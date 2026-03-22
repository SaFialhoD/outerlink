//! Graceful-shutdown server loop.
//!
//! [`Server`] wraps a `TcpListener` + `GpuBackend` and provides:
//!
//! 1. A `shutdown_handle()` that returns a `watch::Sender<()>` the caller
//!    can fire to initiate shutdown.
//! 2. On shutdown signal: stop accepting new connections, wait for in-flight
//!    connections to drain (with a timeout), then call `backend.shutdown()`.
//! 3. Session registry for associating callback channels with sessions.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::net::TcpListener;
use tokio::sync::{watch, Mutex as TokioMutex};

use outerlink_common::error::OuterLinkError;
use outerlink_common::protocol::{MessageHeader, MessageType};
use outerlink_common::tcp_transport::TcpTransportConnection;
use outerlink_common::transport::TransportConnection;

use crate::gpu_backend::GpuBackend;
use crate::handler::handle_request_full;
use crate::session::ConnectionSession;

/// Default maximum time to wait for in-flight connections to finish
/// after the shutdown signal fires.
const DEFAULT_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

/// Generates unique session IDs.
static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);

/// Registry of active sessions, keyed by session_id.
/// Used by callback channels to find their parent session.
pub type SessionRegistry = Arc<TokioMutex<HashMap<u64, Arc<TokioMutex<ConnectionSession>>>>>;

/// A TCP server that can be shut down gracefully.
pub struct Server {
    listener: TcpListener,
    backend: Arc<dyn GpuBackend>,
    shutdown_tx: watch::Sender<()>,
    shutdown_rx: watch::Receiver<()>,
    drain_timeout: Duration,
}

impl Server {
    /// Create a new server bound to the given listener.
    ///
    /// Call [`shutdown_handle()`](Self::shutdown_handle) to get a sender that
    /// triggers graceful shutdown, then call [`run()`](Self::run) to start
    /// the accept loop.
    pub fn new(listener: TcpListener, backend: Arc<dyn GpuBackend>) -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(());
        Self {
            listener,
            backend,
            shutdown_tx,
            shutdown_rx,
            drain_timeout: DEFAULT_DRAIN_TIMEOUT,
        }
    }

    /// Set a custom drain timeout (how long to wait for in-flight connections
    /// after shutdown is signalled).
    #[allow(dead_code)]
    pub fn with_drain_timeout(mut self, timeout: Duration) -> Self {
        self.drain_timeout = timeout;
        self
    }

    /// Return a clone of the shutdown sender.
    ///
    /// Calling `send(())` on this handle triggers graceful shutdown:
    /// the accept loop stops, in-flight connections are given time to finish,
    /// and the GPU backend is cleaned up.
    pub fn shutdown_handle(&self) -> watch::Sender<()> {
        self.shutdown_tx.clone()
    }

    /// Run the accept loop until shutdown is signalled.
    ///
    /// After shutdown:
    /// 1. Stop accepting new connections.
    /// 2. Wait up to `drain_timeout` for in-flight connection tasks to finish.
    /// 3. Call `backend.shutdown()` to release GPU resources.
    pub async fn run(self) {
        let mut shutdown_rx = self.shutdown_rx.clone();
        // We track active connection tasks via a JoinSet.
        let mut join_set = tokio::task::JoinSet::new();
        // Session registry for callback channel association.
        let sessions: SessionRegistry = Arc::new(TokioMutex::new(HashMap::new()));

        tracing::info!(
            addr = %self.listener.local_addr().unwrap_or_else(|_| "unknown".parse().unwrap()),
            "server accept loop starting"
        );

        loop {
            tokio::select! {
                // Bias towards checking shutdown first so we don't accept
                // a new connection when shutdown is already pending.
                biased;

                // Either shutdown was signalled (Ok), or the server dropped
                // the sender (Err(RecvError)).  Either way, stop accepting.
                _ = shutdown_rx.changed() => {
                    tracing::info!("shutdown signal received, stopping accept loop");
                    break;
                }

                accept_result = self.listener.accept() => {
                    match accept_result {
                        Ok((stream, peer)) => {
                            tracing::info!(%peer, "new connection");
                            let backend = Arc::clone(&self.backend);
                            let mut conn_shutdown_rx = self.shutdown_rx.clone();
                            let sessions = Arc::clone(&sessions);

                            join_set.spawn(async move {
                                let conn = match TcpTransportConnection::new(stream) {
                                    Ok(c) => Arc::new(c),
                                    Err(e) => {
                                        tracing::error!(%peer, error = %e, "failed to initialise transport");
                                        return;
                                    }
                                };

                                // Peek at the first message to determine connection type.
                                // If it's CallbackChannelInit, this is a callback channel.
                                // Otherwise, it's a normal client connection.
                                let (first_header, first_payload) = match conn.recv_message().await {
                                    Ok(msg) => msg,
                                    Err(OuterLinkError::ConnectionClosed) => return,
                                    Err(e) => {
                                        tracing::error!(%peer, error = %e, "failed to read first message");
                                        return;
                                    }
                                };

                                if first_header.msg_type == MessageType::CallbackChannelInit {
                                    // This is a callback channel connection.
                                    handle_callback_channel_init(
                                        &conn, &first_header, &first_payload, &sessions,
                                    ).await;
                                    // Callback channel stays open until client disconnects;
                                    // it only receives messages FROM the server (via send_message).
                                    // The channel's lifetime is managed by the session that holds
                                    // its Arc. We just keep the task alive so the connection stays open.
                                    // Wait until the connection drops.
                                    loop {
                                        match conn.recv_message().await {
                                            Err(OuterLinkError::ConnectionClosed) => break,
                                            Err(_) => break,
                                            Ok(_) => {
                                                // Unexpected message on callback channel; ignore.
                                            }
                                        }
                                    }
                                    tracing::info!(%peer, "callback channel closed");
                                } else {
                                    // Normal client connection. Process the first message, then loop.
                                    let session_id = NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed);
                                    let session = Arc::new(TokioMutex::new(
                                        ConnectionSession::with_session_id(session_id),
                                    ));

                                    // Register session for callback channel lookup.
                                    {
                                        let mut reg = sessions.lock().await;
                                        reg.insert(session_id, Arc::clone(&session));
                                    }

                                    // Handle the first message (which we already read).
                                    {
                                        let mut sess = session.lock().await;
                                        let result = handle_request_full(
                                            &*backend, &first_header, &first_payload, &mut sess,
                                        );
                                        if let Err(e) = conn.send_message(
                                            &result.response.0, &result.response.1,
                                        ).await {
                                            tracing::error!(%peer, error = %e, "send failed on first message");
                                            sess.cleanup(&*backend);
                                            let mut reg = sessions.lock().await;
                                            reg.remove(&session_id);
                                            return;
                                        }
                                        // Send callback notification if present
                                        if let Some((cb_id, cb_status)) = result.callback_notification {
                                            send_callback_ready(&sess, cb_id, cb_status).await;
                                        }
                                    }

                                    // Main request loop
                                    if let Err(e) = handle_connection_loop(
                                        &conn, &*backend, &mut conn_shutdown_rx, &session,
                                    ).await {
                                        tracing::error!(%peer, error = %e, "connection handler error");
                                    }

                                    // Cleanup
                                    {
                                        let mut sess = session.lock().await;
                                        let report = sess.cleanup(&*backend);
                                        if report.succeeded > 0 || report.failed > 0 {
                                            tracing::info!(
                                                succeeded = report.succeeded,
                                                failed = report.failed,
                                                "session cleanup complete"
                                            );
                                        }
                                    }
                                    {
                                        let mut reg = sessions.lock().await;
                                        reg.remove(&session_id);
                                    }

                                    tracing::info!(%peer, "connection closed");
                                }
                            });
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "accept failed, retrying");
                        }
                    }
                }
            }
        }

        // --- Drain phase: wait for in-flight connections ---
        let active = join_set.len();
        if active > 0 {
            tracing::info!(
                active_connections = active,
                drain_timeout_secs = self.drain_timeout.as_secs(),
                "waiting for in-flight connections to finish"
            );

            let drain_result = tokio::time::timeout(self.drain_timeout, async {
                while let Some(result) = join_set.join_next().await {
                    if let Err(e) = result {
                        if e.is_panic() {
                            tracing::error!("connection task panicked during drain: {:?}", e);
                        }
                    }
                }
            })
            .await;

            match drain_result {
                Ok(()) => {
                    tracing::info!("all connections drained cleanly");
                }
                Err(_) => {
                    let remaining = join_set.len();
                    tracing::warn!(
                        remaining_connections = remaining,
                        "drain timeout expired, aborting remaining connections"
                    );
                    join_set.abort_all();
                    // Wait for abort to complete (cancelled tasks are expected here).
                    while join_set.join_next().await.is_some() {}
                }
            }
        }

        // --- Cleanup: release GPU resources ---
        tracing::info!("cleaning up GPU backend resources");
        self.backend.shutdown();
        tracing::info!("server shutdown complete");
    }
}

/// Handle a CallbackChannelInit message on a newly accepted connection.
///
/// Looks up the session by session_id and sets the callback channel on it.
/// Sends a CallbackChannelAck response.
async fn handle_callback_channel_init(
    conn: &Arc<TcpTransportConnection>,
    header: &MessageHeader,
    payload: &[u8],
    sessions: &SessionRegistry,
) {
    if payload.len() < 8 {
        tracing::warn!("CallbackChannelInit payload too short");
        let resp = MessageHeader::new_response(header.request_id, 4);
        let _ = conn.send_message(&resp, &1u32.to_le_bytes()).await; // InvalidValue
        return;
    }
    let session_id = u64::from_le_bytes(payload[0..8].try_into().unwrap());

    let reg = sessions.lock().await;
    if let Some(session) = reg.get(&session_id) {
        let mut sess = session.lock().await;
        sess.set_callback_channel(Arc::clone(conn));
        drop(sess);
        drop(reg);

        // Send success ack
        let resp = MessageHeader::new_response(header.request_id, 4);
        let _ = conn.send_message(&resp, &0u32.to_le_bytes()).await;
        tracing::info!(session_id, "callback channel associated with session");
    } else {
        drop(reg);
        tracing::warn!(session_id, "CallbackChannelInit: session not found");
        let resp = MessageHeader::new_response(header.request_id, 4);
        let _ = conn.send_message(&resp, &1u32.to_le_bytes()).await; // InvalidValue
    }
}

/// Send a CallbackReady notification on the session's callback channel.
///
/// If no callback channel is established, the notification is logged and dropped.
async fn send_callback_ready(session: &ConnectionSession, callback_id: u64, cuda_status: u32) {
    if let Some(cb_conn) = session.callback_channel() {
        let mut payload = Vec::with_capacity(12);
        payload.extend_from_slice(&callback_id.to_le_bytes());
        payload.extend_from_slice(&cuda_status.to_le_bytes());
        let header = MessageHeader::new_request(0, MessageType::CallbackReady, payload.len() as u32);
        if let Err(e) = cb_conn.send_message(&header, &payload).await {
            tracing::warn!(
                callback_id,
                error = %e,
                "failed to send CallbackReady on callback channel"
            );
        }
    } else {
        tracing::debug!(
            callback_id,
            "no callback channel, dropping CallbackReady notification"
        );
    }
}

/// Inner loop for a single client connection.
///
/// Uses `handle_request_full` to get both the main response and any callback
/// notifications to send.
async fn handle_connection_loop(
    conn: &Arc<TcpTransportConnection>,
    backend: &dyn GpuBackend,
    shutdown_rx: &mut watch::Receiver<()>,
    session: &Arc<TokioMutex<ConnectionSession>>,
) -> anyhow::Result<()> {
    loop {
        // Wait for either a new message or a shutdown signal.
        let msg = tokio::select! {
            _ = shutdown_rx.changed() => {
                tracing::debug!("connection received shutdown signal, finishing");
                return Ok(());
            }

            result = conn.recv_message() => {
                match result {
                    Ok(msg) => msg,
                    Err(OuterLinkError::ConnectionClosed) => return Ok(()),
                    Err(e) => return Err(anyhow::anyhow!(e)),
                }
            }
        };

        let (header, payload) = msg;

        tracing::debug!(
            request_id = header.request_id,
            msg_type = ?header.msg_type,
            payload_len = header.payload_len,
            "received request"
        );

        // Dispatch
        let result = {
            let mut sess = session.lock().await;
            handle_request_full(backend, &header, &payload, &mut sess)
        };

        // Write the response.
        conn.send_message(&result.response.0, &result.response.1).await?;

        // Send callback notification if present
        if let Some((cb_id, cb_status)) = result.callback_notification {
            let sess = session.lock().await;
            send_callback_ready(&sess, cb_id, cb_status).await;
        }

        // Check shutdown after completing this request, before reading next.
        if shutdown_rx.has_changed().unwrap_or(true) {
            tracing::debug!("shutdown pending, stopping after completed request");
            return Ok(());
        }
    }
}
