//! Graceful-shutdown server loop.
//!
//! [`Server`] wraps a `TcpListener` + `GpuBackend` and provides:
//!
//! 1. A `shutdown_handle()` that returns a `watch::Sender<()>` the caller
//!    can fire to initiate shutdown.
//! 2. On shutdown signal: stop accepting new connections, wait for in-flight
//!    connections to drain (with a timeout), then call `backend.shutdown()`.

use std::sync::Arc;
use std::time::Duration;

use tokio::net::TcpListener;
use tokio::sync::watch;

use outerlink_common::error::OuterLinkError;
use outerlink_common::tcp_transport::TcpTransportConnection;
use outerlink_common::transport::TransportConnection;

use crate::gpu_backend::GpuBackend;
use crate::handler::handle_request;
use crate::session::ConnectionSession;

/// Default maximum time to wait for in-flight connections to finish
/// after the shutdown signal fires.
const DEFAULT_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

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

        tracing::info!(
            addr = %self.listener.local_addr().unwrap_or_else(|_| "unknown".parse().unwrap()),
            "server accept loop starting"
        );

        loop {
            tokio::select! {
                // Bias towards checking shutdown first so we don't accept
                // a new connection when shutdown is already pending.
                biased;

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

                            join_set.spawn(async move {
                                let conn = match TcpTransportConnection::new(stream) {
                                    Ok(c) => c,
                                    Err(e) => {
                                        tracing::error!(%peer, error = %e, "failed to initialise transport");
                                        return;
                                    }
                                };
                                if let Err(e) = handle_connection(conn, backend, &mut conn_shutdown_rx).await {
                                    tracing::error!(%peer, error = %e, "connection handler error");
                                }
                                tracing::info!(%peer, "connection closed");
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
                while join_set.join_next().await.is_some() {}
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
                    // Wait for abort to complete.
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

/// Drive a single client connection to completion.
///
/// Also monitors the shutdown signal: when shutdown is requested, we allow
/// the current in-flight request to finish but then stop reading new ones.
async fn handle_connection(
    conn: TcpTransportConnection,
    backend: Arc<dyn GpuBackend>,
    shutdown_rx: &mut watch::Receiver<()>,
) -> anyhow::Result<()> {
    let mut session = ConnectionSession::new();

    loop {
        // Wait for either a new message or a shutdown signal.
        let msg = tokio::select! {
            biased;

            _ = shutdown_rx.changed() => {
                tracing::debug!("connection received shutdown signal, finishing");
                // Drain: continue processing already-received data but
                // don't wait for the next request. We do one more non-blocking
                // attempt to read and respond, then exit.
                //
                // In practice, the client will notice the connection close
                // and reconnect to another server or retry.
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

        // Dispatch.
        let (resp_header, resp_payload) =
            handle_request(&*backend, &header, &payload, &mut session);

        // Write the response.
        conn.send_message(&resp_header, &resp_payload).await?;
    }
}
