//! OuterLink Client Library
//!
//! This crate provides the Rust side of the CUDA interception layer.
//! The C interposition library (csrc/interpose.c) hooks dlsym and
//! cuGetProcAddress, then calls into this library via FFI.
//!
//! Responsibilities:
//! - Manage connection to remote OuterLink server
//! - Handle translation (synthetic local <-> real remote)
//! - Serialize CUDA calls and send over transport
//! - Deserialize responses and return to application

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use outerlink_common::error::OuterLinkError;
use outerlink_common::handle::HandleStore;
use outerlink_common::protocol::{MessageHeader, MessageType};
use outerlink_common::retry::RetryConfig;
use outerlink_common::tcp_transport::TcpTransportConnection;
use outerlink_common::transport::TransportConnection;

use crate::callback::CallbackRegistry;

/// Global client state, initialized on first CUDA call.
pub struct OuterLinkClient {
    /// Handle translation tables
    pub handles: HandleStore,
    /// Server address
    pub server_addr: String,
    /// Connection state -- AtomicBool so it can be mutated through a shared
    /// (&self) reference, which is required when the client lives in a OnceLock.
    pub connected: AtomicBool,
    /// Tokio runtime for blocking on async transport calls from sync FFI context.
    runtime: tokio::runtime::Runtime,
    /// Active TCP connection to the server. Protected by std::sync::Mutex
    /// (not tokio) because it is accessed from sync FFI functions. Starts as
    /// `None` and is populated by `connect()`.
    connection: std::sync::Mutex<Option<Arc<TcpTransportConnection>>>,
    /// Monotonically increasing request ID for the wire protocol.
    next_request_id: AtomicU64,
    /// The remote context handle that is currently active on this connection.
    /// Updated by cuCtxCreate_v2, cuCtxSetCurrent, and cuCtxDestroy_v2.
    /// Used by cuCtxGetDevice to send the correct context to the server.
    pub current_remote_ctx: AtomicU64,
    /// Retry and reconnect configuration.
    retry_config: RetryConfig,
    /// Mutex to prevent concurrent reconnect calls from creating duplicate
    /// connections. Only one thread can execute the reconnect loop at a time;
    /// others wait and then check if already connected.
    reconnect_in_progress: std::sync::Mutex<()>,
    /// Session ID assigned by the server during handshake.
    /// Used by the callback channel to identify which session it belongs to.
    session_id: AtomicU64,
    /// Client-side callback registry (cuStreamAddCallback / cuLaunchHostFunc).
    /// Stores callback_id -> (fn_ptr, user_data). Only callback_id crosses the wire.
    pub callback_registry: Arc<CallbackRegistry>,
    /// Dedicated TCP connection for receiving CallbackReady notifications.
    /// Established lazily on first callback registration. Protected by a Mutex
    /// to prevent concurrent initialization.
    callback_connection: std::sync::Mutex<Option<Arc<TcpTransportConnection>>>,
    /// Handle for the callback listener thread. Kept to detect if it's running.
    /// Wrapped in Arc so the spawned listener task can safely reference it
    /// without relying on the 'static lifetime invariant of OnceLock.
    callback_listener_running: Arc<AtomicBool>,
}

impl OuterLinkClient {
    /// Create a new client (not yet connected).
    pub fn new(server_addr: String) -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("failed to create tokio runtime for OuterLink client");

        Self {
            handles: HandleStore::new(),
            server_addr,
            connected: AtomicBool::new(false),
            runtime,
            connection: std::sync::Mutex::new(None),
            next_request_id: AtomicU64::new(1),
            current_remote_ctx: AtomicU64::new(0),
            retry_config: RetryConfig::default(),
            reconnect_in_progress: std::sync::Mutex::new(()),
            session_id: AtomicU64::new(0),
            callback_registry: Arc::new(CallbackRegistry::new()),
            callback_connection: std::sync::Mutex::new(None),
            callback_listener_running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Create a new client with a custom retry configuration.
    pub fn with_retry_config(server_addr: String, retry_config: RetryConfig) -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("failed to create tokio runtime for OuterLink client");

        Self {
            handles: HandleStore::new(),
            server_addr,
            connected: AtomicBool::new(false),
            runtime,
            connection: std::sync::Mutex::new(None),
            next_request_id: AtomicU64::new(1),
            current_remote_ctx: AtomicU64::new(0),
            retry_config,
            reconnect_in_progress: std::sync::Mutex::new(()),
            session_id: AtomicU64::new(0),
            callback_registry: Arc::new(CallbackRegistry::new()),
            callback_connection: std::sync::Mutex::new(None),
            callback_listener_running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Default timeout for the initial TCP connect handshake (seconds).
    const CONNECT_TIMEOUT_SECS: u64 = 10;

    /// Attempt to connect to the remote OuterLink server.
    ///
    /// On success, stores the connection and sets `connected` to true.
    /// On failure, returns an error; the client remains in disconnected
    /// (stub) mode. The TCP connect is bounded by a 10-second timeout
    /// to avoid blocking forever on unreachable hosts.
    pub fn connect(&self) -> Result<(), OuterLinkError> {
        self.runtime.block_on(async {
            let timeout_dur = std::time::Duration::from_secs(Self::CONNECT_TIMEOUT_SECS);
            let stream = match tokio::time::timeout(
                timeout_dur,
                tokio::net::TcpStream::connect(&self.server_addr),
            )
            .await
            {
                Ok(Ok(s)) => s,
                Ok(Err(e)) => {
                    return Err(OuterLinkError::Connection(format!(
                        "failed to connect: {}", e
                    )));
                }
                Err(_elapsed) => {
                    tracing::warn!(
                        addr = %self.server_addr,
                        timeout_secs = Self::CONNECT_TIMEOUT_SECS,
                        "connection timed out"
                    );
                    return Err(OuterLinkError::Timeout(
                        Self::CONNECT_TIMEOUT_SECS * 1000,
                    ));
                }
            };
            let conn = Arc::new(TcpTransportConnection::new(stream).map_err(|e| {
                OuterLinkError::Connection(format!("failed to initialize connection: {}", e))
            })?);

            // Perform handshake BEFORE storing connection or setting connected flag.
            // This prevents a race where another thread sees connected=true but the
            // handshake hasn't completed (or fails).
            let req_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
            let header = MessageHeader::new_request(req_id, MessageType::Handshake, 0);
            conn.send_message(&header, &[]).await.map_err(|e| {
                OuterLinkError::Connection(format!("handshake send failed: {}", e))
            })?;
            let (_resp_header, resp_payload) = conn.recv_message().await.map_err(|e| {
                OuterLinkError::Connection(format!("handshake recv failed: {}", e))
            })?;
            // Verify the server accepted the handshake (first 4 LE bytes = CuResult).
            if resp_payload.len() >= 4 {
                let result = u32::from_le_bytes(resp_payload[0..4].try_into().unwrap());
                if result != 0 {
                    return Err(OuterLinkError::Connection(
                        format!("handshake rejected by server (code {})", result),
                    ));
                }
            }
            // Parse session_id from extended handshake response (bytes 4..12).
            // Backward compatible: if server doesn't send it, session_id stays 0.
            if resp_payload.len() >= 12 {
                let sid = u64::from_le_bytes(resp_payload[4..12].try_into().unwrap());
                self.session_id.store(sid, Ordering::Release);
            }

            // Handshake succeeded — now store the connection and mark as connected.
            let mut guard = self.connection.lock().unwrap();
            *guard = Some(Arc::clone(&conn));
            drop(guard);
            self.connected.store(true, Ordering::Release);

            Ok(())
        })
    }

    /// Attempt to reconnect to the server with exponential backoff.
    ///
    /// Tries up to `retry_config.max_reconnect_attempts` times. On success,
    /// replaces the stored connection and sets `connected` to true.
    /// On failure, sets `connected` to false and returns the last error.
    ///
    /// This performs a full connect + handshake, same as [`Self::connect`].
    pub fn reconnect(&self) -> Result<(), OuterLinkError> {
        // Serialize concurrent reconnect calls. Only one thread executes the
        // reconnect loop; others wait and then check if already connected.
        let _guard = self.reconnect_in_progress.lock().unwrap_or_else(|e| e.into_inner());

        // Another thread may have reconnected while we were waiting for the lock.
        if self.is_actually_connected() {
            return Ok(());
        }

        let max_attempts = self.retry_config.max_reconnect_attempts;
        let mut last_err = None;

        for attempt in 0..max_attempts {
            tracing::warn!(
                addr = %self.server_addr,
                attempt = attempt + 1,
                max_attempts = max_attempts,
                "attempting reconnect to server"
            );

            match self.connect() {
                Ok(()) => {
                    // Note: connect() already sets connected=true with Release ordering.
                    tracing::info!(
                        addr = %self.server_addr,
                        attempt = attempt + 1,
                        "reconnected to server"
                    );
                    return Ok(());
                }
                Err(e) => {
                    tracing::warn!(
                        addr = %self.server_addr,
                        attempt = attempt + 1,
                        error = %e,
                        "reconnect attempt failed"
                    );
                    last_err = Some(e);
                    // Sleep between failures, but not after the last attempt.
                    if attempt + 1 < max_attempts {
                        let delay = self.retry_config.reconnect_delay(attempt);
                        tracing::warn!(
                            delay_ms = delay.as_millis() as u64,
                            "waiting before next reconnect attempt"
                        );
                        std::thread::sleep(delay);
                    }
                }
            }
        }

        self.connected.store(false, Ordering::Release);
        Err(last_err.unwrap_or_else(|| {
            OuterLinkError::Connection(format!(
                "reconnect failed after {} attempts", max_attempts
            ))
        }))
    }

    /// Check whether the client is truly connected by inspecting both the
    /// client-level `connected` flag AND the transport's own liveness state.
    ///
    /// This avoids the scenario where the client thinks it is connected but
    /// the underlying TCP stream has already been closed by the peer.
    pub fn is_actually_connected(&self) -> bool {
        if !self.connected.load(Ordering::Acquire) {
            return false;
        }
        let guard = self.connection.lock().unwrap();
        match guard.as_ref() {
            Some(conn) => conn.is_connected(),
            None => false,
        }
    }

    /// Execute a single send+recv attempt on the current connection.
    ///
    /// This is the inner, non-retrying implementation. Returns the connection
    /// error and updates the `connected` flag when the transport reports
    /// disconnection.
    fn send_request_once(
        &self,
        msg_type: MessageType,
        payload: &[u8],
    ) -> Result<(MessageHeader, Vec<u8>), OuterLinkError> {
        // Hold the connection mutex for the entire send+recv cycle to prevent
        // two threads from interleaving their send/recv on the same connection.
        let guard = self.connection.lock().unwrap();
        let conn = match guard.as_ref() {
            Some(c) => Arc::clone(c),
            None => {
                self.connected.store(false, Ordering::Release);
                return Err(OuterLinkError::Connection("not connected".into()));
            }
        };

        let req_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let header =
            MessageHeader::new_request(req_id, msg_type, payload.len() as u32);

        let result = self.runtime.block_on(async {
            conn.send_message(&header, payload).await?;
            conn.recv_message().await
        });

        // Drop the guard only after the full send+recv cycle completes.
        drop(guard);

        match result {
            Ok((resp_header, resp_payload)) => {
                debug_assert_eq!(
                    resp_header.request_id, req_id,
                    "response request_id {} does not match sent request_id {}",
                    resp_header.request_id, req_id
                );
                Ok((resp_header, resp_payload))
            }
            Err(e) => {
                if !conn.is_connected() {
                    self.connected.store(false, Ordering::Release);
                }
                Err(e)
            }
        }
    }

    /// Send a request to the server and wait for the response.
    ///
    /// Returns the response header and payload. The caller is responsible
    /// for parsing the payload according to the message type.
    ///
    /// On transient transport errors, the request is retried up to
    /// `retry_config.max_retries` times with increasing delays. If all
    /// retries fail, a reconnect is attempted before returning an error.
    /// Non-retryable errors (CUDA errors, protocol errors) are returned
    /// immediately without retry.
    pub fn send_request(
        &self,
        msg_type: MessageType,
        payload: &[u8],
    ) -> Result<(MessageHeader, Vec<u8>), OuterLinkError> {
        let max_retries = self.retry_config.max_retries;
        let mut last_err;

        // Initial attempt + retries
        match self.send_request_once(msg_type, payload) {
            Ok(result) => return Ok(result),
            Err(e) if !e.is_retryable() => return Err(e),
            Err(e) => last_err = e,
        }

        for retry in 0..max_retries {
            let delay = self.retry_config.retry_delay(retry);
            tracing::warn!(
                msg_type = ?msg_type,
                retry = retry + 1,
                max_retries = max_retries,
                delay_ms = delay.as_millis() as u64,
                error = %last_err,
                "retrying request after transport error"
            );
            std::thread::sleep(delay);

            match self.send_request_once(msg_type, payload) {
                Ok(result) => return Ok(result),
                Err(e) if !e.is_retryable() => return Err(e),
                Err(e) => last_err = e,
            }
        }

        // All retries exhausted -- attempt reconnect
        tracing::warn!(
            msg_type = ?msg_type,
            error = %last_err,
            "all retries exhausted, attempting reconnect"
        );

        if self.reconnect().is_ok() {
            // One final attempt after successful reconnect
            return self.send_request_once(msg_type, payload);
        }

        Err(last_err)
    }

    /// Execute a single send+bulk+recv attempt on the current connection.
    fn send_request_with_bulk_once(
        &self,
        msg_type: MessageType,
        payload: &[u8],
        bulk_data: &[u8],
    ) -> Result<(MessageHeader, Vec<u8>), OuterLinkError> {
        // Hold the connection mutex for the entire send+bulk+recv cycle to
        // prevent two threads from interleaving on the same connection.
        let guard = self.connection.lock().unwrap();
        let conn = match guard.as_ref() {
            Some(c) => Arc::clone(c),
            None => {
                self.connected.store(false, Ordering::Release);
                return Err(OuterLinkError::Connection("not connected".into()));
            }
        };

        let req_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let header =
            MessageHeader::new_request(req_id, msg_type, payload.len() as u32);

        let result = self.runtime.block_on(async {
            conn.send_message(&header, payload).await?;
            conn.send_bulk(bulk_data).await?;
            conn.recv_message().await
        });

        // Drop the guard only after the full send+bulk+recv cycle completes.
        drop(guard);

        match result {
            Ok((resp_header, resp_payload)) => {
                debug_assert_eq!(
                    resp_header.request_id, req_id,
                    "response request_id {} does not match sent request_id {}",
                    resp_header.request_id, req_id
                );
                Ok((resp_header, resp_payload))
            }
            Err(e) => {
                if !conn.is_connected() {
                    self.connected.store(false, Ordering::Release);
                }
                Err(e)
            }
        }
    }

    /// Send a request followed by raw bulk data (e.g. MemcpyHtoD).
    ///
    /// The server reads the framed message first, then the bulk bytes.
    ///
    /// **Phase 2 scaffolding:** This method exists for chunked/streaming
    /// transfers when payloads exceed `MAX_PAYLOAD_SIZE`. Phase 1 sends
    /// data inline in the protocol payload; Phase 2 will use this path
    /// for large transfers that must be split across multiple frames.
    ///
    /// Retries on transient transport errors, same as [`Self::send_request`].
    pub fn send_request_with_bulk(
        &self,
        msg_type: MessageType,
        payload: &[u8],
        bulk_data: &[u8],
    ) -> Result<(MessageHeader, Vec<u8>), OuterLinkError> {
        let max_retries = self.retry_config.max_retries;
        let mut last_err;

        match self.send_request_with_bulk_once(msg_type, payload, bulk_data) {
            Ok(result) => return Ok(result),
            Err(e) if !e.is_retryable() => return Err(e),
            Err(e) => last_err = e,
        }

        for retry in 0..max_retries {
            let delay = self.retry_config.retry_delay(retry);
            tracing::warn!(
                msg_type = ?msg_type,
                retry = retry + 1,
                max_retries = max_retries,
                delay_ms = delay.as_millis() as u64,
                error = %last_err,
                "retrying bulk request after transport error"
            );
            std::thread::sleep(delay);

            match self.send_request_with_bulk_once(msg_type, payload, bulk_data) {
                Ok(result) => return Ok(result),
                Err(e) if !e.is_retryable() => return Err(e),
                Err(e) => last_err = e,
            }
        }

        tracing::warn!(
            msg_type = ?msg_type,
            error = %last_err,
            "all retries exhausted for bulk request, attempting reconnect"
        );

        if self.reconnect().is_ok() {
            return self.send_request_with_bulk_once(msg_type, payload, bulk_data);
        }

        Err(last_err)
    }

    /// Receive raw bulk data from the server (e.g. MemcpyDtoH response).
    pub fn recv_bulk(&self, size: usize) -> Result<Vec<u8>, OuterLinkError> {
        self.runtime.block_on(async {
            let conn = {
                let guard = self.connection.lock().unwrap();
                match guard.as_ref() {
                    Some(c) => Arc::clone(c),
                    None => {
                        self.connected.store(false, Ordering::Release);
                        return Err(OuterLinkError::Connection("not connected".into()));
                    }
                }
            };
            match conn.recv_bulk(size).await {
                Ok(data) => Ok(data),
                Err(e) => {
                    if !conn.is_connected() {
                        self.connected.store(false, Ordering::Release);
                    }
                    Err(e)
                }
            }
        })
    }

    /// Get the session ID assigned by the server during handshake.
    pub fn session_id(&self) -> u64 {
        self.session_id.load(Ordering::Acquire)
    }

    /// Ensure the callback channel is established. Lazily connects a second
    /// TCP connection to the server and sends a `CallbackChannelInit` message
    /// with the session_id so the server can associate it with the right session.
    ///
    /// Also spawns a background listener thread that receives `CallbackReady`
    /// messages and fires the corresponding callbacks in the registry.
    ///
    /// Returns Ok(()) if the channel is already established or was just created.
    pub fn ensure_callback_channel(&self) -> Result<(), OuterLinkError> {
        // Fast path: already established
        if self.callback_listener_running.load(Ordering::Acquire) {
            return Ok(());
        }

        let mut guard = self.callback_connection.lock().unwrap();
        // Double-check after acquiring lock
        if guard.is_some() {
            return Ok(());
        }

        let sid = self.session_id.load(Ordering::Acquire);
        if sid == 0 {
            return Err(OuterLinkError::Connection(
                "cannot establish callback channel: no session_id (handshake incomplete)".into(),
            ));
        }

        // Connect a second TCP connection to the same server.
        let conn = self.runtime.block_on(async {
            let timeout_dur = std::time::Duration::from_secs(Self::CONNECT_TIMEOUT_SECS);
            let stream = match tokio::time::timeout(
                timeout_dur,
                tokio::net::TcpStream::connect(&self.server_addr),
            )
            .await
            {
                Ok(Ok(s)) => s,
                Ok(Err(e)) => {
                    return Err(OuterLinkError::Connection(format!(
                        "callback channel connect failed: {}",
                        e
                    )));
                }
                Err(_elapsed) => {
                    return Err(OuterLinkError::Timeout(Self::CONNECT_TIMEOUT_SECS * 1000));
                }
            };
            let conn = TcpTransportConnection::new(stream).map_err(|e| {
                OuterLinkError::Connection(format!(
                    "callback channel init failed: {}",
                    e
                ))
            })?;
            Ok(Arc::new(conn))
        })?;

        // Send CallbackChannelInit with session_id
        self.runtime.block_on(async {
            let payload = sid.to_le_bytes();
            let header = MessageHeader::new_request(
                0, // request_id irrelevant for callback channel init
                MessageType::CallbackChannelInit,
                payload.len() as u32,
            );
            conn.send_message(&header, &payload).await.map_err(|e| {
                OuterLinkError::Connection(format!("callback channel init send failed: {}", e))
            })?;

            // Wait for CallbackChannelAck
            let (_resp_hdr, resp_payload) = conn.recv_message().await.map_err(|e| {
                OuterLinkError::Connection(format!("callback channel ack recv failed: {}", e))
            })?;
            if resp_payload.len() >= 4 {
                let result = u32::from_le_bytes(resp_payload[0..4].try_into().unwrap());
                if result != 0 {
                    return Err(OuterLinkError::Connection(format!(
                        "callback channel rejected by server (code {})",
                        result
                    )));
                }
            }
            Ok(())
        })?;

        *guard = Some(Arc::clone(&conn));
        drop(guard); // Release lock before spawning thread

        // Spawn background listener thread
        let registry = Arc::clone(&self.callback_registry);
        let cb_conn = Arc::clone(&conn);
        let running_flag = Arc::clone(&self.callback_listener_running);
        running_flag.store(true, Ordering::Release);

        self.runtime.spawn(async move {
            loop {
                match cb_conn.recv_message().await {
                    Ok((hdr, payload)) => {
                        if hdr.msg_type == MessageType::CallbackReady && payload.len() >= 12 {
                            let callback_id =
                                u64::from_le_bytes(payload[0..8].try_into().unwrap());
                            let cuda_status =
                                u32::from_le_bytes(payload[8..12].try_into().unwrap());
                            registry.fire(callback_id, cuda_status);
                        } else {
                            tracing::warn!(
                                msg_type = ?hdr.msg_type,
                                payload_len = payload.len(),
                                "unexpected message on callback channel"
                            );
                        }
                    }
                    Err(OuterLinkError::ConnectionClosed) => {
                        tracing::debug!("callback channel closed by server");
                        break;
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "callback channel recv error");
                        break;
                    }
                }
            }
            running_flag.store(false, Ordering::Release);
        });

        Ok(())
    }
}

pub mod callback;
pub mod ffi;
pub mod nvml;
pub mod nvml_ffi;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Connecting to a non-routable address should fail within the timeout
    /// rather than blocking indefinitely. RFC-5737 198.51.100.1 is guaranteed
    /// non-routable, so the OS won't immediately refuse -- the timeout must
    /// fire.
    #[test]
    fn test_connect_timeout_fires() {
        // Use a non-routable address; the connect must not block forever.
        let client = OuterLinkClient::new("198.51.100.1:14833".to_string());
        let start = std::time::Instant::now();
        let result = client.connect();
        let elapsed = start.elapsed();

        assert!(result.is_err(), "connect to non-routable addr should fail");
        // Must complete within a reasonable bound (we set 10s timeout,
        // allow up to 15s for CI slack).
        assert!(
            elapsed < std::time::Duration::from_secs(15),
            "connect took too long ({elapsed:?}), timeout may not be working"
        );
    }

    /// `is_actually_connected()` should return false when no connection exists.
    #[test]
    fn test_is_actually_connected_no_connection() {
        let client = OuterLinkClient::new("127.0.0.1:14833".to_string());
        assert!(!client.is_actually_connected());
    }

    /// `is_actually_connected()` should return true after a successful connect,
    /// and should reflect transport-level disconnection.
    #[test]
    fn test_is_actually_connected_after_connect_and_disconnect() {
        // Start a temporary TCP listener to accept the connection.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let client = OuterLinkClient::new(addr.to_string());

        // Accept in a background thread so the handshake can proceed.
        let accept_thread = std::thread::spawn(move || {
            use std::io::{Read, Write};

            let (mut stream, _) = listener.accept().unwrap();

            // Read the handshake request (22-byte header + 0 payload).
            let mut buf = [0u8; 22];
            stream.read_exact(&mut buf).unwrap();

            // Build a valid OLNK response with CuResult::Success (0u32 LE).
            use outerlink_common::protocol::MessageHeader;
            let resp_payload = 0u32.to_le_bytes();
            let resp_header = MessageHeader::new_response(
                // extract request_id from the received header
                u64::from_be_bytes(buf[8..16].try_into().unwrap()),
                resp_payload.len() as u32,
            );
            let header_bytes = resp_header.to_bytes();
            stream.write_all(&header_bytes).unwrap();
            stream.write_all(&resp_payload).unwrap();
            stream.flush().unwrap();

            // Return the stream so we can close it explicitly.
            stream
        });

        // Connect (includes handshake).
        client.connect().unwrap();
        assert!(client.is_actually_connected());

        // Drop the server-side stream to simulate server going away.
        let server_stream = accept_thread.join().unwrap();
        drop(server_stream);

        // A send_request should detect the broken transport and update
        // the connected flag.
        let _result = client.send_request(
            outerlink_common::protocol::MessageType::DeviceGetCount,
            &[],
        );
        assert!(
            !client.is_actually_connected(),
            "connected flag should be false after transport disconnect"
        );
    }

    /// `send_request` should sync the `connected` AtomicBool when the transport
    /// reports an error (e.g. ConnectionClosed or Transport error).
    #[test]
    fn test_send_request_syncs_connected_flag_on_error() {
        let client = OuterLinkClient::new("127.0.0.1:14833".to_string());
        // Force connected=true but no actual connection.
        client.connected.store(true, Ordering::Release);
        // send_request should fail (no transport) and set connected to false.
        let result = client.send_request(
            outerlink_common::protocol::MessageType::DeviceGetCount,
            &[],
        );
        assert!(result.is_err());
        assert!(
            !client.connected.load(Ordering::Acquire),
            "connected should be false after send_request fails"
        );
    }

    // -----------------------------------------------------------------------
    // Retry and reconnect tests
    // -----------------------------------------------------------------------

    /// Helper: spawn a mock server that accepts one connection, responds to
    /// one handshake, then drops. Returns the listener address.
    fn spawn_mock_server_one_shot() -> (std::net::SocketAddr, std::thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            use std::io::{Read, Write};
            if let Ok((mut stream, _)) = listener.accept() {
                // Read handshake request (22-byte header).
                let mut buf = [0u8; 22];
                if stream.read_exact(&mut buf).is_ok() {
                    let resp_payload = 0u32.to_le_bytes();
                    let resp_header = MessageHeader::new_response(
                        u64::from_be_bytes(buf[8..16].try_into().unwrap()),
                        resp_payload.len() as u32,
                    );
                    let _ = stream.write_all(&resp_header.to_bytes());
                    let _ = stream.write_all(&resp_payload);
                    let _ = stream.flush();
                }
                // Drop stream immediately to simulate server going away.
            }
        });
        (addr, handle)
    }

    /// Helper: spawn a mock server that stays alive, accepts connections,
    /// responds to handshake AND one request per connection.
    fn spawn_mock_server_persistent(
    ) -> (std::net::SocketAddr, Arc<AtomicBool>, std::thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();

        listener.set_nonblocking(true).unwrap();

        let handle = std::thread::spawn(move || {
            use std::io::{Read, Write};

            while !shutdown_clone.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream.set_nonblocking(false).unwrap();
                        // Read and respond to messages until connection drops.
                        loop {
                            let mut buf = [0u8; 22];
                            if stream.read_exact(&mut buf).is_err() {
                                break;
                            }
                            let payload_len = u32::from_be_bytes(
                                buf[18..22].try_into().unwrap(),
                            );
                            // Drain payload if any
                            if payload_len > 0 {
                                let mut payload = vec![0u8; payload_len as usize];
                                if stream.read_exact(&mut payload).is_err() {
                                    break;
                                }
                            }
                            let resp_payload = 0u32.to_le_bytes();
                            let resp_header = MessageHeader::new_response(
                                u64::from_be_bytes(buf[8..16].try_into().unwrap()),
                                resp_payload.len() as u32,
                            );
                            if stream.write_all(&resp_header.to_bytes()).is_err() {
                                break;
                            }
                            if stream.write_all(&resp_payload).is_err() {
                                break;
                            }
                            let _ = stream.flush();
                        }
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                    Err(_) => break,
                }
            }
        });
        (addr, shutdown, handle)
    }

    /// `send_request` with no_retry config should fail immediately on transport error
    /// without retrying.
    #[test]
    fn test_send_request_no_retry_fails_immediately() {
        let (addr, server_handle) = spawn_mock_server_one_shot();
        let client = OuterLinkClient::with_retry_config(
            addr.to_string(),
            RetryConfig::no_retry(),
        );
        client.connect().unwrap();
        server_handle.join().unwrap();

        // Server is now gone. The next request should fail without retries.
        let start = std::time::Instant::now();
        let result = client.send_request(MessageType::DeviceGetCount, &[]);
        let elapsed = start.elapsed();

        assert!(result.is_err(), "should fail when server is gone");
        // With no retries, should fail fast (well under 1s).
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "no_retry should fail fast, took {elapsed:?}"
        );
    }

    /// Retry succeeds when server comes back before retries are exhausted.
    #[test]
    fn test_send_request_retry_succeeds_on_reconnect() {
        let (addr, shutdown, _server_handle) = spawn_mock_server_persistent();

        // Use very short retry delays for test speed.
        let retry_config = RetryConfig {
            max_retries: 3,
            retry_delays: vec![
                std::time::Duration::from_millis(10),
                std::time::Duration::from_millis(20),
                std::time::Duration::from_millis(30),
            ],
            max_reconnect_attempts: 3,
            reconnect_initial_delay: std::time::Duration::from_millis(50),
            reconnect_max_delay: std::time::Duration::from_millis(200),
        };

        let client = OuterLinkClient::with_retry_config(
            addr.to_string(),
            retry_config,
        );
        client.connect().unwrap();

        // Verify normal request works.
        let result = client.send_request(MessageType::DeviceGetCount, &[]);
        assert!(result.is_ok(), "initial request should work");

        // Shutdown and cleanup.
        shutdown.store(true, Ordering::Relaxed);
    }

    /// Non-retryable errors (Connection("not connected") when no transport exists)
    /// should not be retried. This tests that the "not connected" path in
    /// send_request_once triggers retry (since Connection is retryable),
    /// but eventually exhausts attempts.
    #[test]
    fn test_send_request_retries_on_connection_error() {
        let client = OuterLinkClient::with_retry_config(
            "127.0.0.1:1".to_string(), // Won't connect (port 1 is discard)
            RetryConfig {
                max_retries: 2,
                retry_delays: vec![std::time::Duration::from_millis(1)],
                max_reconnect_attempts: 0, // No reconnect
                reconnect_initial_delay: std::time::Duration::ZERO,
                reconnect_max_delay: std::time::Duration::ZERO,
            },
        );
        // Force "connected" so send_request_once tries and gets Connection error.
        client.connected.store(true, Ordering::Release);

        let result = client.send_request(MessageType::DeviceGetCount, &[]);
        assert!(result.is_err());
    }

    /// Reconnect with exponential backoff timing: verify the delay grows.
    #[test]
    fn test_reconnect_backoff_timing() {
        // No server at all -- every reconnect attempt will fail.
        let retry_config = RetryConfig {
            max_retries: 0,
            retry_delays: vec![std::time::Duration::ZERO],
            max_reconnect_attempts: 3,
            reconnect_initial_delay: std::time::Duration::from_millis(50),
            reconnect_max_delay: std::time::Duration::from_secs(1),
        };
        let client = OuterLinkClient::with_retry_config(
            "127.0.0.1:1".to_string(),
            retry_config,
        );

        let start = std::time::Instant::now();
        let result = client.reconnect();
        let elapsed = start.elapsed();

        assert!(result.is_err(), "reconnect should fail with no server");
        // 3 attempts: first tries immediately, sleeps between failures.
        // Delays: after attempt 0 = 50ms, after attempt 1 = 100ms. No sleep
        // after the final attempt. Total sleep = 150ms minimum.
        assert!(
            elapsed >= std::time::Duration::from_millis(100),
            "backoff should cause at least 100ms delay, got {elapsed:?}"
        );
    }

    /// After successful reconnect, send_request should work again.
    #[test]
    fn test_reconnect_restores_connectivity() {
        let (addr, shutdown, _server_handle) = spawn_mock_server_persistent();

        let retry_config = RetryConfig {
            max_retries: 0,
            retry_delays: vec![std::time::Duration::ZERO],
            max_reconnect_attempts: 3,
            reconnect_initial_delay: std::time::Duration::from_millis(10),
            reconnect_max_delay: std::time::Duration::from_millis(100),
        };

        let client = OuterLinkClient::with_retry_config(
            addr.to_string(),
            retry_config,
        );
        client.connect().unwrap();
        assert!(client.is_actually_connected());

        // Reconnect (even while already connected) should succeed.
        let result = client.reconnect();
        assert!(result.is_ok(), "reconnect should succeed: {result:?}");
        assert!(client.is_actually_connected());

        // Should still be able to send requests.
        let result = client.send_request(MessageType::DeviceGetCount, &[]);
        assert!(result.is_ok(), "request after reconnect should work");

        shutdown.store(true, Ordering::Relaxed);
    }

    /// Reconnect should NOT sleep before the first attempt -- it should try
    /// connecting immediately, then sleep only between subsequent attempts.
    #[test]
    fn test_reconnect_no_sleep_before_first_attempt() {
        // Use a large reconnect_initial_delay so we can detect if the first
        // attempt sleeps (it shouldn't).
        let retry_config = RetryConfig {
            max_retries: 0,
            retry_delays: vec![std::time::Duration::ZERO],
            max_reconnect_attempts: 1, // Only one attempt, no retries
            reconnect_initial_delay: std::time::Duration::from_secs(5),
            reconnect_max_delay: std::time::Duration::from_secs(5),
        };
        let client = OuterLinkClient::with_retry_config(
            "127.0.0.1:1".to_string(),
            retry_config,
        );

        let start = std::time::Instant::now();
        let _result = client.reconnect();
        let elapsed = start.elapsed();

        // With 1 attempt and no sleep-before-first, should complete well
        // under 5 seconds (the connect timeout is 10s but port 1 usually
        // fails fast). If it slept 5s before trying, this assertion fails.
        assert!(
            elapsed < std::time::Duration::from_secs(3),
            "reconnect should not sleep before first attempt, took {elapsed:?}"
        );
    }

    /// Concurrent reconnect calls should not create duplicate connections.
    /// The reconnect_in_progress mutex should serialize them.
    #[test]
    fn test_reconnect_serialized_by_mutex() {
        let (addr, shutdown, _server_handle) = spawn_mock_server_persistent();

        let retry_config = RetryConfig {
            max_retries: 0,
            retry_delays: vec![std::time::Duration::ZERO],
            max_reconnect_attempts: 3,
            reconnect_initial_delay: std::time::Duration::from_millis(10),
            reconnect_max_delay: std::time::Duration::from_millis(50),
        };

        let client = Arc::new(OuterLinkClient::with_retry_config(
            addr.to_string(),
            retry_config,
        ));
        client.connect().unwrap();

        // Spawn two threads that both call reconnect concurrently.
        let c1 = Arc::clone(&client);
        let c2 = Arc::clone(&client);

        let t1 = std::thread::spawn(move || c1.reconnect());
        let t2 = std::thread::spawn(move || c2.reconnect());

        let r1 = t1.join().unwrap();
        let r2 = t2.join().unwrap();

        // Both should succeed (server is up), and we should still be connected.
        assert!(r1.is_ok(), "thread 1 reconnect failed: {r1:?}");
        assert!(r2.is_ok(), "thread 2 reconnect failed: {r2:?}");
        assert!(client.is_actually_connected());

        shutdown.store(true, Ordering::Relaxed);
    }

    /// `with_retry_config` should use the provided config.
    #[test]
    fn test_with_retry_config_uses_custom_config() {
        let config = RetryConfig {
            max_retries: 7,
            retry_delays: vec![std::time::Duration::from_millis(42)],
            max_reconnect_attempts: 2,
            reconnect_initial_delay: std::time::Duration::from_millis(100),
            reconnect_max_delay: std::time::Duration::from_secs(5),
        };
        let client = OuterLinkClient::with_retry_config(
            "127.0.0.1:1234".to_string(),
            config,
        );
        assert_eq!(client.retry_config.max_retries, 7);
        assert_eq!(client.retry_config.max_reconnect_attempts, 2);
    }
}
