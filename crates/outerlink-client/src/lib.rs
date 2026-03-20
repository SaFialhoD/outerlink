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
use outerlink_common::tcp_transport::TcpTransportConnection;
use outerlink_common::transport::TransportConnection;

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
            let mut guard = self.connection.lock().unwrap();
            *guard = Some(Arc::clone(&conn));
            drop(guard);
            self.connected.store(true, Ordering::Release);

            // Send the one-time per-connection Handshake.
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

            Ok(())
        })
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

    /// Send a request to the server and wait for the response.
    ///
    /// Returns the response header and payload. The caller is responsible
    /// for parsing the payload according to the message type.
    pub fn send_request(
        &self,
        msg_type: MessageType,
        payload: &[u8],
    ) -> Result<(MessageHeader, Vec<u8>), OuterLinkError> {
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

            let req_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
            let header =
                MessageHeader::new_request(req_id, msg_type, payload.len() as u32);

            if let Err(e) = conn.send_message(&header, payload).await {
                if !conn.is_connected() {
                    self.connected.store(false, Ordering::Release);
                }
                return Err(e);
            }
            match conn.recv_message().await {
                Ok((resp_header, resp_payload)) => Ok((resp_header, resp_payload)),
                Err(e) => {
                    if !conn.is_connected() {
                        self.connected.store(false, Ordering::Release);
                    }
                    Err(e)
                }
            }
        })
    }

    /// Send a request followed by raw bulk data (e.g. MemcpyHtoD).
    ///
    /// The server reads the framed message first, then the bulk bytes.
    ///
    /// **Phase 2 scaffolding:** This method exists for chunked/streaming
    /// transfers when payloads exceed `MAX_PAYLOAD_SIZE`. Phase 1 sends
    /// data inline in the protocol payload; Phase 2 will use this path
    /// for large transfers that must be split across multiple frames.
    pub fn send_request_with_bulk(
        &self,
        msg_type: MessageType,
        payload: &[u8],
        bulk_data: &[u8],
    ) -> Result<(MessageHeader, Vec<u8>), OuterLinkError> {
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

            let req_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
            let header =
                MessageHeader::new_request(req_id, msg_type, payload.len() as u32);

            let result = async {
                conn.send_message(&header, payload).await?;
                conn.send_bulk(bulk_data).await?;
                conn.recv_message().await
            }
            .await;

            match result {
                Ok((resp_header, resp_payload)) => Ok((resp_header, resp_payload)),
                Err(e) => {
                    if !conn.is_connected() {
                        self.connected.store(false, Ordering::Release);
                    }
                    Err(e)
                }
            }
        })
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
}

pub mod ffi;

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
}
