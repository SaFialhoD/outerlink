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
        }
    }

    /// Attempt to connect to the remote OuterLink server.
    ///
    /// On success, stores the connection and sets `connected` to true.
    /// On failure, returns an error; the client remains in disconnected
    /// (stub) mode.
    pub fn connect(&self) -> Result<(), OuterLinkError> {
        self.runtime.block_on(async {
            let stream = tokio::net::TcpStream::connect(&self.server_addr)
                .await
                .map_err(|e| {
                    OuterLinkError::Connection(format!("failed to connect: {}", e))
                })?;
            let conn = Arc::new(TcpTransportConnection::new(stream).map_err(|e| {
                OuterLinkError::Connection(format!("failed to initialize connection: {}", e))
            })?);
            let mut guard = self.connection.lock().unwrap();
            *guard = Some(conn);
            self.connected.store(true, Ordering::Release);
            Ok(())
        })
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
                let conn = guard.as_ref().ok_or_else(|| {
                    OuterLinkError::Connection("not connected".into())
                })?;
                Arc::clone(conn)
            };

            let req_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
            let header =
                MessageHeader::new_request(req_id, msg_type, payload.len() as u32);

            conn.send_message(&header, payload).await?;
            let (resp_header, resp_payload) = conn.recv_message().await?;
            Ok((resp_header, resp_payload))
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
                let conn = guard.as_ref().ok_or_else(|| {
                    OuterLinkError::Connection("not connected".into())
                })?;
                Arc::clone(conn)
            };

            let req_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
            let header =
                MessageHeader::new_request(req_id, msg_type, payload.len() as u32);

            conn.send_message(&header, payload).await?;
            conn.send_bulk(bulk_data).await?;
            let (resp_header, resp_payload) = conn.recv_message().await?;
            Ok((resp_header, resp_payload))
        })
    }

    /// Receive raw bulk data from the server (e.g. MemcpyDtoH response).
    pub fn recv_bulk(&self, size: usize) -> Result<Vec<u8>, OuterLinkError> {
        self.runtime.block_on(async {
            let conn = {
                let guard = self.connection.lock().unwrap();
                let conn = guard.as_ref().ok_or_else(|| {
                    OuterLinkError::Connection("not connected".into())
                })?;
                Arc::clone(conn)
            };
            conn.recv_bulk(size).await
        })
    }
}

pub mod ffi;
