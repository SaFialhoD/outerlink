//! Transport trait - pluggable network layer.
//!
//! Defines the interface that all transport backends must implement.
//! Phase 1: TCP (tokio)
//! Phase 2: UCX (auto-negotiates RDMA vs TCP)
//! Phase 5: OpenDMA (direct BAR1 RDMA)

use crate::protocol::MessageHeader;
use crate::Result;

/// A transport connection to a remote node.
///
/// Implementations handle the actual network I/O (TCP, RDMA, etc.)
/// The protocol layer sits on top of this.
#[async_trait::async_trait]
pub trait TransportConnection: Send + Sync {
    /// Send a message header + payload to the remote end.
    async fn send_message(&self, header: &MessageHeader, payload: &[u8]) -> Result<()>;

    /// Receive a message header + payload from the remote end.
    /// Returns (header, payload_bytes).
    async fn recv_message(&self) -> Result<(MessageHeader, Vec<u8>)>;

    /// Send raw bulk data (for large memory transfers).
    /// This may bypass the protocol framing for efficiency.
    async fn send_bulk(&self, data: &[u8]) -> Result<()>;

    /// Receive raw bulk data of known size.
    async fn recv_bulk(&self, size: usize) -> Result<Vec<u8>>;

    /// Check if the connection is still alive.
    fn is_connected(&self) -> bool;

    /// Close the connection gracefully.
    async fn close(&self) -> Result<()>;

    /// Get the remote address as a string (for logging).
    fn remote_addr(&self) -> String;
}

/// Factory for creating transport connections.
///
/// The client uses this to connect to servers.
/// The server uses this to listen for connections.
#[async_trait::async_trait]
pub trait TransportFactory: Send + Sync {
    /// The connection type this factory produces.
    type Connection: TransportConnection;

    /// Connect to a remote server.
    async fn connect(&self, addr: &str) -> Result<Self::Connection>;
}

/// Listener for incoming transport connections.
#[async_trait::async_trait]
pub trait TransportListener: Send + Sync {
    /// The connection type this listener produces.
    type Connection: TransportConnection;

    /// Accept the next incoming connection.
    async fn accept(&self) -> Result<Self::Connection>;

    /// Get the local address being listened on.
    fn local_addr(&self) -> String;
}

// Note: async_trait is used here for simplicity.
// We add the dependency below. In the future, we may switch to
// native async traits (stabilized in Rust 1.75+) for zero-cost.

// Re-export for convenience
pub use async_trait::async_trait;
