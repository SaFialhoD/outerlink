//! Error types for OuterLink.
//!
//! Maps between CUDA errors, network errors, and OuterLink-specific errors.

use crate::cuda_types::CuResult;

/// Main error type for OuterLink operations.
#[derive(Debug, thiserror::Error)]
pub enum OuterLinkError {
    /// CUDA operation returned an error
    #[error("CUDA error: {0:?}")]
    Cuda(CuResult),

    /// Network/transport error
    #[error("Transport error: {0}")]
    Transport(String),

    /// Connection failed or lost
    #[error("Connection error: {0}")]
    Connection(String),

    /// Protocol error (malformed message, version mismatch)
    #[error("Protocol error: {0}")]
    Protocol(String),

    /// Handle not found in translation table
    #[error("Handle not found: {0:#x}")]
    HandleNotFound(u64),

    /// Operation timed out
    #[error("Operation timed out after {0}ms")]
    Timeout(u64),

    /// Server is not ready
    #[error("Server not ready: {0}")]
    NotReady(String),

    /// The remote peer closed the connection (clean disconnect).
    #[error("connection closed by peer")]
    ConnectionClosed,

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Configuration error
    #[error("Config error: {0}")]
    Config(String),
}

impl OuterLinkError {
    /// Convert to the closest CUresult for returning to CUDA applications.
    pub fn to_cuda_result(&self) -> CuResult {
        match self {
            Self::Cuda(r) => *r,
            Self::Transport(_) | Self::Connection(_) | Self::ConnectionClosed | Self::Io(_) => {
                CuResult::TransportError
            }
            Self::Protocol(_) => CuResult::RemoteError,
            Self::HandleNotFound(_) => CuResult::HandleNotFound,
            Self::Timeout(_) => CuResult::Timeout,
            Self::NotReady(_) => CuResult::SystemNotReady,
            Self::Config(_) => CuResult::InvalidValue,
        }
    }
}

/// Convenience Result type for OuterLink operations.
pub type Result<T> = std::result::Result<T, OuterLinkError>;
