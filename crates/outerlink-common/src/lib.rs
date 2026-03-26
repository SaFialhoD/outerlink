//! OuterLink Common Library
//!
//! Shared types, protocol definitions, transport traits, and handle translation
//! used by both the client (interception library) and server (GPU daemon).

pub mod clock_sync;
pub mod cuda_types;
pub mod error;
pub mod fault_tolerance;
pub mod gpu_storage;
pub mod handle;
pub mod memory;
pub mod multicast;
pub mod protocol;
pub mod retry;
pub mod scatter_gather;
pub mod tcp_transport;
pub mod transport;

pub use error::{OuterLinkError, Result};
