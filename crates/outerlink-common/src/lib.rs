//! OuterLink Common Library
//!
//! Shared types, protocol definitions, transport traits, and handle translation
//! used by both the client (interception library) and server (GPU daemon).

pub mod cuda_types;
pub mod error;
pub mod handle;
pub mod health;
pub mod protocol;
pub mod retry;
pub mod tcp_transport;
pub mod transport;

pub use error::{OuterLinkError, Result};
