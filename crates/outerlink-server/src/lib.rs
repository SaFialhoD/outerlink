//! OuterLink Server library.
//!
//! Provides the GPU backend abstraction, request handler, and server loop
//! for the OuterLink GPU node daemon.

pub mod cuda_backend;
pub mod gpu_backend;
pub mod handler;
pub mod session;
