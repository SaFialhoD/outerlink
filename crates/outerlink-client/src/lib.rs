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

use std::sync::atomic::AtomicBool;

use outerlink_common::handle::HandleStore;

/// Global client state, initialized on first CUDA call.
pub struct OuterLinkClient {
    /// Handle translation tables
    pub handles: HandleStore,
    /// Server address
    pub server_addr: String,
    /// Connection state — AtomicBool so it can be mutated through a shared
    /// (&self) reference, which is required when the client lives in a OnceLock.
    pub connected: AtomicBool,
}

impl OuterLinkClient {
    /// Create a new client (not yet connected).
    pub fn new(server_addr: String) -> Self {
        Self {
            handles: HandleStore::new(),
            server_addr,
            connected: AtomicBool::new(false),
        }
    }
}

pub mod ffi;
