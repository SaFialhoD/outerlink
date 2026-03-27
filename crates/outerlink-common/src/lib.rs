//! OuterLink Common Library
//!
//! Shared types, protocol definitions, transport traits, and handle translation
//! used by both the client (interception library) and server (GPU daemon).

pub mod clock_sync;
pub mod connection_pool;
pub mod cuda_graph;
pub mod cuda_types;
pub mod discovery;
pub mod dpu_offload;
pub mod error;
pub mod fault_tolerance;
pub mod gpu_mixing;
pub mod gpu_storage;
pub mod hip_interception;
pub mod handle;
pub mod health;
pub mod kernel_splitting;
pub mod live_migration;
pub mod memory;
pub mod metrics;
pub mod multicast;
pub mod network_resilience;
pub mod nvml_types;
pub mod persistent_kernels;
pub mod pinned_memory;
pub mod protocol;
pub mod retry;
pub mod scatter_gather;
pub mod security;
pub mod tcp_transport;
pub mod time_slicing;
pub mod transport;
pub mod virtual_nvlink;

pub use error::{OuterLinkError, Result};
