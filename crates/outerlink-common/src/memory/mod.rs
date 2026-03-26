//! Memory subsystem: types, traits, and compression.
//!
//! This module provides the memory tier abstractions and transport compression
//! used by OuterLink's memory management layer.

pub mod compression;
pub mod traits;
pub mod types;

pub use compression::{AdaptiveCompressor, CompressionAlgorithm, CompressionConfig};
pub use traits::CompressionHook;
pub use types::TierId;
