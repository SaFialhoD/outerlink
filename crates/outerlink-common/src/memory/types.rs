//! Core memory subsystem type aliases.
//!
//! Lightweight newtypes for tier and node identifiers used throughout
//! the memory management layer.

/// Identifies a memory tier (0 = VRAM, 1 = host pinned, 2 = host paged, etc.)
pub type TierId = u8;

/// Identifies a node in the cluster.
pub type NodeId = u8;
