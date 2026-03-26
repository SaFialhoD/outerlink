//! Memory management subsystem.
//!
//! Provides virtual memory types, page table entries, deduplication,
//! and trait definitions for tiering and migration hooks.

pub mod dedup;
pub mod pte;
pub mod traits;
pub mod types;

pub use dedup::{DedupConfig, DedupManager, DedupStats};
pub use pte::{PageTableEntry, PteFlags};
pub use traits::DedupHook;
pub use types::{NodeId, TierId};
