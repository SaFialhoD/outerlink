//! Core types for the memory tiering subsystem.
//!
//! Defines tier identifiers and access pattern classifications used throughout
//! the memory management layer.

use serde::{Deserialize, Serialize};

/// Identifies a memory tier in the hierarchy.
///
/// Tiers are numbered 0-5:
/// - 0, 1: VRAM (local GPU, remote GPU)
/// - 2, 3: DRAM (local host, remote host)
/// - 4, 5: NVMe (local SSD, remote SSD)
pub type TierId = u8;

/// VRAM tier range (local and remote GPU memory).
pub const VRAM_TIERS: std::ops::RangeInclusive<TierId> = 0..=1;

/// DRAM tier range (local and remote host memory).
pub const DRAM_TIERS: std::ops::RangeInclusive<TierId> = 2..=3;

/// NVMe tier range (local and remote SSD).
pub const NVME_TIERS: std::ops::RangeInclusive<TierId> = 4..=5;

/// Classifies the access pattern of a memory region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccessPatternType {
    /// Sequential reads/writes (e.g., streaming data).
    Sequential,
    /// Random access with no discernible pattern.
    Random,
    /// Strided access (regular intervals).
    Strided,
    /// Read-mostly with infrequent writes.
    ReadMostly,
    /// Write-heavy workload.
    WriteHeavy,
    /// Unknown or not yet classified.
    Unknown,
}
