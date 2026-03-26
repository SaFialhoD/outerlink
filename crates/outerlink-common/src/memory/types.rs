//! Core memory tiering types.

use serde::{Deserialize, Serialize};

/// Identifies a memory tier (0 = local VRAM, 1 = remote VRAM, 2 = host RAM, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TierId(pub u8);

impl TierId {
    pub const LOCAL_VRAM: TierId = TierId(0);
    pub const REMOTE_VRAM: TierId = TierId(1);
    pub const HOST_RAM: TierId = TierId(2);
}

/// Describes the type of access pattern detected for a memory region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccessPatternType {
    /// Sequential stride-based access (e.g., array traversal)
    Stride,
    /// Random access with no discernible pattern
    Random,
    /// Repeating access to the same pages (working set)
    WorkingSet,
    /// Kernel-correlated access pattern
    KernelCorrelated,
}
