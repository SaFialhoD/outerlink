//! Core types for the memory tiering system.
//!
//! Defines tier identifiers, node identifiers, tier constants, access pattern
//! classification, coherency state, pressure levels, and memcpy direction.

use std::fmt;

/// Tier identifier. Each tier in the 6-tier hierarchy has a unique u8 id.
pub type TierId = u8;

/// Node identifier. Each node (PC) in the cluster has a unique u8 id.
pub type NodeId = u8;

// --- Tier constants ---

/// Local GPU VRAM (fastest, most constrained).
pub const LOCAL_VRAM: TierId = 0;
/// Remote GPU VRAM (fast, accessed over network).
pub const REMOTE_VRAM: TierId = 1;
/// Local system DRAM (CPU-attached host memory).
pub const LOCAL_DRAM: TierId = 2;
/// Remote system DRAM (host memory on another node).
pub const REMOTE_DRAM: TierId = 3;
/// Local NVMe storage (persistent, high latency).
pub const LOCAL_NVME: TierId = 4;
/// Remote NVMe storage (persistent, highest latency).
pub const REMOTE_NVME: TierId = 5;
/// Total number of tiers in the hierarchy.
pub const TIER_COUNT: usize = 6;

/// Classification of memory access patterns detected by the access monitor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AccessPatternType {
    /// No pattern detected yet.
    Unknown = 0,
    /// Sequential (streaming) access.
    Sequential = 1,
    /// Strided access with a fixed step.
    Strided = 2,
    /// Random (unpredictable) access.
    Random = 3,
    /// Temporal locality (re-accessed frequently).
    Temporal = 4,
    /// Spatial locality (neighbors accessed together).
    Spatial = 5,
}

impl From<u8> for AccessPatternType {
    fn from(v: u8) -> Self {
        match v {
            0 => Self::Unknown,
            1 => Self::Sequential,
            2 => Self::Strided,
            3 => Self::Random,
            4 => Self::Temporal,
            5 => Self::Spatial,
            _ => Self::Unknown,
        }
    }
}

/// MOESI-inspired coherency state packed into the lower 3 bits of a u16.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CoherencyState {
    /// Modified: sole owner, dirty.
    Modified = 0,
    /// Owned: owner with sharers, dirty.
    Owned = 1,
    /// Exclusive: sole owner, clean.
    Exclusive = 2,
    /// Shared: multiple readers, clean.
    Shared = 3,
    /// Invalid: not usable.
    Invalid = 4,
}

impl From<u8> for CoherencyState {
    fn from(v: u8) -> Self {
        match v {
            0 => Self::Modified,
            1 => Self::Owned,
            2 => Self::Exclusive,
            3 => Self::Shared,
            4 => Self::Invalid,
            _ => Self::Invalid,
        }
    }
}

/// Packed coherency field (u16).
///
/// Layout:
/// - Bits [0..3]: CoherencyState (3 bits)
/// - Bits [3..11]: sharer_mask (8 bits, one per node up to 8 nodes)
/// - Bits [11..16]: reserved (5 bits)
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct CoherencyField(pub u16);

impl CoherencyField {
    /// Create a new coherency field with the given state and no sharers.
    pub fn new(state: CoherencyState) -> Self {
        Self(state as u16)
    }

    /// Create a coherency field with a specific state and sharer mask.
    pub fn with_sharers(state: CoherencyState, sharer_mask: u8) -> Self {
        Self((state as u16) | ((sharer_mask as u16) << 3))
    }

    /// Extract the coherency state (lower 3 bits).
    pub fn state(self) -> CoherencyState {
        CoherencyState::from((self.0 & 0x07) as u8)
    }

    /// Extract the sharer mask (bits 3..11).
    pub fn sharer_mask(self) -> u8 {
        ((self.0 >> 3) & 0xFF) as u8
    }

    /// Set the coherency state, preserving the sharer mask.
    pub fn set_state(&mut self, state: CoherencyState) {
        self.0 = (self.0 & !0x07) | (state as u16);
    }

    /// Set the sharer mask, preserving the state.
    pub fn set_sharer_mask(&mut self, mask: u8) {
        self.0 = (self.0 & !0x07F8) | ((mask as u16) << 3);
    }

    /// Add a node to the sharer set.
    pub fn add_sharer(&mut self, node: NodeId) {
        if node < 8 {
            let mask = self.sharer_mask() | (1 << node);
            self.set_sharer_mask(mask);
        }
    }

    /// Remove a node from the sharer set.
    pub fn remove_sharer(&mut self, node: NodeId) {
        if node < 8 {
            let mask = self.sharer_mask() & !(1 << node);
            self.set_sharer_mask(mask);
        }
    }

    /// Check if a node is in the sharer set.
    pub fn has_sharer(self, node: NodeId) -> bool {
        if node < 8 {
            (self.sharer_mask() & (1 << node)) != 0
        } else {
            false
        }
    }
}

impl fmt::Debug for CoherencyField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CoherencyField")
            .field("state", &self.state())
            .field("sharer_mask", &format_args!("{:#04x}", self.sharer_mask()))
            .finish()
    }
}

impl Default for CoherencyField {
    fn default() -> Self {
        Self::new(CoherencyState::Invalid)
    }
}

/// Memory pressure level for a tier, used by eviction and migration policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum PressureLevel {
    /// Plenty of free space.
    Low = 0,
    /// Approaching capacity, start background eviction.
    Medium = 1,
    /// Near full, aggressive eviction needed.
    High = 2,
    /// At capacity, must evict synchronously before allocating.
    Critical = 3,
}

/// Direction of a CUDA memcpy operation, mirroring cudaMemcpyKind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MemcpyDirection {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
}
