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
    /// Streaming access (sequential, consumed once).
    Streaming = 1,
    /// Working set (re-accessed frequently, fits in a tier).
    WorkingSet = 2,
    /// Phased access (active for a period, then cold).
    Phased = 3,
    /// Random (unpredictable) access.
    Random = 4,
    /// Strided access with a fixed step.
    Strided = 5,
}

impl From<u8> for AccessPatternType {
    fn from(v: u8) -> Self {
        match v {
            0 => Self::Unknown,
            1 => Self::Streaming,
            2 => Self::WorkingSet,
            3 => Self::Phased,
            4 => Self::Random,
            5 => Self::Strided,
            _ => Self::Unknown,
        }
    }
}

/// Coherency state for a page: simplified I/S/E (3-state) protocol.
///
/// Only 2 bits needed (values 0-2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CoherencyState {
    /// Invalid: not usable / no valid copy.
    Invalid = 0,
    /// Shared: multiple readers, clean.
    Shared = 1,
    /// Exclusive: sole owner (may be dirty).
    Exclusive = 2,
}

impl From<u8> for CoherencyState {
    fn from(v: u8) -> Self {
        match v {
            0 => Self::Invalid,
            1 => Self::Shared,
            2 => Self::Exclusive,
            _ => Self::Invalid,
        }
    }
}

/// Packed coherency field (u16).
///
/// Layout (I/S/E with 2-bit state):
/// - Bits [0..2]: CoherencyState (2 bits, values 0-2)
/// - Bits [2..10]: sharer_bitmap (8 bits, one per node up to 8 nodes)
/// - Bits [10..16]: reserved (6 bits)
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct CoherencyField(pub u16);

impl CoherencyField {
    /// Mask for the 2-bit state field.
    const STATE_MASK: u16 = 0x03;
    /// Mask for the 8-bit sharer bitmap (bits 2..10).
    const SHARER_MASK: u16 = 0x03FC;
    /// Bit offset where the sharer bitmap starts.
    const SHARER_SHIFT: u32 = 2;

    /// Create a new coherency field with the given state and no sharers.
    pub fn new(state: CoherencyState) -> Self {
        Self(state as u16)
    }

    /// Create a coherency field with a specific state and sharer bitmap.
    pub fn with_sharers(state: CoherencyState, sharer_bitmap: u8) -> Self {
        Self((state as u16) | ((sharer_bitmap as u16) << Self::SHARER_SHIFT))
    }

    /// Extract the coherency state (lower 2 bits).
    pub fn state(self) -> CoherencyState {
        CoherencyState::from((self.0 & Self::STATE_MASK) as u8)
    }

    /// Extract the sharer bitmap (bits 2..10).
    pub fn sharer_mask(self) -> u8 {
        ((self.0 >> Self::SHARER_SHIFT) & 0xFF) as u8
    }

    /// Set the coherency state, preserving the sharer bitmap.
    pub fn set_state(&mut self, state: CoherencyState) {
        self.0 = (self.0 & !Self::STATE_MASK) | (state as u16);
    }

    /// Set the sharer bitmap, preserving the state.
    pub fn set_sharer_mask(&mut self, mask: u8) {
        self.0 = (self.0 & !Self::SHARER_MASK) | ((mask as u16) << Self::SHARER_SHIFT);
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
