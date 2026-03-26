//! Tier status, allocation, and eviction candidate types.
//!
//! These types represent the runtime state of each tier, allocation results,
//! and eviction/prefetch decisions.

use super::types::{AccessPatternType, PressureLevel, TierId};
use thiserror::Error;

/// Runtime status of a single memory tier.
#[derive(Debug, Clone)]
pub struct TierStatus {
    /// Tier identifier.
    pub tier_id: TierId,
    /// Total capacity in bytes.
    pub capacity_bytes: u64,
    /// Currently used bytes.
    pub used_bytes: u64,
    /// Number of pages currently stored in this tier.
    pub page_count: u64,
    /// Current memory pressure level.
    pub pressure: PressureLevel,
    /// Number of in-flight migrations from/to this tier.
    pub active_migrations: u32,
}

impl TierStatus {
    /// Fraction of capacity used (0.0 to 1.0).
    pub fn utilization(&self) -> f64 {
        if self.capacity_bytes == 0 {
            return 0.0;
        }
        self.used_bytes as f64 / self.capacity_bytes as f64
    }

    /// Free bytes remaining.
    pub fn free_bytes(&self) -> u64 {
        self.capacity_bytes.saturating_sub(self.used_bytes)
    }
}

/// Result of a successful page allocation in a tier.
#[derive(Debug, Clone, Copy)]
pub struct TierAllocation {
    /// The physical page frame number allocated.
    pub phys_pfn: u64,
    /// The tier where allocation was made.
    pub tier_id: TierId,
}

/// Hints provided when requesting an allocation.
#[derive(Debug, Clone, Default)]
pub struct AllocationHints {
    /// Preferred tier (if any).
    pub preferred_tier: Option<TierId>,
    /// Expected access pattern.
    pub access_pattern: Option<AccessPatternType>,
    /// Whether the page should be pinned immediately.
    pub pin: bool,
    /// Allocation size in bytes.
    pub size_bytes: u64,
}

/// Errors that can occur during tier allocation.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum AllocationError {
    /// No space available in the requested tier.
    #[error("tier {0} out of memory")]
    OutOfMemory(TierId),

    /// No space available in any tier.
    #[error("all tiers exhausted")]
    AllTiersExhausted,

    /// The requested tier does not exist.
    #[error("invalid tier id: {0}")]
    InvalidTier(TierId),
}

/// A page selected for eviction by the eviction policy.
#[derive(Debug, Clone)]
pub struct EvictionCandidate {
    /// Virtual page number of the candidate.
    pub vpn: u64,
    /// Current tier of the candidate.
    pub tier_id: TierId,
    /// Score assigned by the eviction policy (lower = better candidate).
    pub score: f64,
    /// Whether the page is dirty and needs writeback.
    pub dirty: bool,
}

/// A page predicted to be accessed soon by the access monitor.
#[derive(Debug, Clone)]
pub struct PrefetchPrediction {
    /// Virtual page number predicted to be accessed.
    pub vpn: u64,
    /// Confidence of the prediction (0.0 to 1.0).
    pub confidence: f64,
    /// Suggested tier to prefetch into.
    pub target_tier: TierId,
}
