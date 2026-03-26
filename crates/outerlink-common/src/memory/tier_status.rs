//! Tier status types for eviction and migration decisions.

use super::types::TierId;

/// A candidate page selected for eviction from a tier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvictionCandidate {
    /// Virtual page number of the page to evict.
    pub vpn: u64,
    /// The tier this page is being evicted from.
    pub source_tier: TierId,
}
