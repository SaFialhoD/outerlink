//! Migration request types for the memory tiering system.

use crate::memory::types::TierId;

/// Priority level for a migration request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MigrationPriority {
    /// Background migration (prefetch, rebalancing)
    Low,
    /// Normal priority (tier demotion)
    Normal,
    /// High priority (page fault, active demand)
    High,
    /// Critical (blocking an active kernel)
    Critical,
}

/// A request to migrate a page between memory tiers.
#[derive(Debug, Clone)]
pub struct MigrationRequest {
    /// Unique migration ID
    pub id: u64,
    /// Virtual page number to migrate
    pub vpn: u64,
    /// Source tier
    pub source_tier: TierId,
    /// Destination tier
    pub dest_tier: TierId,
    /// Priority of this migration
    pub priority: MigrationPriority,
}
