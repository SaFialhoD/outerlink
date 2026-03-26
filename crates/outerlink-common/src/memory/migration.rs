//! Migration request and handle types.
//!
//! These types describe page migration operations between tiers, including
//! priority, status tracking, and error reporting.

use super::types::{NodeId, TierId};
use thiserror::Error;

/// Priority of a migration request, affecting scheduling order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum MigrationPriority {
    /// Background migration (prefetch, rebalancing).
    Low = 0,
    /// Normal priority (eviction-driven).
    Normal = 1,
    /// High priority (access pattern detected, promotion).
    High = 2,
    /// Critical priority (fault handling, must complete ASAP).
    Critical = 3,
}

/// Current status of a migration operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MigrationStatus {
    /// Queued but not yet started.
    Pending,
    /// Currently transferring data.
    InProgress,
    /// Successfully completed.
    Completed,
    /// Failed with an error.
    Failed,
    /// Cancelled before completion.
    Cancelled,
}

/// A request to migrate a page (or batch of pages) between tiers.
#[derive(Debug, Clone)]
pub struct MigrationRequest {
    /// Virtual page number to migrate.
    pub vpn: u64,
    /// Source tier.
    pub src_tier: TierId,
    /// Source node.
    pub src_node: NodeId,
    /// Destination tier.
    pub dst_tier: TierId,
    /// Destination node.
    pub dst_node: NodeId,
    /// Priority of this migration.
    pub priority: MigrationPriority,
    /// Reason for migration (for logging/debugging).
    pub reason: MigrationReason,
}

/// Why a migration was requested.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MigrationReason {
    /// Eviction due to memory pressure.
    Eviction,
    /// Promotion due to hot access pattern.
    Promotion,
    /// Demotion due to cold access pattern.
    Demotion,
    /// Prefetch based on predicted access.
    Prefetch,
    /// Explicit user/application request.
    UserRequest,
    /// Rebalancing across nodes.
    Rebalance,
}

/// Handle to track an in-flight migration operation.
#[derive(Debug, Clone)]
pub struct MigrationHandle {
    /// Unique identifier for this migration.
    pub id: u64,
    /// The original request.
    pub request: MigrationRequest,
    /// Current status.
    pub status: MigrationStatus,
}

/// Errors that can occur during migration.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum MigrationError {
    /// Source page not found in the page table.
    #[error("source page vpn={0:#x} not found")]
    SourceNotFound(u64),

    /// Destination tier is full, cannot allocate.
    #[error("destination tier {0} is full")]
    DestinationFull(TierId),

    /// Page is pinned and cannot be migrated.
    #[error("page vpn={0:#x} is pinned")]
    PagePinned(u64),

    /// Page is already being migrated.
    #[error("page vpn={0:#x} is already migrating")]
    AlreadyMigrating(u64),

    /// Network error during remote migration.
    #[error("network error: {0}")]
    NetworkError(String),

    /// Migration was cancelled.
    #[error("migration id={0} was cancelled")]
    Cancelled(u64),

    /// The migration bandwidth limit has been reached.
    #[error("bandwidth limit reached")]
    BandwidthLimitReached,
}
