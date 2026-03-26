//! Trait definitions for the memory tiering subsystem.
//!
//! These traits define the interfaces that concrete implementations must
//! satisfy. They cover page table operations, tier management, eviction
//! policies, migration, access monitoring, and optional hooks for
//! compression, deduplication, and parity.

use super::migration::{MigrationError, MigrationHandle, MigrationRequest, MigrationStatus};
use super::pte::{PageTableEntry, PteFlags};
use super::tier_status::{
    AllocationError, AllocationHints, EvictionCandidate, PrefetchPrediction, TierAllocation,
    TierStatus,
};
use super::types::{
    AccessPatternType, CoherencyField, MemcpyDirection, NodeId, TierId,
};

// ---------------------------------------------------------------------------
// Core traits
// ---------------------------------------------------------------------------

/// Virtual-to-physical page table with concurrent access support.
pub trait PageTable: Send + Sync {
    /// Look up a single VPN. Returns `None` if not mapped.
    fn lookup(&self, vpn: u64) -> Option<PageTableEntry>;

    /// Insert or update a PTE. Returns the previous entry if one existed.
    fn upsert(&self, entry: PageTableEntry) -> Option<PageTableEntry>;

    /// Remove a mapping by VPN. Returns the removed entry if it existed.
    fn remove(&self, vpn: u64) -> Option<PageTableEntry>;

    /// Atomically update flags on an existing entry.
    /// Returns `true` if the entry was found and updated.
    fn update_flags(&self, vpn: u64, set: PteFlags, clear: PteFlags) -> bool;

    /// Commit a completed migration: update tier_id, node_id, phys_pfn,
    /// clear MIGRATING flag, increment migration_count.
    fn commit_migration(
        &self,
        vpn: u64,
        new_tier: TierId,
        new_node: NodeId,
        new_phys_pfn: u64,
    ) -> bool;

    /// Set the coherency field on an existing entry.
    fn set_coherency(&self, vpn: u64, coherency: CoherencyField) -> bool;

    /// Set the dedup hash on an existing entry.
    fn set_dedup_hash(&self, vpn: u64, hash: u128) -> bool;

    /// Set the parity group on an existing entry.
    fn set_parity_group(&self, vpn: u64, group_id: u32) -> bool;

    /// Set the access pattern and prefetch delta on an existing entry.
    fn set_access_pattern(
        &self,
        vpn: u64,
        pattern: AccessPatternType,
        prefetch_delta: i16,
    ) -> bool;

    /// Scan all entries matching a predicate, returning up to `limit` results.
    fn scan(&self, predicate: &dyn Fn(&PageTableEntry) -> bool, limit: usize) -> Vec<PageTableEntry>;

    /// Number of entries in the page table.
    fn len(&self) -> usize;

    /// Whether the page table is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Look up a contiguous range of VPNs [start_vpn, start_vpn + count).
    /// Returns entries that exist within the range (may be fewer than count).
    fn lookup_range(&self, start_vpn: u64, count: u64) -> Vec<PageTableEntry>;
}

/// Manages allocation and deallocation within tiers, drives promotion/demotion.
pub trait TierManager: Send + Sync {
    /// Allocate a page in a tier, returning the physical location.
    fn allocate(
        &self,
        tier_id: TierId,
        hints: &AllocationHints,
    ) -> Result<TierAllocation, AllocationError>;

    /// Deallocate a page from a tier.
    fn deallocate(&self, tier_id: TierId, phys_pfn: u64);

    /// Request promotion of a page to a higher (faster) tier.
    fn request_promotion(&self, vpn: u64, target_tier: TierId);

    /// Request demotion of a page to a lower (slower) tier.
    fn request_demotion(&self, vpn: u64, target_tier: TierId);

    /// Pin a page so it cannot be evicted.
    fn pin_page(&self, vpn: u64) -> bool;

    /// Unpin a previously pinned page.
    fn unpin_page(&self, vpn: u64) -> bool;

    /// Get the current status of a tier.
    fn tier_status(&self, tier_id: TierId) -> Option<TierStatus>;

    /// Get the remaining migration budget (bytes) for a tier.
    fn migration_budget(&self, tier_id: TierId) -> u64;

    /// Notify the tier manager of a page access (for adaptive policies).
    fn notify_access(&self, vpn: u64, tier_id: TierId);

    /// Register a compression hook.
    fn set_compression_hook(&self, hook: Box<dyn CompressionHook>);

    /// Register a deduplication hook.
    fn set_dedup_hook(&self, hook: Box<dyn DedupHook>);
}

/// Eviction policy that selects victim pages when a tier is under pressure.
pub trait EvictionPolicy: Send + Sync {
    /// Record that a page was accessed (hit).
    fn record_access(&mut self, vpn: u64);

    /// Record that a page was inserted into the tier.
    fn record_insert(&mut self, vpn: u64);

    /// Record that a page was removed from the tier.
    fn record_remove(&mut self, vpn: u64);

    /// Select up to `count` victims for eviction.
    fn select_victim(&self, count: usize) -> Vec<EvictionCandidate>;

    /// Check if a VPN is a "ghost hit" (recently evicted, being accessed again).
    /// Used by ARC/CAR to adapt partition sizes.
    fn is_ghost_hit(&self, vpn: u64) -> bool;

    /// Number of pages currently tracked by this policy.
    fn tracked_count(&self) -> usize;

    /// Approximate memory overhead of this policy in bytes.
    fn memory_overhead(&self) -> usize;

    /// Reset the policy state (e.g., after a major reconfiguration).
    fn reset(&mut self);
}

/// Asynchronous migration engine that moves pages between tiers/nodes.
pub trait MigrationEngine: Send + Sync {
    /// Submit a single migration request. Returns a handle for tracking.
    fn submit(&self, request: MigrationRequest) -> Result<MigrationHandle, MigrationError>;

    /// Submit a batch of migration requests.
    fn submit_batch(
        &self,
        requests: Vec<MigrationRequest>,
    ) -> Vec<Result<MigrationHandle, MigrationError>>;

    /// Poll the status of a migration by handle ID.
    fn poll(&self, handle_id: u64) -> Option<MigrationStatus>;

    /// Block until a migration completes or fails.
    fn wait(&self, handle_id: u64) -> Result<MigrationStatus, MigrationError>;

    /// Cancel an in-flight migration.
    fn cancel(&self, handle_id: u64) -> Result<(), MigrationError>;

    /// Number of currently in-flight migrations.
    fn in_flight_count(&self) -> usize;

    /// Current bandwidth utilization as a fraction (0.0 to 1.0).
    fn bandwidth_utilization(&self) -> f64;

    /// Set the maximum bandwidth (bytes/sec) for migrations.
    fn set_bandwidth_limit(&self, bytes_per_sec: u64);
}

/// Monitors memory access patterns for intelligent tiering decisions.
pub trait AccessMonitor: Send + Sync {
    /// Record a memcpy operation.
    fn record_memcpy(&self, src_vpn: u64, dst_vpn: u64, size_bytes: u64, direction: MemcpyDirection);

    /// Record that a kernel was launched touching the given pages.
    fn record_kernel_launch(&self, kernel_id: u64, vpns: &[u64]);

    /// Record that a kernel completed.
    fn record_kernel_complete(&self, kernel_id: u64);

    /// Get the detected access pattern for a VPN.
    fn get_pattern(&self, vpn: u64) -> AccessPatternType;

    /// Predict the next VPNs likely to be accessed.
    fn predict_next_accesses(&self, vpn: u64, count: usize) -> Vec<PrefetchPrediction>;

    /// Return VPNs considered "hot" (frequently accessed).
    fn hot_pages(&self, limit: usize) -> Vec<u64>;

    /// Return VPNs considered "cold" (rarely accessed).
    fn cold_pages(&self, limit: usize) -> Vec<u64>;
}

// ---------------------------------------------------------------------------
// Hook traits (optional extensions)
// ---------------------------------------------------------------------------

/// Hook for transparent compression of page data during migration/storage.
pub trait CompressionHook: Send + Sync {
    /// Try to compress page data. Returns compressed bytes or None if
    /// compression would not save space.
    fn try_compress(&self, data: &[u8]) -> Option<Vec<u8>>;

    /// Decompress previously compressed data.
    fn decompress(&self, compressed: &[u8], expected_size: usize) -> Result<Vec<u8>, String>;
}

/// Hook for deduplication of identical pages.
pub trait DedupHook: Send + Sync {
    /// Check if a page with the given hash already exists.
    /// Returns the VPN of the canonical copy if found.
    fn check_dedup(&self, hash: u128) -> Option<u64>;

    /// Notify that a deduplicated reference has been freed.
    fn notify_free(&self, hash: u128, vpn: u64);
}

/// Hook for parity/erasure coding across tiers.
pub trait ParityHook: Send + Sync {
    /// Notify that a page has been migrated, so parity may need updating.
    fn notify_migration(&self, vpn: u64, old_tier: TierId, new_tier: TierId);

    /// Rebuild a page from parity data after a failure.
    fn rebuild_page(&self, vpn: u64, parity_group_id: u32) -> Result<Vec<u8>, String>;
}
