//! Traits for the memory tiering subsystem.
//!
//! Defines the EvictionPolicy trait that all eviction algorithms must implement.

use super::tier_status::EvictionCandidate;

/// An eviction policy that tracks page accesses and selects victims for eviction.
///
/// Implementations must be Send + Sync to support concurrent access from
/// multiple CUDA streams (behind a lock in the page table).
pub trait EvictionPolicy: Send + Sync {
    /// Record an access to a page (read or write).
    fn record_access(&mut self, vpn: u64);

    /// Record a new page insertion into the tracked set.
    fn record_insert(&mut self, vpn: u64);

    /// Record removal of a page from the tracked set.
    fn record_remove(&mut self, vpn: u64);

    /// Select the best victim page for eviction.
    ///
    /// Returns `None` if no evictable page exists (e.g., all pages are pinned).
    fn select_victim(&mut self) -> Option<EvictionCandidate>;

    /// Check if a VPN is in a ghost list (recently evicted).
    ///
    /// Ghost hits inform the adaptive parameter in ARC/CAR policies.
    /// Policies without ghost lists always return false.
    fn is_ghost_hit(&self, vpn: u64) -> bool;

    /// Number of pages currently tracked (not including ghosts).
    fn tracked_count(&self) -> usize;

    /// Estimated memory overhead in bytes for this policy's bookkeeping.
    fn memory_overhead(&self) -> usize;

    /// Reset all state, clearing all tracked pages and ghosts.
    fn reset(&mut self);

    /// Mark a page as pinned (cannot be evicted).
    fn pin(&mut self, vpn: u64);

    /// Unpin a page (allow eviction again).
    fn unpin(&mut self, vpn: u64);
}
