//! Traits for pluggable memory subsystem components.
//!
//! These traits define the interfaces that deduplication, migration,
//! and tiering implementations must satisfy.

/// Hook for content-addressable deduplication.
///
/// Implementations examine page content and report whether an identical
/// page already exists, enabling copy-on-write sharing.
pub trait DedupHook: Send + Sync {
    /// Check whether `data` (the content of virtual page `vpn`) already
    /// exists in the dedup table.
    ///
    /// Returns `Some(canonical_vpn)` if a match is found (the caller
    /// should create a reference mapping instead of storing a new copy).
    /// Returns `None` if the content is unique and a new entry was created.
    fn check_dedup(&self, vpn: u64, data: &[u8]) -> Option<u64>;

    /// Notify the dedup subsystem that virtual page `vpn` has been freed.
    ///
    /// This must update reference counts, promote references to canonical
    /// when necessary, and clean up empty entries.
    fn notify_free(&self, vpn: u64);
}
