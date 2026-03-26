//! Page Table Entry (PTE) flags and structures for virtual memory management.
//!
//! These flags track the state of each virtual page in OuterLink's unified
//! virtual memory system, including deduplication metadata.

use bitflags::bitflags;

bitflags! {
    /// Flags stored in each page table entry.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct PteFlags: u32 {
        /// Page is present / valid.
        const PRESENT         = 1 << 0;
        /// Page is writable.
        const WRITABLE        = 1 << 1;
        /// Page is dirty (modified since last flush).
        const DIRTY           = 1 << 2;
        /// Page is pinned (cannot be evicted).
        const PINNED          = 1 << 3;
        /// Page is the canonical (master) copy for dedup.
        const DEDUP_CANONICAL = 1 << 4;
        /// Page is a dedup reference (shares content with canonical).
        const DEDUP_REFERENCE = 1 << 5;
        /// Page is being migrated between tiers.
        const MIGRATING       = 1 << 6;
    }
}

/// A page table entry for a single virtual page.
#[derive(Debug, Clone)]
pub struct PageTableEntry {
    /// Virtual page number.
    pub vpn: u64,
    /// Physical location: node id.
    pub node: u8,
    /// Physical location: tier id.
    pub tier: u8,
    /// PTE flags.
    pub flags: PteFlags,
    /// Content hash for deduplication (xxHash128). Zero if not computed.
    pub dedup_hash: u128,
}
