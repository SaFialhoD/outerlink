//! Page Table Entry (PTE) for the virtual memory system.
//!
//! Each PTE is exactly 64 bytes with 64-byte alignment, designed for cache-line
//! friendly access. The `phys_pfn` field is packed as 6 bytes (u48) to keep the
//! total at 64 bytes. Fields are ordered to eliminate alignment padding under
//! `repr(C)` rules.

use super::types::{AccessPatternType, CoherencyField, NodeId, TierId};
use bitflags::bitflags;

bitflags! {
    /// Page table entry flags stored as a u32 bitfield.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct PteFlags: u32 {
        /// Entry contains a valid mapping.
        const VALID              = 1 << 0;
        /// Page has been written to (dirty).
        const DIRTY              = 1 << 1;
        /// Page is pinned and cannot be evicted.
        const PINNED             = 1 << 2;
        /// Page is currently being migrated to another tier.
        const MIGRATING          = 1 << 3;
        /// Page is shared across multiple allocations or nodes.
        const SHARED             = 1 << 4;
        /// Page is part of a superpage (large page).
        const SUPERPAGE_MEMBER   = 1 << 5;
        /// Page is read-only.
        const READ_ONLY          = 1 << 6;
        /// Page has been accessed since last clear.
        const ACCESSED           = 1 << 7;
        /// Page has been selected as an eviction candidate.
        const EVICTION_CANDIDATE = 1 << 8;
        /// Page is targeted for prefetching.
        const PREFETCH_TARGET    = 1 << 9;
        /// Page data is stored compressed.
        const COMPRESSED         = 1 << 10;
        /// This is the canonical (original) copy of a deduplicated page.
        const DEDUP_CANONICAL    = 1 << 11;
        /// This page references another page's data (dedup).
        const DEDUP_REFERENCE    = 1 << 12;
        /// Parity/erasure coding data is valid for this page.
        const PARITY_VALID       = 1 << 13;
        /// Page is registered with NCCL for collective operations.
        const NCCL_REGISTERED    = 1 << 14;
        /// A page fault is pending resolution for this page.
        const FAULT_PENDING      = 1 << 15;
    }
}

/// A single page table entry, exactly 64 bytes, 64-byte aligned.
///
/// This is the core data structure for virtual-to-physical page mapping
/// in OuterLink's memory tiering system. Fields are ordered to eliminate
/// all alignment padding under `repr(C)` layout rules:
///
/// | Offset | Size | Field                    |
/// |--------|------|--------------------------|
/// | 0      | 8    | vpn: u64                 |
/// | 8      | 1    | tier_id: u8              |
/// | 9      | 1    | node_id: u8              |
/// | 10     | 6    | phys_pfn: [u8; 6]        |
/// | 16     | 4    | flags: PteFlags (u32)    |
/// | 20     | 2    | coherency: u16           |
/// | 22     | 2    | migration_count: u16     |
/// | 24     | 4    | access_count: u32        |
/// | 28     | 4    | last_access_ts: u32      |
/// | 32     | 4    | alloc_id: u32            |
/// | 36     | 2    | last_migration_ts_delta  |
/// | 38     | 1    | preferred_tier: u8       |
/// | 39     | 1    | access_pattern_type: u8  |
/// | 40     | 16   | dedup_hash: [u8; 16]     |
/// | 56     | 4    | parity_group_id: u32     |
/// | 60     | 2    | prefetch_next_vpn_delta  |
/// | 62     | 2    | ref_count: u16           |
/// | Total  | 64   |                          |
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct PageTableEntry {
    /// Virtual page number (the key for lookups).
    pub vpn: u64,
    /// Which tier this page resides in (0-5).
    pub tier_id: TierId,
    /// Which node owns this page.
    pub node_id: NodeId,
    /// Physical page frame number, packed as 6 bytes (48-bit).
    pub phys_pfn: [u8; 6],
    /// Status flags for this page.
    pub flags: PteFlags,
    /// MOESI coherency state + sharer mask.
    pub coherency: CoherencyField,
    /// How many times this page has been migrated between tiers.
    pub migration_count: u16,
    /// Number of accesses recorded by the access monitor.
    pub access_count: u32,
    /// Timestamp (epoch-relative seconds or tick count) of last access.
    pub last_access_ts: u32,
    /// Allocation ID this page belongs to (from cuMemAlloc).
    pub alloc_id: u32,
    /// Delta (in some unit) since last migration, for rate limiting.
    pub last_migration_ts_delta: u16,
    /// The tier this page should ideally reside in.
    pub preferred_tier: u8,
    /// Detected access pattern for this page.
    pub access_pattern_type: AccessPatternType,
    /// Content hash for deduplication (stored as [u8; 16] to avoid u128 alignment padding).
    pub dedup_hash: [u8; 16],
    /// Parity/erasure coding group this page belongs to.
    pub parity_group_id: u32,
    /// Delta to next predicted VPN for prefetching (signed).
    pub prefetch_next_vpn_delta: i16,
    /// Reference count (number of active users of this page).
    pub ref_count: u16,
}

// Static assertion: PTE must be exactly 64 bytes.
const _: () = assert!(std::mem::size_of::<PageTableEntry>() == 64);
// Static assertion: PTE must be 64-byte aligned.
const _: () = assert!(std::mem::align_of::<PageTableEntry>() == 64);

impl PageTableEntry {
    /// Create a new PTE with the given VPN and tier, all other fields zeroed.
    pub fn new(vpn: u64, tier_id: TierId, node_id: NodeId) -> Self {
        Self {
            vpn,
            tier_id,
            node_id,
            phys_pfn: [0; 6],
            flags: PteFlags::VALID,
            coherency: CoherencyField::default(),
            migration_count: 0,
            access_count: 0,
            last_access_ts: 0,
            alloc_id: 0,
            last_migration_ts_delta: 0,
            preferred_tier: tier_id,
            access_pattern_type: AccessPatternType::Unknown,
            dedup_hash: [0; 16],
            parity_group_id: 0,
            prefetch_next_vpn_delta: 0,
            ref_count: 0,
        }
    }

    /// Pack a u64 physical page frame number into the 6-byte phys_pfn field.
    ///
    /// Only the lower 48 bits are stored. The upper 16 bits are discarded.
    pub fn set_phys_pfn(&mut self, pfn: u64) {
        let bytes = pfn.to_le_bytes();
        self.phys_pfn.copy_from_slice(&bytes[..6]);
    }

    /// Unpack the 6-byte phys_pfn field into a u64.
    ///
    /// The upper 16 bits of the returned value are always zero.
    pub fn get_phys_pfn(&self) -> u64 {
        let mut bytes = [0u8; 8];
        bytes[..6].copy_from_slice(&self.phys_pfn);
        u64::from_le_bytes(bytes)
    }

    /// Get the dedup hash as a u128.
    pub fn get_dedup_hash(&self) -> u128 {
        u128::from_le_bytes(self.dedup_hash)
    }

    /// Set the dedup hash from a u128.
    pub fn set_dedup_hash(&mut self, hash: u128) {
        self.dedup_hash = hash.to_le_bytes();
    }

    /// Check if a specific flag is set.
    pub fn has_flag(&self, flag: PteFlags) -> bool {
        self.flags.contains(flag)
    }

    /// Set specific flags (OR with existing).
    pub fn set_flags(&mut self, flags: PteFlags) {
        self.flags |= flags;
    }

    /// Clear specific flags.
    pub fn clear_flags(&mut self, flags: PteFlags) {
        self.flags &= !flags;
    }
}

impl std::fmt::Debug for PageTableEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PageTableEntry")
            .field("vpn", &format_args!("{:#x}", self.vpn))
            .field("tier_id", &self.tier_id)
            .field("node_id", &self.node_id)
            .field("phys_pfn", &format_args!("{:#x}", self.get_phys_pfn()))
            .field("flags", &self.flags)
            .field("coherency", &self.coherency)
            .field("access_count", &self.access_count)
            .field("ref_count", &self.ref_count)
            .finish_non_exhaustive()
    }
}

impl Default for PageTableEntry {
    fn default() -> Self {
        Self {
            vpn: 0,
            tier_id: 0,
            node_id: 0,
            phys_pfn: [0; 6],
            flags: PteFlags::empty(),
            coherency: CoherencyField::default(),
            migration_count: 0,
            access_count: 0,
            last_access_ts: 0,
            alloc_id: 0,
            last_migration_ts_delta: 0,
            preferred_tier: 0,
            access_pattern_type: AccessPatternType::Unknown,
            dedup_hash: [0; 16],
            parity_group_id: 0,
            prefetch_next_vpn_delta: 0,
            ref_count: 0,
        }
    }
}
