//! Tests for the memory tiering module.

use super::*;
use std::thread;

// -----------------------------------------------------------------------
// 1. PTE size and alignment static assertions
// -----------------------------------------------------------------------

#[test]
fn pte_is_64_bytes() {
    assert_eq!(std::mem::size_of::<PageTableEntry>(), 64);
}

#[test]
fn pte_is_64_byte_aligned() {
    assert_eq!(std::mem::align_of::<PageTableEntry>(), 64);
}

// -----------------------------------------------------------------------
// 2. phys_pfn pack/unpack roundtrip
// -----------------------------------------------------------------------

#[test]
fn phys_pfn_roundtrip_zero() {
    let mut pte = PageTableEntry::default();
    pte.set_phys_pfn(0);
    assert_eq!(pte.get_phys_pfn(), 0);
}

#[test]
fn phys_pfn_roundtrip_max_48bit() {
    let mut pte = PageTableEntry::default();
    let max_48 = (1u64 << 48) - 1; // 0xFFFF_FFFF_FFFF
    pte.set_phys_pfn(max_48);
    assert_eq!(pte.get_phys_pfn(), max_48);
}

#[test]
fn phys_pfn_roundtrip_typical_value() {
    let mut pte = PageTableEntry::default();
    let val = 0x0000_DEAD_BEEF_1234u64;
    pte.set_phys_pfn(val);
    assert_eq!(pte.get_phys_pfn(), val);
}

#[test]
fn phys_pfn_truncates_upper_16_bits() {
    let mut pte = PageTableEntry::default();
    // Set a value with upper bits set - they should be discarded
    let val_with_upper = 0xABCD_0000_1234_5678u64;
    pte.set_phys_pfn(val_with_upper);
    // Only the lower 48 bits survive
    assert_eq!(pte.get_phys_pfn(), val_with_upper & 0x0000_FFFF_FFFF_FFFF);
}

// -----------------------------------------------------------------------
// 3. CoherencyField encode/decode roundtrip
// -----------------------------------------------------------------------

#[test]
fn coherency_field_state_roundtrip() {
    for state in [
        CoherencyState::Modified,
        CoherencyState::Owned,
        CoherencyState::Exclusive,
        CoherencyState::Shared,
        CoherencyState::Invalid,
    ] {
        let field = CoherencyField::new(state);
        assert_eq!(field.state(), state);
        assert_eq!(field.sharer_mask(), 0);
    }
}

#[test]
fn coherency_field_with_sharers() {
    let field = CoherencyField::with_sharers(CoherencyState::Shared, 0b1010_0101);
    assert_eq!(field.state(), CoherencyState::Shared);
    assert_eq!(field.sharer_mask(), 0b1010_0101);
}

#[test]
fn coherency_field_add_remove_sharer() {
    let mut field = CoherencyField::new(CoherencyState::Shared);
    assert!(!field.has_sharer(0));
    assert!(!field.has_sharer(3));

    field.add_sharer(0);
    field.add_sharer(3);
    assert!(field.has_sharer(0));
    assert!(field.has_sharer(3));
    assert!(!field.has_sharer(1));

    field.remove_sharer(0);
    assert!(!field.has_sharer(0));
    assert!(field.has_sharer(3));
}

#[test]
fn coherency_field_set_state_preserves_sharers() {
    let mut field = CoherencyField::with_sharers(CoherencyState::Exclusive, 0xFF);
    field.set_state(CoherencyState::Modified);
    assert_eq!(field.state(), CoherencyState::Modified);
    assert_eq!(field.sharer_mask(), 0xFF);
}

// -----------------------------------------------------------------------
// 4. PteFlags bitwise operations
// -----------------------------------------------------------------------

#[test]
fn pte_flags_individual_bits() {
    let flags = PteFlags::VALID | PteFlags::DIRTY | PteFlags::PINNED;
    assert!(flags.contains(PteFlags::VALID));
    assert!(flags.contains(PteFlags::DIRTY));
    assert!(flags.contains(PteFlags::PINNED));
    assert!(!flags.contains(PteFlags::MIGRATING));
}

#[test]
fn pte_flags_set_and_clear() {
    let mut pte = PageTableEntry::new(0x1000, LOCAL_VRAM, 0);
    assert!(pte.has_flag(PteFlags::VALID)); // new() sets VALID

    pte.set_flags(PteFlags::DIRTY | PteFlags::ACCESSED);
    assert!(pte.has_flag(PteFlags::DIRTY));
    assert!(pte.has_flag(PteFlags::ACCESSED));

    pte.clear_flags(PteFlags::DIRTY);
    assert!(!pte.has_flag(PteFlags::DIRTY));
    assert!(pte.has_flag(PteFlags::ACCESSED)); // not cleared
}

#[test]
fn pte_flags_all_defined_flags() {
    // Verify all 16 flags have distinct bit positions
    let all_flags = [
        PteFlags::VALID,
        PteFlags::DIRTY,
        PteFlags::PINNED,
        PteFlags::MIGRATING,
        PteFlags::SHARED,
        PteFlags::SUPERPAGE_MEMBER,
        PteFlags::READ_ONLY,
        PteFlags::ACCESSED,
        PteFlags::EVICTION_CANDIDATE,
        PteFlags::PREFETCH_TARGET,
        PteFlags::COMPRESSED,
        PteFlags::DEDUP_CANONICAL,
        PteFlags::DEDUP_REFERENCE,
        PteFlags::PARITY_VALID,
        PteFlags::NCCL_REGISTERED,
        PteFlags::FAULT_PENDING,
    ];
    // All 16 flags combined should have exactly 16 bits set
    let combined = all_flags.iter().fold(PteFlags::empty(), |acc, &f| acc | f);
    assert_eq!(combined.bits().count_ones(), 16);
}

// -----------------------------------------------------------------------
// 5. RobinHoodPageTable: insert, lookup, remove, update_flags, commit_migration
// -----------------------------------------------------------------------

#[test]
fn page_table_insert_and_lookup() {
    let pt = RobinHoodPageTable::new();
    let pte = PageTableEntry::new(0x1000, LOCAL_VRAM, 0);
    assert!(pt.upsert(pte).is_none()); // first insert

    let found = pt.lookup(0x1000).unwrap();
    assert_eq!(found.vpn, 0x1000);
    assert_eq!(found.tier_id, LOCAL_VRAM);
}

#[test]
fn page_table_upsert_returns_previous() {
    let pt = RobinHoodPageTable::new();
    let pte1 = PageTableEntry::new(0x1000, LOCAL_VRAM, 0);
    let pte2 = PageTableEntry::new(0x1000, REMOTE_VRAM, 1);

    assert!(pt.upsert(pte1).is_none());
    let prev = pt.upsert(pte2).unwrap();
    assert_eq!(prev.tier_id, LOCAL_VRAM);

    let current = pt.lookup(0x1000).unwrap();
    assert_eq!(current.tier_id, REMOTE_VRAM);
}

#[test]
fn page_table_remove() {
    let pt = RobinHoodPageTable::new();
    pt.upsert(PageTableEntry::new(0x1000, LOCAL_VRAM, 0));
    assert_eq!(pt.len(), 1);

    let removed = pt.remove(0x1000).unwrap();
    assert_eq!(removed.vpn, 0x1000);
    assert_eq!(pt.len(), 0);
    assert!(pt.lookup(0x1000).is_none());
}

#[test]
fn page_table_remove_nonexistent() {
    let pt = RobinHoodPageTable::new();
    assert!(pt.remove(0x9999).is_none());
}

#[test]
fn page_table_update_flags() {
    let pt = RobinHoodPageTable::new();
    pt.upsert(PageTableEntry::new(0x1000, LOCAL_VRAM, 0));

    // Set DIRTY, clear VALID
    assert!(pt.update_flags(0x1000, PteFlags::DIRTY, PteFlags::VALID));

    let pte = pt.lookup(0x1000).unwrap();
    assert!(pte.has_flag(PteFlags::DIRTY));
    assert!(!pte.has_flag(PteFlags::VALID));
}

#[test]
fn page_table_update_flags_nonexistent() {
    let pt = RobinHoodPageTable::new();
    assert!(!pt.update_flags(0x9999, PteFlags::DIRTY, PteFlags::empty()));
}

#[test]
fn page_table_commit_migration() {
    let pt = RobinHoodPageTable::new();
    let mut pte = PageTableEntry::new(0x1000, LOCAL_VRAM, 0);
    pte.set_phys_pfn(0xAAAA);
    pte.set_flags(PteFlags::MIGRATING);
    pt.upsert(pte);

    assert!(pt.commit_migration(0x1000, REMOTE_VRAM, 1, 0xBBBB));

    let updated = pt.lookup(0x1000).unwrap();
    assert_eq!(updated.tier_id, REMOTE_VRAM);
    assert_eq!(updated.node_id, 1);
    assert_eq!(updated.get_phys_pfn(), 0xBBBB);
    assert!(!updated.has_flag(PteFlags::MIGRATING));
    assert_eq!(updated.migration_count, 1);
}

#[test]
fn page_table_commit_migration_increments_count() {
    let pt = RobinHoodPageTable::new();
    let pte = PageTableEntry::new(0x1000, LOCAL_VRAM, 0);
    pt.upsert(pte);

    pt.commit_migration(0x1000, REMOTE_VRAM, 1, 0x1111);
    pt.commit_migration(0x1000, LOCAL_DRAM, 0, 0x2222);
    pt.commit_migration(0x1000, REMOTE_DRAM, 1, 0x3333);

    let updated = pt.lookup(0x1000).unwrap();
    assert_eq!(updated.migration_count, 3);
    assert_eq!(updated.tier_id, REMOTE_DRAM);
}

// -----------------------------------------------------------------------
// 6. RobinHoodPageTable: concurrent access
// -----------------------------------------------------------------------

#[test]
fn page_table_concurrent_insert_lookup() {
    use std::sync::Arc;

    let pt = Arc::new(RobinHoodPageTable::new());
    let num_threads = 8;
    let entries_per_thread = 100;

    // Spawn threads that insert entries
    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let pt = Arc::clone(&pt);
            thread::spawn(move || {
                for i in 0..entries_per_thread {
                    let vpn = (t as u64) * 10_000 + (i as u64);
                    let pte = PageTableEntry::new(vpn, (t % 6) as u8, 0);
                    pt.upsert(pte);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(pt.len(), num_threads * entries_per_thread);

    // Verify all entries are present
    for t in 0..num_threads {
        for i in 0..entries_per_thread {
            let vpn = (t as u64) * 10_000 + (i as u64);
            let pte = pt.lookup(vpn).unwrap();
            assert_eq!(pte.vpn, vpn);
        }
    }
}

#[test]
fn page_table_concurrent_mixed_operations() {
    use std::sync::Arc;

    let pt = Arc::new(RobinHoodPageTable::new());

    // Pre-populate
    for i in 0..100u64 {
        pt.upsert(PageTableEntry::new(i, LOCAL_VRAM, 0));
    }

    let handles: Vec<_> = (0..4)
        .map(|t| {
            let pt = Arc::clone(&pt);
            thread::spawn(move || {
                for i in 0..100u64 {
                    match t {
                        0 => {
                            // Reader
                            let _ = pt.lookup(i);
                        }
                        1 => {
                            // Writer (update flags)
                            pt.update_flags(i, PteFlags::ACCESSED, PteFlags::empty());
                        }
                        2 => {
                            // Inserter (new range)
                            let vpn = 1000 + i;
                            pt.upsert(PageTableEntry::new(vpn, REMOTE_VRAM, 1));
                        }
                        3 => {
                            // Migration committer
                            pt.commit_migration(i, LOCAL_DRAM, 0, i * 0x100);
                        }
                        _ => unreachable!(),
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Should have original 100 + 100 new = 200
    assert_eq!(pt.len(), 200);
}

// -----------------------------------------------------------------------
// 7. Default tier configs are valid
// -----------------------------------------------------------------------

#[test]
fn default_tier_configs_has_all_tiers() {
    let configs = default_tier_configs();
    assert_eq!(configs.len(), TIER_COUNT);
}

#[test]
fn default_tier_configs_correct_ids() {
    let configs = default_tier_configs();
    for (i, config) in configs.iter().enumerate() {
        assert_eq!(config.tier_id, i as TierId);
    }
}

#[test]
fn default_tier_configs_correct_ordering() {
    let configs = default_tier_configs();

    // Remote tiers should have higher latency than their local counterparts
    assert!(configs[LOCAL_VRAM as usize].latency_ns < configs[REMOTE_VRAM as usize].latency_ns);
    assert!(configs[LOCAL_DRAM as usize].latency_ns < configs[REMOTE_DRAM as usize].latency_ns);
    assert!(configs[LOCAL_NVME as usize].latency_ns < configs[REMOTE_NVME as usize].latency_ns);
    // NVMe (slowest tier) should have higher latency than VRAM
    assert!(configs[LOCAL_VRAM as usize].latency_ns < configs[REMOTE_NVME as usize].latency_ns);

    // VRAM tiers use ARC
    assert_eq!(configs[LOCAL_VRAM as usize].eviction_policy, EvictionPolicyType::Arc);
    assert_eq!(configs[REMOTE_VRAM as usize].eviction_policy, EvictionPolicyType::Arc);

    // DRAM tiers use CAR
    assert_eq!(configs[LOCAL_DRAM as usize].eviction_policy, EvictionPolicyType::Car);
    assert_eq!(configs[REMOTE_DRAM as usize].eviction_policy, EvictionPolicyType::Car);

    // NVMe tiers use CLOCK
    assert_eq!(configs[LOCAL_NVME as usize].eviction_policy, EvictionPolicyType::Clock);
    assert_eq!(configs[REMOTE_NVME as usize].eviction_policy, EvictionPolicyType::Clock);
}

#[test]
fn default_tier_configs_locality() {
    let configs = default_tier_configs();
    assert!(configs[LOCAL_VRAM as usize].is_local);
    assert!(!configs[REMOTE_VRAM as usize].is_local);
    assert!(configs[LOCAL_DRAM as usize].is_local);
    assert!(!configs[REMOTE_DRAM as usize].is_local);
    assert!(configs[LOCAL_NVME as usize].is_local);
    assert!(!configs[REMOTE_NVME as usize].is_local);
}

#[test]
fn default_tier_configs_nonzero_values() {
    let configs = default_tier_configs();
    for config in &configs {
        assert!(config.capacity_bytes > 0, "tier {} has zero capacity", config.tier_id);
        assert!(config.bandwidth_bytes_per_sec > 0, "tier {} has zero bandwidth", config.tier_id);
        assert!(config.latency_ns > 0, "tier {} has zero latency", config.tier_id);
        assert!(config.migration_batch_size > 0, "tier {} has zero batch size", config.tier_id);
    }
}

// -----------------------------------------------------------------------
// 8. Scan with predicate
// -----------------------------------------------------------------------

#[test]
fn page_table_scan_with_predicate() {
    let pt = RobinHoodPageTable::new();

    // Insert entries in different tiers
    for i in 0..10u64 {
        let tier = if i < 5 { LOCAL_VRAM } else { REMOTE_VRAM };
        pt.upsert(PageTableEntry::new(i, tier, 0));
    }

    // Scan for entries in LOCAL_VRAM only
    let local_entries = pt.scan(&|pte| pte.tier_id == LOCAL_VRAM, 100);
    assert_eq!(local_entries.len(), 5);
    for entry in &local_entries {
        assert_eq!(entry.tier_id, LOCAL_VRAM);
    }
}

#[test]
fn page_table_scan_respects_limit() {
    let pt = RobinHoodPageTable::new();
    for i in 0..100u64 {
        pt.upsert(PageTableEntry::new(i, LOCAL_VRAM, 0));
    }

    let results = pt.scan(&|_| true, 10);
    assert_eq!(results.len(), 10);
}

#[test]
fn page_table_scan_empty_result() {
    let pt = RobinHoodPageTable::new();
    for i in 0..10u64 {
        pt.upsert(PageTableEntry::new(i, LOCAL_VRAM, 0));
    }

    // No entries match tier 5
    let results = pt.scan(&|pte| pte.tier_id == REMOTE_NVME, 100);
    assert_eq!(results.len(), 0);
}

// -----------------------------------------------------------------------
// Additional: lookup_range, set_coherency, set_dedup_hash, set_parity_group
// -----------------------------------------------------------------------

#[test]
fn page_table_lookup_range() {
    let pt = RobinHoodPageTable::new();
    // Insert VPNs 10, 11, 12, 13, 14 (skip 15), 16
    for vpn in [10, 11, 12, 13, 14, 16] {
        pt.upsert(PageTableEntry::new(vpn, LOCAL_VRAM, 0));
    }

    let range = pt.lookup_range(10, 7); // 10..17
    // Should find 10, 11, 12, 13, 14, 16 (6 entries, 15 is missing)
    assert_eq!(range.len(), 6);
}

#[test]
fn page_table_set_coherency() {
    let pt = RobinHoodPageTable::new();
    pt.upsert(PageTableEntry::new(0x1000, LOCAL_VRAM, 0));

    let coh = CoherencyField::with_sharers(CoherencyState::Shared, 0b0000_0011);
    assert!(pt.set_coherency(0x1000, coh));

    let pte = pt.lookup(0x1000).unwrap();
    assert_eq!(pte.coherency.state(), CoherencyState::Shared);
    assert!(pte.coherency.has_sharer(0));
    assert!(pte.coherency.has_sharer(1));
    assert!(!pte.coherency.has_sharer(2));
}

#[test]
fn page_table_set_dedup_hash() {
    let pt = RobinHoodPageTable::new();
    pt.upsert(PageTableEntry::new(0x1000, LOCAL_VRAM, 0));

    let hash: u128 = 0xDEAD_BEEF_CAFE_BABE_1234_5678_9ABC_DEF0;
    assert!(pt.set_dedup_hash(0x1000, hash));

    let pte = pt.lookup(0x1000).unwrap();
    assert_eq!(pte.get_dedup_hash(), hash);
}

#[test]
fn page_table_set_access_pattern() {
    let pt = RobinHoodPageTable::new();
    pt.upsert(PageTableEntry::new(0x1000, LOCAL_VRAM, 0));

    assert!(pt.set_access_pattern(0x1000, AccessPatternType::Sequential, 1));

    let pte = pt.lookup(0x1000).unwrap();
    assert_eq!(pte.access_pattern_type, AccessPatternType::Sequential);
    assert_eq!(pte.prefetch_next_vpn_delta, 1);
}

#[test]
fn page_table_is_empty() {
    let pt = RobinHoodPageTable::new();
    assert!(pt.is_empty());
    pt.upsert(PageTableEntry::new(1, LOCAL_VRAM, 0));
    assert!(!pt.is_empty());
}
