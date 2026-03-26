//! RobinHoodPageTable: concurrent page table backed by DashMap.
//!
//! Uses `dashmap` for lock-free concurrent reads and sharded writes.
//! Keyed by VPN (u64), values are full PageTableEntry structs.

use dashmap::DashMap;

use super::pte::{PageTableEntry, PteFlags};
use super::traits::PageTable;
use super::types::{AccessPatternType, CoherencyField, NodeId, TierId};

/// A concurrent page table implementation using DashMap (sharded hash map).
///
/// Named after Robin Hood hashing which DashMap uses internally for
/// probe sequence optimization.
#[derive(Debug)]
pub struct RobinHoodPageTable {
    /// The underlying concurrent hash map: VPN -> PTE.
    map: DashMap<u64, PageTableEntry>,
}

impl RobinHoodPageTable {
    /// Create a new empty page table.
    pub fn new() -> Self {
        Self {
            map: DashMap::new(),
        }
    }

    /// Create a new page table with a pre-allocated capacity hint.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: DashMap::with_capacity(capacity),
        }
    }
}

impl Default for RobinHoodPageTable {
    fn default() -> Self {
        Self::new()
    }
}

impl PageTable for RobinHoodPageTable {
    fn lookup(&self, vpn: u64) -> Option<PageTableEntry> {
        self.map.get(&vpn).map(|entry| *entry)
    }

    fn upsert(&self, entry: PageTableEntry) -> Option<PageTableEntry> {
        let vpn = entry.vpn;
        self.map.insert(vpn, entry)
    }

    fn remove(&self, vpn: u64) -> Option<PageTableEntry> {
        self.map.remove(&vpn).map(|(_, entry)| entry)
    }

    fn update_flags(&self, vpn: u64, set: PteFlags, clear: PteFlags) -> bool {
        if let Some(mut entry) = self.map.get_mut(&vpn) {
            entry.flags &= !clear;
            entry.flags |= set;
            true
        } else {
            false
        }
    }

    fn commit_migration(
        &self,
        vpn: u64,
        new_tier: TierId,
        new_node: NodeId,
        new_phys_pfn: u64,
    ) -> bool {
        if let Some(mut entry) = self.map.get_mut(&vpn) {
            entry.tier_id = new_tier;
            entry.node_id = new_node;
            entry.set_phys_pfn(new_phys_pfn);
            entry.flags &= !PteFlags::MIGRATING;
            entry.migration_count = entry.migration_count.saturating_add(1);
            true
        } else {
            false
        }
    }

    fn set_coherency(&self, vpn: u64, coherency: CoherencyField) -> bool {
        if let Some(mut entry) = self.map.get_mut(&vpn) {
            entry.coherency = coherency;
            true
        } else {
            false
        }
    }

    fn set_dedup_hash(&self, vpn: u64, hash: u128) -> bool {
        if let Some(mut entry) = self.map.get_mut(&vpn) {
            entry.set_dedup_hash(hash);
            true
        } else {
            false
        }
    }

    fn set_parity_group(&self, vpn: u64, group_id: u32) -> bool {
        if let Some(mut entry) = self.map.get_mut(&vpn) {
            entry.parity_group_id = group_id;
            true
        } else {
            false
        }
    }

    fn set_access_pattern(
        &self,
        vpn: u64,
        pattern: AccessPatternType,
        prefetch_delta: i16,
    ) -> bool {
        if let Some(mut entry) = self.map.get_mut(&vpn) {
            entry.access_pattern_type = pattern;
            entry.prefetch_next_vpn_delta = prefetch_delta;
            true
        } else {
            false
        }
    }

    fn scan(&self, predicate: &dyn Fn(&PageTableEntry) -> bool, limit: usize) -> Vec<PageTableEntry> {
        let mut results = Vec::with_capacity(limit.min(64));
        for entry in self.map.iter() {
            if results.len() >= limit {
                break;
            }
            if predicate(entry.value()) {
                results.push(*entry.value());
            }
        }
        results
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn lookup_range(&self, start_vpn: u64, count: u64) -> Vec<PageTableEntry> {
        let mut results = Vec::new();
        for vpn in start_vpn..start_vpn.saturating_add(count) {
            if let Some(entry) = self.map.get(&vpn) {
                results.push(*entry);
            }
        }
        results
    }
}
