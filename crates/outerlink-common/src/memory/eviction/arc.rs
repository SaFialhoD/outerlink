//! ARC (Adaptive Replacement Cache) eviction policy.
//!
//! Designed for VRAM tiers (0, 1) where the working set is small but access
//! patterns vary between recency-friendly and frequency-friendly workloads.
//!
//! ARC maintains four lists:
//! - T1: recently accessed pages (recency)
//! - T2: frequently accessed pages (pages re-accessed from T1)
//! - B1: ghost list for recently evicted T1 pages
//! - B2: ghost list for recently evicted T2 pages
//!
//! An adaptive parameter `p` in [0, max_size] controls the balance:
//! - B1 ghost hit -> increase p (favor recency, grow T1)
//! - B2 ghost hit -> decrease p (favor frequency, grow T2)
//!
//! Reference: Megiddo & Modha, "ARC: A Self-Tuning, Low Overhead Replacement Cache" (2003).

use std::collections::{HashSet, VecDeque};

use crate::memory::tier_status::EvictionCandidate;
use crate::memory::traits::EvictionPolicy;
use crate::memory::types::TierId;

/// ARC eviction policy.
pub struct ArcPolicy {
    max_size: usize,
    /// Recency list: LRU at front, MRU at back.
    t1: VecDeque<u64>,
    /// Frequency list: LRU at front, MRU at back.
    t2: VecDeque<u64>,
    /// Ghost list for recently evicted T1 pages.
    b1: VecDeque<u64>,
    /// Ghost list for recently evicted T2 pages.
    b2: VecDeque<u64>,
    /// Adaptive parameter: target size for T1. Range [0, max_size].
    p: f64,
    // Fast membership lookups.
    in_t1: HashSet<u64>,
    in_t2: HashSet<u64>,
    in_b1: HashSet<u64>,
    in_b2: HashSet<u64>,
    /// Pages that cannot be evicted.
    pinned: HashSet<u64>,
    /// Tier this policy manages (for EvictionCandidate).
    source_tier: TierId,
}

impl ArcPolicy {
    /// Create a new ARC policy with the given capacity and source tier.
    pub fn new(max_size: usize, source_tier: TierId) -> Self {
        Self {
            max_size,
            t1: VecDeque::new(),
            t2: VecDeque::new(),
            b1: VecDeque::new(),
            b2: VecDeque::new(),
            p: 0.0,
            in_t1: HashSet::new(),
            in_t2: HashSet::new(),
            in_b1: HashSet::new(),
            in_b2: HashSet::new(),
            pinned: HashSet::new(),
            source_tier,
        }
    }

    /// Current value of the adaptive parameter p.
    pub fn adaptive_param(&self) -> f64 {
        self.p
    }

    /// Evict the LRU page from T1, moving it to ghost B1.
    /// Returns the evicted VPN, or None if T1 is empty.
    fn evict_from_t1(&mut self) -> Option<u64> {
        self.evict_from_list_to_ghost(true)
    }

    /// Evict the LRU page from T2, moving it to ghost B2.
    fn evict_from_t2(&mut self) -> Option<u64> {
        self.evict_from_list_to_ghost(false)
    }

    /// Evict LRU from the specified list (true=T1, false=T2) to its ghost.
    /// Skips pinned pages.
    fn evict_from_list_to_ghost(&mut self, from_t1: bool) -> Option<u64> {
        let list = if from_t1 { &mut self.t1 } else { &mut self.t2 };
        let set = if from_t1 {
            &mut self.in_t1
        } else {
            &mut self.in_t2
        };

        // Scan from front (LRU) to find first non-pinned page.
        let len = list.len();
        for _ in 0..len {
            if let Some(vpn) = list.pop_front() {
                if self.pinned.contains(&vpn) {
                    // Move pinned page to back (effectively skip it).
                    list.push_back(vpn);
                    continue;
                }
                set.remove(&vpn);

                // Add to ghost list.
                let (ghost, ghost_set) = if from_t1 {
                    (&mut self.b1, &mut self.in_b1)
                } else {
                    (&mut self.b2, &mut self.in_b2)
                };
                ghost.push_back(vpn);
                ghost_set.insert(vpn);
                // Cap ghost list size.
                while ghost.len() > self.max_size {
                    if let Some(old) = ghost.pop_front() {
                        ghost_set.remove(&old);
                    }
                }
                return Some(vpn);
            }
        }
        None
    }

    /// Internal replacement: make room when T1+T2 is at capacity.
    fn replace(&mut self) {
        let t1_len = self.t1.len();
        if t1_len > 0 && (t1_len as f64 > self.p || (self.t2.is_empty() && t1_len > 0)) {
            self.evict_from_t1();
        } else if !self.t2.is_empty() {
            self.evict_from_t2();
        } else {
            self.evict_from_t1();
        }
    }
}

impl EvictionPolicy for ArcPolicy {
    fn record_access(&mut self, vpn: u64) {
        if self.in_t1.contains(&vpn) {
            // Promote from T1 to MRU of T2.
            // Remove from T1.
            self.t1.retain(|&v| v != vpn);
            self.in_t1.remove(&vpn);
            // Add to T2 MRU.
            self.t2.push_back(vpn);
            self.in_t2.insert(vpn);
        } else if self.in_t2.contains(&vpn) {
            // Move to MRU of T2.
            self.t2.retain(|&v| v != vpn);
            self.t2.push_back(vpn);
        }
        // If not in T1 or T2, access is a no-op (page not tracked).
    }

    fn record_insert(&mut self, vpn: u64) {
        // If already tracked, treat as access.
        if self.in_t1.contains(&vpn) || self.in_t2.contains(&vpn) {
            self.record_access(vpn);
            return;
        }

        // Check ghost lists for adaptive behavior.
        if self.in_b1.contains(&vpn) {
            // B1 ghost hit: increase p (favor recency / grow T1 target).
            let delta = if self.b2.len() >= self.b1.len() && self.b1.len() > 0 {
                (self.b2.len() as f64 / self.b1.len() as f64).max(1.0)
            } else {
                1.0
            };
            self.p = (self.p + delta).min(self.max_size as f64);

            // Remove from B1.
            self.b1.retain(|&v| v != vpn);
            self.in_b1.remove(&vpn);

            // Make room and insert into T2 (it was re-requested, so it's frequent).
            if self.t1.len() + self.t2.len() >= self.max_size {
                self.replace();
            }
            self.t2.push_back(vpn);
            self.in_t2.insert(vpn);
        } else if self.in_b2.contains(&vpn) {
            // B2 ghost hit: decrease p (favor frequency / grow T2 target).
            let delta = if self.b1.len() >= self.b2.len() && self.b2.len() > 0 {
                (self.b1.len() as f64 / self.b2.len() as f64).max(1.0)
            } else {
                1.0
            };
            self.p = (self.p - delta).max(0.0);

            // Remove from B2.
            self.b2.retain(|&v| v != vpn);
            self.in_b2.remove(&vpn);

            // Make room and insert into T2.
            if self.t1.len() + self.t2.len() >= self.max_size {
                self.replace();
            }
            self.t2.push_back(vpn);
            self.in_t2.insert(vpn);
        } else {
            // Completely new page. Insert into T1.
            if self.t1.len() + self.t2.len() >= self.max_size {
                self.replace();
            }
            // Also cap total directory size (T1+T2+B1+B2).
            if self.t1.len() + self.b1.len() >= self.max_size {
                // Remove oldest ghost from B1.
                if let Some(old) = self.b1.pop_front() {
                    self.in_b1.remove(&old);
                }
            }
            self.t1.push_back(vpn);
            self.in_t1.insert(vpn);
        }
    }

    fn record_remove(&mut self, vpn: u64) {
        if self.in_t1.remove(&vpn) {
            self.t1.retain(|&v| v != vpn);
        } else if self.in_t2.remove(&vpn) {
            self.t2.retain(|&v| v != vpn);
        }
        // Also remove from ghost lists if present.
        if self.in_b1.remove(&vpn) {
            self.b1.retain(|&v| v != vpn);
        }
        if self.in_b2.remove(&vpn) {
            self.b2.retain(|&v| v != vpn);
        }
        self.pinned.remove(&vpn);
    }

    fn select_victim(&mut self) -> Option<EvictionCandidate> {
        if self.t1.is_empty() && self.t2.is_empty() {
            return None;
        }

        // Try to evict based on ARC policy.
        let t1_len = self.t1.len();
        let vpn = if t1_len > 0
            && (t1_len as f64 > self.p || (self.t2.is_empty() && t1_len > 0))
        {
            // Try T1 first.
            self.evict_from_t1()
                .or_else(|| self.evict_from_t2())
        } else {
            // Try T2 first.
            self.evict_from_t2()
                .or_else(|| self.evict_from_t1())
        };

        vpn.map(|vpn| EvictionCandidate {
            vpn,
            source_tier: self.source_tier,
        })
    }

    fn is_ghost_hit(&self, vpn: u64) -> bool {
        self.in_b1.contains(&vpn) || self.in_b2.contains(&vpn)
    }

    fn tracked_count(&self) -> usize {
        self.t1.len() + self.t2.len()
    }

    fn memory_overhead(&self) -> usize {
        // Approximate: each VPN is 8 bytes, plus HashSet overhead (~48 bytes per entry).
        let entry_cost = 8 + 48;
        let list_entries = self.t1.len() + self.t2.len() + self.b1.len() + self.b2.len();
        let set_entries =
            self.in_t1.len() + self.in_t2.len() + self.in_b1.len() + self.in_b2.len();
        list_entries * 8 + set_entries * entry_cost + self.pinned.len() * entry_cost
    }

    fn reset(&mut self) {
        self.t1.clear();
        self.t2.clear();
        self.b1.clear();
        self.b2.clear();
        self.in_t1.clear();
        self.in_t2.clear();
        self.in_b1.clear();
        self.in_b2.clear();
        self.pinned.clear();
        self.p = 0.0;
    }

    fn pin(&mut self, vpn: u64) {
        self.pinned.insert(vpn);
    }

    fn unpin(&mut self, vpn: u64) {
        self.pinned.remove(&vpn);
    }
}
