//! CAR (Clock with Adaptive Replacement) eviction policy.
//!
//! Designed for DRAM tiers (2, 3). Provides ARC-like adaptivity between
//! recency and frequency, but uses clock-based scanning for O(1) victim
//! selection instead of ARC's LRU list manipulation.
//!
//! Two clock buffers:
//! - T1: short-term utility (recently inserted pages)
//! - T2: long-term utility (pages promoted from T1)
//!
//! Ghost lists B1/B2 drive the adaptive parameter `p` exactly as in ARC.
//!
//! Reference: Bansal & Modha, "CAR: Clock with Adaptive Replacement" (2004).

use std::collections::{HashMap, HashSet, VecDeque};

use crate::memory::tier_status::EvictionCandidate;
use crate::memory::traits::EvictionPolicy;
use crate::memory::types::TierId;

/// Which list a page currently resides in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ListId {
    T1,
    T2,
}

/// A page entry in a clock buffer.
#[derive(Debug, Clone)]
struct ClockEntry {
    vpn: u64,
    ref_bit: bool,
}

/// CAR eviction policy.
pub struct CarPolicy {
    max_size: usize,
    /// T1 clock buffer (short-term / recency).
    t1: Vec<ClockEntry>,
    /// T2 clock buffer (long-term / frequency).
    t2: Vec<ClockEntry>,
    /// Clock hand position for T1.
    t1_hand: usize,
    /// Clock hand position for T2.
    t2_hand: usize,
    /// Ghost list for recently evicted T1 pages.
    b1: VecDeque<u64>,
    /// Ghost list for recently evicted T2 pages.
    b2: VecDeque<u64>,
    /// Adaptive parameter: target size for T1.
    p: f64,
    /// Maps VPN -> which list it's in.
    membership: HashMap<u64, ListId>,
    /// Fast ghost membership.
    in_b1: HashSet<u64>,
    in_b2: HashSet<u64>,
    /// Pinned pages (cannot be evicted).
    pinned: HashSet<u64>,
    source_tier: TierId,
}

impl CarPolicy {
    /// Create a new CAR policy with the given capacity and source tier.
    pub fn new(max_size: usize, source_tier: TierId) -> Self {
        Self {
            max_size,
            t1: Vec::new(),
            t2: Vec::new(),
            t1_hand: 0,
            t2_hand: 0,
            b1: VecDeque::new(),
            b2: VecDeque::new(),
            p: 0.0,
            membership: HashMap::new(),
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

    /// Find and remove a VPN from the T1 clock buffer.
    fn remove_from_t1(&mut self, vpn: u64) -> Option<ClockEntry> {
        if let Some(pos) = self.t1.iter().position(|e| e.vpn == vpn) {
            let entry = self.t1.remove(pos);
            // Adjust hand if needed.
            if !self.t1.is_empty() {
                if self.t1_hand > pos || self.t1_hand >= self.t1.len() {
                    self.t1_hand = if self.t1.is_empty() {
                        0
                    } else {
                        self.t1_hand.saturating_sub(1) % self.t1.len()
                    };
                }
            } else {
                self.t1_hand = 0;
            }
            Some(entry)
        } else {
            None
        }
    }

    /// Clock-scan T1 to find a victim. Returns the evicted VPN.
    fn evict_from_t1(&mut self) -> Option<u64> {
        if self.t1.is_empty() {
            return None;
        }

        let len = self.t1.len();
        // We may need to scan up to 2*len (one pass to clear ref bits, one to evict).
        for _ in 0..2 * len {
            if self.t1.is_empty() {
                return None;
            }
            self.t1_hand %= self.t1.len();
            let entry = &self.t1[self.t1_hand];
            let vpn = entry.vpn;

            if self.pinned.contains(&vpn) {
                self.t1_hand = (self.t1_hand + 1) % self.t1.len();
                continue;
            }

            if entry.ref_bit {
                // Second chance: clear ref bit and promote to T2.
                let entry = self.t1.remove(self.t1_hand);
                self.membership.insert(entry.vpn, ListId::T2);
                self.t2.push(ClockEntry {
                    vpn: entry.vpn,
                    ref_bit: false,
                });
                if !self.t1.is_empty() {
                    self.t1_hand %= self.t1.len();
                } else {
                    self.t1_hand = 0;
                }
                continue;
            }

            // Evict this page.
            self.t1.remove(self.t1_hand);
            self.membership.remove(&vpn);
            if !self.t1.is_empty() {
                self.t1_hand %= self.t1.len();
            } else {
                self.t1_hand = 0;
            }

            // Add to ghost B1.
            self.b1.push_back(vpn);
            self.in_b1.insert(vpn);
            while self.b1.len() > self.max_size {
                if let Some(old) = self.b1.pop_front() {
                    self.in_b1.remove(&old);
                }
            }

            return Some(vpn);
        }
        None
    }

    /// Clock-scan T2 to find a victim. Returns the evicted VPN.
    fn evict_from_t2(&mut self) -> Option<u64> {
        if self.t2.is_empty() {
            return None;
        }

        let len = self.t2.len();
        for _ in 0..2 * len {
            if self.t2.is_empty() {
                return None;
            }
            self.t2_hand %= self.t2.len();
            let entry = &self.t2[self.t2_hand];
            let vpn = entry.vpn;

            if self.pinned.contains(&vpn) {
                self.t2_hand = (self.t2_hand + 1) % self.t2.len();
                continue;
            }

            if entry.ref_bit {
                // Clear ref bit, advance hand.
                self.t2[self.t2_hand].ref_bit = false;
                self.t2_hand = (self.t2_hand + 1) % self.t2.len();
                continue;
            }

            // Evict this page.
            self.t2.remove(self.t2_hand);
            self.membership.remove(&vpn);
            if !self.t2.is_empty() {
                self.t2_hand %= self.t2.len();
            } else {
                self.t2_hand = 0;
            }

            // Add to ghost B2.
            self.b2.push_back(vpn);
            self.in_b2.insert(vpn);
            while self.b2.len() > self.max_size {
                if let Some(old) = self.b2.pop_front() {
                    self.in_b2.remove(&old);
                }
            }

            return Some(vpn);
        }
        None
    }

    /// Internal replacement when at capacity.
    fn replace(&mut self) {
        let t1_len = self.t1.len();
        if t1_len > 0 && (t1_len as f64 >= self.p || self.t2.is_empty()) {
            if self.evict_from_t1().is_none() {
                self.evict_from_t2();
            }
        } else if !self.t2.is_empty() {
            if self.evict_from_t2().is_none() {
                self.evict_from_t1();
            }
        } else {
            self.evict_from_t1();
        }
    }
}

impl EvictionPolicy for CarPolicy {
    fn record_access(&mut self, vpn: u64) {
        match self.membership.get(&vpn).copied() {
            Some(ListId::T1) => {
                // Find entry in T1 and check/set ref bit.
                if let Some(entry) = self.t1.iter_mut().find(|e| e.vpn == vpn) {
                    if entry.ref_bit {
                        // Already has ref bit set -> promote to T2.
                        if let Some(removed) = self.remove_from_t1(vpn) {
                            let _ = removed; // already removed
                            self.membership.insert(vpn, ListId::T2);
                            self.t2.push(ClockEntry {
                                vpn,
                                ref_bit: false,
                            });
                        }
                    } else {
                        entry.ref_bit = true;
                    }
                }
            }
            Some(ListId::T2) => {
                // Set ref bit in T2.
                if let Some(entry) = self.t2.iter_mut().find(|e| e.vpn == vpn) {
                    entry.ref_bit = true;
                }
            }
            None => {
                // Not tracked, no-op.
            }
        }
    }

    fn record_insert(&mut self, vpn: u64) {
        // If already tracked, treat as access.
        if self.membership.contains_key(&vpn) {
            self.record_access(vpn);
            return;
        }

        // Check ghost lists.
        if self.in_b1.contains(&vpn) {
            // B1 ghost hit: increase p.
            let delta = if self.b2.len() >= self.b1.len() && self.b1.len() > 0 {
                (self.b2.len() as f64 / self.b1.len() as f64).max(1.0)
            } else {
                1.0
            };
            self.p = (self.p + delta).min(self.max_size as f64);

            self.b1.retain(|&v| v != vpn);
            self.in_b1.remove(&vpn);

            if self.t1.len() + self.t2.len() >= self.max_size {
                self.replace();
            }
            self.t2.push(ClockEntry {
                vpn,
                ref_bit: false,
            });
            self.membership.insert(vpn, ListId::T2);
        } else if self.in_b2.contains(&vpn) {
            // B2 ghost hit: decrease p.
            let delta = if self.b1.len() >= self.b2.len() && self.b2.len() > 0 {
                (self.b1.len() as f64 / self.b2.len() as f64).max(1.0)
            } else {
                1.0
            };
            self.p = (self.p - delta).max(0.0);

            self.b2.retain(|&v| v != vpn);
            self.in_b2.remove(&vpn);

            if self.t1.len() + self.t2.len() >= self.max_size {
                self.replace();
            }
            self.t2.push(ClockEntry {
                vpn,
                ref_bit: false,
            });
            self.membership.insert(vpn, ListId::T2);
        } else {
            // New page -> insert into T1.
            if self.t1.len() + self.t2.len() >= self.max_size {
                self.replace();
            }
            if self.t1.len() + self.b1.len() >= self.max_size {
                if let Some(old) = self.b1.pop_front() {
                    self.in_b1.remove(&old);
                }
            }
            self.t1.push(ClockEntry {
                vpn,
                ref_bit: false,
            });
            self.membership.insert(vpn, ListId::T1);
        }
    }

    fn record_remove(&mut self, vpn: u64) {
        if let Some(list) = self.membership.remove(&vpn) {
            match list {
                ListId::T1 => {
                    self.remove_from_t1(vpn);
                }
                ListId::T2 => {
                    if let Some(pos) = self.t2.iter().position(|e| e.vpn == vpn) {
                        self.t2.remove(pos);
                        if !self.t2.is_empty() && self.t2_hand >= self.t2.len() {
                            self.t2_hand = 0;
                        }
                    }
                }
            }
        }
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

        let t1_len = self.t1.len();
        let vpn = if t1_len > 0 && (t1_len as f64 >= self.p || self.t2.is_empty()) {
            self.evict_from_t1().or_else(|| self.evict_from_t2())
        } else {
            self.evict_from_t2().or_else(|| self.evict_from_t1())
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
        let clock_entry_size = 16; // vpn(8) + ref_bit(1) + padding
        let hash_entry_cost = 56; // key(8) + overhead(~48)
        let t1_t2 = (self.t1.len() + self.t2.len()) * clock_entry_size;
        let ghosts = (self.b1.len() + self.b2.len()) * 8;
        let maps = self.membership.len() * (8 + 8 + 48); // key + value + overhead
        let ghost_sets = (self.in_b1.len() + self.in_b2.len()) * hash_entry_cost;
        let pins = self.pinned.len() * hash_entry_cost;
        t1_t2 + ghosts + maps + ghost_sets + pins
    }

    fn reset(&mut self) {
        self.t1.clear();
        self.t2.clear();
        self.t1_hand = 0;
        self.t2_hand = 0;
        self.b1.clear();
        self.b2.clear();
        self.p = 0.0;
        self.membership.clear();
        self.in_b1.clear();
        self.in_b2.clear();
        self.pinned.clear();
    }

    fn pin(&mut self, vpn: u64) {
        self.pinned.insert(vpn);
    }

    fn unpin(&mut self, vpn: u64) {
        self.pinned.remove(&vpn);
    }
}
