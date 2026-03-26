//! Simple CLOCK eviction policy.
//!
//! Designed for NVMe tiers (4, 5) where the page count is large and the
//! overhead of ghost lists is unjustified. Uses a single circular buffer
//! with reference bits.
//!
//! On access: set reference bit.
//! On victim selection: advance clock hand. If ref bit is 0, evict. If 1,
//! clear and advance (second chance).

use std::collections::{HashMap, HashSet};

use crate::memory::tier_status::EvictionCandidate;
use crate::memory::traits::EvictionPolicy;
use crate::memory::types::TierId;

/// Simple CLOCK eviction policy.
pub struct ClockPolicy {
    max_size: usize,
    /// Circular buffer of (vpn, reference_bit). None = empty slot.
    buffer: Vec<Option<(u64, bool)>>,
    /// Clock hand position.
    hand: usize,
    /// VPN -> position in buffer for O(1) lookup.
    index: HashMap<u64, usize>,
    /// Number of occupied slots.
    count: usize,
    /// Pinned pages (cannot be evicted).
    pinned: HashSet<u64>,
    source_tier: TierId,
}

impl ClockPolicy {
    /// Create a new CLOCK policy with the given capacity and source tier.
    pub fn new(max_size: usize, source_tier: TierId) -> Self {
        Self {
            max_size,
            buffer: vec![None; max_size],
            hand: 0,
            index: HashMap::new(),
            count: 0,
            pinned: HashSet::new(),
            source_tier,
        }
    }

    /// Pin a page so it cannot be evicted.
    pub fn pin(&mut self, vpn: u64) {
        self.pinned.insert(vpn);
    }

    /// Unpin a previously pinned page.
    pub fn unpin(&mut self, vpn: u64) {
        self.pinned.remove(&vpn);
    }

    /// Find the next empty slot starting from the hand position.
    fn find_empty_slot(&self) -> Option<usize> {
        for i in 0..self.buffer.len() {
            let pos = (self.hand + i) % self.buffer.len();
            if self.buffer[pos].is_none() {
                return Some(pos);
            }
        }
        None
    }

    /// Internal eviction: advance the clock, clear ref bits (second chance),
    /// and remove the first unpinned page with ref_bit=false.
    /// Returns the evicted VPN, or None if all pages are pinned.
    fn evict_one(&mut self) -> Option<u64> {
        if self.count == 0 {
            return None;
        }

        let unpinned_count = self.count - self.pinned.len().min(self.count);
        if unpinned_count == 0 {
            return None;
        }

        let buf_len = self.buffer.len();
        for _ in 0..2 * buf_len {
            self.hand %= buf_len;
            if let Some((vpn, ref_bit)) = self.buffer[self.hand] {
                if self.pinned.contains(&vpn) {
                    self.hand = (self.hand + 1) % buf_len;
                    continue;
                }

                if ref_bit {
                    // Second chance: clear ref bit, advance.
                    self.buffer[self.hand] = Some((vpn, false));
                    self.hand = (self.hand + 1) % buf_len;
                    continue;
                }

                // Evict this page.
                self.buffer[self.hand] = None;
                self.index.remove(&vpn);
                self.count -= 1;
                self.hand = (self.hand + 1) % buf_len;

                return Some(vpn);
            } else {
                self.hand = (self.hand + 1) % buf_len;
            }
        }
        None
    }
}

impl EvictionPolicy for ClockPolicy {
    fn record_access(&mut self, vpn: u64) {
        if let Some(&pos) = self.index.get(&vpn) {
            if let Some((_, ref mut ref_bit)) = self.buffer[pos] {
                *ref_bit = true;
            }
        }
    }

    fn record_insert(&mut self, vpn: u64) {
        // If already tracked, treat as access.
        if self.index.contains_key(&vpn) {
            self.record_access(vpn);
            return;
        }

        // Find an empty slot, or evict to make room.
        let slot = if self.count < self.max_size {
            self.find_empty_slot().expect("count < max_size but no empty slot")
        } else {
            // Need to evict first.
            if self.evict_one().is_none() {
                // All pages are pinned, can't make room.
                return;
            }
            // evict_one freed a slot.
            self.find_empty_slot().expect("just evicted but no empty slot")
        };

        self.buffer[slot] = Some((vpn, false));
        self.index.insert(vpn, slot);
        self.count += 1;
    }

    fn record_remove(&mut self, vpn: u64) {
        if let Some(pos) = self.index.remove(&vpn) {
            self.buffer[pos] = None;
            self.count -= 1;
        }
        self.pinned.remove(&vpn);
    }

    fn select_victim(&self, count: usize) -> Vec<EvictionCandidate> {
        if self.count == 0 || count == 0 {
            return Vec::new();
        }

        let unpinned_count = self.count - self.pinned.len().min(self.count);
        if unpinned_count == 0 {
            return Vec::new();
        }

        let mut candidates = Vec::with_capacity(count.min(unpinned_count));
        let buf_len = self.buffer.len();
        // Simulate a clock scan without mutating state.
        let mut hand = self.hand;
        // Track which positions we've given a "virtual second chance".
        let mut cleared = HashSet::new();
        let mut seen_vpns = HashSet::new();

        for _ in 0..2 * buf_len {
            if candidates.len() >= count {
                break;
            }
            hand %= buf_len;
            if let Some((vpn, ref_bit)) = self.buffer[hand] {
                if self.pinned.contains(&vpn) || seen_vpns.contains(&vpn) {
                    hand = (hand + 1) % buf_len;
                    continue;
                }

                let effective_ref = if cleared.contains(&hand) {
                    false
                } else {
                    ref_bit
                };

                if effective_ref {
                    // Virtual second chance.
                    cleared.insert(hand);
                    hand = (hand + 1) % buf_len;
                    continue;
                }

                // This page would be evicted.
                seen_vpns.insert(vpn);
                // Score: lower is better candidate. Pages with no ref bit get
                // score 0.0 (best candidate).
                candidates.push(EvictionCandidate {
                    vpn,
                    tier_id: self.source_tier,
                    score: 0.0,
                    dirty: false, // CLOCK does not track dirty state
                });
                hand = (hand + 1) % buf_len;
            } else {
                hand = (hand + 1) % buf_len;
            }
        }
        candidates
    }

    fn is_ghost_hit(&self, _vpn: u64) -> bool {
        // CLOCK has no ghost lists.
        false
    }

    fn tracked_count(&self) -> usize {
        self.count
    }

    fn memory_overhead(&self) -> usize {
        // buffer: max_size * size_of Option<(u64, bool)> ~ 16 bytes each
        // index: HashMap entries ~ 56 bytes each
        // pinned: HashSet entries ~ 56 bytes each
        let buffer_cost = self.max_size * 16;
        let index_cost = self.index.len() * 56;
        let pinned_cost = self.pinned.len() * 56;
        buffer_cost + index_cost + pinned_cost
    }

    fn reset(&mut self) {
        self.buffer.fill(None);
        self.hand = 0;
        self.index.clear();
        self.count = 0;
        self.pinned.clear();
    }
}
