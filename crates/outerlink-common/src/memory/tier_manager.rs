//! DefaultTierManager implementation.
//!
//! Coordinates page placement across all 6 memory tiers, managing allocation,
//! deallocation, promotion/demotion decisions, pinning, and integration with
//! eviction policies and optional compression/dedup hooks.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, RwLock};

use dashmap::DashMap;

use super::config::{default_tier_configs, EvictionPolicyType, TierConfig};
use super::eviction::{ArcPolicy, CarPolicy, ClockPolicy};
use super::tier_status::{AllocationError, AllocationHints, TierAllocation, TierStatus};
use super::traits::{AccessType, CompressionHook, DedupHook, EvictionPolicy, PinError, TierManager};
use super::types::{PressureLevel, TierId, LOCAL_VRAM};

/// Default page size: 64 KB (matches migration engine).
const PAGE_SIZE: usize = 65536;

/// Default max pages per eviction policy (reasonably large).
const DEFAULT_POLICY_MAX_PAGES: usize = 1_000_000;

/// Metadata for a tracked allocation.
#[derive(Debug)]
struct AllocationMeta {
    #[allow(dead_code)]
    alloc_id: u32,
    vpn_start: u64,
    page_count: usize,
    initial_tier: TierId,
    #[allow(dead_code)]
    hints: AllocationHints,
}

/// Coordinates page placement across all 6 tiers.
///
/// Manages allocation, deallocation, pin/unpin, eviction policy integration,
/// promotion/demotion decisions, and optional hooks for compression and dedup.
pub struct DefaultTierManager {
    /// Tier configurations (one per tier).
    configs: Vec<TierConfig>,
    /// Per-tier page tracking: tier_id -> set of VPNs in that tier.
    tier_pages: Vec<DashMap<u64, ()>>,
    /// Per-tier pinned page counts.
    pinned_counts: Vec<AtomicU64>,
    /// Allocation tracking: alloc_id -> allocation metadata.
    allocations: DashMap<u32, AllocationMeta>,
    /// Next allocation ID counter (reserved for future auto-ID generation).
    #[allow(dead_code)]
    next_alloc_id: AtomicU32,
    /// Per-tier VPN counter for assigning VPN ranges.
    next_vpn: AtomicU64,
    /// Per-tier eviction policies (wrapped in Mutex for &self compatibility).
    eviction_policies: Vec<Mutex<Box<dyn EvictionPolicy>>>,
    /// Bandwidth budgets per tier-pair: key = (from << 8 | to) as u16.
    bandwidth_budgets: DashMap<u16, u64>,
    /// Per-tier pinned VPN sets (for pin/unpin tracking).
    pinned_vpns: Vec<DashMap<u64, ()>>,
    /// Optional compression hook.
    compression_hook: RwLock<Option<Box<dyn CompressionHook>>>,
    /// Optional deduplication hook.
    dedup_hook: RwLock<Option<Box<dyn DedupHook>>>,
}

impl DefaultTierManager {
    /// Create a new tier manager with the given tier configurations.
    ///
    /// Creates an eviction policy for each tier based on its configured
    /// `eviction_policy` type (ARC for VRAM, CAR for DRAM, CLOCK for NVMe).
    pub fn new(configs: Vec<TierConfig>) -> Self {
        let tier_count = configs.len();

        let tier_pages: Vec<DashMap<u64, ()>> =
            (0..tier_count).map(|_| DashMap::new()).collect();
        let pinned_counts: Vec<AtomicU64> =
            (0..tier_count).map(|_| AtomicU64::new(0)).collect();
        let pinned_vpns: Vec<DashMap<u64, ()>> =
            (0..tier_count).map(|_| DashMap::new()).collect();

        let eviction_policies: Vec<Mutex<Box<dyn EvictionPolicy>>> = configs
            .iter()
            .map(|cfg| {
                let policy: Box<dyn EvictionPolicy> = match cfg.eviction_policy {
                    EvictionPolicyType::Arc => {
                        Box::new(ArcPolicy::new(DEFAULT_POLICY_MAX_PAGES, cfg.tier_id))
                    }
                    EvictionPolicyType::Car => {
                        Box::new(CarPolicy::new(DEFAULT_POLICY_MAX_PAGES, cfg.tier_id))
                    }
                    EvictionPolicyType::Clock => {
                        Box::new(ClockPolicy::new(DEFAULT_POLICY_MAX_PAGES, cfg.tier_id))
                    }
                };
                Mutex::new(policy)
            })
            .collect();

        Self {
            configs,
            tier_pages,
            pinned_counts,
            allocations: DashMap::new(),
            next_alloc_id: AtomicU32::new(1),
            next_vpn: AtomicU64::new(0x1000), // Start VPNs at a reasonable offset
            eviction_policies,
            bandwidth_budgets: DashMap::new(),
            pinned_vpns,
            compression_hook: RwLock::new(None),
            dedup_hook: RwLock::new(None),
        }
    }

    /// Create a tier manager with the default 6-tier configuration.
    pub fn with_defaults() -> Self {
        Self::new(default_tier_configs().to_vec())
    }

    /// Set a bandwidth budget for migrations between two tiers.
    pub fn set_bandwidth_budget(&self, from: TierId, to: TierId, bytes_per_sec: u64) {
        let key = tier_pair_key(from, to);
        self.bandwidth_budgets.insert(key, bytes_per_sec);
    }

    /// Compute the pressure level for a tier based on utilization.
    fn pressure_level(&self, tier_id: TierId) -> PressureLevel {
        let idx = tier_id as usize;
        if idx >= self.configs.len() {
            return PressureLevel::Low;
        }
        let capacity = self.configs[idx].capacity_bytes;
        if capacity == 0 {
            return PressureLevel::Low;
        }
        let used = self.tier_pages[idx].len() as u64 * PAGE_SIZE as u64;
        let utilization = used as f64 / capacity as f64;
        if utilization >= 0.95 {
            PressureLevel::Critical
        } else if utilization >= 0.85 {
            PressureLevel::High
        } else if utilization >= 0.70 {
            PressureLevel::Medium
        } else {
            PressureLevel::Low
        }
    }

    /// Find the best tier for an allocation given the hints.
    fn select_tier(&self, hints: &AllocationHints, size_bytes: usize) -> Result<TierId, AllocationError> {
        // Pinned allocations must go to LOCAL_VRAM
        if hints.pinned {
            return self.check_tier_capacity(LOCAL_VRAM, size_bytes);
        }

        // If preferred tier is specified, try it first
        if let Some(preferred) = hints.preferred_tier {
            if let Ok(tier) = self.check_tier_capacity(preferred, size_bytes) {
                return Ok(tier);
            }
        }

        // Default: try tiers in order of speed (LOCAL_VRAM first)
        for idx in 0..self.configs.len() {
            let tier_id = self.configs[idx].tier_id;
            if self.check_tier_capacity(tier_id, size_bytes).is_ok() {
                return Ok(tier_id);
            }
        }

        Err(AllocationError::AllTiersExhausted)
    }

    /// Check if a tier has enough capacity for the given size.
    fn check_tier_capacity(&self, tier_id: TierId, size_bytes: usize) -> Result<TierId, AllocationError> {
        let idx = tier_id as usize;
        if idx >= self.configs.len() {
            return Err(AllocationError::InvalidTier(tier_id));
        }
        let capacity = self.configs[idx].capacity_bytes;
        let used = self.tier_pages[idx].len() as u64 * PAGE_SIZE as u64;
        if used + size_bytes as u64 > capacity {
            Err(AllocationError::OutOfMemory(tier_id))
        } else {
            Ok(tier_id)
        }
    }

    /// Find which tier a VPN belongs to.
    fn find_tier_for_vpn(&self, vpn: u64) -> Option<TierId> {
        for (idx, pages) in self.tier_pages.iter().enumerate() {
            if pages.contains_key(&vpn) {
                return Some(idx as TierId);
            }
        }
        None
    }

    /// Get the next slower tier for demotion.
    fn next_slower_tier(&self, tier_id: TierId) -> Option<TierId> {
        let next = tier_id + 1;
        if (next as usize) < self.configs.len() {
            Some(next)
        } else {
            None
        }
    }

    /// Get the next faster tier for promotion.
    fn next_faster_tier(&self, tier_id: TierId) -> Option<TierId> {
        if tier_id > 0 {
            Some(tier_id - 1)
        } else {
            None
        }
    }
}

impl TierManager for DefaultTierManager {
    fn allocate(
        &self,
        alloc_id: u32,
        size_bytes: usize,
        hints: AllocationHints,
    ) -> Result<TierAllocation, AllocationError> {
        let tier_id = self.select_tier(&hints, size_bytes)?;
        let page_count = (size_bytes + PAGE_SIZE - 1) / PAGE_SIZE;

        // Assign VPN range
        let vpn_start = self.next_vpn.fetch_add(page_count as u64, Ordering::Relaxed);

        // Track pages in the tier and record inserts in eviction policy
        let idx = tier_id as usize;
        {
            let mut policy = self.eviction_policies[idx].lock().unwrap();
            for i in 0..page_count {
                let vpn = vpn_start + i as u64;
                self.tier_pages[idx].insert(vpn, ());
                policy.record_insert(vpn);
            }
        }

        // If pinned, track pinned pages
        if hints.pinned {
            for i in 0..page_count {
                let vpn = vpn_start + i as u64;
                self.pinned_vpns[idx].insert(vpn, ());
            }
            self.pinned_counts[idx].fetch_add(page_count as u64, Ordering::Relaxed);
        }

        // Store allocation metadata
        self.allocations.insert(
            alloc_id,
            AllocationMeta {
                alloc_id,
                vpn_start,
                page_count,
                initial_tier: tier_id,
                hints,
            },
        );

        Ok(TierAllocation {
            vpn_start,
            page_count,
            initial_tier: tier_id,
        })
    }

    fn deallocate(&self, alloc_id: u32) -> Result<(), AllocationError> {
        let (_, meta) = self
            .allocations
            .remove(&alloc_id)
            .ok_or(AllocationError::InvalidTier(0))?;

        let idx = meta.initial_tier as usize;

        let mut policy = self.eviction_policies[idx].lock().unwrap();
        for i in 0..meta.page_count {
            let vpn = meta.vpn_start + i as u64;
            self.tier_pages[idx].remove(&vpn);
            policy.record_remove(vpn);

            // Clean up pinned state
            if self.pinned_vpns[idx].remove(&vpn).is_some() {
                self.pinned_counts[idx].fetch_sub(1, Ordering::Relaxed);
            }
        }

        Ok(())
    }

    fn request_promotion(&self, vpn: u64) -> Option<TierId> {
        let current_tier = self.find_tier_for_vpn(vpn)?;
        self.next_faster_tier(current_tier)
    }

    fn request_demotion(&self, tier: TierId) -> Option<(u64, TierId)> {
        let idx = tier as usize;
        if idx >= self.eviction_policies.len() {
            return None;
        }

        let dest = self.next_slower_tier(tier)?;

        let policy = self.eviction_policies[idx].lock().unwrap();
        let victims = policy.select_victim(1);
        if let Some(victim) = victims.first() {
            // Don't demote pinned pages
            if self.pinned_vpns[idx].contains_key(&victim.vpn) {
                return None;
            }
            Some((victim.vpn, dest))
        } else {
            None
        }
    }

    fn pin_page(&self, vpn: u64) -> Result<(), PinError> {
        let tier_id = self.find_tier_for_vpn(vpn).ok_or(PinError::NotFound(vpn))?;
        let idx = tier_id as usize;

        // Check if already pinned
        if self.pinned_vpns[idx].contains_key(&vpn) {
            return Err(PinError::InvalidState(vpn));
        }

        self.pinned_vpns[idx].insert(vpn, ());
        self.pinned_counts[idx].fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn unpin_page(&self, vpn: u64) -> Result<(), PinError> {
        let tier_id = self.find_tier_for_vpn(vpn).ok_or(PinError::NotFound(vpn))?;
        let idx = tier_id as usize;

        // Check if actually pinned
        if self.pinned_vpns[idx].remove(&vpn).is_none() {
            return Err(PinError::InvalidState(vpn));
        }

        self.pinned_counts[idx].fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }

    fn tier_status(&self, tier_id: TierId) -> Option<TierStatus> {
        let idx = tier_id as usize;
        if idx >= self.configs.len() {
            return None;
        }

        let config = &self.configs[idx];
        let page_count = self.tier_pages[idx].len() as u64;
        let used_bytes = page_count * PAGE_SIZE as u64;
        let pinned_pages = self.pinned_counts[idx].load(Ordering::Relaxed);

        let utilization_percent = if config.capacity_bytes > 0 {
            (used_bytes as f32 / config.capacity_bytes as f32) * 100.0
        } else {
            0.0
        };

        Some(TierStatus {
            tier_id,
            capacity_bytes: config.capacity_bytes,
            used_bytes,
            page_count,
            pressure_level: self.pressure_level(tier_id),
            active_migrations: 0,
            pinned_pages,
            migrating_pages: 0,
            compressed_pages: 0,
            dedup_saved_pages: 0,
            utilization_percent,
        })
    }

    fn migration_budget(&self, from: TierId, to: TierId) -> u64 {
        let key = tier_pair_key(from, to);
        self.bandwidth_budgets
            .get(&key)
            .map(|v| *v)
            .unwrap_or(0)
    }

    fn notify_access(&self, vpn: u64, _access_type: AccessType) {
        if let Some(tier_id) = self.find_tier_for_vpn(vpn) {
            let idx = tier_id as usize;
            let mut policy = self.eviction_policies[idx].lock().unwrap();
            policy.record_access(vpn);
        }
    }

    fn set_compression_hook(&mut self, hook: Box<dyn CompressionHook>) {
        *self.compression_hook.write().unwrap() = Some(hook);
    }

    fn set_dedup_hook(&mut self, hook: Box<dyn DedupHook>) {
        *self.dedup_hook.write().unwrap() = Some(hook);
    }
}

/// Compute a key for the bandwidth budget map from a tier pair.
fn tier_pair_key(from: TierId, to: TierId) -> u16 {
    ((from as u16) << 8) | (to as u16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::{LOCAL_DRAM, TIER_COUNT};

    // A trivial compression hook for testing.
    struct TestCompressionHook;
    impl CompressionHook for TestCompressionHook {
        fn try_compress(&self, _data: &[u8]) -> Option<(Vec<u8>, f32)> {
            None
        }
        fn decompress(&self, _compressed: &[u8], _expected_size: usize) -> Result<Vec<u8>, String> {
            Err("not implemented".into())
        }
    }

    // A trivial dedup hook for testing.
    struct TestDedupHook;
    impl DedupHook for TestDedupHook {
        fn check_dedup(&self, _vpn: u64, _data: &[u8]) -> Option<u64> {
            None
        }
        fn notify_free(&self, _vpn: u64) {}
    }

    #[test]
    fn tier_manager_allocate_default() {
        let mgr = DefaultTierManager::with_defaults();
        let hints = AllocationHints::default();
        let result = mgr.allocate(1, PAGE_SIZE, hints);
        assert!(result.is_ok());
        let alloc = result.unwrap();
        assert_eq!(alloc.initial_tier, LOCAL_VRAM);
        assert_eq!(alloc.page_count, 1);
    }

    #[test]
    fn tier_manager_allocate_with_preferred_tier() {
        let mgr = DefaultTierManager::with_defaults();
        let hints = AllocationHints {
            preferred_tier: Some(LOCAL_DRAM),
            ..Default::default()
        };
        let result = mgr.allocate(1, PAGE_SIZE, hints);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().initial_tier, LOCAL_DRAM);
    }

    #[test]
    fn tier_manager_allocate_pinned() {
        let mgr = DefaultTierManager::with_defaults();
        let hints = AllocationHints {
            pinned: true,
            ..Default::default()
        };
        let result = mgr.allocate(1, PAGE_SIZE, hints);
        assert!(result.is_ok());
        let alloc = result.unwrap();
        assert_eq!(alloc.initial_tier, LOCAL_VRAM);

        // Verify pinned count increased
        let status = mgr.tier_status(LOCAL_VRAM).unwrap();
        assert_eq!(status.pinned_pages, 1);
    }

    #[test]
    fn tier_manager_deallocate() {
        let mgr = DefaultTierManager::with_defaults();
        let hints = AllocationHints::default();
        let alloc = mgr.allocate(1, PAGE_SIZE * 3, hints).unwrap();
        assert_eq!(alloc.page_count, 3);

        // Tier should have pages
        let status_before = mgr.tier_status(LOCAL_VRAM).unwrap();
        assert_eq!(status_before.page_count, 3);

        // Deallocate
        mgr.deallocate(1).unwrap();

        // Tier should be empty
        let status_after = mgr.tier_status(LOCAL_VRAM).unwrap();
        assert_eq!(status_after.page_count, 0);
    }

    #[test]
    fn tier_manager_pin_unpin() {
        let mgr = DefaultTierManager::with_defaults();
        let hints = AllocationHints::default();
        let alloc = mgr.allocate(1, PAGE_SIZE, hints).unwrap();
        let vpn = alloc.vpn_start;

        // Pin
        mgr.pin_page(vpn).unwrap();
        let status = mgr.tier_status(LOCAL_VRAM).unwrap();
        assert_eq!(status.pinned_pages, 1);

        // Pin again should fail (already pinned)
        assert!(mgr.pin_page(vpn).is_err());

        // Unpin
        mgr.unpin_page(vpn).unwrap();
        let status = mgr.tier_status(LOCAL_VRAM).unwrap();
        assert_eq!(status.pinned_pages, 0);

        // Unpin again should fail (not pinned)
        assert!(mgr.unpin_page(vpn).is_err());
    }

    #[test]
    fn tier_manager_tier_status() {
        let mgr = DefaultTierManager::with_defaults();
        let hints = AllocationHints::default();

        // Allocate 5 pages
        mgr.allocate(1, PAGE_SIZE * 5, hints).unwrap();

        let status = mgr.tier_status(LOCAL_VRAM).unwrap();
        assert_eq!(status.tier_id, LOCAL_VRAM);
        assert_eq!(status.page_count, 5);
        assert_eq!(status.used_bytes, 5 * PAGE_SIZE as u64);
        assert_eq!(status.pressure_level, PressureLevel::Low);
        assert!(status.utilization_percent > 0.0);

        // Invalid tier should return None
        assert!(mgr.tier_status(99).is_none());
    }

    #[test]
    fn tier_manager_migration_budget() {
        let mgr = DefaultTierManager::with_defaults();

        // No budget set -> returns 0
        assert_eq!(mgr.migration_budget(LOCAL_VRAM, LOCAL_DRAM), 0);

        // Set a budget
        mgr.set_bandwidth_budget(LOCAL_VRAM, LOCAL_DRAM, 1_000_000);
        assert_eq!(mgr.migration_budget(LOCAL_VRAM, LOCAL_DRAM), 1_000_000);

        // Different pair is still 0
        assert_eq!(mgr.migration_budget(LOCAL_DRAM, LOCAL_VRAM), 0);
    }

    #[test]
    fn tier_manager_notify_access() {
        let mgr = DefaultTierManager::with_defaults();
        let hints = AllocationHints::default();
        let alloc = mgr.allocate(1, PAGE_SIZE, hints).unwrap();
        let vpn = alloc.vpn_start;

        // Notify access should not panic, and should update eviction policy
        mgr.notify_access(vpn, AccessType::Read);
        mgr.notify_access(vpn, AccessType::Write);

        // The eviction policy should still track the page
        let policy = mgr.eviction_policies[LOCAL_VRAM as usize].lock().unwrap();
        assert!(policy.tracked_count() >= 1);
    }

    #[test]
    fn tier_manager_request_demotion() {
        let mgr = DefaultTierManager::with_defaults();
        let hints = AllocationHints::default();

        // Allocate several pages
        for i in 1..=10u32 {
            mgr.allocate(i, PAGE_SIZE, hints.clone()).unwrap();
        }

        // Request demotion from LOCAL_VRAM
        let result = mgr.request_demotion(LOCAL_VRAM);
        // Should get a victim and target tier (REMOTE_VRAM = 1)
        assert!(result.is_some());
        let (vpn, dest) = result.unwrap();
        assert!(vpn > 0);
        assert_eq!(dest, 1); // REMOTE_VRAM
    }

    #[test]
    fn tier_manager_compression_hook() {
        let mut mgr = DefaultTierManager::with_defaults();

        // Initially no hook
        {
            let hook = mgr.compression_hook.read().unwrap();
            assert!(hook.is_none());
        }

        // Set hook
        mgr.set_compression_hook(Box::new(TestCompressionHook));

        // Verify it's stored
        let hook = mgr.compression_hook.read().unwrap();
        assert!(hook.is_some());
    }

    #[test]
    fn tier_manager_dedup_hook() {
        let mut mgr = DefaultTierManager::with_defaults();

        // Initially no hook
        {
            let hook = mgr.dedup_hook.read().unwrap();
            assert!(hook.is_none());
        }

        // Set hook
        mgr.set_dedup_hook(Box::new(TestDedupHook));

        // Verify it's stored
        let hook = mgr.dedup_hook.read().unwrap();
        assert!(hook.is_some());
    }

    #[test]
    fn tier_manager_with_defaults() {
        let mgr = DefaultTierManager::with_defaults();

        // Should have 6 tiers
        assert_eq!(mgr.configs.len(), TIER_COUNT);
        assert_eq!(mgr.eviction_policies.len(), TIER_COUNT);
        assert_eq!(mgr.tier_pages.len(), TIER_COUNT);

        // All tiers should report status
        for i in 0..TIER_COUNT {
            let status = mgr.tier_status(i as TierId);
            assert!(status.is_some());
        }
    }
}
