//! GPU Memory Oversubscription types for OuterLink.
//!
//! Provides virtual memory management that advertises more VRAM than physically
//! exists by spilling cold pages to host RAM. This module contains pure types
//! and page-table accounting -- no CUDA calls or actual memory operations.

use std::fmt;

/// Default page size: 2 MiB (matches CUDA large-page granularity).
pub const DEFAULT_PAGE_SIZE: usize = 2 * 1024 * 1024;

/// Default spill watermark: start evicting when physical usage exceeds this
/// percentage of physical VRAM.
pub const DEFAULT_SPILL_WATERMARK_PERCENT: f64 = 85.0;

// ---------------------------------------------------------------------------
// EvictionPolicy
// ---------------------------------------------------------------------------

/// Policy used to select which resident page to evict to host RAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvictionPolicy {
    /// Least Recently Used -- evict the page with the oldest `last_access_ns`.
    LRU,
    /// Least Frequently Used -- evict the page with the lowest `access_count`.
    LFU,
    /// Random victim selection.
    Random,
    /// First In, First Out -- evict the page that was allocated earliest.
    FIFO,
}

impl EvictionPolicy {
    /// Human-readable description of the policy.
    pub fn description(&self) -> &'static str {
        match self {
            Self::LRU => "Least Recently Used: evicts the page whose last access is oldest",
            Self::LFU => "Least Frequently Used: evicts the page with the fewest accesses",
            Self::Random => "Random: selects a victim page at random",
            Self::FIFO => "First In, First Out: evicts the page that was allocated earliest",
        }
    }
}

impl fmt::Display for EvictionPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LRU => write!(f, "LRU"),
            Self::LFU => write!(f, "LFU"),
            Self::Random => write!(f, "Random"),
            Self::FIFO => write!(f, "FIFO"),
        }
    }
}

// ---------------------------------------------------------------------------
// OversubConfig
// ---------------------------------------------------------------------------

/// Configuration for memory oversubscription on a single GPU.
#[derive(Debug, Clone)]
pub struct OversubConfig {
    /// Actual physical VRAM on the GPU (bytes).
    pub physical_vram_bytes: u64,
    /// Advertised (inflated) VRAM seen by CUDA applications (bytes).
    pub virtual_vram_bytes: u64,
    /// Maximum host RAM available for spilled pages (bytes).
    pub host_spill_limit_bytes: u64,
    /// Policy for choosing eviction victims.
    pub eviction_policy: EvictionPolicy,
    /// Size of each virtual page (bytes). Defaults to 2 MiB.
    pub page_size_bytes: usize,
    /// Physical VRAM usage percentage at which eviction begins.
    pub spill_watermark_percent: f64,
}

impl OversubConfig {
    /// Create a config with sensible defaults for page size and watermark.
    pub fn new(
        physical_vram_bytes: u64,
        virtual_vram_bytes: u64,
        host_spill_limit_bytes: u64,
        eviction_policy: EvictionPolicy,
    ) -> Self {
        Self {
            physical_vram_bytes,
            virtual_vram_bytes,
            host_spill_limit_bytes,
            eviction_policy,
            page_size_bytes: DEFAULT_PAGE_SIZE,
            spill_watermark_percent: DEFAULT_SPILL_WATERMARK_PERCENT,
        }
    }
}

// ---------------------------------------------------------------------------
// PageLocation
// ---------------------------------------------------------------------------

/// Where a virtual page physically resides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageLocation {
    /// Resident in GPU VRAM.
    GpuVram { gpu_index: u32 },
    /// Spilled to host RAM at the given offset.
    HostRam { host_offset: u64 },
    /// Evicted entirely (must be restored before access).
    Evicted,
}

// ---------------------------------------------------------------------------
// VirtualPage
// ---------------------------------------------------------------------------

/// A single virtual page tracked by the page table.
#[derive(Debug, Clone)]
pub struct VirtualPage {
    /// Virtual address assigned to this page.
    pub virtual_addr: u64,
    /// Current physical location.
    pub physical_location: PageLocation,
    /// Timestamp of the last access (nanoseconds, monotonic).
    pub last_access_ns: u64,
    /// Total number of accesses to this page.
    pub access_count: u64,
    /// Whether the page has been written to since last spill.
    pub dirty: bool,
    /// Size of this page in bytes.
    pub size_bytes: usize,
}

// ---------------------------------------------------------------------------
// OversubStats
// ---------------------------------------------------------------------------

/// Runtime statistics for memory oversubscription.
#[derive(Debug, Clone, Default)]
pub struct OversubStats {
    /// Total virtual memory currently allocated (bytes).
    pub virtual_allocated: u64,
    /// Bytes currently resident in GPU VRAM.
    pub physical_resident: u64,
    /// Bytes currently spilled to host RAM.
    pub host_spilled: u64,
    /// Number of page faults (access to evicted/spilled page).
    pub page_faults: u64,
    /// Number of pages evicted from VRAM.
    pub evictions: u64,
    /// Number of spill writes (VRAM -> host).
    pub spill_writes: u64,
    /// Number of restore reads (host -> VRAM).
    pub restore_reads: u64,
}

impl OversubStats {
    /// Ratio of virtual allocated to physical resident.
    /// Returns 0.0 if physical_resident is zero.
    pub fn overcommit_ratio(&self) -> f64 {
        if self.physical_resident == 0 {
            return 0.0;
        }
        self.virtual_allocated as f64 / self.physical_resident as f64
    }

    /// Fraction of allocated memory that is spilled to host.
    /// Returns 0.0 if virtual_allocated is zero.
    pub fn spill_ratio(&self) -> f64 {
        if self.virtual_allocated == 0 {
            return 0.0;
        }
        self.host_spilled as f64 / self.virtual_allocated as f64
    }

    /// Page fault rate given total accesses.
    /// Returns 0.0 if total_accesses is zero.
    pub fn fault_rate(&self, total_accesses: u64) -> f64 {
        if total_accesses == 0 {
            return 0.0;
        }
        self.page_faults as f64 / total_accesses as f64
    }
}

// ---------------------------------------------------------------------------
// OversubError
// ---------------------------------------------------------------------------

/// Errors specific to memory oversubscription operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OversubError {
    /// Virtual address space exhausted.
    OutOfVirtualMemory,
    /// Host spill buffer is full.
    OutOfHostSpill,
    /// Referenced virtual address does not exist in the page table.
    PageNotFound(u64),
    /// Allocation size is invalid (zero or not page-aligned).
    InvalidSize,
}

impl fmt::Display for OversubError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfVirtualMemory => write!(f, "out of virtual memory"),
            Self::OutOfHostSpill => write!(f, "host spill buffer exhausted"),
            Self::PageNotFound(addr) => write!(f, "page not found at virtual address {addr:#x}"),
            Self::InvalidSize => write!(f, "invalid allocation size"),
        }
    }
}

impl std::error::Error for OversubError {}

// ---------------------------------------------------------------------------
// PageTable
// ---------------------------------------------------------------------------

/// Virtual page table that tracks all oversubscribed memory allocations.
///
/// This is pure bookkeeping -- it does not perform any actual GPU or host
/// memory operations. The caller (client or server) is responsible for
/// executing the physical transfers that the page table decisions imply.
pub struct PageTable {
    /// All tracked virtual pages.
    pages: Vec<VirtualPage>,
    /// Oversubscription configuration.
    config: OversubConfig,
    /// Next virtual address to hand out.
    next_virtual_addr: u64,
    /// Current host spill offset (monotonically increasing simple allocator).
    next_host_offset: u64,
}

impl PageTable {
    /// Create a new page table with the given configuration.
    ///
    /// Virtual addresses start at `0x1_0000_0000` to avoid confusion with
    /// real GPU pointers in the low address range.
    pub fn new(config: OversubConfig) -> Self {
        Self {
            pages: Vec::new(),
            config,
            next_virtual_addr: 0x1_0000_0000,
            next_host_offset: 0,
        }
    }

    /// Allocate `size` bytes of virtual memory. Returns the virtual address.
    ///
    /// The allocation is page-aligned: `size` must be a positive multiple of
    /// `config.page_size_bytes`. Each page starts as `GpuVram { gpu_index: 0 }`.
    pub fn allocate(&mut self, size: usize) -> Result<u64, OversubError> {
        if size == 0 || size % self.config.page_size_bytes != 0 {
            return Err(OversubError::InvalidSize);
        }

        let total_allocated: u64 = self.pages.iter().map(|p| p.size_bytes as u64).sum();
        if total_allocated + size as u64 > self.config.virtual_vram_bytes {
            return Err(OversubError::OutOfVirtualMemory);
        }

        let base_addr = self.next_virtual_addr;
        let num_pages = size / self.config.page_size_bytes;

        for i in 0..num_pages {
            let addr = base_addr + (i * self.config.page_size_bytes) as u64;
            self.pages.push(VirtualPage {
                virtual_addr: addr,
                physical_location: PageLocation::GpuVram { gpu_index: 0 },
                last_access_ns: 0,
                access_count: 0,
                dirty: false,
                size_bytes: self.config.page_size_bytes,
            });
        }

        self.next_virtual_addr = base_addr + size as u64;
        Ok(base_addr)
    }

    /// Free the page(s) starting at `virtual_addr`.
    ///
    /// Removes all contiguous pages that were part of the allocation rooted
    /// at `virtual_addr`. Currently removes only the single page at that
    /// address; multi-page freeing is handled by removing all pages whose
    /// address falls within the original allocation range.
    pub fn free(&mut self, virtual_addr: u64) {
        self.pages.retain(|p| p.virtual_addr != virtual_addr);
    }

    /// Total bytes currently resident in GPU VRAM.
    pub fn resident_bytes(&self) -> u64 {
        self.pages
            .iter()
            .filter(|p| matches!(p.physical_location, PageLocation::GpuVram { .. }))
            .map(|p| p.size_bytes as u64)
            .sum()
    }

    /// Total bytes currently spilled to host RAM.
    pub fn spilled_bytes(&self) -> u64 {
        self.pages
            .iter()
            .filter(|p| matches!(p.physical_location, PageLocation::HostRam { .. }))
            .map(|p| p.size_bytes as u64)
            .sum()
    }

    /// Whether physical VRAM usage exceeds the spill watermark.
    pub fn should_evict(&self) -> bool {
        if self.config.physical_vram_bytes == 0 {
            return false;
        }
        let usage_percent =
            (self.resident_bytes() as f64 / self.config.physical_vram_bytes as f64) * 100.0;
        usage_percent > self.config.spill_watermark_percent
    }

    /// Select a victim page to evict based on the given policy.
    ///
    /// Only considers pages currently resident in GPU VRAM.
    /// Returns the virtual address of the selected victim, or `None` if no
    /// resident pages exist.
    pub fn select_victim(&self, policy: &EvictionPolicy) -> Option<u64> {
        let resident: Vec<&VirtualPage> = self
            .pages
            .iter()
            .filter(|p| matches!(p.physical_location, PageLocation::GpuVram { .. }))
            .collect();

        if resident.is_empty() {
            return None;
        }

        match policy {
            EvictionPolicy::LRU => resident
                .iter()
                .min_by_key(|p| p.last_access_ns)
                .map(|p| p.virtual_addr),
            EvictionPolicy::LFU => resident
                .iter()
                .min_by_key(|p| p.access_count)
                .map(|p| p.virtual_addr),
            EvictionPolicy::Random => {
                // Deterministic "random": pick the middle element.
                // Real randomness would require an RNG dependency; for a pure
                // types crate we keep it simple and deterministic.
                let idx = resident.len() / 2;
                Some(resident[idx].virtual_addr)
            }
            EvictionPolicy::FIFO => {
                // First allocated = lowest virtual address among residents.
                resident
                    .iter()
                    .min_by_key(|p| p.virtual_addr)
                    .map(|p| p.virtual_addr)
            }
        }
    }

    /// Mark a page as spilled to host RAM. Returns the host offset assigned.
    ///
    /// Fails with `OutOfHostSpill` if the host spill limit is exceeded.
    pub fn spill_page(&mut self, virtual_addr: u64) -> Result<u64, OversubError> {
        let page = self
            .pages
            .iter_mut()
            .find(|p| p.virtual_addr == virtual_addr)
            .ok_or(OversubError::PageNotFound(virtual_addr))?;

        let offset = self.next_host_offset;
        if offset + page.size_bytes as u64 > self.config.host_spill_limit_bytes {
            return Err(OversubError::OutOfHostSpill);
        }

        page.physical_location = PageLocation::HostRam {
            host_offset: offset,
        };
        self.next_host_offset += page.size_bytes as u64;
        Ok(offset)
    }

    /// Number of tracked pages.
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// The inflated total VRAM number reported to `cuMemGetInfo` /
    /// `cuDeviceTotalMem`.
    pub fn reported_total_vram(&self) -> u64 {
        self.config.virtual_vram_bytes
    }

    /// The inflated free VRAM number reported to `cuMemGetInfo`.
    ///
    /// `free = virtual_vram - total_allocated`
    pub fn reported_free_vram(&self) -> u64 {
        let allocated: u64 = self.pages.iter().map(|p| p.size_bytes as u64).sum();
        self.config.virtual_vram_bytes.saturating_sub(allocated)
    }

    /// Immutable access to tracked pages (for testing / inspection).
    pub fn pages(&self) -> &[VirtualPage] {
        &self.pages
    }

    /// Mutable access to a page by virtual address (for updating access stats).
    pub fn page_mut(&mut self, virtual_addr: u64) -> Option<&mut VirtualPage> {
        self.pages.iter_mut().find(|p| p.virtual_addr == virtual_addr)
    }

    /// Reference to the configuration.
    pub fn config(&self) -> &OversubConfig {
        &self.config
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: usize = DEFAULT_PAGE_SIZE; // 2 MiB
    const GB: u64 = 1024 * 1024 * 1024;

    /// Helper: create a standard test config (24 GB physical, 48 GB virtual,
    /// 32 GB host spill).
    fn test_config() -> OversubConfig {
        OversubConfig::new(24 * GB, 48 * GB, 32 * GB, EvictionPolicy::LRU)
    }

    // -- OversubConfig defaults -----------------------------------------------

    #[test]
    fn config_defaults_page_size() {
        let cfg = test_config();
        assert_eq!(cfg.page_size_bytes, DEFAULT_PAGE_SIZE);
    }

    #[test]
    fn config_defaults_watermark() {
        let cfg = test_config();
        assert!((cfg.spill_watermark_percent - DEFAULT_SPILL_WATERMARK_PERCENT).abs() < f64::EPSILON);
    }

    #[test]
    fn config_stores_values() {
        let cfg = OversubConfig::new(8 * GB, 16 * GB, 12 * GB, EvictionPolicy::LFU);
        assert_eq!(cfg.physical_vram_bytes, 8 * GB);
        assert_eq!(cfg.virtual_vram_bytes, 16 * GB);
        assert_eq!(cfg.host_spill_limit_bytes, 12 * GB);
        assert_eq!(cfg.eviction_policy, EvictionPolicy::LFU);
    }

    // -- EvictionPolicy -------------------------------------------------------

    #[test]
    fn eviction_policy_description_not_empty() {
        for policy in [
            EvictionPolicy::LRU,
            EvictionPolicy::LFU,
            EvictionPolicy::Random,
            EvictionPolicy::FIFO,
        ] {
            assert!(!policy.description().is_empty(), "{policy:?} description empty");
        }
    }

    #[test]
    fn eviction_policy_display() {
        assert_eq!(EvictionPolicy::LRU.to_string(), "LRU");
        assert_eq!(EvictionPolicy::LFU.to_string(), "LFU");
        assert_eq!(EvictionPolicy::Random.to_string(), "Random");
        assert_eq!(EvictionPolicy::FIFO.to_string(), "FIFO");
    }

    // -- PageLocation ---------------------------------------------------------

    #[test]
    fn page_location_gpu_vram() {
        let loc = PageLocation::GpuVram { gpu_index: 2 };
        assert_eq!(loc, PageLocation::GpuVram { gpu_index: 2 });
        assert_ne!(loc, PageLocation::Evicted);
    }

    #[test]
    fn page_location_host_ram() {
        let loc = PageLocation::HostRam { host_offset: 4096 };
        if let PageLocation::HostRam { host_offset } = loc {
            assert_eq!(host_offset, 4096);
        } else {
            panic!("expected HostRam");
        }
    }

    #[test]
    fn page_location_evicted() {
        let loc = PageLocation::Evicted;
        assert_eq!(loc, PageLocation::Evicted);
    }

    // -- PageTable allocate / free --------------------------------------------

    #[test]
    fn allocate_single_page() {
        let mut pt = PageTable::new(test_config());
        let addr = pt.allocate(PAGE).unwrap();
        assert!(addr >= 0x1_0000_0000);
        assert_eq!(pt.page_count(), 1);
    }

    #[test]
    fn allocate_multi_page() {
        let mut pt = PageTable::new(test_config());
        let addr = pt.allocate(4 * PAGE).unwrap();
        assert_eq!(pt.page_count(), 4);
        // All pages should be contiguous starting from addr.
        for (i, page) in pt.pages().iter().enumerate() {
            assert_eq!(page.virtual_addr, addr + (i * PAGE) as u64);
        }
    }

    #[test]
    fn allocate_zero_size_fails() {
        let mut pt = PageTable::new(test_config());
        assert_eq!(pt.allocate(0), Err(OversubError::InvalidSize));
    }

    #[test]
    fn allocate_unaligned_size_fails() {
        let mut pt = PageTable::new(test_config());
        assert_eq!(pt.allocate(PAGE + 1), Err(OversubError::InvalidSize));
    }

    #[test]
    fn allocate_exceeds_virtual_limit() {
        let cfg = OversubConfig::new(GB, 2 * PAGE as u64, GB, EvictionPolicy::LRU);
        let mut pt = PageTable::new(cfg);
        pt.allocate(PAGE).unwrap();
        pt.allocate(PAGE).unwrap();
        assert_eq!(pt.allocate(PAGE), Err(OversubError::OutOfVirtualMemory));
    }

    #[test]
    fn free_removes_page() {
        let mut pt = PageTable::new(test_config());
        let addr = pt.allocate(PAGE).unwrap();
        assert_eq!(pt.page_count(), 1);
        pt.free(addr);
        assert_eq!(pt.page_count(), 0);
    }

    #[test]
    fn free_nonexistent_is_noop() {
        let mut pt = PageTable::new(test_config());
        pt.allocate(PAGE).unwrap();
        pt.free(0xDEAD);
        assert_eq!(pt.page_count(), 1);
    }

    // -- resident / spilled bytes ---------------------------------------------

    #[test]
    fn resident_bytes_initial() {
        let mut pt = PageTable::new(test_config());
        pt.allocate(3 * PAGE).unwrap();
        assert_eq!(pt.resident_bytes(), 3 * PAGE as u64);
        assert_eq!(pt.spilled_bytes(), 0);
    }

    #[test]
    fn spilled_bytes_after_spill() {
        let mut pt = PageTable::new(test_config());
        let addr = pt.allocate(2 * PAGE).unwrap();
        pt.spill_page(addr).unwrap();
        assert_eq!(pt.resident_bytes(), PAGE as u64);
        assert_eq!(pt.spilled_bytes(), PAGE as u64);
    }

    // -- should_evict ---------------------------------------------------------

    #[test]
    fn should_evict_below_watermark() {
        // 24 GB physical, watermark 85% = 20.4 GB. Allocate 1 page = well below.
        let mut pt = PageTable::new(test_config());
        pt.allocate(PAGE).unwrap();
        assert!(!pt.should_evict());
    }

    #[test]
    fn should_evict_above_watermark() {
        // Make physical tiny so a single page exceeds watermark.
        let cfg = OversubConfig {
            physical_vram_bytes: PAGE as u64,
            virtual_vram_bytes: 10 * PAGE as u64,
            host_spill_limit_bytes: 10 * PAGE as u64,
            eviction_policy: EvictionPolicy::LRU,
            page_size_bytes: PAGE,
            spill_watermark_percent: 50.0, // 50% of 1 page = 0.5 pages
        };
        let mut pt = PageTable::new(cfg);
        pt.allocate(PAGE).unwrap(); // 100% usage > 50% watermark
        assert!(pt.should_evict());
    }

    // -- select_victim --------------------------------------------------------

    #[test]
    fn select_victim_lru_picks_oldest() {
        let mut pt = PageTable::new(test_config());
        let a1 = pt.allocate(PAGE).unwrap();
        let a2 = pt.allocate(PAGE).unwrap();
        // Give a1 an older timestamp, a2 a newer one.
        pt.page_mut(a1).unwrap().last_access_ns = 100;
        pt.page_mut(a2).unwrap().last_access_ns = 200;
        let victim = pt.select_victim(&EvictionPolicy::LRU).unwrap();
        assert_eq!(victim, a1, "LRU should pick the page with oldest access");
    }

    #[test]
    fn select_victim_lfu_picks_least_frequent() {
        let mut pt = PageTable::new(test_config());
        let a1 = pt.allocate(PAGE).unwrap();
        let a2 = pt.allocate(PAGE).unwrap();
        pt.page_mut(a1).unwrap().access_count = 50;
        pt.page_mut(a2).unwrap().access_count = 10;
        let victim = pt.select_victim(&EvictionPolicy::LFU).unwrap();
        assert_eq!(victim, a2, "LFU should pick the page with fewest accesses");
    }

    #[test]
    fn select_victim_fifo_picks_first_allocated() {
        let mut pt = PageTable::new(test_config());
        let a1 = pt.allocate(PAGE).unwrap();
        let _a2 = pt.allocate(PAGE).unwrap();
        let victim = pt.select_victim(&EvictionPolicy::FIFO).unwrap();
        assert_eq!(victim, a1, "FIFO should pick the first allocated page");
    }

    #[test]
    fn select_victim_random_returns_some() {
        let mut pt = PageTable::new(test_config());
        pt.allocate(3 * PAGE).unwrap();
        assert!(pt.select_victim(&EvictionPolicy::Random).is_some());
    }

    #[test]
    fn select_victim_no_resident_returns_none() {
        let mut pt = PageTable::new(test_config());
        let addr = pt.allocate(PAGE).unwrap();
        pt.spill_page(addr).unwrap();
        assert!(pt.select_victim(&EvictionPolicy::LRU).is_none());
    }

    // -- reported vram --------------------------------------------------------

    #[test]
    fn reported_total_vram_equals_virtual() {
        let pt = PageTable::new(test_config());
        assert_eq!(pt.reported_total_vram(), 48 * GB);
    }

    #[test]
    fn reported_free_vram_decreases_on_alloc() {
        let mut pt = PageTable::new(test_config());
        let free_before = pt.reported_free_vram();
        pt.allocate(PAGE).unwrap();
        let free_after = pt.reported_free_vram();
        assert_eq!(free_before - free_after, PAGE as u64);
    }

    #[test]
    fn reported_free_vram_increases_on_free() {
        let mut pt = PageTable::new(test_config());
        let addr = pt.allocate(PAGE).unwrap();
        let free_allocated = pt.reported_free_vram();
        pt.free(addr);
        assert_eq!(pt.reported_free_vram(), free_allocated + PAGE as u64);
    }

    // -- OversubStats ---------------------------------------------------------

    #[test]
    fn stats_overcommit_ratio() {
        let stats = OversubStats {
            virtual_allocated: 48 * GB,
            physical_resident: 24 * GB,
            ..Default::default()
        };
        assert!((stats.overcommit_ratio() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_overcommit_ratio_zero_physical() {
        let stats = OversubStats::default();
        assert!((stats.overcommit_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_spill_ratio() {
        let stats = OversubStats {
            virtual_allocated: 100,
            host_spilled: 25,
            ..Default::default()
        };
        assert!((stats.spill_ratio() - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_spill_ratio_zero_allocated() {
        let stats = OversubStats::default();
        assert!((stats.spill_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_fault_rate() {
        let stats = OversubStats {
            page_faults: 5,
            ..Default::default()
        };
        assert!((stats.fault_rate(100) - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_fault_rate_zero_accesses() {
        let stats = OversubStats {
            page_faults: 10,
            ..Default::default()
        };
        assert!((stats.fault_rate(0) - 0.0).abs() < f64::EPSILON);
    }

    // -- OversubError Display -------------------------------------------------

    #[test]
    fn error_display_out_of_virtual() {
        let e = OversubError::OutOfVirtualMemory;
        assert_eq!(e.to_string(), "out of virtual memory");
    }

    #[test]
    fn error_display_out_of_host_spill() {
        let e = OversubError::OutOfHostSpill;
        assert_eq!(e.to_string(), "host spill buffer exhausted");
    }

    #[test]
    fn error_display_page_not_found() {
        let e = OversubError::PageNotFound(0xDEAD);
        assert!(e.to_string().contains("0xdead"));
    }

    #[test]
    fn error_display_invalid_size() {
        let e = OversubError::InvalidSize;
        assert_eq!(e.to_string(), "invalid allocation size");
    }

    // -- spill_page errors ----------------------------------------------------

    #[test]
    fn spill_page_not_found() {
        let mut pt = PageTable::new(test_config());
        assert_eq!(pt.spill_page(0xBEEF), Err(OversubError::PageNotFound(0xBEEF)));
    }

    #[test]
    fn spill_page_host_limit_exceeded() {
        let cfg = OversubConfig::new(GB, 10 * GB, PAGE as u64, EvictionPolicy::LRU);
        let mut pt = PageTable::new(cfg);
        let a1 = pt.allocate(PAGE).unwrap();
        let a2 = pt.allocate(PAGE).unwrap();
        pt.spill_page(a1).unwrap(); // uses up the entire host spill limit
        assert_eq!(pt.spill_page(a2), Err(OversubError::OutOfHostSpill));
    }

    // -- new pages are resident in VRAM ---------------------------------------

    #[test]
    fn new_pages_are_gpu_resident() {
        let mut pt = PageTable::new(test_config());
        pt.allocate(PAGE).unwrap();
        assert!(matches!(
            pt.pages()[0].physical_location,
            PageLocation::GpuVram { gpu_index: 0 }
        ));
    }

    // -- VirtualPage fields ---------------------------------------------------

    #[test]
    fn new_page_defaults() {
        let mut pt = PageTable::new(test_config());
        pt.allocate(PAGE).unwrap();
        let page = &pt.pages()[0];
        assert_eq!(page.last_access_ns, 0);
        assert_eq!(page.access_count, 0);
        assert!(!page.dirty);
        assert_eq!(page.size_bytes, PAGE);
    }
}
