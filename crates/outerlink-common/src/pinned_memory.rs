//! Pinned memory pool types for OuterLink.
//!
//! Provides the accounting layer for slab-based pinned (page-locked) host memory
//! management. Actual memory allocation (`cudaHostAlloc`, `mmap`) happens in the
//! server crate; this module defines the bookkeeping types, slab classes, pool
//! configuration, allocation tracking, and pressure/spill policies.

// ---------------------------------------------------------------------------
// SlabClass
// ---------------------------------------------------------------------------

/// Slab size classes for the pinned memory pool.
///
/// Each class represents a fixed-size buffer. Requests are rounded up to the
/// smallest class that fits, avoiding fragmentation in the pinned pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SlabClass {
    /// 4 KiB -- small control messages, tiny tensors.
    Tiny,
    /// 64 KiB -- typical cuMemcpy for metadata.
    Small,
    /// 1 MiB -- medium tensor transfers.
    Medium,
    /// 16 MiB -- large tensor chunks.
    Large,
    /// 256 MiB -- full model layers, huge buffers.
    Huge,
}

impl SlabClass {
    /// Byte size for this slab class.
    pub const fn size_bytes(&self) -> usize {
        match self {
            Self::Tiny => 4 * 1024,                  // 4 KiB
            Self::Small => 64 * 1024,                 // 64 KiB
            Self::Medium => 1 * 1024 * 1024,          // 1 MiB
            Self::Large => 16 * 1024 * 1024,          // 16 MiB
            Self::Huge => 256 * 1024 * 1024,          // 256 MiB
        }
    }

    /// Pick the smallest slab class whose size is >= `bytes`.
    ///
    /// Returns `None` if `bytes` exceeds the largest class (Huge, 256 MiB).
    pub fn from_request_size(bytes: usize) -> Option<SlabClass> {
        if bytes <= Self::Tiny.size_bytes() {
            Some(Self::Tiny)
        } else if bytes <= Self::Small.size_bytes() {
            Some(Self::Small)
        } else if bytes <= Self::Medium.size_bytes() {
            Some(Self::Medium)
        } else if bytes <= Self::Large.size_bytes() {
            Some(Self::Large)
        } else if bytes <= Self::Huge.size_bytes() {
            Some(Self::Huge)
        } else {
            None
        }
    }

    /// All slab classes in ascending size order.
    pub const ALL: [SlabClass; 5] = [
        Self::Tiny,
        Self::Small,
        Self::Medium,
        Self::Large,
        Self::Huge,
    ];
}

// ---------------------------------------------------------------------------
// PoolConfig
// ---------------------------------------------------------------------------

/// Configuration for the pinned memory pool.
///
/// Defines how many buffers of each slab class the pool pre-allocates and the
/// overall memory budget. The pool can optionally grow on demand until the
/// budget is exhausted.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Number of buffers to pre-allocate per slab class.
    /// Index order matches [`SlabClass::ALL`]: Tiny, Small, Medium, Large, Huge.
    pub per_class_capacity: [u32; 5],

    /// Hard upper bound on total pool memory (bytes).
    pub max_total_bytes: u64,

    /// When pool utilization exceeds this percentage, the spill policy activates.
    /// Range 0..=100.
    pub spill_threshold_percent: u8,

    /// What to do when the pool is full and a new allocation is requested.
    pub spill_policy: SpillPolicy,

    /// Whether the pool may allocate additional buffers beyond the initial
    /// per-class capacity, up to `max_total_bytes`.
    pub grow_on_demand: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            // 64 Tiny, 32 Small, 16 Medium, 8 Large (128MB for 100GbE per R49), 1 Huge
            per_class_capacity: [64, 32, 16, 8, 1],
            max_total_bytes: 512 * 1024 * 1024, // 512 MiB
            spill_threshold_percent: 95,
            spill_policy: SpillPolicy::SpillToUnpinned,
            grow_on_demand: true,
        }
    }
}

// ---------------------------------------------------------------------------
// PoolStats
// ---------------------------------------------------------------------------

/// Runtime statistics for a pinned memory pool.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Currently allocated (in-use) bytes across all slab classes.
    pub allocated_bytes: u64,
    /// Total pool capacity in bytes (pre-allocated + grown).
    pub pool_capacity_bytes: u64,
    /// Number of buffers currently checked out.
    pub active_buffers: u32,
    /// Total number of times the pool spilled to unpinned memory.
    pub spill_count: u64,
    /// Highest allocated_bytes value observed.
    pub high_watermark_bytes: u64,
}

impl PoolStats {
    /// Pool utilization as a fraction in `[0.0, 1.0]`.
    ///
    /// Returns 0.0 when pool_capacity_bytes is zero to avoid division by zero.
    pub fn utilization(&self) -> f64 {
        if self.pool_capacity_bytes == 0 {
            return 0.0;
        }
        self.allocated_bytes as f64 / self.pool_capacity_bytes as f64
    }

    /// Returns `true` when utilization exceeds `threshold` (0.0 .. 1.0).
    pub fn is_under_pressure(&self, threshold: f64) -> bool {
        self.utilization() > threshold
    }
}

// ---------------------------------------------------------------------------
// AllocationResult
// ---------------------------------------------------------------------------

/// Outcome of a buffer allocation request.
#[derive(Debug, Clone)]
pub enum AllocationResult {
    /// Successfully obtained a pinned buffer from the pool.
    Pinned {
        /// Slab class of the allocated buffer.
        class: SlabClass,
        /// Index within the slab class.
        index: u32,
    },
    /// Pool was full; fell back to unpinned (regular malloc) memory.
    Spilled {
        /// Size of the unpinned allocation in bytes.
        bytes: usize,
    },
    /// Allocation failed entirely.
    Failed {
        /// Human-readable reason.
        reason: String,
    },
}

impl AllocationResult {
    /// Returns `true` if the allocation produced a pinned buffer.
    pub fn is_pinned(&self) -> bool {
        matches!(self, Self::Pinned { .. })
    }

    /// Returns `true` if the allocation spilled to unpinned memory.
    pub fn is_spilled(&self) -> bool {
        matches!(self, Self::Spilled { .. })
    }

    /// Returns `true` if the allocation failed.
    pub fn is_failed(&self) -> bool {
        matches!(self, Self::Failed { .. })
    }
}

// ---------------------------------------------------------------------------
// MemoryPressureConfig
// ---------------------------------------------------------------------------

/// Thresholds for host memory pressure detection based on available memory
/// percentage (e.g. from `/proc/meminfo` `MemAvailable`).
///
/// When the percentage of system RAM that is *available* drops below these
/// thresholds, the pool should take progressively more aggressive action
/// (shrink, stop growing, reject new allocations).
#[derive(Debug, Clone)]
pub struct MemoryPressureConfig {
    /// When system available RAM is above this percent, the pool is healthy
    /// and may grow freely. Below this threshold, growth stops and the warning
    /// and critical thresholds take over.
    pub healthy_percent: f64,
    /// Below this percent available, emit warnings and stop growing.
    pub warning_percent: f64,
    /// Below this percent available, reject new allocations / force spill.
    pub critical_percent: f64,
}

impl Default for MemoryPressureConfig {
    fn default() -> Self {
        Self {
            healthy_percent: 30.0,
            warning_percent: 15.0,
            critical_percent: 5.0,
        }
    }
}

// ---------------------------------------------------------------------------
// SpillPolicy
// ---------------------------------------------------------------------------

/// Policy for what to do when the pinned pool is full and a new allocation
/// is requested.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpillPolicy {
    /// Block (wait) until a pinned buffer is freed.
    Block,
    /// Fall back to regular (unpinned) `malloc` memory. ~2x slower DMA but
    /// the transfer still works.
    SpillToUnpinned,
    /// Reject the allocation immediately with an error.
    Reject,
}

// ---------------------------------------------------------------------------
// BufferHandle
// ---------------------------------------------------------------------------

/// Lightweight handle to a buffer obtained from the pool.
///
/// Consumers hold this while performing a DMA transfer and return it when done.
/// The pool uses the handle to track which buffers are in-flight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferHandle {
    /// Slab class this buffer belongs to.
    pub class: SlabClass,
    /// Index within the slab class.
    pub index: u32,
    /// Whether this buffer is backed by pinned (page-locked) memory.
    pub pinned: bool,
}

impl BufferHandle {
    /// Size of the underlying buffer in bytes (determined by the slab class).
    pub const fn size_bytes(&self) -> usize {
        self.class.size_bytes()
    }
}

// ---------------------------------------------------------------------------
// PoolSlab
// ---------------------------------------------------------------------------

/// Bookkeeping for a single slab class within the pool.
///
/// Tracks capacity and allocation count. The actual memory is managed by the
/// server crate; this struct only does accounting.
#[derive(Debug, Clone)]
pub struct PoolSlab {
    /// Which slab class this represents.
    pub class: SlabClass,
    /// Maximum number of buffers of this class.
    pub capacity: u32,
    /// Currently allocated (checked-out) buffers.
    pub allocated: u32,
}

impl PoolSlab {
    /// Returns `true` when all buffers are in use.
    pub fn is_full(&self) -> bool {
        self.allocated >= self.capacity
    }

    /// Number of buffers available for allocation.
    pub fn available(&self) -> u32 {
        self.capacity.saturating_sub(self.allocated)
    }

    /// Utilization of this slab as a fraction in `[0.0, 1.0]`.
    ///
    /// Returns 0.0 when capacity is zero.
    pub fn utilization(&self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        self.allocated as f64 / self.capacity as f64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- SlabClass::size_bytes ---

    #[test]
    fn slab_class_tiny_size() {
        assert_eq!(SlabClass::Tiny.size_bytes(), 4096);
    }

    #[test]
    fn slab_class_small_size() {
        assert_eq!(SlabClass::Small.size_bytes(), 65_536);
    }

    #[test]
    fn slab_class_medium_size() {
        assert_eq!(SlabClass::Medium.size_bytes(), 1_048_576);
    }

    #[test]
    fn slab_class_large_size() {
        assert_eq!(SlabClass::Large.size_bytes(), 16_777_216);
    }

    #[test]
    fn slab_class_huge_size() {
        assert_eq!(SlabClass::Huge.size_bytes(), 268_435_456);
    }

    // --- SlabClass::from_request_size ---

    #[test]
    fn from_request_size_zero_returns_tiny() {
        assert_eq!(SlabClass::from_request_size(0), Some(SlabClass::Tiny));
    }

    #[test]
    fn from_request_size_exact_tiny() {
        assert_eq!(SlabClass::from_request_size(4096), Some(SlabClass::Tiny));
    }

    #[test]
    fn from_request_size_one_over_tiny() {
        assert_eq!(SlabClass::from_request_size(4097), Some(SlabClass::Small));
    }

    #[test]
    fn from_request_size_exact_small() {
        assert_eq!(SlabClass::from_request_size(65_536), Some(SlabClass::Small));
    }

    #[test]
    fn from_request_size_just_under_medium() {
        assert_eq!(SlabClass::from_request_size(1_048_575), Some(SlabClass::Medium));
    }

    #[test]
    fn from_request_size_exact_medium() {
        assert_eq!(SlabClass::from_request_size(1_048_576), Some(SlabClass::Medium));
    }

    #[test]
    fn from_request_size_one_over_medium() {
        assert_eq!(SlabClass::from_request_size(1_048_577), Some(SlabClass::Large));
    }

    #[test]
    fn from_request_size_exact_huge() {
        assert_eq!(SlabClass::from_request_size(268_435_456), Some(SlabClass::Huge));
    }

    #[test]
    fn from_request_size_exceeds_huge() {
        assert_eq!(SlabClass::from_request_size(268_435_457), None);
    }

    // --- PoolStats ---

    #[test]
    fn pool_stats_utilization_half() {
        let stats = PoolStats {
            allocated_bytes: 50,
            pool_capacity_bytes: 100,
            ..Default::default()
        };
        assert!((stats.utilization() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn pool_stats_utilization_zero_capacity() {
        let stats = PoolStats::default();
        assert_eq!(stats.utilization(), 0.0);
    }

    #[test]
    fn pool_stats_is_under_pressure_true() {
        let stats = PoolStats {
            allocated_bytes: 96,
            pool_capacity_bytes: 100,
            ..Default::default()
        };
        assert!(stats.is_under_pressure(0.95));
    }

    #[test]
    fn pool_stats_is_under_pressure_false() {
        let stats = PoolStats {
            allocated_bytes: 90,
            pool_capacity_bytes: 100,
            ..Default::default()
        };
        assert!(!stats.is_under_pressure(0.95));
    }

    // --- AllocationResult ---

    #[test]
    fn allocation_result_pinned() {
        let r = AllocationResult::Pinned {
            class: SlabClass::Medium,
            index: 7,
        };
        assert!(r.is_pinned());
        assert!(!r.is_spilled());
        assert!(!r.is_failed());
    }

    #[test]
    fn allocation_result_spilled() {
        let r = AllocationResult::Spilled { bytes: 1024 };
        assert!(!r.is_pinned());
        assert!(r.is_spilled());
        assert!(!r.is_failed());
    }

    #[test]
    fn allocation_result_failed() {
        let r = AllocationResult::Failed {
            reason: "out of memory".into(),
        };
        assert!(!r.is_pinned());
        assert!(!r.is_spilled());
        assert!(r.is_failed());
    }

    // --- BufferHandle ---

    #[test]
    fn buffer_handle_size_bytes() {
        let h = BufferHandle {
            class: SlabClass::Large,
            index: 0,
            pinned: true,
        };
        assert_eq!(h.size_bytes(), 16_777_216);
    }

    #[test]
    fn buffer_handle_unpinned_size() {
        let h = BufferHandle {
            class: SlabClass::Tiny,
            index: 3,
            pinned: false,
        };
        assert_eq!(h.size_bytes(), 4096);
    }

    // --- PoolSlab ---

    #[test]
    fn pool_slab_is_full() {
        let slab = PoolSlab {
            class: SlabClass::Small,
            capacity: 10,
            allocated: 10,
        };
        assert!(slab.is_full());
        assert_eq!(slab.available(), 0);
    }

    #[test]
    fn pool_slab_not_full() {
        let slab = PoolSlab {
            class: SlabClass::Small,
            capacity: 10,
            allocated: 7,
        };
        assert!(!slab.is_full());
        assert_eq!(slab.available(), 3);
    }

    #[test]
    fn pool_slab_utilization() {
        let slab = PoolSlab {
            class: SlabClass::Medium,
            capacity: 4,
            allocated: 1,
        };
        assert!((slab.utilization() - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn pool_slab_zero_capacity_utilization() {
        let slab = PoolSlab {
            class: SlabClass::Tiny,
            capacity: 0,
            allocated: 0,
        };
        assert_eq!(slab.utilization(), 0.0);
        // 0 >= 0 is true, so a zero-capacity slab is technically "full".
        assert!(slab.is_full());
    }

    // --- PoolConfig defaults ---

    #[test]
    fn pool_config_default_values() {
        let cfg = PoolConfig::default();
        assert_eq!(cfg.per_class_capacity, [64, 32, 16, 8, 1]);
        assert_eq!(cfg.spill_policy, SpillPolicy::SpillToUnpinned);
        assert_eq!(cfg.max_total_bytes, 512 * 1024 * 1024);
        assert_eq!(cfg.spill_threshold_percent, 95);
        assert!(cfg.grow_on_demand);
    }

    // --- MemoryPressureConfig defaults ---

    #[test]
    fn memory_pressure_config_defaults() {
        let cfg = MemoryPressureConfig::default();
        assert!((cfg.healthy_percent - 30.0).abs() < f64::EPSILON);
        assert!((cfg.warning_percent - 15.0).abs() < f64::EPSILON);
        assert!((cfg.critical_percent - 5.0).abs() < f64::EPSILON);
    }

    // --- SpillPolicy ---

    #[test]
    fn spill_policy_variants_distinct() {
        assert_ne!(SpillPolicy::Block, SpillPolicy::SpillToUnpinned);
        assert_ne!(SpillPolicy::Block, SpillPolicy::Reject);
        assert_ne!(SpillPolicy::SpillToUnpinned, SpillPolicy::Reject);
    }

    // --- SlabClass::ALL ---

    #[test]
    fn slab_class_all_ascending_size() {
        let sizes: Vec<usize> = SlabClass::ALL.iter().map(|c| c.size_bytes()).collect();
        for w in sizes.windows(2) {
            assert!(w[0] < w[1], "{} should be < {}", w[0], w[1]);
        }
    }
}
