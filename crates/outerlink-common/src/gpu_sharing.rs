//! Fractional GPU sharing types for OuterLink.
//!
//! Provides per-context VRAM quotas and compute quotas so multiple workloads
//! can share a single physical GPU. `VramQuota.try_allocate` is the hot path,
//! called on every `cuMemAlloc`. `QuotaExceeded` maps to `CUDA_ERROR_OUT_OF_MEMORY`
//! at the FFI boundary.

use std::fmt;
use std::time::Instant;

// ---------------------------------------------------------------------------
// QuotaExceeded error
// ---------------------------------------------------------------------------

/// Error returned when a VRAM allocation would exceed the quota.
///
/// At the FFI boundary this maps to `CuResult::OutOfMemory`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuotaExceeded {
    /// Bytes the caller tried to allocate.
    pub requested_bytes: u64,
    /// Hard cap for this quota.
    pub limit_bytes: u64,
    /// Bytes already committed.
    pub allocated_bytes: u64,
}

impl fmt::Display for QuotaExceeded {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VRAM quota exceeded: requested {} bytes, limit {} bytes, already allocated {} bytes ({} bytes remaining)",
            self.requested_bytes,
            self.limit_bytes,
            self.allocated_bytes,
            self.limit_bytes.saturating_sub(self.allocated_bytes),
        )
    }
}

impl std::error::Error for QuotaExceeded {}

// ---------------------------------------------------------------------------
// VramQuota
// ---------------------------------------------------------------------------

/// Tracks VRAM usage against a hard byte limit for one context.
///
/// `try_allocate` is the hot path -- called on every `cuMemAlloc`.
#[derive(Debug, Clone)]
pub struct VramQuota {
    /// Hard cap in bytes.
    pub limit_bytes: u64,
    /// Currently allocated bytes.
    pub allocated_bytes: u64,
    /// High-water mark (peak allocation seen).
    pub peak_bytes: u64,
}

impl VramQuota {
    /// Create a new quota with the given limit and zero allocation.
    pub fn new(limit_bytes: u64) -> Self {
        Self {
            limit_bytes,
            allocated_bytes: 0,
            peak_bytes: 0,
        }
    }

    /// Bytes still available before hitting the limit.
    #[inline]
    pub fn remaining(&self) -> u64 {
        self.limit_bytes.saturating_sub(self.allocated_bytes)
    }

    /// Utilization as a fraction in `[0.0, 1.0]`.
    /// Returns 0.0 when limit is 0 to avoid division by zero.
    #[inline]
    pub fn utilization(&self) -> f64 {
        if self.limit_bytes == 0 {
            return 0.0;
        }
        self.allocated_bytes as f64 / self.limit_bytes as f64
    }

    /// Returns `true` if allocating `bytes` would exceed the limit.
    #[inline]
    pub fn would_exceed(&self, bytes: u64) -> bool {
        self.allocated_bytes.checked_add(bytes).map_or(true, |total| total > self.limit_bytes)
    }

    /// Attempt to allocate `bytes`. Returns `Err(QuotaExceeded)` if it would
    /// exceed the limit. This is the hot path.
    #[inline]
    pub fn try_allocate(&mut self, bytes: u64) -> Result<(), QuotaExceeded> {
        if self.would_exceed(bytes) {
            return Err(QuotaExceeded {
                requested_bytes: bytes,
                limit_bytes: self.limit_bytes,
                allocated_bytes: self.allocated_bytes,
            });
        }
        self.allocated_bytes += bytes;
        if self.allocated_bytes > self.peak_bytes {
            self.peak_bytes = self.allocated_bytes;
        }
        Ok(())
    }

    /// Release `bytes` back. Saturates at zero (never underflows).
    #[inline]
    pub fn release(&mut self, bytes: u64) {
        self.allocated_bytes = self.allocated_bytes.saturating_sub(bytes);
    }
}

// ---------------------------------------------------------------------------
// ComputeQuota
// ---------------------------------------------------------------------------

/// Compute time-share parameters for one context.
#[derive(Debug, Clone)]
pub struct ComputeQuota {
    /// Maximum percentage of GPU compute time (0.0 - 100.0).
    pub max_percent: f64,
    /// Time-slice duration in milliseconds for round-robin scheduling.
    pub time_slice_ms: u32,
    /// Priority level (0 = lowest, 255 = highest). Default 128.
    pub priority: u8,
}

impl ComputeQuota {
    /// Create with defaults: 100% compute, 100ms slice, priority 128.
    pub fn new(max_percent: f64) -> Self {
        Self {
            max_percent,
            time_slice_ms: 100,
            priority: 128,
        }
    }

    /// Create with all parameters specified.
    pub fn with_params(max_percent: f64, time_slice_ms: u32, priority: u8) -> Self {
        Self {
            max_percent,
            time_slice_ms,
            priority,
        }
    }
}

impl Default for ComputeQuota {
    fn default() -> Self {
        Self {
            max_percent: 100.0,
            time_slice_ms: 100,
            priority: 128,
        }
    }
}

// ---------------------------------------------------------------------------
// FractionConfig
// ---------------------------------------------------------------------------

/// Configuration for creating a new GPU fraction. Typically parsed from
/// environment variables (`OUTERLINK_VRAM_LIMIT`) or container CDI specs.
#[derive(Debug, Clone)]
pub struct FractionConfig {
    /// VRAM hard limit in bytes.
    pub vram_limit_bytes: u64,
    /// Compute percentage (0.0 - 100.0).
    pub compute_percent: f64,
    /// Scheduling priority (0 = lowest, 255 = highest).
    pub priority: u8,
}

impl FractionConfig {
    pub fn new(vram_limit_bytes: u64, compute_percent: f64, priority: u8) -> Self {
        Self {
            vram_limit_bytes,
            compute_percent,
            priority,
        }
    }
}

// ---------------------------------------------------------------------------
// GpuFraction
// ---------------------------------------------------------------------------

/// A single "fraction" of a physical GPU allocated to one workload/context.
#[derive(Debug)]
pub struct GpuFraction {
    /// VRAM accounting for this fraction.
    pub vram_quota: VramQuota,
    /// Compute share for this fraction.
    pub compute_quota: ComputeQuota,
    /// Unique identifier for this fraction.
    pub context_id: u64,
    /// When this fraction was created.
    pub created_at: Instant,
}

impl GpuFraction {
    /// Create a new fraction from config, assigning the given context_id.
    pub fn from_config(config: &FractionConfig, context_id: u64) -> Self {
        Self {
            vram_quota: VramQuota::new(config.vram_limit_bytes),
            compute_quota: ComputeQuota::with_params(
                config.compute_percent,
                100,
                config.priority,
            ),
            context_id,
            created_at: Instant::now(),
        }
    }
}

// ---------------------------------------------------------------------------
// GpuPartitionTable
// ---------------------------------------------------------------------------

/// Manages all fractions on one physical GPU.
pub struct GpuPartitionTable {
    /// Physical GPU index.
    pub gpu_index: u32,
    /// Total VRAM on this GPU in bytes.
    pub total_vram: u64,
    /// Active fractions.
    fractions: Vec<GpuFraction>,
    /// Monotonically increasing context ID counter.
    next_context_id: u64,
}

impl GpuPartitionTable {
    /// Create a new partition table for a GPU with the given total VRAM.
    pub fn new(gpu_index: u32, total_vram: u64) -> Self {
        Self {
            gpu_index,
            total_vram,
            fractions: Vec::new(),
            next_context_id: 1,
        }
    }

    /// Add a new fraction. Returns the assigned context_id.
    pub fn add_fraction(&mut self, config: &FractionConfig) -> u64 {
        let id = self.next_context_id;
        self.next_context_id += 1;
        self.fractions.push(GpuFraction::from_config(config, id));
        id
    }

    /// Remove a fraction by context_id. Returns `true` if found and removed.
    pub fn remove_fraction(&mut self, context_id: u64) -> bool {
        let len_before = self.fractions.len();
        self.fractions.retain(|f| f.context_id != context_id);
        self.fractions.len() < len_before
    }

    /// VRAM not yet claimed by any fraction's limit.
    pub fn available_vram(&self) -> u64 {
        let claimed: u64 = self.fractions.iter().map(|f| f.vram_quota.limit_bytes).sum();
        self.total_vram.saturating_sub(claimed)
    }

    /// Sum of all fractions' current allocations.
    pub fn total_allocated_vram(&self) -> u64 {
        self.fractions.iter().map(|f| f.vram_quota.allocated_bytes).sum()
    }

    /// Number of active fractions.
    pub fn fraction_count(&self) -> usize {
        self.fractions.len()
    }

    /// Returns `true` if the sum of fraction limits exceeds physical VRAM.
    pub fn is_overcommitted(&self) -> bool {
        let total_limits: u64 = self.fractions.iter().map(|f| f.vram_quota.limit_bytes).sum();
        total_limits > self.total_vram
    }

    /// Get a reference to a fraction by context_id.
    pub fn get_fraction(&self, context_id: u64) -> Option<&GpuFraction> {
        self.fractions.iter().find(|f| f.context_id == context_id)
    }

    /// Get a mutable reference to a fraction by context_id.
    pub fn get_fraction_mut(&mut self, context_id: u64) -> Option<&mut GpuFraction> {
        self.fractions.iter_mut().find(|f| f.context_id == context_id)
    }
}

// ---------------------------------------------------------------------------
// parse_vram_limit
// ---------------------------------------------------------------------------

/// Parse a VRAM limit string from `OUTERLINK_VRAM_LIMIT` env var.
///
/// Accepted formats (case-insensitive suffixes):
/// - `"4G"` or `"4g"` -- gigabytes
/// - `"4096M"` or `"4096m"` -- megabytes
/// - `"4194304K"` or `"4194304k"` -- kilobytes
/// - `"4294967296"` -- bare bytes
///
/// Returns `None` for empty, zero, or unparseable values.
pub fn parse_vram_limit(env_val: &str) -> Option<u64> {
    let s = env_val.trim();
    if s.is_empty() {
        return None;
    }

    let (num_str, multiplier) = if let Some(n) = s.strip_suffix('G').or_else(|| s.strip_suffix('g')) {
        (n, 1024u64 * 1024 * 1024)
    } else if let Some(n) = s.strip_suffix('M').or_else(|| s.strip_suffix('m')) {
        (n, 1024u64 * 1024)
    } else if let Some(n) = s.strip_suffix('K').or_else(|| s.strip_suffix('k')) {
        (n, 1024u64)
    } else {
        (s, 1u64)
    };

    let value: u64 = num_str.trim().parse().ok()?;
    if value == 0 {
        return None;
    }
    value.checked_mul(multiplier)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- VramQuota tests --------------------------------------------------

    #[test]
    fn vram_quota_new_starts_empty() {
        let q = VramQuota::new(1024);
        assert_eq!(q.limit_bytes, 1024);
        assert_eq!(q.allocated_bytes, 0);
        assert_eq!(q.peak_bytes, 0);
        assert_eq!(q.remaining(), 1024);
    }

    #[test]
    fn vram_quota_try_allocate_success() {
        let mut q = VramQuota::new(1000);
        assert!(q.try_allocate(400).is_ok());
        assert_eq!(q.allocated_bytes, 400);
        assert_eq!(q.remaining(), 600);
        assert!(q.try_allocate(600).is_ok());
        assert_eq!(q.allocated_bytes, 1000);
        assert_eq!(q.remaining(), 0);
    }

    #[test]
    fn vram_quota_try_allocate_exceeds() {
        let mut q = VramQuota::new(1000);
        q.try_allocate(800).unwrap();
        let err = q.try_allocate(300).unwrap_err();
        assert_eq!(err.requested_bytes, 300);
        assert_eq!(err.limit_bytes, 1000);
        assert_eq!(err.allocated_bytes, 800);
    }

    #[test]
    fn vram_quota_try_allocate_exact_limit() {
        let mut q = VramQuota::new(512);
        assert!(q.try_allocate(512).is_ok());
        assert_eq!(q.remaining(), 0);
        // One more byte should fail
        assert!(q.try_allocate(1).is_err());
    }

    #[test]
    fn vram_quota_release() {
        let mut q = VramQuota::new(1000);
        q.try_allocate(600).unwrap();
        q.release(200);
        assert_eq!(q.allocated_bytes, 400);
        assert_eq!(q.remaining(), 600);
    }

    #[test]
    fn vram_quota_release_saturates_at_zero() {
        let mut q = VramQuota::new(1000);
        q.try_allocate(100).unwrap();
        q.release(500); // release more than allocated
        assert_eq!(q.allocated_bytes, 0);
    }

    #[test]
    fn vram_quota_peak_bytes_tracks_high_water() {
        let mut q = VramQuota::new(1000);
        q.try_allocate(700).unwrap();
        assert_eq!(q.peak_bytes, 700);
        q.release(400);
        assert_eq!(q.peak_bytes, 700); // peak unchanged
        q.try_allocate(200).unwrap();
        assert_eq!(q.peak_bytes, 700); // still 700 (500 current)
        q.try_allocate(300).unwrap();
        assert_eq!(q.peak_bytes, 800); // new peak
    }

    #[test]
    fn vram_quota_utilization() {
        let mut q = VramQuota::new(1000);
        assert!((q.utilization() - 0.0).abs() < f64::EPSILON);
        q.try_allocate(500).unwrap();
        assert!((q.utilization() - 0.5).abs() < f64::EPSILON);
        q.try_allocate(500).unwrap();
        assert!((q.utilization() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn vram_quota_utilization_zero_limit() {
        let q = VramQuota::new(0);
        assert!((q.utilization() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn vram_quota_would_exceed() {
        let mut q = VramQuota::new(1000);
        assert!(!q.would_exceed(1000));
        assert!(q.would_exceed(1001));
        q.try_allocate(999).unwrap();
        assert!(!q.would_exceed(1));
        assert!(q.would_exceed(2));
    }

    #[test]
    fn vram_quota_would_exceed_overflow() {
        let mut q = VramQuota::new(u64::MAX);
        q.allocated_bytes = u64::MAX - 10;
        // Adding 20 would overflow u64, should return true
        assert!(q.would_exceed(20));
    }

    // -- QuotaExceeded tests ----------------------------------------------

    #[test]
    fn quota_exceeded_display() {
        let err = QuotaExceeded {
            requested_bytes: 300,
            limit_bytes: 1000,
            allocated_bytes: 800,
        };
        let msg = err.to_string();
        assert!(msg.contains("300"), "should contain requested bytes");
        assert!(msg.contains("1000"), "should contain limit");
        assert!(msg.contains("800"), "should contain allocated");
        assert!(msg.contains("200"), "should contain remaining");
    }

    // -- ComputeQuota tests -----------------------------------------------

    #[test]
    fn compute_quota_defaults() {
        let cq = ComputeQuota::default();
        assert!((cq.max_percent - 100.0).abs() < f64::EPSILON);
        assert_eq!(cq.time_slice_ms, 100);
        assert_eq!(cq.priority, 128);
    }

    #[test]
    fn compute_quota_new_uses_defaults() {
        let cq = ComputeQuota::new(50.0);
        assert!((cq.max_percent - 50.0).abs() < f64::EPSILON);
        assert_eq!(cq.time_slice_ms, 100);
        assert_eq!(cq.priority, 128);
    }

    #[test]
    fn compute_quota_with_params() {
        let cq = ComputeQuota::with_params(25.0, 50, 200);
        assert!((cq.max_percent - 25.0).abs() < f64::EPSILON);
        assert_eq!(cq.time_slice_ms, 50);
        assert_eq!(cq.priority, 200);
    }

    // -- GpuFraction tests ------------------------------------------------

    #[test]
    fn gpu_fraction_from_config() {
        let cfg = FractionConfig::new(4 * 1024 * 1024 * 1024, 50.0, 200);
        let frac = GpuFraction::from_config(&cfg, 42);
        assert_eq!(frac.context_id, 42);
        assert_eq!(frac.vram_quota.limit_bytes, 4 * 1024 * 1024 * 1024);
        assert!((frac.compute_quota.max_percent - 50.0).abs() < f64::EPSILON);
        assert_eq!(frac.compute_quota.priority, 200);
    }

    // -- GpuPartitionTable tests ------------------------------------------

    #[test]
    fn partition_table_new_empty() {
        let pt = GpuPartitionTable::new(0, 24 * 1024 * 1024 * 1024);
        assert_eq!(pt.gpu_index, 0);
        assert_eq!(pt.fraction_count(), 0);
        assert_eq!(pt.available_vram(), 24 * 1024 * 1024 * 1024);
        assert_eq!(pt.total_allocated_vram(), 0);
        assert!(!pt.is_overcommitted());
    }

    #[test]
    fn partition_table_add_fraction_returns_unique_ids() {
        let mut pt = GpuPartitionTable::new(0, 24_000_000_000);
        let cfg = FractionConfig::new(4_000_000_000, 50.0, 128);
        let id1 = pt.add_fraction(&cfg);
        let id2 = pt.add_fraction(&cfg);
        let id3 = pt.add_fraction(&cfg);
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_eq!(pt.fraction_count(), 3);
    }

    #[test]
    fn partition_table_available_vram_decreases() {
        let mut pt = GpuPartitionTable::new(0, 10_000);
        let cfg = FractionConfig::new(3_000, 100.0, 128);
        pt.add_fraction(&cfg);
        assert_eq!(pt.available_vram(), 7_000);
        pt.add_fraction(&cfg);
        assert_eq!(pt.available_vram(), 4_000);
    }

    #[test]
    fn partition_table_remove_fraction() {
        let mut pt = GpuPartitionTable::new(0, 10_000);
        let cfg = FractionConfig::new(3_000, 100.0, 128);
        let id = pt.add_fraction(&cfg);
        assert_eq!(pt.fraction_count(), 1);
        assert!(pt.remove_fraction(id));
        assert_eq!(pt.fraction_count(), 0);
        assert_eq!(pt.available_vram(), 10_000);
    }

    #[test]
    fn partition_table_remove_nonexistent_returns_false() {
        let mut pt = GpuPartitionTable::new(0, 10_000);
        assert!(!pt.remove_fraction(999));
    }

    #[test]
    fn partition_table_total_allocated_vram() {
        let mut pt = GpuPartitionTable::new(0, 10_000);
        let cfg = FractionConfig::new(5_000, 100.0, 128);
        let id1 = pt.add_fraction(&cfg);
        let id2 = pt.add_fraction(&cfg);
        // Allocate within fractions
        pt.get_fraction_mut(id1).unwrap().vram_quota.try_allocate(1_000).unwrap();
        pt.get_fraction_mut(id2).unwrap().vram_quota.try_allocate(2_000).unwrap();
        assert_eq!(pt.total_allocated_vram(), 3_000);
    }

    #[test]
    fn partition_table_overcommit_detection() {
        let mut pt = GpuPartitionTable::new(0, 8_000);
        let cfg = FractionConfig::new(5_000, 100.0, 128);
        pt.add_fraction(&cfg);
        assert!(!pt.is_overcommitted()); // 5000 <= 8000
        pt.add_fraction(&cfg);
        assert!(pt.is_overcommitted()); // 10000 > 8000
    }

    #[test]
    fn partition_table_get_fraction() {
        let mut pt = GpuPartitionTable::new(0, 10_000);
        let cfg = FractionConfig::new(5_000, 75.0, 128);
        let id = pt.add_fraction(&cfg);
        let frac = pt.get_fraction(id).unwrap();
        assert_eq!(frac.context_id, id);
        assert_eq!(frac.vram_quota.limit_bytes, 5_000);
        assert!(pt.get_fraction(999).is_none());
    }

    // -- parse_vram_limit tests -------------------------------------------

    #[test]
    fn parse_vram_limit_gigabytes() {
        assert_eq!(parse_vram_limit("4G"), Some(4 * 1024 * 1024 * 1024));
        assert_eq!(parse_vram_limit("4g"), Some(4 * 1024 * 1024 * 1024));
        assert_eq!(parse_vram_limit("1G"), Some(1024 * 1024 * 1024));
    }

    #[test]
    fn parse_vram_limit_megabytes() {
        assert_eq!(parse_vram_limit("4096M"), Some(4096 * 1024 * 1024));
        assert_eq!(parse_vram_limit("512m"), Some(512 * 1024 * 1024));
    }

    #[test]
    fn parse_vram_limit_kilobytes() {
        assert_eq!(parse_vram_limit("4194304K"), Some(4194304 * 1024));
        assert_eq!(parse_vram_limit("1024k"), Some(1024 * 1024));
    }

    #[test]
    fn parse_vram_limit_bare_bytes() {
        assert_eq!(parse_vram_limit("4294967296"), Some(4294967296));
        assert_eq!(parse_vram_limit("1024"), Some(1024));
    }

    #[test]
    fn parse_vram_limit_with_whitespace() {
        assert_eq!(parse_vram_limit("  4G  "), Some(4 * 1024 * 1024 * 1024));
    }

    #[test]
    fn parse_vram_limit_empty_returns_none() {
        assert_eq!(parse_vram_limit(""), None);
        assert_eq!(parse_vram_limit("   "), None);
    }

    #[test]
    fn parse_vram_limit_zero_returns_none() {
        assert_eq!(parse_vram_limit("0"), None);
        assert_eq!(parse_vram_limit("0G"), None);
    }

    #[test]
    fn parse_vram_limit_invalid_returns_none() {
        assert_eq!(parse_vram_limit("abc"), None);
        assert_eq!(parse_vram_limit("G"), None);
        assert_eq!(parse_vram_limit("-1G"), None);
    }
}
