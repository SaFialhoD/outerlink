//! Checkpoint/Restore types for OuterLink.
//!
//! CRIU cannot checkpoint GPU state, so OuterLink uses framework-level
//! checkpointing (PyTorch/JAX) combined with dirty-page tracking (R10)
//! for incremental VRAM snapshots.

/// Compression method for checkpoint data.
#[derive(Debug, Clone, PartialEq)]
pub enum CompressionMethod {
    /// No compression — raw VRAM bytes.
    None,
    /// LZ4 fast compression (~3 GB/s encode, minimal CPU overhead).
    Lz4,
    /// Zstandard compression with configurable level (1-22).
    Zstd { level: i32 },
}

impl CompressionMethod {
    /// Returns the file extension for this compression method.
    pub fn extension(&self) -> &'static str {
        match self {
            CompressionMethod::None => ".raw",
            CompressionMethod::Lz4 => ".lz4",
            CompressionMethod::Zstd { .. } => ".zst",
        }
    }
}

/// ML framework that initiated the checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub enum FrameworkType {
    PyTorch,
    JAX,
    TensorFlow,
    Custom(String),
    Unknown,
}

/// Configuration for automatic checkpointing.
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Whether checkpointing is enabled.
    pub enabled: bool,
    /// Interval between automatic checkpoints in seconds.
    pub interval_secs: u64,
    /// Maximum number of snapshots to retain before rotation.
    pub max_snapshots: u32,
    /// Whether to write checkpoints asynchronously (non-blocking).
    pub async_write: bool,
    /// Compression method for snapshot data.
    pub compression: CompressionMethod,
    /// Whether to include optimizer state (momentum, etc.) in checkpoints.
    pub include_optimizer_state: bool,
    /// Filesystem path for checkpoint storage.
    pub storage_path: String,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            interval_secs: 300,
            max_snapshots: 3,
            async_write: true,
            compression: CompressionMethod::Lz4, // Lz4 is the safe default: 5% overhead, ~2-3x compression
            include_optimizer_state: true,
            storage_path: "/var/lib/outerlink/checkpoints".to_string(),
        }
    }
}

/// Metadata for a completed VRAM checkpoint.
#[derive(Debug, Clone)]
pub struct CheckpointMetadata {
    /// Unique identifier for this checkpoint (UUID format).
    pub checkpoint_id: String,
    /// CUDA context ID that was checkpointed.
    pub context_id: u64,
    /// Unix timestamp in milliseconds when checkpoint was created.
    pub timestamp_epoch_ms: u64,
    /// Total VRAM bytes captured in this snapshot.
    pub vram_snapshot_bytes: u64,
    /// Number of dirty pages included in this snapshot.
    pub dirty_pages: u64,
    /// Total number of tracked pages at checkpoint time.
    pub total_pages: u64,
    /// Compression method used.
    pub compression: CompressionMethod,
    /// Time taken to complete the checkpoint in milliseconds.
    pub duration_ms: u64,
    /// ML framework that initiated the checkpoint.
    pub framework: FrameworkType,
}

impl CheckpointMetadata {
    /// Returns the ratio of dirty pages to total pages (0.0 if no pages).
    pub fn dirty_ratio(&self) -> f64 {
        if self.total_pages == 0 {
            return 0.0;
        }
        self.dirty_pages as f64 / self.total_pages as f64
    }

    /// Returns the compression ratio (compressed / original).
    /// Returns 0.0 if vram_snapshot_bytes is zero.
    pub fn compression_ratio(&self, compressed_bytes: u64) -> f64 {
        if self.vram_snapshot_bytes == 0 {
            return 0.0;
        }
        compressed_bytes as f64 / self.vram_snapshot_bytes as f64
    }
}

/// Request to restore a checkpoint onto a GPU.
#[derive(Debug, Clone)]
pub struct RestoreRequest {
    /// ID of the checkpoint to restore.
    pub checkpoint_id: String,
    /// Target GPU index to restore onto.
    pub target_gpu_index: u32,
    /// Whether to verify data integrity after restore.
    pub verify_integrity: bool,
}

/// Result of a checkpoint restore operation.
#[derive(Debug, Clone)]
pub struct RestoreResult {
    /// Whether the restore completed successfully.
    pub success: bool,
    /// Number of VRAM pages restored.
    pub restored_pages: u64,
    /// Time taken in milliseconds.
    pub duration_ms: u64,
    /// Result of integrity check (None if not requested).
    pub integrity_check_passed: Option<bool>,
    /// Any errors encountered during restore.
    pub errors: Vec<String>,
}

/// Strategy for VRAM snapshotting.
#[derive(Debug, Clone, PartialEq)]
pub enum SnapshotStrategy {
    /// Snapshot all VRAM pages.
    Full,
    /// Only dirty pages since the last checkpoint.
    Incremental,
    /// Dirty pages since the first (base) checkpoint.
    Differential,
}

impl SnapshotStrategy {
    /// Human-readable description of this strategy.
    pub fn description(&self) -> &'static str {
        match self {
            SnapshotStrategy::Full => "Full snapshot of all VRAM pages",
            SnapshotStrategy::Incremental => {
                "Incremental snapshot of dirty pages since last checkpoint"
            }
            SnapshotStrategy::Differential => {
                "Differential snapshot of dirty pages since base checkpoint"
            }
        }
    }

    /// Whether this strategy requires an existing base snapshot.
    pub fn requires_base_snapshot(&self) -> bool {
        match self {
            SnapshotStrategy::Full => false,
            SnapshotStrategy::Incremental | SnapshotStrategy::Differential => true,
        }
    }
}

/// Tracks checkpoint storage usage and snapshot inventory.
#[derive(Debug, Clone)]
pub struct CheckpointStorage {
    /// Filesystem path where checkpoints are stored.
    pub path: String,
    /// Current bytes used by checkpoint data.
    pub used_bytes: u64,
    /// Total capacity in bytes.
    pub capacity_bytes: u64,
    /// List of stored snapshots, ordered by timestamp.
    pub snapshots: Vec<CheckpointMetadata>,
}

impl CheckpointStorage {
    /// Returns storage utilization as a fraction (0.0 to 1.0).
    /// Returns 0.0 if capacity is zero.
    pub fn utilization(&self) -> f64 {
        if self.capacity_bytes == 0 {
            return 0.0;
        }
        self.used_bytes as f64 / self.capacity_bytes as f64
    }

    /// Returns the oldest snapshot by timestamp, if any.
    pub fn oldest_snapshot(&self) -> Option<&CheckpointMetadata> {
        self.snapshots
            .iter()
            .min_by_key(|s| s.timestamp_epoch_ms)
    }

    /// Returns the newest snapshot by timestamp, if any.
    pub fn newest_snapshot(&self) -> Option<&CheckpointMetadata> {
        self.snapshots
            .iter()
            .max_by_key(|s| s.timestamp_epoch_ms)
    }

    /// Whether the number of stored snapshots meets or exceeds the rotation limit.
    pub fn should_rotate(&self, max: u32) -> bool {
        self.snapshots.len() >= max as usize
    }

    /// Returns the total number of stored snapshots.
    pub fn total_snapshots(&self) -> usize {
        self.snapshots.len()
    }
}

/// Estimates checkpoint duration in milliseconds.
///
/// `bandwidth_bps` is in bits per second.
/// Applies a compression overhead factor: None=0%, Lz4=5%, Zstd=11-32% based on level (10 + level, clamped 1-22).
///
/// Returns `u64::MAX` if bandwidth is zero.
pub fn estimate_checkpoint_time(
    vram_bytes: u64,
    bandwidth_bps: u64,
    compression: &CompressionMethod,
) -> u64 {
    if bandwidth_bps == 0 {
        return u64::MAX;
    }
    if vram_bytes == 0 {
        return 0;
    }

    // Convert bandwidth from bits/s to bytes/s
    let bandwidth_bytes_per_sec = bandwidth_bps / 8;
    if bandwidth_bytes_per_sec == 0 {
        return u64::MAX;
    }

    // Base transfer time in ms
    let base_ms = (vram_bytes * 1000) / bandwidth_bytes_per_sec;

    // Compression overhead factor (percentage added on top of transfer time)
    let overhead_pct = match compression {
        CompressionMethod::None => 0u64,
        CompressionMethod::Lz4 => 5,
        CompressionMethod::Zstd { level } => {
            // Higher levels = more CPU time = more overhead
            let clamped = (*level).clamp(1, 22) as u64;
            10 + clamped // 11% at level 1, up to 32% at level 22
        }
    };

    base_ms + (base_ms * overhead_pct) / 100
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── CheckpointConfig tests ──────────────────────────────────────

    #[test]
    fn checkpoint_config_defaults() {
        let cfg = CheckpointConfig::default();
        assert!(!cfg.enabled);
        assert_eq!(cfg.interval_secs, 300);
        assert_eq!(cfg.max_snapshots, 3);
        assert!(cfg.async_write);
        assert!(matches!(cfg.compression, CompressionMethod::Lz4));
        assert!(cfg.include_optimizer_state);
        assert_eq!(cfg.storage_path, "/var/lib/outerlink/checkpoints");
    }

    #[test]
    fn checkpoint_config_custom() {
        let cfg = CheckpointConfig {
            enabled: true,
            interval_secs: 60,
            max_snapshots: 10,
            async_write: false,
            compression: CompressionMethod::Lz4,
            include_optimizer_state: false,
            storage_path: "/tmp/ckpts".to_string(),
        };
        assert!(cfg.enabled);
        assert_eq!(cfg.interval_secs, 60);
        assert_eq!(cfg.max_snapshots, 10);
        assert!(!cfg.async_write);
        assert!(!cfg.include_optimizer_state);
    }

    // ── CompressionMethod tests ─────────────────────────────────────

    #[test]
    fn compression_none_extension() {
        assert_eq!(CompressionMethod::None.extension(), ".raw");
    }

    #[test]
    fn compression_lz4_extension() {
        assert_eq!(CompressionMethod::Lz4.extension(), ".lz4");
    }

    #[test]
    fn compression_zstd_extension() {
        let zstd = CompressionMethod::Zstd { level: 3 };
        assert_eq!(zstd.extension(), ".zst");
    }

    #[test]
    fn compression_zstd_different_levels() {
        let low = CompressionMethod::Zstd { level: 1 };
        let high = CompressionMethod::Zstd { level: 19 };
        // Both produce same extension regardless of level
        assert_eq!(low.extension(), high.extension());
    }

    // ── FrameworkType tests ─────────────────────────────────────────

    #[test]
    fn framework_type_variants() {
        let pytorch = FrameworkType::PyTorch;
        let jax = FrameworkType::JAX;
        let tf = FrameworkType::TensorFlow;
        let custom = FrameworkType::Custom("DeepSpeed".to_string());
        let unknown = FrameworkType::Unknown;

        // Debug representation works (proves Debug derive)
        assert!(format!("{:?}", pytorch).contains("PyTorch"));
        assert!(format!("{:?}", jax).contains("JAX"));
        assert!(format!("{:?}", tf).contains("TensorFlow"));
        assert!(format!("{:?}", custom).contains("DeepSpeed"));
        assert!(format!("{:?}", unknown).contains("Unknown"));
    }

    #[test]
    fn framework_type_clone() {
        let original = FrameworkType::Custom("MyFramework".to_string());
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    // ── CheckpointMetadata tests ────────────────────────────────────

    #[test]
    fn metadata_dirty_ratio_all_dirty() {
        let meta = CheckpointMetadata {
            checkpoint_id: "ckpt-001".to_string(),
            context_id: 42,
            timestamp_epoch_ms: 1700000000000,
            vram_snapshot_bytes: 1024 * 1024,
            dirty_pages: 100,
            total_pages: 100,
            compression: CompressionMethod::Lz4, // Lz4 is the safe default: 5% overhead, ~2-3x compression
            duration_ms: 500,
            framework: FrameworkType::PyTorch,
        };
        assert!((meta.dirty_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn metadata_dirty_ratio_half() {
        let meta = CheckpointMetadata {
            checkpoint_id: "ckpt-002".to_string(),
            context_id: 1,
            timestamp_epoch_ms: 1700000000000,
            vram_snapshot_bytes: 512 * 1024,
            dirty_pages: 50,
            total_pages: 100,
            compression: CompressionMethod::Lz4,
            duration_ms: 250,
            framework: FrameworkType::JAX,
        };
        assert!((meta.dirty_ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn metadata_dirty_ratio_zero_total() {
        let meta = CheckpointMetadata {
            checkpoint_id: "ckpt-003".to_string(),
            context_id: 1,
            timestamp_epoch_ms: 0,
            vram_snapshot_bytes: 0,
            dirty_pages: 0,
            total_pages: 0,
            compression: CompressionMethod::Lz4, // Lz4 is the safe default: 5% overhead, ~2-3x compression
            duration_ms: 0,
            framework: FrameworkType::Unknown,
        };
        assert!((meta.dirty_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn metadata_compression_ratio() {
        let meta = CheckpointMetadata {
            checkpoint_id: "ckpt-004".to_string(),
            context_id: 1,
            timestamp_epoch_ms: 1700000000000,
            vram_snapshot_bytes: 1000,
            dirty_pages: 10,
            total_pages: 10,
            compression: CompressionMethod::Zstd { level: 3 },
            duration_ms: 100,
            framework: FrameworkType::TensorFlow,
        };
        // 500 compressed from 1000 original = 0.5 ratio
        assert!((meta.compression_ratio(500) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn metadata_compression_ratio_zero_original() {
        let meta = CheckpointMetadata {
            checkpoint_id: "ckpt-005".to_string(),
            context_id: 1,
            timestamp_epoch_ms: 0,
            vram_snapshot_bytes: 0,
            dirty_pages: 0,
            total_pages: 0,
            compression: CompressionMethod::Lz4, // Lz4 is the safe default: 5% overhead, ~2-3x compression
            duration_ms: 0,
            framework: FrameworkType::Unknown,
        };
        assert!((meta.compression_ratio(0) - 0.0).abs() < f64::EPSILON);
    }

    // ── RestoreRequest tests ────────────────────────────────────────

    #[test]
    fn restore_request_defaults() {
        let req = RestoreRequest {
            checkpoint_id: "ckpt-001".to_string(),
            target_gpu_index: 0,
            verify_integrity: true,
        };
        assert!(req.verify_integrity);
        assert_eq!(req.target_gpu_index, 0);
    }

    // ── RestoreResult tests ─────────────────────────────────────────

    #[test]
    fn restore_result_success() {
        let result = RestoreResult {
            success: true,
            restored_pages: 1024,
            duration_ms: 3000,
            integrity_check_passed: Some(true),
            errors: vec![],
        };
        assert!(result.success);
        assert_eq!(result.restored_pages, 1024);
        assert_eq!(result.integrity_check_passed, Some(true));
        assert!(result.errors.is_empty());
    }

    #[test]
    fn restore_result_failure_with_errors() {
        let result = RestoreResult {
            success: false,
            restored_pages: 0,
            duration_ms: 100,
            integrity_check_passed: Some(false),
            errors: vec![
                "CRC mismatch on page 42".to_string(),
                "Missing metadata block".to_string(),
            ],
        };
        assert!(!result.success);
        assert_eq!(result.errors.len(), 2);
    }

    #[test]
    fn restore_result_no_integrity_check() {
        let result = RestoreResult {
            success: true,
            restored_pages: 512,
            duration_ms: 1500,
            integrity_check_passed: None,
            errors: vec![],
        };
        assert!(result.integrity_check_passed.is_none());
    }

    // ── SnapshotStrategy tests ──────────────────────────────────────

    #[test]
    fn snapshot_strategy_full_description() {
        let desc = SnapshotStrategy::Full.description();
        assert!(!desc.is_empty());
    }

    #[test]
    fn snapshot_strategy_incremental_description() {
        let desc = SnapshotStrategy::Incremental.description();
        assert!(!desc.is_empty());
    }

    #[test]
    fn snapshot_strategy_differential_description() {
        let desc = SnapshotStrategy::Differential.description();
        assert!(!desc.is_empty());
    }

    #[test]
    fn snapshot_strategy_requires_base() {
        assert!(!SnapshotStrategy::Full.requires_base_snapshot());
        assert!(SnapshotStrategy::Incremental.requires_base_snapshot());
        assert!(SnapshotStrategy::Differential.requires_base_snapshot());
    }

    // ── CheckpointStorage tests ─────────────────────────────────────

    #[test]
    fn storage_utilization() {
        let storage = CheckpointStorage {
            path: "/data/checkpoints".to_string(),
            used_bytes: 500,
            capacity_bytes: 1000,
            snapshots: vec![],
        };
        assert!((storage.utilization() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn storage_utilization_zero_capacity() {
        let storage = CheckpointStorage {
            path: "/data".to_string(),
            used_bytes: 0,
            capacity_bytes: 0,
            snapshots: vec![],
        };
        assert!((storage.utilization() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn storage_oldest_newest_snapshot() {
        let s1 = CheckpointMetadata {
            checkpoint_id: "old".to_string(),
            context_id: 1,
            timestamp_epoch_ms: 1000,
            vram_snapshot_bytes: 100,
            dirty_pages: 1,
            total_pages: 10,
            compression: CompressionMethod::Lz4, // Lz4 is the safe default: 5% overhead, ~2-3x compression
            duration_ms: 50,
            framework: FrameworkType::PyTorch,
        };
        let s2 = CheckpointMetadata {
            checkpoint_id: "new".to_string(),
            context_id: 1,
            timestamp_epoch_ms: 2000,
            vram_snapshot_bytes: 200,
            dirty_pages: 2,
            total_pages: 10,
            compression: CompressionMethod::Lz4, // Lz4 is the safe default: 5% overhead, ~2-3x compression
            duration_ms: 60,
            framework: FrameworkType::PyTorch,
        };
        let storage = CheckpointStorage {
            path: "/data".to_string(),
            used_bytes: 300,
            capacity_bytes: 10000,
            snapshots: vec![s1, s2],
        };
        assert_eq!(storage.oldest_snapshot().unwrap().checkpoint_id, "old");
        assert_eq!(storage.newest_snapshot().unwrap().checkpoint_id, "new");
    }

    #[test]
    fn storage_oldest_newest_empty() {
        let storage = CheckpointStorage {
            path: "/data".to_string(),
            used_bytes: 0,
            capacity_bytes: 1000,
            snapshots: vec![],
        };
        assert!(storage.oldest_snapshot().is_none());
        assert!(storage.newest_snapshot().is_none());
    }

    #[test]
    fn storage_should_rotate() {
        let snapshots: Vec<CheckpointMetadata> = (0..5)
            .map(|i| CheckpointMetadata {
                checkpoint_id: format!("ckpt-{}", i),
                context_id: 1,
                timestamp_epoch_ms: i as u64 * 1000,
                vram_snapshot_bytes: 100,
                dirty_pages: 1,
                total_pages: 10,
                compression: CompressionMethod::Lz4, // Lz4 is the safe default: 5% overhead, ~2-3x compression
                duration_ms: 50,
                framework: FrameworkType::Unknown,
            })
            .collect();
        let storage = CheckpointStorage {
            path: "/data".to_string(),
            used_bytes: 500,
            capacity_bytes: 10000,
            snapshots,
        };
        assert!(storage.should_rotate(3));
        assert!(storage.should_rotate(5));
        assert!(!storage.should_rotate(6));
    }

    #[test]
    fn storage_total_snapshots() {
        let storage = CheckpointStorage {
            path: "/data".to_string(),
            used_bytes: 0,
            capacity_bytes: 1000,
            snapshots: vec![],
        };
        assert_eq!(storage.total_snapshots(), 0);
    }

    // ── estimate_checkpoint_time tests ──────────────────────────────

    #[test]
    fn estimate_time_basic() {
        // 1 GB at 10 Gbps = ~800ms raw, no compression overhead
        let vram = 1_000_000_000; // 1 GB
        let bw = 10_000_000_000; // 10 Gbps in bits => bytes = 1.25 GB/s
        let ms = estimate_checkpoint_time(vram, bw, &CompressionMethod::None);
        // 1 GB / (10 Gbps / 8) = 1 GB / 1.25 GB/s = 0.8s = 800ms
        assert_eq!(ms, 800);
    }

    #[test]
    fn estimate_time_with_lz4() {
        let vram = 1_000_000_000;
        let bw = 10_000_000_000;
        let ms = estimate_checkpoint_time(vram, bw, &CompressionMethod::Lz4);
        // Lz4 adds some overhead but is very fast
        assert!(ms > 0);
    }

    #[test]
    fn estimate_time_with_zstd() {
        let vram = 1_000_000_000;
        let bw = 10_000_000_000;
        let ms_zstd = estimate_checkpoint_time(vram, bw, &CompressionMethod::Zstd { level: 3 });
        let ms_none = estimate_checkpoint_time(vram, bw, &CompressionMethod::None);
        // Zstd has higher overhead than no compression
        assert!(ms_zstd > ms_none);
    }

    #[test]
    fn estimate_time_zero_bandwidth() {
        let ms = estimate_checkpoint_time(1000, 0, &CompressionMethod::None);
        assert_eq!(ms, u64::MAX);
    }

    #[test]
    fn estimate_time_zero_vram() {
        let ms = estimate_checkpoint_time(0, 10_000_000_000, &CompressionMethod::None);
        assert_eq!(ms, 0);
    }
}
