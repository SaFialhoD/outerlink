//! R22: Live Migration
//!
//! Provides live migration of running GPU workloads between OuterLink nodes.
//! Supports pre-copy (iterative dirty page convergence), post-copy (demand paging),
//! and hybrid strategies. Includes dirty bitmap tracking, CUDA context snapshot
//! serialization, per-page migration state tracking, and a migration orchestrator
//! that coordinates the full lifecycle.
//!
//! Hardware-dependent operations (actual CUDA calls, network transfers, RDMA) are
//! represented as stubs with TODO comments. All state machines, algorithms, and
//! data structures are fully functional and tested.
//!
//! # Integration Points
//!
//! - R10 (Memory Tiering): page table lookups, PTE flag updates, dirty tracking
//! - R15 (Fault Tolerance): failure detection, checkpoint bootstrap
//! - R19 (Network Page Faults): post-copy demand paging
//! - R17 (Topology): bandwidth estimation, destination selection

use std::time::{Duration, Instant};

use crate::fault_tolerance::Vpn;
use crate::memory::types::{NodeId, PressureLevel, TierId};

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// Unique identifier for a live migration operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LiveMigrationId(pub u64);

// ---------------------------------------------------------------------------
// Migration Strategy
// ---------------------------------------------------------------------------

/// Which strategy the migration uses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MigrationStrategy {
    /// Pure pre-copy: iterative dirty page convergence.
    PreCopy,
    /// Pure post-copy: immediate switchover, demand paging.
    PostCopy,
    /// Hybrid: bounded pre-copy, then post-copy if not converged.
    Hybrid {
        /// Maximum pre-copy iterations before switching to post-copy.
        max_precopy_rounds: u32,
        /// Maximum time in pre-copy phase before switching.
        max_precopy_duration: Duration,
        /// Dirty page ratio threshold below which pre-copy is "converged".
        /// Expressed as fraction of total pages (e.g., 0.02 = 2%).
        convergence_threshold: f64,
    },
}

impl MigrationStrategy {
    /// Whether this strategy allows switching to post-copy.
    pub fn allows_postcopy(&self) -> bool {
        matches!(self, MigrationStrategy::PostCopy | MigrationStrategy::Hybrid { .. })
    }
}

impl Default for MigrationStrategy {
    fn default() -> Self {
        Self::Hybrid {
            max_precopy_rounds: 8,
            max_precopy_duration: Duration::from_secs(30),
            convergence_threshold: 0.02,
        }
    }
}

// ---------------------------------------------------------------------------
// Migration Priority
// ---------------------------------------------------------------------------

/// Priority levels for migration operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum LiveMigrationPriority {
    /// Background migration (load balancing, resource reclamation).
    Background = 0,
    /// Normal migration (user-initiated drain).
    Normal = 1,
    /// Urgent migration (pre-failure, thermal throttle).
    Urgent = 2,
    /// Emergency migration (node degrading, imminent failure).
    Emergency = 3,
}

// ---------------------------------------------------------------------------
// Migration Configuration
// ---------------------------------------------------------------------------

/// Configuration for a live migration operation.
#[derive(Debug, Clone)]
pub struct LiveMigrationConfig {
    /// Migration strategy to use.
    pub strategy: MigrationStrategy,
    /// Maximum acceptable blackout (pause) duration.
    /// If exceeded, migration is aborted.
    pub max_blackout: Duration,
    /// Maximum total migration duration (pre-copy + blackout + post-copy).
    pub max_total_duration: Duration,
    /// Whether to compress pages during transfer.
    pub compress_transfers: bool,
    /// Whether to use R15 checkpoints as migration source.
    pub use_checkpoint_bootstrap: bool,
    /// Priority relative to other migration operations.
    pub priority: LiveMigrationPriority,
    /// Network bandwidth limit for migration traffic (bytes/sec, 0 = unlimited).
    pub bandwidth_limit: u64,
    /// Page size for dirty tracking bitmap (must match R10 page size).
    pub page_size: usize,
    /// Whether to validate page checksums after transfer.
    pub verify_checksums: bool,
}

impl Default for LiveMigrationConfig {
    fn default() -> Self {
        Self {
            strategy: MigrationStrategy::default(),
            max_blackout: Duration::from_millis(500),
            max_total_duration: Duration::from_secs(120),
            compress_transfers: true,
            use_checkpoint_bootstrap: true,
            priority: LiveMigrationPriority::Normal,
            bandwidth_limit: 0,
            page_size: 65536, // 64 KB, matching R10
            verify_checksums: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Migration State Machine
// ---------------------------------------------------------------------------

/// The lifecycle states of a live migration operation.
///
/// State transitions:
///
/// Requested -> Preparing -> PreCopy -> (Converged | SwitchToPostCopy)
///                                          |              |
///                                     Blackout        Blackout
///                                          |              |
///                                     Completed     PostCopy -> Completed
///
/// Any state -> Failed (on error)
/// Any state -> Cancelled (on user cancel or R15 failure detection abort)
#[derive(Debug, Clone)]
pub enum LiveMigrationState {
    /// Migration has been requested but not started.
    Requested {
        /// When the migration was requested.
        requested_at: Instant,
    },

    /// Setting up destination node: reserving VRAM, creating context shell,
    /// establishing transfer channels.
    Preparing {
        /// When preparation started.
        started_at: Instant,
    },

    /// Iterative pre-copy phase. Copying pages while workload runs.
    PreCopy {
        /// When pre-copy started.
        started_at: Instant,
        /// Current iteration number (0-indexed).
        round: u32,
        /// Snapshot of dirty bitmap state at the start of this round.
        dirty_pages_this_round: u64,
        /// Total pages that have been successfully transferred.
        pages_transferred: u64,
        /// Total pages in the workload.
        total_pages: u64,
        /// Dirty rate in pages/second (smoothed).
        dirty_rate_pps: f64,
        /// Transfer rate in pages/second (smoothed).
        transfer_rate_pps: f64,
    },

    /// The workload is paused. Final dirty pages and CUDA context are being
    /// transferred. This is the blackout window.
    Blackout {
        /// When the blackout started.
        started_at: Instant,
        /// How many final dirty pages need to be transferred.
        remaining_dirty: u64,
        /// Whether CUDA context state has been serialized.
        context_serialized: bool,
        /// Whether CUDA context state has been restored on destination.
        context_restored: bool,
    },

    /// Post-copy phase: workload resumed on destination, source serving
    /// page faults for pages not yet transferred.
    PostCopy {
        /// When post-copy started.
        started_at: Instant,
        /// Pages that have been demand-faulted from source.
        pages_faulted: u64,
        /// Pages proactively pushed in background.
        pages_pushed: u64,
        /// Pages remaining on source.
        pages_remaining: u64,
        /// Total pages.
        total_pages: u64,
    },

    /// Migration completed successfully. All pages on destination.
    Completed {
        /// When the migration started (overall).
        started_at: Instant,
        /// When the migration completed.
        completed_at: Instant,
        /// Final statistics.
        stats: MigrationStats,
    },

    /// Migration failed. Workload still running on source (or lost if
    /// failure happened during blackout with post-copy).
    Failed {
        /// When the migration started.
        started_at: Instant,
        /// When the failure occurred.
        failed_at: Instant,
        /// What went wrong.
        error: LiveMigrationError,
        /// Whether the workload is still running (source alive).
        workload_intact: bool,
    },

    /// Migration was cancelled (user request, or R15 detected failure).
    Cancelled {
        /// When the migration started.
        started_at: Instant,
        /// When it was cancelled.
        cancelled_at: Instant,
        /// Reason for cancellation.
        reason: CancellationReason,
    },
}

impl LiveMigrationState {
    /// Returns true if the migration is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            LiveMigrationState::Completed { .. }
                | LiveMigrationState::Failed { .. }
                | LiveMigrationState::Cancelled { .. }
        )
    }

    /// Returns true if the migration is in flight (not terminal).
    ///
    /// Includes `Requested` because a requested migration is consuming
    /// resources (slot in the migration table, pending preparations).
    pub fn is_active(&self) -> bool {
        !self.is_terminal()
    }

    /// Returns the state name as a string.
    pub fn name(&self) -> &'static str {
        match self {
            LiveMigrationState::Requested { .. } => "Requested",
            LiveMigrationState::Preparing { .. } => "Preparing",
            LiveMigrationState::PreCopy { .. } => "PreCopy",
            LiveMigrationState::Blackout { .. } => "Blackout",
            LiveMigrationState::PostCopy { .. } => "PostCopy",
            LiveMigrationState::Completed { .. } => "Completed",
            LiveMigrationState::Failed { .. } => "Failed",
            LiveMigrationState::Cancelled { .. } => "Cancelled",
        }
    }
}

// ---------------------------------------------------------------------------
// Migration Statistics
// ---------------------------------------------------------------------------

/// Final statistics for a completed migration.
#[derive(Debug, Clone)]
pub struct MigrationStats {
    /// Total wall-clock time from Requested to Completed.
    pub total_duration: Duration,
    /// Duration of the blackout (pause) window.
    pub blackout_duration: Duration,
    /// Duration of pre-copy phase (0 if pure post-copy).
    pub precopy_duration: Duration,
    /// Duration of post-copy phase (0 if pure pre-copy).
    pub postcopy_duration: Duration,
    /// Number of pre-copy iterations completed.
    pub precopy_rounds: u32,
    /// Total bytes transferred (including re-transfers of dirty pages).
    pub total_bytes_transferred: u64,
    /// Total unique pages transferred (each page counted once).
    pub unique_pages_transferred: u64,
    /// Pages that were transferred more than once (dirty retransmissions).
    pub pages_retransmitted: u64,
    /// Pages served via post-copy demand faults.
    pub pages_demand_faulted: u64,
    /// Pages served from R15 checkpoint (avoided transfer).
    pub pages_from_checkpoint: u64,
    /// Average transfer throughput in MB/s.
    pub avg_throughput_mbps: f64,
    /// Compression ratio (1.0 = no compression).
    pub compression_ratio: f64,
    /// Number of pages that were verified via checksum.
    pub pages_verified: u64,
}

impl Default for MigrationStats {
    fn default() -> Self {
        Self {
            total_duration: Duration::ZERO,
            blackout_duration: Duration::ZERO,
            precopy_duration: Duration::ZERO,
            postcopy_duration: Duration::ZERO,
            precopy_rounds: 0,
            total_bytes_transferred: 0,
            unique_pages_transferred: 0,
            pages_retransmitted: 0,
            pages_demand_faulted: 0,
            pages_from_checkpoint: 0,
            avg_throughput_mbps: 0.0,
            compression_ratio: 1.0,
            pages_verified: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Migration Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during live migration.
#[derive(Debug, Clone)]
pub enum LiveMigrationError {
    /// Destination node does not have enough VRAM.
    InsufficientVram {
        /// VRAM required in bytes.
        required: usize,
        /// VRAM available in bytes.
        available: usize,
    },
    /// Destination GPU is incompatible (different compute capability).
    IncompatibleGpu {
        /// Source compute capability (major, minor).
        source_cc: (u32, u32),
        /// Destination compute capability (major, minor).
        dest_cc: (u32, u32),
    },
    /// Blackout duration exceeded max_blackout.
    BlackoutTimeout {
        /// Time elapsed during blackout.
        elapsed: Duration,
        /// Configured limit.
        limit: Duration,
    },
    /// Total migration duration exceeded max_total_duration.
    TotalTimeout {
        /// Time elapsed.
        elapsed: Duration,
        /// Configured limit.
        limit: Duration,
    },
    /// Network error during page transfer.
    TransferFailed {
        /// Description of the failure.
        reason: String,
    },
    /// CUDA context serialization failed.
    ContextSerializationFailed {
        /// Description of the failure.
        reason: String,
    },
    /// CUDA context restore failed on destination.
    ContextRestoreFailed {
        /// Description of the failure.
        reason: String,
    },
    /// Source node failed during migration (R15 detected).
    SourceNodeFailed {
        /// Cluster generation at time of failure.
        generation: u64,
    },
    /// Destination node failed during migration.
    DestinationNodeFailed {
        /// Cluster generation at time of failure.
        generation: u64,
    },
    /// Pre-copy did not converge and post-copy is disabled.
    PreCopyDidNotConverge {
        /// Final dirty ratio.
        dirty_ratio: f64,
        /// Number of pre-copy rounds completed.
        rounds_completed: u32,
    },
    /// Workload has pinned memory that cannot be migrated.
    PinnedMemoryConflict {
        /// Number of pinned pages.
        pinned_pages: u64,
    },
    /// Internal error.
    Internal {
        /// Description.
        reason: String,
    },
}

/// Reasons for migration cancellation.
#[derive(Debug, Clone)]
pub enum CancellationReason {
    /// User requested cancellation via CLI.
    UserRequested,
    /// R15 detected failure on source -- switching to crash recovery.
    SourceFailureDetected {
        /// Cluster generation at time of detection.
        generation: u64,
    },
    /// R15 detected failure on destination -- aborting migration.
    DestinationFailureDetected {
        /// Cluster generation at time of detection.
        generation: u64,
    },
    /// Higher-priority migration preempted this one.
    Preempted {
        /// The migration that preempted this one.
        by_migration: LiveMigrationId,
    },
    /// System shutting down.
    Shutdown,
}

// ---------------------------------------------------------------------------
// Dirty Bitmap Tracker
// ---------------------------------------------------------------------------

/// Tracks which VRAM pages have been modified since the last scan.
///
/// Uses a bitmap where each bit represents one page (64 KB = R10 page size).
/// The bitmap is updated by the CUDA interception layer: every intercepted
/// cuMemcpy/cuLaunchKernel that writes to device memory marks the
/// corresponding pages as dirty.
///
/// For a 24 GB GPU (RTX 3090): 24 GB / 64 KB = 393,216 pages = 48 KB bitmap.
/// This is tiny and fits in L1 cache.
pub struct DirtyBitmap {
    /// The bitmap itself. Bit N corresponds to VPN base_vpn + N.
    bits: Vec<u64>,
    /// Base VPN for this bitmap (first page tracked).
    base_vpn: u64,
    /// Number of pages tracked.
    num_pages: u64,
    /// Page size in bytes.
    page_size: usize,
    /// Generation counter: incremented each time the bitmap is scanned
    /// and reset. Allows detecting races between dirty marking and scanning.
    generation: u64,
    /// Count of dirty pages (maintained incrementally for O(1) query).
    dirty_count: u64,
}

impl DirtyBitmap {
    /// Create a new bitmap tracking `num_pages` starting at `base_vpn`.
    pub fn new(base_vpn: u64, num_pages: u64, page_size: usize) -> Self {
        let num_words = ((num_pages + 63) / 64) as usize;
        Self {
            bits: vec![0u64; num_words],
            base_vpn,
            num_pages,
            page_size,
            generation: 0,
            dirty_count: 0,
        }
    }

    /// Mark a page as dirty. Called by the CUDA interception layer.
    /// Returns true if the page was previously clean (new dirty).
    #[inline]
    pub fn mark_dirty(&mut self, vpn: u64) -> bool {
        let idx = (vpn - self.base_vpn) as usize;
        let word = idx / 64;
        let bit = idx % 64;
        let was_clean = (self.bits[word] & (1u64 << bit)) == 0;
        self.bits[word] |= 1u64 << bit;
        if was_clean {
            self.dirty_count += 1;
        }
        was_clean
    }

    /// Mark a range of pages as dirty (e.g., for a large cuMemcpy).
    pub fn mark_range_dirty(&mut self, start_vpn: u64, count: u64) {
        for vpn in start_vpn..(start_vpn + count) {
            self.mark_dirty(vpn);
        }
    }

    /// Check if a page is dirty.
    #[inline]
    pub fn is_dirty(&self, vpn: u64) -> bool {
        let idx = (vpn - self.base_vpn) as usize;
        let word = idx / 64;
        let bit = idx % 64;
        (self.bits[word] & (1u64 << bit)) != 0
    }

    /// Scan and reset: atomically return the current dirty set and clear
    /// the bitmap. Returns a list of dirty VPNs.
    ///
    /// This is the core operation for pre-copy iteration: "give me everything
    /// that changed since last scan, and start tracking fresh."
    pub fn scan_and_reset(&mut self) -> Vec<u64> {
        let mut dirty_vpns = Vec::with_capacity(self.dirty_count as usize);
        for (word_idx, word) in self.bits.iter_mut().enumerate() {
            if *word == 0 {
                continue;
            }
            let mut w = *word;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                let vpn = self.base_vpn + (word_idx * 64 + bit) as u64;
                dirty_vpns.push(vpn);
                w &= w - 1; // Clear lowest set bit
            }
            *word = 0;
        }
        self.generation += 1;
        self.dirty_count = 0;
        dirty_vpns
    }

    /// Return the current dirty page count without scanning.
    pub fn dirty_count(&self) -> u64 {
        self.dirty_count
    }

    /// Return the dirty ratio (dirty_count / total_pages).
    pub fn dirty_ratio(&self) -> f64 {
        if self.num_pages == 0 {
            return 0.0;
        }
        self.dirty_count as f64 / self.num_pages as f64
    }

    /// Mark ALL pages as dirty (used for initial round of pre-copy).
    pub fn mark_all_dirty(&mut self) {
        for word in self.bits.iter_mut() {
            *word = u64::MAX;
        }
        // Fix the last word if num_pages is not a multiple of 64
        let remainder = (self.num_pages % 64) as u32;
        if remainder > 0 {
            let last_idx = self.bits.len() - 1;
            self.bits[last_idx] = (1u64 << remainder) - 1;
        }
        self.dirty_count = self.num_pages;
    }

    /// Return the current generation counter.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Return the number of pages tracked.
    pub fn num_pages(&self) -> u64 {
        self.num_pages
    }

    /// Return the base VPN.
    pub fn base_vpn(&self) -> u64 {
        self.base_vpn
    }

    /// Return the page size in bytes.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Clear all dirty bits without returning the dirty set.
    pub fn clear(&mut self) {
        for word in self.bits.iter_mut() {
            *word = 0;
        }
        self.generation += 1;
        self.dirty_count = 0;
    }

    /// Check if a specific VPN is within the tracked range.
    pub fn contains(&self, vpn: u64) -> bool {
        vpn >= self.base_vpn && vpn < self.base_vpn + self.num_pages
    }
}

// ---------------------------------------------------------------------------
// CUDA Context Snapshot
// ---------------------------------------------------------------------------

/// Serialized state of a CUDA context, captured during the blackout window.
///
/// This represents everything needed to reconstruct a running CUDA context
/// on the destination GPU. We capture this at the driver API level through
/// our interception layer -- we already track all of this state because
/// we intercept every cuCtx*, cuStream*, cuEvent*, cuModule*, cuFunc* call.
///
/// Note: we do NOT snapshot running kernel register state. Instead, we
/// wait for all in-flight kernels to complete (cuCtxSynchronize) before
/// capturing the context. This is the key simplification that makes
/// OuterLink's approach tractable.
#[derive(Debug, Clone)]
pub struct CudaContextSnapshot {
    /// Snapshot identifier.
    pub id: u64,
    /// Source node ID.
    pub source_node: NodeId,
    /// Source GPU index on that node.
    pub source_gpu_index: u32,
    /// CUDA compute capability of source GPU.
    pub compute_capability: (u32, u32),
    /// When the snapshot was taken.
    pub captured_at: Instant,
    /// Total VRAM used by this context.
    pub vram_used_bytes: usize,

    // --- Allocation State ---
    /// All active device memory allocations.
    pub allocations: Vec<AllocationRecord>,

    // --- Stream State ---
    /// Active CUDA streams. Stream 0 (default) is implicit.
    pub streams: Vec<StreamRecord>,

    // --- Event State ---
    /// Active CUDA events.
    pub events: Vec<EventRecord>,

    // --- Module/Function State ---
    /// Loaded CUDA modules (PTX/cubin).
    pub modules: Vec<ModuleRecord>,

    // --- Context Flags ---
    /// Flags passed to cuCtxCreate.
    pub ctx_flags: u32,
    /// Shared memory bank size configuration.
    pub shared_mem_config: u32,
    /// Cache configuration.
    pub cache_config: u32,
    /// Memory pool configuration (CUDA 11.2+ memory pools).
    pub mem_pool_config: Option<MemPoolRecord>,
}

/// Record of a single device memory allocation.
#[derive(Debug, Clone)]
pub struct AllocationRecord {
    /// Virtual page number in OuterLink's address space.
    pub vpn: u64,
    /// Size in bytes of this allocation.
    pub size: usize,
    /// The device pointer returned to the application.
    pub device_ptr: u64,
    /// Whether this is pinned (cuMemAllocHost / cuMemHostRegister).
    pub is_pinned: bool,
    /// The CUDA allocation flags.
    pub alloc_flags: u32,
    /// R10 tier where this allocation currently lives.
    pub current_tier: TierId,
    /// R10 node where this allocation currently lives.
    pub current_node: NodeId,
    /// Data class from R15 (for parity-aware migration).
    pub data_class: DataClass,
}

/// Record of a CUDA stream.
#[derive(Debug, Clone)]
pub struct StreamRecord {
    /// Stream handle as seen by the application.
    pub app_handle: u64,
    /// Our internal stream ID.
    pub internal_id: u32,
    /// Stream priority.
    pub priority: i32,
    /// Stream flags (e.g., CU_STREAM_NON_BLOCKING).
    pub flags: u32,
    /// Whether this is the default (null) stream.
    pub is_default: bool,
}

/// Record of a CUDA event.
#[derive(Debug, Clone)]
pub struct EventRecord {
    /// Event handle as seen by the application.
    pub app_handle: u64,
    /// Our internal event ID.
    pub internal_id: u32,
    /// Event flags (e.g., CU_EVENT_DISABLE_TIMING).
    pub flags: u32,
    /// Whether this event has been recorded (cuEventRecord).
    pub is_recorded: bool,
    /// Which stream this event was recorded on (if recorded).
    pub recorded_on_stream: Option<u32>,
    /// Whether this event has completed.
    pub is_completed: bool,
}

/// Record of a loaded CUDA module.
#[derive(Debug, Clone)]
pub struct ModuleRecord {
    /// Module handle as seen by the application.
    pub app_handle: u64,
    /// Our internal module ID.
    pub internal_id: u32,
    /// The PTX or cubin binary data.
    pub binary_data: Vec<u8>,
    /// Whether this was loaded from PTX (needs JIT) or cubin (precompiled).
    pub is_ptx: bool,
    /// Functions defined in this module.
    pub functions: Vec<FunctionRecord>,
    /// Global variables defined in this module.
    pub globals: Vec<GlobalRecord>,
}

/// Record of a CUDA function (kernel) within a module.
#[derive(Debug, Clone)]
pub struct FunctionRecord {
    /// Function handle as seen by the application.
    pub app_handle: u64,
    /// Kernel name (from cuModuleGetFunction).
    pub name: String,
    /// Configured shared memory size.
    pub shared_mem_bytes: u32,
    /// Max dynamic shared memory.
    pub max_dynamic_shared_mem: u32,
    /// Cache preference.
    pub cache_config: u32,
}

/// Record of a global variable in a CUDA module.
#[derive(Debug, Clone)]
pub struct GlobalRecord {
    /// Variable name.
    pub name: String,
    /// Device pointer.
    pub device_ptr: u64,
    /// Size in bytes.
    pub size: usize,
}

/// Record of a CUDA memory pool.
#[derive(Debug, Clone)]
pub struct MemPoolRecord {
    /// Pool handle.
    pub app_handle: u64,
    /// Maximum pool size.
    pub max_size: usize,
    /// Current allocated size.
    pub used_size: usize,
    /// Release threshold.
    pub release_threshold: usize,
}

/// Data class from R15, determines how a page is treated during migration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataClass {
    /// Model weights -- immutable, can be skipped if already on destination.
    Critical,
    /// KV cache, activations -- mutable, must be transferred.
    Recoverable,
    /// Temporary buffers -- can be zeroed on destination instead of transferred.
    Transient,
    /// Optimizer state -- protected by R15 checkpoints.
    Checkpoint,
}

// ---------------------------------------------------------------------------
// Migration Transfer Protocol
// ---------------------------------------------------------------------------

/// A batch of pages being transferred from source to destination.
///
/// Pages are batched for network efficiency. Each batch is a single
/// RDMA write (or TCP send) containing multiple page payloads.
#[derive(Debug)]
pub struct PageTransferBatch {
    /// Migration this batch belongs to.
    pub migration_id: LiveMigrationId,
    /// Sequence number within this migration (for ordering).
    pub sequence: u64,
    /// Pages in this batch.
    pub pages: Vec<PagePayload>,
    /// Whether pages are compressed.
    pub compressed: bool,
    /// Compression algorithm used (if compressed).
    pub compression_algo: Option<CompressionAlgo>,
    /// Total uncompressed size.
    pub uncompressed_size: usize,
    /// Total wire size (after compression).
    pub wire_size: usize,
    /// CRC32 of the batch (for integrity).
    pub checksum: u32,
}

/// A single page's data within a transfer batch.
#[derive(Debug)]
pub struct PagePayload {
    /// Virtual page number.
    pub vpn: u64,
    /// Page data (64 KB uncompressed, variable if compressed).
    pub data: Vec<u8>,
    /// Whether this is a full page or a delta from a previous version.
    pub is_delta: bool,
}

/// Compression algorithms available for migration transfers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionAlgo {
    /// LZ4: fast, ~2 GB/s encode, modest ratio.
    Lz4,
    /// ZSTD level 1: slightly slower, better ratio.
    Zstd1,
    /// No compression (passthrough).
    None,
}

/// Acknowledgement from destination for a received batch.
#[derive(Debug)]
pub struct TransferAck {
    /// Migration this ack belongs to.
    pub migration_id: LiveMigrationId,
    /// Sequence number being acknowledged.
    pub sequence: u64,
    /// Whether the batch was received and applied successfully.
    pub success: bool,
    /// If verification enabled, checksum match result.
    pub checksum_valid: Option<bool>,
    /// Destination's current memory pressure after applying this batch.
    pub dest_pressure: PressureLevel,
}

// ---------------------------------------------------------------------------
// Post-Copy Page Fault Extension
// ---------------------------------------------------------------------------

/// Extension to R19's FaultHandler for post-copy migration page faults.
///
/// During post-copy, the destination node may access pages that haven't
/// been transferred yet. The fault handler recognizes these as migration
/// faults and routes them to the migration source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationFaultType {
    /// Page is still on source, needs to be fetched.
    PostCopyFetch {
        /// Source node to fetch from.
        source_node: NodeId,
        /// VPN of the faulting page.
        vpn: u64,
    },
    /// Page is in-flight (currently being pushed by background thread).
    /// Wait for it to arrive instead of issuing a duplicate fetch.
    InFlightWait {
        /// VPN being waited on.
        vpn: u64,
    },
}

/// Per-page migration state, used by post-copy to know where each page is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PageMigrationState {
    /// Page has been transferred to destination.
    Transferred = 0,
    /// Page is still on source, not yet transferred.
    OnSource = 1,
    /// Page is currently being transferred (in-flight).
    InFlight = 2,
    /// Page was skipped (Transient data class, zeroed on destination).
    Skipped = 3,
    /// Page was served from R15 checkpoint data.
    FromCheckpoint = 4,
}

// ---------------------------------------------------------------------------
// Page Migration Tracker
// ---------------------------------------------------------------------------

/// Tracks the per-page migration state for a live migration.
/// Used during post-copy to route page faults correctly.
pub struct PageMigrationTracker {
    /// Per-page state. Index = VPN - base_vpn.
    states: Vec<PageMigrationState>,
    /// Base VPN (same as DirtyBitmap).
    base_vpn: u64,
    /// Number of pages.
    num_pages: u64,
    /// Count of pages in each state (for fast progress reporting).
    counts: [u64; 5], // Transferred, OnSource, InFlight, Skipped, FromCheckpoint
}

impl PageMigrationTracker {
    /// Create a new tracker with all pages in OnSource state.
    pub fn new(base_vpn: u64, num_pages: u64) -> Self {
        Self {
            states: vec![PageMigrationState::OnSource; num_pages as usize],
            base_vpn,
            num_pages,
            counts: [0, num_pages, 0, 0, 0],
        }
    }

    /// Transition a page from one state to another.
    pub fn transition(&mut self, vpn: u64, new_state: PageMigrationState) {
        let idx = (vpn - self.base_vpn) as usize;
        let old_state = self.states[idx];
        self.counts[old_state as usize] -= 1;
        self.counts[new_state as usize] += 1;
        self.states[idx] = new_state;
    }

    /// Get the state of a page.
    pub fn get(&self, vpn: u64) -> PageMigrationState {
        let idx = (vpn - self.base_vpn) as usize;
        self.states[idx]
    }

    /// How many pages still need to be transferred.
    pub fn remaining(&self) -> u64 {
        self.counts[PageMigrationState::OnSource as usize]
            + self.counts[PageMigrationState::InFlight as usize]
    }

    /// Whether all pages have been transferred (migration complete).
    pub fn is_complete(&self) -> bool {
        self.remaining() == 0
    }

    /// Progress as a fraction [0.0, 1.0].
    pub fn progress(&self) -> f64 {
        if self.num_pages == 0 {
            return 1.0;
        }
        1.0 - (self.remaining() as f64 / self.num_pages as f64)
    }

    /// Get the count of pages in a specific state.
    pub fn count(&self, state: PageMigrationState) -> u64 {
        self.counts[state as usize]
    }

    /// Get the base VPN.
    pub fn base_vpn(&self) -> u64 {
        self.base_vpn
    }

    /// Get the number of pages.
    pub fn num_pages(&self) -> u64 {
        self.num_pages
    }

    /// Get all VPNs in a specific state.
    pub fn get_all(&self, state: PageMigrationState) -> Vec<u64> {
        self.states
            .iter()
            .enumerate()
            .filter(|(_, s)| **s == state)
            .map(|(i, _)| self.base_vpn + i as u64)
            .collect()
    }

    /// Check if a VPN is within the tracked range.
    pub fn contains(&self, vpn: u64) -> bool {
        vpn >= self.base_vpn && vpn < self.base_vpn + self.num_pages
    }
}

// ---------------------------------------------------------------------------
// Convergence Tracking
// ---------------------------------------------------------------------------

/// One data point per pre-copy round, tracking convergence.
#[derive(Debug, Clone)]
pub struct ConvergenceDataPoint {
    /// Pre-copy round number.
    pub round: u32,
    /// Number of dirty pages at the start of this round.
    pub dirty_pages: u64,
    /// Dirty ratio (dirty_pages / total_pages).
    pub dirty_ratio: f64,
    /// Time taken to transfer dirty pages in this round.
    pub transfer_time: Duration,
    /// Bytes transferred in this round.
    pub transfer_bytes: u64,
}

/// Result of convergence analysis after a pre-copy round.
#[derive(Debug, Clone)]
pub enum ConvergenceResult {
    /// Dirty set is small enough to finish in one blackout.
    Converged {
        /// Number of final dirty pages.
        final_dirty_count: u64,
        /// Estimated blackout duration.
        estimated_blackout: Duration,
    },
    /// Dirty set is not converging, switch to post-copy.
    SwitchToPostCopy {
        /// Pages already transferred during pre-copy.
        pages_already_transferred: u64,
    },
    /// Pre-copy failed and post-copy is not allowed.
    Failed {
        /// Final dirty ratio.
        dirty_ratio: f64,
    },
    /// Pre-copy timed out.
    Timeout,
    /// Maximum rounds reached without convergence.
    MaxRoundsReached,
    /// Still converging, continue pre-copy.
    Continue,
}

/// Checks if the dirty page count is diverging (getting worse) based on
/// the convergence history. Uses linear regression slope on the last 3 rounds.
///
/// Returns true if the slope is positive (dirty count increasing).
pub fn is_diverging(history: &[ConvergenceDataPoint]) -> bool {
    if history.len() < 3 {
        return false;
    }
    let last3 = &history[history.len() - 3..];
    let slope = linear_regression_slope(&[
        last3[0].dirty_pages as f64,
        last3[1].dirty_pages as f64,
        last3[2].dirty_pages as f64,
    ]);
    slope > 0.0
}

/// Simple linear regression slope for a small array of values.
/// Input values are treated as y-coordinates at x = 0, 1, 2, ...
pub fn linear_regression_slope(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let sum_x: f64 = (0..values.len()).map(|i| i as f64).sum();
    let sum_y: f64 = values.iter().sum();
    let sum_xy: f64 = values.iter().enumerate().map(|(i, y)| i as f64 * y).sum();
    let sum_x2: f64 = (0..values.len()).map(|i| (i as f64) * (i as f64)).sum();
    let denom = n * sum_x2 - sum_x * sum_x;
    if denom.abs() < f64::EPSILON {
        return 0.0;
    }
    (n * sum_xy - sum_x * sum_y) / denom
}

/// Estimate the blackout time given dirty pages, context size, and transfer rate.
///
/// Returns the estimated duration with a 20% safety margin.
pub fn estimate_blackout_time(
    dirty_pages: u64,
    page_size: usize,
    context_size_bytes: usize,
    transfer_rate_bps: f64,
) -> Duration {
    if transfer_rate_bps <= 0.0 {
        return Duration::from_secs(u64::MAX / 2);
    }
    let page_transfer_time = (dirty_pages as f64 * page_size as f64) / transfer_rate_bps;
    let context_transfer_time = context_size_bytes as f64 / transfer_rate_bps;
    let synchronize_time = 0.005; // 5ms cuCtxSynchronize estimate
    let safety_margin = 1.2;
    let total = (page_transfer_time + context_transfer_time + synchronize_time) * safety_margin;
    Duration::from_secs_f64(total)
}

/// Select a migration strategy based on workload characteristics.
///
/// Heuristic:
/// - If estimated dirty rate < 30% of network bandwidth: use PreCopy
/// - If VRAM footprint < 256 MB: use PreCopy (small enough to just copy)
/// - Otherwise: use Hybrid (default safe choice)
pub fn select_strategy(
    estimated_dirty_rate_bps: f64,
    network_bandwidth_bps: f64,
    vram_footprint_bytes: u64,
) -> MigrationStrategy {
    if estimated_dirty_rate_bps < network_bandwidth_bps * 0.3 {
        MigrationStrategy::PreCopy
    } else if vram_footprint_bytes < 256 * 1024 * 1024 {
        MigrationStrategy::PreCopy
    } else {
        MigrationStrategy::default()
    }
}

// ---------------------------------------------------------------------------
// Migration Orchestrator
// ---------------------------------------------------------------------------

/// The top-level migration controller. Manages the lifecycle of a live
/// migration operation, coordinating between source and destination nodes.
///
/// One `LiveMigrationOrchestrator` exists per active migration. It is created
/// on the coordinator node (which may be source, destination, or a third node).
///
/// Hardware-dependent subsystem references (page_table, migration_engine,
/// failure_detector, etc.) are injected at construction time in the real
/// system. This struct tracks migration-specific state that can be fully
/// exercised in tests.
pub struct LiveMigrationOrchestrator {
    /// Unique ID for this migration.
    pub id: LiveMigrationId,
    /// Configuration.
    pub config: LiveMigrationConfig,
    /// Current state.
    pub state: LiveMigrationState,

    // --- Source-side handles ---
    /// Source node ID.
    pub source_node: NodeId,
    /// Source GPU index on that node.
    pub source_gpu: u32,

    // --- Destination-side handles ---
    /// Destination node ID.
    pub dest_node: NodeId,
    /// Destination GPU index.
    pub dest_gpu: u32,

    // --- Migration-specific state ---
    /// Dirty page bitmap (source side).
    pub dirty_bitmap: DirtyBitmap,
    /// Per-page migration state tracker (used during post-copy).
    pub page_tracker: PageMigrationTracker,
    /// Captured CUDA context snapshot (populated during blackout).
    pub context_snapshot: Option<CudaContextSnapshot>,
    /// Convergence history for strategy adaptation.
    pub convergence_history: Vec<ConvergenceDataPoint>,
    /// When the migration started overall.
    pub started_at: Option<Instant>,
}

impl LiveMigrationOrchestrator {
    /// Create a new migration orchestrator.
    pub fn new(
        id: LiveMigrationId,
        config: LiveMigrationConfig,
        source_node: NodeId,
        source_gpu: u32,
        dest_node: NodeId,
        dest_gpu: u32,
        base_vpn: u64,
        num_pages: u64,
    ) -> Self {
        let page_size = config.page_size;
        Self {
            id,
            config,
            state: LiveMigrationState::Requested {
                requested_at: Instant::now(),
            },
            source_node,
            source_gpu,
            dest_node,
            dest_gpu,
            dirty_bitmap: DirtyBitmap::new(base_vpn, num_pages, page_size),
            page_tracker: PageMigrationTracker::new(base_vpn, num_pages),
            context_snapshot: None,
            convergence_history: Vec::new(),
            started_at: None,
        }
    }

    /// Transition the orchestrator to the Preparing state.
    /// Called after the migration request is accepted.
    pub fn begin_prepare(&mut self) {
        let now = Instant::now();
        self.started_at = Some(now);
        self.state = LiveMigrationState::Preparing { started_at: now };
    }

    /// Transition to the PreCopy state.
    /// Called after destination is prepared and ready to receive pages.
    pub fn begin_precopy(&mut self) {
        let now = Instant::now();
        let total = self.dirty_bitmap.num_pages();
        self.dirty_bitmap.mark_all_dirty();
        self.state = LiveMigrationState::PreCopy {
            started_at: now,
            round: 0,
            dirty_pages_this_round: total,
            pages_transferred: 0,
            total_pages: total,
            dirty_rate_pps: 0.0,
            transfer_rate_pps: 0.0,
        };
    }

    /// Record a completed pre-copy round and evaluate convergence.
    ///
    /// Returns a `ConvergenceResult` indicating whether to continue,
    /// switch to post-copy, or enter blackout.
    pub fn complete_precopy_round(
        &mut self,
        dirty_vpns: &[u64],
        transfer_time: Duration,
        transfer_bytes: u64,
        transfer_rate_pps: f64,
    ) -> ConvergenceResult {
        let total_pages = self.dirty_bitmap.num_pages();
        let dirty_ratio = if total_pages > 0 {
            dirty_vpns.len() as f64 / total_pages as f64
        } else {
            0.0
        };

        let round = self.convergence_history.len() as u32;

        let data_point = ConvergenceDataPoint {
            round,
            dirty_pages: dirty_vpns.len() as u64,
            dirty_ratio,
            transfer_time,
            transfer_bytes,
        };
        self.convergence_history.push(data_point);

        // Update pages transferred in tracker
        for &vpn in dirty_vpns {
            if self.page_tracker.contains(vpn)
                && self.page_tracker.get(vpn) == PageMigrationState::OnSource
            {
                self.page_tracker
                    .transition(vpn, PageMigrationState::Transferred);
            }
        }

        // Check convergence using threshold (primary) and blackout estimate (secondary).
        // Default context metadata estimate: 64MB covers typical cubin binaries + state.
        // If we have an actual snapshot, use its real size.
        let estimated_context_size = self
            .context_snapshot
            .as_ref()
            .map(|ctx| {
                // Rough size: each allocation record ~64B, each module record ~1KB avg
                // (actual binary data is tracked separately during transfer)
                ctx.allocations.len() * 64 + ctx.modules.len() * 1024 + 4096
            })
            .unwrap_or(64 * 1024 * 1024); // 64MB conservative default
        let transfer_rate_bps = if transfer_time.as_secs_f64() > 0.0 {
            transfer_bytes as f64 / transfer_time.as_secs_f64()
        } else {
            transfer_rate_pps * self.config.page_size as f64
        };

        let estimated_blackout = estimate_blackout_time(
            dirty_vpns.len() as u64,
            self.config.page_size,
            estimated_context_size,
            transfer_rate_bps,
        );

        // Primary convergence criterion: dirty ratio below threshold
        let dirty_ratio = if total_pages > 0 {
            dirty_vpns.len() as f64 / total_pages as f64
        } else {
            0.0
        };

        let convergence_threshold = match &self.config.strategy {
            MigrationStrategy::Hybrid {
                convergence_threshold,
                ..
            } => *convergence_threshold,
            _ => 0.02, // default 2% for PreCopy and PostCopy
        };

        // Converged if dirty ratio is below threshold AND blackout fits
        if dirty_ratio < convergence_threshold && estimated_blackout < self.config.max_blackout {
            return ConvergenceResult::Converged {
                final_dirty_count: dirty_vpns.len() as u64,
                estimated_blackout,
            };
        }

        // Secondary: even if threshold not met, converge if blackout is small enough
        if estimated_blackout < self.config.max_blackout {
            return ConvergenceResult::Converged {
                final_dirty_count: dirty_vpns.len() as u64,
                estimated_blackout,
            };
        }

        // Check for divergence
        if is_diverging(&self.convergence_history) {
            if self.config.strategy.allows_postcopy() {
                return ConvergenceResult::SwitchToPostCopy {
                    pages_already_transferred: self
                        .page_tracker
                        .count(PageMigrationState::Transferred),
                };
            } else {
                return ConvergenceResult::Failed { dirty_ratio };
            }
        }

        // Check timeout and max rounds
        match &self.config.strategy {
            MigrationStrategy::Hybrid {
                max_precopy_rounds,
                max_precopy_duration,
                ..
            } => {
                if round + 1 >= *max_precopy_rounds {
                    if self.config.strategy.allows_postcopy() {
                        return ConvergenceResult::SwitchToPostCopy {
                            pages_already_transferred: self
                                .page_tracker
                                .count(PageMigrationState::Transferred),
                        };
                    }
                    return ConvergenceResult::MaxRoundsReached;
                }
                if let Some(started) = self.started_at {
                    if started.elapsed() > *max_precopy_duration {
                        if self.config.strategy.allows_postcopy() {
                            return ConvergenceResult::SwitchToPostCopy {
                                pages_already_transferred: self
                                    .page_tracker
                                    .count(PageMigrationState::Transferred),
                            };
                        }
                        return ConvergenceResult::Timeout;
                    }
                }
            }
            MigrationStrategy::PreCopy => {
                // Pure pre-copy: fail if not converging after 8 rounds
                if round + 1 >= 8 {
                    return ConvergenceResult::MaxRoundsReached;
                }
            }
            MigrationStrategy::PostCopy => {
                // Should not be in pre-copy with PostCopy strategy
                return ConvergenceResult::SwitchToPostCopy {
                    pages_already_transferred: self
                        .page_tracker
                        .count(PageMigrationState::Transferred),
                };
            }
        }

        // Update state
        self.state = LiveMigrationState::PreCopy {
            started_at: self.started_at.unwrap_or_else(Instant::now),
            round: round + 1,
            dirty_pages_this_round: dirty_vpns.len() as u64,
            pages_transferred: self
                .page_tracker
                .count(PageMigrationState::Transferred),
            total_pages,
            dirty_rate_pps: 0.0, // TODO: compute from dirty tracking
            transfer_rate_pps,
        };

        ConvergenceResult::Continue
    }

    /// Transition to the Blackout state.
    /// Called when pre-copy converges or when starting pure post-copy.
    pub fn begin_blackout(&mut self, remaining_dirty: u64) {
        self.state = LiveMigrationState::Blackout {
            started_at: Instant::now(),
            remaining_dirty,
            context_serialized: false,
            context_restored: false,
        };
    }

    /// Mark the CUDA context as serialized during blackout.
    pub fn mark_context_serialized(&mut self) {
        if let LiveMigrationState::Blackout {
            context_serialized, ..
        } = &mut self.state
        {
            *context_serialized = true;
        }
    }

    /// Mark the CUDA context as restored on destination during blackout.
    pub fn mark_context_restored(&mut self) {
        if let LiveMigrationState::Blackout {
            context_restored, ..
        } = &mut self.state
        {
            *context_restored = true;
        }
    }

    /// Transition to the PostCopy state.
    /// Called after blackout when there are still pages on source.
    pub fn begin_postcopy(&mut self) {
        let total = self.page_tracker.num_pages();
        let remaining = self.page_tracker.remaining();
        self.state = LiveMigrationState::PostCopy {
            started_at: Instant::now(),
            pages_faulted: 0,
            pages_pushed: 0,
            pages_remaining: remaining,
            total_pages: total,
        };
    }

    /// Handle a post-copy page fault: determine the fault type based on
    /// the page's current migration state.
    pub fn handle_page_fault(&self, vpn: u64) -> Option<MigrationFaultType> {
        if !self.page_tracker.contains(vpn) {
            return None;
        }
        match self.page_tracker.get(vpn) {
            PageMigrationState::OnSource => Some(MigrationFaultType::PostCopyFetch {
                source_node: self.source_node,
                vpn,
            }),
            PageMigrationState::InFlight => Some(MigrationFaultType::InFlightWait { vpn }),
            PageMigrationState::Transferred
            | PageMigrationState::Skipped
            | PageMigrationState::FromCheckpoint => None, // page is already local
        }
    }

    /// Record that a page was fetched via demand fault during post-copy.
    pub fn record_demand_fault(&mut self, vpn: u64) {
        if self.page_tracker.contains(vpn) {
            self.page_tracker
                .transition(vpn, PageMigrationState::Transferred);
        }
        if let LiveMigrationState::PostCopy {
            pages_faulted,
            pages_remaining,
            ..
        } = &mut self.state
        {
            *pages_faulted += 1;
            *pages_remaining = self.page_tracker.remaining();
        }
    }

    /// Record that a page was proactively pushed in background during post-copy.
    pub fn record_background_push(&mut self, vpn: u64) {
        if self.page_tracker.contains(vpn) {
            self.page_tracker
                .transition(vpn, PageMigrationState::Transferred);
        }
        if let LiveMigrationState::PostCopy {
            pages_pushed,
            pages_remaining,
            ..
        } = &mut self.state
        {
            *pages_pushed += 1;
            *pages_remaining = self.page_tracker.remaining();
        }
    }

    /// Complete the migration successfully.
    pub fn complete(&mut self, stats: MigrationStats) {
        let started = self.started_at.unwrap_or_else(Instant::now);
        self.state = LiveMigrationState::Completed {
            started_at: started,
            completed_at: Instant::now(),
            stats,
        };
    }

    /// Fail the migration with an error.
    pub fn fail(&mut self, error: LiveMigrationError, workload_intact: bool) {
        let started = self.started_at.unwrap_or_else(Instant::now);
        self.state = LiveMigrationState::Failed {
            started_at: started,
            failed_at: Instant::now(),
            error,
            workload_intact,
        };
    }

    /// Cancel the migration.
    pub fn cancel(&mut self, reason: CancellationReason) {
        let started = self.started_at.unwrap_or_else(Instant::now);
        self.state = LiveMigrationState::Cancelled {
            started_at: started,
            cancelled_at: Instant::now(),
            reason,
        };
    }

    /// Check whether migration has timed out overall.
    pub fn is_timed_out(&self) -> bool {
        if let Some(started) = self.started_at {
            started.elapsed() > self.config.max_total_duration
        } else {
            false
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- DirtyBitmap tests ---

    #[test]
    fn dirty_bitmap_new_is_clean() {
        let bm = DirtyBitmap::new(1000, 128, 65536);
        assert_eq!(bm.dirty_count(), 0);
        assert_eq!(bm.dirty_ratio(), 0.0);
        assert_eq!(bm.generation(), 0);
        assert_eq!(bm.num_pages(), 128);
        assert_eq!(bm.base_vpn(), 1000);
    }

    #[test]
    fn dirty_bitmap_mark_single() {
        let mut bm = DirtyBitmap::new(0, 64, 65536);
        assert!(bm.mark_dirty(0));
        assert!(bm.is_dirty(0));
        assert!(!bm.is_dirty(1));
        assert_eq!(bm.dirty_count(), 1);
        // marking again returns false
        assert!(!bm.mark_dirty(0));
        assert_eq!(bm.dirty_count(), 1);
    }

    #[test]
    fn dirty_bitmap_mark_range() {
        let mut bm = DirtyBitmap::new(100, 200, 65536);
        bm.mark_range_dirty(100, 50);
        assert_eq!(bm.dirty_count(), 50);
        for vpn in 100..150 {
            assert!(bm.is_dirty(vpn));
        }
        assert!(!bm.is_dirty(150));
    }

    #[test]
    fn dirty_bitmap_scan_and_reset() {
        let mut bm = DirtyBitmap::new(0, 128, 65536);
        bm.mark_dirty(5);
        bm.mark_dirty(63);
        bm.mark_dirty(64);
        bm.mark_dirty(127);
        let dirty = bm.scan_and_reset();
        assert_eq!(dirty, vec![5, 63, 64, 127]);
        assert_eq!(bm.dirty_count(), 0);
        assert_eq!(bm.generation(), 1);
        assert!(!bm.is_dirty(5));
    }

    #[test]
    fn dirty_bitmap_mark_all_dirty() {
        let mut bm = DirtyBitmap::new(0, 100, 65536);
        bm.mark_all_dirty();
        assert_eq!(bm.dirty_count(), 100);
        for vpn in 0..100 {
            assert!(bm.is_dirty(vpn));
        }
    }

    #[test]
    fn dirty_bitmap_mark_all_dirty_aligned() {
        let mut bm = DirtyBitmap::new(0, 128, 65536);
        bm.mark_all_dirty();
        assert_eq!(bm.dirty_count(), 128);
        let dirty = bm.scan_and_reset();
        assert_eq!(dirty.len(), 128);
    }

    #[test]
    fn dirty_bitmap_mark_all_not_aligned() {
        // Test with num_pages not a multiple of 64
        let mut bm = DirtyBitmap::new(0, 65, 65536);
        bm.mark_all_dirty();
        assert_eq!(bm.dirty_count(), 65);
        let dirty = bm.scan_and_reset();
        assert_eq!(dirty.len(), 65);
        // Ensure no extra bits set beyond num_pages
        assert_eq!(dirty.last(), Some(&64));
    }

    #[test]
    fn dirty_bitmap_clear() {
        let mut bm = DirtyBitmap::new(0, 64, 65536);
        bm.mark_dirty(10);
        bm.mark_dirty(20);
        bm.clear();
        assert_eq!(bm.dirty_count(), 0);
        assert!(!bm.is_dirty(10));
        assert_eq!(bm.generation(), 1);
    }

    #[test]
    fn dirty_bitmap_contains() {
        let bm = DirtyBitmap::new(100, 50, 65536);
        assert!(bm.contains(100));
        assert!(bm.contains(149));
        assert!(!bm.contains(99));
        assert!(!bm.contains(150));
    }

    #[test]
    fn dirty_bitmap_dirty_ratio() {
        let mut bm = DirtyBitmap::new(0, 100, 65536);
        bm.mark_range_dirty(0, 25);
        assert!((bm.dirty_ratio() - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn dirty_bitmap_multiple_scan_and_reset() {
        let mut bm = DirtyBitmap::new(0, 64, 65536);
        bm.mark_dirty(0);
        let d1 = bm.scan_and_reset();
        assert_eq!(d1.len(), 1);
        assert_eq!(bm.generation(), 1);

        bm.mark_dirty(1);
        bm.mark_dirty(2);
        let d2 = bm.scan_and_reset();
        assert_eq!(d2.len(), 2);
        assert_eq!(bm.generation(), 2);
    }

    #[test]
    fn dirty_bitmap_empty() {
        let bm = DirtyBitmap::new(0, 0, 65536);
        assert_eq!(bm.dirty_count(), 0);
        assert_eq!(bm.dirty_ratio(), 0.0);
    }

    #[test]
    fn dirty_bitmap_large_offset() {
        let mut bm = DirtyBitmap::new(1_000_000, 256, 65536);
        bm.mark_dirty(1_000_100);
        assert!(bm.is_dirty(1_000_100));
        assert!(!bm.is_dirty(1_000_101));
        let dirty = bm.scan_and_reset();
        assert_eq!(dirty, vec![1_000_100]);
    }

    #[test]
    fn dirty_bitmap_page_size() {
        let bm = DirtyBitmap::new(0, 64, 4096);
        assert_eq!(bm.page_size(), 4096);
    }

    // --- PageMigrationTracker tests ---

    #[test]
    fn tracker_new_all_on_source() {
        let t = PageMigrationTracker::new(0, 100);
        assert_eq!(t.count(PageMigrationState::OnSource), 100);
        assert_eq!(t.count(PageMigrationState::Transferred), 0);
        assert_eq!(t.remaining(), 100);
        assert!(!t.is_complete());
        assert!((t.progress() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn tracker_transition() {
        let mut t = PageMigrationTracker::new(0, 10);
        t.transition(0, PageMigrationState::InFlight);
        assert_eq!(t.get(0), PageMigrationState::InFlight);
        assert_eq!(t.count(PageMigrationState::OnSource), 9);
        assert_eq!(t.count(PageMigrationState::InFlight), 1);
        assert_eq!(t.remaining(), 10); // InFlight still counts as remaining

        t.transition(0, PageMigrationState::Transferred);
        assert_eq!(t.get(0), PageMigrationState::Transferred);
        assert_eq!(t.count(PageMigrationState::Transferred), 1);
        assert_eq!(t.remaining(), 9);
    }

    #[test]
    fn tracker_complete_all_transferred() {
        let mut t = PageMigrationTracker::new(0, 3);
        t.transition(0, PageMigrationState::Transferred);
        t.transition(1, PageMigrationState::Skipped);
        t.transition(2, PageMigrationState::FromCheckpoint);
        assert!(t.is_complete());
        assert!((t.progress() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn tracker_get_all() {
        let mut t = PageMigrationTracker::new(100, 5);
        t.transition(100, PageMigrationState::Transferred);
        t.transition(102, PageMigrationState::Transferred);
        let transferred = t.get_all(PageMigrationState::Transferred);
        assert_eq!(transferred, vec![100, 102]);
    }

    #[test]
    fn tracker_progress() {
        let mut t = PageMigrationTracker::new(0, 4);
        t.transition(0, PageMigrationState::Transferred);
        t.transition(1, PageMigrationState::Skipped);
        assert!((t.progress() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn tracker_contains() {
        let t = PageMigrationTracker::new(50, 10);
        assert!(t.contains(50));
        assert!(t.contains(59));
        assert!(!t.contains(49));
        assert!(!t.contains(60));
    }

    #[test]
    fn tracker_empty() {
        let t = PageMigrationTracker::new(0, 0);
        assert!(t.is_complete());
        assert!((t.progress() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn tracker_base_vpn_and_num_pages() {
        let t = PageMigrationTracker::new(42, 100);
        assert_eq!(t.base_vpn(), 42);
        assert_eq!(t.num_pages(), 100);
    }

    // --- MigrationStrategy tests ---

    #[test]
    fn strategy_default_is_hybrid() {
        let s = MigrationStrategy::default();
        assert!(matches!(s, MigrationStrategy::Hybrid { .. }));
    }

    #[test]
    fn strategy_allows_postcopy() {
        assert!(!MigrationStrategy::PreCopy.allows_postcopy());
        assert!(MigrationStrategy::PostCopy.allows_postcopy());
        assert!(MigrationStrategy::default().allows_postcopy());
    }

    #[test]
    fn strategy_hybrid_defaults() {
        if let MigrationStrategy::Hybrid {
            max_precopy_rounds,
            max_precopy_duration,
            convergence_threshold,
        } = MigrationStrategy::default()
        {
            assert_eq!(max_precopy_rounds, 8);
            assert_eq!(max_precopy_duration, Duration::from_secs(30));
            assert!((convergence_threshold - 0.02).abs() < f64::EPSILON);
        } else {
            panic!("expected Hybrid");
        }
    }

    // --- LiveMigrationConfig tests ---

    #[test]
    fn config_default() {
        let c = LiveMigrationConfig::default();
        assert_eq!(c.max_blackout, Duration::from_millis(500));
        assert_eq!(c.max_total_duration, Duration::from_secs(120));
        assert!(c.compress_transfers);
        assert!(c.use_checkpoint_bootstrap);
        assert_eq!(c.priority, LiveMigrationPriority::Normal);
        assert_eq!(c.bandwidth_limit, 0);
        assert_eq!(c.page_size, 65536);
        assert!(!c.verify_checksums);
    }

    // --- LiveMigrationState tests ---

    #[test]
    fn state_is_terminal() {
        let now = Instant::now();
        assert!(!LiveMigrationState::Requested { requested_at: now }.is_terminal());
        assert!(!LiveMigrationState::Preparing { started_at: now }.is_terminal());
        assert!(LiveMigrationState::Completed {
            started_at: now,
            completed_at: now,
            stats: MigrationStats::default(),
        }
        .is_terminal());
        assert!(LiveMigrationState::Failed {
            started_at: now,
            failed_at: now,
            error: LiveMigrationError::Internal {
                reason: "test".into()
            },
            workload_intact: true,
        }
        .is_terminal());
        assert!(LiveMigrationState::Cancelled {
            started_at: now,
            cancelled_at: now,
            reason: CancellationReason::UserRequested,
        }
        .is_terminal());
    }

    #[test]
    fn state_is_active() {
        let now = Instant::now();
        // Requested is active (not terminal) — migration is in flight
        assert!(LiveMigrationState::Requested { requested_at: now }.is_active());
        assert!(LiveMigrationState::Preparing { started_at: now }.is_active());
        assert!(LiveMigrationState::PreCopy {
            started_at: now,
            round: 0,
            dirty_pages_this_round: 0,
            pages_transferred: 0,
            total_pages: 100,
            dirty_rate_pps: 0.0,
            transfer_rate_pps: 0.0,
        }
        .is_active());
    }

    #[test]
    fn state_names() {
        let now = Instant::now();
        assert_eq!(
            LiveMigrationState::Requested { requested_at: now }.name(),
            "Requested"
        );
        assert_eq!(
            LiveMigrationState::Preparing { started_at: now }.name(),
            "Preparing"
        );
    }

    // --- Convergence tests ---

    #[test]
    fn linear_regression_slope_flat() {
        let slope = linear_regression_slope(&[10.0, 10.0, 10.0]);
        assert!(slope.abs() < 1e-10);
    }

    #[test]
    fn linear_regression_slope_increasing() {
        let slope = linear_regression_slope(&[10.0, 20.0, 30.0]);
        assert!((slope - 10.0).abs() < 1e-10);
    }

    #[test]
    fn linear_regression_slope_decreasing() {
        let slope = linear_regression_slope(&[30.0, 20.0, 10.0]);
        assert!((slope - (-10.0)).abs() < 1e-10);
    }

    #[test]
    fn linear_regression_slope_single() {
        let slope = linear_regression_slope(&[42.0]);
        assert_eq!(slope, 0.0);
    }

    #[test]
    fn linear_regression_slope_empty() {
        let slope = linear_regression_slope(&[]);
        assert_eq!(slope, 0.0);
    }

    #[test]
    fn is_diverging_too_few_points() {
        let history = vec![
            ConvergenceDataPoint {
                round: 0,
                dirty_pages: 100,
                dirty_ratio: 0.1,
                transfer_time: Duration::from_millis(100),
                transfer_bytes: 1000,
            },
        ];
        assert!(!is_diverging(&history));
    }

    #[test]
    fn is_diverging_true_increasing() {
        let history = vec![
            ConvergenceDataPoint {
                round: 0,
                dirty_pages: 100,
                dirty_ratio: 0.1,
                transfer_time: Duration::from_millis(100),
                transfer_bytes: 1000,
            },
            ConvergenceDataPoint {
                round: 1,
                dirty_pages: 200,
                dirty_ratio: 0.2,
                transfer_time: Duration::from_millis(100),
                transfer_bytes: 1000,
            },
            ConvergenceDataPoint {
                round: 2,
                dirty_pages: 300,
                dirty_ratio: 0.3,
                transfer_time: Duration::from_millis(100),
                transfer_bytes: 1000,
            },
        ];
        assert!(is_diverging(&history));
    }

    #[test]
    fn is_diverging_false_decreasing() {
        let history = vec![
            ConvergenceDataPoint {
                round: 0,
                dirty_pages: 300,
                dirty_ratio: 0.3,
                transfer_time: Duration::from_millis(100),
                transfer_bytes: 1000,
            },
            ConvergenceDataPoint {
                round: 1,
                dirty_pages: 200,
                dirty_ratio: 0.2,
                transfer_time: Duration::from_millis(100),
                transfer_bytes: 1000,
            },
            ConvergenceDataPoint {
                round: 2,
                dirty_pages: 100,
                dirty_ratio: 0.1,
                transfer_time: Duration::from_millis(100),
                transfer_bytes: 1000,
            },
        ];
        assert!(!is_diverging(&history));
    }

    // --- estimate_blackout_time tests ---

    #[test]
    fn estimate_blackout_basic() {
        // 10 pages of 64KB at 1 GB/s -> 640KB / 1e9 = 640us page transfer
        // context_size 4096 bytes -> ~4us
        // sync 5ms
        // total ~5.644ms * 1.2 = ~6.77ms
        let dur = estimate_blackout_time(10, 65536, 4096, 1e9);
        assert!(dur.as_millis() > 5 && dur.as_millis() < 20);
    }

    #[test]
    fn estimate_blackout_zero_rate() {
        let dur = estimate_blackout_time(10, 65536, 4096, 0.0);
        assert!(dur > Duration::from_secs(1_000_000));
    }

    // --- select_strategy tests ---

    #[test]
    fn select_strategy_low_dirty_rate() {
        let s = select_strategy(100.0, 1000.0, 1024 * 1024 * 1024);
        assert!(matches!(s, MigrationStrategy::PreCopy));
    }

    #[test]
    fn select_strategy_small_footprint() {
        let s = select_strategy(1e9, 1e9, 128 * 1024 * 1024); // 128 MB
        assert!(matches!(s, MigrationStrategy::PreCopy));
    }

    #[test]
    fn select_strategy_large_dirty() {
        let s = select_strategy(1e9, 1e9, 1024 * 1024 * 1024); // 1 GB
        assert!(matches!(s, MigrationStrategy::Hybrid { .. }));
    }

    // --- LiveMigrationPriority tests ---

    #[test]
    fn priority_ordering() {
        assert!(LiveMigrationPriority::Background < LiveMigrationPriority::Normal);
        assert!(LiveMigrationPriority::Normal < LiveMigrationPriority::Urgent);
        assert!(LiveMigrationPriority::Urgent < LiveMigrationPriority::Emergency);
    }

    // --- LiveMigrationOrchestrator tests ---

    fn make_orchestrator() -> LiveMigrationOrchestrator {
        LiveMigrationOrchestrator::new(
            LiveMigrationId(1),
            LiveMigrationConfig::default(),
            0, // source_node
            0, // source_gpu
            1, // dest_node
            0, // dest_gpu
            0, // base_vpn
            100, // num_pages
        )
    }

    #[test]
    fn orchestrator_initial_state() {
        let o = make_orchestrator();
        assert!(matches!(o.state, LiveMigrationState::Requested { .. }));
        assert!(o.context_snapshot.is_none());
        assert!(o.convergence_history.is_empty());
        assert_eq!(o.dirty_bitmap.num_pages(), 100);
        assert_eq!(o.page_tracker.num_pages(), 100);
    }

    #[test]
    fn orchestrator_begin_prepare() {
        let mut o = make_orchestrator();
        o.begin_prepare();
        assert!(matches!(o.state, LiveMigrationState::Preparing { .. }));
        assert!(o.started_at.is_some());
    }

    #[test]
    fn orchestrator_begin_precopy() {
        let mut o = make_orchestrator();
        o.begin_prepare();
        o.begin_precopy();
        if let LiveMigrationState::PreCopy {
            round,
            total_pages,
            dirty_pages_this_round,
            ..
        } = &o.state
        {
            assert_eq!(*round, 0);
            assert_eq!(*total_pages, 100);
            assert_eq!(*dirty_pages_this_round, 100);
        } else {
            panic!("expected PreCopy state");
        }
        assert_eq!(o.dirty_bitmap.dirty_count(), 100);
    }

    #[test]
    fn orchestrator_begin_blackout() {
        let mut o = make_orchestrator();
        o.begin_blackout(10);
        if let LiveMigrationState::Blackout {
            remaining_dirty,
            context_serialized,
            context_restored,
            ..
        } = &o.state
        {
            assert_eq!(*remaining_dirty, 10);
            assert!(!*context_serialized);
            assert!(!*context_restored);
        } else {
            panic!("expected Blackout state");
        }
    }

    #[test]
    fn orchestrator_blackout_context_flags() {
        let mut o = make_orchestrator();
        o.begin_blackout(0);
        o.mark_context_serialized();
        if let LiveMigrationState::Blackout {
            context_serialized, ..
        } = &o.state
        {
            assert!(*context_serialized);
        }
        o.mark_context_restored();
        if let LiveMigrationState::Blackout {
            context_restored, ..
        } = &o.state
        {
            assert!(*context_restored);
        }
    }

    #[test]
    fn orchestrator_begin_postcopy() {
        let mut o = make_orchestrator();
        o.begin_postcopy();
        if let LiveMigrationState::PostCopy {
            total_pages,
            pages_remaining,
            pages_faulted,
            pages_pushed,
            ..
        } = &o.state
        {
            assert_eq!(*total_pages, 100);
            assert_eq!(*pages_remaining, 100);
            assert_eq!(*pages_faulted, 0);
            assert_eq!(*pages_pushed, 0);
        } else {
            panic!("expected PostCopy state");
        }
    }

    #[test]
    fn orchestrator_handle_page_fault_on_source() {
        let o = make_orchestrator();
        let fault = o.handle_page_fault(50);
        assert!(matches!(
            fault,
            Some(MigrationFaultType::PostCopyFetch { source_node: 0, vpn: 50 })
        ));
    }

    #[test]
    fn orchestrator_handle_page_fault_in_flight() {
        let mut o = make_orchestrator();
        o.page_tracker
            .transition(50, PageMigrationState::InFlight);
        let fault = o.handle_page_fault(50);
        assert!(matches!(fault, Some(MigrationFaultType::InFlightWait { vpn: 50 })));
    }

    #[test]
    fn orchestrator_handle_page_fault_transferred() {
        let mut o = make_orchestrator();
        o.page_tracker
            .transition(50, PageMigrationState::Transferred);
        let fault = o.handle_page_fault(50);
        assert!(fault.is_none());
    }

    #[test]
    fn orchestrator_handle_page_fault_out_of_range() {
        let o = make_orchestrator();
        let fault = o.handle_page_fault(200);
        assert!(fault.is_none());
    }

    #[test]
    fn orchestrator_record_demand_fault() {
        let mut o = make_orchestrator();
        o.begin_postcopy();
        o.record_demand_fault(5);
        assert_eq!(o.page_tracker.get(5), PageMigrationState::Transferred);
        if let LiveMigrationState::PostCopy {
            pages_faulted,
            pages_remaining,
            ..
        } = &o.state
        {
            assert_eq!(*pages_faulted, 1);
            assert_eq!(*pages_remaining, 99);
        }
    }

    #[test]
    fn orchestrator_record_background_push() {
        let mut o = make_orchestrator();
        o.begin_postcopy();
        o.record_background_push(10);
        assert_eq!(o.page_tracker.get(10), PageMigrationState::Transferred);
        if let LiveMigrationState::PostCopy {
            pages_pushed,
            pages_remaining,
            ..
        } = &o.state
        {
            assert_eq!(*pages_pushed, 1);
            assert_eq!(*pages_remaining, 99);
        }
    }

    #[test]
    fn orchestrator_complete() {
        let mut o = make_orchestrator();
        o.begin_prepare();
        o.complete(MigrationStats::default());
        assert!(o.state.is_terminal());
        assert!(matches!(o.state, LiveMigrationState::Completed { .. }));
    }

    #[test]
    fn orchestrator_fail() {
        let mut o = make_orchestrator();
        o.begin_prepare();
        o.fail(
            LiveMigrationError::TransferFailed {
                reason: "network down".into(),
            },
            true,
        );
        assert!(o.state.is_terminal());
        if let LiveMigrationState::Failed {
            workload_intact, ..
        } = &o.state
        {
            assert!(*workload_intact);
        }
    }

    #[test]
    fn orchestrator_cancel() {
        let mut o = make_orchestrator();
        o.begin_prepare();
        o.cancel(CancellationReason::UserRequested);
        assert!(o.state.is_terminal());
        assert!(matches!(o.state, LiveMigrationState::Cancelled { .. }));
    }

    #[test]
    fn orchestrator_is_timed_out_fresh() {
        let o = make_orchestrator();
        assert!(!o.is_timed_out());
    }

    // --- Convergence round tests ---

    #[test]
    fn orchestrator_convergence_converged() {
        let mut o = LiveMigrationOrchestrator::new(
            LiveMigrationId(1),
            LiveMigrationConfig {
                max_blackout: Duration::from_secs(10), // very generous
                ..Default::default()
            },
            0,
            0,
            1,
            0,
            0,
            100,
        );
        o.begin_prepare();
        o.begin_precopy();

        // Simulate a round with only 1 dirty page, should converge easily
        let result = o.complete_precopy_round(
            &[0],
            Duration::from_millis(1),
            65536,
            1000.0,
        );
        assert!(matches!(result, ConvergenceResult::Converged { .. }));
    }

    #[test]
    fn orchestrator_convergence_continue() {
        let mut o = LiveMigrationOrchestrator::new(
            LiveMigrationId(1),
            LiveMigrationConfig {
                max_blackout: Duration::from_millis(1), // very tight
                ..Default::default()
            },
            0,
            0,
            1,
            0,
            0,
            1000,
        );
        o.begin_prepare();
        o.begin_precopy();

        // Many dirty pages relative to tight blackout
        let dirty: Vec<u64> = (0..500).collect();
        let result = o.complete_precopy_round(
            &dirty,
            Duration::from_secs(1),
            500 * 65536,
            500.0,
        );
        // Should continue (not enough history to diverge, too many dirty for blackout)
        assert!(matches!(result, ConvergenceResult::Continue));
    }

    #[test]
    fn orchestrator_convergence_diverging_switches_postcopy() {
        let mut o = LiveMigrationOrchestrator::new(
            LiveMigrationId(1),
            LiveMigrationConfig {
                max_blackout: Duration::from_millis(1),
                strategy: MigrationStrategy::Hybrid {
                    max_precopy_rounds: 10,
                    max_precopy_duration: Duration::from_secs(60),
                    convergence_threshold: 0.02,
                },
                ..Default::default()
            },
            0,
            0,
            1,
            0,
            0,
            1000,
        );
        o.begin_prepare();
        o.begin_precopy();

        // Simulate 3 rounds with increasing dirty counts (diverging)
        let dirty1: Vec<u64> = (0..100).collect();
        o.complete_precopy_round(&dirty1, Duration::from_secs(1), 100 * 65536, 100.0);

        let dirty2: Vec<u64> = (0..200).collect();
        o.complete_precopy_round(&dirty2, Duration::from_secs(1), 200 * 65536, 200.0);

        let dirty3: Vec<u64> = (0..300).collect();
        let result = o.complete_precopy_round(&dirty3, Duration::from_secs(1), 300 * 65536, 300.0);
        assert!(matches!(
            result,
            ConvergenceResult::SwitchToPostCopy { .. }
        ));
    }

    #[test]
    fn orchestrator_convergence_diverging_fails_precopy_only() {
        let mut o = LiveMigrationOrchestrator::new(
            LiveMigrationId(1),
            LiveMigrationConfig {
                max_blackout: Duration::from_millis(1),
                strategy: MigrationStrategy::PreCopy,
                ..Default::default()
            },
            0,
            0,
            1,
            0,
            0,
            1000,
        );
        o.begin_prepare();
        o.begin_precopy();

        // 3 rounds diverging
        let dirty1: Vec<u64> = (0..100).collect();
        o.complete_precopy_round(&dirty1, Duration::from_secs(1), 100 * 65536, 100.0);
        let dirty2: Vec<u64> = (0..200).collect();
        o.complete_precopy_round(&dirty2, Duration::from_secs(1), 200 * 65536, 200.0);
        let dirty3: Vec<u64> = (0..300).collect();
        let result = o.complete_precopy_round(&dirty3, Duration::from_secs(1), 300 * 65536, 300.0);
        assert!(matches!(result, ConvergenceResult::Failed { .. }));
    }

    // --- PageTransferBatch tests ---

    #[test]
    fn page_transfer_batch_creation() {
        let batch = PageTransferBatch {
            migration_id: LiveMigrationId(42),
            sequence: 0,
            pages: vec![PagePayload {
                vpn: 100,
                data: vec![0u8; 65536],
                is_delta: false,
            }],
            compressed: false,
            compression_algo: None,
            uncompressed_size: 65536,
            wire_size: 65536,
            checksum: 0,
        };
        assert_eq!(batch.pages.len(), 1);
        assert_eq!(batch.migration_id, LiveMigrationId(42));
    }

    // --- TransferAck tests ---

    #[test]
    fn transfer_ack_creation() {
        let ack = TransferAck {
            migration_id: LiveMigrationId(1),
            sequence: 5,
            success: true,
            checksum_valid: Some(true),
            dest_pressure: PressureLevel::Low,
        };
        assert!(ack.success);
        assert_eq!(ack.sequence, 5);
    }

    // --- CudaContextSnapshot tests ---

    #[test]
    fn cuda_context_snapshot_creation() {
        let snap = CudaContextSnapshot {
            id: 1,
            source_node: 0,
            source_gpu_index: 0,
            compute_capability: (8, 6),
            captured_at: Instant::now(),
            vram_used_bytes: 1024 * 1024,
            allocations: vec![AllocationRecord {
                vpn: 0,
                size: 65536,
                device_ptr: 0x7F000000,
                is_pinned: false,
                alloc_flags: 0,
                current_tier: 0,
                current_node: 0,
                data_class: DataClass::Recoverable,
            }],
            streams: vec![StreamRecord {
                app_handle: 0,
                internal_id: 0,
                priority: 0,
                flags: 0,
                is_default: true,
            }],
            events: vec![],
            modules: vec![],
            ctx_flags: 0,
            shared_mem_config: 0,
            cache_config: 0,
            mem_pool_config: None,
        };
        assert_eq!(snap.allocations.len(), 1);
        assert_eq!(snap.streams.len(), 1);
        assert_eq!(snap.compute_capability, (8, 6));
    }

    #[test]
    fn data_class_variants() {
        assert_ne!(DataClass::Critical, DataClass::Transient);
        assert_eq!(DataClass::Checkpoint, DataClass::Checkpoint);
    }

    // --- CompressionAlgo tests ---

    #[test]
    fn compression_algo_variants() {
        assert_ne!(CompressionAlgo::Lz4, CompressionAlgo::Zstd1);
        assert_eq!(CompressionAlgo::None, CompressionAlgo::None);
    }

    // --- MigrationFaultType tests ---

    #[test]
    fn migration_fault_type_postcopy_fetch() {
        let f = MigrationFaultType::PostCopyFetch {
            source_node: 3,
            vpn: 42,
        };
        assert!(matches!(
            f,
            MigrationFaultType::PostCopyFetch {
                source_node: 3,
                vpn: 42
            }
        ));
    }

    #[test]
    fn migration_fault_type_inflight_wait() {
        let f = MigrationFaultType::InFlightWait { vpn: 99 };
        assert!(matches!(f, MigrationFaultType::InFlightWait { vpn: 99 }));
    }

    // --- LiveMigrationError tests ---

    #[test]
    fn error_insufficient_vram() {
        let e = LiveMigrationError::InsufficientVram {
            required: 1000,
            available: 500,
        };
        assert!(matches!(e, LiveMigrationError::InsufficientVram { .. }));
    }

    #[test]
    fn error_incompatible_gpu() {
        let e = LiveMigrationError::IncompatibleGpu {
            source_cc: (8, 6),
            dest_cc: (7, 5),
        };
        assert!(matches!(e, LiveMigrationError::IncompatibleGpu { .. }));
    }

    #[test]
    fn error_blackout_timeout() {
        let e = LiveMigrationError::BlackoutTimeout {
            elapsed: Duration::from_secs(1),
            limit: Duration::from_millis(500),
        };
        assert!(matches!(e, LiveMigrationError::BlackoutTimeout { .. }));
    }

    // --- CancellationReason tests ---

    #[test]
    fn cancellation_reason_variants() {
        let _ = CancellationReason::UserRequested;
        let _ = CancellationReason::SourceFailureDetected { generation: 1 };
        let _ = CancellationReason::DestinationFailureDetected { generation: 2 };
        let _ = CancellationReason::Preempted {
            by_migration: LiveMigrationId(99),
        };
        let _ = CancellationReason::Shutdown;
    }

    // --- MigrationStats tests ---

    #[test]
    fn migration_stats_default() {
        let s = MigrationStats::default();
        assert_eq!(s.total_duration, Duration::ZERO);
        assert_eq!(s.precopy_rounds, 0);
        assert_eq!(s.total_bytes_transferred, 0);
        assert!((s.compression_ratio - 1.0).abs() < f64::EPSILON);
    }

    // --- LiveMigrationId tests ---

    #[test]
    fn migration_id_equality() {
        assert_eq!(LiveMigrationId(1), LiveMigrationId(1));
        assert_ne!(LiveMigrationId(1), LiveMigrationId(2));
    }

    // --- AllocationRecord tests ---

    #[test]
    fn allocation_record_data_class() {
        let rec = AllocationRecord {
            vpn: 0,
            size: 4096,
            device_ptr: 0x100,
            is_pinned: true,
            alloc_flags: 0,
            current_tier: 0,
            current_node: 0,
            data_class: DataClass::Critical,
        };
        assert!(rec.is_pinned);
        assert_eq!(rec.data_class, DataClass::Critical);
    }

    // --- ModuleRecord/FunctionRecord/GlobalRecord tests ---

    #[test]
    fn module_record_with_functions() {
        let module = ModuleRecord {
            app_handle: 1,
            internal_id: 0,
            binary_data: vec![0xDE, 0xAD],
            is_ptx: true,
            functions: vec![FunctionRecord {
                app_handle: 2,
                name: "my_kernel".to_string(),
                shared_mem_bytes: 1024,
                max_dynamic_shared_mem: 4096,
                cache_config: 0,
            }],
            globals: vec![GlobalRecord {
                name: "my_global".to_string(),
                device_ptr: 0x200,
                size: 256,
            }],
        };
        assert!(module.is_ptx);
        assert_eq!(module.functions.len(), 1);
        assert_eq!(module.globals.len(), 1);
    }

    // --- MemPoolRecord tests ---

    #[test]
    fn mem_pool_record() {
        let pool = MemPoolRecord {
            app_handle: 1,
            max_size: 1024 * 1024,
            used_size: 512 * 1024,
            release_threshold: 256 * 1024,
        };
        assert!(pool.used_size < pool.max_size);
    }

    // --- EventRecord tests ---

    #[test]
    fn event_record() {
        let evt = EventRecord {
            app_handle: 10,
            internal_id: 5,
            flags: 0x01,
            is_recorded: true,
            recorded_on_stream: Some(0),
            is_completed: false,
        };
        assert!(evt.is_recorded);
        assert!(!evt.is_completed);
    }
}
