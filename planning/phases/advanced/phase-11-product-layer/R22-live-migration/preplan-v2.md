# R22: Live Migration -- Pre-Plan v2

**Created:** 2026-03-26
**Last Updated:** 2026-03-26
**Status:** Draft
**Priority:** LOW (Phase 11)
**Depends On:** R10 (Memory Tiering), R15 (Fault Tolerance), R19 (Network Page Faults)

## Purpose

Comprehensive pre-plan for live migration of running GPU workloads between OuterLink nodes. This document defines the full data model, algorithms, integration points, and test strategy for moving a CUDA context from one physical GPU to another without stopping the application. The design draws on VM live migration research (pre-copy, post-copy, hybrid), GPU-specific checkpoint/restore advances (NVIDIA cuda-checkpoint, PhoenixOS, CRIUgpu), and OuterLink's existing memory tiering and fault tolerance infrastructure.

---

## 1. Summary and Scope

### What Live Migration Does

Live migration moves a running GPU workload -- all its VRAM contents, CUDA context state, stream/event state, and kernel launch queue -- from a **source node** to a **destination node** while the application continues running. The application sees a brief pause (the "blackout window") but is never killed or restarted.

### Why It Matters

| Use Case | Description |
|----------|-------------|
| **Zero-downtime maintenance** | Drain a node before OS update, GPU driver upgrade, or hardware replacement |
| **Dynamic load balancing** | Move a workload from an overloaded node to an idle one |
| **Thermal management** | Migrate away from a GPU approaching thermal throttle |
| **Graceful pre-failure** | When R15's phi accrual detector shows a node is degrading (not yet dead), migrate proactively instead of waiting for crash recovery |
| **Resource reclamation** | Free a high-end GPU for a higher-priority job by migrating the current workload to a lower-tier GPU |

### What Is NOT In Scope

- **Cross-architecture migration** (e.g., Ampere to Hopper): destination GPU must be the same CUDA compute capability or a compatible superset.
- **Multi-GPU workload migration** (NCCL collectives spanning multiple GPUs): that is R23 territory. R22 handles single-context migration.
- **Storage migration**: only GPU VRAM and associated host-side state. Persistent storage (NVMe checkpoints) stays where it is.

### Dependencies

| Dependency | What R22 Needs From It |
|------------|----------------------|
| **R10 (Memory Tiering)** | Page table, PTE flags, MigrationEngine trait, dirty tracking via ACCESSED/DIRTY flags, tier-aware page placement |
| **R15 (Fault Tolerance)** | Failure detection (MultiLayerDetector), checkpoint infrastructure (CheckpointManager), generation-based fencing, cluster membership |
| **R19 (Network Page Faults)** | Page fault handler for post-copy demand paging, SWMR coherency protocol for ownership transfer, FaultHandler deduplication |
| **R17 (Topology)** | Node health metrics, bandwidth estimates for migration time prediction, placement scoring |

---

## 2. Migration Strategies

### 2.1 Pre-Copy Migration

The classic VM migration approach adapted for GPU VRAM.

**How it works:**
1. Start copying all VRAM pages from source to destination while the workload continues running.
2. Track which pages the workload dirties during the copy (dirty bitmap).
3. Re-copy only the dirty pages. Repeat until the dirty set is small enough.
4. Pause the workload, copy the final dirty pages + CUDA context state, resume on destination.

**Pros:** Source retains a complete working copy throughout. If migration fails, just cancel -- no data loss. Simple rollback.

**Cons:** Write-heavy workloads may never converge (dirtying pages faster than we can copy them). Total migration time can be unpredictable. Pages may be copied multiple times.

**Best for:** Read-heavy inference workloads, model weights that are mostly static.

### 2.2 Post-Copy Migration

**How it works:**
1. Pause the workload briefly, transfer only the CUDA context state and a minimal set of "essential" pages.
2. Resume on destination immediately.
3. When the workload accesses a page that hasn't been copied yet, trigger a network page fault (R19) to fetch it from the source on demand.
4. Background thread proactively pushes remaining pages from source to destination.

**Pros:** Guaranteed bounded pause time (only context + essentials). Each page transferred exactly once. Fast for large VRAM footprints.

**Cons:** If destination node fails during migration, workload state is split across two nodes -- unrecoverable without R15 parity. Post-resume performance degraded by page faults until all pages arrive.

**Best for:** Large-VRAM training workloads, write-heavy workloads where pre-copy won't converge.

### 2.3 Hybrid Migration (Recommended Default)

**How it works:**
1. Run a **bounded** pre-copy phase (max N iterations or T seconds).
2. If the dirty set converges below a threshold, finish with pre-copy (small final pause).
3. If the dirty set does NOT converge, switch to post-copy: pause, transfer context + remaining dirty bitmap, resume on destination, demand-page the rest.

**Pros:** Gets the best of both worlds. Bounded total migration time. Adaptively handles both read-heavy and write-heavy workloads.

**Cons:** More complex state machine. Must track which pages have been copied and which haven't across the strategy switch.

**Decision:** Hybrid is the default strategy. The migration controller monitors convergence rate and decides the switch point automatically.

### 2.4 Strategy Selection Heuristic

```
if estimated_dirty_rate < network_bandwidth * 0.3:
    use PreCopy  // dirty pages converge easily
elif vram_footprint < 256 MB:
    use PreCopy  // small enough to just copy everything
else:
    use Hybrid   // default safe choice
```

The user can also force a specific strategy via the CLI.

---

## 3. Core Data Structures

### 3.1 Migration Identity and Configuration

```rust
use std::time::{Duration, Instant};
use crate::memory::types::{NodeId, TierId};
use crate::fault_tolerance::Vpn;

/// Unique identifier for a live migration operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LiveMigrationId(pub u64);

/// Which strategy the migration uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

impl Default for MigrationStrategy {
    fn default() -> Self {
        Self::Hybrid {
            max_precopy_rounds: 8,
            max_precopy_duration: Duration::from_secs(30),
            convergence_threshold: 0.02,
        }
    }
}

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
    /// Uses R10's AdaptiveCompressor (LZ4 for speed, ZSTD for ratio).
    pub compress_transfers: bool,
    /// Whether to use R15 checkpoints as migration source.
    /// If a recent checkpoint exists, skip copying already-checkpointed pages.
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

/// Priority levels for migration operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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
```

### 3.2 Migration State Machine

```rust
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
///
#[derive(Debug, Clone)]
pub enum LiveMigrationState {
    /// Migration has been requested but not started.
    Requested {
        requested_at: Instant,
    },

    /// Setting up destination node: reserving VRAM, creating context shell,
    /// establishing transfer channels.
    Preparing {
        started_at: Instant,
    },

    /// Iterative pre-copy phase. Copying pages while workload runs.
    PreCopy {
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
        started_at: Instant,
        completed_at: Instant,
        /// Final statistics.
        stats: MigrationStats,
    },

    /// Migration failed. Workload still running on source (or lost if
    /// failure happened during blackout with post-copy).
    Failed {
        started_at: Instant,
        failed_at: Instant,
        /// What went wrong.
        error: LiveMigrationError,
        /// Whether the workload is still running (source alive).
        workload_intact: bool,
    },

    /// Migration was cancelled (user request, or R15 detected failure).
    Cancelled {
        started_at: Instant,
        cancelled_at: Instant,
        /// Reason for cancellation.
        reason: CancellationReason,
    },
}

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

/// Errors that can occur during live migration.
#[derive(Debug, Clone)]
pub enum LiveMigrationError {
    /// Destination node does not have enough VRAM.
    InsufficientVram {
        required: usize,
        available: usize,
    },
    /// Destination GPU is incompatible (different compute capability).
    IncompatibleGpu {
        source_cc: (u32, u32),
        dest_cc: (u32, u32),
    },
    /// Blackout duration exceeded max_blackout.
    BlackoutTimeout {
        elapsed: Duration,
        limit: Duration,
    },
    /// Total migration duration exceeded max_total_duration.
    TotalTimeout {
        elapsed: Duration,
        limit: Duration,
    },
    /// Network error during page transfer.
    TransferFailed {
        reason: String,
    },
    /// CUDA context serialization failed.
    ContextSerializationFailed {
        reason: String,
    },
    /// CUDA context restore failed on destination.
    ContextRestoreFailed {
        reason: String,
    },
    /// Source node failed during migration (R15 detected).
    SourceNodeFailed {
        generation: u64,
    },
    /// Destination node failed during migration.
    DestinationNodeFailed {
        generation: u64,
    },
    /// Pre-copy did not converge and post-copy is disabled.
    PreCopyDidNotConverge {
        dirty_ratio: f64,
        rounds_completed: u32,
    },
    /// Workload has pinned memory that cannot be migrated.
    PinnedMemoryConflict {
        pinned_pages: u64,
    },
    /// Internal error.
    Internal {
        reason: String,
    },
}

/// Reasons for migration cancellation.
#[derive(Debug, Clone)]
pub enum CancellationReason {
    /// User requested cancellation via CLI.
    UserRequested,
    /// R15 detected failure on source -- switching to crash recovery.
    SourceFailureDetected { generation: u64 },
    /// R15 detected failure on destination -- aborting migration.
    DestinationFailureDetected { generation: u64 },
    /// Higher-priority migration preempted this one.
    Preempted { by_migration: LiveMigrationId },
    /// System shutting down.
    Shutdown,
}
```

### 3.3 Dirty Bitmap Tracker

```rust
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
}
```

### 3.4 CUDA Context Snapshot

```rust
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
/// OuterLink's approach tractable -- we control the interception layer
/// and can fence all GPU work.
#[derive(Debug, Clone)]
pub struct CudaContextSnapshot {
    /// Snapshot identifier.
    pub id: u64,
    /// Source node and GPU.
    pub source_node: NodeId,
    pub source_gpu_index: u32,
    /// CUDA compute capability of source GPU.
    pub compute_capability: (u32, u32),
    /// When the snapshot was taken.
    pub captured_at: Instant,
    /// Total VRAM used by this context.
    pub vram_used_bytes: usize,

    // --- Allocation State ---
    /// All active device memory allocations.
    /// VPN -> allocation metadata. The actual page data is transferred
    /// separately via the pre-copy/post-copy mechanism.
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
    /// The CUDA allocation flags (if cuMemAllocManaged, cuMemCreate, etc.).
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
    /// The PTX or cubin binary data. We store this because we intercepted
    /// the cuModuleLoad* call and have the original data.
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
    /// Configured shared memory size (cuFuncSetSharedMemConfig).
    pub shared_mem_bytes: u32,
    /// Max dynamic shared memory.
    pub max_dynamic_shared_mem: u32,
    /// Cache preference (cuFuncSetCacheConfig).
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

/// Data class from R15, duplicated here for migration-specific use.
/// Determines how a page is treated during migration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataClass {
    /// Model weights -- immutable, can be skipped if already on destination
    /// (e.g., from a shared model cache or R15 checkpoint).
    Critical,
    /// KV cache, activations -- mutable, must be transferred.
    Recoverable,
    /// Temporary buffers -- can be zeroed on destination instead of transferred.
    Transient,
    /// Optimizer state -- protected by R15 checkpoints.
    Checkpoint,
}
```

### 3.5 Migration Transfer Protocol

```rust
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
```

### 3.6 Post-Copy Page Fault Extension

```rust
/// Extension to R19's FaultHandler for post-copy migration page faults.
///
/// During post-copy, the destination node may access pages that haven't
/// been transferred yet. The fault handler recognizes these as migration
/// faults (distinct from normal tier-miss faults) and routes them to the
/// migration source instead of R10's normal tier promotion path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationFaultType {
    /// Page is still on source, needs to be fetched.
    PostCopyFetch {
        source_node: NodeId,
        vpn: u64,
    },
    /// Page is in-flight (currently being pushed by background thread).
    /// Wait for it to arrive instead of issuing a duplicate fetch.
    InFlightWait {
        vpn: u64,
    },
}

/// Per-page migration state, used by post-copy to know where each page is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageMigrationState {
    /// Page has been transferred to destination.
    Transferred,
    /// Page is still on source, not yet transferred.
    OnSource,
    /// Page is currently being transferred (in-flight).
    InFlight,
    /// Page was skipped (Transient data class, zeroed on destination).
    Skipped,
    /// Page was served from R15 checkpoint data.
    FromCheckpoint,
}

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
        1.0 - (self.remaining() as f64 / self.num_pages as f64)
    }
}
```

### 3.7 Migration Orchestrator

```rust
/// The top-level migration controller. Manages the lifecycle of a live
/// migration operation, coordinating between source and destination nodes.
///
/// One MigrationOrchestrator exists per active migration. It is created
/// on the coordinator node (which may be source, destination, or a third
/// node).
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

    // --- Subsystem references ---
    /// R10 page table (for VPN lookups, flag updates).
    page_table: Arc<dyn PageTable>,
    /// R10 migration engine (for individual page transfers).
    migration_engine: Arc<dyn MigrationEngine>,
    /// R15 failure detector (abort migration on node failure).
    failure_detector: Arc<dyn FailureDetector>,
    /// R15 checkpoint manager (bootstrap from checkpoint).
    checkpoint_mgr: Arc<CheckpointManager>,
    /// R19 fault handler (post-copy page faults).
    fault_handler: Arc<FaultHandler>,
    /// R17 topology (bandwidth estimation, placement).
    topology: Arc<TopologyGraph>,
    /// Transport layer for bulk transfers.
    transport: Arc<dyn Transport>,

    // --- Migration-specific state ---
    /// Dirty page bitmap (source side).
    dirty_bitmap: DirtyBitmap,
    /// Per-page migration state tracker (used during post-copy).
    page_tracker: PageMigrationTracker,
    /// Captured CUDA context snapshot (populated during blackout).
    context_snapshot: Option<CudaContextSnapshot>,
    /// Convergence history for strategy adaptation.
    convergence_history: Vec<ConvergenceDataPoint>,
}

/// One data point per pre-copy round, tracking convergence.
#[derive(Debug, Clone)]
pub struct ConvergenceDataPoint {
    pub round: u32,
    pub dirty_pages: u64,
    pub dirty_ratio: f64,
    pub transfer_time: Duration,
    pub transfer_bytes: u64,
}
```

---

## 4. Key Algorithms

### 4.1 Pre-Copy Iterative Convergence

The pre-copy algorithm transfers VRAM pages iteratively, with each round copying only the pages dirtied since the last round. The goal is convergence: the dirty set shrinks each round until it is small enough to transfer during the blackout window within the configured `max_blackout` limit.

**Algorithm:**

```
function precopy_loop(migration):
    // Round 0: copy ALL pages
    dirty_bitmap.mark_all_dirty()

    for round in 0..max_rounds:
        // 1. Scan the dirty bitmap and reset it
        dirty_vpns = dirty_bitmap.scan_and_reset()

        // 2. Record convergence data
        convergence_history.push(round, len(dirty_vpns))

        // 3. Check convergence: can we finish in one blackout?
        estimated_blackout = estimate_blackout_time(
            dirty_pages = len(dirty_vpns),
            context_size = estimated_context_size(),
            transfer_rate = measured_transfer_rate,
        )

        if estimated_blackout < config.max_blackout:
            return ConvergenceResult::Converged {
                final_dirty_count: len(dirty_vpns),
                estimated_blackout,
            }

        // 4. Check for divergence (getting worse, not better)
        if round >= 3 and is_diverging(convergence_history):
            if config.strategy.allows_postcopy():
                return ConvergenceResult::SwitchToPostCopy {
                    pages_already_transferred: tracker.transferred_count(),
                }
            else:
                return ConvergenceResult::Failed {
                    dirty_ratio: dirty_bitmap.dirty_ratio(),
                }

        // 5. Check timeout
        if elapsed > config.strategy.max_precopy_duration:
            return ConvergenceResult::Timeout

        // 6. Transfer dirty pages in batches
        for batch in dirty_vpns.chunks(BATCH_SIZE):
            read pages from source VRAM
            compress if enabled
            send to destination via RDMA/TCP
            wait for ack
            update page_tracker: OnSource -> Transferred

        // Note: while we're transferring, the workload keeps running
        // and dirtying pages. Those dirties go into the bitmap and
        // will be picked up in the next round.

    return ConvergenceResult::MaxRoundsReached
```

**Convergence detection:**

```
function is_diverging(history) -> bool:
    // Look at the last 3 rounds. If the dirty count is not decreasing
    // (or is increasing), we are diverging.
    if len(history) < 3:
        return false
    last3 = history[-3:]
    // Linear regression slope on dirty_pages
    slope = linear_regression_slope(last3.dirty_pages)
    return slope > 0  // positive slope = getting worse
```

**Estimated blackout time:**

```
function estimate_blackout_time(dirty_pages, context_size, transfer_rate):
    page_transfer_time = (dirty_pages * PAGE_SIZE) / transfer_rate
    context_transfer_time = context_size / transfer_rate
    synchronize_time = 5ms  // cuCtxSynchronize estimate
    safety_margin = 1.2     // 20% safety margin
    return (page_transfer_time + context_transfer_time + synchronize_time) * safety_margin
```

### 4.2 Post-Copy Demand Paging

When the migration switches to post-copy (either from the start or after pre-copy fails to converge), the workload resumes on the destination GPU. Pages not yet transferred trigger network page faults.

**Algorithm:**

```
function postcopy_main(migration):
    // Two concurrent tasks:
    // 1. Background pusher: proactively sends remaining pages
    // 2. Fault handler: serves on-demand page requests

    spawn background_pusher(migration)

    // The workload is now running on destination.
    // Page faults are handled by the modified R19 fault handler.
    // See Section 5.3 for the fault handler integration.

function background_pusher(migration):
    // Push remaining pages from source to destination in priority order.
    // Priority is based on R10 access pattern data:
    //   1. Hot pages (recently accessed) first
    //   2. Working set pages
    //   3. Sequential/streaming pages
    //   4. Cold pages last

    remaining = page_tracker.get_all(PageMigrationState::OnSource)
    remaining.sort_by(|a, b| access_priority(a).cmp(&access_priority(b)))

    for vpn in remaining:
        // Check if this page was already fetched by a demand fault
        if page_tracker.get(vpn) != PageMigrationState::OnSource:
            continue

        // Mark as in-flight to prevent duplicate fetch
        page_tracker.transition(vpn, PageMigrationState::InFlight)

        // Transfer the page
        data = read_page_from_source(vpn)
        send_to_destination(vpn, data)
        wait_for_ack()

        page_tracker.transition(vpn, PageMigrationState::Transferred)

        // Throttle if configured
        if config.bandwidth_limit > 0:
            throttle()

    // All pages transferred, post-copy complete
    migration.state = Completed

function access_priority(vpn) -> u32:
    pte = page_table.lookup(vpn)
    match pte.access_pattern_type:
        WorkingSet => 0  // highest priority
        Phased => 1
        Streaming => 2
        Random => 3
        Unknown => 4    // lowest priority
```

**Post-copy page fault handling (integration with R19):**

```
function on_page_fault(vpn, fault_type) -> FaultResolution:
    // Called by R19's fault handler when a page access triggers a fault
    // during post-copy migration.

    state = page_tracker.get(vpn)
    match state:
        Transferred:
            // Page is already here. This shouldn't happen unless there's
            // a race with PTE update. Just return the local copy.
            return FaultResolution::Local

        OnSource:
            // Page hasn't been sent yet. Fetch it now (urgent).
            page_tracker.transition(vpn, PageMigrationState::InFlight)
            data = fetch_page_from_source_urgent(vpn)
            write_to_local_vram(vpn, data)
            page_tracker.transition(vpn, PageMigrationState::Transferred)
            return FaultResolution::Fetched

        InFlight:
            // Page is currently being transferred by background pusher.
            // Wait for it to arrive.
            wait_for_page_arrival(vpn)
            return FaultResolution::Fetched

        Skipped:
            // Transient page, was zeroed. Just return the zero page.
            return FaultResolution::Local

        FromCheckpoint:
            // Already restored from checkpoint. Return local.
            return FaultResolution::Local
```

### 4.3 Blackout (Final Switchover) Protocol

The blackout is the critical pause window where the workload is stopped, final state is transferred, and the workload is resumed on the destination. This must be as short as possible.

**Algorithm:**

```
function execute_blackout(migration) -> Result<Duration>:
    blackout_start = now()

    // === Phase 1: Quiesce the workload ===
    // Intercept all new CUDA calls from the application and block them.
    // This prevents new GPU work from being submitted.
    interception_layer.pause_application()

    // Wait for all in-flight GPU kernels to complete.
    // We call cuCtxSynchronize through our interception layer.
    // This ensures all GPU state is quiesced -- no kernels running,
    // no async memcpys in flight.
    cuda_synchronize_all_streams()

    // === Phase 2: Final dirty page transfer ===
    // Scan the dirty bitmap one last time.
    final_dirty = dirty_bitmap.scan_and_reset()

    // Transfer final dirty pages. These are guaranteed to not change
    // (workload is paused).
    for batch in final_dirty.chunks(BATCH_SIZE):
        transfer_batch_to_destination(batch)
    wait_for_all_acks()

    // === Phase 3: Serialize CUDA context ===
    // Capture the full CUDA context state from the interception layer.
    // We already track all this state (see CudaContextSnapshot).
    context = capture_cuda_context_snapshot()

    // Send context to destination.
    send_context_to_destination(context)
    wait_for_context_ack()

    // === Phase 4: Restore on destination ===
    // Destination node reconstructs the CUDA context:
    //   1. Create new CUDA context with same flags
    //   2. Load modules (cuModuleLoadData with stored binary)
    //   3. Get function handles (cuModuleGetFunction)
    //   4. Create streams with same priorities
    //   5. Create events with same flags
    //   6. Re-map device pointers to new VRAM locations
    //   7. Set up address translation table (old device_ptr -> new device_ptr)
    destination.restore_cuda_context(context)

    // === Phase 5: Update routing ===
    // Tell R10 page table: all pages for this workload now live on dest_node.
    // For pre-copy: all pages are already on destination.
    // For hybrid/post-copy: some pages are on destination, rest still on source.
    update_page_table_routing(migration)

    // Update R19 coherency directory: ownership transfers to destination.
    update_coherency_directory(migration)

    // === Phase 6: Resume ===
    // If pure pre-copy: resume on destination, done.
    // If post-copy: resume on destination, start post-copy background push.
    interception_layer.resume_application_on(dest_node)

    blackout_duration = now() - blackout_start

    if blackout_duration > config.max_blackout:
        // Too slow. Log a warning but don't abort -- the workload is
        // already on the destination. The damage is done.
        warn!("Blackout exceeded limit: {:?} > {:?}",
              blackout_duration, config.max_blackout);

    return Ok(blackout_duration)
```

### 4.4 Device Pointer Remapping

A critical challenge: the application holds raw device pointers (CUdeviceptr) to VRAM allocations. After migration, these pointers must map to the new VRAM locations on the destination GPU.

**Approach:** OuterLink already solves this. Because we intercept all CUDA driver API calls via LD_PRELOAD, we return **virtual** device pointers to the application. The mapping from virtual to physical is handled by our page table (R10). When a page moves, we update the PTE -- the application's pointer doesn't change.

```
Application sees:  CUdeviceptr 0x0000_7F00_0000_0000
OuterLink maps:    VPN 0x7F0000 -> {node: 2, tier: REMOTE_VRAM, phys_pfn: 0x1234}

After migration:
Application sees:  CUdeviceptr 0x0000_7F00_0000_0000  (UNCHANGED)
OuterLink maps:    VPN 0x7F0000 -> {node: 5, tier: LOCAL_VRAM, phys_pfn: 0x5678}
```

The application never knows the migration happened. This is the fundamental advantage of OuterLink's interception architecture for live migration -- we already virtualize device pointers, so migration is "just" a page table update.

**Exception: kernel launch arguments.** When the application launches a kernel, it passes device pointers as kernel arguments. If a kernel was queued but not yet launched at the time of migration, those pointers need to be valid on the destination. Since we quiesce all GPU work before migration (cuCtxSynchronize), there are no queued-but-not-launched kernels. After migration, new kernel launches go through our interception layer which resolves virtual pointers to the new physical locations.

### 4.5 Checkpoint Bootstrap Optimization

If R15 has a recent checkpoint of the workload's VRAM state, the migration can skip transferring pages that haven't changed since the checkpoint. This significantly reduces pre-copy transfer volume.

**Algorithm:**

```
function checkpoint_bootstrap(migration) -> BootstrapResult:
    // Ask R15 for the most recent checkpoint of this workload.
    checkpoint = checkpoint_mgr.latest_checkpoint_for(source_node, source_gpu)

    if checkpoint is None:
        return BootstrapResult::NoCheckpoint

    // Calculate checkpoint age.
    age = now() - checkpoint.created_at

    // If checkpoint is very old (> 60s), the dirty set since checkpoint
    // is likely large. Fall back to full pre-copy.
    if age > Duration::from_secs(60):
        return BootstrapResult::TooOld { age }

    // Transfer the checkpoint to the destination if it's not already there.
    // R15 checkpoints often have redundant copies on partner nodes (Gemini
    // placement). If the destination happens to be the partner, we skip this.
    if not destination_has_checkpoint(checkpoint):
        transfer_checkpoint_to_destination(checkpoint)

    // Apply checkpoint on destination: this populates VRAM with the
    // checkpoint state. Pages that changed since the checkpoint will be
    // overwritten during pre-copy.
    destination.apply_checkpoint(checkpoint)

    // Mark checkpoint pages as Transferred in the page tracker.
    for shard in checkpoint.shards:
        for vpn in shard.vpn_range():
            page_tracker.transition(vpn, PageMigrationState::FromCheckpoint)

    // The pre-copy phase only needs to transfer pages dirtied since
    // the checkpoint. This is typically much smaller than the full VRAM
    // footprint.
    return BootstrapResult::Applied {
        checkpoint_id: checkpoint.id,
        pages_bootstrapped: checkpoint.total_pages(),
        estimated_dirty_since: estimate_dirty_since(checkpoint),
    }
```

---

## 5. Integration Points

### 5.1 R10 (Memory Tiering) Integration

| R22 calls R10 | Purpose |
|----------------|---------|
| `page_table.lookup(vpn)` | Get page location, flags, data class during migration planning |
| `page_table.scan(predicate, limit)` | Find all pages belonging to a workload on source node |
| `page_table.update_flags(vpn, set, clear)` | Set MIGRATING flag during transfer, clear on completion |
| `page_table.commit_migration(vpn, new_tier, new_node, new_pfn)` | Atomically update page location after transfer |
| `migration_engine.migrate_page(request)` | Delegate individual page transfers to R10's existing engine |
| `dirty_bitmap` integration | R22's dirty bitmap hooks into R10's write path: when R10 processes a write to a page, it also marks the page dirty in R22's bitmap |

| R10 calls R22 | Purpose |
|----------------|---------|
| `is_migration_in_progress(vpn)` | R10 checks before evicting/promoting a page -- pages in active migration are not evicted |
| `on_write_during_migration(vpn)` | R10 notifies R22 when a page being migrated is written (dirty bitmap update) |

**Key constraint:** R22 must set PteFlags::MIGRATING on pages during transfer. R10's eviction policy already respects this flag and skips MIGRATING pages. R22 must clear the flag when the page is successfully transferred or the migration is cancelled.

### 5.2 R15 (Fault Tolerance) Integration

| R22 calls R15 | Purpose |
|----------------|---------|
| `failure_detector.phi(node)` | Check health of source and destination during migration |
| `failure_detector.on_failure(callback)` | Register callback to abort migration if a node fails |
| `checkpoint_mgr.latest_checkpoint_for(node, gpu)` | Bootstrap from existing checkpoint data |
| `membership.generation` | Validate cluster generation hasn't changed during migration |

| R15 calls R22 | Purpose |
|----------------|---------|
| `abort_migration(reason)` | R15 cancels in-progress migration when it detects failure and needs to begin crash recovery |

**Interaction protocol:**

1. Before starting migration, R22 registers a failure callback with R15's MultiLayerDetector.
2. If R15 detects that the source node has failed mid-migration:
   - If in pre-copy phase: cancel migration, source is dead, trigger R15 crash recovery.
   - If in blackout phase: critical failure. If context was already sent to destination, attempt to continue on destination with whatever pages were transferred + post-copy from parity data. If context was not yet sent, workload is lost -- fall back to R15 checkpoint restore.
   - If in post-copy phase: destination has the context and some pages. Source is dead. Remaining pages must be reconstructed from R15 parity data instead of fetched from source. R22 switches its page fault source from "source node" to "R15 parity reconstruction."
3. If R15 detects the destination node has failed:
   - Cancel migration. Workload is still running on source (pre-copy/blackout) or partially on destination (post-copy). For post-copy, this is the dangerous case -- workload state is split. R15 must reconstruct from parity + whatever the source still has.

### 5.3 R19 (Network Page Faults / SWMR Coherency) Integration

| R22 calls R19 | Purpose |
|----------------|---------|
| `fault_handler.register_migration_source(vpn_range, source_node)` | Tell R19 that page faults in this range during post-copy should fetch from the migration source |
| `coherency_dir.transfer_ownership(vpn, new_owner)` | After migration, transfer SWMR ownership of all migrated pages to destination |
| `coherency_dir.remove_reader(vpn, old_node)` | Clean up reader entries for source node after migration |

| R19 calls R22 | Purpose |
|----------------|---------|
| `on_migration_fault(vpn)` | R19's fault handler delegates to R22 when a fault occurs on a page in the migration VPN range |

**Post-copy fault flow:**

```
1. Destination GPU accesses page VPN X
2. PTE lookup: VPN X has FAULT_PENDING flag (not yet transferred)
3. R19 FaultHandler fires
4. FaultHandler checks: is VPN X in a migration range?
   Yes -> delegate to R22's on_migration_fault(X)
5. R22 checks page_tracker.get(X):
   OnSource -> fetch from source, write to local VRAM, update PTE
   InFlight -> wait for background push to complete
   (see Section 4.2 for full algorithm)
6. CUDA operation resumes with page available locally
```

### 5.4 R17 (Topology) Integration

| R22 calls R17 | Purpose |
|----------------|---------|
| `topology.get_link_bandwidth(src, dst)` | Estimate transfer rate for migration time prediction |
| `topology.get_node_load(node)` | Select least-loaded destination for automatic migration |
| `placement_scorer.score(workload, candidate_nodes)` | Score candidate destination nodes |

R22 uses R17's topology information for:
- **Migration time estimation:** bandwidth between source and destination determines how long pre-copy rounds will take, and whether convergence is feasible.
- **Destination selection:** when migration is triggered automatically (load balancing, pre-failure), R22 uses R17 to pick the best destination.

---

## 6. CUDA Context Serialization Approach

### 6.1 Key Insight: We Already Track Everything

OuterLink's interception layer (LD_PRELOAD on the CUDA driver API) already intercepts and records every context-modifying call. This means we do NOT need to use NVIDIA's cuda-checkpoint utility (which requires driver 570+ and same-chip-type GPUs). Instead, we reconstruct the context from our interception records.

### 6.2 What We Capture (Already Tracked by Interception Layer)

| CUDA State | How We Track It | Migration Action |
|------------|-----------------|-----------------|
| Device memory allocations | Intercepted cuMemAlloc, cuMemAllocManaged, cuMemCreate | Re-allocate on destination with same virtual addresses |
| Memory contents | Page table (R10) + actual VRAM data | Transferred via pre-copy/post-copy |
| Loaded modules | Intercepted cuModuleLoad, cuModuleLoadData | Re-load from stored binary on destination |
| Function handles | Intercepted cuModuleGetFunction | Re-get from re-loaded modules |
| Streams | Intercepted cuStreamCreate | Re-create with same priority/flags |
| Events | Intercepted cuEventCreate, cuEventRecord | Re-create. Recorded events are reset. |
| Context flags | Intercepted cuCtxCreate | Re-create context with same flags |
| Shared mem config | Intercepted cuCtxSetSharedMemConfig | Re-apply |
| Cache config | Intercepted cuCtxSetCacheConfig, cuFuncSetCacheConfig | Re-apply |

### 6.3 What We Do NOT Capture (And Why It's OK)

| CUDA State | Why We Skip It |
|------------|---------------|
| Running kernel register state | We call cuCtxSynchronize before snapshot. No kernels are running. |
| In-flight async memcpy state | cuCtxSynchronize waits for all async ops. Nothing in-flight. |
| GPU hardware counters | Not needed for correctness. Profiling state is lost. |
| CUDA graph instances | We intercept cuGraphLaunch and replay the graph construction on destination. |
| cuFFT/cuBLAS/cuDNN internal state | These libraries are stateless from the application's perspective. Handles are reconstructed by re-calling the library init functions. |

### 6.4 Serialization Format

The CudaContextSnapshot (see Section 3.4) is serialized using a simple binary format:

```
Header (32 bytes):
  magic: u32 = 0x4F4C4D47  ("OLMG" = OuterLink MiGration)
  version: u32 = 1
  snapshot_id: u64
  source_node: u8
  source_gpu: u32
  compute_capability_major: u32
  compute_capability_minor: u32
  num_allocations: u32
  num_streams: u32
  num_events: u32
  num_modules: u32
  total_size: u64

AllocationRecords: [AllocationRecord; num_allocations]
StreamRecords: [StreamRecord; num_streams]
EventRecords: [EventRecord; num_events]
ModuleRecords: [ModuleRecord; num_modules]
  (each ModuleRecord includes inline binary data for PTX/cubin)

Footer:
  checksum: u32 (CRC32 of everything above)
```

Typical size for a single-GPU inference workload: 1-10 KB (metadata only, page data is separate). For a training workload with many modules: 10-100 KB. This is tiny compared to VRAM content.

### 6.5 Context Restore on Destination

```
function restore_cuda_context(snapshot, page_table):
    // 1. Validate compatibility
    assert dest_gpu.compute_capability >= snapshot.compute_capability

    // 2. Create CUDA context
    ctx = cuCtxCreate(snapshot.ctx_flags, dest_gpu)
    cuCtxSetSharedMemConfig(snapshot.shared_mem_config)
    cuCtxSetCacheConfig(snapshot.cache_config)

    // 3. Load modules
    for module_rec in snapshot.modules:
        if module_rec.is_ptx:
            mod = cuModuleLoadDataEx(module_rec.binary_data, jit_options)
        else:
            mod = cuModuleLoadData(module_rec.binary_data)

        // Get function handles
        for func_rec in module_rec.functions:
            func = cuModuleGetFunction(mod, func_rec.name)
            cuFuncSetSharedMemConfig(func, func_rec.shared_mem_bytes)
            cuFuncSetCacheConfig(func, func_rec.cache_config)

            // Store mapping: old_handle -> new_handle
            handle_map.insert(func_rec.app_handle, func)

        // Get global variable handles
        for global_rec in module_rec.globals:
            (ptr, size) = cuModuleGetGlobal(mod, global_rec.name)
            // Map old device_ptr to new device_ptr
            ptr_map.insert(global_rec.device_ptr, ptr)

    // 4. Create streams
    for stream_rec in snapshot.streams:
        if stream_rec.is_default:
            handle_map.insert(stream_rec.app_handle, null_stream)
        else:
            stream = cuStreamCreateWithPriority(stream_rec.flags, stream_rec.priority)
            handle_map.insert(stream_rec.app_handle, stream)

    // 5. Create events
    for event_rec in snapshot.events:
        event = cuEventCreate(event_rec.flags)
        handle_map.insert(event_rec.app_handle, event)

    // 6. Set up device pointer virtualization
    // OuterLink already virtualizes device pointers. We just need to
    // update the page table to point to the new physical locations.
    // The application's virtual pointers remain unchanged.
    for alloc_rec in snapshot.allocations:
        // Allocate physical backing on destination GPU
        dest_phys = allocate_on_dest_gpu(alloc_rec.size)
        // Update page table: VPN now maps to destination
        page_table.commit_migration(
            alloc_rec.vpn,
            LOCAL_VRAM,  // it's local to the destination now
            dest_node,
            dest_phys,
        )

    // 7. Set up interception layer on destination
    // Install the handle_map so that when the application uses old handles,
    // our interception layer translates to new handles.
    interception_layer.install_handle_map(handle_map)
    interception_layer.install_ptr_map(ptr_map)

    return Ok(())
```

---

## 7. Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Blackout duration** | < 500 ms | Network stack timeouts are typically 750ms-1s. Staying under 500ms keeps the application responsive. Matches Microsoft GPU-P live migration target. |
| **Pre-copy convergence** | < 8 rounds for inference workloads | Inference has mostly-static model weights. Only KV cache and activations change. Should converge quickly. |
| **Total migration time (24 GB VRAM)** | < 60 s over 10 Gbps link | 24 GB / 10 Gbps = ~20s raw transfer. With compression (2x typical on weight data) and overhead, 60s budget. |
| **Total migration time (24 GB VRAM)** | < 12 s over 100 Gbps RDMA | 24 GB / 100 Gbps = ~2s raw transfer. 12s budget with overhead. |
| **Post-copy page fault latency** | < 2 ms per page over RDMA | Single 64 KB page over 100 Gbps RDMA. R19 already targets this. |
| **Post-copy page fault latency** | < 10 ms per page over TCP | Single 64 KB page over 10 Gbps TCP. |
| **Migration overhead on running workload** | < 10% throughput reduction | Dirty bitmap tracking is O(1) per write. Pre-copy transfer competes for network bandwidth but is throttleable. |
| **Context serialization** | < 5 ms | Context snapshot is small metadata (1-100 KB). Serialization is trivial. |
| **Context restore** | < 50 ms | Module loading (JIT if PTX) dominates. cubin loading is fast (<5ms). PTX JIT can take 10-50ms per module. |
| **Checkpoint bootstrap savings** | 50-90% fewer pages transferred | For inference (static weights), checkpoint covers nearly everything. For training, optimizer state is checkpointed, only recent activations need transfer. |

### Bandwidth Estimation Table

| Network | Bandwidth | 24 GB raw transfer | With 2x compression | With checkpoint bootstrap (90% hit) |
|---------|-----------|---------------------|---------------------|--------------------------------------|
| 10 Gbps TCP | ~1 GB/s | 24 s | 12 s | 2.4 s |
| 25 Gbps TCP | ~2.5 GB/s | 9.6 s | 4.8 s | 0.96 s |
| 100 Gbps RDMA | ~12 GB/s | 2 s | 1 s | 0.2 s |
| 200 Gbps RDMA | ~24 GB/s | 1 s | 0.5 s | 0.1 s |

---

## 8. Test Plan

### 8.1 Unit Tests

| Test | What It Validates |
|------|-------------------|
| `test_dirty_bitmap_mark_and_scan` | Mark pages dirty, scan_and_reset returns correct VPNs, bitmap is cleared after scan |
| `test_dirty_bitmap_mark_range` | mark_range_dirty correctly sets all bits in a range |
| `test_dirty_bitmap_mark_all` | mark_all_dirty sets all bits, handles non-64-aligned page counts |
| `test_dirty_bitmap_dirty_ratio` | dirty_ratio computes correctly |
| `test_page_tracker_transitions` | PageMigrationTracker transitions update counts correctly |
| `test_page_tracker_progress` | progress() and remaining() return correct values |
| `test_page_tracker_complete` | is_complete() returns true only when all pages transferred |
| `test_migration_state_transitions` | State machine transitions follow allowed paths (no invalid transitions) |
| `test_convergence_detection` | is_diverging correctly detects increasing dirty counts |
| `test_blackout_estimation` | estimate_blackout_time gives reasonable estimates |
| `test_context_snapshot_serialization` | CudaContextSnapshot round-trips through serialize/deserialize |
| `test_context_snapshot_checksum` | Corrupted snapshot data is detected by CRC check |
| `test_strategy_selection_heuristic` | Correct strategy is chosen based on dirty rate and VRAM size |
| `test_migration_config_defaults` | Default config has sensible values |

### 8.2 Integration Tests

| Test | Scenario | Expected Result |
|------|----------|-----------------|
| `test_precopy_static_workload` | Migrate a workload with 100% read-only pages (model weights). No writes during migration. | Converges in round 0 (initial copy). Blackout < 50ms (only context transfer). |
| `test_precopy_low_write_workload` | Migrate with 5% pages dirtied per round. | Converges in 3-5 rounds. Each round's dirty set is ~5% of total. |
| `test_precopy_high_write_divergence` | Migrate with 50% pages dirtied per round (write-heavy training). Strategy = Hybrid. | Pre-copy detects divergence after 3 rounds, switches to post-copy. |
| `test_postcopy_demand_faults` | Pure post-copy. Destination accesses pages in sequence. | Each access triggers a page fault. Pages arrive from source. All pages eventually transferred. |
| `test_postcopy_background_push` | Post-copy. Destination does NOT access most pages. | Background pusher transfers all pages. No demand faults for untouched pages. |
| `test_hybrid_convergence_then_finish` | Hybrid. Low write rate. | Pre-copy converges, finishes as pre-copy. Post-copy phase never entered. |
| `test_hybrid_switch_to_postcopy` | Hybrid. High write rate. | Pre-copy fails to converge, switches to post-copy. Both phases executed. |
| `test_checkpoint_bootstrap` | Migrate with a recent R15 checkpoint available. | 90%+ pages skipped in pre-copy. Only dirty-since-checkpoint pages transferred. |
| `test_checkpoint_bootstrap_stale` | Migrate with an old R15 checkpoint (> 60s). | Checkpoint bootstrap skipped. Full pre-copy. |
| `test_blackout_timeout_abort` | Set max_blackout to 10ms. Migration has too many final dirty pages to transfer in 10ms. | Migration logs warning but completes (workload already moved). |
| `test_total_timeout_abort` | Set max_total_duration to 1s. Migration of 24 GB over slow link. | Migration aborts with TotalTimeout error. Workload stays on source. |
| `test_source_failure_during_precopy` | Simulate source node crash during pre-copy round 3. | R15 callback fires. Migration cancelled with SourceFailureDetected. R15 crash recovery takes over. |
| `test_source_failure_during_postcopy` | Simulate source node crash during post-copy. Some pages not yet transferred. | R22 switches page fault source from "source node" to "R15 parity reconstruction." Remaining pages recovered from parity. |
| `test_destination_failure_during_precopy` | Simulate dest node crash during pre-copy. | Migration cancelled. Workload unaffected (still on source). |
| `test_destination_failure_during_postcopy` | Simulate dest node crash during post-copy. Workload split across nodes. | Critical failure. R15 reconstructs from parity + source residual. |
| `test_concurrent_migrations` | Two migrations running simultaneously on different workloads. | Both complete independently. No interference. |
| `test_migration_with_pinned_pages` | Workload has cuMemHostRegister'd pages (pinned). | Pinned pages are included in migration. PinnedMemoryConflict error if destination can't pin. |
| `test_coherency_transfer` | Migrated pages have SWMR writer on source. | After migration, R19 coherency directory shows destination as new writer. |
| `test_parity_update_after_migration` | Migrated pages are in R15 parity groups. | After migration, R15 parity locations updated to reflect new page locations. |

### 8.3 Stress Tests

| Test | What It Stresses |
|------|-----------------|
| `stress_rapid_dirty_rate` | Workload dirties 100% of VRAM every 100ms. Pre-copy never converges. Validates hybrid switch and post-copy correctness. |
| `stress_large_vram_migration` | Migrate 24 GB VRAM with 10 Gbps link. Validates throughput, memory management (no OOM on destination during gradual population). |
| `stress_postcopy_thundering_herd` | Post-copy: 1000 concurrent kernel launches all faulting on different pages simultaneously. Validates fault handler doesn't deadlock or OOM. |
| `stress_migration_under_memory_pressure` | Destination node is at 80% VRAM utilization. Migration must trigger eviction on destination to make room. |
| `stress_cascading_migration` | Migrate workload A from node 1 to node 2. While in-flight, migrate workload B from node 2 to node 3 (node 2 is both a destination and a source). |

### 8.4 Fault Injection Tests

| Test | Injected Fault | Expected Behavior |
|------|---------------|-------------------|
| `fault_network_partition_during_precopy` | Drop all packets between source and dest for 5s during pre-copy round 2. | Transfer stalls. When network recovers, pre-copy resumes from where it left off. No data corruption. |
| `fault_network_partition_during_blackout` | Drop packets during blackout. | Blackout times out (context can't be sent). Migration fails with TransferFailed. Workload stays on source. |
| `fault_corrupted_page_transfer` | Flip a bit in a transferred page. Verify checksums enabled. | Checksum mismatch detected. Page re-transferred. |
| `fault_destination_oom` | Destination runs out of VRAM mid-migration. | InsufficientVram error. Migration cancelled. Source workload unaffected. |
| `fault_slow_destination` | Destination ACKs are delayed by 500ms. | Migration slows down but completes. Total time increases. No correctness issues. |

---

## 9. Implementation Phases

| Phase | Duration | Dependencies | Deliverables |
|-------|----------|-------------|--------------|
| **R22-A: Core Types & Dirty Bitmap** | 2 weeks | R10 page table API | All structs from Section 3, DirtyBitmap with unit tests, PageMigrationTracker |
| **R22-B: Pre-Copy Engine** | 3 weeks | R22-A, R10 migration engine | Pre-copy iterative loop, convergence detection, batch transfer, integration with R10 write path for dirty marking |
| **R22-C: CUDA Context Snapshot** | 2 weeks | R22-A, interception layer | CudaContextSnapshot capture and restore, serialization format, handle remapping |
| **R22-D: Blackout Protocol** | 2 weeks | R22-B, R22-C | Full blackout sequence (quiesce, final transfer, context send, restore, resume), page table routing update |
| **R22-E: Post-Copy Engine** | 3 weeks | R22-D, R19 fault handler | Post-copy page fault handling, background pusher, PageMigrationTracker integration with R19 |
| **R22-F: Hybrid Strategy & R15 Integration** | 2 weeks | R22-B, R22-E, R15 | Strategy selection heuristic, hybrid switch logic, checkpoint bootstrap, failure detection callbacks |
| **R22-G: Integration & Testing** | 3 weeks | All above | Full integration tests, stress tests, fault injection, CLI commands for triggering migration |

**Total: 17 weeks**

**Parallelism:** R22-B and R22-C can run in parallel (no dependency between them). R22-D depends on both. R22-E depends on R22-D. R22-F depends on R22-B and R22-E.

**Critical path:** R22-A -> R22-B -> R22-D -> R22-E -> R22-G (13 weeks)

---

## 10. Open Questions

### Q1: GPU Compute Capability Compatibility

Can we migrate between different GPU architectures within the same compute capability family? For example, RTX 3090 (SM 8.6) to RTX 3080 (SM 8.6)? They have the same compute capability but different VRAM sizes and core counts.

**Tentative answer:** Yes, as long as:
- Destination has enough VRAM for the workload.
- Compute capability matches (PTX/cubin compatibility).
- VRAM layout differences are handled by OuterLink's virtualization (we don't depend on specific physical addresses).

### Q2: CUDA Graph Migration

CUDA graphs capture a sequence of operations for replay. If the application uses cuGraphLaunch, we need to:
1. Intercept cuGraphCreate, cuGraphAddKernelNode, etc.
2. Store the graph definition.
3. Reconstruct the graph on the destination with new handles.

This is achievable because we already intercept these calls, but adds complexity. Should CUDA graph support be a separate sub-phase?

**Tentative answer:** Yes. CUDA graphs are common in inference (e.g., TensorRT). Support them in R22-C (context snapshot). The graph definition is metadata; the graph execution is just kernel launches that go through our interception layer.

### Q3: Dirty Tracking Granularity for Kernel Launches

When a CUDA kernel is launched, it may write to arbitrary VRAM locations. We know the kernel arguments (which include device pointers), but we don't know exactly which pages the kernel will write to.

**Options:**
1. **Conservative:** Mark all allocation ranges passed as kernel arguments as dirty. Over-estimates dirty set but safe.
2. **Output-only:** Only mark allocations that are output parameters as dirty. Requires inferring which parameters are outputs (hard in general, but cuBLAS/cuDNN have known signatures).
3. **Full-range:** After kernel completion, scan the PTE DIRTY flags set by the GPU hardware. Requires GPU hardware dirty bit support (not available on GeForce).

**Tentative answer:** Option 1 (conservative) as the default. The over-estimation means more pages re-transferred during pre-copy, but no missed dirty pages. Option 2 can be added as an optimization for known library calls. Option 3 is not available on our target hardware (GeForce GPUs).

### Q4: Memory-Mapped Host Buffers

If the application uses cuMemHostRegister to register host memory, that memory is accessible by the GPU via DMA. During migration, we need to either:
- Transfer the host buffer contents too, or
- Ensure the new GPU can DMA from the same host buffer (only works for same-node migration).

**Tentative answer:** For cross-node migration, host-registered buffers must be transferred. Track them in AllocationRecord with `is_pinned = true`. Transfer them alongside VRAM pages.

### Q5: Multi-Process Sharing (CUDA IPC)

If two processes share GPU memory via cuIpcGetMemHandle / cuIpcOpenMemHandle, migrating one process's allocation affects the other.

**Tentative answer:** Out of scope for v1. If CUDA IPC is detected (we intercept cuIpcGetMemHandle), block migration with an error. Support for IPC-aware migration can be added later.

### Q6: Optimal Batch Size for Page Transfer

What is the optimal number of pages per transfer batch? Factors:
- Larger batches: better throughput (amortize per-batch overhead), worse latency (each batch takes longer).
- Smaller batches: lower latency, more overhead.
- RDMA has different optimal batch sizes than TCP.

**Tentative answer:** Start with 64 pages per batch (64 * 64 KB = 4 MB). Tune based on benchmarks. Make configurable.

### Q7: Priority Inversion During Post-Copy

During post-copy, a background push might be sending cold pages while a demand fault needs a hot page urgently. The background push occupies the network channel.

**Tentative answer:** Use separate network channels (or separate RDMA QPs) for demand faults vs. background push. Demand faults get priority. Background push can be paused while demand faults are being served.

### Q8: Handling cuMemcpy During Migration

If the application does a cuMemcpyDtoD (device-to-device copy) during pre-copy, and the source page has already been transferred but the destination page hasn't:
- The source page's copy on the migration destination is now stale (the copy on the source node was read, not the migration destination's copy).
- The destination page on the source node is now dirty.

**Tentative answer:** Our interception layer handles this. cuMemcpyDtoD is intercepted. Both source and destination pages are marked dirty in the bitmap. The next pre-copy round will re-transfer them. This may cause extra retransmissions but is correct.

---

## 11. Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| Write-heavy workloads prevent pre-copy convergence | Medium | High for training | Hybrid strategy auto-switches to post-copy |
| CUDA context restore fails on destination (driver version mismatch) | High | Low | Validate driver version compatibility before starting migration |
| Post-copy destination failure loses workload | Critical | Low | R15 parity data enables reconstruction. Document this as a known risk window. |
| Dirty tracking over-estimation wastes bandwidth | Low | Medium | Conservative is correct. Optimize later with known-output-only tracking. |
| Blackout exceeds 500ms for large dirty sets | Medium | Medium | Hybrid strategy minimizes final dirty set. Checkpoint bootstrap reduces initial transfer. |
| Application uses unsupported CUDA features (IPC, UVM) | Medium | Low | Detect during migration planning, fail fast with clear error. |
| Network congestion during migration | Medium | Medium | Bandwidth limiter prevents migration from starving the workload's own network traffic. |
| Race conditions in dirty bitmap (concurrent mark + scan) | High | Medium | Use atomic operations for bitmap words. scan_and_reset uses compare-and-swap. |

---

## Related Documents

- [README.md](README.md) -- R22 feature overview
- [progress.md](progress.md) -- R22 lifecycle tracker
- R10: Memory Tiering (page table, PTE flags, MigrationEngine, dirty/accessed tracking)
- R15: Fault Tolerance (failure detection, checkpoint manager, parity groups, generation-based fencing)
- R19: Network Page Faults (fault handler, SWMR coherency, demand paging)
- R17: Topology-Aware Scheduling (bandwidth estimation, node health, placement scoring)
- Research references:
  - VM live migration techniques (pre-copy, post-copy, hybrid) from academic literature
  - NVIDIA cuda-checkpoint utility (driver 570+, CUDA checkpoint/restore API)
  - PhoenixOS (SOSP '25): concurrent GPU checkpoint/restore with speculative dirty tracking
  - CRIUgpu: transparent GPU container checkpointing via CRIU integration
  - Microsoft GPU-P dirty bit tracking (WDDM driver interface for VRAM dirty page tracking)
