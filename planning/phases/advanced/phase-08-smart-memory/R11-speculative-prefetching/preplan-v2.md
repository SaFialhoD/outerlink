# R11: Speculative Prefetching -- Pre-Plan v2 (Cross-Topic Refined)

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft
**Priority:** HIGH
**Depends On:** R10 (Memory Tiering)
**Supersedes:** preplan.md (v1)

## Purpose

Second-round refinement of R11's pre-plan, incorporating exact Rust structs matching R10 v2's trait definitions, cross-topic integration points from R13 (CUDA Graphs), R14 (Compression), R17 (Multi-Path Routing), R19 (Page Faults), R26 (PTP Timestamps), and R30 (Persistent Kernels). This document is implementation-ready: every struct, algorithm, and formula is specified precisely enough to code from.

---

## 1. Exact Rust Structs

### 1.1 Core Types (Matching R10 v2)

R10 v2 defines the PTE with fields we consume directly:

```rust
/// From R10 v2 PTE -- fields R11 reads and writes
/// access_pattern_type: AccessPatternType (enum in PTE)
/// prefetch_next_vpn_delta: i16 (in PTE)
/// flags: includes COMPRESSED (relevant for R14 decompression on arrival)

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AccessPatternType {
    Sequential  = 0,
    Strided     = 1,
    Iterative   = 2,  // iteration-repeat (same sequence each training step)
    Random      = 3,
    Unknown     = 4,
}
```

### 1.2 PrefetchScheduler

```rust
/// Central coordinator for all prefetch activity on this node.
/// One instance per GPU context.
pub struct PrefetchScheduler {
    /// Pattern detector -- classifies access streams per kernel
    detector: PatternDetector,

    /// Priority queue of pending prefetch requests (min-heap by deadline)
    queue: BinaryHeap<Reverse<PrefetchRequest>>,

    /// Fast cancel lookup: vpn -> request_id for O(1) cancellation
    pending_by_vpn: HashMap<u64, u64>,

    /// Currently in-flight transfers (bounded by max_in_flight)
    in_flight: Vec<InFlightTransfer>,
    max_in_flight: usize,  // 8 for RDMA, 16 for TCP

    /// Bandwidth budget enforcer (token bucket, 20% of link)
    budget: PrefetchBandwidthBudget,

    /// Ring buffer for staging prefetched pages in pinned DRAM
    dram_buffer: PrefetchRingBuffer,

    /// Small VRAM staging window for imminent-need pages
    vram_window: VramPrefetchWindow,

    /// Stats tracker for adaptive behavior
    stats: PrefetchStats,

    /// CUDA Graph prefetch schedules (R13 integration)
    /// Key: graph_id, Value: pre-computed schedule
    graph_schedules: HashMap<u64, GraphPrefetchSchedule>,

    /// Persistent kernel tracking (R30 integration)
    /// Persistent kernels have continuous predictable access patterns
    persistent_kernel_patterns: HashMap<KernelSignature, PersistentKernelPlan>,

    /// Current scheduler state
    state: SchedulerState,

    /// Reference to R10's AccessMonitor for recording pattern metadata
    access_monitor: Arc<dyn AccessMonitor>,

    /// Reference to R10's PageTable for PTE updates
    page_table: Arc<dyn PageTable>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerState {
    /// First 1-3 iterations: learning patterns
    Profiling { iterations_seen: u32 },
    /// Steady state: generating and executing prefetches
    Active,
    /// Misprediction recovery: re-profiling for 1-2 iterations
    Recovering { iterations_remaining: u32 },
    /// Disabled for this workload (chronic misprediction)
    Disabled,
}
```

### 1.3 PrefetchRequest

```rust
/// A single request to move a page from one tier to another speculatively.
#[derive(Debug, Clone)]
pub struct PrefetchRequest {
    /// Unique request ID for cancellation tracking
    id: u64,

    /// Virtual page number (R10 page table key)
    vpn: u64,

    /// Where the page currently lives
    source_tier: Tier,

    /// Where we want it (usually LocalVram)
    target_tier: Tier,

    /// Source node ID (for topology-aware source selection, R17)
    source_node: NodeId,

    /// Absolute deadline: page must be ready by this timestamp.
    /// Uses PTP-synchronized clock (R26) for cross-node accuracy.
    deadline_ns: u64,

    /// Prediction confidence [0.0, 1.0]
    /// Determines priority and cancellation order
    confidence: f32,

    /// Which pattern generated this request
    pattern_source: PrefetchSource,

    /// Size in bytes (normally 65536 for one 64KB page)
    size_bytes: u32,

    /// Is the source page compressed? (R14 integration)
    /// If true, must decompress on arrival before promoting to VRAM
    source_compressed: bool,

    /// Preferred network path (R17 integration)
    /// None = let transport layer decide; Some = use specific path
    preferred_path: Option<PathId>,

    /// Can this request be canceled mid-transfer?
    /// RDMA posts cannot be canceled; queued requests can.
    cancelable: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum PrefetchSource {
    /// Sequential stride detector predicted this
    StridePredictor { stride_delta: i16 },
    /// Iteration replay predicted this
    IterationReplay { iteration_offset: u32 },
    /// CUDA Graph DAG provided exact schedule (R13)
    GraphSchedule { graph_id: u64, node_idx: u32 },
    /// Persistent kernel continuous access (R30)
    PersistentKernel { kernel_sig: KernelSignature },
    /// Phase-aware predictor (forward/backward/optimizer)
    PhasePredictor { phase: TrainingPhase },
}

impl Ord for PrefetchRequest {
    /// Priority ordering: earliest deadline first.
    /// Tie-break: higher confidence first.
    fn cmp(&self, other: &Self) -> Ordering {
        self.deadline_ns.cmp(&other.deadline_ns)
            .then(other.confidence.partial_cmp(&self.confidence)
                .unwrap_or(Ordering::Equal))
    }
}
```

### 1.4 AccessPattern (Pattern Detector State)

```rust
/// Per-kernel access pattern tracker.
/// Indexed by KernelSignature (hash of function pointer + grid dims).
pub struct KernelAccessPattern {
    /// Kernel identity
    signature: KernelSignature,

    /// Ordered sequence of VPNs accessed in the last observed invocation
    last_access_sequence: Vec<u64>,

    /// Classified pattern type
    pattern: AccessPatternType,

    /// For Sequential/Strided: the detected delta between consecutive VPNs
    stride_delta: i16,

    /// For Iterative: the full VPN sequence that repeats
    replay_sequence: Option<Vec<u64>>,

    /// Confidence in the current classification [0.0, 1.0]
    confidence: f32,

    /// Number of invocations observed
    invocation_count: u64,

    /// Average execution time in nanoseconds (for distance calculation)
    avg_execution_ns: u64,

    /// Exponential moving average of execution time (alpha=0.1)
    ema_execution_ns: u64,

    /// Number of consecutive correct predictions
    streak: u32,

    /// State machine state for pattern detection
    detector_state: DetectorState,
}

/// State machine for the stride/pattern detector.
/// Modeled after CPU RPT (Reference Prediction Table).
#[derive(Debug, Clone)]
pub enum DetectorState {
    /// First invocation: recording baseline
    Initial,

    /// Second invocation: computed first delta, waiting to confirm
    Transient { last_delta: i16 },

    /// Third+ invocation: delta confirmed, actively predicting
    Steady { confirmed_delta: i16, mispredict_count: u32 },

    /// Iteration-level: detected that the full sequence repeats
    IterationLocked { sequence_hash: u64 },

    /// No pattern found after sufficient observations
    NoPattern,
}

/// Iteration-level pattern tracker.
/// Detects repeating kernel sequences across training iterations.
pub struct IterationTracker {
    /// Kernel launch sequence for the current iteration
    current_sequence: Vec<KernelSignature>,

    /// Kernel launch sequence for the previous iteration (for comparison)
    previous_sequence: Vec<KernelSignature>,

    /// Hash of the confirmed repeating sequence (if detected)
    locked_sequence_hash: Option<u64>,

    /// How many iterations matched the locked sequence
    match_count: u32,

    /// Position within current iteration (index into sequence)
    current_position: usize,

    /// Iteration boundaries detected via cuStreamSynchronize / cuCtxSynchronize
    iteration_boundary_count: u64,
}
```

### 1.5 PrefetchBuffer (Ring Buffer + VRAM Window)

```rust
/// Pinned DRAM ring buffer for staging prefetched pages.
/// Pages arrive here from network, then promote to VRAM when needed.
pub struct PrefetchRingBuffer {
    /// Base address of the pinned DRAM allocation
    base_ptr: *mut u8,

    /// Total size in bytes (default: 2 GB)
    capacity_bytes: usize,

    /// Number of 64KB slots
    num_slots: u32,

    /// Write cursor: next slot to receive incoming data
    write_cursor: AtomicU32,

    /// Read cursor: next slot to promote to VRAM
    read_cursor: AtomicU32,

    /// Per-slot metadata
    slots: Vec<RingSlot>,

    /// High-water mark for pressure detection (default: 0.8)
    pressure_threshold: f32,
}

#[derive(Debug)]
pub struct RingSlot {
    /// Which VPN this slot holds data for (0 = empty)
    vpn: AtomicU64,

    /// State of this slot
    state: AtomicU8,  // 0=Empty, 1=Receiving, 2=Ready, 3=Promoting

    /// Timestamp when data arrived (for staleness detection)
    arrival_ns: AtomicU64,

    /// Deadline: when the GPU needs this page
    deadline_ns: u64,
}

/// Small reserved VRAM region for pages needed within 1-2 kernel launches.
pub struct VramPrefetchWindow {
    /// CUDA device pointer to reserved VRAM region
    device_ptr: CUdeviceptr,

    /// Size in bytes (default: 256 MB, configurable)
    capacity_bytes: usize,

    /// Number of 64KB slots
    num_slots: u32,

    /// Slot allocation bitmap (1 = occupied)
    bitmap: Vec<AtomicU64>,

    /// VPN -> slot index mapping for fast lookup
    vpn_to_slot: HashMap<u64, u32>,

    /// LRU list for eviction when window is full
    lru: VecDeque<u32>,
}

/// Buffer sizing formula implementation.
/// buffer_size = prefetch_distance_pages * PAGE_SIZE * pipeline_depth
pub fn calculate_dram_buffer_size(
    pages_per_kernel: usize,
    kernels_ahead: usize,
    pipeline_depth: usize,  // typically 2 (double-buffering)
) -> usize {
    let page_size = 65536; // 64KB
    pages_per_kernel * kernels_ahead * pipeline_depth * page_size
}

/// VRAM window sizing: enough for 1-2 kernels' worth of pages.
/// Capped at max_vram_fraction of total VRAM.
pub fn calculate_vram_window_size(
    pages_per_kernel: usize,
    kernels_in_window: usize,  // typically 2
    total_vram_bytes: usize,
    max_vram_fraction: f32,    // typically 0.02 (2%)
) -> usize {
    let ideal = pages_per_kernel * kernels_in_window * 65536;
    let max_allowed = (total_vram_bytes as f64 * max_vram_fraction as f64) as usize;
    ideal.min(max_allowed)
}
```

### 1.6 InFlightTransfer and Stats

```rust
pub struct InFlightTransfer {
    request: PrefetchRequest,
    /// Which ring buffer slot the data is writing into
    target_slot: u32,
    /// Transfer start timestamp (PTP-synced, R26)
    started_ns: u64,
    /// Transport-layer transfer handle for status queries
    transfer_handle: TransferHandle,
}

/// Prefetch effectiveness tracking.
/// Used for adaptive budget and confidence calibration.
pub struct PrefetchStats {
    /// Rolling window stats (last 1000 prefetches)
    window_size: usize,

    total_issued: u64,
    hits: u64,           // page used before eviction
    misses: u64,         // page evicted unused
    late: u64,           // page arrived after demand fetch already happened
    canceled: u64,       // aborted before completion
    demand_stalls: u64,  // GPU stalled (prefetch failed to prevent)

    /// Derived metrics (recomputed every 100 prefetches)
    hit_rate: f32,       // hits / (hits + misses + late)
    coverage: f32,       // 1.0 - (demand_stalls / total_page_accesses)
    timeliness: f32,     // hits / (hits + late)
    bandwidth_waste_ratio: f32,  // (misses * PAGE_SIZE) / total_bytes_prefetched

    /// Per-source breakdown (which predictor is working)
    per_source_hit_rate: HashMap<PrefetchSourceKind, f32>,

    /// For adaptive budget: EWMA of hit rate (alpha = 0.05)
    ewma_hit_rate: f32,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum PrefetchSourceKind {
    Stride,
    IterationReplay,
    GraphSchedule,
    PersistentKernel,
    PhasePredictor,
}
```

---

## 2. Pattern Detection Algorithm -- Exact State Machine

### 2.1 Per-Kernel Stride Detector (RPT-Based)

Modeled after CPU Reference Prediction Tables, operating at page granularity:

```
Input: stream of (kernel_signature, accessed_vpn_list) events from AccessMonitor

For each kernel invocation with signature S:
  Let current_vpns = sorted list of VPNs accessed
  Let prev_vpns = KernelAccessPattern[S].last_access_sequence

  CASE detector_state:

    Initial:
      // First time seeing this kernel
      Store current_vpns as last_access_sequence
      Transition -> Transient { last_delta = compute_dominant_delta(current_vpns) }

    Transient { last_delta }:
      new_delta = compute_dominant_delta(current_vpns)
      IF new_delta == last_delta AND last_delta != 0:
        // Delta confirmed on second observation
        pattern = if last_delta == 1 { Sequential } else { Strided }
        confidence = 0.6
        Write PTE.access_pattern_type for all pages in current_vpns
        Write PTE.prefetch_next_vpn_delta = last_delta for all pages
        Transition -> Steady { confirmed_delta = last_delta, mispredict_count = 0 }
      ELSE IF sequences_match(current_vpns, prev_vpns, threshold=0.95):
        // Same VPN set repeats -- iteration pattern
        pattern = Iterative
        confidence = 0.7
        Store replay_sequence = current_vpns
        Transition -> IterationLocked { sequence_hash = hash(current_vpns) }
      ELSE:
        // Different delta, try once more
        Transition -> Transient { last_delta = new_delta }

    Steady { confirmed_delta, mispredict_count }:
      new_delta = compute_dominant_delta(current_vpns)
      IF new_delta == confirmed_delta:
        streak += 1
        confidence = min(0.99, 0.6 + 0.02 * streak)
        mispredict_count = 0
        // Generate prefetch requests for next invocation
        EMIT prefetch requests for vpns: last_vpn + confirmed_delta * 1..prefetch_depth
      ELSE:
        mispredict_count += 1
        streak = 0
        confidence *= 0.7
        IF mispredict_count >= 3:
          Transition -> Initial  // re-learn
          Cancel all pending prefetches for this kernel

    IterationLocked { sequence_hash }:
      IF hash(current_vpns) == sequence_hash:
        match_count += 1
        confidence = min(0.99, 0.7 + 0.03 * match_count)
        // Prefetch entire next-iteration sequence
        EMIT prefetch requests for replay_sequence at appropriate deadlines
      ELSE:
        // Sequence changed (learning rate change, epoch boundary, etc.)
        IF match_count > 10:
          // Was stable, try to re-lock
          Transition -> Transient { last_delta = 0 }
        ELSE:
          Transition -> Initial

    NoPattern:
      // Re-check every 100 invocations in case pattern emerges
      IF invocation_count % 100 == 0:
        Transition -> Initial
```

### 2.2 compute_dominant_delta

```rust
/// Given a sorted list of VPNs, find the most common delta between
/// consecutive accesses. Returns 0 if no dominant delta found.
fn compute_dominant_delta(vpns: &[u64]) -> i16 {
    if vpns.len() < 3 { return 0; }

    // Count deltas between consecutive VPNs
    let mut delta_counts: HashMap<i64, u32> = HashMap::new();
    for window in vpns.windows(2) {
        let delta = window[1] as i64 - window[0] as i64;
        if delta.abs() <= i16::MAX as i64 {
            *delta_counts.entry(delta).or_insert(0) += 1;
        }
    }

    // Find most common delta
    let total_pairs = vpns.len() - 1;
    if let Some((&delta, &count)) = delta_counts.iter().max_by_key(|(_, c)| *c) {
        // Dominant if >60% of pairs share this delta
        if count as f64 / total_pairs as f64 > 0.6 {
            return delta.clamp(i16::MIN as i64, i16::MAX as i64) as i16;
        }
    }
    0 // no dominant delta
}
```

### 2.3 Iteration-Level Detection

The `IterationTracker` detects when the entire kernel launch sequence repeats:

```
On each cuStreamSynchronize / cuCtxSynchronize:
  Mark iteration boundary
  Compare current_sequence with previous_sequence:
    IF jaccard_similarity(current_sequence, previous_sequence) > 0.95:
      locked_sequence_hash = hash(current_sequence)
      Enter IterationLocked state for ALL kernels in the sequence
    previous_sequence = current_sequence
    current_sequence = []
    current_position = 0

On each cuLaunchKernel:
  Append kernel signature to current_sequence
  current_position += 1
  IF locked_sequence_hash is set:
    // We know exactly what kernel comes next
    next_kernel = previous_sequence[current_position + 1]
    EMIT prefetch requests for next_kernel's known page set
```

---

## 3. Prefetch Distance Calculator

### 3.1 Formula

```
distance_ns = transfer_time_ns + decompression_time_ns + staging_overhead_ns + safety_margin_ns

Where:
  transfer_time_ns = (num_pages * PAGE_SIZE) / effective_bandwidth_bytes_per_ns
  decompression_time_ns = if source_compressed {
      num_pages * per_page_decompress_ns   // R14: ~3-8us per 64KB page for LZ4
  } else { 0 }
  staging_overhead_ns = DMA_SETUP_NS + PAGE_TABLE_UPDATE_NS  // ~5000 ns typical
  safety_margin_ns = transfer_time_ns * safety_multiplier

  safety_multiplier:
    Initial: 1.5x
    After 10 stall-free iterations: decrease by 0.05 (min 1.1x)
    On any stall: increase by 0.2 (max 2.0x)
```

### 3.2 Implementation

```rust
pub struct PrefetchDistanceCalculator {
    /// Effective bandwidth in bytes per nanosecond for each transport
    /// Populated from R10's migration engine bandwidth reporting
    bandwidth_by_transport: HashMap<TransportKind, f64>,

    /// Per-page decompression cost estimate (R14 integration)
    /// Updated dynamically from measured decompress times
    decompress_ns_per_page: u64,  // 0 if compression disabled

    /// Fixed overhead per transfer batch
    staging_overhead_ns: u64,  // ~5000 ns

    /// Adaptive safety multiplier
    safety_multiplier: f64,

    /// PTP clock reference for cross-node deadline computation (R26)
    ptp_clock: Arc<dyn PtpClock>,
}

impl PrefetchDistanceCalculator {
    /// Calculate how many nanoseconds before the kernel's expected start
    /// we need to begin prefetching its page set.
    pub fn calculate_lead_time(
        &self,
        num_pages: usize,
        source_compressed: bool,
        transport: TransportKind,
    ) -> u64 {
        let bw = self.bandwidth_by_transport[&transport];
        let transfer_ns = (num_pages as f64 * 65536.0 / bw) as u64;

        let decompress_ns = if source_compressed {
            num_pages as u64 * self.decompress_ns_per_page
        } else {
            0
        };

        let base = transfer_ns + decompress_ns + self.staging_overhead_ns;
        (base as f64 * self.safety_multiplier) as u64
    }

    /// Convert lead time into a PTP-synced absolute deadline.
    /// kernel_expected_start_ns is the PTP timestamp when we expect
    /// the consuming kernel to launch.
    pub fn compute_deadline(
        &self,
        kernel_expected_start_ns: u64,
        lead_time_ns: u64,
    ) -> u64 {
        kernel_expected_start_ns.saturating_sub(lead_time_ns)
    }

    /// Adapt safety multiplier based on observed stall/success.
    pub fn feedback(&mut self, stall_occurred: bool) {
        if stall_occurred {
            self.safety_multiplier = (self.safety_multiplier + 0.2).min(2.0);
        } else {
            self.safety_multiplier = (self.safety_multiplier - 0.05).max(1.1);
        }
    }
}
```

### 3.3 Concrete Examples

| Scenario | Pages | Transport | Transfer | Decomp | Overhead | Safety (1.3x) | Total Lead |
|---|---|---|---|---|---|---|---|
| Transformer layer weights (remote) | 150 | RDMA 100Gbps | 873 us | 0 | 5 us | 263 us | 1141 us |
| Transformer layer weights (compressed, remote) | 150 | RDMA 100Gbps | 524 us* | 750 us | 5 us | 384 us | 1663 us |
| Activation checkpoint (local DRAM) | 200 | PCIe 4.0 | 512 us | 0 | 5 us | 155 us | 672 us |
| NVMe -> DRAM -> VRAM chain | 100 | NVMe+PCIe | 1851 us | 0 | 10 us | 558 us | 2419 us |

*Compressed at 0.6x ratio: only 60% of bytes transferred, but decompress cost added.

---

## 4. Integration with R13 CUDA Graph Schedule

### 4.1 Why CUDA Graphs Are Special

When an application calls `cuGraphLaunch`, the entire computation DAG is known ahead of time. This eliminates prediction entirely -- we have a perfect schedule.

### 4.2 Graph Capture Flow

```
Application calls cuGraphInstantiate(graph):
  OuterLink intercepts, receives CUgraph handle

  1. Walk the graph DAG via cuGraphGetNodes / cuGraphGetEdges
  2. For each kernel node:
     - Extract kernel signature and arguments
     - Determine input/output VPN sets (from kernel arg pointers)
     - Record estimated execution time (from profiling DB or first-run measurement)
  3. Build GraphPrefetchSchedule:
     - Topological sort of graph nodes
     - For each node, compute: pages_needed, pages_produced
     - Compute absolute prefetch deadlines relative to graph launch time
  4. Store schedule in graph_schedules[graph_id]
```

### 4.3 GraphPrefetchSchedule Struct

```rust
/// Pre-computed prefetch schedule for an entire CUDA graph.
/// No prediction needed -- this is an exact plan.
pub struct GraphPrefetchSchedule {
    graph_id: u64,

    /// Ordered list of graph nodes with their memory requirements
    nodes: Vec<GraphNodePlan>,

    /// Total pages to prefetch for the entire graph
    total_pages: usize,

    /// Pre-computed: when to start each prefetch batch relative to graph launch
    /// (offset_ns_from_launch, Vec<vpn>)
    prefetch_timeline: Vec<(u64, Vec<u64>)>,

    /// Whether this schedule has been validated (run at least once)
    validated: bool,
}

pub struct GraphNodePlan {
    node_index: u32,
    kernel_signature: KernelSignature,
    /// VPNs that must be in VRAM before this node executes
    required_vpns: Vec<u64>,
    /// VPNs that this node produces (can be evicted after dependents consume)
    produced_vpns: Vec<u64>,
    /// Estimated execution time (ns)
    estimated_duration_ns: u64,
    /// Cumulative offset from graph start (computed from DAG critical path)
    start_offset_ns: u64,
}
```

### 4.4 Graph Launch Execution

```
On cuGraphLaunch(graph_exec, stream):
  IF graph_schedules.contains(graph_id):
    schedule = graph_schedules[graph_id]
    launch_time = ptp_clock.now()

    // Issue all prefetch requests with absolute deadlines
    for (offset_ns, vpns) in schedule.prefetch_timeline:
      deadline = launch_time + offset_ns
      for vpn in vpns:
        if page_not_in_vram(vpn):
          emit PrefetchRequest {
            vpn, deadline, confidence: 1.0,
            pattern_source: GraphSchedule { graph_id, node_idx },
            ..
          }

    // Graph schedules get priority 1 (urgent) since confidence is 1.0
    // and we know the exact timing
  ELSE:
    // First launch: profile, build schedule, fall back to demand paging
    profile_graph_execution(graph_exec)
```

### 4.5 Persistent Kernel Integration (R30)

Persistent kernels run continuously and access memory in predictable loops. They are ideal prefetch targets because their access pattern never changes once launched.

```rust
pub struct PersistentKernelPlan {
    kernel_signature: KernelSignature,
    /// The repeating access cycle (VPNs accessed per loop iteration)
    cycle_vpns: Vec<u64>,
    /// Cycle period in nanoseconds
    cycle_period_ns: u64,
    /// Current position in the cycle (tracked via access monitoring)
    current_cycle_position: usize,
}
```

When a persistent kernel is detected (kernel duration >> typical kernel, continuous access pattern):
1. Profile the first few cycles to learn `cycle_vpns` and `cycle_period_ns`
2. Prefetch one cycle ahead continuously
3. Confidence is 0.99 (persistent kernels are maximally predictable)

---

## 5. Buffer Management

### 5.1 DRAM Ring Buffer Design

```
Memory Layout (2 GB default, 32768 slots of 64KB):

  [Slot 0][Slot 1][Slot 2]...[Slot 32767]
     ^                           ^
     |                           |
  read_cursor              write_cursor

  Network RDMA writes land at: base_ptr + (write_cursor * 65536)
  VRAM promotion reads from:   base_ptr + (read_cursor * 65536)

  Wrap-around: cursors are u32, modulo num_slots

  Slot lifecycle:
    Empty -> Receiving (network writing) -> Ready (data complete) -> Promoting (DMA to VRAM) -> Empty
```

### 5.2 VRAM Window Management

```
VRAM Window (256 MB default, 4096 slots):

  Allocation: cuMemAlloc at startup, reserved from VRAM pool
  Organization: flat array of 64KB slots with bitmap allocator

  Promotion policy:
    - Page enters VRAM window when deadline is within 2 kernel launches
    - Pages that are already in the R10 VRAM tier skip the window entirely
    - If window is full: evict LRU page back to DRAM ring buffer

  Interaction with R10:
    - Pages in the VRAM window are registered in R10's page table as Tier::LocalVram
    - R10 gives them short eviction immunity (immunity_ns = distance_calculator.lead_time)
    - On promotion from window to "real" VRAM: page table entry updated, window slot freed
```

### 5.3 Buffer Sizing Per Tier

| Tier | Default Size | Formula | When to Resize |
|---|---|---|---|
| DRAM ring buffer | 2 GB | `pages_per_iteration * PAGE_SIZE * 2` (double-buffer) | If buffer pressure > 80% for 10+ iterations |
| VRAM window | 256 MB (1% of 24GB) | `pages_per_kernel * 2 * PAGE_SIZE` | If VRAM utilization > 95%, shrink to 128 MB |
| VRAM window (min) | 64 MB | Always reserved | Never shrink below this |
| VRAM window (max) | 2% of total VRAM | Hard cap | Never exceed |

### 5.4 Decompression on Arrival (R14 Integration)

When `source_compressed` is true in a PrefetchRequest:

```
1. Network writes compressed data to DRAM ring buffer slot
2. Before promoting to VRAM, decompress in-place:
   - LZ4 decompress: compressed_slot -> decompressed buffer (CPU, ~3us per 64KB)
   - Or: transfer compressed to VRAM, decompress on GPU (if GPU has spare compute)
3. Clear COMPRESSED flag in PTE after decompression

Adaptive decision per-link (from R14):
  IF link_bandwidth < 50 Gbps: compress (save transfer time)
  IF link_bandwidth >= 50 Gbps AND compression_ratio < 0.7: compress
  IF link_bandwidth >= 50 Gbps AND compression_ratio >= 0.7: skip compression
```

---

## 6. Bandwidth Budget Enforcement

### 6.1 Token Bucket Implementation

R10 v2 establishes a 70:20:10 split (demand:prefetch:writeback). R11 owns the 20% prefetch bucket.

```rust
/// Token bucket rate limiter for prefetch bandwidth.
/// Refills at 20% of the total link bandwidth.
pub struct PrefetchBandwidthBudget {
    /// Available bytes for prefetching (atomic for lock-free check)
    tokens: AtomicI64,

    /// Maximum tokens (burst capacity)
    /// = 20% of link bandwidth * refill_interval
    max_tokens: i64,

    /// Refill rate: bytes per nanosecond
    /// = total_link_bw_bytes_per_ns * 0.20
    refill_rate: f64,

    /// Last refill timestamp (PTP-synced)
    last_refill_ns: AtomicU64,

    /// Refill interval in nanoseconds (100us = 100_000 ns)
    refill_interval_ns: u64,

    /// Adaptive: current effective share [0.05, 0.30]
    /// Adjusted based on hit rate from PrefetchStats
    effective_share: AtomicU32,  // stored as share * 1000 (e.g., 200 = 0.20)

    /// When demand is idle, prefetch can steal up to this much extra
    demand_idle_steal_max: f64,  // 0.50 (can use up to 70% total when demand idle)
}

impl PrefetchBandwidthBudget {
    /// Try to consume `bytes` tokens. Returns true if allowed.
    /// Non-blocking: if insufficient tokens, request is queued, not blocked.
    pub fn try_consume(&self, bytes: u64) -> bool {
        self.refill_if_needed();
        let current = self.tokens.load(Ordering::Relaxed);
        if current >= bytes as i64 {
            // CAS loop for atomic decrement
            self.tokens.fetch_sub(bytes as i64, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Called by R10's migration engine to report demand fetch idle status.
    /// When demand is idle, prefetch budget temporarily expands.
    pub fn notify_demand_idle(&self, idle: bool) {
        if idle {
            // Steal up to demand_idle_steal_max from demand bucket
            let extra = (self.max_tokens as f64 * self.demand_idle_steal_max) as i64;
            self.tokens.fetch_add(extra, Ordering::Relaxed);
        }
    }

    /// Adapt budget share based on prefetch hit rate.
    /// Called every 100 prefetches by PrefetchStats.
    pub fn adapt(&self, hit_rate: f32) {
        let new_share = match hit_rate {
            r if r > 0.90 => 300,  // 30% -- reward excellent prediction
            r if r > 0.70 => 200,  // 20% -- maintain default
            r if r > 0.50 => 100,  // 10% -- reduce for mediocre prediction
            _ => 50,               //  5% -- minimal for poor prediction
        };
        self.effective_share.store(new_share, Ordering::Relaxed);
        // Recalculate max_tokens on next refill
    }
}
```

### 6.2 Multi-Path Routing Integration (R17)

R17 provides multi-path routing where prefetch can use idle links:

```
When issuing a PrefetchRequest:
  1. Query R17 for available paths to source_node
  2. Prefer paths with lowest current utilization
  3. Set preferred_path in PrefetchRequest if a clearly idle path exists
  4. R17 can split large prefetch batches across multiple paths

Key: prefetch traffic is LOWEST priority on all paths.
     R17 preempts prefetch for demand/writeback traffic on any path.
```

### 6.3 Topology-Aware Source Selection (R17)

When a page exists in multiple tiers across the cluster:

```
Example: Page P exists in:
  - Node A, DRAM tier (2 hops away)
  - Node B, VRAM tier (1 hop away)
  - Node C, NVMe tier (1 hop away)

Source selection priority:
  1. Closest node with page in fastest tier: Node B (VRAM, 1 hop)
  2. Closest node regardless of tier: Node B or C (1 hop)
  3. Any node with page in VRAM: Node B
  4. Fallback: any node that has the page

R17 provides the topology graph and hop costs.
The PrefetchScheduler queries R17 to set source_node in each PrefetchRequest.
```

---

## 7. R19 Page Fault Safety Net

R11 prefetching prevents most page faults. R19 (Network Page Faults) is the fallback for mispredictions:

```
Normal flow (R11 working):
  Kernel K needs page P
  R11 predicted this -> P is already in VRAM
  Kernel runs without stall

Misprediction flow (R19 catches it):
  Kernel K needs page P
  R11 did NOT prefetch P (wrong prediction or disabled)
  GPU accesses P -> R19 page fault triggers
  R19 fetches P on demand (higher latency, ~50-100us)
  Kernel resumes

R11 observes the R19 fault as a "demand_stall" event:
  stats.demand_stalls += 1
  Triggers confidence reduction and possible re-profiling
```

This means R11 can be aggressive with prefetching -- the worst case for a wrong prediction is NOT data corruption, it is falling back to demand paging (R19). R11 can never make things worse than having no prefetching at all.

---

## 8. Open Questions (Updated from v1)

### Resolved by Cross-Topic Findings

1. **Page table scan overhead at 375K entries?** -- RESOLVED. R10 v2 provides `AccessMonitor::get_access_pattern()` and `PageTable::scan_by_flags()` which are O(matched) not O(total). No full scan needed.

2. **Bandwidth split?** -- RESOLVED. R10 v2 mandates 70:20:10 (demand:prefetch:writeback). Token bucket implementation defined above.

3. **CUDA Graph support?** -- RESOLVED. Section 4 defines exact integration. Graph DAG gives perfect schedule, confidence=1.0.

4. **How does R14 compression affect prefetch?** -- RESOLVED. Compressed pages require decompression on arrival. Distance calculator accounts for decompression time. Adaptive compression decision per-link.

### Still Open

5. **RDMA queue pair depth on ConnectX-5 for concurrent transfers?** Need hardware benchmarking. Preliminary max_in_flight=8 for RDMA.

6. **Optimal VRAM prefetch window size for different model sizes?** Empirical. Default 256 MB, tune during integration testing.

7. **Batch vs individual prefetch requests?** Will benchmark both. Hypothesis: batch for sequential (coalesce adjacent pages), individual for strided.

8. **PCIe contention between NIC and GPU during prefetch-to-VRAM promotion?** Measure on Pedro's hardware (RTX 3090 + ConnectX-5 on same PCIe root).

---

## 9. Success Criteria (Updated)

| Metric | Target | How Measured |
|---|---|---|
| GPU stall rate (demand faults) | <5% of kernel launches | PrefetchStats.demand_stalls / total_kernels |
| Prefetch hit rate | >90% | PrefetchStats.hit_rate |
| Prefetch timeliness | >90% | PrefetchStats.timeliness |
| Bandwidth waste | <15% | PrefetchStats.bandwidth_waste_ratio |
| CUDA Graph prefetch coverage | 100% (all pages pre-scheduled) | graph_schedule total_pages vs demand faults |
| Profiling overhead per iteration | <2 ms | Wall-clock measurement |
| Prefetch scheduling latency | <50 us (prediction to transfer post) | Timestamp delta |
| Safety multiplier convergence | <1.2x after 50 steady iterations | distance_calculator.safety_multiplier |

---

## 10. Testing Strategy Additions (v2)

### Cross-Topic Integration Tests

| Test | What It Validates |
|---|---|
| Prefetch + R14 compression | Compressed pages arrive, decompress correctly, promote to VRAM |
| Prefetch + R13 CUDA Graph | Graph launch triggers pre-computed schedule, 100% hit rate |
| Prefetch + R17 multi-path | Prefetch uses idle paths, yields to demand traffic on congested paths |
| Prefetch + R19 fallback | Misprediction triggers R19 page fault, kernel completes correctly |
| Prefetch + R26 PTP timing | Cross-node prefetch deadlines are accurate within 1 us |
| Prefetch + R30 persistent kernel | Continuous prefetch keeps up with persistent kernel's access cycle |
| Prefetch + R12 dedup | Deduped pages only prefetched once for canonical copy, references follow |

---

## Related Documents

- [R11 preplan.md](./preplan.md) -- v1 pre-plan (superseded by this document)
- [research/01-existing-prefetching-systems.md](./research/01-existing-prefetching-systems.md)
- [research/02-access-pattern-profiling.md](./research/02-access-pattern-profiling.md)
- [research/03-prefetch-scheduling.md](./research/03-prefetch-scheduling.md)
- R10 v2 preplan -- AccessMonitor trait, PageTable trait, PTE fields
- R13 CUDA Graph capture -- graph DAG extraction for perfect prefetch schedules
- R14 Transport Compression -- decompression cost on prefetch arrival
- R17 Multi-Path Routing -- idle link utilization, topology-aware source selection
- R19 Network Page Faults -- safety net for mispredictions
- R26 PTP Timestamps -- cross-node deadline accuracy
- R30 Persistent Kernels -- continuous predictable access patterns
