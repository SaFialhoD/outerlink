# R15: Fault Tolerance & Erasure Coding -- Pre-Plan v2 (Refined)

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft
**Priority:** MEDIUM (Phase 9)
**Supersedes:** preplan.md (v1)

## Purpose

Second-round refinement of R15's fault tolerance design. This document resolves open questions from v1, provides exact Rust structs and trait implementations, defines concrete algorithms with step-by-step flows, and specifies precise integration points with R10 (memory tiering), R12 (dedup), R17 (topology/failure detection), R19 (coherency), and R22 (live migration).

---

## 1. Resolved Open Questions (from v1)

### Q1: Parity update frequency
**Resolved:** Configurable per data class via `ParityPolicy`.
- **Immutable pages** (model weights, dedup canonical): Parity computed once at allocation, never updated.
- **Mutable hot pages** (KV cache, activations): Asynchronous parity -- write returns immediately, parity update queued. Vulnerability window of ~1-5ms (one parity batch cycle).
- **Training state** (optimizer, gradients): Parity updated at checkpoint boundaries only. Between checkpoints, these pages are protected by checkpoint recovery, not live parity.

### Q2: Quorum for 2-node clusters
**Resolved:** Use NVMe-backed lease file as tiebreaker. Each node attempts to acquire an exclusive lease on a shared NVMe path (or local NVMe with cross-mounted partition). The lease holder is the surviving partition. Lease TTL = 5 seconds. If no shared NVMe exists, fall back to "last writer wins" with manual reconciliation flag.

### Q3: Interaction with CUDA Unified Memory
**Resolved:** OuterLink does not use CUDA UVM. All memory management is explicit through our interception layer. UVM page faults cannot conflict with recovery because we control the entire fault path via R19's SWMR protocol.

### Q4: Checkpoint format compatibility
**Resolved:** Dual format. Internal format is a compact binary layout optimized for in-memory storage and RDMA transfer. Export/import to DCP format for PyTorch interoperability. The internal format stores per-shard metadata (tensor shapes, dtypes, shard boundaries) enabling resharding on load.

### Q5: Cost of protecting everything
**Resolved:** Tiered protection policy. Not all pages are erasure-coded. Protection level is assigned by `DataClass`:
- `Critical` (model weights): RS parity + checkpoint
- `Recoverable` (KV cache, activations): XOR parity only (can be recomputed)
- `Transient` (temporary buffers): No parity (loss is acceptable, application retries)
- `Checkpoint` (optimizer state): Protected by checkpoint system, no live parity

### Q6: Recovery under memory pressure
**Resolved:** Eviction-aware recovery. Before reconstruction, the recovery engine queries R10's ARC cache for eviction candidates. Cold pages on surviving nodes are demoted to NVMe tier to make room. If no room exists even after eviction, reconstruction proceeds to NVMe tier directly (slower but always possible).

---

## 2. Rust Structs and Types

### 2.1 Core Erasure Coding Types

```rust
/// Erasure coding configuration for a parity group
#[derive(Clone, Debug)]
pub struct ErasureConfig {
    /// Number of data fragments
    pub k: u8,
    /// Number of parity fragments
    pub m: u8,
    /// Coding scheme
    pub scheme: CodingScheme,
}

#[derive(Clone, Debug)]
pub enum CodingScheme {
    /// Simple XOR parity (m must be 1)
    Xor,
    /// Reed-Solomon via ISA-L
    ReedSolomon,
}

/// A parity group is a set of pages protected together
#[derive(Clone, Debug)]
pub struct ParityGroup {
    /// Unique group identifier (matches PTE.parity_group_id)
    pub group_id: u32,
    /// Erasure coding configuration
    pub config: ErasureConfig,
    /// VPNs of data pages in this group (ordered, index = fragment index)
    pub data_pages: Vec<Vpn>,
    /// Locations of parity fragments (node_id, memory address)
    pub parity_locations: Vec<ParityLocation>,
    /// Last time parity was verified/updated
    pub last_parity_update: Instant,
    /// Protection policy
    pub policy: ParityPolicy,
}

#[derive(Clone, Debug)]
pub struct ParityLocation {
    pub node_id: NodeId,
    /// Physical address in the node's DRAM (for hot parity)
    /// or NVMe offset (for cold parity)
    pub tier: ParityTier,
    pub address: u64,
    pub size: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum ParityTier {
    /// Parity stored in partner node's DRAM (fast recovery)
    RemoteDram,
    /// Parity stored on NVMe (persistent, slower recovery)
    Nvme,
    /// Both DRAM and NVMe (critical data)
    Both,
}

#[derive(Clone, Copy, Debug)]
pub enum ParityPolicy {
    /// Parity updated synchronously on every write (strongest, highest overhead)
    Synchronous,
    /// Parity updated in background batches (default for mutable data)
    AsyncBatch { batch_interval_ms: u32 },
    /// Parity updated only at checkpoint boundaries
    CheckpointOnly,
    /// No parity (transient data)
    None,
}

/// Data classification determines protection level
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DataClass {
    /// Model weights, embedding tables -- immutable after load
    Critical,
    /// KV cache, activations -- can be recomputed
    Recoverable,
    /// Temporary scratch buffers -- loss is acceptable
    Transient,
    /// Optimizer state, gradients -- protected by checkpoints
    Checkpoint,
}
```

### 2.2 Integration with R10 Page Table Entry

```rust
/// Extended PTE fields for fault tolerance (integrated with R10's PTE)
///
/// R10 v2 defines:
///   - parity_group_id: u32 in PTE
///   - flags include PARITY_VALID
///
/// We use these fields directly.

bitflags! {
    pub struct PageFlags: u32 {
        // ... existing R10 flags ...
        const PARITY_VALID      = 1 << 8;
        const PARITY_STALE      = 1 << 9;   // Page modified since last parity update
        const CHECKPOINT_DIRTY  = 1 << 10;  // Page modified since last checkpoint
        const RECOVERY_IN_PROGRESS = 1 << 11; // Page is being reconstructed
        const DEDUP_CANONICAL   = 1 << 12;  // From R12: this is the canonical dedup copy
    }
}
```

### 2.3 Failure Detection Types

```rust
/// Node health state in the cluster membership
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NodeState {
    /// Normal operation
    Active,
    /// Suspected failure, not yet confirmed (phi threshold approaching)
    Suspected { since: Instant, phi: f64 },
    /// Confirmed dead, fencing in progress
    Fencing { generation: u64 },
    /// Fenced and removed from cluster
    Dead { generation: u64, detected_at: Instant },
    /// Rejoining after recovery
    Rejoining { generation: u64 },
}

/// Cluster membership with generation-based fencing
#[derive(Clone, Debug)]
pub struct ClusterMembership {
    /// Monotonically increasing generation counter
    pub generation: u64,
    /// Current coordinator node
    pub coordinator: NodeId,
    /// All known nodes and their states
    pub members: HashMap<NodeId, NodeState>,
    /// Quorum size (majority of active members)
    pub quorum_size: usize,
    /// NVMe lease path for 2-node tiebreaker
    pub lease_path: Option<PathBuf>,
}

/// Heartbeat payload (fits in single RDMA UD packet)
#[repr(C, packed)]
pub struct HeartbeatPayload {
    pub node_id: u64,
    pub sequence: u64,
    pub timestamp_ns: u64,
    pub generation: u64,
    pub gpu_health_flags: u64,      // Bitmap: 1 bit per GPU (healthy/unhealthy)
    pub memory_pressure_pct: u8,    // 0-100
    pub ptp_offset_ns: i64,         // From R17 v2: NodeInfo.ptp_offset_ns
    pub active_recoveries: u8,      // Number of recoveries in progress
    // Total: 50 bytes
}

/// Phi accrual failure detector state per monitored node
pub struct PhiAccrualDetector {
    /// Sliding window of inter-arrival times (milliseconds)
    arrival_window: VecDeque<f64>,
    /// Window size (default: 1000 samples)
    window_size: usize,
    /// Last heartbeat received timestamp
    last_heartbeat: Instant,
    /// Computed mean and variance of inter-arrival times
    mean: f64,
    variance: f64,
}
```

### 2.4 Recovery Pipeline Types

```rust
/// Recovery plan generated after failure detection
#[derive(Debug)]
pub struct RecoveryPlan {
    pub failed_node: NodeId,
    pub generation: u64,
    /// Pages to reconstruct, ordered by priority
    pub reconstruction_queue: Vec<ReconstructionTask>,
    /// Target node for reconstructed data (hot spare or least-loaded survivor)
    pub target_node: NodeId,
    /// Estimated total reconstruction time
    pub estimated_time: Duration,
}

#[derive(Debug)]
pub struct ReconstructionTask {
    pub vpn: Vpn,
    pub parity_group_id: u32,
    pub method: ReconstructionMethod,
    pub priority: ReconstructionPriority,
    pub data_class: DataClass,
}

#[derive(Debug)]
pub enum ReconstructionMethod {
    /// Reconstruct from XOR parity (single failure)
    XorReconstruct {
        surviving_pages: Vec<(NodeId, Vpn)>,
        parity_location: ParityLocation,
    },
    /// Reconstruct from RS parity (multi-failure capable)
    RsReconstruct {
        available_fragments: Vec<(u8, NodeId, u64)>, // (fragment_idx, node, addr)
        config: ErasureConfig,
    },
    /// Restore from checkpoint
    CheckpointRestore {
        checkpoint_id: u64,
        checkpoint_location: ParityLocation,
        deltas_to_apply: Vec<DeltaRef>,
    },
    /// Page is a dedup reference -- just update pointer to surviving canonical
    DedupRelink {
        canonical_vpn: Vpn,
        canonical_node: NodeId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReconstructionPriority {
    /// Currently being accessed by a CUDA kernel (stalling GPU)
    Immediate = 0,
    /// Hot page (recent access within last 100ms)
    Hot = 1,
    /// Warm page (accessed within last 10s)
    Warm = 2,
    /// Cold page (not recently accessed)
    Cold = 3,
}
```

### 2.5 Checkpoint Types

```rust
/// In-memory checkpoint following Gemini architecture
#[derive(Debug)]
pub struct Checkpoint {
    pub id: u64,
    pub created_at: Instant,
    /// Which training step this checkpoint represents
    pub training_step: u64,
    /// Per-shard checkpoint data
    pub shards: Vec<CheckpointShard>,
    /// Delta chain from previous checkpoint (if incremental)
    pub delta_from: Option<u64>,
    /// Protection: where redundant copies live
    pub redundancy: CheckpointRedundancy,
}

#[derive(Debug)]
pub struct CheckpointShard {
    pub shard_id: u32,
    /// Node that owns this shard
    pub source_node: NodeId,
    /// Location of checkpoint data
    pub locations: Vec<CheckpointLocation>,
    /// Metadata for resharding
    pub tensor_metadata: Vec<TensorMeta>,
    /// Total size in bytes
    pub size: usize,
}

#[derive(Debug)]
pub struct CheckpointLocation {
    pub node_id: NodeId,
    pub tier: CheckpointTier,
    pub address: u64,
    pub size: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum CheckpointTier {
    /// Local DRAM (fastest, volatile)
    LocalDram,
    /// Remote DRAM via RDMA (fast, volatile, cross-node redundancy)
    RemoteDram,
    /// Local NVMe (persistent)
    LocalNvme,
    /// Remote NVMe (persistent, cross-node)
    RemoteNvme,
}

#[derive(Debug)]
pub struct CheckpointRedundancy {
    /// Primary copy location
    pub primary: CheckpointLocation,
    /// Redundant copies (Gemini-style placement optimization)
    pub replicas: Vec<CheckpointLocation>,
    /// RS parity across checkpoint shards (optional, for warm/cold checkpoints)
    pub parity: Option<ParityGroup>,
}

/// Incremental delta between two checkpoints
#[derive(Debug)]
pub struct CheckpointDelta {
    pub from_checkpoint: u64,
    pub to_checkpoint: u64,
    /// Changed regions: (offset_in_shard, data)
    pub regions: Vec<DeltaRegion>,
    /// Compression ratio achieved
    pub compression_ratio: f32,
}

#[derive(Debug)]
pub struct DeltaRegion {
    pub shard_id: u32,
    pub offset: usize,
    pub data: Vec<u8>,  // Compressed delta bytes
}
```

---

## 3. Trait Implementations

### 3.1 ParityHook Trait (R10 Integration)

This trait is called by R10's page table on every page write to maintain parity consistency.

```rust
/// Trait defined by R10 v2 for parity maintenance.
/// R15 provides the implementation.
pub trait ParityHook: Send + Sync {
    /// Called by R10 when a page is written.
    /// Updates parity for the page's group asynchronously or synchronously
    /// based on the ParityPolicy.
    fn on_page_write(&self, vpn: Vpn) -> Result<()>;

    /// Returns all page VPNs in a parity group (for reconstruction).
    fn get_parity_pages(&self, group_id: u32) -> Vec<Vpn>;

    /// Called when a page is allocated -- assigns it to a parity group.
    fn on_page_alloc(&self, vpn: Vpn, data_class: DataClass) -> Result<u32>;

    /// Called when a page is freed -- removes it from its parity group.
    fn on_page_free(&self, vpn: Vpn, group_id: u32) -> Result<()>;

    /// Called when a page migrates between tiers (R10 MigrationEngine).
    /// Parity location may need to update.
    fn on_page_migrate(&self, vpn: Vpn, old_node: NodeId, new_node: NodeId) -> Result<()>;
}
```

**Implementation: `ErasureParityHook`**

```rust
pub struct ErasureParityHook {
    /// All active parity groups
    groups: DashMap<u32, ParityGroup>,
    /// VPN -> group_id mapping (reverse index)
    vpn_to_group: DashMap<Vpn, u32>,
    /// Async parity update queue (batched)
    dirty_queue: crossbeam::queue::SegQueue<Vpn>,
    /// ISA-L encoder handle
    encoder: IsalEncoder,
    /// Background parity update thread handle
    parity_worker: JoinHandle<()>,
    /// Configuration
    config: FaultToleranceConfig,
}

impl ParityHook for ErasureParityHook {
    fn on_page_write(&self, vpn: Vpn) -> Result<()> {
        // 1. Look up group for this VPN
        let group_id = self.vpn_to_group.get(&vpn)
            .ok_or(Error::NoParityGroup)?;

        // 2. Mark PTE as PARITY_STALE
        // (R10 page table access)
        page_table.set_flag(vpn, PageFlags::PARITY_STALE);

        // 3. Based on policy, either update now or queue
        let group = self.groups.get(&group_id).unwrap();
        match group.policy {
            ParityPolicy::Synchronous => {
                self.update_parity_now(*group_id, vpn)?;
            }
            ParityPolicy::AsyncBatch { .. } => {
                self.dirty_queue.push(vpn);
            }
            ParityPolicy::CheckpointOnly | ParityPolicy::None => {
                // No parity update needed now
            }
        }
        Ok(())
    }

    fn get_parity_pages(&self, group_id: u32) -> Vec<Vpn> {
        self.groups.get(&group_id)
            .map(|g| g.data_pages.clone())
            .unwrap_or_default()
    }

    fn on_page_alloc(&self, vpn: Vpn, data_class: DataClass) -> Result<u32> {
        // 1. Find or create a parity group with available slots
        // 2. Group selection considers: node distribution (spread pages
        //    across nodes), data class (don't mix Critical with Transient)
        // 3. Assign group_id and return it for PTE storage
        let group_id = self.find_or_create_group(vpn, data_class)?;
        self.vpn_to_group.insert(vpn, group_id);
        Ok(group_id)
    }
    // ... remaining methods
}
```

### 3.2 FailureDetector Trait

```rust
/// Failure detection interface. R17 provides phi accrual;
/// R15 adds RDMA event monitoring and recovery triggering.
pub trait FailureDetector: Send + Sync {
    /// Process a received heartbeat
    fn on_heartbeat(&self, from: NodeId, payload: &HeartbeatPayload);

    /// Process an RDMA async event (fastest detection path)
    fn on_rdma_event(&self, event: RdmaAsyncEvent);

    /// Get current suspicion level for a node
    fn phi(&self, node: NodeId) -> f64;

    /// Get current node state
    fn node_state(&self, node: NodeId) -> NodeState;

    /// Register callback for node failure events
    fn on_failure(&self, callback: Box<dyn Fn(NodeId, NodeState) + Send + Sync>);
}

/// R15's implementation combining all detection layers
pub struct MultiLayerDetector {
    /// Per-node phi accrual detectors (from R17)
    phi_detectors: DashMap<NodeId, PhiAccrualDetector>,
    /// RDMA event monitoring thread
    rdma_monitor: RdmaEventMonitor,
    /// TCP keepalive fallback
    tcp_fallback: TcpKeepaliveMonitor,
    /// Cluster membership state
    membership: RwLock<ClusterMembership>,
    /// Failure callbacks
    callbacks: Vec<Box<dyn Fn(NodeId, NodeState) + Send + Sync>>,
    /// Configuration
    phi_threshold: f64,       // Default: 6.0
    heartbeat_interval: Duration, // Default: 100ms
}
```

### 3.3 RecoveryOrchestrator

```rust
/// Orchestrates the full recovery pipeline:
/// DETECT -> FENCE -> ASSESS -> RECONSTRUCT -> RESUME
pub trait RecoveryOrchestrator: Send + Sync {
    /// Entry point: called when a node failure is confirmed
    fn initiate_recovery(&self, failed_node: NodeId, generation: u64) -> Result<RecoveryPlan>;

    /// Execute a recovery plan
    fn execute_recovery(&self, plan: RecoveryPlan) -> Result<RecoveryReport>;

    /// Handle a fault during recovery (R19 interaction: pages being
    /// reconstructed that get accessed)
    fn on_fault_during_recovery(&self, vpn: Vpn) -> Result<FaultResolution>;
}

pub struct FaultToleranceOrchestrator {
    /// R10 page table access
    page_table: Arc<dyn PageTable>,
    /// R10 migration engine for moving reconstructed pages
    migration_engine: Arc<dyn MigrationEngine>,
    /// Parity hook for reading parity data
    parity_hook: Arc<ErasureParityHook>,
    /// Checkpoint manager
    checkpoint_mgr: Arc<CheckpointManager>,
    /// Failure detector
    detector: Arc<MultiLayerDetector>,
    /// ISA-L decoder
    decoder: IsalDecoder,
    /// Transport layer for RDMA reads during reconstruction
    transport: Arc<dyn Transport>,
    /// R12 dedup manager (for dedup-aware recovery)
    dedup_mgr: Arc<dyn DedupManager>,
}
```

---

## 4. Algorithms and Protocols

### 4.1 Parity Group Assignment Algorithm

When a new page is allocated, it must be assigned to a parity group that maximizes fault tolerance.

**Step-by-step:**

1. Determine the page's `DataClass` from the allocation context (CUDA alloc interceptor provides hints).
2. If `DataClass::Transient`, return `group_id = 0` (no parity).
3. Look up the node that owns this page (from PTE).
4. Find an existing parity group that:
   - Has the same `DataClass`
   - Has fewer than `k` data pages (not full)
   - Has data pages distributed across at least 2 different nodes (fault isolation)
   - Does NOT already have a page on the same node (spread requirement)
5. If no suitable group exists, create a new group:
   - `k` and `m` chosen from `ErasureConfig` based on cluster size (see table in v1)
   - Scheme: `Xor` if `DataClass::Recoverable`, `ReedSolomon` if `DataClass::Critical`
6. Assign `group_id` to PTE, set `PARITY_VALID` flag.
7. If the group is now full (`k` data pages), trigger initial parity computation.

**R12 dedup integration:** When R12 identifies a page as a dedup candidate:
- The **canonical copy** MUST be in a parity group (it holds the actual data).
- **Reference copies** (pages pointing to canonical) are SKIPPED from parity assignment. They carry no unique data -- if the canonical is lost, it's reconstructed from parity; if a reference is lost, just re-point to canonical.

### 4.2 Parity Update (Async Batch)

The background parity worker processes the dirty queue:

**Step-by-step (runs every `batch_interval_ms`, default 5ms):**

1. Drain up to 256 VPNs from `dirty_queue`.
2. Group them by `parity_group_id`.
3. For each affected parity group:
   a. Read current data for ALL pages in the group (RDMA read from their current locations).
   b. If `Xor` scheme: XOR all data pages together to produce parity.
   c. If `ReedSolomon`: Run ISA-L `ec_encode_data()` with the full data set.
   d. Write new parity to `parity_locations` via RDMA write.
   e. Clear `PARITY_STALE` flag on all pages in the group.
   f. Update `last_parity_update` timestamp.
4. If a page was written AGAIN during parity computation (race), it stays in the dirty queue for the next batch.

**XOR fast-path optimization:** For XOR parity, instead of re-reading all pages, use the incremental update:
```
new_parity = old_parity XOR old_page_data XOR new_page_data
```
This requires caching `old_page_data` or reading it before the write completes. The incremental path avoids reading k-1 other pages.

### 4.3 Failure Detection Protocol

**Multi-layer detection flow:**

```
Time 0ms:     Node B crashes (power loss)
Time ~0.01ms: ConnectX-5 on Node A detects QP errors for Node B connections
              -> ibv_async_event(IBV_EVENT_QP_FATAL) fires
              -> MultiLayerDetector.on_rdma_event() called
              -> Node B state: Active -> Suspected { phi: N/A (RDMA signal) }

Time ~10ms:   Attempt QP recovery (ibv_modify_qp to RESET, then INIT->RTR->RTS)
              -> Recovery fails (remote node is actually dead)
              -> Suspicion level increased

Time ~300ms:  Phi accrual detector confirms (no heartbeats for 200ms+)
              -> phi(Node B) crosses threshold (6.0)
              -> Node B state: Suspected -> Fencing { generation: G+1 }

Time ~350ms:  Fencing begins:
              1. Coordinator broadcasts FENCE(Node B, generation G+1) to all active nodes
              2. Each node: cancel pending RDMA ops to Node B
              3. Each node: invalidate QPs connected to Node B
              4. Each node: mark all PTEs with node=B as INVALID
              5. Coordinator: bump cluster generation to G+1

Time ~400ms:  Fencing complete. Node B state: Dead { generation: G+1 }
              -> RecoveryOrchestrator.initiate_recovery(Node B, G+1) called
```

### 4.4 Recovery Pipeline

**Full recovery flow after fencing completes:**

**Phase: ASSESS (~100-500ms)**

1. Scan page table for all pages that were on the failed node.
2. For each page, classify recovery method:
   - If `PARITY_VALID` and parity group has enough survivors: `XorReconstruct` or `RsReconstruct`
   - If page is a dedup reference (R12, `DEDUP_CANONICAL` flag NOT set): `DedupRelink`
   - If page has `DataClass::Checkpoint` and latest checkpoint exists: `CheckpointRestore`
   - If page has `DataClass::Transient`: skip (application will retry)
3. Sort reconstruction queue by priority:
   - `Immediate`: pages with pending CUDA operations stalled waiting for data
   - `Hot`: pages accessed in last 100ms (from R10 ARC metadata)
   - `Warm`: pages accessed in last 10s
   - `Cold`: everything else
4. Select target node: hot spare if available, otherwise least-loaded surviving node.

**Phase: RECONSTRUCT (~1-30s depending on data volume)**

For each task in priority order:

*XOR Reconstruction:*
1. RDMA read all surviving data pages in the parity group from their current nodes.
2. RDMA read the XOR parity fragment from its storage location.
3. XOR all fragments together: `lost_page = parity XOR page_1 XOR page_2 XOR ... XOR page_{k-1}`
4. RDMA write result to target node (hot spare or survivor).
5. Update PTE: new node, clear `RECOVERY_IN_PROGRESS`.

*RS Reconstruction:*
1. Identify which `k` fragments (data + parity) are available.
2. RDMA read all `k` available fragments.
3. Call ISA-L `ec_encode_data()` with recovery matrix to reconstruct lost fragments.
4. RDMA write reconstructed fragments to target node.
5. Update PTEs.

*Dedup Relink (R12 integration):*
1. The lost page was a reference to a canonical page on another node.
2. Look up canonical page VPN from R12's dedup table.
3. Update the reference PTE to point to the canonical copy's current location.
4. No data transfer needed (just metadata update).

*Checkpoint Restore:*
1. Find latest checkpoint containing this page's data.
2. RDMA read checkpoint shard from its storage location.
3. Apply any incremental deltas between checkpoint and failure time.
4. Write restored data to target node.

**Phase: RESUME (~100ms-1s)**

1. For all reconstructed pages: update R10 page table with new locations.
2. Signal R19 coherency protocol: ownership transfers for pages that were writer-owned by failed node.
3. Unblock any stalled CUDA operations (return from intercepted calls).
4. Notify R17 topology manager: update node count, recompute scheduling weights.
5. If R22 live migration was in progress when failure occurred: abort migration, recovery takes priority.
6. Begin background rebalancing (lazy -- move data to restore even distribution over minutes).

### 4.5 Handling Faults During Recovery (R19 Integration)

When a surviving node accesses a page that is currently being reconstructed:

1. R19's page fault handler fires (page is marked `RECOVERY_IN_PROGRESS`).
2. Fault handler calls `RecoveryOrchestrator::on_fault_during_recovery(vpn)`.
3. The orchestrator checks if this VPN's reconstruction is already in progress:
   - **If yes:** The requesting thread waits on a per-VPN condvar. When reconstruction completes, all waiters are unblocked.
   - **If not yet started:** Promote this VPN to `Immediate` priority and begin reconstruction now.
4. The CUDA call that triggered the fault blocks until data is available (same as a regular network page fault in R19).

This prevents double-reconstruction and ensures pages needed by active compute are recovered first.

### 4.6 Checkpoint Protocol (Gemini-Inspired)

**Hot checkpoint (every iteration):**

1. At end of training step, snapshot engine issues `cudaMemcpyAsync(gpu_state, host_pinned_buf, D2H, checkpoint_stream)`.
2. This overlaps with the next forward pass on the compute stream.
3. When D2H copy completes (callback), write snapshot to local DRAM checkpoint buffer.
4. Mark previous hot checkpoint as superseded.

**Warm checkpoint (every N=10-50 iterations):**

1. Take hot checkpoint as above.
2. Additionally, RDMA write the checkpoint shard to a partner node's DRAM.
3. Partner node selected by Gemini's placement optimization (maximize survival probability given failure distribution).
4. Apply RS parity across checkpoint shards from all nodes.
5. Partner selection considers R17 topology: prefer nodes on different power circuits / network switches.

**Cold checkpoint (every M=100-1000 iterations):**

1. Take warm checkpoint as above.
2. Additionally, write full checkpoint to local NVMe (background, via io_uring).
3. RDMA write to remote NVMe on a different node.
4. This is the only durable checkpoint -- survives total cluster power loss.

**Incremental delta computation:**

1. After hot checkpoint N, XOR with hot checkpoint N-1.
2. Non-zero regions are the delta.
3. Compress delta with LZ4 (fast, ~2 GB/s).
4. For warm checkpoint transfer, send only the compressed delta instead of full snapshot.
5. Remote node applies delta to its copy: `remote_ckpt = remote_ckpt XOR decompressed_delta`.

---

## 5. Integration Points (Exact Function Calls)

### 5.1 R10 (Memory Tiering) Integration

| R15 calls R10 | Purpose |
|----------------|---------|
| `page_table.get_pte(vpn)` | Read page location, flags, parity_group_id during recovery assessment |
| `page_table.set_flag(vpn, PARITY_STALE)` | Mark page dirty for parity update |
| `page_table.set_flag(vpn, RECOVERY_IN_PROGRESS)` | Block access during reconstruction |
| `page_table.update_location(vpn, new_node, new_addr)` | Move page after reconstruction |
| `migration_engine.migrate_page(vpn, target_node, target_tier)` | Move reconstructed page to target |
| `arc_cache.get_access_frequency(vpn)` | Determine reconstruction priority (hot/cold) |
| `arc_cache.evict_cold_pages(node, bytes_needed)` | Make room for reconstructed data |

| R10 calls R15 | Purpose |
|----------------|---------|
| `parity_hook.on_page_write(vpn)` | Trigger parity update on write |
| `parity_hook.on_page_alloc(vpn, data_class)` | Assign parity group on alloc |
| `parity_hook.on_page_free(vpn, group_id)` | Remove from parity group on free |
| `parity_hook.on_page_migrate(vpn, old_node, new_node)` | Update parity after migration |

### 5.2 R12 (Deduplication) Integration

| Interaction | Detail |
|-------------|--------|
| Canonical pages -> parity groups | R12's canonical (unique) copies MUST be assigned to parity groups. R15 calls `dedup_mgr.is_canonical(vpn)` during group assignment. |
| Reference pages -> skip parity | Reference pages (pointing to canonical) are NOT added to parity groups. `dedup_mgr.is_reference(vpn)` returns true -> `on_page_alloc` returns `group_id = 0`. |
| Recovery of dedup references | If a reference page is lost, recovery is instant: just re-point to canonical. `dedup_mgr.get_canonical(vpn)` returns the canonical VPN. |
| Canonical loss | If a canonical page is lost, reconstruct from parity. Then update all references to point to the new location. `dedup_mgr.get_references(canonical_vpn)` returns all reference VPNs. |

### 5.3 R17 (Topology / Failure Detection) Integration

| R15 uses from R17 | Purpose |
|--------------------|---------|
| `topology.get_phi_detector(node_id)` | Access phi accrual detector for each node |
| `topology.heartbeat_interval()` | Get configured heartbeat interval (100ms default) |
| `NodeInfo.ptp_offset_ns` | From R17 v2: used to validate clock sync for coordinated recovery |
| `LinkInfo.ptp_one_way_delay` | From R17 v2: used for recovery time estimation |
| `topology.get_node_role(node_id)` | Determine if failed node was coordinator (triggers re-election) |

| R15 provides to R17 | Purpose |
|----------------------|---------|
| `failure_detector.on_failure(callback)` | R17 registers to be notified of failures for topology updates |
| `membership.generation` | R17 uses generation to tag routing decisions |

### 5.4 R19 (SWMR Consistency) Integration

| Interaction | Detail |
|-------------|--------|
| Writer failure | If the failed node was the SWMR writer of a page, R15 must coordinate with R19 to transfer ownership. Call `coherency.transfer_ownership(vpn, new_owner)` after reconstruction. |
| Reader failure | If the failed node was a reader, call `coherency.remove_reader(vpn, failed_node)` to clean the directory. |
| Directory recovery | If the directory node fails, R15 reconstructs the directory from surviving page table entries: scan all PTEs, rebuild the sharer sets. |
| Faults during recovery | R19's fault handler delegates to R15 when `RECOVERY_IN_PROGRESS` is set (see Section 4.5). |

### 5.5 R22 (Live Migration) Integration

| Interaction | Detail |
|-------------|--------|
| Shared detection | R22 uses R15's `MultiLayerDetector` for failure detection during graceful migration. If a node fails MID-migration, R15 takes over. |
| Checkpoint as migration source | R22 can use R15's checkpoint data as a starting point for migration (avoids re-reading all VRAM). |
| Migration aborts on failure | If a failure is detected during live migration, R22 calls `recovery_orchestrator.initiate_recovery()` which cancels the migration and begins crash recovery instead. |

---

## 6. ISA-L Integration

### Rust FFI Bindings

```rust
/// Thin wrapper around Intel ISA-L erasure coding functions
pub struct IsalEncoder {
    /// Encoding matrix (Cauchy or Vandermonde)
    encode_matrix: Vec<u8>,
    /// Precomputed encoding tables
    encode_tables: Vec<u8>,
    /// k (data fragments) and m (parity fragments)
    k: u32,
    m: u32,
}

impl IsalEncoder {
    pub fn new(k: u32, m: u32) -> Self {
        let mut encode_matrix = vec![0u8; ((k + m) * k) as usize];
        let mut encode_tables = vec![0u8; (32 * k * m) as usize];

        unsafe {
            // Generate Cauchy encoding matrix
            isa_l::gf_gen_cauchy1_matrix(
                encode_matrix.as_mut_ptr(),
                (k + m) as i32,
                k as i32,
            );
            // Precompute encoding tables
            isa_l::ec_init_tables(
                k as i32,
                m as i32,
                encode_matrix[k as usize * k as usize..].as_ptr(),
                encode_tables.as_mut_ptr(),
            );
        }
        Self { encode_matrix, encode_tables, k, m }
    }

    /// Encode data fragments into parity fragments.
    /// `data_ptrs`: array of k pointers to data buffers (each of `len` bytes).
    /// `parity_ptrs`: array of m pointers to output parity buffers.
    pub fn encode(&self, data_ptrs: &[*mut u8], parity_ptrs: &[*mut u8], len: usize) {
        unsafe {
            isa_l::ec_encode_data(
                len as i32,
                self.k as i32,
                self.m as i32,
                self.encode_tables.as_ptr(),
                data_ptrs.as_ptr() as *mut *mut u8,
                parity_ptrs.as_ptr() as *mut *mut u8,
            );
        }
    }
}

/// XOR encoder (trivial, no ISA-L needed)
pub struct XorEncoder;

impl XorEncoder {
    /// XOR all `data` buffers into `parity`.
    /// Uses AVX2 intrinsics when available for memory-bandwidth-limited speed.
    pub fn encode(data: &[&[u8]], parity: &mut [u8]) {
        assert!(data.len() >= 2);
        assert!(data.iter().all(|d| d.len() == parity.len()));

        // Initialize parity with first data buffer
        parity.copy_from_slice(data[0]);

        // XOR remaining buffers
        for buf in &data[1..] {
            for (p, d) in parity.iter_mut().zip(buf.iter()) {
                *p ^= *d;
            }
        }
        // Note: in production, use SIMD intrinsics (AVX2/AVX-512)
        // for memory-bandwidth-limited throughput (~30 GB/s per core).
    }

    /// Incremental XOR update: new_parity = old_parity XOR old_data XOR new_data
    pub fn incremental_update(parity: &mut [u8], old_data: &[u8], new_data: &[u8]) {
        for i in 0..parity.len() {
            parity[i] ^= old_data[i] ^ new_data[i];
        }
    }
}
```

---

## 7. Refined Timeline

| Phase | Duration | Dependencies | Deliverables |
|-------|----------|-------------|--------------|
| **R15-A: Erasure Coding Foundation** | 3 weeks | R10 page table API stable | `IsalEncoder`, `XorEncoder`, `ParityGroup`, parity storage manager, unit tests (encode/decode/recover) |
| **R15-B: Failure Detection** | 2 weeks | R17 phi accrual integrated | `MultiLayerDetector`, RDMA event monitor, `ClusterMembership`, fencing protocol |
| **R15-C: Recovery Pipeline** | 3 weeks | R15-A + R15-B | `FaultToleranceOrchestrator`, reconstruction (XOR, RS, dedup-relink, checkpoint-restore), R10 page table updates, hot spare activation |
| **R15-D: Checkpointing** | 3 weeks | R10 migration, RDMA transport | `CheckpointManager`, hot/warm/cold checkpoint tiers, incremental deltas, Gemini-style placement |
| **R15-E: Integration & Testing** | 2 weeks | All above + R12, R19 | R12 dedup-aware recovery, R19 coherency during recovery, fault injection framework, end-to-end tests |

**Total: 13 weeks** (previously 13-18; tightened by having clearer specs)

**Parallelism:** R15-A and R15-B can run in parallel (no dependency between them). R15-C depends on both. R15-D can start when R15-A is done (shares ISA-L). R15-E requires all others.

---

## 8. Success Criteria (Unchanged from v1, with additions)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Single node failure recovery time | < 30 seconds | Time from crash to all operations resumed |
| Hot page recovery (XOR parity available) | < 5 seconds | Time to reconstruct and resume access |
| Dedup reference recovery | < 100ms | Metadata-only operation |
| Data durability (RS parity) | 99.999% | No data loss for any single-node failure |
| Detection latency (RDMA event path) | < 100ms | Time from failure to suspicion |
| Detection latency (phi accrual) | < 600ms | Time from failure to confirmed dead |
| Normal operation overhead | < 5% | Throughput reduction vs no fault tolerance |
| Checkpoint overhead | < 3% | Training throughput impact |
| Parity storage overhead | 25-50% of protected data | Depends on RS configuration |
| Incremental delta size | < 30% of full checkpoint | For typical LLM training |
| **NEW:** Fault-during-recovery latency | < 2x normal page fault | Page access during active reconstruction |
| **NEW:** Cascading failure tolerance | 2 simultaneous failures | RS(k,2) configuration handles this |

---

## Related Documents

- [preplan.md](preplan.md) -- v1 pre-plan (superseded by this document)
- [research/01-erasure-coding-algorithms.md](research/01-erasure-coding-algorithms.md)
- [research/02-distributed-checkpointing.md](research/02-distributed-checkpointing.md)
- [research/03-failure-detection-recovery.md](research/03-failure-detection-recovery.md)
- R10: Memory Tiering (ParityHook trait, MigrationEngine, 64KB pages, hash-table page table)
- R12: Memory Deduplication (canonical vs reference pages, parity group rules)
- R17: Topology-Aware Scheduling (phi accrual, heartbeat, NodeInfo.ptp_offset_ns)
- R19: Network Page Faults / SWMR Consistency (fault-during-recovery handling)
- R22: Live Migration (shared failure detection, checkpoint-as-migration-source)
