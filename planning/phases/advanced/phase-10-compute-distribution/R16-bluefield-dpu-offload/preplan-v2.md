# R16: BlueField DPU Offload — Pre-Plan v2 (Refined)

**Created:** 2026-03-25
**Updated:** 2026-03-25
**Status:** DRAFT
**Purpose:** Second-round refinement of R16 with exact Rust structs, concrete algorithms, cross-topic integration points, and resolved open questions from v1.

---

## 1. Changes from v1

| Area | v1 | v2 |
|---|---|---|
| Structs | Conceptual `TransportBackend` trait sketch | Full Rust struct definitions with fields, lifetimes, error types |
| Offload decision | "Check data size, skip below threshold" | Concrete algorithm with numeric thresholds and decision tree |
| R10 integration | "Page table on DPU" | Exact `DpuPageTable` struct mirroring R10's `TierManager` + `PageTable` |
| R11 integration | "Prefetch on ARM cores" | `DpuPrefetchScheduler` with traffic monitor, prediction pipeline |
| R14 integration | "DOCA Compress" | `DpuCompressor` with adaptive threshold, BF-2/BF-3 code paths |
| R17 integration | "Routing decisions on DPU" | `DpuRouter` with topology graph in DPU DRAM, path cache |
| R21 integration | Not mentioned | NVMe-oF target offload via DPU ConnectX |
| Build pipeline | "Docker cross-compilation" | Exact `cross-rs` config, Dockerfile outline, CI matrix |
| Open questions | 6 unresolved | All resolved with concrete decisions |

---

## 2. Rust Struct Definitions

### 2.1 DpuService (Top-Level Daemon)

```rust
/// Top-level service running on BlueField ARM cores.
/// Single binary with internal module architecture.
/// Cross-compiled to aarch64-unknown-linux-gnu via cross-rs.
pub struct DpuService {
    /// DOCA device handle (FFI-wrapped)
    device: DocaDevice,

    /// Host communication via DOCA Comm Channel
    host_channel: HostChannel,

    /// All offloaded subsystems
    transport: DpuTransportManager,
    compressor: DpuCompressor,
    page_table: DpuPageTable,
    prefetcher: DpuPrefetchScheduler,
    router: DpuRouter,

    /// Runtime state
    config: DpuConfig,
    stats: DpuStats,
    shutdown: AtomicBool,
}

pub struct DpuConfig {
    /// Which subsystems are enabled (feature flags)
    pub transport_offload: bool,
    pub compression_offload: bool,
    pub prefetch_offload: bool,
    pub bar1_direct: bool,

    /// Resource limits
    pub max_connections: u32,        // default: 256
    pub page_table_max_gpus: u8,     // default: 4
    pub prefetch_cache_mb: u32,      // default: 4096 (4 GB)
    pub compress_staging_mb: u32,    // default: 2048 (2 GB)

    /// BlueField generation (detected at startup)
    pub bf_generation: BfGeneration,
}

pub enum BfGeneration {
    BlueField2 { cores: u8, dram_gb: u8, has_lz4: bool },  // has_lz4 = false
    BlueField3 { cores: u8, dram_gb: u8, has_lz4: bool },  // has_lz4 = true
}

pub struct DpuStats {
    pub bytes_transferred: AtomicU64,
    pub bytes_compressed: AtomicU64,
    pub compression_ratio_avg: AtomicU32,  // fixed-point 8.24
    pub prefetch_hits: AtomicU64,
    pub prefetch_misses: AtomicU64,
    pub page_faults_served: AtomicU64,
    pub connections_active: AtomicU32,
    pub uptime_secs: AtomicU64,
}
```

### 2.2 TransportBackend Trait

```rust
/// Abstraction that allows outerlink-server to work identically
/// with or without a DPU. Host-side code programs against this trait.
#[async_trait]
pub trait TransportBackend: Send + Sync {
    /// Transfer a memory region to/from a remote node.
    /// Returns when the transfer is complete (data landed).
    async fn transfer(&self, req: TransferRequest) -> Result<TransferComplete, TransportError>;

    /// Compress data before sending. Returns compressed buffer.
    /// Implementation may use hardware (DPU) or software (CPU).
    async fn compress(&self, data: &MemRegion, hint: CompressHint) -> Result<CompressedBuf, TransportError>;

    /// Decompress received data. Returns decompressed buffer.
    async fn decompress(&self, data: &CompressedBuf) -> Result<MemRegion, TransportError>;

    /// Look up the best route to a destination node.
    fn route(&self, dest: NodeId) -> Result<Route, TransportError>;

    /// Establish a connection to a remote node.
    async fn connect(&self, peer: PeerAddr) -> Result<ConnectionId, TransportError>;

    /// Disconnect from a remote node.
    async fn disconnect(&self, conn: ConnectionId) -> Result<(), TransportError>;

    /// Query backend capabilities.
    fn capabilities(&self) -> BackendCapabilities;

    /// Collect performance statistics.
    fn stats(&self) -> BackendStats;
}

pub struct BackendCapabilities {
    pub has_hw_compression: bool,
    pub hw_compression_algos: Vec<CompressionAlgo>,  // [Deflate] on BF-2, [Deflate, Lz4] on BF-3
    pub has_bar1_direct: bool,
    pub max_bandwidth_gbps: u32,
    pub supports_prefetch: bool,
}

/// Host-only implementation (current baseline, no DPU)
pub struct HostTransport {
    ucx_context: UcxContext,        // UCX for RDMA
    io_uring: IoUringRuntime,       // io_uring for TCP fallback
    compressor: SoftwareCompressor, // CPU-based LZ4/Zstd
}

/// DPU-offloaded implementation (delegates to outerlink-dpu via Comm Channel)
pub struct DpuTransport {
    channel: CommChannel,           // DOCA Comm Channel to DPU
    pending: DashMap<u64, oneshot::Sender<TransferComplete>>,
    next_request_id: AtomicU64,
}
```

### 2.3 DpuTransportManager (DPU-Side)

```rust
/// Runs on BlueField ARM cores. Manages all network I/O.
pub struct DpuTransportManager {
    /// DOCA RDMA context for network operations
    rdma_ctx: DocaRdma,

    /// Active connections to remote nodes
    connections: HashMap<ConnectionId, RdmaConnection>,

    /// BAR1 regions for direct GPU VRAM access
    gpu_bar1_regions: Vec<Bar1Region>,

    /// Progress engine for async DOCA operations
    progress_engine: DocaProgressEngine,
}

pub struct RdmaConnection {
    pub peer: PeerAddr,
    pub qp: DocaQueuePair,
    pub state: ConnectionState,
    pub created_at: Instant,
    pub bytes_sent: u64,
    pub bytes_recv: u64,
}

pub struct Bar1Region {
    pub gpu_id: u8,
    pub base_addr: u64,        // BAR1 physical address
    pub size: usize,           // Mapped region size
    pub doca_mmap: DocaMmap,   // DOCA memory map for this region
    pub rebar_enabled: bool,   // Whether full VRAM is accessible
}
```

### 2.4 DpuPageTable (R10 Integration)

```rust
/// Mirror of R10's PageTable running on DPU ARM cores.
/// Full copy of page ownership data fits in DPU DRAM:
///   24 GB VRAM / 4 KB pages = 6M entries x 40 bytes = ~240 MB per GPU
///   4 GPUs = ~960 MB (fits in BF-2's 16 GB with room to spare)
pub struct DpuPageTable {
    /// Per-GPU page tables
    tables: Vec<GpuPageMap>,

    /// Tier configuration (mirrors R10 TierManager)
    tier_config: TierConfig,

    /// Migration engine for DPU-initiated page moves
    migration: DpuMigrationEngine,
}

pub struct GpuPageMap {
    pub gpu_id: u8,
    pub node_id: NodeId,
    /// Page number -> PageEntry mapping
    /// Uses a compact array (not HashMap) since pages are dense
    pub entries: Vec<PageEntry>,
    pub total_pages: u32,
}

pub struct PageEntry {
    pub owner_node: NodeId,       // Which node currently has this page
    pub state: PageState,         // Local, Remote, Migrating, Invalid
    pub last_access: u32,         // Timestamp (compact, relative)
    pub access_count: u16,        // For hot/cold classification
    pub flags: u8,                // Pinned, Prefetched, Dirty
    pub _pad: u8,
}
// Size: 12 bytes per entry (compact for cache efficiency)

pub enum PageState {
    Local,        // Page is on this node's GPU VRAM
    Remote(NodeId), // Page is on another node
    Migrating,    // Transfer in progress
    Invalid,      // Page has been freed
}

pub struct DpuMigrationEngine {
    /// Pending migrations (DPU initiates these without host CPU)
    pending: VecDeque<MigrationTask>,
    /// Max concurrent migrations (limited by RDMA resources)
    max_concurrent: u8,  // default: 16
    active: u8,
}

pub struct MigrationTask {
    pub page_range: Range<u64>,
    pub src_node: NodeId,
    pub dst_node: NodeId,
    pub priority: MigrationPriority,
    pub initiated_at: Instant,
}
```

### 2.5 DpuCompressor (R14 Integration)

```rust
/// Hardware compression on DPU using DOCA Compress.
/// BF-2: deflate only. BF-3: deflate + LZ4.
pub struct DpuCompressor {
    /// DOCA Compress context
    compress_ctx: DocaCompress,

    /// Staging buffers in DPU DRAM (LZ4 requires local memory)
    staging_pool: StagingPool,

    /// Adaptive compression config (from R14)
    adaptive: AdaptiveCompressConfig,

    /// Per-connection compression stats for adaptive decisions
    conn_stats: HashMap<ConnectionId, CompressionStats>,
}

pub struct AdaptiveCompressConfig {
    /// Minimum data size to attempt compression (bytes)
    pub min_size: u32,              // default: 4096 (4 KB)
    /// Compressibility ratio below which we skip compression
    /// Sampled periodically, not checked per-transfer
    pub ratio_threshold: f32,       // default: 0.85 (15% savings minimum)
    /// Sample interval: check compressibility every N transfers
    pub sample_interval: u32,       // default: 64
    /// Preferred algorithm (auto-selected based on BF generation)
    pub preferred_algo: CompressionAlgo,
}

pub enum CompressionAlgo {
    None,
    Deflate,   // BF-2 and BF-3
    Lz4,       // BF-3 only
}

pub struct StagingPool {
    /// Pre-allocated DPU DRAM buffers for compression I/O
    buffers: Vec<StagingBuffer>,
    /// Free list (lock-free ring)
    free_indices: ArrayQueue<usize>,
}

pub struct StagingBuffer {
    pub doca_buf: DocaBuf,
    pub ptr: *mut u8,
    pub capacity: usize,  // default: 2 MB per buffer
}

pub struct CompressionStats {
    pub total_bytes_in: u64,
    pub total_bytes_out: u64,
    pub last_ratio: f32,
    pub samples_taken: u32,
    pub algo_used: CompressionAlgo,
}
```

### 2.6 DpuPrefetchScheduler (R11 Integration)

```rust
/// Speculative prefetch running on DPU ARM cores.
/// Monitors traffic patterns and proactively fetches pages
/// before the host requests them.
pub struct DpuPrefetchScheduler {
    /// Access pattern tracker per GPU
    trackers: Vec<AccessPatternTracker>,

    /// Prefetch cache in DPU DRAM
    cache: PrefetchCache,

    /// Active prefetch operations
    in_flight: VecDeque<PrefetchOp>,
    max_in_flight: u16,  // default: 32
}

pub struct AccessPatternTracker {
    pub gpu_id: u8,
    /// Stride detector: tracks sequential page access patterns
    pub stride: StrideDetector,
    /// Markov predictor: probabilistic next-page prediction
    pub markov: MarkovPredictor,
    /// Working set estimator: tracks active page set size
    pub working_set: WorkingSetEstimator,
    /// Recent access history (circular buffer)
    pub history: CircularBuffer<PageAccess, 1024>,
}

pub struct StrideDetector {
    pub last_page: u64,
    pub last_stride: i64,
    pub confidence: f32,      // 0.0-1.0, increases with consistent strides
    pub min_confidence: f32,  // threshold to trigger prefetch (default: 0.7)
}

pub struct MarkovPredictor {
    /// Transition table: page -> [(next_page, probability)]
    /// Capped at 64K entries to bound DPU DRAM usage
    pub transitions: HashMap<u64, SmallVec<[(u64, f32); 4]>>,
    pub max_entries: usize,  // default: 65536
}

pub struct PrefetchOp {
    pub page: u64,
    pub src_node: NodeId,
    pub predicted_by: PredictionSource,
    pub confidence: f32,
    pub initiated_at: Instant,
}

pub enum PredictionSource {
    Stride,
    Markov,
    WorkingSet,
}

pub struct PrefetchCache {
    /// DPU DRAM region for caching prefetched pages
    pub region: DocaMmap,
    pub capacity_pages: u32,     // default: ~1M pages (4 GB / 4 KB)
    pub used_pages: u32,
    /// LRU eviction tracking
    pub lru: LruIndex,
}
```

### 2.7 DpuRouter (R17 Integration)

```rust
/// Topology-aware routing running on DPU ARM cores.
/// Topology graph fits in DPU DRAM (even for ~100 node clusters).
pub struct DpuRouter {
    /// Cluster topology graph
    topology: TopologyGraph,

    /// Pre-computed shortest paths (updated on topology change)
    path_cache: HashMap<(NodeId, NodeId), CachedRoute>,

    /// Multi-path load balancer
    balancer: MultiPathBalancer,

    /// DOCA Flow rules installed for fast-path routing
    flow_rules: Vec<DocaFlowRule>,
}

pub struct TopologyGraph {
    /// Adjacency list: node -> [(neighbor, link_info)]
    pub adjacency: HashMap<NodeId, Vec<LinkInfo>>,
    pub version: u64,  // Incremented on topology change
}

pub struct LinkInfo {
    pub peer: NodeId,
    pub bandwidth_gbps: u32,
    pub latency_us: f32,
    pub link_type: LinkType,
    pub health: LinkHealth,
}

pub enum LinkType {
    RoCEv2,
    InfiniBand,
    Tcp,
}

pub struct CachedRoute {
    pub path: SmallVec<[NodeId; 4]>,  // Typically 1-2 hops
    pub total_latency_us: f32,
    pub bottleneck_bw_gbps: u32,
    pub computed_at: Instant,
    pub ttl_secs: u32,  // Re-compute after this
}

pub struct MultiPathBalancer {
    /// Active paths per destination, weighted by bandwidth
    pub paths: HashMap<NodeId, Vec<WeightedPath>>,
    /// Round-robin or weighted-random selection
    pub strategy: BalanceStrategy,
}
```

### 2.8 HostChannel (Host <-> DPU Protocol)

```rust
/// Messages between outerlink-server (host) and outerlink-dpu (DPU).
/// Serialized with bincode for minimal overhead.
/// Transported via DOCA Comm Channel (Phase A) or PCIe shared memory (Phase B+).

#[derive(Serialize, Deserialize)]
pub enum HostToDpuMsg {
    TransferRequest {
        id: u64,
        src: MemLocation,
        dst: MemLocation,
        size: u64,
        compress: CompressHint,
    },
    AllocNotify {
        gpu_id: u8,
        addr: u64,
        size: u64,
    },
    FreeNotify {
        gpu_id: u8,
        addr: u64,
    },
    SyncBarrier {
        id: u64,
    },
    ConnectPeer {
        peer: PeerAddr,
    },
    DisconnectPeer {
        conn_id: ConnectionId,
    },
    ConfigUpdate {
        config: DpuConfigDelta,
    },
    Shutdown,
}

#[derive(Serialize, Deserialize)]
pub enum DpuToHostMsg {
    TransferComplete {
        id: u64,
        bytes_transferred: u64,
        elapsed_us: u32,
    },
    TransferFailed {
        id: u64,
        error: TransportErrorCode,
    },
    PageFault {
        gpu_id: u8,
        page: u64,
        requestor: NodeId,
    },
    PeerConnected {
        conn_id: ConnectionId,
        peer: PeerAddr,
    },
    PeerDisconnected {
        conn_id: ConnectionId,
        reason: DisconnectReason,
    },
    Stats {
        stats: DpuStats,
    },
    Error {
        code: DpuErrorCode,
        message: String,
    },
}

pub enum MemLocation {
    HostPinned { addr: u64 },
    GpuVram { gpu_id: u8, addr: u64 },
    RemoteNode { node_id: NodeId, gpu_id: u8, addr: u64 },
}
```

---

## 3. Concrete Algorithms

### 3.1 DPU Offload Decision Algorithm

Runs at `outerlink-server` startup and on hot-plug events. Determines whether to use `HostTransport` or `DpuTransport`.

```
FUNCTION detect_and_configure_backend() -> TransportBackend:
    1. Enumerate DOCA devices via doca_devinfo_list_create()
    2. IF no DOCA devices found:
         RETURN HostTransport::new()
    3. FOR each DOCA device:
         a. Check if device is BlueField (PCI vendor/device ID)
         b. Check mode: must be DPU mode (ECPF), not NIC mode
         c. Check if outerlink-dpu service is running (attempt Comm Channel connect)
    4. IF no suitable DPU found:
         RETURN HostTransport::new()
    5. Negotiate capabilities:
         a. Send ConfigUpdate with host's GPU inventory
         b. Receive BackendCapabilities from DPU
    6. Detect BAR1 access:
         a. FOR each local GPU:
              i.  Query PCIe topology (lspci -t)
              ii. Check if GPU and DPU share PCIe root complex
              iii. If yes AND rebar enabled: mark gpu as bar1_accessible
    7. Configure subsystems:
         a. IF dpu.has_hw_compression: enable compression offload
         b. IF any gpu.bar1_accessible: enable BAR1 direct path
         c. IF dpu.cores >= 6: enable prefetch offload (needs 1-2 dedicated cores)
    8. RETURN DpuTransport::new(channel, capabilities)
```

### 3.2 Per-Transfer Offload Decision

Runs on the DPU for each `TransferRequest` received from host.

```
FUNCTION handle_transfer(req: TransferRequest) -> TransferComplete:
    1. Route lookup:
         route = router.lookup(req.dst.node_id)
         IF route.is_none(): RETURN Error(NoRoute)

    2. Compression decision:
         IF req.size < 4096:
             compress = false                    // Too small
         ELSE IF req.compress == CompressHint::Never:
             compress = false
         ELSE IF req.compress == CompressHint::Always:
             compress = true
         ELSE:  // CompressHint::Auto
             stats = conn_stats[req.connection]
             IF stats.samples_taken % 64 == 0:
                 // Sample compressibility
                 sample = take_sample(req.data, 4096)
                 trial_ratio = hw_compress_trial(sample)
                 stats.last_ratio = trial_ratio
             compress = stats.last_ratio < 0.85

    3. Source data access:
         MATCH req.src:
             HostPinned { addr }:
                 // DMA from host memory to DPU staging buffer
                 staging_buf = staging_pool.acquire()
                 doca_dma_copy(host_mmap[addr] -> staging_buf)
             GpuVram { gpu_id, addr }:
                 IF gpu_bar1_regions[gpu_id].is_some():
                     // Direct BAR1 read from GPU VRAM (zero host CPU)
                     staging_buf = staging_pool.acquire()
                     doca_dma_copy(bar1_mmap[gpu_id][addr] -> staging_buf)
                 ELSE:
                     // Fallback: ask host to cudaMemcpy to pinned, then DMA
                     send_to_host(PageFault { ... })
                     wait_for_host_staging()
                     staging_buf = staging_pool.acquire()
                     doca_dma_copy(host_mmap[staged_addr] -> staging_buf)

    4. Compress (if enabled):
         IF compress:
             compressed = hw_compress(staging_buf, preferred_algo)
             send_buf = compressed
         ELSE:
             send_buf = staging_buf

    5. Transmit:
         MATCH route.next_hop:
             DirectPeer(peer_dpu):
                 rdma_write(send_buf -> peer_dpu, route.qp)
             MultiHop(path):
                 // Forward to next DPU in path
                 rdma_write(send_buf -> path[0], route.qp)

    6. Wait for completion, update stats, return TransferComplete
```

### 3.3 ARM Core Allocation Algorithm

Runs at `outerlink-dpu` startup. Pins subsystems to specific cores for deterministic performance.

```
FUNCTION allocate_cores(bf_gen: BfGeneration) -> CoreAllocation:
    total_cores = bf_gen.cores

    // Core 0: always reserved for Linux kernel + DOCA runtime
    allocation[0] = Role::System

    IF total_cores == 8:   // BF-2
        allocation[1] = Role::Transport     // Connection management
        allocation[2] = Role::Transport     // RDMA I/O
        allocation[3] = Role::Compress      // HW compress task management
        allocation[4] = Role::PageTable     // Page ownership tracking
        allocation[5] = Role::Prefetch      // Access pattern monitoring
        allocation[6] = Role::Router        // Routing + flow rules
        allocation[7] = Role::Overflow      // Exception handling, stats
    ELSE IF total_cores == 16:  // BF-3
        allocation[1..4]  = Role::Transport   // 4 cores for higher throughput
        allocation[5]     = Role::Compress
        allocation[6..7]  = Role::PageTable   // 2 cores for faster migration
        allocation[8..9]  = Role::Prefetch    // 2 cores for better prediction
        allocation[10..11]= Role::Router      // 2 cores for complex topologies
        allocation[12..15]= Role::Overflow    // Future features, burst handling

    FOR each (core, role) in allocation:
        set_cpu_affinity(core, role.thread_handle)
        set_scheduling_policy(core, SCHED_FIFO, priority=50)
```

---

## 4. Cross-Topic Integration Points

### 4.1 R10 (Memory Hierarchy) Integration

| R10 Component | DPU Equivalent | Integration |
|---|---|---|
| `TierManager` | `DpuPageTable.tier_config` | DPU mirrors tier configuration. Hot/warm/cold thresholds synced from host. |
| `PageTable` | `DpuPageTable.tables` | Full copy in DPU DRAM. Host sends `AllocNotify`/`FreeNotify` to keep in sync. |
| `MigrationEngine` | `DpuMigrationEngine` | DPU initiates migrations autonomously. DPU-to-DPU RDMA without host CPU. |

**Sync protocol:** Host is the authority for page table mutations (alloc, free, pin). DPU receives deltas via `HostToDpuMsg`. DPU can autonomously initiate migrations (move pages between nodes) but must notify host of ownership changes via `DpuToHostMsg::PageFault`.

**Critical rule from R10 v2:** The DPU migration engine must respect pinned regions. R30's persistent kernel ring buffers are pinned -- the DPU must never evict doorbell or data buffer pages.

### 4.2 R11 (Speculative Prefetch) Integration

| R11 Component | DPU Location | Why DPU Is Better |
|---|---|---|
| Access pattern tracker | `DpuPrefetchScheduler.trackers` | DPU sees all traffic at network edge before host |
| Stride detector | `StrideDetector` on ARM core | Lightweight, fits in A72 cache |
| Markov predictor | `MarkovPredictor` on ARM core | 64K entries x ~40 bytes = ~2.5 MB in DPU DRAM |
| Prefetch cache | `PrefetchCache` in DPU DRAM | Pre-fetched pages stage here before DMA to GPU |

**Data flow:** DPU monitors page requests flowing through it. Stride/Markov predictors run on ARM core 5 (BF-2). When prediction confidence exceeds threshold (0.7), DPU initiates RDMA read to remote node. Fetched page lands in DPU DRAM prefetch cache. When host later requests that page, DPU DMA's it to GPU VRAM via BAR1 (if available) -- effectively zero-latency from the host's perspective.

### 4.3 R14 (Transport Compression) Integration

| R14 Feature | DPU Implementation |
|---|---|
| LZ4 compression | BF-3 DOCA Compress HW engine (line-rate) |
| Deflate fallback | BF-2 DOCA Compress HW engine (~10 GB/s) |
| Adaptive thresholds | `AdaptiveCompressConfig` on DPU, same algorithm as R14 |
| nvCOMP on GPU | NOT offloaded -- still runs on GPU when data is GPU-resident and large |

**Decision tree:** GPU-resident large data uses nvCOMP (avoids PCIe round trip to DPU). Host-staged or DPU-transiting data uses DOCA Compress hardware. DPU-side `DpuCompressor` handles the staging buffer management. BF-3 always prefers LZ4 (matches OuterLink's standard from R14). BF-2 uses deflate (still 100x faster than ARM software).

### 4.4 R17 (Topology-Aware Scheduling) Integration

| R17 Feature | DPU Implementation |
|---|---|
| Topology graph | `TopologyGraph` in DPU DRAM. Even 100-node cluster: ~100 nodes x ~10 links x 64 bytes = ~64 KB |
| Shortest path | Pre-computed at topology change, cached in `path_cache` |
| Multi-path balancing | `MultiPathBalancer` distributes across redundant links |
| DOCA Flow rules | Hardware packet steering for fast-path (known routes bypass ARM cores) |

**Line-rate routing:** For established connections with known routes, DOCA Flow steers packets directly in NIC hardware. ARM cores only handle route computation, topology updates, and exception flows (new connections, link failures).

### 4.5 R21 (NVMe-oF Storage) Integration

The DPU's ConnectX already supports NVMe-oF target offload in hardware. OuterLink can expose GPU VRAM as an NVMe-oF target via the DPU, allowing remote nodes to read/write VRAM using standard NVMe-oF initiators. The DPU adds programmability: access control, compression, and routing decisions that the plain ConnectX hardware path cannot do.

### 4.6 R30 (Persistent Kernels) Integration

The DPU is a natural producer for R30's ring buffer doorbell mechanism. The DPU's ConnectX writes data + descriptor + head pointer to GPU VRAM via BAR1. This is the same OpenDMA write path, but orchestrated by DPU ARM cores instead of a remote node's host CPU. The DPU can pipeline: receive from wire, decompress, write to GPU VRAM ring buffer, update doorbell -- all without host CPU.

---

## 5. Build Pipeline

### 5.1 Cross-Compilation with cross-rs

```toml
# Cross.toml (in outerlink-dpu crate root)
[target.aarch64-unknown-linux-gnu]
image = "outerlink/dpu-build:latest"
pre-build = []

[target.aarch64-unknown-linux-gnu.env]
passthrough = ["DOCA_SDK_VERSION"]
```

### 5.2 Dockerfile for Build Image

```dockerfile
# outerlink/dpu-build Dockerfile
FROM ghcr.io/cross-rs/aarch64-unknown-linux-gnu:main

# Install DOCA SDK (aarch64) for header files and libraries
ARG DOCA_SDK_VERSION=2.9.1
RUN wget -qO- https://linux.mellanox.com/public/keys/GPGKeyA.pub | apt-key add - && \
    echo "deb [arch=arm64] https://linux.mellanox.com/public/repo/doca/${DOCA_SDK_VERSION}/ubuntu22.04/arm64/ /" \
    > /etc/apt/sources.list.d/doca.list && \
    dpkg --add-architecture arm64 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libdoca-common-dev:arm64 \
        libdoca-dma-dev:arm64 \
        libdoca-compress-dev:arm64 \
        libdoca-comm-channel-dev:arm64 \
        libdoca-rdma-dev:arm64 \
        libdoca-flow-dev:arm64

# bindgen needs clang and DOCA headers accessible
ENV BINDGEN_EXTRA_CLANG_ARGS="--sysroot=/usr/aarch64-linux-gnu"
```

### 5.3 CI Matrix

```yaml
# .github/workflows/dpu-build.yml
jobs:
  build-dpu:
    strategy:
      matrix:
        doca_version: ["2.9.1"]  # Pin to LTS
        target: ["aarch64-unknown-linux-gnu"]
    steps:
      - uses: actions/checkout@v4
      - name: Install cross
        run: cargo install cross
      - name: Build outerlink-dpu
        run: cross build --target ${{ matrix.target }} -p outerlink-dpu --release
      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: outerlink-dpu-${{ matrix.target }}
          path: target/${{ matrix.target }}/release/outerlink-dpu
```

---

## 6. Resolved Open Questions

| # | Question (from v1) | Resolution |
|---|---|---|
| 1 | Acquire BF-2 before Phase A or start with SDK exploration? | **Start SDK exploration on x86_64 immediately.** DOCA SDK installs on x86_64 for host-side development. Build the FFI bindings and safe wrappers against x86_64 DOCA headers. Cross-compile to aarch64 once BF-2 arrives. Phase A deliverables split: FFI crate (no hardware needed) then DPU integration test (hardware needed). |
| 2 | Is there a DOCA software emulator? | **No production emulator exists.** NVIDIA provides DOCA Reference VM (QEMU-based) that emulates a BF-2 environment. It runs DOCA applications but without hardware acceleration -- compression runs in software, DMA is simulated. Sufficient for CI testing of control plane logic. Not sufficient for performance benchmarks. |
| 3 | Should doca-sys be open-source? | **Keep internal initially.** DOCA's API surface is large and unstable between versions. Publishing doca-sys creates a maintenance burden. Open-source once the API stabilizes (post-v1.0 of OuterLink). |
| 4 | BF-2 to BF-3 upgrade path? | **Same DOCA API, feature-gated at runtime.** `BfGeneration` enum detected at startup. BF-3 features (LZ4, APP cores) enabled via capability check. Same outerlink-dpu binary runs on both -- Rust `cfg` flags not needed, runtime detection sufficient. |
| 5 | Single binary or separate services? | **Single binary with internal modules.** Simpler deployment, shared memory between modules, single systemd unit. Internal modules (transport, compress, prefetch, router, page_table) run on pinned cores but share address space. |
| 6 | Comm Channel vs shared memory for host<->DPU? | **Start with Comm Channel, migrate to PCIe shared memory for page faults.** Comm Channel: ~5-10 us latency, sufficient for transfer requests and config updates (not on critical path). Page faults are latency-critical: shared memory via NTB gives ~1 us. Phase A uses Comm Channel exclusively. Phase B adds shared memory ring for page faults. |
| 7 | Prefetch model: train on host, deploy to DPU? | **Train and run on DPU.** The prefetch models (stride detection, Markov chains) are lightweight. Training is online (incremental, per-access updates), not batch ML. A72 cores handle this easily. The DPU sees traffic first, so training on DPU has lower-latency access to ground truth. |

---

## 7. Testing Strategy

### Without Hardware (CI)

- **doca-sys / doca-rs crate tests:** Mock DOCA FFI calls. Test that Rust wrappers handle error codes, null pointers, and lifecycle correctly.
- **Protocol tests:** Test `HostToDpuMsg` / `DpuToHostMsg` serialization, message ordering, flow control.
- **Algorithm tests:** Page table operations, routing, prefetch prediction -- pure Rust logic, no hardware needed.
- **DOCA Reference VM:** Run integration tests in QEMU-emulated BF-2 environment. Validates the DOCA call sequences work without real hardware.

### With Hardware (Dev Machine)

- **Phase A validation:** Host <-> DPU DMA at measured bandwidth. Compare against PCIe theoretical maximum.
- **Phase B validation:** End-to-end transfer with DPU handling RDMA. Latency and throughput vs. host-only baseline.
- **Phase D validation:** BAR1 access from DPU ConnectX to GPU VRAM. PCIe topology confirmed working.
- **Stress tests:** 24-hour sustained transfer runs. DPU stability under load. Memory leak detection on ARM cores.

---

## 8. Implementation Phases (Refined)

Unchanged from v1 in structure (A through E), but with concrete acceptance criteria informed by v2 structs:

### Phase A: DOCA Foundation (4-6 weeks)
- **Deliverable:** `doca-sys`, `doca-rs` crates. `outerlink-dpu` boots on BF-2, connects to host via Comm Channel. Proof of concept: DPU copies 1 GB at measured PCIe bandwidth.
- **Acceptance:** Comm Channel round-trip < 10 us. DMA throughput >= 25 GB/s (PCIe Gen4 x16).

### Phase B: Transport Offload (6-8 weeks)
- **Deliverable:** `TransportBackend` trait with `HostTransport` and `DpuTransport` implementations. Connection management on DPU. CUDA app runs unmodified with DPU handling transfers.
- **Acceptance:** Throughput >= 90% of host-only baseline. Latency <= 110% of host-only. Zero host CPU data path involvement (verified by CPU profiling).

### Phase C: Compression Offload (3-4 weeks)
- **Deliverable:** `DpuCompressor` with adaptive logic. Wire traffic compressed on DPU hardware.
- **Acceptance:** BF-2 deflate >= 8 GB/s. BF-3 LZ4 at wire rate. Compression adds < 2 us to transfer latency.

### Phase D: GPU BAR1 Integration (4-6 weeks)
- **Deliverable:** BAR1 regions mapped, DPU ConnectX writes directly to GPU VRAM. Zero-CPU end-to-end path.
- **Acceptance:** Remote VRAM -> wire -> DPU -> local GPU VRAM confirmed working. Latency < 6 us for 4 KB page.

### Phase E: Prefetch on DPU (3-4 weeks)
- **Deliverable:** `DpuPrefetchScheduler` running on ARM cores. Traffic monitoring and proactive fetching.
- **Acceptance:** Prefetch hit rate >= 60% on sequential workloads. Effective latency reduction >= 50% for predictable access patterns.

**Total: 20-28 weeks** (sequential phases, each depends on previous).

---

## Related Documents

- [preplan.md](./preplan.md) -- v1 pre-plan
- [research/01-bluefield-architecture.md](./research/01-bluefield-architecture.md)
- [research/02-programming-models.md](./research/02-programming-models.md)
- [research/03-outerlink-offload-design.md](./research/03-outerlink-offload-design.md)
- [R10: Memory Hierarchy](../R10-memory-hierarchy/)
- [R11: Speculative Prefetch](../R11-speculative-prefetch/)
- [R14: Transport Compression](../R14-transport-compression/)
- [R17: Topology-Aware Scheduling](../R17-topology-aware-scheduling/)
- [R21: NVMe-oF Storage](../R21-nvmeof-storage/)
- [R30: Persistent Kernels](../R30-persistent-kernels/)

## Open Questions

All v1 open questions resolved in Section 6. No new open questions at this stage -- remaining unknowns require hardware (BF-2) to resolve empirically.
