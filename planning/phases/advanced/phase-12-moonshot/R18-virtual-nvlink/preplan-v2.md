# R18: Virtual NVLink Emulation --- Pre-Plan v2 (Refined)

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** DRAFT
**Purpose:** Second-round refinement incorporating cross-topic findings from R10, R12, R17, R19, R20, R26, R29 v2 designs. Defines exact Rust structs, interception flows, RDMA atomic mapping, coherency integration, and performance guardrails.

---

## 1. Cross-Topic Integration Summary

This v2 refinement incorporates concrete designs from:

| Source | What It Provides to R18 |
|--------|------------------------|
| R10 v2 | PageTable with 64-byte PTE, coherency state (I/S/E) stored IN the PTE, 13-bit sharer bitmap |
| R19 v2 | Complete I/S/E state machine with 13 message types, directory-based coherency fully designed |
| R12 v2 | Deduped pages permanently locked in Shared state -- zero coherency overhead for model weights |
| R17 v2 | Topology determines peer connectivity, weighted routing for multi-hop transfers |
| R20 v2 | NCCL backend already handles P2P transport -- NVLink emulation extends this path |
| R26 v2 | PTP clock sync for coordinated atomic operations across nodes |
| R29 v2 | RDMA multicast for bulk invalidation (coherency fast-path) |

**Key insight from cross-topic analysis:** Tier 1 (peer access API) covers ~80% of real NVLink usage at LOW complexity. The remaining 20% requires the full coherency + atomic stack. This v2 focuses on making Tier 1 implementation-ready and defining exact interfaces for Tier 2/3.

---

## 2. Rust Struct Definitions

### 2.1 PeerAccessManager

Owns all peer access state. One instance per OuterLink client process.

```rust
/// Manages virtual NVLink peer access relationships between GPU contexts.
/// Intercepts cuDeviceCanAccessPeer, cuCtxEnablePeerAccess, cuDeviceGetP2PAttribute.
pub struct PeerAccessManager {
    /// Map of (local_device, remote_device) -> peer connection state
    peer_connections: DashMap<(DeviceId, DeviceId), PeerConnection>,

    /// Cluster topology snapshot from R17 TopologyManager
    topology: Arc<TopologySnapshot>,

    /// Performance thresholds for diagnostic warnings
    guardrails: PerformanceGuardrails,

    /// Reference to transport layer for data path
    transport: Arc<TransportManager>,

    /// Reference to R19 page fault handler for Tier 2+ address mapping
    page_fault_handler: Option<Arc<PageFaultHandler>>,

    /// Atomic engine for Tier 3 remote atomics
    atomic_engine: Option<Arc<RemoteAtomicEngine>>,
}

/// State of a single peer access connection (GPU A -> GPU B).
pub struct PeerConnection {
    /// Local device ID
    local_device: DeviceId,

    /// Remote device ID (may be on a different node)
    remote_device: DeviceId,

    /// Remote node ID in the cluster
    remote_node: NodeId,

    /// Whether peer access is currently enabled
    enabled: AtomicBool,

    /// Network path info from R17 topology
    route: RouteInfo,

    /// Measured latency to this peer (updated periodically)
    measured_latency_us: AtomicU32,

    /// Measured bandwidth to this peer in MB/s
    measured_bandwidth_mbps: AtomicU32,

    /// Statistics for diagnostic reporting
    stats: PeerAccessStats,
}

/// Route information from R17 topology-aware scheduling.
pub struct RouteInfo {
    /// Hop count from local to remote
    hop_count: u8,

    /// Weighted cost (lower = better, factors in bandwidth + latency)
    weighted_cost: f32,

    /// Whether this path uses RDMA (true) or TCP fallback (false)
    is_rdma: bool,

    /// If RDMA: the QP (Queue Pair) number for this connection
    rdma_qp: Option<u32>,
}

/// What we report for cuDeviceGetP2PAttribute queries.
pub struct PeerAttributes {
    /// CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED -- always 1 for virtual peers
    access_supported: i32,

    /// CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED
    /// Strategy: 1 if Tier 3 atomic engine is active, 0 otherwise.
    /// Configurable via OUTERLINK_REPORT_ATOMICS env var.
    native_atomic_supported: i32,

    /// CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK
    /// Derived from R17 topology: 0 = same node, 1 = 1-hop, 2 = 2+ hops
    performance_rank: i32,

    /// CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED
    /// 1 if both devices support UVA (same major compute capability)
    cuda_array_access_supported: i32,
}

/// Accumulated statistics per peer connection.
pub struct PeerAccessStats {
    /// Total bytes transferred via cudaMemcpyPeer on this connection
    bytes_transferred: AtomicU64,

    /// Total number of cudaMemcpyPeer calls
    transfer_count: AtomicU64,

    /// Number of small transfers (<4KB) -- latency-dominated, warn if high
    small_transfer_count: AtomicU64,

    /// Number of remote atomics executed (Tier 3)
    atomic_count: AtomicU64,

    /// Number of page faults triggered on this peer's memory (Tier 2)
    page_fault_count: AtomicU64,
}
```

### 2.2 RemoteAtomicEngine

Handles GPU atomic operations that target remote memory. Two modes: page migration (move data to requesting GPU) or remote proxy (execute atomic at home node).

```rust
/// Translates GPU atomic operations into RDMA atomics or proxy requests.
/// Only active when Tier 3 is enabled.
pub struct RemoteAtomicEngine {
    /// Per-node atomic proxy connections
    proxy_connections: DashMap<NodeId, AtomicProxyConnection>,

    /// Hot-atomic tracker: pages that see frequent cross-node atomics
    /// Key = page address, Value = access frequency and home node
    hot_atomics: DashMap<VirtualAddr, HotAtomicEntry>,

    /// Threshold: if a page sees > N atomics/sec from remote nodes,
    /// switch from migration to proxy mode
    migration_to_proxy_threshold: u32,

    /// RDMA memory regions registered for atomic access
    atomic_mr: Vec<RegisteredMemoryRegion>,

    /// PTP clock reference for coordinated atomic ordering (R26)
    clock: Arc<PtpClock>,
}

/// Tracks a hot atomic target page.
pub struct HotAtomicEntry {
    /// Which node owns the authoritative copy
    home_node: NodeId,

    /// Recent atomic frequency (atomics/sec from remote nodes)
    remote_frequency: AtomicU32,

    /// Current mode for this page
    mode: AtomicMode,

    /// Window of recent access timestamps for frequency calculation
    access_window: RingBuffer<Instant, 64>,
}

/// How remote atomics to a given page are handled.
#[derive(Clone, Copy, PartialEq)]
pub enum AtomicMode {
    /// Migrate page to requesting node, execute locally, then it lives there
    Migrate,

    /// Route atomic to home node's proxy server, execute there, return result
    Proxy,

    /// Use RDMA atomic directly (only for CAS/fetch-add on 64-bit values)
    RdmaDirect,
}

/// Maps a CUDA GPU atomic to a network operation.
pub struct AtomicTranslation {
    /// Original CUDA atomic type
    cuda_op: CudaAtomicOp,

    /// Target address (virtual, may be remote)
    target_addr: VirtualAddr,

    /// Operand value(s)
    operands: AtomicOperands,

    /// Chosen network implementation
    network_op: NetworkAtomicOp,
}

/// RDMA connection to a remote node's atomic proxy service.
pub struct AtomicProxyConnection {
    /// Remote node ID
    node_id: NodeId,

    /// RDMA RC QP for atomic operations
    qp: RdmaQueuePair,

    /// Remote proxy's memory region key (for RDMA atomics)
    remote_mr_key: u32,

    /// Doorbell address for proxy requests (RDMA write target)
    doorbell_addr: u64,

    /// Response buffer (RDMA write target for proxy responses)
    response_buffer: RegisteredMemoryRegion,
}

/// All CUDA atomic operations that need network translation.
#[derive(Clone, Copy)]
pub enum CudaAtomicOp {
    Add { size: AtomicSize, is_float: bool },
    Sub { size: AtomicSize },
    Min { size: AtomicSize, is_signed: bool },
    Max { size: AtomicSize, is_signed: bool },
    Exch { size: AtomicSize },
    CAS { size: AtomicSize },
    And { size: AtomicSize },
    Or { size: AtomicSize },
    Xor { size: AtomicSize },
    Inc { modulo: u32 },
    Dec { modulo: u32 },
}

#[derive(Clone, Copy)]
pub enum AtomicSize {
    Bits16,  // SM 7.0+ only
    Bits32,
    Bits64,
}

/// Network-level atomic operation after translation.
pub enum NetworkAtomicOp {
    /// Direct RDMA CAS (64-bit only)
    RdmaCas { compare: u64, swap: u64 },

    /// Direct RDMA Fetch-and-Add (64-bit only)
    RdmaFetchAdd { addend: u64 },

    /// CAS emulation loop for operations without direct RDMA mapping
    CasEmulationLoop { compute_fn: CasComputeFn },

    /// Proxy request: send operation to home node for local execution
    ProxyRequest { op: CudaAtomicOp, operands: AtomicOperands },
}

/// Describes the compute step inside a CAS emulation loop.
/// loop: old = read(addr); new = f(old, operand); if CAS(addr, old, new) == old: done
pub enum CasComputeFn {
    Min { is_signed: bool },
    Max { is_signed: bool },
    And,
    Or,
    Xor,
    Inc { modulo: u32 },
    Dec { modulo: u32 },
    FloatAdd,
    Exchange, // CAS(addr, old, new_value) -- always succeeds
}
```

### 2.3 NvlinkEmulator

Top-level facade that ties together peer access, coherency, and atomics.

```rust
/// Top-level Virtual NVLink emulation layer.
/// Registered as an interception handler during OuterLink client initialization.
pub struct NvlinkEmulator {
    /// Tier 1: Peer access management and P2P memcpy routing
    peer_access: Arc<PeerAccessManager>,

    /// Tier 2: Coherency integration (wraps R19's I/S/E protocol)
    /// None if only Tier 1 is active.
    coherency: Option<Arc<NvlinkCoherencyAdapter>>,

    /// Tier 3: Remote atomic engine
    /// None if only Tier 1/2 is active.
    atomics: Option<Arc<RemoteAtomicEngine>>,

    /// Active tier level (runtime-configurable)
    active_tier: AtomicU8,

    /// Diagnostic reporter for performance warnings
    diagnostics: Arc<PerformanceDiagnostics>,
}

/// Adapts R19's I/S/E coherency protocol for NVLink semantics.
/// This is NOT a new coherency protocol -- it wraps R19's existing one
/// and adds NVLink-specific behavior (atomic-aware transitions, fencing).
pub struct NvlinkCoherencyAdapter {
    /// R19's directory-based coherency manager
    directory: Arc<CoherencyDirectory>,

    /// R10's page table (64-byte PTEs with I/S/E state and 13-bit sharer bitmap)
    page_table: Arc<PageTable>,

    /// R29 multicast group for bulk invalidation fast-path
    invalidation_multicast: Option<MulticastGroup>,

    /// Fencing state per QP for memory ordering guarantees
    fence_state: DashMap<QueuePairId, FenceState>,
}

/// Fencing state to implement NVLink memory ordering over RDMA.
pub struct FenceState {
    /// Last fence sequence number issued
    last_fence_seq: AtomicU64,

    /// Last fence sequence number completed
    completed_fence_seq: AtomicU64,

    /// Pending operations that need fence before proceeding
    pending_after_fence: Mutex<Vec<PendingOp>>,
}
```

---

## 3. Tier 1 Implementation: Peer Access API

### 3.1 Interception Flow: cuDeviceCanAccessPeer

```
Application calls cuDeviceCanAccessPeer(&canAccess, devA, devB)
    |
    v
OuterLink LD_PRELOAD intercepts
    |
    v
Is devA == devB?
    |-- YES: forward to real CUDA driver (local access, always works)
    |-- NO:
        |
        v
    Are both devices managed by OuterLink?
        |-- NO: forward to real CUDA driver (let CUDA handle real NVLink/PCIe)
        |-- YES:
            |
            v
        Query R17 TopologySnapshot: is there a network path devA -> devB?
            |-- NO: *canAccess = 0 (unreachable node)
            |-- YES: *canAccess = 1
            |
            v
        Return CUDA_SUCCESS
```

### 3.2 Interception Flow: cuCtxEnablePeerAccess

```
Application calls cuCtxEnablePeerAccess(peerCtx, flags)
    |
    v
OuterLink intercepts
    |
    v
Identify peerCtx -> (node_id, device_id) from context map
    |
    v
Is this a remote (network) device?
    |-- NO: forward to real CUDA driver
    |-- YES:
        |
        v
    Already enabled? -> return CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED
        |
        v
    Create PeerConnection:
      1. Query R17 for RouteInfo (hop count, bandwidth, RDMA available?)
      2. If RDMA available: establish RC QP for this peer pair
      3. If Tier 2 active: register remote memory range with R19 PageFaultHandler
         - Map remote GPU's virtual address range into local address space
         - Configure page fault handler to fetch from remote_node on access
      4. If Tier 3 active: set up AtomicProxyConnection to remote node
      5. Store PeerConnection in peer_connections map
      6. Mark enabled = true
        |
        v
    Return CUDA_SUCCESS
```

### 3.3 Interception Flow: cudaMemcpyPeer / cudaMemcpyPeerAsync

```
Application calls cudaMemcpyPeer(dst, dstDev, src, srcDev, count)
    |
    v
OuterLink intercepts
    |
    v
Are src and dst on different OuterLink-managed nodes?
    |-- NO: forward to real CUDA driver
    |-- YES:
        |
        v
    Look up PeerConnection for (srcDev, dstDev)
        |
        v
    Is connection enabled?
        |-- NO: return CUDA_ERROR_PEER_ACCESS_NOT_ENABLED
        |-- YES:
            |
            v
        Route through TransportManager:
          1. Pin source memory if not already pinned
          2. For RDMA path:
             a. RDMA READ from source GPU's BAR1 (if OpenDMA Phase 5 available)
             b. OR: GPU->host memcpy + RDMA WRITE + host->GPU memcpy (host-staged)
          3. For TCP path:
             a. GPU->host memcpy + TCP send + host->GPU memcpy
          4. Pipeline: chunk large transfers into 2MB segments, overlap send/recv
            |
            v
        For async variant: return immediately, completion signaled via CUDA event
        For sync variant: block until transfer complete
            |
            v
        Update PeerAccessStats (bytes_transferred, transfer_count)
        If count < 4096: increment small_transfer_count
            |
            v
        Return CUDA_SUCCESS
```

### 3.4 Interception Flow: cuDeviceGetP2PAttribute

```
Application calls cuDeviceGetP2PAttribute(&value, attr, srcDev, dstDev)
    |
    v
OuterLink intercepts
    |
    v
Both devices managed by OuterLink?
    |-- NO: forward to real CUDA driver
    |-- YES:
        |
        v
    Build PeerAttributes from current state:
      - ACCESS_SUPPORTED: 1 (always, for managed peers)
      - NATIVE_ATOMIC_SUPPORTED:
          Check active_tier >= 3 AND env OUTERLINK_REPORT_ATOMICS != "0"
          If yes: 1. If no: 0.
      - PERFORMANCE_RANK:
          route.hop_count (0 = same node, 1 = 1-hop RDMA, 2+ = multi-hop or TCP)
      - CUDA_ARRAY_ACCESS_SUPPORTED:
          1 if both devices have same major compute capability
        |
        v
    *value = relevant attribute
    Return CUDA_SUCCESS
```

---

## 4. RDMA Atomic Mapping: Complete Translation Table

### 4.1 Direct RDMA Mapping (no emulation loop)

| CUDA Atomic | RDMA Operation | Translation |
|-------------|---------------|-------------|
| `atomicCAS(addr, compare, swap)` 64-bit | `IBV_WR_ATOMIC_CMP_AND_SWP` | Direct 1:1. Post: `{compare_add=compare, swap=swap, remote_addr=addr}`. Returns old value. |
| `atomicAdd(addr, val)` 64-bit int | `IBV_WR_ATOMIC_FETCH_AND_ADD` | Direct 1:1. Post: `{compare_add=val, remote_addr=addr}`. Returns old value. |
| `atomicSub(addr, val)` 64-bit int | `IBV_WR_ATOMIC_FETCH_AND_ADD` | Negate val: `compare_add = (0u64.wrapping_sub(val as u64))`. Returns old value. |
| `atomicExch(addr, val)` 64-bit | `IBV_WR_ATOMIC_CMP_AND_SWP` loop | Read current, CAS(current, val). Usually 1 iteration. |

### 4.2 CAS Emulation Loop (software, for everything else)

Every non-directly-mappable atomic follows this pattern:

```rust
fn cas_emulation_loop(
    addr: u64,
    operand: u64,
    compute: impl Fn(u64, u64) -> u64,  // (old_value, operand) -> new_value
    qp: &RdmaQueuePair,
    mr: &MemoryRegion,
) -> u64 {
    let max_retries = 32; // Bound retries to prevent livelock
    for attempt in 0..max_retries {
        // Step 1: Read current value
        let old = rdma_read_64(qp, mr, addr);

        // Step 2: Compute new value locally
        let new_val = compute(old, operand);

        // Step 3: CAS -- if old hasn't changed, swap succeeds
        let actual = rdma_cas_64(qp, mr, addr, old, new_val);

        if actual == old {
            return old; // Success, return old value
        }
        // Someone else modified it -- retry with actual as new old
        // Exponential backoff not needed: RDMA RTT (~2-5us) provides natural spacing
    }
    // Exceeded retries: fall back to proxy mode for this address
    proxy_fallback(addr, operand, compute)
}
```

### 4.3 Complete CAS Emulation Translation

| CUDA Atomic | Compute Function | Latency (no contention) | Latency (moderate contention) |
|-------------|-----------------|------------------------|------------------------------|
| `atomicMin(addr, val)` i32/i64 | `\|old, val\| std::cmp::min(old as i64, val as i64) as u64` | ~4-8us (1 read + 1 CAS) | ~8-20us (2-5 retries) |
| `atomicMax(addr, val)` i32/i64 | `\|old, val\| std::cmp::max(old as i64, val as i64) as u64` | ~4-8us | ~8-20us |
| `atomicMin(addr, val)` u32/u64 | `\|old, val\| std::cmp::min(old, val)` | ~4-8us | ~8-20us |
| `atomicMax(addr, val)` u32/u64 | `\|old, val\| std::cmp::max(old, val)` | ~4-8us | ~8-20us |
| `atomicAnd(addr, val)` | `\|old, val\| old & val` | ~4-8us | ~8-20us |
| `atomicOr(addr, val)` | `\|old, val\| old \| val` | ~4-8us | ~8-20us |
| `atomicXor(addr, val)` | `\|old, val\| old ^ val` | ~4-8us | ~8-20us |
| `atomicInc(addr, modulo)` | `\|old, m\| if old >= m { 0 } else { old + 1 }` | ~4-8us | ~8-20us |
| `atomicDec(addr, modulo)` | `\|old, m\| if old == 0 \|\| old > m { m } else { old - 1 }` | ~4-8us | ~8-20us |
| `atomicAdd(addr, val)` f32 | `\|old, val\| f32::to_bits(f32::from_bits(old as u32) + f32::from_bits(val as u32)) as u64` | ~4-8us | ~8-20us |
| `atomicAdd(addr, val)` f64 | `\|old, val\| f64::to_bits(f64::from_bits(old) + f64::from_bits(val))` | ~4-8us | ~8-20us |

### 4.4 32-bit Atomic Handling

RDMA atomics are 64-bit only. For 32-bit CUDA atomics:

```rust
fn atomic_32bit_via_64bit_cas(
    addr: u64,       // Must be 4-byte aligned
    val: u32,
    compute_32: impl Fn(u32, u32) -> u32,
    qp: &RdmaQueuePair,
    mr: &MemoryRegion,
) -> u32 {
    // Align to 8-byte boundary
    let aligned_addr = addr & !7u64;
    let offset = (addr & 7) as u32; // 0 or 4

    cas_emulation_loop(aligned_addr, val as u64, |old_64, operand| {
        let (lo, hi) = (old_64 as u32, (old_64 >> 32) as u32);
        if offset == 0 {
            let new_lo = compute_32(lo, operand as u32);
            ((hi as u64) << 32) | (new_lo as u64)
        } else {
            let new_hi = compute_32(hi, operand as u32);
            ((new_hi as u64) << 32) | (lo as u64)
        }
    }, qp, mr) as u32
}
```

### 4.5 16-bit Atomic Handling (SM 7.0+)

Same approach as 32-bit but with 16-bit offset within the 64-bit word. Four possible positions (byte offsets 0, 2, 4, 6).

---

## 5. Coherency Integration: How R19's I/S/E Serves NVLink Semantics

### 5.1 R19's Existing Protocol (What We Reuse)

R19 v2 defines a directory-based coherency protocol with 13 message types and an I/S/E state machine. R18 does NOT create a new protocol. It adds an adapter layer on top.

**R10's PTE structure (64 bytes) already contains:**
- Coherency state: I (Invalid), S (Shared), E (Exclusive) -- 2 bits
- Sharer bitmap: 13 bits (one per possible node in cluster)
- Home node ID: identifies the authoritative copy
- Lock bit: for atomic-aware transitions

**R12's dedup optimization:** Pages identified as duplicates (model weights, shared read-only data) are permanently locked in Shared state. The sharer bitmap has all-ones for participating nodes. No coherency transitions ever occur for these pages. This is critical because model weights are the largest memory consumers and the most widely shared -- taking them out of the coherency hot path eliminates the biggest source of potential thrashing.

### 5.2 NVLink Coherency Adapter: State Transitions

The adapter maps NVLink memory operations to R19 coherency protocol messages:

**Read (GPU load from remote pointer):**
```
GPU A loads addr that maps to page P owned by Node B
    |
    v
Page P is in state I (Invalid) at Node A
    |
    v
R19 sends FetchShared(page=P) to home node (directory)
    |
    v
Home node checks PTE:
  - If page is Exclusive at Node B:
      Send Downgrade(P) to Node B
      Node B transitions E -> S, sends page data to directory
      Directory sends page data to Node A
      Node A installs page in state S
  - If page is Shared (possibly at other nodes too):
      Directory sends page data to Node A
      Node A installs page in state S
      Directory updates sharer bitmap: set bit for Node A
    |
    v
GPU A's load completes (page is now local in S state)
```

**Write (GPU store to remote pointer):**
```
GPU A stores to addr that maps to page P
    |
    v
If page P is in state S at Node A:
    R19 sends FetchExclusive(page=P) to home node
        |
        v
    Home node invalidates all other sharers:
      - For each node in sharer bitmap (except Node A):
          If R29 multicast available:
              Single multicast Invalidate(P) to all sharers
          Else:
              Unicast Invalidate(P) to each sharer
      - Wait for InvalidateAck from all sharers
      - Node A transitions S -> E
      - Directory clears all sharer bits except Node A
        |
        v
    GPU A's store proceeds (page is now Exclusive at Node A)

If page P is in state E at Node A:
    Store proceeds immediately (no protocol messages needed)

If page P is in state I at Node A:
    R19 sends FetchExclusive(page=P) to home node
    Same as above but starts from Invalid
```

**Atomic operation on remote page:**
```
GPU A issues atomicAdd to addr on page P
    |
    v
Check HotAtomicEntry for page P:
  - If mode == RdmaDirect (64-bit CAS/fetch-add):
      Execute RDMA atomic directly on home node's memory
      No page migration needed
      No coherency state change (home node retains ownership)

  - If mode == Proxy:
      Send AtomicProxyRequest to home node
      Home node executes atomic on local memory
      Returns old value to Node A
      No page migration, no coherency state change

  - If mode == Migrate:
      FetchExclusive(P) -- same as write path above
      Execute atomic locally (now in E state)
      Page lives on Node A until someone else needs it

Decision logic for mode selection:
  - 64-bit CAS or fetch-add? -> RdmaDirect
  - Page hot (>1000 remote atomics/sec)? -> Proxy
  - Otherwise -> Migrate
```

### 5.3 R29 Multicast Integration for Bulk Invalidation

When a page transitions from Shared-by-many to Exclusive:

**Without R29:** N-1 unicast Invalidate messages, each requiring an RTT (~2-5us). For 8 sharers: 7 * 2us = 14us minimum.

**With R29:** Single multicast Invalidate, one RTT (~2-5us) regardless of sharer count. All sharers receive simultaneously, ACK individually. Total: ~2-5us + straggler latency.

The NvlinkCoherencyAdapter checks `invalidation_multicast.is_some()` and uses the multicast path when available. The sharer bitmap from R10's PTE determines the multicast group membership.

### 5.4 Memory Ordering via RDMA Fences

NVLink guarantees: writes on the same link appear in order. `__threadfence_system()` ensures cross-link ordering.

RDMA mapping:
- **Same QP writes:** Already ordered (RDMA spec guarantees in-order write completion within a QP). No extra work.
- **Post-atomic ordering:** Use `IBV_SEND_FENCE` flag on the operation following an atomic. This ensures the atomic completes before the next operation starts.
- **Cross-GPU fence (__threadfence_system):** Translate to a fence message:
  1. Issue all pending RDMA writes
  2. Wait for all completions (poll CQ)
  3. Send fence-complete notification to all connected peers
  4. Fence returns only after all peers ACK

Cost: ~2-5us per fence (one network RTT). Applications with frequent `__threadfence_system()` will see significant overhead.

---

## 6. Performance Guardrails

### 6.1 Diagnostic Warning System

```rust
pub struct PerformanceGuardrails {
    /// Warn if more than N small (<4KB) transfers per second to a peer
    small_transfer_warn_threshold: u32, // default: 10_000

    /// Warn if page fault rate exceeds N/sec for any peer
    page_fault_warn_threshold: u32, // default: 1_000

    /// Warn if remote atomic rate exceeds N/sec for any address
    atomic_hotspot_warn_threshold: u32, // default: 5_000

    /// Warn if page ping-pong count exceeds N for any page in a 1-sec window
    page_thrash_warn_threshold: u32, // default: 10

    /// Warn if fence frequency exceeds N/sec
    fence_frequency_warn_threshold: u32, // default: 50_000

    /// Whether warnings are currently enabled
    enabled: AtomicBool,

    /// Log output target
    log_target: LogTarget,
}
```

### 6.2 Warning Messages (examples)

```
[OuterLink PERF] WARNING: 15,234 small transfers (<4KB) to GPU 3 in last second.
  NVLink handles these at ~100ns each. Network latency is ~2-5us each.
  Consider batching transfers with cudaMemcpyPeerAsync + larger buffers.

[OuterLink PERF] WARNING: Page 0x7f8000400000 ping-ponging between Node 0 and Node 2.
  Migrated 47 times in last second (S->E->S->E cycle).
  This indicates false sharing -- two GPUs accessing different data on the same 64KB page.
  Consider: (1) aligning data to page boundaries, (2) using OUTERLINK_PAGE_SIZE=4K.

[OuterLink PERF] WARNING: 12,891 remote atomics/sec to address 0x7f8000800100.
  Each costs ~2-5us over network (vs ~100ns on NVLink).
  Consider local accumulation + final reduction instead of per-thread remote atomics.

[OuterLink PERF] WARNING: __threadfence_system() called 85,000 times/sec.
  Each fence costs ~2-5us (network RTT). Total fence overhead: ~170-425ms/sec.
  This workload is synchronization-bound. Consider reducing fence frequency.
```

### 6.3 Configurable Tuning Knobs

| Environment Variable | Values | Default | Effect |
|---------------------|--------|---------|--------|
| `OUTERLINK_NVLINK_TIER` | `1`, `2`, `3` | `1` | Maximum emulation tier |
| `OUTERLINK_REPORT_ATOMICS` | `0`, `1` | `0` | Report NATIVE_ATOMIC_SUPPORTED to apps |
| `OUTERLINK_ATOMIC_MODE` | `migrate`, `proxy`, `auto` | `auto` | Force atomic handling mode |
| `OUTERLINK_PAGE_SIZE` | `4K`, `64K` | `64K` | Coherency page size (smaller = less false sharing, more overhead) |
| `OUTERLINK_PERF_WARNINGS` | `0`, `1`, `verbose` | `1` | Diagnostic output level |
| `OUTERLINK_FENCE_BATCH` | `1`-`100` | `1` | Batch N fence operations into one network RTT |
| `OUTERLINK_PEER_RANK_OFFSET` | `0`-`10` | `0` | Added to PERFORMANCE_RANK (higher = apps may avoid P2P) |

---

## 7. Realistic vs Aspirational Assessment

### What Is Realistic (HIGH confidence, will work)

| Feature | Confidence | Why |
|---------|-----------|-----|
| Peer access API interception (Tier 1) | 99% | Pure software interception, no hardware dependency |
| cudaMemcpyPeer routing over network | 99% | Core OuterLink functionality |
| P2P attribute reporting | 99% | Return whatever values we choose |
| Performance diagnostics | 95% | Instrumentation is straightforward |
| NCCL backend integration (R20) | 90% | NCCL's plugin architecture is designed for this |
| Page-fault-based unified address space (Tier 2) | 80% | Depends on R19 working, which has known risks |

### What Is Aspirational (MEDIUM confidence, might work with caveats)

| Feature | Confidence | Caveat |
|---------|-----------|--------|
| Software I/S/E coherency at 64KB pages | 70% | False sharing will be a problem for some workloads |
| RDMA direct atomics (CAS, fetch-add) | 75% | Depends on RDMA infra; latency is 10-50x NVLink |
| CAS emulation for full atomic set | 65% | Correctness is achievable, but contention degrades badly |
| Atomic proxy server | 60% | Lock-free server design is hard; contention scaling unknown |
| Memory ordering via RDMA fences | 70% | Correct but expensive; apps with many fences will suffer |

### What Is NOT Realistic (LOW confidence, do not promise)

| Feature | Why Not |
|---------|---------|
| NVLink-speed anything | Physics. 40-300x bandwidth gap. |
| Cache-line-level coherency (128B) | Would need 512x more directory entries than 64KB pages. Network latency makes sub-page coherency pointless. |
| Transparent performance for random-access workloads | Every cold access = page fault = 10-50us. No fix for this. |
| Full correctness for CUDA UVM interop | CUDA's own UVM may conflict with our page management. Unknown interaction. |
| >8 node coherent cluster | Protocol message overhead scales linearly with sharers. Beyond 8 nodes, use explicit transfers. |

### Honest Bottom Line

**Tier 1 is a near-certainty and delivers ~80% of value.** Applications using NCCL + cudaMemcpyPeer will work transparently. This covers PyTorch DDP, TensorFlow distributed, most inference engines.

**Tier 2 is medium-risk and depends entirely on R19.** If page faults work reliably, unified address space works. If R19 proves impractical, Tier 2 is dead and we fall back to explicit-transfer-only.

**Tier 3 is research territory.** Correct software atomics over RDMA will work for low-contention patterns. High-contention atomics (reductions, histograms) should NOT use Tier 3 -- they should use local-accumulate-and-merge via R25 or application restructuring.

---

## 8. Revised Milestones (with cross-topic integration)

### M1: Peer Access API (Tier 1a) -- 1-2 weeks after Phase 3

- Intercept cuDeviceCanAccessPeer, cuCtxEnablePeerAccess, cuDeviceGetP2PAttribute
- PeerAccessManager struct, PeerConnection establishment
- All peer access return values correct
- `simpleP2P` CUDA sample passes (canAccessPeer check)
- **No dependency on R10/R19**

### M2: P2P Transfer Routing (Tier 1b) -- 1-2 weeks after Phase 4

- cudaMemcpyPeer/cudaMemcpyPeerAsync routed through TransportManager
- Pipeline chunking (2MB segments) for large transfers
- Async transfers via CUDA stream integration
- Bandwidth within 2x of raw transport bandwidth
- Performance diagnostics for small-transfer warnings
- **Integrates with R20 NCCL backend** (NCCL's P2P send/recv uses the same path)

### M3: Unified Address Space (Tier 2a) -- 3-4 weeks after R19 validated

- NvlinkCoherencyAdapter wraps R19's I/S/E protocol
- Virtual address mapping spans all GPUs in cluster
- Page faults trigger FetchShared/FetchExclusive via R19 protocol
- R12 dedup pages permanently Shared (no coherency transitions for model weights)
- cudaMallocManaged works across network GPUs
- **Requires R10 PageTable + R19 PageFaultHandler**

### M4: Coherency Optimization (Tier 2b) -- 2-3 weeks after M3

- R29 multicast invalidation integration (single message vs N unicast)
- Page thrashing detection and mitigation (warn + suggest page alignment)
- R11 prefetch integration (prefetch adjacent pages on fault pattern)
- R17 topology-aware home node selection (minimize hop count for hot pages)
- Streaming workloads within 3-5x of local VRAM speed

### M5: Remote Atomics (Tier 3a) -- 3-4 weeks after M3 + RDMA transport

- RemoteAtomicEngine with all three modes (RdmaDirect, CasEmulation, Proxy)
- RDMA CAS and Fetch-Add for 64-bit atomics
- CAS emulation loops for all remaining atomic types
- 32-bit atomic handling via 64-bit CAS with alignment
- Hot-atomic detection and automatic mode switching
- Correctness tests under contention (multi-threaded atomic counter)

### M6: Full Integration + Ordering (Tier 3b) -- 4-6 weeks after M5

- Atomic-aware coherency (atomic on S page -> acquire E first)
- Memory ordering via RDMA fences + IBV_SEND_FENCE
- __threadfence_system() translation to network fence
- AtomicProxyConnection for high-contention pages
- R26 PTP integration for cross-node atomic ordering
- Full performance diagnostics suite
- Application benchmarks: NCCL all_reduce_perf, PyTorch DDP training, LLM inference

---

## 9. Resolved Open Questions (from v1)

| # | Question | Resolution |
|---|----------|-----------|
| Q1 | Should M1 be pulled into Phase 6? | YES. Tier 1 has no dependency on R10/R19 and is a natural extension of Phase 3 CUDA interception. Recommend building M1/M2 as part of Phase 6. |
| Q2 | Is TLA+ worth it for coherency? | DEFERRED. R19 v2 already defines the state machine formally. TLA+ verification should happen as part of R19, not R18. R18 only adapts R19's protocol. |
| Q3 | Page size: R10's 64KB or smaller? | CONFIGURABLE. Default 64KB (R10's standard). Expose OUTERLINK_PAGE_SIZE=4K for false-sharing-sensitive workloads. 4KB increases page table entries by 16x but reduces false sharing. |
| Q4 | NVSHMEM concepts for atomic proxy? | YES, adopted. NVSHMEM's fence/quiet semantics map well to our RDMA fence approach. The proxy server concept is similar to NVSHMEM's symmetric heap. |
| Q5 | Minimum viable for alpha? | Tier 1 only (M1 + M2). Peer access API + cudaMemcpyPeer routing. No coherency, no atomics. |

---

## Related Documents

- [preplan.md](preplan.md) -- v1 pre-plan
- [research/01-nvlink-protocol.md](research/01-nvlink-protocol.md) -- NVLink protocol deep dive
- [research/02-rdma-atomics-and-coherency.md](research/02-rdma-atomics-and-coherency.md) -- RDMA atomic analysis
- [research/03-feasibility-and-limitations.md](research/03-feasibility-and-limitations.md) -- Feasibility assessment
- R10: Memory Tiering -- PageTable with 64-byte PTE, sharer bitmap
- R12: Memory Deduplication -- Deduped pages permanently Shared
- R17: Topology-Aware Scheduling -- RouteInfo, weighted routing
- R19: Network Page Faults -- I/S/E state machine, 13 message types
- R20: NCCL Backend -- P2P transport integration
- R26: PTP Clock Sync -- Coordinated atomic ordering
- R29: RDMA Multicast -- Bulk invalidation fast-path

## Open Questions (v2)

| # | Question | Status |
|---|----------|--------|
| Q1 | Can ConnectX-5 DMA engine issue PCIe atomic TLPs to BAR1? | OPEN -- if yes, enables hardware-path remote atomics via OpenDMA |
| Q2 | Actual RDMA CAS latency on our ConnectX-5 hardware? | OPEN -- need benchmarks before M5 |
| Q3 | How does R12 dedup interact with Exclusive transitions? | RESOLVED -- deduped pages are permanently Shared, never transition to E. Writes to dedup pages trigger CoW (copy-on-write) first. |
| Q4 | Impact of 4KB pages on TLB pressure? | OPEN -- need to measure. 16x more PTEs means 16x more TLB misses for large working sets. |
| Q5 | Can we batch atomic proxy requests? | OPEN -- if proxy receives 100 atomicAdd to same address, sum them and apply once. Reduces contention. |
| Q6 | How does NCCL respond to PERFORMANCE_RANK > 0? | OPEN -- need to test. NCCL may adjust ring topology or chunk sizes. |
