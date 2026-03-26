# R18 Research: NVLink Protocol Deep Dive

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Purpose:** Understand exactly what NVLink provides so we know what "Virtual NVLink" must emulate.

---

## 1. NVLink Generations and Bandwidth Progression

| Generation | Year | Arch | Signaling Rate | Links/GPU | BW per Link (bidir) | Total BW per GPU |
|------------|------|------|---------------|-----------|---------------------|-----------------|
| NVLink 1.0 | 2016 | Pascal (P100) | 20 Gbit/s | 4 | 40 GB/s | 160 GB/s |
| NVLink 2.0 | 2017 | Volta (V100) | 25 Gbit/s | 6 | 50 GB/s | 300 GB/s |
| NVLink 3.0 | 2020 | Ampere (A100) | 50 Gbit/s | 12 | 50 GB/s | 600 GB/s |
| NVLink 4.0 | 2022 | Hopper (H100) | 50 Gbit/s | 18 | 50 GB/s | 900 GB/s |
| NVLink 5.0 | 2024 | Blackwell (B200) | 100 Gbit/s | 18 | 100 GB/s | 1,800 GB/s |
| NVLink 6.0 | TBA | Rubin | TBA | TBA | TBA | 3,600 GB/s |

### Consumer NVLink (RTX 3090 / 3090 Ti)

| Parameter | Value |
|-----------|-------|
| NVLink version | 3.0 |
| Links | 4 x4 (reduced from A100's 12) |
| Aggregate bandwidth | 112.5 GB/s |
| Bidirectional per direction | 56.25 GB/s |
| Connector | Single NVLink bridge connector |
| Max peers | 1 (single connector) |
| Status | Last GeForce generation with NVLink (removed in RTX 40 series) |

### Key Bandwidth Context for OuterLink

| Connection | Bandwidth | Ratio to NVLink 4.0 |
|-----------|-----------|---------------------|
| NVLink 4.0 (H100) | 900 GB/s | 1.0x |
| NVLink 3.0 (RTX 3090) | 112.5 GB/s | 0.125x |
| PCIe 5.0 x16 | 64 GB/s | 0.071x |
| PCIe 4.0 x16 | 32 GB/s | 0.036x |
| 4x 100GbE bonded | ~50 GB/s | 0.056x |
| 1x 100GbE RDMA | ~12.5 GB/s | 0.014x |
| 1x 25GbE RDMA | ~3.1 GB/s | 0.003x |

**The gap is 40-300x depending on configuration.** This is the fundamental constraint R18 must acknowledge. We emulate semantics, not bandwidth.

---

## 2. NVLink Protocol: What It Actually Provides

NVLink is not just a faster wire. It provides a fundamentally different memory access model compared to PCIe:

### 2.1 Memory-Mapped Access (Load/Store)

With NVLink, GPU A can issue direct load and store instructions to GPU B's memory. The hardware handles address translation and data movement transparently. This is the same mechanism used to access local VRAM --- the GPU's load/store units simply target a different address range.

**What PCIe provides instead:** PCIe P2P allows MMIO (memory-mapped I/O) access to BAR regions, but this goes through the PCIe fabric with higher latency (~300-500ns vs NVLink's ~100ns). PCIe does NOT provide cache-coherent access --- reads go directly to the device, bypassing CPU caches.

### 2.2 Atomic Operations

NVLink supports native atomic operations between GPUs since NVLink 2.0 (Volta). These are hardware-level atomics that execute at the target GPU's memory controller:

| Atomic Operation | NVLink Support | Notes |
|-----------------|----------------|-------|
| atomicAdd | YES | 32-bit and 64-bit integer, 32-bit and 64-bit float |
| atomicSub | YES | Via atomicAdd with negated value |
| atomicMin / atomicMax | YES | 32-bit and 64-bit integer |
| atomicExch | YES | 32-bit and 64-bit |
| atomicCAS | YES | 32-bit and 64-bit (most general primitive) |
| atomicAnd / atomicOr / atomicXor | YES | 32-bit and 64-bit |
| atomicInc / atomicDec | YES | 32-bit |

Over NVLink, these atomics execute at the remote GPU's memory controller with sub-microsecond latency. The requesting GPU's thread stalls until the atomic completes and the old value is returned.

**What PCIe provides instead:** PCIe 2.1+ defines atomic operations (swap, fetch-and-add, compare-and-swap) but GPU drivers do not generally expose cross-GPU PCIe atomics to CUDA. NVLink atomics are handled at the GPU hardware level, not the PCIe level.

### 2.3 Cache Coherency

This is the most significant NVLink feature for Virtual NVLink emulation.

**NVLink 2.0+ (GPU-GPU coherency):**
- Memory accesses are coherent on 128-byte cache line boundaries
- GPU A's L2 cache can hold copies of GPU B's memory
- When GPU B modifies a line that GPU A has cached, the coherency protocol invalidates GPU A's copy
- This happens in hardware, transparently to CUDA code

**NVLink-C2C (Grace Hopper, GPU-CPU coherency):**
- Implements AMBA CHI cache coherence protocol
- Cache-line granularity (64 bytes on CPU side, 128 bytes on GPU side)
- CPU can cache GPU memory in its hierarchy
- GPU can cache CPU memory in its L1 caches
- All interconnected GH200 units act as a single cache-coherent system
- NO page faults needed --- coherent at cache-line level

**What PCIe provides instead:** PCIe is NOT cache-coherent between endpoints. GPU-to-GPU access over PCIe bypasses caches entirely (non-coherent MMIO). Any software coherency must be implemented explicitly.

### 2.4 Unified Address Space

With NVLink and peer access enabled, all GPU memories appear in a single flat address space. A pointer allocated on GPU 0 can be dereferenced on GPU 1 without any address translation visible to the application.

CUDA's Unified Virtual Addressing (UVA) provides this abstraction, but:
- **With NVLink:** Direct hardware access, low latency, cache-coherent
- **Without NVLink (PCIe):** Still works but may trigger page migration via CUDA UVM, with page fault overhead (~20-50us per fault)

---

## 3. NVSwitch: All-to-All NVLink Fabric

NVSwitch is the scaling mechanism for NVLink, converting point-to-point links into a fully connected fabric.

### NVSwitch Evolution

| System | GPUs | NVSwitch Gen | Switches | Total Fabric BW |
|--------|------|-------------|----------|----------------|
| DGX-2 (Volta) | 16 V100 | Gen 1 (18 ports) | 12 | 2.4 TB/s |
| DGX A100 | 8 A100 | Gen 2 | 6 | 4.8 TB/s |
| DGX H100 | 8 H100 | Gen 3 | 4 | 7.2 TB/s |
| DGX B200 | 8 B200 | Gen 4 | 2 | 14.4 TB/s |
| GB200 NVL72 | 72 B200 | Gen 4 + NVLink Switch | 18 switch trays | 130 TB/s |

### What NVSwitch Provides

1. **Non-blocking any-to-any:** Every GPU can talk to every other GPU at full bandwidth simultaneously
2. **Single-hop latency:** No multi-hop routing through intermediate GPUs
3. **SHARP in-network compute (Gen 3+):** AllReduce operations computed inside the switch itself
4. **Fabric Manager software:** Required for NVSwitch topology setup (proprietary, datacenter only)

### Why NVSwitch Matters for R18

NVSwitch means NVLink-based systems have O(1) latency for any GPU pair. Without NVSwitch, multi-GPU NVLink topologies form rings or meshes with multi-hop paths. Our network topology is inherently multi-hop (router/switch in between), making O(1) any-to-any impossible --- but our topology-aware scheduling (R17) can minimize hop count.

---

## 4. What NVLink Provides That PCIe Does Not

| Capability | NVLink | PCIe (P2P) | Network (RDMA) |
|-----------|--------|-----------|----------------|
| Direct load/store | YES (hardware) | Partial (BAR MMIO, non-coherent) | NO (requires explicit operations) |
| Cache coherency | YES (hardware, 128B lines) | NO | NO (must implement in software) |
| Native GPU atomics | YES (full set, hardware) | NO (not exposed by drivers) | Limited (CAS, fetch-add only) |
| Memory ordering | Sequential within a link | PCIe ordering rules | Fence-based (per QP) |
| Latency | ~100-300ns | ~300-500ns | ~1-2us (RDMA), ~50-100us (TCP) |
| Bandwidth | 112.5-1800 GB/s | 32-64 GB/s | 3-50 GB/s |
| Unified address space | Transparent (hardware) | With UVA (software-managed) | Must build (R19 page faults) |
| Page migration | Not needed (direct access) | CUDA UVM (page fault based) | Must build (R19) |

---

## 5. CUDA Unified Memory: NVLink vs Without

CUDA Unified Memory (cudaMallocManaged) behavior differs dramatically based on interconnect:

### With NVLink (Pascal+)

- **On-demand migration:** Pages migrate to accessing GPU on first touch (~5.4 GB/s on PCIe, higher on NVLink)
- **Prefetching:** `cudaMemPrefetchAsync` moves pages before access (~10.9 GB/s PCIe, better on NVLink)
- **Access counters:** Hardware tracks access frequency per page, driving migration decisions
- **Thrashing mitigation:** GPU throttles access to contested pages, tries to find stable placement
- **NVLink advantage:** 2-5x faster page migration than PCIe, larger effective oversubscription

### With NVLink-C2C (Grace Hopper)

- **No page migration needed:** Cache-line level coherency eliminates page faults entirely
- **CPU caches GPU memory:** CPU L1/L2 can hold GPU memory lines
- **GPU caches CPU memory:** GPU L1 can hold CPU memory lines
- **Transparent:** No cudaMemPrefetchAsync or hints needed

### Without NVLink (PCIe only)

- **Page faults expensive:** ~20-50us per fault on PCIe
- **Migration bandwidth limited:** PCIe bandwidth (~32 GB/s) vs NVLink (~112+ GB/s)
- **Thrashing severe:** Two GPUs repeatedly faulting same page kills performance
- **Hints critical:** Without cudaMemAdvise and cudaMemPrefetchAsync, performance can be 100x worse

### Implication for R18

Our Virtual NVLink operates in the "without NVLink" regime but worse --- network latency is 10-100x higher than PCIe. Page-fault-based migration (R19) is essential but must be combined with aggressive prefetching (R11) and intelligent page placement (R10) to be usable.

---

## 6. The Peer Access API

### cuDeviceCanAccessPeer / cudaDeviceCanAccessPeer

Queries whether GPU A can directly access GPU B's memory.

```c
int canAccess;
cuDeviceCanAccessPeer(&canAccess, deviceA, deviceB);
// canAccess = 1 if NVLink or PCIe P2P path exists
// canAccess = 0 otherwise
```

**What this checks (hardware):**
- Are both GPUs in the same PCIe root complex? (for PCIe P2P)
- Is there an NVLink connection between them?
- Are GPU types compatible? (same major compute capability for UVA)

**What OuterLink must do:** Intercept this call and return 1 for all virtual GPU pairs, regardless of physical connectivity. This is how we convince CUDA applications that peer access exists.

### cuCtxEnablePeerAccess / cudaDeviceEnablePeerAccess

Enables direct memory access between GPU contexts.

```c
cuCtxEnablePeerAccess(peerContext, 0);
// After this: allocations from peerContext are accessible in current context
```

**Key properties:**
- Unidirectional --- must be called symmetrically for bidirectional access
- System limit: 8 peer connections per device (hardware limit)
- Once enabled, all allocations from peer are immediately accessible

**What OuterLink must do:** Intercept this call, set up address mapping for the remote GPU's memory space, and configure the page fault handler (R19) to transparently fetch remote pages on access.

### cuDeviceGetP2PAttribute

Queries link properties between GPU pairs.

```c
int perfRank;
cuDeviceGetP2PAttribute(&perfRank, CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK, devA, devB);
// Also: ACCESS_SUPPORTED, NATIVE_ATOMIC_SUPPORTED, CUDA_ARRAY_ACCESS_SUPPORTED
```

**Critical attribute: `CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED`**

If this returns 1, CUDA apps assume atomics on remote memory work at hardware speed. We must intercept this carefully --- returning 1 means apps will use remote atomics freely, which may perform terribly over the network. Options:
1. Return 1 (optimistic) --- apps work but some may be slow
2. Return 0 (conservative) --- apps may fall back to slower but safer patterns
3. Return based on topology --- 1 for local NVLink pairs, 0 for network-remote

---

## 7. What Applications Actually USE NVLink For

### 7.1 P2P Memory Copy (cudaMemcpyPeer)

The most common use. Training frameworks copy gradients, activations, and model parameters between GPUs. NVLink provides 5-10x the bandwidth of PCIe for these transfers.

**OuterLink impact:** These become network transfers. Performance depends on transfer size:
- Large transfers (>1MB): amortized over bandwidth, works well
- Small transfers (<4KB): dominated by latency, 100x slower than NVLink

### 7.2 NCCL Collective Operations

NCCL (NVIDIA Collective Communications Library) is the primary user of NVLink in ML training:
- **AllReduce:** Sum gradients across GPUs (ring or tree topology over NVLink)
- **AllGather:** Collect tensor shards from all GPUs
- **Broadcast:** Distribute model weights to all GPUs
- **ReduceScatter:** Distributed reduction

NCCL auto-detects NVLink and builds optimal communication rings. With NVSwitch, it uses all-to-all patterns.

**OuterLink impact:** R20 (NCCL Backend) registers as a transport plugin. NCCL doesn't care about the underlying hardware as long as send/recv work. Performance depends on bandwidth, not NVLink-specific features.

### 7.3 Direct Memory Access (GPU Pointers)

Some CUDA code directly dereferences pointers to remote GPU memory:

```cuda
__global__ void kernel(float* remote_data) {
    float val = remote_data[threadIdx.x];  // Direct load from remote GPU
    // ...
}
```

With NVLink, this "just works" with hardware coherency. Without NVLink, it triggers CUDA UVM page faults.

**OuterLink impact:** This is the hardest case. Every load/store to remote memory potentially triggers a page fault. R19's page fault handler + R11's prefetching are essential. Performance is workload-dependent --- streaming access patterns work, random access kills performance.

### 7.4 Remote Atomics

Distributed synchronization primitives (locks, counters, barriers) that operate on memory hosted by other GPUs:

```cuda
atomicAdd(&remote_counter, 1);  // Counter lives on GPU B, executed on GPU A
```

**OuterLink impact:** Must be intercepted and translated to network atomic operations. Latency goes from ~100ns (NVLink) to ~2-10us (RDMA atomic), a 20-100x increase. Fine for occasional synchronization, devastating for tight loops.

### 7.5 Cooperative Groups / Grid Sync

CUDA cooperative groups allow synchronization across thread blocks, potentially across GPUs. `cudaLaunchCooperativeKernelMultiDevice` launches synchronized kernels across GPUs.

**OuterLink impact:** Depends on R25 (Cooperative Kernel Splitting) and R26 (PTP Clock Sync). Cross-GPU synchronization requires network round-trips, limiting sync frequency to ~100K-500K syncs/second (vs millions with NVLink).

---

## 8. Summary: What Virtual NVLink Must Emulate

### Must-Have (correctness)

| Feature | Priority | Mechanism |
|---------|----------|-----------|
| Peer access reporting | CRITICAL | Intercept cuDeviceCanAccessPeer, return 1 |
| Peer access enable/disable | CRITICAL | Intercept cuCtxEnablePeerAccess, set up mapping |
| Unified address space | CRITICAL | R19 page fault handler + address mapping |
| P2P memory copy | CRITICAL | Intercept cudaMemcpyPeer, use transport layer |
| Basic atomics (CAS, fetch-add) | HIGH | RDMA atomics or software emulation |
| Memory ordering | HIGH | Fence-based ordering via transport |

### Nice-to-Have (performance)

| Feature | Priority | Mechanism |
|---------|----------|-----------|
| Cache coherency | MEDIUM | R19 I/S/E protocol (software) |
| Full atomic set (min, max, and, or, xor) | MEDIUM | Software emulation via CAS |
| Direct load/store | LOW | Page fault + prefetch (software, never hardware-speed) |
| NVSwitch-like any-to-any | LOW | Topology-aware routing (R17) |

### Cannot Emulate (honest limitations)

| Feature | Why |
|---------|-----|
| NVLink bandwidth (112-1800 GB/s) | Physics --- network is 40-300x slower |
| Sub-microsecond atomic latency | Network RTT is minimum ~1-2us |
| Hardware cache coherency speed | Software coherency adds microseconds per operation |
| Zero-overhead direct load/store | Page faults add ~2-50us per cold access |

---

## Related Documents

- [R6: NVLink Cross-PC](../../../../research/R6-nvlink-cross-pc.md) --- Physical NVLink limitations
- [R4: ConnectX-5 Transport](../../../../research/R4-connectx5-transport-stack.md) --- Transport capabilities
- [R19: Network Page Faults](../../phase-08-smart-memory/R19-network-page-faults/README.md) --- Page fault mechanism
- [R25: Cooperative Kernel Splitting](../R25-cooperative-kernel-splitting/README.md) --- Cross-GPU kernel execution

## Sources

1. NVIDIA NVLink Wikipedia / WikiChip specifications
2. NVIDIA NVLink & NVSwitch product page
3. NVSwitch Technical Overview (Hot Chips 30)
4. CUDA Programming Guide --- Peer Device Memory Access
5. CUDA Driver API --- cuDeviceCanAccessPeer, cuCtxEnablePeerAccess
6. NVIDIA Grace Hopper Architecture Blog --- NVLink-C2C coherency
7. NVIDIA Maximizing Unified Memory Performance Blog
8. CUDA Programming Guide --- Unified and System Memory

## Open Questions

| # | Question | Status |
|---|----------|--------|
| Q1 | Does CUDA UVM page fault behavior differ between NVLink 3.0 (RTX 3090) and PCIe-only? | OPEN |
| Q2 | What is the actual remote atomic latency over NVLink 3.0 between RTX 3090 pair? | OPEN |
| Q3 | How does NCCL detect NVLink vs PCIe and adjust its algorithms? | OPEN --- relevant for R20 |
| Q4 | What percentage of real ML workloads use direct pointer access vs explicit memcpy? | OPEN |
| Q5 | Can we intercept cuDeviceGetP2PAttribute to report NATIVE_ATOMIC_SUPPORTED=0 without breaking apps? | OPEN |
