# R19: Network Page Faults / Unified Memory --- Pre-Plan

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft
**Priority:** HIGH
**Phase:** 8 --- Smart Memory
**Depends On:** R10 (Memory Tiering), R11 (Speculative Prefetching), R12 (Memory Deduplication)

## Purpose

Define the scope, dependencies, decisions, risks, and implementation phases for R19 Network Page Faults before creating the detailed implementation plan. R19 provides transparent remote memory access: when a GPU accesses memory on a remote node, the system automatically fetches the page, installs it locally, and resumes execution. Combined with R11 prefetching (which prevents >90% of faults), R19 is the correctness safety net that ensures no memory access ever fails silently.

---

## 1. Scope Definition

### 1.1 What R19 IS

- **Pre-launch demand paging:** Before each kernel launch, check that all required pages are locally mapped. Fetch and map any missing pages via RDMA or TCP.
- **Kernel-crash recovery:** If a kernel accesses a truly unmapped page (R11 misprediction), catch the CUDA error, identify the missing page, fetch it, and re-launch the kernel.
- **Directory-based coherency protocol:** Home-node tracking of page ownership with I/S/E (Invalid/Shared/Exclusive) states. Ensures consistency across nodes without hardware coherency support.
- **Thrashing detection and mitigation:** Monitor page bounce rates per page. Escalate response: shared-read promotion, write-broadcast, page pinning.
- **Integration with R10 page table:** R19 extends R10's per-page metadata with coherency state, version numbers, and sharer tracking.
- **Integration with R11 prefetching:** Prefetch requests flow through R19's coherency protocol. Tentative reads for speculative prefetches avoid coherency overhead for mispredictions.
- **Integration with R12 deduplication:** Deduped pages are permanently Shared (read-only), never thrash, and benefit from nearest-copy fetching.
- **userfaultfd for host-staged transfers:** For the host-staged data path (Phase 1 transport), use userfaultfd to intercept CPU-side faults on pinned memory that backs remote data.

### 1.2 What R19 IS NOT

- **Not a kernel module (Phase 1):** Phase 1 operates entirely in userspace via CUDA VMM APIs (cuMemMap) and the interception layer. A kernel module is a future enhancement for true in-kernel-execution fault handling.
- **Not a replacement for NVIDIA UVM:** We do not modify or hook into nvidia-uvm.ko. We build a parallel demand paging system via the CUDA driver API.
- **Not hardware page faults:** ATS (Address Translation Services) requires NVLink to a supported CPU, which is not available on our target hardware (GeForce + PCIe). We implement software-managed paging.
- **Not a general-purpose DSM:** R19 is optimized for GPU workloads (coarse-grained, kernel-launch-aligned access patterns). It does not attempt to provide byte-level shared memory semantics.
- **Not cluster-wide locking:** R19's consistency model is SWMR (Single-Writer/Multiple-Reader), not transactional. There are no distributed locks or atomic compare-and-swap operations across nodes.

### 1.3 Boundary Conditions

| Boundary | Inside R19 | Outside R19 |
|----------|-----------|-------------|
| Fault detection | Pre-launch page checks, CUDA error catching | Application-level error handling |
| Page fetching | Coherency protocol, RDMA/TCP transfer requests | Physical transport implementation (existing) |
| Page installation | cuMemMap/cuMemUnmap, PTE updates | Physical memory allocation (R10) |
| Consistency | I/S/E protocol, directory tracking | Application-level consistency (e.g., all-reduce) |
| Thrashing | Detection, mitigation (pin/replicate/promote) | Workload repartitioning |
| Prefetch integration | Coherency-aware prefetch requests | Prediction engine (R11) |
| Dedup integration | COW fault coherency transitions | Hash computation, DDT (R12) |

---

## 2. Dependencies

### 2.1 Upstream (R19 Requires)

| Dependency | What R19 Needs | Status |
|-----------|---------------|--------|
| **R10 Memory Tiering** | Page table with 64-byte PTEs, page_id, tier tracking, allocation/free, virtual address management | Pre-planned |
| **R11 Speculative Prefetching** | Prefetch hit rate >90%, prefetch requests as coherency operations, pattern profiling | Pre-planned |
| **R12 Memory Deduplication** | DDT (dedup table), COW mechanism, is_shared flag in PTEs | Pre-planned |
| **CUDA Interception Layer** (P1-P3) | cuLaunchKernel hook (pre-launch page check), cuMemcpy* hooks (write detection), CUDA error catching | Planned |
| **Transport Layer** (P4) | Async page transfer API (RDMA READ/WRITE, TCP), bandwidth reporting | Planned |
| **R8 Kernel Param Introspection** | Parse kernel arguments to identify which pages a kernel will access | Research complete |

### 2.2 Downstream (Depends on R19)

| Dependent | What It Needs From R19 |
|-----------|----------------------|
| **R18 Virtual NVLink** | Page fault handling is part of the Virtual NVLink coherency model. R18 builds multi-GPU coherent shared memory on top of R19's I/S/E protocol. |
| **R15 Fault Tolerance** | R19's directory tracks page locations; R15 uses this for replica placement and recovery. |
| **R29 RDMA Multicast** | R19's sharer set identifies which nodes need page updates; R29 can multicast instead of point-to-point. |

### 2.3 Parallel (No Hard Dependency)

| Component | Interaction |
|-----------|------------|
| **R17 Topology-Aware Scheduling** | R19 asks R17 for nearest copy source when fetching a Shared page |
| **R14 Transport Compression** | R19's page transfers can be compressed for bandwidth savings |
| **R20 Memory Protection** | R19's access control (cuMemSetAccess) provides the foundation for memory isolation |

---

## 3. Key Decisions

### Decision 1: Fault Handling Mechanism

| Option | Description | Pros | Cons | Recommendation |
|--------|------------|------|------|---------------|
| **A: cuMemMap pre-launch** | Map pages before kernel launch via interception layer | No kernel module, works on all CUDA 10.2+ GPUs, userspace-only | Cannot catch faults during kernel execution; kernel crashes on miss | **Phase 1** |
| **B: Kernel crash + re-launch** | Catch CUDA_ERROR_ILLEGAL_ADDRESS, fetch missing page, retry | Safety net for mispredictions | Expensive (~500-5000 us per crash+restart); may lose partial kernel work | **Phase 1 fallback** |
| **C: Custom kernel module (HMM)** | Register HMM device, intercept true GPU faults | True demand paging during execution | Requires kernel module, Turing+ only (open driver), fragile across driver versions | **Phase 2** |
| **D: nvidia-uvm.ko extension** | Hook into UVM fault path | Leverages existing infrastructure | Dependent on NVIDIA driver internals, may break between versions | **Not recommended** |

**Decision:** Phase 1 uses A + B (pre-launch mapping with crash recovery fallback). Phase 2 adds C for true in-execution fault handling. D is rejected due to fragility.

### Decision 2: Consistency Model

| Option | Description | Overhead | Recommendation |
|--------|------------|----------|---------------|
| **A: SWMR with I/S/E** | 3-state protocol, home-node directory | Low (invalidation only on write) | **Selected** |
| **B: Full MESI** | 4-state protocol with Exclusive-clean distinction | Medium (track clean vs dirty) | Not worth the complexity for network scale |
| **C: Release Consistency (LRC)** | Propagate changes at synchronization points | Low (batch at sync) | Good for CPU programs, but GPU kernels are already synchronization-aligned |
| **D: Sequential Consistency** | All operations globally ordered | Very high | Completely impractical over network |

**Decision:** Option A (SWMR with I/S/E). GPU workloads naturally follow SWMR patterns (weights are shared-read, activations are exclusive). The 3-state protocol minimizes network traffic while guaranteeing correctness.

### Decision 3: Directory Architecture

| Option | Description | Pros | Cons | Recommendation |
|--------|------------|------|------|---------------|
| **A: Distributed (each home node)** | Page's home node is the directory for that page | No SPOF, local lookups for home pages, scales with nodes | Requires home tracking per page | **Selected** |
| **B: Centralized coordinator** | One node holds all directory entries | Simple, consistent | SPOF, network bottleneck for large clusters | Not recommended |
| **C: Replicated** | All nodes have full directory | Fast lookups everywhere | Sync cost, memory overhead scales with pages x nodes | Not recommended |

**Decision:** Option A (distributed directory). Each page has a home node (initially the allocating node), and that home tracks the I/S/E state and sharer set. This scales linearly with cluster size and has no single point of failure.

### Decision 4: Thrashing Response

| Strategy | When to Use | Cost | Effectiveness |
|----------|------------|------|--------------|
| **Shared-read promotion** | Read-heavy thrashing (many readers, one writer) | O(sharers) per write | High (eliminates read faults) |
| **Write-broadcast** | One writer updates data read by many | O(sharers) bandwidth per update | High for training weights |
| **Page pinning** | Unresolvable contention | Stalls one party | Last resort |
| **Adaptive granularity (sub-page)** | False sharing on 64KB pages | Metadata overhead | High for false sharing |

**Decision:** Escalating response: (1) Shared-read promotion -> (2) Write-broadcast -> (3) Page pinning. Adaptive granularity is Phase 2 only (adds significant complexity).

### Decision 5: Kernel Module Scope

| Option | Description | Recommendation |
|--------|------------|---------------|
| **A: No kernel module** | Entirely userspace, cuMemMap + interception | **Phase 1** |
| **B: Minimal module (userfaultfd helper)** | Small module for host-memory fault forwarding only | Phase 1.5 (if userfaultfd is insufficient) |
| **C: Full HMM device** | Custom kernel module with HMM registration, GPU fault interception | **Phase 2** |
| **D: OpenDMA module extension** | Add fault handling to the already-planned OpenDMA kernel module (Phase 5) | **Phase 3** (when OpenDMA is ready) |

**Decision:** Phase 1 = A (no module). Phase 2 = C if needed. Phase 3 = D (combine with OpenDMA). The OpenDMA kernel module is already planned for Phase 5 of the main project; adding fault handling to it is the most efficient path to true hardware-level demand paging.

---

## 4. Risks and Mitigations

| # | Risk | Severity | Likelihood | Mitigation |
|---|------|----------|-----------|-----------|
| 1 | **Kernel crash on unmapped access is unrecoverable** --- CUDA may not provide the faulting address, making it impossible to identify which page to fetch | Critical | Medium | Test CUDA error reporting exhaustively. Fallback: parse kernel arguments to identify ALL possible pages and pre-map them all (over-fetch). |
| 2 | **Pre-launch mapping adds too much latency** --- if a kernel needs 100+ remote pages, pre-mapping takes 2+ ms | High | Medium | Parallel RDMA fetches (ConnectX-5 supports 100+ outstanding ops). Pipeline: start fetching for kernel N+1 while kernel N is executing. R11 prefetching should reduce this to <10 pages per kernel. |
| 3 | **Coherency protocol overhead exceeds benefit** --- directory messages add latency to every page access | High | Low | Directory lookups are only needed on miss (not on hit). With R11 achieving >90% hit rate, directory traffic is <10% of page accesses. Cache directory results locally. |
| 4 | **Thrashing destroys performance for adversarial workloads** --- random write sharing across nodes | High | Low | Thrashing detection triggers within 10 ms. Mitigation escalation limits damage. Truly adversarial workloads (random cross-node writes) are fundamentally unsuitable for distributed GPU computing regardless of our approach. |
| 5 | **cuMemMap scalability** --- mapping thousands of pages may hit CUDA driver limits or cause performance degradation | Medium | Medium | Benchmark cuMemMap at scale (10K, 100K, 1M mappings). If limits exist, batch mappings or use a smaller virtual address range with LRU eviction. |
| 6 | **Interaction with CUDA memory allocator** --- cuMemMap operates outside CUDA's standard allocator, potentially confusing cudaMalloc and UVM | Medium | Medium | Keep cuMemMap ranges entirely separate from cudaMalloc ranges. Use cuMemAddressReserve for a dedicated region that the application's CUDA code never touches directly. |
| 7 | **Version number races** --- stale page data used due to concurrent write and read | High | Low | Version numbers are monotonically increasing, checked on every page access. Stale reads are detected and trigger re-fetch. Atomic version increment at the home node prevents split writes. |
| 8 | **Home node failure loses directory state** --- if a home node crashes, we lose sharer tracking for all pages homed there | Critical | Low | R15 (Fault Tolerance) provides directory replication. Before R15 is ready: persist directory state to local disk periodically. On failure: reconstruct from surviving nodes (each node knows which pages it holds). |

---

## 5. Implementation Phases

### Phase 1: Pre-Launch Demand Paging (4-5 weeks)

**Goal:** Working demand paging via cuMemMap, with kernel crash recovery as fallback.

| Sub-task | Estimate | Deliverable |
|----------|----------|-------------|
| Virtual address space manager: reserve large GPU VA range, track mapped/unmapped regions | 1 week | VA range allocator integrated with R10 page table |
| Pre-launch page checker: before cuLaunchKernel, identify required pages from kernel arguments (via R8) | 1 week | Page set analyzer that returns list of needed pages |
| Page fetch pipeline: parallel RDMA/TCP fetch for missing pages, with cuMemMap installation | 1 week | Async page fetcher with configurable parallelism |
| Kernel crash recovery: catch CUDA_ERROR_ILLEGAL_ADDRESS, identify missing pages, re-launch | 3-4 days | Error handler with re-launch logic |
| userfaultfd for host-staged path: register pinned memory regions, handle faults via remote fetch | 3-4 days | Host-side fault handler for staging buffers |
| Integration with R10 page table: add coherency fields (state, version, home, sharers) to PTEs | 3-4 days | Extended PTE structure |
| Unit tests: pre-launch mapping, crash recovery, userfaultfd fault handling | 3-4 days | Test suite with synthetic workloads |

**Exit criteria:** A synthetic workload running across 2 nodes can access remote memory transparently. Pre-launch mapping handles >95% of accesses. Crash recovery handles the remainder. No data corruption.

### Phase 2: Coherency Protocol (3-4 weeks)

**Goal:** Full I/S/E coherency with directory-based tracking and invalidation.

| Sub-task | Estimate | Deliverable |
|----------|----------|-------------|
| Directory data structure: per-page I/S/E state, sharer bitset, version tracking | 1 week | PageDirectory with O(1) lookup |
| Coherency message types: READ_REQUEST/REPLY, WRITE_REQUEST/REPLY, INVALIDATE/ACK, FETCH, EVICT | 1 week | Protocol message codec and handlers |
| Read path: I -> S transition via home node | 3-4 days | Read fault handler with directory integration |
| Write path: I/S -> E transition with invalidation | 1 week | Write fault handler with parallel invalidation |
| Eviction path: notify home on eviction, handle dirty writeback | 3-4 days | Eviction coherency integration with R10 ARC |
| R11 prefetch integration: coherency-aware prefetch requests, tentative reads | 3-4 days | Prefetch-coherency bridge |
| Stress test: multi-node concurrent read/write to same pages | 3-4 days | Correctness validation under contention |

**Exit criteria:** 4-node cluster maintains correct coherency under concurrent read/write workloads. No stale reads. No data corruption. Invalidation latency <25 us (RDMA).

### Phase 3: Thrashing Prevention and Optimization (2-3 weeks)

**Goal:** Detect and mitigate thrashing, optimize for real ML workloads.

| Sub-task | Estimate | Deliverable |
|----------|----------|-------------|
| Thrashing detector: per-page bounce counter, epoch-based threshold | 3-4 days | ThrashingDetector component |
| Shared-read promotion: detect read-heavy patterns, switch to write-broadcast | 3-4 days | Adaptive protocol switching |
| Page pinning: min-residency enforcement for thrashing pages | 2-3 days | Pinning with timeout |
| R12 dedup integration: deduped pages always Shared, COW coherency transitions | 3-4 days | Dedup-coherency bridge |
| Nearest-copy optimization: fetch Shared pages from closest node (R17 integration) | 3-4 days | Topology-aware fetch source selection |
| Performance benchmarks: PyTorch training with coherency overhead measurement | 1 week | Benchmark suite with latency/throughput metrics |

**Exit criteria:** Thrashing detected and mitigated within 10 ms. Write-broadcast reduces invalidation traffic by >50% for training workloads. Coherency overhead <5% of total training time.

### Phase 4: Advanced Fault Handling (3-4 weeks, future)

**Goal:** True in-execution fault handling via kernel module (optional).

| Sub-task | Estimate | Deliverable |
|----------|----------|-------------|
| HMM kernel module: register as HMM device, receive GPU fault notifications | 2 weeks | outerlink-faultd.ko kernel module |
| Fault-to-fetch pipeline: on GPU fault, RDMA fetch page, install via cuMemMap, resume | 1 week | Sub-millisecond fault resolution |
| Integration with OpenDMA module (Phase 5): combine fault handling with BAR1 DMA | 1 week | Unified kernel module |

**Exit criteria:** GPU faults during kernel execution are resolved without kernel restart. Fault resolution latency <100 us (RDMA). Kernel module works across kernel versions 5.15-6.x.

### Phase 5: Optimizations (2 weeks, future)

- Home node migration (move home to most-active node)
- Bulk coherency transitions for CUDA graphs
- Sub-page coherency for false sharing scenarios
- Coherency protocol compression (delta encoding for page updates)
- Multi-home for popular pages (replicated directory for read-heavy pages)

**Total estimated effort:** 9-12 weeks for Phases 1-3. Phases 4-5 are future enhancements.

---

## 6. Open Questions

### Must Answer Before Detailed Planning

1. **Does CUDA_ERROR_ILLEGAL_ADDRESS provide the faulting address?** This determines whether crash recovery can identify the specific missing page or must fall back to mapping all possible pages. Must test on actual hardware.

2. **What is the maximum cuMemAddressReserve range?** This determines the maximum cluster memory pool size. If limited to VRAM size, we need address range recycling. If limited to 48-bit VA space (~256 TB), we have plenty of room.

3. **Can cuMemMap be called concurrently from multiple threads?** R19 needs to map pages in parallel. If cuMemMap is single-threaded, it becomes a bottleneck. Must benchmark.

4. **What is the cuMemMap overhead at scale?** With 375K pages (24 GB VRAM / 64KB) and potentially 1M+ pages across the cluster, we need to verify that CUDA handles this many mappings without degradation.

5. **How does R10's ARC eviction interact with coherency?** When R10 evicts a page, R19 must update the directory. If the page is Shared, do we need to invalidate all sharers or just silently evict? (Recommended: silent eviction for Shared, explicit writeback for Exclusive.)

### Can Answer During Implementation

6. **Optimal epoch duration for thrashing detection?** Start with 10 ms, tune based on workload iteration time.

7. **Should version numbers be per-page or global (Lamport clock)?** Per-page is simpler and sufficient for our SWMR model.

8. **What is the break-even point for write-broadcast vs invalidate?** Depends on number of sharers and page write frequency. Benchmark to find the crossover.

9. **Should home node migration be triggered automatically or manually?** Start with manual (administrator configures page placement), add automatic migration based on access pattern analysis later.

10. **How does gradient checkpointing affect coherency?** Forward pass pages are re-accessed during backward pass. Coherency state should persist across kernel launches within an iteration, which it naturally does since R19 only invalidates on explicit write.

---

## 7. Success Criteria

### Quantitative

| Metric | Target (Phase 1) | Target (Phase 2) | Target (Phase 3) |
|--------|-----------------|-----------------|-----------------|
| **Page fault rate** (with R11 prefetching) | <5% of kernel launches | <2% | <1% |
| **Fault handling latency** (RDMA, single page) | <50 us | <30 us | <25 us |
| **Fault handling latency** (TCP, single page) | <200 us | <150 us | <100 us |
| **Kernel crash + re-launch rate** | <1% of kernel launches | <0.1% | <0.01% |
| **Coherency overhead** (% of total training time) | N/A (no coherency in Phase 1) | <10% | <5% |
| **Thrashing detection time** | N/A | N/A | <10 ms from onset |
| **Data corruption** | 0 incidents | 0 incidents | 0 incidents |
| **Pre-launch mapping overhead** (per kernel) | <100 us average | <50 us | <30 us |

### Qualitative

- Transparent to CUDA applications (no code changes required)
- Works with all CUDA 10.2+ GPUs (no hardware requirements beyond standard NVIDIA driver)
- Graceful degradation: if coherency overhead is too high, fall back to demand-only mode (no prefetch coherency)
- Observable: coherency state, fault rate, thrashing detection exposed via metrics API
- No kernel module required for Phase 1 functionality
- Correct under all concurrent access patterns (no stale reads, no lost writes, no data races)

---

## 8. Testing Strategy

### Unit Tests

- Page state machine: verify all I/S/E transitions with correct message generation
- Directory: verify sharer tracking, version increments, concurrent access safety
- Thrashing detector: verify bounce counting, epoch reset, threshold triggering
- Pre-launch page checker: verify page set analysis from kernel arguments
- cuMemMap integration: verify map/unmap/access-control operations

### Integration Tests

- 2-node read sharing: node A writes, node B reads via Shared state
- 2-node write contention: both nodes write, verify invalidation and correctness
- 4-node mixed workload: simulate training iteration (forward + backward + optimizer)
- Thrashing scenario: intentionally create bounce pattern, verify detection and mitigation
- Crash recovery: launch kernel with unmapped pages, verify recovery and correct re-launch
- Dedup + coherency: verify deduped pages stay Shared, COW creates Exclusive correctly

### Benchmarks

- PyTorch ResNet-50 training (2-node, 4-node)
- PyTorch GPT-2 training (pipeline parallel, 2-node)
- LLM inference (model-parallel, 4-node)
- Memory-oversubscribed workload (model > single GPU VRAM)
- Adversarial thrashing workload (random cross-node writes)

---

## Related Documents

- [R19 README](./README.md) --- overview and folder structure
- [research/01-gpu-page-fault-mechanisms.md](./research/01-gpu-page-fault-mechanisms.md) --- GPU fault mechanisms survey
- [research/02-distributed-shared-memory.md](./research/02-distributed-shared-memory.md) --- DSM prior art
- [research/03-coherency-and-thrashing.md](./research/03-coherency-and-thrashing.md) --- coherency and thrashing
- [R10 Memory Tiering](../../../phase-07-memory-intelligence/R10-memory-tiering/README.md) --- upstream dependency
- [R11 Speculative Prefetching](../R11-speculative-prefetching/preplan.md) --- upstream dependency
- [R12 Memory Deduplication](../R12-memory-deduplication/preplan.md) --- upstream dependency
- [R18 Virtual NVLink](../../R18-virtual-nvlink/README.md) --- downstream consumer
