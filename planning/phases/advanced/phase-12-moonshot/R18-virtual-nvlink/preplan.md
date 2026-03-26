# R18: Virtual NVLink Emulation --- Pre-Plan

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** DRAFT
**Purpose:** Define scope, unknowns, dependencies, and implementation approach for Virtual NVLink before writing the detailed plan.

---

## 1. What We Are Planning

Virtual NVLink is a software layer that makes network-distributed GPUs appear NVLink-connected to CUDA applications. It emulates NVLink's peer access API, unified address space, P2P transfers, atomic operations, and cache coherency --- all over standard ethernet/RDMA networking.

This is NOT about matching NVLink bandwidth or latency. It is about matching NVLink SEMANTICS so that applications designed for multi-GPU NVLink systems run unmodified on OuterLink clusters.

---

## 2. Scope Definition

### In Scope

| Component | Description | Research Doc |
|-----------|-------------|-------------|
| Peer Access API Emulation | Intercept cuDeviceCanAccessPeer, cuCtxEnablePeerAccess, cuDeviceGetP2PAttribute | 01-nvlink-protocol.md |
| P2P Memory Transfer | Intercept cudaMemcpyPeer and route through transport layer | 01-nvlink-protocol.md |
| Unified Virtual Address Space | Map all GPU memories into single address space with page fault handling | 01-nvlink-protocol.md, R19 |
| Remote Atomic Operations | CAS and fetch-add via RDMA; remaining atomics via CAS emulation | 02-rdma-atomics-and-coherency.md |
| Atomic Proxy Server | Per-node service for executing remote atomics on local memory | 02-rdma-atomics-and-coherency.md |
| Page-Level Coherency | I/S/E protocol extended for atomic-aware coherency | 02-rdma-atomics-and-coherency.md |
| Memory Ordering / Fencing | RDMA fence-based ordering matching NVLink guarantees | 02-rdma-atomics-and-coherency.md |
| Performance Diagnostics | Detect and report performance-killing patterns | 03-feasibility-and-limitations.md |

### Out of Scope (Acknowledged Limitations)

| Item | Why |
|------|-----|
| NVLink bandwidth matching | Physics --- network is 10-300x slower |
| Cache-line-level coherency (128B) | Would need sub-page tracking, impractical at network latencies |
| Hardware-speed atomics | Network RTT is minimum ~1-2us |
| NVSwitch emulation | All-to-all at full bandwidth requires hardware |
| Cooperative groups across GPUs | Depends on R25 (Cooperative Kernel Splitting), separate effort |

---

## 3. Dependency Map

### Hard Dependencies (must exist before R18 work begins)

| Dependency | What R18 Needs From It | Status |
|-----------|----------------------|--------|
| Phase 3: CUDA Interception | cuDeviceCanAccessPeer, cuCtxEnablePeerAccess interception points | NOT STARTED |
| Phase 4: Transport Layer | Reliable data transfer between nodes | NOT STARTED |
| R19: Network Page Faults | Page fault handler, I/S/E coherency protocol, address mapping | NOT STARTED |
| R10: Memory Tiering | Page management, eviction policies, page table structures | NOT STARTED |

### Soft Dependencies (improve R18 but not required)

| Dependency | What It Adds | Status |
|-----------|-------------|--------|
| R11: Speculative Prefetching | Hides page fault latency for predictable access patterns | NOT STARTED |
| R14: Transport Compression | 2-10x effective bandwidth for compressible transfers | NOT STARTED |
| R17: Topology-Aware Scheduling | Optimal routing for multi-hop transfers | NOT STARTED |
| R20: NCCL Backend | ML framework integration (covers 80% of use cases without R18) | NOT STARTED |
| R25: Cooperative Kernel Splitting | Cross-GPU kernel execution | NOT STARTED |
| R26: PTP Clock Sync | Synchronized cross-GPU operations | NOT STARTED |
| R29: RDMA Multicast | Efficient multicast invalidation for coherency | NOT STARTED |

### Dependency Insight

R18 sits at the top of the dependency tree. Almost everything feeds into it. This is expected for a moonshot feature. However, **Tier 1 features (peer access API + memcpy interception) depend only on Phase 3 + Phase 4**, which are core OuterLink functionality. These can be implemented much earlier than the full R18.

---

## 4. Implementation Tiers

### Tier 1: Peer Access + P2P Transfer (THE 80%)

**Goal:** All applications that use `cudaMemcpyPeer` and check `cuDeviceCanAccessPeer` work transparently.

**What gets built:**
1. Interceptors for all peer access API functions
2. `cudaMemcpyPeer` routing through transport layer
3. P2P attribute reporting (performance rank, access supported)

**Dependencies:** Phase 3 (CUDA interception) + Phase 4 (transport)

**Could be built:** During Phase 6, as an extension to CUDA interception

**Covers:** ~80% of NVLink-using applications (NCCL collectives, PyTorch/TensorFlow distributed training, bulk transfers)

### Tier 2: Unified Address Space + Page Faults

**Goal:** GPU pointers to remote memory are valid and dereferenceable. Pages migrate on demand.

**What gets built:**
1. Virtual address mapping spanning all GPUs
2. Page fault handler integration (R19)
3. Page migration with I/S/E coherency
4. Prefetch integration (R11)

**Dependencies:** R19, R10, R11

**Could be built:** During Phase 8 (Smart Memory), as extension of R19

**Covers:** Applications that use `cudaMallocManaged` across GPUs or directly dereference remote pointers

### Tier 3: Remote Atomics + Full Coherency

**Goal:** Atomic operations on remote memory work correctly. Full coherency protocol.

**What gets built:**
1. Atomic proxy server (per-node)
2. RDMA atomic mapping (CAS, fetch-add)
3. Software CAS emulation for remaining atomics
4. Atomic-aware coherency protocol extensions
5. Memory ordering / fencing layer

**Dependencies:** R19, Tier 2, RDMA transport

**Could be built:** Phase 12 (Moonshot), as the capstone feature

**Covers:** CUDA applications with cross-GPU synchronization, distributed data structures, cooperative groups

---

## 5. Key Technical Decisions Needed

| # | Decision | Options | Impact | When |
|---|----------|---------|--------|------|
| D27 | Coherency protocol | Directory-based (I/S/E from R19) / Snooping / Hybrid | Scalability vs latency | Before Tier 2 implementation |
| D28 | Page size for coherency | 4KB (less false sharing, more overhead) / 64KB (R10 default) / Adaptive | Performance vs memory overhead | Before Tier 2 implementation |
| D29 | Atomic proxy vs page migration | Proxy (lower latency for hot atomics) / Migration (simpler) / Hybrid | Atomic operation latency | Before Tier 3 implementation |
| D30 | P2P attribute reporting | Full NVLink emulation / Conservative (atomics=0) / Configurable | App compatibility vs performance honesty | Before Tier 1 implementation |
| D31 | Home node selection | Fixed (allocation origin) / Dynamic (most accessors) / Hashed | Coherency efficiency | Before Tier 2 implementation |
| D32 | Maximum coherent nodes | 4 / 8 / 16 / Unlimited | Protocol complexity vs cluster size | Before Tier 2 implementation |

---

## 6. Unknowns and Research Gaps

### High-Risk Unknowns

| Unknown | Risk | How to Resolve |
|---------|------|---------------|
| R19 page fault feasibility | If page faults are too slow or unreliable, Tier 2 and 3 are blocked | Prototype userfaultfd + GPU page fault early |
| False sharing severity | 64KB pages may cause constant thrashing for mixed-access workloads | Benchmark with real workloads, try 4KB pages |
| RDMA atomic latency on our hardware | Theory says ~2-5us, actual may differ | Benchmark ConnectX-5 CAS and fetch-add |
| GPU page fault mechanism choice | userfaultfd vs CUDA UVM hooks vs custom kernel handler | R19 research must determine this |

### Medium-Risk Unknowns

| Unknown | Risk | How to Resolve |
|---------|------|---------------|
| Application response to `NATIVE_ATOMIC_SUPPORTED=0` | Apps may refuse P2P or fall back to slow paths | Test with PyTorch, NCCL, common CUDA apps |
| Coherency protocol correctness | Subtle bugs lead to data corruption | Formal specification, model checking (TLA+ or similar) |
| Scalability past 4 nodes | Protocol message overhead may dominate | Start with 2 nodes, measure overhead as nodes increase |
| Interaction with CUDA UVM | OuterLink's page management vs CUDA's own UVM | Determine: cooperate or replace |

### Low-Risk Unknowns

| Unknown | Risk | How to Resolve |
|---------|------|---------------|
| Peer access API interception | Straightforward extension of Phase 3 interception | Implement and test |
| cudaMemcpyPeer routing | Already in OuterLink's core design | Implement and test |

---

## 7. Testing Strategy

### Correctness Testing

| Test Category | What It Verifies |
|--------------|-----------------|
| P2P memcpy correctness | Data arrives intact via network transport |
| Peer access API semantics | All API return values match real NVLink behavior |
| Atomic correctness | CAS, fetch-add, and emulated atomics produce correct results under contention |
| Coherency correctness | I/S/E state transitions maintain data consistency |
| Memory ordering | Fences ensure visibility guarantees |
| Address space integrity | No aliasing, no stale pointers, no corruption |

### Performance Benchmarking

| Benchmark | What It Measures | Target |
|-----------|-----------------|--------|
| P2P bandwidth | Throughput of cudaMemcpyPeer | Within 2x of raw transport bandwidth |
| P2P latency | Round-trip for small transfers | <10us for RDMA, <100us for TCP |
| Page fault latency | Time from fault to page available | <50us for RDMA |
| Atomic latency | Round-trip for CAS/fetch-add | <10us for RDMA direct, <50us for emulated |
| Coherency transition latency | Time for S->E transition (invalidation) | <20us for 2-node, <100us for 4-node |
| Thrashing rate | Pages/second migrating back and forth | Should stabilize under steady-state access |

### Application Testing

| Application | Expected Result |
|------------|----------------|
| NVIDIA P2P sample (simpleP2P) | Works, correct output |
| PyTorch distributed training (DDP) | Trains correctly, 2-10x slower than NVLink |
| LLM inference (tensor parallel) | Correct output, acceptable latency |
| NCCL all_reduce_perf | Works, reports network bandwidth |
| Custom atomic test (counter across GPUs) | Correct final count |

---

## 8. Milestone Breakdown

### M1: Peer Access API (Tier 1a)

**Deliverables:**
- All peer access API functions intercepted and emulated
- `simpleP2P` CUDA sample passes
- No dependency on R19/R10

### M2: P2P Transfer Optimization (Tier 1b)

**Deliverables:**
- `cudaMemcpyPeer` routed through transport with pipelining
- Bandwidth within 2x of raw transport bandwidth
- Async transfers (`cudaMemcpyPeerAsync`) supported

### M3: Unified Address Space (Tier 2a)

**Deliverables:**
- Virtual address mapping spans all GPUs
- Page fault handler fetches remote pages on access
- Basic I/S/E coherency for read/write patterns
- `cudaMallocManaged` works across network-distributed GPUs

### M4: Prefetch-Optimized Access (Tier 2b)

**Deliverables:**
- Access pattern profiler triggers prefetching
- Streaming workloads run within 2-5x of local VRAM speed
- Page thrashing detected and mitigated

### M5: Remote Atomics (Tier 3a)

**Deliverables:**
- RDMA CAS and fetch-add for 64-bit atomics
- CAS emulation for 32-bit and non-standard atomics
- Atomic proxy server for hot pages
- Atomic correctness tests pass under contention

### M6: Full Coherency + Ordering (Tier 3b)

**Deliverables:**
- Atomic-aware I/S/E coherency protocol
- Memory ordering via RDMA fences
- Performance diagnostics for anti-patterns
- LLM training and inference benchmarks pass

---

## 9. Estimated Effort

| Milestone | Complexity | Estimated Time | Can Start After |
|-----------|-----------|----------------|----------------|
| M1 | LOW | 1-2 weeks | Phase 3 complete |
| M2 | LOW-MEDIUM | 1-2 weeks | Phase 4 complete |
| M3 | HIGH | 3-4 weeks | R19 + R10 complete |
| M4 | MEDIUM | 2-3 weeks | M3 + R11 complete |
| M5 | HIGH | 3-4 weeks | M3 + RDMA transport |
| M6 | VERY HIGH | 4-6 weeks | M5 + formal verification |

**Total: ~14-21 weeks** for all milestones, but M1-M2 can start much earlier (Phase 6 timeframe) while M3-M6 wait for the Smart Memory layer.

---

## 10. Recommendation

**Start Tier 1 early.** The peer access API interception and P2P transfer routing are natural extensions of the core CUDA interception (Phase 3) and transport layer (Phase 4). They have no dependency on R19 or R10 and deliver immediate value.

**Do not attempt Tier 3 until Tier 2 is proven.** Software coherency and remote atomics are the hardest parts and most likely to reveal fundamental limitations. Validate R19 (page faults) and R10 (memory management) with real workloads before investing in atomic emulation.

**Prototype the page fault mechanism (R19) as early as possible.** R19 is the keystone for Tier 2 and 3. If page faults prove impractical, the entire R18 strategy must be reconsidered (fall back to explicit-transfer-only model).

---

## Related Documents

- [research/01-nvlink-protocol.md](research/01-nvlink-protocol.md) --- NVLink protocol deep dive
- [research/02-rdma-atomics-and-coherency.md](research/02-rdma-atomics-and-coherency.md) --- RDMA atomic and coherency analysis
- [research/03-feasibility-and-limitations.md](research/03-feasibility-and-limitations.md) --- Honest feasibility assessment
- [R19: Network Page Faults](../phase-08-smart-memory/R19-network-page-faults/README.md) --- Foundation dependency
- [R10: Memory Tiering](../phase-07-memory-intelligence/R10-memory-tiering/README.md) --- Page management foundation
- [04-advanced-features-preplan.md](../../../pre-planning/04-advanced-features-preplan.md) --- Master advanced features plan

## Open Questions

| # | Question | Status |
|---|----------|--------|
| Q1 | Should M1 (peer access API) be pulled into Phase 6 instead of waiting for Phase 12? | OPEN --- strong case for early implementation |
| Q2 | Is formal verification (TLA+) worth the effort for the coherency protocol? | OPEN --- high value but high effort |
| Q3 | Should R18 define its own page size or inherit R10's 64KB? | OPEN --- may need smaller pages for coherency |
| Q4 | Can we use NVSHMEM concepts for the atomic proxy design? | OPEN --- NVSHMEM fence/quiet semantics are relevant |
| Q5 | What is the minimum viable Virtual NVLink for an alpha release? | OPEN --- likely Tier 1 only |
