# PRE-PLAN: Advanced Features & Distributed GPU OS

**Created:** 2026-03-24
**Last Updated:** 2026-03-24
**Status:** DRAFT
**Predecessor:** [02-FINAL-PREPLAN.md](02-FINAL-PREPLAN.md) (covers P1-P13, Phases 0-6)

## Purpose

This pre-plan maps the NEXT wave of OuterLink capabilities beyond the core implementation (P1-P13). Where the original pre-plan gets us a working GPU pooling system, THIS pre-plan turns OuterLink into a **distributed GPU operating system** — smart memory, automatic distribution, fault tolerance, and ecosystem integration.

These features build ON TOP of the Phase 0-6 foundation. Nothing here starts until P5 (PoC) is working.

---

## SECTION A: RESEARCH TOPICS (R10-R30)

### Research Already Completed

| # | Topic | Document | Key Finding |
|---|-------|----------|-------------|
| R9 | Multi-Transport: USB4, OCuLink | [R9](../research/R9-multi-transport-usb4-oculink.md) | ConnectX-5 for RDMA/OpenDMA, USB4 v2 for DRAM/control, OCuLink for raw PCIe. Multi-transport striping gives ~180Gbps aggregate. MS-02 Ultra has all three. |

### Research To Be Done

| # | Topic | Priority | Depends On | Why |
|---|-------|----------|------------|-----|
| R10 | Memory Tiering & NVMe as Tier 3 | **CRITICAL** | P5 working | Foundation for ALL memory features — defines the tier hierarchy (VRAM/DRAM/NVMe), page management, and eviction policies |
| R11 | Speculative Prefetching | HIGH | R10 | Profile-guided data movement — eliminates GPU stalls by pipelining transfers under active compute |
| R12 | Memory Deduplication | HIGH | R10 | Read-only sharing of model weights across GPUs — up to 4x memory savings for LLM inference |
| R13 | CUDA Graph Interception | HIGH | P6 working | Distributed execution planning — intercept full DAG, split across GPUs optimally |
| R14 | Transport-Layer Compression | HIGH | P6 working | LZ4/nvCOMP/delta encoding in the wire — 2-10x effective bandwidth for compressible data |
| R15 | Fault Tolerance & Erasure Coding | MEDIUM | R10 | RAID-like redundancy across GPU memories — survive node failures without data loss |
| R16 | BlueField DPU Offload | MEDIUM | P8 working | Move transport logic onto NIC's ARM core — true zero-CPU data movement |
| R17 | Topology-Aware Scheduling | HIGH | R9, P10 working | Auto-detect link speeds/hops, place data on optimal nodes, route transfers over best paths |
| R18 | Virtual NVLink Emulation | LOW | R19, R25 | The moonshot — atomic ops, cache coherency, unified address space making all GPUs look like one |
| R19 | Network Page Faults / Unified Memory | HIGH | R10 | GPU page fault triggers transparent remote fetch — apps just use pointers, no cudaMemcpy needed |
| R20 | NCCL Backend | **CRITICAL** | P6 working | Register as NCCL transport backend — instant PyTorch/TensorFlow distributed training with zero code changes |
| R21 | GPU Direct Storage Over Network | MEDIUM | P9 working (OpenDMA) | Remote NVMe -> network -> GPU VRAM, no host RAM bounce on either side |
| R22 | Live Migration | LOW | R10, R15 | Move running GPU workloads between nodes without stopping — like VM live migration for GPUs |
| R23 | Heterogeneous GPU Mixing | MEDIUM | P10 working | Mix GPU generations/sizes in same pool, schedule by capability — RTX 3060 + 4070 + 5090 together |
| R24 | Time-Sliced GPU Sharing | MEDIUM | P10, R17 | Multi-user/multi-app GPU pool with quotas — turn your LAN into a GPU cloud |
| R25 | Cooperative Kernel Splitting | LOW | R13, R26 | Split one kernel's thread blocks across physical GPUs — transparent multi-GPU compute |
| R26 | Hardware Clock Sync via PTP | HIGH | P8 working | ConnectX-5 PTP timestamping for sub-μs cross-node synchronization — enables coordinated kernel launches |
| R27 | ROCm/HIP Interception | LOW | P7 working | AMD GPU support via HIP interception — vendor-agnostic GPU pooling |
| R28 | Scatter-Gather DMA | MEDIUM | P8 working | Multi-region non-contiguous transfers in single DMA op — huge win for sparse/fragmented data |
| R29 | RDMA Multicast | MEDIUM | P10 working | One-to-many hardware broadcast — model weight distribution to N GPUs in one send, not N sends |
| R30 | Persistent Kernels with Network Feed | MEDIUM | R13 | Long-running GPU kernels fed via VRAM doorbell from network — zero launch overhead between batches |

---

## SECTION B: DEPENDENCY GRAPH

```
FOUNDATION (P1-P13 from original pre-plan)
    |
    v
LAYER 1: MEMORY FOUNDATION
├── R10: Memory Tiering          <- EVERYTHING builds on this
├── R14: Transport Compression   <- Benefits all transfers immediately
└── R20: NCCL Backend            <- Biggest ecosystem impact, relatively standalone
    |
    v
LAYER 2: SMART MEMORY
├── R12: Memory Deduplication    <- Needs R10 (tier-aware dedup)
├── R19: Network Page Faults     <- Needs R10 (page management)
├── R11: Speculative Prefetching <- Needs R10 (knows which tier to prefetch from)
└── R17: Topology-Aware Sched.   <- Needs R9 (multi-transport) + P10 (multi-node)
    |
    v
LAYER 3: PERFORMANCE & RELIABILITY
├── R26: PTP Clock Sync          <- Needs working multi-node (P8+)
├── R28: Scatter-Gather DMA      <- Needs transport layer (P8+)
├── R15: Fault Tolerance         <- Needs R10 (knows what's stored where)
├── R29: RDMA Multicast          <- Needs P10 (multi-node)
└── R21: GPU Direct Storage      <- Needs P9 (OpenDMA BAR1 path)
    |
    v
LAYER 4: COMPUTE DISTRIBUTION
├── R13: CUDA Graph Interception <- Needs P6 (full CUDA interception)
├── R30: Persistent Kernels      <- Needs R13 (kernel lifecycle control)
├── R23: Heterogeneous GPU Mix   <- Needs R17 (topology-aware scheduling)
└── R16: BlueField DPU Offload   <- Needs P8 (mature transport to offload)
    |
    v
LAYER 5: PRODUCT FEATURES
├── R24: Time-Sliced Sharing     <- Needs R17 + R15 (scheduling + isolation)
├── R22: Live Migration          <- Needs R10 + R15 (memory tracking + fault tolerance)
└── R27: ROCm/HIP Interception   <- Needs P7 (proven CUDA pattern to replicate)
    |
    v
LAYER 6: MOONSHOT
├── R25: Cooperative Kernel Split <- Needs R13 + R26 (graph analysis + clock sync)
└── R18: Virtual NVLink          <- Needs R19 + R25 (page faults + kernel splitting)
```

---

## SECTION C: PROPOSED PHASE MAPPING

These map to future phases AFTER the original P1-P13:

### Phase 7: Memory Intelligence (R10, R14, R20)

**Goal:** Transform OuterLink from "remote GPU access" to "intelligent distributed memory system" + instant ML framework compatibility.

| Component | Research | What Gets Built |
|-----------|----------|-----------------|
| Memory tier manager | R10 | 5-tier hierarchy (VRAM → pinned → DRAM → remote → NVMe), page tables, eviction policies, LRU/LFU tracking |
| Wire compression | R14 | Pluggable compressor in transport layer (LZ4 for CPU, nvCOMP for GPU, delta for iterative workloads) |
| NCCL transport plugin | R20 | `libnccl-net-outerlink.so` — AllReduce, AllGather, Broadcast over OuterLink transport |

**Milestone:** PyTorch distributed training runs across 2 nodes using OuterLink as NCCL backend, with compressed transfers and automatic memory tiering.

**Why first:** R10 is the foundation for 80% of later features. R20 gives us the ML ecosystem instantly. R14 is a multiplier on everything.

---

### Phase 8: Smart Memory (R12, R19, R11, R17)

**Goal:** Memory system that thinks — deduplicates, prefetches, routes optimally, and handles page faults transparently.

| Component | Research | What Gets Built |
|-----------|----------|-----------------|
| Dedup engine | R12 | Content-hash tracking, copy-on-write for shared weights, reference counting |
| Network page fault handler | R19 | GPU fault handler → remote fetch → page install → GPU resume. Zero cudaMemcpy needed. |
| Prefetch engine | R11 | Access pattern profiler, prediction model, background prefetch pipeline |
| Topology router | R17 | Link discovery, latency/bandwidth map, per-transfer routing decisions, multi-path striping |

**Milestone:** LLM inference with 70B model across 4 GPUs uses 140GB (not 560GB) via dedup, zero application-level memory management, transfers routed over optimal links.

**Why second:** These are the features that make OuterLink feel magic. App just runs, OuterLink handles everything.

---

### Phase 9: Hardening (R26, R28, R15, R29, R21)

**Goal:** Production-grade reliability and performance — clock sync, fault tolerance, efficient DMA patterns, multicast, direct storage.

| Component | Research | What Gets Built |
|-----------|----------|-----------------|
| PTP sync daemon | R26 | ConnectX hardware PTP, cross-node clock alignment, coordinated launch API |
| Scatter-gather DMA | R28 | Multi-region DMA descriptor builder, sparse transfer optimization |
| Erasure coding engine | R15 | Reed-Solomon across VRAM pools, parity management, recovery protocol |
| RDMA multicast | R29 | One-to-many broadcast for model weight distribution, IGMP group management |
| Remote GDS | R21 | NVMe → wire → GPU path bypassing both hosts' RAM |

**Milestone:** Cluster survives node failure without data loss, broadcasts model to 8 GPUs in one operation, PTP-synchronized kernel launches across nodes.

---

### Phase 10: Compute Distribution (R13, R30, R23, R16)

**Goal:** Not just shared memory — shared COMPUTE. OuterLink decides where and how kernels run.

| Component | Research | What Gets Built |
|-----------|----------|-----------------|
| CUDA graph analyzer | R13 | Graph capture interception, dependency analysis, optimal split planning |
| Persistent kernel runtime | R30 | Doorbell-based kernel feed, network → VRAM → kernel pipeline |
| Heterogeneous scheduler | R23 | GPU capability database, workload profiling, best-fit placement |
| DPU offload engine | R16 | BlueField ARM runtime, on-NIC transport logic, zero-host-CPU transfers |

**Milestone:** CUDA graph submitted to OuterLink is automatically split across heterogeneous GPUs, persistent kernels process streaming data with zero launch overhead.

---

### Phase 11: Product Layer (R24, R22, R27)

**Goal:** From developer tool to product — multi-tenant, live migration, AMD support.

| Component | Research | What Gets Built |
|-----------|----------|-----------------|
| GPU time-slicer | R24 | Quota system, fair scheduling, isolation, multi-user auth |
| Live migration engine | R22 | VRAM snapshot, incremental state transfer, context handoff |
| ROCm/HIP interceptor | R27 | HIP driver API interception, AMD ↔ NVIDIA unified pool |

**Milestone:** Multiple users share GPU pool with quotas, workloads migrate between nodes without downtime, AMD and NVIDIA GPUs in same cluster.

---

### Phase 12: The Moonshot (R25, R18)

**Goal:** Virtual NVLink — all GPUs behave as one.

| Component | Research | What Gets Built |
|-----------|----------|-----------------|
| Cooperative kernel splitter | R25 | Thread block distribution, cross-GPU shared memory, block synchronization |
| Virtual NVLink protocol | R18 | Atomic operations over network, cache coherency protocol, unified virtual address space |

**Milestone:** A single CUDA kernel launch transparently executes across 8 GPUs on 4 machines as if they were one NVLink-connected GPU. The holy grail.

---

## SECTION D: DECISIONS TO MAKE

| # | Decision | Options | When | Impact |
|---|----------|---------|------|--------|
| D18 | Memory tier eviction policy | LRU / LFU / ML-predicted / hybrid | R10 research | Determines page management performance |
| D19 | Compression strategy | Always-on / adaptive / per-tier / per-data-type | R14 research | CPU overhead vs bandwidth savings tradeoff |
| D20 | NCCL version target | NCCL 2.18+ / 2.21+ | R20 research | API compatibility surface |
| D21 | Page fault mechanism | userfaultfd / custom kernel handler / CUDA UVM hook | R19 research | Complexity vs transparency tradeoff |
| D22 | Dedup granularity | Page-level (4KB) / chunk-level (64KB) / tensor-level | R12 research | Memory overhead of hash tracking |
| D23 | Erasure coding scheme | Reed-Solomon / fountain codes / simple replication | R15 research | Recovery speed vs storage overhead |
| D24 | BlueField programming model | DOCA SDK / raw DPDK / custom ARM runtime | R16 research | Development complexity |
| D25 | ROCm interception strategy | HIP driver API / ROCr / thunk layer | R27 research | Which AMD layer to intercept |
| D26 | Multi-tenant isolation | Process-level / container-level / VM-level | R24 research | Security vs overhead |
| D27 | Virtual NVLink coherency | Directory-based / snooping / hybrid | R18 research | Scalability vs latency |

---

## SECTION E: RISKS

| # | Risk | Impact | Probability | Mitigation |
|---|------|--------|-------------|-----------|
| K11 | Memory tiering adds latency to every allocation | HIGH | MEDIUM | Fast-path for local VRAM, tiering only for overflow |
| K12 | Network page faults too slow for real workloads | HIGH | MEDIUM | Prefetching (R11) hides latency, fallback to explicit transfers |
| K13 | NCCL backend API changes between versions | MEDIUM | MEDIUM | Version-gate, support 2-3 NCCL versions |
| K14 | Dedup hash computation overhead exceeds savings | MEDIUM | LOW | Only dedup read-only regions, lazy hashing |
| K15 | Cooperative kernel splitting breaks CUDA semantics | HIGH | HIGH | Very conservative splitting — only independent blocks, verify correctness exhaustively |
| K16 | BlueField ARM cores too slow for transport logic | MEDIUM | MEDIUM | Hybrid: ARM handles routing, host CPU handles heavy transforms |
| K17 | PTP sync jitter too high for coordinated launches | MEDIUM | LOW | ConnectX hardware PTP is sub-μs, should be fine |
| K18 | ROCm/HIP interception is fundamentally different from CUDA | MEDIUM | MEDIUM | May need separate interceptor architecture, not just port |
| K19 | Live migration state too large for fast handoff | MEDIUM | HIGH | Incremental migration (dirty page tracking), pre-copy approach |
| K20 | Virtual NVLink coherency protocol doesn't scale past 4 GPUs | HIGH | HIGH | This is research — may not be fully achievable. Partial coherency still valuable. |

---

## SECTION F: HARDWARE ADDITIONS

The advanced features benefit from additional hardware:

| # | Hardware | For | Priority | Est. Cost |
|---|----------|-----|----------|-----------|
| H14 | BlueField-2 DPU (1-2 cards) | R16 research | LOW | $100-200 (eBay) |
| H15 | Additional NVMe drives (high-capacity) | R10 tier 4/5, R21 | MEDIUM | $100-300 |
| H16 | USB4 v2 cable | R9 multi-transport testing | HIGH | $20-30 |
| H17 | OCuLink adapter + cable | R9 OCuLink path testing | MEDIUM | $30-50 |
| H18 | PTP-capable switch (or direct DAC) | R26 clock sync | LOW | Already have DAC |
| H19 | AMD GPU (RX 7600 or similar) | R27 ROCm testing | LOW | $200-300 |
| H20 | MS-02 Ultra or similar multi-transport node | Full integration testing | FUTURE | $1500-2900 |
| H21 | PCIe risers / extensions | Physical fitment in MS-02 chassis | HIGH | $20-50 |

---

## SECTION G: THE BIG PICTURE

```
WHAT EXISTS (Original Pre-Plan, P1-P13):
    "GPUs across PCs work as a pool"
    - CUDA interception
    - TCP/RDMA transport
    - Host-staged + OpenDMA
    - Multi-node pooling
    |
    v
WHAT THIS PRE-PLAN ADDS (P14-P19):
    "A distributed GPU operating system"

    Phase 7:  Memory Intelligence    -> Smart tiering + NCCL = usable for ML TODAY
    Phase 8:  Smart Memory           -> Dedup + prefetch + page faults = feels like magic
    Phase 9:  Hardening              -> Fault tolerance + clock sync = production-grade
    Phase 10: Compute Distribution   -> Graph splitting + heterogeneous = true GPU OS
    Phase 11: Product Layer          -> Multi-tenant + live migration + AMD = product
    Phase 12: Moonshot               -> Virtual NVLink = the holy grail
```

### What This Means

When all of this is done, OuterLink is not a GPU sharing tool. It's a **distributed GPU operating system** that:

1. **Any CUDA app runs unmodified** across any number of GPUs on any number of machines
2. **Any ML framework works instantly** via NCCL backend (PyTorch, TensorFlow, JAX)
3. **Memory is infinite** — VRAM + DRAM + NVMe across all nodes, managed automatically
4. **Compute scales linearly** — add a GPU anywhere on the network, cluster gets faster
5. **Hardware agnostic** — NVIDIA + AMD, GeForce + Datacenter, ConnectX + USB4 + OCuLink
6. **Production reliable** — fault tolerance, live migration, multi-tenant isolation
7. **Consumer accessible** — runs on $200 GeForce + $50 ConnectX, not $10k datacenter hardware

Nobody has built this. The closest things (NVIDIA's proprietary stack, VMware vGPU, rCUDA) each do ONE piece and charge enterprise prices. OuterLink does ALL of it, open source, on consumer hardware.

---

## SECTION H: NEXT STEPS

This pre-plan is complete for the advanced features scope. The path forward:

```
NOW:        Complete P1-P13 (original pre-plan) — get the foundation working
            |
THEN:       Write R10 research (Memory Tiering) — it's the keystone
            Write R20 research (NCCL Backend) — it's the biggest ecosystem win
            Write R14 research (Compression) — it's a universal multiplier
            |
THEN:       Plan Phase 7 (Memory Intelligence) — first advanced implementation phase
            |
THEN:       Execute Phase 7, repeat for Phase 8-12
```

**The original pre-plan (P1-P13) is still THE priority.** Nothing here starts until we have a working PoC. But knowing WHERE we're going shapes HOW we build the foundation — every architecture decision in P4-P6 should consider these future layers.

---

## Related Documents

- [02-FINAL-PREPLAN.md](02-FINAL-PREPLAN.md) — Original pre-plan (P1-P13, Phases 0-6)
- [03-contingency-plans.md](03-contingency-plans.md) — Plan B/C/D for core features
- [R9: Multi-Transport](../research/R9-multi-transport-usb4-oculink.md) — USB4/OCuLink/ConnectX research
- [Project Vision](../../docs/architecture/00-project-vision.md)
- [Research Consolidation](../research/CONSOLIDATION-all-research.md)

## Open Questions

| # | Question | Status | Resolution |
|---|----------|--------|------------|
| Q1 | Should Phase 7 (Memory Intelligence) start before P13 (docs) is fully complete? | OPEN | Probably yes — docs can be ongoing |
| Q2 | Is R20 (NCCL Backend) valuable enough to fast-track into Phase 7 even though it's ecosystem, not memory? | OPEN | Strong yes — it's how 90% of ML users would interact with OuterLink |
| Q3 | Should we prototype R19 (network page faults) early to validate feasibility before building R10 fully? | OPEN | Risk reduction — userfaultfd prototype could prove/disprove the concept cheaply |
| Q4 | At what phase does OuterLink need its own website / documentation portal? | OPEN | Probably Phase 7-8, when external users start appearing |
| Q5 | Should the original pre-plan (02) be updated to reference this document? | OPEN | Yes — add a "What comes next" pointer |
