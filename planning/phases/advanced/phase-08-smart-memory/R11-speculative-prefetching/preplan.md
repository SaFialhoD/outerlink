# R11: Speculative Prefetching — Pre-Plan

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft
**Priority:** HIGH
**Depends On:** R10 (Memory Tiering)

## Purpose

Define the scope, dependencies, decisions, risks, and implementation phases for R11 Speculative Prefetching before creating the detailed plan. This document answers "WHAT needs to be planned" so the detailed plan can answer "HOW to build it."

---

## 1. Scope Definition

### 1.1 What R11 IS

- **Access pattern profiler:** instrument the CUDA interception layer to record which pages each kernel accesses, in what order, and whether reads or writes
- **Pattern classifier:** detect sequential, strided, iteration-repeating, and phase-alternating access patterns from profiling data
- **Prediction engine:** given classified patterns, predict which pages will be needed by upcoming kernels
- **Prefetch scheduler:** issue asynchronous data transfers from remote/slow tiers to local VRAM before the GPU needs the data
- **Bandwidth manager:** budget network bandwidth between demand fetches, prefetches, and write-backs to avoid starvation
- **Cancellation system:** abort wrong-prediction prefetches to reclaim bandwidth and buffer space
- **R10 integration:** coordinate prefetches with the ARC eviction policy and page table metadata

### 1.2 What R11 IS NOT

- **Not an ML-based predictor** (in Phase 1): start with deterministic pattern matching (stride, sequence replay). ML-based prediction (transformer model on page sequences) is a future enhancement.
- **Not a replacement for R10:** R11 builds on top of R10's page table, tier system, and eviction policy. R11 tells R10 "move this page here soon" — R10 handles the actual tier mechanics.
- **Not a general-purpose caching system:** R11 prefetches speculatively based on predictions. Non-predictable workloads fall through to R10's demand-based migration.
- **Not cluster-wide scheduling:** R11 operates per-node. Each node prefetches for its own GPU(s). Cross-node coordination (e.g., multicast prefetch) is a future enhancement.
- **Not transport-specific:** R11 generates prefetch requests; the transport layer (TCP, RDMA, OpenDMA) handles actual data movement. R11 works with whatever transport is available.

### 1.3 Boundary Conditions

| Boundary | Inside R11 | Outside R11 |
|---|---|---|
| Pattern detection | Profiling, classification | Modifying application code |
| Data movement | Scheduling, prioritization | Actual transport implementation (existing) |
| VRAM management | Prefetch promotion requests | Eviction policy (R10) |
| Bandwidth | Budgeting, throttling | Physical NIC configuration |
| Multi-node | Per-node prefetch scheduling | Global cluster-wide optimization (future) |

---

## 2. Dependencies

### 2.1 Upstream (R11 requires)

| Dependency | What R11 Needs From It | Status |
|---|---|---|
| **R10 Memory Tiering** | Page table with access metadata (access_count, timestamp, tier, dirty_bit), ARC eviction API, tier migration primitives | Pre-planned |
| **CUDA Interception Layer** (P1-P3) | LD_PRELOAD hooks for cuLaunchKernel, cuMemcpy*, cuMemAlloc, cuStreamSync | Planned |
| **Transport Layer** (P4) | Async transfer API (send page from tier X to tier Y), bandwidth reporting | Planned |
| **R8 Kernel Param Introspection** | Ability to parse kernel arguments for known signatures (cuBLAS, cuDNN) | Research complete |

### 2.2 Downstream (depends on R11)

| Dependent | What It Needs From R11 |
|---|---|
| **R17 Topology-Aware Scheduling** | Prefetch requests include source/destination — R17 picks the optimal network path |
| **R14 Transport Compression** | Can compress prefetch transfers (especially weights — compressible) |
| **R19 Network Page Faults** | R11 reduces page fault rate, making R19's fault handling less performance-critical |
| **R12 Memory Deduplication** | Deduplicated pages only need one prefetch for multiple consumers |

### 2.3 Soft Dependencies

| Dependency | Impact If Unavailable |
|---|---|
| **CUDA Graphs support** | Cannot do ahead-of-time scheduling for graph workloads — fall back to online prediction |
| **Hardware access counters (Volta+)** | Cannot use zero-overhead page-level access tracking — use page table scan instead |
| **CUPTI** | Cannot do detailed warmup profiling — rely solely on interception hooks |

---

## 3. Key Decisions That Need To Be Made

### 3.1 Prediction Model Complexity

| Option | Complexity | Accuracy | Overhead | Recommendation |
|---|---|---|---|---|
| **A: Simple stride/sequential** | Low | Good for regular patterns | <50 us/iteration | Phase 1 — start here |
| **B: Sequence replay (iteration-aware)** | Medium | Excellent for ML training | <200 us/iteration | Phase 1 — implement alongside A |
| **C: Phase-aware (forward/backward/optimizer)** | Medium | Excellent for multi-phase workloads | <500 us/iteration | Phase 2 |
| **D: ML-based (transformer on page sequences)** | High | Best for irregular patterns | 1-5 ms/iteration | Phase 3 (future) |

**Proposed decision:** Start with A+B in Phase 1. These cover 90%+ of ML training workloads with minimal complexity.

### 3.2 Prefetch Buffer Architecture

| Option | Description | Trade-off |
|---|---|---|
| **A: Pinned DRAM ring buffer only** | All prefetches stage through pinned host RAM | Extra copy to VRAM; large buffer capacity |
| **B: VRAM reserved region only** | Prefetch directly into VRAM staging area | No extra copy; reduces usable VRAM |
| **C: Hybrid (DRAM ring + small VRAM window)** | DRAM for bulk staging, VRAM for imminent data | Best of both; more complex |

**Proposed decision:** Option C (hybrid). DRAM ring buffer for most data (2 GB default), small VRAM window (256-512 MB) for data needed within next 1-2 kernels.

### 3.3 Bandwidth Split

| Option | Demand : Prefetch : Writeback | Notes |
|---|---|---|
| **A: Conservative** | 80 : 10 : 10 | Safe but may under-prefetch |
| **B: Balanced** | 70 : 20 : 10 | Good default for predictable workloads |
| **C: Aggressive** | 50 : 40 : 10 | Maximizes prefetch but risks demand starvation |
| **D: Adaptive** | Dynamic based on hit rate | Best performance, most complex |

**Proposed decision:** Start with B (70:20:10) as default, implement D (adaptive) in Phase 2.

### 3.4 Prefetch Granularity

| Option | Description | Trade-off |
|---|---|---|
| **A: Full pages (64KB)** | Always prefetch complete 64KB pages | Simple; may over-fetch for partial access |
| **B: Sub-page (4KB blocks within 64KB page)** | Prefetch only accessed blocks | Less bandwidth waste; complex metadata |

**Proposed decision:** Option A (full pages). At 100Gbps, a 64KB page transfers in ~5.8 us. The overhead of sub-page tracking outweighs the bandwidth savings at this speed. Revisit for slower transports (USB4).

### 3.5 Scheduler Centralization

| Option | Description | Trade-off |
|---|---|---|
| **A: Per-node** | Each node schedules its own prefetches independently | Simple; no coordination overhead; may cause contention |
| **B: Cluster-aware** | Central scheduler or distributed consensus on prefetch priorities | Optimal global bandwidth use; adds latency and complexity |

**Proposed decision:** Option A (per-node) for Phase 1. Cluster awareness in Phase 3 if contention is observed.

---

## 4. Risks and Mitigations

| # | Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|---|
| 1 | **Prediction overhead exceeds benefit** — profiling/prediction cost more than the stalls they prevent | High | Low | Zero-overhead profiling via interception hooks (already on call path). Budget <2 ms/iteration. |
| 2 | **Bandwidth starvation** — prefetches crowd out demand fetches, increasing stalls | High | Medium | Token bucket rate limiter with demand fetch priority. Prefetch is always preemptible. |
| 3 | **VRAM pressure** — prefetch staging area reduces usable VRAM below application needs | Medium | Medium | Configurable staging size. Dynamic shrinking when VRAM pressure detected. Fall back to DRAM-only staging. |
| 4 | **Misprediction cascade** — wrong prediction triggers wrong evictions, causing thrashing | High | Low | Prefetched pages enter ARC's T1 (recency) not T2 (frequency). Immunity window is short. Misprediction triggers re-profiling. |
| 5 | **Interaction with CUDA memory allocator** — prefetched pages conflict with cudaMalloc'd regions | Medium | Low | Prefetch operates on R10-managed pages, which are already outside CUDA's allocator view. |
| 6 | **Multi-GPU contention** — multiple GPUs prefetching from same remote source | Medium | Medium | Per-source rate limiting. Source node tracks total outbound prefetch bandwidth. |
| 7 | **Unpredictable workloads** — graph neural networks, sparse ops, dynamic shapes | Medium | Medium | Graceful degradation: low-confidence predictions → no prefetch → demand-only (same as no R11). |
| 8 | **RDMA transfer cannot be canceled** — NIC DMA in progress wastes bandwidth | Low | High | Accept this — only cancel queued transfers. Keep prefetch queue shallow (2-4 in-flight max). |

---

## 5. Implementation Phases

### Phase 1: Foundation (3-4 weeks)

**Goal:** Working prefetch for simple sequential/iteration-repeat patterns.

| Sub-task | Estimate | Deliverable |
|---|---|---|
| Interception profiler: log kernel launches + memcpy + sync events | 1 week | Structured event log per iteration |
| Page access tracker: correlate events with R10 page table | 3-4 days | Per-kernel page access sets |
| Sequential/stride detector | 3-4 days | Pattern classifier (sequential, strided, unknown) |
| Iteration replay detector | 3-4 days | Detect repeating kernel sequences across iterations |
| Prefetch queue + priority system | 1 week | BinaryHeap-based scheduler with priority levels |
| Pinned DRAM ring buffer | 3-4 days | Staging buffer for prefetched pages |
| Basic bandwidth budgeting (fixed split) | 2-3 days | Token bucket rate limiter |
| Integration test: prefetch reduces stalls on synthetic sequential workload | 3-4 days | Benchmark showing stall reduction |

**Exit criteria:** Prefetching works for a synthetic workload with sequential/strided access. Stall rate reduced by >50% compared to demand-only.

### Phase 2: ML Workload Optimization (2-3 weeks)

**Goal:** Effective prefetching for real transformer/CNN training.

| Sub-task | Estimate | Deliverable |
|---|---|---|
| Kernel argument parser for cuBLAS/cuDNN signatures | 1 week | Extract tensor addresses/sizes from known kernels |
| Phase detection (forward/backward/optimizer) | 3-4 days | Classify iteration sub-phases |
| Adaptive prefetch distance | 3-4 days | Dynamic distance based on kernel timing |
| VRAM prefetch window (hybrid buffer) | 3-4 days | Small VRAM staging area for imminent data |
| Cancellation system | 3-4 days | Cancel queued prefetches on misprediction |
| Adaptive bandwidth budgeting | 2-3 days | Adjust split based on hit rate |
| Integration test: prefetch on real PyTorch transformer training | 1 week | Benchmark on GPT-2 or similar |

**Exit criteria:** Prefetching works for PyTorch transformer training across 2+ nodes. Stall rate <10%. Bandwidth waste <20%.

### Phase 3: Advanced Features (2-3 weeks)

**Goal:** Handle edge cases and optimize.

| Sub-task | Estimate | Deliverable |
|---|---|---|
| CUDA Graph support (ahead-of-time scheduling) | 1 week | Parse graph DAG, pre-compute prefetch schedule |
| Multi-tier chain prefetching (NVMe → DRAM → VRAM pipeline) | 1 week | Pipelined tier traversal |
| R10 ARC integration (immunity, T1 insertion) | 3-4 days | Coordinated prefetch/eviction |
| Prefetch metrics + monitoring dashboard | 2-3 days | Hit rate, coverage, timeliness, waste |
| Performance tuning + profiling | 3-4 days | Minimize overhead, maximize coverage |

**Exit criteria:** Multi-tier prefetching works. ARC integration prevents thrashing. Monitoring shows prefetch effectiveness in real-time.

### Phase 4: Future Enhancements (not in initial plan)

- ML-based prediction model (transformer on page sequences)
- Cluster-wide prefetch coordination
- Multicast prefetch for data-parallel weight distribution
- Compression-aware prefetching (integrate with R14)
- OpenDMA direct-to-VRAM prefetch path

**Total estimated effort: 7-10 weeks** for Phases 1-3.

---

## 6. Open Questions

### Must Answer Before Detailed Planning

1. **What is the actual overhead of R10 page table scanning at 375K entries?** Need to benchmark. If >5 ms, we need a more selective scanning approach (dirty-page bitmap).

2. **Can we parse kernel arguments reliably for the top 20 cuBLAS/cuDNN kernels?** This determines whether we get tensor-level visibility or fall back to page-level-only tracking. R8 research should answer this.

3. **What is the RDMA queue pair depth on ConnectX-5 for concurrent transfers?** This determines max_in_flight for the prefetch queue. Too many concurrent posts may cause NIC queuing delays.

4. **Does pinned DRAM allocation compete with CUDA's own pinned memory pool?** If so, the staging buffer size must be coordinated with the application's pinned memory usage.

### Can Answer During Implementation

5. How does gradient checkpointing affect pattern predictability? (Likely: forward kernels execute twice, which the replay detector will handle.)

6. What is the optimal VRAM prefetch window size for different model sizes? (Empirical — tune during Phase 2.)

7. Should prefetch requests be batched (send 10 page requests at once) or individual? (Benchmark both.)

8. How does PCIe contention between NIC and GPU affect prefetch-to-VRAM promotion latency? (Measure on actual hardware.)

---

## 7. Success Criteria

### Quantitative

| Metric | Target (Phase 1) | Target (Phase 2) | Target (Phase 3) |
|---|---|---|---|
| **GPU stall rate** (kernels waiting for data) | <20% (from ~50%+ baseline) | <10% | <5% |
| **Prefetch hit rate** | >70% | >85% | >90% |
| **Prefetch timeliness** (data ready before kernel) | >60% | >80% | >90% |
| **Bandwidth waste** (prefetched but unused) | <30% | <20% | <15% |
| **Profiling overhead** | <5 ms/iteration | <3 ms/iteration | <2 ms/iteration |
| **Effective throughput improvement** (vs demand-only) | >30% | >50% | >70% |
| **Prefetch scheduling latency** (prediction to transfer start) | <200 us | <100 us | <50 us |

### Qualitative

- Prefetching is transparent to CUDA applications (no code changes required)
- Graceful degradation for unpredictable workloads (never worse than demand-only)
- Configurable aggressiveness (conservative/balanced/aggressive profiles)
- Observable (metrics exposed for monitoring)
- Stable under memory pressure (no thrashing caused by prefetch)

---

## 8. Testing Strategy

### Unit Tests

- Pattern classifier: given synthetic page access sequences, verify correct classification
- Prefetch queue: verify priority ordering, cancellation, deadline management
- Bandwidth budget: verify token refill, stealing, throttling
- Ring buffer: verify wrap-around, overwrite protection, slot management

### Integration Tests

- Synthetic sequential workload: verify stall reduction
- Synthetic strided workload: verify stride detection and prefetch
- Iteration replay: verify pattern learned from iteration 1, applied to iteration 2
- Misprediction: verify cancellation and re-profiling
- Buffer pressure: verify graceful degradation under full buffer

### Benchmarks

- PyTorch ResNet-50 training (CNN baseline)
- PyTorch GPT-2 training (transformer baseline)
- Inference workload with KV cache growth
- Multi-node data-parallel training
- Memory-oversubscribed workload (model > single GPU VRAM)

---

## Related Documents

- [R11 README](./README.md) — summary and folder contents
- [research/01-existing-prefetching-systems.md](./research/01-existing-prefetching-systems.md) — survey of existing approaches
- [research/02-access-pattern-profiling.md](./research/02-access-pattern-profiling.md) — profiling techniques
- [research/03-prefetch-scheduling.md](./research/03-prefetch-scheduling.md) — scheduling strategies
- [R10 Memory Tiering](../../../phase-07-memory-intelligence/R10-memory-tiering/README.md) — upstream dependency
- [R8 Kernel Param Introspection](../../../../research/R8-kernel-param-introspection.md) — kernel argument parsing
