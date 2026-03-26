# R11: Speculative Prefetching

**Phase:** 8 — Smart Memory
**Status:** PRE-PLAN COMPLETE
**Priority:** HIGH
**Depends On:** R10 (Memory Tiering)

## Summary

Profile GPU memory access patterns and speculatively prefetch data from remote tiers before the GPU requests it. Eliminates GPU stalls by pipelining data transfers under active compute. ML workloads are highly predictable (iterate same tensors), making prefetching extremely effective.

## What This Enables

- Zero-stall GPU compute — data arrives before it's needed
- Hides network latency entirely for predictable workloads
- Works across all memory tiers (remote VRAM, DRAM, NVMe)
- Transforms high-latency remote memory into effectively local memory

## Key Findings from Research

- **Explicit prefetch beats fault-driven migration by 2-10x** across all systems surveyed (NVIDIA UVM, AMD HMM, CXL far memory)
- **ML workloads are extremely predictable:** training iterations repeat identical access patterns; after profiling 1 iteration, we can prefetch perfectly for all subsequent iterations
- **Our interception layer is the key advantage:** we see every CUDA call before it happens, giving us zero-overhead profiling data that UVM/HMM systems cannot access
- **Bandwidth budgeting is critical:** CXL research shows demand fetches must always have priority over speculative prefetches (70:20:10 split recommended)
- **At 100Gbps RDMA, a 64KB page transfers in ~5.8 us** — prefetch distance of 1-3 kernels ahead is sufficient for most workloads

## Key Decisions (from Pre-Plan)

| Decision | Recommendation |
|---|---|
| Prediction model | Start with stride detection + iteration replay (covers 90%+ ML workloads) |
| Buffer architecture | Hybrid: 2 GB pinned DRAM ring buffer + 256-512 MB VRAM window |
| Bandwidth split | 70% demand / 20% prefetch / 10% writeback (adaptive later) |
| Prefetch granularity | Full 64KB pages (sub-page not worth it at 100Gbps) |
| Scheduler scope | Per-node initially; cluster-aware in future phase |

## Implementation Estimate

- **Phase 1 (Foundation):** 3-4 weeks — sequential/stride/replay detection + basic scheduler
- **Phase 2 (ML Optimization):** 2-3 weeks — kernel arg parsing, phase detection, adaptive distance
- **Phase 3 (Advanced):** 2-3 weeks — CUDA Graph support, multi-tier chains, ARC integration
- **Total:** 7-10 weeks

## Success Criteria

| Metric | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| GPU stall rate | <20% | <10% | <5% |
| Prefetch hit rate | >70% | >85% | >90% |
| Bandwidth waste | <30% | <20% | <15% |
| Throughput improvement (vs demand-only) | >30% | >50% | >70% |

## Folder Contents

- `research/01-existing-prefetching-systems.md` — Survey of CPU, GPU, storage, and ML prefetching
- `research/02-access-pattern-profiling.md` — How to detect access patterns via CUDA interception
- `research/03-prefetch-scheduling.md` — When, where, and how to schedule prefetches
- `preplan.md` — Scope, dependencies, decisions, risks, phases, success criteria
- `plan.md` — TO BE CREATED (detailed implementation plan)
- `progress.md` — Lifecycle tracker

## Related Topics

- R10 Memory Tiering (which tier to prefetch from)
- R17 Topology-Aware Scheduling (which path to prefetch over)
- R14 Transport Compression (compress prefetched data?)
- R12 Memory Deduplication (deduplicated pages need only one prefetch)
- R19 Network Page Faults (R11 reduces fault rate, making R19 less critical-path)
