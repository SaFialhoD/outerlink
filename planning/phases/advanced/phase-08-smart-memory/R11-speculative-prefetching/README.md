# R11: Speculative Prefetching

**Phase:** 8 — Smart Memory
**Status:** NOT STARTED
**Priority:** HIGH
**Depends On:** R10 (Memory Tiering)

## Summary

Profile GPU memory access patterns and speculatively prefetch data from remote tiers before the GPU requests it. Eliminates GPU stalls by pipelining data transfers under active compute. ML workloads are highly predictable (iterate same tensors), making prefetching extremely effective.

## What This Enables

- Zero-stall GPU compute — data arrives before it's needed
- Hides network latency entirely for predictable workloads
- Works across all memory tiers (remote VRAM, DRAM, NVMe)
- Transforms high-latency remote memory into effectively local memory

## Key Questions

- How to profile access patterns without overhead?
- Prediction model: simple sequential, stride-based, or ML-based?
- How far ahead to prefetch? (too early wastes bandwidth, too late causes stalls)
- Where to buffer prefetched data? (local VRAM, pinned RAM?)

## Folder Contents

- `research/` — Research on prefetching strategies, academic papers
- `side-docs/` — Notes, experiments
- `preplan.md` — TO BE CREATED
- `plan.md` — TO BE CREATED
- `progress.md` — Lifecycle tracker

## Related Topics

- R10 Memory Tiering (which tier to prefetch from)
- R17 Topology-Aware Scheduling (which path to prefetch over)
- R14 Transport Compression (compress prefetched data?)
