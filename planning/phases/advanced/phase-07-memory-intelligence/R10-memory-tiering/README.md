# R10: Memory Tiering & NVMe as Tier 3

**Phase:** 7 — Memory Intelligence
**Status:** NOT STARTED
**Priority:** CRITICAL
**Depends On:** P5 (PoC working)

## Summary

Define and implement a 5-tier memory hierarchy across the OuterLink cluster: Local VRAM → Pinned RAM → Local DRAM → Remote DRAM/VRAM → NVMe. Includes page tables, eviction policies, and automatic tier migration. This is the FOUNDATION for almost every other advanced feature.

## What This Enables

- Unified memory pool up to 16TB+ across cluster
- Automatic overflow from VRAM to DRAM to NVMe
- Transparent to CUDA applications
- Foundation for: dedup (R12), page faults (R19), prefetching (R11), fault tolerance (R15)

## Key Questions

- What page size? (4KB, 64KB, 2MB huge pages?)
- Eviction policy? (LRU, LFU, ML-predicted?)
- How to track page locations across nodes?
- Integration with CUDA's own memory management?

## Folder Contents

- `research/` — Research findings on memory tiering approaches
- `side-docs/` — Notes, experiments, diagrams
- `preplan.md` — Pre-plan (scope, dependencies, decisions) — TO BE CREATED
- `plan.md` — Detailed implementation plan — TO BE CREATED
- `progress.md` — Lifecycle progress tracker

## Related Topics

- R12 Memory Deduplication (builds on this)
- R19 Network Page Faults (builds on this)
- R11 Speculative Prefetching (builds on this)
- R15 Fault Tolerance (builds on this)
