# R22: Live Migration

**Phase:** 11 — Product Layer
**Status:** NOT STARTED
**Priority:** LOW
**Depends On:** R10 (Memory Tiering), R15 (Fault Tolerance)

## Summary
Move a running GPU workload from one node to another without stopping it. Snapshot VRAM state, stream incrementally to destination, flip the context. Like VM live migration but for GPU compute.

## What This Enables
- Zero-downtime node maintenance
- Dynamic load balancing across the cluster
- Graceful failover when a node starts showing problems
- "Drain" a node before shutdown

## Key Questions
- VRAM snapshot: full copy or dirty-page tracking (incremental)?
- Pre-copy vs post-copy migration strategy?
- How to handle in-flight CUDA operations during migration?
- Acceptable downtime window for the final switchover?

## Folder Contents
- `research/` — VM live migration techniques adapted for GPU
- `side-docs/` — Notes, experiments
- `preplan.md` — TO BE CREATED
- `plan.md` — TO BE CREATED
- `progress.md` — Lifecycle tracker

## Related Topics
- R10 Memory Tiering (migration across tiers)
- R15 Fault Tolerance (failover triggers migration)
- R19 Network Page Faults (post-copy uses page faults)
