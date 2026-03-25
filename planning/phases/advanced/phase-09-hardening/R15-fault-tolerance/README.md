# R15: Fault Tolerance & Erasure Coding

**Phase:** 9 — Hardening
**Status:** NOT STARTED
**Priority:** MEDIUM
**Depends On:** R10 (Memory Tiering)

## Summary

Implement RAID-like redundancy across GPU memories using erasure coding. If a node crashes, its data can be reconstructed from parity stored on surviving nodes. Also includes checkpointing for training state recovery.

## What This Enables

- Cluster survives node failures without data loss
- Training checkpoints distributed across VRAM (no disk I/O needed)
- Production-grade reliability for consumer hardware
- Hot-spare node support

## Key Questions

- Erasure coding scheme: Reed-Solomon, fountain codes, or simple replication?
- Parity overhead vs recovery speed tradeoff?
- Where to store parity: VRAM, DRAM, or NVMe?
- Detection latency: how fast do we notice a node is gone?

## Folder Contents

- `research/` — Erasure coding algorithms, distributed recovery
- `side-docs/` — Notes, experiments
- `preplan.md` — TO BE CREATED
- `plan.md` — TO BE CREATED
- `progress.md` — Lifecycle tracker

## Related Topics

- R10 Memory Tiering (parity placement across tiers)
- R22 Live Migration (failover triggers migration)
- R12 Memory Deduplication (dedup interacts with redundancy)
