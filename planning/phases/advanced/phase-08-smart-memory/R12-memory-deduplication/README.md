# R12: Memory Deduplication

**Phase:** 8 — Smart Memory
**Status:** NOT STARTED
**Priority:** HIGH
**Depends On:** R10 (Memory Tiering)

## Summary

Detect identical memory regions across GPUs/nodes and keep a single copy instead of duplicating. Primary target: LLM model weights, which are read-only and identical across all GPUs during inference. Could reduce memory usage by 4x+ for multi-GPU inference.

## What This Enables

- 70B model (140GB) across 4 GPUs: 140GB instead of 560GB
- Copy-on-write for shared regions that occasionally get modified
- More GPUs can participate without hitting memory limits
- Combines with memory tiering for maximum efficiency

## Key Questions

- Dedup granularity: page-level (4KB), chunk-level (64KB), or tensor-level?
- Hash algorithm for content comparison? (xxHash, CRC32, SHA-256?)
- Hash tracking overhead vs memory savings tradeoff?
- How to handle copy-on-write across network?

## Folder Contents

- `research/` — Dedup strategies, overhead analysis
- `side-docs/` — Notes, experiments
- `preplan.md` — TO BE CREATED
- `plan.md` — TO BE CREATED
- `progress.md` — Lifecycle tracker

## Related Topics

- R10 Memory Tiering (tier-aware dedup decisions)
- R29 RDMA Multicast (broadcast shared data efficiently)
- R15 Fault Tolerance (dedup interacts with redundancy)
