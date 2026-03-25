# R30: Persistent Kernels with Network Feed

**Phase:** 10 — Compute Distribution
**Status:** RESEARCH COMPLETE
**Priority:** MEDIUM
**Depends On:** R13 (CUDA Graph Interception)

## Summary
Long-running GPU kernels that don't exit between data batches. They spin-wait on a VRAM "doorbell". When data arrives via RDMA/DMA, the doorbell rings and the kernel immediately processes new data. Zero kernel launch overhead between batches.

## What This Enables
- Zero launch overhead for streaming workloads
- VRAM doorbell = network to GPU pipeline with no CPU
- Ideal for inference serving (continuous batch processing)
- Combined with OpenDMA: network writes directly to doorbell region

## Key Questions
- GPU thread occupancy with persistent kernels?
- How to handle kernel errors/timeouts?
- Doorbell mechanism: memory-mapped flag, atomic counter, or ring buffer?
- Power consumption of spin-waiting GPU threads?

## Folder Contents
- `research/01-persistent-kernel-patterns.md` — CUDA cooperative groups, occupancy, TDR, power/thermal, error handling
- `research/02-doorbell-mechanisms.md` — VRAM ring buffers, atomic counters, OpenDMA integration, cache coherency
- `research/03-network-fed-execution.md` — Pipeline design, buffering strategies, performance analysis, reference architectures
- `side-docs/` — Notes, experiments
- `preplan.md` — Pre-plan with scope, unknowns, dependencies, 3-phase implementation proposal
- `plan.md` — TO BE CREATED
- `progress.md` — Lifecycle tracker

## Related Topics
- R13 CUDA Graph Interception (persistent kernels as graph nodes)
- R26 PTP Clock Sync (coordinated doorbell timing)
- R28 Scatter-Gather DMA (feed data to kernel via scatter-gather)
