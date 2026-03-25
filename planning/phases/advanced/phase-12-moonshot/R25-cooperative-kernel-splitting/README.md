# R25: Cooperative Kernel Splitting

**Phase:** 12 — Moonshot
**Status:** NOT STARTED
**Priority:** LOW (highest complexity)
**Depends On:** R13 (CUDA Graph Interception), R26 (PTP Clock Sync)

## Summary
Split a single CUDA kernel launch across multiple physical GPUs. Blocks 0-127 on GPU A, blocks 128-255 on GPU B. The app launches one kernel and gets Nx the compute. The ultimate expression of distributed GPU pooling.

## What This Enables
- Transparent multi-GPU compute for single-kernel workloads
- Linear compute scaling without application changes
- Combined with Virtual NVLink (R18): true single-GPU illusion
- Any CUDA app scales across the cluster automatically

## Key Questions
- How to partition thread blocks across GPUs?
- Shared memory — cross-GPU shared memory accesses?
- Block synchronization (__syncthreads) across GPUs?
- Data dependencies between blocks: static or runtime detection?

## Folder Contents
- `research/` — CUDA execution model, thread block scheduling, cooperative groups
- `side-docs/` — Notes, experiments
- `preplan.md` — TO BE CREATED
- `plan.md` — TO BE CREATED
- `progress.md` — Lifecycle tracker

## Related Topics
- R13 CUDA Graph Interception (graph-level splitting)
- R18 Virtual NVLink (coherency for shared data)
- R26 PTP Clock Sync (synchronized block launches)
