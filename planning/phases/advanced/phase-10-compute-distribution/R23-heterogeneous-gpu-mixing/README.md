# R23: Heterogeneous GPU Mixing

**Phase:** 10 — Compute Distribution
**Status:** NOT STARTED
**Priority:** MEDIUM
**Depends On:** P10 (Multi-Node working)

## Summary
Support mixing different GPU generations, architectures, and VRAM sizes in the same pool. RTX 3060 + RTX 4070 + RTX 5090 all contribute. OuterLink schedules work based on each GPU's compute capability, memory size, and bandwidth.

## What This Enables
- No need for identical hardware across nodes
- Upgrade incrementally — add new GPUs without replacing old ones
- Memory-heavy work on big-VRAM cards, compute-heavy on fast-shader cards
- Maximum utilization of all available hardware

## Key Questions
- How to normalize performance across GPU generations?
- CUDA compute capability differences — which kernels run where?
- Memory speed differences affect scheduling?
- Driver version compatibility across different GPUs?

## Folder Contents
- `research/` — GPU capability databases, heterogeneous scheduling
- `side-docs/` — Notes, experiments
- `preplan.md` — TO BE CREATED
- `plan.md` — TO BE CREATED
- `progress.md` — Lifecycle tracker

## Related Topics
- R17 Topology-Aware Scheduling (GPU capabilities in topology map)
- R24 Time-Sliced Sharing (heterogeneous quotas)
- R13 CUDA Graph Interception (place graph nodes by GPU capability)
