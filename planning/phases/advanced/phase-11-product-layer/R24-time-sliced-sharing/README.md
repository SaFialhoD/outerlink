# R24: Time-Sliced GPU Sharing

**Phase:** 11 — Product Layer
**Status:** NOT STARTED
**Priority:** MEDIUM
**Depends On:** P10 (Multi-Node), R17 (Topology-Aware Scheduling)

## Summary
Multiple users or applications share the GPU pool with time-slicing and quotas. Turn your LAN into a GPU cloud. Includes authentication, fair scheduling, and resource isolation.

## What This Enables
- Multi-tenant GPU cloud on a LAN
- Fair sharing between users/applications
- Priority levels (production > development > experiment)
- Usage accounting and quota enforcement

## Key Questions
- Isolation mechanism: process-level, container-level, or VM-level?
- Time-slice granularity: per-kernel, per-millisecond, per-batch?
- How to prevent one user from exhausting all VRAM?
- Authentication and authorization model?

## Folder Contents
- `research/` — GPU virtualization, MPS, MIG, time-slicing
- `side-docs/` — Notes, experiments
- `preplan.md` — TO BE CREATED
- `plan.md` — TO BE CREATED
- `progress.md` — Lifecycle tracker

## Related Topics
- R17 Topology-Aware Scheduling (scheduling across users)
- R15 Fault Tolerance (isolation between users)
- R22 Live Migration (rebalance user workloads)
