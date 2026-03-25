# R29: RDMA Multicast

**Phase:** 9 — Hardening
**Status:** NOT STARTED
**Priority:** MEDIUM
**Depends On:** P10 (Multi-Node working)

## Summary

Use ConnectX's RDMA multicast to broadcast data to multiple nodes in a single network operation. The NIC hardware replicates packets — one send reaches N receivers. Primary use case: distributing model weights to all GPUs simultaneously instead of N separate transfers.

## What This Enables

- One-to-many broadcast at line rate (not N sequential unicasts)
- Model weight distribution to 8 GPUs in one send
- Efficient AllBroadcast for NCCL backend
- Linear scaling for broadcast-heavy workloads

## Key Questions

- ConnectX multicast group management (IGMP)?
- Does RDMA multicast work with OpenDMA (BAR1 targets)?
- Reliability: how to handle dropped multicast packets?
- Maximum multicast group size?

## Folder Contents

- `research/` — RDMA multicast, IB multicast groups, ConnectX support
- `side-docs/` — Notes, experiments
- `preplan.md` — TO BE CREATED
- `plan.md` — TO BE CREATED
- `progress.md` — Lifecycle tracker

## Related Topics

- R20 NCCL Backend (multicast for Broadcast collective)
- R12 Memory Deduplication (multicast distributes shared weights)
- R17 Topology-Aware Scheduling (multicast routing)
