# R29: RDMA Multicast

**Phase:** 9 — Hardening
**Status:** RESEARCH COMPLETE
**Priority:** MEDIUM
**Depends On:** P10 (Multi-Node working)

## Summary

Use ConnectX's RDMA multicast to broadcast data to multiple nodes in a single network operation. The NIC hardware replicates packets — one send reaches N receivers. Primary use case: distributing model weights to all GPUs simultaneously instead of N separate transfers.

## What This Enables

- One-to-many broadcast at line rate (not N sequential unicasts)
- Model weight distribution to 8 GPUs in one send
- Efficient AllBroadcast for NCCL backend
- Linear scaling for broadcast-heavy workloads

## Key Findings (Research Phase)

- **ConnectX-5 supports 2M multicast groups, 240 QPs per group** — far exceeds OuterLink's needs
- **RoCE v2 multicast is unreliable** — UD QPs only, no RDMA WRITE, packets can be lost
- **NACK-based reliability is sufficient** at 2-8 node scale — no FEC needed initially
- **7x speedup for 8-node broadcast** (1 send vs 7 sequential unicasts at sender)
- **Tree broadcast is the reliable fallback** — RC RDMA WRITE, O(log N) latency, zero loss
- **NCCL Broadcast is the primary integration point** — multicast for messages > 4MB
- **OpenDMA + multicast is Phase 2** — UD RECV can target BAR1 but GRH handling is complex
- **Multicast doesn't help AllReduce** — tree/ring algorithms are already optimized

## Key Questions (Answered)

- ConnectX multicast group management (IGMP)? **IGMP snooping on RoCE v2, SM-managed on IB. Switch must support IGMP snooping.**
- Does RDMA multicast work with OpenDMA (BAR1 targets)? **Possible but complex — UD RECV to BAR1 requires GRH offset handling. Phase 2.**
- Reliability: how to handle dropped multicast packets? **NACK-based: receivers report missing sequences, sender retransmits via unicast RC RDMA WRITE.**
- Maximum multicast group size? **240 QPs per group on ConnectX-5 — supports clusters up to 240 nodes.**

## Folder Contents

- `research/01-rdma-multicast-fundamentals.md` — IB/RoCE multicast, ConnectX-5 limits, UD QP constraints
- `research/02-reliable-multicast.md` — NACK, FEC, PGM, SHARP, SRD analysis
- `research/03-multicast-for-ml.md` — ML use cases, bandwidth savings, when multicast wins/loses
- `side-docs/` — Notes, experiments
- `preplan.md` — Scope, dependencies, 5-phase implementation roadmap
- `plan.md` — TO BE CREATED
- `progress.md` — Lifecycle tracker

## Related Topics

- R20 NCCL Backend (multicast for Broadcast collective)
- R12 Memory Deduplication (multicast distributes shared weights)
- R17 Topology-Aware Scheduling (multicast routing, tree topology)
- R28 Scatter-Gather DMA (scatter on multicast receive)
