# R17: Topology-Aware Scheduling

**Phase:** 8 — Smart Memory
**Status:** PRE-PLANNED
**Priority:** HIGH
**Depends On:** R9 (Multi-Transport), R10 (Memory Tiering), P10 (Multi-Node working)

## Summary

Auto-discover cluster topology (link types, speeds, latencies, hop counts) and make intelligent routing decisions. Place data on optimal nodes based on access patterns. Route transfers over the best available link. Stripe across multiple links for maximum throughput.

## What This Enables

- Automatic best-path selection per transfer (latency-optimized for small, bandwidth-optimized for large)
- Multi-path striping (ConnectX-5 + USB4 = ~180Gbps aggregate)
- Hot data migrates to closest node automatically via cost-benefit model
- Topology changes (node added/removed, link failure) handled gracefully with <5s failover
- NCCL sees OuterLink's full topology for optimal collective algorithm selection

## Key Decisions Made (from research)

| # | Decision | Answer | Rationale |
|---|---|---|---|
| D1 | Discovery library | hwloc (primary) + libibverbs + sysfs | Unified API, portable, covers GPUs/NICs/NUMA |
| D2 | Topology graph | Custom HashMap-based (not petgraph) | Small graph (2-8 nodes), simpler than generic library |
| D3 | Routing table | Static, recomputed on topology change | <1 us recomputation for 8 nodes; congestion overlay in Phase 2 |
| D4 | Striping granularity | Chunk-based (not per-page) | Lower overhead, simpler reassembly |
| D5 | Failure detection | RDMA events + fixed-timeout heartbeat | Fast hardware detection + backup heartbeat |
| D6 | Placement integration | Passive advisor in Phase 1, active in Phase 2 | Start simple, prove the scoring model works |

## Key Questions (Answered)

- **How to discover topology?** hwloc for intra-node, libibverbs for RDMA, sysfs for USB4, active probing for ground-truth latency/bandwidth
- **Routing algorithm?** Static weighted Dijkstra, size-based path selection (latency for small, bandwidth for large)
- **How to handle asymmetric links?** Priority classes map transfer types to preferred link types
- **Integration with NCCL?** Generate NCCL-compatible topology XML for R20

## Implementation Estimate

| Phase | Scope | Duration |
|---|---|---|
| Phase 1 | Discovery + static routing + single-path selection | 3-4 weeks |
| Phase 2 | Multi-path striping + congestion-aware + NCCL XML | 3-4 weeks |
| Phase 3 | Active placement + cost-benefit migration + phase-aware | 2-3 weeks |
| **Total** | | **8-11 weeks** |

## Folder Contents

- `research/01-topology-discovery.md` — RDMA enumeration, PCIe sysfs, USB4, hwloc, active probing, NCCL reference
- `research/02-routing-algorithms.md` — Path selection, multi-path striping, congestion-aware routing, MPTCP/MP-RDMA references
- `research/03-data-placement.md` — Affinity placement, replication vs migration, cost models, ML-specific patterns
- `preplan.md` — Scope, dependencies, decisions, risks, implementation phases, success criteria
- `progress.md` — Lifecycle tracker
- `side-docs/` — Notes, experiments

## Related Topics

- R9 Multi-Transport (the transports we're routing across)
- R10 Memory Tiering (tier placement decisions depend on topology)
- R11 Speculative Prefetching (bandwidth budgeting depends on available links)
- R20 NCCL Backend (NCCL topology alignment via XML)
- R23 Heterogeneous GPU Mixing (topology includes GPU capabilities)
- R14 Transport Compression (compression value depends on link speed)
