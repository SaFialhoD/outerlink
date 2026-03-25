# R17: Topology-Aware Scheduling

**Phase:** 8 — Smart Memory
**Status:** NOT STARTED
**Priority:** HIGH
**Depends On:** R9 (Multi-Transport), P10 (Multi-Node working)

## Summary

Auto-discover cluster topology (link types, speeds, latencies, hop counts) and make intelligent routing decisions. Place data on optimal nodes based on access patterns. Route transfers over the best available link. Stripe across multiple links for maximum throughput.

## What This Enables

- Automatic best-path selection per transfer
- Multi-path striping (ConnectX + USB4 = ~180Gbps)
- Hot data migrates to closest node automatically
- Topology changes (node added/removed) handled gracefully

## Key Questions

- How to discover topology? (probe latencies, query NIC capabilities?)
- Routing algorithm: static tables, dynamic per-transfer, or learned?
- How to handle asymmetric links (USB4 = 80Gbps, ConnectX = 100Gbps)?
- Integration with NCCL's own topology detection?

## Folder Contents

- `research/` — Network topology discovery, routing algorithms
- `side-docs/` — Notes, experiments
- `preplan.md` — TO BE CREATED
- `plan.md` — TO BE CREATED
- `progress.md` — Lifecycle tracker

## Related Topics

- R9 Multi-Transport (the transports we're routing across)
- R20 NCCL Backend (NCCL has its own topology, must align)
- R23 Heterogeneous GPU Mixing (topology includes GPU capabilities)
