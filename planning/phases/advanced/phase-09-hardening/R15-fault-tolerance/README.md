# R15: Fault Tolerance & Erasure Coding

**Phase:** 9 — Hardening
**Status:** RESEARCH COMPLETE, PRE-PLAN DRAFTED
**Priority:** MEDIUM
**Depends On:** R10 (Memory Tiering), R12 (Deduplication), R17 (Topology/Heartbeat), R19 (Consistency)

## Summary

Implement RAID-like redundancy across GPU memories using erasure coding. If a node crashes, its data can be reconstructed from parity stored on surviving nodes. Also includes checkpointing for training state recovery. Makes consumer GPU clusters production-reliable.

## What This Enables

- Cluster survives node failures without data loss
- Training checkpoints distributed across DRAM (no disk I/O needed for fast recovery)
- Production-grade reliability for consumer hardware
- Hot-spare node support for immediate failover
- Graceful handling of partial failures (GPU crash, NIC failure, process crash)

## Key Decisions (from Pre-Plan)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Erasure coding scheme | Hybrid RS + XOR | XOR for hot data (fast), RS for cold/important data (robust) |
| EC library | Intel ISA-L (CPU) | Saturates 100Gbps wire on single core; battle-tested |
| Parity storage | Tiered: DRAM (hot) + NVMe (cold) | VRAM is too precious; DRAM is fast and plentiful |
| Checkpoint strategy | Gemini-style in-memory + incremental deltas | 13x faster recovery than disk; 70-90% less network traffic |
| Failure detection | Multi-layer: RDMA events + phi accrual + TCP | Each layer covers different failure modes |
| Membership | Generation-based with coordinator | Simple, sufficient for 2-8 node clusters |

## Performance Targets

| Metric | Target |
|--------|--------|
| Single node failure recovery | < 30 seconds |
| Detection latency | < 1 second |
| Normal operation overhead | < 5% |
| Checkpoint overhead | < 3% of training throughput |
| Data durability | 99.999% (single node failure with RS parity) |

## Folder Contents

- `research/01-erasure-coding-algorithms.md` — RS, XOR, Fountain codes, GPU-accelerated EC comparison
- `research/02-distributed-checkpointing.md` — PyTorch DCP, DeepSpeed, Gemini, incremental checkpointing
- `research/03-failure-detection-recovery.md` — RDMA events, phi accrual, recovery workflow, partial failures
- `preplan.md` — Scope, dependencies, decisions, risks, 5-phase implementation plan (13-18 weeks)
- `progress.md` — Lifecycle tracker
- `side-docs/` — Notes, experiments

## Implementation Phases

1. **R15-A: Foundation** (3-4 weeks) — ISA-L integration, XOR + RS encoders, parity storage
2. **R15-B: Failure Detection** (2-3 weeks) — ibv_async_event, phi accrual, fencing, membership
3. **R15-C: Recovery Pipeline** (3-4 weeks) — Orchestrator, page reconstruction, hot spare activation
4. **R15-D: Checkpointing** (3-4 weeks) — Async snapshots, in-memory store, incremental deltas
5. **R15-E: Integration & Testing** (2-3 weeks) — Fault injection, end-to-end tests, benchmarks

## Related Topics

- R10 Memory Tiering (parity placement across tiers)
- R12 Memory Deduplication (dedup interacts with redundancy — shared pages have implicit copies)
- R17 Topology-Aware Scheduling (phi accrual detector, heartbeat infrastructure)
- R19 SWMR Consistency (coherency state affects parity update timing)
- R22 Live Migration (downstream — uses fault tolerance for graceful failover)
