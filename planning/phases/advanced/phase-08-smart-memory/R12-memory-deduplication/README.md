# R12: Memory Deduplication

**Phase:** 8 -- Smart Memory
**Status:** PRE-PLANNED
**Priority:** HIGH
**Depends On:** R10 (Memory Tiering)

## Summary

Detect identical memory regions across GPUs/nodes and keep a single copy instead of duplicating. Primary target: LLM model weights, which are read-only and identical across all GPUs during inference. Could reduce memory usage by 4x+ for multi-GPU inference.

## What This Enables

- 70B model (140GB) across 4 GPUs: 140GB instead of 560GB
- Copy-on-write for shared regions that occasionally get modified
- More GPUs can participate without hitting memory limits
- Combines with memory tiering for maximum efficiency

## Key Decisions (Preliminary)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Hash algorithm | xxHash128 | 31 GB/s throughput, 128-bit output, collision prob ~10^-27 at 1M pages |
| Dedup granularity | 64KB pages | Aligns with R10 page table, minimal overhead |
| Phase 1 scope | Read-only dedup only | Zero runtime overhead, covers primary use case (LLM weights) |
| DDT architecture | Centralized on coordinator | Simple, consistent, acceptable latency for load-time dedup |
| COW mechanism | CUDA interception (proactive) | Detects writes before GPU, avoids GPU fault overhead |
| Verification | Full memcmp on every hash match | Defense in depth, standard practice in all production dedup systems |

## Implementation Phases

1. **Read-Only Dedup, Single Node** (2-3 weeks) -- DDT, page hashing, PTE extensions, cuMemSetAccess enforcement
2. **Read-Only Dedup, Cross-Node** (2-3 weeks) -- Centralized DDT, dedup protocol, network page references
3. **Copy-on-Write** (2-3 weeks) -- COW detection, handler, UVM safety net, invalidation protocol
4. **GPU-Accelerated Hashing** (1-2 weeks) -- CUDA xxHash128 kernel, batch hashing, async result transfer
5. **Optimizations** (1-2 weeks) -- DDT pruning, adaptive dedup, dedup-aware eviction, peer access

## Folder Contents

- `research/01-existing-dedup-systems.md` -- KSM, ZFS, VMware TPS, Windows, Catalyst/Gemina
- `research/02-hashing-and-detection.md` -- Hash algorithms, GPU hashing, false positive analysis, DDT design
- `research/03-copy-on-write-network.md` -- Network COW, DSM protocols, CUDA memory protection, latency analysis
- `preplan.md` -- Scope, dependencies, decisions, risks, phases, success criteria
- `progress.md` -- Lifecycle tracker
- `side-docs/` -- Notes, experiments

## Success Criteria

- Memory savings >= 75% for 4-GPU identical model deployment
- Hash overhead < 1% of model load time
- COW latency < 15us local, < 25us cross-node RDMA
- DDT overhead < 0.1% of managed pool
- Zero data corruption from hash collisions (full memcmp verification)

## Related Topics

- R10 Memory Tiering (tier-aware dedup decisions)
- R29 RDMA Multicast (broadcast shared data efficiently)
- R15 Fault Tolerance (dedup interacts with redundancy)
- R8 Kernel Parameter Introspection (write detection for COW)
