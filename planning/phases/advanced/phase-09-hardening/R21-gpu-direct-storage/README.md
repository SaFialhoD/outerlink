# R21: GPU Direct Storage Over Network

**Phase:** 9 — Hardening
**Status:** RESEARCH COMPLETE
**Priority:** MEDIUM
**Depends On:** P9 (OpenDMA working), R10 (Memory Tiering)

## Summary

Enable remote NVMe-to-GPU VRAM transfers without bouncing through host RAM on either side. Combines OpenDMA (BAR1 RDMA) with storage-side DMA to create a direct NVMe --> network --> GPU pipeline. Load massive datasets directly from remote storage into GPU memory.

## What This Enables

- Load training data from remote NVMe directly into GPU VRAM
- No host RAM touch on sender OR receiver (full P2P path)
- Combined with NVMe tier (R10): transparent dataset streaming from remote storage
- Massive datasets don't need to fit in any single node's RAM

## Key Findings from Research

| Topic | Finding |
|-------|---------|
| NVIDIA GDS on GeForce | Native GDS mode blocked on GeForce GPUs (Data Center/Quadro only). Compatibility mode is just POSIX I/O — useless. |
| OpenDMA bypasses GDS restriction | BAR1 DMA works on ALL GPUs including GeForce. We don't need GDS. |
| ConnectX-5 NVMe-oF target offload | Hardware chains NVMe read + RDMA send with zero CPU. Embedded PCIe switch enables NVMe-to-NIC P2P. |
| P2PDMA upstream status | Kernel 6.2+ with userspace support. Enabled by default in Ubuntu 24.04. AMD Zen fully supports P2PDMA. |
| NVMe CMB | Rare in consumer SSDs. Not required — ConnectX-5 embedded switch or host-staged path work without it. |
| SPDK vs kernel nvmet | Kernel nvmet + ConnectX-5 offload is simpler and achieves similar throughput. SPDK is Plan B. |
| Throughput bottleneck | Single Gen4 NVMe: ~7 GB/s. Network: ~12.5 GB/s. Need 2x NVMe striped to saturate network. |
| NVIDIA moving to P2PDMA | CUDA 12.8 / GDS 1.16 uses upstream kernel P2PDMA instead of nvidia-fs.ko. Validates open-source approach. |

## Key Decisions

1. **Start with host-staged sender** — NVMe → host RAM → RDMA send. Simple, works everywhere. Optimize to NVMe-oF offload later.
2. **OpenDMA on receiver from day one** — RDMA receive → BAR1 DMA → GPU VRAM. Zero host RAM on GPU node.
3. **Kernel nvmet + ConnectX-5 offload** — not SPDK. Hardware offload gives near-zero CPU with lower complexity.
4. **Hybrid protocol** — NVMe-oF on sender (for hardware offload), custom RDMA + OpenDMA on receiver (for GPU delivery).
5. **Three-layer API** — block-level, file-level (cuFile-like), page-level (R10 integration).

## Implementation Phases (7-11 weeks total)

1. **Host-Staged Remote Storage** (2-3 weeks) — NVMe read → pinned buf → RDMA → OpenDMA → GPU VRAM
2. **NVMe-oF Integration** (2-3 weeks) — nvmet subsystem, ConnectX-5 offload, P2PDMA topology validation
3. **Full P2P Pipeline + API** (2-3 weeks) — Zero-copy both sides, batch/async API, CUDA stream integration
4. **R10 Tiering Integration** (1-2 weeks) — Tier 5 page faults, prefetching, eviction to NVMe

## Success Criteria

| Metric | Target |
|--------|--------|
| Single NVMe → GPU throughput (host-staged) | >6 GB/s |
| Single NVMe → GPU throughput (full P2P) | >6.5 GB/s |
| Sender CPU usage with NVMe-oF offload | <5% of baseline |
| Multi-NVMe striped throughput | >12 GB/s (2x NVMe) |
| End-to-end latency (first byte, 4KB) | <200 us |
| R10 Tier 5 page fault latency | <500 us |

## Folder Contents

- `research/01-nvidia-gds-architecture.md` — GDS internals, cuFile API, native vs compat mode, GeForce limitations
- `research/02-p2pdma-and-nvme.md` — Linux P2PDMA framework, NVMe CMB, NVMe-oF, SPDK, ConnectX-5 offload
- `research/03-remote-gds-pipeline.md` — Full pipeline design, throughput analysis, API design, R10 integration
- `preplan.md` — Scope, dependencies, decisions, risks, implementation phases, success criteria
- `progress.md` — Lifecycle tracker
- `side-docs/` — Notes, experiments

## Related Topics

- R10 Memory Tiering (NVMe is Tier 4/5)
- P9 OpenDMA (BAR1 path required — receiver side)
- R28 Scatter-Gather DMA (efficient multi-region loads)
