# R21: GPU Direct Storage Over Network

**Phase:** 9 — Hardening
**Status:** NOT STARTED
**Priority:** MEDIUM
**Depends On:** P9 (OpenDMA working)

## Summary

Enable remote NVMe-to-GPU VRAM transfers without bouncing through host RAM on either side. Combines OpenDMA (BAR1 RDMA) with storage-side DMA to create a direct NVMe → network → GPU pipeline. Load massive datasets directly from remote storage into GPU memory.

## What This Enables

- Load training data from remote NVMe directly into GPU VRAM
- No host RAM touch on sender OR receiver
- Combined with NVMe tier (R10): transparent dataset streaming from remote storage
- Massive datasets don't need to fit in any single node's RAM

## Key Questions

- Does GDS work with our open-source BAR1 approach or only NVIDIA's proprietary stack?
- Can ConnectX DMA engine chain NVMe read + network send?
- What's the realistic throughput? (NVMe ~7GB/s vs network ~12.5GB/s)
- P2PDMA kernel support for NVMe-to-NIC?

## Folder Contents

- `research/` — GDS architecture, P2PDMA, NVMe-oF
- `side-docs/` — Notes, experiments
- `preplan.md` — TO BE CREATED
- `plan.md` — TO BE CREATED
- `progress.md` — Lifecycle tracker

## Related Topics

- R10 Memory Tiering (NVMe is Tier 4/5)
- P9 OpenDMA (BAR1 path required)
- R28 Scatter-Gather DMA (efficient multi-region loads)
