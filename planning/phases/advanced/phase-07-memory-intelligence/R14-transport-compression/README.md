# R14: Transport-Layer Compression

**Phase:** 7 — Memory Intelligence
**Status:** NOT STARTED
**Priority:** HIGH
**Depends On:** P6 (Core Transport working)

## Summary

Add pluggable compression to OuterLink's transport layer. Different strategies for different data: LZ4 for CPU-side DRAM transfers, nvCOMP (GPU-native) for VRAM data, delta encoding for iterative workloads (training gradients). Adaptive selection based on data characteristics and link conditions.

## What This Enables

- 2-10x effective bandwidth for compressible data
- Gradient compression for distributed training (often 90%+ compressible)
- Trade cheap GPU/CPU compute for expensive network bandwidth
- Universal multiplier — benefits every transfer type

## Key Questions

- Where does compression happen? (sender CPU, sender GPU, NIC?)
- How to detect compressibility without overhead?
- Threshold: when is compression slower than sending raw?
- nvCOMP licensing and integration with Rust?

## Folder Contents

- `research/` — Research on compression algorithms and benchmarks
- `side-docs/` — Notes, experiments
- `preplan.md` — TO BE CREATED
- `plan.md` — TO BE CREATED
- `progress.md` — Lifecycle tracker

## Related Topics

- R10 Memory Tiering (which tier benefits most from compression?)
- R28 Scatter-Gather DMA (compress before scatter?)
- R9 Multi-Transport (different compression per link type?)
