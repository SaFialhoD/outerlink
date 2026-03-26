# R28: Scatter-Gather DMA

**Phase:** 9 — Hardening
**Status:** RESEARCH COMPLETE
**Priority:** MEDIUM
**Depends On:** P8 (Performance phase working)

## Summary

Transfer non-contiguous memory regions in a single DMA operation using scatter-gather lists. Instead of N separate transfers for N memory fragments, one DMA descriptor handles all of them. Massive win for sparse tensors, fragmented allocations, and structured data.

## What This Enables

- Single DMA operation for multiple non-contiguous regions
- Huge reduction in transfer count for sparse/fragmented data
- Efficient sparse tensor transfers (only non-zero elements)
- Better utilization of DMA engine bandwidth

## Key Findings (Research Phase)

- **ConnectX-5 supports 30 SGEs per work request** — covers most fragmentation patterns
- **30 x 64KB = 1.875 MB** per single RDMA operation for non-contiguous pages
- **Hardware SG overhead is ~6us** vs ~10us for software pre-pack — both negligible vs network time
- **BAR1 + GPU MMU can provide implicit scatter** — scattered VRAM can appear contiguous to the NIC
- **~40% of ML transfers benefit** from scatter-gather (MoE, sparse models, fragmented VRAM)
- **Software pre-pack fallback** handles unlimited fragments when > 30

## Key Questions (Answered)

- ConnectX scatter-gather list limits? **30 SGEs per WR (max_sge = 30)**
- Does BAR1 (OpenDMA) support scatter-gather targeting? **Yes — GPU MMU remaps BAR1 to scattered VRAM**
- Software overhead of building scatter-gather lists vs multiple simple transfers? **~6us vs ~10us, both negligible**
- Integration with CUDA's sparse matrix/tensor APIs? **cuSPARSE uses 3 separate arrays (CSR), natural fit for 3-SGE transfers**

## Folder Contents

- `research/01-rdma-scatter-gather.md` — RDMA SGE mechanics, ConnectX-5 limits, performance
- `research/02-gpu-sparse-data.md` — GPU sparse formats, ML sparsity patterns, VRAM fragmentation
- `research/03-scatter-gather-pipeline.md` — End-to-end pipeline design, OpenDMA integration
- `side-docs/` — Notes, experiments
- `preplan.md` — Scope, dependencies, implementation phases
- `plan.md` — TO BE CREATED
- `progress.md` — Lifecycle tracker

## Related Topics

- R10 Memory Tiering (page table provides fragment addresses)
- R12 Memory Deduplication (dedup may scatter pages)
- R14 Transport Compression (compress before scatter?)
- R20 NCCL Backend (scatter-gather for non-contiguous collective buffers)
- R21 GPU Direct Storage (scatter-gather for dataset loading)
