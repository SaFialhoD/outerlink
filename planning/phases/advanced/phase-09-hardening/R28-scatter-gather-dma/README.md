# R28: Scatter-Gather DMA

**Phase:** 9 — Hardening
**Status:** NOT STARTED
**Priority:** MEDIUM
**Depends On:** P8 (Performance phase working)

## Summary

Transfer non-contiguous memory regions in a single DMA operation using scatter-gather lists. Instead of N separate transfers for N memory fragments, one DMA descriptor handles all of them. Massive win for sparse tensors, fragmented allocations, and structured data.

## What This Enables

- Single DMA operation for multiple non-contiguous regions
- Huge reduction in transfer count for sparse/fragmented data
- Efficient sparse tensor transfers (only non-zero elements)
- Better utilization of DMA engine bandwidth

## Key Questions

- ConnectX scatter-gather list limits? (max entries per descriptor?)
- Does BAR1 (OpenDMA) support scatter-gather targeting?
- Software overhead of building scatter-gather lists vs multiple simple transfers?
- Integration with CUDA's sparse matrix/tensor APIs?

## Folder Contents

- `research/` — DMA scatter-gather, ConnectX capabilities, RDMA SGE
- `side-docs/` — Notes, experiments
- `preplan.md` — TO BE CREATED
- `plan.md` — TO BE CREATED
- `progress.md` — Lifecycle tracker

## Related Topics

- R14 Transport Compression (compress before scatter?)
- R21 GPU Direct Storage (scatter-gather for dataset loading)
- R30 Persistent Kernels (scatter-gather feed into kernel)
