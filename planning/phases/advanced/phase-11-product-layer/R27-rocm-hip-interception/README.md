# R27: ROCm/HIP Interception

**Phase:** 11 — Product Layer
**Status:** NOT STARTED
**Priority:** LOW
**Depends On:** P7 (CUDA Completeness — proven pattern)

## Summary
Apply the same LD_PRELOAD interception technique to AMD's HIP/ROCm stack. Enables AMD GPUs to join the OuterLink pool alongside NVIDIA GPUs. True vendor-agnostic GPU pooling.

## What This Enables
- AMD + NVIDIA GPUs in the same cluster
- Vendor-agnostic GPU pooling
- Leverage cheaper AMD GPUs where appropriate
- Broader hardware compatibility

## Key Questions
- HIP driver API structure — similar enough to CUDA for same approach?
- ROCr vs HIP vs thunk layer — which to intercept?
- Can AMD and NVIDIA GPUs share the same memory pool?
- How to handle compute capability differences between vendors?

## Folder Contents
- `research/` — HIP/ROCm architecture, driver API differences
- `side-docs/` — Notes, experiments
- `preplan.md` — TO BE CREATED
- `plan.md` — TO BE CREATED
- `progress.md` — Lifecycle tracker

## Related Topics
- P7 CUDA Completeness (proven interception pattern)
- R23 Heterogeneous GPU Mixing (multi-vendor scheduling)
- R13 CUDA Graph Interception (HIP graph equivalent?)
