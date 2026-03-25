# R19: Network Page Faults / Unified Memory

**Phase:** 8 --- Smart Memory
**Status:** RESEARCH COMPLETE
**Priority:** HIGH
**Depends On:** R10 (Memory Tiering), R11 (Speculative Prefetching), R12 (Memory Deduplication)

## Summary

Implement transparent remote memory access via demand paging. When a GPU needs memory that lives on a remote node, the system automatically fetches the page, installs it locally via cuMemMap, and the GPU proceeds. Applications use plain pointers --- no explicit cudaMemcpy calls needed. Combined with R11 prefetching (which prevents >90% of faults), R19 is the correctness safety net ensuring no access ever fails.

## Architecture Decision

**Phase 1 uses cuMemMap pre-launch demand paging (no kernel module).** Before each kernel launch, the interception layer checks that all required pages are locally mapped. Missing pages are fetched via RDMA/TCP and installed via cuMemMap. If a kernel still hits an unmapped page (R11 misprediction), the CUDA error is caught, the page is fetched, and the kernel is re-launched.

**Phase 2 adds a full I/S/E coherency protocol** with directory-based tracking at home nodes, ensuring consistency across nodes for concurrent read/write patterns.

**Phase 3 adds thrashing prevention** via bounce detection, shared-read promotion, write-broadcast, and page pinning.

**Phase 4 (future) adds a kernel module** for true in-execution fault handling without kernel restart.

## What This Enables

- True unified address space across the cluster
- CUDA apps "just work" without any memory management code
- Combined with prefetching (R11), most faults are avoided entirely
- Simplest possible programming model: one flat memory space
- Consistent data across nodes via SWMR coherency protocol

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Fault mechanism (Phase 1) | cuMemMap pre-launch mapping | No kernel module, works on all CUDA 10.2+ GPUs |
| Fallback mechanism | Kernel crash + re-launch | Safety net for R11 mispredictions |
| Consistency model | SWMR (Single-Writer/Multiple-Reader) | Maps to GPU workload patterns (weights = shared-read, activations = exclusive) |
| Page states | I/S/E (Invalid/Shared/Exclusive) | Simpler than MESI, sufficient for network scale |
| Directory | Distributed (home node per page) | No SPOF, scales linearly |
| Thrashing response | Escalating: promotion -> broadcast -> pinning | Proportional response to severity |
| Kernel module | Deferred to Phase 4 | Minimize deployment complexity |

## Folder Contents

- `research/01-gpu-page-fault-mechanisms.md` --- NVIDIA UVM, cuMemMap, ATS, HMM, userfaultfd
- `research/02-distributed-shared-memory.md` --- DSM systems, RDMA memory, CXL
- `research/03-coherency-and-thrashing.md` --- I/S/E protocol, thrashing detection, mitigation
- `preplan.md` --- Scope, dependencies, decisions, risks, implementation phases
- `progress.md` --- Lifecycle tracker
- `side-docs/` --- Notes, prototypes

## Related Topics

- R10 Memory Tiering (page table and tier hierarchy)
- R11 Speculative Prefetching (prefetch prevents >90% of faults)
- R12 Memory Deduplication (deduped pages never thrash)
- R17 Topology-Aware Scheduling (picks nearest copy for page fetch)
- R18 Virtual NVLink (builds multi-GPU coherency on R19's protocol)
