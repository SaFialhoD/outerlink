# R16: BlueField DPU Offload

**Phase:** 10 — Compute Distribution
**Status:** RESEARCH COMPLETE
**Priority:** MEDIUM
**Depends On:** P8 (Performance phase working)

## Summary

Offload OuterLink's transport logic to BlueField DPU's ARM cores. The DPU runs routing decisions, compression, and memory management at the network edge — the host CPU becomes almost unnecessary for data movement. The DPU's integrated ConnectX NIC can also perform OpenDMA (GPU VRAM BAR1 access) natively, achieving true zero-CPU data movement.

## What This Enables
- True zero-CPU data movement (DPU handles entire data path)
- Routing decisions at line rate on the NIC
- Hardware compression/decompression without host CPU (deflate on BF-2, + LZ4 on BF-3)
- Host CPU freed entirely for application work
- Prefetch scheduling at the network edge (sees traffic patterns before host)
- DPU-to-GPU BAR1 direct access (OpenDMA via DPU's ConnectX NIC)

## Key Questions — Answered

| Question | Answer |
|---|---|
| Programming model? | **DOCA SDK** via Rust FFI (bindgen + safe wrappers). DPDK as secondary option for packet I/O. |
| ARM cores sufficient? | **Yes with caveats.** BF-2 (8x A72) handles transport logic. DOCA Flow offloads fast-path to hardware. BF-3 (16x A78 + 128 APP cores) is more comfortable. |
| Host/DPU work split? | **Control plane on host** (CUDA interception, app interface). **Data plane on DPU** (routing, compression, RDMA, BAR1 access). |
| BF-2 vs BF-3? | **Both supported.** BF-2 for dev/test ($150-500 used). BF-3 for production (LZ4 HW, packet processing cores, PCIe Gen5). |

## Folder Contents
- `research/01-bluefield-architecture.md` — Hardware specs, BF-2 vs BF-3, PCIe topology, memory architecture
- `research/02-programming-models.md` — DOCA SDK, Flow, DMA, Compress, DPDK, Rust FFI strategy
- `research/03-outerlink-offload-design.md` — What to offload, data flows, latency analysis, phased implementation
- `preplan.md` — Scope, decisions, unknowns, risk assessment, implementation phases
- `plan.md` — TO BE CREATED (after pre-plan review)
- `progress.md` — Lifecycle tracker
- `side-docs/` — Notes, experiments

## Implementation Phases (from pre-plan)
- **Phase A:** DOCA Foundation — Rust FFI, host↔DPU communication, DMA proof of concept
- **Phase B:** Transport Offload — connection management, routing, RDMA on DPU
- **Phase C:** Compression Offload — DOCA Compress (deflate/LZ4)
- **Phase D:** GPU BAR1 Integration — DPU's ConnectX writes directly to GPU VRAM
- **Phase E:** Prefetch on DPU — traffic pattern monitoring, proactive page fetching

## Related Topics
- R14 Transport Compression (compression offloaded to DPU hardware)
- R17 Topology-Aware Scheduling (routing decisions on DPU)
- R11 Speculative Prefetching (prefetch logic on DPU ARM cores)
- R10 Memory Hierarchy (page tables managed by DPU)
- R9 Multi-Transport (DPU manages transport selection)
- OpenDMA Phase 5 (BAR1 access pattern used natively by DPU's ConnectX)
