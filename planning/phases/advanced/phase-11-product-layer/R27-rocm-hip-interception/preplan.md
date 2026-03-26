# PRE-PLAN: R27 — ROCm/HIP Interception

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** DRAFT
**Depends On:** P7 (CUDA Completeness — proven interception pattern)

## Purpose

Define WHAT needs to be planned and built to extend OuterLink's GPU pooling to AMD GPUs via HIP API interception, enabling vendor-agnostic GPU pooling.

---

## Scope Definition

### In Scope

1. **HIP interception library** — `outerlink_hip.so` that intercepts `libamdhip64.so` via LD_PRELOAD
2. **HIP function coverage** — Phased: ~30 core functions (Phase 1), ~130 (Phase 2), ~350 full (Phase 3)
3. **Server-side HIP dispatch** — GPU server component that executes real HIP calls on AMD hardware
4. **AMD RDMA transport** — ConnectX-5 RDMA to AMD GPU VRAM via ROCnRDMA/dma-buf
5. **Unified scheduler awareness** — Scheduler knows about AMD GPU capabilities alongside NVIDIA
6. **Code generation tooling** — Generate interception stubs from HIP headers (same approach as CUDA)

### Out of Scope (for this phase)

- Cross-vendor kernel translation (PTX <-> AMDGCN) — not feasible, not planned
- AMD-specific optimization (MI300X features, CDNA-specific tuning) — future work
- Windows AMD GPU support — Linux only for now
- Consumer Radeon GPUs without ROCm support — only ROCm-supported hardware
- HIP on NVIDIA backend (HIP apps redirected to CUDA) — NVIDIA apps already use our CUDA interception

---

## Key Decisions Needed

| # | Decision | Options | Research Doc |
|---|---|---|---|
| D1 | Interception layer | HIP (libamdhip64.so) vs ROCr (libhsa-runtime64.so) vs KFD thunk | [01-rocm-hip-architecture.md](research/01-rocm-hip-architecture.md) |
| D2 | RDMA mechanism | Peer Memory Client (ROCnRDMA) vs dma-buf | [02-interception-feasibility.md](research/02-interception-feasibility.md) |
| D3 | Node architecture | Homogeneous (one vendor per node) vs mixed | [03-cross-vendor-challenges.md](research/03-cross-vendor-challenges.md) |
| D4 | Wire protocol | Same protocol as CUDA (unified) vs separate AMD protocol | [02-interception-feasibility.md](research/02-interception-feasibility.md) |
| D5 | Phase 1 function set | Which ~30 functions to intercept first? | [02-interception-feasibility.md](research/02-interception-feasibility.md) |

### Preliminary Recommendations

- **D1:** HIP layer (`libamdhip64.so`). Highest-level public API, stable, well-documented, 1:1 CUDA mapping.
- **D2:** dma-buf (modern, preferred by AMD). Fall back to Peer Memory Client on older kernels.
- **D3:** Recommend homogeneous nodes. Support mixed but don't optimize for it.
- **D4:** Unified wire protocol. Extend existing protocol with vendor-specific opcodes. Transport layer is vendor-agnostic.
- **D5:** Memory (hipMalloc/Free/Memcpy), execution (hipLaunchKernel, hipModuleLoad), device (hipSetDevice, hipGetDeviceProperties), streams, events, sync.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│              OuterLink Scheduler                 │
│  (vendor-aware: knows NVIDIA and AMD GPUs)       │
├────────────────────┬────────────────────────────┤
│   CUDA Client      │       HIP Client            │
│  outerlink_cuda.so │     outerlink_hip.so         │
│  (intercepts       │     (intercepts              │
│   libcuda.so)      │      libamdhip64.so)         │
├────────────────────┴────────────────────────────┤
│            Unified Transport (RDMA/TCP)           │
├────────────────────┬────────────────────────────┤
│  NVIDIA GPU Server │     AMD GPU Server           │
│  (calls real CUDA) │     (calls real HIP)          │
│  GPUDirect/OpenDMA │     ROCnRDMA/dma-buf          │
├────────────────────┼────────────────────────────┤
│   NVIDIA GPU HW    │      AMD GPU HW              │
└────────────────────┴────────────────────────────┘
```

### Key Design Principle: Shared Transport, Separate API Layers

The transport layer (TCP + io_uring, UCX RDMA) is vendor-agnostic. Only the interception library (client-side) and GPU dispatch (server-side) are vendor-specific. This maximizes code reuse.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ROCm consumer GPU support is limited | HIGH | HIGH | Target MI-series or well-supported Radeon (RX 7900). Document supported hardware. |
| HIP API changes across ROCm versions | MEDIUM | MEDIUM | Pin to ROCm LTS releases. Test across versions in CI. |
| dma-buf RDMA not working with ConnectX-5 | LOW | HIGH | Prototype early. Fallback to Peer Memory Client or host-staged transfers. |
| No AMD test hardware available | HIGH | CRITICAL | Acquire at minimum one ROCm-supported GPU for development. Cloud instances as fallback. |
| LD_PRELOAD not intercepting all HIP calls | LOW | MEDIUM | Audit HIP apps for direct HSA calls. Add HSA interception if needed. |
| Cross-vendor scheduling complexity | MEDIUM | MEDIUM | Phase 1: AMD-only pool. Cross-vendor scheduling in Phase 2. |

---

## Dependencies

| Dependency | Status | Why Needed |
|---|---|---|
| P7: CUDA Completeness | Required | Proves the interception pattern works end-to-end |
| Transport layer (P1-P3) | Required | TCP + RDMA transport reused for AMD |
| AMD GPU test hardware | **BLOCKER** | Cannot develop or test without hardware |
| ROCm installation | Required | Development environment with ROCm stack |
| ConnectX-5 + ROCm RDMA | Required for RDMA | UCX with `--with-rocm` for RDMA transport |

---

## Deliverables for Planning Phase

| # | Deliverable | Description |
|---|---|---|
| 1 | HIP function audit | Complete list of HIP functions to intercept, categorized by priority |
| 2 | Code generation spec | How to generate HIP interception stubs from headers |
| 3 | Wire protocol extension | Protocol changes for vendor-aware GPU operations |
| 4 | Server-side HIP dispatch design | How the GPU server executes HIP calls on AMD hardware |
| 5 | RDMA integration design | ROCnRDMA/dma-buf integration with OuterLink transport |
| 6 | Test plan | Hardware requirements, test matrix (ROCm versions, GPU models) |
| 7 | AMD hardware procurement | Identify and acquire test GPU(s) |
| 8 | Acceptance criteria | What "HIP interception works" means, quantitatively |

---

## Estimated Effort

| Component | Complexity | Estimate | Reuse from CUDA |
|---|---|---|---|
| HIP function audit + stub generation | Low | 1-2 weeks | Tooling reused |
| outerlink_hip.so (Phase 1: 30 functions) | Medium | 3-4 weeks | Architecture reused |
| Server-side HIP dispatch | Medium | 2-3 weeks | Pattern reused |
| Wire protocol extension | Low | 1 week | 90% reused |
| RDMA integration (dma-buf) | Medium | 2-3 weeks | Transport reused |
| Testing + debugging | High | 3-4 weeks | Test framework reused |
| outerlink_hip.so Phase 2 (+100 functions) | Medium | 3-4 weeks | Stub generation |
| outerlink_hip.so Phase 3 (full coverage) | Medium | 3-4 weeks | Stub generation |
| **Total (through Phase 2)** | | **~15-21 weeks** | |
| **Total (through Phase 3)** | | **~18-25 weeks** | |

**Note:** ~40-50% less effort than the original CUDA implementation because architecture, protocol, and transport are reusable.

---

## Hardware Requirements

### Minimum for Development

| Option | GPU | ROCm Support | Cost (approx) | Notes |
|---|---|---|---|---|
| Consumer | RX 7900 XTX (24GB VRAM) | ROCm 6.0+ | ~$700-800 | Best consumer option, RDNA3, gfx1100 |
| Consumer | RX 7900 XT (20GB VRAM) | ROCm 6.0+ | ~$600-700 | Good alternative |
| Datacenter | MI100 (32GB HBM2) | ROCm 5.0+ | ~$500-800 (used) | CDNA1, gfx908, good for testing |
| Cloud | Any MI-series instance | Full ROCm | $2-5/hr | For CI/testing, no capital expense |

### Recommended

One consumer RX 7900 XTX for local development + cloud MI-series instances for CI testing. This covers both RDNA and CDNA architectures.

---

## Open Questions

1. **When to start?** R27 depends on P7 (CUDA completeness). Should we start HIP research/prototyping before P7 is done?
2. **Shared code structure:** How do we organize the codebase for dual-vendor support? Separate crates? Feature flags? Trait-based abstraction?
3. **AMD GPU market trajectory:** Is AMD GPU compute growing enough to justify the investment? MI300X adoption is strong in datacenters; consumer ROCm support is improving but still limited.
4. **Community interest:** Would R27 attract AMD GPU community contributors to OuterLink?

---

## Related Documents

- [research/01-rocm-hip-architecture.md](research/01-rocm-hip-architecture.md)
- [research/02-interception-feasibility.md](research/02-interception-feasibility.md)
- [research/03-cross-vendor-challenges.md](research/03-cross-vendor-challenges.md)
- [R23: Heterogeneous GPU Mixing](../R23-heterogeneous-gpu-mixing/) (if in same phase folder, adjust path)
- [R3: CUDA Interception](../../../../research/R3-cuda-interception.md)
