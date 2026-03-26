# R27 Research: ROCm/HIP Architecture

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Priority:** HIGH

## Purpose

Map AMD's GPU software stack from application level down to kernel driver, understand each layer's role, and identify which layer OuterLink should intercept for AMD GPU pooling.

---

## TL;DR

AMD's GPU stack has 5 layers: HIP (user API) -> ROCclr (common runtime) -> ROCr/HSA (hardware abstraction) -> ROCt/KFD thunk (kernel interface) -> amdgpu kernel driver. Unlike CUDA which has separate Runtime and Driver APIs, HIP is a single unified API. The interception target should be **HIP** (via `libamdhip64.so`), mirroring our CUDA Driver API interception strategy. HIP provides the complete surface area (memory management, kernel launch, streams, events) at the highest useful abstraction level.

---

## 1. ROCm Stack Overview

ROCm (Radeon Open Compute) is AMD's open-source GPU computing platform, analogous to NVIDIA's CUDA ecosystem.

### Full Stack (Top to Bottom)

```
┌─────────────────────────────────────┐
│  Applications (PyTorch, TF, etc.)   │
├─────────────────────────────────────┤
│  Libraries (rocBLAS, MIOpen, RCCL)  │
├─────────────────────────────────────┤
│  HIP Runtime API (libamdhip64.so)   │  <-- INTERCEPTION TARGET
├─────────────────────────────────────┤
│  ROCclr (Common Language Runtime)   │
├─────────────────────────────────────┤
│  ROCr / HSA Runtime (libhsa-rt.so)  │
├─────────────────────────────────────┤
│  ROCt / KFD Thunk (libhsakmt.so)   │
├─────────────────────────────────────┤
│  amdgpu Kernel Driver (/dev/kfd)    │
├─────────────────────────────────────┤
│  AMD GPU Hardware                   │
└─────────────────────────────────────┘
```

### Layer Details

#### HIP (Heterogeneous-Compute Interface for Portability)

- **Library:** `libamdhip64.so` (AMD backend) or `libhip_hcc.so` (legacy)
- **Role:** User-facing C++ API for GPU programming. Designed to be nearly 1:1 with CUDA.
- **Key property:** HIP is a PORTABILITY layer. The same HIP code compiles for both AMD (ROCm backend) and NVIDIA (CUDA backend).
- **Compilation:** HIP-Clang (LLVM/Clang-based) for AMD; redirects to NVCC for NVIDIA.
- **Macro:** `__HIP_PLATFORM_AMD__` selects the AMD backend at compile time.

#### ROCclr (Common Language Runtime)

- **Role:** Intermediate layer between programming languages (HIP, OpenCL) and the HSA runtime.
- **Status:** Being gradually absorbed into the upper layers (HIP and OpenCL). Previously handled compiler routing.
- **Note:** This is an INTERNAL implementation detail, not a public API. OuterLink should NOT target this layer.

#### ROCr (ROCm Runtime) / HSA Runtime

- **Library:** `libhsa-runtime64.so`
- **Role:** AMD's implementation of the HSA (Heterogeneous System Architecture) specification. Provides low-level GPU access: memory allocation, signal management, AQL queue dispatch.
- **Key APIs:** `hsa_memory_allocate`, `hsa_signal_create`, `hsa_queue_create`, AQL packet submission
- **Relationship to HIP:** HIP functions internally call ROCr. For example:
  - `hipMalloc()` -> `rocr::AMD::MemoryRegion::AllocateImpl()`
  - `hipLaunchKernel()` -> AQL (Architected Queuing Language) packet dispatch on HSA queues
  - `hipMemcpy()` -> HSA memory copy (via DMA engine or shader blit, controlled by `HSA_ENABLE_SDMA`)

#### ROCt / KFD Thunk

- **Library:** `libhsakmt.so`
- **Role:** Thin userspace interface to the ROC Kernel Fusion Driver (KFD). Translates HSA runtime calls to ioctl calls on `/dev/kfd`.
- **APIs:** `hsaKmt*` functions
- **Note:** Very low-level, kernel-adjacent. Intercepting here would be fragile and version-dependent.

#### amdgpu Kernel Driver

- **Device:** `/dev/kfd` (compute), `/dev/dri` (graphics/display)
- **Role:** Kernel-mode driver managing GPU hardware. Handles memory management, command submission, interrupt handling.
- **Permissions:** Requires `video` group membership.
- **IOMMU:** Required for peer-to-peer DMA between devices.

---

## 2. HIP vs CUDA API Comparison

### Structural Differences

| Aspect | CUDA | HIP |
|---|---|---|
| API levels | Runtime API + Driver API (separate) | Single unified API |
| Runtime library | `libcudart.so` | `libamdhip64.so` |
| Driver library | `libcuda.so` | No separate driver API |
| Compiler | NVCC (two-pass) | HIP-Clang (single-pass) |
| ISA | PTX (intermediate) -> SASS (native) | AMDGCN ISA (direct) |
| Kernel launch syntax | `<<<blocks, threads>>>` | `<<<blocks, threads>>>` (same since ROCm 5.3) |

### API Naming Convention

Nearly all CUDA functions have a HIP equivalent with `cuda` -> `hip` prefix replacement:

| CUDA | HIP | Notes |
|---|---|---|
| `cudaMalloc` | `hipMalloc` | Identical semantics |
| `cudaMemcpy` | `hipMemcpy` | Identical semantics |
| `cudaFree` | `hipFree` | Identical semantics |
| `cudaMemcpyAsync` | `hipMemcpyAsync` | Identical semantics |
| `cudaLaunchKernel` | `hipLaunchKernel` | Identical semantics |
| `cudaStreamCreate` | `hipStreamCreate` | Identical semantics |
| `cudaEventCreate` | `hipEventCreate` | Identical semantics |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` | Identical semantics |
| `cudaGetDeviceProperties` | `hipGetDeviceProperties` | Struct fields differ slightly |
| `cudaSetDevice` | `hipSetDevice` | Identical semantics |
| `cuModuleLoad` | `hipModuleLoad` | Driver API equivalent in HIP |
| `cuModuleGetFunction` | `hipModuleGetFunction` | Driver API equivalent in HIP |
| `cuLaunchKernel` | `hipModuleLaunchKernel` | Driver API equivalent in HIP |

### CUDA Driver API Equivalents in HIP

Unlike CUDA which has separate Runtime and Driver APIs, HIP merges both into one API. The CUDA Driver API functions (`cu*` prefix) have HIP equivalents:

| CUDA Driver API Category | HIP Equivalent | Coverage |
|---|---|---|
| Context management (`cuCtx*`) | `hipCtx*` (deprecated, device-primary context model) | Partial — HIP uses a different context model |
| Module management (`cuModule*`) | `hipModule*` | Good coverage |
| Memory management (`cuMem*`) | `hipMem*` | Good coverage |
| Stream management (`cuStream*`) | `hipStream*` | Good coverage |
| Event management (`cuEvent*`) | `hipEvent*` | Good coverage |
| Execution control (`cuLaunchKernel`) | `hipModuleLaunchKernel` | Full |
| Device management (`cuDevice*`) | `hipDevice*` / `hipGet*` | Good coverage |
| Peer access (`cuCtxEnablePeerAccess`) | `hipDeviceEnablePeerAccess` | Full |

### HIPIFY Coverage

AMD's HIPIFY tool provides comprehensive mapping tables. The HIPIFY documentation tracks support status per function with markers: Added (A), Deprecated (D), Changed (C), Removed (R), Experimental (E).

Key gaps where HIP does NOT have a CUDA equivalent:
- Some CUDA Driver API functions related to NVIDIA-specific features (e.g., certain texture/surface operations)
- CUDA cooperative groups have partial HIP support
- Some CUDA graph API functions are missing or experimental
- NVIDIA-specific library calls (cuBLAS, cuDNN) use different ROCm libraries (rocBLAS, MIOpen)

---

## 3. Which Layer to Intercept?

### Option A: HIP Layer (libamdhip64.so) — RECOMMENDED

| Pros | Cons |
|---|---|
| Highest-level API, closest to application intent | Larger API surface than CUDA Driver API alone |
| 1:1 mapping with CUDA functions (known territory) | Must handle both "runtime" and "driver" functions |
| Well-documented, stable API | Some functions may bypass HIP (direct HSA calls) |
| LD_PRELOAD on `libamdhip64.so` works identically to our CUDA approach | |
| HIPIFY mapping tables give us the exact function list | |
| Most HIP applications link dynamically against `libamdhip64.so` | |

### Option B: ROCr/HSA Layer (libhsa-runtime64.so)

| Pros | Cons |
|---|---|
| Lower-level, fewer functions | HSA is an internal API, less stable across versions |
| Captures ALL GPU operations (including those bypassing HIP) | Harder to map to our existing CUDA interception |
| | Poor documentation compared to HIP |
| | Direct HSA usage by applications is rare |

### Option C: KFD Thunk Layer (libhsakmt.so)

| Pros | Cons |
|---|---|
| Captures absolutely everything | Extremely low-level, kernel-adjacent |
| | Highly version-dependent |
| | Minimal documentation |
| | No application uses this directly |

### Verdict: HIP Layer

Intercept `libamdhip64.so` via LD_PRELOAD. This mirrors our CUDA Driver API strategy:

| CUDA OuterLink | HIP OuterLink |
|---|---|
| `LD_PRELOAD=outerlink_cuda.so` | `LD_PRELOAD=outerlink_hip.so` |
| Intercepts `libcuda.so` (Driver API) | Intercepts `libamdhip64.so` |
| ~222 functions | ~300-400 functions (combined runtime+driver) |
| `cuLaunchKernel` -> forward to remote | `hipLaunchKernel` -> forward to remote |
| `cuMemAlloc` -> track + forward | `hipMalloc` -> track + forward |

---

## 4. Memory Management in HIP

### Allocation Types

| Function | Description | CUDA Equivalent |
|---|---|---|
| `hipMalloc` | Device (VRAM) memory | `cudaMalloc` |
| `hipMallocManaged` | Unified/managed memory (auto-migrates) | `cudaMallocManaged` |
| `hipHostMalloc` | Pinned host memory (DMA-able) | `cudaMallocHost` |
| `hipMallocPitch` | 2D pitched allocation | `cudaMallocPitch` |
| `hipMalloc3D` | 3D allocation | `cudaMalloc3D` |
| `hipExtMallocWithFlags` | Extended allocation with flags | N/A (AMD-specific) |

### Memory Copy

| Function | Description | CUDA Equivalent |
|---|---|---|
| `hipMemcpy` | Synchronous copy (H2D, D2H, D2D, H2H) | `cudaMemcpy` |
| `hipMemcpyAsync` | Asynchronous copy | `cudaMemcpyAsync` |
| `hipMemset` | Fill memory | `cudaMemset` |
| `hipMemcpy2D` | 2D copy | `cudaMemcpy2D` |
| `hipMemcpy3D` | 3D copy | `cudaMemcpy3D` |

### Internal Memory Architecture

- `hipMalloc` internally calls `rocr::AMD::MemoryRegion::AllocateImpl()`
- ROCr uses a block allocator with caching: `hipFree` does NOT immediately release to the driver. Blocks are cached for reuse.
- `HSA_ENABLE_SDMA=0` forces host-to-device copies to use shader blit kernels instead of DMA engines (useful for debugging)
- Memory regions are typed: system (host), GPU local (VRAM), GPU visible (BAR-mapped)

---

## 5. AMD GPU Memory Architecture

### VRAM Access

- AMD GPUs expose VRAM via PCIe BARs, similar to NVIDIA
- Resizable BAR ("Smart Access Memory") maps the entire VRAM to CPU address space
- Without Resizable BAR: only 256MB aperture visible to CPU at a time
- With Resizable BAR: full VRAM (e.g., 24GB on RX 7900 XTX) is CPU-accessible

### BAR Properties

| Property | Details |
|---|---|
| BAR type | Uncached, write-combined |
| CPU reads | SLOW (high latency from PCIe round-trip) |
| CPU writes | Fast if sequential (write-combining buffers) |
| Random access | Poor performance (defeats write combining) |
| Alignment | 64-byte alignment recommended for writes |

### Peer-to-Peer DMA

P2P DMA between PCIe devices requires:
- IOMMU enabled (creates virtual I/O address space per device)
- BAR memory within device's physical addressing limit (e.g., 44-bit for some NICs)
- BIOS configuration: Above-4G decoding, CSM disabled, MMIO aperture placement

---

## 6. AMD GPU + RDMA

### ROCnRDMA (Peer Memory Client)

AMD's equivalent to NVIDIA GPUDirect RDMA. Enables InfiniBand/RoCE NICs to directly read/write GPU VRAM.

**Implementation:** The amdkfd kernel driver implements the Linux Peer Memory Client API, which allows RDMA subsystems (e.g., Mellanox OFED) to query and obtain physical addresses of GPU VRAM for direct DMA.

### Two Mechanisms

| Mechanism | Status | How |
|---|---|---|
| Peer Memory Client API (legacy) | Mature | amdkfd driver exports physical addresses of GPU memory to RDMA subsystem |
| dma-buf (modern) | Preferred | `hsa_amd_portable_export_dmabuf()` exports GPU memory as dma-buf FD, registered with `ibv_reg_dmabuf_mr()` |

### ConnectX-5 + AMD GPU RDMA

ConnectX-5 supports RDMA to AMD GPU VRAM via:
1. **UCX:** Communication library supporting ROCm-aware transports. Compile UCX with `--with-rocm` flag.
2. **Open MPI:** Can be built with UCX backend + ROCm support for GPU-aware MPI.
3. **RCCL:** AMD's equivalent to NCCL, uses RoCE/InfiniBand for multi-node collectives.

### MORI (Modular RDMA Interface) — New from AMD (2025)

AMD released MORI, a composable framework for RDMA + GPU applications. Supports:
- ConnectX-7 (NVIDIA/Mellanox)
- Broadcom Thor2
- AMD Pensando DSC
- IBGDA (InfiniBand GPUDirect Async)
- GDS (GPUDirect Storage)

**Relevance:** MORI provides building blocks for RDMA communication with AMD GPUs. OuterLink could use MORI as an alternative to our custom RDMA transport for AMD GPU nodes.

---

## Open Questions

1. **HIP function count:** What's the exact number of HIP functions we need to intercept? HIPIFY tables suggest 300-400 for full coverage. Need to audit.
2. **HIP context model:** HIP uses device-primary contexts (deprecated explicit context management). How does this affect our interception? CUDA's explicit `cuCtxCreate` maps cleanly; HIP's implicit context may need special handling.
3. **hiprtc and libamdhip64.so overlap:** HIPRTC symbols are currently in `libamdhip64.so` but being moved to separate `libhiprtc.so`. Do we need to intercept both?
4. **Static linking:** Are any AMD GPU applications statically linking HIP? If so, LD_PRELOAD won't work for those.
5. **ROCm version stability:** How stable is the HIP API across ROCm versions? Do functions get removed or change signatures?

---

## Related Documents

- [02-interception-feasibility.md](02-interception-feasibility.md)
- [03-cross-vendor-challenges.md](03-cross-vendor-challenges.md)
- [R3: CUDA Interception](../../../../research/R3-cuda-interception.md)
- [R7: Non-Proprietary GPU DMA](../../../../research/R7-non-proprietary-gpu-dma.md)
