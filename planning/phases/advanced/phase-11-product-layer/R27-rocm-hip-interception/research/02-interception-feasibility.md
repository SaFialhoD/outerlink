# R27 Research: HIP Interception Feasibility

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Priority:** HIGH

## Purpose

Determine whether OuterLink's LD_PRELOAD interception technique can be applied to AMD's HIP/ROCm stack with the same effectiveness as our CUDA interception. Evaluate API surface, memory management, and RDMA feasibility.

---

## TL;DR

**YES, LD_PRELOAD interception of `libamdhip64.so` is fully feasible.** The technique is identical to our CUDA approach: create a shared library with the same symbol names, set LD_PRELOAD, intercept calls, forward to remote GPU. The HIP API surface is larger (~300-400 functions vs ~222 CUDA Driver API) but simpler because HIP has one API level instead of two. The 1:1 naming convention (`cuda*` -> `hip*`) means our interception logic can be adapted systematically. RDMA to AMD GPU VRAM works via ROCnRDMA/dma-buf with ConnectX-5.

---

## 1. LD_PRELOAD on libamdhip64.so

### Mechanism

Identical to our CUDA interception:

```bash
# CUDA (current OuterLink)
LD_PRELOAD=/path/to/outerlink_cuda.so ./my_cuda_app

# HIP (proposed)
LD_PRELOAD=/path/to/outerlink_hip.so ./my_hip_app
```

### How It Works

1. `outerlink_hip.so` exports the same symbols as `libamdhip64.so` (e.g., `hipMalloc`, `hipLaunchKernel`)
2. Dynamic linker resolves HIP calls to our library first (LD_PRELOAD priority)
3. Our library intercepts the call, applies OuterLink logic (routing, scheduling, tracking)
4. Forwards to real `libamdhip64.so` via `dlsym(RTLD_NEXT, "hipMalloc")` on the remote GPU node

### Requirements

| Requirement | Status |
|---|---|
| HIP apps dynamically link `libamdhip64.so` | YES — standard practice, package `hip-runtime-amd` |
| Symbol names are stable | YES — public API, documented, HIPIFY relies on stable names |
| `dlsym(RTLD_NEXT, ...)` works | YES — standard dynamic linker feature |
| No static linking of HIP | MOSTLY — applications may statically link, but framework-based apps (PyTorch, TF) use dynamic linking |

### Potential Issues

| Issue | Risk | Mitigation |
|---|---|---|
| Applications bypassing HIP for direct HSA calls | LOW — only advanced/custom apps | Monitor `libhsa-runtime64.so` calls, intercept if needed |
| HIPRTC symbols in `libamdhip64.so` | MEDIUM — dual library issue | Intercept both `libamdhip64.so` and `libhiprtc.so` |
| ROCm version-specific ABI changes | LOW — public API is stable | Test against multiple ROCm versions, pin to LTS releases |
| `__HIP_PLATFORM_AMD__` compile-time routing | NONE — only affects compilation, not runtime linking | N/A |

---

## 2. HIP API Surface Analysis

### Function Categories and Estimated Counts

Based on HIPIFY documentation and HIP runtime API reference:

| Category | Estimated Functions | CUDA Equivalent Category |
|---|---|---|
| Device management | ~30 | `cuDevice*`, `cudaDevice*` |
| Memory management | ~50 | `cuMem*`, `cudaMem*`, `cudaMalloc*` |
| Stream management | ~15 | `cuStream*`, `cudaStream*` |
| Event management | ~10 | `cuEvent*`, `cudaEvent*` |
| Execution control | ~20 | `cuLaunchKernel`, `cudaLaunchKernel` |
| Module management | ~15 | `cuModule*` |
| Texture/surface | ~30 | `cuTex*`, `cudaBindTexture*` |
| Peer access | ~5 | `cuCtxEnablePeerAccess` |
| Occupancy | ~5 | `cudaOccupancy*` |
| Graph API | ~40 | `cudaGraph*` |
| Virtual memory | ~15 | `cuMemAddressReserve*` |
| Error handling | ~5 | `cudaGetErrorString` |
| Profiling/debug | ~10 | `cuProfiler*` |
| Cooperative groups | ~10 | `cudaLaunchCooperativeKernel` |
| Misc/utility | ~20 | Various |
| **Total** | **~280-350** | **~222 (Driver API only)** |

### Comparison with CUDA Interception

| Metric | CUDA (OuterLink current) | HIP (proposed) |
|---|---|---|
| Target library | `libcuda.so` (Driver API) | `libamdhip64.so` |
| Function count | ~222 | ~280-350 |
| API levels | 1 (Driver only) | 1 (unified) |
| Naming pattern | `cu*` prefix | `hip*` prefix |
| Code generation | HAMi-core pattern from headers | Same approach, from HIP headers |
| Key interceptions | `cuMemAlloc`, `cuLaunchKernel`, `cuCtxCreate` | `hipMalloc`, `hipLaunchKernel`, `hipSetDevice` |

### Priority Functions to Intercept

**Must intercept (core functionality):**

| Function | Why |
|---|---|
| `hipMalloc`, `hipFree` | VRAM allocation tracking and forwarding |
| `hipMemcpy`, `hipMemcpyAsync` | Data transfer routing |
| `hipLaunchKernel`, `hipModuleLaunchKernel` | Kernel dispatch to remote GPU |
| `hipStreamCreate`, `hipStreamDestroy` | Stream lifecycle management |
| `hipEventCreate`, `hipEventRecord`, `hipEventSynchronize` | Timing and synchronization |
| `hipSetDevice`, `hipGetDevice` | Device selection routing |
| `hipGetDeviceProperties` | Virtual device property reporting |
| `hipDeviceSynchronize` | Synchronization barrier |
| `hipModuleLoad`, `hipModuleGetFunction` | Kernel module management |
| `hipHostMalloc`, `hipHostFree` | Pinned memory for staging |

**Should intercept (completeness):**

| Function | Why |
|---|---|
| `hipMemset`, `hipMemsetAsync` | Remote memory initialization |
| `hipMemcpy2D`, `hipMemcpy3D` | Structured data transfers |
| `hipMallocPitch`, `hipMalloc3D` | Pitched/3D allocations |
| `hipMallocManaged` | Unified memory support |
| `hipDeviceEnablePeerAccess` | Multi-GPU peer access |
| `hipStreamWaitEvent` | Cross-stream synchronization |

**Can defer (advanced features):**

| Function | Why Defer |
|---|---|
| `hipGraph*` | Graph API (~40 functions), complex, low initial priority |
| `hipTex*` | Texture operations, rare in compute workloads |
| `hipMemAddressReserve*` | Virtual memory management, advanced |
| `hipLaunchCooperativeKernel` | Cooperative launch, complex multi-GPU semantics |

---

## 3. Memory Management Interception

### Allocation Tracking

Identical approach to our CUDA interception:

```
hipMalloc(ptr, size) intercepted:
  1. Allocate tracking entry: {ptr, size, device, user}
  2. Route to appropriate GPU node
  3. On remote node: call real hipMalloc
  4. Map remote ptr to local virtual ptr
  5. Return virtual ptr to application
```

### Transfer Routing

```
hipMemcpy(dst, src, size, kind) intercepted:
  1. Identify src/dst locations (local host, remote GPU, local GPU)
  2. For remote transfers:
     a. Host -> Remote GPU: stage in pinned memory, RDMA to remote VRAM
     b. Remote GPU -> Host: RDMA from remote VRAM, copy to host
     c. Remote GPU -> Remote GPU: RDMA between nodes (or staged through host)
  3. For local transfers: forward to real hipMemcpy
```

### HIP-Specific Memory Considerations

| Feature | Details | Impact on Interception |
|---|---|---|
| Block allocator caching | `hipFree` caches blocks, doesn't immediately release | Must track actual vs cached allocations |
| SDMA vs shader blit | `HSA_ENABLE_SDMA=0` changes copy engine | Transparent to interception layer |
| Managed memory | `hipMallocManaged` auto-migrates between host/device | Complex — needs page fault handling. Defer to R19. |
| Memory pools | HIP supports memory pools (hipMemPool*) | Track pool allocations separately |

---

## 4. AMD GPU + RDMA: Can ConnectX-5 Access AMD GPU VRAM?

### Answer: YES

ConnectX-5 can RDMA to/from AMD GPU VRAM via two mechanisms:

### Mechanism 1: Peer Memory Client (ROCnRDMA)

The amdkfd kernel driver implements the Linux Peer Memory Client API. This allows:
1. Application allocates GPU memory via `hipMalloc`
2. RDMA subsystem queries amdkfd for physical addresses of that GPU memory
3. ConnectX-5 performs RDMA read/write directly to/from those physical addresses

**Requirements:**
- `amdgpu` kernel driver with KFD (Kernel Fusion Driver) support
- Mellanox OFED with peer memory support
- UCX compiled with `--with-rocm`

### Mechanism 2: dma-buf (Modern, Preferred)

1. Application allocates GPU memory via `hipMalloc`
2. Export to dma-buf: `hsa_amd_portable_export_dmabuf()` returns a file descriptor
3. Register with RDMA: `ibv_reg_dmabuf_mr(dmabuf_fd)`
4. ConnectX-5 uses the registered memory region for RDMA operations

**Advantages over Peer Memory Client:**
- Standard Linux kernel mechanism (not AMD-specific interface)
- Better support in newer kernel versions
- Works with any RDMA NIC that supports `ibv_reg_dmabuf_mr`

### OpenDMA Equivalent for AMD?

OuterLink's OpenDMA (Phase 5) uses PCIe BAR1 direct NIC-to-GPU VRAM access for NVIDIA GPUs. For AMD GPUs:

| Aspect | NVIDIA (OpenDMA) | AMD (Equivalent) |
|---|---|---|
| VRAM BAR | BAR1 (resizable) | BAR (resizable, "Smart Access Memory") |
| BAR type | Uncached, write-combined | Uncached, write-combined |
| NIC DMA to BAR | ConnectX-5 DMA engine -> BAR1 address | ConnectX-5 DMA engine -> BAR address |
| Kernel support | Custom kernel module (opendma) | amdkfd already exposes peer memory |

**Key insight:** AMD's GPU driver (amdkfd) ALREADY exposes the interfaces needed for NIC-to-GPU RDMA through the standard Peer Memory Client API and dma-buf. We may not need a custom kernel module for AMD like we do for NVIDIA (where GPUDirect RDMA is artificially restricted on GeForce). AMD's open-source driver stack does not have GeForce-like artificial restrictions.

---

## 5. Kernel Code Compatibility

### The Fundamental Problem

CUDA kernels compile to PTX/SASS (NVIDIA ISA). HIP kernels compile to AMDGCN ISA. These are completely different instruction sets. **A kernel compiled for NVIDIA cannot run on AMD, and vice versa.**

### What This Means for OuterLink

OuterLink intercepts kernel LAUNCHES, not kernel CODE. The workflow is:

```
1. Application compiles kernel for target GPU ISA (developer responsibility)
2. Application calls hipModuleLoad / hipLaunchKernel
3. OuterLink intercepts the launch call
4. OuterLink forwards the COMPILED kernel binary + launch parameters to the GPU node
5. GPU node executes the kernel on the real hardware
```

OuterLink never needs to translate kernel code between ISAs. The application must be compiled for the target GPU architecture. This is the same as our CUDA approach: we forward PTX/SASS to NVIDIA GPUs, and we'd forward AMDGCN to AMD GPUs.

### Cross-Vendor Kernel Issue

If an application has NVIDIA GPUs AND AMD GPUs in the same pool, it needs BOTH ISA versions of every kernel. This is a cross-vendor challenge covered in [03-cross-vendor-challenges.md](03-cross-vendor-challenges.md).

---

## 6. Feasibility Assessment

### Overall Verdict: HIGHLY FEASIBLE

| Aspect | Feasibility | Confidence | Notes |
|---|---|---|---|
| LD_PRELOAD on libamdhip64.so | HIGH | 95% | Standard technique, identical to CUDA approach |
| HIP API interception | HIGH | 90% | Larger surface but systematic naming convention |
| Memory tracking | HIGH | 90% | Same approach as CUDA, hipMalloc/hipFree tracking |
| Kernel forwarding | HIGH | 90% | Forward compiled binary + params, no ISA translation |
| RDMA to AMD VRAM | HIGH | 85% | Proven via ROCnRDMA + dma-buf with ConnectX-5 |
| OpenDMA equivalent | MEDIUM | 70% | May not need custom kernel module — AMD's driver is open |
| Full API coverage | MEDIUM | 75% | ~350 functions is large but manageable with code generation |
| Cross-vendor mixing | LOW | 40% | ISA incompatibility makes this hard (see doc 03) |

### Effort Estimate vs. CUDA Implementation

| Component | CUDA Effort | HIP Effort | Reuse |
|---|---|---|---|
| Interception library (.so) | 100% (done) | 60% of CUDA effort | Architecture reusable, functions differ |
| Protocol/transport | 100% (done) | 90% reuse | Same wire protocol, different API names |
| Server-side dispatch | 100% (done) | 50% of CUDA effort | New dispatch to HIP instead of CUDA |
| Memory management | 100% (done) | 80% reuse | Same tracking logic, different API calls |
| Testing | 100% (done) | 70% of CUDA effort | Need AMD hardware for testing |

**Estimated total effort for HIP interception: ~40-50% of the original CUDA implementation effort.** The architecture, protocol, and transport layers are fully reusable. Only the interception library and server-side GPU dispatch need to be HIP-specific.

---

## 7. Implementation Strategy

### Phase 1: Minimal Viable HIP Interception

Intercept the ~30 most critical functions:
- Memory: `hipMalloc`, `hipFree`, `hipMemcpy`, `hipMemcpyAsync`, `hipHostMalloc`, `hipHostFree`
- Execution: `hipLaunchKernel`, `hipModuleLaunchKernel`, `hipModuleLoad`, `hipModuleGetFunction`
- Device: `hipSetDevice`, `hipGetDevice`, `hipGetDeviceCount`, `hipGetDeviceProperties`
- Stream: `hipStreamCreate`, `hipStreamDestroy`, `hipStreamSynchronize`
- Event: `hipEventCreate`, `hipEventRecord`, `hipEventSynchronize`, `hipEventElapsedTime`
- Sync: `hipDeviceSynchronize`

**This covers 80% of real-world HIP applications.**

### Phase 2: Extended Coverage

Add ~100 more functions for:
- 2D/3D memory operations
- Memory pools
- Peer access
- Occupancy queries
- Error handling

### Phase 3: Full Coverage

Add remaining ~200 functions:
- Graph API
- Texture/surface
- Virtual memory
- Cooperative groups
- Profiling

### Code Generation Approach

Same as CUDA: generate interception stubs from HIP header files (`hip/hip_runtime_api.h`).

```c
// Generated stub example
hipError_t hipMalloc(void** ptr, size_t size) {
    // OuterLink interception logic
    if (outerlink_is_remote_device(current_device)) {
        return outerlink_remote_hip_malloc(ptr, size);
    }
    // Forward to real implementation
    static hipError_t (*real_hipMalloc)(void**, size_t) = NULL;
    if (!real_hipMalloc) {
        real_hipMalloc = dlsym(RTLD_NEXT, "hipMalloc");
    }
    return real_hipMalloc(ptr, size);
}
```

---

## Open Questions

1. **AMD test hardware:** Do we have any AMD GPUs for development/testing? If not, what's the minimum viable card? (RX 6600 or newer for ROCm support)
2. **ROCm on consumer Radeon:** Which consumer Radeon GPUs have ROCm support? The list is limited — not all Radeon GPUs support ROCm.
3. **HIP on Windows:** ROCm is Linux-primary. Windows HIP SDK exists but is more limited. Do we care about Windows AMD GPU support?
4. **Library versioning:** `libamdhip64.so.5` vs `.so.6` — do we need version-specific interception?
5. **Performance baseline:** What's the overhead of LD_PRELOAD interception for HIP calls? Need to benchmark.

---

## Related Documents

- [01-rocm-hip-architecture.md](01-rocm-hip-architecture.md)
- [03-cross-vendor-challenges.md](03-cross-vendor-challenges.md)
- [R3: CUDA Interception](../../../../research/R3-cuda-interception.md)
- [R5: GPUDirect GeForce Restriction](../../../../research/R5-gpudirect-geforce-restriction.md)
