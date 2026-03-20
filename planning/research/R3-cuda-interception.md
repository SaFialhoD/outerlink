# R3: CUDA Interception Strategies

**Created:** 2026-03-19
**Last Updated:** 2026-03-19
**Status:** Complete
**Priority:** HIGH

## Purpose

Determine how OutterLink will intercept CUDA operations to transparently redirect them to remote GPUs.

## TL;DR - VERDICT: Driver API + LD_PRELOAD + cuGetProcAddress Hooking

The proven approach used by all serious projects:
1. Intercept at the **CUDA Driver API** level (`libcuda.so`) - catches everything
2. Use **LD_PRELOAD** with custom `dlsym()` override as entry point
3. Hook **`cuGetProcAddress`** for CUDA 11.3+ compatibility
4. Follow **HAMi-core's architecture** (222 hooked functions, version-indexed table)
5. Also intercept **NVML** (`libnvidia-ml.so`) to fake GPU properties

---

## CUDA Architecture Layers

```
Application Code
    |
    v
CUDA Runtime API (libcudart.so)  -- High-level, "cuda" prefix (cudaMalloc)
    |                                Implicit context, auto module loading
    v
CUDA Driver API (libcuda.so)     -- Low-level, "cu" prefix (cuMemAlloc)
    |                                Explicit context, manual module mgmt
    v
GPU Hardware (via NVIDIA kernel driver)

NVML (libnvidia-ml.so)           -- Management/monitoring (nvidia-smi)
                                    Separate from both APIs
```

**Key fact:** Runtime API is a thin wrapper over Driver API. Every `cuda*` call becomes one or more `cu*` calls. Intercepting at Driver level catches EVERYTHING.

Since CUDA 11.3, the Runtime resolves Driver functions via `cuGetProcAddress` rather than direct symbol linking. This changes the interception game.

## Why Driver API (Not Runtime API)

| Factor | Runtime API | Driver API |
|--------|-------------|------------|
| Catches all apps | No (static linking bypasses) | **Yes** (libcuda.so always dynamic) |
| API stability | Changes more often | More stable ABI |
| Context control | Implicit (hidden) | Explicit (we control it) |
| Module visibility | Hidden loading | See actual PTX/cubin |
| Industry consensus | rCUDA (older) | HAMi-core, Cricket, FCSP (modern) |

## Interception Mechanism: How It Works

### Step 1: LD_PRELOAD Entry

```bash
LD_PRELOAD=/path/to/outterlink.so ./cuda_application
```

Our library loads before `libcuda.so`, intercepting all symbol lookups.

### Step 2: Override dlsym()

```c
// Pseudocode
void* dlsym(void* handle, const char* symbol) {
    if (strcmp(symbol, "cuMemAlloc_v2") == 0)
        return (void*)our_cuMemAlloc_v2;
    if (strcmp(symbol, "cuLaunchKernel") == 0)
        return (void*)our_cuLaunchKernel;
    // ... 222 functions
    return real_dlsym(handle, symbol);
}
```

### Step 3: Hook cuGetProcAddress (CUDA 11.3+)

```c
CUresult cuGetProcAddress(const char* symbol, void** pfn,
                          int cudaVersion, cuuint64_t flags, ...) {
    CUresult result = real_cuGetProcAddress(symbol, pfn, cudaVersion, flags, ...);
    void* hook = find_hook_for_symbol(symbol);
    if (hook) *pfn = hook;  // Replace with our function
    return result;
}
```

### Step 4: Forward to Remote GPU

Each hooked function: serialize arguments -> send over network -> receive result -> return to app.

## Functions We Need to Intercept

### Must Have (Core - Phase 1)

| Category | Functions | Complexity |
|----------|-----------|-----------|
| **Init** | `cuInit`, `cuDriverGetVersion` | Low |
| **Device Query** | `cuDeviceGet`, `cuDeviceGetCount`, `cuDeviceGetName`, `cuDeviceGetAttribute`, `cuDeviceTotalMem` | Low - return fake values |
| **Context** | `cuCtxCreate`, `cuCtxDestroy`, `cuCtxSetCurrent`, `cuCtxGetCurrent`, `cuCtxSynchronize` | Medium - handle translation |
| **Memory Alloc** | `cuMemAlloc_v2`, `cuMemFree_v2`, `cuMemAllocHost_v2`, `cuMemFreeHost` | High - address mapping |
| **Memory Transfer** | `cuMemcpyHtoD_v2`, `cuMemcpyDtoH_v2`, `cuMemcpyDtoD_v2`, all async variants | **Very High** - bulk network |
| **Module** | `cuModuleLoad`, `cuModuleLoadData`, `cuModuleGetFunction` | High - binary forwarding |
| **Kernel Launch** | `cuLaunchKernel` | **Very High** - serialize + dispatch |

### Must Have (Streams/Events - Phase 2)

| Category | Functions | Complexity |
|----------|-----------|-----------|
| **Streams** | `cuStreamCreate`, `cuStreamDestroy`, `cuStreamSynchronize`, `cuStreamWaitEvent` | High - ordering semantics |
| **Events** | `cuEventCreate`, `cuEventRecord`, `cuEventSynchronize`, `cuEventElapsedTime` | High - timing semantics |

### Nice to Have (Phase 3+)

| Category | Functions | Complexity |
|----------|-----------|-----------|
| **Unified Memory** | `cuMemAllocManaged`, `cuMemPrefetchAsync` | **Very High** - page migration |
| **Graphs** | `cuGraphCreate`, `cuGraphLaunch` | **Very High** - DAG serialization |
| **Texture/Surface** | `cuTexObjectCreate`, `cuSurfObjectCreate` | High |

### Also Required: NVML Interception

PyTorch, TensorFlow, and nvidia-smi all query NVML for GPU info. We must hook `libnvidia-ml.so` to report remote GPU properties as local.

## Handle Translation

Every opaque CUDA handle needs a bidirectional map (local synthetic <-> remote real):

| Handle Type | Challenge |
|-------------|-----------|
| `CUdeviceptr` | 64-bit device pointers - must create synthetic local values mapping to real remote pointers |
| `CUstream` | Must preserve ordering semantics across network |
| `CUevent` | Must return accurate remote timing |
| `CUmodule` | Binary data forwarded to server for remote loading |
| `CUfunction` | Module-relative, server returns its own handles after loading |
| `CUcontext` | 1:1 mapping, per-process |

## Key Challenges

### 1. Network Latency on Every Call
Every `cuMemcpy` = network transfer. Every `cuLaunchKernel` = round-trip.
**Mitigation:** Batch multiple calls without side effects into single network messages (vCUDA's "lazy updates" pattern).

### 2. Kernel Argument Serialization
`cuLaunchKernel` takes `void**` args - need to know types and sizes from module metadata.

### 3. Multi-threaded Context
CUDA contexts have per-thread current state. Thread-safe handle translation required.

### 4. CUDA Version Compatibility
Functions have versioned variants (`_v2`, `_v3`). `cuGetProcAddress` signature itself changed between versions. HAMi-core's version-indexed function table is the robust solution.

### 5. Unified Memory
Page fault handling is internal to NVIDIA driver - extremely hard to virtualize. **Recommendation: don't support in v1.**

### 6. CUDA Graphs
`cuGraphLaunch` bypasses `cuLaunchKernel` hooks. Known gap even in HAMi-core. **Recommendation: Phase 3+.**

## What Existing Projects Do

| Project | Layer | Network | CUDA Version | Open Source | Status |
|---------|-------|---------|--------------|-------------|--------|
| **rCUDA** | Runtime API | TCP, InfiniBand | Modern | No | Research, active |
| **Cricket** | Driver API | Yes | Modern | Yes (GPLv3) | Active |
| **HAMi-core** | Driver API + NVML | N/A (local only) | 10.x-12.x | Yes | Active, production |
| **GVirtuS** | Runtime API | TCP, VMSocket | 12.6+ | Yes | Active |
| **vCUDA** | Runtime API | Shared memory | 1.1 | No | Dead |
| **AVEC** | API interception | TCP/IP | Modern | Unknown | Research |

### Key Takeaways from Existing Projects

1. **HAMi-core** = gold standard for interception mechanics (but no networking)
2. **Cricket** = most relevant to us (Driver API + remote + open source)
3. **rCUDA** = most mature for remote GPU (but closed source, Runtime API)
4. **Lesson from vCUDA:** batching calls ("lazy updates") is a huge win

## NVIDIA Official Mechanisms - None Help Us

| Mechanism | What It Does | Why It Doesn't Help |
|-----------|-------------|-------------------|
| CUDA IPC | Share memory between processes on same machine | Same machine only |
| CUDA MPS | Multi-process GPU sharing | Single node only |
| CUDA VMM | Virtual memory management | Multi-node requires NVLink/NVSwitch |
| GPUDirect RDMA | Direct GPU memory network access | Requires InfiniBand/ConnectX |

**NVIDIA provides NO official remote GPU API for standard networking.** API interception is the only path.

## Recommended Architecture for OutterLink

```
Application
    |
    | LD_PRELOAD
    v
OutterLink Client Library (liboutterlink.so)
    |-- dlsym override (catches all symbol lookups)
    |-- cuGetProcAddress hook (catches CUDA 11.3+ dynamic resolution)
    |-- Handle translation tables
    |-- Call batching/coalescing
    |-- Serialization layer
    |
    | TCP / io_uring (upgradable to RDMA)
    v
OutterLink Server Daemon (per GPU node)
    |-- Deserialize calls
    |-- Execute real CUDA Driver API calls
    |-- Return results
    |-- Manage GPU contexts per client
```

## Implementation Priority Order

1. Device queries and initialization (fake local GPU)
2. Memory allocation (handle translation)
3. Memory transfers (the bandwidth test)
4. Module loading (forward PTX/cubin)
5. Kernel launch (the hard one)
6. Streams and events
7. NVML hooks
8. Unified memory (maybe never)
9. Graphs (maybe never)

## Related Documents

- [Project Vision](../../docs/architecture/00-project-vision.md)
- [R2: SoftRoCE Research](R2-softroce-rdma.md)
- [Pre-Planning Master](../pre-planning/00-master-preplan.md)

## Sources

1. HAMi-core Function Hooking System (DeepWiki) - interception architecture
2. HAMi-core Design Documentation - 222 function hooks
3. CUinterwarp GitHub - Driver API interception example
4. cudahook GitHub - LD_PRELOAD CUDA example
5. rCUDA Wikipedia + ScienceDirect paper
6. Cricket GitHub + Wiley paper - open source remote GPU
7. NVIDIA Driver vs Runtime API docs
8. NVIDIA CUDA Driver API Entry Point Access docs
9. NVIDIA Forums: CUDA 11.4 hooking discussion
10. cuda_hook GitHub - cuGetProcAddress issues

## Open Questions

- [ ] Should we study Cricket's codebase as a starting point?
- [ ] How does NVIDIA's confidential computing mode affect LD_PRELOAD?
- [ ] Can `cudaMallocAsync` (stream-ordered pool allocator) be virtualized?
- [ ] What's the real-world call pattern of ML training? (for batching strategy)
