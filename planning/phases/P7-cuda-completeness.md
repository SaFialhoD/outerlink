# P7: CUDA Completeness -- Full API Coverage for Real-World Applications

**Created:** 2026-03-19
**Last Updated:** 2026-03-19
**Status:** Plan
**Priority:** HIGH - Phase 3 Implementation
**Depends On:** P5 (PoC), P6 (Core Transport)

---

## Goal

Implement comprehensive CUDA Driver API and NVML interception so that real-world CUDA applications (PyTorch, TensorFlow, llama.cpp, etc.) run transparently on remote GPUs without modification. After P7, `torch.cuda.is_available()` returns True, `model.to('cuda')` works, and inference runs end-to-end.

## Milestone

- PyTorch `import torch; torch.cuda.is_available()` returns `True` with remote GPUs
- PyTorch `model.to('cuda:N')` where N is a remote GPU succeeds
- PyTorch simple inference (e.g., ResNet forward pass) completes correctly
- `nvidia-smi` (via NVML interception) shows remote GPUs as local
- CUDA streams and events work correctly over the network
- All versioned function variants (_v2, _v3) handled via cuGetProcAddress
- 150+ CUDA Driver API functions intercepted (covering all MUST HAVE and SHOULD HAVE)
- 25+ NVML functions intercepted

## Prerequisites

- [x] R3: CUDA interception strategy decided (Driver API + LD_PRELOAD + cuGetProcAddress)
- [ ] P5: PoC complete (device query + memory alloc over TCP)
- [ ] P6: Core Transport complete (memory transfers + kernel launch working)
- [ ] D6: Target CUDA version decided (12.x minimum)
- [ ] D14: Handle translation approach decided

---

## Section 1: Complete CUDA Driver API Function Registry

This is the master checklist of every CUDA Driver API function we must consider. Functions are organized by category and tagged with:

- **Phase**: When to implement (PoC / Core / Completeness / Future)
- **Priority**: MUST HAVE (apps break without it), SHOULD HAVE (common frameworks need it), NICE TO HAVE (rare usage)
- **Complexity**: Low / Medium / High / Very High

### 1.1 Initialization (2 functions)

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 1 | `cuInit` | PoC | MUST HAVE | Low | Entry point. Must succeed for anything to work. |
| 2 | `cuDriverGetVersion` | PoC | MUST HAVE | Low | Return version of remote GPU's driver. |

### 1.2 Device Management (14 functions)

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 3 | `cuDeviceGet` | PoC | MUST HAVE | Low | Maps device index to handle. Must handle local+remote numbering. |
| 4 | `cuDeviceGetCount` | PoC | MUST HAVE | Low | Return local + remote GPU count. |
| 5 | `cuDeviceGetName` | PoC | MUST HAVE | Low | Return remote GPU name (e.g., "NVIDIA GeForce RTX 3090 Ti"). |
| 6 | `cuDeviceGetAttribute` | PoC | MUST HAVE | Medium | ~100 attributes. Cache on client after first query. |
| 7 | `cuDeviceTotalMem_v2` | PoC | MUST HAVE | Low | Return remote GPU total memory. |
| 8 | `cuDeviceGetUuid_v2` | Completeness | SHOULD HAVE | Low | PyTorch uses this for device identification. |
| 9 | `cuDeviceGetLuid` | Completeness | NICE TO HAVE | Low | Windows only, not relevant for Linux. |
| 10 | `cuDeviceGetByPCIBusId` | Completeness | SHOULD HAVE | Medium | PyTorch sends this during init. Synthesize PCI bus IDs for remote GPUs. |
| 11 | `cuDeviceGetPCIBusId` | Completeness | SHOULD HAVE | Medium | Inverse of above. Return synthetic PCI bus IDs. |
| 12 | `cuDeviceGetDefaultMemPool` | Completeness | SHOULD HAVE | Medium | Stream-ordered alloc support. |
| 13 | `cuDeviceGetMemPool` | Completeness | SHOULD HAVE | Medium | Stream-ordered alloc support. |
| 14 | `cuDeviceSetMemPool` | Completeness | SHOULD HAVE | Medium | Stream-ordered alloc support. |
| 15 | `cuDeviceGetTexture1DLinearMaxWidth` | Future | NICE TO HAVE | Low | Rarely queried. |
| 16 | `cuDeviceGetExecAffinitySupport` | Future | NICE TO HAVE | Low | Rarely used. |

### 1.3 Primary Context Management (5 functions)

PyTorch and most CUDA Runtime API users rely on primary contexts rather than explicit cuCtxCreate.

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 17 | `cuDevicePrimaryCtxRetain` | Core | MUST HAVE | High | PyTorch's primary entry point for context. Creates remote context on first call. |
| 18 | `cuDevicePrimaryCtxRelease_v2` | Core | MUST HAVE | Medium | Releases primary context reference. |
| 19 | `cuDevicePrimaryCtxGetState` | Completeness | SHOULD HAVE | Low | Query context state. |
| 20 | `cuDevicePrimaryCtxSetFlags_v2` | Completeness | SHOULD HAVE | Low | Set context flags before retain. |
| 21 | `cuDevicePrimaryCtxReset_v2` | Completeness | SHOULD HAVE | Medium | Reset primary context. |

### 1.4 Context Management (12 functions)

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 22 | `cuCtxCreate_v2` | Core | MUST HAVE | High | Creates context on remote GPU. Handle translation required. |
| 23 | `cuCtxCreate_v3` | Completeness | SHOULD HAVE | High | CUDA 12+ variant with exec affinity. |
| 24 | `cuCtxDestroy_v2` | Core | MUST HAVE | Medium | Destroy remote context, clean up handle maps. |
| 25 | `cuCtxSetCurrent` | Core | MUST HAVE | Medium | Thread-local context tracking. |
| 26 | `cuCtxGetCurrent` | Core | MUST HAVE | Low | Return current thread's context. Local only, no network call. |
| 27 | `cuCtxPushCurrent_v2` | Core | MUST HAVE | Medium | Context stack management. |
| 28 | `cuCtxPopCurrent_v2` | Core | MUST HAVE | Medium | Context stack management. |
| 29 | `cuCtxSynchronize` | Core | MUST HAVE | Medium | Flush all remote work, block until complete. |
| 30 | `cuCtxGetDevice` | Completeness | SHOULD HAVE | Low | Return device for current context. Local lookup. |
| 31 | `cuCtxGetFlags` | Completeness | NICE TO HAVE | Low | Return context flags. |
| 32 | `cuCtxGetApiVersion` | Completeness | SHOULD HAVE | Low | Return API version. |
| 33 | `cuCtxGetLimit` | Completeness | SHOULD HAVE | Low | Query remote context limits (stack size, heap size, etc.). |
| 34 | `cuCtxSetLimit` | Completeness | SHOULD HAVE | Low | Set remote context limits. |
| 35 | `cuCtxGetStreamPriorityRange` | Completeness | SHOULD HAVE | Low | Query priority range for stream creation. |
| 36 | `cuCtxGetId` | Completeness | NICE TO HAVE | Low | CUDA 12+ context ID. |

### 1.5 Module Management (10 functions)

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 37 | `cuModuleLoad` | Core | MUST HAVE | High | Load .cubin/.ptx from file path on client. Must transfer binary to server. |
| 38 | `cuModuleLoadData` | Core | MUST HAVE | High | Load from in-memory image. Primary path for PyTorch JIT. |
| 39 | `cuModuleLoadDataEx` | Core | MUST HAVE | High | Load with JIT options (max registers, target arch, etc.). |
| 40 | `cuModuleLoadFatBinary` | Core | SHOULD HAVE | High | Load fat binary (multi-arch). |
| 41 | `cuModuleUnload` | Core | MUST HAVE | Medium | Unload module on server. Clean up function handle maps. |
| 42 | `cuModuleGetFunction` | Core | MUST HAVE | High | Get kernel function handle from loaded module. Handle translation. |
| 43 | `cuModuleGetGlobal_v2` | Completeness | SHOULD HAVE | Medium | Get global variable address. Needed for __constant__ memory. |
| 44 | `cuModuleGetTexRef` | Completeness | NICE TO HAVE | Medium | Deprecated but still used by some apps. |
| 45 | `cuModuleGetSurfRef` | Completeness | NICE TO HAVE | Medium | Deprecated but still used by some apps. |
| 46 | `cuModuleGetLoadingMode` | Completeness | NICE TO HAVE | Low | CUDA 12+ loading mode query. |

### 1.6 Library Management (CUDA 12+, 8 functions)

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 47 | `cuLibraryLoadData` | Future | SHOULD HAVE | High | CUDA 12 replacement for cuModuleLoadData. |
| 48 | `cuLibraryLoadFromFile` | Future | SHOULD HAVE | High | CUDA 12 replacement for cuModuleLoad. |
| 49 | `cuLibraryUnload` | Future | SHOULD HAVE | Medium | Unload library. |
| 50 | `cuLibraryGetKernel` | Future | SHOULD HAVE | Medium | Get kernel from library (replaces cuModuleGetFunction). |
| 51 | `cuLibraryGetModule` | Future | NICE TO HAVE | Low | Get module from library. |
| 52 | `cuLibraryGetGlobal` | Future | NICE TO HAVE | Medium | Get global from library. |
| 53 | `cuLibraryGetManaged` | Future | NICE TO HAVE | Medium | Get managed variable. |
| 54 | `cuKernelGetFunction` | Future | SHOULD HAVE | Low | Get CUfunction from CUkernel. |

### 1.7 Memory Management (52 functions)

#### 1.7.1 Allocation and Deallocation

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 55 | `cuMemAlloc_v2` | Core | MUST HAVE | High | Primary device alloc. Returns synthetic CUdeviceptr. |
| 56 | `cuMemAllocPitch_v2` | Completeness | SHOULD HAVE | High | 2D pitched allocation. Needed for image processing. |
| 57 | `cuMemFree_v2` | Core | MUST HAVE | Medium | Free remote device memory. Remove from handle map. |
| 58 | `cuMemGetInfo_v2` | Completeness | MUST HAVE | Low | Free/total memory. PyTorch queries this frequently. |
| 59 | `cuMemGetAddressRange_v2` | Completeness | SHOULD HAVE | Medium | Query base + size of allocation containing pointer. |
| 60 | `cuMemAllocHost_v2` | Core | MUST HAVE | Medium | Pinned host memory. Critical for transfer performance. |
| 61 | `cuMemFreeHost` | Core | MUST HAVE | Low | Free pinned host memory. |
| 62 | `cuMemHostAlloc` | Core | MUST HAVE | Medium | Pinned + mapped host alloc. |
| 63 | `cuMemHostGetDevicePointer_v2` | Completeness | SHOULD HAVE | Medium | Get device-accessible pointer for mapped host memory. |
| 64 | `cuMemHostGetFlags` | Completeness | NICE TO HAVE | Low | Query host alloc flags. |
| 65 | `cuMemHostRegister_v2` | Completeness | SHOULD HAVE | Medium | Register existing host memory as pinned. PyTorch uses this. |
| 66 | `cuMemHostUnregister` | Completeness | SHOULD HAVE | Low | Unregister pinned host memory. |
| 67 | `cuMemAllocManaged` | Future | NICE TO HAVE | Very High | Unified/managed memory. See Section 5. |

#### 1.7.2 Synchronous Memory Copy

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 68 | `cuMemcpyHtoD_v2` | Core | MUST HAVE | Very High | Host-to-device: bulk network transfer to remote GPU. |
| 69 | `cuMemcpyDtoH_v2` | Core | MUST HAVE | Very High | Device-to-host: bulk network transfer from remote GPU. |
| 70 | `cuMemcpyDtoD_v2` | Core | MUST HAVE | High | Device-to-device on same remote GPU. Forward to server. |
| 71 | `cuMemcpy` | Completeness | SHOULD HAVE | High | Unified addressing variant. Detect direction from pointers. |
| 72 | `cuMemcpyPeer` | Completeness | SHOULD HAVE | Very High | Cross-device copy. If both remote, server-to-server transfer. |
| 73 | `cuMemcpy2D_v2` | Completeness | SHOULD HAVE | High | 2D strided copy. |
| 74 | `cuMemcpy2DUnaligned_v2` | Completeness | NICE TO HAVE | High | Unaligned 2D copy. |
| 75 | `cuMemcpy3D_v2` | Completeness | NICE TO HAVE | High | 3D copy. |
| 76 | `cuMemcpy3DPeer` | Future | NICE TO HAVE | Very High | 3D cross-device copy. |

#### 1.7.3 Asynchronous Memory Copy

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 77 | `cuMemcpyHtoDAsync_v2` | Core | MUST HAVE | Very High | Async HtoD. Must be stream-ordered on remote GPU. |
| 78 | `cuMemcpyDtoHAsync_v2` | Core | MUST HAVE | Very High | Async DtoH. |
| 79 | `cuMemcpyDtoDAsync_v2` | Core | MUST HAVE | High | Async DtoD on same remote GPU. |
| 80 | `cuMemcpyAsync` | Completeness | SHOULD HAVE | High | Unified addressing async copy. |
| 81 | `cuMemcpyPeerAsync` | Completeness | SHOULD HAVE | Very High | Async cross-device copy. |
| 82 | `cuMemcpy2DAsync_v2` | Completeness | SHOULD HAVE | High | Async 2D copy. |
| 83 | `cuMemcpy3DAsync_v2` | Completeness | NICE TO HAVE | High | Async 3D copy. |
| 84 | `cuMemcpy3DPeerAsync` | Future | NICE TO HAVE | Very High | Async 3D peer copy. |

#### 1.7.4 Memory Set

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 85 | `cuMemsetD8_v2` | Core | MUST HAVE | Medium | memset 8-bit on remote. |
| 86 | `cuMemsetD16_v2` | Core | SHOULD HAVE | Medium | memset 16-bit. |
| 87 | `cuMemsetD32_v2` | Core | MUST HAVE | Medium | memset 32-bit. Primary memset used by frameworks. |
| 88 | `cuMemsetD8Async` | Completeness | SHOULD HAVE | Medium | Async memset 8-bit. |
| 89 | `cuMemsetD16Async` | Completeness | SHOULD HAVE | Medium | Async memset 16-bit. |
| 90 | `cuMemsetD32Async` | Completeness | MUST HAVE | Medium | Async memset 32-bit. PyTorch uses this heavily. |
| 91 | `cuMemsetD2D8_v2` | Completeness | NICE TO HAVE | Medium | 2D memset variants. |
| 92 | `cuMemsetD2D16_v2` | Completeness | NICE TO HAVE | Medium | |
| 93 | `cuMemsetD2D32_v2` | Completeness | NICE TO HAVE | Medium | |
| 94 | `cuMemsetD2D8Async` | Future | NICE TO HAVE | Medium | |
| 95 | `cuMemsetD2D16Async` | Future | NICE TO HAVE | Medium | |
| 96 | `cuMemsetD2D32Async` | Future | NICE TO HAVE | Medium | |

#### 1.7.5 CUDA Array Functions

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 97 | `cuArrayCreate_v2` | Completeness | SHOULD HAVE | High | Needed for texture/surface operations. |
| 98 | `cuArray3DCreate_v2` | Completeness | SHOULD HAVE | High | 3D array creation. |
| 99 | `cuArrayDestroy` | Completeness | SHOULD HAVE | Medium | Array cleanup. |
| 100 | `cuArrayGetDescriptor_v2` | Completeness | NICE TO HAVE | Low | Query array properties. |
| 101 | `cuArray3DGetDescriptor_v2` | Completeness | NICE TO HAVE | Low | Query 3D array properties. |
| 102 | `cuArrayGetPlane` | Future | NICE TO HAVE | Medium | Get plane of multiplanar array. |
| 103 | `cuArrayGetMemoryRequirements` | Future | NICE TO HAVE | Low | Query memory requirements. |
| 104 | `cuArrayGetSparseProperties` | Future | NICE TO HAVE | Low | Sparse texture support. |

#### 1.7.6 IPC Memory

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 105 | `cuIpcGetMemHandle` | Future | NICE TO HAVE | High | Not meaningful for remote GPUs. Return error or fake handle. |
| 106 | `cuIpcOpenMemHandle_v2` | Future | NICE TO HAVE | High | Same. |
| 107 | `cuIpcCloseMemHandle` | Future | NICE TO HAVE | Low | Same. |
| 108 | `cuIpcGetEventHandle` | Future | NICE TO HAVE | High | IPC event sharing. |
| 109 | `cuIpcOpenEventHandle` | Future | NICE TO HAVE | High | IPC event sharing. |

### 1.8 Stream-Ordered Memory Allocator (18 functions)

PyTorch 2.x uses stream-ordered allocation heavily via its caching allocator.

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 110 | `cuMemAllocAsync` | Completeness | MUST HAVE | High | Primary async alloc path for PyTorch 2.x. |
| 111 | `cuMemFreeAsync` | Completeness | MUST HAVE | High | Async free. |
| 112 | `cuMemAllocFromPoolAsync` | Completeness | SHOULD HAVE | High | Pool-based async alloc. |
| 113 | `cuMemPoolCreate` | Completeness | SHOULD HAVE | High | Create memory pool on remote GPU. |
| 114 | `cuMemPoolDestroy` | Completeness | SHOULD HAVE | Medium | Destroy remote pool. |
| 115 | `cuMemPoolGetAccess` | Completeness | NICE TO HAVE | Medium | Pool access query. |
| 116 | `cuMemPoolSetAccess` | Completeness | NICE TO HAVE | Medium | Pool access config. |
| 117 | `cuMemPoolGetAttribute` | Completeness | SHOULD HAVE | Low | Pool attribute query. |
| 118 | `cuMemPoolSetAttribute` | Completeness | SHOULD HAVE | Low | Pool attribute config. |
| 119 | `cuMemPoolTrimTo` | Completeness | SHOULD HAVE | Medium | Trim pool memory. |
| 120 | `cuMemPoolExportPointer` | Future | NICE TO HAVE | High | IPC pool pointer export. |
| 121 | `cuMemPoolExportToShareableHandle` | Future | NICE TO HAVE | High | IPC pool handle export. |
| 122 | `cuMemPoolImportFromShareableHandle` | Future | NICE TO HAVE | High | IPC pool handle import. |
| 123 | `cuMemPoolImportPointer` | Future | NICE TO HAVE | High | IPC pool pointer import. |

### 1.9 Virtual Memory Management (VMM) (12 functions)

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 124 | `cuMemAddressReserve` | Future | SHOULD HAVE | Very High | Reserve VA range on remote GPU. |
| 125 | `cuMemAddressFree` | Future | SHOULD HAVE | Medium | Free VA range. |
| 126 | `cuMemCreate` | Future | SHOULD HAVE | Very High | Create physical allocation handle. |
| 127 | `cuMemRelease` | Future | SHOULD HAVE | Medium | Release physical allocation. |
| 128 | `cuMemMap` | Future | SHOULD HAVE | Very High | Map physical to virtual. |
| 129 | `cuMemUnmap` | Future | SHOULD HAVE | Medium | Unmap virtual range. |
| 130 | `cuMemSetAccess` | Future | SHOULD HAVE | Medium | Set access flags for VA range. |
| 131 | `cuMemGetAccess` | Future | NICE TO HAVE | Low | Query access flags. |
| 132 | `cuMemGetAllocationGranularity` | Completeness | SHOULD HAVE | Low | Query allocation granularity. Can answer locally. |
| 133 | `cuMemGetAllocationPropertiesFromHandle` | Future | NICE TO HAVE | Low | Query allocation properties. |
| 134 | `cuMemExportToShareableHandle` | Future | NICE TO HAVE | High | Export for IPC/DMA-BUF. |
| 135 | `cuMemImportFromShareableHandle` | Future | NICE TO HAVE | High | Import IPC handle. |

### 1.10 Execution Control (8 functions)

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 136 | `cuLaunchKernel` | Core | MUST HAVE | Very High | THE critical function. Serialize grid, block, shared mem, args. |
| 137 | `cuLaunchKernelEx` | Completeness | SHOULD HAVE | Very High | Extended launch config (cluster dims, etc.). CUDA 12+. |
| 138 | `cuLaunchCooperativeKernel` | Completeness | SHOULD HAVE | Very High | Cooperative launch (thread blocks can sync). |
| 139 | `cuLaunchHostFunc` | Completeness | SHOULD HAVE | High | Enqueue host callback on stream. Execute on client side. |
| 140 | `cuFuncGetAttribute` | Completeness | MUST HAVE | Low | Query function attributes (shared mem, regs, etc.). |
| 141 | `cuFuncSetAttribute` | Completeness | SHOULD HAVE | Low | Set function attributes. |
| 142 | `cuFuncSetCacheConfig` | Completeness | SHOULD HAVE | Low | Set L1/shared memory preference. |
| 143 | `cuFuncSetSharedMemConfig` | Completeness | NICE TO HAVE | Low | Set shared memory bank config. |
| 144 | `cuFuncGetName` | Completeness | NICE TO HAVE | Low | CUDA 12+ function name query. |

### 1.11 Stream Management (16 functions)

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 145 | `cuStreamCreate` | Completeness | MUST HAVE | High | Create stream on remote GPU. Handle translation. |
| 146 | `cuStreamCreateWithPriority` | Completeness | MUST HAVE | High | Priority stream creation. |
| 147 | `cuStreamDestroy_v2` | Completeness | MUST HAVE | Medium | Destroy remote stream. |
| 148 | `cuStreamSynchronize` | Completeness | MUST HAVE | Medium | Block until all stream work completes on remote GPU. |
| 149 | `cuStreamQuery` | Completeness | MUST HAVE | Medium | Non-blocking check if stream work is done. |
| 150 | `cuStreamWaitEvent` | Completeness | MUST HAVE | High | Make stream wait for event. Critical for multi-stream ordering. |
| 151 | `cuStreamAddCallback` | Completeness | SHOULD HAVE | High | Add host callback to stream. Execute on client when server signals. |
| 152 | `cuStreamAttachMemAsync` | Completeness | NICE TO HAVE | Medium | Attach managed memory to stream. |
| 153 | `cuStreamGetPriority` | Completeness | SHOULD HAVE | Low | Query stream priority. Local lookup. |
| 154 | `cuStreamGetFlags` | Completeness | SHOULD HAVE | Low | Query stream flags. Local lookup. |
| 155 | `cuStreamGetCtx` | Completeness | SHOULD HAVE | Low | Query stream's context. Local lookup. |
| 156 | `cuStreamGetId` | Completeness | NICE TO HAVE | Low | CUDA 12+ stream ID. |
| 157 | `cuStreamBeginCapture_v2` | Future | SHOULD HAVE | Very High | Graph capture start. See Section 6. |
| 158 | `cuStreamEndCapture` | Future | SHOULD HAVE | Very High | Graph capture end. |
| 159 | `cuStreamIsCapturing` | Future | SHOULD HAVE | Low | Query capture status. |
| 160 | `cuStreamGetCaptureInfo_v2` | Future | NICE TO HAVE | Medium | Query capture info. |

### 1.12 Event Management (7 functions)

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 161 | `cuEventCreate` | Completeness | MUST HAVE | Medium | Create event on remote GPU. Handle translation. |
| 162 | `cuEventDestroy_v2` | Completeness | MUST HAVE | Low | Destroy remote event. |
| 163 | `cuEventRecord` | Completeness | MUST HAVE | Medium | Record event in stream on remote GPU. |
| 164 | `cuEventRecordWithFlags` | Completeness | SHOULD HAVE | Medium | Record with flags (e.g., external event). |
| 165 | `cuEventSynchronize` | Completeness | MUST HAVE | Medium | Block until event is recorded on remote GPU. |
| 166 | `cuEventQuery` | Completeness | MUST HAVE | Medium | Non-blocking event status check. |
| 167 | `cuEventElapsedTime` | Completeness | MUST HAVE | Medium | Compute time between two remote events. Server computes, returns float. |

### 1.13 Texture Object Management (4 functions)

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 168 | `cuTexObjectCreate` | Completeness | SHOULD HAVE | High | Create texture object on remote GPU. |
| 169 | `cuTexObjectDestroy` | Completeness | SHOULD HAVE | Medium | Destroy remote texture. |
| 170 | `cuTexObjectGetResourceDesc` | Completeness | NICE TO HAVE | Low | Query texture resource. |
| 171 | `cuTexObjectGetTextureDesc` | Completeness | NICE TO HAVE | Low | Query texture descriptor. |

### 1.14 Surface Object Management (3 functions)

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 172 | `cuSurfObjectCreate` | Future | NICE TO HAVE | High | Create surface object. |
| 173 | `cuSurfObjectDestroy` | Future | NICE TO HAVE | Medium | Destroy surface object. |
| 174 | `cuSurfObjectGetResourceDesc` | Future | NICE TO HAVE | Low | Query surface resource. |

### 1.15 Peer Context Memory Access (4 functions)

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 175 | `cuCtxEnablePeerAccess` | Completeness | SHOULD HAVE | High | Enable P2P between contexts. For multi-GPU on same server. |
| 176 | `cuCtxDisablePeerAccess` | Completeness | SHOULD HAVE | Medium | Disable P2P. |
| 177 | `cuDeviceCanAccessPeer` | Completeness | MUST HAVE | Low | Query P2P capability. Remote GPUs on same server: yes. Cross-server: no. |
| 178 | `cuDeviceGetP2PAttribute` | Completeness | SHOULD HAVE | Low | Query P2P attributes (bandwidth, etc.). |

### 1.16 Occupancy (4 functions)

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 179 | `cuOccupancyMaxActiveBlocksPerMultiprocessor` | Completeness | SHOULD HAVE | Low | Query on remote GPU. Forward to server. |
| 180 | `cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` | Completeness | SHOULD HAVE | Low | Same with flags. |
| 181 | `cuOccupancyMaxPotentialBlockSize` | Completeness | SHOULD HAVE | Medium | Optimal block size query. |
| 182 | `cuOccupancyMaxPotentialBlockSizeWithFlags` | Completeness | NICE TO HAVE | Medium | Same with flags. |

### 1.17 External Resource Interoperability (6 functions)

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 183 | `cuImportExternalMemory` | Future | NICE TO HAVE | High | Import Vulkan/OpenGL/DMA-BUF memory. |
| 184 | `cuExternalMemoryGetMappedBuffer` | Future | NICE TO HAVE | High | Map external memory. |
| 185 | `cuExternalMemoryGetMappedMipmappedArray` | Future | NICE TO HAVE | High | Map as mipmap. |
| 186 | `cuDestroyExternalMemory` | Future | NICE TO HAVE | Low | Cleanup. |
| 187 | `cuImportExternalSemaphore` | Future | NICE TO HAVE | High | Import Vulkan/timeline semaphore. |
| 188 | `cuDestroyExternalSemaphore` | Future | NICE TO HAVE | Low | Cleanup. |

### 1.18 Stream Memory Operations (2 functions)

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 189 | `cuStreamWriteValue32_v2` | Future | NICE TO HAVE | Medium | Write value to memory from stream. |
| 190 | `cuStreamWaitValue32_v2` | Future | NICE TO HAVE | Medium | Wait for memory value from stream. |

### 1.19 Graph Management (20+ functions)

See Section 6 for detailed strategy. Summary of key functions:

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 191 | `cuGraphCreate` | Future | SHOULD HAVE | Very High | Create graph. |
| 192 | `cuGraphDestroy` | Future | SHOULD HAVE | Medium | Destroy graph. |
| 193 | `cuGraphInstantiate_v2` | Future | SHOULD HAVE | Very High | Instantiate graph for launch. |
| 194 | `cuGraphLaunch` | Future | SHOULD HAVE | Very High | Launch instantiated graph. |
| 195 | `cuGraphAddKernelNode_v2` | Future | SHOULD HAVE | High | Add kernel node. |
| 196 | `cuGraphAddMemcpyNode` | Future | SHOULD HAVE | High | Add memcpy node. |
| 197 | `cuGraphAddMemsetNode` | Future | SHOULD HAVE | High | Add memset node. |
| 198 | `cuGraphAddHostNode` | Future | NICE TO HAVE | Medium | Add host callback node. |
| 199 | `cuGraphAddChildGraphNode` | Future | NICE TO HAVE | High | Nested graph. |
| 200 | `cuGraphAddEventRecordNode` | Future | NICE TO HAVE | Medium | Event record in graph. |
| 201 | `cuGraphAddEventWaitNode` | Future | NICE TO HAVE | Medium | Event wait in graph. |
| 202 | `cuGraphExecDestroy` | Future | SHOULD HAVE | Low | Cleanup. |

### 1.20 Unified Addressing (3 functions)

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 203 | `cuPointerGetAttribute` | Completeness | SHOULD HAVE | Medium | Query pointer attributes (which device, host/device, etc.). |
| 204 | `cuPointerGetAttributes` | Completeness | SHOULD HAVE | Medium | Batch query. |
| 205 | `cuPointerSetAttribute` | Future | NICE TO HAVE | Medium | Set pointer attribute. |
| 206 | `cuMemRangeGetAttribute` | Future | NICE TO HAVE | Medium | Query attribute for memory range. |
| 207 | `cuMemRangeGetAttributes` | Future | NICE TO HAVE | Medium | Batch query for range. |

### 1.21 Linker (5 functions)

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 208 | `cuLinkCreate_v2` | Completeness | SHOULD HAVE | High | Create JIT linker. Used by frameworks for PTX compilation. |
| 209 | `cuLinkAddData_v2` | Completeness | SHOULD HAVE | High | Add data to linker. |
| 210 | `cuLinkAddFile_v2` | Completeness | SHOULD HAVE | High | Add file to linker. |
| 211 | `cuLinkComplete` | Completeness | SHOULD HAVE | High | Complete linking, get cubin. |
| 212 | `cuLinkDestroy` | Completeness | SHOULD HAVE | Low | Destroy linker. |

### 1.22 Driver Entry Point Access (2 functions)

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 213 | `cuGetProcAddress` | Core | MUST HAVE | Very High | THE gatekeeper. CUDA 11.3+ resolves all functions through this. Must return our hooks. |
| 214 | `cuGetProcAddress_v2` | Core | MUST HAVE | Very High | Extended variant with symbolStatus output. CUDA 12+. |

### 1.23 Error Handling (3 functions)

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 215 | `cuGetErrorString` | PoC | MUST HAVE | Low | Purely local. Return error string for CUresult. |
| 216 | `cuGetErrorName` | PoC | MUST HAVE | Low | Purely local. Return error name. |
| 217 | `cuGetExportTable` | Core | SHOULD HAVE | High | Internal NVIDIA export table. Opaque but some apps call it. |

### 1.24 Profiler Control (2 functions)

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 218 | `cuProfilerStart` | Future | NICE TO HAVE | Low | Forward to remote or no-op. |
| 219 | `cuProfilerStop` | Future | NICE TO HAVE | Low | Forward to remote or no-op. |

### 1.25 Tensor Map Object Management (CUDA 12+, 2 functions)

| # | Function | Phase | Priority | Complexity | Notes |
|---|----------|-------|----------|-----------|-------|
| 220 | `cuTensorMapEncodeTiled` | Future | NICE TO HAVE | High | Hopper+ TMA support. |
| 221 | `cuTensorMapEncodeIm2col` | Future | NICE TO HAVE | High | Hopper+ TMA support. |
| 222 | `cuTensorMapReplaceAddress` | Future | NICE TO HAVE | Medium | Replace tensor map address. |

### Function Count Summary

| Phase | MUST HAVE | SHOULD HAVE | NICE TO HAVE | Total |
|-------|-----------|-------------|--------------|-------|
| **PoC (P5)** | 9 | 0 | 0 | 9 |
| **Core (P6)** | 24 | 3 | 0 | 27 |
| **Completeness (P7)** | 19 | 63 | 28 | 110 |
| **Future** | 0 | 18 | 58 | 76 |
| **Total** | **52** | **84** | **86** | **222** |

---

## Section 2: NVML Interception Plan

### 2.1 Why NVML Interception is Required

- `nvidia-smi` uses NVML exclusively (not CUDA Driver API)
- PyTorch checks NVML before CUDA init when `PYTORCH_NVML_BASED_CUDA_CHECK=1`
- TensorFlow queries NVML for GPU info during initialization
- Training frameworks use NVML for monitoring (temperature, utilization, memory)

### 2.2 Interception Mechanism

Hook `libnvidia-ml.so` using the same LD_PRELOAD + dlsym approach as CUDA. Our library intercepts symbol lookups for `nvml*` functions.

```
Application / nvidia-smi
    |
    | dlopen("libnvidia-ml.so.1")
    | dlsym("nvmlInit_v2")
    v
OutterLink NVML Hook Layer
    |-- Merge local + remote GPU info
    |-- Cache remote properties
    |-- Forward monitoring queries over network
    v
Real libnvidia-ml.so (for local GPUs only)
```

### 2.3 NVML Functions to Intercept

#### 2.3.1 Initialization (3 functions) -- MUST HAVE

| Function | Strategy | Notes |
|----------|----------|-------|
| `nvmlInit_v2` | Forward to real + connect to remote servers | Initialize both local NVML and remote connections. |
| `nvmlInitWithFlags` | Same as above | Honor flags. |
| `nvmlShutdown` | Disconnect remotes + forward to real | Clean shutdown. |

#### 2.3.2 System Queries (3 functions) -- MUST HAVE

| Function | Strategy | Notes |
|----------|----------|-------|
| `nvmlSystemGetDriverVersion` | Return remote driver version | Or min(local, remote) if mixed. |
| `nvmlSystemGetCudaDriverVersion_v2` | Return remote CUDA driver version | Must match what cuDriverGetVersion returns. |
| `nvmlSystemGetNVMLVersion` | Return our NVML version | Or real NVML version. |

#### 2.3.3 Device Enumeration (6 functions) -- MUST HAVE

| Function | Strategy | Notes |
|----------|----------|-------|
| `nvmlDeviceGetCount_v2` | Return local_count + remote_count | This is how frameworks discover GPUs. |
| `nvmlDeviceGetHandleByIndex_v2` | Map index to local or remote handle | Indices 0..N-1 where N = local+remote. |
| `nvmlDeviceGetHandleByUUID` | Lookup by UUID (synthetic for remote) | Generate deterministic UUIDs for remote GPUs. |
| `nvmlDeviceGetHandleBySerial` | Lookup by serial | Rarely used. |
| `nvmlDeviceGetHandleByPciBusId_v2` | Lookup by PCI bus ID (synthetic) | PyTorch uses this. |
| `nvmlDeviceGetIndex` | Return our assigned index | |

#### 2.3.4 Device Identification (8 functions) -- MUST HAVE

| Function | Strategy | Notes |
|----------|----------|-------|
| `nvmlDeviceGetName` | Query remote, cache | e.g., "NVIDIA GeForce RTX 3090 Ti" |
| `nvmlDeviceGetBrand` | Query remote, cache | e.g., NVML_BRAND_GEFORCE_RTX |
| `nvmlDeviceGetUUID` | Synthetic deterministic UUID | Format: "GPU-{hash(server_id, gpu_index)}" |
| `nvmlDeviceGetSerial` | Synthetic or query remote | |
| `nvmlDeviceGetPciInfo_v3` | Synthetic PCI info | Generate fake bus/device/function. |
| `nvmlDeviceGetMinorNumber` | Synthetic minor number | For /dev/nvidia{N}. |
| `nvmlDeviceGetCudaComputeCapability` | Query remote, cache | e.g., 8.6 for 3090 Ti. |
| `nvmlDeviceGetBoardId` | Query remote, cache | Board identification. |

#### 2.3.5 Device Properties (10 functions) -- SHOULD HAVE

| Function | Strategy | Notes |
|----------|----------|-------|
| `nvmlDeviceGetMemoryInfo_v2` | Query remote (live) | total/free/used. Must reflect real-time remote state. |
| `nvmlDeviceGetUtilizationRates` | Query remote (live) | GPU/memory utilization %. |
| `nvmlDeviceGetTemperature` | Query remote (live) | Current temp. |
| `nvmlDeviceGetPowerUsage` | Query remote (live) | Current watts. |
| `nvmlDeviceGetClockInfo` | Query remote (live) | Current clock MHz. |
| `nvmlDeviceGetMaxClockInfo` | Query remote, cache | Max clock MHz. |
| `nvmlDeviceGetFanSpeed_v2` | Query remote (live) | Fan %. |
| `nvmlDeviceGetPowerManagementLimit` | Query remote, cache | Power limit. |
| `nvmlDeviceGetEnforcedPowerLimit` | Query remote, cache | Enforced limit. |
| `nvmlDeviceGetBAR1MemoryInfo` | Query remote, cache | BAR1 info. |

#### 2.3.6 Process Info (3 functions) -- SHOULD HAVE

| Function | Strategy | Notes |
|----------|----------|-------|
| `nvmlDeviceGetComputeRunningProcesses_v3` | Query remote | Return remote process list. |
| `nvmlDeviceGetGraphicsRunningProcesses_v3` | Query remote | |
| `nvmlDeviceGetMPSComputeRunningProcesses_v3` | Query remote | |

#### 2.3.7 Device Topology (3 functions) -- NICE TO HAVE

| Function | Strategy | Notes |
|----------|----------|-------|
| `nvmlDeviceGetTopologyCommonAncestor` | Synthesize | Remote GPUs are "PCIe" distance from local. |
| `nvmlDeviceGetTopologyNearestGpus` | Synthesize | Group by server. |
| `nvmlSystemGetTopologyGpuSet` | Synthesize | |

#### 2.3.8 ECC and Reliability (4 functions) -- NICE TO HAVE

| Function | Strategy | Notes |
|----------|----------|-------|
| `nvmlDeviceGetEccMode` | Query remote, cache | |
| `nvmlDeviceGetTotalEccErrors` | Query remote (live) | |
| `nvmlDeviceGetDetailedEccErrors` | Query remote (live) | |
| `nvmlDeviceGetRetiredPages` | Query remote (live) | |

### 2.4 Properties: Fake vs Query Remotely

| Property Category | Strategy | Cacheable | Rationale |
|-------------------|----------|-----------|-----------|
| **Name, brand, compute capability** | Query remote once | Yes (immutable) | Never changes. Cache at connection time. |
| **UUID, serial, PCI info** | Synthesize locally | Yes (immutable) | Cannot expose real PCI topology. Generate deterministic fakes. |
| **Total memory, BAR1 size** | Query remote once | Yes (immutable) | Physical property, doesn't change. |
| **Free/used memory** | Query remote on every call | No | Changes constantly during use. |
| **Temperature, power, utilization** | Query remote on every call | No (but can rate-limit) | Live monitoring data. Rate-limit to max 1 query/sec. |
| **Clock speeds** | Query remote on every call | No (but can rate-limit) | Dynamic boost clocks change. |
| **Driver/CUDA version** | Query remote once | Yes (immutable per session) | Doesn't change during runtime. |
| **ECC mode, retired pages** | Query remote once | Yes (rarely changes) | Only changes with admin action. |

### 2.5 Synthetic Identity Generation

For remote GPUs, we must generate deterministic, unique identifiers:

```
UUID format:   GPU-OUTTERLINK-{server_uuid_prefix}-{gpu_index}
               e.g., GPU-OUTTERLINK-a1b2c3d4-0

PCI Bus ID:    0000:ff:{server_index}{gpu_index}:0
               e.g., 0000:ff:10:0  (server 1, GPU 0)

Minor number:  Starting from max_local_minor + 1
               e.g., local GPU is /dev/nvidia0, remote is /dev/nvidia1

Serial:        OUTTERLINK-{server_id}-{gpu_index}
```

---

## Section 3: PyTorch Compatibility Checklist

### 3.1 `import torch; torch.cuda.is_available()`

PyTorch's `is_available()` follows one of two paths:

**Path A: NVML-based check** (when `PYTORCH_NVML_BASED_CUDA_CHECK=1`)
1. `nvmlInit_v2()` -- initialize NVML
2. `nvmlDeviceGetCount_v2()` -- count GPUs; if > 0, return True
3. `nvmlShutdown()` -- cleanup

**Path B: CUDA Runtime check** (default)
1. `cudaGetDeviceCount()` which internally calls:
   - `cuInit(0)` -- initialize driver
   - `cuDeviceGetCount()` -- count devices

**OutterLink requirement:** Both paths must work. NVML path requires our NVML hooks. CUDA path requires our Driver API hooks. Both must report remote GPUs.

**Functions needed:** `cuInit`, `cuDeviceGetCount`, `nvmlInit_v2`, `nvmlDeviceGetCount_v2`, `nvmlShutdown`

### 3.2 `torch.cuda.device_count()` and Device Properties

1. `cuDeviceGetCount()` -- return local + remote count
2. `cuDeviceGet(device_index)` -- get handle
3. `cuDeviceGetName(handle)` -- device name string
4. `cuDeviceGetAttribute(attr, handle)` -- many attributes queried:
   - `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR` / `MINOR`
   - `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`
   - `CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK`
   - `CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR`
   - `CU_DEVICE_ATTRIBUTE_WARP_SIZE`
   - `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK`
   - `CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY`
   - `CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK`
   - `CU_DEVICE_ATTRIBUTE_CLOCK_RATE`
   - `CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE`
   - `CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH`
   - `CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE`
   - `CU_DEVICE_ATTRIBUTE_PCI_BUS_ID` / `PCI_DEVICE_ID` / `PCI_DOMAIN_ID`
   - `CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING`
   - `CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY`
   - `CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS`
   - ~50+ more attributes
5. `cuDeviceTotalMem_v2(handle)` -- total VRAM
6. `cuDeviceGetUuid_v2(handle)` -- UUID for device identification
7. `cuDeviceGetPCIBusId(handle)` -- PCI bus string

**Functions needed:** `cuDeviceGetCount`, `cuDeviceGet`, `cuDeviceGetName`, `cuDeviceGetAttribute` (all ~100 attrs), `cuDeviceTotalMem_v2`, `cuDeviceGetUuid_v2`, `cuDeviceGetPCIBusId`

### 3.3 `model.to('cuda')` / `tensor.cuda()`

This triggers CUDA context creation and memory allocation:

1. **Context creation** (if not already done):
   - `cuDevicePrimaryCtxRetain(device)` -- PyTorch uses primary context
   - `cuCtxSetCurrent(ctx)` -- set as current thread context
   - `cuCtxGetCurrent(&ctx)` -- verify current context

2. **Memory allocation** (for model parameters + buffers):
   - `cuMemAlloc_v2(&ptr, size)` -- allocate device memory for each tensor
   - OR `cuMemAllocAsync(&ptr, size, stream)` -- stream-ordered alloc (PyTorch 2.x)
   - `cuMemGetInfo_v2(&free, &total)` -- check available memory before large allocs

3. **Data transfer** (copy model weights from host to device):
   - `cuMemcpyHtoDAsync_v2(dstDevice, srcHost, byteCount, stream)` -- copy tensor data
   - `cuStreamSynchronize(stream)` -- wait for transfer to complete

4. **Stream creation** (PyTorch creates multiple streams):
   - `cuStreamCreateWithPriority(&stream, flags, priority)` -- create CUDA streams
   - `cuStreamGetPriority(stream, &priority)` -- verify priority

**Functions needed:** `cuDevicePrimaryCtxRetain`, `cuCtxSetCurrent`, `cuCtxGetCurrent`, `cuMemAlloc_v2`, `cuMemAllocAsync`, `cuMemGetInfo_v2`, `cuMemcpyHtoDAsync_v2`, `cuStreamSynchronize`, `cuStreamCreateWithPriority`

### 3.4 `model(input)` -- Forward Pass

1. **Input tensor allocation and copy** (same as 3.3)

2. **Kernel launches** (cuDNN, cuBLAS, custom kernels):
   - `cuModuleLoadData(&module, fatbinData)` -- load JIT-compiled kernels
   - `cuModuleGetFunction(&func, module, "kernel_name")` -- get kernel handle
   - `cuLaunchKernel(func, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMem, stream, args, extra)` -- launch
   - Hundreds to thousands of kernel launches per forward pass

3. **cuBLAS/cuDNN operations** (these use CUDA Driver API internally):
   - All go through `cuLaunchKernel` at the driver level
   - cuBLAS calls `cuMemAlloc` for workspace memory
   - cuDNN calls `cuMemAlloc` for workspace + algorithm selection

4. **Synchronization**:
   - `cuStreamSynchronize(stream)` or `cuEventRecord` + `cuEventSynchronize`
   - `cuEventElapsedTime` for profiling

5. **Memory operations during forward pass**:
   - `cuMemsetD32Async` -- zero out buffers
   - `cuMemcpyDtoDAsync_v2` -- tensor reshape / view operations that require copy

**Functions needed:** `cuModuleLoadData`, `cuModuleGetFunction`, `cuLaunchKernel`, `cuStreamSynchronize`, `cuEventRecord`, `cuEventSynchronize`, `cuEventElapsedTime`, `cuMemsetD32Async`, `cuMemcpyDtoDAsync_v2`, `cuMemAlloc_v2` (workspace)

### 3.5 `loss.backward()` -- Backward Pass

Same set of functions as forward pass, plus:

1. **Gradient allocation**:
   - `cuMemAlloc_v2` -- allocate gradient tensors (same size as parameters)
   - `cuMemAllocAsync` -- stream-ordered gradient alloc

2. **Autograd kernel launches**:
   - `cuLaunchKernel` -- backward kernels for each layer
   - `cuModuleGetFunction` -- may load additional backward-specific kernels

3. **Gradient accumulation**:
   - `cuMemcpyDtoDAsync_v2` -- accumulate gradients
   - `cuLaunchKernel` -- element-wise add kernels

No new function categories beyond forward pass. Same APIs, just more calls.

### 3.6 `optimizer.step()`

1. **Parameter update kernels**:
   - `cuLaunchKernel` -- SGD/Adam/AdamW update kernels
   - These read gradients and update parameters in-place

2. **Optimizer state**:
   - `cuMemAlloc_v2` -- allocate momentum/variance buffers (Adam)
   - `cuMemcpyDtoDAsync_v2` -- copy states if needed

3. **Gradient zeroing** (typically via `optimizer.zero_grad()`):
   - `cuMemsetD32Async` -- zero gradient tensors

No new function categories. Same memory + kernel launch APIs.

### 3.7 Complete PyTorch CUDA Function Usage Summary

| Category | Functions Used | When |
|----------|---------------|------|
| **Init** | `cuInit`, `cuDriverGetVersion` | `import torch` |
| **Device** | `cuDeviceGet`, `cuDeviceGetCount`, `cuDeviceGetName`, `cuDeviceGetAttribute`, `cuDeviceTotalMem_v2`, `cuDeviceGetUuid_v2`, `cuDeviceGetPCIBusId`, `cuDeviceGetByPCIBusId`, `cuDeviceCanAccessPeer` | `torch.cuda.*` queries |
| **Context** | `cuDevicePrimaryCtxRetain`, `cuDevicePrimaryCtxRelease_v2`, `cuCtxSetCurrent`, `cuCtxGetCurrent`, `cuCtxSynchronize`, `cuCtxPushCurrent_v2`, `cuCtxPopCurrent_v2`, `cuCtxGetDevice` | Context management |
| **Memory** | `cuMemAlloc_v2`, `cuMemFree_v2`, `cuMemAllocHost_v2`, `cuMemFreeHost`, `cuMemHostAlloc`, `cuMemHostRegister_v2`, `cuMemHostUnregister`, `cuMemGetInfo_v2`, `cuMemAllocAsync`, `cuMemFreeAsync`, `cuMemAllocManaged` (optional) | Tensor allocation |
| **Transfer** | `cuMemcpyHtoDAsync_v2`, `cuMemcpyDtoHAsync_v2`, `cuMemcpyDtoDAsync_v2`, `cuMemcpyAsync`, `cuMemcpyPeerAsync` | Data movement |
| **Memset** | `cuMemsetD32Async`, `cuMemsetD8Async` | Buffer zeroing |
| **Module** | `cuModuleLoadData`, `cuModuleLoadDataEx`, `cuModuleGetFunction`, `cuModuleUnload` | Kernel loading |
| **Launch** | `cuLaunchKernel`, `cuLaunchKernelEx` | Every computation |
| **Stream** | `cuStreamCreate`, `cuStreamCreateWithPriority`, `cuStreamDestroy_v2`, `cuStreamSynchronize`, `cuStreamQuery`, `cuStreamWaitEvent` | Async execution |
| **Event** | `cuEventCreate`, `cuEventDestroy_v2`, `cuEventRecord`, `cuEventSynchronize`, `cuEventQuery`, `cuEventElapsedTime` | Synchronization + timing |
| **Occupancy** | `cuOccupancyMaxActiveBlocksPerMultiprocessor`, `cuOccupancyMaxPotentialBlockSize` | Kernel config |
| **Linker** | `cuLinkCreate_v2`, `cuLinkAddData_v2`, `cuLinkComplete`, `cuLinkDestroy` | JIT compilation |
| **Entry** | `cuGetProcAddress`, `cuGetProcAddress_v2` | Function resolution |
| **NVML** | `nvmlInit_v2`, `nvmlDeviceGetCount_v2`, `nvmlDeviceGetHandleByIndex_v2`, `nvmlDeviceGetName`, `nvmlDeviceGetMemoryInfo_v2`, `nvmlDeviceGetUUID`, `nvmlShutdown` | GPU detection |
| **Error** | `cuGetErrorString`, `cuGetErrorName` | Error handling |
| **P2P** | `cuDeviceCanAccessPeer`, `cuCtxEnablePeerAccess` | Multi-GPU |
| **Pointer** | `cuPointerGetAttribute`, `cuPointerGetAttributes` | Memory introspection |

**Total unique CUDA Driver API functions PyTorch uses: ~65-75**
**Total unique NVML functions PyTorch uses: ~10-15**

---

## Section 4: Stream and Event Implementation

### 4.1 Stream Architecture

Streams provide ordered execution on the GPU. Our challenge: maintain ordering semantics across the network.

```
Client Process                          Server Process
    |                                       |
    | cuStreamCreate(&stream)               |
    |----[CREATE_STREAM]------------------>|
    |<---[STREAM_HANDLE: 0x1234]-----------|
    |                                       |
    | cuLaunchKernel(func, ..., stream)     |
    |----[LAUNCH: func, stream=0x1234]---->|  <- server enqueues on real stream
    |<---[ACK]------------------------------|
    |                                       |
    | cuMemcpyHtoDAsync(dst, src, n, stream)|
    |----[MEMCPY_HTOD: ..., stream=0x1234]>|  <- server enqueues on same stream
    |<---[ACK]------------------------------|
    |                                       |
    | cuStreamSynchronize(stream)          |
    |----[SYNC_STREAM: 0x1234]----------->|
    |<---[SYNC_COMPLETE]-------------------|  <- blocks until all work done
```

#### Stream Handle Translation

```rust
// Client-side stream handle map
struct StreamMap {
    // Local synthetic handle -> Remote real handle
    local_to_remote: HashMap<CUstream, RemoteStreamHandle>,
    // Remote handle includes server_id for multi-server
    next_local_handle: AtomicU64,
}

struct RemoteStreamHandle {
    server_id: ServerId,
    remote_handle: u64,  // Real CUstream on server
    priority: i32,
    flags: u32,
}
```

#### NULL Stream (Default Stream) Handling

The NULL stream (stream 0) has special semantics -- it synchronizes with all other streams. We must:
1. Map NULL stream to a real default stream on the remote GPU
2. Preserve the implicit synchronization behavior
3. Handle per-thread default stream mode (`CU_STREAM_PER_THREAD`)

### 4.2 Multiple Concurrent Streams

Real applications use multiple streams for overlapping compute and memory transfers:

```
Stream 1: [Kernel A] ---------> [Kernel B]
Stream 2:      [HtoD Copy] --------> [Kernel C]
Stream 3:           [DtoH Copy] ------------>
```

Our implementation must:
1. Create separate real streams on the server for each client stream
2. Send stream-ordered commands to the server, tagging each with its stream handle
3. The server enqueues operations on the correct real stream
4. Stream synchronization only waits for that specific stream's work

#### Batching Stream Operations

Multiple operations on the same stream can be batched into a single network message:

```rust
struct StreamBatch {
    stream: CUstream,
    operations: Vec<StreamOp>,
}

enum StreamOp {
    LaunchKernel { func: CUfunction, grid: Dim3, block: Dim3, args: Vec<u8> },
    MemcpyHtoD { dst: CUdeviceptr, data: Vec<u8> },
    MemsetD32 { dst: CUdeviceptr, value: u32, count: usize },
    EventRecord { event: CUevent },
}
```

Flush the batch on:
- `cuStreamSynchronize` (must execute all pending ops first)
- `cuEventSynchronize` (must execute up to the event)
- `cuStreamQuery` (must check status)
- `cuMemcpyDtoH*` (must read back data)
- Buffer full (configurable batch size limit)

### 4.3 Event Architecture

Events mark points in a stream's execution for synchronization and timing.

#### Event Handle Translation

```rust
struct EventMap {
    local_to_remote: HashMap<CUevent, RemoteEventHandle>,
    next_local_handle: AtomicU64,
}

struct RemoteEventHandle {
    server_id: ServerId,
    remote_handle: u64,
    flags: u32,  // CU_EVENT_DEFAULT, CU_EVENT_BLOCKING_SYNC, CU_EVENT_DISABLE_TIMING
}
```

#### Event Operations Over Network

| Operation | Network Behavior |
|-----------|-----------------|
| `cuEventCreate` | Send to server, get remote handle, store mapping. |
| `cuEventRecord(event, stream)` | Send to server. Server calls real cuEventRecord. |
| `cuEventSynchronize(event)` | Send to server. Server blocks on real cuEventSynchronize, then responds. Client blocks on network recv. |
| `cuEventQuery(event)` | Send to server. Server calls real cuEventQuery, returns CUDA_SUCCESS or CUDA_ERROR_NOT_READY. |
| `cuEventElapsedTime(&ms, start, end)` | Send both event handles to server. Server computes real elapsed time, returns float. |
| `cuEventDestroy(event)` | Send to server. Remove from local map. |

#### cuStreamWaitEvent Over Network

`cuStreamWaitEvent(stream, event)` makes a stream wait until an event has been recorded.

**Same-server case:** Both stream and event are on the same server. Forward directly -- the server calls real `cuStreamWaitEvent`.

**Cross-server case (future):** Stream is on server A, event is on server B. This requires cross-server synchronization protocol. Not implemented in P7 -- return `CUDA_ERROR_NOT_SUPPORTED` for cross-server stream-event dependencies.

### 4.4 Timing Accuracy

`cuEventElapsedTime` returns GPU-side elapsed time between two events. Since both events are on the same remote GPU:
- The server computes the real elapsed time using the real GPU timestamps
- Network latency does NOT affect the timing result
- The returned value is accurate GPU timing, not wall-clock time

---

## Section 5: Unified Memory / Managed Memory

### 5.1 Decision: NOT SUPPORTED in P7

`cuMemAllocManaged` and the unified memory subsystem are **not supported** in the Completeness phase. Reasons:

1. **Page fault handling is kernel-internal.** Unified memory relies on GPU page faults triggering driver-internal page migration. This mechanism is deeply embedded in the NVIDIA proprietary driver and cannot be intercepted or virtualized from userspace.

2. **Semantic impossibility over network.** Unified memory allows CPU code to dereference device pointers directly. Over a network, this would require intercepting every memory access (load/store instruction) -- impossible without hardware support or binary instrumentation.

3. **HAMi-core doesn't support it either.** Even local GPU sharing tools avoid managed memory virtualization.

4. **PyTorch rarely uses it in production.** PyTorch's default allocator uses explicit `cuMemAlloc` + `cuMemcpy`. The `cudaMallocManaged` path is experimental.

### 5.2 Graceful Failure Strategy

When an application calls managed memory functions:

```rust
fn our_cuMemAllocManaged(ptr: *mut CUdeviceptr, size: usize, flags: u32) -> CUresult {
    // Option A: Redirect to explicit device allocation (silent fallback)
    // This works for many use cases but breaks CPU-side pointer derefs.

    // Option B: Return error
    // CUDA_ERROR_NOT_SUPPORTED

    // Decision: Option B (no workarounds policy)
    // Applications that require managed memory will get a clear error.
    // Log a warning message explaining the limitation.
    log::warn!(
        "cuMemAllocManaged called but unified memory is not supported over remote GPU. \
         Use explicit cuMemAlloc + cuMemcpy instead."
    );
    CUDA_ERROR_NOT_SUPPORTED
}
```

### 5.3 Functions Affected

| Function | Response |
|----------|----------|
| `cuMemAllocManaged` | Return `CUDA_ERROR_NOT_SUPPORTED` |
| `cuMemPrefetchAsync` | No-op, return `CUDA_SUCCESS` (prefetch hint is advisory) |
| `cuMemAdvise` | No-op, return `CUDA_SUCCESS` (memory advice is advisory) |
| `cuMemRangeGetAttribute` | Return `CUDA_ERROR_NOT_SUPPORTED` |
| `cuMemRangeGetAttributes` | Return `CUDA_ERROR_NOT_SUPPORTED` |
| `cuPointerGetAttribute` with managed ptr | Return synthetic attributes |

### 5.4 Future Path (P10+)

If unified memory becomes necessary:
- Use `userfaultfd` to intercept CPU page faults on managed memory regions
- On fault: fetch page from remote GPU over network, map locally
- Extremely complex, significant latency, but technically possible
- Would be an entirely separate phase of development

---

## Section 6: CUDA Graphs

### 6.1 Decision: NOT SUPPORTED in P7, SHOULD HAVE for Future

CUDA Graphs record a sequence of operations into a DAG, then replay the entire DAG with a single `cuGraphLaunch` call. This bypasses individual `cuLaunchKernel` hooks.

### 6.2 Why Graphs Are Hard

1. **Bypass interception.** `cuGraphLaunch` replays the recorded graph on the GPU without calling individual kernel launch APIs. Our `cuLaunchKernel` hook never fires for graph-replayed kernels.

2. **Complex DAG serialization.** A graph can contain kernel nodes, memcpy nodes, memset nodes, host callback nodes, child graphs, event nodes, and memory allocation nodes. All must be serialized and reconstructed on the server.

3. **Capture mode breaks hooks.** During `cuStreamBeginCapture`, CUDA records operations instead of executing them. Our hooks would need to detect capture mode and record instead of forward.

### 6.3 Graceful Failure Strategy

```rust
fn our_cuGraphCreate(graph: *mut CUgraph, flags: u32) -> CUresult {
    log::warn!(
        "CUDA Graphs not supported over remote GPU. \
         Operations will execute normally without graph optimization."
    );
    CUDA_ERROR_NOT_SUPPORTED
}

fn our_cuStreamBeginCapture_v2(stream: CUstream, mode: CUstreamCaptureMode) -> CUresult {
    log::warn!("CUDA Graph capture not supported over remote GPU.");
    CUDA_ERROR_NOT_SUPPORTED
}
```

PyTorch falls back to eager execution when graphs are unavailable. TensorFlow similarly degrades gracefully.

### 6.4 Future Implementation Strategy (P10+)

If graph support becomes necessary:

1. **Record phase:** During `cuStreamBeginCapture` through `cuStreamEndCapture`, intercept all stream operations and build our own graph representation (a DAG of operations with dependencies).

2. **Instantiation:** On `cuGraphInstantiate`, serialize the entire graph DAG to the server. Server creates a real CUDA graph from our representation.

3. **Launch:** On `cuGraphLaunch`, send the instantiated graph handle to the server. Server calls real `cuGraphLaunch`.

4. **Update:** Support `cuGraphExecKernelNodeSetParams` for updating graph parameters without re-instantiation.

This is a large effort (estimated 30+ functions, complex DAG serialization) but architecturally sound.

---

## Section 7: Version Compatibility

### 7.1 cuGetProcAddress Version Handling

`cuGetProcAddress` is the critical function resolution mechanism since CUDA 11.3. The runtime no longer links directly to driver symbols -- it resolves them dynamically.

#### How Version Resolution Works

```
Application calls: cuGetProcAddress("cuMemAlloc", &pfn, 3020, flags)
                                     ^             ^      ^
                                     |             |      |
                                     Base name     |   CUDA version 3.2
                                                   |   (= 1000*3 + 10*2)
                                                Function pointer output

The driver returns the _v2 variant because CUDA 3.2 introduced cuMemAlloc_v2.
```

#### Version Mapping Table

We need a version table mapping base names to versioned variants:

```rust
struct VersionedFunction {
    base_name: &'static str,
    versions: &'static [(u32, &'static str)],  // (min_cuda_version, symbol_name)
}

const VERSIONED_FUNCTIONS: &[VersionedFunction] = &[
    VersionedFunction {
        base_name: "cuMemAlloc",
        versions: &[
            (2000, "cuMemAlloc"),       // CUDA 2.0: original
            (3020, "cuMemAlloc_v2"),     // CUDA 3.2: 64-bit pointers
        ],
    },
    VersionedFunction {
        base_name: "cuMemFree",
        versions: &[
            (2000, "cuMemFree"),
            (3020, "cuMemFree_v2"),
        ],
    },
    VersionedFunction {
        base_name: "cuMemcpyHtoD",
        versions: &[
            (2000, "cuMemcpyHtoD"),
            (3020, "cuMemcpyHtoD_v2"),
        ],
    },
    VersionedFunction {
        base_name: "cuCtxCreate",
        versions: &[
            (2000, "cuCtxCreate"),
            (3020, "cuCtxCreate_v2"),
            (11040, "cuCtxCreate_v3"), // CUDA 11.4 (exec affinity)
        ],
    },
    VersionedFunction {
        base_name: "cuCtxDestroy",
        versions: &[
            (2000, "cuCtxDestroy"),
            (4000, "cuCtxDestroy_v2"),
        ],
    },
    VersionedFunction {
        base_name: "cuStreamBeginCapture",
        versions: &[
            (10000, "cuStreamBeginCapture"),    // CUDA 10.0
            (10010, "cuStreamBeginCapture_v2"), // CUDA 10.1
        ],
    },
    VersionedFunction {
        base_name: "cuGetProcAddress",
        versions: &[
            (11030, "cuGetProcAddress"),    // CUDA 11.3: original (4 params)
            (12000, "cuGetProcAddress_v2"), // CUDA 12.0: added symbolStatus (5 params)
        ],
    },
    // ... 50+ more entries
];
```

#### Implementation Strategy

```rust
fn our_cuGetProcAddress_v2(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    cuda_version: c_int,
    flags: u64,
    symbol_status: *mut CUdriverProcAddressQueryResult,
) -> CUresult {
    let sym = unsafe { CStr::from_ptr(symbol) }.to_str().unwrap();

    // 1. Check our hook table for this symbol at this version
    if let Some(hook) = find_hook(sym, cuda_version) {
        unsafe { *pfn = hook as *mut c_void };
        if !symbol_status.is_null() {
            unsafe { *symbol_status = CU_GET_PROC_ADDRESS_SUCCESS };
        }
        return CUDA_SUCCESS;
    }

    // 2. For functions we don't hook, forward to real cuGetProcAddress
    let result = real_cuGetProcAddress_v2(symbol, pfn, cuda_version, flags, symbol_status);
    result
}
```

### 7.2 Handling _v2/_v3 Variants

Both the suffixed AND unsuffixed names must be hooked:

| dlsym lookup | cuGetProcAddress lookup | We must hook |
|---|---|---|
| `cuMemAlloc_v2` | `cuMemAlloc` with version >= 3020 | Both paths |
| `cuCtxDestroy_v2` | `cuCtxDestroy` with version >= 4000 | Both paths |
| `cuEventDestroy_v2` | `cuEventDestroy` with version >= 4000 | Both paths |

The dlsym hook handles explicit symbol lookups. The cuGetProcAddress hook handles runtime-resolved lookups. Both must return our intercepted function.

### 7.3 Per-Thread Default Stream

CUDA supports per-thread default streams (`CU_STREAM_PER_THREAD`). cuGetProcAddress flags include:
- `CU_GET_PROC_ADDRESS_LEGACY_STREAM` -- return legacy (NULL) stream variants
- `CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM` -- return per-thread stream variants

Some functions have per-thread variants (e.g., `cuLaunchKernel_ptsz`, `cuMemcpyAsync_ptsz`). We must handle these flags and return the appropriate hooked variant.

### 7.4 CUDA Version Support Strategy

**Target:** CUDA 12.0 through 12.8+ (and forward-compatible with 13.x)

**Approach:**
1. Build against CUDA 12.x headers for type definitions
2. Use cuGetProcAddress to resolve all function pointers dynamically (never link directly)
3. Maintain version table covering CUDA 10.0 through latest
4. Test against CUDA 12.2, 12.4, 12.6, 12.8 (major point releases)
5. For CUDA 13.x Library API: add support when frameworks start using it

**Deprecated API handling:**
- CUDA 12.3 removed some deprecated CUDA 11.x APIs
- We still hook the old names for backward compatibility
- If the real driver doesn't have them, our hooks return `CUDA_ERROR_NOT_SUPPORTED`

---

## Section 8: Multi-GPU Device Enumeration

### 8.1 Device Numbering Scheme

Remote GPUs appear as additional CUDA devices after any local GPUs:

```
Local PC:
  GPU 0: (local) NVIDIA GeForce RTX 3090 Ti  [/dev/nvidia0]
  GPU 1: (local) NVIDIA GeForce RTX 3090 Ti  [/dev/nvidia1]

Remote Server 1:
  GPU 2: (remote) NVIDIA GeForce RTX 3090 Ti  [synthesized]
  GPU 3: (remote) NVIDIA GeForce RTX 3090 Ti  [synthesized]

Remote Server 2:
  GPU 4: (remote) NVIDIA GeForce RTX 4090     [synthesized]
  GPU 5: (remote) NVIDIA GeForce RTX 4090     [synthesized]
```

So `cuDeviceGetCount()` returns 6, and `cuDeviceGet(4, &dev)` returns a handle for remote Server 2's first GPU.

### 8.2 Device Index Mapping

```rust
struct DeviceRegistry {
    local_count: usize,         // From real cuDeviceGetCount
    remote_devices: Vec<RemoteDevice>,

    // Index mapping: device_index -> DeviceLocation
    index_map: Vec<DeviceLocation>,
}

enum DeviceLocation {
    Local { real_device: CUdevice },
    Remote { server_id: ServerId, server_device_index: u32 },
}

struct RemoteDevice {
    server_id: ServerId,
    server_device_index: u32,  // GPU index on that server (0, 1, ...)
    cached_properties: DeviceProperties,
}

struct DeviceProperties {
    name: String,
    total_mem: usize,
    compute_capability: (i32, i32),
    attributes: HashMap<CUdevice_attribute, i32>,
    uuid: CUuuid,          // Synthesized
    pci_bus_id: String,    // Synthesized
}
```

### 8.3 cuDeviceGet Mapping

```rust
fn our_cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult {
    let registry = DEVICE_REGISTRY.lock();

    if ordinal < 0 || ordinal as usize >= registry.total_count() {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    match &registry.index_map[ordinal as usize] {
        DeviceLocation::Local { real_device } => {
            // Forward to real cuDeviceGet
            real_cuDeviceGet(device, ordinal)
        }
        DeviceLocation::Remote { server_id, server_device_index } => {
            // Return a synthetic device handle that encodes server + index
            let synthetic = encode_remote_device(*server_id, *server_device_index);
            unsafe { *device = synthetic };
            CUDA_SUCCESS
        }
    }
}
```

### 8.4 Device Property Caching

Remote device properties are queried once at connection time and cached:

```rust
fn connect_to_server(server_addr: &str) -> Result<()> {
    let conn = TcpStream::connect(server_addr)?;

    // Query all device info from server
    send_message(&conn, Message::QueryDevices)?;
    let devices: Vec<ServerDeviceInfo> = recv_message(&conn)?;

    for (idx, dev_info) in devices.iter().enumerate() {
        let remote_dev = RemoteDevice {
            server_id: server_id,
            server_device_index: idx as u32,
            cached_properties: DeviceProperties {
                name: dev_info.name.clone(),
                total_mem: dev_info.total_mem,
                compute_capability: dev_info.compute_capability,
                attributes: dev_info.attributes.clone(), // All ~100 attributes
                uuid: generate_synthetic_uuid(server_id, idx),
                pci_bus_id: generate_synthetic_pci_bus_id(server_id, idx),
            },
        };
        registry.add_remote_device(remote_dev);
    }
    Ok(())
}
```

### 8.5 CUDA_VISIBLE_DEVICES Compatibility

Users may set `CUDA_VISIBLE_DEVICES` to control GPU visibility. We must:

1. Parse `CUDA_VISIBLE_DEVICES` before our device enumeration
2. Filter both local and remote GPUs based on the setting
3. Support extended syntax: `CUDA_VISIBLE_DEVICES=0,1,remote:2,remote:3`
4. Or use a separate env var: `OUTTERLINK_VISIBLE_DEVICES=server1:0,server1:1`

### 8.6 P2P Capability Reporting

| Query | Local-Local | Local-Remote | Remote-Remote (same server) | Remote-Remote (different server) |
|-------|-------------|--------------|---------------------------|-------------------------------|
| `cuDeviceCanAccessPeer` | Real result | `0` (no P2P over network) | Query server | `0` (no cross-server P2P) |
| `cuDeviceGetP2PAttribute` | Real result | Not supported | Query server | Not supported |

---

## Section 9: Implementation Priority Order

### 9.1 Implementation Groups

Work is organized into groups that make sense to implement together. Each group is a coherent unit of functionality that can be tested independently.

#### Group 1: Stream Foundation (16 functions)

**Implement first.** Streams are prerequisites for most other P7 features.

| # | Functions | Notes |
|---|-----------|-------|
| 1 | `cuStreamCreate` | Basic stream creation |
| 2 | `cuStreamCreateWithPriority` | Priority streams |
| 3 | `cuStreamDestroy_v2` | Stream cleanup |
| 4 | `cuStreamSynchronize` | Blocking sync |
| 5 | `cuStreamQuery` | Non-blocking status |
| 6 | `cuStreamWaitEvent` | Cross-stream dependency |
| 7 | `cuStreamAddCallback` | Host callback (can defer) |
| 8 | `cuStreamGetPriority` | Priority query |
| 9 | `cuStreamGetFlags` | Flags query |
| 10 | `cuStreamGetCtx` | Context query |
| 11 | `cuStreamGetId` | CUDA 12+ |
| 12 | `cuStreamAttachMemAsync` | Managed memory attach (no-op) |

**Estimated effort:** HIGH (stream ordering semantics over network)
**Test:** Create multiple streams, launch kernels on different streams, verify ordering.

#### Group 2: Event Foundation (7 functions)

**Implement with streams.** Events depend on streams.

| # | Functions |
|---|-----------|
| 1 | `cuEventCreate` |
| 2 | `cuEventDestroy_v2` |
| 3 | `cuEventRecord` |
| 4 | `cuEventRecordWithFlags` |
| 5 | `cuEventSynchronize` |
| 6 | `cuEventQuery` |
| 7 | `cuEventElapsedTime` |

**Estimated effort:** MEDIUM
**Test:** Record events in streams, measure elapsed time, verify correctness.

#### Group 3: NVML Core (20 functions)

**Implement early.** Required for PyTorch GPU detection.

| # | Functions | Count |
|---|-----------|-------|
| 1 | Init/shutdown | 3 |
| 2 | System queries | 3 |
| 3 | Device enumeration | 6 |
| 4 | Device identification | 8 |

**Estimated effort:** MEDIUM (mostly data marshaling, synthetic identity)
**Test:** `nvidia-smi` shows remote GPUs, PyTorch detects them.

#### Group 4: Extended Memory Operations (15 functions)

**Implement after streams work.** Async memset, 2D copies, host registration.

| # | Functions | Count |
|---|-----------|-------|
| 1 | `cuMemsetD8Async`, `cuMemsetD16Async`, `cuMemsetD32Async` | 3 |
| 2 | `cuMemsetD8_v2`, `cuMemsetD16_v2`, `cuMemsetD2D*` (sync) | 6 |
| 3 | `cuMemcpy`, `cuMemcpy2D_v2`, `cuMemcpy2DAsync_v2` | 3 |
| 4 | `cuMemHostRegister_v2`, `cuMemHostUnregister` | 2 |
| 5 | `cuMemGetAddressRange_v2` | 1 |

**Estimated effort:** MEDIUM
**Test:** 2D memory copies, memset operations, host memory registration.

#### Group 5: Stream-Ordered Allocator (10 functions)

**Implement after streams.** PyTorch 2.x depends on this for performance.

| # | Functions | Count |
|---|-----------|-------|
| 1 | `cuMemAllocAsync`, `cuMemFreeAsync` | 2 |
| 2 | `cuMemAllocFromPoolAsync` | 1 |
| 3 | `cuMemPoolCreate`, `cuMemPoolDestroy` | 2 |
| 4 | `cuMemPoolGetAttribute`, `cuMemPoolSetAttribute` | 2 |
| 5 | `cuMemPoolTrimTo`, `cuMemPoolGetAccess`, `cuMemPoolSetAccess` | 3 |

**Estimated effort:** HIGH (pool management + stream ordering)
**Test:** Allocate and free with stream ordering, verify no leaks.

#### Group 6: Device Extended Properties (10 functions)

**Implement alongside NVML.** Fill in remaining device query functions.

| # | Functions | Count |
|---|-----------|-------|
| 1 | `cuDeviceGetUuid_v2` | 1 |
| 2 | `cuDeviceGetByPCIBusId`, `cuDeviceGetPCIBusId` | 2 |
| 3 | `cuDeviceGetDefaultMemPool`, `cuDeviceGetMemPool`, `cuDeviceSetMemPool` | 3 |
| 4 | `cuDeviceCanAccessPeer`, `cuDeviceGetP2PAttribute` | 2 |
| 5 | `cuCtxEnablePeerAccess`, `cuCtxDisablePeerAccess` | 2 |

**Estimated effort:** LOW-MEDIUM
**Test:** PyTorch multi-GPU detection, P2P queries.

#### Group 7: Execution Extensions (8 functions)

**Implement after core kernel launch works.**

| # | Functions | Count |
|---|-----------|-------|
| 1 | `cuLaunchKernelEx`, `cuLaunchCooperativeKernel` | 2 |
| 2 | `cuFuncGetAttribute`, `cuFuncSetAttribute`, `cuFuncSetCacheConfig` | 3 |
| 3 | `cuLaunchHostFunc` | 1 |
| 4 | `cuOccupancyMaxActiveBlocksPerMultiprocessor` (+Flags) | 2 |

**Estimated effort:** MEDIUM
**Test:** Extended kernel launch, occupancy queries.

#### Group 8: JIT Linker (5 functions)

**Implement for PyTorch JIT compilation support.**

| # | Functions | Count |
|---|-----------|-------|
| 1 | `cuLinkCreate_v2`, `cuLinkAddData_v2`, `cuLinkAddFile_v2` | 3 |
| 2 | `cuLinkComplete`, `cuLinkDestroy` | 2 |

**Estimated effort:** HIGH (binary data transfer + remote JIT)
**Test:** JIT-compile PTX on remote GPU, launch resulting kernel.

#### Group 9: NVML Monitoring (10 functions)

**Implement for nvidia-smi compatibility and framework monitoring.**

| # | Functions | Count |
|---|-----------|-------|
| 1 | `nvmlDeviceGetMemoryInfo_v2` | 1 |
| 2 | `nvmlDeviceGetUtilizationRates` | 1 |
| 3 | `nvmlDeviceGetTemperature` | 1 |
| 4 | `nvmlDeviceGetPowerUsage` | 1 |
| 5 | `nvmlDeviceGetClockInfo`, `nvmlDeviceGetMaxClockInfo` | 2 |
| 6 | `nvmlDeviceGetFanSpeed_v2` | 1 |
| 7 | `nvmlDeviceGetPowerManagementLimit`, `nvmlDeviceGetEnforcedPowerLimit` | 2 |
| 8 | `nvmlDeviceGetBAR1MemoryInfo` | 1 |

**Estimated effort:** LOW (straightforward remote query + caching)
**Test:** `nvidia-smi` shows live stats for remote GPUs.

#### Group 10: Texture/Surface/Array (8 functions)

**Implement last.** Needed for image processing but not for ML inference.

| # | Functions | Count |
|---|-----------|-------|
| 1 | `cuArrayCreate_v2`, `cuArray3DCreate_v2`, `cuArrayDestroy` | 3 |
| 2 | `cuTexObjectCreate`, `cuTexObjectDestroy` | 2 |
| 3 | `cuArrayGetDescriptor_v2`, `cuArray3DGetDescriptor_v2` | 2 |
| 4 | `cuTexObjectGetResourceDesc` | 1 |

**Estimated effort:** MEDIUM
**Test:** Texture sampling kernel on remote GPU.

### 9.2 Implementation Sequence

```
Week 1-2:  Group 1 (Streams) + Group 2 (Events)     = 23 functions
Week 3:    Group 3 (NVML Core)                        = 20 functions
Week 4:    Group 4 (Extended Memory)                   = 15 functions
Week 5:    Group 5 (Stream-Ordered Allocator)          = 10 functions
Week 6:    Group 6 (Device Properties)                 = 10 functions
Week 7:    Group 7 (Execution Extensions)              =  8 functions
Week 8:    Group 8 (JIT Linker) + Group 9 (NVML Mon)  = 15 functions
Week 9:    Group 10 (Texture/Surface)                  =  8 functions
Week 10:   Integration testing, PyTorch validation     =  0 new functions
                                                  Total: ~109 functions
```

### 9.3 Validation Gates

| Gate | Test | Pass Criteria |
|------|------|---------------|
| **G1** | After Groups 1+2 | Multi-stream kernel launch with event sync works correctly |
| **G2** | After Group 3 | `nvidia-smi` shows remote GPUs, `torch.cuda.is_available()` returns True |
| **G3** | After Groups 4+5 | PyTorch tensor operations work (alloc, copy, memset) |
| **G4** | After Groups 6+7 | `model.to('cuda')` succeeds, simple forward pass works |
| **G5** | After Groups 8+9 | PyTorch JIT compilation works, nvidia-smi shows live stats |
| **G6** | After Group 10 | Image processing workloads with textures work |
| **FINAL** | After all groups | ResNet inference on remote GPU produces correct results |

---

## Section 10: Testing Strategy

### 10.1 Unit Tests

| Test Category | What to Test |
|---------------|-------------|
| Handle translation | Synthetic handle creation, lookup, removal, thread safety |
| Version table | All versioned function lookups return correct symbols |
| Device numbering | Local + remote indexing, out-of-bounds, CUDA_VISIBLE_DEVICES |
| NVML identity | UUID generation determinism, PCI bus ID format |
| Stream batching | Operations batched correctly, flush triggers work |

### 10.2 Integration Tests

| Test | Description |
|------|-------------|
| `test_stream_ordering` | Launch kernels on multiple streams, verify execution order |
| `test_event_timing` | Record events, verify elapsed time is plausible |
| `test_stream_wait_event` | Stream A waits for event in stream B, verify ordering |
| `test_nvml_device_count` | NVML reports correct total device count |
| `test_nvml_memory_info` | NVML memory info matches cuMemGetInfo |
| `test_stream_ordered_alloc` | cuMemAllocAsync + cuMemFreeAsync lifecycle |
| `test_jit_link` | PTX -> cubin via linker on remote GPU |
| `test_managed_memory_error` | cuMemAllocManaged returns NOT_SUPPORTED |
| `test_graph_error` | cuGraphCreate returns NOT_SUPPORTED |

### 10.3 PyTorch Compatibility Tests

| Test | Command | Expected |
|------|---------|----------|
| GPU detection | `python -c "import torch; print(torch.cuda.is_available())"` | `True` |
| Device count | `python -c "import torch; print(torch.cuda.device_count())"` | Correct count |
| Device name | `python -c "import torch; print(torch.cuda.get_device_name(N))"` | Remote GPU name |
| Tensor alloc | `python -c "import torch; t = torch.zeros(1000).cuda(N)"` | Success |
| Tensor copy | `python -c "import torch; t = torch.randn(1000).cuda(N); print(t.cpu())"` | Correct values |
| Model load | `python -c "import torchvision; m = torchvision.models.resnet18().cuda(N)"` | Success |
| Inference | Run ResNet18 forward pass on remote GPU | Correct output |

### 10.4 Stress Tests

| Test | Description |
|------|-------------|
| Multi-stream stress | 32 concurrent streams, 1000 kernel launches each |
| Event storm | Create/record/sync 10000 events |
| Memory pressure | Allocate until OOM, verify error handling |
| Connection loss | Kill server mid-operation, verify client error handling |

---

## Section 11: Risks and Mitigations

| # | Risk | Impact | Probability | Mitigation |
|---|------|--------|-------------|-----------|
| R1 | cuGetProcAddress changes in CUDA 13.x | HIGH | LOW | Version-indexed table, CI tests against multiple CUDA versions |
| R2 | PyTorch uses internal/undocumented CUDA APIs | HIGH | MEDIUM | ltrace PyTorch to discover all calls, add hooks as needed |
| R3 | Stream ordering semantics incorrect over network | HIGH | MEDIUM | Extensive stream ordering tests, compare results with local GPU |
| R4 | Event timing inaccurate due to network | LOW | LOW | Server computes timing locally, network doesn't affect GPU timestamps |
| R5 | NVML version mismatch between local and remote | MEDIUM | MEDIUM | Use remote NVML version for remote GPUs, local for local |
| R6 | cuGetExportTable returns internal function tables | MEDIUM | MEDIUM | Study what export tables exist, potentially no-op or forward |
| R7 | Stream-ordered allocator edge cases | MEDIUM | HIGH | Extensive testing with PyTorch's caching allocator |
| R8 | Thread safety in handle translation maps | HIGH | MEDIUM | Use concurrent hashmaps (dashmap), thorough multi-thread testing |
| R9 | Per-thread default stream variants (_ptsz) not hooked | HIGH | MEDIUM | Enumerate all _ptsz variants, hook them all via cuGetProcAddress |
| R10 | cuBLAS/cuDNN create internal contexts/streams | MEDIUM | HIGH | These libraries call driver API internally -- our hooks catch them |

---

## Section 12: Estimated Scope

| Component | New Functions | New Files (estimated) | Lines of Code (estimated) |
|-----------|--------------|----------------------|--------------------------|
| Stream management | 12 | 2 | ~800 |
| Event management | 7 | 1 | ~400 |
| NVML hook layer | 30+ | 3 | ~1500 |
| Extended memory ops | 15 | 1 | ~600 |
| Stream-ordered allocator | 10 | 1 | ~500 |
| Device extensions | 10 | 1 | ~400 |
| Execution extensions | 8 | 1 | ~300 |
| JIT linker | 5 | 1 | ~300 |
| Version table | - | 1 | ~500 |
| Texture/surface | 8 | 1 | ~400 |
| Tests | - | 5+ | ~2000 |
| **Total** | **~110** | **~18** | **~7700** |

---

## Related Documents

- [R3: CUDA Interception Strategies](../research/R3-cuda-interception.md)
- [Final Pre-Plan](../pre-planning/02-FINAL-PREPLAN.md)
- [P5: PoC Plan](P5-poc-plan.md)
- [P6: Core Transport](P6-core-transport.md) (pending)

## External References

- [CUDA Driver API Documentation](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)
- [NVML API Reference Guide](https://docs.nvidia.com/deploy/nvml-api/index.html)
- [HAMi-core Function Hooking System](https://github.com/Project-HAMi/HAMi-core/blob/main/src/cuda/hook.c)
- [CUDA Driver Entry Point Access](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DRIVER__ENTRY__POINT.html)
- [PyTorch CUDA Semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [Cricket: GPU Virtualization](https://github.com/RWTH-ACS/cricket)
- [cuGetProcAddress Version Discussion (NVIDIA Forums)](https://forums.developer.nvidia.com/t/how-to-understand-the-cudaversion-arg-in-cugetprocaddress/335722)
