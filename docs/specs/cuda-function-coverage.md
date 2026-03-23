# CUDA Function Coverage

**Created:** 2026-03-23
**Last Updated:** 2026-03-23
**Status:** Draft — reflects Phase 1 implementation

## Purpose

Complete table of all CUDA Driver API functions implemented in OutterLink, organized by category. Each entry includes the CUDA function name, the protocol `MessageType` code, and notes on behavior or limitations.

The table covers ~150 distinct protocol operations. The interposition hook table in `interpose.c` contains ~254 entries because many functions have versioned aliases (e.g., `cuStreamDestroy` and `cuStreamDestroy_v2` both map to the same message type).

---

## Message Type Code Ranges Summary

| Range | Category |
|-------|----------|
| `0x0001-0x0002` | Handshake (internal) |
| `0x0010-0x001B` | Init + Device queries |
| `0x0020-0x002F` | Context management |
| `0x0030-0x003F` | Memory: basic + async memset |
| `0x0040-0x004A` | Module + function attributes |
| `0x0047-0x0048` | Function cache/shared memory config |
| `0x0050-0x0056` | Kernel launch |
| `0x0060-0x0068` | Stream management |
| `0x0070-0x0078` | Event management |
| `0x0080-0x0083` | Occupancy |
| `0x0090-0x0091` | Peer access |
| `0x00A0-0x00A5` | Context: extended config |
| `0x00B0-0x00BB` | Pointer attributes + memory extended |
| `0x00BC-0x00CC` | Managed memory + stream-ordered pools |
| `0x00D0-0x00D4` | Callbacks (internal channel) |
| `0x00D5-0x00D9` | Library API (CUDA 12+) |
| `0x00E0-0x00ED` | JIT Linker + CUDA Graphs |
| `0x00F0` | Response (internal) |
| `0x00FF` | Error (internal) |

---

## Internal Protocol Messages

These are not CUDA functions -- they are OutterLink protocol control messages.

| Message | Code | Direction | Description |
|---------|------|-----------|-------------|
| `Handshake` | `0x0001` | Client -> Server | Initial connection, protocol version negotiation |
| `HandshakeAck` | `0x0002` | Server -> Client | Acknowledgement with assigned session_id |
| `Response` | `0x00F0` | Server -> Client | General response (4-byte CuResult + payload) |
| `Error` | `0x00FF` | Server -> Client | Protocol-level error |
| `CallbackChannelInit` | `0x00D3` | Client -> Server | Initialize dedicated callback channel |
| `CallbackChannelAck` | `0x00D4` | Server -> Client | Acknowledge callback channel |
| `CallbackReady` | `0x00D2` | Server -> Client | Notify client that a registered callback is ready to fire |

---

## 1. Initialization and Driver

| CUDA Function | Message Code | Notes |
|---------------|-------------|-------|
| `cuInit` | `0x0010` (Init) | Flags must be 0. Initializes the CUDA driver on the server. |
| `cuDriverGetVersion` | `0x0011` | Returns the driver version integer (e.g., 12040 = CUDA 12.4). |

---

## 2. Device Queries

| CUDA Function | Message Code | Notes |
|---------------|-------------|-------|
| `cuDeviceGet` | `0x0012` | Returns device ordinal for a given index. Stub returns device 0. |
| `cuDeviceGetCount` | `0x0013` | Number of CUDA-capable devices on the server. |
| `cuDeviceGetName` | `0x0014` | Human-readable device name (e.g., "NVIDIA GeForce RTX 3090"). |
| `cuDeviceGetAttribute` | `0x0015` | Query a `CUdevice_attribute` integer value. |
| `cuDeviceTotalMem_v2` | `0x0016` | Total VRAM in bytes. |
| `cuDeviceGetUuid` | `0x0017` | 16-byte device UUID. |
| `cuDeviceCanAccessPeer` | `0x0018` | Whether device A can access device B memory directly. |
| `cuDeviceGetP2PAttribute` | `0x0019` | P2P attribute between two devices. |
| `cuDeviceGetPCIBusId` | `0x001A` | PCI bus ID string (e.g., "0000:03:00.0"). |
| `cuDeviceGetByPCIBusId` | `0x001B` | Device ordinal from PCI bus ID string. |

---

## 3. Context Management

| CUDA Function | Message Code | Notes |
|---------------|-------------|-------|
| `cuCtxCreate_v2` | `0x0020` | Create a new CUDA context. Returns a synthetic handle. |
| `cuCtxDestroy_v2` | `0x0021` | Destroy a context. Removes it from session resource tracking. |
| `cuCtxSetCurrent` | `0x0022` | Set the current context for this connection. |
| `cuCtxGetCurrent` | `0x0023` | Get the current context handle. |
| `cuCtxGetDevice` | `0x0024` | Get device ordinal for the current context. |
| `cuCtxSynchronize` | `0x0025` | Block until all work in the current context completes. |
| `cuDevicePrimaryCtxRetain` | `0x0026` | Retain the primary context for a device. Idempotent: same remote handle yields same local handle. |
| `cuDevicePrimaryCtxRelease_v2` | `0x0027` | Release the primary context. |
| `cuDevicePrimaryCtxGetState` | `0x0028` | Query primary context flags and active state. |
| `cuDevicePrimaryCtxSetFlags_v2` | `0x0029` | Set primary context flags (only valid before first retain). |
| `cuDevicePrimaryCtxReset_v2` | `0x002A` | Reset the primary context, destroying all resources. |
| `cuCtxPushCurrent_v2` | `0x002B` | Push a context onto the per-thread context stack. |
| `cuCtxPopCurrent_v2` | `0x002C` | Pop the current context from the per-thread stack. |
| `cuCtxGetApiVersion` | `0x002D` | API version associated with the context. |
| `cuCtxGetLimit` | `0x002E` | Query a context resource limit (e.g., stack size). |
| `cuCtxSetLimit` | `0x002F` | Set a context resource limit. |

---

## 4. Context Extended Configuration

| CUDA Function | Message Code | Notes |
|---------------|-------------|-------|
| `cuCtxGetStreamPriorityRange` | `0x00A0` | Returns (leastPriority, greatestPriority). |
| `cuCtxGetFlags` | `0x00A1` | Returns context creation flags. |
| `cuCtxGetCacheConfig` | `0x00A2` | L1/shared memory cache preference for the context. |
| `cuCtxSetCacheConfig` | `0x00A3` | Set L1/shared memory cache preference. |
| `cuCtxGetSharedMemConfig` | `0x00A4` | Shared memory bank size configuration. |
| `cuCtxSetSharedMemConfig` | `0x00A5` | Set shared memory bank size. |

---

## 5. Peer Access

| CUDA Function | Message Code | Notes |
|---------------|-------------|-------|
| `cuCtxEnablePeerAccess` | `0x0090` | Enable P2P access between two contexts. Returns `CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED` on double-enable. |
| `cuCtxDisablePeerAccess` | `0x0091` | Disable P2P access. Returns `CUDA_ERROR_PEER_ACCESS_NOT_ENABLED` if not enabled. |

---

## 6. Memory: Basic Allocation and Transfer

| CUDA Function | Message Code | Notes |
|---------------|-------------|-------|
| `cuMemAlloc_v2` | `0x0030` | Allocate device memory. Returns a synthetic `CUdeviceptr` handle. Zero-size allocations return `CUDA_SUCCESS` without allocating. |
| `cuMemFree_v2` | `0x0031` | Free device memory. Zero pointer is a no-op (`CUDA_SUCCESS`). |
| `cuMemcpyHtoD_v2` | `0x0032` | Host-to-device copy. Data travels: host buffer -> TCP payload -> server -> GPU. Zero-size is a no-op. |
| `cuMemcpyDtoH_v2` | `0x0033` | Device-to-host copy. Data travels: GPU -> server -> TCP payload -> host buffer. Zero-size is a no-op. |
| `cuMemcpyDtoD_v2` | `0x0034` | Device-to-device copy (both pointers on the remote GPU). Zero-size is a no-op. |
| `cuMemGetInfo_v2` | `0x0035` | Returns (free_bytes, total_bytes) on the server GPU. |
| `cuMemAllocHost_v2` | `0x0036` | Allocate page-locked (pinned) host memory on the server. Used for fast transfers. |
| `cuMemFreeHost` | `0x0037` | Free page-locked host memory. Null pointer is a no-op. |
| `cuMemcpyHtoDAsync_v2` | `0x0038` | Asynchronous HtoD copy on a stream. |
| `cuMemcpyDtoHAsync_v2` | `0x0039` | Asynchronous DtoH copy on a stream. |
| `cuMemsetD8_v2` | `0x003A` | Fill device memory with a byte value. Zero-size is a no-op. |
| `cuMemsetD32_v2` | `0x003B` | Fill device memory with a 32-bit value. Zero-size is a no-op. |
| `cuMemsetD8Async` | `0x003C` | Async fill with byte value on a stream. |
| `cuMemsetD32Async` | `0x003D` | Async fill with 32-bit value on a stream. |
| `cuMemsetD16_v2` | `0x003E` | Fill device memory with a 16-bit value. |
| `cuMemsetD16Async` | `0x003F` | Async fill with 16-bit value on a stream. |

---

## 7. Memory: Extended

| CUDA Function | Message Code | Notes |
|---------------|-------------|-------|
| `cuMemGetAddressRange_v2` | `0x00B2` | Returns (base_ptr, allocation_size) for a device pointer. |
| `cuMemHostGetDevicePointer_v2` | `0x00B3` | Get the device-mapped pointer for a pinned host allocation. |
| `cuMemHostGetFlags` | `0x00B4` | Query flags for a pinned host allocation. |
| `cuMemHostRegister_v2` | `0x00B5` | Page-lock and register existing host memory. |
| `cuMemHostUnregister` | `0x00B6` | Unregister and unpin registered host memory. |
| `cuMemcpy` | `0x00B7` | Generic memcpy (auto-detects direction). |
| `cuMemcpyAsync` | `0x00B8` | Async generic memcpy on a stream. |
| `cuMemcpyDtoDAsync_v2` | `0x00B9` | Async device-to-device copy on a stream. |
| `cuMemHostAlloc` | `0x00BA` | Allocate pinned host memory with specific flags. |
| `cuMemAllocPitch_v2` | `0x00BB` | Allocate 2D device memory with pitch alignment. Returns (ptr, pitch). |

---

## 8. Memory: Managed / Unified

| CUDA Function | Message Code | Notes |
|---------------|-------------|-------|
| `cuMemAllocManaged` | `0x00BC` | Allocate unified memory accessible from CPU and GPU. |
| `cuMemPrefetchAsync` | `0x00BD` | Prefetch unified memory to a specific device on a stream. |
| `cuMemAdvise` | `0x00BE` | Provide memory access hints for unified memory. |
| `cuMemRangeGetAttribute` | `0x00BF` | Query a single attribute of a unified memory range. |
| `cuMemRangeGetAttributes` | `0x00CC` | Query multiple attributes of a unified memory range. |

---

## 9. Memory: Stream-Ordered Pools (CUDA 11.2+)

| CUDA Function | Message Code | Notes |
|---------------|-------------|-------|
| `cuMemAllocAsync` | `0x00C0` | Allocate device memory from the default pool, ordered on a stream. |
| `cuMemFreeAsync` | `0x00C1` | Free stream-ordered memory back to its pool. |
| `cuDeviceGetDefaultMemPool` | `0x00C2` | Get the default memory pool for a device. |
| `cuMemPoolCreate` | `0x00C3` | Create a new memory pool with specified properties. |
| `cuMemPoolDestroy` | `0x00C4` | Destroy a memory pool and free all its memory. |
| `cuMemPoolGetAttribute` | `0x00C5` | Query a memory pool attribute. |
| `cuMemPoolSetAttribute` | `0x00C6` | Set a memory pool attribute. |
| `cuMemPoolTrimTo` | `0x00C7` | Release unused memory from a pool back to the OS, keeping at least `minBytesToKeep`. |
| `cuMemAllocFromPoolAsync` | `0x00C8` | Allocate from a specific memory pool, stream-ordered. |
| `cuDeviceGetMemPool` | `0x00C9` | Get the current default memory pool for a device. |
| `cuDeviceSetMemPool` | `0x00CA` | Set the default memory pool for a device. |
| `cuMemGetAllocationGranularity` | `0x00CB` | Query the required allocation granularity for a memory pool. |

---

## 10. Pointer Attributes

| CUDA Function | Message Code | Notes |
|---------------|-------------|-------|
| `cuPointerGetAttribute` | `0x00B0` | Query a single attribute of a device pointer (e.g., memory type, context). |
| `cuPointerGetAttributes` | `0x00B1` | Query multiple pointer attributes in one call. |

---

## 11. Module Loading and Management

| CUDA Function | Message Code | Notes |
|---------------|-------------|-------|
| `cuModuleLoad` | `0x0049` | Load a module from a file path on the server filesystem. |
| `cuModuleLoadData` | `0x0040` | Load a module from a PTX or cubin data buffer. |
| `cuModuleLoadFatBinary` | `0x004A` | Load a fatbinary (multi-architecture) module. |
| `cuModuleLoadDataEx` | `0x0044` | Load a module from data with JIT compiler options. |
| `cuModuleUnload` | `0x0041` | Unload a module and free its GPU resources. |
| `cuModuleGetFunction` | `0x0042` | Look up a kernel function by name in a loaded module. |
| `cuModuleGetGlobal_v2` | `0x0043` | Look up a global variable by name. Returns (device_ptr, size_in_bytes). |

---

## 12. Function / Kernel Attributes and Configuration

| CUDA Function | Message Code | Notes |
|---------------|-------------|-------|
| `cuFuncGetAttribute` | `0x0045` | Query a `CUfunction_attribute` (e.g., register count, shared memory). |
| `cuFuncSetAttribute` | `0x0046` | Set a function attribute (e.g., max dynamic shared memory). |
| `cuFuncSetCacheConfig` | `0x0047` | Set L1/shared memory cache preference for a specific function. |
| `cuFuncSetSharedMemConfig` | `0x0048` | Set shared memory bank size for a specific function. |

---

## 13. Kernel Launch

| CUDA Function | Message Code | Notes |
|---------------|-------------|-------|
| `cuLaunchKernel` | `0x0050` | Launch a kernel with grid/block dimensions, shared memory, and parameters. The parameter buffer (extra or params array) is serialized in the payload. |
| `cuLaunchCooperativeKernel` | `0x0055` | Launch a cooperative kernel where all thread blocks can synchronize. |
| `cuLaunchKernelEx` | `0x0056` | Extended kernel launch (CUDA 12+) with `CUlaunchConfig` struct. |

---

## 14. Stream Management

| CUDA Function | Message Code | Notes |
|---------------|-------------|-------|
| `cuStreamCreate` | `0x0060` | Create a new CUDA stream. Returns a synthetic stream handle. |
| `cuStreamDestroy_v2` | `0x0061` | Destroy a stream. |
| `cuStreamSynchronize` | `0x0062` | Block until all operations on the stream complete. Also waits locally for all pending callbacks on that stream (two-phase). |
| `cuStreamWaitEvent` | `0x0063` | Make a stream wait on an event before executing subsequent operations. |
| `cuStreamQuery` | `0x0064` | Non-blocking check if all stream work is complete. Returns `CUDA_SUCCESS` or `CUDA_ERROR_NOT_READY`. |
| `cuStreamCreateWithPriority` | `0x0065` | Create a stream with a specific scheduling priority. |
| `cuStreamGetPriority` | `0x0066` | Query the priority of an existing stream. |
| `cuStreamGetFlags_v2` | `0x0067` | Query the flags a stream was created with. |
| `cuStreamGetCtx_v2` | `0x0068` | Get the context associated with a stream. |

---

## 15. Stream Callbacks

| CUDA Function | Message Code | Notes |
|---------------|-------------|-------|
| `cuStreamAddCallback` | `0x00D0` (StreamAddCallback) | Register a host callback to fire when the stream reaches this point. The C function pointer and userData stay on the client; only callback_id crosses the wire. Requires callback channel. |
| `cuLaunchHostFunc` | `0x00D1` (LaunchHostFunc) | Enqueue a host function on a stream (CUDA 10.0+). Same wire protocol as `cuStreamAddCallback`. |

---

## 16. Event Management

| CUDA Function | Message Code | Notes |
|---------------|-------------|-------|
| `cuEventCreate` | `0x0070` | Create a CUDA event. Returns a synthetic event handle. |
| `cuEventDestroy_v2` | `0x0071` | Destroy an event. |
| `cuEventRecord` | `0x0072` | Record an event in a stream. |
| `cuEventRecordWithFlags` | `0x0078` | Record an event with additional flags (CUDA 11.1+). |
| `cuEventSynchronize` | `0x0073` | Block until an event completes. |
| `cuEventElapsedTime` | `0x0074` | Get elapsed time in milliseconds between two events. |
| `cuEventQuery` | `0x0075` | Non-blocking check if an event has completed. |

---

## 17. Occupancy

| CUDA Function | Message Code | Notes |
|---------------|-------------|-------|
| `cuOccupancyMaxActiveBlocksPerMultiprocessor` | `0x0080` | Compute maximum resident blocks per SM for given block size and shared mem. |
| `cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` | `0x0081` | Same as above with additional occupancy flags. |
| `cuOccupancyMaxPotentialBlockSize` | `0x0082` | Find the block size that maximizes occupancy. Returns (minGridSize, blockSize). |
| `cuOccupancyMaxPotentialBlockSizeWithFlags` | `0x0083` | Same as above with additional flags. |

---

## 18. JIT Linker

The JIT linker allows combining multiple PTX/cubin modules at runtime before producing a final cubin. Used by frameworks that generate kernels at launch time.

| CUDA Function | Message Code | Notes |
|---------------|-------------|-------|
| `cuLinkCreate_v2` | `0x00E0` | Create a JIT linker state with compiler options. Returns a synthetic `CUlinkState` handle. |
| `cuLinkAddData_v2` | `0x00E1` | Add a data buffer (PTX, cubin, FATBIN, etc.) to the linker. |
| `cuLinkAddFile_v2` | `0x00E2` | Add a file from the server filesystem to the linker. |
| `cuLinkComplete` | `0x00E3` | Complete linking. Returns the cubin data pointer and size. |
| `cuLinkDestroy` | `0x00E4` | Destroy the linker state and free resources. |

Note: `cuLinkCreate` (without `_v2`) and `cuLinkAddData` / `cuLinkAddFile` (without `_v2`) are aliased to the same hooks.

---

## 19. Library API (CUDA 12+)

The Library API provides a higher-level way to load and manage kernel code, separating the library (code container) from the module (GPU context binding). Required for `cuLaunchKernelEx` workflows in CUDA 12+.

| CUDA Function | Message Code | Notes |
|---------------|-------------|-------|
| `cuLibraryLoadData` | `0x00D5` | Load a library from a data buffer (PTX, cubin, or FATBIN). Returns a synthetic `CUlibrary` handle. |
| `cuLibraryUnload` | `0x00D6` | Unload a library and release its GPU resources. |
| `cuLibraryGetKernel` | `0x00D7` | Retrieve a `CUkernel` handle by name from a library. |
| `cuLibraryGetModule` | `0x00D8` | Get the `CUmodule` associated with a library (for backwards compat). |
| `cuKernelGetFunction` | `0x00D9` | Get a `CUfunction` from a `CUkernel` for use with `cuLaunchKernel`. |

---

## 20. CUDA Graph API

CUDA Graphs allow recording a sequence of GPU operations and replaying them with minimal CPU overhead. Used by `torch.compile` with `reduce-overhead` mode.

| CUDA Function | Message Code | Notes |
|---------------|-------------|-------|
| `cuStreamBeginCapture_v2` | `0x00E5` | Begin recording operations into a graph on a stream. |
| `cuStreamEndCapture` | `0x00E6` | End recording and return the captured graph. Returns a synthetic `CUgraph` handle. |
| `cuStreamIsCapturing` | `0x00E7` | Query whether a stream is currently in capture mode. |
| `cuStreamGetCaptureInfo_v2` | `0x00E8` | Get capture status and sequence number for a stream. |
| `cuGraphCreate` | `0x00E9` | Create an empty CUDA graph. |
| `cuGraphInstantiate` | `0x00EA` | Instantiate a graph into an executable graph. Returns a synthetic `CUgraphExec` handle. |
| `cuGraphLaunch` | `0x00EB` | Launch an executable graph on a stream. |
| `cuGraphExecDestroy` | `0x00EC` | Destroy an executable graph. |
| `cuGraphDestroy` | `0x00ED` | Destroy a graph. |

Note: `cuGraphInstantiate_v2` and `cuGraphInstantiateWithFlags` are aliased to the same hooks as `cuGraphInstantiate`.

---

## 21. Error Utilities

These are handled locally by the client without a round-trip to the server (stub responses).

| CUDA Function | Notes |
|---------------|-------|
| `cuGetErrorName` | Returns a string name for a `CUresult` code. Handled locally. |
| `cuGetErrorString` | Returns a human-readable description of a `CUresult` code. Handled locally. |

---

## 22. Export Table and Proc Address (Internal)

These intercept the mechanisms that applications and libraries use to discover CUDA functions, ensuring all CUDA calls route through OutterLink.

| CUDA Function | Notes |
|---------------|-------|
| `cuGetProcAddress` | Hooks the CUDA driver's own function discovery mechanism (CUDA 11.3+). Returns OutterLink hook pointers for known functions. |
| `cuGetProcAddress_v2` | CUDA 12 version of the same. |
| `cuGetExportTable` | Passthrough to real `libcuda.so`. Used for internal driver vtables that do not need interception. |

---

## CUDA Functions Not Yet Implemented

The following categories are not yet covered. Applications using these functions will fall through to the real `libcuda.so` (if available on the client) or receive an error.

| Category | Example Functions | Phase |
|----------|-------------------|-------|
| NVML (GPU management) | `nvmlDeviceGetCount`, `nvmlDeviceGetMemoryInfo` | Future |
| External memory interop | `cuImportExternalMemory`, `cuExternalMemoryGetMappedBuffer` | Future |
| External semaphore interop | `cuImportExternalSemaphore`, `cuSignalExternalSemaphoresAsync` | Future |
| Virtual memory management | `cuMemAddressReserve`, `cuMemCreate`, `cuMemMap` | Future |
| Multi-GPU task graphs | `cuGraphAddMemcpyNode`, `cuGraphAddKernelNode`, etc. | Future |
| IPC memory handles | `cuIpcGetMemHandle`, `cuIpcOpenMemHandle` | Future |
| Surface / texture references | `cuTexRefSetArray`, `cuSurfRefSetArray` (deprecated) | Low priority |
| CUDA Arrays (2D/3D) | `cuArrayCreate`, `cuArray3DCreate`, `cuMemcpy2D` | Future |
| Green contexts (CUDA 12.4+) | `cuGreenCtxCreate`, `cuDeviceGetGreenCtxAttribute` | Future |

---

## Related Documents

- [System Architecture](../architecture/01-system-architecture.md)
- [Installation Guide](../guides/01-installation.md)
- [Project Vision](../architecture/00-project-vision.md)

## Open Questions

- [ ] Are there CUDA functions used by cuDNN / cuBLAS that are not in the current hook table?
- [ ] Does `cuGetExportTable` passthrough cause any correctness issues with libraries that rely on the vtable for function discovery?
- [ ] Should `cuGetErrorName` / `cuGetErrorString` be wired to the server (so errors reported by the server use driver-accurate strings)?
- [ ] What is the behavior when an unimplemented function is called while `libcuda.so` is NOT present on the client machine?
