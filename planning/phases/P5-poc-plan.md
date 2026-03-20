# P5: Phase 1 - Proof of Concept

**Created:** 2026-03-19
**Last Updated:** 2026-03-19
**Status:** Plan
**Priority:** HIGH - First Code Phase

## Purpose

Define the exact implementation plan for OutterLink's proof of concept: intercept a minimal set of CUDA Driver API calls on the client, forward them over TCP to a server daemon, execute them on a real GPU, and return results. By the end of this phase, a simple CUDA program compiles and runs unmodified through OutterLink, allocating memory, loading a kernel, and executing it on a remote GPU.

## Goal

A CUDA application on PC-A (no GPU needed) runs a vector addition kernel on PC-B's GPU via OutterLink, with correct results returned. Zero application code changes.

## Milestone

- `LD_PRELOAD=liboutterlink_client.so ./vector_add` runs on a machine without a GPU
- The vector addition kernel executes on a remote GPU and returns correct results
- Device queries (`cuDeviceGetName`, `cuDeviceTotalMem`, etc.) return the remote GPU's real properties
- Memory allocation, host-to-device copy, kernel launch, device-to-host copy, and free all work
- Round-trip latency for a single `cuLaunchKernel` is measured and logged

## Prerequisites

- [x] P1: GitHub Repository Setup (plan ready)
- [x] P2: Development Environment (plan ready)
- [ ] P4: Project Skeleton (Rust workspace with crate structure)
- [ ] At least one machine with a working NVIDIA GPU + CUDA toolkit
- [ ] Two machines (or same machine for loopback testing) with TCP connectivity

---

## 1. CUDA Functions to Intercept in PoC

### 1.1 Initialization (2 functions)

#### `cuInit`
```c
CUresult cuInit(unsigned int Flags);
```
- **Client:** Sends `Init { flags: u32 }` to server. Blocks until response.
- **Serialized:** `{ op: INIT, flags: u32 }`
- **Server:** Calls real `cuInit(Flags)`. Returns result code.
- **Response:** `{ result: CUresult }`
- **Notes:** Must be called before any other CUDA call. The client also establishes the TCP connection here if not already connected.

#### `cuDriverGetVersion`
```c
CUresult cuDriverGetVersion(int *driverVersion);
```
- **Client:** Sends `DriverGetVersion {}` to server.
- **Serialized:** `{ op: DRIVER_GET_VERSION }`
- **Server:** Calls real `cuDriverGetVersion(&version)`. Returns version.
- **Response:** `{ result: CUresult, version: i32 }`

### 1.2 Device Management (6 functions)

#### `cuDeviceGetCount`
```c
CUresult cuDeviceGetCount(int *count);
```
- **Client:** Sends `DeviceGetCount {}` to server.
- **Serialized:** `{ op: DEVICE_GET_COUNT }`
- **Server:** Calls real `cuDeviceGetCount(&count)`. Returns count.
- **Response:** `{ result: CUresult, count: i32 }`
- **Notes:** In PoC, returns the server's actual GPU count. Multi-server aggregation is Phase 6.

#### `cuDeviceGet`
```c
CUresult cuDeviceGet(CUdevice *device, int ordinal);
```
- **Client:** Sends `DeviceGet { ordinal: i32 }`. Stores returned device handle in local handle table.
- **Serialized:** `{ op: DEVICE_GET, ordinal: i32 }`
- **Server:** Calls real `cuDeviceGet(&dev, ordinal)`. Returns device handle.
- **Response:** `{ result: CUresult, device: i32 }`
- **Notes:** `CUdevice` is a plain `int` (not opaque). No handle translation needed -- pass through directly.

#### `cuDeviceGetName`
```c
CUresult cuDeviceGetName(char *name, int len, CUdevice dev);
```
- **Client:** Sends `DeviceGetName { len: i32, device: i32 }`. Copies returned string to caller's buffer.
- **Serialized:** `{ op: DEVICE_GET_NAME, len: i32, device: i32 }`
- **Server:** Calls real `cuDeviceGetName(buf, len, dev)`. Returns string.
- **Response:** `{ result: CUresult, name: String (null-terminated, max len bytes) }`

#### `cuDeviceGetAttribute`
```c
CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
```
- **Client:** Sends `DeviceGetAttribute { attrib: i32, device: i32 }`.
- **Serialized:** `{ op: DEVICE_GET_ATTRIBUTE, attrib: i32, device: i32 }`
- **Server:** Calls real `cuDeviceGetAttribute(&val, attrib, dev)`. Returns value.
- **Response:** `{ result: CUresult, value: i32 }`

#### `cuDeviceTotalMem_v2`
```c
CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev);
```
- **Client:** Sends `DeviceTotalMem { device: i32 }`.
- **Serialized:** `{ op: DEVICE_TOTAL_MEM, device: i32 }`
- **Server:** Calls real `cuDeviceTotalMem_v2(&bytes, dev)`. Returns size.
- **Response:** `{ result: CUresult, bytes: u64 }`

#### `cuDeviceGetUuid`
```c
CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev);
```
- **Client:** Sends `DeviceGetUuid { device: i32 }`.
- **Serialized:** `{ op: DEVICE_GET_UUID, device: i32 }`
- **Server:** Calls real `cuDeviceGetUuid(&uuid, dev)`. Returns 16-byte UUID.
- **Response:** `{ result: CUresult, uuid: [u8; 16] }`

### 1.3 Context Management (5 functions)

#### `cuCtxCreate_v2`
```c
CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
```
- **Client:** Sends `CtxCreate { flags: u32, device: i32 }`. Receives server-side context handle. Creates local synthetic `CUcontext` and stores mapping: `local_ctx -> server_ctx_id`.
- **Serialized:** `{ op: CTX_CREATE, flags: u32, device: i32 }`
- **Server:** Calls real `cuCtxCreate_v2(&ctx, flags, dev)`. Stores context, returns a context ID (u64).
- **Response:** `{ result: CUresult, ctx_id: u64 }`
- **Handle translation:** Client generates a synthetic `CUcontext` pointer (incrementing counter cast to pointer). Maps it to `ctx_id`.

#### `cuCtxDestroy_v2`
```c
CUresult cuCtxDestroy_v2(CUcontext ctx);
```
- **Client:** Looks up `ctx_id` from local handle table. Sends `CtxDestroy { ctx_id: u64 }`. Removes mapping.
- **Serialized:** `{ op: CTX_DESTROY, ctx_id: u64 }`
- **Server:** Looks up real `CUcontext` by `ctx_id`. Calls real `cuCtxDestroy_v2(ctx)`. Removes from server table.
- **Response:** `{ result: CUresult }`

#### `cuCtxSetCurrent`
```c
CUresult cuCtxSetCurrent(CUcontext ctx);
```
- **Client:** Looks up `ctx_id`. Sends `CtxSetCurrent { ctx_id: u64 }`. Updates thread-local current context.
- **Serialized:** `{ op: CTX_SET_CURRENT, ctx_id: u64 }`
- **Server:** Calls real `cuCtxSetCurrent(ctx)` on the server's worker thread for this client.
- **Response:** `{ result: CUresult }`

#### `cuCtxGetCurrent`
```c
CUresult cuCtxGetCurrent(CUcontext *pctx);
```
- **Client:** Returns the thread-local current context from the local handle table. NO network call needed.
- **Notes:** This is a client-only operation. The client tracks which context is current per thread.

#### `cuCtxGetDevice`
```c
CUresult cuCtxGetDevice(CUdevice *device);
```
- **Client:** Sends `CtxGetDevice {}`.
- **Serialized:** `{ op: CTX_GET_DEVICE }`
- **Server:** Calls real `cuCtxGetDevice(&dev)`. Returns device ordinal.
- **Response:** `{ result: CUresult, device: i32 }`

### 1.4 Memory Management (4 functions)

#### `cuMemAlloc_v2`
```c
CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize);
```
- **Client:** Sends `MemAlloc { size: u64 }`. Receives server-side device pointer. Creates local synthetic `CUdeviceptr` and stores mapping: `local_dptr -> server_dptr`.
- **Serialized:** `{ op: MEM_ALLOC, size: u64 }`
- **Server:** Calls real `cuMemAlloc_v2(&dptr, bytesize)`. Returns the real device pointer value.
- **Response:** `{ result: CUresult, dptr: u64 }`
- **Handle translation:** Client uses an incrementing counter starting at `0x1000_0000_0000` for synthetic device pointers. Maps synthetic -> real server pointer.

#### `cuMemFree_v2`
```c
CUresult cuMemFree_v2(CUdeviceptr dptr);
```
- **Client:** Looks up `server_dptr` from local handle table. Sends `MemFree { dptr: u64 }`. Removes mapping.
- **Serialized:** `{ op: MEM_FREE, dptr: u64 }`
- **Server:** Calls real `cuMemFree_v2(dptr)`.
- **Response:** `{ result: CUresult }`

#### `cuMemcpyHtoD_v2`
```c
CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
```
- **Client:** Looks up `server_dptr` for `dstDevice`. Reads `ByteCount` bytes from `srcHost`. Sends `MemcpyHtoD { dptr: u64, size: u64, data: Vec<u8> }`.
- **Serialized:** `{ op: MEMCPY_HTOD, dptr: u64, size: u64 }` followed by `size` bytes of raw data.
- **Server:** Calls real `cuMemcpyHtoD_v2(dptr, data.as_ptr(), size)`.
- **Response:** `{ result: CUresult }`
- **Notes:** For PoC, the entire payload is sent in one message. No streaming or chunking. The data bytes follow the header directly in the TCP stream.

#### `cuMemcpyDtoH_v2`
```c
CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
```
- **Client:** Looks up `server_dptr` for `srcDevice`. Sends `MemcpyDtoH { dptr: u64, size: u64 }`. Receives data bytes. Copies into `dstHost`.
- **Serialized:** `{ op: MEMCPY_DTOH, dptr: u64, size: u64 }`
- **Server:** Allocates temp buffer. Calls real `cuMemcpyDtoH_v2(buf, dptr, size)`. Returns data.
- **Response:** `{ result: CUresult, size: u64 }` followed by `size` bytes of raw data.

### 1.5 Module Management (5 functions)

#### `cuModuleLoadData`
```c
CUresult cuModuleLoadData(CUmodule *module, const void *image);
```
- **Client:** Reads the module image (PTX or cubin). The image is null-terminated for PTX, or a cubin blob. Determines size by scanning for null (PTX) or by reading the ELF header (cubin). Sends `ModuleLoadData { image: Vec<u8> }`. Receives server-side module handle. Creates local synthetic `CUmodule` and stores mapping.
- **Serialized:** `{ op: MODULE_LOAD_DATA, image_size: u64 }` followed by `image_size` bytes of module data.
- **Server:** Calls real `cuModuleLoadData(&mod, image)`. Stores module, returns module ID.
- **Response:** `{ result: CUresult, module_id: u64 }`
- **Handle translation:** Same incrementing counter pattern as contexts.
- **Notes:** For PTX, the size is `strlen(image) + 1`. For cubin, read the ELF header's `e_shoff + (e_shnum * e_shentsize)` or just scan for the blob size. In PoC, we can send up to the first null for PTX or use a fixed size field.

#### `cuModuleLoadDataEx`
```c
CUresult cuModuleLoadDataEx(CUmodule *module, const void *image,
                            unsigned int numOptions, CUjit_option *options,
                            void **optionValues);
```
- **Client:** Same as `cuModuleLoadData` but also serializes JIT options. For PoC, we forward the image and ignore JIT options (pass `numOptions=0` to server).
- **Serialized:** `{ op: MODULE_LOAD_DATA_EX, image_size: u64, num_options: u32 }` followed by image data. Options serialization deferred to Phase 2.
- **Server:** Calls real `cuModuleLoadData(&mod, image)` (ignoring options for PoC).
- **Response:** `{ result: CUresult, module_id: u64 }`

#### `cuModuleUnload`
```c
CUresult cuModuleUnload(CUmodule hmod);
```
- **Client:** Looks up `module_id`. Sends `ModuleUnload { module_id: u64 }`. Removes mapping. Also removes all function handles associated with this module.
- **Serialized:** `{ op: MODULE_UNLOAD, module_id: u64 }`
- **Server:** Calls real `cuModuleUnload(mod)`. Removes from server table.
- **Response:** `{ result: CUresult }`

#### `cuModuleGetFunction`
```c
CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
```
- **Client:** Looks up `module_id`. Sends `ModuleGetFunction { module_id: u64, name: String }`. Receives server-side function handle. Creates local synthetic `CUfunction` and stores mapping.
- **Serialized:** `{ op: MODULE_GET_FUNCTION, module_id: u64, name_len: u32 }` followed by function name bytes.
- **Server:** Calls real `cuModuleGetFunction(&func, mod, name)`. Stores function, returns function ID.
- **Response:** `{ result: CUresult, function_id: u64 }`

#### `cuModuleGetGlobal_v2`
```c
CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name);
```
- **Client:** Looks up `module_id`. Sends `ModuleGetGlobal { module_id: u64, name: String }`. Receives server-side device pointer and size. Creates synthetic `CUdeviceptr` mapping.
- **Serialized:** `{ op: MODULE_GET_GLOBAL, module_id: u64, name_len: u32 }` followed by name bytes.
- **Server:** Calls real `cuModuleGetGlobal_v2(&dptr, &bytes, mod, name)`. Returns pointer and size.
- **Response:** `{ result: CUresult, dptr: u64, bytes: u64 }`

### 1.6 Kernel Launch (1 function)

#### `cuLaunchKernel`
```c
CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream,
                        void **kernelParams, void **extra);
```
- **Client:** Looks up `function_id` for `f`. For PoC, `hStream` is always `0` (default stream). Serializes kernel parameters. Sends the launch request. Blocks until kernel completes (synchronous in PoC).
- **Serialized:** `{ op: LAUNCH_KERNEL, function_id: u64, grid: [u32; 3], block: [u32; 3], shared_mem: u32, stream_id: u64 }` followed by serialized kernel parameters (see below).
- **Server:** Reconstructs `kernelParams` array. Calls real `cuLaunchKernel(func, gx, gy, gz, bx, by, bz, shared, stream, params, NULL)`. Then calls `cuCtxSynchronize()` to wait for completion. Returns result.
- **Response:** `{ result: CUresult }`

**Kernel Parameter Serialization (PoC approach):**

The `kernelParams` array is `void**` -- an array of pointers to each argument. The challenge: we don't know the types or sizes from the function signature alone.

For PoC, we use the `extra` parameter with `CU_LAUNCH_PARAM_BUFFER_POINTER` + `CU_LAUNCH_PARAM_BUFFER_SIZE`:
- The client tries `extra` first. If the app passes `extra` (as PyTorch and many frameworks do), it contains a flat byte buffer of all kernel args plus a size field.
- If `kernelParams` is used instead, the client must determine parameter sizes. For PoC: query `cuFuncGetAttribute` for `CU_FUNC_ATTRIBUTE_PARAM_SIZE` (total parameter byte size from the server), then send the raw bytes pointed to by `kernelParams` entries. We serialize as: `{ num_params: u32, total_size: u32 }` followed by raw bytes for each param, where the server reconstructs the `void**` array from the known parameter layout.

**Simplified PoC approach:** Since our test program uses simple scalar + pointer args with known sizes, the client sends `{ param_buffer_size: u32 }` followed by the raw concatenated parameter bytes (properly aligned). The server uses `CU_LAUNCH_PARAM_BUFFER_POINTER` to pass the flat buffer.

---

## 2. Client-Server Protocol for PoC

### 2.1 Wire Format

All messages use a simple length-prefixed binary format over TCP:

```
+----------+----------+----------+------------------+
| magic(4) | len(4)   | op(2)    | payload(len - 2) |
+----------+----------+----------+------------------+

magic:   0x4F4C4E4B ("OLNK" - OutterLink)
len:     u32 LE - byte count of op + payload (does NOT include magic or len fields)
op:      u16 LE - operation code
payload: variable length, depends on op
```

For responses:
```
+----------+----------+----------+----------+------------------+
| magic(4) | len(4)   | op(2)    | result(4)| payload(len - 6) |
+----------+----------+----------+----------+------------------+

result:  i32 LE - CUresult value (0 = CUDA_SUCCESS)
```

### 2.2 Operation Codes

```rust
#[repr(u16)]
pub enum OpCode {
    // Init
    Init                = 0x0001,
    DriverGetVersion    = 0x0002,

    // Device
    DeviceGetCount      = 0x0010,
    DeviceGet           = 0x0011,
    DeviceGetName       = 0x0012,
    DeviceGetAttribute  = 0x0013,
    DeviceTotalMem      = 0x0014,
    DeviceGetUuid       = 0x0015,

    // Context
    CtxCreate           = 0x0020,
    CtxDestroy          = 0x0021,
    CtxSetCurrent       = 0x0022,
    CtxGetCurrent       = 0x0023,  // Client-only, not sent
    CtxGetDevice        = 0x0024,

    // Memory
    MemAlloc            = 0x0030,
    MemFree             = 0x0031,
    MemcpyHtoD          = 0x0032,
    MemcpyDtoH          = 0x0033,

    // Module
    ModuleLoadData      = 0x0040,
    ModuleLoadDataEx    = 0x0041,
    ModuleUnload        = 0x0042,
    ModuleGetFunction   = 0x0043,
    ModuleGetGlobal     = 0x0044,

    // Kernel
    LaunchKernel        = 0x0050,
}
```

### 2.3 Message Definitions (Request/Response)

All numeric fields are little-endian. Strings are length-prefixed (u32 len + bytes, no null terminator in wire format).

| Op | Request Payload | Response Payload |
|----|----------------|------------------|
| `Init` | `flags: u32` | (none beyond result) |
| `DriverGetVersion` | (empty) | `version: i32` |
| `DeviceGetCount` | (empty) | `count: i32` |
| `DeviceGet` | `ordinal: i32` | `device: i32` |
| `DeviceGetName` | `len: i32, device: i32` | `name_len: u32, name: [u8]` |
| `DeviceGetAttribute` | `attrib: i32, device: i32` | `value: i32` |
| `DeviceTotalMem` | `device: i32` | `bytes: u64` |
| `DeviceGetUuid` | `device: i32` | `uuid: [u8; 16]` |
| `CtxCreate` | `flags: u32, device: i32` | `ctx_id: u64` |
| `CtxDestroy` | `ctx_id: u64` | (none beyond result) |
| `CtxSetCurrent` | `ctx_id: u64` | (none beyond result) |
| `CtxGetDevice` | (empty) | `device: i32` |
| `MemAlloc` | `size: u64` | `dptr: u64` |
| `MemFree` | `dptr: u64` | (none beyond result) |
| `MemcpyHtoD` | `dptr: u64, size: u64` + `size` bytes data | (none beyond result) |
| `MemcpyDtoH` | `dptr: u64, size: u64` | `size: u64` + `size` bytes data |
| `ModuleLoadData` | `image_size: u64` + image bytes | `module_id: u64` |
| `ModuleLoadDataEx` | `image_size: u64, num_options: u32` + image bytes | `module_id: u64` |
| `ModuleUnload` | `module_id: u64` | (none beyond result) |
| `ModuleGetFunction` | `module_id: u64, name_len: u32` + name bytes | `function_id: u64` |
| `ModuleGetGlobal` | `module_id: u64, name_len: u32` + name bytes | `dptr: u64, bytes: u64` |
| `LaunchKernel` | `function_id: u64, grid: [u32;3], block: [u32;3], shared_mem: u32, stream_id: u64, param_size: u32` + param bytes | (none beyond result) |

### 2.4 Connection Lifecycle

```
1. Client connects TCP to server (e.g., 192.168.100.2:9370)
2. Client sends HANDSHAKE: { magic, protocol_version: u32, client_id: [u8; 16] }
3. Server responds: { magic, protocol_version: u32, server_id: [u8; 16], num_gpus: u32 }
4. Client sends CUDA calls as request messages
5. Server processes each request sequentially (single worker per client in PoC)
6. On disconnect, server cleans up all resources (contexts, modules, memory) for that client
```

Default port: `9370` (OL -> 0x4F4C -> 20300, but 9370 is shorter and memorable: "93" for 3090, "70" for RTX).

---

## 3. Handle Translation in PoC

### 3.1 Overview

The client must translate between local synthetic handles (returned to the application) and remote real handles (on the server). The server must translate between its internal IDs and real CUDA handles.

### 3.2 Client-Side Handle Tables

```rust
/// Thread-safe handle translation tables (client side)
pub struct HandleTables {
    /// CUcontext: synthetic pointer -> server ctx_id
    contexts: RwLock<HashMap<usize, u64>>,

    /// CUdeviceptr: synthetic device pointer -> server device pointer
    device_ptrs: RwLock<HashMap<u64, u64>>,

    /// CUmodule: synthetic pointer -> server module_id
    modules: RwLock<HashMap<usize, u64>>,

    /// CUfunction: synthetic pointer -> server function_id
    functions: RwLock<HashMap<usize, u64>>,

    /// Thread-local current context (per-thread)
    current_ctx: ThreadLocal<Cell<usize>>,

    /// Next synthetic handle value
    next_handle: AtomicU64,
}
```

**Synthetic handle generation:**
- Contexts: `0xCC00_0000_0000_0001`, `0xCC00_0000_0000_0002`, ...
- Device pointers: `0x1000_0000_0000_0000`, `0x1000_0000_0000_1000`, ... (page-aligned)
- Modules: `0xDD00_0000_0000_0001`, `0xDD00_0000_0000_0002`, ...
- Functions: `0xEE00_0000_0000_0001`, `0xEE00_0000_0000_0002`, ...

Using distinct prefixes ensures we can detect if the wrong handle type is passed (debug aid).

### 3.3 Server-Side Handle Tables

```rust
/// Server-side handle tables (per client connection)
pub struct ServerHandles {
    /// ctx_id -> real CUcontext
    contexts: HashMap<u64, CUcontext>,

    /// The server uses real CUdeviceptr values directly (no translation needed)
    /// The client sends server_dptr which IS the real pointer

    /// module_id -> real CUmodule
    modules: HashMap<u64, CUmodule>,

    /// function_id -> real CUfunction
    functions: HashMap<u64, CUfunction>,

    /// Next ID counter
    next_id: u64,
}
```

**Key design decision:** For device pointers, the server sends back the real `CUdeviceptr` value. The client stores a mapping from synthetic to real. When the client sends a memory operation, it sends the real (server-side) pointer. This avoids a server-side device pointer table entirely.

### 3.4 Handle Lifecycle

```
Allocation:
  App calls cuMemAlloc() -> Client generates synthetic_ptr, sends to server
  -> Server calls real cuMemAlloc(), gets real_ptr, returns it
  -> Client stores: synthetic_ptr -> real_ptr
  -> Client returns synthetic_ptr to app

Usage:
  App calls cuMemcpyHtoD(synthetic_ptr, ...) -> Client looks up real_ptr
  -> Client sends MemcpyHtoD { dptr: real_ptr, ... } to server
  -> Server uses real_ptr directly in cuMemcpyHtoD()

Deallocation:
  App calls cuMemFree(synthetic_ptr) -> Client looks up real_ptr
  -> Client sends MemFree { dptr: real_ptr } to server
  -> Server calls real cuMemFree(real_ptr)
  -> Client removes synthetic_ptr from table
```

---

## 4. Step-by-Step Implementation Order

### Step 1: Protocol Types in `outterlink-common`

Define the wire protocol structures:

**Files to create/modify:**
- `crates/outterlink-common/src/protocol.rs` -- OpCode enum, message structs
- `crates/outterlink-common/src/wire.rs` -- Serialization/deserialization (read/write from TCP stream)
- `crates/outterlink-common/src/error.rs` -- Error types, CUresult mapping
- `crates/outterlink-common/src/lib.rs` -- Re-exports

**What to implement:**
1. `OpCode` enum with all PoC operations
2. Request and Response message types for each operation
3. `encode_request()` / `decode_request()` functions
4. `encode_response()` / `decode_response()` functions
5. Length-prefixed framing: `write_frame(stream, bytes)` / `read_frame(stream) -> bytes`
6. Magic number validation
7. Handshake message types

**Acceptance criteria:**
- [ ] Round-trip serialization tests pass for every message type
- [ ] Fuzz-tested deserialization does not panic on invalid input
- [ ] `cargo test -p outterlink-common` all green

### Step 2: Server Daemon in `outterlink-server`

Build the server that listens for TCP connections and executes real CUDA calls.

**Files to create/modify:**
- `crates/outterlink-server/src/main.rs` -- Entry point, TCP listener
- `crates/outterlink-server/src/cuda_executor.rs` -- Executes real CUDA calls via FFI
- `crates/outterlink-server/src/connection.rs` -- Per-client connection handler
- `crates/outterlink-server/src/handles.rs` -- Server-side handle tables
- `crates/outterlink-server/src/ffi.rs` -- Raw CUDA Driver API FFI bindings (or use cudarc)

**What to implement:**
1. TCP listener on configurable port (default 9370) using tokio
2. Accept connections, spawn a task per client
3. Handshake: exchange protocol version, server capabilities
4. Message loop: read frame -> decode -> dispatch to executor -> encode response -> write frame
5. CUDA FFI: load `libcuda.so` with `dlopen`, resolve function pointers with `dlsym`
6. Implement each CUDA call executor (23 functions)
7. Server-side handle tables with per-client isolation
8. Cleanup on client disconnect: destroy contexts, unload modules, free memory
9. Logging: log every CUDA call with timing

**Acceptance criteria:**
- [ ] Server starts, binds port, logs GPU info
- [ ] Handles client connect/disconnect cleanly
- [ ] Executes `cuInit` + `cuDeviceGetCount` for connected client
- [ ] All resources cleaned up on disconnect (no GPU memory leaks)

### Step 3: Client Interception Library in `outterlink-client`

Build the LD_PRELOAD library that intercepts CUDA calls and forwards them to the server.

**Files to create/modify:**
- `crates/outterlink-client/src/lib.rs` -- Library entry, `dlsym` override
- `crates/outterlink-client/src/interception.rs` -- CUDA function hooks (the actual intercepted functions)
- `crates/outterlink-client/src/connection.rs` -- TCP connection to server
- `crates/outterlink-client/src/handles.rs` -- Client-side handle tables
- `crates/outterlink-client/src/init.rs` -- Library initialization (constructor)
- `crates/outterlink-client/build.rs` -- Build script to produce `.so` with correct symbol exports

**What to implement:**
1. Library initialization via `__attribute__((constructor))` or Rust equivalent
2. Read server address from `OUTTERLINK_SERVER` env var (e.g., `192.168.100.2:9370`)
3. Override `dlsym()` to intercept CUDA symbol lookups
4. Override `cuGetProcAddress` to intercept CUDA 11.3+ dynamic resolution
5. Implement each intercepted function (23 functions)
6. Client-side handle tables (thread-safe)
7. TCP connection management (connect on first CUDA call, reconnect on failure)
8. Synchronous request/response: send request, block until response received

**Build considerations:**
- The library MUST be a C-compatible `.so` (cdylib)
- Must export `dlsym` symbol
- Must NOT link against `libcuda.so` (it replaces it)
- Must link against `libdl.so` for `dlvsym` (to call the real dlsym)

**Acceptance criteria:**
- [ ] `LD_PRELOAD=liboutterlink_client.so` loads without crash
- [ ] Device query calls return remote GPU properties
- [ ] Memory alloc/free works through the proxy
- [ ] HtoD/DtoH memory copies transfer data correctly

### Step 4: Integration Testing

**Files to create/modify:**
- `tests/poc/vector_add.cu` -- Test CUDA program
- `tests/poc/run_poc.sh` -- Demo script
- `tests/integration/test_device_query.rs` -- Automated integration test

**What to implement:**
1. Compile test CUDA program
2. Start server on GPU machine
3. Run client with LD_PRELOAD on separate machine (or same machine)
4. Verify correct results
5. Measure and log timing

**Acceptance criteria:**
- [ ] Vector addition produces correct results
- [ ] Device queries match real GPU properties
- [ ] No memory leaks (server-side CUDA memory tracking)

---

## 5. Test Program

The following CUDA program should work unmodified through OutterLink when the PoC is complete.

### `tests/poc/vector_add.cu`

```cuda
// vector_add.cu - OutterLink PoC test program
// Compile: nvcc -o vector_add vector_add.cu -lcuda
// Run: LD_PRELOAD=liboutterlink_client.so OUTTERLINK_SERVER=192.168.100.2:9370 ./vector_add

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// PTX for vector_add kernel (embedded as string so cuModuleLoadData works)
// This avoids needing a separate .ptx file
static const char *vector_add_ptx =
    ".version 7.0\n"
    ".target sm_52\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry vector_add(\n"
    "    .param .u64 a,\n"
    "    .param .u64 b,\n"
    "    .param .u64 c,\n"
    "    .param .u32 n\n"
    ")\n"
    "{\n"
    "    .reg .pred %p<2>;\n"
    "    .reg .f32 %f<4>;\n"
    "    .reg .b32 %r<5>;\n"
    "    .reg .b64 %rd<11>;\n"
    "\n"
    "    ld.param.u64 %rd1, [a];\n"
    "    ld.param.u64 %rd2, [b];\n"
    "    ld.param.u64 %rd3, [c];\n"
    "    ld.param.u32 %r1, [n];\n"
    "\n"
    "    mov.u32 %r2, %ctaid.x;\n"
    "    mov.u32 %r3, %ntid.x;\n"
    "    mov.u32 %r4, %tid.x;\n"
    "    mad.lo.s32 %r2, %r2, %r3, %r4;\n"
    "\n"
    "    setp.ge.s32 %p1, %r2, %r1;\n"
    "    @%p1 bra $done;\n"
    "\n"
    "    cvt.s64.s32 %rd4, %r2;\n"
    "    shl.b64 %rd5, %rd4, 2;\n"
    "\n"
    "    add.s64 %rd6, %rd1, %rd5;\n"
    "    add.s64 %rd7, %rd2, %rd5;\n"
    "    add.s64 %rd8, %rd3, %rd5;\n"
    "\n"
    "    ld.global.f32 %f1, [%rd6];\n"
    "    ld.global.f32 %f2, [%rd7];\n"
    "    add.f32 %f3, %f1, %f2;\n"
    "    st.global.f32 [%rd8], %f3;\n"
    "\n"
    "$done:\n"
    "    ret;\n"
    "}\n";

#define CHECK_CUDA(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char *name; \
        cuGetErrorName(err, &name); \
        fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n", __FILE__, __LINE__, name, err); \
        exit(1); \
    } \
} while(0)

int main(int argc, char **argv) {
    int N = 1024;
    if (argc > 1) N = atoi(argv[1]);

    printf("OutterLink PoC Test: Vector Addition (N=%d)\n", N);
    printf("============================================\n\n");

    // --- Step 1: Initialize CUDA ---
    printf("[1] Initializing CUDA...\n");
    CHECK_CUDA(cuInit(0));

    // --- Step 2: Query device ---
    int device_count;
    CHECK_CUDA(cuDeviceGetCount(&device_count));
    printf("[2] Found %d GPU(s)\n", device_count);

    CUdevice device;
    CHECK_CUDA(cuDeviceGet(&device, 0));

    char name[256];
    CHECK_CUDA(cuDeviceGetName(name, sizeof(name), device));

    size_t total_mem;
    CHECK_CUDA(cuDeviceTotalMem(&total_mem, device));

    printf("    GPU 0: %s (%.1f GB VRAM)\n", name, total_mem / (1024.0 * 1024.0 * 1024.0));

    int driver_version;
    CHECK_CUDA(cuDriverGetVersion(&driver_version));
    printf("    Driver version: %d.%d\n", driver_version / 1000, (driver_version % 100) / 10);

    // --- Step 3: Create context ---
    printf("[3] Creating context...\n");
    CUcontext ctx;
    CHECK_CUDA(cuCtxCreate(&ctx, 0, device));

    // --- Step 4: Load module (PTX) ---
    printf("[4] Loading PTX module...\n");
    CUmodule module;
    CHECK_CUDA(cuModuleLoadData(&module, vector_add_ptx));

    CUfunction kernel;
    CHECK_CUDA(cuModuleGetFunction(&kernel, module, "vector_add"));
    printf("    Kernel 'vector_add' loaded\n");

    // --- Step 5: Allocate memory ---
    printf("[5] Allocating GPU memory (%d floats = %zu bytes)...\n", N, N * sizeof(float));
    CUdeviceptr d_a, d_b, d_c;
    CHECK_CUDA(cuMemAlloc(&d_a, N * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&d_b, N * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&d_c, N * sizeof(float)));

    // --- Step 6: Initialize host data and copy to device ---
    printf("[6] Copying data to GPU...\n");
    float *h_a = (float *)malloc(N * sizeof(float));
    float *h_b = (float *)malloc(N * sizeof(float));
    float *h_c = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    CHECK_CUDA(cuMemcpyHtoD(d_a, h_a, N * sizeof(float)));
    CHECK_CUDA(cuMemcpyHtoD(d_b, h_b, N * sizeof(float)));

    // --- Step 7: Launch kernel ---
    printf("[7] Launching kernel (grid=%d, block=256)...\n", (N + 255) / 256);
    unsigned int grid_x = (N + 255) / 256;
    void *args[] = { &d_a, &d_b, &d_c, &N };
    CHECK_CUDA(cuLaunchKernel(kernel,
        grid_x, 1, 1,    // grid dimensions
        256, 1, 1,        // block dimensions
        0,                // shared memory
        0,                // stream (default)
        args,             // kernel params
        NULL));           // extra

    // --- Step 8: Copy results back ---
    printf("[8] Copying results from GPU...\n");
    CHECK_CUDA(cuMemcpyDtoH(h_c, d_c, N * sizeof(float)));

    // --- Step 9: Verify results ---
    printf("[9] Verifying results...\n");
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = (float)i + (float)(i * 2);
        if (h_c[i] != expected) {
            if (errors < 5) {
                fprintf(stderr, "    MISMATCH at index %d: got %f, expected %f\n",
                        i, h_c[i], expected);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("    ALL %d RESULTS CORRECT\n", N);
    } else {
        printf("    FAILED: %d/%d mismatches\n", errors, N);
    }

    // --- Step 10: Cleanup ---
    printf("[10] Cleaning up...\n");
    CHECK_CUDA(cuMemFree(d_a));
    CHECK_CUDA(cuMemFree(d_b));
    CHECK_CUDA(cuMemFree(d_c));
    CHECK_CUDA(cuModuleUnload(module));
    CHECK_CUDA(cuCtxDestroy(ctx));

    free(h_a);
    free(h_b);
    free(h_c);

    printf("\n============================================\n");
    printf("OutterLink PoC: %s\n", errors == 0 ? "PASS" : "FAIL");
    return errors == 0 ? 0 : 1;
}
```

---

## 6. PoC Demo Script

### `tests/poc/run_poc.sh`

```bash
#!/bin/bash
set -euo pipefail

# OutterLink PoC Demo Script
# Usage:
#   On SERVER (machine with GPU):
#     ./run_poc.sh server
#
#   On CLIENT (machine without GPU, or same machine):
#     ./run_poc.sh client <server_ip>
#
#   Same machine (loopback test):
#     ./run_poc.sh loopback

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PORT=9370

# Build everything
build() {
    echo "=== Building OutterLink ==="
    cd "$PROJECT_ROOT"
    cargo build --release
    echo ""

    echo "=== Compiling test program ==="
    nvcc -o "$SCRIPT_DIR/vector_add" "$SCRIPT_DIR/vector_add.cu" -lcuda 2>/dev/null || {
        echo "nvcc not found. Using pre-compiled binary or compiling on server."
    }
    echo ""
}

# Run the server
run_server() {
    echo "=== Starting OutterLink Server on port $PORT ==="
    echo "    Press Ctrl+C to stop"
    echo ""
    "$PROJECT_ROOT/target/release/outterlink-server" --port "$PORT" --log-level info
}

# Run the client
run_client() {
    local server_ip="${1:-127.0.0.1}"
    echo "=== Running PoC Test via OutterLink ==="
    echo "    Server: $server_ip:$PORT"
    echo ""

    export OUTTERLINK_SERVER="$server_ip:$PORT"
    export LD_PRELOAD="$PROJECT_ROOT/target/release/liboutterlink_client.so"
    export OUTTERLINK_LOG=info

    "$SCRIPT_DIR/vector_add" 1024
    local exit_code=$?

    echo ""
    if [ $exit_code -eq 0 ]; then
        echo "=== PoC DEMO: SUCCESS ==="
    else
        echo "=== PoC DEMO: FAILED (exit code $exit_code) ==="
    fi

    return $exit_code
}

# Loopback test (server and client on same machine)
run_loopback() {
    echo "=== OutterLink PoC Loopback Test ==="
    echo ""

    # Start server in background
    "$PROJECT_ROOT/target/release/outterlink-server" --port "$PORT" --log-level info &
    SERVER_PID=$!
    sleep 1  # Wait for server to start

    # Check server started
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server failed to start"
        exit 1
    fi

    echo "    Server started (PID $SERVER_PID)"
    echo ""

    # Run client
    run_client "127.0.0.1"
    local exit_code=$?

    # Stop server
    echo ""
    echo "    Stopping server..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null

    return $exit_code
}

# Baseline test (direct GPU, no OutterLink)
run_baseline() {
    echo "=== Baseline Test (direct GPU, no OutterLink) ==="
    echo ""
    "$SCRIPT_DIR/vector_add" 1024
}

# Main
case "${1:-help}" in
    server)
        build
        run_server
        ;;
    client)
        build
        run_client "${2:-127.0.0.1}"
        ;;
    loopback)
        build
        run_loopback
        ;;
    baseline)
        build
        run_baseline
        ;;
    help|*)
        echo "OutterLink PoC Demo"
        echo ""
        echo "Usage:"
        echo "  $0 server              Start the OutterLink server (on GPU machine)"
        echo "  $0 client <ip>         Run test through OutterLink (on client machine)"
        echo "  $0 loopback            Run server+client on same machine"
        echo "  $0 baseline            Run test directly on GPU (no OutterLink)"
        echo ""
        echo "Environment:"
        echo "  OUTTERLINK_SERVER      Server address (ip:port)"
        echo "  OUTTERLINK_LOG         Log level (trace, debug, info, warn, error)"
        ;;
esac
```

---

## 7. What is NOT in PoC

The following features are explicitly deferred to later phases:

| Feature | Deferred To | Why |
|---------|-------------|-----|
| Async memory copies (`cuMemcpyHtoDAsync`, etc.) | Phase 2 (P6) | Requires stream support |
| Streams (`cuStreamCreate`, etc.) | Phase 2 (P6) | Adds significant complexity |
| Events (`cuEventCreate`, etc.) | Phase 2 (P6) | Depends on streams |
| `cuCtxSynchronize` | Phase 2 (P6) | PoC is synchronous, every call blocks |
| `cuMemcpyDtoD_v2` | Phase 2 (P6) | Server-local, not needed for PoC |
| NVML interception | Phase 3 (P7) | Not needed for basic CUDA programs |
| Multi-GPU device enumeration | Phase 3 (P7) | PoC uses single GPU |
| Multi-server connections | Phase 6 (P10) | PoC connects to one server |
| io_uring / zero-copy networking | Phase 4 (P8) | TCP is sufficient for PoC |
| Call batching / coalescing | Phase 4 (P8) | PoC is synchronous per call |
| RDMA transport | Phase 4 (P8) | TCP proves the concept |
| Unified Memory | Not planned | Extremely hard to virtualize |
| CUDA Graphs | Phase 3+ (P7) | Bypasses kernel launch hooks |
| JIT compiler options | Phase 2 (P6) | `cuModuleLoadDataEx` options ignored in PoC |
| Error recovery / reconnection | Phase 2 (P6) | PoC aborts on connection failure |
| TLS / authentication | Phase 3+ (P7) | Not needed for local network PoC |
| `cuGetErrorName` / `cuGetErrorString` | Phase 2 (P6) | Can be handled client-side with a lookup table |
| Texture / surface objects | Phase 3+ (P7) | Specialized, not core |
| `cuMemAllocHost` / `cuMemFreeHost` | Phase 2 (P6) | Host memory is local, can be intercepted locally |

---

## Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|-----------|
| `cuLaunchKernel` parameter serialization is wrong | HIGH | MEDIUM | Use `CU_LAUNCH_PARAM_BUFFER_POINTER` for flat buffer approach; test with known kernel |
| PTX compilation fails on server (version mismatch) | MEDIUM | LOW | Embed PTX with lowest reasonable `sm_` target; test with server's GPU arch |
| `dlsym` override conflicts with other LD_PRELOAD libraries | MEDIUM | LOW | Use `dlvsym` with version to get real dlsym; test standalone first |
| Thread safety issues in handle tables | HIGH | MEDIUM | Use `RwLock` from start; test with multi-threaded CUDA program |
| TCP latency makes PoC unusably slow | LOW | LOW | PoC is for correctness, not performance; measure and document latency |
| Module image size detection (PTX vs cubin) | MEDIUM | MEDIUM | Support both: null-terminated scan for PTX, ELF header for cubin |
| `cuGetProcAddress` signature varies across CUDA versions | HIGH | MEDIUM | Support both 3-arg (pre-12.x) and 4-arg variants; version-gate at runtime |

## Estimated Scope

| Component | Files | Approximate Lines |
|-----------|-------|-------------------|
| `outterlink-common` (protocol) | 4 | ~800 |
| `outterlink-server` (daemon) | 5 | ~1200 |
| `outterlink-client` (interception .so) | 6 | ~1500 |
| Test program | 1 | ~150 |
| Demo script | 1 | ~100 |
| Integration tests | 2 | ~300 |
| **Total** | **19** | **~4050** |

## Related Documents

- [R3: CUDA Interception Strategies](../research/R3-cuda-interception.md)
- [R4: ConnectX-5 + Transport Stack](../research/R4-connectx5-transport-stack.md)
- [Final Pre-Plan](../pre-planning/02-FINAL-PREPLAN.md)
- [Project Vision](../../docs/architecture/00-project-vision.md)
- [Research Consolidation](../research/CONSOLIDATION-all-research.md)

## Open Questions

- [ ] Should `cuGetErrorName` / `cuGetErrorString` be intercepted in PoC? (Currently deferred -- the test program uses them in the CHECK_CUDA macro, so we may need a client-side lookup table at minimum)
- [ ] Should the client .so be written in C or Rust? (R3 says "C for interception .so" but Rust cdylib with `#[no_mangle] extern "C"` may work)
- [ ] What is the minimum PTX version / `sm_` target for the test kernel? (Need to match server GPU compute capability)
- [ ] Should we use `cudarc` crate for server-side CUDA FFI or raw bindings? (cudarc adds dependency but provides safety)
- [ ] How to handle `cuGetProcAddress_v2` (CUDA 12.x 4-argument variant) vs original 3-argument variant?
