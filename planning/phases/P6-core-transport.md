# P6: Phase 2 - Core Transport

**Created:** 2026-03-19
**Last Updated:** 2026-03-19
**Status:** Plan
**Priority:** HIGH - Core Implementation Phase

## Goal

Extend the PoC (P5) to support memory transfers (H2D, D2H, D2D), CUDA module loading (PTX/cubin forwarding), kernel launches, and context management over the network, enabling real CUDA compute workloads to execute on remote GPUs.

## Milestone

- `cuMemcpyHtoD` transfers host data to remote GPU correctly (verified with readback)
- `cuMemcpyDtoH` retrieves remote GPU data back to host correctly
- `cuModuleLoadData` forwards PTX/cubin to server, server loads it, returns handle
- `cuLaunchKernel` dispatches a kernel on the remote GPU with correct arguments
- End-to-end test: compile a CUDA kernel that does `C[i] = A[i] + B[i]`, load module remotely, transfer input arrays H2D, launch kernel, transfer result D2H, verify on host
- Large transfer test: 1 GB H2D + D2H round-trip completes without corruption
- Multi-threaded test: 4 threads issuing concurrent CUDA calls without data races

## Prerequisites

- [x] R3: CUDA interception strategy decided (Driver API + LD_PRELOAD)
- [x] R4: Transport stack decided (TCP + tokio Phase 1)
- [ ] P4: Project skeleton with crate structure exists
- [ ] P5: PoC works (device query + memory alloc/free over TCP)
- [ ] D9: Serialization format decided (this plan resolves it: custom binary)
- [ ] D14: Handle translation approach decided (this plan resolves it: `DashMap` concurrent hash map)

---

## Decision D9: Serialization Format

**Decision: Custom binary protocol with manual serialization.**

### Rationale

| Criterion | protobuf | FlatBuffers | Custom Binary |
|-----------|----------|-------------|---------------|
| Serialization latency | ~1800 ns/op | ~710 ns/op | ~100-200 ns/op |
| Zero-copy deserialize | No | Yes | Yes (with care) |
| Schema evolution | Excellent | Good | Manual |
| Bulk data (multi-GB) | Copies to arena | In-place possible | Direct memcpy |
| Code generation deps | Yes (prost) | Yes (flatc) | None |
| Complexity | Medium | Medium | High but contained |

For GPU remoting, the serialization overhead is on the critical path of every CUDA call. At ~20-50us per network round-trip, spending 1.8us on protobuf serialization is 4-9% overhead. Custom binary at ~200ns is negligible (<1%).

Bulk memory transfers (multi-GB) should never be copied into a serialization arena. The protocol must support sending raw byte streams directly after a header. This naturally favors a custom binary framing protocol where we control the wire format exactly.

Schema evolution is less critical because client and server are the same software -- they are always upgraded together.

**Mitigations for custom binary complexity:**
- All wire types defined as Rust enums with `#[repr(u16)]` discriminants
- Serialization/deserialization implemented as a trait with derive macros (future)
- Exhaustive unit tests for every message type with round-trip property testing
- Protocol version field enables future migration

## Decision D14: Handle Translation Approach

**Decision: `DashMap<LocalHandle, RemoteHandle>` per handle type.**

### Rationale

| Approach | Thread-safe | Lookup | Memory | Complexity |
|----------|-------------|--------|--------|-----------|
| `Mutex<HashMap>` | Yes (coarse lock) | O(1) amortized | Good | Low |
| `DashMap` | Yes (sharded) | O(1) amortized | Good | Low |
| Arena allocator | Yes (with atomics) | O(1) direct index | Best | High |
| Array + atomic index | Yes | O(1) | Best | Medium |

`DashMap` provides excellent concurrent read/write performance with sharded locking. It handles the common case (many reads, few writes) well. Arena allocators are faster but add significant complexity for handle recycling and would be premature optimization at this stage.

Local handles are synthetic values we generate. We use an `AtomicU64` counter per handle type to generate unique local handles. The remote handle is whatever the server returns.

---

## Section 1: Binary Protocol Specification

### 1.1 Wire Format Overview

```
TCP Stream:
+--------+--------+---------+------+---------+-----------+---------+
| Magic  | Ver    | Flags   | ReqID| MsgType | PayloadLen| Payload |
| 4 bytes| 2 bytes| 2 bytes | 8 b  | 2 bytes | 4 bytes   | N bytes |
+--------+--------+---------+------+---------+-----------+---------+
|<------------- Header: 22 bytes ----------->|<-- variable ------->|

For bulk data messages (memory transfers), an additional data segment follows:
+-----------+---------+
| DataLen   | Data    |
| 8 bytes   | N bytes |
+-----------+---------+
```

### 1.2 Header Fields

| Field | Type | Bytes | Description |
|-------|------|-------|-------------|
| `magic` | `[u8; 4]` | 4 | `0x4F4C4E4B` ("OLNK") - OutterLink protocol identifier |
| `version` | `u16` | 2 | Protocol version. Currently `1`. Big-endian. |
| `flags` | `u16` | 2 | Bit flags (see below). Big-endian. |
| `request_id` | `u64` | 8 | Unique per-connection request ID for correlation. Big-endian. |
| `msg_type` | `u16` | 2 | Message type discriminant. Big-endian. |
| `payload_len` | `u32` | 4 | Length of the payload section in bytes. Big-endian. |

**Total header: 22 bytes.**

All multi-byte integers are **big-endian** (network byte order) for header fields. Payload contents use **little-endian** (native x86) for CUDA data types to avoid unnecessary byte-swapping of GPU data.

### 1.3 Flags

| Bit | Name | Meaning |
|-----|------|---------|
| 0 | `IS_RESPONSE` | 0 = request, 1 = response |
| 1 | `HAS_BULK_DATA` | 1 = bulk data segment follows payload |
| 2 | `IS_BATCHED` | 1 = payload contains multiple batched calls |
| 3 | `IS_ERROR` | 1 = response carries an error (response only) |
| 4-15 | Reserved | Must be 0 |

### 1.4 Request/Response Correlation

Every request carries a `request_id` (monotonically increasing `u64` generated by the client). The server echoes the same `request_id` in the response. The client maintains a map of `request_id -> oneshot::Sender<Response>` for waking up the calling thread.

This allows pipelining: the client can send multiple requests without waiting for responses, and the server can process them concurrently (or in order per-stream).

### 1.5 Message Types

```rust
#[repr(u16)]
pub enum MessageType {
    // === Handshake (0x00xx) ===
    Handshake           = 0x0001,
    HandshakeAck        = 0x0002,
    Heartbeat           = 0x0003,
    HeartbeatAck        = 0x0004,

    // === Context Management (0x01xx) ===
    CtxCreate           = 0x0100,
    CtxDestroy          = 0x0101,
    CtxSetCurrent       = 0x0102,
    CtxGetCurrent       = 0x0103,
    CtxSynchronize      = 0x0104,
    CtxGetDevice        = 0x0105,

    // === Memory Allocation (0x02xx) ===
    MemAlloc            = 0x0200,
    MemFree             = 0x0201,
    MemAllocHost        = 0x0202,
    MemFreeHost         = 0x0203,
    MemGetInfo          = 0x0204,
    MemAllocPitch       = 0x0205,

    // === Memory Transfer (0x03xx) ===
    MemcpyHtoD          = 0x0300,  // HAS_BULK_DATA
    MemcpyDtoH          = 0x0301,  // Response has HAS_BULK_DATA
    MemcpyDtoD          = 0x0302,
    MemcpyHtoDAsync     = 0x0310,  // HAS_BULK_DATA
    MemcpyDtoHAsync     = 0x0311,  // Response has HAS_BULK_DATA
    MemcpyDtoDAsync     = 0x0312,
    Memset              = 0x0320,
    MemsetAsync         = 0x0321,

    // === Module Management (0x04xx) ===
    ModuleLoadData      = 0x0400,  // HAS_BULK_DATA (PTX/cubin)
    ModuleLoadDataEx    = 0x0401,  // HAS_BULK_DATA
    ModuleUnload        = 0x0402,
    ModuleGetFunction   = 0x0403,
    ModuleGetGlobal     = 0x0404,

    // === Kernel Execution (0x05xx) ===
    LaunchKernel        = 0x0500,  // HAS_BULK_DATA (kernel args)

    // === Batch (0x06xx) ===
    BatchedCalls        = 0x0600,  // IS_BATCHED

    // === Generic Response (0x0Fxx) ===
    ResultOk            = 0x0F00,
    ResultError         = 0x0F01,
    ResultWithData      = 0x0F02,  // HAS_BULK_DATA
}
```

### 1.6 Payload Formats (Field by Field)

All payload fields use **little-endian** encoding. Padding is inserted where needed for 8-byte alignment of u64/pointer fields.

#### Handshake (0x0001)

```
Offset  Size  Field
0       4     client_pid: u32
4       2     cuda_version_major: u16
6       2     cuda_version_minor: u16
8       2     protocol_version: u16
10      2     num_devices_requested: u16  (0 = all available)
12      4     client_name_len: u32
16      N     client_name: [u8; N]        (UTF-8, not null-terminated)
```

#### HandshakeAck (0x0002)

```
Offset  Size  Field
0       4     result: u32                 (CUresult, 0 = success)
4       2     server_protocol_version: u16
6       2     num_devices_available: u16
8       8     server_total_mem: u64       (total VRAM across all devices)
16      4     server_name_len: u32
20      N     server_name: [u8; N]
```

#### CtxCreate (0x0100)

```
Offset  Size  Field
0       4     flags: u32                  (CUctx_flags)
4       4     device: u32                 (CUdevice ordinal, already translated)
```

Response payload:
```
Offset  Size  Field
0       4     result: u32                 (CUresult)
4       4     _pad: u32
8       8     remote_ctx: u64             (CUcontext handle on server)
```

#### CtxDestroy (0x0101)

```
Offset  Size  Field
0       8     remote_ctx: u64
```

#### CtxSetCurrent (0x0102)

```
Offset  Size  Field
0       8     remote_ctx: u64             (0 for NULL context)
```

#### CtxSynchronize (0x0104)

Empty payload (0 bytes). Just synchronizes the current context on the server.

#### MemAlloc (0x0200)

```
Offset  Size  Field
0       8     byte_size: u64              (size_t)
```

Response payload:
```
Offset  Size  Field
0       4     result: u32
4       4     _pad: u32
8       8     remote_devptr: u64          (CUdeviceptr on server)
```

#### MemFree (0x0201)

```
Offset  Size  Field
0       8     remote_devptr: u64
```

#### MemcpyHtoD (0x0300) -- Host to Device

Request payload (header only):
```
Offset  Size  Field
0       8     dst_devptr: u64             (remote CUdeviceptr, already translated)
8       8     byte_count: u64
```

Followed by bulk data segment:
```
Offset  Size  Field
0       8     data_len: u64               (== byte_count)
8       N     data: [u8; N]               (the host memory contents)
```

Response payload:
```
Offset  Size  Field
0       4     result: u32
```

**Chunking for large transfers (>1 MB):**

When `byte_count > CHUNK_SIZE` (default 1 MB), the client sends the transfer as multiple MemcpyHtoD messages with additional chunking metadata in the payload:

```
Offset  Size  Field
0       8     dst_devptr: u64             (base device pointer)
8       8     total_byte_count: u64       (total transfer size)
16      8     chunk_offset: u64           (offset within the total transfer)
24      8     chunk_size: u64             (size of this chunk's data)
32      4     chunk_index: u32            (0-based chunk number)
36      4     total_chunks: u32           (total chunk count)
```

The server accumulates chunks (or writes them directly to device memory at `dst_devptr + chunk_offset`) and sends a single response after the final chunk.

#### MemcpyDtoH (0x0301) -- Device to Host

Request payload:
```
Offset  Size  Field
0       8     src_devptr: u64             (remote CUdeviceptr)
8       8     byte_count: u64
```

Response has `HAS_BULK_DATA` set:
```
Response payload:
Offset  Size  Field
0       4     result: u32
4       4     _pad: u32

Bulk data:
Offset  Size  Field
0       8     data_len: u64
8       N     data: [u8; N]               (the device memory contents)
```

For large responses, the server uses the same chunking protocol (multiple ResultWithData messages with chunk metadata).

#### MemcpyDtoD (0x0302) -- Device to Device

Both pointers are remote (same server). No bulk data transfer over the network.

```
Offset  Size  Field
0       8     dst_devptr: u64
8       8     src_devptr: u64
16      8     byte_count: u64
```

Response: standard `result: u32` only.

#### MemcpyHtoDAsync / MemcpyDtoHAsync (0x0310, 0x0311)

Same payload format as synchronous variants, plus:
```
Additional field at end:
Offset  Size  Field
+0      8     stream: u64                 (remote CUstream handle, 0 = default)
```

The server queues the memcpy on the specified stream. The response is sent immediately after queueing (not after completion). Stream synchronization is a separate call.

#### ModuleLoadData (0x0400)

Request payload:
```
Offset  Size  Field
0       4     data_format: u32            (0 = PTX text, 1 = cubin, 2 = fatbin)
4       4     _pad: u32
```

Bulk data:
```
Offset  Size  Field
0       8     data_len: u64
8       N     data: [u8; N]               (PTX source or cubin binary)
```

Response payload:
```
Offset  Size  Field
0       4     result: u32
4       4     _pad: u32
8       8     remote_module: u64          (CUmodule handle on server)
16      4     num_functions: u32          (number of kernel functions found)
20      4     _pad2: u32
```

Followed by a function info table (so the client can cache parameter metadata):
```
For each function (repeated num_functions times):
Offset  Size  Field
0       4     name_len: u32
4       N     name: [u8; N]               (kernel function name, UTF-8)
N+0     4     num_params: u32
N+4     M*8   param_sizes: [u64; M]       (size of each parameter in bytes)
```

The server extracts this metadata by parsing the `.nv.info` ELF section of the cubin (using the KPARAM_INFO / CBANK_PARAM_OFFSETS attributes) or by loading the module and querying cuFuncGetAttribute. This metadata is critical -- without it, the client cannot serialize kernel arguments for cuLaunchKernel.

#### ModuleUnload (0x0402)

```
Offset  Size  Field
0       8     remote_module: u64
```

#### ModuleGetFunction (0x0403)

```
Offset  Size  Field
0       8     remote_module: u64
4       4     name_len: u32
8       N     name: [u8; N]               (function name, UTF-8)
```

Response:
```
Offset  Size  Field
0       4     result: u32
4       4     _pad: u32
8       8     remote_function: u64        (CUfunction handle on server)
16      4     num_params: u32
20      4     _pad2: u32
24      M*8   param_sizes: [u64; M]       (size of each parameter)
```

#### LaunchKernel (0x0500)

Request payload:
```
Offset  Size  Field
0       8     function: u64               (remote CUfunction handle)
8       4     grid_dim_x: u32
12      4     grid_dim_y: u32
16      4     grid_dim_z: u32
20      4     block_dim_x: u32
24      4     block_dim_y: u32
28      4     block_dim_z: u32
32      4     shared_mem_bytes: u32
36      4     _pad: u32
40      8     stream: u64                 (remote CUstream, 0 = default)
48      4     num_params: u32
52      4     total_params_size: u32      (total bytes in the packed args)
```

Bulk data (the packed kernel arguments):
```
Offset  Size  Field
0       8     data_len: u64               (== total_params_size)
8       N     data: [u8; N]               (packed kernel arguments, see below)
```

**Kernel argument packing format:**

The arguments are packed sequentially with alignment padding matching CUDA's rules. Each argument is packed at its natural alignment (1, 2, 4, or 8 bytes). The client uses the `param_sizes` array from ModuleLoadData/ModuleGetFunction to know the size of each argument.

```
Example for kernel: void add(float* A, float* B, float* C, int N)
param_sizes = [8, 8, 8, 4]  (pointers are 8 bytes, int is 4)

Packed args buffer (28 bytes):
Offset 0:  A_devptr  (u64, 8 bytes) -- translated to remote CUdeviceptr
Offset 8:  B_devptr  (u64, 8 bytes) -- translated to remote CUdeviceptr
Offset 16: C_devptr  (u64, 8 bytes) -- translated to remote CUdeviceptr
Offset 24: N         (i32, 4 bytes) -- passed as-is
```

The critical step: **pointer arguments must be translated from local synthetic handles to remote real device pointers before packing.** The client walks the packed args buffer, and for each argument whose size is 8 bytes (pointer-sized), it checks the handle translation table. If the value matches a known local CUdeviceptr handle, it is replaced with the corresponding remote CUdeviceptr.

This heuristic (8-byte values that match known device pointers) works because:
1. Device pointers are always 8 bytes on 64-bit
2. The set of valid local handles is known
3. Scalar values matching a device pointer by coincidence is astronomically unlikely (pointers are in high address ranges)

For robustness, the server can also validate that pointer-sized arguments point to known allocations.

#### BatchedCalls (0x0600)

When `IS_BATCHED` is set, the payload contains multiple serialized messages concatenated:

```
Offset  Size  Field
0       4     num_calls: u32
4       4     _pad: u32

Then for each call:
0       2     msg_type: u16
2       2     _pad: u16
4       4     call_payload_len: u32
8       N     call_payload: [u8; N]       (individual message payload)
```

The server executes all calls in order and returns a single response with all results:

```
Offset  Size  Field
0       4     num_results: u32
4       4     _pad: u32

Then for each result:
0       4     result: u32                 (CUresult)
4       4     result_payload_len: u32
8       N     result_payload: [u8; N]     (individual result payload, may be empty)
```

#### ResultError (0x0F01)

```
Offset  Size  Field
0       4     cuda_result: u32            (CUresult error code)
4       4     error_kind: u32             (0 = CUDA error, 1 = network error, 2 = protocol error)
8       4     message_len: u32
12      N     message: [u8; N]            (human-readable error, UTF-8)
```

---

## Section 2: Memory Transfer Protocol

### 2.1 cuMemcpyHtoD -- Host to Device

**Flow:**

```
Client                              Server
  |                                   |
  | 1. Look up dst_devptr in          |
  |    handle table -> remote ptr     |
  |                                   |
  | 2. Allocate CUDA pinned buffer    |
  |    (cudaHostAlloc, reuse pool)    |
  |                                   |
  | 3. Copy app's host data into      |
  |    pinned buffer (memcpy)         |
  |                                   |
  | 4. If size <= CHUNK_SIZE:         |
  |    Send single MemcpyHtoD msg  -->|
  |    with HAS_BULK_DATA             | 5. cuMemcpyHtoD(remote_ptr,
  |                                   |    recv_buf, size)
  |                                   |
  | 6. If size > CHUNK_SIZE:          |
  |    For each 1MB chunk:            |
  |      Send MemcpyHtoD chunk    --->| 7. cuMemcpyHtoD(remote_ptr+off,
  |                                   |    chunk_buf, chunk_size)
  |                                   |
  |<-- ResultOk (after final chunk) --| 8. Send response
  |                                   |
  | 9. Return CUDA_SUCCESS to app     |
```

### 2.2 cuMemcpyDtoH -- Device to Host

```
Client                              Server
  |                                   |
  | 1. Look up src_devptr -> remote   |
  |                                   |
  | 2. Send MemcpyDtoH request   --->| 3. Allocate pinned buffer
  |                                   |    cuMemcpyDtoH(buf, remote_ptr, size)
  |                                   |
  |<-- ResultWithData + bulk data --- | 4. Send data back (chunked if large)
  |                                   |
  | 5. Copy received data to app's    |
  |    host buffer                    |
  | 6. Return CUDA_SUCCESS            |
```

### 2.3 cuMemcpyDtoD -- Device to Device

Both device pointers are on the same remote server. No data crosses the network.

```
Client                              Server
  |                                   |
  | 1. Translate both ptrs to remote  |
  | 2. Send MemcpyDtoD request   --->| 3. cuMemcpyDtoD(dst, src, size)
  |<-- ResultOk -------------------- | 4. Return result
```

If source and destination are on different remote servers (future multi-node), this becomes a two-step operation: DtoH from server A, HtoD to server B. That is a P10 concern.

### 2.4 Async Variants

For `cuMemcpyHtoDAsync` and `cuMemcpyDtoHAsync`:

1. The client still sends the data synchronously over TCP (the network transfer must complete to deliver the bytes)
2. The server queues the actual GPU memcpy on the specified CUDA stream using the async API
3. The server responds immediately after queueing (not after GPU completion)
4. The client returns immediately to the application
5. The application must call `cuStreamSynchronize` or `cuCtxSynchronize` to ensure completion

This means "async" in the CUDA sense (GPU side) but the network transfer itself is still synchronous. True async network transfers would require RDMA and are a P8/P9 concern.

### 2.5 Pinned Memory Buffer Pool

To avoid the overhead of `cudaHostAlloc` / `cudaFreeHost` on every transfer, both client and server maintain a pool of pinned memory buffers.

```rust
pub struct PinnedBufferPool {
    /// Free buffers indexed by size class (power-of-2 sizes)
    free_lists: [Mutex<Vec<PinnedBuffer>>; 24],  // 1B to 8GB size classes
    /// Total pinned memory in use
    total_pinned: AtomicU64,
    /// Maximum pinned memory allowed
    max_pinned: u64,
}

pub struct PinnedBuffer {
    ptr: *mut u8,
    capacity: usize,
}

impl PinnedBufferPool {
    /// Get a buffer of at least `size` bytes from the pool, or allocate new
    pub fn acquire(&self, size: usize) -> Result<PinnedBuffer, TransportError> {
        let class = size.next_power_of_two().trailing_zeros() as usize;
        // Try to pop from free list
        if let Some(buf) = self.free_lists[class].lock().pop() {
            return Ok(buf);
        }
        // Allocate new pinned buffer
        if self.total_pinned.load(Ordering::Relaxed) + size as u64 > self.max_pinned {
            return Err(TransportError::PinnedMemoryExhausted);
        }
        let mut ptr: *mut u8 = std::ptr::null_mut();
        let result = unsafe {
            cuda::cuMemAllocHost_v2(&mut ptr as *mut *mut u8 as _, size)
        };
        if result != 0 {
            return Err(TransportError::CudaError(result));
        }
        self.total_pinned.fetch_add(size as u64, Ordering::Relaxed);
        Ok(PinnedBuffer { ptr, capacity: size })
    }

    /// Return a buffer to the pool
    pub fn release(&self, buf: PinnedBuffer) {
        let class = buf.capacity.trailing_zeros() as usize;
        self.free_lists[class].lock().push(buf);
    }
}
```

### 2.6 Chunking and Flow Control

**Chunk size: 1 MB (1,048,576 bytes) default, configurable.**

Why 1 MB:
- Small enough to bound memory usage per in-flight transfer
- Large enough to amortize per-message overhead (22 bytes header / 1MB data = 0.002%)
- Matches TCP window sizes on most systems (4-16 MB window, 1-4 chunks in flight)
- Allows progress reporting for multi-GB transfers

**Flow control:**

TCP itself provides flow control via its sliding window. We do not implement additional application-level flow control in Phase 2. The sending side sends chunks as fast as TCP allows. The receiving side processes them in order.

If the server's GPU memory fills up during a transfer, the server returns `CUDA_ERROR_OUT_OF_MEMORY` in the response to the failing chunk, and the client propagates this to the application.

**Backpressure signal (future P8 optimization):**

In Phase 4, we may add application-level flow control where the server can send a `SLOW_DOWN` signal if its pinned buffer pool is exhausted, causing the client to throttle.

---

## Section 3: Module Loading and Kernel Launch Protocol

### 3.1 Module Loading (cuModuleLoadData)

```
Client                                  Server
  |                                       |
  | 1. App calls cuModuleLoadData(module, |
  |    image) where image is PTX or cubin |
  |                                       |
  | 2. Detect format by inspecting image: |
  |    - Starts with 0x7F 'E' 'L' 'F'    |
  |      -> cubin (ELF)                   |
  |    - Starts with "//" or ".version"   |
  |      -> PTX (text)                    |
  |    - Starts with 0x466154 ("FaT")     |
  |      -> fatbin                        |
  |                                       |
  | 3. Send ModuleLoadData with           |
  |    entire image as bulk data      --->| 4. cuModuleLoadData(&mod, image)
  |                                       |
  |                                       | 5. Enumerate functions in module:
  |                                       |    Parse .nv.info section of cubin
  |                                       |    for each function found,
  |                                       |    extract param count + sizes
  |                                       |
  |<-- Response with remote_module    --- | 6. Return module handle + function
  |    handle + function metadata table   |    metadata table
  |                                       |
  | 7. Store in handle table:             |
  |    local_module -> remote_module      |
  |                                       |
  | 8. Cache function metadata:           |
  |    (module, func_name) -> param_sizes |
  |                                       |
  | 9. Return local_module to app         |
```

**Server-side parameter metadata extraction:**

The server has two strategies for extracting kernel parameter information:

**Strategy A (preferred): Parse cubin ELF directly**

When the loaded image is a cubin or fatbin, the server parses the ELF `.nv.info.<kernel_name>` sections to extract KPARAM_INFO and CBANK_PARAM_OFFSETS attributes. This gives exact parameter counts, sizes, and offsets.

The `cudaparsers` Rust crate (https://github.com/VivekPanyam/cudaparsers) can parse cubin files and has been tested against thousands of binaries. We can either use it directly or implement our own parser for the specific attributes we need.

**Strategy B (fallback): Use CUDA APIs + cuobjdump**

If parsing fails (or for PTX that must be JIT-compiled), the server can:
1. Load the module normally with `cuModuleLoadData`
2. Use `cuobjdump --dump-elf` on the resulting cubin to extract metadata
3. Or use `cuFuncGetAttribute` to query parameter-related attributes

**Strategy C (simplest for initial implementation): Client-side parameter tracking**

For the initial P6 implementation, we can use a simpler approach:
- The client does not need the server to extract parameter metadata
- Instead, the client intercepts `cuLaunchKernel` and uses the `extra` parameter format (`CU_LAUNCH_PARAM_BUFFER_POINTER` + `CU_LAUNCH_PARAM_BUFFER_SIZE`) which includes the total packed args buffer and its size
- The client forwards the packed args buffer as-is, translating only the pointer arguments within it

This works because CUDA Runtime API (which wraps Driver API) always uses the `extra` parameter format, and most applications use the Runtime API. For direct Driver API users who pass `kernelParams` (void** array), the client needs the param_sizes metadata.

**Recommended approach for P6: Implement Strategy C first. Add Strategy A when we encounter applications that use the Driver API directly with `kernelParams`.**

### 3.2 Kernel Launch (cuLaunchKernel)

The CUDA signature:
```c
CUresult cuLaunchKernel(
    CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void **kernelParams,    // Method 1: array of pointers to each arg
    void **extra            // Method 2: packed buffer with CU_LAUNCH_PARAM_*
);
```

**Client-side interception logic:**

```rust
pub unsafe extern "C" fn our_cuLaunchKernel(
    f: CUfunction,
    grid_x: u32, grid_y: u32, grid_z: u32,
    block_x: u32, block_y: u32, block_z: u32,
    shared_mem: u32,
    stream: CUstream,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult {
    let client = get_client();

    // Translate function handle
    let remote_func = match client.handle_table.functions.get(&f) {
        Some(rf) => *rf,
        None => return CUDA_ERROR_INVALID_HANDLE,
    };

    // Translate stream handle
    let remote_stream = if stream.is_null() {
        0u64  // default stream
    } else {
        match client.handle_table.streams.get(&stream) {
            Some(rs) => *rs,
            None => return CUDA_ERROR_INVALID_HANDLE,
        }
    };

    // Serialize kernel arguments
    let packed_args = if !extra.is_null() {
        // Method 2: extra parameter -- already packed
        extract_packed_args_from_extra(extra)
    } else if !kernel_params.is_null() {
        // Method 1: kernelParams -- need param_sizes metadata
        let metadata = match client.param_cache.get(&remote_func) {
            Some(m) => m,
            None => return CUDA_ERROR_INVALID_VALUE,
        };
        pack_kernel_params(kernel_params, &metadata.param_sizes)
    } else {
        // No arguments
        Vec::new()
    };

    // Translate device pointers within packed args
    let translated_args = translate_device_pointers_in_args(
        &packed_args,
        &client.handle_table.device_ptrs,
    );

    // Send to server
    match client.transport.launch_kernel(
        remote_func,
        [grid_x, grid_y, grid_z],
        [block_x, block_y, block_z],
        shared_mem,
        remote_stream,
        &translated_args,
    ) {
        Ok(result) => result,
        Err(_) => CUDA_ERROR_UNKNOWN,
    }
}

/// Extract the packed buffer from the CU_LAUNCH_PARAM_* extra array
unsafe fn extract_packed_args_from_extra(extra: *mut *mut c_void) -> Vec<u8> {
    let mut buf_ptr: *const u8 = std::ptr::null();
    let mut buf_size: usize = 0;

    let mut i = 0;
    loop {
        let token = *extra.add(i) as usize;
        match token {
            0 => break, // CU_LAUNCH_PARAM_END
            1 => {
                // CU_LAUNCH_PARAM_BUFFER_POINTER
                i += 1;
                buf_ptr = *extra.add(i) as *const u8;
            }
            2 => {
                // CU_LAUNCH_PARAM_BUFFER_SIZE
                i += 1;
                buf_size = *(*extra.add(i) as *const usize);
            }
            _ => break,
        }
        i += 1;
    }

    if buf_ptr.is_null() || buf_size == 0 {
        return Vec::new();
    }
    std::slice::from_raw_parts(buf_ptr, buf_size).to_vec()
}

/// Pack kernelParams (void**) into a contiguous buffer using param_sizes
unsafe fn pack_kernel_params(
    params: *mut *mut c_void,
    param_sizes: &[u64],
) -> Vec<u8> {
    let mut buf = Vec::new();
    for (i, &size) in param_sizes.iter().enumerate() {
        let param_ptr = *params.add(i);
        let bytes = std::slice::from_raw_parts(param_ptr as *const u8, size as usize);

        // Align to natural boundary
        let align = match size {
            1 => 1,
            2 => 2,
            s if s <= 4 => 4,
            _ => 8,
        };
        while buf.len() % align as usize != 0 {
            buf.push(0);
        }

        buf.extend_from_slice(bytes);
    }
    buf
}
```

**Server-side execution:**

```rust
fn handle_launch_kernel(
    &self,
    function: CUfunction,
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem: u32,
    stream: CUstream,
    packed_args: &[u8],
) -> CUresult {
    // Use the CU_LAUNCH_PARAM_BUFFER_POINTER method to pass packed args
    let mut arg_buffer_ptr = packed_args.as_ptr() as *mut c_void;
    let mut arg_buffer_size = packed_args.len();

    let mut extra: [*mut c_void; 5] = [
        1 as *mut c_void,  // CU_LAUNCH_PARAM_BUFFER_POINTER
        arg_buffer_ptr,
        2 as *mut c_void,  // CU_LAUNCH_PARAM_BUFFER_SIZE
        &mut arg_buffer_size as *mut usize as *mut c_void,
        0 as *mut c_void,  // CU_LAUNCH_PARAM_END
    ];

    unsafe {
        cuLaunchKernel(
            function,
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
            shared_mem,
            stream,
            std::ptr::null_mut(),  // kernelParams = NULL
            extra.as_mut_ptr(),    // use extra instead
        )
    }
}
```

### 3.3 Device Pointer Translation in Kernel Arguments

When the client packs kernel arguments, some 8-byte values are device pointers that need translation. The translation algorithm:

```rust
fn translate_device_pointers_in_args(
    packed_args: &[u8],
    devptr_table: &DashMap<u64, u64>,  // local -> remote
) -> Vec<u8> {
    let mut result = packed_args.to_vec();

    // Scan for 8-byte aligned values that match known device pointers
    let mut offset = 0;
    while offset + 8 <= result.len() {
        let value = u64::from_le_bytes(result[offset..offset + 8].try_into().unwrap());

        if let Some(remote_ptr) = devptr_table.get(&value) {
            result[offset..offset + 8].copy_from_slice(&remote_ptr.to_le_bytes());
        }

        offset += 8;  // Only check 8-byte aligned positions
    }

    result
}
```

This scan is O(N/8) where N is the packed args size (max 4KB per CUDA spec), so at most 512 lookups. With DashMap this completes in well under 1 microsecond.

---

## Section 4: Handle Translation Design

### 4.1 Handle Types and Data Structures

```rust
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Central handle translation table, shared across all client threads
pub struct HandleTable {
    pub contexts:    HandleMap<CUcontext>,
    pub device_ptrs: HandleMap<CUdeviceptr>,
    pub modules:     HandleMap<CUmodule>,
    pub functions:   HandleMap<CUfunction>,
    pub streams:     HandleMap<CUstream>,
    pub events:      HandleMap<CUevent>,
}

/// A bidirectional concurrent map for one handle type
pub struct HandleMap<H: Copy + Eq + Hash> {
    /// Local synthetic handle -> remote real handle
    local_to_remote: DashMap<u64, u64>,
    /// Remote real handle -> local synthetic handle (for reverse lookups)
    remote_to_local: DashMap<u64, u64>,
    /// Counter for generating unique local handles
    next_local: AtomicU64,
    /// Name for debug logging
    type_name: &'static str,
}

impl<H: Copy + Eq + Hash> HandleMap<H> {
    pub fn new(type_name: &'static str, start: u64) -> Self {
        Self {
            local_to_remote: DashMap::new(),
            remote_to_local: DashMap::new(),
            next_local: AtomicU64::new(start),
            type_name,
        }
    }

    /// Register a new handle pair. Returns the local synthetic handle.
    pub fn register(&self, remote: u64) -> u64 {
        // Check if we already have this remote handle
        if let Some(existing) = self.remote_to_local.get(&remote) {
            return *existing;
        }

        let local = self.next_local.fetch_add(1, Ordering::Relaxed);
        self.local_to_remote.insert(local, remote);
        self.remote_to_local.insert(remote, local);
        local
    }

    /// Translate local -> remote. Returns None if handle is invalid/freed.
    pub fn to_remote(&self, local: u64) -> Option<u64> {
        self.local_to_remote.get(&local).map(|v| *v)
    }

    /// Translate remote -> local.
    pub fn to_local(&self, remote: u64) -> Option<u64> {
        self.remote_to_local.get(&remote).map(|v| *v)
    }

    /// Remove a handle pair (on cuMemFree, cuModuleUnload, etc.)
    pub fn remove(&self, local: u64) -> Option<u64> {
        if let Some((_, remote)) = self.local_to_remote.remove(&local) {
            self.remote_to_local.remove(&remote);
            Some(remote)
        } else {
            None
        }
    }
}
```

### 4.2 Handle Ranges

Each handle type uses a different starting range to avoid collisions and make debugging easier:

| Handle Type | Start Value | Range |
|-------------|-------------|-------|
| CUcontext | 0x0C000000_00000000 | Contexts (few, long-lived) |
| CUdeviceptr | 0x0D000000_00000000 | Device pointers (many, varied lifetime) |
| CUmodule | 0x0E000000_00000000 | Modules (moderate) |
| CUfunction | 0x0F000000_00000000 | Functions (moderate, tied to module lifetime) |
| CUstream | 0x0A000000_00000000 | Streams (few) |
| CUevent | 0x0B000000_00000000 | Events (moderate) |

These ranges are in the high address space, unlikely to collide with real CUDA values, and the prefix byte immediately tells you the handle type during debugging.

### 4.3 Thread Safety

- `DashMap` uses sharded internal locking (16 shards by default). Reads do not block writes to different shards. This is optimal for the CUDA usage pattern where different threads typically operate on different handles.
- `AtomicU64::fetch_add` for handle generation is lock-free.
- The `HandleTable` is created once at initialization and stored in a global `OnceCell<HandleTable>`. All threads access it through the same reference.

### 4.4 Handle Lifecycle

```
Creation:     cuMemAlloc -> server allocates -> returns remote ptr
              -> client registers (local, remote) pair in HandleMap
              -> client returns local handle to application

Use:          cuMemcpyHtoD(local_ptr, ...) -> client translates local -> remote
              -> sends remote ptr to server

Destruction:  cuMemFree(local_ptr) -> client translates local -> remote
              -> sends MemFree(remote) to server
              -> on success, removes from HandleMap

Invalid use:  cuMemcpyHtoD(freed_local_ptr, ...) -> client cannot find in HandleMap
              -> returns CUDA_ERROR_INVALID_VALUE immediately (no network call)
```

### 4.5 Stale Handle Protection

If a handle is used after the remote resource is freed (e.g., use-after-free bug in the CUDA application), the handle will not exist in the HandleMap, and the client returns `CUDA_ERROR_INVALID_VALUE` without making a network call. This is actually safer than real CUDA, which might crash or return garbage.

---

## Section 5: Connection Management

### 5.1 Connection Establishment

```
Client                                  Server
  |                                       |
  | 1. TCP connect to server:port     --->| 2. Accept connection
  |                                       |
  | 3. Send Handshake msg            --->| 4. Validate protocol version
  |    (pid, cuda_ver, protocol_ver)      |    Check device availability
  |                                       |
  |<--- HandshakeAck                 --- | 5. Return device info
  |     (num_devices, total_mem)          |
  |                                       |
  | 6. If HandshakeAck.result != 0,       |
  |    close connection, return error     |
  |                                       |
  | 7. Connection is ESTABLISHED          |
  |    Start heartbeat timer              |
```

### 5.2 Connection Architecture

**Phase 2 (P6): Single connection per server.**

```
Client Process
  |
  +-- OutterLink Client Library
       |
       +-- TcpTransport
            |
            +-- Single TCP connection to Server
            |     (multiplexed via request_id)
            |
            +-- Response dispatcher (background task)
            |     reads responses, matches request_id,
            |     wakes waiting threads via oneshot channels
            |
            +-- Heartbeat task (every 5s)
```

All CUDA threads share the same TCP connection. Writes are serialized through a `tokio::sync::Mutex<OwnedWriteHalf>` (necessary for TCP which is a single byte stream). Reads happen on a single background task that dispatches responses to waiting threads.

**Future (P8): Connection per stream or connection pool.**

Multiple TCP connections could enable parallel transfers when the application uses multiple CUDA streams. This is a performance optimization for P8.

### 5.3 Connection State Machine

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConnectionState {
    /// Initial state, not yet connected
    Disconnected,
    /// TCP connected, handshake in progress
    Connecting,
    /// Handshake complete, ready for CUDA calls
    Ready,
    /// Connection lost, attempting reconnection
    Reconnecting { attempt: u32 },
    /// Permanently failed, all calls will return errors
    Failed,
}
```

### 5.4 Reconnection Strategy

If the TCP connection drops:

1. All in-flight requests receive `TransportError::ConnectionLost`
2. The client enters `Reconnecting { attempt: 0 }` state
3. Reconnection attempts use exponential backoff: 100ms, 200ms, 400ms, ... up to 5s
4. Maximum 10 reconnection attempts before entering `Failed` state
5. During `Reconnecting`, new CUDA calls return `CUDA_ERROR_OPERATING_SYSTEM` (closest standard error code)
6. On successful reconnection, the client must re-establish context:
   - Re-send context creation
   - Re-load modules
   - Handle table is NOT cleared (local handles remain valid)
   - Device memory allocations on the server are LOST (application must handle this)

**Important:** Transparent reconnection with full state recovery is not feasible in P6 because GPU-side state (device memory contents, loaded modules) is lost when the server connection drops. The practical behavior is: reconnection re-establishes the transport, but the CUDA application is likely to fail due to invalid device memory. This is acceptable -- GPU failures are already non-recoverable in most CUDA applications.

### 5.5 Heartbeat

- Client sends `Heartbeat` every 5 seconds
- Server responds with `HeartbeatAck`
- If no heartbeat response within 15 seconds, connection is considered dead
- Heartbeats are low priority and do not block CUDA calls

---

## Section 6: Call Batching (Lazy Updates)

### 6.1 Concept

Many CUDA calls have no immediate side effects observable by the application. They can be queued locally and sent as a single batch to reduce round-trips. This is the "lazy updates" pattern from vCUDA.

### 6.2 Batchable vs Non-Batchable Calls

| Call | Batchable? | Reason |
|------|-----------|--------|
| `cuCtxSetCurrent` | YES | State change, no return value needed |
| `cuMemAlloc` | NO | Application needs the returned pointer immediately |
| `cuMemFree` | YES | No return value needed (fire-and-forget) |
| `cuMemcpyHtoD` | YES* | No return value, but needs data sent. Can batch with other H2D. |
| `cuMemcpyDtoH` | NO | Application needs the returned data |
| `cuModuleLoadData` | NO | Application needs the module handle |
| `cuModuleGetFunction` | NO | Application needs the function handle |
| `cuLaunchKernel` | YES | Asynchronous by nature, no immediate result |
| `cuCtxSynchronize` | FLUSH | This is a synchronization point, flush all batched calls first |
| `cuStreamSynchronize` | FLUSH | Synchronization point |
| `cuMemcpyDtoH` | FLUSH | Needs all prior operations to complete |

### 6.3 Batch Buffer Design

```rust
pub struct BatchBuffer {
    /// Queued operations not yet sent
    operations: Vec<BatchedOperation>,
    /// Total serialized size of queued operations
    total_size: usize,
    /// Maximum batch size before auto-flush (default 64 operations)
    max_operations: usize,
    /// Maximum batch byte size before auto-flush (default 256 KB)
    max_bytes: usize,
}

pub struct BatchedOperation {
    msg_type: MessageType,
    payload: Vec<u8>,
    bulk_data: Option<Vec<u8>>,
}

impl BatchBuffer {
    /// Queue an operation for batched sending
    pub fn enqueue(&mut self, op: BatchedOperation) -> BatchAction {
        self.operations.push(op);
        self.total_size += op.payload.len() + op.bulk_data.as_ref().map_or(0, |d| d.len());

        if self.operations.len() >= self.max_operations
            || self.total_size >= self.max_bytes
        {
            BatchAction::FlushNow
        } else {
            BatchAction::Queued
        }
    }

    /// Take all queued operations for sending
    pub fn drain(&mut self) -> Vec<BatchedOperation> {
        self.total_size = 0;
        std::mem::take(&mut self.operations)
    }

    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }
}

pub enum BatchAction {
    Queued,
    FlushNow,
}
```

### 6.4 Flush Triggers

The batch is flushed (all queued operations sent as a single BatchedCalls message) when:

1. A non-batchable call is made (cuMemAlloc, cuMemcpyDtoH, cuModuleLoadData)
2. A synchronization call is made (cuCtxSynchronize, cuStreamSynchronize)
3. The batch buffer reaches max_operations or max_bytes
4. The application calls cuCtxSynchronize
5. Explicitly flushed by the client (e.g., before shutdown)

**Important for P6:** Batching is designed but may be implemented as a pass-through initially (batch size = 1, effectively no batching). This lets us validate correctness first, then enable batching as a performance optimization. The infrastructure is in place from day one.

---

## Section 7: Error Handling

### 7.1 Error Categories

```rust
#[derive(Debug)]
pub enum TransportError {
    // --- CUDA errors (map to CUresult) ---
    CudaError {
        code: CUresult,
        message: String,
    },

    // --- Network errors ---
    ConnectionRefused,
    ConnectionLost,
    ConnectionTimeout { elapsed_ms: u64 },
    HandshakeFailed { reason: String },

    // --- Protocol errors ---
    InvalidMagic { got: [u8; 4] },
    VersionMismatch { client: u16, server: u16 },
    UnknownMessageType { msg_type: u16 },
    PayloadTooLarge { size: u32, max: u32 },
    CorruptedMessage { detail: String },

    // --- Resource errors ---
    PinnedMemoryExhausted,
    HandleTableFull,

    // --- Internal errors ---
    SerializationError { detail: String },
    InternalError { detail: String },
}
```

### 7.2 Error Mapping to CUresult

| TransportError | CUresult | Rationale |
|----------------|----------|-----------|
| `CudaError { code, .. }` | `code` (pass-through) | Server returned a real CUDA error |
| `ConnectionRefused` | `CUDA_ERROR_OPERATING_SYSTEM` (304) | Best match for "system not available" |
| `ConnectionLost` | `CUDA_ERROR_OPERATING_SYSTEM` (304) | Best match for "system not available" |
| `ConnectionTimeout` | `CUDA_ERROR_OPERATING_SYSTEM` (304) | Timeout looks like OS error to app |
| `HandshakeFailed` | `CUDA_ERROR_NOT_INITIALIZED` (3) | Cannot initialize remote GPU |
| `InvalidMagic` | `CUDA_ERROR_UNKNOWN` (999) | Protocol corruption |
| `VersionMismatch` | `CUDA_ERROR_NOT_SUPPORTED` (801) | Incompatible versions |
| `PinnedMemoryExhausted` | `CUDA_ERROR_OUT_OF_MEMORY` (2) | Host memory exhaustion |
| `HandleTableFull` | `CUDA_ERROR_OUT_OF_MEMORY` (2) | Resource limit |

### 7.3 Error Propagation Flow

```
Server-side CUDA error:
  cuMemAlloc returns CUDA_ERROR_OUT_OF_MEMORY
  -> Server sends ResultError { cuda_result: 2, error_kind: 0, message: "..." }
  -> Client receives, creates TransportError::CudaError { code: 2, message: "..." }
  -> Client returns CUDA_ERROR_OUT_OF_MEMORY to application

Network error during transfer:
  TCP connection drops mid-transfer
  -> Tokio read returns Err(BrokenPipe)
  -> Client creates TransportError::ConnectionLost
  -> Client returns CUDA_ERROR_OPERATING_SYSTEM to application
  -> Client transitions to Reconnecting state

Protocol error:
  Server sends response with wrong magic bytes
  -> Client creates TransportError::InvalidMagic
  -> Client returns CUDA_ERROR_UNKNOWN to application
  -> Client closes connection (protocol state is corrupted)
```

### 7.4 Timeouts

| Operation | Default Timeout | Rationale |
|-----------|----------------|-----------|
| Handshake | 5 seconds | Server should respond quickly |
| Heartbeat response | 15 seconds | 3 missed heartbeats |
| cuMemAlloc | 10 seconds | GPU allocation should be fast |
| cuMemcpyHtoD (per chunk) | 30 seconds | 1MB at even 33 KB/s |
| cuMemcpyDtoH (per chunk) | 30 seconds | Same as H2D |
| cuModuleLoadData | 60 seconds | JIT compilation can be slow |
| cuLaunchKernel | 5 seconds | Just queuing, not execution |
| cuCtxSynchronize | 300 seconds | Kernel execution can take minutes |

---

## Section 8: Rust Implementation Code

### 8.1 Protocol Message Types

```rust
// File: crates/outterlink-common/src/protocol/mod.rs

pub mod message;
pub mod serialize;
pub mod transport;
pub mod connection;

/// Protocol constants
pub const PROTOCOL_MAGIC: [u8; 4] = [0x4F, 0x4C, 0x4E, 0x4B]; // "OLNK"
pub const PROTOCOL_VERSION: u16 = 1;
pub const HEADER_SIZE: usize = 22;
pub const DEFAULT_CHUNK_SIZE: usize = 1_048_576; // 1 MB
pub const MAX_PAYLOAD_SIZE: u32 = 256 * 1024 * 1024; // 256 MB
pub const MAX_BULK_DATA_SIZE: u64 = 8 * 1024 * 1024 * 1024; // 8 GB
```

```rust
// File: crates/outterlink-common/src/protocol/message.rs

use std::io::{self, Read, Write, Cursor};
use byteorder::{BigEndian, LittleEndian, ReadBytesExt, WriteBytesExt};

// --- Header ---

#[derive(Debug, Clone, Copy)]
pub struct MessageHeader {
    pub magic: [u8; 4],
    pub version: u16,
    pub flags: MessageFlags,
    pub request_id: u64,
    pub msg_type: MessageType,
    pub payload_len: u32,
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct MessageFlags: u16 {
        const IS_RESPONSE   = 0b0000_0001;
        const HAS_BULK_DATA = 0b0000_0010;
        const IS_BATCHED    = 0b0000_0100;
        const IS_ERROR      = 0b0000_1000;
    }
}

#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MessageType {
    // Handshake
    Handshake           = 0x0001,
    HandshakeAck        = 0x0002,
    Heartbeat           = 0x0003,
    HeartbeatAck        = 0x0004,

    // Context
    CtxCreate           = 0x0100,
    CtxDestroy          = 0x0101,
    CtxSetCurrent       = 0x0102,
    CtxGetCurrent       = 0x0103,
    CtxSynchronize      = 0x0104,
    CtxGetDevice        = 0x0105,

    // Memory Allocation
    MemAlloc            = 0x0200,
    MemFree             = 0x0201,
    MemAllocHost        = 0x0202,
    MemFreeHost         = 0x0203,
    MemGetInfo          = 0x0204,

    // Memory Transfer
    MemcpyHtoD          = 0x0300,
    MemcpyDtoH          = 0x0301,
    MemcpyDtoD          = 0x0302,
    MemcpyHtoDAsync     = 0x0310,
    MemcpyDtoHAsync     = 0x0311,
    MemcpyDtoDAsync     = 0x0312,

    // Module
    ModuleLoadData      = 0x0400,
    ModuleLoadDataEx    = 0x0401,
    ModuleUnload        = 0x0402,
    ModuleGetFunction   = 0x0403,
    ModuleGetGlobal     = 0x0404,

    // Kernel
    LaunchKernel        = 0x0500,

    // Batch
    BatchedCalls        = 0x0600,

    // Response
    ResultOk            = 0x0F00,
    ResultError         = 0x0F01,
    ResultWithData      = 0x0F02,
}

impl MessageType {
    pub fn from_u16(v: u16) -> Option<Self> {
        // Safety: all valid discriminants are covered
        match v {
            0x0001 => Some(Self::Handshake),
            0x0002 => Some(Self::HandshakeAck),
            0x0003 => Some(Self::Heartbeat),
            0x0004 => Some(Self::HeartbeatAck),
            0x0100 => Some(Self::CtxCreate),
            0x0101 => Some(Self::CtxDestroy),
            0x0102 => Some(Self::CtxSetCurrent),
            0x0103 => Some(Self::CtxGetCurrent),
            0x0104 => Some(Self::CtxSynchronize),
            0x0105 => Some(Self::CtxGetDevice),
            0x0200 => Some(Self::MemAlloc),
            0x0201 => Some(Self::MemFree),
            0x0202 => Some(Self::MemAllocHost),
            0x0203 => Some(Self::MemFreeHost),
            0x0204 => Some(Self::MemGetInfo),
            0x0300 => Some(Self::MemcpyHtoD),
            0x0301 => Some(Self::MemcpyDtoH),
            0x0302 => Some(Self::MemcpyDtoD),
            0x0310 => Some(Self::MemcpyHtoDAsync),
            0x0311 => Some(Self::MemcpyDtoHAsync),
            0x0312 => Some(Self::MemcpyDtoDAsync),
            0x0400 => Some(Self::ModuleLoadData),
            0x0401 => Some(Self::ModuleLoadDataEx),
            0x0402 => Some(Self::ModuleUnload),
            0x0403 => Some(Self::ModuleGetFunction),
            0x0404 => Some(Self::ModuleGetGlobal),
            0x0500 => Some(Self::LaunchKernel),
            0x0600 => Some(Self::BatchedCalls),
            0x0F00 => Some(Self::ResultOk),
            0x0F01 => Some(Self::ResultError),
            0x0F02 => Some(Self::ResultWithData),
            _ => None,
        }
    }
}

// --- Concrete Message Payloads ---

#[derive(Debug, Clone)]
pub enum RequestPayload {
    Handshake {
        client_pid: u32,
        cuda_version_major: u16,
        cuda_version_minor: u16,
        protocol_version: u16,
        num_devices_requested: u16,
        client_name: String,
    },
    CtxCreate {
        flags: u32,
        device: u32,
    },
    CtxDestroy {
        remote_ctx: u64,
    },
    CtxSetCurrent {
        remote_ctx: u64,
    },
    CtxSynchronize,
    MemAlloc {
        byte_size: u64,
    },
    MemFree {
        remote_devptr: u64,
    },
    MemcpyHtoD {
        dst_devptr: u64,
        byte_count: u64,
        // Bulk data follows in the wire message
    },
    MemcpyHtoDChunked {
        dst_devptr: u64,
        total_byte_count: u64,
        chunk_offset: u64,
        chunk_size: u64,
        chunk_index: u32,
        total_chunks: u32,
    },
    MemcpyDtoH {
        src_devptr: u64,
        byte_count: u64,
    },
    MemcpyDtoD {
        dst_devptr: u64,
        src_devptr: u64,
        byte_count: u64,
    },
    ModuleLoadData {
        data_format: u32, // 0=PTX, 1=cubin, 2=fatbin
        // Bulk data (the module image) follows
    },
    ModuleUnload {
        remote_module: u64,
    },
    ModuleGetFunction {
        remote_module: u64,
        name: String,
    },
    LaunchKernel {
        function: u64,
        grid_dim: [u32; 3],
        block_dim: [u32; 3],
        shared_mem_bytes: u32,
        stream: u64,
        num_params: u32,
        total_params_size: u32,
        // Bulk data (packed kernel args) follows
    },
    Heartbeat,
}

#[derive(Debug, Clone)]
pub enum ResponsePayload {
    HandshakeAck {
        result: u32,
        server_protocol_version: u16,
        num_devices_available: u16,
        server_total_mem: u64,
        server_name: String,
    },
    CtxCreated {
        result: u32,
        remote_ctx: u64,
    },
    MemAllocated {
        result: u32,
        remote_devptr: u64,
    },
    ModuleLoaded {
        result: u32,
        remote_module: u64,
        function_metadata: Vec<FunctionMetadata>,
    },
    FunctionFound {
        result: u32,
        remote_function: u64,
        param_sizes: Vec<u64>,
    },
    Ok {
        result: u32,
    },
    Error {
        cuda_result: u32,
        error_kind: u32,
        message: String,
    },
    HeartbeatAck,
}

#[derive(Debug, Clone)]
pub struct FunctionMetadata {
    pub name: String,
    pub param_sizes: Vec<u64>,
}

/// A complete wire message (header + payload + optional bulk data)
#[derive(Debug)]
pub struct WireMessage {
    pub header: MessageHeader,
    pub payload: Vec<u8>,
    pub bulk_data: Option<Vec<u8>>,
}
```

### 8.2 Serialization / Deserialization

```rust
// File: crates/outterlink-common/src/protocol/serialize.rs

use super::message::*;
use super::{PROTOCOL_MAGIC, PROTOCOL_VERSION, HEADER_SIZE};
use byteorder::{BigEndian, LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read, Write};

// --- Header Serialization ---

impl MessageHeader {
    pub fn serialize(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.magic);
        buf.write_u16::<BigEndian>(self.version).unwrap();
        buf.write_u16::<BigEndian>(self.flags.bits()).unwrap();
        buf.write_u64::<BigEndian>(self.request_id).unwrap();
        buf.write_u16::<BigEndian>(self.msg_type as u16).unwrap();
        buf.write_u32::<BigEndian>(self.payload_len).unwrap();
    }

    pub fn deserialize(buf: &[u8]) -> Result<Self, ProtocolError> {
        if buf.len() < HEADER_SIZE {
            return Err(ProtocolError::IncompleteHeader {
                have: buf.len(),
                need: HEADER_SIZE,
            });
        }

        let mut cursor = Cursor::new(buf);

        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic).unwrap();
        if magic != PROTOCOL_MAGIC {
            return Err(ProtocolError::InvalidMagic { got: magic });
        }

        let version = cursor.read_u16::<BigEndian>().unwrap();
        let flags_raw = cursor.read_u16::<BigEndian>().unwrap();
        let flags = MessageFlags::from_bits(flags_raw)
            .ok_or(ProtocolError::InvalidFlags { bits: flags_raw })?;
        let request_id = cursor.read_u64::<BigEndian>().unwrap();
        let msg_type_raw = cursor.read_u16::<BigEndian>().unwrap();
        let msg_type = MessageType::from_u16(msg_type_raw)
            .ok_or(ProtocolError::UnknownMessageType { msg_type: msg_type_raw })?;
        let payload_len = cursor.read_u32::<BigEndian>().unwrap();

        Ok(Self {
            magic,
            version,
            flags,
            request_id,
            msg_type,
            payload_len,
        })
    }
}

// --- Payload Serialization Helpers ---

/// Serialize a RequestPayload into (payload_bytes, optional_bulk_data)
pub fn serialize_request(
    payload: &RequestPayload,
) -> (Vec<u8>, Option<&[u8]>) {
    let mut buf = Vec::with_capacity(64);

    match payload {
        RequestPayload::MemAlloc { byte_size } => {
            buf.write_u64::<LittleEndian>(*byte_size).unwrap();
            (buf, None)
        }
        RequestPayload::MemFree { remote_devptr } => {
            buf.write_u64::<LittleEndian>(*remote_devptr).unwrap();
            (buf, None)
        }
        RequestPayload::MemcpyHtoD { dst_devptr, byte_count } => {
            buf.write_u64::<LittleEndian>(*dst_devptr).unwrap();
            buf.write_u64::<LittleEndian>(*byte_count).unwrap();
            // Bulk data is attached separately by the caller
            (buf, None)
        }
        RequestPayload::MemcpyDtoH { src_devptr, byte_count } => {
            buf.write_u64::<LittleEndian>(*src_devptr).unwrap();
            buf.write_u64::<LittleEndian>(*byte_count).unwrap();
            (buf, None)
        }
        RequestPayload::MemcpyDtoD { dst_devptr, src_devptr, byte_count } => {
            buf.write_u64::<LittleEndian>(*dst_devptr).unwrap();
            buf.write_u64::<LittleEndian>(*src_devptr).unwrap();
            buf.write_u64::<LittleEndian>(*byte_count).unwrap();
            (buf, None)
        }
        RequestPayload::CtxCreate { flags, device } => {
            buf.write_u32::<LittleEndian>(*flags).unwrap();
            buf.write_u32::<LittleEndian>(*device).unwrap();
            (buf, None)
        }
        RequestPayload::CtxDestroy { remote_ctx } => {
            buf.write_u64::<LittleEndian>(*remote_ctx).unwrap();
            (buf, None)
        }
        RequestPayload::CtxSetCurrent { remote_ctx } => {
            buf.write_u64::<LittleEndian>(*remote_ctx).unwrap();
            (buf, None)
        }
        RequestPayload::CtxSynchronize => {
            (buf, None) // empty payload
        }
        RequestPayload::ModuleLoadData { data_format } => {
            buf.write_u32::<LittleEndian>(*data_format).unwrap();
            buf.write_u32::<LittleEndian>(0).unwrap(); // padding
            (buf, None) // bulk data attached separately
        }
        RequestPayload::ModuleUnload { remote_module } => {
            buf.write_u64::<LittleEndian>(*remote_module).unwrap();
            (buf, None)
        }
        RequestPayload::ModuleGetFunction { remote_module, name } => {
            buf.write_u64::<LittleEndian>(*remote_module).unwrap();
            buf.write_u32::<LittleEndian>(name.len() as u32).unwrap();
            buf.extend_from_slice(name.as_bytes());
            (buf, None)
        }
        RequestPayload::LaunchKernel {
            function, grid_dim, block_dim,
            shared_mem_bytes, stream,
            num_params, total_params_size,
        } => {
            buf.write_u64::<LittleEndian>(*function).unwrap();
            for d in grid_dim { buf.write_u32::<LittleEndian>(*d).unwrap(); }
            for d in block_dim { buf.write_u32::<LittleEndian>(*d).unwrap(); }
            buf.write_u32::<LittleEndian>(*shared_mem_bytes).unwrap();
            buf.write_u32::<LittleEndian>(0).unwrap(); // padding
            buf.write_u64::<LittleEndian>(*stream).unwrap();
            buf.write_u32::<LittleEndian>(*num_params).unwrap();
            buf.write_u32::<LittleEndian>(*total_params_size).unwrap();
            (buf, None) // packed args as bulk data
        }
        RequestPayload::Handshake {
            client_pid, cuda_version_major, cuda_version_minor,
            protocol_version, num_devices_requested, client_name,
        } => {
            buf.write_u32::<LittleEndian>(*client_pid).unwrap();
            buf.write_u16::<LittleEndian>(*cuda_version_major).unwrap();
            buf.write_u16::<LittleEndian>(*cuda_version_minor).unwrap();
            buf.write_u16::<LittleEndian>(*protocol_version).unwrap();
            buf.write_u16::<LittleEndian>(*num_devices_requested).unwrap();
            buf.write_u32::<LittleEndian>(client_name.len() as u32).unwrap();
            buf.extend_from_slice(client_name.as_bytes());
            (buf, None)
        }
        RequestPayload::Heartbeat => {
            (buf, None)
        }
        _ => {
            (buf, None)
        }
    }
}

/// Deserialize a response payload from raw bytes
pub fn deserialize_response(
    msg_type: MessageType,
    payload: &[u8],
) -> Result<ResponsePayload, ProtocolError> {
    let mut cursor = Cursor::new(payload);

    match msg_type {
        MessageType::ResultOk => {
            let result = cursor.read_u32::<LittleEndian>()?;
            Ok(ResponsePayload::Ok { result })
        }
        MessageType::ResultError => {
            let cuda_result = cursor.read_u32::<LittleEndian>()?;
            let error_kind = cursor.read_u32::<LittleEndian>()?;
            let msg_len = cursor.read_u32::<LittleEndian>()? as usize;
            let mut msg_bytes = vec![0u8; msg_len];
            cursor.read_exact(&mut msg_bytes)?;
            let message = String::from_utf8_lossy(&msg_bytes).to_string();
            Ok(ResponsePayload::Error { cuda_result, error_kind, message })
        }
        MessageType::HandshakeAck => {
            let result = cursor.read_u32::<LittleEndian>()?;
            let server_protocol_version = cursor.read_u16::<LittleEndian>()?;
            let num_devices_available = cursor.read_u16::<LittleEndian>()?;
            let server_total_mem = cursor.read_u64::<LittleEndian>()?;
            let name_len = cursor.read_u32::<LittleEndian>()? as usize;
            let mut name_bytes = vec![0u8; name_len];
            cursor.read_exact(&mut name_bytes)?;
            let server_name = String::from_utf8_lossy(&name_bytes).to_string();
            Ok(ResponsePayload::HandshakeAck {
                result,
                server_protocol_version,
                num_devices_available,
                server_total_mem,
                server_name,
            })
        }
        _ => Err(ProtocolError::UnexpectedResponseType { msg_type }),
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    #[error("Incomplete header: have {have} bytes, need {need}")]
    IncompleteHeader { have: usize, need: usize },
    #[error("Invalid magic bytes: {got:?}")]
    InvalidMagic { got: [u8; 4] },
    #[error("Invalid flags bits: 0x{bits:04X}")]
    InvalidFlags { bits: u16 },
    #[error("Unknown message type: 0x{msg_type:04X}")]
    UnknownMessageType { msg_type: u16 },
    #[error("Unexpected response type: {msg_type:?}")]
    UnexpectedResponseType { msg_type: MessageType },
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
```

### 8.3 Transport Trait and TCP Implementation

```rust
// File: crates/outterlink-common/src/protocol/transport.rs

use super::message::*;
use super::serialize::*;
use super::{PROTOCOL_MAGIC, PROTOCOL_VERSION, HEADER_SIZE, MAX_PAYLOAD_SIZE};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::TcpStream;
use tokio::sync::{Mutex, oneshot};
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// The core transport trait. Implementations handle the actual I/O.
#[async_trait::async_trait]
pub trait Transport: Send + Sync + 'static {
    /// Send a request and wait for the response.
    async fn request(
        &self,
        msg_type: MessageType,
        payload: &[u8],
        bulk_data: Option<&[u8]>,
    ) -> Result<WireMessage, TransportError>;

    /// Send a request without waiting for a response (fire-and-forget).
    async fn send_oneshot(
        &self,
        msg_type: MessageType,
        payload: &[u8],
        bulk_data: Option<&[u8]>,
    ) -> Result<(), TransportError>;

    /// Check if the connection is alive.
    fn is_connected(&self) -> bool;

    /// Close the connection.
    async fn close(&self) -> Result<(), TransportError>;
}

/// TCP transport implementation using tokio.
pub struct TcpTransport {
    /// Write half of the TCP stream, protected by a mutex for serialized writes
    writer: Mutex<OwnedWriteHalf>,
    /// Monotonically increasing request ID
    next_request_id: AtomicU64,
    /// Pending requests: request_id -> oneshot sender for the response
    pending: Arc<DashMap<u64, oneshot::Sender<Result<WireMessage, TransportError>>>>,
    /// Connection state
    state: Arc<std::sync::RwLock<ConnectionState>>,
    /// Handle to the reader task (for shutdown)
    reader_handle: tokio::task::JoinHandle<()>,
}

impl TcpTransport {
    pub async fn connect(addr: &str) -> Result<Self, TransportError> {
        let stream = TcpStream::connect(addr)
            .await
            .map_err(|_| TransportError::ConnectionRefused)?;

        // Set TCP options for performance
        stream.set_nodelay(true)
            .map_err(|e| TransportError::InternalError {
                detail: format!("Failed to set TCP_NODELAY: {e}"),
            })?;

        // Split into read and write halves
        let (reader, writer) = stream.into_split();

        let pending: Arc<DashMap<u64, oneshot::Sender<Result<WireMessage, TransportError>>>>
            = Arc::new(DashMap::new());
        let state = Arc::new(std::sync::RwLock::new(ConnectionState::Connecting));

        // Spawn the reader task that dispatches responses
        let pending_clone = pending.clone();
        let state_clone = state.clone();
        let reader_handle = tokio::spawn(async move {
            Self::reader_loop(reader, pending_clone, state_clone).await;
        });

        Ok(Self {
            writer: Mutex::new(writer),
            next_request_id: AtomicU64::new(1),
            pending,
            state,
            reader_handle,
        })
    }

    /// Background task: reads responses from the TCP stream and dispatches them
    async fn reader_loop(
        mut reader: OwnedReadHalf,
        pending: Arc<DashMap<u64, oneshot::Sender<Result<WireMessage, TransportError>>>>,
        state: Arc<std::sync::RwLock<ConnectionState>>,
    ) {
        let mut header_buf = vec![0u8; HEADER_SIZE];

        loop {
            // Read header
            match reader.read_exact(&mut header_buf).await {
                Ok(_) => {}
                Err(e) => {
                    // Connection lost
                    *state.write().unwrap() = ConnectionState::Reconnecting { attempt: 0 };
                    // Notify all pending requests
                    for entry in pending.iter() {
                        // Cannot send on already-consumed senders, so we just clear
                    }
                    pending.clear();
                    return;
                }
            }

            let header = match MessageHeader::deserialize(&header_buf) {
                Ok(h) => h,
                Err(e) => {
                    // Protocol corruption, close connection
                    *state.write().unwrap() = ConnectionState::Failed;
                    pending.clear();
                    return;
                }
            };

            // Read payload
            let mut payload = vec![0u8; header.payload_len as usize];
            if header.payload_len > 0 {
                if let Err(_) = reader.read_exact(&mut payload).await {
                    *state.write().unwrap() = ConnectionState::Reconnecting { attempt: 0 };
                    pending.clear();
                    return;
                }
            }

            // Read bulk data if present
            let bulk_data = if header.flags.contains(MessageFlags::HAS_BULK_DATA) {
                let mut len_buf = [0u8; 8];
                if let Err(_) = reader.read_exact(&mut len_buf).await {
                    *state.write().unwrap() = ConnectionState::Reconnecting { attempt: 0 };
                    pending.clear();
                    return;
                }
                let data_len = u64::from_le_bytes(len_buf) as usize;
                let mut data = vec![0u8; data_len];
                if let Err(_) = reader.read_exact(&mut data).await {
                    *state.write().unwrap() = ConnectionState::Reconnecting { attempt: 0 };
                    pending.clear();
                    return;
                }
                Some(data)
            } else {
                None
            };

            let msg = WireMessage { header, payload, bulk_data };

            // Dispatch to waiting request
            if let Some((_, sender)) = pending.remove(&header.request_id) {
                let _ = sender.send(Ok(msg));
            }
            // If no pending request matches, it's a heartbeat ack or stale -- ignore
        }
    }

    /// Serialize and send a complete wire message
    async fn send_message(
        writer: &Mutex<OwnedWriteHalf>,
        header: &MessageHeader,
        payload: &[u8],
        bulk_data: Option<&[u8]>,
    ) -> Result<(), TransportError> {
        let mut wire = Vec::with_capacity(HEADER_SIZE + payload.len() + 8);
        header.serialize(&mut wire);
        wire.extend_from_slice(payload);

        let mut w = writer.lock().await;

        w.write_all(&wire).await.map_err(|_| TransportError::ConnectionLost)?;

        if let Some(data) = bulk_data {
            // Write bulk data length + data
            let len_bytes = (data.len() as u64).to_le_bytes();
            w.write_all(&len_bytes).await.map_err(|_| TransportError::ConnectionLost)?;
            w.write_all(data).await.map_err(|_| TransportError::ConnectionLost)?;
        }

        w.flush().await.map_err(|_| TransportError::ConnectionLost)?;

        Ok(())
    }
}

#[async_trait::async_trait]
impl Transport for TcpTransport {
    async fn request(
        &self,
        msg_type: MessageType,
        payload: &[u8],
        bulk_data: Option<&[u8]>,
    ) -> Result<WireMessage, TransportError> {
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);

        let mut flags = MessageFlags::empty();
        if bulk_data.is_some() {
            flags |= MessageFlags::HAS_BULK_DATA;
        }

        let header = MessageHeader {
            magic: PROTOCOL_MAGIC,
            version: PROTOCOL_VERSION,
            flags,
            request_id,
            msg_type,
            payload_len: payload.len() as u32,
        };

        // Register response channel before sending (avoid race)
        let (tx, rx) = oneshot::channel();
        self.pending.insert(request_id, tx);

        // Send the message
        Self::send_message(&self.writer, &header, payload, bulk_data).await?;

        // Wait for response with timeout
        let timeout = match msg_type {
            MessageType::CtxSynchronize => Duration::from_secs(300),
            MessageType::ModuleLoadData | MessageType::ModuleLoadDataEx => Duration::from_secs(60),
            _ => Duration::from_secs(30),
        };

        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => {
                // oneshot sender was dropped (reader task died)
                self.pending.remove(&request_id);
                Err(TransportError::ConnectionLost)
            }
            Err(_) => {
                // Timeout
                self.pending.remove(&request_id);
                Err(TransportError::ConnectionTimeout {
                    elapsed_ms: timeout.as_millis() as u64,
                })
            }
        }
    }

    async fn send_oneshot(
        &self,
        msg_type: MessageType,
        payload: &[u8],
        bulk_data: Option<&[u8]>,
    ) -> Result<(), TransportError> {
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);

        let mut flags = MessageFlags::empty();
        if bulk_data.is_some() {
            flags |= MessageFlags::HAS_BULK_DATA;
        }

        let header = MessageHeader {
            magic: PROTOCOL_MAGIC,
            version: PROTOCOL_VERSION,
            flags,
            request_id,
            msg_type,
            payload_len: payload.len() as u32,
        };

        Self::send_message(&self.writer, &header, payload, bulk_data).await
    }

    fn is_connected(&self) -> bool {
        matches!(*self.state.read().unwrap(), ConnectionState::Ready)
    }

    async fn close(&self) -> Result<(), TransportError> {
        self.reader_handle.abort();
        // Writer will be dropped
        *self.state.write().unwrap() = ConnectionState::Disconnected;
        Ok(())
    }
}

// --- Error types ---

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("CUDA error: code={code}, {message}")]
    CudaError { code: u32, message: String },

    #[error("Connection refused")]
    ConnectionRefused,

    #[error("Connection lost")]
    ConnectionLost,

    #[error("Connection timeout after {elapsed_ms}ms")]
    ConnectionTimeout { elapsed_ms: u64 },

    #[error("Handshake failed: {reason}")]
    HandshakeFailed { reason: String },

    #[error("Invalid magic bytes: {got:?}")]
    InvalidMagic { got: [u8; 4] },

    #[error("Protocol version mismatch: client={client}, server={server}")]
    VersionMismatch { client: u16, server: u16 },

    #[error("Unknown message type: 0x{msg_type:04X}")]
    UnknownMessageType { msg_type: u16 },

    #[error("Payload too large: {size} bytes (max {max})")]
    PayloadTooLarge { size: u32, max: u32 },

    #[error("Corrupted message: {detail}")]
    CorruptedMessage { detail: String },

    #[error("Pinned memory exhausted")]
    PinnedMemoryExhausted,

    #[error("Handle table full")]
    HandleTableFull,

    #[error("Serialization error: {detail}")]
    SerializationError { detail: String },

    #[error("Internal error: {detail}")]
    InternalError { detail: String },
}
```

### 8.4 Connection State Machine

```rust
// File: crates/outterlink-common/src/protocol/connection.rs

use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConnectionState {
    /// Not connected
    Disconnected,
    /// TCP connected, handshake in progress
    Connecting,
    /// Fully operational
    Ready,
    /// Lost connection, attempting to reconnect
    Reconnecting { attempt: u32 },
    /// Permanently failed
    Failed,
}

impl ConnectionState {
    /// Maximum reconnection attempts before entering Failed state
    pub const MAX_RECONNECT_ATTEMPTS: u32 = 10;

    /// Get the delay before the next reconnection attempt
    pub fn reconnect_delay(&self) -> Option<Duration> {
        match self {
            ConnectionState::Reconnecting { attempt } => {
                if *attempt >= Self::MAX_RECONNECT_ATTEMPTS {
                    return None; // Give up
                }
                // Exponential backoff: 100ms, 200ms, 400ms, ... capped at 5s
                let base_ms = 100u64;
                let delay_ms = (base_ms * (1u64 << attempt.min(&10))).min(5000);
                Some(Duration::from_millis(delay_ms))
            }
            _ => None,
        }
    }

    /// Map this state to a CUresult for CUDA calls during non-Ready states
    pub fn to_cuda_error(&self) -> u32 {
        match self {
            ConnectionState::Disconnected => 3,   // CUDA_ERROR_NOT_INITIALIZED
            ConnectionState::Connecting => 3,      // CUDA_ERROR_NOT_INITIALIZED
            ConnectionState::Ready => 0,           // CUDA_SUCCESS (no error)
            ConnectionState::Reconnecting { .. } => 304, // CUDA_ERROR_OPERATING_SYSTEM
            ConnectionState::Failed => 304,        // CUDA_ERROR_OPERATING_SYSTEM
        }
    }
}
```

---

## Section 9: Implementation Phases

### Phase 2a: Context Management + Memory Transfers (Build First)

**Files to create/modify:**

| File | What |
|------|------|
| `crates/outterlink-common/src/protocol/mod.rs` | Protocol module root |
| `crates/outterlink-common/src/protocol/message.rs` | All message types and enums |
| `crates/outterlink-common/src/protocol/serialize.rs` | Serialization/deserialization |
| `crates/outterlink-common/src/protocol/transport.rs` | Transport trait + TCP implementation |
| `crates/outterlink-common/src/protocol/connection.rs` | Connection state machine |
| `crates/outterlink-common/src/handles.rs` | HandleTable + HandleMap |
| `crates/outterlink-common/src/pinned_pool.rs` | PinnedBufferPool |
| `crates/outterlink-client/src/intercept/context.rs` | cuCtx* interception |
| `crates/outterlink-client/src/intercept/memory.rs` | cuMemcpy* interception |
| `crates/outterlink-server/src/handler/context.rs` | Context request handlers |
| `crates/outterlink-server/src/handler/memory.rs` | Memory request handlers |

**Steps:**

1. Implement `MessageHeader` serialization + deserialization with unit tests
2. Implement `MessageType` enum and `from_u16` conversion
3. Implement `MessageFlags` bitflags
4. Implement `HandleMap` with `DashMap` and unit tests
5. Implement `HandleTable` (all handle types)
6. Implement `TcpTransport` with connection, request/response, and reader loop
7. Implement handshake flow (client sends Handshake, server responds HandshakeAck)
8. Implement `cuCtxCreate` / `cuCtxDestroy` interception (client) + handler (server)
9. Implement `cuMemAlloc` / `cuMemFree` (extend from P5 with proper handle translation)
10. Implement `cuMemcpyHtoD` (single message, no chunking first)
11. Implement `cuMemcpyDtoH`
12. Add chunking for transfers > 1 MB
13. Implement `PinnedBufferPool`
14. Integration test: alloc -> H2D -> D2H -> verify data matches

**Acceptance criteria:**
- [ ] 1 KB H2D + D2H round-trip: data matches exactly
- [ ] 1 MB H2D + D2H round-trip: data matches exactly
- [ ] 100 MB H2D + D2H round-trip: data matches exactly, uses chunking
- [ ] Handle translation works for CUcontext and CUdeviceptr
- [ ] Multiple threads can issue concurrent MemAlloc without races

### Phase 2b: Module Loading + Kernel Launch (Build Second)

**Additional files:**

| File | What |
|------|------|
| `crates/outterlink-client/src/intercept/module.rs` | cuModuleLoadData, cuModuleGetFunction |
| `crates/outterlink-client/src/intercept/kernel.rs` | cuLaunchKernel |
| `crates/outterlink-client/src/param_cache.rs` | Function parameter metadata cache |
| `crates/outterlink-server/src/handler/module.rs` | Module loading + metadata extraction |
| `crates/outterlink-server/src/handler/kernel.rs` | Kernel launch execution |

**Steps:**

1. Implement ModuleLoadData message (client sends PTX/cubin, server loads, returns handle)
2. Implement ModuleGetFunction message
3. Implement parameter metadata extraction on server (start with Strategy C: forward packed args)
4. Implement `extract_packed_args_from_extra` for cuLaunchKernel
5. Implement device pointer translation in packed args
6. Implement LaunchKernel message
7. Implement CtxSynchronize (wait for kernel completion)
8. End-to-end test: vector addition kernel

**Acceptance criteria:**
- [ ] PTX module loads on remote server
- [ ] cubin module loads on remote server
- [ ] Kernel function handle is resolved correctly
- [ ] Vector addition kernel (C[i] = A[i] + B[i]) runs correctly on remote GPU
- [ ] Results match local execution bit-for-bit
- [ ] Multiple kernel launches work in sequence

### Phase 2c: Batching + Error Handling + Polish

**Additional files:**

| File | What |
|------|------|
| `crates/outterlink-common/src/batch.rs` | BatchBuffer, batching logic |
| `crates/outterlink-common/src/error.rs` | TransportError, error mapping |

**Steps:**

1. Implement BatchBuffer with enqueue/drain
2. Add flush triggers to client interception layer
3. Implement heartbeat (client sends every 5s, server responds)
4. Implement reconnection state machine
5. Add timeout handling to all request types
6. Implement error mapping (TransportError -> CUresult)
7. Stress test: 10,000 alloc/free cycles
8. Stress test: 100 sequential kernel launches
9. Stress test: 4 threads doing independent kernel launches

**Acceptance criteria:**
- [ ] Batched cuMemFree + cuLaunchKernel reduces round-trips
- [ ] Server OOM returns CUDA_ERROR_OUT_OF_MEMORY to client
- [ ] Network timeout returns CUDA_ERROR_OPERATING_SYSTEM
- [ ] Heartbeat detects dead connection within 15 seconds
- [ ] 10,000 alloc/free cycles complete without handle leaks

---

## Testing Strategy

### Unit Tests

| Test | What It Validates |
|------|------------------|
| `test_header_round_trip` | Serialize -> deserialize header preserves all fields |
| `test_all_message_types` | Every MessageType serializes/deserializes correctly |
| `test_flags_round_trip` | All flag combinations survive serialization |
| `test_handle_map_concurrent` | 100 threads inserting/removing/looking up handles |
| `test_handle_map_remove_cleans_both_directions` | Both local->remote and remote->local entries removed |
| `test_handle_ranges_no_overlap` | Different handle types use non-overlapping ranges |
| `test_pinned_pool_reuse` | Acquire, release, re-acquire returns same buffer |
| `test_pinned_pool_size_classes` | Different sizes get different size-class buffers |
| `test_batch_buffer_flush_triggers` | Batch flushes at max_operations and max_bytes |
| `test_pointer_translation_in_args` | Device pointers in packed args are correctly translated |
| `test_pointer_translation_leaves_scalars` | Non-pointer 8-byte values are not modified |
| `test_extract_packed_args_from_extra` | CU_LAUNCH_PARAM_* parsing extracts correct buffer |

### Integration Tests (Require Real GPU on Server)

| Test | What It Validates |
|------|------------------|
| `test_handshake` | Client connects, handshakes, gets device info |
| `test_context_lifecycle` | Create context, set current, synchronize, destroy |
| `test_mem_alloc_free` | Allocate, free, verify handle cleanup |
| `test_memcpy_htod_dtoh_small` | 1 KB round-trip |
| `test_memcpy_htod_dtoh_large` | 100 MB round-trip with chunking |
| `test_memcpy_htod_dtoh_1gb` | 1 GB round-trip, verify no corruption |
| `test_memcpy_dtod` | Alloc two buffers, H2D first, D2D, D2H second, verify |
| `test_module_load_ptx` | Load PTX, get function handle |
| `test_module_load_cubin` | Load cubin, get function handle |
| `test_vector_add` | Full pipeline: load module, alloc, H2D, launch, D2H, verify |
| `test_matrix_multiply` | More complex kernel with shared memory |
| `test_concurrent_launches` | 4 threads each launching different kernels |
| `test_reconnection` | Kill server mid-operation, verify error, restart, reconnect |

### Property-Based Tests

| Test | What It Validates |
|------|------------------|
| `prop_serialization_round_trip` | For any random payload, serialize then deserialize == original |
| `prop_handle_map_consistent` | After random insert/remove/lookup sequence, maps are consistent |

---

## Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|-----------|
| Kernel argument sizes unknown (no metadata) | HIGH | MEDIUM | Strategy C: use `extra` parameter format which includes buffer size. Most apps use Runtime API which always uses `extra`. |
| Device pointer translation misidentifies a scalar as a pointer | HIGH | LOW | Pointers are in range 0x0D000000_00000000+, scalar values almost never match. Add server-side validation. |
| TCP head-of-line blocking for multi-threaded apps | MEDIUM | MEDIUM | Acceptable for P6. P8 adds multiple connections. |
| Chunked transfer corruption on network error | MEDIUM | LOW | Each chunk is independently validated. Server discards partial transfers on error. |
| Custom binary protocol has endianness bugs | MEDIUM | MEDIUM | Extensive unit tests with known byte patterns. Big-endian for headers (standard), little-endian for payloads (x86 native). |
| cubin ELF parsing fails for some CUDA versions | MEDIUM | MEDIUM | Strategy C does not require ELF parsing. Strategy A (ELF parsing) is optional and can fall back to Strategy B/C. |
| Pinned memory pool grows unbounded | LOW | LOW | Pool has configurable max_pinned limit. Buffers are recycled by size class. |

---

## Estimated Scope

| Component | Files | Approximate Lines |
|-----------|-------|-------------------|
| Protocol (message types, serialization) | 4 | ~1500 |
| Transport (trait + TCP impl) | 2 | ~800 |
| Handle translation | 1 | ~300 |
| Pinned buffer pool | 1 | ~200 |
| Batch buffer | 1 | ~150 |
| Error types + mapping | 1 | ~200 |
| Client interception (context, memory, module, kernel) | 4 | ~1200 |
| Server handlers (context, memory, module, kernel) | 4 | ~1000 |
| Unit tests | 4 | ~1000 |
| Integration tests | 2 | ~800 |
| **Total** | **~24** | **~7150** |

---

## Interface Contracts

### Input from P5 (PoC)

P6 extends the P5 codebase. It expects:
- Working TCP connection between client and server
- `cuInit`, `cuDeviceGet*` interception already functional
- Basic `cuMemAlloc` / `cuMemFree` working (P6 adds proper handle translation)
- Server framework that dispatches requests to handlers

### Output to P7 (CUDA Completeness)

P6 provides to P7:
- Complete protocol framework (message types, serialization, transport trait)
- Handle translation system ready for CUstream and CUevent handle types
- Batching infrastructure ready for stream-based flush triggers
- Pinned buffer pool for efficient memory transfers
- Tested kernel launch pipeline ready for stream-aware variants

P7 adds: `cuStreamCreate`, `cuStreamSynchronize`, `cuEventCreate`, `cuEventRecord`, `cuEventElapsedTime`, NVML interception, multi-GPU device enumeration.

---

## Dependencies (Rust Crates)

| Crate | Version | Purpose |
|-------|---------|---------|
| `tokio` | 1.x | Async runtime, TCP, timers |
| `tokio-util` | 0.7.x | Optional: LengthDelimitedCodec (may use for simpler framing alternative) |
| `dashmap` | 6.x | Concurrent hash map for handle tables |
| `bitflags` | 2.x | Message flags |
| `byteorder` | 1.x | Big/little-endian integer serialization |
| `thiserror` | 2.x | Error type derive |
| `async-trait` | 0.1.x | Async trait methods |
| `tracing` | 0.1.x | Structured logging |
| `proptest` | 1.x | Property-based testing (dev dependency) |

---

## Related Documents

- [Final Pre-Plan](../pre-planning/02-FINAL-PREPLAN.md)
- [R3: CUDA Interception](../research/R3-cuda-interception.md)
- [R4: Transport Stack](../research/R4-connectx5-transport-stack.md)
- [Research Consolidation](../research/CONSOLIDATION-all-research.md)
- [P1: GitHub Setup](P1-github-repo-setup.md)
