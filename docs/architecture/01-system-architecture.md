# OuterLink System Architecture

**Created:** 2026-03-23
**Last Updated:** 2026-03-23
**Status:** Draft -- reflects Phase 1 (host-staged TCP) implementation

## Purpose

Describe the internal structure of OuterLink: how data flows from a CUDA application through the interception layer, over the network, to the GPU, and back. This document covers every major subsystem with enough detail to understand the codebase without reading every source file.

---

## High-Level Data Flow

```
Application (PyTorch, cuDNN, cuBLAS, raw CUDA, etc.)
    |
    | CUDA Driver API calls: cuMemAlloc, cuLaunchKernel, cuCtxCreate, ...
    v
LD_PRELOAD Interception Layer  (csrc/interpose.c)
    |
    | Hooks dlsym() and cuGetProcAddress()
    | Returns hook_cu* function pointers instead of real libcuda.so symbols
    | On call: forwards to ol_* Rust FFI functions
    v
OuterLink Client Library  (Rust, libouterlink_client.so)
    |
    | Handle translation: local synthetic handle -> remote real handle
    | Serialize request into OLNK binary protocol payload
    | Send over TCP (main connection)
    | Block waiting for response
    | Deserialize response, translate handles back
    | Return result to application
    v
    [Network -- TCP, port 14833 (default)]
    v
OuterLink Server Daemon  (Rust, outerlink-server binary)
    |
    | Tokio async accept loop
    | Handshake: assign session_id, send HandshakeAck
    | Per-connection session (ConnectionSession): tracks all GPU resources
    | Dispatch to handle_request() based on MessageType
    v
GPU Backend  (GpuBackend trait)
    |
    | StubGpuBackend: in-process fake (testing, no GPU needed)
    | CudaGpuBackend: loads libcuda.so via libloading, calls real driver
    v
NVIDIA GPU Driver + Hardware
```

### Callback Data Path (separate channel)

```
Application calls cuStreamAddCallback(stream, fn, userData, flags)
    |
    v
OuterLink Client
    | Registers (callback_id -> fn_ptr, userData, stream) in CallbackRegistry
    | Sends StreamAddCallback message to server (only callback_id crosses wire)
    | On first callback: opens second TCP connection (CallbackChannelInit)
    v
    [Network -- second TCP connection, same port 14833]
    v
OuterLink Server
    | Associates callback channel with session via session_id in the SessionRegistry
    | When GPU executes the stream callback, server sends CallbackReady(callback_id, status)
    | on the callback channel
    v
OuterLink Client callback listener thread
    | Receives CallbackReady
    | Looks up fn_ptr, userData, and CallbackKind in CallbackRegistry
    | Releases registry mutex BEFORE invoking callback (re-entrancy safe)
    | Invokes the C callback: StreamAddCallback or LaunchHostFunc signature
    | Marks callback as completed, signals condvar for StreamSynchronize waiters
```

---

## Component Details

### 1. Interposition Layer (C)

**File:** `crates/outerlink-client/csrc/interpose.c`

The thin C layer that bridges the application's CUDA calls into the Rust client. It does the minimum possible work in C: look up names, forward to Rust.

**How it works:**

1. The `.so` is loaded into the process via `LD_PRELOAD`.
2. It overrides `dlsym()` using glibc's internal `__libc_dlsym(RTLD_NEXT, "dlsym")` to obtain the real implementation without infinite recursion.
3. It hooks `cuGetProcAddress()` and `cuGetProcAddress_v2()` -- the CUDA 11.3+ driver entry point discovery mechanism that modern CUDA runtimes prefer over `dlsym`.
4. It maintains a static `hook_table[]` mapping CUDA function names to `hook_cu*` function pointers.
5. When any code in the process calls `dlsym("cuMemAlloc_v2")`, our override runs first. It looks up the name in the hook table. If found, it returns the hook; otherwise it forwards to the real `dlsym`.
6. Each `hook_cu*` function has the exact CUDA Driver API signature. It calls `ensure_init()` (thread-safe via `pthread_once`) then forwards all arguments to the corresponding `ol_cu*` Rust FFI function.

**Initialization sequence:**

- `pthread_once` guarantees `do_init()` runs exactly once, even under concurrent first calls.
- `do_init()` resolves the real `dlsym` via `__libc_dlsym`, then calls `ol_client_init()` (Rust FFI).
- `ol_client_init()` reads `OUTERLINK_SERVER` from the environment, creates the `OuterLinkClient` singleton via `OnceLock`, and attempts to connect.

**Hook table statistics (169 entries):**

| Category | Entries (incl. version aliases) |
|----------|-------------------------------|
| Init + driver version | 2 |
| Device queries (get, count, name, attribute, UUID) | 6 |
| Context management | 18 |
| Primary context | 7 |
| Peer access | 4 |
| Memory -- basic (alloc, free, copy, set) | 27 |
| Memory -- async/stream-ordered | 4 |
| Memory -- pool (CUDA 11.2+) | 9 |
| Memory -- managed/unified | 5 |
| Memory -- host (alloc, free, register, flags) | 10 |
| Memory -- extended (address range, pitch, copy) | 6 |
| Module + function | 14 |
| Kernel launch (cuLaunchKernel, cooperative, ex) | 3 |
| Stream | 14 |
| Event | 8 |
| Occupancy | 4 |
| Pointer attributes | 2 |
| PCI Bus ID | 2 |
| Error strings | 2 |
| JIT Linker | 8 (covers _v2 aliases) |
| Library API (CUDA 12+) | 5 |
| CUDA Graphs (torch.compile reduce-overhead) | 14 (covers _v2 aliases) |
| Export table + cuGetProcAddress | 3 |
| **Total hook table entries** | **~169** |

Many entries are version aliases pointing to the same hook function (e.g., `cuStreamDestroy` and `cuStreamDestroy_v2` both map to `hook_cuStreamDestroy`). The number of unique intercepted functions (distinct behaviors) is approximately 144.

### 2. OuterLink Client (Rust)

**Crate:** `crates/outerlink-client`
**Build output:** `libouterlink_client.so` (cdylib for LD_PRELOAD) + rlib (for Rust test dependencies)

The global client singleton (`OuterLinkClient`) is initialized once via `OnceLock` on the first CUDA call.

**Key fields:**

| Field | Type | Purpose |
|-------|------|---------|
| `handles` | `HandleStore` | 12 bidirectional handle maps (see Handle Translation) |
| `connection` | `Mutex<Option<Arc<TcpTransportConnection>>>` | Main TCP connection to server |
| `callback_connection` | `Mutex<Option<Arc<TcpTransportConnection>>>` | Dedicated callback channel (lazy) |
| `callback_registry` | `Arc<CallbackRegistry>` | Maps callback_id to (fn_ptr, userData, stream, kind) |
| `callback_listener_running` | `Arc<AtomicBool>` | Tracks whether the listener thread is active |
| `runtime` | `tokio::runtime::Runtime` | 2-worker multi-thread runtime for async transport |
| `next_request_id` | `AtomicU64` | Monotonically increasing request ID for wire protocol |
| `current_remote_ctx` | `AtomicU64` | Tracks active remote context handle |
| `session_id` | `AtomicU64` | Assigned by server during handshake (bytes 4..12 of HandshakeAck) |
| `retry_config` | `RetryConfig` | Retry/reconnect parameters (see Retry section) |
| `reconnect_in_progress` | `Mutex<()>` | Serializes concurrent reconnect attempts |
| `server_addr` | `String` | Read from `OUTERLINK_SERVER` env var, defaults to `localhost:14833` |
| `connected` | `AtomicBool` | Can be checked without locking the connection mutex |

**FFI bridge (ffi.rs):**

Each `ol_cu*` function is `#[no_mangle] pub extern "C"`. The flow is:
1. Call `get_client()` to get the global singleton (lazy init via `OnceLock`).
2. If connected: serialize the request payload (little-endian), send via `send_request()`, receive response, deserialize, translate handles.
3. If disconnected: return stub values (synthetic handles, RTX 3090 properties, `CUDA_SUCCESS`).

**Stub mode:**

When the server is unreachable, the client returns plausible defaults so applications don't crash:
- `cuDeviceGetCount` returns 1
- `cuDeviceGetName` returns "NVIDIA GeForce RTX 3090"
- `cuMemAlloc` returns a synthetic handle from `STUB_HANDLE_COUNTER`
- `cuDeviceTotalMem` returns 24 GB
- Occupancy functions use Ampere/GA102 constants (82 SMs, 2048 threads/SM, 16 blocks/SM)
- Allocation sizes are tracked in a local `HashMap` for `cuMemGetAddressRange` consistency
- Peer access is tracked in a local `HashSet` for correct error code behavior

### 3. Wire Protocol

**Crate:** `crates/outerlink-common`, file `src/protocol.rs`

Custom binary protocol optimized for low overhead and debuggability.

**Header (22 bytes, fields are big-endian):**

```
Offset  Size  Field           Description
------  ----  -----           -----------
0       4     magic           Always "OLNK" (0x4F 0x4C 0x4E 0x4B)
4       2     version         Protocol version = 1
6       2     flags           Reserved = 0
8       8     request_id      Monotonically increasing u64
16      2     msg_type        MessageType enum value
18      4     payload_len     Payload size in bytes (max 256 MB)
```

**Payload (variable, little-endian / x86 native):**

- **Request:** fields specific to the CUDA function being called.
  - Example: `MemAlloc` request = `[size: u64]` (8 bytes)
  - Example: `MemcpyHtoD` request = `[dst: u64, size: u64]` + raw bulk data
- **Response:** always starts with 4-byte `CuResult` (u32 LE), followed by function-specific data.
  - Example: `MemAlloc` response = `[CuResult: u32, device_ptr: u64]` (12 bytes)

**Validation:** The server rejects frames where magic != "OLNK", version != 1, or payload_len > 256 MB (MAX_PAYLOAD_SIZE).

**Message type ranges (143 total message types across all categories):**

| Range | Category |
|-------|----------|
| 0x0001-0x0002 | Handshake / HandshakeAck |
| 0x0010-0x001B | Init + Device queries (12 types) |
| 0x0020-0x002F | Context management (16 types) |
| 0x0030-0x003F | Memory -- basic + memset (16 types) |
| 0x0040-0x004A | Module loading + function attributes (10 types) |
| 0x0047-0x0048 | Function configuration (cache, shared mem) |
| 0x0050-0x0056 | Kernel launch (3 types) |
| 0x0060-0x0068 | Stream (9 types) |
| 0x0070-0x0078 | Event (7 types) |
| 0x0080-0x0083 | Occupancy (4 types) |
| 0x0090-0x0091 | Peer access (2 types) |
| 0x00A0-0x00A5 | Context extended (6 types) |
| 0x00B0-0x00BB | Pointer attributes + memory extended (12 types) |
| 0x00BC-0x00CC | Memory managed + pool (17 types) |
| 0x00D0-0x00D4 | Callbacks (5 types) |
| 0x00D5-0x00D9 | Library API / CUDA 12+ (5 types) |
| 0x00E0-0x00ED | JIT Linker + CUDA Graphs (14 types) |
| 0x00F0 | Response |
| 0x00FF | Error |

**Handler dispatch** (`crates/outerlink-server/src/handler.rs`):

The `handle_request_full()` function matches on `msg_type`, deserializes the request payload (little-endian fields), calls the appropriate `GpuBackend` method, serializes a response with the 4-byte `CuResult` prefix, and returns a `HandleResult`. The `HandleResult` optionally includes a callback notification `(callback_id, cuda_status)` for callbacks that should be sent on the callback channel after the response is sent on the main channel.

### 4. Handle Translation

**File:** `crates/outerlink-common/src/handle.rs`

CUDA handle types (`CUcontext`, `CUdeviceptr`, `CUmodule`, etc.) are opaque pointers. The real handles exist on the server. The client cannot expose them directly to the application because:
- The application runs in a different address space from the GPU process
- The same remote handle value might collide with local memory addresses
- Debugging is easier when handle types are visually distinguishable

**Solution: Synthetic local handles with typed prefixes**

Each handle type gets a distinct high-byte prefix. A monotonic counter fills the lower bytes. This makes handles visually identifiable in debug output and prevents cross-type collisions.

| Handle Type | Prefix | Example |
|-------------|--------|---------|
| `CUcontext` | `0x0C00_0000_0000_0000` | `0x0C00_0000_0000_0001` |
| `CUdeviceptr` | `0x0D00_0000_0000_0000` | `0x0D00_0000_0000_0001` |
| `CUmodule` | `0x0E00_0000_0000_0000` | `0x0E00_0000_0000_0001` |
| `CUfunction` | `0x0F00_0000_0000_0000` | `0x0F00_0000_0000_0001` |
| `CUstream` | `0x1000_0000_0000_0000` | `0x1000_0000_0000_0001` |
| `CUevent` | `0x1100_0000_0000_0000` | `0x1100_0000_0000_0001` |
| `CUmemoryPool` | `0x1200_0000_0000_0000` | `0x1200_0000_0000_0001` |
| `CUlinkState` | `0x1300_0000_0000_0000` | `0x1300_0000_0000_0001` |
| `CUlibrary` (CUDA 12+) | `0x1400_0000_0000_0000` | `0x1400_0000_0000_0001` |
| `CUkernel` (CUDA 12+) | `0x1500_0000_0000_0000` | `0x1500_0000_0000_0001` |
| `CUgraph` | `0x1600_0000_0000_0000` | `0x1600_0000_0000_0001` |
| `CUgraphExec` | `0x1700_0000_0000_0000` | `0x1700_0000_0000_0001` |

**HandleMap internals:**

Each `HandleMap` wraps a pair of `DashMap` instances (sharded concurrent hash maps from the `dashmap` crate) for lock-free bidirectional lookups:
- `local_to_remote`: used when sending requests (translate app's synthetic handle to the real server handle)
- `remote_to_local`: used when receiving responses (translate server's real handle to the app's synthetic handle)
- `next_id`: `AtomicU64` counter for generating unique synthetic IDs

The `insert()` method uses `DashMap::entry()` to atomically check-or-insert, avoiding TOCTOU races. If the same remote handle is inserted twice (e.g., repeated `cuDevicePrimaryCtxRetain` calls), it returns the same local synthetic handle both times.

**HandleStore** contains one `HandleMap` per CUDA handle type (12 total). Created once in the `OuterLinkClient` and shared across all FFI calls.

### 5. OuterLink Server

**Crate:** `crates/outerlink-server`

Async Tokio server that accepts multiple concurrent client connections. Each connection is fully independent.

**Key components:**

| Component | File | Purpose |
|-----------|------|---------|
| `Server` | `src/server.rs` | TCP accept loop, graceful shutdown via `watch::channel`, JoinSet for connection tasks |
| `ConnectionSession` | `src/session.rs` | Per-connection state: current context, resource tracking, callback channel |
| `handle_request_full()` | `src/handler.rs` | Dispatch MessageType to GpuBackend call, serialize response |
| `GpuBackend` trait | `src/gpu_backend.rs` | Abstracts all GPU operations behind a `Send + Sync` trait |
| `StubGpuBackend` | `src/gpu_backend.rs` | In-memory fake GPU (RTX 3090 profile) for testing |
| `CudaGpuBackend` | `src/cuda_backend.rs` | Real NVIDIA driver loaded via `libloading` at runtime |

**Connection handling:**

1. Server binds a `TcpListener` on the configured address (default `0.0.0.0:14833`).
2. On each accepted connection, it peeks at the first message to determine connection type:
   - If `CallbackChannelInit`: this is a callback channel. The server looks up the session by `session_id` in the session registry and attaches the connection to it.
   - Otherwise: this is a normal client connection. The server assigns a unique `session_id`, creates a `ConnectionSession`, registers it in the session registry, handles the first message (usually `Handshake`), then enters the main request loop.
3. The main request loop reads messages, dispatches them through `handle_request_full()`, and sends responses. If a `HandleResult` includes a callback notification, it is sent on the session's callback channel.

**Graceful shutdown:**

1. `watch::Sender<()>` fires the shutdown signal (triggered by Ctrl+C or SIGTERM via `tokio::signal`).
2. The biased `tokio::select!` in the accept loop checks shutdown first, then accept.
3. After shutdown: stop accepting, wait up to 5 seconds (`DEFAULT_DRAIN_TIMEOUT`) for in-flight connection tasks to finish via the `JoinSet`.
4. Call `backend.shutdown()`.

**Session registry:** `Arc<TokioMutex<HashMap<session_id, Arc<TokioMutex<ConnectionSession>>>>>` shared between the main connection handler and the callback channel handler. When a callback channel sends `CallbackChannelInit(session_id)`, the server looks up the session in this registry and calls `set_callback_channel()`.

### 6. Session Management and Resource Cleanup

**File:** `crates/outerlink-server/src/session.rs`

`ConnectionSession` is created per client connection and holds all per-connection state. In CUDA, the "current context" is per-thread. In OuterLink's remote model, each client connection is the equivalent of a thread -- so the current context is tracked per-connection, not in the shared `GpuBackend`.

**Tracked resources:**

| Resource | Field type | Tracking granularity |
|----------|-----------|---------------------|
| Current CUDA context | `u64` | Single value per connection |
| Device memory allocations | `HashSet<u64>` | Device pointer handles |
| Pinned host memory | `HashSet<u64>` | Host pointer handles |
| CUDA contexts | `HashSet<u64>` | Context handles |
| Loaded modules | `HashSet<u64>` | Module handles |
| CUDA streams | `HashSet<u64>` | Stream handles |
| CUDA events | `HashSet<u64>` | Event handles |
| Primary contexts | `HashMap<i32, u64>` | Device ordinal to context handle |
| Registered host memory | `HashSet<u64>` | Registered pointer handles |
| Peer access contexts | `HashSet<u64>` | Peer context handles |
| Memory pools | `HashSet<u64>` | Pool handles |
| JIT link states | `HashSet<u64>` | Link state handles |
| CUDA 12 libraries | `HashSet<u64>` | Library handles |
| CUDA 12 kernels | `HashSet<u64>` | Kernel handles |
| CUDA graphs | `HashSet<u64>` | Graph handles |
| CUDA graph execs | `HashSet<u64>` | Graph execution handles |
| Callback channel | `Option<Arc<TcpTransportConnection>>` | One per session |

**Cleanup on disconnect:**

When a connection drops (gracefully or due to network failure), `cleanup()` frees all tracked resources via the backend in dependency order:

1. Graph execution instances (depend on graphs)
2. Graphs (may reference streams)
3. CUDA 12 libraries (wrap modules; kernels are cleared implicitly)
4. JIT link states (may reference module data)
5. Memory pools (pool-allocated memory should be freed first)
6. Events (depend on streams)
7. Streams
8. Modules
9. Device memory
10. Host memory (pinned + registered)
11. Peer access disabling
12. Primary context release
13. Contexts (last -- other resources depend on them)

Each cleanup step logs success/failure and produces a `CleanupReport` with counts.

### 7. GPU Backend

**File:** `crates/outerlink-server/src/gpu_backend.rs` (trait + stub), `src/cuda_backend.rs` (real)

The `GpuBackend` trait abstracts all GPU operations behind `Send + Sync`. This lets the entire server compile and run without a GPU.

**StubGpuBackend:**
- Simulates an RTX 3090 / GA102 (82 SMs, 2048 threads/SM, 16 blocks/SM, 24 GB VRAM)
- Allocates fake device memory as `Vec<u8>` in host RAM
- Tracks contexts, modules, streams, events, etc. in `HashMap`s protected by `Mutex`
- All operations succeed and return plausible values
- Used for all integration tests that don't require the `real-gpu-test` feature

**CudaGpuBackend:**
- Loads `libcuda.so` (Linux) at runtime via the `libloading` crate
- Resolves every needed CUDA Driver API symbol dynamically using the `require_fn!` macro
- Forwards all calls to the real NVIDIA driver, mapping return codes through `map_cuda_result()`
- Checks VRAM availability at startup (`check_vram()`)
- Selected at runtime via `--real-gpu` CLI flag

---

## Callback Architecture

Callbacks (`cuStreamAddCallback`, `cuLaunchHostFunc`) require the server to notify the client when a stream reaches a specific point in execution. The main request/response channel is strictly synchronous (request in, response out) and does not support server-initiated push.

**Solution: dedicated callback channel**

1. **Registration:** Client calls `cuStreamAddCallback(stream, fn, userData, flags)`. The FFI layer registers `(fn_ptr, userData, local_stream, CallbackKind::StreamAddCallback)` in the `CallbackRegistry` and obtains a unique `callback_id`. Only the `callback_id` + `remote_stream` cross the wire.

2. **Channel setup (lazy):** On the first callback registration, the client opens a second TCP connection to the same server port, sends `CallbackChannelInit(session_id)`. The server looks up the session in the registry, calls `session.set_callback_channel(conn)`, and replies with `CallbackChannelAck`. A listener thread is spawned in the client's Tokio runtime.

3. **Callback firing:** When the GPU executes past the callback point, the server's handler sends `CallbackReady(callback_id, cuda_status)` on the session's callback channel.

4. **Client-side invocation:** The callback listener thread receives `CallbackReady`, looks up `fn_ptr` + `userData` + `CallbackKind` in the registry. It releases the registry mutex BEFORE invoking the callback (re-entrancy safe -- the callback may call `cuStreamAddCallback` again). Based on `CallbackKind`:
   - `StreamAddCallback`: calls `fn_ptr(stream, status, user_data)`
   - `LaunchHostFunc`: calls `fn_ptr(user_data)`

5. **Completion signaling:** The callback is marked completed in the registry. A `Condvar` is signalled to wake any threads blocked in `cuStreamSynchronize` waiting for callbacks on that stream.

**Two-phase StreamSynchronize:** When `cuStreamSynchronize` is called, the client:
1. Sends `StreamSynchronize` to the server and waits for the response (which means the GPU stream is idle).
2. Then waits locally on the `Condvar` for all pending callbacks on that stream to complete.

This two-phase approach ensures the CUDA guarantee that all callbacks have fired before `cuStreamSynchronize` returns.

---

## Retry and Reconnect

**File:** `crates/outerlink-common/src/retry.rs`

**RetryConfig defaults:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `max_retries` | 3 | Per-request retries on transient errors (total 4 attempts) |
| `retry_delays` | [100ms, 500ms, 1000ms] | Delay before each retry |
| `max_reconnect_attempts` | 5 | Full TCP reconnect attempts |
| `reconnect_initial_delay` | 1s | First reconnect delay |
| `reconnect_max_delay` | 30s | Cap on exponential backoff |

**Retryable errors** (from `OuterLinkError::is_retryable()`):
- `Transport` -- network I/O failure
- `Connection` -- connection lost
- `ConnectionClosed` -- clean peer disconnect
- `Io` -- std::io::Error
- `Timeout` -- operation timed out

**Non-retryable errors** (definitive outcomes):
- `Cuda` -- GPU returned an error code
- `Protocol` -- malformed message
- `HandleNotFound` -- invalid handle
- `NotReady` -- server not initialized
- `Config` -- configuration error

**Reconnect behavior:**
- Only one thread can execute the reconnect loop at a time (`reconnect_in_progress` mutex prevents storms).
- Exponential backoff: delay = `initial_delay * 2^attempt`, capped at `max_delay`.
- Each reconnect attempt performs a full TCP connect + handshake.
- If another thread reconnects while we wait for the mutex, we check `is_actually_connected()` and skip.

**Connect timeout:** The initial TCP connect is bounded by a 10-second timeout to avoid blocking forever on unreachable hosts.

---

## Transport Layer

**Files:** `crates/outerlink-common/src/transport.rs`, `src/tcp_transport.rs`

**TransportConnection trait:**
- `send_message(header, payload)` -- send a framed message
- `recv_message() -> (header, payload)` -- receive a framed message
- `send_bulk(data)` -- send raw bulk data (for large memory transfers, bypasses protocol framing)
- `recv_bulk(size)` -- receive raw bulk data of known size
- `is_connected()` -- check connection liveness
- `close()` -- graceful disconnect

**TcpTransportConnection:**
- Wraps a `tokio::net::TcpStream` split into `OwnedReadHalf` / `OwnedWriteHalf` behind `Mutex`es
- `is_connected` tracked via `AtomicBool`, set to false on any I/O error

**TransportFactory** and **TransportListener** traits exist for future transport backends (UCX, OpenDMA). Currently only TCP is implemented.

All transport calls from the synchronous FFI layer are bridged to async Tokio via `runtime.block_on()`.

---

## Error Handling

**File:** `crates/outerlink-common/src/error.rs`

`OuterLinkError` is the central error type with variants for every failure mode. Each variant maps to a `CuResult` code via `to_cuda_result()`, allowing the FFI layer to return a meaningful CUDA error code to the application:

| OuterLinkError variant | Maps to CuResult |
|----------------------|-----------------|
| `Cuda(r)` | Pass-through `r` |
| `Transport` / `Connection` / `ConnectionClosed` / `Io` | `TransportError` |
| `Protocol` | `RemoteError` |
| `HandleNotFound` | `HandleNotFound` |
| `Timeout` | `Timeout` |
| `NotReady` | `SystemNotReady` |
| `Config` | `InvalidValue` |

---

## Crate Dependencies

| Crate | Key dependencies |
|-------|-----------------|
| `outerlink-common` | tokio, dashmap, thiserror, async-trait, bytes |
| `outerlink-client` | outerlink-common, tokio, tracing, cc (build) |
| `outerlink-server` | outerlink-common, tokio, tracing, clap, anyhow, libloading |
| `outerlink-cli` | outerlink-common, clap, tokio (skeleton) |

---

## Related Documents

- [Project Vision](00-project-vision.md)
- [Installation Guide](../guides/01-installation.md)
- [Quickstart Guide](../guides/02-quickstart.md)
- [Testing on Linux](../guides/02-testing-on-linux.md)
- [ADR-001: License](../decisions/ADR-001-license.md)
- [ADR-002: OpenDMA Naming](../decisions/ADR-002-opendma-naming.md)

## Open Questions

- [ ] Should handle translation be moved into the server? (Currently client-side only -- server returns raw CUDA handles)
- [ ] How does handle translation interact with multi-session scenarios where the same remote GPU is shared?
- [ ] What is the reconnect behavior for the callback channel when the main connection reconnects?
- [ ] Does `cuStreamSynchronize` two-phase approach cover all edge cases (e.g., callback fires between the server sync response arriving and the client checking the condvar)?
- [ ] When will the transport layer switch from plain TCP to UCX (auto-RDMA negotiation)?
- [ ] What payload size threshold should trigger `send_bulk` / `recv_bulk` instead of embedding data in the protocol frame?
