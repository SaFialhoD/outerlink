# R20: NCCL Backend -- Pre-Plan v2 (Refined)

**Date Created:** 2026-03-25
**Date Last Updated:** 2026-03-25
**Status:** DRAFT
**Revision:** v2 -- Cross-topic refinement. Resolves open questions from v1, adds FFI interface, transport adapter mapping, gradient compression integration, multi-transport routing, and concrete test plan.

## Purpose

Define the complete implementation blueprint for `libnccl-net-outerlink.so`, a Rust-based NCCL transport plugin that registers OuterLink as a custom NCCL backend. This v2 refines the original preplan by resolving all 9 open questions using findings from R10 (Memory Tiering), R14 (Compression), R17 (Topology), R23 (Heterogeneous GPUs), R25 (Kernel Splitting), R26 (PTP Clock), R28 (Scatter-Gather), R29 (Multicast), and R12 (Dedup).

---

## 1. Resolved Open Questions

### Q1: Should the plugin communicate with OuterLink server directly, or through the client library?

**Answer: Direct connection to the OuterLink server, NOT through the client library.**

Rationale:
- The NCCL plugin is loaded by the application process via `dlopen`. The OuterLink client (`libnccl-net-outerlink.so`) is also loaded via `LD_PRELOAD` in the same process. They coexist in the same address space.
- However, NCCL's plugin is loaded by NCCL itself, which runs inside the application. If the plugin depends on LD_PRELOAD being active, we create a fragile circular dependency: NCCL loads our plugin, our plugin calls the client library, the client library intercepts CUDA calls... this creates re-entrancy hazards.
- The plugin opens its own transport connections to the OuterLink server daemon. This makes the plugin self-contained: `NCCL_NET_PLUGIN=outerlink` works whether or not `LD_PRELOAD` is active.
- The plugin links directly against `outerlink-common` (shared protocol/types) and `outerlink-transport` (TCP/RDMA/USB4 transport layer) crates. No dependency on `outerlink-client`.

**Implementation:** During `init()`, the plugin reads `OUTERLINK_SERVER` environment variable (default `127.0.0.1:9876`) and establishes a control connection to the server. Data connections are created per-comm during `listen`/`connect`/`accept`.

### Q2: Connection pooling -- one per NCCL comm or pooled?

**Answer: One dedicated transport connection per NCCL comm, with shared connection pooling at the transport layer.**

Rationale:
- NCCL creates multiple channels (typically 4-16), each with its own send/recv comm pairs. Each comm handles up to `NCCL_NET_MAX_REQUESTS` (8) concurrent async operations.
- A dedicated transport connection per comm avoids head-of-line blocking between channels and simplifies ordering guarantees (NCCL expects in-order completion per comm).
- The transport layer already supports connection pooling internally. The plugin creates connections; the transport layer manages the underlying socket/QP resources efficiently.
- With 8 channels and 2 peer nodes, we get 16 send + 16 recv = 32 transport connections per node pair. This is well within ConnectX-5 QP limits (~64K QPs) and TCP connection limits.

**Implementation:** `listen()` creates a transport listener. `connect()`/`accept()` each create a new transport connection. The `OuterLinkComm` struct holds one `TransportConnection` handle.

### Q3: Handle format -- what goes in the NCCL handle?

**Answer: 64-byte serialized handle containing transport endpoint info and capability bitmap.**

`NCCL_NET_HANDLE_MAXSIZE` is 128 bytes in current NCCL. Our handle uses 64 bytes:

```
Bytes 0-3:   magic (0x4F4C4E4B = "OLNK")
Byte  4:     version (1)
Byte  5:     transport_type (0=TCP, 1=RDMA, 2=USB4, 3=OpenDMA)
Bytes 6-7:   port (u16, network byte order)
Bytes 8-23:  address (IPv6-mapped, 16 bytes -- covers both IPv4 and IPv6)
Bytes 24-27: RDMA QPN (u32, for RDMA transport) or 0
Bytes 28-43: RDMA GID (16 bytes, for RDMA transport) or 0
Bytes 44-47: capability_flags (u32 bitfield):
               bit 0: supports_compression (R14)
               bit 1: supports_opendma
               bit 2: supports_cuda_ptr
               bit 3: supports_multicast (R29)
               bits 4-7: max_sge_count (R28, 0-15 maps to 1-16)
               bits 8-31: reserved
Bytes 48-55: node_guid (u64, for topology)
Bytes 56-63: reserved (zero-padded)
```

This fits well within the 128-byte limit. The capability flags enable transport negotiation: if both sides support RDMA, use RDMA; otherwise fall back to TCP.

### Q4: Request pool sizing

**Answer: 64 pre-allocated requests per comm.**

Rationale:
- `NCCL_NET_MAX_REQUESTS` = 8 (the maximum concurrent requests NCCL will issue per comm).
- With `maxRecvs` = 8 (grouped receives), the theoretical max is 8 * 8 = 64 concurrent buffer operations per comm.
- Pre-allocate 64 `OuterLinkRequest` structs per comm in a slab allocator. No heap allocation in the data path.
- Each request is ~128 bytes (state enum, completion flag, size, transport request handle, timestamp). Total: 64 * 128 = 8 KB per comm. With 32 comms per peer pair, that is 256 KB -- negligible.

**Implementation:** `RequestPool` with fixed-size array and bitmap allocator. `isend`/`irecv` grab a slot, `test` marks it complete, and the slot is returned when NCCL moves on.

### Q5: regMr caching strategy

**Answer: LRU cache with `regIsGlobal=1`, keyed by (address, size, type) tuple. 4096-entry cache per comm.**

Rationale:
- When `regIsGlobal=1`, NCCL pre-registers buffers and reuses them across operations. The plugin maintains a cache so repeated `regMr` calls for the same region return the existing handle.
- LRU eviction with 4096 entries covers the typical NCCL buffer pool (ring buffers, scratch buffers). NCCL reuses a fixed set of buffers.
- Cache lookup is O(1) via HashMap keyed on `(data_ptr as usize, size, type)`.

**Interaction with R10 (VRAM Manager):**
- When `type == NCCL_PTR_CUDA`, the plugin asks the VRAM manager to pin the page(s) covering the registered region. R10 uses 64KB pages, so a registration of N bytes pins `ceil(N / 64KB)` pages.
- The VRAM manager marks these pages as "NCCL-registered" to prevent eviction during active use.
- `deregMr` unpins the pages, allowing the tier manager to evict them again.
- For host memory (`NCCL_PTR_HOST`), `regMr` pins host pages using `mlock` + RDMA MR registration if on the RDMA transport.

### Q6: Zero-copy threshold

**Answer: 8 KB crossover for RDMA; always copy for TCP.**

Rationale:
- For RDMA, memory registration (`ibv_reg_mr`) costs ~10-50 us. For messages < 8 KB, inline send (copy into WQE) is faster than register-send-deregister.
- With `regIsGlobal=1` and caching (Q5), registrations are amortized -- the threshold only matters for first-time registrations.
- For TCP, all transfers go through pinned host bounce buffers anyway. The copy is inherent.
- For OpenDMA, the threshold is 0 (always zero-copy) since BAR1 regions are pre-mapped.

**Implementation:** The transport adapter checks message size and registration state. Below threshold: memcpy to pre-registered inline buffer, post inline send. Above threshold: use registered MR directly.

### Q7: Minimum NCCL version to test against

**Answer: Test against NCCL 2.19+ (v8 API). Do NOT target v6.**

Rationale:
- v8 adds `getDeviceMr`, `irecvConsumed`, `regIsGlobal`, and `netDeviceType` -- all needed for performance.
- NCCL 2.19 is the minimum in current PyTorch (2.x) and DeepSpeed (0.14+) distributions.
- Supporting v6 means giving up `regIsGlobal` (no registration caching) and `getDeviceMr` (no GPU direct). The complexity of maintaining v6 support does not justify targeting EOL stacks.
- Test matrix: NCCL 2.19 (v8), 2.22 (v9), 2.25 (v10), 2.28 (v11).

### Q8: CUDA version requirements

**Answer: CUDA Driver API only. No CUDA runtime dependency. Works with CUDA 11.0+.**

Rationale:
- The plugin uses `cuMemAlloc`, `cuMemcpyAsync`, `cuStreamSynchronize` for GPU operations. All are driver API.
- No CUDA runtime (`cudart`) dependency. The plugin dynamically loads `libcuda.so` at init time.
- This means the plugin works with any CUDA toolkit version 11.0+ installed on the system.
- For nvCOMP (R14 compression integration), nvCOMP has its own CUDA runtime dependency, but that is optional and loaded dynamically.

### Q9: ARM support (Grace Hopper)

**Answer: Not in scope for initial implementation. x86_64 Linux only.**

Rationale:
- OuterLink targets Pedro's multi-PC setup with x86_64 + GeForce GPUs. ARM/Grace Hopper is a data center platform.
- The Rust crate compiles for x86_64-unknown-linux-gnu. ARM compilation is untested.
- No architectural blockers -- Rust cross-compilation and the NCCL API are platform-agnostic. ARM support can be added later by cross-compiling and testing.
- Grace Hopper has NVLink C2C (900 GB/s chip-to-chip), making OuterLink's transport irrelevant on that platform.

---

## 2. Exact FFI Interface

### 2.1 Core Types (`ffi_types.rs`)

```rust
//! NCCL Net Plugin FFI type definitions.
//! These structs match the C definitions in NCCL's net.h headers exactly.
//! Manual definitions -- no bindgen dependency.

use std::os::raw::{c_char, c_int, c_void};

/// NCCL result codes
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcclResult {
    Success = 0,
    UnhandledCudaError = 1,
    SystemError = 2,
    InternalError = 3,
    InvalidArgument = 4,
    InvalidUsage = 5,
    RemoteError = 6,
    InProgress = 7,
}

/// NCCL pointer types (bitfield)
pub const NCCL_PTR_HOST: c_int = 1;
pub const NCCL_PTR_CUDA: c_int = 2;
pub const NCCL_PTR_DMABUF: c_int = 4;

/// Maximum handle size for connection exchange
pub const NCCL_NET_HANDLE_MAXSIZE: usize = 128;

/// Maximum concurrent async requests per comm
pub const NCCL_NET_MAX_REQUESTS: usize = 8;

/// NCCL debug logger callback type
pub type NcclDebugLoggerFn = Option<
    unsafe extern "C" fn(
        level: c_int,
        flags: u64,
        file: *const c_char,
        line: c_int,
        fmt: *const c_char,
        ...
    ),
>;

/// NCCL profiler callback type (v10+)
pub type NcclProfilerCallbackFn = Option<unsafe extern "C" fn()>; // simplified; full signature TBD

/// Network device handle for GPU-initiated networking (GDR device-side API)
#[repr(C)]
pub struct NcclNetDeviceHandle_v8 {
    pub net_dev_type: c_int,
    pub dev_handle: [u8; 120], // opaque device handle
}

/// Network properties reported per device
#[repr(C)]
pub struct NcclNetProperties_v8 {
    pub name: *const c_char,
    pub pci_path: *const c_char,
    pub guid: u64,
    pub ptr_support: c_int,        // NCCL_PTR_HOST | NCCL_PTR_CUDA | NCCL_PTR_DMABUF
    pub reg_is_global: c_int,      // 1 = registration cache enabled
    pub speed: c_int,              // Link speed in Mbps
    pub port: c_int,               // Port number
    pub latency: f32,              // Network latency in microseconds
    pub max_comms: c_int,          // Maximum connections
    pub max_recvs: c_int,          // Max buffers per grouped receive
    pub net_device_type: c_int,    // Device type for GDR
    pub net_device_version: c_int, // Device API version
}

/// The v8 plugin vtable -- 19 function pointers + name
#[repr(C)]
pub struct NcclNet_v8 {
    pub name: *const c_char,

    pub init: unsafe extern "C" fn(log_fn: NcclDebugLoggerFn) -> NcclResult,

    pub devices: unsafe extern "C" fn(ndev: *mut c_int) -> NcclResult,

    pub get_properties:
        unsafe extern "C" fn(dev: c_int, props: *mut NcclNetProperties_v8) -> NcclResult,

    pub listen: unsafe extern "C" fn(
        dev: c_int,
        handle: *mut c_void,
        listen_comm: *mut *mut c_void,
    ) -> NcclResult,

    pub connect: unsafe extern "C" fn(
        dev: c_int,
        handle: *mut c_void,
        send_comm: *mut *mut c_void,
        send_dev_comm: *mut *mut NcclNetDeviceHandle_v8,
    ) -> NcclResult,

    pub accept: unsafe extern "C" fn(
        listen_comm: *mut c_void,
        recv_comm: *mut *mut c_void,
        recv_dev_comm: *mut *mut NcclNetDeviceHandle_v8,
    ) -> NcclResult,

    pub reg_mr: unsafe extern "C" fn(
        comm: *mut c_void,
        data: *mut c_void,
        size: c_int,
        mr_type: c_int,
        mhandle: *mut *mut c_void,
    ) -> NcclResult,

    pub reg_mr_dma_buf: unsafe extern "C" fn(
        comm: *mut c_void,
        data: *mut c_void,
        size: c_int,
        mr_type: c_int,
        offset: u64,
        fd: c_int,
        mhandle: *mut *mut c_void,
    ) -> NcclResult,

    pub dereg_mr:
        unsafe extern "C" fn(comm: *mut c_void, mhandle: *mut c_void) -> NcclResult,

    pub isend: unsafe extern "C" fn(
        send_comm: *mut c_void,
        data: *mut c_void,
        size: c_int,
        tag: c_int,
        mhandle: *mut c_void,
        request: *mut *mut c_void,
    ) -> NcclResult,

    pub irecv: unsafe extern "C" fn(
        recv_comm: *mut c_void,
        n: c_int,
        data: *mut *mut c_void,
        sizes: *mut c_int,
        tags: *mut c_int,
        mhandles: *mut *mut c_void,
        request: *mut *mut c_void,
    ) -> NcclResult,

    pub iflush: unsafe extern "C" fn(
        recv_comm: *mut c_void,
        n: c_int,
        data: *mut *mut c_void,
        sizes: *mut c_int,
        mhandles: *mut *mut c_void,
        request: *mut *mut c_void,
    ) -> NcclResult,

    pub test: unsafe extern "C" fn(
        request: *mut c_void,
        done: *mut c_int,
        sizes: *mut c_int,
    ) -> NcclResult,

    pub close_send: unsafe extern "C" fn(send_comm: *mut c_void) -> NcclResult,

    pub close_recv: unsafe extern "C" fn(recv_comm: *mut c_void) -> NcclResult,

    pub close_listen: unsafe extern "C" fn(listen_comm: *mut c_void) -> NcclResult,

    pub get_device_mr: unsafe extern "C" fn(
        comm: *mut c_void,
        mhandle: *mut c_void,
        dptr_mhandle: *mut *mut c_void,
    ) -> NcclResult,

    pub irecv_consumed: unsafe extern "C" fn(
        recv_comm: *mut c_void,
        n: c_int,
        request: *mut c_void,
    ) -> NcclResult,
}

// --- Version shim types ---

/// v9: size_t sizes, adds makeVDevice
#[repr(C)]
pub struct NcclNetProperties_v9 {
    // Same as v8 but with additional virtual device fields
    pub v8: NcclNetProperties_v8,
}

/// v9 virtual device properties
#[repr(C)]
pub struct NcclNetVDeviceProps_v9 {
    pub n_devices: c_int,
    pub dev_indices: [c_int; 16],
}

/// v10 communicator config
#[repr(C)]
pub struct NcclNetCommConfig_v10 {
    pub comm_index: c_int,
    pub n_ranks: c_int,
    // Additional config fields TBD from NCCL headers
}

/// v11 communicator config (extends v10)
#[repr(C)]
pub struct NcclNetCommConfig_v11 {
    pub v10: NcclNetCommConfig_v10,
    pub max_multi_request_size: usize,
}
```

### 2.2 FFI Export Functions (`lib.rs`)

```rust
//! libnccl-net-outerlink.so entry point.
//! Exports versioned ncclNet_vX symbols that NCCL discovers via dlsym.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};
use std::panic;

mod ffi_types;
mod plugin;
mod transport_adapter;
mod device_manager;
mod memory_registry;
mod connection;
mod async_ops;
mod version_shims;

use ffi_types::*;

/// Plugin name as a C string (static lifetime)
static PLUGIN_NAME: &[u8] = b"outerlink\0";

// ---------- FFI boundary pattern ----------
//
// Every exported function wraps its body in catch_unwind.
// A Rust panic must NEVER cross the FFI boundary into NCCL's C code.
// On panic, we log the error and return NcclResult::InternalError.

macro_rules! ffi_guard {
    ($body:expr) => {
        match panic::catch_unwind(panic::AssertUnwindSafe(|| $body)) {
            Ok(result) => result,
            Err(_) => {
                // Panic caught at FFI boundary. Log if logger available.
                plugin::log_error("Rust panic caught at FFI boundary");
                NcclResult::InternalError
            }
        }
    };
}

// ---------- v8 function implementations ----------

unsafe extern "C" fn ol_init(log_fn: NcclDebugLoggerFn) -> NcclResult {
    ffi_guard!({
        plugin::initialize(log_fn)
    })
}

unsafe extern "C" fn ol_devices(ndev: *mut c_int) -> NcclResult {
    ffi_guard!({
        if ndev.is_null() { return NcclResult::InvalidArgument; }
        let count = device_manager::get_device_count();
        *ndev = count as c_int;
        NcclResult::Success
    })
}

unsafe extern "C" fn ol_get_properties(
    dev: c_int,
    props: *mut NcclNetProperties_v8,
) -> NcclResult {
    ffi_guard!({
        if props.is_null() { return NcclResult::InvalidArgument; }
        device_manager::fill_properties(dev as usize, &mut *props)
    })
}

unsafe extern "C" fn ol_listen(
    dev: c_int,
    handle: *mut c_void,
    listen_comm: *mut *mut c_void,
) -> NcclResult {
    ffi_guard!({
        connection::listen(dev as usize, handle, listen_comm)
    })
}

unsafe extern "C" fn ol_connect(
    dev: c_int,
    handle: *mut c_void,
    send_comm: *mut *mut c_void,
    send_dev_comm: *mut *mut NcclNetDeviceHandle_v8,
) -> NcclResult {
    ffi_guard!({
        connection::connect(dev as usize, handle, send_comm, send_dev_comm)
    })
}

unsafe extern "C" fn ol_accept(
    listen_comm: *mut c_void,
    recv_comm: *mut *mut c_void,
    recv_dev_comm: *mut *mut NcclNetDeviceHandle_v8,
) -> NcclResult {
    ffi_guard!({
        connection::accept(listen_comm, recv_comm, recv_dev_comm)
    })
}

unsafe extern "C" fn ol_reg_mr(
    comm: *mut c_void,
    data: *mut c_void,
    size: c_int,
    mr_type: c_int,
    mhandle: *mut *mut c_void,
) -> NcclResult {
    ffi_guard!({
        memory_registry::reg_mr(comm, data, size as usize, mr_type, mhandle)
    })
}

unsafe extern "C" fn ol_reg_mr_dma_buf(
    comm: *mut c_void,
    data: *mut c_void,
    size: c_int,
    mr_type: c_int,
    offset: u64,
    fd: c_int,
    mhandle: *mut *mut c_void,
) -> NcclResult {
    ffi_guard!({
        memory_registry::reg_mr_dma_buf(comm, data, size as usize, mr_type, offset, fd, mhandle)
    })
}

unsafe extern "C" fn ol_dereg_mr(
    comm: *mut c_void,
    mhandle: *mut c_void,
) -> NcclResult {
    ffi_guard!({
        memory_registry::dereg_mr(comm, mhandle)
    })
}

unsafe extern "C" fn ol_isend(
    send_comm: *mut c_void,
    data: *mut c_void,
    size: c_int,
    tag: c_int,
    mhandle: *mut c_void,
    request: *mut *mut c_void,
) -> NcclResult {
    ffi_guard!({
        async_ops::isend(send_comm, data, size as usize, tag, mhandle, request)
    })
}

unsafe extern "C" fn ol_irecv(
    recv_comm: *mut c_void,
    n: c_int,
    data: *mut *mut c_void,
    sizes: *mut c_int,
    tags: *mut c_int,
    mhandles: *mut *mut c_void,
    request: *mut *mut c_void,
) -> NcclResult {
    ffi_guard!({
        async_ops::irecv(recv_comm, n, data, sizes, tags, mhandles, request)
    })
}

unsafe extern "C" fn ol_iflush(
    recv_comm: *mut c_void,
    n: c_int,
    data: *mut *mut c_void,
    sizes: *mut c_int,
    mhandles: *mut *mut c_void,
    request: *mut *mut c_void,
) -> NcclResult {
    ffi_guard!({
        async_ops::iflush(recv_comm, n, data, sizes, mhandles, request)
    })
}

unsafe extern "C" fn ol_test(
    request: *mut c_void,
    done: *mut c_int,
    sizes: *mut c_int,
) -> NcclResult {
    ffi_guard!({
        async_ops::test(request, done, sizes)
    })
}

unsafe extern "C" fn ol_close_send(send_comm: *mut c_void) -> NcclResult {
    ffi_guard!({ connection::close_send(send_comm) })
}

unsafe extern "C" fn ol_close_recv(recv_comm: *mut c_void) -> NcclResult {
    ffi_guard!({ connection::close_recv(recv_comm) })
}

unsafe extern "C" fn ol_close_listen(listen_comm: *mut c_void) -> NcclResult {
    ffi_guard!({ connection::close_listen(listen_comm) })
}

unsafe extern "C" fn ol_get_device_mr(
    comm: *mut c_void,
    mhandle: *mut c_void,
    dptr_mhandle: *mut *mut c_void,
) -> NcclResult {
    ffi_guard!({
        // GDR device-side API -- not implemented in Phase 1.
        // Return NULL device handle; NCCL falls back to host path.
        if !dptr_mhandle.is_null() {
            *dptr_mhandle = std::ptr::null_mut();
        }
        NcclResult::Success
    })
}

unsafe extern "C" fn ol_irecv_consumed(
    recv_comm: *mut c_void,
    n: c_int,
    request: *mut c_void,
) -> NcclResult {
    ffi_guard!({
        async_ops::irecv_consumed(recv_comm, n, request)
    })
}

// ---------- Static vtable ----------

#[no_mangle]
pub static ncclNet_v8: NcclNet_v8 = NcclNet_v8 {
    name: PLUGIN_NAME.as_ptr() as *const c_char,
    init: ol_init,
    devices: ol_devices,
    get_properties: ol_get_properties,
    listen: ol_listen,
    connect: ol_connect,
    accept: ol_accept,
    reg_mr: ol_reg_mr,
    reg_mr_dma_buf: ol_reg_mr_dma_buf,
    dereg_mr: ol_dereg_mr,
    isend: ol_isend,
    irecv: ol_irecv,
    iflush: ol_iflush,
    test: ol_test,
    close_send: ol_close_send,
    close_recv: ol_close_recv,
    close_listen: ol_close_listen,
    get_device_mr: ol_get_device_mr,
    irecv_consumed: ol_irecv_consumed,
};

// v9, v10, v11 symbols are generated by version_shims module.
// They wrap the v8 functions with appropriate type conversions.
// See version_shims.rs for the shim implementations.
```

### 2.3 Symbol Export Mechanism

The crate is built as `cdylib`:

```toml
# crates/outerlink-nccl-plugin/Cargo.toml
[package]
name = "outerlink-nccl-plugin"
version = "0.1.0"
edition = "2021"

[lib]
name = "nccl_net_outerlink"   # produces libnccl-net-outerlink.so
crate-type = ["cdylib"]

[dependencies]
outerlink-common = { path = "../outerlink-common" }
# outerlink-transport = { path = "../outerlink-transport" }  # when ready
libc = "0.2"

[profile.release]
lto = true           # link-time optimization for minimal overhead
panic = "abort"      # no unwinding across FFI (catch_unwind still works)
```

Symbol visibility is controlled by `#[no_mangle]` on the static vtable variables. Rust `cdylib` exports all `#[no_mangle]` symbols by default. No linker script needed.

Verification: `nm -D target/release/libnccl-net-outerlink.so | grep ncclNet` must show:
```
D ncclNet_v8
D ncclNet_v9
D ncclNet_v10
D ncclNet_v11
```

### 2.4 Error Handling Pattern

Every FFI function uses the `ffi_guard!` macro shown above. The pattern:

1. `panic::catch_unwind` wraps the entire function body.
2. On panic: log via NCCL's logger (if available), return `NcclResult::InternalError`.
3. On Rust `Result::Err`: convert to appropriate `NcclResult` variant.
4. No `?` operator at the FFI boundary -- all Results are explicitly matched.

The `panic = "abort"` in Cargo.toml is for the final binary. During development, `panic = "unwind"` allows `catch_unwind` to work. In release builds, `catch_unwind` becomes a no-op if panics abort, but we keep the macro for consistency.

---

## 3. Transport Adapter Mapping

### 3.1 NCCL Operation to OuterLink Transport Mapping

| NCCL Plugin Function | OuterLink Transport Call | Notes |
|---------------------|------------------------|-------|
| `init()` | `TransportManager::new()` | Initialize transport layer, discover available transports |
| `devices()` | `TransportManager::list_transports()` | Returns count of available transport endpoints |
| `getProperties(dev)` | `Transport::properties()` | Read PCI path, speed, latency from transport |
| `listen(dev)` | `Transport::bind(addr)` | Create listening socket/QP on the transport |
| `connect(dev, handle)` | `Transport::connect(remote_addr)` | Non-blocking connect to remote endpoint |
| `accept(listenComm)` | `Listener::try_accept()` | Non-blocking accept (returns None if pending) |
| `regMr(data, size, HOST)` | `mlock(data, size)` + `ibv_reg_mr()` (RDMA) | Pin host memory, register with NIC if RDMA |
| `regMr(data, size, CUDA)` | `VramManager::pin_pages(data, size)` (R10) | Pin VRAM pages, prevent tier eviction |
| `regMrDmaBuf(fd)` | `Transport::reg_dmabuf(fd, offset, size)` | DMA-BUF registration for GPU memory |
| `deregMr(mhandle)` | `ibv_dereg_mr()` + `munlock()` / `VramManager::unpin()` | Reverse of regMr |
| `isend(data, size)` | `Transport::send_async(data, size)` | Post async send, return request handle |
| `irecv(n, data[], sizes[])` | `Transport::recv_async(bufs[])` | Post grouped async recv using scatter-gather (R28) |
| `iflush(data[], sizes[])` | `Transport::flush()` or no-op | PCIe write fence for RDMA/OpenDMA paths |
| `test(request)` | `Transport::poll_completion(req)` | Check CQ (RDMA) or io_uring CQE (TCP) |
| `closeSend/Recv/Listen` | `Transport::close()` | Teardown connection, free resources |

### 3.2 Non-Blocking Guarantee

NCCL requires `listen`, `connect`, and `accept` to be non-blocking. Implementation:

- **`listen`:** Creates a socket/QP in non-blocking mode. Always succeeds immediately (bind + listen are synchronous but fast). Returns `listenComm` immediately.
- **`connect`:** Initiates a non-blocking TCP connect or RDMA connection request. If not yet connected, returns `ncclSuccess` with `*sendComm = NULL`. NCCL retries.
- **`accept`:** Calls `try_accept()` on the listener. If no pending connection, returns `ncclSuccess` with `*recvComm = NULL`. NCCL retries.

Internal state machine per comm:
```
Pending -> Connecting -> Connected -> Active -> Closing -> Closed
```

### 3.3 regMr Interaction with R10 VRAM Manager

When NCCL registers GPU memory:

```
regMr(comm, gpu_ptr, size, NCCL_PTR_CUDA, &mhandle)
  |
  v
1. Look up gpu_ptr in R10's page table
   -> Resolves to physical pages (64KB granularity, R10 Decision 1)
   -> Pages may be on Tier 0 (local VRAM), Tier 1 (remote VRAM), or Tier 2 (DRAM)
  |
  v
2. If pages are NOT in local VRAM:
   -> Request R10 migration engine to promote pages to Tier 0
   -> Block until migration completes (regMr can be slow; this is acceptable)
  |
  v
3. Pin pages in VRAM:
   -> Set "NCCL-registered" flag in PTE (prevents eviction)
   -> Record pin count (multiple regMr calls for overlapping regions)
  |
  v
4. If RDMA transport:
   -> Call ibv_reg_mr() on the physical VRAM address
   -> Store IB MR handle in the mhandle struct
  |
  v
5. Return mhandle (contains: page list, pin count, IB MR handle if RDMA)

deregMr(comm, mhandle)
  |
  v
1. If RDMA: ibv_dereg_mr()
2. Decrement pin count on each page
3. If pin count reaches 0: clear "NCCL-registered" flag in PTE
4. Pages are now eligible for eviction by R10 tier manager
```

### 3.4 Scatter-Gather for Grouped Receives (R28 Integration)

NCCL's `irecv` supports grouped receives: `n` buffers received in one call. With ConnectX-5 supporting up to 30 SGEs per Work Request (R28 finding), we can post a single RDMA recv with multiple scatter entries:

```
irecv(recvComm, n=4, data=[buf0, buf1, buf2, buf3], sizes=[...], ...)
  |
  v
If RDMA transport and all buffers are registered:
  -> Build scatter-gather list: [{buf0, size0}, {buf1, size1}, ...]
  -> Post single ibv_post_recv with n SGEs (n <= 30, well within ConnectX-5 limit)
  -> One CQE completes all n buffers

If TCP transport:
  -> Post n individual recv operations
  -> Track completion of all n in the request object
```

This avoids posting n separate recv operations and reduces CQ polling overhead.

### 3.5 Connection Lifecycle Diagram

```
Rank 0 (Sender)                     Rank 1 (Receiver)
     |                                    |
     |                                    | listen(dev, &handle, &listenComm)
     |                                    |   -> binds transport endpoint
     |                                    |   -> serializes endpoint to handle
     |                                    |
     |  <--- NCCL bootstrap: handle --->  |
     |                                    |
     | connect(dev, handle,               |
     |         &sendComm, &devComm)       |
     |   -> initiates async connect       |
     |   -> returns sendComm=NULL         |
     |      (pending)                     |
     |                                    | accept(listenComm, &recvComm, &devComm)
     |                                    |   -> returns recvComm=NULL (pending)
     |                                    |
     | [NCCL retries connect...]          | [NCCL retries accept...]
     |                                    |
     | connect() -> sendComm != NULL      | accept() -> recvComm != NULL
     |   (connected!)                     |   (connected!)
     |                                    |
     | regMr(sendComm, buf, ...)          | regMr(recvComm, buf, ...)
     |                                    |
     | isend(sendComm, data, size, ...)   | irecv(recvComm, n, data, ...)
     | test(request, &done, &sizes)       | test(request, &done, &sizes)
     |   [polls until done=1]             |   [polls until done=1]
     |                                    |
     |                                    | iflush(...) [if RDMA]
     |                                    | test(flush_req, &done, ...)
     |                                    |
     | closeSend(sendComm)                | closeRecv(recvComm)
     |                                    | closeListen(listenComm)
```

---

## 4. Gradient Compression Integration

R14 establishes that gradient-specific compression (Top-K sparsification, PowerSGD, quantization) belongs in R20, not R14. R14 provides the compression infrastructure (nvCOMP, LZ4, Zstd); R20 applies domain-specific intelligence.

### 4.1 Where in the Plugin Pipeline

Compression happens inside `isend` / before the transport call, and decompression happens inside `irecv` / after the transport completes:

```
NCCL calls isend(data, size)
  |
  v
1. Detect buffer type (gradient vs weights vs activations)
2. If gradient + compression enabled:
   a. Apply gradient compression (Top-K or PowerSGD)
   b. Compressed data goes to transport
   c. Record original_size and compressed_size in request
3. If not gradient or compression disabled:
   a. Optionally apply R14 general-purpose compression (LZ4/nvCOMP)
   b. Or send raw
  |
  v
Transport::send_async(compressed_data, compressed_size)


Transport delivers data to receiver
  |
  v
irecv completes -> test() polls transport completion
  |
  v
1. Check compression header
2. If gradient-compressed:
   a. Decompress / reconstruct (Top-K: scatter sparse values; PowerSGD: reconstruct from low-rank)
   b. Write decompressed gradients to NCCL's output buffer
3. Report original_size in test() output
```

### 4.2 Detecting "This is a Gradient"

The plugin cannot introspect buffer contents automatically. Detection strategies:

1. **Environment variable hint:** `OUTERLINK_NCCL_COMPRESS_GRADIENTS=1` enables gradient compression for all AllReduce operations. Since AllReduce is used almost exclusively for gradient aggregation in training, this is a safe heuristic.

2. **Buffer size heuristic:** Gradient buffers in training are typically large (tens of MB to GB) and repeat with the same size every iteration. The plugin can track per-comm buffer sizes: if the same size appears repeatedly on the same comm, it is likely a gradient buffer.

3. **NCCL tag inspection:** NCCL assigns different tags to different collective operations. While tags are not documented as stable, monitoring tag patterns can distinguish AllReduce (gradient) traffic from AllGather (parameter) traffic.

4. **Explicit API (future):** A future OuterLink-specific API could let the application mark buffers as "gradient" at allocation time. The VRAM manager (R10) would propagate this metadata to the NCCL plugin.

**Recommended approach for Phase 1:** Use the environment variable hint (#1). Simple, explicit, no heuristic errors. Users opt in with `OUTERLINK_NCCL_COMPRESS_GRADIENTS=1`.

### 4.3 Implementable Compression Techniques at the Transport Level

| Technique | Where | Implementable in Plugin? | Bandwidth Savings | Accuracy Impact |
|-----------|-------|--------------------------|-------------------|-----------------|
| **Top-K Sparsification** | isend path | YES -- select top K% of values, send indices + values | 10-100x for K=0.1-1% | Requires error feedback (memory across iterations) |
| **PowerSGD** | isend/irecv path | PARTIALLY -- requires two AllReduce passes per gradient (orthogonalization + projection) | 10-100x | Minimal with warmup period |
| **FP32-to-FP16 quantization** | isend path | YES -- trivial conversion | 2x | Small for mixed-precision training |
| **FP32-to-INT8 quantization** | isend path | YES -- scale + quantize | 4x | Needs per-tensor scaling |
| **1-bit SGD (signSGD)** | isend path | YES -- send only sign bits | 32x | Requires error feedback |
| **nvCOMP general lossless** | isend path | YES -- via R14 infrastructure | 1.5-3x for gradients | Zero (lossless) |

**Recommended for initial implementation:**
1. **Lossless nvCOMP** (via R14) -- zero accuracy risk, 1.5-3x savings on structured gradient data.
2. **FP16 quantization** -- 2x savings, well-understood accuracy tradeoff, trivially reversible.
3. **Top-K sparsification** deferred to later phase (requires per-comm state for error feedback buffers).

### 4.4 Compression State Per Comm

For gradient compression that requires iteration state (Top-K error feedback, PowerSGD low-rank factors), the plugin maintains per-comm compression context:

```rust
struct GradientCompressionCtx {
    enabled: bool,
    technique: GradCompressionType,     // Lossless, FP16, TopK, PowerSGD
    error_feedback: Option<Vec<f32>>,   // For Top-K: accumulated errors from previous iterations
    iteration_count: u64,               // For warmup periods
    buffer_size: usize,                 // Expected gradient size (detected from first isend)
    compress_buf: Vec<u8>,              // Scratch buffer for compressed output
}
```

This context is stored in `OuterLinkSendComm` and lives for the lifetime of the NCCL communicator.

### 4.5 Interaction with R12 (Dedup) for Broadcast

R12 (Memory Dedup) identifies duplicate content across nodes (e.g., model weights after initialization). When NCCL broadcasts weights that the receiver already has (detected by R12's content hash), the plugin can short-circuit:

```
NCCL calls isend(weight_buffer, size) for broadcast
  |
  v
Plugin checks: does receiver already have this content?
  -> Query R12's dedup index with content hash
  -> If match: send only a 32-byte "already-present" token
  -> Receiver acknowledges, NCCL continues
  -> Saves entire transfer for model weight distribution
```

This optimization is specific to the broadcast collective during model initialization / checkpoint loading. It requires coordination between the NCCL plugin and R12's dedup index.

---

## 5. Multi-Transport Routing

### 5.1 Device Enumeration

The plugin enumerates available transports at `init()` time and reports each as a separate NCCL device:

```rust
struct TransportDevice {
    index: usize,
    transport_type: TransportType,
    pci_path: String,           // Real PCI path for topology detection
    speed_mbps: u32,            // Reported to NCCL
    latency_us: f32,            // Reported to NCCL
    ptr_support: u32,           // NCCL_PTR_HOST, optionally NCCL_PTR_CUDA
    node_guid: u64,             // Unique per transport endpoint
}

// Example device list on Pedro's setup:
// Device 0: ConnectX-5 (100Gbps RDMA)  - pci: /sys/bus/pci/devices/0000:41:00.0
// Device 1: USB4 port   (80Gbps)       - pci: /sys/bus/pci/devices/0000:0d:00.0
// Device 2: TCP fallback (25Gbps)       - pci: (synthetic, host bridge)
```

### 5.2 Topology Reporting for NCCL's Algorithm Selector

NCCL uses PCI paths to build an internal topology graph. Accurate reporting is critical:

| Transport | `pciPath` | `speed` (Mbps) | `latency` (us) | `ptrSupport` |
|-----------|-----------|----------------|----------------|--------------|
| ConnectX-5 | Actual NIC PCI path from `ibv_get_device_list()` | 100000 | 2.0 | HOST (Phase 1), HOST+CUDA (Phase 2+) |
| USB4 | USB4 controller PCI path from sysfs | 40000 | 5.0 | HOST |
| TCP | Synthetic path: root PCI bridge | 25000 | 100.0 | HOST |
| OpenDMA | Target GPU's PCI path (BAR1 DMA target) | 100000 | 1.5 | HOST+CUDA |

**R23 (Heterogeneous GPU) integration:** When the cluster has GPUs with different speeds (e.g., RTX 3090 vs RTX 4070), the plugin reports asymmetric bandwidth per device based on the GPU pair's capability. The `speed` field reflects the slower GPU's maximum transfer rate for that transport. NCCL's algorithm selector uses these per-device speeds to build asymmetric rings/trees.

**R17 (Topology) integration:** hwloc-based discovery (from R17) feeds PCI topology data to the plugin. The plugin calls `hwloc_get_pcidev_by_busid()` to resolve PCI paths for ConnectX NICs, and uses sysfs for USB4 controllers. This ensures NCCL sees accurate NUMA-to-NIC-to-GPU affinity.

### 5.3 Transport Selection Logic

When NCCL calls `connect(dev, handle)`, the `dev` parameter specifies which device (transport) to use. NCCL has already made the selection based on topology. However, during the connection handshake, both sides negotiate the best available transport:

```
1. NCCL calls connect(dev=0, handle)  [selected ConnectX]
2. Plugin reads handle's capability_flags
3. Both sides support RDMA? -> Use RDMA
4. Only one side has RDMA? -> Fall back to TCP (device 2)
5. Both sides have USB4? -> Use USB4
6. Fallback chain: RDMA -> USB4 -> TCP
```

This negotiation is transparent to NCCL. The plugin reports the connection as using the requested device, but internally may fall back. This ensures NCCL's channel assignment still works correctly.

### 5.4 R26 (PTP Clock) Integration for Synchronized Launches

R26 provides PTP-synchronized clocks across nodes. For NCCL collectives that benefit from coordinated timing (e.g., AllReduce where all ranks should start sending simultaneously):

- The plugin can use PTP timestamps to coordinate send/recv timing across ranks.
- During `isend`, if the collective is a synchronized operation, the plugin may delay transmission until a PTP-coordinated launch time.
- This reduces idle time at the start of each collective step, improving tail latency.

This is an advanced optimization for Phase E/F, not Phase A/B.

### 5.5 R29 (Multicast) Integration Path

R29 identifies UD-based hardware multicast for the broadcast collective. This maps to NCCL's future CollNet API:

- Phase 1: Not implemented. NCCL decomposes broadcast into point-to-point sends via ncclNet.
- Future: Implement `ncclCollNet_vX` with OuterLink's multicast transport. A single `ibv_post_send` on a UD multicast group delivers to all receivers simultaneously.
- The ncclNet plugin and ncclCollNet plugin can coexist in the same .so library.

---

## 6. Concrete Test Plan

### 6.1 Phase A: Skeleton Plugin Verification

```bash
# Build the plugin
cargo build --release -p outerlink-nccl-plugin

# Verify symbol export
nm -D target/release/libnccl-net-outerlink.so | grep ncclNet
# Expected: D ncclNet_v8, D ncclNet_v9, D ncclNet_v10, D ncclNet_v11

# Verify NCCL discovers the plugin (single node, no actual networking)
NCCL_NET_PLUGIN=outerlink \
NCCL_DEBUG=INFO \
LD_LIBRARY_PATH=target/release:$LD_LIBRARY_PATH \
python3 -c "
import torch
import torch.distributed as dist
# Just initialize -- we expect a log line showing plugin discovery
# Will fail at actual communication, but that's OK for Phase A
print('Plugin discovery test')
"
# Look for: "OuterLink plugin loaded" or "Using network outerlink" in NCCL_DEBUG output
```

### 6.2 Phase B: nccl-tests Over TCP

```bash
# Build nccl-tests
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1 NCCL_HOME=/usr/local/nccl

# Environment for OuterLink plugin
export NCCL_NET_PLUGIN=outerlink
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET
export LD_LIBRARY_PATH=/path/to/outerlink/target/release:$LD_LIBRARY_PATH
export OUTERLINK_SERVER=192.168.1.100:9876  # OuterLink server address

# Basic AllReduce (2 nodes, 1 GPU each)
mpirun -np 2 -H node0:1,node1:1 \
  ./build/all_reduce_perf -b 8 -e 1G -f 2 -g 1
# Expected: Bandwidth > 1 Gbps (TCP), no errors, no hangs

# AllGather
mpirun -np 2 -H node0:1,node1:1 \
  ./build/all_gather_perf -b 8 -e 1G -f 2 -g 1

# Broadcast
mpirun -np 2 -H node0:1,node1:1 \
  ./build/broadcast_perf -b 8 -e 1G -f 2 -g 1

# SendRecv (point-to-point, raw transport test)
mpirun -np 2 -H node0:1,node1:1 \
  ./build/sendrecv_perf -b 8 -e 1G -f 2 -g 1

# Stress test: multiple GPUs per node (if available)
mpirun -np 4 -H node0:2,node1:2 \
  ./build/all_reduce_perf -b 1M -e 1G -f 2 -g 1

# Correctness check (nccl-tests has built-in verification)
mpirun -np 2 -H node0:1,node1:1 \
  ./build/all_reduce_perf -b 8 -e 256M -f 2 -g 1 -c 1
# -c 1 enables correctness checking
```

### 6.3 PyTorch Distributed Training Test

```python
# test_pytorch_distributed.py
# Run with: torchrun --nproc_per_node=1 --nnodes=2 --node_rank=$RANK \
#           --master_addr=$MASTER --master_port=29500 test_pytorch_distributed.py

import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    # Test 1: Basic AllReduce
    tensor = torch.ones(1024, 1024, device=device) * (rank + 1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = sum(range(1, world_size + 1))
    assert torch.allclose(tensor, torch.full_like(tensor, expected)), \
        f"AllReduce failed: got {tensor[0,0].item()}, expected {expected}"
    print(f"[Rank {rank}] AllReduce PASSED")

    # Test 2: Broadcast
    if rank == 0:
        tensor = torch.randn(512, 512, device=device)
    else:
        tensor = torch.zeros(512, 512, device=device)
    dist.broadcast(tensor, src=0)
    # All ranks should have same tensor
    print(f"[Rank {rank}] Broadcast PASSED (sum={tensor.sum().item():.4f})")

    # Test 3: DDP training loop (smoke test)
    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
    ).to(device)
    ddp_model = DDP(model, device_ids=[device])
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

    for step in range(10):
        input_data = torch.randn(32, 1024, device=device)
        output = ddp_model(input_data)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if rank == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")

    print(f"[Rank {rank}] DDP training PASSED")

    # Test 4: Large AllReduce (gradient-sized)
    large_tensor = torch.randn(50_000_000, device=device)  # ~200MB
    dist.all_reduce(large_tensor, op=dist.ReduceOp.SUM)
    print(f"[Rank {rank}] Large AllReduce PASSED")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

### 6.4 DeepSpeed ZeRO Config

```json
{
    "train_batch_size": 16,
    "gradient_accumulation_steps": 1,
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 50000000,
        "reduce_scatter": true,
        "reduce_bucket_size": 50000000,
        "overlap_comm": true
    },
    "communication_data_type": "fp16"
}
```

```bash
# Run DeepSpeed with OuterLink NCCL plugin
NCCL_NET_PLUGIN=outerlink \
NCCL_DEBUG=WARN \
LD_LIBRARY_PATH=/path/to/outerlink/target/release:$LD_LIBRARY_PATH \
deepspeed --num_gpus=1 --num_nodes=2 \
  --hostfile hostfile \
  train_script.py \
  --deepspeed --deepspeed_config ds_config.json
```

### 6.5 Performance Benchmarks

```bash
# Measure raw transport bandwidth through NCCL plugin
# Compare: OuterLink TCP vs NCCL built-in TCP socket

# Baseline: NCCL built-in TCP
NCCL_NET=Socket \
NCCL_SOCKET_IFNAME=enp5s0 \
mpirun -np 2 -H node0:1,node1:1 \
  ./build/all_reduce_perf -b 1M -e 4G -f 2 -g 1 -n 50 \
  | tee baseline_tcp.log

# OuterLink TCP
NCCL_NET_PLUGIN=outerlink \
OUTERLINK_TRANSPORT=tcp \
mpirun -np 2 -H node0:1,node1:1 \
  ./build/all_reduce_perf -b 1M -e 4G -f 2 -g 1 -n 50 \
  | tee outerlink_tcp.log

# OuterLink RDMA (Phase D)
NCCL_NET_PLUGIN=outerlink \
OUTERLINK_TRANSPORT=rdma \
mpirun -np 2 -H node0:1,node1:1 \
  ./build/all_reduce_perf -b 1M -e 4G -f 2 -g 1 -n 50 \
  | tee outerlink_rdma.log

# Compare: diff baseline_tcp.log outerlink_tcp.log
# Key metrics: bus bandwidth (GB/s), algorithm bandwidth (GB/s)
```

### 6.6 Stability / Stress Tests

```bash
# Long-running stability test: 1000 iterations, various message sizes
mpirun -np 2 -H node0:1,node1:1 \
  ./build/all_reduce_perf -b 8 -e 1G -f 2 -g 1 -n 1000 -w 50

# Connection teardown/re-establish test
for i in $(seq 1 20); do
  mpirun -np 2 -H node0:1,node1:1 \
    ./build/all_reduce_perf -b 1M -e 1M -g 1 -n 10
  echo "Iteration $i completed"
done

# Multi-collective interleaved test
mpirun -np 2 -H node0:1,node1:1 \
  ./build/all_reduce_perf -b 1M -e 256M -f 2 -g 1 -n 100 &
mpirun -np 2 -H node0:1,node1:1 \
  ./build/all_gather_perf -b 1M -e 256M -f 2 -g 1 -n 100 &
wait
```

---

## 7. Updated Decision Status

All 5 decisions from v1 are now CONFIRMED based on cross-topic analysis:

| Decision | Choice | Confirmed By |
|----------|--------|-------------|
| D1: API Version | v8 primary, v9-v11 shims | Research doc 01, aws-ofi-nccl pattern |
| D2: Language | Rust with C FFI (cdylib) | Direct access to OuterLink transport crates |
| D3: Headers | Copy into tree, manual Rust structs | Section 2 above defines exact structs |
| D4: Device Model | One device per transport | R17 (hwloc discovery validates), R23 (asymmetric bandwidth) |
| D5: GPU Pointer Timeline | NCCL_PTR_HOST first, CUDA later | R10 (VRAM manager handles pinning when ready) |

---

## 8. Updated Risk Matrix

| Risk | Severity | Likelihood | Mitigation | Cross-Topic Reference |
|------|----------|-----------|------------|----------------------|
| NCCL API changes | HIGH | MEDIUM | Multi-version shims (v8-v11) | -- |
| regMr + VRAM manager coordination | HIGH | HIGH | Detailed mapping in Section 3.3; R10 64KB pages align with NCCL buffer sizes | R10 |
| Performance below expectations | HIGH | MEDIUM | Accurate property reporting; request pools; R14 compression for bandwidth amplification | R14 |
| Non-blocking violations | MEDIUM | MEDIUM | State machine per comm; async transport throughout | -- |
| Concurrency bugs | MEDIUM | MEDIUM | Per-comm isolation; no shared mutable state; ffi_guard! macro | -- |
| Handle overflow | LOW | LOW | 64-byte handle fits well within 128-byte NCCL_NET_HANDLE_MAXSIZE | Resolved (Q3) |
| Rust FFI panics | LOW | LOW | catch_unwind at every boundary; panic=abort in release | Section 2.4 |
| Gradient compression accuracy | MEDIUM | LOW | Start with lossless (nvCOMP), FP16 only when user opts in | R14 |
| Multi-transport confusion | MEDIUM | LOW | NCCL handles device selection via topology; verified with NCCL_DEBUG=GRAPH | R17, R23 |

---

## 9. Updated Implementation Phases

### Phase A: Skeleton Plugin (1 week)

- Create `crates/outerlink-nccl-plugin/` crate (`cdylib`)
- Implement all `#[repr(C)]` types from Section 2.1
- Implement all 19 FFI functions as stubs (return `ncclSuccess` with dummy data)
- Export `ncclNet_v8` through `ncclNet_v11` symbols
- `ffi_guard!` macro for panic safety
- **Deliverable:** `nm -D` shows symbols; `NCCL_DEBUG=INFO` shows "outerlink" in logs

### Phase B: TCP Transport Backend (2-3 weeks)

- Wire `listen`/`connect`/`accept` to OuterLink TCP transport
- Implement `regMr` for `NCCL_PTR_HOST` (mlock + registration cache)
- Implement `isend`/`irecv`/`test` using OuterLink async transport
- Request pool (64 entries per comm, slab allocator)
- Non-blocking state machine for connection lifecycle
- Report accurate TCP properties (speed=25000, latency=100.0)
- **Deliverable:** Pass `nccl-tests all_reduce_perf -c 1` (correctness mode) over TCP

### Phase C: Multi-Version Shims (1 week)

- v9 shim: int-to-size_t wrappers for isend/irecv sizes; stub `makeVDevice`
- v10 shim: accept `NcclNetCommConfig_v10` in connect (ignore for now)
- v11 shim: per-communicator `init`/`finalize` with ctx parameter
- Export all version symbols
- **Deliverable:** Same binary passes nccl-tests with NCCL 2.19, 2.22, 2.25, 2.28

### Phase D: RDMA Transport Backend (2-3 weeks)

- Add ConnectX-5 RDMA path in transport adapter
- `regMr` with `ibv_reg_mr` for host memory, `nv_peer_mem` for GPU memory
- `iflush` implementation: `ibv_post_send` with zero-byte RDMA read (fence)
- Scatter-gather irecv using ConnectX-5 SGE support (R28)
- Report `NCCL_PTR_CUDA` when nv_peer_mem is available
- Properties: speed=100000, latency=2.0
- **Deliverable:** Pass nccl-tests over RDMA; bandwidth > 80 Gbps for large AllReduce

### Phase E: Multi-Transport + R10 Integration (2-3 weeks)

- Report multiple devices (ConnectX + USB4 + TCP) with correct PCI paths
- Integrate with R10 VRAM manager for `regMr` with `NCCL_PTR_CUDA`
- Page pinning/unpinning protocol (Section 3.3)
- R23 asymmetric bandwidth reporting for heterogeneous GPUs
- Verify NCCL channel distribution with `NCCL_DEBUG=GRAPH`
- **Deliverable:** nccl-tests shows combined multi-transport bandwidth; R10 pages pinned correctly

### Phase F: Compression + OpenDMA (follows Phase 5)

- Integrate R14 compression: lossless nvCOMP for general buffers
- Gradient compression: FP16 quantization in isend path (opt-in)
- OpenDMA path when PCIe BAR1 transport is ready
- R12 dedup-aware broadcast (skip already-present model weights)
- R29 CollNet path for hardware multicast broadcast
- **Deliverable:** nccl-tests with compression showing 2x+ effective bandwidth; OpenDMA showing < 5us latency

---

## 10. Updated Crate Structure

```
crates/outerlink-nccl-plugin/
  Cargo.toml                  # crate-type = ["cdylib"], lib name = "nccl_net_outerlink"
  nccl-headers/               # Copied NCCL header files (reference only, not used in build)
    net_v8.h
    net_v9.h
    net_v10.h
    net_v11.h
  src/
    lib.rs                    # FFI exports: #[no_mangle] statics, ffi_guard! macro, all 19 extern "C" fns
    ffi_types.rs              # #[repr(C)] structs: NcclNet_v8, NcclNetProperties_v8, NcclResult, etc.
    plugin.rs                 # Global state: logger, transport manager init, env var reading
    transport_adapter.rs      # Maps NCCL ops to OuterLink transport calls (Section 3.1)
    device_manager.rs         # Multi-transport device enumeration, property reporting (Section 5.1)
    memory_registry.rs        # regMr/deregMr: LRU cache, R10 VRAM manager integration (Section 3.3)
    connection.rs             # listen/connect/accept: non-blocking state machine (Section 3.5)
    async_ops.rs              # isend/irecv/iflush/test: request pool, SGE grouping (Sections 3.4, 4.1)
    version_shims.rs          # v9/v10/v11 shim layers: type conversions, added parameters
    compression.rs            # R14 integration + gradient compression (Section 4)
    handle.rs                 # Handle serialization/deserialization (Q3, 64-byte format)
```

---

## Related Documents

- [preplan.md](./preplan.md) -- Original v1 preplan (superseded by this document)
- [research/01-nccl-net-plugin-api.md](./research/01-nccl-net-plugin-api.md) -- Exact API surface
- [research/02-existing-nccl-plugins.md](./research/02-existing-nccl-plugins.md) -- Existing plugin survey
- [research/03-nccl-topology-and-collectives.md](./research/03-nccl-topology-and-collectives.md) -- NCCL internals
- [../R10-memory-tiering/preplan.md](../R10-memory-tiering/preplan.md) -- VRAM manager, 64KB pages, page pinning
- [../R14-transport-compression/preplan.md](../R14-transport-compression/preplan.md) -- Compression infrastructure, nvCOMP
- R17 Topology-Aware Scheduling -- hwloc discovery, PCI path resolution
- R23 Heterogeneous GPUs -- Asymmetric bandwidth reporting
- R25 Kernel Splitting -- NCCL collectives as graph nodes
- R26 PTP Clock -- Synchronized collective launches
- R28 Scatter-Gather DMA -- ConnectX-5 SGE support for grouped receives
- R29 RDMA Multicast -- Hardware broadcast for future CollNet
- R12 Memory Dedup -- Dedup-aware broadcast optimization
