# P4: Project Skeleton - Rust Workspace and Crate Structure

**Created:** 2026-03-19
**Last Updated:** 2026-03-19
**Status:** Plan
**Priority:** HIGH - Do after P1/P2 complete

## Goal

Create the complete Rust workspace skeleton with all crate structures, trait definitions, protocol message types, handle translation types, error types, FFI layer design, the C-based LD_PRELOAD interception library, and build system -- so that P5 (PoC) can immediately start filling in implementations without any architectural decisions remaining.

## Milestone

- `cargo build --all` succeeds on a Linux machine with CUDA installed
- `cargo test --all` passes (skeleton tests)
- `cargo clippy --all-targets -- -D warnings` clean
- The C interception library (`liboutterlink_interpose.so`) compiles via `cc` crate
- Transport trait is defined with a TCP implementation skeleton
- Protocol messages are defined for all Phase 1 operations (device query, memory, module, kernel)
- Handle translation types are defined for all CUDA handle types
- Error types map CUDA errors to/from network errors
- Server binary starts and listens on a configurable port
- CLI binary prints help text
- Config file format is documented and parseable

## Prerequisites

- [x] License decided: Apache 2.0 (ADR-001)
- [x] Pre-planning complete
- [x] CUDA interception strategy decided (R3): Driver API + LD_PRELOAD + cuGetProcAddress
- [x] Transport strategy decided (R4): TCP Phase 1, UCX Phase 2
- [ ] P1 complete (GitHub repo exists)
- [ ] P2 complete (dev environment ready)

---

## Decision: LD_PRELOAD Library in C (Not Rust)

The LD_PRELOAD interception `.so` MUST be written in C. Reasons:

1. **`cdylib` LD_PRELOAD in Rust is unreliable.** Documented issues exist where `cdylib` builds fail to intercept symbols that `dylib` handles correctly (rust-lang/rust#76211). The Rust allocator (jemalloc or system) can cause infinite recursion when `dlsym` calls `malloc` during initialization.

2. **HAMi-core's proven architecture is pure C.** The `dlsym` override, `cuGetProcAddress` double-wrapper, and `cuda_library_entry[]` function table pattern all rely on C-level control over symbol visibility, `__attribute__((constructor))`, `RTLD_NEXT`, and `dlvsym`. Reimplementing this in Rust adds risk for zero benefit.

3. **The C layer is THIN.** It only does: intercept symbol lookup, serialize arguments into a byte buffer, call into the Rust client library (via C FFI), deserialize the response, return the result. All logic lives in Rust.

4. **Architecture:** `C interception .so` --> calls --> `Rust client library (cdylib with C API)` --> sends over network --> `Rust server`.

## Decision: Server-side FFI via `libloading` (Not `cudarc`)

The server needs to call the real CUDA Driver API on the GPU node. Options:

| Option | Pros | Cons |
|--------|------|------|
| **`cudarc` 0.19.x** | Safe wrappers, well-maintained | Wraps only a subset of 222+ functions. We need raw Driver API calls for arbitrary forwarding. Its abstractions (CudaContext, CudaSlice) fight our proxy pattern. |
| **`bindgen`** | Generates raw FFI from `cuda.h` | Requires CUDA headers at build time. CI machines may not have CUDA. |
| **`libloading`** | Load `libcuda.so` at runtime, resolve any symbol | No build-time CUDA dependency. Works for arbitrary function forwarding. We control the call. |

**Decision: `libloading`** for the server's CUDA calls. We load `libcuda.so` at runtime, resolve function pointers by name, and call them with the exact arguments received from the client. This mirrors the proxy pattern perfectly: the server doesn't need safe Rust wrappers around CUDA -- it needs to forward raw calls.

We define our own type aliases for CUDA types (`CUresult`, `CUdevice`, `CUdeviceptr`, etc.) in `outterlink-common` so both client and server share the same representations without depending on CUDA headers.

## Decision: Serialization via `bincode` 2.x

For the wire protocol between client and server:

| Option | Serialize | Deserialize | Size | Ecosystem |
|--------|-----------|-------------|------|-----------|
| **bincode 2.x** | Fastest | Fast | Smallest | Mature, serde optional, `#[derive(Encode, Decode)]` |
| **rkyv** | Fast | Zero-copy (fastest) | Larger | Complex, unsafe for mutable access |
| **protobuf** | Slow | Slow | Moderate | Cross-language (unnecessary for Rust-to-Rust) |

**Decision: `bincode` 2.x** with its native `Encode`/`Decode` derives (no serde dependency). Fastest serialization, smallest wire size, and the simplest integration. We are Rust-to-Rust so cross-language compatibility is irrelevant. For bulk data transfers (memcpy payloads), we skip serialization entirely and send raw bytes after a fixed header.

---

## 1. Cargo Workspace Layout

### Root `Cargo.toml`

```toml
[workspace]
resolver = "2"
members = [
    "crates/outterlink-common",
    "crates/outterlink-server",
    "crates/outterlink-client",
    "crates/outterlink-cli",
]

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "Apache-2.0"
repository = "https://github.com/<owner>/outterlink"
rust-version = "1.75"

[workspace.dependencies]
# Serialization
bincode = "2"

# Async runtime
tokio = { version = "1", features = ["full"] }

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Error handling
thiserror = "2"
anyhow = "1"

# Configuration
toml = "0.8"
serde = { version = "1", features = ["derive"] }

# CLI
clap = { version = "4", features = ["derive"] }

# Dynamic loading (server-side CUDA)
libloading = "0.8"

# Byte manipulation
bytes = "1"

# Unique IDs
uuid = { version = "1", features = ["v4"] }

# Async channels
flume = "0.11"

# Hash maps
dashmap = "6"

[profile.release]
lto = "thin"
codegen-units = 1
strip = "symbols"
```

### `crates/outterlink-common/Cargo.toml`

```toml
[package]
name = "outterlink-common"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
bincode = { workspace = true }
tokio = { workspace = true }
thiserror = { workspace = true }
tracing = { workspace = true }
bytes = { workspace = true }
uuid = { workspace = true }
serde = { workspace = true }
toml = { workspace = true }
dashmap = { workspace = true }

[dev-dependencies]
tokio = { workspace = true, features = ["test-util"] }
```

### `crates/outterlink-server/Cargo.toml`

```toml
[package]
name = "outterlink-server"
version.workspace = true
edition.workspace = true
license.workspace = true

[[bin]]
name = "outterlink-server"
path = "src/main.rs"

[dependencies]
outterlink-common = { path = "../outterlink-common" }
tokio = { workspace = true }
tracing = { workspace = true }
tracing-subscriber = { workspace = true }
anyhow = { workspace = true }
libloading = { workspace = true }
bincode = { workspace = true }
bytes = { workspace = true }
clap = { workspace = true }
serde = { workspace = true }
toml = { workspace = true }

[dev-dependencies]
tokio = { workspace = true, features = ["test-util"] }
```

### `crates/outterlink-client/Cargo.toml`

```toml
[package]
name = "outterlink-client"
version.workspace = true
edition.workspace = true
license.workspace = true

[lib]
# Produces both:
# - A Rust library for outterlink-cli to depend on
# - A C-compatible shared library that the C interposition layer calls into
crate-type = ["lib", "cdylib"]

[dependencies]
outterlink-common = { path = "../outterlink-common" }
tokio = { workspace = true }
tracing = { workspace = true }
tracing-subscriber = { workspace = true }
bincode = { workspace = true }
bytes = { workspace = true }
dashmap = { workspace = true }

[build-dependencies]
cc = "1"

[dev-dependencies]
tokio = { workspace = true, features = ["test-util"] }
```

### `crates/outterlink-cli/Cargo.toml`

```toml
[package]
name = "outterlink-cli"
version.workspace = true
edition.workspace = true
license.workspace = true

[[bin]]
name = "outterlink"
path = "src/main.rs"

[dependencies]
outterlink-common = { path = "../outterlink-common" }
outterlink-client = { path = "../outterlink-client" }
tokio = { workspace = true }
tracing = { workspace = true }
tracing-subscriber = { workspace = true }
clap = { workspace = true }
anyhow = { workspace = true }
serde = { workspace = true }
toml = { workspace = true }

[dev-dependencies]
tokio = { workspace = true, features = ["test-util"] }
```

---

## 2. Directory Structure

```
outterlink/
├── Cargo.toml                          # Workspace root
├── crates/
│   ├── outterlink-common/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # Re-exports all modules
│   │       ├── cuda_types.rs           # CUresult, CUdevice, CUdeviceptr, etc.
│   │       ├── handle.rs               # HandleMap, handle translation types
│   │       ├── error.rs                # OutterLinkError, CUDA error mapping
│   │       ├── protocol.rs             # Request/Response message enums
│   │       ├── transport.rs            # Transport trait definition
│   │       └── config.rs              # Configuration types, TOML parsing
│   │
│   ├── outterlink-server/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── main.rs                 # Entry point, arg parsing, server startup
│   │       ├── server.rs               # Accept loop, connection handling
│   │       ├── dispatcher.rs           # Route incoming requests to GPU executor
│   │       ├── gpu_executor.rs         # Load libcuda.so, execute real CUDA calls
│   │       └── session.rs              # Per-client session state (contexts, allocations)
│   │
│   ├── outterlink-client/
│   │   ├── Cargo.toml
│   │   ├── build.rs                    # Compile C interposition library
│   │   ├── interpose/                  # C source for LD_PRELOAD library
│   │   │   ├── interpose.c             # dlsym override, cuGetProcAddress hook
│   │   │   ├── interpose.h             # Declarations shared with Rust FFI
│   │   │   └── cuda_subset.h           # Minimal CUDA type definitions (no NVIDIA headers needed)
│   │   └── src/
│   │       ├── lib.rs                  # Rust client library + C FFI exports
│   │       ├── ffi.rs                  # #[no_mangle] extern "C" functions called by interpose.c
│   │       ├── connection.rs           # Manages TCP connection to server
│   │       ├── handle_map.rs           # Local handle <-> remote handle translation
│   │       └── call_forwarder.rs       # Serialize CUDA call, send, receive response
│   │
│   └── outterlink-cli/
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs                 # CLI entry point
│           └── commands/
│               ├── mod.rs
│               ├── status.rs           # Show node/GPU status
│               ├── list.rs             # List available GPUs across nodes
│               └── benchmark.rs        # Run basic latency/bandwidth test
│
├── opendma/                            # Future: kernel module (C)
│   ├── module/
│   └── patches/
│
└── benchmarks/                         # Future: benchmark suite
```

---

## 3. CUDA Type Definitions (`cuda_types.rs`)

These are our own definitions, independent of NVIDIA headers. They match CUDA's ABI exactly.

```rust
//! CUDA type definitions matching the CUDA Driver API ABI.
//!
//! These are defined independently of NVIDIA headers so that
//! outterlink-common compiles without CUDA installed.

/// CUDA error codes. Matches CUresult enum values from cuda.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, bincode::Encode, bincode::Decode)]
#[repr(u32)]
pub enum CuResult {
    Success = 0,
    ErrorInvalidValue = 1,
    ErrorOutOfMemory = 2,
    ErrorNotInitialized = 3,
    ErrorDeinitialized = 4,
    ErrorProfilerDisabled = 5,
    ErrorProfilerNotInitialized = 6,
    ErrorProfilerAlreadyStarted = 7,
    ErrorProfilerAlreadyStopped = 8,
    ErrorNoDevice = 100,
    ErrorInvalidDevice = 101,
    ErrorDeviceNotLicensed = 102,
    ErrorInvalidImage = 200,
    ErrorInvalidContext = 201,
    ErrorMapFailed = 205,
    ErrorUnmapFailed = 206,
    ErrorArrayIsMapped = 207,
    ErrorAlreadyMapped = 208,
    ErrorNoBinaryForGpu = 209,
    ErrorAlreadyAcquired = 210,
    ErrorNotMapped = 211,
    ErrorNotMappedAsArray = 212,
    ErrorNotMappedAsPointer = 213,
    ErrorEccUncorrectable = 214,
    ErrorUnsupportedLimit = 215,
    ErrorInvalidSource = 300,
    ErrorFileNotFound = 301,
    ErrorSharedObjectSymbolNotFound = 302,
    ErrorSharedObjectInitFailed = 303,
    ErrorOperatingSystem = 304,
    ErrorInvalidHandle = 400,
    ErrorIllegalAddress = 700,
    ErrorLaunchOutOfResources = 701,
    ErrorLaunchTimeout = 702,
    ErrorLaunchFailed = 719,
    ErrorSystemDriverMismatch = 803,
    ErrorNotFound = 500,
    ErrorNotReady = 600,
    ErrorUnknown = 999,

    // OutterLink-specific error codes (above CUDA's range)
    ErrorNetworkDisconnected = 10000,
    ErrorNetworkTimeout = 10001,
    ErrorProtocolMismatch = 10002,
    ErrorRemoteGpuLost = 10003,
}

impl CuResult {
    pub fn from_raw(code: u32) -> Self {
        // Safety: if unknown code, map to ErrorUnknown
        // In practice, use a match or TryFrom
        match code {
            0 => Self::Success,
            1 => Self::ErrorInvalidValue,
            2 => Self::ErrorOutOfMemory,
            3 => Self::ErrorNotInitialized,
            100 => Self::ErrorNoDevice,
            101 => Self::ErrorInvalidDevice,
            // ... (full mapping in implementation)
            _ => Self::ErrorUnknown,
        }
    }

    pub fn to_raw(self) -> u32 {
        self as u32
    }

    pub fn is_success(self) -> bool {
        self == Self::Success
    }
}

/// Opaque CUDA device index (maps to CUdevice = int in CUDA).
pub type CuDevice = i32;

/// Device pointer. 64-bit on all platforms.
/// On the client side, this holds a synthetic local pointer.
/// On the server side, this holds the real GPU device pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, bincode::Encode, bincode::Decode)]
pub struct CuDevicePtr(pub u64);

/// Opaque handle types. Each wraps a u64 that is either:
/// - A synthetic handle generated by the client (local side)
/// - A real handle from the CUDA driver (server side)
///
/// The mapping between them is maintained by HandleMap.
macro_rules! define_handle {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, bincode::Encode, bincode::Decode)]
        pub struct $name(pub u64);

        impl $name {
            pub const NULL: Self = Self(0);

            pub fn is_null(self) -> bool {
                self.0 == 0
            }
        }
    };
}

define_handle!(CuContext);
define_handle!(CuModule);
define_handle!(CuFunction);
define_handle!(CuStream);
define_handle!(CuEvent);

/// Device attribute identifiers (matches CUdevice_attribute enum).
/// Only the commonly queried ones are listed; the full set has 100+ entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, bincode::Encode, bincode::Decode)]
#[repr(i32)]
pub enum CuDeviceAttribute {
    MaxThreadsPerBlock = 1,
    MaxBlockDimX = 2,
    MaxBlockDimY = 3,
    MaxBlockDimZ = 4,
    MaxGridDimX = 5,
    MaxGridDimY = 6,
    MaxGridDimZ = 7,
    MaxSharedMemoryPerBlock = 8,
    TotalConstantMemory = 9,
    WarpSize = 10,
    MaxPitch = 11,
    MaxRegistersPerBlock = 12,
    ClockRate = 13,
    TextureAlignment = 14,
    MultiprocessorCount = 16,
    KernelExecTimeout = 17,
    Integrated = 18,
    CanMapHostMemory = 19,
    ComputeMode = 20,
    ConcurrentKernels = 31,
    EccEnabled = 32,
    PciBusId = 33,
    PciDeviceId = 34,
    MemoryClockRate = 36,
    GlobalMemoryBusWidth = 37,
    L2CacheSize = 38,
    MaxThreadsPerMultiprocessor = 39,
    ComputeCapabilityMajor = 75,
    ComputeCapabilityMinor = 76,
    // Pass-through: for attributes we don't enumerate, we use the raw i32
    // and forward to the server.
}

impl CuDeviceAttribute {
    pub fn from_raw(v: i32) -> Self {
        // Use transmute only for known values, otherwise store raw
        // Implementation will use a match with a fallback
        unsafe { std::mem::transmute(v) }
    }
}

/// Context creation flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, bincode::Encode, bincode::Decode)]
pub struct CuContextFlags(pub u32);

/// Stream creation flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, bincode::Encode, bincode::Decode)]
pub struct CuStreamFlags(pub u32);

/// Event creation flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, bincode::Encode, bincode::Decode)]
pub struct CuEventFlags(pub u32);
```

---

## 4. Handle Translation (`handle.rs`)

```rust
//! Bidirectional handle translation between local synthetic handles
//! and remote real handles.
//!
//! The client generates synthetic handles to return to the application.
//! Each synthetic handle maps to a real handle on a specific remote server.
//! The server has no knowledge of synthetic handles -- it only works with
//! real CUDA handles.

use std::sync::atomic::{AtomicU64, Ordering};
use dashmap::DashMap;

use crate::cuda_types::*;

/// Identifies which remote server a handle belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, bincode::Encode, bincode::Decode)]
pub struct NodeId(pub u32);

/// A remote handle: the real CUDA handle value plus which node owns it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, bincode::Encode, bincode::Decode)]
pub struct RemoteHandle {
    pub node: NodeId,
    pub value: u64,
}

/// Thread-safe bidirectional map between synthetic local handles and
/// remote real handles.
///
/// Uses DashMap (concurrent HashMap) for lock-free reads under contention.
/// CUDA applications are typically multi-threaded (one thread per stream).
pub struct HandleMap<H: std::hash::Hash + Eq + Copy> {
    /// Synthetic (local) -> Remote (real)
    local_to_remote: DashMap<H, RemoteHandle>,
    /// Remote (real) -> Synthetic (local)
    remote_to_local: DashMap<(NodeId, u64), H>,
    /// Counter for generating unique synthetic handles
    next_synthetic: AtomicU64,
}

impl<H> HandleMap<H>
where
    H: std::hash::Hash + Eq + Copy + From<u64>,
    u64: From<H>,
{
    pub fn new() -> Self {
        Self {
            local_to_remote: DashMap::new(),
            remote_to_local: DashMap::new(),
            // Start at 0x1000 to avoid confusion with NULL (0) and
            // small integers that CUDA sometimes uses for device indices.
            next_synthetic: AtomicU64::new(0x1000),
        }
    }

    /// Register a new remote handle. Generates a synthetic local handle
    /// and returns it. The application receives the synthetic handle.
    pub fn register(&self, node: NodeId, remote_value: u64) -> H {
        let synthetic_value = self.next_synthetic.fetch_add(1, Ordering::Relaxed);
        let synthetic = H::from(synthetic_value);
        let remote = RemoteHandle {
            node,
            value: remote_value,
        };
        self.local_to_remote.insert(synthetic, remote);
        self.remote_to_local.insert((node, remote_value), synthetic);
        synthetic
    }

    /// Look up the remote handle for a local synthetic handle.
    /// Returns None if the handle was never registered (bug or use-after-free).
    pub fn to_remote(&self, local: H) -> Option<RemoteHandle> {
        self.local_to_remote.get(&local).map(|v| *v)
    }

    /// Look up the local synthetic handle for a remote handle.
    pub fn to_local(&self, node: NodeId, remote_value: u64) -> Option<H> {
        self.remote_to_local.get(&(node, remote_value)).map(|v| *v)
    }

    /// Remove a handle mapping (e.g., after cuMemFree or cuCtxDestroy).
    pub fn remove(&self, local: H) -> Option<RemoteHandle> {
        if let Some((_, remote)) = self.local_to_remote.remove(&local) {
            self.remote_to_local.remove(&(remote.node, remote.value));
            Some(remote)
        } else {
            None
        }
    }

    /// Number of active mappings.
    pub fn len(&self) -> usize {
        self.local_to_remote.len()
    }

    pub fn is_empty(&self) -> bool {
        self.local_to_remote.is_empty()
    }
}

/// Convenience type aliases for each CUDA handle type.
pub type DevicePtrMap = HandleMap<CuDevicePtr>;
pub type ContextMap = HandleMap<CuContext>;
pub type ModuleMap = HandleMap<CuModule>;
pub type FunctionMap = HandleMap<CuFunction>;
pub type StreamMap = HandleMap<CuStream>;
pub type EventMap = HandleMap<CuEvent>;

/// All handle maps for a single client session.
pub struct HandleStore {
    pub device_ptrs: DevicePtrMap,
    pub contexts: ContextMap,
    pub modules: ModuleMap,
    pub functions: FunctionMap,
    pub streams: StreamMap,
    pub events: EventMap,
}

impl HandleStore {
    pub fn new() -> Self {
        Self {
            device_ptrs: HandleMap::new(),
            contexts: HandleMap::new(),
            modules: HandleMap::new(),
            functions: HandleMap::new(),
            streams: HandleMap::new(),
            events: HandleMap::new(),
        }
    }
}

// Implement From<u64> for each handle type to satisfy HandleMap bounds
impl From<u64> for CuDevicePtr {
    fn from(v: u64) -> Self { Self(v) }
}
impl From<CuDevicePtr> for u64 {
    fn from(v: CuDevicePtr) -> Self { v.0 }
}
impl From<u64> for CuContext {
    fn from(v: u64) -> Self { Self(v) }
}
impl From<CuContext> for u64 {
    fn from(v: CuContext) -> Self { v.0 }
}
impl From<u64> for CuModule {
    fn from(v: u64) -> Self { Self(v) }
}
impl From<CuModule> for u64 {
    fn from(v: CuModule) -> Self { v.0 }
}
impl From<u64> for CuFunction {
    fn from(v: u64) -> Self { Self(v) }
}
impl From<CuFunction> for u64 {
    fn from(v: CuFunction) -> Self { v.0 }
}
impl From<u64> for CuStream {
    fn from(v: u64) -> Self { Self(v) }
}
impl From<CuStream> for u64 {
    fn from(v: CuStream) -> Self { v.0 }
}
impl From<u64> for CuEvent {
    fn from(v: u64) -> Self { Self(v) }
}
impl From<CuEvent> for u64 {
    fn from(v: CuEvent) -> Self { v.0 }
}
```

---

## 5. Error Types (`error.rs`)

```rust
//! Error types for OutterLink.
//!
//! Design principle: every error must be mappable back to a CUresult
//! so the CUDA application never sees a non-CUDA error type.

use crate::cuda_types::CuResult;

/// The main error type used throughout OutterLink.
#[derive(Debug, thiserror::Error)]
pub enum OutterLinkError {
    // -- Network errors --
    #[error("connection to server failed: {0}")]
    ConnectionFailed(String),

    #[error("connection lost during operation: {0}")]
    ConnectionLost(String),

    #[error("request timed out after {0}ms")]
    Timeout(u64),

    #[error("server at {addr} refused connection")]
    ConnectionRefused { addr: String },

    // -- Protocol errors --
    #[error("protocol version mismatch: local={local}, remote={remote}")]
    ProtocolMismatch { local: u32, remote: u32 },

    #[error("failed to serialize message: {0}")]
    SerializationError(String),

    #[error("failed to deserialize message: {0}")]
    DeserializationError(String),

    #[error("unexpected response type: expected {expected}, got {got}")]
    UnexpectedResponse { expected: String, got: String },

    // -- CUDA errors forwarded from server --
    #[error("remote CUDA error: {0:?}")]
    CudaError(CuResult),

    // -- Handle errors --
    #[error("unknown handle: no mapping exists for the given handle")]
    UnknownHandle,

    #[error("handle already freed")]
    HandleAlreadyFreed,

    // -- Server errors --
    #[error("no GPU available on server")]
    NoGpuAvailable,

    #[error("failed to load libcuda.so: {0}")]
    CudaLibraryLoadFailed(String),

    #[error("CUDA symbol not found: {0}")]
    CudaSymbolNotFound(String),

    // -- Config errors --
    #[error("configuration error: {0}")]
    ConfigError(String),

    // -- IO errors --
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl OutterLinkError {
    /// Map any OutterLink error to a CUresult that the application can handle.
    /// This is the final translation layer -- the CUDA application must always
    /// receive a valid CUresult, never a Rust error.
    pub fn to_cuda_result(&self) -> CuResult {
        match self {
            Self::ConnectionFailed(_)
            | Self::ConnectionLost(_)
            | Self::ConnectionRefused { .. } => CuResult::ErrorNetworkDisconnected,

            Self::Timeout(_) => CuResult::ErrorNetworkTimeout,

            Self::ProtocolMismatch { .. } => CuResult::ErrorProtocolMismatch,

            Self::SerializationError(_)
            | Self::DeserializationError(_)
            | Self::UnexpectedResponse { .. } => CuResult::ErrorUnknown,

            Self::CudaError(result) => *result,

            Self::UnknownHandle
            | Self::HandleAlreadyFreed => CuResult::ErrorInvalidHandle,

            Self::NoGpuAvailable => CuResult::ErrorNoDevice,

            Self::CudaLibraryLoadFailed(_)
            | Self::CudaSymbolNotFound(_) => CuResult::ErrorSharedObjectInitFailed,

            Self::ConfigError(_) => CuResult::ErrorInvalidValue,

            Self::Io(_) => CuResult::ErrorOperatingSystem,
        }
    }
}

/// Convenience type alias.
pub type Result<T> = std::result::Result<T, OutterLinkError>;
```

---

## 6. Protocol Messages (`protocol.rs`)

```rust
//! Wire protocol messages between client and server.
//!
//! Every CUDA call becomes a Request sent from client to server.
//! The server executes it on the real GPU and sends back a Response.
//!
//! Messages are length-prefixed on the wire:
//!   [4 bytes: payload length (u32 big-endian)] [payload bytes]
//!
//! For bulk data (memcpy), the payload contains a fixed header followed
//! by raw bytes (not bincode-encoded) for the data portion.

/// Protocol version. Incremented on breaking wire format changes.
pub const PROTOCOL_VERSION: u32 = 1;

/// Maximum message size (256 MB). Bulk transfers can be larger but are
/// streamed in chunks.
pub const MAX_MESSAGE_SIZE: u32 = 256 * 1024 * 1024;

/// Chunk size for streaming bulk data (1 MB).
pub const BULK_CHUNK_SIZE: usize = 1024 * 1024;

use crate::cuda_types::*;

// ──────────────────────────────────────────────────────────────────
// Handshake (first message on connection)
// ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct Handshake {
    pub protocol_version: u32,
    pub client_id: uuid::Uuid,
    /// Human-readable name for logging
    pub client_name: String,
}

#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct HandshakeResponse {
    pub protocol_version: u32,
    pub server_id: uuid::Uuid,
    pub server_name: String,
    /// Number of GPUs available on this server
    pub gpu_count: i32,
    pub accepted: bool,
    pub reject_reason: Option<String>,
}

// ──────────────────────────────────────────────────────────────────
// Request: client -> server
// ──────────────────────────────────────────────────────────────────

/// Every CUDA call is encoded as a Request variant.
/// The `request_id` is a monotonically increasing u64 used to match
/// responses to requests (important for pipelined/async calls).
#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct RequestEnvelope {
    pub request_id: u64,
    pub payload: Request,
}

#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub enum Request {
    // -- Initialization --
    Init { flags: u32 },
    DriverGetVersion,

    // -- Device queries --
    DeviceGetCount,
    DeviceGet { ordinal: i32 },
    DeviceGetName { device: CuDevice },
    DeviceGetAttribute { attrib: i32, device: CuDevice },
    DeviceTotalMem { device: CuDevice },
    DeviceGetUuid { device: CuDevice },
    DeviceGetProperties { device: CuDevice },
    DeviceComputeCapability { device: CuDevice },

    // -- Context --
    CtxCreate { flags: u32, device: CuDevice },
    CtxDestroy { ctx: u64 },
    CtxSetCurrent { ctx: u64 },
    CtxGetCurrent,
    CtxSynchronize,
    CtxGetDevice,

    // -- Memory allocation --
    MemAlloc { byte_size: u64 },
    MemFree { dptr: u64 },
    MemAllocHost { byte_size: u64 },
    MemFreeHost { ptr: u64 },
    MemGetInfo,

    // -- Memory transfer --
    /// Host-to-Device: data bytes follow the envelope as raw bytes.
    /// The `byte_count` field tells the server how many raw bytes to
    /// read after the envelope.
    MemcpyHtoD {
        dst_device: u64,
        byte_count: u64,
        // Raw data bytes follow this message on the wire
    },
    /// Device-to-Host: server responds with data bytes.
    MemcpyDtoH {
        src_device: u64,
        byte_count: u64,
    },
    /// Device-to-Device (same server).
    MemcpyDtoD {
        dst_device: u64,
        src_device: u64,
        byte_count: u64,
    },

    // -- Module --
    /// Module data (PTX or cubin) follows as raw bytes after envelope.
    ModuleLoadData { data_size: u64 },
    ModuleUnload { module: u64 },
    ModuleGetFunction { module: u64, name: String },
    ModuleGetGlobal { module: u64, name: String },

    // -- Kernel launch --
    LaunchKernel {
        func: u64,
        grid_dim: [u32; 3],
        block_dim: [u32; 3],
        shared_mem_bytes: u32,
        stream: u64,
        /// Serialized kernel arguments.
        /// Each argument is a (offset, size, bytes) tuple.
        /// Device pointers within args reference remote handles.
        kernel_params: Vec<KernelParam>,
    },

    // -- Streams (Phase 2+) --
    StreamCreate { flags: u32 },
    StreamDestroy { stream: u64 },
    StreamSynchronize { stream: u64 },
    StreamQuery { stream: u64 },

    // -- Events (Phase 2+) --
    EventCreate { flags: u32 },
    EventDestroy { event: u64 },
    EventRecord { event: u64, stream: u64 },
    EventSynchronize { event: u64 },
    EventQuery { event: u64 },
    EventElapsedTime { start: u64, end: u64 },

    // -- Misc --
    Ping,
    Disconnect,
}

/// A single kernel parameter.
#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct KernelParam {
    /// Size of this parameter in bytes.
    pub size: u32,
    /// Raw bytes of the parameter value.
    /// For device pointers, this contains the u64 remote handle value
    /// (already translated by the client before sending).
    pub data: Vec<u8>,
}

// ──────────────────────────────────────────────────────────────────
// Response: server -> client
// ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct ResponseEnvelope {
    pub request_id: u64,
    pub payload: Response,
}

#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub enum Response {
    // -- Common --
    /// Simple success with just a CUresult.
    Status { result: CuResult },

    // -- Init --
    InitResult { result: CuResult },
    DriverVersion { result: CuResult, version: i32 },

    // -- Device queries --
    DeviceCount { result: CuResult, count: i32 },
    Device { result: CuResult, device: CuDevice },
    DeviceName { result: CuResult, name: String },
    DeviceAttribute { result: CuResult, value: i32 },
    DeviceTotalMem { result: CuResult, bytes: u64 },
    DeviceUuid { result: CuResult, uuid: [u8; 16] },
    DeviceProperties {
        result: CuResult,
        /// Serialized properties struct -- large, so we use raw bytes.
        properties: Vec<u8>,
    },
    DeviceComputeCapability {
        result: CuResult,
        major: i32,
        minor: i32,
    },

    // -- Context --
    CtxCreated { result: CuResult, ctx: u64 },
    CtxCurrent { result: CuResult, ctx: u64 },
    CtxDevice { result: CuResult, device: CuDevice },

    // -- Memory --
    MemAllocated { result: CuResult, dptr: u64 },
    MemHostAllocated { result: CuResult, ptr: u64 },
    MemInfo { result: CuResult, free: u64, total: u64 },

    /// Device-to-Host response. Raw data bytes follow on the wire
    /// after this envelope.
    MemcpyDtoHResult {
        result: CuResult,
        byte_count: u64,
        // Raw data bytes follow this message on the wire
    },

    // -- Module --
    ModuleLoaded { result: CuResult, module: u64 },
    FunctionHandle { result: CuResult, func: u64 },
    GlobalHandle { result: CuResult, dptr: u64, size: u64 },

    // -- Kernel --
    LaunchResult { result: CuResult },

    // -- Stream --
    StreamCreated { result: CuResult, stream: u64 },
    StreamQueryResult { result: CuResult },

    // -- Event --
    EventCreated { result: CuResult, event: u64 },
    EventQueryResult { result: CuResult },
    EventElapsedTimeResult { result: CuResult, milliseconds: f32 },

    // -- Misc --
    Pong,
    Disconnected,
}
```

---

## 7. Transport Trait (`transport.rs`)

```rust
//! Pluggable transport abstraction.
//!
//! Design: The transport trait abstracts the connection between client
//! and server. Phase 1 uses TCP. Phase 2 adds UCX (which auto-negotiates
//! RDMA vs TCP vs shared memory).
//!
//! The trait operates at the message level, not the byte stream level.
//! This lets transports handle framing, chunking, and zero-copy internally.

use crate::error::Result;
use crate::protocol::{
    Handshake, HandshakeResponse, RequestEnvelope, ResponseEnvelope,
};

/// A connection to a remote peer (either client-side or server-side).
///
/// Trait object safe: can be used as `Box<dyn TransportConnection>`.
#[async_trait::async_trait]
pub trait TransportConnection: Send + Sync {
    /// Send a request message. For messages with trailing bulk data
    /// (MemcpyHtoD, ModuleLoadData), the bulk data is passed separately
    /// to avoid copying it into the envelope.
    async fn send_request(
        &self,
        envelope: &RequestEnvelope,
        bulk_data: Option<&[u8]>,
    ) -> Result<()>;

    /// Receive a request message. Returns the envelope and any trailing
    /// bulk data.
    async fn recv_request(&self) -> Result<(RequestEnvelope, Option<Vec<u8>>)>;

    /// Send a response message with optional bulk data.
    async fn send_response(
        &self,
        envelope: &ResponseEnvelope,
        bulk_data: Option<&[u8]>,
    ) -> Result<()>;

    /// Receive a response message with optional bulk data.
    async fn recv_response(&self) -> Result<(ResponseEnvelope, Option<Vec<u8>>)>;

    /// Close the connection gracefully.
    async fn close(&self) -> Result<()>;

    /// Check if the connection is still alive.
    fn is_connected(&self) -> bool;

    /// Get the remote address as a string (for logging).
    fn remote_addr(&self) -> String;
}

/// Factory for creating transport connections.
///
/// The client uses `connect()` to establish a connection to a server.
/// The server uses `listen()` to accept incoming connections.
#[async_trait::async_trait]
pub trait TransportFactory: Send + Sync {
    /// Connect to a remote server.
    async fn connect(&self, addr: &str) -> Result<Box<dyn TransportConnection>>;

    /// Start listening for incoming connections.
    /// Returns a listener that yields connections.
    async fn listen(&self, addr: &str) -> Result<Box<dyn TransportListener>>;

    /// Human-readable transport name (e.g., "tcp", "ucx").
    fn name(&self) -> &str;
}

/// A listener that accepts incoming connections.
#[async_trait::async_trait]
pub trait TransportListener: Send + Sync {
    /// Accept the next incoming connection.
    /// Performs the handshake and returns the connection + handshake data.
    async fn accept(&self) -> Result<(Box<dyn TransportConnection>, Handshake)>;

    /// The address this listener is bound to.
    fn local_addr(&self) -> String;

    /// Stop listening.
    async fn close(&self) -> Result<()>;
}
```

**Note on `async_trait`:** Add `async-trait = "0.1"` to workspace dependencies. Once Rust stabilizes async trait methods (expected in 2026), we remove this dependency and use native async traits.

---

## 8. Configuration (`config.rs`)

```rust
//! Configuration types for OutterLink nodes.
//!
//! Config file location: /etc/outterlink/config.toml (system-wide)
//! or ~/.config/outterlink/config.toml (user) or OUTTERLINK_CONFIG env var.

use serde::{Deserialize, Serialize};

/// Top-level configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: Option<ServerConfig>,
    pub client: Option<ClientConfig>,
    pub transport: TransportConfig,
    pub logging: LoggingConfig,
}

/// Server-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Address to bind to (e.g., "0.0.0.0:9470").
    pub bind_address: String,
    /// Which GPU devices to expose (e.g., [0, 1]).
    /// Empty means expose all GPUs.
    pub gpu_devices: Vec<i32>,
    /// Maximum number of concurrent client sessions.
    pub max_clients: u32,
    /// Human-readable name for this server node.
    pub name: String,
}

/// Client-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientConfig {
    /// List of server addresses to connect to.
    pub servers: Vec<ServerAddress>,
    /// Connection timeout in milliseconds.
    pub connect_timeout_ms: u64,
    /// Request timeout in milliseconds (0 = no timeout).
    pub request_timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerAddress {
    /// Server address (e.g., "192.168.100.1:9470").
    pub address: String,
    /// Optional human-readable name.
    pub name: Option<String>,
}

/// Transport configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportConfig {
    /// Transport backend: "tcp" (Phase 1) or "ucx" (Phase 2).
    pub backend: String,
    /// TCP-specific settings.
    pub tcp: Option<TcpConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TcpConfig {
    /// Enable TCP_NODELAY (disable Nagle's algorithm).
    pub nodelay: bool,
    /// TCP send buffer size in bytes.
    pub send_buffer_size: Option<u32>,
    /// TCP receive buffer size in bytes.
    pub recv_buffer_size: Option<u32>,
}

/// Logging configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level: "trace", "debug", "info", "warn", "error".
    pub level: String,
    /// Log to file path (optional, defaults to stderr).
    pub file: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: None,
            client: None,
            transport: TransportConfig {
                backend: "tcp".to_string(),
                tcp: Some(TcpConfig {
                    nodelay: true,
                    send_buffer_size: Some(4 * 1024 * 1024),
                    recv_buffer_size: Some(4 * 1024 * 1024),
                }),
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                file: None,
            },
        }
    }
}

impl Config {
    /// Load configuration from the standard search path:
    /// 1. OUTTERLINK_CONFIG env var
    /// 2. ./outterlink.toml (current directory)
    /// 3. ~/.config/outterlink/config.toml
    /// 4. /etc/outterlink/config.toml
    /// Falls back to defaults if no config file found.
    pub fn load() -> Self {
        // Implementation: try each path, parse TOML, merge with defaults
        todo!("implement config loading")
    }

    /// Load from a specific file path.
    pub fn load_from(path: &std::path::Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read config file: {e}"))?;
        toml::from_str(&content)
            .map_err(|e| format!("failed to parse config file: {e}"))
    }
}
```

Example `outterlink.toml`:

```toml
[server]
bind_address = "0.0.0.0:9470"
gpu_devices = [0, 1]
max_clients = 16
name = "gpu-node-1"

[client]
connect_timeout_ms = 5000
request_timeout_ms = 30000

[[client.servers]]
address = "192.168.100.1:9470"
name = "gpu-node-1"

[[client.servers]]
address = "192.168.100.2:9470"
name = "gpu-node-2"

[transport]
backend = "tcp"

[transport.tcp]
nodelay = true
send_buffer_size = 4194304
recv_buffer_size = 4194304

[logging]
level = "info"
```

---

## 9. The C Interposition Layer

### Architecture

```
Application (calls cuInit, cuMemAlloc, etc.)
    |
    | LD_PRELOAD=liboutterlink_interpose.so
    v
liboutterlink_interpose.so  (C, ~500 lines)
    |-- Overrides dlsym() to intercept CUDA symbol lookups
    |-- Overrides cuGetProcAddress() for CUDA 11.3+
    |-- For each intercepted call: marshals arguments into C struct,
    |   calls outterlink_forward_call() (Rust FFI function in liboutterlink_client.so)
    |-- Returns CUresult to application
    |
    v
liboutterlink_client.so  (Rust cdylib, outterlink-client crate)
    |-- outterlink_forward_call() receives marshalled args
    |-- Serializes into protocol message
    |-- Sends over transport to server
    |-- Receives response
    |-- Returns CUresult to C layer
```

### `interpose/cuda_subset.h` -- Minimal CUDA types (no NVIDIA headers)

```c
/* Minimal CUDA type definitions for the interposition layer.
 * We define only what we need to avoid depending on NVIDIA's cuda.h.
 * These match the CUDA Driver API ABI exactly. */

#ifndef OUTTERLINK_CUDA_SUBSET_H
#define OUTTERLINK_CUDA_SUBSET_H

#include <stdint.h>
#include <stddef.h>

typedef int CUresult;
typedef int CUdevice;
typedef uint64_t CUdeviceptr;

/* Opaque handles -- all are pointers in CUDA, but we treat them as u64 */
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;
typedef void* CUevent;

#define CUDA_SUCCESS 0
#define CUDA_ERROR_NOT_INITIALIZED 3
#define CUDA_ERROR_UNKNOWN 999

/* cuGetProcAddress flags */
typedef uint64_t cuuint64_t;

#endif /* OUTTERLINK_CUDA_SUBSET_H */
```

### `interpose/interpose.h` -- Interface to Rust client

```c
/* Functions exported by the Rust client library (liboutterlink_client.so).
 * These are called from the C interposition layer. */

#ifndef OUTTERLINK_INTERPOSE_H
#define OUTTERLINK_INTERPOSE_H

#include "cuda_subset.h"

/* Initialize the OutterLink client runtime.
 * Called once from __attribute__((constructor)). */
int outterlink_init(void);

/* Shutdown the client runtime.
 * Called from __attribute__((destructor)). */
void outterlink_shutdown(void);

/* ── Forwarded CUDA calls ── */

/* Each function corresponds to a CUDA Driver API function.
 * The C layer calls these, which serialize and forward to the server. */

CUresult outterlink_cuInit(unsigned int flags);
CUresult outterlink_cuDriverGetVersion(int* version);
CUresult outterlink_cuDeviceGetCount(int* count);
CUresult outterlink_cuDeviceGet(CUdevice* device, int ordinal);
CUresult outterlink_cuDeviceGetName(char* name, int len, CUdevice dev);
CUresult outterlink_cuDeviceGetAttribute(int* pi, int attrib, CUdevice dev);
CUresult outterlink_cuDeviceTotalMem(size_t* bytes, CUdevice dev);

CUresult outterlink_cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev);
CUresult outterlink_cuCtxDestroy(CUcontext ctx);
CUresult outterlink_cuCtxSetCurrent(CUcontext ctx);
CUresult outterlink_cuCtxGetCurrent(CUcontext* pctx);
CUresult outterlink_cuCtxSynchronize(void);

CUresult outterlink_cuMemAlloc(CUdeviceptr* dptr, size_t bytesize);
CUresult outterlink_cuMemFree(CUdeviceptr dptr);
CUresult outterlink_cuMemcpyHtoD(CUdeviceptr dst, const void* src, size_t count);
CUresult outterlink_cuMemcpyDtoH(void* dst, CUdeviceptr src, size_t count);
CUresult outterlink_cuMemGetInfo(size_t* free, size_t* total);

CUresult outterlink_cuModuleLoadData(CUmodule* module, const void* image);
CUresult outterlink_cuModuleUnload(CUmodule hmod);
CUresult outterlink_cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod,
                                         const char* name);

CUresult outterlink_cuLaunchKernel(CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream,
    void** kernelParams, void** extra);

CUresult outterlink_cuStreamCreate(CUstream* phStream, unsigned int flags);
CUresult outterlink_cuStreamDestroy(CUstream hStream);
CUresult outterlink_cuStreamSynchronize(CUstream hStream);

CUresult outterlink_cuEventCreate(CUevent* phEvent, unsigned int flags);
CUresult outterlink_cuEventDestroy(CUevent hEvent);
CUresult outterlink_cuEventRecord(CUevent hEvent, CUstream hStream);
CUresult outterlink_cuEventSynchronize(CUevent hEvent);
CUresult outterlink_cuEventElapsedTime(float* pMilliseconds,
                                        CUevent hStart, CUevent hEnd);

/* Phase 1 only needs the above. More functions added in later phases. */

#endif /* OUTTERLINK_INTERPOSE_H */
```

### `interpose/interpose.c` -- The LD_PRELOAD library

```c
/* OutterLink CUDA Interposition Layer
 *
 * This library is loaded via LD_PRELOAD before libcuda.so.
 * It intercepts all CUDA Driver API symbol lookups and redirects
 * them to the OutterLink client library.
 *
 * Architecture follows HAMi-core's proven pattern:
 * 1. Override dlsym() to catch direct symbol lookups
 * 2. Override cuGetProcAddress() to catch CUDA 11.3+ dynamic resolution
 * 3. Forward all intercepted calls to outterlink_* functions (Rust FFI)
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "cuda_subset.h"
#include "interpose.h"

/* ── State ── */

static int g_initialized = 0;
static int g_initializing = 0; /* Prevent recursive init */

/* Pointer to the real dlsym (resolved via dlvsym to avoid infinite recursion) */
typedef void* (*real_dlsym_t)(void*, const char*);
static real_dlsym_t real_dlsym_fn = NULL;

/* ── Constructor / Destructor ── */

__attribute__((constructor))
static void outterlink_interpose_init(void) {
    if (g_initialized || g_initializing) return;
    g_initializing = 1;

    /* Resolve real dlsym using dlvsym (which does NOT go through our override) */
    real_dlsym_fn = (real_dlsym_t)dlvsym(RTLD_DEFAULT, "dlsym", "GLIBC_2.2.5");
    if (!real_dlsym_fn) {
        /* Fallback for musl or non-glibc systems */
        real_dlsym_fn = (real_dlsym_t)dlvsym(RTLD_DEFAULT, "dlsym", "GLIBC_2.34");
    }

    /* Initialize the Rust client runtime */
    int ret = outterlink_init();
    if (ret != 0) {
        fprintf(stderr, "[outterlink] WARNING: client initialization failed (%d)\n", ret);
        /* Continue anyway -- calls will return CUDA_ERROR_NOT_INITIALIZED */
    }

    g_initializing = 0;
    g_initialized = 1;
}

__attribute__((destructor))
static void outterlink_interpose_fini(void) {
    if (g_initialized) {
        outterlink_shutdown();
        g_initialized = 0;
    }
}

/* ── Symbol Hook Table ── */

/* Macro to generate hook entries.
 * Each entry maps a CUDA symbol name to our outterlink_* function. */
#define HOOK_ENTRY(cuda_name) \
    { #cuda_name, (void*)outterlink_##cuda_name }

/* Also handle versioned variants (_v2, _v3) that map to the same function */
#define HOOK_ENTRY_V2(cuda_name) \
    { #cuda_name "_v2", (void*)outterlink_##cuda_name }

static struct {
    const char* name;
    void* hook_fn;
} g_hook_table[] = {
    /* Initialization */
    HOOK_ENTRY(cuInit),
    HOOK_ENTRY(cuDriverGetVersion),

    /* Device queries */
    HOOK_ENTRY(cuDeviceGetCount),
    HOOK_ENTRY(cuDeviceGet),
    HOOK_ENTRY(cuDeviceGetName),
    HOOK_ENTRY(cuDeviceGetAttribute),
    HOOK_ENTRY(cuDeviceTotalMem),
    HOOK_ENTRY_V2(cuDeviceTotalMem),

    /* Context */
    HOOK_ENTRY(cuCtxCreate),
    HOOK_ENTRY_V2(cuCtxCreate),
    HOOK_ENTRY(cuCtxDestroy),
    HOOK_ENTRY_V2(cuCtxDestroy),
    HOOK_ENTRY(cuCtxSetCurrent),
    HOOK_ENTRY(cuCtxGetCurrent),
    HOOK_ENTRY(cuCtxSynchronize),

    /* Memory */
    HOOK_ENTRY(cuMemAlloc),
    HOOK_ENTRY_V2(cuMemAlloc),
    HOOK_ENTRY(cuMemFree),
    HOOK_ENTRY_V2(cuMemFree),
    HOOK_ENTRY(cuMemcpyHtoD),
    HOOK_ENTRY_V2(cuMemcpyHtoD),
    HOOK_ENTRY(cuMemcpyDtoH),
    HOOK_ENTRY_V2(cuMemcpyDtoH),
    HOOK_ENTRY(cuMemGetInfo),
    HOOK_ENTRY_V2(cuMemGetInfo),

    /* Module */
    HOOK_ENTRY(cuModuleLoadData),
    HOOK_ENTRY(cuModuleUnload),
    HOOK_ENTRY(cuModuleGetFunction),

    /* Kernel launch */
    HOOK_ENTRY(cuLaunchKernel),

    /* Streams */
    HOOK_ENTRY(cuStreamCreate),
    HOOK_ENTRY(cuStreamDestroy),
    HOOK_ENTRY_V2(cuStreamDestroy),
    HOOK_ENTRY(cuStreamSynchronize),

    /* Events */
    HOOK_ENTRY(cuEventCreate),
    HOOK_ENTRY(cuEventDestroy),
    HOOK_ENTRY(cuEventRecord),
    HOOK_ENTRY(cuEventSynchronize),
    HOOK_ENTRY(cuEventElapsedTime),

    /* Sentinel */
    { NULL, NULL }
};

/* Look up a symbol in our hook table.
 * Returns the hook function pointer, or NULL if not hooked. */
static void* find_hook(const char* symbol) {
    for (int i = 0; g_hook_table[i].name != NULL; i++) {
        if (strcmp(symbol, g_hook_table[i].name) == 0) {
            return g_hook_table[i].hook_fn;
        }
    }
    return NULL;
}

/* ── dlsym Override ── */

/* This is the core of the interposition. When any code calls dlsym()
 * to look up a CUDA symbol, we intercept it and return our hook. */
void* dlsym(void* handle, const char* symbol) {
    /* Prevent recursion during initialization */
    if (g_initializing) {
        if (real_dlsym_fn) return real_dlsym_fn(handle, symbol);
        return NULL;
    }

    /* Check if this is a CUDA symbol we want to hook */
    void* hook = find_hook(symbol);
    if (hook) return hook;

    /* Also intercept cuGetProcAddress itself */
    if (strcmp(symbol, "cuGetProcAddress") == 0 ||
        strcmp(symbol, "cuGetProcAddress_v2") == 0) {
        return (void*)outterlink_cuGetProcAddress;
    }

    /* Not a hooked symbol -- pass through to real dlsym */
    if (real_dlsym_fn) return real_dlsym_fn(handle, symbol);
    return NULL;
}

/* ── cuGetProcAddress Hook ── */

/* CUDA 11.3+ uses cuGetProcAddress to resolve driver functions at runtime.
 * We must intercept this to catch dynamically resolved symbols. */
CUresult outterlink_cuGetProcAddress(const char* symbol, void** pfn,
                                      int cudaVersion, cuuint64_t flags) {
    /* First, check our hook table */
    void* hook = find_hook(symbol);
    if (hook) {
        *pfn = hook;
        return CUDA_SUCCESS;
    }

    /* For cuGetProcAddress itself, return our hook */
    if (strcmp(symbol, "cuGetProcAddress") == 0 ||
        strcmp(symbol, "cuGetProcAddress_v2") == 0) {
        *pfn = (void*)outterlink_cuGetProcAddress;
        return CUDA_SUCCESS;
    }

    /* Symbol not hooked -- let the real CUDA driver resolve it.
     * This requires loading the real libcuda.so. */
    typedef CUresult (*real_cuGetProcAddress_t)(const char*, void**, int, cuuint64_t);
    static real_cuGetProcAddress_t real_fn = NULL;
    if (!real_fn) {
        void* cuda_handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
        if (cuda_handle && real_dlsym_fn) {
            real_fn = (real_cuGetProcAddress_t)real_dlsym_fn(
                cuda_handle, "cuGetProcAddress");
        }
    }

    if (real_fn) {
        return real_fn(symbol, pfn, cudaVersion, flags);
    }

    /* Cannot resolve -- return error */
    return CUDA_ERROR_NOT_INITIALIZED;
}
```

### Build Script (`crates/outterlink-client/build.rs`)

```rust
fn main() {
    // Only compile the C interposition library on Linux
    #[cfg(target_os = "linux")]
    {
        cc::Build::new()
            .file("interpose/interpose.c")
            .include("interpose")
            .flag("-fPIC")
            .flag("-Wall")
            .flag("-Wextra")
            .flag("-Werror")
            .flag("-O2")
            .compile("outterlink_interpose");

        // Link against libdl for dlsym/dlopen
        println!("cargo:rustc-link-lib=dl");
    }

    #[cfg(not(target_os = "linux"))]
    {
        // On non-Linux, skip C compilation (for CI or cross-platform builds).
        // The interposition layer is Linux-only (LD_PRELOAD).
        eprintln!("WARNING: interposition layer only builds on Linux");
    }
}
```

**Important note on the .so output:** The `cc` crate compiles `interpose.c` into a static library that gets linked INTO `liboutterlink_client.so` (the Rust cdylib). The final product is a single `.so` file that:
1. Exports the `dlsym` override (from the C code)
2. Exports the `outterlink_*` functions (from the Rust code, via `#[no_mangle] extern "C"`)
3. Can be loaded via `LD_PRELOAD`

The application sets `LD_PRELOAD=liboutterlink_client.so` and everything works from a single library.

---

## 10. Rust Client FFI Exports (`ffi.rs`)

```rust
//! C FFI functions exported by the client library.
//! These are called from the C interposition layer (interpose.c)
//! and also constitute the public C API of liboutterlink_client.so.

use std::ffi::CStr;
use std::sync::OnceLock;

// The global client state, initialized once
static CLIENT: OnceLock<crate::connection::ClientRuntime> = OnceLock::new();

/// Initialize the OutterLink client runtime.
/// Called from the C constructor in interpose.c.
#[no_mangle]
pub extern "C" fn outterlink_init() -> i32 {
    // Load config, connect to server(s), set up handle maps
    // Returns 0 on success, -1 on failure
    match crate::connection::ClientRuntime::initialize() {
        Ok(runtime) => {
            let _ = CLIENT.set(runtime);
            0
        }
        Err(e) => {
            eprintln!("[outterlink] init failed: {e}");
            -1
        }
    }
}

/// Shutdown the client runtime.
#[no_mangle]
pub extern "C" fn outterlink_shutdown() {
    // Drop the runtime, close connections
    // OnceLock doesn't support take(), so we signal shutdown
    if let Some(client) = CLIENT.get() {
        client.shutdown();
    }
}

/// Example: cuDeviceGetCount
#[no_mangle]
pub extern "C" fn outterlink_cuDeviceGetCount(count: *mut i32) -> u32 {
    let Some(client) = CLIENT.get() else {
        return 3; // CUDA_ERROR_NOT_INITIALIZED
    };
    match client.device_get_count() {
        Ok(n) => {
            unsafe { *count = n };
            0 // CUDA_SUCCESS
        }
        Err(e) => e.to_cuda_result().to_raw(),
    }
}

// ... similar #[no_mangle] extern "C" functions for every hooked CUDA call.
// Each function:
// 1. Gets the CLIENT singleton
// 2. Calls the corresponding method on ClientRuntime
// 3. Writes output values through raw pointers
// 4. Returns CUresult as u32
```

---

## 11. Server Entry Point (`main.rs` skeleton)

```rust
//! OutterLink GPU Server
//!
//! Receives CUDA Driver API calls from remote clients,
//! executes them on the local GPU, and returns results.

use anyhow::Result;
use clap::Parser;
use tracing::{info, error};

#[derive(Parser, Debug)]
#[command(name = "outterlink-server")]
#[command(about = "OutterLink GPU server - exposes local GPUs to remote clients")]
struct Args {
    /// Path to configuration file
    #[arg(short, long)]
    config: Option<String>,

    /// Bind address (overrides config file)
    #[arg(short, long, default_value = "0.0.0.0:9470")]
    bind: String,

    /// GPU device indices to expose (e.g., "0,1")
    #[arg(short, long)]
    gpus: Option<String>,

    /// Log level
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(&args.log_level)
        .init();

    info!("OutterLink server starting");

    // Load configuration
    let config = match &args.config {
        Some(path) => {
            outterlink_common::config::Config::load_from(
                std::path::Path::new(path)
            ).map_err(|e| anyhow::anyhow!(e))?
        }
        None => outterlink_common::config::Config::default(),
    };

    // Initialize GPU executor (loads libcuda.so)
    let gpu_executor = crate::gpu_executor::GpuExecutor::new(&config)?;
    info!("GPU executor initialized with {} GPUs", gpu_executor.gpu_count());

    // Start server
    let server = crate::server::Server::new(config, gpu_executor);
    server.run(&args.bind).await?;

    Ok(())
}

mod server;
mod dispatcher;
mod gpu_executor;
mod session;
```

---

## 12. Server GPU Executor (`gpu_executor.rs` skeleton)

```rust
//! GPU Executor: loads libcuda.so at runtime and executes real CUDA calls.
//!
//! Uses `libloading` to dynamically load the CUDA driver library.
//! This means the server binary compiles without CUDA headers/libraries
//! and only requires libcuda.so at runtime.

use libloading::{Library, Symbol};
use outterlink_common::cuda_types::*;
use outterlink_common::error::{OutterLinkError, Result};
use tracing::{debug, error};

/// Holds function pointers to the real CUDA Driver API functions.
pub struct CudaDriver {
    _lib: Library,
    // Function pointers resolved from libcuda.so
    cu_init: unsafe extern "C" fn(u32) -> u32,
    cu_device_get_count: unsafe extern "C" fn(*mut i32) -> u32,
    cu_device_get: unsafe extern "C" fn(*mut i32, i32) -> u32,
    cu_device_get_name: unsafe extern "C" fn(*mut u8, i32, i32) -> u32,
    cu_device_get_attribute: unsafe extern "C" fn(*mut i32, i32, i32) -> u32,
    cu_device_total_mem: unsafe extern "C" fn(*mut u64, i32) -> u32,
    cu_ctx_create: unsafe extern "C" fn(*mut u64, u32, i32) -> u32,
    cu_ctx_destroy: unsafe extern "C" fn(u64) -> u32,
    cu_mem_alloc: unsafe extern "C" fn(*mut u64, u64) -> u32,
    cu_mem_free: unsafe extern "C" fn(u64) -> u32,
    cu_memcpy_htod: unsafe extern "C" fn(u64, *const u8, u64) -> u32,
    cu_memcpy_dtoh: unsafe extern "C" fn(*mut u8, u64, u64) -> u32,
    cu_module_load_data: unsafe extern "C" fn(*mut u64, *const u8) -> u32,
    cu_module_get_function: unsafe extern "C" fn(*mut u64, u64, *const u8) -> u32,
    cu_launch_kernel: unsafe extern "C" fn(
        u64, u32, u32, u32, u32, u32, u32, u32, u64, *mut *mut std::ffi::c_void, *mut *mut std::ffi::c_void
    ) -> u32,
    // ... more function pointers added as needed
}

impl CudaDriver {
    pub fn load() -> Result<Self> {
        let lib = unsafe {
            Library::new("libcuda.so.1")
                .or_else(|_| Library::new("libcuda.so"))
                .map_err(|e| OutterLinkError::CudaLibraryLoadFailed(e.to_string()))?
        };

        macro_rules! resolve {
            ($lib:expr, $name:expr) => {
                unsafe {
                    let sym: Symbol<unsafe extern "C" fn()> = $lib.get($name)
                        .map_err(|e| OutterLinkError::CudaSymbolNotFound(
                            format!("{}: {}", String::from_utf8_lossy($name), e)
                        ))?;
                    std::mem::transmute(*sym)
                }
            };
        }

        Ok(Self {
            cu_init: resolve!(lib, b"cuInit\0"),
            cu_device_get_count: resolve!(lib, b"cuDeviceGetCount\0"),
            cu_device_get: resolve!(lib, b"cuDeviceGet\0"),
            cu_device_get_name: resolve!(lib, b"cuDeviceGetName\0"),
            cu_device_get_attribute: resolve!(lib, b"cuDeviceGetAttribute\0"),
            cu_device_total_mem: resolve!(lib, b"cuDeviceTotalMem_v2\0"),
            cu_ctx_create: resolve!(lib, b"cuCtxCreate_v2\0"),
            cu_ctx_destroy: resolve!(lib, b"cuCtxDestroy_v2\0"),
            cu_mem_alloc: resolve!(lib, b"cuMemAlloc_v2\0"),
            cu_mem_free: resolve!(lib, b"cuMemFree_v2\0"),
            cu_memcpy_htod: resolve!(lib, b"cuMemcpyHtoD_v2\0"),
            cu_memcpy_dtoh: resolve!(lib, b"cuMemcpyDtoH_v2\0"),
            cu_module_load_data: resolve!(lib, b"cuModuleLoadData\0"),
            cu_module_get_function: resolve!(lib, b"cuModuleGetFunction\0"),
            cu_launch_kernel: resolve!(lib, b"cuLaunchKernel\0"),
            _lib: lib,
        })
    }

    pub fn init(&self, flags: u32) -> CuResult {
        CuResult::from_raw(unsafe { (self.cu_init)(flags) })
    }

    pub fn device_get_count(&self) -> (CuResult, i32) {
        let mut count: i32 = 0;
        let result = unsafe { (self.cu_device_get_count)(&mut count) };
        (CuResult::from_raw(result), count)
    }

    // ... similar safe wrappers for each function pointer
}

/// The GPU executor manages the CudaDriver and per-session state.
pub struct GpuExecutor {
    driver: CudaDriver,
    gpu_count: i32,
}

impl GpuExecutor {
    pub fn new(config: &outterlink_common::config::Config) -> Result<Self> {
        let driver = CudaDriver::load()?;
        let init_result = driver.init(0);
        if !init_result.is_success() {
            return Err(OutterLinkError::CudaError(init_result));
        }

        let (result, count) = driver.device_get_count();
        if !result.is_success() {
            return Err(OutterLinkError::CudaError(result));
        }

        debug!("CUDA initialized with {count} GPU(s)");
        Ok(Self {
            driver,
            gpu_count: count,
        })
    }

    pub fn gpu_count(&self) -> i32 {
        self.gpu_count
    }

    pub fn driver(&self) -> &CudaDriver {
        &self.driver
    }
}
```

---

## 13. CLI Entry Point (`outterlink-cli/src/main.rs` skeleton)

```rust
use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "outterlink")]
#[command(about = "OutterLink management CLI")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Configuration file path
    #[arg(short, long, global = true)]
    config: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Show status of all connected nodes
    Status,
    /// List all available GPUs across nodes
    List,
    /// Run latency and bandwidth benchmarks
    Benchmark {
        /// Target server address
        #[arg(short, long)]
        server: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    match cli.command {
        Commands::Status => {
            crate::commands::status::run(&cli.config).await?;
        }
        Commands::List => {
            crate::commands::list::run(&cli.config).await?;
        }
        Commands::Benchmark { server } => {
            crate::commands::benchmark::run(&server, &cli.config).await?;
        }
    }

    Ok(())
}

mod commands;
```

---

## 14. `outterlink-common/src/lib.rs` -- Module Re-exports

```rust
//! OutterLink Common Library
//!
//! Shared types, protocol messages, transport trait, error types,
//! and configuration used by all OutterLink crates.

pub mod cuda_types;
pub mod error;
pub mod handle;
pub mod protocol;
pub mod transport;
pub mod config;
```

---

## 15. Build System and Linking Strategy

### How the pieces link together

```
Build outputs:
  target/debug/liboutterlink_common.rlib     (Rust static lib, linked into everything)
  target/debug/liboutterlink_client.so        (cdylib -- THE LD_PRELOAD library)
  target/debug/liboutterlink_client.rlib      (Rust static lib, for CLI dependency)
  target/debug/outterlink-server              (binary)
  target/debug/outterlink                     (CLI binary)

Usage:
  # On GPU server machine:
  ./outterlink-server --bind 0.0.0.0:9470

  # On client machine:
  LD_PRELOAD=./liboutterlink_client.so ./my_cuda_app
```

### Workspace `async-trait` dependency

Add to root `Cargo.toml` workspace dependencies:

```toml
async-trait = "0.1"
```

And add to `outterlink-common/Cargo.toml` dependencies:

```toml
async-trait = { workspace = true }
```

### CI considerations

The C interposition code only compiles on Linux. The workspace should build on any platform for the Rust-only crates. The `build.rs` uses `#[cfg(target_os = "linux")]` to skip C compilation on other platforms. CI should test on `ubuntu-latest` for full builds.

The server binary does NOT require CUDA at build time (uses `libloading` for runtime loading). It only requires `libcuda.so` at runtime. This means CI can build and test the protocol/transport code without a GPU.

---

## 16. Implementation Phases for P4

### Phase 4a: Workspace scaffold (files, Cargo.toml, builds)

**Files to create:**
- `Cargo.toml` (workspace root)
- `crates/outterlink-common/Cargo.toml`
- `crates/outterlink-common/src/lib.rs`
- `crates/outterlink-common/src/cuda_types.rs`
- `crates/outterlink-common/src/error.rs`
- `crates/outterlink-common/src/handle.rs`
- `crates/outterlink-common/src/protocol.rs`
- `crates/outterlink-common/src/transport.rs`
- `crates/outterlink-common/src/config.rs`
- `crates/outterlink-server/Cargo.toml`
- `crates/outterlink-server/src/main.rs`
- `crates/outterlink-server/src/server.rs` (skeleton)
- `crates/outterlink-server/src/dispatcher.rs` (skeleton)
- `crates/outterlink-server/src/gpu_executor.rs`
- `crates/outterlink-server/src/session.rs` (skeleton)
- `crates/outterlink-client/Cargo.toml`
- `crates/outterlink-client/build.rs`
- `crates/outterlink-client/interpose/cuda_subset.h`
- `crates/outterlink-client/interpose/interpose.h`
- `crates/outterlink-client/interpose/interpose.c`
- `crates/outterlink-client/src/lib.rs`
- `crates/outterlink-client/src/ffi.rs`
- `crates/outterlink-client/src/connection.rs` (skeleton)
- `crates/outterlink-client/src/handle_map.rs` (skeleton)
- `crates/outterlink-client/src/call_forwarder.rs` (skeleton)
- `crates/outterlink-cli/Cargo.toml`
- `crates/outterlink-cli/src/main.rs`
- `crates/outterlink-cli/src/commands/mod.rs`
- `crates/outterlink-cli/src/commands/status.rs` (skeleton)
- `crates/outterlink-cli/src/commands/list.rs` (skeleton)
- `crates/outterlink-cli/src/commands/benchmark.rs` (skeleton)

**Acceptance criteria:**
- [ ] `cargo build --all` succeeds
- [ ] `cargo test --all` passes
- [ ] `cargo clippy --all-targets -- -D warnings` clean
- [ ] `cargo fmt --all -- --check` passes

### Phase 4b: Transport TCP implementation (skeleton)

**Files to create:**
- `crates/outterlink-common/src/transport/mod.rs` (split from single file)
- `crates/outterlink-common/src/transport/tcp.rs`

**Acceptance criteria:**
- [ ] TCP transport implements `TransportFactory`, `TransportListener`, `TransportConnection`
- [ ] Unit test: TCP connect, send request, receive response, disconnect
- [ ] Integration test: server starts, client connects, handshake succeeds

### Phase 4c: Configuration and integration

**Acceptance criteria:**
- [ ] `outterlink-server --help` prints usage
- [ ] `outterlink --help` prints usage
- [ ] `outterlink.toml` is parsed correctly
- [ ] Server starts and binds to configured address
- [ ] Environment variable `OUTTERLINK_CONFIG` overrides config path

---

## Testing Strategy

### Unit tests (no GPU needed)

| Test | Location | What it tests |
|------|----------|---------------|
| Handle map CRUD | `outterlink-common/src/handle.rs` | Register, lookup, remove, thread safety |
| Error to CuResult mapping | `outterlink-common/src/error.rs` | Every error variant maps to a valid CuResult |
| Protocol serialization roundtrip | `outterlink-common/src/protocol.rs` | Encode then decode every Request/Response variant |
| Config parsing | `outterlink-common/src/config.rs` | Parse valid TOML, reject invalid TOML |
| CuResult from_raw/to_raw | `outterlink-common/src/cuda_types.rs` | Known codes roundtrip, unknown maps to ErrorUnknown |

### Integration tests (no GPU needed)

| Test | Location | What it tests |
|------|----------|---------------|
| TCP transport roundtrip | `outterlink-common/tests/` | Client connects to server, sends request, gets response |
| Handshake flow | `outterlink-common/tests/` | Protocol version negotiation works |
| Multiple concurrent clients | `outterlink-common/tests/` | Handle map and transport work under contention |

### Tests requiring GPU (deferred to P5)

| Test | What it tests |
|------|---------------|
| GPU executor initialization | libcuda.so loads, cuInit succeeds, devices enumerated |
| Full end-to-end device query | Client queries device count, server returns real count |

---

## Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|-----------|
| `cdylib` + C static library linking doesn't export `dlsym` correctly | HIGH | MEDIUM | Test on Linux early. Fallback: ship two separate .so files and document `LD_PRELOAD` for both. |
| `bincode` 2.x API instability | LOW | LOW | Pin exact version. bincode 2.x is stable (1.0 of the 2.x line). |
| `dlvsym` GLIBC version string differs across distros | MEDIUM | MEDIUM | Test on Ubuntu 24.04, Fedora, Arch. Fallback: use `__libc_dlsym` if `dlvsym` fails. |
| `libloading` symbol resolution fails for versioned CUDA functions | MEDIUM | LOW | Try `_v2` first, fall back to unversioned. HAMi-core's `prior_function()` pattern. |
| Thread safety of `OnceLock` + tokio runtime initialization inside `LD_PRELOAD` | MEDIUM | MEDIUM | Initialize tokio runtime lazily on first CUDA call, not in constructor. Constructor only sets up minimal state. |
| CUDA applications that fork() after cuInit | HIGH | LOW | Document limitation. fork() after GPU init is problematic even without OutterLink. |

---

## Estimated Scope

| Component | Files | Lines (est.) | Complexity |
|-----------|-------|-------------|-----------|
| `outterlink-common` | 6 source files | ~1500 | Medium (types + protocol + trait) |
| `outterlink-server` | 5 source files | ~800 skeleton | Medium (entry point + executor) |
| `outterlink-client` | 4 Rust + 3 C files | ~1000 | High (FFI + C interposition) |
| `outterlink-cli` | 5 source files | ~300 skeleton | Low |
| Build system | 2 files (root Cargo.toml + build.rs) | ~100 | Low |
| **Total** | **~25 files** | **~3700** | |

---

## Related Documents

- [P1: GitHub Repository Setup](P1-github-repo-setup.md)
- [P2: Development Environment](P2-dev-environment.md)
- [R3: CUDA Interception Strategies](../research/R3-cuda-interception.md)
- [R4: ConnectX-5 + Transport Stack](../research/R4-connectx5-transport-stack.md)
- [Final Pre-Plan](../pre-planning/02-FINAL-PREPLAN.md)
- [Research Consolidation](../research/CONSOLIDATION-all-research.md)
- [ADR-001: License](../../docs/decisions/ADR-001-license.md)
