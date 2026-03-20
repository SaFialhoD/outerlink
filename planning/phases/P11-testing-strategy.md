# P11: Testing Strategy

**Created:** 2026-03-19
**Last Updated:** 2026-03-19
**Status:** Plan
**Priority:** HIGH - Parallel with P5 (PoC)

## Purpose

Define how every component of OutterLink is tested: unit tests for individual crates, integration tests for client-server communication, end-to-end tests with real GPUs, and CI-compatible testing without GPU hardware. Establish the mock/stub strategy that allows development and CI to proceed without physical GPU access.

## Goal

A comprehensive test suite that catches regressions across CUDA versions, verifies protocol correctness without GPU hardware, and validates real GPU behavior on hardware-equipped machines.

## Milestone

- `cargo test --all` passes without a GPU (using mocks/stubs)
- Integration tests verify client-server protocol over TCP loopback
- End-to-end tests on GPU hardware validate correct CUDA execution
- CI pipeline runs all non-GPU tests on every PR
- GPU tests run on self-hosted runner (or manually triggered)

## Prerequisites

- [ ] P4: Project Skeleton (crate structure established)
- [ ] P5: PoC Plan (defines what functions to test first)

---

## 1. Unit Test Plan

### 1.1 `outterlink-common` Unit Tests

This crate has zero external dependencies (no GPU, no network). All tests run everywhere.

| Test Area | What to Test | Test Count (est.) |
|-----------|-------------|-------------------|
| **Protocol serialization** | Every `OpCode` request/response round-trips through encode/decode | 23 (one per op) |
| **Wire framing** | Length-prefixed frame encoding/decoding | 5 |
| **Magic validation** | Invalid magic rejected, valid magic accepted | 3 |
| **Edge cases** | Zero-length payloads, max-size payloads, truncated frames | 6 |
| **Error mapping** | `CUresult` to/from wire format, all known error codes | 4 |
| **Handshake** | Handshake encode/decode, version mismatch handling | 3 |
| **Fuzz-style** | Deserialization of random bytes does not panic (property test) | 1 (proptest) |

```rust
// Example: protocol round-trip test
#[test]
fn test_mem_alloc_request_roundtrip() {
    let req = Request::MemAlloc { size: 4096 };
    let bytes = req.encode();
    let decoded = Request::decode(&bytes).unwrap();
    assert_eq!(req, decoded);
}

#[test]
fn test_mem_alloc_response_roundtrip() {
    let resp = Response::MemAlloc {
        result: CUDA_SUCCESS,
        dptr: 0x7f0000001000,
    };
    let bytes = resp.encode();
    let decoded = Response::decode(OpCode::MemAlloc, &bytes).unwrap();
    assert_eq!(resp, decoded);
}

#[test]
fn test_memcpy_htod_with_data() {
    let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    let req = Request::MemcpyHtoD {
        dptr: 0x7f0000001000,
        size: 8,
        data: data.clone(),
    };
    let bytes = req.encode();
    let decoded = Request::decode(&bytes).unwrap();
    match decoded {
        Request::MemcpyHtoD { data: d, .. } => assert_eq!(d, data),
        _ => panic!("wrong variant"),
    }
}

#[test]
fn test_invalid_magic_rejected() {
    let mut frame = vec![0xDE, 0xAD, 0xBE, 0xEF]; // wrong magic
    frame.extend_from_slice(&2u32.to_le_bytes()); // len
    frame.extend_from_slice(&1u16.to_le_bytes()); // op
    let result = Frame::decode(&frame);
    assert!(matches!(result, Err(ProtocolError::InvalidMagic)));
}
```

**Location:** `crates/outterlink-common/src/protocol.rs` (inline `#[cfg(test)]` module) and `crates/outterlink-common/tests/`

### 1.2 `outterlink-client` Unit Tests

Most client logic requires mocking the TCP connection. Unit tests focus on handle translation and interception logic.

| Test Area | What to Test | Test Count (est.) |
|-----------|-------------|-------------------|
| **Handle tables** | Insert, lookup, remove for each handle type | 12 |
| **Synthetic handle generation** | Handles have correct prefix, are unique, no collisions | 4 |
| **Thread safety** | Concurrent handle operations from multiple threads | 3 |
| **Current context tracking** | Per-thread context set/get | 3 |
| **Connection config** | Env var parsing (`OUTTERLINK_SERVER`), defaults | 4 |
| **Handle type detection** | Correct prefix for context vs device ptr vs module | 3 |

```rust
#[test]
fn test_handle_table_insert_lookup() {
    let table = HandleTables::new();
    let synthetic = table.create_context(42); // server ctx_id = 42
    let server_id = table.lookup_context(synthetic).unwrap();
    assert_eq!(server_id, 42);
}

#[test]
fn test_handle_table_remove() {
    let table = HandleTables::new();
    let synthetic = table.create_context(42);
    table.remove_context(synthetic);
    assert!(table.lookup_context(synthetic).is_none());
}

#[test]
fn test_device_ptr_handles_are_page_aligned() {
    let table = HandleTables::new();
    let ptr1 = table.create_device_ptr(0xAAAA);
    let ptr2 = table.create_device_ptr(0xBBBB);
    assert_eq!(ptr1 % 4096, 0);
    assert_eq!(ptr2 % 4096, 0);
    assert_ne!(ptr1, ptr2);
}

#[test]
fn test_concurrent_handle_access() {
    let table = Arc::new(HandleTables::new());
    let handles: Vec<_> = (0..100).map(|i| {
        let t = table.clone();
        std::thread::spawn(move || t.create_context(i))
    }).collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    // All unique
    let unique: HashSet<_> = results.iter().collect();
    assert_eq!(unique.len(), 100);
}
```

**Location:** `crates/outterlink-client/src/handles.rs` (inline tests) and `crates/outterlink-client/tests/`

### 1.3 `outterlink-server` Unit Tests

Server unit tests focus on the CUDA executor dispatch logic and handle management, using mock CUDA functions.

| Test Area | What to Test | Test Count (est.) |
|-----------|-------------|-------------------|
| **Server handle tables** | Per-client handle isolation, cleanup on disconnect | 6 |
| **Request dispatch** | Correct executor called for each opcode | 5 |
| **Cleanup on disconnect** | All handles freed when client disconnects | 3 |
| **Config parsing** | Port, log level, bind address | 3 |
| **CUDA FFI loading** | Function pointer resolution from libcuda.so | 2 (GPU-only) |

**Location:** `crates/outterlink-server/src/handles.rs`, `crates/outterlink-server/tests/`

### 1.4 `outterlink-cli` Unit Tests

Minimal in PoC phase. CLI is primarily a thin wrapper.

| Test Area | What to Test | Test Count (est.) |
|-----------|-------------|-------------------|
| **Argument parsing** | Valid and invalid CLI arguments | 4 |
| **Config file** | Parse server config from TOML/env | 3 |

---

## 2. Integration Tests

Integration tests verify the client-server protocol over a real TCP connection, but without requiring a real GPU.

### 2.1 Mock GPU Server

Create a mock GPU server that responds to CUDA requests with predetermined values. This runs in CI without GPU hardware.

```rust
// crates/outterlink-server/src/mock.rs (or tests/common/mock_server.rs)

/// A mock GPU server that responds to CUDA calls with fake data.
/// Used for integration testing without a real GPU.
pub struct MockGpuServer {
    listener: TcpListener,
    /// Fake GPU properties
    gpu_name: String,
    gpu_memory: u64,
    device_count: i32,
    driver_version: i32,
    /// Track allocated memory to verify free calls
    allocations: HashMap<u64, u64>,  // dptr -> size
    next_dptr: u64,
    /// Track loaded modules
    modules: HashMap<u64, Vec<u8>>,  // module_id -> image data
    next_module_id: u64,
}

impl MockGpuServer {
    pub fn new(port: u16) -> Self {
        MockGpuServer {
            listener: TcpListener::bind(format!("127.0.0.1:{}", port)).unwrap(),
            gpu_name: "Mock GPU RTX 3090 Ti".to_string(),
            gpu_memory: 24_696_061_952, // 24 GB
            device_count: 1,
            driver_version: 12060, // CUDA 12.6
            allocations: HashMap::new(),
            next_dptr: 0x7F00_0000_0000,
            modules: HashMap::new(),
            next_module_id: 1,
        }
    }

    /// Handle one request, return a response
    pub fn handle_request(&mut self, req: Request) -> Response {
        match req {
            Request::Init { .. } => Response::Init { result: 0 },
            Request::DeviceGetCount => Response::DeviceGetCount {
                result: 0,
                count: self.device_count,
            },
            Request::DeviceGetName { len, .. } => Response::DeviceGetName {
                result: 0,
                name: self.gpu_name[..len.min(self.gpu_name.len() as i32) as usize].to_string(),
            },
            Request::DeviceTotalMem { .. } => Response::DeviceTotalMem {
                result: 0,
                bytes: self.gpu_memory,
            },
            Request::MemAlloc { size } => {
                let dptr = self.next_dptr;
                self.next_dptr += size.max(4096); // page-align
                self.allocations.insert(dptr, size);
                Response::MemAlloc { result: 0, dptr }
            },
            Request::MemFree { dptr } => {
                let result = if self.allocations.remove(&dptr).is_some() { 0 } else { 1 };
                Response::MemFree { result }
            },
            Request::MemcpyHtoD { dptr, size, .. } => {
                // Verify the allocation exists and size fits
                let result = match self.allocations.get(&dptr) {
                    Some(&alloc_size) if size <= alloc_size => 0,
                    _ => 1, // CUDA_ERROR_INVALID_VALUE
                };
                Response::MemcpyHtoD { result }
            },
            Request::MemcpyDtoH { dptr, size } => {
                // Return zeros (mock doesn't compute anything)
                let data = vec![0u8; size as usize];
                Response::MemcpyDtoH { result: 0, data }
            },
            Request::ModuleLoadData { image } => {
                let id = self.next_module_id;
                self.next_module_id += 1;
                self.modules.insert(id, image);
                Response::ModuleLoadData { result: 0, module_id: id }
            },
            Request::LaunchKernel { .. } => {
                // Mock: just return success (no actual computation)
                Response::LaunchKernel { result: 0 }
            },
            // ... other operations
            _ => Response::Error { result: 1 },
        }
    }
}
```

### 2.2 Integration Test Cases

| Test | Description | GPU Required |
|------|------------|--------------|
| **Full PoC flow (mock)** | Connect -> init -> device query -> alloc -> HtoD -> launch -> DtoH -> free -> disconnect. Uses mock server. | No |
| **Handle translation end-to-end** | Alloc multiple buffers, verify synthetic handles are unique and correctly translated | No |
| **Protocol version mismatch** | Client with version X connects to server with version Y. Expect graceful error. | No |
| **Connection drop** | Server drops mid-request. Client should return CUDA error, not panic. | No |
| **Large memory transfer** | Copy 100MB via MemcpyHtoD, verify all bytes arrive at server | No |
| **Multiple clients** | Two clients connect to same mock server simultaneously | No |
| **Rapid alloc/free** | 1000 alloc/free cycles, verify no handle table leaks | No |
| **Module load/unload** | Load PTX, get function, unload, verify cleanup | No |
| **Invalid handle** | Pass unknown handle to MemFree/CtxDestroy, expect error (not crash) | No |

```rust
// tests/integration/test_poc_flow.rs

#[tokio::test]
async fn test_full_poc_flow_with_mock() {
    // Start mock server
    let server = MockGpuServer::start(0).await; // port 0 = random
    let port = server.port();

    // Create client
    let client = OutterLinkClient::connect(&format!("127.0.0.1:{}", port)).await.unwrap();

    // Init
    let result = client.cu_init(0).await;
    assert_eq!(result, CUDA_SUCCESS);

    // Device query
    let count = client.cu_device_get_count().await.unwrap();
    assert_eq!(count, 1);

    let name = client.cu_device_get_name(0, 256).await.unwrap();
    assert!(name.contains("Mock GPU"));

    // Memory
    let dptr = client.cu_mem_alloc(4096).await.unwrap();
    assert_ne!(dptr, 0);

    let data = vec![42u8; 4096];
    let result = client.cu_memcpy_htod(dptr, &data).await;
    assert_eq!(result, CUDA_SUCCESS);

    let result = client.cu_mem_free(dptr).await;
    assert_eq!(result, CUDA_SUCCESS);

    // Cleanup
    server.shutdown().await;
}

#[tokio::test]
async fn test_connection_drop_returns_error() {
    let server = MockGpuServer::start(0).await;
    let port = server.port();
    let client = OutterLinkClient::connect(&format!("127.0.0.1:{}", port)).await.unwrap();

    // Force disconnect
    server.shutdown().await;

    // Next call should return an error, not panic
    let result = client.cu_device_get_count().await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_large_memcpy() {
    let server = MockGpuServer::start(0).await;
    let port = server.port();
    let client = OutterLinkClient::connect(&format!("127.0.0.1:{}", port)).await.unwrap();
    client.cu_init(0).await;

    let size = 100 * 1024 * 1024; // 100 MB
    let dptr = client.cu_mem_alloc(size).await.unwrap();
    let data = vec![0xABu8; size as usize];
    let result = client.cu_memcpy_htod(dptr, &data).await;
    assert_eq!(result, CUDA_SUCCESS);

    server.shutdown().await;
}
```

**Location:** `tests/integration/`

### 2.3 Test Harness

```rust
// tests/common/mod.rs

/// Start a mock server on a random port, return the server handle and port
pub async fn start_mock_server() -> (MockServerHandle, u16) {
    let server = MockGpuServer::start(0).await;
    let port = server.port();
    (server, port)
}

/// Start a real server (requires GPU)
#[cfg(feature = "gpu-tests")]
pub async fn start_real_server() -> (ServerHandle, u16) {
    let server = OutterLinkServer::start(0).await.unwrap();
    let port = server.port();
    (server, port)
}

/// Create a client connected to the given server
pub async fn connect_client(port: u16) -> OutterLinkClient {
    OutterLinkClient::connect(&format!("127.0.0.1:{}", port))
        .await
        .expect("Failed to connect to server")
}
```

---

## 3. End-to-End Tests (Real GPU)

These tests require a machine with a real NVIDIA GPU and CUDA toolkit installed. They run behind a feature flag.

### 3.1 Feature Gating

```toml
# Cargo.toml (workspace root)
[workspace.metadata.test]
# GPU tests only run when explicitly enabled
# cargo test --features gpu-tests
```

```rust
// In test files:
#[cfg(feature = "gpu-tests")]
mod gpu_tests {
    #[test]
    fn test_vector_add_through_outterlink() { ... }
}
```

### 3.2 End-to-End Test Cases

| Test | Description | Setup |
|------|------------|-------|
| **Vector add (PoC test program)** | Run the full `vector_add.cu` program through OutterLink, verify results | Server on GPU machine, client on same or different machine |
| **Device properties match** | Compare `cuDeviceGetName`, `cuDeviceTotalMem`, `cuDeviceGetAttribute` through OutterLink vs direct | Both on GPU machine |
| **Memory integrity** | Alloc, write pattern, read back, verify pattern matches | Server on GPU machine |
| **Multiple alloc/free** | Allocate N buffers, free in random order, verify no corruption | Server on GPU machine |
| **Module load variants** | Load PTX string, load cubin file, verify both work | Server on GPU machine |
| **Kernel launch correctness** | Vector add, matrix multiply, reduction -- verify numerical results | Server on GPU machine |
| **Error propagation** | Invalid device ordinal, alloc more than VRAM, load invalid PTX | Server on GPU machine |
| **Loopback latency** | Measure round-trip time for each call type on loopback | Same machine |

### 3.3 GPU Test Runner Script

```bash
#!/bin/bash
# tests/run_gpu_tests.sh
# Run end-to-end GPU tests

set -euo pipefail

echo "=== OutterLink GPU Test Suite ==="
echo ""

# Check for GPU
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. GPU tests require NVIDIA GPU + drivers."
    exit 1
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Build
echo "Building..."
cargo build --release --features gpu-tests
echo ""

# Run unit tests (no GPU needed)
echo "=== Unit Tests ==="
cargo test --all -- --nocapture
echo ""

# Run GPU integration tests
echo "=== GPU Integration Tests ==="
cargo test --features gpu-tests -- --nocapture --test-threads=1
echo ""

# Run PoC test program
echo "=== PoC End-to-End Test ==="
tests/poc/run_poc.sh loopback
echo ""

echo "=== ALL TESTS PASSED ==="
```

---

## 4. CUDA Mock/Stub Strategy

### 4.1 The Problem

CI servers (GitHub Actions, etc.) do not have NVIDIA GPUs. The client library links against `libcuda.so` at runtime. The server needs real CUDA to execute calls. We need a way to test without any GPU.

### 4.2 The Solution: Three Layers

```
Layer 1: Protocol-only tests (no CUDA at all)
  -> Test serialization, wire format, handle tables
  -> Works everywhere, no special setup

Layer 2: Mock GPU server (fake CUDA responses)
  -> Test client-server communication over TCP
  -> Mock server returns predetermined values
  -> No libcuda.so needed anywhere

Layer 3: Real GPU tests (feature-gated)
  -> Test with actual CUDA execution
  -> Only on machines with GPU
  -> Feature flag: --features gpu-tests
```

### 4.3 CUDA Stub Library for Client Testing

For testing the LD_PRELOAD interception mechanism itself (not the protocol), create a stub `libcuda.so` that provides the symbols but does nothing.

```c
// tests/stubs/libcuda_stub.c
// Compile: gcc -shared -o libcuda.so.1 libcuda_stub.c -ldl

#include <stddef.h>

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef unsigned long long CUdeviceptr;

#define CUDA_SUCCESS 0

CUresult cuInit(unsigned int flags) { return CUDA_SUCCESS; }
CUresult cuDriverGetVersion(int *v) { *v = 12060; return CUDA_SUCCESS; }
CUresult cuDeviceGetCount(int *c) { *c = 1; return CUDA_SUCCESS; }
CUresult cuDeviceGet(CUdevice *d, int ordinal) { *d = 0; return CUDA_SUCCESS; }
CUresult cuDeviceGetName(char *n, int len, CUdevice d) {
    const char *name = "Stub GPU";
    int i;
    for (i = 0; i < len - 1 && name[i]; i++) n[i] = name[i];
    n[i] = 0;
    return CUDA_SUCCESS;
}
CUresult cuDeviceTotalMem_v2(size_t *b, CUdevice d) { *b = 24696061952ULL; return CUDA_SUCCESS; }
CUresult cuDeviceGetAttribute(int *v, int attr, CUdevice d) { *v = 0; return CUDA_SUCCESS; }

// ... minimal stubs for all PoC functions

CUresult cuGetProcAddress(const char *symbol, void **pfn,
                          int cudaVersion, unsigned long long flags) {
    // Return NULL for all -- forces fallback to dlsym
    *pfn = NULL;
    return CUDA_SUCCESS;
}
```

### 4.4 Server CUDA Abstraction Trait

To enable mock testing of the server's CUDA execution layer:

```rust
// crates/outterlink-server/src/cuda_backend.rs

/// Trait abstracting CUDA operations. Allows mock implementation for testing.
pub trait CudaBackend: Send + Sync {
    fn init(&self, flags: u32) -> CUresult;
    fn device_get_count(&self) -> Result<i32, CUresult>;
    fn device_get_name(&self, device: i32, max_len: i32) -> Result<String, CUresult>;
    fn device_total_mem(&self, device: i32) -> Result<u64, CUresult>;
    fn device_get_attribute(&self, attrib: i32, device: i32) -> Result<i32, CUresult>;
    fn ctx_create(&self, flags: u32, device: i32) -> Result<u64, CUresult>;
    fn ctx_destroy(&self, ctx_id: u64) -> CUresult;
    fn mem_alloc(&self, size: u64) -> Result<u64, CUresult>;
    fn mem_free(&self, dptr: u64) -> CUresult;
    fn memcpy_htod(&self, dptr: u64, data: &[u8]) -> CUresult;
    fn memcpy_dtoh(&self, dptr: u64, size: u64) -> Result<Vec<u8>, CUresult>;
    fn module_load_data(&self, image: &[u8]) -> Result<u64, CUresult>;
    fn module_unload(&self, module_id: u64) -> CUresult;
    fn module_get_function(&self, module_id: u64, name: &str) -> Result<u64, CUresult>;
    fn launch_kernel(&self, function_id: u64, grid: [u32; 3], block: [u32; 3],
                     shared_mem: u32, params: &[u8]) -> CUresult;
    fn cleanup_client(&self, client_id: &[u8; 16]);
}

/// Real CUDA implementation (links to libcuda.so at runtime)
pub struct RealCudaBackend { /* ... */ }

/// Mock implementation for testing
pub struct MockCudaBackend { /* ... */ }
```

This trait boundary is the key testing seam. The server's connection handler is generic over `CudaBackend`, allowing full testing with `MockCudaBackend` in CI.

---

## 5. Test Infrastructure

### 5.1 Test Helpers and Fixtures

**Location:** `tests/common/` (shared across integration tests)

| Helper | Purpose |
|--------|---------|
| `mock_server` | Start/stop mock GPU server on random port |
| `connect_client` | Create client connection to given port |
| `random_data(size)` | Generate random byte vector for memcpy tests |
| `assert_cuda_success(result)` | Assert CUresult == 0 with readable error name |
| `with_server_client(f)` | Setup mock server + client, run closure, teardown |
| `compare_gpu_property(prop, direct, proxied)` | Compare direct vs OutterLink GPU property query |

### 5.2 Test Configuration

```toml
# tests/test_config.toml
[mock]
default_gpu_name = "Mock GPU RTX 3090 Ti"
default_gpu_memory = 24696061952
default_driver_version = 12060
default_device_count = 1

[gpu]
# Used for real GPU tests
server_port = 9370
test_data_sizes = [1024, 65536, 1048576, 104857600]  # 1KB, 64KB, 1MB, 100MB
```

### 5.3 CI Configuration

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  # Standard tests - no GPU
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt
      - run: cargo fmt --all -- --check
      - run: cargo clippy --all-targets -- -D warnings
      - run: cargo test --all  # Unit + mock integration tests
      - run: cargo build --all

  # GPU tests - self-hosted runner with NVIDIA GPU
  gpu-test:
    runs-on: [self-hosted, gpu]
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    needs: test
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: nvidia-smi  # Verify GPU available
      - run: cargo test --all --features gpu-tests -- --test-threads=1
      - run: tests/poc/run_poc.sh loopback
```

### 5.4 Test Directory Structure

```
tests/
├── common/
│   ├── mod.rs              # Shared test utilities
│   ├── mock_server.rs      # Mock GPU server implementation
│   └── fixtures.rs         # Test data generators
├── integration/
│   ├── test_protocol.rs    # Protocol-level tests (encode/decode over TCP)
│   ├── test_poc_flow.rs    # Full PoC flow with mock server
│   ├── test_handle_lifecycle.rs  # Handle create/use/destroy cycles
│   ├── test_error_handling.rs    # Error conditions and edge cases
│   └── test_concurrency.rs      # Multiple clients, thread safety
├── e2e/
│   ├── test_vector_add.rs  # GPU: vector add end-to-end
│   ├── test_device_query.rs # GPU: verify device properties match
│   ├── test_memory.rs      # GPU: memory integrity tests
│   └── test_latency.rs     # GPU: measure per-call latency
├── poc/
│   ├── vector_add.cu       # PoC test program (C/CUDA)
│   └── run_poc.sh          # PoC demo script
└── stubs/
    └── libcuda_stub.c      # Minimal libcuda.so stub for CI
```

---

## 6. Regression Testing

### 6.1 CUDA Version Compatibility

OutterLink must handle CUDA Driver API changes across versions. Regression strategy:

| Risk | Test Approach |
|------|--------------|
| `cuGetProcAddress` signature change (3-arg vs 4-arg) | Test both variants at compile time. Runtime detection via `dlsym` version check. |
| New `_v3` / `_v4` function variants | Version-indexed function table. Test that unknown versions fall through to real implementation. |
| `CUresult` new error codes | Map unknown error codes to `CUDA_ERROR_UNKNOWN`. Test with synthetic error codes. |
| Changed struct layouts | Pin struct sizes in compile-time assertions. |

### 6.2 CUDA Version Matrix Testing

```
CUDA 12.2 - Minimum supported (cuGetProcAddress stable)
CUDA 12.4 - Mid-range
CUDA 12.6 - Current stable
CUDA 13.x - Future (when available)
```

For CI, we test against the stub (version-independent). For GPU tests, we test against whatever version is installed.

The server logs the CUDA version on startup. The client logs the version it was compiled against. If they differ, a warning is logged.

### 6.3 Protocol Version Regression

```rust
// In protocol.rs
pub const PROTOCOL_VERSION: u32 = 1;

// In handshake:
// - If client.version != server.version, return INCOMPATIBLE error
// - Future: support version negotiation (pick min(client, server))
```

Test: Client with protocol version 1 connects to server with protocol version 2. Expect a clear error message, not a crash.

### 6.4 Regression Test Automation

Every bug fix must include a regression test that reproduces the bug before the fix and passes after.

```
Convention:
  - File: tests/regression/test_issue_NNN.rs
  - Naming: #[test] fn regression_issue_NNN_description() { ... }
  - Comment: Link to issue + description of the bug
```

---

## Test Execution Summary

| Test Category | Command | GPU Required | Runs in CI |
|---------------|---------|--------------|-----------|
| Unit tests (all crates) | `cargo test --all` | No | Yes |
| Integration tests (mock) | `cargo test --test '*'` | No | Yes |
| End-to-end tests (GPU) | `cargo test --features gpu-tests` | Yes | Self-hosted only |
| PoC demo | `tests/poc/run_poc.sh loopback` | Yes | Self-hosted only |
| Clippy | `cargo clippy --all-targets -- -D warnings` | No | Yes |
| Format check | `cargo fmt --all -- --check` | No | Yes |

## Risks

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Mock server behavior drifts from real CUDA | MEDIUM | Integration tests on GPU hardware catch this; mock only tests protocol |
| GPU tests flaky due to driver state | LOW | Run GPU tests with `--test-threads=1`; clean state between tests |
| Self-hosted runner GPU contention | LOW | Use per-PR locking or queue system |
| Stub libcuda.so diverges from real API | LOW | Generated from CUDA header files; update on CUDA version bump |

## Related Documents

- [P5: PoC Plan](P5-poc-plan.md)
- [P12: Benchmarking Plan](P12-benchmarking-plan.md)
- [R3: CUDA Interception Strategies](../research/R3-cuda-interception.md)
- [Final Pre-Plan](../pre-planning/02-FINAL-PREPLAN.md)

## Open Questions

- [ ] Should we use `proptest` or `quickcheck` for property-based testing of serialization?
- [ ] Should GPU tests run on every PR or only on main/release branches?
- [ ] Do we need a separate test for the LD_PRELOAD mechanism itself (loading the .so, symbol interception)?
- [ ] Should the mock server support configurable failure injection (e.g., return CUDA_ERROR_OUT_OF_MEMORY on Nth alloc)?
- [ ] How to test the `dlsym` override without conflicting with the test runner's own dlsym usage?
