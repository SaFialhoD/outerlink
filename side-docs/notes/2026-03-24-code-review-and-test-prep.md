# Code Review & Test Preparation

**Date:** 2026-03-24
**Status:** Review
**Purpose:** Full codebase review + inventory of what to test tonight.

---

## Codebase Summary

| Crate | Source Lines | Test Count | Role |
|---|---|---|---|
| outerlink-common | 1,544 | 38 (unit) | Wire protocol, handle maps, transport trait, TCP impl, retry |
| outerlink-server | 17,639 | 68 (unit+integration) | GPU backend trait, StubGpuBackend, CudaGpuBackend, handler, server loop, session, CudaWorker |
| outerlink-client | 10,932 | 369 (unit) | FFI interception layer, callback registry, client connection |
| outerlink-cli | 690 | 11 (integration) | CLI management tool (list, status, bench) |
| **Total** | **30,805** | **486** | |

## Architecture Assessment

The architecture is solid and well-layered:

```
C app -> LD_PRELOAD (interpose.c) -> FFI (ffi.rs) -> OuterLinkClient (lib.rs)
  -> TCP transport -> Server accept loop -> Handler dispatch -> GpuBackend
```

### What's Done Right

1. **Clean trait abstraction** - `GpuBackend` trait with StubGpuBackend + CudaGpuBackend. Tests run against the stub, real hardware uses the CUDA backend. This is textbook.

2. **Handle translation** - Synthetic handle prefixes (0x0C00... through 0x1700...) for different resource types. Bidirectional DashMap lookups. Zero-collision with real GPU addresses. `translate_device_ptrs_in_params` correctly scans kernel parameter buffers and replaces synthetic handles.

3. **Wire protocol** - 22-byte binary header (magic, version, flags, request_id, msg_type, payload_len). Big-endian header, little-endian payload. 170+ message types covering the full CUDA Driver API. Proper validation on recv (magic, version, payload size).

4. **Session management** - Per-connection `ConnectionSession` tracks ALL allocated resources (mem, host mem, contexts, modules, streams, events, mem pools, link states, libraries, kernels, graphs, graph execs). Cleanup on disconnect releases everything. This prevents GPU resource leaks.

5. **CudaWorker thread** - Solves the CUDA context thread-locality problem correctly. Dedicated OS thread per connection with mpsc channel. The worker calls `cuCtxSetCurrent` before each operation.

6. **Callback channel** - Two-connection design: main connection for request/response, second TCP connection for async `CallbackReady` notifications. Client-side `CallbackRegistry` is re-entrant safe (mutex released before invoking callback).

7. **Graceful shutdown** - `tokio::sync::watch` channel for shutdown signal. Drain timeout for in-flight connections. Backend cleanup after all connections finish.

8. **Retry/reconnect** - Configurable exponential backoff with capped delays. Transient vs permanent error classification.

### Issues Found

#### Warnings (6 total, all in outerlink-server)

1. `StubHostAlloc.data_len` - never read
2. `StubFunction.name` - never read
3. `StubEvent.flags` - never read
4. `CaptureState.mode` - never read
5. Two more dead field warnings

These are struct fields stored for future use. **Not bugs, but should be cleaned up** - either mark `#[allow(dead_code)]` with a TODO comment explaining future use, or remove if truly unnecessary.

#### Potential Concerns (not bugs, but worth noting)

1. **Client `OnceLock` singleton** - `OuterLinkClient` in `ffi.rs` uses a global `OnceLock<OuterLinkClient>`. The server address comes from `OUTERLINK_SERVER` env var. No way to reconfigure at runtime. Fine for Phase 1 but will need revisiting for multi-server pools.

2. **`block_on` in FFI** - `OuterLinkClient.connect()` and `send_request()` call `runtime.block_on()`. This works because the FFI functions are called from the application's thread (not a tokio context). But if someone ever calls these from within an async context, it will panic. The integration tests correctly work around this by using raw `TcpTransportConnection`.

3. **Stub occupancy constants** - Both `ffi.rs` and `gpu_backend.rs` define `STUB_NUM_SMS`, `STUB_MAX_THREADS_PER_SM`, `STUB_MAX_BLOCKS_PER_SM`. These are duplicated rather than shared from `outerlink-common`. Low priority.

4. **`DEVICE_ALLOC_SIZES` global in ffi.rs** - Tracks allocation sizes for stub-mode `cuMemGetAddressRange`. Uses `OnceLock<Mutex<HashMap>>`. This only matters in stub mode so it's fine, but it's process-global state.

5. **No timeout on server-side recv** - `handle_connection_loop` blocks on `conn.recv_message()` indefinitely. A slow or dead client holds a connection open until shutdown. Not critical for Phase 1 but worth adding idle timeouts later.

6. **Session cleanup order** - `ConnectionSession::cleanup()` frees resources in a fixed order. If a free fails (e.g., context already destroyed), it logs and continues. This is the right behavior for cleanup, but test coverage of failure paths during cleanup would be valuable.

---

## Test Inventory

### What's Already Tested

**outerlink-common (38 tests)**
- Protocol header roundtrip serialization
- Invalid magic/version rejection
- MessageType roundtrip for all categories
- HandleMap insert, lookup, duplicate insert, remove
- HandleStore prefix separation
- is_synthetic_handle detection
- translate_device_ptrs_in_params (multiple scenarios)
- TCP transport: message roundtrip, bulk roundtrip, invalid magic rejection, oversized payload rejection, close detection, remote_addr, sequential messages, payload length mismatch
- RetryConfig: defaults, delays, exponential backoff, capping, no-retry mode
- Error retryability classification

**outerlink-server integration (34 tests)**
- E2E device queries (count, name, version, attribute, total_mem, uuid)
- Memory alloc/free/copy (HtoD, DtoH, DtoD, memset)
- Context lifecycle (create, destroy, set_current, get_current, get_device, synchronize)
- Module load/unload/get_function/get_global
- Stream lifecycle + operations
- Event lifecycle + record/synchronize/elapsed_time
- Multi-step workflows (alloc -> write -> read -> verify -> free)
- Primary context operations
- Invalid input handling

**outerlink-server unit tests in cuda_backend.rs**
- `map_cuda_result` helper
- `require_fn` helper
- VRAM safety margin calculation
- `deserialize_kernel_params` (empty, single, multi, truncated)

**outerlink-server shutdown tests (5)**
- Graceful shutdown with no connections
- Shutdown with active connection
- Shutdown during request processing
- Multiple connections during shutdown
- Immediate shutdown signal

**outerlink-server cuda_thread tests (10)**
- Worker thread dispatches correctly
- Sequential request ordering
- Worker cleanup on drop

**outerlink-cli tests (11)**
- cmd_list against stub server
- cmd_status against stub server
- cmd_bench against stub server
- JSON output formatting
- Error handling (connection refused)

**outerlink-client callback tests (14)**
- Register returns unique IDs
- has_pending tracking
- Fire removes entry
- Fire invokes stream callback with correct status
- Fire invokes host function
- Zero fn_ptr doesn't crash
- Multi-callback same/different stream
- wait_all_completed: immediate, blocks-until-fired, timeout
- Concurrent register+fire stress test

**outerlink-client FFI tests (343 tests)**
- This is the largest test suite - covers all FFI function stubs

### What's NOT Tested Yet (Gaps)

1. **Real GPU path** - `real_gpu_test.rs` exists (11 tests) but requires `--features real_gpu` or running with `--real-gpu` flag. These need an actual NVIDIA GPU to run.

2. **CudaGpuBackend integration** - The 2,619-line `cuda_backend.rs` loads `libcuda.so` dynamically. Unit tests exist for helpers but no integration test exercises the full driver API flow on real hardware.

3. **Multi-client concurrent connections** - No test exercises multiple simultaneous clients hitting the same server. The server supports it (JoinSet of connection tasks), but it's untested.

4. **Callback channel E2E** - The callback registry has unit tests, but no integration test exercises the full flow: client registers callback -> server fires it -> notification arrives on callback channel -> callback invoked.

5. **Reconnection** - `RetryConfig` is tested, but no integration test actually drops a connection and exercises the reconnect loop.

6. **Transport benchmarks** - `bench_transport.rs` exists but is gated behind feature flags.

---

## Test Plan for Tonight

### Phase 1: Verify Everything Compiles and Existing Tests Pass

```bash
# Check compilation (already done - clean with 6 warnings)
cargo check --workspace

# Run all tests that don't need a GPU
cargo test --workspace --test-threads=1
```

### Phase 2: If You Have a GPU Available

```bash
# Start the server with real GPU
cargo run --bin outerlink-server -- --real-gpu -v

# In another terminal, run the CLI commands
cargo run --bin outerlink -- status -s localhost:14833
cargo run --bin outerlink -- list -s localhost:14833
cargo run --bin outerlink -- bench -s localhost:14833

# Run the real GPU tests (if the feature flag exists)
cargo test -p outerlink-server --test real_gpu_test
```

### Phase 3: Manual Integration Smoke Test

1. Start server: `cargo run --bin outerlink-server -- -v`
2. In another terminal: `cargo run --bin outerlink -- status`
3. Check output: should show 1 GPU (stub), driver version 12040, memory 24576 MiB
4. Run bench: `cargo run --bin outerlink -- bench`
5. Check: should show transfer times for 5 size tiers

### Key Things to Watch For

- **TCP port conflicts** - Tests use random ports (`127.0.0.1:0`) so they shouldn't conflict, but `--test-threads=1` is recommended for the server tests.
- **Timeout on Windows** - The tests were likely developed on Linux. TCP behavior on Windows can differ slightly (e.g., RST vs FIN timing). Watch for flaky connection close tests.
- **The 343 FFI tests** - These run entirely in-process (no network). Should be fast and reliable.

---

## Related Documents

- System architecture: `docs/architecture/00-project-vision.md`
- Testing guide: `docs/guides/testing-guide.md` (if it exists)
- Phase plans: `planning/phases/`
