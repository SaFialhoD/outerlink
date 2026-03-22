# Implementation Report: Real GPU Hardware Tests
Generated: 2026-03-20

## Task
Create a feature-gated integration test file that exercises actual CUDA operations
on real GPU hardware through the full OuterLink stack (client TCP -> server handler ->
CudaGpuBackend -> NVIDIA driver -> RTX 3090 Ti).

## TDD Summary

### Tests Written
All 9 tests in `crates/outerlink-server/tests/real_gpu_test.rs`:

1. `test_real_device_get_count` - Queries real device count (got 1)
2. `test_real_driver_version` - Queries CUDA driver version (got 13010)
3. `test_real_device_get_name` - Gets GPU name ("NVIDIA GeForce RTX 3090 Ti")
4. `test_real_device_total_mem` - Gets VRAM size (23.99 GiB)
5. `test_real_device_attributes` - Gets compute capability (8.6)
6. `test_real_mem_alloc_free` - Allocates 1 MiB, verifies non-zero ptr, frees
7. `test_real_memcpy_roundtrip` - Writes 4 KiB pattern to GPU, reads back, byte-for-byte match
8. `test_real_ctx_create_destroy` - Creates/destroys real CUDA context
9. `test_real_mem_get_info` - Queries free/total VRAM, cross-checks with DeviceTotalMem

### Implementation
- `crates/outerlink-server/tests/real_gpu_test.rs` - NEW: 9 real GPU tests
- `crates/outerlink-server/Cargo.toml` - Added `real-gpu-test` feature

## Test Results
- Workspace (no feature): 190 tests passed, 0 failed
- Real GPU tests (with feature): 9 tests passed, 0 failed
- Total: 199 tests all green

## Changes Made
1. Added `[features] real-gpu-test = []` to `crates/outerlink-server/Cargo.toml`
2. Created `crates/outerlink-server/tests/real_gpu_test.rs` with:
   - `#![cfg(feature = "real-gpu-test")]` gate at file level
   - `try_create_backend()` helper that returns None if no CUDA driver (graceful skip)
   - `require_gpu!()` macro for early-return skip pattern
   - Same helper functions as integration.rs (response_result, assert_success, spawn_real_server, etc.)
   - Memory tests (alloc, memcpy, mem_get_info) create a CUDA context first since real CUDA requires it
   - All tests use `eprintln!` to print hardware values with `--nocapture`

## Notes
- Memory operations (MemAlloc, MemFree, MemcpyHtoD, MemcpyDtoH, MemGetInfo) require an
  active CUDA context on real hardware. The initial run failed 3 tests with `InvalidContext`.
  Fixed by adding CtxCreate at the start of those tests.
- The stub backend does not enforce this requirement, which is why the existing integration
  tests work without context creation for memory ops.
- Run command: `cargo test -p outerlink-server --features real-gpu-test --test real_gpu_test -- --nocapture`

## Hardware Observed
- GPU: NVIDIA GeForce RTX 3090 Ti
- VRAM: 25,756,565,504 bytes (23.99 GiB)
- CUDA Driver: 13010
- Compute Capability: 8.6
- VRAM Free at test time: ~21.98 GiB
