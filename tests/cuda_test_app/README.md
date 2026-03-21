# CUDA Driver API Test Application

Created: 2026-03-21
Status: Draft

## Purpose

Minimal C program that exercises the CUDA Driver API for testing OuterLink's LD_PRELOAD interception layer. Uses ONLY the Driver API (cuXxx), never the Runtime API (cudaXxx).

## API Coverage

| Step | Function | What It Does |
|------|----------|-------------|
| 1 | `cuInit` | Initialize the CUDA driver |
| 2 | `cuDeviceGetCount` | Enumerate available GPUs |
| 3 | `cuDeviceGet`, `cuDeviceGetName` | Select device 0 |
| 4 | `cuCtxCreate_v2` | Create a CUDA context |
| 5 | `cuMemAlloc_v2` (x3) | Allocate device buffers A, B, C |
| 6 | `cuMemcpyHtoD_v2` (x2) | Upload host vectors A and B |
| 7 | `cuModuleLoadData` | Load embedded PTX kernel |
| 8 | `cuModuleGetFunction` | Get `vector_add` entry point |
| 9 | `cuLaunchKernel` | Launch vector_add(A, B, C, N) |
| 10 | `cuCtxSynchronize` | Wait for kernel completion |
| 11 | `cuMemcpyDtoH_v2` | Download result vector C |
| 12 | `cuMemFree_v2`, `cuModuleUnload`, `cuCtxDestroy_v2` | Cleanup |

## Build (Native)

Requires a CUDA toolkit installation with `libcuda.so` (the driver library).

```bash
make native
./test_vector_add
```

Override CUDA path if needed:

```bash
make native CUDA_PATH=/opt/cuda-12.4
```

## Run with OuterLink (LD_PRELOAD)

Start the OuterLink server first, then:

```bash
# Build (links against real libcuda for symbol resolution)
make native

# Run with interception
make interpose OUTERLINK_LIB=../../target/release/libouterlink_client.so \
               OUTERLINK_SERVER=192.168.1.100:9990
```

Or manually:

```bash
OUTERLINK_SERVER=192.168.1.100:9990 \
LD_PRELOAD=../../target/release/libouterlink_client.so \
./test_vector_add
```

## Expected Output

```
=== OuterLink CUDA Driver API Test: Vector Add ===

[1/12] cuInit
[2/12] cuDeviceGetCount
        Found 1 device(s)
[3/12] cuDeviceGet (ordinal=0)
        Device: NVIDIA GeForce RTX 3090 Ti
[4/12] cuCtxCreate_v2
[5/12] cuMemAlloc_v2 (3 buffers, 4096 bytes each)
[6/12] cuMemcpyHtoD_v2 (A and B)
[7/12] cuModuleLoadData (embedded PTX, 734 bytes)
[8/12] cuModuleGetFunction ("vector_add")
[9/12] cuLaunchKernel (grid=4, block=256, params=4)
[10/12] cuCtxSynchronize
[11/12] cuMemcpyDtoH_v2 (C)

--- Verification ---
  PASS: All 1024 elements correct.
  Sample: C[0]=0.0, C[1]=3.0, C[1023]=3069.0

[12/12] Cleanup

=== Test complete (exit code: 0) ===
```

The computation is: `C[i] = A[i] + B[i]` where `A[i] = i` and `B[i] = 2*i`, so `C[i] = 3*i`.

## PTX Kernel

The embedded PTX targets `sm_80` (Ampere) with `.version 7.0`. It is compatible with:
- RTX 3090 / 3090 Ti (sm_86, backward compatible with sm_80 PTX)
- A100 (sm_80)
- RTX 4090 (sm_89, backward compatible)

The CUDA driver JIT-compiles PTX to the actual GPU architecture at load time, so the same PTX works on any GPU with compute capability >= 8.0.

## Related Documents

- `crates/outerlink-client/csrc/interpose.c` -- LD_PRELOAD interception layer
- `crates/outerlink-client/csrc/interpose.h` -- Hook declarations and Rust FFI
- `cuda-stubs/cuda.h` -- CUDA Driver API type stubs

## Open Questions

- The current `hook_cuLaunchKernel` in interpose.c passes NULL for kernel params because param sizes cannot be inferred from the Driver API signature alone. Phase 2 will add PTX metadata introspection to resolve this. Until then, kernel launches under interposition will not forward parameter data.
