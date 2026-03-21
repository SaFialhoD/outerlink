# R8: Kernel Parameter Introspection

**Created:** 2026-03-21
**Last Updated:** 2026-03-21
**Status:** Draft
**Priority:** HIGH

## Purpose

Determine how OuterLink can extract kernel parameter sizes from CUDA functions at runtime, enabling serialization of `cuLaunchKernel` arguments over the network.

## Problem Statement

When intercepting `cuLaunchKernel`, we receive:
- `CUfunction f` — opaque handle to the kernel
- `void **kernelParams` — array of pointers to each argument
- `void **extra` — alternative: packed parameter buffer

The `kernelParams` path gives us pointers but **not the sizes** of what they point to. Without sizes, we cannot serialize the parameters for network transport. The current interpose.c hook passes NULL for params, which breaks any kernel that takes arguments.

## Approach 1: `cuFuncGetParamInfo` (CUDA Driver API)

### How It Works

NVIDIA provides a direct API to query parameter layout:

```c
CUresult cuFuncGetParamInfo(
    CUfunction func,
    size_t     paramIndex,
    size_t*    paramOffset,
    size_t*    paramSize
);
```

Iterate `paramIndex` from 0 upward. Each call returns the offset and size of that parameter in the device-side layout. There is **no API to query total parameter count** — iterate until the call returns `CUDA_ERROR_INVALID_VALUE`.

### CUDA Version Required

**Best estimate: introduced in CUDA 12.3 or 12.4** (late 2023 / early 2024). Minimum driver: likely R550+.

### Works For

Both PTX and cubin. Operates on the loaded `CUfunction` handle.

### Complexity: Low (~20 lines of C)

### Reliability: High for CUDA 12.3+

### Verdict: **Plan A.** Cleanest, most reliable approach.

## Approach 2: PTX Text Parsing

### How It Works

Intercept `cuModuleLoadData` with PTX source, parse `.entry` declarations:

| PTX Type | Size (bytes) |
|----------|-------------|
| `.u8`, `.s8`, `.b8` | 1 |
| `.u16`, `.s16`, `.f16` | 2 |
| `.u32`, `.s32`, `.f32` | 4 |
| `.u64`, `.s64`, `.f64` | 8 |
| `.b128` | 16 |

### Works For: PTX only (not cubin/fatbin)

### Complexity: Medium (~100-200 lines of C)

### Verdict: **Plan B fallback** for pre-12.3 CUDA.

## Approach 3: Cubin/ELF Parsing

Reverse-engineered `.nv.info` section format. Undocumented, fragile across architectures.

### Verdict: **Not recommended.** Too fragile, too complex.

## Approach 4: Binary Utilities (`cuobjdump`, `nvdisasm`)

Shelling out at runtime. Too slow.

### Verdict: **Development/debugging only.**

## Approach 5: `cuLaunchKernelEx` / Newer APIs

No new parameter introspection. Same `kernelParams`/`extra` interface.

### Verdict: **No help.**

## Approach 6: `CU_LAUNCH_PARAM_BUFFER_POINTER` (the `extra` Path)

### How It Works

When `extra != NULL`, the caller provides a complete packed buffer AND its total size:

```c
void* extra[] = {
    CU_LAUNCH_PARAM_BUFFER_POINTER, packed_buffer,
    CU_LAUNCH_PARAM_BUFFER_SIZE,    &buffer_size,
    CU_LAUNCH_PARAM_END
};
```

Serialize the entire buffer as-is. **No per-parameter introspection needed.**

### CUDA Version Required: 4.0+ (universal)

### Complexity: Low

### Verdict: **Critical optimization.** Handle this FIRST — trivial and gives us the buffer for free. Many ML frameworks (TVM, Triton, parts of PyTorch) use this path.

## Comparison Matrix

| Approach | Complexity | Reliability | CUDA Version | PTX | Cubin | Runtime |
|----------|-----------|------------|-------------|-----|-------|---------|
| 1. `cuFuncGetParamInfo` | Low | High | 12.3+ | Yes | Yes | Yes |
| 2. PTX Parsing | Medium | High (PTX) | Any | Yes | No | Yes |
| 3. Cubin/ELF Parsing | High | Medium-Low | Any | No | Yes | Possible |
| 4. Binary Utilities | Medium-High | High (offline) | Any | Yes | Yes | No |
| 5. Newer Launch APIs | N/A | N/A | N/A | N/A | N/A | N/A |
| 6. `extra` Buffer Path | Low | High | 4.0+ | Yes | Yes | Yes |

## Recommended Strategy

### Implementation Order

1. **First**: Handle `extra` path in `cuLaunchKernel` hook (trivial, high impact)
2. **Second**: Implement `cuFuncGetParamInfo` path with caching for `kernelParams`
3. **Third** (if needed): PTX parsing fallback for CUDA < 12.3
4. **Never**: Cubin ELF parsing or shelling out to binary utilities

### Caching Strategy

```
HashMap<CUfunction, Vec<ParamInfo>>
    where ParamInfo = { offset: usize, size: usize }
```

Invalidate only when the module is unloaded. Typical apps: tens to hundreds of unique kernels.

## Related Documents

- [R3: CUDA Interception Strategies](R3-cuda-interception.md)
- [CONSOLIDATION: All Research](CONSOLIDATION-all-research.md)
- [Project Vision](../../docs/architecture/00-project-vision.md)

## Open Questions

1. **Exact CUDA version for `cuFuncGetParamInfo`**: Confirm 12.3 vs 12.4 by resolving the symbol from libcuda.so on dev machines.
2. **What percentage of real workloads use `extra` vs `kernelParams`?** PyTorch, TensorFlow, Triton likely use `extra` for JIT-compiled kernels. cuBLAS/cuDNN may use `kernelParams` internally.
3. **Struct-by-value parameters**: Does `cuFuncGetParamInfo` return the full struct size for `.param .align N .b8 name[SIZE]` parameters?
4. **cuGraphs and kernel nodes**: Does OuterLink need to intercept graph-based launches separately?
5. **Minimum CUDA version policy**: Require CUDA 12.3+ (simplifies everything) or support older with PTX fallback?
