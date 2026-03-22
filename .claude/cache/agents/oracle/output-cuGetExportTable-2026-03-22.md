# Research Report: cuGetExportTable for OuterLink
Generated: 2026-03-22

## Summary

cuGetExportTable is an undocumented CUDA driver API function that returns internal function pointer tables (vtables) identified by UUIDs. Introduced around CUDA 3.0, it is called extensively by NVIDIA libraries (cuBLAS, cuDNN, cuFFT, NCCL) during initialization. For OuterLink, the correct strategy is a hybrid local-forward approach: passthrough to the real local libcuda.so for most tables, with graceful fallback for edge cases.

## Questions Answered

### Q1: What is cuGetExportTable? What does it do?

**Answer:** A private, undocumented CUDA driver API function:

    CUresult cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId);

It accepts a 16-byte UUID identifying a specific internal function table, returns a pointer to a struct of function pointers. These provide access to internal driver functionality NOT in the public API -- context management, TLS, profiling hooks, undocumented memory operations. The CUDA runtime (libcudart) calls it during init; cuBLAS/cuDNN call through cudaGetExportTable indirectly.

**Confidence:** High

### Q2: What UUIDs do cuDNN and cuBLAS request?

**Answer:** Not publicly documented. Reverse-engineered primarily in ZLUDA (zluda/src/impl/export_table.rs). Known categories:

1. **Context Local Storage Interface** (CONTEXT_LOCAL_STORAGE_INTERFACE_V0301) -- per-context thread-local state. Most critical.
2. **Tools Runtime Callback Hooks** -- CUPTI/profiling hooks into runtime API calls.
3. **Tools TLS** -- thread-local storage for profiling infrastructure.
4. **cudart Interface** -- internal libcudart-to-libcuda interface (the dark API).

cuBLAS/cuDNN go through cudaGetExportTable. Table entry counts change across CUDA versions (size went from 2 to 14 between CUDA 10.x and 11.2).

**Confidence:** Medium (reverse-engineered; new versions may add UUIDs)

### Q3: What does the returned table contain?

**Answer:** A C struct of function pointers (vtable). Contents vary by UUID and CUDA version:

- First entry is typically a size/version field
- Subsequent entries are function pointers to internal driver ops
- Pointers are direct addresses into libcuda.so memory -- actual callable code pointers

**Critical for OuterLink: these pointers cannot be forwarded across a network.** A function pointer valid in the server process is meaningless on the client machine.

**Confidence:** High

### Q4: How have other projects handled this?

| Project | Approach | Outcome |
|---------|----------|---------|
| HAMi-core | Passthrough to real driver. Skips pre-init for cuGetExportTable. | Works -- same machine as GPU. |
| ZLUDA | Full re-implementation via reverse engineering. | Labor-intensive. Version-sensitive. |
| Cricket | Runtime API interception (avoids problem entirely). | Cleanest for remote. Requires -cudart shared. |
| rCUDA | Initially unsupported, later compile-time workaround. | Fragile. Abandoned at CUDA 9.0. |
| gVisor | ioctl-level proxy; real libcuda handles it. | Not applicable to remote GPU. |
| Ocelot/gdev | Not implemented (commented out). | Unsolved. |

**Confidence:** High

## Detailed Findings

### Finding 1: The Function Pointer Problem

cuGetExportTable returns a vtable of raw function pointers. Callers call through these directly:

    const void *table;
    cuGetExportTable(&table, &some_uuid);
    ((SomeVtable*)table)->internal_alloc(args...);

Cannot return a dummy pointer or NULL -- the caller WILL dereference and call through it.

### Finding 2: OuterLink Architecture Advantage

OuterLink uses LD_PRELOAD to intercept libcuda.so. The client-side interposition library runs in the same process as the application and CUDA runtime. If the real libcuda.so exists on the client (even without a GPU), we can dlopen it and forward cuGetExportTable calls. Returned function pointers are valid in-process.

### Finding 3: Cricket Lesson

The Cricket paper states: a virtualization layer at the driver API level does not allow the use of the original runtime API on top of it because of cuGetExportTable. Cricket intercepted at runtime API level instead. OuterLink intercepts at driver level but can use local passthrough to sidestep this.

### Finding 4: CUDA Version Sensitivity

Vtable sizes change between CUDA versions. Local passthrough inherits the correct version automatically -- no reverse engineering needed.

## Recommendations

### Primary Strategy: Hybrid Local-Forward with Fallback

#### Tier 1: Local Driver Passthrough (Preferred)

Load the real libcuda.so via dlopen and forward cuGetExportTable calls.

**Why it works:** Most export tables are for runtime-internal bookkeeping (context TLS, callbacks, profiling). These manage in-process state, not GPU state. The real driver can provide these even without a local GPU.

**Implementation:**
1. At client init, use RTLD_NEXT or dlopen real libcuda.so by full path
2. Resolve real cuGetExportTable via dlsym
3. Forward all cuGetExportTable calls to the real one
4. Returned function pointers are valid in-process

**IMPORTANT:** Since OuterLink intercepts libcuda.so via LD_PRELOAD, must use RTLD_NEXT or load by absolute path to avoid recursing into our own interceptor. HAMi-core does exactly this in its dlsym override.

#### Tier 2: Server-Side Forward (Deferred)

If Tier 1 fails for specific UUIDs: send UUID to server, server calls real cuGetExportTable, returns table size, client builds local vtable with RPC stub function pointers. Complex -- defer until empirical testing shows need.

#### Tier 3: Graceful Failure

For unhandled UUIDs: return CUDA_ERROR_NOT_FOUND. Log UUID bytes. NEVER return NULL with CUDA_SUCCESS.

### Implementation Priority

1. **Phase 1 (now):** Local passthrough via real libcuda.so. No network. Handles cuBLAS/cuDNN init.
2. **Phase 2 (when needed):** Diagnostic UUID logging. Run PyTorch + cuDNN, check what fails.
3. **Phase 3 (if needed):** Server-side forwarding for specific failing UUIDs.

### What NOT To Do

1. Do not return a stub/dummy table (segfault or silent corruption)
2. Do not re-implement tables from scratch (ZLUDA-level effort, still fragile)
3. Do not ignore this function (cuBLAS/cuDNN init depends on it)
4. Do not make this a network call by default (latency on hot paths)

### Risk Analysis

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Client has no libcuda.so | Medium | Low | Require CUDA toolkit on client |
| Local driver errors without GPU | Medium | Medium | Fall back to CUDA_ERROR_NOT_FOUND, log |
| Table layout changes between versions | Low | High | Local passthrough inherits correct version |
| Returned pointers try to access local GPU | High | Medium | Test with cuBLAS/cuDNN; most init tables are CPU-only |
| Called before cuInit | Medium | High | HAMi-core confirms pre-init works |

## Sources

1. NVIDIA Forums - cuGetExportTable explanation: https://forums.developer.nvidia.com/t/cugetexporttable-explanation/259109
2. NVIDIA Forums - cudaGetExportTable a total hack: https://forums.developer.nvidia.com/t/cudagetexporttable-a-total-hack/20226
3. NVIDIA Forums - behavior: https://forums.developer.nvidia.com/t/cudagetexporttable-cugetexporttable-behavior/19495
4. ZLUDA Issue #47: https://github.com/vosen/ZLUDA/issues/47
5. xzhangxa dotfiles Issue #1: https://github.com/xzhangxa/dotfiles/issues/1
6. HAMi-core hook.c: https://github.com/Project-HAMi/HAMi-core/blob/main/src/cuda/hook.c
7. HAMi-core DeepWiki: https://deepwiki.com/Project-HAMi/HAMi-core/2-function-hooking-system
8. Cricket paper (Wiley 2022): https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.6474
9. RWTH-ACS/cricket: https://github.com/RWTH-ACS/cricket
10. ZLUDA repository (export_table.rs): https://github.com/vosen/ZLUDA
11. gVisor nvproxy: https://github.com/google/gvisor/blob/master/g3doc/proposals/nvidia_driver_proxy.md
12. dankwiki CUDA traces: https://nick-black.com/dankwiki/index.php/CUDA_traces
13. PyTorch Forums - CUDA callback: https://discuss.pytorch.org/t/best-place-to-subscribe-for-a-cuda-function-callback-in-pytorch/203105

## Open Questions

1. Which specific UUIDs does CUDA 12.x runtime request on first cuBLAS/cuDNN call? Needs ltrace testing.
2. Does cuGetExportTable succeed with libcuda.so but no physical GPU? Needs testing.
3. Are any export table functions called on GPU hot paths (not just init)? Needs profiling.
4. Does CUDA 12.x cuGetExportTable work before cuInit? HAMi-core suggests yes, needs verification.
