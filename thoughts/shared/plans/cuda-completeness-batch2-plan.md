# Feature Plan: CUDA Completeness Batch 2 -- PyTorch Forward Pass Blockers

Created: 2026-03-22
Author: architect-agent

## Overview

With ~83 unique CUDA Driver API hook functions implemented (118 hook table entries
including version aliases), the next batch targets the remaining gaps that would block
a real PyTorch inference workload.

## Gap Analysis

### Critical Missing (crash/abort)

1. **cuMemcpyDtoDAsync_v2** -- THE single biggest gap. Every forward/backward pass uses
   async DtoD copies. Only sync cuMemcpyDtoD exists.

2. **cuMemHostAlloc** -- Pinned+mapped host alloc with flags. DataLoader needs this.

3. **CUDA Graph hooks** -- Unhooked calls go to real libcuda = segfault. Must hook as stubs.

4. **cuLaunchKernelEx** -- CUDA 12+ Triton kernels use extended launch API.

5. **cuModuleLoad + cuModuleLoadFatBinary** -- File-based and fat binary loading.

6. **cuDeviceGetMemPool/SetMemPool** -- PyTorch 2.x pool init.

7. **cuMemAllocPitch_v2** -- cuDNN convolution workspace.

### Important Missing (degraded behavior)

8. cuDeviceGetUuid_v2, cuMemcpyPeerAsync, cuMemcpy2DAsync_v2,
   cuMemGetAllocationGranularity

## Prioritized Implementation (7 items)

| # | Feature | Functions | Effort | Blocks |
|---|---------|-----------|--------|--------|
| 1 | Async DtoD copy | cuMemcpyDtoDAsync_v2 | Small | Inference |
| 2 | Flagged host alloc | cuMemHostAlloc | Small | Training |
| 3 | CUDA Graph stubs | 9 functions | Small | torch.compile safety |
| 4 | Extended launch | cuLaunchKernelEx | Medium | torch.compile perf |
| 5 | File/fat module | cuModuleLoad, LoadFatBinary | Small-Med | Libraries |
| 6 | Device mem pool | 3 functions | Small | PyTorch 2.x init |
| 7 | Pitched alloc | cuMemAllocPitch_v2 | Medium | cuDNN convolutions |

**Total: ~17 real implementations + ~9 stubs = ~26 new entries**
**After completion: ~109 unique hooks**

## Priority Details

### P1: cuMemcpyDtoDAsync_v2 (CRITICAL)
- Follows exact pattern as cuMemcpyHtoDAsync. Small effort.
- Files: protocol.rs, handler.rs, ffi.rs, interpose.c/h

### P2: cuMemHostAlloc (CRITICAL)
- Local operation on client. Small effort.
- Files: ffi.rs, interpose.c/h

### P3: CUDA Graph Stubs (CRITICAL for safety)
- 9 stubs returning CUDA_ERROR_NOT_SUPPORTED. Trivial.
- cuStreamBeginCapture_v2, cuStreamEndCapture, cuStreamIsCapturing,
  cuStreamGetCaptureInfo_v2, cuGraphCreate, cuGraphDestroy,
  cuGraphInstantiate_v2, cuGraphLaunch, cuGraphExecDestroy
- Files: ffi.rs, interpose.c/h

### P4: cuLaunchKernelEx (HIGH)
- CUlaunchConfig struct parsing. Medium effort.
- Files: protocol.rs, cuda_types.rs, handler.rs, ffi.rs, interpose.c/h

### P5: cuModuleLoad + cuModuleLoadFatBinary (HIGH)
- Client reads file, sends bytes via existing LoadData path.
- Fat binary forwarded verbatim (server CUDA parses it).
- Files: ffi.rs, interpose.c/h (server reuses LoadData handler)

### P6: cuDeviceGetMemPool + cuDeviceSetMemPool + cuMemGetAllocationGranularity (MEDIUM)
- GetMemPool returns default pool (already tracked). Small effort.
- Files: ffi.rs, interpose.c/h

### P7: cuMemAllocPitch_v2 (MEDIUM)
- Server round-trip for pitch value. Medium effort.
- Files: protocol.rs, handler.rs, ffi.rs, interpose.c/h

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| CUlaunchConfig varies across CUDA versions | Medium | Target CUDA 12.x |
| Graph stubs hide perf issues | Low | Log warning on first attempt |
| Fat binary format undocumented | Medium | Forward opaque, server parses |
| Pitched alloc alignment varies by GPU | Low | Always query real server GPU |

## Open Questions

- [ ] Does cuDeviceGetUuid_v2 need separate hook or does cuGetProcAddress redirect?
- [ ] cuMemHostAlloc DEVICEMAP: does server need to know about mappings?
- [ ] cuModuleLoad: client reads file, sends bytes (answer: yes, no shared FS)

## Success Criteria

1. cuMemcpyDtoDAsync_v2 works with stream ordering on server
2. cuMemHostAlloc returns usable pinned memory with all flag combos
3. torch.compile runs without segfault (graphs gracefully disabled)
4. Triton kernels launch via cuLaunchKernelEx on CUDA 12
5. cuModuleLoad can load .cubin from disk through client
6. PyTorch CUDACachingAllocator init completes on cuDeviceGetMemPool
7. cuDNN algorithm selection succeeds with cuMemAllocPitch_v2

## Related Documents

- planning/phases/P7-cuda-completeness.md -- Master function registry
- planning/pre-planning/02-FINAL-PREPLAN.md -- Overall project plan
- docs/architecture/00-project-vision.md -- Project vision
