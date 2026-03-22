# Feature Plan: CUDA Completeness Batch 2 -- PyTorch Forward Pass Blockers

Created: 2026-03-22
Author: architect-agent

## Overview

With ~83 unique CUDA Driver API hook functions implemented (118 hook table entries
including version aliases), the next batch targets the remaining gaps that would
block a real PyTorch inference workload.

## Gap Analysis: What Blocks PyTorch

### Critical Missing (crash/abort)

1. **cuMemcpyDtoDAsync_v2** -- THE single biggest gap. Every forward/backward pass
   uses async DtoD copies. Only sync cuMemcpyDtoD exists.

2. **cuMemHostAlloc** -- Pinned+mapped host alloc with flags. DataLoader needs this.

3. **CUDA Graph hooks** -- Unhooked calls go to real libcuda with virtual handles =
   segfault. Must hook even as NOT_SUPPORTED stubs.

4. **cuLaunchKernelEx** -- CUDA 12+ Triton kernels use extended launch API.

5. **cuModuleLoad + cuModuleLoadFatBinary** -- File-based and fat binary loading.

6. **cuDeviceGetMemPool/SetMemPool** -- PyTorch 2.x pool init.

7. **cuMemAllocPitch_v2** -- cuDNN convolution workspace.

## Prioritized Implementation (7 items)

| # | Feature | Functions | Effort | Blocks |
|---|---------|-----------|--------|--------|
| 1 | Async DtoD copy | cuMemcpyDtoDAsync_v2 | Small | Inference |
| 2 | Flagged host alloc | cuMemHostAlloc | Small | Training |
| 3 | CUDA Graph stubs | 9 functions | Small | torch.compile safety |
| 4 | Extended launch | cuLaunchKernelEx | Medium | torch.compile perf |
| 5 | File/fat module | cuModuleLoad, cuModuleLoadFatBinary | Small-Med | Libraries |
| 6 | Device mem pool | 3 functions | Small | PyTorch 2.x init |
| 7 | Pitched alloc | cuMemAllocPitch_v2 | Medium | cuDNN convolutions |

Total: ~17 real implementations + ~9 stubs = ~26 new entries
After completion: ~109 unique hooks

Full details in thoughts/shared/plans/cuda-completeness-batch2-plan.md
