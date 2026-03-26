# R25 Research: Kernel Splitting Strategies

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** DRAFT
**Purpose:** Define the concrete mechanisms for splitting a single CUDA kernel launch across multiple physical GPUs. What we intercept, what we modify, and how we keep correctness.

---

## 1. Block-Level Partitioning: The Core Strategy

The fundamental approach: intercept a kernel launch, divide the grid's blocks across N GPUs, launch a smaller grid on each GPU.

### The Split
```
Application launches: kernel<<<256, 256>>>(args...)   // 256 blocks, 256 threads each
                                     |
                              OuterLink intercepts
                                     |
                    ┌────────────────┴────────────────┐
                    v                                  v
            GPU A: kernel<<<128, 256>>>(args_A...)   GPU B: kernel<<<128, 256>>>(args_B...)
            blocks 0-127                              blocks 128-255 (remapped)
```

### What Changes Per GPU
| Parameter | GPU A | GPU B |
|-----------|-------|-------|
| gridDim.x | 128 | 128 |
| blockDim | 256 (unchanged) | 256 (unchanged) |
| sharedMemBytes | unchanged | unchanged |
| stream | stream_A (local) | stream_B (local) |
| Pointer args | point to GPU A memory | point to GPU B memory |
| Scalar args | copied as-is | copied as-is |
| blockIdx offset | 0 | 128 |

---

## 2. The blockIdx Remapping Problem

This is the single most important technical challenge. When GPU B launches 128 blocks, its `blockIdx.x` ranges from 0 to 127. But the kernel code expects blocks 128-255 to exist:

```c
// Original kernel
__global__ void add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}
```

If GPU B runs this with blockIdx 0-127, it computes indices 0-32767 instead of 32768-65535. **Wrong results.**

### Strategy A: Kernel Argument Injection (Preferred)

Add an invisible offset parameter to the kernel. Requires PTX-level modification.

**How it works:**
1. Intercept `cuModuleLoadData` / `cuModuleLoadFatBinary` — capture the PTX/SASS
2. Parse the PTX to find all reads of `%ctaid.x` (the PTX register for `blockIdx.x`)
3. Insert an add instruction: `add.u32 %ctaid_adjusted, %ctaid.x, %offset_param;`
4. Replace all uses of `%ctaid.x` with `%ctaid_adjusted`
5. Add a new kernel parameter `__blockIdx_offset`
6. Load the modified PTX on each GPU

**PTX example (before):**
```
.entry add(.param .u64 a, .param .u64 b, .param .u64 c, .param .s32 n) {
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mad.lo.u32 %r3, %r1, %r2, %tid.x;  // blockIdx.x * blockDim.x + threadIdx.x
    ...
}
```

**PTX example (after):**
```
.entry add(.param .u64 a, .param .u64 b, .param .u64 c, .param .s32 n, .param .u32 __blkoff) {
    ld.param.u32 %r_off, [__blkoff];
    mov.u32 %r1, %ctaid.x;
    add.u32 %r1, %r1, %r_off;           // adjusted blockIdx
    mov.u32 %r2, %ntid.x;
    mad.lo.u32 %r3, %r1, %r2, %tid.x;  // (blockIdx.x + offset) * blockDim.x + threadIdx.x
    ...
}
```

**Pros:** Clean, correct, works for any kernel. Only needs PTX modification once at module load time.
**Cons:** Requires a PTX parser/transformer. Must handle `%ctaid.y` and `%ctaid.z` too for 2D/3D grids. SASS (compiled binary) is harder to modify — would need to force PTX JIT compilation.

**Complexity:** MODERATE. PTX is a well-documented text format. Writing a regex-based transformer for `%ctaid` references is feasible. The risk is corner cases: inline PTX in C++, or SASS-only modules (no PTX available).

### Strategy B: Memory Remapping (Avoid blockIdx Modification)

Instead of fixing blockIdx, fix the memory so that "wrong" indices land on correct data.

**Concept:** If GPU B's block 0 thinks it's computing index 0, make sure `a[0]` on GPU B actually contains `a[32768]` from the original data.

**How it works:**
1. Analyze the kernel's data access pattern (which indices each block touches)
2. For each GPU, copy only the data slice that GPU's blocks will access
3. Remap pointers so each GPU sees its slice starting at address 0

**Example for the `add` kernel:**
- GPU A gets `a[0..32767]`, `b[0..32767]`, `c[0..32767]`
- GPU B gets `a[32768..65535]`, `b[32768..65535]`, `c[32768..65535]`
- Both GPUs launch blocks 0-127, indexing from 0 — correct, because data is pre-sliced

**Pros:** No PTX modification needed. Works at the memory/pointer level.
**Cons:** Requires knowing the access pattern to slice data correctly. Fails for non-trivial access patterns (indirect indexing, data-dependent access). Requires copying/migrating data before launch. Only works for simple stride patterns.

**Complexity:** EASY for simple kernels, IMPOSSIBLE for complex ones. This is a supplementary strategy, not a primary one.

### Strategy C: Virtual Address Space Tricks (UVM)

Use CUDA Unified Virtual Memory to create a unified address space where both GPUs can access all data.

**How it works:**
1. Allocate all kernel data with `cuMemCreate` + `cuMemMap` (virtual memory management APIs)
2. Map the same virtual address range on both GPUs
3. Each GPU accesses what it needs; page faults trigger remote fetches (R19)

**Pros:** No PTX modification. No data slicing. Transparent.
**Cons:** Page fault latency on every cross-GPU access (~10-50us per fault). Performance is terrible for random access patterns. Requires UVM support and R19 (Network Page Faults). Essentially gives up on performance for correctness.

**Complexity:** LOW to implement (UVM does the work), but HIGH performance cost.

### Recommended: Strategy A + B Hybrid

1. **Always apply Strategy A** (PTX blockIdx offset) — this is the universal correctness solution
2. **When possible, also apply Strategy B** (data slicing) — avoid cross-GPU memory access entirely
3. **Fall back to Strategy C** (UVM/page faults) only when data access patterns are unknown and cross-GPU access is unavoidable

---

## 3. Handling Cross-Block Data Dependencies

### Case 1: No Dependencies (Embarrassingly Parallel)

Each block reads/writes only its own data region. **Just split.** No communication needed.

```c
// Each block processes its own tile — perfect split candidate
c[blockIdx.x * blockDim.x + threadIdx.x] =
    a[blockIdx.x * blockDim.x + threadIdx.x] +
    b[blockIdx.x * blockDim.x + threadIdx.x];
```

### Case 2: Atomic Reductions to Global Memory

Blocks perform `atomicAdd` (or similar) to shared global locations — typically for reductions, histograms, loss accumulation.

```c
// All blocks atomically add to a single global counter
atomicAdd(&total_loss, local_loss);
```

**Splitting strategy: Local Accumulate + Final Merge**

1. Each GPU gets its own copy of the reduction target (`total_loss_A`, `total_loss_B`)
2. Blocks on each GPU atomically update their LOCAL copy (fast, hardware atomics)
3. After both GPUs finish, merge: `total_loss = total_loss_A + total_loss_B`

**Implementation:**
- Detect atomic operations on global memory during PTX analysis
- For each atomic target address, allocate a local copy per GPU
- Rewrite PTX to redirect atomics to local copies
- After kernel completion on all GPUs, perform the merge (host-side or RDMA)

**Cost:** One merge step after kernel completion. For simple reductions (single counter), this is microseconds. For histograms with millions of bins, the merge itself could be a kernel.

**Complexity:** MODERATE for simple atomics (counters, sums). HARD for complex atomics (CAS-based data structures, linked lists). General atomic operations on arbitrary global addresses are NOT automatically splittable.

### Case 3: Producer-Consumer via Global Memory

Block A writes data that Block B reads. Rare in well-structured CUDA code, but exists in:
- Stencil computations (block reads neighbor block's boundary)
- Multi-stage pipelines within a single kernel

**Splitting strategy:** This requires the data sharing mechanism from R18 (Virtual NVLink). Without it:
- Detect the dependency at PTX level (which is extremely hard — see Section 5)
- Place dependent blocks on the same GPU
- Or: accept the cross-GPU latency and use RDMA reads

**Complexity:** VERY HIGH. This is the domain of R18.

### Case 4: Grid-Level Synchronization (Cooperative Groups)

```c
cooperative_groups::grid_group grid = cooperative_groups::this_grid();
// ... compute phase 1 ...
grid.sync();  // ALL blocks must reach here
// ... compute phase 2 using phase 1 results ...
```

**Splitting strategy: Cross-GPU Barrier**

1. Detect cooperative launch (`cudaLaunchCooperativeKernel` instead of `cudaLaunchKernel`)
2. At each `grid.sync()` point:
   - Each GPU reaches the barrier locally
   - Signal the other GPU via RDMA write to a doorbell
   - Wait for signal from all other GPUs
   - Resume

**Cost per sync point:** ~2-5us over RDMA (write + poll). If the kernel has 10 sync points and the compute between them is 100us, overhead is 2-5%. If compute between syncs is 10us, overhead is 20-50%. **Devastating for fine-grained sync.**

**Additional constraint:** Cooperative launches require ALL blocks to be resident simultaneously. Splitting across GPUs actually helps here — more SMs total means the grid size limit increases.

**Complexity:** HIGH. Requires modifying the cooperative group sync mechanism at the driver level, not just PTX rewriting.

---

## 4. Data Locality: Partitioning Blocks to Match Data

The fastest split is one where each GPU's blocks only access data already on that GPU. Zero cross-GPU traffic.

### Locality-Aware Block Assignment

```
Data layout: [Chunk 0 on GPU A] [Chunk 1 on GPU B] [Chunk 2 on GPU A] [Chunk 3 on GPU B]

Block assignment:
  Blocks 0, 2 → GPU A (access Chunks 0, 2 — local)
  Blocks 1, 3 → GPU B (access Chunks 1, 3 — local)
```

### How to Determine Block-to-Data Mapping

1. **Static analysis (PTX):** Parse the kernel to find how blockIdx maps to memory addresses. For `addr = blockIdx.x * stride + offset`, we know exactly which data each block touches.
2. **Profiling:** Run the kernel once on one GPU with memory access logging, then use the profile to assign blocks.
3. **Pattern matching:** Recognize common patterns:
   - Row-major tile: block (bx, by) accesses rows [by*TILE_H .. (by+1)*TILE_H], cols [bx*TILE_W .. (bx+1)*TILE_W]
   - Batch dimension: block bx processes batch item bx
   - Reduction: all blocks access the same data (read-only) + write to one accumulator

### When Locality Fails

Some kernels have all-to-all access patterns:
- FFT butterfly: block i reads from block i XOR stride
- Graph neural networks: block accesses depend on graph topology
- Sparse matrix operations: data-dependent indexing

For these, either:
- Accept cross-GPU traffic (and the performance hit)
- Don't split (redirect entire kernel to one GPU)
- Pre-distribute the data and accept duplication

---

## 5. PTX/SASS Analysis for Dependency Detection

The ambitious goal: automatically determine if a kernel can be split by analyzing its compiled code.

### What to Look For in PTX

| Pattern | Implication | Detection Difficulty |
|---------|-------------|---------------------|
| No atomics on global mem | Blocks likely independent | EASY — grep for `atom.` |
| Only `%ctaid`-derived global addresses | Predictable access pattern | MODERATE — data flow analysis |
| Cooperative group calls | Grid sync present | EASY — grep for `bar.sync` / `cta.sync` |
| `atom.global.add` | Reduction pattern | EASY — can use local accumulate + merge |
| `atom.global.cas` | Lock-free data structure | HARD — cannot easily split |
| Indirect memory access (pointer chasing) | Data-dependent pattern | HARD — need runtime profiling |
| `ld.global` with non-`%ctaid` index | Cross-block data read | MODERATE — data flow analysis |

### Static Analysis: What's Feasible

**Feasible now:**
- Detect presence/absence of atomics
- Detect cooperative group usage
- Detect `blockIdx`-linear access patterns (addr = f(blockIdx))
- Count global memory operations (estimate data volume)

**Feasible with effort:**
- Data flow from `blockIdx` to memory addresses (symbolic execution of the address computation)
- Identify reduction patterns vs general atomics
- Estimate shared memory usage patterns

**Not feasible (research-level):**
- Prove block independence for arbitrary kernels (equivalent to the halting problem for general programs)
- Handle data-dependent control flow (branches based on loaded data)
- Analyze SASS (proprietary binary, undocumented encoding per GPU generation)

### Pragmatic Approach: Classification Not Verification

Instead of trying to PROVE a kernel is safe to split (undecidable), **classify kernels into categories:**

1. **GREEN — Safe to split:** No atomics, no cooperative groups, blockIdx-linear access → split freely
2. **YELLOW — Splittable with work:** Has atomics (use local accumulate + merge), or known reduction pattern → split with modifications
3. **RED — Do not split:** Cooperative groups, complex atomics, data-dependent access, or unknown pattern → run on single GPU
4. **UNCLASSIFIED:** Cannot determine → treat as RED (safe default)

---

## 6. Kernel Argument Modification in Practice

### Pointer Arguments
Every pointer argument needs per-GPU handling:

```c
__global__ void matmul(float* A, float* B, float* C, int M, int N, int K) {
    // A, B, C are pointers to GPU memory
}
```

When splitting:
- If data is pre-distributed: remap pointers to each GPU's local copy
- If data is shared (read-only input like B in matmul): replicate to each GPU
- If data is written (output like C): each GPU gets its own region, merge if needed

### Pointer Classification

| Pointer Role | Strategy |
|-------------|----------|
| Read-only, all blocks read all data | Replicate to each GPU |
| Read-only, each block reads a partition | Distribute matching partitions |
| Write-only, each block writes its partition | Each GPU writes locally, no merge needed |
| Read-write with atomics | Local copies + merge |
| Read-write general | Cannot split without coherency (R18) |

### How to Determine Pointer Role

From the kernel launch alone, we DON'T know which pointers are read vs write. Options:
1. **PTX analysis:** Check if address derived from a pointer appears in `st.global` (write) or only `ld.global` (read)
2. **CUDA annotations:** The programmer may use `const` or `__restrict__` hints
3. **Runtime profiling:** Run once, observe access patterns
4. **Conservative default:** Assume all pointers are read-write (prevents splitting for most kernels — too conservative)

Recommendation: PTX analysis for read/write classification + `__restrict__` hints when available.

---

## 7. Integration with R13 Graph Interception

R13 gives us the CUDA computation graph — the full DAG of kernels, memory operations, and dependencies. This is far more powerful than analyzing individual kernels:

### Graph-Level vs Kernel-Level Splitting

| Aspect | Kernel-Level Split | Graph-Level Split |
|--------|-------------------|-------------------|
| Granularity | Blocks within one kernel | Entire kernels across GPUs |
| Complexity | HIGH (PTX rewriting, blockIdx) | MODERATE (just redirect launches) |
| Data movement | Per-block partition | Per-kernel partition |
| Correctness | Must preserve intra-kernel semantics | Just preserve inter-kernel dependencies |
| Applicability | Only independent-block kernels | Any kernel (just move the whole thing) |

**Graph-level splitting is MUCH easier and should be the primary strategy.** Kernel-level splitting is the secondary, advanced strategy for when a single kernel is the bottleneck.

### Hybrid Approach

1. **Phase 1:** Use R13 to split the graph — different kernels on different GPUs
2. **Phase 2:** For the single hottest kernel (the bottleneck), apply kernel splitting
3. **Phase 3:** For all kernels, apply kernel splitting where safe

This gives us the best ROI — graph splitting covers most cases, kernel splitting handles the rest.

---

## Open Questions

1. **Can we force PTX JIT compilation?** If the app provides only SASS (precompiled binary), we can't modify PTX. Can we force the driver to JIT from PTX by intercepting `cuModuleLoadFatBinary` and stripping the SASS sections? This would need testing.
2. **Multi-dimensional grids:** Most strategies discuss 1D grids. 2D/3D grids need 2D/3D block partitioning. How to partition a 2D grid for data locality in 2D problems (image processing, matrix ops)?
3. **Dynamic shared memory:** When `sharedMemBytes > 0`, does the split affect anything? No — shared memory is per-block, per-SM. But we need to ensure each GPU's SMs have enough shared memory capacity.
4. **Kernel argument size limits:** CUDA has a 4KB limit on kernel parameters. Adding `__blkoff_x/y/z` adds 12 bytes. Should be fine, but verify no kernels are near the limit.

---

## Related Documents

- [01-cuda-execution-model.md](01-cuda-execution-model.md) — How CUDA executes (the foundation)
- [03-practical-feasibility.md](03-practical-feasibility.md) — Which kernels can actually be split
- R13: CUDA Graph Interception — graph-level splitting (easier, higher ROI)
- R18: Virtual NVLink — coherency for cross-GPU data access
- R26: PTP Clock Sync — synchronized kernel launches
