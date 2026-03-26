# R25 Research: CUDA Execution Model Deep Dive

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** DRAFT
**Purpose:** Understand exactly how CUDA kernels execute on hardware — the foundation for knowing what can and cannot be split across GPUs.

---

## 1. The Thread Hierarchy

CUDA organizes parallel work in a strict hierarchy:

```
Grid (the entire kernel launch)
├── Block 0 (aka "thread block" or "CTA")
│   ├── Warp 0 (threads 0-31)
│   ├── Warp 1 (threads 32-63)
│   └── ...
├── Block 1
│   ├── Warp 0
│   └── ...
└── Block N
```

### Thread
- Smallest unit of execution
- Has private registers and local memory
- Identified by `threadIdx.x/y/z` within its block

### Warp (32 threads)
- The actual hardware execution unit — all 32 threads execute the same instruction (SIMT)
- Divergent branches serialize (threads take turns), but reconverge
- Warp-level primitives: `__shfl_sync`, `__ballot_sync`, `__any_sync`
- Warps within a block are completely independent in scheduling

### Block (aka CTA — Cooperative Thread Array)
- 1 to 1024 threads (hardware limit)
- Shares a pool of shared memory (up to 48KB default, 164KB max on Ampere with opt-in)
- Can synchronize all threads with `__syncthreads()`
- Scheduled as a unit onto one SM — **a block NEVER spans multiple SMs**
- Once a block starts on an SM, it runs to completion there (no migration)

### Grid
- Collection of all blocks in one kernel launch
- Defined by `gridDim.x/y/z` (up to 2^31 - 1 in x, 65535 in y/z)
- Blocks in a grid are **not ordered** — the scheduler picks them in any order
- No built-in synchronization between blocks (except cooperative groups)

---

## 2. Block Scheduling: How the GPU Assigns Work

This is critical for kernel splitting.

### The Block Scheduler
- Hardware unit at the GPC (Graphics Processing Cluster) level
- Takes the grid's blocks and distributes them across available SMs
- **Block assignment is opaque and non-deterministic** — the programmer cannot control which SM runs which block
- Blocks are dispatched in roughly sequential order (block 0, 1, 2...) but this is NOT guaranteed
- Multiple blocks can run concurrently on one SM (occupancy)

### Occupancy
- Each SM has limited resources: registers, shared memory, warp slots
- A block "claims" a portion of these resources
- If a block uses 32 registers/thread and 16KB shared memory, the SM might fit 2-4 blocks concurrently
- Higher occupancy = better latency hiding, but not always better performance

### Why This Matters for Splitting
The block scheduler is the key insight: **the GPU already treats blocks as independent schedulable units.** The hardware makes no assumption that block 0 and block 1 will run on the same SM, or even at the same time. This is what makes block-level splitting theoretically possible — we're doing what the GPU already does, just across a wider "pool" of SMs.

---

## 3. Memory Hierarchy

### Registers (per-thread)
- Fastest: ~0 cycle access
- 255 max per thread (Ampere)
- Spill to local memory (slow) if exceeded
- **Private to the thread.** Cannot be shared across blocks or GPUs. Not a splitting concern.

### Shared Memory (per-block, per-SM)
- ~20-30 cycle access
- Up to 164KB per SM (Ampere, opt-in via `cudaFuncSetAttribute`)
- Statically or dynamically allocated per block
- **This is the hard boundary.** Shared memory is physically on-chip in the SM. A block's shared memory CANNOT be accessed by another block, let alone another GPU. Any kernel that uses shared memory for inter-block communication is... well, it can't, by design. But some patterns simulate it through global memory.

### L1 / Texture Cache (per-SM)
- Unified with shared memory pool on modern GPUs
- Automatically caches global memory reads
- **Per-SM, non-coherent across SMs** — same GPU, different SMs may see stale data
- Across GPUs: completely independent. No coherency whatsoever.

### L2 Cache (per-GPU)
- Shared across all SMs in one GPU
- 6MB on RTX 3090, up to 96MB on H100
- Caches global memory traffic
- **Per-GPU.** Splitting means each GPU has its own L2 — cache hits on one GPU won't help the other.

### Global Memory (VRAM)
- Off-chip, high bandwidth (936 GB/s on 3090), high latency (~400-600 cycles)
- Accessible by all threads in all blocks
- **This is where cross-block data sharing happens**
- When splitting: each GPU has its own VRAM. Accesses to the "other GPU's" memory need network transfer.
- Atomic operations on global memory: `atomicAdd`, `atomicCAS`, etc. — hardware-guaranteed on one GPU, need RDMA atomics across GPUs.

### Constant Memory
- 64KB, cached aggressively, read-only during kernel
- Set before kernel launch via `cudaMemcpyToSymbol`
- **Easy to replicate** — just copy to each GPU before launch.

### Texture Memory
- Read-only, 2D/3D spatial locality optimized
- **Easy to replicate** — read-only data, copy to each GPU.

---

## 4. Synchronization Mechanisms

### `__syncthreads()` — Block-Level
- Barrier for all threads in one block
- All threads must reach the barrier before any can proceed
- **Block-local only.** Since a block runs on one SM, this is a hardware barrier.
- Not a splitting concern — each block stays on one GPU, so `__syncthreads()` works as normal.

### `cooperative_groups::grid_group::sync()` — Grid-Level
- Barrier across ALL blocks in the grid
- Requires cooperative launch: `cudaLaunchCooperativeKernel()`
- **All blocks must be resident simultaneously** — limits grid size to (blocks per SM) * (number of SMs)
- **This is the splitting killer.** A grid-level sync requires every block to pause and wait for every other block. Across GPUs, this means a network round-trip for every sync point. At 100Gbps RDMA, that's ~1-2us minimum. If the kernel syncs frequently, overhead destroys performance.

### Atomic Operations — Implicit Sync
- `atomicAdd(&global_var, val)` — hardware read-modify-write
- Used for reductions, histograms, counters
- **Cross-GPU atomics need special handling.** Options:
  - Local atomics + final merge (each GPU accumulates locally, merge at end)
  - RDMA atomics (ConnectX-5 supports remote atomic ops, but ~1-3us per op)
  - Neither is transparent — requires knowing which atomics are cross-GPU

### Memory Fences
- `__threadfence()` — ensures writes visible to all threads on same GPU
- `__threadfence_system()` — ensures writes visible to host and all GPUs (unified memory)
- Across split GPUs: `__threadfence()` only covers the local GPU. Cross-GPU visibility requires explicit flushes or RDMA writes.

---

## 5. Kernel Launch Parameters

When the app calls `cuLaunchKernel()` or `<<<grid, block, sharedMem, stream>>>`:

```c
cuLaunchKernel(
    function,       // CUfunction — the compiled kernel
    gridDimX,       // Number of blocks in X
    gridDimY,       // Number of blocks in Y
    gridDimZ,       // Number of blocks in Z
    blockDimX,      // Threads per block in X
    blockDimY,      // Threads per block in Y
    blockDimZ,      // Threads per block in Z
    sharedMemBytes, // Dynamic shared memory per block
    stream,         // CUDA stream
    kernelParams,   // Array of kernel argument pointers
    extra           // Alternative parameter passing
);
```

### What We Can Intercept and Modify
- **gridDim**: We can reduce this per GPU. GPU A gets gridDim/2, GPU B gets gridDim/2.
- **blockDim**: This stays the same — block internal structure doesn't change.
- **sharedMemBytes**: Same per block — no change needed.
- **stream**: Each GPU gets its own stream. We need to synchronize across streams.
- **kernelParams**: Pointer arguments need to map to the correct GPU's memory. Scalar arguments are just copied.
- **function**: Must be loaded on each GPU separately. Same PTX, loaded independently.

### The blockIdx Problem
When we split a grid of 256 blocks across 2 GPUs:
- GPU A launches blocks 0-127 with `gridDim = 128`
- GPU B launches blocks 128-255... but `blockIdx` starts at 0

The kernel code may use `blockIdx.x` to index into data arrays:
```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
data[idx] = ...;
```

If GPU B's blocks think they are blocks 0-127 instead of 128-255, they'll access the wrong data.

**Solutions:**
1. **Offset argument**: Add a `blockIdxOffset` kernel argument and modify PTX to add it to `blockIdx` reads. Invasive — requires PTX rewriting.
2. **Grid offset**: CUDA Driver API has no built-in "start blockIdx at N" feature. But we can pass the offset as a kernel parameter and let our modified kernel use it.
3. **Virtual addressing**: Map memory so that even with "wrong" blockIdx, accesses land on the right data. Requires understanding the kernel's access pattern.

This is one of the core technical challenges. See `02-kernel-splitting-strategies.md` for detailed solutions.

---

## 6. Independent vs Dependent Blocks

### Truly Independent Blocks (Easy to Split)
Blocks that:
- Read from global memory at addresses derived from `blockIdx` (each block reads its own region)
- Write to global memory at addresses derived from `blockIdx` (each block writes its own region)
- Use shared memory only for intra-block communication
- Never use atomic operations on shared global addresses
- Never use cooperative groups

**Examples:**
- Elementwise operations: `C[i] = A[i] + B[i]`
- Map operations: each block processes one tile of data independently
- Matrix multiply (tiled): each block computes one tile of C from a row of A and column of B
- Convolution: each block handles one output tile
- Batch processing: each block handles one item in a batch

This category covers the **vast majority of GPU kernels in practice**. Most well-written CUDA code is embarrassingly parallel at the block level.

### Partially Dependent Blocks (Splittable with Work)
Blocks that:
- Use atomic operations on global memory (reductions, histograms)
- Read data that another block has written (producer-consumer, but rare in well-structured code)

**Handling atomics:** Each GPU accumulates locally, then merge partial results. For `atomicAdd` on a single counter, this means each GPU gets its own counter, then we add them at the end. For histograms with many bins, each GPU builds a local histogram, then we merge.

### Fully Dependent Blocks (Cannot Split)
Blocks that:
- Use `cooperative_groups::grid_group::sync()` for iterative algorithms
- Implement block-to-block communication through global memory with grid-sync barriers
- Perform multi-pass algorithms where pass N depends on all blocks completing pass N-1

**These kernels cannot be split without fundamentally changing the synchronization model.** The cost of grid-sync across GPUs (network round-trip per sync) would likely destroy any benefit from extra compute.

---

## 7. Key Insight Summary

| Property | Splitting Impact |
|----------|-----------------|
| Blocks are independently scheduled | **Enables splitting** — GPU already treats them as independent |
| Shared memory is per-block | **No impact** — stays within one GPU |
| `__syncthreads` is block-local | **No impact** — stays within one GPU |
| Global memory is per-GPU | **Major challenge** — need data partitioning or coherency |
| Atomics on global memory | **Moderate challenge** — local accumulate + merge |
| Grid-level sync | **Splitting killer** — network round-trip per sync point |
| `blockIdx` numbering | **Must solve** — offset or remap required |
| Kernel arguments (pointers) | **Must remap** — each GPU needs valid pointers to its data |
| Constant/texture memory | **Easy** — replicate to each GPU |

---

## Open Questions

1. **What percentage of real-world kernels use cooperative groups?** Anecdotally very few — it's an advanced feature mostly used in research papers. Need to verify against PyTorch/TensorFlow kernel libraries.
2. **Can we detect grid-level sync at PTX level?** Looking for `bar.sync` with specific parameters, or calls to `cooperative_groups` functions.
3. **How does CUDA Dynamic Parallelism interact with splitting?** Child kernel launches from within a kernel — each split portion would launch children independently. Likely compatible but needs verification.
4. **Warp-level primitives across split boundaries?** Not a concern — warps are within blocks, which are within one GPU.

---

## Related Documents

- [02-kernel-splitting-strategies.md](02-kernel-splitting-strategies.md) — How to actually perform the split
- [03-practical-feasibility.md](03-practical-feasibility.md) — What's realistic
- R13: CUDA Graph Interception — graph-level analysis
- R18: Virtual NVLink — coherency for shared data
- R26: PTP Clock Sync — synchronized launches
