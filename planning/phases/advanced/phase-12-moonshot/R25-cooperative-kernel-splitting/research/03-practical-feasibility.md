# R25 Research: Practical Feasibility Assessment

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** DRAFT
**Purpose:** Brutally honest assessment of what's achievable. Which real workloads benefit from kernel splitting, which don't, and what the actual performance looks like.

---

## 1. Which Real CUDA Kernels CAN Be Split?

### GEMM (Matrix Multiplication)

**Verdict: SPLITTABLE — the best candidate**

Standard tiled GEMM: `C[M,N] = A[M,K] * B[K,N]`
- Each block computes one tile of C (e.g., 128x128)
- Block (bx, by) reads row-tile from A and col-tile from B
- **Blocks are completely independent** — no inter-block communication
- Data locality: partition C tiles across GPUs, replicate A and B (or partition along one dimension)

**Existing art:** cuBLAS-XT already does multi-GPU GEMM by splitting along M or N. It achieves 60-80% efficiency on 2 GPUs. The overhead comes from:
- Replicating the shared dimension (K) to each GPU
- Transferring partial results back for accumulation
- Synchronization between GPUs

**What we'd achieve:** Similar to cuBLAS-XT, but transparent. The app calls `cublasSgemm` for a single GPU, OuterLink splits it across 2.

**Honest efficiency estimate:** 50-75% on 2 GPUs over 100Gbps RDMA. Lower than cuBLAS-XT's NVLink-based multi-GPU because our interconnect is ~50x slower for bulk transfer.

### Convolution (Conv2D)

**Verdict: SPLITTABLE — good candidate**

Standard im2col + GEMM approach:
- Output tensor is spatially decomposable
- Each block computes one output tile
- Input requires halo regions (overlapping border data) between tiles
- Weights (kernel filters) are read-only, replicated to all GPUs

**Splitting strategy:** Partition output spatial dimensions across GPUs. Each GPU needs its input partition PLUS the halo. For a 3x3 kernel, the halo is 1 pixel on each edge — minimal extra data.

**Honest efficiency estimate:** 60-80% for large feature maps. The halo overhead is small relative to the tile size. Worse for many small convolutions (launch overhead dominates).

### Elementwise / Map Operations

**Verdict: PERFECTLY SPLITTABLE — trivial**

`C[i] = f(A[i], B[i])` — ReLU, add, multiply, normalization per-element.
- Zero inter-block dependencies
- Perfect data locality: block i touches only element range i
- No shared data, no atomics, no sync

**Honest efficiency estimate:** 90-95% on 2 GPUs. The only overhead is launch coordination (~2-5us) which is negligible for large tensors. For tiny tensors, the launch overhead exceeds compute time — don't split.

### Attention (Transformer Self-Attention)

**Verdict: PARTIALLY SPLITTABLE — depends on implementation**

Flash Attention (Dao et al.):
- Tiles the Q, K, V matrices into blocks
- Each block computes one output tile of the attention matrix
- **However:** the softmax normalization requires a reduction across the sequence dimension
- The logsumexp correction between tiles is a cross-block dependency

**Splitting strategy:** Split along the batch dimension (trivially parallel) or the head dimension (independent). Splitting along the sequence dimension requires the logsumexp correction across GPUs — doable but adds a merge step.

**Honest efficiency estimate:**
- Batch/head split: 80-90% (trivially parallel)
- Sequence split: 50-70% (logsumexp merge overhead)

### Reduction Kernels (sum, max, mean)

**Verdict: SPLITTABLE — with merge step**

Tree reduction: each block reduces a chunk, then blocks reduce those results.

**Splitting strategy:** Each GPU reduces its blocks locally, then merge partial results.

```
GPU A: blocks 0-127 → partial_sum_A
GPU B: blocks 128-255 → partial_sum_B
Merge: total_sum = partial_sum_A + partial_sum_B  (host-side, ~1us)
```

**Honest efficiency estimate:** 85-95% for large reductions. The merge step is trivial — one RDMA read + one add. For small reductions (few elements), don't bother splitting.

### Histogram / Scatter

**Verdict: SPLITTABLE WITH CAVEATS**

Blocks atomically increment bins: `atomicAdd(&hist[data[idx]], 1)`
- If bins are known in advance: each GPU builds local histogram, merge at end
- If bins are data-dependent and sparse: merge is expensive

**Honest efficiency estimate:** 70-85% for fixed-size histograms. The merge is O(num_bins), which can be large but is a simple elementwise add.

---

## 2. Which Kernels CANNOT Be Split?

### Kernels with Grid-Level Sync

**Examples:** Jacobi iteration, multigrid solvers, some graph algorithms

These kernels perform multiple iterations where each iteration reads the previous iteration's results from ALL blocks:

```c
for (int iter = 0; iter < N; iter++) {
    compute_local();
    grid.sync();  // wait for all blocks
    read_neighbors();
}
```

Cross-GPU grid sync costs 2-5us per sync point via RDMA. If the kernel syncs 1000 times (typical for iterative solvers), that's 2-5ms of pure synchronization overhead. If total compute is 10ms, synchronization adds 20-50%. **Not worth it.**

### Kernels with Heavy Shared-Memory Communication Patterns

Technically, shared memory is block-local and doesn't cross GPU boundaries. But some algorithmic patterns depend on blocks being able to quickly exchange data through global memory with implicit ordering guarantees:

- Scan (prefix sum): block i needs block i-1's partial result
- Wavefront algorithms: diagonal sweep across a 2D grid
- Pipeline parallelism within a kernel

These have strong inter-block data dependencies that make splitting impractical without fundamental algorithmic changes.

### Sequential Algorithms Expressed as Kernels

Some kernels are inherently sequential despite running on a GPU:
- Linked list traversal
- Recursive tree operations
- Some sorting stages (the comparison/swap phase of bitonic sort is parallel, but merge phases may not be)

If there isn't enough block-level parallelism to exploit, splitting provides no benefit.

### Kernels with Data-Dependent Access Patterns

When memory access depends on data values (not just blockIdx), we can't predict which data each block needs:

```c
int neighbor = graph_edges[blockIdx.x];  // data-dependent
float val = data[neighbor];              // can't predict address
```

Graph neural networks, sparse matrix operations, hash table lookups — all fall into this category. Splitting these risks massive cross-GPU traffic because we can't pre-distribute the data.

---

## 3. Performance Model: When Does Splitting Help?

### The Fundamental Equation

```
T_split = max(T_compute_A, T_compute_B) + T_overhead

T_overhead = T_launch_coordination + T_data_distribution + T_result_merge + T_sync_points * T_per_sync
```

Splitting helps when: `T_split < T_single_gpu`

Which simplifies to: `T_compute / 2 + T_overhead < T_compute`

Or: **`T_overhead < T_compute / 2`**

### Overhead Budget at 100Gbps RDMA

| Overhead Source | Cost | Notes |
|----------------|------|-------|
| Launch coordination | 2-5 us | RDMA signal to remote GPU |
| Data distribution (per GB) | ~80 ms | 100 Gbps = 12.5 GB/s |
| Result merge (small) | 1-5 us | RDMA read + host add |
| Result merge (histogram, 1M bins) | ~100 us | Transfer + elementwise add |
| Grid sync (per point) | 2-5 us | RDMA write + poll |
| PTX blockIdx patch | 0 (done at load time) | Amortized |

### Break-Even Analysis

For a kernel with NO cross-GPU data movement (data pre-distributed):

```
Overhead ≈ 5us (launch) + 5us (result merge) = 10us
Break-even: T_compute > 20us  (kernel must take >20us to benefit from splitting)
```

For a kernel requiring 1GB data replication:

```
Overhead ≈ 5us + 80ms + 5us ≈ 80ms
Break-even: T_compute > 160ms  (kernel must take >160ms)
```

For a kernel with 10 grid-sync points:

```
Overhead ≈ 5us + 10 * 5us + 5us = 60us
Break-even: T_compute > 120us
```

### The Data Movement Tax

This is the elephant in the room. **Network bandwidth is the bottleneck, not compute.**

| Scenario | Network Cost | Compute Saved | Net Benefit |
|----------|-------------|---------------|-------------|
| 1GB input, 100ms compute | 80ms transfer | 50ms saved | **-30ms (WORSE)** |
| 1GB input already distributed, 100ms compute | ~0 | 50ms saved | **+50ms (better)** |
| 10MB input, 100ms compute | 0.8ms transfer | 50ms saved | **+49.2ms (better)** |

**Key insight:** Kernel splitting only helps when data is ALREADY distributed across GPUs, or when the compute-to-data ratio is very high (compute >> transfer time).

This is why graph-level splitting (R13) is superior for most cases — it handles data distribution at the graph level, ensuring each GPU has the data it needs before kernels run.

---

## 4. Existing Work and Prior Art

### cuBLAS-XT (NVIDIA)
- Multi-GPU GEMM through the cuBLAS extended API
- Splits along M or N dimension
- Handles data distribution and result gathering
- Achieves 60-80% efficiency on 2 GPUs with NVLink
- **Not transparent** — application must use the XT API explicitly

### NCCL (NVIDIA Collective Communication Library)
- Multi-GPU collective operations (allreduce, broadcast, etc.)
- NOT kernel splitting — it's communication primitives
- But allreduce IS a split-and-merge pattern at the communication level
- OuterLink's R20 (NCCL Backend) builds on this

### DiHydrogen (LLNL/LBANN)
- Distributed tensor operations for deep learning
- Splits tensor operations across nodes
- Uses NCCL/MPI for communication
- Achieves good scaling for training workloads
- **Not transparent** — requires framework integration

### MGARD (Oak Ridge)
- Multi-GPU data compression with grid decomposition
- Splits spatial grids across GPUs
- Demonstrates that domain decomposition works for structured grids

### CuPy Multi-GPU
- Python GPU array library
- Allows splitting array operations across GPUs
- Simple elementwise/reduction splitting
- **Not transparent** — user must explicitly manage multi-GPU

### Academic: Automatic Kernel Fission (Various Papers)

Several research papers explore automatic kernel splitting:
- **Kernel Fission (Wahib & Maruyama, 2014):** Compiler-based block partitioning for multi-GPU. Achieves 1.5-1.8x speedup on 2 GPUs for stencil codes. Limited to regular access patterns.
- **Multi-GPU CUDA Kernel Execution (Lee et al., 2013):** Runtime block scheduling across GPUs. Showed that independent blocks can be distributed but overhead from page migration was significant.
- **Automatic Data Placement (Cabezas & Perez, 2015):** Compiler analysis determines data placement for multi-GPU. Shows that static analysis CAN identify independent blocks for many kernels.

**Common finding across all papers:** 1.3-1.8x speedup on 2 GPUs is typical. Not 2x. The communication overhead always takes a cut.

---

## 5. The Honest Gap: Ideal vs Achievable

### The Dream (Ideal)
- App launches one kernel, OuterLink transparently splits across N GPUs
- Linear scaling: 2 GPUs = 2x compute
- Zero application changes
- Works for all kernels

### The Reality (Achievable)

| Aspect | Dream | Reality |
|--------|-------|---------|
| Scaling | 2x on 2 GPUs | 1.3-1.8x on 2 GPUs (depending on kernel type) |
| Kernel coverage | All kernels | 40-60% of kernels (the embarrassingly parallel ones) |
| Transparency | Fully transparent | Need PTX JIT path (SASS-only modules won't work) |
| Data movement | Automatic | Only efficient when data is pre-distributed |
| Latency | Invisible | Adds 5-100us per split kernel launch |
| Grid sync | Transparent | 2-5us per sync point — viable only for infrequent sync |

### What We're Really Building

Honest framing: **Cooperative Kernel Splitting is an optimization for the subset of kernels where block-level parallelism is already perfect and data is already distributed.** It is NOT a general solution for making all CUDA code run on multiple GPUs.

The general solution is:
1. **R13 Graph-level splitting** (primary — works for all kernels, splits at kernel granularity)
2. **R25 Kernel-level splitting** (secondary — for the single hottest kernel when graph splitting isn't enough)

---

## 6. Recommended Implementation Approach

### Phase A: Kernel Classification Engine

Build the analysis infrastructure first. No splitting yet — just classify.

1. Intercept `cuModuleLoadData` — capture PTX
2. Parse PTX: extract atomic usage, sync usage, memory access patterns
3. Classify each kernel as GREEN / YELLOW / RED
4. Log classifications — build a dataset of real-world kernel properties
5. **Output:** A table showing what COULD be split if we built the splitter

**Value:** Proves the classification is useful before investing in the splitter. If 90% of kernels are RED, kernel splitting isn't worth building.

### Phase B: Trivial Kernel Splitting (GREEN only)

Split only perfectly independent kernels — no atomics, no sync, blockIdx-linear access.

1. PTX blockIdx offset injection
2. Grid dimension splitting
3. Pointer remapping for distributed data
4. Launch on 2 GPUs, synchronize completion
5. **No merge step** — GREEN kernels don't need it

**Value:** Proves the mechanism works end-to-end. Measures actual overhead.

### Phase C: Atomic-Aware Splitting (YELLOW)

Add support for reduction patterns.

1. Detect reduction atomics in PTX
2. Redirect atomics to per-GPU local copies
3. Merge step after kernel completion
4. Handle histogram patterns

### Phase D: Graph-Integrated Splitting

Combine with R13 for maximum impact.

1. R13 identifies the computation graph
2. Graph-level splitting handles most kernels
3. For the hottest kernel (bottleneck), apply kernel-level splitting
4. Data distribution is planned at the graph level, eliminating the data movement problem

**This is where real value emerges.** The graph context tells us which data is where, which kernels are hot, and where splitting actually helps.

---

## 7. Integration Points with Other Research Topics

| Topic | Integration with R25 |
|-------|---------------------|
| **R13 CUDA Graph Interception** | Graph context is the PRIMARY source of splitting decisions. Graph-level split first, kernel-level split for bottleneck kernels. |
| **R18 Virtual NVLink** | Provides coherency for cross-GPU memory access. Without R18, we can only split independent blocks. With R18, YELLOW/RED kernels become accessible (at performance cost). |
| **R26 PTP Clock Sync** | Synchronized kernel launches across GPUs. Ensures both GPUs start their split portions at the same time. Matters for latency-sensitive workloads. |
| **R17 Topology-Aware Scheduling** | Determines WHICH GPUs to split across. Pick the pair with lowest inter-GPU latency. |
| **R23 Heterogeneous GPU Mixing** | Unequal splits for unequal GPUs. A 3090 gets 60% of blocks, a 3060 gets 40%. Block assignment proportional to SM count * clock speed. |

---

## 8. Fallback: Don't Split — Just Redirect

For kernels that can't be split (RED), the simple strategy: **run the entire kernel on whichever GPU currently has the most relevant data.**

This is what OuterLink already does at the basic level (single GPU assignment per kernel). Kernel splitting is an upgrade path for when that's not enough.

The fallback is important because it means kernel splitting is purely additive — it never makes things worse. If a kernel can't be split, it runs exactly as it would without R25.

---

## 9. Verdict: Is R25 Worth Building?

### Yes, BUT...

1. **Build R13 (Graph Interception) first.** Graph-level splitting covers 80% of the benefit at 20% of the complexity.
2. **Build the classifier (Phase A) early.** It's cheap and tells us how much kernel splitting actually matters for real workloads.
3. **Build the splitter (Phases B-D) only after R13 proves there are kernels where graph splitting isn't enough.**
4. **Target GEMM first.** It's the most common compute-bound kernel, it's perfectly splittable, and cuBLAS-XT proves the approach works.
5. **Don't chase generality.** Supporting 50% of kernels at 70% efficiency is worth far more than supporting 90% of kernels after 2 years of work.

### Expected Impact

| Metric | Without R25 | With R25 |
|--------|------------|----------|
| GEMM throughput (2 GPUs) | 1x (one GPU) | 1.5-1.8x |
| Elementwise throughput | 1x | 1.8-1.9x |
| Attention throughput | 1x | 1.3-1.7x (head/batch split) |
| Total training speedup | 1x (graph split only) | +10-30% (kernel split on bottleneck) |

The +10-30% on top of graph splitting is real value — but it comes after R13, not instead of it.

---

## Open Questions

1. **What's the real distribution of GREEN/YELLOW/RED kernels in PyTorch inference?** This determines if R25 is worth the investment. Need to build the classifier and measure.
2. **Can we detect cuBLAS/cuDNN library calls and use their built-in multi-GPU paths instead?** cuBLAS-XT already does multi-GPU GEMM. If we detect a cuBLAS GEMM call, should we redirect to cuBLAS-XT instead of splitting ourselves?
3. **How to handle kernels that exceed one GPU's SM count in the original grid?** If a kernel launches 10000 blocks and one GPU has 82 SMs, blocks are time-multiplexed. Splitting across 2 GPUs reduces the time-multiplexing. This is actually the BEST case for splitting — there's enough work to saturate both GPUs.
4. **SASS-only modules (no PTX available):** Some libraries ship only precompiled SASS. Can we intercept `cuModuleLoadFatBinary` and strip the SASS to force PTX JIT? Or are we blocked for these kernels?

---

## Related Documents

- [01-cuda-execution-model.md](01-cuda-execution-model.md) — CUDA execution model details
- [02-kernel-splitting-strategies.md](02-kernel-splitting-strategies.md) — Splitting mechanisms
- R13: CUDA Graph Interception — graph-level splitting (build first)
- R18: Virtual NVLink — coherency layer
- R26: PTP Clock Sync — synchronized launches
- R17: Topology-Aware Scheduling — GPU selection for splits
- R23: Heterogeneous GPU Mixing — unequal block assignment
