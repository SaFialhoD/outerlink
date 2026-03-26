# R28 Research: GPU Sparse Data Patterns

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Priority:** MEDIUM

## Purpose

Analyze how sparse and fragmented data actually exists in GPU VRAM during ML workloads. This determines when scatter-gather DMA provides real benefit vs when data is naturally contiguous. Covers CUDA sparse matrix formats, cuSPARSE access patterns, ML sparsity patterns, and VRAM fragmentation from allocation churn.

---

## 1. CUDA Sparse Matrix Formats

NVIDIA's cuSPARSE library supports multiple sparse storage formats, each with different memory layouts:

### CSR (Compressed Sparse Row)

The most common format for sparse linear algebra.

**Memory layout (3 arrays):**
- `row_offsets[num_rows + 1]` — Starting position of each row's non-zeros
- `col_indices[nnz]` — Column index for each non-zero
- `values[nnz]` — The actual non-zero values

**Example:** A 1000x1000 matrix with 1% density (10,000 non-zeros):
- `row_offsets`: 4,004 bytes (1001 x int32)
- `col_indices`: 40,000 bytes (10,000 x int32)
- `values`: 40,000 bytes (10,000 x float32)
- **Total: 84KB** vs 4MB dense — 48x savings

**OuterLink relevance:** Three separate arrays, typically non-contiguous in VRAM. A scatter-gather transfer needs 3 SGEs minimum.

### CSC (Compressed Sparse Column)

Transpose of CSR. Same memory layout, same storage requirements. Used for column-major operations. CSR and CSC of a matrix have identical memory footprints — CSC of A is CSR of A^T.

### COO (Coordinate)

Simplest format: explicit (row, col, value) tuples.

**Memory layout (3 arrays):**
- `row_indices[nnz]` — Row index for each non-zero
- `col_indices[nnz]` — Column index for each non-zero
- `values[nnz]` — The actual non-zero values

**Storage:** 12 bytes per non-zero (3 x int32/float32). More memory than CSR but simpler to construct and merge.

cuSPARSE assumes COO is sorted by row, with sorted or unsorted column indices within rows.

### BSR (Block Sparse Row)

Stores fixed-size dense blocks instead of individual elements. Used when sparsity has block structure (common in deep learning).

**Memory layout:**
- `row_offsets[num_block_rows + 1]`
- `col_indices[nnz_blocks]`
- `values[nnz_blocks * block_dim * block_dim]`

**Example:** Block size 32x32, matrix 1024x1024 with 10% block density:
- 32x32 = 1024 blocks, 10% = ~102 non-zero blocks
- Values array: 102 x 32 x 32 x 4 = ~400KB (contiguous!)
- Indices: small

**OuterLink relevance:** BSR values are often one large contiguous allocation — less need for scatter-gather. The structured block pattern is a good fit for simple RDMA WRITE.

---

## 2. cuSPARSE Memory Access Patterns

### Generic API (Modern cuSPARSE)

Since CUDA 11, cuSPARSE uses a generic API with opaque descriptors:

```c
cusparseCreateCsr(&matA, rows, cols, nnz,
                  d_row_offsets, d_col_indices, d_values,
                  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
```

The three arrays (`d_row_offsets`, `d_col_indices`, `d_values`) are separate cudaMalloc allocations — three independent VRAM regions.

### Buffer Workspace

cuSPARSE operations require external workspace buffers:
```c
cusparseSpMM_bufferSize(handle, ..., &bufferSize);
cudaMalloc(&dBuffer, bufferSize);
cusparseSpMM(handle, ..., dBuffer);
```

These workspace buffers are temporary allocations that contribute to VRAM fragmentation over time.

### Access Pattern Summary

| Operation | Input Arrays | Workspace | Total Separate Regions |
|-----------|-------------|-----------|----------------------|
| SpMV (CSR) | 3 (offsets, indices, values) | 1 temp | 4 |
| SpMM (CSR) | 3 sparse + 1 dense | 1 temp | 5 |
| SpGEMM | 2 x 3 sparse | 2+ temp | 8+ |
| Format conversion | 3 source + 3 target | 1 temp | 7 |

---

## 3. Sparse Patterns in Machine Learning

### Attention Masks (Transformers)

Modern transformers use sparse attention to reduce O(n^2) complexity:
- **Block-sparse attention:** Fixed-size blocks in attention matrix (~50-90% sparsity)
- **Sliding window attention:** Band-diagonal pattern
- **Mixture patterns:** Combination of local + global + random

Sparsity in attention is structural (known at compile time), making it predictable for scatter-gather planning.

### Pruned Models

Weight pruning removes near-zero parameters after training:

| Pruning Method | Typical Sparsity | Pattern |
|----------------|-----------------|---------|
| Magnitude pruning | 80-95% | Unstructured (scattered zeros) |
| Structured pruning | 50-80% | Entire channels/filters removed |
| N:M sparsity (NVIDIA) | 50% (2:4) | Hardware-friendly structured pattern |
| Block pruning | 60-90% | Fixed-size zero blocks |

**NVIDIA 2:4 Structured Sparsity:** Ampere+ GPUs have hardware support for 2:4 patterns (2 non-zeros per 4 elements). This is stored as compressed format with separate index and value arrays — two non-contiguous regions.

### Mixture of Experts (MoE)

MoE models activate only a subset of expert layers per input:
- 8-128 experts total, 1-4 active per token
- Only active expert weights need transfer
- Active experts are non-contiguous in VRAM (stored as separate allocations)
- Scatter-gather is ideal: gather N active expert weight tensors in one transfer

### Gradient Sparsity in Training

During distributed training, gradients are often sparse:
- Top-K sparsification: Only transmit the K largest gradients (typically 0.1-1% of total)
- Threshold sparsification: Only transmit gradients above a magnitude threshold
- Both produce non-contiguous data that benefits from scatter-gather

---

## 4. Quantifying Sparsity in Real Workloads

### Inference Workloads

| Model Type | Typical Sparsity | Data Pattern |
|-----------|-----------------|--------------|
| Dense LLM (LLaMA, GPT) | 0% (weights dense) | Contiguous tensors |
| Pruned LLM (SparseGPT) | 50-60% unstructured | COO/CSR indices + values |
| MoE (Mixtral 8x7B) | 75% per-token (6/8 experts inactive) | Non-contiguous expert blocks |
| Sparse attention (BigBird) | 70-90% of attention matrix | Block-sparse mask + values |
| Quantized (GPTQ, AWQ) | 0% sparse, but mixed precision | Contiguous but narrow |

### Training Workloads

| Scenario | Data Needing Transfer | Non-Contiguous? |
|----------|----------------------|-----------------|
| Full gradient sync | All gradients | No — contiguous |
| Top-K gradient sparsification | 0.1-1% of gradients | Yes — scattered locations |
| MoE expert routing | Active expert params | Yes — separate allocations |
| Checkpoint (activation) | Selected layer activations | Yes — per-layer allocations |

### Key Insight

Dense models with dense operations (the majority of current LLM inference) have mostly contiguous VRAM — scatter-gather provides minimal benefit. The big wins come from:
1. **MoE models** — inherently non-contiguous expert weights
2. **Pruned models** — sparse format arrays
3. **Fragmented VRAM** — after allocation churn (see next section)
4. **Multi-tensor operations** — gathering multiple small tensors for batch transfer

---

## 5. VRAM Fragmentation from Allocation Churn

### The Problem

CUDA's `cudaMalloc`/`cudaFree` pattern causes fragmentation over time:
- Training allocates/frees temporary tensors (activations, gradients) every iteration
- Inference with variable-length inputs allocates different-sized KV caches
- Framework-level caching allocators (PyTorch, TensorFlow) mitigate but don't eliminate

### Measured Impact

Research and production reports indicate:
- **Up to 30% VRAM utilization loss** from fragmentation in long-running training jobs on A100 GPUs
- **50% of free VRAM can be unusable** despite appearing free — scattered small gaps between live allocations
- 70B parameter models can fail on RTX 3090 (24GB) with sufficient total free VRAM due to fragmentation

### Mitigation Strategies (Upstream Frameworks)

| Strategy | How It Works | Fragmentation Reduction |
|----------|-------------|------------------------|
| PyTorch caching allocator | Suballocates from large blocks, reuses freed blocks | ~40% fewer fragments |
| `cudaMallocAsync` (CUDA 11.2+) | Stream-ordered pools with automatic coalescing | Significant reduction |
| RAPIDS Memory Manager (RMM) | Arena allocator, per-stream pools | Best for multi-stream |
| `expandable_segments` (PyTorch 2.4+) | Virtual memory-backed growable segments | Addresses large allocation failures |

### OuterLink Implications

OuterLink's VRAM is managed at the page level (R10, 64KB pages). Fragmentation manifests as:
- Pages from the same logical tensor scattered across VRAM
- Gaps between allocated page runs
- Multiple small allocations that are logically related but physically dispersed

Scatter-gather DMA directly addresses this: instead of copying N fragments to a staging buffer, then doing one RDMA WRITE, we do one RDMA WRITE with N SGEs.

---

## 6. Verdict for OuterLink

### When Scatter-Gather Matters Most

1. **MoE inference** — Gathering 2-4 active expert weight tensors (each 100s of MB) from non-contiguous locations. Huge win: eliminates staging copies.

2. **Sparse model transfer** — CSR/COO arrays (indices + values + offsets) are always non-contiguous. 3 SGEs per sparse matrix is trivial.

3. **Fragmented VRAM recovery** — After extended training sessions, the page table may show significant fragmentation. Scatter-gather lets us transfer fragmented pages without defragmentation.

4. **Multi-tensor batch transfer** — Gathering many small tensors (optimizer states, batch norm params, bias vectors) into one RDMA operation instead of many.

### When Scatter-Gather Doesn't Help

- Dense model weight loading (LLaMA, GPT) — weights are contiguous
- Large contiguous tensor transfers — already optimal with single-SGE RDMA WRITE
- Very small transfers (< 4KB) — overhead of building SGE list exceeds benefit

### Estimated Workload Distribution

For a typical OuterLink cluster running mixed inference:
- ~60% of transfers are contiguous (dense weights, large tensors) — no scatter-gather needed
- ~25% of transfers involve 2-5 non-contiguous regions — scatter-gather sweet spot
- ~15% of transfers involve 6-30 fragments — scatter-gather beneficial
- <1% of transfers exceed 30 fragments — need software pre-pack

---

## Related Documents

- [01-rdma-scatter-gather.md](./01-rdma-scatter-gather.md) — RDMA SGE mechanics and limits
- [03-scatter-gather-pipeline.md](./03-scatter-gather-pipeline.md) — Full pipeline design
- [R10 Memory Tiering](../../../R10-memory-tiering/) — Page-level VRAM management
- [R12 Memory Deduplication](../../../R12-memory-dedup/) — Shared read-only pages (contiguous)

## Open Questions

- [ ] Profile actual VRAM fragmentation on Pedro's 3090s after extended training runs
- [ ] What fraction of Mixtral 8x7B inference transfers are non-contiguous?
- [ ] Does cuSPARSE's generic API guarantee the 3-array layout, or can it pack internally?
- [ ] Impact of PyTorch's caching allocator on OuterLink page-level fragmentation
