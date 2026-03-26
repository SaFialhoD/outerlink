# R11 Research: Access Pattern Profiling

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Priority:** HIGH

## Purpose

Determine how OuterLink can detect and classify GPU memory access patterns with minimal overhead, using its unique position as a CUDA interception layer. This drives the prediction engine that powers speculative prefetching.

## TL;DR — Our Unfair Advantage

OuterLink intercepts every CUDA Driver API call via LD_PRELOAD. This gives us **free** profiling data that other systems (UVM, HMM) cannot access without instrumentation overhead:

| Data Source | Overhead | Information Quality |
|---|---|---|
| CUDA interception hooks (our LD_PRELOAD) | ~0 (already on the call path) | Kernel launches, memcpy src/dst/size, allocation patterns |
| R10 page table access bits | ~0 (maintained anyway) | Which pages accessed, when, by which kernel |
| Kernel argument inspection | Low (parse known struct layouts) | Tensor addresses, dimensions, strides |
| CUPTI Activity API | Medium (5-15% overhead) | Detailed per-warp memory access traces |
| Hardware performance counters | Low-Medium | Cache miss rates, memory throughput |
| Shadow page table dirty bits | Low | Write pattern detection |

**Recommendation:** Use interception hooks + page table metadata as primary (zero overhead). Use CUPTI only during an optional profiling warmup pass.

---

## 1. CUDA Interception Hooks — What We Already See

OuterLink intercepts 222+ CUDA Driver API functions. The following are directly relevant to access pattern profiling:

### 1.1 Memory Operations

| Intercepted Call | What It Reveals |
|---|---|
| `cuMemAlloc` / `cuMemAllocManaged` | Allocation address, size — defines the memory map |
| `cuMemFree` | Deallocation — page can be reclaimed |
| `cuMemcpyHtoD` / `cuMemcpyDtoH` | Direction, src/dst address, size — explicit data movement |
| `cuMemcpyDtoD` | Intra-GPU or inter-GPU copy — data reorganization |
| `cuMemcpyAsync` variants | Same but on a stream — indicates overlap intent |
| `cuMemsetD*` | Initialization pattern — pages being zeroed are about to be written |
| `cuMemPrefetchAsync` | Application-level hint — app already knows what it needs |
| `cuMemAdvise` | Access pattern hints (READ_MOSTLY, PREFERRED_LOCATION) |

### 1.2 Kernel Launches

| Intercepted Call | What It Reveals |
|---|---|
| `cuLaunchKernel` | Function pointer, grid/block dims, shared mem, stream, **kernel args** |
| `cuLaunchCooperativeKernel` | Same + cooperative launch semantics |
| `cuGraphLaunch` | Graph execution — entire DAG of operations known ahead of time |

**Kernel arguments are the gold mine.** For known ML frameworks (PyTorch, TensorFlow), kernel arguments contain pointers to input/output tensors. By tracking which device pointers appear in kernel args, we know exactly which memory regions each kernel will access.

### 1.3 Synchronization Points

| Intercepted Call | What It Reveals |
|---|---|
| `cuStreamSynchronize` | Iteration boundary — everything before this is one logical step |
| `cuCtxSynchronize` | Hard synchronization — phase boundary |
| `cuEventSynchronize` | Fine-grained dependency point |
| `cuStreamWaitEvent` | Stream dependency — ordering constraint |

Synchronization calls are natural **epoch markers**. ML training typically follows: `launch kernels → sync → next iteration`. This gives us clean iteration boundaries for pattern detection.

---

## 2. Page Table Access Tracking (R10 Integration)

R10's page table already tracks per-page metadata in 64-byte PTEs:

| PTE Field | Profiling Use |
|---|---|
| `access_count` | Hot/cold page classification |
| `last_access_timestamp` | Recency tracking for LRU/ARC |
| `tier` | Current location (local VRAM, remote VRAM, DRAM, NVMe) |
| `dirty_bit` | Write detection — read-only pages are safer to prefetch speculatively |

### 2.1 Access Bit Sampling

Rather than tracking every access (which requires MMU-level hooks), we can sample access patterns:

1. **Periodic scan:** Every N milliseconds, scan the page table for pages with incremented access counts
2. **Delta tracking:** Record which pages had access_count changes between scans
3. **Sequence construction:** Build ordered list of accessed pages per kernel invocation

Cost: O(active_pages) per scan. With 64KB pages and 24GB VRAM, that's ~375K entries max — scannable in <1ms.

### 2.2 Kernel-to-Page Mapping

By correlating kernel launch timestamps with page access timestamp changes:

```
t=100: cuLaunchKernel(matmul, args=[ptr_A=0x1000, ptr_B=0x5000, ptr_C=0x9000])
t=100-105: Pages 0x1000-0x1FFF, 0x5000-0x5FFF accessed (reads)
t=105: Pages 0x9000-0x9FFF accessed (writes, dirty bit set)
```

This gives us a per-kernel page access profile without any GPU-side instrumentation.

---

## 3. Kernel Argument Inspection

### 3.1 How It Works

`cuLaunchKernel` receives kernel arguments as a `void**` array. For known kernel signatures (GEMM, convolution, attention), we can parse these arguments to extract tensor metadata:

| Argument Type | Example | What We Learn |
|---|---|---|
| Device pointer | `0x7f1234560000` | Base address of tensor in device memory |
| Dimensions | `M=4096, N=4096, K=1024` | Total memory footprint: M*K + K*N + M*N elements |
| Stride | `lda=4096` | Memory access stride (contiguous vs strided) |
| Data type | `CUDA_R_16F` | Bytes per element — total transfer size |
| Alpha/Beta scalars | `alpha=1.0, beta=0.0` | Whether output accumulates (beta != 0) or overwrites |

### 3.2 Framework-Specific Patterns

For PyTorch (the dominant ML framework), CUDA kernels follow predictable patterns:

**Forward pass:**
```
Kernel sequence: [embedding_lookup, layernorm, qkv_projection, attention, ffn, layernorm, ...]
Each kernel: reads input tensor(s), writes output tensor(s)
Output of kernel N = input of kernel N+1 (pipeline)
```

**Backward pass:**
```
Reverse order: [ffn_backward, attention_backward, qkv_backward, layernorm_backward, ...]
Each kernel: reads saved activations + gradients, writes gradient tensors
```

**Optimizer step:**
```
For each parameter: [adam_update(param, grad, momentum, variance)]
Sequential scan through all parameters
```

### 3.3 Kernel Signature Database

OuterLink can maintain a database of known kernel signatures (cuBLAS GEMM variants, cuDNN convolution, etc.) to automatically parse arguments. Unknown kernels fall back to pointer-range-based tracking.

| Library | Key Kernels | Argument Structure |
|---|---|---|
| cuBLAS | `cublasGemmEx`, `cublasSgemm` | A, B, C pointers + M, N, K + strides |
| cuDNN | `cudnnConvolutionForward` | Input, filter, output descriptors + pointers |
| CUTLASS | Template-based GEMM | Similar to cuBLAS but kernel-fused |
| Custom | Framework-specific fused ops | Require framework-specific parsing or fallback to pointer tracking |

---

## 4. CUPTI-Based Profiling (Optional Warmup)

### 4.1 Activity API for Memory Tracing

CUPTI's Activity API provides asynchronous trace collection with structured records:

- `CUpti_ActivityMemcpy` — records every memcpy with src, dst, size, duration
- `CUpti_ActivityKernel` — records kernel execution with timing, grid/block dims
- `CUpti_ActivityMemory4` — tracks memory allocations with source information
- Unified Memory trace: page faults (CPU and GPU), migration events, thrashing

### 4.2 Overhead Characteristics

| CUPTI Feature | Overhead | Notes |
|---|---|---|
| Activity record collection (memcpy, kernel) | 2-5% | Asynchronous buffering, low impact |
| PC sampling (continuous mode) | 3-8% | Does not serialize kernels (new API) |
| Detailed per-kernel metrics | 10-30% | Requires kernel replay — too expensive |
| SASS-level metrics | 5-10% | Offline correlation reduces runtime cost |
| Selective API tracing | 1-3% | Only trace calls of interest |

### 4.3 Recommended CUPTI Usage

Use CUPTI only during an initial profiling warmup (first 1-3 training iterations):

1. Enable Activity API for memcpy + kernel records
2. Collect full trace of iteration 1
3. Build kernel execution graph and memory access map
4. Disable CUPTI
5. Use interception-only tracking for iterations 2+

This gives detailed initial profiling at controlled cost, then zero overhead for steady state.

### 4.4 Buffer Management for Low Overhead

CUPTI uses a callback-based buffer system. Best practices:
- Pre-allocate a pool of 1-10 MB activity buffers
- Return from callbacks as fast as possible (don't process in-line)
- Use the CUPTI worker thread for background processing
- Set `cuptiActivityFlushPeriod` to batch flushes instead of per-event
- Use `cuptiActivityEnableDriverApi` to trace only memory-related APIs

---

## 5. Hardware Performance Counters

### 5.1 Available GPU Counters (Nsight Compute)

| Counter | What It Measures |
|---|---|
| `dram__bytes_read` | Total bytes read from device memory |
| `dram__bytes_write` | Total bytes written to device memory |
| `lts__t_sectors_srcunit_tex_op_read` | L2 read sectors |
| `lts__t_sector_hit_rate` | L2 cache hit rate |
| `sm__warps_active` | Active warps (occupancy indicator) |

### 5.2 NVIDIA Access Counters (Volta+)

Volta introduced hardware access counters specifically for UVM:
- Track remote memory accesses per page
- Used by the UVM driver to detect thrashing
- Trigger migration when access count exceeds threshold

**Open question:** Can these be read from userspace without the UVM driver? If so, they provide zero-overhead per-page access counts.

### 5.3 Practical Limitations

- Reading GPU performance counters requires CUPTI or Nsight, adding overhead
- Counters are aggregate (per-SM or per-kernel), not per-page
- Cannot directly map counter values to specific memory addresses
- Useful for detecting memory-bound vs compute-bound phases, not specific page access patterns

**Verdict:** Hardware counters are supplementary, not primary. Use them to detect when the workload is memory-bound (and thus prefetching is valuable) vs compute-bound (prefetching unnecessary).

---

## 6. Real ML Workload Access Patterns

### 6.1 Transformer Training (e.g., GPT, BERT, LLaMA)

```
Iteration Structure:
  Forward: embedding → [attention → FFN → layernorm] × L layers → loss
  Backward: loss_grad → [layernorm_grad → FFN_grad → attention_grad] × L layers → embedding_grad
  Optimizer: adam_step(param, grad, state) for each parameter

Memory Pattern:
  - Forward activations: written once, read twice (forward use + backward recompute/read)
  - Weights: read every forward + backward pass, written once per optimizer step
  - Gradients: written during backward, read during optimizer step, then discarded
  - Optimizer state (momentum, variance): read + written once per step

  Total unique tensors per layer: ~6-8 (Q, K, V projections, attention output, FFN weights x2, biases, layernorm params)
  Access order: perfectly sequential through layers (forward), then reverse (backward)
```

**Predictability: VERY HIGH.** After seeing one iteration, the next iteration's access pattern is identical with >99% accuracy. Only data parallelism shuffling and dynamic batching introduce minor variations.

### 6.2 CNN Training (e.g., ResNet, EfficientNet)

```
Iteration Structure:
  Forward: input → [conv → batchnorm → relu → pool] × N stages → FC → loss
  Backward: reverse order
  Optimizer: SGD/Adam step

Memory Pattern:
  - Convolution weights: regular strided access on 4D tensors (N, C, H, W)
  - Activations: written forward, stored for backward (or recomputed via gradient checkpointing)
  - Access stride: determined by filter size, stride, and padding

  Very regular, fixed access patterns per layer.
```

**Predictability: VERY HIGH.** Even more regular than transformers because convolution access patterns are fully determined by layer configuration.

### 6.3 GAN Training

```
Iteration Structure:
  Phase 1 (Discriminator): real_batch → D → loss_real; fake_batch → G → D → loss_fake
  Phase 2 (Generator): noise → G → D → loss_G

Memory Pattern:
  - Two alternating compute phases with different memory access sets
  - G weights: read in phase 1 (no grad), read+written in phase 2
  - D weights: read+written in phase 1, read in phase 2 (no grad)
  - Phase boundaries are detectable via different kernel sequences
```

**Predictability: HIGH.** Two-phase pattern is stable across iterations. Within each phase, access is sequential and predictable.

### 6.4 Inference

```
Single forward pass through the model.
Weights: read-only.
Activations: ephemeral, can be overwritten.
KV cache (transformers): grows with sequence length, then resets.
```

**Predictability: VERY HIGH for weights (always same). MEDIUM for KV cache** (depends on generation length).

---

## 7. Pattern Classification Taxonomy

Based on the above analysis, OuterLink should classify detected patterns into categories:

| Pattern Class | Description | Detection Method | Prefetch Strategy |
|---|---|---|---|
| **Sequential** | Pages accessed in address order (e.g., tensor scan) | Stride=1 in page sequence | Stream prefetch: next N pages |
| **Strided** | Constant stride between accesses (e.g., column access on row-major tensor) | Constant delta between page addresses | Stride prefetch: page + k*stride for k=1..N |
| **Iteration-repeat** | Same page sequence repeats (e.g., ML training loop) | Sequence similarity between iterations | Full replay: prefetch entire next-iteration sequence |
| **Phase-alternating** | Two or more distinct phases alternate (e.g., GAN, forward/backward) | Kernel sequence clustering | Phase-aware: identify current phase, prefetch that phase's pages |
| **Random** | No detectable pattern (e.g., hash table, graph traversal) | High entropy in page sequence | No prefetch (demand-only); could hint to keep in fastest tier |
| **Read-heavy** | Pages mostly read, rarely written | Low dirty-bit rate | Aggressive prefetch (safe — no coherence issues) |
| **Write-heavy** | Pages frequently written | High dirty-bit rate | Conservative prefetch (need write-back coordination) |

---

## 8. Profiling Architecture Design

### 8.1 Data Flow

```
                    CUDA Application
                         |
                    LD_PRELOAD hooks
                         |
              ┌──────────┴──────────┐
              |                     |
       API Call Logger        Kernel Arg Parser
       (cuMemcpy, cuLaunch)   (tensor addresses,
                               dimensions, strides)
              |                     |
              └──────────┬──────────┘
                         |
                  Access Pattern Tracker
                  (per-kernel page access sets,
                   sequence builder, stride detector)
                         |
                  Pattern Classifier
                  (sequential, strided, iteration-repeat,
                   phase-alternating, random)
                         |
                  Prediction Engine
                  (next-access predictor based on
                   classified pattern)
                         |
                  Prefetch Scheduler
                  (see 03-prefetch-scheduling.md)
```

### 8.2 Key Data Structures

**Kernel Access Profile:**
```rust
struct KernelAccessProfile {
    kernel_id: KernelSignature,      // hash of function + grid/block dims
    input_pages: Vec<PageId>,        // pages read by this kernel
    output_pages: Vec<PageId>,       // pages written by this kernel
    execution_time_us: u64,          // how long this kernel takes
    invocation_count: u64,           // how many times we've seen it
    last_invocation: Timestamp,
}
```

**Iteration Profile:**
```rust
struct IterationProfile {
    kernel_sequence: Vec<KernelSignature>,  // ordered list of kernels in this iteration
    total_pages_read: usize,
    total_pages_written: usize,
    total_bytes_transferred: u64,
    duration_us: u64,
    detected_pattern: PatternClass,
}
```

**Pattern Predictor:**
```rust
enum PrefetchPrediction {
    Sequential { base_page: PageId, count: usize },
    Strided { base_page: PageId, stride: i64, count: usize },
    Replay { pages: Vec<PageId> },       // full iteration replay
    PhaseSwitch { phase: PhaseId },      // switch to different phase's prediction
    None,                                 // unpredictable — don't prefetch
}
```

### 8.3 Overhead Budget

| Component | Target Overhead | Rationale |
|---|---|---|
| API call logging | <1 us per call | Just timestamp + ID, no parsing |
| Kernel arg parsing | <5 us per launch | Only for known signatures |
| Page table scan | <1 ms per scan | 375K entries at 64 bytes = 24MB scan |
| Pattern classification | <100 us per iteration | Simple stride/sequence detection |
| Total per-iteration | <2 ms | Negligible vs typical iteration time (50-500 ms) |

---

## Related Documents

- [01-existing-prefetching-systems.md](./01-existing-prefetching-systems.md) — survey of prefetching across domains
- [03-prefetch-scheduling.md](./03-prefetch-scheduling.md) — when and where to prefetch
- [R10 Memory Tiering](../../phase-07-memory-intelligence/R10-memory-tiering/README.md) — page table structure and PTE fields
- [R8 Kernel Param Introspection](../../../../research/R8-kernel-param-introspection.md) — kernel argument parsing

## Open Questions

- [ ] Can we intercept `cuGraphLaunch` to get the entire execution DAG ahead of time? This would give perfect prefetch predictions for graph-captured workloads.
- [ ] What is the overhead of CUPTI Activity API with selective tracing on production workloads? Need to benchmark on actual transformer training.
- [ ] For custom/unknown kernels, can we infer read vs write from cuMemcpy patterns before/after the kernel? (If kernel output is always copied out immediately, the destination pages were writes.)
- [ ] How do gradient checkpointing strategies (recomputation vs storage) affect access pattern predictability? Recomputation makes forward-pass patterns appear twice.
- [ ] Should we maintain separate pattern models for forward pass, backward pass, and optimizer step, or one unified model per iteration?
