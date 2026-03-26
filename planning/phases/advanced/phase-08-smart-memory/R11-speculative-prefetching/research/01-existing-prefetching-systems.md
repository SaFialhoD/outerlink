# R11 Research: Existing Prefetching Systems

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Priority:** HIGH

## Purpose

Survey prefetching mechanisms across CPU hardware, GPU memory systems, networked/parallel storage, and ML frameworks to extract proven patterns and lessons applicable to OuterLink's speculative prefetching design.

## TL;DR — What Works and What OuterLink Should Steal

| Source System | Key Technique | OuterLink Applicability |
|---|---|---|
| CPU stride prefetchers | Per-instruction stride detection via Reference Prediction Table | Per-kernel stride detection on page access sequences |
| NVIDIA UVM | `cudaMemPrefetchAsync` + stream overlap | We intercept CUDA — can inject equivalent prefetch commands |
| AMD HMM/XNACK | Page fault retry + `hipMemPrefetchAsync` | Validates that explicit prefetch beats fault-driven migration |
| CXL DRAM cache | Sub-page prefetch from far memory + bandwidth throttling | Direct analog: prefetch from remote VRAM/DRAM into local |
| Lustre/GPFS | Read-ahead with advisory hints (WILL_NEED) | Application-level hints via CUDA interception |
| PyTorch DataLoader | `pin_memory=True` + `non_blocking=True` + double-buffering | Pinned staging buffers + async transfer overlap |
| NVIDIA DALI | GPU-side pipeline with configurable prefetch queue depth | Prefetch queue model with depth tuning |
| ETC (academic) | Coordinated prefetch + eviction under oversubscription | Coordinate with R10 ARC eviction policy |

**The single biggest lesson:** explicit, ahead-of-time prefetching consistently outperforms fault-driven/on-demand migration by 2-10x. OuterLink's interception layer gives us the ideal vantage point to implement this.

---

## 1. CPU Hardware Prefetchers

### 1.1 Types of Hardware Prefetchers

Modern CPUs (Intel, AMD) implement multiple prefetcher types at different cache levels:

| Prefetcher Type | Location | Mechanism | Accuracy |
|---|---|---|---|
| **L1 IP-based stride** | L1D | Tracks per-load-instruction stride via Reference Prediction Table (RPT) | High for regular loops |
| **L2 stream/sequential** | L2 | Detects forward/backward sequential streams | Very high for linear scans |
| **Adjacent-line** | L1/L2 | Fetches neighboring cache line on miss | High for spatial locality |
| **L2 stride** | L2 | Detects constant-stride patterns across cache lines | Good for array traversals |

### 1.2 How Stride Detection Works

The Reference Prediction Table (RPT) is indexed by program counter. When a load instruction executes:
1. First execution: record address in RPT entry
2. Second execution: compute stride = current_address - previous_address
3. Third execution: if stride matches, begin prefetching address + stride ahead

Intel CPUs detect strides with byte-level granularity. AMD CPUs detect strides in multiples of 4 bytes. Both can handle forward and backward strides.

### 1.3 Prefetch Depth and Throttling

- **Intel:** multi-tier prefetching across L1, L2, and LLC with independent prefetchers at each level
- **AMD Zen 5:** dynamic prefetch depth scaling to avoid wasting bandwidth when accuracy is low
- Both architectures throttle prefetching when bandwidth utilization is high or accuracy drops

### 1.4 Limitations

- Non-strided patterns (linked lists, indirect arrays, hash tables) defeat hardware prefetchers
- Prefetchers at L1 have high accuracy but expensive metadata storage
- LLC prefetchers see filtered access streams, reducing prediction accuracy
- Security concerns: recent research (Augury, GoFetch, ZenLeak) shows prefetcher state can leak information

### 1.5 Relevance to OuterLink

OuterLink operates at page granularity (64KB), not cache-line granularity (64B). CPU stride detection maps directly: instead of tracking per-instruction strides on cache lines, we track per-kernel strides on page accesses. The RPT concept translates to a Kernel Prediction Table indexed by kernel signature.

---

## 2. GPU Prefetching

### 2.1 NVIDIA Unified Virtual Memory (UVM)

UVM provides a single address space across CPU and GPU with on-demand page migration:

| Metric | Value |
|---|---|
| Page fault overhead | ~10-50 us per fault |
| Fault batch size | Up to 256 faults per warp |
| On-demand streaming bandwidth | ~5.4 GB/s (PCIe) |
| Prefetched bandwidth | ~10.9 GB/s (PCIe) |
| Explicit memcpy bandwidth | ~11.4 GB/s (PCIe) |
| Performance penalty (no prefetch) | 1.5x-2.4x vs explicit CUDA |
| Performance penalty (with prefetch) | 1.2x-1.3x vs explicit CUDA |

Key APIs:
- `cudaMemPrefetchAsync(ptr, size, device, stream)` — explicit prefetch to device
- `cudaMemAdvise(ptr, size, advice, device)` — hint about access patterns (READ_MOSTLY, PREFERRED_LOCATION, ACCESSED_BY)

Volta introduced **hardware access counters** that track remote accesses to pages, helping detect thrashing.

### 2.2 AMD HMM/XNACK

AMD's approach uses Linux Heterogeneous Memory Management (HMM) kernel infrastructure:

- **XNACK:** GPU hardware retry mechanism for page faults (MI200+, CDNA cards only)
- Without XNACK: managed memory acts as pinned host memory (PCIe access every time) — **up to 4000% performance degradation**
- With XNACK but no prefetching: page fault retry works but still slow
- With `hipMemPrefetchAsync`: degradation reduced to ~200% vs explicit

AMD's data validates that prefetching is mandatory — fault-driven migration alone is catastrophically slow. Consumer GPUs (RDNA) do not support XNACK, making explicit prefetching the only viable strategy for consumer hardware. This is directly relevant to OuterLink targeting GeForce GPUs.

### 2.3 GPU-Internal Prefetching (GPGPU Architecture)

Within GPU execution, warps are the primary latency-hiding mechanism (not prefetching). However, for memory-bound kernels where warp occupancy is insufficient:

- Software prefetching in CUDA requires explicit `__prefetch_l1()` / `__prefetch_l2()` intrinsics
- GPU hardware prefetchers exist but are limited compared to CPU (silicon budget goes to compute units)
- Rolling prefetch pattern: fill buffer before loop, prefetch one value per iteration for use PDIST iterations later

### 2.4 Academic: Deep Learning Prefetch with Transformers

Long et al. (2022) applied transformer models to predict UVM page access sequences:
- Trained a transformer to predict next-page-to-access from historical fault sequences
- Achieved 10.89% performance improvement over baseline UVM
- Improved device memory page hit rate by 16.98%
- Reduced CPU-GPU interconnect traffic by 11.05%

An intelligent oversubscription framework (2023) combined access pattern classification with pattern-specific transformer models, reducing page thrashing by 64.4% under 125% memory oversubscription.

### 2.5 GPUVM (RDMA-Based GPU Virtual Memory)

GPUVM constructs a virtual memory system using RDMA-capable NICs without CPU involvement:
- GPU threads manage page migration directly
- On-demand paging via RDMA for remote pages
- Achieves up to 4x performance over UVM for latency-bound applications
- Directly relevant to OuterLink's OpenDMA path (Phase 5)

---

## 3. Networked/Parallel Storage Prefetching

### 3.1 Lustre

- **Read-ahead:** Automatic for large sequential file reads; prefetches entire stripes at a time for small files
- **ladvise API (2.9+):** Client hints to server for prefetching (`WILL_NEED`) or cache eviction (`DONT_NEED`)
- **Statahead (2.16+):** Prefetches file attributes in parallel for directory traversals using batch RPCs
- **Effectiveness of hints:** POSIX `fadvise(WILL_NEED)` on Lustre does not trigger effective prefetch; explicit `ladvise` or async I/O with 1ms delay achieves up to 3x speedup
- **Cache location:** Uses Linux page cache on compute nodes

### 3.2 GPFS (IBM Spectrum Scale)

- **Internal page pool:** 4GB fixed-size buffer pool separate from Linux page cache
- **Tunable parameters:** `prefetchPct` (percentage of pool for prefetch), `prefetchThreads` (parallel prefetch workers)
- **Read-ahead:** Parallel I/O requests with configurable aggressiveness
- **No direct Linux page cache integration** — manages its own caching entirely

### 3.3 NFS

- Standard Linux NFS client uses kernel read-ahead (configurable via `readahead` mount option)
- Default read-ahead window starts small and grows with sequential access detection
- NFSv4.1+ parallel NFS (pNFS) can stripe reads across multiple servers

### 3.4 CXL Far Memory Prefetching

CXL memory pooling (CXL 3.2, December 2024) faces the same latency challenge as OuterLink:

| Configuration | Latency |
|---|---|
| Local DRAM | ~80-100 ns |
| CXL fabric (ASIC) | ~2.3x local DRAM (~185-230 ns) |
| CXL fabric (FPGA prototype) | 70-210 ns added |

Tirumalasetty & Annapareddy (2024) proposed DRAM cache prefetching for pooled CXL memory:
- **Sub-page block prefetching** from fabric-attached memory (FAM) into local DRAM cache
- **LLC miss observation:** Uses LLC misses visible at root-complex level to trigger prefetches
- **Bandwidth throttling:** Weighted fair queuing (WFQ) between demand and prefetch requests
- **Adaptive rate:** Adjusts prefetch rate based on observed FAM access latency
- **Results:** 7% IPC improvement baseline; 7-10% additional with contention-aware optimizations
- At 50% FAM allocation, DRAM prefetch limited performance loss to just 5% vs all-local

This is the closest architectural analog to OuterLink: local memory as cache for remote memory, with prefetching to hide transfer latency.

### 3.5 Lessons for OuterLink

| Lesson | Source | Application |
|---|---|---|
| Advisory hints alone are unreliable | Lustre/GPFS research | Don't rely on application hints; instrument at interception layer |
| Dedicated prefetch buffer pool | GPFS | Separate prefetch staging area in pinned RAM |
| Bandwidth throttling is essential | CXL research | Budget bandwidth between demand and prefetch transfers |
| Sub-page granularity helps | CXL DRAM cache | Consider 4KB sub-pages within 64KB pages for partial prefetch |
| Parallel prefetch workers | GPFS | Dedicated prefetch threads/async tasks |

---

## 4. ML Framework Prefetching

### 4.1 PyTorch DataLoader

Standard pipeline:
```
CPU: Load batch N → Preprocess → Pin memory
GPU: Train on batch N-1
Transfer: batch N to GPU (overlapped with batch N-1 training)
```

Key mechanisms:
- `pin_memory=True`: allocates page-locked (pinned) host memory for DMA-friendly transfers
- `non_blocking=True` on `.cuda()`: returns immediately, transfer runs on a CUDA stream
- `prefetch_factor`: controls how many batches CPU workers prepare ahead (CPU-side only)
- `num_workers`: parallel CPU data loading processes

Limitation: requires extra VRAM for the prefetched batch. If VRAM is maxed out, prefetching is impossible without reducing batch size.

### 4.2 NVIDIA DALI

DALI moves the entire data pipeline to GPU:
- Decoding, augmentation, and preprocessing run on GPU
- Asynchronous execution engine with configurable `prefetch_queue_depth`
- Transparent prefetching: next batch prepared while current batch trains
- Achieves near-100% GPU utilization vs PyTorch DataLoader's periodic drops to 0%

Architecture:
```
Storage → CPU decode (optional) → GPU pipeline → prefetch queue → Training
                                       ↑
                               DALI execution engine
                           (parallel, async, pipelined)
```

### 4.3 DNN Memory Predictability

A critical insight for OuterLink: DNN computation is inherently predictable.

- Training iterations repeat the same computation graph with different data
- The k-th memory allocation corresponds to the operator at position k mod |V| in the execution sequence
- Memory access patterns are identical across iterations — only data values change
- Systems like TSPLIT and LightSeq exploit this by pre-planning all memory operations before training starts

This means OuterLink can:
1. **Profile iteration 1** to learn the complete memory access sequence
2. **Prefetch perfectly on iteration 2+** using the learned sequence
3. Achieve near-zero stalls for steady-state training

### 4.4 Specific Workload Patterns

| Workload | Pattern | Predictability |
|---|---|---|
| **Transformer training** | Attention: quadratic memory growth; repeated across layers | Very high — layer sequence is fixed |
| **CNN training** | Conv layers: regular strided access on activation tensors | Very high — same access pattern per layer |
| **GAN training** | Generator + discriminator alternate; each internally predictable | High — two alternating predictable phases |
| **Inference** | Single forward pass, no backward; simpler pattern | Very high |

---

## 5. Key Papers and Systems Summary

| Paper/System | Year | Key Contribution | Relevance |
|---|---|---|---|
| Intel/AMD CPU prefetchers | Ongoing | Stride detection, RPT, multi-level prefetch | Pattern detection architecture |
| NVIDIA UVM | 2016+ | `cudaMemPrefetchAsync`, access counters (Volta+) | API model for explicit prefetch |
| AMD HMM/XNACK | 2020+ | Fault retry + prefetch; validates prefetch necessity | Confirms fault-driven is too slow |
| Long et al. | 2022 | Transformer-based UVM page prediction | ML-based prediction viable |
| Oversubscription framework | 2023 | Pattern classifier + transformer + policy engine | Architecture for intelligent prefetch |
| ETC | 2020 | Coordinated prefetch + eviction under oversubscription | Prefetch/eviction coordination |
| CXL DRAM cache prefetch | 2024 | Sub-page prefetch, bandwidth throttling, WFQ | Closest architectural analog |
| GPUVM | - | RDMA-based GPU paging without CPU | OpenDMA-path prefetch model |
| Jog et al. (ISCA 2013) | 2013 | Orchestrated GPU scheduling + prefetching | Warp/prefetch coordination |
| DALI | 2018+ | GPU-accelerated data pipeline with prefetch queue | Queue-depth model |

---

## 6. Verdict: OuterLink Prefetching Architecture

Based on this survey, OuterLink's prefetching should combine:

1. **Interception-based profiling** (like CUPTI but zero-overhead via our LD_PRELOAD hooks)
2. **Per-kernel stride detection** (CPU RPT pattern adapted to page granularity)
3. **Iteration-aware replay** (ML training repeats — learn once, prefetch forever)
4. **Explicit async prefetch** (like `cudaMemPrefetchAsync` but across network tiers)
5. **Bandwidth-throttled scheduling** (CXL WFQ pattern between demand and prefetch)
6. **Coordinated eviction** (work with R10's ARC policy, not against it)
7. **Double-buffering** (DALI/PyTorch pattern — always have next batch ready in staging)

The interception layer is OuterLink's unfair advantage: we see every CUDA call before it happens, giving us prediction data that UVM and HMM systems can only dream of.

---

## Related Documents

- [R10 Memory Tiering](../../phase-07-memory-intelligence/R10-memory-tiering/README.md) — tier definitions, page table, eviction policy
- [R11 Preplan](../preplan.md) — scope, decisions, implementation phases
- [02-access-pattern-profiling.md](./02-access-pattern-profiling.md) — how to detect patterns
- [03-prefetch-scheduling.md](./03-prefetch-scheduling.md) — when and where to prefetch

## Open Questions

- [ ] Can NVIDIA hardware access counters (Volta+) be read from userspace without CUPTI? If so, we could use them for zero-overhead access tracking.
- [ ] Does GPUVM's RDMA-based paging approach apply to ConnectX-5 with OpenDMA?
- [ ] How does CXL sub-page prefetching compare to full-page prefetching at 100Gbps? Is partial-page transfer worth the metadata overhead?
