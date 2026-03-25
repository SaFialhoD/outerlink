# Heterogeneous GPU Scheduling

**Created:** 2026-03-25
**Updated:** 2026-03-25
**Status:** Draft
**Purpose:** Survey scheduling strategies for distributing work across GPUs with different compute capabilities, memory sizes, and bandwidth characteristics.

## 1. The Core Challenge

In a heterogeneous GPU pool, naive scheduling (round-robin, equal partitioning) wastes resources. An RTX 5090 is ~8x faster than an RTX 3060 in FP32 throughput. Assigning them equal work means the 5090 idles while the 3060 finishes. The scheduler must understand asymmetry and allocate work proportionally.

Three dimensions of asymmetry matter:
1. **Compute throughput** (TFLOPS, SM count, clock speed)
2. **Memory bandwidth** (GB/s — dominates memory-bound workloads)
3. **Memory capacity** (VRAM size — determines what fits)

## 2. Performance Normalization Approaches

### 2.1 FLOPS-Based Normalization

**Approach:** Rate each GPU by theoretical peak FLOPS. Assign work proportional to FLOPS rating.

| Pros | Cons |
|------|------|
| Simple to compute from specs | Theoretical != achieved |
| Available before any profiling | Ignores memory bandwidth entirely |
| Stable metric (doesn't change) | Doesn't account for kernel-specific behavior |

**Formula:** `weight_i = TFLOPS_i / sum(TFLOPS_all)`

**Verdict:** Useful as a starting heuristic. Insufficient alone because many workloads are memory-bandwidth-bound, not compute-bound.

### 2.2 Benchmark-Based Normalization (Profiling)

**Approach:** Run representative micro-benchmarks on each GPU to measure actual throughput for different workload classes.

Benchmark categories:
- **Compute-bound:** Dense matrix multiply (SGEMM)
- **Memory-bound:** Stream copy / triad
- **Mixed:** Convolution, attention kernel
- **Transfer:** Host-to-device, device-to-host throughput

Each GPU gets a performance vector, not a single number:
```
gpu_score = {
    compute: measured_gflops,
    memory: measured_bw_gbps,
    transfer: measured_h2d_gbps,
}
```

**Verdict:** Most accurate. Requires a calibration phase at startup. OuterLink should run lightweight benchmarks when a GPU first joins the pool and cache results keyed by (GPU model, driver version).

### 2.3 Workload-Specific Normalization (Gavel's "Effective Throughput")

**Approach:** Profile each (job, GPU type) pair to determine actual throughput. The ratio between GPU types varies per workload.

Example from Gavel research: training ResNet-50 may be 2.1x faster on V100 vs P100, but training Transformer may be 3.5x faster on V100 vs P100. The speedup ratio is unpredictable without profiling.

**Gavel's method:** Express scheduling policies as optimization problems, then transform them into heterogeneity-aware versions using measured "effective throughput" matrices.

**Verdict:** Ideal for long-running ML training jobs. Overkill for OuterLink's initial use case (general CUDA app interception), but the concept of workload-class-specific scoring should inform the design.

### 2.4 Recommended Approach for OuterLink

**Hybrid normalization:**
1. **Static profile** from GPU attributes (TFLOPS, bandwidth, VRAM) — available immediately
2. **Calibration benchmarks** at GPU registration — run once, cache per GPU model
3. **Runtime telemetry** — observe actual kernel execution times, update weights over time

The scheduler starts with static weights, refines with benchmark data, and continuously adjusts based on observed performance. This avoids cold-start problems while converging to accurate scheduling.

## 3. Task-to-GPU Matching

### 3.1 Workload Classification

Before placing a kernel, classify its bottleneck:

| Workload Type | Bottleneck | Best GPU Trait | Example |
|--------------|-----------|---------------|---------|
| Compute-bound | ALU throughput | High SM count, high clock | Matrix multiply, crypto |
| Memory-bound | DRAM bandwidth | High memory bandwidth | LLM inference (decode), reduction |
| Capacity-bound | VRAM size | Large VRAM | Large model weights, large batch |
| Transfer-bound | PCIe/network | High PCIe BW | Frequent host-device sync |
| Tensor-bound | Tensor Core throughput | Latest Tensor Cores | FP8/FP16 matrix ops |

### 3.2 Matching Strategy

```
For each kernel/task:
  1. Classify workload type (from kernel analysis or user hints)
  2. Filter GPUs by hard constraints:
     - Compute capability >= kernel requirement
     - VRAM free >= allocation requirement
  3. Score remaining GPUs by relevant metric:
     - Compute-bound → score by TFLOPS
     - Memory-bound → score by bandwidth
     - Capacity-bound → score by VRAM available
     - Tensor-bound → score by tensor core generation + throughput
  4. Apply locality bonus (data already on GPU → prefer it)
  5. Apply load factor (busy GPU → penalty)
  6. Select highest-scoring GPU
```

### 3.3 Interaction with R17 Topology-Aware Scheduling

R17 handles network topology and data placement. R23 adds GPU capability scoring on top. The combined scoring function:

```
total_score = w1 * capability_score    // R23: GPU fitness for this workload
            + w2 * locality_score      // R17: data proximity
            + w3 * load_score          // inverse of current utilization
            + w4 * network_score       // R17: network path quality
```

The weights (w1-w4) should be tunable and may vary by workload class.

## 4. Existing Heterogeneous GPU Schedulers

### 4.1 Academic Systems

| System | Year | Key Idea | Applicability to OuterLink |
|--------|------|----------|---------------------------|
| **Gavel** (OSDI 2020) | 2020 | Effective throughput matrix, heterogeneity-aware policy transformation | Core concept of profiling per-(job, GPU) pairs is directly applicable |
| **Gandiva** (OSDI 2018) | 2018 | GPU packing, job migration | Migration concept useful for rebalancing |
| **Gandiva_fair** (EuroSys 2020) | 2020 | Second-price auction fairness for heterogeneous GPUs | Fairness model for multi-user pools |
| **AntMan** (OSDI 2020) | 2020 | Dynamic memory bounds, co-location | Memory monitoring for co-located kernels |
| **Sia** (SOSP 2023) | 2023 | Adaptive jobs + heterogeneous resources | Best when jobs can change their parallelism |
| **Helix** (ASPLOS 2025) | 2025 | Max-flow model for LLM layer placement | Directly relevant for LLM workloads |
| **Poplar** (2024) | 2024 | Heterogeneous ZeRO, per-GPU batch sizing | Training with mixed GPU speeds |
| **Cephalo** (ICS 2025) | 2025 | Layered gradient accumulation for mixed GPUs | Training fairness in mixed clusters |

### 4.2 NVIDIA Built-in Mechanisms

**Multi-Process Service (MPS):**
- Allows multiple CUDA processes to share a single GPU concurrently
- Supports SM partitioning on Ampere+ (static allocation of SMs to clients)
- Limited to single user per MPS server
- Max 48 clients per GPU
- Works on CC 3.5+ (all relevant GeForce cards)
- **OuterLink relevance:** Could be used to co-locate multiple OuterLink workloads on a single powerful GPU, with SM limits enforcing QoS

**Multi-Instance GPU (MIG):**
- Hardware-level GPU partitioning into up to 7 isolated instances
- Only on datacenter GPUs (A100, H100) — **not available on GeForce**
- **OuterLink relevance:** Not directly applicable since OuterLink targets GeForce cards. However, the concept of partitioning a powerful GPU for multiple users is relevant and can be approximated with MPS or time-slicing

**MIGER (ICPP 2024):**
- Combines MPS + MIG with hierarchical scheduling
- 36% better job completion time vs MPS-alone
- **OuterLink relevance:** Demonstrates that combining spatial (SM partitioning) and temporal (time-slicing) sharing improves utilization

## 5. Load Balancing with Asymmetric Resources

### 5.1 Proportional Work Distribution

The simplest correct approach: assign work proportional to each GPU's relevant capability metric.

**For data-parallel workloads (e.g., batch inference):**
```
batch_size_i = total_batch * (capability_i / sum(capabilities))
```

Example: 3 GPUs with relative throughput [1.0, 2.5, 6.0]:
- GPU 0 (RTX 3060): 1.0/9.5 = 10.5% of batch
- GPU 1 (RTX 4070): 2.5/9.5 = 26.3% of batch
- GPU 2 (RTX 4090): 6.0/9.5 = 63.2% of batch

### 5.2 Dynamic Rebalancing

Static proportional assignment assumes stable performance. In practice, thermals, power limits, and contention cause performance variation. The scheduler should:

1. Monitor kernel completion times per GPU
2. Compute moving average of throughput per GPU
3. Adjust proportions based on observed (not theoretical) performance
4. React within seconds, not minutes

### 5.3 Avoiding Straggler Effects

In synchronous workloads (all GPUs must complete before proceeding), the slowest GPU determines overall throughput. Strategies:

- **Slight overpartitioning of fast GPUs:** Give fast GPUs slightly more than their fair share, so they finish slightly after slow GPUs rather than well before
- **Work stealing from slow GPUs:** If a fast GPU finishes early, it steals remaining work from the slowest GPU
- **Timeout and reassign:** If a GPU is significantly behind, reassign its remaining work to idle GPUs
- **Pipeline bubbles:** Accept some idle time on fast GPUs in exchange for simpler scheduling

## 6. Work Stealing vs Work Pushing

### 6.1 Work Stealing

**Mechanism:** Each GPU has a local task queue. When a GPU's queue empties, it "steals" tasks from another GPU's queue.

| Aspect | Analysis |
|--------|----------|
| **Strengths** | Self-balancing, low overhead when balanced, decentralized |
| **Weaknesses** | Stealing involves data transfer (slow across network), cache/locality disruption |
| **Best for** | Many small independent tasks, irregular workloads |
| **Worst for** | Tasks with large data dependencies, network-connected GPUs |

**Cross-node work stealing is problematic** in OuterLink because stealing requires moving data across the network. The cost of the data transfer may exceed the benefit of load balancing.

### 6.2 Work Pushing

**Mechanism:** A central scheduler assigns tasks to GPUs based on capability and current load.

| Aspect | Analysis |
|--------|----------|
| **Strengths** | Can optimize placement globally, considers data locality, avoids unnecessary transfers |
| **Weaknesses** | Central scheduler is bottleneck, requires accurate load information |
| **Best for** | Tasks with known data dependencies, heterogeneous clusters |
| **Worst for** | Very dynamic workloads with unpredictable task sizes |

### 6.3 Recommendation for OuterLink

**Primary: Work pushing (centralized scheduler)** for initial kernel placement. The scheduler knows each GPU's capabilities, current load, and data placement.

**Secondary: Local work stealing** within a single node (between GPUs in the same machine). Intra-node transfers are fast (PCIe), so stealing cost is low.

**Avoid:** Cross-node work stealing. The network latency and bandwidth cost of moving data between nodes makes reactive stealing inefficient. Instead, use proactive rebalancing during scheduling.

## 7. CUDA Binary Compatibility

### 7.1 Two Forms of GPU Code

CUDA applications contain GPU code in two forms:

| Form | Description | Compatibility |
|------|------------|--------------|
| **Cubin (SASS)** | Native binary for a specific CC | Same major version, same or higher minor |
| **PTX** | Virtual ISA, JIT-compiled at runtime | Any equal or higher CC (forward-compatible) |

**Example:** A cubin for CC 8.6 runs on CC 8.6 and 8.9 GPUs, but NOT on CC 9.0 or 10.0 GPUs. PTX for CC 8.6 JIT-compiles to any CC >= 8.6.

### 7.2 Fatbins

Modern CUDA applications package multiple cubins and PTX into a "fatbin" container. The CUDA runtime selects the best match:
1. Exact cubin match → use it (fastest load)
2. Compatible cubin (same major, lower minor) → use it
3. PTX available → JIT compile (cached after first run)
4. Nothing compatible → kernel launch fails

### 7.3 OuterLink's Binary Compatibility Strategy

**The problem:** When OuterLink intercepts a CUDA kernel launch and forwards it to a remote GPU, the kernel binary may not be compatible with the remote GPU's CC.

**Strategy:**
1. **At kernel load time** (`cuModuleLoad`, `cuModuleLoadData`), capture both the cubin and PTX from the fatbin
2. **Before dispatching** to a remote GPU, check if the kernel has a compatible cubin for that GPU's CC
3. **If no compatible cubin exists** but PTX is available, the remote GPU's CUDA driver will JIT compile it
4. **If neither cubin nor PTX is compatible**, that GPU cannot run this kernel — the scheduler must exclude it

**JIT compilation overhead:**
- First launch on a new GPU type: can add seconds of latency
- Subsequent launches: cached by the CUDA driver
- OuterLink could pre-warm caches by triggering JIT compilation for all pool GPUs when a module is first loaded

### 7.4 Architecture-Conditional Features (Caveat)

PTX compiled with `compute_100a` or `compute_90a` (architecture-specific features) is NOT forward-compatible. If a kernel uses Hopper-specific features (thread block clusters, TMA), the PTX will not JIT compile on Blackwell. This is rare in practice for consumer workloads but must be handled.

**Detection:** OuterLink can inspect the PTX target version embedded in the module to determine if it uses architecture-conditional features.

### 7.5 What Happens When a Kernel Uses Unavailable Features

| Scenario | Outcome |
|----------|---------|
| Kernel uses FP8 Tensor Cores, GPU is CC 8.6 | Kernel binary won't have a compatible cubin; PTX JIT may succeed but FP8 ops will fail or be emulated (very slow) |
| Kernel uses thread block clusters, GPU is CC 8.9 | Feature not available; kernel launch will fail with `CUDA_ERROR_LAUNCH_INCOMPATIBLE` |
| Kernel uses warp shuffle, GPU is CC 5.0+ | Works on all supported GPUs |
| Kernel targets CC 8.6 cubin, GPU is CC 10.0 | Cubin incompatible (different major version); PTX JIT needed |

**OuterLink must intercept kernel launch errors** and either retry on a compatible GPU or report the error to the application with a clear message.

## 8. Scheduling Algorithm Sketch

```
function schedule_kernel(kernel, requirements):
    // Phase 1: Filter by hard constraints
    candidates = all_gpus.filter(gpu =>
        gpu.compute_capability >= kernel.min_cc AND
        gpu.vram_free >= requirements.vram AND
        kernel.has_compatible_binary(gpu.compute_capability)
    )

    if candidates.empty():
        return Error("No compatible GPU available")

    // Phase 2: Classify workload
    workload_type = classify(kernel)  // compute, memory, tensor, capacity

    // Phase 3: Score candidates
    for gpu in candidates:
        score = 0.0
        match workload_type:
            Compute => score += gpu.fp32_tflops * COMPUTE_WEIGHT
            Memory  => score += gpu.memory_bw_gbps * MEMORY_WEIGHT
            Tensor  => score += gpu.tensor_throughput * TENSOR_WEIGHT
            Capacity => score += gpu.vram_free_gb * CAPACITY_WEIGHT

        // Locality bonus (from R17)
        score += data_locality_score(kernel.input_buffers, gpu) * LOCALITY_WEIGHT

        // Load penalty
        score *= (1.0 - gpu.current_utilization)

        // Network distance penalty (from R17)
        score -= network_distance(client, gpu) * NETWORK_WEIGHT

        gpu.final_score = score

    // Phase 4: Select best
    return candidates.max_by(|gpu| gpu.final_score)
```

## 9. Key Insights for OuterLink

1. **No single metric works** — A GPU that is best for compute-bound work may be mediocre for memory-bound work. Scheduling must be workload-aware.

2. **Profiling pays for itself** — Gavel showed 1.4-3.5x improvement over heterogeneity-agnostic policies. Even lightweight profiling (startup benchmarks) substantially improves scheduling.

3. **Cross-node work stealing is a trap** — It sounds elegant but the data transfer cost dominates in network-connected clusters. Use centralized scheduling with proactive placement instead.

4. **PTX provides forward compatibility** — As long as applications include PTX (most do), OuterLink can JIT compile for any newer GPU. The scheduler must verify binary compatibility before dispatching.

5. **Start simple, add sophistication** — Phase 1: static TFLOPS-proportional scheduling. Phase 2: benchmark-calibrated scoring. Phase 3: workload-classified scheduling. Phase 4: runtime-adaptive weights.

## Related Documents

- [01-gpu-capability-landscape.md](./01-gpu-capability-landscape.md) — GPU hardware specs feeding into scheduler
- [03-practical-mixing-scenarios.md](./03-practical-mixing-scenarios.md) — Real-world scenarios testing these strategies
- [R17 Topology-Aware Scheduling](../../phase-08-network-optimization/R17-topology-scheduling/README.md) — Network topology scoring
- [R13 CUDA Graph Interception](../R13-cuda-graph-interception/README.md) — Graph-level scheduling across GPUs

## Open Questions

- [ ] How to classify workload type at interception time without running the kernel? Heuristics from kernel name, argument sizes, grid dimensions?
- [ ] Should OuterLink maintain a kernel-to-GPU affinity cache (remember where a kernel ran well and prefer that GPU next time)?
- [ ] What is the actual JIT compilation overhead for PTX on consumer GPUs? Need benchmarks.
- [ ] How does Gavel's scalability limitation (expensive solver) affect OuterLink if the pool grows to 50+ GPUs?
- [ ] Can MPS SM partitioning be used to share a powerful GPU (4090) between multiple OuterLink tasks safely?
