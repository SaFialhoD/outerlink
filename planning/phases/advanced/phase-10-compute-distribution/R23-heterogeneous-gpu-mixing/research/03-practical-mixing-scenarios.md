# Practical GPU Mixing Scenarios

**Created:** 2026-03-25
**Updated:** 2026-03-25
**Status:** Draft
**Purpose:** Analyze real-world GPU combinations, use cases, and practical constraints for heterogeneous OuterLink pools.

## 1. Real-World GPU Combinations Users Will Have

### Scenario A: The Incremental Upgrader

The most common heterogeneous pool: a user who upgrades one GPU at a time.

| Configuration | VRAM Total | Compute Range |
|--------------|-----------|---------------|
| RTX 3060 + RTX 4070 | 12 + 12 = 24 GB | 12.7 - 29.1 TFLOPS (2.3x gap) |
| RTX 3080 + RTX 4080 | 10 + 16 = 26 GB | 29.8 - 48.7 TFLOPS (1.6x gap) |
| RTX 3090 + RTX 4090 | 24 + 24 = 48 GB | 35.6 - 82.6 TFLOPS (2.3x gap) |
| RTX 3090 + RTX 5090 | 24 + 32 = 56 GB | 35.6 - 104.8 TFLOPS (2.9x gap) |

**Key insight:** Within 1-2 generation gaps, the compute ratio stays under 3x. This is manageable with proportional scheduling.

### Scenario B: The Multi-PC Home Lab

Pedro's target user: multiple PCs networked together, each with whatever GPU it has.

| PC | GPU | VRAM | Compute | Memory BW |
|----|-----|------|---------|-----------|
| Workstation | RTX 3090 | 24 GB | 35.6 TF | 936 GB/s |
| Gaming PC | RTX 4090 | 24 GB | 82.6 TF | 1,008 GB/s |
| Older PC | RTX 2080 Ti | 11 GB | 13.4 TF | 616 GB/s |
| HTPC | RTX 3060 | 12 GB | 12.7 TF | 360 GB/s |
| **Pool total** | | **71 GB** | **144.3 TF** | |

This 4-GPU pool spans 3 generations (Turing, Ampere, Ada) and has a 6.5x compute gap between weakest and strongest. The pool aggregates 71 GB VRAM — enough for large LLMs.

### Scenario C: The Budget Builder

Buying used GPUs to maximize VRAM per dollar:

| GPU | Used Price (~2026) | VRAM | $/GB |
|-----|-------------------|------|------|
| RTX 3060 12GB | ~$150 | 12 GB | $12.50/GB |
| RTX 3090 24GB | ~$500 | 24 GB | $20.83/GB |
| RTX 4060 8GB | ~$200 | 8 GB | $25.00/GB |
| RTX 3080 10GB | ~$250 | 10 GB | $25.00/GB |

The budget builder might assemble: 2x RTX 3060 + 1x RTX 3090 = 48 GB for ~$800. This is a highly heterogeneous pool (VRAM: 12+12+24, compute ratio: 1:1:2.8).

## 2. LLM Inference with Mixed GPUs

### 2.1 The Opportunity

LLM inference has two distinct resource requirements:
1. **Model weights** — need VRAM capacity (proportional to parameter count)
2. **KV cache** — needs VRAM capacity AND bandwidth (grows with context length)
3. **Attention computation** — needs compute (Tensor Cores ideal)
4. **Decode phase** — memory-bandwidth-bound (one token at a time)

Different GPUs can serve different roles based on their strengths.

### 2.2 Layer-Based Pipeline Parallelism (Uneven Split)

Split the model by layers across GPUs, assigning more layers to faster GPUs:

**Example: 70B LLM across RTX 3090 + RTX 4090 + RTX 3060**
- Total VRAM: 24 + 24 + 12 = 60 GB
- Model size (FP16): ~140 GB — too large even for this pool in FP16
- Model size (INT4 GPTQ): ~35 GB — fits comfortably

Layer distribution by VRAM and compute:
```
RTX 4090 (24 GB, 82.6 TF): 40% of layers (fast compute, ample VRAM)
RTX 3090 (24 GB, 35.6 TF): 35% of layers (large VRAM, moderate compute)
RTX 3060 (12 GB, 12.7 TF): 25% of layers (limited by both VRAM and compute)
```

**Pipeline depth = 3 stages.** Decode throughput limited by the slowest stage. Smart partitioning assigns fewer layers to the RTX 3060 to balance stage times.

### 2.3 KV Cache Distribution

The KV cache grows with context length and batch size. In a heterogeneous pool, the strategy depends on what's scarce:

| Strategy | When to Use | How |
|----------|------------|-----|
| Proportional to VRAM | VRAM-limited regime | Each GPU stores KV proportional to its free VRAM |
| On high-bandwidth GPUs | Decode-bound regime | KV cache on GPUs with highest memory bandwidth (faster token generation) |
| Tiered offload | Very long contexts | Hot KV in VRAM, warm KV in pinned host RAM, cold KV on disk |

**Prima.cpp** (2025 research) demonstrated 70B model inference at 674 ms/token across 4 consumer devices using pipelined-ring parallelism with heterogeneity-aware scheduling.

**Helix** (ASPLOS 2025) models this as a max-flow problem: co-optimize model partition, device placement, and request scheduling to maximize throughput while reserving enough VRAM for KV cache.

### 2.4 Phase-Disaggregated Serving

Modern LLM inference separates prefill and decode:
- **Prefill:** Compute-bound (process entire prompt in parallel) — route to high-TFLOPS GPU
- **Decode:** Memory-bandwidth-bound (one token per step) — route to high-bandwidth GPU

In a mixed pool (RTX 4090 + RTX 3090):
- RTX 4090 handles prefill (higher compute)
- RTX 3090 handles decode (still good bandwidth, frees the 4090 for more prefills)

This is a natural fit for heterogeneous hardware.

## 3. Training with Heterogeneous Hardware

### 3.1 Data Parallelism with Uneven Batch Sizes

Standard data parallelism replicates the model on every GPU and splits the batch equally. With heterogeneous GPUs, assign batch sizes proportional to each GPU's throughput:

```
gpu_batch_i = global_batch * (throughput_i / total_throughput)
```

**Gradient scaling requirement:** Each GPU computes gradients over its local batch. The all-reduce must weight gradients by batch size to maintain correct optimization:
```
global_gradient = sum(gradient_i * batch_size_i) / global_batch_size
```

**Poplar** (2024) extends ZeRO to support per-GPU batch sizes and gradient accumulation steps, enabling fine-grained workload balancing across heterogeneous GPUs.

### 3.2 Pipeline Parallelism with Uneven Layer Splits

For models too large to replicate: split layers across GPUs based on compute and memory capacity.

| GPU | Layers | VRAM for Weights | Activations Budget |
|-----|--------|-----------------|-------------------|
| RTX 4090 (24 GB) | 40% | ~14 GB | ~10 GB |
| RTX 3090 (24 GB) | 35% | ~12.25 GB | ~11.75 GB |
| RTX 3060 (12 GB) | 25% | ~8.75 GB | ~3.25 GB |

**Bubble overhead:** Pipeline parallelism has "bubble" time where stages are idle. With 3 stages and microbatches, the bubble ratio is ~(stages-1)/microbatches. More microbatches reduces bubbles but increases memory pressure.

### 3.3 Synchronization Challenges

| Challenge | Impact | Mitigation |
|-----------|--------|------------|
| All-reduce speed limited by slowest GPU | Training throughput bottleneck | Uneven batch sizes, gradient accumulation |
| Different memory capacity limits batch size | Some GPUs can't fit same micro-batch | Per-GPU micro-batch sizing |
| Gradient precision differences | CC 8.9 supports FP8, CC 8.6 doesn't | Use common precision (FP16/BF16) for gradient communication |
| Thermal throttling on older GPUs | Performance degrades under sustained load | Dynamic rebalancing based on observed throughput |

### 3.4 OuterLink's Role in Training

OuterLink operates at the CUDA API level, below frameworks like PyTorch. For training:
- The application sees N virtual GPUs with uniform APIs
- OuterLink transparently handles data movement between GPUs
- Gradient all-reduce uses OuterLink's transport (host-staged or OpenDMA)
- The application framework (PyTorch DDP, FSDP) handles the training logic

OuterLink doesn't need to understand training semantics — it provides the unified GPU pool, and the application framework handles parallelism strategy. However, exposing GPU capability information through the CUDA API (via intercepted `cudaGetDeviceProperties`) enables frameworks to make heterogeneity-aware decisions.

## 4. Multi-User Pools

### 4.1 GPU Assignment Strategies

When multiple users share a heterogeneous pool:

| Strategy | Description | Fairness |
|----------|------------|---------|
| **Fixed assignment** | User A always gets RTX 4090, User B always gets RTX 3060 | Unfair — some users always get worse hardware |
| **Round-robin assignment** | Alternate GPU assignments per job | Fair over time, but individual jobs may get unlucky |
| **Priority-based** | High-priority jobs get best GPUs | Fair for priority tiers, unfair within tiers |
| **Gandiva_fair auction** | Users bid for GPU time, second-price mechanism | Incentive-compatible, naturally balances demand |
| **Proportional share** | Each user gets a share of total compute proportional to their allocation | Fair, but complex to implement |

### 4.2 GPU Equivalence Units

To make heterogeneous resources commensurable, define a "GPU Equivalent Unit" (GEU):

```
GEU_i = alpha * (TFLOPS_i / TFLOPS_ref)
       + beta * (BW_i / BW_ref)
       + gamma * (VRAM_i / VRAM_ref)
```

Where `ref` is a reference GPU (e.g., RTX 3060 = 1.0 GEU). Users are allocated GEUs, and the scheduler maps GEUs to physical GPUs.

Example GEU values (alpha=0.4, beta=0.3, gamma=0.3):

| GPU | TFLOPS Score | BW Score | VRAM Score | Total GEU |
|-----|-------------|----------|------------|-----------|
| RTX 3060 | 0.40 | 0.30 | 0.30 | 1.00 |
| RTX 3090 | 1.12 | 0.78 | 0.60 | 2.50 |
| RTX 4070 | 0.92 | 0.42 | 0.30 | 1.64 |
| RTX 4090 | 2.60 | 0.84 | 0.60 | 4.04 |
| RTX 5090 | 3.30 | 1.49 | 0.80 | 5.59 |

A user allocated 5.0 GEU could get: one RTX 4090 (4.04) + one RTX 3060 (1.00) = 5.04 GEU.

## 5. Driver Compatibility Across Nodes

### 5.1 The Compatibility Matrix

CUDA has three compatibility modes:

| Mode | Rule | Practical Impact |
|------|------|-----------------|
| **Backward** | Newer driver runs older CUDA apps | Always works — newer drivers support all older apps |
| **Minor version** | CUDA 12.x app on any 12.y driver (y >= min) | Within same major version (11.x or 12.x), broadly compatible |
| **Forward** | Older driver runs newer CUDA toolkit apps via compat package | Data center GPUs only — NOT available on GeForce |

### 5.2 OuterLink's Driver Strategy

**Constraint:** Forward compatibility packages do NOT work on GeForce GPUs. Each node must have a driver version that natively supports the CUDA toolkit version used by applications.

**Recommendation:**
1. All nodes should run the **same major driver version** (e.g., all on 560.x or all on 570.x)
2. The minimum driver version across all nodes determines the maximum CUDA toolkit version supported
3. OuterLink should query driver version at registration and warn about mismatches

**Driver version query:**
```
cuDriverGetVersion(&version)  // Returns e.g., 12080 for CUDA 12.8
```

### 5.3 Practical Compatibility Table

| CUDA Toolkit | Minimum Driver (Linux) | Minimum Driver (Windows) |
|-------------|----------------------|------------------------|
| CUDA 12.0 | 525.60.13 | 527.41 |
| CUDA 12.4 | 550.54.14 | 551.61 |
| CUDA 12.6 | 560.28.03 | 560.70 |
| CUDA 12.8 | 570.86.15 | 571.14 |

**Key risk:** A node with an old driver (e.g., 525.x) cannot run applications compiled with CUDA 12.6+. OuterLink must detect this at registration and either reject the node or limit which kernels it can receive.

### 5.4 Mixed Driver Scenarios

| Node A Driver | Node B Driver | App CUDA Version | Works? |
|--------------|--------------|-----------------|--------|
| 570.x | 570.x | 12.8 | Yes — both support it |
| 570.x | 535.x | 12.8 | Partial — only Node A can run 12.8 apps |
| 560.x | 550.x | 12.4 | Yes — both support 12.4 |
| 560.x | 525.x | 12.0 | Yes for 12.0 — but Node B can't run 12.4+ kernels |

**OuterLink must maintain a per-node compatibility profile** that includes supported CUDA versions, so the scheduler only dispatches kernels to nodes that can execute them.

## 6. Memory Capacity Aggregation

### 6.1 Unified Pool Arithmetic

The headline feature: combining VRAM across multiple GPUs into one addressable pool.

**Example Pool:**
```
RTX 3090:  24 GB
RTX 4090:  24 GB
RTX 3060:  12 GB
RTX 3060:  12 GB
─────────────────
Total:     72 GB unified VRAM
```

72 GB is enough for many 70B LLMs in INT4 quantization (~35 GB) with generous KV cache space.

### 6.2 Not All VRAM Is Equal

While OuterLink presents 72 GB as a unified pool, access latency varies dramatically:

| Access Pattern | Latency | Bandwidth |
|---------------|---------|-----------|
| Local VRAM (same GPU) | ~100 ns | 360-1,792 GB/s (depends on GPU) |
| Remote VRAM via host-staged | ~5-50 us | ~10-25 GB/s (limited by network + copies) |
| Remote VRAM via OpenDMA | ~2-5 us | ~12 GB/s (100Gbps ConnectX-5) |

The scheduler (R10 memory tiering) must be aware of this NUMA-like topology when placing data.

### 6.3 Capacity Planning for Workloads

| Workload | Model Size | KV Cache (4K ctx) | KV Cache (32K ctx) | Total VRAM Needed |
|----------|-----------|-------------------|-------------------|------------------|
| LLaMA-7B FP16 | 14 GB | ~0.5 GB | ~4 GB | 14.5 - 18 GB |
| LLaMA-13B FP16 | 26 GB | ~1 GB | ~8 GB | 27 - 34 GB |
| LLaMA-70B INT4 | 35 GB | ~2.5 GB | ~20 GB | 37.5 - 55 GB |
| Stable Diffusion XL | ~7 GB | N/A | N/A | ~10 GB (with batch) |
| Whisper Large | ~3 GB | N/A | N/A | ~5 GB |

A 70B model with 32K context needs ~55 GB — achievable with a 3-GPU pool (e.g., 3090 + 4090 + 3060 = 60 GB) but not with any single consumer GPU.

## 7. Performance: Bottleneck vs Smart Partitioning

### 7.1 The Slowest-GPU Bottleneck

In synchronous workloads (pipeline parallelism, all-reduce), the slowest GPU determines overall throughput. Adding a slow GPU to the pool can DECREASE performance if work isn't partitioned properly.

**Bad partition (equal split):**
```
RTX 4090: 50% of work → finishes in 1.0s
RTX 3060: 50% of work → finishes in 6.5s
Total time: 6.5s (bottleneck: RTX 3060)
Effective throughput: worse than RTX 4090 alone (1.0s for 50% = 2.0s for 100%)
```

**Smart partition (proportional):**
```
RTX 4090: 87% of work → finishes in 1.7s
RTX 3060: 13% of work → finishes in 1.7s
Total time: 1.7s
Effective throughput: 1.18x of RTX 4090 alone
```

**Key insight:** A properly proportioned heterogeneous pool always outperforms the best single GPU. An improperly proportioned pool can be WORSE than using only the best GPU.

### 7.2 When Heterogeneity Hurts

| Situation | Why It Hurts | Mitigation |
|-----------|-------------|------------|
| Synchronous all-reduce with unequal compute | Fastest GPUs idle waiting for slowest | Proportional batch sizes |
| Small model on many GPUs | Communication overhead exceeds compute | Only use enough GPUs to fit the model |
| Memory-bound work on low-bandwidth GPU | GPU becomes a bottleneck stage | Place memory-bound layers on high-BW GPUs |
| Frequent inter-GPU data exchange | Network cost scales with number of GPUs | Minimize cross-GPU communication |

### 7.3 When Heterogeneity Helps

| Situation | Why It Helps | Example |
|-----------|-------------|---------|
| VRAM aggregation | Pool supports models that don't fit on one GPU | 70B model across 3 GPUs |
| Workload disaggregation | Different tasks → different optimal GPUs | Prefill on 4090, decode on 3090 |
| Multi-user serving | Each user gets a GPU; total capacity scales | 4 users served simultaneously |
| Redundancy | If one GPU fails, others continue | Fault tolerance |
| Cost efficiency | Old GPUs still contribute value | RTX 3060 adds 12 GB to pool for $150 |

### 7.4 The Threshold Rule

**Rule of thumb:** A GPU should only be added to the pool for a synchronous workload if its performance is within 10x of the fastest GPU. Beyond that, the scheduling overhead and straggler effects outweigh the benefit.

For async workloads (independent jobs, batch inference), any GPU adds value regardless of speed difference.

## 8. Implications for OuterLink Design

### 8.1 GPU Pool Registration

When a GPU joins the pool, OuterLink must:
1. Query all hardware attributes (CC, VRAM, bandwidth, BAR1, PCIe)
2. Run calibration benchmarks (SGEMM, stream copy, H2D transfer)
3. Check driver compatibility with current pool baseline
4. Compute capability profile and GEU rating
5. Register with the cluster coordinator
6. Warn about potential issues (no ReBAR, old driver, low VRAM)

### 8.2 Workload Routing

The scheduler needs a decision tree:
```
1. Can this kernel run on this GPU? (CC compatibility, binary availability)
2. Does this GPU have enough VRAM? (capacity check)
3. How fast will this kernel run? (workload-type-aware scoring)
4. Where is the data? (locality from R17)
5. How busy is this GPU? (load balancing)
```

### 8.3 Graceful Degradation

When the best GPU isn't available, OuterLink should:
- Use the next-best GPU (reduced performance, not failure)
- Warn the user about expected performance impact
- Queue the work if a suitable GPU will be available soon
- NEVER silently fail — always report why a GPU was chosen

### 8.4 User-Facing GPU Policies

Users should be able to express preferences:
```
# Pin to a specific GPU (debugging)
OUTERLINK_GPU_PIN=node1:gpu0

# Exclude slow GPUs
OUTERLINK_MIN_TFLOPS=20.0

# Prefer high-VRAM GPUs
OUTERLINK_PREFER=vram

# Exclude specific nodes
OUTERLINK_EXCLUDE_NODES=htpc
```

## Related Documents

- [01-gpu-capability-landscape.md](./01-gpu-capability-landscape.md) — Hardware specs and capabilities
- [02-heterogeneous-scheduling.md](./02-heterogeneous-scheduling.md) — Scheduling algorithms and strategies
- [R10 Memory Tiering](../../phase-07-memory-intelligence/R10-memory-tiering/README.md) — Memory hierarchy for unified pool
- [R17 Topology-Aware Scheduling](../../phase-08-network-optimization/R17-topology-scheduling/README.md) — Network topology in scheduling
- [R20 NCCL Backend](../../phase-09-collective-communication/R20-nccl-backend/README.md) — NCCL topology reporting for heterogeneous devices

## Open Questions

- [ ] What is the minimum pool size (# of GPUs) where heterogeneous scheduling overhead becomes worthwhile vs just using the best single GPU?
- [ ] Should OuterLink auto-detect and report optimal workload partitioning to the user, or just execute whatever partition the application requests?
- [ ] How to handle the case where a user upgrades their GPU mid-session? Dynamic pool membership changes.
- [ ] Is there a market for "GPU lending" where users contribute idle GPUs to other users' pools? Security and trust implications?
- [ ] How does thermal throttling affect scheduling? A GPU in a poorly ventilated case may start fast and slow down under sustained load.
- [ ] For LLM inference specifically, should OuterLink integrate with vLLM/llama.cpp or remain framework-agnostic?
