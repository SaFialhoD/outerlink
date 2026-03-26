# R17 Research: Data Placement Optimization

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Priority:** HIGH

## Purpose

Survey strategies for deciding where data (pages, tensors, model weights) should live across the cluster: affinity-based placement, replication vs migration tradeoffs, load balancing across tiers and nodes, ML-specific placement patterns, and cost models for migration decisions. This connects topology knowledge (from 01-topology-discovery) and routing capability (from 02-routing-algorithms) into actionable placement policy.

## TL;DR — What Works and What OuterLink Should Use

| Technique | What It Does | OuterLink Applicability |
|---|---|---|
| Access-frequency affinity | Place data near the GPU that accesses it most | Primary placement heuristic for all workloads |
| Cost-benefit migration (ARMS model) | Migrate only when benefit > cost | Gate all migrations through cost check: avoid thrashing |
| Replication for read-heavy data | Copy weights to every node that reads them | Model weights in data-parallel training (read by all, written by none during forward) |
| Computation migration | Move compute to data instead of data to compute | For small kernels on large remote tensors; limited applicability in GPU context |
| First-fit with affinity scoring | Place new allocations on best-scoring node | cuMemAlloc interception decides which node gets the allocation |
| Bandwidth-proportional placement | Distribute data proportional to link bandwidth | Spread optimizer states across nodes connected by fast links |
| BATMAN bandwidth-aware tiering | Distribute accesses proportional to tier bandwidth | Direct application to R10's 5-tier hierarchy |

**The single biggest lesson:** Data placement is not a one-time decision. The optimal location for a tensor changes during training (forward pass: weights read everywhere; backward pass: gradients written locally then reduced; optimizer: states updated locally). OuterLink needs phase-aware placement that adapts within each training iteration.

---

## 1. Affinity-Based Placement

### 1.1 What is Affinity?

Affinity measures how much a specific GPU benefits from having data local. High affinity = GPU accesses the data frequently. Low affinity = GPU rarely or never touches it.

**Affinity score per page:**
```
affinity(page, gpu) = access_count(page, gpu) / total_accesses(page)
```

A page with affinity(page, GPU_A) = 0.9 should live on GPU_A's node. A page with affinity split 0.5/0.5 between two GPUs is a candidate for replication.

### 1.2 Tracking Access Patterns

R10's page table already tracks per-page metadata:
- `access_count`: total accesses since last reset
- `last_accessed`: timestamp of most recent access
- `accessing_gpus`: bitmap of which GPUs have accessed this page

R17 extends this with:
- `access_count_per_gpu[gpu_id]`: per-GPU access frequency
- `access_pattern`: read-only, write-only, read-write
- `phase`: which training phase last accessed (forward, backward, optimizer)

### 1.3 Affinity-Driven Migration

When a page's access pattern shifts (new GPU starts accessing it frequently):

1. Compute new affinity scores after N accesses (e.g., N=100)
2. If the highest-affinity GPU is not the current host, evaluate migration cost
3. Migrate if benefit exceeds cost (see Section 5)

**Hysteresis:** Don't migrate on the first affinity shift. Require the new pattern to persist for at least 3 measurement windows before triggering migration. This prevents thrashing during transient patterns.

---

## 2. Replication vs Migration

### 2.1 When to Replicate

Replication creates copies of a page on multiple nodes. Best for:

| Scenario | Why Replicate |
|---|---|
| Read-only data accessed by multiple GPUs | No write coherence needed; all nodes read local copy |
| Model weights during forward pass | Every GPU in data-parallel reads the same weights |
| Lookup tables / embeddings (read-heavy) | Shared across all GPU inference workers |
| Small data accessed frequently from many nodes | Cheaper to copy once than transfer repeatedly |

**Cost of replication:**
- Memory: N copies * page_size (64 KB * N nodes)
- Bandwidth: one-time broadcast (64 KB * (N-1) links)
- No ongoing cost if data is truly read-only

### 2.2 When to Migrate

Migration moves a page from one node to another (single copy). Best for:

| Scenario | Why Migrate |
|---|---|
| Write-heavy data used by one GPU | Avoids write coherence protocol |
| Large tensors with shifting access locality | Replication would waste too much memory |
| Memory pressure on current host | Evict to a node with more free VRAM |
| Phase transition (forward -> backward) | Gradient tensors move to where backward runs |

**Cost of migration:**
- Bandwidth: page_size / link_speed (64 KB / 12.5 GB/s = 5.1 us for RDMA)
- Latency: migration_latency + metadata_update
- Disruption: page is unavailable during transfer (or use copy-on-migrate)

### 2.3 Hybrid: Replicate-then-Invalidate

For ML training iterations:
1. **Forward pass start:** Replicate weights to all nodes (broadcast)
2. **Forward pass:** All nodes read local weight copies
3. **Backward pass:** Each node computes local gradients (no replication needed)
4. **AllReduce:** Gradients reduced across nodes (NCCL handles this)
5. **Optimizer step:** Updated weights written at a single node
6. **Next iteration:** Re-broadcast updated weights

This is exactly what data-parallel training does. OuterLink should recognize this pattern and optimize for it.

### 2.4 Write Coherence for Replicated Pages

If a replicated page is written:
1. **Invalidate:** Mark all remote copies as stale (simple, low overhead)
2. **Update:** Push the write to all copies (higher bandwidth, consistent reads)

**Recommendation:** Invalidate-on-write for Phase 1. OuterLink's page table marks replicas as "stale" when a write is detected. Next read from a remote node triggers a fresh fetch.

---

## 3. Placement Algorithms

### 3.1 First-Fit with Affinity Scoring

When a CUDA application calls `cuMemAlloc`, OuterLink intercepts and decides which node hosts the allocation:

```rust
fn place_allocation(size: usize, requesting_gpu: GpuId) -> NodeId {
    let candidates: Vec<(NodeId, f64)> = cluster.nodes()
        .filter(|n| n.free_vram() >= size)
        .map(|n| {
            let score = compute_placement_score(n, requesting_gpu, size);
            (n.id, score)
        })
        .collect();

    candidates.max_by_key(|(_, score)| score).node_id
}

fn compute_placement_score(node: &Node, gpu: GpuId, size: usize) -> f64 {
    let locality = if node.has_gpu(gpu) { 1.0 } else { 0.0 };
    let bw_to_gpu = topology.bandwidth(node.id, gpu.node_id);
    let memory_pressure = 1.0 - (node.used_vram() as f64 / node.total_vram() as f64);

    // Weights: locality is king, then bandwidth, then memory availability
    0.6 * locality + 0.25 * (bw_to_gpu / max_bw) + 0.15 * memory_pressure
}
```

**Result:** Allocations land on the local node when possible. When local VRAM is full, they spill to the node with the best connectivity.

### 3.2 Best-Fit (Memory Efficient)

Instead of first available node, choose the node where the allocation fits most tightly (least remaining free memory after allocation). This reduces fragmentation across nodes.

**Trade-off:** Best-fit optimizes memory utilization but may place data on a poorly-connected node. OuterLink should combine best-fit with a minimum connectivity threshold.

### 3.3 Constraint-Based Placement

For ML workloads, the framework often provides constraints:
- "These tensors must be on the same GPU" (operator fusion)
- "These tensors must be on different GPUs" (model parallelism)
- "This tensor is temporary and will be freed after backward" (gradient checkpointing)

OuterLink can detect some constraints automatically:
- Tensors allocated in the same `cuMemAlloc` batch -> same node
- Tensors used as arguments to the same kernel -> co-locate if possible
- Tensors in a cuBLAS GEMM -> matrices A, B, C should be on the same GPU

### 3.4 Delay Scheduling (Dally-Inspired)

When no ideal placement is available (preferred node is full), delay the placement decision briefly:
1. Wait up to T milliseconds (default: 5 ms) for preferred node to free memory
2. If memory becomes available within T, place there
3. If timeout, fall back to best alternative node

The Dally system showed this can improve placement quality by up to 69% for distributed training jobs, at the cost of slightly higher allocation latency.

---

## 4. ML-Specific Placement Strategies

### 4.1 Data Parallelism

Each GPU processes different data batches with the same model:

| Data Type | Placement | Count | Size (7B model) |
|---|---|---|---|
| Model weights | Replicated on all nodes | N copies | ~14 GB each |
| Activations | Local to computing GPU | 1 per GPU | 2-8 GB each |
| Gradients | Local, then AllReduced | 1 per GPU | ~14 GB each |
| Optimizer states (Adam) | Local to each GPU | 1 per GPU | ~28 GB each (2x weights) |

**Total per GPU:** ~60 GB for a 7B model with Adam. Exceeds 24 GB VRAM of RTX 3090.

**OuterLink optimization:** Distribute optimizer states to DRAM tier (R10), keep weights and activations in VRAM. Prefetch optimizer states from DRAM when needed (R11).

### 4.2 Pipeline Parallelism

Model split into stages, each stage on a different GPU:

```
GPU 0: Layers 0-7    ->  GPU 1: Layers 8-15  ->  GPU 2: Layers 16-23  ->  GPU 3: Layers 24-31
```

**Placement:** Each stage's weights, activations, and gradients stay on its assigned GPU. Inter-stage communication (activation tensors between pipeline stages) flows through OuterLink's transport.

**Topology relevance:** Pipeline stages that are adjacent should be placed on GPUs connected by the fastest link. If GPU 0 and GPU 1 are on different nodes, they should be connected by RDMA or OCuLink, not TCP.

### 4.3 Tensor Parallelism

Single layers split across GPUs. Each GPU holds a slice of the weight matrix:

```
Matrix A (4096 x 4096) split across 4 GPUs:
  GPU 0: A[:, 0:1024]
  GPU 1: A[:, 1024:2048]
  GPU 2: A[:, 2048:3072]
  GPU 3: A[:, 3072:4096]
```

**Communication pattern:** AllReduce after each layer. Very bandwidth-intensive.

**Topology relevance:** Tensor parallelism should only be used across GPUs connected by the highest-bandwidth links (NVLink within a node, RDMA/OCuLink between nodes). Using TCP for tensor-parallel communication would be catastrophic for performance.

### 4.4 ZeRO / FSDP (Sharded Data Parallelism)

ZeRO partitions optimizer states (Stage 1), gradients (Stage 2), and parameters (Stage 3) across GPUs:

| ZeRO Stage | What's Sharded | Communication | Memory Savings |
|---|---|---|---|
| Stage 1 | Optimizer states | AllGather for params when needed | ~4x per GPU |
| Stage 2 | + Gradients | + ReduceScatter for gradients | ~8x per GPU |
| Stage 3 | + Parameters | + AllGather before each forward/backward | ~Nx per GPU (N = GPU count) |

**OuterLink placement:** ZeRO Stage 3 is ideal for OuterLink because it naturally distributes memory across nodes. Each node holds 1/N of the model. OuterLink's transport layer handles the AllGather/ReduceScatter operations.

**Topology relevance:** ZeRO Stage 3 generates heavy AllGather traffic before each layer's forward pass. The topology-aware scheduler should ensure sufficient bandwidth is reserved for these operations.

---

## 5. Cost Models for Migration Decisions

### 5.1 The ARMS Cost-Benefit Framework

ARMS (Adaptive and Robust Memory Tiering System) provides a concrete framework:

**Migrate if:** `expected_benefit > migration_cost`

**Expected benefit:**
```
benefit = hotness_score * hot_age * latency_reduction
```
Where:
- `hotness_score` = access frequency (accesses per second)
- `hot_age` = how long the page has been "hot" (persistent heat = higher benefit)
- `latency_reduction` = latency(current_tier) - latency(target_tier)

**Migration cost:**
```
cost = page_size / link_bandwidth + disruption_penalty
```
Where:
- `page_size / link_bandwidth` = actual transfer time
- `disruption_penalty` = estimated stall time if page is accessed during migration

### 5.2 OuterLink Migration Cost Model

Adapting ARMS for OuterLink's multi-transport cluster:

```rust
fn should_migrate(page: &Page, target_node: NodeId) -> bool {
    let current_node = page.location.node;
    let link = topology.best_link(current_node, target_node);

    // Cost: transfer time + metadata update
    let transfer_time_us = (PAGE_SIZE as f64 / link.bw_bytes_per_us()) + link.rtt_us;
    let metadata_cost_us = 5.0;  // Page table update + invalidation
    let total_cost_us = transfer_time_us + metadata_cost_us;

    // Benefit: future access savings over the page's expected lifetime at new location
    let access_rate = page.accesses_per_second();
    let latency_saved_per_access = page.remote_access_latency_us - page.local_access_latency_us;
    let expected_lifetime_us = estimate_page_lifetime(page);
    let total_benefit_us = access_rate * latency_saved_per_access * expected_lifetime_us;

    total_benefit_us > total_cost_us * MIGRATION_THRESHOLD  // threshold = 2.0 (require 2x benefit)
}
```

### 5.3 Concrete Numbers

For a page on Node B accessed by GPU on Node A, connected by ConnectX-5 RDMA:

| Metric | Value |
|---|---|
| Remote access latency (per access) | ~7 us (RDMA read) |
| Local access latency (per access) | ~0.3 us (VRAM) |
| Latency saved per access | ~6.7 us |
| Migration cost (64 KB / 12.5 GB/s + 2 us RTT) | ~7.1 us |
| Break-even: accesses to recoup migration | ~2 accesses |

**Conclusion:** If a page will be accessed more than twice after migration, it's worth migrating. This is almost always true for ML training workloads where pages are accessed hundreds of times per iteration.

For slower links (TCP/25GbE):

| Metric | Value |
|---|---|
| Remote access latency (per access) | ~73 us (TCP) |
| Local access latency (per access) | ~0.3 us (VRAM) |
| Migration cost (64 KB / 2.8 GB/s + 50 us RTT) | ~73 us |
| Break-even | ~2 accesses |

The break-even is still very low because the access latency savings are proportionally larger for slow links.

### 5.4 When NOT to Migrate

| Condition | Why Not |
|---|---|
| Page accessed by multiple GPUs roughly equally | Migration just shifts the problem; replicate instead |
| Page will be freed soon (optimizer scratch space) | Migration cost wasted |
| Destination node VRAM is >90% full | Migration may trigger eviction cascade |
| Link is >80% saturated | Migration adds congestion, worsening all transfers |
| Page is being migrated already | Avoid double-migration |

---

## 6. Load Balancing Across Nodes

### 6.1 Memory Load Balancing

Ensure no single node is overwhelmed while others have free capacity:

```
imbalance = max(node.used_vram%) - min(node.used_vram%)
```

If `imbalance > 20%`, trigger rebalancing: migrate cold pages from the most-loaded node to the least-loaded node.

### 6.2 Bandwidth Load Balancing

Ensure no single link is the bottleneck:

```
link_utilization[i] = link.current_traffic / link.capacity
imbalance = max(link_utilization) - mean(link_utilization)
```

If `imbalance > 0.3`, shift new placements to nodes reachable via underutilized links.

### 6.3 BATMAN-Style Bandwidth-Proportional Distribution

BATMAN distributes memory accesses proportional to tier bandwidth. Applied to OuterLink:

If Node A has RDMA (100G) and USB4 (80G) to Node B:
- Place 56% of frequently-migrating pages to paths that use RDMA
- Place 44% to paths that use USB4
- This ensures both links contribute proportionally

### 6.4 Rebalancing is Low Priority

Rebalancing is background work. It should:
1. Use lowest-priority transfer class (Priority 4)
2. Only run when total link utilization < 50%
3. Migrate at most 1% of total pages per minute
4. Pause immediately if demand traffic increases

---

## 7. Integration with R10 Memory Tiering

### 7.1 Placement as Tier Selection

R10 defines 5 tiers: Local VRAM -> Remote VRAM -> Local DRAM -> Remote DRAM -> NVMe.

R17's placement decision IS the tier selection:
- Place on local VRAM = Tier 0
- Place on remote VRAM of best-connected node = Tier 1
- Place on local DRAM = Tier 2
- Place on remote DRAM of best-connected node = Tier 3
- Place on NVMe = Tier 4

**The topology graph determines which remote node is "best-connected"** for each tier. This is not hardcoded — it's dynamically computed based on current link bandwidth and latency.

### 7.2 Eviction Destination Selection

When R10 needs to evict a page from VRAM, R17 determines where it goes:

```
1. Local DRAM (if available) — zero network cost
2. Remote VRAM on best-connected node (if that node has free VRAM)
3. Remote DRAM on best-connected node
4. NVMe (last resort)
```

The "best-connected" node changes based on current topology and congestion.

### 7.3 Promotion Source Selection

When R10 promotes a page to VRAM, R17 determines the fetch path:

```
1. From local DRAM — zero network cost
2. From remote VRAM — use best available link
3. From remote DRAM — use best available link
4. From NVMe — local I/O
```

If the page exists in multiple locations (replicated), R17 fetches from the closest replica.

---

## 8. Advanced: Predictive Placement

### 8.1 Training Iteration Pattern

ML training is highly predictable: the same kernels execute in the same order every iteration. R11 (prefetching) already profiles this pattern. R17 can use the same profile for placement:

**Before iteration N starts:**
1. R11 profile says kernel K1 will access pages {P1, P2, P3} on GPU_A
2. R17 checks: are P1, P2, P3 on GPU_A's node?
3. If not, R17 tells R11: "prefetch these to GPU_A's node before K1 starts"
4. R11 issues the prefetch via R17's routing layer

This creates a feedback loop: R17 (topology + routing) -> R11 (prefetch scheduling) -> R10 (actual data movement).

### 8.2 Phase-Aware Placement Presets

Instead of per-page decisions, recognize workload phases and apply bulk placement policies:

| Phase | Placement Policy |
|---|---|
| Forward pass start | Broadcast weights to all nodes |
| Forward pass execution | All reads are local (replicated weights) |
| Backward pass | Gradients allocated locally, no migration needed |
| AllReduce | NCCL handles via R20; OuterLink provides transport |
| Optimizer step | Each node updates its weight shard locally |
| Data loading | Prefetch next batch to local DRAM/VRAM |

### 8.3 Reinforcement Learning for Placement (Future)

Academic research (HDP, Post) uses RL to learn optimal device placement. This requires:
- State: current placement, topology, access patterns
- Action: migrate page P from node A to node B
- Reward: -1 * (total_transfer_time + total_stall_time)

**This is a Phase 3+ enhancement.** Phase 1 should use heuristic placement (affinity + cost-benefit model). Only invest in RL if heuristics leave significant performance on the table.

---

## Open Questions

### Must Answer Before Detailed Planning

1. **How much VRAM overhead does per-GPU access tracking add?** Storing `access_count_per_gpu[4]` for 375K pages = 375K * 4 * 8 bytes = ~12 MB. Acceptable.

2. **What is the write coherence cost for replicated pages?** Invalidation messages are small (~100 bytes per page), but at scale (1000 pages invalidated simultaneously), this could spike. Need bounded invalidation rate.

3. **Can we detect ZeRO Stage 3 AllGather patterns automatically?** If OuterLink sees a burst of AllGather-like traffic at iteration boundaries, it can optimize placement for the next iteration.

4. **What is the actual memory overhead of replication?** A 7B model replicated across 4 nodes = 56 GB of VRAM used for weights alone. With 24 GB per GPU (4 x 24 = 96 GB total VRAM), this leaves only 40 GB for activations, gradients, and optimizer states. ZeRO may be more memory-efficient.

### Can Answer During Implementation

5. What is the optimal migration threshold (MIGRATION_THRESHOLD = 2.0, or higher)?

6. How frequently should rebalancing run? (Every 10 seconds? Every minute?)

7. Should placement scores include power/thermal state? (Hot GPUs may throttle.)

---

## Related Documents

- [R17 README](../README.md) — summary and folder contents
- [research/01-topology-discovery.md](./01-topology-discovery.md) — topology graph that placement queries
- [research/02-routing-algorithms.md](./02-routing-algorithms.md) — routing decisions for data movement
- [R10 Memory Tiering](../../../phase-07-memory-intelligence/R10-memory-tiering/README.md) — tier system this integrates with
- [R11 Speculative Prefetching](../../R11-speculative-prefetching/preplan.md) — prefetch scheduling coordination
- [R12 Memory Deduplication](../../R12-memory-deduplication/README.md) — dedup interaction with replication
- [CONSOLIDATION](../../../../research/CONSOLIDATION-all-research.md) — system architecture and transport stack
