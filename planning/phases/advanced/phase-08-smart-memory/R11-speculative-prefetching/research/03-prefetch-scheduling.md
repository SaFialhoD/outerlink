# R11 Research: Prefetch Scheduling

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Priority:** HIGH

## Purpose

Define the scheduling strategy for speculative prefetches: how far ahead to prefetch, where to buffer data, how to budget bandwidth, how to prioritize competing requests, and how to handle wrong predictions. This is the execution engine that turns predictions into actual data movement.

## TL;DR — The Scheduling Model

```
                    Prediction Engine
                    (from 02-access-pattern-profiling)
                           |
                    ┌──────┴──────┐
                    |             |
              Prefetch Queue   Cancellation
              (priority-ordered) Queue
                    |             |
              Bandwidth Budget    |
              (demand gets 70%,   |
               prefetch gets 30%) |
                    |             |
              ┌─────┴─────┐      |
              |           |      |
         Network      Local Tier |
         Transfer     Migration  |
              |           |      |
              └─────┬─────┘      |
                    |             |
              Staging Buffer ◄───┘ (cancel if prediction wrong)
              (pinned DRAM ring buffer)
                    |
              Promote to Target Tier
              (VRAM when ready)
```

**Key numbers for OuterLink at 100Gbps (ConnectX-5 RDMA):**

| Metric | Value |
|---|---|
| Raw link bandwidth | 12.5 GB/s |
| Practical throughput (RDMA) | ~11 GB/s |
| 64KB page transfer time | ~5.8 us |
| Typical ML kernel duration | 50-5000 us |
| Pages transferable during one kernel | 8-860 pages (0.5-55 MB) |
| Prefetch distance needed | 1-3 kernels ahead |

---

## 1. Prefetch Distance

### 1.1 Definition

Prefetch distance is how far ahead of the current execution point we initiate data transfers. Measured in:
- **Kernels:** number of kernel launches ahead
- **Time:** microseconds/milliseconds before data is needed
- **Pages:** number of pages in the prefetch queue

### 1.2 The Goldilocks Problem

| Distance | Problem |
|---|---|
| Too short | Data not ready when needed — GPU stalls (demand fault) |
| Too far | Prefetched data evicted before use — wasted bandwidth + VRAM pressure |
| Just right | Data arrives just before kernel needs it — zero stalls, minimal VRAM waste |

### 1.3 Calculating Optimal Distance

For OuterLink, the optimal prefetch distance depends on:

```
prefetch_distance_time = transfer_latency + network_rtt + staging_overhead

Where:
  transfer_latency = pages_needed * page_size / effective_bandwidth
  network_rtt      = ~1-5 us (RDMA) or ~50-200 us (TCP)
  staging_overhead = ~2-10 us (DMA setup + page table update)
```

**Example: Transformer layer with 100 pages (6.4 MB) of weights on remote node:**

| Transport | Transfer Time | RTT | Staging | Total | Distance (kernels) |
|---|---|---|---|---|---|
| RDMA (100Gbps) | 582 us | 2 us | 5 us | 589 us | 1-2 kernels |
| TCP (100Gbps) | 640 us | 100 us | 10 us | 750 us | 2-3 kernels |
| OpenDMA (Phase 5) | 582 us | 2 us | 2 us | 586 us | 1-2 kernels |
| USB4 (40Gbps) | 1455 us | 5 us | 5 us | 1465 us | 3-5 kernels |

### 1.4 Adaptive Distance

Static prefetch distance is suboptimal because kernel durations vary. Adaptive approach:

1. **Measure actual kernel execution times** during profiling warmup
2. **Build a timeline model:** for each kernel, know its typical duration and data requirements
3. **Schedule prefetches by deadline:** `prefetch_start = kernel_start - transfer_time - safety_margin`
4. **Adjust safety margin dynamically:** if any stalls detected, increase margin by 20%; if no stalls for 10 iterations, decrease by 10%

Target: safety_margin = 1.5x transfer_time initially, converging to 1.1x in steady state.

### 1.5 Multi-Tier Distance

When data must traverse multiple tiers (e.g., NVMe -> DRAM -> VRAM), the prefetch chain must be scheduled with additive latencies:

```
NVMe -> Remote DRAM: ~500 us (NVMe read) + 50 us (DMA to DRAM)
Remote DRAM -> Network: ~589 us (100Gbps RDMA)
Network -> Local DRAM: ~2 us (RDMA write)
Local DRAM -> Local VRAM: ~10 us (cudaMemcpy pinned)
Total chain: ~1151 us

Prefetch must start 1.2-1.5 ms before data is needed.
```

For multi-tier chains, OuterLink should pipeline the stages: start NVMe read for iteration N+3 data while RDMA-transferring iteration N+2 data while staging iteration N+1 data into VRAM.

---

## 2. Buffer Management

### 2.1 Where Prefetched Data Lives

Prefetched data needs a staging area before being promoted to the target tier. Options:

| Buffer Location | Capacity | Access Latency | Pros | Cons |
|---|---|---|---|---|
| **Local VRAM (reserved region)** | 256 MB - 1 GB | 0 (already in VRAM) | Instant use by GPU | Reduces usable VRAM |
| **Pinned host DRAM** | 1-8 GB | ~5-10 us (cudaMemcpy) | Large, cheap, no VRAM pressure | Extra copy to VRAM needed |
| **Regular host DRAM** | Unlimited | ~10-50 us (page fault path) | No pinning overhead | Slower path, unpinned |

### 2.2 Recommended: Hybrid Ring Buffer

```
Pinned DRAM Ring Buffer (primary staging):
  Size: 2 GB (configurable)
  Structure: circular buffer of 64KB page slots
  Capacity: ~32,768 pages

  Write pointer: RDMA/network writes prefetched data here
  Read pointer: DMA engine copies from here to VRAM

  Pages stay here until:
    a) Promoted to VRAM (when kernel is about to need them)
    b) Evicted (if prediction was wrong or buffer is full)

VRAM Prefetch Window (secondary, for critical-path data):
  Size: 256 MB - 512 MB (reserved from VRAM allocation)
  For: data needed within next 1-2 kernel launches
  Managed as: LRU with pinning for active prefetch targets
```

### 2.3 Buffer Sizing Formula

```
buffer_size = prefetch_distance_pages * page_size * pipeline_depth

Where:
  prefetch_distance_pages = pages needed per kernel * kernels_ahead
  page_size = 64 KB
  pipeline_depth = 2 (double-buffering: one set in use, one being filled)

Example (transformer training, one layer):
  Pages per layer: ~150 (weights) + ~200 (activations) = 350 pages
  Kernels ahead: 2
  Pipeline depth: 2
  Buffer = 350 * 2 * 2 * 64KB = 89.6 MB per layer being prefetched

  For 2-layer lookahead: ~180 MB in staging buffer
```

### 2.4 Buffer Pressure Management

When the staging buffer is >80% full:
1. **Promote aggressively:** move ready data to VRAM even if kernel is 3+ launches away
2. **Drop low-confidence prefetches:** cancel transfers for predictions with confidence < threshold
3. **Throttle new prefetches:** reduce prefetch rate by 50%
4. **Signal to R10:** request eviction of cold VRAM pages to make room

---

## 3. Bandwidth Budgeting

### 3.1 Why Budget Bandwidth

The network link is shared between:
- **Demand fetches:** GPU stalled, waiting for data — highest priority
- **Speculative prefetches:** proactive data movement — lower priority
- **Write-backs:** dirty pages being flushed to remote tiers — medium priority
- **Control traffic:** page table updates, health checks — minimal bandwidth

Without budgeting, aggressive prefetching can starve demand fetches, causing worse performance than no prefetching at all.

### 3.2 Bandwidth Allocation

| Traffic Class | Bandwidth Share | Priority | Preemptible |
|---|---|---|---|
| **Demand fetch** | 70% (guaranteed minimum) | Highest | No |
| **Write-back** | 10% (guaranteed minimum) | High | Only by demand |
| **Speculative prefetch** | 20% (best-effort) | Low | Yes (by demand or write-back) |
| **Control** | <1% | Medium | No |

When demand fetch is idle (no GPU stalls), prefetch can use up to 90% of bandwidth.

### 3.3 Token Bucket Rate Limiter

Implement bandwidth budgeting with a token bucket per traffic class:

```
struct BandwidthBudget {
    demand_tokens: AtomicU64,       // bytes available for demand fetches
    prefetch_tokens: AtomicU64,     // bytes available for prefetches
    writeback_tokens: AtomicU64,    // bytes available for write-backs

    refill_rate_bytes_per_us: u64,  // total link bandwidth
    refill_interval_us: u64,        // typically 100 us

    // Demand steals from prefetch when needed
    demand_can_steal_from: [TrafficClass; 2], // [Prefetch, Writeback]
}
```

### 3.4 Adaptive Budget Based on Prediction Accuracy

Track prefetch hit rate (prefetched pages that were actually used):

| Hit Rate | Action |
|---|---|
| >90% | Increase prefetch budget to 30% |
| 70-90% | Maintain default 20% |
| 50-70% | Reduce to 10% |
| <50% | Reduce to 5% and re-enter profiling mode |

This prevents wasting bandwidth on bad predictions while rewarding accurate prediction with more aggressive prefetching.

---

## 4. Priority Management

### 4.1 Priority Levels

```
Priority 0 (Critical):  Demand fetch — GPU thread is stalled RIGHT NOW
Priority 1 (Urgent):    Prefetch for kernel launching in < 100 us
Priority 2 (Normal):    Prefetch for kernel launching in < 1 ms
Priority 3 (Low):       Prefetch for kernel launching in < 10 ms
Priority 4 (Background): Speculative tier migration (warm data moving closer)
```

### 4.2 Priority Queue Implementation

```rust
struct PrefetchQueue {
    // Min-heap ordered by deadline (earliest deadline first)
    queue: BinaryHeap<PrefetchRequest>,

    // Fast lookup for cancellation
    pending: HashMap<PageId, PrefetchRequestId>,

    // In-flight transfers (being transferred right now)
    in_flight: Vec<PrefetchRequest>,
    max_in_flight: usize,  // concurrency limit (e.g., 8-16 simultaneous transfers)
}

struct PrefetchRequest {
    page_id: PageId,
    source_tier: Tier,
    target_tier: Tier,
    deadline: Timestamp,       // when the page must be ready
    confidence: f32,           // prediction confidence (0.0 - 1.0)
    priority: u8,              // computed from deadline proximity
    cancelable: bool,          // can this be aborted mid-transfer?
    size_bytes: u32,           // typically 64KB
}
```

### 4.3 Preemption Policy

When a demand fetch arrives and bandwidth is saturated by prefetches:

1. **Check in-flight prefetches:** find the lowest-priority cancelable transfer
2. **If cancelable:** abort the transfer, reclaim bandwidth for demand fetch
3. **If not cancelable** (e.g., RDMA transfer already committed): queue demand fetch as Priority 0, it goes next
4. **Track preemption frequency:** if preemptions happen often, reduce prefetch aggressiveness

RDMA transfers are generally not cancelable mid-flight (the NIC DMA engine is committed). However, queued-but-not-started prefetches can always be canceled.

---

## 5. Cancellation and Wrong Predictions

### 5.1 When to Cancel

| Trigger | Action |
|---|---|
| Prediction invalidated (different kernel launched than expected) | Cancel all queued prefetches for the predicted sequence |
| Buffer pressure >90% | Cancel lowest-confidence queued prefetches |
| Demand fetch for different page set | Suggests phase change — cancel and re-profile |
| Iteration boundary (sync point) | Clear stale prefetch queue entries from previous iteration |
| Application calls `cudaFree` on prefetch target | Cancel + discard any staged data for those pages |

### 5.2 Cancellation Cost

| Transfer State | Cancellation Cost |
|---|---|
| Queued (not started) | Free — just remove from queue |
| In-flight (RDMA write ongoing) | Cannot cancel — let it complete, discard data |
| Staged in DRAM buffer | Free — mark buffer slot as available |
| Promoted to VRAM | Must evict — cost is eviction + wasted VRAM time |

### 5.3 Misprediction Recovery

When a misprediction is detected:

1. **Immediate:** flush prefetch queue, issue demand fetch for actual needed pages
2. **Short-term:** mark the prediction model as unreliable for this phase, increase demand fetch budget
3. **Medium-term:** collect new profiling data for the unexpected pattern
4. **Long-term:** if mispredictions are chronic (>20% rate over 100 iterations), disable prefetching for this workload and fall back to demand-only

### 5.4 Prefetch Accounting

Track metrics per prediction model:

```rust
struct PrefetchStats {
    total_prefetches: u64,
    hits: u64,              // prefetched page was used before eviction
    misses: u64,            // prefetched page evicted before use
    late: u64,              // page used but arrived after demand fetch already happened
    canceled: u64,          // prefetch aborted before completion
    demand_stalls: u64,     // GPU stalled waiting for data (prefetch failed to prevent)

    hit_rate: f32,          // hits / total
    coverage: f32,          // 1.0 - (demand_stalls / total_accesses)
    timeliness: f32,        // hits / (hits + late)
    bandwidth_waste: f32,   // (misses * page_size) / total_bytes_transferred
}
```

---

## 6. Multi-Tier Prefetch Chains

### 6.1 The Pipeline Model

For data in deep tiers (NVMe), a single-hop prefetch is too slow. Instead, pipeline across tiers:

```
Time: ──────────────────────────────────────────────────►

NVMe Read (iter N+3):  [████████████]
                              ↓
Remote DRAM Stage:            [████]
                                ↓
RDMA Transfer:                  [████████]
                                       ↓
Local DRAM Stage:                      [██]
                                        ↓
VRAM Promote:                           [█]
                                         ↓
GPU Uses (iter N+3):                     [████████████████]

Meanwhile, iter N+2 data is already in local DRAM,
and iter N+1 data is already in VRAM.
```

### 6.2 Chain Scheduling

Each tier transition is scheduled independently with its own deadline:

```rust
struct TierChain {
    page_id: PageId,
    current_tier: Tier,
    target_tier: Tier,     // usually Tier::LocalVRAM

    hops: Vec<TierHop>,    // ordered list of transitions
}

struct TierHop {
    from: Tier,
    to: Tier,
    estimated_latency_us: u64,
    deadline: Timestamp,   // must complete by this time
    status: HopStatus,     // Pending, InFlight, Complete, Failed
}
```

### 6.3 Tier Latency Table

| Hop | Estimated Latency | Bandwidth Limit |
|---|---|---|
| NVMe -> Local DRAM | 100-500 us (depends on read size) | ~3.5 GB/s (Gen4 NVMe) |
| Local DRAM -> Local VRAM | 5-15 us (pinned cudaMemcpy) | ~25 GB/s (PCIe 4.0 x16) |
| Remote DRAM -> Network -> Local DRAM | 50-600 us (depends on size + transport) | ~11 GB/s (100Gbps RDMA) |
| Remote VRAM -> Remote DRAM -> Network -> Local DRAM | 70-650 us | Limited by slowest hop |
| Remote VRAM -> Network -> Local VRAM (OpenDMA) | 50-600 us | ~11 GB/s (direct DMA) |

### 6.4 Chain Optimization

- **Skip tiers when possible:** If data is in remote VRAM and OpenDMA is available, go direct to local VRAM (skip DRAM staging)
- **Batch transfers:** Group pages going the same route into bulk transfers (amortize setup overhead)
- **Coalesce adjacent pages:** If pages 100-110 are all needed, transfer as one 704KB bulk instead of 11 separate 64KB transfers

---

## 7. Integration with R10 Eviction Policy

### 7.1 Coordination Points

Prefetching and eviction are two sides of the same coin. They must coordinate:

| Situation | Required Coordination |
|---|---|
| Prefetch wants to promote page to VRAM, but VRAM is full | Tell R10 to evict a cold page first |
| R10 wants to evict a page that has an active prefetch targeting it | Cancel the prefetch or defer the eviction |
| Prefetched page is marked "about to be used" | R10 should give it eviction immunity for a window |
| Prefetch staging buffer is full | Signal R10 to proactively evict cold pages from staging |

### 7.2 Prefetch Immunity

Pages that are prefetched and awaiting use should have temporary eviction immunity:

```rust
struct PageMetadata {
    // ... existing R10 fields ...
    prefetch_immunity_until: Option<Timestamp>,  // don't evict before this time
    prefetch_confidence: f32,                     // low confidence = shorter immunity
}
```

Immunity duration = estimated time until kernel needs the page + safety margin. If the kernel doesn't use it within the immunity window, immunity expires and R10 can evict normally.

### 7.3 ARC Integration

R10 uses ARC (Adaptive Replacement Cache) for eviction. Prefetched pages should enter ARC's recency list (T1) rather than frequency list (T2), so they don't pollute the frequency-based cache with speculative data. If the prefetched page is actually used, it naturally promotes to T2.

---

## 8. Putting It All Together: Scheduler State Machine

```
                         ┌─────────────┐
                         │   PROFILING  │ ◄── First 1-3 iterations
                         │ (learn patterns) │
                         └──────┬──────┘
                                │ Pattern detected
                                ▼
                         ┌─────────────┐
                    ┌───►│  PREDICTING  │
                    │    │ (generating   │
                    │    │  prefetch     │
                    │    │  requests)    │
                    │    └──────┬──────┘
                    │           │ Requests queued
                    │           ▼
                    │    ┌─────────────┐
                    │    │ TRANSFERRING │
                    │    │ (data moving │
                    │    │  across tiers)│
                    │    └──────┬──────┘
                    │           │ Data staged
                    │           ▼
                    │    ┌─────────────┐
                    │    │  PROMOTING   │
                    │    │ (staging →   │
                    │    │  VRAM)       │
                    │    └──────┬──────┘
                    │           │ Page ready
                    │           ▼
                    │    ┌─────────────┐
                    │    │   ACTIVE     │
                    │    │ (page in use)│
                    │    └──────┬──────┘
                    │           │ Iteration end
                    └───────────┘

Misprediction at any stage → drop back to PROFILING for 1-2 iterations.
```

### 8.1 Scheduler Thread Model

```
Thread 1: Prediction Thread
  - Runs on CPU
  - Reads interception log from LD_PRELOAD hooks
  - Maintains pattern model
  - Generates prefetch requests
  - Adds to prefetch queue

Thread 2: Transfer Thread
  - Pulls from prefetch queue (priority-ordered)
  - Issues RDMA/network transfers
  - Manages bandwidth budget
  - Handles cancellations

Thread 3: Staging Thread
  - Monitors staging buffer
  - Promotes data to VRAM when deadline approaches
  - Coordinates with R10 for VRAM space
  - Issues cudaMemcpyAsync for DRAM→VRAM moves
```

### 8.2 Latency Targets

| Metric | Target | Rationale |
|---|---|---|
| Prediction latency | <100 us | Must complete before next kernel launch |
| Queue-to-transfer latency | <50 us | RDMA post should be near-instant |
| Staging-to-VRAM latency | <15 us | Single pinned cudaMemcpy |
| End-to-end (remote VRAM → local VRAM) | <700 us | Must be < typical kernel duration |
| Stall rate | <5% of kernels | 95%+ kernels should find data ready |
| Bandwidth waste | <15% | 85%+ of prefetched bytes should be used |

---

## 9. Special Cases

### 9.1 CUDA Graphs

When the application uses CUDA Graphs (`cuGraphLaunch`), the entire execution DAG is known ahead of time. This is the best case for prefetching:
- Parse the graph to extract complete kernel sequence and data dependencies
- Pre-compute the entire prefetch schedule for the graph
- Execute prefetches in perfect lockstep with graph execution

### 9.2 Dynamic Batching

Some inference workloads use dynamic batching (variable batch size per iteration). This changes the number of pages needed per kernel. Handle by:
- Detecting batch size from kernel arguments (grid dimensions often encode batch size)
- Scaling prefetch page count proportionally to batch size
- Keeping a per-batch-size prediction model

### 9.3 Multi-GPU (Data Parallel)

In data-parallel training, each GPU processes a different data shard but the same model weights. Prefetching model weights can be shared/coordinated:
- All GPUs need the same weights at the same time
- One prefetch request can serve multiple consumers (multicast prefetch)
- Data shards are unique per GPU — prefetch independently

### 9.4 Pipeline Parallelism

In pipeline-parallel training, different GPUs process different layers. As activations flow forward:
- GPU N finishes layer L, sends activations to GPU N+1
- GPU N+1 needs layer L+1 weights ready before activations arrive
- Prefetch weights for layer L+1 as soon as GPU N starts layer L

This is the ideal case: the pipeline structure itself tells us exactly what to prefetch and when.

---

## Related Documents

- [01-existing-prefetching-systems.md](./01-existing-prefetching-systems.md) — survey of existing approaches
- [02-access-pattern-profiling.md](./02-access-pattern-profiling.md) — how patterns are detected
- [R10 Memory Tiering](../../phase-07-memory-intelligence/R10-memory-tiering/README.md) — tier definitions, ARC eviction
- [R17 Topology-Aware Scheduling](../../R17-topology-aware-scheduling/README.md) — which network path to use

## Open Questions

- [ ] What is the right max_in_flight count for RDMA prefetches? Too many concurrent RDMA operations can saturate the NIC's queue pairs. Need ConnectX-5 benchmarking data.
- [ ] Should prefetch scheduling be centralized (one scheduler for the whole cluster) or distributed (each node schedules its own prefetches)? Centralized gives global optimality but adds coordination overhead.
- [ ] How does PCIe bandwidth contention between RDMA NIC and GPU affect prefetch-to-VRAM promotion? Both compete for PCIe lanes.
- [ ] For the OpenDMA path (NIC writes directly to VRAM via BAR1), does the prefetch bypass the staging buffer entirely? If so, the scheduling pipeline simplifies significantly.
- [ ] What happens when two nodes prefetch from the same remote node simultaneously? Need fairness policy at the source node.
