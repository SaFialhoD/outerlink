# R19 Research: Coherency Protocols and Thrashing Prevention

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft
**Purpose:** Define the coherency protocol for OuterLink's distributed GPU memory, analyze thrashing scenarios and prevention mechanisms, and specify how R19 interacts with R11 (prefetching) and R12 (deduplication) to maintain consistency without destroying performance.

---

## 1. Page Ownership Models

### 1.1 Single-Writer / Multiple-Reader (SWMR)

The fundamental invariant for correctness: at any point in time, a memory page is either:
- **Shared-Read:** Multiple nodes hold read-only copies, no node may write
- **Exclusive:** Exactly one node holds a read-write copy, no other node has a valid copy

This is the simplest model that guarantees consistency. It maps directly to GPU workload patterns:

| Data Type | Access Pattern | SWMR State |
|-----------|---------------|-----------|
| Model weights (inference) | All GPUs read, none write | Shared-Read |
| Model weights (training) | All GPUs read, optimizer writes | Shared-Read -> Exclusive (optimizer step) -> Shared-Read |
| Activations (pipeline) | GPU N writes, GPU N+1 reads | Exclusive (GPU N) -> Exclusive (GPU N+1) |
| Gradients (all-reduce) | Each GPU writes local, then reduce | Exclusive per GPU, then coordinated reduce |
| KV cache (inference) | One GPU writes, same GPU reads | Exclusive (never shared) |
| Input data (data-parallel) | Read by assigned GPU | Exclusive (each GPU gets its own partition) |

### 1.2 Home Node Model

Every page has a **home node** --- the node that owns the authoritative copy and tracks who has cached copies:

- **Home node responsibilities:**
  - Maintain the canonical page data
  - Track the set of nodes holding read copies (the "sharer set")
  - Process ownership requests (Shared-Read -> Exclusive transitions)
  - Send invalidation messages when ownership changes
  - Resolve conflicts when multiple nodes request Exclusive simultaneously

- **Home assignment:** Initially, the node that allocated the page is the home. Home can be transferred (e.g., when a page is heavily accessed from a remote node, migrate the home there).

### 1.3 Why Not MESI?

Full MESI (Modified/Exclusive/Shared/Invalid) is designed for hardware caches with snooping buses. At network scale:

| MESI Feature | Network Scale Problem | Our Simplification |
|-------------|----------------------|-------------------|
| Exclusive (clean) state | Requires tracking clean vs dirty on every access | Not worth the protocol overhead; treat Exclusive as always dirty |
| Snooping (broadcast) | O(N) messages per write | Use directory (home node) instead |
| Cache-line granularity (64B) | Too many entries to track | 64KB pages reduce tracking overhead by 1000x |
| Hardware-speed transitions | Network latency makes transitions 1000x slower | Batch transitions, amortize with prefetching |

We use a simplified protocol with three states per page:

| State | Meaning | Allowed Operations |
|-------|---------|-------------------|
| **I** (Invalid) | No local copy | Must fetch before access |
| **S** (Shared) | Read-only local copy, home tracks us | Read only; write requires upgrade to E |
| **E** (Exclusive) | We are the sole owner | Read and write; home has no copy |

---

## 2. Directory-Based Coherency Protocol

### 2.1 Directory Structure

Each page's directory entry (stored at the home node) contains:

```
PageDirectory {
    page_id: u64,              // Global page identifier
    state: PageState,          // I, S, or E
    owner: Option<NodeId>,     // Current exclusive owner (if state == E)
    sharers: BitSet<NodeId>,   // Set of nodes with S copies (if state == S)
    home_node: NodeId,         // Where the canonical copy lives
    version: u64,              // Incremented on every write (for stale detection)
    last_writer: NodeId,       // Who last wrote this page
    fault_count: u32,          // Thrashing detection counter
    last_fault_time: Instant,  // When the last fault/transition occurred
}
```

### 2.2 State Transitions

#### Read Miss (node N wants to read page P, currently Invalid at N)

```
N -> Home: READ_REQUEST(P)
Home checks state:
  If state == I (no one has it):
    Home -> N: READ_REPLY(P, data, version)
    Set state = S, add N to sharers
  If state == S (others reading):
    Home -> N: READ_REPLY(P, data, version)
    Add N to sharers
  If state == E (owned by O):
    Home -> O: FETCH_REQUEST(P)  // Get latest data from owner
    O -> Home: FETCH_REPLY(P, data)
    Home -> N: READ_REPLY(P, data, version)
    Set state = S, add {O, N} to sharers
    O transitions from E to S
```

#### Write Miss (node N wants to write page P, currently Invalid or S at N)

```
N -> Home: WRITE_REQUEST(P)
Home checks state:
  If state == I:
    Home -> N: WRITE_REPLY(P, data, version)
    Set state = E, owner = N, clear sharers
  If state == S:
    For each sharer S_i in sharers (except N):
      Home -> S_i: INVALIDATE(P, version)
      S_i -> Home: INVALIDATE_ACK(P)
    Home -> N: WRITE_REPLY(P, data, version)
    Set state = E, owner = N, clear sharers
  If state == E (owned by O, O != N):
    Home -> O: INVALIDATE(P, version)
    O -> Home: INVALIDATE_ACK(P, data)  // Flush dirty data
    Home -> N: WRITE_REPLY(P, data, version)
    Set state = E, owner = N
```

#### Eviction (node N wants to free local copy of page P)

```
N -> Home: EVICT_NOTIFY(P, dirty_data_if_modified)
Home removes N from sharers or clears owner
If data was dirty, home updates canonical copy
```

### 2.3 Optimizations

**Migratory sharing:** If a page repeatedly transitions S -> E -> S -> E between two nodes, the home node can detect this pattern and migrate the home to the node that writes most frequently, reducing the number of network hops.

**Silent eviction for Shared pages:** When a node evicts a Shared copy, it can simply discard it without notifying the home (since the home still has the canonical copy). The home will learn the node no longer has the page when it tries to send an invalidation and gets no response. This is a common DSM optimization that reduces eviction traffic.

**Bulk invalidation:** When transitioning a page from S to E, if the sharer set is large, send a single multicast invalidation instead of N point-to-point messages. This is especially relevant for model weights going from shared-read to exclusive during optimizer step.

---

## 3. Thrashing Detection and Prevention

### 3.1 What Causes Thrashing

Thrashing occurs when two or more nodes repeatedly request exclusive access to the same page, causing it to "bounce" between nodes. Each transition involves:
1. Invalidation message to current owner (~1-2 us RDMA)
2. Data transfer from old owner to new owner (~2-5 us for 64KB over RDMA)
3. Page table updates on both sides (~10-20 us)
4. Total per-bounce: ~15-30 us

If two nodes bounce a page every kernel launch (~100 us), thrashing overhead consumes 15-30% of compute time. With more nodes, it gets worse.

### 3.2 Thrashing Scenarios in GPU Workloads

| Scenario | Pattern | Severity |
|----------|---------|----------|
| **False sharing:** Two GPUs write different regions of the same 64KB page | High frequency bouncing | High (but rare with 64KB pages) |
| **Producer-consumer:** GPU A writes output, GPU B reads it as input | Alternating E(A) -> E(B) -> E(A) | Medium (normal, not pathological) |
| **Optimizer step:** All GPUs read weights (S), optimizer writes (E), all read again (S) | Periodic S -> E -> S | Low (infrequent, ~1/iteration) |
| **Random write sharing:** Multiple GPUs write to same page unpredictably | True contention | High (should be avoided by scheduling) |

### 3.3 Detection Mechanisms

#### Bounce Counter

Track per-page state transitions at the home node:

```rust
struct ThrashingDetector {
    bounce_count: u32,        // State transitions in current epoch
    epoch_start: Instant,     // When current counting epoch began
    epoch_duration: Duration, // How long an epoch lasts (e.g., 10 ms)
    threshold: u32,           // Bounces per epoch before declaring thrashing
}

impl ThrashingDetector {
    fn record_transition(&mut self) -> ThrashingAction {
        self.bounce_count += 1;
        if self.bounce_count >= self.threshold {
            return ThrashingAction::Mitigate;
        }
        ThrashingAction::None
    }

    fn epoch_reset(&mut self) {
        self.bounce_count = 0;
        self.epoch_start = Instant::now();
    }
}
```

**Recommended parameters:**
- Epoch duration: 10 ms (covers ~100 kernel launches)
- Threshold: 5 bounces per epoch
- These are conservative --- a page that bounces 5 times in 10 ms is definitely thrashing

#### Per-Node Fault Rate

Monitor the rate of page faults (READ_REQUEST / WRITE_REQUEST) per node. A sudden spike in fault rate indicates potential thrashing or working set change. If combined with a high invalidation rate, it is thrashing.

### 3.4 Mitigation Strategies

#### Strategy 1: Shared-Read Promotion (Primary)

When a page is bouncing between readers and writers, promote it to shared-read with explicit synchronization:

- If the page is being read by many and written by one, switch to a **write-update** protocol for that page: the writer broadcasts diffs to all readers instead of invalidating them
- This amortizes the write cost across the update interval instead of per-read-request

#### Strategy 2: Page Pinning with Minimum Residency

When thrashing is detected, pin the page to the current owner for a minimum duration:

```
On thrashing detection for page P at home:
  Set P.min_residency = 2 * epoch_duration
  Reject ownership transfer requests until min_residency expires
  Requestors receive NACK with retry_after timestamp
  Stalled requestors use stale data (if they have an S copy) or block
```

**Caution:** Page pinning can cause priority inversion. If a high-priority kernel needs the page but it is pinned to a low-priority node, the high-priority kernel stalls. Use pinning only as a last resort.

#### Strategy 3: Replication for Read-Heavy Thrashing

If a page is thrashing because many nodes want to read it and one node wants to write it (common for training weights):

1. Detect the pattern: many S -> E -> S transitions with the same writer
2. Switch to **write-broadcast** for that page: writer sends update to all sharers instead of invalidating
3. Sharers receive the update and apply it locally (no need to re-fetch)
4. Cost: O(sharers) writes per update, but eliminates all read faults

This is particularly effective for model weights during training:
- Forward pass: all GPUs read weights (Shared)
- Optimizer step: one GPU (or all GPUs with identical updates) writes new weights
- Write-broadcast sends the weight delta to all GPUs simultaneously
- Next forward pass: no faults needed, all GPUs already have updated weights

#### Strategy 4: Adaptive Granularity

If thrashing is caused by false sharing (two GPUs accessing different parts of the same 64KB page):
- Split the 64KB page into sub-pages (e.g., 4KB blocks) for coherency tracking
- Each sub-page has its own state (I/S/E) within the 64KB page
- Only invalidate the sub-page that was actually written
- This adds metadata overhead (16 sub-page entries per page) but eliminates false sharing

**When to activate:** Only when thrashing is detected AND the access pattern shows different byte ranges being written by different nodes (detectable from kernel argument analysis).

---

## 4. Interaction with R11 Prefetching

### 4.1 Prefetching Prevents Most Faults

R11's speculative prefetching is the primary mechanism for avoiding page faults. The relationship:

```
Page Fault Rate = (1 - Prefetch Hit Rate) * Access Rate
```

If R11 achieves 90% hit rate and a kernel accesses 100 pages, only ~10 pages trigger faults. At 95% hit rate, only ~5. The goal is to make R19's fault handling the rare exception, not the common case.

### 4.2 Prefetch as Coherency-Aware Operation

When R11 issues a prefetch request, it must respect coherency:

1. **Prefetch for read:** Equivalent to a READ_REQUEST. Home adds the prefetching node to the sharer set. If page is Exclusive elsewhere, the owner must downgrade to Shared.

2. **Prefetch for write:** Equivalent to a WRITE_REQUEST. Home invalidates all sharers and grants Exclusive to the prefetching node. Use this when R11 predicts that the kernel will write to the page.

3. **Speculative prefetch:** If the prefetch might be wrong (low confidence), use a **tentative read** that doesn't add the node to the sharer set. The page is fetched into a local staging buffer but not installed in the page table. If the prediction is correct, a fast local install happens; if wrong, the data is discarded without coherency cleanup.

### 4.3 Prefetch Cancellation and Coherency

If R11 cancels a prefetch (misprediction):
- If the page was prefetched as Shared: notify home to remove from sharer set
- If the page was prefetched as Exclusive: notify home to release ownership
- If using tentative read: no coherency action needed (page was never registered)

### 4.4 Combined Latency Analysis

| Scenario | Latency | Frequency (target) |
|----------|---------|-------------------|
| Prefetch hit (page already local) | 0 us (no fault) | >90% |
| Pre-launch map (page in staging buffer) | 10-30 us (cuMemMap) | ~5% |
| Demand fault (page not local, RDMA fetch) | 20-50 us (fetch + map) | <5% |
| Demand fault (TCP fetch) | 100-200 us | <1% (fallback only) |
| Kernel crash + restart (worst case) | 500-5000 us | <0.1% |

---

## 5. Interaction with R12 Deduplication

### 5.1 Deduped Pages Are Always Shared-Read

R12 deduplicates identical pages across nodes. By definition, deduped pages are read-only (any write triggers COW, creating a private copy). This means:

- Deduped pages are always in Shared state in the coherency protocol
- They never need invalidation (no one writes to them)
- They never cause thrashing (no write contention)
- The home node for a deduped page serves as the canonical copy provider

### 5.2 COW and Coherency

When a node writes to a deduped page:
1. R12's COW mechanism creates a private copy
2. The private copy starts in Exclusive state (owned by the writing node)
3. The original deduped page remains Shared for all other nodes
4. The home node's directory is updated: the writing node is removed from the sharer set of the deduped page and becomes the owner of a new (non-deduped) page

### 5.3 Dedup + Prefetch Integration

When R11 prefetches a page that is deduped:
- If any node in the cluster already has it, it can be fetched from the nearest copy (not necessarily the home)
- This is a read-sharing optimization: the nearest node with a Shared copy is the prefetch source
- R17 (Topology-Aware Scheduling) picks the nearest source based on network topology

---

## 6. NVIDIA UVM's Anti-Thrashing Approach (Reference)

NVIDIA's UVM driver implements thrashing detection internally:

### 6.1 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `uvm_perf_thrashing_enable` | true | Enable thrashing detection |
| `uvm_perf_thrashing_threshold` | 3 | Faults before declaring thrashing |
| `uvm_perf_thrashing_pin_threshold` | 10 | Faults before pinning |
| `uvm_perf_thrashing_lapse_usec` | 1000 | Time window for counting (1 ms) |
| `uvm_perf_thrashing_nap` | 1 ms | Throttle delay for thrashing pages |
| `uvm_perf_thrashing_epoch` | 20 ms | Epoch for resetting counters |

### 6.2 UVM's Response to Thrashing

When thrashing is detected (threshold exceeded within lapse window):
1. **Dual mapping:** Both CPU and GPU get simultaneous access to the page. The page is not migrated; instead, both memory systems can access it (at lower bandwidth due to PCIe/NVLink coherency overhead).
2. **Throttling:** SM throttling reduces the rate of GPU accesses to the thrashing page, giving the CPU time to work.
3. **Pinning:** If thrashing persists past the pin_threshold, the page is pinned to its current location and not migrated for the duration of the epoch.

### 6.3 Lessons for OuterLink

- A threshold of 3 faults is very aggressive --- suitable for CPU-GPU within one machine
- For network-scale thrashing, our threshold should be higher (5-10) due to higher per-transition cost
- Dual mapping is not possible across network nodes (no hardware coherency) --- we must use replication or pinning instead
- Throttling the GPU is a last resort that reduces compute throughput; prefer replication or scheduling changes

---

## 7. Worst-Case Latency Analysis

### 7.1 Single Page Fault (Cold Miss)

```
Event                           Time (RDMA)    Time (TCP)
-----------------------------------------------
Detect access to unmapped page:    0 us          0 us
  (via pre-launch check or crash)
Send READ_REQUEST to home:         ~1 us         ~25 us
Home processes request:            ~1 us         ~1 us
Home sends data (64KB):            ~2 us         ~50 us
  (RDMA: inline or DMA)
  (TCP: copy + send + receive)
Install page via cuMemMap:         ~15 us        ~15 us
Update local page table:           ~1 us         ~1 us
-----------------------------------------------
Total (single fault):              ~20 us        ~92 us
```

### 7.2 Write Fault with Invalidation

```
Event                           Time (RDMA)    Time (TCP)
-----------------------------------------------
Send WRITE_REQUEST to home:        ~1 us         ~25 us
Home sends INVALIDATE to N sharers:
  (parallel, wait for slowest)     ~2 us         ~50 us
Each sharer sends ACK:             ~1 us         ~25 us
Home sends data + grant:           ~2 us         ~50 us
Install page via cuMemMap:         ~15 us        ~15 us
-----------------------------------------------
Total (write fault, N sharers):    ~21 us        ~165 us
  (N doesn't affect much with RDMA due to parallel invalidation)
```

### 7.3 Thrashing Scenario (Worst Case)

If two nodes alternate exclusive access every kernel launch:
- Each transition: ~21 us (RDMA)
- 100 kernel launches/second = 100 transitions = 2.1 ms overhead/second
- At 10% of compute time threshold: thrashing detection triggers
- Mitigation: pin for 20 ms, one node uses stale data or blocks

### 7.4 Cascading Fault (Multiple Pages)

If a kernel needs M pages that are all remote:
- Pre-launch mapping can issue M parallel RDMA READ requests
- With ConnectX-5 supporting ~100+ outstanding RDMA operations
- M pages fetched in parallel: ~max(2 us * M / bandwidth, per-page-overhead * M)
- For M=100, 64KB pages: 6.4 MB of data, at 12.5 GB/s (100 Gbps) = ~0.5 ms transfer + ~1.5 ms cuMemMap overhead
- Total: ~2 ms for 100 pages (bulk pre-launch mapping)

---

## 8. Recommended Coherency Design for OuterLink

### 8.1 Protocol Summary

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Consistency model | SWMR with home nodes | Maps to GPU workload patterns |
| Page states | I / S / E (3-state) | Simpler than MESI; E implies dirty |
| Directory | Distributed (each home node tracks its pages) | No single point of failure |
| Coherency granularity | 64KB pages (matching R10) | Amortize network cost per transition |
| Read protocol | One-sided RDMA READ from nearest copy | Bypass remote CPU for data access |
| Write protocol | Request Exclusive from home, invalidate sharers | Standard directory-based invalidation |
| Thrashing detection | Bounce counter per page, 5-bounce threshold in 10ms epoch | Detect and respond within one training iteration |
| Thrashing mitigation | Priority: (1) shared-read promotion, (2) write-broadcast, (3) page pinning | Escalating response based on pattern |
| Prefetch coherency | Tentative read for speculative prefetch; full S/E for confirmed | Avoid coherency overhead for mispredictions |
| Dedup coherency | Deduped pages are permanently S until COW break | Zero write contention for shared data |

### 8.2 Page State Machine

```
           READ_REQUEST
    I -----------------------> S
    |                          |
    |  WRITE_REQUEST           | WRITE_REQUEST
    |                          | (invalidate others)
    v                          v
    E <----------------------- S
         UPGRADE_REQUEST
    |
    | EVICT / INVALIDATE
    v
    I
```

Transitions:
- I -> S: Read fault, fetch from home (READ_REQUEST)
- I -> E: Write fault, fetch + get exclusive (WRITE_REQUEST)
- S -> E: Write upgrade, invalidate other sharers (UPGRADE_REQUEST)
- S -> I: Eviction (local decision) or invalidation (from home on behalf of new writer)
- E -> S: Another node reads, owner downgrades (FETCH_REQUEST from home)
- E -> I: Eviction or transfer of ownership

### 8.3 Message Types

| Message | Direction | Payload | Purpose |
|---------|-----------|---------|---------|
| READ_REQUEST | Node -> Home | page_id | Request Shared copy |
| READ_REPLY | Home -> Node | page_id, data, version | Provide data + Shared state |
| WRITE_REQUEST | Node -> Home | page_id | Request Exclusive ownership |
| WRITE_REPLY | Home -> Node | page_id, data, version | Grant Exclusive + data |
| UPGRADE_REQUEST | Node -> Home | page_id | S -> E upgrade (node already has data) |
| UPGRADE_REPLY | Home -> Node | page_id, version | Grant Exclusive (no data transfer) |
| INVALIDATE | Home -> Sharer | page_id, version | Demand invalidation of S copy |
| INVALIDATE_ACK | Sharer -> Home | page_id | Confirm invalidation |
| FETCH_REQUEST | Home -> Owner | page_id | Request data from E owner (for downgrade) |
| FETCH_REPLY | Owner -> Home | page_id, data | Return data + downgrade to S |
| EVICT_NOTIFY | Node -> Home | page_id, dirty_flag, data_if_dirty | Voluntary eviction |
| THRASH_NOTIFY | Home -> Node | page_id, action | Thrashing mitigation instruction |

---

## 9. Open Questions

1. **Home node migration policy:** When should a page's home node be transferred? Options: never (simplest), when the majority of accesses come from a remote node (adaptive), or when the page is evicted from the original home (forced).

2. **Silent eviction timeout:** If a node silently evicts a Shared page, how long before the home node detects this? The home will discover the stale sharer entry when it tries to send an INVALIDATE and gets no response. Should we have a heartbeat mechanism instead?

3. **Version number overflow:** With u64 versions, overflow is not a practical concern (2^64 writes). But should versions be per-page or global (Lamport clock)?

4. **Interaction with CUDA graphs:** CUDA graphs execute a pre-recorded sequence of kernels. Can we pre-compute the coherency transitions for an entire graph and batch them?

5. **Partial page writes:** If a kernel writes only part of a 64KB page, do we send the entire page on FETCH_REPLY or just the dirty sub-range? Sending the full page is simpler; sending diffs saves bandwidth but adds complexity (like TreadMarks).

6. **Multi-home for popular pages:** For pages read by many nodes (e.g., model weights), having a single home creates a hot spot. Should popular pages have multiple "homes" that can serve read requests? This is similar to DNS-style replication.

---

## Related Documents

- [01-gpu-page-fault-mechanisms.md](./01-gpu-page-fault-mechanisms.md) --- GPU-side fault handling mechanisms
- [02-distributed-shared-memory.md](./02-distributed-shared-memory.md) --- prior art survey
- [R10 Memory Tiering](../../R10-memory-tiering/README.md) --- page table and tier design
- [R11 Speculative Prefetching](../R11-speculative-prefetching/preplan.md) --- prevents most faults
- [R12 Memory Deduplication](../R12-memory-deduplication/preplan.md) --- shared read-only pages
- [R17 Topology-Aware Scheduling](../R17-topology-aware-scheduling/README.md) --- picks nearest copy for prefetch
- [NVIDIA UVM Thrashing Source Analysis](https://eunomia.dev/zh/blog/posts/nvidia-open-driver-analysis/)
- [DSM Thrashing Overview](https://www.geeksforgeeks.org/distributed-system-thrashing-in-distributed-shared-memory/)
