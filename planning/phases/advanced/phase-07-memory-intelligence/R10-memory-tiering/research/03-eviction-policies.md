# R10 Research: Eviction Policies for GPU Memory Tiering

**Date Created:** 2026-03-25
**Date Updated:** 2026-03-25
**Status:** DRAFT
**Author:** Research Agent

## Purpose

Evaluate cache eviction/replacement policies for use in OuterLink's tier manager. The eviction policy determines which pages to demote (move to a slower tier) when a faster tier is full. This directly impacts performance — a bad policy causes thrashing, a good policy keeps hot data where it belongs.

---

## 1. Policy Landscape

### Classical Policies

| Policy | Tracks | Time Complexity | Space Overhead | Scan Resistant | Adaptive |
|--------|--------|----------------|----------------|----------------|----------|
| LRU | Last access time | O(1) | Low (linked list) | No | No |
| LFU | Access frequency | O(log n) | Medium (heap) | Yes | No |
| CLOCK | Access bit | O(1) amortized | Very low (bit per page) | No | No |
| ARC | Recency + frequency | O(1) | 2x LRU | Yes | Yes |
| CAR | Recency + frequency | O(1) | 2x CLOCK | Yes | Yes |
| 2Q | Short-term + long-term | O(1) | ~2x LRU | Yes | Partially |
| LeCaR | Recency + frequency (ML) | O(1) | ~2x LRU | Yes | Yes (online learning) |

### GPU-Specific Context

NVIDIA GPU hardware caches (L1/L2) use LRU natively with optional priority hints (`evict_first`, `evict_last`). NVIDIA UVM's page eviction between VRAM and host memory also uses LRU. No GPU system in production uses anything more sophisticated for inter-tier eviction.

This is an opportunity — OuterLink can differentiate by using a smarter policy.

---

## 2. Policy Deep Dive

### LRU (Least Recently Used)

**How it works:** Evict the page that was accessed least recently. Maintained as a doubly-linked list — accessed pages move to the head, eviction takes from the tail.

**For GPU workloads:**
- Works well for streaming/sequential access (iterating through data once)
- Fails catastrophically for scan patterns (large sequential scan evicts everything, even frequently-used pages)
- Fails for cyclic patterns larger than the cache (all pages equally "old")
- NVIDIA UVM uses LRU — and NVIDIA's own documentation acknowledges severe thrashing with random access patterns under oversubscription

**Strengths:** Simple, O(1), low overhead, well-understood.
**Weaknesses:** Not scan-resistant. A single table scan or data load operation evicts the entire working set of all concurrent GPU kernels.

**Verdict for OuterLink:** Insufficient as the sole policy. Too vulnerable to scan patterns, which are common in data loading, initialization, and checkpoint operations.

### LFU (Least Frequently Used)

**How it works:** Evict the page with the lowest access count. Pages that are accessed many times stay cached.

**For GPU workloads:**
- Works well for workloads with stable hot sets (model weights in inference)
- Fails for phase changes — a page that was hot in training phase 1 stays cached through phase 2 even if never accessed (cache pollution)
- Frequency counters accumulate over time, making it hard for new-but-hot pages to compete with historically-hot pages

**Strengths:** Resistant to scans (a single scan doesn't inflate frequency counts).
**Weaknesses:** Slow to adapt to phase changes. Counter management is complex (aging, overflow).

**Verdict for OuterLink:** Good for stable workloads (inference). Poor for training or mixed workloads where access patterns shift between kernel launches.

### CLOCK (Second-Chance)

**How it works:** Circular buffer with a "reference bit" per page. On eviction scan, pages with the bit set get a second chance (bit cleared). Pages with the bit unset are evicted.

**For GPU workloads:**
- Extremely low overhead (1 bit per page)
- Approximates LRU with less bookkeeping
- Used by most OS kernels for physical page management

**Strengths:** Very fast, tiny memory overhead, good enough for most cases.
**Weaknesses:** Same scan vulnerability as LRU. No frequency awareness.

**Verdict for OuterLink:** Good as a fast fallback or for NVMe tiers where the page count is enormous (134M pages per 8TB NVMe). The per-page overhead matters at that scale.

### ARC (Adaptive Replacement Cache)

**How it works:** Maintains four lists:
1. **T1:** Pages accessed recently (once) — recency list
2. **T2:** Pages accessed recently (more than once) — frequency list
3. **B1:** Ghost entries for recently evicted T1 pages
4. **B2:** Ghost entries for recently evicted T2 pages

A tuning parameter `p` dynamically adjusts how much of the cache is devoted to recency vs frequency. When a ghost hit occurs in B1, increase `p` (favor recency). When a ghost hit occurs in B2, decrease `p` (favor frequency).

**For GPU workloads:**
- Self-tuning: adapts to workload phase changes without manual configuration
- Scan-resistant: scanned pages enter T1 but don't pollute T2
- Ghost entries are small (just the page key, no data), so overhead is bounded
- Time complexity is O(1) per operation, same as LRU

**Strengths:**
- Outperforms LRU across a wide range of workloads
- Self-tuning handles the recency-vs-frequency tradeoff automatically
- Ghost entries provide "learning" without storing actual data
- Scan-resistant and loop-resistant

**Weaknesses:**
- 2x the bookkeeping of LRU (four lists instead of one)
- Patent concerns (IBM patent, though widely implemented)
- Slightly more complex implementation
- Ghost list size needs bounding (typically same as cache size)

**Verdict for OuterLink:** Strong candidate for VRAM and DRAM tiers. The self-tuning property is critical for GPU workloads that alternate between training phases, data loading, and inference. Ghost entries provide exactly the "what did we evict that we shouldn't have" signal needed for adaptive tiering.

### CAR (CLOCK with Adaptive Replacement)

**How it works:** Combines CLOCK's efficient scanning with ARC's adaptive recency/frequency balancing. Uses two circular buffers (T1, T2) with reference bits instead of LRU lists, plus ghost buffers (B1, B2) for adaptation.

**For GPU workloads:**
- Same adaptive behavior as ARC
- Lower constant-factor overhead than ARC (no list pointer maintenance)
- Uses CLOCK's bit-scanning instead of LRU's list manipulation

**Strengths:** ARC's adaptiveness with CLOCK's simplicity. Self-tuning, scan-resistant, O(1).
**Weaknesses:** Slightly less precise than ARC (CLOCK is an approximation of LRU). Same 2x bookkeeping.

**Verdict for OuterLink:** Excellent alternative to ARC for high page count tiers. The reduced per-operation overhead matters when managing millions of pages.

### 2Q (Two-Queue)

**How it works:** Two queues — A1 (admission queue, FIFO) and Am (main queue, LRU). New pages enter A1. Pages accessed again while in A1 promote to Am. Eviction prefers A1 (eliminates one-time accesses).

**Strengths:** Simple, good scan resistance. New data doesn't immediately displace hot data.
**Weaknesses:** Fixed split between A1 and Am (not adaptive). Requires tuning the A1 size.

**Verdict for OuterLink:** Simpler than ARC but less adaptive. Acceptable for Phase 1 but ARC/CAR would be better long-term.

### LeCaR (Learning Cache Replacement)

**How it works:** Uses online machine learning (regret minimization) to dynamically weight LRU and LFU policies. At each eviction, consults both policies and selects the eviction candidate using learned weights. Updates weights based on whether evicted pages are re-accessed (regret signal from ghost entries, similar to ARC).

**Performance:** Outperforms ARC by over 18x in some workloads, particularly when cache size is small relative to working set — exactly the situation in GPU VRAM tiering.

**Strengths:** Best-of-breed adaptive performance. Learns workload-specific patterns.
**Weaknesses:** Higher computational cost per eviction decision. Requires tuning the learning rate. More complex to implement correctly.

**Verdict for OuterLink:** Promising for future optimization (R11 prefetching integration). Too complex for initial R10 implementation. Could replace ARC in a later phase when access pattern data is available for training.

---

## 3. GPU Workload Access Pattern Analysis

Understanding GPU workload access patterns is essential for choosing the right policy.

### Common GPU Memory Access Patterns

| Pattern | Example | Best Policy |
|---------|---------|-------------|
| Streaming (sequential, one-pass) | Data loading, preprocessing | LRU works fine |
| Working set (repeated access to subset) | Model weights in inference | LFU or ARC |
| Phased (different sets in different phases) | Training epochs with different data | ARC (adapts to phase changes) |
| Scan + hot set (bulk scan with concurrent hot data) | Data load while inference runs | ARC/CAR (scan-resistant) |
| Random (uniform random access) | Hash tables, graph algorithms | No policy helps — must fit in tier |
| Strided (regular but non-sequential) | Convolution, matrix operations | Spatial prefetching helps more than eviction policy |

### Key Insight: GPU Workloads Are Phase-Driven

Unlike CPU workloads that often have steady-state access patterns, GPU workloads are driven by kernel launches with distinct phases:

1. **Data loading phase:** Stream data from host/storage to VRAM (streaming pattern)
2. **Compute phase:** Repeated access to weights + activations (working set pattern)
3. **Gradient phase:** Write gradients, read weights (mixed pattern)
4. **Communication phase:** All-reduce, parameter sync (network-bound, memory idle)

A good eviction policy must handle transitions between these phases. ARC's adaptive parameter `p` naturally tracks these phase transitions through ghost hits.

---

## 4. Multi-Tier Eviction Strategy

### The Cascade Problem

In a 6-tier system, eviction is not binary (cached vs not-cached). When VRAM is full:
- Evict from VRAM to... where? Remote VRAM? Local DRAM? It depends on the page's characteristics.
- A cold-but-large page should go to NVMe (cheap storage).
- A warm page that might be needed soon should go to DRAM (fast retrieval).
- A page shared with another node should go to that node's VRAM (avoid future network transfer).

### Proposed: Tiered Eviction with Destination Scoring

When evicting a page from tier N, score potential destination tiers:

```
score(page, dest_tier) =
    w1 * capacity_available(dest_tier)
  + w2 * retrieval_speed(dest_tier)    // if page is re-accessed, how fast can we get it back?
  + w3 * access_probability(page)       // how likely is re-access? (from eviction policy)
  + w4 * affinity(page, dest_tier)      // is the page's data already partially in this tier?
  - w5 * migration_cost(current_tier, dest_tier)  // cost to move it there
```

**Hot evictees** (high re-access probability) go to fast tiers (remote VRAM, local DRAM).
**Cold evictees** (low re-access probability) go to cheap tiers (NVMe).
**Shared evictees** go to the node that also needs them (network affinity).

### Eviction Policy Per Tier

Not all tiers need the same eviction policy:

| Tier | Page Count | Recommended Policy | Rationale |
|------|-----------|-------------------|-----------|
| Tier 0: Local VRAM (12 GB) | ~196K | ARC | Highest value, most dynamic. Adaptive policy critical. |
| Tier 1: Remote VRAM (12 GB) | ~196K | ARC | Same as local VRAM but for remote-accessed pages. |
| Tier 2: Local DRAM (256 GB) | ~4.2M | CAR | Large page count, CAR's CLOCK base is more efficient. |
| Tier 3: Remote DRAM (256 GB) | ~4.2M | CAR | Same rationale as local DRAM. |
| Tier 4: Local NVMe (8 TB) | ~134M | CLOCK | Massive page count. Simple policy with minimal overhead. |
| Tier 5: Remote NVMe (8 TB) | ~134M | CLOCK | Same. At this scale, eviction to "nowhere" (discard). |

### Ghost Entry Memory Overhead

ARC and CAR require ghost entries (evicted page keys for learning). Size estimate:

| Tier | Pages | Ghost Entries | Ghost Memory (16 bytes/entry) |
|------|-------|---------------|-------------------------------|
| VRAM (ARC) | 196K | 196K | 3 MB |
| DRAM (CAR) | 4.2M | 4.2M | 67 MB |
| NVMe (CLOCK) | 134M | 0 (no ghosts) | 0 |

Total ghost overhead: ~70 MB per node. Negligible relative to 256 GB DRAM.

---

## 5. Access Monitoring for Eviction Decisions

### How to Track Access Patterns

The eviction policy needs access information. How does OuterLink get it?

**Option A: Intercept-time tracking.** Every cudaMemcpy, cuLaunchKernel, and cuMemPrefetch call tells us which pages are being accessed. Update access metadata in the PTE at interception time.

- Pros: Exact, no overhead during kernel execution
- Cons: Only sees explicit API calls, not implicit GPU memory accesses during kernel execution

**Option B: Periodic PTE scanning.** Similar to Linux NUMA balancing — periodically revoke access permissions and track which pages fault back.

- Pros: Catches all access patterns including in-kernel accesses
- Cons: Introduces latency spikes from artificial faults. Complex interaction with CUDA.

**Option C: Access counters.** Use hardware performance counters (if available) to sample memory access addresses.

- Pros: Low overhead, catches real access patterns
- Cons: Hardware-dependent, sampling may miss patterns, not available on all GPUs

**Option D: Hybrid — Intercept tracking + kernel-launch-based inference.** Use interception to track explicit API accesses. For kernel execution, infer accessed pages from kernel arguments (pointer parameters to cuLaunchKernel).

- Pros: No artificial faults, no hardware dependency. Kernel arguments often contain all the buffer pointers.
- Cons: Doesn't catch indirect access (pointer chasing in kernels). May over-count (marks all kernel argument pages as accessed even if only some are).

**Recommended:** Option D for Phase 1. Intercept tracking gives us explicit copies and prefetches. Kernel argument analysis gives us kernel-phase access patterns. This combination covers the vast majority of GPU memory access patterns without introducing overhead during kernel execution.

Add Option B (periodic PTE scanning) as an optional refinement in a later phase, gated behind a configuration flag for workloads where kernel argument analysis is insufficient (e.g., graph algorithms with indirect memory access).

---

## 6. Anti-Thrashing Integration

The eviction policy must integrate with anti-thrashing mechanisms from the page management design:

1. **Minimum residency time:** After a page is placed in a tier, the eviction scanner skips it until the minimum time has elapsed. This is checked via `last_migration_timestamp` in the PTE.

2. **Migration budget:** The tier manager maintains a per-tier migration budget (pages/second). When the budget is exhausted, the eviction policy returns "no eviction" even if the tier is full. Incoming allocations queue or go to a fallback tier.

3. **Hysteresis in ARC's `p` parameter:** Dampen changes to `p` to prevent rapid oscillation. Instead of immediately adjusting on ghost hits, use exponential moving average.

4. **Eviction backoff:** If a page is evicted and re-accessed within a short window (detected via ghost hit), increase the minimum residency time for that page class.

---

## Summary: Recommended Eviction Architecture

```
                    +-----------------+
                    |  Tier Manager   |
                    |  (per-node)     |
                    +-----------------+
                           |
            +--------------+--------------+
            |              |              |
    +-------+-------+ +---+---+ +--------+--------+
    | VRAM Eviction | | DRAM  | | NVMe Eviction   |
    | (ARC)         | | (CAR) | | (CLOCK)         |
    +-------+-------+ +---+---+ +--------+--------+
            |              |              |
    +-------+-------+ +---+---+ +--------+--------+
    | Ghost Lists   | | Ghost | | (no ghosts)     |
    | B1, B2        | | B1,B2 | |                 |
    +---------------+ +-------+ +-----------------+
            |              |              |
            +--------------+--------------+
                           |
                    +------+------+
                    | Destination |
                    | Scorer      |
                    +------+------+
                           |
              "evict page X to tier Y"
```

### Key Design Points

1. **ARC for VRAM/remote VRAM** — most critical tier, needs adaptive policy
2. **CAR for DRAM/remote DRAM** — large page count, CLOCK efficiency + ARC adaptiveness
3. **CLOCK for NVMe** — massive page count, minimal overhead
4. **Destination scoring** — eviction decides both WHAT to evict and WHERE to send it
5. **Access monitoring via interception + kernel argument analysis** — no artificial overhead
6. **Anti-thrashing** — minimum residency, migration budgets, hysteresis

---

## Related Documents

- `01-existing-tiering-systems.md` — How existing systems handle eviction
- `02-page-management-strategies.md` — Page table design that these policies operate on
- `../preplan.md` — R10 pre-plan integrating all research
- R11 (Speculative Prefetching) — Will use access pattern data from eviction policy ghost entries
- R12 (Memory Dedup) — Dedup-aware eviction: don't evict a page that's the canonical copy for dedup

## Open Questions

1. **Should ARC's `p` parameter be shared across all VRAM allocations, or per-CUDA-context?** Different contexts may have very different access patterns. Per-context `p` would adapt better but increases complexity.

2. **How do we handle eviction during active kernel execution?** Ideally we never evict pages that an active kernel is using. Kernel argument tracking helps, but indirect access patterns could still cause faults. Do we pin all kernel-referenced pages?

3. **Can ghost entries from VRAM eviction be shared with the DRAM eviction policy?** When VRAM evicts a page to DRAM and it's later evicted from DRAM, the VRAM ghost list could inform whether to evict it further to NVMe or bring it back to VRAM.

4. **What is the right ghost list size?** ARC theory says ghost list = cache size. But for VRAM (196K pages), that's manageable. For DRAM (4.2M pages), ghost entries use 67MB — still fine but worth monitoring.

5. **IBM's ARC patent — is it an issue?** The patent (US 6,996,676) expired in 2024. Verify expiration before implementation. ZFS's use of ARC (after initial patent dispute with IBM) suggests the patent landscape has cleared.
