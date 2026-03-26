# R12: Memory Deduplication -- Pre-Plan

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft
**Phase:** 8 -- Smart Memory
**Priority:** HIGH

## Purpose

Define the scope, dependencies, decisions, risks, and implementation phases for R12 Memory Deduplication before creating the detailed implementation plan.

---

## 1. Scope Definition

### What R12 IS

- **Content-based deduplication** of 64KB memory pages across GPUs and nodes
- **Inline dedup** at allocation/load time: when a page is written to VRAM, hash it and check for duplicates
- **Copy-on-write** semantics for deduped pages that are later modified
- **Dedup table (DDT)** tracking unique page hashes, reference counts, and canonical copy locations
- **Integration with R10's page table** to mark pages as shared/deduped with reference counting
- **Primary target:** LLM model weights (read-only, identical across all GPUs during inference)
- **Secondary targets:** Embedding tables, lookup tables, static configuration data

### What R12 IS NOT

- **Not data compression:** R12 finds identical pages, not similar ones. Compression (LZ4, ZSTD) is a separate optimization for the NVMe tier
- **Not tensor-level awareness:** R12 operates at 64KB page granularity, blind to tensor boundaries. Tensor-aware optimizations are future work
- **Not a distributed cache:** R12 deduplicates content, not access patterns. R11 (Prefetching) handles access optimization
- **Not multi-tenant isolation:** R12 assumes a single-owner cluster. Multi-tenant dedup with salting (VMware-style) is future work

### Scope Boundary

R12 covers the dedup detection engine, the DDT, the COW mechanism, and the integration points with R10's page table. It does NOT cover the transport layer changes needed for R29 (RDMA Multicast) or the fault tolerance implications (R15).

---

## 2. Dependencies

### Upstream (R12 Depends On)

| Dependency | What We Need | Status |
|-----------|-------------|--------|
| **R10 Memory Tiering** | Page table with 64-byte PTEs, tier tracking, page allocation/free | Pre-planned (R10 research complete) |
| **P5 PoC** | Working CUDA interception, basic transport | Not started |
| **CUDA Interception Layer** | cuMemcpy*, cuLaunchKernel hooks for write detection | Part of P1-P5 |

### Downstream (Depends On R12)

| Component | What They Need From R12 | Notes |
|-----------|------------------------|-------|
| **R29 RDMA Multicast** | List of deduped pages to broadcast efficiently | R29 can multicast a single write to all nodes that share a page |
| **R15 Fault Tolerance** | Dedup-aware redundancy (don't replicate deduped pages N times) | R15 must know which pages are shared to avoid over-replication |
| **R11 Prefetching** | Dedup hints (if page X is loaded, pages with same hash are already available) | Prefetch can skip fetching pages that are already deduped locally |

### Parallel (No Hard Dependency)

| Component | Interaction | Notes |
|-----------|------------|-------|
| **R17 Topology-Aware Scheduling** | Place deduped pages on nodes with fast interconnect | Optimization, not required |
| **R19 Network Page Faults** | COW faults may trigger network page faults | R19 and R12 share fault handling infrastructure |

---

## 3. Key Decisions To Make

### Decision 1: Hash Algorithm

| Option | Speed | Output | Collision Risk (1M pages) | Recommendation |
|--------|-------|--------|--------------------------|----------------|
| xxHash128 | ~31 GB/s | 128-bit | 1.5 x 10^-27 | **Plan A** |
| BLAKE3 | ~8.4 GB/s | 256-bit | ~0 | Plan B (if crypto needed) |
| CRC32C pre-filter + xxHash128 | ~32 GB/s + ~31 GB/s | 32+128-bit | Same as xxHash128 | Unnecessary complexity |

**Preliminary decision:** xxHash128. Full verification (memcmp) on every hash match provides defense in depth.

### Decision 2: Dedup Scope (Phase 1)

| Option | Complexity | Memory Savings | Runtime Overhead |
|--------|-----------|---------------|-----------------|
| Read-only pages only | Low | High for LLM inference | Zero (no COW needed) |
| All pages with COW | High | Higher overall | COW fault cost on writes |

**Preliminary decision:** Start with read-only pages only (Phase 1), add COW for writable pages later (Phase 2).

### Decision 3: DDT Architecture

| Option | Pros | Cons |
|--------|------|------|
| Centralized (coordinator node) | Simple, consistent | Single point of failure, network hop for lookups |
| Distributed (each node tracks own) | No SPOF, local lookups | Merge/reconciliation complexity |
| Replicated (all nodes have full DDT) | Fast lookups everywhere | Memory overhead on every node, sync cost |

**Preliminary decision:** Centralized DDT on coordinator, with async replication to one backup node for fault tolerance. DDT lookups happen at load time (not on the hot path), so the network hop is acceptable.

### Decision 4: COW Implementation

| Option | Latency | Complexity | GPU Compatibility |
|--------|---------|-----------|-------------------|
| CUDA interception (proactive COW) | ~5us | Medium | All GPUs |
| UVM page faults (reactive COW) | 10-50us | Low | Pascal+ (2016+) |
| cuMemSetAccess (CUDA VMM API) | ~5us | Medium | CUDA 10.2+ |

**Preliminary decision:** CUDA interception (proactive COW) as primary, with UVM page faults as safety net for undetectable writes. cuMemSetAccess for VRAM read-only enforcement.

### Decision 5: Intra-Node vs Cross-Node Dedup

| Scope | Benefit | Cost |
|-------|---------|------|
| Intra-node only (multi-GPU in one PC) | VRAM savings, no network cost | Limited savings (typically 2-4 GPUs per node) |
| Cross-node (multi-PC cluster) | Maximum savings (4x+ for LLM) | Network COW cost, distributed DDT |
| Both | Full benefit | Full complexity |

**Preliminary decision:** Implement both, but intra-node first (simpler, validates the core logic), then extend to cross-node.

---

## 4. Risks and Mitigations

### Risk 1: Hash Collision Causes Data Corruption

**Severity:** Critical (silent data corruption)
**Probability:** Astronomically low (10^-27 with xxHash128)
**Mitigation:** Full byte-level verification (memcmp) on every hash match. This is standard practice in all production dedup systems (ZFS, KSM, VMware TPS). The hash is a fast pre-filter; memcmp is the source of truth.

### Risk 2: COW Latency Spikes During Inference

**Severity:** High (latency-sensitive workload)
**Probability:** Low if model weights are truly read-only
**Mitigation:**
- Phase 1 only deduplicates pages explicitly marked read-only -- COW never triggers
- Phase 2 adds COW with proactive detection at the interception layer (~5us, comparable to kernel launch overhead)
- Monitor COW frequency; if a page COWs more than N times, remove it from dedup

### Risk 3: DDT Memory Overhead Exceeds Budget

**Severity:** Medium
**Probability:** Low (0.05% of pool at 32 bytes/entry)
**Mitigation:**
- DDT pruning (remove entries with refcount=1 that have been unique for >N minutes, ZFS-style)
- DDT quota: hard limit on DDT size, stop deduplicating new pages when reached
- DDT stored in host DRAM, not VRAM

### Risk 4: Dedup Ratio Too Low to Justify Overhead

**Severity:** Medium
**Probability:** Low for LLM use case (identical weights = near-perfect dedup)
**Mitigation:**
- Minimum dedup threshold: only activate dedup when expected savings > 2x the overhead
- Disable dedup at runtime if measured savings ratio drops below threshold
- Start with the highest-value target (model weights) where dedup ratio is guaranteed to be high

### Risk 5: Interaction with R10 Tier Migration

**Severity:** Medium
**Probability:** Certain (deduped pages will be migrated between tiers)
**Mitigation:**
- When the canonical copy of a deduped page is evicted from VRAM to DRAM (tier migration), all reference holders must be notified or their virtual mappings updated
- Solution: deduped pages get a higher eviction priority bias (less likely to be evicted) because evicting them affects multiple consumers
- If evicted anyway: reference holders fall back to fetching from the new tier (transparent via R10's page fault mechanism)

### Risk 6: GPU-Side Hashing Interferes with Compute

**Severity:** Low-Medium
**Probability:** Manageable
**Mitigation:**
- Hash computation uses low-priority CUDA streams
- Schedule hashing during model loading (before inference starts)
- For inline dedup at load time, hashing is on the critical path anyway (loading the model is I/O-bound, not compute-bound)

---

## 5. Implementation Phases

### Phase 1: Read-Only Dedup, Single Node (2-3 weeks)

**Goal:** Deduplicate identical pages across GPUs within a single machine.

**Deliverables:**
- DDT implementation (hashbrown-based hash map, xxHash128 keys)
- Page hashing on CPU (xxHash128 for pages in host DRAM before GPU upload)
- R10 PTE extension: `dedup_group_id` field, `is_shared` flag, reference count
- Inline dedup check on `cuMemcpyHtoD`: hash source buffer, check DDT, share if match
- cuMemSetAccess enforcement: deduped VRAM pages mapped read-only
- Unit tests: dedup detection, reference counting, page sharing

**Success criteria:**
- Load a 7B model (14GB) on 2 GPUs: uses ~14GB total VRAM instead of ~28GB
- Hash overhead < 1% of model load time
- DDT memory < 1MB for a 14GB model

### Phase 2: Read-Only Dedup, Cross-Node (2-3 weeks)

**Goal:** Extend dedup across network-connected nodes.

**Deliverables:**
- Centralized DDT on coordinator node
- Dedup protocol messages: `DEDUP_CHECK`, `DEDUP_SHARE`, `DEDUP_BREAK`
- Network page reference: PTE points to remote canonical copy with node ID
- Cross-node memcmp verification (hash match -> request 64KB sample -> verify)
- Integration with transport layer (TCP Phase 1, RDMA Phase 2)

**Success criteria:**
- Load a 70B model (140GB) across 4 nodes x 1 GPU: uses ~140GB total VRAM instead of ~560GB
- Dedup check adds < 5% to model load time
- Cross-node reference resolution latency < 15us (RDMA)

### Phase 3: Copy-on-Write (2-3 weeks)

**Goal:** Support COW for deduped pages that are written (fine-tuning, gradient updates).

**Deliverables:**
- COW detection in interception layer (cuMemcpyHtoD, cuMemcpyDtoD write targets)
- COW handler: allocate private page, copy content, update PTE, decrement refcount
- CUDA UVM fault handler (safety net for undetectable writes)
- Async COW break notification to coordinator
- Directory-based invalidation protocol

**Success criteria:**
- COW latency < 15us for local copy, < 25us for remote fetch
- COW correctly handles concurrent writes from multiple nodes
- No data corruption under stress testing (millions of COW operations)

### Phase 4: GPU-Accelerated Hashing (1-2 weeks)

**Goal:** Hash VRAM-resident pages on the GPU instead of transferring to host.

**Deliverables:**
- CUDA kernel for xxHash128 computation on 64KB pages
- Batch hashing API: hash N pages in one kernel launch
- Async hash result transfer (device -> host, 16 bytes per page)
- Integration with DDT lookup pipeline

**Success criteria:**
- GPU hashing throughput > 500 GB/s on RTX 3090 (memory bandwidth limited)
- Hash 1M pages (64GB) in < 200ms including DDT lookup
- Zero interference with inference compute (low-priority stream)

### Phase 5: Optimizations (1-2 weeks)

**Goal:** DDT pruning, adaptive dedup, dedup-aware eviction.

**Deliverables:**
- DDT pruning: remove unique entries after timeout (configurable)
- Adaptive dedup: skip hashing for page ranges with low dedup ratio
- Dedup-aware eviction in R10: prefer evicting non-shared pages
- Dedup statistics and monitoring (dedup ratio, COW frequency, DDT size)
- Intra-node peer access optimization (zero-copy dedup for multi-GPU)

**Success criteria:**
- DDT size stays within configured quota under all workloads
- Dedup ratio reported accurately in monitoring
- Peer access dedup shows zero overhead compared to dedicated copies

---

## 6. Open Questions

1. **Dedup at load time vs continuous scanning?** Research strongly favors load-time (inline) dedup for our use case. Model weights are loaded once. But should we also periodically scan for newly-matching pages (e.g., after partial weight sharing in multi-model serving)?

2. **Tensor-level hints from the application?** If the application tells us "this is a weight tensor, it's read-only", we can skip hashing entirely and just track by allocation identity. Is this worth the API surface?

3. **Dedup across different data types?** FP16 and FP32 representations of the same weight are NOT identical at the byte level. Should R12 handle format-aware dedup, or is that out of scope?

4. **Interaction with quantization?** Quantized models (GPTQ, AWQ) have different byte patterns. Dedup still works (identical quantized weights are identical bytes), but mixed-precision serving (some nodes FP16, some INT4) won't dedup across formats.

5. **DDT persistence across restarts?** Should the DDT be saved to disk so that restarting OuterLink doesn't require re-hashing everything? ZFS does this; KSM does not.

6. **Reference counting overflow?** With 4 bytes for refcount (max 4 billion), this is not a practical concern. But should we use atomic operations for concurrent refcount updates?

7. **How does dedup interact with encryption?** If pages are encrypted at rest or in transit, we must hash the plaintext (before encryption) for dedup to work. This means dedup must happen at a layer above encryption.

---

## 7. Success Criteria

### Memory Savings

| Metric | Target | Method |
|--------|--------|--------|
| Dedup ratio (identical weights, N GPUs) | >= (N-1)/N (e.g., 75% for 4 GPUs) | Measure VRAM used vs naive replication |
| Dedup ratio (mixed workloads) | >= 30% | Measure across diverse LLM serving scenarios |
| Maximum memory overhead from DDT | < 0.1% of managed pool | DDT size / total pool size |

### Performance

| Metric | Target | Method |
|--------|--------|--------|
| Hash overhead at load time | < 1% of model load time | Benchmark model load with/without dedup |
| Dedup check latency (inline) | < 5us per page | Microbenchmark: hash + DDT lookup |
| COW latency (local) | < 15us | Microbenchmark: fault detection + copy |
| COW latency (cross-node RDMA) | < 25us | Microbenchmark: fault + network fetch + copy |
| Inference throughput impact | < 1% regression | Compare tokens/sec with/without dedup |

### Correctness

| Metric | Target | Method |
|--------|--------|--------|
| Data corruption from false hash match | 0 incidents | Stress test with full memcmp verification |
| COW correctness (concurrent writes) | 0 data races | Multi-threaded/multi-node stress test |
| Reference count accuracy | Exact (no leaks, no double-frees) | Long-running test with random alloc/free |

---

## Related Documents

- `research/01-existing-dedup-systems.md` -- Survey of KSM, ZFS, VMware TPS, Windows, Catalyst
- `research/02-hashing-and-detection.md` -- Hash algorithms, GPU hashing, false positive analysis
- `research/03-copy-on-write-network.md` -- Network COW, invalidation protocols, CUDA memory protection
- R10 Memory Tiering (upstream dependency)
- R29 RDMA Multicast (downstream consumer)
- R15 Fault Tolerance (downstream consumer)
- R8 Kernel Parameter Introspection (write detection for COW)
