# R10 Pre-Plan: Memory Tiering & NVMe as Tier 3

**Date Created:** 2026-03-25
**Date Updated:** 2026-03-25
**Status:** DRAFT
**Author:** Research Agent

## Purpose

Define the scope, dependencies, decisions, risks, and open questions for R10 — OuterLink's foundational memory tiering system. This pre-plan synthesizes findings from three research documents and lays out what must be decided before detailed planning begins.

R10 is the **single most critical foundation** in OuterLink's advanced feature set. R11 (prefetching), R12 (dedup), R15 (fault tolerance), and R19 (network page faults) all build directly on it.

---

## 1. Scope Definition

### What R10 IS

R10 implements a 6-tier memory hierarchy with automatic page management:

```
Tier 0: Local VRAM     — 12 GB,   ~900 GB/s   (GPU internal bandwidth)
Tier 1: Remote VRAM    — 12 GB,   ~12-22 GB/s  (ConnectX RDMA / USB4)
Tier 2: Local DRAM     — 256 GB,  ~76 GB/s     (DDR5 dual-channel)
Tier 3: Remote DRAM    — 256 GB,  ~10-22 GB/s  (RDMA/USB4)
Tier 4: Local NVMe     — 8 TB,    ~28 GB/s     (4x NVMe RAID-0)
Tier 5: Remote NVMe    — 8 TB,    ~10 GB/s     (RDMA to remote NVMe)
```

**Core components:**

1. **Page Table** — Cluster-wide virtual-to-physical mapping with per-page metadata (tier location, access info, state flags). Hash-table-based, 64-byte entries, 64KB page granularity.

2. **Tier Manager** — Per-node daemon that decides page placement and triggers migrations. Maintains eviction policies (ARC for VRAM, CAR for DRAM, CLOCK for NVMe) and anti-thrashing mechanisms.

3. **Migration Engine** — Executes page transfers between tiers using DMA (local) and RDMA/network (remote). Handles pinning, transfer, PTE update, and TLB invalidation atomically.

4. **CUDA Interception Extensions** — Extends the existing LD_PRELOAD interception layer to route cudaMalloc/cudaFree through the tier manager and track memory access patterns from cudaMemcpy and cuLaunchKernel calls.

5. **Access Monitor** — Tracks which pages are hot/cold/warm via interception-time metadata updates and kernel argument analysis. Feeds data to eviction policies.

6. **NVMe Tier Driver** — Manages NVMe RAID-0 volumes as memory-tier backing stores with DMA-optimized read/write paths. Batches small page operations into larger NVMe I/O requests.

### What R10 is NOT

- **Not a prefetching system** — R10 reacts to capacity pressure and access patterns. Proactive prefetching is R11.
- **Not a deduplication system** — R10 tracks pages individually. Cross-page content deduplication is R12.
- **Not a fault tolerance system** — R10 does not replicate pages across nodes. Redundancy is R15.
- **Not a page fault handler** — R10 migrates pages proactively based on policy. Network page faults (remote access triggering automatic migration) are R19.
- **Not OpenDMA** — R10 uses existing transport (TCP/RDMA). OpenDMA (Phase 5) provides hardware-accelerated paths that R10 can later leverage.

### Deliverables

| Deliverable | Description |
|-------------|-------------|
| `PageTable` | Hash-table data structure with 64-byte PTEs, supporting insert/lookup/update/delete at O(1) |
| `TierManager` | Per-node service managing page placement decisions for all 6 tiers |
| `MigrationEngine` | Async page transfer system with batching, pinning, and atomic PTE updates |
| `EvictionPolicy` trait | Trait with ARC, CAR, and CLOCK implementations |
| `AccessMonitor` | Intercept-based access tracking integrated with CUDA interception layer |
| `NvmeTierDriver` | NVMe RAID-0 backing store with batched I/O and optional GDS integration |
| CUDA interception extensions | Modified cudaMalloc/cudaFree/cudaMemcpy handlers routing through tier manager |
| Benchmarks | Migration latency, eviction accuracy, throughput under pressure for each tier pair |
| Integration tests | End-to-end: CUDA app allocates > VRAM, data overflows to DRAM/NVMe transparently |

---

## 2. Dependency Mapping

### What R10 Needs (Upstream Dependencies)

| Dependency | Status | Notes |
|------------|--------|-------|
| P5: PoC working end-to-end | Required | Basic CUDA interception, transport, and GPU backend must work |
| CUDA interception layer (LD_PRELOAD) | Implemented | Already intercepts 222+ driver API functions |
| Transport layer (TCP + io_uring) | Implemented | Phase 1 transport for remote tier access |
| ConnectX-5 RDMA (optional) | Phase 2 | Enables Tier 1/3 at full bandwidth; TCP fallback works |
| NVMe RAID-0 setup | Infrastructure | 4x NVMe drives configured as RAID-0 on each node |
| CUDA VMM APIs availability | To verify | Need cuMemCreate/cuMemMap on GeForce GPUs |

### What Needs R10 (Downstream Dependents)

| Dependent | What It Needs From R10 |
|-----------|----------------------|
| R11: Speculative Prefetching | Page table, access monitor, migration engine, tier capacity info |
| R12: Memory Deduplication | Page table (dedup hash field), page content access, migration engine |
| R15: Fault Tolerance | Page table (for replication tracking), migration engine (for recovery) |
| R19: Network Page Faults | Page table (for remote page lookup), migration engine (for fault servicing) |
| R13: Compression | Page content access, migration engine (compress during migration) |
| R14: Bandwidth Aggregation | Tier bandwidth info, migration scheduling |

### Circular Dependencies (None)

R10 has no circular dependencies. It is a pure foundation layer.

---

## 3. Decision Inventory

These decisions MUST be made before detailed planning. Each needs explicit confirmation.

### Decision 1: Base Page Size

**Options:** 4KB, 64KB, 2MB, or hybrid

**Research Recommendation:** 64KB base with optional 2MB super-pages

**Rationale:**
- 4KB: Too fine — 268M entries per TB, 16 GB metadata per TB
- 64KB: Balanced — 16.7M entries per TB, 1 GB metadata per TB, validated by GPUVM research, WDDM standard
- 2MB: Too coarse — internal fragmentation, 80 us migration latency per page
- Hybrid: 64KB default, coalesce to 2MB when contiguous pages share access patterns

**Impact:** Affects every other component. PTE design, migration engine, NVMe I/O alignment, dedup granularity (R12), fault handling (R19).

**Status:** NEEDS CONFIRMATION

### Decision 2: Page Table Structure

**Options:** Multi-level radix table, flat hash table, hybrid

**Research Recommendation:** Flat hash table (Robin Hood hashing)

**Rationale:**
- O(1) single-page lookup vs O(3-4) for radix
- Easy to shard across cluster nodes
- GPU research (FS-HPT, PACT 2024) validates hash tables for GPU page management
- Radix index can be added later for range queries (R12 dedup scanning)

**Impact:** Core data structure for all dependent features.

**Status:** NEEDS CONFIRMATION

### Decision 3: Eviction Policy Architecture

**Options:** Single policy everywhere, per-tier policies, ML-based

**Research Recommendation:** Per-tier policies — ARC (VRAM), CAR (DRAM), CLOCK (NVMe)

**Rationale:**
- Different tiers have vastly different page counts (196K vs 134M)
- ARC's ghost lists provide learning signal, critical for the high-value VRAM tier
- CLOCK's minimal overhead is necessary for NVMe's 134M pages
- Per-tier policies connected by a destination scorer for eviction routing

**Impact:** Determines eviction quality, thrashing behavior, and memory overhead.

**Status:** NEEDS CONFIRMATION

### Decision 4: Access Monitoring Approach

**Options:** Intercept-only, periodic PTE scanning, hardware counters, hybrid

**Research Recommendation:** Intercept tracking + kernel argument analysis (hybrid Option D from research)

**Rationale:**
- No artificial fault overhead during kernel execution
- Kernel arguments contain buffer pointers — gives us per-kernel access patterns
- Intercept tracking covers cudaMemcpy, cudaMemPrefetch, and other explicit operations
- Hardware counters are GPU-model-dependent and sampling-based (lossy)

**Impact:** Determines quality of hot/cold classification, foundation for R11 prefetching.

**Status:** NEEDS CONFIRMATION

### Decision 5: PTE Ownership Model (Multi-Node)

**Options:** Allocation-based, hash-based, current-tier-based

**Research Recommendation:** Allocation-based for Phase 1

**Rationale:**
- Matches existing OuterLink client/server model (client allocates, server manages)
- Simple — no distributed consensus needed
- Migrate to hash-based when scaling beyond 4 nodes

**Impact:** Determines how page table lookups work across nodes, coherence protocol complexity.

**Status:** NEEDS CONFIRMATION

### Decision 6: NVMe Access Method

**Options:** Standard file I/O, io_uring, GPUDirect Storage, SPDK

**Research Recommendation:** io_uring for Phase 1 with GDS evaluation

**Rationale:**
- io_uring is already used in OuterLink's transport layer
- GDS requires pinned GPU memory (cudaMalloc, not managed) — compatible with our interception approach
- GDS provides 2-8x bandwidth improvement but has driver dependencies
- SPDK bypasses the kernel entirely but requires dedicated NVMe devices

**Impact:** Determines NVMe tier throughput and latency. Affects whether GPU-to-NVMe can be direct (GDS) or requires CPU bounce (io_uring).

**Status:** NEEDS CONFIRMATION

### Decision 7: Virtual Address Space Strategy

**Options:** Per-node address spaces, cluster-wide unified space, hybrid

**Research Recommendation:** Cluster-wide unified space with per-node non-overlapping ranges

**Rationale:**
- Foundation for R19 (network page faults) — a virtual address must be globally meaningful
- CUDA VMM APIs allow reserving address ranges without backing memory
- Each node reserves a range; page table maps to physical location on any node

**Impact:** Fundamental architectural decision. Affects how R19, R15, and all cross-node features work.

**Status:** NEEDS CONFIRMATION

---

## 4. Risk Assessment

### Risk 1: CUDA VMM API Availability on GeForce

**Severity:** HIGH
**Probability:** MEDIUM

CUDA's low-level virtual memory management APIs (cuMemCreate, cuMemMap) may be restricted on consumer GPUs. If unavailable, we cannot dynamically map/unmap physical pages without cudaFree's synchronization penalty.

**Mitigation:** Test VMM APIs on RTX 3060/4070 immediately. If restricted, fall back to a shadow address space approach where OuterLink maintains its own virtual-to-physical mapping separate from CUDA's.

### Risk 2: Page Migration Overhead Exceeds Budget

**Severity:** HIGH
**Probability:** MEDIUM

Research shows 10-50 us per page fault in UVM. OuterLink adds network latency for remote tiers. If migration overhead exceeds useful compute time, the tiering system becomes a net negative.

**Mitigation:**
- Proactive migration (between kernel launches, not during)
- Batch migrations (move multiple pages in one DMA operation)
- Migration budgets that cap overhead as a percentage of available bandwidth
- R11 prefetching reduces reactive migrations

### Risk 3: Thrashing Between Tiers

**Severity:** HIGH
**Probability:** HIGH (if not addressed)

Every surveyed system identifies thrashing as the primary failure mode. Pages bouncing between VRAM and DRAM can saturate the PCIe bus and tank performance.

**Mitigation:**
- ARC's adaptive policy reduces thrashing vs naive LRU
- Minimum residency times prevent rapid bouncing
- Migration rate limiting caps bandwidth consumption
- Hysteresis bands for promotion/demotion thresholds
- Kernel-level page pinning during execution

### Risk 4: Metadata Memory Consumption at Scale

**Severity:** MEDIUM
**Probability:** LOW

At 64 bytes per PTE and 64KB pages, metadata is 0.1% of managed capacity. For a 16 TB cluster, that is 16 GB — fits in one node's DRAM. Ghost entries add ~70 MB per node.

**Mitigation:** Already within budget. Monitor actual usage during benchmarking. If needed, reduce PTE size by making some fields optional (dedup hash, prefetch hints).

### Risk 5: NVMe Endurance Under Tier Migration

**Severity:** MEDIUM
**Probability:** MEDIUM

Using NVMe as a memory tier means frequent writes. Consumer NVMe drives have limited write endurance (typically 600 TBW for 2TB drives). High migration rates could burn through endurance in months.

**Mitigation:**
- Prefer NVMe for cold pages (infrequent writes)
- Track write volume per drive, alert on endurance consumption rate
- Use enterprise/data-center NVMe drives for production
- Migration rate limiting naturally limits NVMe write volume

### Risk 6: Concurrent Access During Migration

**Severity:** MEDIUM
**Probability:** HIGH

When a page is being migrated (in-flight DMA), what happens if a CUDA kernel accesses it? The data is in an inconsistent state across source and destination.

**Mitigation:**
- "Migrating" flag in PTE prevents concurrent access
- Stall or redirect accesses to the source copy until migration completes
- Only migrate pages not referenced by active kernels (kernel argument tracking)
- Copy-on-write semantics: keep source valid until destination is confirmed

---

## 5. Implementation Phases (Proposed)

### Phase R10-A: Core Infrastructure (Estimated: 2-3 weeks)

1. Page Table implementation (hash table + PTE structure)
2. Basic Tier Manager with 2-tier support (VRAM + DRAM only)
3. Migration Engine for local VRAM <-> DRAM transfers
4. CUDA interception extensions for cudaMalloc/cudaFree routing
5. Basic access tracking (intercept-time only)
6. LRU eviction (simplest policy, replaced by ARC in Phase B)

**Acceptance:** CUDA app allocates 2x VRAM, computation completes correctly with automatic overflow to DRAM.

### Phase R10-B: Eviction & Monitoring (Estimated: 2-3 weeks)

1. ARC eviction policy for VRAM tier
2. CAR eviction policy for DRAM tier
3. Kernel argument analysis for access monitoring
4. Anti-thrashing mechanisms (minimum residency, rate limiting, hysteresis)
5. Destination scoring for eviction routing
6. Benchmarking framework for eviction accuracy

**Acceptance:** Workload with phase changes (training then inference) shows ARC adapting. No thrashing under controlled tests.

### Phase R10-C: NVMe Tier (Estimated: 2-3 weeks)

1. NVMe tier driver with io_uring backend
2. CLOCK eviction policy for NVMe tier
3. Batched I/O for NVMe operations
4. NVMe RAID-0 configuration and management
5. GDS evaluation (if available on target hardware)
6. Write endurance monitoring

**Acceptance:** CUDA app with 300 GB working set runs on a 12 GB GPU + 256 GB DRAM + NVMe overflow. Correct results.

### Phase R10-D: Remote Tiers (Estimated: 2-3 weeks)

1. Remote VRAM tier (Tier 1) via existing transport
2. Remote DRAM tier (Tier 3) via existing transport
3. Remote NVMe tier (Tier 5) via network + NVMe
4. Distributed PTE management (allocation-based ownership)
5. Cross-node migration coordination
6. PTE cache for remote page lookups

**Acceptance:** 2-node cluster with combined memory pool. Workload larger than any single node's resources completes correctly.

### Phase R10-E: Optimization & Hardening (Estimated: 1-2 weeks)

1. 2MB super-page support for contiguous hot regions
2. Migration batching optimization
3. Concurrent migration support (multiple pages in-flight)
4. Stress testing under adversarial workloads (random access, rapid phase changes)
5. Performance regression test suite

**Acceptance:** Performance within 2x of native (no tiering) for workloads that fit in VRAM. Graceful degradation for larger workloads.

---

## 6. Open Questions

### Architecture Questions

1. **How do we handle cudaMallocManaged (UVM) allocations?** UVM has its own page migration. Options: (a) intercept UVM faults and route through our tier manager, (b) pass UVM allocations through unmodified, (c) convert UVM allocations to our managed allocations transparently.

2. **Should the tier manager be a separate process or embedded in the server?** Separate process allows independent scaling and failure isolation. Embedded is simpler and lower latency.

3. **What happens when all tiers are full?** Options: (a) fail the allocation with cudaErrorMemoryAllocation, (b) evict to remote NVMe (always space available if cluster is large enough), (c) compress pages (R13) to make room.

### Performance Questions

4. **What is the real PCIe 5.0 x16 bandwidth on the MS-02 Ultra under concurrent GPU + NVMe + ConnectX traffic?** The theoretical 64 GB/s may be reduced by contention.

5. **Can we overlap migration DMA with GPU kernel execution on the same GPU?** NVIDIA's copy engines are separate from compute engines, but PCIe bandwidth is shared.

6. **What is the practical NVMe RAID-0 bandwidth with 4x Gen4 drives?** Theoretical is 28 GB/s but real-world with small random I/O (page-sized) will be lower.

### Design Questions

7. **Should pages have a "preferred tier" based on their allocation context?** E.g., constant data (model weights) prefer VRAM/DRAM, temporary buffers (activations) prefer being evictable.

8. **How does the tier manager interact with CUDA streams?** Migrations should be asynchronous. Should they use a dedicated CUDA stream?

9. **Can we use Linux's cgroups or memory.tier sysfs to leverage kernel tiering for the DRAM/NVMe tiers?** This would offload some management to the kernel for local tiers.

---

## 7. Success Criteria

| Criterion | Target |
|-----------|--------|
| Correctness | All CUDA operations produce identical results with and without tiering |
| VRAM-fits performance | < 5% overhead when working set fits in VRAM (tiering layer cost) |
| 2x VRAM performance | < 3x slowdown for working set = 2x VRAM (24 GB on 12 GB GPU) |
| 10x VRAM performance | Completes successfully (any speed) for working set = 10x VRAM |
| Migration latency (local) | < 20 us per 64KB page (VRAM <-> DRAM) |
| Migration latency (remote) | < 50 us per 64KB page (VRAM <-> remote DRAM via RDMA) |
| Thrashing detection | System detects and mitigates thrashing within 100ms |
| Page table lookup | < 1 us per lookup (hash table hit) |
| Metadata overhead | < 0.5% of managed capacity |

---

## Related Documents

- `research/01-existing-tiering-systems.md` — Survey of tiering approaches
- `research/02-page-management-strategies.md` — Page size, table, PTE design
- `research/03-eviction-policies.md` — Eviction policy analysis
- `README.md` — R10 overview and folder structure
- `../../R11-speculative-prefetching/` — Builds on R10 access monitoring
- `../../R12-memory-dedup/` — Builds on R10 page table
- `../../R15-fault-tolerance/` — Builds on R10 migration engine
- `../../R19-network-page-faults/` — Builds on R10 page table + migration
- `../../../../docs/architecture/00-project-vision.md` — Project vision
- `../../../../planning/pre-planning/02-FINAL-PREPLAN.md` — Overall project pre-plan
