# R10 Research: Existing Memory Tiering Systems

**Date Created:** 2026-03-25
**Date Updated:** 2026-03-25
**Status:** DRAFT
**Author:** Research Agent

## Purpose

Survey existing memory tiering systems to understand proven approaches, their trade-offs, and lessons applicable to OuterLink's 6-tier memory hierarchy. This is foundational research — R10 underpins R11 (prefetching), R12 (dedup), R15 (fault tolerance), and R19 (page faults).

---

## 1. Linux Kernel Memory Tiering (DAMON + CXL)

### What It Is

The Linux kernel (v5.15+) has built-in support for tiered memory management across heterogeneous memory types (DRAM, CXL-attached memory, persistent memory). The core mechanism uses NUMA node abstraction — different memory types appear as separate NUMA nodes with different performance characteristics.

### How It Works

**Demotion (fast to slow):** Since kernel 5.15, cold DRAM pages are demoted to slower memory (PMem/CXL) proactively under memory pressure via the reclaim path.

**Promotion (slow to fast):** Hot pages in slow memory are identified via NUMA-balancing scans and migrated to fast DRAM. A two-access heuristic prevents rapid bouncing between tiers.

**DAMON (Data Access Monitor):** The key innovation for CXL tiering. DAMON monitors memory access patterns asynchronously and uses DAMOS (DAMON-based Operation Schemes) to define promotion/demotion actions:
- `DAMOS_MIGRATE_COLD` — demote cold pages from DRAM to CXL
- `DAMOS_MIGRATE_HOT` — promote hot pages from CXL to DRAM
- Self-tuning patches (March 2025) auto-adjust hot/cold thresholds, yielding ~4.4% improvement over manual tuning

**TPP (Transparent Page Placement):** Sets slow-memory pages as inaccessible; user access triggers minor page faults that drive promotion decisions.

### Performance

- DAMON self-tuned tiering: +4.42% over baseline
- NUMAB-2 hot page promotion: -7.34% (worse — highlights that naive promotion hurts)
- Software-based profiling (PTE scanning, hint faults) struggles with accuracy for CXL memory

### Relevance to OuterLink

| Aspect | Linux Tiering | OuterLink R10 |
|--------|--------------|---------------|
| Number of tiers | 2-3 (DRAM/CXL/PMem) | 6 (local VRAM through remote NVMe) |
| Granularity | OS page (4KB/2MB) | Custom (likely 64KB-2MB) |
| Migration trigger | Memory pressure + access monitoring | Access monitoring + capacity thresholds |
| Hardware assist | HMAT for latency info, NeoMem for profiling | None — pure software |
| Scope | Single machine | Cluster-wide |

**Key Lesson:** Asynchronous access monitoring (DAMON-style) outperforms reactive fault-based approaches. Self-tuning thresholds are essential — manual tuning is fragile.

---

## 2. NVIDIA Unified Virtual Memory (UVM)

### What It Is

UVM provides a single virtual address space accessible from both CPU and GPU. Pages migrate on-demand via hardware page faults (Pascal+), enabling GPU memory oversubscription — allocations can exceed physical VRAM.

### How It Works

1. GPU TLB miss triggers fault in GPU Memory Management Unit (GMMU)
2. GMMU writes fault info to a fault buffer
3. PCIe interrupt notifies the UVM kernel driver
4. Driver retrieves batch of faults (up to 256), deduplicates, resolves
5. OS updates page tables (host + GPU), performs TLB shootdown
6. DMA engine migrates pages to GPU memory

**Eviction:** When VRAM is full, UVM evicts coldest pages to system RAM using LRU. Evicted pages remain accessible — accessing them triggers another fault cycle.

**Replayable faults (Volta+):** GPU continues executing other warps while waiting for page migration, hiding latency.

### Performance Characteristics

- Page fault latency: **10-50 us per fault**
- Random access under oversubscription: only hundreds of KB/s (catastrophic)
- Sequential/structured access: reasonable, especially with prefetching hints
- `cudaMemPrefetchAsync()` can pre-migrate pages to avoid faults
- Zero-copy (mapping host memory to GPU address space) sometimes faster than migration for infrequent access

### Relevance to OuterLink

| Aspect | NVIDIA UVM | OuterLink R10 |
|--------|-----------|---------------|
| Scope | Single GPU + host | Multi-node cluster |
| Migration unit | 4KB-2MB pages | Custom (likely 64KB) |
| Fault handling | Hardware GMMU + kernel driver | Software interception layer |
| Eviction policy | LRU | Configurable (ARC recommended) |
| Transparency | Requires `cudaMallocManaged` | Works with standard `cudaMalloc` |

**Key Lessons:**
- Fault-driven migration is expensive (10-50 us per fault). Prefetching is critical.
- Batch processing of faults amortizes overhead significantly.
- Random access patterns under oversubscription are catastrophic — tier placement must account for access patterns, not just recency.
- Page thrashing between tiers is the primary failure mode.

---

## 3. Intel Optane Persistent Memory (Discontinued but Instructive)

### What It Is

Optane PMem sat between DRAM and SSD in the hierarchy: byte-addressable like DRAM, persistent like SSD, with latency ~3-4x DRAM (~305ns random vs ~81ns DRAM).

### Operating Modes

- **Memory Mode:** DRAM acts as L4 cache for Optane (hardware-managed). Simple but inflexible.
- **App Direct Mode:** Both DRAM and Optane visible as separate address spaces. Software decides placement.

### Key Performance Findings

- Random load: 305 ns (vs 81 ns DRAM) — 3.8x penalty
- Sequential load: 169 ns — internal buffering helps
- Asymmetric read/write: writes are significantly slower and more power-hungry
- When working set fits in DRAM cache (Memory Mode), performance is near-DRAM
- When working set exceeds DRAM, latency degrades dramatically
- Workload-aware tiering can move 90% of data to Optane with only ~11% latency increase

### Relevance to OuterLink

| Aspect | Optane Tiering | OuterLink R10 |
|--------|---------------|---------------|
| Tier gap | 3.8x latency (DRAM to Optane) | 40-900x latency gaps (VRAM to NVMe) |
| Migration cost | Nanoseconds (same bus) | Microseconds to milliseconds (network) |
| Access pattern sensitivity | High | Extreme |
| DRAM as cache | Hardware-managed option | Software-managed required |

**Key Lessons:**
- DRAM augmentation is essential — pure slow-tier configurations are unusable.
- Workload-aware placement vastly outperforms naive tiering.
- Simple drop-in replacement fails. The software layer matters more than the hardware.
- Access skew is the key insight: most workloads have highly skewed access patterns, so smart placement can move most data to slow tiers with minimal impact.

---

## 4. GPUDirect Storage (GDS) — NVMe as GPU Memory Tier

### What It Is

GDS creates a direct DMA data path between NVMe storage and GPU memory, bypassing the CPU bounce buffer entirely. Part of CUDA since recent releases.

### How It Works

- NVMe controller DMA engine writes directly to GPU BAR memory
- Requires pinned GPU memory (`cudaMalloc`, not `cudaMallocManaged`)
- cuFile APIs replace POSIX read/write for GPU-resident buffers

### Performance

- 2x-8x higher bandwidth than CPU-bounce path
- 3.8x lower end-to-end latency
- Requires GPU memory to be pinned (not compatible with UVM managed memory)

### Relevance to OuterLink

GDS is directly relevant for Tier 4 (Local NVMe) and Tier 5 (Remote NVMe via NVMe-oF). However:
- GDS requires NVIDIA proprietary drivers and GPUDirect RDMA support
- OuterLink's OpenDMA approach may provide an alternative path via PCIe BAR1
- GDS works for bulk transfers but not fine-grained page migration
- For NVMe-as-swap, a page-level approach with DMA batching is more appropriate

---

## 5. NVIDIA Inference Context Memory Storage Platform (ICMSP)

### What It Is

Announced at CES 2026, ICMSP standardizes offloading inference KV cache to NVMe SSDs. Formalizes a tiered hierarchy: HBM -> CPU DRAM -> NVMe SSD -> networked storage.

### Key Claims

- 5x greater power efficiency than traditional storage approaches
- 5x higher tokens-per-second
- NVMe-resident KV cache as part of the context memory address space
- Persistent across inference runs

### Relevance to OuterLink

ICMSP validates the exact tiered architecture R10 proposes. It confirms industry direction toward HBM/VRAM -> DRAM -> NVMe hierarchies with transparent management. OuterLink extends this beyond a single node to a cluster.

---

## 6. Research Systems: vDNN, SwapAdvisor, RT-Swap, DeepNVMe

### vDNN (Virtualized DNN)
- Swaps feature maps between GPU and CPU memory during training
- Reduced GPU memory usage by 89-95% for large networks
- Foundation for all modern GPU memory management research

### SwapAdvisor
- Uses computation graph analysis to determine optimal swap schedule
- Plans swaps ahead of time (proactive, not reactive)
- Key insight: if you know the execution schedule, you can prefetch perfectly

### RT-Swap (2024)
- Uses CUDA VMM APIs for fine-grained virtual/physical address separation
- Minimizes internal fragmentation via uniform physical chunks
- Real-time multi-DNN inference with memory sharing

### DeepNVMe + ZeRO-Inference
- Combines NVMe RAID-0 volumes with GDS for GPU-direct NVMe access
- ZeRO-Inference: inference of 100B+ parameter models on 1 GPU via NVMe offloading
- With Gen5 NVMe + GDS: 17-26 tokens/sec (vs 7 tokens/sec Gen4)

### Relevance to OuterLink

These systems prove that proactive scheduling (knowing what to swap when) dramatically outperforms reactive faulting. For R10:
- The CUDA interception layer already knows allocation patterns from cudaMalloc calls
- Kernel launch interception provides the computation graph
- This enables SwapAdvisor-style proactive tiering in later phases (R11)

---

## 7. Hardware-Assisted Approaches

### NeoMem (CXL Device-Side Profiling)
- Hardware profiler (NeoProf) on CXL device snoops memory access requests
- Generates page hotness statistics with near-zero CPU overhead
- 32-67% speedup over software-only tiering
- Validated on FPGA-based CXL platform

### Intel Flat Memory Mode
- Hardware-managed DRAM-as-cache for CXL memory
- No OS intervention, single NUMA node
- Works well when working set is close to DRAM size; degrades when it exceeds

### GPUVM (GPU-Driven Virtual Memory)
- Uses RDMA NIC to manage GPU virtual memory without CPU involvement
- 4x performance improvement over UVM for latency-bound workloads
- Host involvement overhead is 7x higher than actual transfer time at 64KB page size

### Relevance to OuterLink

OuterLink cannot rely on hardware-assisted profiling (no CXL devices in target hardware). However:
- The 7x host overhead finding from GPUVM confirms that minimizing CPU involvement in the migration path is critical
- ConnectX-5 RDMA could serve a similar role to GPUVM's NIC-mediated approach for remote tiers
- OpenDMA (Phase 5) could eventually provide hardware-assist for remote tier access

---

## Summary: Comparative Table

| System | Tiers | Migration Trigger | Granularity | Overhead | Strengths | Weaknesses |
|--------|-------|-------------------|-------------|----------|-----------|------------|
| Linux DAMON | 2-3 | Access monitoring | 4KB/2MB | Low | Self-tuning, async | Inaccurate for GPU |
| NVIDIA UVM | 2 (VRAM/DRAM) | Page fault | 4KB-2MB | 10-50 us/fault | Hardware support | Thrashing, no NVMe tier |
| Optane tiering | 2 | Workload-aware | Cacheline-page | ~300ns access | Low latency gap | Discontinued |
| GPUDirect Storage | 2 (NVMe/VRAM) | Explicit API | Bulk transfer | Low (DMA) | High bandwidth | No page-level mgmt |
| vDNN/SwapAdvisor | 2 (VRAM/DRAM) | Proactive schedule | Tensor-level | Pre-planned | Near-optimal | Requires graph analysis |
| NeoMem | 2-3 | HW profiling | Page | Very low | Accurate hotness | Requires CXL hardware |
| GPUVM | 2+ | GPU-driven fault | 64KB | Low (no CPU) | No CPU overhead | Requires custom RNIC |

---

## Recommendations for OuterLink R10

1. **Adopt DAMON-style asynchronous access monitoring** rather than fault-driven migration. Faults are too expensive for the multi-tier case.

2. **Design for proactive tiering from day one.** The CUDA interception layer gives us unique visibility into allocation patterns and kernel launches — use this for prediction (R11 foundation).

3. **Use 64KB as the base page size.** Balances TLB pressure, metadata overhead, and migration granularity. Aligns with GPUVM findings and WDDM GPU page size.

4. **NVMe tiers (4/5) need bulk-oriented migration.** Page-at-a-time NVMe access is wasteful. Batch migrations of multiple pages per DMA operation.

5. **Anti-thrashing mechanisms are non-negotiable.** Every system studied shows thrashing as the primary failure mode. Migration rate limiting, hysteresis bands, and minimum residency times are essential.

6. **CPU involvement in the hot path must be minimized.** GPUVM showed 7x overhead from host involvement. Use DMA engines and RDMA where possible.

---

## Related Documents

- `02-page-management-strategies.md` — Page sizes, tables, metadata design
- `03-eviction-policies.md` — Which eviction policies fit GPU workloads
- `../preplan.md` — R10 pre-plan (scope, decisions, risks)
- `../../R11-speculative-prefetching/` — Builds on R10 access monitoring
- `../../R12-memory-dedup/` — Builds on R10 page table
- `../../R19-network-page-faults/` — Builds on R10 fault handling

## Open Questions

1. **Can we use NVIDIA's UVM fault buffer for access monitoring?** The open-source nvidia-uvm.ko module exposes fault information — could we piggyback on this for hot/cold detection without implementing our own monitoring?

2. **What is the real-world migration bandwidth between our tiers?** The theoretical numbers (28 GB/s NVMe, 76 GB/s DRAM) need validation under concurrent workload conditions on the MS-02 Ultra.

3. **How does ConnectX-5 RDMA compare to GPUVM's NIC-mediated approach?** Could we achieve similar CPU-bypass benefits for remote tier access?

4. **Is there a hybrid approach?** Reactive faults for initial placement + proactive migration for optimization, similar to how Linux DAMON combines TPP fault-based tracking with asynchronous DAMOS migration.
