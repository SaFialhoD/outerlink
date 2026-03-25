# R12 Research: Existing Deduplication Systems

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft

## Purpose

Survey existing memory and block-level deduplication systems to extract design patterns, overhead numbers, and lessons applicable to OuterLink's cross-GPU memory deduplication at 64KB page granularity.

---

## 1. Linux KSM (Kernel Same-page Merging)

### How It Works

KSM is a Linux kernel feature (merged in 2.6.32, December 2009) that scans physical memory for identical 4KB pages across processes or VMs, merges them into a single physical page, and marks all virtual mappings as copy-on-write.

**Scanning Architecture:**
- A kernel daemon (`ksmd`) periodically scans memory regions that applications have registered via `madvise(MADV_MERGEABLE)`
- Each candidate page is hashed and inserted into one of two Red-Black trees:
  - **Stable tree:** Pages already merged and confirmed identical (searched first)
  - **Unstable tree:** Candidate pages not yet merged (rebuilt every full scan)
- Pages are only inserted into the unstable tree if their hash has NOT changed since the previous scan (filters volatile pages)
- On hash match, a full byte-by-byte memcmp confirms identity before merging
- Merged pages are marked read-only; writes trigger a COW page fault

**Key Parameters:**
- `pages_to_scan`: How many pages to scan per cycle (default: 100)
- `sleep_millisecs`: Delay between scan cycles (default: 20ms)
- `max_page_sharing`: Maximum VMs sharing a single page (default: 256)

### Performance Numbers

| Metric | Value | Source/Context |
|--------|-------|----------------|
| Memory savings | 30-50% typical | General workloads |
| VM density gain | 52 WinXP VMs on 16GB | Red Hat experiment |
| CPU overhead | 5-10% typical | Depends on scan rate |
| CPU overhead (peak) | Up to 60% during startup | Instagram/Meta Skylake systems |
| CPU overhead (steady) | ~30% | Instagram/Meta steady state |
| Merge speed | ~560 MB/s | 100MB region, selftest benchmark |

### NUMA Awareness

RHEL 7+ KSM is NUMA-aware: it avoids merging pages across NUMA nodes by default, preventing remote memory access penalties. This is directly relevant to OuterLink where "NUMA nodes" are entire separate machines.

### Interaction with Huge Pages (THP)

KSM breaks 2MB huge pages into 4KB base pages to find duplicates within them. This destroys the TLB benefits of huge pages and can cause significant performance regression. **Lesson for OuterLink:** Our 64KB pages are a middle ground -- large enough to reduce TLB pressure, small enough to find meaningful duplicates.

### Lessons for OuterLink

1. **Two-tree approach is clever:** Stable tree (confirmed merges) vs unstable tree (candidates) avoids wasting time re-verifying known-good merges
2. **Hash-then-verify is essential:** Hash match alone is not sufficient; full comparison is needed. At 64KB, this is 64KB memcmp per candidate -- trivial on CPU, worth considering for GPU VRAM
3. **Scan rate vs CPU tradeoff is real:** KSM's biggest criticism is CPU overhead. OuterLink should use event-driven detection (hash on allocation/load) rather than continuous scanning
4. **NUMA locality matters:** Cross-node merging has hidden costs. OuterLink must account for the remote access latency when deciding whether to dedup cross-node
5. **Volatile page filtering saves work:** Only considering pages whose hash is stable between scans avoids wasting effort on frequently-written data

---

## 2. ZFS Block-Level Deduplication

### How It Works

ZFS performs **inline (synchronous) deduplication** at the block level. Every block written to the pool is hashed, and the hash is checked against a pool-wide Deduplication Table (DDT). If the hash exists, ZFS increments a reference count instead of writing the block. If not, the block is written and a new DDT entry is created.

**DDT Structure:**
- Key: 40 bytes (32 bytes checksum + 8 bytes metadata)
- Each cached entry: ~320 bytes of RAM
- Stored on disk but must be cached in RAM for performance
- Hash algorithm: configurable (SHA-256 default, fletcher4 for performance)

### Memory Overhead

| Scale | DDT Entries | RAM Required |
|-------|-------------|-------------|
| 100K blocks | 100,000 | 32 MB |
| 1M blocks | 1,000,000 | 320 MB |
| 10M blocks | 10,000,000 | 3.2 GB |
| 100M blocks | 100,000,000 | 32 GB |

**Rule of thumb:** 5 GB RAM per TB of deduplicated storage, on top of ARC and system needs.

### Inline vs Post-Process

ZFS only does inline dedup -- it checks at write time. This means:
- **Pro:** Duplicate data is never written to disk (saves I/O bandwidth)
- **Con:** Every write incurs a hash computation and DDT lookup
- **Con:** DDT must fit in RAM or performance collapses (100 IOPS on a 7200RPM disk for uncached DDT reads = ~400 KB/s throughput)

### Fast Dedup (OpenZFS 2.3.0+, 2024)

New improvements address longstanding DDT performance issues:
- **DDT Log:** Batches DDT updates instead of synchronous writes
- **Prefetch:** Pre-loads DDT entries likely to be needed
- **Pruning:** Removes unique entries older than a threshold (assumes if a block had no duplicates for N days, it never will)
- **Quota:** Limits DDT size to prevent unbounded RAM consumption

### Lessons for OuterLink

1. **Memory cost of hash tables is significant:** At 320 bytes per entry and 64KB pages, 1M pages = 320 MB just for the dedup table. OuterLink must budget for this
2. **Inline dedup is ideal for our use case:** Model weights are loaded once and read many times. Checking at load time (not continuously scanning) is natural
3. **DDT pruning is a good idea:** Pages that have been unique for a long time can be removed from the dedup index to save memory
4. **Compression is often better than dedup:** ZFS community consensus is that LZ4 compression gives most of the space savings at a fraction of the overhead. OuterLink should consider compressed tiers (NVMe tier) separately from dedup
5. **Hash table caching is critical:** If the dedup index doesn't fit in fast memory, performance craters. Keep the index in host DRAM, not VRAM

---

## 3. VMware TPS (Transparent Page Sharing)

### How It Works

VMware's hypervisor (ESXi) scans VM memory pages, computes hashes, and merges identical pages with COW protection. Unlike KSM, TPS operates at the hypervisor level with no guest cooperation needed.

**Two-Phase Process:**
1. **Hash Phase:** Compute hash of each VM page, store in a global hash table
2. **Share Phase:** On hash match, perform full byte comparison, then remap to shared physical page with read-only protection. Writes trigger COW via VMkernel

**Scope Changes (Security Response):**
- Pre-2014: Inter-VM TPS enabled by default (pages shared across VMs)
- Post-2014: Only intra-VM TPS by default (pages shared within a single VM)
- Controlled by `Mem.ShareForceSalting` setting (0/1/2)
- "Salting" gives each VM a unique salt mixed into the hash, preventing cross-VM matches unless VMs share the same salt value

### Security Concerns

Academic researchers demonstrated that TPS could be exploited as a side channel:
- **Timing attack:** By observing COW fault latency, an attacker VM could determine whether a specific page exists in another VM's memory
- **Cache-based attack:** Flush+Reload on shared pages could leak AES keys across VMs
- These attacks require controlled, non-standard configurations but were enough for VMware to disable inter-VM sharing by default

### Performance Numbers

| Metric | Value | Context |
|--------|-------|---------|
| Memory savings | 10-40% | VDI environments |
| Impact of disabling | Increased ballooning/swapping | When host memory is overcommitted |

### Lessons for OuterLink

1. **Side-channel attacks are real but contextual:** OuterLink nodes are owned by the same user (not multi-tenant), so the security model is fundamentally different from VMware's. Inter-node dedup is safe in our threat model
2. **Salting for selective sharing is elegant:** If OuterLink ever supports multi-tenant GPU pools, salt-based sharing groups would work well
3. **Intra-node dedup is free win:** Even without cross-node dedup, deduplicating within a single node (across GPU contexts) has zero network cost
4. **Full byte comparison after hash match is standard:** Every major system does hash-then-verify. The hash is a fast pre-filter, not the final word

---

## 4. Windows Memory Combining

### How It Works

Windows 8.1+ includes kernel-level memory combining as a default-on feature:
- A kernel thread (`MiCombineAllPhysicalMemory`) scans all physical memory every ~15 minutes
- Identifies identical private, pageable pages
- Merges them with COW semantics (write triggers page fault and private copy)
- Excludes file-backed pages and huge pages (non-pageable on Windows)

### Key Differences from KSM

| Aspect | Windows | KSM |
|--------|---------|-----|
| Default state | On | Off (requires madvise) |
| Scan interval | ~15 minutes | Configurable (20ms default sleep) |
| Page types | Private pageable only | madvise-marked regions |
| Huge pages | Excluded | Broken into 4KB pages |
| Approach | Batch periodic | Continuous incremental |

### Lessons for OuterLink

1. **Batch periodic scanning works for stable data:** Model weights don't change, so a one-time scan at load time is better than periodic rescanning
2. **Excluding non-pageable memory is sensible:** GPU VRAM is analogous to non-pageable memory. Dedup decisions should be made at the software layer before data reaches VRAM, not by scanning VRAM directly

---

## 5. GPU-Specific Deduplication Research

### Catalyst (VEE 2017)

**GPU-Assisted Rapid Memory Deduplication in Virtualization Environments**

Catalyst offloads the hashing phase of memory deduplication to the GPU:
- **Phase 1 (GPU):** Pages from VMs are transferred to GPU memory, hashed in parallel by GPU cores, and hash results are returned to the host. The GPU identifies likely sharing candidates
- **Phase 2 (CPU):** Targeted byte-by-byte comparison on the candidate set identified by the GPU
- **Result:** Achieves higher dedup ratios in less time than KSM at comparable or lower compute cost, because the GPU can hash thousands of pages simultaneously

### Gemina (HPCA 2025)

**Coordinated and High-Performance Memory Deduplication Engine**

A more recent system building on Catalyst's approach, presented at HPCA 2025. Focuses on coordinated dedup across the memory hierarchy for cloud/virtualization workloads.

### CUDA Unified Memory (Relevant Mechanism)

NVIDIA's UVM provides page-fault-based migration between CPU and GPU:
- Pages migrate on-demand at 2MB granularity (GPU preferred) or 4KB (CPU)
- Fault buffer processes up to 256 faults per batch
- Page fault latency: 10-50 us per fault
- Oversubscription supported via LRU eviction

**Relevance:** UVM's page fault mechanism is how CUDA currently handles memory that needs to be on multiple devices. However, UVM does NOT deduplicate -- each GPU gets its own physical copy. OuterLink's dedup would sit above UVM, preventing redundant copies from being created in the first place.

### Lessons for OuterLink

1. **GPU-accelerated hashing is proven:** Catalyst demonstrated that offloading hash computation to the GPU is faster than CPU-only scanning. OuterLink could hash pages on the GPU where they already reside
2. **Two-phase (hash then verify) works at GPU scale:** Even with GPU acceleration, the full comparison is still done on CPU. For VRAM-resident pages, OuterLink could do both phases on GPU
3. **UVM provides the mechanism but not the policy:** CUDA's page migration infrastructure could potentially be leveraged, but the dedup logic is entirely new

---

## 6. Comparative Analysis for OuterLink

### What Works at 64KB Granularity

| System | Granularity | Overhead | Savings | Applicability to 64KB |
|--------|-------------|----------|---------|----------------------|
| KSM | 4KB | 5-10% CPU | 30-50% | 64KB gives 16x fewer pages to track, lower overhead |
| ZFS | Variable (default 128KB) | 320 bytes/block RAM | Depends on data | 64KB is close to ZFS default, directly applicable |
| VMware TPS | 4KB | Minimal (hypervisor) | 10-40% | Same benefits as KSM with larger pages |
| Windows | 4KB | Low (15min batch) | Varies | Batch approach maps well to load-time dedup |
| Catalyst | 4KB | GPU compute | Better than KSM | GPU hashing at 64KB is even more efficient (better bandwidth utilization) |

### Overhead vs Savings Tradeoff Summary

**For LLM model weight dedup (primary target):**
- Weights are loaded once, read millions of times, never written during inference
- Dedup is checked once at load time (inline, like ZFS)
- No continuous scanning needed (unlike KSM)
- Expected savings: ~(N-1)/N where N = number of GPUs with identical weights
- Overhead: One-time hash computation + hash table storage

**Overhead budget (1M 64KB pages = 64GB pool):**
- Hash computation: ~2ms at 31 GB/s (xxHash3) for 64GB
- Hash table: 1M entries x ~48 bytes (128-bit hash + metadata) = 48 MB
- COW tracking: Minimal for read-only data (model weights)
- Total overhead: <0.1% of managed memory

---

## Open Questions

1. Should OuterLink support dedup across heterogeneous page sizes (e.g., dedup a 64KB page against a 2MB huge page that contains it)?
2. How does dedup interact with R10's tier migration? If a deduped page is migrated to a different tier, do all references follow?
3. What is the minimum dedup ratio that makes the overhead worthwhile? (Likely >2x, i.e., at least 2 copies)

## Related Documents

- R10 Memory Tiering (page table format, PTE structure)
- R12 Research 02 — Hashing and Detection
- R12 Research 03 — Copy-on-Write Network
- R12 Preplan
