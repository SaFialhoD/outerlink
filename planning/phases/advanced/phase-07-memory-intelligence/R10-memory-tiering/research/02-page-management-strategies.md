# R10 Research: Page Management Strategies

**Date Created:** 2026-03-25
**Date Updated:** 2026-03-25
**Status:** DRAFT
**Author:** Research Agent

## Purpose

Determine the optimal page size, page table structure, and metadata design for OuterLink's 6-tier memory hierarchy. Page management decisions directly impact every dependent research topic (R11, R12, R15, R19) and are among the hardest to change later.

---

## 1. Page Size Analysis

### Available Options

| Page Size | Pages per GB | Page Table Entries per TB | TLB Coverage (1024 entries) | Migration Cost (PCIe 4.0 ~25 GB/s) |
|-----------|-------------|--------------------------|----------------------------|-------------------------------------|
| 4 KB | 262,144 | ~268 million | 4 MB | ~0.16 us |
| 64 KB | 16,384 | ~16.7 million | 64 MB | ~2.6 us |
| 2 MB | 512 | ~524,288 | 2 GB | ~80 us |

### 4KB Pages

**Pros:**
- Finest granularity — minimal internal fragmentation
- Standard OS page size (x86 Linux default)
- Best for workloads with scattered small allocations

**Cons:**
- Massive metadata overhead: 268 million entries per TB of managed memory
- At 64 bytes per page table entry, that is ~16 GB of metadata per TB — unacceptable
- TLB pressure is extreme: 1024 TLB entries cover only 4 MB
- Migration overhead per page is dominated by DMA setup cost (~5 us), not transfer time
- Every page migration requires a page table update, TLB invalidation, and potentially network round-trip

**Verdict:** Too fine-grained for a multi-tier system spanning terabytes. The metadata and management overhead alone would consume significant DRAM.

### 64KB Pages

**Pros:**
- Good balance: 16.7 million entries per TB is manageable
- At 64 bytes per entry: ~1 GB metadata per TB — acceptable
- GPUVM research validated 64KB as optimal for GPU virtual memory with NIC-mediated access
- Windows WDDM 2.0+ uses 64KB as the GPU large page size
- 16x fewer page table entries than 4KB, 16x less migration overhead
- DMA transfer of 64KB amortizes the ~5 us setup cost well (2.6 us transfer + 5 us setup = ~7.6 us total)
- Power9 systems use 64KB default pages for better UVM performance

**Cons:**
- Some internal fragmentation for small allocations (< 64KB)
- Not aligned with x86 OS page size (4KB), requiring custom mapping
- CUDA minimum allocation granularity is 2MB (cuMemGetAllocationGranularity), so sub-64KB allocations are rare in practice

**Verdict:** Strong candidate for the primary page size. Validated by multiple GPU memory research systems.

### 2MB Pages (Huge Pages)

**Pros:**
- Only 524,288 entries per TB — extremely low metadata
- At 64 bytes per entry: ~32 MB metadata per TB — negligible
- Matches Linux huge page and CUDA allocation granularity (2MB on NVIDIA)
- Excellent TLB coverage: 1024 entries cover 2 GB
- Fewer migrations needed, each migration moves more useful data (if locality is good)

**Cons:**
- Significant internal fragmentation: a 100KB allocation wastes 1.9 MB
- Migration latency is high: ~80 us per page over PCIe 4.0
- Over network (10 GB/s): ~200 us per page — blocks the requesting warp for a long time
- Demand paging with 2MB pages causes huge stalls on first access
- Memory pressure response is coarse — must evict 2MB at a time even if only 64KB is cold

**Verdict:** Good for bulk data (large tensors, buffers) but too coarse as the sole page size. Consider as a secondary "super-page" for contiguous hot regions.

### Recommended: Hybrid Approach

Use **64KB as the base page size** with optional **2MB super-pages** for contiguous hot allocations:

1. All allocations tracked at 64KB granularity by default
2. When the tier manager detects contiguous 64KB pages with similar access patterns, promote them to a 2MB super-page for reduced metadata and TLB pressure
3. Super-pages can be split back to 64KB pages when access patterns diverge
4. This mirrors the Linux transparent huge pages (THP) approach and the Mosaic research for GPU page sizes

---

## 2. Page Table Design

### Requirements for OuterLink

The page table must track:
1. **Location:** Which tier (0-5) and which node holds each page
2. **Physical address:** Where in that tier's address space the page lives
3. **State:** Valid, migrating, evicted, pinned, shared
4. **Access metadata:** Last access time, access count, hotness score
5. **Ownership:** Which CUDA context/allocation owns the page
6. **Cross-references:** For dedup (R12) — hash or reference to canonical copy

### Design Options

#### Option A: Multi-Level Radix Page Table (Traditional)

Structure mirrors traditional OS page tables with levels indexed by virtual address bits.

```
Level 3: Root table (indexed by VA[47:39]) -> 512 entries
Level 2: Directory (indexed by VA[38:30]) -> 512 entries
Level 1: Table (indexed by VA[29:16]) -> 16384 entries (for 64KB pages)
Level 0: Entry (page metadata)
```

**Pros:**
- Well-understood, maps directly to hardware MMU concepts
- Sparse — only allocates entries for used address ranges
- Can mix page sizes at different levels

**Cons:**
- Multi-level lookup on every access check (3-4 memory accesses)
- Poor cache behavior for scattered access patterns
- Complex to distribute across cluster nodes

#### Option B: Flat Hash Table

Single hash table mapping virtual page number to page metadata.

```
hash(VPN) -> bucket -> {VPN, tier, node, phys_addr, state, access_info}
```

**Pros:**
- O(1) lookup (single memory access + potential collision chain)
- Simple to distribute — shard by hash range across nodes
- Recent GPU research (FS-HPT, PACT 2024) shows hashed page tables outperform radix for GPUs
- Easy to iterate for scanning/eviction decisions

**Cons:**
- Not sparse for metadata — fixed memory allocation for hash table
- Hash collisions require chain walking
- No inherent support for super-pages (need separate table or multi-level hash)

#### Option C: Hybrid — Hash Table + Radix Index

Use a flat hash table for the primary lookup path, with a radix tree as a secondary index for range queries (finding all pages in a contiguous allocation).

**Pros:**
- Best of both: O(1) single-page lookup, efficient range operations
- Hash table distributes well across cluster nodes
- Radix index stays local to the allocating node

**Cons:**
- Two data structures to maintain consistency
- More memory for dual indexing

### Recommended: Option B (Flat Hash Table) for Phase 1

For initial implementation:
- Flat hash table with open addressing (Robin Hood hashing for low variance)
- 64 bytes per entry, pre-allocated for expected page count
- Shard key = virtual page number, so each node manages pages it hosts
- Add radix index in Phase 2 when range operations become necessary (R12 dedup scanning)

---

## 3. Page Table Entry (PTE) Design

### Proposed PTE Layout (64 bytes)

```
Bytes 0-7:   Virtual Page Number (VPN)           — 8 bytes
Bytes 8-9:   Tier (0-5) + Node ID (0-255)        — 2 bytes
Bytes 10-15: Physical Page Frame Number           — 6 bytes (48-bit, covers 16 TB at 64KB pages)
Bytes 16-17: Flags                                — 2 bytes
Bytes 18-21: Access Count (saturating)            — 4 bytes
Bytes 22-25: Last Access Timestamp (seconds)      — 4 bytes
Bytes 26-29: Migration Count                      — 4 bytes
Bytes 30-33: Allocation ID                        — 4 bytes (links to CUDA allocation metadata)
Bytes 34-37: Dedup Hash (partial, for R12)        — 4 bytes
Bytes 38-41: Reference Count (for shared pages)   — 4 bytes
Bytes 42-45: Last Migration Timestamp             — 4 bytes
Bytes 46-63: Reserved (future: hotness score,     — 18 bytes
             prefetch hints, fault count)
```

### Flags Field (2 bytes = 16 bits)

```
Bit 0:    Valid
Bit 1:    Dirty (modified since last migration)
Bit 2:    Pinned (cannot be evicted)
Bit 3:    Migrating (in-flight transfer)
Bit 4:    Shared (referenced by multiple contexts)
Bit 5:    Super-page member (part of a 2MB super-page)
Bit 6:    Read-only
Bit 7:    Accessed (set on access, cleared by scanner)
Bit 8:    Eviction candidate (marked by eviction scan)
Bit 9:    Prefetch target (marked for proactive migration)
Bit 10-15: Reserved
```

### Metadata Overhead Analysis

| Managed Capacity | Pages (64KB) | PTE Memory | Overhead % |
|-----------------|-------------|-----------|-----------|
| 12 GB (1 GPU VRAM) | 196,608 | 12 MB | 0.10% |
| 256 GB (1 node DRAM) | 4,194,304 | 256 MB | 0.10% |
| 8 TB (1 node NVMe) | 134,217,728 | 8 GB | 0.10% |
| 16 TB (2-node cluster) | 268,435,456 | 16 GB | 0.10% |

The 0.1% overhead is consistent across scales because page size and PTE size have a fixed ratio. 16 GB of metadata for a 16 TB cluster is acceptable — it fits comfortably in the 256 GB DRAM of a single node.

---

## 4. CUDA Integration: Intercepting Memory Operations

### cudaMalloc Interception

OuterLink already intercepts CUDA driver API calls via LD_PRELOAD. For R10, the memory interception must:

1. **`cuMemAlloc` / `cudaMalloc`:** Allocate virtual address space, create page table entries (initially unmapped to any tier). Record allocation metadata (size, CUDA context, creation time).

2. **`cuMemFree` / `cudaFree`:** Mark pages as invalid, free physical backing in all tiers. Important: `cudaFree` normally synchronizes all GPU work — OuterLink should use the CUDA VMM APIs (`cuMemUnmap`, `cuMemRelease`) to avoid this synchronization penalty where possible.

3. **`cuMemcpy*` / `cudaMemcpy`:** Key opportunity for tier awareness. When data is copied to GPU memory, the tier manager can decide which tier to place it in based on the destination context's current VRAM pressure.

4. **`cuLaunchKernel`:** Kernel launches tell the tier manager which memory regions will be accessed. This is the foundation for R11 prefetching — knowing what memory a kernel needs before it executes.

### CUDA VMM APIs (cuMem*)

CUDA 10.2+ provides low-level virtual memory management:
- `cuMemCreate` — Create physical allocation handle
- `cuMemMap` — Map physical memory to virtual address range
- `cuMemUnmap` — Unmap without synchronization
- `cuMemSetAccess` — Control access permissions per-GPU

These APIs enable OuterLink to:
- Reserve large virtual address ranges without backing physical memory
- Map/unmap physical pages from different tiers dynamically
- Avoid the synchronization penalty of cudaFree
- Potentially share virtual address ranges across GPUs in the cluster

### The Virtual Address Space Strategy

OuterLink should maintain a **cluster-wide virtual address space**:
- Each node reserves a non-overlapping range of virtual addresses
- Page table entries map these virtual addresses to physical locations in any tier on any node
- CUDA applications see a flat address space; the tier manager handles physical placement
- This is the foundation for R19 (network page faults) — a virtual address on one node can fault to bring data from a remote node

---

## 5. Migration Mechanics

### Migration Pipeline

```
1. Decision: Tier manager selects page for migration (eviction or promotion)
2. Pin: Mark page as "migrating" in PTE (prevent concurrent migration)
3. Allocate: Reserve space in destination tier
4. Transfer: DMA copy from source to destination
5. Update: Atomic PTE update (new tier, new physical address)
6. Invalidate: TLB/cache invalidation on source
7. Release: Free space in source tier, clear "migrating" flag
```

### Migration Costs by Tier Transition

| From | To | Bandwidth | 64KB Page | 2MB Page |
|------|----|-----------|-----------|----------|
| Local VRAM | Local DRAM | ~25 GB/s (PCIe) | ~7.5 us | ~85 us |
| Local VRAM | Remote VRAM | ~12-22 GB/s (RDMA) | ~8-13 us | ~95-170 us |
| Local DRAM | Local NVMe | ~28 GB/s (NVMe RAID-0) | ~7.3 us | ~73 us |
| Local DRAM | Remote DRAM | ~10-22 GB/s (RDMA) | ~8-16 us | ~95-210 us |
| Local NVMe | Remote NVMe | ~10 GB/s (RDMA+NVMe) | ~11.4 us | ~205 us |

Note: These are transfer-time only. Add ~5-10 us for DMA setup, PTE update, and TLB invalidation overhead per migration.

### Anti-Thrashing Mechanisms

Every surveyed system identifies thrashing as the primary failure mode. Required mechanisms:

1. **Minimum residency time:** A page must stay in its current tier for at least N microseconds before being eligible for migration. Prevents ping-pong.

2. **Migration rate limiting:** Cap the number of migrations per second per tier pair. Prevents migration traffic from saturating interconnects.

3. **Hysteresis bands:** Different thresholds for promotion vs demotion. A page must be "significantly hot" to promote but only "somewhat cold" to demote. Prevents oscillation near the boundary.

4. **Migration cost accounting:** Track the total bandwidth consumed by migrations. If migrations are consuming > X% of available bandwidth, back off.

5. **Pinning for active kernels:** While a CUDA kernel is executing, pages it accesses should be pinned in their current tier. Migrate only between kernel launches.

---

## 6. Distributed Page Table Considerations

### Ownership Model

For a multi-node cluster, each page must have a clear "home node" that owns its authoritative PTE:

- **Option 1: Allocation-based ownership.** The node that called cudaMalloc owns all PTEs for that allocation. Simple but creates hot spots if one node allocates most memory.

- **Option 2: Hash-based distribution.** PTE ownership is determined by hash(VPN) mod N. Distributes evenly but requires network lookup for non-local pages.

- **Option 3: Home-node = current-tier-node.** The PTE lives on whichever node currently hosts the physical page. Migration moves PTE ownership. Locality-optimal but complex.

**Recommended:** Option 1 for Phase 1 (simple, matches existing OuterLink client/server model). Migrate to Option 2 when scaling beyond 2-4 nodes.

### Coherence Protocol

When a page migrates, all nodes that cache its PTE must be invalidated. For a small cluster (2-4 nodes):
- Broadcast invalidation is acceptable
- Each node maintains a "page directory cache" of recently accessed remote PTEs
- On migration, source node sends invalidation to all nodes (or tracked sharers)

For larger clusters, a directory-based coherence protocol will be needed (future work beyond R10).

---

## Summary: Key Design Decisions for R10

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Base page size | 64 KB | Validated by GPUVM, WDDM; good balance of overhead and granularity |
| Super-page support | 2 MB (optional) | For contiguous hot regions; reduces metadata and TLB pressure |
| Page table structure | Flat hash table | O(1) lookup, easy to distribute, GPU-research-validated |
| PTE size | 64 bytes | Fits all required metadata + room for R11/R12 extensions |
| Metadata overhead | 0.1% of managed capacity | Acceptable at all scales up to 16+ TB |
| CUDA integration | Driver API interception + VMM APIs | Existing LD_PRELOAD approach + low-level VMM for dynamic mapping |
| Virtual address space | Cluster-wide, non-overlapping per node | Foundation for R19 network page faults |
| PTE ownership | Allocation-based (Phase 1) | Simple, matches client/server model |
| Anti-thrashing | Minimum residency + rate limiting + hysteresis | Non-negotiable based on all surveyed systems |

---

## Related Documents

- `01-existing-tiering-systems.md` — Survey of tiering approaches informing these decisions
- `03-eviction-policies.md` — Which eviction policies work with this page structure
- `../preplan.md` — R10 pre-plan integrating all research findings

## Open Questions

1. **Should the PTE hash table be stored in VRAM or DRAM?** DRAM is larger but adds PCIe latency for GPU-side lookups. Could keep a small cache of hot PTEs in VRAM.

2. **How do we handle CUDA unified memory (`cudaMallocManaged`) allocations?** These already have their own page migration via UVM. Do we intercept UVM faults, replace UVM entirely, or treat managed allocations as a special case?

3. **What is the optimal hash table load factor?** Robin Hood hashing works well up to 0.9 load factor, but GPU memory access patterns may cause clustering. Need benchmarking.

4. **Can we use CUDA VMM APIs on GeForce GPUs?** Some VMM APIs may be restricted to data-center GPUs. Need to verify on RTX 3060/4070.

5. **How does the 64KB page size interact with CUDA's 2MB allocation granularity?** A single cudaMalloc(2MB) would create 32 pages. Is sub-allocation tracking (multiple cudaMalloc calls sharing one 2MB physical allocation) needed?
