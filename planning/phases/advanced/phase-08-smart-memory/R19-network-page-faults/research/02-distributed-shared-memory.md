# R19 Research: Distributed Shared Memory Systems

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft
**Purpose:** Survey prior art in distributed shared memory (DSM), remote memory / memory disaggregation, and CXL memory pooling to inform OuterLink's network page fault design. Extract lessons on consistency models, coherency protocols, migration vs caching, and real-world performance numbers.

---

## 1. Classic Distributed Shared Memory Systems

### 1.1 Ivy (1986)

The first software DSM system. Provided a shared virtual address space across networked workstations using the CPU's MMU to trap accesses to remote pages.

- **Mechanism:** mprotect + SIGSEGV handler. Pages marked PROT_NONE until accessed; fault triggers network fetch.
- **Consistency:** Sequential consistency (strict --- all processors see same order of operations)
- **Granularity:** OS page size (4KB)
- **Problem:** False sharing. Two processors writing to different variables on the same page cause ping-pong migration.
- **Lesson for OuterLink:** Sequential consistency is too expensive over a network. False sharing at 4KB is severe; OuterLink's 64KB pages make false sharing even worse unless we use multiple-writer protocols or sub-page tracking.

### 1.2 TreadMarks (1994)

Major improvement over Ivy. Introduced **Lazy Release Consistency (LRC)** and **multiple-writer protocol**:

- **LRC:** Propagate modifications only at synchronization points (lock release/acquire), not on every write. This dramatically reduces network traffic for programs with locks.
- **Multiple writers:** Instead of migrating the entire page, each writer records diffs (list of byte-level modifications). On fault, the reader receives diffs from all writers and applies them. This eliminates false sharing.
- **Performance:** 7.4x speedup on 8 processors for Jacobi (100 Mbps ATM), 4.0x for Water (high synchronization rate).

**Lessons for OuterLink:**
- Relaxed consistency models are essential for network-scale shared memory
- Diff-based protocols avoid page-level false sharing (but add complexity)
- Synchronization latency dominates for fine-grained sharing --- GPU kernels are coarse-grained, which works in our favor
- The "home node" concept (one node owns each page, sends diffs on request) scales better than homeless protocols

### 1.3 Grappa (2015)

Modern software DSM for data-intensive computing, built on partitioned global address space (PGAS):

- **Delegate model:** Every piece of global memory is owned by a specific core. Remote access is performed by sending a "delegate" operation to the owning core, which executes it locally and returns the result.
- **Fine-grained access:** Instead of page-level sharing, Grappa provides word-level or cache-line-level access via active messages.
- **Performance:** High throughput for graph analytics and data-intensive workloads via message aggregation and work-stealing.

**Lessons for OuterLink:**
- Delegate operations (execute-at-data) can be more efficient than data migration for small accesses
- But for GPU workloads, we need bulk data (64KB+ pages) locally in VRAM --- delegate operations don't help when the GPU needs contiguous data for a kernel launch

---

## 2. RDMA-Based Distributed Memory Systems

### 2.1 FaRM (Microsoft Research, 2014)

Fast Remote Memory --- an RDMA-based distributed computing platform:

- **Design:** Exposes cluster memory as a shared address space. Applications execute distributed transactions using RDMA.
- **Performance:** 6.3 million ops/sec per machine. Average latency 41 us at peak throughput. 140 million TATP transactions/sec on 90 machines.
- **RDMA usage:** Lock-free reads via one-sided RDMA READ. Optimistic concurrency control with 2-phase commit.
- **Key insight:** RDMA gives 145x lower latency than TCP/IP at peak rates, and 12x lower unloaded. But local memory is still 23x faster than RDMA --- locality matters.

**Lessons for OuterLink:**
- RDMA one-sided operations bypass remote CPU entirely --- ideal for page fetching
- Data locality is paramount even with RDMA; this validates R11 prefetching
- Carefully manage NIC cache and queue pair resources for peak RDMA performance
- 41 us average latency for a full transactional operation gives a realistic bound for page fault + network fetch

### 2.2 Pilaf (2013)

RDMA-based key-value store using one-sided READ operations:

- **Design:** Clients read directly from server memory via RDMA READ. Puts go through the server (to handle synchronization).
- **Self-verifying data structures:** Detect read-write races without coordination. Hash table includes checksums that clients verify after reading.
- **Performance:** 1.3 million ops/sec with single CPU core (90% gets). Two round-trips per GET (index lookup + data fetch).

**Lessons for OuterLink:**
- Self-verifying structures (checksums/versioning) can detect stale reads without locks
- Two RDMA round-trips per operation is a significant overhead --- OuterLink should fetch pages in a single RDMA READ since the page address is already known from the page table

### 2.3 HERD (2014)

Contrarian RDMA key-value store that uses WRITE + messaging instead of READ:

- **Design:** Client sends request via RDMA WRITE to server. Server processes and replies via SEND.
- **Performance:** 26 million ops/sec with 5 us average latency. Outperforms READ-based designs at high throughput.
- **Why not READ?** Multiple READs per request (for index + data) underperform a single WRITE + SEND round-trip when the server CPU is available.

**Lessons for OuterLink:**
- For page faults where the requesting node knows the exact remote address, one-sided RDMA READ is optimal (single round trip, no remote CPU involvement)
- For page table lookups where the remote address is unknown, a request-reply pattern (like HERD) may be faster than multiple READs
- The choice between one-sided vs two-sided RDMA depends on whether the remote address is known

---

## 3. Remote Memory / Memory Disaggregation Systems

### 3.1 Infiniswap (NSDI 2017)

Remote memory paging system using RDMA as a swap backend:

- **Design:** Replaces local disk swap with RDMA to remote machines. Divides swap space into slabs distributed across remote nodes. Implemented as a Linux block device driver.
- **Mechanism:** Page-out uses RDMA WRITE (synchronous to remote, async to local disk). Page-in uses RDMA READ.
- **Performance:** 4x-15.4x throughput improvement over disk swap. Tail latency improved by up to 61x. Cluster memory utilization increased from 40.8% to 60%.
- **Limitation:** Linux block layer adds overhead unsuitable for microsecond-scale operations. The block layer was designed for disk I/O (millisecond scale), not RDMA (microsecond scale).

**Lessons for OuterLink:**
- RDMA as a swap/paging backend works well for bulk data
- The Linux VMM and block layer are NOT optimized for microsecond paging --- bypass them
- Remote memory paging increases cluster memory utilization significantly
- Prefetching on top of remote paging (follow-up work VANDI) improved performance by 15-102x --- validates R11's approach

### 3.2 LegoOS (OSDI 2018)

Disaggregated OS with separate processor (pComponent), memory (mComponent), and storage (sComponent) nodes:

- **Splitkernel model:** Each hardware component runs its own monitor. Memory functionality (page tables, MMU, TLBs) moved to mComponents; only caches remain at pComponents.
- **Key design insight:** Separate "memory for performance" (local cache) from "memory for capacity" (remote pool). Local cache provides fast access; remote pool provides large address space.
- **Performance:** Comparable to monolithic Linux servers with improved resource packing.

**Lessons for OuterLink:**
- The separation of performance memory (local VRAM) and capacity memory (remote VRAM/DRAM) maps directly to our tiered architecture
- Local caching of hot pages is essential --- remote memory alone cannot meet performance requirements
- This validates R10's tier model: Tier 0 (local VRAM) for performance, Tiers 1-4 for capacity

### 3.3 AIFM (OSDI 2020)

Application-Integrated Far Memory --- high-performance userspace far memory:

- **Design:** C++ STL-like interfaces for far memory. User-level implementation with user-space networking stack.
- **Approach:** Application explicitly marks data structures as "far-memory-capable." Runtime manages local vs remote placement transparently.
- **Performance:** Outperforms kernel-based approaches (LegoOS) due to elimination of kernel crossings and direct RDMA access.

**Lessons for OuterLink:**
- Userspace implementation outperforms kernel-based for far memory (fewer context switches)
- Application-level hints about data access patterns improve placement decisions
- Our interception layer provides these hints implicitly (kernel arguments reveal data access patterns)

### 3.4 Carbink (OSDI 2022)

Fault-tolerant far memory with erasure coding:

- **Design:** Uses erasure coding (not replication) for space-efficient fault tolerance. One-sided RDMA for data access. Remote memory compaction to reduce fragmentation.
- **Performance:** 29% lower tail latency and 48% higher application performance than Hydra (prior fault-tolerant far memory system), with at most 35% higher memory usage.

**Lessons for OuterLink:**
- Erasure coding is more space-efficient than replication for fault tolerance (relevant to R15)
- One-sided RDMA with compaction keeps remote memory efficient
- Tail latency matters as much as average latency for production workloads

---

## 4. CXL Memory Pooling

### 4.1 CXL Type 3 Devices

CXL (Compute Express Link) Type 3 devices provide host-managed device memory (HDM) accessible via CXL.mem protocol:

- **CXL 2.0:** Memory pooling (multiple hosts share a memory pool, but each partition is exclusive to one host)
- **CXL 3.0:** True memory sharing (multiple hosts can share the same memory region with hardware coherency)
- **CXL 4.0 (Nov 2025):** 128 GT/s bandwidth, 100+ TB shared pools

### 4.2 Coherency Mechanisms

| CXL Version | Coherency | Mechanism |
|-------------|-----------|-----------|
| CXL 2.0 | Software-managed | No hardware coherency for shared regions |
| CXL 3.0 | Hardware-managed | Back Invalidation (BI) snooping |
| CXL 4.0 | Enhanced hardware | Improved snoop filters, disaggregated coherency |

CXL 3.0's Back Invalidation (BI) protocol:
- When a device modifies shared memory, it sends BI snoop messages to all hosts caching that line
- Hosts invalidate their cached copies
- Similar to directory-based coherency but at the fabric level
- Snoop filter tracks which hosts cache which lines (precise at 64B, imprecise at 4KB)

### 4.3 CXL vs OuterLink's Approach

| Aspect | CXL | OuterLink |
|--------|-----|-----------|
| Interconnect | CXL 3.0/4.0 fabric | Ethernet/RDMA (ConnectX-5) |
| Latency | ~100-300 ns (CXL) | ~2 us (RDMA), ~50 us (TCP) |
| Coherency | Hardware (BI protocol) | Software (directory-based) |
| Granularity | 64B cache lines | 64KB pages |
| Scope | Rack-scale (CXL switches) | Datacenter/LAN scale |
| GPU support | CXL 3.0 Type 2 (future) | All CUDA GPUs (today) |
| Availability | 2026-2027 (production) | Now (software-only) |

**Key insight:** CXL solves the same problem OuterLink solves, but with hardware support at rack scale. OuterLink operates at larger scale (cross-rack, cross-building) and works with existing hardware. The approaches are complementary --- OuterLink could use CXL as a faster local interconnect in the future while maintaining RDMA for cross-rack communication.

### 4.4 Lessons for OuterLink

- CXL's snoop filter design (precise for critical regions, imprecise for bulk data) is a practical approach to coherency at scale
- Hardware coherency at 64B granularity is impractical over Ethernet --- our 64KB page granularity is the right choice for network-scale coherency
- The transition from exclusive partitioning (CXL 2.0) to true sharing (CXL 3.0) mirrors our phased approach: start with exclusive ownership, add shared-read later

---

## 5. Performance Summary

### 5.1 Fault/Access Latency Comparison

| System | Operation | Latency | Medium |
|--------|-----------|---------|--------|
| Local VRAM access | Memory read | ~0.3 us | HBM/GDDR6X |
| NVIDIA UVM fault (single) | GPU page fault | 10-50 us | PCIe |
| NVIDIA UVM fault (batch) | 32 faults batched | 223-553 us | PCIe |
| CXL Type 3 read | Cache-line fetch | 0.1-0.3 us | CXL fabric |
| RDMA READ (ConnectX-5) | 64KB page fetch | ~2-5 us | InfiniBand/RoCE |
| TCP page fetch | 64KB page | ~50-100 us | Ethernet |
| Infiniswap page-in | 4KB page via RDMA | ~3-5 us | InfiniBand |
| FaRM key-value read | Object fetch | ~5 us | RDMA |
| HERD key-value op | Full request-response | ~5 us | RDMA |
| userfaultfd resolve | UFFDIO_COPY | ~5-20 us | Local |
| Disk swap page-in | 4KB page from SSD | ~50-100 us | NVMe |
| Disk swap page-in | 4KB page from HDD | ~5000-10000 us | SATA |

### 5.2 Key Takeaway

For OuterLink with RDMA (ConnectX-5, 100Gbps):
- **Best case (RDMA, 64KB page):** ~2-5 us per page fetch
- **Including software overhead (fault detection, page table update, mapping):** ~10-20 us total
- **With cuMemMap installation:** add ~10-30 us for mapping
- **Total fault handling latency target:** 20-50 us per page (RDMA), 100-200 us (TCP)

Compare to: GPU kernel execution time of ~10-1000 us for typical operations. A single page fault adds 20-50 us, but if it stalls the entire kernel, the cost is the kernel's full execution time (since the kernel crashes and must restart in our cuMemMap approach). This is why R11 prefetching is critical --- preventing faults is far cheaper than handling them.

---

## 6. Consistency Models for OuterLink

### 6.1 Options

| Model | Description | Overhead | Suitability |
|-------|------------|----------|-------------|
| Sequential Consistency | All processors see same operation order | Very high (every write is globally visible) | Not suitable for network scale |
| Release Consistency | Writes visible at synchronization points | Medium (batch at sync) | Good for lock-based programs |
| Lazy Release Consistency | Writes propagated on demand at acquire | Low (only pull what's needed) | Good for DSM |
| Entry Consistency | Per-object consistency at lock acquire | Lowest (only sync locked object) | Complex to program |
| Single-Writer/Multiple-Reader | One writer at a time, many readers | Low (no coherency for reads) | Best for GPU workloads |

### 6.2 Recommended Model: Single-Writer/Multiple-Reader (SWMR) with Home Nodes

GPU workloads have a natural consistency model:
- **Model weights:** Read by all GPUs, written by none during inference (or by one optimizer during training)
- **Activations:** Written by one GPU, read by the next GPU in the pipeline
- **Gradients:** Written by one GPU, reduced across GPUs (all-reduce pattern)

SWMR maps perfectly:
- Pages are either **shared-read** (multiple readers, no writers) or **exclusive** (one owner, read-write)
- Transitioning from shared-read to exclusive requires invalidating all other copies
- Home node (the page's origin node in R10's tier system) tracks who has copies via a directory

This is simpler than MESI because we don't need the Exclusive-clean state --- at network scale, the distinction between Exclusive and Modified is not worth the protocol overhead.

---

## 7. Architecture Implications for OuterLink

### 7.1 What We Should Adopt

| Technique | From System | Application in OuterLink |
|-----------|-------------|------------------------|
| Directory-based coherency | Classic DSM, CXL 3.0 | Home node tracks page copies |
| One-sided RDMA for page fetch | FaRM, Pilaf, Infiniswap | Fetch remote pages without involving remote CPU |
| Userspace implementation | AIFM | Avoid kernel crossings for page management |
| Pre-launch mapping | Our cuMemMap approach | Map pages before kernel launch |
| Prefetching on top of paging | Infiniswap + VANDI | R11 prefetching prevents most faults |
| SWMR consistency | GPU workload semantics | Natural fit for ML workloads |
| Erasure coding for fault tolerance | Carbink | R15 future work |
| Tiered local/remote memory | LegoOS | R10 tier hierarchy |

### 7.2 What We Should Avoid

| Anti-Pattern | Why It Fails at Network Scale |
|-------------|------------------------------|
| Sequential consistency | Too expensive (every write = network round-trip) |
| 4KB page granularity | Too many pages to track, too many faults for GPU workloads |
| Kernel-level paging (block device) | Linux VMM not optimized for microsecond operations |
| Symmetric page migration (any node can pull from any node) | Leads to thrashing; home-node model is simpler |
| Hardware-only coherency | Not available on our target hardware (no CXL, no ATS) |

---

## Related Documents

- [01-gpu-page-fault-mechanisms.md](./01-gpu-page-fault-mechanisms.md) --- GPU-side fault handling
- [03-coherency-and-thrashing.md](./03-coherency-and-thrashing.md) --- coherency protocols and thrashing prevention
- [R10 Memory Tiering](../../R10-memory-tiering/README.md) --- tier hierarchy and page table
- [R11 Speculative Prefetching](../R11-speculative-prefetching/preplan.md) --- prevents most faults
- [FaRM Paper](https://www.microsoft.com/en-us/research/publication/farm-fast-remote-memory/)
- [Infiniswap Paper](https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/gu)
- [LegoOS Paper](https://www.usenix.org/system/files/osdi18-shan.pdf)
- [CXL Memory Sharing Paper](https://arxiv.org/html/2404.03245v1)
