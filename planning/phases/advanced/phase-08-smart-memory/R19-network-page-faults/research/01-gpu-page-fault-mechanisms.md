# R19 Research: GPU Page Fault Mechanisms

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft
**Purpose:** Survey the mechanisms available for intercepting and handling GPU memory faults, from hardware page faults to userspace API tricks, to determine which approach OuterLink should use for transparent remote memory.

---

## 1. NVIDIA Unified Virtual Memory (UVM) Internals

### 1.1 Architecture

NVIDIA UVM is implemented by the `nvidia-uvm.ko` kernel module, which works alongside the main `nvidia.ko` driver. UVM provides a single unified virtual address space accessible from both CPU and GPU. Instead of explicit `cudaMemcpy`, UVM automatically migrates pages on demand via page faulting.

The kernel module maintains per-process `uvm_va_space_t` structures that track where each page resides (CPU memory, GPU memory, or both). The fundamental unit of management is the `uvm_va_block_t`, which covers a contiguous range of virtual addresses.

### 1.2 GPU Page Fault Flow (Volta+)

Starting with Volta (2017), NVIDIA GPUs support **hardware replayable page faults**:

1. **TLB Miss:** GPU Memory Management Unit (GMMU) detects unmapped address
2. **Fault Buffer Write:** GMMU writes fault information (address, access type, SM ID) into a circular fault buffer in GPU memory
3. **PCIe Interrupt:** Hardware interrupt notifies the UVM kernel driver on the host CPU
4. **Batch Retrieval:** Driver reads a batch of faults (up to 32) from the fault buffer into host memory
5. **Preprocessing:** Faults are sorted by address, deduplicated, and prefetch requests are inserted
6. **Page Table Walk:** CPU page table walks, allocation, and eviction scheduling
7. **DMA Migration:** DMA engine copies pages from source to destination (CPU-to-GPU or GPU-to-GPU)
8. **PTE Update:** GPU page table entries are updated, TLB invalidated
9. **Warp Resume:** Stalled GPU warps are replayed and resume execution

Key insight: while faulting warps stall, other warps on the same SM can continue executing (latency hiding via warp scheduling).

### 1.3 Performance Characteristics

| Metric | Value | Source |
|--------|-------|--------|
| Per-fault latency | 10-50 us | NVIDIA docs, academic measurements |
| Batch processing time | 223-553 us (median 313 us) | SC'21 paper (Talendev et al.) |
| GPU runtime fault handling | ~47% of batch processing time | SC'21 paper |
| Batch size | Up to 32 faults | nvidia-uvm.ko source |
| PCIe round-trip (interrupt) | ~1-2 us | PCIe spec |
| DMA transfer (4KB page, PCIe 4.0) | ~0.3 us | Bandwidth-limited |
| DMA transfer (64KB page, PCIe 4.0) | ~2.5 us | Bandwidth-limited |

The batch processing overhead dominates: even a single fault pays the full batch cost (~300 us) because of software processing, page table walks, and TLB management. This is why prefetching is critical --- avoiding faults altogether is far cheaper than handling them quickly.

### 1.4 Eviction and Oversubscription

When GPU VRAM is full, UVM evicts least-recently-used (LRU) pages to system RAM. Evicted pages remain accessible --- accessing them later triggers another page fault. This enables memory oversubscription (allocating more managed memory than physical VRAM).

### 1.5 Hardware Access Counters (Volta+)

Starting with Volta, hardware access counters track page-level access frequency. When a page becomes "hot" through repeated access, the counter generates a notification to the UVM driver, which can proactively migrate the page before a fault occurs. This transforms migration from reactive to proactive.

### 1.6 Thrashing Detection

The UVM driver has built-in thrashing detection with configurable parameters:

- `uvm_perf_thrashing_enable`: Enable/disable thrashing detection
- `uvm_perf_thrashing_threshold`: Number of faults before a page is considered thrashing
- `uvm_perf_thrashing_pin_threshold`: Faults before a page is pinned to avoid further migration
- `uvm_perf_thrashing_lapse_usec`: Time window for counting faults
- `uvm_perf_thrashing_nap`: Throttle delay for thrashing pages

When thrashing is detected, the driver establishes **dual mappings** --- both CPU and GPU can access the page simultaneously (at reduced bandwidth), eliminating the ping-pong migration.

### 1.7 Relevance to OuterLink

**Cannot use directly:** UVM's page fault handling is entirely internal to the nvidia-uvm.ko module. There is no public API to intercept or redirect GPU page faults. We cannot hook into the fault buffer or inject custom fault handlers.

**Can cooperate with:** UVM's migration policies could be extended if we wrote a custom kernel module that hooks into the UVM fault path. However, this would be fragile (dependent on NVIDIA driver internals that change between versions) and would require the open-source driver (only supports Turing+ GPUs).

**Lessons learned:**
- Batch processing is essential to amortize per-fault overhead
- Hardware access counters enable proactive migration (similar to R11's role)
- Thrashing detection with adaptive response (dual mapping) is a proven technique
- Per-fault latency of 10-50 us is the floor for any software-based approach

---

## 2. CUDA Virtual Memory Management API (cuMemMap)

### 2.1 Overview

CUDA 10.2 introduced low-level VMM APIs that decouple virtual address space from physical memory:

1. **`cuMemCreate`**: Allocate physical memory (returns a handle, no mapping yet)
2. **`cuMemAddressReserve`**: Reserve a virtual address range (no physical backing)
3. **`cuMemMap`**: Map a physical allocation to a reserved virtual address range
4. **`cuMemSetAccess`**: Set access permissions (which devices can read/write)
5. **`cuMemUnmap`**: Remove the mapping (virtual range becomes unmapped again)

### 2.2 Key Property: No Automatic Page Faults

Unlike UVM, the VMM API does **not** provide lazy/on-demand page fault handling. If a GPU kernel accesses an unmapped or access-denied region, it **crashes** (not faults gracefully). This means we cannot use cuMemMap alone to implement demand paging.

### 2.3 OuterLink's "Trap-and-Map" Pattern

However, we can build a **software demand paging** system on top of cuMemMap:

1. Reserve a large virtual address range covering the entire cluster's memory
2. Only map regions that are locally resident (backed by local VRAM or pinned DRAM)
3. When a kernel needs remote data, the **interception layer** (which sees kernel launches and their arguments) pre-maps the required pages before launch
4. After kernel completion, unmap pages that should be reclaimed

This is not true demand paging (faults during kernel execution), but it is **pre-launch demand paging** that works entirely in userspace without a kernel module.

### 2.4 Access Control for COW

`cuMemSetAccess` can mark regions as read-only. Combined with R12's deduplication, shared pages can be mapped read-only on all nodes. A write attempt would crash the kernel --- but our interception layer detects writes (via cuMemcpy* hooks and kernel argument analysis) before they happen, triggering COW proactively.

### 2.5 Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| cuMemCreate | ~50-100 us | Physical allocation |
| cuMemMap | ~10-30 us | Virtual-to-physical binding |
| cuMemSetAccess | ~5-15 us | Permission change |
| cuMemUnmap | ~10-30 us | Remove mapping |

These operations are fast enough for pre-launch mapping but too slow to call per-page during a running kernel.

### 2.6 Verdict for OuterLink

**Primary mechanism for Phase 1:** cuMemMap-based pre-launch demand paging. This works entirely in userspace, requires no kernel module, and is compatible with all CUDA 10.2+ GPUs. The limitation is that faults during kernel execution cannot be caught --- R11 prefetching must ensure data is mapped before launch. R19's role becomes ensuring the page table is correct before each kernel launch.

---

## 3. GPU Hardware Page Faults: ATS (Address Translation Services)

### 3.1 What ATS Provides

ATS is a PCIe capability that allows devices to request address translations from the system IOMMU. Combined with PRI (Page Request Interface), it enables:

1. Device sends ATS translation request to IOMMU via PCIe
2. If page is not mapped, IOMMU returns a fault
3. Device sends a Page Request to the IOMMU
4. IOMMU triggers a host interrupt for fault handling
5. Host resolves the fault (maps the page) and responds
6. Device retries the access

### 3.2 GPU Support

| Architecture | ATS Support | Connection Required |
|-------------|-------------|-------------------|
| Volta (V100, 2017) | Yes | NVLink to POWER9 |
| Ampere (A100, 2020) | Limited | NVLink preferred |
| Hopper (H100, 2022) | Full via NVLink-C2C | Grace CPU required |
| GeForce (all) | No ATS exposed | N/A |

Critical limitation: **ATS on NVIDIA GPUs requires NVLink to the host CPU**, not PCIe. On standard x86 systems with PCIe-connected GPUs (including all GeForce cards), ATS is not available. The Grace Hopper Superchip has full ATS via NVLink-C2C, but that is a datacenter product.

### 3.3 Relevance to OuterLink

**Not usable for our target hardware.** OuterLink targets GeForce GPUs on consumer PCs connected via PCIe. ATS requires NVLink to a supported CPU (POWER9 or Grace). This mechanism is off the table for our use case.

However, ATS is worth understanding because:
- It shows what "true" hardware-managed GPU page faults look like
- CXL-connected accelerators may use similar mechanisms in the future
- If OuterLink ever supports datacenter GPUs with NVLink, ATS becomes relevant

---

## 4. Linux HMM (Heterogeneous Memory Management)

### 4.1 What HMM Provides

HMM is a Linux kernel framework that helps device drivers mirror CPU page tables into device-specific page tables and keep them synchronized. Key features:

- **Address space mirroring:** Duplicate CPU page table entries into device format
- **MMU notifiers:** Get callbacks when CPU page table changes (unmap, migration, permission change)
- **Device page faults:** When device accesses unmapped address, driver uses `hmm_range_fault()` to resolve
- **ZONE_DEVICE memory:** Allocate `struct page` for device memory, enabling standard migration mechanisms
- **Migration helpers:** `migrate_vma_setup()`, `migrate_vma_pages()`, `migrate_vma_finalize()`

### 4.2 How It Works

1. Device driver registers `mmu_interval_notifier` for a virtual address range
2. When CPU page tables change, driver receives invalidation callbacks
3. On device page fault, driver calls `hmm_range_fault()` to snapshot current CPU page table state
4. Driver populates device page table from the HMM snapshot
5. For migration: driver allocates ZONE_DEVICE pages, uses migration helpers to move pages into device memory

### 4.3 NVIDIA's Use of HMM

NVIDIA's open-source UVM driver (nvidia-uvm.ko, Turing+ only) integrates with Linux HMM when available:
- Uses `hmm_range_fault()` for querying memory residency
- Uses MMU interval notifiers for invalidation tracking
- Requires kernel 6.1.24+ and CUDA 12.2+ with open-source r535 driver

### 4.4 AMD's Use of HMM (Comparison)

AMD's open-source `amdgpu` driver uses HMM for its Shared Virtual Memory (SVM) implementation:
- XNACK (retry-on-fault) allows GPU to fault and recover gracefully
- Supported on MI200+ and Vega (with `HSA_XNACK=1`)
- Page faults trigger automatic migration between CPU and GPU memory
- Standard `malloc`/`new` allocations are accessible from GPU (with HMM-enabled kernel)

AMD's implementation is notable because it is **fully open-source** and integrated into the mainline Linux kernel, making it a useful reference for how GPU page faults can be handled via HMM.

### 4.5 Relevance to OuterLink

**Requires a kernel module.** HMM is a kernel framework --- using it requires writing a kernel module that registers as an HMM device and implements the device page table operations. This is the "right" way to do GPU demand paging on Linux, but it adds significant deployment complexity.

**Could be used for Phase 2/3:** A custom kernel module that:
1. Registers with HMM for the application's address space
2. Receives fault notifications when GPU accesses unmapped remote pages
3. Fetches pages from remote nodes via RDMA
4. Maps them into GPU-accessible memory via cuMemMap
5. Updates device page table and resumes GPU

This would give us true in-kernel-execution fault handling, but at the cost of kernel module maintenance across kernel versions.

---

## 5. userfaultfd for Host-Side Fault Handling

### 5.1 Mechanism

`userfaultfd` (Linux 4.3+) allows userspace to handle page faults on registered memory regions:

1. Create file descriptor via `userfaultfd(2)` syscall
2. Register memory ranges with `UFFDIO_REGISTER`
3. When a thread accesses an unpopulated page, kernel delivers a 32-byte fault event
4. Handler resolves fault via `UFFDIO_COPY` (provide data), `UFFDIO_ZEROPAGE` (zero-fill), or `UFFDIO_CONTINUE` (map existing page)

### 5.2 Registration Modes

| Mode | Trigger | Resolution |
|------|---------|-----------|
| `UFFDIO_REGISTER_MODE_MISSING` | Access to unmapped page | `UFFDIO_COPY` or `UFFDIO_ZEROPAGE` |
| `UFFDIO_REGISTER_MODE_MINOR` | Access to page in cache but no PTE | `UFFDIO_CONTINUE` |
| `UFFDIO_REGISTER_MODE_WP` | Write to write-protected page | `UFFDIO_WRITEPROTECT` |

### 5.3 Existing Use Cases

- **KVM live migration:** VM moves to new host, memory follows via userfaultfd as guest faults
- **CRIU:** Lazy process restore, serve checkpoint pages on demand
- **Distributed shared memory:** Fetch pages from remote nodes on access
- **UMap (LLNL):** mmap-like interface for out-of-core execution

### 5.4 Performance

Typical userfaultfd fault handling latency: 5-20 us for local resolution (UFFDIO_COPY with data ready). Add network RTT for remote fetch.

### 5.5 Relevance to OuterLink

**Useful for host-side pinned memory:** When a GPU needs to DMA from pinned host memory that is backed by remote data, userfaultfd can intercept the host-side access and fetch from the network. This is useful for the host-staged data path (Phase 1 transport).

**Not useful for VRAM faults:** userfaultfd only handles CPU page faults on host memory. GPU page faults on VRAM are handled by the GPU's own fault mechanism (which goes through nvidia-uvm.ko, not userfaultfd).

**Hybrid approach:** Use userfaultfd for host memory regions that back GPU DMA. When the DMA engine accesses a host page that isn't populated, userfaultfd traps, fetches from remote node via RDMA, and resolves. The DMA engine then finds the data and proceeds.

---

## 6. Custom Kernel Module Approach

### 6.1 What a Kernel Module Enables

A custom OuterLink kernel module could:

1. **Hook GPU fault buffer:** Intercept GPU page faults before nvidia-uvm.ko processes them
2. **Implement custom migration:** On fault, fetch page from remote node via RDMA, install into GPU page table
3. **Use HMM framework:** Register as an HMM device, get standard page table mirroring
4. **Access PCIe BAR1:** Direct RDMA writes into GPU VRAM via OpenDMA without going through CUDA APIs

### 6.2 What Requires a Kernel Module vs What Does Not

| Capability | Kernel Module? | Alternative |
|-----------|---------------|-------------|
| True GPU page fault interception | **Yes** | Pre-launch mapping via cuMemMap |
| Custom fault handler for GPU faults | **Yes** | Interception layer detects access patterns |
| userfaultfd for host memory | No | Standard syscall |
| cuMemMap/cuMemUnmap | No | CUDA driver API |
| RDMA transfers | No (with libibverbs) | Userspace verbs |
| PCIe BAR1 access (OpenDMA) | **Yes** (Phase 5) | N/A |
| HMM integration | **Yes** | N/A |
| Hardware access counter reading | **Yes** | Use CUPTI (limited) |

### 6.3 Phased Approach

**Phase 1 (No kernel module):**
- cuMemMap-based pre-launch mapping
- Interception layer ensures pages are mapped before kernel launch
- userfaultfd for host-staged memory regions
- R11 prefetching prevents most faults

**Phase 2 (Optional kernel module):**
- Custom module intercepts actual GPU faults for cases where pre-mapping misses
- HMM integration for proper page table synchronization
- Hardware access counter integration for proactive migration

**Phase 3 (OpenDMA kernel module, already planned for Phase 5):**
- Direct RDMA to/from GPU VRAM
- Combined with fault handling for sub-microsecond page installation

---

## 7. Comparison of Approaches

| Approach | Fault Handling | Latency | Kernel Module? | GPU Compatibility | Complexity |
|----------|---------------|---------|---------------|-------------------|-----------|
| NVIDIA UVM (nvidia-uvm.ko) | True GPU faults | 10-50 us/fault, 300 us/batch | Requires NVIDIA's | All CUDA GPUs | N/A (proprietary) |
| cuMemMap pre-launch | Pre-launch mapping | 0 (no fault) or kernel crash | No | CUDA 10.2+ | Medium |
| ATS (PCIe) | Hardware page fault | ~1 us PCIe + resolution | No (hardware) | NVLink only | N/A (not available) |
| Linux HMM + custom module | True device faults | 5-20 us + migration | **Yes** | Turing+ (open driver) | High |
| userfaultfd (host only) | Host memory faults | 5-20 us + remote fetch | No | All (host-side) | Medium |
| Custom kernel module | Full control | Depends on implementation | **Yes** | All (with driver hooks) | Very High |

---

## 8. Recommended Architecture for OuterLink

### Primary Mechanism: cuMemMap Pre-Launch Demand Paging

The recommended approach for R19 is **interception-layer demand paging** using the cuMemMap API:

1. OuterLink reserves a large GPU virtual address range covering the entire cluster's memory pool
2. The R10 page table tracks which 64KB pages are locally resident (mapped) vs remote (unmapped)
3. Before each kernel launch (intercepted via cuLaunchKernel hook):
   a. R11 prefetching has already moved predicted pages to local VRAM
   b. R19 checks the kernel's memory arguments to identify any remaining unmapped pages
   c. For each unmapped page needed: fetch from remote node, cuMemMap into the virtual range
4. If a kernel accesses a truly unpredicted page, the kernel crashes --- this is the failure mode that R11 prefetching must prevent
5. Fallback: re-launch the kernel after fetching the missing pages (detected via CUDA error)

### Safety Net: Kernel Re-Launch on Fault

When a kernel crashes due to accessing unmapped memory:
1. Catch the CUDA error (`CUDA_ERROR_ILLEGAL_ADDRESS`)
2. Parse the error to identify the faulting address (if available from CUDA error info)
3. Fetch the missing page from the remote node
4. Map it via cuMemMap
5. Re-launch the kernel

This is expensive (kernel restart overhead) but functions as a correctness safety net. With R11 prefetching achieving >90% hit rate, this path should be rare (<1% of kernel launches).

### Future Enhancement: Custom Kernel Module

For Phase 2, a custom kernel module can provide true fault handling without kernel restart:
1. Hook into the GPU fault path (either via nvidia-uvm.ko extension or custom HMM device)
2. On fault: pause GPU warp, fetch page via RDMA, install mapping, resume
3. This eliminates the kernel-restart overhead but adds deployment complexity

---

## 9. Open Questions

1. **Can CUDA_ERROR_ILLEGAL_ADDRESS provide the faulting address?** If not, we need a different mechanism to identify which page caused the crash. NVIDIA's error reporting may only say "illegal access" without specifying the address.

2. **Can cuMemMap be called while a kernel is running on a different stream?** If yes, we could map pages concurrently with kernel execution, reducing the window for faults.

3. **What is the maximum virtual address range we can reserve with cuMemAddressReserve?** This determines the maximum cluster memory pool size we can support.

4. **Does the open-source nvidia-uvm.ko expose any hooks or extension points?** If so, we might avoid writing a fully custom kernel module.

5. **Can we use CUPTI to detect which pages a kernel accessed post-execution?** This would feed into R11's profiling without needing hardware access counters.

---

## Related Documents

- [R10 Memory Tiering](../../R10-memory-tiering/README.md) --- page table design
- [R11 Speculative Prefetching](../R11-speculative-prefetching/preplan.md) --- prevents most faults
- [R12 Memory Deduplication](../R12-memory-deduplication/preplan.md) --- read-only shared pages
- [nvidia-uvm.ko DeepWiki Analysis](https://deepwiki.com/NVIDIA/open-gpu-kernel-modules/3.4-nvidia-uvm.ko-unified-memory)
- [NVIDIA CUDA VMM API Documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html)
- [Linux HMM Documentation](https://www.kernel.org/doc/html/v5.0/vm/hmm.html)
- [SC'21 UVM Analysis Paper](https://tallendev.github.io/assets/papers/sc21.pdf)
