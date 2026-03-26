# R12 Research: Copy-on-Write Across the Network

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft

## Purpose

Analyze how copy-on-write semantics work across a network of GPU nodes, including write detection, invalidation protocols, consistency models, and latency implications. This is the hardest part of cross-node dedup -- detecting that a shared page is being written and creating a private copy before the write corrupts the canonical shared data.

---

## 1. Local CoW Mechanisms (Background)

### CPU Page Table CoW

All major operating systems implement CoW via page table protection bits:

1. **Shared page is mapped read-only** in all processes that reference it
2. **Write attempt triggers a page fault** (hardware exception)
3. **OS fault handler:**
   - Allocates a new physical page
   - Copies content from the shared page to the new page
   - Updates the faulting process's page table entry to point to the new page (read-write)
   - Decrements the shared page's reference count
   - Resumes the faulting instruction (which now succeeds on the private copy)

**Latency:** 1-5 us for the fault handling path (allocate + copy + remap). The write itself then completes normally.

### How VMware Does CoW

VMware's TPS marks shared pages as read-only in the VM's shadow page table:
- VM writes to shared page -> VM exit (hardware trap to hypervisor)
- VMkernel allocates new page, copies content, remaps, decrements refcount
- VM re-enters with private page

### How KSM Does CoW

Linux KSM uses the standard `PROT_WRITE` removal + fault handler approach:
- Merged pages get their write bit cleared in all referencing PTEs
- `do_wp_page()` in the kernel handles the COW fault
- New page allocated, content copied, PTE updated

**Key observation:** COW on the local machine is a well-solved problem. The challenge is doing it across a network.

---

## 2. Network CoW: The Core Challenge

### The Problem

When Node A and Node B share a deduplicated page (both read-only references to a single canonical copy on Node A), and Node B writes to it:

1. Node B must detect the write BEFORE it happens (or simultaneously)
2. Node B must get a private copy of the page content
3. Node A's canonical copy must remain untouched
4. The dedup table must be updated to reflect that Node B now has a unique page

### Why This Is Hard

- **Latency:** Network round-trip (RDMA: ~2-5us, TCP: ~50-100us) is added to every COW fault
- **Atomicity:** The write must not proceed until the private copy is ready
- **GPU writes:** CUDA kernels can write at memory bandwidth speeds (936 GB/s on 3090). We cannot afford a network round-trip per write
- **False sharing:** If only part of a 64KB page is written, we still COW the entire page

---

## 3. Existing DSM Approaches

### Ivy (Yale, 1986) -- Sequential Consistency

The first software DSM system. Provides the illusion of shared memory across networked workstations:

- **Granularity:** 1KB pages
- **Protocol:** Single-writer, multiple-reader (SWMR)
- **On write:** Send invalidation to ALL nodes with copies. Wait for acknowledgments. Then write
- **Problem:** Excessive communication. Every write invalidates every copy. "Ping-pong effect" when multiple nodes write to the same page alternately

**Lesson for OuterLink:** Sequential consistency is too expensive for our use case. We need a relaxed model.

### TreadMarks (Rice University, 1994) -- Lazy Release Consistency

Major improvement over Ivy using lazy release consistency (LRC):

- **Consistency enforcement:** Deferred until synchronization points (lock acquire/release)
- **Write detection:** Pages mapped read-only. First write triggers protection fault. The DSM handler:
  1. Creates a "twin" (copy of the page before modification)
  2. Removes write protection so subsequent writes proceed without faults
  3. At release point, computes a "diff" between twin and current version
- **Diff propagation:** Only the CHANGED BYTES are sent, not the full page
- **Multiple writers:** Different nodes can write to different parts of the same page simultaneously (resolved at synchronization points)

**Key innovation -- Twin/Diff mechanism:**
```
Node B wants to write to shared page P:
  1. P is mapped read-only (COW trap)
  2. Fault handler copies P -> twin_P (snapshot before writes)
  3. P is remapped read-write (writes proceed at full speed)
  4. ... Node B writes to P many times ...
  5. At sync point: diff = P XOR twin_P (identifies changed bytes)
  6. Diff sent to owner/other nodes (compact, only changes)
```

**Performance characteristics:**
- First write to a shared page: ~5-50us (fault + twin copy)
- Subsequent writes: zero overhead (page is now writable)
- Sync point: diff computation (~1us for 4KB) + network send

**Lesson for OuterLink:** The twin/diff approach is highly relevant. For model weights that are read-only, COW faults will be extremely rare (only on fine-tuning or gradient updates). When they do happen, the twin approach minimizes impact.

### Munin (University of Rochester) -- Release Consistency

Similar to TreadMarks but with per-object consistency annotations:
- **Read-only objects:** No coherence overhead (exactly our model weight case)
- **Migratory objects:** Full page migrates to writer (good for exclusive write patterns)
- **Write-shared objects:** Multiple writers, merge at barriers

**Lesson:** Annotating pages as "read-only shared" at allocation time eliminates all COW overhead for those pages. This is exactly what model weights should be tagged as.

---

## 4. CUDA Memory Protection for GPU Writes

### Can We Trap GPU Writes to Shared Pages?

This is the critical question for VRAM-resident deduped pages.

### Method 1: CUDA UVM Page Faults (Post-Pascal)

NVIDIA GPUs since Pascal (2016) support hardware page faults for Unified Virtual Memory:
- GPU MMU detects access to unmapped/protected page
- Fault information written to a **fault buffer** (circular array in device memory)
- **nvidia-uvm driver** on host fetches faults, services them
- Supports **replayable faults:** GPU continues executing other warps while waiting

**For COW:**
1. Map deduped VRAM page as read-only in GPU page table
2. GPU kernel writes to page -> GPU page fault
3. nvidia-uvm driver intercepts fault
4. OuterLink handler: allocate new VRAM page, copy content, remap as read-write
5. GPU replays the faulting instruction

**Latency:** 10-50us per GPU page fault (fault buffer drain + host processing + page allocation + copy + remap). This is significant but amortized:
- Each page faults only ONCE (on first write)
- Subsequent writes to the same page proceed at full speed
- For model weights (read-only during inference), this path is never triggered

**Batch processing:** UVM processes faults in batches of up to 256. If multiple deduped pages are written simultaneously (e.g., gradient update touching many weight pages), the faults are batched efficiently.

### Method 2: Interception Layer Write Tracking

OuterLink intercepts all CUDA API calls. We can detect writes at the API level:

- **cuMemcpyHtoD(dst, src, size):** We know exactly which VRAM pages in `[dst, dst+size)` will be written
- **cuMemcpyDtoD(dst, src, size):** Same -- destination pages are written
- **cuLaunchKernel(f, ..., params):** If we can determine output pointers (via R8 kernel parameter introspection), we know which pages will be written

**For COW:**
1. Before dispatching the CUDA call, check if any destination pages are deduped
2. If yes: allocate private copy, copy content, update page table
3. Dispatch the original CUDA call (now writing to the private copy)

**Latency:** ~5-15us (page table lookup + VRAM allocation + device-to-device copy of 64KB). This happens BEFORE the CUDA call, so the kernel/memcpy sees a clean, writable private page.

**Advantage over UVM faults:** No GPU fault overhead. The COW is resolved proactively on the host before the GPU ever tries to write. This is OuterLink's unique advantage -- we control the entire API surface.

### Method 3: Lazy COW with Network Notification

For cross-node dedup, the writing node must also notify the network:

1. Node B detects write to deduped page (via Method 1 or 2)
2. Node B creates private copy locally
3. Node B sends `DEDUP_COW_BREAK` message to coordinator/owner (async, fire-and-forget)
4. Coordinator decrements reference count on the shared page
5. If refcount drops to 1, the remaining reference becomes the sole owner (no longer deduped)

This notification is **asynchronous** -- Node B does not wait for it. The write proceeds immediately on the private copy. The coordinator updates its bookkeeping in the background.

### Recommendation

**Primary: Method 2 (interception layer) + Method 3 (async network notification)**

Rationale:
- We already intercept all CUDA calls; adding COW checks is minimal overhead
- Proactive COW resolution avoids GPU faults entirely
- Async network notification means no write-path latency penalty
- Method 1 (UVM faults) serves as a safety net for writes we can't detect at the API level (e.g., kernels with unknown output parameters)

---

## 5. Invalidation Protocols

### Protocol Options for Cross-Node Dedup

When a deduped page is modified (COW break), the system must decide how to handle the coherence:

**Option A: No Invalidation (Read-Only Dedup)**
- Only dedup pages explicitly marked as read-only (model weights)
- No invalidation needed because writes are not allowed
- Simplest, zero runtime overhead
- **This is sufficient for our primary use case (LLM inference)**

**Option B: Directory-Based Invalidation**
- Coordinator maintains a directory: for each deduped page, tracks which nodes hold references
- On COW break: coordinator sends invalidations only to nodes with references
- Scales well (no broadcast)
- Overhead: directory entry per node per deduped page (~8 bytes per reference)

**Option C: Broadcast Invalidation**
- On COW break: sender broadcasts to all nodes
- Simple but doesn't scale (O(N) messages per write)
- Acceptable for small clusters (4-8 nodes)

**Option D: Lazy Invalidation (TreadMarks-style)**
- Don't invalidate immediately
- At next synchronization point (barrier, lock acquire), the acquiring node checks for pending invalidations
- Minimizes message count, maximizes staleness window

### Recommended Protocol

**Phase 1: Option A (read-only dedup only)**
- Only dedup pages that are guaranteed read-only (model weights, lookup tables)
- No invalidation protocol needed
- Zero runtime overhead
- Covers the primary use case (4x memory savings for LLM inference)

**Phase 2: Option B (directory-based invalidation)**
- Extend to writable pages with COW semantics
- Coordinator tracks references per page
- On COW break: notify coordinator, coordinator invalidates stale references
- This enables dedup for KV cache and other read-mostly data

---

## 6. Consistency Models

### Strict Consistency
- Every write is instantly visible to all nodes
- Requires synchronous invalidation on every write
- **Not viable** for network-connected nodes (latency too high)

### Sequential Consistency
- All nodes see the same order of operations
- Used by Ivy DSM
- Still requires synchronous invalidation
- **Too expensive** for our use case

### Release Consistency
- Writes are only visible at synchronization points (lock release / barrier)
- Used by TreadMarks and Munin
- Allows writes to accumulate without network communication
- **Good fit** for CUDA kernel launches (natural synchronization points: kernel launch = acquire, kernel completion = release)

### Entry Consistency
- Consistency enforced per-object, only when the object's lock is acquired
- Most relaxed, least communication
- **Best fit** for OuterLink: each deduped page group has its own consistency domain

### Recommendation

**For read-only dedup (model weights):** No consistency protocol needed. Data is immutable.

**For writable dedup (future):** Release consistency, where CUDA kernel launches are the synchronization points. Diffs are computed and propagated after kernel completion, not during execution.

---

## 7. Impact on Latency When CoW Triggers

### Worst Case: Full Page Copy Over Network

If Node B writes to a deduped page and the only copy is on Node A:

```
Node B: detect write to deduped page    [~1us, interception check]
Node B: request page from Node A         [network round-trip: 2-5us RDMA]
Node A: DMA read 64KB from VRAM          [~1us, memory bandwidth]
         transfer 64KB over RDMA          [~5us at 100Gbps]
Node B: receive page, store in VRAM      [~1us, DMA write]
Node B: update local page table           [~0.5us]
Node B: proceed with write                [0]
-----------------------------------------
Total: ~10-13us via RDMA
Total: ~60-80us via TCP
```

### Optimized Case: Local Cache Hit

If Node B has a cached copy of the page in local DRAM (from R10 tiering):

```
Node B: detect write to deduped page     [~1us]
Node B: allocate new VRAM page            [~1us]
Node B: copy from local DRAM to VRAM      [~2us for 64KB via DMA]
Node B: update local page table           [~0.5us]
Node B: async notify coordinator          [fire-and-forget, ~0us on write path]
Node B: proceed with write                [0]
----------------------------------------------
Total: ~4.5us
```

### Best Case: Read-Only Dedup (Model Weights)

```
Write attempt: NEVER (model weights are read-only during inference)
COW overhead: 0
```

### Latency Summary

| Scenario | COW Latency | Frequency |
|----------|-------------|-----------|
| Read-only dedup (model weights) | 0 | Never triggers |
| COW with local DRAM copy | ~4.5us | Rare (only on write to shared page) |
| COW with remote fetch (RDMA) | ~10-13us | Very rare (only if no local copy) |
| COW with remote fetch (TCP) | ~60-80us | Very rare, fallback path |

For context, a single CUDA kernel launch takes ~5-20us of overhead. A COW fault of ~10us is comparable to one kernel launch -- noticeable but not catastrophic if it happens rarely.

---

## 8. CUDA-Specific Considerations

### GPU Page Table Manipulation

- NVIDIA does not expose a public API for directly manipulating GPU page tables
- UVM manages GPU page tables internally via the nvidia-uvm kernel module
- OuterLink's approach: use CUDA interception to control memory mapping BEFORE the GPU sees it, rather than manipulating GPU page tables directly

### cuMemAddressReserve / cuMemCreate (CUDA Virtual Memory Management)

CUDA 10.2+ provides virtual memory management APIs:
- `cuMemAddressReserve`: Reserve virtual address range
- `cuMemCreate`: Create physical allocation
- `cuMemMap`: Map physical allocation to virtual address
- `cuMemSetAccess`: Control per-GPU access permissions

These APIs allow OuterLink to:
1. Reserve a virtual address range for a deduped page group
2. Map the canonical physical copy at that address on all GPUs
3. Set access to READ_ONLY on all GPUs
4. On COW: create new physical allocation, copy content, remap the writing GPU's virtual address to the new physical allocation, set READ_WRITE

**This is the cleanest path for VRAM COW** -- it uses official CUDA APIs without needing to hack GPU page tables.

### Peer Access and VRAM Sharing

With NVLink or PCIe peer access enabled (`cuCtxEnablePeerAccess`), one GPU can directly read another GPU's VRAM. This could allow dedup without copying:
- GPU 0 holds the canonical copy of a weight tensor
- GPUs 1-3 read directly from GPU 0's VRAM via peer access
- No copy needed, no dedup table needed

**Limitation:** Peer access only works within a single machine (PCIe/NVLink fabric). For cross-node (network) dedup, we still need the full COW mechanism.

**Optimization for single-node multi-GPU:** Use peer access for intra-node dedup (zero-copy), use COW for cross-node dedup (copy-on-write with network notification).

---

## Open Questions

1. Can we use `cuMemSetAccess` with `CU_MEM_ACCESS_FLAGS_PROT_READ` to enforce read-only on deduped VRAM pages without UVM overhead?
2. How does peer access latency compare to local VRAM copy latency for the canonical shared page?
3. For multi-writer scenarios (e.g., parallel fine-tuning), should we use diff-based updates (TreadMarks-style) or full page replacement?
4. How does R10's eviction policy interact with COW? If the canonical copy of a deduped page is evicted from VRAM, what happens to all references?

## Related Documents

- R12 Research 01 -- Existing Dedup Systems
- R12 Research 02 -- Hashing and Detection
- R10 Memory Tiering (page table, eviction policy)
- R8 Kernel Parameter Introspection (detecting kernel outputs for write tracking)
- R12 Preplan
