# R18 Research: RDMA Atomics and Coherency

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Purpose:** Determine whether RDMA atomic operations and software coherency protocols can emulate NVLink's hardware features over the network.

---

## 1. RDMA Atomic Operations

### 1.1 InfiniBand/RoCE Atomic Capabilities

RDMA provides two native atomic operations:

| Operation | Verb | Size | Description |
|-----------|------|------|-------------|
| Compare-and-Swap | `IBV_WR_ATOMIC_CMP_AND_SWP` | 8 bytes (64-bit) | If value at address == compare_value, replace with swap_value. Returns old value. |
| Fetch-and-Add | `IBV_WR_ATOMIC_FETCH_AND_ADD` | 8 bytes (64-bit) | Add value to address atomically. Returns old value. |

**Transport requirement:** RDMA atomics require Reliable Connection (RC) queue pairs. They are NOT supported on Unreliable Datagram (UD) or other transport types.

**Memory registration:** Target memory must be registered with `IBV_ACCESS_REMOTE_ATOMIC` and `IBV_ACCESS_LOCAL_WRITE` flags.

### 1.2 ConnectX-5 Extended Atomics

ConnectX devices extend standard RDMA atomics with:

| Extended Operation | Description |
|-------------------|-------------|
| Masked Compare-and-Swap | CAS with a bitmask --- only compare/swap specified bits |
| Multi-field Fetch-and-Add | Fetch-and-add with bit boundary control (carry doesn't propagate across fields) |
| Variable-size arguments | Arguments can be 8 bytes, with max inline argument sizes defined by device caps |

Extended atomics are exposed via `ibv_query_device_ex()` capability flags. They expand what can be done in a single atomic operation but are still fundamentally limited to CAS and fetch-add variants.

### 1.3 PCI Atomic Operations (PCIe 2.1+)

PCIe itself defines atomic operations at the TLP (Transaction Layer Packet) level:

| PCIe Atomic | Sizes |
|-------------|-------|
| Swap | 32-bit, 64-bit |
| Fetch-and-Add | 32-bit, 64-bit |
| Compare-and-Swap | 32-bit, 64-bit, 128-bit |

These operate on PCIe BAR addresses and do NOT go through the verbs API. They are relevant for OpenDMA (Phase 5) where the ConnectX-5 DMA engine directly accesses GPU BAR1 --- PCIe atomics could theoretically be used on BAR1 addresses.

**Open question:** Do ConnectX-5 NIC DMA engines support issuing PCIe atomic TLPs to remote BAR addresses? If yes, this provides a hardware path for remote GPU atomics without going through the RDMA verbs stack.

---

## 2. GPU Atomic Operations (CUDA)

### 2.1 Full CUDA Atomic Set

CUDA provides a rich set of atomic operations, all operating on global or shared memory:

| CUDA Atomic | Sizes | Hardware Impl |
|-------------|-------|---------------|
| `atomicAdd` | 32i, 64i, 32f, 64f | Direct hardware |
| `atomicSub` | 32i | Via atomicAdd with negation |
| `atomicMin` | 32i, 64i, 32u, 64u | Direct hardware (Volta+) |
| `atomicMax` | 32i, 64i, 32u, 64u | Direct hardware (Volta+) |
| `atomicExch` | 32i, 64i, 32f | Direct hardware |
| `atomicCAS` | 32i, 64i, 16i (sm_70+) | Direct hardware |
| `atomicAnd` | 32i, 64i | Direct hardware |
| `atomicOr` | 32i, 64i | Direct hardware |
| `atomicXor` | 32i, 64i | Direct hardware |
| `atomicInc` | 32u | Direct hardware |
| `atomicDec` | 32u | Direct hardware |

### 2.2 GPU Atomic Implementation

- **Shared memory atomics:** Execute in the SM's L1/shared memory unit. Sub-microsecond latency. Block-local only.
- **Global memory atomics:** Execute at the L2 cache (same GPU) or at the remote GPU's memory controller (NVLink peer). GPU thread stalls until the atomic completes and the old value is returned.
- **Cross-GPU atomics (NVLink):** The atomic operation packet is sent over NVLink to the target GPU's memory controller, which executes the operation and returns the result. Latency ~100-300ns.

### 2.3 Memory Ordering (Volta+ / SM 7.0+)

Since Volta, CUDA supports explicit memory ordering via `cuda::std::atomic` and PTX-level qualifiers:

| Ordering | PTX Qualifier | Meaning |
|----------|--------------|---------|
| Relaxed | `.relaxed` | No ordering guarantees relative to other operations |
| Acquire | `.acquire` | Subsequent reads see all writes before the release |
| Release | `.release` | All prior writes are visible to threads that acquire |
| Acq_Rel | `.acq_rel` | Both acquire and release |

**Scope qualifiers:** Control visibility of atomic operations:

| Scope | Qualifier | Visible To |
|-------|-----------|-----------|
| Block | `_block` | Threads within same thread block |
| Device | (default) | All threads on same GPU |
| System | `_system` | All threads in system (including CPU) |

**Critical for R18:** The `_system` scope is what NVLink uses for cross-GPU atomics. Over the network, we must ensure that system-scope atomics trigger remote operations rather than operating on stale local copies.

---

## 3. Mapping GPU Atomics to RDMA Atomics

### 3.1 Direct Mapping (RDMA hardware can do it)

| CUDA Atomic | RDMA Mapping | Notes |
|-------------|-------------|-------|
| `atomicCAS` (64-bit) | `IBV_WR_ATOMIC_CMP_AND_SWP` | Direct 1:1 mapping |
| `atomicAdd` (64-bit int) | `IBV_WR_ATOMIC_FETCH_AND_ADD` | Direct 1:1 mapping |
| `atomicExch` (64-bit) | CAS loop (compare with read value, swap to new) | 1-2 RDMA operations |

### 3.2 Software Emulation via CAS (possible but expensive)

Any atomic can be emulated with a CAS loop:

```
retry:
    old = RDMA_READ(address)
    new = compute(old, operand)  // e.g., min(old, operand)
    result = RDMA_CAS(address, old, new)
    if result != old: goto retry
```

| CUDA Atomic | RDMA Emulation | Operations per Success | Latency |
|-------------|---------------|----------------------|---------|
| `atomicMin/Max` | CAS loop | 1 RDMA_READ + 1+ RDMA_CAS | ~4-20us |
| `atomicAnd/Or/Xor` | CAS loop | 1 RDMA_READ + 1+ RDMA_CAS | ~4-20us |
| `atomicInc/Dec` | CAS loop | 1 RDMA_READ + 1+ RDMA_CAS | ~4-20us |
| `atomicAdd` (32-bit) | CAS loop or split to 64-bit | 1-2 operations | ~2-10us |
| `atomicAdd` (float) | CAS loop with float reinterpret | 1 RDMA_READ + 1+ RDMA_CAS | ~4-20us |
| `atomicSub` | Fetch-and-add with negated value | 1 operation | ~2-5us |

### 3.3 What Cannot Be Directly Mapped

| Problem | Why |
|---------|-----|
| 32-bit atomics | RDMA atomics are 64-bit only. 32-bit ops need padding/alignment tricks or CAS emulation |
| Float atomics | RDMA has no float atomic support. Must use CAS with float-to-int reinterpretation |
| 16-bit atomics (SM 7.0+) | No RDMA equivalent. CAS emulation with masking |
| Contention under CAS loops | With N threads contesting same address, CAS loops degrade to O(N) round trips |
| Shared memory atomics | These are block-local and should never hit the network. Interception must distinguish shared vs global memory targets |

### 3.4 Latency Comparison

| Path | atomicAdd Latency | atomicCAS Latency |
|------|------------------|------------------|
| Same GPU (L2) | ~10-50ns | ~10-50ns |
| NVLink peer | ~100-300ns | ~100-300ns |
| RDMA (direct mapping) | ~2-5us | ~2-5us |
| RDMA (CAS emulation) | ~4-20us (no contention) | ~2-5us |
| TCP (software) | ~50-100us | ~50-100us |

**The RDMA atomic path is 10-100x slower than NVLink.** This is the fundamental constraint. Applications that do occasional synchronization (locks, barriers, counters) will work. Applications that do tight atomic loops (reductions, histograms) will not.

---

## 4. Cache Coherency Over RDMA

### 4.1 R19's I/S/E Coherency Protocol

R19 (Network Page Faults) defines a software coherency protocol based on directory-based tracking:

| State | Meaning | Transitions |
|-------|---------|-------------|
| **I** (Invalid) | Page not present locally | Fetch on access -> S or E |
| **S** (Shared) | Read-only copy, may exist on multiple nodes | Write request -> invalidate others -> E |
| **E** (Exclusive) | Read-write, only this node has it | Remote read request -> downgrade to S, send copy |

This is a page-level (64KB in R10) coherency protocol, NOT cache-line level (128 bytes like NVLink). The granularity difference matters enormously:

| Granularity | Size | Overhead per Transition | False Sharing Risk |
|-------------|------|------------------------|-------------------|
| NVLink cache line | 128 bytes | ~100-300ns | Very low (fine-grained) |
| R19 page | 64 KB | ~2-50us (network transfer) | High (many unrelated accesses per page) |

### 4.2 Extending I/S/E for R18

R18 needs to extend R19's coherency protocol with:

**4.2.1 Atomic-Aware Coherency**

When a remote atomic targets a page in Shared state on multiple nodes, the protocol must:
1. Acquire Exclusive ownership of the page (invalidate all other copies)
2. Perform the atomic locally
3. The page remains Exclusive until another node requests it

This is expensive --- every remote atomic triggers a full page ownership transfer. Optimization: for pages that are atomic-hot (frequently targeted by atomics from multiple nodes), maintain the page at a designated "home node" and route all atomics there.

**4.2.2 Write Combining / Relaxed Coherency**

Not all writes need immediate coherency. For performance, R18 could offer:

| Mode | Coherency | Latency | Use Case |
|------|-----------|---------|----------|
| Strict | Immediate invalidation on write | ~5-50us | Synchronization variables |
| Relaxed | Batch invalidations, eventual consistency | ~100us-1ms | Streaming writes (gradients) |
| Write-through | Writes go to home + local cache | ~2-10us per write | Append-only structures |

### 4.3 Directory-Based Coherency Scaling

The directory tracks, for each page, which nodes have copies and in what state. Scaling concerns:

| Nodes | Directory Entries | Memory Overhead | Protocol Messages per Write |
|-------|------------------|----------------|---------------------------|
| 2 | 1 bit per page per node = 2 bits | Negligible | 1 invalidation |
| 4 | 4 bits per page | Negligible | Up to 3 invalidations |
| 8 | 8 bits per page | ~1 byte per 64KB page | Up to 7 invalidations |
| 16 | 16 bits per page | ~2 bytes per 64KB page | Up to 15 invalidations |
| 32 | 32 bits per page | ~4 bytes per 64KB page | Up to 31 invalidations |

**Storage overhead:** With 64KB pages and a 32-bit sharer vector, the directory overhead is 4 bytes / 65536 bytes = 0.006%. Storage is NOT the scaling bottleneck.

**Message overhead IS the bottleneck:** A write to a widely-shared page requires invalidating N-1 sharers. Each invalidation is a network round-trip (~2-5us). For 8 nodes sharing a page, that is 7 * 2us = 14us minimum before the write can proceed.

**Mitigation strategies:**
1. **Multicast invalidation (R29):** Send one RDMA multicast message instead of N-1 unicast. Reduces latency to ~1 network RTT regardless of sharer count.
2. **Coarsened sharing:** Track sharing at larger granularity (e.g., per-allocation rather than per-page) to reduce directory size.
3. **Limited sharer vector:** Cap at 4-8 sharers. If more nodes want access, fall back to broadcast invalidation.
4. **Home-node protocol:** Each page has a fixed home node. All coherency messages go through home, which serializes operations. Adds one hop but simplifies protocol.

### 4.4 At What Scale Does Coherency Break?

| Nodes | Feasibility | Bottleneck |
|-------|------------|-----------|
| 2 | Solid | None --- simple ping-pong protocol |
| 4 | Good | Manageable invalidation traffic |
| 8 | Possible | Invalidation storms on widely-shared pages |
| 16 | Difficult | Protocol messages dominate latency |
| 32+ | Impractical | Software coherency cannot keep up with hardware access rates |

**Practical limit: 4-8 nodes for fine-grained coherency.** Beyond that, coherency should be coarsened to per-allocation level or replaced with explicit synchronization (barriers + bulk transfers).

This aligns with real NVLink systems: even NVIDIA's GB200 NVL72 (72 GPUs) does not provide fine-grained coherency across all GPUs. Coherency is within NVSwitch domains (typically 8 GPUs), with explicit data movement between domains.

---

## 5. Memory Ordering: NVLink vs RDMA

### 5.1 NVLink Memory Ordering

NVLink provides sequential consistency within a link:
- All operations from GPU A to GPU B appear in program order at GPU B
- atomicAdd followed by store to same link: store sees the add result
- Cross-link ordering requires GPU-side memory fences (`__threadfence_system()`)

### 5.2 RDMA Memory Ordering

RDMA has a more complex ordering model:

| Operation Type | Ordering Within Same QP |
|---------------|------------------------|
| RDMA Write followed by RDMA Write | Ordered (writes complete in post order) |
| RDMA Write followed by Send | Ordered (send completes after writes) |
| RDMA Read followed by anything | NOT ordered without `IBV_SEND_FENCE` |
| RDMA Atomic followed by anything | NOT ordered without `IBV_SEND_FENCE` |

**Key difference from NVLink:**
- RDMA Reads and Atomics can complete out-of-order relative to subsequent operations
- The Fence indicator (`IBV_SEND_FENCE`) forces ordering but adds latency
- Cross-QP ordering is undefined --- no guarantees between different connections
- CPU memory barriers (`mfence`, etc.) do NOT guarantee ordering visible to RDMA hardware

### 5.3 GPU RDMA Memory Ordering Limitation

A critical limitation of current GPUDirect RDMA: memory ordering between a PCIe peer device and GPU kernel threads is only enforced at kernel boundaries. Within a running kernel, there is no mechanism to guarantee that a GPU store is visible to an RDMA peer or vice versa without ending the kernel.

**Impact for R18:** If a kernel writes data and then signals (via atomic) that the data is ready, an RDMA peer may see the signal before the data. This requires careful fencing:
1. GPU kernel writes data
2. GPU kernel issues `__threadfence_system()`
3. GPU kernel writes signal (atomic)
4. RDMA transport sees signal, issues fenced read of data

### 5.4 Mapping NVLink Ordering to RDMA

| NVLink Guarantee | RDMA Implementation |
|-----------------|-------------------|
| Write ordering (same link) | RDMA Writes are naturally ordered within QP |
| Atomic completion before subsequent ops | Use `IBV_SEND_FENCE` on post-atomic operations |
| System-scope visibility | Use completion notification + fence |
| Cross-GPU fence | Network round-trip (explicit fence message) |

**Overhead:** Each fence operation adds ~1 network RTT (~2-5us). NVLink fences are ~10-100ns. Applications that issue many fences (fine-grained synchronization) will see 20-500x slowdown on fence operations.

---

## 6. Implementation Architecture for R18 Atomics

### 6.1 Atomic Interception Strategy

GPU atomics to remote memory must be intercepted at the page fault level (R19). When a GPU thread issues an atomic to a page that is remote:

```
1. GPU thread issues atomicAdd(&remote_addr, val)
2. Page fault: remote_addr is not locally present
3. OuterLink page fault handler determines: this is an atomic operation
4. TWO PATHS:

PATH A: Page Migration (simple, high latency)
   - Fetch page to local GPU in Exclusive state
   - Perform atomic locally
   - Page remains local until someone else needs it
   - Latency: ~10-50us (page fetch) + ~10ns (local atomic) = ~10-50us

PATH B: Remote Atomic Proxy (complex, lower latency for hot pages)
   - Send atomic operation to home node via RDMA
   - Home node executes atomic on local memory
   - Return old value to requesting GPU
   - Latency: ~2-5us (RDMA round-trip) + ~10ns (remote execution) = ~2-5us
```

### 6.2 Decision: When to Use Path A vs Path B

| Criterion | Path A (Migrate) | Path B (Remote Proxy) |
|-----------|------------------|----------------------|
| Page will be accessed again soon | YES | NO |
| Many threads accessing same remote page | YES | MAYBE (contention on proxy) |
| Occasional atomic to remote counter | NO | YES |
| Page shared by many nodes | NO (thrashing) | YES |
| Bandwidth-sensitive (large page) | NO (64KB transfer) | YES (8-byte round-trip) |

A hybrid approach: maintain a "hot atomic" set. Pages that see frequent atomics from multiple nodes are kept at their home node with proxy atomics. Pages with sequential access patterns are migrated normally.

### 6.3 Atomic Proxy Server

Each node runs an atomic proxy service that executes remote atomic requests:

```
Remote GPU -> RDMA Send (atomic_op, address, operand)
           -> Home Node Proxy
           -> Execute atomic on local memory
           -> RDMA Send (old_value)
           -> Remote GPU resumes
```

The proxy must be lock-free for scalability. Since it runs on the CPU (not GPU), it operates on pinned memory that is also mapped into GPU VRAM (via cudaHostRegister or similar).

---

## 7. Summary: Can RDMA Emulate NVLink Features?

| NVLink Feature | RDMA Emulation | Feasibility | Performance Gap |
|---------------|---------------|-------------|----------------|
| Direct load/store | Page fault + migration (R19) | HIGH | 100-1000x latency on cold access |
| atomicCAS (64-bit) | RDMA CAS (direct) | HIGH | 10-50x latency |
| atomicAdd (64-bit int) | RDMA Fetch-Add (direct) | HIGH | 10-50x latency |
| atomicAdd (32-bit, float) | CAS emulation loop | MEDIUM | 20-200x latency |
| atomicMin/Max/And/Or/Xor | CAS emulation loop | MEDIUM | 20-200x latency |
| Cache coherency (line-level) | Software page-level I/S/E | MEDIUM | 1000x+ granularity difference |
| Memory ordering | RDMA fences | HIGH | 20-500x fence latency |
| Unified address space | Virtual address mapping + page faults | HIGH | No overhead once mapped |

**Bottom line:** All NVLink features can be emulated in software over RDMA. Correctness is achievable. Performance will be 10-1000x worse than hardware NVLink depending on the operation. The key is identifying which workloads can tolerate this gap and optimizing for them.

---

## Related Documents

- [01-nvlink-protocol.md](01-nvlink-protocol.md) --- What NVLink provides
- [R19: Network Page Faults](../../phase-08-smart-memory/R19-network-page-faults/README.md) --- I/S/E coherency protocol
- [R7: Non-Proprietary GPU DMA](../../../../research/R7-non-proprietary-gpu-dma.md) --- BAR1 direct access
- [R4: ConnectX-5 Transport](../../../../research/R4-connectx5-transport-stack.md) --- RDMA capabilities
- [R29: RDMA Multicast](../../phase-09-hardening/R29-rdma-multicast/README.md) --- Multicast invalidation

## Sources

1. RDMA Aware Networks Programming User Manual (NVIDIA)
2. DOCA RDMA Programming Guide
3. RDMAmojo --- Fence operations
4. CUDA Programming Guide --- Atomic Functions, Memory Ordering
5. InfiniBand Architecture Specification 1.2.1, Section 10.8.3.3 (Work Request Operation Ordering)
6. CMU 15-418 Directory-Based Cache Coherence Lecture
7. GAM: Efficient Distributed Memory Management with RDMA and Caching (VLDB 2018)
8. NVIDIA Developer Forums --- GPU RDMA Memory Ordering Limitations
9. PCIe Base Specification 2.1+ --- Atomic Operations

## Open Questions

| # | Question | Status |
|---|----------|--------|
| Q1 | Can ConnectX-5 DMA engine issue PCIe atomic TLPs to BAR1 addresses? | OPEN --- would enable hardware remote atomics via OpenDMA |
| Q2 | What is the actual latency of RDMA CAS on ConnectX-5 in our hardware? | OPEN --- need benchmarks |
| Q3 | Does ConnectX-5 support extended (masked) CAS? | OPEN --- check ibv_query_device_ex |
| Q4 | How does GAM's PSO (partial store order) consistency impact CUDA correctness? | OPEN --- CUDA expects stricter ordering |
| Q5 | Can we detect at interception time whether an allocation will be used for atomics vs bulk access? | OPEN --- affects migration vs proxy decision |
| Q6 | What is the practical contention limit for the atomic proxy server? | OPEN --- need benchmarks |
