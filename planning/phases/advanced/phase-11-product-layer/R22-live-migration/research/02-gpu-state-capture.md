# R22 Research: GPU State Capture for Live Migration

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft

## Purpose

Enumerate every piece of GPU state that must be captured, transferred, and reconstructed on the destination node for a successful live migration. Classify each by size, complexity, and whether OuterLink's interception layer already tracks it.

---

## 1. State Categories Overview

| Category | Typical Size | Tracked by OuterLink? | Migration Difficulty |
|----------|-------------|----------------------|---------------------|
| VRAM contents (allocations) | 1-24+ GB | Yes — R10 page table | High (bulk data) |
| CUDA context metadata | ~KB | Yes — interception layer | Low |
| Loaded modules (PTX/cubin) | ~100KB-10MB | Yes — cuModuleLoad intercepted | Low |
| Function handles | ~KB | Yes — cuModuleGetFunction intercepted | Low |
| Stream state | ~KB | Partially — creation tracked, ordering implicit | Medium |
| Event state | ~KB | Yes — cuEventCreate intercepted | Low |
| cuMemMap virtual mappings | ~KB | Yes — cuMemMap intercepted | Medium |
| In-flight kernel state | N/A | No — cannot capture | **Cannot migrate** |
| CUDA Runtime internal state | ~KB-MB | No — opaque | Medium (avoid via Driver API) |

---

## 2. Detailed State Analysis

### 2.1 VRAM Contents — The Bulk of Migration Data

This is 99%+ of the data that must move. Every cuMemAlloc, cuMemAllocManaged, and cuMemCreate allocation has backing VRAM.

**What OuterLink tracks (via interception layer):**
- Every allocation: base address, size, allocation flags
- R10 page table: 64KB page granularity, tier location, access patterns, PTE flags
- Memory type: device, host, managed, mapped

**What needs to transfer:**
- Raw VRAM contents for every live allocation
- R10 page table entries (metadata, not just data)
- Allocation metadata (so destination can re-create allocations at matching virtual addresses)

**Size estimation:**

| Workload | Typical VRAM Used | Transfer Time at 12.5 GB/s |
|----------|------------------|---------------------------|
| LLM inference (7B params, FP16) | ~14 GB | ~1.1s |
| LLM inference (70B params, quantized) | ~24 GB | ~1.9s |
| Stable Diffusion | ~8 GB | ~0.6s |
| Training (7B, Adam optimizer) | ~24 GB | ~1.9s |
| Light workload | ~2 GB | ~0.16s |

These are raw transfer times. Pre-copy reduces the final delta to a fraction of total VRAM.

**Key optimization — skip read-only data:**
If R12 (Memory Deduplication) is active, model weights shared across GPUs are already present on the destination (or can be referenced from any node). Only mutable state (KV cache, activations, gradients, optimizer state) needs to transfer. For inference workloads, this can reduce migration data by 50-90%.

### 2.2 CUDA Context State

The CUDA context is the top-level container. OuterLink's interception layer creates/manages contexts via cuCtxCreate, cuDevicePrimaryCtxRetain.

**What we track:**
- Context handle → device mapping
- Context flags (scheduling mode, mapping flags)
- Current context per thread

**Migration approach:**
1. Create new context on destination device
2. Map old context handle → new context handle in translation table
3. All subsequent CUDA calls through interception layer use translated handle

**Size:** Negligible (~100 bytes of metadata)

### 2.3 Loaded Modules (PTX/cubin)

CUDA modules contain compiled GPU code. Loaded via cuModuleLoad, cuModuleLoadData, cuModuleLoadFatBinary.

**What we track:**
- Module handle
- Source: file path, embedded data pointer, or fat binary pointer

**Migration approach — two options:**

| Approach | Pros | Cons |
|----------|------|------|
| **Re-load from original source** | No data transfer needed for modules | Requires source path/data still accessible on destination |
| **Transfer module binary** | Destination doesn't need access to original files | Adds ~100KB-10MB to transfer |

**Recommended:** Cache module binaries at load time in our interception layer. During migration, send the cached binary. This decouples migration from filesystem state and handles cuModuleLoadData (where the source is an in-memory pointer that may not persist).

**Size:** Typically 100KB-10MB total across all modules. Negligible compared to VRAM.

### 2.4 Function Handles

Every cuModuleGetFunction returns a CUfunction handle used for kernel launches.

**What we track:**
- Function name (string passed to cuModuleGetFunction)
- Module it belongs to
- Our internal handle → real handle mapping

**Migration approach:**
1. After reloading modules on destination, call cuModuleGetFunction with same names
2. Map old function handle → new function handle
3. All subsequent cuLaunchKernel calls use translated handle

**Size:** Negligible (~bytes per function, typically 10-100 functions)

### 2.5 Stream State

CUDA streams represent ordered execution queues. Created via cuStreamCreate.

**What we track:**
- Stream handle
- Priority (if cuStreamCreateWithPriority)
- Associated context

**What we DON'T explicitly track:**
- Pending work in the stream — but at quiesce point, all work is completed (cuStreamSynchronize)
- Stream ordering relationships (cuStreamWaitEvent) — but these are transient; once the waited event completes, the dependency is gone

**Migration approach:**
1. At quiesce point, all streams are synchronized (empty)
2. Create new streams on destination with same priorities
3. Map old stream handle → new stream handle
4. Post-migration, any cuStreamWaitEvent calls use translated event handles

**Subtlety:** The default stream (stream 0/NULL) is implicit per-context. It doesn't need explicit migration — the new context automatically has one.

**Size:** Negligible metadata

### 2.6 Event State

CUDA events are synchronization points. Created via cuEventCreate.

**What we track:**
- Event handle
- Flags (BLOCKING_SYNC, DISABLE_TIMING)
- Associated context

**Migration approach:**
1. At quiesce point, all events either have been recorded and completed, or were never recorded
2. Create new events on destination with same flags
3. Map old event handle → new event handle
4. Timing information (cuEventElapsedTime) for completed events can be preserved as metadata

**Size:** Negligible

### 2.7 cuMemMap Virtual Address Mappings

R19 and OuterLink's advanced memory management use cuMemMap to create virtual-to-physical mappings. This is the most complex metadata to migrate.

**What we track:**
- cuMemAddressReserve: virtual address range reservations
- cuMemCreate: physical allocation handles
- cuMemMap: virtual → physical mappings
- cuMemSetAccess: access permissions per allocation

**Migration approach:**
1. Reserve same virtual address ranges on destination (cuMemAddressReserve with same base address)
2. Create physical allocations on destination (cuMemCreate)
3. Map virtual → physical (cuMemMap)
4. Set access permissions (cuMemSetAccess)
5. Transfer VRAM contents to the new physical allocations

**Critical requirement:** Virtual addresses must match. Applications hold pointers to these addresses. If we can't reserve the same virtual range on destination (unlikely but possible if destination's address space is fragmented), we need pointer rewriting — a much harder problem.

**Mitigation:** OuterLink controls all memory allocation through interception. We can reserve the needed virtual ranges on destination BEFORE migration starts (proactive reservation during pre-copy phase).

**Size:** Metadata is ~KB. VRAM contents covered in 2.1.

### 2.8 In-Flight Kernel State — CANNOT MIGRATE

A running CUDA kernel has state distributed across:
- Thousands of thread registers (256 registers x 2048 threads x N SMs)
- Shared memory per block (up to 164KB per SM)
- Warp program counters and convergence masks
- L1/L2 cache contents
- Texture/surface cache
- Instruction pipeline state

**None of this is accessible from the host.** The GPU provides no mechanism to:
- Read the register file of a running kernel
- Capture shared memory contents during execution
- Snapshot warp progress
- Serialize the execution pipeline

**This is a hard constraint.** Migration MUST wait for all in-flight kernels to complete. This is the same constraint that NVIDIA vGPU migration, Cricket, and every other GPU migration system operates under.

**Practical impact:** The quiesce point is the boundary between kernel completions. For inference, this is natural — each forward pass completes in 10-100ms. For training, each iteration (forward + backward + optimizer step) completes in 100ms-10s. The migration system waits for this boundary.

### 2.9 Additional State

**Texture/Surface objects:** Created via cuTexObjectCreate. We track creation parameters. Re-create on destination with same parameters, rebind to migrated memory.

**Graphics interop (cuGraphicsResource):** If CUDA is interoperating with OpenGL/Vulkan, migration becomes vastly more complex. OuterLink can initially refuse to migrate workloads with active graphics interop.

**Peer access (cuCtxEnablePeerAccess):** Must be re-established on destination with the destination's peer set. Our cluster topology manager handles this.

**CUDA graphs (cuGraphLaunch):** If R13 (CUDA Graph Interception) is implemented, we have the full graph structure. Graphs must be re-instantiated on destination from our captured structure.

---

## 3. What OuterLink's Interception Layer Already Tracks

Because OuterLink intercepts all CUDA Driver API calls, we have a remarkably complete picture of GPU state. Here's the coverage:

| State Component | Intercepted Call(s) | Tracked? | Notes |
|----------------|-------------------|----------|-------|
| Memory allocations | cuMemAlloc, cuMemAllocManaged, cuMemCreate | **Yes** | Size, flags, address |
| VRAM contents | cuMemcpy*, cuLaunchKernel (outputs) | **Indirectly** | We know what's there, must read to transfer |
| Contexts | cuCtxCreate, cuDevicePrimaryCtxRetain | **Yes** | Handle, flags, device |
| Modules | cuModuleLoad, cuModuleLoadData | **Yes** | Handle, binary data |
| Functions | cuModuleGetFunction | **Yes** | Handle, name, module |
| Streams | cuStreamCreate | **Yes** | Handle, priority, context |
| Events | cuEventCreate, cuEventRecord | **Yes** | Handle, flags, status |
| Virtual mappings | cuMemAddressReserve, cuMemMap, cuMemSetAccess | **Yes** | Full virtual memory map |
| Kernel launches | cuLaunchKernel | **Yes** | Function, grid, block, args, stream |
| Synchronization | cuStreamSynchronize, cuCtxSynchronize | **Yes** | Know when GPU is idle |

**Coverage assessment:** OuterLink tracks ~95% of the state needed for migration through its interception layer. The remaining 5% is either inaccessible (in-flight kernel state) or implicit (CUDA Runtime internal state, which we bypass by using Driver API).

---

## 4. State Capture Sequence

The ordered steps to capture all migratable state:

```
1. QUIESCE
   ├── cuStreamSynchronize on ALL streams
   ├── Verify all kernels completed
   └── Hold all new cuLaunchKernel calls in interception layer

2. CAPTURE METADATA (fast, ~microseconds)
   ├── Serialize context list + flags
   ├── Serialize module list + cached binaries
   ├── Serialize function handle map (name → handle)
   ├── Serialize stream list + priorities
   ├── Serialize event list + flags
   ├── Serialize cuMemMap virtual address mappings
   └── Serialize R10 page table entries

3. CAPTURE VRAM (slow, bulk data)
   ├── For each live allocation:
   │   ├── cuMemcpyDtoH to pinned staging buffer
   │   └── Send over network to destination
   └── (Most of this is already done during pre-copy)

4. FINAL DELTA
   └── Send only pages dirtied since last pre-copy round
```

---

## 5. Size Budget

For a typical 24GB workload:

| Component | Size | Transfer Time @ 12.5 GB/s |
|-----------|------|--------------------------|
| VRAM contents | 24 GB (pre-copy reduces final delta) | ~1.9s full, ~100ms final delta |
| Module binaries | ~5 MB | <1ms |
| All metadata (contexts, streams, events, mappings) | ~100 KB | <0.01ms |
| R10 page table (24GB / 64KB pages = 384K entries) | ~15 MB at ~40 bytes/entry | ~1ms |
| **Total metadata** | **~20 MB** | **~2ms** |

**Conclusion:** Migration is entirely dominated by VRAM data transfer. All metadata combined is <0.1% of the total. The optimization challenge is reducing the VRAM delta at switchover, not the metadata.

---

## Related Documents

- [01-vm-migration-techniques.md](01-vm-migration-techniques.md) — Migration algorithms and dirty tracking
- [R10: Memory Tiering](../../../phase-07-memory-intelligence/R10-memory-tiering/README.md) — Page table structure, PTE flags for dirty tracking
- [R12: Memory Deduplication](../../../phase-07-memory-intelligence/README.md) — Dedup reduces migration data for shared model weights
- [R19: Network Page Faults](../../../phase-08-smart-memory/R19-network-page-faults/README.md) — Post-copy mechanism
- [R1: Existing Projects](../../../../research/R1-existing-projects.md) — Cricket's checkpoint/restore approach

## Open Questions

- [ ] cuMemCreate physical allocation handles — can we guarantee same virtual addresses on destination? Need to test cuMemAddressReserve with specific base address hints.
- [ ] CUDA Runtime internal state (cudaMalloc metadata, internal stream pools) — does our Driver API interception fully bypass this, or are there leaks? Need empirical verification.
- [ ] Texture object state — are the creation parameters sufficient, or do we need to capture internal compiled sampler state?
- [ ] What happens to in-progress cuMemcpyAsync when we quiesce? Does cuStreamSynchronize wait for pending copies? (Yes, it should — verify.)
- [ ] CUDA IPC handles (cuIpcGetMemHandle) — if the workload uses IPC, handles are process-specific and won't survive migration. Need to invalidate and re-issue.
