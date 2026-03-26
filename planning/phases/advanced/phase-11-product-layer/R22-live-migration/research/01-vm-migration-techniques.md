# R22 Research: VM & GPU Live Migration Techniques

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft

## Purpose

Survey the prior art in live migration — both traditional VM migration and emerging GPU-specific approaches — to extract patterns, metrics, and lessons applicable to OuterLink's GPU workload migration.

---

## 1. Traditional VM Live Migration

### 1.1 Pre-Copy Migration (KVM/QEMU, VMware vMotion)

The dominant approach since Clark et al. (2005). The algorithm:

1. **Iterative push phase:** Copy all memory pages to destination while VM continues running on source
2. **Track dirty pages:** After each round, only resend pages that were modified (dirtied) since last transfer
3. **Convergence:** Each round transfers fewer pages as the dirty set shrinks
4. **Stop-and-copy:** When dirty set is small enough (or max rounds reached), pause VM, send final delta, resume on destination

| Parameter | Typical Value | Notes |
|-----------|--------------|-------|
| Total migration time | 10s - 60s for 8-32GB RAM | Depends on dirty rate and bandwidth |
| Downtime | 50-300ms | Final stop-and-copy phase |
| Convergence rounds | 3-8 | Diminishing returns after ~5 |
| Dirty page tracking | Hardware-assisted (EPT/NPT dirty bits) | Near-zero overhead via MMU |

**Key insight for OuterLink:** Pre-copy works well when dirty rate is lower than network bandwidth. For GPU VRAM at 100Gbps, we can transfer ~12.5 GB/s. If a workload dirties less than that per second, pre-copy converges. Most inference workloads dirty very little VRAM after initial load. Training workloads with large batch sizes dirty significant fractions per iteration but have natural pause points between iterations.

**VMware vMotion specifics:**
- Uses a bitmap to track page modifications at 4KB granularity
- Pre-copies pages in order of modification frequency (hot pages last)
- Supports "stun" threshold: if dirty rate exceeds bandwidth, briefly stun the VM to let pre-copy catch up
- Can migrate storage simultaneously (Storage vMotion) — analogous to migrating NVMe tier data

**Hyper-V Live Migration:**
- Similar pre-copy model but adds "smart paging" that compresses pages during transfer
- Supports live migration over SMB3 with RDMA (SMB Direct) — directly relevant to our ConnectX-5 transport
- Tracks dirty pages at 4KB granularity via Second Level Address Translation (SLAT)

### 1.2 Post-Copy Migration

The inverse approach (Hines et al., 2009):

1. **Pause VM** on source, transfer minimal state (CPU registers, device state) to destination
2. **Resume immediately** on destination — most memory pages still reside on source
3. **On page fault:** When destination accesses a page not yet transferred, trap the fault, fetch from source over network, install locally, resume
4. **Background push:** Concurrently push remaining pages from source in the background

| Parameter | Typical Value | Notes |
|-----------|--------------|-------|
| Total migration time | Can be longer (pages fetched on demand) | But workload runs on dest sooner |
| Downtime | Very short (~10-50ms) | Only CPU state + minimal context |
| Page fault penalty | 50-500us per fault (depends on network) | RDMA brings this to ~2-10us |
| Risk | Source failure = data loss | Pages not yet transferred are lost |

**Key insight for OuterLink:** Post-copy maps perfectly to R19's network page fault mechanism. If destination GPU accesses a page still on source, the existing SWMR coherency protocol can serve it. This means post-copy migration is essentially free if R19 is already implemented — we just "move" ownership and let page faults handle the data movement.

**Risk:** If source node fails during post-copy, pages not yet transferred are lost. This is mitigated by R15's erasure coding — parity data on other nodes can reconstruct lost pages.

### 1.3 Hybrid Pre+Post Copy

Combines both approaches to get the benefits of each:

1. **Pre-copy hot pages:** Transfer frequently-accessed pages first (using access frequency tracking from R10's page table)
2. **Switch execution:** Move context to destination
3. **Post-copy cold pages:** Remaining pages fetched on demand via page faults (R19)

This is the most promising approach for OuterLink because:
- R10's page table already tracks access patterns (hot/cold classification)
- R19 provides the page fault mechanism for post-copy
- Pre-copying hot pages minimizes page fault storms after switchover
- Cold pages (often model weights that rarely change) can trickle in the background

### 1.4 Convergence Analysis

Pre-copy converges when: `dirty_rate < transfer_bandwidth`

For OuterLink with ConnectX-5 at 100Gbps (12.5 GB/s effective):

| Workload Type | Estimated VRAM Dirty Rate | Pre-Copy Converges? |
|--------------|--------------------------|-------------------|
| LLM inference (batch) | ~100 MB/s (KV cache updates) | Easily — ~1% of bandwidth |
| Stable Diffusion inference | ~500 MB/s (denoising steps) | Yes — ~4% of bandwidth |
| LLM training (per iteration) | 2-8 GB/s (gradient updates) | Tight — 16-64% of bandwidth |
| Real-time rendering | 5-12 GB/s (framebuffer + textures) | May not converge — post-copy needed |

---

## 2. GPU-Specific Migration

### 2.1 NVIDIA vGPU Live Migration

NVIDIA's enterprise solution for vGPU workloads in virtualized environments:

- **Mechanism:** Hypervisor-level, uses NVIDIA's proprietary VFIO mediated device framework
- **State captured:** VRAM contents, GPU context registers, channel state, page tables
- **Pre-copy:** Iterative dirty-page transfer similar to VM memory migration
- **Dirty tracking:** Hardware-level — the GPU MMU tracks which VRAM pages are modified
- **Pause point:** Waits for all GPU channels to idle (no running kernels)
- **Downtime:** Typically 100-500ms depending on VRAM size and dirty rate
- **Limitations:** Requires enterprise NVIDIA drivers (vGPU license), only works in hypervisor context, closed source

**Relevant takeaway:** Even NVIDIA must wait for kernel completion before the final switchover. There is no way to migrate a running kernel mid-execution. The GPU pipeline (thousands of in-flight warps, register files, shared memory) has no checkpoint mechanism.

### 2.2 Cricket (RWTH Aachen) — Checkpoint/Restart for CUDA

The most relevant open-source prior art. Cricket intercepts Driver API calls and can checkpoint/restore GPU state:

- **State tracked:** All cuMem allocations, module loads, function handles, CUDA contexts
- **Checkpoint:** Pause execution (wait for kernels to finish), dump all allocated VRAM to disk, save context metadata
- **Restore:** Re-create CUDA context on target GPU, reload modules, re-allocate memory at same virtual addresses (if possible), reload VRAM contents
- **Handle translation:** Maps original CUDA handles to new handles on restored context
- **License:** GPLv3 (cannot use code, but can study approach)

**Key lessons from Cricket:**
1. VRAM contents are the bulk of migration data — context metadata is tiny (~KB)
2. Virtual address preservation is important but not always possible — need a handle/pointer translation layer
3. Waiting for kernel completion is the only safe quiesce point
4. Module reload (cuModuleLoad) is fast — PTX/cubin is typically small
5. Stream and event state is reconstructible — just create new ones and remap handles

### 2.3 GPUswap (Univ. of Pittsburgh)

Research system for GPU memory overcommitment:

- **Approach:** Swap GPU memory pages to host memory or disk when GPU is overcommitted
- **Mechanism:** Intercepts CUDA memory allocation, maintains shadow page table, swaps pages at sub-allocation granularity
- **Relevance:** Demonstrates that GPU memory can be managed in pages and swapped transparently — similar to our tier migration in R10
- **Not live migration per se,** but the page-level VRAM management is directly applicable

### 2.4 CRIU + GPU (Checkpoint/Restore In Userspace)

CRIU is the standard Linux process checkpoint tool. GPU support is emerging:

- **NVIDIA's nvidia-persistenced** can preserve GPU state across driver reloads
- **CUDA checkpoint experiments** by various groups show that full CUDA context can be serialized if all kernels are complete
- **Challenge:** CUDA Runtime API maintains internal state that isn't fully accessible via Driver API — another reason OuterLink's Driver API interception is the right choice (we track everything ourselves)

---

## 3. Key Metrics for Migration Quality

| Metric | Definition | Target for OuterLink |
|--------|-----------|---------------------|
| **Total migration time** | Wall clock from start to completion | < 30s for 24GB VRAM |
| **Downtime** | Time workload is paused (no progress) | < 500ms (1 iteration gap) |
| **Application impact** | Throughput degradation during migration | < 20% during pre-copy |
| **Network overhead** | Extra bandwidth consumed by migration | Configurable rate limit |
| **Dirty page convergence** | Rounds needed for dirty set to shrink below threshold | 3-5 rounds typical |
| **Post-migration performance** | Any lingering penalty after migration completes | None (all pages local within 30s) |

---

## 4. Dirty Page Tracking Mechanisms

This is the critical enabler for efficient pre-copy migration.

### 4.1 Hardware-Assisted (Not Available to Us)

- **CPU MMU dirty bits:** Used by VM hypervisors — hardware sets dirty bit on every write. Near-zero overhead.
- **GPU MMU dirty bits:** Used by NVIDIA vGPU — proprietary, not exposed to userspace.
- **Not available to OuterLink:** We don't control the GPU MMU. We need software tracking.

### 4.2 Software Dirty Tracking via R10's Page Table

OuterLink's R10 page table manages VRAM in 64KB pages with PTE flags. We can add migration-specific tracking:

| Approach | Mechanism | Overhead | Accuracy |
|----------|-----------|----------|----------|
| **Write-protect + trap** | Mark pages read-only in our virtual mapping, trap on write, log dirty, unprotect | Per-first-write fault (~1-5us) | Exact |
| **PTE dirty flag** | R10 PTEs already have flags — add a `DIRTY` bit, set on every write through our interception layer | Per-write flag set (~10ns if in interception path) | Exact |
| **Periodic scan** | Snapshot page hashes, compare after interval to find changed pages | Hash compute cost (~1us per 64KB page) | Approximate (misses changes between scans) |
| **Interception-level tracking** | Our cuMemcpy/cuLaunchKernel interception knows which buffers are written — mark those pages dirty | Zero extra overhead (piggyback on existing interception) | Conservative (marks whole buffer, not just changed pages) |

**Recommended approach:** Interception-level tracking for Phase 1 (zero overhead, conservative), PTE dirty flag for Phase 2 (exact, minimal overhead). The interception layer already knows which memory regions are kernel output buffers — marking those dirty is essentially free.

---

## 5. Verdict for OuterLink

### Recommended Strategy: Hybrid Pre+Post Copy

1. **Pre-copy phase** using interception-level dirty tracking (Phase 1) or PTE dirty flags (Phase 2)
2. **Hot-page prioritization** using R10's access frequency data
3. **Quiesce at kernel boundary** — wait for current kernel to complete, hold new launches
4. **Final delta transfer** of remaining dirty pages
5. **Resume on destination** with R19 page fault mechanism as safety net for any missed pages
6. **Background completion** — remaining cold pages trickle via post-copy page faults

This leverages every relevant OuterLink subsystem: R10 (page table + access tracking), R15 (erasure coding for fault tolerance during migration), R19 (page fault mechanism for post-copy).

---

## Related Documents

- [R10: Memory Tiering](../../../phase-07-memory-intelligence/R10-memory-tiering/README.md) — Page table, dirty tracking, tier hierarchy
- [R15: Fault Tolerance](../../../phase-09-hardening/R15-fault-tolerance/README.md) — Erasure coding protects against source failure during post-copy
- [R19: Network Page Faults](../../../phase-08-smart-memory/R19-network-page-faults/README.md) — Page fault mechanism enables post-copy migration
- [R1: Existing Projects](../../../../research/R1-existing-projects.md) — Cricket's checkpoint/restore approach

## Open Questions

- [ ] Can we leverage cuMemMap's virtual address remapping to avoid pointer translation on destination? (Likely yes — cuMemMap separates virtual from physical)
- [ ] What's the actual dirty rate for common ML workloads? Need empirical measurement.
- [ ] Should migration rate-limit be automatic (based on available bandwidth) or user-configured?
- [ ] How does CUDA's internal context state (not tracked by us) affect migration? Need to verify our interception layer captures everything.
