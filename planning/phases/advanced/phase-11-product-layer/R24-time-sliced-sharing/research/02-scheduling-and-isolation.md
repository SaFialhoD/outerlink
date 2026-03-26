# R24 Research: Multi-Tenant Scheduling & Isolation

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Priority:** HIGH

## Purpose

Define how OuterLink time-slices GPU access across multiple users and applications, what isolation guarantees it can enforce at the software level, and what requires kernel-level support.

---

## TL;DR

OuterLink can enforce **VRAM quotas, compute time quotas, and priority scheduling** entirely in software at the CUDA interception layer. Memory isolation requires separate CUDA contexts (Volta+ gives address space isolation via MPS). Preemption of running kernels is NOT possible without kernel driver support, but cooperative scheduling (between kernel launches) is fully achievable. The recommended approach is a **two-level scheduler**: OuterLink's central scheduler assigns GPU slots to users, and per-GPU MPS handles concurrent execution within a slot.

---

## 1. Time-Slice Granularity Options

### Option A: Per-Kernel Scheduling

Each kernel launch is a scheduling decision point. The scheduler decides which user's kernel runs next.

| Pros | Cons |
|---|---|
| Finest granularity, fairest sharing | High scheduling overhead per kernel launch |
| Can implement per-kernel priority | Requires intercepting every `cuLaunchKernel` call |
| Natural fit for OuterLink's interception model | Long-running kernels block other users until completion |
| Can track per-user GPU-time at kernel level | Cannot preempt a running kernel |

**Feasibility for OuterLink:** HIGH. We already intercept `cuLaunchKernel`. Adding a scheduling check before forwarding to the real GPU is straightforward. The limitation is that once a kernel is dispatched, it runs to completion.

### Option B: Fixed Time Quantum (Round-Robin)

Each user gets a fixed time slice (e.g., 10ms, 50ms, 100ms). When the quantum expires, the next user's work runs.

| Pros | Cons |
|---|---|
| Predictable latency bounds | Requires kernel-level preemption (NVIDIA driver feature, not user-controllable) |
| Fair sharing regardless of kernel size | Context switch overhead: 25-50us per switch |
| Similar to CPU scheduling | Resources (shared memory, registers) not transferred between slices |

**Feasibility for OuterLink:** LOW for true preemption. NVIDIA's GPU driver does round-robin time-slicing of CUDA contexts internally, but the time quantum is not user-configurable. OuterLink cannot control this. We CAN approximate it by queuing kernel batches per user and alternating which user's batch we dispatch.

### Option C: Cooperative Scheduling

Users voluntarily yield the GPU at natural breakpoints (between training steps, between inference batches). The scheduler queues work and dispatches cooperatively.

| Pros | Cons |
|---|---|
| Zero preemption overhead | Relies on well-behaved applications |
| Natural for ML workloads (step-based) | One bad actor can hog the GPU |
| Can batch multiple kernels per scheduling slot | Not suitable for untrusted users |
| Simple to implement | No hard latency guarantees |

**Feasibility for OuterLink:** HIGH for trusted environments (e.g., team/lab sharing). OuterLink can insert scheduling barriers between forwarded CUDA operations, effectively creating cooperative yield points.

### Recommended Approach: Hybrid Per-Kernel + Cooperative

1. **Between kernel launches:** OuterLink's scheduler checks if the current user has exceeded their time quantum. If yes, queue their next kernels and dispatch the next user's work.
2. **Between synchronization points:** When a user calls `cuStreamSynchronize` or `cuCtxSynchronize`, this is a natural yield point.
3. **Kernel launch admission control:** Before dispatching a kernel, the scheduler estimates execution time (based on grid size, historical data) and decides whether to admit or queue it.

---

## 2. Memory Isolation

### Separate CUDA Contexts (Baseline)

Each user's work runs in a separate CUDA context on the GPU. This is the default behavior when different processes access the GPU.

| Property | Pre-Volta | Volta+ |
|---|---|---|
| Address space | Shared (no isolation) | Fully isolated |
| Out-of-range writes | Corrupt other processes | Contained to own context |
| Error containment | Fatal fault kills ALL contexts | Contained to fault-causing context |

**Our RTX 3090s (Ampere CC 8.6)** have full Volta+ isolation between CUDA contexts.

### MPS for Concurrent Execution

With MPS, multiple clients share a single hardware context but get separate GPU address spaces (Volta+).

| Property | Without MPS | With MPS |
|---|---|---|
| Kernel execution | Sequential (context switch between users) | Concurrent (Hyper-Q) |
| Context switch cost | 25-50us | None (shared HW context) |
| Memory isolation | Full (separate contexts) | Volta+: full address space isolation |
| Max clients | Unlimited (but serialized) | 48-60 per GPU |

**Recommendation:** Use MPS on each GPU node. OuterLink routes each user's CUDA calls to the MPS server, which provides concurrent execution with address space isolation.

### Address Space Separation Without MPS

If MPS is not used, each user gets a full CUDA context. The GPU time-slices between contexts. Isolation is strong but concurrency is zero (only one context executes at a time).

---

## 3. VRAM Quota Enforcement

OuterLink intercepts ALL memory allocation calls (`cuMemAlloc`, `cuMemAllocManaged`, `cuMemAllocHost`, etc.). This gives us complete control over VRAM usage per user.

### Enforcement Mechanism

```
User A requests cuMemAlloc(1GB):
  1. OuterLink scheduler checks User A's current VRAM usage: 3GB
  2. User A's VRAM quota: 4GB
  3. 3GB + 1GB = 4GB <= 4GB quota -> ALLOW
  4. Forward cuMemAlloc to real GPU
  5. Track allocation in User A's ledger

User B requests cuMemAlloc(2GB):
  1. OuterLink scheduler checks User B's current VRAM usage: 5GB
  2. User B's VRAM quota: 6GB
  3. 5GB + 2GB = 7GB > 6GB quota -> DENY
  4. Return CUDA_ERROR_OUT_OF_MEMORY to User B
```

### What We Track Per User

| Resource | Tracked Via |
|---|---|
| VRAM allocations | Intercept `cuMemAlloc*`, `cuMemFree` |
| Pinned host memory | Intercept `cuMemAllocHost`, `cuMemFreeHost` |
| CUDA contexts | Intercept `cuCtxCreate`, `cuCtxDestroy` |
| Streams | Intercept `cuStreamCreate`, `cuStreamDestroy` |
| Modules/kernels | Intercept `cuModuleLoad`, `cuModuleUnload` |

### Quota Policies

| Policy | Description |
|---|---|
| Hard limit | Allocation fails if quota exceeded |
| Soft limit | Allocation succeeds but triggers eviction/migration from tier |
| Burst | Temporary over-quota allowed if GPU has free VRAM, reclaimed when needed |
| Reserved | Minimum VRAM guaranteed per user, even if others want more |

---

## 4. Priority-Based Scheduling

### Priority Levels

| Priority | Use Case | Scheduling Behavior |
|---|---|---|
| CRITICAL | Production inference serving | Preempts all lower priorities at scheduling points |
| HIGH | Training runs with deadlines | Preempts NORMAL and LOW |
| NORMAL | Standard training/experiments | Fair-share among NORMAL users |
| LOW | Background/batch jobs | Only runs when no higher-priority work pending |
| IDLE | Opportunistic (use spare cycles) | Evicted immediately when any other work arrives |

### Preemption: What OuterLink Can and Cannot Do

**Can do (software-level):**
- Queue a lower-priority user's NEXT kernel launches while dispatching higher-priority work
- Delay forwarding of `cuLaunchKernel` calls from lower-priority users
- Insert synchronization barriers before lower-priority dispatches
- Evict lower-priority user's VRAM allocations to host memory (via R10 memory tiering)

**Cannot do (requires kernel/driver support):**
- Kill a currently-executing GPU kernel mid-flight
- Preempt a running kernel to give the GPU to another user
- Change the GPU's internal time-slice quantum
- Modify MPS's internal scheduling

### Implication

OuterLink's scheduling is **non-preemptive at the kernel level**. A long-running kernel (e.g., 500ms training kernel) will block higher-priority users until it completes. Mitigation: encourage users to break work into smaller kernels, or implement kernel time estimation and refuse to dispatch kernels that would violate latency SLOs.

---

## 5. Fair-Share Algorithms

### Dominant Resource Fairness (DRF)

DRF is the gold standard for multi-resource fair sharing (originally from Apache Mesos). It generalizes max-min fairness to multiple resource types.

**Resources in OuterLink's GPU pool:**
- GPU compute time (SM-seconds)
- VRAM capacity (GB)
- Network bandwidth (Gbps)
- Host memory for staging (GB)

**DRF assigns each user a share proportional to their dominant resource consumption.** If User A is GPU-compute-bound and User B is VRAM-bound, DRF balances so neither dominates.

### Simplified Fair-Share for OuterLink

For initial implementation, a simpler approach:

1. **Weight-based GPU time:** Each user has a weight (default 1.0). GPU time is proportional to weight.
2. **Per-user GPU-second accounting:** Track cumulative GPU-seconds used per user per time window.
3. **Deficit scheduling:** Users who have used less than their fair share get priority for next scheduling slot.
4. **Decay:** Usage history decays over a configurable window (e.g., last 1 hour) to prevent permanent penalization.

### Example: 3 Users, 2 GPUs

| User | Weight | Fair Share | Actual Usage (last hour) | Deficit | Priority |
|---|---|---|---|---|---|
| Alice | 2 | 50% | 30% | +20% | HIGH |
| Bob | 1 | 25% | 40% | -15% | LOW |
| Carol | 1 | 25% | 30% | -5% | MEDIUM |

Alice gets priority because she's used less than her fair share. Bob gets deprioritized because he's over-consumed.

---

## 6. CUDA Context Switching Overhead

### Measured Overhead

| Operation | Cost | Notes |
|---|---|---|
| Full GPU context switch | 25-50us | Register file save + cache flush |
| CUDA thread block launch | ~100s of cycles | Within existing context |
| MPS context (no switch) | 0us overhead | Shared hardware context |
| Kernel dispatch latency | 5-10us | Driver overhead for kernel launch |

### Impact on Scheduling Design

- At 25-50us per context switch and 10ms time slices, overhead is ~0.25-0.5% (acceptable)
- At 1ms time slices, overhead rises to 2.5-5% (still acceptable for sharing)
- Below 1ms time slices, context switching overhead becomes significant

**Recommendation:** Use MPS to eliminate context switch overhead entirely. Multiple users' kernels run concurrently without switching. OuterLink manages the SCHEDULING (which kernels to dispatch), MPS manages the EXECUTION (concurrent kernel running).

---

## 7. What OuterLink Can Enforce vs. What Requires Kernel Support

### OuterLink Can Enforce (Software Layer)

| Capability | Mechanism |
|---|---|
| VRAM quotas | Intercept `cuMemAlloc*`, reject over-quota |
| Kernel admission control | Intercept `cuLaunchKernel`, queue if not user's turn |
| Priority scheduling | Dispatch higher-priority users' work first |
| Fair-share accounting | Track GPU-seconds per user, adjust scheduling |
| Stream isolation | Ensure each user's work goes to separate streams |
| Context lifecycle | Create/destroy CUDA contexts per user session |
| Usage metering | Log all intercepted calls with timestamps |

### Requires Kernel/Driver Support

| Capability | Why Software Can't Do It |
|---|---|
| Kernel preemption | Running kernels cannot be interrupted from userspace |
| SM partitioning | Hardware feature (MIG only, datacenter GPUs) |
| Memory bandwidth isolation | Hardware memory controller, not software-controllable |
| L2 cache partitioning | Hardware feature (Hopper+ MIG only) |
| True compute time guarantees | GPU driver controls actual time-slicing quantum |

### Middle Ground (Requires MPS)

| Capability | How |
|---|---|
| Concurrent kernel execution | MPS enables multiple users' kernels to run simultaneously |
| Per-client resource limits | MPS on Volta+ supports active thread percentage limits |
| Memory isolation | MPS on Volta+ provides separate address spaces |

---

## Recommended Architecture for R24

```
Users/Applications
        |
        v
[OuterLink Central Scheduler]
  - Authenticates users
  - Enforces quotas (VRAM, compute time)
  - Priority scheduling
  - Fair-share accounting
  - Job queue management
        |
        v
[OuterLink Per-GPU Agent]
  - Manages MPS server on local GPU
  - Routes user kernels to MPS clients
  - Tracks per-user resource usage
  - Reports metrics to central scheduler
        |
        v
[NVIDIA MPS Server]
  - Concurrent kernel execution
  - Address space isolation (Volta+)
        |
        v
[Physical GPU]
```

---

## Open Questions

1. **MPS + LD_PRELOAD interaction:** Does our LD_PRELOAD interception work correctly when MPS is active? Need to test.
2. **Kernel time estimation:** How accurately can we estimate kernel execution time from grid dimensions and historical data? This is critical for admission control.
3. **Memory eviction under pressure:** When a high-priority user needs VRAM and a low-priority user holds it, how do we handle eviction? Depends on R10 (memory tiering).
4. **Multi-GPU scheduling:** When a user needs multiple GPUs (distributed training), how does the fair-share algorithm account for gang scheduling?

---

## Related Documents

- [01-gpu-virtualization-landscape.md](01-gpu-virtualization-landscape.md)
- [03-gpu-cloud-architecture.md](03-gpu-cloud-architecture.md)
- [R10: Memory Tiering](../../phase-07-memory-intelligence/R10-memory-tiering/)
- [R17: Topology-Aware Scheduling](../../phase-09-distributed-os/R17-topology-aware-scheduling/)
