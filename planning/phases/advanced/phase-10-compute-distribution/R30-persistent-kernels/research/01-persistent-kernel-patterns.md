# R30 Research: Persistent Kernel Patterns

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Priority:** MEDIUM

## Purpose

Document existing persistent kernel designs, CUDA cooperative groups, GPU thread scheduling behavior, occupancy implications, power/thermal concerns, and TDR handling. This provides the foundation for understanding how OuterLink can use persistent kernels to eliminate kernel launch overhead for network-fed GPU workloads.

---

## TL;DR

Persistent kernels are long-running GPU programs that stay resident on the GPU, self-scheduling work from a shared queue instead of exiting and relaunching. They eliminate kernel launch overhead (~5-20us per launch) and enable continuous data processing. CUDA cooperative groups provide grid-wide synchronization (`grid.sync()`) needed for persistent patterns. Key constraints: the grid must fit entirely in the GPU's resident capacity (all blocks active simultaneously), TDR must be disabled on Linux via persistence mode, and spin-waiting threads consume power. For OuterLink, persistent kernels paired with VRAM doorbells create a zero-overhead pipeline: network data arrives via RDMA into VRAM, doorbell wakes the kernel, kernel processes immediately.

---

## What Are Persistent Kernels

A persistent kernel is a GPU kernel that launches once and runs indefinitely (or for a very long time), processing multiple batches of work without returning control to the CPU. Instead of the traditional pattern of:

```
for each batch:
    cudaMemcpy(host_to_device)
    kernel<<<grid, block>>>(data)
    cudaDeviceSynchronize()
```

A persistent kernel does:

```
// Launched once
__global__ void persistent_kernel(volatile int* doorbell, float* data) {
    while (*doorbell != SHUTDOWN) {
        if (*doorbell == DATA_READY) {
            process(data);
            __threadfence_system();
            *doorbell = DONE;
        }
    }
}
```

### Why This Matters for OuterLink

| Traditional Pattern | Persistent Pattern |
|---|---|
| Kernel launch per batch (~5-20us overhead) | Single launch, zero per-batch overhead |
| CPU must orchestrate each batch | GPU self-schedules, CPU optional |
| Data must be staged before launch | Data arrives while kernel is running |
| Cannot overlap network I/O with kernel start | Network writes to VRAM, kernel sees it immediately |

The persistent pattern is essential for OuterLink's vision of network-to-GPU pipelines where RDMA writes data directly into VRAM and the kernel processes it with no CPU involvement.

---

## CUDA Cooperative Groups

Cooperative groups are CUDA's mechanism for synchronization beyond the thread block level. They are the foundation for persistent kernel patterns that need grid-wide coordination.

### Hierarchy

| Group Level | Scope | Sync Mechanism |
|---|---|---|
| `thread_block_tile<N>` | Sub-warp (N threads) | `tile.sync()` |
| `thread_block` | Single block (up to 1024 threads) | `__syncthreads()` |
| `thread_block_cluster` | Multi-block cluster (CC 9.0+) | `cluster.sync()` |
| `grid_group` | Entire grid | `grid.sync()` |
| `multi_grid_group` | Multiple GPUs | `multi_grid.sync()` (deprecated) |

### Grid-Level Synchronization

The key API for persistent kernels is `grid.sync()`, which provides a barrier across ALL thread blocks in the grid. This requires:

1. **Cooperative launch**: Use `cudaLaunchCooperativeKernel()` instead of `kernel<<<grid, block>>>()`
2. **Occupancy-bound grid**: The grid size must not exceed the maximum number of simultaneously resident blocks
3. **Device capability check**: Query `cudaDevAttrCooperativeLaunch` before use

```cpp
// Query max cooperative grid size
int max_blocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_blocks, persistent_kernel, block_size, shared_mem);
int total_blocks = max_blocks * num_SMs;

// Launch cooperatively
void* args[] = {&doorbell, &data};
cudaLaunchCooperativeKernel(
    (void*)persistent_kernel, total_blocks, block_size, args);
```

### Grid Size Constraint

This is the critical constraint: cooperative launches require all blocks to be resident simultaneously. The grid cannot be larger than what the GPU can schedule at once.

For an RTX 3090 (82 SMs):
- With 256 threads/block and moderate register usage: ~16 blocks/SM
- Maximum cooperative grid: 82 x 16 = 1,312 blocks
- Total threads: 1,312 x 256 = ~336K threads

This is sufficient for most persistent kernel workloads, but it means the persistent kernel occupies the ENTIRE GPU. No other kernels can run concurrently.

### Thread Block Clusters (Compute Capability 9.0+)

Hopper (H100) and later introduce an intermediate level: clusters of thread blocks that can synchronize and share memory via distributed shared memory (DSMEM). This provides a middle ground between block-level and grid-level sync. Not available on our RTX 3090 targets (CC 8.6).

---

## Existing Uses of Persistent Kernels

### GPU Databases (HeavyDB/MapD)

GPU databases use persistent kernels to keep query execution resident on the GPU. The kernel loops over incoming query fragments, processing them without relaunching. Benefits: eliminates kernel launch latency for interactive queries, keeps the GPU "warm" with pre-loaded data structures.

### Ray Tracing (Megakernel Pattern)

Ray tracing engines often use a single "megakernel" that contains all shader stages. Persistent threads pull ray work items from a shared queue. This avoids the overhead of launching separate kernels for ray generation, intersection, and shading stages.

### Graph Processing

Graph analytics frameworks (Gunrock, CUDA-based BFS/SSSP) use persistent kernels with work-stealing queues. Thread blocks pull vertex/edge work items and push newly discovered work back to the queue. The kernel runs until the work queue is empty.

### Inference Serving

vLLM and similar LLM serving systems use techniques inspired by persistent kernels:
- Continuous batching at the iteration level
- CUDA graph replay to minimize launch overhead
- PagedAttention manages KV-cache like virtual memory pages
- Tokens from different requests batched in a single kernel invocation

### NVSHMEM with IBGDA

NVIDIA's NVSHMEM library uses persistent kernels combined with GPU-initiated networking (IBGDA transport). The GPU kernel directly writes work queue entries (WQEs) to NIC memory, rings the NIC doorbell, and initiates RDMA transfers without any CPU involvement. This is the closest existing system to what OuterLink aims to achieve.

---

## Thread Block Scheduling and Occupancy

### How the GPU Scheduler Works

The GPU's thread block scheduler assigns blocks to Streaming Multiprocessors (SMs) based on resource availability:

1. **Registers per thread**: Each SM has a fixed register file (e.g., 65536 registers on Ampere SM)
2. **Shared memory per block**: Limited per SM (e.g., 100KB configurable on Ampere)
3. **Thread slots**: Max 2048 threads per SM (64 warps)
4. **Block slots**: Max 16 blocks per SM (Ampere)

The scheduler fills SMs greedily. Once a block completes, its slot opens for the next block. With persistent kernels, blocks never complete, so the scheduler never reclaims slots.

### Occupancy with Persistent Kernels

Persistent kernels lock the GPU at whatever occupancy they launch with. Key implications:

| Aspect | Impact |
|---|---|
| **SM utilization** | 100% of SMs occupied by persistent blocks |
| **Dynamic scheduling** | Disabled — no new blocks can be scheduled |
| **Concurrent kernels** | Impossible while persistent kernel runs |
| **MPS (Multi-Process Service)** | Cannot share GPU with other processes |
| **Memory overhead** | All blocks hold resources for entire duration |

**Occupancy optimization strategy**: Launch with minimum blocks needed, not maximum. If the persistent kernel processes one data buffer at a time, one block per SM may suffice, leaving theoretical room for other work (though cooperative launch still blocks the GPU).

### Non-Cooperative Persistent Kernels

An alternative that avoids the cooperative launch constraint: launch a persistent kernel using standard `<<<grid, block>>>` syntax, but only launch enough blocks to fill the GPU. Use device-side work queues instead of `grid.sync()` for coordination. This approach:
- Allows partial GPU occupation (launch fewer blocks)
- Does not require cooperative launch capability
- Loses grid-wide barrier (must use atomics/flags instead)
- Is the pattern used by GPUrdma and graph processing frameworks

---

## Power and Thermal Considerations

### Spin-Waiting Costs

When a persistent kernel spin-waits on a doorbell flag, the GPU threads are actively executing instructions (load, compare, branch). This is not idle — the SMs consume power even though no useful work is being done.

**Measured impacts from research:**
- Active spin-waiting can consume 60-80% of peak GPU power
- Lower thread block count reduces power proportionally
- Fewer active warps per SM means the warp scheduler has less to do, but ALUs still cycle

### Mitigation Strategies

| Strategy | How It Works | Trade-off |
|---|---|---|
| **Reduced grid size** | Launch fewer blocks, only what's needed | Less parallelism for burst processing |
| **`__nanosleep()`** | CUDA intrinsic to pause thread for N nanoseconds | Adds latency to doorbell detection |
| **Warp-level polling** | Only one warp per block polls, others sleep | Requires intra-block signaling |
| **`cuStreamWaitValue32/64`** | Hardware-level polling, SMs not involved | Not available inside kernels (CPU-side only) |
| **Conditional SM sleep** | Non-target SMs exit immediately, target SMs work | Complex scheduling logic |

### Recommended Approach for OuterLink

Use **warp-level polling**: designate one warp per block (32 threads) as the "doorbell watcher." That warp polls the doorbell flag. When data arrives, it signals the other warps via shared memory flag + `__syncthreads()`. This reduces spin-wait power consumption by ~(block_size - 32) / block_size while keeping doorbell detection latency under 1us.

The `__nanosleep()` intrinsic (available since CUDA 10, CC 7.0+) can further reduce power:

```cpp
// Low-power polling loop
while (atomicAdd(doorbell, 0) == 0) {
    __nanosleep(100);  // 100ns sleep between polls
}
```

This adds ~100ns worst-case latency to doorbell detection but dramatically reduces power consumption during idle periods.

---

## CUDA Timeout Detection and Recovery (TDR)

### The Problem

Operating systems have watchdog timers that kill GPU kernels exceeding a time limit. This is designed to prevent a hung GPU from freezing the display:

- **Windows**: TDR (Timeout Detection and Recovery), default 2 seconds
- **Linux with X11**: Similar timeout for display-connected GPUs
- **Linux headless**: No timeout by default

A persistent kernel that runs indefinitely will trigger TDR unless properly configured.

### Linux Configuration (Our Target)

OuterLink targets Linux servers. The configuration path:

**1. Enable Persistence Mode:**
```bash
sudo nvidia-smi -pm 1
```
Keeps the driver loaded even when no CUDA applications are running. Eliminates driver load/unload overhead between sessions.

**2. Set Compute Mode:**
```bash
sudo nvidia-smi -c EXCLUSIVE_PROCESS
```
Restricts the GPU to one process at a time, which is appropriate for an OuterLink server node.

**3. Headless Operation (Recommended):**
Run without X11/Wayland display server on compute GPUs:
```bash
sudo systemctl stop gdm  # or lightdm
```
With no display manager, there is no watchdog timer for the GPU.

**4. X11 Timeout Override (If Display Needed):**
In `/etc/X11/xorg.conf`, add to the device section:
```
Option "Interactive" "0"
```
This disables the X11 interactive timeout for that GPU.

### Windows Configuration (Reference Only)

For development/testing on Windows:
- TCC mode (Tesla/Quadro only): Disables TDR entirely
- Registry modification: Set `TdrLevel=0` at `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers` (not recommended for production — system instability risk)
- GeForce cards cannot use TCC mode, making long-running kernels problematic on Windows

### OuterLink Recommendation

Deploy OuterLink server nodes in headless Linux mode with persistence mode enabled. This eliminates TDR entirely and provides the most reliable environment for persistent kernels. The CLI tool should verify this configuration during setup:

```bash
# OuterLink server startup check
nvidia-smi --query-gpu=persistence_mode,compute_mode --format=csv
# Expected: Enabled, Exclusive_Process
```

---

## Error Handling for Long-Running Kernels

### CUDA Error Categories

| Error Type | Recoverable? | Impact |
|---|---|---|
| **Non-sticky** (e.g., `cudaErrorNotReady`) | Yes | Cleared by `cudaGetLastError()` |
| **Sticky** (e.g., illegal memory access) | No | Requires `cudaDeviceReset()` |
| **ECC memory error** | Sometimes | May self-correct or require reset |
| **GPU hang** | No | Requires driver reset or reboot |

### Sticky Error Problem

If a persistent kernel hits an illegal memory access or other fatal error, the CUDA context becomes irrecoverable. The only fix is `cudaDeviceReset()`, which destroys all allocations — including RDMA registrations, pinned memory mappings, and VRAM doorbells.

### OuterLink Error Recovery Strategy

1. **Heartbeat mechanism**: The persistent kernel periodically writes a timestamp to a VRAM location. The host-side monitor reads this via mapped memory. If the heartbeat stops, assume kernel failure.

2. **Graceful shutdown path**: The kernel checks a "shutdown" flag each iteration. Host sets this flag to request clean exit before error recovery.

3. **Full context recovery**:
   - `cudaDeviceReset()`
   - Re-initialize CUDA context
   - Re-register RDMA memory regions
   - Re-create doorbell mappings
   - Re-launch persistent kernel
   - Notify connected clients of temporary disruption

4. **Checkpoint/restart (advanced)**: For long-running stateful kernels, periodically checkpoint kernel state (accumulator values, model weights, etc.) to host memory. After recovery, restore from checkpoint.

---

## Key Findings for OuterLink

### Architecture Decision Points

| Decision | Recommendation | Rationale |
|---|---|---|
| Cooperative vs. non-cooperative | Non-cooperative (standard launch) | More flexible, allows partial GPU usage, no cooperative launch requirement |
| Grid size | 1-2 blocks per SM | Minimizes power while providing enough parallelism for burst processing |
| Polling strategy | Warp-level with `__nanosleep()` | Balances power consumption and detection latency |
| TDR handling | Headless Linux + persistence mode | Eliminates timeout entirely |
| Error recovery | Heartbeat + full context reset | Sticky errors leave no other option |
| Grid sync | Atomic flags + `__threadfence_system()` | Works without cooperative launch |

### Performance Expectations

| Metric | Traditional Launch | Persistent Kernel |
|---|---|---|
| Per-batch overhead | 5-20us (kernel launch) | ~0us (already running) |
| First-batch latency | 5-20us | 0.1-1us (doorbell detection) |
| Power during idle | ~0W (kernel not running) | 60-80% peak (full spin), 10-20% (nanosleep) |
| GPU sharing | Possible between launches | Not possible during persistent run |

### Integration with OuterLink Components

- **R13 CUDA Graphs**: A persistent kernel can be a node in a CUDA graph, but this limits the graph to a single invocation. More useful: the persistent kernel replaces graph-based execution for streaming workloads.
- **R26 PTP Clock Sync**: Coordinated doorbell timing across nodes — all persistent kernels begin processing at the same synchronized time.
- **R28 Scatter-Gather DMA**: Scatter-gather writes populate multiple VRAM regions, doorbell rings after all regions are filled.

---

## Open Questions

- [ ] What is the optimal block count per SM for OuterLink's persistent kernel? Needs benchmarking with actual workloads.
- [ ] Can we use `cuStreamWaitValue32` as a doorbell mechanism from the host side to avoid kernel spin-wait entirely? Needs testing with RDMA-written VRAM.
- [ ] How does MIG (Multi-Instance GPU) interact with persistent kernels? Could we isolate a persistent kernel to one MIG partition?
- [ ] What is the actual power draw difference between `__nanosleep(100)` polling and full spin on an RTX 3090?

---

## Related Documents

- [R13: CUDA Graph Interception](../../R13-cuda-graph-interception/research/)
- [R26: PTP Clock Sync](../../R26-ptp-clock-sync/)
- [R28: Scatter-Gather DMA](../../R28-scatter-gather-dma/)
- [CUDA Cooperative Groups Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)
- [GPUrdma: GPU-side RDMA from persistent kernels](https://marksilberstein.com/wp-content/uploads/2020/04/ross16net.pdf)
- [A Study of Persistent Threads Style GPU Programming](https://www.classes.cs.uchicago.edu/archive/2016/winter/32001-1/papers/AStudyofPersistentThreadsStyleGPUProgrammingforGPGPUWorkloads.pdf)
