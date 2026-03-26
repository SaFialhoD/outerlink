# R26 Research: GPU Clock Integration with PTP

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** DRAFT

## Purpose

Determine how to bridge the gap between PTP-synchronized system clocks and GPU-internal timing. Cover CUDA event timestamps, GPU clock domains, correlation techniques, and practical strategies for coordinated kernel launches across PTP-synced nodes.

---

## 1. The Clock Domain Problem

There are three distinct clock domains in a GPU compute node:

```
┌─────────────────────────────────────────────┐
│  Clock Domain 1: NIC PHC (PTP Hardware Clock)│
│  - Free-running at NIC core frequency        │
│  - Synced to grandmaster via PTP             │
│  - Accessed via /dev/ptpN                    │
└──────────────────────┬──────────────────────┘
                       │ phc2sys
┌──────────────────────▼──────────────────────┐
│  Clock Domain 2: System Clock (CLOCK_REALTIME)│
│  - Kernel-maintained wall clock              │
│  - Disciplined by phc2sys to follow PHC      │
│  - Accessible from both host and device code │
│  - Typical offset from PHC: <100ns           │
└──────────────────────┬──────────────────────┘
                       │ (no automatic sync)
┌──────────────────────▼──────────────────────┐
│  Clock Domain 3: GPU Internal Clock          │
│  - %globaltimer (PTX register)               │
│  - Initialized from host system clock at     │
│    device attach (or CUDA persistence mode)  │
│  - Free-runs independently after init        │
│  - Frequency matches real-time (not GPU core)│
│  - NOT updated when host clock changes       │
└─────────────────────────────────────────────┘
```

**The gap:** PTP synchronizes Domain 1 (PHC) to sub-100ns. phc2sys propagates to Domain 2 (system clock) with <100ns additional error. But Domain 3 (GPU timer) is initialized once and then drifts independently. The GPU timer cannot be adjusted after initialization.

---

## 2. CUDA Event Timestamps

### How CUDA Events Work

```c
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
// ... kernel launch ...
cudaEventRecord(stop, stream);

cudaEventSynchronize(stop);
float milliseconds;
cudaEventElapsedTime(&milliseconds, start, stop);
```

### Key Properties

| Property | Value |
|----------|-------|
| Resolution | ~0.5 microseconds |
| Clock source | GPU internal timer (not system clock) |
| Cross-GPU comparison | **Not valid** — each GPU has independent clock |
| Cross-node comparison | **Not valid** — different clock domains |
| Precision for single-GPU timing | Excellent (~0.5us) |

**Critical limitation:** `cudaEventElapsedTime` only measures duration between two events on the **same GPU**. It cannot be used to correlate timing across GPUs or across nodes. For distributed timing, we need a different approach.

### cuEventRecord Precision

`cudaEventRecord` (or `cuEventRecord` at driver API level) inserts a timestamp marker into a CUDA stream. The GPU records the timestamp when the stream execution reaches that point. The timestamp uses the GPU's internal clock, which has ~0.5us resolution.

---

## 3. GPU Global Timer (%globaltimer)

### PTX %globaltimer Register

NVIDIA GPUs expose a `%globaltimer` register in PTX (parallel thread execution) ISA:

- **Initialized from host system clock** at device attach time
- **Counts real-world time** (not GPU core clock cycles)
- **Same tick rate as host system clock** (nanosecond resolution)
- **Monotonically increasing** — never goes backwards
- **Cannot be updated** — once initialized, it free-runs

### Implications for PTP Correlation

Since `%globaltimer` is initialized from the host system clock:

1. **At initialization time**, the GPU timer matches the system clock (within a few microseconds)
2. **After initialization**, the GPU timer drifts independently
3. **phc2sys adjustments to the system clock are NOT reflected in the GPU timer**
4. **PTP time corrections are invisible to the GPU**

This means: the GPU timer will slowly drift from PTP time. How fast?

### GPU Timer Drift Rate

GPU crystal oscillators are similar in quality to NIC oscillators:
- Typical drift: 1-10 ppm (parts per million)
- At 10 ppm: 10 microseconds per second, 600 microseconds per minute, 36 milliseconds per hour

For sub-microsecond coordinated launches, this drift is unacceptable for workloads longer than ~100 milliseconds without correction.

---

## 4. Correlation Strategies

### Strategy A: Host-Side Scheduling with PTP Time (Recommended)

Don't try to use GPU clocks for cross-node coordination. Instead:

```
                    PTP-synced system clock
                           │
                    ┌──────▼──────┐
                    │  OuterLink   │
                    │  Coordinator │
                    └──────┬──────┘
                           │
          "Launch kernel at PTP time T+5ms"
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         ┌────────┐  ┌────────┐  ┌────────┐
         │ Node A │  │ Node B │  │ Node C │
         └───┬────┘  └───┬────┘  └───┬────┘
             │            │            │
    wait until            wait until   wait until
    clock_gettime         clock_gettime clock_gettime
    == T+5ms              == T+5ms     == T+5ms
             │            │            │
    cudaLaunchKernel      cudaLaunch   cudaLaunch
```

**How it works:**
1. Coordinator announces: "All nodes launch kernel at PTP time T+5ms"
2. Each node spins on `clock_gettime(CLOCK_REALTIME)` until the target time
3. At target time, each node calls `cudaLaunchKernel`
4. Since all system clocks are PTP-synced to <100ns, launches are coordinated

**Launch jitter budget:**
- PTP sync error: ~50-100ns
- `clock_gettime` call overhead: ~20-50ns
- Kernel launch latency variation: ~5-20 microseconds (this is the bottleneck!)

**Key insight:** The CUDA kernel launch latency (~5-20us) dominates over PTP sync error (~100ns). PTP gives us synchronized trigger times, but the GPU's kernel launch pipeline adds its own jitter. This is still far better than uncoordinated launches.

### Strategy B: GPU-Side Spin-Wait on Global Timer

For tighter coordination, we can use the GPU's `%globaltimer`:

```cuda
__global__ void coordinated_kernel(uint64_t target_ns, ...) {
    // Spin until GPU global timer reaches target time
    uint64_t now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    while (now < target_ns) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    }

    // All threads across all GPUs start actual work here
    // (if global timers are correlated to PTP)
    do_actual_work(...);
}
```

**Problem:** This requires the GPU global timers to be correlated with PTP time. Since they're initialized from the system clock at attach time and then drift, we need periodic re-calibration.

### Strategy C: Periodic GPU-to-PTP Calibration (Best Accuracy)

Periodically measure the offset between GPU `%globaltimer` and PTP-synced system clock:

```
Every N seconds:
  1. Read system clock (PTP-synced): T_host
  2. Read GPU global timer:          T_gpu
  3. Compute offset:                 delta = T_host - T_gpu
  4. Store delta for this GPU

To convert GPU time to PTP time:
  PTP_time = GPU_time + delta

To set a GPU-side target time:
  GPU_target = PTP_target - delta
```

**Calibration accuracy depends on:**
- `clock_gettime` call latency: ~20-50ns
- PCIe round-trip for GPU timer read: ~1-2 microseconds
- Variability of PCIe latency: ~100-500ns

**Improvement: Bracketing technique:**

```rust
let t1_host = clock_gettime(CLOCK_REALTIME);
let t_gpu = read_gpu_globaltimer();  // via CUPTI or custom kernel
let t2_host = clock_gettime(CLOCK_REALTIME);

let host_midpoint = (t1_host + t2_host) / 2;
let pcie_rtt = t2_host - t1_host;  // should be ~1-2us

if pcie_rtt < threshold {
    // Good sample — PCIe wasn't delayed
    delta = host_midpoint - t_gpu;
    // Accuracy: ~pcie_rtt/2, typically ~0.5-1us
}
```

By taking multiple samples and filtering for lowest PCIe round-trip, we can get GPU-to-PTP correlation within ~500ns.

---

## 5. CUPTI for GPU Timestamps

### cuptiGetTimestamp

CUPTI provides `cuptiGetTimestamp()` which returns a normalized timestamp usable for correlating CPU and GPU activity:

| API | Clock Source | Resolution | Use Case |
|-----|-------------|-----------|----------|
| `cuptiGetTimestamp` | Normalized to activity records | ~100ns granularity | Profiling correlation |
| `cuptiDeviceGetTimestamp` | GPU device timer | Nanosecond | Device-specific timing |
| `cudaEventElapsedTime` | GPU internal clock | ~0.5us | Single-GPU duration |
| `clock_gettime(CLOCK_REALTIME)` | System clock (PTP-synced) | ~1ns resolution | Cross-node correlation |

**For OuterLink distributed profiling:** Use `clock_gettime(CLOCK_REALTIME)` on the host side (PTP-synced) to timestamp kernel launches and completions. Use CUDA events for on-GPU durations. Correlate using the Strategy C calibration offset.

### CUPTI Correlation IDs

CUPTI assigns correlation IDs that link CUDA API calls to GPU activity records. This is more reliable than timestamp matching for tracing which host call produced which GPU work:

```
Host: cuLaunchKernel(correlation_id=42) at PTP_time T1
GPU:  kernel started (correlation_id=42) at GPU_time G1
GPU:  kernel finished (correlation_id=42) at GPU_time G2
Host: cuStreamSynchronize returns at PTP_time T2

Duration on GPU: G2 - G1 (via CUDA events)
Total latency:   T2 - T1 (via PTP-synced system clock)
Launch overhead:  T2 - T1 - (G2 - G1)
```

---

## 6. Coordinated Kernel Launch Strategies

### Strategy 1: Future-Time Scheduling (Primary Approach)

```
Coordinator -> All nodes: "Launch kernel K at PTP time T+10ms"
Each node:
  1. Prepare kernel arguments
  2. Spin on clock_gettime until T+10ms
  3. cudaLaunchKernel(K)
```

**Jitter: ~5-20us** (dominated by cudaLaunchKernel variability)

### Strategy 2: Barrier-Then-Launch

```
Coordinator -> All nodes: "Prepare kernel K, wait for GO"
Each node:
  1. Prepare kernel, record readiness
  2. Wait for coordinator "GO" message (RDMA send)
Coordinator: Once all ready, send "GO" at PTP time T
Each node:
  3. Receive GO (RDMA latency ~2us)
  4. cudaLaunchKernel(K)
```

**Jitter: ~7-25us** (RDMA message latency + launch latency)

### Strategy 3: GPU-Side Spin Barrier (Tightest Coordination)

```
All nodes: Launch coordinated_kernel with target_time = PTP_T + delta_correction
GPU: Spin on %globaltimer until target_time, then execute
```

**Jitter: ~0.5-2us** (GPU timer read granularity + calibration error)

This is the tightest coordination possible but wastes GPU cycles spinning. Best for short synchronization points within long-running kernels.

### Strategy 4: Hybrid — Host Schedule + GPU Spin

```
Host: Launch kernel at PTP time T (Strategy 1, ~10us jitter)
GPU:  Spin on %globaltimer until T + safe_margin (Strategy 3, ~1us jitter)
```

The host gets the kernel "close" (within 10us), then the GPU spin-waits for fine alignment. This combines the reliability of host scheduling with the precision of GPU-side synchronization.

**This is likely the best approach for R25 (Cooperative Kernel Splitting).**

---

## 7. How Data Centers Handle This

### Current Practice

Most distributed GPU training does **not** use clock synchronization for kernel coordination. Instead:

- **NCCL All-Reduce** acts as an implicit barrier — all GPUs must contribute before any can proceed
- **cudaStreamSynchronize** on the host provides inter-iteration barriers
- **No attempt at simultaneous kernel launch** — the collective operations provide synchronization naturally

### Why PTP Matters Anyway

Even though NCCL/collective ops handle synchronization for standard training, PTP-synced clocks enable:

1. **Accurate distributed profiling**: Know exactly when each GPU started and finished, across all nodes, on a common timeline
2. **Pipeline parallelism optimization**: Different stages of a pipeline should start at precise offsets
3. **R25 (Cooperative Kernel Splitting)**: Splitting one kernel across GPUs requires threads to start simultaneously
4. **R30 (Persistent Kernels)**: Network-fed persistent kernels need coordinated timing for data handoff
5. **Latency measurement**: True one-way latency measurement (not just round-trip / 2)

### NVIDIA Firefly (DPU-Based)

NVIDIA's DOCA Firefly service on BlueField DPUs provides data-center-scale clock synchronization. It leverages BlueField's hardware PTP with UTC-format real-time clock. This is the "enterprise" version of what we're building with ConnectX-5 PTP.

---

## 8. Distributed Profiling with PTP Timestamps

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                  PTP Timeline (common)               │
│  T0────T1────T2────T3────T4────T5────T6────T7───>   │
└─────────────────────────────────────────────────────┘

Node A:  [alloc]──[H2D transfer]──[kernel A1]──[D2H]──
Node B:       [alloc]──[H2D]──[kernel B1]──────[D2H]──
Node C:            [alloc]──[H2D]──[kernel C1]──[D2H]─

With PTP, we can draw this timeline accurately.
Without PTP, we can only measure durations per-node.
```

### Implementation

```rust
struct DistributedEvent {
    node_id: NodeId,
    gpu_id: GpuId,
    event_type: EventType,
    ptp_timestamp_ns: u64,    // clock_gettime(CLOCK_REALTIME)
    gpu_timestamp_ns: u64,    // CUDA event or CUPTI
    correlation_id: u64,      // Links host call to GPU work
}
```

Each node records events with PTP-synced system timestamps. The coordinator collects all events and can reconstruct a global timeline with <1us accuracy.

---

## 9. Drift Compensation for Long-Running Workloads

### The Problem

GPU `%globaltimer` drifts from PTP-synced system clock at 1-10 ppm:
- After 1 second: 1-10us drift
- After 1 minute: 60-600us drift
- After 1 hour: 3.6-36ms drift

### Compensation Approach

1. **Periodic re-calibration**: Every 1-10 seconds, re-measure GPU-to-PTP offset using Strategy C (bracketing technique)
2. **Linear drift model**: Between calibration points, model drift as linear: `corrected_time = gpu_time + offset + drift_rate * (gpu_time - last_calibration_time)`
3. **No GPU clock adjustment needed**: We never change the GPU clock. We just maintain a correction table on the host side.

### Calibration Frequency vs Accuracy

| Calibration Interval | Expected Drift Between Calibrations | Achievable Accuracy |
|---------------------|-------------------------------------|---------------------|
| 100ms | 0.1-1us | ~1.5us (calibration error + drift) |
| 1 second | 1-10us | ~1us (calibration dominates) |
| 10 seconds | 10-100us | ~1us (with linear drift model) |
| 1 minute | 60-600us | ~5us (drift model less accurate) |

**Recommendation:** Calibrate every 1-10 seconds. The calibration overhead is negligible (one host-GPU round-trip).

---

## 10. Key Takeaways for OuterLink

1. **GPU clocks are a separate domain** — PTP syncs NIC and system clocks, but GPU timers drift independently
2. **Host-side PTP scheduling is the primary approach** — use `clock_gettime(CLOCK_REALTIME)` for coordinated launches
3. **GPU-side spin-wait provides tightest coordination** — ~1us jitter, but needs periodic calibration
4. **Hybrid approach is best** — host schedules launch, GPU spin-waits for fine alignment
5. **CUDA events are single-GPU only** — use PTP-synced system timestamps for cross-node correlation
6. **Periodic calibration handles drift** — re-measure GPU-to-PTP offset every 1-10 seconds
7. **Kernel launch jitter (~5-20us) is the real bottleneck** — PTP sync error (~100ns) is negligible by comparison
8. **CUPTI correlation IDs** are more reliable than timestamp matching for linking host calls to GPU work

---

## Related Documents

- [01-ptp-protocol-and-hardware.md](01-ptp-protocol-and-hardware.md) — PTP protocol and ConnectX-5 support
- [02-linux-ptp-stack.md](02-linux-ptp-stack.md) — linuxptp configuration
- [R25: Cooperative Kernel Splitting](../../phase-12-moonshot/R25-cooperative-kernel-splitting/README.md) — Primary consumer of coordinated launches
- [R30: Persistent Kernels](../../phase-10-compute-distribution/R30-persistent-kernels/README.md) — Timing coordination for network-fed kernels

## Open Questions

- [ ] What's the actual kernel launch jitter on our 3090s? (Need to measure with CUDA events)
- [ ] Can we reduce launch jitter via CUDA graphs? (Pre-recorded launch sequences should have lower jitter)
- [ ] Is %globaltimer accessible from Rust via inline PTX in cubin? (Need to test)
- [ ] What's the PCIe round-trip latency for GPU timer reads on our hardware? (Determines calibration accuracy)
- [ ] Can CUDA MPS (Multi-Process Service) affect launch coordination? (If multiple processes share a GPU)
- [ ] Does nvidia-persistenced affect %globaltimer initialization? (Persistence mode keeps GPU attached)
