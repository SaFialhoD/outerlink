# R30: Persistent Kernels with Network Feed — Pre-Plan

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft
**Priority:** MEDIUM

## Purpose

Define WHAT needs to be planned for implementing persistent kernels with VRAM-based doorbell notification and network-fed continuous execution in OuterLink, before writing the detailed implementation plan.

---

## Scope Definition

### What R30 Covers

1. **Persistent kernel framework** — a reusable kernel template that runs indefinitely, polling a VRAM doorbell for work
2. **VRAM ring buffer system** — input and output ring buffers in GPU VRAM with head/tail pointers, batch descriptors, and data slot pools
3. **Doorbell mechanism** — atomic counter + ring buffer descriptor pattern for NIC-to-GPU notification
4. **OpenDMA integration** — NIC writes data and doorbell directly to VRAM via PCIe BAR1 (connects to Phase 5 OpenDMA)
5. **Double/triple buffering** — overlapping ingest, process, and emit stages
6. **Error handling and recovery** — heartbeat monitoring, graceful shutdown, full context reset on fatal errors
7. **Host-side management** — kernel lifecycle (launch, monitor, shutdown, restart), configuration, statistics collection
8. **Server integration** — connecting the persistent kernel pipeline to OuterLink's server daemon and client protocol

### What R30 Does NOT Cover (Adjacent Topics)

| Topic | Covered By | R30 Interaction |
|-------|-----------|-----------------|
| CUDA graph capture/replay | R13 (CUDA Graph Interception) | R30 is an alternative execution model; apps choose graph mode or persistent mode |
| PTP time synchronization | R26 (PTP Clock Sync) | R30 uses R26's synchronized clocks for coordinated doorbell timing across nodes |
| Scatter-gather DMA | R28 (Scatter-Gather DMA) | R30 uses R28 for multi-region data ingest into ring buffer slots |
| Kernel argument introspection | R8 (Kernel Param Introspection) | R30 persistent kernels have fixed arguments (ring pointers); R8 not needed |
| Transport layer setup | P6 (Core Transport) | R30 assumes transport is established; uses RDMA writes provided by transport |
| OpenDMA kernel module | P5 (OpenDMA) | R30 depends on P5 for BAR1 VRAM access from NIC |
| GPU memory management | R10 (Memory Hierarchy) | R30 allocates fixed VRAM pools; R10 manages the broader memory space |

---

## Key Technical Unknowns

### Unknown 1: GPU Cache Coherency with NIC VRAM Writes

**Risk:** HIGH
**Question:** When the NIC writes to VRAM via BAR1, does the GPU persistent kernel reliably see the new data? Under what conditions might it see stale cached values?
**Research finding:** NVIDIA docs confirm that a concurrently running kernel may observe stale data from PCIe BAR1 writes. `volatile` reads bypass L1 but not necessarily L2. `__threadfence_system()` after doorbell detection should ensure visibility of preceding data writes.
**Required before planning:** Prototype test — NIC writes to VRAM via BAR1, persistent kernel polls and reads. Measure correctness and latency.
**Proposed resolution:** Use `volatile` doorbell + `__threadfence_system()` after detection. If L2 stale data persists, fall back to `atomicAdd(addr, 0)` reads which force uncached loads.

### Unknown 2: RDMA Atomic Operations to VRAM via BAR1

**Risk:** MEDIUM
**Question:** Can ConnectX-5 perform RDMA atomic operations (e.g., `atomicAdd`) targeting GPU VRAM mapped through BAR1? Standard RDMA atomics work on host memory, but VRAM-mapped regions via BAR1 are non-standard.
**Research finding:** GPUrdma and NVSHMEM IBGDA both use NIC-to-VRAM atomics, but they rely on GPUDirect RDMA registration (which requires Tesla/A100 hardware). OpenDMA's BAR1 path may not support NIC-side atomics.
**Required before planning:** Test RDMA atomic fetch-and-add targeting a BAR1-mapped VRAM address. If unsupported, fall back to regular RDMA write for doorbell (write new head value instead of atomic increment).
**Proposed resolution:** Primary path uses RDMA write (not atomic) for doorbell — write the new head value directly. This is simpler and guaranteed to work over BAR1. Atomic path as optimization if supported.

### Unknown 3: Persistent Kernel Power Consumption

**Risk:** LOW-MEDIUM
**Question:** How much power does a persistent kernel consume while idle (spin-waiting on doorbell)? Is it acceptable for always-on server deployment?
**Research finding:** Full spin-wait can consume 60-80% of peak GPU power. `__nanosleep(100)` reduces this significantly but needs measurement on RTX 3090.
**Required before planning:** Benchmark idle power with different polling strategies (full spin, nanosleep 100ns, nanosleep 1us, warp-level polling).
**Proposed resolution:** Default to `__nanosleep(100)` with configurable interval. Document power vs. latency trade-off for operators.

### Unknown 4: TDR on Development Machines

**Risk:** LOW
**Question:** How to develop and test persistent kernels on Windows workstations where TDR is active and cannot use TCC mode (GeForce)?
**Research finding:** Windows TDR kills kernels after ~2 seconds. Registry modification (`TdrLevel=0`) disables this but risks system instability. Linux headless mode has no timeout.
**Required before planning:** None — development can use short-lived "persistent" kernels (run for N iterations then exit) on Windows, full persistent on Linux CI.
**Proposed resolution:** Build configuration flag: `OUTERLINK_PERSISTENT_MAX_ITERATIONS` — defaults to infinite on Linux, configurable on Windows for testing.

### Unknown 5: Multiple Persistent Kernels on One GPU

**Risk:** MEDIUM
**Question:** Can we run multiple independent persistent kernels on one GPU (e.g., one per client connection)? Or must we use a single kernel that multiplexes all connections?
**Research finding:** Cooperative launch occupies the entire GPU. Non-cooperative persistent kernels can coexist if total block count fits in GPU capacity. However, resource fragmentation and scheduling fairness are concerns.
**Required before planning:** Test launching 2-4 non-cooperative persistent kernels with reduced block counts. Measure whether they all make progress (no starvation).
**Proposed resolution:** Start with single-kernel-per-GPU design that multiplexes connections via tagged ring entries. Multi-kernel as future optimization.

---

## Dependencies

### Hard Dependencies (Must Exist Before R30)

| Dependency | Status | What R30 Needs |
|---|---|---|
| **P5: OpenDMA** | Not started | BAR1 VRAM access for NIC-to-VRAM doorbell writes |
| **P6: Core Transport** | Not started | RDMA connection setup, memory registration |
| **Basic CUDA context** | Available | `cudaMalloc`, `cudaLaunchKernel`, persistence mode |

### Soft Dependencies (Can Develop in Parallel)

| Dependency | Status | What R30 Gets |
|---|---|---|
| R13: CUDA Graph Interception | Research complete | Complementary execution model — not required |
| R26: PTP Clock Sync | Research complete | Coordinated timing for multi-node pipelines |
| R28: Scatter-Gather DMA | Research complete | Efficient multi-region data ingest |

### What Depends on R30

| Dependent | What It Needs from R30 |
|---|---|
| Inference serving workloads | Continuous batching pipeline |
| Stream processing workloads | Low-latency data processing pipeline |
| Multi-node GPU pipelines | Persistent kernel + RDMA ring buffer chaining |

---

## Implementation Phases (Proposed)

### Phase A: Standalone Persistent Kernel (No Network)

**Goal:** Validate persistent kernel pattern works correctly on target hardware.

**Deliverables:**
1. Persistent kernel template (C/CUDA) with doorbell polling and nanosleep
2. VRAM ring buffer allocation and management (host-side Rust)
3. Host-side doorbell writer (simulates NIC writes via `cudaMemcpy` or mapped pointer)
4. Heartbeat monitor and graceful shutdown
5. Benchmark: launch overhead comparison (persistent vs. repeated launch)
6. Power consumption measurement with different polling strategies

**Acceptance criteria:**
- Persistent kernel runs for >1 hour without TDR or errors on headless Linux
- Doorbell latency <1us (host write to kernel detection)
- Graceful shutdown completes within 100ms
- Heartbeat detects simulated hang within 200ms

### Phase B: OpenDMA-Fed Persistent Kernel

**Goal:** NIC writes data and doorbell directly to VRAM, persistent kernel processes.

**Deliverables:**
1. BAR1 memory registration for ring buffer and data pool
2. RDMA write path: remote node writes data + descriptor + head to VRAM
3. Cache coherency validation (NIC writes visible to persistent kernel)
4. End-to-end latency benchmark: remote write to kernel processing start
5. Throughput benchmark: sustained data rate through the pipeline

**Acceptance criteria:**
- NIC-to-kernel doorbell latency <2us (BAR1 write to kernel detection)
- Data integrity verified (CRC or checksum per batch)
- Sustained throughput matches network bandwidth (approaching 100Gbps for large batches)
- No stale data observed over 1M+ doorbell cycles

### Phase C: Full Pipeline Integration

**Goal:** Connect persistent kernel pipeline to OuterLink server daemon.

**Deliverables:**
1. Server-side persistent kernel manager (launch, monitor, restart)
2. Client protocol extension: "persistent stream" mode
3. Output path: result ring buffer with host-side or RDMA emission
4. Multi-connection multiplexing in single persistent kernel
5. Error recovery with automatic kernel restart
6. Configuration API for ring sizes, poll intervals, buffer counts

**Acceptance criteria:**
- Multiple clients stream data through a single persistent kernel simultaneously
- Kernel crash recovery completes in <500ms, clients reconnect automatically
- Configuration changes take effect without full restart
- Integration tests pass with simulated network failures

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| BAR1 cache coherency issues | Medium | High | Test early (Phase B). Fall back to host-staged doorbell if VRAM doorbell unreliable |
| GPU hangs from persistent kernel bugs | Medium | High | Heartbeat + automatic recovery. Extensive testing in Phase A |
| Power consumption unacceptable | Low | Medium | Configurable nanosleep interval, warp-level polling, document trade-offs |
| RDMA atomics unsupported via BAR1 | Medium | Low | Use RDMA write (not atomic) for doorbell — simpler, always works |
| Single kernel per GPU limits flexibility | Low | Medium | Multiplexing via tagged ring entries handles most multi-client scenarios |

---

## Resource Estimates

| Phase | Engineering Time | Hardware Needed |
|---|---|---|
| Phase A | 2-3 weeks | Single GPU (RTX 3090), Linux |
| Phase B | 3-4 weeks | Two PCs with RTX 3090 + ConnectX-5, Linux |
| Phase C | 3-4 weeks | Same as Phase B + OuterLink server running |
| **Total** | **8-11 weeks** | |

---

## Open Questions for Planning

- [ ] Should the persistent kernel be written in pure CUDA C, or use a Rust wrapper with `cuda-sys` bindings for the host-side management?
- [ ] What is the minimum viable ring buffer size for a useful demo? (4 entries? 16?)
- [ ] Should we support heterogeneous data slot sizes (different batches have different sizes) or fixed-size slots?
- [ ] How does this interact with OuterLink's memory allocation tracking? The ring buffers are pre-allocated, not managed by the intercepted `cudaMalloc` path.
- [ ] Priority question: should R30 Phase A begin before or after P5 (OpenDMA) has basic BAR1 access working? Phase A is standalone, but Phase B requires P5.

---

## Related Documents

- [research/01-persistent-kernel-patterns.md](./research/01-persistent-kernel-patterns.md)
- [research/02-doorbell-mechanisms.md](./research/02-doorbell-mechanisms.md)
- [research/03-network-fed-execution.md](./research/03-network-fed-execution.md)
- [R13: CUDA Graph Interception Pre-Plan](../R13-cuda-graph-interception/preplan.md)
- [R26: PTP Clock Sync](../R26-ptp-clock-sync/)
- [R28: Scatter-Gather DMA](../R28-scatter-gather-dma/)
- [02-FINAL-PREPLAN.md](../../../pre-planning/02-FINAL-PREPLAN.md)
