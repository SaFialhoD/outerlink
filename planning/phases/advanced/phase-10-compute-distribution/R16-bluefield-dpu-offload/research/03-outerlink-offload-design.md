# R16 Research: OuterLink Offload Design

**Created:** 2026-03-25
**Updated:** 2026-03-25
**Status:** APPROVED
**Purpose:** Define what OuterLink functionality to offload to BlueField DPU, how to split work between host and DPU, and the data flow architecture.

---

## 1. Design Principle: DPU as Transparent Accelerator

The DPU offload must be **optional**. OuterLink runs identically on a system with a plain ConnectX NIC — the DPU adds acceleration, not new functionality. The architecture:

```
WITH DPU:
  App → LD_PRELOAD → outerlink-client → outerlink-server → outerlink-dpu → Network
                                              ↑                    ↑
                                         (control plane)     (data plane)

WITHOUT DPU:
  App → LD_PRELOAD → outerlink-client → outerlink-server → ConnectX NIC → Network
                                              ↑
                                     (control + data plane)
```

When a DPU is present, the host-side `outerlink-server` detects it (via DOCA device enumeration or a config flag) and delegates transport operations to `outerlink-dpu` running on the ARM cores. When no DPU is present, `outerlink-server` handles everything itself, as it does today.

---

## 2. What to Offload

### 2.1 Transport Layer Offload

**What moves to DPU:**
- Connection state machine (establish, maintain, teardown RDMA connections)
- Packet routing decisions (which data goes to which remote node)
- Retransmission logic (detect lost packets, re-send from DPU buffers)
- Congestion control (DCQCN/ECN processing at line rate)
- Multi-path load balancing (R17 routing across multiple links)
- ACK generation and processing

**What stays on host:**
- CUDA API interception (LD_PRELOAD, driver API hooks)
- CUDA context management (must stay in host process address space)
- Application-visible CUDA operations

**Data Flow:**
```
Host GPU VRAM                    DPU                         Remote
┌──────────┐    PCIe BAR1    ┌──────────┐    RDMA Wire    ┌──────────┐
│          │ ◄──────────────►│ ConnectX │ ◄──────────────►│ Remote   │
│  VRAM    │   GPUDirect     │ NIC/DMA  │   100-400G      │ DPU/NIC  │
│          │                 │          │                  │          │
└──────────┘                 │ ARM Cores│                  └──────────┘
                             │ (routing,│
Host CPU                     │  compress,│
┌──────────┐   Comm Channel  │  schedule)│
│ outerlink│ ◄──────────────►│          │
│ -server  │   (control)     │ outerlink│
│          │                 │ -dpu     │
└──────────┘                 └──────────┘
```

**Key insight:** In the DPU model, the host CPU is involved only in the control plane (CUDA interception, deciding what transfers are needed). The actual data movement is DPU→GPU and DPU→Network with zero host CPU data path involvement.

### 2.2 Compression Offload

**Offload strategy (builds on R14):**

| Scenario | Without DPU | With DPU (BF-2) | With DPU (BF-3) |
|---|---|---|---|
| **GPU-resident data** | nvCOMP on GPU | nvCOMP on GPU (preferred) or DMA to DPU → HW deflate | nvCOMP on GPU (preferred) or DMA to DPU → HW LZ4 |
| **Host-staged data** | LZ4/Zstd on CPU | DMA to DPU → HW deflate → send | DMA to DPU → HW LZ4 → send |
| **Small transfers (<4KB)** | No compression | No compression | No compression |
| **Receive path** | CPU decompress | HW decompress on DPU → DMA to host | HW decompress on DPU → DMA to host |

**Decision logic on DPU:**
1. Check data size — skip compression below threshold
2. Check compressibility hint (from R14's adaptive compression)
3. If GPU-resident and large: prefer nvCOMP on source GPU (avoids PCIe round trip)
4. If host-staged or already in DPU memory: use hardware compression engine
5. BF-3: always prefer LZ4 (faster, OuterLink's standard)
6. BF-2: deflate only (still beneficial for compressible data)

**Performance estimate:**
- BF-2 HW deflate: ~100x faster than ARM core software, ~10 GB/s throughput
- BF-3 HW LZ4: line-rate compression at 400 Gbps for compressible data
- Compression engine shares memory bandwidth with NIC, not NIC wire bandwidth

### 2.3 Memory Management Offload

**Page table operations on DPU:**
- Track which VRAM pages are on which node
- Process page fault notifications (remote node requests a page)
- Initiate page migration (DPU-to-DPU RDMA transfer)
- Update local and remote page tables
- Coordinate with host's `outerlink-server` for CUDA-level consistency

**Why this helps:**
- Page faults are latency-sensitive — DPU processes them without host CPU scheduling delays
- Page table is moderate size: 24 GB VRAM / 4 KB pages = 6M entries × ~32 bytes = ~192 MB per GPU
- Even with 4 GPUs per node: ~768 MB — fits easily in DPU's 16-32 GB DRAM
- DPU can pre-compute migration plans and execute them without waking the host

**What stays on host:**
- CUDA memory allocation tracking (cuMemAlloc interception)
- Virtual address space management
- Coherency fencing (CUDA stream synchronization)

### 2.4 Prefetch Scheduling on DPU (R11 Integration)

**Why prefetching belongs on the DPU:**

The DPU sits at the network edge — it sees all incoming and outgoing traffic patterns before the host does. This makes it the ideal location for speculative prefetch decisions:

1. **Traffic pattern observation**: DPU monitors which pages are being accessed by remote nodes
2. **Prediction**: ARM cores run the prefetch model (access pattern → likely next pages)
3. **Proactive fetch**: DPU initiates RDMA reads to pull predicted pages from remote nodes
4. **Zero host involvement**: Pages arrive in DPU buffer, then DMA to host/GPU — host never knew it happened

**ARM core suitability:**
- R11's prefetch models (stride detection, Markov chains, working set estimation) are lightweight
- 1-2 ARM cores dedicated to prefetch scheduling is sufficient
- BF-2: leaves 6 cores for transport + compression
- BF-3: leaves 14 cores (more than enough)

### 2.5 DPU-to-GPU BAR1 Access (OpenDMA Integration)

**This is the crown jewel of DPU offload for OuterLink.**

The DPU's integrated ConnectX NIC can perform GPUDirect RDMA to the local GPU's VRAM via PCIe BAR1. This is exactly the same mechanism as OuterLink's OpenDMA — but the DPU handles it natively:

```
Remote GPU VRAM                                          Local GPU VRAM
┌──────────────┐                                       ┌──────────────┐
│              │    RDMA over wire    ┌──────────┐     │              │
│   Page X     │ ────────────────────►│ DPU      │     │   Page X     │
│              │    (100-400 Gbps)    │ ConnectX │────►│   (BAR1)     │
└──────────────┘                      │ NIC      │     └──────────────┘
                                      │          │
                                      │ ARM cores│ ← routing decision
                                      └──────────┘     made here

                                      Host CPU: completely uninvolved
```

**Data path for remote VRAM read (with DPU):**
1. App on host does `cudaMemcpy` intercepted by OuterLink
2. `outerlink-server` tells `outerlink-dpu`: "need page X from node B"
3. DPU ARM cores: check page table, determine node B has it
4. DPU ConnectX: RDMA Read from node B's DPU → arrives in DPU buffer
5. (Optional) DPU: decompress if data was compressed
6. DPU ConnectX: DMA write to local GPU VRAM via BAR1
7. DPU tells `outerlink-server`: "page X is ready in local VRAM"
8. `outerlink-server` completes the intercepted cudaMemcpy

**Host CPU involvement: one control message in, one completion out.** All data movement is DPU.

**Requirements:**
- GPU and DPU in same PCIe root complex
- Resizable BAR enabled in BIOS (for large BAR1 aperture)
- BF-3: `RmDmaAdjustPeerMmioBF3=1` in NVIDIA driver config
- CUDA driver loaded on host (for GPU memory management)

---

## 3. Host↔DPU Work Split

### Control Plane (Host → DPU)

| Message | Direction | Purpose |
|---|---|---|
| `TransferRequest` | Host → DPU | "Move page X to/from node Y" |
| `AllocNotify` | Host → DPU | "CUDA allocated region at addr A, size S" |
| `FreeNotify` | Host → DPU | "CUDA freed region at addr A" |
| `SyncBarrier` | Host → DPU | "Ensure all pending transfers complete" |
| `ConnectPeer` | Host → DPU | "Establish connection to node Y" |
| `ConfigUpdate` | Host → DPU | "Update compression settings, routing table" |

### Control Plane (DPU → Host)

| Message | Direction | Purpose |
|---|---|---|
| `TransferComplete` | DPU → Host | "Page X is now in local VRAM/host memory" |
| `PageFault` | DPU → Host | "Remote node requested page X, need CUDA context action" |
| `PeerConnected` | DPU → Host | "Connection to node Y established" |
| `Error` | DPU → Host | "Transfer failed, connection lost, etc." |
| `Stats` | DPU → Host | "Bandwidth, latency, compression ratio metrics" |

### Communication Mechanism
- **DOCA Comm Channel**: Purpose-built for host↔DPU control messaging
- **Shared memory via PCIe BAR**: Lower latency for high-frequency messages (page faults)
- **Recommendation**: Start with Comm Channel, optimize to shared memory if latency is a bottleneck

---

## 4. Latency Analysis

### Without DPU (Current OuterLink Design)
```
cudaMemcpy intercept             → 0.1 us
outerlink-server processes       → 1-5 us  (user-space scheduling)
Host CPU: prepare RDMA WR        → 0.5 us
ConnectX NIC: RDMA over wire     → 2-5 us  (100G, typical)
Remote ConnectX → host memory    → 0.5 us
Host CPU: copy to GPU VRAM       → 1-2 us  (cudaMemcpy)
Total                            → ~5-14 us
```

### With DPU (Offloaded)
```
cudaMemcpy intercept             → 0.1 us
outerlink-server → DPU msg       → 1-2 us  (Comm Channel)
DPU ARM: routing decision        → 0.2 us  (page table lookup)
DPU ConnectX: RDMA over wire     → 2-5 us  (same wire latency)
DPU ConnectX: DMA to GPU BAR1    → 0.5 us  (PCIe, no host CPU)
DPU → outerlink-server msg       → 1-2 us  (completion)
Total                            → ~5-10 us
```

### With DPU + Prefetch Hit (Best Case)
```
cudaMemcpy intercept             → 0.1 us
Page already in local VRAM       → 0 us  (prefetch predicted correctly)
outerlink-server: page hit       → 0.1 us
Total                            → ~0.2 us
```

**Net improvement:**
- Standard path: ~20-30% latency reduction (host CPU removed from data path)
- Prefetch hit: ~98% latency reduction (data already local)
- Throughput: host CPU freed for application work, DPU handles transfers at line rate

---

## 5. DPU Resource Allocation

### BlueField-2 (8 cores, 16-32 GB DRAM)

| Resource | Allocation | Purpose |
|---|---|---|
| Core 0 | OS + DOCA runtime | Linux kernel, DOCA services |
| Cores 1-2 | Transport manager | Connection state, routing, retransmission |
| Core 3 | Compression handler | Manage HW compress tasks, adaptive decisions |
| Core 4 | Page table manager | Page ownership, migration planning |
| Core 5 | Prefetch scheduler | Access pattern monitoring, prediction |
| Cores 6-7 | DOCA Flow + overflow | Packet processing exceptions, flow rule updates |
| DRAM 2 GB | OS + DOCA runtime | — |
| DRAM 2 GB | Page tables | 4 GPUs × 192 MB = ~768 MB + metadata |
| DRAM 4 GB | Packet buffers | Staging for compress/decompress |
| DRAM 4 GB | Prefetch cache | Pre-fetched pages awaiting DMA to GPU |
| DRAM 4 GB | Connection state | RDMA QP state, routing tables |

Total: fits in 16 GB with headroom. 32 GB model allows larger prefetch cache.

### BlueField-3 (16 cores, 32 GB DDR5)

More headroom: transport can use 4 cores for higher throughput, prefetch gets 2 cores, 4 cores for overflow and future features. DDR5 bandwidth helps with large page table operations.

---

## 6. Fallback: No DPU Present

OuterLink **must** work without a DPU. The design:

### Detection
```rust
// In outerlink-server startup
fn detect_dpu() -> Option<DpuHandle> {
    // Try to enumerate DOCA devices
    // If DOCA SDK not installed or no BF device: return None
    // If BF device in NIC mode: return None (treat as plain ConnectX)
    // If BF device in DPU mode: connect to outerlink-dpu service, return handle
}
```

### Abstraction Layer
```rust
trait TransportBackend {
    fn transfer(&self, req: TransferRequest) -> Future<TransferResult>;
    fn compress(&self, data: &[u8]) -> Future<Vec<u8>>;
    fn route(&self, dest: NodeId) -> Route;
}

// Host-only implementation (current)
struct HostTransport { /* uses UCX/io_uring directly */ }

// DPU-offloaded implementation
struct DpuTransport { /* delegates to outerlink-dpu via Comm Channel */ }
```

This is a clean trait boundary. All OuterLink code above the transport layer is identical regardless of whether a DPU is present.

---

## 7. Deployment Architecture

### DPU-Side Service
```
outerlink-dpu.service (systemd on BlueField Linux)
├── Binary: /opt/outerlink/outerlink-dpu
├── Config: /etc/outerlink/dpu.toml
├── Logs: journald
├── Auto-start on boot
└── Talks to host via DOCA Comm Channel
```

### Installation Flow
1. Flash BlueField with NVIDIA BSP (Ubuntu-based)
2. Install DOCA SDK on BlueField
3. Deploy `outerlink-dpu` binary (cross-compiled aarch64)
4. Configure: which host GPUs to manage, compression settings, prefetch policy
5. Set BlueField to DPU mode (if not already)
6. Start `outerlink-dpu.service`
7. Host-side `outerlink-server` detects DPU on next start

### Update Path
- New `outerlink-dpu` binary deployed via SCP/rsync to BlueField filesystem
- Restart service — DPU back online in seconds
- Host-side OuterLink continues running during DPU update (falls back to host transport momentarily)

---

## 8. Phased Implementation

### Phase A: DOCA Foundation (Minimum Viable DPU)
- Rust FFI bindings for DOCA Core, DMA, Comm Channel
- Host↔DPU control plane communication
- DPU-initiated DMA to host memory (prove the data path works)
- No compression, no GPU BAR1 — just host memory
- **Deliverable**: DPU can copy data between its DRAM and host pinned memory

### Phase B: Transport Offload
- Move connection management to DPU
- DPU handles RDMA send/recv for OuterLink protocol
- Routing decisions on DPU ARM cores
- Host CPU no longer touches data path
- **Deliverable**: OuterLink transfers work with DPU handling all network I/O

### Phase C: Compression Offload
- DOCA Compress integration (deflate on BF-2, + LZ4 on BF-3)
- Adaptive compression logic on DPU (from R14)
- Compress before send, decompress after receive
- **Deliverable**: Wire traffic is compressed/decompressed entirely on DPU

### Phase D: GPU BAR1 Integration (OpenDMA on DPU)
- DPU's ConnectX writes directly to local GPU VRAM via BAR1
- Eliminates host-staged transfer for GPU-bound data
- Requires PCIe topology validation per system
- **Deliverable**: Remote VRAM → wire → DPU → local GPU VRAM, zero host CPU

### Phase E: Prefetch on DPU
- Migrate R11's prefetch scheduling logic to DPU ARM cores
- DPU monitors traffic patterns and proactively fetches predicted pages
- Pre-fetched pages staged in DPU DRAM or written directly to GPU VRAM
- **Deliverable**: Reduced effective latency through intelligent prefetching at the network edge

---

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| ARM cores too slow for transport at line rate | Data path bottleneck | Use DOCA Flow for fast-path hardware steering, ARM only handles exceptions |
| 32 GB DRAM insufficient for large clusters | Cannot track all pages | Tiered page table: hot pages on DPU, cold pages queried from host |
| PCIe topology prevents GPUDirect | No BAR1 access from DPU | Fall back to host-staged transfers (still get compression + routing offload) |
| DOCA SDK breaking changes | Build failures | Pin DOCA version, test on both BF-2 and BF-3 in CI |
| DPU hardware failure | Data path down | Auto-fallback to host transport (trait abstraction handles this) |
| DOCA license restrictions | Cannot distribute | DOCA SDK is freely available, Apache 2.0 compatible |

---

## 10. Verdict

BlueField DPU offload is a strong fit for OuterLink. The architecture is clean:

1. **Control plane stays on host** — CUDA interception, application interface
2. **Data plane moves to DPU** — routing, compression, RDMA, GPU BAR1 access
3. **Fallback is transparent** — same trait interface, host handles everything when no DPU

The phased approach (A→E) lets us validate each layer independently. Phase A (DOCA Foundation) is the only prerequisite — once host↔DPU communication works, everything else builds on it.

The BF-2 is sufficient for development and validates the architecture. BF-3 unlocks production-grade performance with LZ4 hardware compression and dedicated packet processing cores.

---

## Related Documents
- [01-bluefield-architecture.md](./01-bluefield-architecture.md) — Hardware specs
- [02-programming-models.md](./02-programming-models.md) — DOCA SDK, Rust FFI
- [R14: Transport Compression](../../R14-transport-compression/) — Compression algorithms
- [R17: Topology-Aware Scheduling](../../R17-topology-aware-scheduling/) — Routing logic
- [R11: Speculative Prefetching](../../R11-speculative-prefetch/) — Prefetch models
- [R10: Memory Hierarchy](../../R10-memory-hierarchy/) — Page management

## Open Questions
- [ ] Should outerlink-dpu be a single monolithic binary or separate services per function (transport, compress, prefetch)?
- [ ] How to handle DPU firmware updates that might change DOCA behavior?
- [ ] Can we run outerlink-dpu in a container on the DPU for easier deployment, or does DOCA require host-level access?
- [ ] What is the minimum DOCA SDK version that supports all features we need?
- [ ] How to test DPU offload in CI without physical BlueField hardware? (DOCA has a software emulator?)
- [ ] Should the prefetch model be trained on the host (more compute) and deployed to the DPU (inference only)?
