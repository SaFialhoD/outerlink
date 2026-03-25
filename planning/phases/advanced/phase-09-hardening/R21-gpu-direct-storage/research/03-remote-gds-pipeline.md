# R21 Research: Remote GPU Direct Storage Pipeline

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** DRAFT

## Purpose

Design the full remote NVMe-to-GPU pipeline for OuterLink. This combines research from 01 (GDS architecture) and 02 (P2PDMA/NVMe) into a concrete data path: remote NVMe SSD --> network --> GPU VRAM, with minimal or zero host RAM involvement on either side.

---

## 1. The Full Pipeline: End-to-End Data Path

### Target Architecture

```
STORAGE NODE                          GPU NODE
============                          ========

NVMe SSD                              GPU VRAM (BAR1)
    |                                     ^
    | PCIe P2P (via embedded switch       | PCIe BAR1 write
    | or root complex forwarding)         | (OpenDMA)
    v                                     |
ConnectX-5 NIC  ---- 100Gbps RDMA ----> ConnectX-5 NIC
(sender)              wire               (receiver)
```

### Two Approaches: Two-Hop vs One-Hop

#### Two-Hop (Host-Staged on Sender)

```
Sender:   NVMe --DMA--> host RAM --RDMA send--> ConnectX-5 --> wire
Receiver: wire --> ConnectX-5 --DMA to BAR1--> GPU VRAM (OpenDMA)
```

- Sender uses standard NVMe read to host pinned memory, then RDMA send
- Receiver uses OpenDMA (ConnectX-5 DMA to GPU BAR1) — zero host RAM touch
- **Host RAM touched on sender side only**

#### One-Hop (Full P2P on Sender)

```
Sender:   NVMe --P2P DMA--> ConnectX-5 (NVMe-oF target offload) --> wire
Receiver: wire --> ConnectX-5 --DMA to BAR1--> GPU VRAM (OpenDMA)
```

- Sender uses ConnectX-5 NVMe-oF target offload — NVMe data goes directly from SSD through the ConnectX-5 embedded PCIe switch to the wire
- Receiver uses OpenDMA — data goes directly from wire to GPU VRAM
- **Zero host RAM touch on either side**

---

## 2. Sender Side Analysis

### Option A: Host-Staged Sender (Simpler, Always Works)

```
1. read(nvme_fd, pinned_buffer, size)     -- NVMe DMA to pinned host RAM
2. ibv_post_send(pinned_buffer, ...)      -- RDMA send from pinned RAM
```

| Aspect | Detail |
|--------|--------|
| **Complexity** | Low — standard NVMe read + RDMA send |
| **Host RAM** | Yes — uses pinned memory as staging buffer |
| **CPU involvement** | Moderate — orchestrates read + send |
| **Throughput** | Limited by min(NVMe read, RDMA send) = ~7 GB/s |
| **PCIe topology** | No P2P requirements |
| **Compatibility** | Works with any NVMe SSD and any NIC |

**Pipeline optimization:** Can overlap NVMe reads and RDMA sends with double-buffering:
```
Buffer A: NVMe reading chunk N+1    |  Buffer B: RDMA sending chunk N
Buffer A: RDMA sending chunk N+1    |  Buffer B: NVMe reading chunk N+2
```
This hides latency and keeps both NVMe and network busy.

### Option B: NVMe-oF Target Offload (Zero-Copy, Hardware Requirements)

```
1. Configure nvmet subsystem exposing local NVMe
2. Enable ConnectX-5 NVMe-oF target offload
3. Remote initiator sends NVMe-oF read command
4. ConnectX-5 hardware reads from NVMe via embedded switch, sends over RDMA
```

| Aspect | Detail |
|--------|--------|
| **Complexity** | Medium — requires nvmet + offload configuration |
| **Host RAM** | No — data stays in PCIe fabric |
| **CPU involvement** | Zero for data path (hardware handles everything) |
| **Throughput** | Limited by min(NVMe read, RDMA send) = ~7 GB/s |
| **PCIe topology** | NVMe must be accessible via ConnectX-5 embedded switch or same PCIe hierarchy |
| **Compatibility** | ConnectX-5+ with NVMe-oF offload, NVMe behind compatible PCIe topology |

### Option C: SPDK Userspace NVMe-oF Target

```
1. SPDK takes ownership of NVMe device (via VFIO)
2. SPDK NVMe-oF target serves over RDMA
3. Optional: use P2P memory (CMB or NIC BAR) for zero-copy
```

| Aspect | Detail |
|--------|--------|
| **Complexity** | High — SPDK setup, dedicated cores, VFIO |
| **Host RAM** | Optional — can use P2P memory if CMB available |
| **CPU involvement** | Dedicated core(s) for polling |
| **Throughput** | Highest — polled mode, lowest latency |
| **PCIe topology** | Flexible — SPDK manages DMA directly |
| **Compatibility** | Requires VFIO, dedicated cores, NVMe not available to kernel |

### Sender Recommendation

| Phase | Approach | Rationale |
|-------|----------|-----------|
| **Initial** | Option A (host-staged) | Works everywhere, simple, uses existing RDMA transport |
| **Optimized** | Option B (NVMe-oF target offload) | Zero CPU, zero host RAM, but requires PCIe topology validation |
| **Advanced** | Option C (SPDK) | Only if we need sub-5us latency or kernel NVMe-oF is insufficient |

---

## 3. Receiver Side Analysis

The receiver side is simpler because we already have OpenDMA (R10/P9):

### OpenDMA Receiver (Primary Path)

```
wire --> ConnectX-5 RDMA receive --> DMA to GPU BAR1 address --> data in VRAM
```

This is the existing OpenDMA data path. The ConnectX-5 DMA engine writes received RDMA data directly to GPU BAR1 physical addresses, which map to VRAM. No host RAM involved.

| Aspect | Detail |
|--------|--------|
| **Host RAM** | Zero |
| **CPU involvement** | Completion notification only |
| **Throughput** | Up to 12.5 GB/s (100Gbps line rate) |
| **Requirements** | OpenDMA working (P9), GPU BAR1 mapped |

### Host-Staged Receiver (Fallback)

```
wire --> ConnectX-5 RDMA receive --> pinned host RAM --> cudaMemcpy --> GPU VRAM
```

This is Phase 1 transport — the fallback when OpenDMA is unavailable.

---

## 4. Can the ConnectX-5 DMA Engine Chain NVMe Read + Network Send?

This is a key question from R21's scope.

### Answer: Yes, via NVMe-oF Target Offload

ConnectX-5's NVMe-oF target offload **is** the chaining of NVMe read + network send in hardware:

1. Incoming NVMe-oF read command arrives via RDMA
2. ConnectX-5 hardware translates it to a local NVMe read via the embedded PCIe switch
3. NVMe data flows through the embedded switch directly to the RDMA send engine
4. Data goes out on the wire

This is not a software pipeline — it happens entirely within the ConnectX-5 hardware. The CPU is not involved in the data path at all.

### Limitations

- NVMe drive must be accessible via the ConnectX-5's embedded PCIe switch or a compatible P2P topology
- The NVMe must be configured as an nvmet subsystem
- Standard NVMe-oF protocol semantics (block-level access, not file-level)

### Can We Chain in the Opposite Direction? (Network Receive + NVMe Write)

Yes — NVMe-oF target offload handles both reads and writes. An incoming NVMe-oF write command causes the ConnectX-5 to:
1. Receive data via RDMA
2. Write data to local NVMe via embedded PCIe switch P2P DMA

This means we can also do GPU VRAM --> network --> remote NVMe for checkpointing.

---

## 5. Throughput Analysis and Bottleneck Identification

### Component Bandwidth Budget

| Component | Bandwidth | Bottleneck? |
|-----------|-----------|-------------|
| Single NVMe Gen4 SSD | ~7.0-7.5 GB/s seq read | **Yes — usually the bottleneck** |
| Single NVMe Gen5 SSD | ~12-14 GB/s seq read | Matches network |
| PCIe Gen4 x4 (NVMe slot) | ~8 GB/s | Slight overhead above NVMe |
| PCIe Gen4 x16 (GPU slot) | ~32 GB/s | Not a bottleneck |
| ConnectX-5 100GbE | ~12.5 GB/s | Not bottleneck for single NVMe |
| GPU BAR1 write bandwidth | ~12-25 GB/s | Not a bottleneck |
| System RAM (DDR4 dual-ch) | ~50 GB/s | Not a bottleneck (if host-staged) |
| ConnectX-5 embedded switch | PCIe Gen3 x16 = ~16 GB/s | Not bottleneck for single NVMe |

### End-to-End Throughput Estimates

| Configuration | Expected Throughput | Bottleneck |
|---------------|-------------------|------------|
| 1x Gen4 NVMe → network → GPU | ~6.5-7 GB/s | NVMe sequential read |
| 2x Gen4 NVMe (striped) → network → GPU | ~12-12.5 GB/s | Network bandwidth |
| 4x Gen4 NVMe (striped) → network → GPU | ~12.5 GB/s | Network bandwidth |
| 1x Gen5 NVMe → network → GPU | ~12-12.5 GB/s | Network bandwidth |

### Latency Budget

| Stage | Latency |
|-------|---------|
| NVMe read (4KB-1MB) | 10-100 us |
| PCIe P2P transfer | ~1-2 us |
| RDMA network transfer | 2-5 us (100GbE) |
| GPU BAR1 write | ~1-2 us |
| **Total (small I/O)** | **~15-110 us** |
| **Total (large sequential)** | Throughput-bound, latency amortized |

### Scaling Strategy

To saturate 100Gbps network from storage:
- **2x Gen4 NVMe drives** striped = ~14 GB/s raw → ~12.5 GB/s network-limited
- **RAID0 or application-level striping** across drives
- Each GPU node can receive from multiple storage nodes simultaneously

---

## 6. cuFile-Like API Design for OuterLink

### Why Build Our Own API

- NVIDIA's cuFile requires Data Center/Quadro GPUs for native mode
- We need remote storage support (cuFile is local-only without complex NVMe-oF setup)
- Integration with OuterLink's memory tiering (R10) and transport layer

### Proposed API

```rust
/// OuterLink Storage API (olStorage)

/// Open a remote storage file
fn ol_storage_open(node_id: NodeId, path: &str, flags: OpenFlags) -> Result<StorageHandle>;

/// Read from remote storage directly to GPU VRAM
fn ol_storage_read_to_gpu(
    handle: &StorageHandle,
    gpu_buffer: CudaDevicePtr,      // destination in GPU VRAM
    size: usize,
    file_offset: u64,
    buf_offset: usize,
) -> Result<usize>;

/// Batch read (multiple regions in one call)
fn ol_storage_batch_read(
    handle: &StorageHandle,
    requests: &[StorageReadRequest],  // vec of (gpu_ptr, size, file_offset)
) -> Result<Vec<usize>>;

/// Async read integrated with CUDA stream
fn ol_storage_read_async(
    handle: &StorageHandle,
    gpu_buffer: CudaDevicePtr,
    size: usize,
    file_offset: u64,
    stream: CudaStream,
) -> Result<StorageFuture>;

/// Write GPU VRAM to remote storage (for checkpointing)
fn ol_storage_write_from_gpu(
    handle: &StorageHandle,
    gpu_buffer: CudaDevicePtr,
    size: usize,
    file_offset: u64,
) -> Result<usize>;
```

### Alignment Requirements

Like GDS, direct DMA requires alignment:

| Parameter | Requirement |
|-----------|------------|
| File offset | 4KB aligned (for O_DIRECT) |
| Buffer offset | 4KB aligned (for DMA) |
| Transfer size | Multiple of 4KB (optimal), 512B minimum |
| GPU buffer | Must be in BAR1-mappable region |

When alignment requirements aren't met, fall back to host-staged transfer (or GPU-side bounce buffer).

---

## 7. Integration with R10 Memory Tiering

### R10 Memory Hierarchy

| Tier | Location | Bandwidth | Latency |
|------|----------|-----------|---------|
| Tier 0 | Local GPU VRAM | ~900 GB/s | ~300 ns |
| Tier 1 | Remote GPU VRAM (OpenDMA) | ~12.5 GB/s | ~2-5 us |
| Tier 2 | Local host DRAM | ~50 GB/s | ~100 ns |
| Tier 3 | Remote host DRAM | ~12.5 GB/s | ~2-5 us |
| **Tier 4** | **Local NVMe** | **~7 GB/s** | **~10-100 us** |
| **Tier 5** | **Remote NVMe** | **~7 GB/s** | **~15-110 us** |

### How R21 Enables Tier 5

Without R21, accessing remote NVMe requires:
```
Remote NVMe → Remote host RAM → Network → Local host RAM → GPU VRAM
(4 hops, 4 copies, ~7 GB/s bottlenecked by NVMe + extra latency)
```

With R21 (best case, full P2P):
```
Remote NVMe → ConnectX-5 P2P → Network → ConnectX-5 OpenDMA → GPU VRAM
(2 hops, 0 host RAM copies, ~7 GB/s bottlenecked by NVMe, lower latency)
```

### Automatic Tiering Behavior

The memory tiering system (R10) would use R21's storage API when:
- A page is evicted from all RAM tiers and needs to be read from NVMe backing store
- A dataset is too large for any node's RAM and must be streamed from storage
- Prefetching predictions indicate upcoming access to storage-resident data

---

## 8. Comparison: SPDK + RDMA vs Kernel P2PDMA + RDMA

| Aspect | Kernel P2PDMA + nvmet RDMA | SPDK + RDMA |
|--------|---------------------------|-------------|
| **Setup complexity** | Low — kernel modules, sysfs config | High — VFIO, dedicated cores, userspace |
| **NVMe availability** | Kernel still owns NVMe | SPDK takes exclusive ownership |
| **Latency** | ~10-15 us (interrupt-driven) | ~2-5 us (polled) |
| **Throughput** | Near line rate with offload | Near line rate |
| **CPU usage** | Near zero with ConnectX-5 offload | Dedicated core(s) for polling |
| **P2P support** | Native kernel P2PDMA | SPDK's own P2P with CMB |
| **Filesystem access** | Full — ext4, XFS, etc. | BlobFS only, or raw blocks |
| **Hardware offload** | ConnectX-5 NVMe-oF target offload | No hardware offload (CPU polled) |
| **Maintenance** | Kernel-maintained | Must track SPDK releases |
| **GPU integration** | Via our OpenDMA (BAR1 DMA) | Via our OpenDMA (BAR1 DMA) |

### Recommendation

**Use kernel P2PDMA + nvmet RDMA with ConnectX-5 offload** as the primary approach:
- Lower complexity
- ConnectX-5 hardware offload provides near-zero CPU usage (same benefit as SPDK's polling)
- NVMe remains available to the kernel for filesystem operations
- P2PDMA is upstream and well-maintained

**SPDK is a Plan B** if:
- We need sub-5us storage latency for specific workloads
- Kernel nvmet performance is insufficient
- We want to build a dedicated storage appliance mode

---

## 9. Implementation Strategy

### Phase 1: Host-Staged Remote Storage (2-3 weeks)

The simplest path that still demonstrates remote NVMe-to-GPU:

```
Sender:   read(nvme, pinned_buf) → RDMA send(pinned_buf)
Receiver: RDMA recv → OpenDMA to GPU BAR1
```

- Host RAM touched on sender side only
- Receiver side is pure OpenDMA (zero-copy to GPU)
- Uses existing transport infrastructure

### Phase 2: NVMe-oF Integration (2-3 weeks)

Expose remote NVMe as a block device via NVMe-oF:

```
Sender:   nvmet subsystem + ConnectX-5 offload
Receiver: nvme-cli connect → /dev/nvmeXnY appears locally
```

Then our storage API reads from the NVMe-oF device and transfers to GPU via OpenDMA.

### Phase 3: Full P2P Pipeline (2-3 weeks)

Zero-copy on both sides:

```
Sender:   ConnectX-5 NVMe-oF target offload (NVMe → wire, zero CPU)
Receiver: Custom RDMA receive → OpenDMA to GPU BAR1
```

This requires combining NVMe-oF on the sender with our custom RDMA-to-BAR1 on the receiver. The challenge is bridging NVMe-oF protocol (block semantics) with our GPU memory management.

### Phase 4: API and Tiering Integration (1-2 weeks)

- Build the `ol_storage_*` API described in Section 6
- Integrate with R10 memory tiering as Tier 4/5
- Add prefetching support for predictable access patterns

---

## 10. Key Takeaways

1. **The full pipeline is achievable** — all components exist (NVMe-oF target offload, RDMA, OpenDMA). The innovation is combining them.

2. **Start with host-staged sender + OpenDMA receiver** — gets 90% of the benefit (zero-copy on GPU side) with minimal complexity. Optimize sender later.

3. **ConnectX-5 NVMe-oF target offload is the key to zero-copy on sender** — hardware chains NVMe read + RDMA send with zero CPU involvement.

4. **NVMe is always the bottleneck** — ~7 GB/s per Gen4 drive. Stripe across drives or use Gen5 to match network bandwidth.

5. **Kernel approach over SPDK** — ConnectX-5 hardware offload gives us the same CPU savings as SPDK's polling, with lower complexity.

6. **API should mirror cuFile patterns** — batch I/O, stream integration, alignment handling. Makes it familiar for CUDA developers.

7. **R10 integration is natural** — remote NVMe becomes Tier 5 in the memory hierarchy, accessed transparently by the page manager.

---

## Related Documents

- [01-nvidia-gds-architecture.md](01-nvidia-gds-architecture.md) — NVIDIA GDS internals and why we can't use it on GeForce
- [02-p2pdma-and-nvme.md](02-p2pdma-and-nvme.md) — P2PDMA framework, NVMe CMB, NVMe-oF details
- [R10: Memory Tiering](../../../../phase-08-smart-memory/R10-memory-tiering/README.md) — NVMe as Tier 4/5
- [P9: OpenDMA](../../../../phase-05-opendma/README.md) — BAR1 RDMA (receiver side of this pipeline)

## Open Questions

- [ ] Can we use NVMe-oF target offload on the sender AND custom RDMA (non-NVMe-oF protocol) on the receiver? Or does NVMe-oF offload require the remote side to speak NVMe-oF?
- [ ] What's the overhead of NVMe-oF protocol encapsulation vs raw RDMA for large sequential reads?
- [ ] Can we do scatter-gather: read from multiple NVMe regions and DMA to a single contiguous GPU buffer? (Relates to R28)
- [ ] How does NVMe-oF handle multiple concurrent readers accessing the same file? (Important for multi-GPU training data loading)
- [ ] Can we expose a POSIX-like file API on top of NVMe-oF, or is it strictly block-level?
