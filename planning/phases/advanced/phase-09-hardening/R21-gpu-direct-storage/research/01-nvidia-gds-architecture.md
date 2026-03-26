# R21 Research: NVIDIA GPUDirect Storage Architecture

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** DRAFT

## Purpose

Document NVIDIA's GPUDirect Storage (GDS) architecture, cuFile API, performance characteristics, and compatibility requirements. Assess what parts of GDS can be reused with OuterLink's open-source BAR1 approach vs what requires NVIDIA's proprietary stack.

---

## 1. How GPUDirect Storage Works

### The Problem GDS Solves

Traditional storage-to-GPU data path:

```
NVMe SSD --DMA--> System RAM (bounce buffer) --cudaMemcpy--> GPU VRAM
```

This requires two copies and consumes system memory bandwidth. The CPU orchestrates both transfers, and the bounce buffer in system RAM limits throughput.

### GDS Direct Path

```
NVMe SSD --DMA--> GPU VRAM (via PCIe P2P)
```

GDS eliminates the system RAM bounce buffer by enabling the NVMe controller's DMA engine to write directly to GPU BAR1 memory (which maps to VRAM). The storage controller targets GPU BAR1 addresses as the DMA destination.

### Architecture Components

| Component | Role |
|-----------|------|
| **cuFile API** | Userspace library providing `cuFileRead()`, `cuFileWrite()` and batch/stream variants |
| **nvidia-fs.ko** | Kernel module that registers GPU BAR1 memory with the filesystem/block layer (being deprecated in favor of P2PDMA) |
| **GDS driver** | Translates cuFile calls into DMA operations targeting GPU BAR1 addresses |
| **NVMe driver** | Issues DMA commands with GPU BAR1 physical addresses as source/destination |
| **PCIe fabric** | Routes DMA transactions between NVMe controller and GPU BAR1 |

### Data Flow (Native GDS Mode)

1. Application calls `cuFileRead(file_handle, gpu_buffer, size, file_offset, buf_offset)`
2. GDS library translates `gpu_buffer` (CUDA device pointer) to a BAR1 physical address
3. GDS issues an O_DIRECT read request with the BAR1 address as the DMA target
4. The NVMe controller's DMA engine reads data from flash and writes it directly to the GPU BAR1 address
5. Data appears in GPU VRAM without ever touching system RAM

---

## 2. cuFile API Overview

### Core API Functions

| Function | Description |
|----------|-------------|
| `cuFileDriverOpen()` | Initialize GDS driver |
| `cuFileHandleRegister()` | Register a file descriptor for GDS operations |
| `cuFileBufRegister()` | Register a GPU buffer (maps BAR1 for DMA) |
| `cuFileRead()` | Synchronous read: storage to GPU buffer |
| `cuFileWrite()` | Synchronous write: GPU buffer to storage |
| `cuFileBatchIOSubmit()` | Asynchronous batch I/O (like Linux AIO) |
| `cuFileReadAsync()` | Stream-ordered async read (CUDA 12.2+) |
| `cuFileWriteAsync()` | Stream-ordered async write (CUDA 12.2+) |

### Stream API (CUDA 12.2+)

Stream APIs integrate with CUDA streams for ordering guarantees:
- **IO after compute**: GPU kernel produces data, then writes to storage
- **Compute after IO**: Load data from storage, then launch kernel
- **Concurrent streams**: Multiple DMA engines operate in parallel across CUDA streams

### Buffer Registration

`cuFileBufRegister()` is critical: it maps the GPU buffer's BAR1 region and pins it for DMA. Without registration, GDS falls back to internal GPU bounce buffers (still avoids system RAM, but adds an extra GPU-side copy).

---

## 3. Native Mode vs Compatibility Mode

### Native (P2P DMA) Mode

| Aspect | Detail |
|--------|--------|
| **Data path** | NVMe DMA engine writes directly to GPU BAR1 |
| **Bounce buffer** | None (system RAM completely bypassed) |
| **Throughput** | Up to 2x-8x higher than traditional path |
| **Latency** | Lowest — single DMA hop |
| **CPU involvement** | Minimal — setup only, no data copy |
| **Requirements** | Data center / Quadro GPU, specific filesystem, PCIe topology, IOMMU off or configured |

### Compatibility Mode

| Aspect | Detail |
|--------|--------|
| **Data path** | NVMe --> System RAM --> cudaMemcpy --> GPU VRAM (standard POSIX path) |
| **Bounce buffer** | Yes, system RAM bounce buffer used |
| **Throughput** | Same as traditional read() + cudaMemcpy |
| **Latency** | Higher — two copy operations |
| **CPU involvement** | Full — CPU orchestrates both copies |
| **Requirements** | Any GPU, any filesystem, works everywhere |

### When Native Mode Falls Back to Bounce Buffers

Even in native mode, GDS uses internal **GPU bounce buffers** (not system RAM) in certain cases:
- Buffer not registered with `cuFileBufRegister()`
- File offset not 4KB aligned
- Buffer offset not aligned
- Storage and GPU cross NUMA node boundaries
- Default pool: 128 bounce buffers of 1MB each (128MB total), configurable up to 16MB per buffer

### Key Insight for OuterLink

Compatibility mode is just a fancy wrapper around standard POSIX I/O — it provides **zero benefit** over what we could implement ourselves. The value is entirely in native mode's direct DMA path.

---

## 4. GPU and Hardware Requirements

### Supported GPUs for Native GDS

| GPU Class | Native GDS Support | Compatibility Mode |
|-----------|-------------------|-------------------|
| **Data Center (Tesla)** — V100, A100, H100 | Yes | Yes |
| **Quadro** — P6000, RTX A6000 | Yes (compute capability >= 6) | Yes |
| **GeForce** — RTX 3090, 4090, etc. | **NO** | Yes |
| **Laptop GPUs** | **NO** | Yes |

**Critical finding: GeForce GPUs (including our RTX 3090s) cannot use native GDS mode.** They are limited to compatibility mode, which is just standard POSIX I/O with a CPU bounce buffer.

### Other Requirements for Native GDS

| Requirement | Detail |
|-------------|--------|
| **Kernel module** | nvidia-fs.ko (being replaced by kernel P2PDMA in CUDA 12.8+) |
| **NVIDIA driver** | Open kernel module required (proprietary not supported for GDS) |
| **Filesystem** | ext4 (ordered mode), XFS on NVMe/NVMe-oF devices |
| **File open mode** | O_DIRECT required (relaxed in GDS 1.7 / CUDA 12.2) |
| **PCIe ACS** | Must be disabled for best performance |
| **IOMMU** | Recommended disabled (can cause failures or poor performance) |
| **PCIe topology** | GPU and NVMe should be on same PCIe root complex / behind same switch |

### P2PDMA Mode (New in CUDA 12.8 / GDS 1.16)

Starting with CUDA 12.8 and driver 570.x:
- GDS can use the upstream Linux kernel P2PDMA framework instead of nvidia-fs.ko
- Requires Linux kernel 6.2+
- Eliminates dependency on custom NVMe patches and nvidia-fs kernel module
- Works with ext4 and XFS on NVMe devices
- Currently x86_64 only

This is significant: it shows NVIDIA is moving toward the same kernel P2PDMA infrastructure that OuterLink's OpenDMA approach would use.

---

## 5. Performance Numbers

### GDS Throughput (from NVIDIA Benchmarks)

| Configuration | Throughput | Notes |
|---------------|-----------|-------|
| GDS native, optimal topology | **6.5 GB/s** per GPU | GPU and NVMe on same PCIe switch |
| GDS native, cross-NUMA | **3.9 GB/s** per GPU | 40% reduction from topology mismatch |
| GDS + NIC (remote storage) | Up to **50 GB/s** aggregate | Multiple GPUs + multiple NICs |
| RAPIDS cuDF with GDS | **30-50% speedup** over non-GDS | For datasets >= 512MB |
| Compatibility mode | Same as POSIX + cudaMemcpy | No improvement over traditional path |

### Throughput vs Traditional Path

| Metric | Traditional (bounce buffer) | GDS Native |
|--------|---------------------------|------------|
| Bandwidth | Limited by system RAM BW | Up to 2x-8x higher |
| CPU utilization | High (orchestrates copies) | Near zero |
| Latency | Higher (two hops) | Lower (single DMA) |

### Bottleneck Analysis

| Component | Bandwidth |
|-----------|-----------|
| Single NVMe Gen4 SSD | ~7.0-7.5 GB/s sequential read |
| PCIe Gen4 x4 (NVMe slot) | ~8 GB/s theoretical |
| PCIe Gen4 x16 (GPU slot) | ~32 GB/s theoretical |
| GPU BAR1 write bandwidth | ~12-25 GB/s (depends on GPU) |
| System RAM bandwidth | ~50 GB/s (DDR4 dual-channel) |

The bottleneck for local GDS is always the NVMe SSD (~7 GB/s). GPU BAR1 and PCIe x16 have capacity to spare.

---

## 6. What Can We Reuse for OuterLink?

### What Requires NVIDIA's Proprietary Stack

| Component | Proprietary? | Alternative for OuterLink |
|-----------|-------------|--------------------------|
| cuFile API | Yes (closed-source library) | We build our own API |
| nvidia-fs.ko | Yes (but being replaced) | Not needed — we use kernel P2PDMA or OpenDMA directly |
| GPU class restriction (no GeForce) | Yes (artificial lock) | **OpenDMA bypasses this** — BAR1 access works on all GPUs |
| GDS driver (BAR1 translation) | Yes | OpenDMA does this via ConnectX-5 DMA to BAR1 |

### What We Can Reuse (Concepts, Not Code)

| Concept | How It Applies to OuterLink |
|---------|----------------------------|
| **BAR1 as DMA target** | Same approach as OpenDMA — storage DMA writes to GPU BAR1 addresses |
| **O_DIRECT for alignment** | We need aligned I/O too when doing direct DMA from NVMe |
| **Batch I/O pattern** | Our cuFile-like API should support batched reads for throughput |
| **Stream ordering** | Integration with CUDA streams for compute-after-IO ordering |
| **GPU bounce buffers for unaligned** | Fallback when alignment requirements aren't met |

### Why OuterLink's Approach Is Better for Our Use Case

| Aspect | NVIDIA GDS | OuterLink OpenDMA + Storage |
|--------|-----------|---------------------------|
| **GeForce support** | No | Yes (BAR1 works on all GPUs) |
| **Kernel module** | nvidia-fs.ko required (being replaced) | No custom kernel modules for network path |
| **Remote storage** | Limited (NVMe-oF with specific setups) | Native — designed for network-first |
| **Combined with RDMA** | Separate feature (GPUDirect RDMA) | Unified — same ConnectX-5 DMA engine |
| **Open source** | cuFile is closed, nvidia-fs is open | Fully open source |

---

## 7. Key Takeaways for R21

1. **GDS native mode is irrelevant for GeForce GPUs** — NVIDIA artificially restricts it to Data Center/Quadro. OuterLink's OpenDMA approach bypasses this restriction.

2. **The BAR1 DMA concept is sound and validated** — GDS proves that NVMe DMA to GPU BAR1 works. We just take a different path to achieve it.

3. **NVIDIA is moving to kernel P2PDMA** — CUDA 12.8 / GDS 1.16 uses upstream P2PDMA instead of nvidia-fs.ko. This validates the open-source P2P approach.

4. **cuFile API design is a good reference** — batch I/O, stream integration, alignment handling are all patterns we should adopt.

5. **Remote GDS (storage over network) is the gap** — GDS does local NVMe-to-GPU well. OuterLink's value-add is doing this across the network: remote NVMe --> network --> GPU VRAM.

6. **Performance ceiling is ~7 GB/s per NVMe** — the SSD is always the bottleneck for single-drive reads. Striping across multiple NVMe drives can scale linearly.

---

## Related Documents

- [02-p2pdma-and-nvme.md](02-p2pdma-and-nvme.md) — Linux P2PDMA framework and NVMe direct access
- [03-remote-gds-pipeline.md](03-remote-gds-pipeline.md) — Building the full remote storage pipeline
- [R10: Memory Tiering](../../../../phase-08-smart-memory/R10-memory-tiering/README.md) — NVMe as Tier 4/5 in memory hierarchy

## Open Questions

- [ ] Does CUDA 12.8 P2PDMA mode work on GeForce GPUs? (It uses kernel P2PDMA, which doesn't have the artificial GPU class restriction)
- [ ] Can we intercept cuFile calls via LD_PRELOAD the same way we intercept CUDA driver API? (For apps already using GDS)
- [ ] What BAR1 size do our RTX 3090s expose? (Determines max concurrent DMA region size)
- [x] Does GDS work with our BAR1 approach? **No — GDS requires nvidia-fs.ko or P2PDMA kernel support with registered GPU memory. But we don't need GDS; OpenDMA provides the same DMA-to-BAR1 capability.**
