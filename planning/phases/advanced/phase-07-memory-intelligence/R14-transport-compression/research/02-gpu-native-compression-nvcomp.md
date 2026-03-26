# R14 Research: GPU-Native Compression with NVIDIA nvCOMP

**Date Created:** 2026-03-25
**Date Last Updated:** 2026-03-25
**Status:** DRAFT

## Purpose

Evaluate NVIDIA nvCOMP as the GPU-side compression engine for OuterLink's transport layer. nvCOMP is the only production-ready GPU compression library and is critical for compressing VRAM-resident data at speeds that match or exceed network wire rates.

## Why GPU Compression Matters for OuterLink

As established in R14-01, CPU compression algorithms top out at ~0.7 GB/s (LZ4, single core) to ~5 GB/s (Zstd, 16 cores). OuterLink's RDMA path runs at 12.5 GB/s (100Gbps ConnectX-5). CPU compression is a bottleneck on fast links.

GPU compression changes the equation entirely. An A100 running nvCOMP LZ4 achieves **90+ GB/s compression** and **300+ GB/s decompression** -- far exceeding any network link speed. The GPU can compress data faster than the NIC can send it, making compression essentially "free" in terms of latency added.

## nvCOMP Overview

| Property | Value |
|---|---|
| **Developer** | NVIDIA |
| **License** | **Proprietary NVIDIA SDK** (NOT open source) |
| **Source Availability** | Closed source since v2.3 (headers + binary only) |
| **Supported Architectures** | Pascal (sm60)+, Volta (sm70)+ recommended |
| **Language** | C/C++ with C API suitable for FFI |
| **CUDA Integration** | Stream-based async, device-side API (nvCOMPDx) |
| **Python Bindings** | Official, interop with PyTorch/TensorFlow |

## Supported Algorithms

| Algorithm | Type | Best For | GPU-Specific |
|---|---|---|---|
| **LZ4** | General-purpose, byte-level | Wide range of data | No (standard format) |
| **Snappy** | General-purpose, byte-level | Tabular data | No (standard format) |
| **GDeflate** | GPU-optimized DEFLATE | Broad compatibility | Yes |
| **Zstandard** | Huffman + LZ77 + ANS | High ratio needs | No (standard format) |
| **Cascaded** | Run-length + delta + bitpacking | Structured/analytical data | Yes |
| **Bitcomp** | Proprietary, scientific data | Scientific computing | Yes (NVIDIA proprietary) |
| **gANS** | GPU-optimized ANS entropy coding | High-ratio needs | Yes |

## Benchmark Results (A100 GPU)

### LZ4 on A100

| Interface | Compression Ratio | Compression Throughput | Decompression Throughput |
|---|---|---|---|
| High-Level | 38.34x | 90.48 GB/s | 312.81 GB/s |
| Low-Level (chunked) | 38.89x | 95.87 GB/s | 320.70 GB/s |

*Dataset: Mortgage 2009Q2 (structured numerical data -- highly compressible)*

### Cascaded vs LZ4 on A100 (2000Q4 dataset)

| Algorithm | Compression Ratio | Compression Throughput | Decompression Throughput |
|---|---|---|---|
| Cascaded (auto) | 39.71x | 225.60 GB/s | 374.95 GB/s |
| LZ4 | 21.22x | 36.64 GB/s | 118.47 GB/s |

### Key Performance Notes

- H100 delivers up to **2.2x faster Zstd decompression** and **1.4x faster LZ4 decompression** vs A100
- GDeflate high-compression mode is up to **2x faster** in recent versions
- ANS achieves up to **7x higher throughput** for small files (~few MB) through strong scaling
- Low-level batch API significantly outperforms high-level API for many small buffers (OuterLink's typical workload pattern)

## Relevance to OuterLink

### The Speed Advantage

| Transport | Wire Speed | nvCOMP LZ4 Compress | nvCOMP LZ4 Decompress |
|---|---|---|---|
| ConnectX-5 RDMA (100Gbps) | 12.5 GB/s | 90+ GB/s | 312+ GB/s |
| USB4 v2 (80Gbps) | 10 GB/s | 90+ GB/s | 312+ GB/s |
| TCP + io_uring (25Gbps) | 3.1 GB/s | 90+ GB/s | 312+ GB/s |

GPU compression throughput exceeds wire speed by **7-100x**. Compression is effectively free on the GPU path. The 38x ratio on structured data means a 100Gbps link behaves like a 3.8 Tbps link for compressible workloads.

### Integration Architecture

```
Sender GPU:
  VRAM data -> nvCOMP compress (same GPU, async stream) -> compressed buffer in VRAM
  -> DMA to NIC (RDMA) or cudaMemcpy to host -> network

Receiver GPU:
  network -> NIC DMA to VRAM or cudaMemcpy from host -> compressed buffer in VRAM
  -> nvCOMP decompress (same GPU, async stream) -> original data in VRAM
```

### Impact on RDMA Zero-Copy (OpenDMA)

**Compression fundamentally breaks zero-copy.** With OpenDMA (BAR1 direct), the NIC reads directly from VRAM. Inserting compression means:

1. Application writes to VRAM buffer A
2. GPU compresses A -> B (a GPU kernel, uses SM compute)
3. NIC reads from B via BAR1

This adds one GPU kernel launch and one VRAM-to-VRAM copy. However, because nvCOMP compression is **7x faster than wire speed**, the net effect is still positive for compressible data. The GPU kernel completes before the NIC could have finished sending the uncompressed data.

**Trade-off matrix:**

| Scenario | Zero-Copy? | Latency | Throughput |
|---|---|---|---|
| Incompressible data, no compression | Yes (OpenDMA direct) | Minimal (~2 us) | Wire speed |
| Compressible data, with nvCOMP | No (compress then send) | +kernel launch (~5-10 us) | Effective throughput > wire speed |
| Small messages (<4KB) | Yes (skip compression) | Minimal | Wire speed |

The adaptive layer should detect compressibility and choose: raw OpenDMA for incompressible data, compress-then-send for compressible data.

## API Architecture for OuterLink Integration

### Low-Level Batch API (Recommended)

OuterLink should use the **low-level (chunked) batch API**, not the high-level manager API:

- OuterLink manages its own VRAM buffers already (CUDA memory interception)
- Batch API processes multiple chunks in parallel (matches OuterLink's scatter-gather pattern)
- Higher throughput for many small-to-medium buffers
- Full control over CUDA streams for async pipeline integration

### Device-Side API (nvCOMPDx)

nvCOMPDx allows calling compression/decompression from **within a CUDA kernel**. This is interesting for:
- Fusing compression with data preparation (e.g., gradient packing + compression in one kernel)
- Reducing kernel launch overhead
- Integration with custom CUDA kernels OuterLink may use

### Rust FFI Strategy

No existing Rust crate for nvCOMP. Strategy:

1. **Generate bindings** with `bindgen` against nvCOMP C API headers
2. **Link against** pre-built nvCOMP shared library (`libnvcomp.so`)
3. **Create safe Rust wrapper** (`outerlink-nvcomp` or within `outerlink-common`)
4. **Buffer management**: nvCOMP operates on device pointers; OuterLink already manages these via CUDA driver API interception

Key C API functions to wrap:
- `nvcompBatchedLZ4CompressAsync` / `nvcompBatchedLZ4DecompressAsync`
- `nvcompBatchedLZ4CompressGetTempSize` / `nvcompBatchedLZ4CompressGetMaxOutputChunkSize`
- Equivalent for Zstd, Cascaded, Bitcomp as needed

## Licensing Concern

**nvCOMP is proprietary.** Since v2.3, source code is not released. This creates a dependency on NVIDIA's binary distribution.

**Impact on OuterLink (Apache 2.0):**
- OuterLink can link against nvCOMP as a runtime dependency (similar to linking against `libcuda.so`)
- nvCOMP must be installed separately by the user (like CUDA itself)
- OuterLink's code remains Apache 2.0; the nvCOMP integration is via dynamic linking
- This is the same model as NCCL, cuDNN, etc.

**Risk mitigation:** nvCOMP algorithms that produce standard formats (LZ4, Zstd) allow CPU-side decompression as fallback. If nvCOMP is unavailable, OuterLink can fall back to CPU compression with `lz4_flex`/`zstd` crates at reduced throughput.

## Algorithm Selection for OuterLink

| Data Type | Recommended nvCOMP Algorithm | Rationale |
|---|---|---|
| **CUDA memory (general)** | LZ4 | Standard format, fast, good general ratio |
| **Structured GPU data (tensors, buffers)** | Cascaded | Highest throughput (225 GB/s) for structured data |
| **Model weights** | Zstd or Bitcomp | Better ratio for weight-like data |
| **Gradient buffers** | Cascaded or Bitcomp | Structured numerical data, high compressibility |
| **Texture/image data** | GDeflate | Good ratio for image-like data |

## Related Documents

- [R14-01: CPU Compression Algorithms](./01-cpu-compression-algorithms.md) -- CPU-side alternatives
- [R14-03: Gradient Compression Techniques](./03-gradient-compression-techniques.md) -- ML-specific compression
- [R10: Memory Tiering](../../R10-memory-tiering/) -- Compression per tier
- [R14 Pre-Plan](../preplan.md) -- Scope and implementation decisions

## Open Questions

1. **What is nvCOMP performance on GeForce GPUs (non-datacenter)?** Benchmarks are on A100/H100. OuterLink targets consumer hardware too. Need to measure on RTX 3090/4090.
2. **nvCOMP version compatibility:** Which minimum version of nvCOMP should OuterLink target? v4.0 is latest but requires recent CUDA toolkit.
3. **Compression ratio on actual CUDA application data:** Benchmarks use financial/analytical datasets. How well does VRAM content from games, simulations, ML training actually compress?
4. **Can nvCOMP and compute kernels share the GPU?** If the application is fully utilizing SMs, compression kernels may be starved. Need to investigate CUDA stream priorities and MPS.
5. **nvCOMPDx viability:** Is the device-side API mature enough for production use? It could eliminate kernel launch overhead.
6. **Cascaded auto-tuning overhead:** Cascaded algorithm has an "auto" mode that selects parameters. What is the cost of this auto-selection, and can it be cached for repeated data patterns?
