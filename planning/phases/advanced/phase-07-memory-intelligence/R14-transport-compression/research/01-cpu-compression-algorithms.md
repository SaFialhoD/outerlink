# R14 Research: CPU Compression Algorithms for Network Transport

**Date Created:** 2026-03-25
**Date Last Updated:** 2026-03-25
**Status:** DRAFT

## Purpose

Evaluate CPU-side lossless compression algorithms (LZ4, Zstd, Snappy) for suitability in OuterLink's transport layer, focusing on whether they can keep up with network wire speeds (12.5 GB/s for 100Gbps ConnectX-5, 10 GB/s for USB4 v2, ~3.1 GB/s for 25Gbps TCP).

## The Core Question

OuterLink needs compression that is **faster than the network**. If compression throughput < wire speed, compression adds latency instead of saving it. The break-even equation is:

```
compression_time + (compressed_size / wire_speed) < (original_size / wire_speed)
```

Rearranged: compression is worthwhile when:

```
throughput_compress > wire_speed / (1 - 1/ratio)
```

For a 100Gbps link (12.5 GB/s) with a modest 2:1 ratio, the compressor must sustain **25 GB/s** to break even on a single stream. For a 25Gbps TCP link (3.1 GB/s) with 2:1, the compressor needs ~6.2 GB/s. These are **per-stream** requirements.

## Algorithm Comparison

### LZ4

| Property | Value |
|---|---|
| **Type** | LZ77-family, byte-level, no entropy coding |
| **Compression Speed (single core)** | 500-800 MB/s |
| **Decompression Speed (single core)** | 1-5 GB/s (~1 byte/cycle) |
| **Compression Ratio (typical)** | 2.0-2.5x on structured data, 1.1-1.5x on random-ish data |
| **Multithreading** | Supported since v1.10.0 (5-10x boost on multi-core) |
| **Memory Usage** | ~16 KB for fast mode, ~64 KB dictionary |
| **Streaming** | Yes, frame format supports streaming |
| **Rust Crate** | `lz4_flex` (pure Rust), `lz4` (C bindings) |

**Strengths for OuterLink:**
- Decompression is extremely fast (receiver-side is cheap)
- Minimal memory footprint (good for many concurrent streams)
- Zero-allocation fast mode available
- Well-tested in networked systems (Kafka, Cassandra, ElasticSearch default)

**Weaknesses for OuterLink:**
- Single-core compression (500-800 MB/s) is far below 100Gbps wire speed (12.5 GB/s)
- Even 4-thread compression (~2-3 GB/s) insufficient for 100Gbps
- Compression ratio modest for non-structured data
- Would need 4-16 cores dedicated to compression to match wire speed

### Zstd (Zstandard)

| Property | Value |
|---|---|
| **Type** | LZ77 + Huffman + FSE (Finite State Entropy) |
| **Compression Speed (level 1, single core)** | ~340 MB/s |
| **Compression Speed (level 1, 4 threads)** | ~1.4 GB/s |
| **Decompression Speed (single core)** | 500-1500 MB/s |
| **Compression Ratio (level 1)** | 2.5-3.0x on structured data |
| **Compression Ratio (level 3, default)** | 3.0-3.5x |
| **Multithreading** | Native multi-threaded compression |
| **Memory Usage** | Higher than LZ4 (~128 KB - several MB depending on level) |
| **Streaming** | Yes, native streaming API |
| **Rust Crate** | `zstd` (C bindings), `zstd-safe` |

**Strengths for OuterLink:**
- Best ratio-to-speed tradeoff available
- Level 1 approaches LZ4 speed with significantly better ratios
- Dictionary mode: pre-train on data patterns (e.g., gradient distributions) for better ratio on small buffers
- Native streaming reduces latency vs block-only compression

**Weaknesses for OuterLink:**
- Even level 1 at ~340 MB/s single-core is far below wire speed
- Multi-thread scaling helps but still maxes ~5-8 GB/s with 16 threads
- Higher memory usage per stream
- Decompression slower than LZ4

### Snappy

| Property | Value |
|---|---|
| **Type** | LZ77 variant, designed by Google |
| **Compression Speed (single core)** | ~250 MB/s |
| **Decompression Speed (single core)** | ~500 MB/s |
| **Compression Ratio** | ~2.0x on structured data |
| **Multithreading** | Not native |
| **Rust Crate** | `snap` (pure Rust) |

**Verdict:** Strictly inferior to LZ4 on all metrics (confirmed by lzbench benchmarks). Lower speed, lower ratio, no multi-threading. Legacy choice being phased out.

**Recommendation:** Do not consider Snappy for OuterLink. LZ4 dominates on speed; Zstd dominates on ratio.

## Throughput vs Wire Speed Analysis

| Transport | Wire Speed | LZ4 1-core | LZ4 4-core | Zstd-1 1-core | Zstd-1 4-core | Zstd-1 16-core |
|---|---|---|---|---|---|---|
| **ConnectX-5 RDMA (100Gbps)** | 12.5 GB/s | 0.7 GB/s | ~2.8 GB/s | 0.34 GB/s | ~1.4 GB/s | ~5.4 GB/s |
| **USB4 v2 (80Gbps)** | 10 GB/s | 0.7 GB/s | ~2.8 GB/s | 0.34 GB/s | ~1.4 GB/s | ~5.4 GB/s |
| **TCP + io_uring (25Gbps)** | 3.1 GB/s | 0.7 GB/s | ~2.8 GB/s | 0.34 GB/s | ~1.4 GB/s | ~5.4 GB/s |

### Key Insight: CPU Compression Cannot Keep Up With High-Speed Links

At 100Gbps, no CPU compression algorithm can match wire speed on reasonable core counts. This leads to two strategies:

1. **Use GPU compression (nvCOMP)** for VRAM-resident data going over fast links (see R14-02)
2. **Use CPU compression only on slower links** (TCP) or when CPU cores are plentiful
3. **Use CPU compression for specific data types** that are highly compressible (10x+ ratio makes even slow compression worthwhile)

### When CPU Compression Wins

For the 25Gbps TCP link, LZ4 with 4 cores (~2.8 GB/s) is close to wire speed. If data compresses 2x, effective throughput becomes ~5.6 GB/s on a 3.1 GB/s link. The math works:

```
Time without compression: 1 GB / 3.1 GB/s = 322 ms
Time with LZ4 (4-core, 2x ratio): compress 1 GB at 2.8 GB/s (357 ms) + send 0.5 GB at 3.1 GB/s (161 ms) = 518 ms
```

Wait, that is worse. Compression only wins when pipelined (compress chunk N while sending chunk N-1):

```
Pipelined: max(compress_time, send_time) per chunk
= max(chunk/2.8, chunk*0.5/3.1) per chunk
= effectively limited by the slower of compression or sending compressed data
```

With 2x ratio and 4-core LZ4, the bottleneck shifts to compression at 2.8 GB/s effective. But the effective throughput through the wire is doubled. **Compression wins when it can be overlapped with network I/O via pipelining.**

For high-ratio data (gradients at 10x+), even single-core LZ4 is a massive win on any link:

```
1 GB gradients, 10x ratio, 100Gbps link:
Without: 1 GB / 12.5 GB/s = 80 ms
With LZ4: compress at 0.7 GB/s (overlap) + send 0.1 GB / 12.5 GB/s = ~8 ms wire time
Pipelined effective: ~1.43 s total for 1 GB (limited by compression), but only 0.1 GB on wire
```

The key insight: **compression's value is in reducing wire utilization**, not reducing total transfer time for a single stream. When multiple streams share a link, compression reduces contention.

## Recommended Strategy for OuterLink

| Data Type | Algorithm | Rationale |
|---|---|---|
| **VRAM data over RDMA (100Gbps)** | nvCOMP on GPU (see R14-02) | CPU too slow for wire speed |
| **DRAM data over TCP (25Gbps)** | LZ4 (multi-threaded) | Fast enough to pipeline with TCP |
| **High-ratio data (gradients, sparse)** | Zstd level 1 or domain-specific | Ratio matters more than raw speed |
| **Small messages (<4KB)** | None | Compression overhead exceeds savings |
| **Control plane / metadata** | Zstd with dictionary | Small payloads benefit from trained dictionaries |

## Rust Ecosystem

| Crate | Type | Performance Notes |
|---|---|---|
| `lz4_flex` | Pure Rust LZ4 | ~80-90% of C LZ4 speed, no unsafe dependencies |
| `lz4` | C bindings | Full C performance, requires system lib |
| `zstd` | C bindings (libzstd) | Full performance, well-maintained |
| `zstd-safe` | Safe Rust wrapper over zstd | Good for OuterLink's safety preferences |
| `snap` | Pure Rust Snappy | Not recommended (inferior to LZ4) |

**Recommendation:** Use `lz4_flex` for the LZ4 path (pure Rust, no C dependency, good performance) and `zstd` crate for the Zstd path (C bindings acceptable for mature, stable library).

## Related Documents

- [R14-02: GPU-Native Compression (nvCOMP)](./02-gpu-native-compression-nvcomp.md) -- GPU compression for VRAM data
- [R14-03: Gradient Compression Techniques](./03-gradient-compression-techniques.md) -- ML-specific compression
- [R10: Memory Tiering](../../R10-memory-tiering/) -- Which tier benefits most from compression
- [R14 Pre-Plan](../preplan.md) -- Scope and implementation decisions

## Open Questions

1. **What is LZ4 throughput on the actual OuterLink server hardware?** Benchmarks vary by CPU generation. Need to measure on the target system.
2. **Is `lz4_flex` performance sufficient, or do we need C bindings?** The ~10-20% penalty may matter at scale.
3. **Can we use SIMD-accelerated compression?** Some LZ4/Zstd implementations use AVX2/AVX-512 for higher throughput. Rust support for explicit SIMD is evolving.
4. **What chunk size optimizes the compression-to-network pipeline?** Too small = poor ratio + overhead. Too large = latency. Likely 64KB-256KB range.
5. **Dictionary training for Zstd:** Can we pre-train dictionaries on typical CUDA memory patterns for better small-buffer compression?
