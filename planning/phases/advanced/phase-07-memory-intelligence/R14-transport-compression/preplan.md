# R14: Transport-Layer Compression -- Pre-Plan

**Date Created:** 2026-03-25
**Date Last Updated:** 2026-03-25
**Status:** DRAFT

## Purpose

Define the scope, dependencies, decision points, risks, and implementation phases for adding pluggable compression to OuterLink's transport layer. This pre-plan covers WHAT needs to be planned before writing the detailed implementation plan.

## Scope Definition

### In Scope

1. **General-purpose lossless compression** on the transport path (LZ4, Zstd via CPU; nvCOMP on GPU)
2. **Adaptive compression selection** -- detect compressibility, choose algorithm, or skip compression
3. **GPU-native compression pipeline** -- nvCOMP integration for VRAM-to-wire compression
4. **Pluggable compression trait** -- Rust trait abstracting compression algorithm selection
5. **Per-stream compression configuration** -- different strategies per data type / link condition
6. **Compression bypass for small messages** -- skip compression below a configurable threshold (likely 4-16 KB)
7. **Metrics and observability** -- compression ratio, throughput, CPU/GPU utilization per stream
8. **Pipelining** -- overlap compression with network I/O (compress chunk N while sending chunk N-1)

### Out of Scope (Handled by Other Research Topics)

| Topic | Owner | Interaction |
|---|---|---|
| Gradient-specific compression (Top-K, PowerSGD, quantization) | **R20 (NCCL Backend)** | R14 provides the compression infrastructure; R20 uses it for collectives |
| Per-tier compression policy (which tier compresses what) | **R10 (Memory Tiering)** | R10 decides policy; R14 provides the mechanism |
| Scatter-gather + compression fusion | **R28 (Scatter-Gather DMA)** | R28 integrates compression into DMA scatter patterns |
| NCCL compressed collective operations | **R20 (NCCL Backend)** | R20 builds compressed AllReduce on top of R14's compression |

### Boundary: R14 vs R20

R14 implements **transport-layer compression**: any buffer going over the wire can be compressed, regardless of what it contains. R20 implements **semantic compression**: understanding that a buffer contains gradients and applying domain-specific techniques (sparsification, quantization). R14 provides the plumbing; R20 provides the intelligence.

## Dependencies

### Upstream (R14 Depends On)

| Dependency | Status | Why Needed |
|---|---|---|
| **P6: Core Transport** | Must be working | Compression sits on top of the transport layer |
| **CUDA Driver API interception** | Done (P1-P3) | nvCOMP integration uses CUDA streams from the intercepted context |
| **nvCOMP SDK** | External dependency | Must be installed on systems with GPUs (runtime, not build-time) |
| **Rust lz4/zstd crates** | Available on crates.io | CPU compression path |

### Downstream (Depends on R14)

| Dependent | What It Needs |
|---|---|
| **R20: NCCL Backend** | Compression API to build compressed collectives |
| **R10: Memory Tiering** | Compression mechanism for tier-specific policies |
| **R28: Scatter-Gather DMA** | Compression integration for scatter operations |
| **R9: Multi-Transport** | Per-transport compression configuration |

## Decision Inventory

These decisions must be made during planning. Each needs research backing.

### D1: Where Does Compression Happen?

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **Sender-side only** | Simpler, receiver does less work | Receiver must always decompress | Baseline |
| **Receiver-side only** | Sender stays fast | Doesn't reduce wire traffic (nonsensical) | Rejected |
| **Both sides (symmetric)** | Required for collectives (AllReduce) | More complex | Required for R20 |

**Research finding:** Sender compresses, receiver decompresses. Both sides need the compression library. For collectives, intermediate nodes may decompress-accumulate-recompress.

### D2: GPU Compression vs CPU Compression

| Data Location | Compression Engine | Rationale |
|---|---|---|
| VRAM | nvCOMP (GPU) | 90+ GB/s, matches wire speed. CPU cannot keep up. |
| Host DRAM | LZ4/Zstd (CPU) | Data already on host. GPU roundtrip would be slower. |
| VRAM on slow link (TCP) | Either | CPU might suffice for 3.1 GB/s TCP. GPU preferred if available. |

**Decision:** Use GPU compression for VRAM-resident data, CPU compression for host-resident data. When both are possible, prefer GPU (faster, doesn't consume CPU cores needed for io_uring/networking).

### D3: Which Algorithms?

Based on research (R14-01, R14-02):

| Path | Algorithm | Why |
|---|---|---|
| GPU general | nvCOMP LZ4 | Standard format, fast, fallback to CPU decompression possible |
| GPU structured data | nvCOMP Cascaded | 225 GB/s compress, best for numerical/tensor data |
| CPU general | LZ4 (`lz4_flex` crate) | Fastest CPU algorithm, pure Rust |
| CPU high-ratio | Zstd level 1 (`zstd` crate) | Better ratio than LZ4, still fast |
| Small metadata | Zstd with dictionary | Trained dictionaries improve small-buffer compression |

### D4: Adaptive vs Fixed Compression

**Decision: Adaptive.** Fixed compression wastes cycles on incompressible data and under-compresses highly compressible data.

Adaptive strategy:
1. **Sample first chunk** of a stream: compress a small sample (1-4 KB) and measure ratio
2. **If ratio > threshold (e.g., 1.5x):** enable compression for this stream
3. **If ratio < threshold:** send raw, re-evaluate periodically
4. **Cache decision per stream type:** once a memory region's compressibility is known, reuse the decision
5. **Entropy estimation as fast pre-filter:** count distinct byte values in a sample. High entropy (>7.5 bits/byte) = incompressible = skip.

Cost of adaptivity: one small trial compression per new stream. At nvCOMP speeds (90+ GB/s), compressing a 4KB sample takes ~44 nanoseconds. Negligible.

### D5: Chunk Size

| Chunk Size | Compression Ratio | Throughput | Latency |
|---|---|---|---|
| 4 KB | Poor (small window) | Low (kernel launch overhead) | Lowest |
| 64 KB | Good | Good | Low |
| 256 KB | Best | Best | Medium |
| 1 MB | Best | Best | Higher |

**Decision:** Default chunk size of **64-256 KB**, configurable per transport. Smaller for latency-sensitive paths (RDMA), larger for throughput-oriented paths (TCP).

### D6: Protocol Wire Format

Compressed messages need a header:

```
[1 byte: flags] [3 bytes: original_size] [compressed_payload]

Flags:
  bit 0: compressed (0=raw, 1=compressed)
  bit 1-3: algorithm (0=LZ4, 1=Zstd, 2=nvCOMP-Cascaded, 3=nvCOMP-Bitcomp, 4-7=reserved)
  bit 4-7: reserved
```

Total overhead: 4 bytes per chunk. For 64KB chunks, overhead is 0.006%. For algorithm negotiation, sender and receiver handshake supported algorithms at connection time.

## Risk Assessment

### R1: Compression Overhead Exceeds Savings (MEDIUM)

**Risk:** For incompressible data or very fast links, compression adds latency without reducing transfer time.

**Mitigation:** Adaptive compression with fast compressibility detection. Skip compression when ratio < 1.5x. The sample-based approach costs ~44ns on GPU, ~5us on CPU -- negligible.

### R2: nvCOMP Unavailability (LOW)

**Risk:** nvCOMP is proprietary and may not be installed, may change licensing, or may not support target GPU.

**Mitigation:** All GPU compression paths fall back to CPU compression (LZ4/Zstd). nvCOMP uses standard formats (LZ4, Zstd) so CPU can decompress GPU-compressed data. OuterLink works without nvCOMP, just slower.

### R3: GPU Compute Contention (MEDIUM)

**Risk:** nvCOMP kernels compete with application CUDA kernels for GPU SMs. Compression could slow down the application.

**Mitigation:**
- Use low-priority CUDA streams for compression kernels
- Monitor SM utilization; throttle compression when GPU is busy
- For latency-critical paths, prefer CPU compression even if slower overall
- CUDA MPS can partition SMs between application and compression

### R4: Compression Breaks Zero-Copy RDMA (LOW-MEDIUM)

**Risk:** OpenDMA (BAR1 direct) is zero-copy NIC-to-VRAM. Compression requires a copy (compress to separate buffer).

**Mitigation:** This is inherent and acceptable. The extra kernel launch (~5-10us) is offset by reduced wire time for compressible data. For incompressible data, adaptive detection skips compression and preserves zero-copy. The adaptive layer makes this transparent.

### R5: Memory Pressure from Compression Buffers (LOW)

**Risk:** Compression requires temporary buffers (input + output). For many concurrent streams, this adds VRAM/DRAM pressure.

**Mitigation:**
- Pre-allocate a compression buffer pool (configurable size)
- Reuse buffers across streams (ring buffer pattern)
- nvCOMP temp buffer sizes are known at setup time
- Worst case: compression buffer pool is bounded, excess streams wait

### R6: Latency Regression for Small Transfers (LOW)

**Risk:** Compression adds latency to small messages that don't benefit from compression.

**Mitigation:** Configurable minimum size threshold (default: 4-16 KB). Messages below threshold are always sent raw. This is a simple size check with zero overhead.

## Implementation Phases

### Phase R14-A: Compression Trait and CPU Path (2-3 weeks)

**Goal:** Pluggable compression infrastructure with CPU algorithms working.

1. Define `Compressor` trait in `outerlink-common`:
   - `compress(input: &[u8], output: &mut Vec<u8>) -> Result<CompressionResult>`
   - `decompress(input: &[u8], output: &mut Vec<u8>) -> Result<()>`
   - `estimate_compressibility(sample: &[u8]) -> f32`
2. Implement LZ4 compressor (using `lz4_flex`)
3. Implement Zstd compressor (using `zstd` crate)
4. Wire into transport layer with compression/decompression at send/receive boundaries
5. Add compression bypass for small messages
6. Add wire format header (4-byte overhead per chunk)
7. Unit tests and benchmarks

### Phase R14-B: GPU Compression (nvCOMP) (3-4 weeks)

**Goal:** GPU-native compression for VRAM-resident data.

1. Create nvCOMP FFI bindings (bindgen against C API headers)
2. Implement nvCOMP compressor behind the `Compressor` trait (async, stream-based)
3. Buffer pool management for compression temp buffers
4. Integration with CUDA stream management (from P1-P3 interception)
5. Fallback to CPU compression when nvCOMP unavailable
6. Benchmarks on target hardware (RTX 3090/4090 + A100 if available)

### Phase R14-C: Adaptive Selection (2 weeks)

**Goal:** Automatic algorithm and compression-on/off selection.

1. Implement compressibility sampling (entropy estimation + trial compression)
2. Per-stream compression state machine (probe -> decide -> steady-state -> re-probe)
3. Algorithm selection logic (GPU vs CPU, LZ4 vs Cascaded vs Zstd)
4. Configuration: thresholds, chunk sizes, algorithm preferences per transport
5. Metrics: compression ratio, throughput, decision distribution

### Phase R14-D: Pipelining and Optimization (2-3 weeks)

**Goal:** Overlap compression with network I/O for maximum throughput.

1. Chunk pipeline: compress chunk N while sending chunk N-1
2. Double-buffering for GPU compression (ping-pong buffers)
3. Integration with io_uring submission queues (TCP path)
4. Integration with RDMA verbs (RDMA path)
5. End-to-end benchmarks: measure actual throughput improvement per transport type
6. Tuning: chunk size, buffer pool size, thread count

## Success Criteria

| Metric | Target |
|---|---|
| **Effective throughput increase** | 2x+ for compressible data (gradients, structured buffers) |
| **Latency regression for incompressible data** | < 1% (adaptive bypass works) |
| **Latency regression for small messages** | 0% (size threshold bypass) |
| **CPU overhead (CPU compression path)** | < 2 cores dedicated to compression |
| **GPU overhead (nvCOMP path)** | < 5% SM utilization for compression |
| **Compression works with all transports** | TCP, RDMA, USB4 all supported |
| **Fallback works** | System functions without nvCOMP (CPU-only mode) |

## Open Questions

1. **Should compression be opt-in or opt-out?** Default-on with adaptive bypass is recommended, but some users may want to guarantee zero compression overhead.
2. **How to handle compression across heterogeneous nodes?** Node A has GPU (nvCOMP), Node B has CPU only. Sender compresses with nvCOMP LZ4, receiver decompresses with CPU LZ4. Standard formats make this work, but need to verify interop.
3. **Compression dictionary sharing:** Can sender train a Zstd dictionary on representative data and share it with the receiver for better small-buffer compression? Protocol needs dictionary negotiation.
4. **Integration point in transport layer:** Does compression wrap the transport (compress-then-send) or is it embedded in the transport (compress-within-send)? The former is simpler; the latter allows tighter pipelining.
5. **Testing strategy:** How to benchmark compression benefit across diverse CUDA workloads? Need a suite of representative memory patterns (ML training, rendering, simulation, general compute).

## Related Documents

- [R14-01: CPU Compression Algorithms](./research/01-cpu-compression-algorithms.md)
- [R14-02: GPU-Native Compression (nvCOMP)](./research/02-gpu-native-compression-nvcomp.md)
- [R14-03: Gradient Compression Techniques](./research/03-gradient-compression-techniques.md)
- [R14 README](./README.md)
- [R10: Memory Tiering Pre-Plan](../R10-memory-tiering/preplan.md)
- [R20: NCCL Backend Pre-Plan](../R20-nccl-backend/preplan.md)
