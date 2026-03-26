# R14: Transport-Layer Compression -- Pre-Plan v2 (Refined)

**Date Created:** 2026-03-25
**Date Last Updated:** 2026-03-25
**Status:** DRAFT
**Revision:** v2 -- Cross-topic refinement. Resolves open questions from v1, defines exact CompressionHook implementation matching R10 v2's trait, adaptive decision tree with concrete thresholds, wire format specification, nvCOMP FFI design, and per-tier compression strategy.
**Supersedes:** `preplan.md` (v1, preserved for reference)

## Purpose

Second-round refinement of the R14 pre-plan. Integrates findings from R10 (Memory Tiering), R11 (Prefetching), R12 (Dedup), R17 (Multi-Path), R20 (NCCL Backend), R21 (NVMe Tiers), R28 (Scatter-Gather), and R30 (Persistent Kernels) to produce a precise implementation blueprint for transport-layer compression.

---

## 1. Resolved Open Questions from v1

### Q1: Should compression be opt-in or opt-out?

**RESOLVED: Default-on with adaptive bypass. Opt-out via config.**

Rationale:
- R10 v2 shows that tier migration uses a `compress: bool` field on `MigrationRequest`. The migration engine defaults this to `true` when crossing tiers with bandwidth differentials > 2x.
- R17 multi-path striping makes per-link decisions. Fast RDMA links skip compression; slow TCP links enable it. This is adaptive, not opt-in.
- User override: `OUTERLINK_COMPRESSION=off` environment variable disables all compression. `OUTERLINK_COMPRESSION=cpu_only` disables GPU compression but keeps CPU path.

### Q2: How to handle compression across heterogeneous nodes?

**RESOLVED: Standard-format algorithms only on the wire. GPU-specific formats stay local.**

Rationale:
- nvCOMP LZ4 and nvCOMP Zstd produce standard LZ4/Zstd frames. A GPU-compressed buffer can be CPU-decompressed on the receiver. This is verified in nvCOMP documentation.
- nvCOMP Cascaded and Bitcomp are GPU-proprietary formats. These are used only for in-tier storage (e.g., compressing a page in VRAM to save space) where both compress and decompress happen on the same GPU.
- Wire protocol advertises supported algorithms via capability bits (see R20 v2 handle format, bit 0: `supports_compression`). Sender picks the best algorithm the receiver can decompress.

Algorithm negotiation at connection time:
```
Sender capabilities: [LZ4_CPU, LZ4_GPU, ZSTD_CPU, ZSTD_GPU, CASCADED_GPU]
Receiver capabilities: [LZ4_CPU, ZSTD_CPU]
Negotiated set: [LZ4, ZSTD]  (standard formats both sides can handle)
```

### Q3: Compression dictionary sharing?

**RESOLVED: Deferred to Phase R14-D optimization. Not in initial implementation.**

Rationale:
- Zstd dictionary training requires representative samples and adds protocol complexity (dictionary sync, versioning).
- The primary wins come from LZ4/nvCOMP without dictionaries. Dictionary support is an optimization for small metadata buffers (< 4 KB), which are already below the compression bypass threshold.
- If implemented later, dictionaries are negotiated during connection handshake and cached per peer.

### Q4: Integration point -- compress-then-send or compress-within-send?

**RESOLVED: Compress-within-send (embedded in transport pipeline).**

Rationale:
- R28 (Scatter-Gather) finding: gather first, then compress the contiguous buffer. This means compression sits between the gather stage and the send stage.
- R30 (Persistent Kernels) finding: compress/decompress in VRAM ring buffer pipeline. Compression is a pipeline stage, not a wrapper.
- The transport pipeline becomes: `gather -> compress -> send` on the sender, `recv -> decompress -> scatter` on the receiver.
- This allows double-buffering: compress chunk N while sending chunk N-1, which is only possible when compression is a pipeline stage.

### Q5: Testing strategy?

**RESOLVED: Three-tier benchmark suite.**

1. **Microbenchmarks:** Raw compression throughput per algorithm per data type (random, zeros, gradients, textures, weights). Run on target hardware (RTX 3090, RTX 4090).
2. **Transport benchmarks:** End-to-end throughput with compression enabled/disabled across TCP, RDMA, USB4. Measure effective throughput, CPU utilization, GPU SM utilization.
3. **Application benchmarks:** Run real CUDA workloads (PyTorch training, CUDA samples) with and without compression. Measure wall-clock time, memory transfer reduction, GPU utilization impact.

---

## 2. CompressionHook Implementation

R10 v2 defines this trait that R14 must implement:

```rust
/// R14 compression hook, called during migration.
/// Defined in R10 v2 preplan-v2.md, implemented by R14.
pub trait CompressionHook: Send + Sync {
    /// Attempt to compress page data. Returns compressed data and ratio,
    /// or None if compression is not beneficial (ratio < 1.5x).
    fn try_compress(&self, data: &[u8]) -> Option<(Vec<u8>, f32)>;

    /// Decompress page data.
    fn decompress(&self, compressed: &[u8], original_size: usize) -> Vec<u8>;
}
```

R14 implements this with a broader interface that also serves the transport layer. The `CompressionHook` trait is satisfied by delegating to the full compression engine.

### Full Compression Engine Trait (R14 owns this)

```rust
/// Algorithm identifier. Matches wire format algorithm field.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompressionAlgorithm {
    None      = 0,
    Lz4Cpu    = 1,
    ZstdCpu   = 2,
    Lz4Gpu    = 3,   // nvCOMP LZ4 (standard format, GPU-accelerated)
    ZstdGpu   = 4,   // nvCOMP Zstd (standard format, GPU-accelerated)
    Cascaded  = 5,   // nvCOMP Cascaded (GPU-only format, local storage only)
    Bitcomp   = 6,   // nvCOMP Bitcomp (GPU-only format, local storage only)
}

/// Result of a compression operation.
pub struct CompressionResult {
    pub algorithm: CompressionAlgorithm,
    pub compressed_data: Vec<u8>,
    pub original_size: u32,
    pub compressed_size: u32,
    pub ratio: f32,              // original / compressed
    pub compress_time_ns: u64,   // for metrics
}

/// Decision from the adaptive engine.
pub struct CompressionDecision {
    pub algorithm: CompressionAlgorithm,
    pub estimated_ratio: f32,
    pub estimated_compress_throughput_gbps: f32,
    pub reason: &'static str,   // human-readable reason for observability
}

/// Full compression engine. Transport layer and migration engine use this.
pub trait CompressionEngine: Send + Sync {
    /// Decide whether and how to compress, given context.
    fn should_compress(&self, ctx: &CompressionContext) -> CompressionDecision;

    /// Compress data using the specified algorithm.
    /// Returns None if compression is not beneficial (ratio < threshold).
    fn compress(
        &self,
        data: &[u8],
        algorithm: CompressionAlgorithm,
    ) -> Option<CompressionResult>;

    /// Decompress data. Algorithm is encoded in the wire header.
    fn decompress(
        &self,
        compressed: &[u8],
        algorithm: CompressionAlgorithm,
        original_size: usize,
    ) -> Vec<u8>;

    /// Fast compressibility estimate without full compression.
    /// Returns estimated ratio (1.0 = incompressible).
    fn estimate_ratio(&self, sample: &[u8]) -> f32;

    /// Report supported algorithms on this node.
    fn supported_algorithms(&self) -> &[CompressionAlgorithm];
}

/// Context for compression decisions.
pub struct CompressionContext {
    pub data_size: usize,
    pub source_tier: TierId,
    pub dest_tier: TierId,
    pub link_bandwidth_gbps: f32,    // current link bandwidth
    pub link_utilization: f32,        // 0.0-1.0
    pub gpu_available: bool,          // nvCOMP loaded and GPU has spare SMs
    pub gpu_sm_utilization: f32,      // 0.0-1.0, from NVML
    pub cached_ratio: Option<f32>,    // from previous compression of similar data
    pub is_dedup_page: bool,          // R12: deduped pages are read-only, one-time compress
}
```

### CompressionHook Adapter

The `CompressionHook` trait (from R10) is implemented by wrapping the `CompressionEngine`:

```rust
/// Adapter that satisfies R10's CompressionHook using R14's full engine.
pub struct TransportCompressionHook {
    engine: Arc<dyn CompressionEngine>,
    /// Default context for tier migration (no link info needed).
    default_ctx: CompressionContext,
}

impl CompressionHook for TransportCompressionHook {
    fn try_compress(&self, data: &[u8]) -> Option<(Vec<u8>, f32)> {
        let decision = self.engine.should_compress(&self.default_ctx);
        if decision.algorithm == CompressionAlgorithm::None {
            return None;
        }
        self.engine.compress(data, decision.algorithm)
            .map(|r| (r.compressed_data, r.ratio))
    }

    fn decompress(&self, compressed: &[u8], original_size: usize) -> Vec<u8> {
        // Algorithm is stored in the compressed page header (first byte after magic).
        let algorithm = parse_algorithm_from_header(compressed);
        self.engine.decompress(compressed, algorithm, original_size)
    }
}
```

Registration with R10's migration engine:

```rust
// During server initialization:
let compression_engine = Arc::new(AdaptiveCompressionEngine::new(config));
let hook = Box::new(TransportCompressionHook {
    engine: compression_engine.clone(),
    default_ctx: CompressionContext::for_tier_migration(),
});
tier_manager.set_compression_hook(hook);
```

---

## 3. Adaptive Compression Decision Tree

### Input Parameters

| Parameter | Source | Range |
|---|---|---|
| `data_size` | Caller | 0 - 64 KB (page) or 0 - N MB (transport buffer) |
| `source_tier` | R10 PTE | 0-5 (VRAM local/remote, DRAM local/remote, NVMe local/remote) |
| `dest_tier` | Migration request | 0-5 |
| `link_bandwidth_gbps` | R17 topology discovery | 0.1 - 100 |
| `link_utilization` | Transport layer metrics | 0.0 - 1.0 |
| `gpu_available` | nvCOMP probe at startup | bool |
| `gpu_sm_utilization` | NVML polling (1s interval) | 0.0 - 1.0 |
| `cached_ratio` | Per-alloc-id cache | None or 1.0 - 100.0 |
| `is_dedup_page` | R12 PTE flag (DEDUP_CANONICAL) | bool |

### Decision Algorithm (Pseudocode)

```
function should_compress(ctx: CompressionContext) -> CompressionDecision:

    // GATE 1: Size threshold — skip tiny buffers
    if ctx.data_size < 4096:  // 4 KB minimum
        return Decision(None, reason="below_size_threshold")

    // GATE 2: Use cached decision if available
    if ctx.cached_ratio is Some(ratio):
        if ratio < 1.3:
            return Decision(None, reason="cached_incompressible")
        // Else: use cached ratio to pick algorithm, skip sampling

    // GATE 3: Fast entropy estimation (256-byte sample)
    sample = data[0..256]
    distinct_bytes = count_distinct(sample)
    if distinct_bytes > 240:  // >7.5 bits/byte entropy
        return Decision(None, reason="high_entropy")

    // GATE 4: Dedup pages always compress (one-time cost, stored compressed)
    if ctx.is_dedup_page:
        return pick_best_algorithm(ctx, force=true)

    // GATE 5: Tier-pair specific logic
    match (ctx.source_tier, ctx.dest_tier):
        // NVMe transfers: ALWAYS compress (NVMe is the bottleneck at ~7 GB/s)
        (_, Tier::NvmeLocal) | (_, Tier::NvmeRemote) |
        (Tier::NvmeLocal, _) | (Tier::NvmeRemote, _):
            return pick_best_algorithm(ctx, force=true)

        // VRAM-to-VRAM remote: compress if link is slow or congested
        (Tier::VramLocal, Tier::VramRemote) | (Tier::VramRemote, Tier::VramLocal):
            if ctx.link_bandwidth_gbps < 25.0 or ctx.link_utilization > 0.7:
                return pick_best_algorithm(ctx, force=false)
            else:
                return Decision(None, reason="fast_uncongested_link")

        // VRAM-to-DRAM (same node): rarely worth it (PCIe 4.0 = 25 GB/s)
        (Tier::VramLocal, Tier::DramLocal) | (Tier::DramLocal, Tier::VramLocal):
            return Decision(None, reason="local_pcie_fast_enough")

        // DRAM-to-DRAM remote: compress for TCP, skip for RDMA
        (Tier::DramLocal, Tier::DramRemote) | (Tier::DramRemote, Tier::DramLocal):
            if ctx.link_bandwidth_gbps < 25.0:
                return pick_best_algorithm(ctx, force=false)
            else:
                return Decision(None, reason="rdma_fast_enough")

        // Default: try if bandwidth constrained
        _:
            if ctx.link_bandwidth_gbps < 10.0 or ctx.link_utilization > 0.8:
                return pick_best_algorithm(ctx, force=false)
            return Decision(None, reason="default_skip")


function pick_best_algorithm(ctx, force) -> CompressionDecision:

    // Trial compress 4 KB sample to measure actual ratio
    if not force and ctx.cached_ratio is None:
        sample = data[0..min(4096, data_size)]
        trial_ratio = trial_compress_lz4(sample)
        if trial_ratio < 1.5 and not force:
            cache_ratio(ctx.alloc_id, trial_ratio)
            return Decision(None, reason="trial_ratio_too_low")

    // Pick engine: GPU or CPU
    if ctx.gpu_available and data_is_in_vram(ctx.source_tier):
        if ctx.gpu_sm_utilization < 0.85:
            // GPU has headroom — use nvCOMP
            if ctx.data_size >= 65536:   // 64 KB+: Cascaded for local storage
                if is_local_storage(ctx.dest_tier) and not crosses_wire(ctx):
                    return Decision(Cascaded, reason="gpu_cascaded_local")
                else:
                    return Decision(Lz4Gpu, reason="gpu_lz4_wire_compatible")
            else:
                return Decision(Lz4Gpu, reason="gpu_lz4_small_buffer")
        else:
            // GPU busy — fall back to CPU if data accessible from host
            return pick_cpu_algorithm(ctx)
    else:
        return pick_cpu_algorithm(ctx)


function pick_cpu_algorithm(ctx) -> CompressionDecision:
    // For NVMe-bound transfers, prefer Zstd (better ratio, NVMe is slow anyway)
    if involves_nvme(ctx):
        return Decision(ZstdCpu, reason="zstd_for_nvme_ratio")
    // For network transfers, prefer LZ4 (speed matters)
    return Decision(Lz4Cpu, reason="lz4_for_speed")
```

### Concrete Threshold Summary

| Threshold | Value | Rationale |
|---|---|---|
| Min size for compression | 4 KB | Below this, overhead > savings |
| Entropy skip (distinct bytes in 256-byte sample) | > 240 | 7.5+ bits/byte = incompressible |
| Min ratio to enable compression | 1.5x | Below this, compression time > transfer time savings |
| Link utilization threshold for "congested" | > 0.7 | 70% utilization = compression reduces contention |
| GPU SM utilization threshold for nvCOMP | < 0.85 | Above 85%, compression kernels compete too much |
| Link bandwidth threshold for "slow" | < 25 Gbps | Below 25G, CPU LZ4 can keep up with pipelining |
| NVMe: always compress threshold | Always | NVMe at ~7 GB/s is always the bottleneck |
| Ratio cache TTL | 10 seconds | Re-evaluate after pattern may have changed |

---

## 4. Wire Format Specification

### Compressed Page Header (16 bytes)

```
Offset  Size  Field               Description
------  ----  -----               -----------
0       2     magic               0xC014 ("COmpress 14") — identifies compressed frame
2       1     version             Protocol version (1)
3       1     algorithm           CompressionAlgorithm enum (u8, see Section 2)
4       4     original_size       Uncompressed size in bytes (u32, max 4 GB)
8       4     compressed_size     Compressed payload size in bytes (u32)
12      4     checksum            xxHash32 of compressed payload (for integrity)
```

Total header: **16 bytes**. Payload follows immediately.

### Design Rationale

**Why 16 bytes instead of v1's 4 bytes:**
- v1 used 3-byte original_size (max 16 MB) which is insufficient for large transport buffers (multi-MB scatter-gather buffers from R28).
- Checksum is essential for detecting corruption. Compressed data is more fragile than raw data — a single bit flip in compressed data causes decompression failure or silent corruption. xxHash32 adds 4 bytes but catches errors.
- Magic bytes enable fast detection: if first two bytes are not 0xC014, the frame is uncompressed. No ambiguity.
- Version field allows future evolution without breaking backward compatibility.

**Alignment:**
- Header is 16-byte aligned (natural for VRAM DMA operations, matches GPU memory transaction size).
- Compressed payload starts at offset 16, which is 16-byte aligned.
- For NIC DMA (R28 scatter-gather), 16-byte alignment is sufficient for ConnectX-5 SGE requirements.

### Streaming Support

**Can the receiver decompress before the full page arrives?**

- **LZ4:** Yes, partially. LZ4 frame format supports streaming decompression with 64 KB blocks. The receiver can decompress each block as it arrives, provided blocks are sent intact.
- **Zstd:** Yes. Zstd streaming API (`ZSTD_decompressStream`) processes data incrementally.
- **nvCOMP:** No. nvCOMP batch API requires the full compressed buffer before decompression. GPU kernels process the entire buffer in parallel.
- **Cascaded/Bitcomp:** No. GPU-proprietary, requires full buffer.

**Implementation:** For CPU-compressed wire transfers, use LZ4/Zstd block boundaries aligned to network chunks. For GPU-compressed transfers, the full compressed page must arrive before decompression starts. Since GPU decompression is 300+ GB/s, the decompression time for a 64 KB page (~0.2 us) is negligible compared to network transfer time.

### Stored Compressed Pages (In-Tier, R10 Integration)

When a page is stored compressed within a tier (PTE has COMPRESSED flag set), the same 16-byte header is prepended to the compressed data. This means:

- The tier driver reads the header to know the algorithm and original size.
- `CompressionHook::decompress()` parses the header to determine the algorithm automatically.
- A compressed 64 KB page with 2x ratio occupies ~32 KB + 16 bytes = ~32 KB in the tier.

---

## 5. nvCOMP Integration Design

### FFI Bindings

R14 creates a `outerlink-nvcomp-sys` crate with raw C bindings and a safe `outerlink-nvcomp` wrapper crate.

#### Key nvCOMP C APIs to Wrap

**LZ4 Batch (primary wire-compatible algorithm):**
```c
// Get required temp buffer size for compression
nvcompStatus_t nvcompBatchedLZ4CompressGetTempSize(
    size_t batch_size,
    size_t max_uncompressed_chunk_bytes,
    nvcompBatchedLZ4FormatOpts format_opts,
    size_t* temp_bytes
);

// Get max compressed chunk size (for output buffer allocation)
nvcompStatus_t nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
    size_t max_uncompressed_chunk_bytes,
    nvcompBatchedLZ4FormatOpts format_opts,
    size_t* max_compressed_bytes
);

// Async compress — runs on a CUDA stream
nvcompStatus_t nvcompBatchedLZ4CompressAsync(
    const void* const* device_uncompressed_ptrs,
    const size_t* device_uncompressed_bytes,
    size_t max_uncompressed_chunk_bytes,
    size_t batch_size,
    void* device_temp_ptr,
    size_t temp_bytes,
    void* const* device_compressed_ptrs,
    size_t* device_compressed_bytes,
    nvcompBatchedLZ4FormatOpts format_opts,
    cudaStream_t stream
);

// Async decompress
nvcompStatus_t nvcompBatchedLZ4DecompressAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    const size_t* device_uncompressed_bytes,
    size_t* device_actual_uncompressed_bytes,
    size_t batch_size,
    void* device_temp_ptr,
    size_t temp_bytes,
    void* const* device_uncompressed_ptrs,
    cudaStream_t stream
);
```

**Cascaded Batch (local storage, structured data):**
- `nvcompBatchedCascadedCompressAsync` / `nvcompBatchedCascadedDecompressAsync`
- Same pattern as LZ4 batch API. Format opts include `num_RLEs`, `num_deltas`, `use_bp` (bitpacking).

#### Rust Wrapper Design

```rust
/// Safe wrapper around nvCOMP. Dynamically loaded at runtime.
pub struct NvcompEngine {
    /// dlopen handle to libnvcomp.so
    lib: libloading::Library,
    /// Function pointers loaded from the library
    fns: NvcompFunctions,
    /// Pre-allocated scratch space per GPU
    scratch_pools: HashMap<i32, ScratchPool>,  // gpu_id -> pool
    /// CUDA stream dedicated to compression (low priority)
    streams: HashMap<i32, CudaStream>,         // gpu_id -> stream
}

/// Pre-allocated scratch buffers for nvCOMP operations.
struct ScratchPool {
    /// Temp buffer for compression (size determined by GetTempSize)
    compress_temp: DeviceBuffer,
    /// Temp buffer for decompression
    decompress_temp: DeviceBuffer,
    /// Output buffer (max compressed size, reused across operations)
    output_buffer: DeviceBuffer,
    /// Total VRAM allocated for this pool
    total_bytes: usize,
}
```

### Buffer Management

**Scratch space sizing (per GPU):**
- nvCOMP `GetTempSize` returns the required scratch for a given batch size and chunk size.
- For 64 KB pages, LZ4 batch temp size is ~1-2 MB.
- For Cascaded, temp size is ~4-8 MB (more complex algorithm).
- Pre-allocate at startup: **16 MB per GPU** covers LZ4 + Cascaded scratch for batch sizes up to 256 chunks.
- Output buffer: max compressed size for 64 KB input is ~65 KB (LZ4 worst case is slightly larger than input). Pre-allocate 128 KB per output slot.

**Buffer pool sizing:**
- 4 concurrent compression operations per GPU (matching transport pipeline depth).
- Total per GPU: 16 MB scratch + 4 * 128 KB output = ~16.5 MB VRAM dedicated to compression.
- This is 0.07% of a 24 GB RTX 3090's VRAM. Negligible.

### Async Compression Pipeline

```
Time --->

Chunk 0:  [compress on GPU stream] -> [DMA to host/NIC]
Chunk 1:                [compress on GPU stream] -> [DMA to host/NIC]
Chunk 2:                               [compress on GPU stream] -> [DMA to host/NIC]
```

Implementation:
1. Ping-pong between two output buffers (double buffering).
2. Compress chunk N into buffer A on the compression CUDA stream.
3. While compression runs, DMA chunk N-1 from buffer B to NIC (or host pinned memory).
4. CUDA event synchronization: DMA waits on compression event for the current buffer.

```rust
/// Pipeline state for async compress-and-send.
struct CompressionPipeline {
    buffers: [DeviceBuffer; 2],    // ping-pong output buffers
    events: [CudaEvent; 2],        // signal when compression completes
    current: usize,                 // 0 or 1
    stream: CudaStream,            // low-priority compression stream
}

impl CompressionPipeline {
    fn compress_and_send(&mut self, chunk: &DeviceSlice, transport: &Transport) {
        let buf_idx = self.current;
        let next_idx = 1 - buf_idx;

        // Wait for previous send to finish using this buffer
        self.events[buf_idx].synchronize();

        // Launch async compression into current buffer
        nvcomp_compress_async(chunk, &mut self.buffers[buf_idx], &self.stream);
        self.events[buf_idx].record(&self.stream);

        // Meanwhile, send the previously compressed chunk from the other buffer
        // (this was compressed in the previous iteration)
        if self.has_pending(next_idx) {
            transport.send_async(&self.buffers[next_idx]);
        }

        self.current = next_idx;
    }
}
```

### Fallback When nvCOMP Unavailable

nvCOMP is unavailable in two scenarios:
1. **Not installed:** `libnvcomp.so` not found at runtime. Common on GeForce systems without CUDA toolkit.
2. **GPU not suitable:** Compute capability < 6.0 (pre-Pascal).

**Detection at startup:**
```rust
fn probe_nvcomp() -> NvcompAvailability {
    match libloading::Library::new("libnvcomp.so") {
        Err(_) => NvcompAvailability::NotInstalled,
        Ok(lib) => {
            // Verify we can call a basic function
            match lib.get::<fn() -> u32>(b"nvcompGetVersion") {
                Err(_) => NvcompAvailability::IncompatibleVersion,
                Ok(get_version) => {
                    let version = get_version();
                    if version < MIN_NVCOMP_VERSION {
                        NvcompAvailability::TooOld(version)
                    } else {
                        NvcompAvailability::Available(version)
                    }
                }
            }
        }
    }
}
```

**Fallback behavior:**
- All GPU-resident data falls back to: `cudaMemcpy D2H -> LZ4/Zstd CPU compress -> send`.
- This adds a `cudaMemcpy` round-trip (~5-10 us for 64 KB on PCIe 4.0) plus CPU compression time.
- For TCP links (3.1 GB/s), CPU LZ4 at 2.8 GB/s (4 cores) is adequate. For RDMA (12.5 GB/s), skip compression entirely (CPU cannot keep up and there is no GPU engine).

**Fallback decision matrix:**

| Link Speed | nvCOMP Available | Action |
|---|---|---|
| < 25 Gbps (TCP) | No | CPU LZ4/Zstd (D2H copy + compress) |
| < 25 Gbps (TCP) | Yes | nvCOMP LZ4 (compress in VRAM, then send) |
| >= 25 Gbps (RDMA) | No | Skip compression (CPU too slow, DMA direct) |
| >= 25 Gbps (RDMA) | Yes | nvCOMP LZ4 (compress in VRAM, then send) |

---

## 6. Per-Tier Compression Strategy

R10 defines 6 tiers. Here is the compression strategy for each tier pair.

### Tier Definitions (from R10 v2)

| Tier ID | Name | Medium | Bandwidth (approx) |
|---|---|---|---|
| 0 | VRAM Local | Local GPU VRAM | Internal: 936 GB/s (3090) |
| 1 | VRAM Remote | Remote GPU VRAM (network) | RDMA: 12.5 GB/s, TCP: 3.1 GB/s |
| 2 | DRAM Local | Local host pinned DRAM | PCIe 4.0: ~25 GB/s |
| 3 | DRAM Remote | Remote host DRAM (network) | RDMA: 12.5 GB/s, TCP: 3.1 GB/s |
| 4 | NVMe Local | Local NVMe SSD | ~7 GB/s (4x Gen4 RAID-0) |
| 5 | NVMe Remote | Remote NVMe (network + disk) | min(network, ~7 GB/s) |

### Transfer Matrix

| Source -> Dest | Compress? | Engine | Algorithm | Rationale |
|---|---|---|---|---|
| **T0 -> T1** (VRAM local -> VRAM remote) | Conditional | GPU (nvCOMP) | LZ4 | Compress if link < 25 Gbps or utilization > 70%. On 100G RDMA, skip unless data is highly compressible (cached ratio > 3x). |
| **T0 -> T2** (VRAM -> local DRAM) | No | -- | -- | PCIe 4.0 at ~25 GB/s. Compression adds latency without bandwidth benefit on same-node PCIe. Exception: compress for in-tier storage if DRAM is under pressure (R10 "all tiers full" cascade). |
| **T0 -> T4** (VRAM -> local NVMe) | **Always** | GPU (nvCOMP) | LZ4 or Cascaded | NVMe is the bottleneck at ~7 GB/s. GPU compression at 90 GB/s is free. 2x ratio doubles effective NVMe bandwidth to ~14 GB/s. Cascaded for structured data (tensor weights). |
| **T0 -> T5** (VRAM -> remote NVMe) | **Always** | GPU (nvCOMP) | LZ4 | Double bottleneck: network + NVMe. Compression helps both. Wire-compatible format (LZ4) required. |
| **T2 -> T3** (DRAM local -> DRAM remote) | Conditional | CPU | LZ4 or Zstd | Compress if link < 25 Gbps. R17: per-link decision — compress for slow TCP links, skip for RDMA. Zstd level 1 if ratio matters more than speed. |
| **T2 -> T4** (DRAM -> local NVMe) | **Always** | CPU | Zstd level 1 | NVMe bottleneck. CPU Zstd at ~340 MB/s single-core is fine since NVMe sequential write is ~7 GB/s but 64 KB random is ~3-4 GB/s. Use 4 threads for ~1.4 GB/s Zstd which matches NVMe effective speed. Better ratio than LZ4 saves NVMe write endurance. |
| **T2 -> T5** (DRAM -> remote NVMe) | **Always** | CPU | Zstd level 1 | Same as T2->T4 but also reduces network transfer. Double win. |
| **T1 -> T0** (VRAM remote -> VRAM local) | Conditional | GPU (nvCOMP) | LZ4 | Same as T0->T1 (symmetric). Remote side compresses, local side decompresses. |
| **T4 -> T0** (NVMe -> VRAM) | **Always** | CPU read, GPU decompress | LZ4 | Page stored compressed on NVMe (from T0->T4). Read compressed from NVMe, DMA to VRAM, GPU decompresses. |
| **T4 -> T2** (NVMe -> DRAM) | **Always** | CPU | Zstd | Page stored compressed. Read compressed, CPU decompress. |

### R12 Dedup Integration

Deduped pages (PTE flags: `DEDUP_CANONICAL` or `DEDUP_REFERENCE`) are read-only. This means:
- Compression is a **one-time cost** (compress once when the canonical page is stored).
- Decompression happens on every read, but since deduped pages are shared across contexts, the amortized cost is even lower.
- Force compression for all dedup canonical pages regardless of ratio threshold (even 1.2x saves space that multiplies across all references).

### R17 Multi-Path Integration

R17 multi-path striping makes per-link compression decisions:
- **Fast link (RDMA, 100 Gbps):** Skip compression by default. The overhead is not worth it.
- **Slow link (TCP, 25 Gbps):** Enable compression. CPU LZ4 can keep up with pipelining.
- **Mixed paths:** When R17 stripes across both fast and slow links, each stripe independently decides. Chunks routed via TCP are compressed; chunks via RDMA are not.

This requires the compression decision to be made **per-chunk, per-link**, not per-page. The transport pipeline queries `should_compress()` for each chunk before sending, passing the link's bandwidth and utilization.

### R30 Persistent Kernel Ring Buffer Integration

R30 defines VRAM ring buffers for persistent kernel feed. Compression fits as a pipeline stage:

```
Producer (GPU kernel) -> Ring Buffer Slot -> [Compress] -> DMA to NIC -> Wire
Wire -> DMA to VRAM -> [Decompress] -> Ring Buffer Slot -> Consumer (GPU kernel)
```

The compress/decompress stages operate on ring buffer slots using the dedicated compression CUDA stream. Double-buffering applies: compress slot N while NIC sends slot N-1.

Ring buffer slot size must account for compression overhead: allocate slots at max_compressed_size (slightly larger than uncompressed size) even though most compressed data will be smaller.

---

## 7. Updated Risk Assessment

### Risks Carried from v1 (Updated)

**R1: Compression Overhead Exceeds Savings — DOWNGRADED to LOW**

Cross-topic mitigation: The adaptive decision tree (Section 3) has 7 gates before compression is attempted. Fast entropy estimation (256 bytes, ~50 ns) and cached decisions per allocation ID mean overhead for incompressible data is < 100 ns per page. NVMe paths always compress because the bottleneck guarantees savings.

**R2: nvCOMP Unavailability — UNCHANGED, LOW**

Fallback path fully specified (Section 5). CPU LZ4 handles TCP links. RDMA links skip compression entirely without nvCOMP. System is fully functional without nvCOMP, just slower on compressible workloads over fast links.

**R3: GPU Compute Contention — DOWNGRADED to LOW**

Decision tree checks GPU SM utilization (NVML polling) and skips GPU compression when SM utilization > 85%. Compression uses a dedicated low-priority CUDA stream. R30 persistent kernel workloads are the main concern, but R30's ring buffer pipeline explicitly accounts for compression as a pipeline stage.

**R4: Compression Breaks Zero-Copy RDMA — UNCHANGED, LOW-MEDIUM**

Inherent tradeoff. Adaptive bypass preserves zero-copy for incompressible data. For compressible data, the extra kernel launch (~5-10 us) is offset by reduced wire time. Decision tree evaluates this per-transfer.

**R5: Memory Pressure from Compression Buffers — DOWNGRADED to LOW**

Section 5 sizes the buffer pool: 16.5 MB per GPU (0.07% of 24 GB). Fixed-size pool with bounded concurrency (4 operations). No unbounded allocation.

**R6: Latency Regression for Small Transfers — UNCHANGED, LOW**

4 KB threshold with zero-overhead size check.

### New Risks from Cross-Topic Analysis

**R7: Algorithm Negotiation Mismatch (LOW)**

Two nodes may support different algorithm sets. If sender picks nvCOMP Cascaded but receiver has no GPU, decompression fails.

Mitigation: Cascaded/Bitcomp are never used on the wire (Section 1, Q2 resolution). Only standard-format algorithms (LZ4, Zstd) cross the wire. Both sides always have CPU fallback decompression. Negotiation happens at connection time via capability bits.

**R8: Compression Ratio Cache Staleness (LOW)**

Cached ratio per alloc_id may become stale if the application changes data patterns (e.g., training phase changes from dense to sparse gradients).

Mitigation: Cache TTL of 10 seconds forces re-evaluation. The trial compression sample (4 KB, ~44 ns on GPU or ~5 us on CPU) is cheap enough to re-run periodically.

**R9: R28 Scatter-Gather Interaction Complexity (MEDIUM)**

R28 defines scatter-gather DMA where data is gathered from multiple source buffers into a contiguous wire buffer. Compression must happen after gather (cannot compress non-contiguous data). This means the gather buffer must be allocated, then compressed in-place or into a second buffer.

Mitigation: The pipeline is: gather into contiguous buffer -> compress -> send. The gather buffer is pre-allocated as part of the transport buffer pool. Compression operates on the contiguous result. No architectural conflict, but the buffer pool must account for both gathered and compressed copies.

---

## 8. Updated Implementation Phases

### Phase R14-A: Compression Engine Core + CPU Path (2-3 weeks)

**Goal:** Full `CompressionEngine` trait, CPU implementations, wire format, R10 hook integration.

Deliverables:
1. `CompressionEngine` trait and `CompressionContext` types in `outerlink-common`
2. `Lz4CpuCompressor` using `lz4_flex` crate
3. `ZstdCpuCompressor` using `zstd` crate (level 1 default)
4. Wire format header (16 bytes, Section 4)
5. `TransportCompressionHook` adapter for R10's `CompressionHook` trait
6. Entropy estimator (256-byte sample, distinct byte count)
7. Integration into transport send/receive path (compress-within-send)
8. Config: `OUTERLINK_COMPRESSION` env var, minimum size threshold
9. Unit tests: compress/decompress round-trip, wire format parsing, entropy estimation
10. Benchmarks: CPU compression throughput on target hardware

**Acceptance criteria:** CPU compression works end-to-end on TCP transport. Compressible data shows measurable throughput improvement. R10 hook compiles and passes stub tests.

### Phase R14-B: nvCOMP GPU Compression (3-4 weeks)

**Goal:** nvCOMP integration with fallback, buffer pool, async pipeline.

Deliverables:
1. `outerlink-nvcomp-sys` crate: `bindgen` against nvCOMP C headers
2. `outerlink-nvcomp` crate: safe wrapper with `NvcompEngine`
3. Runtime probing: `dlopen` + version check + capability detection
4. `NvcompLz4Compressor` implementing `CompressionEngine` for GPU LZ4
5. `NvcompCascadedCompressor` for local-storage structured data
6. `ScratchPool`: pre-allocated 16 MB per GPU scratch space
7. Dedicated low-priority CUDA stream per GPU for compression
8. Fallback logic: nvCOMP unavailable -> CPU path (Section 5)
9. Benchmarks on RTX 3090: measure actual compression throughput, SM utilization

**Acceptance criteria:** GPU compression works for VRAM-resident data. Fallback to CPU works when nvCOMP is absent. Scratch pool stays within 16.5 MB per GPU.

### Phase R14-C: Adaptive Decision Engine (2 weeks)

**Goal:** Full adaptive decision tree with per-transfer decisions.

Deliverables:
1. `AdaptiveCompressionEngine` implementing full decision tree (Section 3)
2. Per-alloc-id ratio cache with 10-second TTL
3. GPU SM utilization polling via NVML (1-second interval)
4. Link bandwidth and utilization integration from transport layer
5. R12 dedup integration: force compression for dedup canonical pages
6. R17 multi-path integration: per-link compression decisions
7. Metrics: compression ratio histogram, algorithm selection distribution, bypass rate
8. Config tuning: expose all thresholds via config file

**Acceptance criteria:** Decision tree correctly routes different data types to different algorithms. Incompressible data bypassed with < 100 ns overhead. NVMe transfers always compressed. Per-link decisions work with R17 striping.

### Phase R14-D: Pipelining, Optimization, Integration (2-3 weeks)

**Goal:** Overlap compression with I/O. Full cross-topic integration.

Deliverables:
1. `CompressionPipeline` with double-buffered ping-pong GPU buffers
2. Integration with io_uring submission (TCP path): compress chunk N while sending N-1
3. Integration with RDMA verbs (RDMA path): compress chunk N while RDMA-posting N-1
4. R28 scatter-gather integration: gather -> compress -> send pipeline
5. R30 ring buffer integration: compression as pipeline stage in VRAM ring buffer
6. R11 prefetch integration: decompress-on-arrival for prefetched pages
7. End-to-end benchmarks: effective throughput per transport, per data type
8. Tuning: chunk size per transport, pipeline depth, thread count for CPU compression

**Acceptance criteria:** Pipelined compression shows > 1.5x effective throughput on compressible data over TCP. No measurable regression on incompressible data. R28/R30 integration passes integration tests.

### Timeline Summary

| Phase | Duration | Dependencies | Can Overlap With |
|---|---|---|---|
| R14-A | 2-3 weeks | P6 transport working | R10-B (eviction policies) |
| R14-B | 3-4 weeks | R14-A, CUDA driver interception | R10-C (NVMe tier) |
| R14-C | 2 weeks | R14-A + R14-B, R17 topology | R10-D (remote tiers) |
| R14-D | 2-3 weeks | R14-C, R28, R30 stubs | R10-D (hooks integration) |

**Total: 9-12 weeks** (unchanged from v1, but scope is now precisely defined).

---

## 9. Updated Success Criteria

| Metric | Target | Measurement Method |
|---|---|---|
| Effective throughput increase (compressible data, TCP) | >= 2x | Transport benchmark, gradient-like data |
| Effective throughput increase (compressible data, NVMe) | >= 2x | Tier migration benchmark |
| Latency regression (incompressible data) | < 100 ns per page | Entropy check + cache lookup timing |
| Latency regression (small messages < 4 KB) | 0 ns | Size check is unconditional bypass |
| CPU overhead (CPU compression path, 4 cores) | < 2 cores sustained | `perf stat` during transport benchmark |
| GPU overhead (nvCOMP path) | < 5% SM utilization | NVML monitoring during application benchmark |
| VRAM overhead (scratch + buffers) | < 20 MB per GPU | Memory accounting in ScratchPool |
| Compression works with all transports | TCP, RDMA, USB4 | Integration tests per transport |
| Fallback works without nvCOMP | CPU-only mode functional | Test with nvCOMP uninstalled |
| R10 CompressionHook integration | Compressed pages stored/retrieved correctly | Migration round-trip test |
| R17 per-link decisions | Different algorithms per link type | Multi-transport integration test |
| Wire format interoperability | GPU-compressed data decompressible by CPU | Cross-node heterogeneous test |

---

## 10. Related Documents

- `preplan.md` -- Original v1 pre-plan (preserved for reference)
- `research/01-cpu-compression-algorithms.md` -- CPU algorithm benchmarks
- `research/02-gpu-native-compression-nvcomp.md` -- nvCOMP evaluation
- `research/03-gradient-compression-techniques.md` -- Gradient-specific (owned by R20)
- `../../R10-memory-tiering/preplan-v2.md` -- PTE layout, CompressionHook trait, COMPRESSED flag
- `../../R20-nccl-backend/preplan-v2.md` -- NCCL handle capability bits, gradient compression boundary
- `../../../phase-08-smart-memory/R11-speculative-prefetching/` -- Prefetch decompress-on-arrival
- `../../../phase-08-smart-memory/R12-memory-deduplication/` -- Dedup read-only page compression
- `../../../phase-09-hardening/R15-fault-tolerance/` -- Parity + compression interaction
