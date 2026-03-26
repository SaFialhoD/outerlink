# R28: Scatter-Gather DMA — Pre-Plan v2

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft
**Priority:** MEDIUM (Phase 9)
**Supersedes:** preplan.md (v1)

## Purpose

Second-round refinement of scatter-gather DMA planning. v1 established scope, decisions, and phases. v2 adds exact Rust structs, concrete algorithms integrated with R10 page table, performance crossover models, and resolved open questions from v1.

---

## 1. Rust Data Structures

### Core Types

```rust
/// A single scatter-gather element mapping to one contiguous memory run.
/// Mirrors ibv_sge but with Rust safety and OuterLink page-table awareness.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct SgeEntry {
    /// Virtual address (BAR1 offset or host pinned addr)
    pub addr: u64,
    /// Byte length of this contiguous run
    pub length: u32,
    /// Local memory region key (single MR covers all VRAM)
    pub lkey: u32,
}

/// A list of SGEs representing one logical transfer.
/// Invariant: total data = sum of all entry lengths.
/// Invariant: entries.len() <= max_sge (30 on ConnectX-5).
pub struct ScatterGatherList {
    pub entries: ArrayVec<SgeEntry, 30>,
    /// Total bytes across all entries
    pub total_bytes: u64,
    /// Number of contiguous runs (may differ from entries.len() if
    /// adjacent pages were merged)
    pub run_count: u32,
}

/// Result of analyzing a page list for scatter-gather feasibility.
pub struct FragmentAnalysis {
    /// Contiguous runs after merging adjacent pages
    pub runs: Vec<ContiguousRun>,
    /// Total fragment count (runs.len())
    pub fragment_count: u32,
    /// Total data size in bytes
    pub total_bytes: u64,
    /// Recommended transfer method
    pub recommendation: TransferMethod,
}

/// A contiguous run of pages in physical VRAM.
#[derive(Clone, Debug)]
pub struct ContiguousRun {
    /// Starting physical address (or BAR1 offset)
    pub start_addr: u64,
    /// Byte length of this run (page_count * PAGE_SIZE)
    pub length: u64,
    /// Number of pages in this run
    pub page_count: u32,
}

/// Transfer method decision — output of the decision engine.
#[derive(Clone, Debug)]
pub enum TransferMethod {
    /// Single SGE, data is already contiguous
    SingleSge,
    /// Hardware scatter-gather: <= 30 runs, all >= MIN_SGE_FRAGMENT_SIZE
    HardwareScatterGather {
        sge_list: ScatterGatherList,
    },
    /// Software pre-pack: gather to staging, then single-SGE send
    SoftwarePrePack {
        reason: PrePackReason,
    },
    /// Software pre-pack followed by R14 compression
    CompressedPrePack {
        estimated_ratio: f32,
    },
}

#[derive(Clone, Debug)]
pub enum PrePackReason {
    /// More than 30 fragments
    ExceedsMaxSge,
    /// Fragments too small for efficient SGE processing
    FragmentsTooSmall,
    /// Compression requested and requires contiguous input
    CompressionRequired,
}

/// Transfer descriptor sent as control message before bulk data.
/// Fits in a single RDMA SEND (< 500 bytes for up to 30 fragments).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransferDescriptor {
    pub transfer_id: u64,
    pub total_size: u64,
    pub fragment_count: u32,
    pub method: WireTransferMethod,
    /// Per-fragment layout metadata (only for receiver-scatter cases)
    pub fragments: SmallVec<[FragmentInfo; 8]>,
    /// CRC32 of the descriptor itself for integrity
    pub descriptor_crc: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WireTransferMethod {
    /// Contiguous on wire, contiguous at destination
    Contiguous,
    /// Gathered by sender, contiguous on wire
    HardwareGathered,
    /// Pre-packed by sender, contiguous on wire
    SoftwarePacked,
    /// Pre-packed + compressed
    Compressed {
        algorithm: CompressionAlgorithm,
        compressed_size: u64,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FragmentInfo {
    /// Offset within the logical tensor
    pub logical_offset: u64,
    /// Size of this fragment in bytes
    pub size: u64,
    /// Destination address on receiver (for receiver-scatter)
    pub dest_addr: u64,
}

/// Staging buffer pool for software pre-pack path.
pub struct StagingPool {
    /// Pre-allocated buffers, each `buf_size` bytes
    buffers: Vec<StagingBuffer>,
    /// Available buffer indices
    free_list: Mutex<VecDeque<usize>>,
    /// Semaphore for backpressure when pool exhausted
    available: Semaphore,
}

pub struct StagingBuffer {
    /// Pinned host memory or contiguous VRAM
    pub addr: u64,
    pub size: u64,
    pub lkey: u32,
    /// CUDA stream for async gather/scatter kernels
    pub stream: CudaStream,
}
```

### Configuration Constants

```rust
/// ConnectX-5 hardware limit
pub const MAX_SGE_PER_WR: usize = 30;

/// R10 page size
pub const PAGE_SIZE: u64 = 64 * 1024; // 64KB

/// Maximum data per single hardware scatter-gather WR
/// 30 SGEs * 64KB = 1.875 MB (worst case, all single-page runs)
/// With merged runs, can be much larger
pub const MAX_HW_SG_DATA: u64 = MAX_SGE_PER_WR as u64 * PAGE_SIZE;

/// Minimum fragment size for hardware SGE to be efficient.
/// Below this, NIC descriptor read overhead dominates.
pub const MIN_SGE_FRAGMENT_SIZE: u64 = 4096; // 4KB

/// Staging buffer count and size
pub const STAGING_POOL_COUNT: usize = 8;
pub const STAGING_BUFFER_SIZE: u64 = 4 * 1024 * 1024; // 4MB each = 32MB total

/// Threshold above which compression is considered
pub const COMPRESSION_SIZE_THRESHOLD: u64 = 2 * 1024 * 1024; // 2MB

/// Minimum estimated compression ratio to justify pre-pack + compress
pub const MIN_COMPRESSION_RATIO: f32 = 1.5;
```

---

## 2. Algorithms

### 2.1 Fragment Analyzer: Page List to Contiguous Runs

This is the core algorithm. Input comes from R10's `PageTable::bulk_lookup()`.

```rust
/// Build contiguous runs from a page table lookup result.
///
/// Input: page_addrs from PageTable::bulk_lookup(tensor_id)
///        — physical addresses of pages backing a logical tensor,
///          returned in logical order.
///
/// Output: Vec<ContiguousRun> with adjacent pages merged.
///
/// Complexity: O(N) where N = page count. Single pass.
pub fn analyze_fragments(page_addrs: &[u64]) -> FragmentAnalysis {
    if page_addrs.is_empty() {
        return FragmentAnalysis::empty();
    }

    let mut runs = Vec::with_capacity(page_addrs.len() / 4); // estimate
    let mut run_start = page_addrs[0];
    let mut run_pages: u32 = 1;

    for i in 1..page_addrs.len() {
        let expected_next = page_addrs[i - 1] + PAGE_SIZE;
        if page_addrs[i] == expected_next {
            // Contiguous with previous — extend current run
            run_pages += 1;
        } else {
            // Gap — close current run, start new one
            runs.push(ContiguousRun {
                start_addr: run_start,
                length: run_pages as u64 * PAGE_SIZE,
                page_count: run_pages,
            });
            run_start = page_addrs[i];
            run_pages = 1;
        }
    }
    // Close final run
    runs.push(ContiguousRun {
        start_addr: run_start,
        length: run_pages as u64 * PAGE_SIZE,
        page_count: run_pages,
    });

    let total_bytes = page_addrs.len() as u64 * PAGE_SIZE;
    let fragment_count = runs.len() as u32;
    let recommendation = decide_method(&runs, total_bytes);

    FragmentAnalysis { runs, fragment_count, total_bytes, recommendation }
}
```

### 2.2 Decision Engine

```rust
/// Decide transfer method based on fragment analysis.
///
/// Decision tree:
///   1 run -> SingleSge
///   2-30 runs, all >= 4KB -> HardwareScatterGather
///   2-30 runs, any < 4KB  -> SoftwarePrePack (FragmentsTooSmall)
///   > 30 runs             -> SoftwarePrePack (ExceedsMaxSge)
///   Compressible + > 2MB  -> CompressedPrePack
fn decide_method(runs: &[ContiguousRun], total_bytes: u64) -> TransferMethod {
    if runs.len() == 1 {
        return TransferMethod::SingleSge;
    }

    let all_fragments_large = runs.iter().all(|r| r.length >= MIN_SGE_FRAGMENT_SIZE);

    if runs.len() <= MAX_SGE_PER_WR && all_fragments_large {
        let sge_list = build_sge_list(runs);
        return TransferMethod::HardwareScatterGather { sge_list };
    }

    if runs.len() > MAX_SGE_PER_WR {
        return TransferMethod::SoftwarePrePack {
            reason: PrePackReason::ExceedsMaxSge,
        };
    }

    // Fragments exist but some are too small
    TransferMethod::SoftwarePrePack {
        reason: PrePackReason::FragmentsTooSmall,
    }
}

/// Overlay compression decision on top of base method.
/// Called after decide_method when compressibility info is available.
pub fn consider_compression(
    base: TransferMethod,
    total_bytes: u64,
    estimated_ratio: f32,
) -> TransferMethod {
    if total_bytes >= COMPRESSION_SIZE_THRESHOLD
        && estimated_ratio >= MIN_COMPRESSION_RATIO
    {
        // Compression requires contiguous input, so pre-pack first
        return TransferMethod::CompressedPrePack { estimated_ratio };
    }
    base
}
```

### 2.3 SGE List Builder

```rust
/// Build an ibv_sge-compatible list from contiguous runs.
/// Precondition: runs.len() <= MAX_SGE_PER_WR.
fn build_sge_list(runs: &[ContiguousRun]) -> ScatterGatherList {
    debug_assert!(runs.len() <= MAX_SGE_PER_WR);

    let mut entries = ArrayVec::new();
    let mut total_bytes = 0u64;

    for run in runs {
        entries.push(SgeEntry {
            addr: run.start_addr,
            length: run.length as u32,
            lkey: VRAM_MR_LKEY, // single MR over entire VRAM
        });
        total_bytes += run.length;
    }

    ScatterGatherList {
        entries,
        total_bytes,
        run_count: runs.len() as u32,
    }
}
```

### 2.4 Multi-WR Chunking for Large Scatter-Gather

When a tensor has more than 30 runs but we still want hardware SGE (avoiding full software pre-pack), chunk into multiple WRs:

```rust
/// Split runs into chunks of <= MAX_SGE_PER_WR, each becoming one WR.
/// Returns a Vec of ScatterGatherList, each suitable for one ibv_post_send WR.
/// All WRs are chained in a single ibv_post_send call (one doorbell ring).
pub fn chunk_into_wrs(runs: &[ContiguousRun]) -> Vec<ScatterGatherList> {
    runs.chunks(MAX_SGE_PER_WR)
        .map(|chunk| build_sge_list(chunk))
        .collect()
}
```

This gives a middle ground: 60 fragments = 2 WRs chained in one `ibv_post_send()` call = 1 doorbell. Only falls to full software pre-pack when compression is needed or fragments are < 4KB.

---

## 3. Integration with Other Topics

### 3.1 R10 (Memory Tiering) Integration

R10 provides the `PageTable` trait. The key integration points:

```rust
// R10 PageTable trait (relevant methods)
pub trait PageTable {
    /// Look up multiple PTEs in one call. Returns physical addresses
    /// in logical order for the given tensor.
    fn bulk_lookup(&self, tensor_id: TensorId) -> Vec<PageTableEntry>;

    /// Scan pages matching flags (e.g., DIRTY, REMOTE, LOCAL).
    /// Returns page addresses grouped by contiguous runs.
    fn scan_by_flags(&self, flags: PageFlags) -> Vec<ContiguousRun>;
}

// R10 MigrationEngine (relevant methods)
pub trait MigrationEngine {
    /// Migrate multiple pages in a batch. R28 uses this for
    /// pre-positioning pages before scatter-gather if needed.
    fn migrate_batch(&self, pages: &[PageId], target: Tier) -> Result<()>;
}
```

**Data flow: R10 -> R28**

```
1. Application requests transfer of tensor T
2. R28 calls page_table.bulk_lookup(T.tensor_id)
   -> Returns Vec<PageTableEntry> with physical addresses
3. R28 extracts addresses: ptes.iter().map(|p| p.phys_addr).collect()
4. R28 calls analyze_fragments(&addrs)
   -> Returns FragmentAnalysis with runs + recommendation
5. R28 executes transfer per recommendation
```

**Optimization via scan_by_flags:**

When R28 needs to transfer all dirty pages (not a specific tensor), it uses R10's `scan_by_flags(DIRTY)` which returns pre-grouped contiguous runs. This avoids the sort+merge step since R10's hash-table already knows page adjacency.

```rust
/// Transfer all dirty pages to remote node.
/// Uses R10's scan_by_flags for efficient run detection.
pub fn flush_dirty_pages(
    page_table: &dyn PageTable,
    remote: &RemoteNode,
) -> Result<()> {
    // R10 returns runs already grouped — no fragment analysis needed
    let dirty_runs = page_table.scan_by_flags(PageFlags::DIRTY);
    let total_bytes: u64 = dirty_runs.iter().map(|r| r.length).sum();

    if dirty_runs.len() <= MAX_SGE_PER_WR {
        // Direct hardware scatter-gather
        let sge_list = build_sge_list(&dirty_runs);
        rdma_write_gathered(remote, &sge_list)?;
    } else {
        // Chunk into multiple WRs
        let wr_chunks = chunk_into_wrs(&dirty_runs);
        rdma_write_chained(remote, &wr_chunks)?;
    }
    Ok(())
}
```

### 3.2 R14 (Transport Compression) Integration

R14 provides compress/decompress on contiguous buffers. The integration point is the "gather first, compress second" pipeline:

```rust
// R14 compression interface (relevant)
pub trait Compressor {
    fn compress(&self, input: &[u8], output: &mut [u8]) -> Result<usize>;
    fn estimate_ratio(&self, sample: &[u8]) -> f32;
}

/// Pipeline: gather -> compress -> send
/// Used when total_bytes >= 2MB AND estimated_ratio >= 1.5x
pub fn scatter_gather_compressed_send(
    analysis: &FragmentAnalysis,
    compressor: &dyn Compressor,
    staging_pool: &StagingPool,
    remote: &RemoteNode,
) -> Result<()> {
    let staging = staging_pool.acquire()?;

    // Phase 1: GPU gather kernel — fragments -> contiguous staging
    gpu_gather_to_staging(&analysis.runs, &staging)?;

    // Phase 2: Compress contiguous staging buffer (R14)
    let mut compressed_buf = staging_pool.acquire()?;
    let compressed_size = compressor.compress(
        staging.as_slice(),
        compressed_buf.as_mut_slice(),
    )?;

    // Phase 3: Single-SGE RDMA WRITE of compressed data
    let descriptor = TransferDescriptor {
        transfer_id: next_transfer_id(),
        total_size: analysis.total_bytes,
        fragment_count: analysis.fragment_count,
        method: WireTransferMethod::Compressed {
            algorithm: CompressionAlgorithm::Lz4,
            compressed_size: compressed_size as u64,
        },
        fragments: build_fragment_info(&analysis.runs),
        descriptor_crc: 0, // computed before send
    };

    send_descriptor(remote, &descriptor)?;
    rdma_write_single(remote, &compressed_buf, compressed_size)?;

    staging_pool.release(staging);
    staging_pool.release(compressed_buf);
    Ok(())
}
```

**Why gather-first-then-compress (from R14 cross-findings):** Compressing a contiguous buffer yields significantly better ratios than compressing individual fragments. LZ4/Zstandard's dictionary window works across the full buffer, finding patterns that span fragment boundaries. Typical improvement: 1.3x -> 1.8x ratio for weight data.

### 3.3 R20 (NCCL Backend) Integration

R20's NCCL plugin uses scatter-gather for non-contiguous collective buffers:

```rust
/// NCCL backend uses R28 for grouped receives in AllGather.
/// When NCCL calls ncclAllGather with a non-contiguous output buffer
/// (e.g., tensor-parallel weight shards), R28 builds SGEs for the
/// receive side.
///
/// Data flow:
///   NCCL calls ncclAllGather(sendbuf, recvbuf, count, ...)
///   R20 plugin detects non-contiguous recvbuf layout
///   R20 calls R28::build_recv_sge_list(recvbuf_pages)
///   R20 posts ibv_post_recv with multi-SGE list
///   Incoming data is hardware-scattered to correct locations
pub fn build_recv_sge_list(
    page_table: &dyn PageTable,
    recv_regions: &[MemoryRegion],
) -> ScatterGatherList {
    let mut all_runs = Vec::new();
    for region in recv_regions {
        let ptes = page_table.bulk_lookup(region.tensor_id);
        let addrs: Vec<u64> = ptes.iter().map(|p| p.phys_addr).collect();
        let analysis = analyze_fragments(&addrs);
        all_runs.extend(analysis.runs);
    }

    // If total runs fit in one WR, use hardware scatter
    if all_runs.len() <= MAX_SGE_PER_WR {
        build_sge_list(&all_runs)
    } else {
        // For receive side, we must use SEND/RECV with scatter,
        // or fall back to staging + scatter kernel
        build_sge_list(&all_runs[..MAX_SGE_PER_WR])
        // Remaining runs handled by overflow scatter kernel
    }
}
```

### 3.4 R21 (GPU Direct Storage) Integration

R21 uses scatter-gather for loading dataset chunks into non-contiguous GPU regions:

```
NVMe -> host staging -> scatter-gather RDMA WRITE -> multiple GPU VRAM regions

The SGE list for R21 is built the same way as for R28,
but the source is host memory (NVMe read target) and the
destinations are GPU VRAM pages. R28 provides the
analyze_fragments + build_sge_list primitives; R21 uses them
with NVMe source addresses instead of VRAM source addresses.
```

### 3.5 R30 (Persistent Kernels) Integration

R30's persistent kernel ring buffer can be fed via scatter-gather:

```
Persistent kernel expects data in a ring buffer in VRAM.
Ring buffer slots may wrap around physical pages.
R28 builds SGE list for ring buffer write positions,
enabling the NIC to scatter incoming data directly
into the correct ring buffer slots without CPU involvement.
```

---

## 4. Performance Model

### 4.1 Hardware SG vs Software Pre-Pack Crossover

```
Legend:
  t_hw(N)   = time for N-SGE hardware scatter-gather
  t_sw(S)   = time for software pre-pack of S bytes
  t_net(S)  = network transfer time for S bytes

Hardware scatter-gather total time:
  t_hw_total = t_sge_build(N) + t_nic_descriptor(N) + t_net(S)
  t_sge_build(N) = ~100ns * N  (CPU, trivial array fill)
  t_nic_descriptor(N) = ~150ns * N  (NIC reads N SGE descriptors)
  t_net(S) = S / wire_bandwidth

Software pre-pack total time:
  t_sw_total = t_kernel_launch + t_gather(S) + t_net(S)
  t_kernel_launch = ~5us  (CUDA kernel launch overhead)
  t_gather(S) = S / gpu_internal_bw  (1.5 TB/s on 3090)
  t_net(S) = S / wire_bandwidth
```

**Crossover analysis for ConnectX-5 at 100Gbps, RTX 3090:**

| Fragments (N) | Fragment Size | Total (S) | HW SG Overhead | SW PrePack Overhead | Winner |
|---------------|--------------|-----------|----------------|--------------------|---------|
| 1 | 64KB | 64KB | 0 (single SGE) | N/A | SingleSge |
| 5 | 64KB | 320KB | ~1.25us | ~5.2us | HW SG (4x faster) |
| 10 | 64KB | 640KB | ~2.5us | ~5.4us | HW SG (2x faster) |
| 30 | 64KB | 1.875MB | ~7.5us | ~6.25us | ~Tie |
| 30 | 256KB | 7.5MB | ~7.5us | ~10us | HW SG |
| 60 (2 WRs) | 64KB | 3.75MB | ~15us | ~7.5us | SW PrePack |
| 5 | 1KB | 5KB | ~1.25us | ~5us | HW SG (data too small for kernel) |
| 30 | 512B | 15KB | ~7.5us | ~5us | SW PrePack (tiny frags) |

**Key findings:**
- HW SG wins for 2-30 fragments with fragments >= 4KB (the common case for 64KB pages)
- SW pre-pack wins when fragment count > 30 or fragments < 4KB
- At exactly 30 fragments of 64KB, it is roughly a tie -- HW SG still preferred (avoids staging buffer allocation)
- Multi-WR chaining (2 WRs for 31-60 fragments) is worse than SW pre-pack due to 2 doorbells
- **Correction from v1:** The 30-fragment crossover in v1's Decision 2 leaned toward "always use HW SG when <= 30." This holds, but with the sub-rule: skip HW SG if all fragments are < 4KB

### 4.2 Network Transfer Time Dominance

For any transfer > ~100KB at 100Gbps, network time dominates:

```
100KB  at 100Gbps = 8us
1MB   at 100Gbps = 80us
10MB  at 100Gbps = 800us
100MB at 100Gbps = 8ms
```

SG overhead (1-8us) and pre-pack overhead (5-10us) are both negligible for transfers > 1MB. The decision matters primarily for small-to-medium transfers (64KB - 2MB) where overhead is a measurable fraction of total time.

### 4.3 Staging Pool Sizing

```
Pool: 8 buffers x 4MB = 32MB total

Why 4MB per buffer:
  - MAX_HW_SG_DATA = 1.875MB (30 * 64KB worst case)
  - Buffers serve the > 30 fragment case, which can be larger
  - 4MB covers up to 64 pages (common fragmentation ceiling)
  - Oversized transfers (> 4MB) use multiple staging rounds

Why 8 buffers:
  - Supports up to 8 concurrent scatter-gather transfers
  - Triple-buffering for pipelined transfers: gather, compress, send
  - 2 buffers per pipeline stage + 2 spare
  - Backpressure via semaphore if exhausted (queue, don't fail)
```

---

## 5. Resolved Open Questions from v1

### v1 Q1: "Should we expose scatter-gather as a user-facing API?"
**Resolved: No.** Scatter-gather is purely internal to the transport layer. The application/CUDA layer sees standard `cuMemcpy` semantics. OuterLink's interception layer transparently uses SG when the page table indicates fragmentation. No user API change needed.

### v1 Q2: "How does scatter-gather interact with QoS and flow control?"
**Resolved:** Large multi-SGE WRs do not dominate bandwidth unfairly. A 30-SGE WR carrying 1.875MB is equivalent to a single 1.875MB WR from the NIC's perspective -- it occupies the wire for the same duration. QoS priority is set per-QP, not per-WR. Scatter-gather WRs inherit the QP's traffic class and are scheduled identically to single-SGE WRs.

### v1 Q3: "Can we overlap SGE list building with previous transfer completion?"
**Resolved: Yes.** SGE list building is pure CPU work (~100ns * N). It runs on the completion polling thread while the previous WR is in-flight. The pattern:

```
While WR[k] is on the wire:
    CPU builds SGE list for WR[k+1]
    CPU prepares TransferDescriptor for WR[k+1]
When CQE for WR[k] arrives:
    Post WR[k+1] immediately (SGE list already ready)
```

This pipelining adds zero latency to the steady-state transfer path.

### Research Q: "Does UCX abstract scatter-gather?"
**Resolved: UCX supports multi-iov (scatter-gather) via `ucp_tag_send_nbx` with `UCP_DATATYPE_IOV`.** UCX maps iov entries to SGEs internally. However, UCX may add its own coalescing/packing logic. For maximum control (especially for the 30-SGE limit exploitation), we use raw ibv_verbs for the hardware SG path and UCX for simple transfers. This is the hybrid approach from v1 Decision 1.

### Research Q: "Performance delta: 30 SGEs in 1 WR vs software pre-pack?"
**Resolved** in Section 4.1 above. Hardware SG saves ~3-5us for typical transfers. Not huge, but eliminates staging buffer allocation and GPU memory bandwidth consumption, which matters under load.

---

## 6. Implementation Phases (Refined from v1)

### Phase 1: Fragment Analyzer + Software Pre-Pack (2-3 weeks)

**Deliverables:**
- `FragmentAnalysis` struct and `analyze_fragments()` function
- `StagingPool` with 8 x 4MB buffers
- CUDA gather kernel (non-contiguous VRAM -> contiguous staging)
- CUDA scatter kernel (contiguous staging -> non-contiguous VRAM)
- `TransferDescriptor` protocol (control message before bulk data)
- Decision engine (`decide_method()`)

**Integration test:**
- R10 page table returns 50 non-contiguous pages for a test tensor
- Software pre-pack gathers them, sends via single-SGE RDMA WRITE
- Receiver scatter kernel places data in correct locations
- Byte-level verification passes

### Phase 2: Hardware Scatter-Gather (1-2 weeks)

**Deliverables:**
- `build_sge_list()` from contiguous runs
- `ibv_post_send` with multi-SGE WR
- `chunk_into_wrs()` for > 30 fragment case (multi-WR chaining)
- Query `ibv_query_device` for actual `max_sge` at startup

**Integration test:**
- 5-fragment, 15-fragment, and 30-fragment transfers all use hardware SG
- Latency measurement confirms < 10us overhead vs single-SGE baseline
- 31-fragment transfer correctly falls back to 2-WR chain or software pre-pack

### Phase 3: R14 Compression Pipeline (1-2 weeks)

**Deliverables:**
- `scatter_gather_compressed_send()` pipeline
- `consider_compression()` overlay on decision engine
- Triple-buffering for gather/compress/send overlap
- Receiver decompress + scatter pipeline

**Integration test:**
- 10MB fragmented tensor with 2x compressible data
- Gather -> compress -> send takes less time than gather -> send (uncompressed)
- End-to-end correctness with compression

### Phase 4: OpenDMA BAR1 Integration (2-3 weeks)

**Deliverables:**
- BAR1-aware SGE builder (SGEs use BAR1 offsets)
- GPU MMU contiguity detection (check if scattered VRAM is contiguous in BAR1 space)
- OpenDMA scatter-gather path (NIC gathers from BAR1)

**Integration test:**
- Zero-CPU scatter-gather: NIC reads from scattered BAR1 regions
- Latency matches or beats host-staged path

---

## 7. Success Metrics (Refined from v1)

| Metric | Baseline | Target | How to Measure |
|--------|----------|--------|----------------|
| 5-fragment 320KB transfer | 2 separate WRs + staging | 1 WR, 5 SGEs, no staging | Latency diff < 3us |
| 30-fragment 1.875MB transfer | 30 separate WRs | 1 WR, 30 SGEs | Latency reduction ~20us |
| MoE 4-expert gather (4 x 500MB) | 4 separate RDMA WRITEs | 4 SGEs in 1 WR | Wall time reduction ~15% |
| Staging memory overhead | 0 (no pool) | Fixed 32MB pool | Memory accounting |
| > 30 fragment transfer | Fails or requires manual defrag | Software pre-pack transparent | Correctness test |
| Compressed fragmented send | N/A | gather -> compress saves bandwidth | Effective BW > 100Gbps |

---

## Related Documents

- [preplan.md](./preplan.md) -- v1 pre-plan (superseded by this document)
- [research/01-rdma-scatter-gather.md](./research/01-rdma-scatter-gather.md) -- RDMA SGE details
- [research/02-gpu-sparse-data.md](./research/02-gpu-sparse-data.md) -- Sparse data patterns
- [research/03-scatter-gather-pipeline.md](./research/03-scatter-gather-pipeline.md) -- Pipeline design
- R10 Memory Tiering -- PageTable trait, bulk_lookup, scan_by_flags
- R14 Transport Compression -- Compressor trait, gather-then-compress pipeline
- R20 NCCL Backend -- Grouped receives via scatter-gather
- R21 GPU Direct Storage -- NVMe scatter-gather loading
- R30 Persistent Kernels -- Ring buffer scatter-gather feeding

## Open Questions

- [ ] Exact `ibv_devinfo -v` output from Pedro's ConnectX-5 to confirm max_sge = 30 (carried from v1, blocked on hardware access)
- [ ] Benchmark 30-SGE ibv_post_send overhead on actual hardware (quantify NIC descriptor read time)
- [ ] Profile VRAM fragmentation on Pedro's 3090s after 1-hour training run to validate the 60/25/15/<1 distribution estimate
