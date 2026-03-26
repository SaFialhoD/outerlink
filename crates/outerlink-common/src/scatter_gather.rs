//! Scatter-Gather DMA support for non-contiguous memory transfers.
//!
//! Implements fragment analysis, transfer method decision engine, and SGE list
//! building for efficient RDMA scatter-gather transfers. Integrates with the
//! R10 page table system to handle fragmented VRAM transparently.
//!
//! # Architecture
//!
//! When a tensor's pages are not physically contiguous in VRAM, a naive approach
//! would copy each fragment individually. This module analyzes the fragmentation
//! pattern and selects the optimal transfer strategy:
//!
//! - **SingleSge**: Data is already contiguous, use one SGE entry.
//! - **HardwareScatterGather**: Up to 30 contiguous runs, NIC gathers them in
//!   a single work request (zero CPU copy).
//! - **SoftwarePrePack**: Too many fragments or fragments too small for hardware
//!   SGE; gather into a staging buffer first, then send contiguously.
//! - **CompressedPrePack**: Pre-pack followed by compression (R14) when data is
//!   large and compressible.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, RwLock};

// ---------------------------------------------------------------------------
// Configuration constants
// ---------------------------------------------------------------------------

/// ConnectX-5 hardware limit for scatter-gather entries per work request.
pub const MAX_SGE_PER_WR: usize = 30;

/// R10 page size (64KB).
pub const PAGE_SIZE: u64 = 64 * 1024;

/// Maximum data per single hardware scatter-gather work request.
/// Worst case: 30 SGEs * 64KB = 1.875 MB (all single-page runs).
pub const MAX_HW_SG_DATA: u64 = MAX_SGE_PER_WR as u64 * PAGE_SIZE;

/// Minimum fragment size for hardware SGE to be efficient.
/// Below this threshold, NIC descriptor read overhead dominates.
pub const MIN_SGE_FRAGMENT_SIZE: u64 = 4096;

/// Number of pre-allocated staging buffers.
pub const STAGING_POOL_COUNT: usize = 8;

/// Size of each staging buffer (4MB). Total pool: 32MB.
pub const STAGING_BUFFER_SIZE: u64 = 4 * 1024 * 1024;

/// Size threshold above which compression is considered.
pub const COMPRESSION_SIZE_THRESHOLD: u64 = 2 * 1024 * 1024;

/// Minimum estimated compression ratio to justify pre-pack + compress.
pub const MIN_COMPRESSION_RATIO: f32 = 1.5;

/// Placeholder memory region key for VRAM (single MR covers all VRAM).
pub const VRAM_MR_LKEY: u32 = 0;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A single scatter-gather element mapping to one contiguous memory run.
/// Mirrors `ibv_sge` but with Rust safety and OuterLink page-table awareness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct SgeEntry {
    /// Virtual address (BAR1 offset or host pinned addr).
    pub addr: u64,
    /// Byte length of this contiguous run.
    pub length: u32,
    /// Local memory region key (single MR covers all VRAM).
    pub lkey: u32,
}

/// A list of SGEs representing one logical transfer.
///
/// Invariants:
/// - `total_bytes` equals the sum of all entry lengths.
/// - `entries.len() <= MAX_SGE_PER_WR` (30 on ConnectX-5).
#[derive(Clone, Debug)]
pub struct ScatterGatherList {
    /// SGE entries, capped at MAX_SGE_PER_WR.
    pub entries: Vec<SgeEntry>,
    /// Total bytes across all entries.
    pub total_bytes: u64,
    /// Number of contiguous runs (may differ from entries.len() if
    /// adjacent pages were merged).
    pub run_count: u32,
}

impl ScatterGatherList {
    /// Validate internal invariants.
    pub fn is_valid(&self) -> bool {
        let sum: u64 = self.entries.iter().map(|e| e.length as u64).sum();
        sum == self.total_bytes && self.entries.len() <= MAX_SGE_PER_WR
    }
}

/// A contiguous run of pages in physical VRAM.
#[derive(Clone, Debug, PartialEq)]
pub struct ContiguousRun {
    /// Starting physical address (or BAR1 offset).
    pub start_addr: u64,
    /// Byte length of this run (`page_count * PAGE_SIZE`).
    pub length: u64,
    /// Number of pages in this run.
    pub page_count: u32,
}

/// Result of analyzing a page list for scatter-gather feasibility.
#[derive(Clone, Debug)]
pub struct FragmentAnalysis {
    /// Contiguous runs after merging adjacent pages.
    pub runs: Vec<ContiguousRun>,
    /// Total fragment count (`runs.len()`).
    pub fragment_count: u32,
    /// Total data size in bytes.
    pub total_bytes: u64,
    /// Recommended transfer method.
    pub recommendation: TransferMethod,
}

impl FragmentAnalysis {
    /// Create an empty analysis (zero pages).
    pub fn empty() -> Self {
        Self {
            runs: Vec::new(),
            fragment_count: 0,
            total_bytes: 0,
            recommendation: TransferMethod::SingleSge,
        }
    }
}

/// Transfer method decision -- output of the decision engine.
#[derive(Clone, Debug)]
pub enum TransferMethod {
    /// Single SGE, data is already contiguous.
    SingleSge,
    /// Hardware scatter-gather: <= 30 runs, all >= MIN_SGE_FRAGMENT_SIZE.
    HardwareScatterGather { sge_list: ScatterGatherList },
    /// Software pre-pack: gather to staging, then single-SGE send.
    SoftwarePrePack { reason: PrePackReason },
    /// Software pre-pack followed by R14 compression.
    CompressedPrePack { estimated_ratio: f32 },
}

/// Reason for falling back to software pre-pack.
#[derive(Clone, Debug, PartialEq)]
pub enum PrePackReason {
    /// More than 30 fragments.
    ExceedsMaxSge,
    /// Fragments too small for efficient SGE processing.
    FragmentsTooSmall,
}

/// Wire-format transfer method descriptor.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum WireTransferMethod {
    /// Contiguous on wire, contiguous at destination.
    Contiguous,
    /// Gathered by sender, contiguous on wire.
    HardwareGathered,
    /// Pre-packed by sender, contiguous on wire.
    SoftwarePacked,
    /// Pre-packed + compressed.
    Compressed {
        algorithm: CompressionAlgorithm,
        compressed_size: u64,
    },
}

/// Compression algorithm identifier.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum CompressionAlgorithm {
    Lz4,
    Zstd,
}

/// Per-fragment layout metadata for receiver-scatter cases.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FragmentInfo {
    /// Offset within the logical tensor.
    pub logical_offset: u64,
    /// Size of this fragment in bytes.
    pub size: u64,
    /// Destination address on receiver (for receiver-scatter).
    pub dest_addr: u64,
}

/// Transfer descriptor sent as control message before bulk data.
/// Fits in a single RDMA SEND (< 500 bytes for up to 30 fragments).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransferDescriptor {
    pub transfer_id: u64,
    pub total_size: u64,
    pub fragment_count: u32,
    pub method: WireTransferMethod,
    /// Per-fragment layout metadata (only for receiver-scatter cases).
    pub fragments: Vec<FragmentInfo>,
    /// CRC32 of the descriptor itself for integrity.
    pub descriptor_crc: u32,
}

// ---------------------------------------------------------------------------
// Staging buffer pool (software pre-pack path)
// ---------------------------------------------------------------------------

/// Configuration for the staging buffer pool.
#[derive(Clone, Debug)]
pub struct StagingPoolConfig {
    /// Number of buffers in the pool.
    pub buffer_count: usize,
    /// Size of each buffer in bytes.
    pub buffer_size: u64,
}

impl Default for StagingPoolConfig {
    fn default() -> Self {
        Self {
            buffer_count: STAGING_POOL_COUNT,
            buffer_size: STAGING_BUFFER_SIZE,
        }
    }
}

/// A staging buffer handle. In production, this represents pinned host memory
/// or contiguous VRAM allocated by the transport layer.
#[derive(Debug)]
pub struct StagingBuffer {
    /// Buffer index in the pool.
    pub index: usize,
    /// Simulated address (in production: pinned host memory or contiguous VRAM).
    pub addr: u64,
    /// Size of the buffer.
    pub size: u64,
    /// Local memory region key.
    pub lkey: u32,
    // NOTE: In production, this would also hold a CUDA stream handle
    // for async gather/scatter kernels.
    // pub stream: CudaStream, // TODO: requires CUDA runtime
}

/// Pool of pre-allocated staging buffers for software pre-pack transfers.
///
/// When hardware scatter-gather is not feasible (too many fragments or fragments
/// too small), data is first gathered into a contiguous staging buffer before
/// being sent over the wire.
pub struct StagingPool {
    /// Pre-allocated buffers.
    buffers: Vec<StagingBuffer>,
    /// Free list + in-use tracking under a single mutex to prevent AB-BA deadlocks.
    inner: Mutex<StagingPoolInner>,
    /// Total number of buffers.
    capacity: usize,
    /// Statistics.
    stats: RwLock<StagingPoolStats>,
}

/// Internal state for StagingPool, protected by a single Mutex.
struct StagingPoolInner {
    free_list: VecDeque<usize>,
    in_use: Vec<bool>,
}

/// Statistics for staging pool usage.
#[derive(Debug, Default, Clone)]
pub struct StagingPoolStats {
    pub acquires: u64,
    pub releases: u64,
    pub exhausted_count: u64,
}

impl StagingPool {
    /// Create a new staging pool with the given configuration.
    ///
    /// In production, each buffer would be allocated as pinned host memory
    /// via `cuMemAllocHost` or as contiguous VRAM. Here we simulate with
    /// placeholder addresses.
    pub fn new(config: StagingPoolConfig) -> Self {
        let mut buffers = Vec::with_capacity(config.buffer_count);
        let mut free_list = VecDeque::with_capacity(config.buffer_count);

        for i in 0..config.buffer_count {
            buffers.push(StagingBuffer {
                index: i,
                // Placeholder address; in production this would be a real pinned address.
                // TODO: requires CUDA pinned memory allocation
                addr: 0x1000_0000 + (i as u64 * config.buffer_size),
                size: config.buffer_size,
                lkey: VRAM_MR_LKEY,
            });
            free_list.push_back(i);
        }

        let in_use = vec![false; config.buffer_count];

        Self {
            buffers,
            inner: Mutex::new(StagingPoolInner { free_list, in_use }),
            capacity: config.buffer_count,
            stats: RwLock::new(StagingPoolStats::default()),
        }
    }

    /// Acquire a staging buffer from the pool.
    ///
    /// Returns `None` if all buffers are in use. In production, callers should
    /// wait on a semaphore for backpressure rather than spinning.
    pub fn acquire(&self) -> Option<usize> {
        let mut inner = self.inner.lock().expect("staging pool lock poisoned");
        let result = inner.free_list.pop_front();

        if let Some(idx) = result {
            inner.in_use[idx] = true;
        }
        drop(inner);

        let mut stats = self.stats.write().expect("staging stats lock poisoned");
        if result.is_some() {
            stats.acquires += 1;
        } else {
            stats.exhausted_count += 1;
        }

        result
    }

    /// Release a staging buffer back to the pool.
    ///
    /// # Panics
    /// Panics if `index` is out of range or if the buffer was already released
    /// (double-release detection).
    pub fn release(&self, index: usize) {
        assert!(index < self.capacity, "staging buffer index out of range");

        let mut inner = self.inner.lock().expect("staging pool lock poisoned");
        assert!(
            inner.in_use[index],
            "staging buffer {index} released twice (not currently in use)"
        );
        inner.in_use[index] = false;
        inner.free_list.push_back(index);
        drop(inner);

        let mut stats = self.stats.write().expect("staging stats lock poisoned");
        stats.releases += 1;
    }

    /// Get buffer info by index.
    pub fn buffer(&self, index: usize) -> Option<&StagingBuffer> {
        self.buffers.get(index)
    }

    /// Number of currently available buffers.
    pub fn available(&self) -> usize {
        let inner = self.inner.lock().expect("staging pool lock poisoned");
        inner.free_list.len()
    }

    /// Get a snapshot of pool statistics.
    pub fn stats(&self) -> StagingPoolStats {
        self.stats.read().expect("staging stats lock poisoned").clone()
    }
}

// ---------------------------------------------------------------------------
// Transfer ID generator
// ---------------------------------------------------------------------------

static TRANSFER_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate a unique transfer ID.
pub fn next_transfer_id() -> u64 {
    TRANSFER_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Core algorithms
// ---------------------------------------------------------------------------

/// Build contiguous runs from a page table lookup result.
///
/// Input: `page_addrs` from `PageTable::bulk_lookup(tensor_id)` -- physical
/// addresses of pages backing a logical tensor, returned in logical order.
///
/// Output: `FragmentAnalysis` with adjacent pages merged into contiguous runs
/// and a recommended transfer method.
///
/// Complexity: O(N) where N = page count. Single pass.
pub fn analyze_fragments(page_addrs: &[u64]) -> FragmentAnalysis {
    if page_addrs.is_empty() {
        return FragmentAnalysis::empty();
    }

    let mut runs = Vec::with_capacity(page_addrs.len() / 4);
    let mut run_start = page_addrs[0];
    let mut run_pages: u32 = 1;

    for i in 1..page_addrs.len() {
        let expected_next = page_addrs[i - 1] + PAGE_SIZE;
        if page_addrs[i] == expected_next {
            // Contiguous with previous -- extend current run.
            run_pages += 1;
        } else {
            // Gap -- close current run, start new one.
            runs.push(ContiguousRun {
                start_addr: run_start,
                length: run_pages as u64 * PAGE_SIZE,
                page_count: run_pages,
            });
            run_start = page_addrs[i];
            run_pages = 1;
        }
    }
    // Close final run.
    runs.push(ContiguousRun {
        start_addr: run_start,
        length: run_pages as u64 * PAGE_SIZE,
        page_count: run_pages,
    });

    let total_bytes = page_addrs.len() as u64 * PAGE_SIZE;
    let fragment_count = runs.len() as u32;
    let recommendation = decide_method(&runs, total_bytes);

    FragmentAnalysis {
        runs,
        fragment_count,
        total_bytes,
        recommendation,
    }
}

/// Decide transfer method based on fragment analysis.
///
/// Decision tree:
/// - 1 run -> SingleSge
/// - 2-30 runs, all >= 4KB -> HardwareScatterGather
/// - 2-30 runs, any < 4KB -> SoftwarePrePack (FragmentsTooSmall)
/// - > 30 runs -> SoftwarePrePack (ExceedsMaxSge)
fn decide_method(runs: &[ContiguousRun], _total_bytes: u64) -> TransferMethod {
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

    // Fragments exist but some are too small.
    TransferMethod::SoftwarePrePack {
        reason: PrePackReason::FragmentsTooSmall,
    }
}

/// Overlay compression decision on top of base method.
///
/// Called after `decide_method` when compressibility info is available.
/// If the data is large enough and compressible enough, overrides the
/// base method with `CompressedPrePack`.
pub fn consider_compression(
    base: TransferMethod,
    total_bytes: u64,
    estimated_ratio: f32,
) -> TransferMethod {
    // Only overlay compression on SoftwarePrePack — compression requires a
    // contiguous staging buffer which is already created in the pre-pack path.
    // Upgrading SingleSge or HardwareScatterGather to compressed would silently
    // regress a zero-copy hardware path to a gather-then-compress path.
    if matches!(base, TransferMethod::SoftwarePrePack { .. })
        && total_bytes >= COMPRESSION_SIZE_THRESHOLD
        && estimated_ratio >= MIN_COMPRESSION_RATIO
    {
        return TransferMethod::CompressedPrePack { estimated_ratio };
    }
    base
}

/// Build an ibv_sge-compatible list from contiguous runs.
///
/// # Panics
/// Debug-asserts that `runs.len() <= MAX_SGE_PER_WR`.
pub fn build_sge_list(runs: &[ContiguousRun]) -> ScatterGatherList {
    debug_assert!(runs.len() <= MAX_SGE_PER_WR);

    let mut entries = Vec::with_capacity(runs.len());
    let mut total_bytes = 0u64;

    for run in runs {
        assert!(
            run.length <= u32::MAX as u64,
            "contiguous run length {} exceeds u32::MAX for SgeEntry",
            run.length
        );
        entries.push(SgeEntry {
            addr: run.start_addr,
            length: run.length as u32,
            lkey: VRAM_MR_LKEY,
        });
        total_bytes += run.length;
    }

    ScatterGatherList {
        entries,
        total_bytes,
        run_count: runs.len() as u32,
    }
}

/// Split runs into chunks of <= MAX_SGE_PER_WR, each becoming one work request.
///
/// Returns a Vec of `ScatterGatherList`, each suitable for one `ibv_post_send` WR.
/// All WRs can be chained in a single `ibv_post_send` call (one doorbell ring).
pub fn chunk_into_wrs(runs: &[ContiguousRun]) -> Vec<ScatterGatherList> {
    runs.chunks(MAX_SGE_PER_WR)
        .map(|chunk| build_sge_list(chunk))
        .collect()
}

/// Build fragment info metadata from contiguous runs.
///
/// Used when constructing a `TransferDescriptor` for receiver-scatter cases.
pub fn build_fragment_info(runs: &[ContiguousRun]) -> Vec<FragmentInfo> {
    let mut offset = 0u64;
    runs.iter()
        .map(|run| {
            let info = FragmentInfo {
                logical_offset: offset,
                size: run.length,
                dest_addr: run.start_addr,
            };
            offset += run.length;
            info
        })
        .collect()
}

/// Build a `TransferDescriptor` for a given fragment analysis and wire method.
pub fn build_transfer_descriptor(
    analysis: &FragmentAnalysis,
    method: WireTransferMethod,
) -> TransferDescriptor {
    TransferDescriptor {
        transfer_id: next_transfer_id(),
        total_size: analysis.total_bytes,
        fragment_count: analysis.fragment_count,
        method,
        fragments: build_fragment_info(&analysis.runs),
        descriptor_crc: 0, // CRC computed by transport layer before send
    }
}

/// Scatter-gather transfer statistics.
#[derive(Debug, Default, Clone)]
pub struct ScatterGatherStats {
    pub single_sge_transfers: u64,
    pub hw_sg_transfers: u64,
    pub sw_prepack_transfers: u64,
    pub compressed_transfers: u64,
    pub total_bytes_transferred: u64,
    pub total_fragments_processed: u64,
}

/// Scatter-gather transfer engine that tracks statistics.
pub struct ScatterGatherEngine {
    stats: RwLock<ScatterGatherStats>,
    staging_pool: StagingPool,
}

impl ScatterGatherEngine {
    /// Create a new scatter-gather engine with default configuration.
    pub fn new() -> Self {
        Self::with_config(StagingPoolConfig::default())
    }

    /// Create a new scatter-gather engine with custom staging pool config.
    pub fn with_config(staging_config: StagingPoolConfig) -> Self {
        Self {
            stats: RwLock::new(ScatterGatherStats::default()),
            staging_pool: StagingPool::new(staging_config),
        }
    }

    /// Analyze page addresses and prepare a transfer plan.
    ///
    /// This is the main entry point. It analyzes fragmentation, decides the
    /// optimal transfer method, and updates internal statistics.
    pub fn prepare_transfer(&self, page_addrs: &[u64]) -> FragmentAnalysis {
        let analysis = analyze_fragments(page_addrs);

        let mut stats = self.stats.write().expect("sg stats lock poisoned");
        stats.total_fragments_processed += analysis.fragment_count as u64;
        stats.total_bytes_transferred += analysis.total_bytes;

        match &analysis.recommendation {
            TransferMethod::SingleSge => stats.single_sge_transfers += 1,
            TransferMethod::HardwareScatterGather { .. } => stats.hw_sg_transfers += 1,
            TransferMethod::SoftwarePrePack { .. } => stats.sw_prepack_transfers += 1,
            TransferMethod::CompressedPrePack { .. } => stats.compressed_transfers += 1,
        }

        analysis
    }

    /// Get a reference to the staging pool.
    pub fn staging_pool(&self) -> &StagingPool {
        &self.staging_pool
    }

    /// Get a snapshot of scatter-gather statistics.
    pub fn stats(&self) -> ScatterGatherStats {
        self.stats.read().expect("sg stats lock poisoned").clone()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create contiguous page addresses starting at `base`.
    fn contiguous_pages(base: u64, count: usize) -> Vec<u64> {
        (0..count).map(|i| base + i as u64 * PAGE_SIZE).collect()
    }

    // Helper: create page addresses with a gap after every `run_len` pages.
    fn fragmented_pages(base: u64, run_len: usize, num_runs: usize) -> Vec<u64> {
        let gap = PAGE_SIZE * 100; // large gap between runs
        let mut addrs = Vec::new();
        for r in 0..num_runs {
            let run_base = base + r as u64 * (run_len as u64 * PAGE_SIZE + gap);
            for p in 0..run_len {
                addrs.push(run_base + p as u64 * PAGE_SIZE);
            }
        }
        addrs
    }

    // -----------------------------------------------------------------------
    // Fragment analysis tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_pages() {
        let analysis = analyze_fragments(&[]);
        assert_eq!(analysis.fragment_count, 0);
        assert_eq!(analysis.total_bytes, 0);
        assert!(analysis.runs.is_empty());
        assert!(matches!(analysis.recommendation, TransferMethod::SingleSge));
    }

    #[test]
    fn test_single_page() {
        let addrs = vec![0x1000_0000];
        let analysis = analyze_fragments(&addrs);

        assert_eq!(analysis.fragment_count, 1);
        assert_eq!(analysis.total_bytes, PAGE_SIZE);
        assert_eq!(analysis.runs.len(), 1);
        assert_eq!(analysis.runs[0].page_count, 1);
        assert_eq!(analysis.runs[0].start_addr, 0x1000_0000);
        assert!(matches!(analysis.recommendation, TransferMethod::SingleSge));
    }

    #[test]
    fn test_contiguous_pages_single_run() {
        let addrs = contiguous_pages(0x2000_0000, 10);
        let analysis = analyze_fragments(&addrs);

        assert_eq!(analysis.fragment_count, 1);
        assert_eq!(analysis.total_bytes, 10 * PAGE_SIZE);
        assert_eq!(analysis.runs[0].page_count, 10);
        assert!(matches!(analysis.recommendation, TransferMethod::SingleSge));
    }

    #[test]
    fn test_two_fragments() {
        // 5 contiguous pages, gap, 5 more contiguous pages
        let addrs = fragmented_pages(0x1000_0000, 5, 2);
        let analysis = analyze_fragments(&addrs);

        assert_eq!(analysis.fragment_count, 2);
        assert_eq!(analysis.total_bytes, 10 * PAGE_SIZE);
        assert_eq!(analysis.runs[0].page_count, 5);
        assert_eq!(analysis.runs[1].page_count, 5);
        assert!(matches!(
            analysis.recommendation,
            TransferMethod::HardwareScatterGather { .. }
        ));
    }

    #[test]
    fn test_exactly_30_fragments_uses_hw_sg() {
        let addrs = fragmented_pages(0x1000_0000, 1, 30);
        let analysis = analyze_fragments(&addrs);

        assert_eq!(analysis.fragment_count, 30);
        // Each fragment is 64KB (one page) which is >= MIN_SGE_FRAGMENT_SIZE
        assert!(matches!(
            analysis.recommendation,
            TransferMethod::HardwareScatterGather { .. }
        ));

        if let TransferMethod::HardwareScatterGather { ref sge_list } = analysis.recommendation {
            assert_eq!(sge_list.entries.len(), 30);
            assert!(sge_list.is_valid());
        }
    }

    #[test]
    fn test_31_fragments_exceeds_max_sge() {
        let addrs = fragmented_pages(0x1000_0000, 1, 31);
        let analysis = analyze_fragments(&addrs);

        assert_eq!(analysis.fragment_count, 31);
        assert!(matches!(
            analysis.recommendation,
            TransferMethod::SoftwarePrePack {
                reason: PrePackReason::ExceedsMaxSge
            }
        ));
    }

    #[test]
    fn test_adjacent_pages_merged() {
        // Pages: 0, 64K, 128K (contiguous) then gap then 1MB, 1MB+64K (contiguous)
        let addrs = vec![
            0x0000_0000,
            0x0001_0000, // PAGE_SIZE = 64K = 0x10000
            0x0002_0000,
            0x0010_0000, // gap
            0x0011_0000,
        ];
        let analysis = analyze_fragments(&addrs);

        assert_eq!(analysis.fragment_count, 2);
        assert_eq!(analysis.runs[0].page_count, 3);
        assert_eq!(analysis.runs[0].length, 3 * PAGE_SIZE);
        assert_eq!(analysis.runs[1].page_count, 2);
        assert_eq!(analysis.runs[1].length, 2 * PAGE_SIZE);
    }

    // -----------------------------------------------------------------------
    // Decision engine tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compression_overlay_triggers() {
        let base = TransferMethod::SoftwarePrePack {
            reason: PrePackReason::ExceedsMaxSge,
        };
        let result = consider_compression(base, 3 * 1024 * 1024, 2.0);

        assert!(matches!(
            result,
            TransferMethod::CompressedPrePack { estimated_ratio } if (estimated_ratio - 2.0).abs() < f32::EPSILON
        ));
    }

    #[test]
    fn test_compression_overlay_below_threshold_size() {
        let base = TransferMethod::SoftwarePrePack {
            reason: PrePackReason::ExceedsMaxSge,
        };
        // 1MB is below the 2MB threshold
        let result = consider_compression(base, 1024 * 1024, 2.0);

        assert!(matches!(
            result,
            TransferMethod::SoftwarePrePack {
                reason: PrePackReason::ExceedsMaxSge
            }
        ));
    }

    #[test]
    fn test_compression_overlay_below_threshold_ratio() {
        let base = TransferMethod::SoftwarePrePack {
            reason: PrePackReason::ExceedsMaxSge,
        };
        // Ratio 1.2 is below the 1.5 minimum
        let result = consider_compression(base, 3 * 1024 * 1024, 1.2);

        assert!(matches!(
            result,
            TransferMethod::SoftwarePrePack {
                reason: PrePackReason::ExceedsMaxSge
            }
        ));
    }

    // -----------------------------------------------------------------------
    // SGE list builder tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_sge_list_basic() {
        let runs = vec![
            ContiguousRun {
                start_addr: 0x1000,
                length: PAGE_SIZE,
                page_count: 1,
            },
            ContiguousRun {
                start_addr: 0x2000_0000,
                length: 3 * PAGE_SIZE,
                page_count: 3,
            },
        ];

        let sge_list = build_sge_list(&runs);

        assert_eq!(sge_list.entries.len(), 2);
        assert_eq!(sge_list.run_count, 2);
        assert_eq!(sge_list.total_bytes, 4 * PAGE_SIZE);
        assert!(sge_list.is_valid());
        assert_eq!(sge_list.entries[0].addr, 0x1000);
        assert_eq!(sge_list.entries[0].length, PAGE_SIZE as u32);
        assert_eq!(sge_list.entries[1].addr, 0x2000_0000);
    }

    #[test]
    fn test_chunk_into_wrs_single_chunk() {
        let runs: Vec<_> = (0..10)
            .map(|i| ContiguousRun {
                start_addr: i * 0x100_0000,
                length: PAGE_SIZE,
                page_count: 1,
            })
            .collect();

        let wrs = chunk_into_wrs(&runs);
        assert_eq!(wrs.len(), 1);
        assert_eq!(wrs[0].entries.len(), 10);
    }

    #[test]
    fn test_chunk_into_wrs_multiple_chunks() {
        let runs: Vec<_> = (0..60)
            .map(|i| ContiguousRun {
                start_addr: i * 0x100_0000,
                length: PAGE_SIZE,
                page_count: 1,
            })
            .collect();

        let wrs = chunk_into_wrs(&runs);
        assert_eq!(wrs.len(), 2);
        assert_eq!(wrs[0].entries.len(), 30);
        assert_eq!(wrs[1].entries.len(), 30);
    }

    #[test]
    fn test_chunk_into_wrs_uneven() {
        let runs: Vec<_> = (0..35)
            .map(|i| ContiguousRun {
                start_addr: i * 0x100_0000,
                length: PAGE_SIZE,
                page_count: 1,
            })
            .collect();

        let wrs = chunk_into_wrs(&runs);
        assert_eq!(wrs.len(), 2);
        assert_eq!(wrs[0].entries.len(), 30);
        assert_eq!(wrs[1].entries.len(), 5);
    }

    // -----------------------------------------------------------------------
    // Transfer descriptor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_fragment_info() {
        let runs = vec![
            ContiguousRun {
                start_addr: 0x1000,
                length: 2 * PAGE_SIZE,
                page_count: 2,
            },
            ContiguousRun {
                start_addr: 0x5000_0000,
                length: PAGE_SIZE,
                page_count: 1,
            },
        ];

        let infos = build_fragment_info(&runs);
        assert_eq!(infos.len(), 2);
        assert_eq!(infos[0].logical_offset, 0);
        assert_eq!(infos[0].size, 2 * PAGE_SIZE);
        assert_eq!(infos[0].dest_addr, 0x1000);
        assert_eq!(infos[1].logical_offset, 2 * PAGE_SIZE);
        assert_eq!(infos[1].size, PAGE_SIZE);
    }

    #[test]
    fn test_build_transfer_descriptor() {
        let addrs = fragmented_pages(0x1000_0000, 3, 5);
        let analysis = analyze_fragments(&addrs);
        let desc = build_transfer_descriptor(&analysis, WireTransferMethod::HardwareGathered);

        assert_eq!(desc.total_size, 15 * PAGE_SIZE);
        assert_eq!(desc.fragment_count, 5);
        assert_eq!(desc.fragments.len(), 5);
        assert!(desc.transfer_id > 0);
    }

    #[test]
    fn test_transfer_descriptor_serialization() {
        let desc = TransferDescriptor {
            transfer_id: 42,
            total_size: 1024,
            fragment_count: 1,
            method: WireTransferMethod::Compressed {
                algorithm: CompressionAlgorithm::Lz4,
                compressed_size: 512,
            },
            fragments: vec![FragmentInfo {
                logical_offset: 0,
                size: 1024,
                dest_addr: 0x1000,
            }],
            descriptor_crc: 0,
        };

        let encoded = bincode::serialize(&desc).expect("serialize");
        let decoded: TransferDescriptor = bincode::deserialize(&encoded).expect("deserialize");

        assert_eq!(decoded.transfer_id, 42);
        assert_eq!(decoded.total_size, 1024);
        assert_eq!(
            decoded.method,
            WireTransferMethod::Compressed {
                algorithm: CompressionAlgorithm::Lz4,
                compressed_size: 512,
            }
        );
    }

    // -----------------------------------------------------------------------
    // Staging pool tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_staging_pool_acquire_release() {
        let pool = StagingPool::new(StagingPoolConfig {
            buffer_count: 4,
            buffer_size: 1024,
        });

        assert_eq!(pool.available(), 4);

        let b0 = pool.acquire().expect("should get buffer");
        assert_eq!(pool.available(), 3);

        let b1 = pool.acquire().expect("should get buffer");
        let b2 = pool.acquire().expect("should get buffer");
        let b3 = pool.acquire().expect("should get buffer");
        assert_eq!(pool.available(), 0);

        // Pool exhausted
        assert!(pool.acquire().is_none());

        pool.release(b1);
        assert_eq!(pool.available(), 1);

        let b4 = pool.acquire().expect("should get buffer after release");
        assert_eq!(b4, b1); // FIFO reuse

        pool.release(b0);
        pool.release(b2);
        pool.release(b3);
        pool.release(b4);
        assert_eq!(pool.available(), 4);
    }

    #[test]
    fn test_staging_pool_stats() {
        let pool = StagingPool::new(StagingPoolConfig {
            buffer_count: 2,
            buffer_size: 1024,
        });

        let b0 = pool.acquire().unwrap();
        let _b1 = pool.acquire().unwrap();
        assert!(pool.acquire().is_none()); // exhausted

        pool.release(b0);

        let stats = pool.stats();
        assert_eq!(stats.acquires, 2);
        assert_eq!(stats.exhausted_count, 1);
        assert_eq!(stats.releases, 1);
    }

    #[test]
    fn test_staging_pool_buffer_info() {
        let pool = StagingPool::new(StagingPoolConfig {
            buffer_count: 3,
            buffer_size: 4096,
        });

        let buf = pool.buffer(0).expect("buffer 0 should exist");
        assert_eq!(buf.size, 4096);
        assert!(pool.buffer(3).is_none());
    }

    // -----------------------------------------------------------------------
    // Engine tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_engine_tracks_stats() {
        let engine = ScatterGatherEngine::new();

        // Single contiguous transfer
        let addrs = contiguous_pages(0x1000_0000, 5);
        let analysis = engine.prepare_transfer(&addrs);
        assert!(matches!(analysis.recommendation, TransferMethod::SingleSge));

        // Fragmented transfer (5 runs)
        let addrs = fragmented_pages(0x2000_0000, 2, 5);
        let analysis = engine.prepare_transfer(&addrs);
        assert!(matches!(
            analysis.recommendation,
            TransferMethod::HardwareScatterGather { .. }
        ));

        let stats = engine.stats();
        assert_eq!(stats.single_sge_transfers, 1);
        assert_eq!(stats.hw_sg_transfers, 1);
        assert_eq!(stats.total_bytes_transferred, 5 * PAGE_SIZE + 10 * PAGE_SIZE);
    }

    #[test]
    fn test_engine_large_fragmentation() {
        let engine = ScatterGatherEngine::new();

        // 50 single-page fragments (exceeds MAX_SGE_PER_WR)
        let addrs = fragmented_pages(0x1000_0000, 1, 50);
        let analysis = engine.prepare_transfer(&addrs);
        assert!(matches!(
            analysis.recommendation,
            TransferMethod::SoftwarePrePack {
                reason: PrePackReason::ExceedsMaxSge
            }
        ));

        let stats = engine.stats();
        assert_eq!(stats.sw_prepack_transfers, 1);
    }

    // -----------------------------------------------------------------------
    // Edge case tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_pages_same_address_not_contiguous() {
        // Pathological: all pages at the same address (not realistic but tests logic)
        let addrs = vec![0x1000_0000, 0x1000_0000, 0x1000_0000];
        let analysis = analyze_fragments(&addrs);

        // Each page is NOT contiguous with the previous (same addr != addr + PAGE_SIZE)
        assert_eq!(analysis.fragment_count, 3);
    }

    #[test]
    fn test_sge_list_validity() {
        let runs: Vec<_> = (0..5)
            .map(|i| ContiguousRun {
                start_addr: i * 0x100_0000,
                length: PAGE_SIZE * 2,
                page_count: 2,
            })
            .collect();

        let sge_list = build_sge_list(&runs);
        assert!(sge_list.is_valid());
        assert_eq!(sge_list.total_bytes, 5 * 2 * PAGE_SIZE);
    }

    #[test]
    fn test_transfer_id_monotonic() {
        let id1 = next_transfer_id();
        let id2 = next_transfer_id();
        let id3 = next_transfer_id();

        assert!(id2 > id1);
        assert!(id3 > id2);
    }
}
