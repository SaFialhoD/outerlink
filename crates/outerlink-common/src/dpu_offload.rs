//! R16: BlueField DPU Offload
//!
//! Types, traits, and algorithms for offloading network transport, compression,
//! routing, and memory management to NVIDIA BlueField DPU ARM cores.
//!
//! The DPU sits between the NIC and the host CPU/GPU, providing hardware-accelerated
//! data path operations (RDMA, compression, flow steering) without host CPU involvement.
//!
//! # Architecture
//!
//! ```text
//! Host CPU/GPU <-> PCIe <-> BlueField DPU (ARM cores + ConnectX NIC) <-> Network
//! ```
//!
//! The `TransportBackend` trait abstracts over host-only (`HostTransport`) and
//! DPU-offloaded (`DpuTransport`) implementations, so `outerlink-server` works
//! identically with or without a DPU present.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// BlueField generation and capability detection
// ---------------------------------------------------------------------------

/// BlueField DPU generation with hardware capabilities.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BfGeneration {
    /// BlueField-2: 8 ARM A72 cores, 16 GB DRAM, deflate-only HW compress.
    BlueField2 {
        cores: u8,
        dram_gb: u8,
        has_lz4: bool,
    },
    /// BlueField-3: 16 ARM A78 cores, 32 GB DRAM, deflate + LZ4 HW compress.
    BlueField3 {
        cores: u8,
        dram_gb: u8,
        has_lz4: bool,
    },
}

impl BfGeneration {
    /// Create a standard BlueField-2 configuration.
    pub fn bf2() -> Self {
        Self::BlueField2 {
            cores: 8,
            dram_gb: 16,
            has_lz4: false,
        }
    }

    /// Create a standard BlueField-3 configuration.
    pub fn bf3() -> Self {
        Self::BlueField3 {
            cores: 16,
            dram_gb: 32,
            has_lz4: true,
        }
    }

    /// Number of ARM cores available.
    pub fn cores(&self) -> u8 {
        match self {
            Self::BlueField2 { cores, .. } => *cores,
            Self::BlueField3 { cores, .. } => *cores,
        }
    }

    /// DRAM capacity in gigabytes.
    pub fn dram_gb(&self) -> u8 {
        match self {
            Self::BlueField2 { dram_gb, .. } => *dram_gb,
            Self::BlueField3 { dram_gb, .. } => *dram_gb,
        }
    }

    /// Whether the hardware LZ4 compression engine is available.
    pub fn has_lz4(&self) -> bool {
        match self {
            Self::BlueField2 { has_lz4, .. } => *has_lz4,
            Self::BlueField3 { has_lz4, .. } => *has_lz4,
        }
    }
}

// ---------------------------------------------------------------------------
// DPU Configuration
// ---------------------------------------------------------------------------

/// Configuration for the DPU offload subsystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpuConfig {
    /// Enable transport (RDMA/network) offload to DPU.
    pub transport_offload: bool,
    /// Enable hardware compression offload to DPU.
    pub compression_offload: bool,
    /// Enable speculative prefetch on DPU ARM cores.
    pub prefetch_offload: bool,
    /// Enable direct BAR1 GPU VRAM access from DPU.
    pub bar1_direct: bool,

    /// Maximum number of concurrent connections managed by the DPU.
    pub max_connections: u32,
    /// Maximum number of GPUs the DPU page table tracks.
    pub page_table_max_gpus: u8,
    /// Prefetch cache size in megabytes (in DPU DRAM).
    pub prefetch_cache_mb: u32,
    /// Compression staging buffer pool size in megabytes (in DPU DRAM).
    pub compress_staging_mb: u32,

    /// BlueField generation (detected at startup).
    pub bf_generation: BfGeneration,
}

impl Default for DpuConfig {
    fn default() -> Self {
        Self {
            transport_offload: true,
            compression_offload: true,
            prefetch_offload: true,
            bar1_direct: false,

            max_connections: 256,
            page_table_max_gpus: 4,
            prefetch_cache_mb: 4096,
            compress_staging_mb: 2048,

            bf_generation: BfGeneration::bf2(),
        }
    }
}

// ---------------------------------------------------------------------------
// DPU Statistics
// ---------------------------------------------------------------------------

/// Runtime statistics for the DPU offload subsystem.
///
/// All fields are atomic for lock-free concurrent updates from multiple
/// DPU subsystems (transport, compress, prefetch, router).
#[derive(Debug)]
pub struct DpuStats {
    pub bytes_transferred: AtomicU64,
    pub bytes_compressed: AtomicU64,
    /// Average compression ratio as fixed-point 8.24 (multiply by 2^-24 for float).
    pub compression_ratio_avg: AtomicU32,
    pub prefetch_hits: AtomicU64,
    pub prefetch_misses: AtomicU64,
    pub page_faults_served: AtomicU64,
    pub connections_active: AtomicU32,
    pub uptime_secs: AtomicU64,
}

impl DpuStats {
    /// Create zeroed statistics.
    pub fn new() -> Self {
        Self {
            bytes_transferred: AtomicU64::new(0),
            bytes_compressed: AtomicU64::new(0),
            compression_ratio_avg: AtomicU32::new(0),
            prefetch_hits: AtomicU64::new(0),
            prefetch_misses: AtomicU64::new(0),
            page_faults_served: AtomicU64::new(0),
            connections_active: AtomicU32::new(0),
            uptime_secs: AtomicU64::new(0),
        }
    }

    /// Get prefetch hit rate as a fraction (0.0 to 1.0).
    /// Returns 0.0 if no prefetch operations have occurred.
    pub fn prefetch_hit_rate(&self) -> f64 {
        let hits = self.prefetch_hits.load(Ordering::Relaxed);
        let misses = self.prefetch_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            return 0.0;
        }
        hits as f64 / total as f64
    }

    /// Get the average compression ratio as a float.
    /// Value of 0.5 means data was compressed to 50% of original size.
    pub fn compression_ratio(&self) -> f64 {
        let raw = self.compression_ratio_avg.load(Ordering::Relaxed);
        raw as f64 / (1u64 << 24) as f64
    }

    /// Set the compression ratio from a float value (0.0 to 1.0).
    pub fn set_compression_ratio(&self, ratio: f64) {
        let fixed = (ratio * (1u64 << 24) as f64) as u32;
        self.compression_ratio_avg.store(fixed, Ordering::Relaxed);
    }

    /// Take a snapshot of all stats as a plain struct (for serialization).
    pub fn snapshot(&self) -> DpuStatsSnapshot {
        DpuStatsSnapshot {
            bytes_transferred: self.bytes_transferred.load(Ordering::Relaxed),
            bytes_compressed: self.bytes_compressed.load(Ordering::Relaxed),
            compression_ratio: self.compression_ratio(),
            prefetch_hits: self.prefetch_hits.load(Ordering::Relaxed),
            prefetch_misses: self.prefetch_misses.load(Ordering::Relaxed),
            prefetch_hit_rate: self.prefetch_hit_rate(),
            page_faults_served: self.page_faults_served.load(Ordering::Relaxed),
            connections_active: self.connections_active.load(Ordering::Relaxed),
            uptime_secs: self.uptime_secs.load(Ordering::Relaxed),
        }
    }
}

impl Default for DpuStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable snapshot of DPU statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpuStatsSnapshot {
    pub bytes_transferred: u64,
    pub bytes_compressed: u64,
    pub compression_ratio: f64,
    pub prefetch_hits: u64,
    pub prefetch_misses: u64,
    pub prefetch_hit_rate: f64,
    pub page_faults_served: u64,
    pub connections_active: u32,
    pub uptime_secs: u64,
}

// ---------------------------------------------------------------------------
// Transport Backend Abstraction
// ---------------------------------------------------------------------------

/// Compression algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgo {
    /// No compression.
    None,
    /// Deflate (available on BF-2 and BF-3).
    Deflate,
    /// LZ4 (available on BF-3 only).
    Lz4,
}

/// Hint for whether to compress a transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressHint {
    /// Let the backend decide based on data size and compressibility.
    Auto,
    /// Always compress.
    Always,
    /// Never compress.
    Never,
}

/// Capabilities reported by a transport backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendCapabilities {
    /// Whether hardware compression is available.
    pub has_hw_compression: bool,
    /// Available hardware compression algorithms.
    pub hw_compression_algos: Vec<CompressionAlgo>,
    /// Whether direct BAR1 GPU VRAM access is available.
    pub has_bar1_direct: bool,
    /// Maximum link bandwidth in Gbps.
    pub max_bandwidth_gbps: u32,
    /// Whether speculative prefetch is supported.
    pub supports_prefetch: bool,
}

impl BackendCapabilities {
    /// Capabilities for a host-only transport (no DPU).
    pub fn host_only() -> Self {
        Self {
            has_hw_compression: false,
            hw_compression_algos: vec![],
            has_bar1_direct: false,
            max_bandwidth_gbps: 100, // Typical NIC speed
            supports_prefetch: false,
        }
    }

    /// Capabilities for a BlueField-2 DPU.
    pub fn bf2() -> Self {
        Self {
            has_hw_compression: true,
            hw_compression_algos: vec![CompressionAlgo::Deflate],
            has_bar1_direct: true,
            max_bandwidth_gbps: 200,
            supports_prefetch: true,
        }
    }

    /// Capabilities for a BlueField-3 DPU.
    pub fn bf3() -> Self {
        Self {
            has_hw_compression: true,
            hw_compression_algos: vec![CompressionAlgo::Deflate, CompressionAlgo::Lz4],
            has_bar1_direct: true,
            max_bandwidth_gbps: 400,
            supports_prefetch: true,
        }
    }
}

/// Opaque node identifier within the OuterLink cluster.
pub type NodeId = u32;

/// Opaque connection identifier.
pub type ConnectionId = u64;

// ---------------------------------------------------------------------------
// Memory Location
// ---------------------------------------------------------------------------

/// Where a memory region lives -- used in transfer requests.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemLocation {
    /// Host-pinned (page-locked) CPU memory.
    HostPinned { addr: u64 },
    /// GPU VRAM on a specific local GPU.
    GpuVram { gpu_id: u8, addr: u64 },
    /// VRAM on a remote node's GPU.
    RemoteNode {
        node_id: NodeId,
        gpu_id: u8,
        addr: u64,
    },
}

// ---------------------------------------------------------------------------
// Transfer Request / Response
// ---------------------------------------------------------------------------

/// A data transfer request from host to DPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferRequest {
    /// Unique request ID for correlation.
    pub id: u64,
    /// Source memory location.
    pub src: MemLocation,
    /// Destination memory location.
    pub dst: MemLocation,
    /// Transfer size in bytes.
    pub size: u64,
    /// Compression hint.
    pub compress: CompressHint,
}

/// Result of a completed transfer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferComplete {
    /// Request ID that was completed.
    pub id: u64,
    /// Actual bytes transferred (may differ from request if compressed).
    pub bytes_transferred: u64,
    /// Transfer duration in microseconds.
    pub elapsed_us: u32,
}

// ---------------------------------------------------------------------------
// Host <-> DPU Protocol Messages
// ---------------------------------------------------------------------------

/// Messages sent from the host to the DPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HostToDpuMsg {
    /// Request a data transfer.
    TransferRequest {
        id: u64,
        src: MemLocation,
        dst: MemLocation,
        size: u64,
        compress: CompressHint,
    },
    /// Notify DPU of a new GPU memory allocation (keeps page table in sync).
    AllocNotify { gpu_id: u8, addr: u64, size: u64 },
    /// Notify DPU of a GPU memory free.
    FreeNotify { gpu_id: u8, addr: u64, size: u64 },
    /// Synchronization barrier.
    SyncBarrier { id: u64 },
    /// Connect to a remote peer.
    ConnectPeer { peer: String },
    /// Disconnect from a remote peer.
    DisconnectPeer { conn_id: ConnectionId },
    /// Update DPU configuration at runtime.
    ConfigUpdate { config: DpuConfig },
    /// Request graceful shutdown.
    Shutdown,
}

/// Messages sent from the DPU to the host.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DpuToHostMsg {
    /// Transfer completed successfully.
    TransferComplete {
        id: u64,
        bytes_transferred: u64,
        elapsed_us: u32,
    },
    /// Transfer failed.
    TransferFailed { id: u64, error: TransportErrorCode },
    /// Page fault: a remote node is requesting a page we own.
    PageFault {
        gpu_id: u8,
        page: u64,
        requestor: NodeId,
    },
    /// A remote peer connected.
    PeerConnected { conn_id: ConnectionId, peer: String },
    /// A remote peer disconnected.
    PeerDisconnected {
        conn_id: ConnectionId,
        reason: DisconnectReason,
    },
    /// Periodic stats push from DPU.
    Stats { stats: DpuStatsSnapshot },
    /// Error on the DPU.
    Error { code: DpuErrorCode, message: String },
}

/// Transport error codes sent from DPU to host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransportErrorCode {
    NoRoute,
    ConnectionLost,
    Timeout,
    CompressionFailed,
    RdmaError,
    Bar1AccessDenied,
    ResourceExhausted,
}

/// Reason for peer disconnection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisconnectReason {
    /// Clean shutdown initiated by either side.
    Graceful,
    /// Connection lost (network failure, peer crash).
    LinkDown,
    /// Idle timeout.
    IdleTimeout,
    /// Resource pressure on DPU.
    ResourcePressure,
}

/// Error codes originating from the DPU itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DpuErrorCode {
    DocaInitFailed,
    CommChannelDown,
    DramExhausted,
    TooManyConnections,
    Bar1MapFailed,
    InternalError,
}

// ---------------------------------------------------------------------------
// BAR1 Region Descriptor
// ---------------------------------------------------------------------------

/// Describes a BAR1-mapped GPU VRAM region accessible from the DPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bar1Region {
    /// Which GPU this region belongs to.
    pub gpu_id: u8,
    /// BAR1 physical base address.
    pub base_addr: u64,
    /// Mapped region size in bytes.
    pub size: usize,
    /// Whether resizable BAR (ReBAR) is enabled, giving full VRAM access.
    pub rebar_enabled: bool,
}

impl Bar1Region {
    /// Check if an address range falls within this BAR1 mapping.
    pub fn contains(&self, addr: u64, len: usize) -> bool {
        addr >= self.base_addr && (addr + len as u64) <= (self.base_addr + self.size as u64)
    }
}

// ---------------------------------------------------------------------------
// Page Table Types (R10 integration on DPU)
// ---------------------------------------------------------------------------

/// State of a memory page in the DPU's mirror of the host page table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PageState {
    /// Page resides on the local node's GPU VRAM.
    Local,
    /// Page resides on a remote node.
    Remote(NodeId),
    /// Page is currently being migrated.
    Migrating,
    /// Page has been freed and is invalid.
    Invalid,
}

/// Flags for page entries.
pub mod page_flags {
    /// Page is pinned and must not be evicted/migrated.
    pub const PINNED: u8 = 1 << 0;
    /// Page was brought in by prefetch.
    pub const PREFETCHED: u8 = 1 << 1;
    /// Page has been modified since last sync.
    pub const DIRTY: u8 = 1 << 2;
}

/// A single page entry in the DPU's page table mirror.
/// Compact (12 bytes) for cache efficiency -- 6M entries (24 GB / 4 KB) = ~72 MB.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageEntry {
    /// Which node currently owns this page.
    pub owner_node: NodeId,
    /// Current page state.
    pub state: PageState,
    /// Compact timestamp (relative, wrapping).
    pub last_access: u32,
    /// Access counter for hot/cold classification.
    pub access_count: u16,
    /// Flags (pinned, prefetched, dirty).
    pub flags: u8,
}

impl PageEntry {
    /// Create a new local page entry.
    pub fn new_local(owner: NodeId) -> Self {
        Self {
            owner_node: owner,
            state: PageState::Local,
            last_access: 0,
            access_count: 0,
            flags: 0,
        }
    }

    /// Whether this page is pinned (cannot be migrated).
    pub fn is_pinned(&self) -> bool {
        self.flags & page_flags::PINNED != 0
    }

    /// Whether this page was brought in by prefetch.
    pub fn is_prefetched(&self) -> bool {
        self.flags & page_flags::PREFETCHED != 0
    }

    /// Whether this page has been modified.
    pub fn is_dirty(&self) -> bool {
        self.flags & page_flags::DIRTY != 0
    }

    /// Record an access to this page (increments counter, updates timestamp).
    pub fn record_access(&mut self, timestamp: u32) {
        self.last_access = timestamp;
        self.access_count = self.access_count.saturating_add(1);
    }
}

/// Per-GPU page map on the DPU.
#[derive(Debug)]
pub struct GpuPageMap {
    pub gpu_id: u8,
    pub node_id: NodeId,
    /// Dense array of page entries (page number is the index).
    pub entries: Vec<PageEntry>,
    pub total_pages: u32,
}

impl GpuPageMap {
    /// Create a new page map for a GPU with the given number of pages.
    /// All pages start as Local, owned by the specified node.
    pub fn new(gpu_id: u8, node_id: NodeId, total_pages: u32) -> Self {
        let entries = (0..total_pages)
            .map(|_| PageEntry::new_local(node_id))
            .collect();
        Self {
            gpu_id,
            node_id,
            entries,
            total_pages,
        }
    }

    /// Get a page entry by page number.
    pub fn get(&self, page: u32) -> Option<&PageEntry> {
        self.entries.get(page as usize)
    }

    /// Get a mutable page entry by page number.
    pub fn get_mut(&mut self, page: u32) -> Option<&mut PageEntry> {
        self.entries.get_mut(page as usize)
    }

    /// Count pages in a given state.
    pub fn count_in_state(&self, state: PageState) -> usize {
        self.entries.iter().filter(|e| e.state == state).count()
    }

    /// Count pinned pages.
    pub fn count_pinned(&self) -> usize {
        self.entries.iter().filter(|e| e.is_pinned()).count()
    }
}

// ---------------------------------------------------------------------------
// Compression Types (R14 integration on DPU)
// ---------------------------------------------------------------------------

/// Adaptive compression configuration for the DPU compressor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveCompressConfig {
    /// Minimum data size (bytes) to attempt compression.
    pub min_size: u32,
    /// Compressibility ratio below which we skip compression.
    /// E.g., 0.85 means we need at least 15% savings.
    pub ratio_threshold: f32,
    /// Sample interval: check compressibility every N transfers.
    pub sample_interval: u32,
    /// Preferred algorithm (auto-selected based on BF generation).
    pub preferred_algo: CompressionAlgo,
}

impl Default for AdaptiveCompressConfig {
    fn default() -> Self {
        Self {
            min_size: 4096,
            ratio_threshold: 0.85,
            sample_interval: 64,
            preferred_algo: CompressionAlgo::Deflate,
        }
    }
}

impl AdaptiveCompressConfig {
    /// Create config tuned for a BlueField-3 (prefers LZ4).
    pub fn for_bf3() -> Self {
        Self {
            preferred_algo: CompressionAlgo::Lz4,
            ..Self::default()
        }
    }
}

/// Per-connection compression statistics tracked on the DPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    pub total_bytes_in: u64,
    pub total_bytes_out: u64,
    pub last_ratio: f32,
    pub samples_taken: u32,
    pub algo_used: CompressionAlgo,
}

impl CompressionStats {
    /// Create new zeroed stats for a given algorithm.
    pub fn new(algo: CompressionAlgo) -> Self {
        Self {
            total_bytes_in: 0,
            total_bytes_out: 0,
            last_ratio: 1.0,
            samples_taken: 0,
            algo_used: algo,
        }
    }

    /// Overall compression ratio (compressed / original). Lower is better.
    pub fn overall_ratio(&self) -> f64 {
        if self.total_bytes_in == 0 {
            return 1.0;
        }
        self.total_bytes_out as f64 / self.total_bytes_in as f64
    }
}

// ---------------------------------------------------------------------------
// Offload Decision Algorithm
// ---------------------------------------------------------------------------

/// Result of the DPU detection and configuration algorithm.
#[derive(Debug, Clone)]
pub enum DetectedBackend {
    /// No DPU found; use host-only transport.
    HostOnly(BackendCapabilities),
    /// DPU found and configured.
    DpuOffloaded {
        capabilities: BackendCapabilities,
        config: DpuConfig,
        bar1_regions: Vec<Bar1Region>,
    },
}

/// Per-transfer offload decision made by the DPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffloadDecision {
    /// Send without compression.
    SendDirect,
    /// Compress then send.
    CompressThenSend(CompressionAlgo),
    /// No route to destination.
    NoRoute,
}

/// Decide whether to compress a transfer based on size, hint, and connection stats.
///
/// Implements the per-transfer offload decision algorithm from the R16 preplan:
/// 1. If size < min_size, skip compression.
/// 2. If hint is Never, skip. If Always, compress.
/// 3. For Auto: use sampled compressibility ratio from connection stats.
pub fn decide_transfer_compression(
    size: u64,
    hint: CompressHint,
    config: &AdaptiveCompressConfig,
    conn_stats: Option<&CompressionStats>,
) -> OffloadDecision {
    // Too small to be worth compressing
    if size < config.min_size as u64 {
        return OffloadDecision::SendDirect;
    }

    match hint {
        CompressHint::Never => OffloadDecision::SendDirect,
        CompressHint::Always => OffloadDecision::CompressThenSend(config.preferred_algo),
        CompressHint::Auto => {
            // Check connection stats for last sampled ratio
            if let Some(stats) = conn_stats {
                if stats.last_ratio < config.ratio_threshold {
                    OffloadDecision::CompressThenSend(config.preferred_algo)
                } else {
                    OffloadDecision::SendDirect
                }
            } else {
                // No stats yet -- try compression to establish baseline
                OffloadDecision::CompressThenSend(config.preferred_algo)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ARM Core Allocation
// ---------------------------------------------------------------------------

/// Role assigned to a DPU ARM core.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreRole {
    /// Reserved for Linux kernel and DOCA runtime.
    System,
    /// Network transport (RDMA I/O, connection management).
    Transport,
    /// Hardware compression task management.
    Compress,
    /// Page table tracking and migration.
    PageTable,
    /// Speculative prefetch pattern monitoring.
    Prefetch,
    /// Routing decisions and flow rule management.
    Router,
    /// Overflow: exception handling, stats, future features.
    Overflow,
}

/// ARM core allocation plan for a BlueField DPU.
#[derive(Debug, Clone)]
pub struct CoreAllocation {
    /// Core index -> assigned role.
    pub assignments: Vec<(u8, CoreRole)>,
}

/// Allocate ARM cores to DPU subsystem roles based on BlueField generation.
///
/// Core 0 is always reserved for the Linux kernel and DOCA runtime.
/// BF-2 (8 cores) gets one core per subsystem.
/// BF-3 (16 cores) gets multiple cores for high-throughput subsystems.
pub fn allocate_cores(bf_gen: &BfGeneration) -> CoreAllocation {
    let total_cores = bf_gen.cores();
    let mut assignments = Vec::with_capacity(total_cores as usize);

    // Core 0 is always system
    assignments.push((0, CoreRole::System));

    if total_cores <= 8 {
        // BF-2 layout: 1 core per subsystem
        let roles = [
            CoreRole::Transport,
            CoreRole::Transport,
            CoreRole::Compress,
            CoreRole::PageTable,
            CoreRole::Prefetch,
            CoreRole::Router,
            CoreRole::Overflow,
        ];
        for (i, role) in roles.iter().enumerate() {
            let core_id = (i + 1) as u8;
            if core_id < total_cores {
                assignments.push((core_id, *role));
            }
        }
    } else {
        // BF-3 layout: multiple cores for high-throughput subsystems
        // Cores 1-3: Transport (3 cores)
        for i in 1..=3 {
            assignments.push((i, CoreRole::Transport));
        }
        // Core 4: Transport (4th core for BF-3's higher throughput)
        assignments.push((4, CoreRole::Transport));
        // Core 5: Compress
        assignments.push((5, CoreRole::Compress));
        // Cores 6-7: PageTable
        assignments.push((6, CoreRole::PageTable));
        assignments.push((7, CoreRole::PageTable));
        // Cores 8-9: Prefetch
        assignments.push((8, CoreRole::Prefetch));
        assignments.push((9, CoreRole::Prefetch));
        // Cores 10-11: Router
        assignments.push((10, CoreRole::Router));
        assignments.push((11, CoreRole::Router));
        // Cores 12-15: Overflow
        for i in 12..total_cores {
            assignments.push((i, CoreRole::Overflow));
        }
    }

    CoreAllocation { assignments }
}

impl CoreAllocation {
    /// Get all cores assigned to a specific role.
    pub fn cores_for_role(&self, role: CoreRole) -> Vec<u8> {
        self.assignments
            .iter()
            .filter(|(_, r)| *r == role)
            .map(|(c, _)| *c)
            .collect()
    }

    /// Total number of cores assigned.
    pub fn total_cores(&self) -> usize {
        self.assignments.len()
    }
}

// ---------------------------------------------------------------------------
// DPU Capability Detection (stub for real hardware)
// ---------------------------------------------------------------------------

/// Result of probing for a BlueField DPU on the system.
#[derive(Debug, Clone)]
pub enum DpuProbeResult {
    /// No BlueField DPU detected.
    NoDpu,
    /// DPU found but not in DPU mode (e.g., NIC-only mode).
    WrongMode { pci_addr: String },
    /// DPU found but outerlink-dpu service is not running.
    ServiceNotRunning { pci_addr: String, generation: BfGeneration },
    /// DPU is ready and fully operational.
    Ready {
        pci_addr: String,
        generation: BfGeneration,
        capabilities: BackendCapabilities,
    },
}

/// Probe the system for BlueField DPU hardware.
///
/// In production this would enumerate DOCA devices via `doca_devinfo_list_create()`.
/// This implementation is a stub that returns `NoDpu` -- real hardware interaction
/// requires the DOCA SDK and a physical BlueField DPU.
///
/// # TODO: requires hardware
/// Real implementation needs:
/// - DOCA SDK FFI bindings
/// - PCI device enumeration
/// - Comm Channel handshake with outerlink-dpu service
pub fn probe_dpu() -> DpuProbeResult {
    // TODO: requires hardware -- DOCA SDK device enumeration
    DpuProbeResult::NoDpu
}

/// Configure the DPU subsystems based on detected capabilities.
///
/// This is the second half of the `detect_and_configure_backend()` algorithm:
/// after probing, configure which offload subsystems to enable based on
/// the DPU generation and available resources.
pub fn configure_backend(
    probe: &DpuProbeResult,
    gpu_count: u8,
) -> DetectedBackend {
    match probe {
        DpuProbeResult::NoDpu | DpuProbeResult::WrongMode { .. } | DpuProbeResult::ServiceNotRunning { .. } => {
            DetectedBackend::HostOnly(BackendCapabilities::host_only())
        }
        DpuProbeResult::Ready {
            generation,
            capabilities,
            ..
        } => {
            let mut config = DpuConfig {
                bf_generation: generation.clone(),
                page_table_max_gpus: gpu_count,
                ..DpuConfig::default()
            };

            // Enable compression offload if hardware supports it
            config.compression_offload = capabilities.has_hw_compression;

            // Enable prefetch only if enough cores (need at least 6 for prefetch core)
            config.prefetch_offload = generation.cores() >= 6;

            // BAR1 direct requires explicit detection of PCIe topology
            // (probe doesn't determine this -- separate step)
            config.bar1_direct = capabilities.has_bar1_direct;

            DetectedBackend::DpuOffloaded {
                capabilities: capabilities.clone(),
                config,
                bar1_regions: Vec::new(), // Populated by BAR1 enumeration
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DPU Service State (top-level, for the DPU daemon)
// ---------------------------------------------------------------------------

/// Top-level state for the outerlink-dpu service running on BlueField ARM cores.
///
/// This is the "god object" that owns all DPU subsystems. In production, this
/// would be a long-running daemon with async event loops on each assigned core.
///
/// # TODO: requires hardware
/// The actual service implementation requires:
/// - DOCA SDK initialization
/// - DOCA Comm Channel for host communication
/// - RDMA context creation
/// - ARM core pinning
pub struct DpuService {
    /// Configuration.
    pub config: DpuConfig,
    /// Runtime statistics.
    pub stats: Arc<DpuStats>,
    /// Core allocation plan.
    pub core_allocation: CoreAllocation,
    /// Per-GPU page maps (R10 integration).
    pub page_maps: Vec<GpuPageMap>,
    /// Per-connection compression stats (R14 integration).
    pub compression_stats: HashMap<ConnectionId, CompressionStats>,
    /// Active connection IDs (for disconnect tracking / underflow prevention).
    active_connections: std::collections::HashSet<ConnectionId>,
    /// Next connection ID for allocation.
    next_conn_id: ConnectionId,
    /// Adaptive compression config.
    pub compress_config: AdaptiveCompressConfig,
}

impl DpuService {
    /// Create a new DPU service with the given configuration.
    pub fn new(config: DpuConfig) -> Self {
        let core_allocation = allocate_cores(&config.bf_generation);
        let stats = Arc::new(DpuStats::new());

        let compress_config = if config.bf_generation.has_lz4() {
            AdaptiveCompressConfig::for_bf3()
        } else {
            AdaptiveCompressConfig::default()
        };

        Self {
            config,
            stats,
            core_allocation,
            page_maps: Vec::new(),
            compression_stats: HashMap::new(),
            active_connections: std::collections::HashSet::new(),
            next_conn_id: 1,
            compress_config,
        }
    }

    /// Register a GPU's page table on the DPU.
    pub fn register_gpu(&mut self, gpu_id: u8, node_id: NodeId, total_pages: u32) {
        self.page_maps.push(GpuPageMap::new(gpu_id, node_id, total_pages));
    }

    /// Handle a host-to-DPU message.
    ///
    /// Returns an optional response message to send back to the host.
    pub fn handle_message(&mut self, msg: HostToDpuMsg) -> Option<DpuToHostMsg> {
        match msg {
            HostToDpuMsg::AllocNotify { gpu_id, addr, size } => {
                // Update page table for the allocation
                let page_size: u64 = 4096;
                let start_page = addr / page_size;
                let num_pages = (size + page_size - 1) / page_size;
                if let Some(map) = self.page_maps.iter_mut().find(|m| m.gpu_id == gpu_id) {
                    for p in start_page..(start_page + num_pages) {
                        if let Some(entry) = map.get_mut(p as u32) {
                            entry.state = PageState::Local;
                            entry.access_count = 0;
                        }
                    }
                }
                None
            }
            HostToDpuMsg::FreeNotify { gpu_id, addr, size } => {
                let page_size: u64 = 4096;
                let start_page = addr / page_size;
                let num_pages = (size + page_size - 1) / page_size;
                if let Some(map) = self.page_maps.iter_mut().find(|m| m.gpu_id == gpu_id) {
                    for p in start_page..(start_page + num_pages) {
                        if let Some(entry) = map.get_mut(p as u32) {
                            entry.state = PageState::Invalid;
                        }
                    }
                }
                None
            }
            HostToDpuMsg::Shutdown => {
                // Graceful shutdown -- return stats before stopping
                Some(DpuToHostMsg::Stats {
                    stats: self.stats.snapshot(),
                })
            }
            HostToDpuMsg::TransferRequest {
                id,
                size,
                compress,
                ..
            } => {
                // Decide on compression
                let conn_stats = None; // Would look up by connection
                let decision =
                    decide_transfer_compression(size, compress, &self.compress_config, conn_stats);

                // In production, this would initiate the actual RDMA transfer.
                // For now, return a simulated completion.
                // TODO: requires hardware -- DOCA RDMA transfer
                match decision {
                    OffloadDecision::NoRoute => Some(DpuToHostMsg::TransferFailed {
                        id,
                        error: TransportErrorCode::NoRoute,
                    }),
                    _ => {
                        self.stats
                            .bytes_transferred
                            .fetch_add(size, Ordering::Relaxed);
                        Some(DpuToHostMsg::TransferComplete {
                            id,
                            bytes_transferred: size,
                            elapsed_us: 0, // TODO: actual timing
                        })
                    }
                }
            }
            HostToDpuMsg::ConnectPeer { peer } => {
                // TODO: requires hardware -- DOCA RDMA connection
                let conn_id = self.next_conn_id;
                self.next_conn_id += 1;
                self.active_connections.insert(conn_id);
                self.stats.connections_active.fetch_add(1, Ordering::Relaxed);
                Some(DpuToHostMsg::PeerConnected { conn_id, peer })
            }
            HostToDpuMsg::DisconnectPeer { conn_id } => {
                // Only decrement if connection was actually tracked (prevents underflow).
                if self.active_connections.remove(&conn_id) {
                    self.stats.connections_active.fetch_sub(1, Ordering::Relaxed);
                }
                self.compression_stats.remove(&conn_id);
                Some(DpuToHostMsg::PeerDisconnected {
                    conn_id,
                    reason: DisconnectReason::Graceful,
                })
            }
            HostToDpuMsg::SyncBarrier { .. } | HostToDpuMsg::ConfigUpdate { .. } => {
                // TODO: SyncBarrier should return an ack when all prior messages are processed.
                // TODO: ConfigUpdate should apply to self.config.
                None
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- BfGeneration tests --

    #[test]
    fn test_bf2_defaults() {
        let gen = BfGeneration::bf2();
        assert_eq!(gen.cores(), 8);
        assert_eq!(gen.dram_gb(), 16);
        assert!(!gen.has_lz4());
    }

    #[test]
    fn test_bf3_defaults() {
        let gen = BfGeneration::bf3();
        assert_eq!(gen.cores(), 16);
        assert_eq!(gen.dram_gb(), 32);
        assert!(gen.has_lz4());
    }

    #[test]
    fn test_bf_generation_custom() {
        let gen = BfGeneration::BlueField2 {
            cores: 4,
            dram_gb: 8,
            has_lz4: false,
        };
        assert_eq!(gen.cores(), 4);
        assert_eq!(gen.dram_gb(), 8);
    }

    // -- DpuConfig tests --

    #[test]
    fn test_dpu_config_default() {
        let config = DpuConfig::default();
        assert!(config.transport_offload);
        assert!(config.compression_offload);
        assert!(config.prefetch_offload);
        assert!(!config.bar1_direct);
        assert_eq!(config.max_connections, 256);
        assert_eq!(config.page_table_max_gpus, 4);
        assert_eq!(config.prefetch_cache_mb, 4096);
        assert_eq!(config.compress_staging_mb, 2048);
    }

    // -- DpuStats tests --

    #[test]
    fn test_dpu_stats_new_zeroed() {
        let stats = DpuStats::new();
        assert_eq!(stats.bytes_transferred.load(Ordering::Relaxed), 0);
        assert_eq!(stats.connections_active.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_dpu_stats_prefetch_hit_rate_no_ops() {
        let stats = DpuStats::new();
        assert_eq!(stats.prefetch_hit_rate(), 0.0);
    }

    #[test]
    fn test_dpu_stats_prefetch_hit_rate() {
        let stats = DpuStats::new();
        stats.prefetch_hits.store(75, Ordering::Relaxed);
        stats.prefetch_misses.store(25, Ordering::Relaxed);
        assert!((stats.prefetch_hit_rate() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_dpu_stats_compression_ratio_roundtrip() {
        let stats = DpuStats::new();
        stats.set_compression_ratio(0.65);
        let ratio = stats.compression_ratio();
        assert!((ratio - 0.65).abs() < 0.001);
    }

    #[test]
    fn test_dpu_stats_snapshot() {
        let stats = DpuStats::new();
        stats.bytes_transferred.store(1000, Ordering::Relaxed);
        stats.prefetch_hits.store(8, Ordering::Relaxed);
        stats.prefetch_misses.store(2, Ordering::Relaxed);
        let snap = stats.snapshot();
        assert_eq!(snap.bytes_transferred, 1000);
        assert!((snap.prefetch_hit_rate - 0.8).abs() < 1e-10);
    }

    // -- BackendCapabilities tests --

    #[test]
    fn test_host_only_capabilities() {
        let caps = BackendCapabilities::host_only();
        assert!(!caps.has_hw_compression);
        assert!(caps.hw_compression_algos.is_empty());
        assert!(!caps.has_bar1_direct);
        assert!(!caps.supports_prefetch);
    }

    #[test]
    fn test_bf2_capabilities() {
        let caps = BackendCapabilities::bf2();
        assert!(caps.has_hw_compression);
        assert_eq!(caps.hw_compression_algos, vec![CompressionAlgo::Deflate]);
        assert!(!caps.hw_compression_algos.contains(&CompressionAlgo::Lz4));
        assert_eq!(caps.max_bandwidth_gbps, 200);
    }

    #[test]
    fn test_bf3_capabilities() {
        let caps = BackendCapabilities::bf3();
        assert!(caps.hw_compression_algos.contains(&CompressionAlgo::Lz4));
        assert_eq!(caps.max_bandwidth_gbps, 400);
    }

    // -- BAR1 Region tests --

    #[test]
    fn test_bar1_region_contains() {
        let region = Bar1Region {
            gpu_id: 0,
            base_addr: 0x1000,
            size: 0x2000,
            rebar_enabled: true,
        };
        assert!(region.contains(0x1000, 0x100));
        assert!(region.contains(0x2FFF, 1));
        assert!(!region.contains(0x3000, 1)); // Just past end
        assert!(!region.contains(0x0FFF, 1)); // Just before start
        assert!(!region.contains(0x1000, 0x2001)); // Overflows end
    }

    // -- PageEntry tests --

    #[test]
    fn test_page_entry_new_local() {
        let entry = PageEntry::new_local(42);
        assert_eq!(entry.owner_node, 42);
        assert_eq!(entry.state, PageState::Local);
        assert!(!entry.is_pinned());
        assert!(!entry.is_dirty());
    }

    #[test]
    fn test_page_entry_flags() {
        let mut entry = PageEntry::new_local(1);
        entry.flags = page_flags::PINNED | page_flags::DIRTY;
        assert!(entry.is_pinned());
        assert!(entry.is_dirty());
        assert!(!entry.is_prefetched());
    }

    #[test]
    fn test_page_entry_record_access() {
        let mut entry = PageEntry::new_local(1);
        entry.record_access(100);
        assert_eq!(entry.last_access, 100);
        assert_eq!(entry.access_count, 1);
        entry.record_access(200);
        assert_eq!(entry.access_count, 2);
    }

    #[test]
    fn test_page_entry_access_count_saturates() {
        let mut entry = PageEntry::new_local(1);
        entry.access_count = u16::MAX;
        entry.record_access(1);
        assert_eq!(entry.access_count, u16::MAX); // Does not overflow
    }

    // -- GpuPageMap tests --

    #[test]
    fn test_gpu_page_map_new() {
        let map = GpuPageMap::new(0, 1, 100);
        assert_eq!(map.total_pages, 100);
        assert_eq!(map.entries.len(), 100);
        assert_eq!(map.count_in_state(PageState::Local), 100);
    }

    #[test]
    fn test_gpu_page_map_get_and_mutate() {
        let mut map = GpuPageMap::new(0, 1, 10);
        {
            let entry = map.get_mut(5).expect("page 5 exists");
            entry.state = PageState::Remote(2);
        }
        assert_eq!(map.get(5).unwrap().state, PageState::Remote(2));
        assert_eq!(map.count_in_state(PageState::Remote(2)), 1);
    }

    #[test]
    fn test_gpu_page_map_out_of_range() {
        let map = GpuPageMap::new(0, 1, 10);
        assert!(map.get(10).is_none());
        assert!(map.get(100).is_none());
    }

    // -- Compression decision tests --

    #[test]
    fn test_compression_decision_too_small() {
        let config = AdaptiveCompressConfig::default();
        let decision = decide_transfer_compression(100, CompressHint::Auto, &config, None);
        assert_eq!(decision, OffloadDecision::SendDirect);
    }

    #[test]
    fn test_compression_decision_never() {
        let config = AdaptiveCompressConfig::default();
        let decision = decide_transfer_compression(1_000_000, CompressHint::Never, &config, None);
        assert_eq!(decision, OffloadDecision::SendDirect);
    }

    #[test]
    fn test_compression_decision_always() {
        let config = AdaptiveCompressConfig::default();
        let decision = decide_transfer_compression(1_000_000, CompressHint::Always, &config, None);
        assert_eq!(
            decision,
            OffloadDecision::CompressThenSend(CompressionAlgo::Deflate)
        );
    }

    #[test]
    fn test_compression_decision_auto_no_stats() {
        let config = AdaptiveCompressConfig::default();
        let decision = decide_transfer_compression(1_000_000, CompressHint::Auto, &config, None);
        // No stats yet, so it tries compression to establish baseline
        assert_eq!(
            decision,
            OffloadDecision::CompressThenSend(CompressionAlgo::Deflate)
        );
    }

    #[test]
    fn test_compression_decision_auto_good_ratio() {
        let config = AdaptiveCompressConfig::default();
        let stats = CompressionStats {
            last_ratio: 0.5, // 50% compression -- very good
            ..CompressionStats::new(CompressionAlgo::Deflate)
        };
        let decision =
            decide_transfer_compression(1_000_000, CompressHint::Auto, &config, Some(&stats));
        assert_eq!(
            decision,
            OffloadDecision::CompressThenSend(CompressionAlgo::Deflate)
        );
    }

    #[test]
    fn test_compression_decision_auto_poor_ratio() {
        let config = AdaptiveCompressConfig::default();
        let stats = CompressionStats {
            last_ratio: 0.95, // Only 5% savings -- not worth it
            ..CompressionStats::new(CompressionAlgo::Deflate)
        };
        let decision =
            decide_transfer_compression(1_000_000, CompressHint::Auto, &config, Some(&stats));
        assert_eq!(decision, OffloadDecision::SendDirect);
    }

    #[test]
    fn test_compression_decision_bf3_uses_lz4() {
        let config = AdaptiveCompressConfig::for_bf3();
        let decision =
            decide_transfer_compression(1_000_000, CompressHint::Always, &config, None);
        assert_eq!(
            decision,
            OffloadDecision::CompressThenSend(CompressionAlgo::Lz4)
        );
    }

    // -- Core allocation tests --

    #[test]
    fn test_core_allocation_bf2() {
        let gen = BfGeneration::bf2();
        let alloc = allocate_cores(&gen);
        assert_eq!(alloc.total_cores(), 8);
        assert_eq!(alloc.assignments[0], (0, CoreRole::System));
        assert_eq!(alloc.cores_for_role(CoreRole::Transport).len(), 2);
        assert_eq!(alloc.cores_for_role(CoreRole::Compress).len(), 1);
        assert_eq!(alloc.cores_for_role(CoreRole::PageTable).len(), 1);
        assert_eq!(alloc.cores_for_role(CoreRole::Prefetch).len(), 1);
        assert_eq!(alloc.cores_for_role(CoreRole::Router).len(), 1);
        assert_eq!(alloc.cores_for_role(CoreRole::Overflow).len(), 1);
    }

    #[test]
    fn test_core_allocation_bf3() {
        let gen = BfGeneration::bf3();
        let alloc = allocate_cores(&gen);
        assert_eq!(alloc.total_cores(), 16);
        assert_eq!(alloc.cores_for_role(CoreRole::System).len(), 1);
        assert_eq!(alloc.cores_for_role(CoreRole::Transport).len(), 4);
        assert_eq!(alloc.cores_for_role(CoreRole::PageTable).len(), 2);
        assert_eq!(alloc.cores_for_role(CoreRole::Prefetch).len(), 2);
        assert_eq!(alloc.cores_for_role(CoreRole::Router).len(), 2);
        assert!(alloc.cores_for_role(CoreRole::Overflow).len() >= 4);
    }

    // -- DPU probe and backend configuration tests --

    #[test]
    fn test_probe_dpu_returns_no_dpu() {
        let result = probe_dpu();
        matches!(result, DpuProbeResult::NoDpu);
    }

    #[test]
    fn test_configure_backend_no_dpu() {
        let probe = DpuProbeResult::NoDpu;
        let backend = configure_backend(&probe, 2);
        match backend {
            DetectedBackend::HostOnly(caps) => {
                assert!(!caps.has_hw_compression);
            }
            _ => panic!("Expected HostOnly backend"),
        }
    }

    #[test]
    fn test_configure_backend_wrong_mode() {
        let probe = DpuProbeResult::WrongMode {
            pci_addr: "0000:03:00.0".to_string(),
        };
        let backend = configure_backend(&probe, 2);
        assert!(matches!(backend, DetectedBackend::HostOnly(_)));
    }

    #[test]
    fn test_configure_backend_service_not_running() {
        let probe = DpuProbeResult::ServiceNotRunning {
            pci_addr: "0000:03:00.0".to_string(),
            generation: BfGeneration::bf2(),
        };
        let backend = configure_backend(&probe, 2);
        assert!(matches!(backend, DetectedBackend::HostOnly(_)));
    }

    #[test]
    fn test_configure_backend_ready_bf2() {
        let probe = DpuProbeResult::Ready {
            pci_addr: "0000:03:00.0".to_string(),
            generation: BfGeneration::bf2(),
            capabilities: BackendCapabilities::bf2(),
        };
        let backend = configure_backend(&probe, 2);
        match backend {
            DetectedBackend::DpuOffloaded {
                config,
                capabilities,
                ..
            } => {
                assert!(config.compression_offload);
                assert!(config.prefetch_offload); // 8 cores >= 6
                assert_eq!(config.page_table_max_gpus, 2);
                assert!(capabilities.has_hw_compression);
            }
            _ => panic!("Expected DpuOffloaded backend"),
        }
    }

    #[test]
    fn test_configure_backend_ready_bf3() {
        let probe = DpuProbeResult::Ready {
            pci_addr: "0000:03:00.0".to_string(),
            generation: BfGeneration::bf3(),
            capabilities: BackendCapabilities::bf3(),
        };
        let backend = configure_backend(&probe, 4);
        match backend {
            DetectedBackend::DpuOffloaded {
                config,
                capabilities,
                ..
            } => {
                assert_eq!(config.page_table_max_gpus, 4);
                assert!(capabilities.hw_compression_algos.contains(&CompressionAlgo::Lz4));
            }
            _ => panic!("Expected DpuOffloaded backend"),
        }
    }

    // -- DpuService tests --

    #[test]
    fn test_dpu_service_new() {
        let service = DpuService::new(DpuConfig::default());
        assert_eq!(service.core_allocation.total_cores(), 8);
        assert!(service.page_maps.is_empty());
    }

    #[test]
    fn test_dpu_service_register_gpu() {
        let mut service = DpuService::new(DpuConfig::default());
        service.register_gpu(0, 1, 1000);
        assert_eq!(service.page_maps.len(), 1);
        assert_eq!(service.page_maps[0].total_pages, 1000);
    }

    #[test]
    fn test_dpu_service_handle_alloc_notify() {
        let mut service = DpuService::new(DpuConfig::default());
        service.register_gpu(0, 1, 100);

        // Mark page 5 as invalid first
        service.page_maps[0].get_mut(5).unwrap().state = PageState::Invalid;

        let msg = HostToDpuMsg::AllocNotify {
            gpu_id: 0,
            addr: 5 * 4096,
            size: 4096,
        };
        let resp = service.handle_message(msg);
        assert!(resp.is_none()); // No response for alloc notify
        assert_eq!(service.page_maps[0].get(5).unwrap().state, PageState::Local);
    }

    #[test]
    fn test_dpu_service_handle_free_notify() {
        let mut service = DpuService::new(DpuConfig::default());
        service.register_gpu(0, 1, 100);

        let msg = HostToDpuMsg::FreeNotify {
            gpu_id: 0,
            addr: 10 * 4096,
            size: 4096,
        };
        let resp = service.handle_message(msg);
        assert!(resp.is_none());
        assert_eq!(
            service.page_maps[0].get(10).unwrap().state,
            PageState::Invalid
        );
    }

    #[test]
    fn test_dpu_service_handle_shutdown() {
        let service = DpuService::new(DpuConfig::default());
        let msg = HostToDpuMsg::Shutdown;
        // Need mutable for handle_message
        let mut service = service;
        let resp = service.handle_message(msg);
        assert!(matches!(resp, Some(DpuToHostMsg::Stats { .. })));
    }

    #[test]
    fn test_dpu_service_handle_transfer() {
        let mut service = DpuService::new(DpuConfig::default());
        let msg = HostToDpuMsg::TransferRequest {
            id: 42,
            src: MemLocation::GpuVram {
                gpu_id: 0,
                addr: 0x1000,
            },
            dst: MemLocation::RemoteNode {
                node_id: 2,
                gpu_id: 0,
                addr: 0x2000,
            },
            size: 65536,
            compress: CompressHint::Never,
        };
        let resp = service.handle_message(msg);
        match resp {
            Some(DpuToHostMsg::TransferComplete {
                id,
                bytes_transferred,
                ..
            }) => {
                assert_eq!(id, 42);
                assert_eq!(bytes_transferred, 65536);
            }
            _ => panic!("Expected TransferComplete"),
        }
        assert_eq!(
            service.stats.bytes_transferred.load(Ordering::Relaxed),
            65536
        );
    }

    #[test]
    fn test_dpu_service_handle_connect_disconnect() {
        let mut service = DpuService::new(DpuConfig::default());

        let resp = service
            .handle_message(HostToDpuMsg::ConnectPeer {
                peer: "10.0.0.2:9000".to_string(),
            });
        assert!(matches!(resp, Some(DpuToHostMsg::PeerConnected { .. })));
        assert_eq!(service.stats.connections_active.load(Ordering::Relaxed), 1);

        let resp = service.handle_message(HostToDpuMsg::DisconnectPeer { conn_id: 1 });
        assert!(matches!(resp, Some(DpuToHostMsg::PeerDisconnected { .. })));
        assert_eq!(service.stats.connections_active.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_dpu_service_bf3_uses_lz4() {
        let config = DpuConfig {
            bf_generation: BfGeneration::bf3(),
            ..DpuConfig::default()
        };
        let service = DpuService::new(config);
        assert_eq!(
            service.compress_config.preferred_algo,
            CompressionAlgo::Lz4
        );
    }

    // -- CompressionStats tests --

    #[test]
    fn test_compression_stats_overall_ratio_empty() {
        let stats = CompressionStats::new(CompressionAlgo::Deflate);
        assert_eq!(stats.overall_ratio(), 1.0);
    }

    #[test]
    fn test_compression_stats_overall_ratio() {
        let mut stats = CompressionStats::new(CompressionAlgo::Lz4);
        stats.total_bytes_in = 1000;
        stats.total_bytes_out = 600;
        assert!((stats.overall_ratio() - 0.6).abs() < 1e-10);
    }

    // -- MemLocation serialization --

    #[test]
    fn test_mem_location_variants() {
        let locations = vec![
            MemLocation::HostPinned { addr: 0x1000 },
            MemLocation::GpuVram {
                gpu_id: 0,
                addr: 0x2000,
            },
            MemLocation::RemoteNode {
                node_id: 3,
                gpu_id: 1,
                addr: 0x3000,
            },
        ];
        // Verify bincode roundtrip
        for loc in &locations {
            let encoded = bincode::serialize(loc).expect("serialize");
            let decoded: MemLocation = bincode::deserialize(&encoded).expect("deserialize");
            assert_eq!(*loc, decoded);
        }
    }

    // -- Protocol message serialization --

    #[test]
    fn test_host_to_dpu_msg_serialization() {
        let msg = HostToDpuMsg::TransferRequest {
            id: 1,
            src: MemLocation::GpuVram {
                gpu_id: 0,
                addr: 0x1000,
            },
            dst: MemLocation::RemoteNode {
                node_id: 2,
                gpu_id: 0,
                addr: 0x2000,
            },
            size: 4096,
            compress: CompressHint::Auto,
        };
        let encoded = bincode::serialize(&msg).expect("serialize");
        let decoded: HostToDpuMsg = bincode::deserialize(&encoded).expect("deserialize");
        match decoded {
            HostToDpuMsg::TransferRequest { id, size, .. } => {
                assert_eq!(id, 1);
                assert_eq!(size, 4096);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_dpu_to_host_msg_serialization() {
        let msg = DpuToHostMsg::TransferComplete {
            id: 99,
            bytes_transferred: 65536,
            elapsed_us: 150,
        };
        let encoded = bincode::serialize(&msg).expect("serialize");
        let decoded: DpuToHostMsg = bincode::deserialize(&encoded).expect("deserialize");
        match decoded {
            DpuToHostMsg::TransferComplete { id, elapsed_us, .. } => {
                assert_eq!(id, 99);
                assert_eq!(elapsed_us, 150);
            }
            _ => panic!("Wrong variant"),
        }
    }
}
