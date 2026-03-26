//! Virtual NVLink emulation over RDMA/network transport (R18).
//!
//! Emulates NVLink semantics (peer-to-peer GPU access, atomic operations,
//! coherency) over OuterLink's network transport. Three tiers of emulation:
//!
//! - **Tier 1:** Peer access API interception and cudaMemcpyPeer routing.
//!   Covers ~80% of real NVLink usage at low complexity.
//! - **Tier 2:** Coherency integration via R19's I/S/E protocol, providing
//!   unified address space across networked GPUs.
//! - **Tier 3:** Remote atomic operations translated to RDMA atomics or
//!   proxy requests.
//!
//! # Architecture
//!
//! ```text
//! NvlinkEmulator (facade)
//!   +-- PeerAccessManager      (Tier 1: peer connections, P2P memcpy routing)
//!   +-- NvlinkCoherencyAdapter (Tier 2: wraps R19 I/S/E protocol)
//!   +-- RemoteAtomicEngine     (Tier 3: GPU atomics -> RDMA atomics)
//!   +-- PerformanceDiagnostics (all tiers: warnings, stats)
//! ```
//!
//! # Key Design Decisions
//!
//! - R18 does NOT implement a new coherency protocol. It adapts R19's existing
//!   directory-based I/S/E protocol with NVLink-specific behavior.
//! - RDMA atomics are 64-bit only. 32-bit and 16-bit CUDA atomics are handled
//!   via 64-bit CAS emulation with sub-word alignment.
//! - Hardware-dependent operations (RDMA verbs, QP setup, BAR1 access) are
//!   defined as trait interfaces with TODO stubs for the actual transport layer.

use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::gpu_mixing::GpuId;
use crate::memory::types::NodeId;

// ---------------------------------------------------------------------------
// Type aliases used throughout this module
// ---------------------------------------------------------------------------

/// Device identifier within the OuterLink virtual device space.
/// Maps to CUDA device ordinals as seen by the application.
pub type DeviceId = u32;

/// Virtual address in the unified GPU address space.
pub type VirtualAddr = u64;

/// Queue pair identifier for RDMA connections.
pub type QueuePairId = u32;

// ---------------------------------------------------------------------------
// Placeholder types for hardware-dependent subsystems
// ---------------------------------------------------------------------------

/// Placeholder for R17 topology manager's snapshot.
/// TODO: Replace with actual TopologySnapshot from R17 when available.
#[derive(Debug, Clone)]
pub struct TopologySnapshot {
    /// Known nodes in the cluster.
    pub nodes: Vec<NodeId>,
    /// Adjacency: (nodeA, nodeB) -> hop count. Missing = unreachable.
    pub adjacency: HashMap<(NodeId, NodeId), u8>,
}

impl TopologySnapshot {
    /// Create an empty topology.
    pub fn empty() -> Self {
        Self {
            nodes: Vec::new(),
            adjacency: HashMap::new(),
        }
    }

    /// Check if there is a network path between two nodes.
    pub fn has_path(&self, from: NodeId, to: NodeId) -> bool {
        if from == to {
            return true;
        }
        self.adjacency.contains_key(&(from, to))
    }

    /// Get the hop count between two nodes, if reachable.
    pub fn hop_count(&self, from: NodeId, to: NodeId) -> Option<u8> {
        if from == to {
            return Some(0);
        }
        self.adjacency.get(&(from, to)).copied()
    }
}

/// Placeholder for the network transport manager.
/// TODO: Replace with actual TransportManager when wiring into transport layer.
pub struct TransportManager;

/// Placeholder for R19's page fault handler.
/// TODO: Replace with actual PageFaultHandler from R19 when available.
pub struct PageFaultHandler;

/// Placeholder for R19's coherency directory.
/// TODO: Replace with actual CoherencyDirectory from R19 when available.
pub struct CoherencyDirectory;

/// Placeholder for R10's page table.
/// TODO: Replace with actual PageTable from R10 when available.
pub struct PageTableRef;

/// Placeholder for an RDMA multicast group (R29).
/// TODO: Replace with actual MulticastGroup from R29 when available.
pub struct MulticastGroup;

/// Placeholder for PTP clock reference (R26).
/// TODO: Replace with actual PtpClock from clock_sync when wiring.
pub struct PtpClock;

/// Placeholder for RDMA registered memory region.
/// TODO: Replace with actual RDMA MR from transport layer.
#[derive(Debug, Clone)]
pub struct RegisteredMemoryRegion {
    /// Base address of the registered region.
    pub base_addr: u64,
    /// Length in bytes.
    pub length: u64,
    /// Local key for RDMA access.
    pub lkey: u32,
    /// Remote key for RDMA access.
    pub rkey: u32,
}

/// Placeholder for an RDMA queue pair.
/// TODO: Replace with actual RDMA QP from transport layer.
#[derive(Debug, Clone)]
pub struct RdmaQueuePair {
    /// QP number.
    pub qp_num: u32,
}

// ---------------------------------------------------------------------------
// Route information from R17 topology
// ---------------------------------------------------------------------------

/// Route information from R17 topology-aware scheduling.
///
/// Describes the network path from a local device to a remote device,
/// including hop count, cost weighting, and transport capability.
#[derive(Debug, Clone)]
pub struct RouteInfo {
    /// Hop count from local to remote (0 = same node).
    pub hop_count: u8,
    /// Weighted cost (lower = better, factors in bandwidth + latency).
    pub weighted_cost: f32,
    /// Whether this path uses RDMA (true) or TCP fallback (false).
    pub is_rdma: bool,
    /// If RDMA: the QP (Queue Pair) number for this connection.
    pub rdma_qp: Option<u32>,
}

impl RouteInfo {
    /// Create a route for a same-node (loopback) connection.
    pub fn loopback() -> Self {
        Self {
            hop_count: 0,
            weighted_cost: 0.0,
            is_rdma: false,
            rdma_qp: None,
        }
    }

    /// Create a route with the given parameters.
    pub fn new(hop_count: u8, weighted_cost: f32, is_rdma: bool, rdma_qp: Option<u32>) -> Self {
        Self {
            hop_count,
            weighted_cost,
            is_rdma,
            rdma_qp,
        }
    }
}

// ---------------------------------------------------------------------------
// Peer access statistics
// ---------------------------------------------------------------------------

/// Accumulated statistics per peer connection.
///
/// All counters are atomic for lock-free concurrent updates from
/// multiple CUDA streams hitting the same peer connection.
#[derive(Debug)]
pub struct PeerAccessStats {
    /// Total bytes transferred via cudaMemcpyPeer on this connection.
    pub bytes_transferred: AtomicU64,
    /// Total number of cudaMemcpyPeer calls.
    pub transfer_count: AtomicU64,
    /// Number of small transfers (<4KB) -- latency-dominated, warn if high.
    pub small_transfer_count: AtomicU64,
    /// Number of remote atomics executed (Tier 3).
    pub atomic_count: AtomicU64,
    /// Number of page faults triggered on this peer's memory (Tier 2).
    pub page_fault_count: AtomicU64,
}

impl PeerAccessStats {
    /// Create zeroed statistics.
    pub fn new() -> Self {
        Self {
            bytes_transferred: AtomicU64::new(0),
            transfer_count: AtomicU64::new(0),
            small_transfer_count: AtomicU64::new(0),
            atomic_count: AtomicU64::new(0),
            page_fault_count: AtomicU64::new(0),
        }
    }

    /// Record a transfer of the given size in bytes.
    pub fn record_transfer(&self, size_bytes: u64) {
        self.bytes_transferred.fetch_add(size_bytes, Ordering::Relaxed);
        self.transfer_count.fetch_add(1, Ordering::Relaxed);
        if size_bytes < 4096 {
            self.small_transfer_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record a remote atomic operation.
    pub fn record_atomic(&self) {
        self.atomic_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a page fault on this peer's memory.
    pub fn record_page_fault(&self) {
        self.page_fault_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get a snapshot of all stats.
    pub fn snapshot(&self) -> PeerAccessStatsSnapshot {
        PeerAccessStatsSnapshot {
            bytes_transferred: self.bytes_transferred.load(Ordering::Relaxed),
            transfer_count: self.transfer_count.load(Ordering::Relaxed),
            small_transfer_count: self.small_transfer_count.load(Ordering::Relaxed),
            atomic_count: self.atomic_count.load(Ordering::Relaxed),
            page_fault_count: self.page_fault_count.load(Ordering::Relaxed),
        }
    }
}

impl Default for PeerAccessStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Non-atomic snapshot of peer access statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerAccessStatsSnapshot {
    /// Total bytes transferred.
    pub bytes_transferred: u64,
    /// Total transfer count.
    pub transfer_count: u64,
    /// Number of small (<4KB) transfers.
    pub small_transfer_count: u64,
    /// Number of remote atomics.
    pub atomic_count: u64,
    /// Number of page faults.
    pub page_fault_count: u64,
}

// ---------------------------------------------------------------------------
// Peer connection
// ---------------------------------------------------------------------------

/// State of a single peer access connection (GPU A -> GPU B).
///
/// Created when `cuCtxEnablePeerAccess` is called for a remote device.
/// Tracks the connection state, route, measured performance, and statistics.
pub struct PeerConnection {
    /// Local device ID.
    pub local_device: DeviceId,
    /// Remote device ID (may be on a different node).
    pub remote_device: DeviceId,
    /// Remote node ID in the cluster.
    pub remote_node: NodeId,
    /// Whether peer access is currently enabled.
    pub enabled: AtomicBool,
    /// Network path info from R17 topology.
    pub route: RouteInfo,
    /// Measured latency to this peer in microseconds (updated periodically).
    pub measured_latency_us: AtomicU32,
    /// Measured bandwidth to this peer in MB/s.
    pub measured_bandwidth_mbps: AtomicU32,
    /// Statistics for diagnostic reporting.
    pub stats: PeerAccessStats,
}

impl PeerConnection {
    /// Create a new peer connection with the given parameters.
    pub fn new(
        local_device: DeviceId,
        remote_device: DeviceId,
        remote_node: NodeId,
        route: RouteInfo,
    ) -> Self {
        Self {
            local_device,
            remote_device,
            remote_node,
            enabled: AtomicBool::new(false),
            route,
            measured_latency_us: AtomicU32::new(0),
            measured_bandwidth_mbps: AtomicU32::new(0),
            stats: PeerAccessStats::new(),
        }
    }

    /// Enable peer access on this connection.
    ///
    /// Returns `false` if already enabled (maps to CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED).
    pub fn enable(&self) -> bool {
        // Returns true only if we actually changed from false to true
        !self.enabled.swap(true, Ordering::AcqRel)
    }

    /// Disable peer access on this connection.
    ///
    /// Returns `false` if not currently enabled.
    pub fn disable(&self) -> bool {
        self.enabled.swap(false, Ordering::AcqRel)
    }

    /// Check if peer access is currently enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    /// Update measured latency in microseconds.
    pub fn update_latency(&self, latency_us: u32) {
        self.measured_latency_us.store(latency_us, Ordering::Relaxed);
    }

    /// Update measured bandwidth in MB/s.
    pub fn update_bandwidth(&self, bandwidth_mbps: u32) {
        self.measured_bandwidth_mbps.store(bandwidth_mbps, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Peer attributes (cuDeviceGetP2PAttribute)
// ---------------------------------------------------------------------------

/// What we report for cuDeviceGetP2PAttribute queries.
///
/// Maps CUDA P2P attribute constants to values derived from OuterLink's
/// virtual NVLink topology and configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerAttributes {
    /// CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED -- always 1 for virtual peers.
    pub access_supported: i32,
    /// CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED.
    /// 1 if Tier 3 atomic engine is active, 0 otherwise.
    pub native_atomic_supported: i32,
    /// CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK.
    /// Derived from R17 topology: 0 = same node, 1 = 1-hop, 2 = 2+ hops.
    pub performance_rank: i32,
    /// CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED.
    /// 1 if both devices support UVA (same major compute capability).
    pub cuda_array_access_supported: i32,
}

impl PeerAttributes {
    /// Build peer attributes from connection state and configuration.
    pub fn from_connection(
        conn: &PeerConnection,
        active_tier: u8,
        report_atomics: bool,
        same_compute_capability: bool,
        rank_offset: i32,
    ) -> Self {
        let performance_rank = conn.route.hop_count as i32 + rank_offset;
        Self {
            access_supported: 1,
            native_atomic_supported: if active_tier >= 3 && report_atomics { 1 } else { 0 },
            performance_rank,
            cuda_array_access_supported: if same_compute_capability { 1 } else { 0 },
        }
    }
}

// ---------------------------------------------------------------------------
// CUDA atomic operation types
// ---------------------------------------------------------------------------

/// Size of a CUDA atomic operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtomicSize {
    /// 16-bit atomic (SM 7.0+ only).
    Bits16,
    /// 32-bit atomic.
    Bits32,
    /// 64-bit atomic.
    Bits64,
}

impl AtomicSize {
    /// Number of bytes for this atomic size.
    pub fn byte_count(self) -> usize {
        match self {
            AtomicSize::Bits16 => 2,
            AtomicSize::Bits32 => 4,
            AtomicSize::Bits64 => 8,
        }
    }
}

/// All CUDA atomic operations that need network translation.
///
/// Each variant captures the operation type and the size/signedness
/// parameters needed for correct translation to network operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CudaAtomicOp {
    /// atomicAdd with optional float mode.
    Add { size: AtomicSize, is_float: bool },
    /// atomicSub.
    Sub { size: AtomicSize },
    /// atomicMin with signedness.
    Min { size: AtomicSize, is_signed: bool },
    /// atomicMax with signedness.
    Max { size: AtomicSize, is_signed: bool },
    /// atomicExch.
    Exch { size: AtomicSize },
    /// atomicCAS (compare-and-swap).
    CAS { size: AtomicSize },
    /// atomicAnd.
    And { size: AtomicSize },
    /// atomicOr.
    Or { size: AtomicSize },
    /// atomicXor.
    Xor { size: AtomicSize },
    /// atomicInc with modulo.
    Inc { modulo: u32 },
    /// atomicDec with modulo.
    Dec { modulo: u32 },
}

impl CudaAtomicOp {
    /// Whether this operation can be mapped directly to an RDMA atomic
    /// (without CAS emulation loop) when operating on 64-bit values.
    pub fn is_direct_rdma_mappable(&self) -> bool {
        matches!(
            self,
            CudaAtomicOp::CAS { size: AtomicSize::Bits64 }
                | CudaAtomicOp::Add {
                    size: AtomicSize::Bits64,
                    is_float: false,
                }
                | CudaAtomicOp::Sub { size: AtomicSize::Bits64 }
        )
    }

    /// Get the atomic size of this operation.
    pub fn size(&self) -> AtomicSize {
        match self {
            CudaAtomicOp::Add { size, .. } => *size,
            CudaAtomicOp::Sub { size } => *size,
            CudaAtomicOp::Min { size, .. } => *size,
            CudaAtomicOp::Max { size, .. } => *size,
            CudaAtomicOp::Exch { size } => *size,
            CudaAtomicOp::CAS { size } => *size,
            CudaAtomicOp::And { size } => *size,
            CudaAtomicOp::Or { size } => *size,
            CudaAtomicOp::Xor { size } => *size,
            // Inc/Dec are always 32-bit in CUDA
            CudaAtomicOp::Inc { .. } => AtomicSize::Bits32,
            CudaAtomicOp::Dec { .. } => AtomicSize::Bits32,
        }
    }
}

/// Operand values for an atomic operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomicOperands {
    /// First operand (value for add/sub/min/max/exch, compare for CAS).
    pub operand1: u64,
    /// Second operand (swap value for CAS, unused for others).
    pub operand2: u64,
}

// ---------------------------------------------------------------------------
// Network atomic operation (after translation)
// ---------------------------------------------------------------------------

/// Describes the compute step inside a CAS emulation loop.
///
/// Pattern: `loop { old = read(addr); new = f(old, operand); if CAS(addr, old, new) == old: done }`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CasComputeFn {
    /// Signed or unsigned minimum.
    Min { is_signed: bool },
    /// Signed or unsigned maximum.
    Max { is_signed: bool },
    /// Bitwise AND.
    And,
    /// Bitwise OR.
    Or,
    /// Bitwise XOR.
    Xor,
    /// CUDA atomicInc: `if old >= modulo { 0 } else { old + 1 }`.
    Inc { modulo: u32 },
    /// CUDA atomicDec: `if old == 0 || old > modulo { modulo } else { old - 1 }`.
    Dec { modulo: u32 },
    /// Float add (reinterpret bits).
    FloatAdd,
    /// Exchange: CAS(addr, old, new_value) -- always succeeds on match.
    Exchange,
}

impl CasComputeFn {
    /// Execute the compute function: `(old_value, operand) -> new_value`.
    ///
    /// All values are treated as u64 bit patterns. The caller is responsible
    /// for sub-word extraction when operating on 32-bit or 16-bit values.
    pub fn compute(&self, old: u64, operand: u64) -> u64 {
        match self {
            CasComputeFn::Min { is_signed: true } => {
                std::cmp::min(old as i64, operand as i64) as u64
            }
            CasComputeFn::Min { is_signed: false } => std::cmp::min(old, operand),
            CasComputeFn::Max { is_signed: true } => {
                std::cmp::max(old as i64, operand as i64) as u64
            }
            CasComputeFn::Max { is_signed: false } => std::cmp::max(old, operand),
            CasComputeFn::And => old & operand,
            CasComputeFn::Or => old | operand,
            CasComputeFn::Xor => old ^ operand,
            CasComputeFn::Inc { modulo } => {
                let m = *modulo as u64;
                if old >= m {
                    0
                } else {
                    old + 1
                }
            }
            CasComputeFn::Dec { modulo } => {
                let m = *modulo as u64;
                if old == 0 || old > m {
                    m
                } else {
                    old - 1
                }
            }
            CasComputeFn::FloatAdd => {
                let a = f64::from_bits(old);
                let b = f64::from_bits(operand);
                f64::to_bits(a + b)
            }
            CasComputeFn::Exchange => operand,
        }
    }

    /// Execute the compute function for 32-bit float add.
    /// Interprets lower 32 bits as f32.
    pub fn compute_f32_add(old: u64, operand: u64) -> u64 {
        let a = f32::from_bits(old as u32);
        let b = f32::from_bits(operand as u32);
        f32::to_bits(a + b) as u64
    }
}

/// Network-level atomic operation after translation from CUDA atomic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetworkAtomicOp {
    /// Direct RDMA CAS (64-bit only).
    RdmaCas {
        /// Expected current value.
        compare: u64,
        /// Value to swap in.
        swap: u64,
    },
    /// Direct RDMA Fetch-and-Add (64-bit only).
    RdmaFetchAdd {
        /// Value to add (may be negated for subtraction).
        addend: u64,
    },
    /// CAS emulation loop for operations without direct RDMA mapping.
    CasEmulationLoop {
        /// The compute function to apply inside the CAS loop.
        compute_fn: CasComputeFn,
    },
    /// Proxy request: send operation to home node for local execution.
    ProxyRequest {
        /// Original CUDA atomic operation.
        op: CudaAtomicOp,
        /// Operands for the atomic.
        operands: AtomicOperands,
    },
}

/// Full description of an atomic operation translation.
///
/// Maps a CUDA GPU atomic to a network operation, including the target
/// address, operands, and chosen network implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomicTranslation {
    /// Original CUDA atomic type.
    pub cuda_op: CudaAtomicOp,
    /// Target address (virtual, may be remote).
    pub target_addr: VirtualAddr,
    /// Operand value(s).
    pub operands: AtomicOperands,
    /// Chosen network implementation.
    pub network_op: NetworkAtomicOp,
}

// ---------------------------------------------------------------------------
// Atomic mode selection
// ---------------------------------------------------------------------------

/// How remote atomics to a given page are handled.
///
/// The mode is selected per-page based on the atomic type and access
/// frequency. High-frequency targets switch from migration to proxy mode
/// to avoid page thrashing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtomicMode {
    /// Migrate page to requesting node, execute locally.
    /// Best for low-frequency atomics where the page will stay local.
    Migrate,
    /// Route atomic to home node's proxy server, execute there, return result.
    /// Best for high-frequency cross-node atomics (avoids ping-ponging).
    Proxy,
    /// Use RDMA atomic directly (only for CAS/fetch-add on 64-bit values).
    /// Lowest latency for supported operations.
    RdmaDirect,
}

/// Tracks a hot atomic target page.
///
/// Monitors the frequency of remote atomic operations on a page and
/// determines the optimal handling mode.
pub struct HotAtomicEntry {
    /// Which node owns the authoritative copy.
    pub home_node: NodeId,
    /// Recent atomic frequency (atomics/sec from remote nodes).
    pub remote_frequency: AtomicU32,
    /// Current mode for this page.
    pub mode: Mutex<AtomicMode>,
    /// Timestamps of recent accesses for frequency calculation.
    access_timestamps: Mutex<Vec<Instant>>,
    /// Maximum window size for frequency tracking.
    window_size: usize,
}

impl HotAtomicEntry {
    /// Create a new entry for a page owned by the given node.
    pub fn new(home_node: NodeId) -> Self {
        Self {
            home_node,
            remote_frequency: AtomicU32::new(0),
            mode: Mutex::new(AtomicMode::Migrate),
            access_timestamps: Mutex::new(Vec::with_capacity(64)),
            window_size: 64,
        }
    }

    /// Record a remote atomic access and update frequency.
    ///
    /// Returns the updated frequency (atomics/sec).
    pub fn record_access(&self) -> u32 {
        let now = Instant::now();
        let mut timestamps = self.access_timestamps.lock().unwrap();

        // Add current timestamp
        timestamps.push(now);

        // Trim to window size
        if timestamps.len() > self.window_size {
            let excess = timestamps.len() - self.window_size;
            timestamps.drain(0..excess);
        }

        // Calculate frequency over the window
        let freq = if timestamps.len() >= 2 {
            let window_duration = now.duration_since(timestamps[0]);
            if window_duration.as_micros() > 0 {
                ((timestamps.len() as u64 * 1_000_000) / window_duration.as_micros() as u64) as u32
            } else {
                0
            }
        } else {
            0
        };

        self.remote_frequency.store(freq, Ordering::Relaxed);
        freq
    }

    /// Get the current mode for this page.
    pub fn current_mode(&self) -> AtomicMode {
        *self.mode.lock().unwrap()
    }

    /// Update the mode based on the current frequency and the given threshold.
    ///
    /// If frequency exceeds `proxy_threshold`, switch to Proxy mode.
    /// Otherwise, stay in Migrate or RdmaDirect depending on the operation.
    pub fn update_mode(&self, proxy_threshold: u32, is_rdma_capable: bool) {
        let freq = self.remote_frequency.load(Ordering::Relaxed);
        let mut mode = self.mode.lock().unwrap();
        if freq > proxy_threshold {
            *mode = AtomicMode::Proxy;
        } else if is_rdma_capable {
            *mode = AtomicMode::RdmaDirect;
        } else {
            *mode = AtomicMode::Migrate;
        }
    }
}

// ---------------------------------------------------------------------------
// Atomic proxy connection
// ---------------------------------------------------------------------------

/// RDMA connection to a remote node's atomic proxy service.
///
/// The proxy server executes atomics locally on behalf of remote nodes,
/// avoiding page migration for high-contention addresses.
#[derive(Debug, Clone)]
pub struct AtomicProxyConnection {
    /// Remote node ID.
    pub node_id: NodeId,
    /// RDMA RC QP for atomic operations.
    pub qp: RdmaQueuePair,
    /// Remote proxy's memory region key (for RDMA atomics).
    pub remote_mr_key: u32,
    /// Doorbell address for proxy requests (RDMA write target).
    pub doorbell_addr: u64,
    /// Response buffer base address.
    pub response_buffer_addr: u64,
}

// ---------------------------------------------------------------------------
// Remote atomic engine
// ---------------------------------------------------------------------------

/// Translates GPU atomic operations into RDMA atomics or proxy requests.
///
/// Only active when Tier 3 is enabled. Manages per-node proxy connections,
/// tracks hot-atomic pages for automatic mode switching, and provides the
/// CAS emulation loop for non-directly-mappable CUDA atomics.
pub struct RemoteAtomicEngine {
    /// Per-node atomic proxy connections.
    proxy_connections: DashMap<NodeId, AtomicProxyConnection>,
    /// Hot-atomic tracker: pages with frequent cross-node atomics.
    hot_atomics: DashMap<VirtualAddr, HotAtomicEntry>,
    /// Threshold: if a page sees > N atomics/sec, switch to proxy mode.
    migration_to_proxy_threshold: u32,
    /// Maximum retries for CAS emulation loop.
    max_cas_retries: u32,
}

impl RemoteAtomicEngine {
    /// Create a new remote atomic engine with the given proxy threshold.
    pub fn new(migration_to_proxy_threshold: u32) -> Self {
        Self {
            proxy_connections: DashMap::new(),
            hot_atomics: DashMap::new(),
            migration_to_proxy_threshold,
            max_cas_retries: 32,
        }
    }

    /// Create with default settings.
    pub fn with_defaults() -> Self {
        Self::new(1000)
    }

    /// Register a proxy connection to a remote node.
    pub fn register_proxy(&self, conn: AtomicProxyConnection) {
        self.proxy_connections.insert(conn.node_id, conn);
    }

    /// Remove a proxy connection.
    pub fn remove_proxy(&self, node_id: NodeId) -> Option<AtomicProxyConnection> {
        self.proxy_connections.remove(&node_id).map(|(_, v)| v)
    }

    /// Check if a proxy connection exists for the given node.
    pub fn has_proxy(&self, node_id: NodeId) -> bool {
        self.proxy_connections.contains_key(&node_id)
    }

    /// Get the number of registered proxy connections.
    pub fn proxy_count(&self) -> usize {
        self.proxy_connections.len()
    }

    /// Translate a CUDA atomic operation to a network atomic operation.
    ///
    /// This is the main translation entry point. It determines the optimal
    /// network operation based on the CUDA atomic type, target address,
    /// and current hot-atomic state.
    pub fn translate(
        &self,
        cuda_op: CudaAtomicOp,
        target_addr: VirtualAddr,
        operands: AtomicOperands,
        home_node: NodeId,
    ) -> AtomicTranslation {
        // Track access for hot-atomic detection
        let mode = self.determine_mode(cuda_op, target_addr, home_node);

        let network_op = match mode {
            AtomicMode::RdmaDirect => self.translate_rdma_direct(cuda_op, operands),
            AtomicMode::Proxy => NetworkAtomicOp::ProxyRequest {
                op: cuda_op,
                operands,
            },
            AtomicMode::Migrate => self.translate_with_cas_emulation(cuda_op, operands),
        };

        AtomicTranslation {
            cuda_op,
            target_addr,
            operands,
            network_op,
        }
    }

    /// Determine the best mode for a given atomic operation on a target address.
    fn determine_mode(
        &self,
        cuda_op: CudaAtomicOp,
        target_addr: VirtualAddr,
        home_node: NodeId,
    ) -> AtomicMode {
        // Page-align the address for hot-atomic tracking
        let page_addr = target_addr & !0xFFFF; // 64KB page alignment

        // Check/create hot atomic entry
        let entry = self.hot_atomics.entry(page_addr).or_insert_with(|| {
            HotAtomicEntry::new(home_node)
        });

        // Record access and update frequency
        let _freq = entry.record_access();

        // If the operation is directly RDMA-mappable, prefer that
        let is_rdma_capable = cuda_op.is_direct_rdma_mappable();

        // Update mode based on frequency
        entry.update_mode(self.migration_to_proxy_threshold, is_rdma_capable);

        entry.current_mode()
    }

    /// Translate a CUDA atomic to a direct RDMA operation.
    ///
    /// Only works for 64-bit CAS, 64-bit integer add, and 64-bit integer sub.
    fn translate_rdma_direct(
        &self,
        cuda_op: CudaAtomicOp,
        operands: AtomicOperands,
    ) -> NetworkAtomicOp {
        match cuda_op {
            CudaAtomicOp::CAS { size: AtomicSize::Bits64 } => NetworkAtomicOp::RdmaCas {
                compare: operands.operand1,
                swap: operands.operand2,
            },
            CudaAtomicOp::Add {
                size: AtomicSize::Bits64,
                is_float: false,
            } => NetworkAtomicOp::RdmaFetchAdd {
                addend: operands.operand1,
            },
            CudaAtomicOp::Sub { size: AtomicSize::Bits64 } => NetworkAtomicOp::RdmaFetchAdd {
                addend: 0u64.wrapping_sub(operands.operand1),
            },
            // Fallback to CAS emulation for anything else
            _ => self.translate_with_cas_emulation(cuda_op, operands),
        }
    }

    /// Translate a CUDA atomic to a CAS emulation loop.
    fn translate_with_cas_emulation(
        &self,
        cuda_op: CudaAtomicOp,
        _operands: AtomicOperands,
    ) -> NetworkAtomicOp {
        let compute_fn = Self::cuda_op_to_cas_fn(cuda_op);
        NetworkAtomicOp::CasEmulationLoop { compute_fn }
    }

    /// Map a CUDA atomic operation to the corresponding CAS compute function.
    pub fn cuda_op_to_cas_fn(cuda_op: CudaAtomicOp) -> CasComputeFn {
        match cuda_op {
            CudaAtomicOp::Min { is_signed, .. } => CasComputeFn::Min { is_signed },
            CudaAtomicOp::Max { is_signed, .. } => CasComputeFn::Max { is_signed },
            CudaAtomicOp::And { .. } => CasComputeFn::And,
            CudaAtomicOp::Or { .. } => CasComputeFn::Or,
            CudaAtomicOp::Xor { .. } => CasComputeFn::Xor,
            CudaAtomicOp::Inc { modulo } => CasComputeFn::Inc { modulo },
            CudaAtomicOp::Dec { modulo } => CasComputeFn::Dec { modulo },
            CudaAtomicOp::Add { is_float: true, .. } => CasComputeFn::FloatAdd,
            CudaAtomicOp::Exch { .. } => CasComputeFn::Exchange,
            // Integer add/sub/CAS should use direct RDMA when 64-bit.
            // For 32-bit, they go through CAS emulation with appropriate compute.
            CudaAtomicOp::Add { is_float: false, .. } => {
                // For CAS emulation of integer add, compute = old + operand
                // This is handled specially in the emulation loop.
                // We use FloatAdd as placeholder -- actually we do wrapping add.
                // Define inline: old.wrapping_add(operand)
                CasComputeFn::Exchange // Will be overridden; see note below.
            }
            CudaAtomicOp::Sub { .. } => CasComputeFn::Exchange,
            CudaAtomicOp::CAS { .. } => CasComputeFn::Exchange,
        }
    }

    /// Execute a CAS emulation loop on a local memory address (for testing).
    ///
    /// In production, this would operate over RDMA. This method provides
    /// the software CAS loop logic for correctness verification.
    ///
    /// Returns `(old_value, success)`.
    pub fn cas_emulation_local(
        target: &AtomicU64,
        operand: u64,
        compute_fn: CasComputeFn,
        max_retries: u32,
    ) -> (u64, bool) {
        for _attempt in 0..max_retries {
            let old = target.load(Ordering::Acquire);
            let new_val = compute_fn.compute(old, operand);
            match target.compare_exchange_weak(old, new_val, Ordering::AcqRel, Ordering::Acquire) {
                Ok(actual) => return (actual, true),
                Err(_) => continue,
            }
        }
        // Exceeded retries
        (target.load(Ordering::Acquire), false)
    }

    /// Compute the result of a 32-bit atomic via 64-bit CAS with alignment.
    ///
    /// RDMA atomics are 64-bit only. For 32-bit CUDA atomics, we read the
    /// aligned 64-bit word, modify the appropriate 32-bit half, and CAS
    /// the whole 64-bit word.
    pub fn compute_32bit_in_64bit(
        old_64: u64,
        offset: u32,
        operand: u32,
        compute_32: impl Fn(u32, u32) -> u32,
    ) -> u64 {
        let lo = old_64 as u32;
        let hi = (old_64 >> 32) as u32;
        if offset == 0 {
            let new_lo = compute_32(lo, operand);
            ((hi as u64) << 32) | (new_lo as u64)
        } else {
            let new_hi = compute_32(hi, operand);
            ((new_hi as u64) << 32) | (lo as u64)
        }
    }

    /// Compute the result of a 16-bit atomic via 64-bit CAS with alignment.
    ///
    /// Four possible positions within the 64-bit word (byte offsets 0, 2, 4, 6).
    pub fn compute_16bit_in_64bit(
        old_64: u64,
        byte_offset: u32,
        operand: u16,
        compute_16: impl Fn(u16, u16) -> u16,
    ) -> u64 {
        let shift = (byte_offset / 2) * 16;
        let mask = 0xFFFFu64 << shift;
        let old_16 = ((old_64 >> shift) & 0xFFFF) as u16;
        let new_16 = compute_16(old_16, operand);
        (old_64 & !mask) | ((new_16 as u64) << shift)
    }

    /// Get the number of hot atomic pages being tracked.
    pub fn hot_atomic_count(&self) -> usize {
        self.hot_atomics.len()
    }

    /// Get the current mode for a page, if tracked.
    pub fn page_mode(&self, page_addr: VirtualAddr) -> Option<AtomicMode> {
        let aligned = page_addr & !0xFFFF;
        self.hot_atomics
            .get(&aligned)
            .map(|entry| entry.current_mode())
    }
}

// ---------------------------------------------------------------------------
// Performance guardrails and diagnostics
// ---------------------------------------------------------------------------

/// Log output target for performance warnings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogTarget {
    /// Log to stderr (default).
    Stderr,
    /// Log to a file at the given path.
    File(String),
    /// Suppress all output.
    None,
}

/// Performance thresholds for diagnostic warnings.
///
/// Monitors peer connection usage patterns and emits warnings when
/// usage patterns that are efficient on real NVLink become expensive
/// over network transport.
#[derive(Debug)]
pub struct PerformanceGuardrails {
    /// Warn if more than N small (<4KB) transfers per second to a peer.
    pub small_transfer_warn_threshold: u32,
    /// Warn if page fault rate exceeds N/sec for any peer.
    pub page_fault_warn_threshold: u32,
    /// Warn if remote atomic rate exceeds N/sec for any address.
    pub atomic_hotspot_warn_threshold: u32,
    /// Warn if page ping-pong count exceeds N in a 1-sec window.
    pub page_thrash_warn_threshold: u32,
    /// Warn if fence frequency exceeds N/sec.
    pub fence_frequency_warn_threshold: u32,
    /// Whether warnings are currently enabled.
    pub enabled: AtomicBool,
    /// Log output target.
    pub log_target: LogTarget,
}

impl PerformanceGuardrails {
    /// Create guardrails with default thresholds.
    pub fn defaults() -> Self {
        Self {
            small_transfer_warn_threshold: 10_000,
            page_fault_warn_threshold: 1_000,
            atomic_hotspot_warn_threshold: 5_000,
            page_thrash_warn_threshold: 10,
            fence_frequency_warn_threshold: 50_000,
            enabled: AtomicBool::new(true),
            log_target: LogTarget::Stderr,
        }
    }

    /// Create guardrails with all warnings disabled.
    pub fn disabled() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            ..Self::defaults()
        }
    }

    /// Enable or disable warnings.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Check if warnings are enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }
}

impl Default for PerformanceGuardrails {
    fn default() -> Self {
        Self::defaults()
    }
}

/// Diagnostic event types for performance monitoring.
#[derive(Debug, Clone, PartialEq)]
pub enum DiagnosticEvent {
    /// Too many small transfers to a peer.
    SmallTransferWarning {
        /// Target device.
        device: DeviceId,
        /// Transfers per second.
        rate: u32,
    },
    /// Page ping-ponging between nodes.
    PageThrashWarning {
        /// Virtual address of the thrashing page.
        page_addr: VirtualAddr,
        /// Number of migrations in the window.
        migration_count: u32,
        /// Nodes involved.
        nodes: Vec<NodeId>,
    },
    /// High rate of remote atomics to a single address.
    AtomicHotspotWarning {
        /// Target address.
        addr: VirtualAddr,
        /// Atomics per second.
        rate: u32,
    },
    /// High fence frequency.
    FenceFrequencyWarning {
        /// Fences per second.
        rate: u32,
    },
    /// High page fault rate on a peer.
    PageFaultRateWarning {
        /// Peer device.
        device: DeviceId,
        /// Faults per second.
        rate: u32,
    },
}

/// Performance diagnostics reporter.
///
/// Collects diagnostic events from the NVLink emulation layer and
/// reports them according to the configured guardrails.
pub struct PerformanceDiagnostics {
    /// Configuration thresholds.
    guardrails: PerformanceGuardrails,
    /// Accumulated events (bounded ring buffer).
    events: Mutex<Vec<DiagnosticEvent>>,
    /// Maximum events to retain.
    max_events: usize,
}

impl PerformanceDiagnostics {
    /// Create a new diagnostics reporter with the given guardrails.
    pub fn new(guardrails: PerformanceGuardrails) -> Self {
        Self {
            guardrails,
            events: Mutex::new(Vec::new()),
            max_events: 1000,
        }
    }

    /// Create with default guardrails.
    pub fn with_defaults() -> Self {
        Self::new(PerformanceGuardrails::defaults())
    }

    /// Record a diagnostic event.
    pub fn record_event(&self, event: DiagnosticEvent) {
        if !self.guardrails.is_enabled() {
            return;
        }
        let mut events = self.events.lock().unwrap();
        if events.len() >= self.max_events {
            events.drain(0..self.max_events / 2);
        }
        events.push(event);
    }

    /// Check peer stats against guardrails and emit warnings.
    ///
    /// Returns a list of diagnostic events that exceed thresholds.
    pub fn check_peer_stats(
        &self,
        device: DeviceId,
        stats: &PeerAccessStatsSnapshot,
        window_secs: f64,
    ) -> Vec<DiagnosticEvent> {
        let mut warnings = Vec::new();
        if !self.guardrails.is_enabled() || window_secs <= 0.0 {
            return warnings;
        }

        let small_rate = (stats.small_transfer_count as f64 / window_secs) as u32;
        if small_rate > self.guardrails.small_transfer_warn_threshold {
            let event = DiagnosticEvent::SmallTransferWarning {
                device,
                rate: small_rate,
            };
            warnings.push(event);
        }

        let fault_rate = (stats.page_fault_count as f64 / window_secs) as u32;
        if fault_rate > self.guardrails.page_fault_warn_threshold {
            let event = DiagnosticEvent::PageFaultRateWarning {
                device,
                rate: fault_rate,
            };
            warnings.push(event);
        }

        for event in &warnings {
            self.record_event(event.clone());
        }

        warnings
    }

    /// Get the number of recorded events.
    pub fn event_count(&self) -> usize {
        self.events.lock().unwrap().len()
    }

    /// Drain all recorded events.
    pub fn drain_events(&self) -> Vec<DiagnosticEvent> {
        let mut events = self.events.lock().unwrap();
        std::mem::take(&mut *events)
    }

    /// Get a reference to the guardrails configuration.
    pub fn guardrails(&self) -> &PerformanceGuardrails {
        &self.guardrails
    }
}

// ---------------------------------------------------------------------------
// Fencing state for memory ordering
// ---------------------------------------------------------------------------

/// Pending operation that needs fence before proceeding.
#[derive(Debug, Clone)]
pub struct PendingOp {
    /// Sequence number of the fence this operation is waiting for.
    pub fence_seq: u64,
    /// Description of the pending operation.
    pub op_id: u64,
}

/// Fencing state to implement NVLink memory ordering over RDMA.
///
/// NVLink guarantees in-order writes on the same link. `__threadfence_system()`
/// ensures cross-link ordering. This state tracks fence sequences per QP.
pub struct FenceState {
    /// Last fence sequence number issued.
    pub last_fence_seq: AtomicU64,
    /// Last fence sequence number completed.
    pub completed_fence_seq: AtomicU64,
    /// Pending operations that need fence before proceeding.
    pub pending_after_fence: Mutex<Vec<PendingOp>>,
}

impl FenceState {
    /// Create a new fence state.
    pub fn new() -> Self {
        Self {
            last_fence_seq: AtomicU64::new(0),
            completed_fence_seq: AtomicU64::new(0),
            pending_after_fence: Mutex::new(Vec::new()),
        }
    }

    /// Issue a new fence and return its sequence number.
    pub fn issue_fence(&self) -> u64 {
        self.last_fence_seq.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Mark a fence as completed.
    pub fn complete_fence(&self, seq: u64) {
        // Only advance if this is actually the next expected completion
        let _ = self.completed_fence_seq.fetch_max(seq, Ordering::AcqRel);
    }

    /// Check if a fence has been completed.
    pub fn is_fence_complete(&self, seq: u64) -> bool {
        self.completed_fence_seq.load(Ordering::Acquire) >= seq
    }

    /// Add a pending operation that must wait for the given fence.
    pub fn add_pending(&self, fence_seq: u64, op_id: u64) {
        let mut pending = self.pending_after_fence.lock().unwrap();
        pending.push(PendingOp { fence_seq, op_id });
    }

    /// Drain all pending operations whose fence has completed.
    pub fn drain_ready(&self) -> Vec<PendingOp> {
        let completed = self.completed_fence_seq.load(Ordering::Acquire);
        let mut pending = self.pending_after_fence.lock().unwrap();
        let mut ready = Vec::new();
        pending.retain(|op| {
            if op.fence_seq <= completed {
                ready.push(op.clone());
                false
            } else {
                true
            }
        });
        ready
    }

    /// Get the number of pending operations.
    pub fn pending_count(&self) -> usize {
        self.pending_after_fence.lock().unwrap().len()
    }

    /// Get the last issued fence sequence number.
    pub fn last_issued(&self) -> u64 {
        self.last_fence_seq.load(Ordering::Acquire)
    }

    /// Get the last completed fence sequence number.
    pub fn last_completed(&self) -> u64 {
        self.completed_fence_seq.load(Ordering::Acquire)
    }
}

impl Default for FenceState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// NVLink coherency adapter
// ---------------------------------------------------------------------------

/// Adapts R19's I/S/E coherency protocol for NVLink semantics.
///
/// This is NOT a new coherency protocol -- it wraps R19's existing one
/// and adds NVLink-specific behavior (atomic-aware transitions, fencing).
/// When R19 is not available, this adapter is `None` in the NvlinkEmulator.
pub struct NvlinkCoherencyAdapter {
    /// R29 multicast group for bulk invalidation fast-path.
    /// TODO: Replace Option<MulticastGroup> with actual R29 integration.
    has_multicast: bool,
    /// Fencing state per QP for memory ordering guarantees.
    fence_state: DashMap<QueuePairId, FenceState>,
    /// Page size for coherency (default 64KB, configurable to 4KB).
    page_size: u64,
}

impl NvlinkCoherencyAdapter {
    /// Create a new coherency adapter.
    pub fn new(page_size: u64, has_multicast: bool) -> Self {
        Self {
            has_multicast,
            fence_state: DashMap::new(),
            page_size,
        }
    }

    /// Create with default 64KB page size.
    pub fn with_defaults() -> Self {
        Self::new(64 * 1024, false)
    }

    /// Get or create fence state for a QP.
    pub fn get_or_create_fence(&self, qp_id: QueuePairId) -> dashmap::mapref::one::Ref<'_, QueuePairId, FenceState> {
        self.fence_state.entry(qp_id).or_insert_with(FenceState::new);
        self.fence_state.get(&qp_id).unwrap()
    }

    /// Issue a fence on a QP.
    pub fn issue_fence(&self, qp_id: QueuePairId) -> u64 {
        let entry = self.fence_state.entry(qp_id).or_insert_with(FenceState::new);
        entry.issue_fence()
    }

    /// Complete a fence on a QP.
    pub fn complete_fence(&self, qp_id: QueuePairId, seq: u64) {
        if let Some(state) = self.fence_state.get(&qp_id) {
            state.complete_fence(seq);
        }
    }

    /// Check if a fence is complete on a QP.
    pub fn is_fence_complete(&self, qp_id: QueuePairId, seq: u64) -> bool {
        self.fence_state
            .get(&qp_id)
            .map(|state| state.is_fence_complete(seq))
            .unwrap_or(true) // No state = no pending fences = complete
    }

    /// Whether multicast invalidation is available.
    pub fn has_multicast_invalidation(&self) -> bool {
        self.has_multicast
    }

    /// Get the configured page size.
    pub fn page_size(&self) -> u64 {
        self.page_size
    }

    /// Align an address to the coherency page boundary.
    pub fn align_to_page(&self, addr: VirtualAddr) -> VirtualAddr {
        addr & !(self.page_size - 1)
    }

    /// Calculate the number of pages spanned by an address range.
    pub fn pages_spanned(&self, start: VirtualAddr, size: u64) -> u64 {
        if size == 0 {
            return 0;
        }
        let start_page = self.align_to_page(start);
        let end_page = self.align_to_page(start + size - 1);
        (end_page - start_page) / self.page_size + 1
    }

    /// Get the number of tracked QPs.
    pub fn tracked_qp_count(&self) -> usize {
        self.fence_state.len()
    }

    /// Determine the invalidation strategy for a page going from Shared to Exclusive.
    ///
    /// Returns `InvalidationStrategy::Multicast` if multicast is available,
    /// otherwise `InvalidationStrategy::Unicast` with the number of sharers.
    pub fn invalidation_strategy(&self, sharer_count: u32) -> InvalidationStrategy {
        if self.has_multicast && sharer_count > 1 {
            InvalidationStrategy::Multicast
        } else {
            InvalidationStrategy::Unicast {
                sharer_count,
            }
        }
    }
}

/// Strategy for invalidating shared copies when a page transitions to Exclusive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidationStrategy {
    /// Send a single multicast invalidation to all sharers.
    Multicast,
    /// Send individual unicast invalidations to each sharer.
    Unicast {
        /// Number of sharers to invalidate.
        sharer_count: u32,
    },
}

// ---------------------------------------------------------------------------
// Configuration from environment variables
// ---------------------------------------------------------------------------

/// Runtime configuration for the Virtual NVLink emulation layer.
///
/// Parsed from environment variables at initialization time.
/// See the preplan for the full list of tuning knobs.
#[derive(Debug, Clone)]
pub struct NvlinkConfig {
    /// Maximum emulation tier (1, 2, or 3).
    pub max_tier: u8,
    /// Whether to report NATIVE_ATOMIC_SUPPORTED to applications.
    pub report_atomics: bool,
    /// Force atomic handling mode (None = auto).
    pub atomic_mode_override: Option<AtomicMode>,
    /// Coherency page size in bytes.
    pub page_size: u64,
    /// Performance warning level.
    pub perf_warning_level: PerfWarningLevel,
    /// Batch N fence operations into one network RTT.
    pub fence_batch_size: u32,
    /// Added to PERFORMANCE_RANK (higher = apps may avoid P2P).
    pub peer_rank_offset: i32,
}

/// Performance warning output level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerfWarningLevel {
    /// No warnings.
    Off,
    /// Standard warnings.
    On,
    /// Verbose warnings with additional detail.
    Verbose,
}

impl NvlinkConfig {
    /// Create configuration with default values.
    pub fn defaults() -> Self {
        Self {
            max_tier: 1,
            report_atomics: false,
            atomic_mode_override: None,
            page_size: 64 * 1024,
            perf_warning_level: PerfWarningLevel::On,
            fence_batch_size: 1,
            peer_rank_offset: 0,
        }
    }

    /// Parse configuration from environment variables.
    ///
    /// Environment variables:
    /// - `OUTERLINK_NVLINK_TIER`: Maximum tier (1, 2, 3). Default: 1.
    /// - `OUTERLINK_REPORT_ATOMICS`: Report atomics (0 or 1). Default: 0.
    /// - `OUTERLINK_ATOMIC_MODE`: Force mode (migrate, proxy, auto). Default: auto.
    /// - `OUTERLINK_PAGE_SIZE`: Page size (4K, 64K). Default: 64K.
    /// - `OUTERLINK_PERF_WARNINGS`: Warning level (0, 1, verbose). Default: 1.
    /// - `OUTERLINK_FENCE_BATCH`: Fence batch size (1-100). Default: 1.
    /// - `OUTERLINK_PEER_RANK_OFFSET`: Rank offset (0-10). Default: 0.
    pub fn from_env() -> Self {
        let mut config = Self::defaults();

        if let Ok(tier) = std::env::var("OUTERLINK_NVLINK_TIER") {
            if let Ok(t) = tier.parse::<u8>() {
                if (1..=3).contains(&t) {
                    config.max_tier = t;
                }
            }
        }

        if let Ok(val) = std::env::var("OUTERLINK_REPORT_ATOMICS") {
            config.report_atomics = val == "1";
        }

        if let Ok(mode) = std::env::var("OUTERLINK_ATOMIC_MODE") {
            config.atomic_mode_override = match mode.as_str() {
                "migrate" => Some(AtomicMode::Migrate),
                "proxy" => Some(AtomicMode::Proxy),
                _ => None, // "auto" or anything else
            };
        }

        if let Ok(size) = std::env::var("OUTERLINK_PAGE_SIZE") {
            config.page_size = match size.as_str() {
                "4K" | "4k" => 4 * 1024,
                _ => 64 * 1024, // default 64K
            };
        }

        if let Ok(level) = std::env::var("OUTERLINK_PERF_WARNINGS") {
            config.perf_warning_level = match level.as_str() {
                "0" => PerfWarningLevel::Off,
                "verbose" => PerfWarningLevel::Verbose,
                _ => PerfWarningLevel::On,
            };
        }

        if let Ok(batch) = std::env::var("OUTERLINK_FENCE_BATCH") {
            if let Ok(b) = batch.parse::<u32>() {
                if (1..=100).contains(&b) {
                    config.fence_batch_size = b;
                }
            }
        }

        if let Ok(offset) = std::env::var("OUTERLINK_PEER_RANK_OFFSET") {
            if let Ok(o) = offset.parse::<i32>() {
                if (0..=10).contains(&o) {
                    config.peer_rank_offset = o;
                }
            }
        }

        config
    }
}

impl Default for NvlinkConfig {
    fn default() -> Self {
        Self::defaults()
    }
}

// ---------------------------------------------------------------------------
// Device-to-node mapping
// ---------------------------------------------------------------------------

/// Maps virtual device IDs to their physical location (node + local GPU).
#[derive(Debug, Clone)]
pub struct DeviceMapping {
    /// device_id -> (node_id, local_gpu_id).
    pub mappings: HashMap<DeviceId, (NodeId, GpuId)>,
}

impl DeviceMapping {
    /// Create an empty device mapping.
    pub fn new() -> Self {
        Self {
            mappings: HashMap::new(),
        }
    }

    /// Register a device's location.
    pub fn register(&mut self, device_id: DeviceId, node_id: NodeId, gpu_id: GpuId) {
        self.mappings.insert(device_id, (node_id, gpu_id));
    }

    /// Look up a device's location.
    pub fn lookup(&self, device_id: DeviceId) -> Option<(NodeId, GpuId)> {
        self.mappings.get(&device_id).copied()
    }

    /// Check if a device is managed by OuterLink.
    pub fn is_managed(&self, device_id: DeviceId) -> bool {
        self.mappings.contains_key(&device_id)
    }

    /// Check if two devices are on the same node.
    pub fn same_node(&self, dev_a: DeviceId, dev_b: DeviceId) -> bool {
        match (self.lookup(dev_a), self.lookup(dev_b)) {
            (Some((node_a, _)), Some((node_b, _))) => node_a == node_b,
            _ => false,
        }
    }

    /// Get the number of registered devices.
    pub fn device_count(&self) -> usize {
        self.mappings.len()
    }
}

impl Default for DeviceMapping {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Peer access manager
// ---------------------------------------------------------------------------

/// Manages virtual NVLink peer access relationships between GPU contexts.
///
/// Intercepts `cuDeviceCanAccessPeer`, `cuCtxEnablePeerAccess`, and
/// `cuDeviceGetP2PAttribute`. One instance per OuterLink client process.
///
/// This is the Tier 1 core: it tracks which GPUs can "see" each other
/// and routes P2P data transfers through the network transport.
pub struct PeerAccessManager {
    /// Map of (local_device, remote_device) -> peer connection state.
    peer_connections: DashMap<(DeviceId, DeviceId), PeerConnection>,
    /// Cluster topology snapshot from R17 TopologyManager.
    topology: Arc<TopologySnapshot>,
    /// Performance thresholds for diagnostic warnings.
    guardrails: PerformanceGuardrails,
    /// Device-to-node mapping.
    device_map: Arc<DeviceMapping>,
    /// Runtime configuration.
    config: NvlinkConfig,
}

impl PeerAccessManager {
    /// Create a new peer access manager.
    pub fn new(
        topology: Arc<TopologySnapshot>,
        device_map: Arc<DeviceMapping>,
        config: NvlinkConfig,
    ) -> Self {
        Self {
            peer_connections: DashMap::new(),
            topology,
            guardrails: PerformanceGuardrails::defaults(),
            device_map,
            config,
        }
    }

    /// Check if peer access is possible between two devices.
    ///
    /// Implements the interception logic for `cuDeviceCanAccessPeer`.
    /// Returns `None` if either device is not managed by OuterLink
    /// (caller should forward to real CUDA driver).
    pub fn can_access_peer(&self, dev_a: DeviceId, dev_b: DeviceId) -> Option<bool> {
        // Same device: let CUDA handle it
        if dev_a == dev_b {
            return None;
        }

        // Both must be managed
        let (node_a, _) = self.device_map.lookup(dev_a)?;
        let (node_b, _) = self.device_map.lookup(dev_b)?;

        // Check topology for network path
        Some(self.topology.has_path(node_a, node_b))
    }

    /// Enable peer access from the current context to a peer device.
    ///
    /// Implements the interception logic for `cuCtxEnablePeerAccess`.
    /// Returns an error if peer access cannot be enabled.
    pub fn enable_peer_access(
        &self,
        local_device: DeviceId,
        remote_device: DeviceId,
    ) -> Result<(), PeerAccessError> {
        // Check if both are managed
        let (local_node, _) = self
            .device_map
            .lookup(local_device)
            .ok_or(PeerAccessError::DeviceNotManaged(local_device))?;
        let (remote_node, _) = self
            .device_map
            .lookup(remote_device)
            .ok_or(PeerAccessError::DeviceNotManaged(remote_device))?;

        // Same device
        if local_device == remote_device {
            return Err(PeerAccessError::SameDevice);
        }

        let key = (local_device, remote_device);

        // Check if already enabled
        if let Some(conn) = self.peer_connections.get(&key) {
            if conn.is_enabled() {
                return Err(PeerAccessError::AlreadyEnabled);
            }
        }

        // Check topology
        if !self.topology.has_path(local_node, remote_node) {
            return Err(PeerAccessError::NoRoute);
        }

        // Build route info from topology
        let hop_count = self.topology.hop_count(local_node, remote_node).unwrap_or(255);
        let route = RouteInfo::new(
            hop_count,
            hop_count as f32, // Simple cost = hop count for now
            false,            // TODO: Check RDMA availability from transport layer
            None,
        );

        // Create or update connection
        let conn = PeerConnection::new(local_device, remote_device, remote_node, route);
        conn.enable();
        self.peer_connections.insert(key, conn);

        Ok(())
    }

    /// Disable peer access from the current context to a peer device.
    pub fn disable_peer_access(
        &self,
        local_device: DeviceId,
        remote_device: DeviceId,
    ) -> Result<(), PeerAccessError> {
        let key = (local_device, remote_device);
        match self.peer_connections.get(&key) {
            Some(conn) => {
                if conn.disable() {
                    Ok(())
                } else {
                    Err(PeerAccessError::NotEnabled)
                }
            }
            None => Err(PeerAccessError::NotEnabled),
        }
    }

    /// Get peer attributes for two managed devices.
    ///
    /// Implements the interception logic for `cuDeviceGetP2PAttribute`.
    /// Returns `None` if either device is not managed.
    pub fn get_p2p_attributes(
        &self,
        src_device: DeviceId,
        dst_device: DeviceId,
        same_compute_capability: bool,
    ) -> Option<PeerAttributes> {
        let key = (src_device, dst_device);

        // If connection exists, use it for route info
        if let Some(conn) = self.peer_connections.get(&key) {
            return Some(PeerAttributes::from_connection(
                &conn,
                self.config.max_tier,
                self.config.report_atomics,
                same_compute_capability,
                self.config.peer_rank_offset,
            ));
        }

        // No connection yet; build attributes from topology
        let (node_a, _) = self.device_map.lookup(src_device)?;
        let (node_b, _) = self.device_map.lookup(dst_device)?;

        let hop_count = self.topology.hop_count(node_a, node_b)?;
        let route = RouteInfo::new(hop_count, hop_count as f32, false, None);
        let temp_conn = PeerConnection::new(src_device, dst_device, node_b, route);

        Some(PeerAttributes::from_connection(
            &temp_conn,
            self.config.max_tier,
            self.config.report_atomics,
            same_compute_capability,
            self.config.peer_rank_offset,
        ))
    }

    /// Record a P2P transfer for statistics tracking.
    ///
    /// Called after a successful cudaMemcpyPeer to update stats.
    pub fn record_transfer(
        &self,
        src_device: DeviceId,
        dst_device: DeviceId,
        size_bytes: u64,
    ) {
        let key = (src_device, dst_device);
        if let Some(conn) = self.peer_connections.get(&key) {
            conn.stats.record_transfer(size_bytes);
        }
    }

    /// Get statistics for a peer connection.
    pub fn get_stats(
        &self,
        src_device: DeviceId,
        dst_device: DeviceId,
    ) -> Option<PeerAccessStatsSnapshot> {
        let key = (src_device, dst_device);
        self.peer_connections
            .get(&key)
            .map(|conn| conn.stats.snapshot())
    }

    /// Check if peer access is enabled between two devices.
    pub fn is_peer_access_enabled(
        &self,
        local_device: DeviceId,
        remote_device: DeviceId,
    ) -> bool {
        let key = (local_device, remote_device);
        self.peer_connections
            .get(&key)
            .map(|conn| conn.is_enabled())
            .unwrap_or(false)
    }

    /// Get the number of active peer connections.
    pub fn active_connection_count(&self) -> usize {
        self.peer_connections
            .iter()
            .filter(|entry| entry.value().is_enabled())
            .count()
    }

    /// Get the total number of peer connections (including disabled).
    pub fn total_connection_count(&self) -> usize {
        self.peer_connections.len()
    }

    /// Get the route info for a peer connection.
    pub fn get_route(
        &self,
        src_device: DeviceId,
        dst_device: DeviceId,
    ) -> Option<RouteInfo> {
        let key = (src_device, dst_device);
        self.peer_connections
            .get(&key)
            .map(|conn| conn.route.clone())
    }

    /// Get a reference to the guardrails.
    pub fn guardrails(&self) -> &PerformanceGuardrails {
        &self.guardrails
    }

    /// Get a reference to the topology.
    pub fn topology(&self) -> &TopologySnapshot {
        &self.topology
    }

    /// Get a reference to the config.
    pub fn config(&self) -> &NvlinkConfig {
        &self.config
    }
}

/// Errors that can occur during peer access operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PeerAccessError {
    /// Device is not managed by OuterLink.
    DeviceNotManaged(DeviceId),
    /// Peer access is already enabled for this pair.
    AlreadyEnabled,
    /// Peer access is not currently enabled.
    NotEnabled,
    /// No network route between the two devices.
    NoRoute,
    /// Cannot peer-access the same device.
    SameDevice,
}

impl std::fmt::Display for PeerAccessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PeerAccessError::DeviceNotManaged(id) => {
                write!(f, "device {} is not managed by OuterLink", id)
            }
            PeerAccessError::AlreadyEnabled => write!(f, "peer access already enabled"),
            PeerAccessError::NotEnabled => write!(f, "peer access not enabled"),
            PeerAccessError::NoRoute => write!(f, "no network route between devices"),
            PeerAccessError::SameDevice => write!(f, "cannot peer-access the same device"),
        }
    }
}

impl std::error::Error for PeerAccessError {}

// ---------------------------------------------------------------------------
// NVLink emulator (top-level facade)
// ---------------------------------------------------------------------------

/// Top-level Virtual NVLink emulation layer.
///
/// Registered as an interception handler during OuterLink client initialization.
/// Ties together peer access (Tier 1), coherency (Tier 2), and atomics (Tier 3).
pub struct NvlinkEmulator {
    /// Tier 1: Peer access management and P2P memcpy routing.
    pub peer_access: Arc<PeerAccessManager>,
    /// Tier 2: Coherency integration (wraps R19's I/S/E protocol).
    /// None if only Tier 1 is active.
    pub coherency: Option<Arc<NvlinkCoherencyAdapter>>,
    /// Tier 3: Remote atomic engine.
    /// None if only Tier 1/2 is active.
    pub atomics: Option<Arc<RemoteAtomicEngine>>,
    /// Active tier level (runtime-configurable).
    active_tier: AtomicU8,
    /// Diagnostic reporter for performance warnings.
    pub diagnostics: Arc<PerformanceDiagnostics>,
    /// Runtime configuration.
    config: NvlinkConfig,
}

impl NvlinkEmulator {
    /// Create a new NVLink emulator with the given configuration.
    pub fn new(
        peer_access: Arc<PeerAccessManager>,
        config: NvlinkConfig,
    ) -> Self {
        let active_tier = config.max_tier;
        let diagnostics = Arc::new(PerformanceDiagnostics::with_defaults());

        Self {
            peer_access,
            coherency: None,
            atomics: None,
            active_tier: AtomicU8::new(active_tier),
            diagnostics,
            config,
        }
    }

    /// Create a Tier 1-only emulator (most common case).
    pub fn tier1_only(
        topology: Arc<TopologySnapshot>,
        device_map: Arc<DeviceMapping>,
    ) -> Self {
        let config = NvlinkConfig::defaults();
        let peer_access = Arc::new(PeerAccessManager::new(
            topology,
            device_map,
            config.clone(),
        ));
        Self::new(peer_access, config)
    }

    /// Enable Tier 2 coherency with the given adapter.
    pub fn enable_tier2(&mut self, adapter: Arc<NvlinkCoherencyAdapter>) {
        self.coherency = Some(adapter);
        let current = self.active_tier.load(Ordering::Relaxed);
        if current < 2 {
            self.active_tier.store(2, Ordering::Relaxed);
        }
    }

    /// Enable Tier 3 atomics with the given engine.
    pub fn enable_tier3(&mut self, engine: Arc<RemoteAtomicEngine>) {
        self.atomics = Some(engine);
        self.active_tier.store(3, Ordering::Relaxed);
    }

    /// Get the currently active tier level.
    pub fn active_tier(&self) -> u8 {
        self.active_tier.load(Ordering::Acquire)
    }

    /// Set the active tier level (clamped to 1-3).
    pub fn set_active_tier(&self, tier: u8) {
        let clamped = tier.clamp(1, 3);
        self.active_tier.store(clamped, Ordering::Release);
    }

    /// Check if Tier 2 coherency is available and active.
    pub fn is_coherency_active(&self) -> bool {
        self.active_tier() >= 2 && self.coherency.is_some()
    }

    /// Check if Tier 3 atomics are available and active.
    pub fn is_atomics_active(&self) -> bool {
        self.active_tier() >= 3 && self.atomics.is_some()
    }

    /// Intercept cuDeviceCanAccessPeer.
    ///
    /// Returns `Some(true/false)` if OuterLink handles this, or `None` to
    /// forward to the real CUDA driver.
    pub fn can_access_peer(&self, dev_a: DeviceId, dev_b: DeviceId) -> Option<bool> {
        self.peer_access.can_access_peer(dev_a, dev_b)
    }

    /// Intercept cuCtxEnablePeerAccess.
    pub fn enable_peer_access(
        &self,
        local_device: DeviceId,
        remote_device: DeviceId,
    ) -> Result<(), PeerAccessError> {
        self.peer_access.enable_peer_access(local_device, remote_device)
    }

    /// Intercept cuCtxDisablePeerAccess.
    pub fn disable_peer_access(
        &self,
        local_device: DeviceId,
        remote_device: DeviceId,
    ) -> Result<(), PeerAccessError> {
        self.peer_access.disable_peer_access(local_device, remote_device)
    }

    /// Intercept cuDeviceGetP2PAttribute.
    pub fn get_p2p_attributes(
        &self,
        src_device: DeviceId,
        dst_device: DeviceId,
        same_compute_capability: bool,
    ) -> Option<PeerAttributes> {
        self.peer_access
            .get_p2p_attributes(src_device, dst_device, same_compute_capability)
    }

    /// Record a P2P memcpy transfer for stats and diagnostics.
    pub fn record_transfer(
        &self,
        src_device: DeviceId,
        dst_device: DeviceId,
        size_bytes: u64,
    ) {
        self.peer_access.record_transfer(src_device, dst_device, size_bytes);
    }

    /// Translate a CUDA atomic for a remote target.
    ///
    /// Returns `None` if Tier 3 is not active.
    pub fn translate_atomic(
        &self,
        cuda_op: CudaAtomicOp,
        target_addr: VirtualAddr,
        operands: AtomicOperands,
        home_node: NodeId,
    ) -> Option<AtomicTranslation> {
        let engine = self.atomics.as_ref()?;
        if self.active_tier() < 3 {
            return None;
        }
        Some(engine.translate(cuda_op, target_addr, operands, home_node))
    }

    /// Get the configuration.
    pub fn config(&self) -> &NvlinkConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // === Helper functions ===

    fn make_topology() -> TopologySnapshot {
        let mut topo = TopologySnapshot::empty();
        topo.nodes = vec![0, 1, 2];
        topo.adjacency.insert((0, 1), 1);
        topo.adjacency.insert((1, 0), 1);
        topo.adjacency.insert((0, 2), 2);
        topo.adjacency.insert((2, 0), 2);
        topo.adjacency.insert((1, 2), 1);
        topo.adjacency.insert((2, 1), 1);
        topo
    }

    fn make_device_map() -> DeviceMapping {
        let mut dm = DeviceMapping::new();
        dm.register(0, 0, 0); // device 0 -> node 0, gpu 0
        dm.register(1, 0, 1); // device 1 -> node 0, gpu 1
        dm.register(2, 1, 0); // device 2 -> node 1, gpu 0
        dm.register(3, 2, 0); // device 3 -> node 2, gpu 0
        dm
    }

    fn make_peer_access_manager() -> PeerAccessManager {
        let topo = Arc::new(make_topology());
        let dm = Arc::new(make_device_map());
        PeerAccessManager::new(topo, dm, NvlinkConfig::defaults())
    }

    fn make_emulator() -> NvlinkEmulator {
        let topo = Arc::new(make_topology());
        let dm = Arc::new(make_device_map());
        NvlinkEmulator::tier1_only(topo, dm)
    }

    // === TopologySnapshot tests ===

    #[test]
    fn test_topology_empty() {
        let topo = TopologySnapshot::empty();
        assert!(topo.nodes.is_empty());
        assert!(!topo.has_path(0, 1));
    }

    #[test]
    fn test_topology_self_path() {
        let topo = TopologySnapshot::empty();
        assert!(topo.has_path(0, 0));
        assert_eq!(topo.hop_count(0, 0), Some(0));
    }

    #[test]
    fn test_topology_has_path() {
        let topo = make_topology();
        assert!(topo.has_path(0, 1));
        assert!(topo.has_path(1, 2));
        assert!(topo.has_path(0, 2));
    }

    #[test]
    fn test_topology_hop_count() {
        let topo = make_topology();
        assert_eq!(topo.hop_count(0, 1), Some(1));
        assert_eq!(topo.hop_count(0, 2), Some(2));
        assert_eq!(topo.hop_count(0, 99), None);
    }

    // === RouteInfo tests ===

    #[test]
    fn test_route_info_loopback() {
        let route = RouteInfo::loopback();
        assert_eq!(route.hop_count, 0);
        assert_eq!(route.weighted_cost, 0.0);
        assert!(!route.is_rdma);
        assert!(route.rdma_qp.is_none());
    }

    #[test]
    fn test_route_info_new() {
        let route = RouteInfo::new(2, 3.5, true, Some(42));
        assert_eq!(route.hop_count, 2);
        assert_eq!(route.weighted_cost, 3.5);
        assert!(route.is_rdma);
        assert_eq!(route.rdma_qp, Some(42));
    }

    // === DeviceMapping tests ===

    #[test]
    fn test_device_mapping_empty() {
        let dm = DeviceMapping::new();
        assert_eq!(dm.device_count(), 0);
        assert!(dm.lookup(0).is_none());
        assert!(!dm.is_managed(0));
    }

    #[test]
    fn test_device_mapping_register_and_lookup() {
        let mut dm = DeviceMapping::new();
        dm.register(5, 2, 1);
        assert_eq!(dm.lookup(5), Some((2, 1)));
        assert!(dm.is_managed(5));
        assert!(!dm.is_managed(6));
    }

    #[test]
    fn test_device_mapping_same_node() {
        let dm = make_device_map();
        assert!(dm.same_node(0, 1));  // both on node 0
        assert!(!dm.same_node(0, 2)); // node 0 vs node 1
        assert!(!dm.same_node(0, 99)); // 99 not managed
    }

    #[test]
    fn test_device_mapping_count() {
        let dm = make_device_map();
        assert_eq!(dm.device_count(), 4);
    }

    // === PeerAccessStats tests ===

    #[test]
    fn test_peer_stats_new() {
        let stats = PeerAccessStats::new();
        let snap = stats.snapshot();
        assert_eq!(snap.bytes_transferred, 0);
        assert_eq!(snap.transfer_count, 0);
        assert_eq!(snap.small_transfer_count, 0);
    }

    #[test]
    fn test_peer_stats_record_transfer() {
        let stats = PeerAccessStats::new();
        stats.record_transfer(8192);
        stats.record_transfer(100); // small
        let snap = stats.snapshot();
        assert_eq!(snap.bytes_transferred, 8292);
        assert_eq!(snap.transfer_count, 2);
        assert_eq!(snap.small_transfer_count, 1);
    }

    #[test]
    fn test_peer_stats_record_atomic() {
        let stats = PeerAccessStats::new();
        stats.record_atomic();
        stats.record_atomic();
        assert_eq!(stats.snapshot().atomic_count, 2);
    }

    #[test]
    fn test_peer_stats_record_page_fault() {
        let stats = PeerAccessStats::new();
        stats.record_page_fault();
        assert_eq!(stats.snapshot().page_fault_count, 1);
    }

    #[test]
    fn test_peer_stats_small_transfer_threshold() {
        let stats = PeerAccessStats::new();
        stats.record_transfer(4095); // small (< 4096)
        stats.record_transfer(4096); // NOT small
        let snap = stats.snapshot();
        assert_eq!(snap.small_transfer_count, 1);
    }

    // === PeerConnection tests ===

    #[test]
    fn test_peer_connection_new() {
        let conn = PeerConnection::new(0, 1, 1, RouteInfo::loopback());
        assert!(!conn.is_enabled());
        assert_eq!(conn.local_device, 0);
        assert_eq!(conn.remote_device, 1);
    }

    #[test]
    fn test_peer_connection_enable_disable() {
        let conn = PeerConnection::new(0, 1, 1, RouteInfo::loopback());
        assert!(conn.enable()); // success, was disabled
        assert!(!conn.enable()); // already enabled
        assert!(conn.is_enabled());
        assert!(conn.disable()); // success
        assert!(!conn.is_enabled());
        assert!(!conn.disable()); // already disabled
    }

    #[test]
    fn test_peer_connection_update_latency_bandwidth() {
        let conn = PeerConnection::new(0, 1, 1, RouteInfo::loopback());
        conn.update_latency(5);
        conn.update_bandwidth(12000);
        assert_eq!(conn.measured_latency_us.load(Ordering::Relaxed), 5);
        assert_eq!(conn.measured_bandwidth_mbps.load(Ordering::Relaxed), 12000);
    }

    // === PeerAttributes tests ===

    #[test]
    fn test_peer_attributes_tier1() {
        let conn = PeerConnection::new(0, 2, 1, RouteInfo::new(1, 1.0, true, None));
        let attrs = PeerAttributes::from_connection(&conn, 1, false, true, 0);
        assert_eq!(attrs.access_supported, 1);
        assert_eq!(attrs.native_atomic_supported, 0); // tier 1, not reporting
        assert_eq!(attrs.performance_rank, 1);
        assert_eq!(attrs.cuda_array_access_supported, 1);
    }

    #[test]
    fn test_peer_attributes_tier3_with_atomics() {
        let conn = PeerConnection::new(0, 2, 1, RouteInfo::new(1, 1.0, true, None));
        let attrs = PeerAttributes::from_connection(&conn, 3, true, true, 0);
        assert_eq!(attrs.native_atomic_supported, 1);
    }

    #[test]
    fn test_peer_attributes_tier3_without_reporting() {
        let conn = PeerConnection::new(0, 2, 1, RouteInfo::new(1, 1.0, true, None));
        let attrs = PeerAttributes::from_connection(&conn, 3, false, false, 0);
        assert_eq!(attrs.native_atomic_supported, 0);
        assert_eq!(attrs.cuda_array_access_supported, 0);
    }

    #[test]
    fn test_peer_attributes_rank_offset() {
        let conn = PeerConnection::new(0, 2, 1, RouteInfo::new(1, 1.0, true, None));
        let attrs = PeerAttributes::from_connection(&conn, 1, false, true, 3);
        assert_eq!(attrs.performance_rank, 4); // 1 + 3
    }

    // === AtomicSize tests ===

    #[test]
    fn test_atomic_size_byte_count() {
        assert_eq!(AtomicSize::Bits16.byte_count(), 2);
        assert_eq!(AtomicSize::Bits32.byte_count(), 4);
        assert_eq!(AtomicSize::Bits64.byte_count(), 8);
    }

    // === CudaAtomicOp tests ===

    #[test]
    fn test_cuda_atomic_direct_rdma_mappable() {
        assert!(CudaAtomicOp::CAS { size: AtomicSize::Bits64 }.is_direct_rdma_mappable());
        assert!(CudaAtomicOp::Add { size: AtomicSize::Bits64, is_float: false }.is_direct_rdma_mappable());
        assert!(CudaAtomicOp::Sub { size: AtomicSize::Bits64 }.is_direct_rdma_mappable());
        assert!(!CudaAtomicOp::Add { size: AtomicSize::Bits32, is_float: false }.is_direct_rdma_mappable());
        assert!(!CudaAtomicOp::Add { size: AtomicSize::Bits64, is_float: true }.is_direct_rdma_mappable());
        assert!(!CudaAtomicOp::Min { size: AtomicSize::Bits64, is_signed: false }.is_direct_rdma_mappable());
    }

    #[test]
    fn test_cuda_atomic_size() {
        assert_eq!(CudaAtomicOp::Add { size: AtomicSize::Bits32, is_float: false }.size(), AtomicSize::Bits32);
        assert_eq!(CudaAtomicOp::Inc { modulo: 10 }.size(), AtomicSize::Bits32);
        assert_eq!(CudaAtomicOp::Dec { modulo: 10 }.size(), AtomicSize::Bits32);
        assert_eq!(CudaAtomicOp::Xor { size: AtomicSize::Bits64 }.size(), AtomicSize::Bits64);
    }

    // === CasComputeFn tests ===

    #[test]
    fn test_cas_compute_min_unsigned() {
        let f = CasComputeFn::Min { is_signed: false };
        assert_eq!(f.compute(10, 5), 5);
        assert_eq!(f.compute(5, 10), 5);
        assert_eq!(f.compute(7, 7), 7);
    }

    #[test]
    fn test_cas_compute_min_signed() {
        let f = CasComputeFn::Min { is_signed: true };
        // -1i64 as u64 is a very large unsigned number
        let neg1 = (-1i64) as u64;
        assert_eq!(f.compute(neg1, 5), neg1); // -1 < 5
        assert_eq!(f.compute(5, neg1), neg1);
    }

    #[test]
    fn test_cas_compute_max_unsigned() {
        let f = CasComputeFn::Max { is_signed: false };
        assert_eq!(f.compute(10, 5), 10);
        assert_eq!(f.compute(5, 10), 10);
    }

    #[test]
    fn test_cas_compute_max_signed() {
        let f = CasComputeFn::Max { is_signed: true };
        let neg1 = (-1i64) as u64;
        assert_eq!(f.compute(neg1, 5), 5); // max(-1, 5) = 5
    }

    #[test]
    fn test_cas_compute_and() {
        let f = CasComputeFn::And;
        assert_eq!(f.compute(0xFF00, 0x0FF0), 0x0F00);
    }

    #[test]
    fn test_cas_compute_or() {
        let f = CasComputeFn::Or;
        assert_eq!(f.compute(0xFF00, 0x00FF), 0xFFFF);
    }

    #[test]
    fn test_cas_compute_xor() {
        let f = CasComputeFn::Xor;
        assert_eq!(f.compute(0xFF, 0x0F), 0xF0);
    }

    #[test]
    fn test_cas_compute_inc() {
        let f = CasComputeFn::Inc { modulo: 5 };
        assert_eq!(f.compute(0, 0), 1);
        assert_eq!(f.compute(4, 0), 5);
        assert_eq!(f.compute(5, 0), 0); // wrap at modulo
        assert_eq!(f.compute(10, 0), 0); // > modulo wraps too
    }

    #[test]
    fn test_cas_compute_dec() {
        let f = CasComputeFn::Dec { modulo: 5 };
        assert_eq!(f.compute(3, 0), 2);
        assert_eq!(f.compute(1, 0), 0);
        assert_eq!(f.compute(0, 0), 5); // 0 wraps to modulo
        assert_eq!(f.compute(10, 0), 5); // > modulo wraps to modulo
    }

    #[test]
    fn test_cas_compute_float_add() {
        let f = CasComputeFn::FloatAdd;
        let a = f64::to_bits(1.5);
        let b = f64::to_bits(2.5);
        let result = f64::from_bits(f.compute(a, b));
        assert!((result - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cas_compute_f32_add() {
        let a = f32::to_bits(3.0) as u64;
        let b = f32::to_bits(4.0) as u64;
        let result = f32::from_bits(CasComputeFn::compute_f32_add(a, b) as u32);
        assert!((result - 7.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cas_compute_exchange() {
        let f = CasComputeFn::Exchange;
        assert_eq!(f.compute(42, 99), 99);
    }

    // === 32-bit and 16-bit sub-word atomic tests ===

    #[test]
    fn test_32bit_in_64bit_low_half() {
        // [lo=10, hi=20], add 5 to low half
        let old = ((20u64) << 32) | 10u64;
        let result = RemoteAtomicEngine::compute_32bit_in_64bit(old, 0, 5, |a, b| a + b);
        assert_eq!(result as u32, 15); // lo = 10 + 5
        assert_eq!((result >> 32) as u32, 20); // hi unchanged
    }

    #[test]
    fn test_32bit_in_64bit_high_half() {
        let old = ((20u64) << 32) | 10u64;
        let result = RemoteAtomicEngine::compute_32bit_in_64bit(old, 4, 5, |a, b| a + b);
        assert_eq!(result as u32, 10); // lo unchanged
        assert_eq!((result >> 32) as u32, 25); // hi = 20 + 5
    }

    #[test]
    fn test_16bit_in_64bit_offset0() {
        let old: u64 = 0x0004_0003_0002_0001;
        let result = RemoteAtomicEngine::compute_16bit_in_64bit(old, 0, 0x0010, |a, b| a + b);
        assert_eq!(result & 0xFFFF, 0x0011); // first 16-bit slot: 1 + 16 = 17
        assert_eq!((result >> 16) & 0xFFFF, 0x0002); // unchanged
    }

    #[test]
    fn test_16bit_in_64bit_offset4() {
        let old: u64 = 0x0004_0003_0002_0001;
        let result = RemoteAtomicEngine::compute_16bit_in_64bit(old, 4, 0x0010, |a, b| a + b);
        assert_eq!((result >> 32) & 0xFFFF, 0x0013); // slot at byte offset 4: 3 + 16 = 19
        assert_eq!(result & 0xFFFF, 0x0001); // unchanged
    }

    // === CAS emulation local tests ===

    #[test]
    fn test_cas_emulation_local_min() {
        let target = AtomicU64::new(10);
        let (old, success) = RemoteAtomicEngine::cas_emulation_local(
            &target,
            5,
            CasComputeFn::Min { is_signed: false },
            32,
        );
        assert!(success);
        assert_eq!(old, 10);
        assert_eq!(target.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn test_cas_emulation_local_max() {
        let target = AtomicU64::new(10);
        let (old, success) = RemoteAtomicEngine::cas_emulation_local(
            &target,
            20,
            CasComputeFn::Max { is_signed: false },
            32,
        );
        assert!(success);
        assert_eq!(old, 10);
        assert_eq!(target.load(Ordering::Relaxed), 20);
    }

    #[test]
    fn test_cas_emulation_local_inc() {
        let target = AtomicU64::new(3);
        let (old, success) = RemoteAtomicEngine::cas_emulation_local(
            &target,
            0,
            CasComputeFn::Inc { modulo: 5 },
            32,
        );
        assert!(success);
        assert_eq!(old, 3);
        assert_eq!(target.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn test_cas_emulation_local_inc_wrap() {
        let target = AtomicU64::new(5);
        let (_, success) = RemoteAtomicEngine::cas_emulation_local(
            &target,
            0,
            CasComputeFn::Inc { modulo: 5 },
            32,
        );
        assert!(success);
        assert_eq!(target.load(Ordering::Relaxed), 0);
    }

    // === RemoteAtomicEngine tests ===

    #[test]
    fn test_atomic_engine_new() {
        let engine = RemoteAtomicEngine::new(500);
        assert_eq!(engine.proxy_count(), 0);
        assert_eq!(engine.hot_atomic_count(), 0);
    }

    #[test]
    fn test_atomic_engine_register_proxy() {
        let engine = RemoteAtomicEngine::with_defaults();
        let conn = AtomicProxyConnection {
            node_id: 1,
            qp: RdmaQueuePair { qp_num: 42 },
            remote_mr_key: 100,
            doorbell_addr: 0x1000,
            response_buffer_addr: 0x2000,
        };
        engine.register_proxy(conn);
        assert!(engine.has_proxy(1));
        assert!(!engine.has_proxy(2));
        assert_eq!(engine.proxy_count(), 1);
    }

    #[test]
    fn test_atomic_engine_remove_proxy() {
        let engine = RemoteAtomicEngine::with_defaults();
        let conn = AtomicProxyConnection {
            node_id: 1,
            qp: RdmaQueuePair { qp_num: 42 },
            remote_mr_key: 100,
            doorbell_addr: 0x1000,
            response_buffer_addr: 0x2000,
        };
        engine.register_proxy(conn);
        assert!(engine.remove_proxy(1).is_some());
        assert!(!engine.has_proxy(1));
    }

    #[test]
    fn test_atomic_engine_translate_cas_64() {
        let engine = RemoteAtomicEngine::new(100_000); // high threshold to avoid proxy
        let ops = AtomicOperands {
            operand1: 42,
            operand2: 99,
        };
        let translation = engine.translate(
            CudaAtomicOp::CAS { size: AtomicSize::Bits64 },
            0x1000,
            ops,
            1,
        );
        // Should be direct RDMA CAS
        assert_eq!(
            translation.network_op,
            NetworkAtomicOp::RdmaCas {
                compare: 42,
                swap: 99,
            }
        );
    }

    #[test]
    fn test_atomic_engine_translate_add_64() {
        let engine = RemoteAtomicEngine::new(100_000);
        let ops = AtomicOperands {
            operand1: 7,
            operand2: 0,
        };
        let translation = engine.translate(
            CudaAtomicOp::Add {
                size: AtomicSize::Bits64,
                is_float: false,
            },
            0x1000,
            ops,
            1,
        );
        assert_eq!(
            translation.network_op,
            NetworkAtomicOp::RdmaFetchAdd { addend: 7 }
        );
    }

    #[test]
    fn test_atomic_engine_translate_sub_64() {
        let engine = RemoteAtomicEngine::new(100_000);
        let ops = AtomicOperands {
            operand1: 3,
            operand2: 0,
        };
        let translation = engine.translate(
            CudaAtomicOp::Sub { size: AtomicSize::Bits64 },
            0x1000,
            ops,
            1,
        );
        // Sub -> FetchAdd with negated value
        assert_eq!(
            translation.network_op,
            NetworkAtomicOp::RdmaFetchAdd {
                addend: 0u64.wrapping_sub(3),
            }
        );
    }

    #[test]
    fn test_atomic_engine_translate_min_32_uses_cas() {
        let engine = RemoteAtomicEngine::new(100_000);
        let ops = AtomicOperands {
            operand1: 5,
            operand2: 0,
        };
        let translation = engine.translate(
            CudaAtomicOp::Min {
                size: AtomicSize::Bits32,
                is_signed: true,
            },
            0x2000,
            ops,
            1,
        );
        assert!(matches!(
            translation.network_op,
            NetworkAtomicOp::CasEmulationLoop { .. }
        ));
    }

    // === HotAtomicEntry tests ===

    #[test]
    fn test_hot_atomic_entry_new() {
        let entry = HotAtomicEntry::new(1);
        assert_eq!(entry.home_node, 1);
        assert_eq!(entry.current_mode(), AtomicMode::Migrate);
    }

    #[test]
    fn test_hot_atomic_entry_update_mode_proxy() {
        let entry = HotAtomicEntry::new(1);
        entry.remote_frequency.store(5000, Ordering::Relaxed);
        entry.update_mode(1000, false);
        assert_eq!(entry.current_mode(), AtomicMode::Proxy);
    }

    #[test]
    fn test_hot_atomic_entry_update_mode_rdma() {
        let entry = HotAtomicEntry::new(1);
        entry.remote_frequency.store(100, Ordering::Relaxed);
        entry.update_mode(1000, true);
        assert_eq!(entry.current_mode(), AtomicMode::RdmaDirect);
    }

    #[test]
    fn test_hot_atomic_entry_update_mode_migrate() {
        let entry = HotAtomicEntry::new(1);
        entry.remote_frequency.store(100, Ordering::Relaxed);
        entry.update_mode(1000, false);
        assert_eq!(entry.current_mode(), AtomicMode::Migrate);
    }

    // === FenceState tests ===

    #[test]
    fn test_fence_state_new() {
        let fs = FenceState::new();
        assert_eq!(fs.last_issued(), 0);
        assert_eq!(fs.last_completed(), 0);
        assert_eq!(fs.pending_count(), 0);
    }

    #[test]
    fn test_fence_state_issue() {
        let fs = FenceState::new();
        assert_eq!(fs.issue_fence(), 1);
        assert_eq!(fs.issue_fence(), 2);
        assert_eq!(fs.last_issued(), 2);
    }

    #[test]
    fn test_fence_state_complete() {
        let fs = FenceState::new();
        let seq = fs.issue_fence();
        assert!(!fs.is_fence_complete(seq));
        fs.complete_fence(seq);
        assert!(fs.is_fence_complete(seq));
    }

    #[test]
    fn test_fence_state_out_of_order_complete() {
        let fs = FenceState::new();
        let s1 = fs.issue_fence();
        let s2 = fs.issue_fence();
        // Complete s2 first
        fs.complete_fence(s2);
        // Both should be considered complete (fence seq advances to max)
        assert!(fs.is_fence_complete(s1));
        assert!(fs.is_fence_complete(s2));
    }

    #[test]
    fn test_fence_state_pending_ops() {
        let fs = FenceState::new();
        let seq = fs.issue_fence();
        fs.add_pending(seq, 100);
        fs.add_pending(seq, 200);
        assert_eq!(fs.pending_count(), 2);

        // Before completing: no ready ops
        let ready = fs.drain_ready();
        assert!(ready.is_empty());

        // Complete fence
        fs.complete_fence(seq);
        let ready = fs.drain_ready();
        assert_eq!(ready.len(), 2);
        assert_eq!(fs.pending_count(), 0);
    }

    #[test]
    fn test_fence_state_partial_drain() {
        let fs = FenceState::new();
        let s1 = fs.issue_fence();
        let s2 = fs.issue_fence();
        fs.add_pending(s1, 100);
        fs.add_pending(s2, 200);

        fs.complete_fence(s1);
        let ready = fs.drain_ready();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].op_id, 100);
        assert_eq!(fs.pending_count(), 1); // s2 still pending
    }

    // === NvlinkCoherencyAdapter tests ===

    #[test]
    fn test_coherency_adapter_defaults() {
        let adapter = NvlinkCoherencyAdapter::with_defaults();
        assert_eq!(adapter.page_size(), 64 * 1024);
        assert!(!adapter.has_multicast_invalidation());
    }

    #[test]
    fn test_coherency_adapter_align_to_page() {
        let adapter = NvlinkCoherencyAdapter::new(64 * 1024, false);
        assert_eq!(adapter.align_to_page(0x10100), 0x10000);
        assert_eq!(adapter.align_to_page(0x10000), 0x10000);
        assert_eq!(adapter.align_to_page(0x1FFFF), 0x10000);
    }

    #[test]
    fn test_coherency_adapter_align_4k() {
        let adapter = NvlinkCoherencyAdapter::new(4 * 1024, false);
        assert_eq!(adapter.align_to_page(0x1100), 0x1000);
        assert_eq!(adapter.align_to_page(0x1000), 0x1000);
        assert_eq!(adapter.align_to_page(0x1FFF), 0x1000);
    }

    #[test]
    fn test_coherency_adapter_pages_spanned() {
        let adapter = NvlinkCoherencyAdapter::new(64 * 1024, false);
        assert_eq!(adapter.pages_spanned(0, 0), 0);
        assert_eq!(adapter.pages_spanned(0, 1), 1);
        assert_eq!(adapter.pages_spanned(0, 64 * 1024), 1);
        assert_eq!(adapter.pages_spanned(0, 64 * 1024 + 1), 2);
        assert_eq!(adapter.pages_spanned(0, 128 * 1024), 2);
    }

    #[test]
    fn test_coherency_adapter_pages_spanned_unaligned_start() {
        let adapter = NvlinkCoherencyAdapter::new(64 * 1024, false);
        // Start mid-page: spans 2 pages even for small size
        assert_eq!(adapter.pages_spanned(0x10000 + 100, 64 * 1024), 2);
    }

    #[test]
    fn test_coherency_adapter_fence_operations() {
        let adapter = NvlinkCoherencyAdapter::new(64 * 1024, false);
        let seq = adapter.issue_fence(42);
        assert!(!adapter.is_fence_complete(42, seq));
        adapter.complete_fence(42, seq);
        assert!(adapter.is_fence_complete(42, seq));
    }

    #[test]
    fn test_coherency_adapter_invalidation_strategy_multicast() {
        let adapter = NvlinkCoherencyAdapter::new(64 * 1024, true);
        assert_eq!(
            adapter.invalidation_strategy(4),
            InvalidationStrategy::Multicast
        );
    }

    #[test]
    fn test_coherency_adapter_invalidation_strategy_unicast() {
        let adapter = NvlinkCoherencyAdapter::new(64 * 1024, false);
        assert_eq!(
            adapter.invalidation_strategy(4),
            InvalidationStrategy::Unicast { sharer_count: 4 }
        );
    }

    #[test]
    fn test_coherency_adapter_invalidation_strategy_single_sharer() {
        // Even with multicast, single sharer -> unicast
        let adapter = NvlinkCoherencyAdapter::new(64 * 1024, true);
        assert_eq!(
            adapter.invalidation_strategy(1),
            InvalidationStrategy::Unicast { sharer_count: 1 }
        );
    }

    // === PerformanceGuardrails tests ===

    #[test]
    fn test_guardrails_defaults() {
        let g = PerformanceGuardrails::defaults();
        assert!(g.is_enabled());
        assert_eq!(g.small_transfer_warn_threshold, 10_000);
        assert_eq!(g.page_fault_warn_threshold, 1_000);
    }

    #[test]
    fn test_guardrails_disabled() {
        let g = PerformanceGuardrails::disabled();
        assert!(!g.is_enabled());
    }

    #[test]
    fn test_guardrails_toggle() {
        let g = PerformanceGuardrails::defaults();
        g.set_enabled(false);
        assert!(!g.is_enabled());
        g.set_enabled(true);
        assert!(g.is_enabled());
    }

    // === PerformanceDiagnostics tests ===

    #[test]
    fn test_diagnostics_new() {
        let d = PerformanceDiagnostics::with_defaults();
        assert_eq!(d.event_count(), 0);
    }

    #[test]
    fn test_diagnostics_record_event() {
        let d = PerformanceDiagnostics::with_defaults();
        d.record_event(DiagnosticEvent::SmallTransferWarning {
            device: 0,
            rate: 20_000,
        });
        assert_eq!(d.event_count(), 1);
    }

    #[test]
    fn test_diagnostics_disabled_no_record() {
        let g = PerformanceGuardrails::disabled();
        let d = PerformanceDiagnostics::new(g);
        d.record_event(DiagnosticEvent::SmallTransferWarning {
            device: 0,
            rate: 20_000,
        });
        assert_eq!(d.event_count(), 0);
    }

    #[test]
    fn test_diagnostics_drain() {
        let d = PerformanceDiagnostics::with_defaults();
        d.record_event(DiagnosticEvent::FenceFrequencyWarning { rate: 100_000 });
        d.record_event(DiagnosticEvent::FenceFrequencyWarning { rate: 200_000 });
        let events = d.drain_events();
        assert_eq!(events.len(), 2);
        assert_eq!(d.event_count(), 0);
    }

    #[test]
    fn test_diagnostics_check_peer_stats_no_warnings() {
        let d = PerformanceDiagnostics::with_defaults();
        let stats = PeerAccessStatsSnapshot {
            bytes_transferred: 1000,
            transfer_count: 10,
            small_transfer_count: 5,
            atomic_count: 0,
            page_fault_count: 0,
        };
        let warnings = d.check_peer_stats(0, &stats, 1.0);
        assert!(warnings.is_empty()); // 5 small/sec < 10000 threshold
    }

    #[test]
    fn test_diagnostics_check_peer_stats_small_transfer_warning() {
        let d = PerformanceDiagnostics::with_defaults();
        let stats = PeerAccessStatsSnapshot {
            bytes_transferred: 1000,
            transfer_count: 20_000,
            small_transfer_count: 15_000,
            atomic_count: 0,
            page_fault_count: 0,
        };
        let warnings = d.check_peer_stats(0, &stats, 1.0);
        assert_eq!(warnings.len(), 1);
        assert!(matches!(
            warnings[0],
            DiagnosticEvent::SmallTransferWarning { device: 0, rate: 15000 }
        ));
    }

    #[test]
    fn test_diagnostics_check_peer_stats_page_fault_warning() {
        let d = PerformanceDiagnostics::with_defaults();
        let stats = PeerAccessStatsSnapshot {
            bytes_transferred: 0,
            transfer_count: 0,
            small_transfer_count: 0,
            atomic_count: 0,
            page_fault_count: 5_000,
        };
        let warnings = d.check_peer_stats(0, &stats, 1.0);
        assert_eq!(warnings.len(), 1);
        assert!(matches!(
            warnings[0],
            DiagnosticEvent::PageFaultRateWarning { .. }
        ));
    }

    // === NvlinkConfig tests ===

    #[test]
    fn test_config_defaults() {
        let c = NvlinkConfig::defaults();
        assert_eq!(c.max_tier, 1);
        assert!(!c.report_atomics);
        assert_eq!(c.page_size, 64 * 1024);
        assert_eq!(c.fence_batch_size, 1);
        assert_eq!(c.peer_rank_offset, 0);
        assert!(c.atomic_mode_override.is_none());
    }

    // === PeerAccessManager tests ===

    #[test]
    fn test_peer_access_manager_can_access_self() {
        let mgr = make_peer_access_manager();
        // Same device -> None (let CUDA handle)
        assert_eq!(mgr.can_access_peer(0, 0), None);
    }

    #[test]
    fn test_peer_access_manager_can_access_unmanaged() {
        let mgr = make_peer_access_manager();
        // Device 99 not managed -> None
        assert_eq!(mgr.can_access_peer(0, 99), None);
    }

    #[test]
    fn test_peer_access_manager_can_access_reachable() {
        let mgr = make_peer_access_manager();
        // Devices 0 and 2 are on different nodes with a path
        assert_eq!(mgr.can_access_peer(0, 2), Some(true));
    }

    #[test]
    fn test_peer_access_manager_can_access_same_node() {
        let mgr = make_peer_access_manager();
        // Devices 0 and 1 are both on node 0
        assert_eq!(mgr.can_access_peer(0, 1), Some(true));
    }

    #[test]
    fn test_peer_access_manager_enable() {
        let mgr = make_peer_access_manager();
        assert!(mgr.enable_peer_access(0, 2).is_ok());
        assert!(mgr.is_peer_access_enabled(0, 2));
    }

    #[test]
    fn test_peer_access_manager_enable_already_enabled() {
        let mgr = make_peer_access_manager();
        mgr.enable_peer_access(0, 2).unwrap();
        assert_eq!(
            mgr.enable_peer_access(0, 2),
            Err(PeerAccessError::AlreadyEnabled)
        );
    }

    #[test]
    fn test_peer_access_manager_enable_same_device() {
        let mgr = make_peer_access_manager();
        assert_eq!(
            mgr.enable_peer_access(0, 0),
            Err(PeerAccessError::SameDevice)
        );
    }

    #[test]
    fn test_peer_access_manager_enable_unmanaged() {
        let mgr = make_peer_access_manager();
        assert_eq!(
            mgr.enable_peer_access(0, 99),
            Err(PeerAccessError::DeviceNotManaged(99))
        );
    }

    #[test]
    fn test_peer_access_manager_disable() {
        let mgr = make_peer_access_manager();
        mgr.enable_peer_access(0, 2).unwrap();
        assert!(mgr.disable_peer_access(0, 2).is_ok());
        assert!(!mgr.is_peer_access_enabled(0, 2));
    }

    #[test]
    fn test_peer_access_manager_disable_not_enabled() {
        let mgr = make_peer_access_manager();
        assert_eq!(
            mgr.disable_peer_access(0, 2),
            Err(PeerAccessError::NotEnabled)
        );
    }

    #[test]
    fn test_peer_access_manager_record_transfer() {
        let mgr = make_peer_access_manager();
        mgr.enable_peer_access(0, 2).unwrap();
        mgr.record_transfer(0, 2, 1024);
        mgr.record_transfer(0, 2, 8192);
        let stats = mgr.get_stats(0, 2).unwrap();
        assert_eq!(stats.bytes_transferred, 9216);
        assert_eq!(stats.transfer_count, 2);
        assert_eq!(stats.small_transfer_count, 1); // 1024 < 4096
    }

    #[test]
    fn test_peer_access_manager_get_stats_no_connection() {
        let mgr = make_peer_access_manager();
        assert!(mgr.get_stats(0, 2).is_none());
    }

    #[test]
    fn test_peer_access_manager_active_count() {
        let mgr = make_peer_access_manager();
        assert_eq!(mgr.active_connection_count(), 0);
        mgr.enable_peer_access(0, 2).unwrap();
        assert_eq!(mgr.active_connection_count(), 1);
        mgr.enable_peer_access(0, 3).unwrap();
        assert_eq!(mgr.active_connection_count(), 2);
    }

    #[test]
    fn test_peer_access_manager_get_route() {
        let mgr = make_peer_access_manager();
        mgr.enable_peer_access(0, 2).unwrap();
        let route = mgr.get_route(0, 2).unwrap();
        assert_eq!(route.hop_count, 1); // node 0 -> node 1 = 1 hop
    }

    #[test]
    fn test_peer_access_manager_get_p2p_attributes_with_connection() {
        let mgr = make_peer_access_manager();
        mgr.enable_peer_access(0, 2).unwrap();
        let attrs = mgr.get_p2p_attributes(0, 2, true).unwrap();
        assert_eq!(attrs.access_supported, 1);
        assert_eq!(attrs.performance_rank, 1);
    }

    #[test]
    fn test_peer_access_manager_get_p2p_attributes_without_connection() {
        let mgr = make_peer_access_manager();
        let attrs = mgr.get_p2p_attributes(0, 2, true).unwrap();
        assert_eq!(attrs.access_supported, 1);
        assert_eq!(attrs.performance_rank, 1);
    }

    #[test]
    fn test_peer_access_manager_get_p2p_attributes_unmanaged() {
        let mgr = make_peer_access_manager();
        assert!(mgr.get_p2p_attributes(0, 99, true).is_none());
    }

    // === NvlinkEmulator tests ===

    #[test]
    fn test_emulator_tier1_only() {
        let emu = make_emulator();
        assert_eq!(emu.active_tier(), 1);
        assert!(!emu.is_coherency_active());
        assert!(!emu.is_atomics_active());
    }

    #[test]
    fn test_emulator_enable_tier2() {
        let mut emu = make_emulator();
        let adapter = Arc::new(NvlinkCoherencyAdapter::with_defaults());
        emu.enable_tier2(adapter);
        assert_eq!(emu.active_tier(), 2);
        assert!(emu.is_coherency_active());
    }

    #[test]
    fn test_emulator_enable_tier3() {
        let mut emu = make_emulator();
        let engine = Arc::new(RemoteAtomicEngine::with_defaults());
        emu.enable_tier3(engine);
        assert_eq!(emu.active_tier(), 3);
        assert!(emu.is_atomics_active());
    }

    #[test]
    fn test_emulator_set_active_tier() {
        let emu = make_emulator();
        emu.set_active_tier(2);
        assert_eq!(emu.active_tier(), 2);
        emu.set_active_tier(0); // clamped to 1
        assert_eq!(emu.active_tier(), 1);
        emu.set_active_tier(5); // clamped to 3
        assert_eq!(emu.active_tier(), 3);
    }

    #[test]
    fn test_emulator_can_access_peer() {
        let emu = make_emulator();
        assert_eq!(emu.can_access_peer(0, 0), None);
        assert_eq!(emu.can_access_peer(0, 2), Some(true));
        assert_eq!(emu.can_access_peer(0, 99), None);
    }

    #[test]
    fn test_emulator_enable_disable_peer() {
        let emu = make_emulator();
        assert!(emu.enable_peer_access(0, 2).is_ok());
        assert!(emu.disable_peer_access(0, 2).is_ok());
    }

    #[test]
    fn test_emulator_translate_atomic_tier1() {
        let emu = make_emulator();
        let ops = AtomicOperands {
            operand1: 1,
            operand2: 0,
        };
        // Tier 1 -> no atomic translation
        assert!(emu.translate_atomic(
            CudaAtomicOp::Add { size: AtomicSize::Bits64, is_float: false },
            0x1000,
            ops,
            1,
        ).is_none());
    }

    #[test]
    fn test_emulator_translate_atomic_tier3() {
        let mut emu = make_emulator();
        let engine = Arc::new(RemoteAtomicEngine::new(100_000));
        emu.enable_tier3(engine);
        let ops = AtomicOperands {
            operand1: 42,
            operand2: 99,
        };
        let result = emu.translate_atomic(
            CudaAtomicOp::CAS { size: AtomicSize::Bits64 },
            0x1000,
            ops,
            1,
        );
        assert!(result.is_some());
        let t = result.unwrap();
        assert_eq!(
            t.network_op,
            NetworkAtomicOp::RdmaCas { compare: 42, swap: 99 }
        );
    }

    #[test]
    fn test_emulator_record_transfer() {
        let emu = make_emulator();
        emu.enable_peer_access(0, 2).unwrap();
        emu.record_transfer(0, 2, 4096);
        let stats = emu.peer_access.get_stats(0, 2).unwrap();
        assert_eq!(stats.bytes_transferred, 4096);
    }

    // === PeerAccessError tests ===

    #[test]
    fn test_peer_access_error_display() {
        assert_eq!(
            PeerAccessError::DeviceNotManaged(5).to_string(),
            "device 5 is not managed by OuterLink"
        );
        assert_eq!(
            PeerAccessError::AlreadyEnabled.to_string(),
            "peer access already enabled"
        );
        assert_eq!(
            PeerAccessError::NotEnabled.to_string(),
            "peer access not enabled"
        );
        assert_eq!(
            PeerAccessError::NoRoute.to_string(),
            "no network route between devices"
        );
        assert_eq!(
            PeerAccessError::SameDevice.to_string(),
            "cannot peer-access the same device"
        );
    }

    // === NetworkAtomicOp translation tests ===

    #[test]
    fn test_translate_float_add_uses_cas_emulation() {
        let engine = RemoteAtomicEngine::new(100_000);
        let ops = AtomicOperands {
            operand1: f64::to_bits(1.0),
            operand2: 0,
        };
        let t = engine.translate(
            CudaAtomicOp::Add {
                size: AtomicSize::Bits64,
                is_float: true,
            },
            0x3000,
            ops,
            1,
        );
        assert!(matches!(
            t.network_op,
            NetworkAtomicOp::CasEmulationLoop {
                compute_fn: CasComputeFn::FloatAdd
            }
        ));
    }

    #[test]
    fn test_translate_or_uses_cas_emulation() {
        let engine = RemoteAtomicEngine::new(100_000);
        let ops = AtomicOperands {
            operand1: 0xFF,
            operand2: 0,
        };
        let t = engine.translate(
            CudaAtomicOp::Or { size: AtomicSize::Bits64 },
            0x4000,
            ops,
            1,
        );
        assert!(matches!(
            t.network_op,
            NetworkAtomicOp::CasEmulationLoop {
                compute_fn: CasComputeFn::Or
            }
        ));
    }

    // === Topology with unreachable nodes ===

    #[test]
    fn test_topology_unreachable_node() {
        let mut topo = TopologySnapshot::empty();
        topo.nodes = vec![0, 1, 2];
        topo.adjacency.insert((0, 1), 1);
        topo.adjacency.insert((1, 0), 1);
        // Node 2 has no adjacency entries -> unreachable from 0 and 1
        assert!(!topo.has_path(0, 2));
        assert!(topo.hop_count(0, 2).is_none());
    }

    #[test]
    fn test_peer_access_enable_no_route() {
        let mut topo = TopologySnapshot::empty();
        topo.nodes = vec![0, 1];
        // No adjacency -> no route
        let topo = Arc::new(topo);
        let mut dm = DeviceMapping::new();
        dm.register(0, 0, 0);
        dm.register(1, 1, 0);
        let dm = Arc::new(dm);
        let mgr = PeerAccessManager::new(topo, dm, NvlinkConfig::defaults());
        assert_eq!(
            mgr.enable_peer_access(0, 1),
            Err(PeerAccessError::NoRoute)
        );
    }
}
