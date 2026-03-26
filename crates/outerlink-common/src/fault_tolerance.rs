//! R15: Fault Tolerance and Erasure Coding
//!
//! Provides erasure coding (XOR and Reed-Solomon via trait), parity group
//! management, phi accrual failure detection, cluster membership with
//! generation-based fencing, recovery planning, and checkpoint management.
//!
//! Hardware-dependent functionality (ISA-L bindings, RDMA heartbeats, actual
//! network I/O) is represented as traits with software fallback implementations.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{Duration, Instant};

use dashmap::DashMap;

use crate::memory::types::NodeId;

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// Virtual page number (matches u64 used throughout the memory subsystem).
pub type Vpn = u64;

// ---------------------------------------------------------------------------
// 1. Erasure Coding Types
// ---------------------------------------------------------------------------

/// Erasure coding configuration for a parity group.
#[derive(Clone, Debug)]
pub struct ErasureConfig {
    /// Number of data fragments.
    pub k: u8,
    /// Number of parity fragments.
    pub m: u8,
    /// Coding scheme.
    pub scheme: CodingScheme,
}

/// The coding scheme used to compute parity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CodingScheme {
    /// Simple XOR parity (m must be 1).
    Xor,
    /// Reed-Solomon (via ISA-L or software fallback).
    ReedSolomon,
}

/// A parity group -- a set of pages protected together.
#[derive(Clone, Debug)]
pub struct ParityGroup {
    /// Unique group identifier (stored in PTE.parity_group_id).
    pub group_id: u32,
    /// Erasure coding configuration.
    pub config: ErasureConfig,
    /// VPNs of data pages in this group (ordered; index = fragment index).
    pub data_pages: Vec<Vpn>,
    /// Locations of parity fragments.
    pub parity_locations: Vec<ParityLocation>,
    /// Last time parity was verified/updated.
    pub last_parity_update: Instant,
    /// Protection policy.
    pub policy: ParityPolicy,
}

/// Where a parity fragment is stored.
#[derive(Clone, Debug)]
pub struct ParityLocation {
    pub node_id: NodeId,
    pub tier: ParityTier,
    pub address: u64,
    pub size: usize,
}

/// Storage tier for parity data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParityTier {
    /// Partner node's DRAM (fast recovery).
    RemoteDram,
    /// NVMe (persistent, slower recovery).
    Nvme,
    /// Both DRAM and NVMe (critical data).
    Both,
}

/// Parity update policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParityPolicy {
    /// Parity updated synchronously on every write.
    Synchronous,
    /// Parity updated in background batches.
    AsyncBatch { batch_interval_ms: u32 },
    /// Parity updated only at checkpoint boundaries.
    CheckpointOnly,
    /// No parity (transient data).
    None,
}

/// Data classification determines protection level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataClass {
    /// Model weights, embedding tables -- immutable after load.
    Critical,
    /// KV cache, activations -- can be recomputed.
    Recoverable,
    /// Temporary scratch buffers -- loss is acceptable.
    Transient,
    /// Optimizer state, gradients -- protected by checkpoints.
    Checkpoint,
}

// ---------------------------------------------------------------------------
// 2. Failure Detection Types
// ---------------------------------------------------------------------------

/// Node health state in the cluster membership.
#[derive(Clone, Debug, PartialEq)]
pub enum NodeState {
    /// Normal operation.
    Active,
    /// Suspected failure, not yet confirmed.
    Suspected { since: Instant, phi: f64 },
    /// Confirmed dead, fencing in progress.
    Fencing { generation: u64 },
    /// Fenced and removed from cluster.
    Dead { generation: u64, detected_at: Instant },
    /// Rejoining after recovery.
    Rejoining { generation: u64 },
}

/// Cluster membership with generation-based fencing.
#[derive(Clone, Debug)]
pub struct ClusterMembership {
    /// Monotonically increasing generation counter.
    pub generation: u64,
    /// Current coordinator node.
    pub coordinator: NodeId,
    /// All known nodes and their states.
    pub members: HashMap<NodeId, NodeState>,
    /// Quorum size (majority of active members).
    pub quorum_size: usize,
    /// NVMe lease path for 2-node tiebreaker.
    pub lease_path: Option<PathBuf>,
}

impl ClusterMembership {
    /// Create a new membership with a single coordinator.
    pub fn new(coordinator: NodeId) -> Self {
        let mut members = HashMap::new();
        members.insert(coordinator, NodeState::Active);
        Self {
            generation: 1,
            coordinator,
            members,
            quorum_size: 1,
            lease_path: None,
        }
    }

    /// Add a node to the cluster.
    pub fn add_node(&mut self, node_id: NodeId) {
        self.members.insert(node_id, NodeState::Active);
        self.recompute_quorum();
    }

    /// Count the number of active members.
    pub fn active_count(&self) -> usize {
        self.members
            .values()
            .filter(|s| matches!(s, NodeState::Active))
            .count()
    }

    /// Recompute the quorum size based on total known members.
    ///
    /// Uses total membership (including dead) for static quorum safety:
    /// a single surviving node cannot unilaterally claim quorum.
    /// Dead nodes must be explicitly removed via `remove_node()` to shrink
    /// the quorum requirement (administrative action after confirmed failure).
    fn recompute_quorum(&mut self) {
        let total = self.members.len();
        self.quorum_size = total / 2 + 1;
    }

    /// Remove a dead node from the membership entirely, shrinking the quorum.
    ///
    /// This is an administrative action: after confirming a node is permanently
    /// gone, call this to allow the remaining cluster to re-establish quorum.
    pub fn remove_node(&mut self, node_id: NodeId) -> Result<(), FaultToleranceError> {
        match self.members.get(&node_id) {
            Some(NodeState::Dead { .. }) => {
                self.members.remove(&node_id);
                self.recompute_quorum();
                Ok(())
            }
            Some(_) => Err(FaultToleranceError::InvalidStateTransition(
                format!("node {} is not Dead, cannot remove", node_id),
            )),
            None => Err(FaultToleranceError::UnknownNode(node_id)),
        }
    }

    /// Fence a node: mark it as Fencing and bump generation.
    pub fn fence_node(&mut self, node_id: NodeId) -> Result<u64, FaultToleranceError> {
        if !self.members.contains_key(&node_id) {
            return Err(FaultToleranceError::UnknownNode(node_id));
        }
        self.generation += 1;
        self.members
            .insert(node_id, NodeState::Fencing { generation: self.generation });
        Ok(self.generation)
    }

    /// Mark a fencing node as dead.
    pub fn mark_dead(&mut self, node_id: NodeId) -> Result<(), FaultToleranceError> {
        match self.members.get(&node_id) {
            Some(NodeState::Fencing { generation }) => {
                let gen = *generation;
                self.members.insert(
                    node_id,
                    NodeState::Dead {
                        generation: gen,
                        detected_at: Instant::now(),
                    },
                );
                self.recompute_quorum();
                Ok(())
            }
            _ => Err(FaultToleranceError::InvalidStateTransition(
                format!("node {} is not in Fencing state", node_id),
            )),
        }
    }

    /// Check if the cluster has quorum.
    pub fn has_quorum(&self) -> bool {
        self.active_count() >= self.quorum_size
    }
}

/// Heartbeat payload (compact, fits in single RDMA UD packet).
#[derive(Debug, Clone)]
pub struct HeartbeatPayload {
    pub node_id: u64,
    pub sequence: u64,
    pub timestamp_ns: u64,
    pub generation: u64,
    /// Bitmap: 1 bit per GPU (healthy/unhealthy).
    pub gpu_health_flags: u64,
    /// Memory pressure 0-100.
    pub memory_pressure_pct: u8,
    /// PTP clock offset from R26.
    pub ptp_offset_ns: i64,
    /// Number of active recoveries on this node.
    pub active_recoveries: u8,
}

// ---------------------------------------------------------------------------
// 3. Phi Accrual Failure Detector
// ---------------------------------------------------------------------------

/// Phi accrual failure detector state per monitored node.
///
/// Implements the phi accrual algorithm: the "suspicion level" phi grows
/// continuously based on how long it has been since the last heartbeat
/// relative to the statistical distribution of past inter-arrival times.
pub struct PhiAccrualDetector {
    /// Sliding window of inter-arrival times (milliseconds).
    arrival_window: VecDeque<f64>,
    /// Maximum window size.
    window_size: usize,
    /// Last heartbeat received timestamp.
    last_heartbeat: Option<Instant>,
    /// Computed mean of inter-arrival times.
    mean: f64,
    /// Computed variance of inter-arrival times.
    variance: f64,
}

impl PhiAccrualDetector {
    /// Create a new detector with the given window size.
    pub fn new(window_size: usize) -> Self {
        Self {
            arrival_window: VecDeque::with_capacity(window_size),
            window_size,
            last_heartbeat: None,
            mean: 0.0,
            variance: 0.0,
        }
    }

    /// Record a heartbeat arrival.
    pub fn record_heartbeat(&mut self, now: Instant) {
        if let Some(last) = self.last_heartbeat {
            let interval = now.duration_since(last).as_secs_f64() * 1000.0;
            if self.arrival_window.len() >= self.window_size {
                self.arrival_window.pop_front();
            }
            self.arrival_window.push_back(interval);
            self.recompute_stats();
        }
        self.last_heartbeat = Some(now);
    }

    /// Compute the current phi value.
    ///
    /// Phi represents the suspicion level. Higher values mean higher
    /// probability that the node has failed. Typical threshold: phi >= 6.0
    /// corresponds to ~99.97% probability of failure.
    pub fn phi(&self, now: Instant) -> f64 {
        let last = match self.last_heartbeat {
            Some(t) => t,
            None => return 0.0, // No data yet, cannot suspect.
        };

        if self.arrival_window.is_empty() {
            return 0.0;
        }

        let elapsed_ms = now.duration_since(last).as_secs_f64() * 1000.0;

        // Phi = -log10(1 - CDF(elapsed))
        // Using normal distribution approximation:
        // CDF(x) = 0.5 * (1 + erf((x - mean) / (sqrt(2) * stddev)))
        let stddev = self.variance.sqrt().max(0.01); // avoid division by zero
        let z = (elapsed_ms - self.mean) / stddev;

        // Approximate phi using log10 of complementary CDF
        // For z > 0 (elapsed > mean): phi grows rapidly
        // phi = -log10(1 - Phi(z)) where Phi is the standard normal CDF
        let p = 0.5 * erfc(z / std::f64::consts::SQRT_2);
        if p < 1e-15 {
            15.0 // Cap at 15 to avoid infinity
        } else {
            -p.log10()
        }
    }

    /// Get the last heartbeat time.
    pub fn last_heartbeat(&self) -> Option<Instant> {
        self.last_heartbeat
    }

    /// Get the current mean inter-arrival time in milliseconds.
    pub fn mean_interval_ms(&self) -> f64 {
        self.mean
    }

    /// Recompute mean and variance from the window.
    fn recompute_stats(&mut self) {
        if self.arrival_window.is_empty() {
            self.mean = 0.0;
            self.variance = 0.0;
            return;
        }
        let n = self.arrival_window.len() as f64;
        let sum: f64 = self.arrival_window.iter().sum();
        self.mean = sum / n;

        let var_sum: f64 = self.arrival_window
            .iter()
            .map(|x| (x - self.mean).powi(2))
            .sum();
        self.variance = var_sum / n;
    }
}

/// Complementary error function approximation (Horner form).
/// Abramowitz and Stegun approximation 7.1.26.
fn erfc(x: f64) -> f64 {
    if x >= 0.0 {
        erfc_positive(x)
    } else {
        2.0 - erfc_positive(-x)
    }
}

fn erfc_positive(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736
                + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    poly * (-x * x).exp()
}

// ---------------------------------------------------------------------------
// 4. XOR Encoder (software, no ISA-L dependency)
// ---------------------------------------------------------------------------

/// XOR encoder for erasure coding with m=1.
///
/// Computes parity by XOR-ing all data buffers together.
/// In production, SIMD intrinsics (AVX2/AVX-512) would be used for
/// memory-bandwidth-limited throughput (~30 GB/s per core).
pub struct XorEncoder;

impl XorEncoder {
    /// XOR all `data` buffers into `parity`.
    ///
    /// # Panics
    /// Panics if data has fewer than 2 buffers or if lengths do not match.
    pub fn encode(data: &[&[u8]], parity: &mut [u8]) -> Result<(), FaultToleranceError> {
        if data.is_empty() {
            return Err(FaultToleranceError::InvalidConfig(
                "encode requires at least 1 data buffer".to_string(),
            ));
        }
        if !data.iter().all(|d| d.len() == parity.len()) {
            return Err(FaultToleranceError::InvalidConfig(
                "all data buffers and parity must have the same length".to_string(),
            ));
        }

        // Initialize parity with first data buffer.
        parity.copy_from_slice(data[0]);

        // XOR remaining buffers.
        for buf in &data[1..] {
            for (p, d) in parity.iter_mut().zip(buf.iter()) {
                *p ^= *d;
            }
        }
        Ok(())
    }

    /// Reconstruct a lost data fragment from surviving fragments and parity.
    ///
    /// `survivors` are the data fragments that are still available.
    /// `parity` is the XOR parity fragment.
    /// `output` receives the reconstructed data.
    pub fn reconstruct(
        survivors: &[&[u8]],
        parity: &[u8],
        output: &mut [u8],
    ) -> Result<(), FaultToleranceError> {
        if output.len() != parity.len() {
            return Err(FaultToleranceError::InvalidConfig(
                "output and parity must have the same length".to_string(),
            ));
        }
        if !survivors.iter().all(|s| s.len() == parity.len()) {
            return Err(FaultToleranceError::InvalidConfig(
                "all buffers must have the same length".to_string(),
            ));
        }

        // lost_data = parity XOR survivor_0 XOR survivor_1 XOR ...
        output.copy_from_slice(parity);
        for s in survivors {
            for (o, d) in output.iter_mut().zip(s.iter()) {
                *o ^= *d;
            }
        }
        Ok(())
    }

    /// Incremental parity update: new_parity = old_parity XOR old_data XOR new_data.
    pub fn incremental_update(
        parity: &mut [u8],
        old_data: &[u8],
        new_data: &[u8],
    ) -> Result<(), FaultToleranceError> {
        if parity.len() != old_data.len() || parity.len() != new_data.len() {
            return Err(FaultToleranceError::InvalidConfig(
                "all buffers must have the same length".to_string(),
            ));
        }
        for i in 0..parity.len() {
            parity[i] ^= old_data[i] ^ new_data[i];
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 5. ErasureEncoder trait (for Reed-Solomon and other schemes)
// ---------------------------------------------------------------------------

/// Trait for erasure coding implementations.
///
/// XorEncoder implements this for m=1. Reed-Solomon (via ISA-L) would
/// implement this for m>1.
pub trait ErasureEncoder: Send + Sync {
    /// Encode data fragments into parity fragments.
    ///
    /// `data` contains `k` data buffers of equal length.
    /// Returns `m` parity buffers.
    fn encode(&self, data: &[&[u8]], fragment_size: usize) -> Result<Vec<Vec<u8>>, FaultToleranceError>;

    /// Reconstruct missing fragments from available ones.
    ///
    /// `available` maps fragment index (0..k+m) to data.
    /// `missing_indices` are the fragment indices to reconstruct.
    /// Returns the reconstructed fragments in order of missing_indices.
    fn reconstruct(
        &self,
        available: &HashMap<u8, &[u8]>,
        missing_indices: &[u8],
        fragment_size: usize,
    ) -> Result<Vec<Vec<u8>>, FaultToleranceError>;

    /// Get the (k, m) configuration.
    fn config(&self) -> (u8, u8);
}

/// Software XOR erasure encoder (k data fragments, m=1 parity).
pub struct XorErasureEncoder {
    k: u8,
}

impl XorErasureEncoder {
    pub fn new(k: u8) -> Self {
        Self { k }
    }
}

impl ErasureEncoder for XorErasureEncoder {
    fn encode(&self, data: &[&[u8]], fragment_size: usize) -> Result<Vec<Vec<u8>>, FaultToleranceError> {
        if data.len() != self.k as usize {
            return Err(FaultToleranceError::InvalidConfig(
                format!("expected {} data fragments, got {}", self.k, data.len()),
            ));
        }
        if !data.iter().all(|d| d.len() == fragment_size) {
            return Err(FaultToleranceError::InvalidConfig(
                "all fragments must have the specified size".to_string(),
            ));
        }

        let mut parity = vec![0u8; fragment_size];
        XorEncoder::encode(data, &mut parity)?;
        Ok(vec![parity])
    }

    fn reconstruct(
        &self,
        available: &HashMap<u8, &[u8]>,
        missing_indices: &[u8],
        fragment_size: usize,
    ) -> Result<Vec<Vec<u8>>, FaultToleranceError> {
        if missing_indices.len() != 1 {
            return Err(FaultToleranceError::InvalidConfig(
                "XOR can only reconstruct exactly 1 missing fragment".to_string(),
            ));
        }
        if available.len() != self.k as usize {
            return Err(FaultToleranceError::InvalidConfig(
                format!(
                    "XOR reconstruction requires exactly {} available fragments, got {}",
                    self.k,
                    available.len()
                ),
            ));
        }

        // Collect all available fragment data (both data and parity)
        let bufs: Vec<&[u8]> = available.values().copied().collect();
        let mut output = vec![0u8; fragment_size];

        // XOR all available fragments together to reconstruct the missing one.
        output.copy_from_slice(bufs[0]);
        for buf in &bufs[1..] {
            for (o, d) in output.iter_mut().zip(buf.iter()) {
                *o ^= *d;
            }
        }
        Ok(vec![output])
    }

    fn config(&self) -> (u8, u8) {
        (self.k, 1)
    }
}

/// Placeholder for ISA-L Reed-Solomon encoder.
///
/// In production, this would use FFI bindings to Intel ISA-L's
/// `ec_encode_data()` and `ec_encode_data_update()` functions.
/// For now, it returns an error indicating hardware support is required.
pub struct ReedSolomonEncoder {
    k: u8,
    m: u8,
}

impl ReedSolomonEncoder {
    pub fn new(k: u8, m: u8) -> Self {
        Self { k, m }
    }
}

impl ErasureEncoder for ReedSolomonEncoder {
    fn encode(&self, _data: &[&[u8]], _fragment_size: usize) -> Result<Vec<Vec<u8>>, FaultToleranceError> {
        // TODO: requires ISA-L C library bindings
        Err(FaultToleranceError::HardwareRequired(
            "Reed-Solomon encoding requires ISA-L library".to_string(),
        ))
    }

    fn reconstruct(
        &self,
        _available: &HashMap<u8, &[u8]>,
        _missing_indices: &[u8],
        _fragment_size: usize,
    ) -> Result<Vec<Vec<u8>>, FaultToleranceError> {
        // TODO: requires ISA-L C library bindings
        Err(FaultToleranceError::HardwareRequired(
            "Reed-Solomon reconstruction requires ISA-L library".to_string(),
        ))
    }

    fn config(&self) -> (u8, u8) {
        (self.k, self.m)
    }
}

// ---------------------------------------------------------------------------
// 6. Recovery Pipeline Types
// ---------------------------------------------------------------------------

/// Recovery plan generated after failure detection.
#[derive(Debug)]
pub struct RecoveryPlan {
    pub failed_node: NodeId,
    pub generation: u64,
    /// Pages to reconstruct, ordered by priority.
    pub reconstruction_queue: Vec<ReconstructionTask>,
    /// Target node for reconstructed data.
    pub target_node: NodeId,
    /// Estimated total reconstruction time.
    pub estimated_time: Duration,
}

/// A single page reconstruction task.
#[derive(Debug)]
pub struct ReconstructionTask {
    pub vpn: Vpn,
    pub parity_group_id: u32,
    pub method: ReconstructionMethod,
    pub priority: ReconstructionPriority,
    pub data_class: DataClass,
}

/// How to reconstruct a lost page.
#[derive(Debug)]
pub enum ReconstructionMethod {
    /// Reconstruct from XOR parity (single failure).
    XorReconstruct {
        surviving_pages: Vec<(NodeId, Vpn)>,
        parity_location: ParityLocation,
    },
    /// Reconstruct from RS parity (multi-failure capable).
    RsReconstruct {
        /// (fragment_idx, node, address).
        available_fragments: Vec<(u8, NodeId, u64)>,
        config: ErasureConfig,
    },
    /// Restore from checkpoint.
    CheckpointRestore {
        checkpoint_id: u64,
        checkpoint_location: ParityLocation,
    },
    /// Page is a dedup reference -- just update pointer.
    DedupRelink {
        canonical_vpn: Vpn,
        canonical_node: NodeId,
    },
}

/// Reconstruction priority (lower value = higher priority).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReconstructionPriority {
    /// Currently accessed by a CUDA kernel (stalling GPU).
    Immediate = 0,
    /// Hot page (recent access within last 100ms).
    Hot = 1,
    /// Warm page (accessed within last 10s).
    Warm = 2,
    /// Cold page (not recently accessed).
    Cold = 3,
}

/// Result of a fault encountered during recovery.
#[derive(Debug)]
pub enum FaultResolution {
    /// Page is being reconstructed; caller should wait.
    WaitForReconstruction { vpn: Vpn },
    /// Page was promoted to immediate priority.
    PromotedToImmediate { vpn: Vpn },
    /// Page is transient; return error to application.
    TransientLoss { vpn: Vpn },
}

// ---------------------------------------------------------------------------
// 7. Checkpoint Types
// ---------------------------------------------------------------------------

/// In-memory checkpoint following Gemini architecture.
#[derive(Debug)]
pub struct Checkpoint {
    pub id: u64,
    pub created_at: Instant,
    /// Which training step this checkpoint represents.
    pub training_step: u64,
    /// Per-shard checkpoint data.
    pub shards: Vec<CheckpointShard>,
    /// Delta chain from previous checkpoint (if incremental).
    pub delta_from: Option<u64>,
    /// Protection: where redundant copies live.
    pub redundancy: CheckpointRedundancy,
}

/// One shard of a checkpoint.
#[derive(Debug)]
pub struct CheckpointShard {
    pub shard_id: u32,
    /// Node that owns this shard.
    pub source_node: NodeId,
    /// Locations of checkpoint data.
    pub locations: Vec<CheckpointLocation>,
    /// Metadata for resharding.
    pub tensor_metadata: Vec<TensorMeta>,
    /// Total size in bytes.
    pub size: usize,
}

/// Tensor metadata for checkpoint resharding.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype_size: usize,
    pub offset: usize,
}

/// Where a checkpoint copy lives.
#[derive(Debug, Clone)]
pub struct CheckpointLocation {
    pub node_id: NodeId,
    pub tier: CheckpointTier,
    pub address: u64,
    pub size: usize,
}

/// Checkpoint storage tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointTier {
    /// Local DRAM (fastest, volatile).
    LocalDram,
    /// Remote DRAM via RDMA (fast, volatile, cross-node redundancy).
    RemoteDram,
    /// Local NVMe (persistent).
    LocalNvme,
    /// Remote NVMe (persistent, cross-node).
    RemoteNvme,
}

/// Checkpoint redundancy placement.
#[derive(Debug)]
pub struct CheckpointRedundancy {
    /// Primary copy location.
    pub primary: CheckpointLocation,
    /// Redundant copies.
    pub replicas: Vec<CheckpointLocation>,
}

/// Incremental delta between two checkpoints.
#[derive(Debug)]
pub struct CheckpointDelta {
    pub from_checkpoint: u64,
    pub to_checkpoint: u64,
    /// Changed regions.
    pub regions: Vec<DeltaRegion>,
    /// Compression ratio achieved.
    pub compression_ratio: f32,
}

/// A changed region within a checkpoint shard.
#[derive(Debug, Clone)]
pub struct DeltaRegion {
    pub shard_id: u32,
    pub offset: usize,
    pub data: Vec<u8>,
}

// ---------------------------------------------------------------------------
// 8. Parity Hook (R10 integration interface)
// ---------------------------------------------------------------------------

/// Erasure parity lifecycle trait for R15.
///
/// Called by the page table on page writes, allocations, frees, and
/// migrations to maintain parity consistency. Extends the base
/// `memory::traits::ParityHook` with R15-specific operations.
pub trait ErasureParityHook: Send + Sync {
    /// Called when a page is written. Updates parity based on policy.
    fn on_page_write(&self, vpn: Vpn) -> Result<(), FaultToleranceError>;

    /// Returns all page VPNs in a parity group.
    fn get_parity_pages(&self, group_id: u32) -> Vec<Vpn>;

    /// Called when a page is allocated -- assigns it to a parity group.
    /// Returns the assigned group_id.
    fn on_page_alloc(&self, vpn: Vpn, data_class: DataClass) -> Result<u32, FaultToleranceError>;

    /// Called when a page is freed -- removes it from its parity group.
    fn on_page_free(&self, vpn: Vpn, group_id: u32) -> Result<(), FaultToleranceError>;
}

/// Trait for failure detection.
pub trait FailureDetector: Send + Sync {
    /// Process a received heartbeat.
    fn on_heartbeat(&self, from: NodeId, payload: &HeartbeatPayload);

    /// Get current suspicion level for a node.
    fn phi(&self, node: NodeId) -> f64;

    /// Get current node state.
    fn node_state(&self, node: NodeId) -> Option<NodeState>;
}

/// Trait for recovery orchestration.
pub trait RecoveryOrchestrator: Send + Sync {
    /// Entry point: called when a node failure is confirmed.
    fn initiate_recovery(
        &self,
        failed_node: NodeId,
        generation: u64,
    ) -> Result<RecoveryPlan, FaultToleranceError>;

    /// Handle a fault during recovery.
    fn on_fault_during_recovery(&self, vpn: Vpn) -> Result<FaultResolution, FaultToleranceError>;
}

// ---------------------------------------------------------------------------
// 9. ErasureParityHook (the main parity manager)
// ---------------------------------------------------------------------------

/// Configuration for the fault tolerance subsystem.
#[derive(Debug, Clone)]
pub struct FaultToleranceConfig {
    /// Default erasure config for Critical data.
    pub critical_config: ErasureConfig,
    /// Default erasure config for Recoverable data.
    pub recoverable_config: ErasureConfig,
    /// Async parity batch interval (ms).
    pub async_batch_interval_ms: u32,
    /// Phi threshold for failure detection.
    pub phi_threshold: f64,
    /// Heartbeat interval.
    pub heartbeat_interval: Duration,
    /// Phi accrual detector window size.
    pub phi_window_size: usize,
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            critical_config: ErasureConfig {
                k: 3,
                m: 1,
                // XOR is the only working encoder. ReedSolomon requires ISA-L
                // hardware library and returns HardwareRequired at runtime.
                scheme: CodingScheme::Xor,
            },
            recoverable_config: ErasureConfig {
                k: 2,
                m: 1,
                scheme: CodingScheme::Xor,
            },
            async_batch_interval_ms: 5,
            phi_threshold: 6.0,
            heartbeat_interval: Duration::from_millis(100),
            phi_window_size: 1000,
        }
    }
}

/// Statistics for the fault tolerance subsystem.
#[derive(Debug, Default)]
pub struct FaultToleranceStats {
    /// Total parity groups created.
    pub groups_created: AtomicU64,
    /// Total parity updates performed.
    pub parity_updates: AtomicU64,
    /// Total pages protected by parity.
    pub pages_protected: AtomicU64,
    /// Total pages in dirty queue awaiting parity update.
    pub dirty_queue_size: AtomicU64,
    /// Total recoveries initiated.
    pub recoveries_initiated: AtomicU64,
    /// Total pages reconstructed.
    pub pages_reconstructed: AtomicU64,
}

impl FaultToleranceStats {
    pub fn snapshot(&self) -> FaultToleranceStatsSnapshot {
        FaultToleranceStatsSnapshot {
            groups_created: self.groups_created.load(Ordering::Relaxed),
            parity_updates: self.parity_updates.load(Ordering::Relaxed),
            pages_protected: self.pages_protected.load(Ordering::Relaxed),
            dirty_queue_size: self.dirty_queue_size.load(Ordering::Relaxed),
            recoveries_initiated: self.recoveries_initiated.load(Ordering::Relaxed),
            pages_reconstructed: self.pages_reconstructed.load(Ordering::Relaxed),
        }
    }
}

/// Immutable snapshot of fault tolerance statistics.
#[derive(Debug, Clone)]
pub struct FaultToleranceStatsSnapshot {
    pub groups_created: u64,
    pub parity_updates: u64,
    pub pages_protected: u64,
    pub dirty_queue_size: u64,
    pub recoveries_initiated: u64,
    pub pages_reconstructed: u64,
}

/// The main parity management engine.
///
/// Manages parity groups, processes dirty pages, and coordinates with
/// the R10 page table via the `ParityHook` trait.
pub struct ErasureParityManager {
    /// All active parity groups.
    groups: DashMap<u32, ParityGroup>,
    /// VPN -> group_id mapping (reverse index).
    vpn_to_group: DashMap<Vpn, u32>,
    /// Next group ID counter.
    next_group_id: AtomicU32,
    /// Dirty VPNs awaiting parity update.
    dirty_queue: DashMap<Vpn, ()>,
    /// Configuration.
    config: FaultToleranceConfig,
    /// Statistics.
    stats: FaultToleranceStats,
    /// Cluster membership (for parity group node distribution).
    membership: RwLock<ClusterMembership>,
}

impl ErasureParityManager {
    /// Create a new parity manager.
    pub fn new(config: FaultToleranceConfig, coordinator: NodeId) -> Self {
        Self {
            groups: DashMap::new(),
            vpn_to_group: DashMap::new(),
            next_group_id: AtomicU32::new(1),
            dirty_queue: DashMap::new(),
            config,
            stats: FaultToleranceStats::default(),
            membership: RwLock::new(ClusterMembership::new(coordinator)),
        }
    }

    /// Get a reference to the statistics.
    pub fn stats(&self) -> &FaultToleranceStats {
        &self.stats
    }

    /// Get a parity group by ID.
    pub fn get_group(&self, group_id: u32) -> Option<ParityGroup> {
        self.groups.get(&group_id).map(|g| g.clone())
    }

    /// Get the group ID for a VPN.
    pub fn get_group_for_vpn(&self, vpn: Vpn) -> Option<u32> {
        self.vpn_to_group.get(&vpn).map(|g| *g)
    }

    /// Get the number of dirty pages in the queue.
    pub fn dirty_queue_len(&self) -> usize {
        self.dirty_queue.len()
    }

    /// Process dirty queue: drain up to `max_batch` VPNs and return them
    /// grouped by parity group ID.
    ///
    /// In production, this is called by a background worker thread every
    /// `async_batch_interval_ms`.
    pub fn drain_dirty_batch(&self, max_batch: usize) -> HashMap<u32, Vec<Vpn>> {
        let mut result: HashMap<u32, Vec<Vpn>> = HashMap::new();
        let mut count = 0;

        // Collect keys first then remove, to avoid holding references during removal.
        let keys: Vec<Vpn> = self
            .dirty_queue
            .iter()
            .take(max_batch)
            .map(|entry| *entry.key())
            .collect();

        for vpn in keys {
            if count >= max_batch {
                break;
            }
            if self.dirty_queue.remove(&vpn).is_some() {
                if let Some(group_id) = self.vpn_to_group.get(&vpn) {
                    result.entry(*group_id).or_default().push(vpn);
                }
                count += 1;
            }
        }

        self.stats
            .dirty_queue_size
            .store(self.dirty_queue.len() as u64, Ordering::Relaxed);

        result
    }

    /// Find or create a parity group for the given VPN and data class.
    fn find_or_create_group(
        &self,
        _vpn: Vpn,
        data_class: DataClass,
        _owner_node: NodeId,
    ) -> Result<u32, FaultToleranceError> {
        let ec_config = match data_class {
            DataClass::Critical => &self.config.critical_config,
            DataClass::Recoverable => &self.config.recoverable_config,
            DataClass::Transient | DataClass::Checkpoint => {
                return Ok(0); // No parity for these classes
            }
        };

        // Search for an existing group with space.
        for entry in self.groups.iter() {
            let group = entry.value();
            if group.data_pages.len() < group.config.k as usize {
                // Check data class compatibility (determined by coding scheme match).
                if group.config.scheme == ec_config.scheme
                    && group.config.k == ec_config.k
                    && group.config.m == ec_config.m
                {
                    // Spread requirement: don't put two pages from the same node.
                    // For test simplicity, we skip this check if there's only one node.
                    let membership = self.membership.read().map_err(|_| {
                        FaultToleranceError::Internal("lock poisoned".to_string())
                    })?;
                    let multi_node = membership.members.len() > 1;
                    if !multi_node
                        || !group.data_pages.iter().any(|_| {
                            // In production, we'd check which node owns each VPN.
                            // For now, just allow it.
                            false
                        })
                    {
                        return Ok(group.group_id);
                    }
                }
            }
        }

        // No suitable group found -- create a new one.
        let group_id = self.next_group_id.fetch_add(1, Ordering::Relaxed) as u32;
        let policy = match data_class {
            DataClass::Critical => ParityPolicy::Synchronous,
            DataClass::Recoverable => ParityPolicy::AsyncBatch {
                batch_interval_ms: self.config.async_batch_interval_ms,
            },
            _ => ParityPolicy::None,
        };

        let group = ParityGroup {
            group_id,
            config: ec_config.clone(),
            data_pages: Vec::new(),
            parity_locations: Vec::new(),
            last_parity_update: Instant::now(),
            policy,
        };

        self.groups.insert(group_id, group);
        self.stats.groups_created.fetch_add(1, Ordering::Relaxed);

        Ok(group_id)
    }

    /// Access the cluster membership for updates.
    pub fn membership(&self) -> &RwLock<ClusterMembership> {
        &self.membership
    }
}

impl ErasureParityHook for ErasureParityManager {
    fn on_page_write(&self, vpn: Vpn) -> Result<(), FaultToleranceError> {
        let group_id = match self.vpn_to_group.get(&vpn) {
            Some(g) => *g,
            None => return Ok(()), // Not in a parity group.
        };

        if group_id == 0 {
            return Ok(()); // No parity for this page.
        }

        let group = self.groups.get(&group_id).ok_or_else(|| {
            FaultToleranceError::Internal(format!("parity group {} not found", group_id))
        })?;

        match group.policy {
            ParityPolicy::Synchronous => {
                // In production: immediately recompute parity.
                // Here we just record the update count.
                self.stats.parity_updates.fetch_add(1, Ordering::Relaxed);
            }
            ParityPolicy::AsyncBatch { .. } => {
                self.dirty_queue.insert(vpn, ());
                self.stats
                    .dirty_queue_size
                    .store(self.dirty_queue.len() as u64, Ordering::Relaxed);
            }
            ParityPolicy::CheckpointOnly | ParityPolicy::None => {
                // No parity update needed.
            }
        }
        Ok(())
    }

    fn get_parity_pages(&self, group_id: u32) -> Vec<Vpn> {
        self.groups
            .get(&group_id)
            .map(|g| g.data_pages.clone())
            .unwrap_or_default()
    }

    fn on_page_alloc(&self, vpn: Vpn, data_class: DataClass) -> Result<u32, FaultToleranceError> {
        if data_class == DataClass::Transient {
            self.vpn_to_group.insert(vpn, 0);
            return Ok(0);
        }

        let group_id = self.find_or_create_group(vpn, data_class, 0)?;

        if group_id == 0 {
            self.vpn_to_group.insert(vpn, 0);
            return Ok(0);
        }

        // Add VPN to the group.
        if let Some(mut group) = self.groups.get_mut(&group_id) {
            group.data_pages.push(vpn);
        }
        self.vpn_to_group.insert(vpn, group_id);
        self.stats.pages_protected.fetch_add(1, Ordering::Relaxed);

        Ok(group_id)
    }

    fn on_page_free(&self, vpn: Vpn, group_id: u32) -> Result<(), FaultToleranceError> {
        self.vpn_to_group.remove(&vpn);
        self.dirty_queue.remove(&vpn);

        if group_id == 0 {
            return Ok(());
        }

        if let Some(mut group) = self.groups.get_mut(&group_id) {
            group.data_pages.retain(|v| *v != vpn);
            // If group is now empty, remove it.
            if group.data_pages.is_empty() {
                drop(group);
                self.groups.remove(&group_id);
            }
        }

        // Only decrement if this page was actually protected (non-Transient).
        // Transient pages have group_id == 0 and were never counted.
        if group_id != 0 {
            self.stats.pages_protected.fetch_sub(1, Ordering::Relaxed);
        }
        Ok(())
    }
}

/// Bridge implementation: allows `ErasureParityManager` to be used as the
/// R10 memory system's `ParityHook` (defined in `memory::traits`).
impl crate::memory::traits::ParityHook for ErasureParityManager {
    fn notify_migration(&self, vpn: u64, _old_tier: crate::memory::types::TierId, _new_tier: crate::memory::types::TierId) {
        // When a page migrates between tiers, parity may need updating
        // since the parity data might reference the old location.
        // Treat migration as a write for parity consistency.
        let _ = self.on_page_write(vpn);
    }

    fn rebuild_page(&self, vpn: u64, parity_group_id: u32) -> Result<Vec<u8>, String> {
        // Collect all other data pages in the group and use XOR to reconstruct.
        let group = self
            .groups
            .get(&parity_group_id)
            .ok_or_else(|| format!("parity group {} not found", parity_group_id))?;

        // Verify the page is actually in this group.
        if !group.data_pages.contains(&vpn) {
            return Err(format!("vpn {} not in parity group {}", vpn, parity_group_id));
        }

        // In production: read the parity shard and surviving data shards,
        // then XOR-reconstruct the missing page. Here we return an error
        // indicating the data shards are needed from the caller.
        // TODO: requires actual page data access via PageTable
        Err(format!(
            "rebuild requires {} surviving data pages + parity from group {}",
            group.config.k - 1,
            parity_group_id
        ))
    }
}

// ---------------------------------------------------------------------------
// 10. MultiLayerDetector (failure detection)
// ---------------------------------------------------------------------------

/// Multi-layer failure detector combining phi accrual and RDMA events.
pub struct MultiLayerDetector {
    /// Per-node phi accrual detectors.
    phi_detectors: DashMap<NodeId, RwLock<PhiAccrualDetector>>,
    /// Cluster membership state.
    membership: RwLock<ClusterMembership>,
    /// Configuration.
    phi_threshold: f64,
    /// Interval at which heartbeats should be sent. Exposed via
    /// `heartbeat_interval()` so external schedulers can drive the
    /// heartbeat loop at the correct rate.
    heartbeat_interval: Duration,
    /// Per-node overridden states (e.g., from RDMA events).
    overridden_states: DashMap<NodeId, NodeState>,
}

impl MultiLayerDetector {
    pub fn new(config: &FaultToleranceConfig, coordinator: NodeId) -> Self {
        Self {
            phi_detectors: DashMap::new(),
            membership: RwLock::new(ClusterMembership::new(coordinator)),
            phi_threshold: config.phi_threshold,
            heartbeat_interval: config.heartbeat_interval,
            overridden_states: DashMap::new(),
        }
    }

    /// Register a node for monitoring.
    pub fn register_node(&self, node_id: NodeId) {
        let detector = PhiAccrualDetector::new(1000);
        self.phi_detectors.insert(node_id, RwLock::new(detector));
        if let Ok(mut m) = self.membership.write() {
            m.add_node(node_id);
        }
    }

    /// Get the phi threshold.
    pub fn phi_threshold(&self) -> f64 {
        self.phi_threshold
    }

    /// Get the configured heartbeat interval for external schedulers.
    ///
    /// The phi accrual detector requires heartbeats at this interval
    /// to maintain accurate failure detection. External code should
    /// call `on_heartbeat()` at this rate for each node.
    pub fn heartbeat_interval(&self) -> Duration {
        self.heartbeat_interval
    }

    /// Access the cluster membership.
    pub fn membership(&self) -> &RwLock<ClusterMembership> {
        &self.membership
    }

    /// Signal an RDMA connection failure for a node.
    /// This immediately moves the node to Suspected state.
    pub fn on_rdma_connection_failure(&self, node_id: NodeId) {
        self.overridden_states.insert(
            node_id,
            NodeState::Suspected {
                since: Instant::now(),
                phi: f64::MAX,
            },
        );
    }

    /// Initiate fencing for a suspected node.
    pub fn initiate_fencing(&self, node_id: NodeId) -> Result<u64, FaultToleranceError> {
        let mut membership = self.membership.write().map_err(|_| {
            FaultToleranceError::Internal("lock poisoned".to_string())
        })?;
        let gen = membership.fence_node(node_id)?;
        self.overridden_states
            .insert(node_id, NodeState::Fencing { generation: gen });
        Ok(gen)
    }
}

impl FailureDetector for MultiLayerDetector {
    fn on_heartbeat(&self, from: NodeId, _payload: &HeartbeatPayload) {
        if let Some(detector_lock) = self.phi_detectors.get(&from) {
            if let Ok(mut detector) = detector_lock.write() {
                detector.record_heartbeat(Instant::now());
            }
        }
        // Clear any overridden state if we receive a valid heartbeat.
        self.overridden_states.remove(&from);
    }

    fn phi(&self, node: NodeId) -> f64 {
        // Check overridden state first.
        if let Some(state) = self.overridden_states.get(&node) {
            if let NodeState::Suspected { phi, .. } = state.value() {
                return *phi;
            }
        }

        if let Some(detector_lock) = self.phi_detectors.get(&node) {
            if let Ok(detector) = detector_lock.read() {
                return detector.phi(Instant::now());
            }
        }
        0.0
    }

    fn node_state(&self, node: NodeId) -> Option<NodeState> {
        // Check overridden states first.
        if let Some(state) = self.overridden_states.get(&node) {
            return Some(state.value().clone());
        }

        // Check phi threshold.
        let phi_val = self.phi(node);
        if phi_val >= self.phi_threshold {
            return Some(NodeState::Suspected {
                since: Instant::now(),
                phi: phi_val,
            });
        }

        // Check membership.
        if let Ok(membership) = self.membership.read() {
            return membership.members.get(&node).cloned();
        }

        None
    }
}

// ---------------------------------------------------------------------------
// 11. Error Type
// ---------------------------------------------------------------------------

/// Errors for the fault tolerance subsystem.
#[derive(Debug, thiserror::Error)]
pub enum FaultToleranceError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("unknown node: {0}")]
    UnknownNode(NodeId),

    #[error("invalid state transition: {0}")]
    InvalidStateTransition(String),

    #[error("no parity group for VPN")]
    NoParityGroup,

    #[error("reconstruction failed: {0}")]
    ReconstructionFailed(String),

    #[error("hardware required: {0}")]
    HardwareRequired(String),

    #[error("no quorum")]
    NoQuorum,

    #[error("internal error: {0}")]
    Internal(String),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- XOR Encoder Tests --

    #[test]
    fn test_xor_encode_basic() {
        let a = vec![0xAA, 0xBB, 0xCC, 0xDD];
        let b = vec![0x11, 0x22, 0x33, 0x44];
        let c = vec![0xFF, 0x00, 0xFF, 0x00];
        let mut parity = vec![0u8; 4];

        XorEncoder::encode(&[&a, &b, &c], &mut parity).unwrap();

        // parity = a ^ b ^ c
        let expected: Vec<u8> = a
            .iter()
            .zip(b.iter())
            .zip(c.iter())
            .map(|((a, b), c)| a ^ b ^ c)
            .collect();
        assert_eq!(parity, expected);
    }

    #[test]
    fn test_xor_encode_single_buffer() {
        let a = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let mut parity = vec![0u8; 4];
        XorEncoder::encode(&[&a], &mut parity).unwrap();
        assert_eq!(parity, a);
    }

    #[test]
    fn test_xor_encode_mismatched_lengths_returns_error() {
        let a = vec![0xAA, 0xBB];
        let b = vec![0x11, 0x22, 0x33];
        let mut parity = vec![0u8; 2];
        let result = XorEncoder::encode(&[&a, &b], &mut parity);
        assert!(result.is_err());
    }

    #[test]
    fn test_xor_reconstruct_single_loss() {
        let a = vec![0xAA, 0xBB, 0xCC, 0xDD];
        let b = vec![0x11, 0x22, 0x33, 0x44];
        let c = vec![0xFF, 0x00, 0xFF, 0x00];
        let mut parity = vec![0u8; 4];
        XorEncoder::encode(&[&a, &b, &c], &mut parity).unwrap();

        // Lose fragment `b`, reconstruct from survivors (a, c) and parity.
        let mut recovered = vec![0u8; 4];
        XorEncoder::reconstruct(&[&a, &c], &parity, &mut recovered).unwrap();
        assert_eq!(recovered, b);
    }

    #[test]
    fn test_xor_incremental_update() {
        let a = vec![0xAA, 0xBB, 0xCC, 0xDD];
        let b = vec![0x11, 0x22, 0x33, 0x44];
        let mut parity = vec![0u8; 4];
        XorEncoder::encode(&[&a, &b], &mut parity).unwrap();

        // Update b -> b2.
        let b2 = vec![0x55, 0x66, 0x77, 0x88];
        XorEncoder::incremental_update(&mut parity, &b, &b2).unwrap();

        // Verify: parity should now match a ^ b2.
        let mut expected = vec![0u8; 4];
        XorEncoder::encode(&[&a, &b2], &mut expected).unwrap();
        assert_eq!(parity, expected);
    }

    #[test]
    fn test_xor_reconstruct_after_incremental_update() {
        let a = vec![0x01, 0x02, 0x03, 0x04];
        let b = vec![0x10, 0x20, 0x30, 0x40];
        let c = vec![0xF0, 0xE0, 0xD0, 0xC0];
        let mut parity = vec![0u8; 4];
        XorEncoder::encode(&[&a, &b, &c], &mut parity).unwrap();

        // Update c -> c2.
        let c2 = vec![0x0F, 0x0E, 0x0D, 0x0C];
        XorEncoder::incremental_update(&mut parity, &c, &c2).unwrap();

        // Lose a, reconstruct.
        let mut recovered = vec![0u8; 4];
        XorEncoder::reconstruct(&[&b, &c2], &parity, &mut recovered).unwrap();
        assert_eq!(recovered, a);
    }

    // -- XorErasureEncoder Tests --

    #[test]
    fn test_xor_erasure_encoder_encode_and_reconstruct() {
        let encoder = XorErasureEncoder::new(3);
        let a = vec![0xAA; 64];
        let b = vec![0xBB; 64];
        let c = vec![0xCC; 64];

        let parity_frags = encoder.encode(&[&a, &b, &c], 64).unwrap();
        assert_eq!(parity_frags.len(), 1);

        // Lose fragment 1 (b). Available: fragment 0 (a), fragment 2 (c), parity (index 3).
        let mut available = HashMap::new();
        available.insert(0u8, a.as_slice());
        available.insert(2u8, c.as_slice());
        available.insert(3u8, parity_frags[0].as_slice()); // parity is at index k=3

        let reconstructed = encoder.reconstruct(&available, &[1], 64).unwrap();
        assert_eq!(reconstructed.len(), 1);
        assert_eq!(reconstructed[0], b);
    }

    #[test]
    fn test_xor_erasure_encoder_wrong_k() {
        let encoder = XorErasureEncoder::new(3);
        let a = vec![0xAA; 64];
        let b = vec![0xBB; 64];
        let result = encoder.encode(&[&a, &b], 64);
        assert!(result.is_err());
    }

    // -- ReedSolomonEncoder Tests --

    #[test]
    fn test_reed_solomon_requires_hardware() {
        let encoder = ReedSolomonEncoder::new(4, 2);
        let data: Vec<Vec<u8>> = (0..4).map(|i| vec![i; 64]).collect();
        let refs: Vec<&[u8]> = data.iter().map(|d| d.as_slice()).collect();

        let result = encoder.encode(&refs, 64);
        assert!(matches!(result, Err(FaultToleranceError::HardwareRequired(_))));
    }

    // -- Phi Accrual Detector Tests --

    #[test]
    fn test_phi_zero_with_no_data() {
        let detector = PhiAccrualDetector::new(100);
        assert_eq!(detector.phi(Instant::now()), 0.0);
    }

    #[test]
    fn test_phi_low_with_regular_heartbeats() {
        let mut detector = PhiAccrualDetector::new(100);
        let start = Instant::now();

        // Simulate 20 regular heartbeats at 100ms intervals.
        for i in 0..20 {
            let t = start + Duration::from_millis(i * 100);
            detector.record_heartbeat(t);
        }

        // Check phi right after last heartbeat: should be very low.
        let last = start + Duration::from_millis(19 * 100);
        let phi = detector.phi(last + Duration::from_millis(10));
        assert!(phi < 2.0, "phi should be low shortly after heartbeat, got {}", phi);
    }

    #[test]
    fn test_phi_high_with_missed_heartbeats() {
        let mut detector = PhiAccrualDetector::new(100);
        let start = Instant::now();

        // 20 regular heartbeats at 100ms intervals.
        for i in 0..20 {
            let t = start + Duration::from_millis(i * 100);
            detector.record_heartbeat(t);
        }

        // No heartbeat for 2 seconds -- phi should be very high.
        let check_time = start + Duration::from_millis(19 * 100 + 2000);
        let phi = detector.phi(check_time);
        assert!(
            phi > 5.0,
            "phi should be high after long silence, got {}",
            phi
        );
    }

    #[test]
    fn test_phi_mean_interval_tracking() {
        let mut detector = PhiAccrualDetector::new(100);
        let start = Instant::now();

        for i in 0..10 {
            detector.record_heartbeat(start + Duration::from_millis(i * 50));
        }

        let mean = detector.mean_interval_ms();
        assert!(
            (mean - 50.0).abs() < 1.0,
            "mean should be ~50ms, got {}",
            mean
        );
    }

    // -- ClusterMembership Tests --

    #[test]
    fn test_cluster_membership_basic() {
        let mut m = ClusterMembership::new(0);
        assert_eq!(m.active_count(), 1);
        assert!(m.has_quorum());

        m.add_node(1);
        m.add_node(2);
        assert_eq!(m.active_count(), 3);
        assert_eq!(m.quorum_size, 2); // majority of 3
        assert!(m.has_quorum());
    }

    #[test]
    fn test_cluster_membership_fencing() {
        let mut m = ClusterMembership::new(0);
        m.add_node(1);
        m.add_node(2);

        let gen = m.fence_node(1).unwrap();
        assert_eq!(gen, 2); // generation bumped from 1 to 2
        assert!(matches!(m.members.get(&1), Some(NodeState::Fencing { .. })));
        assert_eq!(m.active_count(), 2); // Fencing node is not Active
    }

    #[test]
    fn test_cluster_membership_mark_dead() {
        let mut m = ClusterMembership::new(0);
        m.add_node(1);
        m.add_node(2);

        m.fence_node(1).unwrap();
        m.mark_dead(1).unwrap();
        assert!(matches!(m.members.get(&1), Some(NodeState::Dead { .. })));
    }

    #[test]
    fn test_cluster_membership_mark_dead_without_fencing_fails() {
        let mut m = ClusterMembership::new(0);
        m.add_node(1);

        let result = m.mark_dead(1);
        assert!(result.is_err());
    }

    #[test]
    fn test_fence_unknown_node_fails() {
        let mut m = ClusterMembership::new(0);
        let result = m.fence_node(99);
        assert!(matches!(result, Err(FaultToleranceError::UnknownNode(99))));
    }

    #[test]
    fn test_quorum_lost_after_failures() {
        let mut m = ClusterMembership::new(0);
        m.add_node(1);
        m.add_node(2);
        // 3 nodes, quorum = 2

        m.fence_node(1).unwrap();
        m.mark_dead(1).unwrap();
        // 2 active nodes, quorum still met (2 >= 2)
        assert!(m.has_quorum());

        m.fence_node(2).unwrap();
        m.mark_dead(2).unwrap();
        // 1 active node, quorum = 2 (still based on total members)
        // After recompute, quorum should be majority of total members.
        // With 3 members, quorum is 2, but only 1 active. No quorum.
        assert!(!m.has_quorum());
    }

    // -- ErasureParityManager Tests --

    #[test]
    fn test_parity_manager_alloc_transient_returns_zero() {
        let mgr = ErasureParityManager::new(FaultToleranceConfig::default(), 0);
        let group_id = mgr.on_page_alloc(100, DataClass::Transient).unwrap();
        assert_eq!(group_id, 0);
    }

    #[test]
    fn test_parity_manager_alloc_critical() {
        let mgr = ErasureParityManager::new(FaultToleranceConfig::default(), 0);
        let gid = mgr.on_page_alloc(100, DataClass::Critical).unwrap();
        assert!(gid > 0);

        // Verify VPN is tracked.
        assert_eq!(mgr.get_group_for_vpn(100), Some(gid));

        // Verify the group exists and contains the VPN.
        let group = mgr.get_group(gid).unwrap();
        assert!(group.data_pages.contains(&100));
    }

    #[test]
    fn test_parity_manager_alloc_reuses_group() {
        let mgr = ErasureParityManager::new(FaultToleranceConfig::default(), 0);
        let gid1 = mgr.on_page_alloc(100, DataClass::Recoverable).unwrap();
        let gid2 = mgr.on_page_alloc(101, DataClass::Recoverable).unwrap();
        // Should reuse the same group (it has capacity for k=3 pages).
        assert_eq!(gid1, gid2);

        let group = mgr.get_group(gid1).unwrap();
        assert_eq!(group.data_pages.len(), 2);
    }

    #[test]
    fn test_parity_manager_alloc_creates_new_group_when_full() {
        let mut config = FaultToleranceConfig::default();
        config.recoverable_config.k = 2;
        let mgr = ErasureParityManager::new(config, 0);

        let gid1 = mgr.on_page_alloc(100, DataClass::Recoverable).unwrap();
        let gid2 = mgr.on_page_alloc(101, DataClass::Recoverable).unwrap();
        assert_eq!(gid1, gid2); // Same group (k=2, has 2 slots).

        // Third alloc should create a new group.
        let gid3 = mgr.on_page_alloc(102, DataClass::Recoverable).unwrap();
        assert_ne!(gid1, gid3);
    }

    #[test]
    fn test_parity_manager_free_removes_vpn() {
        let mgr = ErasureParityManager::new(FaultToleranceConfig::default(), 0);
        let gid = mgr.on_page_alloc(100, DataClass::Recoverable).unwrap();
        assert!(mgr.get_group_for_vpn(100).is_some());

        mgr.on_page_free(100, gid).unwrap();
        assert!(mgr.get_group_for_vpn(100).is_none());
    }

    #[test]
    fn test_parity_manager_write_sync_policy() {
        let mgr = ErasureParityManager::new(FaultToleranceConfig::default(), 0);
        let gid = mgr.on_page_alloc(200, DataClass::Critical).unwrap();
        assert!(gid > 0);

        // Write should be synchronous (Critical -> Synchronous policy).
        mgr.on_page_write(200).unwrap();
        let snap = mgr.stats().snapshot();
        assert_eq!(snap.parity_updates, 1);
        assert_eq!(snap.dirty_queue_size, 0); // Not queued.
    }

    #[test]
    fn test_parity_manager_write_async_policy() {
        let mgr = ErasureParityManager::new(FaultToleranceConfig::default(), 0);
        let gid = mgr.on_page_alloc(300, DataClass::Recoverable).unwrap();
        assert!(gid > 0);

        // Write should be async (Recoverable -> AsyncBatch policy).
        mgr.on_page_write(300).unwrap();
        let snap = mgr.stats().snapshot();
        assert_eq!(snap.parity_updates, 0); // Not immediate.
        assert_eq!(snap.dirty_queue_size, 1); // Queued.
    }

    #[test]
    fn test_parity_manager_drain_dirty_batch() {
        let mgr = ErasureParityManager::new(FaultToleranceConfig::default(), 0);
        let gid = mgr.on_page_alloc(400, DataClass::Recoverable).unwrap();
        mgr.on_page_alloc(401, DataClass::Recoverable).unwrap();

        mgr.on_page_write(400).unwrap();
        mgr.on_page_write(401).unwrap();
        assert_eq!(mgr.dirty_queue_len(), 2);

        let batch = mgr.drain_dirty_batch(10);
        assert!(batch.contains_key(&gid));
        assert_eq!(batch[&gid].len(), 2);
        assert_eq!(mgr.dirty_queue_len(), 0);
    }

    #[test]
    fn test_parity_manager_stats_tracking() {
        let mgr = ErasureParityManager::new(FaultToleranceConfig::default(), 0);

        mgr.on_page_alloc(500, DataClass::Critical).unwrap();
        mgr.on_page_alloc(501, DataClass::Recoverable).unwrap();

        let snap = mgr.stats().snapshot();
        assert_eq!(snap.pages_protected, 2);
        assert!(snap.groups_created >= 1); // At least 1 group for each data class.
    }

    // -- MultiLayerDetector Tests --

    #[test]
    fn test_detector_register_and_heartbeat() {
        let config = FaultToleranceConfig::default();
        let detector = MultiLayerDetector::new(&config, 0);
        detector.register_node(1);

        let payload = HeartbeatPayload {
            node_id: 1,
            sequence: 1,
            timestamp_ns: 0,
            generation: 1,
            gpu_health_flags: 0xFF,
            memory_pressure_pct: 30,
            ptp_offset_ns: 0,
            active_recoveries: 0,
        };

        detector.on_heartbeat(1, &payload);

        let state = detector.node_state(1);
        assert!(matches!(state, Some(NodeState::Active)));
    }

    #[test]
    fn test_detector_rdma_failure_overrides_state() {
        let config = FaultToleranceConfig::default();
        let detector = MultiLayerDetector::new(&config, 0);
        detector.register_node(1);

        detector.on_rdma_connection_failure(1);

        let state = detector.node_state(1);
        assert!(matches!(state, Some(NodeState::Suspected { .. })));
    }

    #[test]
    fn test_detector_heartbeat_clears_override() {
        let config = FaultToleranceConfig::default();
        let detector = MultiLayerDetector::new(&config, 0);
        detector.register_node(1);

        detector.on_rdma_connection_failure(1);
        assert!(matches!(
            detector.node_state(1),
            Some(NodeState::Suspected { .. })
        ));

        let payload = HeartbeatPayload {
            node_id: 1,
            sequence: 2,
            timestamp_ns: 0,
            generation: 1,
            gpu_health_flags: 0xFF,
            memory_pressure_pct: 30,
            ptp_offset_ns: 0,
            active_recoveries: 0,
        };
        detector.on_heartbeat(1, &payload);

        let state = detector.node_state(1);
        assert!(matches!(state, Some(NodeState::Active)));
    }

    #[test]
    fn test_detector_fencing() {
        let config = FaultToleranceConfig::default();
        let detector = MultiLayerDetector::new(&config, 0);
        detector.register_node(1);

        let gen = detector.initiate_fencing(1).unwrap();
        assert!(gen >= 2);

        let state = detector.node_state(1);
        assert!(matches!(state, Some(NodeState::Fencing { .. })));
    }

    // -- Recovery Types Tests --

    #[test]
    fn test_reconstruction_priority_ordering() {
        assert!(ReconstructionPriority::Immediate < ReconstructionPriority::Hot);
        assert!(ReconstructionPriority::Hot < ReconstructionPriority::Warm);
        assert!(ReconstructionPriority::Warm < ReconstructionPriority::Cold);
    }

    // -- Config Tests --

    #[test]
    fn test_default_config() {
        let config = FaultToleranceConfig::default();
        assert_eq!(config.phi_threshold, 6.0);
        assert_eq!(config.heartbeat_interval, Duration::from_millis(100));
        assert_eq!(config.critical_config.k, 3);
        assert_eq!(config.critical_config.m, 1);
        assert_eq!(config.recoverable_config.scheme, CodingScheme::Xor);
    }

    // -- erfc approximation test --

    #[test]
    fn test_erfc_known_values() {
        // erfc(0) = 1.0
        assert!((erfc(0.0) - 1.0).abs() < 0.001);
        // erfc(large) -> 0
        assert!(erfc(5.0) < 0.001);
        // erfc(-large) -> 2
        assert!((erfc(-5.0) - 2.0).abs() < 0.001);
    }

    // -- Checkpoint type construction tests --

    #[test]
    fn test_checkpoint_types() {
        let ckpt = Checkpoint {
            id: 1,
            created_at: Instant::now(),
            training_step: 100,
            shards: vec![CheckpointShard {
                shard_id: 0,
                source_node: 0,
                locations: vec![CheckpointLocation {
                    node_id: 0,
                    tier: CheckpointTier::LocalDram,
                    address: 0x1000,
                    size: 65536,
                }],
                tensor_metadata: vec![TensorMeta {
                    name: "weight".to_string(),
                    shape: vec![1024, 1024],
                    dtype_size: 2,
                    offset: 0,
                }],
                size: 65536,
            }],
            delta_from: None,
            redundancy: CheckpointRedundancy {
                primary: CheckpointLocation {
                    node_id: 0,
                    tier: CheckpointTier::LocalDram,
                    address: 0x1000,
                    size: 65536,
                },
                replicas: vec![],
            },
        };
        assert_eq!(ckpt.training_step, 100);
        assert_eq!(ckpt.shards.len(), 1);
    }

    #[test]
    fn test_checkpoint_delta_computation() {
        let old_data = vec![0xAA; 1024];
        let new_data = {
            let mut d = old_data.clone();
            // Change bytes 100-199.
            for i in 100..200 {
                d[i] = 0xBB;
            }
            d
        };

        // Compute delta regions.
        let mut regions = Vec::new();
        let chunk_size = 64;
        for chunk_start in (0..old_data.len()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(old_data.len());
            if old_data[chunk_start..chunk_end] != new_data[chunk_start..chunk_end] {
                regions.push(DeltaRegion {
                    shard_id: 0,
                    offset: chunk_start,
                    data: new_data[chunk_start..chunk_end].to_vec(),
                });
            }
        }

        let delta = CheckpointDelta {
            from_checkpoint: 0,
            to_checkpoint: 1,
            regions,
            compression_ratio: 1.0,
        };

        // Changes are at bytes 100-199, spanning 64-byte chunks at offsets 64 and 128.
        assert!(delta.regions.len() >= 2);
        assert!(delta.regions.len() <= 3); // 64-127, 128-191, possibly 192-255
    }

    // -- Data class / parity policy mapping --

    #[test]
    fn test_data_class_to_parity_policy() {
        let mgr = ErasureParityManager::new(FaultToleranceConfig::default(), 0);

        // Critical -> Synchronous
        let gid = mgr.on_page_alloc(1000, DataClass::Critical).unwrap();
        let group = mgr.get_group(gid).unwrap();
        assert_eq!(group.policy, ParityPolicy::Synchronous);

        // Recoverable -> AsyncBatch
        let gid2 = mgr.on_page_alloc(2000, DataClass::Recoverable).unwrap();
        let group2 = mgr.get_group(gid2).unwrap();
        assert!(matches!(group2.policy, ParityPolicy::AsyncBatch { .. }));
    }

    // -- Heartbeat payload construction --

    #[test]
    fn test_heartbeat_payload() {
        let payload = HeartbeatPayload {
            node_id: 42,
            sequence: 100,
            timestamp_ns: 1_000_000_000,
            generation: 5,
            gpu_health_flags: 0b1111,
            memory_pressure_pct: 65,
            ptp_offset_ns: -1234,
            active_recoveries: 2,
        };
        assert_eq!(payload.node_id, 42);
        assert_eq!(payload.memory_pressure_pct, 65);
    }
}
