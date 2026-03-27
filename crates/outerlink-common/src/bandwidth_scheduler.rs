//! Bandwidth-aware scheduler types for OuterLink compute distribution.
//!
//! Provides EWMA-based adaptive bandwidth estimation, hysteresis-damped weight
//! adjustment, and transfer classification for the scheduler that distributes
//! work across networked GPU nodes.
//!
//! Key design choices (from R38 research):
//! - Three-layer oscillation damping: EWMA smoothing, hysteresis threshold, minimum hold time
//! - Separate handling for small (latency-sensitive) vs large (bandwidth-sensitive) transfers
//! - Deferred weight adjustment: changes apply to the NEXT kernel, not the running one

use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// TransferSize
// ---------------------------------------------------------------------------

/// Classifies a transfer by size to select the appropriate scheduling algorithm.
///
/// Small transfers are latency-sensitive (tree/scatter patterns), while large
/// transfers are bandwidth-sensitive (ring/pipeline patterns), following the
/// NCCL collective algorithm selection pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferSize {
    /// < 256 KB -- latency-sensitive, prefer tree/scatter topology.
    Small,
    /// 256 KB .. 16 MB -- balanced, use adaptive routing.
    Medium,
    /// > 16 MB -- bandwidth-sensitive, prefer ring/pipeline topology.
    Large,
}

impl TransferSize {
    /// Classify a byte count into a [`TransferSize`] category.
    pub fn from_bytes(n: u64) -> Self {
        const SMALL_THRESHOLD: u64 = 256 * 1024; // 256 KB
        const LARGE_THRESHOLD: u64 = 16 * 1024 * 1024; // 16 MB
        if n < SMALL_THRESHOLD {
            Self::Small
        } else if n > LARGE_THRESHOLD {
            Self::Large
        } else {
            Self::Medium
        }
    }

    /// Returns `true` when the transfer should be optimised for latency rather
    /// than raw throughput.
    pub fn is_latency_sensitive(&self) -> bool {
        matches!(self, Self::Small)
    }
}

// ---------------------------------------------------------------------------
// NodeBandwidth
// ---------------------------------------------------------------------------

/// Per-node bandwidth estimate maintained via exponential weighted moving average.
///
/// Each measurement updates `ewma_bps` using:
///   ewma_bps = alpha * sample + (1 - alpha) * ewma_bps
///
/// The `weight()` method is only meaningful when called via [`compute_weights`]
/// which normalises across the cluster.
#[derive(Debug, Clone)]
pub struct NodeBandwidth {
    /// Unique identifier for this node.
    pub node_id: String,
    /// Most recent raw measurement in bytes per second.
    pub measured_bps: u64,
    /// EWMA-smoothed estimate in bytes per second.
    pub ewma_bps: u64,
    /// Smoothing factor for EWMA (default 0.2, range 0.0..=1.0).
    pub ewma_alpha: f64,
    /// When the last measurement was recorded.
    pub last_measurement: Option<Instant>,
}

impl NodeBandwidth {
    /// Create a new node bandwidth tracker with the given alpha.
    pub fn new(node_id: impl Into<String>, ewma_alpha: f64) -> Self {
        Self {
            node_id: node_id.into(),
            measured_bps: 0,
            ewma_bps: 0,
            ewma_alpha,
            last_measurement: None,
        }
    }

    /// Record a new bandwidth sample and update the EWMA estimate.
    ///
    /// On the first sample the EWMA is seeded directly (no smoothing).
    pub fn update(&mut self, sample_bps: u64) {
        self.measured_bps = sample_bps;
        if self.last_measurement.is_none() {
            // First sample: seed directly.
            self.ewma_bps = sample_bps;
        } else {
            let alpha = self.ewma_alpha;
            let smoothed =
                alpha * sample_bps as f64 + (1.0 - alpha) * self.ewma_bps as f64;
            self.ewma_bps = smoothed.round() as u64;
        }
        self.last_measurement = Some(Instant::now());
    }
}

// ---------------------------------------------------------------------------
// HysteresisGuard
// ---------------------------------------------------------------------------

/// Prevents oscillation by gating weight changes behind a threshold percentage
/// and a minimum hold time.
///
/// A weight update is only allowed when:
/// 1. The relative change exceeds `threshold_percent`, AND
/// 2. At least `hold_time` has elapsed since the last accepted change.
#[derive(Debug, Clone)]
pub struct HysteresisGuard {
    /// Currently active weight (0.0..=1.0).
    pub current_weight: f64,
    /// When the weight was last changed.
    pub last_change: Option<Instant>,
    /// Minimum relative change (%) to allow an update.
    pub threshold_percent: f64,
    /// Minimum time between accepted changes.
    pub hold_time: Duration,
}

impl HysteresisGuard {
    /// Create a guard with default thresholds (15%, 2 seconds).
    pub fn new(initial_weight: f64) -> Self {
        Self::with_params(initial_weight, 15.0, Duration::from_secs(2))
    }

    /// Create a guard with custom thresholds.
    pub fn with_params(initial_weight: f64, threshold_percent: f64, hold_time: Duration) -> Self {
        Self {
            current_weight: initial_weight,
            last_change: None,
            threshold_percent,
            hold_time,
        }
    }

    /// Returns `true` if the proposed `new_weight` should be accepted.
    ///
    /// Rejects the update when:
    /// - The relative change from `current_weight` is below `threshold_percent`, OR
    /// - `hold_time` has not elapsed since the last accepted change.
    ///
    /// When returning `true`, the caller is expected to apply the weight and call
    /// [`accept`](Self::accept) to record the transition.
    pub fn should_update(&self, new_weight: f64, now: Instant) -> bool {
        // Hold time check.
        if let Some(last) = self.last_change {
            if now.duration_since(last) < self.hold_time {
                return false;
            }
        }

        // Threshold check -- relative change.
        if self.current_weight == 0.0 {
            // Any non-zero proposal is a significant change from zero.
            return new_weight != 0.0;
        }
        let relative_change =
            ((new_weight - self.current_weight) / self.current_weight).abs() * 100.0;
        relative_change >= self.threshold_percent
    }

    /// Record that a weight update was accepted at `now`.
    pub fn accept(&mut self, new_weight: f64, now: Instant) {
        self.current_weight = new_weight;
        self.last_change = Some(now);
    }
}

// ---------------------------------------------------------------------------
// SchedulerConfig
// ---------------------------------------------------------------------------

/// Tunable parameters for the bandwidth-aware scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// EWMA smoothing factor (0.0 = ignore new samples, 1.0 = no smoothing).
    pub ewma_alpha: f64,
    /// Hysteresis threshold in percent — applied to HysteresisGuard.
    pub hysteresis_percent: f64,
    /// Minimum hold time between weight changes — applied to HysteresisGuard.
    pub min_hold_time: Duration,
    // NOTE: TransferSize boundaries (256KB small, 16MB large) are fixed constants
    // in TransferSize::from_bytes. They are not configurable because they map to
    // fundamental algorithm boundaries (latency-optimal vs bandwidth-optimal).
    /// How often the scheduler should recompute weights.
    pub rebalance_interval: Duration,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            ewma_alpha: 0.2,
            hysteresis_percent: 15.0,
            min_hold_time: Duration::from_secs(2),
            rebalance_interval: Duration::from_secs(10),
        }
    }
}

// ---------------------------------------------------------------------------
// TransferRequest
// ---------------------------------------------------------------------------

/// A pending transfer that the scheduler must route or split.
#[derive(Debug, Clone)]
pub struct TransferRequest {
    /// Source node identifier.
    pub source_node: String,
    /// Destination node identifier.
    pub dest_node: String,
    /// Number of bytes to transfer.
    pub bytes: u64,
    /// Priority level (0 = lowest, 255 = highest).
    pub priority: u8,
    /// Pre-computed size classification.
    pub transfer_size: TransferSize,
}

impl TransferRequest {
    /// Create a new transfer request, automatically classifying the size.
    pub fn new(
        source_node: impl Into<String>,
        dest_node: impl Into<String>,
        bytes: u64,
        priority: u8,
    ) -> Self {
        Self {
            source_node: source_node.into(),
            dest_node: dest_node.into(),
            bytes,
            priority,
            transfer_size: TransferSize::from_bytes(bytes),
        }
    }
}

// ---------------------------------------------------------------------------
// SchedulerDecision
// ---------------------------------------------------------------------------

/// The outcome of a scheduling request.
#[derive(Debug, Clone)]
pub enum SchedulerDecision {
    /// Route the entire transfer through a single node.
    Route {
        node_id: String,
        weight: f64,
    },
    /// Split the transfer across multiple nodes with the given weights.
    Split {
        weights: Vec<(String, f64)>,
    },
    /// Defer the decision (e.g., adjust next kernel, not the running one).
    Defer {
        reason: String,
    },
    /// Reject the transfer entirely.
    Reject {
        reason: String,
    },
}

impl SchedulerDecision {
    /// Returns `true` for [`Route`](Self::Route) and [`Split`](Self::Split).
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Route { .. } | Self::Split { .. })
    }
}

// ---------------------------------------------------------------------------
// compute_weights
// ---------------------------------------------------------------------------

/// Compute normalised bandwidth weights from a set of node estimates.
///
/// Each node's EWMA estimate is divided by the sum of all estimates so the
/// returned weights sum to 1.0. Nodes with zero bandwidth receive zero weight.
///
/// Returns an empty vec when `nodes` is empty. Returns equal weights when all
/// EWMA values are zero.
pub fn compute_weights(nodes: &[NodeBandwidth]) -> Vec<(String, f64)> {
    if nodes.is_empty() {
        return Vec::new();
    }

    let total: u64 = nodes.iter().map(|n| n.ewma_bps).sum();

    if total == 0 {
        // All zero -- equal distribution.
        let equal = 1.0 / nodes.len() as f64;
        return nodes
            .iter()
            .map(|n| (n.node_id.clone(), equal))
            .collect();
    }

    nodes
        .iter()
        .map(|n| {
            let w = n.ewma_bps as f64 / total as f64;
            (n.node_id.clone(), w)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// ClusterBandwidthState
// ---------------------------------------------------------------------------

/// Aggregate bandwidth state for the entire cluster.
///
/// Tracks per-node EWMA estimates and provides the main entry points for
/// recording measurements and querying weights.
#[derive(Debug, Clone)]
pub struct ClusterBandwidthState {
    /// Per-node bandwidth trackers.
    pub nodes: Vec<NodeBandwidth>,
    /// When weights were last recomputed.
    pub last_rebalance: Option<Instant>,
    /// Scheduler configuration.
    pub config: SchedulerConfig,
}

impl ClusterBandwidthState {
    /// Create an empty cluster state with default config.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            last_rebalance: None,
            config: SchedulerConfig::default(),
        }
    }

    /// Create a cluster state with a custom config.
    pub fn with_config(config: SchedulerConfig) -> Self {
        Self {
            nodes: Vec::new(),
            last_rebalance: None,
            config,
        }
    }

    /// Record a bandwidth measurement for a node, creating the tracker if new.
    pub fn record_measurement(&mut self, node_id: &str, bps: u64) {
        if let Some(node) = self.nodes.iter_mut().find(|n| n.node_id == node_id) {
            node.update(bps);
        } else {
            let mut node = NodeBandwidth::new(node_id, self.config.ewma_alpha);
            node.update(bps);
            self.nodes.push(node);
        }
    }

    /// Get the current normalised weights for all tracked nodes.
    pub fn get_weights(&self) -> Vec<(String, f64)> {
        compute_weights(&self.nodes)
    }

    /// Returns `true` when at least `rebalance_interval` has elapsed since the
    /// last rebalance (or if no rebalance has ever occurred).
    pub fn should_rebalance(&self, now: Instant) -> bool {
        match self.last_rebalance {
            None => true,
            Some(last) => now.duration_since(last) >= self.config.rebalance_interval,
        }
    }

    /// Record that a rebalance occurred at `now`.
    pub fn mark_rebalanced(&mut self, now: Instant) {
        self.last_rebalance = Some(now);
    }

    /// Build a HysteresisGuard pre-configured from this cluster's scheduler config.
    pub fn make_hysteresis_guard(&self, initial_weight: f64) -> HysteresisGuard {
        HysteresisGuard::with_params(
            initial_weight,
            self.config.hysteresis_percent,
            self.config.min_hold_time,
        )
    }

    /// Sum of all nodes' EWMA bandwidth estimates.
    pub fn total_bandwidth_bps(&self) -> u64 {
        self.nodes.iter().map(|n| n.ewma_bps).sum()
    }
}

impl Default for ClusterBandwidthState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- TransferSize -------------------------------------------------------

    #[test]
    fn transfer_size_small_below_256kb() {
        assert_eq!(TransferSize::from_bytes(0), TransferSize::Small);
        assert_eq!(TransferSize::from_bytes(1), TransferSize::Small);
        assert_eq!(TransferSize::from_bytes(256 * 1024 - 1), TransferSize::Small);
    }

    #[test]
    fn transfer_size_medium_256kb_to_16mb() {
        assert_eq!(TransferSize::from_bytes(256 * 1024), TransferSize::Medium);
        assert_eq!(TransferSize::from_bytes(1_000_000), TransferSize::Medium);
        assert_eq!(TransferSize::from_bytes(16 * 1024 * 1024), TransferSize::Medium);
    }

    #[test]
    fn transfer_size_large_above_16mb() {
        assert_eq!(
            TransferSize::from_bytes(16 * 1024 * 1024 + 1),
            TransferSize::Large
        );
        assert_eq!(TransferSize::from_bytes(1_000_000_000), TransferSize::Large);
    }

    #[test]
    fn transfer_size_latency_sensitive() {
        assert!(TransferSize::Small.is_latency_sensitive());
        assert!(!TransferSize::Medium.is_latency_sensitive());
        assert!(!TransferSize::Large.is_latency_sensitive());
    }

    // -- NodeBandwidth EWMA -------------------------------------------------

    #[test]
    fn ewma_first_sample_seeds_directly() {
        let mut node = NodeBandwidth::new("n1", 0.2);
        node.update(1_000_000);
        assert_eq!(node.ewma_bps, 1_000_000);
        assert_eq!(node.measured_bps, 1_000_000);
    }

    #[test]
    fn ewma_second_sample_applies_smoothing() {
        let mut node = NodeBandwidth::new("n1", 0.2);
        node.update(1_000_000); // seed
        node.update(2_000_000); // 0.2 * 2M + 0.8 * 1M = 1.2M
        assert_eq!(node.ewma_bps, 1_200_000);
    }

    #[test]
    fn ewma_multiple_updates_converge() {
        let mut node = NodeBandwidth::new("n1", 0.2);
        node.update(1_000_000);
        // Feed the same value repeatedly -- should converge to it.
        for _ in 0..50 {
            node.update(5_000_000);
        }
        // After 50 updates at alpha=0.2 towards 5M, should be very close.
        let diff = (node.ewma_bps as f64 - 5_000_000.0).abs();
        assert!(
            diff < 1000.0,
            "Expected ~5000000, got {}, diff={}",
            node.ewma_bps,
            diff
        );
    }

    #[test]
    fn ewma_alpha_one_means_no_smoothing() {
        let mut node = NodeBandwidth::new("n1", 1.0);
        node.update(1_000_000);
        node.update(5_000_000);
        assert_eq!(node.ewma_bps, 5_000_000);
    }

    #[test]
    fn ewma_alpha_zero_means_ignore_new_samples() {
        let mut node = NodeBandwidth::new("n1", 0.0);
        node.update(1_000_000); // seed
        node.update(5_000_000); // 0.0 * 5M + 1.0 * 1M = 1M
        assert_eq!(node.ewma_bps, 1_000_000);
    }

    // -- HysteresisGuard ----------------------------------------------------

    #[test]
    fn hysteresis_blocks_small_changes() {
        let guard = HysteresisGuard::new(0.5);
        let now = Instant::now();
        // 5% change: should be blocked (below 15% threshold).
        assert!(!guard.should_update(0.525, now));
    }

    #[test]
    fn hysteresis_allows_large_changes() {
        let guard = HysteresisGuard::new(0.5);
        let now = Instant::now();
        // 20% change: should be allowed.
        assert!(guard.should_update(0.6, now));
    }

    #[test]
    fn hysteresis_blocks_during_hold_time() {
        let mut guard = HysteresisGuard::new(0.5);
        let now = Instant::now();
        guard.accept(0.6, now);
        // Even a large change should be blocked within hold time.
        assert!(!guard.should_update(0.9, now));
        // Still blocked 1 second later (hold is 2s).
        assert!(!guard.should_update(0.9, now + Duration::from_secs(1)));
    }

    #[test]
    fn hysteresis_allows_after_hold_time() {
        let mut guard = HysteresisGuard::new(0.5);
        let now = Instant::now();
        guard.accept(0.6, now);
        // After hold time, large change should be allowed.
        let later = now + Duration::from_secs(3);
        assert!(guard.should_update(0.9, later));
    }

    #[test]
    fn hysteresis_from_zero_weight() {
        let guard = HysteresisGuard::new(0.0);
        let now = Instant::now();
        // Any non-zero should be allowed from zero baseline.
        assert!(guard.should_update(0.1, now));
        // Zero to zero should not trigger.
        assert!(!guard.should_update(0.0, now));
    }

    // -- compute_weights ----------------------------------------------------

    #[test]
    fn compute_weights_empty() {
        let result = compute_weights(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn compute_weights_single_node() {
        let mut n = NodeBandwidth::new("solo", 0.2);
        n.update(1_000_000);
        let w = compute_weights(&[n]);
        assert_eq!(w.len(), 1);
        assert!((w[0].1 - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn compute_weights_sum_to_one() {
        let mut a = NodeBandwidth::new("a", 0.2);
        let mut b = NodeBandwidth::new("b", 0.2);
        let mut c = NodeBandwidth::new("c", 0.2);
        a.update(1_000_000);
        b.update(2_000_000);
        c.update(3_000_000);
        let w = compute_weights(&[a, b, c]);
        let sum: f64 = w.iter().map(|(_, v)| v).sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Weights must sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn compute_weights_proportional() {
        let mut a = NodeBandwidth::new("a", 0.2);
        let mut b = NodeBandwidth::new("b", 0.2);
        a.update(1_000_000);
        b.update(3_000_000);
        let w = compute_weights(&[a, b]);
        // a should get 0.25, b should get 0.75
        assert!((w[0].1 - 0.25).abs() < 1e-10);
        assert!((w[1].1 - 0.75).abs() < 1e-10);
    }

    #[test]
    fn compute_weights_all_zero_equal_distribution() {
        let a = NodeBandwidth::new("a", 0.2);
        let b = NodeBandwidth::new("b", 0.2);
        let w = compute_weights(&[a, b]);
        assert!((w[0].1 - 0.5).abs() < 1e-10);
        assert!((w[1].1 - 0.5).abs() < 1e-10);
    }

    // -- SchedulerDecision --------------------------------------------------

    #[test]
    fn decision_is_success() {
        let route = SchedulerDecision::Route {
            node_id: "n1".into(),
            weight: 1.0,
        };
        let split = SchedulerDecision::Split {
            weights: vec![("n1".into(), 0.5), ("n2".into(), 0.5)],
        };
        let defer = SchedulerDecision::Defer {
            reason: "running kernel".into(),
        };
        let reject = SchedulerDecision::Reject {
            reason: "no capacity".into(),
        };

        assert!(route.is_success());
        assert!(split.is_success());
        assert!(!defer.is_success());
        assert!(!reject.is_success());
    }

    // -- ClusterBandwidthState ----------------------------------------------

    #[test]
    fn cluster_record_and_get_weights() {
        let mut cluster = ClusterBandwidthState::new();
        cluster.record_measurement("n1", 1_000_000);
        cluster.record_measurement("n2", 3_000_000);
        let w = cluster.get_weights();
        assert_eq!(w.len(), 2);
        let sum: f64 = w.iter().map(|(_, v)| v).sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cluster_record_updates_existing_node() {
        let mut cluster = ClusterBandwidthState::new();
        cluster.record_measurement("n1", 1_000_000);
        cluster.record_measurement("n1", 2_000_000);
        // Should still have only one node.
        assert_eq!(cluster.nodes.len(), 1);
        // EWMA should be 0.2 * 2M + 0.8 * 1M = 1.2M
        assert_eq!(cluster.nodes[0].ewma_bps, 1_200_000);
    }

    #[test]
    fn cluster_total_bandwidth() {
        let mut cluster = ClusterBandwidthState::new();
        cluster.record_measurement("n1", 1_000_000);
        cluster.record_measurement("n2", 2_000_000);
        assert_eq!(cluster.total_bandwidth_bps(), 3_000_000);
    }

    #[test]
    fn cluster_should_rebalance_first_time() {
        let cluster = ClusterBandwidthState::new();
        assert!(cluster.should_rebalance(Instant::now()));
    }

    #[test]
    fn cluster_should_rebalance_respects_interval() {
        let mut cluster = ClusterBandwidthState::new();
        let now = Instant::now();
        cluster.last_rebalance = Some(now);
        // Too soon.
        assert!(!cluster.should_rebalance(now + Duration::from_secs(5)));
        // After interval (10s default).
        assert!(cluster.should_rebalance(now + Duration::from_secs(10)));
    }

    // -- SchedulerConfig defaults -------------------------------------------

    #[test]
    fn scheduler_config_defaults() {
        let cfg = SchedulerConfig::default();
        assert!((cfg.ewma_alpha - 0.2).abs() < f64::EPSILON);
        assert!((cfg.hysteresis_percent - 15.0).abs() < f64::EPSILON);
        assert_eq!(cfg.min_hold_time, Duration::from_secs(2));
        assert_eq!(cfg.rebalance_interval, Duration::from_secs(10));
    }

    // -- TransferRequest auto-classification --------------------------------

    #[test]
    fn transfer_request_auto_classifies() {
        let small = TransferRequest::new("a", "b", 100, 0);
        assert_eq!(small.transfer_size, TransferSize::Small);

        let large = TransferRequest::new("a", "b", 100_000_000, 5);
        assert_eq!(large.transfer_size, TransferSize::Large);
    }
}
