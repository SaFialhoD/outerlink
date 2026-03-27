//! NIC/Network Resilience types for OuterLink.
//!
//! Pure types and state machines for monitoring network link health,
//! estimating bandwidth via EWMA, and defining degradation policies.
//! No networking code, no async, no tokio — just state management.

use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// LinkState
// ---------------------------------------------------------------------------

/// Represents the health state of a network link.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkState {
    /// Link is fully operational.
    Up,
    /// Link is operational but experiencing issues (missed heartbeats, high error rate).
    Degraded,
    /// Link is not operational.
    Down,
}

impl LinkState {
    /// Returns `true` if the link is `Up` (fully healthy, no issues).
    pub fn is_healthy(&self) -> bool {
        matches!(self, Self::Up)
    }

    /// Returns `true` if the link can still carry data transfers (`Up` or `Degraded`).
    pub fn can_transfer(&self) -> bool {
        matches!(self, Self::Up | Self::Degraded)
    }
}

// ---------------------------------------------------------------------------
// LinkEvent
// ---------------------------------------------------------------------------

/// Events emitted by NIC monitoring that can trigger state transitions.
#[derive(Debug, Clone, PartialEq)]
pub enum LinkEvent {
    /// Physical link came up.
    LinkUp,
    /// Physical link went down.
    LinkDown,
    /// NIC negotiated a different speed (in bits per second).
    SpeedChange(u64),
    /// Error rate exceeded the configured threshold.
    ErrorThreshold,
}

// ---------------------------------------------------------------------------
// NetworkConfig
// ---------------------------------------------------------------------------

/// Configuration parameters for network health monitoring.
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Interval between heartbeat probes in milliseconds.
    pub heartbeat_interval_ms: u64,
    /// Time in milliseconds before a heartbeat is considered timed out.
    pub heartbeat_timeout_ms: u64,
    /// Number of consecutive missed heartbeats before entering `Degraded` state.
    pub suspect_after_missed: u32,
    /// Number of consecutive missed heartbeats before entering `Down` state.
    pub failed_after_missed: u32,
    /// Maximum Transmission Unit in bytes (9000 for jumbo frames).
    pub mtu: u32,
    /// TCP_USER_TIMEOUT socket option value in milliseconds.
    pub tcp_user_timeout_ms: u64,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval_ms: 1000,
            heartbeat_timeout_ms: 3000,
            suspect_after_missed: 3, // R47: phi-8 ≈ 3 missed before suspicion
            failed_after_missed: 5, // R47: phi-8 ≈ 5 missed before declaration
            mtu: 9000,
            tcp_user_timeout_ms: 15_000,
        }
    }
}

// ---------------------------------------------------------------------------
// BandwidthEstimate
// ---------------------------------------------------------------------------

/// Tracks estimated bandwidth using Exponentially Weighted Moving Average (EWMA).
#[derive(Debug, Clone)]
pub struct BandwidthEstimate {
    /// Current smoothed estimate of bandwidth in bits per second.
    pub current_bps: u64,
    /// Maximum link bandwidth in bits per second (theoretical max).
    pub max_bps: u64,
    /// EWMA smoothing factor. Higher values react faster to changes.
    /// Must be in (0.0, 1.0].
    pub ewma_alpha: f64,
}

impl BandwidthEstimate {
    /// Create a new estimate with the given maximum bandwidth.
    ///
    /// `current_bps` starts at `max_bps` (optimistic), `ewma_alpha` defaults to 0.2.
    pub fn new(max_bps: u64) -> Self {
        Self {
            current_bps: max_bps,
            max_bps,
            ewma_alpha: 0.2,
        }
    }

    /// Update the estimate with a new sample using EWMA smoothing.
    ///
    /// Formula: `new = alpha * sample + (1 - alpha) * old`
    pub fn update(&mut self, sample_bps: u64) {
        let new = self.ewma_alpha * (sample_bps as f64)
            + (1.0 - self.ewma_alpha) * (self.current_bps as f64);
        self.current_bps = new as u64;
    }

    /// Returns the current utilization as a ratio in [0.0, 1.0].
    ///
    /// Returns 0.0 if `max_bps` is zero to avoid division by zero.
    pub fn utilization(&self) -> f64 {
        if self.max_bps == 0 {
            return 0.0;
        }
        let ratio = self.current_bps as f64 / self.max_bps as f64;
        ratio.clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// DegradationPolicy
// ---------------------------------------------------------------------------

/// Policy defining how the system reacts when bandwidth degrades.
#[derive(Debug, Clone)]
pub struct DegradationPolicy {
    /// Minimum transfer size in bytes; transfers below this are deferred
    /// or batched when bandwidth drops.
    pub min_transfer_size: u64,
    /// If `true`, stop proactive migration of GPU memory pages when
    /// bandwidth is degraded.
    pub stop_proactive_migration: bool,
    /// If `true`, adjust kernel weight distribution to prefer local
    /// execution when bandwidth is degraded.
    pub adjust_kernel_weights: bool,
}

impl Default for DegradationPolicy {
    fn default() -> Self {
        Self {
            min_transfer_size: 4096,
            stop_proactive_migration: true,
            adjust_kernel_weights: true,
        }
    }
}

// ---------------------------------------------------------------------------
// ConnectionHealth
// ---------------------------------------------------------------------------

/// Tracks the health of a single connection to a remote peer.
#[derive(Debug, Clone)]
pub struct ConnectionHealth {
    /// Current link state.
    pub state: LinkState,
    /// When this connection was created (used as reference when no heartbeat received yet).
    pub created_at: Instant,
    /// Timestamp of the last successful heartbeat, if any.
    pub last_heartbeat: Option<Instant>,
    /// Number of consecutive missed heartbeats.
    pub missed_heartbeats: u32,
    /// Bandwidth estimate for this connection.
    pub bandwidth: BandwidthEstimate,
}

impl ConnectionHealth {
    /// Create a new `ConnectionHealth` in `Up` state with the given max bandwidth.
    pub fn new(max_bps: u64) -> Self {
        Self::new_at(max_bps, Instant::now())
    }

    /// Create with an explicit creation time (for testing).
    pub fn new_at(max_bps: u64, created_at: Instant) -> Self {
        Self {
            state: LinkState::Up,
            created_at,
            last_heartbeat: None,
            missed_heartbeats: 0,
            bandwidth: BandwidthEstimate::new(max_bps),
        }
    }

    /// Record a successful heartbeat at the given instant.
    ///
    /// Resets missed heartbeat counter and transitions state to `Up`.
    pub fn record_heartbeat(&mut self, now: Instant) {
        self.last_heartbeat = Some(now);
        self.missed_heartbeats = 0;
        self.state = LinkState::Up;
    }

    /// Check the health of this connection based on the given config and current time.
    ///
    /// Increments missed heartbeat counter if the last heartbeat is too old,
    /// and transitions state accordingly.
    pub fn check_health(&mut self, config: &NetworkConfig, now: Instant) -> LinkState {
        let timeout = Duration::from_millis(config.heartbeat_timeout_ms);

        // Use last heartbeat time, or creation time if no heartbeat received yet.
        // This ensures the first "miss" only fires after a real timeout elapses.
        let reference = self.last_heartbeat.unwrap_or(self.created_at);
        let timed_out = now.duration_since(reference) > timeout;

        if timed_out {
            self.missed_heartbeats = self.missed_heartbeats.saturating_add(1);
        }

        self.state = if self.missed_heartbeats >= config.failed_after_missed {
            LinkState::Down
        } else if self.missed_heartbeats >= config.suspect_after_missed {
            LinkState::Degraded
        } else {
            LinkState::Up
        };

        self.state
    }

    /// Record a completed data transfer and update bandwidth estimate.
    ///
    /// `bytes` is the number of bytes transferred, `duration` is the wall-clock
    /// time the transfer took. Ignores zero-duration transfers to avoid division
    /// by zero.
    pub fn record_transfer(&mut self, bytes: u64, duration: Duration) {
        let secs = duration.as_secs_f64();
        if secs <= 0.0 {
            return;
        }
        let bits = bytes * 8;
        let sample_bps = (bits as f64 / secs) as u64;
        self.bandwidth.update(sample_bps);
    }

    /// Apply a link event, immediately updating state.
    pub fn apply_event(&mut self, event: &LinkEvent) {
        match event {
            LinkEvent::LinkDown => {
                self.state = LinkState::Down;
            }
            LinkEvent::LinkUp => {
                // Physical link restored; wait for heartbeat to confirm app-level health.
                if self.state == LinkState::Down {
                    self.state = LinkState::Degraded;
                    self.missed_heartbeats = 0; // Reset so check_health doesn't immediately revert to Down
                    self.last_heartbeat = None; // Force re-evaluation from created_at
                }
            }
            LinkEvent::SpeedChange(new_bps) => {
                // Update max bandwidth; state unchanged.
                self.bandwidth.max_bps = *new_bps;
            }
            LinkEvent::ErrorThreshold => {
                if self.state == LinkState::Up {
                    self.state = LinkState::Degraded;
                }
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

    // -- LinkState ----------------------------------------------------------

    #[test]
    fn test_link_state_is_healthy() {
        assert!(LinkState::Up.is_healthy());
        assert!(!LinkState::Degraded.is_healthy());
        assert!(!LinkState::Down.is_healthy());
    }

    #[test]
    fn test_link_state_can_transfer() {
        assert!(LinkState::Up.can_transfer());
        assert!(LinkState::Degraded.can_transfer());
        assert!(!LinkState::Down.can_transfer());
    }

    #[test]
    fn test_link_state_clone_eq() {
        let a = LinkState::Degraded;
        let b = a;
        assert_eq!(a, b);
    }

    // -- LinkEvent ----------------------------------------------------------

    #[test]
    fn test_link_event_variants() {
        let up = LinkEvent::LinkUp;
        let down = LinkEvent::LinkDown;
        let speed = LinkEvent::SpeedChange(25_000_000_000);
        let err = LinkEvent::ErrorThreshold;

        assert_eq!(up, LinkEvent::LinkUp);
        assert_eq!(down, LinkEvent::LinkDown);
        assert_eq!(speed, LinkEvent::SpeedChange(25_000_000_000));
        assert_eq!(err, LinkEvent::ErrorThreshold);
    }

    // -- NetworkConfig ------------------------------------------------------

    #[test]
    fn test_network_config_defaults() {
        let cfg = NetworkConfig::default();
        assert_eq!(cfg.heartbeat_interval_ms, 1000);
        assert_eq!(cfg.heartbeat_timeout_ms, 3000);
        assert_eq!(cfg.suspect_after_missed, 3);
        assert_eq!(cfg.failed_after_missed, 5);
        assert_eq!(cfg.mtu, 9000);
        assert_eq!(cfg.tcp_user_timeout_ms, 15_000);
    }

    // -- BandwidthEstimate --------------------------------------------------

    #[test]
    fn test_bandwidth_estimate_new() {
        let bw = BandwidthEstimate::new(10_000_000_000);
        assert_eq!(bw.current_bps, 10_000_000_000);
        assert_eq!(bw.max_bps, 10_000_000_000);
        assert!((bw.ewma_alpha - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bandwidth_ewma_single_update() {
        let mut bw = BandwidthEstimate::new(1_000_000);
        // EWMA: 0.2 * 500_000 + 0.8 * 1_000_000 = 100_000 + 800_000 = 900_000
        bw.update(500_000);
        assert_eq!(bw.current_bps, 900_000);
    }

    #[test]
    fn test_bandwidth_ewma_multiple_updates() {
        let mut bw = BandwidthEstimate::new(1_000_000);
        // Update 1: 0.2 * 500_000 + 0.8 * 1_000_000 = 900_000
        bw.update(500_000);
        assert_eq!(bw.current_bps, 900_000);

        // Update 2: 0.2 * 500_000 + 0.8 * 900_000 = 100_000 + 720_000 = 820_000
        bw.update(500_000);
        assert_eq!(bw.current_bps, 820_000);

        // Update 3: 0.2 * 500_000 + 0.8 * 820_000 = 100_000 + 656_000 = 756_000
        bw.update(500_000);
        assert_eq!(bw.current_bps, 756_000);
    }

    #[test]
    fn test_bandwidth_ewma_converges_upward() {
        let mut bw = BandwidthEstimate {
            current_bps: 0,
            max_bps: 1_000_000,
            ewma_alpha: 0.5,
        };
        // With alpha=0.5 and constant 1M samples, should converge toward 1M
        for _ in 0..20 {
            bw.update(1_000_000);
        }
        // After 20 iterations with alpha=0.5, should be very close to 1M
        assert!(bw.current_bps > 990_000);
    }

    #[test]
    fn test_bandwidth_utilization_full() {
        let bw = BandwidthEstimate::new(1_000_000);
        assert!((bw.utilization() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bandwidth_utilization_partial() {
        let bw = BandwidthEstimate {
            current_bps: 500_000,
            max_bps: 1_000_000,
            ewma_alpha: 0.2,
        };
        assert!((bw.utilization() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bandwidth_utilization_zero_max() {
        let bw = BandwidthEstimate {
            current_bps: 100,
            max_bps: 0,
            ewma_alpha: 0.2,
        };
        assert!((bw.utilization() - 0.0).abs() < f64::EPSILON);
    }

    // -- DegradationPolicy --------------------------------------------------

    #[test]
    fn test_degradation_policy_defaults() {
        let policy = DegradationPolicy::default();
        assert_eq!(policy.min_transfer_size, 4096);
        assert!(policy.stop_proactive_migration);
        assert!(policy.adjust_kernel_weights);
    }

    // -- ConnectionHealth ---------------------------------------------------

    #[test]
    fn test_connection_health_new() {
        let ch = ConnectionHealth::new(10_000_000_000);
        assert_eq!(ch.state, LinkState::Up);
        assert!(ch.last_heartbeat.is_none());
        assert_eq!(ch.missed_heartbeats, 0);
        assert_eq!(ch.bandwidth.max_bps, 10_000_000_000);
    }

    #[test]
    fn test_connection_health_record_heartbeat() {
        let mut ch = ConnectionHealth::new(1_000_000);
        ch.missed_heartbeats = 5;
        ch.state = LinkState::Down;

        let now = Instant::now();
        ch.record_heartbeat(now);

        assert_eq!(ch.state, LinkState::Up);
        assert_eq!(ch.missed_heartbeats, 0);
        assert_eq!(ch.last_heartbeat, Some(now));
    }

    #[test]
    fn test_connection_health_check_no_heartbeat_ever() {
        // Create connection 10s in the past so timeout fires
        let past = Instant::now() - Duration::from_secs(10);
        let mut ch = ConnectionHealth::new_at(1_000_000, past);
        let config = NetworkConfig::default(); // suspect=3, failed=5
        let now = Instant::now();

        // First check: no heartbeat ever, 10s > 3s timeout = 1 missed
        let state = ch.check_health(&config, now);
        assert_eq!(ch.missed_heartbeats, 1);
        assert_eq!(state, LinkState::Up); // < suspect_after_missed(3)
    }

    #[test]
    fn test_connection_health_transitions_to_degraded() {
        let past = Instant::now() - Duration::from_secs(10);
        let mut ch = ConnectionHealth::new_at(1_000_000, past);
        let config = NetworkConfig::default(); // suspect_after_missed = 3
        let now = Instant::now();

        ch.check_health(&config, now); // missed=1 -> Up
        ch.check_health(&config, now); // missed=2 -> Up
        let state = ch.check_health(&config, now); // missed=3 -> Degraded
        assert_eq!(ch.missed_heartbeats, 3);
        assert_eq!(state, LinkState::Degraded);
    }

    #[test]
    fn test_connection_health_transitions_to_down() {
        let past = Instant::now() - Duration::from_secs(10);
        let mut ch = ConnectionHealth::new_at(1_000_000, past);
        let config = NetworkConfig::default(); // failed_after_missed = 5
        let now = Instant::now();

        for _ in 0..4 {
            ch.check_health(&config, now);
        }
        assert_eq!(ch.missed_heartbeats, 4);
        assert_eq!(ch.state, LinkState::Degraded);

        let state = ch.check_health(&config, now); // missed=5 -> Down
        assert_eq!(ch.missed_heartbeats, 5);
        assert_eq!(state, LinkState::Down);
    }

    #[test]
    fn test_connection_health_recovery_after_heartbeat() {
        let past = Instant::now() - Duration::from_secs(10);
        let mut ch = ConnectionHealth::new_at(1_000_000, past);
        let config = NetworkConfig::default(); // failed=5
        let now = Instant::now();

        // Drive to Down (5 misses)
        for _ in 0..5 {
            ch.check_health(&config, now);
        }
        assert_eq!(ch.state, LinkState::Down);

        // Heartbeat arrives — recovery
        ch.record_heartbeat(now);
        assert_eq!(ch.state, LinkState::Up);
        assert_eq!(ch.missed_heartbeats, 0);

        // Next check should be Up since heartbeat is fresh (now == last_heartbeat)
        let state = ch.check_health(&config, now);
        assert_eq!(state, LinkState::Up);
    }

    #[test]
    fn test_connection_health_record_transfer() {
        let mut ch = ConnectionHealth::new(10_000_000_000); // 10 Gbps
        // Transfer 1 GB in 1 second = 8 Gbps
        ch.record_transfer(1_000_000_000, Duration::from_secs(1));
        // EWMA: 0.2 * 8_000_000_000 + 0.8 * 10_000_000_000 = 9_600_000_000
        assert_eq!(ch.bandwidth.current_bps, 9_600_000_000);
    }

    #[test]
    fn test_connection_health_record_transfer_zero_duration() {
        let mut ch = ConnectionHealth::new(1_000_000);
        let original = ch.bandwidth.current_bps;
        // Zero-duration transfer should be ignored
        ch.record_transfer(1000, Duration::ZERO);
        assert_eq!(ch.bandwidth.current_bps, original);
    }
}
