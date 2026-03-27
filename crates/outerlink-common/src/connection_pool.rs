//! Connection pool types and priority lane management for OuterLink transports.
//!
//! Provides pure types and state machines for managing pooled TCP connections
//! across priority lanes (Control, Data, Bulk). No networking code, no async,
//! no tokio -- just types, config, and selection logic.
//!
//! Design rationale (R37): QUIC rejected (8% of 100GbE capacity). Separate TCP
//! connections per priority lane avoids HOL blocking. Custom pool because
//! connections are GPU-node-specific (bb8/deadpool don't fit).

use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// PriorityLane
// ---------------------------------------------------------------------------

/// Priority lane for connection traffic.
///
/// Each lane uses separate TCP connections to avoid head-of-line blocking.
/// On PFC/ECN LANs, HOL blocking within a lane is negligible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PriorityLane {
    /// Low-latency lane for control messages (context creation, device queries, etc.).
    /// Typically 1 connection -- these are small, infrequent messages that must
    /// never be delayed behind bulk transfers.
    Control,

    /// High-throughput lane for regular data transfers (kernel launches, small
    /// memcpy operations). Typically 4 connections for parallelism.
    Data,

    /// Background lane for large bulk transfers (model loading, checkpoint saves).
    /// Typically 2 connections. Lower priority, won't starve control/data.
    Bulk,
}

impl PriorityLane {
    /// Default number of connections for this lane.
    pub fn default_connections(&self) -> u32 {
        match self {
            PriorityLane::Control => 1,
            PriorityLane::Data => 4,
            PriorityLane::Bulk => 2,
        }
    }

    /// Human-readable description of this lane's purpose.
    pub fn description(&self) -> &'static str {
        match self {
            PriorityLane::Control => "Low-latency control messages",
            PriorityLane::Data => "High-throughput data transfers",
            PriorityLane::Bulk => "Background bulk transfers",
        }
    }

    /// Whether this lane is latency-sensitive (should never be blocked by bulk).
    pub fn is_latency_sensitive(&self) -> bool {
        match self {
            PriorityLane::Control => true,
            PriorityLane::Data => false,
            PriorityLane::Bulk => false,
        }
    }
}

// ---------------------------------------------------------------------------
// ConnectionId
// ---------------------------------------------------------------------------

/// Uniquely identifies a connection within the pool.
///
/// Composed of the lane it belongs to, its index within that lane, and the
/// remote node it connects to.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConnectionId {
    /// Which priority lane this connection belongs to.
    pub lane: PriorityLane,

    /// Index within the lane (0-based). For example, Data lane index 2 of 4.
    pub index: u32,

    /// Identifier of the remote GPU node (e.g. hostname or UUID).
    pub node_id: String,
}

impl ConnectionId {
    /// Create a new connection identifier.
    pub fn new(lane: PriorityLane, index: u32, node_id: impl Into<String>) -> Self {
        Self {
            lane,
            index,
            node_id: node_id.into(),
        }
    }
}

impl std::fmt::Display for ConnectionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}[{}]@{}", self.lane, self.index, self.node_id)
    }
}

// ---------------------------------------------------------------------------
// ConnectionState
// ---------------------------------------------------------------------------

/// State of a single connection in the pool.
///
/// Follows the lifecycle: Connecting -> Ready <-> Busy -> Draining -> Closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConnectionState {
    /// TCP handshake in progress.
    Connecting,

    /// Connected and idle, available for use.
    Ready,

    /// Currently handling a request/transfer.
    Busy,

    /// Gracefully shutting down (finishing in-flight work, refusing new work).
    Draining,

    /// Connection is closed (terminal state).
    Closed,
}

impl ConnectionState {
    /// Whether this connection can accept new work.
    pub fn is_available(&self) -> bool {
        matches!(self, ConnectionState::Ready)
    }

    /// Whether this is a terminal state (no further transitions possible).
    pub fn is_terminal(&self) -> bool {
        matches!(self, ConnectionState::Closed)
    }
}

// ---------------------------------------------------------------------------
// ConnectionInfo
// ---------------------------------------------------------------------------

/// Runtime information about a single pooled connection.
///
/// Tracks identity, state, timing, and traffic statistics. Used by the pool
/// manager to make scheduling and health decisions.
#[derive(Debug, Clone)]
pub struct ConnectionInfo {
    /// Unique identifier for this connection.
    pub id: ConnectionId,

    /// Current state.
    pub state: ConnectionState,

    /// When this connection was established.
    pub created_at: Instant,

    /// Last time this connection was used for a request.
    pub last_used: Option<Instant>,

    /// Total bytes sent over this connection.
    pub bytes_sent: u64,

    /// Total bytes received over this connection.
    pub bytes_received: u64,

    /// Number of errors encountered on this connection.
    pub error_count: u32,
}

impl ConnectionInfo {
    /// Create a new ConnectionInfo in the Connecting state.
    pub fn new(id: ConnectionId) -> Self {
        Self {
            id,
            state: ConnectionState::Connecting,
            created_at: Instant::now(),
            last_used: None,
            bytes_sent: 0,
            bytes_received: 0,
            error_count: 0,
        }
    }

    /// Duration since this connection was last used, or since creation if never used.
    pub fn idle_duration(&self) -> Duration {
        let reference = self.last_used.unwrap_or(self.created_at);
        reference.elapsed()
    }

    /// Whether this connection is healthy (error count below threshold).
    pub fn is_healthy(&self, max_errors: u32) -> bool {
        self.error_count < max_errors
    }

    /// Fraction of traffic that is outbound (sent / total). Returns 0.0 if no traffic.
    pub fn send_ratio(&self) -> f64 {
        let total = self.bytes_sent + self.bytes_received;
        if total == 0 {
            return 0.0;
        }
        self.bytes_sent as f64 / total as f64
    }
}

// ---------------------------------------------------------------------------
// PoolConfig
// ---------------------------------------------------------------------------

/// Configuration for the connection pool.
///
/// Defines how many connections per lane, idle timeouts, and reconnect policy.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Number of TCP connections for the Control lane.
    pub control_connections: u32,

    /// Number of TCP connections for the Data lane.
    pub data_connections: u32,

    /// Number of TCP connections for the Bulk lane.
    pub bulk_connections: u32,

    /// Maximum idle time (seconds) before a connection is closed.
    pub max_idle_secs: u64,

    /// Interval (seconds) between health check probes.
    pub health_check_interval_secs: u64,
    // NOTE: Reconnect delay and max attempts live in ReconnectPolicy (single source of truth).
    // Do NOT duplicate them here — use ReconnectPolicy::new() directly.
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            control_connections: 1,
            data_connections: 4,
            bulk_connections: 2,
            max_idle_secs: 300,
            health_check_interval_secs: 30,
        }
    }
}

impl PoolConfig {
    /// Total number of connections across all lanes.
    pub fn total_connections(&self) -> u32 {
        self.control_connections + self.data_connections + self.bulk_connections
    }

    /// Get the configured connection count for a specific lane.
    pub fn connections_for_lane(&self, lane: PriorityLane) -> u32 {
        match lane {
            PriorityLane::Control => self.control_connections,
            PriorityLane::Data => self.data_connections,
            PriorityLane::Bulk => self.bulk_connections,
        }
    }
}

// ---------------------------------------------------------------------------
// PoolStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for the entire connection pool.
///
/// Provides a snapshot view for monitoring and health decisions.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total connections in the pool (all states).
    pub total_connections: u32,

    /// Connections in Ready state.
    pub ready_connections: u32,

    /// Connections in Busy state.
    pub busy_connections: u32,

    /// Connections in Closed state (failed or terminated).
    pub failed_connections: u32,

    /// Total bytes sent across all connections.
    pub total_bytes_sent: u64,

    /// Total bytes received across all connections.
    pub total_bytes_received: u64,
}

impl PoolStats {
    /// Ratio of healthy (non-failed) connections to total. Returns 1.0 if no connections.
    pub fn health_ratio(&self) -> f64 {
        if self.total_connections == 0 {
            return 1.0;
        }
        let healthy = self.total_connections.saturating_sub(self.failed_connections);
        healthy as f64 / self.total_connections as f64
    }
}

// ---------------------------------------------------------------------------
// LaneSelector
// ---------------------------------------------------------------------------

/// Threshold above which payloads are routed to the Bulk lane (1 MiB).
const BULK_THRESHOLD_BYTES: usize = 1024 * 1024;

/// Selects the appropriate priority lane for a message.
///
/// Routing rules:
/// - Control messages always go to the Control lane regardless of size.
/// - Payloads larger than 1 MiB go to the Bulk lane.
/// - Everything else goes to the Data lane.
pub struct LaneSelector;

impl LaneSelector {
    /// Pick the right lane for a message given its payload size and control flag.
    pub fn select_lane(payload_size: usize, is_control: bool) -> PriorityLane {
        if is_control {
            PriorityLane::Control
        } else if payload_size > BULK_THRESHOLD_BYTES {
            PriorityLane::Bulk
        } else {
            PriorityLane::Data
        }
    }
}

// ---------------------------------------------------------------------------
// ReconnectPolicy
// ---------------------------------------------------------------------------

/// Tracks reconnection state with exponential backoff and jitter.
///
/// Each call to [`next_delay`] returns the delay for the current attempt and
/// advances the attempt counter. The delay formula:
///
///   delay = min(base_delay_ms * 2^current_attempt, max_delay_ms) + jitter
///
/// Jitter is deterministic based on the attempt number (0..25% of computed delay)
/// to avoid thundering herd without requiring a random number generator.
/// The runtime layer may add true randomness on top.
#[derive(Debug, Clone)]
pub struct ReconnectPolicy {
    /// Current attempt number (0-indexed, incremented after each call to `next_delay`).
    pub current_attempt: u32,

    /// Maximum attempts before giving up.
    pub max_attempts: u32,

    /// Base delay in milliseconds (first attempt delay before backoff).
    pub base_delay_ms: u64,

    /// Maximum delay in milliseconds (caps exponential growth).
    pub max_delay_ms: u64,
}

impl ReconnectPolicy {
    /// Create a policy with explicit parameters.
    pub fn new(max_attempts: u32, base_delay_ms: u64, max_delay_ms: u64) -> Self {
        Self {
            current_attempt: 0,
            max_attempts,
            base_delay_ms,
            max_delay_ms,
        }
    }

    /// Whether another reconnect attempt should be made.
    pub fn should_retry(&self) -> bool {
        self.current_attempt < self.max_attempts
    }

    /// Compute the delay for the current attempt, then advance the counter.
    ///
    /// Returns `None` if `should_retry()` is false (attempts exhausted).
    /// Jitter is deterministic: `(attempt * 7 + 13) % 25` percent of the base delay.
    pub fn next_delay(&mut self) -> Option<Duration> {
        if !self.should_retry() {
            return None;
        }

        let attempt = self.current_attempt;
        self.current_attempt += 1;

        // Exponential backoff: base * 2^attempt, capped at max
        let multiplier = 1u64.checked_shl(attempt).unwrap_or(u64::MAX);
        let delay_ms = self.base_delay_ms.saturating_mul(multiplier).min(self.max_delay_ms);

        // Deterministic jitter: 0-25% of delay based on attempt number
        let jitter_pct = ((attempt as u64).wrapping_mul(7).wrapping_add(13)) % 26;
        let jitter_ms = delay_ms * jitter_pct / 100;

        Some(Duration::from_millis(delay_ms + jitter_ms))
    }

    /// Reset the policy to its initial state (e.g., after a successful connection).
    pub fn reset(&mut self) {
        self.current_attempt = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- PriorityLane tests --

    #[test]
    fn test_priority_lane_default_connections() {
        assert_eq!(PriorityLane::Control.default_connections(), 1);
        assert_eq!(PriorityLane::Data.default_connections(), 4);
        assert_eq!(PriorityLane::Bulk.default_connections(), 2);
    }

    #[test]
    fn test_priority_lane_descriptions_non_empty() {
        assert!(!PriorityLane::Control.description().is_empty());
        assert!(!PriorityLane::Data.description().is_empty());
        assert!(!PriorityLane::Bulk.description().is_empty());
    }

    #[test]
    fn test_priority_lane_latency_sensitive() {
        assert!(PriorityLane::Control.is_latency_sensitive());
        assert!(!PriorityLane::Data.is_latency_sensitive());
        assert!(!PriorityLane::Bulk.is_latency_sensitive());
    }

    #[test]
    fn test_priority_lane_clone_eq() {
        let a = PriorityLane::Data;
        let b = a;
        assert_eq!(a, b);
        assert_ne!(PriorityLane::Control, PriorityLane::Data);
    }

    // -- ConnectionId tests --

    #[test]
    fn test_connection_id_new() {
        let id = ConnectionId::new(PriorityLane::Data, 2, "node-1");
        assert_eq!(id.lane, PriorityLane::Data);
        assert_eq!(id.index, 2);
        assert_eq!(id.node_id, "node-1");
    }

    #[test]
    fn test_connection_id_equality() {
        let a = ConnectionId::new(PriorityLane::Control, 0, "gpu-server-1");
        let b = ConnectionId::new(PriorityLane::Control, 0, "gpu-server-1");
        let c = ConnectionId::new(PriorityLane::Control, 1, "gpu-server-1");
        let d = ConnectionId::new(PriorityLane::Data, 0, "gpu-server-1");
        assert_eq!(a, b);
        assert_ne!(a, c); // different index
        assert_ne!(a, d); // different lane
    }

    #[test]
    fn test_connection_id_display() {
        let id = ConnectionId::new(PriorityLane::Bulk, 1, "worker-3");
        let s = format!("{}", id);
        assert!(s.contains("Bulk"));
        assert!(s.contains("1"));
        assert!(s.contains("worker-3"));
    }

    // -- ConnectionState tests --

    #[test]
    fn test_connection_state_is_available() {
        assert!(!ConnectionState::Connecting.is_available());
        assert!(ConnectionState::Ready.is_available());
        assert!(!ConnectionState::Busy.is_available());
        assert!(!ConnectionState::Draining.is_available());
        assert!(!ConnectionState::Closed.is_available());
    }

    #[test]
    fn test_connection_state_is_terminal() {
        assert!(!ConnectionState::Connecting.is_terminal());
        assert!(!ConnectionState::Ready.is_terminal());
        assert!(!ConnectionState::Busy.is_terminal());
        assert!(!ConnectionState::Draining.is_terminal());
        assert!(ConnectionState::Closed.is_terminal());
    }

    // -- ConnectionInfo tests --

    #[test]
    fn test_connection_info_new_defaults() {
        let id = ConnectionId::new(PriorityLane::Data, 0, "node-1");
        let info = ConnectionInfo::new(id.clone());
        assert_eq!(info.id, id);
        assert_eq!(info.state, ConnectionState::Connecting);
        assert!(info.last_used.is_none());
        assert_eq!(info.bytes_sent, 0);
        assert_eq!(info.bytes_received, 0);
        assert_eq!(info.error_count, 0);
    }

    #[test]
    fn test_connection_info_idle_duration_without_use() {
        let id = ConnectionId::new(PriorityLane::Control, 0, "node-1");
        let info = ConnectionInfo::new(id);
        // Just created -- idle duration should be very small (< 1 second)
        assert!(info.idle_duration() < Duration::from_secs(1));
    }

    #[test]
    fn test_connection_info_is_healthy() {
        let id = ConnectionId::new(PriorityLane::Data, 0, "node-1");
        let mut info = ConnectionInfo::new(id);
        assert!(info.is_healthy(5));
        info.error_count = 4;
        assert!(info.is_healthy(5));
        info.error_count = 5;
        assert!(!info.is_healthy(5));
        info.error_count = 10;
        assert!(!info.is_healthy(5));
    }

    #[test]
    fn test_connection_info_utilization_no_traffic() {
        let id = ConnectionId::new(PriorityLane::Data, 0, "node-1");
        let info = ConnectionInfo::new(id);
        assert_eq!(info.send_ratio(), 0.0);
    }

    #[test]
    fn test_connection_info_utilization_send_only() {
        let id = ConnectionId::new(PriorityLane::Data, 0, "node-1");
        let mut info = ConnectionInfo::new(id);
        info.bytes_sent = 1000;
        info.bytes_received = 0;
        assert_eq!(info.send_ratio(), 1.0);
    }

    #[test]
    fn test_connection_info_utilization_balanced() {
        let id = ConnectionId::new(PriorityLane::Data, 0, "node-1");
        let mut info = ConnectionInfo::new(id);
        info.bytes_sent = 500;
        info.bytes_received = 500;
        assert!((info.send_ratio() - 0.5).abs() < f64::EPSILON);
    }

    // -- PoolConfig tests --

    #[test]
    fn test_pool_config_defaults() {
        let cfg = PoolConfig::default();
        assert_eq!(cfg.control_connections, 1);
        assert_eq!(cfg.data_connections, 4);
        assert_eq!(cfg.bulk_connections, 2);
        assert_eq!(cfg.max_idle_secs, 300);
        assert_eq!(cfg.health_check_interval_secs, 30);
        assert_eq!(cfg.health_check_interval_secs, 30);
    }

    #[test]
    fn test_pool_config_total_connections() {
        let cfg = PoolConfig::default();
        assert_eq!(cfg.total_connections(), 7); // 1 + 4 + 2
    }

    #[test]
    fn test_pool_config_connections_for_lane() {
        let cfg = PoolConfig::default();
        assert_eq!(cfg.connections_for_lane(PriorityLane::Control), 1);
        assert_eq!(cfg.connections_for_lane(PriorityLane::Data), 4);
        assert_eq!(cfg.connections_for_lane(PriorityLane::Bulk), 2);
    }

    // -- PoolStats tests --

    #[test]
    fn test_pool_stats_health_ratio_no_connections() {
        let stats = PoolStats::default();
        assert_eq!(stats.health_ratio(), 1.0);
    }

    #[test]
    fn test_pool_stats_health_ratio_all_healthy() {
        let stats = PoolStats {
            total_connections: 7,
            ready_connections: 5,
            busy_connections: 2,
            failed_connections: 0,
            total_bytes_sent: 0,
            total_bytes_received: 0,
        };
        assert_eq!(stats.health_ratio(), 1.0);
    }

    #[test]
    fn test_pool_stats_health_ratio_some_failed() {
        let stats = PoolStats {
            total_connections: 10,
            ready_connections: 5,
            busy_connections: 2,
            failed_connections: 3,
            total_bytes_sent: 0,
            total_bytes_received: 0,
        };
        assert!((stats.health_ratio() - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pool_stats_health_ratio_all_failed() {
        let stats = PoolStats {
            total_connections: 5,
            ready_connections: 0,
            busy_connections: 0,
            failed_connections: 5,
            total_bytes_sent: 0,
            total_bytes_received: 0,
        };
        assert_eq!(stats.health_ratio(), 0.0);
    }

    // -- LaneSelector tests --

    #[test]
    fn test_lane_selector_control_always_control() {
        assert_eq!(LaneSelector::select_lane(0, true), PriorityLane::Control);
        assert_eq!(LaneSelector::select_lane(1024, true), PriorityLane::Control);
        // Even huge control messages go to Control lane
        assert_eq!(
            LaneSelector::select_lane(10 * 1024 * 1024, true),
            PriorityLane::Control
        );
    }

    #[test]
    fn test_lane_selector_small_payload_data() {
        assert_eq!(LaneSelector::select_lane(0, false), PriorityLane::Data);
        assert_eq!(LaneSelector::select_lane(1024, false), PriorityLane::Data);
        assert_eq!(
            LaneSelector::select_lane(1024 * 1024, false),
            PriorityLane::Data
        );
    }

    #[test]
    fn test_lane_selector_large_payload_bulk() {
        assert_eq!(
            LaneSelector::select_lane(1024 * 1024 + 1, false),
            PriorityLane::Bulk
        );
        assert_eq!(
            LaneSelector::select_lane(100 * 1024 * 1024, false),
            PriorityLane::Bulk
        );
    }

    #[test]
    fn test_lane_selector_boundary_exactly_1mib() {
        // Exactly 1 MiB goes to Data (threshold is >, not >=)
        assert_eq!(
            LaneSelector::select_lane(BULK_THRESHOLD_BYTES, false),
            PriorityLane::Data
        );
        assert_eq!(
            LaneSelector::select_lane(BULK_THRESHOLD_BYTES + 1, false),
            PriorityLane::Bulk
        );
    }

    // -- ReconnectPolicy tests --

    #[test]
    fn test_reconnect_policy_new() {
        let policy = ReconnectPolicy::new(5, 1000, 30_000);
        assert_eq!(policy.current_attempt, 0);
        assert_eq!(policy.max_attempts, 5);
        assert_eq!(policy.base_delay_ms, 1000);
        assert_eq!(policy.max_delay_ms, 30_000);
    }

    #[test]
    fn test_reconnect_policy_new() {
        let policy = ReconnectPolicy::new(5, 1000, 30_000);
        assert_eq!(policy.max_attempts, 5);
        assert_eq!(policy.base_delay_ms, 1000);
        assert_eq!(policy.current_attempt, 0);
    }

    #[test]
    fn test_reconnect_policy_should_retry() {
        let mut policy = ReconnectPolicy::new(3, 1000, 30_000);
        assert!(policy.should_retry());
        policy.current_attempt = 2;
        assert!(policy.should_retry());
        policy.current_attempt = 3;
        assert!(!policy.should_retry());
    }

    #[test]
    fn test_reconnect_policy_exponential_backoff() {
        let mut policy = ReconnectPolicy::new(10, 1000, 60_000);
        // Collect delays (without jitter component, just verify they grow)
        let d0 = policy.next_delay().unwrap();
        let d1 = policy.next_delay().unwrap();
        let d2 = policy.next_delay().unwrap();

        // Base delay for attempt 0: 1000ms * 2^0 = 1000ms + jitter
        // Base delay for attempt 1: 1000ms * 2^1 = 2000ms + jitter
        // Base delay for attempt 2: 1000ms * 2^2 = 4000ms + jitter
        // Jitter adds 0-25%, so:
        assert!(d0.as_millis() >= 1000 && d0.as_millis() <= 1250);
        assert!(d1.as_millis() >= 2000 && d1.as_millis() <= 2500);
        assert!(d2.as_millis() >= 4000 && d2.as_millis() <= 5000);

        // Verify strictly increasing base
        // Even with jitter, each step doubles so d1 > d0 always holds
        assert!(d1 > d0);
        assert!(d2 > d1);
    }

    #[test]
    fn test_reconnect_policy_capped_at_max() {
        let mut policy = ReconnectPolicy::new(20, 1000, 5000);
        // Burn through attempts until we hit the cap
        for _ in 0..5 {
            let _ = policy.next_delay();
        }
        // At attempt 5: base * 2^5 = 32000 > 5000, so capped at 5000 + jitter
        let d = policy.next_delay().unwrap();
        // 5000 + up to 25% jitter = max 6250
        assert!(d.as_millis() >= 5000 && d.as_millis() <= 6250);
    }

    #[test]
    fn test_reconnect_policy_exhausted_returns_none() {
        let mut policy = ReconnectPolicy::new(2, 1000, 30_000);
        assert!(policy.next_delay().is_some()); // attempt 0
        assert!(policy.next_delay().is_some()); // attempt 1
        assert!(policy.next_delay().is_none()); // exhausted
        assert!(policy.next_delay().is_none()); // still exhausted
    }

    #[test]
    fn test_reconnect_policy_reset() {
        let mut policy = ReconnectPolicy::new(3, 1000, 30_000);
        let _ = policy.next_delay();
        let _ = policy.next_delay();
        assert_eq!(policy.current_attempt, 2);
        policy.reset();
        assert_eq!(policy.current_attempt, 0);
        assert!(policy.should_retry());
    }

    #[test]
    fn test_reconnect_policy_zero_attempts() {
        let mut policy = ReconnectPolicy::new(0, 1000, 30_000);
        assert!(!policy.should_retry());
        assert!(policy.next_delay().is_none());
    }

    #[test]
    fn test_reconnect_policy_advances_attempt_counter() {
        let mut policy = ReconnectPolicy::new(5, 1000, 30_000);
        assert_eq!(policy.current_attempt, 0);
        let _ = policy.next_delay();
        assert_eq!(policy.current_attempt, 1);
        let _ = policy.next_delay();
        assert_eq!(policy.current_attempt, 2);
    }
}
