//! Power management types for OuterLink GPU nodes.
//!
//! Defines power states, idle detection policies, Wake-on-LAN configuration,
//! GPU P-state mappings, power budgets, and power-related events. These are
//! pure data types — no actual WoL packet sending or NVML calls happen here.

use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// PowerState
// ---------------------------------------------------------------------------

/// Power state of a GPU node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PowerState {
    /// Actively processing workloads.
    Active,
    /// No current workloads but immediately available.
    Idle,
    /// Reduced clock / partial sleep — quick resume.
    LowPower,
    /// Deep sleep — requires wake sequence.
    Suspended,
    /// Machine is powered off — requires Wake-on-LAN.
    Off,
}

impl PowerState {
    /// Returns `true` if this state can serve new work immediately.
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Active | Self::Idle)
    }

    /// Returns `true` if work *can* be scheduled (possibly after a short wake).
    pub fn can_schedule(&self) -> bool {
        matches!(self, Self::Active | Self::Idle | Self::LowPower)
    }

    /// Estimated milliseconds to reach [`PowerState::Active`] from this state.
    pub fn wakeup_time_estimate_ms(&self) -> u64 {
        match self {
            Self::Active => 0,
            Self::Idle => 0,
            Self::LowPower => 100,
            Self::Suspended => 5_000,
            Self::Off => 30_000,
        }
    }
}

impl std::fmt::Display for PowerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Active => write!(f, "Active"),
            Self::Idle => write!(f, "Idle"),
            Self::LowPower => write!(f, "LowPower"),
            Self::Suspended => write!(f, "Suspended"),
            Self::Off => write!(f, "Off"),
        }
    }
}

// ---------------------------------------------------------------------------
// GpuPState
// ---------------------------------------------------------------------------

/// NVIDIA GPU performance state (P-state).
///
/// P-states are hardware-level power/performance modes. Lower numbers mean
/// higher performance and power draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuPState {
    /// Maximum performance — full clocks.
    P0,
    /// Balanced — slight clock reduction for power savings.
    P2,
    /// Low power — significant clock reduction.
    P5,
    /// Idle minimum — minimal clocks, lowest power.
    P8,
}

impl GpuPState {
    /// Approximate clock frequency reduction as a percentage relative to P0.
    pub fn clock_reduction_percent(&self) -> u32 {
        match self {
            Self::P0 => 0,
            Self::P2 => 15,
            Self::P5 => 50,
            Self::P8 => 90,
        }
    }
}

impl std::fmt::Display for GpuPState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::P0 => write!(f, "P0"),
            Self::P2 => write!(f, "P2"),
            Self::P5 => write!(f, "P5"),
            Self::P8 => write!(f, "P8"),
        }
    }
}

// ---------------------------------------------------------------------------
// IdlePolicy
// ---------------------------------------------------------------------------

/// Policy controlling automatic power-state transitions based on idle time.
#[derive(Debug, Clone)]
pub struct IdlePolicy {
    /// Seconds of inactivity before transitioning from Active to Idle.
    pub idle_threshold_secs: u64,
    /// Seconds of inactivity before transitioning from Idle to LowPower.
    pub low_power_after_secs: u64,
    /// Seconds of inactivity before transitioning from LowPower to Suspended.
    pub suspend_after_secs: u64,
    /// Seconds of inactivity before transitioning from Suspended to Off.
    /// `None` means never auto-power-off.
    pub off_after_secs: Option<u64>,
}

impl Default for IdlePolicy {
    fn default() -> Self {
        Self {
            idle_threshold_secs: 300,
            low_power_after_secs: 900,
            suspend_after_secs: 3600,
            off_after_secs: None,
        }
    }
}

// ---------------------------------------------------------------------------
// WolConfig
// ---------------------------------------------------------------------------

/// Wake-on-LAN configuration for a remote node.
#[derive(Debug, Clone)]
pub struct WolConfig {
    /// MAC address of the target NIC (e.g. "AA:BB:CC:DD:EE:FF").
    pub mac_address: String,
    /// Broadcast address to send magic packets to.
    pub broadcast_address: String,
    /// UDP port for Wake-on-LAN magic packets.
    pub magic_packet_port: u16,
    /// Number of times to re-send the magic packet if no acknowledgment.
    pub retry_count: u32,
    /// Milliseconds between retries.
    pub retry_delay_ms: u64,
}

impl WolConfig {
    /// Create a new WoL config with the given MAC address and sensible defaults.
    pub fn new(mac_address: impl Into<String>) -> Self {
        Self {
            mac_address: mac_address.into(),
            broadcast_address: "255.255.255.255:9".to_string(),
            magic_packet_port: 9,
            retry_count: 3,
            retry_delay_ms: 2000,
        }
    }
}

// ---------------------------------------------------------------------------
// PowerEvent
// ---------------------------------------------------------------------------

/// Events related to power-state changes and wake-up operations.
#[derive(Debug, Clone)]
pub enum PowerEvent {
    /// A GPU has become idle (no active kernels or transfers).
    GpuBecameIdle { gpu_index: u32 },
    /// A GPU has started processing a workload.
    GpuWorkloadStarted { gpu_index: u32 },
    /// The entire node has been idle long enough to trigger a timeout.
    NodeIdleTimeout,
    /// A Wake-on-LAN magic packet was sent to the target.
    WakeOnLanSent { mac: String },
    /// The target acknowledged the WoL (came online).
    WakeOnLanAcked { mac: String, latency_ms: u64 },
    /// A node's power state changed.
    PowerStateChanged { from: PowerState, to: PowerState },
}

impl PowerEvent {
    /// Returns `true` if this event represents something waking up / starting work.
    pub fn is_wakeup_event(&self) -> bool {
        matches!(
            self,
            Self::GpuWorkloadStarted { .. }
                | Self::WakeOnLanAcked { .. }
                | Self::PowerStateChanged { to: PowerState::Active, .. }
        )
    }
}

// ---------------------------------------------------------------------------
// IdleDetector
// ---------------------------------------------------------------------------

/// Tracks activity and determines the current power state based on idle time.
///
/// Call [`IdleDetector::record_activity`] whenever work happens. Call
/// [`IdleDetector::check_state`] periodically to get the appropriate
/// power-state transition.
#[derive(Debug, Clone)]
pub struct IdleDetector {
    /// Timestamp of the last recorded activity, or `None` if no activity yet.
    last_activity: Option<Instant>,
    /// Current power state as determined by the most recent `check_state` call.
    current_state: PowerState,
    /// Policy governing idle-time thresholds.
    policy: IdlePolicy,
}

impl IdleDetector {
    /// Create a new detector with the given policy, starting in `Active` state.
    pub fn new(policy: IdlePolicy) -> Self {
        Self {
            last_activity: None,
            current_state: PowerState::Active,
            policy,
        }
    }

    /// Record that activity occurred at the given instant.
    pub fn record_activity(&mut self, now: Instant) {
        self.last_activity = Some(now);
        self.current_state = PowerState::Active;
    }

    /// Evaluate how long the node has been idle and return the appropriate
    /// power state. Also updates `current_state` internally.
    pub fn check_state(&mut self, now: Instant) -> PowerState {
        let idle_secs = match self.idle_duration(now) {
            Some(d) => d.as_secs(),
            None => {
                // No activity ever recorded — remain in current state.
                return self.current_state;
            }
        };

        let new_state = if let Some(off_secs) = self.policy.off_after_secs {
            if idle_secs >= off_secs {
                PowerState::Off
            } else if idle_secs >= self.policy.suspend_after_secs {
                PowerState::Suspended
            } else if idle_secs >= self.policy.low_power_after_secs {
                PowerState::LowPower
            } else if idle_secs >= self.policy.idle_threshold_secs {
                PowerState::Idle
            } else {
                PowerState::Active
            }
        } else if idle_secs >= self.policy.suspend_after_secs {
            PowerState::Suspended
        } else if idle_secs >= self.policy.low_power_after_secs {
            PowerState::LowPower
        } else if idle_secs >= self.policy.idle_threshold_secs {
            PowerState::Idle
        } else {
            PowerState::Active
        };

        self.current_state = new_state;
        new_state
    }

    /// Duration since the last recorded activity, or `None` if no activity
    /// has ever been recorded.
    pub fn idle_duration(&self, now: Instant) -> Option<Duration> {
        self.last_activity.map(|last| now.duration_since(last))
    }

    /// The current power state (as of the last `check_state` or `record_activity` call).
    pub fn current_state(&self) -> PowerState {
        self.current_state
    }
}

// ---------------------------------------------------------------------------
// PowerBudget
// ---------------------------------------------------------------------------

/// Tracks power consumption against a configured budget.
#[derive(Debug, Clone)]
pub struct PowerBudget {
    /// Maximum allowed power draw in watts.
    pub max_watts: f64,
    /// Current total power draw in watts.
    pub current_watts: f64,
    /// Per-GPU power draw in watts.
    pub gpu_watts: Vec<f64>,
}

impl PowerBudget {
    /// Current utilization as a fraction (0.0 .. 1.0+).
    pub fn utilization(&self) -> f64 {
        if self.max_watts <= 0.0 {
            return 0.0;
        }
        self.current_watts / self.max_watts
    }

    /// Remaining watts before hitting the budget cap.
    /// Negative values mean over budget.
    pub fn headroom_watts(&self) -> f64 {
        self.max_watts - self.current_watts
    }

    /// Returns `true` if current draw exceeds the budget.
    pub fn is_over_budget(&self) -> bool {
        self.current_watts > self.max_watts
    }
}

// ---------------------------------------------------------------------------
// Formatting helper
// ---------------------------------------------------------------------------

/// Format a power state and duration into a human-readable string.
///
/// Examples: `"Active for 2h 15m"`, `"Idle for 5m 30s"`, `"Off for 0s"`.
pub fn format_power_state_duration(state: PowerState, duration_secs: u64) -> String {
    let hours = duration_secs / 3600;
    let minutes = (duration_secs % 3600) / 60;
    let seconds = duration_secs % 60;

    let duration_str = if hours > 0 {
        format!("{}h {}m", hours, minutes)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, seconds)
    } else {
        format!("{}s", seconds)
    };

    format!("{} for {}", state, duration_str)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- PowerState ---------------------------------------------------------

    #[test]
    fn power_state_is_available() {
        assert!(PowerState::Active.is_available());
        assert!(PowerState::Idle.is_available());
        assert!(!PowerState::LowPower.is_available());
        assert!(!PowerState::Suspended.is_available());
        assert!(!PowerState::Off.is_available());
    }

    #[test]
    fn power_state_can_schedule() {
        assert!(PowerState::Active.can_schedule());
        assert!(PowerState::Idle.can_schedule());
        assert!(PowerState::LowPower.can_schedule());
        assert!(!PowerState::Suspended.can_schedule());
        assert!(!PowerState::Off.can_schedule());
    }

    #[test]
    fn power_state_wakeup_estimates() {
        assert_eq!(PowerState::Active.wakeup_time_estimate_ms(), 0);
        assert_eq!(PowerState::Idle.wakeup_time_estimate_ms(), 0);
        assert_eq!(PowerState::LowPower.wakeup_time_estimate_ms(), 100);
        assert_eq!(PowerState::Suspended.wakeup_time_estimate_ms(), 5_000);
        assert_eq!(PowerState::Off.wakeup_time_estimate_ms(), 30_000);
    }

    #[test]
    fn power_state_display() {
        assert_eq!(PowerState::Active.to_string(), "Active");
        assert_eq!(PowerState::Off.to_string(), "Off");
    }

    // -- GpuPState ----------------------------------------------------------

    #[test]
    fn gpu_pstate_clock_reduction() {
        assert_eq!(GpuPState::P0.clock_reduction_percent(), 0);
        assert_eq!(GpuPState::P2.clock_reduction_percent(), 15);
        assert_eq!(GpuPState::P5.clock_reduction_percent(), 50);
        assert_eq!(GpuPState::P8.clock_reduction_percent(), 90);
    }

    #[test]
    fn gpu_pstate_display() {
        assert_eq!(GpuPState::P0.to_string(), "P0");
        assert_eq!(GpuPState::P8.to_string(), "P8");
    }

    #[test]
    fn gpu_pstate_equality() {
        assert_eq!(GpuPState::P2, GpuPState::P2);
        assert_ne!(GpuPState::P0, GpuPState::P8);
    }

    // -- IdlePolicy ---------------------------------------------------------

    #[test]
    fn idle_policy_defaults() {
        let p = IdlePolicy::default();
        assert_eq!(p.idle_threshold_secs, 300);
        assert_eq!(p.low_power_after_secs, 900);
        assert_eq!(p.suspend_after_secs, 3600);
        assert_eq!(p.off_after_secs, None);
    }

    #[test]
    fn idle_policy_custom() {
        let p = IdlePolicy {
            idle_threshold_secs: 60,
            low_power_after_secs: 120,
            suspend_after_secs: 300,
            off_after_secs: Some(600),
        };
        assert_eq!(p.idle_threshold_secs, 60);
        assert_eq!(p.off_after_secs, Some(600));
    }

    // -- WolConfig ----------------------------------------------------------

    #[test]
    fn wol_config_defaults() {
        let cfg = WolConfig::new("AA:BB:CC:DD:EE:FF");
        assert_eq!(cfg.mac_address, "AA:BB:CC:DD:EE:FF");
        assert_eq!(cfg.broadcast_address, "255.255.255.255:9");
        assert_eq!(cfg.magic_packet_port, 9);
        assert_eq!(cfg.retry_count, 3);
        assert_eq!(cfg.retry_delay_ms, 2000);
    }

    #[test]
    fn wol_config_custom_mac() {
        let cfg = WolConfig::new("11:22:33:44:55:66");
        assert_eq!(cfg.mac_address, "11:22:33:44:55:66");
    }

    // -- PowerEvent ---------------------------------------------------------

    #[test]
    fn power_event_is_wakeup_workload_started() {
        let e = PowerEvent::GpuWorkloadStarted { gpu_index: 0 };
        assert!(e.is_wakeup_event());
    }

    #[test]
    fn power_event_is_wakeup_wol_acked() {
        let e = PowerEvent::WakeOnLanAcked {
            mac: "AA:BB:CC:DD:EE:FF".into(),
            latency_ms: 1500,
        };
        assert!(e.is_wakeup_event());
    }

    #[test]
    fn power_event_is_wakeup_state_change_to_active() {
        let e = PowerEvent::PowerStateChanged {
            from: PowerState::Suspended,
            to: PowerState::Active,
        };
        assert!(e.is_wakeup_event());
    }

    #[test]
    fn power_event_is_not_wakeup_idle() {
        let e = PowerEvent::GpuBecameIdle { gpu_index: 0 };
        assert!(!e.is_wakeup_event());
    }

    #[test]
    fn power_event_is_not_wakeup_wol_sent() {
        let e = PowerEvent::WakeOnLanSent {
            mac: "AA:BB:CC:DD:EE:FF".into(),
        };
        assert!(!e.is_wakeup_event());
    }

    #[test]
    fn power_event_is_not_wakeup_node_idle_timeout() {
        let e = PowerEvent::NodeIdleTimeout;
        assert!(!e.is_wakeup_event());
    }

    #[test]
    fn power_event_state_change_to_suspended_not_wakeup() {
        let e = PowerEvent::PowerStateChanged {
            from: PowerState::Active,
            to: PowerState::Suspended,
        };
        assert!(!e.is_wakeup_event());
    }

    // -- IdleDetector -------------------------------------------------------

    #[test]
    fn idle_detector_starts_active() {
        let det = IdleDetector::new(IdlePolicy::default());
        assert_eq!(det.current_state(), PowerState::Active);
    }

    #[test]
    fn idle_detector_no_activity_stays_current() {
        let mut det = IdleDetector::new(IdlePolicy::default());
        let now = Instant::now();
        // Never called record_activity, so check_state keeps current state.
        assert_eq!(det.check_state(now), PowerState::Active);
    }

    #[test]
    fn idle_detector_active_before_threshold() {
        let policy = IdlePolicy {
            idle_threshold_secs: 10,
            low_power_after_secs: 20,
            suspend_after_secs: 30,
            off_after_secs: None,
        };
        let mut det = IdleDetector::new(policy);
        let start = Instant::now();
        det.record_activity(start);
        // 5 seconds later — still active
        let state = det.check_state(start + Duration::from_secs(5));
        assert_eq!(state, PowerState::Active);
    }

    #[test]
    fn idle_detector_transitions_to_idle() {
        let policy = IdlePolicy {
            idle_threshold_secs: 10,
            low_power_after_secs: 20,
            suspend_after_secs: 30,
            off_after_secs: None,
        };
        let mut det = IdleDetector::new(policy);
        let start = Instant::now();
        det.record_activity(start);
        let state = det.check_state(start + Duration::from_secs(10));
        assert_eq!(state, PowerState::Idle);
    }

    #[test]
    fn idle_detector_transitions_to_low_power() {
        let policy = IdlePolicy {
            idle_threshold_secs: 10,
            low_power_after_secs: 20,
            suspend_after_secs: 30,
            off_after_secs: None,
        };
        let mut det = IdleDetector::new(policy);
        let start = Instant::now();
        det.record_activity(start);
        let state = det.check_state(start + Duration::from_secs(25));
        assert_eq!(state, PowerState::LowPower);
    }

    #[test]
    fn idle_detector_transitions_to_suspended() {
        let policy = IdlePolicy {
            idle_threshold_secs: 10,
            low_power_after_secs: 20,
            suspend_after_secs: 30,
            off_after_secs: None,
        };
        let mut det = IdleDetector::new(policy);
        let start = Instant::now();
        det.record_activity(start);
        let state = det.check_state(start + Duration::from_secs(35));
        assert_eq!(state, PowerState::Suspended);
    }

    #[test]
    fn idle_detector_never_goes_off_when_none() {
        let policy = IdlePolicy {
            idle_threshold_secs: 10,
            low_power_after_secs: 20,
            suspend_after_secs: 30,
            off_after_secs: None,
        };
        let mut det = IdleDetector::new(policy);
        let start = Instant::now();
        det.record_activity(start);
        // Even after a very long time, stays Suspended (never Off).
        let state = det.check_state(start + Duration::from_secs(100_000));
        assert_eq!(state, PowerState::Suspended);
    }

    #[test]
    fn idle_detector_transitions_to_off_when_configured() {
        let policy = IdlePolicy {
            idle_threshold_secs: 10,
            low_power_after_secs: 20,
            suspend_after_secs: 30,
            off_after_secs: Some(60),
        };
        let mut det = IdleDetector::new(policy);
        let start = Instant::now();
        det.record_activity(start);
        let state = det.check_state(start + Duration::from_secs(65));
        assert_eq!(state, PowerState::Off);
    }

    #[test]
    fn idle_detector_activity_resets_to_active() {
        let policy = IdlePolicy {
            idle_threshold_secs: 10,
            low_power_after_secs: 20,
            suspend_after_secs: 30,
            off_after_secs: None,
        };
        let mut det = IdleDetector::new(policy);
        let start = Instant::now();
        det.record_activity(start);
        // Go idle
        assert_eq!(
            det.check_state(start + Duration::from_secs(15)),
            PowerState::Idle
        );
        // New activity resets to active
        det.record_activity(start + Duration::from_secs(15));
        assert_eq!(det.current_state(), PowerState::Active);
        // Still active shortly after
        assert_eq!(
            det.check_state(start + Duration::from_secs(16)),
            PowerState::Active
        );
    }

    #[test]
    fn idle_detector_idle_duration_none_without_activity() {
        let det = IdleDetector::new(IdlePolicy::default());
        assert!(det.idle_duration(Instant::now()).is_none());
    }

    #[test]
    fn idle_detector_idle_duration_with_activity() {
        let mut det = IdleDetector::new(IdlePolicy::default());
        let start = Instant::now();
        det.record_activity(start);
        let dur = det.idle_duration(start + Duration::from_secs(42));
        assert_eq!(dur.unwrap().as_secs(), 42);
    }

    // -- PowerBudget --------------------------------------------------------

    #[test]
    fn power_budget_utilization() {
        let b = PowerBudget {
            max_watts: 1000.0,
            current_watts: 750.0,
            gpu_watts: vec![350.0, 400.0],
        };
        assert!((b.utilization() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn power_budget_headroom() {
        let b = PowerBudget {
            max_watts: 1000.0,
            current_watts: 750.0,
            gpu_watts: vec![350.0, 400.0],
        };
        assert!((b.headroom_watts() - 250.0).abs() < f64::EPSILON);
    }

    #[test]
    fn power_budget_over_budget() {
        let b = PowerBudget {
            max_watts: 500.0,
            current_watts: 600.0,
            gpu_watts: vec![300.0, 300.0],
        };
        assert!(b.is_over_budget());
        assert!(b.headroom_watts() < 0.0);
    }

    #[test]
    fn power_budget_under_budget() {
        let b = PowerBudget {
            max_watts: 1000.0,
            current_watts: 999.0,
            gpu_watts: vec![999.0],
        };
        assert!(!b.is_over_budget());
    }

    #[test]
    fn power_budget_zero_max_watts() {
        let b = PowerBudget {
            max_watts: 0.0,
            current_watts: 100.0,
            gpu_watts: vec![100.0],
        };
        // utilization returns 0.0 when max is zero to avoid division by zero
        assert_eq!(b.utilization(), 0.0);
    }

    // -- format_power_state_duration ----------------------------------------

    #[test]
    fn format_hours_and_minutes() {
        let s = format_power_state_duration(PowerState::Active, 8100);
        assert_eq!(s, "Active for 2h 15m");
    }

    #[test]
    fn format_minutes_and_seconds() {
        let s = format_power_state_duration(PowerState::Idle, 330);
        assert_eq!(s, "Idle for 5m 30s");
    }

    #[test]
    fn format_seconds_only() {
        let s = format_power_state_duration(PowerState::Off, 0);
        assert_eq!(s, "Off for 0s");
    }

    #[test]
    fn format_exactly_one_hour() {
        let s = format_power_state_duration(PowerState::Suspended, 3600);
        assert_eq!(s, "Suspended for 1h 0m");
    }

    #[test]
    fn format_low_power_seconds() {
        let s = format_power_state_duration(PowerState::LowPower, 45);
        assert_eq!(s, "LowPower for 45s");
    }
}
