//! Health monitoring types and GPU state machine.
//!
//! Defines the health states for GPUs, NICs, and nodes, plus the
//! state machine transitions triggered by monitoring events.

use std::time::{Duration, Instant};

/// GPU health state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuHealthState {
    /// GPU is functioning normally.
    Available,
    /// GPU is throttling (thermal or power).
    Throttled,
    /// GPU is temporarily unavailable (driver error, context reset).
    Unavailable,
    /// GPU has a permanent failure (Xid 43/48/64/79/95).
    Failed,
}

impl GpuHealthState {
    /// Whether this GPU can accept new work.
    pub fn can_schedule(&self) -> bool {
        matches!(self, Self::Available | Self::Throttled)
    }

    /// Whether this GPU is considered alive (not permanently failed).
    pub fn is_alive(&self) -> bool {
        !matches!(self, Self::Failed)
    }
}

/// NIC health state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NicHealthState {
    /// Link is up and healthy.
    Up,
    /// Link is degraded (high error rate, reduced speed).
    Degraded,
    /// Link is down.
    Down,
}

/// Node health state (aggregate).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeHealthState {
    /// Node is healthy and reachable.
    Healthy,
    /// Node heartbeat is suspect (phi > 8).
    Suspect,
    /// Node is unreachable (phi > 12).
    Unreachable,
}

/// Xid error severity classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XidSeverity {
    /// Informational, no action needed (Xid 57, 63, 94).
    Info,
    /// Warning, track frequency (Xid 31).
    Warning,
    /// Critical, immediate GPU removal (Xid 43, 48, 64, 79, 95).
    Critical,
}

/// Classify an NVIDIA Xid error code.
pub fn classify_xid(xid: u32) -> XidSeverity {
    match xid {
        // Informational
        57 | 63 | 94 => XidSeverity::Info,
        // Warning - track frequency
        13 | 31 | 32 | 61 | 69 => XidSeverity::Warning,
        // Critical - immediate removal
        43 | 48 | 64 | 79 | 95 => XidSeverity::Critical,
        // Unknown Xid - treat as warning
        _ => XidSeverity::Warning,
    }
}

/// GPU thermal thresholds (Celsius).
#[derive(Debug, Clone, Copy)]
pub struct ThermalThresholds {
    /// Temperature at which to log a warning and reduce scheduling priority.
    pub throttle_warn: f64,
    /// Temperature at which to stop scheduling new work.
    pub throttle_stop: f64,
    /// Temperature at which to migrate existing work away.
    pub migrate: f64,
    /// Emergency shutdown temperature.
    pub emergency: f64,
}

impl Default for ThermalThresholds {
    fn default() -> Self {
        Self {
            throttle_warn: 80.0,
            throttle_stop: 85.0,
            migrate: 90.0,
            emergency: 95.0,
        }
    }
}

/// GPU health snapshot at a point in time.
#[derive(Debug, Clone)]
pub struct GpuHealthSnapshot {
    /// GPU index.
    pub gpu_index: u32,
    /// Current health state.
    pub state: GpuHealthState,
    /// GPU temperature in Celsius.
    pub temperature: f64,
    /// GPU utilization 0-100%.
    pub utilization: u32,
    /// VRAM used in bytes.
    pub vram_used: u64,
    /// VRAM total in bytes.
    pub vram_total: u64,
    /// Power draw in watts.
    pub power_watts: f64,
    /// Number of Xid errors since last check.
    pub xid_error_count: u32,
    /// Last Xid error code (0 = none).
    pub last_xid: u32,
    /// When this snapshot was taken.
    pub timestamp: Instant,
}

impl GpuHealthSnapshot {
    /// Determine health state from thermal readings.
    pub fn thermal_state(&self, thresholds: &ThermalThresholds) -> GpuHealthState {
        if self.temperature >= thresholds.emergency {
            GpuHealthState::Failed
        } else if self.temperature >= thresholds.migrate {
            GpuHealthState::Unavailable
        } else if self.temperature >= thresholds.throttle_warn {
            GpuHealthState::Throttled
        } else {
            GpuHealthState::Available
        }
    }
}

/// Phi accrual failure detector state.
///
/// Tracks heartbeat arrival times and computes the phi value,
/// which represents the suspicion level that the monitored node has failed.
#[derive(Debug, Clone)]
pub struct PhiAccrualDetector {
    /// Recent heartbeat intervals in milliseconds.
    intervals: Vec<f64>,
    /// Maximum number of intervals to track.
    max_samples: usize,
    /// Time of last heartbeat.
    last_heartbeat: Option<Instant>,
    /// Phi threshold for "suspect" state.
    pub suspect_threshold: f64,
    /// Phi threshold for "failed" state.
    pub failed_threshold: f64,
}

impl PhiAccrualDetector {
    /// Create a new detector.
    pub fn new(max_samples: usize, suspect_threshold: f64, failed_threshold: f64) -> Self {
        Self {
            intervals: Vec::with_capacity(max_samples),
            max_samples,
            last_heartbeat: None,
            suspect_threshold,
            failed_threshold,
        }
    }

    /// Record a heartbeat arrival.
    pub fn heartbeat(&mut self) {
        self.heartbeat_at(Instant::now());
    }

    /// Record a heartbeat with a specific timestamp (for testing).
    pub fn heartbeat_at(&mut self, time: Instant) {
        if let Some(last) = self.last_heartbeat {
            let interval = time.duration_since(last).as_secs_f64() * 1000.0;
            if self.intervals.len() >= self.max_samples {
                self.intervals.remove(0);
            }
            self.intervals.push(interval);
        }
        self.last_heartbeat = Some(time);
    }

    /// Compute the current phi value.
    /// Returns None if not enough data (need at least 2 intervals).
    pub fn phi(&self) -> Option<f64> {
        if self.intervals.len() < 2 {
            return None;
        }
        let last = self.last_heartbeat?;
        let elapsed = Instant::now().duration_since(last).as_secs_f64() * 1000.0;
        Some(self.compute_phi(elapsed))
    }

    /// Compute phi at a specific time offset from last heartbeat (for testing).
    pub fn phi_at_offset(&self, offset: Duration) -> Option<f64> {
        if self.intervals.len() < 2 {
            return None;
        }
        let elapsed = offset.as_secs_f64() * 1000.0;
        Some(self.compute_phi(elapsed))
    }

    /// Internal phi computation from elapsed milliseconds since last heartbeat.
    fn compute_phi(&self, elapsed_ms: f64) -> f64 {
        let mean = self.mean();
        let std_dev = self.std_dev();

        // Phi = -log10(Q(y)) where Q is the complementary normal CDF
        // y = (elapsed - mean) / std_dev
        let y = (elapsed_ms - mean) / std_dev.max(1.0);
        let phi = log10_normal_cdf_complement(y);
        phi.max(0.0)
    }

    /// Get the current node health state based on phi.
    pub fn node_state(&self) -> NodeHealthState {
        match self.phi() {
            None => NodeHealthState::Healthy, // Not enough data, assume healthy
            Some(phi) if phi >= self.failed_threshold => NodeHealthState::Unreachable,
            Some(phi) if phi >= self.suspect_threshold => NodeHealthState::Suspect,
            _ => NodeHealthState::Healthy,
        }
    }

    /// Mean of recorded intervals.
    fn mean(&self) -> f64 {
        if self.intervals.is_empty() {
            return 1000.0; // Default 1s
        }
        self.intervals.iter().sum::<f64>() / self.intervals.len() as f64
    }

    /// Standard deviation of recorded intervals.
    fn std_dev(&self) -> f64 {
        if self.intervals.len() < 2 {
            return 100.0; // Default 100ms
        }
        let mean = self.mean();
        let variance = self.intervals
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / (self.intervals.len() - 1) as f64;
        variance.sqrt()
    }

    /// Number of recorded intervals.
    pub fn sample_count(&self) -> usize {
        self.intervals.len()
    }

    /// Whether enough data has been collected for reliable detection.
    pub fn is_ready(&self) -> bool {
        self.intervals.len() >= 2
    }
}

/// Compute -log10(Q(y)) where Q(y) = 1 - Phi(y) is the complementary
/// standard normal CDF. Uses the Abramowitz and Stegun rational approximation
/// for the complementary error function.
fn log10_normal_cdf_complement(y: f64) -> f64 {
    if y <= -40.0 {
        return 0.0; // Very negative = phi is 0
    }
    if y >= 40.0 {
        return 40.0; // Cap at reasonable max
    }

    // Rational approximation of Q(y) for the standard normal.
    // Q(y) = 1 - Phi(y) where Phi is the CDF.
    let abs_y = y.abs();
    let t = 1.0 / (1.0 + 0.2316419 * abs_y);
    let d = 0.398_942_280_401_432_7; // 1/sqrt(2*pi)
    let p = d * (-abs_y * abs_y / 2.0).exp()
        * (t
            * (0.319_381_530
                + t * (-0.356_563_782
                    + t * (1.781_477_937
                        + t * (-1.821_255_978 + t * 1.330_274_429)))));

    let q = if y >= 0.0 { p } else { 1.0 - p };

    if q <= 1e-100 {
        100.0 // -log10(very small) = very large
    } else {
        -q.log10()
    }
}

/// System memory pressure level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryPressure {
    /// >30% available -- normal.
    Normal,
    /// 15-30% available -- caution.
    Warning,
    /// 5-15% available -- reduce allocations.
    Critical,
    /// <5% available -- emergency, free memory.
    Emergency,
}

impl MemoryPressure {
    /// Classify from percentage of available memory.
    pub fn from_available_percent(percent: f64) -> Self {
        if percent > 30.0 {
            Self::Normal
        } else if percent > 15.0 {
            Self::Warning
        } else if percent > 5.0 {
            Self::Critical
        } else {
            Self::Emergency
        }
    }
}
