//! Tests for the health monitoring module.

use std::time::{Duration, Instant};

use outerlink_common::health::{
    classify_xid, GpuHealthSnapshot, GpuHealthState, HostMemoryPressure, NicHealthState,
    NodeHealthState, PhiAccrualDetector, ThermalThresholds, XidSeverity,
};

// ---------------------------------------------------------------------------
// GpuHealthState
// ---------------------------------------------------------------------------

#[test]
fn gpu_available_can_schedule() {
    assert!(GpuHealthState::Available.can_schedule());
}

#[test]
fn gpu_throttled_can_schedule() {
    assert!(GpuHealthState::Throttled.can_schedule());
}

#[test]
fn gpu_unavailable_cannot_schedule() {
    assert!(!GpuHealthState::Unavailable.can_schedule());
}

#[test]
fn gpu_failed_cannot_schedule() {
    assert!(!GpuHealthState::Failed.can_schedule());
}

#[test]
fn gpu_available_is_alive() {
    assert!(GpuHealthState::Available.is_alive());
}

#[test]
fn gpu_throttled_is_alive() {
    assert!(GpuHealthState::Throttled.is_alive());
}

#[test]
fn gpu_unavailable_is_alive() {
    assert!(GpuHealthState::Unavailable.is_alive());
}

#[test]
fn gpu_failed_is_not_alive() {
    assert!(!GpuHealthState::Failed.is_alive());
}

// ---------------------------------------------------------------------------
// NicHealthState
// ---------------------------------------------------------------------------

#[test]
fn nic_states_are_distinct() {
    let up = NicHealthState::Up;
    let degraded = NicHealthState::Degraded;
    let down = NicHealthState::Down;
    assert_ne!(up, degraded);
    assert_ne!(degraded, down);
    assert_ne!(up, down);
}

#[test]
fn nic_state_debug_and_clone() {
    let state = NicHealthState::Up;
    let cloned = state;
    assert_eq!(format!("{:?}", cloned), "Up");
}

// ---------------------------------------------------------------------------
// NodeHealthState
// ---------------------------------------------------------------------------

#[test]
fn node_states_are_distinct() {
    let healthy = NodeHealthState::Healthy;
    let suspect = NodeHealthState::Suspect;
    let unreachable = NodeHealthState::Unreachable;
    assert_ne!(healthy, suspect);
    assert_ne!(suspect, unreachable);
    assert_ne!(healthy, unreachable);
}

#[test]
fn node_state_debug_and_clone() {
    let state = NodeHealthState::Healthy;
    let cloned = state;
    assert_eq!(format!("{:?}", cloned), "Healthy");
}

// ---------------------------------------------------------------------------
// XidSeverity / classify_xid
// ---------------------------------------------------------------------------

#[test]
fn xid_info_codes() {
    for code in [57, 63, 94] {
        assert_eq!(
            classify_xid(code),
            XidSeverity::Info,
            "Xid {} should be Info",
            code
        );
    }
}

#[test]
fn xid_warning_codes() {
    for code in [13, 31, 32, 61, 69] {
        assert_eq!(
            classify_xid(code),
            XidSeverity::Warning,
            "Xid {} should be Warning",
            code
        );
    }
}

#[test]
fn xid_critical_codes() {
    for code in [43, 48, 64, 79, 95] {
        assert_eq!(
            classify_xid(code),
            XidSeverity::Critical,
            "Xid {} should be Critical",
            code
        );
    }
}

#[test]
fn xid_unknown_code_is_warning() {
    assert_eq!(classify_xid(999), XidSeverity::Warning);
    assert_eq!(classify_xid(0), XidSeverity::Warning);
    assert_eq!(classify_xid(1), XidSeverity::Warning);
}

// ---------------------------------------------------------------------------
// ThermalThresholds
// ---------------------------------------------------------------------------

#[test]
fn thermal_defaults() {
    let t = ThermalThresholds::default();
    assert_eq!(t.throttle_warn, 80.0);
    assert_eq!(t.throttle_stop, 85.0);
    assert_eq!(t.migrate, 90.0);
    assert_eq!(t.emergency, 95.0);
}

// ---------------------------------------------------------------------------
// GpuHealthSnapshot: thermal_state
// ---------------------------------------------------------------------------

fn make_snapshot(temperature: f64) -> GpuHealthSnapshot {
    GpuHealthSnapshot {
        gpu_index: 0,
        state: GpuHealthState::Available,
        temperature,
        utilization: 50,
        vram_used: 1_000_000,
        vram_total: 10_000_000,
        power_watts: 200.0,
        xid_error_count: 0,
        last_xid: 0,
        timestamp: Instant::now(),
    }
}

#[test]
fn thermal_state_normal() {
    let snap = make_snapshot(70.0);
    let thresholds = ThermalThresholds::default();
    assert_eq!(snap.thermal_state(&thresholds), GpuHealthState::Available);
}

#[test]
fn thermal_state_at_throttle_warn() {
    let snap = make_snapshot(80.0);
    let thresholds = ThermalThresholds::default();
    assert_eq!(snap.thermal_state(&thresholds), GpuHealthState::Throttled);
}

#[test]
fn thermal_state_at_throttle_stop() {
    let snap = make_snapshot(85.0);
    let thresholds = ThermalThresholds::default();
    // 85.0 == throttle_stop: stop scheduling, begin draining -> Unavailable
    assert_eq!(snap.thermal_state(&thresholds), GpuHealthState::Unavailable);
}

#[test]
fn thermal_state_between_warn_and_stop() {
    let snap = make_snapshot(82.0);
    let thresholds = ThermalThresholds::default();
    // 80 <= 82 < 85: throttled, reduced priority
    assert_eq!(snap.thermal_state(&thresholds), GpuHealthState::Throttled);
}

#[test]
fn thermal_state_at_migrate() {
    let snap = make_snapshot(90.0);
    let thresholds = ThermalThresholds::default();
    assert_eq!(
        snap.thermal_state(&thresholds),
        GpuHealthState::Unavailable
    );
}

#[test]
fn thermal_state_at_emergency() {
    let snap = make_snapshot(95.0);
    let thresholds = ThermalThresholds::default();
    assert_eq!(snap.thermal_state(&thresholds), GpuHealthState::Failed);
}

#[test]
fn thermal_state_above_emergency() {
    let snap = make_snapshot(100.0);
    let thresholds = ThermalThresholds::default();
    assert_eq!(snap.thermal_state(&thresholds), GpuHealthState::Failed);
}

// ---------------------------------------------------------------------------
// PhiAccrualDetector
// ---------------------------------------------------------------------------

#[test]
fn phi_detector_new_has_no_samples() {
    let det = PhiAccrualDetector::new(100, 8.0, 12.0);
    assert_eq!(det.sample_count(), 0);
    assert!(!det.is_ready());
}

#[test]
fn phi_detector_not_ready_after_one_heartbeat() {
    let mut det = PhiAccrualDetector::new(100, 8.0, 12.0);
    det.heartbeat();
    assert_eq!(det.sample_count(), 0); // Need 2 heartbeats for 1 interval
    assert!(!det.is_ready());
}

#[test]
fn phi_detector_ready_after_three_heartbeats() {
    let mut det = PhiAccrualDetector::new(100, 8.0, 12.0);
    let t0 = Instant::now();
    det.heartbeat_at(t0);
    det.heartbeat_at(t0 + Duration::from_secs(1));
    det.heartbeat_at(t0 + Duration::from_secs(2));
    assert_eq!(det.sample_count(), 2);
    assert!(det.is_ready());
}

#[test]
fn phi_none_when_not_ready() {
    let det = PhiAccrualDetector::new(100, 8.0, 12.0);
    assert!(det.phi().is_none());
}

#[test]
fn phi_low_with_recent_heartbeat() {
    // Use phi_at_offset for deterministic testing
    let mut det = PhiAccrualDetector::new(100, 8.0, 12.0);
    let t0 = Instant::now();
    // Record regular 1-second heartbeats
    for i in 0..10 {
        det.heartbeat_at(t0 + Duration::from_millis(i * 1000));
    }
    // Check phi at a small offset (just arrived)
    let phi = det.phi_at_offset(Duration::from_millis(100)).unwrap();
    assert!(phi < 1.0, "phi should be low right after heartbeat, got {}", phi);
}

#[test]
fn phi_high_with_missing_heartbeat() {
    let mut det = PhiAccrualDetector::new(100, 8.0, 12.0);
    let t0 = Instant::now();
    for i in 0..10 {
        det.heartbeat_at(t0 + Duration::from_millis(i * 1000));
    }
    // Check phi at 30 seconds (way past expected 1s interval)
    let phi = det.phi_at_offset(Duration::from_secs(30)).unwrap();
    assert!(
        phi > 8.0,
        "phi should be high with long gap, got {}",
        phi
    );
}

#[test]
fn phi_detector_max_samples() {
    let mut det = PhiAccrualDetector::new(5, 8.0, 12.0);
    let t0 = Instant::now();
    for i in 0..20 {
        det.heartbeat_at(t0 + Duration::from_millis(i * 1000));
    }
    assert_eq!(det.sample_count(), 5);
}

#[test]
fn node_state_healthy_when_not_ready() {
    let det = PhiAccrualDetector::new(100, 8.0, 12.0);
    assert_eq!(det.node_state(), NodeHealthState::Healthy);
}

// ---------------------------------------------------------------------------
// HostMemoryPressure
// ---------------------------------------------------------------------------

#[test]
fn memory_pressure_normal() {
    assert_eq!(
        HostMemoryPressure::from_available_percent(50.0),
        HostMemoryPressure::Normal
    );
    assert_eq!(
        HostMemoryPressure::from_available_percent(31.0),
        HostMemoryPressure::Normal
    );
}

#[test]
fn memory_pressure_warning() {
    assert_eq!(
        HostMemoryPressure::from_available_percent(30.0),
        HostMemoryPressure::Warning
    );
    assert_eq!(
        HostMemoryPressure::from_available_percent(16.0),
        HostMemoryPressure::Warning
    );
}

#[test]
fn memory_pressure_critical() {
    assert_eq!(
        HostMemoryPressure::from_available_percent(15.0),
        HostMemoryPressure::Critical
    );
    assert_eq!(
        HostMemoryPressure::from_available_percent(6.0),
        HostMemoryPressure::Critical
    );
}

#[test]
fn memory_pressure_emergency() {
    assert_eq!(
        HostMemoryPressure::from_available_percent(5.0),
        HostMemoryPressure::Emergency
    );
    assert_eq!(
        HostMemoryPressure::from_available_percent(0.0),
        HostMemoryPressure::Emergency
    );
}

#[test]
fn memory_pressure_ordering() {
    assert!(HostMemoryPressure::Normal < HostMemoryPressure::Warning);
    assert!(HostMemoryPressure::Warning < HostMemoryPressure::Critical);
    assert!(HostMemoryPressure::Critical < HostMemoryPressure::Emergency);
}

// ---------------------------------------------------------------------------
// CDF Math Verification (phi accrual numerical accuracy)
// ---------------------------------------------------------------------------

#[test]
fn phi_at_mean_is_approximately_0_301() {
    // At y=0 (elapsed == mean), Q(0) = 0.5, -log10(0.5) = 0.301
    // Create detector with regular 1s heartbeats, check phi at exactly mean offset
    let mut det = PhiAccrualDetector::new(100, 8.0, 12.0);
    let t0 = Instant::now();
    // Seed with regular 1s intervals
    for i in 0..10 {
        det.heartbeat_at(t0 + Duration::from_secs(i));
    }
    // Phi at exactly 1 second after last heartbeat (at the mean)
    let phi = det.phi_at_offset(Duration::from_secs(1)).unwrap();
    assert!(
        (phi - 0.301).abs() < 0.1,
        "phi at mean should be ~0.301, got {}",
        phi
    );
}

#[test]
fn phi_increases_monotonically_with_delay() {
    let mut det = PhiAccrualDetector::new(100, 8.0, 12.0);
    let t0 = Instant::now();
    for i in 0..20 {
        det.heartbeat_at(t0 + Duration::from_secs(i));
    }
    let mut prev_phi = 0.0;
    for secs in [1, 2, 3, 5, 8, 13, 21] {
        let phi = det.phi_at_offset(Duration::from_secs(secs)).unwrap();
        assert!(
            phi >= prev_phi,
            "phi must be monotonically non-decreasing: {} < {} at {}s",
            phi,
            prev_phi,
            secs
        );
        prev_phi = phi;
    }
}

#[test]
fn phi_exceeds_suspect_threshold_after_sufficient_delay() {
    let mut det = PhiAccrualDetector::new(100, 8.0, 12.0);
    let t0 = Instant::now();
    for i in 0..20 {
        det.heartbeat_at(t0 + Duration::from_secs(i));
    }
    // At 5x the mean interval, phi should be well above the suspect threshold (8)
    let phi_5x = det.phi_at_offset(Duration::from_secs(5)).unwrap();
    assert!(
        phi_5x > 8.0,
        "phi at 5x mean should exceed suspect threshold 8, got {}",
        phi_5x
    );
}

#[test]
fn phi_exceeds_failed_threshold_after_long_delay() {
    let mut det = PhiAccrualDetector::new(100, 8.0, 12.0);
    let t0 = Instant::now();
    for i in 0..20 {
        det.heartbeat_at(t0 + Duration::from_secs(i));
    }
    // At 10x the mean interval, phi should exceed the failed threshold (12)
    let phi_10x = det.phi_at_offset(Duration::from_secs(10)).unwrap();
    assert!(
        phi_10x > 12.0,
        "phi at 10x mean should exceed failed threshold 12, got {}",
        phi_10x
    );
}

#[test]
fn phi_near_zero_when_heartbeat_arrives_early() {
    let mut det = PhiAccrualDetector::new(100, 8.0, 12.0);
    let t0 = Instant::now();
    for i in 0..20 {
        det.heartbeat_at(t0 + Duration::from_secs(i));
    }
    // Heartbeat arrives 500ms early (half the 1s interval)
    let phi = det.phi_at_offset(Duration::from_millis(500)).unwrap();
    assert!(
        phi < 1.0,
        "phi when heartbeat arrives early should be near 0, got {}",
        phi
    );
}

#[test]
fn phi_cdf_complement_is_non_negative() {
    // The CDF complement function should never return negative
    let mut det = PhiAccrualDetector::new(100, 8.0, 12.0);
    let t0 = Instant::now();
    for i in 0..10 {
        det.heartbeat_at(t0 + Duration::from_secs(i));
    }
    for ms in [0, 100, 500, 1000, 2000, 5000, 10000, 60000] {
        let phi = det.phi_at_offset(Duration::from_millis(ms)).unwrap();
        assert!(
            phi >= 0.0,
            "phi must be non-negative, got {} at {}ms",
            phi,
            ms
        );
    }
}
