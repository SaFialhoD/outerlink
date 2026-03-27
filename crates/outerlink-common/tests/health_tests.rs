//! Tests for the health monitoring module.

use std::time::{Duration, Instant};

use outerlink_common::health::{
    classify_xid, GpuHealthSnapshot, GpuHealthState, MemoryPressure, NicHealthState,
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
fn thermal_state_between_warn_and_migrate() {
    let snap = make_snapshot(85.0);
    let thresholds = ThermalThresholds::default();
    // 85 >= throttle_warn(80) but < migrate(90)
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
// MemoryPressure
// ---------------------------------------------------------------------------

#[test]
fn memory_pressure_normal() {
    assert_eq!(
        MemoryPressure::from_available_percent(50.0),
        MemoryPressure::Normal
    );
    assert_eq!(
        MemoryPressure::from_available_percent(31.0),
        MemoryPressure::Normal
    );
}

#[test]
fn memory_pressure_warning() {
    assert_eq!(
        MemoryPressure::from_available_percent(30.0),
        MemoryPressure::Warning
    );
    assert_eq!(
        MemoryPressure::from_available_percent(16.0),
        MemoryPressure::Warning
    );
}

#[test]
fn memory_pressure_critical() {
    assert_eq!(
        MemoryPressure::from_available_percent(15.0),
        MemoryPressure::Critical
    );
    assert_eq!(
        MemoryPressure::from_available_percent(6.0),
        MemoryPressure::Critical
    );
}

#[test]
fn memory_pressure_emergency() {
    assert_eq!(
        MemoryPressure::from_available_percent(5.0),
        MemoryPressure::Emergency
    );
    assert_eq!(
        MemoryPressure::from_available_percent(0.0),
        MemoryPressure::Emergency
    );
}

#[test]
fn memory_pressure_ordering() {
    assert!(MemoryPressure::Normal < MemoryPressure::Warning);
    assert!(MemoryPressure::Warning < MemoryPressure::Critical);
    assert!(MemoryPressure::Critical < MemoryPressure::Emergency);
}
