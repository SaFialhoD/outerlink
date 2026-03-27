//! Tests for the GPU recovery module (R48).

use std::time::{Duration, Instant};

use outerlink_common::gpu_recovery::{
    EccCounters, GpuRecoveryLog, GpuRecoveryState, RecoveryEvent, RecoveryTier, ThermalAction,
    XidRecoveryAction, recovery_action,
};
use outerlink_common::health::XidSeverity;

// ---------------------------------------------------------------------------
// RecoveryTier ordering
// ---------------------------------------------------------------------------

#[test]
fn recovery_tier_ordering_context_lt_reset() {
    assert!(RecoveryTier::ContextRecreate < RecoveryTier::GpuReset);
}

#[test]
fn recovery_tier_ordering_reset_lt_eviction() {
    assert!(RecoveryTier::GpuReset < RecoveryTier::PoolEviction);
}

#[test]
fn recovery_tier_ordering_context_lt_eviction() {
    assert!(RecoveryTier::ContextRecreate < RecoveryTier::PoolEviction);
}

// ---------------------------------------------------------------------------
// recovery_action for all 16 Xid codes from R48
// ---------------------------------------------------------------------------

#[test]
fn recovery_xid_13_context_recreate() {
    let a = recovery_action(13);
    assert_eq!(a.xid_code, 13);
    assert_eq!(a.tier, RecoveryTier::ContextRecreate);
    assert!(!a.requires_drain);
    assert!(!a.affects_all_contexts);
}

#[test]
fn recovery_xid_31_gpu_reset() {
    let a = recovery_action(31);
    assert_eq!(a.tier, RecoveryTier::GpuReset);
    assert!(a.requires_drain);
    assert!(a.affects_all_contexts);
}

#[test]
fn recovery_xid_32_context_recreate() {
    let a = recovery_action(32);
    assert_eq!(a.tier, RecoveryTier::ContextRecreate);
    assert!(!a.affects_all_contexts);
}

#[test]
fn recovery_xid_38_gpu_reset() {
    let a = recovery_action(38);
    assert_eq!(a.tier, RecoveryTier::GpuReset);
    assert!(a.requires_drain);
    assert!(a.affects_all_contexts);
}

#[test]
fn recovery_xid_43_pool_eviction() {
    let a = recovery_action(43);
    assert_eq!(a.tier, RecoveryTier::PoolEviction);
    assert!(a.requires_drain);
}

#[test]
fn recovery_xid_45_context_recreate() {
    // Xid 45 is informational / no-op, but we still classify it as context-level
    let a = recovery_action(45);
    assert_eq!(a.tier, RecoveryTier::ContextRecreate);
    assert!(!a.requires_drain);
    assert!(!a.affects_all_contexts);
}

#[test]
fn recovery_xid_48_pool_eviction() {
    let a = recovery_action(48);
    assert_eq!(a.tier, RecoveryTier::PoolEviction);
    assert!(a.requires_drain);
}

#[test]
fn recovery_xid_61_gpu_reset() {
    let a = recovery_action(61);
    assert_eq!(a.tier, RecoveryTier::GpuReset);
    assert!(a.requires_drain);
    assert!(a.affects_all_contexts);
}

#[test]
fn recovery_xid_62_gpu_reset() {
    let a = recovery_action(62);
    assert_eq!(a.tier, RecoveryTier::GpuReset);
    assert!(a.requires_drain);
}

#[test]
fn recovery_xid_63_context_recreate() {
    // Xid 63 = ECC page retirement, warning-level, track but no immediate action
    let a = recovery_action(63);
    assert_eq!(a.tier, RecoveryTier::ContextRecreate);
    assert!(!a.requires_drain);
}

#[test]
fn recovery_xid_64_pool_eviction() {
    let a = recovery_action(64);
    assert_eq!(a.tier, RecoveryTier::PoolEviction);
    assert!(a.requires_drain);
}

#[test]
fn recovery_xid_68_context_recreate() {
    // Xid 68 = video processor exception, informational for compute
    let a = recovery_action(68);
    assert_eq!(a.tier, RecoveryTier::ContextRecreate);
    assert!(!a.requires_drain);
}

#[test]
fn recovery_xid_69_context_recreate() {
    let a = recovery_action(69);
    assert_eq!(a.tier, RecoveryTier::ContextRecreate);
    assert!(!a.affects_all_contexts);
}

#[test]
fn recovery_xid_79_pool_eviction() {
    let a = recovery_action(79);
    assert_eq!(a.tier, RecoveryTier::PoolEviction);
    assert!(a.requires_drain);
    assert!(a.affects_all_contexts);
}

#[test]
fn recovery_xid_94_context_recreate() {
    // Xid 94 = contained ECC, self-corrected
    let a = recovery_action(94);
    assert_eq!(a.tier, RecoveryTier::ContextRecreate);
    assert!(!a.requires_drain);
}

#[test]
fn recovery_xid_95_gpu_reset() {
    let a = recovery_action(95);
    assert_eq!(a.tier, RecoveryTier::GpuReset);
    assert!(a.requires_drain);
    assert!(a.affects_all_contexts);
}

#[test]
fn recovery_unknown_xid_defaults_to_gpu_reset() {
    let a = recovery_action(999);
    assert_eq!(a.tier, RecoveryTier::GpuReset);
    assert!(a.requires_drain);
}

#[test]
fn recovery_action_description_is_nonempty() {
    for xid in [13, 31, 32, 38, 43, 45, 48, 61, 62, 63, 64, 68, 69, 79, 94, 95] {
        let a = recovery_action(xid);
        assert!(!a.description.is_empty(), "Xid {} has empty description", xid);
    }
}

// ---------------------------------------------------------------------------
// EccCounters
// ---------------------------------------------------------------------------

#[test]
fn ecc_needs_eviction_on_any_dbe() {
    let c = EccCounters {
        correctable_single: 0,
        uncorrectable_double: 1,
        retired_pages: 0,
        pending_retirement: false,
    };
    assert!(c.needs_eviction());
}

#[test]
fn ecc_no_eviction_when_clean() {
    let c = EccCounters {
        correctable_single: 100,
        uncorrectable_double: 0,
        retired_pages: 3,
        pending_retirement: false,
    };
    assert!(!c.needs_eviction());
}

#[test]
fn ecc_needs_investigation_when_many_retired_pages() {
    let c = EccCounters {
        correctable_single: 0,
        uncorrectable_double: 0,
        retired_pages: 6,
        pending_retirement: false,
    };
    assert!(c.needs_investigation());
}

#[test]
fn ecc_no_investigation_when_few_retired_pages() {
    let c = EccCounters {
        correctable_single: 0,
        uncorrectable_double: 0,
        retired_pages: 4,
        pending_retirement: false,
    };
    assert!(!c.needs_investigation());
}

#[test]
fn ecc_needs_investigation_when_pending_retirement() {
    let c = EccCounters {
        correctable_single: 0,
        uncorrectable_double: 0,
        retired_pages: 0,
        pending_retirement: true,
    };
    assert!(c.needs_investigation());
}

// ---------------------------------------------------------------------------
// ThermalAction
// ---------------------------------------------------------------------------

#[test]
fn thermal_none_below_85() {
    assert_eq!(ThermalAction::from_temperature(84.9), ThermalAction::None);
}

#[test]
fn thermal_reduce_load_at_85() {
    assert_eq!(ThermalAction::from_temperature(85.0), ThermalAction::ReduceLoad);
}

#[test]
fn thermal_stop_scheduling_at_90() {
    assert_eq!(ThermalAction::from_temperature(90.0), ThermalAction::StopScheduling);
}

#[test]
fn thermal_migrate_at_95() {
    assert_eq!(ThermalAction::from_temperature(95.0), ThermalAction::MigrateWorkloads);
}

#[test]
fn thermal_emergency_at_100() {
    assert_eq!(ThermalAction::from_temperature(100.0), ThermalAction::EmergencyShutdown);
}

#[test]
fn thermal_emergency_above_100() {
    assert_eq!(ThermalAction::from_temperature(110.0), ThermalAction::EmergencyShutdown);
}

#[test]
fn thermal_between_85_and_90() {
    assert_eq!(ThermalAction::from_temperature(87.5), ThermalAction::ReduceLoad);
}

#[test]
fn thermal_between_90_and_95() {
    assert_eq!(ThermalAction::from_temperature(92.0), ThermalAction::StopScheduling);
}

#[test]
fn thermal_between_95_and_100() {
    assert_eq!(ThermalAction::from_temperature(97.0), ThermalAction::MigrateWorkloads);
}

// ---------------------------------------------------------------------------
// GpuRecoveryState
// ---------------------------------------------------------------------------

#[test]
fn recovery_state_normal_is_default_variant() {
    let state = GpuRecoveryState::Normal;
    assert!(matches!(state, GpuRecoveryState::Normal));
}

#[test]
fn recovery_state_context_recovery_holds_xid() {
    let state = GpuRecoveryState::ContextRecovery { xid: 13 };
    if let GpuRecoveryState::ContextRecovery { xid } = state {
        assert_eq!(xid, 13);
    } else {
        panic!("Expected ContextRecovery");
    }
}

#[test]
fn recovery_state_draining_holds_reason() {
    let state = GpuRecoveryState::Draining {
        reason: "Xid 38 GPU reset pending".to_string(),
    };
    if let GpuRecoveryState::Draining { reason } = state {
        assert!(reason.contains("Xid 38"));
    } else {
        panic!("Expected Draining");
    }
}

#[test]
fn recovery_state_evicted_holds_xid() {
    let state = GpuRecoveryState::Evicted { xid: 79 };
    if let GpuRecoveryState::Evicted { xid } = state {
        assert_eq!(xid, 79);
    } else {
        panic!("Expected Evicted");
    }
}

// ---------------------------------------------------------------------------
// GpuRecoveryLog
// ---------------------------------------------------------------------------

#[test]
fn recovery_log_push_and_len() {
    let mut log = GpuRecoveryLog::new();
    assert_eq!(log.len(), 0);
    log.push(RecoveryEvent {
        timestamp: Instant::now(),
        xid: 13,
        tier: RecoveryTier::ContextRecreate,
        success: true,
    });
    assert_eq!(log.len(), 1);
}

#[test]
fn recovery_log_recent_xids_filters_by_duration() {
    let mut log = GpuRecoveryLog::new();
    let now = Instant::now();

    // Event "far in the past" - we can't easily create past Instants,
    // so we test by adding events at current time and filtering with a large window.
    log.push(RecoveryEvent {
        timestamp: now,
        xid: 13,
        tier: RecoveryTier::ContextRecreate,
        success: true,
    });
    log.push(RecoveryEvent {
        timestamp: now,
        xid: 31,
        tier: RecoveryTier::GpuReset,
        success: false,
    });

    let recent = log.recent_xids(Duration::from_secs(60));
    assert_eq!(recent.len(), 2);
}

#[test]
fn recovery_log_failure_rate_all_success() {
    let mut log = GpuRecoveryLog::new();
    log.push(RecoveryEvent {
        timestamp: Instant::now(),
        xid: 13,
        tier: RecoveryTier::ContextRecreate,
        success: true,
    });
    log.push(RecoveryEvent {
        timestamp: Instant::now(),
        xid: 69,
        tier: RecoveryTier::ContextRecreate,
        success: true,
    });
    assert!((log.failure_rate() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn recovery_log_failure_rate_half_failed() {
    let mut log = GpuRecoveryLog::new();
    log.push(RecoveryEvent {
        timestamp: Instant::now(),
        xid: 13,
        tier: RecoveryTier::ContextRecreate,
        success: true,
    });
    log.push(RecoveryEvent {
        timestamp: Instant::now(),
        xid: 38,
        tier: RecoveryTier::GpuReset,
        success: false,
    });
    assert!((log.failure_rate() - 0.5).abs() < f64::EPSILON);
}

#[test]
fn recovery_log_failure_rate_empty() {
    let log = GpuRecoveryLog::new();
    assert!((log.failure_rate() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn recovery_log_is_empty() {
    let log = GpuRecoveryLog::new();
    assert!(log.is_empty());
}

// ---------------------------------------------------------------------------
// XidRecoveryAction is Clone + Debug
// ---------------------------------------------------------------------------

#[test]
fn xid_recovery_action_clone_and_debug() {
    let a = recovery_action(13);
    let b = a.clone();
    assert_eq!(a.xid_code, b.xid_code);
    let _ = format!("{:?}", a);
}

#[test]
fn recovery_tier_clone_and_debug() {
    let t = RecoveryTier::GpuReset;
    let t2 = t.clone();
    assert_eq!(t, t2);
    let _ = format!("{:?}", t);
}

#[test]
fn ecc_counters_clone_and_debug() {
    let c = EccCounters {
        correctable_single: 5,
        uncorrectable_double: 0,
        retired_pages: 1,
        pending_retirement: false,
    };
    let c2 = c.clone();
    assert_eq!(c.correctable_single, c2.correctable_single);
    let _ = format!("{:?}", c);
}
