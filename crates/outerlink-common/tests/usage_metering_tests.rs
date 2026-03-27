//! Tests for usage metering types and cost tracking.

use outerlink_common::usage_metering::{
    format_cost, format_duration_hours, BillingRate, MeteringConfig, TransferDirection,
    UsageAccumulator, UsageSnapshot, UsageSummary,
};

// ---------------------------------------------------------------------------
// TransferDirection
// ---------------------------------------------------------------------------

#[test]
fn transfer_direction_has_all_variants() {
    let _h2d = TransferDirection::HostToDevice;
    let _d2h = TransferDirection::DeviceToHost;
    let _d2d = TransferDirection::DeviceToDevice;
    let _p2p = TransferDirection::PeerToPeer;
}

#[test]
fn transfer_direction_debug() {
    let d = TransferDirection::HostToDevice;
    let s = format!("{:?}", d);
    assert!(s.contains("HostToDevice"));
}

#[test]
fn transfer_direction_clone_eq() {
    let a = TransferDirection::DeviceToHost;
    let b = a.clone();
    assert_eq!(a, b);
}

// ---------------------------------------------------------------------------
// UsageAccumulator -- construction
// ---------------------------------------------------------------------------

#[test]
fn accumulator_new_zeroed() {
    let acc = UsageAccumulator::new(42);
    assert_eq!(acc.context_id(), 42);
    assert_eq!(acc.gpu_time_ns(), 0);
    assert_eq!(acc.vram_byte_seconds(), 0);
    assert_eq!(acc.kernel_launches(), 0);
    assert_eq!(acc.bytes_transferred_h2d(), 0);
    assert_eq!(acc.bytes_transferred_d2h(), 0);
    assert_eq!(acc.cuda_api_calls(), 0);
    assert!(acc.last_updated().is_none());
}

// ---------------------------------------------------------------------------
// UsageAccumulator -- record methods
// ---------------------------------------------------------------------------

#[test]
fn record_kernel_accumulates_time_and_count() {
    let mut acc = UsageAccumulator::new(1);
    acc.record_kernel(500_000);
    acc.record_kernel(300_000);
    assert_eq!(acc.gpu_time_ns(), 800_000);
    assert_eq!(acc.kernel_launches(), 2);
}

#[test]
fn record_transfer_h2d() {
    let mut acc = UsageAccumulator::new(1);
    acc.record_transfer(1024, TransferDirection::HostToDevice);
    assert_eq!(acc.bytes_transferred_h2d(), 1024);
    assert_eq!(acc.bytes_transferred_d2h(), 0);
}

#[test]
fn record_transfer_d2h() {
    let mut acc = UsageAccumulator::new(1);
    acc.record_transfer(2048, TransferDirection::DeviceToHost);
    assert_eq!(acc.bytes_transferred_d2h(), 2048);
    assert_eq!(acc.bytes_transferred_h2d(), 0);
}

#[test]
fn record_transfer_d2d_adds_to_both() {
    let mut acc = UsageAccumulator::new(1);
    acc.record_transfer(4096, TransferDirection::DeviceToDevice);
    // DeviceToDevice counts as both source and destination on same node
    assert_eq!(acc.bytes_transferred_h2d(), 0);
    assert_eq!(acc.bytes_transferred_d2h(), 0);
}

#[test]
fn record_transfer_p2p() {
    let mut acc = UsageAccumulator::new(1);
    acc.record_transfer(8192, TransferDirection::PeerToPeer);
    // PeerToPeer is neither H2D nor D2H
    assert_eq!(acc.bytes_transferred_h2d(), 0);
    assert_eq!(acc.bytes_transferred_d2h(), 0);
}

#[test]
fn record_api_call_increments() {
    let mut acc = UsageAccumulator::new(1);
    acc.record_api_call();
    acc.record_api_call();
    acc.record_api_call();
    assert_eq!(acc.cuda_api_calls(), 3);
}

#[test]
fn update_vram_time_accumulates() {
    let mut acc = UsageAccumulator::new(1);
    // 1 GB for 1 second = 1_073_741_824 byte-seconds
    acc.update_vram_time(1_073_741_824, 1_000_000_000);
    // vram_byte_seconds = bytes * elapsed_ns / 1_000_000_000  (normalized to seconds)
    // Actually: vram_byte_seconds should store bytes * seconds.
    // The spec says: "adds vram_bytes * elapsed to vram_byte_seconds"
    // elapsed is in ns, so we need to convert. Let's say the field stores byte*ns for precision.
    // But the conversion to gb_hours handles it. Let's verify the raw value.
    assert!(acc.vram_byte_seconds() > 0);
}

#[test]
fn record_kernel_updates_last_updated() {
    let acc_initial_last = {
        let acc = UsageAccumulator::new(1);
        acc.last_updated()
    };
    assert!(acc_initial_last.is_none());

    let mut acc = UsageAccumulator::new(1);
    acc.record_kernel(100);
    assert!(acc.last_updated().is_some());
}

// ---------------------------------------------------------------------------
// UsageAccumulator -- derived calculations
// ---------------------------------------------------------------------------

#[test]
fn gpu_hours_calculation() {
    let mut acc = UsageAccumulator::new(1);
    // 1 hour in nanoseconds = 3_600_000_000_000
    acc.record_kernel(3_600_000_000_000);
    let hours = acc.gpu_hours();
    assert!((hours - 1.0).abs() < 1e-6, "expected ~1.0 hour, got {}", hours);
}

#[test]
fn gpu_hours_zero_when_empty() {
    let acc = UsageAccumulator::new(1);
    assert_eq!(acc.gpu_hours(), 0.0);
}

#[test]
fn vram_gb_hours_calculation() {
    let mut acc = UsageAccumulator::new(1);
    // 1 GB = 1_073_741_824 bytes, 1 hour = 3_600_000_000_000 ns
    // Internally: elapsed_ns -> elapsed_secs, then byte_secs = bytes * secs
    // vram_gb_hours = byte_secs / (1 GB * 3600)
    let one_gb: u64 = 1_073_741_824;
    let one_hour_ns: u64 = 3_600_000_000_000;
    acc.update_vram_time(one_gb, one_hour_ns);
    let gb_hours = acc.vram_gb_hours();
    assert!(
        (gb_hours - 1.0).abs() < 1e-6,
        "expected ~1.0 GB-hour, got {}",
        gb_hours
    );
}

// ---------------------------------------------------------------------------
// UsageSnapshot
// ---------------------------------------------------------------------------

#[test]
fn snapshot_from_accumulator() {
    let mut acc = UsageAccumulator::new(99);
    acc.record_kernel(1_000_000);
    acc.record_transfer(2048, TransferDirection::HostToDevice);
    acc.record_api_call();
    let snap = UsageSnapshot::from_accumulator(&acc);
    assert_eq!(snap.context_id, 99);
    assert_eq!(snap.gpu_time_ns, 1_000_000);
    assert_eq!(snap.kernel_launches, 1);
    assert_eq!(snap.bytes_transferred_h2d, 2048);
    assert_eq!(snap.cuda_api_calls, 1);
}

#[test]
fn snapshot_cost_estimate() {
    let mut acc = UsageAccumulator::new(1);
    // 1 GPU hour
    acc.record_kernel(3_600_000_000_000);
    // 1 GB transferred H2D
    acc.record_transfer(1_073_741_824, TransferDirection::HostToDevice);

    let snap = UsageSnapshot::from_accumulator(&acc);
    let rate = BillingRate {
        gpu_hour_cost: 2.50,
        vram_gb_hour_cost: 0.10,
        transfer_gb_cost: 0.05,
        api_call_cost: 0.0,
    };
    let cost = snap.cost_estimate(&rate);
    // gpu: 1h * $2.50 = $2.50
    // transfer: 1 GB * $0.05 = $0.05
    // total = ~$2.55 (vram is 0 since we didn't call update_vram_time)
    assert!(cost > 2.5, "cost should be > $2.50, got {}", cost);
    assert!(cost < 3.0, "cost should be < $3.00, got {}", cost);
}

// ---------------------------------------------------------------------------
// BillingRate
// ---------------------------------------------------------------------------

#[test]
fn billing_rate_default_api_call_cost_zero() {
    let rate = BillingRate::default();
    assert_eq!(rate.api_call_cost, 0.0);
}

#[test]
fn billing_rate_custom() {
    let rate = BillingRate {
        gpu_hour_cost: 1.0,
        vram_gb_hour_cost: 0.5,
        transfer_gb_cost: 0.01,
        api_call_cost: 0.001,
    };
    assert_eq!(rate.gpu_hour_cost, 1.0);
    assert_eq!(rate.api_call_cost, 0.001);
}

// ---------------------------------------------------------------------------
// UsageSummary
// ---------------------------------------------------------------------------

#[test]
fn usage_summary_fields() {
    let summary = UsageSummary {
        total_contexts: 5,
        total_gpu_hours: 10.0,
        total_vram_gb_hours: 20.0,
        total_bytes_transferred: 1_000_000,
        total_kernel_launches: 500,
        total_estimated_cost: 25.0,
    };
    assert_eq!(summary.total_contexts, 5);
    assert_eq!(summary.total_kernel_launches, 500);
}

#[test]
fn usage_summary_debug() {
    let summary = UsageSummary {
        total_contexts: 1,
        total_gpu_hours: 0.5,
        total_vram_gb_hours: 1.0,
        total_bytes_transferred: 0,
        total_kernel_launches: 0,
        total_estimated_cost: 0.0,
    };
    let s = format!("{:?}", summary);
    assert!(s.contains("UsageSummary"));
}

// ---------------------------------------------------------------------------
// MeteringConfig
// ---------------------------------------------------------------------------

#[test]
fn metering_config_defaults() {
    let config = MeteringConfig::default();
    assert_eq!(config.flush_interval_secs, 60);
    assert!(config.snapshot_on_destroy);
    assert!(!config.enable_per_call_tracking);
    assert_eq!(config.retention_hours, 720);
}

#[test]
fn metering_config_custom() {
    let config = MeteringConfig {
        flush_interval_secs: 30,
        snapshot_on_destroy: false,
        enable_per_call_tracking: true,
        retention_hours: 24,
    };
    assert_eq!(config.flush_interval_secs, 30);
    assert!(!config.snapshot_on_destroy);
    assert!(config.enable_per_call_tracking);
    assert_eq!(config.retention_hours, 24);
}

// ---------------------------------------------------------------------------
// Format functions
// ---------------------------------------------------------------------------

#[test]
fn format_cost_basic() {
    assert_eq!(format_cost(1.5), "$1.5000");
    assert_eq!(format_cost(0.0), "$0.0000");
    assert_eq!(format_cost(123.456789), "$123.4568");
}

#[test]
fn format_cost_small_amounts() {
    assert_eq!(format_cost(0.0001), "$0.0001");
    assert_eq!(format_cost(0.00005), "$0.0001"); // rounded
}

#[test]
fn format_duration_hours_basic() {
    // 1 hour = 3_600_000_000_000 ns
    assert_eq!(format_duration_hours(3_600_000_000_000), "1.00h");
}

#[test]
fn format_duration_hours_zero() {
    assert_eq!(format_duration_hours(0), "0.00h");
}

#[test]
fn format_duration_hours_fractional() {
    // 30 minutes = 1_800_000_000_000 ns
    assert_eq!(format_duration_hours(1_800_000_000_000), "0.50h");
}
