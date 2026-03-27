//! OutterLink metrics definitions.
//!
//! All metric names are defined here as constants so they are consistent
//! across crates. Uses the `metrics` crate facade pattern: callers record
//! metrics through these helpers, and the actual exporter (Prometheus, etc.)
//! is installed by the binary entrypoint. When no recorder is installed the
//! macros compile down to no-ops.

use metrics::{counter, gauge, histogram};

// ---------------------------------------------------------------------------
// GPU metrics
// ---------------------------------------------------------------------------
pub const GPU_UTILIZATION: &str = "outerlink_gpu_utilization_percent";
pub const GPU_TEMPERATURE: &str = "outerlink_gpu_temperature_celsius";
pub const GPU_VRAM_USED: &str = "outerlink_gpu_vram_used_bytes";
pub const GPU_VRAM_TOTAL: &str = "outerlink_gpu_vram_total_bytes";
pub const GPU_POWER_WATTS: &str = "outerlink_gpu_power_watts";

// ---------------------------------------------------------------------------
// Transfer metrics
// ---------------------------------------------------------------------------
pub const TRANSFER_BYTES_TOTAL: &str = "outerlink_transfer_bytes_total";
pub const TRANSFER_DURATION_SECONDS: &str = "outerlink_transfer_duration_seconds";
pub const TRANSFER_BANDWIDTH_BYTES_PER_SEC: &str = "outerlink_transfer_bandwidth_bps";

// ---------------------------------------------------------------------------
// CUDA interception metrics
// ---------------------------------------------------------------------------
pub const CUDA_CALLS_TOTAL: &str = "outerlink_cuda_calls_total";
pub const CUDA_CALL_DURATION_SECONDS: &str = "outerlink_cuda_call_duration_seconds";
pub const CUDA_ERRORS_TOTAL: &str = "outerlink_cuda_errors_total";

// ---------------------------------------------------------------------------
// Cluster metrics
// ---------------------------------------------------------------------------
pub const CLUSTER_NODES_TOTAL: &str = "outerlink_cluster_nodes_total";
pub const CLUSTER_GPUS_TOTAL: &str = "outerlink_cluster_gpus_total";
pub const CLUSTER_VRAM_AVAILABLE: &str = "outerlink_cluster_vram_available_bytes";

// ---------------------------------------------------------------------------
// Connection metrics
// ---------------------------------------------------------------------------
pub const CONNECTIONS_ACTIVE: &str = "outerlink_connections_active";
pub const CONNECTION_ERRORS_TOTAL: &str = "outerlink_connection_errors_total";
pub const HEARTBEAT_RTT_SECONDS: &str = "outerlink_heartbeat_rtt_seconds";

// ---------------------------------------------------------------------------
// Memory pool metrics
// ---------------------------------------------------------------------------
pub const PINNED_MEMORY_ALLOCATED: &str = "outerlink_pinned_memory_allocated_bytes";
pub const PINNED_MEMORY_POOL_SIZE: &str = "outerlink_pinned_memory_pool_size_bytes";
pub const PINNED_MEMORY_SPILLS: &str = "outerlink_pinned_memory_spills_total";

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Record a GPU utilization sample.
pub fn record_gpu_utilization(gpu_index: u32, percent: f64) {
    gauge!(GPU_UTILIZATION, "gpu" => gpu_index.to_string()).set(percent);
}

/// Record GPU temperature.
pub fn record_gpu_temperature(gpu_index: u32, celsius: f64) {
    gauge!(GPU_TEMPERATURE, "gpu" => gpu_index.to_string()).set(celsius);
}

/// Record GPU VRAM usage.
pub fn record_gpu_vram(gpu_index: u32, used: u64, total: u64) {
    gauge!(GPU_VRAM_USED, "gpu" => gpu_index.to_string()).set(used as f64);
    gauge!(GPU_VRAM_TOTAL, "gpu" => gpu_index.to_string()).set(total as f64);
}

/// Record GPU power draw.
pub fn record_gpu_power(gpu_index: u32, watts: f64) {
    gauge!(GPU_POWER_WATTS, "gpu" => gpu_index.to_string()).set(watts);
}

/// Record a data transfer.
pub fn record_transfer(direction: &str, bytes: u64, duration_secs: f64) {
    counter!(TRANSFER_BYTES_TOTAL, "direction" => direction.to_owned()).increment(bytes);
    histogram!(TRANSFER_DURATION_SECONDS, "direction" => direction.to_owned())
        .record(duration_secs);
    if duration_secs > 0.0 {
        gauge!(TRANSFER_BANDWIDTH_BYTES_PER_SEC, "direction" => direction.to_owned())
            .set(bytes as f64 / duration_secs);
    }
}

/// Record a CUDA API call.
pub fn record_cuda_call(function: &str, duration_secs: f64, success: bool) {
    counter!(CUDA_CALLS_TOTAL, "function" => function.to_owned()).increment(1);
    histogram!(CUDA_CALL_DURATION_SECONDS, "function" => function.to_owned())
        .record(duration_secs);
    if !success {
        counter!(CUDA_ERRORS_TOTAL, "function" => function.to_owned()).increment(1);
    }
}

/// Update cluster-level gauges.
pub fn record_cluster_state(nodes: u32, gpus: u32, vram_available: u64) {
    gauge!(CLUSTER_NODES_TOTAL).set(nodes as f64);
    gauge!(CLUSTER_GPUS_TOTAL).set(gpus as f64);
    gauge!(CLUSTER_VRAM_AVAILABLE).set(vram_available as f64);
}

/// Record active connection count.
pub fn record_connections(active: u32) {
    gauge!(CONNECTIONS_ACTIVE).set(active as f64);
}

/// Record a connection error.
pub fn record_connection_error(reason: &str) {
    counter!(CONNECTION_ERRORS_TOTAL, "reason" => reason.to_owned()).increment(1);
}

/// Record heartbeat round-trip time.
pub fn record_heartbeat_rtt(rtt_secs: f64) {
    histogram!(HEARTBEAT_RTT_SECONDS).record(rtt_secs);
}

/// Record pinned memory pool state.
pub fn record_pinned_memory(allocated: u64, pool_size: u64) {
    gauge!(PINNED_MEMORY_ALLOCATED).set(allocated as f64);
    gauge!(PINNED_MEMORY_POOL_SIZE).set(pool_size as f64);
}

/// Record a pinned memory spill (fallback to unpinned allocation).
pub fn record_pinned_memory_spill() {
    counter!(PINNED_MEMORY_SPILLS).increment(1);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// Collect every metric name constant into a list for systematic validation.
    fn all_metric_names() -> Vec<&'static str> {
        vec![
            GPU_UTILIZATION,
            GPU_TEMPERATURE,
            GPU_VRAM_USED,
            GPU_VRAM_TOTAL,
            GPU_POWER_WATTS,
            TRANSFER_BYTES_TOTAL,
            TRANSFER_DURATION_SECONDS,
            TRANSFER_BANDWIDTH_BYTES_PER_SEC,
            CUDA_CALLS_TOTAL,
            CUDA_CALL_DURATION_SECONDS,
            CUDA_ERRORS_TOTAL,
            CLUSTER_NODES_TOTAL,
            CLUSTER_GPUS_TOTAL,
            CLUSTER_VRAM_AVAILABLE,
            CONNECTIONS_ACTIVE,
            CONNECTION_ERRORS_TOTAL,
            HEARTBEAT_RTT_SECONDS,
            PINNED_MEMORY_ALLOCATED,
            PINNED_MEMORY_POOL_SIZE,
            PINNED_MEMORY_SPILLS,
        ]
    }

    #[test]
    fn metric_names_are_non_empty() {
        for name in all_metric_names() {
            assert!(!name.is_empty(), "metric name must not be empty");
        }
    }

    #[test]
    fn metric_names_have_outerlink_prefix() {
        for name in all_metric_names() {
            assert!(
                name.starts_with("outerlink_"),
                "metric '{}' must start with 'outerlink_'",
                name
            );
        }
    }

    #[test]
    fn metric_names_are_lowercase_with_underscores() {
        for name in all_metric_names() {
            assert!(
                name.chars()
                    .all(|c| c.is_ascii_lowercase() || c == '_'),
                "metric '{}' contains invalid characters (only lowercase + underscore allowed)",
                name
            );
        }
    }

    #[test]
    fn metric_names_are_unique() {
        let names = all_metric_names();
        let mut seen = std::collections::HashSet::new();
        for name in &names {
            assert!(
                seen.insert(name),
                "duplicate metric name: '{}'",
                name
            );
        }
    }

    // The following tests verify that helper functions do not panic
    // even when no metrics recorder is installed (noop path).

    #[test]
    fn gpu_helpers_do_not_panic_without_recorder() {
        record_gpu_utilization(0, 75.0);
        record_gpu_temperature(0, 65.0);
        record_gpu_vram(0, 4_000_000_000, 8_000_000_000);
        record_gpu_power(0, 250.0);
    }

    #[test]
    fn transfer_helper_does_not_panic_without_recorder() {
        record_transfer("send", 1024, 0.001);
        record_transfer("recv", 2048, 0.002);
    }

    #[test]
    fn transfer_zero_duration_does_not_divide_by_zero() {
        // Should skip the bandwidth gauge, not panic.
        record_transfer("send", 1024, 0.0);
    }

    #[test]
    fn cuda_call_helpers_do_not_panic_without_recorder() {
        record_cuda_call("cuMemAlloc", 0.0001, true);
        record_cuda_call("cuLaunchKernel", 0.005, false);
    }

    #[test]
    fn cluster_helpers_do_not_panic_without_recorder() {
        record_cluster_state(3, 6, 48_000_000_000);
    }

    #[test]
    fn connection_helpers_do_not_panic_without_recorder() {
        record_connections(5);
        record_connection_error("timeout");
        record_heartbeat_rtt(0.002);
    }

    #[test]
    fn pinned_memory_helpers_do_not_panic_without_recorder() {
        record_pinned_memory(1_000_000_000, 2_000_000_000);
        record_pinned_memory_spill();
    }
}
