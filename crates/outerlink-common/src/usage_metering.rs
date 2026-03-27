//! Usage metering and cost tracking for multi-tenant GPU sharing.
//!
//! Provides per-context accounting of GPU time, VRAM usage, data transfers,
//! and kernel launches. Pure types and accounting math -- no CUDA calls, no I/O.

use std::time::Instant;

use serde::{Deserialize, Serialize};

/// Direction of a memory transfer operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransferDirection {
    /// Host (CPU) to device (GPU) transfer.
    HostToDevice,
    /// Device (GPU) to host (CPU) transfer.
    DeviceToHost,
    /// Device-to-device transfer on the same node.
    DeviceToDevice,
    /// Peer-to-peer transfer between GPUs (possibly across nodes).
    PeerToPeer,
}

/// Per-context usage accumulator for billing and chargeback.
///
/// Tracks GPU time, VRAM usage, transfers, and API calls for a single
/// CUDA context. All counters are monotonically increasing.
#[derive(Debug)]
pub struct UsageAccumulator {
    context_id: u64,
    gpu_time_ns: u64,
    /// Accumulated VRAM usage in byte-seconds.
    /// Converted to GB-hours by [`Self::vram_gb_hours`].
    vram_byte_secs: u64,
    /// Sub-second nanosecond remainder for VRAM time accumulation.
    vram_ns_remainder: u64,
    kernel_launches: u64,
    bytes_transferred_h2d: u64,
    bytes_transferred_d2h: u64,
    bytes_transferred_d2d: u64,
    bytes_transferred_p2p: u64,
    cuda_api_calls: u64,
    created_at: Instant,
    last_updated: Option<Instant>,
}

impl UsageAccumulator {
    /// Create a new zeroed accumulator for the given context.
    pub fn new(context_id: u64) -> Self {
        Self {
            context_id,
            gpu_time_ns: 0,
            vram_byte_secs: 0,
            vram_ns_remainder: 0,
            kernel_launches: 0,
            bytes_transferred_h2d: 0,
            bytes_transferred_d2h: 0,
            bytes_transferred_d2d: 0,
            bytes_transferred_p2p: 0,
            cuda_api_calls: 0,
            created_at: Instant::now(),
            last_updated: None,
        }
    }

    // -- Accessors --

    pub fn context_id(&self) -> u64 {
        self.context_id
    }

    pub fn gpu_time_ns(&self) -> u64 {
        self.gpu_time_ns
    }

    /// Raw VRAM byte-seconds. Use [`Self::vram_gb_hours`] for human units.
    pub fn vram_byte_seconds(&self) -> u64 {
        self.vram_byte_secs
    }

    pub fn kernel_launches(&self) -> u64 {
        self.kernel_launches
    }

    pub fn bytes_transferred_h2d(&self) -> u64 {
        self.bytes_transferred_h2d
    }

    pub fn bytes_transferred_d2h(&self) -> u64 {
        self.bytes_transferred_d2h
    }

    pub fn bytes_transferred_d2d(&self) -> u64 {
        self.bytes_transferred_d2d
    }

    pub fn bytes_transferred_p2p(&self) -> u64 {
        self.bytes_transferred_p2p
    }

    pub fn cuda_api_calls(&self) -> u64 {
        self.cuda_api_calls
    }

    pub fn created_at(&self) -> Instant {
        self.created_at
    }

    pub fn last_updated(&self) -> Option<Instant> {
        self.last_updated
    }

    // -- Recording methods --

    /// Record a kernel launch with the given execution duration in nanoseconds.
    pub fn record_kernel(&mut self, duration_ns: u64) {
        self.gpu_time_ns = self.gpu_time_ns.saturating_add(duration_ns);
        self.kernel_launches = self.kernel_launches.saturating_add(1);
        self.last_updated = Some(Instant::now());
    }

    /// Record a memory transfer of `bytes` in the given `direction`.
    pub fn record_transfer(&mut self, bytes: u64, direction: TransferDirection) {
        match direction {
            TransferDirection::HostToDevice => {
                self.bytes_transferred_h2d = self.bytes_transferred_h2d.saturating_add(bytes);
            }
            TransferDirection::DeviceToHost => {
                self.bytes_transferred_d2h = self.bytes_transferred_d2h.saturating_add(bytes);
            }
            TransferDirection::DeviceToDevice => {
                self.bytes_transferred_d2d = self.bytes_transferred_d2d.saturating_add(bytes);
            }
            TransferDirection::PeerToPeer => {
                self.bytes_transferred_p2p = self.bytes_transferred_p2p.saturating_add(bytes);
            }
        }
        self.last_updated = Some(Instant::now());
    }

    /// Record a single CUDA API call.
    pub fn record_api_call(&mut self) {
        self.cuda_api_calls = self.cuda_api_calls.saturating_add(1);
        self.last_updated = Some(Instant::now());
    }

    /// Update VRAM time tracking.
    ///
    /// `current_vram_bytes` is the number of VRAM bytes currently allocated
    /// by this context, and `elapsed_ns` is the time since the last update.
    /// Accumulates byte-seconds with sub-second precision via a nanosecond
    /// remainder that carries across calls.
    pub fn update_vram_time(&mut self, current_vram_bytes: u64, elapsed_ns: u64) {
        let total_ns = self.vram_ns_remainder.saturating_add(elapsed_ns);
        let elapsed_secs = total_ns / 1_000_000_000;
        self.vram_ns_remainder = total_ns % 1_000_000_000;
        self.vram_byte_secs = self
            .vram_byte_secs
            .saturating_add(current_vram_bytes.saturating_mul(elapsed_secs));
        self.last_updated = Some(Instant::now());
    }

    // -- Derived calculations --

    /// Total GPU time expressed in hours.
    pub fn gpu_hours(&self) -> f64 {
        self.gpu_time_ns as f64 / NS_PER_HOUR
    }

    /// Total VRAM usage expressed in GB-hours.
    ///
    /// Internally stored as byte-seconds; this divides by (1 GB * 3600 s/h).
    pub fn vram_gb_hours(&self) -> f64 {
        self.vram_byte_secs as f64 / (BYTES_PER_GB * SECS_PER_HOUR)
    }
}

/// Nanoseconds in one hour.
const NS_PER_HOUR: f64 = 3_600_000_000_000.0;

/// Seconds in one hour.
const SECS_PER_HOUR: f64 = 3_600.0;

/// Bytes in one gibibyte (GiB, used as "GB" in GPU contexts).
const BYTES_PER_GB: f64 = 1_073_741_824.0;

/// Serializable snapshot of a [`UsageAccumulator`] at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageSnapshot {
    pub context_id: u64,
    pub gpu_time_ns: u64,
    pub vram_byte_seconds: u64,
    pub kernel_launches: u64,
    pub bytes_transferred_h2d: u64,
    pub bytes_transferred_d2h: u64,
    pub bytes_transferred_d2d: u64,
    pub bytes_transferred_p2p: u64,
    pub cuda_api_calls: u64,
    pub snapshot_at_epoch_ms: u64,
}

impl UsageSnapshot {
    /// Create a snapshot from the current state of an accumulator.
    pub fn from_accumulator(acc: &UsageAccumulator) -> Self {
        Self {
            context_id: acc.context_id,
            gpu_time_ns: acc.gpu_time_ns,
            vram_byte_seconds: acc.vram_byte_secs,
            kernel_launches: acc.kernel_launches,
            bytes_transferred_h2d: acc.bytes_transferred_h2d,
            bytes_transferred_d2h: acc.bytes_transferred_d2h,
            bytes_transferred_d2d: acc.bytes_transferred_d2d,
            bytes_transferred_p2p: acc.bytes_transferred_p2p,
            cuda_api_calls: acc.cuda_api_calls,
            snapshot_at_epoch_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
                .min(u64::MAX as u128) as u64,
        }
    }

    /// Estimate cost based on a billing rate.
    pub fn cost_estimate(&self, rate: &BillingRate) -> f64 {
        let gpu_hours = self.gpu_time_ns as f64 / NS_PER_HOUR;
        let vram_gb_hours = self.vram_byte_seconds as f64 / (BYTES_PER_GB * SECS_PER_HOUR);
        let total_transfer_bytes = (self.bytes_transferred_h2d as f64)
            + (self.bytes_transferred_d2h as f64)
            + (self.bytes_transferred_d2d as f64)
            + (self.bytes_transferred_p2p as f64);
        let transfer_gb = total_transfer_bytes / BYTES_PER_GB;

        (gpu_hours * rate.gpu_hour_cost)
            + (vram_gb_hours * rate.vram_gb_hour_cost)
            + (transfer_gb * rate.transfer_gb_cost)
            + (self.cuda_api_calls as f64 * rate.api_call_cost)
    }
}

/// Configurable billing rates for cost estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingRate {
    /// Cost per GPU-hour in dollars.
    pub gpu_hour_cost: f64,
    /// Cost per VRAM GB-hour in dollars.
    pub vram_gb_hour_cost: f64,
    /// Cost per GB of data transferred in dollars.
    pub transfer_gb_cost: f64,
    /// Cost per API call in dollars (default 0.0).
    pub api_call_cost: f64,
}

impl Default for BillingRate {
    fn default() -> Self {
        Self {
            gpu_hour_cost: 0.0,
            vram_gb_hour_cost: 0.0,
            transfer_gb_cost: 0.0,
            api_call_cost: 0.0,
        }
    }
}

/// Aggregate usage summary across all contexts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageSummary {
    pub total_contexts: u32,
    pub total_gpu_hours: f64,
    pub total_vram_gb_hours: f64,
    pub total_bytes_transferred: u64,
    pub total_kernel_launches: u64,
    pub total_estimated_cost: f64,
}

/// Configuration for the metering subsystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeteringConfig {
    /// How often to flush accumulated metrics (seconds).
    pub flush_interval_secs: u64,
    /// Whether to take a snapshot when a context is destroyed.
    pub snapshot_on_destroy: bool,
    /// Whether to track every individual API call (expensive).
    pub enable_per_call_tracking: bool,
    /// How long to retain metering data (hours). Default 720 (30 days).
    pub retention_hours: u64,
}

impl Default for MeteringConfig {
    fn default() -> Self {
        Self {
            flush_interval_secs: 60,
            snapshot_on_destroy: true,
            enable_per_call_tracking: false,
            retention_hours: 720,
        }
    }
}

/// Format a dollar amount with 4 decimal places: `"$X.XXXX"`.
pub fn format_cost(amount: f64) -> String {
    format!("${:.4}", amount)
}

/// Format a duration in nanoseconds as hours: `"X.XXh"`.
pub fn format_duration_hours(ns: u64) -> String {
    let hours = ns as f64 / NS_PER_HOUR;
    format!("{:.2}h", hours)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_sanity() {
        assert_eq!(NS_PER_HOUR, 3.6e12);
        assert_eq!(BYTES_PER_GB, 1_073_741_824.0);
    }

    #[test]
    fn saturating_kernel_no_panic() {
        let mut acc = UsageAccumulator::new(1);
        acc.record_kernel(u64::MAX);
        acc.record_kernel(1);
        assert_eq!(acc.gpu_time_ns(), u64::MAX);
        assert_eq!(acc.kernel_launches(), 2);
    }

    #[test]
    fn saturating_transfer_no_panic() {
        let mut acc = UsageAccumulator::new(1);
        acc.record_transfer(u64::MAX, TransferDirection::HostToDevice);
        acc.record_transfer(1, TransferDirection::HostToDevice);
        assert_eq!(acc.bytes_transferred_h2d(), u64::MAX);
    }
}
