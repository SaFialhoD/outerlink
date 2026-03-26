//! Page fault handler for pre-launch page resolution.
//!
//! Manages in-flight page fault requests, deduplicates concurrent requests
//! for the same page, and tracks resolution statistics. Also includes
//! thrashing detection to identify pages bouncing between nodes.

use dashmap::DashMap;
use std::sync::RwLock;
use std::time::{Duration, Instant};

/// Pre-launch page fault resolution handler.
pub struct FaultHandler {
    config: FaultConfig,
    /// Pages currently being fetched: vpn -> FaultState
    pending_faults: DashMap<u64, FaultState>,
    stats: RwLock<FaultStats>,
}

/// Configuration for the fault handler.
pub struct FaultConfig {
    /// Maximum number of concurrent fault resolutions (default: 64).
    pub max_concurrent_faults: usize,
    /// Maximum retries per fault before giving up (default: 3).
    pub max_retry_count: u32,
}

impl Default for FaultConfig {
    fn default() -> Self {
        Self {
            max_concurrent_faults: 64,
            max_retry_count: 3,
        }
    }
}

/// The state of an in-flight page fault.
///
/// Only `Fetching` entries live in `pending_faults`. When a fault completes
/// or fails, the entry is removed (not transitioned). This keeps the map
/// clean and avoids accumulating stale terminal states.
pub enum FaultState {
    Fetching { from_node: u8, started_at: Instant },
}

/// Statistics for fault handling operations.
#[derive(Debug, Default, Clone)]
pub struct FaultStats {
    pub total_faults: u64,
    pub faults_resolved: u64,
    pub faults_failed: u64,
    pub prefetch_avoided: u64,
    pub avg_resolution_us: f64,
}

/// Result of a single page fault request.
pub enum FaultResult {
    /// Page is already present locally.
    AlreadyLocal,
    /// A fetch for this page is already in-flight (dedup). Caller must wait
    /// for the existing fetch to complete before accessing the page.
    AlreadyFetching,
    /// Fetch has been initiated.
    Fetching,
    /// Request failed.
    Failed(String),
}

impl FaultHandler {
    /// Create a new fault handler with the given configuration.
    pub fn new(config: FaultConfig) -> Self {
        Self {
            config,
            pending_faults: DashMap::new(),
            stats: RwLock::new(FaultStats::default()),
        }
    }

    /// Batch request pages by VPN. Deduplicates requests and respects capacity.
    ///
    /// For each VPN:
    /// - If already pending (fetching), returns `AlreadyLocal` (dedup).
    /// - If capacity is exceeded, returns `Failed`.
    /// - Otherwise, initiates a fetch and returns `Fetching`.
    pub fn request_pages(&self, vpns: &[u64]) -> Vec<FaultResult> {
        let mut results = Vec::with_capacity(vpns.len());

        for &vpn in vpns {
            // Check if already pending (dedup): page is being fetched by another
            // request. Caller must wait for the in-flight fetch to complete.
            if self.pending_faults.contains_key(&vpn) {
                results.push(FaultResult::AlreadyFetching);
                continue;
            }

            // Check capacity
            if self.pending_faults.len() >= self.config.max_concurrent_faults {
                results.push(FaultResult::Failed(
                    "max concurrent faults reached".to_string(),
                ));
                continue;
            }

            // Initiate fetch
            self.pending_faults.insert(
                vpn,
                FaultState::Fetching {
                    from_node: 0, // Home node; real implementation would resolve this
                    started_at: Instant::now(),
                },
            );

            {
                let mut stats = self.stats.write().unwrap();
                stats.total_faults += 1;
            }

            results.push(FaultResult::Fetching);
        }

        results
    }

    /// Mark a fault as successfully resolved.
    pub fn on_fault_complete(&self, vpn: u64) {
        if let Some((_, FaultState::Fetching { started_at, .. })) =
            self.pending_faults.remove(&vpn)
        {
            let resolution_us = started_at.elapsed().as_micros() as f64;

            // Welford running average: increment count first (new N),
            // then update the mean. This is numerically stable.
            let mut stats = self.stats.write().unwrap();
            stats.faults_resolved += 1;
            let total_resolved = stats.faults_resolved as f64;
            stats.avg_resolution_us = stats.avg_resolution_us
                + (resolution_us - stats.avg_resolution_us) / total_resolved;
        }
    }

    /// Mark a fault as failed.
    ///
    /// The reason is logged via `tracing` (when integrated) for diagnostics.
    /// Retry logic is the caller's responsibility; `max_retry_count` in
    /// `FaultConfig` provides the recommended limit.
    pub fn on_fault_failed(&self, vpn: u64, reason: String) {
        if self.pending_faults.remove(&vpn).is_some() {
            // Log the failure reason for diagnostics. In production this
            // will use tracing::warn!; for now we just consume the value
            // to avoid the unused-variable lint.
            let _reason = reason;
            let mut stats = self.stats.write().unwrap();
            stats.faults_failed += 1;
        }
    }

    /// Check whether a fault is currently pending (being fetched) for the given VPN.
    pub fn is_fault_pending(&self, vpn: u64) -> bool {
        self.pending_faults.contains_key(&vpn)
    }

    /// Number of currently pending faults.
    pub fn pending_count(&self) -> usize {
        self.pending_faults.len()
    }

    /// Get a snapshot of the fault handling statistics.
    pub fn stats(&self) -> FaultStats {
        self.stats.read().unwrap().clone()
    }
}

// ---------------------------------------------------------------------------
// Thrashing detection
// ---------------------------------------------------------------------------

/// Detects pages that bounce between nodes excessively.
///
/// Tracks per-page bounce counts within a sliding time window.
/// When bounces exceed configured thresholds, escalating mitigation
/// levels are triggered:
/// - Level1: Shared-read promotion
/// - Level2: Write-broadcast
/// - Level3: Page pin
pub struct ThrashingDetector {
    /// Per-page bounce counter: vpn -> (bounce_count, window_start)
    bounces: DashMap<u64, (u32, Instant)>,
    config: ThrashingConfig,
}

/// Configuration for thrashing detection thresholds.
pub struct ThrashingConfig {
    /// Detection window in milliseconds (default: 10_000).
    pub window_ms: u64,
    /// Bounces within window to trigger Level1 (default: 5).
    pub level1_threshold: u32,
    /// Bounces within window to trigger Level2 (default: 10).
    pub level2_threshold: u32,
    /// Bounces within window to trigger Level3 (default: 20).
    pub level3_threshold: u32,
    /// Duration to pin a page in milliseconds (default: 20_000).
    pub pin_duration_ms: u64,
}

impl Default for ThrashingConfig {
    fn default() -> Self {
        Self {
            window_ms: 10_000,
            level1_threshold: 5,
            level2_threshold: 10,
            level3_threshold: 20,
            pin_duration_ms: 20_000,
        }
    }
}

/// Severity level of detected thrashing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThrashingLevel {
    None,
    Level1,
    Level2,
    Level3,
}

impl ThrashingDetector {
    /// Create a new thrashing detector with the given configuration.
    pub fn new(config: ThrashingConfig) -> Self {
        Self {
            bounces: DashMap::new(),
            config,
        }
    }

    /// Record a bounce for a page and return the current thrashing level.
    ///
    /// If the window has expired since the last bounce, the counter resets.
    pub fn record_bounce(&self, vpn: u64) -> ThrashingLevel {
        let now = Instant::now();
        let window = Duration::from_millis(self.config.window_ms);

        let mut entry = self.bounces.entry(vpn).or_insert((0, now));
        let (count, window_start) = entry.value_mut();

        // Reset if window has expired
        if now.duration_since(*window_start) > window {
            *count = 0;
            *window_start = now;
        }

        *count += 1;
        self.classify(*count)
    }

    /// Check the current thrashing level for a page without recording a bounce.
    pub fn check_level(&self, vpn: u64) -> ThrashingLevel {
        let now = Instant::now();
        let window = Duration::from_millis(self.config.window_ms);

        match self.bounces.get(&vpn) {
            Some(entry) => {
                let (count, window_start) = *entry.value();
                if now.duration_since(window_start) > window {
                    ThrashingLevel::None
                } else {
                    self.classify(count)
                }
            }
            None => ThrashingLevel::None,
        }
    }

    /// Map a bounce count to its thrashing level based on configured thresholds.
    fn classify(&self, count: u32) -> ThrashingLevel {
        if count >= self.config.level3_threshold {
            ThrashingLevel::Level3
        } else if count >= self.config.level2_threshold {
            ThrashingLevel::Level2
        } else if count >= self.config.level1_threshold {
            ThrashingLevel::Level1
        } else {
            ThrashingLevel::None
        }
    }
}
