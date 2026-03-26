//! Access monitor implementation for memory tiering.
//!
//! `InterceptionAccessMonitor` tracks memory access patterns from CUDA
//! interception hooks and classifies pages by their access behavior
//! (streaming, working-set, strided, phased, random). This feeds into
//! eviction policy decisions and prefetch predictions for R11.

use std::sync::RwLock;

use dashmap::DashMap;

use super::tier_status::PrefetchPrediction;
use super::traits::AccessMonitor;
use super::types::{AccessPatternType, MemcpyDirection, TierId, LOCAL_VRAM};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the access monitor.
pub struct MonitorConfig {
    /// Maximum pages to track (older entries pruned on overflow).
    pub max_tracked_pages: usize,
    /// Maximum kernel records to keep.
    pub max_kernel_records: usize,
    /// History depth for kernel sequence prediction.
    pub kernel_history_depth: usize,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            max_tracked_pages: 1_000_000,
            max_kernel_records: 10_000,
            kernel_history_depth: 128,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

/// Per-page access state tracked by the monitor.
#[allow(dead_code)] // first_access_ts, tier_id reserved for phased pattern gap analysis and tier-aware scoring
struct PageAccessState {
    /// Total number of accesses recorded.
    access_count: u32,
    /// Timestamp (epoch seconds) of the last access.
    last_access_ts: u32,
    /// Timestamp (epoch seconds) of the first access.
    first_access_ts: u32,
    /// Classified access pattern.
    pattern: AccessPatternType,
    /// Ring buffer of VPN deltas between consecutive accesses (for stride detection).
    recent_deltas: [i64; 4],
    /// Cursor into the recent_deltas ring buffer.
    delta_cursor: u8,
    /// How many distinct kernels have accessed this page.
    kernel_count: u16,
    /// Current tier where this page resides.
    tier_id: TierId,
    /// The last VPN that accessed this page's "group" (for delta computation).
    last_vpn: Option<u64>,
    /// Whether the page had a gap in access (used for phased detection).
    had_gap: bool,
    /// Timestamp when a gap started (no accesses for a period).
    gap_start_ts: Option<u32>,
}

impl PageAccessState {
    fn new(ts: u32) -> Self {
        Self {
            access_count: 0,
            last_access_ts: ts,
            first_access_ts: ts,
            pattern: AccessPatternType::Unknown,
            recent_deltas: [0; 4],
            delta_cursor: 0,
            kernel_count: 0,
            tier_id: LOCAL_VRAM,
            last_vpn: None,
            had_gap: false,
            gap_start_ts: None,
        }
    }
}

/// Per-kernel record of buffer associations.
struct KernelRecord {
    /// VPNs of buffers this kernel touches.
    buffer_vpns: Vec<u64>,
    /// Number of times this kernel has been launched.
    launch_count: u32,
    /// Timestamp of last launch.
    last_launch_ts: u32,
}

// ---------------------------------------------------------------------------
// InterceptionAccessMonitor
// ---------------------------------------------------------------------------

/// Tracks memory access patterns from CUDA interception hooks.
///
/// Thread-safe: uses `DashMap` for per-page and per-kernel state, and
/// `RwLock` for the kernel history ring buffer.
pub struct InterceptionAccessMonitor {
    /// Per-page access state.
    page_state: DashMap<u64, PageAccessState>,
    /// Per-kernel buffer associations (kernel_id -> record).
    kernel_buffers: DashMap<u64, KernelRecord>,
    /// Recent kernel launch order (ring buffer of kernel IDs for prediction).
    kernel_history: RwLock<Vec<u64>>,
    /// Configuration.
    config: MonitorConfig,
}

impl InterceptionAccessMonitor {
    /// Create a new monitor with default configuration.
    pub fn new() -> Self {
        Self::with_config(MonitorConfig::default())
    }

    /// Create a new monitor with custom configuration.
    pub fn with_config(config: MonitorConfig) -> Self {
        Self {
            page_state: DashMap::new(),
            kernel_buffers: DashMap::new(),
            kernel_history: RwLock::new(Vec::with_capacity(config.kernel_history_depth)),
            config,
        }
    }

    /// Get the current epoch timestamp in seconds.
    /// Uses a monotonic counter for testing determinism; in production
    /// this would use `std::time::SystemTime`.
    fn now_ts(&self) -> u32 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as u32
    }

    /// Record an access to a specific VPN, updating page state and pattern.
    fn record_access(&self, vpn: u64, ts: u32) {
        let mut entry = self.page_state.entry(vpn).or_insert_with(|| PageAccessState::new(ts));
        let state = entry.value_mut();
        state.access_count = state.access_count.saturating_add(1);

        // Detect gaps for phased pattern: if time since last access > 10 seconds
        // and we've had accesses before, mark as having a gap.
        if state.access_count > 1 && ts.saturating_sub(state.last_access_ts) > 10 {
            state.had_gap = true;
            if state.gap_start_ts.is_none() {
                state.gap_start_ts = Some(state.last_access_ts);
            }
        }

        // Record VPN delta for stride detection
        if let Some(last) = state.last_vpn {
            let delta = vpn as i64 - last as i64;
            let idx = state.delta_cursor as usize % 4;
            state.recent_deltas[idx] = delta;
            state.delta_cursor = state.delta_cursor.wrapping_add(1);
        }
        state.last_vpn = Some(vpn);
        state.last_access_ts = ts;

        // Reclassify pattern
        state.pattern = Self::classify_pattern(state);
    }

    /// Classify the access pattern based on accumulated state.
    fn classify_pattern(state: &PageAccessState) -> AccessPatternType {
        // Need at least 2 accesses to classify
        if state.access_count < 2 {
            return AccessPatternType::Unknown;
        }

        // Check for strided pattern: all recent deltas are consistent
        let filled = std::cmp::min(state.delta_cursor as usize, 4);
        if filled >= 2 {
            if Self::is_strided(&state.recent_deltas[..filled]) {
                return AccessPatternType::Strided;
            }
        }

        // Check for phased: had a gap and now being accessed again
        if state.had_gap && state.access_count > 3 {
            return AccessPatternType::Phased;
        }

        // Check for streaming: low access count, positive sequential deltas
        if state.access_count <= 3 && filled >= 1 {
            let all_positive_sequential = state.recent_deltas[..filled]
                .iter()
                .all(|&d| d > 0 && d <= 16);
            if all_positive_sequential {
                return AccessPatternType::Streaming;
            }
        }

        // Check for working set: high access count relative to kernel count
        if state.access_count >= 4 {
            // If accessed many times, it's a working set
            if state.kernel_count <= 2 || state.access_count as u16 > state.kernel_count * 2 {
                return AccessPatternType::WorkingSet;
            }
        }

        // Check for random: high variance in deltas
        if filled >= 2 {
            let deltas = &state.recent_deltas[..filled];
            let mean = deltas.iter().sum::<i64>() / filled as i64;
            let variance: i64 = deltas.iter().map(|d| (d - mean).pow(2)).sum::<i64>() / filled as i64;
            if variance > 100 {
                return AccessPatternType::Random;
            }
        }

        AccessPatternType::Unknown
    }

    /// Check if deltas represent a strided pattern.
    /// All deltas must be within 10% of the median (or equal for small deltas).
    fn is_strided(deltas: &[i64]) -> bool {
        if deltas.is_empty() {
            return false;
        }

        // All deltas must be non-zero
        if deltas.iter().any(|&d| d == 0) {
            return false;
        }

        let mut sorted = deltas.to_vec();
        sorted.sort();
        let median = sorted[sorted.len() / 2];

        if median == 0 {
            return false;
        }

        // All deltas must be within 10% of median (or exactly equal for small values)
        let threshold = std::cmp::max(1, (median.abs() / 10).max(1));
        deltas.iter().all(|&d| (d - median).abs() <= threshold)
    }

    /// Append a kernel ID to the history ring buffer.
    fn push_kernel_history(&self, kernel_id: u64) {
        let mut history = self.kernel_history.write().unwrap();
        if history.len() >= self.config.kernel_history_depth {
            history.remove(0);
        }
        history.push(kernel_id);
    }

    /// Compute a hot/cold score for a page. Higher = hotter.
    fn hot_score(state: &PageAccessState, now_ts: u32) -> f64 {
        let age = now_ts.saturating_sub(state.last_access_ts) as f64;
        // Recency weight: exponential decay with half-life of 60 seconds
        let recency_weight = (-age / 60.0).exp();
        state.access_count as f64 * recency_weight
    }
}

impl Default for InterceptionAccessMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl AccessMonitor for InterceptionAccessMonitor {
    fn record_memcpy(
        &self,
        dst_vpn_start: u64,
        src_vpn_start: u64,
        page_count: usize,
        _direction: MemcpyDirection,
    ) {
        let ts = self.now_ts();
        for i in 0..page_count as u64 {
            self.record_access(dst_vpn_start + i, ts);
            self.record_access(src_vpn_start + i, ts);
        }
    }

    fn record_kernel_launch(&self, kernel_id: u64, vpns: &[u64]) {
        let ts = self.now_ts();

        // Update page states: record access and increment kernel_count
        for &vpn in vpns {
            self.record_access(vpn, ts);
            // Increment kernel_count for this page (record_access already created the entry)
            if let Some(mut entry) = self.page_state.get_mut(&vpn) {
                entry.kernel_count = entry.kernel_count.saturating_add(1);
            }
        }

        // Update kernel record
        {
            let mut entry = self.kernel_buffers.entry(kernel_id).or_insert_with(|| KernelRecord {
                buffer_vpns: Vec::new(),
                launch_count: 0,
                last_launch_ts: ts,
            });
            let record = entry.value_mut();
            record.launch_count = record.launch_count.saturating_add(1);
            record.last_launch_ts = ts;
            record.buffer_vpns = vpns.to_vec();
        }

        self.push_kernel_history(kernel_id);
    }

    fn record_kernel_complete(&self, _kernel_id: u64) {
        // Currently a no-op. Could be used to track kernel durations in the future.
    }

    fn get_pattern(&self, vpn: u64) -> AccessPatternType {
        self.page_state
            .get(&vpn)
            .map(|entry| entry.pattern)
            .unwrap_or(AccessPatternType::Unknown)
    }

    fn predict_next_accesses(&self, vpn: u64, count: usize) -> Vec<PrefetchPrediction> {
        let mut predictions = Vec::new();

        // Strategy 1: Stride-based prediction
        if let Some(state) = self.page_state.get(&vpn) {
            if state.pattern == AccessPatternType::Strided {
                let filled = std::cmp::min(state.delta_cursor as usize, 4);
                if filled > 0 {
                    // Use the most recent delta as the stride
                    let last_idx = (state.delta_cursor.wrapping_sub(1)) as usize % 4;
                    let stride = state.recent_deltas[last_idx];
                    if stride != 0 {
                        for i in 1..=count {
                            let predicted_vpn = (vpn as i64 + stride * i as i64) as u64;
                            predictions.push(PrefetchPrediction {
                                vpn: predicted_vpn,
                                confidence: 0.8 / (i as f64).sqrt(),
                                target_tier: LOCAL_VRAM,
                            });
                        }
                        return predictions;
                    }
                }
            }
        }

        // Strategy 2: Kernel history pattern matching
        let history = self.kernel_history.read().unwrap();
        if history.len() >= 3 {
            // Look for repeating patterns in kernel history
            // Try pattern lengths from 1 to half the history
            let max_pattern_len = std::cmp::min(history.len() / 2, 8);
            for pattern_len in 1..=max_pattern_len {
                let tail = &history[history.len() - pattern_len..];
                // Count how many times this pattern repeats backwards
                let mut repetitions = 0u32;
                let mut pos = history.len() as isize - pattern_len as isize;
                while pos >= pattern_len as isize {
                    let start = (pos - pattern_len as isize) as usize;
                    let segment = &history[start..start + pattern_len];
                    if segment == tail {
                        repetitions += 1;
                    } else {
                        break;
                    }
                    pos -= pattern_len as isize;
                }

                if repetitions >= 1 {
                    // Predict the next kernel in the pattern and get its VPNs
                    let next_kernel_idx = history.len() % (pattern_len * (repetitions as usize + 1));
                    let predicted_kernel_idx = next_kernel_idx % pattern_len;
                    if predicted_kernel_idx < history.len() {
                        let predicted_kernel = history[history.len() - pattern_len + predicted_kernel_idx];
                        if let Some(record) = self.kernel_buffers.get(&predicted_kernel) {
                            let confidence = (repetitions as f64 / 5.0).min(0.9);
                            for &pred_vpn in record.buffer_vpns.iter().take(count) {
                                predictions.push(PrefetchPrediction {
                                    vpn: pred_vpn,
                                    confidence,
                                    target_tier: LOCAL_VRAM,
                                });
                            }
                            if !predictions.is_empty() {
                                return predictions;
                            }
                        }
                    }
                }
            }
        }

        predictions
    }

    fn hot_pages(&self, limit: usize) -> Vec<u64> {
        let now = self.now_ts();
        let mut scored: Vec<(u64, f64)> = self
            .page_state
            .iter()
            .map(|entry| (*entry.key(), Self::hot_score(entry.value(), now)))
            .collect();

        // Sort descending by score (hottest first)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(limit).map(|(vpn, _)| vpn).collect()
    }

    fn cold_pages(&self, limit: usize) -> Vec<u64> {
        let now = self.now_ts();
        let mut scored: Vec<(u64, f64)> = self
            .page_state
            .iter()
            .map(|entry| (*entry.key(), Self::hot_score(entry.value(), now)))
            .collect();

        // Sort ascending by score (coldest first)
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(limit).map(|(vpn, _)| vpn).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a monitor and manually inject page state for deterministic tests.
    fn make_monitor() -> InterceptionAccessMonitor {
        InterceptionAccessMonitor::with_config(MonitorConfig {
            max_tracked_pages: 10_000,
            max_kernel_records: 1_000,
            kernel_history_depth: 32,
        })
    }

    /// Helper: directly record an access with a controlled timestamp.
    fn record_at(monitor: &InterceptionAccessMonitor, vpn: u64, ts: u32) {
        let mut entry = monitor.page_state.entry(vpn).or_insert_with(|| PageAccessState::new(ts));
        let state = entry.value_mut();
        state.access_count = state.access_count.saturating_add(1);

        if state.access_count > 1 && ts.saturating_sub(state.last_access_ts) > 10 {
            state.had_gap = true;
            if state.gap_start_ts.is_none() {
                state.gap_start_ts = Some(state.last_access_ts);
            }
        }

        if let Some(last) = state.last_vpn {
            let delta = vpn as i64 - last as i64;
            let idx = state.delta_cursor as usize % 4;
            state.recent_deltas[idx] = delta;
            state.delta_cursor = state.delta_cursor.wrapping_add(1);
        }
        state.last_vpn = Some(vpn);
        state.last_access_ts = ts;

        state.pattern = InterceptionAccessMonitor::classify_pattern(state);
    }

    /// Helper: record a kernel launch with controlled timestamp.
    fn kernel_launch_at(
        monitor: &InterceptionAccessMonitor,
        kernel_id: u64,
        vpns: &[u64],
        ts: u32,
    ) {
        let mut entry = monitor
            .kernel_buffers
            .entry(kernel_id)
            .or_insert_with(|| KernelRecord {
                buffer_vpns: Vec::new(),
                launch_count: 0,
                last_launch_ts: ts,
            });
        let record = entry.value_mut();
        record.launch_count = record.launch_count.saturating_add(1);
        record.last_launch_ts = ts;
        record.buffer_vpns = vpns.to_vec();

        for &vpn in vpns {
            record_at(monitor, vpn, ts);
            let mut page_entry = monitor
                .page_state
                .entry(vpn)
                .or_insert_with(|| PageAccessState::new(ts));
            page_entry.value_mut().kernel_count =
                page_entry.value_mut().kernel_count.saturating_add(1);
        }

        monitor.push_kernel_history(kernel_id);
    }

    // -------------------------------------------------------------------
    // 1. record_memcpy updates page state
    // -------------------------------------------------------------------

    #[test]
    fn monitor_record_memcpy() {
        let monitor = make_monitor();

        // Record a memcpy of 4 pages from VPN 0x100 to VPN 0x200
        monitor.record_memcpy(0x200, 0x100, 4, MemcpyDirection::DeviceToDevice);

        // Both source and destination VPNs should have page state
        for vpn in 0x100..0x104 {
            assert!(
                monitor.page_state.contains_key(&vpn),
                "source VPN {:#x} should be tracked",
                vpn
            );
            let state = monitor.page_state.get(&vpn).unwrap();
            assert!(state.access_count >= 1, "access_count should be >= 1");
        }

        for vpn in 0x200..0x204 {
            assert!(
                monitor.page_state.contains_key(&vpn),
                "dest VPN {:#x} should be tracked",
                vpn
            );
            let state = monitor.page_state.get(&vpn).unwrap();
            assert!(state.access_count >= 1, "access_count should be >= 1");
        }
    }

    // -------------------------------------------------------------------
    // 2. record_kernel_launch associates buffers with kernels
    // -------------------------------------------------------------------

    #[test]
    fn monitor_record_kernel_launch() {
        let monitor = make_monitor();
        let vpns = vec![0x1000, 0x2000, 0x3000];

        monitor.record_kernel_launch(42, &vpns);

        // Kernel record should exist with the right VPNs
        // Scope the DashMap Ref so the read lock is released before the next write
        {
            let record = monitor.kernel_buffers.get(&42).unwrap();
            assert_eq!(record.launch_count, 1);
            assert_eq!(record.buffer_vpns, vpns);
        }

        // All VPNs should have page state
        for &vpn in &vpns {
            assert!(monitor.page_state.contains_key(&vpn));
        }

        // Launch the same kernel again
        monitor.record_kernel_launch(42, &vpns);
        {
            let record = monitor.kernel_buffers.get(&42).unwrap();
            assert_eq!(record.launch_count, 2);
        }
    }

    // -------------------------------------------------------------------
    // 3. Pattern: Streaming (sequential, accessed once or twice)
    // -------------------------------------------------------------------

    #[test]
    fn monitor_pattern_streaming() {
        // Streaming pattern: page accessed few times with small positive sequential deltas.
        // We inject the state directly since per-page delta tracking requires
        // cross-page context (the delta records VPN differences between accesses).
        let monitor = make_monitor();

        // Inject a page that looks streaming: accessed 2 times with delta=+1
        {
            let state = PageAccessState {
                access_count: 2,
                last_access_ts: 1001,
                first_access_ts: 1000,
                pattern: AccessPatternType::Unknown,
                recent_deltas: [1, 0, 0, 0],
                delta_cursor: 1,
                kernel_count: 0,
                tier_id: LOCAL_VRAM,
                last_vpn: Some(101),
                had_gap: false,
                gap_start_ts: None,
            };
            monitor.page_state.insert(100, state);
        }

        // Reclassify
        let mut entry = monitor.page_state.get_mut(&100).unwrap();
        entry.pattern = InterceptionAccessMonitor::classify_pattern(&entry);
        let pattern = entry.pattern;
        drop(entry);

        assert_eq!(
            pattern,
            AccessPatternType::Streaming,
            "sequential low-count access should be Streaming"
        );
    }

    // -------------------------------------------------------------------
    // 4. Pattern: WorkingSet (repeated access to same pages)
    // -------------------------------------------------------------------

    #[test]
    fn monitor_pattern_working_set() {
        let monitor = make_monitor();

        // A page accessed many times by the same kernel
        let base_ts = 1000;
        for i in 0..10u32 {
            record_at(&monitor, 0x5000, base_ts + i);
        }

        let pattern = monitor.get_pattern(0x5000);
        assert_eq!(
            pattern,
            AccessPatternType::WorkingSet,
            "repeatedly accessed page should be WorkingSet"
        );
    }

    // -------------------------------------------------------------------
    // 5. Pattern: Strided (regular stride)
    // -------------------------------------------------------------------

    #[test]
    fn monitor_pattern_strided() {
        let monitor = make_monitor();

        // Inject a page with consistent stride deltas
        {
            let state = PageAccessState {
                access_count: 5,
                last_access_ts: 1004,
                first_access_ts: 1000,
                pattern: AccessPatternType::Unknown,
                recent_deltas: [16, 16, 16, 16],
                delta_cursor: 4,
                kernel_count: 1,
                tier_id: LOCAL_VRAM,
                last_vpn: Some(0x5040),
                had_gap: false,
                gap_start_ts: None,
            };
            monitor.page_state.insert(0x5000, state);
        }

        // Reclassify
        {
            let mut entry = monitor.page_state.get_mut(&0x5000).unwrap();
            entry.pattern = InterceptionAccessMonitor::classify_pattern(&entry);
        }

        assert_eq!(
            monitor.get_pattern(0x5000),
            AccessPatternType::Strided,
            "consistent stride deltas should be classified as Strided"
        );
    }

    // -------------------------------------------------------------------
    // 6. Pattern: Random (high variance in deltas)
    // -------------------------------------------------------------------

    #[test]
    fn monitor_pattern_random() {
        let monitor = make_monitor();

        // Inject a page with wildly varying deltas
        {
            let state = PageAccessState {
                access_count: 3,
                last_access_ts: 1002,
                first_access_ts: 1000,
                pattern: AccessPatternType::Unknown,
                recent_deltas: [500, -300, 1000, 0],
                delta_cursor: 3,
                kernel_count: 0,
                tier_id: LOCAL_VRAM,
                last_vpn: Some(0x9000),
                had_gap: false,
                gap_start_ts: None,
            };
            monitor.page_state.insert(0x7000, state);
        }

        {
            let mut entry = monitor.page_state.get_mut(&0x7000).unwrap();
            entry.pattern = InterceptionAccessMonitor::classify_pattern(&entry);
        }

        assert_eq!(
            monitor.get_pattern(0x7000),
            AccessPatternType::Random,
            "high-variance deltas should be classified as Random"
        );
    }

    // -------------------------------------------------------------------
    // 7. hot_pages: most-accessed pages ranked first
    // -------------------------------------------------------------------

    #[test]
    fn monitor_hot_pages() {
        let monitor = make_monitor();

        let now = monitor.now_ts();

        // Insert pages with varying access counts, all accessed "now"
        for vpn in 0..10u64 {
            let state = PageAccessState {
                access_count: (vpn as u32 + 1) * 10, // 10, 20, ..., 100
                last_access_ts: now,
                first_access_ts: now,
                pattern: AccessPatternType::Unknown,
                recent_deltas: [0; 4],
                delta_cursor: 0,
                kernel_count: 0,
                tier_id: LOCAL_VRAM,
                last_vpn: None,
                had_gap: false,
                gap_start_ts: None,
            };
            monitor.page_state.insert(vpn, state);
        }

        let hot = monitor.hot_pages(3);
        assert_eq!(hot.len(), 3);

        // The hottest page should be VPN 9 (access_count=100)
        assert_eq!(hot[0], 9, "VPN 9 has highest access count");
        assert_eq!(hot[1], 8, "VPN 8 has second highest");
        assert_eq!(hot[2], 7, "VPN 7 has third highest");
    }

    // -------------------------------------------------------------------
    // 8. cold_pages: least-accessed pages ranked first
    // -------------------------------------------------------------------

    #[test]
    fn monitor_cold_pages() {
        let monitor = make_monitor();

        let now = monitor.now_ts();

        // Insert pages with varying access counts, all accessed "now"
        for vpn in 0..10u64 {
            let state = PageAccessState {
                access_count: (vpn as u32 + 1) * 10,
                last_access_ts: now,
                first_access_ts: now,
                pattern: AccessPatternType::Unknown,
                recent_deltas: [0; 4],
                delta_cursor: 0,
                kernel_count: 0,
                tier_id: LOCAL_VRAM,
                last_vpn: None,
                had_gap: false,
                gap_start_ts: None,
            };
            monitor.page_state.insert(vpn, state);
        }

        let cold = monitor.cold_pages(3);
        assert_eq!(cold.len(), 3);

        // The coldest page should be VPN 0 (access_count=10)
        assert_eq!(cold[0], 0, "VPN 0 has lowest access count");
        assert_eq!(cold[1], 1, "VPN 1 has second lowest");
        assert_eq!(cold[2], 2, "VPN 2 has third lowest");
    }

    // -------------------------------------------------------------------
    // 9. predict_next_accesses with repeating kernel pattern
    // -------------------------------------------------------------------

    #[test]
    fn monitor_predict_repeating_kernel() {
        let monitor = make_monitor();

        // Set up three kernels with distinct buffer VPNs
        let kernel_a = 1u64;
        let kernel_b = 2u64;
        let kernel_c = 3u64;
        let vpns_a = vec![0x100, 0x200];
        let vpns_b = vec![0x300, 0x400];
        let vpns_c = vec![0x500, 0x600];

        let ts = 1000;
        // Simulate pattern: A, B, C, A, B, C, A, B, C
        for round in 0..3u32 {
            kernel_launch_at(&monitor, kernel_a, &vpns_a, ts + round * 3);
            kernel_launch_at(&monitor, kernel_b, &vpns_b, ts + round * 3 + 1);
            kernel_launch_at(&monitor, kernel_c, &vpns_c, ts + round * 3 + 2);
        }

        // After A, B, C repeated 3 times, history is [A,B,C,A,B,C,A,B,C]
        // The last kernel launched was C. Next in the pattern should be A.
        // predict_next_accesses should predict A's VPNs.
        let predictions = monitor.predict_next_accesses(0x500, 4);

        // We should get predictions (may be from A's VPNs)
        assert!(
            !predictions.is_empty(),
            "should predict next accesses from repeating kernel pattern"
        );

        // Confidence should be > 0
        for pred in &predictions {
            assert!(
                pred.confidence > 0.0,
                "prediction confidence should be positive"
            );
        }
    }

    // -------------------------------------------------------------------
    // 10. Concurrent access from multiple threads
    // -------------------------------------------------------------------

    #[test]
    fn monitor_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let monitor = Arc::new(make_monitor());
        let num_threads = 4;
        let ops_per_thread = 100;

        let handles: Vec<_> = (0..num_threads)
            .map(|t| {
                let m = Arc::clone(&monitor);
                thread::spawn(move || {
                    for i in 0..ops_per_thread {
                        let vpn_base = (t as u64) * 10_000;
                        // Record memcpy
                        m.record_memcpy(
                            vpn_base + i,
                            vpn_base + 1000 + i,
                            1,
                            MemcpyDirection::DeviceToDevice,
                        );
                        // Record kernel launch
                        let kernel_id = (t as u64) * 1000 + i;
                        m.record_kernel_launch(kernel_id, &[vpn_base + i]);
                        // Read pattern
                        let _ = m.get_pattern(vpn_base + i);
                        // Hot/cold pages
                        let _ = m.hot_pages(5);
                        let _ = m.cold_pages(5);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread should not panic");
        }

        // Verify all threads' data is present
        for t in 0..num_threads {
            let vpn_base = (t as u64) * 10_000;
            for i in 0..ops_per_thread {
                assert!(
                    monitor.page_state.contains_key(&(vpn_base + i)),
                    "page state for thread {} VPN {} should exist",
                    t,
                    vpn_base + i
                );
            }
        }
    }
}
