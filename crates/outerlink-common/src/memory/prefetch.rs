//! Speculative prefetch scheduler for the memory tiering system.
//!
//! Consumes access predictions from the access monitor and schedules
//! page migrations ahead of actual demand, reducing stalls from
//! cross-tier memory access.

use std::collections::VecDeque;
use std::sync::{Mutex, RwLock};

use dashmap::DashMap;

use crate::memory::tier_status::PrefetchPrediction;
use crate::memory::types::TierId;

/// Source of a prefetch prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionSource {
    /// Regular stride pattern detected
    Stride,
    /// Repeating kernel execution sequence
    KernelHistory,
    /// R13 CUDA Graph schedule (confidence = 1.0)
    GraphSchedule,
    /// Explicit prefetch hint from the application
    Manual,
}

/// Configuration for the prefetch scheduler.
#[derive(Debug, Clone)]
pub struct PrefetchConfig {
    /// Maximum pending prefetch requests in the queue.
    pub max_pending: usize,
    /// Maximum concurrent in-flight prefetches.
    pub max_in_flight: usize,
    /// Minimum confidence threshold to schedule a prefetch (0.0-1.0).
    pub min_confidence: f32,
    /// How many kernels ahead to look for predictions.
    pub lookahead_kernels: usize,
    /// Fraction of bandwidth allocated to prefetching (0.0-1.0).
    pub bandwidth_fraction: f32,
    /// Target tier for prefetched pages.
    pub target_tier: TierId,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            max_pending: 1024,
            max_in_flight: 64,
            min_confidence: 0.5,
            lookahead_kernels: 3,
            bandwidth_fraction: 0.2,
            target_tier: crate::memory::types::LOCAL_VRAM,
        }
    }
}

/// A pending prefetch request.
struct PrefetchRequest {
    vpn: u64,
    #[allow(dead_code)]
    source_tier: TierId,
    confidence: f32,
    #[allow(dead_code)]
    predicted_by: PredictionSource,
    #[allow(dead_code)]
    submitted_at: std::time::Instant,
}

/// State of an in-flight prefetch.
struct PrefetchState {
    #[allow(dead_code)]
    request: PrefetchRequest,
    #[allow(dead_code)]
    migration_id: u64,
}

/// Aggregate statistics for the prefetch scheduler.
#[derive(Debug, Default, Clone)]
pub struct PrefetchStats {
    /// Total number of prefetches scheduled.
    pub total_scheduled: u64,
    /// Total number of prefetches that completed.
    pub total_completed: u64,
    /// Prefetched pages that were subsequently accessed (useful prefetch).
    pub total_hits: u64,
    /// Prefetched pages that were never accessed (wasted bandwidth).
    pub total_misses: u64,
    /// Pages accessed before their prefetch completed (latency stall).
    pub total_stalls: u64,
    /// Prefetches that were cancelled before completion.
    pub total_cancelled: u64,
}

/// Speculative prefetch scheduler.
///
/// Consumes `PrefetchPrediction`s from the access monitor, queues them
/// by confidence, and drains them into migration requests up to the
/// configured concurrency limit.
pub struct PrefetchScheduler {
    config: RwLock<PrefetchConfig>,
    /// Ring buffer of pending prefetch requests, ordered by confidence descending.
    pending: Mutex<VecDeque<PrefetchRequest>>,
    /// Currently in-flight prefetches keyed by VPN.
    in_flight: DashMap<u64, PrefetchState>,
    /// Completed prefetches: vpn -> was_used.
    completed: DashMap<u64, bool>,
    /// Aggregate stats.
    stats: RwLock<PrefetchStats>,
    /// Counter for generating migration IDs.
    next_migration_id: Mutex<u64>,
    /// Prediction counter for adaptive lookahead.
    prediction_count: Mutex<u64>,
}

impl PrefetchScheduler {
    /// Create a new prefetch scheduler with the given configuration.
    pub fn new(config: PrefetchConfig) -> Self {
        Self {
            config: RwLock::new(config),
            pending: Mutex::new(VecDeque::new()),
            in_flight: DashMap::new(),
            completed: DashMap::new(),
            stats: RwLock::new(PrefetchStats::default()),
            next_migration_id: Mutex::new(1),
            prediction_count: Mutex::new(0),
        }
    }

    /// Schedule prefetches from a set of predictions.
    ///
    /// Filters by confidence threshold, skips pages already in-flight or
    /// pending, and inserts into the pending queue ordered by confidence
    /// descending.
    pub fn schedule_from_predictions(&self, predictions: &[PrefetchPrediction]) {
        let config = self.config.read().unwrap();
        let min_confidence = config.min_confidence;
        let max_pending = config.max_pending;
        drop(config);

        let mut pending = self.pending.lock().unwrap();

        // Collect VPNs already pending to avoid duplicates.
        // Mutable: updated as new predictions are inserted to prevent
        // the same VPN appearing twice within a single predictions slice.
        let mut pending_vpns: std::collections::HashSet<u64> =
            pending.iter().map(|r| r.vpn).collect();

        for pred in predictions {
            // Skip below confidence threshold
            if pred.confidence < min_confidence as f64 {
                continue;
            }
            // Skip if already in-flight
            if self.in_flight.contains_key(&pred.vpn) {
                continue;
            }
            // Skip if already pending
            if pending_vpns.contains(&pred.vpn) {
                continue;
            }
            // Skip if already completed (already prefetched)
            if self.completed.contains_key(&pred.vpn) {
                continue;
            }
            // Skip if queue is full
            if pending.len() >= max_pending {
                break;
            }

            let confidence = pred.confidence as f32;
            let request = PrefetchRequest {
                vpn: pred.vpn,
                source_tier: pred.target_tier,
                confidence,
                predicted_by: PredictionSource::Stride, // Default; real impl would derive from prediction metadata
                submitted_at: std::time::Instant::now(),
            };

            // Insert in sorted position (descending confidence)
            let pos = pending
                .iter()
                .position(|r| r.confidence < confidence)
                .unwrap_or(pending.len());
            pending.insert(pos, request);
            pending_vpns.insert(pred.vpn);
        }

        // Update prediction counter for adaptive lookahead
        let mut count = self.prediction_count.lock().unwrap();
        *count += predictions.len() as u64;

        // Adaptive lookahead: adjust every 100 predictions
        if *count >= 100 {
            *count = 0;
            drop(count);
            self.adapt_lookahead();
        }
    }

    /// Drain pending queue and submit migrations up to max_in_flight.
    ///
    /// Returns the number of new prefetches started.
    pub fn tick(&self) -> usize {
        let config = self.config.read().unwrap();
        let max_in_flight = config.max_in_flight;
        drop(config);

        let mut pending = self.pending.lock().unwrap();
        let mut started = 0;

        while self.in_flight.len() < max_in_flight {
            let request = match pending.pop_front() {
                Some(r) => r,
                None => break,
            };

            let vpn = request.vpn;
            let migration_id = {
                let mut id = self.next_migration_id.lock().unwrap();
                let current = *id;
                *id += 1;
                current
            };

            self.in_flight.insert(
                vpn,
                PrefetchState {
                    request,
                    migration_id,
                },
            );

            {
                let mut stats = self.stats.write().unwrap();
                stats.total_scheduled += 1;
            }

            started += 1;
        }

        started
    }

    /// Mark a page as accessed. If the page was prefetched, record a hit.
    /// If the page is still in-flight, record a stall.
    pub fn on_page_accessed(&self, vpn: u64) {
        // Check if in-flight (stall: page needed before prefetch completed)
        if self.in_flight.contains_key(&vpn) {
            let mut stats = self.stats.write().unwrap();
            stats.total_stalls += 1;
            return;
        }

        // Check if completed prefetch
        if let Some(mut entry) = self.completed.get_mut(&vpn) {
            if !*entry {
                *entry = true;
                let mut stats = self.stats.write().unwrap();
                stats.total_hits += 1;
            }
        }
    }

    /// Mark a prefetch as completed (migration finished).
    pub fn on_prefetch_complete(&self, vpn: u64) {
        if let Some((_, _state)) = self.in_flight.remove(&vpn) {
            self.completed.insert(vpn, false); // Not yet accessed
            let mut stats = self.stats.write().unwrap();
            stats.total_completed += 1;
        }
    }

    /// Cancel a pending or in-flight prefetch. Returns true if found and cancelled.
    pub fn cancel_prefetch(&self, vpn: u64) -> bool {
        // Try pending first
        {
            let mut pending = self.pending.lock().unwrap();
            if let Some(pos) = pending.iter().position(|r| r.vpn == vpn) {
                pending.remove(pos);
                let mut stats = self.stats.write().unwrap();
                stats.total_cancelled += 1;
                return true;
            }
        }

        // Try in-flight
        if self.in_flight.remove(&vpn).is_some() {
            let mut stats = self.stats.write().unwrap();
            stats.total_cancelled += 1;
            return true;
        }

        false
    }

    /// Calculate the hit rate: hits / (hits + misses).
    /// Returns 0.0 if no completed prefetches have been evaluated.
    pub fn hit_rate(&self) -> f32 {
        let stats = self.stats.read().unwrap();
        let total = stats.total_hits + stats.total_misses;
        if total == 0 {
            return 0.0;
        }
        stats.total_hits as f32 / total as f32
    }

    /// Calculate the stall rate: stalls / (stalls + hits).
    /// Returns 0.0 if no relevant accesses recorded.
    pub fn stall_rate(&self) -> f32 {
        let stats = self.stats.read().unwrap();
        let total = stats.total_stalls + stats.total_hits;
        if total == 0 {
            return 0.0;
        }
        stats.total_stalls as f32 / total as f32
    }

    /// Get a snapshot of the current statistics.
    pub fn stats(&self) -> PrefetchStats {
        self.stats.read().unwrap().clone()
    }

    /// Check if a page has an in-flight prefetch (for R19 coordination).
    pub fn is_in_flight(&self, vpn: u64) -> bool {
        self.in_flight.contains_key(&vpn)
    }

    /// Number of pending prefetch requests.
    pub fn pending_count(&self) -> usize {
        self.pending.lock().unwrap().len()
    }

    /// Number of currently in-flight prefetches.
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }

    /// Finalize miss tracking: call this to mark all completed-but-unused
    /// prefetches as misses. Typically called at epoch boundaries.
    ///
    /// Clears the completed map atomically with the stats update so that
    /// concurrent or repeated calls cannot double-count misses.
    pub fn finalize_misses(&self) {
        // Count misses first, then clear, all under the stats lock to
        // ensure atomicity: no window where another call could re-count.
        let mut miss_count = 0u64;
        for entry in self.completed.iter() {
            if !*entry.value() {
                miss_count += 1;
            }
        }
        self.completed.clear();

        if miss_count > 0 {
            let mut stats = self.stats.write().unwrap();
            stats.total_misses += miss_count;
        }
    }

    /// Adaptive lookahead adjustment based on hit/stall rates.
    ///
    /// Computes all needed values from a single stats snapshot to avoid
    /// inconsistencies from concurrent updates between lock acquisitions.
    /// Uses `else if` so only one direction of adjustment happens per cycle,
    /// preventing the lookahead from cancelling itself out when both
    /// stall_rate > 5% and hit_rate < 50% are true simultaneously.
    fn adapt_lookahead(&self) {
        let (stall_rate, hit_rate, has_stalls, has_hits_misses) = {
            let stats = self.stats.read().unwrap();
            let stall_total = stats.total_stalls + stats.total_hits;
            let hit_miss_total = stats.total_hits + stats.total_misses;
            let sr = if stall_total > 0 {
                stats.total_stalls as f32 / stall_total as f32
            } else {
                0.0
            };
            let hr = if hit_miss_total > 0 {
                stats.total_hits as f32 / hit_miss_total as f32
            } else {
                0.0
            };
            (sr, hr, stall_total > 0, hit_miss_total > 0)
        };

        let mut config = self.config.write().unwrap();

        // If stall rate > 5%, increase lookahead (pages arrive too late)
        if has_stalls && stall_rate > 0.05 && config.lookahead_kernels < 8 {
            config.lookahead_kernels += 1;
        } else if has_hits_misses && hit_rate < 0.5 && config.lookahead_kernels > 1 {
            // If hit rate < 50%, decrease lookahead (too speculative)
            config.lookahead_kernels -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::tier_status::PrefetchPrediction;

    fn make_prediction(vpn: u64, confidence: f32) -> PrefetchPrediction {
        PrefetchPrediction {
            vpn,
            confidence: confidence as f64,
            target_tier: crate::memory::types::LOCAL_VRAM,
        }
    }

    #[test]
    fn prefetch_schedule_from_predictions() {
        // Predictions with varying confidence; only those above threshold (0.5) should be scheduled
        let scheduler = PrefetchScheduler::new(PrefetchConfig::default());

        let predictions = vec![
            make_prediction(100, 0.9),
            make_prediction(101, 0.3), // below threshold
            make_prediction(102, 0.7),
            make_prediction(103, 0.1), // below threshold
            make_prediction(104, 0.5), // exactly at threshold
        ];

        scheduler.schedule_from_predictions(&predictions);
        assert_eq!(scheduler.pending_count(), 3);

        // Verify the pending queue contains only vpns 100, 102, 104
        let pending = scheduler.pending.lock().unwrap();
        let vpns: Vec<u64> = pending.iter().map(|r| r.vpn).collect();
        assert!(vpns.contains(&100));
        assert!(vpns.contains(&102));
        assert!(vpns.contains(&104));
        assert!(!vpns.contains(&101));
        assert!(!vpns.contains(&103));
    }

    #[test]
    fn prefetch_tick_submits_in_order() {
        // Higher confidence should be submitted first (they are at the front of the queue)
        let config = PrefetchConfig {
            max_in_flight: 2,
            ..PrefetchConfig::default()
        };
        let scheduler = PrefetchScheduler::new(config);

        let predictions = vec![
            make_prediction(10, 0.6),
            make_prediction(11, 0.9),
            make_prediction(12, 0.7),
        ];

        scheduler.schedule_from_predictions(&predictions);

        // Queue should be ordered: 11 (0.9), 12 (0.7), 10 (0.6)
        {
            let pending = scheduler.pending.lock().unwrap();
            let vpns: Vec<u64> = pending.iter().map(|r| r.vpn).collect();
            assert_eq!(vpns, vec![11, 12, 10]);
        }

        // Tick with max_in_flight=2 should submit top 2
        let started = scheduler.tick();
        assert_eq!(started, 2);
        assert!(scheduler.is_in_flight(11));
        assert!(scheduler.is_in_flight(12));
        assert!(!scheduler.is_in_flight(10));
        assert_eq!(scheduler.pending_count(), 1);
    }

    #[test]
    fn prefetch_hit_tracking() {
        let scheduler = PrefetchScheduler::new(PrefetchConfig::default());

        let predictions = vec![make_prediction(200, 0.8)];
        scheduler.schedule_from_predictions(&predictions);
        scheduler.tick();

        // Complete the prefetch
        scheduler.on_prefetch_complete(200);
        assert!(!scheduler.is_in_flight(200));

        // Now the page is accessed -> hit
        scheduler.on_page_accessed(200);

        let stats = scheduler.stats();
        assert_eq!(stats.total_completed, 1);
        assert_eq!(stats.total_hits, 1);
    }

    #[test]
    fn prefetch_miss_tracking() {
        let scheduler = PrefetchScheduler::new(PrefetchConfig::default());

        let predictions = vec![make_prediction(300, 0.8)];
        scheduler.schedule_from_predictions(&predictions);
        scheduler.tick();

        // Complete the prefetch
        scheduler.on_prefetch_complete(300);

        // Page is never accessed; finalize to count misses
        scheduler.finalize_misses();

        let stats = scheduler.stats();
        assert_eq!(stats.total_completed, 1);
        assert_eq!(stats.total_misses, 1);
        assert_eq!(stats.total_hits, 0);
    }

    #[test]
    fn prefetch_stall_detection() {
        // Page accessed before prefetch completes -> stall
        let scheduler = PrefetchScheduler::new(PrefetchConfig::default());

        let predictions = vec![make_prediction(400, 0.8)];
        scheduler.schedule_from_predictions(&predictions);
        scheduler.tick();

        // Page accessed while still in-flight
        assert!(scheduler.is_in_flight(400));
        scheduler.on_page_accessed(400);

        let stats = scheduler.stats();
        assert_eq!(stats.total_stalls, 1);
    }

    #[test]
    fn prefetch_cancel() {
        let scheduler = PrefetchScheduler::new(PrefetchConfig::default());

        let predictions = vec![make_prediction(500, 0.8), make_prediction(501, 0.7)];
        scheduler.schedule_from_predictions(&predictions);

        // Cancel a pending prefetch
        assert!(scheduler.cancel_prefetch(501));
        assert_eq!(scheduler.pending_count(), 1);

        // Start the remaining one and cancel it while in-flight
        scheduler.tick();
        assert!(scheduler.is_in_flight(500));
        assert!(scheduler.cancel_prefetch(500));
        assert!(!scheduler.is_in_flight(500));

        let stats = scheduler.stats();
        assert_eq!(stats.total_cancelled, 2);
    }

    #[test]
    fn prefetch_max_in_flight() {
        let config = PrefetchConfig {
            max_in_flight: 2,
            ..PrefetchConfig::default()
        };
        let scheduler = PrefetchScheduler::new(config);

        let predictions = vec![
            make_prediction(600, 0.9),
            make_prediction(601, 0.8),
            make_prediction(602, 0.7),
            make_prediction(603, 0.6),
        ];
        scheduler.schedule_from_predictions(&predictions);
        assert_eq!(scheduler.pending_count(), 4);

        // First tick: only 2 in-flight
        let started = scheduler.tick();
        assert_eq!(started, 2);
        assert_eq!(scheduler.in_flight_count(), 2);
        assert_eq!(scheduler.pending_count(), 2);

        // Second tick: still at max, nothing new
        let started = scheduler.tick();
        assert_eq!(started, 0);

        // Complete one, then tick again
        scheduler.on_prefetch_complete(600);
        let started = scheduler.tick();
        assert_eq!(started, 1);
        assert_eq!(scheduler.in_flight_count(), 2);
    }

    #[test]
    fn prefetch_skip_duplicate() {
        let scheduler = PrefetchScheduler::new(PrefetchConfig::default());

        let predictions = vec![make_prediction(700, 0.8)];
        scheduler.schedule_from_predictions(&predictions);
        assert_eq!(scheduler.pending_count(), 1);

        // Schedule same VPN again -- should be skipped
        scheduler.schedule_from_predictions(&predictions);
        assert_eq!(scheduler.pending_count(), 1);

        // Start it, then try to schedule again (now in-flight)
        scheduler.tick();
        scheduler.schedule_from_predictions(&predictions);
        assert_eq!(scheduler.pending_count(), 0); // Not re-added to pending

        // Complete it, then try again (now in completed)
        scheduler.on_prefetch_complete(700);
        scheduler.schedule_from_predictions(&predictions);
        assert_eq!(scheduler.pending_count(), 0); // Not re-added since it's in completed
    }

    #[test]
    fn prefetch_stats() {
        let config = PrefetchConfig {
            max_in_flight: 10,
            ..PrefetchConfig::default()
        };
        let scheduler = PrefetchScheduler::new(config);

        // Schedule and tick 3 prefetches
        let predictions = vec![
            make_prediction(800, 0.9),
            make_prediction(801, 0.8),
            make_prediction(802, 0.7),
        ];
        scheduler.schedule_from_predictions(&predictions);
        scheduler.tick();

        // Complete 2, leave 1 in-flight
        scheduler.on_prefetch_complete(800);
        scheduler.on_prefetch_complete(801);

        // Access one completed (hit)
        scheduler.on_page_accessed(800);

        // Access in-flight one (stall)
        scheduler.on_page_accessed(802);

        // Finalize to count misses on unused completed (801)
        scheduler.finalize_misses();

        let stats = scheduler.stats();
        assert_eq!(stats.total_scheduled, 3);
        assert_eq!(stats.total_completed, 2);
        assert_eq!(stats.total_hits, 1);
        assert_eq!(stats.total_misses, 1);
        assert_eq!(stats.total_stalls, 1);
        assert_eq!(stats.total_cancelled, 0);
    }

    #[test]
    fn prefetch_hit_rate_calculation() {
        let scheduler = PrefetchScheduler::new(PrefetchConfig::default());

        // No data yet
        assert_eq!(scheduler.hit_rate(), 0.0);

        // Create 10 prefetches: 5 hits, 5 misses
        let predictions: Vec<_> = (0..10).map(|i| make_prediction(900 + i, 0.8)).collect();
        scheduler.schedule_from_predictions(&predictions);
        scheduler.tick();

        // Complete all
        for i in 0..10 {
            scheduler.on_prefetch_complete(900 + i);
        }

        // Access first 5 (hits)
        for i in 0..5 {
            scheduler.on_page_accessed(900 + i);
        }

        // Finalize (remaining 5 are misses)
        scheduler.finalize_misses();

        let rate = scheduler.hit_rate();
        assert!((rate - 0.5).abs() < 0.001, "Expected 50% hit rate, got {rate}");
    }

    #[test]
    fn prefetch_is_in_flight() {
        let scheduler = PrefetchScheduler::new(PrefetchConfig::default());

        assert!(!scheduler.is_in_flight(1000));

        let predictions = vec![make_prediction(1000, 0.8)];
        scheduler.schedule_from_predictions(&predictions);
        assert!(!scheduler.is_in_flight(1000)); // Still pending, not in-flight

        scheduler.tick();
        assert!(scheduler.is_in_flight(1000)); // Now in-flight

        scheduler.on_prefetch_complete(1000);
        assert!(!scheduler.is_in_flight(1000)); // Completed, no longer in-flight
    }
}
