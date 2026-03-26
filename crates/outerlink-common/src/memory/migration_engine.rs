//! TieredMigrationEngine: concrete implementation of the MigrationEngine trait.
//!
//! Manages asynchronous page migrations between memory tiers with priority
//! scheduling, bandwidth limiting, and concurrent migration tracking.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use dashmap::DashMap;

use super::migration::{
    MigrationError, MigrationHandle, MigrationPriority, MigrationRequest, MigrationStatus,
};
use super::pte::PteFlags;
use super::traits::{MigrationEngine, PageTable};

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// Default page size for migration transfers (64 KiB).
const DEFAULT_PAGE_SIZE: usize = 65536;

/// A migration request wrapped with scheduling metadata for the priority queue.
struct PrioritizedRequest {
    /// The scheduling priority.
    priority: MigrationPriority,
    /// The original migration request.
    request: MigrationRequest,
    /// Assigned migration ID.
    id: u64,
    /// When this request was submitted.
    submitted_at: Instant,
}

impl Eq for PrioritizedRequest {}

impl PartialEq for PrioritizedRequest {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

/// Ordering: higher priority first, then earlier submission (FIFO within same priority).
impl Ord for PrioritizedRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => {
                // Earlier submitted_at = higher priority (reverse ordering on time).
                other.submitted_at.cmp(&self.submitted_at)
            }
            ord => ord,
        }
    }
}

impl PartialOrd for PrioritizedRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Tracks the state of an in-flight migration.
struct InFlightState {
    /// The original request.
    request: MigrationRequest,
    /// Bytes transferred so far.
    bytes_transferred: usize,
    /// Total bytes to transfer.
    total_bytes: usize,
    /// When the transfer started.
    #[allow(dead_code)]
    started_at: Instant,
}

/// Token bucket for bandwidth limiting.
struct BandwidthLimiter {
    /// Maximum bytes per second (0 = unlimited).
    limit_bytes_per_sec: u64,
    /// Available tokens (bytes we can still send this period).
    tokens: u64,
    /// When tokens were last refilled.
    last_refill: Instant,
}

impl BandwidthLimiter {
    fn new(limit: u64) -> Self {
        Self {
            limit_bytes_per_sec: limit,
            tokens: limit,
            last_refill: Instant::now(),
        }
    }

    /// Refill tokens based on elapsed time, then try to consume `bytes`.
    /// Returns true if the bytes were consumed.
    fn try_consume(&mut self, bytes: u64) -> bool {
        if self.limit_bytes_per_sec == 0 {
            return true; // unlimited
        }
        self.refill();
        if self.tokens >= bytes {
            self.tokens -= bytes;
            true
        } else {
            false
        }
    }

    /// Refill tokens based on elapsed time since last refill.
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill);
        let new_tokens =
            (elapsed.as_secs_f64() * self.limit_bytes_per_sec as f64) as u64;
        if new_tokens > 0 {
            self.tokens = (self.tokens + new_tokens).min(self.limit_bytes_per_sec);
            self.last_refill = now;
        }
    }

    /// Current utilization as fraction of limit. Returns bytes consumed recently.
    fn utilization(&self) -> f64 {
        if self.limit_bytes_per_sec == 0 {
            return 0.0;
        }
        // Utilization = 1.0 - (remaining_tokens / limit)
        let remaining_fraction =
            self.tokens as f64 / self.limit_bytes_per_sec as f64;
        (1.0 - remaining_fraction).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// TieredMigrationEngine
// ---------------------------------------------------------------------------

/// Concrete migration engine that manages page migrations between tiers.
///
/// Uses a priority queue for pending migrations, DashMap for in-flight tracking,
/// and a token bucket for bandwidth limiting. Thread-safe for concurrent access.
pub struct TieredMigrationEngine {
    /// Pending migrations ordered by priority.
    pending: Mutex<BinaryHeap<PrioritizedRequest>>,
    /// In-flight migrations indexed by migration ID.
    in_flight: DashMap<u64, InFlightState>,
    /// Completed/cancelled/failed migrations (kept for poll).
    completed: DashMap<u64, MigrationStatus>,
    /// Bandwidth limiter.
    bandwidth: Mutex<BandwidthLimiter>,
    /// Monotonic migration ID counter.
    next_id: AtomicU64,
    /// Maximum concurrent in-flight migrations.
    max_concurrent: usize,
    /// Optional page table for PTE flag updates during migration.
    page_table: Option<Arc<dyn PageTable>>,
    /// Total bytes transferred (for bandwidth tracking).
    total_bytes_transferred: AtomicU64,
}

impl TieredMigrationEngine {
    /// Create a new migration engine with the given concurrency limit.
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            pending: Mutex::new(BinaryHeap::new()),
            in_flight: DashMap::new(),
            completed: DashMap::new(),
            bandwidth: Mutex::new(BandwidthLimiter::new(0)), // unlimited by default
            next_id: AtomicU64::new(1),
            max_concurrent,
            page_table: None,
            total_bytes_transferred: AtomicU64::new(0),
        }
    }

    /// Create a migration engine with a page table reference for PTE updates.
    pub fn with_page_table(max_concurrent: usize, page_table: Arc<dyn PageTable>) -> Self {
        Self {
            pending: Mutex::new(BinaryHeap::new()),
            in_flight: DashMap::new(),
            completed: DashMap::new(),
            bandwidth: Mutex::new(BandwidthLimiter::new(0)),
            next_id: AtomicU64::new(1),
            max_concurrent,
            page_table: Some(page_table),
            total_bytes_transferred: AtomicU64::new(0),
        }
    }

    /// Allocate a new unique migration ID.
    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, AtomicOrdering::Relaxed)
    }

    /// Try to start pending migrations if there are available slots and bandwidth.
    fn try_drain_pending(&self) {
        let mut pending = self.pending.lock().unwrap();
        let mut bw = self.bandwidth.lock().unwrap();

        while self.in_flight.len() < self.max_concurrent {
            if pending.peek().is_some() {
                let page_bytes = DEFAULT_PAGE_SIZE as u64;
                if !bw.try_consume(page_bytes) {
                    break; // no bandwidth available
                }
                let req = pending.pop().unwrap();
                self.start_migration(req);
            } else {
                break;
            }
        }
    }

    /// Move a prioritized request into the in-flight state.
    fn start_migration(&self, req: PrioritizedRequest) {
        // Set MIGRATING flag on the source PTE if we have a page table.
        if let Some(ref pt) = self.page_table {
            pt.update_flags(req.request.vpn, PteFlags::MIGRATING, PteFlags::empty());
        }

        self.in_flight.insert(
            req.id,
            InFlightState {
                request: req.request,
                bytes_transferred: 0,
                total_bytes: DEFAULT_PAGE_SIZE,
                started_at: Instant::now(),
            },
        );
    }

    /// Complete a migration: move from in-flight to completed, update PTE.
    pub fn complete_migration(&self, handle_id: u64) -> Result<(), MigrationError> {
        let state = self
            .in_flight
            .remove(&handle_id)
            .map(|(_, v)| v)
            .ok_or(MigrationError::InvalidState(handle_id))?;

        // Track bandwidth usage.
        self.total_bytes_transferred
            .fetch_add(state.total_bytes as u64, AtomicOrdering::Relaxed);

        // Update PTE: commit the migration in the page table.
        if let Some(ref pt) = self.page_table {
            // Use commit_migration which clears MIGRATING and updates tier/node.
            // We use a placeholder phys_pfn of 0 since the actual physical address
            // is determined by the transport layer, not the engine.
            let _ = pt.commit_migration(
                state.request.vpn,
                state.request.dst_tier,
                state.request.dst_node,
                0, // phys_pfn set by transport
            );
        }

        self.completed.insert(handle_id, MigrationStatus::Completed);

        // Try to start more pending migrations now that a slot freed up.
        self.try_drain_pending();

        Ok(())
    }

    /// Simulate progress on an in-flight migration (for testing / polling).
    /// In production, the transport layer would call this as bytes arrive.
    pub fn report_progress(&self, handle_id: u64, bytes: usize) {
        if let Some(mut entry) = self.in_flight.get_mut(&handle_id) {
            entry.bytes_transferred = entry.bytes_transferred.saturating_add(bytes);
            if entry.bytes_transferred >= entry.total_bytes {
                drop(entry);
                let _ = self.complete_migration(handle_id);
            }
        }
    }
}

impl MigrationEngine for TieredMigrationEngine {
    fn submit(&self, request: MigrationRequest) -> Result<MigrationHandle, MigrationError> {
        // Check if page is already migrating (if we have a page table).
        if let Some(ref pt) = self.page_table {
            if let Some(pte) = pt.lookup(request.vpn) {
                if pte.has_flag(PteFlags::MIGRATING) {
                    return Err(MigrationError::AlreadyMigrating(request.vpn));
                }
                if pte.has_flag(PteFlags::PINNED) {
                    return Err(MigrationError::PagePinned(request.vpn));
                }
            }
        }

        let id = self.next_id();
        let now = Instant::now();
        let priority = request.priority;

        let handle = MigrationHandle {
            id,
            request: request.clone(),
            status: MigrationStatus::Pending,
            submitted_at: now,
        };

        let prioritized = PrioritizedRequest {
            priority,
            request,
            id,
            submitted_at: now,
        };

        {
            let mut pending = self.pending.lock().unwrap();
            pending.push(prioritized);
        }

        // Try to start it immediately if there's capacity.
        self.try_drain_pending();

        Ok(handle)
    }

    fn submit_batch(
        &self,
        requests: Vec<MigrationRequest>,
    ) -> Vec<Result<MigrationHandle, MigrationError>> {
        let mut results = Vec::with_capacity(requests.len());

        // Insert all into pending queue first (avoids repeated lock/unlock of drain).
        let now = Instant::now();
        {
            let mut pending = self.pending.lock().unwrap();
            for request in requests {
                // Check constraints.
                if let Some(ref pt) = self.page_table {
                    if let Some(pte) = pt.lookup(request.vpn) {
                        if pte.has_flag(PteFlags::MIGRATING) {
                            results.push(Err(MigrationError::AlreadyMigrating(request.vpn)));
                            continue;
                        }
                        if pte.has_flag(PteFlags::PINNED) {
                            results.push(Err(MigrationError::PagePinned(request.vpn)));
                            continue;
                        }
                    }
                }

                let id = self.next_id();
                let priority = request.priority;
                let handle = MigrationHandle {
                    id,
                    request: request.clone(),
                    status: MigrationStatus::Pending,
                    submitted_at: now,
                };

                pending.push(PrioritizedRequest {
                    priority,
                    request,
                    id,
                    submitted_at: now,
                });

                results.push(Ok(handle));
            }
        }

        // Now drain pending into in-flight.
        self.try_drain_pending();

        results
    }

    fn poll(&self, handle_id: u64) -> Option<MigrationStatus> {
        // Check completed first.
        if let Some(entry) = self.completed.get(&handle_id) {
            return Some(entry.value().clone());
        }

        // Check in-flight.
        if let Some(entry) = self.in_flight.get(&handle_id) {
            return Some(MigrationStatus::InFlight {
                bytes_transferred: entry.bytes_transferred,
                total_bytes: entry.total_bytes,
            });
        }

        // Check pending queue (linear scan, but pending queues should be small).
        {
            let pending = self.pending.lock().unwrap();
            for req in pending.iter() {
                if req.id == handle_id {
                    return Some(MigrationStatus::Pending);
                }
            }
        }

        None
    }

    fn wait(&self, handle_id: u64) -> Result<MigrationStatus, MigrationError> {
        // Busy-poll with small sleep. In production, this would use a condvar.
        loop {
            match self.poll(handle_id) {
                Some(MigrationStatus::Completed) => return Ok(MigrationStatus::Completed),
                Some(MigrationStatus::Failed) => return Ok(MigrationStatus::Failed),
                Some(MigrationStatus::Cancelled) => {
                    return Err(MigrationError::Cancelled(handle_id))
                }
                Some(MigrationStatus::Pending) | Some(MigrationStatus::InFlight { .. }) => {
                    std::thread::sleep(std::time::Duration::from_micros(100));
                }
                None => return Err(MigrationError::NotFound(handle_id)),
            }
        }
    }

    fn cancel(&self, handle_id: u64) -> Result<(), MigrationError> {
        // Try to remove from in-flight first.
        if let Some((_, state)) = self.in_flight.remove(&handle_id) {
            // Clear MIGRATING flag on the PTE.
            if let Some(ref pt) = self.page_table {
                pt.update_flags(state.request.vpn, PteFlags::empty(), PteFlags::MIGRATING);
            }
            self.completed
                .insert(handle_id, MigrationStatus::Cancelled);
            // Free up a slot.
            self.try_drain_pending();
            return Ok(());
        }

        // Try to remove from pending queue.
        {
            let mut pending = self.pending.lock().unwrap();
            let before_len = pending.len();
            let items: Vec<_> = std::mem::take(&mut *pending)
                .into_vec()
                .into_iter()
                .filter(|r| r.id != handle_id)
                .collect();
            let removed = before_len > items.len();
            *pending = BinaryHeap::from(items);

            if removed {
                drop(pending);
                self.completed
                    .insert(handle_id, MigrationStatus::Cancelled);
                return Ok(());
            }
        }

        // Check if already completed.
        if self.completed.contains_key(&handle_id) {
            return Err(MigrationError::InvalidState(handle_id));
        }

        Err(MigrationError::NotFound(handle_id))
    }

    fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }

    fn bandwidth_utilization(&self) -> f64 {
        let bw = self.bandwidth.lock().unwrap();
        bw.utilization()
    }

    fn set_bandwidth_limit(&self, bytes_per_sec: u64) {
        let mut bw = self.bandwidth.lock().unwrap();
        bw.limit_bytes_per_sec = bytes_per_sec;
        bw.tokens = bytes_per_sec;
        bw.last_refill = Instant::now();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::migration::{MigrationPriority, MigrationReason, MigrationRequest};
    use crate::memory::page_table::RobinHoodPageTable;
    use crate::memory::pte::PageTableEntry;

    /// Helper to create a migration request with sensible defaults.
    fn make_request(vpn: u64, priority: MigrationPriority) -> MigrationRequest {
        MigrationRequest {
            vpn,
            src_tier: 0,
            src_node: 0,
            dst_tier: 2,
            dst_node: 0,
            priority,
            reason: MigrationReason::Promotion,
            compress: false,
            update_parity: false,
        }
    }

    /// Helper to create a page table with some pages.
    fn make_page_table(vpns: &[u64]) -> Arc<RobinHoodPageTable> {
        let pt = Arc::new(RobinHoodPageTable::new());
        for &vpn in vpns {
            pt.upsert(PageTableEntry::new(vpn, 0, 0));
        }
        pt
    }

    // -------------------------------------------------------------------
    // 1. submit and poll
    // -------------------------------------------------------------------

    #[test]
    fn migration_engine_submit_and_poll() {
        let engine = TieredMigrationEngine::new(4);
        let req = make_request(0x1000, MigrationPriority::Normal);
        let handle = engine.submit(req).unwrap();

        // Should be either Pending or InFlight (since max_concurrent=4, it should start).
        let status = engine.poll(handle.id).unwrap();
        assert!(
            matches!(status, MigrationStatus::InFlight { .. }),
            "expected InFlight, got {:?}",
            status
        );
    }

    // -------------------------------------------------------------------
    // 2. submit_batch
    // -------------------------------------------------------------------

    #[test]
    fn migration_engine_submit_batch() {
        let engine = TieredMigrationEngine::new(10);
        let requests: Vec<_> = (0..5)
            .map(|i| make_request(0x1000 + i, MigrationPriority::Normal))
            .collect();

        let results = engine.submit_batch(requests);
        assert_eq!(results.len(), 5);

        for result in &results {
            assert!(result.is_ok());
        }

        // All 5 should be handles with unique IDs.
        let ids: Vec<u64> = results.iter().map(|r| r.as_ref().unwrap().id).collect();
        let mut unique_ids = ids.clone();
        unique_ids.sort();
        unique_ids.dedup();
        assert_eq!(unique_ids.len(), 5, "all IDs should be unique");
    }

    // -------------------------------------------------------------------
    // 3. priority ordering
    // -------------------------------------------------------------------

    #[test]
    fn migration_engine_priority_ordering() {
        // max_concurrent=1 so only one can start at a time.
        let engine = TieredMigrationEngine::new(1);

        // Submit a low-priority request first - it will start immediately (slot available).
        let low_handle = engine
            .submit(make_request(0x1000, MigrationPriority::Background))
            .unwrap();

        // Now submit both Normal and Fault while slot is occupied.
        let normal_handle = engine
            .submit(make_request(0x2000, MigrationPriority::Normal))
            .unwrap();
        let fault_handle = engine
            .submit(make_request(0x3000, MigrationPriority::Fault))
            .unwrap();

        // low is in-flight, normal and fault are pending.
        assert!(matches!(
            engine.poll(low_handle.id),
            Some(MigrationStatus::InFlight { .. })
        ));
        assert!(matches!(
            engine.poll(normal_handle.id),
            Some(MigrationStatus::Pending)
        ));
        assert!(matches!(
            engine.poll(fault_handle.id),
            Some(MigrationStatus::Pending)
        ));

        // Complete the low-priority one. Fault should start next (higher priority).
        engine.complete_migration(low_handle.id).unwrap();

        assert!(
            matches!(
                engine.poll(fault_handle.id),
                Some(MigrationStatus::InFlight { .. })
            ),
            "fault priority should start before normal"
        );
        assert!(matches!(
            engine.poll(normal_handle.id),
            Some(MigrationStatus::Pending)
        ));
    }

    // -------------------------------------------------------------------
    // 4. cancel
    // -------------------------------------------------------------------

    #[test]
    fn migration_engine_cancel() {
        let engine = TieredMigrationEngine::new(4);
        let handle = engine
            .submit(make_request(0x1000, MigrationPriority::Normal))
            .unwrap();

        engine.cancel(handle.id).unwrap();

        let status = engine.poll(handle.id).unwrap();
        assert_eq!(status, MigrationStatus::Cancelled);
    }

    #[test]
    fn migration_engine_cancel_pending() {
        // max_concurrent=1, fill the slot, then cancel a pending one.
        let engine = TieredMigrationEngine::new(1);
        let _first = engine
            .submit(make_request(0x1000, MigrationPriority::Normal))
            .unwrap();
        let second = engine
            .submit(make_request(0x2000, MigrationPriority::Normal))
            .unwrap();

        assert!(matches!(
            engine.poll(second.id),
            Some(MigrationStatus::Pending)
        ));

        engine.cancel(second.id).unwrap();
        assert_eq!(
            engine.poll(second.id).unwrap(),
            MigrationStatus::Cancelled
        );
    }

    // -------------------------------------------------------------------
    // 5. bandwidth limit
    // -------------------------------------------------------------------

    #[test]
    fn migration_engine_bandwidth_limit() {
        let engine = TieredMigrationEngine::new(100);
        // Set a bandwidth limit smaller than one page (64KB).
        // This means only 0 pages can be transferred per refill.
        engine.set_bandwidth_limit(1024); // 1 KB/sec - way less than 64KB page

        // First submit might consume initial tokens.
        let req1 = make_request(0x1000, MigrationPriority::Normal);
        let _h1 = engine.submit(req1);

        // The second should be stuck pending because bandwidth is exhausted.
        let req2 = make_request(0x2000, MigrationPriority::Normal);
        let h2 = engine.submit(req2).unwrap();

        // h2 should remain pending since bandwidth is exhausted.
        let status = engine.poll(h2.id).unwrap();
        assert_eq!(
            status,
            MigrationStatus::Pending,
            "should be pending due to bandwidth limit"
        );
    }

    // -------------------------------------------------------------------
    // 6. max concurrent
    // -------------------------------------------------------------------

    #[test]
    fn migration_engine_max_concurrent() {
        let engine = TieredMigrationEngine::new(2);

        let h1 = engine
            .submit(make_request(0x1000, MigrationPriority::Normal))
            .unwrap();
        let h2 = engine
            .submit(make_request(0x2000, MigrationPriority::Normal))
            .unwrap();
        let h3 = engine
            .submit(make_request(0x3000, MigrationPriority::Normal))
            .unwrap();

        assert!(matches!(
            engine.poll(h1.id),
            Some(MigrationStatus::InFlight { .. })
        ));
        assert!(matches!(
            engine.poll(h2.id),
            Some(MigrationStatus::InFlight { .. })
        ));
        assert!(matches!(
            engine.poll(h3.id),
            Some(MigrationStatus::Pending)
        ));

        assert_eq!(engine.in_flight_count(), 2);
    }

    // -------------------------------------------------------------------
    // 7. in_flight_count
    // -------------------------------------------------------------------

    #[test]
    fn migration_engine_in_flight_count() {
        let engine = TieredMigrationEngine::new(10);
        assert_eq!(engine.in_flight_count(), 0);

        let h1 = engine
            .submit(make_request(0x1000, MigrationPriority::Normal))
            .unwrap();
        assert_eq!(engine.in_flight_count(), 1);

        let _h2 = engine
            .submit(make_request(0x2000, MigrationPriority::Normal))
            .unwrap();
        assert_eq!(engine.in_flight_count(), 2);

        engine.cancel(h1.id).unwrap();
        assert_eq!(engine.in_flight_count(), 1);
    }

    // -------------------------------------------------------------------
    // 8. complete_migration
    // -------------------------------------------------------------------

    #[test]
    fn migration_engine_complete_migration() {
        let engine = TieredMigrationEngine::new(4);
        let handle = engine
            .submit(make_request(0x1000, MigrationPriority::Normal))
            .unwrap();

        // Should be in-flight.
        assert!(matches!(
            engine.poll(handle.id),
            Some(MigrationStatus::InFlight { .. })
        ));

        // Complete it.
        engine.complete_migration(handle.id).unwrap();

        // Should now be completed.
        assert_eq!(
            engine.poll(handle.id).unwrap(),
            MigrationStatus::Completed
        );
        assert_eq!(engine.in_flight_count(), 0);
    }

    // -------------------------------------------------------------------
    // 9. PTE flag updates with page table
    // -------------------------------------------------------------------

    #[test]
    fn migration_engine_sets_migrating_flag() {
        let pt = make_page_table(&[0x1000]);
        let engine = TieredMigrationEngine::with_page_table(4, pt.clone());

        let handle = engine
            .submit(make_request(0x1000, MigrationPriority::Normal))
            .unwrap();

        // PTE should have MIGRATING flag set.
        let pte = pt.lookup(0x1000).unwrap();
        assert!(
            pte.has_flag(PteFlags::MIGRATING),
            "MIGRATING flag should be set on in-flight page"
        );

        // Complete migration.
        engine.complete_migration(handle.id).unwrap();

        // MIGRATING flag should be cleared.
        let pte = pt.lookup(0x1000).unwrap();
        assert!(
            !pte.has_flag(PteFlags::MIGRATING),
            "MIGRATING flag should be cleared after completion"
        );
    }

    #[test]
    fn migration_engine_cancel_clears_migrating_flag() {
        let pt = make_page_table(&[0x1000]);
        let engine = TieredMigrationEngine::with_page_table(4, pt.clone());

        let handle = engine
            .submit(make_request(0x1000, MigrationPriority::Normal))
            .unwrap();

        // MIGRATING should be set.
        let pte = pt.lookup(0x1000).unwrap();
        assert!(pte.has_flag(PteFlags::MIGRATING));

        // Cancel.
        engine.cancel(handle.id).unwrap();

        // MIGRATING should be cleared.
        let pte = pt.lookup(0x1000).unwrap();
        assert!(
            !pte.has_flag(PteFlags::MIGRATING),
            "MIGRATING flag should be cleared after cancel"
        );
    }

    #[test]
    fn migration_engine_rejects_already_migrating() {
        let pt = make_page_table(&[0x1000]);
        let engine = TieredMigrationEngine::with_page_table(4, pt.clone());

        let _h1 = engine
            .submit(make_request(0x1000, MigrationPriority::Normal))
            .unwrap();

        // Submitting the same VPN again should fail.
        let result = engine.submit(make_request(0x1000, MigrationPriority::Normal));
        assert!(matches!(result, Err(MigrationError::AlreadyMigrating(0x1000))));
    }

    #[test]
    fn migration_engine_rejects_pinned_page() {
        let pt = make_page_table(&[0x1000]);
        // Set PINNED flag.
        pt.update_flags(0x1000, PteFlags::PINNED, PteFlags::empty());

        let engine = TieredMigrationEngine::with_page_table(4, pt.clone());

        let result = engine.submit(make_request(0x1000, MigrationPriority::Normal));
        assert!(matches!(result, Err(MigrationError::PagePinned(0x1000))));
    }

    #[test]
    fn migration_engine_commit_updates_pte_tier() {
        let pt = make_page_table(&[0x1000]);
        let engine = TieredMigrationEngine::with_page_table(4, pt.clone());

        // Request: migrate from tier 0 to tier 2.
        let handle = engine
            .submit(make_request(0x1000, MigrationPriority::Normal))
            .unwrap();

        engine.complete_migration(handle.id).unwrap();

        let pte = pt.lookup(0x1000).unwrap();
        assert_eq!(pte.tier_id, 2, "PTE tier should be updated to destination");
        assert_eq!(pte.node_id, 0, "PTE node should be updated to destination");
    }

    // -------------------------------------------------------------------
    // 10. report_progress auto-completes
    // -------------------------------------------------------------------

    #[test]
    fn migration_engine_report_progress_auto_completes() {
        let engine = TieredMigrationEngine::new(4);
        let handle = engine
            .submit(make_request(0x1000, MigrationPriority::Normal))
            .unwrap();

        // Report full page transferred.
        engine.report_progress(handle.id, DEFAULT_PAGE_SIZE);

        // Should be completed now.
        assert_eq!(
            engine.poll(handle.id).unwrap(),
            MigrationStatus::Completed
        );
    }

    // -------------------------------------------------------------------
    // 11. bandwidth utilization reporting
    // -------------------------------------------------------------------

    #[test]
    fn migration_engine_bandwidth_utilization_default() {
        let engine = TieredMigrationEngine::new(4);
        // No limit set = 0.0 utilization.
        assert_eq!(engine.bandwidth_utilization(), 0.0);
    }

    #[test]
    fn migration_engine_set_bandwidth_limit() {
        let engine = TieredMigrationEngine::new(4);
        engine.set_bandwidth_limit(1_000_000);

        // Submit a migration to consume some tokens.
        let _h = engine
            .submit(make_request(0x1000, MigrationPriority::Normal))
            .unwrap();

        // Utilization should be > 0 since we consumed tokens.
        let util = engine.bandwidth_utilization();
        assert!(util > 0.0, "utilization should be non-zero after transfer");
    }

    // -------------------------------------------------------------------
    // 12. drain promotes highest priority first from pending
    // -------------------------------------------------------------------

    #[test]
    fn migration_engine_drain_after_complete_starts_next() {
        let engine = TieredMigrationEngine::new(1);

        let h1 = engine
            .submit(make_request(0x1000, MigrationPriority::Normal))
            .unwrap();
        let h2 = engine
            .submit(make_request(0x2000, MigrationPriority::Normal))
            .unwrap();

        // h1 in flight, h2 pending.
        assert_eq!(engine.in_flight_count(), 1);

        engine.complete_migration(h1.id).unwrap();

        // h2 should now be in flight.
        assert!(matches!(
            engine.poll(h2.id),
            Some(MigrationStatus::InFlight { .. })
        ));
    }
}
