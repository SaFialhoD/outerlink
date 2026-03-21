//! Per-connection session state and resource tracking.
//!
//! In CUDA, the "current context" is per-thread. In OuterLink's remote model,
//! each client connection is the equivalent of a thread -- so the current
//! context must be tracked per-connection, NOT in the shared `GpuBackend`.
//!
//! `ConnectionSession` holds all state that is specific to a single client
//! connection and must NOT leak between connections. It tracks every GPU
//! resource allocated by the connection so they can be cleaned up when the
//! connection drops (gracefully or not).

use std::collections::HashSet;

use outerlink_common::cuda_types::CuResult;
use crate::gpu_backend::GpuBackend;

/// Per-connection state that is NOT shared between connections.
///
/// Created when a client connects, dropped when the connection closes.
/// Passed as `&mut` to [`handle_request`](crate::handler::handle_request) on
/// every request so context-related operations read/write the correct
/// per-connection context.
///
/// Tracks all GPU resources allocated by this connection so they can be
/// cleaned up when the connection drops.
pub struct ConnectionSession {
    /// The current CUDA context handle for this connection (0 = none).
    current_ctx: u64,

    /// Device memory allocations owned by this session.
    mem_allocations: HashSet<u64>,
    /// Pinned host memory allocations owned by this session.
    host_allocations: HashSet<u64>,
    /// CUDA contexts created by this session.
    contexts: HashSet<u64>,
    /// Loaded modules owned by this session.
    modules: HashSet<u64>,
    /// CUDA streams created by this session.
    streams: HashSet<u64>,
    /// CUDA events created by this session.
    events: HashSet<u64>,
}

impl ConnectionSession {
    /// Create a new session with no current context and no tracked resources.
    pub fn new() -> Self {
        Self {
            current_ctx: 0,
            mem_allocations: HashSet::new(),
            host_allocations: HashSet::new(),
            contexts: HashSet::new(),
            modules: HashSet::new(),
            streams: HashSet::new(),
            events: HashSet::new(),
        }
    }

    // --- Current context management (unchanged) ---

    /// Get the current context handle for this connection.
    pub fn current_ctx(&self) -> u64 {
        self.current_ctx
    }

    /// Set the current context handle for this connection.
    ///
    /// Setting to 0 clears the current context (equivalent to
    /// `cuCtxSetCurrent(NULL)`).
    pub fn set_current_ctx(&mut self, ctx: u64) {
        self.current_ctx = ctx;
    }

    /// If the current context matches `ctx`, clear it (set to 0).
    ///
    /// Called after a context is destroyed so the session does not hold
    /// a dangling handle.
    pub fn clear_if_current(&mut self, ctx: u64) {
        if self.current_ctx == ctx {
            self.current_ctx = 0;
        }
    }

    /// Validate that a context handle can be set as current.
    ///
    /// `ctx == 0` is always valid (means "unset current"). For non-zero
    /// handles, the caller must verify the handle exists in the backend
    /// before calling [`set_current_ctx`](Self::set_current_ctx).
    pub fn validate_set_current(
        &self,
        ctx: u64,
        ctx_exists: bool,
    ) -> Result<(), CuResult> {
        if ctx == 0 {
            return Ok(());
        }
        if !ctx_exists {
            return Err(CuResult::InvalidContext);
        }
        Ok(())
    }

    // --- Resource tracking ---

    /// Record that this session allocated device memory at `ptr`.
    pub fn track_mem_alloc(&mut self, ptr: u64) {
        self.mem_allocations.insert(ptr);
    }

    /// Remove `ptr` from this session's tracked device memory (e.g. after explicit free).
    pub fn untrack_mem_alloc(&mut self, ptr: u64) {
        self.mem_allocations.remove(&ptr);
    }

    /// Record that this session allocated pinned host memory at `ptr`.
    pub fn track_host_alloc(&mut self, ptr: u64) {
        self.host_allocations.insert(ptr);
    }

    /// Remove `ptr` from this session's tracked host memory.
    pub fn untrack_host_alloc(&mut self, ptr: u64) {
        self.host_allocations.remove(&ptr);
    }

    /// Record that this session created a CUDA context with handle `ctx`.
    pub fn track_context(&mut self, ctx: u64) {
        self.contexts.insert(ctx);
    }

    /// Remove `ctx` from this session's tracked contexts.
    pub fn untrack_context(&mut self, ctx: u64) {
        self.contexts.remove(&ctx);
    }

    /// Record that this session loaded a module with handle `module`.
    pub fn track_module(&mut self, module: u64) {
        self.modules.insert(module);
    }

    /// Remove `module` from this session's tracked modules.
    pub fn untrack_module(&mut self, module: u64) {
        self.modules.remove(&module);
    }

    /// Record that this session created a stream with handle `stream`.
    pub fn track_stream(&mut self, stream: u64) {
        self.streams.insert(stream);
    }

    /// Remove `stream` from this session's tracked streams.
    pub fn untrack_stream(&mut self, stream: u64) {
        self.streams.remove(&stream);
    }

    /// Record that this session created an event with handle `event`.
    pub fn track_event(&mut self, event: u64) {
        self.events.insert(event);
    }

    /// Remove `event` from this session's tracked events.
    pub fn untrack_event(&mut self, event: u64) {
        self.events.remove(&event);
    }

    // --- Resource queries ---

    /// Number of device memory allocations tracked by this session.
    pub fn mem_alloc_count(&self) -> usize {
        self.mem_allocations.len()
    }

    /// Number of host memory allocations tracked by this session.
    pub fn host_alloc_count(&self) -> usize {
        self.host_allocations.len()
    }

    /// Number of contexts tracked by this session.
    pub fn context_count(&self) -> usize {
        self.contexts.len()
    }

    /// Number of modules tracked by this session.
    pub fn module_count(&self) -> usize {
        self.modules.len()
    }

    /// Number of streams tracked by this session.
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }

    /// Number of events tracked by this session.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Total number of tracked resources across all categories.
    pub fn total_tracked_resources(&self) -> usize {
        self.mem_allocations.len()
            + self.host_allocations.len()
            + self.contexts.len()
            + self.modules.len()
            + self.streams.len()
            + self.events.len()
    }

    // --- Cleanup ---

    /// Clean up all GPU resources owned by this session.
    ///
    /// Called when a client connection drops. Frees all tracked resources
    /// via the backend in the correct order:
    /// 1. Events (depend on streams)
    /// 2. Streams
    /// 3. Modules
    /// 4. Device memory
    /// 5. Host memory
    /// 6. Contexts (should be last -- other resources may depend on them)
    ///
    /// Returns the number of resources that were successfully cleaned up
    /// and the number that failed.
    pub fn cleanup(&mut self, backend: &dyn GpuBackend) -> CleanupReport {
        let mut report = CleanupReport::default();

        // 1. Events
        for event in self.events.drain() {
            match backend.event_destroy(event) {
                Ok(()) => {
                    tracing::info!(handle = event, "session cleanup: destroyed event");
                    report.succeeded += 1;
                }
                Err(e) => {
                    tracing::warn!(handle = event, error = ?e, "session cleanup: failed to destroy event");
                    report.failed += 1;
                }
            }
        }

        // 2. Streams
        for stream in self.streams.drain() {
            match backend.stream_destroy(stream) {
                Ok(()) => {
                    tracing::info!(handle = stream, "session cleanup: destroyed stream");
                    report.succeeded += 1;
                }
                Err(e) => {
                    tracing::warn!(handle = stream, error = ?e, "session cleanup: failed to destroy stream");
                    report.failed += 1;
                }
            }
        }

        // 3. Modules
        for module in self.modules.drain() {
            match backend.module_unload(module) {
                Ok(()) => {
                    tracing::info!(handle = module, "session cleanup: unloaded module");
                    report.succeeded += 1;
                }
                Err(e) => {
                    tracing::warn!(handle = module, error = ?e, "session cleanup: failed to unload module");
                    report.failed += 1;
                }
            }
        }

        // 4. Device memory
        for ptr in self.mem_allocations.drain() {
            match backend.mem_free(ptr) {
                Ok(()) => {
                    tracing::info!(ptr = ptr, "session cleanup: freed device memory");
                    report.succeeded += 1;
                }
                Err(e) => {
                    tracing::warn!(ptr = ptr, error = ?e, "session cleanup: failed to free device memory");
                    report.failed += 1;
                }
            }
        }

        // 5. Host memory
        for ptr in self.host_allocations.drain() {
            match backend.mem_free_host(ptr) {
                Ok(()) => {
                    tracing::info!(ptr = ptr, "session cleanup: freed host memory");
                    report.succeeded += 1;
                }
                Err(e) => {
                    tracing::warn!(ptr = ptr, error = ?e, "session cleanup: failed to free host memory");
                    report.failed += 1;
                }
            }
        }

        // 6. Contexts (last)
        for ctx in self.contexts.drain() {
            match backend.ctx_destroy(ctx) {
                Ok(()) => {
                    tracing::info!(handle = ctx, "session cleanup: destroyed context");
                    report.succeeded += 1;
                }
                Err(e) => {
                    tracing::warn!(handle = ctx, error = ?e, "session cleanup: failed to destroy context");
                    report.failed += 1;
                }
            }
        }

        self.current_ctx = 0;

        tracing::debug!(
            succeeded = report.succeeded,
            failed = report.failed,
            "session cleanup complete"
        );

        report
    }
}

/// Summary of a session cleanup operation.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct CleanupReport {
    /// Number of resources successfully freed/destroyed.
    pub succeeded: usize,
    /// Number of resources that failed to free/destroy.
    pub failed: usize,
}

impl Default for ConnectionSession {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_backend::StubGpuBackend;

    // --- Current context tests (unchanged) ---

    #[test]
    fn test_new_session_has_no_current_ctx() {
        let session = ConnectionSession::new();
        assert_eq!(session.current_ctx(), 0);
    }

    #[test]
    fn test_set_and_get_current_ctx() {
        let mut session = ConnectionSession::new();
        session.set_current_ctx(0xC000_0000_0000_0001);
        assert_eq!(session.current_ctx(), 0xC000_0000_0000_0001);
    }

    #[test]
    fn test_clear_if_current_matches() {
        let mut session = ConnectionSession::new();
        session.set_current_ctx(42);
        session.clear_if_current(42);
        assert_eq!(session.current_ctx(), 0);
    }

    #[test]
    fn test_clear_if_current_no_match() {
        let mut session = ConnectionSession::new();
        session.set_current_ctx(42);
        session.clear_if_current(99);
        assert_eq!(session.current_ctx(), 42);
    }

    #[test]
    fn test_set_current_to_zero_clears() {
        let mut session = ConnectionSession::new();
        session.set_current_ctx(42);
        session.set_current_ctx(0);
        assert_eq!(session.current_ctx(), 0);
    }

    #[test]
    fn test_validate_set_current_zero_always_ok() {
        let session = ConnectionSession::new();
        assert!(session.validate_set_current(0, false).is_ok());
    }

    #[test]
    fn test_validate_set_current_nonzero_exists() {
        let session = ConnectionSession::new();
        assert!(session.validate_set_current(42, true).is_ok());
    }

    #[test]
    fn test_validate_set_current_nonzero_not_exists() {
        let session = ConnectionSession::new();
        assert_eq!(
            session.validate_set_current(42, false),
            Err(CuResult::InvalidContext)
        );
    }

    #[test]
    fn test_two_sessions_are_independent() {
        let mut session_a = ConnectionSession::new();
        let mut session_b = ConnectionSession::new();

        session_a.set_current_ctx(0xAAAA);
        session_b.set_current_ctx(0xBBBB);

        // Each session has its own current context.
        assert_eq!(session_a.current_ctx(), 0xAAAA);
        assert_eq!(session_b.current_ctx(), 0xBBBB);

        // Clearing one does not affect the other.
        session_a.set_current_ctx(0);
        assert_eq!(session_a.current_ctx(), 0);
        assert_eq!(session_b.current_ctx(), 0xBBBB);
    }

    // --- Resource tracking tests ---

    #[test]
    fn test_new_session_has_no_tracked_resources() {
        let session = ConnectionSession::new();
        assert_eq!(session.total_tracked_resources(), 0);
        assert_eq!(session.mem_alloc_count(), 0);
        assert_eq!(session.host_alloc_count(), 0);
        assert_eq!(session.context_count(), 0);
        assert_eq!(session.module_count(), 0);
        assert_eq!(session.stream_count(), 0);
        assert_eq!(session.event_count(), 0);
    }

    #[test]
    fn test_track_and_untrack_mem_alloc() {
        let mut session = ConnectionSession::new();
        session.track_mem_alloc(0xDEAD_0001);
        session.track_mem_alloc(0xDEAD_0002);
        assert_eq!(session.mem_alloc_count(), 2);

        session.untrack_mem_alloc(0xDEAD_0001);
        assert_eq!(session.mem_alloc_count(), 1);

        // Untracking a non-existent handle is a no-op.
        session.untrack_mem_alloc(0xDEAD_9999);
        assert_eq!(session.mem_alloc_count(), 1);
    }

    #[test]
    fn test_track_and_untrack_host_alloc() {
        let mut session = ConnectionSession::new();
        session.track_host_alloc(0xCAFE_0001);
        assert_eq!(session.host_alloc_count(), 1);
        session.untrack_host_alloc(0xCAFE_0001);
        assert_eq!(session.host_alloc_count(), 0);
    }

    #[test]
    fn test_track_and_untrack_context() {
        let mut session = ConnectionSession::new();
        session.track_context(0xC001);
        session.track_context(0xC002);
        assert_eq!(session.context_count(), 2);
        session.untrack_context(0xC001);
        assert_eq!(session.context_count(), 1);
    }

    #[test]
    fn test_track_and_untrack_module() {
        let mut session = ConnectionSession::new();
        session.track_module(0xA001);
        assert_eq!(session.module_count(), 1);
        session.untrack_module(0xA001);
        assert_eq!(session.module_count(), 0);
    }

    #[test]
    fn test_track_and_untrack_stream() {
        let mut session = ConnectionSession::new();
        session.track_stream(0x5001);
        session.track_stream(0x5002);
        assert_eq!(session.stream_count(), 2);
        session.untrack_stream(0x5002);
        assert_eq!(session.stream_count(), 1);
    }

    #[test]
    fn test_track_and_untrack_event() {
        let mut session = ConnectionSession::new();
        session.track_event(0xE001);
        assert_eq!(session.event_count(), 1);
        session.untrack_event(0xE001);
        assert_eq!(session.event_count(), 0);
    }

    #[test]
    fn test_total_tracked_resources() {
        let mut session = ConnectionSession::new();
        session.track_mem_alloc(1);
        session.track_host_alloc(2);
        session.track_context(3);
        session.track_module(4);
        session.track_stream(5);
        session.track_event(6);
        assert_eq!(session.total_tracked_resources(), 6);
    }

    #[test]
    fn test_duplicate_track_is_idempotent() {
        let mut session = ConnectionSession::new();
        session.track_mem_alloc(0xDEAD_0001);
        session.track_mem_alloc(0xDEAD_0001); // duplicate
        assert_eq!(session.mem_alloc_count(), 1);
    }

    // --- Two sessions with independent handle spaces ---

    #[test]
    fn test_two_sessions_independent_resource_tracking() {
        let mut session_a = ConnectionSession::new();
        let mut session_b = ConnectionSession::new();

        session_a.track_mem_alloc(0xDEAD_0001);
        session_a.track_stream(0x5001);
        session_b.track_mem_alloc(0xDEAD_0002);
        session_b.track_event(0xE001);

        assert_eq!(session_a.mem_alloc_count(), 1);
        assert_eq!(session_a.stream_count(), 1);
        assert_eq!(session_a.event_count(), 0);

        assert_eq!(session_b.mem_alloc_count(), 1);
        assert_eq!(session_b.stream_count(), 0);
        assert_eq!(session_b.event_count(), 1);
    }

    // --- Cleanup tests using the real StubGpuBackend ---

    #[test]
    fn test_cleanup_frees_all_resources() {
        let backend = StubGpuBackend::new();
        let mut session = ConnectionSession::new();

        // Allocate real resources via the backend and track them.
        let ptr = backend.mem_alloc(1024).unwrap();
        session.track_mem_alloc(ptr);

        let host_ptr = backend.mem_alloc_host(512).unwrap();
        session.track_host_alloc(host_ptr);

        let ctx = backend.ctx_create(0, 0).unwrap();
        session.track_context(ctx);
        session.set_current_ctx(ctx);

        let module = backend.module_load_data(b"fake_ptx").unwrap();
        session.track_module(module);

        let stream = backend.stream_create(0).unwrap();
        session.track_stream(stream);

        let event = backend.event_create(0).unwrap();
        session.track_event(event);

        assert_eq!(session.total_tracked_resources(), 6);

        let report = session.cleanup(&backend);
        assert_eq!(report.succeeded, 6);
        assert_eq!(report.failed, 0);
        assert_eq!(session.total_tracked_resources(), 0);
        assert_eq!(session.current_ctx(), 0);

        // Verify the backend actually freed the resources: re-freeing should fail.
        assert!(backend.mem_free(ptr).is_err());
        assert!(backend.ctx_destroy(ctx).is_err());
        assert!(backend.stream_destroy(stream).is_err());
        assert!(backend.event_destroy(event).is_err());
        assert!(backend.module_unload(module).is_err());
        assert!(backend.mem_free_host(host_ptr).is_err());
    }

    #[test]
    fn test_cleanup_one_session_does_not_affect_other() {
        let backend = StubGpuBackend::new();
        let mut session_a = ConnectionSession::new();
        let mut session_b = ConnectionSession::new();

        // Session A allocates memory.
        let ptr_a = backend.mem_alloc(1024).unwrap();
        session_a.track_mem_alloc(ptr_a);

        // Session B allocates memory.
        let ptr_b = backend.mem_alloc(2048).unwrap();
        session_b.track_mem_alloc(ptr_b);

        // Clean up session A.
        let report = session_a.cleanup(&backend);
        assert_eq!(report.succeeded, 1);

        // Session B's memory should still be accessible.
        let data = backend.memcpy_dtoh(ptr_b, 8);
        assert!(data.is_ok(), "session B's allocation should still be valid");

        // Session A's memory should be gone.
        assert!(backend.mem_free(ptr_a).is_err());
    }

    #[test]
    fn test_cleanup_empty_session_is_noop() {
        let backend = StubGpuBackend::new();
        let mut session = ConnectionSession::new();
        let report = session.cleanup(&backend);
        assert_eq!(report.succeeded, 0);
        assert_eq!(report.failed, 0);
    }

    #[test]
    fn test_cleanup_with_already_freed_resource_reports_failure() {
        let backend = StubGpuBackend::new();
        let mut session = ConnectionSession::new();

        // Allocate and track.
        let ptr = backend.mem_alloc(256).unwrap();
        session.track_mem_alloc(ptr);

        // Free it manually (simulating backend-level corruption — not a
        // normal multi-session case, since resources are strictly per-session).
        let _ = backend.mem_free(ptr);

        // Cleanup should report a failure for this resource.
        let report = session.cleanup(&backend);
        assert_eq!(report.failed, 1);
        assert_eq!(report.succeeded, 0);
    }
}
