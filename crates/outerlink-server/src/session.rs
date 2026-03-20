//! Per-connection session state.
//!
//! In CUDA, the "current context" is per-thread. In OuterLink's remote model,
//! each client connection is the equivalent of a thread -- so the current
//! context must be tracked per-connection, NOT in the shared `GpuBackend`.
//!
//! `ConnectionSession` holds all state that is specific to a single client
//! connection and must NOT leak between connections.

use outerlink_common::cuda_types::CuResult;

/// Per-connection state that is NOT shared between connections.
///
/// Created when a client connects, dropped when the connection closes.
/// Passed as `&mut` to [`handle_request`](crate::handler::handle_request) on
/// every request so context-related operations read/write the correct
/// per-connection context.
pub struct ConnectionSession {
    /// The current CUDA context handle for this connection (0 = none).
    current_ctx: u64,
}

impl ConnectionSession {
    /// Create a new session with no current context.
    pub fn new() -> Self {
        Self { current_ctx: 0 }
    }

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
}
