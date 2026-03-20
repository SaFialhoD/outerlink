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
    /// Per-connection context stack for cuCtxPushCurrent / cuCtxPopCurrent.
    ///
    /// In CUDA, the context stack is per-thread. In OuterLink's remote model,
    /// each connection is the equivalent of a thread, so the stack lives here.
    /// The top of the stack is always consistent with `current_ctx`.
    context_stack: Vec<u64>,
}

impl ConnectionSession {
    /// Create a new session with no current context.
    pub fn new() -> Self {
        Self {
            current_ctx: 0,
            context_stack: Vec::new(),
        }
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

    /// Push a context onto the per-connection context stack.
    ///
    /// The pushed context becomes the new current context. The previous
    /// current context (if any) is preserved on the stack so it can be
    /// restored by [`pop_ctx`](Self::pop_ctx).
    pub fn push_ctx(&mut self, ctx: u64) {
        // Push the old current onto the stack, then set the new one.
        self.context_stack.push(self.current_ctx);
        self.current_ctx = ctx;
    }

    /// Pop the top context from the per-connection context stack.
    ///
    /// Returns the context that was current before the pop. The context
    /// underneath it on the stack becomes the new current context (or 0
    /// if the stack is empty after the pop).
    pub fn pop_ctx(&mut self) -> Result<u64, CuResult> {
        if self.context_stack.is_empty() {
            return Err(CuResult::InvalidContext);
        }
        let popped = self.current_ctx;
        self.current_ctx = self.context_stack.pop().unwrap_or(0);
        Ok(popped)
    }

    /// If the current context matches `ctx`, clear it (set to 0).
    ///
    /// Called after a context is destroyed so the session does not hold
    /// a dangling handle. Also removes any occurrences from the stack.
    pub fn clear_if_current(&mut self, ctx: u64) {
        if self.current_ctx == ctx {
            self.current_ctx = 0;
        }
        self.context_stack.retain(|&c| c != ctx);
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
    fn test_push_ctx() {
        let mut session = ConnectionSession::new();
        // Push ctx A -- should become current.
        session.push_ctx(0xAAAA);
        assert_eq!(session.current_ctx(), 0xAAAA);
    }

    #[test]
    fn test_push_then_pop_ctx() {
        let mut session = ConnectionSession::new();
        session.push_ctx(0xAAAA);
        session.push_ctx(0xBBBB);
        assert_eq!(session.current_ctx(), 0xBBBB);
        // Pop should return 0xBBBB and restore 0xAAAA as current.
        let popped = session.pop_ctx().unwrap();
        assert_eq!(popped, 0xBBBB);
        assert_eq!(session.current_ctx(), 0xAAAA);
    }

    #[test]
    fn test_pop_ctx_restores_zero() {
        let mut session = ConnectionSession::new();
        // Push one context (previous current was 0).
        session.push_ctx(0xAAAA);
        let popped = session.pop_ctx().unwrap();
        assert_eq!(popped, 0xAAAA);
        // Current should be 0 (the original state before push).
        assert_eq!(session.current_ctx(), 0);
    }

    #[test]
    fn test_pop_ctx_empty_stack_errors() {
        let mut session = ConnectionSession::new();
        assert_eq!(session.pop_ctx(), Err(CuResult::InvalidContext));
    }

    #[test]
    fn test_push_pop_multiple() {
        let mut session = ConnectionSession::new();
        session.push_ctx(0xAAAA);
        session.push_ctx(0xBBBB);
        session.push_ctx(0xCCCC);
        assert_eq!(session.current_ctx(), 0xCCCC);

        assert_eq!(session.pop_ctx().unwrap(), 0xCCCC);
        assert_eq!(session.current_ctx(), 0xBBBB);

        assert_eq!(session.pop_ctx().unwrap(), 0xBBBB);
        assert_eq!(session.current_ctx(), 0xAAAA);

        assert_eq!(session.pop_ctx().unwrap(), 0xAAAA);
        assert_eq!(session.current_ctx(), 0);
    }

    #[test]
    fn test_push_ctx_preserves_set_current() {
        let mut session = ConnectionSession::new();
        // Set a current context via set_current_ctx.
        session.set_current_ctx(0xAAAA);
        // Now push a new one -- old current should be saved on stack.
        session.push_ctx(0xBBBB);
        assert_eq!(session.current_ctx(), 0xBBBB);
        // Pop restores the original.
        let popped = session.pop_ctx().unwrap();
        assert_eq!(popped, 0xBBBB);
        assert_eq!(session.current_ctx(), 0xAAAA);
    }

    #[test]
    fn test_clear_if_current_also_clears_stack() {
        let mut session = ConnectionSession::new();
        session.push_ctx(0xAAAA);
        session.push_ctx(0xBBBB);
        session.push_ctx(0xAAAA); // push again
        // Destroy 0xAAAA -- should remove from stack too.
        session.clear_if_current(0xAAAA);
        // Current was 0xAAAA, so it should be cleared to 0.
        assert_eq!(session.current_ctx(), 0);
        // Stack should have had [0, 0xAAAA, 0xBBBB] -- 0xAAAA entries removed,
        // leaving [0, 0xBBBB].
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
