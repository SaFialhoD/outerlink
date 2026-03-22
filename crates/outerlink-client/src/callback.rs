//! Client-side callback registry for cuStreamAddCallback and cuLaunchHostFunc.
//!
//! Stores callback_id -> (fn_ptr, user_data) mappings. Only the callback_id
//! crosses the wire. When the server sends a `CallbackReady` notification,
//! the client looks up the callback and invokes it locally.
//!
//! **Re-entrant safety:** The mutex is released BEFORE invoking the callback
//! function, because the callback may call cuStreamAddCallback again.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};

/// The two kinds of stream callbacks CUDA supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallbackKind {
    /// `cuStreamAddCallback(stream, callback, userData, flags)`
    /// Signature: `void (*CUstreamCallback)(CUstream, CUresult, void*)`
    StreamAddCallback,
    /// `cuLaunchHostFunc(stream, fn, userData)`
    /// Signature: `void (*CUhostFn)(void*)`
    LaunchHostFunc,
}

/// A registered callback entry.
struct CallbackEntry {
    kind: CallbackKind,
    /// Raw function pointer (cast from the C callback).
    fn_ptr: u64,
    /// Raw user_data pointer.
    user_data: u64,
    /// The local stream handle this callback was registered on.
    local_stream: u64,
    /// Whether this callback has been executed.
    completed: bool,
}

/// Thread-safe registry mapping callback_id -> callback metadata.
///
/// The registry is shared between the FFI thread (which registers callbacks)
/// and the callback listener thread (which fires them when the server sends
/// CallbackReady notifications).
pub struct CallbackRegistry {
    /// Next callback ID to assign. Monotonically increasing.
    next_id: AtomicU64,
    /// Protected state: map of pending callbacks + condvar for sync waits.
    inner: Mutex<RegistryInner>,
    /// Signalled when any callback completes (for StreamSynchronize waiting).
    completed_cond: Condvar,
}

struct RegistryInner {
    entries: HashMap<u64, CallbackEntry>,
}

impl CallbackRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            inner: Mutex::new(RegistryInner {
                entries: HashMap::new(),
            }),
            completed_cond: Condvar::new(),
        }
    }

    /// Register a callback and return its unique ID.
    ///
    /// The ID is what gets sent over the wire to the server.
    pub fn register(
        &self,
        kind: CallbackKind,
        fn_ptr: u64,
        user_data: u64,
        local_stream: u64,
    ) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let entry = CallbackEntry {
            kind,
            fn_ptr,
            user_data,
            local_stream,
            completed: false,
        };
        let mut inner = self.inner.lock().unwrap();
        inner.entries.insert(id, entry);
        id
    }

    /// Fire a callback by ID with the given CUDA status code.
    ///
    /// Called by the callback listener thread when a `CallbackReady` message
    /// arrives from the server.
    ///
    /// **CRITICAL:** The mutex is released BEFORE invoking the callback to
    /// allow re-entrant callback registration (e.g., a callback that calls
    /// cuStreamAddCallback).
    pub fn fire(&self, callback_id: u64, cuda_status: u32) {
        // Extract callback data under lock, then release before invoking.
        let (kind, fn_ptr, user_data, local_stream) = {
            let mut inner = self.inner.lock().unwrap();
            let entry = match inner.entries.get_mut(&callback_id) {
                Some(e) => e,
                None => {
                    tracing::warn!(
                        callback_id,
                        "CallbackReady for unknown callback_id, ignoring"
                    );
                    return;
                }
            };
            entry.completed = true;
            let data = (entry.kind, entry.fn_ptr, entry.user_data, entry.local_stream);
            // Remove the entry now that it's been consumed.
            inner.entries.remove(&callback_id);
            data
        };
        // Mutex is released here. Invoke callback BEFORE notifying waiters
        // so StreamSynchronize blocks until the callback has fully returned,
        // matching CUDA semantics ("all preceding work is complete").
        match kind {
            CallbackKind::StreamAddCallback => {
                // CUstreamCallback signature: void(CUstream, CUresult, void*)
                // CUstream = void*, CUresult = unsigned int
                type StreamCb = unsafe extern "C" fn(u64, u32, u64);
                if fn_ptr != 0 {
                    let cb: StreamCb = unsafe { std::mem::transmute(fn_ptr) };
                    unsafe { cb(local_stream, cuda_status, user_data) };
                }
            }
            CallbackKind::LaunchHostFunc => {
                // CUhostFn signature: void(void*)
                type HostFn = unsafe extern "C" fn(u64);
                if fn_ptr != 0 {
                    let cb: HostFn = unsafe { std::mem::transmute(fn_ptr) };
                    unsafe { cb(user_data) };
                }
            }
        }
        // Notify AFTER callback returns so StreamSynchronize blocks until done.
        self.completed_cond.notify_all();
    }

    /// Check if there are any pending (not yet completed) callbacks for a
    /// given stream. Stream 0 matches all callbacks.
    pub fn has_pending(&self, stream: u64) -> bool {
        let inner = self.inner.lock().unwrap();
        if stream == 0 {
            // Default stream: any pending callback counts
            !inner.entries.is_empty()
        } else {
            inner.entries.values().any(|e| e.local_stream == stream)
        }
    }

    /// Block until all pending callbacks for `stream` have been fired.
    ///
    /// Called by StreamSynchronize after the server-side sync completes.
    /// Uses a Condvar to avoid busy-waiting.
    ///
    /// `timeout` bounds the wait to prevent infinite blocking.
    pub fn wait_all_completed(&self, stream: u64, timeout: std::time::Duration) -> bool {
        let mut inner = self.inner.lock().unwrap();
        let deadline = std::time::Instant::now() + timeout;

        loop {
            let has_pending = if stream == 0 {
                !inner.entries.is_empty()
            } else {
                inner.entries.values().any(|e| e.local_stream == stream)
            };

            if !has_pending {
                return true;
            }

            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return false; // Timed out
            }

            let result = self.completed_cond.wait_timeout(inner, remaining).unwrap();
            inner = result.0;
            if result.1.timed_out() {
                // Check one more time after wakeup
                let has_pending = if stream == 0 {
                    !inner.entries.is_empty()
                } else {
                    inner.entries.values().any(|e| e.local_stream == stream)
                };
                return !has_pending;
            }
        }
    }

    /// Number of pending callbacks (for testing/diagnostics).
    pub fn pending_count(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner.entries.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;
    use std::sync::Arc;

    #[test]
    fn test_register_returns_unique_ids() {
        let reg = CallbackRegistry::new();
        let id1 = reg.register(CallbackKind::StreamAddCallback, 0, 0, 100);
        let id2 = reg.register(CallbackKind::LaunchHostFunc, 0, 0, 100);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_has_pending_after_register() {
        let reg = CallbackRegistry::new();
        assert!(!reg.has_pending(100));

        let _id = reg.register(CallbackKind::StreamAddCallback, 0, 0, 100);
        assert!(reg.has_pending(100));
        assert!(!reg.has_pending(200)); // Different stream
        assert!(reg.has_pending(0)); // Default stream matches all
    }

    #[test]
    fn test_fire_removes_entry() {
        let reg = CallbackRegistry::new();
        let id = reg.register(CallbackKind::LaunchHostFunc, 0, 0, 100);
        assert_eq!(reg.pending_count(), 1);

        reg.fire(id, 0);
        assert_eq!(reg.pending_count(), 0);
        assert!(!reg.has_pending(100));
    }

    #[test]
    fn test_fire_unknown_id_is_noop() {
        let reg = CallbackRegistry::new();
        reg.fire(999, 0); // Should not panic
    }

    #[test]
    fn test_fire_invokes_stream_add_callback() {
        // Use a static atomic to verify the callback was invoked.
        static CALLED: AtomicU32 = AtomicU32::new(0);
        static RECEIVED_STATUS: AtomicU32 = AtomicU32::new(u32::MAX);

        unsafe extern "C" fn my_callback(_stream: u64, status: u32, _user_data: u64) {
            CALLED.store(1, Ordering::SeqCst);
            RECEIVED_STATUS.store(status, Ordering::SeqCst);
        }

        let reg = CallbackRegistry::new();
        let fn_ptr = my_callback as *const () as u64;
        let id = reg.register(CallbackKind::StreamAddCallback, fn_ptr, 0, 100);

        reg.fire(id, 0); // CUDA_SUCCESS = 0

        assert_eq!(CALLED.load(Ordering::SeqCst), 1);
        assert_eq!(RECEIVED_STATUS.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_fire_invokes_launch_host_func() {
        static CALLED: AtomicU32 = AtomicU32::new(0);

        unsafe extern "C" fn my_host_fn(_user_data: u64) {
            CALLED.store(1, Ordering::SeqCst);
        }

        let reg = CallbackRegistry::new();
        let fn_ptr = my_host_fn as *const () as u64;
        let id = reg.register(CallbackKind::LaunchHostFunc, fn_ptr, 0, 200);

        reg.fire(id, 0);

        assert_eq!(CALLED.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_wait_all_completed_returns_immediately_when_empty() {
        let reg = CallbackRegistry::new();
        let ok = reg.wait_all_completed(100, std::time::Duration::from_millis(10));
        assert!(ok);
    }

    #[test]
    fn test_wait_all_completed_blocks_until_fired() {
        let reg = Arc::new(CallbackRegistry::new());
        let id = reg.register(CallbackKind::LaunchHostFunc, 0, 0, 100);

        let reg2 = Arc::clone(&reg);
        let t = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(50));
            reg2.fire(id, 0);
        });

        let ok = reg.wait_all_completed(100, std::time::Duration::from_secs(2));
        assert!(ok);
        t.join().unwrap();
    }

    #[test]
    fn test_wait_all_completed_times_out() {
        let reg = CallbackRegistry::new();
        let _id = reg.register(CallbackKind::LaunchHostFunc, 0, 0, 100);

        let ok = reg.wait_all_completed(100, std::time::Duration::from_millis(50));
        assert!(!ok, "should time out since callback was never fired");
    }

    #[test]
    fn test_fire_with_zero_fn_ptr_does_not_crash() {
        let reg = CallbackRegistry::new();
        let id = reg.register(CallbackKind::StreamAddCallback, 0, 42, 100);
        reg.fire(id, 0); // fn_ptr == 0, should skip invocation
        assert_eq!(reg.pending_count(), 0);
    }

    #[test]
    fn test_multiple_callbacks_same_stream() {
        let reg = CallbackRegistry::new();
        let id1 = reg.register(CallbackKind::LaunchHostFunc, 0, 0, 100);
        let id2 = reg.register(CallbackKind::StreamAddCallback, 0, 0, 100);
        assert_eq!(reg.pending_count(), 2);
        assert!(reg.has_pending(100));

        reg.fire(id1, 0);
        assert_eq!(reg.pending_count(), 1);
        assert!(reg.has_pending(100));

        reg.fire(id2, 0);
        assert_eq!(reg.pending_count(), 0);
        assert!(!reg.has_pending(100));
    }

    #[test]
    fn test_callbacks_on_different_streams() {
        let reg = CallbackRegistry::new();
        let id1 = reg.register(CallbackKind::LaunchHostFunc, 0, 0, 100);
        let _id2 = reg.register(CallbackKind::StreamAddCallback, 0, 0, 200);

        // Firing callback on stream 100 should not affect stream 200
        reg.fire(id1, 0);
        assert!(!reg.has_pending(100));
        assert!(reg.has_pending(200));
    }

    #[test]
    fn test_wait_only_waits_for_target_stream() {
        let reg = Arc::new(CallbackRegistry::new());
        let _id_other = reg.register(CallbackKind::LaunchHostFunc, 0, 0, 200);

        // Waiting on stream 100 should return immediately (no callbacks for it)
        let ok = reg.wait_all_completed(100, std::time::Duration::from_millis(50));
        assert!(ok);

        // Stream 200 still has a pending callback
        assert!(reg.has_pending(200));
    }

    #[test]
    fn test_concurrent_register_and_fire() {
        let reg = Arc::new(CallbackRegistry::new());
        let num_callbacks = 100;

        // Register all callbacks
        let mut ids = Vec::new();
        for i in 0..num_callbacks {
            let id = reg.register(CallbackKind::LaunchHostFunc, 0, 0, i as u64);
            ids.push(id);
        }
        assert_eq!(reg.pending_count(), num_callbacks);

        // Fire them from multiple threads
        let mut threads = Vec::new();
        for chunk in ids.chunks(10) {
            let reg2 = Arc::clone(&reg);
            let chunk_ids: Vec<u64> = chunk.to_vec();
            threads.push(std::thread::spawn(move || {
                for id in chunk_ids {
                    reg2.fire(id, 0);
                }
            }));
        }

        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(reg.pending_count(), 0);
    }
}
