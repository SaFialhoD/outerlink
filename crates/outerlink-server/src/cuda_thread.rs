//! Dedicated OS thread for CUDA backend operations.
//!
//! CUDA contexts are thread-local: when `cuCtxCreate` is called, it pushes
//! the new context onto the *calling thread's* context stack. Subsequent
//! CUDA calls (memcpy, kernel launch, etc.) operate on whatever context is
//! current on *that* thread.
//!
//! In the OuterLink server, requests arrive on tokio worker threads which
//! are multiplexed by the runtime. A request that creates a context on
//! thread A might be followed by a memcpy dispatched to thread B, where
//! no context is current -> `CUDA_ERROR_INVALID_CONTEXT`.
//!
//! [`CudaWorker`] solves this by spawning a single dedicated `std::thread`
//! per connection. All GPU backend calls for that connection are serialised
//! through an mpsc channel and executed on the dedicated thread, which calls
//! `cuCtxSetCurrent` before each operation to ensure the right context is
//! active.

use std::sync::Arc;
use std::thread;

use tokio::sync::{mpsc, oneshot};

use outerlink_common::protocol::MessageHeader;

use crate::gpu_backend::GpuBackend;
use crate::handler::{handle_request_full, HandleResult};
use crate::session::ConnectionSession;

/// A work item sent from the tokio task to the dedicated CUDA thread.
struct WorkItem {
    header: MessageHeader,
    payload: Vec<u8>,
    /// The CUDA context handle to set as current before executing.
    /// Comes from the session's `current_ctx()` at send time.
    current_ctx: u64,
    /// Channel to send the result back to the tokio task.
    reply: oneshot::Sender<HandleResult>,
}

/// A dedicated OS thread that processes all CUDA backend calls for a single
/// connection.
///
/// The tokio task reads messages from TCP, wraps them into `WorkItem`s, and
/// sends them through `tx`. The worker thread receives them, calls
/// `ctx_set_current` to pin the right context, runs the handler, and sends
/// the result back through the oneshot.
pub struct CudaWorker {
    tx: mpsc::UnboundedSender<WorkItem>,
    /// Handle to the background OS thread (used for clean shutdown).
    _thread: Option<thread::JoinHandle<()>>,
}

impl CudaWorker {
    /// Spawn a new dedicated worker thread for CUDA operations.
    ///
    /// The worker owns a [`ConnectionSession`] and processes requests
    /// sequentially on a single OS thread, ensuring CUDA context
    /// thread-locality.
    /// Spawn a new worker with session_id 0 (for tests or when ID is not known).
    pub fn new(backend: Arc<dyn GpuBackend>) -> Self {
        Self::with_session_id(backend, 0)
    }

    /// Spawn a new dedicated worker thread with a specific session ID.
    ///
    /// The session ID is assigned to the internal [`ConnectionSession`] so
    /// that handshake responses include the correct ID for callback channel
    /// association.
    pub fn with_session_id(backend: Arc<dyn GpuBackend>, session_id: u64) -> Self {
        let (tx, rx) = mpsc::unbounded_channel::<WorkItem>();

        let handle = thread::Builder::new()
            .name(format!("cuda-worker-{}", session_id))
            .spawn(move || {
                Self::worker_loop(backend, rx, session_id);
            })
            .expect("failed to spawn CUDA worker thread");

        Self {
            tx,
            _thread: Some(handle),
        }
    }

    /// Send a request to the worker thread and await the response.
    ///
    /// `current_ctx` is the session's current CUDA context handle;
    /// the worker will call `ctx_set_current` with this value before
    /// dispatching the request.
    ///
    /// Returns the response header + payload, plus any callback notification.
    pub async fn send_request(
        &self,
        header: MessageHeader,
        payload: Vec<u8>,
        current_ctx: u64,
    ) -> Result<(MessageHeader, Vec<u8>), CudaWorkerError> {
        let (reply_tx, reply_rx) = oneshot::channel();

        self.tx
            .send(WorkItem {
                header,
                payload,
                current_ctx,
                reply: reply_tx,
            })
            .map_err(|_| CudaWorkerError::WorkerGone)?;

        let result = reply_rx.await.map_err(|_| CudaWorkerError::WorkerGone)?;
        Ok(result.response)
    }

    /// Send a request and get the full HandleResult (including callback info).
    pub async fn send_request_full(
        &self,
        header: MessageHeader,
        payload: Vec<u8>,
        current_ctx: u64,
    ) -> Result<HandleResult, CudaWorkerError> {
        let (reply_tx, reply_rx) = oneshot::channel();

        self.tx
            .send(WorkItem {
                header,
                payload,
                current_ctx,
                reply: reply_tx,
            })
            .map_err(|_| CudaWorkerError::WorkerGone)?;

        reply_rx.await.map_err(|_| CudaWorkerError::WorkerGone)
    }

    /// The main loop running on the dedicated OS thread.
    ///
    /// Blocks on the mpsc receiver. For each work item:
    /// 1. Call `ctx_set_current` to pin the session's CUDA context
    /// 2. Run `handle_request_full` with the backend and session
    /// 3. Send the result back via oneshot
    ///
    /// Exits when the sender side is dropped (all CudaWorker clones gone).
    fn worker_loop(
        backend: Arc<dyn GpuBackend>,
        mut rx: mpsc::UnboundedReceiver<WorkItem>,
        session_id: u64,
    ) {
        let mut session = ConnectionSession::with_session_id(session_id);

        // Use blocking_recv since this is a plain OS thread, not a tokio task.
        while let Some(item) = rx.blocking_recv() {
            // Pin the correct CUDA context on this thread before every call.
            if item.current_ctx != 0 {
                if let Err(e) = backend.ctx_set_current(item.current_ctx) {
                    tracing::warn!(
                        ctx = item.current_ctx,
                        error = ?e,
                        "ctx_set_current failed on worker thread"
                    );
                    // Continue anyway - the handler will likely fail with
                    // InvalidContext which is the correct error to propagate.
                }
            }

            let result = handle_request_full(&*backend, &item.header, &item.payload, &mut session);

            // Update our record of the current ctx from the session.
            // (ctx_create / ctx_set_current in the handler may have changed it.)
            // The caller gets the response; on the next call they should pass
            // session.current_ctx() - but since we own the session internally,
            // we handle this automatically.

            // Send result back. If the receiver was dropped, just discard.
            let _ = item.reply.send(result);
        }

        // Channel closed - clean up session resources.
        session.cleanup(&*backend);
        tracing::debug!("CUDA worker thread exiting after session cleanup");
    }

    /// Get access to the session's current context.
    /// Since the session lives inside the worker thread, we track it
    /// by updating after each request. For callers that need to read
    /// session state, we provide this through the response.
    ///
    /// Note: The caller doesn't need to track current_ctx externally
    /// because the CudaWorker owns the session. The `current_ctx`
    /// parameter in send_request is only used for the initial
    /// ctx_set_current call on the worker thread. For the worker-owned
    /// session, the handler updates current_ctx internally.
    pub fn is_alive(&self) -> bool {
        !self.tx.is_closed()
    }
}

/// Errors from communicating with the CudaWorker thread.
#[derive(Debug)]
pub enum CudaWorkerError {
    /// The worker thread has exited (channel closed).
    WorkerGone,
}

impl std::fmt::Display for CudaWorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaWorkerError::WorkerGone => write!(f, "CUDA worker thread has exited"),
        }
    }
}

impl std::error::Error for CudaWorkerError {}
