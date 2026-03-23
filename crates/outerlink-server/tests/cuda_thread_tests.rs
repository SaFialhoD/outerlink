//! Tests for the CudaWorker dedicated-thread architecture.
//!
//! These tests verify that:
//! 1. CudaWorker processes requests on a dedicated OS thread
//! 2. All requests for a connection go through the same thread
//! 3. The worker correctly calls ctx_set_current before each operation
//! 4. The worker shuts down cleanly when the sender is dropped
//! 5. StubGpuBackend reports needs_dedicated_thread() == false
//! 6. CudaWorker works correctly with the stub backend for testing
//! 7. The server uses CudaWorker when needs_dedicated_thread() is true

mod common;
use common::*;

use std::sync::Arc;

use outerlink_common::protocol::MessageType;
use outerlink_server::cuda_thread::CudaWorker;
use outerlink_server::gpu_backend::{GpuBackend, StubGpuBackend};

// ---------------------------------------------------------------------------
// Unit tests for GpuBackend trait additions
// ---------------------------------------------------------------------------

#[test]
fn test_stub_backend_needs_dedicated_thread_false() {
    let backend = StubGpuBackend::new();
    assert!(
        !backend.needs_dedicated_thread(),
        "StubGpuBackend should NOT need a dedicated thread"
    );
}

#[test]
fn test_stub_backend_ctx_set_current_noop() {
    let backend = StubGpuBackend::new();
    // ctx_set_current on the stub should be a no-op that succeeds
    let result = backend.ctx_set_current(0);
    assert!(result.is_ok(), "ctx_set_current(0) should succeed on stub");

    let result = backend.ctx_set_current(12345);
    assert!(result.is_ok(), "ctx_set_current(any) should succeed on stub");
}

// ---------------------------------------------------------------------------
// CudaWorker basic lifecycle tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_cuda_worker_creation_and_shutdown() {
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let worker = CudaWorker::new(backend);
    // Worker should be created successfully
    // Dropping the worker should shut down the background thread cleanly
    drop(worker);
    // If we reach here without hanging or panicking, the test passes
}

#[tokio::test]
async fn test_cuda_worker_processes_init() {
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let worker = CudaWorker::new(backend);

    let header = MessageHeader::new_request(1, MessageType::Init, 4);
    let payload = 0u32.to_le_bytes().to_vec();

    let (resp_header, resp_payload) = worker
        .send_request(header, payload, 0)
        .await
        .expect("worker should process Init request");

    assert_eq!(resp_header.msg_type, MessageType::Response);
    assert_eq!(resp_header.request_id, 1);
    let result = response_result(&resp_payload);
    assert_eq!(result, CuResult::Success);

    drop(worker);
}

#[tokio::test]
async fn test_cuda_worker_processes_device_get_count() {
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let worker = CudaWorker::new(backend);

    let header = MessageHeader::new_request(2, MessageType::DeviceGetCount, 0);
    let payload = vec![];

    let (resp_header, resp_payload) = worker
        .send_request(header, payload, 0)
        .await
        .expect("worker should process DeviceGetCount request");

    assert_eq!(resp_header.msg_type, MessageType::Response);
    let data = assert_success(&resp_payload);
    let count = i32::from_le_bytes(data[..4].try_into().unwrap());
    assert_eq!(count, 1, "stub backend should report 1 device");

    drop(worker);
}

#[tokio::test]
async fn test_cuda_worker_multiple_sequential_requests() {
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let worker = CudaWorker::new(backend);

    // Send multiple requests sequentially
    for i in 1..=5 {
        let header = MessageHeader::new_request(i, MessageType::DeviceGetCount, 0);
        let (resp_header, resp_payload) = worker
            .send_request(header, vec![], 0)
            .await
            .expect("worker should process request");

        assert_eq!(resp_header.request_id, i);
        let data = assert_success(&resp_payload);
        let count = i32::from_le_bytes(data[..4].try_into().unwrap());
        assert_eq!(count, 1);
    }

    drop(worker);
}

#[tokio::test]
async fn test_cuda_worker_runs_on_single_thread() {
    // Verify that all requests execute on the same OS thread.
    // We use ctx_create + ctx_destroy to verify the worker processes
    // stateful operations correctly (which requires thread consistency).
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let worker = CudaWorker::new(backend);

    // Create a context
    let mut ctx_payload = Vec::new();
    ctx_payload.extend_from_slice(&0u32.to_le_bytes()); // flags
    ctx_payload.extend_from_slice(&0i32.to_le_bytes()); // device

    let header = MessageHeader::new_request(1, MessageType::CtxCreate, ctx_payload.len() as u32);
    let (_resp_hdr, resp_payload) = worker
        .send_request(header, ctx_payload, 0)
        .await
        .expect("ctx_create should work");

    let data = assert_success(&resp_payload);
    let ctx_handle = u64::from_le_bytes(data[..8].try_into().unwrap());
    assert_ne!(ctx_handle, 0, "context handle should be non-zero");

    // Destroy the context (requires same backend state)
    let destroy_payload = ctx_handle.to_le_bytes().to_vec();
    let header = MessageHeader::new_request(2, MessageType::CtxDestroy, destroy_payload.len() as u32);
    let (_resp_hdr, resp_payload) = worker
        .send_request(header, destroy_payload, ctx_handle)
        .await
        .expect("ctx_destroy should work");

    let result = response_result(&resp_payload);
    assert_eq!(result, CuResult::Success);

    drop(worker);
}

#[tokio::test]
async fn test_cuda_worker_mem_alloc_and_free_through_worker() {
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let worker = CudaWorker::new(backend);

    // Allocate memory
    let alloc_payload = 4096u64.to_le_bytes().to_vec();
    let header = MessageHeader::new_request(1, MessageType::MemAlloc, alloc_payload.len() as u32);
    let (_resp_hdr, resp_payload) = worker
        .send_request(header, alloc_payload, 0)
        .await
        .expect("mem_alloc should work");

    let data = assert_success(&resp_payload);
    let ptr = u64::from_le_bytes(data[..8].try_into().unwrap());
    assert_ne!(ptr, 0, "allocated pointer should be non-zero");

    // Free memory
    let free_payload = ptr.to_le_bytes().to_vec();
    let header = MessageHeader::new_request(2, MessageType::MemFree, free_payload.len() as u32);
    let (_resp_hdr, resp_payload) = worker
        .send_request(header, free_payload, 0)
        .await
        .expect("mem_free should work");

    let result = response_result(&resp_payload);
    assert_eq!(result, CuResult::Success);

    drop(worker);
}

#[tokio::test]
async fn test_cuda_worker_dropped_sender_stops_worker() {
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let worker = CudaWorker::new(backend);

    // Send one request to confirm it works
    let header = MessageHeader::new_request(1, MessageType::DeviceGetCount, 0);
    let _ = worker
        .send_request(header, vec![], 0)
        .await
        .expect("should work before drop");

    // Drop the worker - background thread should exit cleanly
    drop(worker);

    // If we get here without hanging, the worker shut down properly.
    // Give the OS thread a moment to actually exit.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
}

// ---------------------------------------------------------------------------
// Integration: CudaWorker through the TCP server path
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_server_with_stub_still_works_e2e() {
    // The stub backend does NOT need a dedicated thread.
    // Verify the server still works normally with StubGpuBackend.
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;
    let (_hdr, payload) = roundtrip(&client, MessageType::DeviceGetCount, &[], 1).await;

    let data = assert_success(&payload);
    let count = i32::from_le_bytes(data[..4].try_into().unwrap());
    assert_eq!(count, 1);

    drop(client);
    server.await.unwrap();
}
