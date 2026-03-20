//! End-to-end integration tests for the OuterLink client-server protocol.
//!
//! Each test starts a real TCP server on localhost (port chosen by the OS),
//! connects a client via `TcpTransportConnection`, and exercises the full
//! request-response cycle through the wire protocol and handler.
//!
//! We use raw `TcpTransportConnection` on the client side (Option A) rather
//! than `OuterLinkClient` because `OuterLinkClient::send_request` calls
//! `runtime.block_on()`, which panics when called from within a tokio context.

use std::sync::Arc;

use tokio::net::TcpListener;

use outerlink_common::cuda_types::CuResult;
use outerlink_common::error::OuterLinkError;
use outerlink_common::protocol::{MessageHeader, MessageType};
use outerlink_common::tcp_transport::TcpTransportConnection;
use outerlink_common::transport::TransportConnection;
use outerlink_server::gpu_backend::{GpuBackend, StubGpuBackend};
use outerlink_server::handler::handle_request;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the `CuResult` from the first 4 bytes of a response payload.
fn response_result(payload: &[u8]) -> CuResult {
    assert!(
        payload.len() >= 4,
        "response payload too short ({} bytes)",
        payload.len()
    );
    CuResult::from_raw(u32::from_le_bytes(payload[..4].try_into().unwrap()))
}

/// Assert the response indicates success and return the data portion (after
/// the 4-byte `CuResult` prefix).
fn assert_success(payload: &[u8]) -> &[u8] {
    let result = response_result(payload);
    assert_eq!(
        result,
        CuResult::Success,
        "expected CuResult::Success, got {:?}",
        result
    );
    &payload[4..]
}

/// Spawn a server task that accepts ONE connection on `listener`, runs the
/// recv-handle-send loop until the client disconnects, then exits.
///
/// Returns a `JoinHandle` so the test can await server completion and catch
/// any panics.
fn spawn_server(
    listener: TcpListener,
    backend: Arc<dyn GpuBackend>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let (stream, _peer) = listener.accept().await.expect("accept failed");
        let conn = TcpTransportConnection::new(stream).expect("server conn init failed");

        loop {
            let (header, payload) = match conn.recv_message().await {
                Ok(msg) => msg,
                Err(OuterLinkError::ConnectionClosed) => break,
                Err(e) => panic!("server recv error: {e:?}"),
            };

            let (resp_header, resp_payload) = handle_request(&*backend, &header, &payload);
            conn.send_message(&resp_header, &resp_payload)
                .await
                .expect("server send failed");
        }
    })
}

/// Bind a TCP listener on localhost with an OS-assigned port and return
/// both the listener and the address string suitable for client connection.
async fn bind_server() -> (TcpListener, String) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind");
    let addr = listener.local_addr().unwrap().to_string();
    (listener, addr)
}

/// Connect a raw `TcpTransportConnection` to the given address.
async fn connect_client(addr: &str) -> TcpTransportConnection {
    let stream = tokio::net::TcpStream::connect(addr)
        .await
        .expect("client connect failed");
    TcpTransportConnection::new(stream).expect("client conn init failed")
}

/// Send a request and receive the response, using a monotonic request ID.
async fn roundtrip(
    conn: &TcpTransportConnection,
    msg_type: MessageType,
    payload: &[u8],
) -> (MessageHeader, Vec<u8>) {
    // Use a fixed request_id of 1 -- each test runs in isolation so there
    // is no conflict.
    let header = MessageHeader::new_request(1, msg_type, payload.len() as u32);
    conn.send_message(&header, payload)
        .await
        .expect("client send failed");
    conn.recv_message().await.expect("client recv failed")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_device_get_count_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;
    let (_hdr, payload) = roundtrip(&client, MessageType::DeviceGetCount, &[]).await;

    let data = assert_success(&payload);
    let count = i32::from_le_bytes(data[..4].try_into().unwrap());
    assert_eq!(count, 1, "stub backend exposes exactly 1 device");

    drop(client); // close connection so server task exits
    server.await.unwrap();
}

#[tokio::test]
async fn test_driver_get_version_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;
    let (_hdr, payload) = roundtrip(&client, MessageType::DriverGetVersion, &[]).await;

    let data = assert_success(&payload);
    let version = i32::from_le_bytes(data[..4].try_into().unwrap());
    assert_eq!(version, 12040, "stub backend returns CUDA 12.4 (12040)");

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_device_get_name_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;
    // Request payload: i32 device = 0
    let device_payload = 0_i32.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::DeviceGetName, &device_payload).await;

    let data = assert_success(&payload);
    // Response: u32 name_len + UTF-8 bytes
    let name_len = u32::from_le_bytes(data[..4].try_into().unwrap()) as usize;
    assert!(name_len > 0, "device name should not be empty");
    let name = std::str::from_utf8(&data[4..4 + name_len]).expect("name should be valid UTF-8");
    assert!(
        !name.is_empty(),
        "device name string should not be empty"
    );
    // The stub returns "OuterLink Virtual GPU"
    assert_eq!(name, "OuterLink Virtual GPU");

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_device_total_mem_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;
    let device_payload = 0_i32.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::DeviceTotalMem, &device_payload).await;

    let data = assert_success(&payload);
    let total_bytes = u64::from_le_bytes(data[..8].try_into().unwrap());
    let expected_24gb: u64 = 24 * 1024 * 1024 * 1024;
    assert_eq!(
        total_bytes, expected_24gb,
        "stub backend reports 24 GiB VRAM"
    );

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_mem_alloc_free_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // MemAlloc: request payload is u64 size = 1024
    let alloc_payload = 1024_u64.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::MemAlloc, &alloc_payload).await;

    let data = assert_success(&payload);
    let device_ptr = u64::from_le_bytes(data[..8].try_into().unwrap());
    assert_ne!(device_ptr, 0, "allocated pointer should be non-zero");

    // MemFree: request payload is u64 device_ptr
    let free_payload = device_ptr.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::MemFree, &free_payload).await;

    let result = response_result(&payload);
    assert_eq!(result, CuResult::Success, "MemFree should succeed");

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_memcpy_roundtrip_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // 1. Allocate 256 bytes
    let alloc_payload = 256_u64.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::MemAlloc, &alloc_payload).await;
    let data = assert_success(&payload);
    let device_ptr = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 2. MemcpyHtoD: write a recognisable pattern
    let test_data: Vec<u8> = (0..256).map(|i| (i & 0xFF) as u8).collect();
    let mut htod_payload = device_ptr.to_le_bytes().to_vec();
    htod_payload.extend_from_slice(&test_data);
    let (_hdr, payload) = roundtrip(&client, MessageType::MemcpyHtoD, &htod_payload).await;
    let result = response_result(&payload);
    assert_eq!(result, CuResult::Success, "MemcpyHtoD should succeed");

    // 3. MemcpyDtoH: read back the data
    let mut dtoh_payload = device_ptr.to_le_bytes().to_vec();
    dtoh_payload.extend_from_slice(&256_u64.to_le_bytes());
    let (_hdr, payload) = roundtrip(&client, MessageType::MemcpyDtoH, &dtoh_payload).await;
    let data = assert_success(&payload);
    assert_eq!(
        data, &test_data[..],
        "data read back from device should match what was written"
    );

    // 4. Free
    let free_payload = device_ptr.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::MemFree, &free_payload).await;
    assert_eq!(response_result(&payload), CuResult::Success);

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_ctx_create_destroy_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // CtxCreate: flags=0, device=0
    let mut ctx_create_payload = 0_u32.to_le_bytes().to_vec(); // flags
    ctx_create_payload.extend_from_slice(&0_i32.to_le_bytes()); // device
    let (_hdr, payload) =
        roundtrip(&client, MessageType::CtxCreate, &ctx_create_payload).await;

    let data = assert_success(&payload);
    let ctx_handle = u64::from_le_bytes(data[..8].try_into().unwrap());
    assert_ne!(ctx_handle, 0, "context handle should be non-zero");

    // CtxDestroy
    let ctx_destroy_payload = ctx_handle.to_le_bytes();
    let (_hdr, payload) =
        roundtrip(&client, MessageType::CtxDestroy, &ctx_destroy_payload).await;
    let result = response_result(&payload);
    assert_eq!(result, CuResult::Success, "CtxDestroy should succeed");

    drop(client);
    server.await.unwrap();
}
