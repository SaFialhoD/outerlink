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
use outerlink_server::session::ConnectionSession;

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

        let mut session = ConnectionSession::new();
        loop {
            let (header, payload) = match conn.recv_message().await {
                Ok(msg) => msg,
                Err(OuterLinkError::ConnectionClosed) => break,
                Err(e) => panic!("server recv error: {e:?}"),
            };

            let (resp_header, resp_payload) = handle_request(&*backend, &header, &payload, &mut session);
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

/// Send a request and receive the response. The caller provides the
/// `request_id` so that multi-request tests can use incrementing IDs.
///
/// Asserts that the response header has `msg_type == Response` and that the
/// `request_id` in the response matches what was sent.
async fn roundtrip(
    conn: &TcpTransportConnection,
    msg_type: MessageType,
    payload: &[u8],
    request_id: u64,
) -> (MessageHeader, Vec<u8>) {
    let header = MessageHeader::new_request(request_id, msg_type, payload.len() as u32);
    conn.send_message(&header, payload)
        .await
        .expect("client send failed");
    let (resp_hdr, resp_payload) = conn.recv_message().await.expect("client recv failed");
    assert_eq!(
        resp_hdr.msg_type,
        MessageType::Response,
        "expected Response msg_type, got {:?}",
        resp_hdr.msg_type
    );
    assert_eq!(
        resp_hdr.request_id, request_id,
        "response request_id should match the sent request_id"
    );
    (resp_hdr, resp_payload)
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
    let (_hdr, payload) = roundtrip(&client, MessageType::DeviceGetCount, &[], 1).await;

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
    let (_hdr, payload) = roundtrip(&client, MessageType::DriverGetVersion, &[], 1).await;

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
    let (_hdr, payload) = roundtrip(&client, MessageType::DeviceGetName, &device_payload, 1).await;

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
    let (_hdr, payload) = roundtrip(&client, MessageType::DeviceTotalMem, &device_payload, 1).await;

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
    let (_hdr, payload) = roundtrip(&client, MessageType::MemAlloc, &alloc_payload, 1).await;

    let data = assert_success(&payload);
    let device_ptr = u64::from_le_bytes(data[..8].try_into().unwrap());
    assert_ne!(device_ptr, 0, "allocated pointer should be non-zero");

    // MemFree: request payload is u64 device_ptr
    let free_payload = device_ptr.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::MemFree, &free_payload, 2).await;

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
    let (_hdr, payload) = roundtrip(&client, MessageType::MemAlloc, &alloc_payload, 1).await;
    let data = assert_success(&payload);
    let device_ptr = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 2. MemcpyHtoD: write a recognisable pattern
    let test_data: Vec<u8> = (0..256).map(|i| (i & 0xFF) as u8).collect();
    let mut htod_payload = device_ptr.to_le_bytes().to_vec();
    htod_payload.extend_from_slice(&test_data);
    let (_hdr, payload) = roundtrip(&client, MessageType::MemcpyHtoD, &htod_payload, 2).await;
    let result = response_result(&payload);
    assert_eq!(result, CuResult::Success, "MemcpyHtoD should succeed");

    // 3. MemcpyDtoH: read back the data
    let mut dtoh_payload = device_ptr.to_le_bytes().to_vec();
    dtoh_payload.extend_from_slice(&256_u64.to_le_bytes());
    let (_hdr, payload) = roundtrip(&client, MessageType::MemcpyDtoH, &dtoh_payload, 3).await;
    let data = assert_success(&payload);
    assert_eq!(
        data, &test_data[..],
        "data read back from device should match what was written"
    );

    // 4. Free
    let free_payload = device_ptr.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::MemFree, &free_payload, 4).await;
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
        roundtrip(&client, MessageType::CtxCreate, &ctx_create_payload, 1).await;

    let data = assert_success(&payload);
    let ctx_handle = u64::from_le_bytes(data[..8].try_into().unwrap());
    assert_ne!(ctx_handle, 0, "context handle should be non-zero");

    // CtxDestroy
    let ctx_destroy_payload = ctx_handle.to_le_bytes();
    let (_hdr, payload) =
        roundtrip(&client, MessageType::CtxDestroy, &ctx_destroy_payload, 2).await;
    let result = response_result(&payload);
    assert_eq!(result, CuResult::Success, "CtxDestroy should succeed");

    drop(client);
    server.await.unwrap();
}

// ---------------------------------------------------------------------------
// Error path tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_double_free_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // Allocate
    let alloc_payload = 512_u64.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::MemAlloc, &alloc_payload, 1).await;
    let data = assert_success(&payload);
    let device_ptr = u64::from_le_bytes(data[..8].try_into().unwrap());

    // First free -- should succeed
    let free_payload = device_ptr.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::MemFree, &free_payload, 2).await;
    assert_eq!(response_result(&payload), CuResult::Success, "first MemFree should succeed");

    // Second free of the same pointer -- should return an error
    let (_hdr, payload) = roundtrip(&client, MessageType::MemFree, &free_payload, 3).await;
    let result = response_result(&payload);
    assert_ne!(
        result,
        CuResult::Success,
        "double MemFree should fail, got {:?}",
        result
    );
    assert_eq!(
        result,
        CuResult::InvalidValue,
        "double free should return InvalidValue"
    );

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_ctx_create_invalid_device_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // CtxCreate with device=99 (invalid -- stub only has device 0)
    let mut ctx_create_payload = 0_u32.to_le_bytes().to_vec(); // flags
    ctx_create_payload.extend_from_slice(&99_i32.to_le_bytes()); // invalid device
    let (_hdr, payload) =
        roundtrip(&client, MessageType::CtxCreate, &ctx_create_payload, 1).await;

    let result = response_result(&payload);
    assert_ne!(
        result,
        CuResult::Success,
        "CtxCreate with invalid device should fail, got {:?}",
        result
    );
    assert_eq!(
        result,
        CuResult::InvalidDevice,
        "CtxCreate with invalid device should return InvalidDevice"
    );

    drop(client);
    server.await.unwrap();
}

// ---------------------------------------------------------------------------
// Handshake and Init tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_handshake_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;
    let (_hdr, payload) = roundtrip(&client, MessageType::Handshake, &[], 1).await;

    assert_success(&payload);

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_init_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;
    // Init payload: 4-byte flags = 0
    let init_payload = 0_u32.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::Init, &init_payload, 1).await;

    assert_success(&payload);

    drop(client);
    server.await.unwrap();
}

// ---------------------------------------------------------------------------
// Module / Stream / Event / Kernel E2E tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_module_load_data_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // Send ModuleLoadData with dummy PTX bytes.
    let ptx = b"fake ptx data for e2e test";
    let (_hdr, payload) = roundtrip(&client, MessageType::ModuleLoadData, ptx, 1).await;

    let data = assert_success(&payload);
    let module_handle = u64::from_le_bytes(data[..8].try_into().unwrap());
    assert_ne!(module_handle, 0, "module handle should be non-zero");

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_module_get_function_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // 1. Load a module.
    let ptx = b"fake ptx";
    let (_hdr, payload) = roundtrip(&client, MessageType::ModuleLoadData, ptx, 1).await;
    let data = assert_success(&payload);
    let module_handle = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 2. Get function: [8B module_handle][4B name_len][name bytes]
    let name = b"my_kernel";
    let mut get_func_payload = module_handle.to_le_bytes().to_vec();
    get_func_payload.extend_from_slice(&(name.len() as u32).to_le_bytes());
    get_func_payload.extend_from_slice(name);
    let (_hdr, payload) =
        roundtrip(&client, MessageType::ModuleGetFunction, &get_func_payload, 2).await;

    let data = assert_success(&payload);
    let func_handle = u64::from_le_bytes(data[..8].try_into().unwrap());
    assert_ne!(func_handle, 0, "function handle should be non-zero");

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_module_unload_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // 1. Load a module.
    let ptx = b"fake ptx";
    let (_hdr, payload) = roundtrip(&client, MessageType::ModuleLoadData, ptx, 1).await;
    let data = assert_success(&payload);
    let module_handle = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 2. Unload the module.
    let unload_payload = module_handle.to_le_bytes();
    let (_hdr, payload) =
        roundtrip(&client, MessageType::ModuleUnload, &unload_payload, 2).await;
    assert_eq!(
        response_result(&payload),
        CuResult::Success,
        "ModuleUnload should succeed"
    );

    // 3. Try to get a function from the unloaded module -- should fail.
    let name = b"my_kernel";
    let mut get_func_payload = module_handle.to_le_bytes().to_vec();
    get_func_payload.extend_from_slice(&(name.len() as u32).to_le_bytes());
    get_func_payload.extend_from_slice(name);
    let (_hdr, payload) =
        roundtrip(&client, MessageType::ModuleGetFunction, &get_func_payload, 3).await;
    let result = response_result(&payload);
    assert_eq!(
        result,
        CuResult::InvalidValue,
        "ModuleGetFunction on unloaded module should return InvalidValue"
    );

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_module_get_global_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // 1. Load a module.
    let ptx = b"fake ptx";
    let (_hdr, payload) = roundtrip(&client, MessageType::ModuleLoadData, ptx, 1).await;
    let data = assert_success(&payload);
    let module_handle = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 2. Get global: [8B module][4B name_len][name bytes]
    let name = b"my_global";
    let mut get_global_payload = module_handle.to_le_bytes().to_vec();
    get_global_payload.extend_from_slice(&(name.len() as u32).to_le_bytes());
    get_global_payload.extend_from_slice(name);
    let (_hdr, payload) =
        roundtrip(&client, MessageType::ModuleGetGlobal, &get_global_payload, 2).await;

    let data = assert_success(&payload);
    let devptr = u64::from_le_bytes(data[..8].try_into().unwrap());
    let size = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
    assert_ne!(devptr, 0, "global devptr should be non-zero");
    assert!(size > 0, "global size should be > 0");

    // 3. Verify the allocation is usable: MemcpyHtoD then MemcpyDtoH.
    let test_data = vec![0xABu8; size];
    let mut htod_payload = devptr.to_le_bytes().to_vec();
    htod_payload.extend_from_slice(&test_data);
    let (_hdr, payload) =
        roundtrip(&client, MessageType::MemcpyHtoD, &htod_payload, 3).await;
    assert_eq!(
        response_result(&payload),
        CuResult::Success,
        "MemcpyHtoD to global pointer should succeed"
    );

    let mut dtoh_payload = devptr.to_le_bytes().to_vec();
    dtoh_payload.extend_from_slice(&(size as u64).to_le_bytes());
    let (_hdr, payload) =
        roundtrip(&client, MessageType::MemcpyDtoH, &dtoh_payload, 4).await;
    let data = assert_success(&payload);
    assert_eq!(
        data, &test_data[..],
        "data read back from global should match what was written"
    );

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_stream_lifecycle_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // 1. Create a stream (flags=0).
    let create_payload = 0_u32.to_le_bytes();
    let (_hdr, payload) =
        roundtrip(&client, MessageType::StreamCreate, &create_payload, 1).await;
    let data = assert_success(&payload);
    let stream = u64::from_le_bytes(data[..8].try_into().unwrap());
    assert_ne!(stream, 0, "stream handle should be non-zero");

    // 2. Synchronize the stream.
    let stream_payload = stream.to_le_bytes();
    let (_hdr, payload) =
        roundtrip(&client, MessageType::StreamSynchronize, &stream_payload, 2).await;
    assert_eq!(
        response_result(&payload),
        CuResult::Success,
        "StreamSynchronize should succeed"
    );

    // 3. Query the stream.
    let (_hdr, payload) =
        roundtrip(&client, MessageType::StreamQuery, &stream_payload, 3).await;
    assert_eq!(
        response_result(&payload),
        CuResult::Success,
        "StreamQuery should succeed"
    );

    // 4. Destroy the stream.
    let (_hdr, payload) =
        roundtrip(&client, MessageType::StreamDestroy, &stream_payload, 4).await;
    assert_eq!(
        response_result(&payload),
        CuResult::Success,
        "StreamDestroy should succeed"
    );

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_event_lifecycle_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // 1. Create two events (flags=0).
    let flags_payload = 0_u32.to_le_bytes();
    let (_hdr, payload) =
        roundtrip(&client, MessageType::EventCreate, &flags_payload, 1).await;
    let data = assert_success(&payload);
    let event1 = u64::from_le_bytes(data[..8].try_into().unwrap());
    assert_ne!(event1, 0, "event1 handle should be non-zero");

    let (_hdr, payload) =
        roundtrip(&client, MessageType::EventCreate, &flags_payload, 2).await;
    let data = assert_success(&payload);
    let event2 = u64::from_le_bytes(data[..8].try_into().unwrap());
    assert_ne!(event2, 0, "event2 handle should be non-zero");

    // 2. Record event1 on default stream (stream=0).
    let mut record_payload = event1.to_le_bytes().to_vec();
    record_payload.extend_from_slice(&0_u64.to_le_bytes());
    let (_hdr, payload) =
        roundtrip(&client, MessageType::EventRecord, &record_payload, 3).await;
    assert_eq!(
        response_result(&payload),
        CuResult::Success,
        "EventRecord for event1 should succeed"
    );

    // 3. Record event2 on default stream (stream=0).
    let mut record_payload = event2.to_le_bytes().to_vec();
    record_payload.extend_from_slice(&0_u64.to_le_bytes());
    let (_hdr, payload) =
        roundtrip(&client, MessageType::EventRecord, &record_payload, 4).await;
    assert_eq!(
        response_result(&payload),
        CuResult::Success,
        "EventRecord for event2 should succeed"
    );

    // 4. Get elapsed time between event1 and event2.
    let mut elapsed_payload = event1.to_le_bytes().to_vec();
    elapsed_payload.extend_from_slice(&event2.to_le_bytes());
    let (_hdr, payload) =
        roundtrip(&client, MessageType::EventElapsedTime, &elapsed_payload, 5).await;
    let data = assert_success(&payload);
    let ms = f32::from_le_bytes(data[..4].try_into().unwrap());
    assert!(
        ms >= 0.0,
        "elapsed time should be non-negative, got {ms}"
    );

    // 5. Destroy both events.
    let (_hdr, payload) =
        roundtrip(&client, MessageType::EventDestroy, &event1.to_le_bytes(), 6).await;
    assert_eq!(
        response_result(&payload),
        CuResult::Success,
        "EventDestroy for event1 should succeed"
    );

    let (_hdr, payload) =
        roundtrip(&client, MessageType::EventDestroy, &event2.to_le_bytes(), 7).await;
    assert_eq!(
        response_result(&payload),
        CuResult::Success,
        "EventDestroy for event2 should succeed"
    );

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_launch_kernel_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // 1. Load module.
    let ptx = b"fake ptx";
    let (_hdr, payload) = roundtrip(&client, MessageType::ModuleLoadData, ptx, 1).await;
    let data = assert_success(&payload);
    let module = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 2. Get function.
    let name = b"my_kernel";
    let mut func_payload = module.to_le_bytes().to_vec();
    func_payload.extend_from_slice(&(name.len() as u32).to_le_bytes());
    func_payload.extend_from_slice(name);
    let (_hdr, payload) =
        roundtrip(&client, MessageType::ModuleGetFunction, &func_payload, 2).await;
    let data = assert_success(&payload);
    let func = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 3. Create a stream.
    let create_stream_payload = 0_u32.to_le_bytes();
    let (_hdr, payload) =
        roundtrip(&client, MessageType::StreamCreate, &create_stream_payload, 3).await;
    let data = assert_success(&payload);
    let stream = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 4. Launch kernel:
    // [8B func][4B gx=1][4B gy=1][4B gz=1][4B bx=32][4B by=1][4B bz=1][4B smem=0][8B stream]
    let mut launch_payload = func.to_le_bytes().to_vec();
    launch_payload.extend_from_slice(&1_u32.to_le_bytes()); // gridX
    launch_payload.extend_from_slice(&1_u32.to_le_bytes()); // gridY
    launch_payload.extend_from_slice(&1_u32.to_le_bytes()); // gridZ
    launch_payload.extend_from_slice(&32_u32.to_le_bytes()); // blockX
    launch_payload.extend_from_slice(&1_u32.to_le_bytes()); // blockY
    launch_payload.extend_from_slice(&1_u32.to_le_bytes()); // blockZ
    launch_payload.extend_from_slice(&0_u32.to_le_bytes()); // sharedMem
    launch_payload.extend_from_slice(&stream.to_le_bytes()); // stream
    launch_payload.extend_from_slice(&0_u32.to_le_bytes()); // num_params = 0 (matches real client wire format)
    let (_hdr, payload) =
        roundtrip(&client, MessageType::LaunchKernel, &launch_payload, 4).await;
    assert_eq!(
        response_result(&payload),
        CuResult::Success,
        "LaunchKernel should succeed"
    );

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_launch_kernel_with_params_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // 1. Load module.
    let ptx = b"fake ptx";
    let (_hdr, payload) = roundtrip(&client, MessageType::ModuleLoadData, ptx, 1).await;
    let data = assert_success(&payload);
    let module = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 2. Get function.
    let name = b"doubling";
    let mut func_payload = module.to_le_bytes().to_vec();
    func_payload.extend_from_slice(&(name.len() as u32).to_le_bytes());
    func_payload.extend_from_slice(name);
    let (_hdr, payload) =
        roundtrip(&client, MessageType::ModuleGetFunction, &func_payload, 2).await;
    let data = assert_success(&payload);
    let func = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 3. Launch kernel with serialized params (2 params: u64 ptr + u32 count)
    let mut launch_payload = func.to_le_bytes().to_vec();
    launch_payload.extend_from_slice(&1_u32.to_le_bytes()); // gridX
    launch_payload.extend_from_slice(&1_u32.to_le_bytes()); // gridY
    launch_payload.extend_from_slice(&1_u32.to_le_bytes()); // gridZ
    launch_payload.extend_from_slice(&32_u32.to_le_bytes()); // blockX
    launch_payload.extend_from_slice(&1_u32.to_le_bytes()); // blockY
    launch_payload.extend_from_slice(&1_u32.to_le_bytes()); // blockZ
    launch_payload.extend_from_slice(&0_u32.to_le_bytes()); // sharedMem
    launch_payload.extend_from_slice(&0_u64.to_le_bytes()); // stream (default)
    // Serialized kernel params:
    launch_payload.extend_from_slice(&2_u32.to_le_bytes()); // num_params = 2
    launch_payload.extend_from_slice(&8_u32.to_le_bytes()); // param 0 size = 8
    launch_payload.extend_from_slice(&0xABCD_0000_u64.to_le_bytes()); // param 0: device ptr
    launch_payload.extend_from_slice(&4_u32.to_le_bytes()); // param 1 size = 4
    launch_payload.extend_from_slice(&512_u32.to_le_bytes()); // param 1: count
    let (_hdr, payload) =
        roundtrip(&client, MessageType::LaunchKernel, &launch_payload, 3).await;
    assert_eq!(
        response_result(&payload),
        CuResult::Success,
        "LaunchKernel with params should succeed"
    );

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_ctx_session_persistence_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // 1. Create a context (flags=0, device=0).
    let mut ctx_create_payload = 0_u32.to_le_bytes().to_vec();
    ctx_create_payload.extend_from_slice(&0_i32.to_le_bytes());
    let (_hdr, payload) =
        roundtrip(&client, MessageType::CtxCreate, &ctx_create_payload, 1).await;
    let data = assert_success(&payload);
    let ctx_handle = u64::from_le_bytes(data[..8].try_into().unwrap());
    assert_ne!(ctx_handle, 0, "context handle should be non-zero");

    // 2. CtxGetCurrent should return the same handle (session persistence).
    let (_hdr, payload) =
        roundtrip(&client, MessageType::CtxGetCurrent, &[], 2).await;
    let data = assert_success(&payload);
    let current_ctx = u64::from_le_bytes(data[..8].try_into().unwrap());
    assert_eq!(
        current_ctx, ctx_handle,
        "CtxGetCurrent should return the handle from CtxCreate (session persistence)"
    );

    drop(client);
    server.await.unwrap();
}

// ---------------------------------------------------------------------------
// MemcpyDtoD / MemAllocHost / MemFreeHost / StreamWaitEvent E2E tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_memcpy_dtod_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // 1. Allocate buffer A (256 bytes).
    let alloc_payload = 256_u64.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::MemAlloc, &alloc_payload, 1).await;
    let data = assert_success(&payload);
    let buf_a = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 2. Allocate buffer B (256 bytes).
    let (_hdr, payload) = roundtrip(&client, MessageType::MemAlloc, &alloc_payload, 2).await;
    let data = assert_success(&payload);
    let buf_b = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 3. Write known data to buffer A via MemcpyHtoD.
    let test_data: Vec<u8> = (0..256).map(|i| (i & 0xFF) as u8).collect();
    let mut htod_payload = buf_a.to_le_bytes().to_vec();
    htod_payload.extend_from_slice(&test_data);
    let (_hdr, payload) = roundtrip(&client, MessageType::MemcpyHtoD, &htod_payload, 3).await;
    assert_eq!(response_result(&payload), CuResult::Success, "MemcpyHtoD should succeed");

    // 4. Copy A -> B via MemcpyDtoD: [8B dst][8B src][8B size].
    let mut dtod_payload = buf_b.to_le_bytes().to_vec();
    dtod_payload.extend_from_slice(&buf_a.to_le_bytes());
    dtod_payload.extend_from_slice(&256_u64.to_le_bytes());
    let (_hdr, payload) = roundtrip(&client, MessageType::MemcpyDtoD, &dtod_payload, 4).await;
    assert_eq!(response_result(&payload), CuResult::Success, "MemcpyDtoD should succeed");

    // 5. Read B back via MemcpyDtoH.
    let mut dtoh_payload = buf_b.to_le_bytes().to_vec();
    dtoh_payload.extend_from_slice(&256_u64.to_le_bytes());
    let (_hdr, payload) = roundtrip(&client, MessageType::MemcpyDtoH, &dtoh_payload, 5).await;
    let data = assert_success(&payload);
    assert_eq!(
        data, &test_data[..],
        "data read from buffer B should match what was written to buffer A"
    );

    // 6. Free both buffers.
    let (_hdr, payload) = roundtrip(&client, MessageType::MemFree, &buf_a.to_le_bytes(), 6).await;
    assert_eq!(response_result(&payload), CuResult::Success);
    let (_hdr, payload) = roundtrip(&client, MessageType::MemFree, &buf_b.to_le_bytes(), 7).await;
    assert_eq!(response_result(&payload), CuResult::Success);

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_mem_alloc_host_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // 1. MemAllocHost: [8B size=1024].
    let alloc_payload = 1024_u64.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::MemAllocHost, &alloc_payload, 1).await;
    let data = assert_success(&payload);
    let host_ptr = u64::from_le_bytes(data[..8].try_into().unwrap());
    assert_ne!(host_ptr, 0, "host pointer should be non-zero");

    // 2. MemFreeHost: [8B ptr].
    let free_payload = host_ptr.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::MemFreeHost, &free_payload, 2).await;
    assert_eq!(
        response_result(&payload),
        CuResult::Success,
        "MemFreeHost should succeed"
    );

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_mem_free_host_double_free_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // 1. AllocHost.
    let alloc_payload = 512_u64.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::MemAllocHost, &alloc_payload, 1).await;
    let data = assert_success(&payload);
    let host_ptr = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 2. First FreeHost -- should succeed.
    let free_payload = host_ptr.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::MemFreeHost, &free_payload, 2).await;
    assert_eq!(
        response_result(&payload),
        CuResult::Success,
        "first MemFreeHost should succeed"
    );

    // 3. Second FreeHost of the same pointer -- should return InvalidValue.
    let (_hdr, payload) = roundtrip(&client, MessageType::MemFreeHost, &free_payload, 3).await;
    assert_eq!(
        response_result(&payload),
        CuResult::InvalidValue,
        "double MemFreeHost should return InvalidValue"
    );

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_stream_wait_event_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // 1. Create a stream (flags=0).
    let create_stream_payload = 0_u32.to_le_bytes();
    let (_hdr, payload) =
        roundtrip(&client, MessageType::StreamCreate, &create_stream_payload, 1).await;
    let data = assert_success(&payload);
    let stream = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 2. Create an event (flags=0).
    let create_event_payload = 0_u32.to_le_bytes();
    let (_hdr, payload) =
        roundtrip(&client, MessageType::EventCreate, &create_event_payload, 2).await;
    let data = assert_success(&payload);
    let event = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 3. Record the event on stream 0 (default stream).
    let mut record_payload = event.to_le_bytes().to_vec();
    record_payload.extend_from_slice(&0_u64.to_le_bytes());
    let (_hdr, payload) =
        roundtrip(&client, MessageType::EventRecord, &record_payload, 3).await;
    assert_eq!(
        response_result(&payload),
        CuResult::Success,
        "EventRecord should succeed"
    );

    // 4. StreamWaitEvent: [8B stream][8B event][4B flags=0].
    let mut wait_payload = stream.to_le_bytes().to_vec();
    wait_payload.extend_from_slice(&event.to_le_bytes());
    wait_payload.extend_from_slice(&0_u32.to_le_bytes());
    let (_hdr, payload) =
        roundtrip(&client, MessageType::StreamWaitEvent, &wait_payload, 4).await;
    assert_eq!(
        response_result(&payload),
        CuResult::Success,
        "StreamWaitEvent should succeed"
    );

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_stream_wait_event_invalid_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // 1. Create a stream (flags=0).
    let create_stream_payload = 0_u32.to_le_bytes();
    let (_hdr, payload) =
        roundtrip(&client, MessageType::StreamCreate, &create_stream_payload, 1).await;
    let data = assert_success(&payload);
    let stream = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 2. StreamWaitEvent with an invalid event handle.
    let mut wait_payload = stream.to_le_bytes().to_vec();
    wait_payload.extend_from_slice(&0xBAD_u64.to_le_bytes());
    wait_payload.extend_from_slice(&0_u32.to_le_bytes());
    let (_hdr, payload) =
        roundtrip(&client, MessageType::StreamWaitEvent, &wait_payload, 2).await;
    assert_eq!(
        response_result(&payload),
        CuResult::InvalidValue,
        "StreamWaitEvent with invalid event should return InvalidValue"
    );

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_memcpy_dtod_invalid_src_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // 1. Allocate a valid destination buffer.
    let alloc_payload = 64_u64.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::MemAlloc, &alloc_payload, 1).await;
    let data = assert_success(&payload);
    let dst_ptr = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 2. MemcpyDtoD with invalid source pointer: [8B dst][8B src=0xBAD][8B size].
    let mut dtod_payload = dst_ptr.to_le_bytes().to_vec();
    dtod_payload.extend_from_slice(&0xBAD_u64.to_le_bytes());
    dtod_payload.extend_from_slice(&64_u64.to_le_bytes());
    let (_hdr, payload) = roundtrip(&client, MessageType::MemcpyDtoD, &dtod_payload, 2).await;
    assert_eq!(
        response_result(&payload),
        CuResult::InvalidValue,
        "MemcpyDtoD with invalid source should return InvalidValue"
    );

    drop(client);
    server.await.unwrap();
}

// ---------------------------------------------------------------------------
// Context push/pop E2E tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_ctx_push_pop_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // 1. Create two contexts.
    let mut create_payload = 0_u32.to_le_bytes().to_vec();
    create_payload.extend_from_slice(&0_i32.to_le_bytes());
    let (_hdr, payload) =
        roundtrip(&client, MessageType::CtxCreate, &create_payload, 1).await;
    let data = assert_success(&payload);
    let ctx1 = u64::from_le_bytes(data[..8].try_into().unwrap());

    let (_hdr, payload) =
        roundtrip(&client, MessageType::CtxCreate, &create_payload, 2).await;
    let data = assert_success(&payload);
    let ctx2 = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 2. Push ctx1 onto the stack.
    let push_payload = ctx1.to_le_bytes();
    let (_hdr, payload) =
        roundtrip(&client, MessageType::CtxPushCurrent, &push_payload, 3).await;
    assert_eq!(
        response_result(&payload),
        CuResult::Success,
        "CtxPushCurrent should succeed"
    );

    // 3. Get current -- should be ctx1.
    let (_hdr, payload) =
        roundtrip(&client, MessageType::CtxGetCurrent, &[], 4).await;
    let data = assert_success(&payload);
    let current = u64::from_le_bytes(data[..8].try_into().unwrap());
    assert_eq!(current, ctx1, "current context should be ctx1 after push");

    // 4. Push ctx2 on top.
    let push_payload = ctx2.to_le_bytes();
    let (_hdr, payload) =
        roundtrip(&client, MessageType::CtxPushCurrent, &push_payload, 5).await;
    assert_eq!(response_result(&payload), CuResult::Success);

    // 5. Get current -- should be ctx2.
    let (_hdr, payload) =
        roundtrip(&client, MessageType::CtxGetCurrent, &[], 6).await;
    let data = assert_success(&payload);
    let current = u64::from_le_bytes(data[..8].try_into().unwrap());
    assert_eq!(current, ctx2, "current context should be ctx2 after second push");

    // 6. Pop -- should return ctx2, current becomes ctx1.
    let (_hdr, payload) =
        roundtrip(&client, MessageType::CtxPopCurrent, &[], 7).await;
    let data = assert_success(&payload);
    let popped = u64::from_le_bytes(data[..8].try_into().unwrap());
    assert_eq!(popped, ctx2, "popped context should be ctx2");

    // 7. Verify current is now ctx1.
    let (_hdr, payload) =
        roundtrip(&client, MessageType::CtxGetCurrent, &[], 8).await;
    let data = assert_success(&payload);
    let current = u64::from_le_bytes(data[..8].try_into().unwrap());
    assert_eq!(current, ctx1, "current context should be ctx1 after pop");

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_ctx_push_invalid_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // Push an invalid context handle.
    let push_payload = 0xBAD_u64.to_le_bytes();
    let (_hdr, payload) =
        roundtrip(&client, MessageType::CtxPushCurrent, &push_payload, 1).await;
    assert_eq!(
        response_result(&payload),
        CuResult::InvalidContext,
        "CtxPushCurrent with invalid handle should return InvalidContext"
    );

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_ctx_pop_empty_stack_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // Pop from empty stack.
    let (_hdr, payload) =
        roundtrip(&client, MessageType::CtxPopCurrent, &[], 1).await;
    assert_eq!(
        response_result(&payload),
        CuResult::InvalidContext,
        "CtxPopCurrent on empty stack should return InvalidContext"
    );

    drop(client);
    server.await.unwrap();
}

// ---------------------------------------------------------------------------
// FuncGetAttribute E2E tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_func_get_attribute_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // 1. Load a module.
    let mod_data = b"fake ptx data";
    let (_hdr, payload) =
        roundtrip(&client, MessageType::ModuleLoadData, mod_data, 1).await;
    let data = assert_success(&payload);
    let module = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 2. Get a function from the module.
    let kern_name = b"my_kernel";
    let mut func_payload = module.to_le_bytes().to_vec();
    func_payload.extend_from_slice(&(kern_name.len() as u32).to_le_bytes());
    func_payload.extend_from_slice(kern_name);
    let (_hdr, payload) =
        roundtrip(&client, MessageType::ModuleGetFunction, &func_payload, 2).await;
    let data = assert_success(&payload);
    let func = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 3. Query MAX_THREADS_PER_BLOCK (attrib 0).
    let mut attr_payload = func.to_le_bytes().to_vec();
    attr_payload.extend_from_slice(&0_i32.to_le_bytes());
    let (_hdr, payload) =
        roundtrip(&client, MessageType::FuncGetAttribute, &attr_payload, 3).await;
    let data = assert_success(&payload);
    let max_threads = i32::from_le_bytes(data[..4].try_into().unwrap());
    assert_eq!(max_threads, 1024, "max threads per block should be 1024");

    // 4. Query SHARED_SIZE_BYTES (attrib 1).
    let mut attr_payload = func.to_le_bytes().to_vec();
    attr_payload.extend_from_slice(&1_i32.to_le_bytes());
    let (_hdr, payload) =
        roundtrip(&client, MessageType::FuncGetAttribute, &attr_payload, 4).await;
    let data = assert_success(&payload);
    let shared_mem = i32::from_le_bytes(data[..4].try_into().unwrap());
    assert_eq!(shared_mem, 49152, "shared mem size should be 49152");

    // 5. Query NUM_REGS (attrib 4).
    let mut attr_payload = func.to_le_bytes().to_vec();
    attr_payload.extend_from_slice(&4_i32.to_le_bytes());
    let (_hdr, payload) =
        roundtrip(&client, MessageType::FuncGetAttribute, &attr_payload, 5).await;
    let data = assert_success(&payload);
    let num_regs = i32::from_le_bytes(data[..4].try_into().unwrap());
    assert_eq!(num_regs, 32, "num regs should be 32");

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_func_get_attribute_invalid_func_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // Query attribute on a non-existent function.
    let mut attr_payload = 0xBAD_u64.to_le_bytes().to_vec();
    attr_payload.extend_from_slice(&0_i32.to_le_bytes());
    let (_hdr, payload) =
        roundtrip(&client, MessageType::FuncGetAttribute, &attr_payload, 1).await;
    assert_eq!(
        response_result(&payload),
        CuResult::InvalidValue,
        "FuncGetAttribute with invalid func should return InvalidValue"
    );

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_func_get_attribute_invalid_attrib_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // 1. Load a module and get a function.
    let mod_data = b"ptx";
    let (_hdr, payload) =
        roundtrip(&client, MessageType::ModuleLoadData, mod_data, 1).await;
    let data = assert_success(&payload);
    let module = u64::from_le_bytes(data[..8].try_into().unwrap());

    let kern_name = b"kern";
    let mut func_payload = module.to_le_bytes().to_vec();
    func_payload.extend_from_slice(&(kern_name.len() as u32).to_le_bytes());
    func_payload.extend_from_slice(kern_name);
    let (_hdr, payload) =
        roundtrip(&client, MessageType::ModuleGetFunction, &func_payload, 2).await;
    let data = assert_success(&payload);
    let func = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 2. Query with an invalid attribute code.
    let mut attr_payload = func.to_le_bytes().to_vec();
    attr_payload.extend_from_slice(&99999_i32.to_le_bytes());
    let (_hdr, payload) =
        roundtrip(&client, MessageType::FuncGetAttribute, &attr_payload, 3).await;
    assert_eq!(
        response_result(&payload),
        CuResult::InvalidValue,
        "FuncGetAttribute with invalid attrib should return InvalidValue"
    );

    drop(client);
    server.await.unwrap();
}

// ---------------------------------------------------------------------------
// Expanded DeviceGetAttribute E2E test
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_device_get_attribute_expanded_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // Query a range of attributes that real apps commonly request.
    let attributes_and_expected: Vec<(i32, i32)> = vec![
        (1, 1024),   // MaxThreadsPerBlock
        (10, 32),    // WarpSize
        (16, 82),    // MultiprocessorCount
        (75, 8),     // ComputeCapabilityMajor
        (76, 6),     // ComputeCapabilityMinor
        (8, 49152),  // MaxSharedMemoryPerBlock
        (36, 1),     // ConcurrentKernels
        (47, 1),     // UnifiedAddressing
    ];

    for (i, (attrib, expected)) in attributes_and_expected.iter().enumerate() {
        let mut payload = attrib.to_le_bytes().to_vec();
        payload.extend_from_slice(&0_i32.to_le_bytes()); // device 0
        let (_hdr, resp) =
            roundtrip(&client, MessageType::DeviceGetAttribute, &payload, (i + 1) as u64).await;
        let data = assert_success(&resp);
        let val = i32::from_le_bytes(data[..4].try_into().unwrap());
        assert_eq!(
            val, *expected,
            "attribute {} should return {}, got {}",
            attrib, expected, val
        );
    }

    drop(client);
    server.await.unwrap();
}
