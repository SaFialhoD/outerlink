//! End-to-end integration tests for the OuterLink client-server protocol.
//!
//! Each test starts a real TCP server on localhost (port chosen by the OS),
//! connects a client via `TcpTransportConnection`, and exercises the full
//! request-response cycle through the wire protocol and handler.
//!
//! We use raw `TcpTransportConnection` on the client side (Option A) rather
//! than `OuterLinkClient` because `OuterLinkClient::send_request` calls
//! `runtime.block_on()`, which panics when called from within a tokio context.

mod common;
use common::*;

use outerlink_common::protocol::MessageType;
use outerlink_server::gpu_backend::StubGpuBackend;

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
// Async memory copy E2E tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_memcpy_htod_async_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // 1. Allocate device memory.
    let alloc_payload = 64_u64.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::MemAlloc, &alloc_payload, 1).await;
    let data = assert_success(&payload);
    let dev_ptr = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 2. Create a stream.
    let stream_payload = 0_u32.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::StreamCreate, &stream_payload, 2).await;
    let data = assert_success(&payload);
    let stream = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 3. MemcpyHtoDAsync: [8B dst][8B stream][data...]
    let src_data = vec![0xABu8; 32];
    let mut htod_payload = Vec::new();
    htod_payload.extend_from_slice(&dev_ptr.to_le_bytes());
    htod_payload.extend_from_slice(&stream.to_le_bytes());
    htod_payload.extend_from_slice(&src_data);
    let (_hdr, payload) =
        roundtrip(&client, MessageType::MemcpyHtoDAsync, &htod_payload, 3).await;
    assert_success(&payload);

    // 4. Verify data was written by reading it back (sync DtoH).
    let mut dtoh_payload = [0u8; 16];
    dtoh_payload[0..8].copy_from_slice(&dev_ptr.to_le_bytes());
    dtoh_payload[8..16].copy_from_slice(&32_u64.to_le_bytes());
    let (_hdr, payload) = roundtrip(&client, MessageType::MemcpyDtoH, &dtoh_payload, 4).await;
    let data = assert_success(&payload);
    assert_eq!(&data[..32], &src_data[..]);

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_memcpy_dtoh_async_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // 1. Allocate device memory and write data.
    let alloc_payload = 64_u64.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::MemAlloc, &alloc_payload, 1).await;
    let data = assert_success(&payload);
    let dev_ptr = u64::from_le_bytes(data[..8].try_into().unwrap());

    let src_data = vec![0xCDu8; 16];
    let mut htod_payload = Vec::new();
    htod_payload.extend_from_slice(&dev_ptr.to_le_bytes());
    htod_payload.extend_from_slice(&src_data);
    let (_hdr, _payload) = roundtrip(&client, MessageType::MemcpyHtoD, &htod_payload, 2).await;

    // 2. Create a stream.
    let stream_payload = 0_u32.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::StreamCreate, &stream_payload, 3).await;
    let data = assert_success(&payload);
    let stream = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 3. MemcpyDtoHAsync: [8B src][8B size][8B stream]
    let mut dtoh_payload = [0u8; 24];
    dtoh_payload[0..8].copy_from_slice(&dev_ptr.to_le_bytes());
    dtoh_payload[8..16].copy_from_slice(&16_u64.to_le_bytes());
    dtoh_payload[16..24].copy_from_slice(&stream.to_le_bytes());
    let (_hdr, payload) =
        roundtrip(&client, MessageType::MemcpyDtoHAsync, &dtoh_payload, 4).await;
    let data = assert_success(&payload);
    assert_eq!(&data[..16], &src_data[..]);

    drop(client);
    server.await.unwrap();
}

// ---------------------------------------------------------------------------
// Memset E2E tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_memset_d8_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // 1. Allocate device memory.
    let alloc_payload = 64_u64.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::MemAlloc, &alloc_payload, 1).await;
    let data = assert_success(&payload);
    let dev_ptr = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 2. MemsetD8: [8B dst][1B value][8B count]
    let mut memset_payload = [0u8; 17];
    memset_payload[0..8].copy_from_slice(&dev_ptr.to_le_bytes());
    memset_payload[8] = 0x42;
    memset_payload[9..17].copy_from_slice(&32_u64.to_le_bytes());
    let (_hdr, payload) = roundtrip(&client, MessageType::MemsetD8, &memset_payload, 2).await;
    assert_success(&payload);

    // 3. Read back and verify.
    let mut dtoh_payload = [0u8; 16];
    dtoh_payload[0..8].copy_from_slice(&dev_ptr.to_le_bytes());
    dtoh_payload[8..16].copy_from_slice(&32_u64.to_le_bytes());
    let (_hdr, payload) = roundtrip(&client, MessageType::MemcpyDtoH, &dtoh_payload, 3).await;
    let data = assert_success(&payload);
    assert!(data[..32].iter().all(|&b| b == 0x42));

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_memset_d32_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // 1. Allocate device memory (must be multiple of 4).
    let alloc_payload = 64_u64.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::MemAlloc, &alloc_payload, 1).await;
    let data = assert_success(&payload);
    let dev_ptr = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 2. MemsetD32: [8B dst][4B value][8B count] -- count is # of u32 elements
    let mut memset_payload = [0u8; 20];
    memset_payload[0..8].copy_from_slice(&dev_ptr.to_le_bytes());
    memset_payload[8..12].copy_from_slice(&0xDEADBEEF_u32.to_le_bytes());
    memset_payload[12..20].copy_from_slice(&8_u64.to_le_bytes()); // 8 u32s = 32 bytes
    let (_hdr, payload) = roundtrip(&client, MessageType::MemsetD32, &memset_payload, 2).await;
    assert_success(&payload);

    // 3. Read back and verify.
    let mut dtoh_payload = [0u8; 16];
    dtoh_payload[0..8].copy_from_slice(&dev_ptr.to_le_bytes());
    dtoh_payload[8..16].copy_from_slice(&32_u64.to_le_bytes()); // 8 * 4 = 32 bytes
    let (_hdr, payload) = roundtrip(&client, MessageType::MemcpyDtoH, &dtoh_payload, 3).await;
    let data = assert_success(&payload);
    // Verify each u32 element
    for i in 0..8 {
        let val = u32::from_le_bytes(data[i * 4..(i + 1) * 4].try_into().unwrap());
        assert_eq!(val, 0xDEADBEEF, "u32 element {i} mismatch");
    }

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_memset_d8_async_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // 1. Allocate device memory.
    let alloc_payload = 64_u64.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::MemAlloc, &alloc_payload, 1).await;
    let data = assert_success(&payload);
    let dev_ptr = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 2. Create a stream.
    let stream_payload = 0_u32.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::StreamCreate, &stream_payload, 2).await;
    let data = assert_success(&payload);
    let stream = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 3. MemsetD8Async: [8B dst][1B value][8B count][8B stream]
    let mut memset_payload = [0u8; 25];
    memset_payload[0..8].copy_from_slice(&dev_ptr.to_le_bytes());
    memset_payload[8] = 0x77;
    memset_payload[9..17].copy_from_slice(&16_u64.to_le_bytes());
    memset_payload[17..25].copy_from_slice(&stream.to_le_bytes());
    let (_hdr, payload) =
        roundtrip(&client, MessageType::MemsetD8Async, &memset_payload, 3).await;
    assert_success(&payload);

    // 4. Read back and verify.
    let mut dtoh_payload = [0u8; 16];
    dtoh_payload[0..8].copy_from_slice(&dev_ptr.to_le_bytes());
    dtoh_payload[8..16].copy_from_slice(&16_u64.to_le_bytes());
    let (_hdr, payload) = roundtrip(&client, MessageType::MemcpyDtoH, &dtoh_payload, 4).await;
    let data = assert_success(&payload);
    assert!(data[..16].iter().all(|&b| b == 0x77));

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_memset_d32_async_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // 1. Allocate device memory.
    let alloc_payload = 64_u64.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::MemAlloc, &alloc_payload, 1).await;
    let data = assert_success(&payload);
    let dev_ptr = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 2. Create a stream.
    let stream_payload = 0_u32.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::StreamCreate, &stream_payload, 2).await;
    let data = assert_success(&payload);
    let stream = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 3. MemsetD32Async: [8B dst][4B value][8B count][8B stream]
    let mut memset_payload = [0u8; 28];
    memset_payload[0..8].copy_from_slice(&dev_ptr.to_le_bytes());
    memset_payload[8..12].copy_from_slice(&0xCAFEBABE_u32.to_le_bytes());
    memset_payload[12..20].copy_from_slice(&4_u64.to_le_bytes()); // 4 u32s = 16 bytes
    memset_payload[20..28].copy_from_slice(&stream.to_le_bytes());
    let (_hdr, payload) =
        roundtrip(&client, MessageType::MemsetD32Async, &memset_payload, 3).await;
    assert_success(&payload);

    // 4. Read back and verify.
    let mut dtoh_payload = [0u8; 16];
    dtoh_payload[0..8].copy_from_slice(&dev_ptr.to_le_bytes());
    dtoh_payload[8..16].copy_from_slice(&16_u64.to_le_bytes());
    let (_hdr, payload) = roundtrip(&client, MessageType::MemcpyDtoH, &dtoh_payload, 4).await;
    let data = assert_success(&payload);
    for i in 0..4 {
        let val = u32::from_le_bytes(data[i * 4..(i + 1) * 4].try_into().unwrap());
        assert_eq!(val, 0xCAFEBABE, "u32 element {i} mismatch");
    }

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_memset_d8_short_payload_e2e() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let client = connect_client(&addr).await;

    // Send a too-short payload for MemsetD8 (needs 17 bytes minimum).
    let short_payload = [0u8; 8];
    let (_hdr, payload) =
        roundtrip(&client, MessageType::MemsetD8, &short_payload, 1).await;
    assert_eq!(
        response_result(&payload),
        CuResult::InvalidValue,
        "MemsetD8 with short payload should return InvalidValue"
    );

    drop(client);
    server.await.unwrap();
}
