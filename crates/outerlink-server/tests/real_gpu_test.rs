//! Real GPU hardware tests for the OuterLink server.
//!
//! These tests exercise the full stack: client TCP transport -> server handler
//! -> CudaGpuBackend -> real NVIDIA driver -> real GPU hardware.
//!
//! Gated behind `real-gpu-test` feature so they never run in normal CI.
//!
//! Run with:
//! ```
//! cargo test -p outerlink-server --features real-gpu-test --test real_gpu_test -- --nocapture
//! ```

#![cfg(feature = "real-gpu-test")]

use std::sync::Arc;

use tokio::net::TcpListener;

use outerlink_common::cuda_types::CuResult;
use outerlink_common::error::OuterLinkError;
use outerlink_common::protocol::{MessageHeader, MessageType};
use outerlink_common::tcp_transport::TcpTransportConnection;
use outerlink_common::transport::TransportConnection;
use outerlink_server::cuda_backend::CudaGpuBackend;
use outerlink_server::gpu_backend::GpuBackend;
use outerlink_server::handler::handle_request;
use outerlink_server::session::ConnectionSession;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Try to create a CudaGpuBackend.  Returns `None` if the CUDA driver is
/// not available (e.g. no GPU, no driver installed), allowing tests to skip
/// gracefully rather than fail.
fn try_create_backend() -> Option<Arc<dyn GpuBackend>> {
    let backend = match CudaGpuBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[SKIP] CudaGpuBackend::new() failed: {e}");
            eprintln!("[SKIP] No CUDA driver available -- skipping real GPU tests");
            return None;
        }
    };
    let init_result = backend.init();
    if !init_result.is_success() {
        eprintln!("[SKIP] CudaGpuBackend::init() returned {init_result:?} -- skipping");
        return None;
    }
    Some(Arc::new(backend))
}

/// Macro that skips the test early if CUDA is unavailable.
macro_rules! require_gpu {
    () => {
        match try_create_backend() {
            Some(b) => b,
            None => return,
        }
    };
}

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
fn spawn_real_server(
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

            let (resp_header, resp_payload) =
                handle_request(&*backend, &header, &payload, &mut session);
            conn.send_message(&resp_header, &resp_payload)
                .await
                .expect("server send failed");
        }
    })
}

/// Bind a TCP listener on localhost with an OS-assigned port.
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

/// Send a request and receive the response.
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
// Real GPU Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_real_device_get_count() {
    let backend = require_gpu!();
    let (listener, addr) = bind_server().await;
    let server = spawn_real_server(listener, backend);
    let client = connect_client(&addr).await;

    let (_hdr, payload) = roundtrip(&client, MessageType::DeviceGetCount, &[], 1).await;
    let data = assert_success(&payload);
    let count = i32::from_le_bytes(data[..4].try_into().unwrap());

    eprintln!("[REAL GPU] Device count: {count}");
    assert!(count >= 1, "expected at least 1 GPU, got {count}");

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_real_driver_version() {
    let backend = require_gpu!();
    let (listener, addr) = bind_server().await;
    let server = spawn_real_server(listener, backend);
    let client = connect_client(&addr).await;

    let (_hdr, payload) = roundtrip(&client, MessageType::DriverGetVersion, &[], 1).await;
    let data = assert_success(&payload);
    let version = i32::from_le_bytes(data[..4].try_into().unwrap());

    eprintln!("[REAL GPU] CUDA driver version: {version}");
    assert!(
        version >= 11000,
        "expected CUDA driver version >= 11000, got {version}"
    );

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_real_device_get_name() {
    let backend = require_gpu!();
    let (listener, addr) = bind_server().await;
    let server = spawn_real_server(listener, backend);
    let client = connect_client(&addr).await;

    let device_payload = 0_i32.to_le_bytes();
    let (_hdr, payload) =
        roundtrip(&client, MessageType::DeviceGetName, &device_payload, 1).await;
    let data = assert_success(&payload);

    let name_len = u32::from_le_bytes(data[..4].try_into().unwrap()) as usize;
    assert!(name_len > 0, "device name should not be empty");
    let name = std::str::from_utf8(&data[4..4 + name_len]).expect("name should be valid UTF-8");

    eprintln!("[REAL GPU] Device name: {name}");
    let name_upper = name.to_uppercase();
    assert!(
        name_upper.contains("RTX")
            || name_upper.contains("GEFORCE")
            || name_upper.contains("NVIDIA")
            || name_upper.contains("TESLA")
            || name_upper.contains("QUADRO"),
        "expected GPU name to contain a known NVIDIA identifier, got: {name}"
    );

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_real_device_total_mem() {
    let backend = require_gpu!();
    let (listener, addr) = bind_server().await;
    let server = spawn_real_server(listener, backend);
    let client = connect_client(&addr).await;

    let device_payload = 0_i32.to_le_bytes();
    let (_hdr, payload) =
        roundtrip(&client, MessageType::DeviceTotalMem, &device_payload, 1).await;
    let data = assert_success(&payload);
    let total_bytes = u64::from_le_bytes(data[..8].try_into().unwrap());

    let total_gib = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!("[REAL GPU] Total VRAM: {total_bytes} bytes ({total_gib:.2} GiB)");

    let one_gib: u64 = 1024 * 1024 * 1024;
    assert!(
        total_bytes > one_gib,
        "expected > 1 GiB VRAM, got {total_bytes} bytes"
    );

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_real_device_attributes() {
    let backend = require_gpu!();
    let (listener, addr) = bind_server().await;
    let server = spawn_real_server(listener, backend);
    let client = connect_client(&addr).await;

    // Query compute capability major (attribute 75)
    let mut attr_payload = 75_i32.to_le_bytes().to_vec(); // ComputeCapabilityMajor
    attr_payload.extend_from_slice(&0_i32.to_le_bytes()); // device 0
    let (_hdr, payload) =
        roundtrip(&client, MessageType::DeviceGetAttribute, &attr_payload, 1).await;
    let data = assert_success(&payload);
    let major = i32::from_le_bytes(data[..4].try_into().unwrap());

    // Query compute capability minor (attribute 76)
    let mut attr_payload = 76_i32.to_le_bytes().to_vec(); // ComputeCapabilityMinor
    attr_payload.extend_from_slice(&0_i32.to_le_bytes()); // device 0
    let (_hdr, payload) =
        roundtrip(&client, MessageType::DeviceGetAttribute, &attr_payload, 2).await;
    let data = assert_success(&payload);
    let minor = i32::from_le_bytes(data[..4].try_into().unwrap());

    eprintln!("[REAL GPU] Compute capability: {major}.{minor}");
    assert!(
        major >= 1,
        "expected compute capability major >= 1, got {major}"
    );

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_real_mem_alloc_free() {
    let backend = require_gpu!();
    let (listener, addr) = bind_server().await;
    let server = spawn_real_server(listener, backend);
    let client = connect_client(&addr).await;

    // Create a context first -- memory operations require an active context
    let mut ctx_payload = 0_u32.to_le_bytes().to_vec();
    ctx_payload.extend_from_slice(&0_i32.to_le_bytes());
    let (_hdr, payload) = roundtrip(&client, MessageType::CtxCreate, &ctx_payload, 1).await;
    assert_success(&payload);

    // Allocate 1 MiB on the real GPU
    let one_mib: u64 = 1024 * 1024;
    let alloc_payload = one_mib.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::MemAlloc, &alloc_payload, 2).await;
    let data = assert_success(&payload);
    let device_ptr = u64::from_le_bytes(data[..8].try_into().unwrap());

    eprintln!("[REAL GPU] Allocated 1 MiB at device ptr: 0x{device_ptr:016X}");
    assert_ne!(device_ptr, 0, "allocated pointer should be non-zero");

    // Free it
    let free_payload = device_ptr.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::MemFree, &free_payload, 3).await;
    let result = response_result(&payload);
    assert_eq!(result, CuResult::Success, "MemFree should succeed");
    eprintln!("[REAL GPU] MemFree succeeded");

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_real_memcpy_roundtrip() {
    let backend = require_gpu!();
    let (listener, addr) = bind_server().await;
    let server = spawn_real_server(listener, backend);
    let client = connect_client(&addr).await;

    // Create a context first -- memory operations require an active context
    let mut ctx_payload = 0_u32.to_le_bytes().to_vec();
    ctx_payload.extend_from_slice(&0_i32.to_le_bytes());
    let (_hdr, payload) = roundtrip(&client, MessageType::CtxCreate, &ctx_payload, 1).await;
    assert_success(&payload);

    // 1. Allocate 4 KiB
    let size: u64 = 4096;
    let alloc_payload = size.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::MemAlloc, &alloc_payload, 2).await;
    let data = assert_success(&payload);
    let device_ptr = u64::from_le_bytes(data[..8].try_into().unwrap());
    eprintln!("[REAL GPU] Allocated 4 KiB at 0x{device_ptr:016X}");

    // 2. Write a known pattern via MemcpyHtoD
    let test_data: Vec<u8> = (0..size as usize).map(|i| (i % 251) as u8).collect();
    let mut htod_payload = device_ptr.to_le_bytes().to_vec();
    htod_payload.extend_from_slice(&test_data);
    let (_hdr, payload) = roundtrip(&client, MessageType::MemcpyHtoD, &htod_payload, 3).await;
    let result = response_result(&payload);
    assert_eq!(result, CuResult::Success, "MemcpyHtoD should succeed");
    eprintln!("[REAL GPU] MemcpyHtoD: wrote 4 KiB to GPU");

    // 3. Read back via MemcpyDtoH
    let mut dtoh_payload = device_ptr.to_le_bytes().to_vec();
    dtoh_payload.extend_from_slice(&size.to_le_bytes());
    let (_hdr, payload) = roundtrip(&client, MessageType::MemcpyDtoH, &dtoh_payload, 4).await;
    let data = assert_success(&payload);

    assert_eq!(
        data.len(),
        size as usize,
        "read-back should return {size} bytes"
    );
    assert_eq!(
        data, &test_data[..],
        "DATA MISMATCH: bytes read back from GPU do not match what was written!"
    );
    eprintln!("[REAL GPU] MemcpyDtoH: read 4 KiB back -- DATA MATCHES BYTE-FOR-BYTE");

    // 4. Free
    let free_payload = device_ptr.to_le_bytes();
    let (_hdr, payload) = roundtrip(&client, MessageType::MemFree, &free_payload, 5).await;
    assert_eq!(response_result(&payload), CuResult::Success);
    eprintln!("[REAL GPU] MemFree succeeded");

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_real_ctx_create_destroy() {
    let backend = require_gpu!();
    let (listener, addr) = bind_server().await;
    let server = spawn_real_server(listener, backend);
    let client = connect_client(&addr).await;

    // CtxCreate: flags=0, device=0
    let mut ctx_create_payload = 0_u32.to_le_bytes().to_vec();
    ctx_create_payload.extend_from_slice(&0_i32.to_le_bytes());
    let (_hdr, payload) =
        roundtrip(&client, MessageType::CtxCreate, &ctx_create_payload, 1).await;
    let data = assert_success(&payload);
    let ctx_handle = u64::from_le_bytes(data[..8].try_into().unwrap());

    eprintln!("[REAL GPU] Created CUDA context: 0x{ctx_handle:016X}");
    assert_ne!(ctx_handle, 0, "context handle should be non-zero");

    // CtxDestroy
    let ctx_destroy_payload = ctx_handle.to_le_bytes();
    let (_hdr, payload) =
        roundtrip(&client, MessageType::CtxDestroy, &ctx_destroy_payload, 2).await;
    let result = response_result(&payload);
    assert_eq!(result, CuResult::Success, "CtxDestroy should succeed");
    eprintln!("[REAL GPU] Destroyed CUDA context");

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_real_mem_get_info() {
    let backend = require_gpu!();
    let (listener, addr) = bind_server().await;
    let server = spawn_real_server(listener, backend);
    let client = connect_client(&addr).await;

    // Create a context first -- memory info queries require an active context
    let mut ctx_payload = 0_u32.to_le_bytes().to_vec();
    ctx_payload.extend_from_slice(&0_i32.to_le_bytes());
    let (_hdr, payload) = roundtrip(&client, MessageType::CtxCreate, &ctx_payload, 1).await;
    assert_success(&payload);

    // MemGetInfo returns free and total VRAM
    let (_hdr, payload) = roundtrip(&client, MessageType::MemGetInfo, &[], 2).await;
    let data = assert_success(&payload);

    let free = u64::from_le_bytes(data[..8].try_into().unwrap());
    let total = u64::from_le_bytes(data[8..16].try_into().unwrap());

    let free_gib = free as f64 / (1024.0 * 1024.0 * 1024.0);
    let total_gib = total as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!("[REAL GPU] VRAM free: {free} bytes ({free_gib:.2} GiB)");
    eprintln!("[REAL GPU] VRAM total: {total} bytes ({total_gib:.2} GiB)");

    assert!(free > 0, "free VRAM should be > 0, got {free}");
    assert!(total > 0, "total VRAM should be > 0, got {total}");
    assert!(
        free <= total,
        "free ({free}) should not exceed total ({total})"
    );

    // Cross-check: total from MemGetInfo should match DeviceTotalMem
    let device_payload = 0_i32.to_le_bytes();
    let (_hdr, payload) =
        roundtrip(&client, MessageType::DeviceTotalMem, &device_payload, 3).await;
    let data = assert_success(&payload);
    let device_total = u64::from_le_bytes(data[..8].try_into().unwrap());

    assert_eq!(
        total, device_total,
        "MemGetInfo total ({total}) should match DeviceTotalMem ({device_total})"
    );
    eprintln!("[REAL GPU] MemGetInfo total matches DeviceTotalMem -- consistent");

    drop(client);
    server.await.unwrap();
}

#[tokio::test]
async fn test_real_kernel_launch() {
    let backend = require_gpu!();
    let (listener, addr) = bind_server().await;
    let server = spawn_real_server(listener, backend);
    let client = connect_client(&addr).await;

    // ---------------------------------------------------------------
    // 1. Create a CUDA context on device 0
    // ---------------------------------------------------------------
    let mut ctx_payload = 0_u32.to_le_bytes().to_vec(); // flags
    ctx_payload.extend_from_slice(&0_i32.to_le_bytes()); // device 0
    let (_hdr, payload) = roundtrip(&client, MessageType::CtxCreate, &ctx_payload, 1).await;
    let data = assert_success(&payload);
    let ctx_handle = u64::from_le_bytes(data[..8].try_into().unwrap());
    eprintln!("[REAL GPU] Created context: 0x{ctx_handle:016X}");

    // ---------------------------------------------------------------
    // 2. Load PTX module
    // ---------------------------------------------------------------
    // A simple kernel that doubles every element: output[tid] = input[tid] * 2
    // Launched with grid=(1,1,1), block=(N,1,1) — N threads in one block using %tid.x.
    let ptx = b"\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry double_elements(\n\
    .param .u64 input,\n\
    .param .u64 output,\n\
    .param .u32 n\n\
)\n\
{\n\
    .reg .pred %p0;\n\
    .reg .u32 %idx, %n;\n\
    .reg .u64 %in_ptr, %out_ptr, %offset;\n\
    .reg .f32 %val, %result;\n\
\n\
    ld.param.u64 %in_ptr, [input];\n\
    ld.param.u64 %out_ptr, [output];\n\
    ld.param.u32 %n, [n];\n\
\n\
    mov.u32 %idx, %tid.x;\n\
    setp.ge.u32 %p0, %idx, %n;\n\
    @%p0 bra done;\n\
\n\
    mul.wide.u32 %offset, %idx, 4;\n\
    add.u64 %in_ptr, %in_ptr, %offset;\n\
    add.u64 %out_ptr, %out_ptr, %offset;\n\
\n\
    ld.global.f32 %val, [%in_ptr];\n\
    add.f32 %result, %val, %val;\n\
    st.global.f32 [%out_ptr], %result;\n\
\n\
done:\n\
    ret;\n\
}\n\0";

    let (_hdr, payload) =
        roundtrip(&client, MessageType::ModuleLoadData, ptx, 2).await;
    let data = assert_success(&payload);
    let module = u64::from_le_bytes(data[..8].try_into().unwrap());
    eprintln!("[REAL GPU] Loaded PTX module: 0x{module:016X}");

    // ---------------------------------------------------------------
    // 3. Get function handle for "double_elements"
    // ---------------------------------------------------------------
    let func_name = b"double_elements";
    let mut gf_payload = module.to_le_bytes().to_vec();
    gf_payload.extend_from_slice(&(func_name.len() as u32).to_le_bytes());
    gf_payload.extend_from_slice(func_name);
    let (_hdr, payload) =
        roundtrip(&client, MessageType::ModuleGetFunction, &gf_payload, 3).await;
    let data = assert_success(&payload);
    let kernel_func = u64::from_le_bytes(data[..8].try_into().unwrap());
    eprintln!("[REAL GPU] Got kernel function: 0x{kernel_func:016X}");

    // ---------------------------------------------------------------
    // 4. Allocate input and output GPU buffers
    // ---------------------------------------------------------------
    let n: u32 = 256;
    let buf_size: u64 = (n as u64) * 4; // N * sizeof(f32)

    // Allocate input buffer
    let (_hdr, payload) =
        roundtrip(&client, MessageType::MemAlloc, &buf_size.to_le_bytes(), 4).await;
    let data = assert_success(&payload);
    let input_ptr = u64::from_le_bytes(data[..8].try_into().unwrap());
    eprintln!("[REAL GPU] Input buffer at: 0x{input_ptr:016X} ({buf_size} bytes)");

    // Allocate output buffer
    let (_hdr, payload) =
        roundtrip(&client, MessageType::MemAlloc, &buf_size.to_le_bytes(), 5).await;
    let data = assert_success(&payload);
    let output_ptr = u64::from_le_bytes(data[..8].try_into().unwrap());
    eprintln!("[REAL GPU] Output buffer at: 0x{output_ptr:016X} ({buf_size} bytes)");

    // ---------------------------------------------------------------
    // 5. Create input data [1.0, 2.0, 3.0, ..., 256.0] and upload
    // ---------------------------------------------------------------
    let input_data: Vec<f32> = (1..=n).map(|i| i as f32).collect();
    let input_bytes: Vec<u8> = input_data.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut htod_payload = input_ptr.to_le_bytes().to_vec();
    htod_payload.extend_from_slice(&input_bytes);
    let (_hdr, payload) =
        roundtrip(&client, MessageType::MemcpyHtoD, &htod_payload, 6).await;
    assert_success(&payload);
    eprintln!("[REAL GPU] Uploaded {n} floats to input buffer");

    // ---------------------------------------------------------------
    // 6. Launch kernel: grid=(N,1,1), block=(1,1,1), shared=0, stream=0
    // ---------------------------------------------------------------
    // Kernel params layout for cuLaunchKernel:
    //   param 0: u64 input_ptr  (CUdeviceptr, 8 bytes)
    //   param 1: u64 output_ptr (CUdeviceptr, 8 bytes)
    //   param 2: u32 n          (int, 4 bytes)
    //
    // Wire format after the 44-byte fixed header:
    //   [4B num_params: u32 LE]
    //   [4B size][size bytes]  -- repeated for each param
    //
    // The server's launch_kernel deserializes this into a Vec<Vec<u8>>,
    // then builds the void** array that cuLaunchKernel expects (each
    // pointer targets the raw bytes of one parameter).
    let mut launch_payload = kernel_func.to_le_bytes().to_vec(); // 8B func
    launch_payload.extend_from_slice(&1_u32.to_le_bytes());      // gridX = 1
    launch_payload.extend_from_slice(&1_u32.to_le_bytes());      // gridY
    launch_payload.extend_from_slice(&1_u32.to_le_bytes());      // gridZ
    launch_payload.extend_from_slice(&n.to_le_bytes());          // blockX = N (256 threads)
    launch_payload.extend_from_slice(&1_u32.to_le_bytes());      // blockY
    launch_payload.extend_from_slice(&1_u32.to_le_bytes());      // blockZ
    launch_payload.extend_from_slice(&0_u32.to_le_bytes());      // shared_mem
    launch_payload.extend_from_slice(&0_u64.to_le_bytes());      // stream (default)
    // Kernel params (serialized with size-prefix format):
    launch_payload.extend_from_slice(&3_u32.to_le_bytes());      // num_params = 3
    launch_payload.extend_from_slice(&8_u32.to_le_bytes());      // param 0 size = 8
    launch_payload.extend_from_slice(&input_ptr.to_le_bytes());  // param 0: input ptr
    launch_payload.extend_from_slice(&8_u32.to_le_bytes());      // param 1 size = 8
    launch_payload.extend_from_slice(&output_ptr.to_le_bytes()); // param 1: output ptr
    launch_payload.extend_from_slice(&4_u32.to_le_bytes());      // param 2 size = 4
    launch_payload.extend_from_slice(&n.to_le_bytes());          // param 2: n (u32)

    let (_hdr, payload) =
        roundtrip(&client, MessageType::LaunchKernel, &launch_payload, 7).await;
    let result = response_result(&payload);
    assert_eq!(
        result,
        CuResult::Success,
        "LaunchKernel failed with: {result:?}"
    );
    eprintln!("[REAL GPU] Kernel launched successfully!");

    // Synchronize to ensure kernel completes before reading back
    let (_hdr, payload) =
        roundtrip(&client, MessageType::CtxSynchronize, &[], 70).await;
    assert_eq!(response_result(&payload), CuResult::Success);
    eprintln!("[REAL GPU] Context synchronized");

    // ---------------------------------------------------------------
    // 7. Read back the output buffer
    // ---------------------------------------------------------------
    let mut dtoh_payload = output_ptr.to_le_bytes().to_vec();
    dtoh_payload.extend_from_slice(&buf_size.to_le_bytes());
    let (_hdr, payload) =
        roundtrip(&client, MessageType::MemcpyDtoH, &dtoh_payload, 8).await;
    let data = assert_success(&payload);

    assert_eq!(
        data.len(),
        buf_size as usize,
        "expected {buf_size} bytes back from GPU, got {}",
        data.len()
    );

    // ---------------------------------------------------------------
    // 8. Verify: output[i] == input[i] * 2.0
    // ---------------------------------------------------------------
    let output_floats: Vec<f32> = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    eprintln!("[REAL GPU] First 8 results: {:?}", &output_floats[..8]);
    eprintln!(
        "[REAL GPU] Last 8 results:  {:?}",
        &output_floats[output_floats.len() - 8..]
    );

    for (i, (&out, &inp)) in output_floats.iter().zip(input_data.iter()).enumerate() {
        let expected = inp * 2.0;
        assert!(
            (out - expected).abs() < f32::EPSILON,
            "MISMATCH at index {i}: expected {expected}, got {out}"
        );
    }
    eprintln!(
        "[REAL GPU] ALL {n} elements verified: output[i] == input[i] * 2.0  COMPUTE WORKS!"
    );

    // ---------------------------------------------------------------
    // 9. Cleanup: free buffers, unload module, destroy context
    // ---------------------------------------------------------------
    let (_hdr, payload) =
        roundtrip(&client, MessageType::MemFree, &input_ptr.to_le_bytes(), 9).await;
    assert_eq!(response_result(&payload), CuResult::Success);

    let (_hdr, payload) =
        roundtrip(&client, MessageType::MemFree, &output_ptr.to_le_bytes(), 10).await;
    assert_eq!(response_result(&payload), CuResult::Success);

    let (_hdr, payload) =
        roundtrip(&client, MessageType::ModuleUnload, &module.to_le_bytes(), 11).await;
    assert_eq!(response_result(&payload), CuResult::Success);

    let (_hdr, payload) =
        roundtrip(&client, MessageType::CtxDestroy, &ctx_handle.to_le_bytes(), 12).await;
    assert_eq!(response_result(&payload), CuResult::Success);

    eprintln!("[REAL GPU] Cleanup complete: buffers freed, module unloaded, context destroyed");

    drop(client);
    server.await.unwrap();
}
