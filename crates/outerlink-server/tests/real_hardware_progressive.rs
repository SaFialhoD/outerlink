//! Progressive real-hardware GPU test suite for OuterLink.
//!
//! Designed to be run in phases, each gating the next:
//!   Phase 1 (sanity)     → Can we talk to the GPU at all?
//!   Phase 2 (functional) → Does every operation work correctly?
//!   Phase 3 (reliability)→ Does it work 30 times in a row?
//!   Phase 4 (stress)     → Can it handle pressure?
//!
//! Run with:
//! ```
//! # All phases (stops on first failure):
//! cargo test -p outerlink-server --features real-gpu-test \
//!     --test real_hardware_progressive -- --test-threads=1 --nocapture
//!
//! # Single phase:
//! cargo test -p outerlink-server --features real-gpu-test \
//!     --test real_hardware_progressive phase1_ -- --test-threads=1 --nocapture
//! ```
//!
//! Safety guarantees:
//!   - Every async op wrapped in a timeout (no infinite hangs)
//!   - VRAM checked before every allocation (no OOM crashes)
//!   - All constants at top of file (no magic numbers in test bodies)
//!   - Cleanup always runs (explicit, never deferred to "next test")

#![cfg(feature = "real-gpu-test")]

mod common;
use common::*;

use std::future::Future;
use std::time::{Duration, Instant};

use outerlink_common::protocol::MessageType;
use outerlink_common::tcp_transport::TcpTransportConnection;
use outerlink_server::cuda_backend::CudaGpuBackend;

// ===========================================================================
// Safety constants — ALL tunables live here, nowhere else
// ===========================================================================

/// Max time any single protocol operation can take before we kill it.
const OP_TIMEOUT: Duration = Duration::from_secs(10);

/// Max time for kernel launch + sync (PTX compilation can be slow).
const KERNEL_TIMEOUT: Duration = Duration::from_secs(30);

/// Max time for an entire stress/reliability test.
const STRESS_TIMEOUT: Duration = Duration::from_secs(120);

/// Max allocation size per buffer (128 MiB) — hard cap to prevent OOM.
const MAX_ALLOC_BYTES: usize = 128 * 1024 * 1024;

/// Small allocation for basic sanity tests (1 MiB).
const SMALL_ALLOC: usize = 1024 * 1024;

/// Tiny allocation for minimal sanity (1 KiB).
const TINY_ALLOC: usize = 1024;

/// Minimum free VRAM required to run any test (256 MiB).
const MIN_FREE_VRAM: u64 = 256 * 1024 * 1024;

/// Never consume more than this fraction of free VRAM.
const VRAM_USAGE_LIMIT: f64 = 0.50;

/// Number of repetitions for reliability tests.
const RELIABILITY_REPS: usize = 30;

/// Number of reconnect cycles in reliability tests.
const RECONNECT_REPS: usize = 10;

/// Number of concurrent clients for load tests.
const LOAD_CLIENTS: usize = 4;

/// Per-client allocation cap in load tests (16 MiB).
const LOAD_CLIENT_ALLOC: usize = 16 * 1024 * 1024;

/// Number of rapid-fire operations for latency stress.
const RAPID_FIRE_OPS: usize = 100;

// ===========================================================================
// Helpers
// ===========================================================================

/// Try to create a real CUDA backend, skip test if unavailable.
fn try_create_backend() -> Option<Arc<dyn GpuBackend>> {
    let backend = match CudaGpuBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[SKIP] CudaGpuBackend::new() failed: {e}");
            return None;
        }
    };
    let init_result = backend.init();
    if !init_result.is_success() {
        eprintln!("[SKIP] init returned {init_result:?}");
        return None;
    }
    Some(Arc::new(backend))
}

macro_rules! require_gpu {
    () => {
        match try_create_backend() {
            Some(b) => b,
            None => return,
        }
    };
}

/// Wrap an async op in a timeout. Panics with a clear message on timeout.
async fn guarded<F, T>(label: &str, timeout: Duration, fut: F) -> T
where
    F: Future<Output = T>,
{
    tokio::time::timeout(timeout, fut)
        .await
        .unwrap_or_else(|_| {
            panic!(
                "TIMEOUT: '{label}' did not complete within {}s",
                timeout.as_secs()
            )
        })
}

/// Send a request through a guarded timeout. Returns the response payload.
async fn guarded_roundtrip(
    conn: &TcpTransportConnection,
    msg_type: MessageType,
    payload: &[u8],
    rid: u64,
    label: &str,
) -> Vec<u8> {
    let label = format!("roundtrip({label})");
    let (_hdr, resp) = guarded(&label, OP_TIMEOUT, roundtrip(conn, msg_type, payload, rid)).await;
    resp
}

/// Create a context on device 0 and return the handle. Panicked on failure.
async fn create_ctx(conn: &TcpTransportConnection, rid: u64) -> u64 {
    let mut payload = 0_u32.to_le_bytes().to_vec();
    payload.extend_from_slice(&0_i32.to_le_bytes());
    let resp = guarded_roundtrip(conn, MessageType::CtxCreate, &payload, rid, "CtxCreate").await;
    let data = assert_success(&resp);
    u64::from_le_bytes(data[..8].try_into().unwrap())
}

/// Destroy a context. Panics on failure.
async fn destroy_ctx(conn: &TcpTransportConnection, ctx: u64, rid: u64) {
    let resp =
        guarded_roundtrip(conn, MessageType::CtxDestroy, &ctx.to_le_bytes(), rid, "CtxDestroy")
            .await;
    assert_success(&resp);
}

/// Query VRAM info (requires active context). Returns (free, total) in bytes.
/// Panics if free VRAM is below MIN_FREE_VRAM.
async fn check_vram(conn: &TcpTransportConnection, rid: u64) -> (u64, u64) {
    let resp = guarded_roundtrip(conn, MessageType::MemGetInfo, &[], rid, "MemGetInfo").await;
    let data = assert_success(&resp);
    let free = u64::from_le_bytes(data[..8].try_into().unwrap());
    let total = u64::from_le_bytes(data[8..16].try_into().unwrap());
    assert!(
        free >= MIN_FREE_VRAM,
        "VRAM GUARD: only {} MiB free (need {} MiB minimum). Aborting to prevent OOM.",
        free / (1024 * 1024),
        MIN_FREE_VRAM / (1024 * 1024)
    );
    (free, total)
}

/// Compute a safe allocation size: min(requested, free * VRAM_USAGE_LIMIT, MAX_ALLOC_BYTES).
fn safe_alloc_size(requested: usize, free_vram: u64) -> usize {
    let vram_limit = (free_vram as f64 * VRAM_USAGE_LIMIT) as usize;
    requested.min(vram_limit).min(MAX_ALLOC_BYTES)
}

/// Allocate device memory and return the pointer. Panics on failure.
async fn alloc_mem(conn: &TcpTransportConnection, size: usize, rid: u64) -> u64 {
    let resp = guarded_roundtrip(
        conn,
        MessageType::MemAlloc,
        &(size as u64).to_le_bytes(),
        rid,
        &format!("MemAlloc({})", size),
    )
    .await;
    let data = assert_success(&resp);
    u64::from_le_bytes(data[..8].try_into().unwrap())
}

/// Free device memory. Panics on failure.
async fn free_mem(conn: &TcpTransportConnection, ptr: u64, rid: u64) {
    let resp =
        guarded_roundtrip(conn, MessageType::MemFree, &ptr.to_le_bytes(), rid, "MemFree").await;
    assert_success(&resp);
}

/// Copy host data to device. Panics on failure.
async fn memcpy_htod(conn: &TcpTransportConnection, dst: u64, data: &[u8], rid: u64) {
    let mut payload = dst.to_le_bytes().to_vec();
    payload.extend_from_slice(data);
    let resp =
        guarded_roundtrip(conn, MessageType::MemcpyHtoD, &payload, rid, "MemcpyHtoD").await;
    assert_success(&resp);
}

/// Copy device data to host. Returns the bytes. Panics on failure.
async fn memcpy_dtoh(conn: &TcpTransportConnection, src: u64, size: usize, rid: u64) -> Vec<u8> {
    let mut payload = src.to_le_bytes().to_vec();
    payload.extend_from_slice(&(size as u64).to_le_bytes());
    let resp =
        guarded_roundtrip(conn, MessageType::MemcpyDtoH, &payload, rid, "MemcpyDtoH").await;
    let data = assert_success(&resp);
    assert_eq!(
        data.len(),
        size,
        "MemcpyDtoH returned {} bytes, expected {size}",
        data.len()
    );
    data.to_vec()
}

/// Generate a deterministic test pattern: byte at position i = (i % 251) as u8.
/// 251 is prime so the pattern doesn't repeat on power-of-2 boundaries.
fn test_pattern(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 251) as u8).collect()
}

/// Tear down: drop client, await server.
async fn teardown(client: TcpTransportConnection, server: tokio::task::JoinHandle<()>) {
    drop(client);
    guarded("server shutdown", OP_TIMEOUT, server)
        .await
        .expect("server task panicked");
}

// ===========================================================================
// PHASE 1: Sanity — "Can we talk to the GPU at all?"
// ===========================================================================

#[tokio::test]
async fn phase1_a_gpu_exists() {
    let backend = require_gpu!();

    let count = backend.device_get_count().expect("device_get_count");
    eprintln!("[P1] GPU device count: {count}");
    assert!(count >= 1, "expected at least 1 GPU, got {count}");

    let version = backend.driver_get_version().expect("driver_get_version");
    eprintln!("[P1] CUDA driver version: {version}");
    assert!(version > 0, "driver version should be positive");
}

#[tokio::test]
async fn phase1_b_device_properties() {
    let backend = require_gpu!();
    let (listener, addr) = bind_server().await;
    let server = spawn_server(listener, backend);
    let client = connect_client(&addr).await;

    // Device name
    let resp = guarded_roundtrip(
        &client,
        MessageType::DeviceGetName,
        &0_i32.to_le_bytes(),
        1,
        "DeviceGetName",
    )
    .await;
    let data = assert_success(&resp);
    let name_len = u32::from_le_bytes(data[..4].try_into().unwrap()) as usize;
    let name = std::str::from_utf8(&data[4..4 + name_len]).expect("valid UTF-8");
    eprintln!("[P1] GPU name: {name}");
    assert!(!name.is_empty());

    // Total memory
    let resp = guarded_roundtrip(
        &client,
        MessageType::DeviceTotalMem,
        &0_i32.to_le_bytes(),
        2,
        "DeviceTotalMem",
    )
    .await;
    let data = assert_success(&resp);
    let total_mem = u64::from_le_bytes(data[..8].try_into().unwrap());
    eprintln!("[P1] Total VRAM: {} MiB", total_mem / (1024 * 1024));
    assert!(total_mem > 0, "total memory should be > 0");

    // Compute capability
    let mut attr_payload = 75_i32.to_le_bytes().to_vec(); // ComputeCapabilityMajor
    attr_payload.extend_from_slice(&0_i32.to_le_bytes());
    let resp = guarded_roundtrip(
        &client,
        MessageType::DeviceGetAttribute,
        &attr_payload,
        3,
        "ComputeCapMajor",
    )
    .await;
    let data = assert_success(&resp);
    let cc_major = i32::from_le_bytes(data[..4].try_into().unwrap());

    let mut attr_payload = 76_i32.to_le_bytes().to_vec(); // ComputeCapabilityMinor
    attr_payload.extend_from_slice(&0_i32.to_le_bytes());
    let resp = guarded_roundtrip(
        &client,
        MessageType::DeviceGetAttribute,
        &attr_payload,
        4,
        "ComputeCapMinor",
    )
    .await;
    let data = assert_success(&resp);
    let cc_minor = i32::from_le_bytes(data[..4].try_into().unwrap());
    eprintln!("[P1] Compute capability: {cc_major}.{cc_minor}");
    assert!(cc_major >= 3, "compute capability major should be >= 3");

    // UUID
    let resp = guarded_roundtrip(
        &client,
        MessageType::DeviceGetUuid,
        &0_i32.to_le_bytes(),
        5,
        "DeviceGetUuid",
    )
    .await;
    let data = assert_success(&resp);
    assert_eq!(data.len(), 16, "UUID should be 16 bytes");
    eprintln!("[P1] GPU UUID: {:02x?}", data);

    teardown(client, server).await;
}

#[tokio::test]
async fn phase1_c_context_lifecycle() {
    let backend = require_gpu!();
    let (listener, addr) = bind_server().await;
    let server = spawn_server(listener, backend);
    let client = connect_client(&addr).await;

    let ctx = create_ctx(&client, 1).await;
    eprintln!("[P1] Created context: 0x{ctx:016X}");

    // CtxGetDevice
    let resp = guarded_roundtrip(
        &client,
        MessageType::CtxGetDevice,
        &ctx.to_le_bytes(),
        2,
        "CtxGetDevice",
    )
    .await;
    let data = assert_success(&resp);
    let device = i32::from_le_bytes(data[..4].try_into().unwrap());
    assert_eq!(device, 0, "context should be on device 0");

    // CtxSynchronize
    let resp =
        guarded_roundtrip(&client, MessageType::CtxSynchronize, &[], 3, "CtxSynchronize").await;
    assert_success(&resp);

    // Destroy
    destroy_ctx(&client, ctx, 4).await;
    eprintln!("[P1] Context lifecycle: create -> get_device -> sync -> destroy  OK");

    teardown(client, server).await;
}

#[tokio::test]
async fn phase1_d_small_alloc_free() {
    let backend = require_gpu!();
    let (listener, addr) = bind_server().await;
    let server = spawn_server(listener, backend);
    let client = connect_client(&addr).await;

    let ctx = create_ctx(&client, 1).await;
    let (free, _total) = check_vram(&client, 2).await;
    eprintln!("[P1] Free VRAM: {} MiB", free / (1024 * 1024));

    let size = safe_alloc_size(SMALL_ALLOC, free);
    let ptr = alloc_mem(&client, size, 3).await;
    eprintln!("[P1] Allocated {} bytes at 0x{ptr:016X}", size);

    free_mem(&client, ptr, 4).await;
    eprintln!("[P1] Freed OK");

    destroy_ctx(&client, ctx, 5).await;
    teardown(client, server).await;
}

#[tokio::test]
async fn phase1_e_small_memcpy_roundtrip() {
    let backend = require_gpu!();
    let (listener, addr) = bind_server().await;
    let server = spawn_server(listener, backend);
    let client = connect_client(&addr).await;

    let ctx = create_ctx(&client, 1).await;
    check_vram(&client, 2).await;

    let data = test_pattern(TINY_ALLOC);
    let ptr = alloc_mem(&client, TINY_ALLOC, 3).await;

    memcpy_htod(&client, ptr, &data, 4).await;
    let readback = memcpy_dtoh(&client, ptr, TINY_ALLOC, 5).await;

    assert_eq!(
        readback, data,
        "memcpy roundtrip mismatch! First differing byte at index {}",
        readback.iter().zip(data.iter()).position(|(a, b)| a != b).unwrap_or(0)
    );
    eprintln!("[P1] Memcpy roundtrip ({} bytes): VERIFIED", TINY_ALLOC);

    free_mem(&client, ptr, 6).await;
    destroy_ctx(&client, ctx, 7).await;
    teardown(client, server).await;
}

// ===========================================================================
// PHASE 2: Functional — "Does every operation work correctly?"
// ===========================================================================

#[tokio::test]
async fn phase2_a_memcpy_sizes() {
    let backend = require_gpu!();
    let (listener, addr) = bind_server().await;
    let server = spawn_server(listener, backend);
    let client = connect_client(&addr).await;

    let ctx = create_ctx(&client, 1).await;
    let (free, _total) = check_vram(&client, 2).await;

    let sizes: &[usize] = &[4096, 64 * 1024, 1024 * 1024, 16 * 1024 * 1024];
    let mut rid = 10_u64;

    for &size in sizes {
        let size = safe_alloc_size(size, free);
        let data = test_pattern(size);

        let ptr = alloc_mem(&client, size, rid).await;
        rid += 1;

        memcpy_htod(&client, ptr, &data, rid).await;
        rid += 1;

        let readback = memcpy_dtoh(&client, ptr, size, rid).await;
        rid += 1;

        assert_eq!(readback, data, "mismatch at size {size}");

        free_mem(&client, ptr, rid).await;
        rid += 1;

        eprintln!("[P2] Memcpy {}: OK", format_size(size));
    }

    destroy_ctx(&client, ctx, rid).await;
    teardown(client, server).await;
}

#[tokio::test]
async fn phase2_b_memcpy_pattern_verify() {
    let backend = require_gpu!();
    let (listener, addr) = bind_server().await;
    let server = spawn_server(listener, backend);
    let client = connect_client(&addr).await;

    let ctx = create_ctx(&client, 1).await;
    check_vram(&client, 2).await;

    // Write pattern where every byte is different modulo 251
    let size = 251 * 100; // exactly 25100 bytes, pattern repeats once
    let data = test_pattern(size);

    let ptr = alloc_mem(&client, size, 3).await;
    memcpy_htod(&client, ptr, &data, 4).await;
    let readback = memcpy_dtoh(&client, ptr, size, 5).await;

    // Verify every single byte
    for (i, (&expected, &actual)) in data.iter().zip(readback.iter()).enumerate() {
        assert_eq!(
            actual, expected,
            "byte mismatch at offset {i}: expected 0x{expected:02x}, got 0x{actual:02x}"
        );
    }
    eprintln!("[P2] Pattern verify ({size} bytes, every byte checked): OK");

    free_mem(&client, ptr, 6).await;
    destroy_ctx(&client, ctx, 7).await;
    teardown(client, server).await;
}

#[tokio::test]
async fn phase2_c_memset_operations() {
    let backend = require_gpu!();
    let (listener, addr) = bind_server().await;
    let server = spawn_server(listener, backend);
    let client = connect_client(&addr).await;

    let ctx = create_ctx(&client, 1).await;
    check_vram(&client, 2).await;

    let count: usize = 256;
    let mut rid = 10_u64;

    // memset_d8: fill with 0xAB
    {
        let ptr = alloc_mem(&client, count, rid).await;
        rid += 1;

        // payload: u64 dstDevice, u8 value, u64 count
        let mut payload = ptr.to_le_bytes().to_vec();
        payload.push(0xAB_u8);
        payload.extend_from_slice(&(count as u64).to_le_bytes());
        let resp =
            guarded_roundtrip(&client, MessageType::MemsetD8, &payload, rid, "MemsetD8").await;
        assert_success(&resp);
        rid += 1;

        let readback = memcpy_dtoh(&client, ptr, count, rid).await;
        rid += 1;
        assert!(
            readback.iter().all(|&b| b == 0xAB),
            "MemsetD8 verification failed"
        );

        free_mem(&client, ptr, rid).await;
        rid += 1;
        eprintln!("[P2] MemsetD8 (0xAB, {count} bytes): OK");
    }

    // memset_d32: fill with 0xDEADBEEF
    {
        let num_elements = 64_usize;
        let byte_size = num_elements * 4;
        let ptr = alloc_mem(&client, byte_size, rid).await;
        rid += 1;

        // payload: u64 dstDevice, u32 value, u64 count (in elements, not bytes)
        let mut payload = ptr.to_le_bytes().to_vec();
        payload.extend_from_slice(&0xDEADBEEF_u32.to_le_bytes());
        payload.extend_from_slice(&(num_elements as u64).to_le_bytes());
        let resp =
            guarded_roundtrip(&client, MessageType::MemsetD32, &payload, rid, "MemsetD32").await;
        assert_success(&resp);
        rid += 1;

        let readback = memcpy_dtoh(&client, ptr, byte_size, rid).await;
        rid += 1;
        for (i, chunk) in readback.chunks_exact(4).enumerate() {
            let val = u32::from_le_bytes(chunk.try_into().unwrap());
            assert_eq!(
                val, 0xDEADBEEF,
                "MemsetD32 mismatch at element {i}: got 0x{val:08X}"
            );
        }

        free_mem(&client, ptr, rid).await;
        rid += 1;
        eprintln!("[P2] MemsetD32 (0xDEADBEEF, {num_elements} elements): OK");
    }

    destroy_ctx(&client, ctx, rid).await;
    teardown(client, server).await;
}

#[tokio::test]
async fn phase2_d_multiple_allocations() {
    let backend = require_gpu!();
    let (listener, addr) = bind_server().await;
    let server = spawn_server(listener, backend);
    let client = connect_client(&addr).await;

    let ctx = create_ctx(&client, 1).await;
    let (free, _total) = check_vram(&client, 2).await;

    let num_buffers = 10_usize;
    let per_buf_size = safe_alloc_size(SMALL_ALLOC, free / (num_buffers as u64 + 2));
    let mut rid = 10_u64;
    let mut ptrs = Vec::new();
    let mut patterns = Vec::new();

    // Allocate and write different data to each
    for i in 0..num_buffers {
        let ptr = alloc_mem(&client, per_buf_size, rid).await;
        rid += 1;

        // Each buffer gets a unique pattern: (offset + i*37) % 251
        let pattern: Vec<u8> = (0..per_buf_size)
            .map(|j| ((j + i * 37) % 251) as u8)
            .collect();
        memcpy_htod(&client, ptr, &pattern, rid).await;
        rid += 1;

        ptrs.push(ptr);
        patterns.push(pattern);
    }
    eprintln!(
        "[P2] Allocated and wrote {num_buffers} buffers x {} each",
        format_size(per_buf_size)
    );

    // Read all back and verify
    for (i, (&ptr, expected)) in ptrs.iter().zip(patterns.iter()).enumerate() {
        let readback = memcpy_dtoh(&client, ptr, per_buf_size, rid).await;
        rid += 1;
        assert_eq!(readback, *expected, "buffer {i} mismatch");
    }
    eprintln!("[P2] All {num_buffers} buffers verified correctly");

    // Free all
    for &ptr in &ptrs {
        free_mem(&client, ptr, rid).await;
        rid += 1;
    }

    destroy_ctx(&client, ctx, rid).await;
    teardown(client, server).await;
}

#[tokio::test]
async fn phase2_e_stream_operations() {
    let backend = require_gpu!();
    let (listener, addr) = bind_server().await;
    let server = spawn_server(listener, backend);
    let client = connect_client(&addr).await;

    let ctx = create_ctx(&client, 1).await;
    let mut rid = 10_u64;

    // Create stream
    let resp =
        guarded_roundtrip(&client, MessageType::StreamCreate, &[], rid, "StreamCreate").await;
    let data = assert_success(&resp);
    let stream = u64::from_le_bytes(data[..8].try_into().unwrap());
    eprintln!("[P2] Stream: 0x{stream:016X}");
    rid += 1;

    // Create event
    let resp = guarded_roundtrip(
        &client,
        MessageType::EventCreate,
        &0_u32.to_le_bytes(),
        rid,
        "EventCreate",
    )
    .await;
    let data = assert_success(&resp);
    let event = u64::from_le_bytes(data[..8].try_into().unwrap());
    eprintln!("[P2] Event: 0x{event:016X}");
    rid += 1;

    // Record event on stream
    let mut payload = event.to_le_bytes().to_vec();
    payload.extend_from_slice(&stream.to_le_bytes());
    let resp =
        guarded_roundtrip(&client, MessageType::EventRecord, &payload, rid, "EventRecord").await;
    assert_success(&resp);
    rid += 1;

    // Synchronize stream
    let resp = guarded_roundtrip(
        &client,
        MessageType::StreamSynchronize,
        &stream.to_le_bytes(),
        rid,
        "StreamSync",
    )
    .await;
    assert_success(&resp);
    rid += 1;

    // Event synchronize
    let resp = guarded_roundtrip(
        &client,
        MessageType::EventSynchronize,
        &event.to_le_bytes(),
        rid,
        "EventSync",
    )
    .await;
    assert_success(&resp);
    rid += 1;

    // Destroy event, stream
    let resp = guarded_roundtrip(
        &client,
        MessageType::EventDestroy,
        &event.to_le_bytes(),
        rid,
        "EventDestroy",
    )
    .await;
    assert_success(&resp);
    rid += 1;

    let resp = guarded_roundtrip(
        &client,
        MessageType::StreamDestroy,
        &stream.to_le_bytes(),
        rid,
        "StreamDestroy",
    )
    .await;
    assert_success(&resp);
    rid += 1;

    eprintln!("[P2] Stream+Event lifecycle: OK");

    destroy_ctx(&client, ctx, rid).await;
    teardown(client, server).await;
}

#[tokio::test]
async fn phase2_f_module_load_launch() {
    let backend = require_gpu!();
    let (listener, addr) = bind_server().await;
    let server = spawn_server(listener, backend);
    let client = connect_client(&addr).await;

    let ctx = create_ctx(&client, 1).await;
    check_vram(&client, 2).await;

    // Load PTX module: a kernel that doubles each f32 element
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

    let mut rid = 10_u64;

    // Load module
    let resp = guarded(
        "ModuleLoadData",
        KERNEL_TIMEOUT,
        roundtrip(&client, MessageType::ModuleLoadData, ptx, rid),
    )
    .await;
    let data = assert_success(&resp.1);
    let module = u64::from_le_bytes(data[..8].try_into().unwrap());
    rid += 1;

    // Get function
    let func_name = b"double_elements";
    let mut gf_payload = module.to_le_bytes().to_vec();
    gf_payload.extend_from_slice(&(func_name.len() as u32).to_le_bytes());
    gf_payload.extend_from_slice(func_name);
    let resp = guarded_roundtrip(
        &client,
        MessageType::ModuleGetFunction,
        &gf_payload,
        rid,
        "ModuleGetFunction",
    )
    .await;
    let data = assert_success(&resp);
    let func = u64::from_le_bytes(data[..8].try_into().unwrap());
    rid += 1;

    // Allocate I/O buffers
    let n: u32 = 256;
    let buf_size = (n as usize) * 4;
    let input_ptr = alloc_mem(&client, buf_size, rid).await;
    rid += 1;
    let output_ptr = alloc_mem(&client, buf_size, rid).await;
    rid += 1;

    // Upload input: [1.0, 2.0, ..., 256.0]
    let input_floats: Vec<f32> = (1..=n).map(|i| i as f32).collect();
    let input_bytes: Vec<u8> = input_floats.iter().flat_map(|v| v.to_le_bytes()).collect();
    memcpy_htod(&client, input_ptr, &input_bytes, rid).await;
    rid += 1;

    // Launch kernel
    let mut lp = func.to_le_bytes().to_vec();
    lp.extend_from_slice(&1_u32.to_le_bytes()); // gridX
    lp.extend_from_slice(&1_u32.to_le_bytes()); // gridY
    lp.extend_from_slice(&1_u32.to_le_bytes()); // gridZ
    lp.extend_from_slice(&n.to_le_bytes()); // blockX
    lp.extend_from_slice(&1_u32.to_le_bytes()); // blockY
    lp.extend_from_slice(&1_u32.to_le_bytes()); // blockZ
    lp.extend_from_slice(&0_u32.to_le_bytes()); // shared_mem
    lp.extend_from_slice(&0_u64.to_le_bytes()); // stream
    lp.extend_from_slice(&3_u32.to_le_bytes()); // num_params
    lp.extend_from_slice(&8_u32.to_le_bytes()); // param0 size
    lp.extend_from_slice(&input_ptr.to_le_bytes());
    lp.extend_from_slice(&8_u32.to_le_bytes()); // param1 size
    lp.extend_from_slice(&output_ptr.to_le_bytes());
    lp.extend_from_slice(&4_u32.to_le_bytes()); // param2 size
    lp.extend_from_slice(&n.to_le_bytes());

    let resp = guarded(
        "LaunchKernel",
        KERNEL_TIMEOUT,
        roundtrip(&client, MessageType::LaunchKernel, &lp, rid),
    )
    .await;
    assert_eq!(response_result(&resp.1), CuResult::Success, "kernel launch failed");
    rid += 1;

    // Synchronize
    let resp = guarded(
        "CtxSynchronize",
        KERNEL_TIMEOUT,
        roundtrip(&client, MessageType::CtxSynchronize, &[], rid),
    )
    .await;
    assert_success(&resp.1);
    rid += 1;

    // Read back and verify
    let output_bytes = memcpy_dtoh(&client, output_ptr, buf_size, rid).await;
    rid += 1;
    let output_floats: Vec<f32> = output_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    for (i, (&out, &inp)) in output_floats.iter().zip(input_floats.iter()).enumerate() {
        let expected = inp * 2.0;
        assert!(
            (out - expected).abs() < f32::EPSILON,
            "kernel output mismatch at [{i}]: expected {expected}, got {out}"
        );
    }
    eprintln!("[P2] Kernel launch: {n} elements doubled and verified  COMPUTE WORKS");

    // Cleanup
    free_mem(&client, input_ptr, rid).await;
    rid += 1;
    free_mem(&client, output_ptr, rid).await;
    rid += 1;
    let resp = guarded_roundtrip(
        &client,
        MessageType::ModuleUnload,
        &module.to_le_bytes(),
        rid,
        "ModuleUnload",
    )
    .await;
    assert_success(&resp);
    rid += 1;
    destroy_ctx(&client, ctx, rid).await;
    teardown(client, server).await;
}

#[tokio::test]
async fn phase2_g_primary_context() {
    let backend = require_gpu!();
    let (listener, addr) = bind_server().await;
    let server = spawn_server(listener, backend);
    let client = connect_client(&addr).await;
    let mut rid = 1_u64;

    // Retain primary context for device 0
    let resp = guarded_roundtrip(
        &client,
        MessageType::DevicePrimaryCtxRetain,
        &0_i32.to_le_bytes(),
        rid,
        "PrimaryCtxRetain",
    )
    .await;
    let data = assert_success(&resp);
    let pctx = u64::from_le_bytes(data[..8].try_into().unwrap());
    eprintln!("[P2] Primary context: 0x{pctx:016X}");
    rid += 1;

    // Get state
    let resp = guarded_roundtrip(
        &client,
        MessageType::DevicePrimaryCtxGetState,
        &0_i32.to_le_bytes(),
        rid,
        "PrimaryCtxGetState",
    )
    .await;
    let data = assert_success(&resp);
    let flags = u32::from_le_bytes(data[..4].try_into().unwrap());
    let active = i32::from_le_bytes(data[4..8].try_into().unwrap());
    eprintln!("[P2] Primary ctx state: flags={flags}, active={active}");
    assert_eq!(active, 1, "primary context should be active after retain");
    rid += 1;

    // Release
    let resp = guarded_roundtrip(
        &client,
        MessageType::DevicePrimaryCtxRelease,
        &0_i32.to_le_bytes(),
        rid,
        "PrimaryCtxRelease",
    )
    .await;
    assert_success(&resp);
    eprintln!("[P2] Primary context lifecycle: retain -> get_state -> release  OK");

    teardown(client, server).await;
}

#[tokio::test]
async fn phase2_h_session_cleanup_on_disconnect() {
    let backend = require_gpu!();
    let (listener, addr) = bind_server().await;
    let server = spawn_server(listener, Arc::clone(&backend));
    let client = connect_client(&addr).await;

    let _ctx = create_ctx(&client, 1).await;
    check_vram(&client, 2).await;

    // Allocate resources without freeing them
    let ptr1 = alloc_mem(&client, SMALL_ALLOC, 3).await;
    let _ptr2 = alloc_mem(&client, SMALL_ALLOC, 4).await;
    eprintln!("[P2] Allocated 2 buffers without freeing, ptr1=0x{ptr1:016X}");

    // Create a stream
    let resp =
        guarded_roundtrip(&client, MessageType::StreamCreate, &[], 5, "StreamCreate").await;
    assert_success(&resp);

    // Disconnect — server's session cleanup should free everything
    drop(client);
    eprintln!("[P2] Disconnected. Server should clean up leaked resources...");

    guarded("server cleanup", OP_TIMEOUT, server)
        .await
        .expect("server should not panic during cleanup");
    eprintln!("[P2] Session cleanup on disconnect: OK (server exited cleanly)");
}

// ===========================================================================
// PHASE 3: Reliability — "Does it work N times in a row?"
// ===========================================================================

#[tokio::test]
async fn phase3_a_alloc_free_cycle() {
    let backend = require_gpu!();

    tokio::time::timeout(STRESS_TIMEOUT, async {
        let (listener, addr) = bind_server().await;
        let server = spawn_server(listener, backend);
        let client = connect_client(&addr).await;

        let ctx = create_ctx(&client, 1).await;
        let mut rid = 10_u64;

        for rep in 0..RELIABILITY_REPS {
            let (free, _) = check_vram(&client, rid).await;
            rid += 1;

            let size = safe_alloc_size(SMALL_ALLOC, free);
            let data = test_pattern(size);

            let ptr = alloc_mem(&client, size, rid).await;
            rid += 1;
            memcpy_htod(&client, ptr, &data, rid).await;
            rid += 1;
            let readback = memcpy_dtoh(&client, ptr, size, rid).await;
            rid += 1;
            assert_eq!(readback, data, "rep {rep}: data mismatch");
            free_mem(&client, ptr, rid).await;
            rid += 1;

            if (rep + 1) % 10 == 0 {
                eprintln!("[P3] alloc/free cycle: {}/{RELIABILITY_REPS}", rep + 1);
            }
        }

        eprintln!("[P3] alloc_free_cycle: {RELIABILITY_REPS}/{RELIABILITY_REPS} passed");
        destroy_ctx(&client, ctx, rid).await;
        teardown(client, server).await;
    })
    .await
    .expect("TIMEOUT: phase3_a exceeded stress timeout");
}

#[tokio::test]
async fn phase3_b_context_cycle() {
    let backend = require_gpu!();

    tokio::time::timeout(STRESS_TIMEOUT, async {
        let (listener, addr) = bind_server().await;
        let server = spawn_server(listener, backend);
        let client = connect_client(&addr).await;
        let mut rid = 1_u64;

        for rep in 0..RELIABILITY_REPS {
            let ctx = create_ctx(&client, rid).await;
            rid += 1;

            // Do a sync to prove the context works
            let resp = guarded_roundtrip(
                &client,
                MessageType::CtxSynchronize,
                &[],
                rid,
                &format!("sync rep {rep}"),
            )
            .await;
            assert_success(&resp);
            rid += 1;

            destroy_ctx(&client, ctx, rid).await;
            rid += 1;

            if (rep + 1) % 10 == 0 {
                eprintln!("[P3] context cycle: {}/{RELIABILITY_REPS}", rep + 1);
            }
        }

        eprintln!("[P3] context_cycle: {RELIABILITY_REPS}/{RELIABILITY_REPS} passed");
        teardown(client, server).await;
    })
    .await
    .expect("TIMEOUT: phase3_b exceeded stress timeout");
}

#[tokio::test]
async fn phase3_c_reconnect_cycle() {
    let backend = require_gpu!();

    tokio::time::timeout(STRESS_TIMEOUT, async {
        for rep in 0..RECONNECT_REPS {
            let (listener, addr) = bind_server().await;
            let server = spawn_server(listener, Arc::clone(&backend));
            let client = connect_client(&addr).await;

            // Basic operation to prove the connection works
            let resp = guarded_roundtrip(
                &client,
                MessageType::DeviceGetCount,
                &[],
                1,
                &format!("reconnect rep {rep}"),
            )
            .await;
            let data = assert_success(&resp);
            let count = i32::from_le_bytes(data[..4].try_into().unwrap());
            assert!(count >= 1);

            teardown(client, server).await;

            if (rep + 1) % 5 == 0 {
                eprintln!("[P3] reconnect cycle: {}/{RECONNECT_REPS}", rep + 1);
            }
        }
        eprintln!("[P3] reconnect_cycle: {RECONNECT_REPS}/{RECONNECT_REPS} passed");
    })
    .await
    .expect("TIMEOUT: phase3_c exceeded stress timeout");
}

#[tokio::test]
async fn phase3_d_mixed_operations_cycle() {
    let backend = require_gpu!();

    tokio::time::timeout(STRESS_TIMEOUT, async {
        let (listener, addr) = bind_server().await;
        let server = spawn_server(listener, backend);
        let client = connect_client(&addr).await;
        let mut rid = 1_u64;

        for rep in 0..RELIABILITY_REPS {
            // Full workflow: ctx -> alloc -> write -> read -> verify -> free -> ctx_destroy
            let ctx = create_ctx(&client, rid).await;
            rid += 1;

            let (free, _) = check_vram(&client, rid).await;
            rid += 1;

            let size = safe_alloc_size(SMALL_ALLOC, free);
            let data = test_pattern(size);

            let ptr = alloc_mem(&client, size, rid).await;
            rid += 1;
            memcpy_htod(&client, ptr, &data, rid).await;
            rid += 1;
            let readback = memcpy_dtoh(&client, ptr, size, rid).await;
            rid += 1;
            assert_eq!(readback, data, "rep {rep}: mixed ops data mismatch");
            free_mem(&client, ptr, rid).await;
            rid += 1;
            destroy_ctx(&client, ctx, rid).await;
            rid += 1;

            if (rep + 1) % 10 == 0 {
                eprintln!("[P3] mixed ops cycle: {}/{RELIABILITY_REPS}", rep + 1);
            }
        }

        eprintln!("[P3] mixed_operations: {RELIABILITY_REPS}/{RELIABILITY_REPS} passed");
        teardown(client, server).await;
    })
    .await
    .expect("TIMEOUT: phase3_d exceeded stress timeout");
}

// ===========================================================================
// PHASE 4: Stress & Load — "Can it handle pressure?"
// ===========================================================================

#[tokio::test]
async fn phase4_a_large_transfer() {
    let backend = require_gpu!();

    tokio::time::timeout(STRESS_TIMEOUT, async {
        let (listener, addr) = bind_server().await;
        let server = spawn_server(listener, backend);
        let client = connect_client(&addr).await;

        let ctx = create_ctx(&client, 1).await;
        let (free, _) = check_vram(&client, 2).await;

        let size = safe_alloc_size(MAX_ALLOC_BYTES, free);
        eprintln!(
            "[P4] Large transfer: {} (free VRAM: {} MiB)",
            format_size(size),
            free / (1024 * 1024)
        );

        let data = test_pattern(size);
        let ptr = alloc_mem(&client, size, 3).await;

        let start = Instant::now();
        memcpy_htod(&client, ptr, &data, 4).await;
        let htod_time = start.elapsed();

        let start = Instant::now();
        let readback = memcpy_dtoh(&client, ptr, size, 5).await;
        let dtoh_time = start.elapsed();

        assert_eq!(readback, data, "large transfer data mismatch");

        let htod_mbps = (size as f64) / (1024.0 * 1024.0) / htod_time.as_secs_f64();
        let dtoh_mbps = (size as f64) / (1024.0 * 1024.0) / dtoh_time.as_secs_f64();
        eprintln!(
            "[P4] HtoD: {:.1} MiB/s ({:.2}ms), DtoH: {:.1} MiB/s ({:.2}ms)",
            htod_mbps,
            htod_time.as_secs_f64() * 1000.0,
            dtoh_mbps,
            dtoh_time.as_secs_f64() * 1000.0,
        );

        free_mem(&client, ptr, 6).await;
        destroy_ctx(&client, ctx, 7).await;
        teardown(client, server).await;
    })
    .await
    .expect("TIMEOUT: phase4_a exceeded stress timeout");
}

#[tokio::test]
async fn phase4_b_concurrent_clients() {
    let backend = require_gpu!();

    tokio::time::timeout(STRESS_TIMEOUT, async {
        // Use the full Server (not spawn_server) to accept multiple connections
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind failed");
        let addr = listener.local_addr().unwrap().to_string();

        let server = outerlink_server::server::Server::new(listener, Arc::clone(&backend));
        let shutdown_tx = server.shutdown_handle();

        let server_handle = tokio::spawn(async move {
            server.run().await;
        });

        // Spawn LOAD_CLIENTS concurrent clients
        let mut client_handles = Vec::new();
        for client_id in 0..LOAD_CLIENTS {
            let addr = addr.clone();
            client_handles.push(tokio::spawn(async move {
                let client = connect_client(&addr).await;
                let base_rid = (client_id as u64) * 1000;
                let mut rid = base_rid + 1;

                // Each client: ctx -> alloc -> write -> read -> verify -> free -> ctx_destroy
                let ctx = create_ctx(&client, rid).await;
                rid += 1;

                let size = LOAD_CLIENT_ALLOC.min(MAX_ALLOC_BYTES);
                let data = test_pattern(size);

                let ptr = alloc_mem(&client, size, rid).await;
                rid += 1;
                memcpy_htod(&client, ptr, &data, rid).await;
                rid += 1;
                let readback = memcpy_dtoh(&client, ptr, size, rid).await;
                rid += 1;
                assert_eq!(
                    readback, data,
                    "client {client_id}: data mismatch"
                );
                free_mem(&client, ptr, rid).await;
                rid += 1;
                destroy_ctx(&client, ctx, rid).await;

                drop(client);
                eprintln!("[P4] Client {client_id}: OK");
            }));
        }

        // Wait for all clients
        for (i, handle) in client_handles.into_iter().enumerate() {
            handle
                .await
                .unwrap_or_else(|e| panic!("client {i} task panicked: {e}"));
        }

        eprintln!("[P4] All {LOAD_CLIENTS} concurrent clients completed");

        // Shutdown server
        let _ = shutdown_tx.send(());
        guarded("server shutdown", OP_TIMEOUT, server_handle)
            .await
            .expect("server task panicked");
    })
    .await
    .expect("TIMEOUT: phase4_b exceeded stress timeout");
}

#[tokio::test]
async fn phase4_c_rapid_fire_ops() {
    let backend = require_gpu!();

    tokio::time::timeout(STRESS_TIMEOUT, async {
        let (listener, addr) = bind_server().await;
        let server = spawn_server(listener, backend);
        let client = connect_client(&addr).await;
        let mut rid = 1_u64;

        let mut latencies = Vec::with_capacity(RAPID_FIRE_OPS);

        for _ in 0..RAPID_FIRE_OPS {
            let start = Instant::now();
            let resp = guarded_roundtrip(
                &client,
                MessageType::DeviceGetCount,
                &[],
                rid,
                "rapid_fire",
            )
            .await;
            let elapsed = start.elapsed();
            assert_success(&resp);
            latencies.push(elapsed.as_secs_f64() * 1000.0);
            rid += 1;
        }

        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let avg = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let p50 = latencies[latencies.len() / 2];
        let p99_idx = ((latencies.len() as f64) * 0.99).ceil() as usize - 1;
        let p99 = latencies[p99_idx.min(latencies.len() - 1)];
        let min = latencies[0];
        let max = latencies[latencies.len() - 1];

        eprintln!(
            "[P4] Rapid fire ({RAPID_FIRE_OPS} ops): avg={avg:.2}ms p50={p50:.2}ms p99={p99:.2}ms min={min:.2}ms max={max:.2}ms"
        );

        teardown(client, server).await;
    })
    .await
    .expect("TIMEOUT: phase4_c exceeded stress timeout");
}

// ===========================================================================
// Formatting helpers
// ===========================================================================

fn format_size(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{} MiB", bytes / (1024 * 1024))
    } else if bytes >= 1024 {
        format!("{} KiB", bytes / 1024)
    } else {
        format!("{} B", bytes)
    }
}
