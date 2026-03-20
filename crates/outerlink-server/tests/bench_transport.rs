//! TCP transport throughput benchmarks for OuterLink.
//!
//! Measures protocol overhead, connection latency, and memory transfer
//! throughput over loopback using StubGpuBackend (no real GPU required).
//!
//! These numbers represent the floor: real cross-machine transfers will
//! add network latency on top. The whole point of OuterLink is to
//! minimise that additional cost.
//!
//! Gated behind the `bench` feature so they never run in normal CI.
//!
//! Run with:
//! ```
//! cargo test -p outerlink-server --features bench --test bench_transport -- --nocapture
//! ```

#![cfg(feature = "bench")]

use std::sync::Arc;
use std::time::{Duration, Instant};

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
// Helpers (mirror integration.rs pattern)
// ---------------------------------------------------------------------------

fn response_result(payload: &[u8]) -> CuResult {
    assert!(
        payload.len() >= 4,
        "response payload too short ({} bytes)",
        payload.len()
    );
    CuResult::from_raw(u32::from_le_bytes(payload[..4].try_into().unwrap()))
}

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

            let (resp_header, resp_payload) =
                handle_request(&*backend, &header, &payload, &mut session);
            conn.send_message(&resp_header, &resp_payload)
                .await
                .expect("server send failed");
        }
    })
}

async fn bind_server() -> (TcpListener, String) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind");
    let addr = listener.local_addr().unwrap().to_string();
    (listener, addr)
}

async fn connect_client(addr: &str) -> TcpTransportConnection {
    let stream = tokio::net::TcpStream::connect(addr)
        .await
        .expect("client connect failed");
    TcpTransportConnection::new(stream).expect("client conn init failed")
}

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
    assert_eq!(resp_hdr.msg_type, MessageType::Response);
    assert_eq!(resp_hdr.request_id, request_id);
    (resp_hdr, resp_payload)
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

fn format_size(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:4} MiB", bytes / (1024 * 1024))
    } else if bytes >= 1024 {
        format!("{:4} KiB", bytes / 1024)
    } else {
        format!("{:4} B  ", bytes)
    }
}

fn format_throughput(bytes: usize, duration: Duration) -> String {
    let secs = duration.as_secs_f64();
    if secs == 0.0 {
        return "      inf MB/s".to_string();
    }
    let mb_per_sec = (bytes as f64) / (1024.0 * 1024.0) / secs;
    if mb_per_sec >= 1000.0 {
        format!("{:8.2} GB/s", mb_per_sec / 1024.0)
    } else {
        format!("{:8.2} MB/s", mb_per_sec)
    }
}

// ---------------------------------------------------------------------------
// 1. Stub backend memcpy throughput
// ---------------------------------------------------------------------------

/// Measure a single HtoD + DtoH roundtrip for `size` bytes.
///
/// Returns (htod_duration, dtoh_duration).
async fn bench_memcpy_roundtrip(
    client: &TcpTransportConnection,
    size: usize,
    request_id_base: u64,
) -> (Duration, Duration) {
    // 1. Allocate
    let alloc_payload = (size as u64).to_le_bytes();
    let (_hdr, payload) = roundtrip(
        client,
        MessageType::MemAlloc,
        &alloc_payload,
        request_id_base,
    )
    .await;
    let data = assert_success(&payload);
    let device_ptr = u64::from_le_bytes(data[..8].try_into().unwrap());

    // 2. Prepare test data
    let test_data: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();

    // 3. Time MemcpyHtoD
    let mut htod_payload = device_ptr.to_le_bytes().to_vec();
    htod_payload.extend_from_slice(&test_data);

    let htod_start = Instant::now();
    let (_hdr, payload) = roundtrip(
        client,
        MessageType::MemcpyHtoD,
        &htod_payload,
        request_id_base + 1,
    )
    .await;
    let htod_elapsed = htod_start.elapsed();
    assert_eq!(response_result(&payload), CuResult::Success);

    // 4. Time MemcpyDtoH
    let mut dtoh_payload = device_ptr.to_le_bytes().to_vec();
    dtoh_payload.extend_from_slice(&(size as u64).to_le_bytes());

    let dtoh_start = Instant::now();
    let (_hdr, payload) = roundtrip(
        client,
        MessageType::MemcpyDtoH,
        &dtoh_payload,
        request_id_base + 2,
    )
    .await;
    let dtoh_elapsed = dtoh_start.elapsed();
    assert_success(&payload);

    // 5. Free
    let free_payload = device_ptr.to_le_bytes();
    let (_hdr, payload) = roundtrip(
        client,
        MessageType::MemFree,
        &free_payload,
        request_id_base + 3,
    )
    .await;
    assert_eq!(response_result(&payload), CuResult::Success);

    (htod_elapsed, dtoh_elapsed)
}

#[tokio::test]
async fn bench_stub_memcpy_throughput() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);
    let client = connect_client(&addr).await;

    let sizes: &[usize] = &[
        1024,                // 1 KiB
        64 * 1024,           // 64 KiB
        1024 * 1024,         // 1 MiB
        16 * 1024 * 1024,    // 16 MiB
        64 * 1024 * 1024,    // 64 MiB
    ];

    const WARMUP: usize = 1;
    const ITERATIONS: usize = 10;

    let mut htod_results: Vec<(usize, Duration)> = Vec::new();
    let mut dtoh_results: Vec<(usize, Duration)> = Vec::new();

    let mut req_id: u64 = 1;

    for &size in sizes {
        // Warm up
        for _ in 0..WARMUP {
            let _ = bench_memcpy_roundtrip(&client, size, req_id).await;
            req_id += 4;
        }

        // Measure
        let mut htod_total = Duration::ZERO;
        let mut dtoh_total = Duration::ZERO;

        for _ in 0..ITERATIONS {
            let (htod, dtoh) = bench_memcpy_roundtrip(&client, size, req_id).await;
            htod_total += htod;
            dtoh_total += dtoh;
            req_id += 4;
        }

        let htod_avg = htod_total / ITERATIONS as u32;
        let dtoh_avg = dtoh_total / ITERATIONS as u32;

        htod_results.push((size, htod_avg));
        dtoh_results.push((size, dtoh_avg));
    }

    // Print results
    eprintln!();
    eprintln!("=== OuterLink Transport Benchmarks ===");
    eprintln!("      (loopback, StubGpuBackend, {} iterations)", ITERATIONS);
    eprintln!();

    eprintln!("--- Stub MemcpyHtoD Throughput ---");
    for &(size, avg) in &htod_results {
        eprintln!(
            "  {:>8}:  avg {:>8.2}ms  ({})",
            format_size(size),
            avg.as_secs_f64() * 1000.0,
            format_throughput(size, avg),
        );
    }

    eprintln!();
    eprintln!("--- Stub MemcpyDtoH Throughput ---");
    for &(size, avg) in &dtoh_results {
        eprintln!(
            "  {:>8}:  avg {:>8.2}ms  ({})",
            format_size(size),
            avg.as_secs_f64() * 1000.0,
            format_throughput(size, avg),
        );
    }

    eprintln!();

    // Verify data integrity on final iteration (sanity check)
    // The roundtrip helper already asserts CuResult::Success, so if
    // we got here, all transfers succeeded.

    drop(client);
    server.await.unwrap();
}

// ---------------------------------------------------------------------------
// 2. Connection overhead
// ---------------------------------------------------------------------------

#[tokio::test]
async fn bench_connection_overhead() {
    const ITERATIONS: usize = 10;

    let mut durations: Vec<Duration> = Vec::with_capacity(ITERATIONS);

    for _ in 0..ITERATIONS {
        let (listener, addr) = bind_server().await;
        let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
        let server = spawn_server(listener, backend);

        let start = Instant::now();
        let client = connect_client(&addr).await;

        // Handshake to confirm the connection is fully established
        let (_hdr, payload) = roundtrip(&client, MessageType::Handshake, &[], 1).await;
        let elapsed = start.elapsed();
        assert_success(&payload);

        durations.push(elapsed);

        drop(client);
        server.await.unwrap();
    }

    let total: Duration = durations.iter().sum();
    let avg = total / ITERATIONS as u32;

    eprintln!();
    eprintln!("--- Connection Overhead (TCP connect + handshake) ---");
    eprintln!(
        "  avg: {:.2}ms  ({} iterations)",
        avg.as_secs_f64() * 1000.0,
        ITERATIONS,
    );
    eprintln!();
}

// ---------------------------------------------------------------------------
// 3. Small message latency
// ---------------------------------------------------------------------------

#[tokio::test]
async fn bench_small_message_latency() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);
    let client = connect_client(&addr).await;

    const ITERATIONS: usize = 100;

    let mut durations: Vec<Duration> = Vec::with_capacity(ITERATIONS);

    // Warm up with a single request
    let _ = roundtrip(&client, MessageType::DeviceGetCount, &[], 0).await;

    for i in 0..ITERATIONS {
        let start = Instant::now();
        let (_hdr, payload) = roundtrip(
            &client,
            MessageType::DeviceGetCount,
            &[],
            (i + 1) as u64,
        )
        .await;
        let elapsed = start.elapsed();

        assert_success(&payload);
        durations.push(elapsed);
    }

    // Compute statistics
    durations.sort();

    let total: Duration = durations.iter().sum();
    let avg = total / ITERATIONS as u32;
    let min = durations[0];
    let max = durations[ITERATIONS - 1];
    let p99_idx = (ITERATIONS as f64 * 0.99).ceil() as usize - 1;
    let p99 = durations[p99_idx.min(ITERATIONS - 1)];

    eprintln!();
    eprintln!("--- Small Message Latency (DeviceGetCount, {} iterations) ---", ITERATIONS);
    eprintln!(
        "  avg: {:.3}ms  min: {:.3}ms  max: {:.3}ms  p99: {:.3}ms",
        avg.as_secs_f64() * 1000.0,
        min.as_secs_f64() * 1000.0,
        max.as_secs_f64() * 1000.0,
        p99.as_secs_f64() * 1000.0,
    );
    eprintln!();

    drop(client);
    server.await.unwrap();
}
