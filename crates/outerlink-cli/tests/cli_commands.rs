//! Integration tests for CLI command logic.
//!
//! Each test starts a real TCP server with `StubGpuBackend` and exercises
//! the CLI command functions (list, status, bench) against it.

use std::sync::Arc;

use tokio::net::TcpListener;

use outerlink_common::error::OuterLinkError;
use outerlink_common::protocol::MessageType;
use outerlink_common::tcp_transport::TcpTransportConnection;
use outerlink_common::transport::TransportConnection;
use outerlink_server::gpu_backend::{GpuBackend, StubGpuBackend};
use outerlink_server::handler::handle_request;
use outerlink_server::session::ConnectionSession;

use outerlink_cli::commands::{cmd_bench, cmd_list, cmd_status};
use outerlink_cli::commands::{BenchResult, GpuInfo, StatusInfo};

// ---------------------------------------------------------------------------
// Helpers (same pattern as outerlink-server integration tests)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// list command tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_list_returns_gpu_info() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let server = spawn_server(listener, backend);

    let gpus = cmd_list(&addr).await.expect("cmd_list should succeed");

    assert_eq!(gpus.len(), 1, "stub backend has exactly 1 GPU");

    let gpu = &gpus[0];
    assert_eq!(gpu.id, 0);
    assert_eq!(gpu.name, "OuterLink Virtual GPU");
    assert_eq!(gpu.total_mem_bytes, 24 * 1024 * 1024 * 1024);
    assert_eq!(gpu.compute_major, 8);
    assert_eq!(gpu.compute_minor, 6);

    // Wait for server to finish
    // (cmd_list drops its connection internally, so server loop should exit)
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), server).await;
}

#[tokio::test]
async fn test_list_formats_vram_as_mib() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let _server = spawn_server(listener, backend);

    let gpus = cmd_list(&addr).await.unwrap();
    let gpu = &gpus[0];

    // 24 GiB = 24 * 1024 MiB = 24576 MiB
    let mib = gpu.total_mem_bytes / (1024 * 1024);
    assert_eq!(mib, 24576);
}

#[tokio::test]
async fn test_list_unreachable_server() {
    // Try to connect to a port that nobody is listening on.
    let result = cmd_list("127.0.0.1:1").await;
    assert!(result.is_err(), "connecting to unreachable server should fail");
}

// ---------------------------------------------------------------------------
// status command tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_status_returns_driver_version() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let _server = spawn_server(listener, backend);

    let status = cmd_status(&addr).await.expect("cmd_status should succeed");

    assert_eq!(status.driver_version, 12040);
    assert!(status.latency_ms > 0.0, "latency should be positive");
    assert_eq!(status.gpus.len(), 1);

    let gpu = &status.gpus[0];
    assert_eq!(gpu.name, "OuterLink Virtual GPU");
    assert!(gpu.free_bytes > 0, "free VRAM should be non-zero");
    assert_eq!(gpu.total_bytes, 24 * 1024 * 1024 * 1024);
    assert!(gpu.free_bytes <= gpu.total_bytes, "free should not exceed total");
}

#[tokio::test]
async fn test_status_unreachable_server() {
    let result = cmd_status("127.0.0.1:1").await;
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// bench command tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_bench_returns_results() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let _server = spawn_server(listener, backend);

    let result = cmd_bench(&addr, None).await.expect("cmd_bench should succeed");

    // Should have HtoD and DtoH results for default sizes
    assert!(!result.htod.is_empty(), "should have HtoD results");
    assert!(!result.dtoh.is_empty(), "should have DtoH results");
    assert_eq!(result.htod.len(), result.dtoh.len(), "htod and dtoh should have same number of entries");

    // Verify latency stats
    assert!(result.latency_avg_ms > 0.0, "avg latency should be positive");
    assert!(result.latency_p99_ms > 0.0, "p99 latency should be positive");
    assert!(result.latency_p99_ms >= result.latency_avg_ms, "p99 should be >= avg");
    assert_eq!(result.latency_samples, 100);
}

#[tokio::test]
async fn test_bench_custom_size() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let _server = spawn_server(listener, backend);

    let result = cmd_bench(&addr, Some(4 * 1024 * 1024)).await.expect("cmd_bench should succeed");

    // With a custom size, should have exactly 1 entry for HtoD and DtoH
    assert_eq!(result.htod.len(), 1, "custom size should produce exactly 1 HtoD entry");
    assert_eq!(result.dtoh.len(), 1, "custom size should produce exactly 1 DtoH entry");
    assert_eq!(result.htod[0].size_bytes, 4 * 1024 * 1024);
}

#[tokio::test]
async fn test_bench_unreachable_server() {
    let result = cmd_bench("127.0.0.1:1", None).await;
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// JSON serialization tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_gpu_info_serializes_to_json() {
    let info = GpuInfo {
        id: 0,
        name: "Test GPU".to_string(),
        total_mem_bytes: 24 * 1024 * 1024 * 1024,
        compute_major: 8,
        compute_minor: 6,
    };

    let json = serde_json::to_string(&info).expect("should serialize");
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed["id"], 0);
    assert_eq!(parsed["name"], "Test GPU");
    assert_eq!(parsed["compute_major"], 8);
}

#[tokio::test]
async fn test_status_info_serializes_to_json() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let _server = spawn_server(listener, backend);

    let status = cmd_status(&addr).await.unwrap();
    let json = serde_json::to_string(&status).expect("StatusInfo should serialize");
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed["driver_version"], 12040);
    assert!(parsed["gpus"].is_array());
}

#[tokio::test]
async fn test_bench_result_serializes_to_json() {
    let (listener, addr) = bind_server().await;
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let _server = spawn_server(listener, backend);

    let result = cmd_bench(&addr, Some(1024)).await.unwrap();
    let json = serde_json::to_string(&result).expect("BenchResult should serialize");
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(parsed["htod"].is_array());
    assert!(parsed["dtoh"].is_array());
    assert!(parsed["latency_avg_ms"].is_number());
}
