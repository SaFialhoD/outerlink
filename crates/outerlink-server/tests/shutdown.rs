//! Tests for graceful shutdown of the OuterLink server.
//!
//! Validates that:
//! - The server shuts down cleanly when the shutdown signal fires
//! - In-flight requests complete before shutdown
//! - GPU resources are cleaned up on shutdown

mod common;
use common::*;

use std::sync::Arc;
use std::time::Duration;

use tokio::net::TcpListener;

use outerlink_common::protocol::MessageType;
use outerlink_server::gpu_backend::{GpuBackend, StubGpuBackend};
use outerlink_server::server::Server;

// ---------------------------------------------------------------------------
// Helper: build a Server on a random port for testing
// ---------------------------------------------------------------------------

async fn test_server() -> (Server, String) {
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind failed");
    let addr = listener.local_addr().unwrap().to_string();
    let server = Server::new(listener, backend);
    (server, addr)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// The server exits cleanly when the shutdown signal fires with no clients.
#[tokio::test]
async fn test_shutdown_no_clients() {
    let (server, _addr) = test_server().await;
    let shutdown_tx = server.shutdown_handle();

    let handle = tokio::spawn(async move {
        server.run().await;
    });

    // Signal shutdown immediately.
    shutdown_tx.send(()).expect("send shutdown");

    // Server should exit promptly.
    let result = tokio::time::timeout(Duration::from_secs(3), handle).await;
    assert!(result.is_ok(), "server should exit within 3 seconds");
    result.unwrap().expect("server task should not panic");
}

/// In-flight requests complete before the connection handler exits.
/// After shutdown, the accept loop stops but existing connections finish
/// their current work before the drain phase begins.
#[tokio::test]
async fn test_shutdown_completes_inflight_request() {
    let (server, addr) = test_server().await;
    let shutdown_tx = server.shutdown_handle();

    let handle = tokio::spawn(async move {
        server.run().await;
    });

    // Connect a client and verify multiple requests work.
    let client = connect_client(&addr).await;
    let (_hdr, payload) = roundtrip(&client, MessageType::DeviceGetCount, &[], 1).await;
    let data = assert_success(&payload);
    let count = i32::from_le_bytes(data[..4].try_into().unwrap());
    assert_eq!(count, 1);

    let (_hdr2, payload2) = roundtrip(&client, MessageType::DriverGetVersion, &[], 2).await;
    let data2 = assert_success(&payload2);
    let version = i32::from_le_bytes(data2[..4].try_into().unwrap());
    assert_eq!(version, 12040);

    // Signal shutdown. The server stops accepting new connections and
    // existing connection handlers exit after their current request.
    shutdown_tx.send(()).expect("send shutdown");

    // Drop the client so the connection handler finishes.
    drop(client);

    // Server should exit within the drain timeout.
    let result = tokio::time::timeout(Duration::from_secs(5), handle).await;
    assert!(result.is_ok(), "server should exit within 5 seconds after client disconnects");
    result.unwrap().expect("server task should not panic");
}

/// After shutdown, new connections are NOT accepted.
#[tokio::test]
async fn test_shutdown_rejects_new_connections() {
    let (server, addr) = test_server().await;
    let shutdown_tx = server.shutdown_handle();

    let handle = tokio::spawn(async move {
        server.run().await;
    });

    // Signal shutdown.
    shutdown_tx.send(()).expect("send shutdown");

    // Wait for the server to actually stop.
    let result = tokio::time::timeout(Duration::from_secs(3), handle).await;
    assert!(result.is_ok(), "server should exit within 3 seconds");

    // A new connection attempt should fail (nobody listening).
    let connect_result = tokio::time::timeout(
        Duration::from_secs(1),
        tokio::net::TcpStream::connect(&addr),
    )
    .await;
    // Either times out or gets connection refused -- both mean no server.
    match connect_result {
        Err(_timeout) => {} // timed out -- OK, server is gone
        Ok(Err(_)) => {}    // connection refused -- OK
        Ok(Ok(_)) => panic!("should not be able to connect after shutdown"),
    }
}

/// GPU backend cleanup is called on shutdown.
#[tokio::test]
async fn test_shutdown_calls_backend_cleanup() {
    // Create a backend, allocate some resources, then shut down.
    let backend = Arc::new(StubGpuBackend::new());
    backend.init();

    // Allocate a context and some memory to verify cleanup.
    let ctx = backend.ctx_create(0, 0).expect("ctx_create");
    let _ptr = backend.mem_alloc(1024).expect("mem_alloc");
    assert!(backend.ctx_exists(ctx));

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind failed");

    let backend_clone: Arc<dyn GpuBackend> = backend.clone();
    let server = Server::new(listener, backend_clone);
    let shutdown_tx = server.shutdown_handle();

    let handle = tokio::spawn(async move {
        server.run().await;
    });

    // Signal shutdown.
    shutdown_tx.send(()).expect("send shutdown");

    let result = tokio::time::timeout(Duration::from_secs(3), handle).await;
    assert!(result.is_ok(), "server should exit within 3 seconds");

    // After shutdown, the backend's shutdown() should have been called,
    // clearing all contexts and allocations.
    assert!(
        !backend.ctx_exists(ctx),
        "context should be destroyed after shutdown"
    );
}

/// Server with a drain timeout: connections that do not close in time
/// are dropped.
#[tokio::test]
async fn test_shutdown_drain_timeout() {
    let (server, addr) = test_server().await;
    let shutdown_tx = server.shutdown_handle();

    let handle = tokio::spawn(async move {
        server.run().await;
    });

    // Connect a client but do NOT close it.
    let client = connect_client(&addr).await;

    // Send one request to make sure the connection is established.
    let (_hdr, payload) = roundtrip(&client, MessageType::DeviceGetCount, &[], 1).await;
    assert_success(&payload);

    // Signal shutdown. The server has a drain timeout (5s by default).
    shutdown_tx.send(()).expect("send shutdown");

    // The server should exit within drain_timeout + some slack, even though
    // the client is still connected.
    let result = tokio::time::timeout(Duration::from_secs(8), handle).await;
    assert!(
        result.is_ok(),
        "server should exit within drain timeout even with lingering client"
    );
}
