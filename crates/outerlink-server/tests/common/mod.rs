//! Shared test helpers for OuterLink server integration tests.
//!
//! Used by `integration.rs`, `bench_transport.rs`, and `real_gpu_test.rs`
//! via `mod common;` + `use common::*;`.

use std::sync::Arc;

use tokio::net::TcpListener;

use outerlink_common::cuda_types::CuResult;
use outerlink_common::error::OuterLinkError;
use outerlink_common::protocol::{MessageHeader, MessageType};
use outerlink_common::tcp_transport::TcpTransportConnection;
use outerlink_common::transport::TransportConnection;
use outerlink_server::gpu_backend::GpuBackend;
use outerlink_server::handler::handle_request;
use outerlink_server::session::ConnectionSession;

/// Extract the `CuResult` from the first 4 bytes of a response payload.
#[allow(dead_code)]
pub fn response_result(payload: &[u8]) -> CuResult {
    assert!(
        payload.len() >= 4,
        "response payload too short ({} bytes)",
        payload.len()
    );
    CuResult::from_raw(u32::from_le_bytes(payload[..4].try_into().unwrap()))
}

/// Assert the response indicates success and return the data portion (after
/// the 4-byte `CuResult` prefix).
#[allow(dead_code)]
pub fn assert_success(payload: &[u8]) -> &[u8] {
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
#[allow(dead_code)]
pub fn spawn_server(
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
#[allow(dead_code)]
pub async fn bind_server() -> (TcpListener, String) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind");
    let addr = listener.local_addr().unwrap().to_string();
    (listener, addr)
}

/// Connect a raw `TcpTransportConnection` to the given address.
#[allow(dead_code)]
pub async fn connect_client(addr: &str) -> TcpTransportConnection {
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
#[allow(dead_code)]
pub async fn roundtrip(
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
