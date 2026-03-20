//! OuterLink Server - GPU Node Daemon
//!
//! Listens for incoming connections from OuterLink clients,
//! receives serialised CUDA operations, executes them on the
//! local GPU(s) via the [`GpuBackend`] trait, and returns results.

use std::sync::Arc;

use clap::Parser;
use tokio::net::TcpListener;

use outerlink_common::error::OuterLinkError;
use outerlink_common::tcp_transport::TcpTransportConnection;
use outerlink_common::transport::TransportConnection;
use outerlink_server::gpu_backend::{GpuBackend, StubGpuBackend};
use outerlink_server::handler::handle_request;
use outerlink_server::session::ConnectionSession;

/// Command-line arguments for the server.
#[derive(Parser, Debug)]
#[command(name = "outerlink-server", about = "OuterLink GPU node daemon")]
struct Args {
    /// Address to listen on (ip:port).
    #[arg(short, long, default_value = "0.0.0.0:14833")]
    listen: String,

    // TODO: Add config file support (--config outerlink.toml) when needed.

    /// Enable verbose (debug-level) logging.
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Initialise tracing.
    let filter = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .init();

    tracing::info!("OuterLink Server starting on {}", args.listen);
    tracing::info!("OpenDMA: non-proprietary GPU direct access");

    // Create the GPU backend.  For the PoC we always use the stub.
    let backend: Arc<dyn GpuBackend> = Arc::new(StubGpuBackend::new());
    let init_result = backend.init();
    if !init_result.is_success() {
        anyhow::bail!("GPU backend init failed: {:?}", init_result);
    }
    tracing::info!("GPU backend initialised (stub mode)");

    // Bind the TCP listener.
    let listener = TcpListener::bind(&args.listen).await?;
    tracing::info!("Listening on {}", listener.local_addr()?);

    // Accept loop.
    loop {
        let (stream, peer) = match listener.accept().await {
            Ok(conn) => conn,
            Err(e) => {
                // Transient accept errors (fd exhaustion, connection aborted before
                // accept completes, etc.) should not kill the daemon.
                tracing::warn!(error = %e, "accept failed, retrying");
                continue;
            }
        };
        tracing::info!(%peer, "new connection");

        let backend = Arc::clone(&backend);
        tokio::spawn(async move {
            let conn = match TcpTransportConnection::new(stream) {
                Ok(c) => c,
                Err(e) => {
                    tracing::error!(%peer, error = %e, "failed to initialise transport");
                    return;
                }
            };
            if let Err(e) = handle_connection(conn, backend).await {
                tracing::error!(%peer, error = %e, "connection handler error");
            }
            tracing::info!(%peer, "connection closed");
        });
    }
}

/// Drive a single client connection to completion.
///
/// Reads messages in a loop via [`TcpTransportConnection`] (which provides
/// TCP_NODELAY, magic/version validation, and payload-size bounds checking),
/// dispatches each to [`handle_request`], and writes the response back.
/// Returns when the client disconnects or a fatal I/O error occurs.
async fn handle_connection(
    conn: TcpTransportConnection,
    backend: Arc<dyn GpuBackend>,
) -> anyhow::Result<()> {
    // Each connection gets its own session so per-thread state (like the
    // current CUDA context) is isolated between clients.
    let mut session = ConnectionSession::new();

    loop {
        // 1. Receive the next framed message (header + payload).
        let (header, payload) = match conn.recv_message().await {
            Ok(msg) => msg,
            Err(OuterLinkError::ConnectionClosed) => {
                // Client closed the connection gracefully.
                return Ok(());
            }
            Err(e) => {
                return Err(anyhow::anyhow!(e));
            }
        };

        tracing::debug!(
            request_id = header.request_id,
            msg_type = ?header.msg_type,
            payload_len = header.payload_len,
            "received request"
        );

        // 2. Dispatch.
        let (resp_header, resp_payload) =
            handle_request(&*backend, &header, &payload, &mut session);

        // 3. Write the response.
        conn.send_message(&resp_header, &resp_payload).await?;
    }
}
