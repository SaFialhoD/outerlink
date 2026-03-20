//! OuterLink Server - GPU Node Daemon
//!
//! Listens for incoming connections from OuterLink clients,
//! receives serialised CUDA operations, executes them on the
//! local GPU(s) via the [`GpuBackend`] trait, and returns results.

use std::sync::Arc;

use clap::Parser;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use outerlink_common::protocol::{MessageHeader, HEADER_SIZE};
use outerlink_server::gpu_backend::{GpuBackend, StubGpuBackend};
use outerlink_server::handler::handle_request;

/// Command-line arguments for the server.
#[derive(Parser, Debug)]
#[command(name = "outerlink-server", about = "OuterLink GPU node daemon")]
struct Args {
    /// Address to listen on (ip:port).
    #[arg(short, long, default_value = "0.0.0.0:14833")]
    listen: String,

    /// Path to the configuration file.
    #[arg(short, long, default_value = "outerlink.toml")]
    config: String,

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
        let (stream, peer) = listener.accept().await?;
        tracing::info!(%peer, "new connection");

        let backend = Arc::clone(&backend);
        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, backend).await {
                tracing::error!(%peer, error = %e, "connection handler error");
            }
            tracing::info!(%peer, "connection closed");
        });
    }
}

/// Drive a single client connection to completion.
///
/// Reads messages in a loop, dispatches each to [`handle_request`],
/// and writes the response back. Returns when the client disconnects
/// or a fatal I/O error occurs.
async fn handle_connection(
    mut stream: tokio::net::TcpStream,
    backend: Arc<dyn GpuBackend>,
) -> anyhow::Result<()> {
    loop {
        // 1. Read the fixed-size header.
        let mut hdr_buf = [0u8; HEADER_SIZE];
        match stream.read_exact(&mut hdr_buf).await {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                // Client closed the connection gracefully.
                return Ok(());
            }
            Err(e) => return Err(e.into()),
        }

        let header = match MessageHeader::from_bytes(&hdr_buf) {
            Some(h) => h,
            None => {
                tracing::warn!("invalid header, closing connection");
                return Ok(());
            }
        };

        tracing::debug!(
            request_id = header.request_id,
            msg_type = ?header.msg_type,
            payload_len = header.payload_len,
            "received request"
        );

        // 2. Read the payload.
        let mut payload = vec![0u8; header.payload_len as usize];
        if !payload.is_empty() {
            stream.read_exact(&mut payload).await?;
        }

        // 3. Dispatch.
        let (resp_header, resp_payload) = handle_request(&*backend, &header, &payload);

        // 4. Write the response.
        let resp_hdr_bytes = resp_header.to_bytes();
        stream.write_all(&resp_hdr_bytes).await?;
        if !resp_payload.is_empty() {
            stream.write_all(&resp_payload).await?;
        }
        stream.flush().await?;
    }
}
