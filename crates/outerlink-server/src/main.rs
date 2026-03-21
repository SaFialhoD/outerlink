//! OuterLink Server - GPU Node Daemon
//!
//! Listens for incoming connections from OuterLink clients,
//! receives serialised CUDA operations, executes them on the
//! local GPU(s) via the [`GpuBackend`] trait, and returns results.

use std::sync::Arc;

use clap::Parser;
use tokio::net::TcpListener;

use outerlink_server::cuda_backend::CudaGpuBackend;
use outerlink_server::gpu_backend::{GpuBackend, StubGpuBackend};
use outerlink_server::server::Server;

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

    /// Use the real CUDA GPU backend (requires NVIDIA driver).
    /// When not set, the stub backend is used (no GPU required).
    #[arg(long)]
    real_gpu: bool,
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

    // Create the GPU backend.
    let backend: Arc<dyn GpuBackend> = if args.real_gpu {
        tracing::info!("Loading real CUDA GPU backend...");
        let cuda = CudaGpuBackend::new()?;
        Arc::new(cuda)
    } else {
        Arc::new(StubGpuBackend::new())
    };
    let init_result = backend.init();
    if !init_result.is_success() {
        anyhow::bail!("GPU backend init failed: {:?}", init_result);
    }
    let mode = if args.real_gpu { "real GPU" } else { "stub" };
    tracing::info!("GPU backend initialised ({mode} mode)");

    // Bind the TCP listener.
    let listener = TcpListener::bind(&args.listen).await?;
    tracing::info!("Listening on {}", listener.local_addr()?);

    // Build the server with graceful shutdown support.
    let server = Server::new(listener, backend);
    let shutdown_tx = server.shutdown_handle();

    // Spawn a task that listens for Ctrl+C (SIGINT) or SIGTERM and
    // fires the shutdown signal.
    tokio::spawn(async move {
        let ctrl_c = tokio::signal::ctrl_c();

        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            let mut sigterm =
                signal(SignalKind::terminate()).expect("failed to register SIGTERM handler");
            tokio::select! {
                _ = ctrl_c => {
                    tracing::info!("received Ctrl+C, initiating shutdown");
                }
                _ = sigterm.recv() => {
                    tracing::info!("received SIGTERM, initiating shutdown");
                }
            }
        }

        #[cfg(not(unix))]
        {
            ctrl_c.await.expect("failed to listen for Ctrl+C");
            tracing::info!("received Ctrl+C, initiating shutdown");
        }

        let _ = shutdown_tx.send(());
    });

    // Run the server (blocks until shutdown completes).
    server.run().await;

    tracing::info!("OuterLink Server exited");
    Ok(())
}
