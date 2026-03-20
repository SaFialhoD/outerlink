//! OuterLink Server - GPU Node Daemon
//!
//! Listens for incoming connections from OuterLink clients,
//! receives serialized CUDA operations, executes them on the
//! local GPU(s), and returns results.

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "outerlink-server", about = "OuterLink GPU node daemon")]
struct Args {
    /// Address to listen on
    #[arg(short, long, default_value = "0.0.0.0:14833")]
    listen: String,

    /// Config file path
    #[arg(short, long, default_value = "outerlink.toml")]
    config: String,

    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Initialize tracing
    let filter = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .init();

    tracing::info!("OuterLink Server starting on {}", args.listen);
    tracing::info!("OpenDMA: non-proprietary GPU direct access");

    // TODO: Initialize CUDA via libloading
    // TODO: Start TCP listener
    // TODO: Accept connections and dispatch requests

    tracing::info!("Server skeleton ready - implementation coming in Phase 1 (P5)");

    Ok(())
}
