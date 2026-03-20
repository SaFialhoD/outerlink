//! OuterLink CLI - Management Tool
//!
//! Command-line interface for managing the OuterLink GPU pool.
//! - List available GPUs across all nodes
//! - Check node status and connectivity
//! - Run benchmarks
//! - View pool statistics

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "outerlink", about = "OuterLink GPU pool management")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// List all GPUs in the pool
    List {
        /// Server address
        #[arg(short, long, default_value = "localhost:14833")]
        server: String,
    },

    /// Check server status
    Status {
        /// Server address
        #[arg(short, long, default_value = "localhost:14833")]
        server: String,
    },

    /// Run bandwidth benchmark
    Bench {
        /// Server address
        #[arg(short, long, default_value = "localhost:14833")]
        server: String,

        /// Transfer size in bytes
        #[arg(long, default_value = "1048576")]
        size: usize,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    match cli.command {
        Commands::List { server } => {
            println!("Listing GPUs on {server}...");
            // TODO: Connect to server, query GPU list
            println!("(not yet implemented - see P5 PoC plan)");
        }
        Commands::Status { server } => {
            println!("Checking status of {server}...");
            // TODO: Connect and check health
            println!("(not yet implemented)");
        }
        Commands::Bench { server, size } => {
            println!("Running benchmark against {server} with {size} byte transfers...");
            // TODO: Connect and run transfer benchmark
            println!("(not yet implemented - see P12 benchmarking plan)");
        }
    }

    Ok(())
}
