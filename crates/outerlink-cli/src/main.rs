//! OuterLink CLI - Management Tool
//!
//! Command-line interface for managing the OuterLink GPU pool.
//! - List available GPUs across all nodes
//! - Check node status and connectivity
//! - Run benchmarks

use clap::{Parser, Subcommand};

use outerlink_cli::commands;

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

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Check server status
    Status {
        /// Server address
        #[arg(short, long, default_value = "localhost:14833")]
        server: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Run transfer benchmarks
    Bench {
        /// Server address
        #[arg(short, long, default_value = "localhost:14833")]
        server: String,

        /// Transfer size in MiB (if omitted, benchmarks 1K, 64K, 1M, 16M, 64M)
        #[arg(long)]
        size: Option<usize>,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Minimal logging -- only show warnings/errors (CLI output should be clean).
    tracing_subscriber::fmt()
        .with_env_filter("warn")
        .init();

    let result = run(cli).await;
    if let Err(e) = result {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

async fn run(cli: Cli) -> anyhow::Result<()> {
    match cli.command {
        Commands::List { server, json } => {
            let gpus = commands::cmd_list(&server).await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&gpus)?);
            } else {
                commands::print_list_table(&server, &gpus);
            }
        }
        Commands::Status { server, json } => {
            let status = commands::cmd_status(&server).await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&status)?);
            } else {
                commands::print_status(&server, &status);
            }
        }
        Commands::Bench { server, size, json } => {
            let custom_size = size.map(|s| s * 1024 * 1024);
            let result = commands::cmd_bench(&server, custom_size).await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                commands::print_bench(&server, &result);
            }
        }
    }
    Ok(())
}
