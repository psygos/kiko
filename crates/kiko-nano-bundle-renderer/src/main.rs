use std::path::PathBuf;

use clap::{Parser, Subcommand};
use kiko_nano_bundle_renderer::{RenderMode, render_bundle};

#[derive(Debug, Parser)]
#[command(about = "Build an offline Kiko Nano staging bundle; never installs it")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Check {
        #[arg(long)]
        input: PathBuf,
    },
    Stage {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        destination: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();
    let result = match &cli.command {
        Command::Check { input } => render_bundle(input, RenderMode::DryRun),
        Command::Stage { input, destination } => render_bundle(
            input,
            RenderMode::Stage {
                destination: destination.as_path(),
            },
        ),
    };
    match result {
        Ok(evidence) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&evidence)
                    .expect("bundle plan evidence serialization cannot fail")
            );
        }
        Err(error) => {
            eprintln!("kiko-nano-bundle-renderer: {error}");
            std::process::exit(1);
        }
    }
}
