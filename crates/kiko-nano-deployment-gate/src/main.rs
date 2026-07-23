#![cfg(unix)]
#![forbid(unsafe_code)]

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use kiko_nano_deployment_gate::{
    DEFAULT_QUALIFICATION_MARKER, ROOT_GID, ROOT_UID, verify_qualification_marker,
};

#[derive(Debug, Parser)]
#[command(
    name = "kiko-nano-deployment-gate",
    about = "Verify one root-owned, exact-byte Nano offline-install qualification"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Verify the marker and every exact installed byte without device I/O.
    Verify {
        #[arg(long, default_value = DEFAULT_QUALIFICATION_MARKER)]
        marker: PathBuf,
    },
}

fn main() {
    let Cli { command } = Cli::parse();
    let result = match command {
        Command::Verify { marker } => {
            verify_qualification_marker(&marker, ROOT_UID, ROOT_GID).map(|_| {
                eprintln!(
                    "offline install qualification verified; no hardware qualification is implied"
                );
            })
        }
    };
    if let Err(source) = result {
        eprintln!("{source}");
        std::process::exit(1);
    }
}
