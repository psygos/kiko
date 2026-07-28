//! Offline-only review and publication of a completed base-commissioning
//! proposal. This executable has no live device or motion-authority surface.

use std::path::PathBuf;

use clap::Parser;
use kiko_slam::navigation::nano_plant_promotion::promote_review_file;
use serde_json::json;

#[derive(Debug, Parser)]
#[command(
    name = "kiko-nano-plant-promote",
    about = "Verify and publish reviewed Nano plant artifacts (offline; no motion authority)"
)]
struct Args {
    /// Strict V1 review JSON binding every commissioning artifact and human declaration.
    #[arg(long)]
    review: PathBuf,

    /// Existing absolute, current-user-owned directory with mode 0700.
    #[arg(long)]
    output_root: PathBuf,
}

fn main() {
    let args = Args::parse();
    match promote_review_file(&args.review, &args.output_root) {
        Ok(result) => {
            let output = json!({
                "schema_version": 1,
                "status": "reviewed_bundle_input_motion_authority_withheld",
                "directory": result.directory,
                "production_plant": artifact(&result.production_plant),
                "promotion_evidence": artifact(&result.promotion_evidence),
                "renderer_values": artifact(&result.renderer_values),
                "completion_marker": artifact(&result.completion_marker),
            });
            println!(
                "{}",
                serde_json::to_string(&output).expect("fixed promotion result is serializable")
            );
        }
        Err(error) => {
            eprintln!("kiko-nano-plant-promote failed: {error}");
            std::process::exit(1);
        }
    }
}

fn artifact(
    artifact: &kiko_slam::navigation::nano_plant_promotion::PublishedPromotionArtifact,
) -> serde_json::Value {
    json!({
        "path": artifact.path,
        "sha256_hex": lower_hex(artifact.sha256),
        "bytes": artifact.bytes,
    })
}

fn lower_hex(digest: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_exposes_only_review_document_and_output_root() {
        let args = Args::try_parse_from([
            "kiko-nano-plant-promote",
            "--review",
            "/tmp/review.json",
            "--output-root",
            "/tmp/promotions",
        ])
        .expect("bounded arguments");
        assert_eq!(args.review, PathBuf::from("/tmp/review.json"));
        assert_eq!(args.output_root, PathBuf::from("/tmp/promotions"));

        let command = <Args as clap::CommandFactory>::command();
        let argument_ids: Vec<_> = command
            .get_arguments()
            .map(|argument| argument.get_id().as_str())
            .collect();
        assert_eq!(argument_ids, ["review", "output_root"]);
        assert!(
            command
                .get_arguments()
                .all(|argument| argument.get_env().is_none())
        );
    }

    #[test]
    fn cli_has_no_flag_that_can_assert_or_bypass_physical_review() {
        for forbidden in [
            "--approve",
            "--confirm-physical-review",
            "--skip-journal-review",
            "--activate",
            "--grant-motion-authority",
        ] {
            let error = Args::try_parse_from([
                "kiko-nano-plant-promote",
                "--review",
                "/tmp/review.json",
                "--output-root",
                "/tmp/promotions",
                forbidden,
            ])
            .expect_err("physical review cannot be asserted by a CLI flag");
            assert_eq!(error.kind(), clap::error::ErrorKind::UnknownArgument);
        }
    }
}
