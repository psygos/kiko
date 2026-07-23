//! Host-only generator for a freshly provisioned STM32F446 boot journal.
//!
//! The resulting 128 KiB image targets sector 7 at `0x08060000`. It contains
//! one CSPRNG-generated, nonzero provisioning seed and no boot records. The
//! deployment procedure must flash the complete image and compare a complete
//! sector readback before building firmware with `flash-boot-journal`.

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;

use clap::Parser;
use embedded::boot_journal::{
    BOOT_JOURNAL_ERASED_BYTE, BOOT_JOURNAL_HEADER_BYTES, BootJournalProvisioningSeed,
    encode_provisioned_header, parse_boot_journal,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

const STM32F446_SECTOR_7_ADDRESS: u32 = 0x0806_0000;
const STM32F446_SECTOR_7_BYTES: usize = 128 * 1024;
const MAX_ENTROPY_ATTEMPTS: usize = 8;

#[derive(Parser, Debug)]
#[command(
    name = "kiko-boot-journal-image",
    about = "Generate a fresh STM32F446 sector-7 boot-journal image"
)]
struct Cli {
    /// New image destination. Existing files and symlinks are never replaced.
    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug)]
enum ToolError {
    Entropy(getrandom::Error),
    EntropyProducedOnlyZero { attempts: usize },
    Create(std::io::Error),
    Write(std::io::Error),
    Sync(std::io::Error),
    Reopen(std::io::Error),
    Readback(std::io::Error),
    UnexpectedReadbackLength { expected: usize, actual: usize },
    ReadbackMismatch,
    GeneratedJournalInvalid(embedded::boot_journal::BootJournalError),
    Json(serde_json::Error),
}

impl fmt::Display for ToolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("boot-journal image generation failed: ")?;
        match self {
            Self::Entropy(source) => write!(formatter, "CSPRNG failed: {source}"),
            Self::EntropyProducedOnlyZero { attempts } => write!(
                formatter,
                "CSPRNG produced only zero seeds across {attempts} bounded attempts"
            ),
            Self::Create(source) => write!(formatter, "create-new output failed: {source}"),
            Self::Write(source) => write!(formatter, "complete image write failed: {source}"),
            Self::Sync(source) => write!(formatter, "image fsync failed: {source}"),
            Self::Reopen(source) => write!(formatter, "image reopen failed: {source}"),
            Self::Readback(source) => write!(formatter, "local image readback failed: {source}"),
            Self::UnexpectedReadbackLength { expected, actual } => write!(
                formatter,
                "local readback length {actual} differs from expected {expected}"
            ),
            Self::ReadbackMismatch => {
                formatter.write_str("local readback differs from generated image")
            }
            Self::GeneratedJournalInvalid(source) => {
                write!(formatter, "generated journal did not parse: {source}")
            }
            Self::Json(source) => write!(formatter, "evidence JSON encode failed: {source}"),
        }
    }
}

impl std::error::Error for ToolError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Entropy(source) => Some(source),
            Self::Create(source)
            | Self::Write(source)
            | Self::Sync(source)
            | Self::Reopen(source)
            | Self::Readback(source) => Some(source),
            Self::GeneratedJournalInvalid(source) => Some(source),
            Self::Json(source) => Some(source),
            Self::EntropyProducedOnlyZero { .. }
            | Self::UnexpectedReadbackLength { .. }
            | Self::ReadbackMismatch => None,
        }
    }
}

#[derive(Serialize)]
struct Evidence {
    schema_version: u32,
    artifact_kind: &'static str,
    output_path: String,
    image_bytes: usize,
    image_sha256_hex: String,
    provisioning_seed_hex: String,
    target_flash_address_hex: &'static str,
    target_flash_sector: u8,
    target_flash_sector_bytes: usize,
    initial_valid_boot_records: usize,
    required_deployment_evidence: &'static str,
    safety_boundary: &'static str,
}

fn fresh_seed() -> Result<BootJournalProvisioningSeed, ToolError> {
    for _ in 0..MAX_ENTROPY_ATTEMPTS {
        let mut bytes = [0_u8; 8];
        getrandom::fill(&mut bytes).map_err(ToolError::Entropy)?;
        if let Ok(seed) = BootJournalProvisioningSeed::try_new(u64::from_le_bytes(bytes)) {
            return Ok(seed);
        }
    }
    Err(ToolError::EntropyProducedOnlyZero {
        attempts: MAX_ENTROPY_ATTEMPTS,
    })
}

fn build_image(
    seed: BootJournalProvisioningSeed,
) -> Result<Vec<u8>, embedded::boot_journal::BootJournalError> {
    let mut image = vec![BOOT_JOURNAL_ERASED_BYTE; STM32F446_SECTOR_7_BYTES];
    image[..BOOT_JOURNAL_HEADER_BYTES].copy_from_slice(&encode_provisioned_header(seed));
    let scan = parse_boot_journal(&image)?;
    debug_assert_eq!(scan.provisioning_seed(), seed);
    debug_assert!(scan.last_counter().is_none());
    Ok(image)
}

fn write_new_and_verify(path: &PathBuf, image: &[u8]) -> Result<(), ToolError> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600);
    let mut file = options.open(path).map_err(ToolError::Create)?;
    file.write_all(image).map_err(ToolError::Write)?;
    file.sync_all().map_err(ToolError::Sync)?;
    drop(file);

    let mut readback = Vec::with_capacity(image.len().saturating_add(1));
    File::open(path)
        .map_err(ToolError::Reopen)?
        .take(
            u64::try_from(image.len())
                .unwrap_or(u64::MAX)
                .saturating_add(1),
        )
        .read_to_end(&mut readback)
        .map_err(ToolError::Readback)?;
    if readback.len() != image.len() {
        return Err(ToolError::UnexpectedReadbackLength {
            expected: image.len(),
            actual: readback.len(),
        });
    }
    if readback != image {
        return Err(ToolError::ReadbackMismatch);
    }
    Ok(())
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

fn main() -> Result<(), ToolError> {
    let cli = Cli::parse();
    let seed = fresh_seed()?;
    let image = build_image(seed).map_err(ToolError::GeneratedJournalInvalid)?;
    write_new_and_verify(&cli.output, &image)?;
    let image_sha256 = Sha256::digest(&image);
    let evidence = Evidence {
        schema_version: 1,
        artifact_kind: "stm32f446_kiko_boot_journal_sector_image",
        output_path: cli.output.display().to_string(),
        image_bytes: image.len(),
        image_sha256_hex: encode_hex(&image_sha256),
        provisioning_seed_hex: format!("{:016x}", seed.get()),
        target_flash_address_hex: "0x08060000",
        target_flash_sector: 7,
        target_flash_sector_bytes: STM32F446_SECTOR_7_BYTES,
        initial_valid_boot_records: 0,
        required_deployment_evidence: "flash the complete sector image, read back exactly 131072 bytes from 0x08060000, and compare SHA-256 before starting journal firmware",
        safety_boundary: "this artifact provides reset-unique software session identity after verified append; it grants no motor authority and proves no wiring or physical stop behavior",
    };
    debug_assert_eq!(STM32F446_SECTOR_7_ADDRESS, 0x0806_0000);
    serde_json::to_writer_pretty(std::io::stdout().lock(), &evidence).map_err(ToolError::Json)?;
    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_is_exact_sector_shape_with_one_valid_header_and_no_records() {
        let seed = BootJournalProvisioningSeed::try_new(7).expect("seed");
        let image = build_image(seed).expect("image");
        assert_eq!(image.len(), STM32F446_SECTOR_7_BYTES);
        let scan = parse_boot_journal(&image).expect("scan");
        assert_eq!(scan.provisioning_seed(), seed);
        assert_eq!(scan.last_counter(), None);
        assert_eq!(scan.next_record_offset(), Some(BOOT_JOURNAL_HEADER_BYTES));
        assert_eq!(
            scan.record_capacity(),
            (STM32F446_SECTOR_7_BYTES - BOOT_JOURNAL_HEADER_BYTES)
                / embedded::boot_journal::BOOT_JOURNAL_RECORD_BYTES
        );
    }

    #[test]
    fn encoded_hash_and_seed_are_fixed_width_lower_hex() {
        assert_eq!(encode_hex(&[0x00, 0xab, 0xff]), "00abff");
        assert_eq!(format!("{:016x}", 7_u64), "0000000000000007");
    }
}
