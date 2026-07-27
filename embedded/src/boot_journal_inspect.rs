//! Host-only inspection of STM32F446 Kiko boot-journal readbacks.
//!
//! Snapshot mode parses one complete 128 KiB sector with the canonical journal
//! implementation. Transition mode proves that the second sector differs from
//! the first by exactly the one record planned by that implementation. The
//! tool performs no device I/O and writes no files.

use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use embedded::boot_journal::{
    BOOT_JOURNAL_SCHEMA_V1, BootJournalError, BootJournalScan, parse_boot_journal, plan_next_boot,
    verify_commit,
};
use robot_protocol::v2::ControllerBootId;
use serde::Serialize;
use sha2::{Digest, Sha256};

const STM32F446_SECTOR_7_ADDRESS_HEX: &str = "0x08060000";
const STM32F446_SECTOR_7_BYTES: usize = 128 * 1024;

#[derive(Parser, Debug)]
#[command(
    name = "kiko-boot-journal-inspect",
    about = "Inspect exact STM32F446 Kiko boot-journal sector readbacks"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Parse one exact sector image and plan the next boot record.
    Snapshot {
        /// Existing exact 128 KiB sector-7 readback.
        #[arg(long)]
        input: PathBuf,
    },
    /// Prove one exact canonical append between two sector images.
    Transition {
        /// Sector image immediately before one controlled boot.
        #[arg(long)]
        previous: PathBuf,
        /// Sector image after that boot has been halted.
        #[arg(long)]
        current: PathBuf,
    },
}

#[derive(Debug)]
enum ToolError {
    Open {
        path: PathBuf,
        source: std::io::Error,
    },
    Metadata {
        path: PathBuf,
        source: std::io::Error,
    },
    NotRegularFile {
        path: PathBuf,
    },
    UnexpectedInputLength {
        path: PathBuf,
        expected: usize,
        actual: u64,
    },
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    LengthConversion,
    Journal(BootJournalError),
    TransitionRange,
    TransitionDiffersOutsidePlannedRecord,
    Json(serde_json::Error),
}

impl fmt::Display for ToolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("boot-journal inspection failed: ")?;
        match self {
            Self::Open { path, source } => {
                write!(formatter, "could not open {}: {source}", path.display())
            }
            Self::Metadata { path, source } => {
                write!(formatter, "could not stat {}: {source}", path.display())
            }
            Self::NotRegularFile { path } => {
                write!(formatter, "{} is not a regular file", path.display())
            }
            Self::UnexpectedInputLength {
                path,
                expected,
                actual,
            } => write!(
                formatter,
                "{} has {actual} bytes, expected exactly {expected}",
                path.display()
            ),
            Self::Read { path, source } => {
                write!(
                    formatter,
                    "could not completely read {}: {source}",
                    path.display()
                )
            }
            Self::LengthConversion => {
                formatter.write_str("exact sector length cannot be represented by the host")
            }
            Self::Journal(source) => write!(formatter, "{source}"),
            Self::TransitionRange => {
                formatter.write_str("planned record range is outside the exact sector image")
            }
            Self::TransitionDiffersOutsidePlannedRecord => formatter.write_str(
                "current sector differs from previous by more than the one planned record",
            ),
            Self::Json(source) => write!(formatter, "evidence JSON encode failed: {source}"),
        }
    }
}

impl std::error::Error for ToolError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Open { source, .. }
            | Self::Metadata { source, .. }
            | Self::Read { source, .. } => Some(source),
            Self::Journal(source) => Some(source),
            Self::Json(source) => Some(source),
            Self::NotRegularFile { .. }
            | Self::UnexpectedInputLength { .. }
            | Self::LengthConversion
            | Self::TransitionRange
            | Self::TransitionDiffersOutsidePlannedRecord => None,
        }
    }
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct JournalState {
    provisioning_seed_hex: String,
    last_counter: Option<u64>,
    last_boot_id: Option<u64>,
    next_record_offset_bytes: Option<usize>,
    aborted_records: usize,
    record_capacity: usize,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct PlannedBoot {
    counter: u64,
    boot_id: u64,
    record_offset_bytes: usize,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct SnapshotObservation {
    schema_version: u32,
    observation_kind: &'static str,
    input_path: String,
    image_bytes: usize,
    image_sha256_hex: String,
    target_flash_address_hex: &'static str,
    journal_schema_version: u32,
    state: JournalState,
    planned_next_boot: PlannedBoot,
    evidence_boundary: &'static str,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct TransitionObservation {
    schema_version: u32,
    observation_kind: &'static str,
    previous_path: String,
    previous_sha256_hex: String,
    current_path: String,
    current_sha256_hex: String,
    image_bytes: usize,
    target_flash_address_hex: &'static str,
    journal_schema_version: u32,
    previous_state: JournalState,
    current_state: JournalState,
    committed_boot: PlannedBoot,
    current_planned_next_boot: PlannedBoot,
    exact_planned_record_only: bool,
    evidence_boundary: &'static str,
}

fn read_exact_sector(path: &Path) -> Result<Vec<u8>, ToolError> {
    let mut file = File::open(path).map_err(|source| ToolError::Open {
        path: path.to_path_buf(),
        source,
    })?;
    let metadata = file.metadata().map_err(|source| ToolError::Metadata {
        path: path.to_path_buf(),
        source,
    })?;
    if !metadata.is_file() {
        return Err(ToolError::NotRegularFile {
            path: path.to_path_buf(),
        });
    }
    let expected =
        u64::try_from(STM32F446_SECTOR_7_BYTES).map_err(|_| ToolError::LengthConversion)?;
    if metadata.len() != expected {
        return Err(ToolError::UnexpectedInputLength {
            path: path.to_path_buf(),
            expected: STM32F446_SECTOR_7_BYTES,
            actual: metadata.len(),
        });
    }

    let mut image = Vec::with_capacity(STM32F446_SECTOR_7_BYTES);
    file.read_to_end(&mut image)
        .map_err(|source| ToolError::Read {
            path: path.to_path_buf(),
            source,
        })?;
    let actual = u64::try_from(image.len()).map_err(|_| ToolError::LengthConversion)?;
    if actual != expected {
        return Err(ToolError::UnexpectedInputLength {
            path: path.to_path_buf(),
            expected: STM32F446_SECTOR_7_BYTES,
            actual,
        });
    }
    Ok(image)
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

fn image_sha256_hex(image: &[u8]) -> String {
    encode_hex(&Sha256::digest(image))
}

fn controller_boot_id_value(boot_id: ControllerBootId) -> u64 {
    boot_id.get()
}

fn state(scan: BootJournalScan) -> Result<JournalState, ToolError> {
    Ok(JournalState {
        provisioning_seed_hex: format!("{:016x}", scan.provisioning_seed().get()),
        last_counter: scan.last_counter().map(|counter| counter.get()),
        last_boot_id: scan
            .last_boot_id()
            .map_err(ToolError::Journal)?
            .map(controller_boot_id_value),
        next_record_offset_bytes: scan.next_record_offset(),
        aborted_records: scan.aborted_records(),
        record_capacity: scan.record_capacity(),
    })
}

fn planned_boot(image: &[u8]) -> Result<PlannedBoot, ToolError> {
    let planned = plan_next_boot(image).map_err(ToolError::Journal)?;
    Ok(PlannedBoot {
        counter: planned.counter().get(),
        boot_id: planned.boot_id().get(),
        record_offset_bytes: planned.record_offset(),
    })
}

fn inspect_snapshot(path: &Path, image: &[u8]) -> Result<SnapshotObservation, ToolError> {
    let scan = parse_boot_journal(image).map_err(ToolError::Journal)?;
    Ok(SnapshotObservation {
        schema_version: 1,
        observation_kind: "parsed_stm32f446_kiko_boot_journal_snapshot",
        input_path: path.display().to_string(),
        image_bytes: image.len(),
        image_sha256_hex: image_sha256_hex(image),
        target_flash_address_hex: STM32F446_SECTOR_7_ADDRESS_HEX,
        journal_schema_version: BOOT_JOURNAL_SCHEMA_V1,
        state: state(scan)?,
        planned_next_boot: planned_boot(image)?,
        evidence_boundary: "one exact local sector image was parsed; this observation performs no device I/O, grants no motor authority, and proves no reset cause or physical behavior",
    })
}

fn inspect_transition(
    previous_path: &Path,
    previous: &[u8],
    current_path: &Path,
    current: &[u8],
) -> Result<TransitionObservation, ToolError> {
    let previous_scan = parse_boot_journal(previous).map_err(ToolError::Journal)?;
    let planned = plan_next_boot(previous).map_err(ToolError::Journal)?;
    let record_offset = planned.record_offset();
    let record = planned.record();
    let record_end = record_offset
        .checked_add(record.len())
        .ok_or(ToolError::TransitionRange)?;
    let destination = previous
        .get(record_offset..record_end)
        .ok_or(ToolError::TransitionRange)?;
    if destination.iter().any(|byte| *byte != 0xff) {
        return Err(ToolError::TransitionRange);
    }

    let mut expected = previous.to_vec();
    expected
        .get_mut(record_offset..record_end)
        .ok_or(ToolError::TransitionRange)?
        .copy_from_slice(&record);
    if expected != current {
        return Err(ToolError::TransitionDiffersOutsidePlannedRecord);
    }
    let current_scan = parse_boot_journal(current).map_err(ToolError::Journal)?;
    let committed_boot_id = verify_commit(current, planned).map_err(ToolError::Journal)?;

    Ok(TransitionObservation {
        schema_version: 1,
        observation_kind: "verified_stm32f446_kiko_boot_journal_transition",
        previous_path: previous_path.display().to_string(),
        previous_sha256_hex: image_sha256_hex(previous),
        current_path: current_path.display().to_string(),
        current_sha256_hex: image_sha256_hex(current),
        image_bytes: current.len(),
        target_flash_address_hex: STM32F446_SECTOR_7_ADDRESS_HEX,
        journal_schema_version: BOOT_JOURNAL_SCHEMA_V1,
        previous_state: state(previous_scan)?,
        current_state: state(current_scan)?,
        committed_boot: PlannedBoot {
            counter: planned.counter().get(),
            boot_id: committed_boot_id.get(),
            record_offset_bytes: planned.record_offset(),
        },
        current_planned_next_boot: planned_boot(current)?,
        exact_planned_record_only: true,
        evidence_boundary: "two exact local sector images differ by exactly one canonically planned and verified record; the surrounding runbook must prove which controlled reset separated the snapshots",
    })
}

fn write_json<T: Serialize>(value: &T) -> Result<(), ToolError> {
    serde_json::to_writer_pretty(std::io::stdout().lock(), value).map_err(ToolError::Json)?;
    println!();
    Ok(())
}

fn main() -> Result<(), ToolError> {
    match Cli::parse().command {
        Command::Snapshot { input } => {
            let image = read_exact_sector(&input)?;
            write_json(&inspect_snapshot(&input, &image)?)
        }
        Command::Transition { previous, current } => {
            let previous_image = read_exact_sector(&previous)?;
            let current_image = read_exact_sector(&current)?;
            write_json(&inspect_transition(
                &previous,
                &previous_image,
                &current,
                &current_image,
            )?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use embedded::boot_journal::{
        BOOT_JOURNAL_ERASED_BYTE, BOOT_JOURNAL_HEADER_BYTES, BootJournalProvisioningSeed,
        encode_provisioned_header,
    };

    fn provisioned_sector(seed: BootJournalProvisioningSeed) -> Vec<u8> {
        let mut image = vec![BOOT_JOURNAL_ERASED_BYTE; STM32F446_SECTOR_7_BYTES];
        image[..BOOT_JOURNAL_HEADER_BYTES].copy_from_slice(&encode_provisioned_header(seed));
        image
    }

    fn apply_next_record(image: &mut [u8]) -> PlannedBoot {
        let planned = plan_next_boot(image).expect("space remains");
        let record_offset = planned.record_offset();
        let record = planned.record();
        image[record_offset..record_offset + record.len()].copy_from_slice(&record);
        PlannedBoot {
            counter: planned.counter().get(),
            boot_id: planned.boot_id().get(),
            record_offset_bytes: record_offset,
        }
    }

    #[test]
    fn fresh_sector_becomes_one_typed_snapshot() {
        let seed = BootJournalProvisioningSeed::try_new(7).expect("nonzero seed");
        let image = provisioned_sector(seed);
        let observed = inspect_snapshot(Path::new("sector7.bin"), &image).expect("valid journal");

        assert_eq!(observed.schema_version, 1);
        assert_eq!(observed.input_path, "sector7.bin");
        assert_eq!(observed.image_bytes, STM32F446_SECTOR_7_BYTES);
        assert_eq!(observed.state.provisioning_seed_hex, "0000000000000007");
        assert_eq!(observed.state.last_counter, None);
        assert_eq!(observed.state.last_boot_id, None);
        assert_eq!(
            observed.state.next_record_offset_bytes,
            Some(BOOT_JOURNAL_HEADER_BYTES)
        );
        assert_eq!(observed.planned_next_boot.counter, 1);
        assert_eq!(
            observed.planned_next_boot.record_offset_bytes,
            BOOT_JOURNAL_HEADER_BYTES
        );
        assert_eq!(observed.image_sha256_hex.len(), 64);
    }

    #[test]
    fn one_exact_planned_append_is_accepted() {
        let seed = BootJournalProvisioningSeed::try_new(7).expect("nonzero seed");
        let previous = provisioned_sector(seed);
        let mut current = previous.clone();
        let committed = apply_next_record(&mut current);

        let observed = inspect_transition(
            Path::new("previous.bin"),
            &previous,
            Path::new("current.bin"),
            &current,
        )
        .expect("exact transition");
        assert_eq!(observed.committed_boot, committed);
        assert_eq!(observed.current_state.last_counter, Some(1));
        assert_eq!(observed.current_state.last_boot_id, Some(committed.boot_id));
        assert_eq!(observed.current_planned_next_boot.counter, 2);
        assert!(observed.exact_planned_record_only);
    }

    #[test]
    fn planned_transition_preserves_and_skips_an_aborted_record() {
        let seed = BootJournalProvisioningSeed::try_new(7).expect("nonzero seed");
        let mut previous = provisioned_sector(seed);
        previous[BOOT_JOURNAL_HEADER_BYTES] = 0;
        let mut current = previous.clone();
        let committed = apply_next_record(&mut current);
        assert_eq!(committed.counter, 2);
        assert_eq!(
            committed.record_offset_bytes,
            BOOT_JOURNAL_HEADER_BYTES + 16
        );

        let observed = inspect_transition(
            Path::new("previous.bin"),
            &previous,
            Path::new("current.bin"),
            &current,
        )
        .expect("exact transition after an aborted slot");
        assert_eq!(observed.previous_state.aborted_records, 1);
        assert_eq!(observed.current_state.aborted_records, 1);
        assert_eq!(observed.committed_boot, committed);
    }

    #[test]
    fn any_change_outside_the_planned_record_is_rejected() {
        let seed = BootJournalProvisioningSeed::try_new(7).expect("nonzero seed");
        let previous = provisioned_sector(seed);
        let mut current = previous.clone();
        apply_next_record(&mut current);
        current[STM32F446_SECTOR_7_BYTES - 1] = 0;

        let error = inspect_transition(
            Path::new("previous.bin"),
            &previous,
            Path::new("current.bin"),
            &current,
        )
        .expect_err("extra mutation must fail");
        assert!(matches!(
            error,
            ToolError::TransitionDiffersOutsidePlannedRecord
        ));
    }

    #[test]
    fn malformed_sector_is_rejected_by_the_canonical_parser() {
        let image = vec![BOOT_JOURNAL_ERASED_BYTE; STM32F446_SECTOR_7_BYTES];
        let error = inspect_snapshot(Path::new("erased.bin"), &image)
            .expect_err("an erased header is invalid");
        assert!(matches!(
            error,
            ToolError::Journal(BootJournalError::HeaderErased)
        ));
    }

    #[test]
    fn encoded_hash_is_fixed_width_lower_hex() {
        assert_eq!(encode_hex(&[0x00, 0xab, 0xff]), "00abff");
    }
}
