//! Pure parser and commit planner for a provisioned STM32 boot-identity journal.
//!
//! The STM32F446 has no hardware random-number generator. Motion-capable
//! firmware therefore cannot derive a session-unique boot identity from the
//! device UID or a timer. A deployment tool provisions one random nonzero seed
//! in a flash sector that is outside the linked firmware image. Every boot
//! appends one counter record before serial or motor admission.
//!
//! This module performs no flash I/O. The target adapter must:
//!
//! 1. parse the complete reserved sector;
//! 2. program exactly the returned erased record;
//! 3. read the sector back; and
//! 4. admit the boot identity only if [`verify_commit`] succeeds.
//!
//! An interrupted record is retained as an unusable slot. Later records may
//! continue after it, but a valid counter must always equal its one-based slot
//! index. Burning a slot therefore also burns its identity; a corrupted old
//! record cannot make a later boot reuse that counter. Erased space after any
//! programmed space is the only valid append boundary.

use core::fmt;
use core::num::NonZeroU64;

use robot_protocol::v2::ControllerBootId;

pub const BOOT_JOURNAL_SCHEMA_V1: u32 = 1;
pub const BOOT_JOURNAL_HEADER_BYTES: usize = 32;
pub const BOOT_JOURNAL_RECORD_BYTES: usize = 16;
pub const BOOT_JOURNAL_ERASED_BYTE: u8 = 0xff;

const HEADER_MAGIC: [u8; 8] = *b"KIKOBOOT";
const HEADER_RESERVED_ZERO: u32 = 0;
const HEADER_COMMIT_MARKER: u32 = 0x5a3c_c3a5;
const RECORD_COMMIT_MARKER: u32 = 0xc35a_a53c;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BootJournalProvisioningSeed(NonZeroU64);

impl BootJournalProvisioningSeed {
    pub fn try_new(value: u64) -> Result<Self, BootJournalError> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or(BootJournalError::ZeroProvisioningSeed)
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BootJournalCounter(NonZeroU64);

impl BootJournalCounter {
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BootJournalScan {
    provisioning_seed: BootJournalProvisioningSeed,
    last_counter: Option<BootJournalCounter>,
    next_record_offset: Option<usize>,
    aborted_records: usize,
    record_capacity: usize,
}

impl BootJournalScan {
    pub const fn provisioning_seed(self) -> BootJournalProvisioningSeed {
        self.provisioning_seed
    }

    pub const fn last_counter(self) -> Option<BootJournalCounter> {
        self.last_counter
    }

    pub const fn next_record_offset(self) -> Option<usize> {
        self.next_record_offset
    }

    pub const fn aborted_records(self) -> usize {
        self.aborted_records
    }

    pub const fn record_capacity(self) -> usize {
        self.record_capacity
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BootJournalCommit {
    provisioning_seed: BootJournalProvisioningSeed,
    counter: BootJournalCounter,
    boot_id: ControllerBootId,
    record_offset: usize,
    record: [u8; BOOT_JOURNAL_RECORD_BYTES],
}

impl BootJournalCommit {
    pub const fn provisioning_seed(self) -> BootJournalProvisioningSeed {
        self.provisioning_seed
    }

    pub const fn counter(self) -> BootJournalCounter {
        self.counter
    }

    pub const fn boot_id(self) -> ControllerBootId {
        self.boot_id
    }

    pub const fn record_offset(self) -> usize {
        self.record_offset
    }

    pub const fn record(self) -> [u8; BOOT_JOURNAL_RECORD_BYTES] {
        self.record
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BootJournalError {
    RegionTooSmall {
        actual_bytes: usize,
        minimum_bytes: usize,
    },
    RegionLengthNotRecordAligned {
        actual_bytes: usize,
        header_bytes: usize,
        record_bytes: usize,
    },
    HeaderErased,
    HeaderMagicMismatch,
    HeaderSchemaMismatch {
        actual: u32,
        expected: u32,
    },
    HeaderReservedNonzero {
        actual: u32,
    },
    ZeroProvisioningSeed,
    HeaderChecksumMismatch,
    HeaderCommitMarkerMismatch,
    ProgrammedRecordAfterErasedBoundary {
        record_index: usize,
    },
    ValidCounterDoesNotMatchSlot {
        record_index: usize,
        expected: u64,
        actual: u64,
    },
    CounterExhausted,
    JournalFull {
        record_capacity: usize,
    },
    BootIdDerivationInvariant {
        seed: u64,
        counter: u64,
    },
    CommitRangeOutsideRegion {
        offset: usize,
        record_bytes: usize,
        region_bytes: usize,
    },
    CommitDestinationNotErased {
        offset: usize,
    },
    CommitReadbackMismatch {
        offset: usize,
    },
    CommitNotLastValidRecord {
        expected: u64,
        actual: Option<u64>,
    },
    CommitSeedChanged {
        expected: u64,
        actual: u64,
    },
}

impl fmt::Display for BootJournalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid boot-identity journal: {self:?}")
    }
}

impl core::error::Error for BootJournalError {}

pub fn parse_boot_journal(region: &[u8]) -> Result<BootJournalScan, BootJournalError> {
    let minimum_bytes = BOOT_JOURNAL_HEADER_BYTES
        .checked_add(BOOT_JOURNAL_RECORD_BYTES)
        .ok_or(BootJournalError::RegionTooSmall {
            actual_bytes: region.len(),
            minimum_bytes: usize::MAX,
        })?;
    if region.len() < minimum_bytes {
        return Err(BootJournalError::RegionTooSmall {
            actual_bytes: region.len(),
            minimum_bytes,
        });
    }
    let record_region_bytes = region.len() - BOOT_JOURNAL_HEADER_BYTES;
    if !record_region_bytes.is_multiple_of(BOOT_JOURNAL_RECORD_BYTES) {
        return Err(BootJournalError::RegionLengthNotRecordAligned {
            actual_bytes: region.len(),
            header_bytes: BOOT_JOURNAL_HEADER_BYTES,
            record_bytes: BOOT_JOURNAL_RECORD_BYTES,
        });
    }

    let header = &region[..BOOT_JOURNAL_HEADER_BYTES];
    if is_erased(header) {
        return Err(BootJournalError::HeaderErased);
    }
    if header[..8] != HEADER_MAGIC {
        return Err(BootJournalError::HeaderMagicMismatch);
    }
    let schema = read_u32(&header[8..12]);
    if schema != BOOT_JOURNAL_SCHEMA_V1 {
        return Err(BootJournalError::HeaderSchemaMismatch {
            actual: schema,
            expected: BOOT_JOURNAL_SCHEMA_V1,
        });
    }
    let reserved = read_u32(&header[12..16]);
    if reserved != HEADER_RESERVED_ZERO {
        return Err(BootJournalError::HeaderReservedNonzero { actual: reserved });
    }
    let seed_raw = read_u64(&header[16..24]);
    let provisioning_seed = BootJournalProvisioningSeed::try_new(seed_raw)?;
    let header_checksum = read_u32(&header[24..28]);
    if header_checksum != crc32c(&header[..24]) {
        return Err(BootJournalError::HeaderChecksumMismatch);
    }
    if read_u32(&header[28..32]) != HEADER_COMMIT_MARKER {
        return Err(BootJournalError::HeaderCommitMarkerMismatch);
    }

    let records = &region[BOOT_JOURNAL_HEADER_BYTES..];
    let record_capacity = records.len() / BOOT_JOURNAL_RECORD_BYTES;
    let mut last_counter: Option<BootJournalCounter> = None;
    let mut next_record_offset = None;
    let mut aborted_records = 0_usize;
    let mut saw_erased_boundary = false;

    for (record_index, record) in records.chunks_exact(BOOT_JOURNAL_RECORD_BYTES).enumerate() {
        let record_offset = BOOT_JOURNAL_HEADER_BYTES
            .checked_add(
                record_index
                    .checked_mul(BOOT_JOURNAL_RECORD_BYTES)
                    .ok_or(BootJournalError::JournalFull { record_capacity })?,
            )
            .ok_or(BootJournalError::JournalFull { record_capacity })?;
        if is_erased(record) {
            saw_erased_boundary = true;
            if next_record_offset.is_none() {
                next_record_offset = Some(record_offset);
            }
            continue;
        }
        if saw_erased_boundary {
            return Err(BootJournalError::ProgrammedRecordAfterErasedBoundary { record_index });
        }

        let counter_raw = read_u64(&record[..8]);
        let checksum = read_u32(&record[8..12]);
        let marker = read_u32(&record[12..16]);
        let Some(counter) = NonZeroU64::new(counter_raw)
            .filter(|counter| {
                checksum == record_checksum(provisioning_seed, counter.get())
                    && marker == RECORD_COMMIT_MARKER
            })
            .map(BootJournalCounter)
        else {
            aborted_records = aborted_records
                .checked_add(1)
                .ok_or(BootJournalError::JournalFull { record_capacity })?;
            continue;
        };
        let expected = u64::try_from(record_index)
            .ok()
            .and_then(|index| index.checked_add(1))
            .ok_or(BootJournalError::CounterExhausted)?;
        if counter.get() != expected {
            return Err(BootJournalError::ValidCounterDoesNotMatchSlot {
                record_index,
                expected,
                actual: counter.get(),
            });
        }
        last_counter = Some(counter);
    }

    Ok(BootJournalScan {
        provisioning_seed,
        last_counter,
        next_record_offset,
        aborted_records,
        record_capacity,
    })
}

pub fn plan_next_boot(region: &[u8]) -> Result<BootJournalCommit, BootJournalError> {
    let scan = parse_boot_journal(region)?;
    let record_index = scan
        .next_record_offset
        .and_then(|offset| offset.checked_sub(BOOT_JOURNAL_HEADER_BYTES))
        .map(|offset| offset / BOOT_JOURNAL_RECORD_BYTES)
        .ok_or(BootJournalError::JournalFull {
            record_capacity: scan.record_capacity,
        })?;
    let counter_raw = u64::try_from(record_index)
        .ok()
        .and_then(|index| index.checked_add(1))
        .ok_or(BootJournalError::CounterExhausted)?;
    let counter =
        BootJournalCounter(NonZeroU64::new(counter_raw).ok_or(BootJournalError::CounterExhausted)?);
    let record_offset = scan
        .next_record_offset
        .ok_or(BootJournalError::JournalFull {
            record_capacity: scan.record_capacity,
        })?;
    let end = record_offset.checked_add(BOOT_JOURNAL_RECORD_BYTES).ok_or(
        BootJournalError::CommitRangeOutsideRegion {
            offset: record_offset,
            record_bytes: BOOT_JOURNAL_RECORD_BYTES,
            region_bytes: region.len(),
        },
    )?;
    let destination =
        region
            .get(record_offset..end)
            .ok_or(BootJournalError::CommitRangeOutsideRegion {
                offset: record_offset,
                record_bytes: BOOT_JOURNAL_RECORD_BYTES,
                region_bytes: region.len(),
            })?;
    if !is_erased(destination) {
        return Err(BootJournalError::CommitDestinationNotErased {
            offset: record_offset,
        });
    }

    let boot_id = derive_boot_id(scan.provisioning_seed, counter)?;
    let mut record = [0_u8; BOOT_JOURNAL_RECORD_BYTES];
    record[..8].copy_from_slice(&counter.get().to_le_bytes());
    record[8..12]
        .copy_from_slice(&record_checksum(scan.provisioning_seed, counter.get()).to_le_bytes());
    record[12..16].copy_from_slice(&RECORD_COMMIT_MARKER.to_le_bytes());
    Ok(BootJournalCommit {
        provisioning_seed: scan.provisioning_seed,
        counter,
        boot_id,
        record_offset,
        record,
    })
}

pub fn verify_commit(
    region: &[u8],
    commit: BootJournalCommit,
) -> Result<ControllerBootId, BootJournalError> {
    let end = commit
        .record_offset
        .checked_add(BOOT_JOURNAL_RECORD_BYTES)
        .ok_or(BootJournalError::CommitRangeOutsideRegion {
            offset: commit.record_offset,
            record_bytes: BOOT_JOURNAL_RECORD_BYTES,
            region_bytes: region.len(),
        })?;
    let readback = region.get(commit.record_offset..end).ok_or(
        BootJournalError::CommitRangeOutsideRegion {
            offset: commit.record_offset,
            record_bytes: BOOT_JOURNAL_RECORD_BYTES,
            region_bytes: region.len(),
        },
    )?;
    if readback != commit.record {
        return Err(BootJournalError::CommitReadbackMismatch {
            offset: commit.record_offset,
        });
    }
    let scan = parse_boot_journal(region)?;
    if scan.provisioning_seed != commit.provisioning_seed {
        return Err(BootJournalError::CommitSeedChanged {
            expected: commit.provisioning_seed.get(),
            actual: scan.provisioning_seed.get(),
        });
    }
    if scan.last_counter != Some(commit.counter) {
        return Err(BootJournalError::CommitNotLastValidRecord {
            expected: commit.counter.get(),
            actual: scan.last_counter.map(BootJournalCounter::get),
        });
    }
    Ok(commit.boot_id)
}

pub fn encode_provisioned_header(
    seed: BootJournalProvisioningSeed,
) -> [u8; BOOT_JOURNAL_HEADER_BYTES] {
    let mut header = [0_u8; BOOT_JOURNAL_HEADER_BYTES];
    header[..8].copy_from_slice(&HEADER_MAGIC);
    header[8..12].copy_from_slice(&BOOT_JOURNAL_SCHEMA_V1.to_le_bytes());
    header[12..16].copy_from_slice(&HEADER_RESERVED_ZERO.to_le_bytes());
    header[16..24].copy_from_slice(&seed.get().to_le_bytes());
    let checksum = crc32c(&header[..24]);
    header[24..28].copy_from_slice(&checksum.to_le_bytes());
    header[28..32].copy_from_slice(&HEADER_COMMIT_MARKER.to_le_bytes());
    header
}

fn is_erased(bytes: &[u8]) -> bool {
    bytes.iter().all(|byte| *byte == BOOT_JOURNAL_ERASED_BYTE)
}

fn read_u32(bytes: &[u8]) -> u32 {
    let mut value = [0_u8; 4];
    value.copy_from_slice(bytes);
    u32::from_le_bytes(value)
}

fn read_u64(bytes: &[u8]) -> u64 {
    let mut value = [0_u8; 8];
    value.copy_from_slice(bytes);
    u64::from_le_bytes(value)
}

fn record_checksum(seed: BootJournalProvisioningSeed, counter: u64) -> u32 {
    let mut material = [0_u8; 16];
    material[..8].copy_from_slice(&seed.get().to_le_bytes());
    material[8..].copy_from_slice(&counter.to_le_bytes());
    crc32c(&material)
}

fn derive_boot_id(
    seed: BootJournalProvisioningSeed,
    counter: BootJournalCounter,
) -> Result<ControllerBootId, BootJournalError> {
    // Addition modulo the size of the nonzero u64 domain is injective for all
    // counters in one journal. Unlike `value | 1`, it cannot collapse two
    // successive counters onto one boot ID.
    let nonzero_domain = u128::from(u64::MAX);
    let zero_based_seed = u128::from(seed.get() - 1);
    let zero_based = (zero_based_seed + u128::from(counter.get())) % nonzero_domain;
    let raw =
        u64::try_from(zero_based + 1).map_err(|_| BootJournalError::BootIdDerivationInvariant {
            seed: seed.get(),
            counter: counter.get(),
        })?;
    ControllerBootId::try_new(raw).map_err(|_| BootJournalError::BootIdDerivationInvariant {
        seed: seed.get(),
        counter: counter.get(),
    })
}

fn crc32c(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = 0_u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0x82f6_3b78 & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    const RECORDS: usize = 8;
    const REGION_BYTES: usize = BOOT_JOURNAL_HEADER_BYTES + RECORDS * BOOT_JOURNAL_RECORD_BYTES;

    fn seed() -> BootJournalProvisioningSeed {
        BootJournalProvisioningSeed::try_new(0x5f3c_a924_1bd8_7701).expect("nonzero seed")
    }

    fn region() -> [u8; REGION_BYTES] {
        let mut region = [BOOT_JOURNAL_ERASED_BYTE; REGION_BYTES];
        region[..BOOT_JOURNAL_HEADER_BYTES].copy_from_slice(&encode_provisioned_header(seed()));
        region
    }

    fn apply(region: &mut [u8], commit: BootJournalCommit) {
        let start = commit.record_offset();
        let end = start + BOOT_JOURNAL_RECORD_BYTES;
        region[start..end].copy_from_slice(&commit.record());
    }

    #[test]
    fn provisioned_empty_journal_commits_and_verifies_first_boot() {
        let mut region = region();
        let commit = plan_next_boot(&region).expect("first plan");
        assert_eq!(commit.counter().get(), 1);
        assert_eq!(commit.record_offset(), BOOT_JOURNAL_HEADER_BYTES);
        apply(&mut region, commit);
        assert_eq!(
            verify_commit(&region, commit).expect("verified first commit"),
            commit.boot_id()
        );
        let scan = parse_boot_journal(&region).expect("scan");
        assert_eq!(scan.last_counter().map(BootJournalCounter::get), Some(1));
        assert_eq!(scan.aborted_records(), 0);
    }

    #[test]
    fn every_successive_boot_is_exact_and_has_a_distinct_identity() {
        let mut region = region();
        let mut previous_boot = None;
        for expected_counter in 1..=RECORDS as u64 {
            let commit = plan_next_boot(&region).expect("next plan");
            assert_eq!(commit.counter().get(), expected_counter);
            assert_ne!(Some(commit.boot_id()), previous_boot);
            apply(&mut region, commit);
            verify_commit(&region, commit).expect("commit verifies");
            previous_boot = Some(commit.boot_id());
        }
        assert!(matches!(
            plan_next_boot(&region),
            Err(BootJournalError::JournalFull {
                record_capacity: RECORDS
            })
        ));
    }

    #[test]
    fn interrupted_nonempty_record_burns_its_counter_and_advances_the_slot() {
        let mut region = region();
        let first = plan_next_boot(&region).expect("first");
        apply(&mut region, first);
        let interrupted_offset = first.record_offset() + BOOT_JOURNAL_RECORD_BYTES;
        region[interrupted_offset] = 0x7f;

        let next = plan_next_boot(&region).expect("skip interrupted slot");
        assert_eq!(next.counter().get(), 3);
        assert_eq!(
            next.record_offset(),
            interrupted_offset + BOOT_JOURNAL_RECORD_BYTES
        );
        apply(&mut region, next);
        let scan = parse_boot_journal(&region).expect("scan with aborted record");
        assert_eq!(scan.aborted_records(), 1);
        assert_eq!(scan.last_counter().map(BootJournalCounter::get), Some(3));
    }

    #[test]
    fn corruption_of_an_old_committed_record_cannot_reuse_its_identity() {
        let mut region = region();
        let first = plan_next_boot(&region).expect("first");
        apply(&mut region, first);
        let second = plan_next_boot(&region).expect("second");
        apply(&mut region, second);

        // Model a retained but no-longer-valid programmed record. Slot two was
        // already consumed even though its checksum no longer validates.
        region[second.record_offset() + 8] ^= 1;
        let third = plan_next_boot(&region).expect("third slot");
        assert_eq!(third.counter().get(), 3);
        assert_ne!(third.boot_id(), first.boot_id());
        assert_ne!(third.boot_id(), second.boot_id());
    }

    #[test]
    fn programmed_data_after_an_erased_gap_is_rejected() {
        let mut region = region();
        let third = BOOT_JOURNAL_HEADER_BYTES + 2 * BOOT_JOURNAL_RECORD_BYTES;
        region[third] = 0;
        assert!(matches!(
            parse_boot_journal(&region),
            Err(BootJournalError::ProgrammedRecordAfterErasedBoundary { record_index: 2 })
        ));
    }

    #[test]
    fn a_valid_counter_must_match_its_one_based_slot() {
        let mut region = region();
        let counter = 2_u64;
        region[BOOT_JOURNAL_HEADER_BYTES..BOOT_JOURNAL_HEADER_BYTES + 8]
            .copy_from_slice(&counter.to_le_bytes());
        region[BOOT_JOURNAL_HEADER_BYTES + 8..BOOT_JOURNAL_HEADER_BYTES + 12]
            .copy_from_slice(&record_checksum(seed(), counter).to_le_bytes());
        region[BOOT_JOURNAL_HEADER_BYTES + 12..BOOT_JOURNAL_HEADER_BYTES + 16]
            .copy_from_slice(&RECORD_COMMIT_MARKER.to_le_bytes());
        assert!(matches!(
            parse_boot_journal(&region),
            Err(BootJournalError::ValidCounterDoesNotMatchSlot {
                record_index: 0,
                expected: 1,
                actual: 2
            })
        ));
    }

    #[test]
    fn malformed_or_unprovisioned_headers_never_plan_a_boot() {
        let erased = [BOOT_JOURNAL_ERASED_BYTE; REGION_BYTES];
        assert_eq!(plan_next_boot(&erased), Err(BootJournalError::HeaderErased));

        let mut wrong_magic = region();
        wrong_magic[0] ^= 1;
        assert_eq!(
            plan_next_boot(&wrong_magic),
            Err(BootJournalError::HeaderMagicMismatch)
        );

        let mut wrong_checksum = region();
        wrong_checksum[24] ^= 1;
        assert_eq!(
            plan_next_boot(&wrong_checksum),
            Err(BootJournalError::HeaderChecksumMismatch)
        );
    }

    #[test]
    fn all_single_prefix_power_losses_are_either_erased_aborted_or_valid() {
        let base = region();
        let commit = plan_next_boot(&base).expect("plan");
        for programmed_prefix in 0..=BOOT_JOURNAL_RECORD_BYTES {
            let mut interrupted = base;
            let start = commit.record_offset();
            interrupted[start..start + programmed_prefix]
                .copy_from_slice(&commit.record()[..programmed_prefix]);
            let scan = parse_boot_journal(&interrupted).expect("prefix is structurally bounded");
            if programmed_prefix == 0 {
                assert_eq!(scan.last_counter(), None);
                assert_eq!(scan.aborted_records(), 0);
            } else if scan.last_counter() == Some(commit.counter()) {
                assert_eq!(programmed_prefix, BOOT_JOURNAL_RECORD_BYTES);
            } else {
                assert_eq!(scan.last_counter(), None);
                assert_eq!(scan.aborted_records(), 1);
            }
        }
    }

    #[test]
    fn verification_rejects_partial_or_modified_readback() {
        let mut region = region();
        let commit = plan_next_boot(&region).expect("plan");
        region[commit.record_offset()] = commit.record()[0];
        assert!(matches!(
            verify_commit(&region, commit),
            Err(BootJournalError::CommitReadbackMismatch { .. })
        ));

        apply(&mut region, commit);
        region[commit.record_offset() + 7] ^= 1;
        assert!(matches!(
            verify_commit(&region, commit),
            Err(BootJournalError::CommitReadbackMismatch { .. })
        ));
    }

    #[test]
    fn region_shape_is_checked_before_header_parsing() {
        assert!(matches!(
            parse_boot_journal(&[]),
            Err(BootJournalError::RegionTooSmall { .. })
        ));
        let malformed = [BOOT_JOURNAL_ERASED_BYTE; BOOT_JOURNAL_HEADER_BYTES + 1];
        assert!(matches!(
            parse_boot_journal(&malformed),
            Err(BootJournalError::RegionTooSmall { .. })
                | Err(BootJournalError::RegionLengthNotRecordAligned { .. })
        ));
    }
}
