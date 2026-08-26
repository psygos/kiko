//! Bounded, durable production journal for completed pet episodes.
//!
//! The accessory owner encodes one complete typed record and performs only a
//! nonblocking bounded enqueue. A named OS thread owns the file descriptor,
//! appends and synchronizes records in FIFO order, and publishes the first
//! failure. The file is opened relative to a component-wise `O_NOFOLLOW`
//! state-root descriptor and its path identity is rechecked around every
//! append. No write or `fsync` runs on the accessory control thread.

use std::ffi::OsStr;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::num::NonZeroU64;
use std::os::fd::OwnedFd;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crossbeam_channel::{Receiver, Sender, TryRecvError};
use rustix::fs::{AtFlags, FileType, Mode, OFlags, Stat, fstat, fsync, open, openat, statat};
use rustix::io::Errno;

use super::{
    MAX_NANO_PET_EVIDENCE_RECORD_BYTES, NanoPetEpisodeEvidence, NanoPetEvidenceDecodeError,
    NanoPetEvidenceEncodeError,
};
use kiko_head_runtime::compliant_hold::CompliantPetEpisodeSummary;

pub const NANO_PET_EVIDENCE_JOURNAL_FILENAME: &str = "pet-episodes-v1.ndjson";
pub const MAX_NANO_PET_EVIDENCE_JOURNAL_BYTES: u64 = 16 * 1_024 * 1_024;
pub const NANO_PET_EVIDENCE_JOURNAL_QUEUE_CAPACITY: usize = 8;
const JOURNAL_MODE: Mode = Mode::from_raw_mode(0o600);
const JOIN_POLL_INTERVAL: Duration = Duration::from_millis(1);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoPetEvidenceJournalConfig {
    state_root: PathBuf,
    maximum_bytes: NonZeroU64,
}

impl NanoPetEvidenceJournalConfig {
    pub fn for_state_root(state_root: &Path) -> Result<Self, NanoPetEvidenceJournalConfigError> {
        validate_absolute_path(state_root)?;
        Ok(Self {
            state_root: state_root.to_path_buf(),
            maximum_bytes: NonZeroU64::new(MAX_NANO_PET_EVIDENCE_JOURNAL_BYTES)
                .expect("journal byte bound is nonzero"),
        })
    }

    pub fn state_root(&self) -> &Path {
        &self.state_root
    }

    pub const fn maximum_bytes(&self) -> NonZeroU64 {
        self.maximum_bytes
    }

    pub fn path(&self) -> PathBuf {
        self.state_root.join(NANO_PET_EVIDENCE_JOURNAL_FILENAME)
    }

    #[cfg(test)]
    fn with_maximum_bytes(mut self, maximum_bytes: NonZeroU64) -> Self {
        self.maximum_bytes = maximum_bytes;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NanoPetEvidenceJournalConfigError {
    NotAbsolute { path: PathBuf },
    NonCanonicalComponent { path: PathBuf },
}

impl fmt::Display for NanoPetEvidenceJournalConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid pet-evidence journal configuration: {self:?}"
        )
    }
}

impl std::error::Error for NanoPetEvidenceJournalConfigError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoPetEvidenceJournalIoOperation {
    OpenStateRootComponent { component_index: usize },
    InspectJournal,
    OpenJournal,
    SynchronizeStateRoot,
    ReadExistingJournal,
    AppendRecord,
    SynchronizeRecord,
}

#[derive(Clone, Debug)]
pub enum NanoPetEvidenceJournalRuntimeError {
    Encode(Arc<NanoPetEvidenceEncodeError>),
    WallClockBeforeUnixEpoch(Arc<std::time::SystemTimeError>),
    WallClockMillisecondsOutOfRange {
        milliseconds: u128,
    },
    QueueFull,
    WriterDisconnected,
    Io {
        operation: NanoPetEvidenceJournalIoOperation,
        kind: std::io::ErrorKind,
        raw_os_error: Option<i32>,
        message: Arc<str>,
    },
    JournalPathMissing,
    JournalPathNotRegular,
    JournalPathReplaced {
        expected_device: i128,
        expected_inode: i128,
        observed_device: i128,
        observed_inode: i128,
    },
    JournalHasMultipleLinks {
        links: u64,
    },
    JournalOwnerChanged {
        expected_uid: u32,
        observed_uid: u32,
    },
    JournalModeChanged {
        expected: u32,
        observed: u32,
    },
    JournalLengthOutOfRange {
        bytes: i128,
    },
    JournalLengthOverflow,
    JournalByteLimitExceeded {
        attempted: u64,
        maximum: u64,
    },
}

impl PartialEq for NanoPetEvidenceJournalRuntimeError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Encode(left), Self::Encode(right)) => Arc::ptr_eq(left, right),
            (Self::WallClockBeforeUnixEpoch(left), Self::WallClockBeforeUnixEpoch(right)) => {
                Arc::ptr_eq(left, right)
            }
            (
                Self::WallClockMillisecondsOutOfRange { milliseconds: left },
                Self::WallClockMillisecondsOutOfRange {
                    milliseconds: right,
                },
            ) => left == right,
            (Self::QueueFull, Self::QueueFull)
            | (Self::WriterDisconnected, Self::WriterDisconnected)
            | (Self::JournalPathMissing, Self::JournalPathMissing)
            | (Self::JournalPathNotRegular, Self::JournalPathNotRegular)
            | (Self::JournalLengthOverflow, Self::JournalLengthOverflow) => true,
            (
                Self::Io {
                    operation: left_operation,
                    kind: left_kind,
                    raw_os_error: left_raw,
                    message: left_message,
                },
                Self::Io {
                    operation: right_operation,
                    kind: right_kind,
                    raw_os_error: right_raw,
                    message: right_message,
                },
            ) => {
                left_operation == right_operation
                    && left_kind == right_kind
                    && left_raw == right_raw
                    && left_message == right_message
            }
            (
                Self::JournalPathReplaced {
                    expected_device: left_expected_device,
                    expected_inode: left_expected_inode,
                    observed_device: left_observed_device,
                    observed_inode: left_observed_inode,
                },
                Self::JournalPathReplaced {
                    expected_device: right_expected_device,
                    expected_inode: right_expected_inode,
                    observed_device: right_observed_device,
                    observed_inode: right_observed_inode,
                },
            ) => {
                left_expected_device == right_expected_device
                    && left_expected_inode == right_expected_inode
                    && left_observed_device == right_observed_device
                    && left_observed_inode == right_observed_inode
            }
            (
                Self::JournalHasMultipleLinks { links: left },
                Self::JournalHasMultipleLinks { links: right },
            ) => left == right,
            (
                Self::JournalOwnerChanged {
                    expected_uid: left_expected,
                    observed_uid: left_observed,
                },
                Self::JournalOwnerChanged {
                    expected_uid: right_expected,
                    observed_uid: right_observed,
                },
            ) => left_expected == right_expected && left_observed == right_observed,
            (
                Self::JournalModeChanged {
                    expected: left_expected,
                    observed: left_observed,
                },
                Self::JournalModeChanged {
                    expected: right_expected,
                    observed: right_observed,
                },
            ) => left_expected == right_expected && left_observed == right_observed,
            (
                Self::JournalLengthOutOfRange { bytes: left },
                Self::JournalLengthOutOfRange { bytes: right },
            ) => left == right,
            (
                Self::JournalByteLimitExceeded {
                    attempted: left_attempted,
                    maximum: left_maximum,
                },
                Self::JournalByteLimitExceeded {
                    attempted: right_attempted,
                    maximum: right_maximum,
                },
            ) => left_attempted == right_attempted && left_maximum == right_maximum,
            _ => false,
        }
    }
}

impl Eq for NanoPetEvidenceJournalRuntimeError {}

impl NanoPetEvidenceJournalRuntimeError {
    fn io(operation: NanoPetEvidenceJournalIoOperation, source: impl Into<std::io::Error>) -> Self {
        let source = source.into();
        Self::Io {
            operation,
            kind: source.kind(),
            raw_os_error: source.raw_os_error(),
            message: Arc::from(source.to_string()),
        }
    }
}

impl fmt::Display for NanoPetEvidenceJournalRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "pet-evidence journal failed: {self:?}")
    }
}

impl std::error::Error for NanoPetEvidenceJournalRuntimeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Encode(source) => Some(source.as_ref()),
            Self::WallClockBeforeUnixEpoch(source) => Some(source.as_ref()),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum NanoPetEvidenceJournalStartError {
    Config(NanoPetEvidenceJournalConfigError),
    Io {
        operation: NanoPetEvidenceJournalIoOperation,
        path: PathBuf,
        source: std::io::Error,
    },
    StateRootNotDirectory {
        path: PathBuf,
    },
    JournalNotRegular {
        path: PathBuf,
    },
    JournalHasMultipleLinks {
        path: PathBuf,
        links: u64,
    },
    JournalWrongOwner {
        path: PathBuf,
        expected_uid: u32,
        observed_uid: u32,
    },
    JournalWrongMode {
        path: PathBuf,
        expected: u32,
        observed: u32,
    },
    JournalLengthOutOfRange {
        path: PathBuf,
        bytes: i128,
    },
    JournalAlreadyAboveLimit {
        path: PathBuf,
        bytes: u64,
        maximum: u64,
    },
    ExistingRecord {
        path: PathBuf,
        record_index: u64,
        source: NanoPetEvidenceDecodeError,
    },
    ExistingRecordCountOverflow {
        path: PathBuf,
    },
    ThreadSpawn(std::io::Error),
}

impl fmt::Display for NanoPetEvidenceJournalStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "pet-evidence journal startup failed: {self:?}")
    }
}

impl std::error::Error for NanoPetEvidenceJournalStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Config(source) => Some(source),
            Self::Io { source, .. } | Self::ThreadSpawn(source) => Some(source),
            Self::ExistingRecord { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoPetEvidenceJournalReadyEvidence {
    path: PathBuf,
    existing_records: u64,
    existing_bytes: u64,
    maximum_bytes: NonZeroU64,
}

impl NanoPetEvidenceJournalReadyEvidence {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub const fn existing_records(&self) -> u64 {
        self.existing_records
    }

    pub const fn existing_bytes(&self) -> u64 {
        self.existing_bytes
    }

    pub const fn maximum_bytes(&self) -> NonZeroU64 {
        self.maximum_bytes
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NanoPetEvidenceJournalStats {
    initial_records: u64,
    appended_records: u64,
    durable_bytes: u64,
}

impl NanoPetEvidenceJournalStats {
    pub const fn initial_records(self) -> u64 {
        self.initial_records
    }

    pub const fn appended_records(self) -> u64 {
        self.appended_records
    }

    pub const fn durable_bytes(self) -> u64 {
        self.durable_bytes
    }
}

#[derive(Clone, Debug)]
pub enum NanoPetEvidenceJournalExit {
    Shutdown(NanoPetEvidenceJournalStats),
    ProducerDisconnected(NanoPetEvidenceJournalStats),
    Fault(NanoPetEvidenceJournalRuntimeError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoPetEvidenceJournalShutdownSignalError {
    TimedOut,
    WriterDisconnected,
}

#[derive(Clone, Debug)]
pub enum NanoPetEvidenceJournalJoinEvidence {
    Joined(NanoPetEvidenceJournalExit),
    ThreadPanicked,
    DetachedAfterTimeout { timeout: Duration },
}

#[derive(Clone, Debug)]
pub struct NanoPetEvidenceJournalShutdownEvidence {
    signal: Result<(), NanoPetEvidenceJournalShutdownSignalError>,
    join: NanoPetEvidenceJournalJoinEvidence,
}

impl NanoPetEvidenceJournalShutdownEvidence {
    pub const fn signal(&self) -> &Result<(), NanoPetEvidenceJournalShutdownSignalError> {
        &self.signal
    }

    pub const fn join(&self) -> &NanoPetEvidenceJournalJoinEvidence {
        &self.join
    }

    pub fn clean(&self) -> bool {
        matches!(self.signal, Ok(()))
            && matches!(
                self.join,
                NanoPetEvidenceJournalJoinEvidence::Joined(NanoPetEvidenceJournalExit::Shutdown(_))
            )
    }
}

enum JournalCommand {
    Append(Vec<u8>),
    Shutdown,
}

struct JournalFile {
    parent: OwnedFd,
    file: File,
    identity: JournalIdentity,
    expected_uid: u32,
    bytes: u64,
    maximum_bytes: NonZeroU64,
    initial_records: u64,
    appended_records: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct JournalIdentity {
    device: i128,
    inode: i128,
}

impl JournalIdentity {
    fn from_stat(stat: &Stat) -> Self {
        Self {
            device: i128::from(stat.st_dev),
            inode: i128::from(stat.st_ino),
        }
    }
}

impl JournalFile {
    fn stats(&self) -> NanoPetEvidenceJournalStats {
        NanoPetEvidenceJournalStats {
            initial_records: self.initial_records,
            appended_records: self.appended_records,
            durable_bytes: self.bytes,
        }
    }

    fn verify_named_identity(&self) -> Result<(), NanoPetEvidenceJournalRuntimeError> {
        let stat = match statat(
            &self.parent,
            NANO_PET_EVIDENCE_JOURNAL_FILENAME,
            AtFlags::SYMLINK_NOFOLLOW,
        ) {
            Ok(stat) => stat,
            Err(Errno::NOENT) => {
                return Err(NanoPetEvidenceJournalRuntimeError::JournalPathMissing);
            }
            Err(source) => {
                return Err(NanoPetEvidenceJournalRuntimeError::io(
                    NanoPetEvidenceJournalIoOperation::InspectJournal,
                    errno_as_io(source),
                ));
            }
        };
        if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile {
            return Err(NanoPetEvidenceJournalRuntimeError::JournalPathNotRegular);
        }
        let observed = JournalIdentity::from_stat(&stat);
        if observed != self.identity {
            return Err(NanoPetEvidenceJournalRuntimeError::JournalPathReplaced {
                expected_device: self.identity.device,
                expected_inode: self.identity.inode,
                observed_device: observed.device,
                observed_inode: observed.inode,
            });
        }
        let links = u64::from(stat.st_nlink);
        if links != 1 {
            return Err(NanoPetEvidenceJournalRuntimeError::JournalHasMultipleLinks { links });
        }
        let observed_uid = stat.st_uid;
        if observed_uid != self.expected_uid {
            return Err(NanoPetEvidenceJournalRuntimeError::JournalOwnerChanged {
                expected_uid: self.expected_uid,
                observed_uid,
            });
        }
        let observed_mode = u32::from(stat.st_mode & 0o777);
        if observed_mode != 0o600 {
            return Err(NanoPetEvidenceJournalRuntimeError::JournalModeChanged {
                expected: 0o600,
                observed: observed_mode,
            });
        }
        Ok(())
    }

    fn append(&mut self, record: &[u8]) -> Result<(), NanoPetEvidenceJournalRuntimeError> {
        self.verify_named_identity()?;
        let record_bytes = u64::try_from(record.len())
            .map_err(|_| NanoPetEvidenceJournalRuntimeError::JournalLengthOverflow)?;
        let attempted = self
            .bytes
            .checked_add(record_bytes)
            .ok_or(NanoPetEvidenceJournalRuntimeError::JournalLengthOverflow)?;
        if attempted > self.maximum_bytes.get() {
            return Err(
                NanoPetEvidenceJournalRuntimeError::JournalByteLimitExceeded {
                    attempted,
                    maximum: self.maximum_bytes.get(),
                },
            );
        }
        self.file.write_all(record).map_err(|source| {
            NanoPetEvidenceJournalRuntimeError::io(
                NanoPetEvidenceJournalIoOperation::AppendRecord,
                source,
            )
        })?;
        self.file.sync_data().map_err(|source| {
            NanoPetEvidenceJournalRuntimeError::io(
                NanoPetEvidenceJournalIoOperation::SynchronizeRecord,
                source,
            )
        })?;
        self.verify_named_identity()?;
        self.bytes = attempted;
        self.appended_records = self
            .appended_records
            .checked_add(1)
            .ok_or(NanoPetEvidenceJournalRuntimeError::JournalLengthOverflow)?;
        Ok(())
    }
}

/// Running writer and its bounded producer edge.
#[must_use = "the pet-evidence writer must be explicitly shut down and joined"]
pub struct NanoPetEvidenceJournal {
    ready: NanoPetEvidenceJournalReadyEvidence,
    sender: Option<Sender<JournalCommand>>,
    fault_receiver: Receiver<NanoPetEvidenceJournalRuntimeError>,
    latched_fault: Option<NanoPetEvidenceJournalRuntimeError>,
    thread: Option<JoinHandle<NanoPetEvidenceJournalExit>>,
}

impl NanoPetEvidenceJournal {
    pub fn start(
        config: NanoPetEvidenceJournalConfig,
    ) -> Result<Self, NanoPetEvidenceJournalStartError> {
        validate_absolute_path(&config.state_root)
            .map_err(NanoPetEvidenceJournalStartError::Config)?;
        let path = config.path();
        let mut journal = open_journal(&config)?;
        let ready = NanoPetEvidenceJournalReadyEvidence {
            path,
            existing_records: journal.initial_records,
            existing_bytes: journal.bytes,
            maximum_bytes: config.maximum_bytes,
        };
        let (sender, receiver) =
            crossbeam_channel::bounded(NANO_PET_EVIDENCE_JOURNAL_QUEUE_CAPACITY);
        let (fault_sender, fault_receiver) = crossbeam_channel::bounded(1);
        let thread = thread::Builder::new()
            .name("kiko-pet-evidence".into())
            .spawn(move || writer_main(&mut journal, &receiver, &fault_sender))
            .map_err(NanoPetEvidenceJournalStartError::ThreadSpawn)?;
        Ok(Self {
            ready,
            sender: Some(sender),
            fault_receiver,
            latched_fault: None,
            thread: Some(thread),
        })
    }

    pub const fn readiness(&self) -> &NanoPetEvidenceJournalReadyEvidence {
        &self.ready
    }

    pub fn poll_fault(&mut self) -> Option<NanoPetEvidenceJournalRuntimeError> {
        if let Some(fault) = &self.latched_fault {
            return Some(fault.clone());
        }
        match self.fault_receiver.try_recv() {
            Ok(fault) => {
                self.latched_fault = Some(fault.clone());
                Some(fault)
            }
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                let fault = NanoPetEvidenceJournalRuntimeError::WriterDisconnected;
                self.latched_fault = Some(fault.clone());
                Some(fault)
            }
        }
    }

    pub fn try_append(
        &mut self,
        evidence: NanoPetEpisodeEvidence,
    ) -> Result<(), NanoPetEvidenceJournalRuntimeError> {
        if let Some(fault) = self.poll_fault() {
            return Err(fault);
        }
        let bytes = evidence
            .encode_ndjson_line()
            .map_err(|source| NanoPetEvidenceJournalRuntimeError::Encode(Arc::new(source)))?;
        match self
            .sender
            .as_ref()
            .ok_or(NanoPetEvidenceJournalRuntimeError::WriterDisconnected)?
            .try_send(JournalCommand::Append(bytes))
        {
            Ok(()) => Ok(()),
            Err(crossbeam_channel::TrySendError::Full(_)) => {
                Err(NanoPetEvidenceJournalRuntimeError::QueueFull)
            }
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => Err(self
                .poll_fault()
                .unwrap_or(NanoPetEvidenceJournalRuntimeError::WriterDisconnected)),
        }
    }

    /// Convert one committed controller summary and one wall-clock sample at
    /// the boundary, then enqueue the exact V1 record without waiting for I/O.
    pub fn try_append_completed(
        &mut self,
        summary: CompliantPetEpisodeSummary,
        completed_wall: SystemTime,
    ) -> Result<(), NanoPetEvidenceJournalRuntimeError> {
        let milliseconds = completed_wall
            .duration_since(UNIX_EPOCH)
            .map_err(|source| {
                NanoPetEvidenceJournalRuntimeError::WallClockBeforeUnixEpoch(Arc::new(source))
            })?
            .as_millis();
        let milliseconds = u64::try_from(milliseconds).map_err(|_| {
            NanoPetEvidenceJournalRuntimeError::WallClockMillisecondsOutOfRange { milliseconds }
        })?;
        let evidence = NanoPetEpisodeEvidence::from_completed_summary(summary, milliseconds)
            .map_err(|source| NanoPetEvidenceJournalRuntimeError::Encode(Arc::new(source)))?;
        self.try_append(evidence)
    }

    pub fn shutdown(mut self, timeout: Duration) -> NanoPetEvidenceJournalShutdownEvidence {
        let sender = self.sender.take().expect("journal shuts down exactly once");
        let signal = match sender.send_timeout(JournalCommand::Shutdown, timeout) {
            Ok(()) => Ok(()),
            Err(crossbeam_channel::SendTimeoutError::Timeout(_)) => {
                Err(NanoPetEvidenceJournalShutdownSignalError::TimedOut)
            }
            Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => {
                Err(NanoPetEvidenceJournalShutdownSignalError::WriterDisconnected)
            }
        };
        drop(sender);
        let join = join_bounded(
            self.thread.take().expect("journal thread is retained"),
            timeout,
        );
        NanoPetEvidenceJournalShutdownEvidence { signal, join }
    }
}

fn writer_main(
    journal: &mut JournalFile,
    receiver: &Receiver<JournalCommand>,
    fault_sender: &Sender<NanoPetEvidenceJournalRuntimeError>,
) -> NanoPetEvidenceJournalExit {
    loop {
        match receiver.recv() {
            Ok(JournalCommand::Append(record)) => {
                if let Err(source) = journal.append(&record) {
                    let _first_fault_published = fault_sender.try_send(source.clone());
                    return NanoPetEvidenceJournalExit::Fault(source);
                }
            }
            Ok(JournalCommand::Shutdown) => {
                return NanoPetEvidenceJournalExit::Shutdown(journal.stats());
            }
            Err(_) => {
                return NanoPetEvidenceJournalExit::ProducerDisconnected(journal.stats());
            }
        }
    }
}

fn join_bounded(
    thread: JoinHandle<NanoPetEvidenceJournalExit>,
    timeout: Duration,
) -> NanoPetEvidenceJournalJoinEvidence {
    let deadline = Instant::now().checked_add(timeout);
    while !thread.is_finished() {
        if deadline.is_none_or(|deadline| Instant::now() >= deadline) {
            return NanoPetEvidenceJournalJoinEvidence::DetachedAfterTimeout { timeout };
        }
        thread::sleep(JOIN_POLL_INTERVAL.min(timeout));
    }
    match thread.join() {
        Ok(exit) => NanoPetEvidenceJournalJoinEvidence::Joined(exit),
        Err(_) => NanoPetEvidenceJournalJoinEvidence::ThreadPanicked,
    }
}

fn open_journal(
    config: &NanoPetEvidenceJournalConfig,
) -> Result<JournalFile, NanoPetEvidenceJournalStartError> {
    let root = open_absolute_directory_nofollow(&config.state_root)?;
    let root_stat = fstat(&root).map_err(|source| NanoPetEvidenceJournalStartError::Io {
        operation: NanoPetEvidenceJournalIoOperation::InspectJournal,
        path: config.state_root.clone(),
        source: errno_as_io(source),
    })?;
    if FileType::from_raw_mode(root_stat.st_mode) != FileType::Directory {
        return Err(NanoPetEvidenceJournalStartError::StateRootNotDirectory {
            path: config.state_root.clone(),
        });
    }
    let name = OsStr::new(NANO_PET_EVIDENCE_JOURNAL_FILENAME);
    let existed = match statat(&root, name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(_) => true,
        Err(Errno::NOENT) => false,
        Err(source) => {
            return Err(NanoPetEvidenceJournalStartError::Io {
                operation: NanoPetEvidenceJournalIoOperation::InspectJournal,
                path: config.path(),
                source: errno_as_io(source),
            });
        }
    };
    let descriptor = openat(
        &root,
        name,
        OFlags::RDWR | OFlags::APPEND | OFlags::CREATE | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        JOURNAL_MODE,
    )
    .map_err(|source| NanoPetEvidenceJournalStartError::Io {
        operation: NanoPetEvidenceJournalIoOperation::OpenJournal,
        path: config.path(),
        source: errno_as_io(source),
    })?;
    let stat = fstat(&descriptor).map_err(|source| NanoPetEvidenceJournalStartError::Io {
        operation: NanoPetEvidenceJournalIoOperation::InspectJournal,
        path: config.path(),
        source: errno_as_io(source),
    })?;
    require_admitted_journal_stat(config, &stat)?;
    let named = statat(&root, name, AtFlags::SYMLINK_NOFOLLOW).map_err(|source| {
        NanoPetEvidenceJournalStartError::Io {
            operation: NanoPetEvidenceJournalIoOperation::InspectJournal,
            path: config.path(),
            source: errno_as_io(source),
        }
    })?;
    if JournalIdentity::from_stat(&named) != JournalIdentity::from_stat(&stat) {
        return Err(NanoPetEvidenceJournalStartError::Io {
            operation: NanoPetEvidenceJournalIoOperation::InspectJournal,
            path: config.path(),
            source: std::io::Error::other("journal path changed while it was opened"),
        });
    }
    if !existed {
        fsync(&root).map_err(|source| NanoPetEvidenceJournalStartError::Io {
            operation: NanoPetEvidenceJournalIoOperation::SynchronizeStateRoot,
            path: config.state_root.clone(),
            source: errno_as_io(source),
        })?;
    }
    let file = File::from(descriptor);
    let bytes = stat_size(config, &stat)?;
    if bytes > config.maximum_bytes.get() {
        return Err(NanoPetEvidenceJournalStartError::JournalAlreadyAboveLimit {
            path: config.path(),
            bytes,
            maximum: config.maximum_bytes.get(),
        });
    }
    let initial_records = scan_existing_records(config, &file)?;
    Ok(JournalFile {
        parent: root,
        file,
        identity: JournalIdentity::from_stat(&stat),
        expected_uid: rustix::process::geteuid().as_raw(),
        bytes,
        maximum_bytes: config.maximum_bytes,
        initial_records,
        appended_records: 0,
    })
}

fn scan_existing_records(
    config: &NanoPetEvidenceJournalConfig,
    file: &File,
) -> Result<u64, NanoPetEvidenceJournalStartError> {
    let reader = file
        .try_clone()
        .map_err(|source| NanoPetEvidenceJournalStartError::Io {
            operation: NanoPetEvidenceJournalIoOperation::ReadExistingJournal,
            path: config.path(),
            source,
        })?;
    let mut reader = BufReader::new(reader);
    let mut record = Vec::with_capacity(MAX_NANO_PET_EVIDENCE_RECORD_BYTES);
    let mut count = 0_u64;
    loop {
        record.clear();
        let read = reader.read_until(b'\n', &mut record).map_err(|source| {
            NanoPetEvidenceJournalStartError::Io {
                operation: NanoPetEvidenceJournalIoOperation::ReadExistingJournal,
                path: config.path(),
                source,
            }
        })?;
        if read == 0 {
            return Ok(count);
        }
        NanoPetEpisodeEvidence::parse_ndjson_line(&record).map_err(|source| {
            NanoPetEvidenceJournalStartError::ExistingRecord {
                path: config.path(),
                record_index: count,
                source,
            }
        })?;
        count = count.checked_add(1).ok_or_else(|| {
            NanoPetEvidenceJournalStartError::ExistingRecordCountOverflow {
                path: config.path(),
            }
        })?;
    }
}

fn require_admitted_journal_stat(
    config: &NanoPetEvidenceJournalConfig,
    stat: &Stat,
) -> Result<(), NanoPetEvidenceJournalStartError> {
    let path = config.path();
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile {
        return Err(NanoPetEvidenceJournalStartError::JournalNotRegular { path });
    }
    let links = u64::from(stat.st_nlink);
    if links != 1 {
        return Err(NanoPetEvidenceJournalStartError::JournalHasMultipleLinks { path, links });
    }
    let expected_uid = rustix::process::geteuid().as_raw();
    if stat.st_uid != expected_uid {
        return Err(NanoPetEvidenceJournalStartError::JournalWrongOwner {
            path,
            expected_uid,
            observed_uid: stat.st_uid,
        });
    }
    let observed_mode = u32::from(stat.st_mode & 0o777);
    if observed_mode != 0o600 {
        return Err(NanoPetEvidenceJournalStartError::JournalWrongMode {
            path,
            expected: 0o600,
            observed: observed_mode,
        });
    }
    Ok(())
}

fn stat_size(
    config: &NanoPetEvidenceJournalConfig,
    stat: &Stat,
) -> Result<u64, NanoPetEvidenceJournalStartError> {
    u64::try_from(stat.st_size).map_err(|_| {
        NanoPetEvidenceJournalStartError::JournalLengthOutOfRange {
            path: config.path(),
            bytes: i128::from(stat.st_size),
        }
    })
}

fn open_absolute_directory_nofollow(
    path: &Path,
) -> Result<OwnedFd, NanoPetEvidenceJournalStartError> {
    let flags =
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK;
    let mut current =
        open("/", flags, Mode::empty()).map_err(|source| NanoPetEvidenceJournalStartError::Io {
            operation: NanoPetEvidenceJournalIoOperation::OpenStateRootComponent {
                component_index: 0,
            },
            path: PathBuf::from("/"),
            source: errno_as_io(source),
        })?;
    let mut opened = PathBuf::from("/");
    for (index, component) in path.components().enumerate() {
        let Component::Normal(name) = component else {
            continue;
        };
        opened.push(name);
        current = openat(&current, name, flags, Mode::empty()).map_err(|source| {
            NanoPetEvidenceJournalStartError::Io {
                operation: NanoPetEvidenceJournalIoOperation::OpenStateRootComponent {
                    component_index: index,
                },
                path: opened.clone(),
                source: errno_as_io(source),
            }
        })?;
    }
    Ok(current)
}

fn validate_absolute_path(path: &Path) -> Result<(), NanoPetEvidenceJournalConfigError> {
    if !path.is_absolute() {
        return Err(NanoPetEvidenceJournalConfigError::NotAbsolute {
            path: path.to_path_buf(),
        });
    }
    if path.components().any(|component| {
        matches!(
            component,
            Component::CurDir | Component::ParentDir | Component::Prefix(_)
        )
    }) {
        return Err(NanoPetEvidenceJournalConfigError::NonCanonicalComponent {
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

fn errno_as_io(source: Errno) -> std::io::Error {
    std::io::Error::from_raw_os_error(source.raw_os_error())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::os::unix::fs::{PermissionsExt, symlink};
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn create(label: &str) -> Self {
            let base = fs::canonicalize(std::env::temp_dir()).expect("canonical temp root");
            for _ in 0..1_000 {
                let serial = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
                let path = base.join(format!(
                    "kiko-pet-journal-{label}-{}-{serial}",
                    std::process::id()
                ));
                match fs::create_dir(&path) {
                    Ok(()) => return Self(path),
                    Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => {}
                    Err(source) => panic!("create test state root: {source}"),
                }
            }
            panic!("could not allocate test state root")
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn v1_record() -> NanoPetEpisodeEvidence {
        let line = concat!(
            r#"{"schema_version":1,"wall":1787759000.25,"wall_unix_ms":1787759000250,"completed_monotonic_ns":14200000000,"episode":{"started_at":10.0,"started_monotonic_ns":10000000000,"completed_monotonic_ns":14200000000,"yield_entries":2,"samples":41,"peak_residual":[12,4,3,2],"delta_accum":240,"delta_samples":40,"reached_rest":true,"reached_comfy":false,"tap":false,"duration_s":4.2,"duration_ns":4200000000,"mean_delta":6.0}}"#,
            "\n"
        );
        NanoPetEpisodeEvidence::parse_ndjson_line(line.as_bytes()).expect("V1 fixture")
    }

    #[test]
    fn writer_appends_syncs_and_joins_with_exact_evidence() {
        let root = TestDirectory::create("append");
        let config = NanoPetEvidenceJournalConfig::for_state_root(&root.0).unwrap();
        let path = config.path();
        let mut journal = NanoPetEvidenceJournal::start(config).expect("journal startup");
        assert_eq!(journal.readiness().existing_records(), 0);
        journal.try_append(v1_record()).expect("bounded enqueue");
        let shutdown = journal.shutdown(Duration::from_secs(1));
        assert!(shutdown.clean());
        let NanoPetEvidenceJournalJoinEvidence::Joined(NanoPetEvidenceJournalExit::Shutdown(stats)) =
            shutdown.join()
        else {
            panic!("unexpected shutdown evidence: {shutdown:?}");
        };
        assert_eq!(stats.initial_records(), 0);
        assert_eq!(stats.appended_records(), 1);

        let bytes = fs::read(&path).expect("durable journal");
        NanoPetEpisodeEvidence::parse_ndjson_line(&bytes).expect("written record parses");
        assert_eq!(
            fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }

    #[test]
    fn startup_validates_every_existing_record_and_retains_legacy_corpus() {
        let root = TestDirectory::create("legacy");
        let config = NanoPetEvidenceJournalConfig::for_state_root(&root.0).unwrap();
        let path = config.path();
        let legacy = concat!(
            r#"{"wall":1787759000.25,"episode":{"started_at":1234.5,"yield_entries":1,"samples":41,"peak_residual":[12,4,3,2],"delta_accum":20.0,"delta_samples":40,"reached_rest":true,"reached_comfy":true,"tap":false,"duration_s":4.2,"mean_delta":0.5}}"#,
            "\n"
        );
        fs::write(&path, legacy).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();

        let mut journal = NanoPetEvidenceJournal::start(config).expect("legacy admission");
        assert_eq!(journal.readiness().existing_records(), 1);
        journal.try_append(v1_record()).expect("V1 append");
        assert!(journal.shutdown(Duration::from_secs(1)).clean());

        let bytes = fs::read(path).unwrap();
        let mut records = bytes.split_inclusive(|byte| *byte == b'\n');
        assert_eq!(
            NanoPetEpisodeEvidence::parse_ndjson_line(records.next().unwrap())
                .unwrap()
                .format(),
            super::super::NanoPetEvidenceFormat::FableLegacy
        );
        assert_eq!(
            NanoPetEpisodeEvidence::parse_ndjson_line(records.next().unwrap())
                .unwrap()
                .format(),
            super::super::NanoPetEvidenceFormat::NanoV1
        );
        assert!(records.next().is_none());
    }

    #[test]
    fn symlink_truncated_corpus_and_size_limit_fail_closed() {
        let root = TestDirectory::create("reject");
        let config = NanoPetEvidenceJournalConfig::for_state_root(&root.0).unwrap();
        let path = config.path();
        let target = root.0.join("target");
        fs::write(&target, []).unwrap();
        symlink(&target, &path).unwrap();
        assert!(matches!(
            NanoPetEvidenceJournal::start(config.clone()),
            Err(NanoPetEvidenceJournalStartError::Io {
                operation: NanoPetEvidenceJournalIoOperation::OpenJournal,
                ..
            })
        ));
        fs::remove_file(&path).unwrap();
        fs::write(&path, b"{}").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        assert!(matches!(
            NanoPetEvidenceJournal::start(config.clone()),
            Err(NanoPetEvidenceJournalStartError::ExistingRecord { .. })
        ));
        fs::remove_file(&path).unwrap();
        let tiny = config.with_maximum_bytes(NonZeroU64::new(1).unwrap());
        let mut journal = NanoPetEvidenceJournal::start(tiny).expect("empty tiny journal");
        journal
            .try_append(v1_record())
            .expect("enqueue is nonblocking");
        let shutdown = journal.shutdown(Duration::from_secs(1));
        assert!(matches!(
            shutdown.join(),
            NanoPetEvidenceJournalJoinEvidence::Joined(NanoPetEvidenceJournalExit::Fault(
                NanoPetEvidenceJournalRuntimeError::JournalByteLimitExceeded { .. }
            ))
        ));
    }
}
