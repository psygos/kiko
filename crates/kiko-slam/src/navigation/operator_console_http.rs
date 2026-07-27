//! Loopback-only HTTP task for [`super::OperatorConsoleHandle`].
//!
//! Static assets are public to the local machine. Every status or control API
//! requires an injected 256-bit per-boot capability, and every session gets a
//! second independent capability. Neither capability is accepted in a URL.

use std::convert::Infallible;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::io::Write;
use std::net::SocketAddr;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use bytes::Bytes;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore, oneshot};
use warp::http::StatusCode;
use warp::hyper::Body;
use warp::{Filter, Reply};

use super::{
    ConsoleDownstreamRequestId, ConsoleIdempotencyKey, ConsoleIntentRequestDto,
    ConsoleSessionCapability, ConsoleSessionId, ConsoleSourceKind, ConsoleSourceSequence,
    MAX_OPERATOR_CONSOLE_REQUEST_BYTES, OperatorConsoleBind, OperatorConsoleHandle,
    OperatorConsoleResponseRecord, OperatorConsoleSnapshot, OperatorConsoleSubmitError,
    OperatorConsoleSubmitOutcome,
};
use crate::HostMonotonicTimestamp;

pub(super) const INDEX_HTML: &str = include_str!("../operator-console/index.html");
pub(super) const STYLES_CSS: &str = include_str!("../operator-console/styles.css");
pub(super) const VIEW_MODEL_JS: &str = include_str!("../operator-console/view-model.js");
pub(super) const APP_JS: &str = include_str!("../operator-console/app.js");
const CAPABILITY_BYTES: usize = 32;
const CAPABILITY_HEX_BYTES: usize = CAPABILITY_BYTES * 2;
const MAX_DEADMAN_TICK_MS: u64 = 100;
const MIN_DEADMAN_TICK_MS: u64 = 5;
const MAX_CONCURRENT_API_REQUESTS: usize = 32;
const HTTP_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(3);
const HTTP_FORCE_SHUTDOWN_AFTER: Duration = Duration::from_secs(1);
#[cfg(unix)]
const PRIVATE_CAPABILITY_PARENT_MODE: u32 = 0o700;
#[cfg(unix)]
const PRIVATE_CAPABILITY_FILE_MODE: u32 = 0o600;

/// Secret intentionally redacted from `Debug`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct OperatorConsoleAccessCapability([u8; CAPABILITY_BYTES]);

impl OperatorConsoleAccessCapability {
    pub fn generate() -> Result<Self, OperatorConsoleCapabilityError> {
        let mut bytes = [0_u8; CAPABILITY_BYTES];
        getrandom::fill(&mut bytes).map_err(OperatorConsoleCapabilityError::Random)?;
        Self::parse(bytes)
    }

    pub fn parse(bytes: [u8; CAPABILITY_BYTES]) -> Result<Self, OperatorConsoleCapabilityError> {
        if bytes == [0; CAPABILITY_BYTES] {
            return Err(OperatorConsoleCapabilityError::AllZero);
        }
        Ok(Self(bytes))
    }

    /// Atomically publish a fresh per-process capability in an already-private
    /// directory. An existing safe owned target is replaced, never read or
    /// reused. Publication and cleanup are descriptor-relative and durable.
    #[cfg(unix)]
    pub fn generate_and_persist_new(
        path: &Path,
    ) -> Result<OperatorConsolePersistedAccessCapability, OperatorConsoleCapabilityPersistError>
    {
        persist_fresh_capability(path)
    }

    pub(super) fn parse_hex(raw: &str) -> Option<Self> {
        if raw.len() != CAPABILITY_HEX_BYTES || !raw.is_ascii() {
            return None;
        }
        let mut bytes = [0_u8; CAPABILITY_BYTES];
        let raw = raw.as_bytes();
        for (index, output) in bytes.iter_mut().enumerate() {
            let high = decode_hex(raw[index * 2])?;
            let low = decode_hex(raw[index * 2 + 1])?;
            *output = (high << 4) | low;
        }
        Self::parse(bytes).ok()
    }

    pub(super) fn constant_time_matches(self, candidate: Self) -> bool {
        let mut difference = 0_u8;
        for (expected, actual) in self.0.iter().zip(candidate.0.iter()) {
            difference |= expected ^ actual;
        }
        difference == 0
    }

    #[cfg(feature = "nano-wheels-off-qualification")]
    pub(super) const fn as_bytes_for_session(&self) -> &[u8; CAPABILITY_BYTES] {
        &self.0
    }

    pub(super) fn to_hex(self) -> String {
        encode_hex(&self.0)
    }
}

impl fmt::Debug for OperatorConsoleAccessCapability {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("OperatorConsoleAccessCapability([REDACTED])")
    }
}

/// Linear owner of one exact published capability file.
///
/// The HTTP config receives a copy of [`Self::access_capability`]. After the
/// HTTP server has stopped, production shutdown consumes this owner with
/// [`Self::cleanup`] and records the returned inode-guarded evidence. Dropping
/// it also attempts the same exact cleanup, but explicit cleanup is required
/// when the caller needs reportable evidence.
#[cfg(unix)]
pub struct OperatorConsolePersistedAccessCapability {
    capability: OperatorConsoleAccessCapability,
    parent: rustix::fd::OwnedFd,
    target_path: PathBuf,
    target_name: OsString,
    identity: CapabilityFileIdentity,
    terminal: bool,
}

#[cfg(unix)]
impl OperatorConsolePersistedAccessCapability {
    pub const fn access_capability(&self) -> OperatorConsoleAccessCapability {
        self.capability
    }

    pub fn target_path(&self) -> &Path {
        &self.target_path
    }

    pub fn cleanup(mut self) -> OperatorConsoleCapabilityCleanupEvidence {
        let evidence =
            cleanup_exact_capability_entry(&self.parent, &self.target_name, self.identity);
        self.terminal = true;
        evidence
    }
}

#[cfg(unix)]
impl fmt::Debug for OperatorConsolePersistedAccessCapability {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OperatorConsolePersistedAccessCapability")
            .field("capability", &"[REDACTED]")
            .field("target_path", &self.target_path)
            .field("identity", &self.identity)
            .field("terminal", &self.terminal)
            .finish()
    }
}

#[cfg(unix)]
impl Drop for OperatorConsolePersistedAccessCapability {
    fn drop(&mut self) {
        if !self.terminal {
            let _ = cleanup_exact_capability_entry(&self.parent, &self.target_name, self.identity);
            self.terminal = true;
        }
    }
}

#[derive(Debug)]
pub enum OperatorConsoleCapabilityError {
    Random(getrandom::Error),
    AllZero,
}

impl fmt::Display for OperatorConsoleCapabilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("could not create a strong operator-console capability")
    }
}

impl std::error::Error for OperatorConsoleCapabilityError {}

#[derive(Debug)]
pub enum OperatorConsoleCapabilityPersistError {
    Capability(OperatorConsoleCapabilityError),
    InvalidAbsoluteTarget(PathBuf),
    ParentNotPrivate {
        path: PathBuf,
        observed_mode: u32,
    },
    ParentOwnerMismatch {
        path: PathBuf,
        expected_uid: u32,
        observed_uid: u32,
    },
    UnsafeExistingTarget {
        path: PathBuf,
        reason: OperatorConsoleCapabilityTargetSafetyError,
        cleanup: OperatorConsoleCapabilityCleanupEvidence,
    },
    Io {
        operation: OperatorConsoleCapabilityFilesystemOperation,
        source: std::io::Error,
        cleanup: OperatorConsoleCapabilityCleanupEvidence,
    },
}

impl fmt::Display for OperatorConsoleCapabilityPersistError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("could not persist the operator-console capability")
    }
}

impl std::error::Error for OperatorConsoleCapabilityPersistError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Capability(source) => Some(source),
            Self::Io { source, .. } => Some(source),
            Self::InvalidAbsoluteTarget(_)
            | Self::ParentNotPrivate { .. }
            | Self::ParentOwnerMismatch { .. }
            | Self::UnsafeExistingTarget { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorConsoleCapabilityFilesystemOperation {
    OpenParent,
    InspectParent,
    InspectTarget,
    GenerateTemporaryName,
    CreateTemporary,
    SetTemporaryMode,
    InspectTemporary,
    WriteTemporary,
    SyncTemporary,
    Publish,
    InspectPublished,
    SyncParent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorConsoleCapabilityTargetSafetyError {
    NotRegularFile,
    OwnerMismatch {
        expected_uid: u32,
        observed_uid: u32,
    },
    ModeMismatch {
        observed_mode: u32,
    },
    MultipleHardLinks {
        observed_links: u64,
    },
    CrossDevice,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorConsoleCapabilityCleanupEvidence {
    NotNeeded,
    EntryAlreadyAbsent,
    ExactEntryRemovedAndParentSynced,
    RefusedIdentityMismatch,
    InspectFailed { raw_os_error: i32 },
    RemoveFailed { raw_os_error: i32 },
    ParentSyncFailed { raw_os_error: i32 },
}

#[cfg(unix)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CapabilityFileIdentity {
    device: i128,
    inode: u128,
}

#[cfg(unix)]
fn capability_file_identity(stat: &rustix::fs::Stat) -> CapabilityFileIdentity {
    CapabilityFileIdentity {
        device: stat.st_dev.into(),
        inode: stat.st_ino.into(),
    }
}

#[cfg(unix)]
fn errno_as_io(source: rustix::io::Errno) -> std::io::Error {
    std::io::Error::from_raw_os_error(source.raw_os_error())
}

#[cfg(unix)]
fn capability_io_error(
    operation: OperatorConsoleCapabilityFilesystemOperation,
    source: std::io::Error,
    cleanup: OperatorConsoleCapabilityCleanupEvidence,
) -> OperatorConsoleCapabilityPersistError {
    OperatorConsoleCapabilityPersistError::Io {
        operation,
        source,
        cleanup,
    }
}

#[cfg(unix)]
fn inspect_safe_existing_capability(
    parent: &rustix::fd::OwnedFd,
    target_name: &OsStr,
    target_path: &Path,
    parent_device: i128,
    expected_uid: u32,
) -> Result<(), OperatorConsoleCapabilityPersistError> {
    use rustix::fs::{AtFlags, FileType, statat};
    use rustix::io::Errno;

    let stat = match statat(parent, target_name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(stat) => stat,
        Err(Errno::NOENT) => return Ok(()),
        Err(source) => {
            return Err(capability_io_error(
                OperatorConsoleCapabilityFilesystemOperation::InspectTarget,
                errno_as_io(source),
                OperatorConsoleCapabilityCleanupEvidence::NotNeeded,
            ));
        }
    };
    let reason = if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile {
        Some(OperatorConsoleCapabilityTargetSafetyError::NotRegularFile)
    } else if stat.st_uid != expected_uid {
        Some(OperatorConsoleCapabilityTargetSafetyError::OwnerMismatch {
            expected_uid,
            observed_uid: stat.st_uid,
        })
    } else {
        let observed_mode = u32::from(stat.st_mode) & 0o7777;
        if observed_mode != PRIVATE_CAPABILITY_FILE_MODE {
            Some(OperatorConsoleCapabilityTargetSafetyError::ModeMismatch { observed_mode })
        } else if stat.st_nlink != 1 {
            Some(
                OperatorConsoleCapabilityTargetSafetyError::MultipleHardLinks {
                    observed_links: stat.st_nlink as u64,
                },
            )
        } else if i128::from(stat.st_dev) != parent_device {
            Some(OperatorConsoleCapabilityTargetSafetyError::CrossDevice)
        } else {
            None
        }
    };
    match reason {
        Some(reason) => Err(
            OperatorConsoleCapabilityPersistError::UnsafeExistingTarget {
                path: target_path.to_path_buf(),
                reason,
                cleanup: OperatorConsoleCapabilityCleanupEvidence::NotNeeded,
            },
        ),
        None => Ok(()),
    }
}

#[cfg(unix)]
fn cleanup_exact_capability_entry(
    parent: &rustix::fd::OwnedFd,
    name: &OsStr,
    expected: CapabilityFileIdentity,
) -> OperatorConsoleCapabilityCleanupEvidence {
    use rustix::fs::{AtFlags, fsync, statat, unlinkat};
    use rustix::io::Errno;
    let observed = match statat(parent, name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(stat) => capability_file_identity(&stat),
        Err(Errno::NOENT) => {
            return OperatorConsoleCapabilityCleanupEvidence::EntryAlreadyAbsent;
        }
        Err(source) => {
            return OperatorConsoleCapabilityCleanupEvidence::InspectFailed {
                raw_os_error: source.raw_os_error(),
            };
        }
    };
    if observed != expected {
        return OperatorConsoleCapabilityCleanupEvidence::RefusedIdentityMismatch;
    }
    if let Err(source) = unlinkat(parent, name, AtFlags::empty()) {
        return OperatorConsoleCapabilityCleanupEvidence::RemoveFailed {
            raw_os_error: source.raw_os_error(),
        };
    }
    match fsync(parent) {
        Ok(()) => OperatorConsoleCapabilityCleanupEvidence::ExactEntryRemovedAndParentSynced,
        Err(source) => OperatorConsoleCapabilityCleanupEvidence::ParentSyncFailed {
            raw_os_error: source.raw_os_error(),
        },
    }
}

#[cfg(unix)]
fn persist_fresh_capability(
    path: &Path,
) -> Result<OperatorConsolePersistedAccessCapability, OperatorConsoleCapabilityPersistError> {
    use rustix::fs::{
        AtFlags, FileType, Mode, OFlags, fchmod, fstat, fsync, open, openat, renameat, statat,
    };

    if !path.is_absolute() {
        return Err(
            OperatorConsoleCapabilityPersistError::InvalidAbsoluteTarget(path.to_path_buf()),
        );
    }
    let Some(parent_path) = path.parent() else {
        return Err(
            OperatorConsoleCapabilityPersistError::InvalidAbsoluteTarget(path.to_path_buf()),
        );
    };
    let Some(target_name) = path.file_name() else {
        return Err(
            OperatorConsoleCapabilityPersistError::InvalidAbsoluteTarget(path.to_path_buf()),
        );
    };
    let parent = open(
        parent_path,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK,
        Mode::empty(),
    )
    .map_err(|source| {
        capability_io_error(
            OperatorConsoleCapabilityFilesystemOperation::OpenParent,
            errno_as_io(source),
            OperatorConsoleCapabilityCleanupEvidence::NotNeeded,
        )
    })?;
    let parent_stat = fstat(&parent).map_err(|source| {
        capability_io_error(
            OperatorConsoleCapabilityFilesystemOperation::InspectParent,
            errno_as_io(source),
            OperatorConsoleCapabilityCleanupEvidence::NotNeeded,
        )
    })?;
    if FileType::from_raw_mode(parent_stat.st_mode) != FileType::Directory {
        return Err(capability_io_error(
            OperatorConsoleCapabilityFilesystemOperation::InspectParent,
            std::io::Error::other("capability parent is not a directory"),
            OperatorConsoleCapabilityCleanupEvidence::NotNeeded,
        ));
    }
    let observed_parent_mode = u32::from(parent_stat.st_mode) & 0o7777;
    if observed_parent_mode != PRIVATE_CAPABILITY_PARENT_MODE {
        return Err(OperatorConsoleCapabilityPersistError::ParentNotPrivate {
            path: parent_path.to_path_buf(),
            observed_mode: observed_parent_mode,
        });
    }
    let expected_uid = rustix::process::geteuid().as_raw();
    if parent_stat.st_uid != expected_uid {
        return Err(OperatorConsoleCapabilityPersistError::ParentOwnerMismatch {
            path: parent_path.to_path_buf(),
            expected_uid,
            observed_uid: parent_stat.st_uid,
        });
    }
    let parent_device = i128::from(parent_stat.st_dev);
    inspect_safe_existing_capability(&parent, target_name, path, parent_device, expected_uid)?;

    let capability = OperatorConsoleAccessCapability::generate()
        .map_err(OperatorConsoleCapabilityPersistError::Capability)?;
    let mut random_name = [0_u8; 16];
    getrandom::fill(&mut random_name).map_err(|source| {
        OperatorConsoleCapabilityPersistError::Capability(OperatorConsoleCapabilityError::Random(
            source,
        ))
    })?;
    let temporary_name = OsString::from(format!(
        ".operator-console-capability-{}.tmp",
        encode_hex(&random_name)
    ));
    let temporary = openat(
        &parent,
        &temporary_name,
        OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::from_raw_mode(PRIVATE_CAPABILITY_FILE_MODE as _),
    )
    .map_err(|source| {
        capability_io_error(
            OperatorConsoleCapabilityFilesystemOperation::CreateTemporary,
            errno_as_io(source),
            OperatorConsoleCapabilityCleanupEvidence::NotNeeded,
        )
    })?;
    let initial_stat = fstat(&temporary).map_err(|source| {
        capability_io_error(
            OperatorConsoleCapabilityFilesystemOperation::InspectTemporary,
            errno_as_io(source),
            OperatorConsoleCapabilityCleanupEvidence::NotNeeded,
        )
    })?;
    let temporary_identity = capability_file_identity(&initial_stat);
    let fail_with_temp = |operation: OperatorConsoleCapabilityFilesystemOperation,
                          source: std::io::Error| {
        let cleanup = cleanup_exact_capability_entry(&parent, &temporary_name, temporary_identity);
        capability_io_error(operation, source, cleanup)
    };
    fchmod(
        &temporary,
        Mode::from_raw_mode(PRIVATE_CAPABILITY_FILE_MODE as _),
    )
    .map_err(|source| {
        fail_with_temp(
            OperatorConsoleCapabilityFilesystemOperation::SetTemporaryMode,
            errno_as_io(source),
        )
    })?;
    let temporary_stat = fstat(&temporary).map_err(|source| {
        fail_with_temp(
            OperatorConsoleCapabilityFilesystemOperation::InspectTemporary,
            errno_as_io(source),
        )
    })?;
    let temporary_mode = u32::from(temporary_stat.st_mode) & 0o7777;
    if FileType::from_raw_mode(temporary_stat.st_mode) != FileType::RegularFile
        || temporary_stat.st_uid != expected_uid
        || temporary_stat.st_nlink != 1
        || temporary_mode != PRIVATE_CAPABILITY_FILE_MODE
        || i128::from(temporary_stat.st_dev) != parent_device
        || capability_file_identity(&temporary_stat) != temporary_identity
    {
        return Err(fail_with_temp(
            OperatorConsoleCapabilityFilesystemOperation::InspectTemporary,
            std::io::Error::other("temporary capability object failed identity checks"),
        ));
    }
    let mut temporary_file: std::fs::File = temporary.into();
    temporary_file
        .write_all(capability.to_hex().as_bytes())
        .and_then(|()| temporary_file.write_all(b"\n"))
        .map_err(|source| {
            fail_with_temp(
                OperatorConsoleCapabilityFilesystemOperation::WriteTemporary,
                source,
            )
        })?;
    temporary_file.sync_all().map_err(|source| {
        fail_with_temp(
            OperatorConsoleCapabilityFilesystemOperation::SyncTemporary,
            source,
        )
    })?;

    // Re-check immediately before replacement. The admitted private parent is
    // the trust boundary; unsafe symlinks, devices, and hard links are never
    // overwritten.
    inspect_safe_existing_capability(&parent, target_name, path, parent_device, expected_uid)
        .map_err(|error| {
            let cleanup =
                cleanup_exact_capability_entry(&parent, &temporary_name, temporary_identity);
            match error {
                OperatorConsoleCapabilityPersistError::Io {
                    operation, source, ..
                } => OperatorConsoleCapabilityPersistError::Io {
                    operation,
                    source,
                    cleanup,
                },
                OperatorConsoleCapabilityPersistError::UnsafeExistingTarget {
                    path,
                    reason,
                    ..
                } => OperatorConsoleCapabilityPersistError::UnsafeExistingTarget {
                    path,
                    reason,
                    cleanup,
                },
                other => other,
            }
        })?;
    renameat(&parent, &temporary_name, &parent, target_name).map_err(|source| {
        fail_with_temp(
            OperatorConsoleCapabilityFilesystemOperation::Publish,
            errno_as_io(source),
        )
    })?;
    let published_stat =
        statat(&parent, target_name, AtFlags::SYMLINK_NOFOLLOW).map_err(|source| {
            capability_io_error(
                OperatorConsoleCapabilityFilesystemOperation::InspectPublished,
                errno_as_io(source),
                cleanup_exact_capability_entry(&parent, target_name, temporary_identity),
            )
        })?;
    if capability_file_identity(&published_stat) != temporary_identity {
        return Err(capability_io_error(
            OperatorConsoleCapabilityFilesystemOperation::InspectPublished,
            std::io::Error::other("published capability identity changed"),
            OperatorConsoleCapabilityCleanupEvidence::RefusedIdentityMismatch,
        ));
    }
    fsync(&parent).map_err(|source| {
        capability_io_error(
            OperatorConsoleCapabilityFilesystemOperation::SyncParent,
            errno_as_io(source),
            cleanup_exact_capability_entry(&parent, target_name, temporary_identity),
        )
    })?;
    Ok(OperatorConsolePersistedAccessCapability {
        capability,
        parent,
        target_path: path.to_path_buf(),
        target_name: target_name.to_os_string(),
        identity: temporary_identity,
        terminal: false,
    })
}

fn decode_hex(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

pub(super) fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

/// Nonblocking console surface consumed by the HTTP task.
pub trait OperatorConsoleHttpBackend: Send + Sync + 'static {
    fn open_session(
        &self,
        source: ConsoleSourceKind,
        capability: ConsoleSessionCapability,
        received_at: HostMonotonicTimestamp,
    ) -> Result<ConsoleSessionId, OperatorConsoleSubmitError>;

    #[allow(clippy::too_many_arguments)]
    fn submit(
        &self,
        session_id: ConsoleSessionId,
        capability: ConsoleSessionCapability,
        source_sequence: ConsoleSourceSequence,
        idempotency_key: ConsoleIdempotencyKey,
        intent: super::OperatorConsoleIntent,
        received_at: HostMonotonicTimestamp,
    ) -> Result<OperatorConsoleSubmitOutcome, OperatorConsoleSubmitError>;

    fn close_session(
        &self,
        session_id: ConsoleSessionId,
        capability: ConsoleSessionCapability,
    ) -> Result<bool, OperatorConsoleSubmitError>;

    fn session_capability_matches(
        &self,
        session_id: ConsoleSessionId,
        capability: ConsoleSessionCapability,
    ) -> bool;

    fn tick_deadman(&self, now: HostMonotonicTimestamp)
    -> Result<bool, OperatorConsoleSubmitError>;

    fn fail_closed(&self);

    fn latest_snapshot(&self) -> Arc<OperatorConsoleSnapshot>;

    fn exact_grid(
        &self,
        map_epoch_id: NonZeroU64,
        revision: u64,
    ) -> Option<Arc<super::ConsoleOccupancyGrid>>;

    fn observe_response_record(
        &self,
        id: ConsoleDownstreamRequestId,
        source_session_id: ConsoleSessionId,
    ) -> Option<OperatorConsoleResponseRecord>;
}

impl OperatorConsoleHttpBackend for OperatorConsoleHandle {
    fn open_session(
        &self,
        source: ConsoleSourceKind,
        capability: ConsoleSessionCapability,
        received_at: HostMonotonicTimestamp,
    ) -> Result<ConsoleSessionId, OperatorConsoleSubmitError> {
        self.open_session(source, capability, received_at)
    }

    fn session_capability_matches(
        &self,
        session_id: ConsoleSessionId,
        capability: ConsoleSessionCapability,
    ) -> bool {
        self.session_capability_matches(session_id, capability)
    }

    fn submit(
        &self,
        session_id: ConsoleSessionId,
        capability: ConsoleSessionCapability,
        source_sequence: ConsoleSourceSequence,
        idempotency_key: ConsoleIdempotencyKey,
        intent: super::OperatorConsoleIntent,
        received_at: HostMonotonicTimestamp,
    ) -> Result<OperatorConsoleSubmitOutcome, OperatorConsoleSubmitError> {
        self.submit(
            session_id,
            capability,
            source_sequence,
            idempotency_key,
            intent,
            received_at,
        )
    }

    fn close_session(
        &self,
        session_id: ConsoleSessionId,
        capability: ConsoleSessionCapability,
    ) -> Result<bool, OperatorConsoleSubmitError> {
        self.close_session(session_id, capability)
    }

    fn tick_deadman(
        &self,
        now: HostMonotonicTimestamp,
    ) -> Result<bool, OperatorConsoleSubmitError> {
        self.tick_deadman(now)
    }

    fn fail_closed(&self) {
        self.signal_internal_fail_closed();
    }

    fn latest_snapshot(&self) -> Arc<OperatorConsoleSnapshot> {
        self.latest_snapshot()
    }

    fn exact_grid(
        &self,
        map_epoch_id: NonZeroU64,
        revision: u64,
    ) -> Option<Arc<super::ConsoleOccupancyGrid>> {
        self.exact_grid(map_epoch_id, revision)
    }

    fn observe_response_record(
        &self,
        id: ConsoleDownstreamRequestId,
        source_session_id: ConsoleSessionId,
    ) -> Option<OperatorConsoleResponseRecord> {
        self.observe_response_record_for_http(id, source_session_id)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OperatorConsoleHttpServerConfig {
    bind: OperatorConsoleBind,
    access_capability: OperatorConsoleAccessCapability,
    clock: super::AgentControlMonotonicOrigin,
    deadman_tick: Duration,
}

impl OperatorConsoleHttpServerConfig {
    pub fn parse(
        bind: OperatorConsoleBind,
        access_capability: OperatorConsoleAccessCapability,
        clock: super::AgentControlMonotonicOrigin,
        deadman_tick: Duration,
    ) -> Result<Self, OperatorConsoleHttpConfigError> {
        let millis = u64::try_from(deadman_tick.as_millis())
            .map_err(|_| OperatorConsoleHttpConfigError::DeadmanTickOutOfRange)?;
        if !(MIN_DEADMAN_TICK_MS..=MAX_DEADMAN_TICK_MS).contains(&millis)
            || Duration::from_millis(millis) != deadman_tick
        {
            return Err(OperatorConsoleHttpConfigError::DeadmanTickOutOfRange);
        }
        Ok(Self {
            bind,
            access_capability,
            clock,
            deadman_tick,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorConsoleHttpConfigError {
    DeadmanTickOutOfRange,
}

impl fmt::Display for OperatorConsoleHttpConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("operator-console HTTP deadman tick must be an exact 5..=100 ms")
    }
}

impl std::error::Error for OperatorConsoleHttpConfigError {}

struct HttpContext {
    backend: Arc<dyn OperatorConsoleHttpBackend>,
    access_capability: OperatorConsoleAccessCapability,
    clock: super::AgentControlMonotonicOrigin,
    bound_port: AtomicU16,
    request_permits: Arc<Semaphore>,
}

#[derive(Debug, Serialize)]
struct ErrorBody<'a> {
    error: &'a str,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenSessionDto {
    schema_version: u32,
    source: ConsoleSourceKind,
}

#[derive(Debug, Serialize)]
struct OpenSessionResponse {
    schema_version: u32,
    session_id: ConsoleSessionId,
    session_capability: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CloseSessionDto {
    schema_version: u32,
    session_id: String,
}

#[derive(Debug, Serialize)]
struct CloseSessionResponse {
    schema_version: u32,
    closed: bool,
}

#[derive(Debug, Serialize)]
struct SubmitResponse {
    schema_version: u32,
    state: &'static str,
    downstream_request_id: ConsoleDownstreamRequestId,
    applied: bool,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    schema_version: u32,
    http_ready: bool,
    runtime_known: bool,
}

struct AuthorizedPost {
    path: warp::path::FullPath,
    session: Option<(ConsoleSessionId, ConsoleSessionCapability)>,
    context: Arc<HttpContext>,
    _permit: OwnedSemaphorePermit,
}

#[derive(Debug)]
pub struct OperatorConsoleHttpServer {
    bound_addr: SocketAddr,
    shutdown: Option<oneshot::Sender<()>>,
    join: Option<thread::JoinHandle<OperatorConsoleHttpServerExit>>,
}

impl OperatorConsoleHttpServer {
    pub fn start(
        config: OperatorConsoleHttpServerConfig,
        backend: Arc<dyn OperatorConsoleHttpBackend>,
    ) -> Result<Self, OperatorConsoleHttpServerStartError> {
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let thread = thread::Builder::new()
            .name("kiko-operator-console-http".to_string())
            .spawn(move || run_server(config, backend, shutdown_rx, ready_tx))
            .map_err(OperatorConsoleHttpServerStartError::Spawn)?;
        let ready = ready_rx
            .recv_timeout(Duration::from_secs(5))
            .map_err(|_| OperatorConsoleHttpServerStartError::ReadinessLost)?;
        let bound_addr = match ready {
            Ok(address) => address,
            Err(message) => {
                let _ = thread.join();
                return Err(OperatorConsoleHttpServerStartError::Bind(message));
            }
        };
        if !bound_addr.ip().is_loopback() {
            let _ = shutdown_tx.send(());
            let _ = thread.join();
            return Err(OperatorConsoleHttpServerStartError::NonLoopbackBound(
                bound_addr,
            ));
        }
        Ok(Self {
            bound_addr,
            shutdown: Some(shutdown_tx),
            join: Some(thread),
        })
    }

    pub const fn bound_addr(&self) -> SocketAddr {
        self.bound_addr
    }

    /// Stop accepting new work without waiting for in-flight HTTP ownership to
    /// drain. Motion shutdown calls this before touching the controller owner.
    pub fn request_shutdown(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
    }

    /// Join the HTTP owner. A timeout retains the join handle in `self`, so the
    /// caller can retry and must not revoke the capability while that owner may
    /// still be serving it.
    pub fn shutdown(
        &mut self,
    ) -> Result<OperatorConsoleHttpServerExit, OperatorConsoleHttpJoinError> {
        self.shutdown_with_timeout(HTTP_SHUTDOWN_TIMEOUT)
    }

    /// Nonblocking owner-health probe. `Ok(None)` means the HTTP thread is
    /// still alive; any completed result consumes and joins the exact owner.
    pub fn try_join(
        &mut self,
    ) -> Result<Option<OperatorConsoleHttpServerExit>, OperatorConsoleHttpJoinError> {
        let join = self
            .join
            .as_ref()
            .ok_or(OperatorConsoleHttpJoinError::AlreadyJoined)?;
        if !join.is_finished() {
            return Ok(None);
        }
        let join = self
            .join
            .take()
            .ok_or(OperatorConsoleHttpJoinError::AlreadyJoined)?;
        join.join()
            .map(Some)
            .map_err(|_| OperatorConsoleHttpJoinError::Panicked)
    }

    pub(crate) fn shutdown_with_timeout(
        &mut self,
        timeout: Duration,
    ) -> Result<OperatorConsoleHttpServerExit, OperatorConsoleHttpJoinError> {
        self.request_shutdown();
        let join = self
            .join
            .as_ref()
            .ok_or(OperatorConsoleHttpJoinError::AlreadyJoined)?;
        let deadline = Instant::now() + timeout;
        while !join.is_finished() {
            if Instant::now() >= deadline {
                return Err(OperatorConsoleHttpJoinError::TimedOut);
            }
            thread::sleep(Duration::from_millis(10));
        }
        let join = self
            .join
            .take()
            .ok_or(OperatorConsoleHttpJoinError::AlreadyJoined)?;
        join.join()
            .map_err(|_| OperatorConsoleHttpJoinError::Panicked)
    }
}

impl Drop for OperatorConsoleHttpServer {
    fn drop(&mut self) {
        self.request_shutdown();
    }
}

#[derive(Debug)]
pub enum OperatorConsoleHttpServerStartError {
    Spawn(std::io::Error),
    ReadinessLost,
    Bind(String),
    NonLoopbackBound(SocketAddr),
}

impl fmt::Display for OperatorConsoleHttpServerStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("operator-console HTTP server did not become ready")
    }
}

impl std::error::Error for OperatorConsoleHttpServerStartError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorConsoleHttpJoinError {
    AlreadyJoined,
    Panicked,
    TimedOut,
}

impl fmt::Display for OperatorConsoleHttpJoinError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "operator-console HTTP join failed: {self:?}")
    }
}

impl std::error::Error for OperatorConsoleHttpJoinError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperatorConsoleHttpServerExit {
    pub bound_addr: SocketAddr,
    pub graceful_shutdown: bool,
    pub forced_shutdown: bool,
    pub deadman_ticks: u64,
    pub deadman_stops_enqueued: u64,
    pub clock_faulted: bool,
}

fn run_server(
    config: OperatorConsoleHttpServerConfig,
    backend: Arc<dyn OperatorConsoleHttpBackend>,
    shutdown_rx: oneshot::Receiver<()>,
    ready_tx: mpsc::SyncSender<Result<SocketAddr, String>>,
) -> OperatorConsoleHttpServerExit {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            let _ = ready_tx.send(Err(error.to_string()));
            return OperatorConsoleHttpServerExit {
                bound_addr: config.bind.address(),
                graceful_shutdown: false,
                forced_shutdown: false,
                deadman_ticks: 0,
                deadman_stops_enqueued: 0,
                clock_faulted: true,
            };
        }
    };
    let context = Arc::new(HttpContext {
        backend: Arc::clone(&backend),
        access_capability: config.access_capability,
        clock: config.clock,
        bound_port: AtomicU16::new(0),
        request_permits: Arc::new(Semaphore::new(MAX_CONCURRENT_API_REQUESTS)),
    });
    let routes = routes(Arc::clone(&context));
    let shutdown_observed = Arc::new(AtomicBool::new(false));
    let forced_shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_notify = Arc::new(Notify::new());
    let shutdown_for_signal = Arc::clone(&shutdown_observed);
    let notify_for_signal = Arc::clone(&shutdown_notify);
    let graceful = async move {
        let _ = shutdown_rx.await;
        shutdown_for_signal.store(true, Ordering::Release);
        notify_for_signal.notify_one();
    };
    let bind_result = {
        // Hyper creates its listener through the current Tokio reactor.
        let _runtime_guard = runtime.enter();
        warp::serve(routes).try_bind_with_graceful_shutdown(config.bind.address(), graceful)
    };
    let (bound_addr, server) = match bind_result {
        Ok(parts) => parts,
        Err(error) => {
            let _ = ready_tx.send(Err(error.to_string()));
            return OperatorConsoleHttpServerExit {
                bound_addr: config.bind.address(),
                graceful_shutdown: false,
                forced_shutdown: false,
                deadman_ticks: 0,
                deadman_stops_enqueued: 0,
                clock_faulted: false,
            };
        }
    };
    context
        .bound_port
        .store(bound_addr.port(), Ordering::Release);
    if ready_tx.send(Ok(bound_addr)).is_err() {
        return OperatorConsoleHttpServerExit {
            bound_addr,
            graceful_shutdown: false,
            forced_shutdown: false,
            deadman_ticks: 0,
            deadman_stops_enqueued: 0,
            clock_faulted: false,
        };
    }
    let ticks = Arc::new(AtomicU64::new(0));
    let stops = Arc::new(AtomicU64::new(0));
    let clock_faulted = Arc::new(AtomicBool::new(false));
    let tick_counts = Arc::clone(&ticks);
    let stop_counts = Arc::clone(&stops);
    let tick_clock_faulted = Arc::clone(&clock_faulted);
    let force_notify = Arc::clone(&shutdown_notify);
    let force_evidence = Arc::clone(&forced_shutdown);
    let force_shutdown = async move {
        force_notify.notified().await;
        tokio::time::sleep(HTTP_FORCE_SHUTDOWN_AFTER).await;
        force_evidence.store(true, Ordering::Release);
    };
    let deadman = async move {
        let mut interval = tokio::time::interval(config.deadman_tick);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            tick_counts.fetch_add(1, Ordering::Relaxed);
            match config.clock.try_now() {
                Ok(now) => {
                    if backend.tick_deadman(now).unwrap_or_else(|_| {
                        backend.fail_closed();
                        false
                    }) {
                        stop_counts.fetch_add(1, Ordering::Relaxed);
                    }
                }
                Err(_) => {
                    tick_clock_faulted.store(true, Ordering::Release);
                    backend.fail_closed();
                }
            }
        }
    };
    runtime.block_on(async {
        tokio::select! {
            () = server => {}
            () = deadman => {}
            () = force_shutdown => {}
        }
    });
    let forced_shutdown = forced_shutdown.load(Ordering::Acquire);
    OperatorConsoleHttpServerExit {
        bound_addr,
        graceful_shutdown: shutdown_observed.load(Ordering::Acquire) && !forced_shutdown,
        forced_shutdown,
        deadman_ticks: ticks.load(Ordering::Relaxed),
        deadman_stops_enqueued: stops.load(Ordering::Relaxed),
        clock_faulted: clock_faulted.load(Ordering::Acquire),
    }
}

fn routes(context: Arc<HttpContext>) -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    let get_context = Arc::clone(&context);
    let get = warp::get()
        .and(warp::path::full())
        .and(warp::header::optional::<String>("host"))
        .and(warp::header::optional::<String>("origin"))
        .and(warp::header::optional::<String>(
            "x-kiko-console-capability",
        ))
        .and(warp::header::optional::<String>("x-kiko-session-id"))
        .and(warp::header::optional::<String>(
            "x-kiko-session-capability",
        ))
        .and(warp::header::optional::<String>("if-none-match"))
        .and(warp::any().map(move || Arc::clone(&get_context)))
        .and_then(handle_get);

    let post = warp::post()
        .and(warp::path::full())
        .and(warp::header::optional::<String>("host"))
        .and(warp::header::optional::<String>("origin"))
        .and(warp::header::optional::<String>("content-type"))
        .and(warp::header::optional::<String>(
            "x-kiko-console-capability",
        ))
        .and(warp::header::optional::<String>("x-kiko-session-id"))
        .and(warp::header::optional::<String>(
            "x-kiko-session-capability",
        ))
        .and(warp::any().map(move || Arc::clone(&context)))
        .and_then(preflight_post)
        .and(warp::body::content_length_limit(
            MAX_OPERATOR_CONSOLE_REQUEST_BYTES as u64,
        ))
        .and(warp::body::bytes())
        .and_then(handle_post);

    get.or(post).unify().recover(recover_http).unify().boxed()
}

#[allow(clippy::too_many_arguments)]
async fn handle_get(
    path: warp::path::FullPath,
    host: Option<String>,
    origin: Option<String>,
    access_header: Option<String>,
    session_id_header: Option<String>,
    session_header: Option<String>,
    if_none_match: Option<String>,
    context: Arc<HttpContext>,
) -> Result<warp::reply::Response, Infallible> {
    let port = context.bound_port.load(Ordering::Acquire);
    let Some(host) = host.filter(|host| valid_host(host, port)) else {
        return Ok(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_loopback_host",
        ));
    };
    let path = path.as_str();
    let static_response = match path {
        "/" | "/index.html" => Some(text_response(
            StatusCode::OK,
            "text/html; charset=utf-8",
            INDEX_HTML,
        )),
        "/assets/styles.css" => Some(text_response(
            StatusCode::OK,
            "text/css; charset=utf-8",
            STYLES_CSS,
        )),
        "/assets/app.js" => Some(text_response(
            StatusCode::OK,
            "text/javascript; charset=utf-8",
            APP_JS,
        )),
        "/assets/view-model.js" => Some(text_response(
            StatusCode::OK,
            "text/javascript; charset=utf-8",
            VIEW_MODEL_JS,
        )),
        _ => None,
    };
    if let Some(response) = static_response {
        return Ok(response);
    }
    if !path.starts_with("/api/v1/") {
        return Ok(error_response(StatusCode::NOT_FOUND, "not_found"));
    }
    let access_valid = access_header
        .as_deref()
        .and_then(OperatorConsoleAccessCapability::parse_hex)
        .is_some_and(|candidate| context.access_capability.constant_time_matches(candidate));
    if !access_valid {
        return Ok(error_response(StatusCode::UNAUTHORIZED, "unauthorized"));
    }
    let origin_valid = origin
        .as_deref()
        .is_none_or(|origin| origin == format!("http://{host}"));
    if !origin_valid {
        return Ok(error_response(StatusCode::FORBIDDEN, "invalid_origin"));
    }
    let Ok(_permit) = Arc::clone(&context.request_permits).try_acquire_owned() else {
        return Ok(error_response(
            StatusCode::TOO_MANY_REQUESTS,
            "too_many_requests",
        ));
    };

    let response = match path {
        "/api/v1/health" => {
            let snapshot = context.backend.latest_snapshot();
            json_response(
                StatusCode::OK,
                &HealthResponse {
                    schema_version: 1,
                    http_ready: true,
                    runtime_known: snapshot.runtime.is_some(),
                },
            )
        }
        "/api/v1/snapshot" => {
            json_response(StatusCode::OK, context.backend.latest_snapshot().as_ref())
        }
        _ if path.starts_with("/api/v1/responses/") => response_record_response(
            &context,
            path,
            session_id_header.as_deref(),
            session_header.as_deref(),
        ),
        _ if path.starts_with("/api/v1/maps/") => {
            grid_response(&context, path, if_none_match.as_deref())
        }
        _ => error_response(StatusCode::NOT_FOUND, "not_found"),
    };
    Ok(response)
}

#[allow(clippy::too_many_arguments)]
async fn preflight_post(
    path: warp::path::FullPath,
    host: Option<String>,
    origin: Option<String>,
    content_type: Option<String>,
    access_header: Option<String>,
    session_id_header: Option<String>,
    session_header: Option<String>,
    context: Arc<HttpContext>,
) -> Result<AuthorizedPost, warp::Rejection> {
    let port = context.bound_port.load(Ordering::Acquire);
    let host = host
        .filter(|host| valid_host(host, port))
        .ok_or_else(|| reject_http(StatusCode::BAD_REQUEST, "invalid_loopback_host"))?;
    if !matches!(
        path.as_str(),
        "/api/v1/sessions" | "/api/v1/intents" | "/api/v1/sessions/close"
    ) {
        return Err(reject_http(StatusCode::NOT_FOUND, "not_found"));
    }
    let access_valid = access_header
        .as_deref()
        .and_then(OperatorConsoleAccessCapability::parse_hex)
        .is_some_and(|candidate| context.access_capability.constant_time_matches(candidate));
    if !access_valid {
        return Err(reject_http(StatusCode::UNAUTHORIZED, "unauthorized"));
    }
    let expected_origin = format!("http://{host}");
    if origin.as_deref() != Some(expected_origin.as_str()) {
        return Err(reject_http(StatusCode::FORBIDDEN, "invalid_origin"));
    }
    if !content_type.as_deref().is_some_and(|value| {
        value
            .split(';')
            .next()
            .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("application/json"))
    }) {
        return Err(reject_http(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "application_json_required",
        ));
    }
    let session = if path.as_str() == "/api/v1/sessions" {
        None
    } else {
        let session_id = session_id_header
            .as_deref()
            .and_then(|raw| raw.parse::<u64>().ok())
            .and_then(|raw| ConsoleSessionId::parse(raw).ok())
            .ok_or_else(|| reject_http(StatusCode::UNAUTHORIZED, "invalid_session"))?;
        let session_capability = session_header
            .as_deref()
            .and_then(OperatorConsoleAccessCapability::parse_hex)
            .map(|capability| ConsoleSessionCapability::from_bytes(capability.0))
            .ok_or_else(|| reject_http(StatusCode::UNAUTHORIZED, "invalid_session"))?;
        if !context
            .backend
            .session_capability_matches(session_id, session_capability)
        {
            return Err(reject_http(StatusCode::UNAUTHORIZED, "invalid_session"));
        }
        Some((session_id, session_capability))
    };
    let permit = Arc::clone(&context.request_permits)
        .try_acquire_owned()
        .map_err(|_| reject_http(StatusCode::TOO_MANY_REQUESTS, "too_many_requests"))?;
    Ok(AuthorizedPost {
        path,
        session,
        context,
        _permit: permit,
    })
}

async fn handle_post(
    request: AuthorizedPost,
    body: Bytes,
) -> Result<warp::reply::Response, Infallible> {
    let response = match request.path.as_str() {
        "/api/v1/sessions" => open_session_response(&request.context, &body),
        "/api/v1/intents" => intent_response(&request.context, request.session, &body),
        "/api/v1/sessions/close" => {
            close_session_response(&request.context, request.session, &body)
        }
        _ => error_response(StatusCode::NOT_FOUND, "not_found"),
    };
    Ok(response)
}

fn open_session_response(context: &HttpContext, body: &[u8]) -> warp::reply::Response {
    let dto: OpenSessionDto = match parse_exact_json::<OpenSessionDto>(body) {
        Ok(dto) if dto.schema_version == 1 => dto,
        _ => return error_response(StatusCode::BAD_REQUEST, "invalid_session_request"),
    };
    let mut session_bytes = [0_u8; CAPABILITY_BYTES];
    if getrandom::fill(&mut session_bytes).is_err() || session_bytes == [0; CAPABILITY_BYTES] {
        context.backend.fail_closed();
        return error_response(StatusCode::INTERNAL_SERVER_ERROR, "entropy_unavailable");
    }
    let capability = ConsoleSessionCapability::from_bytes(session_bytes);
    let now = match context.clock.try_now() {
        Ok(now) => now,
        Err(_) => {
            context.backend.fail_closed();
            return error_response(StatusCode::SERVICE_UNAVAILABLE, "host_clock_fault");
        }
    };
    match context.backend.open_session(dto.source, capability, now) {
        Ok(session_id) => json_response(
            StatusCode::CREATED,
            &OpenSessionResponse {
                schema_version: 1,
                session_id,
                session_capability: encode_hex(capability.as_bytes()),
            },
        ),
        Err(_) => error_response(StatusCode::CONFLICT, "session_unavailable"),
    }
}

fn intent_response(
    context: &HttpContext,
    authorized_session: Option<(ConsoleSessionId, ConsoleSessionCapability)>,
    body: &[u8],
) -> warp::reply::Response {
    let Some((authorized_session_id, session_capability)) = authorized_session else {
        return error_response(StatusCode::UNAUTHORIZED, "invalid_session_capability");
    };
    let dto: ConsoleIntentRequestDto = match parse_exact_json(body) {
        Ok(dto) => dto,
        Err(()) => return error_response(StatusCode::BAD_REQUEST, "invalid_intent_json"),
    };
    let (session_id, source_sequence, idempotency_key, intent) = match dto.parse() {
        Ok(parsed) => parsed,
        Err(_) => return error_response(StatusCode::BAD_REQUEST, "invalid_typed_intent"),
    };
    if session_id != authorized_session_id {
        return error_response(StatusCode::UNAUTHORIZED, "invalid_session");
    }
    let received_at = match context.clock.try_now() {
        Ok(received_at) => received_at,
        Err(_) => {
            context.backend.fail_closed();
            return error_response(StatusCode::SERVICE_UNAVAILABLE, "host_clock_fault");
        }
    };
    match context.backend.submit(
        session_id,
        session_capability,
        source_sequence,
        idempotency_key,
        intent,
        received_at,
    ) {
        Ok(outcome) => {
            let state = match outcome {
                OperatorConsoleSubmitOutcome::AcceptedForProcessing { .. } => {
                    "accepted_for_processing"
                }
                OperatorConsoleSubmitOutcome::IdempotentReplay { .. } => "idempotent_replay",
                OperatorConsoleSubmitOutcome::SoftwareSafetyStopLatched { .. } => {
                    "software_safety_stop_latched"
                }
            };
            json_response(
                StatusCode::ACCEPTED,
                &SubmitResponse {
                    schema_version: 1,
                    state,
                    downstream_request_id: outcome.downstream_request_id(),
                    applied: false,
                },
            )
        }
        Err(
            OperatorConsoleSubmitError::NormalQueueFull
            | OperatorConsoleSubmitError::ResponseCapacityReached,
        ) => error_response(StatusCode::TOO_MANY_REQUESTS, "console_backpressure"),
        Err(
            OperatorConsoleSubmitError::SessionCapabilityMismatch
            | OperatorConsoleSubmitError::UnknownSession(_),
        ) => error_response(StatusCode::UNAUTHORIZED, "invalid_session"),
        Err(OperatorConsoleSubmitError::SoftwareSafetyStopLatched) => {
            error_response(StatusCode::LOCKED, "software_safety_stop_latched")
        }
        Err(_) => error_response(StatusCode::CONFLICT, "intent_rejected"),
    }
}

fn close_session_response(
    context: &HttpContext,
    authorized_session: Option<(ConsoleSessionId, ConsoleSessionCapability)>,
    body: &[u8],
) -> warp::reply::Response {
    let Some((authorized_session_id, session_capability)) = authorized_session else {
        return error_response(StatusCode::UNAUTHORIZED, "invalid_session_capability");
    };
    let dto: CloseSessionDto = match parse_exact_json::<CloseSessionDto>(body) {
        Ok(dto) if dto.schema_version == 1 => dto,
        _ => return error_response(StatusCode::BAD_REQUEST, "invalid_close_request"),
    };
    let session_id = match dto
        .session_id
        .parse::<u64>()
        .ok()
        .and_then(|value| ConsoleSessionId::parse(value).ok())
    {
        Some(session_id) => session_id,
        None => return error_response(StatusCode::BAD_REQUEST, "invalid_session_id"),
    };
    if session_id != authorized_session_id {
        return error_response(StatusCode::UNAUTHORIZED, "invalid_session");
    }
    match context
        .backend
        .close_session(session_id, session_capability)
    {
        Ok(closed) => json_response(
            StatusCode::OK,
            &CloseSessionResponse {
                schema_version: 1,
                closed,
            },
        ),
        Err(_) => error_response(StatusCode::UNAUTHORIZED, "invalid_session"),
    }
}

fn response_record_response(
    context: &HttpContext,
    path: &str,
    session_id_header: Option<&str>,
    session_header: Option<&str>,
) -> warp::reply::Response {
    let Some(session_id) = session_id_header
        .and_then(|raw| raw.parse::<u64>().ok())
        .and_then(|raw| ConsoleSessionId::parse(raw).ok())
    else {
        return error_response(StatusCode::UNAUTHORIZED, "invalid_session");
    };
    let Some(session_capability) = session_header
        .and_then(OperatorConsoleAccessCapability::parse_hex)
        .map(|capability| ConsoleSessionCapability::from_bytes(capability.0))
    else {
        return error_response(StatusCode::UNAUTHORIZED, "invalid_session");
    };
    if !context
        .backend
        .session_capability_matches(session_id, session_capability)
    {
        return error_response(StatusCode::UNAUTHORIZED, "invalid_session");
    }
    let Some(raw) = path.strip_prefix("/api/v1/responses/") else {
        return error_response(StatusCode::NOT_FOUND, "not_found");
    };
    let Some(id) = raw
        .parse::<u64>()
        .ok()
        .and_then(NonZeroU64::new)
        .map(ConsoleDownstreamRequestId::from_nonzero_for_http)
    else {
        return error_response(StatusCode::BAD_REQUEST, "invalid_response_id");
    };
    match context.backend.observe_response_record(id, session_id) {
        Some(record) => json_response(StatusCode::OK, &record),
        None => error_response(StatusCode::NOT_FOUND, "response_not_found"),
    }
}

struct GridBytesOwner(Arc<super::ConsoleOccupancyGrid>);

impl AsRef<[u8]> for GridBytesOwner {
    fn as_ref(&self) -> &[u8] {
        &self.0.cells
    }
}

fn grid_response(
    context: &HttpContext,
    path: &str,
    if_none_match: Option<&str>,
) -> warp::reply::Response {
    let fields: Vec<_> = path.trim_matches('/').split('/').collect();
    if fields.len() != 7
        || fields[0..3] != ["api", "v1", "maps"]
        || fields[4] != "revisions"
        || fields[6] != "grid"
    {
        return error_response(StatusCode::NOT_FOUND, "not_found");
    }
    let Some(epoch) = fields[3].parse::<u64>().ok().and_then(NonZeroU64::new) else {
        return error_response(StatusCode::BAD_REQUEST, "invalid_map_epoch");
    };
    let Ok(revision) = fields[5].parse::<u64>() else {
        return error_response(StatusCode::BAD_REQUEST, "invalid_map_revision");
    };
    let Some(grid) = context.backend.exact_grid(epoch, revision) else {
        return error_response(StatusCode::NOT_FOUND, "exact_grid_not_available");
    };
    let etag = format!("\"{}:{}\"", epoch.get(), revision);
    if if_none_match == Some(etag.as_str()) {
        return secure_response(StatusCode::NOT_MODIFIED, None, Vec::new(), &[]);
    }
    let metadata = grid.metadata;
    let headers = [
        ("etag", etag),
        ("x-kiko-map-epoch", epoch.get().to_string()),
        ("x-kiko-map-revision", revision.to_string()),
        ("x-kiko-grid-width", metadata.width.get().to_string()),
        ("x-kiko-grid-height", metadata.height.get().to_string()),
        (
            "x-kiko-grid-encoding",
            "u8_unknown0_free1_occupied2".to_string(),
        ),
        (
            "x-kiko-grid-row-order",
            "row_major_x_fast_rows_increase_positive_map_y".to_string(),
        ),
        (
            "x-kiko-grid-origin",
            "minimum_xy_corner_of_cell_0_0".to_string(),
        ),
    ];
    secure_response(
        StatusCode::OK,
        Some("application/octet-stream"),
        Bytes::from_owner(GridBytesOwner(grid)),
        &headers,
    )
}

pub(super) fn parse_exact_json<T: DeserializeOwned>(body: &[u8]) -> Result<T, ()> {
    if body.is_empty()
        || body.len() > MAX_OPERATOR_CONSOLE_REQUEST_BYTES
        || body.first() != Some(&b'{')
        || body.last() != Some(&b'}')
    {
        return Err(());
    }
    serde_json::from_slice(body).map_err(|_| ())
}

pub(super) fn valid_host(host: &str, port: u16) -> bool {
    host == format!("127.0.0.1:{port}")
        || host == format!("localhost:{port}")
        || host == format!("[::1]:{port}")
}

pub(super) fn text_response(
    status: StatusCode,
    content_type: &'static str,
    body: &'static str,
) -> warp::reply::Response {
    secure_response(status, Some(content_type), body, &[])
}

pub(super) fn json_response<T: Serialize + ?Sized>(
    status: StatusCode,
    value: &T,
) -> warp::reply::Response {
    match serde_json::to_vec(value) {
        Ok(body) => secure_response(status, Some("application/json"), body, &[]),
        Err(_) => error_response(StatusCode::INTERNAL_SERVER_ERROR, "serialization_failed"),
    }
}

pub(super) fn error_response(status: StatusCode, code: &'static str) -> warp::reply::Response {
    let body = serde_json::to_vec(&ErrorBody { error: code })
        .unwrap_or_else(|_| b"{\"error\":\"internal_fault\"}".to_vec());
    secure_response(status, Some("application/json"), body, &[])
}

pub(super) fn secure_response(
    status: StatusCode,
    content_type: Option<&str>,
    body: impl Into<Body>,
    extra_headers: &[(&str, String)],
) -> warp::reply::Response {
    let mut builder = warp::http::Response::builder()
        .status(status)
        .header("cache-control", "no-store, max-age=0")
        .header("pragma", "no-cache")
        .header("x-content-type-options", "nosniff")
        .header("x-frame-options", "DENY")
        .header("referrer-policy", "no-referrer")
        .header("cross-origin-opener-policy", "same-origin")
        .header("cross-origin-resource-policy", "same-origin")
        .header(
            "content-security-policy",
            "default-src 'self'; base-uri 'none'; object-src 'none'; frame-ancestors 'none'; form-action 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'",
        );
    if let Some(content_type) = content_type {
        builder = builder.header("content-type", content_type);
    }
    for (name, value) in extra_headers {
        builder = builder.header(*name, value);
    }
    match builder.body(body.into()) {
        Ok(response) => response,
        Err(_) => {
            warp::reply::with_status("internal response error", StatusCode::INTERNAL_SERVER_ERROR)
                .into_response()
        }
    }
}

#[derive(Debug)]
struct HttpPreflightRejection {
    status: StatusCode,
    code: &'static str,
}

impl warp::reject::Reject for HttpPreflightRejection {}

fn reject_http(status: StatusCode, code: &'static str) -> warp::Rejection {
    warp::reject::custom(HttpPreflightRejection { status, code })
}

async fn recover_http(rejection: warp::Rejection) -> Result<warp::reply::Response, Infallible> {
    let response = if let Some(rejection) = rejection.find::<HttpPreflightRejection>() {
        error_response(rejection.status, rejection.code)
    } else if rejection.find::<warp::reject::PayloadTooLarge>().is_some() {
        error_response(StatusCode::PAYLOAD_TOO_LARGE, "request_body_too_large")
    } else {
        error_response(StatusCode::BAD_REQUEST, "invalid_http_request")
    };
    Ok(response)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpStream;
    #[cfg(unix)]
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
    use std::sync::atomic::AtomicU64;

    use super::*;
    use crate::navigation::{
        ConsoleResponseRejectionCode, ConsoleSnapshotRevision, OperatorConsoleIngressItem,
        OperatorConsoleLimits, OperatorConsoleSnapshot, operator_console,
    };

    fn request(address: SocketAddr, raw: &str) -> String {
        let mut stream = TcpStream::connect(address).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        stream.write_all(raw.as_bytes()).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        response
    }

    #[cfg(unix)]
    struct PrivateTestDirectory(PathBuf);

    #[cfg(unix)]
    impl PrivateTestDirectory {
        fn create() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(1);
            let suffix = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "kiko-console-capability-test-{}-{suffix}",
                std::process::id()
            ));
            let mut builder = fs::DirBuilder::new();
            builder.mode(PRIVATE_CAPABILITY_PARENT_MODE);
            builder.create(&path).unwrap();
            fs::set_permissions(
                &path,
                fs::Permissions::from_mode(PRIVATE_CAPABILITY_PARENT_MODE),
            )
            .unwrap();
            Self(path)
        }
    }

    #[cfg(unix)]
    impl Drop for PrivateTestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn real_listener_requires_capabilities_and_shuts_down_with_evidence() {
        let (console, receiver) = operator_console(
            OperatorConsoleLimits::default(),
            OperatorConsoleSnapshot::unknown(ConsoleSnapshotRevision::parse(1).unwrap()),
        );
        let access = OperatorConsoleAccessCapability::parse([0x33; 32]).unwrap();
        let origin = super::super::AgentControlMonotonicOrigin::new(
            Instant::now(),
            HostMonotonicTimestamp::from_nanos(1_000_000_000),
        );
        let config = OperatorConsoleHttpServerConfig::parse(
            OperatorConsoleBind::parse("127.0.0.1:0".parse().unwrap()).unwrap(),
            access,
            origin,
            Duration::from_millis(20),
        )
        .unwrap();
        let mut server =
            OperatorConsoleHttpServer::start(config, Arc::new(console.clone())).unwrap();
        let address = server.bound_addr();
        let host = format!("127.0.0.1:{}", address.port());

        let static_response = request(
            address,
            &format!("GET / HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"),
        );
        assert!(static_response.starts_with("HTTP/1.1 200"));
        assert!(static_response.contains("content-security-policy:"));
        assert!(static_response.contains("cache-control: no-store"));

        let model_asset = request(
            address,
            &format!(
                "GET /assets/view-model.js HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"
            ),
        );
        assert!(model_asset.starts_with("HTTP/1.1 200"), "{model_asset}");
        assert!(model_asset.contains("content-type: text/javascript; charset=utf-8"));
        assert!(model_asset.contains("KikoOperatorConsoleModel"));

        let unauthorized = request(
            address,
            &format!("GET /api/v1/snapshot HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"),
        );
        assert!(unauthorized.starts_with("HTTP/1.1 401"));

        let open_body = r#"{"schema_version":1,"source":"operator"}"#;
        let opened = request(
            address,
            &format!(
                "POST /api/v1/sessions HTTP/1.1\r\nHost: {host}\r\nOrigin: http://{host}\r\nX-Kiko-Console-Capability: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                access.to_hex(),
                open_body.len(),
                open_body
            ),
        );
        assert!(opened.starts_with("HTTP/1.1 201"), "{opened}");
        let body = opened.split("\r\n\r\n").nth(1).unwrap();
        let opened_json: serde_json::Value = serde_json::from_str(body).unwrap();
        let session_id = opened_json["session_id"].as_str().unwrap();
        let session_capability = opened_json["session_capability"].as_str().unwrap();
        let intent_body = format!(
            "{{\"schema_version\":1,\"session_id\":\"{session_id}\",\"source_sequence\":\"1\",\"idempotency_key\":\"1\",\"intent\":{{\"kind\":\"arm\"}}}}"
        );
        let accepted = request(
            address,
            &format!(
                "POST /api/v1/intents HTTP/1.1\r\nHost: {host}\r\nOrigin: http://{host}\r\nX-Kiko-Console-Capability: {}\r\nX-Kiko-Session-Id: {session_id}\r\nX-Kiko-Session-Capability: {session_capability}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                access.to_hex(),
                intent_body.len(),
                intent_body
            ),
        );
        assert!(accepted.starts_with("HTTP/1.1 202"), "{accepted}");
        assert!(accepted.contains("\"applied\":false"));
        let accepted_body = accepted.split("\r\n\r\n").nth(1).unwrap();
        let accepted_json: serde_json::Value = serde_json::from_str(accepted_body).unwrap();
        let response_id = accepted_json["downstream_request_id"].as_str().unwrap();
        let dispatch = receiver.try_next().unwrap();
        let OperatorConsoleIngressItem::Dispatch(dispatch) = dispatch else {
            panic!("ordinary arm dispatch expected");
        };
        assert!(dispatch.received_at().as_nanos() >= 1_000_000_000);
        let (_, _, _, response) = dispatch.into_parts();
        response.reject(ConsoleResponseRejectionCode::RuntimeRejected);

        let unscoped_record = request(
            address,
            &format!(
                "GET /api/v1/responses/{response_id} HTTP/1.1\r\nHost: {host}\r\nX-Kiko-Console-Capability: {}\r\nConnection: close\r\n\r\n",
                access.to_hex(),
            ),
        );
        assert!(unscoped_record.starts_with("HTTP/1.1 401"));
        let scoped_record = request(
            address,
            &format!(
                "GET /api/v1/responses/{response_id} HTTP/1.1\r\nHost: {host}\r\nX-Kiko-Console-Capability: {}\r\nX-Kiko-Session-Id: {session_id}\r\nX-Kiko-Session-Capability: {session_capability}\r\nConnection: close\r\n\r\n",
                access.to_hex(),
            ),
        );
        assert!(scoped_record.starts_with("HTTP/1.1 200"), "{scoped_record}");
        assert!(scoped_record.contains("\"state\":\"rejected\""));

        let unauthorized_large_body = request(
            address,
            &format!(
                "POST /api/v1/intents HTTP/1.1\r\nHost: {host}\r\nOrigin: http://{host}\r\nContent-Type: application/json\r\nContent-Length: 999999\r\nConnection: close\r\n\r\n"
            ),
        );
        assert!(unauthorized_large_body.starts_with("HTTP/1.1 401"));
        let invalid_session_large_body = request(
            address,
            &format!(
                "POST /api/v1/intents HTTP/1.1\r\nHost: {host}\r\nOrigin: http://{host}\r\nX-Kiko-Console-Capability: {}\r\nX-Kiko-Session-Id: 999999\r\nX-Kiko-Session-Capability: {session_capability}\r\nContent-Type: application/json\r\nContent-Length: 999999\r\nConnection: close\r\n\r\n",
                access.to_hex(),
            ),
        );
        assert!(invalid_session_large_body.starts_with("HTTP/1.1 401"));

        let exit = server.shutdown().unwrap();
        assert!(exit.graceful_shutdown);
        assert_eq!(exit.bound_addr, address);
    }

    #[test]
    fn save_map_completion_is_http_observed_only_by_its_owning_session() {
        let (console, receiver) = operator_console(
            OperatorConsoleLimits::default(),
            OperatorConsoleSnapshot::unknown(ConsoleSnapshotRevision::parse(1).unwrap()),
        );
        let access = OperatorConsoleAccessCapability::parse([0x45; 32]).unwrap();
        let origin = super::super::AgentControlMonotonicOrigin::new(
            Instant::now(),
            HostMonotonicTimestamp::from_nanos(1_000_000_000),
        );
        let config = OperatorConsoleHttpServerConfig::parse(
            OperatorConsoleBind::parse("127.0.0.1:0".parse().unwrap()).unwrap(),
            access,
            origin,
            Duration::from_millis(20),
        )
        .unwrap();
        let mut server =
            OperatorConsoleHttpServer::start(config, Arc::new(console.clone())).unwrap();
        let address = server.bound_addr();
        let host = format!("127.0.0.1:{}", address.port());
        let open_session = |source: &str| {
            let body = format!(r#"{{"schema_version":1,"source":"{source}"}}"#);
            let opened = request(
                address,
                &format!(
                    "POST /api/v1/sessions HTTP/1.1\r\nHost: {host}\r\nOrigin: http://{host}\r\nX-Kiko-Console-Capability: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    access.to_hex(),
                    body.len(),
                    body
                ),
            );
            assert!(opened.starts_with("HTTP/1.1 201"), "{opened}");
            let parsed: serde_json::Value =
                serde_json::from_str(opened.split("\r\n\r\n").nth(1).unwrap()).unwrap();
            (
                parsed["session_id"].as_str().unwrap().to_owned(),
                parsed["session_capability"].as_str().unwrap().to_owned(),
            )
        };
        let (owner_id, owner_capability) = open_session("operator");
        let (foreign_id, foreign_capability) = open_session("agent");
        let body = format!(
            "{{\"schema_version\":1,\"session_id\":\"{owner_id}\",\"source_sequence\":\"1\",\"idempotency_key\":\"1\",\"intent\":{{\"kind\":\"save_map\"}}}}"
        );
        let accepted = request(
            address,
            &format!(
                "POST /api/v1/intents HTTP/1.1\r\nHost: {host}\r\nOrigin: http://{host}\r\nX-Kiko-Console-Capability: {}\r\nX-Kiko-Session-Id: {owner_id}\r\nX-Kiko-Session-Capability: {owner_capability}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                access.to_hex(),
                body.len(),
                body
            ),
        );
        assert!(accepted.starts_with("HTTP/1.1 202"), "{accepted}");
        let accepted_json: serde_json::Value = serde_json::from_str(
            accepted
                .split("\r\n\r\n")
                .nth(1)
                .expect("accepted response body"),
        )
        .unwrap();
        let response_id_raw = accepted_json["downstream_request_id"]
            .as_str()
            .expect("downstream response identity")
            .parse::<u64>()
            .unwrap();
        let response_id = ConsoleDownstreamRequestId::from_nonzero_for_http(
            NonZeroU64::new(response_id_raw).unwrap(),
        );

        let dispatch = receiver.try_next().unwrap();
        let OperatorConsoleIngressItem::Dispatch(dispatch) = dispatch else {
            panic!("save-map dispatch expected");
        };
        let (_, _, _, response) = dispatch.into_parts();
        response.completed().unwrap();
        assert!(!console.response_record_was_http_observed(response_id));

        let unauthorized = request(
            address,
            &format!(
                "GET /api/v1/responses/{response_id_raw} HTTP/1.1\r\nHost: {host}\r\nX-Kiko-Console-Capability: {}\r\nX-Kiko-Session-Id: {owner_id}\r\nX-Kiko-Session-Capability: {foreign_capability}\r\nConnection: close\r\n\r\n",
                access.to_hex(),
            ),
        );
        assert!(unauthorized.starts_with("HTTP/1.1 401"), "{unauthorized}");
        assert!(!console.response_record_was_http_observed(response_id));

        let foreign = request(
            address,
            &format!(
                "GET /api/v1/responses/{response_id_raw} HTTP/1.1\r\nHost: {host}\r\nX-Kiko-Console-Capability: {}\r\nX-Kiko-Session-Id: {foreign_id}\r\nX-Kiko-Session-Capability: {foreign_capability}\r\nConnection: close\r\n\r\n",
                access.to_hex(),
            ),
        );
        assert!(foreign.starts_with("HTTP/1.1 404"), "{foreign}");
        assert!(!console.response_record_was_http_observed(response_id));

        let missing_response_id = response_id_raw.checked_add(1).unwrap();
        let missing = request(
            address,
            &format!(
                "GET /api/v1/responses/{missing_response_id} HTTP/1.1\r\nHost: {host}\r\nX-Kiko-Console-Capability: {}\r\nX-Kiko-Session-Id: {owner_id}\r\nX-Kiko-Session-Capability: {owner_capability}\r\nConnection: close\r\n\r\n",
                access.to_hex(),
            ),
        );
        assert!(missing.starts_with("HTTP/1.1 404"), "{missing}");
        assert!(!console.response_record_was_http_observed(response_id));

        let final_record = request(
            address,
            &format!(
                "GET /api/v1/responses/{response_id_raw} HTTP/1.1\r\nHost: {host}\r\nX-Kiko-Console-Capability: {}\r\nX-Kiko-Session-Id: {owner_id}\r\nX-Kiko-Session-Capability: {owner_capability}\r\nConnection: close\r\n\r\n",
                access.to_hex(),
            ),
        );
        assert!(final_record.starts_with("HTTP/1.1 200"), "{final_record}");
        assert!(final_record.contains("\"state\":\"completed\""));
        assert!(console.response_record_was_http_observed(response_id));

        assert!(server.shutdown().unwrap().graceful_shutdown);
    }

    #[test]
    fn foreign_session_observes_latched_stop_without_receiving_its_response_id() {
        let (console, _receiver) = operator_console(
            OperatorConsoleLimits::default(),
            OperatorConsoleSnapshot::unknown(ConsoleSnapshotRevision::parse(1).unwrap()),
        );
        let access = OperatorConsoleAccessCapability::parse([0x34; 32]).unwrap();
        let origin = super::super::AgentControlMonotonicOrigin::new(
            Instant::now(),
            HostMonotonicTimestamp::from_nanos(1_000_000_000),
        );
        let config = OperatorConsoleHttpServerConfig::parse(
            OperatorConsoleBind::parse("127.0.0.1:0".parse().unwrap()).unwrap(),
            access,
            origin,
            Duration::from_millis(20),
        )
        .unwrap();
        let mut server =
            OperatorConsoleHttpServer::start(config, Arc::new(console.clone())).unwrap();
        let address = server.bound_addr();
        let host = format!("127.0.0.1:{}", address.port());
        let open_session = |source: &str| {
            let body = format!(r#"{{"schema_version":1,"source":"{source}"}}"#);
            let opened = request(
                address,
                &format!(
                    "POST /api/v1/sessions HTTP/1.1\r\nHost: {host}\r\nOrigin: http://{host}\r\nX-Kiko-Console-Capability: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    access.to_hex(),
                    body.len(),
                    body
                ),
            );
            assert!(opened.starts_with("HTTP/1.1 201"), "{opened}");
            let parsed: serde_json::Value =
                serde_json::from_str(opened.split("\r\n\r\n").nth(1).unwrap()).unwrap();
            (
                parsed["session_id"].as_str().unwrap().to_owned(),
                parsed["session_capability"].as_str().unwrap().to_owned(),
            )
        };
        let (owner_id, owner_capability) = open_session("operator");
        let (observer_id, observer_capability) = open_session("agent");
        let submit_stop = |session_id: &str, session_capability: &str| {
            let body = format!(
                "{{\"schema_version\":1,\"session_id\":\"{session_id}\",\"source_sequence\":\"1\",\"idempotency_key\":\"1\",\"intent\":{{\"kind\":\"software_safety_stop\"}}}}"
            );
            request(
                address,
                &format!(
                    "POST /api/v1/intents HTTP/1.1\r\nHost: {host}\r\nOrigin: http://{host}\r\nX-Kiko-Console-Capability: {}\r\nX-Kiko-Session-Id: {session_id}\r\nX-Kiko-Session-Capability: {session_capability}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    access.to_hex(),
                    body.len(),
                    body
                ),
            )
        };

        let accepted = submit_stop(&owner_id, &owner_capability);
        assert!(accepted.starts_with("HTTP/1.1 202"), "{accepted}");
        let accepted_json: serde_json::Value = serde_json::from_str(
            accepted
                .split("\r\n\r\n")
                .nth(1)
                .expect("accepted response body"),
        )
        .unwrap();
        let response_id = accepted_json["downstream_request_id"]
            .as_str()
            .expect("owning response identity");

        let already_latched = submit_stop(&observer_id, &observer_capability);
        assert!(
            already_latched.starts_with("HTTP/1.1 423"),
            "{already_latched}"
        );
        let already_latched_body = already_latched
            .split("\r\n\r\n")
            .nth(1)
            .expect("locked response body");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(already_latched_body).unwrap(),
            serde_json::json!({"error": "software_safety_stop_latched"})
        );
        assert!(!already_latched_body.contains("downstream_request_id"));

        let foreign_record = request(
            address,
            &format!(
                "GET /api/v1/responses/{response_id} HTTP/1.1\r\nHost: {host}\r\nX-Kiko-Console-Capability: {}\r\nX-Kiko-Session-Id: {observer_id}\r\nX-Kiko-Session-Capability: {observer_capability}\r\nConnection: close\r\n\r\n",
                access.to_hex(),
            ),
        );
        assert!(
            foreign_record.starts_with("HTTP/1.1 404"),
            "{foreign_record}"
        );
        let observer_snapshot = request(
            address,
            &format!(
                "GET /api/v1/snapshot HTTP/1.1\r\nHost: {host}\r\nX-Kiko-Console-Capability: {}\r\nX-Kiko-Session-Id: {observer_id}\r\nX-Kiko-Session-Capability: {observer_capability}\r\nConnection: close\r\n\r\n",
                access.to_hex(),
            ),
        );
        assert!(
            observer_snapshot.starts_with("HTTP/1.1 200"),
            "{observer_snapshot}"
        );
        assert!(observer_snapshot.contains("\"software_safety_stop_latched\":true"));

        assert!(server.shutdown().unwrap().graceful_shutdown);
    }

    #[test]
    fn capability_compare_and_host_origin_checks_are_exact() {
        let capability = OperatorConsoleAccessCapability::parse([7; 32]).unwrap();
        assert!(capability.constant_time_matches(capability));
        assert!(
            !capability
                .constant_time_matches(OperatorConsoleAccessCapability::parse([8; 32]).unwrap())
        );
        assert!(valid_host("127.0.0.1:8123", 8123));
        assert!(!valid_host("0.0.0.0:8123", 8123));
        assert!(!valid_host("127.0.0.1:9", 8123));
    }

    #[test]
    fn embedded_map_renderer_uses_the_five_argument_draw_image_overload() {
        let expected = "ctx.drawImage(\n      state.gridRaster,\n      transform.offsetX,\n      transform.offsetY,\n      transform.displayWidth,\n      transform.displayHeight,\n    );";
        assert!(APP_JS.contains(expected));
        assert!(!APP_JS.contains(
            "state.gridRaster,\n      transform.offsetX,\n      transform.offsetX,\n      transform.offsetY"
        ));
    }

    #[test]
    fn embedded_console_bounds_requests_and_inhibits_stale_snapshots() {
        assert!(APP_JS.contains("const API_REQUEST_TIMEOUT_MILLISECONDS = 750;"));
        assert!(APP_JS.contains("const SNAPSHOT_STALE_AFTER_MILLISECONDS = 1500;"));
        assert!(APP_JS.contains("const abort = new AbortController();"));
        assert!(APP_JS.contains("signal: abort.signal,"));
        assert!(APP_JS.contains("const pendingResponseBodies = new WeakMap();"));
        assert!(APP_JS.contains("return await consume();"));
        assert!(APP_JS.contains("return responseJson(response);"));
        assert!(APP_JS.contains("await responseArrayBuffer(response)"));
        assert!(APP_JS.contains("driveSafety.requestTimeoutMessage("));
        assert!(APP_JS.contains("kind: \"snapshot_observed\""));
        assert!(APP_JS.contains("kind: \"transport_failed\""));
        assert!(APP_JS.contains("await executeDriveSafetyEffects(transition.effects);"));
        assert!(VIEW_MODEL_JS.contains("now - lastAdvance > staleAfter;"));
        assert!(VIEW_MODEL_JS.contains("connectionKind: \"stale\""));
        assert!(VIEW_MODEL_JS.contains(r#"pill: "STATE STALE""#));
        assert!(VIEW_MODEL_JS.contains("connectionKind: \"disconnected\""));
        assert!(VIEW_MODEL_JS.contains("localInhibit: true"));
        assert!(VIEW_MODEL_JS.contains("snapshotFresh: false"));
        assert!(VIEW_MODEL_JS.contains("releaseManualBestEffort"));
        assert!(VIEW_MODEL_JS.contains(r#"pill: "CONNECTION LOST""#));
    }

    #[test]
    fn embedded_console_routes_browser_loss_and_terminal_stop_through_pure_safety_logic() {
        assert!(VIEW_MODEL_JS.contains("function reduceDriveSafety(state, event)"));
        assert!(VIEW_MODEL_JS.contains("function reduceTerminalStop(state, outcome)"));
        assert!(
            APP_JS.contains("const transition = driveSafety.reduce(state.driveSafety, event);")
        );
        for expected in [
            r#"window.addEventListener("blur", () => releaseForLifecycleLoss("blur"));"#,
            r#"window.addEventListener("offline", () => releaseForLifecycleLoss("offline"));"#,
            r#"window.addEventListener("pagehide", () => releaseForLifecycleLoss("pagehide"));"#,
            r#"releaseForLifecycleLoss("visibility_hidden");"#,
            r#"kind: "key_released""#,
            "driveSafety.createTerminalStopState()",
            r#"release ? "release_confirmed" : "release_unavailable""#,
            r#""terminal_stop_confirmed""#,
        ] {
            assert!(
                APP_JS.contains(expected),
                "missing safety wiring: {expected}"
            );
        }
        let drive_loop_start = APP_JS
            .find("async function driveLoop(generation)")
            .expect("manual drive loop");
        let drive_loop_end = APP_JS[drive_loop_start..]
            .find("\n  function ensureDriveLoop()")
            .map(|offset| drive_loop_start + offset)
            .expect("manual drive loop boundary");
        let drive_loop = &APP_JS[drive_loop_start..drive_loop_end];
        assert!(drive_loop.contains("if (error instanceof ApiError)"));
        assert!(drive_loop.contains("await failClosedForTransport(error);"));
        assert_eq!(
            APP_JS
                .matches("await failClosedForTransport(error);")
                .count(),
            2,
            "polling and manual drive must share one fail-closed transport path"
        );
    }

    #[test]
    fn embedded_console_distinguishes_local_safety_inhibit_from_server_latch() {
        let safety_handler_start = APP_JS
            .find(r#"$("software-stop").addEventListener("click", async () => {"#)
            .expect("software-stop click handler");
        let safety_handler_end = APP_JS[safety_handler_start..]
            .find("\n  function health(")
            .map(|offset| safety_handler_start + offset)
            .expect("software-stop handler terminates before health rendering");
        let safety_handler = &APP_JS[safety_handler_start..safety_handler_end];
        assert!(APP_JS.contains("function applyLocalSafetyInhibit()"));
        assert!(APP_JS.contains("LOCAL BROWSER CONTROLS INHIBITED. Server latch is unconfirmed"));
        assert!(APP_JS.contains("snapshot?.software_safety_signal_state"));
        assert!(APP_JS.contains("runtime_drained_awaiting_completion:"));
        assert!(safety_handler.contains("await submit({ kind: \"software_safety_stop\" });"));
        assert!(APP_JS.contains("class ApiError extends Error"));
        assert!(safety_handler.contains("error.status === 423"));
        assert!(safety_handler.contains("error.code === \"software_safety_stop_latched\""));
        assert!(safety_handler.contains("state.lastResponseId = null;"));
        assert!(safety_handler.contains("already latched outside this session"));
        assert!(safety_handler.contains("no session-scoped receipt"));
        assert!(
            safety_handler
                .contains("// the one-way local inhibit, but still attempt the ordinary manual")
        );
        assert!(safety_handler.contains("await releaseManual({ bestEffort: true });"));
        assert!(!APP_JS.contains("SOFTWARE STOP LATCHED for this process."));
    }

    #[test]
    fn embedded_console_exposes_save_map_only_through_the_exact_stop_gate() {
        assert_eq!(INDEX_HTML.matches(r#"data-intent="save_map""#).count(), 1);
        assert!(
            INDEX_HTML
                .contains(r#"<button data-intent="save_map">Finalize map &amp; stop</button>"#)
        );
        let handler_start = APP_JS
            .find(r#"document.querySelectorAll("[data-intent]").forEach((button) => {"#)
            .expect("operating-intent handler");
        let handler_end = APP_JS[handler_start..]
            .find("\n  function applyLocalSafetyInhibit(")
            .map(|offset| handler_start + offset)
            .expect("operating-intent handler terminates before safety state");
        let handler = &APP_JS[handler_start..handler_end];
        assert!(
            handler.contains(
                "} else {\n          await ensureExactStopped();\n          await submit({ kind: intent });\n        }"
            ),
            "save_map must use the generic stopped-intent path"
        );
    }

    #[test]
    fn embedded_console_restarts_a_rapid_manual_regrab_after_old_release() {
        let finally = APP_JS
            .find("} finally {\n      state.driveLoopRunning = false;")
            .expect("drive loop has one explicit terminal handoff");
        let restart = APP_JS
            .find("if (state.held.size && !state.driveSafety.localInhibit) ensureDriveLoop();")
            .expect("held desired state is restarted after the old loop");
        assert!(restart > finally);
        assert!(
            APP_JS.contains("if (generation === state.driveGeneration) clearDriveUi();"),
            "an obsolete generation must not clear a newer held desired state"
        );
    }

    #[test]
    fn embedded_console_does_not_reacquire_for_cancelled_held_directions() {
        let ensure_start = APP_JS
            .find("function ensureDriveLoop()")
            .expect("manual drive loop admission");
        let ensure_end = APP_JS[ensure_start..]
            .find("\n  function startDrive(")
            .map(|offset| ensure_start + offset)
            .expect("manual drive loop admission ends before input handling");
        let ensure = &APP_JS[ensure_start..ensure_end];
        let cancelled = ensure
            .find("if (!desiredManualIntent()) return;")
            .expect("cancelled held directions remain released");
        let spawn = ensure
            .find("void driveLoop(generation);")
            .expect("admitted nonzero intention starts the loop");
        assert!(
            cancelled < spawn,
            "the exact-zero held intention must be rejected before reacquiring authority"
        );
    }

    #[test]
    fn embedded_console_separates_global_actuation_from_session_response() {
        assert!(INDEX_HTML.contains(r#"id="request-state""#));
        assert!(INDEX_HTML.contains(r#"id="global-request-state""#));
        assert!(INDEX_HTML.contains(r#"id="actuation-request-state""#));
        assert!(APP_JS.contains("const requested = snapshot.last_requested;"));
        assert!(APP_JS.contains("const actuation = snapshot.last_requested_actuation;"));
        assert!(APP_JS.contains("actuation.downstream_request_id"));
        assert!(APP_JS.contains("actuation.decision_id"));
        assert!(APP_JS.contains("actuation.left_timer_pwm_percent"));
        assert!(APP_JS.contains("actuation.right_timer_pwm_percent"));
    }

    #[test]
    fn embedded_console_exposes_one_typed_unified_status_surface() {
        let model_script = r#"<script src="/assets/view-model.js"></script>"#;
        let app_script = r#"<script src="/assets/app.js" defer></script>"#;
        let model_index = INDEX_HTML.find(model_script).expect("view model asset");
        let app_index = INDEX_HTML.find(app_script).expect("application asset");
        assert!(
            model_index < app_index,
            "view model must load before the application"
        );
        for id in [
            "requested-owner-state",
            "readiness-state",
            "physical-estop-state",
            "fault-state",
            "telemetry-state",
            "map-freshness",
            "mode-state",
        ] {
            assert_eq!(
                INDEX_HTML.matches(&format!(r#"id="{id}""#)).count(),
                1,
                "{id} must be a unique dashboard projection",
            );
        }
        assert!(VIEW_MODEL_JS.contains("function parseConsoleSnapshot(raw)"));
        assert!(VIEW_MODEL_JS.contains("map.grid geometry contract is unsupported"));
        assert!(VIEW_MODEL_JS.contains("function authorityView(snapshot, sessionId)"));
        assert!(VIEW_MODEL_JS.contains("function readinessView(snapshot)"));
        assert!(VIEW_MODEL_JS.contains("function mpcView(snapshot)"));
        assert!(VIEW_MODEL_JS.contains("function physicalStopView(snapshot)"));
        assert!(VIEW_MODEL_JS.contains("function faultView(snapshot)"));
        assert!(APP_JS.contains(
            "const snapshot = model.parseConsoleSnapshot(await responseJson(response));"
        ));
        assert!(APP_JS.contains("renderAuthorityAndPipeline(snapshot);"));
        assert!(APP_JS.contains("setConnectionView(transition.state.connectionKind);"));
        assert!(APP_JS.contains("setConnectionView(transition.state.connectionKind, detail);"));
        assert!(APP_JS.contains(
            "Point goals require a currently free cell; selected cell is ${selectedCell}."
        ));
    }

    #[test]
    fn embedded_console_inhibits_stale_persistence_but_keeps_stop_actions_available() {
        assert!(APP_JS.contains(
            "&& [\"arm\", \"autonomous_frontier_explore\", \"save_map\"].includes(intent)"
        ));
        assert!(APP_JS.contains(
            "production && !terminal && !state.driveSafety.localInhibit && mapAvailable"
        ));
        assert!(APP_JS.contains("disarm: production && !terminal"));
        assert!(APP_JS.contains("autonomous_map_only: production && !terminal"));
        assert!(APP_JS.contains("stop: !terminal"));
        let handler_start = APP_JS
            .find("document.querySelectorAll(\"[data-intent]\").forEach((button) => {")
            .expect("mode handler");
        let handler_end = APP_JS[handler_start..]
            .find("function applyLocalSafetyInhibit()")
            .map(|offset| handler_start + offset)
            .expect("mode handler end");
        let handler = &APP_JS[handler_start..handler_end];
        assert!(handler.contains("await ensureExactStopped();"));
        assert!(handler.contains("await submit({ kind: intent });"));
    }

    #[cfg(unix)]
    #[test]
    fn capability_persistence_is_private_atomic_and_never_reuses_old_value() {
        let directory = PrivateTestDirectory::create();
        let target = directory.0.join("operator-console.cap");
        let first_owner =
            OperatorConsoleAccessCapability::generate_and_persist_new(&target).unwrap();
        let first = first_owner.access_capability();
        let first_text = fs::read_to_string(&target).unwrap();
        assert_eq!(first_text, format!("{}\n", first.to_hex()));
        assert_eq!(
            fs::metadata(&target).unwrap().permissions().mode() & 0o7777,
            PRIVATE_CAPABILITY_FILE_MODE
        );

        let second_owner =
            OperatorConsoleAccessCapability::generate_and_persist_new(&target).unwrap();
        let second = second_owner.access_capability();
        assert_ne!(first, second);
        assert_eq!(
            fs::read_to_string(&target).unwrap(),
            format!("{}\n", second.to_hex())
        );
        assert!(fs::read_dir(&directory.0).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".tmp")
        }));
        assert_eq!(
            first_owner.cleanup(),
            OperatorConsoleCapabilityCleanupEvidence::RefusedIdentityMismatch
        );
        assert!(target.exists());
        assert_eq!(
            second_owner.cleanup(),
            OperatorConsoleCapabilityCleanupEvidence::ExactEntryRemovedAndParentSynced
        );
        assert!(!target.exists());
    }

    #[cfg(unix)]
    #[test]
    fn capability_persistence_rejects_public_parent_and_symlink_target() {
        use std::os::unix::fs::symlink;

        let directory = PrivateTestDirectory::create();
        let target = directory.0.join("operator-console.cap");
        fs::set_permissions(&directory.0, fs::Permissions::from_mode(0o755)).unwrap();
        assert!(matches!(
            OperatorConsoleAccessCapability::generate_and_persist_new(&target),
            Err(OperatorConsoleCapabilityPersistError::ParentNotPrivate { .. })
        ));
        fs::set_permissions(
            &directory.0,
            fs::Permissions::from_mode(PRIVATE_CAPABILITY_PARENT_MODE),
        )
        .unwrap();
        symlink("/dev/null", &target).unwrap();
        assert!(matches!(
            OperatorConsoleAccessCapability::generate_and_persist_new(&target),
            Err(
                OperatorConsoleCapabilityPersistError::UnsafeExistingTarget {
                    reason: OperatorConsoleCapabilityTargetSafetyError::NotRegularFile,
                    ..
                }
            )
        ));
    }

    #[test]
    fn hanging_request_is_forcibly_cancelled_with_bounded_join_evidence() {
        let (console, _receiver) = operator_console(
            OperatorConsoleLimits::default(),
            OperatorConsoleSnapshot::unknown(ConsoleSnapshotRevision::parse(1).unwrap()),
        );
        let access = OperatorConsoleAccessCapability::parse([0x44; 32]).unwrap();
        let origin = super::super::AgentControlMonotonicOrigin::new(
            Instant::now(),
            HostMonotonicTimestamp::from_nanos(1_000_000_000),
        );
        let config = OperatorConsoleHttpServerConfig::parse(
            OperatorConsoleBind::parse("127.0.0.1:0".parse().unwrap()).unwrap(),
            access,
            origin,
            Duration::from_millis(20),
        )
        .unwrap();
        let mut server = OperatorConsoleHttpServer::start(config, Arc::new(console)).unwrap();
        let address = server.bound_addr();
        let host = format!("127.0.0.1:{}", address.port());
        let mut hanging = TcpStream::connect(address).unwrap();
        hanging
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        write!(
            hanging,
            "POST /api/v1/sessions HTTP/1.1\r\nHost: {host}\r\nOrigin: http://{host}\r\nX-Kiko-Console-Capability: {}\r\nContent-Type: application/json\r\nContent-Length: 128\r\nExpect: 100-continue\r\n\r\n",
            access.to_hex()
        )
        .unwrap();
        hanging.flush().unwrap();
        let mut interim = [0_u8; 128];
        let read = hanging
            .read(&mut interim)
            .expect("server accepts headers and requests the pending body");
        assert!(
            String::from_utf8_lossy(&interim[..read]).starts_with("HTTP/1.1 100 Continue"),
            "shutdown race barrier must observe request acceptance"
        );

        let began = Instant::now();
        let exit = server.shutdown().unwrap();
        assert!(exit.forced_shutdown);
        assert!(!exit.graceful_shutdown);
        assert!(began.elapsed() < HTTP_SHUTDOWN_TIMEOUT);
    }

    #[test]
    fn join_timeout_retains_http_owner_for_retry() {
        let (console, _receiver) = operator_console(
            OperatorConsoleLimits::default(),
            OperatorConsoleSnapshot::unknown(ConsoleSnapshotRevision::parse(1).unwrap()),
        );
        let access = OperatorConsoleAccessCapability::parse([0x45; 32]).unwrap();
        let origin = super::super::AgentControlMonotonicOrigin::new(
            Instant::now(),
            HostMonotonicTimestamp::from_nanos(1_000_000_000),
        );
        let config = OperatorConsoleHttpServerConfig::parse(
            OperatorConsoleBind::parse("127.0.0.1:0".parse().unwrap()).unwrap(),
            access,
            origin,
            Duration::from_millis(20),
        )
        .unwrap();
        let mut server = OperatorConsoleHttpServer::start(config, Arc::new(console)).unwrap();
        let address = server.bound_addr();
        let host = format!("127.0.0.1:{}", address.port());
        let mut hanging = TcpStream::connect(address).unwrap();
        hanging
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        write!(
            hanging,
            "POST /api/v1/sessions HTTP/1.1\r\nHost: {host}\r\nOrigin: http://{host}\r\nX-Kiko-Console-Capability: {}\r\nContent-Type: application/json\r\nContent-Length: 128\r\nExpect: 100-continue\r\n\r\n",
            access.to_hex()
        )
        .unwrap();
        hanging.flush().unwrap();
        let mut interim = [0_u8; 128];
        let read = hanging.read(&mut interim).unwrap();
        assert!(String::from_utf8_lossy(&interim[..read]).starts_with("HTTP/1.1 100 Continue"));

        assert!(matches!(
            server.shutdown_with_timeout(Duration::from_millis(20)),
            Err(OperatorConsoleHttpJoinError::TimedOut)
        ));
        assert!(
            server.join.is_some(),
            "timeout must retain the joinable HTTP owner"
        );
        let exit = server.shutdown().expect("retry joins retained owner");
        assert!(exit.forced_shutdown);
    }
}
