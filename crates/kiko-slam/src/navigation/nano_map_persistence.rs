//! Bounded, durable map persistence for the production Nano runtime.
//!
//! The owner retains exactly one live, epoch-bound occupancy snapshot. Saves
//! are synchronous and require `&mut self`, so snapshot replacement and a
//! second save cannot overlap the durable temporary-file/sync/rename/directory
//! sync sequence.
//!
//! Production warm start resolves one atomically selected, finalized session.
//! The selection binds the session manifest and its session-local occupancy
//! artifact by exact byte lengths and SHA-256 digests. Replay rechecks those
//! bindings before and after reconstruction, then still requires exact final
//! occupancy equality. None of this claims current-camera localization; that
//! remains a separate live relocalization transition.

use std::fmt;
use std::fs::{self, File};
use std::io;
use std::os::unix::fs::DirBuilderExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

#[cfg(feature = "nano-agent")]
use std::ffi::OsStr;
#[cfg(feature = "nano-agent")]
use std::io::{Read, Seek, SeekFrom, Write};
#[cfg(feature = "nano-agent")]
use std::os::fd::AsFd;
#[cfg(feature = "nano-agent")]
use std::os::unix::fs::OpenOptionsExt;
#[cfg(feature = "nano-agent")]
use std::path::Component;
#[cfg(feature = "nano-agent")]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "nano-agent")]
use rustix::fs::{
    AtFlags, FileType, Mode, OFlags, RenameFlags, fstat, fsync, openat, renameat, renameat_with,
    statat, unlinkat,
};
#[cfg(feature = "nano-agent")]
use rustix::io::Errno;
#[cfg(feature = "nano-agent")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "nano-agent")]
use sha2::{Digest, Sha256};

use crate::dense::occupancy::OccupancyGridSnapshot;
#[cfg(feature = "nano-agent")]
use crate::dense::occupancy_persistence::load_persisted_occupancy_map_from_reader;
#[cfg(feature = "nano-agent")]
use crate::dense::occupancy_persistence::save_occupancy_map_atomic_at;
use crate::dense::occupancy_persistence::{
    OccupancyMapEncodeError, OccupancyMapLimits, OccupancyMapLoadError, OccupancyMapSaveError,
    OccupancyReplayBindError, PersistedOccupancyMap, ReplayMatchedOccupancyMap,
    ReplayOccupancyEvidence, occupancy_map_encoded_len,
};
#[cfg(not(feature = "nano-agent"))]
use crate::dense::occupancy_persistence::{
    load_persisted_occupancy_map, save_occupancy_map_atomic,
};
use crate::map::MapInstanceId;

use super::agent_config::{NanoMapPersistenceConfig, NanoMapWarmStart};
use super::control_api::{AgentControlCommandKindV1, AgentControlRejectionCodeV1};
use super::control_socket::{AgentControlClaimedRequest, AgentControlDispatchResponseError};
use super::ingress::{CurrentMapEpochBinding, RecordedMapEpochId};
use super::nano_bootstrap::NanoBootstrapRoots;
use super::nano_state_quota::{
    NanoStateQuotaCommitError, NanoStateQuotaOwner, NanoStateQuotaReserveError,
    NanoStateQuotaWriteReceipt,
};

#[cfg(feature = "nano-agent")]
const NANO_WARM_SELECTION_SCHEMA_VERSION: u32 = 1;
#[cfg(feature = "nano-agent")]
const NANO_WARM_SELECTION_FILE: &str = "selected-warm-start-v1.json";
#[cfg(feature = "nano-agent")]
const NANO_WARM_OCCUPANCY_FILE: &str = "occupancy.kmap";
#[cfg(feature = "nano-agent")]
const DATASET_MANIFEST_FILE: &str = "manifest.json";
#[cfg(feature = "nano-agent")]
pub const MAX_NANO_WARM_SELECTION_BYTES: u64 = 4 * 1_024;
#[cfg(feature = "nano-agent")]
pub const MAX_NANO_DATASET_MANIFEST_BYTES: u64 =
    crate::dataset::MAX_PRODUCTION_DATASET_MANIFEST_BYTES;
#[cfg(feature = "nano-agent")]
const MAX_NANO_SELECTED_OCCUPANCY_BYTES: u64 = 256 * 1_024 * 1_024;
const SHA256_BYTES: usize = 32;
#[cfg(feature = "nano-agent")]
const SHA256_HEX_BYTES: usize = SHA256_BYTES * 2;
#[cfg(feature = "nano-agent")]
const MAX_DATASET_DIRECTORY_NAME_BYTES: usize = 128;
#[cfg(feature = "nano-agent")]
static NEXT_SELECTION_TEMPORARY: AtomicU64 = AtomicU64::new(0);

/// Filesystem roles reported by persistence path admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoMapPathRole {
    StateRoot,
    SaveParent,
    SaveSnapshot,
    WarmOccupancySnapshot,
    WarmSlamDatasetDirectory,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoMapPathOperation {
    Inspect,
    Canonicalize,
    CreateDirectChildDirectory,
    OpenDirectoryForSync,
    SyncDirectory,
}

impl fmt::Display for NanoMapPathOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Inspect => "inspect",
            Self::Canonicalize => "canonicalize",
            Self::CreateDirectChildDirectory => "create direct-child directory",
            Self::OpenDirectoryForSync => "open directory for synchronization",
            Self::SyncDirectory => "synchronize directory",
        })
    }
}

/// Exact path failure; no variant silently substitutes another location.
#[derive(Debug)]
pub enum NanoMapPersistencePathError {
    OutsideStateRoot {
        role: NanoMapPathRole,
        state_root: PathBuf,
        configured: PathBuf,
    },
    AliasesStateRoot {
        role: NanoMapPathRole,
        state_root: PathBuf,
    },
    MissingParent {
        role: NanoMapPathRole,
        path: PathBuf,
    },
    SaveParentCreationIsNotDirectChild {
        state_root: PathBuf,
        save_parent: PathBuf,
    },
    Missing {
        role: NanoMapPathRole,
        path: PathBuf,
    },
    Symlink {
        role: NanoMapPathRole,
        path: PathBuf,
    },
    NotDirectory {
        role: NanoMapPathRole,
        path: PathBuf,
    },
    NotRegularFile {
        role: NanoMapPathRole,
        path: PathBuf,
    },
    FilesystemPathIsNotCanonical {
        role: NanoMapPathRole,
        configured: PathBuf,
        resolved: PathBuf,
    },
    FilesystemObjectChanged {
        role: NanoMapPathRole,
        path: PathBuf,
        expected_device: u64,
        expected_inode: u64,
        actual_device: u64,
        actual_inode: u64,
    },
    Io {
        operation: NanoMapPathOperation,
        role: NanoMapPathRole,
        path: PathBuf,
        source: io::Error,
    },
}

impl fmt::Display for NanoMapPersistencePathError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutsideStateRoot {
                role,
                state_root,
                configured,
            } => write!(
                formatter,
                "{role:?} path '{}' is outside Nano state root '{}'",
                configured.display(),
                state_root.display()
            ),
            Self::AliasesStateRoot { role, state_root } => write!(
                formatter,
                "{role:?} path may not alias Nano state root '{}'",
                state_root.display()
            ),
            Self::MissingParent { role, path } => write!(
                formatter,
                "{role:?} path '{}' has no parent directory",
                path.display()
            ),
            Self::SaveParentCreationIsNotDirectChild {
                state_root,
                save_parent,
            } => write!(
                formatter,
                "refusing to create save parent '{}' because it is not one direct child beneath state root '{}'",
                save_parent.display(),
                state_root.display()
            ),
            Self::Missing { role, path } => {
                write!(
                    formatter,
                    "{role:?} path '{}' does not exist",
                    path.display()
                )
            }
            Self::Symlink { role, path } => write!(
                formatter,
                "{role:?} path '{}' is a symbolic link",
                path.display()
            ),
            Self::NotDirectory { role, path } => {
                write!(
                    formatter,
                    "{role:?} path '{}' is not a directory",
                    path.display()
                )
            }
            Self::NotRegularFile { role, path } => write!(
                formatter,
                "{role:?} path '{}' is not a regular file",
                path.display()
            ),
            Self::FilesystemPathIsNotCanonical {
                role,
                configured,
                resolved,
            } => write!(
                formatter,
                "{role:?} path '{}' resolves to '{}'",
                configured.display(),
                resolved.display()
            ),
            Self::FilesystemObjectChanged {
                role,
                path,
                expected_device,
                expected_inode,
                actual_device,
                actual_inode,
            } => write!(
                formatter,
                "{role:?} path '{}' changed filesystem identity from {expected_device}:{expected_inode} to {actual_device}:{actual_inode}",
                path.display()
            ),
            Self::Io {
                operation,
                role,
                path,
                source,
            } => write!(
                formatter,
                "failed to {operation} {role:?} path '{}': {source}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for NanoMapPersistencePathError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FilesystemObjectIdentity {
    device: u64,
    inode: u64,
}

impl FilesystemObjectIdentity {
    fn from_metadata(metadata: &fs::Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
        }
    }
}

/// Exact live snapshot identity retained by the owner.
///
/// The recorded epoch orders mapping-session transitions. The process-local
/// map ID proves that the snapshot belongs to that epoch, and the revision
/// orders snapshots within the epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NanoMapSnapshotIdentity {
    map_epoch_id: RecordedMapEpochId,
    map_instance_id: MapInstanceId,
    revision: u64,
}

impl NanoMapSnapshotIdentity {
    pub const fn map_epoch_id(self) -> RecordedMapEpochId {
        self.map_epoch_id
    }

    pub const fn map_instance_id(self) -> MapInstanceId {
        self.map_instance_id
    }

    pub const fn revision(self) -> u64 {
        self.revision
    }
}

/// Final accepted map identity parsed back from the synchronized navigation
/// journal. This deliberately excludes the process-local map instance: the
/// journal's wire contract records only the stable epoch and mapper revision.
#[cfg(feature = "nano-agent")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NanoFinalizedJournalMapIdentity {
    map_epoch_id: RecordedMapEpochId,
    revision: u64,
}

#[cfg(feature = "nano-agent")]
impl NanoFinalizedJournalMapIdentity {
    pub const fn new(map_epoch_id: RecordedMapEpochId, revision: u64) -> Self {
        Self {
            map_epoch_id,
            revision,
        }
    }

    pub const fn map_epoch_id(self) -> RecordedMapEpochId {
        self.map_epoch_id
    }

    pub const fn revision(self) -> u64 {
        self.revision
    }
}

#[cfg(feature = "nano-agent")]
fn require_finalized_journal_map_matches_retained(
    finalized_map_identity: Option<NanoFinalizedJournalMapIdentity>,
    retained: NanoMapSnapshotIdentity,
) -> Result<(), NanoWarmCheckpointError> {
    let finalized =
        finalized_map_identity.ok_or(NanoWarmCheckpointError::FinalizedJournalHasNoAcceptedMap)?;
    if finalized.map_epoch_id() != retained.map_epoch_id()
        || finalized.revision() != retained.revision()
    {
        return Err(NanoWarmCheckpointError::FinalizedJournalMapMismatch {
            finalized,
            retained_epoch_id: retained.map_epoch_id(),
            retained_revision: retained.revision(),
        });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NanoMapRetentionOutcome {
    retained: NanoMapSnapshotIdentity,
    replaced: Option<NanoMapSnapshotIdentity>,
}

impl NanoMapRetentionOutcome {
    pub const fn retained(self) -> NanoMapSnapshotIdentity {
        self.retained
    }

    pub const fn replaced(self) -> Option<NanoMapSnapshotIdentity> {
        self.replaced
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoMapSnapshotRetentionError {
    TooManyCells {
        actual_cells: usize,
        maximum_cells: usize,
    },
    UnboundSnapshot,
    MapInstanceMismatch {
        map_epoch_id: RecordedMapEpochId,
        bound: MapInstanceId,
        snapshot: MapInstanceId,
    },
    StaleEpoch {
        retained: RecordedMapEpochId,
        offered: RecordedMapEpochId,
    },
    EpochMapInstanceChanged {
        map_epoch_id: RecordedMapEpochId,
        retained: MapInstanceId,
        offered: MapInstanceId,
    },
    StaleRevision {
        map_epoch_id: RecordedMapEpochId,
        retained: u64,
        offered: u64,
    },
}

impl fmt::Display for NanoMapSnapshotRetentionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "cannot retain Nano occupancy snapshot: {self:?}")
    }
}

impl std::error::Error for NanoMapSnapshotRetentionError {}

struct RetainedNanoMapSnapshot {
    identity: NanoMapSnapshotIdentity,
    snapshot: OccupancyGridSnapshot,
}

/// Evidence returned only after the dense atomic writer has synchronized the
/// temporary file, published it by rename, and synchronized its parent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoMapSaveReceipt {
    identity: NanoMapSnapshotIdentity,
    destination: PathBuf,
    quota_verification: Option<Box<NanoStateQuotaWriteReceipt>>,
}

impl NanoMapSaveReceipt {
    pub const fn identity(&self) -> NanoMapSnapshotIdentity {
        self.identity
    }

    pub fn destination(&self) -> &Path {
        &self.destination
    }

    /// Present only when the launch-bound map quota owner admitted the exact
    /// planned length and transient headroom before publication, then verified
    /// the exact length and post-save free-space floor afterwards.
    pub fn quota_verification(&self) -> Option<&NanoStateQuotaWriteReceipt> {
        self.quota_verification.as_deref()
    }
}

#[derive(Debug)]
pub enum NanoMapSaveError {
    NoSnapshot,
    StaleSelection {
        requested: NanoMapSnapshotIdentity,
        retained: NanoMapSnapshotIdentity,
    },
    Path(NanoMapPersistencePathError),
    QuotaReservation(NanoStateQuotaReserveError),
    Persistence(OccupancyMapSaveError),
    PublishedButPersistenceFailed {
        source: OccupancyMapSaveError,
        quota_verification: Result<Box<NanoStateQuotaWriteReceipt>, Box<NanoStateQuotaCommitError>>,
    },
    PublishedButQuotaVerificationFailed {
        receipt: NanoMapSaveReceipt,
        source: Box<NanoStateQuotaCommitError>,
    },
}

impl NanoMapSaveError {
    fn rejection(&self) -> (AgentControlRejectionCodeV1, bool) {
        match self {
            Self::NoSnapshot => (AgentControlRejectionCodeV1::MapUnavailable, true),
            Self::StaleSelection { .. } => (AgentControlRejectionCodeV1::StaleMapSelection, true),
            Self::Path(_) | Self::QuotaReservation(_) => {
                (AgentControlRejectionCodeV1::PersistenceFailed, false)
            }
            Self::Persistence(source) => (
                AgentControlRejectionCodeV1::PersistenceFailed,
                persistence_error_may_be_retryable(source),
            ),
            Self::PublishedButPersistenceFailed { .. }
            | Self::PublishedButQuotaVerificationFailed { .. } => {
                (AgentControlRejectionCodeV1::PersistenceFailed, false)
            }
        }
    }

    /// A rename may have published the destination even though directory sync
    /// failed. Such an error still never produces a completion response.
    pub fn destination_may_have_been_published(&self) -> bool {
        matches!(
            self,
            Self::Persistence(source) if persistence_error_was_published(source)
        ) || matches!(
            self,
            Self::PublishedButPersistenceFailed { .. }
                | Self::PublishedButQuotaVerificationFailed { .. }
        )
    }
}

impl fmt::Display for NanoMapSaveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSnapshot => formatter.write_str("no live occupancy snapshot is retained"),
            Self::StaleSelection {
                requested,
                retained,
            } => write!(
                formatter,
                "requested map snapshot {requested:?} is not the retained latest snapshot {retained:?}"
            ),
            Self::Path(source) => write!(formatter, "map save path is not admitted: {source}"),
            Self::QuotaReservation(source) => {
                write!(formatter, "map save quota was not admitted: {source}")
            }
            Self::Persistence(source) => write!(formatter, "durable map save failed: {source}"),
            Self::PublishedButPersistenceFailed {
                source,
                quota_verification: Ok(_),
            } => write!(
                formatter,
                "map destination was published and quota-verified, but durable map save failed: {source}"
            ),
            Self::PublishedButPersistenceFailed {
                source,
                quota_verification: Err(quota),
            } => write!(
                formatter,
                "map destination was published, durable map save failed ({source}), and quota verification also failed: {quota}"
            ),
            Self::PublishedButQuotaVerificationFailed { source, .. } => write!(
                formatter,
                "durable map publication completed but quota verification failed: {source}"
            ),
        }
    }
}

impl std::error::Error for NanoMapSaveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Path(source) => Some(source),
            Self::QuotaReservation(source) => Some(source),
            Self::Persistence(source) => Some(source),
            Self::PublishedButPersistenceFailed { source, .. } => Some(source),
            Self::PublishedButQuotaVerificationFailed { source, .. } => Some(source),
            Self::NoSnapshot | Self::StaleSelection { .. } => None,
        }
    }
}

fn persistence_error_was_published(source: &OccupancyMapSaveError) -> bool {
    matches!(source, OccupancyMapSaveError::Io(source) if source.published())
}

fn persistence_error_may_be_retryable(source: &OccupancyMapSaveError) -> bool {
    match source {
        OccupancyMapSaveError::Io(_) | OccupancyMapSaveError::TemporaryNameCollisions { .. } => {
            true
        }
        OccupancyMapSaveError::Encode(OccupancyMapEncodeError::AllocationFailed { .. }) => true,
        OccupancyMapSaveError::Encode(OccupancyMapEncodeError::EncodedLengthOverflow {
            ..
        })
        | OccupancyMapSaveError::InvalidDestination { .. } => false,
    }
}

/// Save plus control-response outcome. A receipt in the final variant means
/// disk publication completed but the completion response did not reach the
/// socket client.
#[derive(Debug)]
pub enum NanoMapSaveCommandError {
    WrongCommandRejected {
        actual: AgentControlCommandKindV1,
    },
    WrongCommandResponseFailed {
        actual: AgentControlCommandKindV1,
        response: AgentControlDispatchResponseError,
    },
    SaveRejected {
        source: NanoMapSaveError,
    },
    SaveAndRejectionResponseFailed {
        source: NanoMapSaveError,
        response: AgentControlDispatchResponseError,
    },
    SavedButCompletionResponseFailed {
        receipt: NanoMapSaveReceipt,
        response: AgentControlDispatchResponseError,
    },
}

impl fmt::Display for NanoMapSaveCommandError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Nano save-map command failed: {self:?}")
    }
}

impl std::error::Error for NanoMapSaveCommandError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::WrongCommandResponseFailed { response, .. }
            | Self::SavedButCompletionResponseFailed { response, .. } => Some(response),
            Self::SaveRejected { source } | Self::SaveAndRejectionResponseFailed { source, .. } => {
                Some(source)
            }
            Self::WrongCommandRejected { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoDatasetContentBindingStatus {
    /// Non-production compatibility builds retain the legacy pathname-only
    /// contract. The production Nano build never admits this status.
    MissingImmutableContentIdentity,
    /// A durably published selection binds the exact finalized dataset
    /// manifest and occupancy artifact by SHA-256. Exact replay equality is
    /// still required, and remains distinct from live-camera localization.
    FinalizedSelectionV1 {
        dataset_manifest_sha256: [u8; SHA256_BYTES],
        occupancy_sha256: [u8; SHA256_BYTES],
    },
}

impl NanoDatasetContentBindingStatus {
    pub const fn dataset_manifest_sha256(self) -> Option<[u8; 32]> {
        match self {
            Self::MissingImmutableContentIdentity => None,
            Self::FinalizedSelectionV1 {
                dataset_manifest_sha256,
                ..
            } => Some(dataset_manifest_sha256),
        }
    }

    pub const fn occupancy_sha256(self) -> Option<[u8; 32]> {
        match self {
            Self::MissingImmutableContentIdentity => None,
            Self::FinalizedSelectionV1 {
                occupancy_sha256, ..
            } => Some(occupancy_sha256),
        }
    }
}

impl fmt::Display for NanoDatasetContentBindingStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingImmutableContentIdentity => {
                formatter.write_str("missing_immutable_content_identity")
            }
            Self::FinalizedSelectionV1 {
                dataset_manifest_sha256,
                occupancy_sha256,
            } => {
                formatter.write_str("finalized_selection_v1:manifest_sha256=")?;
                for byte in dataset_manifest_sha256 {
                    write!(formatter, "{byte:02x}")?;
                }
                formatter.write_str(",occupancy_sha256=")?;
                for byte in occupancy_sha256 {
                    write!(formatter, "{byte:02x}")?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug)]
enum AdmittedNanoWarmStart {
    None,
    DatasetReplay {
        occupancy_snapshot_path: PathBuf,
        slam_dataset_directory_path: PathBuf,
    },
}

#[cfg(feature = "nano-agent")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NanoWarmSelectionContentIdentity {
    dataset_manifest_sha256: [u8; SHA256_BYTES],
    occupancy_sha256: [u8; SHA256_BYTES],
    dataset_manifest_bytes: u64,
    occupancy_bytes: u64,
}

#[cfg(feature = "nano-agent")]
impl NanoWarmSelectionContentIdentity {
    const fn binding_status(self) -> NanoDatasetContentBindingStatus {
        NanoDatasetContentBindingStatus::FinalizedSelectionV1 {
            dataset_manifest_sha256: self.dataset_manifest_sha256,
            occupancy_sha256: self.occupancy_sha256,
        }
    }
}

/// Durable proof that one final map revision and one finalized recording were
/// selected together. Completion never implies current-camera localization.
#[cfg(feature = "nano-agent")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoWarmCheckpointReceipt {
    staged_map: NanoMapSaveReceipt,
    occupancy_snapshot_path: PathBuf,
    dataset_directory: PathBuf,
    selection_path: PathBuf,
    binding: NanoDatasetContentBindingStatus,
}

#[cfg(feature = "nano-agent")]
impl NanoWarmCheckpointReceipt {
    pub const fn map_identity(&self) -> NanoMapSnapshotIdentity {
        self.staged_map.identity()
    }

    /// Exact immutable occupancy selected for restart.
    pub fn occupancy_snapshot_path(&self) -> &Path {
        &self.occupancy_snapshot_path
    }

    /// Launch-bound staging path at which quota publication and exact length
    /// were verified before the same inode was relocated into the finalized
    /// session. This pathname no longer names the artifact after success.
    pub fn quota_staging_path(&self) -> &Path {
        self.staged_map.destination()
    }

    pub fn quota_verification(&self) -> Option<&NanoStateQuotaWriteReceipt> {
        self.staged_map.quota_verification()
    }

    pub fn dataset_directory(&self) -> &Path {
        &self.dataset_directory
    }

    pub fn selection_path(&self) -> &Path {
        &self.selection_path
    }

    pub const fn dataset_content_binding_status(&self) -> NanoDatasetContentBindingStatus {
        self.binding
    }
}

#[cfg(feature = "nano-agent")]
#[derive(Debug)]
pub enum NanoWarmCheckpointError {
    FinalizedJournalHasNoAcceptedMap,
    FinalizedJournalMapMismatch {
        finalized: NanoFinalizedJournalMapIdentity,
        retained_epoch_id: RecordedMapEpochId,
        retained_revision: u64,
    },
    DatasetHasNoSelectionParent {
        dataset_directory: PathBuf,
    },
    DatasetPath(NanoMapPersistencePathError),
    DatasetIsNotDirectSelectionChild {
        selection_root: PathBuf,
        dataset_directory: PathBuf,
    },
    DatasetManifestMissing {
        path: PathBuf,
    },
    OccupancyAlreadyExists {
        path: PathBuf,
    },
    MissingQuotaVerification,
    SaveMap(NanoMapSaveError),
    Selection(NanoWarmSelectionError),
}

#[cfg(feature = "nano-agent")]
impl NanoWarmCheckpointError {
    fn rejection(&self) -> (AgentControlRejectionCodeV1, bool) {
        match self {
            Self::SaveMap(source) => source.rejection(),
            Self::FinalizedJournalHasNoAcceptedMap
            | Self::FinalizedJournalMapMismatch { .. }
            | Self::DatasetHasNoSelectionParent { .. }
            | Self::DatasetPath(_)
            | Self::DatasetIsNotDirectSelectionChild { .. }
            | Self::DatasetManifestMissing { .. }
            | Self::OccupancyAlreadyExists { .. }
            | Self::MissingQuotaVerification
            | Self::Selection(_) => (AgentControlRejectionCodeV1::PersistenceFailed, false),
        }
    }
}

#[cfg(feature = "nano-agent")]
impl fmt::Display for NanoWarmCheckpointError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FinalizedJournalHasNoAcceptedMap => formatter
                .write_str("finalized navigation journal has no accepted map in its final epoch"),
            Self::FinalizedJournalMapMismatch {
                finalized,
                retained_epoch_id,
                retained_revision,
            } => write!(
                formatter,
                "finalized journal map epoch/revision {}:{} does not equal retained occupancy {}:{}",
                finalized.map_epoch_id().as_u64(),
                finalized.revision(),
                retained_epoch_id.as_u64(),
                retained_revision,
            ),
            Self::DatasetHasNoSelectionParent { dataset_directory } => write!(
                formatter,
                "finalized dataset '{}' has no parent selection directory",
                dataset_directory.display()
            ),
            Self::DatasetPath(source) => {
                write!(formatter, "dataset path is not admitted: {source}")
            }
            Self::DatasetIsNotDirectSelectionChild {
                selection_root,
                dataset_directory,
            } => write!(
                formatter,
                "finalized dataset '{}' is not one direct child of warm selection root '{}'",
                dataset_directory.display(),
                selection_root.display()
            ),
            Self::DatasetManifestMissing { path } => {
                write!(
                    formatter,
                    "finalized dataset manifest '{}' is missing",
                    path.display()
                )
            }
            Self::OccupancyAlreadyExists { path } => write!(
                formatter,
                "refusing to replace immutable checkpoint occupancy '{}'",
                path.display()
            ),
            Self::MissingQuotaVerification => formatter
                .write_str("quota-bound map staging returned no quota verification evidence"),
            Self::SaveMap(source) => write!(formatter, "latest-map publication failed: {source}"),
            Self::Selection(source) => write!(formatter, "warm selection failed: {source}"),
        }
    }
}

#[cfg(feature = "nano-agent")]
impl std::error::Error for NanoWarmCheckpointError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::DatasetPath(source) => Some(source),
            Self::SaveMap(source) => Some(source),
            Self::Selection(source) => Some(source),
            Self::DatasetHasNoSelectionParent { .. }
            | Self::FinalizedJournalHasNoAcceptedMap
            | Self::FinalizedJournalMapMismatch { .. }
            | Self::DatasetIsNotDirectSelectionChild { .. }
            | Self::DatasetManifestMissing { .. }
            | Self::OccupancyAlreadyExists { .. } => None,
            Self::MissingQuotaVerification => None,
        }
    }
}

#[cfg(feature = "nano-agent")]
#[derive(Debug)]
pub enum NanoWarmCheckpointCommandError {
    WrongCommandRejected {
        actual: AgentControlCommandKindV1,
    },
    WrongCommandResponseFailed {
        actual: AgentControlCommandKindV1,
        response: AgentControlDispatchResponseError,
    },
    CheckpointRejected {
        source: NanoWarmCheckpointError,
    },
    CheckpointAndRejectionResponseFailed {
        source: NanoWarmCheckpointError,
        response: AgentControlDispatchResponseError,
    },
    CheckpointedButCompletionResponseFailed {
        receipt: Box<NanoWarmCheckpointReceipt>,
        response: AgentControlDispatchResponseError,
    },
}

#[cfg(feature = "nano-agent")]
impl fmt::Display for NanoWarmCheckpointCommandError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Nano warm-checkpoint command failed: {self:?}")
    }
}

#[cfg(feature = "nano-agent")]
impl std::error::Error for NanoWarmCheckpointCommandError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::WrongCommandResponseFailed { response, .. }
            | Self::CheckpointedButCompletionResponseFailed { response, .. } => Some(response),
            Self::CheckpointRejected { source }
            | Self::CheckpointAndRejectionResponseFailed { source, .. } => Some(source),
            Self::WrongCommandRejected { .. } => None,
        }
    }
}

#[cfg(feature = "nano-agent")]
#[derive(Debug)]
pub enum NanoWarmSelectionError {
    Path(NanoMapPersistencePathError),
    Io {
        operation: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    NotRegularFile {
        path: PathBuf,
    },
    FilesystemObjectChanged {
        path: PathBuf,
        expected_device: u64,
        expected_inode: u64,
        observed_device: u64,
        observed_inode: u64,
    },
    FileTooLarge {
        path: PathBuf,
        actual_bytes: u64,
        maximum_bytes: u64,
    },
    Truncated {
        path: PathBuf,
        expected_bytes: u64,
        actual_bytes: usize,
    },
    Json(serde_json::Error),
    UnsupportedSchema {
        actual: u32,
    },
    InvalidDatasetDirectoryName,
    InvalidSha256 {
        field: &'static str,
    },
    ZeroMapEpoch,
    DigestMismatch {
        path: PathBuf,
        expected: [u8; SHA256_BYTES],
        observed: [u8; SHA256_BYTES],
    },
    LengthMismatch {
        path: PathBuf,
        expected: u64,
        observed: u64,
    },
    MapRevisionMismatch {
        selected: u64,
        observed: u64,
    },
    MapEpochMismatch {
        selected: u64,
        observed: u64,
    },
    SelectionEncodingTooLarge {
        actual_bytes: usize,
        maximum_bytes: u64,
    },
    TemporaryNameCollisions {
        parent: PathBuf,
    },
    SelectionPublishedButDurabilityUnconfirmed {
        selection_path: PathBuf,
        operation: NanoWarmSelectionPostPublishOperation,
        source: io::Error,
    },
}

#[cfg(feature = "nano-agent")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoWarmSelectionPostPublishOperation {
    InspectPublishedSelection,
    SynchronizeSelectionRoot,
}

#[cfg(feature = "nano-agent")]
impl NanoWarmSelectionError {
    /// The selection rename completed, but parent-directory durability could
    /// not be proven. No completion response may be emitted for this state.
    pub const fn selection_may_have_been_published(&self) -> bool {
        matches!(
            self,
            Self::SelectionPublishedButDurabilityUnconfirmed { .. }
        )
    }
}

#[cfg(feature = "nano-agent")]
impl fmt::Display for NanoWarmSelectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid Nano warm selection: {self:?}")
    }
}

#[cfg(feature = "nano-agent")]
impl std::error::Error for NanoWarmSelectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Path(source) => Some(source),
            Self::Io { source, .. } => Some(source),
            Self::Json(source) => Some(source),
            Self::SelectionPublishedButDurabilityUnconfirmed { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[cfg(feature = "nano-agent")]
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoWarmSelectionV1Dto {
    schema_version: u32,
    dataset_directory_name: String,
    dataset_manifest_sha256_hex: String,
    dataset_manifest_bytes: u64,
    occupancy_file_name: String,
    occupancy_sha256_hex: String,
    occupancy_bytes: u64,
    map_epoch_id: u64,
    map_revision: u64,
}

#[cfg(feature = "nano-agent")]
#[derive(Clone, Debug)]
struct ParsedNanoWarmSelectionV1 {
    dataset_directory_name: String,
    content: NanoWarmSelectionContentIdentity,
    map_epoch_id: RecordedMapEpochId,
    map_revision: u64,
}

/// Typed warm-start load result. There is intentionally no localized variant.
#[derive(Debug)]
pub enum NanoMapWarmStartLoad {
    Disabled,
    DatasetReplayRequired(Box<NanoDatasetReplayRequired>),
}

/// Loaded, unbound occupancy plus the exact configured replay location.
///
/// The persisted snapshot can be inspected but not extracted. The only
/// consuming transition is exact replay verification.
#[derive(Debug)]
pub struct NanoDatasetReplayRequired {
    persisted: PersistedOccupancyMap,
    occupancy_snapshot_path: PathBuf,
    slam_dataset_directory_path: PathBuf,
    state_root: PathBuf,
    state_root_identity: FilesystemObjectIdentity,
    dataset_directory_identity: FilesystemObjectIdentity,
    #[cfg(feature = "nano-agent")]
    selection_root: PathBuf,
    #[cfg(feature = "nano-agent")]
    selection_root_identity: FilesystemObjectIdentity,
    #[cfg(feature = "nano-agent")]
    selection_root_file: File,
    #[cfg(feature = "nano-agent")]
    selection_file: File,
    #[cfg(feature = "nano-agent")]
    selection_identity: FilesystemObjectIdentity,
    #[cfg(feature = "nano-agent")]
    dataset_directory_file: File,
    #[cfg(feature = "nano-agent")]
    manifest_file: File,
    #[cfg(feature = "nano-agent")]
    manifest_identity: FilesystemObjectIdentity,
    #[cfg(feature = "nano-agent")]
    occupancy_file: File,
    #[cfg(feature = "nano-agent")]
    occupancy_identity: FilesystemObjectIdentity,
    #[cfg(feature = "nano-agent")]
    content_identity: NanoWarmSelectionContentIdentity,
    #[cfg(feature = "nano-agent")]
    selected_map_epoch_id: RecordedMapEpochId,
}

impl NanoDatasetReplayRequired {
    pub fn occupancy_snapshot_path(&self) -> &Path {
        &self.occupancy_snapshot_path
    }

    pub fn slam_dataset_directory_path(&self) -> &Path {
        &self.slam_dataset_directory_path
    }

    pub const fn dataset_content_binding_status(&self) -> NanoDatasetContentBindingStatus {
        #[cfg(feature = "nano-agent")]
        {
            self.content_identity.binding_status()
        }
        #[cfg(not(feature = "nano-agent"))]
        {
            NanoDatasetContentBindingStatus::MissingImmutableContentIdentity
        }
    }

    pub fn persisted_snapshot(&self) -> &OccupancyGridSnapshot {
        self.persisted.snapshot()
    }

    #[cfg(feature = "nano-agent")]
    pub const fn selected_map_epoch_id(&self) -> RecordedMapEpochId {
        self.selected_map_epoch_id
    }

    #[cfg(feature = "nano-agent")]
    pub fn selected_map_revision(&self) -> u64 {
        self.persisted.snapshot().revision()
    }

    /// Rewind the exact retained manifest descriptor and bind every byte
    /// consumed by `DatasetReader` to the atomic selection digest.
    #[cfg(feature = "nano-agent")]
    pub(crate) fn selected_manifest_reader(
        &mut self,
    ) -> Result<SelectedManifestReader<'_>, NanoWarmSelectionError> {
        self.require_selected_handles_current()?;
        self.manifest_file
            .seek(SeekFrom::Start(0))
            .map_err(|source| NanoWarmSelectionError::Io {
                operation: "rewind retained selected manifest",
                path: self.slam_dataset_directory_path.join(DATASET_MANIFEST_FILE),
                source,
            })?;
        Ok(SelectedManifestReader {
            reader: DigestingReader::new(&mut self.manifest_file),
            expected_digest: self.content_identity.dataset_manifest_sha256,
            expected_bytes: self.content_identity.dataset_manifest_bytes,
            path: self.slam_dataset_directory_path.join(DATASET_MANIFEST_FILE),
        })
    }

    /// Require the canonical final map event derived from the selected
    /// dataset's bounded navigation journal to match the atomic selection.
    ///
    /// This is independent restart evidence for the wire-stable epoch and
    /// revision. It does not prove a current live-camera pose.
    #[cfg(feature = "nano-agent")]
    pub fn verify_selected_dataset_map_identity(
        &self,
        observed_map_epoch_id: RecordedMapEpochId,
        observed_map_revision: u64,
    ) -> Result<(), NanoWarmSelectionError> {
        if observed_map_epoch_id != self.selected_map_epoch_id {
            return Err(NanoWarmSelectionError::MapEpochMismatch {
                selected: self.selected_map_epoch_id.as_u64(),
                observed: observed_map_epoch_id.as_u64(),
            });
        }
        let selected_map_revision = self.selected_map_revision();
        if observed_map_revision != selected_map_revision {
            return Err(NanoWarmSelectionError::MapRevisionMismatch {
                selected: selected_map_revision,
                observed: observed_map_revision,
            });
        }
        Ok(())
    }

    /// Revalidate the retained directory identity, then require exact equality
    /// with final sparse/occupancy replay evidence. Success is replay matching,
    /// not current-camera relocalization.
    pub fn verify_exact_replay(
        mut self,
        replay: ReplayOccupancyEvidence,
    ) -> Result<NanoReplayMatchedWarmStart, NanoWarmStartReplayBindError> {
        require_unchanged_directory(
            NanoMapPathRole::StateRoot,
            &self.state_root,
            self.state_root_identity,
        )
        .map_err(NanoWarmStartReplayBindError::Path)?;
        require_unchanged_directory(
            NanoMapPathRole::WarmSlamDatasetDirectory,
            &self.slam_dataset_directory_path,
            self.dataset_directory_identity,
        )
        .map_err(NanoWarmStartReplayBindError::Path)?;
        #[cfg(feature = "nano-agent")]
        {
            self.require_selected_handles_current()
                .map_err(NanoWarmStartReplayBindError::Selection)?;
            verify_selected_content_from_handles(
                &mut self.manifest_file,
                &mut self.occupancy_file,
                &self.slam_dataset_directory_path,
                &self.occupancy_snapshot_path,
                self.content_identity,
            )
            .map_err(NanoWarmStartReplayBindError::Selection)?;
        }
        let replay_matched = self
            .persisted
            .verify_replay_and_bind(replay)
            .map_err(NanoWarmStartReplayBindError::ExactReplay)?;
        Ok(NanoReplayMatchedWarmStart {
            replay_matched,
            occupancy_snapshot_path: self.occupancy_snapshot_path,
            slam_dataset_directory_path: self.slam_dataset_directory_path,
            #[cfg(feature = "nano-agent")]
            content_identity: self.content_identity,
        })
    }

    #[cfg(feature = "nano-agent")]
    fn require_selected_handles_current(&self) -> Result<(), NanoWarmSelectionError> {
        require_unchanged_directory(
            NanoMapPathRole::WarmSlamDatasetDirectory,
            &self.selection_root,
            self.selection_root_identity,
        )
        .map_err(NanoWarmSelectionError::Path)?;
        require_directory_identity_at(
            &self.selection_root_file,
            NanoMapPathRole::WarmSlamDatasetDirectory,
            &self.selection_root,
            self.selection_root_identity,
        )
        .map_err(NanoWarmSelectionError::Path)?;
        require_directory_identity_at(
            &self.dataset_directory_file,
            NanoMapPathRole::WarmSlamDatasetDirectory,
            &self.slam_dataset_directory_path,
            self.dataset_directory_identity,
        )
        .map_err(NanoWarmSelectionError::Path)?;
        require_open_regular_file_identity(
            &self.selection_file,
            &self.selection_root.join(NANO_WARM_SELECTION_FILE),
            self.selection_identity,
        )?;
        require_regular_file_identity_at(
            &self.selection_root_file,
            OsStr::new(NANO_WARM_SELECTION_FILE),
            &self.selection_root.join(NANO_WARM_SELECTION_FILE),
            self.selection_identity,
        )?;
        require_open_regular_file_identity(
            &self.manifest_file,
            &self.slam_dataset_directory_path.join(DATASET_MANIFEST_FILE),
            self.manifest_identity,
        )?;
        require_open_regular_file_identity(
            &self.occupancy_file,
            &self.occupancy_snapshot_path,
            self.occupancy_identity,
        )?;
        require_unchanged_directory(
            NanoMapPathRole::WarmSlamDatasetDirectory,
            &self.slam_dataset_directory_path,
            self.dataset_directory_identity,
        )
        .map_err(NanoWarmSelectionError::Path)?;
        require_regular_file_identity_at(
            &self.dataset_directory_file,
            OsStr::new(DATASET_MANIFEST_FILE),
            &self.slam_dataset_directory_path.join(DATASET_MANIFEST_FILE),
            self.manifest_identity,
        )?;
        require_regular_file_identity_at(
            &self.dataset_directory_file,
            OsStr::new(NANO_WARM_OCCUPANCY_FILE),
            &self.occupancy_snapshot_path,
            self.occupancy_identity,
        )
    }
}

/// Occupancy proven equal to final replay output. This type deliberately does
/// not carry or expose a live localization state.
#[derive(Debug)]
pub struct NanoReplayMatchedWarmStart {
    replay_matched: ReplayMatchedOccupancyMap,
    occupancy_snapshot_path: PathBuf,
    slam_dataset_directory_path: PathBuf,
    #[cfg(feature = "nano-agent")]
    content_identity: NanoWarmSelectionContentIdentity,
}

impl NanoReplayMatchedWarmStart {
    pub fn replay_matched_map(&self) -> &ReplayMatchedOccupancyMap {
        &self.replay_matched
    }

    pub fn occupancy_snapshot_path(&self) -> &Path {
        &self.occupancy_snapshot_path
    }

    pub fn slam_dataset_directory_path(&self) -> &Path {
        &self.slam_dataset_directory_path
    }

    pub const fn dataset_content_binding_status(&self) -> NanoDatasetContentBindingStatus {
        #[cfg(feature = "nano-agent")]
        {
            self.content_identity.binding_status()
        }
        #[cfg(not(feature = "nano-agent"))]
        {
            NanoDatasetContentBindingStatus::MissingImmutableContentIdentity
        }
    }

    pub fn into_replay_matched_map(self) -> ReplayMatchedOccupancyMap {
        self.replay_matched
    }
}

#[derive(Debug)]
pub enum NanoMapWarmStartLoadError {
    Path(NanoMapPersistencePathError),
    #[cfg(feature = "nano-agent")]
    Selection(NanoWarmSelectionError),
    Occupancy(OccupancyMapLoadError),
}

impl fmt::Display for NanoMapWarmStartLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Path(source) => write!(formatter, "warm-start path is not admitted: {source}"),
            #[cfg(feature = "nano-agent")]
            Self::Selection(source) => {
                write!(formatter, "warm-start selection is not admitted: {source}")
            }
            Self::Occupancy(source) => write!(formatter, "cannot load warm occupancy: {source}"),
        }
    }
}

impl std::error::Error for NanoMapWarmStartLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Path(source) => Some(source),
            #[cfg(feature = "nano-agent")]
            Self::Selection(source) => Some(source),
            Self::Occupancy(source) => Some(source),
        }
    }
}

#[derive(Debug)]
pub enum NanoWarmStartReplayBindError {
    Path(NanoMapPersistencePathError),
    #[cfg(feature = "nano-agent")]
    Selection(NanoWarmSelectionError),
    ExactReplay(OccupancyReplayBindError),
}

impl fmt::Display for NanoWarmStartReplayBindError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Path(source) => write!(formatter, "replay dataset path changed: {source}"),
            #[cfg(feature = "nano-agent")]
            Self::Selection(source) => write!(
                formatter,
                "selected replay dataset content changed during replay: {source}"
            ),
            Self::ExactReplay(source) => write!(formatter, "occupancy replay mismatch: {source}"),
        }
    }
}

impl std::error::Error for NanoWarmStartReplayBindError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Path(source) => Some(source),
            #[cfg(feature = "nano-agent")]
            Self::Selection(source) => Some(source),
            Self::ExactReplay(source) => Some(source),
        }
    }
}

/// Sole production owner for the bounded live snapshot and map writer.
pub struct NanoMapPersistenceOwner {
    state_root: PathBuf,
    state_root_identity: FilesystemObjectIdentity,
    save_snapshot_path: PathBuf,
    save_parent_path: PathBuf,
    save_parent_identity: FilesystemObjectIdentity,
    warm_start: AdmittedNanoWarmStart,
    limits: OccupancyMapLimits,
    latest: Option<RetainedNanoMapSnapshot>,
}

impl fmt::Debug for NanoMapPersistenceOwner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NanoMapPersistenceOwner")
            .field("state_root", &self.state_root)
            .field("save_snapshot_path", &self.save_snapshot_path)
            .field("warm_start", &self.warm_start)
            .field("limits", &self.limits)
            .field("latest", &self.latest.as_ref().map(|value| value.identity))
            .finish()
    }
}

impl NanoMapPersistenceOwner {
    /// Admit the exact policy paths beneath the canonical service state root.
    ///
    /// The state root and save parent must already exist. The destination file
    /// may be absent, but an existing destination must be a canonical regular
    /// file and never a symlink.
    pub fn try_new(
        roots: &NanoBootstrapRoots,
        config: &NanoMapPersistenceConfig,
        limits: OccupancyMapLimits,
    ) -> Result<Self, NanoMapPersistencePathError> {
        Self::try_new_with_save_parent_policy(
            roots,
            config,
            limits,
            SaveParentAdmission::RequireExisting,
        )
    }

    /// Explicitly permit creation of one missing direct child of the admitted
    /// state root as the save parent.
    ///
    /// This never creates multiple components. The new directory is requested
    /// with mode `0700`, an `AlreadyExists` race is re-admitted rather than
    /// trusted, and the state-root directory is synchronized before success.
    pub fn try_new_with_direct_save_parent_creation(
        roots: &NanoBootstrapRoots,
        config: &NanoMapPersistenceConfig,
        limits: OccupancyMapLimits,
    ) -> Result<Self, NanoMapPersistencePathError> {
        Self::try_new_with_save_parent_policy(
            roots,
            config,
            limits,
            SaveParentAdmission::CreateMissingDirectChild,
        )
    }

    fn try_new_with_save_parent_policy(
        roots: &NanoBootstrapRoots,
        config: &NanoMapPersistenceConfig,
        limits: OccupancyMapLimits,
        save_parent_admission: SaveParentAdmission,
    ) -> Result<Self, NanoMapPersistencePathError> {
        let state_root = roots.state_root().to_path_buf();
        let state_root_identity =
            require_canonical_directory(NanoMapPathRole::StateRoot, &state_root)?;

        let save_snapshot_path = config.save_snapshot_path().as_path().to_path_buf();
        require_strict_descendant(
            NanoMapPathRole::SaveSnapshot,
            &state_root,
            &save_snapshot_path,
        )?;
        let save_parent_path = save_snapshot_path
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| NanoMapPersistencePathError::MissingParent {
                role: NanoMapPathRole::SaveSnapshot,
                path: save_snapshot_path.clone(),
            })?;
        let save_parent_identity = admit_save_parent(
            &state_root,
            state_root_identity,
            &save_parent_path,
            save_parent_admission,
        )?;
        require_optional_canonical_regular_file(
            NanoMapPathRole::SaveSnapshot,
            &save_snapshot_path,
        )?;

        let warm_start = match config.warm_start() {
            NanoMapWarmStart::None => AdmittedNanoWarmStart::None,
            NanoMapWarmStart::DatasetReplay {
                occupancy_snapshot_path,
                slam_dataset_directory_path,
            } => {
                let occupancy_snapshot_path = occupancy_snapshot_path.as_path().to_path_buf();
                let slam_dataset_directory_path =
                    slam_dataset_directory_path.as_path().to_path_buf();
                require_strict_descendant(
                    NanoMapPathRole::WarmOccupancySnapshot,
                    &state_root,
                    &occupancy_snapshot_path,
                )?;
                require_strict_descendant(
                    NanoMapPathRole::WarmSlamDatasetDirectory,
                    &state_root,
                    &slam_dataset_directory_path,
                )?;
                AdmittedNanoWarmStart::DatasetReplay {
                    occupancy_snapshot_path,
                    slam_dataset_directory_path,
                }
            }
        };

        Ok(Self {
            state_root,
            state_root_identity,
            save_snapshot_path,
            save_parent_path,
            save_parent_identity,
            warm_start,
            limits,
            latest: None,
        })
    }

    pub fn save_snapshot_path(&self) -> &Path {
        &self.save_snapshot_path
    }

    pub fn latest_identity(&self) -> Option<NanoMapSnapshotIdentity> {
        self.latest.as_ref().map(|retained| retained.identity)
    }

    pub const fn retained_snapshot_capacity(&self) -> usize {
        1
    }

    /// Move one live snapshot into the sole slot without a grid copy.
    pub fn retain_latest(
        &mut self,
        binding: CurrentMapEpochBinding,
        snapshot: OccupancyGridSnapshot,
    ) -> Result<NanoMapRetentionOutcome, NanoMapSnapshotRetentionError> {
        let actual_cells = snapshot.class_ids().len();
        if actual_cells > self.limits.maximum_cells() {
            return Err(NanoMapSnapshotRetentionError::TooManyCells {
                actual_cells,
                maximum_cells: self.limits.maximum_cells(),
            });
        }
        let snapshot_map_instance_id = snapshot
            .map_instance_id()
            .ok_or(NanoMapSnapshotRetentionError::UnboundSnapshot)?;
        if snapshot_map_instance_id != binding.map_instance_id() {
            return Err(NanoMapSnapshotRetentionError::MapInstanceMismatch {
                map_epoch_id: binding.map_epoch_id(),
                bound: binding.map_instance_id(),
                snapshot: snapshot_map_instance_id,
            });
        }
        let offered = NanoMapSnapshotIdentity {
            map_epoch_id: binding.map_epoch_id(),
            map_instance_id: binding.map_instance_id(),
            revision: snapshot.revision(),
        };

        if let Some(retained) = self.latest.as_ref() {
            let retained_epoch = retained.identity.map_epoch_id.as_u64();
            let offered_epoch = offered.map_epoch_id.as_u64();
            if offered_epoch < retained_epoch {
                return Err(NanoMapSnapshotRetentionError::StaleEpoch {
                    retained: retained.identity.map_epoch_id,
                    offered: offered.map_epoch_id,
                });
            }
            if offered_epoch == retained_epoch {
                if offered.map_instance_id != retained.identity.map_instance_id {
                    return Err(NanoMapSnapshotRetentionError::EpochMapInstanceChanged {
                        map_epoch_id: offered.map_epoch_id,
                        retained: retained.identity.map_instance_id,
                        offered: offered.map_instance_id,
                    });
                }
                if offered.revision <= retained.identity.revision {
                    return Err(NanoMapSnapshotRetentionError::StaleRevision {
                        map_epoch_id: offered.map_epoch_id,
                        retained: retained.identity.revision,
                        offered: offered.revision,
                    });
                }
            }
        }

        let replaced = self.latest.replace(RetainedNanoMapSnapshot {
            identity: offered,
            snapshot,
        });
        Ok(NanoMapRetentionOutcome {
            retained: offered,
            replaced: replaced.map(|value| value.identity),
        })
    }

    /// Durably save the slot's current value. No receipt exists on any error.
    pub fn save_latest(&mut self) -> Result<NanoMapSaveReceipt, NanoMapSaveError> {
        self.save_with_selection(None)
    }

    /// Durably save only if the slot still contains the exact selected epoch
    /// and revision.
    pub fn save_selected(
        &mut self,
        selected: NanoMapSnapshotIdentity,
    ) -> Result<NanoMapSaveReceipt, NanoMapSaveError> {
        self.save_with_selection(Some(selected))
    }

    /// Quota-bound form required by the production Nano save-map path.
    pub fn save_latest_with_quota(
        &mut self,
        quota: &mut NanoStateQuotaOwner,
    ) -> Result<NanoMapSaveReceipt, NanoMapSaveError> {
        self.save_with_quota(None, quota)
    }

    /// Exact-selection quota-bound form required after a displayed map
    /// identity was captured.
    pub fn save_selected_with_quota(
        &mut self,
        selected: NanoMapSnapshotIdentity,
        quota: &mut NanoStateQuotaOwner,
    ) -> Result<NanoMapSaveReceipt, NanoMapSaveError> {
        self.save_with_quota(Some(selected), quota)
    }

    fn save_with_selection(
        &mut self,
        selected: Option<NanoMapSnapshotIdentity>,
    ) -> Result<NanoMapSaveReceipt, NanoMapSaveError> {
        self.save_atomically(selected)
    }

    fn save_with_quota(
        &mut self,
        selected: Option<NanoMapSnapshotIdentity>,
        quota: &mut NanoStateQuotaOwner,
    ) -> Result<NanoMapSaveReceipt, NanoMapSaveError> {
        let retained = self.retained_for_selection(selected)?;
        let cells = retained.snapshot.class_ids().len();
        let encoded_bytes = occupancy_map_encoded_len(&retained.snapshot)
            .map_err(OccupancyMapSaveError::from)
            .map_err(NanoMapSaveError::Persistence)?;
        let encoded_bytes = u64::try_from(encoded_bytes).map_err(|_| {
            NanoMapSaveError::Persistence(OccupancyMapSaveError::Encode(
                OccupancyMapEncodeError::EncodedLengthOverflow { cells },
            ))
        })?;
        let reservation = quota
            .reserve_map_replacement(&self.save_snapshot_path, encoded_bytes)
            .map_err(NanoMapSaveError::QuotaReservation)?;
        #[cfg(feature = "nano-agent")]
        let save_result =
            self.save_with_parent_descriptor(selected, reservation.publication_parent());
        #[cfg(not(feature = "nano-agent"))]
        let save_result = self.save_atomically(selected);
        let mut receipt = match save_result {
            Ok(receipt) => receipt,
            Err(NanoMapSaveError::Persistence(source))
                if persistence_error_was_published(&source) =>
            {
                let quota_verification = reservation
                    .verify_committed()
                    .map(Box::new)
                    .map_err(Box::new);
                return Err(NanoMapSaveError::PublishedButPersistenceFailed {
                    source,
                    quota_verification,
                });
            }
            Err(source) => return Err(source),
        };
        let quota_verification = reservation.verify_committed().map_err(|source| {
            NanoMapSaveError::PublishedButQuotaVerificationFailed {
                receipt: receipt.clone(),
                source: Box::new(source),
            }
        })?;
        receipt.quota_verification = Some(Box::new(quota_verification));
        Ok(receipt)
    }

    fn retained_for_selection(
        &self,
        selected: Option<NanoMapSnapshotIdentity>,
    ) -> Result<&RetainedNanoMapSnapshot, NanoMapSaveError> {
        let retained = self.latest.as_ref().ok_or(NanoMapSaveError::NoSnapshot)?;
        if let Some(requested) = selected
            && requested != retained.identity
        {
            return Err(NanoMapSaveError::StaleSelection {
                requested,
                retained: retained.identity,
            });
        }
        Ok(retained)
    }

    fn save_atomically(
        &mut self,
        selected: Option<NanoMapSnapshotIdentity>,
    ) -> Result<NanoMapSaveReceipt, NanoMapSaveError> {
        #[cfg(feature = "nano-agent")]
        {
            self.save_with_admitted_parent(selected)
        }
        #[cfg(not(feature = "nano-agent"))]
        {
            self.save_with(selected, |path, snapshot| {
                save_occupancy_map_atomic(path, snapshot)
            })
        }
    }

    #[cfg(feature = "nano-agent")]
    fn save_with_admitted_parent(
        &mut self,
        selected: Option<NanoMapSnapshotIdentity>,
    ) -> Result<NanoMapSaveReceipt, NanoMapSaveError> {
        require_unchanged_directory(
            NanoMapPathRole::StateRoot,
            &self.state_root,
            self.state_root_identity,
        )
        .map_err(NanoMapSaveError::Path)?;
        let save_parent = open_unchanged_directory_file(
            NanoMapPathRole::SaveParent,
            &self.save_parent_path,
            self.save_parent_identity,
        )
        .map_err(NanoMapSaveError::Path)?;
        self.save_with_parent_descriptor(selected, &save_parent)
    }

    #[cfg(feature = "nano-agent")]
    fn save_with_parent_descriptor(
        &mut self,
        selected: Option<NanoMapSnapshotIdentity>,
        save_parent: impl AsFd,
    ) -> Result<NanoMapSaveReceipt, NanoMapSaveError> {
        let retained = self.retained_for_selection(selected)?;
        require_directory_identity_at(
            &save_parent,
            NanoMapPathRole::SaveParent,
            &self.save_parent_path,
            self.save_parent_identity,
        )
        .map_err(NanoMapSaveError::Path)?;
        require_optional_regular_file_at(
            &save_parent,
            self.save_snapshot_path
                .file_name()
                .expect("admitted save snapshot has one file name"),
            NanoMapPathRole::SaveSnapshot,
            &self.save_snapshot_path,
        )
        .map_err(NanoMapSaveError::Path)?;
        save_occupancy_map_atomic_at(
            &save_parent,
            self.save_snapshot_path
                .file_name()
                .expect("admitted save snapshot has one file name"),
            &self.save_snapshot_path,
            &retained.snapshot,
        )
        .map_err(NanoMapSaveError::Persistence)?;
        Ok(NanoMapSaveReceipt {
            identity: retained.identity,
            destination: self.save_snapshot_path.clone(),
            quota_verification: None,
        })
    }

    #[cfg(any(test, not(feature = "nano-agent")))]
    fn save_with<F>(
        &mut self,
        selected: Option<NanoMapSnapshotIdentity>,
        save: F,
    ) -> Result<NanoMapSaveReceipt, NanoMapSaveError>
    where
        F: FnOnce(&Path, &OccupancyGridSnapshot) -> Result<(), OccupancyMapSaveError>,
    {
        let retained = self.retained_for_selection(selected)?;
        require_unchanged_directory(
            NanoMapPathRole::StateRoot,
            &self.state_root,
            self.state_root_identity,
        )
        .map_err(NanoMapSaveError::Path)?;
        require_unchanged_directory(
            NanoMapPathRole::SaveParent,
            &self.save_parent_path,
            self.save_parent_identity,
        )
        .map_err(NanoMapSaveError::Path)?;
        require_optional_canonical_regular_file(
            NanoMapPathRole::SaveSnapshot,
            &self.save_snapshot_path,
        )
        .map_err(NanoMapSaveError::Path)?;

        save(&self.save_snapshot_path, &retained.snapshot)
            .map_err(NanoMapSaveError::Persistence)?;
        Ok(NanoMapSaveReceipt {
            identity: retained.identity,
            destination: self.save_snapshot_path.clone(),
            quota_verification: None,
        })
    }

    /// Execute a claimed `save_map` and emit `Completed` only after durable
    /// save success. Save failures are converted to truthful final rejections.
    pub fn respond_to_claimed_save_map(
        &mut self,
        claimed: AgentControlClaimedRequest,
    ) -> Result<NanoMapSaveReceipt, NanoMapSaveCommandError> {
        finish_claimed_save(claimed, || self.save_latest()).map_err(Into::into)
    }

    /// Exact-selection form used when the outer runtime captured a displayed
    /// map identity before dispatch.
    pub fn respond_to_claimed_selected_save_map(
        &mut self,
        claimed: AgentControlClaimedRequest,
        selected: NanoMapSnapshotIdentity,
    ) -> Result<NanoMapSaveReceipt, NanoMapSaveCommandError> {
        finish_claimed_save(claimed, || self.save_selected(selected)).map_err(Into::into)
    }

    /// Production command form: completion is emitted only after both durable
    /// atomic publication and exact post-write map quota verification.
    pub fn respond_to_claimed_save_map_with_quota(
        &mut self,
        claimed: AgentControlClaimedRequest,
        quota: &mut NanoStateQuotaOwner,
    ) -> Result<NanoMapSaveReceipt, NanoMapSaveCommandError> {
        finish_claimed_save(claimed, || self.save_latest_with_quota(quota)).map_err(Into::into)
    }

    /// Exact-selection production command form with launch-bound map quota
    /// admission and post-write verification.
    pub fn respond_to_claimed_selected_save_map_with_quota(
        &mut self,
        claimed: AgentControlClaimedRequest,
        selected: NanoMapSnapshotIdentity,
        quota: &mut NanoStateQuotaOwner,
    ) -> Result<NanoMapSaveReceipt, NanoMapSaveCommandError> {
        finish_claimed_save(claimed, || self.save_selected_with_quota(selected, quota))
            .map_err(Into::into)
    }

    /// Whether `save_map` must terminate capture and bind the drained final
    /// map to the finalized session dataset before it may complete.
    #[cfg(feature = "nano-agent")]
    pub const fn requires_quiescent_warm_checkpoint(&self) -> bool {
        true
    }

    /// Publish one restart-safe terminal checkpoint.
    ///
    /// The caller must first stop capture, drain inference/dense/navigation,
    /// finalize the exact session dataset manifest, and retain the final
    /// occupancy snapshot. This method then:
    ///
    /// 1. writes and quota-verifies the configured staging pathname;
    /// 2. relocates that exact inode, without replacement or a second encode,
    ///    into the immutable session-local occupancy pathname;
    /// 3. hashes that artifact and the finalized dataset manifest; and
    /// 4. atomically selects the pair by a synchronized rename.
    ///
    /// Every error before the selection rename leaves the previous selection
    /// untouched. An error after rename but before directory synchronization
    /// is explicitly reported as publication uncertainty and never receives a
    /// completion response. A successful receipt proves replay inputs, not
    /// current-camera localization.
    #[cfg(feature = "nano-agent")]
    pub fn publish_quiescent_warm_checkpoint_with_quota(
        &mut self,
        finalized_dataset_directory: &Path,
        finalized_map_identity: Option<NanoFinalizedJournalMapIdentity>,
        quota: &mut NanoStateQuotaOwner,
    ) -> Result<NanoWarmCheckpointReceipt, NanoWarmCheckpointError> {
        let selected = self
            .latest_identity()
            .ok_or(NanoWarmCheckpointError::SaveMap(
                NanoMapSaveError::NoSnapshot,
            ))?;
        require_finalized_journal_map_matches_retained(finalized_map_identity, selected)?;
        let selection_root = match &self.warm_start {
            AdmittedNanoWarmStart::None => finalized_dataset_directory
                .parent()
                .map(Path::to_path_buf)
                .ok_or_else(|| NanoWarmCheckpointError::DatasetHasNoSelectionParent {
                    dataset_directory: finalized_dataset_directory.to_path_buf(),
                })?,
            AdmittedNanoWarmStart::DatasetReplay {
                slam_dataset_directory_path,
                ..
            } => slam_dataset_directory_path.clone(),
        };
        require_unchanged_directory(
            NanoMapPathRole::StateRoot,
            &self.state_root,
            self.state_root_identity,
        )
        .map_err(NanoWarmCheckpointError::DatasetPath)?;
        require_strict_descendant(
            NanoMapPathRole::WarmSlamDatasetDirectory,
            &self.state_root,
            &selection_root,
        )
        .map_err(NanoWarmCheckpointError::DatasetPath)?;
        let selection_root_identity =
            require_canonical_directory(NanoMapPathRole::WarmSlamDatasetDirectory, &selection_root)
                .map_err(NanoWarmCheckpointError::DatasetPath)?;
        require_direct_child(&selection_root, finalized_dataset_directory).map_err(|()| {
            NanoWarmCheckpointError::DatasetIsNotDirectSelectionChild {
                selection_root: selection_root.clone(),
                dataset_directory: finalized_dataset_directory.to_path_buf(),
            }
        })?;
        let dataset_identity = require_canonical_directory(
            NanoMapPathRole::WarmSlamDatasetDirectory,
            finalized_dataset_directory,
        )
        .map_err(NanoWarmCheckpointError::DatasetPath)?;
        let manifest_path = finalized_dataset_directory.join(DATASET_MANIFEST_FILE);
        match require_canonical_regular_file(
            NanoMapPathRole::WarmSlamDatasetDirectory,
            &manifest_path,
        ) {
            Ok(_) => {}
            Err(NanoMapPersistencePathError::Missing { .. }) => {
                return Err(NanoWarmCheckpointError::DatasetManifestMissing {
                    path: manifest_path,
                });
            }
            Err(source) => return Err(NanoWarmCheckpointError::DatasetPath(source)),
        }

        let occupancy_path = finalized_dataset_directory.join(NANO_WARM_OCCUPANCY_FILE);
        match fs::symlink_metadata(&occupancy_path) {
            Ok(_) => {
                return Err(NanoWarmCheckpointError::OccupancyAlreadyExists {
                    path: occupancy_path,
                });
            }
            Err(source) if source.kind() == io::ErrorKind::NotFound => {}
            Err(source) => {
                return Err(NanoWarmCheckpointError::Selection(
                    NanoWarmSelectionError::Io {
                        operation: "inspect checkpoint occupancy destination",
                        path: occupancy_path,
                        source,
                    },
                ));
            }
        }
        let staged_map = self
            .save_selected_with_quota(selected, quota)
            .map_err(NanoWarmCheckpointError::SaveMap)?;
        let maximum_occupancy_bytes = staged_map
            .quota_verification()
            .map(NanoStateQuotaWriteReceipt::exact_bytes)
            .ok_or(NanoWarmCheckpointError::MissingQuotaVerification)?;
        relocate_staged_checkpoint_occupancy(
            staged_map.destination(),
            &self.save_parent_path,
            self.save_parent_identity,
            &occupancy_path,
            finalized_dataset_directory,
            dataset_identity,
        )
        .map_err(NanoWarmCheckpointError::Selection)?;

        let (dataset_manifest_sha256, dataset_manifest_bytes) =
            hash_bounded_regular_file(&manifest_path, MAX_NANO_DATASET_MANIFEST_BYTES)
                .map_err(NanoWarmCheckpointError::Selection)?;
        let (occupancy_sha256, occupancy_bytes) =
            hash_bounded_regular_file(&occupancy_path, maximum_occupancy_bytes)
                .map_err(NanoWarmCheckpointError::Selection)?;
        if occupancy_bytes != maximum_occupancy_bytes {
            return Err(NanoWarmCheckpointError::Selection(
                NanoWarmSelectionError::LengthMismatch {
                    path: occupancy_path,
                    expected: maximum_occupancy_bytes,
                    observed: occupancy_bytes,
                },
            ));
        }
        require_unchanged_directory(
            NanoMapPathRole::WarmSlamDatasetDirectory,
            finalized_dataset_directory,
            dataset_identity,
        )
        .map_err(NanoWarmCheckpointError::DatasetPath)?;
        require_unchanged_directory(
            NanoMapPathRole::WarmSlamDatasetDirectory,
            &selection_root,
            selection_root_identity,
        )
        .map_err(NanoWarmCheckpointError::DatasetPath)?;

        let dataset_directory_name = finalized_dataset_directory
            .file_name()
            .and_then(OsStr::to_str)
            .ok_or(NanoWarmCheckpointError::Selection(
                NanoWarmSelectionError::InvalidDatasetDirectoryName,
            ))?
            .to_owned();
        let content = NanoWarmSelectionContentIdentity {
            dataset_manifest_sha256,
            occupancy_sha256,
            dataset_manifest_bytes,
            occupancy_bytes,
        };
        let selection = ParsedNanoWarmSelectionV1 {
            dataset_directory_name,
            content,
            map_epoch_id: selected.map_epoch_id,
            map_revision: selected.revision,
        };
        let selection_path =
            publish_warm_selection(&selection_root, selection_root_identity, &selection)
                .map_err(NanoWarmCheckpointError::Selection)?;
        Ok(NanoWarmCheckpointReceipt {
            staged_map,
            occupancy_snapshot_path: occupancy_path,
            dataset_directory: finalized_dataset_directory.to_path_buf(),
            selection_path,
            binding: content.binding_status(),
        })
    }

    /// Claimed-command form. The accepted completion is sent only after the
    /// dataset/map selection rename and its parent-directory synchronization.
    #[cfg(feature = "nano-agent")]
    pub fn respond_to_claimed_quiescent_warm_checkpoint_with_quota(
        &mut self,
        claimed: AgentControlClaimedRequest,
        finalized_dataset_directory: &Path,
        finalized_map_identity: Option<NanoFinalizedJournalMapIdentity>,
        quota: &mut NanoStateQuotaOwner,
        require_wire_delivery: bool,
    ) -> Result<NanoWarmCheckpointReceipt, NanoWarmCheckpointCommandError> {
        let actual = claimed.request().command().kind();
        if actual != AgentControlCommandKindV1::SaveMap {
            let response = if require_wire_delivery {
                claimed
                    .reject_after_wire_delivery(AgentControlRejectionCodeV1::InternalFault, false)
            } else {
                claimed.reject(AgentControlRejectionCodeV1::InternalFault, false)
            };
            return match response {
                Ok(()) => Err(NanoWarmCheckpointCommandError::WrongCommandRejected { actual }),
                Err(response) => Err(NanoWarmCheckpointCommandError::WrongCommandResponseFailed {
                    actual,
                    response,
                }),
            };
        }
        match self.publish_quiescent_warm_checkpoint_with_quota(
            finalized_dataset_directory,
            finalized_map_identity,
            quota,
        ) {
            Ok(receipt) => match if require_wire_delivery {
                claimed.respond_completed_after_wire_delivery()
            } else {
                claimed.respond_completed()
            } {
                Ok(()) => Ok(receipt),
                Err(response) => Err(
                    NanoWarmCheckpointCommandError::CheckpointedButCompletionResponseFailed {
                        receipt: Box::new(receipt),
                        response,
                    },
                ),
            },
            Err(source) => {
                let (code, retryable) = source.rejection();
                let response = if require_wire_delivery {
                    claimed.reject_after_wire_delivery(code, retryable)
                } else {
                    claimed.reject(code, retryable)
                };
                match response {
                    Ok(()) => Err(NanoWarmCheckpointCommandError::CheckpointRejected { source }),
                    Err(response) => Err(
                        NanoWarmCheckpointCommandError::CheckpointAndRejectionResponseFailed {
                            source,
                            response,
                        },
                    ),
                }
            }
        }
    }

    /// Load the configured warm artifact once.
    ///
    /// Production resolves only the durably selected finalized session. The
    /// configured dataset pathname is the selection root, never a mutable
    /// recording directory and never a symlink. The selected manifest and
    /// occupancy digests are checked before replay and again when exact replay
    /// is bound.
    pub fn load_warm_start(&self) -> Result<NanoMapWarmStartLoad, NanoMapWarmStartLoadError> {
        require_unchanged_directory(
            NanoMapPathRole::StateRoot,
            &self.state_root,
            self.state_root_identity,
        )
        .map_err(NanoMapWarmStartLoadError::Path)?;
        let AdmittedNanoWarmStart::DatasetReplay {
            occupancy_snapshot_path,
            slam_dataset_directory_path,
        } = &self.warm_start
        else {
            return Ok(NanoMapWarmStartLoad::Disabled);
        };

        #[cfg(feature = "nano-agent")]
        {
            let _ = occupancy_snapshot_path;
            self.load_selected_warm_start(slam_dataset_directory_path)
        }

        #[cfg(not(feature = "nano-agent"))]
        {
            let dataset_directory_identity = require_canonical_directory(
                NanoMapPathRole::WarmSlamDatasetDirectory,
                slam_dataset_directory_path,
            )
            .map_err(NanoMapWarmStartLoadError::Path)?;
            let snapshot_identity = require_canonical_regular_file(
                NanoMapPathRole::WarmOccupancySnapshot,
                occupancy_snapshot_path,
            )
            .map_err(NanoMapWarmStartLoadError::Path)?;
            let persisted = load_persisted_occupancy_map(occupancy_snapshot_path, self.limits)
                .map_err(NanoMapWarmStartLoadError::Occupancy)?;
            require_unchanged_regular_file(
                NanoMapPathRole::WarmOccupancySnapshot,
                occupancy_snapshot_path,
                snapshot_identity,
            )
            .map_err(NanoMapWarmStartLoadError::Path)?;

            Ok(NanoMapWarmStartLoad::DatasetReplayRequired(Box::new(
                NanoDatasetReplayRequired {
                    persisted,
                    occupancy_snapshot_path: occupancy_snapshot_path.clone(),
                    slam_dataset_directory_path: slam_dataset_directory_path.clone(),
                    state_root: self.state_root.clone(),
                    state_root_identity: self.state_root_identity,
                    dataset_directory_identity,
                },
            )))
        }
    }

    #[cfg(feature = "nano-agent")]
    fn load_selected_warm_start(
        &self,
        selection_root: &Path,
    ) -> Result<NanoMapWarmStartLoad, NanoMapWarmStartLoadError> {
        let selection_root_identity =
            require_canonical_directory(NanoMapPathRole::WarmSlamDatasetDirectory, selection_root)
                .map_err(NanoMapWarmStartLoadError::Path)?;
        let selection_root_file = open_unchanged_directory_nofollow(
            NanoMapPathRole::WarmSlamDatasetDirectory,
            selection_root,
            selection_root_identity,
        )
        .map_err(NanoMapWarmStartLoadError::Selection)?;
        let selection_path = selection_root.join(NANO_WARM_SELECTION_FILE);
        let (mut selection_file, selection_identity) = open_regular_file_at_nofollow(
            &selection_root_file,
            OsStr::new(NANO_WARM_SELECTION_FILE),
            &selection_path,
        )
        .map_err(NanoMapWarmStartLoadError::Selection)?;
        let selection_bytes = read_bounded_open_file(
            &mut selection_file,
            &selection_path,
            MAX_NANO_WARM_SELECTION_BYTES,
        )
        .map_err(NanoMapWarmStartLoadError::Selection)?;
        let parsed = parse_warm_selection_bytes(selection_root, &selection_bytes)
            .map_err(NanoMapWarmStartLoadError::Selection)?;
        let dataset_directory = selection_root.join(&parsed.dataset_directory_name);
        require_direct_child(selection_root, &dataset_directory).map_err(|()| {
            NanoMapWarmStartLoadError::Selection(
                NanoWarmSelectionError::InvalidDatasetDirectoryName,
            )
        })?;
        let (dataset_directory_file, dataset_directory_identity) = open_directory_at_nofollow(
            &selection_root_file,
            OsStr::new(&parsed.dataset_directory_name),
            &dataset_directory,
        )
        .map_err(NanoMapWarmStartLoadError::Selection)?;
        require_unchanged_directory(
            NanoMapPathRole::WarmSlamDatasetDirectory,
            &dataset_directory,
            dataset_directory_identity,
        )
        .map_err(NanoMapWarmStartLoadError::Path)?;
        let manifest_path = dataset_directory.join(DATASET_MANIFEST_FILE);
        let (mut manifest_file, manifest_identity) = open_regular_file_at_nofollow(
            &dataset_directory_file,
            OsStr::new(DATASET_MANIFEST_FILE),
            &manifest_path,
        )
        .map_err(NanoMapWarmStartLoadError::Selection)?;
        let occupancy_snapshot_path = dataset_directory.join(NANO_WARM_OCCUPANCY_FILE);
        let (mut occupancy_file, occupancy_identity) = open_regular_file_at_nofollow(
            &dataset_directory_file,
            OsStr::new(NANO_WARM_OCCUPANCY_FILE),
            &occupancy_snapshot_path,
        )
        .map_err(NanoMapWarmStartLoadError::Selection)?;

        let (manifest_digest, manifest_bytes) = hash_bounded_open_file(
            &mut manifest_file,
            &manifest_path,
            MAX_NANO_DATASET_MANIFEST_BYTES,
        )
        .map_err(NanoMapWarmStartLoadError::Selection)?;
        verify_selected_digest(
            &manifest_path,
            parsed.content.dataset_manifest_sha256,
            parsed.content.dataset_manifest_bytes,
            manifest_digest,
            manifest_bytes,
        )
        .map_err(NanoMapWarmStartLoadError::Selection)?;

        let occupancy_bytes = occupancy_file
            .metadata()
            .map_err(|source| {
                NanoMapWarmStartLoadError::Selection(NanoWarmSelectionError::Io {
                    operation: "inspect retained selected occupancy",
                    path: occupancy_snapshot_path.clone(),
                    source,
                })
            })?
            .len();
        if occupancy_bytes != parsed.content.occupancy_bytes {
            return Err(NanoMapWarmStartLoadError::Selection(
                NanoWarmSelectionError::LengthMismatch {
                    path: occupancy_snapshot_path,
                    expected: parsed.content.occupancy_bytes,
                    observed: occupancy_bytes,
                },
            ));
        }
        occupancy_file.seek(SeekFrom::Start(0)).map_err(|source| {
            NanoMapWarmStartLoadError::Selection(NanoWarmSelectionError::Io {
                operation: "rewind retained selected occupancy",
                path: occupancy_snapshot_path.clone(),
                source,
            })
        })?;
        let mut occupancy_reader = DigestingReader::new(&mut occupancy_file);
        let persisted = load_persisted_occupancy_map_from_reader(
            &mut occupancy_reader,
            occupancy_bytes,
            &occupancy_snapshot_path,
            self.limits,
        )
        .map_err(NanoMapWarmStartLoadError::Occupancy)?;
        let (occupancy_digest, observed_occupancy_bytes) = occupancy_reader.finish();
        verify_selected_digest(
            &occupancy_snapshot_path,
            parsed.content.occupancy_sha256,
            parsed.content.occupancy_bytes,
            occupancy_digest,
            observed_occupancy_bytes,
        )
        .map_err(NanoMapWarmStartLoadError::Selection)?;
        if persisted.snapshot().revision() != parsed.map_revision {
            return Err(NanoMapWarmStartLoadError::Selection(
                NanoWarmSelectionError::MapRevisionMismatch {
                    selected: parsed.map_revision,
                    observed: persisted.snapshot().revision(),
                },
            ));
        }
        require_regular_file_identity_at(
            &selection_root_file,
            OsStr::new(NANO_WARM_SELECTION_FILE),
            &selection_path,
            selection_identity,
        )
        .map_err(NanoMapWarmStartLoadError::Selection)?;
        require_regular_file_identity_at(
            &dataset_directory_file,
            OsStr::new(DATASET_MANIFEST_FILE),
            &manifest_path,
            manifest_identity,
        )
        .map_err(NanoMapWarmStartLoadError::Selection)?;
        require_regular_file_identity_at(
            &dataset_directory_file,
            OsStr::new(NANO_WARM_OCCUPANCY_FILE),
            &occupancy_snapshot_path,
            occupancy_identity,
        )
        .map_err(NanoMapWarmStartLoadError::Selection)?;

        Ok(NanoMapWarmStartLoad::DatasetReplayRequired(Box::new(
            NanoDatasetReplayRequired {
                persisted,
                occupancy_snapshot_path,
                slam_dataset_directory_path: dataset_directory,
                state_root: self.state_root.clone(),
                state_root_identity: self.state_root_identity,
                dataset_directory_identity,
                selection_root: selection_root.to_path_buf(),
                selection_root_identity,
                selection_root_file,
                selection_file,
                selection_identity,
                dataset_directory_file,
                manifest_file,
                manifest_identity,
                occupancy_file,
                occupancy_identity,
                content_identity: parsed.content,
                selected_map_epoch_id: parsed.map_epoch_id,
            },
        )))
    }
}

#[cfg(feature = "nano-agent")]
fn require_direct_child(parent: &Path, child: &Path) -> Result<(), ()> {
    if child.parent() != Some(parent) {
        return Err(());
    }
    let Some(name) = child.file_name().and_then(OsStr::to_str) else {
        return Err(());
    };
    parse_dataset_directory_name(name).map(|_| ())
}

#[cfg(feature = "nano-agent")]
fn parse_dataset_directory_name(value: &str) -> Result<&str, ()> {
    if value.is_empty()
        || value.len() > MAX_DATASET_DIRECTORY_NAME_BYTES
        || Path::new(value).components().count() != 1
        || !matches!(
            Path::new(value).components().next(),
            Some(Component::Normal(_))
        )
    {
        return Err(());
    }
    Ok(value)
}

#[cfg(feature = "nano-agent")]
fn parse_sha256_hex(
    field: &'static str,
    value: &str,
) -> Result<[u8; SHA256_BYTES], NanoWarmSelectionError> {
    if value.len() != SHA256_HEX_BYTES || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(NanoWarmSelectionError::InvalidSha256 { field });
    }
    let mut digest = [0_u8; SHA256_BYTES];
    for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
        let high =
            decode_hex_nibble(chunk[0]).ok_or(NanoWarmSelectionError::InvalidSha256 { field })?;
        let low =
            decode_hex_nibble(chunk[1]).ok_or(NanoWarmSelectionError::InvalidSha256 { field })?;
        digest[index] = (high << 4) | low;
    }
    Ok(digest)
}

#[cfg(feature = "nano-agent")]
fn decode_hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

#[cfg(feature = "nano-agent")]
fn encode_sha256_hex(value: [u8; SHA256_BYTES]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(SHA256_HEX_BYTES);
    for byte in value {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

#[cfg(feature = "nano-agent")]
fn open_regular_file_nofollow(path: &Path) -> Result<File, NanoWarmSelectionError> {
    let mut options = fs::OpenOptions::new();
    options.read(true).custom_flags(libc::O_NOFOLLOW);
    let file = options
        .open(path)
        .map_err(|source| NanoWarmSelectionError::Io {
            operation: "open regular file without following links",
            path: path.to_path_buf(),
            source,
        })?;
    let metadata = file
        .metadata()
        .map_err(|source| NanoWarmSelectionError::Io {
            operation: "inspect opened regular file",
            path: path.to_path_buf(),
            source,
        })?;
    if !metadata.is_file() {
        return Err(NanoWarmSelectionError::NotRegularFile {
            path: path.to_path_buf(),
        });
    }
    Ok(file)
}

#[cfg(feature = "nano-agent")]
fn open_regular_file_at_nofollow(
    parent: impl AsFd,
    name: &OsStr,
    path: &Path,
) -> Result<(File, FilesystemObjectIdentity), NanoWarmSelectionError> {
    let descriptor = openat(
        parent.as_fd(),
        name,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|source| NanoWarmSelectionError::Io {
        operation: "open retained regular file without following links",
        path: path.to_path_buf(),
        source: rustix_errno_as_io(source),
    })?;
    let file = File::from(descriptor);
    let metadata = file
        .metadata()
        .map_err(|source| NanoWarmSelectionError::Io {
            operation: "inspect retained regular file",
            path: path.to_path_buf(),
            source,
        })?;
    if !metadata.is_file() {
        return Err(NanoWarmSelectionError::NotRegularFile {
            path: path.to_path_buf(),
        });
    }
    Ok((file, FilesystemObjectIdentity::from_metadata(&metadata)))
}

#[cfg(feature = "nano-agent")]
fn open_directory_at_nofollow(
    parent: impl AsFd,
    name: &OsStr,
    path: &Path,
) -> Result<(File, FilesystemObjectIdentity), NanoWarmSelectionError> {
    let descriptor = openat(
        parent.as_fd(),
        name,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|source| NanoWarmSelectionError::Io {
        operation: "open retained directory without following links",
        path: path.to_path_buf(),
        source: rustix_errno_as_io(source),
    })?;
    let file = File::from(descriptor);
    let metadata = file
        .metadata()
        .map_err(|source| NanoWarmSelectionError::Io {
            operation: "inspect retained directory",
            path: path.to_path_buf(),
            source,
        })?;
    if !metadata.is_dir() {
        return Err(NanoWarmSelectionError::Path(
            NanoMapPersistencePathError::NotDirectory {
                role: NanoMapPathRole::WarmSlamDatasetDirectory,
                path: path.to_path_buf(),
            },
        ));
    }
    Ok((file, FilesystemObjectIdentity::from_metadata(&metadata)))
}

#[cfg(feature = "nano-agent")]
fn require_regular_file_identity_at(
    parent: impl AsFd,
    name: &OsStr,
    path: &Path,
    expected: FilesystemObjectIdentity,
) -> Result<(), NanoWarmSelectionError> {
    let stat = statat(parent.as_fd(), name, AtFlags::SYMLINK_NOFOLLOW).map_err(|source| {
        NanoWarmSelectionError::Io {
            operation: "inspect retained selected file name",
            path: path.to_path_buf(),
            source: rustix_errno_as_io(source),
        }
    })?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile {
        return Err(NanoWarmSelectionError::NotRegularFile {
            path: path.to_path_buf(),
        });
    }
    let observed_device = u64::try_from(stat.st_dev).map_err(|_| NanoWarmSelectionError::Io {
        operation: "parse retained selected file device number",
        path: path.to_path_buf(),
        source: io::Error::new(
            io::ErrorKind::InvalidData,
            "file device number is not representable as u64",
        ),
    })?;
    let observed_inode = stat.st_ino;
    if (observed_device, observed_inode) != (expected.device, expected.inode) {
        return Err(NanoWarmSelectionError::FilesystemObjectChanged {
            path: path.to_path_buf(),
            expected_device: expected.device,
            expected_inode: expected.inode,
            observed_device,
            observed_inode,
        });
    }
    Ok(())
}

#[cfg(feature = "nano-agent")]
fn require_open_regular_file_identity(
    file: &File,
    path: &Path,
    expected: FilesystemObjectIdentity,
) -> Result<(), NanoWarmSelectionError> {
    let metadata = file
        .metadata()
        .map_err(|source| NanoWarmSelectionError::Io {
            operation: "inspect retained selected file descriptor",
            path: path.to_path_buf(),
            source,
        })?;
    if !metadata.is_file() {
        return Err(NanoWarmSelectionError::NotRegularFile {
            path: path.to_path_buf(),
        });
    }
    let observed = FilesystemObjectIdentity::from_metadata(&metadata);
    if observed != expected {
        return Err(NanoWarmSelectionError::FilesystemObjectChanged {
            path: path.to_path_buf(),
            expected_device: expected.device,
            expected_inode: expected.inode,
            observed_device: observed.device,
            observed_inode: observed.inode,
        });
    }
    Ok(())
}

#[cfg(feature = "nano-agent")]
fn open_unchanged_directory_nofollow(
    role: NanoMapPathRole,
    path: &Path,
    expected: FilesystemObjectIdentity,
) -> Result<File, NanoWarmSelectionError> {
    open_unchanged_directory_file(role, path, expected).map_err(NanoWarmSelectionError::Path)
}

#[cfg(feature = "nano-agent")]
fn open_unchanged_directory_file(
    role: NanoMapPathRole,
    path: &Path,
    expected: FilesystemObjectIdentity,
) -> Result<File, NanoMapPersistencePathError> {
    let mut options = fs::OpenOptions::new();
    options
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_DIRECTORY);
    let directory = options
        .open(path)
        .map_err(|source| NanoMapPersistencePathError::Io {
            operation: NanoMapPathOperation::OpenDirectoryForSync,
            role,
            path: path.to_path_buf(),
            source,
        })?;
    let metadata = directory
        .metadata()
        .map_err(|source| NanoMapPersistencePathError::Io {
            operation: NanoMapPathOperation::Inspect,
            role,
            path: path.to_path_buf(),
            source,
        })?;
    require_same_identity(
        role,
        path,
        expected,
        FilesystemObjectIdentity::from_metadata(&metadata),
    )?;
    Ok(directory)
}

#[cfg(feature = "nano-agent")]
fn require_directory_identity_at(
    directory: impl AsFd,
    role: NanoMapPathRole,
    path: &Path,
    expected: FilesystemObjectIdentity,
) -> Result<(), NanoMapPersistencePathError> {
    let stat = fstat(directory.as_fd()).map_err(|source| NanoMapPersistencePathError::Io {
        operation: NanoMapPathOperation::Inspect,
        role,
        path: path.to_path_buf(),
        source: rustix_errno_as_io(source),
    })?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::Directory {
        return Err(NanoMapPersistencePathError::NotDirectory {
            role,
            path: path.to_path_buf(),
        });
    }
    let actual_device =
        u64::try_from(stat.st_dev).map_err(|_| NanoMapPersistencePathError::Io {
            operation: NanoMapPathOperation::Inspect,
            role,
            path: path.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::InvalidData,
                "directory device number is not representable as u64",
            ),
        })?;
    let actual_inode = stat.st_ino;
    require_same_identity(
        role,
        path,
        expected,
        FilesystemObjectIdentity {
            device: actual_device,
            inode: actual_inode,
        },
    )
}

#[cfg(feature = "nano-agent")]
fn require_optional_regular_file_at(
    parent: impl AsFd,
    name: &OsStr,
    role: NanoMapPathRole,
    path: &Path,
) -> Result<(), NanoMapPersistencePathError> {
    let stat = match statat(parent.as_fd(), name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(stat) => stat,
        Err(Errno::NOENT) => return Ok(()),
        Err(source) => {
            return Err(NanoMapPersistencePathError::Io {
                operation: NanoMapPathOperation::Inspect,
                role,
                path: path.to_path_buf(),
                source: rustix_errno_as_io(source),
            });
        }
    };
    match FileType::from_raw_mode(stat.st_mode) {
        FileType::RegularFile => Ok(()),
        FileType::Symlink => Err(NanoMapPersistencePathError::Symlink {
            role,
            path: path.to_path_buf(),
        }),
        _ => Err(NanoMapPersistencePathError::NotRegularFile {
            role,
            path: path.to_path_buf(),
        }),
    }
}

/// Move the quota-verified encoded artifact into its immutable session with
/// no second encode, allocation, copy, or overwrite.
#[cfg(feature = "nano-agent")]
fn relocate_staged_checkpoint_occupancy(
    staged_path: &Path,
    staged_parent: &Path,
    staged_parent_identity: FilesystemObjectIdentity,
    occupancy_path: &Path,
    dataset_directory: &Path,
    dataset_directory_identity: FilesystemObjectIdentity,
) -> Result<(), NanoWarmSelectionError> {
    let staged_name = staged_path
        .file_name()
        .ok_or(NanoWarmSelectionError::InvalidDatasetDirectoryName)?;
    let occupancy_name = occupancy_path
        .file_name()
        .ok_or(NanoWarmSelectionError::InvalidDatasetDirectoryName)?;
    if staged_path.parent() != Some(staged_parent)
        || occupancy_path.parent() != Some(dataset_directory)
    {
        return Err(NanoWarmSelectionError::InvalidDatasetDirectoryName);
    }
    let staged_identity =
        require_canonical_regular_file(NanoMapPathRole::SaveSnapshot, staged_path)
            .map_err(NanoWarmSelectionError::Path)?;
    let staged_parent_file = open_unchanged_directory_nofollow(
        NanoMapPathRole::SaveParent,
        staged_parent,
        staged_parent_identity,
    )?;
    let dataset_directory_file = open_unchanged_directory_nofollow(
        NanoMapPathRole::WarmSlamDatasetDirectory,
        dataset_directory,
        dataset_directory_identity,
    )?;
    renameat_with(
        &staged_parent_file,
        staged_name,
        &dataset_directory_file,
        occupancy_name,
        RenameFlags::NOREPLACE,
    )
    .map_err(|source| NanoWarmSelectionError::Io {
        operation: "relocate quota-verified occupancy without replacement",
        path: occupancy_path.to_path_buf(),
        source: io::Error::from_raw_os_error(source.raw_os_error()),
    })?;
    let relocated_identity =
        require_canonical_regular_file(NanoMapPathRole::WarmOccupancySnapshot, occupancy_path)
            .map_err(NanoWarmSelectionError::Path)?;
    require_same_identity(
        NanoMapPathRole::WarmOccupancySnapshot,
        occupancy_path,
        staged_identity,
        relocated_identity,
    )
    .map_err(NanoWarmSelectionError::Path)?;
    dataset_directory_file
        .sync_all()
        .map_err(|source| NanoWarmSelectionError::Io {
            operation: "synchronize checkpoint occupancy directory",
            path: dataset_directory.to_path_buf(),
            source,
        })?;
    staged_parent_file
        .sync_all()
        .map_err(|source| NanoWarmSelectionError::Io {
            operation: "synchronize emptied quota-staging directory",
            path: staged_parent.to_path_buf(),
            source,
        })
}

#[cfg(all(feature = "nano-agent", test))]
fn read_bounded_regular_file(
    path: &Path,
    maximum_bytes: u64,
) -> Result<Vec<u8>, NanoWarmSelectionError> {
    let mut file = open_regular_file_nofollow(path)?;
    read_bounded_open_file(&mut file, path, maximum_bytes)
}

#[cfg(feature = "nano-agent")]
fn read_bounded_open_file(
    file: &mut File,
    path: &Path,
    maximum_bytes: u64,
) -> Result<Vec<u8>, NanoWarmSelectionError> {
    let metadata = file
        .metadata()
        .map_err(|source| NanoWarmSelectionError::Io {
            operation: "inspect bounded input",
            path: path.to_path_buf(),
            source,
        })?;
    let expected_bytes = metadata.len();
    if expected_bytes > maximum_bytes {
        return Err(NanoWarmSelectionError::FileTooLarge {
            path: path.to_path_buf(),
            actual_bytes: expected_bytes,
            maximum_bytes,
        });
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|source| NanoWarmSelectionError::Io {
            operation: "rewind bounded input",
            path: path.to_path_buf(),
            source,
        })?;
    let capacity =
        usize::try_from(expected_bytes).map_err(|_| NanoWarmSelectionError::FileTooLarge {
            path: path.to_path_buf(),
            actual_bytes: expected_bytes,
            maximum_bytes,
        })?;
    let mut bytes = Vec::with_capacity(capacity);
    Read::by_ref(file)
        .take(expected_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|source| NanoWarmSelectionError::Io {
            operation: "read bounded input",
            path: path.to_path_buf(),
            source,
        })?;
    if bytes.len() != capacity {
        return if bytes.len() < capacity {
            Err(NanoWarmSelectionError::Truncated {
                path: path.to_path_buf(),
                expected_bytes,
                actual_bytes: bytes.len(),
            })
        } else {
            Err(NanoWarmSelectionError::LengthMismatch {
                path: path.to_path_buf(),
                expected: expected_bytes,
                observed: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            })
        };
    }
    Ok(bytes)
}

#[cfg(feature = "nano-agent")]
fn hash_bounded_regular_file(
    path: &Path,
    maximum_bytes: u64,
) -> Result<([u8; SHA256_BYTES], u64), NanoWarmSelectionError> {
    const HASH_BUFFER_BYTES: usize = 64 * 1_024;

    let mut file = open_regular_file_nofollow(path)?;
    let metadata = file
        .metadata()
        .map_err(|source| NanoWarmSelectionError::Io {
            operation: "inspect hash input",
            path: path.to_path_buf(),
            source,
        })?;
    let expected_bytes = metadata.len();
    if expected_bytes > maximum_bytes {
        return Err(NanoWarmSelectionError::FileTooLarge {
            path: path.to_path_buf(),
            actual_bytes: expected_bytes,
            maximum_bytes,
        });
    }
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; HASH_BUFFER_BYTES];
    let mut observed = 0_u64;
    loop {
        let remaining = expected_bytes.saturating_sub(observed);
        if remaining == 0 {
            break;
        }
        let request = usize::try_from(remaining.min(HASH_BUFFER_BYTES as u64))
            .expect("bounded hash chunk fits usize");
        let read =
            file.read(&mut buffer[..request])
                .map_err(|source| NanoWarmSelectionError::Io {
                    operation: "hash bounded input",
                    path: path.to_path_buf(),
                    source,
                })?;
        if read == 0 {
            return Err(NanoWarmSelectionError::Truncated {
                path: path.to_path_buf(),
                expected_bytes,
                actual_bytes: usize::try_from(observed).unwrap_or(usize::MAX),
            });
        }
        hasher.update(&buffer[..read]);
        observed = observed
            .checked_add(u64::try_from(read).expect("read length fits u64"))
            .expect("observed bytes cannot exceed the admitted file length");
    }
    let mut trailing = [0_u8; 1];
    let trailing_bytes = file
        .read(&mut trailing)
        .map_err(|source| NanoWarmSelectionError::Io {
            operation: "verify hash input end",
            path: path.to_path_buf(),
            source,
        })?;
    if trailing_bytes != 0 {
        return Err(NanoWarmSelectionError::LengthMismatch {
            path: path.to_path_buf(),
            expected: expected_bytes,
            observed: expected_bytes.saturating_add(1),
        });
    }
    Ok((hasher.finalize().into(), observed))
}

#[cfg(feature = "nano-agent")]
struct DigestingReader<R> {
    inner: R,
    hasher: Sha256,
    observed_bytes: u64,
}

#[cfg(feature = "nano-agent")]
impl<R> DigestingReader<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
            observed_bytes: 0,
        }
    }

    fn finish(self) -> ([u8; SHA256_BYTES], u64) {
        (self.hasher.finalize().into(), self.observed_bytes)
    }
}

#[cfg(feature = "nano-agent")]
impl<R: Read> Read for DigestingReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let read = self.inner.read(buffer)?;
        self.hasher.update(&buffer[..read]);
        self.observed_bytes = self
            .observed_bytes
            .checked_add(u64::try_from(read).expect("read length fits u64"))
            .ok_or_else(|| io::Error::other("digested byte count overflow"))?;
        Ok(read)
    }
}

#[cfg(feature = "nano-agent")]
pub(crate) struct SelectedManifestReader<'file> {
    reader: DigestingReader<&'file mut File>,
    expected_digest: [u8; SHA256_BYTES],
    expected_bytes: u64,
    path: PathBuf,
}

#[cfg(feature = "nano-agent")]
impl Read for SelectedManifestReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.reader.read(buffer)
    }
}

#[cfg(feature = "nano-agent")]
impl SelectedManifestReader<'_> {
    pub(crate) fn verify(mut self) -> Result<(), NanoWarmSelectionError> {
        let mut trailing = [0_u8; 8 * 1_024];
        loop {
            let read = self
                .read(&mut trailing)
                .map_err(|source| NanoWarmSelectionError::Io {
                    operation: "finish digesting retained selected manifest",
                    path: self.path.clone(),
                    source,
                })?;
            if read == 0 {
                break;
            }
        }
        let (observed_digest, observed_bytes) = self.reader.finish();
        verify_selected_digest(
            &self.path,
            self.expected_digest,
            self.expected_bytes,
            observed_digest,
            observed_bytes,
        )
    }
}

#[cfg(feature = "nano-agent")]
fn hash_bounded_open_file(
    file: &mut File,
    path: &Path,
    maximum_bytes: u64,
) -> Result<([u8; SHA256_BYTES], u64), NanoWarmSelectionError> {
    const HASH_BUFFER_BYTES: usize = 64 * 1_024;

    let metadata = file
        .metadata()
        .map_err(|source| NanoWarmSelectionError::Io {
            operation: "inspect retained hash input",
            path: path.to_path_buf(),
            source,
        })?;
    if !metadata.is_file() {
        return Err(NanoWarmSelectionError::NotRegularFile {
            path: path.to_path_buf(),
        });
    }
    if metadata.len() > maximum_bytes {
        return Err(NanoWarmSelectionError::FileTooLarge {
            path: path.to_path_buf(),
            actual_bytes: metadata.len(),
            maximum_bytes,
        });
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|source| NanoWarmSelectionError::Io {
            operation: "rewind retained hash input",
            path: path.to_path_buf(),
            source,
        })?;
    let mut reader = DigestingReader::new(file);
    let mut buffer = [0_u8; HASH_BUFFER_BYTES];
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|source| NanoWarmSelectionError::Io {
                operation: "hash retained input",
                path: path.to_path_buf(),
                source,
            })?;
        if read == 0 {
            break;
        }
        if reader.observed_bytes > maximum_bytes {
            return Err(NanoWarmSelectionError::FileTooLarge {
                path: path.to_path_buf(),
                actual_bytes: reader.observed_bytes,
                maximum_bytes,
            });
        }
    }
    Ok(reader.finish())
}

#[cfg(feature = "nano-agent")]
fn verify_selected_digest(
    path: &Path,
    expected_digest: [u8; SHA256_BYTES],
    expected_bytes: u64,
    observed_digest: [u8; SHA256_BYTES],
    observed_bytes: u64,
) -> Result<(), NanoWarmSelectionError> {
    if observed_bytes != expected_bytes {
        return Err(NanoWarmSelectionError::LengthMismatch {
            path: path.to_path_buf(),
            expected: expected_bytes,
            observed: observed_bytes,
        });
    }
    if observed_digest != expected_digest {
        return Err(NanoWarmSelectionError::DigestMismatch {
            path: path.to_path_buf(),
            expected: expected_digest,
            observed: observed_digest,
        });
    }
    Ok(())
}

#[cfg(feature = "nano-agent")]
fn verify_selected_content_from_handles(
    manifest_file: &mut File,
    occupancy_file: &mut File,
    dataset_directory: &Path,
    occupancy_path: &Path,
    expected: NanoWarmSelectionContentIdentity,
) -> Result<(), NanoWarmSelectionError> {
    let manifest_path = dataset_directory.join(DATASET_MANIFEST_FILE);
    let (observed_manifest, observed_manifest_bytes) = hash_bounded_open_file(
        manifest_file,
        &manifest_path,
        MAX_NANO_DATASET_MANIFEST_BYTES,
    )?;
    verify_selected_digest(
        &manifest_path,
        expected.dataset_manifest_sha256,
        expected.dataset_manifest_bytes,
        observed_manifest,
        observed_manifest_bytes,
    )?;
    let (observed_occupancy, observed_occupancy_bytes) = hash_bounded_open_file(
        occupancy_file,
        occupancy_path,
        MAX_NANO_SELECTED_OCCUPANCY_BYTES,
    )?;
    verify_selected_digest(
        occupancy_path,
        expected.occupancy_sha256,
        expected.occupancy_bytes,
        observed_occupancy,
        observed_occupancy_bytes,
    )
}

#[cfg(all(feature = "nano-agent", test))]
fn load_warm_selection(
    selection_root: &Path,
) -> Result<ParsedNanoWarmSelectionV1, NanoWarmSelectionError> {
    let path = selection_root.join(NANO_WARM_SELECTION_FILE);
    let bytes = read_bounded_regular_file(&path, MAX_NANO_WARM_SELECTION_BYTES)?;
    parse_warm_selection_bytes(selection_root, &bytes)
}

#[cfg(feature = "nano-agent")]
fn parse_warm_selection_bytes(
    selection_root: &Path,
    bytes: &[u8],
) -> Result<ParsedNanoWarmSelectionV1, NanoWarmSelectionError> {
    let dto: NanoWarmSelectionV1Dto =
        serde_json::from_slice(bytes).map_err(NanoWarmSelectionError::Json)?;
    if dto.schema_version != NANO_WARM_SELECTION_SCHEMA_VERSION {
        return Err(NanoWarmSelectionError::UnsupportedSchema {
            actual: dto.schema_version,
        });
    }
    let dataset_directory_name = parse_dataset_directory_name(&dto.dataset_directory_name)
        .map_err(|()| NanoWarmSelectionError::InvalidDatasetDirectoryName)?
        .to_owned();
    if dto.occupancy_file_name != NANO_WARM_OCCUPANCY_FILE {
        return Err(NanoWarmSelectionError::InvalidDatasetDirectoryName);
    }
    if dto.dataset_manifest_bytes == 0
        || dto.dataset_manifest_bytes > MAX_NANO_DATASET_MANIFEST_BYTES
    {
        return Err(NanoWarmSelectionError::FileTooLarge {
            path: selection_root
                .join(&dataset_directory_name)
                .join(DATASET_MANIFEST_FILE),
            actual_bytes: dto.dataset_manifest_bytes,
            maximum_bytes: MAX_NANO_DATASET_MANIFEST_BYTES,
        });
    }
    if dto.occupancy_bytes == 0 || dto.occupancy_bytes > MAX_NANO_SELECTED_OCCUPANCY_BYTES {
        return Err(NanoWarmSelectionError::FileTooLarge {
            path: selection_root
                .join(&dataset_directory_name)
                .join(NANO_WARM_OCCUPANCY_FILE),
            actual_bytes: dto.occupancy_bytes,
            maximum_bytes: MAX_NANO_SELECTED_OCCUPANCY_BYTES,
        });
    }
    let map_epoch_id = RecordedMapEpochId::try_new(dto.map_epoch_id)
        .map_err(|_| NanoWarmSelectionError::ZeroMapEpoch)?;
    Ok(ParsedNanoWarmSelectionV1 {
        dataset_directory_name,
        content: NanoWarmSelectionContentIdentity {
            dataset_manifest_sha256: parse_sha256_hex(
                "dataset_manifest_sha256_hex",
                &dto.dataset_manifest_sha256_hex,
            )?,
            occupancy_sha256: parse_sha256_hex("occupancy_sha256_hex", &dto.occupancy_sha256_hex)?,
            dataset_manifest_bytes: dto.dataset_manifest_bytes,
            occupancy_bytes: dto.occupancy_bytes,
        },
        map_epoch_id,
        map_revision: dto.map_revision,
    })
}

#[cfg(feature = "nano-agent")]
fn publish_warm_selection(
    selection_root: &Path,
    selection_root_identity: FilesystemObjectIdentity,
    selection: &ParsedNanoWarmSelectionV1,
) -> Result<PathBuf, NanoWarmSelectionError> {
    parse_dataset_directory_name(&selection.dataset_directory_name)
        .map_err(|()| NanoWarmSelectionError::InvalidDatasetDirectoryName)?;
    let dto = NanoWarmSelectionV1Dto {
        schema_version: NANO_WARM_SELECTION_SCHEMA_VERSION,
        dataset_directory_name: selection.dataset_directory_name.clone(),
        dataset_manifest_sha256_hex: encode_sha256_hex(selection.content.dataset_manifest_sha256),
        dataset_manifest_bytes: selection.content.dataset_manifest_bytes,
        occupancy_file_name: NANO_WARM_OCCUPANCY_FILE.to_owned(),
        occupancy_sha256_hex: encode_sha256_hex(selection.content.occupancy_sha256),
        occupancy_bytes: selection.content.occupancy_bytes,
        map_epoch_id: selection.map_epoch_id.as_u64(),
        map_revision: selection.map_revision,
    };
    let bytes = serde_json::to_vec_pretty(&dto).map_err(NanoWarmSelectionError::Json)?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_NANO_WARM_SELECTION_BYTES {
        return Err(NanoWarmSelectionError::SelectionEncodingTooLarge {
            actual_bytes: bytes.len(),
            maximum_bytes: MAX_NANO_WARM_SELECTION_BYTES,
        });
    }
    let destination = selection_root.join(NANO_WARM_SELECTION_FILE);
    let root = open_unchanged_directory_nofollow(
        NanoMapPathRole::WarmSlamDatasetDirectory,
        selection_root,
        selection_root_identity,
    )?;
    require_optional_regular_file_at(
        &root,
        OsStr::new(NANO_WARM_SELECTION_FILE),
        NanoMapPathRole::WarmSlamDatasetDirectory,
        &destination,
    )
    .map_err(NanoWarmSelectionError::Path)?;
    publish_warm_selection_bytes_at(&root, selection_root, &destination, &bytes)?;
    Ok(destination)
}

#[cfg(feature = "nano-agent")]
fn publish_warm_selection_bytes_at(
    root: &File,
    selection_root: &Path,
    destination: &Path,
    bytes: &[u8],
) -> Result<(), NanoWarmSelectionError> {
    let mut temporary = None;
    for _ in 0..16 {
        let serial = NEXT_SELECTION_TEMPORARY.fetch_add(1, Ordering::Relaxed);
        let name = format!(
            ".{NANO_WARM_SELECTION_FILE}.tmp-{}-{serial:016x}",
            std::process::id()
        );
        let path = selection_root.join(&name);
        match openat(
            root,
            &name,
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::from_raw_mode(0o600),
        ) {
            Ok(file) => {
                temporary = Some((name, path, file));
                break;
            }
            Err(Errno::EXIST) => {}
            Err(source) => {
                return Err(NanoWarmSelectionError::Io {
                    operation: "create warm selection temporary file",
                    path,
                    source: rustix_errno_as_io(source),
                });
            }
        }
    }
    let Some((temporary_name, temporary_path, temporary_fd)) = temporary else {
        return Err(NanoWarmSelectionError::TemporaryNameCollisions {
            parent: selection_root.to_path_buf(),
        });
    };
    let mut file = File::from(temporary_fd);
    let temporary_identity =
        FilesystemObjectIdentity::from_metadata(&file.metadata().map_err(|source| {
            warm_selection_error_with_cleanup(
                root,
                &temporary_name,
                "inspect warm selection temporary file",
                &temporary_path,
                source,
            )
        })?);
    file.write_all(bytes).map_err(|source| {
        warm_selection_error_with_cleanup(
            root,
            &temporary_name,
            "write warm selection temporary file",
            &temporary_path,
            source,
        )
    })?;
    file.sync_all().map_err(|source| {
        warm_selection_error_with_cleanup(
            root,
            &temporary_name,
            "synchronize warm selection temporary file",
            &temporary_path,
            source,
        )
    })?;
    let synchronized_identity =
        FilesystemObjectIdentity::from_metadata(&file.metadata().map_err(|source| {
            warm_selection_error_with_cleanup(
                root,
                &temporary_name,
                "inspect synchronized warm selection temporary file",
                &temporary_path,
                source,
            )
        })?);
    if synchronized_identity != temporary_identity {
        return Err(warm_selection_error_with_cleanup(
            root,
            &temporary_name,
            "inspect synchronized warm selection temporary file",
            &temporary_path,
            io::Error::other("temporary warm selection identity changed"),
        ));
    }
    renameat(
        root,
        &temporary_name,
        root,
        OsStr::new(NANO_WARM_SELECTION_FILE),
    )
    .map_err(|source| {
        warm_selection_error_with_cleanup(
            root,
            &temporary_name,
            "atomically publish warm selection",
            destination,
            rustix_errno_as_io(source),
        )
    })?;
    require_published_selection_identity(root, destination, temporary_identity)?;
    fsync(root).map_err(|source| {
        NanoWarmSelectionError::SelectionPublishedButDurabilityUnconfirmed {
            selection_path: destination.to_path_buf(),
            operation: NanoWarmSelectionPostPublishOperation::SynchronizeSelectionRoot,
            source: rustix_errno_as_io(source),
        }
    })?;
    require_published_selection_identity(root, destination, temporary_identity)?;
    drop(file);
    Ok(())
}

#[cfg(feature = "nano-agent")]
fn warm_selection_error_with_cleanup(
    root: &File,
    temporary_name: &str,
    operation: &'static str,
    path: &Path,
    source: io::Error,
) -> NanoWarmSelectionError {
    let cleanup = unlinkat(root, temporary_name, AtFlags::empty());
    let source = match cleanup {
        Ok(()) | Err(Errno::NOENT) => source,
        Err(cleanup) => io::Error::new(
            source.kind(),
            format!("{source}; temporary cleanup also failed: {cleanup}"),
        ),
    };
    NanoWarmSelectionError::Io {
        operation,
        path: path.to_path_buf(),
        source,
    }
}

#[cfg(feature = "nano-agent")]
fn require_published_selection_identity(
    root: &File,
    destination: &Path,
    expected: FilesystemObjectIdentity,
) -> Result<(), NanoWarmSelectionError> {
    let observed = statat(
        root,
        OsStr::new(NANO_WARM_SELECTION_FILE),
        AtFlags::SYMLINK_NOFOLLOW,
    )
    .map_err(
        |source| NanoWarmSelectionError::SelectionPublishedButDurabilityUnconfirmed {
            selection_path: destination.to_path_buf(),
            operation: NanoWarmSelectionPostPublishOperation::InspectPublishedSelection,
            source: rustix_errno_as_io(source),
        },
    )?;
    if u64::try_from(observed.st_dev).ok() != Some(expected.device)
        || observed.st_ino != expected.inode
    {
        return Err(
            NanoWarmSelectionError::SelectionPublishedButDurabilityUnconfirmed {
                selection_path: destination.to_path_buf(),
                operation: NanoWarmSelectionPostPublishOperation::InspectPublishedSelection,
                source: io::Error::other("published warm selection identity changed"),
            },
        );
    }
    Ok(())
}

#[cfg(feature = "nano-agent")]
fn rustix_errno_as_io(source: Errno) -> io::Error {
    io::Error::from_raw_os_error(source.raw_os_error())
}

trait SaveMapResponder {
    type Error;

    fn command_kind(&self) -> AgentControlCommandKindV1;
    fn completed(self) -> Result<(), Self::Error>;
    fn rejected(
        self,
        code: AgentControlRejectionCodeV1,
        retryable: bool,
    ) -> Result<(), Self::Error>;
}

impl SaveMapResponder for AgentControlClaimedRequest {
    type Error = AgentControlDispatchResponseError;

    fn command_kind(&self) -> AgentControlCommandKindV1 {
        self.request().command().kind()
    }

    fn completed(self) -> Result<(), Self::Error> {
        self.respond_completed()
    }

    fn rejected(
        self,
        code: AgentControlRejectionCodeV1,
        retryable: bool,
    ) -> Result<(), Self::Error> {
        self.reject(code, retryable)
    }
}

#[derive(Debug)]
enum ClaimedSaveFlowError<ResponseError> {
    WrongCommandRejected {
        actual: AgentControlCommandKindV1,
    },
    WrongCommandResponseFailed {
        actual: AgentControlCommandKindV1,
        response: ResponseError,
    },
    SaveRejected {
        source: NanoMapSaveError,
    },
    SaveAndRejectionResponseFailed {
        source: NanoMapSaveError,
        response: ResponseError,
    },
    SavedButCompletionResponseFailed {
        receipt: NanoMapSaveReceipt,
        response: ResponseError,
    },
}

fn finish_claimed_save<R, F>(
    responder: R,
    save: F,
) -> Result<NanoMapSaveReceipt, ClaimedSaveFlowError<R::Error>>
where
    R: SaveMapResponder,
    F: FnOnce() -> Result<NanoMapSaveReceipt, NanoMapSaveError>,
{
    let actual = responder.command_kind();
    if actual != AgentControlCommandKindV1::SaveMap {
        return match responder.rejected(AgentControlRejectionCodeV1::InternalFault, false) {
            Ok(()) => Err(ClaimedSaveFlowError::WrongCommandRejected { actual }),
            Err(response) => {
                Err(ClaimedSaveFlowError::WrongCommandResponseFailed { actual, response })
            }
        };
    }

    match save() {
        Ok(receipt) => match responder.completed() {
            Ok(()) => Ok(receipt),
            Err(response) => {
                Err(ClaimedSaveFlowError::SavedButCompletionResponseFailed { receipt, response })
            }
        },
        Err(source) => {
            let (code, retryable) = source.rejection();
            match responder.rejected(code, retryable) {
                Ok(()) => Err(ClaimedSaveFlowError::SaveRejected { source }),
                Err(response) => {
                    Err(ClaimedSaveFlowError::SaveAndRejectionResponseFailed { source, response })
                }
            }
        }
    }
}

impl From<ClaimedSaveFlowError<AgentControlDispatchResponseError>> for NanoMapSaveCommandError {
    fn from(source: ClaimedSaveFlowError<AgentControlDispatchResponseError>) -> Self {
        match source {
            ClaimedSaveFlowError::WrongCommandRejected { actual } => {
                Self::WrongCommandRejected { actual }
            }
            ClaimedSaveFlowError::WrongCommandResponseFailed { actual, response } => {
                Self::WrongCommandResponseFailed { actual, response }
            }
            ClaimedSaveFlowError::SaveRejected { source } => Self::SaveRejected { source },
            ClaimedSaveFlowError::SaveAndRejectionResponseFailed { source, response } => {
                Self::SaveAndRejectionResponseFailed { source, response }
            }
            ClaimedSaveFlowError::SavedButCompletionResponseFailed { receipt, response } => {
                Self::SavedButCompletionResponseFailed { receipt, response }
            }
        }
    }
}

#[derive(Clone, Copy)]
enum SaveParentAdmission {
    RequireExisting,
    CreateMissingDirectChild,
}

fn admit_save_parent(
    state_root: &Path,
    state_root_identity: FilesystemObjectIdentity,
    save_parent: &Path,
    admission: SaveParentAdmission,
) -> Result<FilesystemObjectIdentity, NanoMapPersistencePathError> {
    match fs::symlink_metadata(save_parent) {
        Ok(_) => require_canonical_directory(NanoMapPathRole::SaveParent, save_parent),
        Err(source) if source.kind() == io::ErrorKind::NotFound => match admission {
            SaveParentAdmission::RequireExisting => Err(NanoMapPersistencePathError::Missing {
                role: NanoMapPathRole::SaveParent,
                path: save_parent.to_path_buf(),
            }),
            SaveParentAdmission::CreateMissingDirectChild => create_direct_child_save_parent(
                state_root,
                state_root_identity,
                save_parent,
                |path| {
                    let mut builder = fs::DirBuilder::new();
                    builder.recursive(false).mode(0o700);
                    builder.create(path)
                },
                sync_directory,
            ),
        },
        Err(source) => Err(NanoMapPersistencePathError::Io {
            operation: NanoMapPathOperation::Inspect,
            role: NanoMapPathRole::SaveParent,
            path: save_parent.to_path_buf(),
            source,
        }),
    }
}

fn create_direct_child_save_parent<Create, Sync>(
    state_root: &Path,
    state_root_identity: FilesystemObjectIdentity,
    save_parent: &Path,
    create: Create,
    sync: Sync,
) -> Result<FilesystemObjectIdentity, NanoMapPersistencePathError>
where
    Create: FnOnce(&Path) -> io::Result<()>,
    Sync: FnOnce(&Path, FilesystemObjectIdentity) -> Result<(), NanoMapPersistencePathError>,
{
    if save_parent.parent() != Some(state_root) {
        return Err(
            NanoMapPersistencePathError::SaveParentCreationIsNotDirectChild {
                state_root: state_root.to_path_buf(),
                save_parent: save_parent.to_path_buf(),
            },
        );
    }

    match create(save_parent) {
        Ok(()) => {}
        Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {}
        Err(source) => {
            return Err(NanoMapPersistencePathError::Io {
                operation: NanoMapPathOperation::CreateDirectChildDirectory,
                role: NanoMapPathRole::SaveParent,
                path: save_parent.to_path_buf(),
                source,
            });
        }
    }

    let save_parent_identity =
        require_canonical_directory(NanoMapPathRole::SaveParent, save_parent)?;
    require_unchanged_directory(NanoMapPathRole::StateRoot, state_root, state_root_identity)?;
    sync(state_root, state_root_identity)?;
    Ok(save_parent_identity)
}

fn sync_directory(
    path: &Path,
    expected: FilesystemObjectIdentity,
) -> Result<(), NanoMapPersistencePathError> {
    let directory = File::open(path).map_err(|source| NanoMapPersistencePathError::Io {
        operation: NanoMapPathOperation::OpenDirectoryForSync,
        role: NanoMapPathRole::StateRoot,
        path: path.to_path_buf(),
        source,
    })?;
    let metadata = directory
        .metadata()
        .map_err(|source| NanoMapPersistencePathError::Io {
            operation: NanoMapPathOperation::Inspect,
            role: NanoMapPathRole::StateRoot,
            path: path.to_path_buf(),
            source,
        })?;
    require_same_identity(
        NanoMapPathRole::StateRoot,
        path,
        expected,
        FilesystemObjectIdentity::from_metadata(&metadata),
    )?;
    directory
        .sync_all()
        .map_err(|source| NanoMapPersistencePathError::Io {
            operation: NanoMapPathOperation::SyncDirectory,
            role: NanoMapPathRole::StateRoot,
            path: path.to_path_buf(),
            source,
        })
}

fn require_strict_descendant(
    role: NanoMapPathRole,
    state_root: &Path,
    configured: &Path,
) -> Result<(), NanoMapPersistencePathError> {
    let Ok(relative) = configured.strip_prefix(state_root) else {
        return Err(NanoMapPersistencePathError::OutsideStateRoot {
            role,
            state_root: state_root.to_path_buf(),
            configured: configured.to_path_buf(),
        });
    };
    if relative.as_os_str().is_empty() {
        return Err(NanoMapPersistencePathError::AliasesStateRoot {
            role,
            state_root: state_root.to_path_buf(),
        });
    }
    Ok(())
}

fn inspect(
    role: NanoMapPathRole,
    path: &Path,
) -> Result<fs::Metadata, NanoMapPersistencePathError> {
    fs::symlink_metadata(path).map_err(|source| {
        if source.kind() == io::ErrorKind::NotFound {
            NanoMapPersistencePathError::Missing {
                role,
                path: path.to_path_buf(),
            }
        } else {
            NanoMapPersistencePathError::Io {
                operation: NanoMapPathOperation::Inspect,
                role,
                path: path.to_path_buf(),
                source,
            }
        }
    })
}

fn require_not_symlink(
    role: NanoMapPathRole,
    path: &Path,
    metadata: &fs::Metadata,
) -> Result<(), NanoMapPersistencePathError> {
    if metadata.file_type().is_symlink() {
        Err(NanoMapPersistencePathError::Symlink {
            role,
            path: path.to_path_buf(),
        })
    } else {
        Ok(())
    }
}

fn require_exact_canonical_path(
    role: NanoMapPathRole,
    path: &Path,
) -> Result<(), NanoMapPersistencePathError> {
    let resolved = fs::canonicalize(path).map_err(|source| NanoMapPersistencePathError::Io {
        operation: NanoMapPathOperation::Canonicalize,
        role,
        path: path.to_path_buf(),
        source,
    })?;
    if resolved != path {
        return Err(NanoMapPersistencePathError::FilesystemPathIsNotCanonical {
            role,
            configured: path.to_path_buf(),
            resolved,
        });
    }
    Ok(())
}

fn require_canonical_directory(
    role: NanoMapPathRole,
    path: &Path,
) -> Result<FilesystemObjectIdentity, NanoMapPersistencePathError> {
    let metadata = inspect(role, path)?;
    require_not_symlink(role, path, &metadata)?;
    if !metadata.is_dir() {
        return Err(NanoMapPersistencePathError::NotDirectory {
            role,
            path: path.to_path_buf(),
        });
    }
    require_exact_canonical_path(role, path)?;
    Ok(FilesystemObjectIdentity::from_metadata(&metadata))
}

fn require_canonical_regular_file(
    role: NanoMapPathRole,
    path: &Path,
) -> Result<FilesystemObjectIdentity, NanoMapPersistencePathError> {
    let metadata = inspect(role, path)?;
    require_not_symlink(role, path, &metadata)?;
    if !metadata.is_file() {
        return Err(NanoMapPersistencePathError::NotRegularFile {
            role,
            path: path.to_path_buf(),
        });
    }
    require_exact_canonical_path(role, path)?;
    Ok(FilesystemObjectIdentity::from_metadata(&metadata))
}

fn require_optional_canonical_regular_file(
    role: NanoMapPathRole,
    path: &Path,
) -> Result<(), NanoMapPersistencePathError> {
    match fs::symlink_metadata(path) {
        Ok(_) => require_canonical_regular_file(role, path).map(|_| ()),
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(NanoMapPersistencePathError::Io {
            operation: NanoMapPathOperation::Inspect,
            role,
            path: path.to_path_buf(),
            source,
        }),
    }
}

fn require_same_identity(
    role: NanoMapPathRole,
    path: &Path,
    expected: FilesystemObjectIdentity,
    actual: FilesystemObjectIdentity,
) -> Result<(), NanoMapPersistencePathError> {
    if actual == expected {
        Ok(())
    } else {
        Err(NanoMapPersistencePathError::FilesystemObjectChanged {
            role,
            path: path.to_path_buf(),
            expected_device: expected.device,
            expected_inode: expected.inode,
            actual_device: actual.device,
            actual_inode: actual.inode,
        })
    }
}

fn require_unchanged_directory(
    role: NanoMapPathRole,
    path: &Path,
    expected: FilesystemObjectIdentity,
) -> Result<(), NanoMapPersistencePathError> {
    let actual = require_canonical_directory(role, path)?;
    require_same_identity(role, path, expected, actual)
}

#[cfg(not(feature = "nano-agent"))]
fn require_unchanged_regular_file(
    role: NanoMapPathRole,
    path: &Path,
    expected: FilesystemObjectIdentity,
) -> Result<(), NanoMapPersistencePathError> {
    let actual = require_canonical_regular_file(role, path)?;
    require_same_identity(role, path, expected, actual)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::os::unix::fs::{PermissionsExt, symlink};
    use std::rc::Rc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use serde_json::json;

    use super::*;
    use crate::HostMonotonicTimestamp;
    use crate::dense::occupancy::{OccupancyCell, OccupancyGridGeometry};
    use crate::dense::occupancy_persistence::load_occupancy_map;
    use crate::map::SlamMap;
    use crate::navigation::{NavigationClockEpoch, NavigationMapEpochCoordinator};

    static NEXT_TEST_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory {
        path: PathBuf,
    }

    impl TestDirectory {
        fn create(label: &str) -> Self {
            let serial = NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = fs::canonicalize(std::env::temp_dir())
                .expect("canonical temporary root")
                .join(format!(
                    "kiko-nano-map-{label}-{}-{serial}",
                    std::process::id()
                ));
            fs::create_dir(&path).expect("create test root");
            Self { path }
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn policy(state_root: &Path, save_path: &Path, warm_start: bool) -> NanoMapPersistenceConfig {
        let warm = if warm_start {
            json!({
                "kind": "dataset_replay",
                "occupancy_snapshot_path": save_path,
                "slam_dataset_directory_path": state_root.join("dataset")
            })
        } else {
            json!({"kind": "none"})
        };
        let value = json!({
            "schema_version": 3,
            "control": {
                "socket_path": "/tmp/kiko-agent/control.sock",
                "read_timeout_ms": 100,
                "write_timeout_ms": 100,
                "runtime_response_timeout_ms": 500,
                "terminal_response_timeout_ms": 300000,
                "runtime_queue_capacity": 8,
                "operator_console": {
                    "bind_address": "127.0.0.1:9877",
                    "capability_path": "/tmp/kiko-agent/operator-console.capability",
                    "deadman_tick_ms": 20,
                    "manual_command_forward_mm_per_s": 100,
                    "manual_command_yaw_millirad_per_s": 500
                }
            },
            "inventory": {
                "manifest_path": "/opt/kiko/deployment/manifest.json",
                "artifact_root_path": "/opt/kiko/deployment/artifacts",
                "artifact_bindings": [
                    {
                        "kind": "calibration",
                        "artifact_id": "stereo-v1",
                        "relative_path": "calibration/stereo.json"
                    },
                    {
                        "kind": "plant",
                        "artifact_id": "drive-v1",
                        "relative_path": "plant/drive.json"
                    }
                ]
            },
            "map_persistence": {
                "save_snapshot_path": save_path,
                "warm_start": warm
            },
            "eye": {"mode": "disabled"},
            "head": {"mode": "disabled"},
            "rgb_expression": {"mode": "disabled"},
            "supervisor": {
                "maximum_authority_lease_ms": 1000,
                "maximum_zero_age_ms": 250
            },
            "live_mode_policy": {
                "startup": "disarmed_map_only",
                "manual": {"permission": "disabled"},
                "point_goal": {"permission": "disabled"},
                "frontier_explore": {"permission": "disabled"}
            }
        });
        super::super::NanoAgentPolicyConfigV3::parse_json(
            &serde_json::to_vec(&value).expect("policy JSON"),
        )
        .expect("test policy")
        .map_persistence()
        .clone()
    }

    fn owner(
        directory: &TestDirectory,
        warm_start: bool,
        maximum_cells: usize,
    ) -> NanoMapPersistenceOwner {
        fs::create_dir_all(directory.path.join("maps")).expect("map directory");
        if warm_start {
            fs::create_dir_all(directory.path.join("dataset")).expect("dataset directory");
        }
        let roots = NanoBootstrapRoots::try_new(
            PathBuf::from("/opt/kiko/deployment"),
            directory.path.clone(),
        )
        .expect("test roots");
        let config = policy(
            &directory.path,
            &directory.path.join("maps/current.kmap"),
            warm_start,
        );
        NanoMapPersistenceOwner::try_new(
            &roots,
            &config,
            OccupancyMapLimits::try_new(maximum_cells).expect("limits"),
        )
        .expect("persistence owner")
    }

    fn snapshot(map: &SlamMap, revision: u64, occupied_index: usize) -> OccupancyGridSnapshot {
        let geometry =
            OccupancyGridGeometry::try_new(0.1, [-0.2, -0.2], 2, 2, 4).expect("geometry");
        let mut cells = [OccupancyCell::Free; 4];
        cells[occupied_index] = OccupancyCell::Occupied;
        OccupancyGridSnapshot::from_test_cells(
            geometry,
            &cells,
            map.snapshot().instance_id(),
            revision,
        )
    }

    fn bindings(maps: &[&SlamMap]) -> Vec<CurrentMapEpochBinding> {
        let mut coordinator = NavigationMapEpochCoordinator::new();
        let clock = NavigationClockEpoch::new(HostMonotonicTimestamp::from_nanos(0));
        maps.iter()
            .enumerate()
            .map(|(index, map)| {
                coordinator
                    .start_epoch(
                        clock,
                        HostMonotonicTimestamp::from_nanos(index as u64),
                        map.snapshot().instance_id(),
                    )
                    .expect("map epoch")
                    .binding()
            })
            .collect()
    }

    #[test]
    fn one_slot_rejects_unbound_oversize_and_stale_snapshots_without_replacement() {
        let directory = TestDirectory::create("retention");
        let mut retention_owner = owner(&directory, false, 4);
        let first_map = SlamMap::new();
        let second_map = SlamMap::new();
        let epoch = bindings(&[&first_map, &second_map]);

        let first = retention_owner
            .retain_latest(epoch[0], snapshot(&first_map, 7, 0))
            .expect("first snapshot");
        assert_eq!(retention_owner.retained_snapshot_capacity(), 1);
        assert_eq!(first.replaced(), None);
        let first_identity = first.retained();

        assert!(matches!(
            retention_owner.retain_latest(epoch[0], snapshot(&first_map, 7, 1)),
            Err(NanoMapSnapshotRetentionError::StaleRevision {
                retained: 7,
                offered: 7,
                ..
            })
        ));
        let second = retention_owner
            .retain_latest(epoch[1], snapshot(&second_map, 1, 2))
            .expect("new epoch");
        assert_eq!(second.replaced(), Some(first_identity));
        assert!(matches!(
            retention_owner.retain_latest(epoch[0], snapshot(&first_map, 99, 3)),
            Err(NanoMapSnapshotRetentionError::StaleEpoch { .. })
        ));
        assert!(matches!(
            retention_owner.retain_latest(epoch[1], snapshot(&first_map, 100, 0)),
            Err(NanoMapSnapshotRetentionError::MapInstanceMismatch { .. })
        ));
        assert_eq!(retention_owner.latest_identity(), Some(second.retained()));

        let unbound = crate::dense::occupancy_persistence::decode_occupancy_map(
            &crate::dense::occupancy_persistence::encode_occupancy_map(&snapshot(
                &second_map,
                2,
                0,
            ))
            .expect("encode"),
            OccupancyMapLimits::try_new(4).expect("limits"),
        )
        .expect("decode without live identity");
        assert!(matches!(
            retention_owner.retain_latest(epoch[1], unbound),
            Err(NanoMapSnapshotRetentionError::UnboundSnapshot)
        ));

        let mut too_small = owner(&directory, false, 3);
        assert!(matches!(
            too_small.retain_latest(epoch[1], snapshot(&second_map, 2, 0)),
            Err(NanoMapSnapshotRetentionError::TooManyCells {
                actual_cells: 4,
                maximum_cells: 3
            })
        ));
    }

    #[test]
    fn revision_retention_property_keeps_the_strict_maximum() {
        let directory = TestDirectory::create("revision-property");
        let map = SlamMap::new();
        let binding = bindings(&[&map])[0];

        for pivot in 0..16_u64 {
            let mut owner = owner(&directory, false, 4);
            owner
                .retain_latest(binding, snapshot(&map, pivot, 0))
                .expect("initial revision");
            for offered in (0..16_u64).rev() {
                let previous = owner.latest_identity().expect("retained").revision();
                let result = owner.retain_latest(binding, snapshot(&map, offered, 1));
                if offered > previous {
                    result.expect("strictly newer revision");
                } else {
                    assert!(matches!(
                        result,
                        Err(NanoMapSnapshotRetentionError::StaleRevision { .. })
                    ));
                }
            }
            assert_eq!(
                owner.latest_identity().expect("latest").revision(),
                pivot.max(15)
            );
        }
    }

    #[test]
    #[cfg(feature = "nano-agent")]
    fn finalized_journal_epoch_and_revision_must_exactly_match_retained_occupancy() {
        let map = SlamMap::new();
        let retained = NanoMapSnapshotIdentity {
            map_epoch_id: bindings(&[&map])[0].map_epoch_id(),
            map_instance_id: map.snapshot().instance_id(),
            revision: 17,
        };
        let exact = NanoFinalizedJournalMapIdentity::new(retained.map_epoch_id(), 17);
        require_finalized_journal_map_matches_retained(Some(exact), retained)
            .expect("exact finalized journal identity");

        assert!(matches!(
            require_finalized_journal_map_matches_retained(None, retained),
            Err(NanoWarmCheckpointError::FinalizedJournalHasNoAcceptedMap)
        ));
        let stale = NanoFinalizedJournalMapIdentity::new(retained.map_epoch_id(), 16);
        assert!(matches!(
            require_finalized_journal_map_matches_retained(Some(stale), retained),
            Err(NanoWarmCheckpointError::FinalizedJournalMapMismatch {
                finalized,
                retained_epoch_id,
                retained_revision: 17,
            }) if finalized == stale && retained_epoch_id == retained.map_epoch_id()
        ));

        let other_map = SlamMap::new();
        let other_epoch = bindings(&[&map, &other_map])[1].map_epoch_id();
        let wrong_epoch = NanoFinalizedJournalMapIdentity::new(other_epoch, 17);
        assert!(matches!(
            require_finalized_journal_map_matches_retained(Some(wrong_epoch), retained),
            Err(NanoWarmCheckpointError::FinalizedJournalMapMismatch { finalized, .. })
                if finalized == wrong_epoch
        ));
    }

    #[test]
    fn save_has_explicit_empty_and_stale_errors_then_round_trips_exact_latest() {
        let directory = TestDirectory::create("save");
        let mut owner = owner(&directory, false, 4);
        assert!(matches!(
            owner.save_latest(),
            Err(NanoMapSaveError::NoSnapshot)
        ));

        let map = SlamMap::new();
        let binding = bindings(&[&map])[0];
        let old = owner
            .retain_latest(binding, snapshot(&map, 3, 0))
            .expect("old")
            .retained();
        let current = owner
            .retain_latest(binding, snapshot(&map, 4, 2))
            .expect("current")
            .retained();
        assert!(matches!(
            owner.save_selected(old),
            Err(NanoMapSaveError::StaleSelection {
                requested,
                retained
            }) if requested == old && retained == current
        ));
        assert!(!owner.save_snapshot_path().exists());

        let receipt = owner.save_selected(current).expect("durable save");
        assert_eq!(receipt.identity(), current);
        let loaded = load_occupancy_map(
            receipt.destination(),
            OccupancyMapLimits::try_new(4).expect("limits"),
        )
        .expect("load durable result");
        assert_eq!(loaded.revision(), 4);
        assert_eq!(loaded.class_ids(), &[1, 1, 2, 1]);
        assert_eq!(loaded.map_instance_id(), None);
        assert_eq!(
            fs::read_dir(directory.path.join("maps"))
                .expect("map directory")
                .count(),
            1
        );
    }

    #[test]
    fn injected_persistence_failure_never_returns_a_receipt_and_retains_retry_state() {
        let directory = TestDirectory::create("save-failure");
        let mut owner = owner(&directory, false, 4);
        let map = SlamMap::new();
        let binding = bindings(&[&map])[0];
        let identity = owner
            .retain_latest(binding, snapshot(&map, 1, 0))
            .expect("snapshot")
            .retained();

        let result = owner.save_with(Some(identity), |path, _| {
            Err(OccupancyMapSaveError::InvalidDestination {
                path: path.to_path_buf(),
            })
        });
        assert!(matches!(result, Err(NanoMapSaveError::Persistence(_))));
        assert_eq!(owner.latest_identity(), Some(identity));
        assert!(!owner.save_snapshot_path().exists());
    }

    #[test]
    fn policy_paths_are_state_contained_and_symlinks_are_rejected() {
        let directory = TestDirectory::create("paths");
        fs::create_dir(directory.path.join("maps")).expect("map directory");
        let roots = NanoBootstrapRoots::try_new(
            PathBuf::from("/opt/kiko/deployment"),
            directory.path.clone(),
        )
        .expect("roots");
        let outside = directory
            .path
            .parent()
            .expect("parent")
            .join("outside.kmap");
        let config = policy(&directory.path, &outside, false);
        assert!(matches!(
            NanoMapPersistenceOwner::try_new(
                &roots,
                &config,
                OccupancyMapLimits::try_new(4).expect("limits")
            ),
            Err(NanoMapPersistencePathError::OutsideStateRoot {
                role: NanoMapPathRole::SaveSnapshot,
                ..
            })
        ));

        let target = directory.path.join("target");
        fs::create_dir(&target).expect("target");
        let linked_parent = directory.path.join("linked");
        symlink(&target, &linked_parent).expect("directory symlink");
        let config = policy(&directory.path, &linked_parent.join("current.kmap"), false);
        assert!(matches!(
            NanoMapPersistenceOwner::try_new(
                &roots,
                &config,
                OccupancyMapLimits::try_new(4).expect("limits")
            ),
            Err(NanoMapPersistencePathError::Symlink {
                role: NanoMapPathRole::SaveParent,
                ..
            }) | Err(NanoMapPersistencePathError::FilesystemPathIsNotCanonical {
                role: NanoMapPathRole::SaveParent,
                ..
            })
        ));
    }

    #[test]
    fn explicit_constructor_creates_only_one_private_direct_child() {
        let directory = TestDirectory::create("create-parent");
        let roots = NanoBootstrapRoots::try_new(
            PathBuf::from("/opt/kiko/deployment"),
            directory.path.clone(),
        )
        .expect("roots");
        let save_path = directory.path.join("maps/current.kmap");
        let config = policy(&directory.path, &save_path, false);
        let limits = OccupancyMapLimits::try_new(4).expect("limits");

        assert!(matches!(
            NanoMapPersistenceOwner::try_new(&roots, &config, limits),
            Err(NanoMapPersistencePathError::Missing {
                role: NanoMapPathRole::SaveParent,
                ..
            })
        ));
        let owner = NanoMapPersistenceOwner::try_new_with_direct_save_parent_creation(
            &roots, &config, limits,
        )
        .expect("create direct save parent");
        assert_eq!(owner.save_snapshot_path(), save_path);
        let mode = fs::symlink_metadata(directory.path.join("maps"))
            .expect("created parent")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o700);

        let nested_save_path = directory.path.join("nested/maps/current.kmap");
        let nested_config = policy(&directory.path, &nested_save_path, false);
        assert!(matches!(
            NanoMapPersistenceOwner::try_new_with_direct_save_parent_creation(
                &roots,
                &nested_config,
                limits
            ),
            Err(NanoMapPersistencePathError::SaveParentCreationIsNotDirectChild { .. })
        ));
        assert!(!directory.path.join("nested").exists());
    }

    #[test]
    fn direct_child_creation_readmits_races_and_rejects_a_symlink_winner() {
        let directory = TestDirectory::create("create-race");
        let state_identity =
            require_canonical_directory(NanoMapPathRole::StateRoot, &directory.path)
                .expect("state identity");
        let target = directory.path.join("target");
        fs::create_dir(&target).expect("race target");
        let save_parent = directory.path.join("maps");

        let result = create_direct_child_save_parent(
            &directory.path,
            state_identity,
            &save_parent,
            |path| {
                symlink(&target, path)?;
                Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "simulated create race",
                ))
            },
            sync_directory,
        );
        assert!(matches!(
            result,
            Err(NanoMapPersistencePathError::Symlink {
                role: NanoMapPathRole::SaveParent,
                ..
            })
        ));
    }

    #[test]
    fn state_root_replacement_is_detected_before_save() {
        let directory = TestDirectory::create("root-replaced");
        let mut owner = owner(&directory, false, 4);
        let map = SlamMap::new();
        let binding = bindings(&[&map])[0];
        owner
            .retain_latest(binding, snapshot(&map, 1, 0))
            .expect("snapshot");

        let old_root = directory.path.with_extension("old-root");
        fs::rename(&directory.path, &old_root).expect("retain old root inode");
        fs::create_dir(&directory.path).expect("replacement root");
        fs::create_dir(directory.path.join("maps")).expect("replacement map parent");
        assert!(matches!(
            owner.save_latest(),
            Err(NanoMapSaveError::Path(
                NanoMapPersistencePathError::FilesystemObjectChanged {
                    role: NanoMapPathRole::StateRoot,
                    ..
                }
            ))
        ));

        fs::remove_dir_all(&directory.path).expect("remove replacement");
        fs::rename(&old_root, &directory.path).expect("restore test root");
    }

    #[test]
    fn parent_sync_failure_returns_no_admitted_parent_identity() {
        let directory = TestDirectory::create("parent-sync");
        let state_identity =
            require_canonical_directory(NanoMapPathRole::StateRoot, &directory.path)
                .expect("state identity");
        let save_parent = directory.path.join("maps");
        let result = create_direct_child_save_parent(
            &directory.path,
            state_identity,
            &save_parent,
            |path| fs::create_dir(path),
            |path, _| {
                Err(NanoMapPersistencePathError::Io {
                    operation: NanoMapPathOperation::SyncDirectory,
                    role: NanoMapPathRole::StateRoot,
                    path: path.to_path_buf(),
                    source: io::Error::other("injected directory sync failure"),
                })
            },
        );
        assert!(matches!(
            result,
            Err(NanoMapPersistencePathError::Io {
                operation: NanoMapPathOperation::SyncDirectory,
                ..
            })
        ));
    }

    #[test]
    #[cfg(not(feature = "nano-agent"))]
    fn warm_start_requires_exact_replay_and_never_claims_localization() {
        let directory = TestDirectory::create("warm");
        let mut owner = owner(&directory, true, 4);
        let map = SlamMap::new();
        let binding = bindings(&[&map])[0];
        let identity = owner
            .retain_latest(binding, snapshot(&map, 11, 2))
            .expect("live snapshot")
            .retained();
        owner.save_selected(identity).expect("save warm artifact");

        let NanoMapWarmStartLoad::DatasetReplayRequired(required) =
            owner.load_warm_start().expect("load warm request")
        else {
            panic!("dataset replay required");
        };
        assert_eq!(required.persisted_snapshot().map_instance_id(), None);
        assert_eq!(
            required.dataset_content_binding_status(),
            NanoDatasetContentBindingStatus::MissingImmutableContentIdentity
        );

        let replay = ReplayOccupancyEvidence::try_new(map.snapshot(), snapshot(&map, 11, 2))
            .expect("same replay map");
        let matched = required.verify_exact_replay(replay).expect("exact replay");
        assert_eq!(
            matched.replay_matched_map().map_instance_id(),
            map.snapshot().instance_id()
        );
        assert_eq!(
            matched.dataset_content_binding_status(),
            NanoDatasetContentBindingStatus::MissingImmutableContentIdentity
        );
    }

    #[test]
    #[cfg(not(feature = "nano-agent"))]
    fn warm_start_rejects_mismatched_replay_and_changed_dataset_directory() {
        let directory = TestDirectory::create("warm-mismatch");
        let mut owner = owner(&directory, true, 4);
        let map = SlamMap::new();
        let binding = bindings(&[&map])[0];
        owner
            .retain_latest(binding, snapshot(&map, 2, 0))
            .expect("snapshot");
        owner.save_latest().expect("save");

        let NanoMapWarmStartLoad::DatasetReplayRequired(required) =
            owner.load_warm_start().expect("load request")
        else {
            panic!("replay request");
        };
        let replay = ReplayOccupancyEvidence::try_new(map.snapshot(), snapshot(&map, 2, 1))
            .expect("replay evidence");
        assert!(matches!(
            required.verify_exact_replay(replay),
            Err(NanoWarmStartReplayBindError::ExactReplay(
                OccupancyReplayBindError::CellClassMismatch { .. }
            ))
        ));

        let NanoMapWarmStartLoad::DatasetReplayRequired(required) =
            owner.load_warm_start().expect("second request")
        else {
            panic!("replay request");
        };
        fs::rename(
            directory.path.join("dataset"),
            directory.path.join("dataset-replaced"),
        )
        .expect("retain old dataset inode");
        fs::create_dir(directory.path.join("dataset")).expect("replace dataset");
        let replay = ReplayOccupancyEvidence::try_new(map.snapshot(), snapshot(&map, 2, 0))
            .expect("replay evidence");
        assert!(matches!(
            required.verify_exact_replay(replay),
            Err(NanoWarmStartReplayBindError::Path(
                NanoMapPersistencePathError::FilesystemObjectChanged { .. }
            ))
        ));
    }

    #[cfg(feature = "nano-agent")]
    fn publish_test_warm_selection(
        directory: &TestDirectory,
        session_name: &str,
        map: &SlamMap,
        binding: CurrentMapEpochBinding,
        revision: u64,
        occupied_index: usize,
        manifest: &[u8],
    ) -> (PathBuf, NanoWarmSelectionContentIdentity) {
        let selection_root = directory.path.join("dataset");
        let session = selection_root.join(session_name);
        fs::create_dir(&session).expect("session directory");
        let manifest_path = session.join(DATASET_MANIFEST_FILE);
        fs::write(&manifest_path, manifest).expect("manifest");
        let occupancy_path = session.join(NANO_WARM_OCCUPANCY_FILE);
        let occupancy = snapshot(map, revision, occupied_index);
        crate::dense::occupancy_persistence::save_occupancy_map_atomic(&occupancy_path, &occupancy)
            .expect("session occupancy");
        let (dataset_manifest_sha256, dataset_manifest_bytes) =
            hash_bounded_regular_file(&manifest_path, MAX_NANO_DATASET_MANIFEST_BYTES)
                .expect("manifest digest");
        let (occupancy_sha256, occupancy_bytes) =
            hash_bounded_regular_file(&occupancy_path, MAX_NANO_SELECTED_OCCUPANCY_BYTES)
                .expect("occupancy digest");
        let content = NanoWarmSelectionContentIdentity {
            dataset_manifest_sha256,
            occupancy_sha256,
            dataset_manifest_bytes,
            occupancy_bytes,
        };
        let selection_root_identity =
            require_canonical_directory(NanoMapPathRole::WarmSlamDatasetDirectory, &selection_root)
                .expect("selection root identity");
        publish_warm_selection(
            &selection_root,
            selection_root_identity,
            &ParsedNanoWarmSelectionV1 {
                dataset_directory_name: session_name.to_owned(),
                content,
                map_epoch_id: binding.map_epoch_id(),
                map_revision: revision,
            },
        )
        .expect("selection");
        (session, content)
    }

    #[test]
    #[cfg(feature = "nano-agent")]
    fn production_selection_loads_only_the_exact_session_and_never_claims_localization() {
        let directory = TestDirectory::create("selected-warm");
        let owner = owner(&directory, true, 4);
        assert!(owner.requires_quiescent_warm_checkpoint());
        fs::create_dir(directory.path.join("dataset/session-unselected"))
            .expect("unselected session");

        let map = SlamMap::new();
        let binding = bindings(&[&map])[0];
        let (selected_session, content) = publish_test_warm_selection(
            &directory,
            "session-selected",
            &map,
            binding,
            11,
            2,
            br#"{"session":1}"#,
        );

        let NanoMapWarmStartLoad::DatasetReplayRequired(required) =
            owner.load_warm_start().expect("selected warm request")
        else {
            panic!("dataset replay required");
        };
        assert_eq!(
            required.slam_dataset_directory_path(),
            selected_session.as_path()
        );
        assert_eq!(
            required.occupancy_snapshot_path(),
            selected_session.join(NANO_WARM_OCCUPANCY_FILE)
        );
        assert_eq!(required.selected_map_epoch_id(), binding.map_epoch_id());
        assert_eq!(required.selected_map_revision(), 11);
        assert!(matches!(
            required.verify_selected_dataset_map_identity(
                RecordedMapEpochId::try_new(2).expect("different epoch"),
                11,
            ),
            Err(NanoWarmSelectionError::MapEpochMismatch {
                selected: 1,
                observed: 2,
            })
        ));
        assert!(matches!(
            required.verify_selected_dataset_map_identity(binding.map_epoch_id(), 10),
            Err(NanoWarmSelectionError::MapRevisionMismatch {
                selected: 11,
                observed: 10,
            })
        ));
        required
            .verify_selected_dataset_map_identity(binding.map_epoch_id(), 11)
            .expect("journal identity matches selection");
        let status = required.dataset_content_binding_status();
        assert_eq!(
            status.dataset_manifest_sha256(),
            Some(content.dataset_manifest_sha256)
        );
        assert_eq!(status.occupancy_sha256(), Some(content.occupancy_sha256));

        let replay = ReplayOccupancyEvidence::try_new(map.snapshot(), snapshot(&map, 11, 2))
            .expect("exact replay evidence");
        let matched = required.verify_exact_replay(replay).expect("exact replay");
        assert_eq!(
            matched.slam_dataset_directory_path(),
            selected_session.as_path()
        );
        assert_eq!(matched.dataset_content_binding_status(), status);
    }

    #[test]
    #[cfg(feature = "nano-agent")]
    fn quota_staging_inode_is_relocated_once_without_copy_or_replacement() {
        let directory = TestDirectory::create("checkpoint-relocation");
        let staging_parent = directory.path.join("maps");
        let dataset_directory = directory.path.join("dataset/session-1");
        fs::create_dir(&staging_parent).expect("staging parent");
        fs::create_dir_all(&dataset_directory).expect("dataset directory");
        let staged = staging_parent.join("current.kmap");
        let destination = dataset_directory.join(NANO_WARM_OCCUPANCY_FILE);
        fs::write(&staged, b"exact-encoded-occupancy").expect("staged map");
        let before = fs::metadata(&staged).expect("staged metadata");
        relocate_staged_checkpoint_occupancy(
            &staged,
            &staging_parent,
            require_canonical_directory(NanoMapPathRole::SaveParent, &staging_parent)
                .expect("staging parent identity"),
            &destination,
            &dataset_directory,
            require_canonical_directory(
                NanoMapPathRole::WarmSlamDatasetDirectory,
                &dataset_directory,
            )
            .expect("dataset identity"),
        )
        .expect("relocation");
        assert!(!staged.exists());
        assert_eq!(
            fs::read(&destination).expect("relocated bytes"),
            b"exact-encoded-occupancy"
        );
        let after = fs::metadata(&destination).expect("relocated metadata");
        assert_eq!((before.dev(), before.ino()), (after.dev(), after.ino()));

        fs::write(&staged, b"new").expect("second staged map");
        assert!(matches!(
            relocate_staged_checkpoint_occupancy(
                &staged,
                &staging_parent,
                require_canonical_directory(NanoMapPathRole::SaveParent, &staging_parent)
                    .expect("staging parent identity"),
                &destination,
                &dataset_directory,
                require_canonical_directory(
                    NanoMapPathRole::WarmSlamDatasetDirectory,
                    &dataset_directory,
                )
                .expect("dataset identity"),
            ),
            Err(NanoWarmSelectionError::Io { .. })
        ));
        assert_eq!(
            fs::read(&destination).expect("original destination retained"),
            b"exact-encoded-occupancy"
        );
        assert_eq!(fs::read(&staged).expect("new staging retained"), b"new");
    }

    #[test]
    #[cfg(feature = "nano-agent")]
    fn selected_content_is_rechecked_after_load_before_replay_binding() {
        let directory = TestDirectory::create("selected-warm-tamper");
        let owner = owner(&directory, true, 4);
        let map = SlamMap::new();
        let binding = bindings(&[&map])[0];
        let manifest = br#"{"session":1}"#;
        let (session, _) = publish_test_warm_selection(
            &directory,
            "session-selected",
            &map,
            binding,
            7,
            1,
            manifest,
        );
        let NanoMapWarmStartLoad::DatasetReplayRequired(required) =
            owner.load_warm_start().expect("selected warm request")
        else {
            panic!("dataset replay required");
        };
        let replacement = br#"{"session":2}"#;
        assert_eq!(replacement.len(), manifest.len());
        fs::write(session.join(DATASET_MANIFEST_FILE), replacement)
            .expect("same-length manifest replacement");
        let replay = ReplayOccupancyEvidence::try_new(map.snapshot(), snapshot(&map, 7, 1))
            .expect("exact occupancy replay");
        assert!(matches!(
            required.verify_exact_replay(replay),
            Err(NanoWarmStartReplayBindError::Selection(
                NanoWarmSelectionError::DigestMismatch { .. }
            ))
        ));
    }

    #[test]
    #[cfg(feature = "nano-agent")]
    fn selected_manifest_parse_stream_stays_on_one_handle_across_path_replacement() {
        let directory = TestDirectory::create("selected-manifest-handle");
        let owner = owner(&directory, true, 4);
        let map = SlamMap::new();
        let binding = bindings(&[&map])[0];
        let manifest = br#"{"session":"retained"}"#;
        let (session, _) = publish_test_warm_selection(
            &directory,
            "session-selected",
            &map,
            binding,
            7,
            1,
            manifest,
        );
        let NanoMapWarmStartLoad::DatasetReplayRequired(mut required) =
            owner.load_warm_start().expect("selected warm request")
        else {
            panic!("dataset replay required");
        };
        let manifest_path = session.join(DATASET_MANIFEST_FILE);
        let retained_path = session.join("manifest-retained.json");
        let mut selected_reader = required
            .selected_manifest_reader()
            .expect("retained manifest reader");
        fs::rename(&manifest_path, &retained_path).expect("move selected manifest pathname");
        fs::write(&manifest_path, br#"{"session":"attacker"}"#)
            .expect("replace selected manifest pathname");
        let mut parsed_bytes = Vec::new();
        selected_reader
            .read_to_end(&mut parsed_bytes)
            .expect("read retained manifest handle");
        assert_eq!(parsed_bytes, manifest);
        selected_reader
            .verify()
            .expect("retained bytes match selected digest");

        let replay = ReplayOccupancyEvidence::try_new(map.snapshot(), snapshot(&map, 7, 1))
            .expect("exact occupancy replay");
        assert!(matches!(
            required.verify_exact_replay(replay),
            Err(NanoWarmStartReplayBindError::Selection(
                NanoWarmSelectionError::FilesystemObjectChanged { .. }
            ))
        ));
    }

    #[test]
    #[cfg(feature = "nano-agent")]
    fn weak_selection_json_is_bounded_and_parsed_fail_closed_once() {
        let directory = TestDirectory::create("selection-parse");
        fs::create_dir(directory.path.join("dataset")).expect("selection root");
        let selection_root = directory.path.join("dataset");
        let path = selection_root.join(NANO_WARM_SELECTION_FILE);
        let digest = "00".repeat(SHA256_BYTES);
        let valid = json!({
            "schema_version": 1,
            "dataset_directory_name": "../escape",
            "dataset_manifest_sha256_hex": digest,
            "dataset_manifest_bytes": 1,
            "occupancy_file_name": NANO_WARM_OCCUPANCY_FILE,
            "occupancy_sha256_hex": "00".repeat(SHA256_BYTES),
            "occupancy_bytes": 1,
            "map_epoch_id": 1,
            "map_revision": 0
        });
        fs::write(&path, serde_json::to_vec(&valid).expect("json")).expect("selection");
        assert!(matches!(
            load_warm_selection(&selection_root),
            Err(NanoWarmSelectionError::InvalidDatasetDirectoryName)
        ));

        let mut unknown = valid;
        unknown["dataset_directory_name"] = json!("session-1");
        unknown["unexpected"] = json!(true);
        fs::write(&path, serde_json::to_vec(&unknown).expect("json")).expect("selection");
        assert!(matches!(
            load_warm_selection(&selection_root),
            Err(NanoWarmSelectionError::Json(_))
        ));

        fs::write(
            &path,
            vec![b' '; usize::try_from(MAX_NANO_WARM_SELECTION_BYTES).unwrap() + 1],
        )
        .expect("oversized selection");
        assert!(matches!(
            load_warm_selection(&selection_root),
            Err(NanoWarmSelectionError::FileTooLarge { .. })
        ));
    }

    #[test]
    #[cfg(feature = "nano-agent")]
    fn invalid_next_selection_cannot_replace_the_previous_atomic_selection() {
        let directory = TestDirectory::create("selection-preserved");
        let mut owner = owner(&directory, true, 4);
        let map = SlamMap::new();
        let binding = bindings(&[&map])[0];
        owner
            .retain_latest(binding, snapshot(&map, 3, 0))
            .expect("retained map");
        let (_, content) = publish_test_warm_selection(
            &directory,
            "session-selected",
            &map,
            binding,
            3,
            0,
            br#"{"session":1}"#,
        );
        let selection_root = directory.path.join("dataset");
        let selection_path = selection_root.join(NANO_WARM_SELECTION_FILE);
        let previous = fs::read(&selection_path).expect("previous selection");
        let invalid = ParsedNanoWarmSelectionV1 {
            dataset_directory_name: "x".repeat(MAX_DATASET_DIRECTORY_NAME_BYTES + 1),
            content,
            map_epoch_id: binding.map_epoch_id(),
            map_revision: 3,
        };
        let selection_root_identity =
            require_canonical_directory(NanoMapPathRole::WarmSlamDatasetDirectory, &selection_root)
                .expect("selection root identity");
        assert!(matches!(
            publish_warm_selection(&selection_root, selection_root_identity, &invalid),
            Err(NanoWarmSelectionError::InvalidDatasetDirectoryName)
        ));
        assert_eq!(
            fs::read(&selection_path).expect("retained selection"),
            previous
        );
    }

    #[test]
    #[cfg(feature = "nano-agent")]
    fn descriptor_relative_selection_publish_cannot_follow_a_replaced_root_path() {
        let directory = TestDirectory::create("selection-root-replacement");
        let selection_root = directory.path.join("dataset");
        let retained_root_path = directory.path.join("dataset-retained");
        let attacker_root = directory.path.join("dataset-attacker");
        fs::create_dir(&selection_root).expect("selection root");
        fs::create_dir(&attacker_root).expect("attacker root");
        let root_identity =
            require_canonical_directory(NanoMapPathRole::WarmSlamDatasetDirectory, &selection_root)
                .expect("selection root identity");
        let retained_root = open_unchanged_directory_nofollow(
            NanoMapPathRole::WarmSlamDatasetDirectory,
            &selection_root,
            root_identity,
        )
        .expect("retained selection root");
        fs::rename(&selection_root, &retained_root_path).expect("move retained root");
        symlink(&attacker_root, &selection_root).expect("replace selection root by symlink");

        let destination = selection_root.join(NANO_WARM_SELECTION_FILE);
        let selection_bytes = br#"{"schema_version":1,"test":"descriptor-relative"}"#;
        publish_warm_selection_bytes_at(
            &retained_root,
            &selection_root,
            &destination,
            selection_bytes,
        )
        .expect("descriptor-relative selection publication");

        assert_eq!(
            fs::read(retained_root_path.join(NANO_WARM_SELECTION_FILE))
                .expect("retained selection bytes"),
            selection_bytes
        );
        assert!(
            !attacker_root.join(NANO_WARM_SELECTION_FILE).exists(),
            "the replacement selection root must never receive publication"
        );
    }

    #[derive(Clone)]
    struct FakeResponder {
        kind: AgentControlCommandKindV1,
        events: Rc<RefCell<Vec<&'static str>>>,
        response_fails: bool,
    }

    impl SaveMapResponder for FakeResponder {
        type Error = &'static str;

        fn command_kind(&self) -> AgentControlCommandKindV1 {
            self.kind
        }

        fn completed(self) -> Result<(), Self::Error> {
            self.events.borrow_mut().push("completed");
            if self.response_fails {
                Err("response")
            } else {
                Ok(())
            }
        }

        fn rejected(
            self,
            _code: AgentControlRejectionCodeV1,
            _retryable: bool,
        ) -> Result<(), Self::Error> {
            self.events.borrow_mut().push("rejected");
            if self.response_fails {
                Err("response")
            } else {
                Ok(())
            }
        }
    }

    fn test_receipt() -> NanoMapSaveReceipt {
        let map = SlamMap::new();
        let binding = bindings(&[&map])[0];
        NanoMapSaveReceipt {
            identity: NanoMapSnapshotIdentity {
                map_epoch_id: binding.map_epoch_id(),
                map_instance_id: binding.map_instance_id(),
                revision: 1,
            },
            destination: PathBuf::from("/var/lib/kiko/maps/current.kmap"),
            quota_verification: None,
        }
    }

    #[test]
    fn command_completion_is_strictly_after_save_and_never_sent_on_failure() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let responder = FakeResponder {
            kind: AgentControlCommandKindV1::SaveMap,
            events: Rc::clone(&events),
            response_fails: false,
        };
        let save_events = Rc::clone(&events);
        finish_claimed_save(responder, || {
            save_events.borrow_mut().push("durable_save");
            Ok(test_receipt())
        })
        .expect("save and response");
        assert_eq!(&*events.borrow(), &["durable_save", "completed"]);

        events.borrow_mut().clear();
        let responder = FakeResponder {
            kind: AgentControlCommandKindV1::SaveMap,
            events: Rc::clone(&events),
            response_fails: false,
        };
        let save_events = Rc::clone(&events);
        assert!(matches!(
            finish_claimed_save(responder, || {
                save_events.borrow_mut().push("failed_save");
                Err(NanoMapSaveError::NoSnapshot)
            }),
            Err(ClaimedSaveFlowError::SaveRejected { .. })
        ));
        assert_eq!(&*events.borrow(), &["failed_save", "rejected"]);

        events.borrow_mut().clear();
        let responder = FakeResponder {
            kind: AgentControlCommandKindV1::QueryStatus,
            events: Rc::clone(&events),
            response_fails: false,
        };
        let save_events = Rc::clone(&events);
        assert!(matches!(
            finish_claimed_save(responder, || {
                save_events.borrow_mut().push("must_not_save");
                Ok(test_receipt())
            }),
            Err(ClaimedSaveFlowError::WrongCommandRejected {
                actual: AgentControlCommandKindV1::QueryStatus
            })
        ));
        assert_eq!(&*events.borrow(), &["rejected"]);

        events.borrow_mut().clear();
        let responder = FakeResponder {
            kind: AgentControlCommandKindV1::SaveMap,
            events: Rc::clone(&events),
            response_fails: true,
        };
        let save_events = Rc::clone(&events);
        assert!(matches!(
            finish_claimed_save(responder, || {
                save_events.borrow_mut().push("durable_save");
                Ok(test_receipt())
            }),
            Err(
                ClaimedSaveFlowError::SavedButCompletionResponseFailed {
                    receipt,
                    response: "response"
                }
            ) if receipt.destination() == Path::new("/var/lib/kiko/maps/current.kmap")
        ));
        assert_eq!(&*events.borrow(), &["durable_save", "completed"]);
    }
}
