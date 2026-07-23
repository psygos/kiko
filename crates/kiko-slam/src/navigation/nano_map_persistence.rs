//! Bounded, durable map persistence for the production Nano runtime.
//!
//! The owner retains exactly one live, epoch-bound occupancy snapshot. Saves
//! are synchronous and require `&mut self`, so snapshot replacement and a
//! second save cannot overlap the durable temporary-file/sync/rename/directory
//! sync sequence.
//!
//! Warm start deliberately stops at a dataset-replay request. The current
//! policy binds only mutable pathnames, not an immutable dataset content
//! identity. Consequently a loaded occupancy artifact cannot claim
//! localization, and it cannot acquire a live map identity until the existing
//! dense persistence verifier proves exact equality with final replay output.

use std::fmt;
use std::fs::{self, File};
use std::io;
use std::os::unix::fs::DirBuilderExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use crate::dense::occupancy::OccupancyGridSnapshot;
use crate::dense::occupancy_persistence::{
    OccupancyMapEncodeError, OccupancyMapLimits, OccupancyMapLoadError, OccupancyMapSaveError,
    OccupancyReplayBindError, PersistedOccupancyMap, ReplayMatchedOccupancyMap,
    ReplayOccupancyEvidence, load_persisted_occupancy_map, save_occupancy_map_atomic,
};
use crate::map::MapInstanceId;

use super::agent_config::{NanoMapPersistenceConfig, NanoMapWarmStart};
use super::control_api::{AgentControlCommandKindV1, AgentControlRejectionCodeV1};
use super::control_socket::{AgentControlClaimedRequest, AgentControlDispatchResponseError};
use super::ingress::{CurrentMapEpochBinding, RecordedMapEpochId};
use super::nano_bootstrap::NanoBootstrapRoots;

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
}

impl NanoMapSaveReceipt {
    pub const fn identity(&self) -> NanoMapSnapshotIdentity {
        self.identity
    }

    pub fn destination(&self) -> &Path {
        &self.destination
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
    Persistence(OccupancyMapSaveError),
}

impl NanoMapSaveError {
    fn rejection(&self) -> (AgentControlRejectionCodeV1, bool) {
        match self {
            Self::NoSnapshot => (AgentControlRejectionCodeV1::MapUnavailable, true),
            Self::StaleSelection { .. } => (AgentControlRejectionCodeV1::StaleMapSelection, true),
            Self::Path(_) => (AgentControlRejectionCodeV1::PersistenceFailed, false),
            Self::Persistence(source) => (
                AgentControlRejectionCodeV1::PersistenceFailed,
                persistence_error_may_be_retryable(source),
            ),
        }
    }

    /// A rename may have published the destination even though directory sync
    /// failed. Such an error still never produces a completion response.
    pub fn destination_may_have_been_published(&self) -> bool {
        matches!(
            self,
            Self::Persistence(OccupancyMapSaveError::Io(source)) if source.published()
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
            Self::Persistence(source) => write!(formatter, "durable map save failed: {source}"),
        }
    }
}

impl std::error::Error for NanoMapSaveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Path(source) => Some(source),
            Self::Persistence(source) => Some(source),
            Self::NoSnapshot | Self::StaleSelection { .. } => None,
        }
    }
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
    /// The schema carries a path but no digest or manifest content identity.
    MissingImmutableContentIdentity,
}

#[derive(Clone, Debug)]
enum AdmittedNanoWarmStart {
    None,
    DatasetReplay {
        occupancy_snapshot_path: PathBuf,
        slam_dataset_directory_path: PathBuf,
    },
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
}

impl NanoDatasetReplayRequired {
    pub fn occupancy_snapshot_path(&self) -> &Path {
        &self.occupancy_snapshot_path
    }

    pub fn slam_dataset_directory_path(&self) -> &Path {
        &self.slam_dataset_directory_path
    }

    pub const fn dataset_content_binding_status(&self) -> NanoDatasetContentBindingStatus {
        NanoDatasetContentBindingStatus::MissingImmutableContentIdentity
    }

    pub fn persisted_snapshot(&self) -> &OccupancyGridSnapshot {
        self.persisted.snapshot()
    }

    /// Revalidate the retained directory identity, then require exact equality
    /// with final sparse/occupancy replay evidence. Success is replay matching,
    /// not current-camera relocalization.
    pub fn verify_exact_replay(
        self,
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
        let replay_matched = self
            .persisted
            .verify_replay_and_bind(replay)
            .map_err(NanoWarmStartReplayBindError::ExactReplay)?;
        Ok(NanoReplayMatchedWarmStart {
            replay_matched,
            occupancy_snapshot_path: self.occupancy_snapshot_path,
            slam_dataset_directory_path: self.slam_dataset_directory_path,
        })
    }
}

/// Occupancy proven equal to final replay output. This type deliberately does
/// not carry or expose a live localization state.
#[derive(Debug)]
pub struct NanoReplayMatchedWarmStart {
    replay_matched: ReplayMatchedOccupancyMap,
    occupancy_snapshot_path: PathBuf,
    slam_dataset_directory_path: PathBuf,
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
        NanoDatasetContentBindingStatus::MissingImmutableContentIdentity
    }

    pub fn into_replay_matched_map(self) -> ReplayMatchedOccupancyMap {
        self.replay_matched
    }
}

#[derive(Debug)]
pub enum NanoMapWarmStartLoadError {
    Path(NanoMapPersistencePathError),
    Occupancy(OccupancyMapLoadError),
}

impl fmt::Display for NanoMapWarmStartLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Path(source) => write!(formatter, "warm-start path is not admitted: {source}"),
            Self::Occupancy(source) => write!(formatter, "cannot load warm occupancy: {source}"),
        }
    }
}

impl std::error::Error for NanoMapWarmStartLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Path(source) => Some(source),
            Self::Occupancy(source) => Some(source),
        }
    }
}

#[derive(Debug)]
pub enum NanoWarmStartReplayBindError {
    Path(NanoMapPersistencePathError),
    ExactReplay(OccupancyReplayBindError),
}

impl fmt::Display for NanoWarmStartReplayBindError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Path(source) => write!(formatter, "replay dataset path changed: {source}"),
            Self::ExactReplay(source) => write!(formatter, "occupancy replay mismatch: {source}"),
        }
    }
}

impl std::error::Error for NanoWarmStartReplayBindError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Path(source) => Some(source),
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

    fn save_with_selection(
        &mut self,
        selected: Option<NanoMapSnapshotIdentity>,
    ) -> Result<NanoMapSaveReceipt, NanoMapSaveError> {
        self.save_with(selected, |path, snapshot| {
            save_occupancy_map_atomic(path, snapshot)
        })
    }

    fn save_with<F>(
        &mut self,
        selected: Option<NanoMapSnapshotIdentity>,
        save: F,
    ) -> Result<NanoMapSaveReceipt, NanoMapSaveError>
    where
        F: FnOnce(&Path, &OccupancyGridSnapshot) -> Result<(), OccupancyMapSaveError>,
    {
        let retained = self.latest.as_ref().ok_or(NanoMapSaveError::NoSnapshot)?;
        if let Some(requested) = selected
            && requested != retained.identity
        {
            return Err(NanoMapSaveError::StaleSelection {
                requested,
                retained: retained.identity,
            });
        }
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

    /// Load the configured warm artifact once. A dataset path is admitted and
    /// retained, but it is explicitly not treated as content identity.
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
            "schema_version": 1,
            "control": {
                "socket_path": "/tmp/kiko-agent/control.sock",
                "read_timeout_ms": 100,
                "write_timeout_ms": 100,
                "runtime_response_timeout_ms": 500,
                "runtime_queue_capacity": 8
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
        super::super::NanoAgentPolicyConfigV1::parse_json(
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
