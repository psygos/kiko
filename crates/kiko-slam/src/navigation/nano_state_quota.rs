//! Descriptor-relative quota admission for the production Nano map snapshot.
//!
//! The launch document parses the exact map location, encoded-byte ceiling,
//! and post-save free-space floor into [`NanoLaunchStorage`]. This module
//! turns that structural policy into a process-lifetime owner which:
//!
//! - opens the absolute state root one component at a time with `O_NOFOLLOW`;
//! - creates only the configured map parent with descriptor-relative
//!   `mkdirat`/`openat`;
//! - rejects symbolic links, special files, hard-linked regular files, and
//!   cross-filesystem components on the exact configured map path;
//! - queries free space from the admitted root descriptor without a fallback;
//! - serializes exact map replacement through a borrowing reservation; and
//! - verifies the exact destination length, map ceiling, and post-save
//!   free-space floor again after commit.
//!
//! The navigation dataset has a separate launch-bound streaming quota. This
//! map-only owner deliberately neither scans nor double-counts that dataset,
//! startup evidence, unrelated files, or total state-root usage.
//!
//! The reservation retains the exact state-root and map-parent descriptors used
//! for admission. The production writer publishes relative to that retained
//! parent and post-write verification observes the same objects before checking
//! that the parent is still reachable at the configured path. A concurrent
//! writer can still consume free space between observations. Dataset capture
//! retains terminal headroom for this map path, while post-write verification
//! detects later races but cannot undo an already published map.

use std::ffi::OsStr;
use std::fmt;
use std::os::fd::{AsFd, BorrowedFd};
use std::path::{Component, Path, PathBuf};

use rustix::fd::OwnedFd;
use rustix::fs::{
    AtFlags, FileType, Mode, OFlags, Stat, fstat, fstatvfs, fsync, mkdirat, open, openat, statat,
};
use rustix::io::{Errno, dup};

use super::{NanoBootstrapRoots, NanoLaunchStorage};

const PRIVATE_DIRECTORY_MODE: Mode = Mode::from_raw_mode(0o700);

fn directory_open_flags() -> OFlags {
    OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK
}

fn file_open_flags() -> OFlags {
    OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoStatePathRole {
    StateRoot,
    MapParent,
    MapSnapshot,
    WriteDestination,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoStateFilesystemOperation {
    OpenAbsoluteComponent { component_index: usize },
    DuplicateDirectory,
    Inspect,
    CreateDirectory,
    OpenDirectory,
    OpenFile,
    SynchronizeDirectory,
    QueryFreeSpace,
}

/// Exact descriptor-relative filesystem failure.
#[derive(Debug)]
pub enum NanoStateFilesystemError {
    Io {
        operation: NanoStateFilesystemOperation,
        role: NanoStatePathRole,
        path: PathBuf,
        source: Errno,
    },
    NotDirectory {
        role: NanoStatePathRole,
        path: PathBuf,
        observed: FileType,
    },
    NotRegularFile {
        role: NanoStatePathRole,
        path: PathBuf,
        observed: FileType,
    },
    CrossFilesystem {
        role: NanoStatePathRole,
        path: PathBuf,
        state_root_device: i128,
        observed_device: i128,
    },
    ObjectChanged {
        role: NanoStatePathRole,
        path: PathBuf,
        inspected_device: i128,
        inspected_inode: u128,
        opened_device: i128,
        opened_inode: u128,
    },
    StateRootChanged {
        path: PathBuf,
        admitted_device: i128,
        admitted_inode: u128,
        observed_device: i128,
        observed_inode: u128,
    },
    HardLinkedRegularFile {
        role: NanoStatePathRole,
        path: PathBuf,
        link_count: u128,
    },
    NegativeFileSize {
        role: NanoStatePathRole,
        path: PathBuf,
        bytes: i128,
    },
    ZeroFilesystemFragmentSize {
        state_root: PathBuf,
    },
    FreeSpaceOverflow {
        state_root: PathBuf,
        available_fragments: u64,
        fragment_size_bytes: u64,
    },
}

impl fmt::Display for NanoStateFilesystemError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Nano state filesystem error: {self:?}")
    }
}

impl std::error::Error for NanoStateFilesystemError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FilesystemIdentity {
    device: i128,
    inode: u128,
}

impl FilesystemIdentity {
    fn from_stat(stat: &Stat) -> Self {
        Self {
            device: i128::from(stat.st_dev),
            inode: u128::from(stat.st_ino),
        }
    }
}

/// One exact map-path and filesystem observation used to admit or verify a
/// map replacement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NanoStateQuotaSnapshot {
    map_snapshot_bytes: u64,
    available_bytes: u64,
    filesystem_fragment_size_bytes: u64,
}

impl NanoStateQuotaSnapshot {
    pub const fn map_snapshot_bytes(self) -> u64 {
        self.map_snapshot_bytes
    }

    pub const fn available_bytes(self) -> u64 {
        self.available_bytes
    }

    pub const fn filesystem_fragment_size_bytes(self) -> u64 {
        self.filesystem_fragment_size_bytes
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NanoStateQuotaLimits {
    maximum_map_snapshot_bytes: u64,
    minimum_free_bytes_after_map_save: u64,
}

impl NanoStateQuotaLimits {
    fn from_storage(storage: &NanoLaunchStorage) -> Self {
        Self {
            maximum_map_snapshot_bytes: storage.maximum_map_snapshot_bytes(),
            minimum_free_bytes_after_map_save: storage.minimum_free_bytes_after_map_save(),
        }
    }
}

/// Existing or projected state violates an exact parsed quota.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoStateQuotaViolation {
    MapSnapshotMaximumExceeded {
        actual_bytes: u64,
        maximum_bytes: u64,
    },
    PostSaveMinimumFreeNotMet {
        available_bytes: u64,
        minimum_free_bytes_after_map_save: u64,
    },
}

impl fmt::Display for NanoStateQuotaViolation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Nano map-storage quota violation: {self:?}")
    }
}

impl std::error::Error for NanoStateQuotaViolation {}

#[derive(Debug)]
pub enum NanoStateQuotaAdmissionError {
    Filesystem(NanoStateFilesystemError),
    ExistingState(NanoStateQuotaViolation),
}

impl fmt::Display for NanoStateQuotaAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Nano map-storage admission failed: {self:?}")
    }
}

impl std::error::Error for NanoStateQuotaAdmissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Filesystem(source) => Some(source),
            Self::ExistingState(source) => Some(source),
        }
    }
}

impl From<NanoStateFilesystemError> for NanoStateQuotaAdmissionError {
    fn from(source: NanoStateFilesystemError) -> Self {
        Self::Filesystem(source)
    }
}

#[derive(Debug)]
pub enum NanoStateQuotaReserveError {
    Filesystem(NanoStateFilesystemError),
    ExistingState(NanoStateQuotaViolation),
    MapDestinationMismatch {
        configured: PathBuf,
        requested: PathBuf,
    },
    PlannedMapSnapshotMaximumExceeded {
        planned_bytes: u64,
        maximum_bytes: u64,
    },
    InsufficientTransientFreeSpace {
        available_bytes: u64,
        planned_file_bytes: u64,
        transient_allocation_bytes: u64,
        filesystem_fragment_size_bytes: u64,
        minimum_free_bytes_after_map_save: u64,
        required_available_bytes: u64,
    },
    RequiredAvailableArithmeticOverflow,
}

impl fmt::Display for NanoStateQuotaReserveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Nano map replacement reservation failed: {self:?}"
        )
    }
}

impl std::error::Error for NanoStateQuotaReserveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Filesystem(source) => Some(source),
            Self::ExistingState(source) => Some(source),
            _ => None,
        }
    }
}

impl From<NanoStateFilesystemError> for NanoStateQuotaReserveError {
    fn from(source: NanoStateFilesystemError) -> Self {
        Self::Filesystem(source)
    }
}

#[derive(Debug)]
pub enum NanoStateQuotaCommitError {
    Filesystem(NanoStateFilesystemError),
    DestinationMissing {
        path: PathBuf,
    },
    LengthMismatch {
        path: PathBuf,
        planned_bytes: u64,
        actual_bytes: u64,
    },
    PostWriteViolation(NanoStateQuotaViolation),
}

impl fmt::Display for NanoStateQuotaCommitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Nano map replacement verification failed: {self:?}"
        )
    }
}

impl std::error::Error for NanoStateQuotaCommitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Filesystem(source) => Some(source),
            Self::PostWriteViolation(source) => Some(source),
            _ => None,
        }
    }
}

impl From<NanoStateFilesystemError> for NanoStateQuotaCommitError {
    fn from(source: NanoStateFilesystemError) -> Self {
        Self::Filesystem(source)
    }
}

/// Verified result of one exact write.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoStateQuotaWriteReceipt {
    destination: PathBuf,
    exact_bytes: u64,
    before: NanoStateQuotaSnapshot,
    after: NanoStateQuotaSnapshot,
}

impl NanoStateQuotaWriteReceipt {
    pub fn destination(&self) -> &Path {
        &self.destination
    }

    pub const fn exact_bytes(&self) -> u64 {
        self.exact_bytes
    }

    pub const fn before(&self) -> NanoStateQuotaSnapshot {
        self.before
    }

    pub const fn after(&self) -> NanoStateQuotaSnapshot {
        self.after
    }
}

/// Sole admitted production map-replacement quota owner.
pub struct NanoStateQuotaOwner {
    state_root: PathBuf,
    state_root_identity: FilesystemIdentity,
    map_relative_path: PathBuf,
    map_absolute_path: PathBuf,
    limits: NanoStateQuotaLimits,
    last_snapshot: NanoStateQuotaSnapshot,
}

impl fmt::Debug for NanoStateQuotaOwner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NanoStateQuotaOwner")
            .field("state_root", &self.state_root)
            .field("map_absolute_path", &self.map_absolute_path)
            .field("limits", &self.limits)
            .field("last_snapshot", &self.last_snapshot)
            .finish()
    }
}

impl NanoStateQuotaOwner {
    /// Admit the exact existing map path and create only its parsed parent.
    pub fn admit(
        roots: &NanoBootstrapRoots,
        storage: &NanoLaunchStorage,
    ) -> Result<Self, NanoStateQuotaAdmissionError> {
        let state_root = roots.state_root().to_path_buf();
        let state_root_fd =
            open_absolute_directory_nofollow(NanoStatePathRole::StateRoot, &state_root)?;
        let root_stat = fstat(&state_root_fd).map_err(|source| NanoStateFilesystemError::Io {
            operation: NanoStateFilesystemOperation::Inspect,
            role: NanoStatePathRole::StateRoot,
            path: state_root.clone(),
            source,
        })?;
        require_file_type(
            NanoStatePathRole::StateRoot,
            &state_root,
            &root_stat,
            FileType::Directory,
        )?;
        let state_root_identity = FilesystemIdentity::from_stat(&root_stat);

        let map_relative_path = storage.map_snapshot().as_path().to_path_buf();
        let map_absolute_path = state_root.join(&map_relative_path);
        let map_parent = map_relative_path.parent().unwrap_or_else(|| Path::new(""));
        ensure_directory_beneath(
            &state_root_fd,
            &state_root,
            state_root_identity.device,
            NanoStatePathRole::MapParent,
            map_parent,
        )?;

        let limits = NanoStateQuotaLimits::from_storage(storage);
        let mut owner = Self {
            state_root,
            state_root_identity,
            map_relative_path,
            map_absolute_path,
            limits,
            last_snapshot: NanoStateQuotaSnapshot {
                map_snapshot_bytes: 0,
                available_bytes: 0,
                filesystem_fragment_size_bytes: 0,
            },
        };
        let snapshot = owner.observe()?;
        validate_map_size(snapshot, limits).map_err(NanoStateQuotaAdmissionError::ExistingState)?;
        owner.last_snapshot = snapshot;
        Ok(owner)
    }

    pub fn map_snapshot_path(&self) -> &Path {
        &self.map_absolute_path
    }

    pub const fn last_snapshot(&self) -> NanoStateQuotaSnapshot {
        self.last_snapshot
    }

    /// Reserve an atomic replacement of the one configured map artifact.
    ///
    /// `destination` must exactly equal [`Self::map_snapshot_path`]. The full
    /// planned file length is retained as transient headroom because atomic
    /// replacement temporarily coexists with the old map.
    pub fn reserve_map_replacement(
        &mut self,
        destination: &Path,
        exact_final_bytes: u64,
    ) -> Result<NanoStateQuotaReservation<'_>, NanoStateQuotaReserveError> {
        if destination != self.map_absolute_path {
            return Err(NanoStateQuotaReserveError::MapDestinationMismatch {
                configured: self.map_absolute_path.clone(),
                requested: destination.to_path_buf(),
            });
        }
        self.reserve_map(exact_final_bytes)
    }

    fn reserve_map(
        &mut self,
        exact_final_bytes: u64,
    ) -> Result<NanoStateQuotaReservation<'_>, NanoStateQuotaReserveError> {
        let state_root_fd = self.open_current_root()?;
        let map_parent_fd = self.open_map_parent_from_root(&state_root_fd)?;
        let before = self.observe_from_descriptors(&state_root_fd, &map_parent_fd)?;
        validate_map_size(before, self.limits)
            .map_err(NanoStateQuotaReserveError::ExistingState)?;
        project_write(before, self.limits, exact_final_bytes)?;

        self.last_snapshot = before;
        Ok(NanoStateQuotaReservation {
            owner: self,
            exact_final_bytes,
            before,
            state_root_fd,
            map_parent_fd,
        })
    }

    fn open_current_root(&self) -> Result<OwnedFd, NanoStateFilesystemError> {
        let root_fd =
            open_absolute_directory_nofollow(NanoStatePathRole::StateRoot, &self.state_root)?;
        let stat = fstat(&root_fd).map_err(|source| NanoStateFilesystemError::Io {
            operation: NanoStateFilesystemOperation::Inspect,
            role: NanoStatePathRole::StateRoot,
            path: self.state_root.clone(),
            source,
        })?;
        require_file_type(
            NanoStatePathRole::StateRoot,
            &self.state_root,
            &stat,
            FileType::Directory,
        )?;
        let observed = FilesystemIdentity::from_stat(&stat);
        if observed != self.state_root_identity {
            return Err(NanoStateFilesystemError::StateRootChanged {
                path: self.state_root.clone(),
                admitted_device: self.state_root_identity.device,
                admitted_inode: self.state_root_identity.inode,
                observed_device: observed.device,
                observed_inode: observed.inode,
            });
        }
        Ok(root_fd)
    }

    fn observe(&self) -> Result<NanoStateQuotaSnapshot, NanoStateFilesystemError> {
        let root_fd = self.open_current_root()?;
        self.observe_from_root(&root_fd)
    }

    fn observe_from_root(
        &self,
        root_fd: &OwnedFd,
    ) -> Result<NanoStateQuotaSnapshot, NanoStateFilesystemError> {
        let map_parent_fd = self.open_map_parent_from_root(root_fd)?;
        self.observe_from_descriptors(root_fd, &map_parent_fd)
    }

    fn open_map_parent_from_root(
        &self,
        root_fd: &OwnedFd,
    ) -> Result<OwnedFd, NanoStateFilesystemError> {
        let parent = self
            .map_relative_path
            .parent()
            .unwrap_or_else(|| Path::new(""));
        open_directory_beneath(
            root_fd,
            &self.state_root,
            self.state_root_identity.device,
            NanoStatePathRole::MapParent,
            parent,
        )
    }

    fn map_file_name(&self) -> Result<&OsStr, NanoStateFilesystemError> {
        self.map_relative_path
            .file_name()
            .ok_or_else(|| NanoStateFilesystemError::NotRegularFile {
                role: NanoStatePathRole::MapSnapshot,
                path: self.map_absolute_path.clone(),
                observed: FileType::Directory,
            })
    }

    fn observe_from_descriptors(
        &self,
        root_fd: &OwnedFd,
        map_parent_fd: &OwnedFd,
    ) -> Result<NanoStateQuotaSnapshot, NanoStateFilesystemError> {
        let map_snapshot_bytes = inspect_optional_regular_file_in_parent(
            map_parent_fd,
            &self.state_root,
            self.state_root_identity.device,
            NanoStatePathRole::MapSnapshot,
            &self.map_relative_path,
            self.map_file_name()?,
        )?
        .unwrap_or(0);
        let filesystem = fstatvfs(root_fd).map_err(|source| NanoStateFilesystemError::Io {
            operation: NanoStateFilesystemOperation::QueryFreeSpace,
            role: NanoStatePathRole::StateRoot,
            path: self.state_root.clone(),
            source,
        })?;
        if filesystem.f_frsize == 0 {
            return Err(NanoStateFilesystemError::ZeroFilesystemFragmentSize {
                state_root: self.state_root.clone(),
            });
        }
        let available_bytes = filesystem
            .f_bavail
            .checked_mul(filesystem.f_frsize)
            .ok_or_else(|| NanoStateFilesystemError::FreeSpaceOverflow {
                state_root: self.state_root.clone(),
                available_fragments: filesystem.f_bavail,
                fragment_size_bytes: filesystem.f_frsize,
            })?;
        Ok(NanoStateQuotaSnapshot {
            map_snapshot_bytes,
            available_bytes,
            filesystem_fragment_size_bytes: filesystem.f_frsize,
        })
    }

    fn require_retained_parent_is_current(
        &self,
        retained_parent: &OwnedFd,
    ) -> Result<(), NanoStateFilesystemError> {
        let current_root = self.open_current_root()?;
        let current_parent = self.open_map_parent_from_root(&current_root)?;
        let retained = fstat(retained_parent).map_err(|source| NanoStateFilesystemError::Io {
            operation: NanoStateFilesystemOperation::Inspect,
            role: NanoStatePathRole::MapParent,
            path: self
                .map_absolute_path
                .parent()
                .unwrap_or(&self.state_root)
                .to_path_buf(),
            source,
        })?;
        let current = fstat(&current_parent).map_err(|source| NanoStateFilesystemError::Io {
            operation: NanoStateFilesystemOperation::Inspect,
            role: NanoStatePathRole::MapParent,
            path: self
                .map_absolute_path
                .parent()
                .unwrap_or(&self.state_root)
                .to_path_buf(),
            source,
        })?;
        require_same_object(
            NanoStatePathRole::MapParent,
            self.map_absolute_path.parent().unwrap_or(&self.state_root),
            &retained,
            &current,
        )
    }
}

/// A borrowing reservation prevents a second quota admission from overlapping
/// the planned write. Dropping it without verification grants no receipt; the
/// next reservation re-observes the exact map and filesystem headroom.
#[must_use = "the authoritative writer must verify the exact committed file"]
pub struct NanoStateQuotaReservation<'owner> {
    owner: &'owner mut NanoStateQuotaOwner,
    exact_final_bytes: u64,
    before: NanoStateQuotaSnapshot,
    state_root_fd: OwnedFd,
    map_parent_fd: OwnedFd,
}

impl fmt::Debug for NanoStateQuotaReservation<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NanoStateQuotaReservation")
            .field("destination", &self.owner.map_absolute_path)
            .field("exact_final_bytes", &self.exact_final_bytes)
            .field("before", &self.before)
            .finish()
    }
}

impl NanoStateQuotaReservation<'_> {
    /// The exact map parent admitted for this reservation. The authoritative
    /// writer must create, rename, and synchronize relative to this descriptor.
    pub(crate) fn publication_parent(&self) -> BorrowedFd<'_> {
        self.map_parent_fd.as_fd()
    }

    /// Inspect the destination through the same retained descriptors used for
    /// admission and publication, then verify that the parent is still the
    /// configured live path.
    pub fn verify_committed(self) -> Result<NanoStateQuotaWriteReceipt, NanoStateQuotaCommitError> {
        let actual = inspect_optional_regular_file_in_parent(
            &self.map_parent_fd,
            &self.owner.state_root,
            self.owner.state_root_identity.device,
            NanoStatePathRole::WriteDestination,
            &self.owner.map_relative_path,
            self.owner.map_file_name()?,
        )?
        .ok_or_else(|| NanoStateQuotaCommitError::DestinationMissing {
            path: self.owner.map_absolute_path.clone(),
        })?;
        if actual != self.exact_final_bytes {
            return Err(NanoStateQuotaCommitError::LengthMismatch {
                path: self.owner.map_absolute_path.clone(),
                planned_bytes: self.exact_final_bytes,
                actual_bytes: actual,
            });
        }
        let after = self
            .owner
            .observe_from_descriptors(&self.state_root_fd, &self.map_parent_fd)?;
        validate_committed_map(after, self.owner.limits)
            .map_err(NanoStateQuotaCommitError::PostWriteViolation)?;
        self.owner
            .require_retained_parent_is_current(&self.map_parent_fd)?;
        self.owner.last_snapshot = after;
        Ok(NanoStateQuotaWriteReceipt {
            destination: self.owner.map_absolute_path.clone(),
            exact_bytes: actual,
            before: self.before,
            after,
        })
    }
}

fn validate_map_size(
    snapshot: NanoStateQuotaSnapshot,
    limits: NanoStateQuotaLimits,
) -> Result<(), NanoStateQuotaViolation> {
    let actual_bytes = snapshot.map_snapshot_bytes;
    let maximum_bytes = limits.maximum_map_snapshot_bytes;
    if actual_bytes > maximum_bytes {
        return Err(NanoStateQuotaViolation::MapSnapshotMaximumExceeded {
            actual_bytes,
            maximum_bytes,
        });
    }
    Ok(())
}

fn validate_committed_map(
    snapshot: NanoStateQuotaSnapshot,
    limits: NanoStateQuotaLimits,
) -> Result<(), NanoStateQuotaViolation> {
    validate_map_size(snapshot, limits)?;
    if snapshot.available_bytes < limits.minimum_free_bytes_after_map_save {
        return Err(NanoStateQuotaViolation::PostSaveMinimumFreeNotMet {
            available_bytes: snapshot.available_bytes,
            minimum_free_bytes_after_map_save: limits.minimum_free_bytes_after_map_save,
        });
    }
    Ok(())
}

fn project_write(
    before: NanoStateQuotaSnapshot,
    limits: NanoStateQuotaLimits,
    planned_bytes: u64,
) -> Result<(), NanoStateQuotaReserveError> {
    let maximum_map_snapshot_bytes = limits.maximum_map_snapshot_bytes;
    if planned_bytes > maximum_map_snapshot_bytes {
        return Err(
            NanoStateQuotaReserveError::PlannedMapSnapshotMaximumExceeded {
                planned_bytes,
                maximum_bytes: maximum_map_snapshot_bytes,
            },
        );
    }

    let transient_allocation_bytes =
        round_up_to_fragment(planned_bytes, before.filesystem_fragment_size_bytes)
            .ok_or(NanoStateQuotaReserveError::RequiredAvailableArithmeticOverflow)?;
    let required_available = limits
        .minimum_free_bytes_after_map_save
        .checked_add(transient_allocation_bytes)
        .ok_or(NanoStateQuotaReserveError::RequiredAvailableArithmeticOverflow)?;
    if before.available_bytes < required_available {
        return Err(NanoStateQuotaReserveError::InsufficientTransientFreeSpace {
            available_bytes: before.available_bytes,
            planned_file_bytes: planned_bytes,
            transient_allocation_bytes,
            filesystem_fragment_size_bytes: before.filesystem_fragment_size_bytes,
            minimum_free_bytes_after_map_save: limits.minimum_free_bytes_after_map_save,
            required_available_bytes: required_available,
        });
    }
    Ok(())
}

fn round_up_to_fragment(bytes: u64, fragment_size_bytes: u64) -> Option<u64> {
    if bytes == 0 {
        return Some(0);
    }
    bytes
        .checked_sub(1)?
        .checked_div(fragment_size_bytes)?
        .checked_add(1)?
        .checked_mul(fragment_size_bytes)
}

fn open_absolute_directory_nofollow(
    role: NanoStatePathRole,
    path: &Path,
) -> Result<OwnedFd, NanoStateFilesystemError> {
    let mut current = open("/", directory_open_flags(), Mode::empty()).map_err(|source| {
        NanoStateFilesystemError::Io {
            operation: NanoStateFilesystemOperation::OpenAbsoluteComponent { component_index: 0 },
            role,
            path: PathBuf::from("/"),
            source,
        }
    })?;
    let mut opened_path = PathBuf::from("/");
    for (index, component) in path
        .components()
        .filter_map(|component| match component {
            Component::Normal(name) => Some(name),
            _ => None,
        })
        .enumerate()
    {
        opened_path.push(component);
        current = openat(&current, component, directory_open_flags(), Mode::empty()).map_err(
            |source| NanoStateFilesystemError::Io {
                operation: NanoStateFilesystemOperation::OpenAbsoluteComponent {
                    component_index: index + 1,
                },
                role,
                path: opened_path.clone(),
                source,
            },
        )?;
    }
    Ok(current)
}

fn ensure_directory_beneath(
    root: &OwnedFd,
    state_root: &Path,
    state_root_device: i128,
    role: NanoStatePathRole,
    relative: &Path,
) -> Result<OwnedFd, NanoStateFilesystemError> {
    let mut current = dup(root).map_err(|source| NanoStateFilesystemError::Io {
        operation: NanoStateFilesystemOperation::DuplicateDirectory,
        role,
        path: state_root.to_path_buf(),
        source,
    })?;
    let mut relative_opened = PathBuf::new();
    for component in relative
        .components()
        .filter_map(|component| match component {
            Component::Normal(name) => Some(name),
            _ => None,
        })
    {
        relative_opened.push(component);
        let absolute = state_root.join(&relative_opened);
        let created = match mkdirat(&current, component, PRIVATE_DIRECTORY_MODE) {
            Ok(()) => true,
            Err(Errno::EXIST) => false,
            Err(source) => {
                return Err(NanoStateFilesystemError::Io {
                    operation: NanoStateFilesystemOperation::CreateDirectory,
                    role,
                    path: absolute,
                    source,
                });
            }
        };
        let child = openat(&current, component, directory_open_flags(), Mode::empty()).map_err(
            |source| NanoStateFilesystemError::Io {
                operation: NanoStateFilesystemOperation::OpenDirectory,
                role,
                path: absolute.clone(),
                source,
            },
        )?;
        let stat = fstat(&child).map_err(|source| NanoStateFilesystemError::Io {
            operation: NanoStateFilesystemOperation::Inspect,
            role,
            path: absolute.clone(),
            source,
        })?;
        require_file_type(role, &absolute, &stat, FileType::Directory)?;
        require_same_filesystem(role, &absolute, state_root_device, &stat)?;
        if created {
            fsync(&child).map_err(|source| NanoStateFilesystemError::Io {
                operation: NanoStateFilesystemOperation::SynchronizeDirectory,
                role,
                path: absolute.clone(),
                source,
            })?;
            fsync(&current).map_err(|source| NanoStateFilesystemError::Io {
                operation: NanoStateFilesystemOperation::SynchronizeDirectory,
                role,
                path: absolute
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| state_root.to_path_buf()),
                source,
            })?;
        }
        current = child;
    }
    Ok(current)
}

fn inspect_optional_regular_file_in_parent(
    parent_fd: &OwnedFd,
    state_root: &Path,
    state_root_device: i128,
    role: NanoStatePathRole,
    relative: &Path,
    file_name: &OsStr,
) -> Result<Option<u64>, NanoStateFilesystemError> {
    let absolute = state_root.join(relative);
    let inspected = match statat(parent_fd, file_name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(stat) => stat,
        Err(Errno::NOENT) => return Ok(None),
        Err(source) => {
            return Err(NanoStateFilesystemError::Io {
                operation: NanoStateFilesystemOperation::Inspect,
                role,
                path: absolute,
                source,
            });
        }
    };
    require_file_type(role, &absolute, &inspected, FileType::RegularFile)?;
    let file =
        openat(parent_fd, file_name, file_open_flags(), Mode::empty()).map_err(|source| {
            NanoStateFilesystemError::Io {
                operation: NanoStateFilesystemOperation::OpenFile,
                role,
                path: absolute.clone(),
                source,
            }
        })?;
    let opened = fstat(&file).map_err(|source| NanoStateFilesystemError::Io {
        operation: NanoStateFilesystemOperation::Inspect,
        role,
        path: absolute.clone(),
        source,
    })?;
    require_same_object(role, &absolute, &inspected, &opened)?;
    require_same_filesystem(role, &absolute, state_root_device, &opened)?;
    require_single_link(role, &absolute, &opened)?;
    regular_file_size(role, &absolute, &opened).map(Some)
}

fn open_directory_beneath(
    root: &OwnedFd,
    state_root: &Path,
    state_root_device: i128,
    role: NanoStatePathRole,
    relative: &Path,
) -> Result<OwnedFd, NanoStateFilesystemError> {
    let mut current = dup(root).map_err(|source| NanoStateFilesystemError::Io {
        operation: NanoStateFilesystemOperation::DuplicateDirectory,
        role,
        path: state_root.to_path_buf(),
        source,
    })?;
    let mut relative_opened = PathBuf::new();
    for component in relative
        .components()
        .filter_map(|component| match component {
            Component::Normal(name) => Some(name),
            _ => None,
        })
    {
        relative_opened.push(component);
        let absolute = state_root.join(&relative_opened);
        current = openat(&current, component, directory_open_flags(), Mode::empty()).map_err(
            |source| NanoStateFilesystemError::Io {
                operation: NanoStateFilesystemOperation::OpenDirectory,
                role,
                path: absolute.clone(),
                source,
            },
        )?;
        let stat = fstat(&current).map_err(|source| NanoStateFilesystemError::Io {
            operation: NanoStateFilesystemOperation::Inspect,
            role,
            path: absolute.clone(),
            source,
        })?;
        require_file_type(role, &absolute, &stat, FileType::Directory)?;
        require_same_filesystem(role, &absolute, state_root_device, &stat)?;
    }
    Ok(current)
}

fn require_file_type(
    role: NanoStatePathRole,
    path: &Path,
    stat: &Stat,
    expected: FileType,
) -> Result<(), NanoStateFilesystemError> {
    let observed = FileType::from_raw_mode(stat.st_mode);
    if observed == expected {
        Ok(())
    } else if expected == FileType::Directory {
        Err(NanoStateFilesystemError::NotDirectory {
            role,
            path: path.to_path_buf(),
            observed,
        })
    } else {
        Err(NanoStateFilesystemError::NotRegularFile {
            role,
            path: path.to_path_buf(),
            observed,
        })
    }
}

fn require_same_filesystem(
    role: NanoStatePathRole,
    path: &Path,
    state_root_device: i128,
    stat: &Stat,
) -> Result<(), NanoStateFilesystemError> {
    let observed_device = i128::from(stat.st_dev);
    if observed_device == state_root_device {
        Ok(())
    } else {
        Err(NanoStateFilesystemError::CrossFilesystem {
            role,
            path: path.to_path_buf(),
            state_root_device,
            observed_device,
        })
    }
}

fn require_same_object(
    role: NanoStatePathRole,
    path: &Path,
    inspected: &Stat,
    opened: &Stat,
) -> Result<(), NanoStateFilesystemError> {
    let inspected = FilesystemIdentity::from_stat(inspected);
    let opened = FilesystemIdentity::from_stat(opened);
    if inspected == opened {
        Ok(())
    } else {
        Err(NanoStateFilesystemError::ObjectChanged {
            role,
            path: path.to_path_buf(),
            inspected_device: inspected.device,
            inspected_inode: inspected.inode,
            opened_device: opened.device,
            opened_inode: opened.inode,
        })
    }
}

fn require_single_link(
    role: NanoStatePathRole,
    path: &Path,
    stat: &Stat,
) -> Result<(), NanoStateFilesystemError> {
    let link_count = u128::from(stat.st_nlink);
    if link_count == 1 {
        Ok(())
    } else {
        Err(NanoStateFilesystemError::HardLinkedRegularFile {
            role,
            path: path.to_path_buf(),
            link_count,
        })
    }
}

fn regular_file_size(
    role: NanoStatePathRole,
    path: &Path,
    stat: &Stat,
) -> Result<u64, NanoStateFilesystemError> {
    u64::try_from(stat.st_size).map_err(|_| NanoStateFilesystemError::NegativeFileSize {
        role,
        path: path.to_path_buf(),
        bytes: i128::from(stat.st_size),
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Write;
    use std::os::unix::fs::{PermissionsExt, symlink};
    use std::sync::atomic::{AtomicU64, Ordering};

    use serde_json::{Value, json};

    use super::*;
    use crate::HostMonotonicTimestamp;
    use crate::dense::occupancy::{OccupancyCell, OccupancyGridGeometry, OccupancyGridSnapshot};
    use crate::dense::occupancy_persistence::OccupancyMapLimits;
    use crate::map::SlamMap;
    use crate::navigation::{
        NanoAgentLaunchV2, NanoAgentPolicyConfigV3, NanoMapPersistenceOwner, NavigationClockEpoch,
        NavigationMapEpochCoordinator,
    };

    static NEXT_TEST_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory {
        path: PathBuf,
    }

    impl TestDirectory {
        fn create(label: &str) -> Self {
            let base = fs::canonicalize(std::env::temp_dir()).expect("canonical temp root");
            for _ in 0..1_000 {
                let serial = NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed);
                let path = base.join(format!(
                    "kiko-nano-quota-{label}-{}-{serial}",
                    std::process::id()
                ));
                match fs::create_dir(&path) {
                    Ok(()) => return Self { path },
                    Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => {}
                    Err(source) => panic!("create test state root: {source}"),
                }
            }
            panic!("could not allocate unique quota test directory")
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn digest(seed: u8) -> String {
        format!("{seed:02x}").repeat(32)
    }

    fn asset(path: &str, seed: u8) -> Value {
        json!({
            "relative_path": path,
            "maximum_bytes": 1024,
            "sha256_hex": digest(seed)
        })
    }

    fn storage(
        maximum_map_snapshot_bytes: u64,
        minimum_free_bytes_after_map_save: u64,
    ) -> super::super::NanoLaunchStorage {
        let rounded_map_bytes = maximum_map_snapshot_bytes
            .checked_add(4_095)
            .expect("fixture map size rounds without overflow")
            / 4_096
            * 4_096;
        let terminal_reserve_bytes = rounded_map_bytes
            .checked_add(64 * 1_024 * 1_024 + 4_096)
            .expect("fixture terminal reserve fits");
        let launch = json!({
            "schema_version": 2,
            "agent_policy_asset": asset("config/agent-policy.json", 1),
            "navigation_shadow_config_asset": asset("config/navigation.json", 2),
            "physical_actuation_config_asset": asset("config/actuation.json", 3),
            "controller_server": {
                "contract_asset": asset("config/controller.json", 4),
                "command_udp_endpoint": "127.0.0.1:8080"
            },
            "plant_artifact": {
                "artifact_id": "plant-v1",
                "asset": asset("artifacts/plant.json", 5)
            },
            "calibration_artifact": {
                "artifact_id": "calibration-v1",
                "asset": asset("artifacts/calibration.json", 9)
            },
            "oak": {
                "selector_source": "exact_inventory_oak_mxid",
                "maximum_usb_speed": "SUPER",
                "minimum_usb_speed": "SUPER",
                "rgb": {"width_px": 640, "height_px": 480, "fps": 30},
                "rectified_stereo": {
                    "width_px": 640,
                    "height_px": 400,
                    "fps": 30,
                    "rectified": true
                },
                "depth": {
                    "width_px": 640,
                    "height_px": 400,
                    "fps": 30,
                    "alignment": "rectified_left"
                },
                "imu": {"rate_hz": 400},
                "queue": {"size": 4, "blocking": false}
            },
            "occupancy": {
                "resolution_m": 0.05,
                "lower_x_m": -1.0,
                "lower_y_m": -1.0,
                "width_cells": 40,
                "height_cells": 40,
                "maximum_cells": 1600,
                "maximum_keyframes": 10,
                "snapshot_every_keyframes": 1
            },
            "inference": {
                "onnx_runtime_library_asset": asset("runtime/ort.so", 6),
                "superpoint_model_asset": asset("models/sp.onnx", 7),
                "lightglue_model_asset": asset("models/lg.onnx", 8),
                "superpoint_backend": "cpu",
                "lightglue_backend": "cpu",
                "downscale_factor": 1,
                "maximum_keypoints": 128
            },
            "rerun": {
                "kind": "serve_loopback",
                "bind": "127.0.0.1:9876",
                "decimation": 1,
                "memory_limit_bytes": 1_048_576,
                "flush_timeout_ms": 1000
            },
            "storage": {
                "map_snapshot_relative_path": "maps/current.kmap",
                "navigation_dataset_directory_relative_path": "records/navigation",
                "maximum_map_snapshot_bytes": maximum_map_snapshot_bytes,
                "minimum_free_bytes_after_map_save": minimum_free_bytes_after_map_save,
                "maximum_navigation_dataset_bytes": terminal_reserve_bytes + 1_073_741_824,
                "maximum_navigation_dataset_files": 65_536,
                "maximum_navigation_ingress_records": 100_000,
                "minimum_free_bytes_after_navigation_dataset_write": 1,
                "navigation_dataset_terminal_reserve_bytes": terminal_reserve_bytes
            }
        });
        NanoAgentLaunchV2::parse_json(
            &serde_json::to_vec(&launch).expect("serialize launch fixture"),
        )
        .expect("parse launch fixture")
        .storage()
        .clone()
    }

    fn roots(state_root: &Path) -> NanoBootstrapRoots {
        NanoBootstrapRoots::try_new(
            PathBuf::from("/opt/kiko/deployment"),
            state_root.to_path_buf(),
        )
        .expect("test roots")
    }

    fn map_policy(state_root: &Path) -> super::super::NanoMapPersistenceConfig {
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
                "save_snapshot_path": state_root.join("maps/current.kmap"),
                "warm_start": {"kind": "none"}
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
        NanoAgentPolicyConfigV3::parse_json(
            &serde_json::to_vec(&value).expect("serialize map policy"),
        )
        .expect("parse map policy")
        .map_persistence()
        .clone()
    }

    fn owner(directory: &TestDirectory) -> NanoStateQuotaOwner {
        NanoStateQuotaOwner::admit(&roots(&directory.path), &storage(64, 1)).expect("quota owner")
    }

    #[test]
    fn admission_creates_only_private_map_parent_and_observes_empty_state() {
        let directory = TestDirectory::create("admit");
        let owner = owner(&directory);

        let map_parent = directory.path.join("maps");
        let metadata = fs::symlink_metadata(&map_parent).expect("created map parent");
        assert!(metadata.is_dir());
        assert_eq!(metadata.permissions().mode() & 0o777, 0o700);
        assert!(!directory.path.join("records").exists());
        assert_eq!(
            owner.map_snapshot_path(),
            directory.path.join("maps/current.kmap")
        );
        let observed = owner.last_snapshot();
        assert_eq!(observed.map_snapshot_bytes(), 0);
        assert!(observed.available_bytes() >= observed.filesystem_fragment_size_bytes());
    }

    #[test]
    fn map_replacement_is_bound_to_exact_path_and_verified_at_exact_length() {
        let directory = TestDirectory::create("map");
        let mut owner = owner(&directory);
        fs::write(owner.map_snapshot_path(), b"old").expect("old map");

        let wrong = directory.path.join("maps/other.kmap");
        assert!(matches!(
            owner.reserve_map_replacement(&wrong, 4),
            Err(NanoStateQuotaReserveError::MapDestinationMismatch { .. })
        ));

        let destination = owner.map_snapshot_path().to_path_buf();
        let reservation = owner
            .reserve_map_replacement(&destination, 4)
            .expect("map replacement reservation");
        fs::write(&destination, b"next").expect("replace fixture");
        let receipt = reservation.verify_committed().expect("quota receipt");
        assert_eq!(receipt.exact_bytes(), 4);
        assert_eq!(receipt.before().map_snapshot_bytes(), 3);
        assert_eq!(receipt.after().map_snapshot_bytes(), 4);
    }

    #[test]
    fn reservation_retains_one_parent_for_admission_publication_and_verification() {
        let directory = TestDirectory::create("retained-parent");
        let mut owner = owner(&directory);
        let destination = owner.map_snapshot_path().to_path_buf();
        let configured_parent = destination.parent().expect("map parent").to_path_buf();
        let retained_parent = directory.path.join("maps-retained");
        let replacement_parent = directory.path.join("maps-replacement");
        let reservation = owner
            .reserve_map_replacement(&destination, 4)
            .expect("map replacement reservation");

        fs::rename(&configured_parent, &retained_parent).expect("move admitted map parent");
        fs::create_dir(&replacement_parent).expect("replacement backing directory");
        symlink(&replacement_parent, &configured_parent).expect("replace map parent pathname");
        let descriptor = openat(
            reservation.publication_parent(),
            OsStr::new("current.kmap"),
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::from_raw_mode(0o600),
        )
        .expect("descriptor-relative test publication");
        let mut file = std::fs::File::from(descriptor);
        file.write_all(b"next").expect("write retained publication");
        file.sync_all().expect("sync retained publication");
        drop(file);

        assert_eq!(
            fs::read(retained_parent.join("current.kmap")).expect("retained publication"),
            b"next"
        );
        assert!(
            !replacement_parent.join("current.kmap").exists(),
            "replacement pathname must not receive the publication"
        );
        assert!(matches!(
            reservation.verify_committed(),
            Err(NanoStateQuotaCommitError::Filesystem(_))
        ));
    }

    #[test]
    fn quota_aware_map_owner_publishes_only_with_post_write_receipt() {
        let directory = TestDirectory::create("map-owner");
        let roots = roots(&directory.path);
        let mut quota =
            NanoStateQuotaOwner::admit(&roots, &storage(1_024, 1)).expect("quota owner");
        let mut persistence = NanoMapPersistenceOwner::try_new(
            &roots,
            &map_policy(&directory.path),
            OccupancyMapLimits::try_new(4).expect("map limits"),
        )
        .expect("map persistence owner");

        let map = SlamMap::new();
        let mut epochs = NavigationMapEpochCoordinator::new();
        let binding = epochs
            .start_epoch(
                NavigationClockEpoch::new(HostMonotonicTimestamp::from_nanos(0)),
                HostMonotonicTimestamp::from_nanos(1),
                map.snapshot().instance_id(),
            )
            .expect("map epoch")
            .binding();
        let geometry =
            OccupancyGridGeometry::try_new(0.1, [-0.2, -0.2], 2, 2, 4).expect("geometry");
        let snapshot = OccupancyGridSnapshot::from_test_cells(
            geometry,
            &[
                OccupancyCell::Free,
                OccupancyCell::Occupied,
                OccupancyCell::Unknown,
                OccupancyCell::Free,
            ],
            map.snapshot().instance_id(),
            1,
        );
        persistence
            .retain_latest(binding, snapshot)
            .expect("retain snapshot");

        let receipt = persistence
            .save_latest_with_quota(&mut quota)
            .expect("quota-bound publication");
        let quota_receipt = receipt
            .quota_verification()
            .expect("post-write quota receipt");
        assert_eq!(quota_receipt.destination(), receipt.destination());
        assert_eq!(
            quota_receipt.exact_bytes(),
            fs::metadata(receipt.destination())
                .expect("published map metadata")
                .len()
        );
        assert_eq!(
            quota_receipt.after().map_snapshot_bytes(),
            quota_receipt.exact_bytes()
        );
    }

    #[test]
    fn navigation_dataset_is_explicitly_outside_the_map_quota_contract() {
        let directory = TestDirectory::create("independent-navigation-dataset-quota");
        fs::create_dir_all(directory.path.join("records/navigation")).expect("record root");
        fs::write(
            directory.path.join("records/navigation/run.bin"),
            [0_u8; 1_024],
        )
        .expect("record");
        let mut quota = NanoStateQuotaOwner::admit(&roots(&directory.path), &storage(4, 1))
            .expect("navigation dataset is not misrepresented as quota-bound");
        let map = quota.map_snapshot_path().to_path_buf();
        let reservation = quota
            .reserve_map_replacement(&map, 4)
            .expect("map reservation ignores unrelated dataset bytes");
        fs::write(&map, b"map!").expect("map fixture");
        reservation
            .verify_committed()
            .expect("map verification ignores unrelated dataset bytes");
    }

    #[test]
    fn map_ceiling_fails_before_writes() {
        let directory = TestDirectory::create("limits");
        let policy = storage(5, 1);
        let mut owner =
            NanoStateQuotaOwner::admit(&roots(&directory.path), &policy).expect("quota owner");

        let map = owner.map_snapshot_path().to_path_buf();
        assert!(matches!(
            owner.reserve_map_replacement(&map, 6),
            Err(
                NanoStateQuotaReserveError::PlannedMapSnapshotMaximumExceeded {
                    planned_bytes: 6,
                    maximum_bytes: 5
                }
            )
        ));
        assert!(!map.exists());
    }

    #[test]
    fn projection_reserves_fragment_rounded_transient_space_without_fallback() {
        let limits = NanoStateQuotaLimits {
            maximum_map_snapshot_bytes: 10_000,
            minimum_free_bytes_after_map_save: 10_000,
        };
        let snapshot = NanoStateQuotaSnapshot {
            map_snapshot_bytes: 0,
            available_bytes: 14_095,
            filesystem_fragment_size_bytes: 4_096,
        };
        assert!(matches!(
            project_write(snapshot, limits, 1),
            Err(NanoStateQuotaReserveError::InsufficientTransientFreeSpace {
                available_bytes: 14_095,
                planned_file_bytes: 1,
                transient_allocation_bytes: 4_096,
                filesystem_fragment_size_bytes: 4_096,
                minimum_free_bytes_after_map_save: 10_000,
                required_available_bytes: 14_096
            })
        ));
        assert_eq!(round_up_to_fragment(0, 4_096), Some(0));
        assert_eq!(round_up_to_fragment(4_096, 4_096), Some(4_096));
        assert_eq!(round_up_to_fragment(4_097, 4_096), Some(8_192));
        assert_eq!(round_up_to_fragment(u64::MAX, 4_096), None);
        assert_eq!(round_up_to_fragment(1, 0), None);
    }

    #[test]
    fn free_space_floor_is_a_post_save_contract_not_a_startup_claim() {
        let limits = NanoStateQuotaLimits {
            maximum_map_snapshot_bytes: 10,
            minimum_free_bytes_after_map_save: 10_000,
        };
        let snapshot = NanoStateQuotaSnapshot {
            map_snapshot_bytes: 4,
            available_bytes: 9_999,
            filesystem_fragment_size_bytes: 4_096,
        };

        validate_map_size(snapshot, limits).expect("existing map size is admitted");
        assert_eq!(
            validate_committed_map(snapshot, limits),
            Err(NanoStateQuotaViolation::PostSaveMinimumFreeNotMet {
                available_bytes: 9_999,
                minimum_free_bytes_after_map_save: 10_000
            })
        );
    }

    #[test]
    fn symlinked_root_components_and_descendants_are_never_followed() {
        let real = TestDirectory::create("real-root");
        let root_link = real.path.with_extension("link");
        symlink(&real.path, &root_link).expect("root symlink");
        let result = NanoStateQuotaOwner::admit(&roots(&root_link), &storage(64, 1));
        assert!(matches!(
            result,
            Err(NanoStateQuotaAdmissionError::Filesystem(
                NanoStateFilesystemError::Io {
                    operation: NanoStateFilesystemOperation::OpenAbsoluteComponent { .. },
                    ..
                }
            ))
        ));
        fs::remove_file(&root_link).expect("remove root symlink");

        let directory = TestDirectory::create("descendant-link");
        let outside = TestDirectory::create("outside");
        let mut owner = owner(&directory);
        let map = owner.map_snapshot_path().to_path_buf();
        fs::write(outside.path.join("map"), b"x").expect("outside map");
        symlink(outside.path.join("map"), &map).expect("map symlink");
        assert!(matches!(
            owner.reserve_map_replacement(&map, 1),
            Err(NanoStateQuotaReserveError::Filesystem(
                NanoStateFilesystemError::NotRegularFile {
                    observed: FileType::Symlink,
                    ..
                }
            ))
        ));
    }

    #[test]
    fn hard_links_and_state_root_replacement_fail_closed() {
        let directory = TestDirectory::create("identity");
        let mut owner = owner(&directory);
        let first = owner.map_snapshot_path().to_path_buf();
        let second = directory.path.join("second");
        fs::write(&first, b"x").expect("first hard-link fixture");
        fs::hard_link(&first, &second).expect("second hard link");
        let map = owner.map_snapshot_path().to_path_buf();
        assert!(matches!(
            owner.reserve_map_replacement(&map, 1),
            Err(NanoStateQuotaReserveError::Filesystem(
                NanoStateFilesystemError::HardLinkedRegularFile { link_count: 2, .. }
            ))
        ));
        fs::remove_file(first).expect("remove first hard link");
        fs::remove_file(second).expect("remove second hard link");

        let moved = directory.path.with_extension("moved");
        fs::rename(&directory.path, &moved).expect("move admitted root");
        fs::create_dir(&directory.path).expect("replacement root");
        assert!(matches!(
            owner.reserve_map_replacement(&map, 1),
            Err(NanoStateQuotaReserveError::Filesystem(
                NanoStateFilesystemError::StateRootChanged { .. }
            ))
        ));
        drop(owner);
        fs::remove_dir_all(moved).expect("remove moved root");
    }

    #[test]
    fn preexisting_usage_is_checked_at_admission() {
        let directory = TestDirectory::create("existing");
        fs::create_dir_all(directory.path.join("maps")).expect("map directory");
        fs::write(directory.path.join("maps/current.kmap"), b"12345")
            .expect("oversize existing map");
        assert!(matches!(
            NanoStateQuotaOwner::admit(&roots(&directory.path), &storage(4, 1)),
            Err(NanoStateQuotaAdmissionError::ExistingState(
                NanoStateQuotaViolation::MapSnapshotMaximumExceeded {
                    actual_bytes: 5,
                    maximum_bytes: 4
                }
            ))
        ));
    }
}
