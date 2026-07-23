use std::collections::BTreeMap;
use std::ffi::{CString, OsStr};
use std::fs::{File, Metadata};
use std::io::{self, Write};
use std::num::NonZeroU64;
use std::os::fd::{AsRawFd, FromRawFd, RawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DatasetStorageLimits {
    maximum_logical_bytes: NonZeroU64,
    maximum_files: NonZeroU64,
    minimum_free_bytes_after_write: NonZeroU64,
    terminal_reserve_bytes: NonZeroU64,
    terminal_component_maximum_bytes: [u64; 3],
}

impl DatasetStorageLimits {
    pub fn try_new(
        maximum_logical_bytes: u64,
        maximum_files: u64,
        minimum_free_bytes_after_write: u64,
        terminal_reserve_bytes: u64,
        maximum_map_snapshot_bytes: u64,
        maximum_manifest_bytes: u64,
        maximum_selection_bytes: u64,
    ) -> Result<Self, DatasetStorageQuotaError> {
        let maximum_logical_bytes = NonZeroU64::new(maximum_logical_bytes).ok_or(
            DatasetStorageQuotaError::InvalidLimits {
                reason: "maximum logical bytes must be nonzero",
            },
        )?;
        let maximum_files =
            NonZeroU64::new(maximum_files).ok_or(DatasetStorageQuotaError::InvalidLimits {
                reason: "maximum files must be nonzero",
            })?;
        let minimum_free_bytes_after_write = NonZeroU64::new(minimum_free_bytes_after_write)
            .ok_or(DatasetStorageQuotaError::InvalidLimits {
                reason: "minimum free bytes after write must be nonzero",
            })?;
        let terminal_reserve_bytes = NonZeroU64::new(terminal_reserve_bytes).ok_or(
            DatasetStorageQuotaError::InvalidLimits {
                reason: "terminal reserve bytes must be nonzero",
            },
        )?;
        if terminal_reserve_bytes >= maximum_logical_bytes {
            return Err(DatasetStorageQuotaError::InvalidLimits {
                reason: "terminal reserve must be smaller than the dataset logical-byte ceiling",
            });
        }
        if [
            maximum_map_snapshot_bytes,
            maximum_manifest_bytes,
            maximum_selection_bytes,
        ]
        .contains(&0)
        {
            return Err(DatasetStorageQuotaError::InvalidLimits {
                reason: "terminal artifact maxima must be nonzero",
            });
        }
        if maximum_files.get() <= 1 {
            return Err(DatasetStorageQuotaError::InvalidLimits {
                reason: "maximum files must leave one file for terminal manifest publication",
            });
        }
        Ok(Self {
            maximum_logical_bytes,
            maximum_files,
            minimum_free_bytes_after_write,
            terminal_reserve_bytes,
            terminal_component_maximum_bytes: [
                maximum_map_snapshot_bytes,
                maximum_manifest_bytes,
                maximum_selection_bytes,
            ],
        })
    }

    pub const fn maximum_logical_bytes(self) -> u64 {
        self.maximum_logical_bytes.get()
    }

    pub const fn maximum_files(self) -> u64 {
        self.maximum_files.get()
    }

    pub const fn minimum_free_bytes_after_write(self) -> u64 {
        self.minimum_free_bytes_after_write.get()
    }

    pub const fn terminal_reserve_bytes(self) -> u64 {
        self.terminal_reserve_bytes.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DatasetStoragePhase {
    Live,
    Terminal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FilesystemIdentity {
    device: u64,
    inode: u64,
}

impl FilesystemIdentity {
    fn from_metadata(metadata: &Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
        }
    }
}

#[derive(Debug)]
struct QuotaState {
    logical_bytes: u64,
    files: u64,
    tracked_lengths: BTreeMap<PathBuf, u64>,
    poisoned: bool,
}

/// Linear per-session dataset storage owner.
///
/// All production dataset files are created relative to retained directory
/// descriptors. Exact regular-file lengths are accounted separately from the
/// fragment-rounded free-space reservation.
#[derive(Debug)]
pub struct DatasetStorageQuota {
    dataset_path: PathBuf,
    frames_path: PathBuf,
    dataset_directory: File,
    frames_directory: File,
    dataset_identity: FilesystemIdentity,
    frames_identity: FilesystemIdentity,
    limits: DatasetStorageLimits,
    state: Mutex<QuotaState>,
}

impl DatasetStorageQuota {
    pub(crate) fn open(
        dataset_path: PathBuf,
        frames_path: PathBuf,
        limits: DatasetStorageLimits,
    ) -> Result<Self, DatasetStorageQuotaError> {
        if frames_path != dataset_path.join(super::format::FRAMES_DIR) {
            return Err(DatasetStorageQuotaError::InvalidDirectory);
        }
        let dataset_directory = open_directory_path_nofollow(&dataset_path, "dataset directory")?;
        let frames_directory = open_directory_at_nofollow(
            dataset_directory.as_raw_fd(),
            OsStr::new(super::format::FRAMES_DIR),
            &frames_path,
            "frames directory",
        )?;
        let dataset_metadata =
            dataset_directory
                .metadata()
                .map_err(|source| DatasetStorageQuotaError::Io {
                    operation: "inspect dataset directory",
                    path: dataset_path.clone(),
                    source,
                })?;
        let frames_metadata =
            frames_directory
                .metadata()
                .map_err(|source| DatasetStorageQuotaError::Io {
                    operation: "inspect frames directory",
                    path: frames_path.clone(),
                    source,
                })?;
        if !dataset_metadata.is_dir() || !frames_metadata.is_dir() {
            return Err(DatasetStorageQuotaError::InvalidDirectory);
        }
        require_same_filesystem(
            FilesystemIdentity::from_metadata(&dataset_metadata),
            FilesystemIdentity::from_metadata(&frames_metadata),
        )?;
        let quota = Self {
            dataset_path,
            frames_path,
            dataset_directory,
            frames_directory,
            dataset_identity: FilesystemIdentity::from_metadata(&dataset_metadata),
            frames_identity: FilesystemIdentity::from_metadata(&frames_metadata),
            limits,
            state: Mutex::new(QuotaState {
                logical_bytes: 0,
                files: 0,
                tracked_lengths: BTreeMap::new(),
                poisoned: false,
            }),
        };
        let snapshot = filesystem_snapshot(quota.dataset_directory.as_raw_fd())?;
        let minimum_terminal_reserve = terminal_reserve_required(
            quota.limits.terminal_component_maximum_bytes,
            snapshot.fragment_bytes,
        )?;
        if quota.limits.terminal_reserve_bytes() < minimum_terminal_reserve {
            return Err(
                DatasetStorageQuotaError::TerminalReserveTooSmallForFilesystem {
                    configured: quota.limits.terminal_reserve_bytes(),
                    required: minimum_terminal_reserve,
                    fragment_bytes: snapshot.fragment_bytes,
                },
            );
        }
        quota.require_directories_unchanged()?;
        Ok(quota)
    }

    pub(crate) fn create_new_live_file(
        &self,
        relative_path: &Path,
        bytes: &[u8],
    ) -> Result<File, DatasetStorageQuotaError> {
        self.create_new_file(relative_path, bytes, DatasetStoragePhase::Live)
    }

    pub(crate) fn create_empty_live_file(
        &self,
        relative_path: &Path,
    ) -> Result<File, DatasetStorageQuotaError> {
        self.create_new_file(relative_path, &[], DatasetStoragePhase::Live)
    }

    pub(crate) fn append_live(
        &self,
        file: &mut File,
        relative_path: &Path,
        bytes: &[u8],
    ) -> Result<(), DatasetStorageQuotaError> {
        self.append(file, relative_path, bytes, DatasetStoragePhase::Live)
    }

    pub(crate) fn write_live_with(
        &self,
        relative_path: &Path,
        growth: u64,
        write_and_observe_length: impl FnOnce() -> io::Result<u64>,
    ) -> Result<(), DatasetStorageQuotaError> {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        self.require_healthy(&state)?;
        self.require_directories_unchanged()
            .map_err(|error| poison(&mut state, error))?;
        let previous = state
            .tracked_lengths
            .get(relative_path)
            .copied()
            .ok_or_else(|| {
                poison(
                    &mut state,
                    DatasetStorageQuotaError::UntrackedFile {
                        path: relative_path.to_path_buf(),
                    },
                )
            })?;
        self.admit(&state, previous, growth, false, DatasetStoragePhase::Live)
            .map_err(|error| poison(&mut state, error))?;
        let observed = write_and_observe_length().map_err(|source| {
            poison(
                &mut state,
                DatasetStorageQuotaError::Io {
                    operation: "write quota-bound dataset stream",
                    path: relative_path.to_path_buf(),
                    source,
                },
            )
        })?;
        let expected = previous
            .checked_add(growth)
            .ok_or_else(|| poison(&mut state, DatasetStorageQuotaError::ArithmeticOverflow))?;
        if observed != expected {
            return Err(poison(
                &mut state,
                DatasetStorageQuotaError::FileLengthChanged {
                    path: relative_path.to_path_buf(),
                    expected,
                    actual: observed,
                },
            ));
        }
        state.logical_bytes = state
            .logical_bytes
            .checked_add(growth)
            .ok_or_else(|| poison(&mut state, DatasetStorageQuotaError::ArithmeticOverflow))?;
        state
            .tracked_lengths
            .insert(relative_path.to_path_buf(), expected);
        self.require_free_floor(DatasetStoragePhase::Live)
            .map_err(|error| poison(&mut state, error))
    }

    pub(crate) fn publish_terminal_file(
        &self,
        temporary_relative_path: &Path,
        final_relative_path: &Path,
        bytes: &[u8],
    ) -> Result<(), DatasetStorageQuotaError> {
        let file = self.create_new_file(
            temporary_relative_path,
            bytes,
            DatasetStoragePhase::Terminal,
        )?;
        file.sync_all().map_err(|source| {
            self.poison_io(
                "sync terminal dataset file",
                temporary_relative_path,
                source,
            )
        })?;
        drop(file);
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        self.require_healthy(&state)?;
        self.require_directories_unchanged()
            .map_err(|error| poison(&mut state, error))?;
        descriptor_relative_rename(
            self.dataset_directory.as_raw_fd(),
            temporary_relative_path,
            final_relative_path,
        )
        .map_err(|source| {
            poison(
                &mut state,
                DatasetStorageQuotaError::Io {
                    operation: "publish terminal dataset file",
                    path: final_relative_path.to_path_buf(),
                    source,
                },
            )
        })?;
        let length = state
            .tracked_lengths
            .remove(temporary_relative_path)
            .ok_or_else(|| {
                poison(
                    &mut state,
                    DatasetStorageQuotaError::UntrackedFile {
                        path: temporary_relative_path.to_path_buf(),
                    },
                )
            })?;
        if state
            .tracked_lengths
            .insert(final_relative_path.to_path_buf(), length)
            .is_some()
        {
            return Err(poison(
                &mut state,
                DatasetStorageQuotaError::DestinationAlreadyTracked {
                    path: final_relative_path.to_path_buf(),
                },
            ));
        }
        self.dataset_directory.sync_all().map_err(|source| {
            poison(
                &mut state,
                DatasetStorageQuotaError::Io {
                    operation: "sync dataset directory after publication",
                    path: self.dataset_path.clone(),
                    source,
                },
            )
        })
    }

    pub(crate) fn rename_live_file(
        &self,
        temporary_relative_path: &Path,
        final_relative_path: &Path,
    ) -> Result<(), DatasetStorageQuotaError> {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        self.require_healthy(&state)?;
        self.require_directories_unchanged()
            .map_err(|error| poison(&mut state, error))?;
        descriptor_relative_rename(
            self.dataset_directory.as_raw_fd(),
            temporary_relative_path,
            final_relative_path,
        )
        .map_err(|source| {
            poison(
                &mut state,
                DatasetStorageQuotaError::Io {
                    operation: "rename live dataset file",
                    path: final_relative_path.to_path_buf(),
                    source,
                },
            )
        })?;
        let length = state
            .tracked_lengths
            .remove(temporary_relative_path)
            .ok_or_else(|| {
                poison(
                    &mut state,
                    DatasetStorageQuotaError::UntrackedFile {
                        path: temporary_relative_path.to_path_buf(),
                    },
                )
            })?;
        if state
            .tracked_lengths
            .insert(final_relative_path.to_path_buf(), length)
            .is_some()
        {
            return Err(poison(
                &mut state,
                DatasetStorageQuotaError::DestinationAlreadyTracked {
                    path: final_relative_path.to_path_buf(),
                },
            ));
        }
        Ok(())
    }

    pub(crate) fn verify_tracked_file(
        &self,
        file: &File,
        relative_path: &Path,
    ) -> Result<(), DatasetStorageQuotaError> {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        self.require_healthy(&state)?;
        self.require_directories_unchanged()
            .map_err(|error| poison(&mut state, error))?;
        let expected = state
            .tracked_lengths
            .get(relative_path)
            .copied()
            .ok_or_else(|| {
                poison(
                    &mut state,
                    DatasetStorageQuotaError::UntrackedFile {
                        path: relative_path.to_path_buf(),
                    },
                )
            })?;
        let actual = file
            .metadata()
            .map_err(|source| {
                poison(
                    &mut state,
                    DatasetStorageQuotaError::Io {
                        operation: "inspect tracked dataset file",
                        path: relative_path.to_path_buf(),
                        source,
                    },
                )
            })?
            .len();
        if actual != expected {
            return Err(poison(
                &mut state,
                DatasetStorageQuotaError::FileLengthChanged {
                    path: relative_path.to_path_buf(),
                    expected,
                    actual,
                },
            ));
        }
        Ok(())
    }

    pub(crate) fn is_poisoned(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .poisoned
    }

    fn create_new_file(
        &self,
        relative_path: &Path,
        bytes: &[u8],
        phase: DatasetStoragePhase,
    ) -> Result<File, DatasetStorageQuotaError> {
        let byte_count =
            u64::try_from(bytes.len()).map_err(|_| DatasetStorageQuotaError::ArithmeticOverflow)?;
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        self.require_healthy(&state)?;
        self.require_directories_unchanged()
            .map_err(|error| poison(&mut state, error))?;
        if state.tracked_lengths.contains_key(relative_path) {
            return Err(poison(
                &mut state,
                DatasetStorageQuotaError::DestinationAlreadyTracked {
                    path: relative_path.to_path_buf(),
                },
            ));
        }
        self.admit(&state, 0, byte_count, true, phase)
            .map_err(|error| poison(&mut state, error))?;
        let (directory_fd, file_name) = self
            .parent_and_name(relative_path)
            .map_err(|error| poison(&mut state, error))?;
        let mut file = descriptor_relative_create(directory_fd, file_name).map_err(|source| {
            poison(
                &mut state,
                DatasetStorageQuotaError::Io {
                    operation: "create dataset file",
                    path: relative_path.to_path_buf(),
                    source,
                },
            )
        })?;
        if let Err(source) = file.write_all(bytes) {
            return Err(poison(
                &mut state,
                DatasetStorageQuotaError::Io {
                    operation: "write dataset file",
                    path: relative_path.to_path_buf(),
                    source,
                },
            ));
        }
        self.commit_file_length(&mut state, &file, relative_path, 0, byte_count, true)?;
        self.require_free_floor(phase)
            .map_err(|error| poison(&mut state, error))?;
        Ok(file)
    }

    fn append(
        &self,
        file: &mut File,
        relative_path: &Path,
        bytes: &[u8],
        phase: DatasetStoragePhase,
    ) -> Result<(), DatasetStorageQuotaError> {
        let growth =
            u64::try_from(bytes.len()).map_err(|_| DatasetStorageQuotaError::ArithmeticOverflow)?;
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        self.require_healthy(&state)?;
        self.require_directories_unchanged()
            .map_err(|error| poison(&mut state, error))?;
        let previous = state
            .tracked_lengths
            .get(relative_path)
            .copied()
            .ok_or_else(|| {
                poison(
                    &mut state,
                    DatasetStorageQuotaError::UntrackedFile {
                        path: relative_path.to_path_buf(),
                    },
                )
            })?;
        let observed = file
            .metadata()
            .map_err(|source| {
                poison(
                    &mut state,
                    DatasetStorageQuotaError::Io {
                        operation: "inspect dataset file before append",
                        path: relative_path.to_path_buf(),
                        source,
                    },
                )
            })?
            .len();
        if observed != previous {
            return Err(poison(
                &mut state,
                DatasetStorageQuotaError::FileLengthChanged {
                    path: relative_path.to_path_buf(),
                    expected: previous,
                    actual: observed,
                },
            ));
        }
        self.admit(&state, previous, growth, false, phase)
            .map_err(|error| poison(&mut state, error))?;
        if let Err(source) = file.write_all(bytes) {
            return Err(poison(
                &mut state,
                DatasetStorageQuotaError::Io {
                    operation: "append dataset file",
                    path: relative_path.to_path_buf(),
                    source,
                },
            ));
        }
        self.commit_file_length(&mut state, file, relative_path, previous, growth, false)?;
        self.require_free_floor(phase)
            .map_err(|error| poison(&mut state, error))
    }

    fn admit(
        &self,
        state: &QuotaState,
        previous_file_bytes: u64,
        growth_bytes: u64,
        new_file: bool,
        phase: DatasetStoragePhase,
    ) -> Result<(), DatasetStorageQuotaError> {
        let next_bytes = state
            .logical_bytes
            .checked_add(growth_bytes)
            .ok_or(DatasetStorageQuotaError::ArithmeticOverflow)?;
        let logical_ceiling = match phase {
            DatasetStoragePhase::Live => self
                .limits
                .maximum_logical_bytes()
                .checked_sub(self.limits.terminal_reserve_bytes())
                .ok_or(DatasetStorageQuotaError::ArithmeticOverflow)?,
            DatasetStoragePhase::Terminal => self.limits.maximum_logical_bytes(),
        };
        if next_bytes > logical_ceiling {
            return Err(DatasetStorageQuotaError::LogicalByteLimitExceeded {
                attempted: next_bytes,
                maximum: logical_ceiling,
                phase,
            });
        }
        let next_files = state
            .files
            .checked_add(u64::from(new_file))
            .ok_or(DatasetStorageQuotaError::ArithmeticOverflow)?;
        let file_ceiling = match phase {
            DatasetStoragePhase::Live => self
                .limits
                .maximum_files()
                .checked_sub(1)
                .ok_or(DatasetStorageQuotaError::ArithmeticOverflow)?,
            DatasetStoragePhase::Terminal => self.limits.maximum_files(),
        };
        if next_files > file_ceiling {
            return Err(DatasetStorageQuotaError::FileLimitExceeded {
                attempted: next_files,
                maximum: file_ceiling,
            });
        }
        let snapshot = filesystem_snapshot(self.dataset_directory.as_raw_fd())?;
        let next_file_bytes = previous_file_bytes
            .checked_add(growth_bytes)
            .ok_or(DatasetStorageQuotaError::ArithmeticOverflow)?;
        let allocation_before = round_up(previous_file_bytes, snapshot.fragment_bytes)?;
        let allocation_after = round_up(next_file_bytes, snapshot.fragment_bytes)?;
        let allocation_growth = allocation_after
            .checked_sub(allocation_before)
            .ok_or(DatasetStorageQuotaError::ArithmeticOverflow)?;
        let reserve = phase_physical_reserve(self.limits, snapshot.fragment_bytes, phase)?;
        let required = self
            .limits
            .minimum_free_bytes_after_write()
            .checked_add(reserve)
            .and_then(|value| value.checked_add(allocation_growth))
            .ok_or(DatasetStorageQuotaError::ArithmeticOverflow)?;
        if snapshot.available_bytes < required {
            return Err(DatasetStorageQuotaError::InsufficientFreeSpace {
                available: snapshot.available_bytes,
                required,
                fragment_bytes: snapshot.fragment_bytes,
                phase,
            });
        }
        let inode_reserve: u64 = match phase {
            DatasetStoragePhase::Live => 3,
            DatasetStoragePhase::Terminal => 2,
        };
        let required_inodes = inode_reserve
            .checked_add(u64::from(new_file))
            .ok_or(DatasetStorageQuotaError::ArithmeticOverflow)?;
        if snapshot.available_inodes < required_inodes {
            return Err(DatasetStorageQuotaError::InsufficientFreeInodes {
                available: snapshot.available_inodes,
                required: required_inodes,
                phase,
            });
        }
        Ok(())
    }

    fn commit_file_length(
        &self,
        state: &mut QuotaState,
        file: &File,
        relative_path: &Path,
        previous: u64,
        growth: u64,
        new_file: bool,
    ) -> Result<(), DatasetStorageQuotaError> {
        let expected = previous
            .checked_add(growth)
            .ok_or_else(|| poison(state, DatasetStorageQuotaError::ArithmeticOverflow))?;
        let metadata = file.metadata().map_err(|source| {
            poison(
                state,
                DatasetStorageQuotaError::Io {
                    operation: "inspect dataset file after write",
                    path: relative_path.to_path_buf(),
                    source,
                },
            )
        })?;
        if !metadata.is_file() || metadata.len() != expected {
            return Err(poison(
                state,
                DatasetStorageQuotaError::FileLengthChanged {
                    path: relative_path.to_path_buf(),
                    expected,
                    actual: metadata.len(),
                },
            ));
        }
        state.logical_bytes = state
            .logical_bytes
            .checked_add(growth)
            .ok_or_else(|| poison(state, DatasetStorageQuotaError::ArithmeticOverflow))?;
        if new_file {
            state.files = state
                .files
                .checked_add(1)
                .ok_or_else(|| poison(state, DatasetStorageQuotaError::ArithmeticOverflow))?;
        }
        state
            .tracked_lengths
            .insert(relative_path.to_path_buf(), expected);
        Ok(())
    }

    fn require_free_floor(
        &self,
        phase: DatasetStoragePhase,
    ) -> Result<(), DatasetStorageQuotaError> {
        let snapshot = filesystem_snapshot(self.dataset_directory.as_raw_fd())?;
        validate_post_write_headroom(snapshot, self.limits, phase)
    }

    fn require_healthy(&self, state: &QuotaState) -> Result<(), DatasetStorageQuotaError> {
        if state.poisoned {
            Err(DatasetStorageQuotaError::Poisoned)
        } else {
            Ok(())
        }
    }

    fn require_directories_unchanged(&self) -> Result<(), DatasetStorageQuotaError> {
        let current_dataset = require_current_directory(
            &self.dataset_path,
            self.dataset_identity,
            "dataset directory",
        )?;
        let current_frames = open_directory_at_nofollow(
            current_dataset.as_raw_fd(),
            OsStr::new(super::format::FRAMES_DIR),
            &self.frames_path,
            "frames directory",
        )?;
        let current_frames_metadata =
            current_frames
                .metadata()
                .map_err(|source| DatasetStorageQuotaError::Io {
                    operation: "inspect current frames directory",
                    path: self.frames_path.clone(),
                    source,
                })?;
        let current_frames_identity = FilesystemIdentity::from_metadata(&current_frames_metadata);
        if !current_frames_metadata.is_dir() || current_frames_identity != self.frames_identity {
            return Err(DatasetStorageQuotaError::DirectoryReplaced {
                role: "frames directory",
                path: self.frames_path.clone(),
            });
        }
        require_same_filesystem(self.dataset_identity, current_frames_identity)
    }

    fn parent_and_name<'path>(
        &self,
        relative_path: &'path Path,
    ) -> Result<(RawFd, &'path OsStr), DatasetStorageQuotaError> {
        let components = relative_path.components().collect::<Vec<_>>();
        match components.as_slice() {
            [Component::Normal(name)] => Ok((self.dataset_directory.as_raw_fd(), name)),
            [Component::Normal(parent), Component::Normal(name)]
                if *parent == OsStr::new(super::format::FRAMES_DIR) =>
            {
                Ok((self.frames_directory.as_raw_fd(), name))
            }
            _ => Err(DatasetStorageQuotaError::InvalidRelativePath {
                path: relative_path.to_path_buf(),
            }),
        }
    }

    fn poison_io(
        &self,
        operation: &'static str,
        path: &Path,
        source: io::Error,
    ) -> DatasetStorageQuotaError {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        poison(
            &mut state,
            DatasetStorageQuotaError::Io {
                operation,
                path: path.to_path_buf(),
                source,
            },
        )
    }
}

fn validate_post_write_headroom(
    snapshot: FilesystemSnapshot,
    limits: DatasetStorageLimits,
    phase: DatasetStoragePhase,
) -> Result<(), DatasetStorageQuotaError> {
    let reserve = phase_physical_reserve(limits, snapshot.fragment_bytes, phase)?;
    let required = limits
        .minimum_free_bytes_after_write()
        .checked_add(reserve)
        .ok_or(DatasetStorageQuotaError::ArithmeticOverflow)?;
    if snapshot.available_bytes < required {
        return Err(DatasetStorageQuotaError::PostWriteFreeSpaceRace {
            available: snapshot.available_bytes,
            required,
            phase,
        });
    }
    let required_inodes = match phase {
        DatasetStoragePhase::Live => 3,
        DatasetStoragePhase::Terminal => 2,
    };
    if snapshot.available_inodes < required_inodes {
        return Err(DatasetStorageQuotaError::PostWriteInodeRace {
            available: snapshot.available_inodes,
            required: required_inodes,
            phase,
        });
    }
    Ok(())
}

fn poison(state: &mut QuotaState, error: DatasetStorageQuotaError) -> DatasetStorageQuotaError {
    state.poisoned = true;
    error
}

#[derive(Clone, Copy, Debug)]
struct FilesystemSnapshot {
    available_bytes: u64,
    fragment_bytes: u64,
    available_inodes: u64,
}

#[allow(unsafe_code)]
fn filesystem_snapshot(
    directory_fd: RawFd,
) -> Result<FilesystemSnapshot, DatasetStorageQuotaError> {
    let mut raw = std::mem::MaybeUninit::<libc::statvfs>::zeroed();
    // SAFETY: `raw` points to writable storage for `statvfs`; the fd is a
    // retained live directory descriptor owned by the quota.
    if unsafe { libc::fstatvfs(directory_fd, raw.as_mut_ptr()) } != 0 {
        return Err(DatasetStorageQuotaError::FilesystemInspect(
            io::Error::last_os_error(),
        ));
    }
    // SAFETY: a zero return from `fstatvfs` initialized the structure.
    let raw = unsafe { raw.assume_init() };
    let fragment_u128 = if raw.f_frsize == 0 {
        raw.f_bsize as u128
    } else {
        raw.f_frsize as u128
    };
    let fragment_bytes =
        u64::try_from(fragment_u128).map_err(|_| DatasetStorageQuotaError::ArithmeticOverflow)?;
    if fragment_bytes == 0 {
        return Err(DatasetStorageQuotaError::ZeroFilesystemFragment);
    }
    let available_u128 = (raw.f_bavail as u128)
        .checked_mul(fragment_u128)
        .ok_or(DatasetStorageQuotaError::ArithmeticOverflow)?;
    let available_bytes =
        u64::try_from(available_u128).map_err(|_| DatasetStorageQuotaError::ArithmeticOverflow)?;
    let available_inodes = u64::try_from(raw.f_favail as u128)
        .map_err(|_| DatasetStorageQuotaError::ArithmeticOverflow)?;
    Ok(FilesystemSnapshot {
        available_bytes,
        fragment_bytes,
        available_inodes,
    })
}

fn round_up(value: u64, fragment: u64) -> Result<u64, DatasetStorageQuotaError> {
    if value == 0 {
        return Ok(0);
    }
    value
        .checked_add(fragment - 1)
        .map(|sum| (sum / fragment) * fragment)
        .ok_or(DatasetStorageQuotaError::ArithmeticOverflow)
}

fn terminal_reserve_required(
    components: [u64; 3],
    fragment: u64,
) -> Result<u64, DatasetStorageQuotaError> {
    components.into_iter().try_fold(0_u64, |total, component| {
        total
            .checked_add(round_up(component, fragment)?)
            .ok_or(DatasetStorageQuotaError::ArithmeticOverflow)
    })
}

fn phase_physical_reserve(
    limits: DatasetStorageLimits,
    fragment: u64,
    phase: DatasetStoragePhase,
) -> Result<u64, DatasetStorageQuotaError> {
    match phase {
        DatasetStoragePhase::Live => round_up(limits.terminal_reserve_bytes(), fragment),
        DatasetStoragePhase::Terminal => [
            limits.terminal_component_maximum_bytes[0],
            limits.terminal_component_maximum_bytes[2],
        ]
        .into_iter()
        .try_fold(0_u64, |total, component| {
            total
                .checked_add(round_up(component, fragment)?)
                .ok_or(DatasetStorageQuotaError::ArithmeticOverflow)
        }),
    }
}

fn require_current_directory(
    path: &Path,
    expected: FilesystemIdentity,
    role: &'static str,
) -> Result<File, DatasetStorageQuotaError> {
    let current = open_directory_path_nofollow(path, role)?;
    let metadata = current
        .metadata()
        .map_err(|source| DatasetStorageQuotaError::Io {
            operation: "inspect current retained dataset directory",
            path: path.to_path_buf(),
            source,
        })?;
    let actual = FilesystemIdentity::from_metadata(&metadata);
    if !metadata.is_dir() || actual != expected {
        return Err(DatasetStorageQuotaError::DirectoryReplaced {
            role,
            path: path.to_path_buf(),
        });
    }
    Ok(current)
}

fn require_same_filesystem(
    dataset: FilesystemIdentity,
    frames: FilesystemIdentity,
) -> Result<(), DatasetStorageQuotaError> {
    if dataset.device != frames.device {
        Err(DatasetStorageQuotaError::FramesFilesystemMismatch {
            dataset_device: dataset.device,
            frames_device: frames.device,
        })
    } else {
        Ok(())
    }
}

fn open_directory_path_nofollow(
    path: &Path,
    role: &'static str,
) -> Result<File, DatasetStorageQuotaError> {
    let link_metadata =
        std::fs::symlink_metadata(path).map_err(|source| DatasetStorageQuotaError::Io {
            operation: "inspect dataset directory path without following links",
            path: path.to_path_buf(),
            source,
        })?;
    if link_metadata.file_type().is_symlink() {
        return Err(DatasetStorageQuotaError::DirectorySymlink {
            role,
            path: path.to_path_buf(),
        });
    }
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK)
        .open(path)
        .map_err(|source| DatasetStorageQuotaError::Io {
            operation: "open dataset directory without following links",
            path: path.to_path_buf(),
            source,
        })
}

#[allow(unsafe_code)]
fn descriptor_relative_create(directory_fd: RawFd, name: &OsStr) -> io::Result<File> {
    let name = CString::new(name.as_bytes()).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidInput, "dataset filename contains NUL")
    })?;
    // SAFETY: `directory_fd` is retained by the quota and `name` is a valid
    // NUL-terminated relative filename. A successful fd is uniquely owned.
    let fd = unsafe {
        libc::openat(
            directory_fd,
            name.as_ptr(),
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            0o600,
        )
    };
    if fd < 0 {
        Err(io::Error::last_os_error())
    } else {
        // SAFETY: successful `openat` returned one newly owned descriptor.
        Ok(unsafe { File::from_raw_fd(fd) })
    }
}

#[allow(unsafe_code)]
fn open_directory_at_nofollow(
    directory_fd: RawFd,
    name: &OsStr,
    path: &Path,
    role: &'static str,
) -> Result<File, DatasetStorageQuotaError> {
    let name = CString::new(name.as_bytes()).map_err(|_| {
        DatasetStorageQuotaError::InvalidRelativePath {
            path: path.to_path_buf(),
        }
    })?;
    // SAFETY: the parent descriptor is retained for the call, the child name
    // is one NUL-terminated component, and a successful fd is uniquely owned.
    let fd = unsafe {
        libc::openat(
            directory_fd,
            name.as_ptr(),
            libc::O_RDONLY
                | libc::O_DIRECTORY
                | libc::O_NOFOLLOW
                | libc::O_CLOEXEC
                | libc::O_NONBLOCK,
        )
    };
    if fd < 0 {
        return Err(DatasetStorageQuotaError::Io {
            operation: "open child dataset directory without following links",
            path: path.to_path_buf(),
            source: io::Error::last_os_error(),
        });
    }
    // SAFETY: successful `openat` returned one newly owned descriptor.
    let file = unsafe { File::from_raw_fd(fd) };
    let metadata = file
        .metadata()
        .map_err(|source| DatasetStorageQuotaError::Io {
            operation: "inspect child dataset directory",
            path: path.to_path_buf(),
            source,
        })?;
    if !metadata.is_dir() {
        return Err(DatasetStorageQuotaError::DirectoryReplaced {
            role,
            path: path.to_path_buf(),
        });
    }
    Ok(file)
}

#[allow(unsafe_code)]
fn descriptor_relative_rename(directory_fd: RawFd, from: &Path, to: &Path) -> io::Result<()> {
    let from = single_component_c_string(from)?;
    let to = single_component_c_string(to)?;
    // SAFETY: both names are valid relative C strings and the retained
    // directory descriptor remains alive for the call.
    if unsafe { libc::renameat(directory_fd, from.as_ptr(), directory_fd, to.as_ptr()) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

fn single_component_c_string(path: &Path) -> io::Result<CString> {
    match path.components().collect::<Vec<_>>().as_slice() {
        [Component::Normal(name)] => CString::new(name.as_bytes()).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "dataset filename contains NUL")
        }),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "dataset publication name must be one relative component",
        )),
    }
}

#[derive(Debug)]
pub enum DatasetStorageQuotaError {
    InvalidLimits {
        reason: &'static str,
    },
    InvalidDirectory,
    InvalidRelativePath {
        path: PathBuf,
    },
    DirectoryReplaced {
        role: &'static str,
        path: PathBuf,
    },
    DirectorySymlink {
        role: &'static str,
        path: PathBuf,
    },
    FramesFilesystemMismatch {
        dataset_device: u64,
        frames_device: u64,
    },
    DestinationAlreadyTracked {
        path: PathBuf,
    },
    UntrackedFile {
        path: PathBuf,
    },
    FileLengthChanged {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    LogicalByteLimitExceeded {
        attempted: u64,
        maximum: u64,
        phase: DatasetStoragePhase,
    },
    FileLimitExceeded {
        attempted: u64,
        maximum: u64,
    },
    InsufficientFreeSpace {
        available: u64,
        required: u64,
        fragment_bytes: u64,
        phase: DatasetStoragePhase,
    },
    PostWriteFreeSpaceRace {
        available: u64,
        required: u64,
        phase: DatasetStoragePhase,
    },
    InsufficientFreeInodes {
        available: u64,
        required: u64,
        phase: DatasetStoragePhase,
    },
    PostWriteInodeRace {
        available: u64,
        required: u64,
        phase: DatasetStoragePhase,
    },
    TerminalReserveTooSmallForFilesystem {
        configured: u64,
        required: u64,
        fragment_bytes: u64,
    },
    ArithmeticOverflow,
    ZeroFilesystemFragment,
    FilesystemInspect(io::Error),
    Io {
        operation: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    Poisoned,
}

impl std::fmt::Display for DatasetStorageQuotaError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "navigation dataset storage quota failed: {self:?}"
        )
    }
}

impl std::error::Error for DatasetStorageQuotaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::FilesystemInspect(source) | Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DatasetStorageLimits, DatasetStoragePhase, DatasetStorageQuota, DatasetStorageQuotaError,
        FilesystemIdentity, FilesystemSnapshot, require_same_filesystem, terminal_reserve_required,
        validate_post_write_headroom,
    };
    use std::path::Path;

    fn temporary_directory(name: &str) -> std::path::PathBuf {
        let root =
            std::env::temp_dir().join(format!("kiko-dataset-quota-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("frames")).expect("create quota fixture");
        root
    }

    #[test]
    fn exact_logical_fill_is_accepted_and_next_byte_poisons() {
        let root = temporary_directory("fill");
        let limits = DatasetStorageLimits::try_new(32_775, 4, 1, 32_768, 4_096, 4_096, 4_096)
            .expect("limits");
        let quota =
            DatasetStorageQuota::open(root.clone(), root.join("frames"), limits).expect("quota");
        quota
            .create_new_live_file(Path::new("meta.json"), &[7; 7])
            .expect("exact live ceiling");
        assert!(matches!(
            quota.create_new_live_file(Path::new("calibration.json"), &[8; 1]),
            Err(DatasetStorageQuotaError::LogicalByteLimitExceeded { .. })
        ));
        assert!(quota.is_poisoned());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn file_boundary_and_root_replacement_fail_closed() {
        let root = temporary_directory("replacement");
        let limits = DatasetStorageLimits::try_new(65_536, 2, 1, 12_288, 4_096, 4_096, 4_096)
            .expect("limits");
        let quota =
            DatasetStorageQuota::open(root.clone(), root.join("frames"), limits).expect("quota");
        quota
            .create_new_live_file(Path::new("meta.json"), &[1])
            .expect("first file");
        assert!(matches!(
            quota.create_new_live_file(Path::new("calibration.json"), &[2]),
            Err(DatasetStorageQuotaError::FileLimitExceeded { .. })
        ));
        let _ = std::fs::remove_dir_all(root);

        let root = temporary_directory("replacement-identity");
        let moved = root.with_extension("moved");
        let _ = std::fs::remove_dir_all(&moved);
        let limits = DatasetStorageLimits::try_new(65_536, 4, 1, 12_288, 4_096, 4_096, 4_096)
            .expect("limits");
        let quota =
            DatasetStorageQuota::open(root.clone(), root.join("frames"), limits).expect("quota");
        std::fs::rename(&root, &moved).expect("replace pathname");
        std::fs::create_dir_all(root.join("frames")).expect("replacement tree");
        assert!(matches!(
            quota.create_new_live_file(Path::new("meta.json"), &[1]),
            Err(DatasetStorageQuotaError::DirectoryReplaced { .. })
        ));
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(moved);

        let root = temporary_directory("symlink-alias");
        let moved = root.with_extension("retained");
        let _ = std::fs::remove_dir_all(&moved);
        let limits = DatasetStorageLimits::try_new(65_536, 4, 1, 12_288, 4_096, 4_096, 4_096)
            .expect("limits");
        let quota =
            DatasetStorageQuota::open(root.clone(), root.join("frames"), limits).expect("quota");
        std::fs::rename(&root, &moved).expect("move retained directory");
        std::os::unix::fs::symlink(&moved, &root).expect("alias retained directory");
        assert!(matches!(
            quota.create_new_live_file(Path::new("meta.json"), &[1]),
            Err(DatasetStorageQuotaError::DirectorySymlink { .. })
        ));
        std::fs::remove_file(root).expect("remove alias");
        std::fs::remove_dir_all(moved).expect("remove retained directory");
    }

    #[test]
    fn terminal_reserve_uses_actual_larger_fragment_and_checked_arithmetic() {
        assert_eq!(
            terminal_reserve_required([1, 65_537, 4_096], 65_536).expect("rounded reserve"),
            4 * 65_536
        );
        assert!(matches!(
            terminal_reserve_required([u64::MAX, 1, 1], 4_096),
            Err(DatasetStorageQuotaError::ArithmeticOverflow)
        ));
    }

    #[test]
    fn frames_directory_must_share_the_quota_filesystem() {
        assert!(
            require_same_filesystem(
                FilesystemIdentity {
                    device: 7,
                    inode: 1,
                },
                FilesystemIdentity {
                    device: 7,
                    inode: 2,
                },
            )
            .is_ok()
        );
        assert!(matches!(
            require_same_filesystem(
                FilesystemIdentity {
                    device: 7,
                    inode: 1,
                },
                FilesystemIdentity {
                    device: 8,
                    inode: 2,
                },
            ),
            Err(DatasetStorageQuotaError::FramesFilesystemMismatch {
                dataset_device: 7,
                frames_device: 8,
            })
        ));
    }

    #[test]
    fn post_write_free_block_and_inode_races_are_typed() {
        let limits = DatasetStorageLimits::try_new(65_536, 4, 10, 12_288, 4_096, 4_096, 4_096)
            .expect("limits");
        assert!(matches!(
            validate_post_write_headroom(
                FilesystemSnapshot {
                    available_bytes: 12_297,
                    fragment_bytes: 4_096,
                    available_inodes: 3,
                },
                limits,
                DatasetStoragePhase::Live,
            ),
            Err(DatasetStorageQuotaError::PostWriteFreeSpaceRace {
                available: 12_297,
                required: 12_298,
                ..
            })
        ));
        assert!(matches!(
            validate_post_write_headroom(
                FilesystemSnapshot {
                    available_bytes: 12_298,
                    fragment_bytes: 4_096,
                    available_inodes: 2,
                },
                limits,
                DatasetStoragePhase::Live,
            ),
            Err(DatasetStorageQuotaError::PostWriteInodeRace {
                available: 2,
                required: 3,
                ..
            })
        ));
        assert!(matches!(
            validate_post_write_headroom(
                FilesystemSnapshot {
                    available_bytes: 8_201,
                    fragment_bytes: 4_096,
                    available_inodes: 2,
                },
                limits,
                DatasetStoragePhase::Terminal,
            ),
            Err(DatasetStorageQuotaError::PostWriteFreeSpaceRace {
                available: 8_201,
                required: 8_202,
                phase: DatasetStoragePhase::Terminal,
            })
        ));
    }
}
