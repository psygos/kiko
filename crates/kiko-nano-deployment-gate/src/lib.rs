//! Root-owned, content-bound boot gate for one qualified Nano installation.
//!
//! This crate deliberately has no dependency on the camera, inference,
//! controller, or SLAM stacks. The offline qualifier uses their typed parsers
//! to mint one exact marker; this small verifier can then reject install drift
//! before the production binary and its deployment-native libraries start.

#![cfg(unix)]
#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const QUALIFICATION_SCHEMA_VERSION: u32 = 1;
pub const QUALIFICATION_SCOPE: &str = "offline_install_only_not_hardware_qualification_v1";
pub const QUALIFICATION_ACKNOWLEDGEMENT: &str =
    "I reviewed this exact offline install; hardware and wheels-off gates remain separate";
pub const DEFAULT_QUALIFICATION_MARKER: &str =
    "/etc/kiko/nano-agent-offline-install-qualified-v1.json";
pub const ROOT_UID: u32 = 0;
pub const ROOT_GID: u32 = 0;
pub const MARKER_MODE: u32 = 0o400;
pub const MARKER_PARENT_MODE: u32 = 0o755;
pub const MAX_MARKER_BYTES: u64 = 256 * 1_024;
pub const MAX_BOUND_FILE_BYTES: u64 = 512 * 1_024 * 1_024;
const MAX_QUALIFIED_FILES: usize = 96;
const MAX_BINDINGS: usize = 192;
const MAX_ROLE_BYTES: usize = 256;
const FILESYSTEM_TRUST_ROOT: &str = "/";

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct QualifiedFileV1 {
    path: PathBuf,
    mode: u32,
    byte_len: u64,
    sha256: [u8; 32],
}

impl QualifiedFileV1 {
    pub fn inspect(
        path: PathBuf,
        required_uid: u32,
        required_gid: u32,
    ) -> Result<Self, DeploymentGateError> {
        Self::inspect_with_required_mode(path, required_uid, required_gid, None)
    }

    pub fn inspect_with_required_mode(
        path: PathBuf,
        required_uid: u32,
        required_gid: u32,
        required_mode: Option<u32>,
    ) -> Result<Self, DeploymentGateError> {
        Self::inspect_with_required_mode_beneath(
            path,
            required_uid,
            required_gid,
            required_mode,
            Path::new(FILESYSTEM_TRUST_ROOT),
        )
    }

    fn inspect_with_required_mode_beneath(
        path: PathBuf,
        required_uid: u32,
        required_gid: u32,
        required_mode: Option<u32>,
        trust_root: &Path,
    ) -> Result<Self, DeploymentGateError> {
        validate_canonical_absolute_path(&path)?;
        let observed = inspect_trusted_regular_file_beneath(
            &path,
            required_uid,
            required_gid,
            required_mode,
            MAX_BOUND_FILE_BYTES,
            trust_root,
        )?;
        Ok(Self {
            path,
            mode: observed.mode,
            byte_len: observed.byte_len,
            sha256: observed.sha256,
        })
    }

    pub fn from_retained_bytes(
        path: PathBuf,
        bytes: &[u8],
        sha256: [u8; 32],
        required_uid: u32,
        required_gid: u32,
    ) -> Result<Self, DeploymentGateError> {
        Self::from_retained_bytes_beneath(
            path,
            bytes,
            sha256,
            required_uid,
            required_gid,
            Path::new(FILESYSTEM_TRUST_ROOT),
        )
    }

    fn from_retained_bytes_beneath(
        path: PathBuf,
        bytes: &[u8],
        sha256: [u8; 32],
        required_uid: u32,
        required_gid: u32,
        trust_root: &Path,
    ) -> Result<Self, DeploymentGateError> {
        validate_canonical_absolute_path(&path)?;
        let expected_bytes = u64::try_from(bytes.len()).map_err(|_| {
            DeploymentGateError::FileSizeNotRepresentable {
                path: path.clone(),
                bytes: bytes.len(),
            }
        })?;
        let observed = inspect_trusted_regular_file_beneath(
            &path,
            required_uid,
            required_gid,
            None,
            MAX_BOUND_FILE_BYTES,
            trust_root,
        )?;
        if observed.byte_len != expected_bytes || observed.sha256 != sha256 {
            return Err(DeploymentGateError::RetainedFileMismatch {
                path,
                retained_bytes: expected_bytes,
                observed_bytes: observed.byte_len,
                retained_sha256: sha256,
                observed_sha256: observed.sha256,
            });
        }
        Ok(Self {
            path,
            mode: observed.mode,
            byte_len: observed.byte_len,
            sha256: observed.sha256,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub const fn mode(&self) -> u32 {
        self.mode
    }

    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    pub const fn sha256(&self) -> &[u8; 32] {
        &self.sha256
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct QualifiedFileBindingV1 {
    role: String,
    path: PathBuf,
}

impl QualifiedFileBindingV1 {
    pub fn new(role: String, path: PathBuf) -> Result<Self, DeploymentGateError> {
        validate_role(&role)?;
        validate_canonical_absolute_path(&path)?;
        Ok(Self { role, path })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OfflineInstallQualificationV1 {
    schema_version: u32,
    qualification_scope: String,
    deployment_root: PathBuf,
    state_root: PathBuf,
    launch_relative_path: String,
    native_runtime_manifest_relative_path: String,
    native_library_search_relative_path: String,
    files: Vec<QualifiedFileV1>,
    bindings: Vec<QualifiedFileBindingV1>,
}

impl OfflineInstallQualificationV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        deployment_root: PathBuf,
        state_root: PathBuf,
        launch_relative_path: String,
        native_runtime_manifest_relative_path: String,
        native_library_search_relative_path: String,
        mut files: Vec<QualifiedFileV1>,
        mut bindings: Vec<QualifiedFileBindingV1>,
    ) -> Result<Self, DeploymentGateError> {
        validate_canonical_absolute_path(&deployment_root)?;
        validate_canonical_absolute_path(&state_root)?;
        validate_canonical_relative_path(&launch_relative_path)?;
        validate_canonical_relative_path(&native_runtime_manifest_relative_path)?;
        validate_canonical_relative_path(&native_library_search_relative_path)?;
        files.sort_by(|left, right| left.path.cmp(&right.path));
        bindings.sort_by(|left, right| left.role.cmp(&right.role));
        let marker = Self {
            schema_version: QUALIFICATION_SCHEMA_VERSION,
            qualification_scope: QUALIFICATION_SCOPE.to_owned(),
            deployment_root,
            state_root,
            launch_relative_path,
            native_runtime_manifest_relative_path,
            native_library_search_relative_path,
            files,
            bindings,
        };
        marker.validate_structure()?;
        Ok(marker)
    }

    pub fn files(&self) -> &[QualifiedFileV1] {
        &self.files
    }

    fn validate_structure(&self) -> Result<(), DeploymentGateError> {
        if self.schema_version != QUALIFICATION_SCHEMA_VERSION {
            return Err(DeploymentGateError::UnsupportedSchema {
                actual: self.schema_version,
                supported: QUALIFICATION_SCHEMA_VERSION,
            });
        }
        if self.qualification_scope != QUALIFICATION_SCOPE {
            return Err(DeploymentGateError::WrongQualificationScope {
                actual: self.qualification_scope.clone(),
            });
        }
        validate_canonical_absolute_path(&self.deployment_root)?;
        validate_canonical_absolute_path(&self.state_root)?;
        if self.deployment_root.starts_with(&self.state_root)
            || self.state_root.starts_with(&self.deployment_root)
        {
            return Err(DeploymentGateError::OverlappingRoots {
                deployment_root: self.deployment_root.clone(),
                state_root: self.state_root.clone(),
            });
        }
        validate_canonical_relative_path(&self.launch_relative_path)?;
        validate_canonical_relative_path(&self.native_runtime_manifest_relative_path)?;
        validate_canonical_relative_path(&self.native_library_search_relative_path)?;
        if self.files.is_empty() || self.files.len() > MAX_QUALIFIED_FILES {
            return Err(DeploymentGateError::InvalidFileCount {
                actual: self.files.len(),
                maximum: MAX_QUALIFIED_FILES,
            });
        }
        if self.bindings.is_empty() || self.bindings.len() > MAX_BINDINGS {
            return Err(DeploymentGateError::InvalidBindingCount {
                actual: self.bindings.len(),
                maximum: MAX_BINDINGS,
            });
        }

        let mut prior_path: Option<&Path> = None;
        let mut known_paths = BTreeSet::new();
        for file in &self.files {
            validate_canonical_absolute_path(&file.path)?;
            if file.byte_len > MAX_BOUND_FILE_BYTES {
                return Err(DeploymentGateError::FileTooLarge {
                    path: file.path.clone(),
                    actual_bytes: file.byte_len,
                    maximum_bytes: MAX_BOUND_FILE_BYTES,
                });
            }
            if file.mode & 0o022 != 0 {
                return Err(DeploymentGateError::WritableQualifiedFileMode {
                    path: file.path.clone(),
                    mode: file.mode,
                });
            }
            if prior_path.is_some_and(|prior| prior >= file.path.as_path()) {
                return Err(DeploymentGateError::FilesNotStrictlySorted);
            }
            prior_path = Some(&file.path);
            known_paths.insert(file.path.clone());
        }

        let mut prior_role: Option<&str> = None;
        for binding in &self.bindings {
            validate_role(&binding.role)?;
            validate_canonical_absolute_path(&binding.path)?;
            if prior_role.is_some_and(|prior| prior >= binding.role.as_str()) {
                return Err(DeploymentGateError::BindingsNotStrictlySorted);
            }
            if !known_paths.contains(&binding.path) {
                return Err(DeploymentGateError::BindingReferencesUnknownFile {
                    role: binding.role.clone(),
                    path: binding.path.clone(),
                });
            }
            prior_role = Some(&binding.role);
        }
        for file in &self.files {
            if !self
                .bindings
                .iter()
                .any(|binding| binding.path == file.path)
            {
                return Err(DeploymentGateError::UnboundFile {
                    path: file.path.clone(),
                });
            }
        }
        Ok(())
    }
}

pub fn write_qualification_marker(
    marker_path: &Path,
    marker: &OfflineInstallQualificationV1,
    required_uid: u32,
    required_gid: u32,
) -> Result<(), DeploymentGateError> {
    write_qualification_marker_beneath(
        marker_path,
        marker,
        required_uid,
        required_gid,
        Path::new(FILESYSTEM_TRUST_ROOT),
    )
}

fn write_qualification_marker_beneath(
    marker_path: &Path,
    marker: &OfflineInstallQualificationV1,
    required_uid: u32,
    required_gid: u32,
    trust_root: &Path,
) -> Result<(), DeploymentGateError> {
    marker.validate_structure()?;
    validate_canonical_absolute_path(marker_path)?;
    let parent = marker_path
        .parent()
        .ok_or_else(|| DeploymentGateError::MarkerHasNoParent {
            path: marker_path.to_path_buf(),
        })?;
    require_trusted_directory_chain_beneath(
        parent,
        required_uid,
        required_gid,
        Some(MARKER_PARENT_MODE),
        trust_root,
    )?;
    match fs::symlink_metadata(marker_path) {
        Ok(metadata) => {
            require_trusted_regular_metadata(
                marker_path,
                &metadata,
                required_uid,
                required_gid,
                Some(MARKER_MODE),
            )?;
        }
        Err(source) if source.kind() == io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(DeploymentGateError::Metadata {
                path: marker_path.to_path_buf(),
                source,
            });
        }
    }

    let mut bytes = serde_json::to_vec_pretty(marker)
        .map_err(|source| DeploymentGateError::MarkerEncode { source })?;
    bytes.push(b'\n');
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_MARKER_BYTES {
        return Err(DeploymentGateError::MarkerTooLarge {
            actual_bytes: bytes.len(),
            maximum_bytes: MAX_MARKER_BYTES,
        });
    }

    let file_name = marker_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| DeploymentGateError::InvalidMarkerFileName {
            path: marker_path.to_path_buf(),
        })?;
    let mut selected = None;
    for attempt in 0_u8..16 {
        let candidate = parent.join(format!(".{file_name}.{}.{attempt}.tmp", std::process::id()));
        let mut options = OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .mode(MARKER_MODE)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
        match options.open(&candidate) {
            Ok(file) => {
                selected = Some((candidate, file));
                break;
            }
            Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {}
            Err(source) => {
                return Err(DeploymentGateError::MarkerTempOpen {
                    path: candidate,
                    source,
                });
            }
        }
    }
    let (temporary, mut temporary_file) =
        selected.ok_or_else(|| DeploymentGateError::MarkerTempNamesExhausted {
            parent: parent.to_path_buf(),
        })?;
    let result = (|| {
        temporary_file
            .write_all(&bytes)
            .map_err(|source| DeploymentGateError::MarkerWrite {
                path: temporary.clone(),
                source,
            })?;
        temporary_file
            .sync_all()
            .map_err(|source| DeploymentGateError::MarkerSync {
                path: temporary.clone(),
                source,
            })?;
        let metadata =
            temporary_file
                .metadata()
                .map_err(|source| DeploymentGateError::Metadata {
                    path: temporary.clone(),
                    source,
                })?;
        require_trusted_regular_metadata(
            &temporary,
            &metadata,
            required_uid,
            required_gid,
            Some(MARKER_MODE),
        )?;
        drop(temporary_file);
        fs::rename(&temporary, marker_path).map_err(|source| {
            DeploymentGateError::MarkerPublish {
                temporary: temporary.clone(),
                marker: marker_path.to_path_buf(),
                source,
            }
        })?;
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|source| DeploymentGateError::MarkerParentSync {
                path: parent.to_path_buf(),
                source,
            })
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

pub fn verify_qualification_marker(
    marker_path: &Path,
    required_uid: u32,
    required_gid: u32,
) -> Result<OfflineInstallQualificationV1, DeploymentGateError> {
    verify_qualification_marker_beneath(
        marker_path,
        required_uid,
        required_gid,
        Path::new(FILESYSTEM_TRUST_ROOT),
    )
}

fn verify_qualification_marker_beneath(
    marker_path: &Path,
    required_uid: u32,
    required_gid: u32,
    trust_root: &Path,
) -> Result<OfflineInstallQualificationV1, DeploymentGateError> {
    let marker =
        read_qualification_marker_beneath(marker_path, required_uid, required_gid, trust_root)?;
    marker.validate_structure()?;
    require_trusted_directory_chain_beneath(
        &marker.deployment_root,
        required_uid,
        required_gid,
        None,
        trust_root,
    )?;
    for expected in &marker.files {
        let observed = inspect_trusted_regular_file_beneath(
            &expected.path,
            required_uid,
            required_gid,
            Some(expected.mode),
            MAX_BOUND_FILE_BYTES,
            trust_root,
        )?;
        if observed.byte_len != expected.byte_len || observed.sha256 != expected.sha256 {
            return Err(DeploymentGateError::QualifiedFileChanged {
                path: expected.path.clone(),
                expected_bytes: expected.byte_len,
                observed_bytes: observed.byte_len,
                expected_sha256: expected.sha256,
                observed_sha256: observed.sha256,
            });
        }
    }
    Ok(marker)
}

fn read_qualification_marker_beneath(
    marker_path: &Path,
    required_uid: u32,
    required_gid: u32,
    trust_root: &Path,
) -> Result<OfflineInstallQualificationV1, DeploymentGateError> {
    validate_canonical_absolute_path(marker_path)?;
    let parent = marker_path
        .parent()
        .ok_or_else(|| DeploymentGateError::MarkerHasNoParent {
            path: marker_path.to_path_buf(),
        })?;
    require_trusted_directory_chain_beneath(
        parent,
        required_uid,
        required_gid,
        Some(MARKER_PARENT_MODE),
        trust_root,
    )?;
    let metadata =
        fs::symlink_metadata(marker_path).map_err(|source| DeploymentGateError::Metadata {
            path: marker_path.to_path_buf(),
            source,
        })?;
    require_trusted_regular_metadata(
        marker_path,
        &metadata,
        required_uid,
        required_gid,
        Some(MARKER_MODE),
    )?;
    if metadata.len() > MAX_MARKER_BYTES {
        return Err(DeploymentGateError::MarkerTooLarge {
            actual_bytes: usize::try_from(metadata.len()).unwrap_or(usize::MAX),
            maximum_bytes: MAX_MARKER_BYTES,
        });
    }
    let mut options = OpenOptions::new();
    options
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    let file = options
        .open(marker_path)
        .map_err(|source| DeploymentGateError::FileOpen {
            path: marker_path.to_path_buf(),
            source,
        })?;
    let opened = file
        .metadata()
        .map_err(|source| DeploymentGateError::Metadata {
            path: marker_path.to_path_buf(),
            source,
        })?;
    require_trusted_regular_metadata(
        marker_path,
        &opened,
        required_uid,
        required_gid,
        Some(MARKER_MODE),
    )?;
    require_same_opened_file(marker_path, &metadata, &opened)?;
    let capacity = usize::try_from(metadata.len()).map_err(|_| {
        DeploymentGateError::DeclaredSizeNotRepresentable {
            path: marker_path.to_path_buf(),
            declared_bytes: metadata.len(),
        }
    })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|source| DeploymentGateError::Allocation {
            path: marker_path.to_path_buf(),
            requested_bytes: capacity,
            source,
        })?;
    file.take(MAX_MARKER_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|source| DeploymentGateError::FileRead {
            path: marker_path.to_path_buf(),
            source,
        })?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) != metadata.len() {
        return Err(DeploymentGateError::FileLengthChanged {
            path: marker_path.to_path_buf(),
            declared_bytes: metadata.len(),
            observed_bytes: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        });
    }
    let mut deserializer = serde_json::Deserializer::from_slice(&bytes);
    let marker = OfflineInstallQualificationV1::deserialize(&mut deserializer)
        .map_err(|source| DeploymentGateError::MarkerDecode { source })?;
    deserializer
        .end()
        .map_err(|source| DeploymentGateError::MarkerTrailingData { source })?;
    Ok(marker)
}

#[derive(Clone, Copy)]
struct ObservedFile {
    mode: u32,
    byte_len: u64,
    sha256: [u8; 32],
}

fn inspect_trusted_regular_file_beneath(
    path: &Path,
    required_uid: u32,
    required_gid: u32,
    required_mode: Option<u32>,
    maximum_bytes: u64,
    trust_root: &Path,
) -> Result<ObservedFile, DeploymentGateError> {
    let parent = path
        .parent()
        .ok_or_else(|| DeploymentGateError::FileHasNoParent {
            path: path.to_path_buf(),
        })?;
    require_trusted_directory_chain_beneath(parent, required_uid, required_gid, None, trust_root)?;
    let metadata = fs::symlink_metadata(path).map_err(|source| DeploymentGateError::Metadata {
        path: path.to_path_buf(),
        source,
    })?;
    require_trusted_regular_metadata(path, &metadata, required_uid, required_gid, required_mode)?;
    if metadata.len() > maximum_bytes {
        return Err(DeploymentGateError::FileTooLarge {
            path: path.to_path_buf(),
            actual_bytes: metadata.len(),
            maximum_bytes,
        });
    }
    let mut options = OpenOptions::new();
    options
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    let mut file = options
        .open(path)
        .map_err(|source| DeploymentGateError::FileOpen {
            path: path.to_path_buf(),
            source,
        })?;
    let opened = file
        .metadata()
        .map_err(|source| DeploymentGateError::Metadata {
            path: path.to_path_buf(),
            source,
        })?;
    require_trusted_regular_metadata(path, &opened, required_uid, required_gid, required_mode)?;
    require_same_opened_file(path, &metadata, &opened)?;
    if opened.len() != metadata.len() {
        return Err(DeploymentGateError::FileLengthChanged {
            path: path.to_path_buf(),
            declared_bytes: metadata.len(),
            observed_bytes: opened.len(),
        });
    }

    let mut hasher = Sha256::new();
    let mut observed_bytes = 0_u64;
    let mut buffer = [0_u8; 32 * 1_024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| DeploymentGateError::FileRead {
                path: path.to_path_buf(),
                source,
            })?;
        if read == 0 {
            break;
        }
        observed_bytes = observed_bytes
            .checked_add(u64::try_from(read).expect("buffer read length fits u64"))
            .ok_or_else(|| DeploymentGateError::ObservedLengthOverflow {
                path: path.to_path_buf(),
            })?;
        if observed_bytes > maximum_bytes || observed_bytes > metadata.len() {
            return Err(DeploymentGateError::FileGrewDuringRead {
                path: path.to_path_buf(),
                declared_bytes: metadata.len(),
                observed_bytes,
            });
        }
        hasher.update(&buffer[..read]);
    }
    let final_metadata = file
        .metadata()
        .map_err(|source| DeploymentGateError::Metadata {
            path: path.to_path_buf(),
            source,
        })?;
    require_same_opened_file(path, &opened, &final_metadata)?;
    if observed_bytes != metadata.len() || final_metadata.len() != metadata.len() {
        return Err(DeploymentGateError::FileLengthChanged {
            path: path.to_path_buf(),
            declared_bytes: metadata.len(),
            observed_bytes,
        });
    }
    Ok(ObservedFile {
        mode: metadata.permissions().mode() & 0o7777,
        byte_len: observed_bytes,
        sha256: hasher.finalize().into(),
    })
}

fn require_same_opened_file(
    path: &Path,
    expected: &fs::Metadata,
    observed: &fs::Metadata,
) -> Result<(), DeploymentGateError> {
    if expected.dev() != observed.dev() || expected.ino() != observed.ino() {
        return Err(DeploymentGateError::FileIdentityChanged {
            path: path.to_path_buf(),
            expected_device: expected.dev(),
            expected_inode: expected.ino(),
            observed_device: observed.dev(),
            observed_inode: observed.ino(),
        });
    }
    Ok(())
}

fn require_trusted_regular_metadata(
    path: &Path,
    metadata: &fs::Metadata,
    required_uid: u32,
    required_gid: u32,
    required_mode: Option<u32>,
) -> Result<(), DeploymentGateError> {
    if !metadata.file_type().is_file() {
        return Err(DeploymentGateError::NotRegularFile {
            path: path.to_path_buf(),
        });
    }
    let mode = metadata.permissions().mode() & 0o7777;
    if metadata.uid() != required_uid || metadata.gid() != required_gid {
        return Err(DeploymentGateError::WrongOwner {
            path: path.to_path_buf(),
            required_uid,
            required_gid,
            actual_uid: metadata.uid(),
            actual_gid: metadata.gid(),
        });
    }
    if let Some(required) = required_mode
        && mode != required
    {
        return Err(DeploymentGateError::WrongMode {
            path: path.to_path_buf(),
            required,
            actual: mode,
        });
    }
    if mode & 0o022 != 0 {
        return Err(DeploymentGateError::WritableQualifiedFileMode {
            path: path.to_path_buf(),
            mode,
        });
    }
    if metadata.nlink() != 1 {
        return Err(DeploymentGateError::UnexpectedLinkCount {
            path: path.to_path_buf(),
            actual: metadata.nlink(),
        });
    }
    Ok(())
}

fn require_trusted_directory_entry(
    path: &Path,
    required_uid: u32,
    required_gid: u32,
    required_mode: Option<u32>,
) -> Result<(), DeploymentGateError> {
    validate_absolute_directory_path(path)?;
    let metadata = fs::symlink_metadata(path).map_err(|source| DeploymentGateError::Metadata {
        path: path.to_path_buf(),
        source,
    })?;
    if !metadata.file_type().is_dir() {
        return Err(DeploymentGateError::NotDirectory {
            path: path.to_path_buf(),
        });
    }
    let mode = metadata.permissions().mode() & 0o7777;
    if metadata.uid() != required_uid || metadata.gid() != required_gid {
        return Err(DeploymentGateError::WrongOwner {
            path: path.to_path_buf(),
            required_uid,
            required_gid,
            actual_uid: metadata.uid(),
            actual_gid: metadata.gid(),
        });
    }
    if let Some(required) = required_mode
        && mode != required
    {
        return Err(DeploymentGateError::WrongMode {
            path: path.to_path_buf(),
            required,
            actual: mode,
        });
    }
    if mode & 0o022 != 0 {
        return Err(DeploymentGateError::WritableQualifiedDirectoryMode {
            path: path.to_path_buf(),
            mode,
        });
    }
    Ok(())
}

fn require_trusted_directory_chain_beneath(
    path: &Path,
    required_uid: u32,
    required_gid: u32,
    required_final_mode: Option<u32>,
    trust_root: &Path,
) -> Result<(), DeploymentGateError> {
    validate_absolute_directory_path(path)?;
    validate_absolute_directory_path(trust_root)?;
    let relative =
        path.strip_prefix(trust_root)
            .map_err(|_| DeploymentGateError::PathOutsideTrustRoot {
                path: path.to_path_buf(),
                trust_root: trust_root.to_path_buf(),
            })?;

    require_trusted_directory_entry(
        trust_root,
        required_uid,
        required_gid,
        if relative.as_os_str().is_empty() {
            required_final_mode
        } else {
            None
        },
    )?;
    let mut current = trust_root.to_path_buf();
    for component in relative.components() {
        let Component::Normal(component) = component else {
            return Err(DeploymentGateError::InvalidAbsolutePath {
                path: path.to_path_buf(),
            });
        };
        current.push(component);
        let is_final = current == path;
        require_trusted_directory_entry(
            &current,
            required_uid,
            required_gid,
            is_final.then_some(required_final_mode).flatten(),
        )?;
    }
    Ok(())
}

fn validate_role(role: &str) -> Result<(), DeploymentGateError> {
    if role.is_empty()
        || role.len() > MAX_ROLE_BYTES
        || !role
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b':' | b'.'))
    {
        return Err(DeploymentGateError::InvalidRole {
            role: role.to_owned(),
        });
    }
    Ok(())
}

fn validate_canonical_absolute_path(path: &Path) -> Result<(), DeploymentGateError> {
    if !path.is_absolute() || path == Path::new("/") {
        return Err(DeploymentGateError::InvalidAbsolutePath {
            path: path.to_path_buf(),
        });
    }
    if path
        .components()
        .any(|component| !matches!(component, Component::RootDir | Component::Normal(_)))
    {
        return Err(DeploymentGateError::InvalidAbsolutePath {
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

fn validate_absolute_directory_path(path: &Path) -> Result<(), DeploymentGateError> {
    if !path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::RootDir | Component::Normal(_)))
    {
        return Err(DeploymentGateError::InvalidAbsolutePath {
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

fn validate_canonical_relative_path(path: &str) -> Result<(), DeploymentGateError> {
    let parsed = Path::new(path);
    if path.is_empty()
        || path.ends_with('/')
        || path.contains("//")
        || parsed.is_absolute()
        || parsed
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(DeploymentGateError::InvalidRelativePath {
            path: path.to_owned(),
        });
    }
    Ok(())
}

#[derive(Debug)]
pub enum DeploymentGateError {
    UnsupportedSchema {
        actual: u32,
        supported: u32,
    },
    WrongQualificationScope {
        actual: String,
    },
    InvalidAbsolutePath {
        path: PathBuf,
    },
    InvalidRelativePath {
        path: String,
    },
    OverlappingRoots {
        deployment_root: PathBuf,
        state_root: PathBuf,
    },
    InvalidFileCount {
        actual: usize,
        maximum: usize,
    },
    InvalidBindingCount {
        actual: usize,
        maximum: usize,
    },
    InvalidRole {
        role: String,
    },
    FilesNotStrictlySorted,
    BindingsNotStrictlySorted,
    BindingReferencesUnknownFile {
        role: String,
        path: PathBuf,
    },
    UnboundFile {
        path: PathBuf,
    },
    WritableQualifiedFileMode {
        path: PathBuf,
        mode: u32,
    },
    WritableQualifiedDirectoryMode {
        path: PathBuf,
        mode: u32,
    },
    FileTooLarge {
        path: PathBuf,
        actual_bytes: u64,
        maximum_bytes: u64,
    },
    FileSizeNotRepresentable {
        path: PathBuf,
        bytes: usize,
    },
    RetainedFileMismatch {
        path: PathBuf,
        retained_bytes: u64,
        observed_bytes: u64,
        retained_sha256: [u8; 32],
        observed_sha256: [u8; 32],
    },
    MarkerHasNoParent {
        path: PathBuf,
    },
    FileHasNoParent {
        path: PathBuf,
    },
    PathOutsideTrustRoot {
        path: PathBuf,
        trust_root: PathBuf,
    },
    InvalidMarkerFileName {
        path: PathBuf,
    },
    MarkerTooLarge {
        actual_bytes: usize,
        maximum_bytes: u64,
    },
    MarkerEncode {
        source: serde_json::Error,
    },
    MarkerDecode {
        source: serde_json::Error,
    },
    MarkerTrailingData {
        source: serde_json::Error,
    },
    MarkerTempOpen {
        path: PathBuf,
        source: io::Error,
    },
    MarkerTempNamesExhausted {
        parent: PathBuf,
    },
    MarkerWrite {
        path: PathBuf,
        source: io::Error,
    },
    MarkerSync {
        path: PathBuf,
        source: io::Error,
    },
    MarkerPublish {
        temporary: PathBuf,
        marker: PathBuf,
        source: io::Error,
    },
    MarkerParentSync {
        path: PathBuf,
        source: io::Error,
    },
    Metadata {
        path: PathBuf,
        source: io::Error,
    },
    FileOpen {
        path: PathBuf,
        source: io::Error,
    },
    FileRead {
        path: PathBuf,
        source: io::Error,
    },
    Allocation {
        path: PathBuf,
        requested_bytes: usize,
        source: std::collections::TryReserveError,
    },
    NotRegularFile {
        path: PathBuf,
    },
    NotDirectory {
        path: PathBuf,
    },
    WrongOwner {
        path: PathBuf,
        required_uid: u32,
        required_gid: u32,
        actual_uid: u32,
        actual_gid: u32,
    },
    WrongMode {
        path: PathBuf,
        required: u32,
        actual: u32,
    },
    UnexpectedLinkCount {
        path: PathBuf,
        actual: u64,
    },
    DeclaredSizeNotRepresentable {
        path: PathBuf,
        declared_bytes: u64,
    },
    ObservedLengthOverflow {
        path: PathBuf,
    },
    FileGrewDuringRead {
        path: PathBuf,
        declared_bytes: u64,
        observed_bytes: u64,
    },
    FileLengthChanged {
        path: PathBuf,
        declared_bytes: u64,
        observed_bytes: u64,
    },
    FileIdentityChanged {
        path: PathBuf,
        expected_device: u64,
        expected_inode: u64,
        observed_device: u64,
        observed_inode: u64,
    },
    QualifiedFileChanged {
        path: PathBuf,
        expected_bytes: u64,
        observed_bytes: u64,
        expected_sha256: [u8; 32],
        observed_sha256: [u8; 32],
    },
}

impl fmt::Display for DeploymentGateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Nano offline-install gate rejected input: {self:?}"
        )
    }
}

impl std::error::Error for DeploymentGateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::MarkerEncode { source }
            | Self::MarkerDecode { source }
            | Self::MarkerTrailingData { source } => Some(source),
            Self::MarkerTempOpen { source, .. }
            | Self::MarkerWrite { source, .. }
            | Self::MarkerSync { source, .. }
            | Self::MarkerPublish { source, .. }
            | Self::MarkerParentSync { source, .. }
            | Self::Metadata { source, .. }
            | Self::FileOpen { source, .. }
            | Self::FileRead { source, .. } => Some(source),
            Self::Allocation { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    struct TestRoot {
        path: PathBuf,
        uid: u32,
        gid: u32,
    }

    impl TestRoot {
        fn new(name: &str) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("test clock after epoch")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "kiko-nano-deployment-gate-{name}-{}-{nonce}",
                std::process::id()
            ));
            fs::create_dir(&path).expect("create test root");
            fs::set_permissions(&path, fs::Permissions::from_mode(MARKER_PARENT_MODE))
                .expect("set test root mode");
            let metadata = fs::metadata(&path).expect("test root metadata");
            Self {
                path,
                uid: metadata.uid(),
                gid: metadata.gid(),
            }
        }

        fn file(&self, name: &str, bytes: &[u8]) -> PathBuf {
            let path = self.path.join(name);
            fs::write(&path, bytes).expect("write test file");
            fs::set_permissions(&path, fs::Permissions::from_mode(0o444))
                .expect("set test file mode");
            path
        }
    }

    impl Drop for TestRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn marker(root: &TestRoot, file: &Path) -> OfflineInstallQualificationV1 {
        let deployment_root = root.path.join("deployment");
        fs::create_dir_all(&deployment_root).expect("create deployment root");
        fs::set_permissions(
            &deployment_root,
            fs::Permissions::from_mode(MARKER_PARENT_MODE),
        )
        .expect("set deployment mode");
        let qualified = QualifiedFileV1::inspect_with_required_mode_beneath(
            file.to_path_buf(),
            root.uid,
            root.gid,
            None,
            &root.path,
        )
        .expect("inspect");
        OfflineInstallQualificationV1::try_new(
            deployment_root,
            root.path.join("state"),
            "launch.json".into(),
            "native-runtime-v1.json".into(),
            "lib".into(),
            vec![qualified],
            vec![
                QualifiedFileBindingV1::new("fixture".into(), file.to_path_buf()).expect("binding"),
            ],
        )
        .expect("marker")
    }

    #[test]
    fn marker_round_trip_is_strict_and_file_drift_fails_closed() {
        let root = TestRoot::new("round-trip");
        let file = root.file("bound", b"version one");
        let marker_path = root.path.join("qualification.json");
        let expected = marker(&root, &file);
        assert!(matches!(
            verify_qualification_marker_beneath(&marker_path, root.uid, root.gid, &root.path),
            Err(DeploymentGateError::Metadata { source, .. })
                if source.kind() == io::ErrorKind::NotFound
        ));
        write_qualification_marker_beneath(&marker_path, &expected, root.uid, root.gid, &root.path)
            .expect("write marker");
        assert_eq!(
            verify_qualification_marker_beneath(&marker_path, root.uid, root.gid, &root.path,)
                .expect("verify marker"),
            expected
        );

        fs::set_permissions(&file, fs::Permissions::from_mode(0o644)).expect("make writable");
        fs::write(&file, b"version two").expect("change file");
        fs::set_permissions(&file, fs::Permissions::from_mode(0o444)).expect("restore mode");
        assert!(matches!(
            verify_qualification_marker_beneath(&marker_path, root.uid, root.gid, &root.path),
            Err(DeploymentGateError::QualifiedFileChanged { .. })
        ));
    }

    #[test]
    fn marker_symlink_and_group_writable_bound_file_are_rejected() {
        let root = TestRoot::new("metadata");
        let file = root.file("bound", b"content");
        fs::set_permissions(&file, fs::Permissions::from_mode(0o664)).expect("weaken mode");
        assert!(matches!(
            QualifiedFileV1::inspect_with_required_mode_beneath(
                file, root.uid, root.gid, None, &root.path,
            ),
            Err(DeploymentGateError::WritableQualifiedFileMode { .. })
        ));

        let target = root.file("target", b"{}");
        let link = root.path.join("link");
        std::os::unix::fs::symlink(target, &link).expect("create symlink");
        assert!(matches!(
            verify_qualification_marker_beneath(&link, root.uid, root.gid, &root.path),
            Err(DeploymentGateError::NotRegularFile { .. })
        ));
    }

    #[test]
    fn writable_or_symlinked_parent_components_are_rejected() {
        let root = TestRoot::new("parent-chain");
        let assets = root.path.join("assets");
        fs::create_dir(&assets).expect("create asset directory");
        fs::set_permissions(&assets, fs::Permissions::from_mode(0o755))
            .expect("set asset directory mode");
        let bound = assets.join("bound");
        fs::write(&bound, b"bound bytes").expect("write bound file");
        fs::set_permissions(&bound, fs::Permissions::from_mode(0o444))
            .expect("set bound file mode");

        QualifiedFileV1::inspect_with_required_mode_beneath(
            bound.clone(),
            root.uid,
            root.gid,
            None,
            &root.path,
        )
        .expect("trusted chain is admitted");
        let expected = marker(&root, &bound);
        let marker_path = root.path.join("qualification.json");
        write_qualification_marker_beneath(&marker_path, &expected, root.uid, root.gid, &root.path)
            .expect("write qualification marker");
        fs::set_permissions(&assets, fs::Permissions::from_mode(0o775))
            .expect("make parent group-writable");
        assert!(matches!(
            verify_qualification_marker_beneath(
                &marker_path,
                root.uid,
                root.gid,
                &root.path,
            ),
            Err(DeploymentGateError::WritableQualifiedDirectoryMode { path, .. })
                if path == assets
        ));

        let real = root.path.join("real");
        fs::create_dir(&real).expect("create real directory");
        fs::set_permissions(&real, fs::Permissions::from_mode(0o755))
            .expect("set real directory mode");
        let target = real.join("target");
        fs::write(&target, b"target bytes").expect("write target");
        fs::set_permissions(&target, fs::Permissions::from_mode(0o444)).expect("set target mode");
        let alias = root.path.join("alias");
        std::os::unix::fs::symlink(&real, &alias).expect("create directory symlink");
        assert!(matches!(
            QualifiedFileV1::inspect_with_required_mode_beneath(
                alias.join("target"),
                root.uid,
                root.gid,
                None,
                &root.path,
            ),
            Err(DeploymentGateError::NotDirectory { path }) if path == alias
        ));
    }

    #[test]
    fn unknown_marker_fields_and_trailing_documents_are_rejected() {
        let root = TestRoot::new("strict-json");
        let file = root.file("bound", b"version one");
        let marker_path = root.path.join("qualification.json");
        let expected = marker(&root, &file);
        write_qualification_marker_beneath(&marker_path, &expected, root.uid, root.gid, &root.path)
            .expect("write marker");

        let mut value = serde_json::to_value(&expected).expect("marker JSON");
        value
            .as_object_mut()
            .expect("marker object")
            .insert("unexpected".into(), serde_json::json!(true));
        fs::set_permissions(&marker_path, fs::Permissions::from_mode(0o600))
            .expect("permit fixture rewrite");
        fs::write(
            &marker_path,
            serde_json::to_vec(&value).expect("encode fixture"),
        )
        .expect("write unknown field");
        fs::set_permissions(&marker_path, fs::Permissions::from_mode(MARKER_MODE))
            .expect("restore marker mode");
        assert!(matches!(
            verify_qualification_marker_beneath(&marker_path, root.uid, root.gid, &root.path),
            Err(DeploymentGateError::MarkerDecode { .. })
        ));

        fs::set_permissions(&marker_path, fs::Permissions::from_mode(0o600))
            .expect("permit fixture rewrite");
        let mut trailing = serde_json::to_vec(&expected).expect("encode marker");
        trailing.extend_from_slice(b"\n{}");
        fs::write(&marker_path, trailing).expect("write trailing document");
        fs::set_permissions(&marker_path, fs::Permissions::from_mode(MARKER_MODE))
            .expect("restore marker mode");
        assert!(matches!(
            verify_qualification_marker_beneath(&marker_path, root.uid, root.gid, &root.path),
            Err(DeploymentGateError::MarkerTrailingData { .. })
        ));
    }

    #[test]
    fn source_systemd_topology_is_always_gated_and_only_drop_in_enables_boot() {
        let base = include_str!("../../../deploy/systemd/kiko-nano-agent.service");
        assert!(!base.lines().any(|line| line.trim() == "[Install]"));
        let assignment_count = |key: &str| {
            base.lines()
                .filter(|line| line.trim_start().starts_with(key))
                .count()
        };
        assert_eq!(assignment_count("User="), 1);
        assert_eq!(
            base.lines()
                .filter(|line| *line == "User=makerspace")
                .count(),
            1
        );
        assert_eq!(assignment_count("Restart="), 1);
        assert_eq!(base.lines().filter(|line| *line == "Restart=no").count(), 1);
        assert_eq!(assignment_count("ExecStartPre="), 1);
        assert_eq!(
            base.lines()
                .filter(|line| {
                    *line == "ExecStartPre=+/opt/kiko/bin/kiko-nano-deployment-gate verify --marker /etc/kiko/nano-agent-offline-install-qualified-v1.json"
                })
                .count(),
            1
        );
        assert!(
            !base
                .lines()
                .any(|line| line.trim_start().starts_with("ConditionPath"))
        );
        assert_eq!(assignment_count("ExecStart="), 1);
        assert_eq!(
            base.lines()
                .filter(|line| {
                    *line == "ExecStart=/usr/bin/env LD_LIBRARY_PATH=/opt/kiko/deployment/lib /opt/kiko/bin/kiko-slam nano-agent --deployment-root /opt/kiko/deployment --launch-config nano-agent-launch-v2.json --state-root /var/lib/kiko-nano-agent"
                })
                .count(),
            1
        );
        assert!(
            !base
                .lines()
                .any(|line| line.starts_with("Environment=LD_LIBRARY_PATH="))
        );

        let qualified = include_str!("../../../deploy/systemd/kiko-nano-agent-qualified-boot.conf");
        assert_eq!(
            qualified,
            "[Install]\n\
WantedBy=multi-user.target\n"
        );
    }

    #[test]
    fn qualified_enablement_cannot_bypass_exact_prestart_admission() {
        let base = include_str!("../../../deploy/systemd/kiko-nano-agent.service");
        let qualified = include_str!("../../../deploy/systemd/kiko-nano-agent-qualified-boot.conf");
        let effective = format!("{base}\n{qualified}");

        assert!(
            !base.lines().any(|line| line.trim() == "[Install]"),
            "the unqualified base unit must not be enableable"
        );
        assert_eq!(
            effective
                .lines()
                .filter(|line| line.trim() == "[Install]")
                .count(),
            1
        );
        assert_eq!(
            effective
                .lines()
                .filter(|line| line.trim() == "WantedBy=multi-user.target")
                .count(),
            1
        );
        assert_eq!(
            effective
                .lines()
                .filter(|line| {
                    line.trim()
                        == "ExecStartPre=+/opt/kiko/bin/kiko-nano-deployment-gate verify --marker /etc/kiko/nano-agent-offline-install-qualified-v1.json"
                })
                .count(),
            1,
            "qualified enablement must retain the exact invariant pre-start gate"
        );
        assert!(
            !qualified
                .lines()
                .any(|line| line.trim_start().starts_with("ExecStart")),
            "the enablement-only drop-in may not reset or replace the gate"
        );
        assert_eq!(
            effective
                .lines()
                .filter(|line| line.trim_start().starts_with("Restart="))
                .count(),
            1
        );
        assert!(effective.lines().any(|line| line.trim() == "Restart=no"));
        assert!(
            !effective.contains("kiko-robot-server.service"),
            "production must retain one integrated controller owner"
        );

        // This is a component simulation of ExecStartPre, not a systemd or
        // hardware execution. Merely adding the enablement drop-in cannot make
        // startup admission succeed without the exact marker and bound bytes.
        let root = TestRoot::new("qualified-enablement");
        let bound = root.file("bound", b"qualified bytes");
        let marker_path = root.path.join("qualification.json");
        let expected = marker(&root, &bound);
        assert!(matches!(
            verify_qualification_marker_beneath(
                &marker_path,
                root.uid,
                root.gid,
                &root.path,
            ),
            Err(DeploymentGateError::Metadata { source, .. })
                if source.kind() == io::ErrorKind::NotFound
        ));
        write_qualification_marker_beneath(&marker_path, &expected, root.uid, root.gid, &root.path)
            .expect("publish exact simulated qualification marker");
        verify_qualification_marker_beneath(&marker_path, root.uid, root.gid, &root.path)
            .expect("exact simulated install is admitted");

        fs::set_permissions(&bound, fs::Permissions::from_mode(0o644))
            .expect("make fixture writable");
        fs::write(&bound, b"drifted bytes").expect("drift fixture");
        fs::set_permissions(&bound, fs::Permissions::from_mode(0o444))
            .expect("restore fixture mode");
        assert!(matches!(
            verify_qualification_marker_beneath(&marker_path, root.uid, root.gid, &root.path,),
            Err(DeploymentGateError::QualifiedFileChanged { .. })
        ));
    }
}
