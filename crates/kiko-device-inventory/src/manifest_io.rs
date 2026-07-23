use std::collections::TryReserveError;
use std::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::secure_fs::{
    OpenedPathKind, SecureOpenError, is_canonical_absolute_path, open_absolute_nofollow,
};
use crate::{
    DeviceInventoryManifestV1, DeviceInventoryManifestV1Dto, DeviceInventoryManifestV2,
    DeviceInventoryManifestV2Dto, DeviceInventoryManifestV3, DeviceInventoryManifestV3Dto,
    InventoryParseError,
};

pub const MAX_MANIFEST_JSON_BYTES: usize = 64 * 1_024;
pub const MAX_MANIFEST_PATH_BYTES: usize = 1_024;

#[derive(Debug)]
pub struct LoadedExpectedManifestV1 {
    manifest: DeviceInventoryManifestV1,
    json_bytes: usize,
    content_sha256: ManifestContentSha256,
    source_path: Option<PathBuf>,
}

#[derive(Debug)]
pub struct LoadedExpectedManifestV2 {
    manifest: DeviceInventoryManifestV2,
    json_bytes: usize,
    content_sha256: ManifestContentSha256,
}

#[derive(Debug)]
pub struct LoadedExpectedManifestV3 {
    manifest: DeviceInventoryManifestV3,
    json_bytes: usize,
    content_sha256: ManifestContentSha256,
}

/// SHA-256 identity of the exact admitted JSON bytes.
///
/// This is available only after the bytes have passed the bounded JSON and
/// manifest-domain parsers. It deliberately includes whitespace and key order;
/// it identifies one loaded file representation rather than a canonicalized
/// semantic manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ManifestContentSha256([u8; 32]);

impl ManifestContentSha256 {
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl LoadedExpectedManifestV1 {
    pub const fn manifest(&self) -> &DeviceInventoryManifestV1 {
        &self.manifest
    }

    pub const fn json_bytes(&self) -> usize {
        self.json_bytes
    }

    pub const fn content_sha256(&self) -> ManifestContentSha256 {
        self.content_sha256
    }

    /// Exact no-follow source path when the manifest crossed the file
    /// boundary. `None` identifies an in-memory parse and cannot satisfy a
    /// production policy which names a deployment path.
    pub fn source_path(&self) -> Option<&Path> {
        self.source_path.as_deref()
    }

    pub fn into_manifest(self) -> DeviceInventoryManifestV1 {
        self.manifest
    }
}

impl LoadedExpectedManifestV2 {
    pub const fn manifest(&self) -> &DeviceInventoryManifestV2 {
        &self.manifest
    }

    pub const fn json_bytes(&self) -> usize {
        self.json_bytes
    }

    pub const fn content_sha256(&self) -> ManifestContentSha256 {
        self.content_sha256
    }

    pub fn into_manifest(self) -> DeviceInventoryManifestV2 {
        self.manifest
    }
}

impl LoadedExpectedManifestV3 {
    pub const fn manifest(&self) -> &DeviceInventoryManifestV3 {
        &self.manifest
    }

    pub const fn json_bytes(&self) -> usize {
        self.json_bytes
    }

    pub const fn content_sha256(&self) -> ManifestContentSha256 {
        self.content_sha256
    }

    pub fn into_manifest(self) -> DeviceInventoryManifestV3 {
        self.manifest
    }
}

pub fn load_expected_manifest_v1_from_slice(
    json: &[u8],
) -> Result<LoadedExpectedManifestV1, ManifestLoadError> {
    if json.len() > MAX_MANIFEST_JSON_BYTES {
        return Err(ManifestLoadError::JsonTooLarge {
            actual_bytes: host_usize_to_u64(json.len()),
            maximum_bytes: host_usize_to_u64(MAX_MANIFEST_JSON_BYTES),
        });
    }
    let mut deserializer = serde_json::Deserializer::from_slice(json);
    let dto = DeviceInventoryManifestV1Dto::deserialize(&mut deserializer)
        .map_err(|source| ManifestLoadError::Json(ManifestJsonError::Decode { source }))?;
    deserializer
        .end()
        .map_err(|source| ManifestLoadError::Json(ManifestJsonError::TrailingData { source }))?;
    let manifest = DeviceInventoryManifestV1::parse(dto)
        .map_err(|source| ManifestLoadError::Domain { source })?;
    Ok(LoadedExpectedManifestV1 {
        manifest,
        json_bytes: json.len(),
        content_sha256: ManifestContentSha256(Sha256::digest(json).into()),
        source_path: None,
    })
}

/// Parse one bounded schema-V2 candidate manifest. File ownership and
/// no-follow policy remain the caller's responsibility; Nano startup already
/// satisfies that boundary by passing bytes from a `LoadedDeploymentAsset`.
pub fn load_expected_manifest_v2_from_slice(
    json: &[u8],
) -> Result<LoadedExpectedManifestV2, ManifestLoadError> {
    if json.len() > MAX_MANIFEST_JSON_BYTES {
        return Err(ManifestLoadError::JsonTooLarge {
            actual_bytes: host_usize_to_u64(json.len()),
            maximum_bytes: host_usize_to_u64(MAX_MANIFEST_JSON_BYTES),
        });
    }
    let mut deserializer = serde_json::Deserializer::from_slice(json);
    let dto = DeviceInventoryManifestV2Dto::deserialize(&mut deserializer)
        .map_err(|source| ManifestLoadError::Json(ManifestJsonError::Decode { source }))?;
    deserializer
        .end()
        .map_err(|source| ManifestLoadError::Json(ManifestJsonError::TrailingData { source }))?;
    let manifest = DeviceInventoryManifestV2::parse(dto)
        .map_err(|source| ManifestLoadError::Domain { source })?;
    Ok(LoadedExpectedManifestV2 {
        manifest,
        json_bytes: json.len(),
        content_sha256: ManifestContentSha256(Sha256::digest(json).into()),
    })
}

/// Parse one bounded schema-V3 attended wheel-on commissioning manifest.
///
/// File ownership and no-follow policy remain the caller's responsibility;
/// Nano startup passes bytes from one already loaded deployment asset.
pub fn load_expected_manifest_v3_from_slice(
    json: &[u8],
) -> Result<LoadedExpectedManifestV3, ManifestLoadError> {
    if json.len() > MAX_MANIFEST_JSON_BYTES {
        return Err(ManifestLoadError::JsonTooLarge {
            actual_bytes: host_usize_to_u64(json.len()),
            maximum_bytes: host_usize_to_u64(MAX_MANIFEST_JSON_BYTES),
        });
    }
    let mut deserializer = serde_json::Deserializer::from_slice(json);
    let dto = DeviceInventoryManifestV3Dto::deserialize(&mut deserializer)
        .map_err(|source| ManifestLoadError::Json(ManifestJsonError::Decode { source }))?;
    deserializer
        .end()
        .map_err(|source| ManifestLoadError::Json(ManifestJsonError::TrailingData { source }))?;
    let manifest = DeviceInventoryManifestV3::parse(dto)
        .map_err(|source| ManifestLoadError::Domain { source })?;
    Ok(LoadedExpectedManifestV3 {
        manifest,
        json_bytes: json.len(),
        content_sha256: ManifestContentSha256(Sha256::digest(json).into()),
    })
}

pub fn load_expected_manifest_v1_file(
    path: &Path,
) -> Result<LoadedExpectedManifestV1, ManifestLoadError> {
    validate_manifest_path(path)?;
    let descriptor = open_absolute_nofollow(path, OpenedPathKind::File).map_err(|source| {
        ManifestLoadError::Open {
            path: path.to_path_buf(),
            source,
        }
    })?;
    let mut file = File::from(descriptor);
    let metadata = file
        .metadata()
        .map_err(|source| ManifestLoadError::Metadata {
            path: path.to_path_buf(),
            source,
        })?;
    if !metadata.is_file() {
        return Err(ManifestLoadError::NotRegularFile {
            path: path.to_path_buf(),
            actual: FileKind::from_metadata(&metadata),
        });
    }
    let declared_bytes = metadata.len();
    if declared_bytes > host_usize_to_u64(MAX_MANIFEST_JSON_BYTES) {
        return Err(ManifestLoadError::JsonTooLarge {
            actual_bytes: declared_bytes,
            maximum_bytes: host_usize_to_u64(MAX_MANIFEST_JSON_BYTES),
        });
    }

    let mut json = Vec::new();
    let initial_capacity =
        usize::try_from(declared_bytes).expect("manifest byte limit always fits the host usize");
    json.try_reserve_exact(initial_capacity)
        .map_err(|source| ManifestLoadError::Allocation {
            requested_bytes: initial_capacity,
            source,
        })?;
    let mut buffer = [0_u8; 8 * 1_024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| ManifestLoadError::Read {
                path: path.to_path_buf(),
                bytes_read: json.len(),
                source,
            })?;
        if read == 0 {
            break;
        }
        let next_len = json
            .len()
            .checked_add(read)
            .expect("bounded manifest reads cannot overflow usize");
        if next_len > MAX_MANIFEST_JSON_BYTES {
            return Err(ManifestLoadError::JsonTooLarge {
                actual_bytes: host_usize_to_u64(next_len),
                maximum_bytes: host_usize_to_u64(MAX_MANIFEST_JSON_BYTES),
            });
        }
        if next_len > json.capacity() {
            json.try_reserve_exact(next_len - json.len())
                .map_err(|source| ManifestLoadError::Allocation {
                    requested_bytes: next_len,
                    source,
                })?;
        }
        json.extend_from_slice(&buffer[..read]);
    }
    if host_usize_to_u64(json.len()) != declared_bytes {
        return Err(ManifestLoadError::FileLengthChanged {
            path: path.to_path_buf(),
            metadata_bytes: declared_bytes,
            bytes_read: host_usize_to_u64(json.len()),
        });
    }
    let mut loaded = load_expected_manifest_v1_from_slice(&json)?;
    loaded.source_path = Some(path.to_path_buf());
    Ok(loaded)
}

fn host_usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).expect("Linux and macOS usize values fit u64")
}

fn validate_manifest_path(path: &Path) -> Result<(), ManifestLoadError> {
    if !path.is_absolute() {
        return Err(ManifestLoadError::PathNotAbsolute {
            path: path.to_path_buf(),
        });
    }
    let path_bytes = path.as_os_str().as_bytes().len();
    if path_bytes > MAX_MANIFEST_PATH_BYTES {
        return Err(ManifestLoadError::PathTooLong {
            actual_bytes: path_bytes,
            maximum_bytes: MAX_MANIFEST_PATH_BYTES,
        });
    }
    if !is_canonical_absolute_path(path) {
        return Err(ManifestLoadError::NonCanonicalPath {
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileKind {
    Directory,
    Symlink,
    Other,
}

impl FileKind {
    fn from_metadata(metadata: &std::fs::Metadata) -> Self {
        let file_type = metadata.file_type();
        if file_type.is_dir() {
            Self::Directory
        } else if file_type.is_symlink() {
            Self::Symlink
        } else {
            Self::Other
        }
    }
}

#[derive(Debug)]
pub enum ManifestJsonError {
    Decode { source: serde_json::Error },
    TrailingData { source: serde_json::Error },
}

impl fmt::Display for ManifestJsonError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "expected-manifest JSON was rejected: {self:?}")
    }
}

impl std::error::Error for ManifestJsonError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Decode { source } | Self::TrailingData { source } => Some(source),
        }
    }
}

#[derive(Debug)]
pub enum ManifestLoadError {
    PathNotAbsolute {
        path: PathBuf,
    },
    PathTooLong {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    NonCanonicalPath {
        path: PathBuf,
    },
    Open {
        path: PathBuf,
        source: SecureOpenError,
    },
    Metadata {
        path: PathBuf,
        source: io::Error,
    },
    NotRegularFile {
        path: PathBuf,
        actual: FileKind,
    },
    JsonTooLarge {
        actual_bytes: u64,
        maximum_bytes: u64,
    },
    Allocation {
        requested_bytes: usize,
        source: TryReserveError,
    },
    Read {
        path: PathBuf,
        bytes_read: usize,
        source: io::Error,
    },
    FileLengthChanged {
        path: PathBuf,
        metadata_bytes: u64,
        bytes_read: u64,
    },
    Json(ManifestJsonError),
    Domain {
        source: InventoryParseError,
    },
}

impl fmt::Display for ManifestLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "could not load expected Kiko manifest V1: {self:?}"
        )
    }
}

impl std::error::Error for ManifestLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Open { source, .. } => Some(source),
            Self::Metadata { source, .. } | Self::Read { source, .. } => Some(source),
            Self::Allocation { source, .. } => Some(source),
            Self::Json(source) => Some(source),
            Self::Domain { source } => Some(source),
            Self::PathNotAbsolute { .. }
            | Self::PathTooLong { .. }
            | Self::NonCanonicalPath { .. }
            | Self::NotRegularFile { .. }
            | Self::JsonTooLarge { .. }
            | Self::FileLengthChanged { .. } => None,
        }
    }
}
