use std::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::secure_fs::{
    OpenedPathKind, SecureOpenError, is_canonical_absolute_path, open_absolute_nofollow,
    open_beneath_nofollow,
};
use crate::{
    ArtifactDigestDto, ArtifactId, ArtifactKind, DeviceInventoryManifestV1, MAX_ARTIFACTS,
};

pub const MAX_ARTIFACT_FILE_BYTES: u64 = 128 * 1_024 * 1_024;
pub const MAX_ARTIFACT_RELATIVE_PATH_BYTES: usize = 512;
pub const MAX_ARTIFACT_PATH_COMPONENTS: usize = 64;
pub const MAX_ARTIFACT_ROOT_PATH_BYTES: usize = 1_024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactFileBindingInput {
    pub kind: ArtifactKind,
    pub artifact_id: String,
    pub relative_path: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArtifactRelativePath(String);

impl ArtifactRelativePath {
    fn parse(value: String) -> Result<Self, ArtifactRelativePathError> {
        if value.is_empty() {
            return Err(ArtifactRelativePathError::Empty);
        }
        if value.len() > MAX_ARTIFACT_RELATIVE_PATH_BYTES {
            return Err(ArtifactRelativePathError::TooLong {
                actual_bytes: value.len(),
                maximum_bytes: MAX_ARTIFACT_RELATIVE_PATH_BYTES,
            });
        }
        if value.as_bytes().contains(&0) {
            return Err(ArtifactRelativePathError::ContainsNul);
        }
        if value.contains('\\') {
            return Err(ArtifactRelativePathError::PlatformAmbiguousSeparator);
        }
        if value.starts_with('/') || value.ends_with('/') || value.contains("//") {
            return Err(ArtifactRelativePathError::NotCanonicalRelativePath);
        }
        let mut component_count = 0_usize;
        for component in value.split('/') {
            if matches!(component, "." | "..") {
                return Err(ArtifactRelativePathError::DotComponent);
            }
            component_count += 1;
        }
        if component_count > MAX_ARTIFACT_PATH_COMPONENTS {
            return Err(ArtifactRelativePathError::TooManyComponents {
                actual: component_count,
                maximum: MAX_ARTIFACT_PATH_COMPONENTS,
            });
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn as_path(&self) -> &Path {
        Path::new(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactRelativePathError {
    Empty,
    TooLong {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    ContainsNul,
    PlatformAmbiguousSeparator,
    NotCanonicalRelativePath,
    DotComponent,
    TooManyComponents {
        actual: usize,
        maximum: usize,
    },
}

impl fmt::Display for ArtifactRelativePathError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid manifest artifact relative path: {self:?}"
        )
    }
}

impl std::error::Error for ArtifactRelativePathError {}

#[derive(Debug, PartialEq, Eq)]
struct ParsedArtifactBinding {
    kind: ArtifactKind,
    artifact_id: ArtifactId,
    relative_path: ArtifactRelativePath,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactContentIdentity {
    kind: ArtifactKind,
    artifact_id: ArtifactId,
    relative_path: ArtifactRelativePath,
    expected_sha256: [u8; 32],
    observed_sha256: [u8; 32],
    bytes_hashed: u64,
}

impl ArtifactContentIdentity {
    pub const fn kind(&self) -> ArtifactKind {
        self.kind
    }

    pub const fn artifact_id(&self) -> &ArtifactId {
        &self.artifact_id
    }

    pub const fn relative_path(&self) -> &ArtifactRelativePath {
        &self.relative_path
    }

    pub const fn expected_sha256(&self) -> &[u8; 32] {
        &self.expected_sha256
    }

    pub const fn observed_sha256(&self) -> &[u8; 32] {
        &self.observed_sha256
    }

    pub const fn bytes_hashed(&self) -> u64 {
        self.bytes_hashed
    }

    pub fn content_matches_manifest(&self) -> bool {
        self.expected_sha256 == self.observed_sha256
    }

    pub fn to_observed_digest_dto(&self) -> ArtifactDigestDto {
        ArtifactDigestDto {
            artifact_id: self.artifact_id.as_str().to_owned(),
            sha256: self.observed_sha256,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManifestArtifactHashes {
    entries: [Option<ArtifactContentIdentity>; MAX_ARTIFACTS],
    len: u8,
}

impl ManifestArtifactHashes {
    pub fn len(&self) -> usize {
        usize::from(self.len)
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = &ArtifactContentIdentity> + '_ {
        self.entries[..self.len()].iter().map(|entry| {
            entry
                .as_ref()
                .expect("manifest-bound hash prefix is initialized")
        })
    }

    pub fn all_content_matches_manifest(&self) -> bool {
        self.iter()
            .all(ArtifactContentIdentity::content_matches_manifest)
    }
}

pub fn hash_manifest_artifacts(
    manifest: &DeviceInventoryManifestV1,
    artifact_root: &Path,
    bindings: Vec<ArtifactFileBindingInput>,
) -> Result<ManifestArtifactHashes, ArtifactHashError> {
    if bindings.len() != manifest.artifacts().len() {
        return Err(ArtifactHashError::BindingCountMismatch {
            actual: bindings.len(),
            expected: manifest.artifacts().len(),
            maximum: MAX_ARTIFACTS,
        });
    }

    let mut parsed: [Option<ParsedArtifactBinding>; MAX_ARTIFACTS] = core::array::from_fn(|_| None);
    let mut parsed_len = 0_usize;
    for (index, input) in bindings.into_iter().enumerate() {
        let artifact_id = ArtifactId::parse(input.artifact_id).map_err(|source| {
            ArtifactHashError::InvalidArtifactId {
                index,
                kind: input.kind,
                source,
            }
        })?;
        let relative_path = ArtifactRelativePath::parse(input.relative_path).map_err(|source| {
            ArtifactHashError::InvalidRelativePath {
                index,
                artifact_id,
                source,
            }
        })?;
        if parsed[..parsed_len]
            .iter()
            .flatten()
            .any(|existing| existing.kind == input.kind && existing.artifact_id == artifact_id)
        {
            return Err(ArtifactHashError::DuplicateBinding {
                kind: input.kind,
                artifact_id,
            });
        }
        if parsed[..parsed_len]
            .iter()
            .flatten()
            .any(|existing| existing.relative_path == relative_path)
        {
            return Err(ArtifactHashError::DuplicateRelativePath { relative_path });
        }
        if manifest
            .artifacts()
            .find(input.kind, &artifact_id)
            .is_none()
        {
            return Err(ArtifactHashError::UnexpectedBinding {
                kind: input.kind,
                artifact_id,
            });
        }
        parsed[parsed_len] = Some(ParsedArtifactBinding {
            kind: input.kind,
            artifact_id,
            relative_path,
        });
        parsed_len += 1;
    }
    validate_root_path(artifact_root)?;
    let root =
        open_absolute_nofollow(artifact_root, OpenedPathKind::Directory).map_err(|source| {
            ArtifactHashError::OpenRoot {
                path: artifact_root.to_path_buf(),
                source,
            }
        })?;
    let mut output = ManifestArtifactHashes {
        entries: core::array::from_fn(|_| None),
        len: 0,
    };
    for expected in manifest.artifacts().iter() {
        let binding_index = parsed[..parsed_len]
            .iter()
            .position(|binding| {
                binding.as_ref().is_some_and(|binding| {
                    binding.kind == expected.kind()
                        && binding.artifact_id == *expected.artifact_id()
                })
            })
            .expect("exact manifest binding set was established before filesystem access");
        let binding = parsed[binding_index]
            .as_ref()
            .expect("matching manifest binding is initialized");
        let descriptor =
            open_beneath_nofollow(&root, binding.relative_path.as_path()).map_err(|source| {
                ArtifactHashError::OpenArtifact {
                    kind: binding.kind,
                    artifact_id: binding.artifact_id,
                    relative_path: binding.relative_path.clone(),
                    source,
                }
            })?;
        let mut file = File::from(descriptor);
        let metadata = file
            .metadata()
            .map_err(|source| ArtifactHashError::Metadata {
                kind: binding.kind,
                artifact_id: binding.artifact_id,
                relative_path: binding.relative_path.clone(),
                source,
            })?;
        if !metadata.is_file() {
            return Err(ArtifactHashError::NotRegularFile {
                kind: binding.kind,
                artifact_id: binding.artifact_id,
                relative_path: binding.relative_path.clone(),
            });
        }
        if metadata.len() > MAX_ARTIFACT_FILE_BYTES {
            return Err(ArtifactHashError::ArtifactTooLarge {
                kind: binding.kind,
                artifact_id: binding.artifact_id,
                relative_path: binding.relative_path.clone(),
                actual_bytes: metadata.len(),
                maximum_bytes: MAX_ARTIFACT_FILE_BYTES,
            });
        }
        let (observed_sha256, bytes_hashed) = hash_file(&mut file, binding, metadata.len())?;
        let binding = parsed[binding_index]
            .take()
            .expect("matching manifest binding is consumed once");
        let output_index = usize::from(output.len);
        output.entries[output_index] = Some(ArtifactContentIdentity {
            kind: expected.kind(),
            artifact_id: *expected.artifact_id(),
            relative_path: binding.relative_path,
            expected_sha256: *expected.sha256().as_bytes(),
            observed_sha256,
            bytes_hashed,
        });
        output.len += 1;
    }
    Ok(output)
}

fn hash_file(
    file: &mut File,
    binding: &ParsedArtifactBinding,
    metadata_bytes: u64,
) -> Result<([u8; 32], u64), ArtifactHashError> {
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1_024];
    let mut bytes_hashed = 0_u64;
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| ArtifactHashError::Read {
                kind: binding.kind,
                artifact_id: binding.artifact_id,
                relative_path: binding.relative_path.clone(),
                bytes_hashed,
                source,
            })?;
        if read == 0 {
            break;
        }
        bytes_hashed = bytes_hashed
            .checked_add(u64::try_from(read).expect("Linux and macOS usize values fit u64"))
            .expect("bounded artifact byte count cannot overflow u64");
        if bytes_hashed > MAX_ARTIFACT_FILE_BYTES {
            return Err(ArtifactHashError::ArtifactTooLarge {
                kind: binding.kind,
                artifact_id: binding.artifact_id,
                relative_path: binding.relative_path.clone(),
                actual_bytes: bytes_hashed,
                maximum_bytes: MAX_ARTIFACT_FILE_BYTES,
            });
        }
        hasher.update(&buffer[..read]);
    }
    if bytes_hashed != metadata_bytes {
        return Err(ArtifactHashError::FileLengthChanged {
            kind: binding.kind,
            artifact_id: binding.artifact_id,
            relative_path: binding.relative_path.clone(),
            metadata_bytes,
            bytes_hashed,
        });
    }
    Ok((hasher.finalize().into(), bytes_hashed))
}

fn validate_root_path(path: &Path) -> Result<(), ArtifactHashError> {
    if !path.is_absolute() {
        return Err(ArtifactHashError::RootNotAbsolute {
            path: path.to_path_buf(),
        });
    }
    let path_bytes = path.as_os_str().as_bytes().len();
    if path_bytes > MAX_ARTIFACT_ROOT_PATH_BYTES {
        return Err(ArtifactHashError::RootPathTooLong {
            actual_bytes: path_bytes,
            maximum_bytes: MAX_ARTIFACT_ROOT_PATH_BYTES,
        });
    }
    if !is_canonical_absolute_path(path) {
        return Err(ArtifactHashError::NonCanonicalRootPath {
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

#[derive(Debug)]
pub enum ArtifactHashError {
    BindingCountMismatch {
        actual: usize,
        expected: usize,
        maximum: usize,
    },
    InvalidArtifactId {
        index: usize,
        kind: ArtifactKind,
        source: crate::BoundedTextError,
    },
    InvalidRelativePath {
        index: usize,
        artifact_id: ArtifactId,
        source: ArtifactRelativePathError,
    },
    DuplicateBinding {
        kind: ArtifactKind,
        artifact_id: ArtifactId,
    },
    DuplicateRelativePath {
        relative_path: ArtifactRelativePath,
    },
    UnexpectedBinding {
        kind: ArtifactKind,
        artifact_id: ArtifactId,
    },
    RootNotAbsolute {
        path: PathBuf,
    },
    RootPathTooLong {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    NonCanonicalRootPath {
        path: PathBuf,
    },
    OpenRoot {
        path: PathBuf,
        source: SecureOpenError,
    },
    OpenArtifact {
        kind: ArtifactKind,
        artifact_id: ArtifactId,
        relative_path: ArtifactRelativePath,
        source: SecureOpenError,
    },
    Metadata {
        kind: ArtifactKind,
        artifact_id: ArtifactId,
        relative_path: ArtifactRelativePath,
        source: io::Error,
    },
    NotRegularFile {
        kind: ArtifactKind,
        artifact_id: ArtifactId,
        relative_path: ArtifactRelativePath,
    },
    ArtifactTooLarge {
        kind: ArtifactKind,
        artifact_id: ArtifactId,
        relative_path: ArtifactRelativePath,
        actual_bytes: u64,
        maximum_bytes: u64,
    },
    Read {
        kind: ArtifactKind,
        artifact_id: ArtifactId,
        relative_path: ArtifactRelativePath,
        bytes_hashed: u64,
        source: io::Error,
    },
    FileLengthChanged {
        kind: ArtifactKind,
        artifact_id: ArtifactId,
        relative_path: ArtifactRelativePath,
        metadata_bytes: u64,
        bytes_hashed: u64,
    },
}

impl fmt::Display for ArtifactHashError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "could not hash manifest-bound artifacts: {self:?}"
        )
    }
}

impl std::error::Error for ArtifactHashError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidArtifactId { source, .. } => Some(source),
            Self::InvalidRelativePath { source, .. } => Some(source),
            Self::OpenRoot { source, .. } | Self::OpenArtifact { source, .. } => Some(source),
            Self::Metadata { source, .. } | Self::Read { source, .. } => Some(source),
            Self::BindingCountMismatch { .. }
            | Self::DuplicateBinding { .. }
            | Self::DuplicateRelativePath { .. }
            | Self::UnexpectedBinding { .. }
            | Self::RootNotAbsolute { .. }
            | Self::RootPathTooLong { .. }
            | Self::NonCanonicalRootPath { .. }
            | Self::NotRegularFile { .. }
            | Self::ArtifactTooLarge { .. }
            | Self::FileLengthChanged { .. } => None,
        }
    }
}
