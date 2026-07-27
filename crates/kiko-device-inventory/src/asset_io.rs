use std::collections::TryReserveError;
use std::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::secure_fs::{
    OpenedPathKind, SecureOpenError, is_canonical_absolute_path, open_absolute_nofollow,
    open_beneath_nofollow,
};
use crate::{ArtifactRelativePath, MAX_ARTIFACT_FILE_BYTES, MAX_ARTIFACT_ROOT_PATH_BYTES};

/// Global upper bound for one startup/deployment asset snapshot.
pub const MAX_DEPLOYMENT_ASSET_BYTES: u64 = MAX_ARTIFACT_FILE_BYTES;

/// Parsed nonzero byte limit for one deployment asset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeploymentAssetByteLimit(u64);

impl DeploymentAssetByteLimit {
    pub fn try_new(bytes: u64) -> Result<Self, DeploymentAssetByteLimitError> {
        if bytes == 0 {
            return Err(DeploymentAssetByteLimitError::Zero);
        }
        if bytes > MAX_DEPLOYMENT_ASSET_BYTES {
            return Err(DeploymentAssetByteLimitError::AboveMaximum {
                actual_bytes: bytes,
                maximum_bytes: MAX_DEPLOYMENT_ASSET_BYTES,
            });
        }
        Ok(Self(bytes))
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeploymentAssetByteLimitError {
    Zero,
    AboveMaximum {
        actual_bytes: u64,
        maximum_bytes: u64,
    },
}

impl fmt::Display for DeploymentAssetByteLimitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid deployment asset byte limit: {self:?}")
    }
}

impl std::error::Error for DeploymentAssetByteLimitError {}

/// SHA-256 of the exact retained file bytes.
///
/// No digest value is reserved: an all-zero digest, while infeasible in
/// normal operation, is still a mathematically valid SHA-256 output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeploymentAssetContentSha256([u8; 32]);

impl DeploymentAssetContentSha256 {
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Stable Unix identity of one already opened regular file.
///
/// This is process-local loader evidence, not content identity: callers must
/// still compare the independently bound length and SHA-256. Device/inode
/// identity lets Linux runtime admission prove that the file already mapped by
/// the loader is the same file descriptor object that was content-verified.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnixFileIdentity {
    device: u64,
    inode: u64,
}

impl UnixFileIdentity {
    pub fn from_metadata(metadata: &std::fs::Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
        }
    }

    pub const fn device(self) -> u64 {
        self.device
    }

    pub const fn inode(self) -> u64 {
        self.inode
    }
}

/// One bounded retained byte sequence opened without following symlinks.
///
/// The content identity covers these retained bytes only. It does not prove
/// publisher authenticity, physical correctness, a stable single-instant file
/// state under concurrent in-place writes, or that a later pathname open
/// returns the same file. Consumers must parse/use `bytes()` (or consume
/// `into_bytes`) and compare the digest with an independently trusted binding
/// instead of reopening the path after admission.
#[derive(PartialEq, Eq)]
pub struct LoadedDeploymentAsset {
    relative_path: ArtifactRelativePath,
    bytes: Vec<u8>,
    content_sha256: DeploymentAssetContentSha256,
    file_identity: UnixFileIdentity,
}

/// Content identity produced by one bounded streaming read.
///
/// Unlike [`LoadedDeploymentAsset`], this evidence retains no file bytes. It
/// is intended for large executable and native-library admission where the
/// consumer needs exact path/length/digest evidence but will not consume the
/// content from Rust memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamedDeploymentAssetIdentity {
    relative_path: ArtifactRelativePath,
    byte_len: u64,
    content_sha256: DeploymentAssetContentSha256,
    file_identity: UnixFileIdentity,
}

impl StreamedDeploymentAssetIdentity {
    pub const fn relative_path(&self) -> &ArtifactRelativePath {
        &self.relative_path
    }

    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    pub const fn content_sha256(&self) -> DeploymentAssetContentSha256 {
        self.content_sha256
    }

    pub const fn file_identity(&self) -> UnixFileIdentity {
        self.file_identity
    }
}

impl fmt::Debug for LoadedDeploymentAsset {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LoadedDeploymentAsset")
            .field("relative_path", &self.relative_path)
            .field("byte_len", &self.bytes.len())
            .field("content_sha256", &self.content_sha256)
            .field("file_identity", &self.file_identity)
            .finish()
    }
}

impl LoadedDeploymentAsset {
    pub const fn relative_path(&self) -> &ArtifactRelativePath {
        &self.relative_path
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }

    pub const fn content_sha256(&self) -> DeploymentAssetContentSha256 {
        self.content_sha256
    }

    pub const fn file_identity(&self) -> UnixFileIdentity {
        self.file_identity
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// Open and retain one bounded regular file beneath an exact absolute root.
///
/// Every root and relative-path component is opened with `O_NOFOLLOW`; already
/// opened descriptors remain pinned if a pathname is subsequently renamed.
/// A normal file or directory atomically substituted before its component is
/// opened can still be selected, so callers must compare the returned digest
/// with an independently trusted content binding. The digest is accumulated
/// during the only content read. A changed length, a non-regular file,
/// allocation failure, or short/long read is a typed failure.
pub fn load_deployment_asset(
    root: &Path,
    relative_path: ArtifactRelativePath,
    byte_limit: DeploymentAssetByteLimit,
) -> Result<LoadedDeploymentAsset, DeploymentAssetLoadError> {
    let (mut file, declared_bytes, declared_bytes_usize, file_identity) =
        open_bounded_deployment_asset(root, &relative_path, byte_limit)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(declared_bytes_usize)
        .map_err(|source| DeploymentAssetLoadError::Allocation {
            relative_path: relative_path.clone(),
            requested_bytes: declared_bytes_usize,
            source,
        })?;
    let (bytes_read, content_sha256) = read_declared_content(
        &mut file,
        &relative_path,
        declared_bytes,
        declared_bytes_usize,
        |chunk| bytes.extend_from_slice(chunk),
    )?;
    verify_final_length(&file, &relative_path, declared_bytes, bytes_read)?;
    debug_assert_eq!(bytes.len(), declared_bytes_usize);

    Ok(LoadedDeploymentAsset {
        relative_path,
        bytes,
        content_sha256,
        file_identity,
    })
}

/// Open and hash one bounded regular file without retaining its bytes.
///
/// Root and path confinement are identical to [`load_deployment_asset`].
/// The returned identity is not publisher authentication; callers must compare
/// its digest with an independently bound expectation.
pub fn stream_deployment_asset_identity(
    root: &Path,
    relative_path: ArtifactRelativePath,
    byte_limit: DeploymentAssetByteLimit,
) -> Result<StreamedDeploymentAssetIdentity, DeploymentAssetLoadError> {
    let (mut file, declared_bytes, declared_bytes_usize, file_identity) =
        open_bounded_deployment_asset(root, &relative_path, byte_limit)?;
    let (bytes_read, content_sha256) = read_declared_content(
        &mut file,
        &relative_path,
        declared_bytes,
        declared_bytes_usize,
        |_| {},
    )?;
    verify_final_length(&file, &relative_path, declared_bytes, bytes_read)?;
    Ok(StreamedDeploymentAssetIdentity {
        relative_path,
        byte_len: declared_bytes,
        content_sha256,
        file_identity,
    })
}

fn open_bounded_deployment_asset(
    root: &Path,
    relative_path: &ArtifactRelativePath,
    byte_limit: DeploymentAssetByteLimit,
) -> Result<(File, u64, usize, UnixFileIdentity), DeploymentAssetLoadError> {
    validate_root(root)?;
    let root_descriptor =
        open_absolute_nofollow(root, OpenedPathKind::Directory).map_err(|source| {
            DeploymentAssetLoadError::OpenRoot {
                root: root.to_path_buf(),
                source,
            }
        })?;
    let descriptor =
        open_beneath_nofollow(&root_descriptor, relative_path.as_path()).map_err(|source| {
            DeploymentAssetLoadError::OpenAsset {
                relative_path: relative_path.clone(),
                source,
            }
        })?;
    let file = File::from(descriptor);
    let initial_metadata =
        file.metadata()
            .map_err(|source| DeploymentAssetLoadError::Metadata {
                relative_path: relative_path.clone(),
                source,
            })?;
    if !initial_metadata.is_file() {
        return Err(DeploymentAssetLoadError::NotRegularFile {
            relative_path: relative_path.clone(),
        });
    }
    let declared_bytes = initial_metadata.len();
    if declared_bytes > byte_limit.get() {
        return Err(DeploymentAssetLoadError::TooLarge {
            relative_path: relative_path.clone(),
            actual_bytes: declared_bytes,
            maximum_bytes: byte_limit.get(),
        });
    }
    let declared_bytes_usize = usize::try_from(declared_bytes).map_err(|_| {
        DeploymentAssetLoadError::SizeNotRepresentable {
            relative_path: relative_path.clone(),
            declared_bytes,
        }
    })?;
    let file_identity = UnixFileIdentity::from_metadata(&initial_metadata);
    Ok((file, declared_bytes, declared_bytes_usize, file_identity))
}

fn verify_final_length(
    file: &File,
    relative_path: &ArtifactRelativePath,
    declared_bytes: u64,
    bytes_read: usize,
) -> Result<(), DeploymentAssetLoadError> {
    let final_metadata = file
        .metadata()
        .map_err(|source| DeploymentAssetLoadError::Metadata {
            relative_path: relative_path.clone(),
            source,
        })?;
    let bytes_read_u64 = u64::try_from(bytes_read).map_err(|_| {
        DeploymentAssetLoadError::ObservedLengthNotRepresentable {
            relative_path: relative_path.clone(),
            observed_bytes: bytes_read,
        }
    })?;
    if final_metadata.len() != declared_bytes || bytes_read_u64 != declared_bytes {
        return Err(DeploymentAssetLoadError::FileLengthChanged {
            relative_path: relative_path.clone(),
            initial_metadata_bytes: declared_bytes,
            final_metadata_bytes: final_metadata.len(),
            bytes_read: bytes_read_u64,
        });
    }
    Ok(())
}

fn read_declared_content<R, F>(
    reader: &mut R,
    relative_path: &ArtifactRelativePath,
    declared_bytes: u64,
    declared_bytes_usize: usize,
    mut consume: F,
) -> Result<(usize, DeploymentAssetContentSha256), DeploymentAssetLoadError>
where
    R: Read,
    F: FnMut(&[u8]),
{
    let mut observed_bytes = 0_usize;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 8 * 1_024];
    loop {
        // Read at most the declared remainder plus one probe byte. Growth can
        // therefore do no caller-sized allocation or hashing work before it
        // fails, even when the configured global limit is much larger.
        let remaining = declared_bytes_usize.saturating_sub(observed_bytes);
        let read_limit = remaining.saturating_add(1).min(buffer.len());
        let read = reader.read(&mut buffer[..read_limit]).map_err(|source| {
            DeploymentAssetLoadError::Read {
                relative_path: relative_path.clone(),
                bytes_read: observed_bytes,
                source,
            }
        })?;
        if read == 0 {
            break;
        }
        let next_len = observed_bytes.checked_add(read).ok_or_else(|| {
            DeploymentAssetLoadError::ObservedLengthOverflow {
                relative_path: relative_path.clone(),
            }
        })?;
        if next_len > declared_bytes_usize {
            return Err(DeploymentAssetLoadError::FileGrewDuringRead {
                relative_path: relative_path.clone(),
                initial_metadata_bytes: declared_bytes,
                observed_bytes: next_len,
            });
        }
        hasher.update(&buffer[..read]);
        consume(&buffer[..read]);
        observed_bytes = next_len;
    }
    Ok((
        observed_bytes,
        DeploymentAssetContentSha256(hasher.finalize().into()),
    ))
}

fn validate_root(root: &Path) -> Result<(), DeploymentAssetLoadError> {
    let actual_bytes = root.as_os_str().as_bytes().len();
    if actual_bytes > MAX_ARTIFACT_ROOT_PATH_BYTES {
        return Err(DeploymentAssetLoadError::RootTooLong {
            actual_bytes,
            maximum_bytes: MAX_ARTIFACT_ROOT_PATH_BYTES,
        });
    }
    if !root.is_absolute() {
        return Err(DeploymentAssetLoadError::RootNotAbsolute {
            root: root.to_path_buf(),
        });
    }
    if root == Path::new("/") {
        return Err(DeploymentAssetLoadError::RootDirectoryNotAllowed);
    }
    if !is_canonical_absolute_path(root) {
        return Err(DeploymentAssetLoadError::RootNotCanonical {
            root: root.to_path_buf(),
        });
    }
    Ok(())
}

#[derive(Debug)]
pub enum DeploymentAssetLoadError {
    RootNotAbsolute {
        root: PathBuf,
    },
    RootTooLong {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    RootDirectoryNotAllowed,
    RootNotCanonical {
        root: PathBuf,
    },
    OpenRoot {
        root: PathBuf,
        source: SecureOpenError,
    },
    OpenAsset {
        relative_path: ArtifactRelativePath,
        source: SecureOpenError,
    },
    Metadata {
        relative_path: ArtifactRelativePath,
        source: io::Error,
    },
    NotRegularFile {
        relative_path: ArtifactRelativePath,
    },
    TooLarge {
        relative_path: ArtifactRelativePath,
        actual_bytes: u64,
        maximum_bytes: u64,
    },
    SizeNotRepresentable {
        relative_path: ArtifactRelativePath,
        declared_bytes: u64,
    },
    ObservedLengthOverflow {
        relative_path: ArtifactRelativePath,
    },
    ObservedLengthNotRepresentable {
        relative_path: ArtifactRelativePath,
        observed_bytes: usize,
    },
    Allocation {
        relative_path: ArtifactRelativePath,
        requested_bytes: usize,
        source: TryReserveError,
    },
    Read {
        relative_path: ArtifactRelativePath,
        bytes_read: usize,
        source: io::Error,
    },
    FileGrewDuringRead {
        relative_path: ArtifactRelativePath,
        initial_metadata_bytes: u64,
        observed_bytes: usize,
    },
    FileLengthChanged {
        relative_path: ArtifactRelativePath,
        initial_metadata_bytes: u64,
        final_metadata_bytes: u64,
        bytes_read: u64,
    },
}

impl fmt::Display for DeploymentAssetLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "could not load deployment asset: {self:?}")
    }
}

impl std::error::Error for DeploymentAssetLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::OpenRoot { source, .. } | Self::OpenAsset { source, .. } => Some(source),
            Self::Metadata { source, .. } | Self::Read { source, .. } => Some(source),
            Self::Allocation { source, .. } => Some(source),
            Self::RootNotAbsolute { .. }
            | Self::RootTooLong { .. }
            | Self::RootDirectoryNotAllowed
            | Self::RootNotCanonical { .. }
            | Self::NotRegularFile { .. }
            | Self::TooLarge { .. }
            | Self::SizeNotRepresentable { .. }
            | Self::ObservedLengthOverflow { .. }
            | Self::ObservedLengthNotRepresentable { .. }
            | Self::FileGrewDuringRead { .. }
            | Self::FileLengthChanged { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Cursor;
    use std::os::unix::fs::symlink;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(1);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let base = fs::canonicalize(std::env::temp_dir()).expect("canonical temp directory");
            for _ in 0..1_000 {
                let suffix = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
                let path = base.join(format!(
                    "kiko-deployment-asset-{}-{suffix}",
                    std::process::id()
                ));
                match fs::create_dir(&path) {
                    Ok(()) => return Self(path),
                    Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {}
                    Err(source) => panic!("create test directory: {source}"),
                }
            }
            panic!("could not allocate unique deployment-asset test directory")
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn path(value: &str) -> ArtifactRelativePath {
        ArtifactRelativePath::parse(value.to_owned()).expect("canonical relative path")
    }

    fn limit(bytes: u64) -> DeploymentAssetByteLimit {
        DeploymentAssetByteLimit::try_new(bytes).expect("bounded byte limit")
    }

    #[test]
    fn exact_bytes_and_digest_are_retained_from_one_bounded_read() {
        let root = TestDirectory::new();
        fs::create_dir(root.0.join("config")).expect("config directory");
        fs::write(root.0.join("config/policy.json"), b"abc").expect("asset fixture");

        let loaded = load_deployment_asset(&root.0, path("config/policy.json"), limit(3))
            .expect("bounded asset");
        assert_eq!(loaded.bytes(), b"abc");
        assert_eq!(loaded.byte_len(), 3);
        assert_eq!(
            loaded.content_sha256().as_bytes(),
            &[
                0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
                0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
                0xf2, 0x00, 0x15, 0xad,
            ]
        );
    }

    #[test]
    fn streaming_identity_matches_retained_identity_without_content_storage() {
        let root = TestDirectory::new();
        let contents = vec![0xa5; 8 * 1_024 + 1];
        fs::write(root.0.join("native.so"), &contents).expect("asset fixture");
        let content_bytes = u64::try_from(contents.len()).expect("fixture length");
        let retained = load_deployment_asset(&root.0, path("native.so"), limit(content_bytes))
            .expect("retained asset");
        let streamed =
            stream_deployment_asset_identity(&root.0, path("native.so"), limit(content_bytes))
                .expect("streamed identity");
        assert_eq!(streamed.relative_path().as_str(), "native.so");
        assert_eq!(
            streamed.byte_len(),
            u64::try_from(retained.byte_len()).expect("retained length")
        );
        assert_eq!(streamed.content_sha256(), retained.content_sha256());
        assert_eq!(streamed.file_identity(), retained.file_identity());
        assert!(matches!(
            stream_deployment_asset_identity(&root.0, path("native.so"), limit(content_bytes - 1)),
            Err(DeploymentAssetLoadError::TooLarge { .. })
        ));
    }

    #[test]
    fn empty_and_chunk_boundary_content_hash_exactly_the_returned_bytes() {
        let root = TestDirectory::new();
        for (name, contents) in [
            ("empty", Vec::new()),
            ("chunk", vec![0x5a; 8 * 1_024]),
            ("chunk-plus-one", vec![0xa5; 8 * 1_024 + 1]),
        ] {
            fs::write(root.0.join(name), &contents).expect("asset fixture");
            let loaded = load_deployment_asset(
                &root.0,
                path(name),
                limit(u64::try_from(contents.len()).unwrap().max(1)),
            )
            .expect("bounded asset");
            assert_eq!(loaded.bytes(), contents);
            let expected: [u8; 32] = Sha256::digest(&contents).into();
            assert_eq!(loaded.content_sha256().as_bytes(), &expected);
        }
    }

    #[test]
    fn read_time_growth_fails_after_only_one_probe_byte() {
        let relative_path = path("asset");
        let mut reader = Cursor::new(b"abcd");
        assert!(matches!(
            read_declared_content(&mut reader, &relative_path, 3, 3, |_| {}),
            Err(DeploymentAssetLoadError::FileGrewDuringRead {
                initial_metadata_bytes: 3,
                observed_bytes: 4,
                ..
            })
        ));
        assert_eq!(reader.position(), 4);
    }

    #[test]
    fn size_limit_is_closed_and_growth_beyond_it_is_rejected() {
        let root = TestDirectory::new();
        fs::write(root.0.join("asset"), b"1234").expect("asset fixture");
        assert!(load_deployment_asset(&root.0, path("asset"), limit(4)).is_ok());
        assert!(matches!(
            load_deployment_asset(&root.0, path("asset"), limit(3)),
            Err(DeploymentAssetLoadError::TooLarge {
                actual_bytes: 4,
                maximum_bytes: 3,
                ..
            })
        ));
    }

    #[test]
    fn symlinked_root_or_asset_component_is_never_followed() {
        let root = TestDirectory::new();
        let outside = TestDirectory::new();
        fs::write(outside.0.join("asset"), b"secret").expect("outside fixture");
        symlink(&outside.0, root.0.join("linked")).expect("relative component symlink");
        assert!(matches!(
            load_deployment_asset(&root.0, path("linked/asset"), limit(16)),
            Err(DeploymentAssetLoadError::OpenAsset { .. })
        ));
        symlink(outside.0.join("asset"), root.0.join("asset-link"))
            .expect("final component symlink");
        assert!(matches!(
            load_deployment_asset(&root.0, path("asset-link"), limit(16)),
            Err(DeploymentAssetLoadError::OpenAsset { .. })
        ));

        let root_link = root.0.with_extension("link");
        symlink(&root.0, &root_link).expect("root symlink");
        assert!(matches!(
            load_deployment_asset(&root_link, path("missing"), limit(16)),
            Err(DeploymentAssetLoadError::OpenRoot { .. })
        ));
        fs::remove_file(root_link).expect("remove root symlink");
    }

    #[test]
    fn weak_roots_paths_and_limits_are_rejected_before_io() {
        assert!(matches!(
            DeploymentAssetByteLimit::try_new(0),
            Err(DeploymentAssetByteLimitError::Zero)
        ));
        assert!(matches!(
            DeploymentAssetByteLimit::try_new(MAX_DEPLOYMENT_ASSET_BYTES + 1),
            Err(DeploymentAssetByteLimitError::AboveMaximum { .. })
        ));
        assert!(matches!(
            load_deployment_asset(Path::new("relative"), path("asset"), limit(1)),
            Err(DeploymentAssetLoadError::RootNotAbsolute { .. })
        ));
        assert!(matches!(
            load_deployment_asset(Path::new("/"), path("asset"), limit(1)),
            Err(DeploymentAssetLoadError::RootDirectoryNotAllowed)
        ));
        assert!(ArtifactRelativePath::parse("../escape".to_owned()).is_err());

        let oversized_relative_root = "x".repeat(MAX_ARTIFACT_ROOT_PATH_BYTES + 1);
        assert!(matches!(
            load_deployment_asset(Path::new(&oversized_relative_root), path("asset"), limit(1)),
            Err(DeploymentAssetLoadError::RootTooLong { .. })
        ));
    }

    #[test]
    fn debug_is_bounded_and_does_not_include_asset_content() {
        let root = TestDirectory::new();
        fs::write(root.0.join("secret"), b"do-not-log-this-content").expect("asset fixture");
        let loaded = load_deployment_asset(&root.0, path("secret"), limit(64)).unwrap();
        let debug = format!("{loaded:?}");
        assert!(debug.contains("byte_len: 23"));
        assert!(!debug.contains("do-not-log-this-content"));
    }
}
