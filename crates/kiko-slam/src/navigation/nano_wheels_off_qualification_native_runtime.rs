//! Strict native-runtime boundary for the attended wheels-off qualifier.
//!
//! Production has a separate offline-install qualification path. This module
//! deliberately admits only the seven closed required native-runtime roles emitted by the
//! wheels-off bundle renderer and never widens the production contract.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::OpenOptions;
use std::io;
use std::io::Read;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

use kiko_device_inventory::{
    ArtifactRelativePath, ArtifactRelativePathError, DeploymentAssetByteLimit,
    DeploymentAssetByteLimitError, StreamedDeploymentAssetIdentity, UnixFileIdentity,
};
use serde::Deserialize;

use super::{
    NanoLaunchAssetBinding, NanoLaunchBoundAssetLoadError, NanoLaunchSha256Error, parse_sha256,
};

pub const MAX_NANO_WHEELS_OFF_NATIVE_RUNTIME_JSON_BYTES: u64 = 64 * 1_024;
const NATIVE_RUNTIME_SCHEMA_VERSION: u32 = 1;
const NATIVE_LIBRARY_SEARCH_RELATIVE_PATH: &str = "lib";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum NanoWheelsOffNativeLibraryRole {
    DepthAiCore,
    DynamicCalibration,
    LibUsb1_0,
    OnnxRuntime,
    OpenCvCore,
    OpenCvImgproc,
    OpenCvObjdetect,
}

impl NanoWheelsOffNativeLibraryRole {
    pub const ALL: [Self; 7] = [
        Self::DepthAiCore,
        Self::DynamicCalibration,
        Self::LibUsb1_0,
        Self::OnnxRuntime,
        Self::OpenCvCore,
        Self::OpenCvImgproc,
        Self::OpenCvObjdetect,
    ];

    fn parse(value: String) -> Result<Self, NanoWheelsOffNativeRuntimeParseError> {
        match value.as_str() {
            "depthai_core" => Ok(Self::DepthAiCore),
            "dynamic_calibration" => Ok(Self::DynamicCalibration),
            "libusb_1_0" => Ok(Self::LibUsb1_0),
            "onnxruntime" => Ok(Self::OnnxRuntime),
            "opencv_core" => Ok(Self::OpenCvCore),
            "opencv_imgproc" => Ok(Self::OpenCvImgproc),
            "opencv_objdetect" => Ok(Self::OpenCvObjdetect),
            _ => Err(NanoWheelsOffNativeRuntimeParseError::UnknownLibraryRole { actual: value }),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DepthAiCore => "depthai_core",
            Self::DynamicCalibration => "dynamic_calibration",
            Self::LibUsb1_0 => "libusb_1_0",
            Self::OnnxRuntime => "onnxruntime",
            Self::OpenCvCore => "opencv_core",
            Self::OpenCvImgproc => "opencv_imgproc",
            Self::OpenCvObjdetect => "opencv_objdetect",
        }
    }

    const fn exact_nano_soname(self) -> &'static str {
        match self {
            Self::DepthAiCore => "libdepthai-core.so",
            Self::DynamicCalibration => "libdynamic_calibration.so",
            Self::LibUsb1_0 => "libusb-1.0.so.0",
            Self::OnnxRuntime => "libonnxruntime.so.1",
            Self::OpenCvCore => "libopencv_core.so.4.5d",
            Self::OpenCvImgproc => "libopencv_imgproc.so.4.5d",
            Self::OpenCvObjdetect => "libopencv_objdetect.so.4.5d",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoWheelsOffNativeLibraryBinding {
    role: NanoWheelsOffNativeLibraryRole,
    soname: Box<str>,
    asset: NanoLaunchAssetBinding,
}

impl NanoWheelsOffNativeLibraryBinding {
    pub const fn role(&self) -> NanoWheelsOffNativeLibraryRole {
        self.role
    }

    pub fn soname(&self) -> &str {
        &self.soname
    }

    pub const fn asset(&self) -> &NanoLaunchAssetBinding {
        &self.asset
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoWheelsOffNativeRuntimeV1 {
    library_search_relative_path: ArtifactRelativePath,
    libraries: [NanoWheelsOffNativeLibraryBinding; 7],
}

impl NanoWheelsOffNativeRuntimeV1 {
    pub fn parse_json(json: &[u8]) -> Result<Self, NanoWheelsOffNativeRuntimeParseError> {
        if json.len()
            > usize::try_from(MAX_NANO_WHEELS_OFF_NATIVE_RUNTIME_JSON_BYTES)
                .expect("the 64 KiB manifest bound fits every supported host")
        {
            return Err(NanoWheelsOffNativeRuntimeParseError::InputTooLarge {
                actual_bytes: json.len(),
                maximum_bytes: MAX_NANO_WHEELS_OFF_NATIVE_RUNTIME_JSON_BYTES,
            });
        }
        let mut deserializer = serde_json::Deserializer::from_slice(json);
        let dto = NativeRuntimeManifestV1Dto::deserialize(&mut deserializer)
            .map_err(NanoWheelsOffNativeRuntimeParseError::JsonDecode)?;
        deserializer
            .end()
            .map_err(NanoWheelsOffNativeRuntimeParseError::JsonTrailingData)?;
        if dto.schema_version != NATIVE_RUNTIME_SCHEMA_VERSION {
            return Err(NanoWheelsOffNativeRuntimeParseError::UnsupportedSchema {
                actual: dto.schema_version,
                supported: NATIVE_RUNTIME_SCHEMA_VERSION,
            });
        }
        let library_search_relative_path =
            ArtifactRelativePath::parse(dto.library_search_relative_path)
                .map_err(NanoWheelsOffNativeRuntimeParseError::InvalidLibrarySearchPath)?;
        if library_search_relative_path.as_str() != NATIVE_LIBRARY_SEARCH_RELATIVE_PATH {
            return Err(
                NanoWheelsOffNativeRuntimeParseError::WrongLibrarySearchPath {
                    actual: library_search_relative_path.as_str().to_owned(),
                },
            );
        }
        if dto.libraries.len() != NanoWheelsOffNativeLibraryRole::ALL.len() {
            return Err(NanoWheelsOffNativeRuntimeParseError::WrongLibraryCount {
                actual: dto.libraries.len(),
                required: NanoWheelsOffNativeLibraryRole::ALL.len(),
            });
        }

        let mut parsed = BTreeMap::new();
        let mut paths = BTreeSet::new();
        for library in dto.libraries {
            let role = NanoWheelsOffNativeLibraryRole::parse(library.role)?;
            validate_soname(role, &library.soname)?;
            let relative_path =
                ArtifactRelativePath::parse(library.relative_path).map_err(|source| {
                    NanoWheelsOffNativeRuntimeParseError::InvalidLibraryPath { role, source }
                })?;
            if relative_path.as_path().parent()
                != Some(Path::new(NATIVE_LIBRARY_SEARCH_RELATIVE_PATH))
                || relative_path.as_path().file_name()
                    != Some(std::ffi::OsStr::new(&library.soname))
            {
                return Err(
                    NanoWheelsOffNativeRuntimeParseError::LibraryOutsideSearchDirectory {
                        role,
                        soname: library.soname,
                        relative_path,
                    },
                );
            }
            if !paths.insert(relative_path.clone()) {
                return Err(NanoWheelsOffNativeRuntimeParseError::DuplicateLibraryPath {
                    relative_path,
                });
            }
            let byte_limit =
                DeploymentAssetByteLimit::try_new(library.maximum_bytes).map_err(|source| {
                    NanoWheelsOffNativeRuntimeParseError::InvalidLibraryByteLimit { role, source }
                })?;
            let expected_sha256 = parse_sha256(&library.sha256_hex).map_err(|source| {
                NanoWheelsOffNativeRuntimeParseError::InvalidLibrarySha256 { role, source }
            })?;
            let binding = NanoWheelsOffNativeLibraryBinding {
                role,
                soname: library.soname.into_boxed_str(),
                asset: NanoLaunchAssetBinding::from_parsed_parts(
                    relative_path,
                    byte_limit,
                    expected_sha256,
                ),
            };
            if parsed.insert(role, binding).is_some() {
                return Err(NanoWheelsOffNativeRuntimeParseError::DuplicateLibraryRole { role });
            }
        }

        let ordered = NanoWheelsOffNativeLibraryRole::ALL
            .map(|role| parsed.remove(&role).expect("all seven roles were parsed"));
        debug_assert!(parsed.is_empty());
        Ok(Self {
            library_search_relative_path,
            libraries: ordered,
        })
    }

    pub const fn library_search_relative_path(&self) -> &ArtifactRelativePath {
        &self.library_search_relative_path
    }

    pub const fn libraries(&self) -> &[NanoWheelsOffNativeLibraryBinding; 7] {
        &self.libraries
    }

    pub fn library(
        &self,
        role: NanoWheelsOffNativeLibraryRole,
    ) -> &NanoWheelsOffNativeLibraryBinding {
        &self.libraries[role_index(role)]
    }

    pub fn bind_onnx_runtime_launch(
        &self,
        launch: &NanoLaunchAssetBinding,
    ) -> Result<(), NanoWheelsOffNativeRuntimeBindingError> {
        let native = self
            .library(NanoWheelsOffNativeLibraryRole::OnnxRuntime)
            .asset();
        if native.relative_path() != launch.relative_path()
            || native.byte_limit() != launch.byte_limit()
            || native.expected_sha256() != launch.expected_sha256()
        {
            return Err(NanoWheelsOffNativeRuntimeBindingError::OnnxRuntimeLaunchMismatch);
        }
        Ok(())
    }

    pub fn reject_non_onnx_launch_aliases<'path>(
        &self,
        launch_paths: impl IntoIterator<Item = &'path ArtifactRelativePath>,
    ) -> Result<(), NanoWheelsOffNativeRuntimeBindingError> {
        let launch_paths = launch_paths.into_iter().collect::<BTreeSet<_>>();
        for library in &self.libraries {
            if library.role != NanoWheelsOffNativeLibraryRole::OnnxRuntime
                && launch_paths.contains(library.asset.relative_path())
            {
                return Err(
                    NanoWheelsOffNativeRuntimeBindingError::LibraryAliasesLaunchAsset {
                        role: library.role,
                        relative_path: library.asset.relative_path().clone(),
                    },
                );
            }
        }
        Ok(())
    }

    pub fn verify_dependencies_reusing_onnx(
        &self,
        deployment_root: &Path,
        verified_onnx: &StreamedDeploymentAssetIdentity,
    ) -> Result<
        VerifiedNanoWheelsOffNativeRuntimeDependencies,
        NanoWheelsOffNativeRuntimeVerificationError,
    > {
        let onnx = self
            .library(NanoWheelsOffNativeLibraryRole::OnnxRuntime)
            .asset();
        if onnx.relative_path() != verified_onnx.relative_path()
            || onnx.expected_sha256() != verified_onnx.content_sha256().as_bytes()
            || verified_onnx.byte_len() > onnx.byte_limit().get()
        {
            return Err(NanoWheelsOffNativeRuntimeVerificationError::VerifiedOnnxRuntimeMismatch);
        }

        let mut dependencies = Vec::with_capacity(NanoWheelsOffNativeLibraryRole::ALL.len() - 1);
        for library in &self.libraries {
            if library.role == NanoWheelsOffNativeLibraryRole::OnnxRuntime {
                continue;
            }
            let identity = library
                .asset
                .verify_exact_streaming(deployment_root)
                .map_err(
                    |source| NanoWheelsOffNativeRuntimeVerificationError::Library {
                        role: library.role,
                        source,
                    },
                )?;
            dependencies.push((library.role, identity));
        }
        debug_assert_eq!(
            dependencies.len(),
            NanoWheelsOffNativeLibraryRole::ALL.len() - 1
        );
        Ok(VerifiedNanoWheelsOffNativeRuntimeDependencies { dependencies })
    }
}

const fn role_index(role: NanoWheelsOffNativeLibraryRole) -> usize {
    match role {
        NanoWheelsOffNativeLibraryRole::DepthAiCore => 0,
        NanoWheelsOffNativeLibraryRole::DynamicCalibration => 1,
        NanoWheelsOffNativeLibraryRole::LibUsb1_0 => 2,
        NanoWheelsOffNativeLibraryRole::OnnxRuntime => 3,
        NanoWheelsOffNativeLibraryRole::OpenCvCore => 4,
        NanoWheelsOffNativeLibraryRole::OpenCvImgproc => 5,
        NanoWheelsOffNativeLibraryRole::OpenCvObjdetect => 6,
    }
}

fn validate_soname(
    role: NanoWheelsOffNativeLibraryRole,
    soname: &str,
) -> Result<(), NanoWheelsOffNativeRuntimeParseError> {
    if soname.is_empty()
        || soname.len() > 128
        || !soname
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_' | b'+'))
        || !soname.starts_with("lib")
        || !soname.contains(".so")
    {
        return Err(NanoWheelsOffNativeRuntimeParseError::InvalidSoname {
            role,
            actual: soname.to_owned(),
        });
    }
    let expected = role.exact_nano_soname();
    if soname != expected {
        return Err(NanoWheelsOffNativeRuntimeParseError::WrongSonameForRole {
            role,
            expected,
            actual: soname.to_owned(),
        });
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeRuntimeManifestV1Dto {
    schema_version: u32,
    library_search_relative_path: String,
    libraries: Vec<NativeLibraryDto>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeLibraryDto {
    role: String,
    soname: String,
    relative_path: String,
    maximum_bytes: u64,
    sha256_hex: String,
}

#[derive(Debug)]
pub struct VerifiedNanoWheelsOffNativeRuntimeDependencies {
    dependencies: Vec<(
        NanoWheelsOffNativeLibraryRole,
        StreamedDeploymentAssetIdentity,
    )>,
}

impl VerifiedNanoWheelsOffNativeRuntimeDependencies {
    pub fn dependency(
        &self,
        role: NanoWheelsOffNativeLibraryRole,
    ) -> Option<&StreamedDeploymentAssetIdentity> {
        self.dependencies
            .iter()
            .find_map(|(candidate, loaded)| (*candidate == role).then_some(loaded))
    }

    pub fn dependencies(
        &self,
    ) -> impl ExactSizeIterator<
        Item = (
            NanoWheelsOffNativeLibraryRole,
            &StreamedDeploymentAssetIdentity,
        ),
    > {
        self.dependencies
            .iter()
            .map(|(role, loaded)| (*role, loaded))
    }
}

/// One qualification image whose verified file identity must be present in
/// the Linux loader map before any hardware is opened.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum NanoWheelsOffMappedImageRole {
    QualificationExecutable,
    NativeLibrary(NanoWheelsOffNativeLibraryRole),
}

/// Admission evidence that the exact verified executable and seven native
/// library inodes each had a file-backed executable mapping in this process
/// at the qualification boundary.
///
/// This evidence does not claim an exact `DT_NEEDED` graph. Release tooling
/// must inspect that graph independently.
#[derive(Debug)]
pub struct VerifiedNanoWheelsOffMappedImages {
    images: BTreeMap<NanoWheelsOffMappedImageRole, UnixFileIdentity>,
}

impl VerifiedNanoWheelsOffMappedImages {
    pub fn file_identity(&self, role: NanoWheelsOffMappedImageRole) -> Option<UnixFileIdentity> {
        self.images.get(&role).copied()
    }

    pub fn images(
        &self,
    ) -> impl ExactSizeIterator<Item = (NanoWheelsOffMappedImageRole, UnixFileIdentity)> + '_ {
        self.images
            .iter()
            .map(|(role, identity)| (*role, *identity))
    }
}

const MAX_LINUX_PROCESS_MAPS_BYTES: u64 = 4 * 1_024 * 1_024;
const LINUX_PROCESS_MAPS_PATH: &str = "/proc/self/maps";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct LinuxMappedFileIdentity {
    device_major: u64,
    device_minor: u64,
    inode: u64,
}

impl LinuxMappedFileIdentity {
    fn from_unix(identity: UnixFileIdentity) -> Self {
        // Linux's userspace `major(3)` and `minor(3)` encoding. Keeping the
        // conversion here avoids comparing the textual `/proc` device pair to
        // an ambiguously encoded `dev_t`.
        let device = identity.device();
        Self {
            device_major: ((device >> 8) & 0x0fff) | ((device >> 32) & !0x0fff),
            device_minor: (device & 0x00ff) | ((device >> 12) & !0x00ff),
            inode: identity.inode(),
        }
    }
}

/// Require every independently content-verified qualification image to be
/// mapped from that same inode in the current Linux process.
pub fn verify_linux_mapped_qualification_images(
    qualification_executable: &StreamedDeploymentAssetIdentity,
    dependencies: &VerifiedNanoWheelsOffNativeRuntimeDependencies,
    onnx_runtime: &StreamedDeploymentAssetIdentity,
) -> Result<VerifiedNanoWheelsOffMappedImages, NanoWheelsOffMappedImageError> {
    if !cfg!(target_os = "linux") {
        return Err(NanoWheelsOffMappedImageError::UnsupportedPlatform {
            target_os: std::env::consts::OS,
        });
    }

    let mut expected = Vec::with_capacity(NanoWheelsOffNativeLibraryRole::ALL.len() + 1);
    expected.push((
        NanoWheelsOffMappedImageRole::QualificationExecutable,
        qualification_executable.file_identity(),
    ));
    for role in NanoWheelsOffNativeLibraryRole::ALL {
        let identity = if role == NanoWheelsOffNativeLibraryRole::OnnxRuntime {
            onnx_runtime.file_identity()
        } else {
            dependencies
                .dependency(role)
                .ok_or(NanoWheelsOffMappedImageError::MissingVerifiedDependency { role })?
                .file_identity()
        };
        expected.push((NanoWheelsOffMappedImageRole::NativeLibrary(role), identity));
    }

    let mapped = read_linux_process_maps()?;
    admit_mapped_images(expected, &mapped)
}

fn admit_mapped_images(
    expected: Vec<(NanoWheelsOffMappedImageRole, UnixFileIdentity)>,
    mapped: &BTreeSet<LinuxMappedFileIdentity>,
) -> Result<VerifiedNanoWheelsOffMappedImages, NanoWheelsOffMappedImageError> {
    let mut identities = BTreeMap::new();
    let mut admitted = BTreeMap::new();
    for (role, identity) in expected {
        let linux_identity = LinuxMappedFileIdentity::from_unix(identity);
        if let Some(first_role) = identities.insert(linux_identity, role) {
            return Err(NanoWheelsOffMappedImageError::ExpectedIdentityAliased {
                first_role,
                second_role: role,
                device: identity.device(),
                inode: identity.inode(),
            });
        }
        if !mapped.contains(&linux_identity) {
            return Err(
                NanoWheelsOffMappedImageError::ExpectedImageNotExecutablyMapped {
                    role,
                    device: identity.device(),
                    inode: identity.inode(),
                },
            );
        }
        admitted.insert(role, identity);
    }
    Ok(VerifiedNanoWheelsOffMappedImages { images: admitted })
}

fn read_linux_process_maps()
-> Result<BTreeSet<LinuxMappedFileIdentity>, NanoWheelsOffMappedImageError> {
    let path = Path::new(LINUX_PROCESS_MAPS_PATH);
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC)
        .open(path)
        .map_err(|source| NanoWheelsOffMappedImageError::OpenProcessMaps { source })?;
    let mut bytes = Vec::with_capacity(64 * 1_024);
    Read::by_ref(&mut file)
        .take(MAX_LINUX_PROCESS_MAPS_BYTES.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|source| NanoWheelsOffMappedImageError::ReadProcessMaps { source })?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_LINUX_PROCESS_MAPS_BYTES {
        return Err(NanoWheelsOffMappedImageError::ProcessMapsTooLarge {
            maximum_bytes: MAX_LINUX_PROCESS_MAPS_BYTES,
        });
    }
    let text = std::str::from_utf8(&bytes)
        .map_err(|source| NanoWheelsOffMappedImageError::ProcessMapsNotUtf8 { source })?;
    parse_linux_process_maps(text)
}

fn parse_linux_process_maps(
    text: &str,
) -> Result<BTreeSet<LinuxMappedFileIdentity>, NanoWheelsOffMappedImageError> {
    let mut identities = BTreeSet::new();
    for (index, line) in text.lines().enumerate() {
        let line_number = index.saturating_add(1);
        let mut fields = line.split_whitespace();
        let range = fields
            .next()
            .ok_or(NanoWheelsOffMappedImageError::MalformedProcessMapsLine { line_number })?;
        let permissions = fields
            .next()
            .ok_or(NanoWheelsOffMappedImageError::MalformedProcessMapsLine { line_number })?;
        let offset = fields
            .next()
            .ok_or(NanoWheelsOffMappedImageError::MalformedProcessMapsLine { line_number })?;
        let device = fields
            .next()
            .ok_or(NanoWheelsOffMappedImageError::MalformedProcessMapsLine { line_number })?;
        let inode = fields
            .next()
            .ok_or(NanoWheelsOffMappedImageError::MalformedProcessMapsLine { line_number })?;

        let Some((range_start, range_end)) = range.split_once('-') else {
            return Err(NanoWheelsOffMappedImageError::MalformedProcessMapsLine { line_number });
        };
        let permissions = permissions.as_bytes();
        if range_end.contains('-')
            || u64::from_str_radix(range_start, 16).is_err()
            || u64::from_str_radix(range_end, 16).is_err()
            || permissions.len() != 4
            || !matches!(permissions[0], b'r' | b'-')
            || !matches!(permissions[1], b'w' | b'-')
            || !matches!(permissions[2], b'x' | b'-')
            || !matches!(permissions[3], b'p' | b's')
            || u64::from_str_radix(offset, 16).is_err()
        {
            return Err(NanoWheelsOffMappedImageError::MalformedProcessMapsLine { line_number });
        }
        let Some((device_major, device_minor)) = device.split_once(':') else {
            return Err(NanoWheelsOffMappedImageError::MalformedProcessMapsLine { line_number });
        };
        if device_minor.contains(':') {
            return Err(NanoWheelsOffMappedImageError::MalformedProcessMapsLine { line_number });
        }
        let device_major = u64::from_str_radix(device_major, 16)
            .map_err(|_| NanoWheelsOffMappedImageError::MalformedProcessMapsLine { line_number })?;
        let device_minor = u64::from_str_radix(device_minor, 16)
            .map_err(|_| NanoWheelsOffMappedImageError::MalformedProcessMapsLine { line_number })?;
        let inode = inode
            .parse::<u64>()
            .map_err(|_| NanoWheelsOffMappedImageError::MalformedProcessMapsLine { line_number })?;
        if inode != 0 && permissions[2] == b'x' {
            identities.insert(LinuxMappedFileIdentity {
                device_major,
                device_minor,
                inode,
            });
        }
    }
    Ok(identities)
}

#[derive(Debug)]
pub enum NanoWheelsOffMappedImageError {
    UnsupportedPlatform {
        target_os: &'static str,
    },
    MissingVerifiedDependency {
        role: NanoWheelsOffNativeLibraryRole,
    },
    ExpectedIdentityAliased {
        first_role: NanoWheelsOffMappedImageRole,
        second_role: NanoWheelsOffMappedImageRole,
        device: u64,
        inode: u64,
    },
    OpenProcessMaps {
        source: io::Error,
    },
    ReadProcessMaps {
        source: io::Error,
    },
    ProcessMapsTooLarge {
        maximum_bytes: u64,
    },
    ProcessMapsNotUtf8 {
        source: std::str::Utf8Error,
    },
    MalformedProcessMapsLine {
        line_number: usize,
    },
    ExpectedImageNotExecutablyMapped {
        role: NanoWheelsOffMappedImageRole,
        device: u64,
        inode: u64,
    },
}

impl fmt::Display for NanoWheelsOffMappedImageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Linux wheels-off mapped-image admission failed: {self:?}"
        )
    }
}

impl std::error::Error for NanoWheelsOffMappedImageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::OpenProcessMaps { source } | Self::ReadProcessMaps { source } => Some(source),
            Self::ProcessMapsNotUtf8 { source } => Some(source),
            Self::UnsupportedPlatform { .. }
            | Self::MissingVerifiedDependency { .. }
            | Self::ExpectedIdentityAliased { .. }
            | Self::ProcessMapsTooLarge { .. }
            | Self::MalformedProcessMapsLine { .. }
            | Self::ExpectedImageNotExecutablyMapped { .. } => None,
        }
    }
}

#[derive(Debug)]
pub enum NanoWheelsOffNativeRuntimeParseError {
    InputTooLarge {
        actual_bytes: usize,
        maximum_bytes: u64,
    },
    JsonDecode(serde_json::Error),
    JsonTrailingData(serde_json::Error),
    UnsupportedSchema {
        actual: u32,
        supported: u32,
    },
    InvalidLibrarySearchPath(ArtifactRelativePathError),
    WrongLibrarySearchPath {
        actual: String,
    },
    WrongLibraryCount {
        actual: usize,
        required: usize,
    },
    UnknownLibraryRole {
        actual: String,
    },
    DuplicateLibraryRole {
        role: NanoWheelsOffNativeLibraryRole,
    },
    InvalidSoname {
        role: NanoWheelsOffNativeLibraryRole,
        actual: String,
    },
    WrongSonameForRole {
        role: NanoWheelsOffNativeLibraryRole,
        expected: &'static str,
        actual: String,
    },
    InvalidLibraryPath {
        role: NanoWheelsOffNativeLibraryRole,
        source: ArtifactRelativePathError,
    },
    LibraryOutsideSearchDirectory {
        role: NanoWheelsOffNativeLibraryRole,
        soname: String,
        relative_path: ArtifactRelativePath,
    },
    DuplicateLibraryPath {
        relative_path: ArtifactRelativePath,
    },
    InvalidLibraryByteLimit {
        role: NanoWheelsOffNativeLibraryRole,
        source: DeploymentAssetByteLimitError,
    },
    InvalidLibrarySha256 {
        role: NanoWheelsOffNativeLibraryRole,
        source: NanoLaunchSha256Error,
    },
}

impl fmt::Display for NanoWheelsOffNativeRuntimeParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid wheels-off native-runtime manifest: {self:?}"
        )
    }
}

impl std::error::Error for NanoWheelsOffNativeRuntimeParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::JsonDecode(source) | Self::JsonTrailingData(source) => Some(source),
            Self::InvalidLibrarySearchPath(source) => Some(source),
            Self::InvalidLibraryPath { source, .. } => Some(source),
            Self::InvalidLibraryByteLimit { source, .. } => Some(source),
            Self::InvalidLibrarySha256 { source, .. } => Some(source),
            Self::InputTooLarge { .. }
            | Self::UnsupportedSchema { .. }
            | Self::WrongLibrarySearchPath { .. }
            | Self::WrongLibraryCount { .. }
            | Self::UnknownLibraryRole { .. }
            | Self::DuplicateLibraryRole { .. }
            | Self::InvalidSoname { .. }
            | Self::WrongSonameForRole { .. }
            | Self::LibraryOutsideSearchDirectory { .. }
            | Self::DuplicateLibraryPath { .. } => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NanoWheelsOffNativeRuntimeBindingError {
    OnnxRuntimeLaunchMismatch,
    LibraryAliasesLaunchAsset {
        role: NanoWheelsOffNativeLibraryRole,
        relative_path: ArtifactRelativePath,
    },
}

impl fmt::Display for NanoWheelsOffNativeRuntimeBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "wheels-off native-runtime binding differs from the launch graph: {self:?}"
        )
    }
}

impl std::error::Error for NanoWheelsOffNativeRuntimeBindingError {}

#[derive(Debug)]
pub enum NanoWheelsOffNativeRuntimeVerificationError {
    VerifiedOnnxRuntimeMismatch,
    Library {
        role: NanoWheelsOffNativeLibraryRole,
        source: NanoLaunchBoundAssetLoadError,
    },
}

impl fmt::Display for NanoWheelsOffNativeRuntimeVerificationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "wheels-off native-runtime asset verification failed: {self:?}"
        )
    }
}

impl std::error::Error for NanoWheelsOffNativeRuntimeVerificationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Library { source, .. } => Some(source),
            Self::VerifiedOnnxRuntimeMismatch => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    use kiko_device_inventory::stream_deployment_asset_identity;
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};

    use super::*;

    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(1);

    fn manifest() -> Value {
        let libraries = [
            ("depthai_core", "libdepthai-core.so"),
            ("dynamic_calibration", "libdynamic_calibration.so"),
            ("libusb_1_0", "libusb-1.0.so.0"),
            ("onnxruntime", "libonnxruntime.so.1"),
            ("opencv_core", "libopencv_core.so.4.5d"),
            ("opencv_imgproc", "libopencv_imgproc.so.4.5d"),
            ("opencv_objdetect", "libopencv_objdetect.so.4.5d"),
        ]
        .map(|(role, soname)| {
            json!({
                "role": role,
                "soname": soname,
                "relative_path": format!("lib/{soname}"),
                "maximum_bytes": 1,
                "sha256_hex": "11".repeat(32),
            })
        });
        json!({
            "schema_version": 1,
            "library_search_relative_path": "lib",
            "libraries": libraries,
        })
    }

    #[test]
    fn parses_exact_closed_seven_role_contract() {
        let parsed = NanoWheelsOffNativeRuntimeV1::parse_json(
            &serde_json::to_vec(&manifest()).expect("fixture"),
        )
        .expect("strict manifest");
        assert_eq!(
            parsed.libraries().each_ref().map(|library| library.role()),
            NanoWheelsOffNativeLibraryRole::ALL
        );
    }

    #[test]
    fn missing_role_duplicate_role_and_wrong_search_path_fail_closed() {
        let mut missing = manifest();
        missing["libraries"]
            .as_array_mut()
            .expect("libraries")
            .pop();
        assert!(matches!(
            NanoWheelsOffNativeRuntimeV1::parse_json(
                &serde_json::to_vec(&missing).expect("fixture")
            ),
            Err(NanoWheelsOffNativeRuntimeParseError::WrongLibraryCount { .. })
        ));

        let mut duplicate = manifest();
        duplicate["libraries"][6]["role"] = json!("opencv_imgproc");
        duplicate["libraries"][6]["soname"] = json!("libopencv_imgproc.so.4.5d");
        duplicate["libraries"][6]["relative_path"] = json!("lib/libopencv_imgproc.so.4.5d");
        assert!(matches!(
            NanoWheelsOffNativeRuntimeV1::parse_json(
                &serde_json::to_vec(&duplicate).expect("fixture")
            ),
            Err(
                NanoWheelsOffNativeRuntimeParseError::DuplicateLibraryPath { .. }
                    | NanoWheelsOffNativeRuntimeParseError::DuplicateLibraryRole { .. }
            )
        ));

        let mut wrong_search = manifest();
        wrong_search["library_search_relative_path"] = json!("other");
        assert!(matches!(
            NanoWheelsOffNativeRuntimeV1::parse_json(
                &serde_json::to_vec(&wrong_search).expect("fixture")
            ),
            Err(NanoWheelsOffNativeRuntimeParseError::WrongLibrarySearchPath { .. })
        ));
    }

    #[test]
    fn onnx_launch_binding_must_match_path_limit_and_digest() {
        let parsed = NanoWheelsOffNativeRuntimeV1::parse_json(
            &serde_json::to_vec(&manifest()).expect("fixture"),
        )
        .expect("strict manifest");
        let onnx = parsed
            .library(NanoWheelsOffNativeLibraryRole::OnnxRuntime)
            .asset();
        parsed
            .bind_onnx_runtime_launch(onnx)
            .expect("same exact binding");
        let wrong = NanoLaunchAssetBinding::from_parsed_parts(
            ArtifactRelativePath::parse("lib/libonnxruntime-other.so".to_owned())
                .expect("fixture path"),
            onnx.byte_limit(),
            *onnx.expected_sha256(),
        );
        assert_eq!(
            parsed.bind_onnx_runtime_launch(&wrong),
            Err(NanoWheelsOffNativeRuntimeBindingError::OnnxRuntimeLaunchMismatch)
        );
        let aliased =
            ArtifactRelativePath::parse("lib/libdepthai-core.so".to_owned()).expect("fixture path");
        assert!(matches!(
            parsed.reject_non_onnx_launch_aliases([&aliased]),
            Err(
                NanoWheelsOffNativeRuntimeBindingError::LibraryAliasesLaunchAsset {
                    role: NanoWheelsOffNativeLibraryRole::DepthAiCore,
                    ..
                }
            )
        ));
    }

    #[test]
    fn required_native_roles_are_stream_verified_without_retaining_their_bytes() {
        let requested = std::env::temp_dir().join(format!(
            "kiko-native-runtime-{}-{}",
            std::process::id(),
            NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(requested.join("lib")).expect("fixture directory");
        let root = fs::canonicalize(&requested).expect("canonical fixture root");
        let mut value = manifest();
        for library in value["libraries"]
            .as_array_mut()
            .expect("fixture libraries")
        {
            let role = library["role"].as_str().expect("fixture role");
            let soname = library["soname"].as_str().expect("fixture soname");
            let bytes = format!("exact-{role}").into_bytes();
            fs::write(root.join("lib").join(soname), &bytes).expect("native fixture");
            library["maximum_bytes"] =
                json!(u64::try_from(bytes.len()).expect("fixture byte length"));
            library["sha256_hex"] = json!(
                Sha256::digest(&bytes)
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>()
            );
        }
        let parsed =
            NanoWheelsOffNativeRuntimeV1::parse_json(&serde_json::to_vec(&value).expect("fixture"))
                .expect("strict native runtime");
        let onnx = parsed
            .library(NanoWheelsOffNativeLibraryRole::OnnxRuntime)
            .asset();
        let verified_onnx = stream_deployment_asset_identity(
            &root,
            onnx.relative_path().clone(),
            onnx.byte_limit(),
        )
        .expect("stream-verified ONNX identity");
        let verified = parsed
            .verify_dependencies_reusing_onnx(&root, &verified_onnx)
            .expect("stream-verified required native roles");
        assert_eq!(verified.dependencies().len(), 6);
        assert!(
            verified
                .dependency(NanoWheelsOffNativeLibraryRole::OnnxRuntime)
                .is_none()
        );
        for role in NanoWheelsOffNativeLibraryRole::ALL {
            if role != NanoWheelsOffNativeLibraryRole::OnnxRuntime {
                let identity = verified.dependency(role).expect("verified dependency");
                assert_eq!(
                    identity.relative_path(),
                    parsed.library(role).asset().relative_path()
                );
                assert_eq!(
                    identity.content_sha256().as_bytes(),
                    parsed.library(role).asset().expected_sha256()
                );
            }
        }
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn process_maps_parser_uses_device_and_inode_not_path_text() {
        let parsed = parse_linux_process_maps(
            "00400000-00452000 r-xp 00000000 08:02 123 /opt/kiko/lib/a.so\n\
             7f000000-7f001000 r-xs 00001000 00:2a 456 /path with spaces/lib b.so (deleted)\n\
             7f100000-7f101000 r--p 00000000 08:02 789 /data-only.so\n\
             7fff0000-7fff1000 rw-p 00000000 00:00 0 [stack]\n",
        )
        .expect("well-formed Linux maps");
        assert_eq!(
            parsed,
            BTreeSet::from([
                LinuxMappedFileIdentity {
                    device_major: 8,
                    device_minor: 2,
                    inode: 123,
                },
                LinuxMappedFileIdentity {
                    device_major: 0,
                    device_minor: 42,
                    inode: 456,
                },
            ])
        );
        assert!(matches!(
            parse_linux_process_maps(
                "00400000-00452000 r-xp not-hex 08:02 123 /opt/kiko/lib/a.so\n"
            ),
            Err(NanoWheelsOffMappedImageError::MalformedProcessMapsLine { line_number: 1 })
        ));
    }

    #[test]
    fn mapped_image_admission_rejects_missing_and_aliased_expected_inodes() {
        let requested = std::env::temp_dir().join(format!(
            "kiko-mapped-image-{}-{}",
            std::process::id(),
            NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&requested).expect("fixture directory");
        let image = requested.join("image");
        fs::write(&image, b"mapped").expect("fixture image");
        let identity =
            UnixFileIdentity::from_metadata(&fs::metadata(&image).expect("fixture image metadata"));
        let linux_identity = LinuxMappedFileIdentity::from_unix(identity);
        let mapped = BTreeSet::from([linux_identity]);
        let role = NanoWheelsOffMappedImageRole::QualificationExecutable;

        let admitted =
            admit_mapped_images(vec![(role, identity)], &mapped).expect("same mapped device/inode");
        assert_eq!(admitted.file_identity(role), Some(identity));
        assert!(matches!(
            admit_mapped_images(vec![(role, identity)], &BTreeSet::new()),
            Err(
                NanoWheelsOffMappedImageError::ExpectedImageNotExecutablyMapped {
                    role: NanoWheelsOffMappedImageRole::QualificationExecutable,
                    ..
                }
            )
        ));
        assert!(matches!(
            admit_mapped_images(
                vec![
                    (role, identity),
                    (
                        NanoWheelsOffMappedImageRole::NativeLibrary(
                            NanoWheelsOffNativeLibraryRole::DepthAiCore
                        ),
                        identity,
                    ),
                ],
                &mapped,
            ),
            Err(NanoWheelsOffMappedImageError::ExpectedIdentityAliased { .. })
        ));
        fs::remove_dir_all(requested).expect("remove fixture");
    }
}
