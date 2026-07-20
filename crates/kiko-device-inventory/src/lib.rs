#![forbid(unsafe_code)]

//! Typed inventory domains plus a bounded Unix host boundary for one Kiko robot.
//!
//! Manifest loading and artifact hashing establish structural validity and
//! content identity only. Neither result proves provenance or authenticity.

mod artifact;
#[cfg(unix)]
mod artifact_io;
mod bounded;
mod compare;
mod manifest;
#[cfg(unix)]
mod manifest_io;
mod model;
mod observed;
#[cfg(unix)]
mod secure_fs;

use core::fmt;

use kiko_head_protocol::{HeadJoint, ServoId};

pub use artifact::{
    ArtifactDigest, ArtifactDigestDto, ArtifactKind, ArtifactSet, MAX_ARTIFACTS,
    MAX_CALIBRATION_ARTIFACTS, MAX_PLANT_ARTIFACTS,
};
#[cfg(unix)]
pub use artifact_io::{
    ArtifactContentIdentity, ArtifactFileBinding, ArtifactFileBindingInput,
    ArtifactFileBindingParseError, ArtifactFileBindingSet, ArtifactHashError, ArtifactRelativePath,
    ArtifactRelativePathError, CalibrationBundleHashError, ExactCalibrationBundleSha256,
    MAX_ARTIFACT_FILE_BYTES, MAX_ARTIFACT_PATH_COMPONENTS, MAX_ARTIFACT_RELATIVE_PATH_BYTES,
    MAX_ARTIFACT_ROOT_PATH_BYTES, ManifestArtifactHashes, hash_manifest_artifacts,
};
pub use bounded::{
    ArtifactId, BoundedTextError, BuildProvenance, ControlEndpointIdentity, MAX_ARTIFACT_ID_BYTES,
    MAX_BUILD_PROVENANCE_BYTES, MAX_CONTROL_ENDPOINT_ID_BYTES, MAX_OAK_MXID_BYTES,
    MAX_ROBOT_ID_BYTES, MAX_SERIAL_BY_ID_PATH_BYTES, OakMxid, PersistentSerialPath, RobotId,
    Sha256Id,
};
pub use compare::{InventoryComparison, InventoryMismatch, MAX_INVENTORY_MISMATCHES};
pub use manifest::{
    DEVICE_INVENTORY_MANIFEST_V1, DeviceInventoryManifestV1, DeviceInventoryManifestV1Dto,
    EyeManifestV1Dto, HeadManifestV1Dto, OakManifestV1Dto, REQUIRED_EYE_CAPABILITY_BITS,
    Stm32ManifestV1Dto,
};
#[cfg(unix)]
pub use manifest_io::{
    FileKind, LoadedExpectedManifestV1, MAX_MANIFEST_JSON_BYTES, MAX_MANIFEST_PATH_BYTES,
    ManifestContentSha256, ManifestJsonError, ManifestLoadError, load_expected_manifest_v1_file,
    load_expected_manifest_v1_from_slice,
};
pub use model::{
    DeviceRole, EyeStaticIdentity, HeadExpectedIdentity, MAX_OBSERVED_HEAD_SERVOS, OakIdentity,
    ObservedEye, ObservedHead, ObservedServoIds, ObservedStm32, Stm32StaticIdentity,
};
pub use observed::{
    OBSERVED_DEVICE_INVENTORY_V1, ObservedDeviceInventoryV1, ObservedDeviceInventoryV1Dto,
    ObservedEyeV1Dto, ObservedHeadV1Dto, ObservedStm32V1Dto,
};
#[cfg(unix)]
pub use secure_fs::SecureOpenError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextField {
    RobotId,
    OakMxid,
    OakRuntimeProvenance,
    OakSdkBuildProvenance,
    OakAdapterBuildProvenance,
    SerialPath(DeviceRole),
    Stm32ControlEndpoint,
    ArtifactId { kind: ArtifactKind, index: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InventoryParseError {
    UnsupportedManifestSchema {
        actual: u32,
        supported: u32,
    },
    UnsupportedObservedSchema {
        actual: u32,
        supported: u32,
    },
    InvalidText {
        field: TextField,
        source: BoundedTextError,
    },
    InvalidControllerUid(robot_protocol::v2::DomainError),
    InvalidControllerBootId(robot_protocol::v2::DomainError),
    ZeroStm32FirmwareAbi,
    ZeroStm32FirmwareBuildId,
    InvalidHardwareProfile(robot_protocol::v2::DomainError),
    InvalidControllerCapabilities(robot_protocol::v2::DomainError),
    Stm32FirmwareAbiContractMismatch {
        actual: u16,
        required: u16,
    },
    MissingControllerSafetyCapabilities {
        actual_bits: u32,
        required_bits: u32,
    },
    InvalidHeadServoId {
        index: usize,
        source: kiko_head_protocol::FrameBuildError,
    },
    DuplicateHeadServoId {
        servo_id: ServoId,
    },
    HeadServoContractMismatch {
        joint: HeadJoint,
        expected: ServoId,
        actual: ServoId,
    },
    HeadElectricalContractMismatch {
        actual_baud_rate_bps: u32,
        required_baud_rate_bps: u32,
        actual_dtr_asserted: bool,
        required_dtr_asserted: bool,
        actual_rts_asserted: bool,
        required_rts_asserted: bool,
    },
    ZeroObservedHeadBaudRate,
    TooManyObservedHeadServos {
        actual: usize,
        maximum: usize,
    },
    ZeroEyeProtocolVersion,
    InvalidEyeDeviceUid(kiko_eye_protocol::DomainError),
    InvalidEyeFirmwareBuildId(kiko_eye_protocol::DomainError),
    InvalidEyeBootId(kiko_eye_protocol::DomainError),
    InvalidEyeCapabilities(kiko_eye_protocol::DomainError),
    EyeProtocolContractMismatch {
        actual: u8,
        required: u8,
    },
    MissingEyeCapabilities {
        actual_bits: u32,
        required_bits: u32,
    },
    DuplicatePhysicalPath {
        first: DeviceRole,
        second: DeviceRole,
    },
    MissingRequiredArtifactKind {
        kind: ArtifactKind,
    },
    TooManyArtifacts {
        kind: ArtifactKind,
        actual: usize,
        maximum: usize,
    },
    ZeroArtifactDigest {
        kind: ArtifactKind,
        index: usize,
        artifact_id: ArtifactId,
    },
    DuplicateArtifactId {
        artifact_id: ArtifactId,
    },
    DuplicateArtifactDigest {
        sha256: Sha256Id,
    },
}

impl fmt::Display for InventoryParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid Kiko device inventory: {self:?}")
    }
}

impl std::error::Error for InventoryParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidText { source, .. } => Some(source),
            Self::InvalidControllerUid(source)
            | Self::InvalidControllerBootId(source)
            | Self::InvalidHardwareProfile(source)
            | Self::InvalidControllerCapabilities(source) => Some(source),
            Self::InvalidHeadServoId { source, .. } => Some(source),
            Self::InvalidEyeDeviceUid(source)
            | Self::InvalidEyeFirmwareBuildId(source)
            | Self::InvalidEyeBootId(source)
            | Self::InvalidEyeCapabilities(source) => Some(source),
            _ => None,
        }
    }
}

pub(crate) fn ensure_unique_physical_paths(
    stm32: &PersistentSerialPath,
    head: Option<&PersistentSerialPath>,
    eye: Option<&PersistentSerialPath>,
) -> Result<(), InventoryParseError> {
    for (first_role, first_path, second_role, second_path) in [
        (DeviceRole::Stm32, Some(stm32), DeviceRole::Head, head),
        (DeviceRole::Stm32, Some(stm32), DeviceRole::Eye, eye),
        (DeviceRole::Head, head, DeviceRole::Eye, eye),
    ] {
        if let (Some(first_path), Some(second_path)) = (first_path, second_path)
            && first_path == second_path
        {
            return Err(InventoryParseError::DuplicatePhysicalPath {
                first: first_role,
                second: second_role,
            });
        }
    }
    Ok(())
}
