use core::num::{NonZeroU16, NonZeroU32};

use kiko_eye_protocol::{Capabilities as EyeCapabilities, DeviceUid, FirmwareBuildId};
use kiko_head_protocol::{HeadJoint, ServoId};
use robot_protocol::v2::{
    ActuatorConfigFingerprint, ControllerCapabilities, ControllerUid, VERSION as ROBOT_PROTOCOL_V2,
};
use serde::Deserialize;

use crate::{
    ArtifactDigestDto, ArtifactSet, BuildProvenance, ControlEndpointIdentity, DeviceRole,
    EyeStaticIdentity, HeadExpectedIdentity, InventoryParseError, OakIdentity, OakMxid,
    PersistentSerialPath, RobotId, Stm32StaticIdentity, TextField, ensure_unique_physical_paths,
};

pub const DEVICE_INVENTORY_MANIFEST_V1: u32 = 1;
pub const REQUIRED_EYE_CAPABILITY_BITS: u32 = EyeCapabilities::KNOWN_BITS;

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OakManifestV1Dto {
    pub mxid: String,
    pub compiled_depthai_header_sdk_version: String,
    pub compiled_depthai_header_sdk_commit: String,
    pub compiled_depthai_header_embedded_device_artifact_version: String,
    pub compiled_depthai_header_embedded_bootloader_artifact_version: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Stm32ManifestV1Dto {
    pub serial_by_id_path: String,
    pub control_endpoint_identity: String,
    pub controller_uid: [u8; 12],
    pub firmware_abi: u16,
    pub firmware_build_id: u32,
    pub hardware_profile_fingerprint: [u8; 16],
    pub capabilities_bits: u32,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HeadManifestV1Dto {
    pub adapter_serial_by_id_path: String,
    pub bow_servo_id: u8,
    pub curl_servo_id: u8,
    pub yaw_servo_id: u8,
    pub roll_servo_id: u8,
    pub baud_rate_bps: u32,
    pub dtr_asserted: bool,
    pub rts_asserted: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EyeManifestV1Dto {
    pub serial_by_id_path: String,
    pub kep_protocol_version: u8,
    pub device_uid: [u8; 16],
    pub firmware_build_id: [u8; 32],
    pub capabilities_bits: u32,
}

/// Weak, versioned expectation document for one exact robot.
///
/// OAK, STM32, calibration artifacts, and plant artifacts are mandatory for
/// the mobile SLAM base. Head and eye are the only optional expected devices,
/// because a base can be physically built without either expressive accessory.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DeviceInventoryManifestV1Dto {
    pub schema_version: u32,
    pub robot_id: String,
    pub oak: OakManifestV1Dto,
    pub stm32: Stm32ManifestV1Dto,
    pub head: Option<HeadManifestV1Dto>,
    pub eye: Option<EyeManifestV1Dto>,
    pub calibration_artifacts: Vec<ArtifactDigestDto>,
    pub plant_artifacts: Vec<ArtifactDigestDto>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInventoryManifestV1 {
    robot_id: RobotId,
    oak: OakIdentity,
    stm32: Stm32StaticIdentity,
    head: Option<HeadExpectedIdentity>,
    eye: Option<EyeStaticIdentity>,
    artifacts: ArtifactSet,
}

impl DeviceInventoryManifestV1 {
    pub fn parse(dto: DeviceInventoryManifestV1Dto) -> Result<Self, InventoryParseError> {
        if dto.schema_version != DEVICE_INVENTORY_MANIFEST_V1 {
            return Err(InventoryParseError::UnsupportedManifestSchema {
                actual: dto.schema_version,
                supported: DEVICE_INVENTORY_MANIFEST_V1,
            });
        }
        let robot_id =
            RobotId::parse(dto.robot_id).map_err(|source| InventoryParseError::InvalidText {
                field: TextField::RobotId,
                source,
            })?;
        let oak = parse_oak(dto.oak)?;
        let stm32 = parse_expected_stm32(dto.stm32)?;
        let head = dto.head.map(parse_expected_head).transpose()?;
        let eye = dto.eye.map(parse_expected_eye).transpose()?;
        ensure_unique_physical_paths(
            stm32.serial_path(),
            head.as_ref().map(HeadExpectedIdentity::serial_path),
            eye.as_ref().map(EyeStaticIdentity::serial_path),
        )?;
        let artifacts =
            ArtifactSet::parse_expected(dto.calibration_artifacts, dto.plant_artifacts)?;
        Ok(Self {
            robot_id,
            oak,
            stm32,
            head,
            eye,
            artifacts,
        })
    }

    pub fn robot_id(&self) -> &RobotId {
        &self.robot_id
    }

    pub fn oak(&self) -> &OakIdentity {
        &self.oak
    }

    pub fn stm32(&self) -> &Stm32StaticIdentity {
        &self.stm32
    }

    pub fn head(&self) -> Option<&HeadExpectedIdentity> {
        self.head.as_ref()
    }

    pub fn eye(&self) -> Option<&EyeStaticIdentity> {
        self.eye.as_ref()
    }

    pub fn artifacts(&self) -> &ArtifactSet {
        &self.artifacts
    }
}

pub(crate) fn parse_oak(dto: OakManifestV1Dto) -> Result<OakIdentity, InventoryParseError> {
    let mxid = OakMxid::parse(dto.mxid).map_err(|source| InventoryParseError::InvalidText {
        field: TextField::OakMxid,
        source,
    })?;
    let compiled_depthai_header_sdk_version =
        BuildProvenance::parse(dto.compiled_depthai_header_sdk_version).map_err(|source| {
            InventoryParseError::InvalidText {
                field: TextField::OakCompiledDepthAiHeaderSdkVersion,
                source,
            }
        })?;
    let compiled_depthai_header_sdk_commit =
        BuildProvenance::parse(dto.compiled_depthai_header_sdk_commit).map_err(|source| {
            InventoryParseError::InvalidText {
                field: TextField::OakCompiledDepthAiHeaderSdkCommit,
                source,
            }
        })?;
    let compiled_depthai_header_embedded_device_artifact_version =
        BuildProvenance::parse(dto.compiled_depthai_header_embedded_device_artifact_version)
            .map_err(|source| InventoryParseError::InvalidText {
                field: TextField::OakCompiledDepthAiHeaderEmbeddedDeviceArtifactVersion,
                source,
            })?;
    let compiled_depthai_header_embedded_bootloader_artifact_version =
        BuildProvenance::parse(dto.compiled_depthai_header_embedded_bootloader_artifact_version)
            .map_err(|source| InventoryParseError::InvalidText {
                field: TextField::OakCompiledDepthAiHeaderEmbeddedBootloaderArtifactVersion,
                source,
            })?;
    Ok(OakIdentity::new(
        mxid,
        compiled_depthai_header_sdk_version,
        compiled_depthai_header_sdk_commit,
        compiled_depthai_header_embedded_device_artifact_version,
        compiled_depthai_header_embedded_bootloader_artifact_version,
    ))
}

pub(crate) fn parse_stm32_static(
    serial_by_id_path: String,
    control_endpoint_identity: String,
    controller_uid: [u8; 12],
    firmware_abi: u16,
    firmware_build_id: u32,
    hardware_profile_fingerprint: [u8; 16],
    capabilities_bits: u32,
) -> Result<Stm32StaticIdentity, InventoryParseError> {
    let serial_path = PersistentSerialPath::parse(serial_by_id_path).map_err(|source| {
        InventoryParseError::InvalidText {
            field: TextField::SerialPath(DeviceRole::Stm32),
            source,
        }
    })?;
    let control_endpoint =
        ControlEndpointIdentity::parse(control_endpoint_identity).map_err(|source| {
            InventoryParseError::InvalidText {
                field: TextField::Stm32ControlEndpoint,
                source,
            }
        })?;
    let controller_uid = ControllerUid::try_new(controller_uid)
        .map_err(InventoryParseError::InvalidControllerUid)?;
    let firmware_abi =
        NonZeroU16::new(firmware_abi).ok_or(InventoryParseError::ZeroStm32FirmwareAbi)?;
    let firmware_build_id =
        NonZeroU32::new(firmware_build_id).ok_or(InventoryParseError::ZeroStm32FirmwareBuildId)?;
    let hardware_profile = ActuatorConfigFingerprint::try_new(hardware_profile_fingerprint)
        .map_err(InventoryParseError::InvalidHardwareProfile)?;
    let capabilities = ControllerCapabilities::try_from_bits(capabilities_bits)
        .map_err(InventoryParseError::InvalidControllerCapabilities)?;
    Ok(Stm32StaticIdentity::new(
        serial_path,
        control_endpoint,
        controller_uid,
        firmware_abi,
        firmware_build_id,
        hardware_profile,
        capabilities,
    ))
}

fn parse_expected_stm32(
    dto: Stm32ManifestV1Dto,
) -> Result<Stm32StaticIdentity, InventoryParseError> {
    let parsed = parse_stm32_static(
        dto.serial_by_id_path,
        dto.control_endpoint_identity,
        dto.controller_uid,
        dto.firmware_abi,
        dto.firmware_build_id,
        dto.hardware_profile_fingerprint,
        dto.capabilities_bits,
    )?;
    let required_abi = u16::from(ROBOT_PROTOCOL_V2);
    if parsed.firmware_abi() != required_abi {
        return Err(InventoryParseError::Stm32FirmwareAbiContractMismatch {
            actual: parsed.firmware_abi(),
            required: required_abi,
        });
    }
    if !parsed.capabilities().supports_required_safety() {
        return Err(InventoryParseError::MissingControllerSafetyCapabilities {
            actual_bits: parsed.capabilities().bits(),
            required_bits: ControllerCapabilities::REQUIRED_BITS,
        });
    }
    Ok(parsed)
}

fn parse_expected_head(
    dto: HeadManifestV1Dto,
) -> Result<HeadExpectedIdentity, InventoryParseError> {
    let serial_path =
        PersistentSerialPath::parse(dto.adapter_serial_by_id_path).map_err(|source| {
            InventoryParseError::InvalidText {
                field: TextField::SerialPath(DeviceRole::Head),
                source,
            }
        })?;
    let raw_ids = [
        dto.bow_servo_id,
        dto.curl_servo_id,
        dto.yaw_servo_id,
        dto.roll_servo_id,
    ];
    let mut servo_ids = [HeadJoint::Bow.servo_id(); 4];
    for (index, raw) in raw_ids.into_iter().enumerate() {
        servo_ids[index] = ServoId::try_new(raw)
            .map_err(|source| InventoryParseError::InvalidHeadServoId { index, source })?;
    }
    for right in 0..servo_ids.len() {
        if servo_ids[..right].contains(&servo_ids[right]) {
            return Err(InventoryParseError::DuplicateHeadServoId {
                servo_id: servo_ids[right],
            });
        }
    }
    for (index, joint) in HeadJoint::ALL.into_iter().enumerate() {
        let expected = joint.servo_id();
        if servo_ids[index] != expected {
            return Err(InventoryParseError::HeadServoContractMismatch {
                joint,
                expected,
                actual: servo_ids[index],
            });
        }
    }
    if dto.baud_rate_bps != kiko_head_protocol::BUS_BAUD_RATE_BPS
        || dto.dtr_asserted != kiko_head_protocol::ADAPTER_DTR_ASSERTED
        || dto.rts_asserted != kiko_head_protocol::ADAPTER_RTS_ASSERTED
    {
        return Err(InventoryParseError::HeadElectricalContractMismatch {
            actual_baud_rate_bps: dto.baud_rate_bps,
            required_baud_rate_bps: kiko_head_protocol::BUS_BAUD_RATE_BPS,
            actual_dtr_asserted: dto.dtr_asserted,
            required_dtr_asserted: kiko_head_protocol::ADAPTER_DTR_ASSERTED,
            actual_rts_asserted: dto.rts_asserted,
            required_rts_asserted: kiko_head_protocol::ADAPTER_RTS_ASSERTED,
        });
    }
    Ok(HeadExpectedIdentity::new(serial_path, servo_ids))
}

pub(crate) fn parse_eye_static(
    serial_by_id_path: String,
    protocol_version: u8,
    device_uid: [u8; 16],
    firmware_build_id: [u8; 32],
    capabilities_bits: u32,
) -> Result<EyeStaticIdentity, InventoryParseError> {
    let serial_path = PersistentSerialPath::parse(serial_by_id_path).map_err(|source| {
        InventoryParseError::InvalidText {
            field: TextField::SerialPath(DeviceRole::Eye),
            source,
        }
    })?;
    if protocol_version == 0 {
        return Err(InventoryParseError::ZeroEyeProtocolVersion);
    }
    let device_uid =
        DeviceUid::try_new(device_uid).map_err(InventoryParseError::InvalidEyeDeviceUid)?;
    let firmware_build_id = FirmwareBuildId::try_new(firmware_build_id)
        .map_err(InventoryParseError::InvalidEyeFirmwareBuildId)?;
    let capabilities = EyeCapabilities::try_from_bits(capabilities_bits)
        .map_err(InventoryParseError::InvalidEyeCapabilities)?;
    Ok(EyeStaticIdentity::new(
        serial_path,
        protocol_version,
        device_uid,
        firmware_build_id,
        capabilities,
    ))
}

fn parse_expected_eye(dto: EyeManifestV1Dto) -> Result<EyeStaticIdentity, InventoryParseError> {
    let parsed = parse_eye_static(
        dto.serial_by_id_path,
        dto.kep_protocol_version,
        dto.device_uid,
        dto.firmware_build_id,
        dto.capabilities_bits,
    )?;
    if parsed.protocol_version() != kiko_eye_protocol::PROTOCOL_VERSION {
        return Err(InventoryParseError::EyeProtocolContractMismatch {
            actual: parsed.protocol_version(),
            required: kiko_eye_protocol::PROTOCOL_VERSION,
        });
    }
    if parsed.capabilities().bits() & REQUIRED_EYE_CAPABILITY_BITS != REQUIRED_EYE_CAPABILITY_BITS {
        return Err(InventoryParseError::MissingEyeCapabilities {
            actual_bits: parsed.capabilities().bits(),
            required_bits: REQUIRED_EYE_CAPABILITY_BITS,
        });
    }
    Ok(parsed)
}
