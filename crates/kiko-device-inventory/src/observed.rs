use kiko_eye_protocol::DeviceBootId as EyeBootId;
use kiko_head_protocol::ServoId;
use robot_protocol::v2::ControllerBootId;

use crate::{
    ArtifactDigestDto, ArtifactSet, DeviceRole, InventoryParseError, OakIdentity, OakManifestV1Dto,
    ObservedEye, ObservedHead, ObservedServoIds, ObservedStm32, PersistentSerialPath, RobotId,
    TextField, ensure_unique_physical_paths,
    manifest::{parse_eye_static, parse_oak, parse_stm32_static},
    model::MAX_OBSERVED_HEAD_SERVOS,
};

pub const OBSERVED_DEVICE_INVENTORY_V1: u32 = 1;

/// Caller-supplied OAK evidence with the same field semantics as the manifest.
///
/// `mxid` must come from the identity of the device that was actually opened.
/// The remaining values come from `dai::build::*` constants in the DepthAI
/// header used to compile the native bridge. They prove neither the identity
/// of a linked/runtime DepthAI library nor firmware or bootloader readback from
/// the physical device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservedOakV1Dto {
    pub mxid: String,
    pub compiled_depthai_header_sdk_version: String,
    pub compiled_depthai_header_sdk_commit: String,
    pub compiled_depthai_header_embedded_device_artifact_version: String,
    pub compiled_depthai_header_embedded_bootloader_artifact_version: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservedStm32V1Dto {
    pub serial_by_id_path: String,
    pub control_endpoint_identity: String,
    pub controller_uid: [u8; 12],
    pub controller_boot_id: u64,
    pub firmware_abi: u16,
    pub firmware_build_id: u32,
    pub hardware_profile_fingerprint: [u8; 16],
    pub capabilities_bits: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservedHeadV1Dto {
    pub adapter_serial_by_id_path: String,
    pub baud_rate_bps: u32,
    pub dtr_asserted: bool,
    pub rts_asserted: bool,
    pub responding_servo_ids: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservedEyeV1Dto {
    pub serial_by_id_path: String,
    pub kep_protocol_version: u8,
    pub device_uid: [u8; 16],
    pub firmware_build_id: [u8; 32],
    pub device_boot_id: u64,
    pub capabilities_bits: u32,
}

/// Caller-supplied results from one external inventory probe.
///
/// `None` means the caller observed no exact device for that role. Parsing does
/// not probe hardware and does not elevate these claims to physical evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservedDeviceInventoryV1Dto {
    pub schema_version: u32,
    pub robot_id: String,
    pub oak: Option<ObservedOakV1Dto>,
    pub stm32: Option<ObservedStm32V1Dto>,
    pub head: Option<ObservedHeadV1Dto>,
    pub eye: Option<ObservedEyeV1Dto>,
    pub calibration_artifacts: Vec<ArtifactDigestDto>,
    pub plant_artifacts: Vec<ArtifactDigestDto>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservedDeviceInventoryV1 {
    robot_id: RobotId,
    oak: Option<OakIdentity>,
    stm32: Option<ObservedStm32>,
    head: Option<ObservedHead>,
    eye: Option<ObservedEye>,
    artifacts: ArtifactSet,
}

impl ObservedDeviceInventoryV1 {
    pub fn parse(dto: ObservedDeviceInventoryV1Dto) -> Result<Self, InventoryParseError> {
        if dto.schema_version != OBSERVED_DEVICE_INVENTORY_V1 {
            return Err(InventoryParseError::UnsupportedObservedSchema {
                actual: dto.schema_version,
                supported: OBSERVED_DEVICE_INVENTORY_V1,
            });
        }
        let robot_id =
            RobotId::parse(dto.robot_id).map_err(|source| InventoryParseError::InvalidText {
                field: TextField::RobotId,
                source,
            })?;
        let oak = dto.oak.map(parse_observed_oak).transpose()?;
        let stm32 = dto.stm32.map(parse_observed_stm32).transpose()?;
        let head = dto.head.map(parse_observed_head).transpose()?;
        let eye = dto.eye.map(parse_observed_eye).transpose()?;
        let artifacts =
            ArtifactSet::parse_observed(dto.calibration_artifacts, dto.plant_artifacts)?;
        let stm32_path = stm32
            .as_ref()
            .map(|value| value.static_identity().serial_path());
        if let Some(stm32_path) = stm32_path {
            ensure_unique_physical_paths(
                stm32_path,
                head.as_ref().map(ObservedHead::serial_path),
                eye.as_ref()
                    .map(|value| value.static_identity().serial_path()),
            )?;
        } else if let (Some(head_path), Some(eye_path)) = (
            head.as_ref().map(ObservedHead::serial_path),
            eye.as_ref()
                .map(|value| value.static_identity().serial_path()),
        ) && head_path == eye_path
        {
            return Err(InventoryParseError::DuplicatePhysicalPath {
                first: DeviceRole::Head,
                second: DeviceRole::Eye,
            });
        }
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

    pub fn oak(&self) -> Option<&OakIdentity> {
        self.oak.as_ref()
    }

    pub fn stm32(&self) -> Option<&ObservedStm32> {
        self.stm32.as_ref()
    }

    pub fn head(&self) -> Option<&ObservedHead> {
        self.head.as_ref()
    }

    pub fn eye(&self) -> Option<&ObservedEye> {
        self.eye.as_ref()
    }

    pub fn artifacts(&self) -> &ArtifactSet {
        &self.artifacts
    }
}

fn parse_observed_oak(dto: ObservedOakV1Dto) -> Result<OakIdentity, InventoryParseError> {
    parse_oak(OakManifestV1Dto {
        mxid: dto.mxid,
        compiled_depthai_header_sdk_version: dto.compiled_depthai_header_sdk_version,
        compiled_depthai_header_sdk_commit: dto.compiled_depthai_header_sdk_commit,
        compiled_depthai_header_embedded_device_artifact_version: dto
            .compiled_depthai_header_embedded_device_artifact_version,
        compiled_depthai_header_embedded_bootloader_artifact_version: dto
            .compiled_depthai_header_embedded_bootloader_artifact_version,
    })
}

fn parse_observed_stm32(dto: ObservedStm32V1Dto) -> Result<ObservedStm32, InventoryParseError> {
    let static_identity = parse_stm32_static(
        dto.serial_by_id_path,
        dto.control_endpoint_identity,
        dto.controller_uid,
        dto.firmware_abi,
        dto.firmware_build_id,
        dto.hardware_profile_fingerprint,
        dto.capabilities_bits,
    )?;
    let boot_id = ControllerBootId::try_new(dto.controller_boot_id)
        .map_err(InventoryParseError::InvalidControllerBootId)?;
    Ok(ObservedStm32::new(static_identity, boot_id))
}

fn parse_observed_head(dto: ObservedHeadV1Dto) -> Result<ObservedHead, InventoryParseError> {
    let serial_path =
        PersistentSerialPath::parse(dto.adapter_serial_by_id_path).map_err(|source| {
            InventoryParseError::InvalidText {
                field: TextField::SerialPath(DeviceRole::Head),
                source,
            }
        })?;
    if dto.baud_rate_bps == 0 {
        return Err(InventoryParseError::ZeroObservedHeadBaudRate);
    }
    let servo_count = dto.responding_servo_ids.len();
    if servo_count > MAX_OBSERVED_HEAD_SERVOS {
        return Err(InventoryParseError::TooManyObservedHeadServos {
            actual: servo_count,
            maximum: MAX_OBSERVED_HEAD_SERVOS,
        });
    }
    let mut entries = [None; MAX_OBSERVED_HEAD_SERVOS];
    for (index, raw) in dto.responding_servo_ids.into_iter().enumerate() {
        let servo_id = ServoId::try_new(raw)
            .map_err(|source| InventoryParseError::InvalidHeadServoId { index, source })?;
        if entries[..index].contains(&Some(servo_id)) {
            return Err(InventoryParseError::DuplicateHeadServoId { servo_id });
        }
        entries[index] = Some(servo_id);
    }
    Ok(ObservedHead::new(
        serial_path,
        dto.baud_rate_bps,
        dto.dtr_asserted,
        dto.rts_asserted,
        ObservedServoIds::from_parsed(entries, servo_count),
    ))
}

fn parse_observed_eye(dto: ObservedEyeV1Dto) -> Result<ObservedEye, InventoryParseError> {
    let static_identity = parse_eye_static(
        dto.serial_by_id_path,
        dto.kep_protocol_version,
        dto.device_uid,
        dto.firmware_build_id,
        dto.capabilities_bits,
    )?;
    let boot_id =
        EyeBootId::try_new(dto.device_boot_id).map_err(InventoryParseError::InvalidEyeBootId)?;
    Ok(ObservedEye::new(static_identity, boot_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiko_head_protocol::HeadJoint;

    #[test]
    fn observed_head_sorts_ids_and_retains_no_unparsed_tail() {
        let observed = parse_observed_head(ObservedHeadV1Dto {
            adapter_serial_by_id_path: "/dev/serial/by-id/usb-head".into(),
            baud_rate_bps: kiko_head_protocol::BUS_BAUD_RATE_BPS,
            dtr_asserted: false,
            rts_asserted: true,
            responding_servo_ids: vec![4, 2, 1, 3],
        })
        .expect("observed head");
        assert_eq!(
            observed
                .servo_ids()
                .iter()
                .map(ServoId::get)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        assert_eq!(HeadJoint::ALL.len(), observed.servo_ids().len());
    }
}
