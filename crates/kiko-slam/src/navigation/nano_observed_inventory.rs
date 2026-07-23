//! Production observed-inventory construction from retained runtime evidence.
//!
//! This boundary deliberately does not accept an expected device manifest.
//! The robot and command endpoint come from the parsed actuation deployment;
//! the OAK fields come from the device that was actually opened plus the
//! DepthAI header compiled into `oak-sys`; controller fields come from one
//! verified acquisition; head and eye fields come from read-only probes; and
//! artifact digests come from the bytes actually hashed without following
//! symlinks.
//!
//! Head probing performs only torque-switch and telemetry reads. Eye probing
//! performs one nonce-bound identity query and never acquires expression
//! control. Starting either accessory actor belongs after exact inventory
//! admission.

use std::fmt;
#[cfg(feature = "nano-wheels-off-qualification")]
use std::net::SocketAddr;
use std::path::Path;

use kiko_device_inventory::{
    ArtifactDigestDto, ArtifactKind, InventoryParseError, ManifestArtifactHashes,
    OBSERVED_DEVICE_INVENTORY_V1, ObservedDeviceInventoryV1, ObservedDeviceInventoryV1Dto,
    ObservedEyeV1Dto, ObservedHeadV1Dto, ObservedOakV1Dto, ObservedStm32V1Dto,
};
use kiko_eye_runtime::{
    EyeIdentityObservation, SerialConfigurationEvidence as EyeSerialConfigurationEvidence,
};
use kiko_head_protocol::{
    ADAPTER_DTR_ASSERTED, ADAPTER_RTS_ASSERTED, BUS_BAUD_RATE_BPS, HeadJoint,
};
use kiko_head_runtime::{
    HeadProbeReport, SerialConfigurationEvidence as HeadSerialConfigurationEvidence,
};
use oak_sys::{
    ConnectedDeviceIdentity, DepthAiBuildMetadata, UsbTransportAdmissionEvidence, UsbTransportSpeed,
};
use robot_command_client::VerifiedControllerAcquisition;

use super::NavigationActuationConfigV1;

const KEP2_SERIAL_BAUD_RATE_BPS: u32 = 115_200;

/// One required source of observed production inventory evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoObservedInventoryEvidenceKind {
    Deployment,
    Oak,
    Stm32,
    Head,
    Eye,
    Artifacts,
}

/// Retained proof that the already-opened OAK link both required and observed
/// USB 3 SuperSpeed or better.
///
/// Construction is private. This fact is intentionally kept beside the
/// existing inventory domain because `ObservedDeviceInventoryV1` has no USB
/// transport field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmittedOakSuperSpeedEvidence {
    requested_maximum: UsbTransportSpeed,
    required_minimum: UsbTransportSpeed,
    observed: UsbTransportSpeed,
}

impl AdmittedOakSuperSpeedEvidence {
    pub const fn requested_maximum(self) -> UsbTransportSpeed {
        self.requested_maximum
    }

    pub const fn required_minimum(self) -> UsbTransportSpeed {
        self.required_minimum
    }

    pub const fn observed(self) -> UsbTransportSpeed {
        self.observed
    }
}

/// Observed inventory that can be constructed only through live production
/// evidence, plus the admitted OAK transport fact absent from the V1 DTO.
///
/// This is an instantaneous startup observation. It does not claim continuing
/// device liveness after the underlying probes and acquisitions completed.
#[derive(Debug)]
pub struct ProductionObservedDeviceInventoryV1 {
    inventory: ObservedDeviceInventoryV1,
    oak_super_speed: AdmittedOakSuperSpeedEvidence,
}

impl ProductionObservedDeviceInventoryV1 {
    pub const fn inventory(&self) -> &ObservedDeviceInventoryV1 {
        &self.inventory
    }

    pub const fn oak_super_speed(&self) -> AdmittedOakSuperSpeedEvidence {
        self.oak_super_speed
    }

    /// Consume the proof-carrying production snapshot for exact comparison.
    ///
    /// Production admission should accept this wrapper directly and call this
    /// only inside that boundary; accepting arbitrary parsed observed DTOs
    /// would reopen the manifest-copying path this type closes.
    pub fn into_inventory(self) -> ObservedDeviceInventoryV1 {
        self.inventory
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DeploymentObservation {
    robot_id: String,
    control_endpoint_identity: String,
    controller_expectation: DeploymentControllerExpectation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeploymentControllerExpectation {
    ProductionActuation {
        controller_uid: [u8; 12],
        firmware_abi: u16,
        firmware_build_id: u32,
        hardware_profile_fingerprint: [u8; 16],
    },
    /// Qualification binds the candidate server/client/manifest before
    /// acquisition, then lets exact inventory comparison bind the acquisition
    /// fields. No expected controller identity is copied into observation.
    #[cfg(feature = "nano-wheels-off-qualification")]
    CandidateExactInventory,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct OakObservation {
    mxid: String,
    compiled_depthai_header_sdk_version: String,
    compiled_depthai_header_sdk_commit: String,
    compiled_depthai_header_embedded_device_artifact_version: String,
    compiled_depthai_header_embedded_bootloader_artifact_version: String,
    super_speed: AdmittedOakSuperSpeedEvidence,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stm32Observation {
    serial_by_id_path: String,
    controller_uid: [u8; 12],
    controller_boot_id: u64,
    firmware_abi: u16,
    firmware_build_id: u32,
    hardware_profile_fingerprint: [u8; 16],
    capabilities_bits: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct HeadObservation {
    adapter_serial_by_id_path: String,
    baud_rate_bps: u32,
    dtr_asserted: bool,
    rts_asserted: bool,
    responding_servo_ids: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct EyeObservation {
    serial_by_id_path: String,
    kep_protocol_version: u8,
    device_uid: [u8; 16],
    firmware_build_id: [u8; 32],
    device_boot_id: u64,
    capabilities_bits: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ArtifactObservation {
    calibration: Vec<ArtifactDigestDto>,
    plant: Vec<ArtifactDigestDto>,
}

/// Single-use accumulator for all production observation sources.
///
/// Each source may be retained once. `build` reports the first missing source
/// in a stable order and performs the only parse into the existing inventory
/// domain.
#[derive(Default)]
pub struct NanoObservedInventoryBuilder {
    deployment: Option<DeploymentObservation>,
    oak: Option<OakObservation>,
    stm32: Option<Stm32Observation>,
    head: Option<HeadObservation>,
    eye: Option<EyeObservation>,
    artifacts: Option<ArtifactObservation>,
}

impl NanoObservedInventoryBuilder {
    pub const fn new() -> Self {
        Self {
            deployment: None,
            oak: None,
            stm32: None,
            head: None,
            eye: None,
            artifacts: None,
        }
    }

    /// Retain the exact deployment-selected robot identity, controller
    /// endpoint, and controller identity claims used to acquire motion.
    pub fn observe_deployment(
        &mut self,
        deployment: &NavigationActuationConfigV1,
    ) -> Result<(), NanoObservedInventoryEvidenceError> {
        retain_once(
            &mut self.deployment,
            DeploymentObservation {
                robot_id: deployment.robot_id().to_owned(),
                control_endpoint_identity: format!(
                    "udp://{}",
                    deployment.command_endpoint().socket_addr()
                ),
                controller_expectation: DeploymentControllerExpectation::ProductionActuation {
                    controller_uid: *deployment.controller_uid().as_bytes(),
                    firmware_abi: deployment.firmware_abi().get(),
                    firmware_build_id: deployment.firmware_build_id().get(),
                    hardware_profile_fingerprint: *deployment
                        .actuator_config_fingerprint()
                        .as_bytes(),
                },
            },
            NanoObservedInventoryEvidenceKind::Deployment,
        )
    }

    /// Retain only qualification deployment facts which cannot be observed
    /// from a device: the logical robot selector and command route.
    ///
    /// Controller UID/build/fingerprint are deliberately absent. Those fields
    /// enter the observed snapshot only through [`Self::observe_stm32`] and
    /// are subsequently bound by exact comparison with the V2 manifest's
    /// inner V1 inventory.
    #[cfg(feature = "nano-wheels-off-qualification")]
    pub fn observe_candidate_deployment(
        &mut self,
        robot_id: &str,
        command_endpoint: SocketAddr,
    ) -> Result<(), NanoObservedInventoryEvidenceError> {
        retain_once(
            &mut self.deployment,
            DeploymentObservation {
                robot_id: robot_id.to_owned(),
                control_endpoint_identity: format!("udp://{command_endpoint}"),
                controller_expectation: DeploymentControllerExpectation::CandidateExactInventory,
            },
            NanoObservedInventoryEvidenceKind::Deployment,
        )
    }

    /// Retain identity from the already-opened exact OAK, build provenance
    /// from the native header used by `oak-sys`, and separately admitted USB
    /// transport evidence.
    pub fn observe_oak(
        &mut self,
        opened_identity: &ConnectedDeviceIdentity,
        compiled_header: &DepthAiBuildMetadata,
        transport: UsbTransportAdmissionEvidence,
    ) -> Result<(), NanoObservedInventoryEvidenceError> {
        ensure_empty(&self.oak, NanoObservedInventoryEvidenceKind::Oak)?;
        let super_speed = admit_oak_super_speed(
            transport.requested_maximum(),
            transport.required_minimum(),
            transport.observed(),
        )?;
        self.oak = Some(OakObservation {
            mxid: opened_identity.mxid().to_owned(),
            compiled_depthai_header_sdk_version: compiled_header.sdk_version().to_owned(),
            compiled_depthai_header_sdk_commit: compiled_header.sdk_commit().to_owned(),
            compiled_depthai_header_embedded_device_artifact_version: compiled_header
                .embedded_device_artifact_version()
                .to_owned(),
            compiled_depthai_header_embedded_bootloader_artifact_version: compiled_header
                .embedded_bootloader_artifact_version()
                .to_owned(),
            super_speed,
        });
        Ok(())
    }

    /// Retain controller identity from one verified acquisition and the exact
    /// persistent serial route configured for the sole robot-server owner.
    ///
    /// The serial pathname is configuration evidence, not USB descriptor
    /// readback. The controller identity itself is always taken from the
    /// acquisition, never from the deployment document.
    pub fn observe_stm32(
        &mut self,
        configured_serial_by_id_path: &Path,
        acquisition: VerifiedControllerAcquisition,
    ) -> Result<(), NanoObservedInventoryEvidenceError> {
        ensure_empty(&self.stm32, NanoObservedInventoryEvidenceKind::Stm32)?;
        let serial_by_id_path = configured_serial_by_id_path
            .to_str()
            .ok_or(NanoObservedInventoryEvidenceError::Stm32SerialPathNotUtf8)?;
        self.stm32 = Some(Stm32Observation {
            serial_by_id_path: serial_by_id_path.to_owned(),
            controller_uid: *acquisition.controller_uid().as_bytes(),
            controller_boot_id: acquisition.boot_id().get(),
            firmware_abi: acquisition.firmware_abi(),
            firmware_build_id: acquisition.firmware_build_id(),
            hardware_profile_fingerprint: *acquisition.actuator_config_fingerprint().as_bytes(),
            capabilities_bits: acquisition.capabilities().bits(),
        });
        Ok(())
    }

    /// Retain the successful fixed read-only head probe. No expected servo ID
    /// is copied: IDs come from the parsed telemetry responses themselves.
    pub fn observe_head(
        &mut self,
        probe: &HeadProbeReport,
    ) -> Result<(), NanoObservedInventoryEvidenceError> {
        ensure_empty(&self.head, NanoObservedInventoryEvidenceKind::Head)?;
        validate_head_serial(probe.serial())?;
        let mut responding_servo_ids = Vec::with_capacity(probe.servos().len());
        for (index, report) in probe.servos().iter().copied().enumerate() {
            let expected_joint = HeadJoint::ALL[index];
            if report.joint() != expected_joint {
                return Err(
                    NanoObservedInventoryEvidenceError::HeadProbeJointOrderMismatch {
                        index,
                        expected: expected_joint,
                        actual: report.joint(),
                    },
                );
            }
            let telemetry_id = report.telemetry().id();
            if telemetry_id != report.joint().servo_id() {
                return Err(
                    NanoObservedInventoryEvidenceError::HeadProbeServoIdMismatch {
                        joint: report.joint(),
                        expected: report.joint().servo_id().get(),
                        actual: telemetry_id.get(),
                    },
                );
            }
            responding_servo_ids.push(telemetry_id.get());
        }
        self.head = Some(HeadObservation {
            adapter_serial_by_id_path: probe.serial().device.path().to_owned(),
            baud_rate_bps: probe.serial().baud_rate_bps_readback,
            dtr_asserted: ADAPTER_DTR_ASSERTED,
            rts_asserted: ADAPTER_RTS_ASSERTED,
            responding_servo_ids,
        });
        Ok(())
    }

    /// Retain one nonce-bound, identity-only KEP2 probe. This does not start
    /// the eye actor or acquire expression control.
    pub fn observe_eye(
        &mut self,
        serial: &EyeSerialConfigurationEvidence,
        observation: EyeIdentityObservation,
    ) -> Result<(), NanoObservedInventoryEvidenceError> {
        ensure_empty(&self.eye, NanoObservedInventoryEvidenceKind::Eye)?;
        validate_eye_serial(serial)?;
        let report = observation.report();
        if report.nonce != observation.challenge() {
            return Err(NanoObservedInventoryEvidenceError::EyeChallengeMismatch);
        }
        self.eye = Some(EyeObservation {
            serial_by_id_path: serial.device().path().to_owned(),
            kep_protocol_version: kiko_eye_protocol::PROTOCOL_VERSION,
            device_uid: *report.device_uid.as_bytes(),
            firmware_build_id: *report.firmware_build_id.as_bytes(),
            device_boot_id: report.boot_id.get(),
            capabilities_bits: report.capabilities.bits(),
        });
        Ok(())
    }

    /// Retain observed digests from the files actually opened and hashed.
    ///
    /// Contrary content remains contrary: `observed_sha256`, not the expected
    /// manifest digest, is copied into the observed inventory. Exact inventory
    /// comparison is responsible for rejecting a mismatch.
    pub fn observe_artifacts(
        &mut self,
        hashes: &ManifestArtifactHashes,
    ) -> Result<(), NanoObservedInventoryEvidenceError> {
        ensure_empty(
            &self.artifacts,
            NanoObservedInventoryEvidenceKind::Artifacts,
        )?;
        let mut calibration = Vec::new();
        let mut plant = Vec::new();
        calibration.reserve(
            hashes
                .iter()
                .filter(|artifact| artifact.kind() == ArtifactKind::Calibration)
                .count(),
        );
        plant.reserve(
            hashes
                .iter()
                .filter(|artifact| artifact.kind() == ArtifactKind::Plant)
                .count(),
        );
        for artifact in hashes.iter() {
            match artifact.kind() {
                ArtifactKind::Calibration => {
                    calibration.push(artifact.to_observed_digest_dto());
                }
                ArtifactKind::Plant => plant.push(artifact.to_observed_digest_dto()),
            }
        }
        self.artifacts = Some(ArtifactObservation { calibration, plant });
        Ok(())
    }

    /// Cross-bind all retained observations and parse the existing V1 domain
    /// exactly once.
    pub fn build(
        self,
    ) -> Result<ProductionObservedDeviceInventoryV1, NanoObservedInventoryBuildError> {
        let deployment = required(
            self.deployment,
            NanoObservedInventoryEvidenceKind::Deployment,
        )?;
        let oak = required(self.oak, NanoObservedInventoryEvidenceKind::Oak)?;
        let stm32 = required(self.stm32, NanoObservedInventoryEvidenceKind::Stm32)?;
        let head = required(self.head, NanoObservedInventoryEvidenceKind::Head)?;
        let eye = required(self.eye, NanoObservedInventoryEvidenceKind::Eye)?;
        let artifacts = required(self.artifacts, NanoObservedInventoryEvidenceKind::Artifacts)?;

        cross_bind_controller(&deployment, &stm32)?;

        let oak_super_speed = oak.super_speed;
        let inventory = ObservedDeviceInventoryV1::parse(ObservedDeviceInventoryV1Dto {
            schema_version: OBSERVED_DEVICE_INVENTORY_V1,
            robot_id: deployment.robot_id,
            oak: Some(ObservedOakV1Dto {
                mxid: oak.mxid,
                compiled_depthai_header_sdk_version: oak.compiled_depthai_header_sdk_version,
                compiled_depthai_header_sdk_commit: oak.compiled_depthai_header_sdk_commit,
                compiled_depthai_header_embedded_device_artifact_version: oak
                    .compiled_depthai_header_embedded_device_artifact_version,
                compiled_depthai_header_embedded_bootloader_artifact_version: oak
                    .compiled_depthai_header_embedded_bootloader_artifact_version,
            }),
            stm32: Some(ObservedStm32V1Dto {
                serial_by_id_path: stm32.serial_by_id_path,
                control_endpoint_identity: deployment.control_endpoint_identity,
                controller_uid: stm32.controller_uid,
                controller_boot_id: stm32.controller_boot_id,
                firmware_abi: stm32.firmware_abi,
                firmware_build_id: stm32.firmware_build_id,
                hardware_profile_fingerprint: stm32.hardware_profile_fingerprint,
                capabilities_bits: stm32.capabilities_bits,
            }),
            head: Some(ObservedHeadV1Dto {
                adapter_serial_by_id_path: head.adapter_serial_by_id_path,
                baud_rate_bps: head.baud_rate_bps,
                dtr_asserted: head.dtr_asserted,
                rts_asserted: head.rts_asserted,
                responding_servo_ids: head.responding_servo_ids,
            }),
            eye: Some(ObservedEyeV1Dto {
                serial_by_id_path: eye.serial_by_id_path,
                kep_protocol_version: eye.kep_protocol_version,
                device_uid: eye.device_uid,
                firmware_build_id: eye.firmware_build_id,
                device_boot_id: eye.device_boot_id,
                capabilities_bits: eye.capabilities_bits,
            }),
            calibration_artifacts: artifacts.calibration,
            plant_artifacts: artifacts.plant,
        })
        .map_err(NanoObservedInventoryBuildError::Inventory)?;

        Ok(ProductionObservedDeviceInventoryV1 {
            inventory,
            oak_super_speed,
        })
    }
}

fn ensure_empty<T>(
    slot: &Option<T>,
    kind: NanoObservedInventoryEvidenceKind,
) -> Result<(), NanoObservedInventoryEvidenceError> {
    if slot.is_some() {
        Err(NanoObservedInventoryEvidenceError::DuplicateEvidence { kind })
    } else {
        Ok(())
    }
}

fn retain_once<T>(
    slot: &mut Option<T>,
    value: T,
    kind: NanoObservedInventoryEvidenceKind,
) -> Result<(), NanoObservedInventoryEvidenceError> {
    ensure_empty(slot, kind)?;
    *slot = Some(value);
    Ok(())
}

fn required<T>(
    value: Option<T>,
    kind: NanoObservedInventoryEvidenceKind,
) -> Result<T, NanoObservedInventoryBuildError> {
    value.ok_or(NanoObservedInventoryBuildError::MissingEvidence { kind })
}

fn admit_oak_super_speed(
    requested_maximum: UsbTransportSpeed,
    required_minimum: UsbTransportSpeed,
    observed: UsbTransportSpeed,
) -> Result<AdmittedOakSuperSpeedEvidence, NanoObservedInventoryEvidenceError> {
    if required_minimum < UsbTransportSpeed::Super {
        return Err(
            NanoObservedInventoryEvidenceError::OakDidNotRequireSuperSpeed { required_minimum },
        );
    }
    if observed < UsbTransportSpeed::Super {
        return Err(NanoObservedInventoryEvidenceError::OakDidNotObserveSuperSpeed { observed });
    }
    Ok(AdmittedOakSuperSpeedEvidence {
        requested_maximum,
        required_minimum,
        observed,
    })
}

fn validate_head_serial(
    serial: &HeadSerialConfigurationEvidence,
) -> Result<(), NanoObservedInventoryEvidenceError> {
    if !serial.exclusive_owner_claimed {
        return Err(NanoObservedInventoryEvidenceError::HeadSerialNotExclusive);
    }
    if serial.baud_rate_bps_readback != BUS_BAUD_RATE_BPS {
        return Err(NanoObservedInventoryEvidenceError::HeadSerialBaudMismatch {
            expected: BUS_BAUD_RATE_BPS,
            actual: serial.baud_rate_bps_readback,
        });
    }
    if !serial.data_bits_8_readback
        || !serial.parity_none_readback
        || !serial.stop_bits_1_readback
        || !serial.flow_control_none_readback
    {
        return Err(NanoObservedInventoryEvidenceError::HeadSerialFramingMismatch);
    }
    if !serial.dtr_false_setter_accepted || !serial.rts_true_setter_accepted {
        return Err(NanoObservedInventoryEvidenceError::HeadSerialModemControlNotApplied);
    }
    Ok(())
}

fn validate_eye_serial(
    serial: &EyeSerialConfigurationEvidence,
) -> Result<(), NanoObservedInventoryEvidenceError> {
    if !serial.exclusive_owner_claimed() {
        return Err(NanoObservedInventoryEvidenceError::EyeSerialNotExclusive);
    }
    if serial.baud_rate_bps_readback() != KEP2_SERIAL_BAUD_RATE_BPS {
        return Err(NanoObservedInventoryEvidenceError::EyeSerialBaudMismatch {
            expected: KEP2_SERIAL_BAUD_RATE_BPS,
            actual: serial.baud_rate_bps_readback(),
        });
    }
    if !serial.data_bits_8_readback()
        || !serial.parity_none_readback()
        || !serial.stop_bits_1_readback()
        || !serial.flow_control_none_readback()
    {
        return Err(NanoObservedInventoryEvidenceError::EyeSerialFramingMismatch);
    }
    Ok(())
}

fn cross_bind_controller(
    deployment: &DeploymentObservation,
    observed: &Stm32Observation,
) -> Result<(), NanoObservedInventoryBuildError> {
    let (controller_uid, firmware_abi, firmware_build_id, hardware_profile_fingerprint) =
        match deployment.controller_expectation {
            DeploymentControllerExpectation::ProductionActuation {
                controller_uid,
                firmware_abi,
                firmware_build_id,
                hardware_profile_fingerprint,
            } => (
                controller_uid,
                firmware_abi,
                firmware_build_id,
                hardware_profile_fingerprint,
            ),
            #[cfg(feature = "nano-wheels-off-qualification")]
            DeploymentControllerExpectation::CandidateExactInventory => return Ok(()),
        };
    if controller_uid != observed.controller_uid {
        return Err(NanoObservedInventoryBuildError::ControllerUidMismatch {
            deployment: controller_uid,
            observed: observed.controller_uid,
        });
    }
    if firmware_abi != observed.firmware_abi {
        return Err(
            NanoObservedInventoryBuildError::ControllerFirmwareAbiMismatch {
                deployment: firmware_abi,
                observed: observed.firmware_abi,
            },
        );
    }
    if firmware_build_id != observed.firmware_build_id {
        return Err(
            NanoObservedInventoryBuildError::ControllerFirmwareBuildMismatch {
                deployment: firmware_build_id,
                observed: observed.firmware_build_id,
            },
        );
    }
    if hardware_profile_fingerprint != observed.hardware_profile_fingerprint {
        return Err(
            NanoObservedInventoryBuildError::ControllerHardwareProfileMismatch {
                deployment: hardware_profile_fingerprint,
                observed: observed.hardware_profile_fingerprint,
            },
        );
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoObservedInventoryEvidenceError {
    DuplicateEvidence {
        kind: NanoObservedInventoryEvidenceKind,
    },
    Stm32SerialPathNotUtf8,
    OakDidNotRequireSuperSpeed {
        required_minimum: UsbTransportSpeed,
    },
    OakDidNotObserveSuperSpeed {
        observed: UsbTransportSpeed,
    },
    HeadSerialNotExclusive,
    HeadSerialBaudMismatch {
        expected: u32,
        actual: u32,
    },
    HeadSerialFramingMismatch,
    HeadSerialModemControlNotApplied,
    HeadProbeJointOrderMismatch {
        index: usize,
        expected: HeadJoint,
        actual: HeadJoint,
    },
    HeadProbeServoIdMismatch {
        joint: HeadJoint,
        expected: u8,
        actual: u8,
    },
    EyeSerialNotExclusive,
    EyeSerialBaudMismatch {
        expected: u32,
        actual: u32,
    },
    EyeSerialFramingMismatch,
    EyeChallengeMismatch,
}

impl fmt::Display for NanoObservedInventoryEvidenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid Nano observed-inventory evidence: {self:?}"
        )
    }
}

impl std::error::Error for NanoObservedInventoryEvidenceError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoObservedInventoryBuildError {
    MissingEvidence {
        kind: NanoObservedInventoryEvidenceKind,
    },
    ControllerUidMismatch {
        deployment: [u8; 12],
        observed: [u8; 12],
    },
    ControllerFirmwareAbiMismatch {
        deployment: u16,
        observed: u16,
    },
    ControllerFirmwareBuildMismatch {
        deployment: u32,
        observed: u32,
    },
    ControllerHardwareProfileMismatch {
        deployment: [u8; 16],
        observed: [u8; 16],
    },
    Inventory(InventoryParseError),
}

impl fmt::Display for NanoObservedInventoryBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "could not build Nano observed inventory: {self:?}"
        )
    }
}

impl std::error::Error for NanoObservedInventoryBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Inventory(source) => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiko_device_inventory::DeviceRole;

    const ROBOT_ID: &str = "kiko-production-01";
    const OAK_MXID: &str = "19443010F1B43A2E00";
    const STM32_PATH: &str = "/dev/serial/by-id/usb-kiko-stm32-if00";
    const HEAD_PATH: &str = "/dev/serial/by-id/usb-kiko-head-if00";
    const EYE_PATH: &str = "/dev/serial/by-id/usb-kiko-eye-if00";
    const CONTROLLER_UID: [u8; 12] = [1; 12];
    const HARDWARE_PROFILE: [u8; 16] = [2; 16];

    fn deployment() -> DeploymentObservation {
        DeploymentObservation {
            robot_id: ROBOT_ID.into(),
            control_endpoint_identity: "udp://127.0.0.1:8080".into(),
            controller_expectation: DeploymentControllerExpectation::ProductionActuation {
                controller_uid: CONTROLLER_UID,
                firmware_abi: u16::from(robot_protocol::v2::VERSION),
                firmware_build_id: 7,
                hardware_profile_fingerprint: HARDWARE_PROFILE,
            },
        }
    }

    fn oak() -> OakObservation {
        OakObservation {
            mxid: OAK_MXID.into(),
            compiled_depthai_header_sdk_version: "3.4.0".into(),
            compiled_depthai_header_sdk_commit: "ba7a920".into(),
            compiled_depthai_header_embedded_device_artifact_version: "device-2026".into(),
            compiled_depthai_header_embedded_bootloader_artifact_version: "boot-2026".into(),
            super_speed: AdmittedOakSuperSpeedEvidence {
                requested_maximum: UsbTransportSpeed::SuperPlus,
                required_minimum: UsbTransportSpeed::Super,
                observed: UsbTransportSpeed::SuperPlus,
            },
        }
    }

    fn stm32() -> Stm32Observation {
        Stm32Observation {
            serial_by_id_path: STM32_PATH.into(),
            controller_uid: CONTROLLER_UID,
            controller_boot_id: 11,
            firmware_abi: u16::from(robot_protocol::v2::VERSION),
            firmware_build_id: 7,
            hardware_profile_fingerprint: HARDWARE_PROFILE,
            capabilities_bits: robot_protocol::v2::ControllerCapabilities::REQUIRED_BITS,
        }
    }

    fn head() -> HeadObservation {
        HeadObservation {
            adapter_serial_by_id_path: HEAD_PATH.into(),
            baud_rate_bps: BUS_BAUD_RATE_BPS,
            dtr_asserted: ADAPTER_DTR_ASSERTED,
            rts_asserted: ADAPTER_RTS_ASSERTED,
            responding_servo_ids: vec![1, 2, 3, 4],
        }
    }

    fn eye() -> EyeObservation {
        EyeObservation {
            serial_by_id_path: EYE_PATH.into(),
            kep_protocol_version: kiko_eye_protocol::PROTOCOL_VERSION,
            device_uid: [3; 16],
            firmware_build_id: [4; 32],
            device_boot_id: 13,
            capabilities_bits: kiko_eye_protocol::Capabilities::KNOWN_BITS,
        }
    }

    fn artifacts() -> ArtifactObservation {
        ArtifactObservation {
            calibration: vec![ArtifactDigestDto {
                artifact_id: "camera-main".into(),
                sha256: [5; 32],
            }],
            plant: vec![ArtifactDigestDto {
                artifact_id: "drive-main".into(),
                sha256: [6; 32],
            }],
        }
    }

    fn complete_builder() -> NanoObservedInventoryBuilder {
        NanoObservedInventoryBuilder {
            deployment: Some(deployment()),
            oak: Some(oak()),
            stm32: Some(stm32()),
            head: Some(head()),
            eye: Some(eye()),
            artifacts: Some(artifacts()),
        }
    }

    #[test]
    fn exact_extraction_uses_observed_fields_and_retains_super_speed() {
        let production = complete_builder().build().expect("production inventory");
        let inventory = production.inventory();

        assert_eq!(inventory.robot_id().as_str(), ROBOT_ID);
        let observed_oak = inventory.oak().expect("OAK");
        assert_eq!(observed_oak.mxid().as_str(), OAK_MXID);
        assert_eq!(
            observed_oak.compiled_depthai_header_sdk_version().as_str(),
            "3.4.0"
        );
        assert_eq!(
            observed_oak
                .compiled_depthai_header_embedded_device_artifact_version()
                .as_str(),
            "device-2026"
        );

        let observed_stm32 = inventory.stm32().expect("STM32");
        assert_eq!(
            observed_stm32.static_identity().serial_path().as_str(),
            STM32_PATH
        );
        assert_eq!(
            observed_stm32.static_identity().control_endpoint().as_str(),
            "udp://127.0.0.1:8080"
        );
        assert_eq!(observed_stm32.boot_id().get(), 11);

        let observed_head = inventory.head().expect("head");
        assert_eq!(observed_head.serial_path().as_str(), HEAD_PATH);
        assert_eq!(
            observed_head
                .servo_ids()
                .iter()
                .map(|id| id.get())
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );

        let observed_eye = inventory.eye().expect("eye");
        assert_eq!(
            observed_eye.static_identity().serial_path().as_str(),
            EYE_PATH
        );
        assert_eq!(observed_eye.boot_id().get(), 13);

        let digests = inventory
            .artifacts()
            .iter()
            .map(|artifact| {
                (
                    artifact.kind(),
                    artifact.artifact_id().as_str(),
                    *artifact.sha256().as_bytes(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            digests,
            vec![
                (ArtifactKind::Calibration, "camera-main", [5; 32]),
                (ArtifactKind::Plant, "drive-main", [6; 32]),
            ]
        );
        assert_eq!(
            production.oak_super_speed().required_minimum(),
            UsbTransportSpeed::Super
        );
        assert_eq!(
            production.oak_super_speed().observed(),
            UsbTransportSpeed::SuperPlus
        );
    }

    #[test]
    fn each_missing_live_source_is_reported_before_dto_parsing() {
        let cases = [
            NanoObservedInventoryEvidenceKind::Deployment,
            NanoObservedInventoryEvidenceKind::Oak,
            NanoObservedInventoryEvidenceKind::Stm32,
            NanoObservedInventoryEvidenceKind::Head,
            NanoObservedInventoryEvidenceKind::Eye,
            NanoObservedInventoryEvidenceKind::Artifacts,
        ];
        for missing in cases {
            let mut builder = complete_builder();
            match missing {
                NanoObservedInventoryEvidenceKind::Deployment => builder.deployment = None,
                NanoObservedInventoryEvidenceKind::Oak => builder.oak = None,
                NanoObservedInventoryEvidenceKind::Stm32 => builder.stm32 = None,
                NanoObservedInventoryEvidenceKind::Head => builder.head = None,
                NanoObservedInventoryEvidenceKind::Eye => builder.eye = None,
                NanoObservedInventoryEvidenceKind::Artifacts => builder.artifacts = None,
            }
            assert_eq!(
                builder.build().expect_err("missing evidence"),
                NanoObservedInventoryBuildError::MissingEvidence { kind: missing }
            );
        }
    }

    #[test]
    fn duplicate_evidence_cannot_replace_the_first_observation() {
        let mut slot = Some(1_u8);
        assert_eq!(
            retain_once(&mut slot, 2, NanoObservedInventoryEvidenceKind::Artifacts),
            Err(NanoObservedInventoryEvidenceError::DuplicateEvidence {
                kind: NanoObservedInventoryEvidenceKind::Artifacts
            })
        );
        assert_eq!(slot, Some(1));
    }

    #[test]
    fn controller_acquisition_must_match_the_deployment_authority() {
        let mut builder = complete_builder();
        builder.stm32.as_mut().expect("STM32").controller_uid = [9; 12];
        assert!(matches!(
            builder.build(),
            Err(NanoObservedInventoryBuildError::ControllerUidMismatch { .. })
        ));

        let mut builder = complete_builder();
        builder.stm32.as_mut().expect("STM32").firmware_abi += 1;
        assert!(matches!(
            builder.build(),
            Err(NanoObservedInventoryBuildError::ControllerFirmwareAbiMismatch { .. })
        ));

        let mut builder = complete_builder();
        builder.stm32.as_mut().expect("STM32").firmware_build_id += 1;
        assert!(matches!(
            builder.build(),
            Err(NanoObservedInventoryBuildError::ControllerFirmwareBuildMismatch { .. })
        ));

        let mut builder = complete_builder();
        builder
            .stm32
            .as_mut()
            .expect("STM32")
            .hardware_profile_fingerprint = [8; 16];
        assert!(matches!(
            builder.build(),
            Err(NanoObservedInventoryBuildError::ControllerHardwareProfileMismatch { .. })
        ));
    }

    #[cfg(feature = "nano-wheels-off-qualification")]
    #[test]
    fn candidate_observation_does_not_copy_controller_expectations() {
        let mut builder = complete_builder();
        builder
            .deployment
            .as_mut()
            .expect("deployment")
            .controller_expectation = DeploymentControllerExpectation::CandidateExactInventory;
        let stm32 = builder.stm32.as_mut().expect("STM32");
        stm32.controller_uid = [9; 12];
        stm32.firmware_build_id = 99;
        stm32.hardware_profile_fingerprint = [8; 16];

        let observed = builder
            .build()
            .expect("candidate identity remains live evidence");
        let observed_stm32 = observed.inventory().stm32().expect("STM32");
        assert_eq!(
            observed_stm32.static_identity().controller_uid().as_bytes(),
            &[9; 12]
        );
        assert_eq!(observed_stm32.static_identity().firmware_build_id(), 99);
        assert_eq!(
            observed_stm32
                .static_identity()
                .hardware_profile()
                .as_bytes(),
            &[8; 16]
        );
    }

    #[test]
    fn super_speed_must_be_both_required_and_observed() {
        assert_eq!(
            admit_oak_super_speed(
                UsbTransportSpeed::Super,
                UsbTransportSpeed::High,
                UsbTransportSpeed::Super,
            ),
            Err(
                NanoObservedInventoryEvidenceError::OakDidNotRequireSuperSpeed {
                    required_minimum: UsbTransportSpeed::High
                }
            )
        );
        assert_eq!(
            admit_oak_super_speed(
                UsbTransportSpeed::Super,
                UsbTransportSpeed::Super,
                UsbTransportSpeed::High,
            ),
            Err(
                NanoObservedInventoryEvidenceError::OakDidNotObserveSuperSpeed {
                    observed: UsbTransportSpeed::High
                }
            )
        );
    }

    #[test]
    fn duplicate_physical_paths_remain_a_typed_inventory_failure() {
        let mut builder = complete_builder();
        builder.eye.as_mut().expect("eye").serial_by_id_path = HEAD_PATH.into();
        assert_eq!(
            builder.build().expect_err("duplicate serial path"),
            NanoObservedInventoryBuildError::Inventory(
                InventoryParseError::DuplicatePhysicalPath {
                    first: DeviceRole::Head,
                    second: DeviceRole::Eye,
                }
            )
        );
    }

    #[test]
    fn observed_artifact_digest_is_never_substituted_with_an_expectation() {
        let mut builder = complete_builder();
        builder.artifacts.as_mut().expect("artifacts").calibration[0].sha256 = [0xA5; 32];
        let inventory = builder.build().expect("inventory");
        let calibration = inventory
            .inventory()
            .artifacts()
            .iter()
            .find(|artifact| artifact.kind() == ArtifactKind::Calibration)
            .expect("calibration");
        assert_eq!(calibration.sha256().as_bytes(), &[0xA5; 32]);
    }
}
