use kiko_eye_protocol::{
    Capabilities as EyeCapabilities, DeviceUid as EyeDeviceUid,
    FirmwareBuildId as EyeFirmwareBuildId,
};
use kiko_head_protocol::ServoId;
use robot_protocol::v2::{ActuatorConfigFingerprint, ControllerCapabilities, ControllerUid};

use crate::{
    ArtifactId, ArtifactKind, BuildProvenance, ControlEndpointIdentity, DeviceInventoryManifestV1,
    EyeStaticIdentity, HeadExpectedIdentity, MAX_ARTIFACTS, OakIdentity, OakMxid,
    ObservedDeviceInventoryV1, ObservedHead, ObservedServoIds, PersistentSerialPath, RobotId,
    Sha256Id, Stm32StaticIdentity,
};

/// Tight upper bound for two successfully parsed inventories.
///
/// There are 22 scalar/device fields and at most two mismatch entries for each
/// artifact slot (one missing expected artifact and one unexpected observed
/// artifact).
pub const MAX_INVENTORY_MISMATCHES: usize = 22 + 2 * MAX_ARTIFACTS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InventoryMismatch<'inventory> {
    RobotId {
        expected: &'inventory RobotId,
        observed: &'inventory RobotId,
    },
    MissingOak,
    OakMxid {
        expected: &'inventory OakMxid,
        observed: &'inventory OakMxid,
    },
    OakRuntimeProvenance {
        expected: &'inventory BuildProvenance,
        observed: &'inventory BuildProvenance,
    },
    OakSdkBuildProvenance {
        expected: &'inventory BuildProvenance,
        observed: &'inventory BuildProvenance,
    },
    OakAdapterBuildProvenance {
        expected: &'inventory BuildProvenance,
        observed: &'inventory BuildProvenance,
    },
    MissingStm32,
    Stm32SerialPath {
        expected: &'inventory PersistentSerialPath,
        observed: &'inventory PersistentSerialPath,
    },
    Stm32ControlEndpoint {
        expected: &'inventory ControlEndpointIdentity,
        observed: &'inventory ControlEndpointIdentity,
    },
    Stm32ControllerUid {
        expected: &'inventory ControllerUid,
        observed: &'inventory ControllerUid,
    },
    Stm32FirmwareAbi {
        expected: u16,
        observed: u16,
    },
    Stm32FirmwareBuildId {
        expected: u32,
        observed: u32,
    },
    Stm32HardwareProfile {
        expected: &'inventory ActuatorConfigFingerprint,
        observed: &'inventory ActuatorConfigFingerprint,
    },
    Stm32Capabilities {
        expected: ControllerCapabilities,
        observed: ControllerCapabilities,
    },
    MissingHead,
    UnexpectedHead,
    HeadSerialPath {
        expected: &'inventory PersistentSerialPath,
        observed: &'inventory PersistentSerialPath,
    },
    HeadBaudRate {
        expected_bps: u32,
        observed_bps: u32,
    },
    HeadDtrState {
        expected_asserted: bool,
        observed_asserted: bool,
    },
    HeadRtsState {
        expected_asserted: bool,
        observed_asserted: bool,
    },
    HeadServoIds {
        expected: &'inventory [ServoId; 4],
        observed: &'inventory ObservedServoIds,
    },
    MissingEye,
    UnexpectedEye,
    EyeSerialPath {
        expected: &'inventory PersistentSerialPath,
        observed: &'inventory PersistentSerialPath,
    },
    EyeProtocolVersion {
        expected: u8,
        observed: u8,
    },
    EyeDeviceUid {
        expected: &'inventory EyeDeviceUid,
        observed: &'inventory EyeDeviceUid,
    },
    EyeFirmwareBuildId {
        expected: &'inventory EyeFirmwareBuildId,
        observed: &'inventory EyeFirmwareBuildId,
    },
    EyeCapabilities {
        expected: EyeCapabilities,
        observed: EyeCapabilities,
    },
    MissingArtifact {
        kind: ArtifactKind,
        artifact_id: &'inventory ArtifactId,
        expected_sha256: &'inventory Sha256Id,
    },
    ArtifactDigest {
        kind: ArtifactKind,
        artifact_id: &'inventory ArtifactId,
        expected_sha256: &'inventory Sha256Id,
        observed_sha256: &'inventory Sha256Id,
    },
    UnexpectedArtifact {
        kind: ArtifactKind,
        artifact_id: &'inventory ArtifactId,
        observed_sha256: &'inventory Sha256Id,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InventoryComparison<'inventory> {
    mismatches: [Option<InventoryMismatch<'inventory>>; MAX_INVENTORY_MISMATCHES],
    len: u8,
}

impl<'inventory> InventoryComparison<'inventory> {
    pub fn compare(
        expected: &'inventory DeviceInventoryManifestV1,
        observed: &'inventory ObservedDeviceInventoryV1,
    ) -> Self {
        let mut output = Self {
            mismatches: core::array::from_fn(|_| None),
            len: 0,
        };
        if expected.robot_id() != observed.robot_id() {
            output.push(InventoryMismatch::RobotId {
                expected: expected.robot_id(),
                observed: observed.robot_id(),
            });
        }
        match observed.oak() {
            Some(actual) => compare_oak(&mut output, expected.oak(), actual),
            None => output.push(InventoryMismatch::MissingOak),
        }
        match observed.stm32() {
            Some(actual) => compare_stm32(&mut output, expected.stm32(), actual.static_identity()),
            None => output.push(InventoryMismatch::MissingStm32),
        }
        match (expected.head(), observed.head()) {
            (Some(expected), Some(actual)) => compare_head(&mut output, expected, actual),
            (Some(_), None) => output.push(InventoryMismatch::MissingHead),
            (None, Some(_)) => output.push(InventoryMismatch::UnexpectedHead),
            (None, None) => {}
        }
        match (expected.eye(), observed.eye()) {
            (Some(expected), Some(actual)) => {
                compare_eye(&mut output, expected, actual.static_identity())
            }
            (Some(_), None) => output.push(InventoryMismatch::MissingEye),
            (None, Some(_)) => output.push(InventoryMismatch::UnexpectedEye),
            (None, None) => {}
        }
        for artifact in expected.artifacts().iter() {
            match observed
                .artifacts()
                .find(artifact.kind(), artifact.artifact_id())
            {
                None => output.push(InventoryMismatch::MissingArtifact {
                    kind: artifact.kind(),
                    artifact_id: artifact.artifact_id(),
                    expected_sha256: artifact.sha256(),
                }),
                Some(actual) if actual.sha256() != artifact.sha256() => {
                    output.push(InventoryMismatch::ArtifactDigest {
                        kind: artifact.kind(),
                        artifact_id: artifact.artifact_id(),
                        expected_sha256: artifact.sha256(),
                        observed_sha256: actual.sha256(),
                    })
                }
                Some(_) => {}
            }
        }
        for artifact in observed.artifacts().iter() {
            if expected
                .artifacts()
                .find(artifact.kind(), artifact.artifact_id())
                .is_none()
            {
                output.push(InventoryMismatch::UnexpectedArtifact {
                    kind: artifact.kind(),
                    artifact_id: artifact.artifact_id(),
                    observed_sha256: artifact.sha256(),
                });
            }
        }
        output
    }

    pub fn is_exact_match(&self) -> bool {
        self.len == 0
    }

    pub fn len(&self) -> usize {
        usize::from(self.len)
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = &InventoryMismatch<'inventory>> + '_ {
        self.mismatches[..self.len()]
            .iter()
            .map(|mismatch| mismatch.as_ref().expect("comparison prefix is initialized"))
    }

    fn push(&mut self, mismatch: InventoryMismatch<'inventory>) {
        let index = usize::from(self.len);
        let slot = self
            .mismatches
            .get_mut(index)
            .expect("mismatch capacity is derived from all parsed inventory fields");
        *slot = Some(mismatch);
        self.len += 1;
    }
}

fn compare_oak<'inventory>(
    output: &mut InventoryComparison<'inventory>,
    expected: &'inventory OakIdentity,
    observed: &'inventory OakIdentity,
) {
    if expected.mxid() != observed.mxid() {
        output.push(InventoryMismatch::OakMxid {
            expected: expected.mxid(),
            observed: observed.mxid(),
        });
    }
    if expected.runtime_provenance() != observed.runtime_provenance() {
        output.push(InventoryMismatch::OakRuntimeProvenance {
            expected: expected.runtime_provenance(),
            observed: observed.runtime_provenance(),
        });
    }
    if expected.sdk_build_provenance() != observed.sdk_build_provenance() {
        output.push(InventoryMismatch::OakSdkBuildProvenance {
            expected: expected.sdk_build_provenance(),
            observed: observed.sdk_build_provenance(),
        });
    }
    if expected.adapter_build_provenance() != observed.adapter_build_provenance() {
        output.push(InventoryMismatch::OakAdapterBuildProvenance {
            expected: expected.adapter_build_provenance(),
            observed: observed.adapter_build_provenance(),
        });
    }
}

fn compare_stm32<'inventory>(
    output: &mut InventoryComparison<'inventory>,
    expected: &'inventory Stm32StaticIdentity,
    observed: &'inventory Stm32StaticIdentity,
) {
    if expected.serial_path() != observed.serial_path() {
        output.push(InventoryMismatch::Stm32SerialPath {
            expected: expected.serial_path(),
            observed: observed.serial_path(),
        });
    }
    if expected.control_endpoint() != observed.control_endpoint() {
        output.push(InventoryMismatch::Stm32ControlEndpoint {
            expected: expected.control_endpoint(),
            observed: observed.control_endpoint(),
        });
    }
    if expected.controller_uid() != observed.controller_uid() {
        output.push(InventoryMismatch::Stm32ControllerUid {
            expected: expected.controller_uid(),
            observed: observed.controller_uid(),
        });
    }
    if expected.firmware_abi() != observed.firmware_abi() {
        output.push(InventoryMismatch::Stm32FirmwareAbi {
            expected: expected.firmware_abi(),
            observed: observed.firmware_abi(),
        });
    }
    if expected.firmware_build_id() != observed.firmware_build_id() {
        output.push(InventoryMismatch::Stm32FirmwareBuildId {
            expected: expected.firmware_build_id(),
            observed: observed.firmware_build_id(),
        });
    }
    if expected.hardware_profile() != observed.hardware_profile() {
        output.push(InventoryMismatch::Stm32HardwareProfile {
            expected: expected.hardware_profile(),
            observed: observed.hardware_profile(),
        });
    }
    if expected.capabilities() != observed.capabilities() {
        output.push(InventoryMismatch::Stm32Capabilities {
            expected: expected.capabilities(),
            observed: observed.capabilities(),
        });
    }
}

fn compare_head<'inventory>(
    output: &mut InventoryComparison<'inventory>,
    expected: &'inventory HeadExpectedIdentity,
    observed: &'inventory ObservedHead,
) {
    if expected.serial_path() != observed.serial_path() {
        output.push(InventoryMismatch::HeadSerialPath {
            expected: expected.serial_path(),
            observed: observed.serial_path(),
        });
    }
    if expected.baud_rate_bps() != observed.baud_rate_bps() {
        output.push(InventoryMismatch::HeadBaudRate {
            expected_bps: expected.baud_rate_bps(),
            observed_bps: observed.baud_rate_bps(),
        });
    }
    if expected.dtr_asserted() != observed.dtr_asserted() {
        output.push(InventoryMismatch::HeadDtrState {
            expected_asserted: expected.dtr_asserted(),
            observed_asserted: observed.dtr_asserted(),
        });
    }
    if expected.rts_asserted() != observed.rts_asserted() {
        output.push(InventoryMismatch::HeadRtsState {
            expected_asserted: expected.rts_asserted(),
            observed_asserted: observed.rts_asserted(),
        });
    }
    let expected_ids = expected.servo_ids();
    let observed_ids = observed.servo_ids();
    let exact_ids = observed_ids.len() == expected_ids.len()
        && expected_ids
            .iter()
            .all(|expected_id| observed_ids.iter().any(|actual| actual == *expected_id));
    if !exact_ids {
        output.push(InventoryMismatch::HeadServoIds {
            expected: expected_ids,
            observed: observed_ids,
        });
    }
}

fn compare_eye<'inventory>(
    output: &mut InventoryComparison<'inventory>,
    expected: &'inventory EyeStaticIdentity,
    observed: &'inventory EyeStaticIdentity,
) {
    if expected.serial_path() != observed.serial_path() {
        output.push(InventoryMismatch::EyeSerialPath {
            expected: expected.serial_path(),
            observed: observed.serial_path(),
        });
    }
    if expected.protocol_version() != observed.protocol_version() {
        output.push(InventoryMismatch::EyeProtocolVersion {
            expected: expected.protocol_version(),
            observed: observed.protocol_version(),
        });
    }
    if expected.device_uid() != observed.device_uid() {
        output.push(InventoryMismatch::EyeDeviceUid {
            expected: expected.device_uid(),
            observed: observed.device_uid(),
        });
    }
    if expected.firmware_build_id() != observed.firmware_build_id() {
        output.push(InventoryMismatch::EyeFirmwareBuildId {
            expected: expected.firmware_build_id(),
            observed: observed.firmware_build_id(),
        });
    }
    if expected.capabilities() != observed.capabilities() {
        output.push(InventoryMismatch::EyeCapabilities {
            expected: expected.capabilities(),
            observed: observed.capabilities(),
        });
    }
}
