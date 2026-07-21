use core::fmt;

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
/// There are 23 scalar/device fields and at most two mismatch entries for each
/// artifact slot (one missing expected artifact and one unexpected observed
/// artifact).
pub const MAX_INVENTORY_MISMATCHES: usize = 23 + 2 * MAX_ARTIFACTS;

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
    OakCompiledDepthAiHeaderSdkVersion {
        expected: &'inventory BuildProvenance,
        observed: &'inventory BuildProvenance,
    },
    OakCompiledDepthAiHeaderSdkCommit {
        expected: &'inventory BuildProvenance,
        observed: &'inventory BuildProvenance,
    },
    OakCompiledDepthAiHeaderEmbeddedDeviceArtifactVersion {
        expected: &'inventory BuildProvenance,
        observed: &'inventory BuildProvenance,
    },
    OakCompiledDepthAiHeaderEmbeddedBootloaderArtifactVersion {
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

#[derive(Clone, Debug)]
pub struct InventoryComparison<'inventory> {
    expected: &'inventory DeviceInventoryManifestV1,
    observed: &'inventory ObservedDeviceInventoryV1,
    mismatches: [Option<InventoryMismatch<'inventory>>; MAX_INVENTORY_MISMATCHES],
    len: u8,
}

impl<'inventory> PartialEq for InventoryComparison<'inventory> {
    fn eq(&self, other: &Self) -> bool {
        // Expected/observed references exist only to mint owned admission.
        // Comparison semantics are exactly the reported mismatch sequence;
        // fields intentionally excluded from comparison (such as boot IDs)
        // must not leak into equality through those references.
        self.mismatches[..self.len()] == other.mismatches[..other.len()]
    }
}

impl<'inventory> Eq for InventoryComparison<'inventory> {}

/// Owned capability proving that two immutable parsed snapshots compared
/// exactly when this value was created.
///
/// Fields are private, so external callers can obtain this type only through
/// [`admit_exact_inventory`] or
/// [`InventoryComparison::into_exact_admission`], both of which require an
/// exact comparison. This is snapshot admission evidence, not proof of
/// continuing connectivity, authenticity, or liveness.
#[derive(Debug, PartialEq, Eq)]
pub struct ExactInventoryAdmission {
    expected: DeviceInventoryManifestV1,
    observed: ObservedDeviceInventoryV1,
}

impl ExactInventoryAdmission {
    pub fn expected(&self) -> &DeviceInventoryManifestV1 {
        &self.expected
    }

    pub fn observed(&self) -> &ObservedDeviceInventoryV1 {
        &self.observed
    }
}

/// Owned, bounded, lossless evidence for a failed exact-inventory admission.
///
/// Both parsed snapshots are retained, rather than copying or truncating
/// mismatch text. [`Self::comparison`] deterministically reconstructs every
/// typed mismatch while borrowing this report.
#[derive(Debug, PartialEq, Eq)]
pub struct InventoryMismatchReport(Box<InventoryMismatchEvidence>);

#[derive(Debug, PartialEq, Eq)]
struct InventoryMismatchEvidence {
    expected: DeviceInventoryManifestV1,
    observed: ObservedDeviceInventoryV1,
    mismatch_count: u8,
}

impl InventoryMismatchReport {
    pub fn expected(&self) -> &DeviceInventoryManifestV1 {
        &self.0.expected
    }

    pub fn observed(&self) -> &ObservedDeviceInventoryV1 {
        &self.0.observed
    }

    pub fn comparison(&self) -> InventoryComparison<'_> {
        let comparison = InventoryComparison::compare(&self.0.expected, &self.0.observed);
        debug_assert_eq!(comparison.len, self.0.mismatch_count);
        comparison
    }

    pub fn len(&self) -> usize {
        usize::from(self.0.mismatch_count)
    }

    pub fn is_empty(&self) -> bool {
        self.0.mismatch_count == 0
    }
}

impl fmt::Display for InventoryMismatchReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "exact Kiko inventory admission failed with {} mismatch(es)",
            self.len()
        )
    }
}

impl std::error::Error for InventoryMismatchReport {}

/// Compare two owned parsed snapshots and move them into the exact result.
///
/// This is the preferred production entrypoint when the caller owns both
/// values: success performs no full-snapshot clone, while failure allocates
/// one bounded report payload containing the complete original snapshots.
pub fn admit_exact_inventory(
    expected: DeviceInventoryManifestV1,
    observed: ObservedDeviceInventoryV1,
) -> Result<ExactInventoryAdmission, InventoryMismatchReport> {
    let mismatch_count = InventoryComparison::compare(&expected, &observed).len;
    if mismatch_count == 0 {
        Ok(ExactInventoryAdmission { expected, observed })
    } else {
        Err(InventoryMismatchReport(Box::new(
            InventoryMismatchEvidence {
                expected,
                observed,
                mismatch_count,
            },
        )))
    }
}

impl<'inventory> InventoryComparison<'inventory> {
    pub fn compare(
        expected: &'inventory DeviceInventoryManifestV1,
        observed: &'inventory ObservedDeviceInventoryV1,
    ) -> Self {
        let mut output = Self {
            expected,
            observed,
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

    /// Convert an exact borrowed comparison into owned admission evidence.
    ///
    /// A non-exact comparison returns an owned report containing both complete
    /// bounded snapshots, so no mismatch values or identities are lost when
    /// the original inputs go out of scope.
    pub fn into_exact_admission(self) -> Result<ExactInventoryAdmission, InventoryMismatchReport> {
        if self.is_exact_match() {
            Ok(ExactInventoryAdmission {
                expected: self.expected.clone(),
                observed: self.observed.clone(),
            })
        } else {
            Err(InventoryMismatchReport(Box::new(
                InventoryMismatchEvidence {
                    expected: self.expected.clone(),
                    observed: self.observed.clone(),
                    mismatch_count: self.len,
                },
            )))
        }
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
    if expected.compiled_depthai_header_sdk_version()
        != observed.compiled_depthai_header_sdk_version()
    {
        output.push(InventoryMismatch::OakCompiledDepthAiHeaderSdkVersion {
            expected: expected.compiled_depthai_header_sdk_version(),
            observed: observed.compiled_depthai_header_sdk_version(),
        });
    }
    if expected.compiled_depthai_header_sdk_commit()
        != observed.compiled_depthai_header_sdk_commit()
    {
        output.push(InventoryMismatch::OakCompiledDepthAiHeaderSdkCommit {
            expected: expected.compiled_depthai_header_sdk_commit(),
            observed: observed.compiled_depthai_header_sdk_commit(),
        });
    }
    if expected.compiled_depthai_header_embedded_device_artifact_version()
        != observed.compiled_depthai_header_embedded_device_artifact_version()
    {
        output.push(
            InventoryMismatch::OakCompiledDepthAiHeaderEmbeddedDeviceArtifactVersion {
                expected: expected.compiled_depthai_header_embedded_device_artifact_version(),
                observed: observed.compiled_depthai_header_embedded_device_artifact_version(),
            },
        );
    }
    if expected.compiled_depthai_header_embedded_bootloader_artifact_version()
        != observed.compiled_depthai_header_embedded_bootloader_artifact_version()
    {
        output.push(
            InventoryMismatch::OakCompiledDepthAiHeaderEmbeddedBootloaderArtifactVersion {
                expected: expected.compiled_depthai_header_embedded_bootloader_artifact_version(),
                observed: observed.compiled_depthai_header_embedded_bootloader_artifact_version(),
            },
        );
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
