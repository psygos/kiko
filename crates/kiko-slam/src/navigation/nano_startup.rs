//! Proof-carrying Nano startup admission.
//!
//! This boundary combines already-parsed policy, no-follow file evidence,
//! exact observed inventory, and a verified controller acquisition exactly
//! once. A successful value can enter the supervisor only in `Disarmed`; it
//! cannot arm the robot or claim that any device remains live.

use std::fmt;
use std::path::PathBuf;

use kiko_device_inventory::{
    ArtifactId, ArtifactKind, CalibrationBundleHashError, ExactInventoryAdmission,
    LoadedExpectedManifestV1, ManifestArtifactHashes,
};
use kiko_supervisor_core::{
    EvidenceValueError, ReadinessBinding, ReadinessEpoch, Sha256Digest, SupervisorAction,
};
use robot_command_client::VerifiedControllerAcquisition;

use super::{
    AgentAuthorityError, AgentAuthoritySupervisor, ManifestBoundNanoAgentPolicyConfigV1,
    NanoAccessoryManifestBindingError, NanoAgentPolicyConfigV1, NavigationClockEpoch,
};
use crate::HostMonotonicTimestamp;

/// Startup evidence whose file locations, contents, parsed policy, exact
/// inventory, and live controller acquisition all describe the same robot.
pub struct AdmittedNanoStartup {
    policy: ManifestBoundNanoAgentPolicyConfigV1,
    exact_inventory: ExactInventoryAdmission,
    artifact_hashes: ManifestArtifactHashes,
    controller_acquisition: VerifiedControllerAcquisition,
    readiness: ReadinessBinding,
}

impl AdmittedNanoStartup {
    #[allow(clippy::too_many_arguments)]
    pub fn admit(
        policy: NanoAgentPolicyConfigV1,
        loaded_manifest: LoadedExpectedManifestV1,
        artifact_hashes: ManifestArtifactHashes,
        exact_inventory: ExactInventoryAdmission,
        controller_acquisition: VerifiedControllerAcquisition,
        readiness_epoch: ReadinessEpoch,
    ) -> Result<Self, NanoStartupAdmissionError> {
        let configured_manifest_path = policy.inventory().manifest_path().as_path();
        let Some(loaded_manifest_path) = loaded_manifest.source_path() else {
            return Err(NanoStartupAdmissionError::ManifestWasNotLoadedFromFile {
                configured: configured_manifest_path.to_path_buf(),
            });
        };
        if configured_manifest_path != loaded_manifest_path {
            return Err(NanoStartupAdmissionError::ManifestPathMismatch {
                configured: configured_manifest_path.to_path_buf(),
                loaded: loaded_manifest_path.to_path_buf(),
            });
        }

        let configured_artifact_root = policy.inventory().artifact_root_path().as_path();
        if configured_artifact_root != artifact_hashes.artifact_root_path() {
            return Err(NanoStartupAdmissionError::ArtifactRootPathMismatch {
                configured: configured_artifact_root.to_path_buf(),
                hashed: artifact_hashes.artifact_root_path().to_path_buf(),
            });
        }

        if loaded_manifest.manifest() != exact_inventory.expected() {
            return Err(NanoStartupAdmissionError::ExactInventoryManifestMismatch);
        }

        let expected_artifacts = loaded_manifest.manifest().artifacts();
        let configured_bindings = policy.inventory().artifact_bindings();
        if artifact_hashes.len() != expected_artifacts.len()
            || artifact_hashes.len() != configured_bindings.len()
        {
            return Err(NanoStartupAdmissionError::artifact(
                NanoStartupArtifactError::EvidenceCountMismatch {
                    hashed: artifact_hashes.len(),
                    manifest: expected_artifacts.len(),
                    configured_bindings: configured_bindings.len(),
                },
            ));
        }
        for hashed in artifact_hashes.iter() {
            let Some(expected) = expected_artifacts.find(hashed.kind(), hashed.artifact_id())
            else {
                return Err(NanoStartupAdmissionError::artifact(
                    NanoStartupArtifactError::UnexpectedHashedArtifact {
                        kind: hashed.kind(),
                        artifact_id: *hashed.artifact_id(),
                    },
                ));
            };
            if hashed.expected_sha256() != expected.sha256().as_bytes() {
                return Err(NanoStartupAdmissionError::artifact(
                    NanoStartupArtifactError::ManifestDigestMismatch {
                        kind: hashed.kind(),
                        artifact_id: *hashed.artifact_id(),
                        manifest_sha256: *expected.sha256().as_bytes(),
                        hashing_input_sha256: *hashed.expected_sha256(),
                    },
                ));
            }
            let Some(binding) = configured_bindings.iter().find(|binding| {
                binding.kind() == hashed.kind() && binding.artifact_id() == hashed.artifact_id()
            }) else {
                return Err(NanoStartupAdmissionError::artifact(
                    NanoStartupArtifactError::MissingConfiguredBinding {
                        kind: hashed.kind(),
                        artifact_id: *hashed.artifact_id(),
                    },
                ));
            };
            if binding.relative_path() != hashed.relative_path() {
                return Err(NanoStartupAdmissionError::artifact(
                    NanoStartupArtifactError::RelativePathMismatch {
                        kind: hashed.kind(),
                        artifact_id: *hashed.artifact_id(),
                        configured: binding.relative_path().as_str().into(),
                        hashed: hashed.relative_path().as_str().into(),
                    },
                ));
            }
            if !hashed.content_matches_manifest() {
                return Err(NanoStartupAdmissionError::artifact(
                    NanoStartupArtifactError::ContentMismatch {
                        kind: hashed.kind(),
                        artifact_id: *hashed.artifact_id(),
                        expected_sha256: *hashed.expected_sha256(),
                        observed_sha256: *hashed.observed_sha256(),
                    },
                ));
            }
        }

        let calibration_bundle =
            artifact_hashes
                .exact_calibration_bundle_sha256()
                .map_err(|source| match source {
                    CalibrationBundleHashError::ContentMismatch { artifact } => {
                        NanoStartupAdmissionError::artifact(
                            NanoStartupArtifactError::ContentMismatch {
                                kind: artifact.kind(),
                                artifact_id: *artifact.artifact_id(),
                                expected_sha256: *artifact.expected_sha256(),
                                observed_sha256: *artifact.observed_sha256(),
                            },
                        )
                    }
                    CalibrationBundleHashError::MissingCalibrationArtifact => {
                        NanoStartupAdmissionError::artifact(
                            NanoStartupArtifactError::MissingCalibrationArtifact,
                        )
                    }
                })?;

        let observed_stm32 = exact_inventory.observed_stm32();
        let observed_static = observed_stm32.static_identity();
        if controller_acquisition.controller_uid() != *observed_static.controller_uid() {
            return Err(NanoStartupAdmissionError::ControllerUidMismatch);
        }
        if controller_acquisition.boot_id() != observed_stm32.boot_id() {
            return Err(NanoStartupAdmissionError::ControllerBootIdMismatch {
                inventory: observed_stm32.boot_id(),
                acquisition: controller_acquisition.boot_id(),
            });
        }
        if controller_acquisition.firmware_abi() != observed_static.firmware_abi() {
            return Err(NanoStartupAdmissionError::ControllerFirmwareAbiMismatch {
                inventory: observed_static.firmware_abi(),
                acquisition: controller_acquisition.firmware_abi(),
            });
        }
        if controller_acquisition.firmware_build_id() != observed_static.firmware_build_id() {
            return Err(NanoStartupAdmissionError::ControllerFirmwareBuildMismatch {
                inventory: observed_static.firmware_build_id(),
                acquisition: controller_acquisition.firmware_build_id(),
            });
        }
        if controller_acquisition.actuator_config_fingerprint()
            != *observed_static.hardware_profile()
        {
            return Err(NanoStartupAdmissionError::ControllerHardwareProfileMismatch);
        }
        if controller_acquisition.capabilities() != observed_static.capabilities() {
            return Err(NanoStartupAdmissionError::ControllerCapabilitiesMismatch {
                inventory_bits: observed_static.capabilities().bits(),
                acquisition_bits: controller_acquisition.capabilities().bits(),
            });
        }

        let policy = policy
            .bind_accessories_to_manifest(loaded_manifest.manifest())
            .map_err(NanoStartupAdmissionError::AccessoryManifestBinding)?;
        let hardware_manifest = Sha256Digest::try_new(*loaded_manifest.content_sha256().as_bytes())
            .map_err(NanoStartupAdmissionError::HardwareManifestDigest)?;
        let calibration_bundle = Sha256Digest::try_new(*calibration_bundle.as_bytes())
            .map_err(NanoStartupAdmissionError::CalibrationBundleDigest)?;
        let readiness = ReadinessBinding::new(
            readiness_epoch,
            controller_acquisition.controller_uid(),
            controller_acquisition.boot_id(),
            controller_acquisition.control_epoch(),
            hardware_manifest,
            calibration_bundle,
        );

        Ok(Self {
            policy,
            exact_inventory,
            artifact_hashes,
            controller_acquisition,
            readiness,
        })
    }

    pub const fn policy(&self) -> &ManifestBoundNanoAgentPolicyConfigV1 {
        &self.policy
    }

    pub const fn exact_inventory(&self) -> &ExactInventoryAdmission {
        &self.exact_inventory
    }

    pub const fn artifact_hashes(&self) -> &ManifestArtifactHashes {
        &self.artifact_hashes
    }

    pub const fn controller_acquisition(&self) -> VerifiedControllerAcquisition {
        self.controller_acquisition
    }

    pub const fn readiness(&self) -> ReadinessBinding {
        self.readiness
    }

    /// Enter the process-lifetime authority state machine without arming.
    pub fn enter_disarmed(
        self,
        clock_epoch: NavigationClockEpoch,
        inventory_started_at: HostMonotonicTimestamp,
        readiness_admitted_at: HostMonotonicTimestamp,
    ) -> Result<DisarmedNanoStartup, NanoStartupSupervisorError> {
        let mut authority = AgentAuthoritySupervisor::new(self.policy.supervisor(), clock_epoch);
        let action = authority
            .begin_inventory(inventory_started_at)
            .map_err(NanoStartupSupervisorError::Authority)?;
        if action != SupervisorAction::InventoryRequired {
            return Err(NanoStartupSupervisorError::UnexpectedAction {
                stage: NanoStartupSupervisorStage::BeginInventory,
                expected: SupervisorAction::InventoryRequired,
                actual: action,
            });
        }
        let action = authority
            .admit_readiness(self.readiness, readiness_admitted_at)
            .map_err(NanoStartupSupervisorError::Authority)?;
        if action != SupervisorAction::Disarmed {
            return Err(NanoStartupSupervisorError::UnexpectedAction {
                stage: NanoStartupSupervisorStage::AdmitReadiness,
                expected: SupervisorAction::Disarmed,
                actual: action,
            });
        }
        Ok(DisarmedNanoStartup {
            policy: self.policy,
            exact_inventory: self.exact_inventory,
            artifact_hashes: self.artifact_hashes,
            controller_acquisition: self.controller_acquisition,
            readiness: self.readiness,
            authority,
        })
    }
}

/// Startup capability after exact readiness has entered the supervisor. The
/// contained authority is provably `Disarmed`; arming still requires an
/// explicit command and a newly applied zero receipt.
pub struct DisarmedNanoStartup {
    policy: ManifestBoundNanoAgentPolicyConfigV1,
    exact_inventory: ExactInventoryAdmission,
    artifact_hashes: ManifestArtifactHashes,
    controller_acquisition: VerifiedControllerAcquisition,
    readiness: ReadinessBinding,
    authority: AgentAuthoritySupervisor,
}

impl DisarmedNanoStartup {
    pub const fn policy(&self) -> &ManifestBoundNanoAgentPolicyConfigV1 {
        &self.policy
    }

    pub const fn exact_inventory(&self) -> &ExactInventoryAdmission {
        &self.exact_inventory
    }

    pub const fn artifact_hashes(&self) -> &ManifestArtifactHashes {
        &self.artifact_hashes
    }

    pub const fn controller_acquisition(&self) -> VerifiedControllerAcquisition {
        self.controller_acquisition
    }

    pub const fn readiness(&self) -> ReadinessBinding {
        self.readiness
    }

    pub const fn authority(&self) -> &AgentAuthoritySupervisor {
        &self.authority
    }

    pub fn into_parts(self) -> DisarmedNanoStartupParts {
        DisarmedNanoStartupParts {
            policy: self.policy,
            exact_inventory: self.exact_inventory,
            artifact_hashes: self.artifact_hashes,
            controller_acquisition: self.controller_acquisition,
            readiness: self.readiness,
            authority: self.authority,
        }
    }
}

/// Owned parts for construction of the sole Nano runtime owner.
pub struct DisarmedNanoStartupParts {
    pub policy: ManifestBoundNanoAgentPolicyConfigV1,
    pub exact_inventory: ExactInventoryAdmission,
    pub artifact_hashes: ManifestArtifactHashes,
    pub controller_acquisition: VerifiedControllerAcquisition,
    pub readiness: ReadinessBinding,
    pub authority: AgentAuthoritySupervisor,
}

#[derive(Debug)]
pub enum NanoStartupAdmissionError {
    ManifestWasNotLoadedFromFile {
        configured: PathBuf,
    },
    ManifestPathMismatch {
        configured: PathBuf,
        loaded: PathBuf,
    },
    ArtifactRootPathMismatch {
        configured: PathBuf,
        hashed: PathBuf,
    },
    ExactInventoryManifestMismatch,
    Artifact(Box<NanoStartupArtifactError>),
    ControllerUidMismatch,
    ControllerBootIdMismatch {
        inventory: robot_protocol::v2::ControllerBootId,
        acquisition: robot_protocol::v2::ControllerBootId,
    },
    ControllerFirmwareAbiMismatch {
        inventory: u16,
        acquisition: u16,
    },
    ControllerFirmwareBuildMismatch {
        inventory: u32,
        acquisition: u32,
    },
    ControllerHardwareProfileMismatch,
    ControllerCapabilitiesMismatch {
        inventory_bits: u32,
        acquisition_bits: u32,
    },
    AccessoryManifestBinding(NanoAccessoryManifestBindingError),
    HardwareManifestDigest(EvidenceValueError),
    CalibrationBundleDigest(EvidenceValueError),
}

impl NanoStartupAdmissionError {
    fn artifact(source: NanoStartupArtifactError) -> Self {
        Self::Artifact(Box::new(source))
    }
}

#[derive(Debug)]
pub enum NanoStartupArtifactError {
    EvidenceCountMismatch {
        hashed: usize,
        manifest: usize,
        configured_bindings: usize,
    },
    UnexpectedHashedArtifact {
        kind: ArtifactKind,
        artifact_id: ArtifactId,
    },
    ManifestDigestMismatch {
        kind: ArtifactKind,
        artifact_id: ArtifactId,
        manifest_sha256: [u8; 32],
        hashing_input_sha256: [u8; 32],
    },
    MissingConfiguredBinding {
        kind: ArtifactKind,
        artifact_id: ArtifactId,
    },
    RelativePathMismatch {
        kind: ArtifactKind,
        artifact_id: ArtifactId,
        configured: Box<str>,
        hashed: Box<str>,
    },
    ContentMismatch {
        kind: ArtifactKind,
        artifact_id: ArtifactId,
        expected_sha256: [u8; 32],
        observed_sha256: [u8; 32],
    },
    MissingCalibrationArtifact,
}

impl fmt::Display for NanoStartupAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Nano startup admission failed: {self:?}")
    }
}

impl std::error::Error for NanoStartupAdmissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Artifact(source) => Some(source.as_ref()),
            Self::AccessoryManifestBinding(source) => Some(source),
            Self::HardwareManifestDigest(source) | Self::CalibrationBundleDigest(source) => {
                Some(source)
            }
            _ => None,
        }
    }
}

impl fmt::Display for NanoStartupArtifactError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Nano startup artifact admission failed: {self:?}"
        )
    }
}

impl std::error::Error for NanoStartupArtifactError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoStartupSupervisorStage {
    BeginInventory,
    AdmitReadiness,
}

#[derive(Debug)]
pub enum NanoStartupSupervisorError {
    Authority(AgentAuthorityError),
    UnexpectedAction {
        stage: NanoStartupSupervisorStage,
        expected: SupervisorAction,
        actual: SupervisorAction,
    },
}

impl fmt::Display for NanoStartupSupervisorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Nano startup supervisor transition failed: {self:?}"
        )
    }
}

impl std::error::Error for NanoStartupSupervisorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Authority(source) => Some(source),
            Self::UnexpectedAction { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    use kiko_device_inventory::{
        ArtifactDigestDto, ObservedDeviceInventoryV1, ObservedDeviceInventoryV1Dto,
        ObservedOakV1Dto, ObservedStm32V1Dto, admit_exact_inventory, hash_manifest_artifacts,
        load_expected_manifest_v1_file, load_expected_manifest_v1_from_slice,
    };
    use kiko_supervisor_core::{ReadinessEpoch, SupervisorState};
    use robot_command_client::fake::{FakeClock, FakeStep, FakeTransport};
    use robot_command_client::{
        ClientConfig, ClientConfigInput, DisarmedCommandClient, VerifiedControllerAcquisition,
    };
    use robot_protocol::ControllerUptimeMsWrapping;
    use robot_protocol::v2::{
        AcquireResult, AcquireResultCode, ActuatorConfigFingerprint, ControlEpoch,
        ControllerBootId, ControllerCapabilities, ControllerDeadlineMsWrapping, ControllerFaults,
        ControllerUid, HostCommandResult, HostCommandResultCode, HostStopResult, Message,
        MessageKind, OutputState, RemainingLeaseMs, RequestId, StatusCode, StatusReport,
        StopResultCode, TargetBootId, TimerPwm, V2CommandSequence,
    };
    use serde_json::json;
    use sha2::{Digest, Sha256};

    use super::*;
    use crate::navigation::{
        AgentMapStateV1, AgentRuntimeStateV1, NanoAgentPolicyConfigV1, NavigationClockEpoch,
    };

    const UID_BYTES: [u8; 12] = [0x11; 12];
    const FINGERPRINT_BYTES: [u8; 16] = [0x22; 16];
    const FIRMWARE_ABI: u16 = 2;
    const FIRMWARE_BUILD_ID: u32 = 9;
    const BOOT_ID: u64 = 17;
    const CONTROL_EPOCH: u32 = 23;
    const CALIBRATION_BYTES: &[u8] = b"startup calibration v1";
    const PLANT_BYTES: &[u8] = b"startup plant v1";
    const RESPONSE_DELAY: Duration = Duration::from_millis(1);

    static NEXT_TEMP_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    struct DeploymentFixture {
        root: PathBuf,
        manifest_path: PathBuf,
        artifact_root: PathBuf,
        manifest_json: Vec<u8>,
    }

    impl DeploymentFixture {
        fn new() -> Self {
            let sequence = NEXT_TEMP_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let root = fs::canonicalize(std::env::temp_dir())
                .expect("canonical temp root")
                .join(format!(
                    "kiko-nano-startup-{}-{sequence}",
                    std::process::id()
                ));
            let artifact_root = root.join("artifacts");
            fs::create_dir_all(artifact_root.join("calibration")).expect("calibration directory");
            fs::create_dir_all(artifact_root.join("plant")).expect("plant directory");
            fs::write(
                artifact_root.join("calibration/main.bin"),
                CALIBRATION_BYTES,
            )
            .expect("calibration artifact");
            fs::write(artifact_root.join("plant/main.bin"), PLANT_BYTES).expect("plant artifact");
            let manifest_path = root.join("device-inventory-v1.json");
            let manifest_json = manifest_json();
            fs::write(&manifest_path, &manifest_json).expect("manifest file");
            Self {
                root,
                manifest_path,
                artifact_root,
                manifest_json,
            }
        }

        fn policy(&self) -> NanoAgentPolicyConfigV1 {
            let socket_path = self.root.join("agent-control.sock");
            let map_path = self.root.join("current.kmap");
            let value = json!({
                "schema_version": 1,
                "control": {
                    "socket_path": socket_path,
                    "read_timeout_ms": 100,
                    "write_timeout_ms": 100,
                    "runtime_response_timeout_ms": 500,
                    "runtime_queue_capacity": 8
                },
                "inventory": {
                    "manifest_path": self.manifest_path,
                    "artifact_root_path": self.artifact_root,
                    "artifact_bindings": [
                        {
                            "kind": "calibration",
                            "artifact_id": "camera-main",
                            "relative_path": "calibration/main.bin"
                        },
                        {
                            "kind": "plant",
                            "artifact_id": "drive-main",
                            "relative_path": "plant/main.bin"
                        }
                    ]
                },
                "map_persistence": {
                    "save_snapshot_path": map_path,
                    "warm_start": {"kind": "none"}
                },
                "eye": {"mode": "disabled"},
                "head": {"mode": "disabled"},
                "rgb_expression": {"mode": "disabled"},
                "supervisor": {
                    "maximum_authority_lease_ms": 1000,
                    "maximum_zero_age_ms": 250
                },
                "live_mode_policy": {
                    "startup": "disarmed_map_only",
                    "manual": {"permission": "disabled"},
                    "point_goal": {"permission": "disabled"},
                    "frontier_explore": {"permission": "disabled"}
                }
            });
            NanoAgentPolicyConfigV1::parse_json(&serde_json::to_vec(&value).expect("policy JSON"))
                .expect("parsed policy")
        }

        fn loaded_file(&self) -> LoadedExpectedManifestV1 {
            load_expected_manifest_v1_file(&self.manifest_path).expect("loaded manifest file")
        }

        fn loaded_slice(&self) -> LoadedExpectedManifestV1 {
            load_expected_manifest_v1_from_slice(&self.manifest_json)
                .expect("loaded manifest bytes")
        }

        fn hashes(
            &self,
            policy: &NanoAgentPolicyConfigV1,
            loaded: &LoadedExpectedManifestV1,
        ) -> ManifestArtifactHashes {
            hash_manifest_artifacts(
                loaded.manifest(),
                &self.artifact_root,
                policy.inventory().artifact_bindings().clone(),
            )
            .expect("exact artifact hashes")
        }

        fn exact_inventory(&self, loaded: &LoadedExpectedManifestV1) -> ExactInventoryAdmission {
            let observed = ObservedDeviceInventoryV1::parse(ObservedDeviceInventoryV1Dto {
                schema_version: kiko_device_inventory::OBSERVED_DEVICE_INVENTORY_V1,
                robot_id: "kiko-startup-test".into(),
                oak: Some(ObservedOakV1Dto {
                    mxid: "A1B2C3D4E5F60708".into(),
                    compiled_depthai_header_sdk_version: "3.6.1".into(),
                    compiled_depthai_header_sdk_commit: "abc123".into(),
                    compiled_depthai_header_embedded_device_artifact_version: "device-1".into(),
                    compiled_depthai_header_embedded_bootloader_artifact_version: "bootloader-1"
                        .into(),
                }),
                stm32: Some(ObservedStm32V1Dto {
                    serial_by_id_path: "/dev/serial/by-id/usb-Kiko_STM32_A1-if00".into(),
                    control_endpoint_identity: "udp://127.0.0.1:8080".into(),
                    controller_uid: UID_BYTES,
                    controller_boot_id: BOOT_ID,
                    firmware_abi: FIRMWARE_ABI,
                    firmware_build_id: FIRMWARE_BUILD_ID,
                    hardware_profile_fingerprint: FINGERPRINT_BYTES,
                    capabilities_bits: ControllerCapabilities::REQUIRED_BITS,
                }),
                head: None,
                eye: None,
                calibration_artifacts: vec![ArtifactDigestDto {
                    artifact_id: "camera-main".into(),
                    sha256: sha256(CALIBRATION_BYTES),
                }],
                plant_artifacts: vec![ArtifactDigestDto {
                    artifact_id: "drive-main".into(),
                    sha256: sha256(PLANT_BYTES),
                }],
            })
            .expect("observed inventory");
            admit_exact_inventory(loaded.manifest().clone(), observed)
                .expect("exact inventory admission")
        }
    }

    impl Drop for DeploymentFixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn sha256(bytes: &[u8]) -> [u8; 32] {
        Sha256::digest(bytes).into()
    }

    fn manifest_json() -> Vec<u8> {
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "robot_id": "kiko-startup-test",
            "oak": {
                "mxid": "A1B2C3D4E5F60708",
                "compiled_depthai_header_sdk_version": "3.6.1",
                "compiled_depthai_header_sdk_commit": "abc123",
                "compiled_depthai_header_embedded_device_artifact_version": "device-1",
                "compiled_depthai_header_embedded_bootloader_artifact_version": "bootloader-1"
            },
            "stm32": {
                "serial_by_id_path": "/dev/serial/by-id/usb-Kiko_STM32_A1-if00",
                "control_endpoint_identity": "udp://127.0.0.1:8080",
                "controller_uid": UID_BYTES,
                "firmware_abi": FIRMWARE_ABI,
                "firmware_build_id": FIRMWARE_BUILD_ID,
                "hardware_profile_fingerprint": FINGERPRINT_BYTES,
                "capabilities_bits": ControllerCapabilities::REQUIRED_BITS
            },
            "head": null,
            "eye": null,
            "calibration_artifacts": [{
                "artifact_id": "camera-main",
                "sha256": sha256(CALIBRATION_BYTES)
            }],
            "plant_artifacts": [{
                "artifact_id": "drive-main",
                "sha256": sha256(PLANT_BYTES)
            }]
        }))
        .expect("manifest JSON")
    }

    fn uid() -> ControllerUid {
        ControllerUid::try_new(UID_BYTES).expect("UID")
    }

    fn boot() -> ControllerBootId {
        ControllerBootId::try_new(BOOT_ID).expect("boot ID")
    }

    fn control_epoch() -> ControlEpoch {
        ControlEpoch::try_new(CONTROL_EPOCH).expect("control epoch")
    }

    fn fingerprint() -> ActuatorConfigFingerprint {
        ActuatorConfigFingerprint::try_new(FINGERPRINT_BYTES).expect("fingerprint")
    }

    fn capabilities() -> ControllerCapabilities {
        ControllerCapabilities::try_from_bits(ControllerCapabilities::REQUIRED_BITS)
            .expect("capabilities")
    }

    fn verified_controller_acquisition() -> VerifiedControllerAcquisition {
        let clock = FakeClock::default();
        let steps = [
            FakeStep::respond(
                MessageKind::StatusQuery,
                RESPONSE_DELAY,
                Message::StatusReport(StatusReport {
                    controller_uid: uid(),
                    observed_boot_id: TargetBootId::Exact(boot()),
                    request_id: RequestId::new(0),
                    status: StatusCode::ReadyStopped,
                    control_epoch: None,
                    controller_uptime: ControllerUptimeMsWrapping::new(1_000),
                    capabilities: capabilities(),
                    output_state: OutputState::Disabled,
                    controller_timer_pwm: TimerPwm::ZERO,
                    remaining_lease: RemainingLeaseMs::ZERO,
                    faults: ControllerFaults::NONE,
                }),
            ),
            FakeStep::respond(
                MessageKind::AcquireControl,
                RESPONSE_DELAY,
                Message::AcquireResult(AcquireResult {
                    controller_uid: uid(),
                    boot_id: boot(),
                    request_id: RequestId::new(1),
                    control_epoch: Some(control_epoch()),
                    result: AcquireResultCode::Granted,
                    capabilities: capabilities(),
                    faults: ControllerFaults::NONE,
                    observed_firmware_abi: FIRMWARE_ABI,
                    observed_firmware_build_id: FIRMWARE_BUILD_ID,
                    observed_actuator_config_fingerprint: fingerprint(),
                }),
            ),
            FakeStep::respond(
                MessageKind::HostCommand,
                RESPONSE_DELAY,
                Message::HostCommandResult(HostCommandResult {
                    controller_uid: uid(),
                    boot_id: boot(),
                    control_epoch: control_epoch(),
                    sequence: V2CommandSequence::FIRST,
                    result: HostCommandResultCode::AppliedNew,
                    requested_timer_pwm: TimerPwm::ZERO,
                    controller_timer_pwm: TimerPwm::ZERO,
                    output_state: OutputState::ZeroPwm,
                    controller_applied_at: ControllerUptimeMsWrapping::new(2_000),
                    controller_expires_at: ControllerDeadlineMsWrapping::new(2_100),
                    remaining_lease: RemainingLeaseMs::try_new(90).expect("remaining lease"),
                    faults: ControllerFaults::NONE,
                }),
            ),
            FakeStep::respond(
                MessageKind::HostStop,
                RESPONSE_DELAY,
                Message::HostStopResult(HostStopResult {
                    controller_uid: uid(),
                    observed_boot_id: TargetBootId::Exact(boot()),
                    request_id: RequestId::new(2),
                    result: StopResultCode::ControllerConfirmed,
                    output_state: OutputState::Disabled,
                    controller_uptime: ControllerUptimeMsWrapping::new(3_000),
                    faults: ControllerFaults::NONE,
                }),
            ),
        ];
        let (transport, _probe) = FakeTransport::scripted(clock.clone(), steps);
        let client = DisarmedCommandClient::new(
            transport,
            clock,
            ClientConfig::parse(ClientConfigInput {
                command_endpoint: "127.0.0.1:8080",
                controller_uid_hex: "111111111111111111111111",
                expected_firmware_abi: "2",
                expected_firmware_build_id: "9",
                expected_actuator_config_fingerprint_hex: "22222222222222222222222222222222",
                status_timeout_ns: "50000000",
                acquire_timeout_ns: "50000000",
                applied_ack_timeout_ns: "50000000",
                stop_attempt_timeout_ns: "50000000",
                max_stop_recovery_attempts: "3",
                zero_acquisition_lease_ms: "100",
            })
            .expect("client config"),
        );
        let (armed, _initial_zero) = client.acquire_zero().ok().expect("controller acquisition");
        let acquisition = armed.verified_acquisition();
        let (_disarmed, _stop) = armed.disarm().ok().expect("bounded explicit stop");
        acquisition
    }

    #[test]
    fn exact_startup_evidence_enters_only_disarmed() {
        let fixture = DeploymentFixture::new();
        let policy = fixture.policy();
        let loaded = fixture.loaded_file();
        let hashes = fixture.hashes(&policy, &loaded);
        let exact_inventory = fixture.exact_inventory(&loaded);
        let admitted = AdmittedNanoStartup::admit(
            policy,
            loaded,
            hashes,
            exact_inventory,
            verified_controller_acquisition(),
            ReadinessEpoch::try_new(1).expect("readiness epoch"),
        )
        .expect("startup admission");
        assert_eq!(
            admitted.readiness().control_epoch(),
            control_epoch(),
            "readiness must retain the observed acquisition epoch"
        );

        let disarmed = admitted
            .enter_disarmed(
                NavigationClockEpoch::new(HostMonotonicTimestamp::from_nanos(10)),
                HostMonotonicTimestamp::from_nanos(11),
                HostMonotonicTimestamp::from_nanos(12),
            )
            .expect("disarmed startup");
        assert!(matches!(
            disarmed.authority().state(),
            SupervisorState::Disarmed { .. }
        ));
        assert_eq!(
            disarmed
                .authority()
                .control_status(AgentMapStateV1::UNAVAILABLE)
                .runtime(),
            AgentRuntimeStateV1::Disarmed
        );
    }

    #[test]
    fn in_memory_manifest_cannot_impersonate_configured_deployment_file() {
        let fixture = DeploymentFixture::new();
        let policy = fixture.policy();
        let loaded = fixture.loaded_slice();
        let hashes = fixture.hashes(&policy, &loaded);
        let exact_inventory = fixture.exact_inventory(&loaded);
        assert!(matches!(
            AdmittedNanoStartup::admit(
                policy,
                loaded,
                hashes,
                exact_inventory,
                verified_controller_acquisition(),
                ReadinessEpoch::try_new(1).expect("readiness epoch"),
            ),
            Err(NanoStartupAdmissionError::ManifestWasNotLoadedFromFile { .. })
        ));
    }

    #[test]
    fn controller_acquisition_is_bound_to_exact_inventory_boot() {
        let fixture = DeploymentFixture::new();
        let policy = fixture.policy();
        let loaded = fixture.loaded_file();
        let hashes = fixture.hashes(&policy, &loaded);
        let exact_inventory = fixture.exact_inventory(&loaded);

        // Exact inventory itself is immutable and non-forgeable. Build a
        // second semantically exact pair whose only intentionally un-compared
        // field, STM32 boot ID, differs from the live acquisition.
        let mismatched_observed = ObservedDeviceInventoryV1::parse(ObservedDeviceInventoryV1Dto {
            schema_version: kiko_device_inventory::OBSERVED_DEVICE_INVENTORY_V1,
            robot_id: exact_inventory.observed().robot_id().as_str().into(),
            oak: Some(ObservedOakV1Dto {
                mxid: exact_inventory.observed_oak().mxid().as_str().into(),
                compiled_depthai_header_sdk_version: "3.6.1".into(),
                compiled_depthai_header_sdk_commit: "abc123".into(),
                compiled_depthai_header_embedded_device_artifact_version: "device-1".into(),
                compiled_depthai_header_embedded_bootloader_artifact_version: "bootloader-1".into(),
            }),
            stm32: Some(ObservedStm32V1Dto {
                serial_by_id_path: exact_inventory
                    .observed_stm32()
                    .static_identity()
                    .serial_path()
                    .as_str()
                    .into(),
                control_endpoint_identity: "udp://127.0.0.1:8080".into(),
                controller_uid: UID_BYTES,
                controller_boot_id: BOOT_ID + 1,
                firmware_abi: FIRMWARE_ABI,
                firmware_build_id: FIRMWARE_BUILD_ID,
                hardware_profile_fingerprint: FINGERPRINT_BYTES,
                capabilities_bits: ControllerCapabilities::REQUIRED_BITS,
            }),
            head: None,
            eye: None,
            calibration_artifacts: vec![ArtifactDigestDto {
                artifact_id: "camera-main".into(),
                sha256: sha256(CALIBRATION_BYTES),
            }],
            plant_artifacts: vec![ArtifactDigestDto {
                artifact_id: "drive-main".into(),
                sha256: sha256(PLANT_BYTES),
            }],
        })
        .expect("boot-mismatched observation");
        let observed =
            admit_exact_inventory(exact_inventory.expected().clone(), mismatched_observed)
                .expect("boot ID is intentionally dynamic inventory evidence");

        assert!(matches!(
            AdmittedNanoStartup::admit(
                policy,
                loaded,
                hashes,
                observed,
                verified_controller_acquisition(),
                ReadinessEpoch::try_new(1).expect("readiness epoch"),
            ),
            Err(NanoStartupAdmissionError::ControllerBootIdMismatch { .. })
        ));
    }
}
