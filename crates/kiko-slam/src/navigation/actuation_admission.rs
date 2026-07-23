//! Exact deployment admission for a parsed navigation actuation authority.
//!
//! [`NavigationActuationConfigV2`] intentionally remains a weak operator
//! authority document. This module binds it to one no-follow manifest load,
//! one exact inventory snapshot, and one exact plant artifact before the
//! production runtime may acquire a physical driver.

use std::fmt;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use kiko_device_inventory::{
    ArtifactId, ArtifactKind, ArtifactRelativePath, ControlEndpointTransport,
    ExactInventoryAdmission, LoadedExpectedManifestV1, ManifestArtifactHashes,
    ManifestContentSha256,
};
use robot_protocol::v2::ControllerCapabilities;

use super::NavigationActuationConfigV2;

#[cfg(test)]
const SHA256_HEX_PREFIX: &str = "sha256:";
#[cfg(test)]
const SHA256_HEX_DIGITS: usize = 64;

/// Immutable identity of the exact plant artifact admitted for actuation.
///
/// The selected artifact is the exact serialized plant model. Its digest is
/// independently bound by the actuation config and never reused as the
/// physical evidence-dataset identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdmittedPlantArtifactIdentity {
    artifact_root_path: PathBuf,
    artifact_id: ArtifactId,
    relative_path: ArtifactRelativePath,
    content_sha256: [u8; 32],
    bytes_hashed: u64,
}

impl AdmittedPlantArtifactIdentity {
    pub fn artifact_root_path(&self) -> &Path {
        &self.artifact_root_path
    }

    pub const fn artifact_id(&self) -> &ArtifactId {
        &self.artifact_id
    }

    pub const fn relative_path(&self) -> &ArtifactRelativePath {
        &self.relative_path
    }

    pub const fn content_sha256(&self) -> &[u8; 32] {
        &self.content_sha256
    }

    pub const fn bytes_hashed(&self) -> u64 {
        self.bytes_hashed
    }
}

/// Non-forgeable production authority for one exact parsed actuation config.
///
/// Construction proves that the config's robot and controller identities
/// match an exact admitted inventory, that model evidence matches the distinct
/// physical dataset identity, and that the separately configured plant
/// artifact digest names the exact selected no-follow hashed artifact.
#[derive(Debug)]
pub struct AdmittedNavigationActuationConfigV2 {
    config: NavigationActuationConfigV2,
    manifest_source_path: PathBuf,
    manifest_content_sha256: ManifestContentSha256,
    controller_capabilities: ControllerCapabilities,
    plant_artifact: AdmittedPlantArtifactIdentity,
}

impl AdmittedNavigationActuationConfigV2 {
    pub fn admit(
        config: NavigationActuationConfigV2,
        loaded_manifest: &LoadedExpectedManifestV1,
        exact_inventory: &ExactInventoryAdmission,
        artifact_hashes: &ManifestArtifactHashes,
        plant_artifact_id: &ArtifactId,
        plant_artifact_relative_path: &ArtifactRelativePath,
    ) -> Result<Self, ActuationAdmissionError> {
        let Some(manifest_source_path) = loaded_manifest.source_path() else {
            return Err(ActuationAdmissionError::ManifestWasNotLoadedFromFile);
        };
        if loaded_manifest.manifest() != exact_inventory.expected() {
            return Err(ActuationAdmissionError::ExactInventoryManifestMismatch);
        }
        if config.robot_id() != exact_inventory.expected().robot_id().as_str() {
            return Err(ActuationAdmissionError::RobotIdMismatch);
        }

        let observed_stm32 = exact_inventory.observed_stm32().static_identity();
        let inventory_endpoint = observed_stm32.control_endpoint();
        if inventory_endpoint.transport() != ControlEndpointTransport::Udp {
            return Err(
                ActuationAdmissionError::ControllerEndpointTransportMismatch {
                    inventory_transport: inventory_endpoint.transport(),
                },
            );
        }
        let config_endpoint = config.command_endpoint().socket_addr();
        if inventory_endpoint.socket_addr() != Some(config_endpoint) {
            return Err(ActuationAdmissionError::ControllerEndpointMismatch {
                config: config_endpoint,
                inventory: inventory_endpoint.socket_addr(),
            });
        }
        if config.controller_uid() != *observed_stm32.controller_uid() {
            return Err(ActuationAdmissionError::ControllerUidMismatch);
        }
        if config.firmware_abi().get() != observed_stm32.firmware_abi() {
            return Err(ActuationAdmissionError::ControllerFirmwareAbiMismatch {
                config: config.firmware_abi().get(),
                inventory: observed_stm32.firmware_abi(),
            });
        }
        if config.firmware_build_id().get() != observed_stm32.firmware_build_id() {
            return Err(ActuationAdmissionError::ControllerFirmwareBuildMismatch {
                config: config.firmware_build_id().get(),
                inventory: observed_stm32.firmware_build_id(),
            });
        }
        if config.actuator_config_fingerprint() != *observed_stm32.hardware_profile() {
            return Err(ActuationAdmissionError::ControllerHardwareProfileMismatch);
        }
        // Capabilities are deliberately not duplicated in the weak V1
        // actuation document. Exact-inventory construction already proves
        // that the observed capabilities equal the manifest's mandatory
        // safety-capability set for this exact controller identity.
        debug_assert!(observed_stm32.capabilities().supports_required_safety());

        let manifest_artifacts = loaded_manifest.manifest().artifacts();
        if artifact_hashes.len() != manifest_artifacts.len() {
            return Err(ActuationAdmissionError::ArtifactEvidenceCountMismatch {
                hashed: artifact_hashes.len(),
                manifest: manifest_artifacts.len(),
            });
        }
        for hashed in artifact_hashes.iter() {
            let Some(expected) = manifest_artifacts.find(hashed.kind(), hashed.artifact_id())
            else {
                return Err(ActuationAdmissionError::ArtifactEvidenceNotInManifest {
                    kind: hashed.kind(),
                    artifact_id: *hashed.artifact_id(),
                });
            };
            if expected.sha256().as_bytes() != hashed.expected_sha256() {
                return Err(ActuationAdmissionError::ArtifactManifestDigestMismatch {
                    kind: hashed.kind(),
                    artifact_id: *hashed.artifact_id(),
                });
            }
            if !hashed.content_matches_manifest() {
                return Err(ActuationAdmissionError::ArtifactContentMismatch(Box::new(
                    ArtifactContentMismatch {
                        kind: hashed.kind(),
                        artifact_id: *hashed.artifact_id(),
                        expected_sha256: *hashed.expected_sha256(),
                        observed_sha256: *hashed.observed_sha256(),
                    },
                )));
            }
        }

        let Some(manifest_plant) = manifest_artifacts.find(ArtifactKind::Plant, plant_artifact_id)
        else {
            return Err(ActuationAdmissionError::SelectedPlantNotInManifest {
                artifact_id: *plant_artifact_id,
            });
        };
        let Some(hashed_plant) = artifact_hashes.iter().find(|artifact| {
            artifact.kind() == ArtifactKind::Plant && artifact.artifact_id() == plant_artifact_id
        }) else {
            return Err(ActuationAdmissionError::SelectedPlantWasNotHashed {
                artifact_id: *plant_artifact_id,
            });
        };
        if hashed_plant.relative_path() != plant_artifact_relative_path {
            return Err(ActuationAdmissionError::SelectedPlantPathMismatch {
                artifact_id: *plant_artifact_id,
                selected: plant_artifact_relative_path.clone(),
                hashed: hashed_plant.relative_path().clone(),
            });
        }
        if manifest_plant.sha256().as_bytes() != hashed_plant.observed_sha256() {
            return Err(ActuationAdmissionError::SelectedPlantDigestMismatch {
                artifact_id: *plant_artifact_id,
            });
        }

        let configured_plant_sha256 = config.plant_artifact_content_sha256();
        if configured_plant_sha256.as_bytes() != hashed_plant.observed_sha256() {
            return Err(
                ActuationAdmissionError::ConfiguredPlantArtifactDigestMismatch(Box::new(
                    ConfiguredPlantArtifactDigestMismatch {
                        artifact_id: *plant_artifact_id,
                        configured_sha256: *configured_plant_sha256.as_bytes(),
                        observed_sha256: *hashed_plant.observed_sha256(),
                    },
                )),
            );
        }

        Ok(Self {
            config,
            manifest_source_path: manifest_source_path.to_path_buf(),
            manifest_content_sha256: loaded_manifest.content_sha256(),
            controller_capabilities: observed_stm32.capabilities(),
            plant_artifact: AdmittedPlantArtifactIdentity {
                artifact_root_path: artifact_hashes.artifact_root_path().to_path_buf(),
                artifact_id: *plant_artifact_id,
                relative_path: plant_artifact_relative_path.clone(),
                content_sha256: *hashed_plant.observed_sha256(),
                bytes_hashed: hashed_plant.bytes_hashed(),
            },
        })
    }

    pub const fn config(&self) -> &NavigationActuationConfigV2 {
        &self.config
    }

    pub fn manifest_source_path(&self) -> &Path {
        &self.manifest_source_path
    }

    pub const fn manifest_content_sha256(&self) -> ManifestContentSha256 {
        self.manifest_content_sha256
    }

    pub const fn controller_capabilities(&self) -> ControllerCapabilities {
        self.controller_capabilities
    }

    pub const fn plant_artifact(&self) -> &AdmittedPlantArtifactIdentity {
        &self.plant_artifact
    }
}

/// Compatibility name for call sites that predate actuation schema V2.
pub type AdmittedNavigationActuationConfigV1 = AdmittedNavigationActuationConfigV2;

#[cfg(test)]
fn parse_canonical_sha256_id(value: &str) -> Option<[u8; 32]> {
    let hex = value.strip_prefix(SHA256_HEX_PREFIX)?;
    if hex.len() != SHA256_HEX_DIGITS {
        return None;
    }
    let mut digest = [0_u8; 32];
    for (index, pair) in hex.as_bytes().chunks_exact(2).enumerate() {
        let high = canonical_hex_nibble(pair[0])?;
        let low = canonical_hex_nibble(pair[1])?;
        digest[index] = (high << 4) | low;
    }
    Some(digest)
}

#[cfg(test)]
const fn canonical_hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ActuationAdmissionError {
    ManifestWasNotLoadedFromFile,
    ExactInventoryManifestMismatch,
    RobotIdMismatch,
    ControllerEndpointTransportMismatch {
        inventory_transport: ControlEndpointTransport,
    },
    ControllerEndpointMismatch {
        config: SocketAddr,
        inventory: Option<SocketAddr>,
    },
    ControllerUidMismatch,
    ControllerFirmwareAbiMismatch {
        config: u16,
        inventory: u16,
    },
    ControllerFirmwareBuildMismatch {
        config: u32,
        inventory: u32,
    },
    ControllerHardwareProfileMismatch,
    ArtifactEvidenceCountMismatch {
        hashed: usize,
        manifest: usize,
    },
    ArtifactEvidenceNotInManifest {
        kind: ArtifactKind,
        artifact_id: ArtifactId,
    },
    ArtifactManifestDigestMismatch {
        kind: ArtifactKind,
        artifact_id: ArtifactId,
    },
    ArtifactContentMismatch(Box<ArtifactContentMismatch>),
    SelectedPlantNotInManifest {
        artifact_id: ArtifactId,
    },
    SelectedPlantWasNotHashed {
        artifact_id: ArtifactId,
    },
    SelectedPlantPathMismatch {
        artifact_id: ArtifactId,
        selected: ArtifactRelativePath,
        hashed: ArtifactRelativePath,
    },
    SelectedPlantDigestMismatch {
        artifact_id: ArtifactId,
    },
    ConfiguredPlantArtifactDigestMismatch(Box<ConfiguredPlantArtifactDigestMismatch>),
}

#[derive(Debug, PartialEq, Eq)]
pub struct ArtifactContentMismatch {
    pub kind: ArtifactKind,
    pub artifact_id: ArtifactId,
    pub expected_sha256: [u8; 32],
    pub observed_sha256: [u8; 32],
}

#[derive(Debug, PartialEq, Eq)]
pub struct ConfiguredPlantArtifactDigestMismatch {
    pub artifact_id: ArtifactId,
    pub configured_sha256: [u8; 32],
    pub observed_sha256: [u8; 32],
}

impl fmt::Display for ActuationAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "navigation actuation deployment admission failed: {self:?}"
        )
    }
}

impl std::error::Error for ActuationAdmissionError {}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::num::NonZeroU64;
    use std::sync::atomic::{AtomicU64, Ordering};

    use kiko_device_inventory::{
        ArtifactDigestDto, ArtifactFileBindingInput, ArtifactFileBindingSet,
        DeviceInventoryManifestV1, ExactInventoryAdmission, ObservedDeviceInventoryV1,
        ObservedDeviceInventoryV1Dto, ObservedOakV1Dto, ObservedStm32V1Dto, admit_exact_inventory,
        hash_manifest_artifacts, load_expected_manifest_v1_file,
        load_expected_manifest_v1_from_slice,
    };
    use robot_protocol::v2::ControllerCapabilities;
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};

    use super::*;
    use crate::navigation::mpc::{
        FitResidualsV1Dto, PLANT_MODEL_V1, PlantEvidenceV1Dto, PlantModelV1, PlantModelV1Dto,
        PlantValidityEnvelopeV1Dto, WheelPlantV1Dto,
    };
    use crate::navigation::{ControlPeriodNs, SolverBudgetNs};

    const ROBOT_ID: &str = "kiko-actuation-admission";
    const NAVIGATION_BYTES: &[u8] = br#"{"schema":"navigation-fixture"}"#;
    const CALIBRATION_BYTES: &[u8] = b"actuation admission calibration";
    const PLANT_BYTES: &[u8] = b"actuation admission physical dataset";
    const UID: [u8; 12] = [0x11; 12];
    const FINGERPRINT: [u8; 16] = [0x22; 16];
    const FIRMWARE_ABI: u16 = 2;
    const FIRMWARE_BUILD: u32 = 42;
    const ENDPOINT: &str = "127.0.0.1:8080";
    const INVENTORY_ENDPOINT: &str = "udp://127.0.0.1:8080";

    static NEXT_TEMP_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    struct Fixture {
        root: PathBuf,
        manifest_path: PathBuf,
        artifact_root: PathBuf,
        manifest_json: Vec<u8>,
    }

    impl Fixture {
        fn new() -> Self {
            let sequence = NEXT_TEMP_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let root = fs::canonicalize(std::env::temp_dir())
                .expect("canonical temp directory")
                .join(format!(
                    "kiko-actuation-admission-{}-{sequence}",
                    std::process::id()
                ));
            let artifact_root = root.join("artifacts");
            fs::create_dir_all(artifact_root.join("calibration")).expect("calibration directory");
            fs::create_dir_all(artifact_root.join("plant")).expect("plant directory");
            fs::write(
                artifact_root.join("calibration/camera.bin"),
                CALIBRATION_BYTES,
            )
            .expect("calibration file");
            fs::write(artifact_root.join("plant/drive.bin"), PLANT_BYTES).expect("plant file");

            let manifest_json = manifest_json();
            let manifest_path = root.join("device-inventory-v1.json");
            fs::write(&manifest_path, &manifest_json).expect("manifest file");
            Self {
                root,
                manifest_path,
                artifact_root,
                manifest_json,
            }
        }

        fn loaded(&self) -> LoadedExpectedManifestV1 {
            load_expected_manifest_v1_file(&self.manifest_path).expect("loaded manifest")
        }

        fn inventory(&self, manifest: &DeviceInventoryManifestV1) -> ExactInventoryAdmission {
            admit_exact_inventory(
                manifest.clone(),
                ObservedDeviceInventoryV1::parse(observed_dto()).expect("observed inventory"),
            )
            .expect("exact inventory")
        }

        fn hashes(&self, manifest: &DeviceInventoryManifestV1) -> ManifestArtifactHashes {
            hash_manifest_artifacts(manifest, &self.artifact_root, bindings())
                .expect("artifact hashes")
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn sha256(bytes: &[u8]) -> [u8; 32] {
        Sha256::digest(bytes).into()
    }

    fn canonical_sha256_id(bytes: &[u8]) -> String {
        let mut output = String::from(SHA256_HEX_PREFIX);
        for byte in sha256(bytes) {
            use fmt::Write;
            write!(output, "{byte:02x}").expect("write to String");
        }
        output
    }

    fn manifest_json() -> Vec<u8> {
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "robot_id": ROBOT_ID,
            "oak": {
                "mxid": "A1B2C3D4E5F60708",
                "compiled_depthai_header_sdk_version": "3.6.1",
                "compiled_depthai_header_sdk_commit": "abc123",
                "compiled_depthai_header_embedded_device_artifact_version": "device-1",
                "compiled_depthai_header_embedded_bootloader_artifact_version": "bootloader-1"
            },
            "stm32": {
                "serial_by_id_path": "/dev/serial/by-id/usb-Kiko_STM32_A1-if00",
                "control_endpoint_identity": INVENTORY_ENDPOINT,
                "controller_uid": UID,
                "firmware_abi": FIRMWARE_ABI,
                "firmware_build_id": FIRMWARE_BUILD,
                "hardware_profile_fingerprint": FINGERPRINT,
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

    fn observed_dto() -> ObservedDeviceInventoryV1Dto {
        ObservedDeviceInventoryV1Dto {
            schema_version: 1,
            robot_id: ROBOT_ID.into(),
            oak: Some(ObservedOakV1Dto {
                mxid: "A1B2C3D4E5F60708".into(),
                compiled_depthai_header_sdk_version: "3.6.1".into(),
                compiled_depthai_header_sdk_commit: "abc123".into(),
                compiled_depthai_header_embedded_device_artifact_version: "device-1".into(),
                compiled_depthai_header_embedded_bootloader_artifact_version: "bootloader-1".into(),
            }),
            stm32: Some(ObservedStm32V1Dto {
                serial_by_id_path: "/dev/serial/by-id/usb-Kiko_STM32_A1-if00".into(),
                control_endpoint_identity: INVENTORY_ENDPOINT.into(),
                controller_uid: UID,
                controller_boot_id: 7,
                firmware_abi: FIRMWARE_ABI,
                firmware_build_id: FIRMWARE_BUILD,
                hardware_profile_fingerprint: FINGERPRINT,
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
        }
    }

    fn bindings() -> ArtifactFileBindingSet {
        ArtifactFileBindingSet::parse(vec![
            ArtifactFileBindingInput {
                kind: ArtifactKind::Calibration,
                artifact_id: "camera-main".into(),
                relative_path: "calibration/camera.bin".into(),
            },
            ArtifactFileBindingInput {
                kind: ArtifactKind::Plant,
                artifact_id: "drive-main".into(),
                relative_path: "plant/drive.bin".into(),
            },
        ])
        .expect("artifact bindings")
    }

    fn physical_model(dataset_content_id: String) -> PlantModelV1 {
        PlantModelV1::parse(PlantModelV1Dto {
            schema_version: PLANT_MODEL_V1,
            model_id: "kiko-physical-v1".into(),
            model_version: 1,
            sample_period_s: 0.1,
            wheelbase_m: 0.3,
            left: WheelPlantV1Dto {
                velocity_gain_mps_per_pwm_percent: 0.01,
                time_constant_s: 0.2,
            },
            right: WheelPlantV1Dto {
                velocity_gain_mps_per_pwm_percent: 0.01,
                time_constant_s: 0.2,
            },
            validity: PlantValidityEnvelopeV1Dto {
                left_pwm_min_percent: -50,
                left_pwm_max_percent: 50,
                right_pwm_min_percent: -50,
                right_pwm_max_percent: 50,
                left_velocity_min_mps: -0.5,
                left_velocity_max_mps: 0.5,
                right_velocity_min_mps: -0.5,
                right_velocity_max_mps: 0.5,
                max_abs_yaw_rate_rad_s: 3.0,
                max_abs_lateral_velocity_mps: 0.1,
            },
            evidence: PlantEvidenceV1Dto::ClaimedPhysicalIdentification {
                dataset_content_id,
                identification_method_id: "method-v1".into(),
                sample_count: 100,
                residuals: FitResidualsV1Dto {
                    left_velocity_rmse_mps: 0.01,
                    right_velocity_rmse_mps: 0.02,
                    yaw_rate_rmse_rad_s: 0.03,
                    max_abs_velocity_error_mps: 0.04,
                },
            },
        })
        .expect("physical model")
    }

    fn actuation_json(dataset_content_id: &str) -> Value {
        let navigation_hash = sha256(NAVIGATION_BYTES);
        let navigation_hash_hex: String = navigation_hash
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        json!({
            "schema_version": 2,
            "robot_id": ROBOT_ID,
            "command_endpoint": ENDPOINT,
            "navigation_config_sha256_hex": navigation_hash_hex,
            "controller_uid_hex": "111111111111111111111111",
            "firmware_abi": FIRMWARE_ABI,
            "firmware_build_id": FIRMWARE_BUILD,
            "actuator_config_fingerprint_hex": "22222222222222222222222222222222",
            "plant_model_id": "kiko-physical-v1",
            "plant_model_version": 1,
            "plant_artifact_sha256_hex": canonical_sha256_id(PLANT_BYTES)
                .strip_prefix(SHA256_HEX_PREFIX)
                .expect("canonical prefix"),
            "operator_claimed_physical_approval": {
                "approval_id": "approval-v1",
                "approver_id": "operator@example.com",
                "plant_dataset_content_id": dataset_content_id,
                "plant_identification_method_id": "method-v1",
                "plant_sample_count": 100,
                "plant_fit_residuals": {
                    "left_velocity_rmse_mps": 0.01,
                    "right_velocity_rmse_mps": 0.02,
                    "yaw_rate_rmse_rad_s": 0.03,
                    "max_abs_velocity_error_mps": 0.04
                },
                "imu_calibration_id": "imu-cal-v1",
                "stereo_calibration_id": "stereo-cal-v1",
                "tracking_camera_to_base_calibration_id": "extrinsic-v1"
            },
            "apply_ack_budget_ns": 20_000_000,
            "stop_ack_budget_ns": 30_000_000,
            "scheduling_guard_ns": 5_000_000,
            "controller_motion_lease_ms": 200,
            "controller_deadline_tolerance_ns": 2_000_000,
            "maximum_uncommanded_motion_ns": 222_000_000
        })
    }

    fn config(dataset_content_id: String) -> NavigationActuationConfigV2 {
        config_from_json(actuation_json(&dataset_content_id), dataset_content_id)
    }

    fn config_from_json(value: Value, dataset_content_id: String) -> NavigationActuationConfigV2 {
        config_from_json_for_robot(value, dataset_content_id, ROBOT_ID)
    }

    fn config_from_json_for_robot(
        value: Value,
        dataset_content_id: String,
        requested_robot_id: &str,
    ) -> NavigationActuationConfigV2 {
        let bytes = serde_json::to_vec(&value).expect("actuation JSON");
        NavigationActuationConfigV2::parse_and_authorize(
            &bytes,
            requested_robot_id,
            NAVIGATION_BYTES,
            physical_model(dataset_content_id),
            SolverBudgetNs::try_new(50_000_000).expect("solver budget"),
            ControlPeriodNs::from_nonzero(NonZeroU64::new(100_000_000).expect("control period")),
        )
        .expect("actuation config")
    }

    fn plant_id(manifest: &DeviceInventoryManifestV1) -> ArtifactId {
        *manifest
            .artifacts()
            .iter()
            .find(|artifact| artifact.kind() == ArtifactKind::Plant)
            .expect("plant artifact")
            .artifact_id()
    }

    fn plant_path() -> ArtifactRelativePath {
        ArtifactRelativePath::parse("plant/drive.bin".into()).expect("plant path")
    }

    fn distinct_dataset_content_id() -> String {
        canonical_sha256_id(b"distinct physical evidence dataset")
    }

    #[test]
    fn exact_controller_manifest_and_plant_evidence_admit() {
        let fixture = Fixture::new();
        let loaded = fixture.loaded();
        let inventory = fixture.inventory(loaded.manifest());
        let hashes = fixture.hashes(loaded.manifest());
        let plant_id = plant_id(loaded.manifest());
        let dataset_content_id = distinct_dataset_content_id();
        let admitted = AdmittedNavigationActuationConfigV1::admit(
            config(dataset_content_id.clone()),
            &loaded,
            &inventory,
            &hashes,
            &plant_id,
            &plant_path(),
        )
        .expect("exact actuation admission");

        assert_eq!(admitted.config().robot_id(), ROBOT_ID);
        assert_eq!(admitted.manifest_source_path(), fixture.manifest_path);
        assert_eq!(
            admitted.controller_capabilities().bits(),
            ControllerCapabilities::REQUIRED_BITS
        );
        assert_eq!(admitted.plant_artifact().artifact_id(), &plant_id);
        assert_eq!(
            admitted.plant_artifact().content_sha256(),
            &sha256(PLANT_BYTES)
        );
        assert_eq!(
            admitted.plant_artifact().bytes_hashed(),
            PLANT_BYTES.len() as u64
        );
        assert_eq!(
            admitted
                .config()
                .approval()
                .plant_dataset_content_id()
                .as_str(),
            dataset_content_id
        );
        assert_ne!(
            admitted
                .config()
                .approval()
                .plant_dataset_content_id()
                .sha256(),
            admitted.plant_artifact().content_sha256()
        );
    }

    #[test]
    fn in_memory_manifest_and_wrong_plant_artifact_binding_fail_closed() {
        let fixture = Fixture::new();
        let loaded_file = fixture.loaded();
        let inventory = fixture.inventory(loaded_file.manifest());
        let hashes = fixture.hashes(loaded_file.manifest());
        let loaded_slice = load_expected_manifest_v1_from_slice(&fixture.manifest_json)
            .expect("in-memory manifest");
        let plant_id = plant_id(loaded_file.manifest());

        assert_eq!(
            AdmittedNavigationActuationConfigV1::admit(
                config(distinct_dataset_content_id()),
                &loaded_slice,
                &inventory,
                &hashes,
                &plant_id,
                &plant_path(),
            )
            .expect_err("in-memory manifest must not admit"),
            ActuationAdmissionError::ManifestWasNotLoadedFromFile
        );
        let dataset_content_id = distinct_dataset_content_id();
        let mut wrong_artifact = actuation_json(&dataset_content_id);
        wrong_artifact["plant_artifact_sha256_hex"] = json!("00".repeat(32));
        assert!(matches!(
            AdmittedNavigationActuationConfigV1::admit(
                config_from_json(wrong_artifact, dataset_content_id),
                &loaded_file,
                &inventory,
                &hashes,
                &plant_id,
                &plant_path(),
            ),
            Err(ActuationAdmissionError::ConfiguredPlantArtifactDigestMismatch(_))
        ));
    }

    #[test]
    fn wrong_selector_path_and_changed_content_fail_closed() {
        let fixture = Fixture::new();
        let loaded = fixture.loaded();
        let inventory = fixture.inventory(loaded.manifest());
        let hashes = fixture.hashes(loaded.manifest());
        let plant_id = plant_id(loaded.manifest());
        let wrong_path =
            ArtifactRelativePath::parse("plant/not-drive.bin".into()).expect("wrong path");
        assert!(matches!(
            AdmittedNavigationActuationConfigV1::admit(
                config(distinct_dataset_content_id()),
                &loaded,
                &inventory,
                &hashes,
                &plant_id,
                &wrong_path,
            ),
            Err(ActuationAdmissionError::SelectedPlantPathMismatch { .. })
        ));

        fs::write(
            fixture.artifact_root.join("plant/drive.bin"),
            b"changed plant",
        )
        .expect("changed plant");
        let changed_hashes = fixture.hashes(loaded.manifest());
        assert!(matches!(
            AdmittedNavigationActuationConfigV1::admit(
                config(distinct_dataset_content_id()),
                &loaded,
                &inventory,
                &changed_hashes,
                &plant_id,
                &plant_path(),
            ),
            Err(ActuationAdmissionError::ArtifactContentMismatch(source))
                if source.kind == ArtifactKind::Plant
        ));
    }

    #[test]
    fn controller_endpoint_and_static_identity_must_match_exact_inventory() {
        let fixture = Fixture::new();
        let loaded = fixture.loaded();
        let inventory = fixture.inventory(loaded.manifest());
        let hashes = fixture.hashes(loaded.manifest());
        let plant_id = plant_id(loaded.manifest());
        let dataset_content_id = distinct_dataset_content_id();

        let mut wrong_endpoint = actuation_json(&dataset_content_id);
        wrong_endpoint["command_endpoint"] = json!("127.0.0.1:8081");
        assert!(matches!(
            AdmittedNavigationActuationConfigV1::admit(
                config_from_json(wrong_endpoint, dataset_content_id.clone()),
                &loaded,
                &inventory,
                &hashes,
                &plant_id,
                &plant_path(),
            ),
            Err(ActuationAdmissionError::ControllerEndpointMismatch { .. })
        ));

        let mut wrong_build = actuation_json(&dataset_content_id);
        wrong_build["firmware_build_id"] = json!(FIRMWARE_BUILD + 1);
        assert_eq!(
            AdmittedNavigationActuationConfigV1::admit(
                config_from_json(wrong_build, dataset_content_id),
                &loaded,
                &inventory,
                &hashes,
                &plant_id,
                &plant_path(),
            )
            .expect_err("different controller build must not admit"),
            ActuationAdmissionError::ControllerFirmwareBuildMismatch {
                config: FIRMWARE_BUILD + 1,
                inventory: FIRMWARE_BUILD,
            }
        );

        let mut wrong_uid = actuation_json(&distinct_dataset_content_id());
        wrong_uid["controller_uid_hex"] = json!("333333333333333333333333");
        assert_eq!(
            AdmittedNavigationActuationConfigV1::admit(
                config_from_json(wrong_uid, distinct_dataset_content_id()),
                &loaded,
                &inventory,
                &hashes,
                &plant_id,
                &plant_path(),
            )
            .expect_err("different controller UID must not admit"),
            ActuationAdmissionError::ControllerUidMismatch
        );

        let mut wrong_fingerprint = actuation_json(&distinct_dataset_content_id());
        wrong_fingerprint["actuator_config_fingerprint_hex"] =
            json!("44444444444444444444444444444444");
        assert_eq!(
            AdmittedNavigationActuationConfigV1::admit(
                config_from_json(wrong_fingerprint, distinct_dataset_content_id()),
                &loaded,
                &inventory,
                &hashes,
                &plant_id,
                &plant_path(),
            )
            .expect_err("different hardware profile must not admit"),
            ActuationAdmissionError::ControllerHardwareProfileMismatch
        );

        let other_robot = "kiko-other";
        let mut wrong_robot = actuation_json(&distinct_dataset_content_id());
        wrong_robot["robot_id"] = json!(other_robot);
        assert_eq!(
            AdmittedNavigationActuationConfigV1::admit(
                config_from_json_for_robot(
                    wrong_robot,
                    distinct_dataset_content_id(),
                    other_robot,
                ),
                &loaded,
                &inventory,
                &hashes,
                &plant_id,
                &plant_path(),
            )
            .expect_err("different robot must not admit"),
            ActuationAdmissionError::RobotIdMismatch
        );
    }

    #[test]
    fn canonical_sha256_parser_rejects_uppercase_and_wrong_length() {
        let digest = sha256(PLANT_BYTES);
        assert_eq!(
            parse_canonical_sha256_id(&canonical_sha256_id(PLANT_BYTES)),
            Some(digest)
        );
        assert_eq!(
            parse_canonical_sha256_id(&canonical_sha256_id(PLANT_BYTES).to_uppercase()),
            None
        );
        assert_eq!(parse_canonical_sha256_id("sha256:00"), None);
    }
}
