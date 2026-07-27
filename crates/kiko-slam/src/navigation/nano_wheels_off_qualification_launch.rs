//! Strict launch boundary for the manually invoked wheels-off full-stack
//! qualifier.
//!
//! This schema deliberately has no production physical-actuation document.
//! The V3 launch binds the common OAK/SLAM/occupancy/inference graph to a
//! schema-V2 candidate inventory, a candidate-only controller contract, and
//! a candidate-only host policy. Production and qualification documents are
//! different Rust types and reject each other's schemas.

use std::fmt;
use std::path::Path;

use kiko_device_inventory::{
    ArtifactRelativePath, ArtifactRelativePathError, DeploymentAssetByteLimit,
    DeploymentAssetByteLimitError, DeploymentAssetContentSha256, DeploymentAssetLoadError,
    LoadedDeploymentAsset, MAX_DEPLOYMENT_ASSET_BYTES, MAX_MANIFEST_JSON_BYTES, MAX_ROBOT_ID_BYTES,
    load_deployment_asset,
};
use serde::Deserialize;

use super::nano_agent_launch::{
    NanoAgentLaunchParseError, NanoLaunchAssetBinding, NanoLaunchAssetBindingDto,
    NanoLaunchAssetRole, NanoLaunchCalibrationArtifact, NanoLaunchCalibrationArtifactDto,
    NanoLaunchControllerServer, NanoLaunchControllerServerDto, NanoLaunchInference,
    NanoLaunchInferenceDto, NanoLaunchOccupancy, NanoLaunchOccupancyDto, NanoLaunchPlantArtifact,
    NanoLaunchPlantArtifactDto, NanoLaunchRerun, NanoLaunchRerunDto, NanoLaunchSha256Error,
    NanoLaunchStorage, NanoLaunchStorageDto, NanoOakStreamGraph, NanoOakStreamGraphDto,
    parse_asset, parse_calibration_artifact, parse_controller_server, parse_inference, parse_oak,
    parse_occupancy, parse_plant_artifact, parse_rerun, parse_sha256, parse_storage,
};
use super::{
    MAX_NANO_AGENT_POLICY_CONFIG_JSON_BYTES, MAX_NANO_CALIBRATION_ARTIFACT_JSON_BYTES,
    MAX_NANO_WHEELS_OFF_NATIVE_RUNTIME_JSON_BYTES, MAX_SHADOW_NAVIGATION_CONFIG_JSON_BYTES,
    MAX_WHEELS_OFF_CANDIDATE_POLICY_JSON_BYTES,
};

pub const NANO_WHEELS_OFF_QUALIFICATION_LAUNCH_V3: u32 = 3;
pub const MAX_NANO_WHEELS_OFF_QUALIFICATION_LAUNCH_JSON_BYTES: usize = 64 * 1_024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoWheelsOffQualificationAssetRole {
    QualificationExecutable,
    NativeRuntimeManifest,
    AgentPolicy,
    NavigationShadowConfig,
    CandidateInventoryManifest,
    CandidateControllerPolicy,
    ControllerServerContract,
    CalibrationArtifact,
    PlantArtifact,
    OnnxRuntimeLibrary,
    SuperpointModel,
    LightglueModel,
}

impl NanoWheelsOffQualificationAssetRole {
    pub(crate) const ALL: [Self; 12] = [
        Self::QualificationExecutable,
        Self::NativeRuntimeManifest,
        Self::AgentPolicy,
        Self::NavigationShadowConfig,
        Self::CandidateInventoryManifest,
        Self::CandidateControllerPolicy,
        Self::ControllerServerContract,
        Self::CalibrationArtifact,
        Self::PlantArtifact,
        Self::OnnxRuntimeLibrary,
        Self::SuperpointModel,
        Self::LightglueModel,
    ];

    const fn maximum_bytes(self) -> u64 {
        match self {
            Self::QualificationExecutable => MAX_DEPLOYMENT_ASSET_BYTES,
            Self::NativeRuntimeManifest => MAX_NANO_WHEELS_OFF_NATIVE_RUNTIME_JSON_BYTES,
            Self::AgentPolicy => MAX_NANO_AGENT_POLICY_CONFIG_JSON_BYTES as u64,
            Self::NavigationShadowConfig => MAX_SHADOW_NAVIGATION_CONFIG_JSON_BYTES as u64,
            Self::CandidateInventoryManifest => MAX_MANIFEST_JSON_BYTES as u64,
            Self::CandidateControllerPolicy => MAX_WHEELS_OFF_CANDIDATE_POLICY_JSON_BYTES as u64,
            Self::ControllerServerContract => super::MAX_CONTROLLER_SERVER_CONTRACT_JSON_BYTES,
            Self::CalibrationArtifact => MAX_NANO_CALIBRATION_ARTIFACT_JSON_BYTES as u64,
            Self::PlantArtifact
            | Self::OnnxRuntimeLibrary
            | Self::SuperpointModel
            | Self::LightglueModel => MAX_DEPLOYMENT_ASSET_BYTES,
        }
    }
}

/// A bounded logical robot selector independent of the candidate manifest.
///
/// It is cross-bound to the manifest during bootstrap. Hardware identity still
/// comes from the connected devices and the controller acquisition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QualificationRobotId(Box<str>);

impl QualificationRobotId {
    fn parse(value: String) -> Result<Self, NanoWheelsOffQualificationLaunchParseError> {
        if value.is_empty()
            || value.len() > MAX_ROBOT_ID_BYTES
            || value.bytes().all(|byte| byte == b'0')
            || !value.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':')
            })
        {
            return Err(NanoWheelsOffQualificationLaunchParseError::InvalidRobotId {
                actual_bytes: value.len(),
                maximum_bytes: MAX_ROBOT_ID_BYTES,
            });
        }
        Ok(Self(value.into_boxed_str()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Fully parsed qualification-only launch document.
#[derive(Clone, Debug, PartialEq)]
pub struct NanoWheelsOffQualificationLaunchV3 {
    robot_id: QualificationRobotId,
    qualification_executable: NanoLaunchAssetBinding,
    native_runtime_manifest: NanoLaunchAssetBinding,
    agent_policy: NanoLaunchAssetBinding,
    navigation_shadow_config: NanoLaunchAssetBinding,
    candidate_inventory_manifest: NanoLaunchAssetBinding,
    candidate_controller_policy: NanoLaunchAssetBinding,
    controller_server: NanoLaunchControllerServer,
    calibration_artifact: NanoLaunchCalibrationArtifact,
    plant_artifact: NanoLaunchPlantArtifact,
    oak: NanoOakStreamGraph,
    occupancy: NanoLaunchOccupancy,
    inference: NanoLaunchInference,
    rerun: NanoLaunchRerun,
    storage: NanoLaunchStorage,
}

impl NanoWheelsOffQualificationLaunchV3 {
    pub fn parse_json(json: &[u8]) -> Result<Self, NanoWheelsOffQualificationLaunchParseError> {
        if json.len() > MAX_NANO_WHEELS_OFF_QUALIFICATION_LAUNCH_JSON_BYTES {
            return Err(NanoWheelsOffQualificationLaunchParseError::InputTooLarge {
                actual_bytes: json.len(),
                maximum_bytes: MAX_NANO_WHEELS_OFF_QUALIFICATION_LAUNCH_JSON_BYTES,
            });
        }
        let mut deserializer = serde_json::Deserializer::from_slice(json);
        let dto = NanoWheelsOffQualificationLaunchV3Dto::deserialize(&mut deserializer)
            .map_err(NanoWheelsOffQualificationLaunchParseError::JsonDecode)?;
        deserializer
            .end()
            .map_err(NanoWheelsOffQualificationLaunchParseError::JsonTrailingData)?;
        if dto.schema_version != NANO_WHEELS_OFF_QUALIFICATION_LAUNCH_V3 {
            return Err(
                NanoWheelsOffQualificationLaunchParseError::UnsupportedSchema {
                    actual: dto.schema_version,
                    supported: NANO_WHEELS_OFF_QUALIFICATION_LAUNCH_V3,
                },
            );
        }

        let launch = Self {
            robot_id: QualificationRobotId::parse(dto.robot_id)?,
            qualification_executable: parse_candidate_asset(
                NanoWheelsOffQualificationAssetRole::QualificationExecutable,
                dto.qualification_executable_asset.ok_or(
                    NanoWheelsOffQualificationLaunchParseError::MissingRequiredAsset {
                        role: NanoWheelsOffQualificationAssetRole::QualificationExecutable,
                    },
                )?,
            )?,
            native_runtime_manifest: parse_candidate_asset(
                NanoWheelsOffQualificationAssetRole::NativeRuntimeManifest,
                dto.native_runtime_manifest_asset.ok_or(
                    NanoWheelsOffQualificationLaunchParseError::MissingRequiredAsset {
                        role: NanoWheelsOffQualificationAssetRole::NativeRuntimeManifest,
                    },
                )?,
            )?,
            agent_policy: parse_asset(NanoLaunchAssetRole::AgentPolicy, dto.agent_policy_asset)
                .map_err(NanoWheelsOffQualificationLaunchParseError::Common)?,
            navigation_shadow_config: parse_asset(
                NanoLaunchAssetRole::NavigationShadowConfig,
                dto.navigation_shadow_config_asset,
            )
            .map_err(NanoWheelsOffQualificationLaunchParseError::Common)?,
            candidate_inventory_manifest: parse_candidate_asset(
                NanoWheelsOffQualificationAssetRole::CandidateInventoryManifest,
                dto.candidate_inventory_manifest_asset,
            )?,
            candidate_controller_policy: parse_candidate_asset(
                NanoWheelsOffQualificationAssetRole::CandidateControllerPolicy,
                dto.candidate_controller_policy_asset,
            )?,
            controller_server: parse_controller_server(dto.controller_server)
                .map_err(NanoWheelsOffQualificationLaunchParseError::Common)?,
            calibration_artifact: parse_calibration_artifact(dto.calibration_artifact)
                .map_err(NanoWheelsOffQualificationLaunchParseError::Common)?,
            plant_artifact: parse_plant_artifact(dto.plant_artifact)
                .map_err(NanoWheelsOffQualificationLaunchParseError::Common)?,
            oak: parse_oak(dto.oak).map_err(NanoWheelsOffQualificationLaunchParseError::Common)?,
            occupancy: parse_occupancy(dto.occupancy)
                .map_err(NanoWheelsOffQualificationLaunchParseError::Common)?,
            inference: parse_inference(dto.inference)
                .map_err(NanoWheelsOffQualificationLaunchParseError::Common)?,
            rerun: parse_rerun(dto.rerun)
                .map_err(NanoWheelsOffQualificationLaunchParseError::Common)?,
            storage: parse_storage(dto.storage)
                .map_err(NanoWheelsOffQualificationLaunchParseError::Common)?,
        };
        ensure_distinct_assets(&launch)?;
        Ok(launch)
    }

    pub const fn robot_id(&self) -> &QualificationRobotId {
        &self.robot_id
    }

    pub const fn qualification_executable(&self) -> &NanoLaunchAssetBinding {
        &self.qualification_executable
    }

    pub const fn native_runtime_manifest(&self) -> &NanoLaunchAssetBinding {
        &self.native_runtime_manifest
    }

    pub const fn agent_policy(&self) -> &NanoLaunchAssetBinding {
        &self.agent_policy
    }

    pub const fn navigation_shadow_config(&self) -> &NanoLaunchAssetBinding {
        &self.navigation_shadow_config
    }

    pub const fn candidate_inventory_manifest(&self) -> &NanoLaunchAssetBinding {
        &self.candidate_inventory_manifest
    }

    pub const fn candidate_controller_policy(&self) -> &NanoLaunchAssetBinding {
        &self.candidate_controller_policy
    }

    pub const fn controller_server(&self) -> &NanoLaunchControllerServer {
        &self.controller_server
    }

    pub const fn plant_artifact(&self) -> &NanoLaunchPlantArtifact {
        &self.plant_artifact
    }

    pub const fn calibration_artifact(&self) -> &NanoLaunchCalibrationArtifact {
        &self.calibration_artifact
    }

    pub const fn oak(&self) -> &NanoOakStreamGraph {
        &self.oak
    }

    pub const fn occupancy(&self) -> &NanoLaunchOccupancy {
        &self.occupancy
    }

    pub const fn inference(&self) -> &NanoLaunchInference {
        &self.inference
    }

    pub const fn rerun(&self) -> NanoLaunchRerun {
        self.rerun
    }

    pub const fn storage(&self) -> &NanoLaunchStorage {
        &self.storage
    }

    pub fn asset(&self, role: NanoWheelsOffQualificationAssetRole) -> &NanoLaunchAssetBinding {
        match role {
            NanoWheelsOffQualificationAssetRole::QualificationExecutable => {
                &self.qualification_executable
            }
            NanoWheelsOffQualificationAssetRole::NativeRuntimeManifest => {
                &self.native_runtime_manifest
            }
            NanoWheelsOffQualificationAssetRole::AgentPolicy => &self.agent_policy,
            NanoWheelsOffQualificationAssetRole::NavigationShadowConfig => {
                &self.navigation_shadow_config
            }
            NanoWheelsOffQualificationAssetRole::CandidateInventoryManifest => {
                &self.candidate_inventory_manifest
            }
            NanoWheelsOffQualificationAssetRole::CandidateControllerPolicy => {
                &self.candidate_controller_policy
            }
            NanoWheelsOffQualificationAssetRole::ControllerServerContract => {
                self.controller_server.contract_asset()
            }
            NanoWheelsOffQualificationAssetRole::CalibrationArtifact => {
                self.calibration_artifact.asset()
            }
            NanoWheelsOffQualificationAssetRole::PlantArtifact => self.plant_artifact.asset(),
            NanoWheelsOffQualificationAssetRole::OnnxRuntimeLibrary => {
                self.inference.onnx_runtime_library()
            }
            NanoWheelsOffQualificationAssetRole::SuperpointModel => {
                self.inference.superpoint_model()
            }
            NanoWheelsOffQualificationAssetRole::LightglueModel => self.inference.lightglue_model(),
        }
    }
}

#[derive(Debug)]
pub struct LoadedNanoWheelsOffQualificationLaunchV3 {
    launch: NanoWheelsOffQualificationLaunchV3,
    source: LoadedDeploymentAsset,
}

impl LoadedNanoWheelsOffQualificationLaunchV3 {
    pub const fn launch(&self) -> &NanoWheelsOffQualificationLaunchV3 {
        &self.launch
    }

    pub const fn source(&self) -> &LoadedDeploymentAsset {
        &self.source
    }

    pub const fn content_sha256(&self) -> DeploymentAssetContentSha256 {
        self.source.content_sha256()
    }

    pub fn into_parts(self) -> (NanoWheelsOffQualificationLaunchV3, LoadedDeploymentAsset) {
        (self.launch, self.source)
    }
}

pub fn load_nano_wheels_off_qualification_launch_v3(
    deployment_root: &Path,
    launch_relative_path: ArtifactRelativePath,
) -> Result<LoadedNanoWheelsOffQualificationLaunchV3, NanoWheelsOffQualificationLaunchLoadError> {
    let byte_limit = DeploymentAssetByteLimit::try_new(
        u64::try_from(MAX_NANO_WHEELS_OFF_QUALIFICATION_LAUNCH_JSON_BYTES)
            .expect("launch JSON bound fits every supported host"),
    )
    .expect("launch JSON bound is within the deployment-asset domain");
    let source = load_deployment_asset(deployment_root, launch_relative_path, byte_limit)
        .map_err(NanoWheelsOffQualificationLaunchLoadError::Load)?;
    let launch = NanoWheelsOffQualificationLaunchV3::parse_json(source.bytes())
        .map_err(NanoWheelsOffQualificationLaunchLoadError::Parse)?;
    for role in NanoWheelsOffQualificationAssetRole::ALL {
        if launch.asset(role).relative_path() == source.relative_path() {
            return Err(
                NanoWheelsOffQualificationLaunchLoadError::InputAliasesLaunchDocument {
                    role,
                    relative_path: source.relative_path().clone(),
                },
            );
        }
    }
    Ok(LoadedNanoWheelsOffQualificationLaunchV3 { launch, source })
}

#[derive(Debug)]
pub enum NanoWheelsOffQualificationLaunchLoadError {
    Load(DeploymentAssetLoadError),
    Parse(NanoWheelsOffQualificationLaunchParseError),
    InputAliasesLaunchDocument {
        role: NanoWheelsOffQualificationAssetRole,
        relative_path: ArtifactRelativePath,
    },
}

impl fmt::Display for NanoWheelsOffQualificationLaunchLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "wheels-off qualification launch load failed: {self:?}"
        )
    }
}

impl std::error::Error for NanoWheelsOffQualificationLaunchLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Load(source) => Some(source),
            Self::Parse(source) => Some(source),
            Self::InputAliasesLaunchDocument { .. } => None,
        }
    }
}

#[derive(Debug)]
pub enum NanoWheelsOffQualificationLaunchParseError {
    InputTooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    JsonDecode(serde_json::Error),
    JsonTrailingData(serde_json::Error),
    UnsupportedSchema {
        actual: u32,
        supported: u32,
    },
    InvalidRobotId {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    InvalidAssetPath {
        role: NanoWheelsOffQualificationAssetRole,
        source: ArtifactRelativePathError,
    },
    InvalidAssetByteLimit {
        role: NanoWheelsOffQualificationAssetRole,
        source: DeploymentAssetByteLimitError,
    },
    AssetByteLimitAboveRoleMaximum {
        role: NanoWheelsOffQualificationAssetRole,
        actual_bytes: u64,
        maximum_bytes: u64,
    },
    InvalidAssetSha256 {
        role: NanoWheelsOffQualificationAssetRole,
        source: NanoLaunchSha256Error,
    },
    MissingRequiredAsset {
        role: NanoWheelsOffQualificationAssetRole,
    },
    DuplicateInputAssetPath {
        first: NanoWheelsOffQualificationAssetRole,
        second: NanoWheelsOffQualificationAssetRole,
        relative_path: ArtifactRelativePath,
    },
    Common(NanoAgentLaunchParseError),
}

impl fmt::Display for NanoWheelsOffQualificationLaunchParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid wheels-off qualification launch document: {self:?}"
        )
    }
}

impl std::error::Error for NanoWheelsOffQualificationLaunchParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::JsonDecode(source) | Self::JsonTrailingData(source) => Some(source),
            Self::InvalidAssetPath { source, .. } => Some(source),
            Self::InvalidAssetByteLimit { source, .. } => Some(source),
            Self::InvalidAssetSha256 { source, .. } => Some(source),
            Self::Common(source) => Some(source),
            Self::InputTooLarge { .. }
            | Self::UnsupportedSchema { .. }
            | Self::InvalidRobotId { .. }
            | Self::MissingRequiredAsset { .. }
            | Self::AssetByteLimitAboveRoleMaximum { .. }
            | Self::DuplicateInputAssetPath { .. } => None,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoWheelsOffQualificationLaunchV3Dto {
    schema_version: u32,
    robot_id: String,
    qualification_executable_asset: Option<CandidateAssetBindingDto>,
    native_runtime_manifest_asset: Option<CandidateAssetBindingDto>,
    agent_policy_asset: NanoLaunchAssetBindingDto,
    navigation_shadow_config_asset: NanoLaunchAssetBindingDto,
    candidate_inventory_manifest_asset: CandidateAssetBindingDto,
    candidate_controller_policy_asset: CandidateAssetBindingDto,
    controller_server: NanoLaunchControllerServerDto,
    calibration_artifact: NanoLaunchCalibrationArtifactDto,
    plant_artifact: NanoLaunchPlantArtifactDto,
    oak: NanoOakStreamGraphDto,
    occupancy: NanoLaunchOccupancyDto,
    inference: NanoLaunchInferenceDto,
    rerun: NanoLaunchRerunDto,
    storage: NanoLaunchStorageDto,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateAssetBindingDto {
    relative_path: String,
    maximum_bytes: u64,
    sha256_hex: String,
}

fn parse_candidate_asset(
    role: NanoWheelsOffQualificationAssetRole,
    dto: CandidateAssetBindingDto,
) -> Result<NanoLaunchAssetBinding, NanoWheelsOffQualificationLaunchParseError> {
    let relative_path = ArtifactRelativePath::parse(dto.relative_path).map_err(|source| {
        NanoWheelsOffQualificationLaunchParseError::InvalidAssetPath { role, source }
    })?;
    let byte_limit = DeploymentAssetByteLimit::try_new(dto.maximum_bytes).map_err(|source| {
        NanoWheelsOffQualificationLaunchParseError::InvalidAssetByteLimit { role, source }
    })?;
    let maximum_bytes = role.maximum_bytes();
    if byte_limit.get() > maximum_bytes {
        return Err(
            NanoWheelsOffQualificationLaunchParseError::AssetByteLimitAboveRoleMaximum {
                role,
                actual_bytes: byte_limit.get(),
                maximum_bytes,
            },
        );
    }
    let expected_sha256 = parse_sha256(&dto.sha256_hex).map_err(|source| {
        NanoWheelsOffQualificationLaunchParseError::InvalidAssetSha256 { role, source }
    })?;
    Ok(NanoLaunchAssetBinding::from_parsed_parts(
        relative_path,
        byte_limit,
        expected_sha256,
    ))
}

fn ensure_distinct_assets(
    launch: &NanoWheelsOffQualificationLaunchV3,
) -> Result<(), NanoWheelsOffQualificationLaunchParseError> {
    for (first_index, first) in NanoWheelsOffQualificationAssetRole::ALL
        .into_iter()
        .enumerate()
    {
        for second in NanoWheelsOffQualificationAssetRole::ALL
            .into_iter()
            .skip(first_index + 1)
        {
            let first_path = launch.asset(first).relative_path();
            if first_path == launch.asset(second).relative_path() {
                return Err(
                    NanoWheelsOffQualificationLaunchParseError::DuplicateInputAssetPath {
                        first,
                        second,
                        relative_path: first_path.clone(),
                    },
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;
    use crate::navigation::NanoAgentLaunchV3;

    fn asset(path: &str) -> Value {
        json!({
            "relative_path": path,
            "maximum_bytes": 1024,
            "sha256_hex": "11".repeat(32),
        })
    }

    fn launch_document() -> Vec<u8> {
        serde_json::to_vec(&json!({
            "schema_version": 3,
            "robot_id": "kiko-candidate",
            "qualification_executable_asset":
                asset("bin/kiko-nano-wheels-off-qualification"),
            "native_runtime_manifest_asset": asset("native-runtime-v1.json"),
            "agent_policy_asset": asset("agent-policy-v3.json"),
            "navigation_shadow_config_asset": asset("navigation-shadow-v1.json"),
            "candidate_inventory_manifest_asset": asset("device-inventory-candidate-v2.json"),
            "candidate_controller_policy_asset": asset("candidate-controller-policy-v1.json"),
            "controller_server": {
                "contract_asset": asset("controller-server-candidate-v2.json"),
                "command_udp_endpoint": "127.0.0.1:8080"
            },
            "calibration_artifact": {
                "artifact_id": "kiko-calibration-v1",
                "asset": asset("calibration/kiko-calibration-v1.json")
            },
            "plant_artifact": {
                "artifact_id": "candidate-shadow-plant",
                "asset": asset("plant/candidate.json")
            },
            "oak": {
                "selector_source": "exact_inventory_oak_mxid",
                "maximum_usb_speed": "SUPER",
                "minimum_usb_speed": "SUPER",
                "rgb": {"width_px": 640, "height_px": 400, "fps": 10},
                "rectified_stereo": {"width_px": 640, "height_px": 400, "fps": 10, "rectified": true},
                "depth": {"width_px": 640, "height_px": 400, "fps": 10, "alignment": "rectified_left"},
                "imu": {"rate_hz": 400},
                "queue": {"size": 4, "blocking": false}
            },
            "occupancy": {
                "resolution_m": 0.05,
                "lower_x_m": -10.0,
                "lower_y_m": -10.0,
                "width_cells": 400,
                "height_cells": 400,
                "maximum_cells": 160000,
                "maximum_keyframes": 10000,
                "snapshot_every_keyframes": 1
            },
            "inference": {
                "onnx_runtime_library_asset": asset("lib/libonnxruntime.so.1"),
                "superpoint_model_asset": asset("models/superpoint.onnx"),
                "lightglue_model_asset": asset("models/lightglue.onnx"),
                "superpoint_backend": "cpu",
                "lightglue_backend": "cpu",
                "downscale_factor": 1,
                "maximum_keypoints": 512
            },
            "rerun": {
                "kind": "serve_loopback",
                "bind": "127.0.0.1:9876",
                "decimation": 1,
                "memory_limit_bytes": 1048576,
                "flush_timeout_ms": 1000
            },
            "storage": {
                "map_snapshot_relative_path": "maps/current.kmap",
                "navigation_dataset_directory_relative_path": "navigation",
                "maximum_map_snapshot_bytes": 1048576,
                "minimum_free_bytes_after_map_save": 1048576,
                "maximum_navigation_dataset_bytes": 8589934592_u64,
                "maximum_navigation_dataset_files": 65536_u64,
                "maximum_navigation_ingress_records": 100000_u64,
                "minimum_free_bytes_after_navigation_dataset_write": 1073741824_u64,
                "navigation_dataset_terminal_reserve_bytes": 68161536_u64
            }
        }))
        .expect("qualification launch fixture")
    }

    #[test]
    fn qualification_and_production_launch_schemas_are_disjoint() {
        let qualification = launch_document();
        NanoWheelsOffQualificationLaunchV3::parse_json(&qualification)
            .expect("qualification schema");
        assert!(NanoAgentLaunchV3::parse_json(&qualification).is_err());

        let mut wrong: Value =
            serde_json::from_slice(&qualification).expect("qualification fixture");
        wrong["schema_version"] = json!(1);
        wrong
            .as_object_mut()
            .expect("launch fixture")
            .remove("qualification_executable_asset");
        wrong
            .as_object_mut()
            .expect("launch fixture")
            .remove("native_runtime_manifest_asset");
        assert!(matches!(
            NanoWheelsOffQualificationLaunchV3::parse_json(
                &serde_json::to_vec(&wrong).expect("wrong-schema fixture")
            ),
            Err(
                NanoWheelsOffQualificationLaunchParseError::UnsupportedSchema {
                    actual: 1,
                    supported: 3
                }
            )
        ));

        let mut retired_v2: Value =
            serde_json::from_slice(&qualification).expect("qualification fixture");
        retired_v2["schema_version"] = json!(2);
        assert!(matches!(
            NanoWheelsOffQualificationLaunchV3::parse_json(
                &serde_json::to_vec(&retired_v2).expect("retired-schema fixture")
            ),
            Err(
                NanoWheelsOffQualificationLaunchParseError::UnsupportedSchema {
                    actual: 2,
                    supported: 3
                }
            )
        ));

        let mut missing: Value =
            serde_json::from_slice(&qualification).expect("qualification fixture");
        missing
            .as_object_mut()
            .expect("launch fixture")
            .remove("qualification_executable_asset");
        assert!(matches!(
            NanoWheelsOffQualificationLaunchV3::parse_json(
                &serde_json::to_vec(&missing).expect("missing-asset fixture")
            ),
            Err(
                NanoWheelsOffQualificationLaunchParseError::MissingRequiredAsset {
                    role: NanoWheelsOffQualificationAssetRole::QualificationExecutable
                }
            )
        ));
    }

    #[test]
    fn every_input_path_is_unique_and_source_aliasing_is_checked_after_load() {
        let mut duplicate: Value =
            serde_json::from_slice(&launch_document()).expect("qualification fixture");
        duplicate["candidate_controller_policy_asset"] =
            duplicate["candidate_inventory_manifest_asset"].clone();
        assert!(matches!(
            NanoWheelsOffQualificationLaunchV3::parse_json(
                &serde_json::to_vec(&duplicate).expect("duplicate fixture")
            ),
            Err(
                NanoWheelsOffQualificationLaunchParseError::DuplicateInputAssetPath {
                    first: NanoWheelsOffQualificationAssetRole::CandidateInventoryManifest,
                    second: NanoWheelsOffQualificationAssetRole::CandidateControllerPolicy,
                    ..
                }
            )
        ));
    }

    #[test]
    fn weak_robot_identity_and_trailing_json_fail_at_the_boundary() {
        let mut invalid: Value =
            serde_json::from_slice(&launch_document()).expect("qualification fixture");
        invalid["robot_id"] = json!("../../kiko");
        assert!(
            NanoWheelsOffQualificationLaunchV3::parse_json(
                &serde_json::to_vec(&invalid).expect("invalid robot fixture")
            )
            .is_err()
        );

        let mut trailing = launch_document();
        trailing.extend_from_slice(b" {}");
        assert!(matches!(
            NanoWheelsOffQualificationLaunchV3::parse_json(&trailing),
            Err(NanoWheelsOffQualificationLaunchParseError::JsonTrailingData(_))
        ));
    }
}
