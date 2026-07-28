//! Strict launch boundary for the manually invoked wheels-off full-stack
//! qualifier.
//!
//! This schema deliberately has no production physical-actuation document.
//! V3 binds the common OAK/SLAM/occupancy/inference graph to a schema-V2
//! candidate inventory, a candidate-only controller contract, and a
//! candidate-only host policy. V4 adds exact frontal/profile face-cascade
//! bindings and an optional head-gaze-policy binding without reinterpreting
//! V3. Bootstrap separately admits any bound policy only as proposal-only.
//! Production and qualification documents are different Rust types and reject
//! each other's schemas.

use std::fmt;
use std::path::Path;

use kiko_device_inventory::{
    ArtifactRelativePath, ArtifactRelativePathError, DeploymentAssetByteLimit,
    DeploymentAssetByteLimitError, DeploymentAssetContentSha256, DeploymentAssetLoadError,
    LoadedDeploymentAsset, MAX_DEPLOYMENT_ASSET_BYTES, MAX_MANIFEST_JSON_BYTES, MAX_ROBOT_ID_BYTES,
    load_deployment_asset,
};
use serde::{Deserialize, Deserializer};

use super::nano_agent_launch::{
    MAX_OPENCV_HAAR_CASCADE_BYTES, NanoAgentLaunchParseError, NanoFaceCascadeAssetRole,
    NanoLaunchAssetBinding, NanoLaunchAssetBindingDto, NanoLaunchAssetRole,
    NanoLaunchCalibrationArtifact, NanoLaunchCalibrationArtifactDto, NanoLaunchControllerServer,
    NanoLaunchControllerServerDto, NanoLaunchInference, NanoLaunchInferenceDto,
    NanoLaunchOccupancy, NanoLaunchOccupancyDto, NanoLaunchPlantArtifact,
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
pub const NANO_WHEELS_OFF_QUALIFICATION_LAUNCH_V4: u32 = 4;
pub const MAX_NANO_WHEELS_OFF_QUALIFICATION_LAUNCH_JSON_BYTES: usize = 64 * 1_024;
pub const MAX_NANO_HEAD_GAZE_POLICY_JSON_BYTES: u64 = 256 * 1_024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoWheelsOffQualificationV4AssetRole {
    HeadGazePolicy,
}

impl NanoWheelsOffQualificationV4AssetRole {
    pub const ALL: [Self; 1] = [Self::HeadGazePolicy];

    const fn maximum_bytes(self) -> u64 {
        match self {
            Self::HeadGazePolicy => MAX_NANO_HEAD_GAZE_POLICY_JSON_BYTES,
        }
    }
}

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
        Self::from_dto(dto)
    }

    fn from_dto(
        dto: NanoWheelsOffQualificationLaunchV3Dto,
    ) -> Result<Self, NanoWheelsOffQualificationLaunchParseError> {
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

/// Exact face-cascade bindings added by qualification launch V4.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoWheelsOffQualificationFacePerception {
    frontal_face_cascade: NanoLaunchAssetBinding,
    profile_face_cascade: NanoLaunchAssetBinding,
}

impl NanoWheelsOffQualificationFacePerception {
    pub const fn frontal_face_cascade(&self) -> &NanoLaunchAssetBinding {
        &self.frontal_face_cascade
    }

    pub const fn profile_face_cascade(&self) -> &NanoLaunchAssetBinding {
        &self.profile_face_cascade
    }

    pub const fn asset(&self, role: NanoFaceCascadeAssetRole) -> &NanoLaunchAssetBinding {
        match role {
            NanoFaceCascadeAssetRole::FrontalFace => &self.frontal_face_cascade,
            NanoFaceCascadeAssetRole::ProfileFace => &self.profile_face_cascade,
        }
    }
}

/// Qualification V4 retains the complete V3 graph, exact face cascades, and
/// optionally one exact head-gaze-policy binding.
///
/// This is a separate type so a V3 document can never silently gain face
/// semantics. Runtime wiring must explicitly select and load V4.
#[derive(Clone, Debug, PartialEq)]
pub struct NanoWheelsOffQualificationLaunchV4 {
    common: NanoWheelsOffQualificationLaunchV3,
    head_gaze_policy: Option<NanoLaunchAssetBinding>,
    face_perception: NanoWheelsOffQualificationFacePerception,
}

impl NanoWheelsOffQualificationLaunchV4 {
    pub fn parse_json(json: &[u8]) -> Result<Self, NanoWheelsOffQualificationLaunchParseError> {
        if json.len() > MAX_NANO_WHEELS_OFF_QUALIFICATION_LAUNCH_JSON_BYTES {
            return Err(NanoWheelsOffQualificationLaunchParseError::InputTooLarge {
                actual_bytes: json.len(),
                maximum_bytes: MAX_NANO_WHEELS_OFF_QUALIFICATION_LAUNCH_JSON_BYTES,
            });
        }
        let mut deserializer = serde_json::Deserializer::from_slice(json);
        let dto = NanoWheelsOffQualificationLaunchV4Dto::deserialize(&mut deserializer)
            .map_err(NanoWheelsOffQualificationLaunchParseError::JsonDecode)?;
        deserializer
            .end()
            .map_err(NanoWheelsOffQualificationLaunchParseError::JsonTrailingData)?;
        if dto.schema_version != NANO_WHEELS_OFF_QUALIFICATION_LAUNCH_V4 {
            return Err(
                NanoWheelsOffQualificationLaunchParseError::UnsupportedSchema {
                    actual: dto.schema_version,
                    supported: NANO_WHEELS_OFF_QUALIFICATION_LAUNCH_V4,
                },
            );
        }

        let (common_dto, head_gaze_policy_dto, face_perception_dto) = dto.into_parts();
        let face_perception = parse_qualification_face_perception(
            face_perception_dto
                .ok_or(NanoWheelsOffQualificationLaunchParseError::MissingFacePerception)?,
        )?;
        let head_gaze_policy = match head_gaze_policy_dto {
            JsonFieldPresence::Absent => None,
            JsonFieldPresence::Null => {
                return Err(
                    NanoWheelsOffQualificationLaunchParseError::NullV4AssetForbidden {
                        role: NanoWheelsOffQualificationV4AssetRole::HeadGazePolicy,
                    },
                );
            }
            JsonFieldPresence::Value(dto) => Some(parse_v4_asset(
                NanoWheelsOffQualificationV4AssetRole::HeadGazePolicy,
                dto,
            )?),
        };
        let launch = Self {
            common: NanoWheelsOffQualificationLaunchV3::from_dto(common_dto)?,
            head_gaze_policy,
            face_perception,
        };
        ensure_distinct_v4_assets(&launch)?;
        Ok(launch)
    }

    pub const fn robot_id(&self) -> &QualificationRobotId {
        self.common.robot_id()
    }

    /// The complete, already parsed V3-compatible portion retained by V4.
    ///
    /// This accessor does not reinterpret V4 as V3; callers still own the V4
    /// type and must separately consume every V4-only asset before runtime
    /// construction.
    pub const fn common(&self) -> &NanoWheelsOffQualificationLaunchV3 {
        &self.common
    }

    pub const fn qualification_executable(&self) -> &NanoLaunchAssetBinding {
        self.common.qualification_executable()
    }

    pub const fn native_runtime_manifest(&self) -> &NanoLaunchAssetBinding {
        self.common.native_runtime_manifest()
    }

    pub const fn agent_policy(&self) -> &NanoLaunchAssetBinding {
        self.common.agent_policy()
    }

    pub const fn navigation_shadow_config(&self) -> &NanoLaunchAssetBinding {
        self.common.navigation_shadow_config()
    }

    pub const fn candidate_inventory_manifest(&self) -> &NanoLaunchAssetBinding {
        self.common.candidate_inventory_manifest()
    }

    pub const fn candidate_controller_policy(&self) -> &NanoLaunchAssetBinding {
        self.common.candidate_controller_policy()
    }

    pub const fn controller_server(&self) -> &NanoLaunchControllerServer {
        self.common.controller_server()
    }

    pub const fn plant_artifact(&self) -> &NanoLaunchPlantArtifact {
        self.common.plant_artifact()
    }

    pub const fn calibration_artifact(&self) -> &NanoLaunchCalibrationArtifact {
        self.common.calibration_artifact()
    }

    pub const fn oak(&self) -> &NanoOakStreamGraph {
        self.common.oak()
    }

    pub const fn occupancy(&self) -> &NanoLaunchOccupancy {
        self.common.occupancy()
    }

    pub const fn inference(&self) -> &NanoLaunchInference {
        self.common.inference()
    }

    pub const fn rerun(&self) -> NanoLaunchRerun {
        self.common.rerun()
    }

    pub const fn storage(&self) -> &NanoLaunchStorage {
        self.common.storage()
    }

    pub fn asset(&self, role: NanoWheelsOffQualificationAssetRole) -> &NanoLaunchAssetBinding {
        self.common.asset(role)
    }

    pub const fn head_gaze_policy(&self) -> Option<&NanoLaunchAssetBinding> {
        self.head_gaze_policy.as_ref()
    }

    pub const fn v4_asset(
        &self,
        role: NanoWheelsOffQualificationV4AssetRole,
    ) -> Option<&NanoLaunchAssetBinding> {
        match role {
            NanoWheelsOffQualificationV4AssetRole::HeadGazePolicy => self.head_gaze_policy.as_ref(),
        }
    }

    pub const fn face_perception(&self) -> &NanoWheelsOffQualificationFacePerception {
        &self.face_perception
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
pub struct LoadedNanoWheelsOffQualificationLaunchV4 {
    launch: NanoWheelsOffQualificationLaunchV4,
    source: LoadedDeploymentAsset,
}

impl LoadedNanoWheelsOffQualificationLaunchV4 {
    pub const fn launch(&self) -> &NanoWheelsOffQualificationLaunchV4 {
        &self.launch
    }

    pub const fn source(&self) -> &LoadedDeploymentAsset {
        &self.source
    }

    pub const fn content_sha256(&self) -> DeploymentAssetContentSha256 {
        self.source.content_sha256()
    }

    pub fn into_parts(self) -> (NanoWheelsOffQualificationLaunchV4, LoadedDeploymentAsset) {
        (self.launch, self.source)
    }
}

pub fn load_nano_wheels_off_qualification_launch_v4(
    deployment_root: &Path,
    launch_relative_path: ArtifactRelativePath,
) -> Result<LoadedNanoWheelsOffQualificationLaunchV4, NanoWheelsOffQualificationLaunchLoadError> {
    let byte_limit = DeploymentAssetByteLimit::try_new(
        u64::try_from(MAX_NANO_WHEELS_OFF_QUALIFICATION_LAUNCH_JSON_BYTES)
            .expect("launch JSON bound fits every supported host"),
    )
    .expect("launch JSON bound is within the deployment-asset domain");
    let source = load_deployment_asset(deployment_root, launch_relative_path, byte_limit)
        .map_err(NanoWheelsOffQualificationLaunchLoadError::Load)?;
    let launch = NanoWheelsOffQualificationLaunchV4::parse_json(source.bytes())
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
    for role in NanoWheelsOffQualificationV4AssetRole::ALL {
        if launch
            .v4_asset(role)
            .is_some_and(|asset| asset.relative_path() == source.relative_path())
        {
            return Err(
                NanoWheelsOffQualificationLaunchLoadError::V4InputAliasesLaunchDocument {
                    role,
                    relative_path: source.relative_path().clone(),
                },
            );
        }
    }
    for role in NanoFaceCascadeAssetRole::ALL {
        if launch.face_perception().asset(role).relative_path() == source.relative_path() {
            return Err(
                NanoWheelsOffQualificationLaunchLoadError::FaceInputAliasesLaunchDocument {
                    role,
                    relative_path: source.relative_path().clone(),
                },
            );
        }
    }
    Ok(LoadedNanoWheelsOffQualificationLaunchV4 { launch, source })
}

#[derive(Debug)]
pub enum NanoWheelsOffQualificationLaunchLoadError {
    Load(DeploymentAssetLoadError),
    Parse(NanoWheelsOffQualificationLaunchParseError),
    InputAliasesLaunchDocument {
        role: NanoWheelsOffQualificationAssetRole,
        relative_path: ArtifactRelativePath,
    },
    FaceInputAliasesLaunchDocument {
        role: NanoFaceCascadeAssetRole,
        relative_path: ArtifactRelativePath,
    },
    V4InputAliasesLaunchDocument {
        role: NanoWheelsOffQualificationV4AssetRole,
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
            Self::InputAliasesLaunchDocument { .. }
            | Self::FaceInputAliasesLaunchDocument { .. }
            | Self::V4InputAliasesLaunchDocument { .. } => None,
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
    MissingFacePerception,
    InvalidFaceAssetPath {
        role: NanoFaceCascadeAssetRole,
        source: ArtifactRelativePathError,
    },
    InvalidFaceAssetByteLimit {
        role: NanoFaceCascadeAssetRole,
        source: DeploymentAssetByteLimitError,
    },
    FaceAssetByteLimitAboveMaximum {
        role: NanoFaceCascadeAssetRole,
        actual_bytes: u64,
        maximum_bytes: u64,
    },
    InvalidFaceAssetSha256 {
        role: NanoFaceCascadeAssetRole,
        source: NanoLaunchSha256Error,
    },
    DuplicateFaceAssetPath {
        relative_path: ArtifactRelativePath,
    },
    DuplicateFaceAssetContent {
        expected_sha256: [u8; 32],
    },
    FaceAssetAliasesInputAsset {
        face: NanoFaceCascadeAssetRole,
        input: NanoWheelsOffQualificationAssetRole,
        relative_path: ArtifactRelativePath,
    },
    FaceAssetAliasesInputContent {
        face: NanoFaceCascadeAssetRole,
        input: NanoWheelsOffQualificationAssetRole,
        expected_sha256: [u8; 32],
    },
    FaceAssetAliasesV4Content {
        face: NanoFaceCascadeAssetRole,
        input: NanoWheelsOffQualificationV4AssetRole,
        expected_sha256: [u8; 32],
    },
    NullV4AssetForbidden {
        role: NanoWheelsOffQualificationV4AssetRole,
    },
    InvalidV4AssetPath {
        role: NanoWheelsOffQualificationV4AssetRole,
        source: ArtifactRelativePathError,
    },
    InvalidV4AssetByteLimit {
        role: NanoWheelsOffQualificationV4AssetRole,
        source: DeploymentAssetByteLimitError,
    },
    V4AssetByteLimitAboveRoleMaximum {
        role: NanoWheelsOffQualificationV4AssetRole,
        actual_bytes: u64,
        maximum_bytes: u64,
    },
    InvalidV4AssetSha256 {
        role: NanoWheelsOffQualificationV4AssetRole,
        source: NanoLaunchSha256Error,
    },
    V4AssetAliasesInputPath {
        role: NanoWheelsOffQualificationV4AssetRole,
        input: NanoWheelsOffQualificationAssetRole,
        relative_path: ArtifactRelativePath,
    },
    V4AssetAliasesFacePath {
        role: NanoWheelsOffQualificationV4AssetRole,
        face: NanoFaceCascadeAssetRole,
        relative_path: ArtifactRelativePath,
    },
    V4AssetAliasesInputContent {
        role: NanoWheelsOffQualificationV4AssetRole,
        input: NanoWheelsOffQualificationAssetRole,
        expected_sha256: [u8; 32],
    },
    V4AssetAliasesFaceContent {
        role: NanoWheelsOffQualificationV4AssetRole,
        face: NanoFaceCascadeAssetRole,
        expected_sha256: [u8; 32],
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
            Self::InvalidFaceAssetPath { source, .. } => Some(source),
            Self::InvalidFaceAssetByteLimit { source, .. } => Some(source),
            Self::InvalidFaceAssetSha256 { source, .. } => Some(source),
            Self::InvalidV4AssetPath { source, .. } => Some(source),
            Self::InvalidV4AssetByteLimit { source, .. } => Some(source),
            Self::InvalidV4AssetSha256 { source, .. } => Some(source),
            Self::Common(source) => Some(source),
            Self::InputTooLarge { .. }
            | Self::UnsupportedSchema { .. }
            | Self::InvalidRobotId { .. }
            | Self::MissingRequiredAsset { .. }
            | Self::AssetByteLimitAboveRoleMaximum { .. }
            | Self::DuplicateInputAssetPath { .. }
            | Self::MissingFacePerception
            | Self::FaceAssetByteLimitAboveMaximum { .. }
            | Self::DuplicateFaceAssetPath { .. }
            | Self::DuplicateFaceAssetContent { .. }
            | Self::FaceAssetAliasesInputAsset { .. }
            | Self::FaceAssetAliasesInputContent { .. }
            | Self::FaceAssetAliasesV4Content { .. }
            | Self::NullV4AssetForbidden { .. }
            | Self::V4AssetByteLimitAboveRoleMaximum { .. }
            | Self::V4AssetAliasesInputPath { .. }
            | Self::V4AssetAliasesFacePath { .. }
            | Self::V4AssetAliasesInputContent { .. }
            | Self::V4AssetAliasesFaceContent { .. } => None,
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
struct NanoWheelsOffQualificationLaunchV4Dto {
    schema_version: u32,
    robot_id: String,
    qualification_executable_asset: Option<CandidateAssetBindingDto>,
    native_runtime_manifest_asset: Option<CandidateAssetBindingDto>,
    agent_policy_asset: NanoLaunchAssetBindingDto,
    navigation_shadow_config_asset: NanoLaunchAssetBindingDto,
    candidate_inventory_manifest_asset: CandidateAssetBindingDto,
    candidate_controller_policy_asset: CandidateAssetBindingDto,
    #[serde(default)]
    head_gaze_policy_asset: JsonFieldPresence<CandidateAssetBindingDto>,
    controller_server: NanoLaunchControllerServerDto,
    calibration_artifact: NanoLaunchCalibrationArtifactDto,
    plant_artifact: NanoLaunchPlantArtifactDto,
    oak: NanoOakStreamGraphDto,
    occupancy: NanoLaunchOccupancyDto,
    inference: NanoLaunchInferenceDto,
    face_perception: Option<QualificationFacePerceptionDto>,
    rerun: NanoLaunchRerunDto,
    storage: NanoLaunchStorageDto,
}

impl NanoWheelsOffQualificationLaunchV4Dto {
    fn into_parts(
        self,
    ) -> (
        NanoWheelsOffQualificationLaunchV3Dto,
        JsonFieldPresence<CandidateAssetBindingDto>,
        Option<QualificationFacePerceptionDto>,
    ) {
        (
            NanoWheelsOffQualificationLaunchV3Dto {
                schema_version: NANO_WHEELS_OFF_QUALIFICATION_LAUNCH_V3,
                robot_id: self.robot_id,
                qualification_executable_asset: self.qualification_executable_asset,
                native_runtime_manifest_asset: self.native_runtime_manifest_asset,
                agent_policy_asset: self.agent_policy_asset,
                navigation_shadow_config_asset: self.navigation_shadow_config_asset,
                candidate_inventory_manifest_asset: self.candidate_inventory_manifest_asset,
                candidate_controller_policy_asset: self.candidate_controller_policy_asset,
                controller_server: self.controller_server,
                calibration_artifact: self.calibration_artifact,
                plant_artifact: self.plant_artifact,
                oak: self.oak,
                occupancy: self.occupancy,
                inference: self.inference,
                rerun: self.rerun,
                storage: self.storage,
            },
            self.head_gaze_policy_asset,
            self.face_perception,
        )
    }
}

#[derive(Debug, Default)]
enum JsonFieldPresence<T> {
    #[default]
    Absent,
    Null,
    Value(T),
}

impl<'de, T> Deserialize<'de> for JsonFieldPresence<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(match Option::<T>::deserialize(deserializer)? {
            Some(value) => Self::Value(value),
            None => Self::Null,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QualificationFacePerceptionDto {
    frontal_face_cascade_asset: QualificationFaceAssetBindingDto,
    profile_face_cascade_asset: QualificationFaceAssetBindingDto,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QualificationFaceAssetBindingDto {
    relative_path: String,
    maximum_bytes: u64,
    sha256_hex: String,
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

fn parse_v4_asset(
    role: NanoWheelsOffQualificationV4AssetRole,
    dto: CandidateAssetBindingDto,
) -> Result<NanoLaunchAssetBinding, NanoWheelsOffQualificationLaunchParseError> {
    let relative_path = ArtifactRelativePath::parse(dto.relative_path).map_err(|source| {
        NanoWheelsOffQualificationLaunchParseError::InvalidV4AssetPath { role, source }
    })?;
    let byte_limit = DeploymentAssetByteLimit::try_new(dto.maximum_bytes).map_err(|source| {
        NanoWheelsOffQualificationLaunchParseError::InvalidV4AssetByteLimit { role, source }
    })?;
    let maximum_bytes = role.maximum_bytes();
    if byte_limit.get() > maximum_bytes {
        return Err(
            NanoWheelsOffQualificationLaunchParseError::V4AssetByteLimitAboveRoleMaximum {
                role,
                actual_bytes: byte_limit.get(),
                maximum_bytes,
            },
        );
    }
    let expected_sha256 = parse_sha256(&dto.sha256_hex).map_err(|source| {
        NanoWheelsOffQualificationLaunchParseError::InvalidV4AssetSha256 { role, source }
    })?;
    Ok(NanoLaunchAssetBinding::from_parsed_parts(
        relative_path,
        byte_limit,
        expected_sha256,
    ))
}

fn parse_qualification_face_perception(
    dto: QualificationFacePerceptionDto,
) -> Result<NanoWheelsOffQualificationFacePerception, NanoWheelsOffQualificationLaunchParseError> {
    Ok(NanoWheelsOffQualificationFacePerception {
        frontal_face_cascade: parse_qualification_face_asset(
            NanoFaceCascadeAssetRole::FrontalFace,
            dto.frontal_face_cascade_asset,
        )?,
        profile_face_cascade: parse_qualification_face_asset(
            NanoFaceCascadeAssetRole::ProfileFace,
            dto.profile_face_cascade_asset,
        )?,
    })
}

fn parse_qualification_face_asset(
    role: NanoFaceCascadeAssetRole,
    dto: QualificationFaceAssetBindingDto,
) -> Result<NanoLaunchAssetBinding, NanoWheelsOffQualificationLaunchParseError> {
    let relative_path = ArtifactRelativePath::parse(dto.relative_path).map_err(|source| {
        NanoWheelsOffQualificationLaunchParseError::InvalidFaceAssetPath { role, source }
    })?;
    let byte_limit = DeploymentAssetByteLimit::try_new(dto.maximum_bytes).map_err(|source| {
        NanoWheelsOffQualificationLaunchParseError::InvalidFaceAssetByteLimit { role, source }
    })?;
    if byte_limit.get() > MAX_OPENCV_HAAR_CASCADE_BYTES {
        return Err(
            NanoWheelsOffQualificationLaunchParseError::FaceAssetByteLimitAboveMaximum {
                role,
                actual_bytes: byte_limit.get(),
                maximum_bytes: MAX_OPENCV_HAAR_CASCADE_BYTES,
            },
        );
    }
    let expected_sha256 = parse_sha256(&dto.sha256_hex).map_err(|source| {
        NanoWheelsOffQualificationLaunchParseError::InvalidFaceAssetSha256 { role, source }
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

fn ensure_distinct_v4_assets(
    launch: &NanoWheelsOffQualificationLaunchV4,
) -> Result<(), NanoWheelsOffQualificationLaunchParseError> {
    let frontal = launch
        .face_perception()
        .asset(NanoFaceCascadeAssetRole::FrontalFace);
    let profile = launch
        .face_perception()
        .asset(NanoFaceCascadeAssetRole::ProfileFace);
    if frontal.relative_path() == profile.relative_path() {
        return Err(
            NanoWheelsOffQualificationLaunchParseError::DuplicateFaceAssetPath {
                relative_path: frontal.relative_path().clone(),
            },
        );
    }
    if frontal.expected_sha256() == profile.expected_sha256() {
        return Err(
            NanoWheelsOffQualificationLaunchParseError::DuplicateFaceAssetContent {
                expected_sha256: *frontal.expected_sha256(),
            },
        );
    }
    for face in NanoFaceCascadeAssetRole::ALL {
        let face_asset = launch.face_perception().asset(face);
        for input in NanoWheelsOffQualificationAssetRole::ALL {
            let existing = launch.asset(input);
            if face_asset.relative_path() == existing.relative_path() {
                return Err(
                    NanoWheelsOffQualificationLaunchParseError::FaceAssetAliasesInputAsset {
                        face,
                        input,
                        relative_path: face_asset.relative_path().clone(),
                    },
                );
            }
            if face_asset.expected_sha256() == existing.expected_sha256() {
                return Err(
                    NanoWheelsOffQualificationLaunchParseError::FaceAssetAliasesInputContent {
                        face,
                        input,
                        expected_sha256: *face_asset.expected_sha256(),
                    },
                );
            }
        }
        for input in NanoWheelsOffQualificationV4AssetRole::ALL {
            let Some(existing) = launch.v4_asset(input) else {
                continue;
            };
            if face_asset.expected_sha256() == existing.expected_sha256() {
                return Err(
                    NanoWheelsOffQualificationLaunchParseError::FaceAssetAliasesV4Content {
                        face,
                        input,
                        expected_sha256: *face_asset.expected_sha256(),
                    },
                );
            }
        }
    }
    for role in NanoWheelsOffQualificationV4AssetRole::ALL {
        let Some(additional) = launch.v4_asset(role) else {
            continue;
        };
        for input in NanoWheelsOffQualificationAssetRole::ALL {
            let existing = launch.asset(input);
            if additional.relative_path() == existing.relative_path() {
                return Err(
                    NanoWheelsOffQualificationLaunchParseError::V4AssetAliasesInputPath {
                        role,
                        input,
                        relative_path: additional.relative_path().clone(),
                    },
                );
            }
            if additional.expected_sha256() == existing.expected_sha256() {
                return Err(
                    NanoWheelsOffQualificationLaunchParseError::V4AssetAliasesInputContent {
                        role,
                        input,
                        expected_sha256: *additional.expected_sha256(),
                    },
                );
            }
        }
        for face in NanoFaceCascadeAssetRole::ALL {
            let existing = launch.face_perception().asset(face);
            if additional.relative_path() == existing.relative_path() {
                return Err(
                    NanoWheelsOffQualificationLaunchParseError::V4AssetAliasesFacePath {
                        role,
                        face,
                        relative_path: additional.relative_path().clone(),
                    },
                );
            }
            if additional.expected_sha256() == existing.expected_sha256() {
                return Err(
                    NanoWheelsOffQualificationLaunchParseError::V4AssetAliasesFaceContent {
                        role,
                        face,
                        expected_sha256: *additional.expected_sha256(),
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
            "navigation_shadow_config_asset": asset("navigation-shadow-v2.json"),
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

    fn face_asset(path: &str, digest_byte: &str) -> Value {
        json!({
            "relative_path": path,
            "maximum_bytes": 1024,
            "sha256_hex": digest_byte.repeat(32),
        })
    }

    fn launch_v4_document() -> Vec<u8> {
        let mut value: Value =
            serde_json::from_slice(&launch_document()).expect("qualification V3 fixture");
        value["schema_version"] = json!(4);
        value["head_gaze_policy_asset"] = face_asset("head-gaze-policy-v1.json", "44");
        value["face_perception"] = json!({
            "frontal_face_cascade_asset":
                face_asset("models/opencv/haarcascade_frontalface_default.xml", "22"),
            "profile_face_cascade_asset":
                face_asset("models/opencv/haarcascade_profileface.xml", "33"),
        });
        serde_json::to_vec(&value).expect("qualification V4 launch fixture")
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

    #[test]
    fn v4_adds_exact_face_assets_without_reinterpreting_v3() {
        let v4 = launch_v4_document();
        let launch = NanoWheelsOffQualificationLaunchV4::parse_json(&v4).expect("qualification V4");
        assert_eq!(
            launch
                .face_perception()
                .frontal_face_cascade()
                .relative_path()
                .as_str(),
            "models/opencv/haarcascade_frontalface_default.xml"
        );
        assert_eq!(
            launch
                .face_perception()
                .profile_face_cascade()
                .relative_path()
                .as_str(),
            "models/opencv/haarcascade_profileface.xml"
        );
        assert_eq!(
            launch
                .head_gaze_policy()
                .expect("fixture head-gaze policy")
                .relative_path()
                .as_str(),
            "head-gaze-policy-v1.json"
        );
        assert!(
            NanoWheelsOffQualificationLaunchV3::parse_json(&v4).is_err(),
            "V3 must not silently adopt V4 face semantics"
        );
        assert!(
            NanoAgentLaunchV3::parse_json(&v4).is_err(),
            "production launch remains a disjoint type"
        );

        for retired in [1_u32, 2, 3] {
            let mut value: Value = serde_json::from_slice(&v4).expect("qualification V4 fixture");
            value["schema_version"] = json!(retired);
            assert!(matches!(
                NanoWheelsOffQualificationLaunchV4::parse_json(
                    &serde_json::to_vec(&value).expect("retired version fixture")
                ),
                Err(
                    NanoWheelsOffQualificationLaunchParseError::UnsupportedSchema {
                        actual,
                        supported: 4
                    }
                ) if actual == retired
            ));
        }
    }

    #[test]
    fn v4_requires_distinct_bounded_face_assets_that_alias_nothing() {
        let original: Value =
            serde_json::from_slice(&launch_v4_document()).expect("qualification V4 fixture");

        let mut missing = original.clone();
        missing
            .as_object_mut()
            .expect("V4 launch object")
            .remove("face_perception");
        assert!(matches!(
            NanoWheelsOffQualificationLaunchV4::parse_json(
                &serde_json::to_vec(&missing).expect("missing face fixture")
            ),
            Err(NanoWheelsOffQualificationLaunchParseError::MissingFacePerception)
        ));

        let mut same_path = original.clone();
        same_path["face_perception"]["profile_face_cascade_asset"]["relative_path"] =
            same_path["face_perception"]["frontal_face_cascade_asset"]["relative_path"].clone();
        assert!(matches!(
            NanoWheelsOffQualificationLaunchV4::parse_json(
                &serde_json::to_vec(&same_path).expect("same face path fixture")
            ),
            Err(NanoWheelsOffQualificationLaunchParseError::DuplicateFaceAssetPath { .. })
        ));

        let mut same_content = original.clone();
        same_content["face_perception"]["profile_face_cascade_asset"]["sha256_hex"] =
            same_content["face_perception"]["frontal_face_cascade_asset"]["sha256_hex"].clone();
        assert!(matches!(
            NanoWheelsOffQualificationLaunchV4::parse_json(
                &serde_json::to_vec(&same_content).expect("same face content fixture")
            ),
            Err(NanoWheelsOffQualificationLaunchParseError::DuplicateFaceAssetContent { .. })
        ));

        let mut aliases_common_content = original.clone();
        aliases_common_content["face_perception"]["frontal_face_cascade_asset"]["sha256_hex"] =
            aliases_common_content["agent_policy_asset"]["sha256_hex"].clone();
        assert!(matches!(
            NanoWheelsOffQualificationLaunchV4::parse_json(
                &serde_json::to_vec(&aliases_common_content)
                    .expect("face/common content alias fixture")
            ),
            Err(
                NanoWheelsOffQualificationLaunchParseError::FaceAssetAliasesInputContent {
                    face: NanoFaceCascadeAssetRole::FrontalFace,
                    input: NanoWheelsOffQualificationAssetRole::QualificationExecutable,
                    ..
                }
            )
        ));

        let mut aliases_head_content = original.clone();
        aliases_head_content["face_perception"]["profile_face_cascade_asset"]["sha256_hex"] =
            aliases_head_content["head_gaze_policy_asset"]["sha256_hex"].clone();
        assert!(matches!(
            NanoWheelsOffQualificationLaunchV4::parse_json(
                &serde_json::to_vec(&aliases_head_content)
                    .expect("face/head content alias fixture")
            ),
            Err(
                NanoWheelsOffQualificationLaunchParseError::FaceAssetAliasesV4Content {
                    face: NanoFaceCascadeAssetRole::ProfileFace,
                    input: NanoWheelsOffQualificationV4AssetRole::HeadGazePolicy,
                    ..
                }
            )
        ));

        let mut aliases_input = original.clone();
        aliases_input["face_perception"]["frontal_face_cascade_asset"]["relative_path"] =
            aliases_input["agent_policy_asset"]["relative_path"].clone();
        assert!(matches!(
            NanoWheelsOffQualificationLaunchV4::parse_json(
                &serde_json::to_vec(&aliases_input).expect("aliased face fixture")
            ),
            Err(
                NanoWheelsOffQualificationLaunchParseError::FaceAssetAliasesInputAsset {
                    face: NanoFaceCascadeAssetRole::FrontalFace,
                    input: NanoWheelsOffQualificationAssetRole::AgentPolicy,
                    ..
                }
            )
        ));

        let mut oversized = original;
        oversized["face_perception"]["profile_face_cascade_asset"]["maximum_bytes"] =
            json!(MAX_OPENCV_HAAR_CASCADE_BYTES + 1);
        assert!(matches!(
            NanoWheelsOffQualificationLaunchV4::parse_json(
                &serde_json::to_vec(&oversized).expect("oversized face fixture")
            ),
            Err(
                NanoWheelsOffQualificationLaunchParseError::FaceAssetByteLimitAboveMaximum {
                    role: NanoFaceCascadeAssetRole::ProfileFace,
                    ..
                }
            )
        ));
    }

    #[test]
    fn v4_allows_absent_head_gaze_and_bounds_any_present_policy() {
        let original: Value =
            serde_json::from_slice(&launch_v4_document()).expect("qualification V4 fixture");

        let mut missing = original.clone();
        missing
            .as_object_mut()
            .expect("V4 launch object")
            .remove("head_gaze_policy_asset");
        let without_head_gaze = NanoWheelsOffQualificationLaunchV4::parse_json(
            &serde_json::to_vec(&missing).expect("missing head-gaze fixture"),
        )
        .expect("head gaze is optional for Gate A");
        assert!(without_head_gaze.head_gaze_policy().is_none());

        let mut null = original.clone();
        null["head_gaze_policy_asset"] = Value::Null;
        assert!(matches!(
            NanoWheelsOffQualificationLaunchV4::parse_json(
                &serde_json::to_vec(&null).expect("null head-gaze fixture")
            ),
            Err(
                NanoWheelsOffQualificationLaunchParseError::NullV4AssetForbidden {
                    role: NanoWheelsOffQualificationV4AssetRole::HeadGazePolicy
                }
            )
        ));

        let mut aliases_path = original.clone();
        aliases_path["head_gaze_policy_asset"]["relative_path"] =
            aliases_path["agent_policy_asset"]["relative_path"].clone();
        assert!(matches!(
            NanoWheelsOffQualificationLaunchV4::parse_json(
                &serde_json::to_vec(&aliases_path).expect("head-gaze path alias fixture")
            ),
            Err(
                NanoWheelsOffQualificationLaunchParseError::V4AssetAliasesInputPath {
                    role: NanoWheelsOffQualificationV4AssetRole::HeadGazePolicy,
                    input: NanoWheelsOffQualificationAssetRole::AgentPolicy,
                    ..
                }
            )
        ));

        let mut aliases_content = original.clone();
        aliases_content["head_gaze_policy_asset"]["sha256_hex"] =
            aliases_content["face_perception"]["frontal_face_cascade_asset"]["sha256_hex"].clone();
        assert!(matches!(
            NanoWheelsOffQualificationLaunchV4::parse_json(
                &serde_json::to_vec(&aliases_content).expect("head-gaze content alias fixture")
            ),
            Err(
                NanoWheelsOffQualificationLaunchParseError::FaceAssetAliasesV4Content {
                    face: NanoFaceCascadeAssetRole::FrontalFace,
                    input: NanoWheelsOffQualificationV4AssetRole::HeadGazePolicy,
                    ..
                }
            )
        ));

        let mut oversized = original;
        oversized["head_gaze_policy_asset"]["maximum_bytes"] =
            json!(MAX_NANO_HEAD_GAZE_POLICY_JSON_BYTES + 1);
        assert!(matches!(
            NanoWheelsOffQualificationLaunchV4::parse_json(
                &serde_json::to_vec(&oversized).expect("oversized head-gaze fixture")
            ),
            Err(
                NanoWheelsOffQualificationLaunchParseError::V4AssetByteLimitAboveRoleMaximum {
                    role: NanoWheelsOffQualificationV4AssetRole::HeadGazePolicy,
                    ..
                }
            )
        ));
    }
}
