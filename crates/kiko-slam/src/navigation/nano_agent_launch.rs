//! Strict deployment document for the production Nano agent.
//!
//! This is a structural, content-binding boundary. It parses one bounded JSON
//! document into canonical deployment-relative paths, exact SHA-256 identities,
//! finite resource bounds, and a single complete OAK stream graph. Parsing or
//! loading this document does not prove publisher authenticity, filesystem
//! ownership, device presence, USB speed, model compatibility, controller
//! liveness, calibration quality, plant validity, or safe physical motion.

use std::fmt;
use std::net::SocketAddr;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::Path;

use kiko_device_inventory::{
    ArtifactRelativePath, ArtifactRelativePathError, DeploymentAssetByteLimit,
    DeploymentAssetByteLimitError, DeploymentAssetContentSha256, DeploymentAssetLoadError,
    LoadedDeploymentAsset, MAX_DEPLOYMENT_ASSET_BYTES, load_deployment_asset,
};
use oak_sys::{
    DepthAlignment, DepthConfig, DeviceConfig, DeviceConfigError, ImuConfig, MonoConfig,
    QueueConfig, RgbConfig, UsbTransportPolicy,
};
use serde::Deserialize;

use super::{
    MAX_NANO_AGENT_POLICY_CONFIG_JSON_BYTES, MAX_NAVIGATION_ACTUATION_CONFIG_JSON_BYTES,
    MAX_SHADOW_NAVIGATION_CONFIG_JSON_BYTES,
};
use crate::InferenceBackend;
use crate::dense::occupancy::OccupancyGridGeometry;
use crate::dense::occupancy_runtime::OccupancySnapshotCadence;
use crate::live_runtime::{LiveOccupancyHostPolicy, LiveOccupancyHostPolicyError};

/// The only supported production Nano launch-document schema.
pub const NANO_AGENT_LAUNCH_V1: u32 = 1;

/// Hard bound applied before JSON decoding can allocate caller-sized values.
pub const MAX_NANO_AGENT_LAUNCH_JSON_BYTES: usize = 64 * 1_024;

/// The robot-server parser's current hard JSON limit.
///
/// This module does not depend on the optional `robot-server` crate. Production
/// integration must also pass the retained bytes through
/// `ControllerServerConfigV1::parse_json`; this local ceiling only prevents a
/// larger allocation before that authoritative parser runs.
pub const MAX_CONTROLLER_SERVER_CONTRACT_JSON_BYTES: u64 = 8 * 1_024;

pub const MAX_OAK_IMAGE_WIDTH_PX: u32 = 4_096;
pub const MAX_OAK_IMAGE_HEIGHT_PX: u32 = 3_072;
pub const MAX_OAK_FRAME_RATE_HZ: u32 = 240;
pub const MAX_OAK_IMU_RATE_HZ: u32 = 2_000;
pub const MAX_OAK_QUEUE_SIZE: u32 = 64;
pub const MAX_INFERENCE_DOWNSCALE_FACTOR: u32 = 16;
pub const MAX_INFERENCE_KEYPOINTS: u32 = 65_535;
pub const MAX_RERUN_DECIMATION: u32 = 10_000;
pub const MAX_RERUN_MEMORY_BYTES: u64 = 4 * 1_024 * 1_024 * 1_024;
pub const MAX_RERUN_FLUSH_TIMEOUT_MS: u64 = 120_000;
pub const MAX_NANO_STATE_BYTES: u64 = 1_099_511_627_776;
pub const MIN_NANO_OCCUPANCY_RESOLUTION_M: f64 = 0.001;
pub const MAX_NANO_OCCUPANCY_RESOLUTION_M: f64 = 10.0;
pub const MAX_NANO_OCCUPANCY_ABS_LOWER_BOUND_M: f64 = 100_000.0;
pub const MAX_NANO_OCCUPANCY_AXIS_CELLS: u32 = 100_000;
pub const MAX_NANO_OCCUPANCY_CELLS: usize = 16_000_000;
pub const MAX_NANO_OCCUPANCY_KEYFRAMES: usize = 1_000_000;
pub const MAX_NANO_OCCUPANCY_SNAPSHOT_CADENCE: usize = 1_000_000;

const SHA256_HEX_BYTES: usize = 64;
const MAX_PLANT_ARTIFACT_ID_BYTES: usize = 64;
const MIN_RERUN_MEMORY_BYTES: u64 = 1_048_576;

/// Roles are retained in errors so a rejected weak field is unambiguous.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoLaunchAssetRole {
    AgentPolicy,
    NavigationShadowConfig,
    PhysicalActuationConfig,
    ControllerServerContract,
    PlantArtifact,
    OnnxRuntimeLibrary,
    SuperpointModel,
    LightglueModel,
}

impl NanoLaunchAssetRole {
    const ALL: [Self; 8] = [
        Self::AgentPolicy,
        Self::NavigationShadowConfig,
        Self::PhysicalActuationConfig,
        Self::ControllerServerContract,
        Self::PlantArtifact,
        Self::OnnxRuntimeLibrary,
        Self::SuperpointModel,
        Self::LightglueModel,
    ];

    const fn maximum_bytes(self) -> u64 {
        match self {
            Self::AgentPolicy => MAX_NANO_AGENT_POLICY_CONFIG_JSON_BYTES as u64,
            Self::NavigationShadowConfig => MAX_SHADOW_NAVIGATION_CONFIG_JSON_BYTES as u64,
            Self::PhysicalActuationConfig => MAX_NAVIGATION_ACTUATION_CONFIG_JSON_BYTES as u64,
            Self::ControllerServerContract => MAX_CONTROLLER_SERVER_CONTRACT_JSON_BYTES,
            Self::PlantArtifact
            | Self::OnnxRuntimeLibrary
            | Self::SuperpointModel
            | Self::LightglueModel => MAX_DEPLOYMENT_ASSET_BYTES,
        }
    }
}

/// Canonical relative path, bounded load size, and exact expected file bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoLaunchAssetBinding {
    relative_path: ArtifactRelativePath,
    byte_limit: DeploymentAssetByteLimit,
    expected_sha256: [u8; 32],
}

impl NanoLaunchAssetBinding {
    pub const fn relative_path(&self) -> &ArtifactRelativePath {
        &self.relative_path
    }

    pub const fn byte_limit(&self) -> DeploymentAssetByteLimit {
        self.byte_limit
    }

    pub const fn expected_sha256(&self) -> &[u8; 32] {
        &self.expected_sha256
    }

    /// Open one exact asset beneath `deployment_root`, without following any
    /// path component, retain its bytes, and compare its content identity.
    ///
    /// This establishes equality with this launch document only. It does not
    /// authenticate who supplied either file.
    pub fn load_exact(
        &self,
        deployment_root: &Path,
    ) -> Result<LoadedDeploymentAsset, NanoLaunchBoundAssetLoadError> {
        let loaded =
            load_deployment_asset(deployment_root, self.relative_path.clone(), self.byte_limit)
                .map_err(NanoLaunchBoundAssetLoadError::Load)?;
        let observed = *loaded.content_sha256().as_bytes();
        if observed != self.expected_sha256 {
            return Err(NanoLaunchBoundAssetLoadError::ContentMismatch {
                relative_path: self.relative_path.clone(),
                expected_sha256: self.expected_sha256,
                observed_sha256: observed,
            });
        }
        Ok(loaded)
    }
}

#[derive(Debug)]
pub enum NanoLaunchBoundAssetLoadError {
    Load(DeploymentAssetLoadError),
    ContentMismatch {
        relative_path: ArtifactRelativePath,
        expected_sha256: [u8; 32],
        observed_sha256: [u8; 32],
    },
}

impl fmt::Display for NanoLaunchBoundAssetLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load(source) => write!(formatter, "deployment asset load failed: {source}"),
            Self::ContentMismatch { relative_path, .. } => write!(
                formatter,
                "deployment asset {} does not match the launch-document SHA-256",
                relative_path.as_str()
            ),
        }
    }
}

impl std::error::Error for NanoLaunchBoundAssetLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Load(source) => Some(source),
            Self::ContentMismatch { .. } => None,
        }
    }
}

/// Bounded plant artifact identifier retained for a later exact manifest
/// lookup. It is a launch claim until production admission finds the same ID
/// and content digest in the admitted device manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoLaunchPlantArtifactId(String);

impl NanoLaunchPlantArtifactId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoLaunchPlantArtifact {
    artifact_id: NanoLaunchPlantArtifactId,
    asset: NanoLaunchAssetBinding,
}

impl NanoLaunchPlantArtifact {
    pub const fn artifact_id(&self) -> &NanoLaunchPlantArtifactId {
        &self.artifact_id
    }

    pub const fn asset(&self) -> &NanoLaunchAssetBinding {
        &self.asset
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoLaunchControllerServer {
    contract_asset: NanoLaunchAssetBinding,
    command_udp_endpoint: SocketAddr,
}

impl NanoLaunchControllerServer {
    pub const fn contract_asset(&self) -> &NanoLaunchAssetBinding {
        &self.contract_asset
    }

    pub const fn command_udp_endpoint(&self) -> SocketAddr {
        self.command_udp_endpoint
    }
}

/// The OAK selector must come from the exact admitted inventory; there is no
/// first-device or environment fallback in this launch schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoOakSelectorSource {
    ExactInventoryOakMxid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NanoOakImageStream {
    width_px: NonZeroU32,
    height_px: NonZeroU32,
    fps: NonZeroU32,
}

impl NanoOakImageStream {
    pub const fn width_px(self) -> u32 {
        self.width_px.get()
    }

    pub const fn height_px(self) -> u32 {
        self.height_px.get()
    }

    pub const fn fps(self) -> u32 {
        self.fps.get()
    }
}

/// One mandatory production OAK graph. Invalid or incomplete stream sets
/// cannot be represented.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoOakStreamGraph {
    selector_source: NanoOakSelectorSource,
    rgb: NanoOakImageStream,
    rectified_stereo: NanoOakImageStream,
    depth: NanoOakImageStream,
    imu_rate_hz: NonZeroU32,
    queue_size: NonZeroU32,
}

impl NanoOakStreamGraph {
    pub const fn selector_source(&self) -> NanoOakSelectorSource {
        self.selector_source
    }

    pub const fn rgb(&self) -> NanoOakImageStream {
        self.rgb
    }

    pub const fn rectified_stereo(&self) -> NanoOakImageStream {
        self.rectified_stereo
    }

    pub const fn depth(&self) -> NanoOakImageStream {
        self.depth
    }

    pub const fn imu_rate_hz(&self) -> u32 {
        self.imu_rate_hz.get()
    }

    pub const fn queue_size(&self) -> u32 {
        self.queue_size.get()
    }

    /// Convert the already-parsed graph to the OAK bridge's runtime type.
    ///
    /// Connection must still read back and admit the observed USB speed and
    /// exact connected MXID.
    pub fn device_config(&self) -> DeviceConfig {
        DeviceConfig {
            usb_transport: UsbTransportPolicy::super_speed_required(),
            rgb: Some(RgbConfig {
                width: self.rgb.width_px(),
                height: self.rgb.height_px(),
                fps: self.rgb.fps(),
            }),
            mono: Some(MonoConfig {
                width: self.rectified_stereo.width_px(),
                height: self.rectified_stereo.height_px(),
                fps: self.rectified_stereo.fps(),
                rectified: true,
            }),
            depth: Some(DepthConfig {
                width: self.depth.width_px(),
                height: self.depth.height_px(),
                fps: self.depth.fps(),
                alignment: DepthAlignment::RectifiedLeft,
            }),
            imu: Some(ImuConfig {
                rate_hz: self.imu_rate_hz(),
            }),
            queue: QueueConfig {
                size: self.queue_size(),
                blocking: false,
            },
        }
    }
}

/// Global occupancy resource policy selected by the launch document.
///
/// This deliberately does not duplicate projection or obstacle semantics. The
/// exact parsed [`crate::navigation::ShadowNavigationConfigV1`] owns
/// `world_to_occupancy` (including the declared level
/// optical-world/camera-height transform), the runtime depth
/// camera/intrinsics, height and depth ranges, and sampling block. The fixed
/// integer evidence model remains executable code. This launch component owns
/// only global extent, retained-evidence capacity, and publication cadence.
#[derive(Clone, Debug, PartialEq)]
pub struct NanoLaunchOccupancy {
    host_policy: LiveOccupancyHostPolicy,
}

impl NanoLaunchOccupancy {
    pub const fn geometry(&self) -> OccupancyGridGeometry {
        self.host_policy.geometry()
    }

    pub const fn maximum_keyframes(&self) -> usize {
        self.host_policy.maximum_keyframes().get()
    }

    pub const fn snapshot_cadence(&self) -> OccupancySnapshotCadence {
        self.host_policy.snapshot_cadence()
    }

    /// Domain policy consumed directly by the production live-runtime
    /// preparation boundary after the shadow document has been parsed with the
    /// admitted runtime depth camera.
    pub const fn host_policy(&self) -> LiveOccupancyHostPolicy {
        self.host_policy
    }
}

/// Requested inference provider. This is a configuration choice, not evidence
/// that the provider is installed, compatible, faster, or selected at runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoInferenceBackend {
    Auto,
    Cpu,
    Cuda,
    TensorRt,
}

impl NanoInferenceBackend {
    pub const fn runtime(self) -> InferenceBackend {
        match self {
            Self::Auto => InferenceBackend::Auto,
            Self::Cpu => InferenceBackend::Cpu,
            Self::Cuda => InferenceBackend::Cuda,
            Self::TensorRt => InferenceBackend::TensorRT,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoLaunchInference {
    onnx_runtime_library: NanoLaunchAssetBinding,
    superpoint_model: NanoLaunchAssetBinding,
    lightglue_model: NanoLaunchAssetBinding,
    superpoint_backend: NanoInferenceBackend,
    lightglue_backend: NanoInferenceBackend,
    downscale_factor: NonZeroU32,
    maximum_keypoints: NonZeroU32,
}

impl NanoLaunchInference {
    pub const fn onnx_runtime_library(&self) -> &NanoLaunchAssetBinding {
        &self.onnx_runtime_library
    }

    pub const fn superpoint_model(&self) -> &NanoLaunchAssetBinding {
        &self.superpoint_model
    }

    pub const fn lightglue_model(&self) -> &NanoLaunchAssetBinding {
        &self.lightglue_model
    }

    pub const fn superpoint_backend(&self) -> NanoInferenceBackend {
        self.superpoint_backend
    }

    pub const fn lightglue_backend(&self) -> NanoInferenceBackend {
        self.lightglue_backend
    }

    pub const fn downscale_factor(&self) -> u32 {
        self.downscale_factor.get()
    }

    pub const fn maximum_keypoints(&self) -> u32 {
        self.maximum_keypoints.get()
    }
}

/// Loopback-only diagnostic Rerun output. It is never an authority or control
/// input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NanoLaunchRerun {
    bind: SocketAddr,
    decimation: NonZeroU32,
    memory_limit_bytes: NonZeroU64,
    flush_timeout_ms: NonZeroU64,
}

impl NanoLaunchRerun {
    pub const fn bind(self) -> SocketAddr {
        self.bind
    }

    pub const fn decimation(self) -> u32 {
        self.decimation.get()
    }

    pub const fn memory_limit_bytes(self) -> u64 {
        self.memory_limit_bytes.get()
    }

    pub const fn flush_timeout_ms(self) -> u64 {
        self.flush_timeout_ms.get()
    }
}

/// Output paths are canonical and relative to the state root supplied by the
/// service, never to the deployment root or current working directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoLaunchStorage {
    map_snapshot: ArtifactRelativePath,
    navigation_records: ArtifactRelativePath,
    startup_evidence: ArtifactRelativePath,
    maximum_map_bytes: NonZeroU64,
    maximum_navigation_record_bytes: NonZeroU64,
    maximum_startup_evidence_bytes: NonZeroU64,
    maximum_total_state_bytes: NonZeroU64,
    minimum_free_bytes: NonZeroU64,
}

impl NanoLaunchStorage {
    pub const fn map_snapshot(&self) -> &ArtifactRelativePath {
        &self.map_snapshot
    }

    pub const fn navigation_records(&self) -> &ArtifactRelativePath {
        &self.navigation_records
    }

    pub const fn startup_evidence(&self) -> &ArtifactRelativePath {
        &self.startup_evidence
    }

    pub const fn maximum_map_bytes(&self) -> u64 {
        self.maximum_map_bytes.get()
    }

    pub const fn maximum_navigation_record_bytes(&self) -> u64 {
        self.maximum_navigation_record_bytes.get()
    }

    pub const fn maximum_startup_evidence_bytes(&self) -> u64 {
        self.maximum_startup_evidence_bytes.get()
    }

    pub const fn maximum_total_state_bytes(&self) -> u64 {
        self.maximum_total_state_bytes.get()
    }

    pub const fn minimum_free_bytes(&self) -> u64 {
        self.minimum_free_bytes.get()
    }
}

/// Fully parsed production launch document.
#[derive(Clone, Debug, PartialEq)]
pub struct NanoAgentLaunchV1 {
    agent_policy: NanoLaunchAssetBinding,
    navigation_shadow_config: NanoLaunchAssetBinding,
    physical_actuation_config: NanoLaunchAssetBinding,
    controller_server: NanoLaunchControllerServer,
    plant_artifact: NanoLaunchPlantArtifact,
    oak: NanoOakStreamGraph,
    occupancy: NanoLaunchOccupancy,
    inference: NanoLaunchInference,
    rerun: NanoLaunchRerun,
    storage: NanoLaunchStorage,
}

impl NanoAgentLaunchV1 {
    /// Parse one exact bounded JSON byte sequence exactly once.
    pub fn parse_json(json: &[u8]) -> Result<Self, NanoAgentLaunchParseError> {
        if json.len() > MAX_NANO_AGENT_LAUNCH_JSON_BYTES {
            return Err(NanoAgentLaunchParseError::InputTooLarge {
                actual_bytes: json.len(),
                maximum_bytes: MAX_NANO_AGENT_LAUNCH_JSON_BYTES,
            });
        }
        let mut deserializer = serde_json::Deserializer::from_slice(json);
        let dto = NanoAgentLaunchV1Dto::deserialize(&mut deserializer)
            .map_err(NanoAgentLaunchParseError::JsonDecode)?;
        deserializer
            .end()
            .map_err(NanoAgentLaunchParseError::JsonTrailingData)?;
        if dto.schema_version != NANO_AGENT_LAUNCH_V1 {
            return Err(NanoAgentLaunchParseError::UnsupportedSchema {
                actual: dto.schema_version,
                supported: NANO_AGENT_LAUNCH_V1,
            });
        }

        let agent_policy = parse_asset(NanoLaunchAssetRole::AgentPolicy, dto.agent_policy_asset)?;
        let navigation_shadow_config = parse_asset(
            NanoLaunchAssetRole::NavigationShadowConfig,
            dto.navigation_shadow_config_asset,
        )?;
        let physical_actuation_config = parse_asset(
            NanoLaunchAssetRole::PhysicalActuationConfig,
            dto.physical_actuation_config_asset,
        )?;
        let controller_server = parse_controller_server(dto.controller_server)?;
        let plant_artifact = parse_plant_artifact(dto.plant_artifact)?;
        let oak = parse_oak(dto.oak)?;
        let occupancy = parse_occupancy(dto.occupancy)?;
        let inference = parse_inference(dto.inference)?;
        let rerun = parse_rerun(dto.rerun)?;
        let storage = parse_storage(dto.storage)?;

        let launch = Self {
            agent_policy,
            navigation_shadow_config,
            physical_actuation_config,
            controller_server,
            plant_artifact,
            oak,
            occupancy,
            inference,
            rerun,
            storage,
        };
        ensure_distinct_input_assets(&launch)?;
        Ok(launch)
    }

    pub const fn agent_policy(&self) -> &NanoLaunchAssetBinding {
        &self.agent_policy
    }

    pub const fn navigation_shadow_config(&self) -> &NanoLaunchAssetBinding {
        &self.navigation_shadow_config
    }

    pub const fn physical_actuation_config(&self) -> &NanoLaunchAssetBinding {
        &self.physical_actuation_config
    }

    pub const fn controller_server(&self) -> &NanoLaunchControllerServer {
        &self.controller_server
    }

    pub const fn plant_artifact(&self) -> &NanoLaunchPlantArtifact {
        &self.plant_artifact
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

    pub fn asset(&self, role: NanoLaunchAssetRole) -> &NanoLaunchAssetBinding {
        match role {
            NanoLaunchAssetRole::AgentPolicy => &self.agent_policy,
            NanoLaunchAssetRole::NavigationShadowConfig => &self.navigation_shadow_config,
            NanoLaunchAssetRole::PhysicalActuationConfig => &self.physical_actuation_config,
            NanoLaunchAssetRole::ControllerServerContract => &self.controller_server.contract_asset,
            NanoLaunchAssetRole::PlantArtifact => &self.plant_artifact.asset,
            NanoLaunchAssetRole::OnnxRuntimeLibrary => &self.inference.onnx_runtime_library,
            NanoLaunchAssetRole::SuperpointModel => &self.inference.superpoint_model,
            NanoLaunchAssetRole::LightglueModel => &self.inference.lightglue_model,
        }
    }
}

/// Loaded launch document retaining the exact no-follow source bytes and
/// digest used by the parser.
#[derive(Debug)]
pub struct LoadedNanoAgentLaunchV1 {
    launch: NanoAgentLaunchV1,
    source: LoadedDeploymentAsset,
}

impl LoadedNanoAgentLaunchV1 {
    pub const fn launch(&self) -> &NanoAgentLaunchV1 {
        &self.launch
    }

    pub const fn source(&self) -> &LoadedDeploymentAsset {
        &self.source
    }

    pub const fn content_sha256(&self) -> DeploymentAssetContentSha256 {
        self.source.content_sha256()
    }

    pub fn into_parts(self) -> (NanoAgentLaunchV1, LoadedDeploymentAsset) {
        (self.launch, self.source)
    }
}

/// Load the launch document beneath one canonical absolute deployment root.
///
/// Every path component is opened without following symlinks. The retained
/// content digest identifies the exact bytes parsed, but does not authenticate
/// the root or its publisher.
pub fn load_nano_agent_launch_v1(
    deployment_root: &Path,
    launch_relative_path: ArtifactRelativePath,
) -> Result<LoadedNanoAgentLaunchV1, NanoAgentLaunchLoadError> {
    let byte_limit = DeploymentAssetByteLimit::try_new(
        u64::try_from(MAX_NANO_AGENT_LAUNCH_JSON_BYTES)
            .expect("launch JSON bound fits every supported host"),
    )
    .expect("launch JSON bound is nonzero and below the global asset limit");
    let source = load_deployment_asset(deployment_root, launch_relative_path, byte_limit)
        .map_err(NanoAgentLaunchLoadError::Load)?;
    let launch =
        NanoAgentLaunchV1::parse_json(source.bytes()).map_err(NanoAgentLaunchLoadError::Parse)?;
    for role in NanoLaunchAssetRole::ALL {
        if launch.asset(role).relative_path() == source.relative_path() {
            return Err(NanoAgentLaunchLoadError::InputAliasesLaunchDocument {
                role,
                relative_path: source.relative_path().clone(),
            });
        }
    }
    Ok(LoadedNanoAgentLaunchV1 { launch, source })
}

#[derive(Debug)]
pub enum NanoAgentLaunchLoadError {
    Load(DeploymentAssetLoadError),
    Parse(NanoAgentLaunchParseError),
    InputAliasesLaunchDocument {
        role: NanoLaunchAssetRole,
        relative_path: ArtifactRelativePath,
    },
}

impl fmt::Display for NanoAgentLaunchLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load(source) => write!(formatter, "Nano launch document load failed: {source}"),
            Self::Parse(source) => write!(formatter, "Nano launch document parse failed: {source}"),
            Self::InputAliasesLaunchDocument {
                role,
                relative_path,
            } => write!(
                formatter,
                "{role:?} asset aliases launch document {}",
                relative_path.as_str()
            ),
        }
    }
}

impl std::error::Error for NanoAgentLaunchLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Load(source) => Some(source),
            Self::Parse(source) => Some(source),
            Self::InputAliasesLaunchDocument { .. } => None,
        }
    }
}

#[derive(Debug)]
pub enum NanoAgentLaunchParseError {
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
    InvalidAssetPath {
        role: NanoLaunchAssetRole,
        source: ArtifactRelativePathError,
    },
    InvalidAssetByteLimit {
        role: NanoLaunchAssetRole,
        source: DeploymentAssetByteLimitError,
    },
    AssetByteLimitAboveRoleMaximum {
        role: NanoLaunchAssetRole,
        actual_bytes: u64,
        maximum_bytes: u64,
    },
    InvalidAssetSha256 {
        role: NanoLaunchAssetRole,
        source: NanoLaunchSha256Error,
    },
    DuplicateInputAssetPath {
        first: NanoLaunchAssetRole,
        second: NanoLaunchAssetRole,
        relative_path: ArtifactRelativePath,
    },
    InvalidPlantArtifactId,
    InvalidSocket {
        field: &'static str,
        source: std::net::AddrParseError,
    },
    NonLoopbackSocket {
        field: &'static str,
        address: SocketAddr,
    },
    ZeroSocketPort {
        field: &'static str,
    },
    UnsupportedOakSelectorSource,
    ProductionOakUsbPolicyRequired,
    RectifiedStereoRequired,
    RectifiedLeftDepthRequired,
    NonblockingOakQueueRequired,
    NumericOutOfRange {
        field: &'static str,
        value: u64,
        minimum: u64,
        maximum: u64,
    },
    OccupancyResolutionOutOfRange {
        resolution_m: f64,
        minimum_m: f64,
        maximum_m: f64,
    },
    OccupancyLowerBoundOutOfRange {
        axis: usize,
        value_m: f64,
        maximum_absolute_m: f64,
    },
    OccupancyCountNotRepresentable {
        field: &'static str,
        value: u64,
    },
    OccupancyHostPolicy(LiveOccupancyHostPolicyError),
    OakStereoDepthContractMismatch,
    OakDeviceConfig(DeviceConfigError),
    UnsupportedInferenceBackend {
        field: &'static str,
    },
    UnsupportedRerunKind,
    InvalidStoragePath {
        field: &'static str,
        source: ArtifactRelativePathError,
    },
    OverlappingStoragePaths {
        first: &'static str,
        second: &'static str,
    },
    StateQuotaArithmeticOverflow,
    ReservedStateExceedsTotal {
        reserved_bytes: u64,
        maximum_total_state_bytes: u64,
    },
}

impl fmt::Display for NanoAgentLaunchParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid production Nano launch document: {self:?}"
        )
    }
}

impl std::error::Error for NanoAgentLaunchParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::JsonDecode(source) | Self::JsonTrailingData(source) => Some(source),
            Self::InvalidAssetPath { source, .. } | Self::InvalidStoragePath { source, .. } => {
                Some(source)
            }
            Self::InvalidAssetByteLimit { source, .. } => Some(source),
            Self::InvalidAssetSha256 { source, .. } => Some(source),
            Self::InvalidSocket { source, .. } => Some(source),
            Self::OakDeviceConfig(source) => Some(source),
            Self::OccupancyHostPolicy(source) => Some(source),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoLaunchSha256Error {
    WrongLength {
        actual_bytes: usize,
        expected_bytes: usize,
    },
    NonLowercaseHex {
        index: usize,
        byte: u8,
    },
}

impl fmt::Display for NanoLaunchSha256Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid SHA-256 identity: {self:?}")
    }
}

impl std::error::Error for NanoLaunchSha256Error {}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoAgentLaunchV1Dto {
    schema_version: u32,
    agent_policy_asset: NanoLaunchAssetBindingDto,
    navigation_shadow_config_asset: NanoLaunchAssetBindingDto,
    physical_actuation_config_asset: NanoLaunchAssetBindingDto,
    controller_server: NanoLaunchControllerServerDto,
    plant_artifact: NanoLaunchPlantArtifactDto,
    oak: NanoOakStreamGraphDto,
    occupancy: NanoLaunchOccupancyDto,
    inference: NanoLaunchInferenceDto,
    rerun: NanoLaunchRerunDto,
    storage: NanoLaunchStorageDto,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoLaunchAssetBindingDto {
    relative_path: String,
    maximum_bytes: u64,
    sha256_hex: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoLaunchControllerServerDto {
    contract_asset: NanoLaunchAssetBindingDto,
    command_udp_endpoint: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoLaunchPlantArtifactDto {
    artifact_id: String,
    asset: NanoLaunchAssetBindingDto,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoOakStreamGraphDto {
    selector_source: String,
    maximum_usb_speed: String,
    minimum_usb_speed: String,
    rgb: NanoOakImageStreamDto,
    rectified_stereo: NanoOakRectifiedStereoStreamDto,
    depth: NanoOakDepthStreamDto,
    imu: NanoOakImuStreamDto,
    queue: NanoOakQueueDto,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoOakImageStreamDto {
    width_px: u32,
    height_px: u32,
    fps: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoOakRectifiedStereoStreamDto {
    width_px: u32,
    height_px: u32,
    fps: u32,
    rectified: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoOakDepthStreamDto {
    width_px: u32,
    height_px: u32,
    fps: u32,
    alignment: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoOakImuStreamDto {
    rate_hz: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoOakQueueDto {
    size: u32,
    blocking: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoLaunchOccupancyDto {
    resolution_m: f64,
    lower_x_m: f64,
    lower_y_m: f64,
    width_cells: u32,
    height_cells: u32,
    maximum_cells: u64,
    maximum_keyframes: u64,
    snapshot_every_keyframes: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoLaunchInferenceDto {
    onnx_runtime_library_asset: NanoLaunchAssetBindingDto,
    superpoint_model_asset: NanoLaunchAssetBindingDto,
    lightglue_model_asset: NanoLaunchAssetBindingDto,
    superpoint_backend: String,
    lightglue_backend: String,
    downscale_factor: u32,
    maximum_keypoints: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoLaunchRerunDto {
    kind: String,
    bind: String,
    decimation: u32,
    memory_limit_bytes: u64,
    flush_timeout_ms: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoLaunchStorageDto {
    map_snapshot_relative_path: String,
    navigation_records_relative_path: String,
    startup_evidence_relative_path: String,
    maximum_map_bytes: u64,
    maximum_navigation_record_bytes: u64,
    maximum_startup_evidence_bytes: u64,
    maximum_total_state_bytes: u64,
    minimum_free_bytes: u64,
}

fn parse_asset(
    role: NanoLaunchAssetRole,
    dto: NanoLaunchAssetBindingDto,
) -> Result<NanoLaunchAssetBinding, NanoAgentLaunchParseError> {
    let relative_path = ArtifactRelativePath::parse(dto.relative_path)
        .map_err(|source| NanoAgentLaunchParseError::InvalidAssetPath { role, source })?;
    let byte_limit = DeploymentAssetByteLimit::try_new(dto.maximum_bytes)
        .map_err(|source| NanoAgentLaunchParseError::InvalidAssetByteLimit { role, source })?;
    let role_maximum = role.maximum_bytes();
    if byte_limit.get() > role_maximum {
        return Err(NanoAgentLaunchParseError::AssetByteLimitAboveRoleMaximum {
            role,
            actual_bytes: byte_limit.get(),
            maximum_bytes: role_maximum,
        });
    }
    let expected_sha256 = parse_sha256(&dto.sha256_hex)
        .map_err(|source| NanoAgentLaunchParseError::InvalidAssetSha256 { role, source })?;
    Ok(NanoLaunchAssetBinding {
        relative_path,
        byte_limit,
        expected_sha256,
    })
}

fn parse_controller_server(
    dto: NanoLaunchControllerServerDto,
) -> Result<NanoLaunchControllerServer, NanoAgentLaunchParseError> {
    Ok(NanoLaunchControllerServer {
        contract_asset: parse_asset(
            NanoLaunchAssetRole::ControllerServerContract,
            dto.contract_asset,
        )?,
        command_udp_endpoint: parse_loopback_socket(
            "controller_server.command_udp_endpoint",
            dto.command_udp_endpoint,
        )?,
    })
}

fn parse_plant_artifact(
    dto: NanoLaunchPlantArtifactDto,
) -> Result<NanoLaunchPlantArtifact, NanoAgentLaunchParseError> {
    if dto.artifact_id.is_empty()
        || dto.artifact_id.len() > MAX_PLANT_ARTIFACT_ID_BYTES
        || dto.artifact_id.bytes().all(|byte| byte == b'0')
        || !dto
            .artifact_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
    {
        return Err(NanoAgentLaunchParseError::InvalidPlantArtifactId);
    }
    Ok(NanoLaunchPlantArtifact {
        artifact_id: NanoLaunchPlantArtifactId(dto.artifact_id),
        asset: parse_asset(NanoLaunchAssetRole::PlantArtifact, dto.asset)?,
    })
}

fn parse_oak(dto: NanoOakStreamGraphDto) -> Result<NanoOakStreamGraph, NanoAgentLaunchParseError> {
    if dto.selector_source != "exact_inventory_oak_mxid" {
        return Err(NanoAgentLaunchParseError::UnsupportedOakSelectorSource);
    }
    if dto.maximum_usb_speed != "SUPER" || dto.minimum_usb_speed != "SUPER" {
        return Err(NanoAgentLaunchParseError::ProductionOakUsbPolicyRequired);
    }
    if !dto.rectified_stereo.rectified {
        return Err(NanoAgentLaunchParseError::RectifiedStereoRequired);
    }
    if dto.depth.alignment != "rectified_left" {
        return Err(NanoAgentLaunchParseError::RectifiedLeftDepthRequired);
    }
    if dto.queue.blocking {
        return Err(NanoAgentLaunchParseError::NonblockingOakQueueRequired);
    }

    let rgb = parse_image_stream("oak.rgb", dto.rgb.width_px, dto.rgb.height_px, dto.rgb.fps)?;
    let rectified_stereo = parse_image_stream(
        "oak.rectified_stereo",
        dto.rectified_stereo.width_px,
        dto.rectified_stereo.height_px,
        dto.rectified_stereo.fps,
    )?;
    let depth = parse_image_stream(
        "oak.depth",
        dto.depth.width_px,
        dto.depth.height_px,
        dto.depth.fps,
    )?;
    if rectified_stereo != depth {
        return Err(NanoAgentLaunchParseError::OakStereoDepthContractMismatch);
    }
    let imu_rate_hz = bounded_nonzero_u32("oak.imu.rate_hz", dto.imu.rate_hz, MAX_OAK_IMU_RATE_HZ)?;
    let queue_size = bounded_nonzero_u32("oak.queue.size", dto.queue.size, MAX_OAK_QUEUE_SIZE)?;

    let graph = NanoOakStreamGraph {
        selector_source: NanoOakSelectorSource::ExactInventoryOakMxid,
        rgb,
        rectified_stereo,
        depth,
        imu_rate_hz,
        queue_size,
    };
    graph
        .device_config()
        .validate()
        .map_err(NanoAgentLaunchParseError::OakDeviceConfig)?;
    Ok(graph)
}

fn parse_occupancy(
    dto: NanoLaunchOccupancyDto,
) -> Result<NanoLaunchOccupancy, NanoAgentLaunchParseError> {
    if !dto.resolution_m.is_finite()
        || dto.resolution_m < MIN_NANO_OCCUPANCY_RESOLUTION_M
        || dto.resolution_m > MAX_NANO_OCCUPANCY_RESOLUTION_M
    {
        return Err(NanoAgentLaunchParseError::OccupancyResolutionOutOfRange {
            resolution_m: dto.resolution_m,
            minimum_m: MIN_NANO_OCCUPANCY_RESOLUTION_M,
            maximum_m: MAX_NANO_OCCUPANCY_RESOLUTION_M,
        });
    }
    for (axis, value_m) in [dto.lower_x_m, dto.lower_y_m].into_iter().enumerate() {
        if !value_m.is_finite() || value_m.abs() > MAX_NANO_OCCUPANCY_ABS_LOWER_BOUND_M {
            return Err(NanoAgentLaunchParseError::OccupancyLowerBoundOutOfRange {
                axis,
                value_m,
                maximum_absolute_m: MAX_NANO_OCCUPANCY_ABS_LOWER_BOUND_M,
            });
        }
    }
    let width_cells = bounded_nonzero_u32(
        "occupancy.width_cells",
        dto.width_cells,
        MAX_NANO_OCCUPANCY_AXIS_CELLS,
    )?;
    let height_cells = bounded_nonzero_u32(
        "occupancy.height_cells",
        dto.height_cells,
        MAX_NANO_OCCUPANCY_AXIS_CELLS,
    )?;
    let maximum_cells = bounded_nonzero_usize(
        "occupancy.maximum_cells",
        dto.maximum_cells,
        MAX_NANO_OCCUPANCY_CELLS,
    )?;
    let maximum_keyframes = bounded_nonzero_usize(
        "occupancy.maximum_keyframes",
        dto.maximum_keyframes,
        MAX_NANO_OCCUPANCY_KEYFRAMES,
    )?;
    let snapshot_every_keyframes = bounded_nonzero_usize(
        "occupancy.snapshot_every_keyframes",
        dto.snapshot_every_keyframes,
        MAX_NANO_OCCUPANCY_SNAPSHOT_CADENCE,
    )?;
    let host_policy = LiveOccupancyHostPolicy::try_new(
        dto.resolution_m,
        dto.lower_x_m,
        dto.lower_y_m,
        width_cells.get(),
        height_cells.get(),
        maximum_cells.get(),
        maximum_keyframes.get(),
        snapshot_every_keyframes.get(),
    )
    .map_err(NanoAgentLaunchParseError::OccupancyHostPolicy)?;
    Ok(NanoLaunchOccupancy { host_policy })
}

fn parse_image_stream(
    field: &'static str,
    width_px: u32,
    height_px: u32,
    fps: u32,
) -> Result<NanoOakImageStream, NanoAgentLaunchParseError> {
    Ok(NanoOakImageStream {
        width_px: bounded_nonzero_u32(field, width_px, MAX_OAK_IMAGE_WIDTH_PX)?,
        height_px: bounded_nonzero_u32(field, height_px, MAX_OAK_IMAGE_HEIGHT_PX)?,
        fps: bounded_nonzero_u32(field, fps, MAX_OAK_FRAME_RATE_HZ)?,
    })
}

fn bounded_nonzero_usize(
    field: &'static str,
    value: u64,
    maximum: usize,
) -> Result<NonZeroUsize, NanoAgentLaunchParseError> {
    let converted = usize::try_from(value)
        .map_err(|_| NanoAgentLaunchParseError::OccupancyCountNotRepresentable { field, value })?;
    if converted == 0 || converted > maximum {
        return Err(NanoAgentLaunchParseError::NumericOutOfRange {
            field,
            value,
            minimum: 1,
            maximum: u64::try_from(maximum).expect("Linux and macOS usize resource bounds fit u64"),
        });
    }
    Ok(NonZeroUsize::new(converted).expect("nonzero checked above"))
}

fn parse_inference(
    dto: NanoLaunchInferenceDto,
) -> Result<NanoLaunchInference, NanoAgentLaunchParseError> {
    Ok(NanoLaunchInference {
        onnx_runtime_library: parse_asset(
            NanoLaunchAssetRole::OnnxRuntimeLibrary,
            dto.onnx_runtime_library_asset,
        )?,
        superpoint_model: parse_asset(
            NanoLaunchAssetRole::SuperpointModel,
            dto.superpoint_model_asset,
        )?,
        lightglue_model: parse_asset(
            NanoLaunchAssetRole::LightglueModel,
            dto.lightglue_model_asset,
        )?,
        superpoint_backend: parse_inference_backend(
            "inference.superpoint_backend",
            &dto.superpoint_backend,
        )?,
        lightglue_backend: parse_inference_backend(
            "inference.lightglue_backend",
            &dto.lightglue_backend,
        )?,
        downscale_factor: bounded_nonzero_u32(
            "inference.downscale_factor",
            dto.downscale_factor,
            MAX_INFERENCE_DOWNSCALE_FACTOR,
        )?,
        maximum_keypoints: bounded_nonzero_u32(
            "inference.maximum_keypoints",
            dto.maximum_keypoints,
            MAX_INFERENCE_KEYPOINTS,
        )?,
    })
}

fn parse_inference_backend(
    field: &'static str,
    value: &str,
) -> Result<NanoInferenceBackend, NanoAgentLaunchParseError> {
    match value {
        "auto" => Ok(NanoInferenceBackend::Auto),
        "cpu" => Ok(NanoInferenceBackend::Cpu),
        "cuda" => Ok(NanoInferenceBackend::Cuda),
        "tensorrt" => Ok(NanoInferenceBackend::TensorRt),
        _ => Err(NanoAgentLaunchParseError::UnsupportedInferenceBackend { field }),
    }
}

fn parse_rerun(dto: NanoLaunchRerunDto) -> Result<NanoLaunchRerun, NanoAgentLaunchParseError> {
    if dto.kind != "serve_loopback" {
        return Err(NanoAgentLaunchParseError::UnsupportedRerunKind);
    }
    Ok(NanoLaunchRerun {
        bind: parse_loopback_socket("rerun.bind", dto.bind)?,
        decimation: bounded_nonzero_u32("rerun.decimation", dto.decimation, MAX_RERUN_DECIMATION)?,
        memory_limit_bytes: bounded_nonzero_u64(
            "rerun.memory_limit_bytes",
            dto.memory_limit_bytes,
            MIN_RERUN_MEMORY_BYTES,
            MAX_RERUN_MEMORY_BYTES,
        )?,
        flush_timeout_ms: bounded_nonzero_u64(
            "rerun.flush_timeout_ms",
            dto.flush_timeout_ms,
            1,
            MAX_RERUN_FLUSH_TIMEOUT_MS,
        )?,
    })
}

fn parse_storage(
    dto: NanoLaunchStorageDto,
) -> Result<NanoLaunchStorage, NanoAgentLaunchParseError> {
    let map_snapshot = parse_storage_path(
        "storage.map_snapshot_relative_path",
        dto.map_snapshot_relative_path,
    )?;
    let navigation_records = parse_storage_path(
        "storage.navigation_records_relative_path",
        dto.navigation_records_relative_path,
    )?;
    let startup_evidence = parse_storage_path(
        "storage.startup_evidence_relative_path",
        dto.startup_evidence_relative_path,
    )?;
    ensure_nonoverlapping_storage_paths([
        ("map_snapshot", &map_snapshot),
        ("navigation_records", &navigation_records),
        ("startup_evidence", &startup_evidence),
    ])?;

    let maximum_map_bytes = bounded_nonzero_u64(
        "storage.maximum_map_bytes",
        dto.maximum_map_bytes,
        1,
        MAX_NANO_STATE_BYTES,
    )?;
    let maximum_navigation_record_bytes = bounded_nonzero_u64(
        "storage.maximum_navigation_record_bytes",
        dto.maximum_navigation_record_bytes,
        1,
        MAX_NANO_STATE_BYTES,
    )?;
    let maximum_startup_evidence_bytes = bounded_nonzero_u64(
        "storage.maximum_startup_evidence_bytes",
        dto.maximum_startup_evidence_bytes,
        1,
        MAX_NANO_STATE_BYTES,
    )?;
    let maximum_total_state_bytes = bounded_nonzero_u64(
        "storage.maximum_total_state_bytes",
        dto.maximum_total_state_bytes,
        1,
        MAX_NANO_STATE_BYTES,
    )?;
    let minimum_free_bytes = bounded_nonzero_u64(
        "storage.minimum_free_bytes",
        dto.minimum_free_bytes,
        1,
        MAX_NANO_STATE_BYTES,
    )?;
    let reserved_bytes = maximum_map_bytes
        .get()
        .checked_add(maximum_navigation_record_bytes.get())
        .and_then(|value| value.checked_add(maximum_startup_evidence_bytes.get()))
        .ok_or(NanoAgentLaunchParseError::StateQuotaArithmeticOverflow)?;
    if reserved_bytes > maximum_total_state_bytes.get() {
        return Err(NanoAgentLaunchParseError::ReservedStateExceedsTotal {
            reserved_bytes,
            maximum_total_state_bytes: maximum_total_state_bytes.get(),
        });
    }
    Ok(NanoLaunchStorage {
        map_snapshot,
        navigation_records,
        startup_evidence,
        maximum_map_bytes,
        maximum_navigation_record_bytes,
        maximum_startup_evidence_bytes,
        maximum_total_state_bytes,
        minimum_free_bytes,
    })
}

fn parse_storage_path(
    field: &'static str,
    value: String,
) -> Result<ArtifactRelativePath, NanoAgentLaunchParseError> {
    ArtifactRelativePath::parse(value)
        .map_err(|source| NanoAgentLaunchParseError::InvalidStoragePath { field, source })
}

fn ensure_nonoverlapping_storage_paths(
    paths: [(&'static str, &ArtifactRelativePath); 3],
) -> Result<(), NanoAgentLaunchParseError> {
    for first_index in 0..paths.len() {
        for second_index in (first_index + 1)..paths.len() {
            let (first_name, first) = paths[first_index];
            let (second_name, second) = paths[second_index];
            if first.as_path().starts_with(second.as_path())
                || second.as_path().starts_with(first.as_path())
            {
                return Err(NanoAgentLaunchParseError::OverlappingStoragePaths {
                    first: first_name,
                    second: second_name,
                });
            }
        }
    }
    Ok(())
}

fn ensure_distinct_input_assets(
    launch: &NanoAgentLaunchV1,
) -> Result<(), NanoAgentLaunchParseError> {
    for (first_index, first_role) in NanoLaunchAssetRole::ALL.iter().copied().enumerate() {
        for second_role in NanoLaunchAssetRole::ALL[(first_index + 1)..]
            .iter()
            .copied()
        {
            let first = launch.asset(first_role);
            let second = launch.asset(second_role);
            if first.relative_path == second.relative_path {
                return Err(NanoAgentLaunchParseError::DuplicateInputAssetPath {
                    first: first_role,
                    second: second_role,
                    relative_path: first.relative_path.clone(),
                });
            }
        }
    }
    Ok(())
}

fn parse_loopback_socket(
    field: &'static str,
    value: String,
) -> Result<SocketAddr, NanoAgentLaunchParseError> {
    let address = value
        .parse::<SocketAddr>()
        .map_err(|source| NanoAgentLaunchParseError::InvalidSocket { field, source })?;
    if !address.ip().is_loopback() {
        return Err(NanoAgentLaunchParseError::NonLoopbackSocket { field, address });
    }
    if address.port() == 0 {
        return Err(NanoAgentLaunchParseError::ZeroSocketPort { field });
    }
    Ok(address)
}

fn bounded_nonzero_u32(
    field: &'static str,
    value: u32,
    maximum: u32,
) -> Result<NonZeroU32, NanoAgentLaunchParseError> {
    if value == 0 || value > maximum {
        return Err(NanoAgentLaunchParseError::NumericOutOfRange {
            field,
            value: u64::from(value),
            minimum: 1,
            maximum: u64::from(maximum),
        });
    }
    Ok(NonZeroU32::new(value).expect("nonzero checked above"))
}

fn bounded_nonzero_u64(
    field: &'static str,
    value: u64,
    minimum: u64,
    maximum: u64,
) -> Result<NonZeroU64, NanoAgentLaunchParseError> {
    if value < minimum || value > maximum {
        return Err(NanoAgentLaunchParseError::NumericOutOfRange {
            field,
            value,
            minimum,
            maximum,
        });
    }
    Ok(NonZeroU64::new(value).expect("positive minimum checked above"))
}

fn parse_sha256(value: &str) -> Result<[u8; 32], NanoLaunchSha256Error> {
    if value.len() != SHA256_HEX_BYTES {
        return Err(NanoLaunchSha256Error::WrongLength {
            actual_bytes: value.len(),
            expected_bytes: SHA256_HEX_BYTES,
        });
    }
    let mut digest = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = lowercase_hex_nibble(pair[0]).ok_or(NanoLaunchSha256Error::NonLowercaseHex {
            index: index * 2,
            byte: pair[0],
        })?;
        let low = lowercase_hex_nibble(pair[1]).ok_or(NanoLaunchSha256Error::NonLowercaseHex {
            index: index * 2 + 1,
            byte: pair[1],
        })?;
        digest[index] = (high << 4) | low;
    }
    Ok(digest)
}

const fn lowercase_hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::os::unix::fs::symlink;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};

    use super::*;

    fn digest(seed: u8) -> String {
        format!("{seed:02x}").repeat(32)
    }

    fn asset(path: &str, maximum_bytes: u64, seed: u8) -> Value {
        json!({
            "relative_path": path,
            "maximum_bytes": maximum_bytes,
            "sha256_hex": digest(seed)
        })
    }

    fn valid_value() -> Value {
        json!({
            "schema_version": 1,
            "agent_policy_asset": asset("config/agent-policy-v1.json", 65_536, 1),
            "navigation_shadow_config_asset": asset(
                "config/navigation-shadow-v1.json",
                262_144,
                2
            ),
            "physical_actuation_config_asset": asset(
                "config/navigation-actuation-v1.json",
                16_384,
                3
            ),
            "controller_server": {
                "contract_asset": asset("config/controller-server-v1.json", 8_192, 4),
                "command_udp_endpoint": "127.0.0.1:8080"
            },
            "plant_artifact": {
                "artifact_id": "qualified-drive-v1",
                "asset": asset("artifacts/plant/qualified-drive-v1.json", 1_048_576, 5)
            },
            "oak": {
                "selector_source": "exact_inventory_oak_mxid",
                "maximum_usb_speed": "SUPER",
                "minimum_usb_speed": "SUPER",
                "rgb": {"width_px": 640, "height_px": 480, "fps": 30},
                "rectified_stereo": {
                    "width_px": 640,
                    "height_px": 400,
                    "fps": 30,
                    "rectified": true
                },
                "depth": {
                    "width_px": 640,
                    "height_px": 400,
                    "fps": 30,
                    "alignment": "rectified_left"
                },
                "imu": {"rate_hz": 400},
                "queue": {"size": 8, "blocking": false}
            },
            "occupancy": {
                "resolution_m": 0.05,
                "lower_x_m": -10.0,
                "lower_y_m": -5.0,
                "width_cells": 400,
                "height_cells": 400,
                "maximum_cells": 4_000_000,
                "maximum_keyframes": 300,
                "snapshot_every_keyframes": 5
            },
            "inference": {
                "onnx_runtime_library_asset": asset(
                    "runtime/libonnxruntime.so",
                    134_217_728,
                    6
                ),
                "superpoint_model_asset": asset("models/sp.onnx", 134_217_728, 7),
                "lightglue_model_asset": asset("models/lg.onnx", 134_217_728, 8),
                "superpoint_backend": "auto",
                "lightglue_backend": "auto",
                "downscale_factor": 1,
                "maximum_keypoints": 1024
            },
            "rerun": {
                "kind": "serve_loopback",
                "bind": "127.0.0.1:9876",
                "decimation": 1,
                "memory_limit_bytes": 134_217_728,
                "flush_timeout_ms": 10_000
            },
            "storage": {
                "map_snapshot_relative_path": "maps/current.kiko-map",
                "navigation_records_relative_path": "records/navigation",
                "startup_evidence_relative_path": "evidence/startup",
                "maximum_map_bytes": 67_108_864,
                "maximum_navigation_record_bytes": 536_870_912,
                "maximum_startup_evidence_bytes": 16_777_216,
                "maximum_total_state_bytes": 1_073_741_824,
                "minimum_free_bytes": 134_217_728
            }
        })
    }

    fn parse(value: &Value) -> Result<NanoAgentLaunchV1, NanoAgentLaunchParseError> {
        NanoAgentLaunchV1::parse_json(&serde_json::to_vec(value).expect("fixture serializes"))
    }

    #[test]
    fn valid_document_builds_one_complete_production_graph() {
        let launch = parse(&valid_value()).expect("valid launch");
        assert_eq!(
            launch.oak().selector_source(),
            NanoOakSelectorSource::ExactInventoryOakMxid
        );
        assert!(launch.oak().device_config().rgb.is_some());
        assert!(launch.oak().device_config().mono.is_some());
        assert!(launch.oak().device_config().depth.is_some());
        assert!(launch.oak().device_config().imu.is_some());
        assert_eq!(
            launch.oak().device_config().usb_transport,
            UsbTransportPolicy::super_speed_required()
        );
        assert_eq!(
            launch.inference().superpoint_backend().runtime(),
            InferenceBackend::Auto
        );
        assert_eq!(
            launch.controller_server().command_udp_endpoint(),
            "127.0.0.1:8080".parse().expect("valid socket")
        );
        assert_eq!(launch.occupancy().geometry().resolution_m(), 0.05);
        assert_eq!(launch.occupancy().geometry().lower_bound_m(), [-10.0, -5.0]);
        assert_eq!(launch.occupancy().maximum_keyframes(), 300);
        assert_eq!(launch.occupancy().snapshot_cadence().get(), 5);
        assert_eq!(
            launch.occupancy().host_policy().geometry(),
            launch.occupancy().geometry()
        );
    }

    #[test]
    fn every_object_rejects_unknown_fields_and_trailing_json() {
        let mut top = valid_value();
        top["unexpected"] = json!(true);
        assert!(matches!(
            parse(&top),
            Err(NanoAgentLaunchParseError::JsonDecode(_))
        ));

        let mut nested = valid_value();
        nested["oak"]["rgb"]["unexpected"] = json!(true);
        assert!(matches!(
            parse(&nested),
            Err(NanoAgentLaunchParseError::JsonDecode(_))
        ));

        let mut bytes = serde_json::to_vec(&valid_value()).expect("fixture serializes");
        bytes.extend_from_slice(b"{}");
        assert!(matches!(
            NanoAgentLaunchV1::parse_json(&bytes),
            Err(NanoAgentLaunchParseError::JsonTrailingData(_))
        ));
    }

    #[test]
    fn all_asset_paths_are_canonical_bounded_and_unique() {
        let mut traversal = valid_value();
        traversal["agent_policy_asset"]["relative_path"] = json!("../policy.json");
        assert!(matches!(
            parse(&traversal),
            Err(NanoAgentLaunchParseError::InvalidAssetPath {
                role: NanoLaunchAssetRole::AgentPolicy,
                ..
            })
        ));

        let mut duplicate = valid_value();
        duplicate["inference"]["lightglue_model_asset"]["relative_path"] =
            duplicate["inference"]["superpoint_model_asset"]["relative_path"].clone();
        assert!(matches!(
            parse(&duplicate),
            Err(NanoAgentLaunchParseError::DuplicateInputAssetPath {
                first: NanoLaunchAssetRole::SuperpointModel,
                second: NanoLaunchAssetRole::LightglueModel,
                ..
            })
        ));

        let mut oversized_policy = valid_value();
        oversized_policy["agent_policy_asset"]["maximum_bytes"] =
            json!(MAX_NANO_AGENT_POLICY_CONFIG_JSON_BYTES as u64 + 1);
        assert!(matches!(
            parse(&oversized_policy),
            Err(NanoAgentLaunchParseError::AssetByteLimitAboveRoleMaximum {
                role: NanoLaunchAssetRole::AgentPolicy,
                ..
            })
        ));
    }

    #[test]
    fn sha256_is_canonical_lowercase_hex_and_reserves_no_valid_digest() {
        let mut uppercase = valid_value();
        uppercase["agent_policy_asset"]["sha256_hex"] = json!(digest(10).to_uppercase());
        assert!(matches!(
            parse(&uppercase),
            Err(NanoAgentLaunchParseError::InvalidAssetSha256 {
                source: NanoLaunchSha256Error::NonLowercaseHex { .. },
                ..
            })
        ));

        let mut zero = valid_value();
        zero["agent_policy_asset"]["sha256_hex"] = json!("0".repeat(64));
        assert_eq!(
            parse(&zero)
                .expect("all-zero SHA-256 is mathematically valid")
                .agent_policy()
                .expected_sha256(),
            &[0; 32]
        );
    }

    #[test]
    fn production_oak_graph_cannot_drop_required_contracts() {
        let mut usb2 = valid_value();
        usb2["oak"]["minimum_usb_speed"] = json!("HIGH");
        assert!(matches!(
            parse(&usb2),
            Err(NanoAgentLaunchParseError::ProductionOakUsbPolicyRequired)
        ));

        let mut unrectified = valid_value();
        unrectified["oak"]["rectified_stereo"]["rectified"] = json!(false);
        assert!(matches!(
            parse(&unrectified),
            Err(NanoAgentLaunchParseError::RectifiedStereoRequired)
        ));

        let mut mismatched_depth = valid_value();
        mismatched_depth["oak"]["depth"]["height_px"] = json!(480);
        assert!(matches!(
            parse(&mismatched_depth),
            Err(NanoAgentLaunchParseError::OakStereoDepthContractMismatch)
        ));

        let mut blocking = valid_value();
        blocking["oak"]["queue"]["blocking"] = json!(true);
        assert!(matches!(
            parse(&blocking),
            Err(NanoAgentLaunchParseError::NonblockingOakQueueRequired)
        ));
    }

    #[test]
    fn occupancy_host_resources_are_mandatory_finite_bounded_and_coherent() {
        let mut missing = valid_value();
        missing
            .as_object_mut()
            .expect("top-level fixture object")
            .remove("occupancy");
        assert!(matches!(
            parse(&missing),
            Err(NanoAgentLaunchParseError::JsonDecode(_))
        ));

        let mut resolution = valid_value();
        resolution["occupancy"]["resolution_m"] = json!(MAX_NANO_OCCUPANCY_RESOLUTION_M + 1.0);
        assert!(matches!(
            parse(&resolution),
            Err(NanoAgentLaunchParseError::OccupancyResolutionOutOfRange { .. })
        ));

        let mut lower_bound = valid_value();
        lower_bound["occupancy"]["lower_x_m"] = json!(MAX_NANO_OCCUPANCY_ABS_LOWER_BOUND_M + 1.0);
        assert!(matches!(
            parse(&lower_bound),
            Err(NanoAgentLaunchParseError::OccupancyLowerBoundOutOfRange { axis: 0, .. })
        ));

        let mut undersized_maximum = valid_value();
        undersized_maximum["occupancy"]["maximum_cells"] = json!(100);
        assert!(matches!(
            parse(&undersized_maximum),
            Err(NanoAgentLaunchParseError::OccupancyHostPolicy(
                LiveOccupancyHostPolicyError::Geometry(_)
            ))
        ));

        for (field, maximum) in [
            (
                "maximum_keyframes",
                u64::try_from(MAX_NANO_OCCUPANCY_KEYFRAMES)
                    .expect("host resource maximum fits u64"),
            ),
            (
                "snapshot_every_keyframes",
                u64::try_from(MAX_NANO_OCCUPANCY_SNAPSHOT_CADENCE)
                    .expect("host resource maximum fits u64"),
            ),
        ] {
            let mut excessive = valid_value();
            excessive["occupancy"][field] = json!(maximum + 1);
            assert!(matches!(
                parse(&excessive),
                Err(NanoAgentLaunchParseError::NumericOutOfRange { .. })
            ));
        }
    }

    #[test]
    fn transports_are_loopback_only_and_numeric_fields_are_bounded() {
        let mut public_controller = valid_value();
        public_controller["controller_server"]["command_udp_endpoint"] = json!("0.0.0.0:8080");
        assert!(matches!(
            parse(&public_controller),
            Err(NanoAgentLaunchParseError::NonLoopbackSocket {
                field: "controller_server.command_udp_endpoint",
                ..
            })
        ));

        let mut public_rerun = valid_value();
        public_rerun["rerun"]["bind"] = json!("[::]:9876");
        assert!(matches!(
            parse(&public_rerun),
            Err(NanoAgentLaunchParseError::NonLoopbackSocket {
                field: "rerun.bind",
                ..
            })
        ));

        let mut fps = valid_value();
        fps["oak"]["rgb"]["fps"] = json!(MAX_OAK_FRAME_RATE_HZ + 1);
        assert!(matches!(
            parse(&fps),
            Err(NanoAgentLaunchParseError::NumericOutOfRange {
                field: "oak.rgb",
                ..
            })
        ));
    }

    #[test]
    fn state_paths_and_quotas_do_not_overlap_or_overcommit() {
        let mut overlap = valid_value();
        overlap["storage"]["startup_evidence_relative_path"] = json!("records/navigation/evidence");
        assert!(matches!(
            parse(&overlap),
            Err(NanoAgentLaunchParseError::OverlappingStoragePaths { .. })
        ));

        let mut overcommit = valid_value();
        overcommit["storage"]["maximum_total_state_bytes"] = json!(100);
        assert!(matches!(
            parse(&overcommit),
            Err(NanoAgentLaunchParseError::ReservedStateExceedsTotal { .. })
        ));
    }

    #[test]
    fn exact_asset_loader_rejects_content_mismatch() {
        let requested_directory = unique_test_directory("bound-asset");
        fs::create_dir_all(requested_directory.join("config")).expect("create test directory");
        let directory =
            fs::canonicalize(requested_directory).expect("canonicalize test directory root");
        fs::write(directory.join("config/policy.json"), b"actual").expect("write asset");
        let binding = NanoLaunchAssetBinding {
            relative_path: ArtifactRelativePath::parse("config/policy.json".to_owned())
                .expect("relative path"),
            byte_limit: DeploymentAssetByteLimit::try_new(64).expect("byte limit"),
            expected_sha256: Sha256::digest(b"different").into(),
        };
        assert!(matches!(
            binding.load_exact(&directory),
            Err(NanoLaunchBoundAssetLoadError::ContentMismatch { .. })
        ));
        fs::remove_dir_all(directory).expect("remove test directory");
    }

    #[test]
    fn launch_loader_rejects_symlinks_and_self_aliases() {
        let requested_directory = unique_test_directory("launch-loader");
        fs::create_dir_all(&requested_directory).expect("create test directory");
        let directory =
            fs::canonicalize(requested_directory).expect("canonicalize test directory root");
        let bytes = serde_json::to_vec(&valid_value()).expect("fixture serializes");
        fs::write(directory.join("launch.json"), &bytes).expect("write launch");
        symlink("launch.json", directory.join("launch-link.json")).expect("create symlink");
        let linked = load_nano_agent_launch_v1(
            &directory,
            ArtifactRelativePath::parse("launch-link.json".to_owned()).expect("relative path"),
        );
        assert!(matches!(linked, Err(NanoAgentLaunchLoadError::Load(_))));

        let mut aliases = valid_value();
        aliases["agent_policy_asset"]["relative_path"] = json!("launch.json");
        fs::write(
            directory.join("launch.json"),
            serde_json::to_vec(&aliases).expect("fixture serializes"),
        )
        .expect("rewrite launch");
        let aliased = load_nano_agent_launch_v1(
            &directory,
            ArtifactRelativePath::parse("launch.json".to_owned()).expect("relative path"),
        );
        assert!(matches!(
            aliased,
            Err(NanoAgentLaunchLoadError::InputAliasesLaunchDocument {
                role: NanoLaunchAssetRole::AgentPolicy,
                ..
            })
        ));
        fs::remove_dir_all(directory).expect("remove test directory");
    }

    fn unique_test_directory(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "kiko-nano-launch-{label}-{}-{nonce}",
            std::process::id()
        ))
    }
}
