//! Strict deployment document for the production Nano agent.
//!
//! This is a structural, content-binding boundary. It parses one bounded JSON
//! document into canonical deployment-relative paths, exact SHA-256 identities,
//! finite resource bounds, and a single complete OAK stream graph. Parsing or
//! loading this document does not prove publisher authenticity, filesystem
//! ownership, device presence, USB speed, model compatibility, controller
//! liveness, calibration quality, plant validity, or safe physical motion.

use std::fmt;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::num::{NonZeroU16, NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::Path;

use kiko_device_inventory::{
    ArtifactRelativePath, ArtifactRelativePathError, DeploymentAssetByteLimit,
    DeploymentAssetByteLimitError, DeploymentAssetContentSha256, DeploymentAssetLoadError,
    LoadedDeploymentAsset, MAX_DEPLOYMENT_ASSET_BYTES, StreamedDeploymentAssetIdentity,
    load_deployment_asset, stream_deployment_asset_identity,
};
use oak_sys::{
    DepthAlignment, DepthConfig, DeviceConfig, DeviceConfigError, ImuConfig, MonoConfig,
    QueueConfig, RgbConfig, UsbTransportPolicy, UsbTransportSpeed,
};
use serde::Deserialize;

use super::{
    ConsoleRerunDiagnosticsUrl, MAX_HEAD_GAZE_POLICY_JSON_BYTES,
    MAX_NANO_AGENT_POLICY_CONFIG_JSON_BYTES, MAX_NANO_CALIBRATION_ARTIFACT_JSON_BYTES,
    MAX_NANO_OCCUPANCY_CELLS, MAX_NAVIGATION_ACTUATION_CONFIG_JSON_BYTES,
    MAX_NAVIGATION_INGRESS_RECORDS, MAX_SHADOW_NAVIGATION_CONFIG_JSON_BYTES,
};
use crate::InferenceBackend;
use crate::dense::occupancy::OccupancyGridGeometry;
use crate::dense::occupancy_runtime::OccupancySnapshotCadence;
use crate::live_runtime::{LiveOccupancyHostPolicy, LiveOccupancyHostPolicyError};

/// Legacy launch schema retained only for explicit compatibility tooling.
pub const NANO_AGENT_LAUNCH_V2: u32 = 2;

/// Production launch schema that adds mandatory, content-bound face cascades.
///
/// V2 remains a separate parser and type so adding face perception does not
/// silently weaken or reinterpret an already deployed V2 document.
pub const NANO_AGENT_LAUNCH_V3: u32 = 3;

/// Production launch schema that makes physically reviewed head gaze part of
/// the exact deployment graph.
///
/// V3 remains separately parseable, but production startup uses V4 so a
/// natural-hold-only bundle cannot silently claim expressive head control.
pub const NANO_AGENT_LAUNCH_V4: u32 = 4;

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
pub const MAX_NANO_NAVIGATION_DATASET_FILES: u64 = 65_536;
pub const MIN_NANO_OCCUPANCY_RESOLUTION_M: f64 = 0.001;
pub const MAX_NANO_OCCUPANCY_RESOLUTION_M: f64 = 10.0;
pub const MAX_NANO_OCCUPANCY_ABS_LOWER_BOUND_M: f64 = 100_000.0;
pub const MAX_NANO_OCCUPANCY_AXIS_CELLS: u32 = 100_000;
pub const MAX_NANO_OCCUPANCY_KEYFRAMES: usize = 1_000_000;
pub const MAX_NANO_OCCUPANCY_SNAPSHOT_CADENCE: usize = 1_000_000;
/// Per-file ceiling for the XML cascade inputs retained by launch V3.
pub const MAX_OPENCV_HAAR_CASCADE_BYTES: u64 = 4 * 1_024 * 1_024;
/// Physical-review evidence is retained as an opaque, exact-byte artifact.
/// Admission interprets only the digest claimed by the typed policy.
pub const MAX_HEAD_GAZE_REVIEW_EVIDENCE_BYTES: u64 = 1_024 * 1_024;

const SHA256_HEX_BYTES: usize = 64;
const MAX_LAUNCH_ARTIFACT_ID_BYTES: usize = 64;
const MIN_RERUN_MEMORY_BYTES: u64 = 1_048_576;
const NAVIGATION_DATASET_ADMISSION_FRAGMENT_BYTES: u64 = 4_096;
const MAX_NAVIGATION_DATASET_MANIFEST_BYTES: u64 = 64 * 1_024 * 1_024;
const MAX_WARM_START_SELECTION_BYTES: u64 = 4 * 1_024;

/// Roles are retained in errors so a rejected weak field is unambiguous.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoLaunchAssetRole {
    AgentPolicy,
    NavigationShadowConfig,
    PhysicalActuationConfig,
    ControllerServerContract,
    CalibrationArtifact,
    PlantArtifact,
    OnnxRuntimeLibrary,
    SuperpointModel,
    LightglueModel,
}

/// The two distinct OpenCV cascade roles required by launch V3.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoFaceCascadeAssetRole {
    FrontalFace,
    ProfileFace,
}

impl NanoFaceCascadeAssetRole {
    pub const ALL: [Self; 2] = [Self::FrontalFace, Self::ProfileFace];
}

/// The two distinct inputs needed to activate physical head gaze in launch V4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoHeadGazeAssetRole {
    Policy,
    PhysicalReviewEvidence,
}

impl NanoHeadGazeAssetRole {
    pub const ALL: [Self; 2] = [Self::Policy, Self::PhysicalReviewEvidence];

    const fn maximum_bytes(self) -> u64 {
        match self {
            Self::Policy => MAX_HEAD_GAZE_POLICY_JSON_BYTES as u64,
            Self::PhysicalReviewEvidence => MAX_HEAD_GAZE_REVIEW_EVIDENCE_BYTES,
        }
    }
}

impl NanoLaunchAssetRole {
    const ALL: [Self; 9] = [
        Self::AgentPolicy,
        Self::NavigationShadowConfig,
        Self::PhysicalActuationConfig,
        Self::ControllerServerContract,
        Self::CalibrationArtifact,
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
            Self::CalibrationArtifact => MAX_NANO_CALIBRATION_ARTIFACT_JSON_BYTES as u64,
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
    #[cfg(feature = "nano-wheels-off-qualification")]
    pub(crate) const fn from_parsed_parts(
        relative_path: ArtifactRelativePath,
        byte_limit: DeploymentAssetByteLimit,
        expected_sha256: [u8; 32],
    ) -> Self {
        Self {
            relative_path,
            byte_limit,
            expected_sha256,
        }
    }

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

    /// Stream and compare one exact bound asset without retaining its bytes.
    ///
    /// Use this for executable and native-library evidence that is consumed by
    /// the OS loader rather than by Rust. Assets parsed or consumed from memory
    /// must continue to use [`Self::load_exact`].
    pub fn verify_exact_streaming(
        &self,
        deployment_root: &Path,
    ) -> Result<StreamedDeploymentAssetIdentity, NanoLaunchBoundAssetLoadError> {
        let identity = stream_deployment_asset_identity(
            deployment_root,
            self.relative_path.clone(),
            self.byte_limit,
        )
        .map_err(NanoLaunchBoundAssetLoadError::Load)?;
        let observed = *identity.content_sha256().as_bytes();
        if observed != self.expected_sha256 {
            return Err(NanoLaunchBoundAssetLoadError::ContentMismatch {
                relative_path: self.relative_path.clone(),
                expected_sha256: self.expected_sha256,
                observed_sha256: observed,
            });
        }
        Ok(identity)
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

/// Mandatory launch-V3 bindings for the exact frontal and profile cascades.
///
/// These bindings retain canonical deployment-relative identity and exact
/// content digests. `load_exact` retains the verified bytes, and the native
/// detector parses those exact in-memory byte slices with OpenCV
/// `FileStorage`; it does not reopen either deployment path. The immutable
/// deployment root remains part of install admission, while detector
/// construction is bound to the already verified content rather than a second
/// pathname lookup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoLaunchFacePerception {
    frontal_face_cascade: NanoLaunchAssetBinding,
    profile_face_cascade: NanoLaunchAssetBinding,
}

impl NanoLaunchFacePerception {
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

/// Exact launch-V4 inputs for physically reviewed gaze and character motion.
///
/// The policy is parsed later into domain types. The review evidence remains
/// opaque: only its retained content identity is cross-bound to the policy's
/// operator-claimed review before motion types can be constructed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoLaunchPhysicalHeadGaze {
    policy: NanoLaunchAssetBinding,
    review_evidence: NanoLaunchAssetBinding,
}

impl NanoLaunchPhysicalHeadGaze {
    pub const fn policy(&self) -> &NanoLaunchAssetBinding {
        &self.policy
    }

    pub const fn review_evidence(&self) -> &NanoLaunchAssetBinding {
        &self.review_evidence
    }

    pub const fn asset(&self, role: NanoHeadGazeAssetRole) -> &NanoLaunchAssetBinding {
        match role {
            NanoHeadGazeAssetRole::Policy => &self.policy,
            NanoHeadGazeAssetRole::PhysicalReviewEvidence => &self.review_evidence,
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

/// Bounded calibration artifact identifier retained for exact manifest,
/// policy, launch-path, and content-digest binding during bootstrap.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoLaunchCalibrationArtifactId(String);

impl NanoLaunchCalibrationArtifactId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoLaunchCalibrationArtifact {
    artifact_id: NanoLaunchCalibrationArtifactId,
    asset: NanoLaunchAssetBinding,
}

impl NanoLaunchCalibrationArtifact {
    pub const fn artifact_id(&self) -> &NanoLaunchCalibrationArtifactId {
        &self.artifact_id
    }

    pub const fn asset(&self) -> &NanoLaunchAssetBinding {
        &self.asset
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
    usb_transport: UsbTransportPolicy,
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

    pub const fn usb_transport(&self) -> UsbTransportPolicy {
        self.usb_transport
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
            usb_transport: self.usb_transport,
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
/// exact parsed [`crate::navigation::ShadowNavigationConfigV2`] owns
/// the explicit, reviewed world-to-floor-occupancy transform, the runtime
/// depth camera/intrinsics, height and depth ranges, and sampling block. The
/// fixed integer evidence model remains executable code. This launch
/// component owns only global extent, retained-evidence capacity, and
/// publication cadence.
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
    diagnostics_url: ConsoleRerunDiagnosticsUrl,
    decimation: NonZeroU32,
    memory_limit_bytes: NonZeroU64,
    flush_timeout_ms: NonZeroU64,
}

impl NanoLaunchRerun {
    pub fn bind(self) -> SocketAddr {
        self.diagnostics_url.serve_loopback_bind()
    }

    pub const fn diagnostics_url(self) -> ConsoleRerunDiagnosticsUrl {
        self.diagnostics_url
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NanoNavigationDatasetStorageLimits {
    maximum_bytes: NonZeroU64,
    maximum_files: NonZeroU64,
    maximum_ingress_records: NonZeroUsize,
    minimum_free_bytes_after_write: NonZeroU64,
    terminal_reserve_bytes: NonZeroU64,
}

impl NanoNavigationDatasetStorageLimits {
    /// Maximum cumulative logical bytes admitted for dataset-owned payloads,
    /// sidecars, IMU samples, journal records, and the final manifest.
    pub const fn maximum_bytes(self) -> u64 {
        self.maximum_bytes.get()
    }

    /// Maximum cumulative regular-file count admitted beneath the dataset.
    pub const fn maximum_files(self) -> u64 {
        self.maximum_files.get()
    }

    /// Independent upper bound for journal records admitted in this dataset.
    pub const fn maximum_ingress_records(self) -> usize {
        self.maximum_ingress_records.get()
    }

    /// Descriptor-relative filesystem free-space floor required after every
    /// dataset-owned write.
    pub const fn minimum_free_bytes_after_write(self) -> u64 {
        self.minimum_free_bytes_after_write.get()
    }

    /// Admission bytes withheld from open-ended capture so terminal manifest,
    /// map, and selection publication can still allocate their bounded
    /// transient files. Map and selection bytes remain owned and counted by
    /// their existing persistence quota, not by the dataset logical-byte
    /// counter.
    pub const fn terminal_reserve_bytes(self) -> u64 {
        self.terminal_reserve_bytes.get()
    }
}

/// Output paths are canonical and relative to the state root supplied by the
/// service, never to the deployment root or current working directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoLaunchStorage {
    map_snapshot: ArtifactRelativePath,
    navigation_dataset_directory: ArtifactRelativePath,
    maximum_map_snapshot_bytes: NonZeroU64,
    minimum_free_bytes_after_map_save: NonZeroU64,
    navigation_dataset_limits: NanoNavigationDatasetStorageLimits,
}

impl NanoLaunchStorage {
    pub const fn map_snapshot(&self) -> &ArtifactRelativePath {
        &self.map_snapshot
    }

    /// Quota-controlled recording destination for the current navigation
    /// session.
    pub const fn navigation_dataset_directory(&self) -> &ArtifactRelativePath {
        &self.navigation_dataset_directory
    }

    pub const fn navigation_dataset_limits(&self) -> NanoNavigationDatasetStorageLimits {
        self.navigation_dataset_limits
    }

    /// Exact encoded-byte ceiling enforced for the configured map snapshot.
    pub const fn maximum_map_snapshot_bytes(&self) -> u64 {
        self.maximum_map_snapshot_bytes.get()
    }

    /// Free-space floor enforced after map publication and reserved, in
    /// addition to fragment-rounded transient map bytes, before publication.
    pub const fn minimum_free_bytes_after_map_save(&self) -> u64 {
        self.minimum_free_bytes_after_map_save.get()
    }
}

/// Fully parsed legacy V2 launch document.
///
/// Canonical production and attended commissioning use `NanoAgentLaunchV4`;
/// this type never invents or implies face-perception or physical-head-gaze
/// assets.
#[derive(Clone, Debug, PartialEq)]
pub struct NanoAgentLaunchV2 {
    agent_policy: NanoLaunchAssetBinding,
    navigation_shadow_config: NanoLaunchAssetBinding,
    physical_actuation_config: NanoLaunchAssetBinding,
    controller_server: NanoLaunchControllerServer,
    calibration_artifact: NanoLaunchCalibrationArtifact,
    plant_artifact: NanoLaunchPlantArtifact,
    oak: NanoOakStreamGraph,
    occupancy: NanoLaunchOccupancy,
    inference: NanoLaunchInference,
    rerun: NanoLaunchRerun,
    storage: NanoLaunchStorage,
}

impl NanoAgentLaunchV2 {
    /// Parse one exact bounded JSON byte sequence exactly once.
    pub fn parse_json(json: &[u8]) -> Result<Self, NanoAgentLaunchParseError> {
        if json.len() > MAX_NANO_AGENT_LAUNCH_JSON_BYTES {
            return Err(NanoAgentLaunchParseError::InputTooLarge {
                actual_bytes: json.len(),
                maximum_bytes: MAX_NANO_AGENT_LAUNCH_JSON_BYTES,
            });
        }
        let mut deserializer = serde_json::Deserializer::from_slice(json);
        let dto = NanoAgentLaunchV2Dto::deserialize(&mut deserializer)
            .map_err(NanoAgentLaunchParseError::JsonDecode)?;
        deserializer
            .end()
            .map_err(NanoAgentLaunchParseError::JsonTrailingData)?;
        Self::from_dto(dto)
    }

    fn from_dto(dto: NanoAgentLaunchV2Dto) -> Result<Self, NanoAgentLaunchParseError> {
        if dto.schema_version != NANO_AGENT_LAUNCH_V2 {
            return Err(NanoAgentLaunchParseError::UnsupportedSchema {
                actual: dto.schema_version,
                supported: NANO_AGENT_LAUNCH_V2,
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
        let calibration_artifact = parse_calibration_artifact(dto.calibration_artifact)?;
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
            calibration_artifact,
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

    pub fn asset(&self, role: NanoLaunchAssetRole) -> &NanoLaunchAssetBinding {
        match role {
            NanoLaunchAssetRole::AgentPolicy => &self.agent_policy,
            NanoLaunchAssetRole::NavigationShadowConfig => &self.navigation_shadow_config,
            NanoLaunchAssetRole::PhysicalActuationConfig => &self.physical_actuation_config,
            NanoLaunchAssetRole::ControllerServerContract => &self.controller_server.contract_asset,
            NanoLaunchAssetRole::CalibrationArtifact => &self.calibration_artifact.asset,
            NanoLaunchAssetRole::PlantArtifact => &self.plant_artifact.asset,
            NanoLaunchAssetRole::OnnxRuntimeLibrary => &self.inference.onnx_runtime_library,
            NanoLaunchAssetRole::SuperpointModel => &self.inference.superpoint_model,
            NanoLaunchAssetRole::LightglueModel => &self.inference.lightglue_model,
        }
    }
}

/// Launch V3 retains the complete established runtime graph and exact face assets.
#[derive(Clone, Debug, PartialEq)]
pub struct NanoAgentLaunchV3 {
    common: NanoAgentLaunchV2,
    face_perception: NanoLaunchFacePerception,
}

impl NanoAgentLaunchV3 {
    /// Parse one exact bounded V3 JSON byte sequence exactly once.
    pub fn parse_json(json: &[u8]) -> Result<Self, NanoAgentLaunchParseError> {
        if json.len() > MAX_NANO_AGENT_LAUNCH_JSON_BYTES {
            return Err(NanoAgentLaunchParseError::InputTooLarge {
                actual_bytes: json.len(),
                maximum_bytes: MAX_NANO_AGENT_LAUNCH_JSON_BYTES,
            });
        }
        let mut deserializer = serde_json::Deserializer::from_slice(json);
        let dto = NanoAgentLaunchV3Dto::deserialize(&mut deserializer)
            .map_err(NanoAgentLaunchParseError::JsonDecode)?;
        deserializer
            .end()
            .map_err(NanoAgentLaunchParseError::JsonTrailingData)?;
        Self::from_dto(dto)
    }

    fn from_dto(dto: NanoAgentLaunchV3Dto) -> Result<Self, NanoAgentLaunchParseError> {
        if dto.schema_version != NANO_AGENT_LAUNCH_V3 {
            return Err(NanoAgentLaunchParseError::UnsupportedSchema {
                actual: dto.schema_version,
                supported: NANO_AGENT_LAUNCH_V3,
            });
        }

        let (common_dto, face_perception_dto) = dto.into_parts();
        let face_perception_dto =
            face_perception_dto.ok_or(NanoAgentLaunchParseError::MissingFacePerception)?;
        let face_perception = parse_face_perception(face_perception_dto)?;
        let common = NanoAgentLaunchV2::from_dto(common_dto)?;
        let launch = Self {
            common,
            face_perception,
        };
        ensure_distinct_v3_input_assets(&launch)?;
        Ok(launch)
    }

    pub const fn agent_policy(&self) -> &NanoLaunchAssetBinding {
        self.common.agent_policy()
    }

    pub const fn navigation_shadow_config(&self) -> &NanoLaunchAssetBinding {
        self.common.navigation_shadow_config()
    }

    pub const fn physical_actuation_config(&self) -> &NanoLaunchAssetBinding {
        self.common.physical_actuation_config()
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

    pub fn asset(&self, role: NanoLaunchAssetRole) -> &NanoLaunchAssetBinding {
        self.common.asset(role)
    }

    pub const fn face_perception(&self) -> &NanoLaunchFacePerception {
        &self.face_perception
    }
}

/// Canonical production launch: the established V3 graph plus mandatory,
/// exact-byte physical head-gaze inputs.
#[derive(Clone, Debug, PartialEq)]
pub struct NanoAgentLaunchV4 {
    common: NanoAgentLaunchV3,
    physical_head_gaze: NanoLaunchPhysicalHeadGaze,
}

impl NanoAgentLaunchV4 {
    /// Parse one exact bounded V4 JSON byte sequence exactly once.
    pub fn parse_json(json: &[u8]) -> Result<Self, NanoAgentLaunchParseError> {
        if json.len() > MAX_NANO_AGENT_LAUNCH_JSON_BYTES {
            return Err(NanoAgentLaunchParseError::InputTooLarge {
                actual_bytes: json.len(),
                maximum_bytes: MAX_NANO_AGENT_LAUNCH_JSON_BYTES,
            });
        }
        let mut deserializer = serde_json::Deserializer::from_slice(json);
        let dto = NanoAgentLaunchV4Dto::deserialize(&mut deserializer)
            .map_err(NanoAgentLaunchParseError::JsonDecode)?;
        deserializer
            .end()
            .map_err(NanoAgentLaunchParseError::JsonTrailingData)?;
        if dto.schema_version != NANO_AGENT_LAUNCH_V4 {
            return Err(NanoAgentLaunchParseError::UnsupportedSchema {
                actual: dto.schema_version,
                supported: NANO_AGENT_LAUNCH_V4,
            });
        }

        let (common_dto, physical_head_gaze_dto) = dto.into_parts();
        let common = NanoAgentLaunchV3::from_dto(common_dto)?;
        let physical_head_gaze = parse_physical_head_gaze(physical_head_gaze_dto)?;
        let launch = Self {
            common,
            physical_head_gaze,
        };
        ensure_distinct_v4_input_assets(&launch)?;
        Ok(launch)
    }

    pub const fn agent_policy(&self) -> &NanoLaunchAssetBinding {
        self.common.agent_policy()
    }

    pub const fn navigation_shadow_config(&self) -> &NanoLaunchAssetBinding {
        self.common.navigation_shadow_config()
    }

    pub const fn physical_actuation_config(&self) -> &NanoLaunchAssetBinding {
        self.common.physical_actuation_config()
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

    pub fn asset(&self, role: NanoLaunchAssetRole) -> &NanoLaunchAssetBinding {
        self.common.asset(role)
    }

    pub const fn face_perception(&self) -> &NanoLaunchFacePerception {
        self.common.face_perception()
    }

    pub const fn physical_head_gaze(&self) -> &NanoLaunchPhysicalHeadGaze {
        &self.physical_head_gaze
    }
}

/// Loaded launch document retaining the exact no-follow source bytes and
/// digest used by the parser.
#[derive(Debug)]
pub struct LoadedNanoAgentLaunchV2 {
    launch: NanoAgentLaunchV2,
    source: LoadedDeploymentAsset,
}

impl LoadedNanoAgentLaunchV2 {
    pub const fn launch(&self) -> &NanoAgentLaunchV2 {
        &self.launch
    }

    pub const fn source(&self) -> &LoadedDeploymentAsset {
        &self.source
    }

    pub const fn content_sha256(&self) -> DeploymentAssetContentSha256 {
        self.source.content_sha256()
    }

    pub fn into_parts(self) -> (NanoAgentLaunchV2, LoadedDeploymentAsset) {
        (self.launch, self.source)
    }
}

/// Loaded V3 launch retaining the exact no-follow source bytes and digest.
#[derive(Debug)]
pub struct LoadedNanoAgentLaunchV3 {
    launch: NanoAgentLaunchV3,
    source: LoadedDeploymentAsset,
}

impl LoadedNanoAgentLaunchV3 {
    pub const fn launch(&self) -> &NanoAgentLaunchV3 {
        &self.launch
    }

    pub const fn source(&self) -> &LoadedDeploymentAsset {
        &self.source
    }

    pub const fn content_sha256(&self) -> DeploymentAssetContentSha256 {
        self.source.content_sha256()
    }

    pub fn into_parts(self) -> (NanoAgentLaunchV3, LoadedDeploymentAsset) {
        (self.launch, self.source)
    }
}

/// Loaded V4 launch retaining the exact no-follow source bytes and digest.
#[derive(Debug)]
pub struct LoadedNanoAgentLaunchV4 {
    launch: NanoAgentLaunchV4,
    source: LoadedDeploymentAsset,
}

impl LoadedNanoAgentLaunchV4 {
    pub const fn launch(&self) -> &NanoAgentLaunchV4 {
        &self.launch
    }

    pub const fn source(&self) -> &LoadedDeploymentAsset {
        &self.source
    }

    pub const fn content_sha256(&self) -> DeploymentAssetContentSha256 {
        self.source.content_sha256()
    }

    pub fn into_parts(self) -> (NanoAgentLaunchV4, LoadedDeploymentAsset) {
        (self.launch, self.source)
    }
}

/// Compatibility-only V2 loader beneath one canonical absolute deployment root.
///
/// Every path component is opened without following symlinks. The retained
/// content digest identifies the exact bytes parsed, but does not authenticate
/// the root or its publisher.
pub fn load_nano_agent_launch_v2(
    deployment_root: &Path,
    launch_relative_path: ArtifactRelativePath,
) -> Result<LoadedNanoAgentLaunchV2, NanoAgentLaunchLoadError> {
    let byte_limit = DeploymentAssetByteLimit::try_new(
        u64::try_from(MAX_NANO_AGENT_LAUNCH_JSON_BYTES)
            .expect("launch JSON bound fits every supported host"),
    )
    .expect("launch JSON bound is nonzero and below the global asset limit");
    let source = load_deployment_asset(deployment_root, launch_relative_path, byte_limit)
        .map_err(NanoAgentLaunchLoadError::Load)?;
    let launch =
        NanoAgentLaunchV2::parse_json(source.bytes()).map_err(NanoAgentLaunchLoadError::Parse)?;
    for role in NanoLaunchAssetRole::ALL {
        if launch.asset(role).relative_path() == source.relative_path() {
            return Err(NanoAgentLaunchLoadError::InputAliasesLaunchDocument {
                role,
                relative_path: source.relative_path().clone(),
            });
        }
    }
    Ok(LoadedNanoAgentLaunchV2 { launch, source })
}

/// Load an exact V3 launch document beneath one canonical absolute root.
///
/// This only loads the launch document. Callers must separately use
/// `NanoLaunchAssetBinding::load_exact` for every referenced input before
/// constructing a runtime.
pub fn load_nano_agent_launch_v3(
    deployment_root: &Path,
    launch_relative_path: ArtifactRelativePath,
) -> Result<LoadedNanoAgentLaunchV3, NanoAgentLaunchLoadError> {
    let byte_limit = DeploymentAssetByteLimit::try_new(
        u64::try_from(MAX_NANO_AGENT_LAUNCH_JSON_BYTES)
            .expect("launch JSON bound fits every supported host"),
    )
    .expect("launch JSON bound is nonzero and below the global asset limit");
    let source = load_deployment_asset(deployment_root, launch_relative_path, byte_limit)
        .map_err(NanoAgentLaunchLoadError::Load)?;
    let launch =
        NanoAgentLaunchV3::parse_json(source.bytes()).map_err(NanoAgentLaunchLoadError::Parse)?;
    for role in NanoLaunchAssetRole::ALL {
        if launch.asset(role).relative_path() == source.relative_path() {
            return Err(NanoAgentLaunchLoadError::InputAliasesLaunchDocument {
                role,
                relative_path: source.relative_path().clone(),
            });
        }
    }
    for role in NanoFaceCascadeAssetRole::ALL {
        if launch.face_perception().asset(role).relative_path() == source.relative_path() {
            return Err(NanoAgentLaunchLoadError::FaceInputAliasesLaunchDocument {
                role,
                relative_path: source.relative_path().clone(),
            });
        }
    }
    Ok(LoadedNanoAgentLaunchV3 { launch, source })
}

/// Load the canonical production V4 launch document beneath one deployment
/// root. Every referenced input remains separately exact-loaded by bootstrap.
pub fn load_nano_agent_launch_v4(
    deployment_root: &Path,
    launch_relative_path: ArtifactRelativePath,
) -> Result<LoadedNanoAgentLaunchV4, NanoAgentLaunchLoadError> {
    let byte_limit = DeploymentAssetByteLimit::try_new(
        u64::try_from(MAX_NANO_AGENT_LAUNCH_JSON_BYTES)
            .expect("launch JSON bound fits every supported host"),
    )
    .expect("launch JSON bound is nonzero and below the global asset limit");
    let source = load_deployment_asset(deployment_root, launch_relative_path, byte_limit)
        .map_err(NanoAgentLaunchLoadError::Load)?;
    let launch =
        NanoAgentLaunchV4::parse_json(source.bytes()).map_err(NanoAgentLaunchLoadError::Parse)?;
    for role in NanoLaunchAssetRole::ALL {
        if launch.asset(role).relative_path() == source.relative_path() {
            return Err(NanoAgentLaunchLoadError::InputAliasesLaunchDocument {
                role,
                relative_path: source.relative_path().clone(),
            });
        }
    }
    for role in NanoFaceCascadeAssetRole::ALL {
        if launch.face_perception().asset(role).relative_path() == source.relative_path() {
            return Err(NanoAgentLaunchLoadError::FaceInputAliasesLaunchDocument {
                role,
                relative_path: source.relative_path().clone(),
            });
        }
    }
    for role in NanoHeadGazeAssetRole::ALL {
        if launch.physical_head_gaze().asset(role).relative_path() == source.relative_path() {
            return Err(
                NanoAgentLaunchLoadError::HeadGazeInputAliasesLaunchDocument {
                    role,
                    relative_path: source.relative_path().clone(),
                },
            );
        }
    }
    Ok(LoadedNanoAgentLaunchV4 { launch, source })
}

#[derive(Debug)]
pub enum NanoAgentLaunchLoadError {
    Load(DeploymentAssetLoadError),
    Parse(NanoAgentLaunchParseError),
    InputAliasesLaunchDocument {
        role: NanoLaunchAssetRole,
        relative_path: ArtifactRelativePath,
    },
    FaceInputAliasesLaunchDocument {
        role: NanoFaceCascadeAssetRole,
        relative_path: ArtifactRelativePath,
    },
    HeadGazeInputAliasesLaunchDocument {
        role: NanoHeadGazeAssetRole,
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
            Self::FaceInputAliasesLaunchDocument {
                role,
                relative_path,
            } => write!(
                formatter,
                "{role:?} face-cascade asset aliases launch document {}",
                relative_path.as_str()
            ),
            Self::HeadGazeInputAliasesLaunchDocument {
                role,
                relative_path,
            } => write!(
                formatter,
                "{role:?} head-gaze asset aliases launch document {}",
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
            Self::InputAliasesLaunchDocument { .. }
            | Self::FaceInputAliasesLaunchDocument { .. }
            | Self::HeadGazeInputAliasesLaunchDocument { .. } => None,
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
    InvalidHeadGazeAssetPath {
        role: NanoHeadGazeAssetRole,
        source: ArtifactRelativePathError,
    },
    InvalidHeadGazeAssetByteLimit {
        role: NanoHeadGazeAssetRole,
        source: DeploymentAssetByteLimitError,
    },
    HeadGazeAssetByteLimitAboveMaximum {
        role: NanoHeadGazeAssetRole,
        actual_bytes: u64,
        maximum_bytes: u64,
    },
    InvalidHeadGazeAssetSha256 {
        role: NanoHeadGazeAssetRole,
        source: NanoLaunchSha256Error,
    },
    MissingFacePerception,
    DuplicateInputAssetPath {
        first: NanoLaunchAssetRole,
        second: NanoLaunchAssetRole,
        relative_path: ArtifactRelativePath,
    },
    DuplicateFaceAssetPath {
        relative_path: ArtifactRelativePath,
    },
    DuplicateFaceAssetContent {
        expected_sha256: [u8; 32],
    },
    FaceAssetAliasesInputAsset {
        face: NanoFaceCascadeAssetRole,
        input: NanoLaunchAssetRole,
        relative_path: ArtifactRelativePath,
    },
    DuplicateHeadGazeAssetPath {
        relative_path: ArtifactRelativePath,
    },
    DuplicateHeadGazeAssetContent {
        expected_sha256: [u8; 32],
    },
    HeadGazeAssetAliasesInputAsset {
        head_gaze: NanoHeadGazeAssetRole,
        input: NanoLaunchAssetRole,
        relative_path: ArtifactRelativePath,
    },
    HeadGazeAssetAliasesInputContent {
        head_gaze: NanoHeadGazeAssetRole,
        input: NanoLaunchAssetRole,
        expected_sha256: [u8; 32],
    },
    HeadGazeAssetAliasesFaceAsset {
        head_gaze: NanoHeadGazeAssetRole,
        face: NanoFaceCascadeAssetRole,
        relative_path: ArtifactRelativePath,
    },
    HeadGazeAssetAliasesFaceContent {
        head_gaze: NanoHeadGazeAssetRole,
        face: NanoFaceCascadeAssetRole,
        expected_sha256: [u8; 32],
    },
    InvalidPlantArtifactId,
    InvalidCalibrationArtifactId,
    InvalidSocket {
        field: &'static str,
        source: std::net::AddrParseError,
    },
    NonLoopbackSocket {
        field: &'static str,
        address: SocketAddr,
    },
    NonCanonicalRerunSocket {
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
    NavigationDatasetCountNotRepresentable {
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
    MissingStorageField {
        field: &'static str,
    },
    LegacyStorageField {
        field: &'static str,
    },
    OverlappingStoragePaths {
        first: &'static str,
        second: &'static str,
    },
    NavigationDatasetTerminalReserveNotBelowMaximum {
        reserve_bytes: u64,
        maximum_dataset_bytes: u64,
    },
    NavigationDatasetTerminalReserveTooSmall {
        reserve_bytes: u64,
        minimum_bytes: u64,
    },
    NavigationDatasetTerminalReserveArithmeticOverflow,
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
            Self::InvalidFaceAssetPath { source, .. }
            | Self::InvalidHeadGazeAssetPath { source, .. } => Some(source),
            Self::InvalidAssetByteLimit { source, .. }
            | Self::InvalidFaceAssetByteLimit { source, .. }
            | Self::InvalidHeadGazeAssetByteLimit { source, .. } => Some(source),
            Self::InvalidAssetSha256 { source, .. }
            | Self::InvalidFaceAssetSha256 { source, .. }
            | Self::InvalidHeadGazeAssetSha256 { source, .. } => Some(source),
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
struct NanoAgentLaunchV2Dto {
    schema_version: u32,
    agent_policy_asset: NanoLaunchAssetBindingDto,
    navigation_shadow_config_asset: NanoLaunchAssetBindingDto,
    physical_actuation_config_asset: NanoLaunchAssetBindingDto,
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
struct NanoAgentLaunchV3Dto {
    schema_version: u32,
    agent_policy_asset: NanoLaunchAssetBindingDto,
    navigation_shadow_config_asset: NanoLaunchAssetBindingDto,
    physical_actuation_config_asset: NanoLaunchAssetBindingDto,
    controller_server: NanoLaunchControllerServerDto,
    calibration_artifact: NanoLaunchCalibrationArtifactDto,
    plant_artifact: NanoLaunchPlantArtifactDto,
    oak: NanoOakStreamGraphDto,
    occupancy: NanoLaunchOccupancyDto,
    inference: NanoLaunchInferenceDto,
    face_perception: Option<NanoLaunchFacePerceptionDto>,
    rerun: NanoLaunchRerunDto,
    storage: NanoLaunchStorageDto,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoAgentLaunchV4Dto {
    schema_version: u32,
    agent_policy_asset: NanoLaunchAssetBindingDto,
    navigation_shadow_config_asset: NanoLaunchAssetBindingDto,
    physical_actuation_config_asset: NanoLaunchAssetBindingDto,
    controller_server: NanoLaunchControllerServerDto,
    calibration_artifact: NanoLaunchCalibrationArtifactDto,
    plant_artifact: NanoLaunchPlantArtifactDto,
    oak: NanoOakStreamGraphDto,
    occupancy: NanoLaunchOccupancyDto,
    inference: NanoLaunchInferenceDto,
    face_perception: Option<NanoLaunchFacePerceptionDto>,
    physical_head_gaze: NanoLaunchPhysicalHeadGazeDto,
    rerun: NanoLaunchRerunDto,
    storage: NanoLaunchStorageDto,
}

impl NanoAgentLaunchV4Dto {
    fn into_parts(self) -> (NanoAgentLaunchV3Dto, NanoLaunchPhysicalHeadGazeDto) {
        (
            NanoAgentLaunchV3Dto {
                schema_version: NANO_AGENT_LAUNCH_V3,
                agent_policy_asset: self.agent_policy_asset,
                navigation_shadow_config_asset: self.navigation_shadow_config_asset,
                physical_actuation_config_asset: self.physical_actuation_config_asset,
                controller_server: self.controller_server,
                calibration_artifact: self.calibration_artifact,
                plant_artifact: self.plant_artifact,
                oak: self.oak,
                occupancy: self.occupancy,
                inference: self.inference,
                face_perception: self.face_perception,
                rerun: self.rerun,
                storage: self.storage,
            },
            self.physical_head_gaze,
        )
    }
}

impl NanoAgentLaunchV3Dto {
    fn into_parts(self) -> (NanoAgentLaunchV2Dto, Option<NanoLaunchFacePerceptionDto>) {
        (
            NanoAgentLaunchV2Dto {
                schema_version: NANO_AGENT_LAUNCH_V2,
                agent_policy_asset: self.agent_policy_asset,
                navigation_shadow_config_asset: self.navigation_shadow_config_asset,
                physical_actuation_config_asset: self.physical_actuation_config_asset,
                controller_server: self.controller_server,
                calibration_artifact: self.calibration_artifact,
                plant_artifact: self.plant_artifact,
                oak: self.oak,
                occupancy: self.occupancy,
                inference: self.inference,
                rerun: self.rerun,
                storage: self.storage,
            },
            self.face_perception,
        )
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NanoLaunchAssetBindingDto {
    relative_path: String,
    maximum_bytes: u64,
    sha256_hex: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoLaunchFacePerceptionDto {
    frontal_face_cascade_asset: NanoLaunchAssetBindingDto,
    profile_face_cascade_asset: NanoLaunchAssetBindingDto,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoLaunchPhysicalHeadGazeDto {
    policy_asset: NanoLaunchAssetBindingDto,
    physical_review_evidence_asset: NanoLaunchAssetBindingDto,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NanoLaunchControllerServerDto {
    contract_asset: NanoLaunchAssetBindingDto,
    command_udp_endpoint: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NanoLaunchPlantArtifactDto {
    artifact_id: String,
    asset: NanoLaunchAssetBindingDto,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NanoLaunchCalibrationArtifactDto {
    artifact_id: String,
    asset: NanoLaunchAssetBindingDto,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NanoOakStreamGraphDto {
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
pub(crate) struct NanoOakImageStreamDto {
    width_px: u32,
    height_px: u32,
    fps: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NanoOakRectifiedStereoStreamDto {
    width_px: u32,
    height_px: u32,
    fps: u32,
    rectified: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NanoOakDepthStreamDto {
    width_px: u32,
    height_px: u32,
    fps: u32,
    alignment: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NanoOakImuStreamDto {
    rate_hz: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NanoOakQueueDto {
    size: u32,
    blocking: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NanoLaunchOccupancyDto {
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
pub(crate) struct NanoLaunchInferenceDto {
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
pub(crate) struct NanoLaunchRerunDto {
    kind: String,
    bind: String,
    decimation: u32,
    memory_limit_bytes: u64,
    flush_timeout_ms: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NanoLaunchStorageDto {
    map_snapshot_relative_path: String,
    navigation_dataset_directory_relative_path: Option<String>,
    maximum_map_snapshot_bytes: Option<u64>,
    minimum_free_bytes_after_map_save: Option<u64>,
    maximum_navigation_dataset_bytes: Option<u64>,
    maximum_navigation_dataset_files: Option<u64>,
    maximum_navigation_ingress_records: Option<u64>,
    minimum_free_bytes_after_navigation_dataset_write: Option<u64>,
    navigation_dataset_terminal_reserve_bytes: Option<u64>,
    navigation_records_relative_path: Option<String>,
    startup_evidence_relative_path: Option<String>,
    maximum_map_bytes: Option<u64>,
    maximum_navigation_record_bytes: Option<u64>,
    maximum_startup_evidence_bytes: Option<u64>,
    maximum_total_state_bytes: Option<u64>,
    minimum_free_bytes: Option<u64>,
}

pub(crate) fn parse_asset(
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

fn parse_face_perception(
    dto: NanoLaunchFacePerceptionDto,
) -> Result<NanoLaunchFacePerception, NanoAgentLaunchParseError> {
    Ok(NanoLaunchFacePerception {
        frontal_face_cascade: parse_face_asset(
            NanoFaceCascadeAssetRole::FrontalFace,
            dto.frontal_face_cascade_asset,
        )?,
        profile_face_cascade: parse_face_asset(
            NanoFaceCascadeAssetRole::ProfileFace,
            dto.profile_face_cascade_asset,
        )?,
    })
}

fn parse_face_asset(
    role: NanoFaceCascadeAssetRole,
    dto: NanoLaunchAssetBindingDto,
) -> Result<NanoLaunchAssetBinding, NanoAgentLaunchParseError> {
    let relative_path = ArtifactRelativePath::parse(dto.relative_path)
        .map_err(|source| NanoAgentLaunchParseError::InvalidFaceAssetPath { role, source })?;
    let byte_limit = DeploymentAssetByteLimit::try_new(dto.maximum_bytes)
        .map_err(|source| NanoAgentLaunchParseError::InvalidFaceAssetByteLimit { role, source })?;
    if byte_limit.get() > MAX_OPENCV_HAAR_CASCADE_BYTES {
        return Err(NanoAgentLaunchParseError::FaceAssetByteLimitAboveMaximum {
            role,
            actual_bytes: byte_limit.get(),
            maximum_bytes: MAX_OPENCV_HAAR_CASCADE_BYTES,
        });
    }
    let expected_sha256 = parse_sha256(&dto.sha256_hex)
        .map_err(|source| NanoAgentLaunchParseError::InvalidFaceAssetSha256 { role, source })?;
    Ok(NanoLaunchAssetBinding {
        relative_path,
        byte_limit,
        expected_sha256,
    })
}

fn parse_physical_head_gaze(
    dto: NanoLaunchPhysicalHeadGazeDto,
) -> Result<NanoLaunchPhysicalHeadGaze, NanoAgentLaunchParseError> {
    Ok(NanoLaunchPhysicalHeadGaze {
        policy: parse_head_gaze_asset(NanoHeadGazeAssetRole::Policy, dto.policy_asset)?,
        review_evidence: parse_head_gaze_asset(
            NanoHeadGazeAssetRole::PhysicalReviewEvidence,
            dto.physical_review_evidence_asset,
        )?,
    })
}

fn parse_head_gaze_asset(
    role: NanoHeadGazeAssetRole,
    dto: NanoLaunchAssetBindingDto,
) -> Result<NanoLaunchAssetBinding, NanoAgentLaunchParseError> {
    let relative_path = ArtifactRelativePath::parse(dto.relative_path)
        .map_err(|source| NanoAgentLaunchParseError::InvalidHeadGazeAssetPath { role, source })?;
    let byte_limit = DeploymentAssetByteLimit::try_new(dto.maximum_bytes).map_err(|source| {
        NanoAgentLaunchParseError::InvalidHeadGazeAssetByteLimit { role, source }
    })?;
    let role_maximum = role.maximum_bytes();
    if byte_limit.get() > role_maximum {
        return Err(
            NanoAgentLaunchParseError::HeadGazeAssetByteLimitAboveMaximum {
                role,
                actual_bytes: byte_limit.get(),
                maximum_bytes: role_maximum,
            },
        );
    }
    let expected_sha256 = parse_sha256(&dto.sha256_hex)
        .map_err(|source| NanoAgentLaunchParseError::InvalidHeadGazeAssetSha256 { role, source })?;
    Ok(NanoLaunchAssetBinding {
        relative_path,
        byte_limit,
        expected_sha256,
    })
}

pub(crate) fn parse_controller_server(
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

pub(crate) fn parse_plant_artifact(
    dto: NanoLaunchPlantArtifactDto,
) -> Result<NanoLaunchPlantArtifact, NanoAgentLaunchParseError> {
    if !valid_launch_artifact_id(&dto.artifact_id) {
        return Err(NanoAgentLaunchParseError::InvalidPlantArtifactId);
    }
    Ok(NanoLaunchPlantArtifact {
        artifact_id: NanoLaunchPlantArtifactId(dto.artifact_id),
        asset: parse_asset(NanoLaunchAssetRole::PlantArtifact, dto.asset)?,
    })
}

pub(crate) fn parse_calibration_artifact(
    dto: NanoLaunchCalibrationArtifactDto,
) -> Result<NanoLaunchCalibrationArtifact, NanoAgentLaunchParseError> {
    if !valid_launch_artifact_id(&dto.artifact_id) {
        return Err(NanoAgentLaunchParseError::InvalidCalibrationArtifactId);
    }
    Ok(NanoLaunchCalibrationArtifact {
        artifact_id: NanoLaunchCalibrationArtifactId(dto.artifact_id),
        asset: parse_asset(NanoLaunchAssetRole::CalibrationArtifact, dto.asset)?,
    })
}

fn valid_launch_artifact_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_LAUNCH_ARTIFACT_ID_BYTES
        && !value.bytes().all(|byte| byte == b'0')
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
}

pub(crate) fn parse_oak(
    dto: NanoOakStreamGraphDto,
) -> Result<NanoOakStreamGraph, NanoAgentLaunchParseError> {
    if dto.selector_source != "exact_inventory_oak_mxid" {
        return Err(NanoAgentLaunchParseError::UnsupportedOakSelectorSource);
    }
    let maximum_usb_speed = UsbTransportSpeed::parse(&dto.maximum_usb_speed)
        .map_err(|_| NanoAgentLaunchParseError::ProductionOakUsbPolicyRequired)?;
    let minimum_usb_speed = UsbTransportSpeed::parse(&dto.minimum_usb_speed)
        .map_err(|_| NanoAgentLaunchParseError::ProductionOakUsbPolicyRequired)?;
    let usb_transport = UsbTransportPolicy::try_new(maximum_usb_speed, minimum_usb_speed)
        .map_err(|_| NanoAgentLaunchParseError::ProductionOakUsbPolicyRequired)?;
    if minimum_usb_speed != UsbTransportSpeed::Super
        || !matches!(
            maximum_usb_speed,
            UsbTransportSpeed::Super | UsbTransportSpeed::SuperPlus
        )
    {
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
        usb_transport,
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

pub(crate) fn parse_occupancy(
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

fn bounded_navigation_dataset_record_count(
    field: &'static str,
    value: u64,
    maximum: usize,
) -> Result<NonZeroUsize, NanoAgentLaunchParseError> {
    let converted = usize::try_from(value).map_err(|_| {
        NanoAgentLaunchParseError::NavigationDatasetCountNotRepresentable { field, value }
    })?;
    if converted == 0 || converted > maximum {
        return Err(NanoAgentLaunchParseError::NumericOutOfRange {
            field,
            value,
            minimum: 1,
            maximum: u64::try_from(maximum).expect("dataset record bound fits u64"),
        });
    }
    Ok(NonZeroUsize::new(converted).expect("nonzero checked above"))
}

pub(crate) fn parse_inference(
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

pub(crate) fn parse_rerun(
    dto: NanoLaunchRerunDto,
) -> Result<NanoLaunchRerun, NanoAgentLaunchParseError> {
    if dto.kind != "serve_loopback" {
        return Err(NanoAgentLaunchParseError::UnsupportedRerunKind);
    }
    Ok(NanoLaunchRerun {
        diagnostics_url: parse_rerun_diagnostics_url(dto.bind)?,
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

pub(crate) fn parse_storage(
    dto: NanoLaunchStorageDto,
) -> Result<NanoLaunchStorage, NanoAgentLaunchParseError> {
    for (field, present) in [
        (
            "storage.navigation_records_relative_path",
            dto.navigation_records_relative_path.is_some(),
        ),
        (
            "storage.startup_evidence_relative_path",
            dto.startup_evidence_relative_path.is_some(),
        ),
        ("storage.maximum_map_bytes", dto.maximum_map_bytes.is_some()),
        (
            "storage.maximum_navigation_record_bytes",
            dto.maximum_navigation_record_bytes.is_some(),
        ),
        (
            "storage.maximum_startup_evidence_bytes",
            dto.maximum_startup_evidence_bytes.is_some(),
        ),
        (
            "storage.maximum_total_state_bytes",
            dto.maximum_total_state_bytes.is_some(),
        ),
        (
            "storage.minimum_free_bytes",
            dto.minimum_free_bytes.is_some(),
        ),
    ] {
        if present {
            return Err(NanoAgentLaunchParseError::LegacyStorageField { field });
        }
    }
    let map_snapshot = parse_storage_path(
        "storage.map_snapshot_relative_path",
        dto.map_snapshot_relative_path,
    )?;
    let navigation_dataset_directory = parse_storage_path(
        "storage.navigation_dataset_directory_relative_path",
        dto.navigation_dataset_directory_relative_path.ok_or(
            NanoAgentLaunchParseError::MissingStorageField {
                field: "storage.navigation_dataset_directory_relative_path",
            },
        )?,
    )?;
    ensure_nonoverlapping_storage_paths(&[
        ("map_snapshot", &map_snapshot),
        (
            "navigation_dataset_directory",
            &navigation_dataset_directory,
        ),
    ])?;

    let maximum_map_snapshot_bytes = bounded_nonzero_u64(
        "storage.maximum_map_snapshot_bytes",
        dto.maximum_map_snapshot_bytes
            .ok_or(NanoAgentLaunchParseError::MissingStorageField {
                field: "storage.maximum_map_snapshot_bytes",
            })?,
        1,
        MAX_NANO_STATE_BYTES,
    )?;
    let minimum_free_bytes_after_map_save = bounded_nonzero_u64(
        "storage.minimum_free_bytes_after_map_save",
        dto.minimum_free_bytes_after_map_save.ok_or(
            NanoAgentLaunchParseError::MissingStorageField {
                field: "storage.minimum_free_bytes_after_map_save",
            },
        )?,
        1,
        MAX_NANO_STATE_BYTES,
    )?;
    let maximum_navigation_dataset_bytes = bounded_nonzero_u64(
        "storage.maximum_navigation_dataset_bytes",
        dto.maximum_navigation_dataset_bytes.ok_or(
            NanoAgentLaunchParseError::MissingStorageField {
                field: "storage.maximum_navigation_dataset_bytes",
            },
        )?,
        1,
        MAX_NANO_STATE_BYTES,
    )?;
    let maximum_navigation_dataset_files = bounded_nonzero_u64(
        "storage.maximum_navigation_dataset_files",
        dto.maximum_navigation_dataset_files.ok_or(
            NanoAgentLaunchParseError::MissingStorageField {
                field: "storage.maximum_navigation_dataset_files",
            },
        )?,
        1,
        MAX_NANO_NAVIGATION_DATASET_FILES,
    )?;
    let maximum_navigation_ingress_records_u64 = dto.maximum_navigation_ingress_records.ok_or(
        NanoAgentLaunchParseError::MissingStorageField {
            field: "storage.maximum_navigation_ingress_records",
        },
    )?;
    let maximum_navigation_ingress_records = bounded_navigation_dataset_record_count(
        "storage.maximum_navigation_ingress_records",
        maximum_navigation_ingress_records_u64,
        MAX_NAVIGATION_INGRESS_RECORDS,
    )?;
    let minimum_free_bytes_after_navigation_dataset_write = bounded_nonzero_u64(
        "storage.minimum_free_bytes_after_navigation_dataset_write",
        dto.minimum_free_bytes_after_navigation_dataset_write
            .ok_or(NanoAgentLaunchParseError::MissingStorageField {
                field: "storage.minimum_free_bytes_after_navigation_dataset_write",
            })?,
        1,
        MAX_NANO_STATE_BYTES,
    )?;
    let navigation_dataset_terminal_reserve_bytes = bounded_nonzero_u64(
        "storage.navigation_dataset_terminal_reserve_bytes",
        dto.navigation_dataset_terminal_reserve_bytes.ok_or(
            NanoAgentLaunchParseError::MissingStorageField {
                field: "storage.navigation_dataset_terminal_reserve_bytes",
            },
        )?,
        1,
        MAX_NANO_STATE_BYTES,
    )?;
    if navigation_dataset_terminal_reserve_bytes.get() >= maximum_navigation_dataset_bytes.get() {
        return Err(
            NanoAgentLaunchParseError::NavigationDatasetTerminalReserveNotBelowMaximum {
                reserve_bytes: navigation_dataset_terminal_reserve_bytes.get(),
                maximum_dataset_bytes: maximum_navigation_dataset_bytes.get(),
            },
        );
    }
    let minimum_terminal_reserve =
        minimum_navigation_dataset_terminal_reserve(maximum_map_snapshot_bytes.get())?;
    if navigation_dataset_terminal_reserve_bytes.get() < minimum_terminal_reserve {
        return Err(
            NanoAgentLaunchParseError::NavigationDatasetTerminalReserveTooSmall {
                reserve_bytes: navigation_dataset_terminal_reserve_bytes.get(),
                minimum_bytes: minimum_terminal_reserve,
            },
        );
    }
    Ok(NanoLaunchStorage {
        map_snapshot,
        navigation_dataset_directory,
        maximum_map_snapshot_bytes,
        minimum_free_bytes_after_map_save,
        navigation_dataset_limits: NanoNavigationDatasetStorageLimits {
            maximum_bytes: maximum_navigation_dataset_bytes,
            maximum_files: maximum_navigation_dataset_files,
            maximum_ingress_records: maximum_navigation_ingress_records,
            minimum_free_bytes_after_write: minimum_free_bytes_after_navigation_dataset_write,
            terminal_reserve_bytes: navigation_dataset_terminal_reserve_bytes,
        },
    })
}

fn minimum_navigation_dataset_terminal_reserve(
    maximum_map_snapshot_bytes: u64,
) -> Result<u64, NanoAgentLaunchParseError> {
    let unrounded = maximum_map_snapshot_bytes
        .checked_add(MAX_NAVIGATION_DATASET_MANIFEST_BYTES)
        .and_then(|bytes| bytes.checked_add(MAX_WARM_START_SELECTION_BYTES))
        .ok_or(NanoAgentLaunchParseError::NavigationDatasetTerminalReserveArithmeticOverflow)?;
    let remainder = unrounded % NAVIGATION_DATASET_ADMISSION_FRAGMENT_BYTES;
    if remainder == 0 {
        return Ok(unrounded);
    }
    unrounded
        .checked_add(NAVIGATION_DATASET_ADMISSION_FRAGMENT_BYTES - remainder)
        .ok_or(NanoAgentLaunchParseError::NavigationDatasetTerminalReserveArithmeticOverflow)
}

fn parse_storage_path(
    field: &'static str,
    value: String,
) -> Result<ArtifactRelativePath, NanoAgentLaunchParseError> {
    ArtifactRelativePath::parse(value)
        .map_err(|source| NanoAgentLaunchParseError::InvalidStoragePath { field, source })
}

fn ensure_nonoverlapping_storage_paths(
    paths: &[(&'static str, &ArtifactRelativePath)],
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
    launch: &NanoAgentLaunchV2,
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

fn ensure_distinct_v3_input_assets(
    launch: &NanoAgentLaunchV3,
) -> Result<(), NanoAgentLaunchParseError> {
    let frontal = launch
        .face_perception()
        .asset(NanoFaceCascadeAssetRole::FrontalFace);
    let profile = launch
        .face_perception()
        .asset(NanoFaceCascadeAssetRole::ProfileFace);
    if frontal.relative_path() == profile.relative_path() {
        return Err(NanoAgentLaunchParseError::DuplicateFaceAssetPath {
            relative_path: frontal.relative_path().clone(),
        });
    }
    if frontal.expected_sha256() == profile.expected_sha256() {
        return Err(NanoAgentLaunchParseError::DuplicateFaceAssetContent {
            expected_sha256: *frontal.expected_sha256(),
        });
    }
    for face in NanoFaceCascadeAssetRole::ALL {
        let face_asset = launch.face_perception().asset(face);
        for input in NanoLaunchAssetRole::ALL {
            if face_asset.relative_path() == launch.asset(input).relative_path() {
                return Err(NanoAgentLaunchParseError::FaceAssetAliasesInputAsset {
                    face,
                    input,
                    relative_path: face_asset.relative_path().clone(),
                });
            }
        }
    }
    Ok(())
}

fn ensure_distinct_v4_input_assets(
    launch: &NanoAgentLaunchV4,
) -> Result<(), NanoAgentLaunchParseError> {
    let policy = launch
        .physical_head_gaze()
        .asset(NanoHeadGazeAssetRole::Policy);
    let review = launch
        .physical_head_gaze()
        .asset(NanoHeadGazeAssetRole::PhysicalReviewEvidence);
    if policy.relative_path() == review.relative_path() {
        return Err(NanoAgentLaunchParseError::DuplicateHeadGazeAssetPath {
            relative_path: policy.relative_path().clone(),
        });
    }
    if policy.expected_sha256() == review.expected_sha256() {
        return Err(NanoAgentLaunchParseError::DuplicateHeadGazeAssetContent {
            expected_sha256: *policy.expected_sha256(),
        });
    }
    for head_gaze in NanoHeadGazeAssetRole::ALL {
        let head_gaze_asset = launch.physical_head_gaze().asset(head_gaze);
        for input in NanoLaunchAssetRole::ALL {
            if head_gaze_asset.relative_path() == launch.asset(input).relative_path() {
                return Err(NanoAgentLaunchParseError::HeadGazeAssetAliasesInputAsset {
                    head_gaze,
                    input,
                    relative_path: head_gaze_asset.relative_path().clone(),
                });
            }
            if head_gaze_asset.expected_sha256() == launch.asset(input).expected_sha256() {
                return Err(
                    NanoAgentLaunchParseError::HeadGazeAssetAliasesInputContent {
                        head_gaze,
                        input,
                        expected_sha256: *head_gaze_asset.expected_sha256(),
                    },
                );
            }
        }
        for face in NanoFaceCascadeAssetRole::ALL {
            if head_gaze_asset.relative_path()
                == launch.face_perception().asset(face).relative_path()
            {
                return Err(NanoAgentLaunchParseError::HeadGazeAssetAliasesFaceAsset {
                    head_gaze,
                    face,
                    relative_path: head_gaze_asset.relative_path().clone(),
                });
            }
            if head_gaze_asset.expected_sha256()
                == launch.face_perception().asset(face).expected_sha256()
            {
                return Err(NanoAgentLaunchParseError::HeadGazeAssetAliasesFaceContent {
                    head_gaze,
                    face,
                    expected_sha256: *head_gaze_asset.expected_sha256(),
                });
            }
        }
    }
    Ok(())
}

fn parse_rerun_diagnostics_url(
    value: String,
) -> Result<ConsoleRerunDiagnosticsUrl, NanoAgentLaunchParseError> {
    let address =
        value
            .parse::<SocketAddr>()
            .map_err(|source| NanoAgentLaunchParseError::InvalidSocket {
                field: "rerun.bind",
                source,
            })?;
    if address.ip() != IpAddr::V4(Ipv4Addr::LOCALHOST) {
        return Err(NanoAgentLaunchParseError::NonCanonicalRerunSocket { address });
    }
    let forwarded_port =
        NonZeroU16::new(address.port()).ok_or(NanoAgentLaunchParseError::ZeroSocketPort {
            field: "rerun.bind",
        })?;
    Ok(ConsoleRerunDiagnosticsUrl::from_admitted_forwarded_port(
        forwarded_port,
    ))
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

pub(crate) fn parse_sha256(value: &str) -> Result<[u8; 32], NanoLaunchSha256Error> {
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
    use std::path::{Path, PathBuf};
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
            "schema_version": 2,
            "agent_policy_asset": asset("config/agent-policy-v3.json", 65_536, 1),
            "navigation_shadow_config_asset": asset(
                "config/navigation-shadow-v2.json",
                262_144,
                2
            ),
            "physical_actuation_config_asset": asset(
                "config/navigation-actuation-v2.json",
                16_384,
                3
            ),
            "controller_server": {
                "contract_asset": asset("config/controller-server-v1.json", 8_192, 4),
                "command_udp_endpoint": "127.0.0.1:8080"
            },
            "calibration_artifact": {
                "artifact_id": "kiko-calibration-v1",
                "asset": asset(
                    "artifacts/calibration/kiko-calibration-v1.json",
                    65_536,
                    9
                )
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
                "navigation_dataset_directory_relative_path": "records/navigation",
                "maximum_map_snapshot_bytes": 67_108_864,
                "minimum_free_bytes_after_map_save": 134_217_728,
                "maximum_navigation_dataset_bytes": 8_589_934_592_u64,
                "maximum_navigation_dataset_files": 65_536,
                "maximum_navigation_ingress_records": 100_000,
                "minimum_free_bytes_after_navigation_dataset_write": 1_073_741_824,
                "navigation_dataset_terminal_reserve_bytes": 268_435_456
            }
        })
    }

    fn valid_v3_value() -> Value {
        let mut value = valid_value();
        value["schema_version"] = json!(NANO_AGENT_LAUNCH_V3);
        value["face_perception"] = json!({
            "frontal_face_cascade_asset": asset(
                "models/opencv/haarcascade_frontalface_default.xml",
                1_048_576,
                10
            ),
            "profile_face_cascade_asset": asset(
                "models/opencv/haarcascade_profileface.xml",
                1_048_576,
                11
            )
        });
        value
    }

    fn valid_v4_value() -> Value {
        let mut value = valid_v3_value();
        value["schema_version"] = json!(NANO_AGENT_LAUNCH_V4);
        value["physical_head_gaze"] = json!({
            "policy_asset": asset("head-gaze-policy-v1.json", 16_384, 12),
            "physical_review_evidence_asset": asset(
                "evidence/head-gaze-physical-review-v1.json",
                65_536,
                13
            )
        });
        value
    }

    fn parse(value: &Value) -> Result<NanoAgentLaunchV2, NanoAgentLaunchParseError> {
        NanoAgentLaunchV2::parse_json(&serde_json::to_vec(value).expect("fixture serializes"))
    }

    fn parse_v3(value: &Value) -> Result<NanoAgentLaunchV3, NanoAgentLaunchParseError> {
        NanoAgentLaunchV3::parse_json(&serde_json::to_vec(value).expect("fixture serializes"))
    }

    fn parse_v4(value: &Value) -> Result<NanoAgentLaunchV4, NanoAgentLaunchParseError> {
        NanoAgentLaunchV4::parse_json(&serde_json::to_vec(value).expect("fixture serializes"))
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
        assert_eq!(
            launch.storage().navigation_dataset_directory().as_path(),
            Path::new("records/navigation")
        );
        assert_eq!(launch.storage().maximum_map_snapshot_bytes(), 67_108_864);
        assert_eq!(
            launch.storage().minimum_free_bytes_after_map_save(),
            134_217_728
        );
        let dataset_limits = launch.storage().navigation_dataset_limits();
        assert_eq!(dataset_limits.maximum_bytes(), 8_589_934_592);
        assert_eq!(dataset_limits.maximum_files(), 65_536);
        assert_eq!(dataset_limits.maximum_ingress_records(), 100_000);
        assert_eq!(
            dataset_limits.minimum_free_bytes_after_write(),
            1_073_741_824
        );
        assert_eq!(dataset_limits.terminal_reserve_bytes(), 268_435_456);
    }

    #[test]
    fn v3_requires_two_exact_distinct_face_cascades_without_changing_v2() {
        let v2 = parse(&valid_value()).expect("unchanged V2 launch");
        assert_eq!(
            v2.agent_policy().relative_path().as_str(),
            "config/agent-policy-v3.json"
        );

        let launch = parse_v3(&valid_v3_value()).expect("valid V3 launch");
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
                .face_perception()
                .frontal_face_cascade()
                .byte_limit()
                .get(),
            1_048_576
        );

        let mut missing = valid_v3_value();
        missing
            .as_object_mut()
            .expect("top-level fixture")
            .remove("face_perception");
        assert!(matches!(
            parse_v3(&missing),
            Err(NanoAgentLaunchParseError::MissingFacePerception)
        ));

        assert!(matches!(
            parse_v3(&valid_value()),
            Err(NanoAgentLaunchParseError::UnsupportedSchema {
                actual: NANO_AGENT_LAUNCH_V2,
                supported: NANO_AGENT_LAUNCH_V3
            })
        ));
    }

    #[test]
    fn v3_face_assets_are_canonical_bounded_and_do_not_alias_any_input() {
        let mut traversal = valid_v3_value();
        traversal["face_perception"]["frontal_face_cascade_asset"]["relative_path"] =
            json!("../frontal.xml");
        assert!(matches!(
            parse_v3(&traversal),
            Err(NanoAgentLaunchParseError::InvalidFaceAssetPath {
                role: NanoFaceCascadeAssetRole::FrontalFace,
                ..
            })
        ));

        let mut oversized = valid_v3_value();
        oversized["face_perception"]["profile_face_cascade_asset"]["maximum_bytes"] =
            json!(MAX_OPENCV_HAAR_CASCADE_BYTES + 1);
        assert!(matches!(
            parse_v3(&oversized),
            Err(NanoAgentLaunchParseError::FaceAssetByteLimitAboveMaximum {
                role: NanoFaceCascadeAssetRole::ProfileFace,
                ..
            })
        ));

        let mut same_face_path = valid_v3_value();
        same_face_path["face_perception"]["profile_face_cascade_asset"]["relative_path"] =
            same_face_path["face_perception"]["frontal_face_cascade_asset"]["relative_path"]
                .clone();
        assert!(matches!(
            parse_v3(&same_face_path),
            Err(NanoAgentLaunchParseError::DuplicateFaceAssetPath { .. })
        ));

        let mut same_face_content = valid_v3_value();
        same_face_content["face_perception"]["profile_face_cascade_asset"]["sha256_hex"] =
            same_face_content["face_perception"]["frontal_face_cascade_asset"]["sha256_hex"]
                .clone();
        assert!(matches!(
            parse_v3(&same_face_content),
            Err(NanoAgentLaunchParseError::DuplicateFaceAssetContent { .. })
        ));

        let mut aliases_existing = valid_v3_value();
        aliases_existing["face_perception"]["frontal_face_cascade_asset"]["relative_path"] =
            aliases_existing["inference"]["superpoint_model_asset"]["relative_path"].clone();
        assert!(matches!(
            parse_v3(&aliases_existing),
            Err(NanoAgentLaunchParseError::FaceAssetAliasesInputAsset {
                face: NanoFaceCascadeAssetRole::FrontalFace,
                input: NanoLaunchAssetRole::SuperpointModel,
                ..
            })
        ));
    }

    #[test]
    fn v4_requires_distinct_bounded_head_gaze_policy_and_review_evidence() {
        let launch = parse_v4(&valid_v4_value()).expect("valid V4 launch");
        assert_eq!(
            launch
                .physical_head_gaze()
                .policy()
                .relative_path()
                .as_str(),
            "head-gaze-policy-v1.json"
        );
        assert_eq!(
            launch
                .physical_head_gaze()
                .review_evidence()
                .relative_path()
                .as_str(),
            "evidence/head-gaze-physical-review-v1.json"
        );

        let mut missing = valid_v4_value();
        missing
            .as_object_mut()
            .expect("top-level fixture")
            .remove("physical_head_gaze");
        assert!(matches!(
            parse_v4(&missing),
            Err(NanoAgentLaunchParseError::JsonDecode(_))
        ));

        let mut oversized = valid_v4_value();
        oversized["physical_head_gaze"]["policy_asset"]["maximum_bytes"] =
            json!(MAX_HEAD_GAZE_POLICY_JSON_BYTES as u64 + 1);
        assert!(matches!(
            parse_v4(&oversized),
            Err(
                NanoAgentLaunchParseError::HeadGazeAssetByteLimitAboveMaximum {
                    role: NanoHeadGazeAssetRole::Policy,
                    ..
                }
            )
        ));

        let mut same_path = valid_v4_value();
        same_path["physical_head_gaze"]["physical_review_evidence_asset"]["relative_path"] =
            same_path["physical_head_gaze"]["policy_asset"]["relative_path"].clone();
        assert!(matches!(
            parse_v4(&same_path),
            Err(NanoAgentLaunchParseError::DuplicateHeadGazeAssetPath { .. })
        ));

        let mut same_content = valid_v4_value();
        same_content["physical_head_gaze"]["physical_review_evidence_asset"]["sha256_hex"] =
            same_content["physical_head_gaze"]["policy_asset"]["sha256_hex"].clone();
        assert!(matches!(
            parse_v4(&same_content),
            Err(NanoAgentLaunchParseError::DuplicateHeadGazeAssetContent { .. })
        ));

        let mut aliases_face = valid_v4_value();
        aliases_face["physical_head_gaze"]["policy_asset"]["relative_path"] =
            aliases_face["face_perception"]["frontal_face_cascade_asset"]["relative_path"].clone();
        assert!(matches!(
            parse_v4(&aliases_face),
            Err(NanoAgentLaunchParseError::HeadGazeAssetAliasesFaceAsset {
                head_gaze: NanoHeadGazeAssetRole::Policy,
                face: NanoFaceCascadeAssetRole::FrontalFace,
                ..
            })
        ));

        let mut aliases_input_content = valid_v4_value();
        aliases_input_content["physical_head_gaze"]["policy_asset"]["sha256_hex"] =
            aliases_input_content["agent_policy_asset"]["sha256_hex"].clone();
        assert!(matches!(
            parse_v4(&aliases_input_content),
            Err(
                NanoAgentLaunchParseError::HeadGazeAssetAliasesInputContent {
                    head_gaze: NanoHeadGazeAssetRole::Policy,
                    input: NanoLaunchAssetRole::AgentPolicy,
                    ..
                }
            )
        ));

        let mut aliases_face_content = valid_v4_value();
        aliases_face_content["physical_head_gaze"]["policy_asset"]["sha256_hex"] =
            aliases_face_content["face_perception"]["frontal_face_cascade_asset"]["sha256_hex"]
                .clone();
        assert!(matches!(
            parse_v4(&aliases_face_content),
            Err(NanoAgentLaunchParseError::HeadGazeAssetAliasesFaceContent {
                head_gaze: NanoHeadGazeAssetRole::Policy,
                face: NanoFaceCascadeAssetRole::FrontalFace,
                ..
            })
        ));
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
            NanoAgentLaunchV2::parse_json(&bytes),
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
        let production = parse(&valid_value()).expect("production USB 3 graph parses");
        assert_eq!(
            production.oak().usb_transport(),
            UsbTransportPolicy::super_speed_required()
        );

        let mut explicit_super_plus = valid_value();
        explicit_super_plus["oak"]["maximum_usb_speed"] = json!("SUPER_PLUS");
        let explicit_super_plus =
            parse(&explicit_super_plus).expect("retained explicit 10 Gbit/s input remains valid");
        assert_eq!(
            explicit_super_plus.oak().usb_transport(),
            UsbTransportPolicy::try_new(UsbTransportSpeed::SuperPlus, UsbTransportSpeed::Super)
                .expect("ordered explicit 10 Gbit/s policy")
        );

        let mut usb2 = valid_value();
        usb2["oak"]["minimum_usb_speed"] = json!("HIGH");
        assert!(matches!(
            parse(&usb2),
            Err(NanoAgentLaunchParseError::ProductionOakUsbPolicyRequired)
        ));

        let mut forced_usb2 = valid_value();
        forced_usb2["oak"]["maximum_usb_speed"] = json!("HIGH");
        forced_usb2["oak"]["minimum_usb_speed"] = json!("HIGH");
        assert!(matches!(
            parse(&forced_usb2),
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

        for address in ["[::]:9876", "[::1]:9876", "127.0.0.2:9876"] {
            let mut noncanonical_rerun = valid_value();
            noncanonical_rerun["rerun"]["bind"] = json!(address);
            assert!(matches!(
                parse(&noncanonical_rerun),
                Err(NanoAgentLaunchParseError::NonCanonicalRerunSocket { .. })
            ));
        }

        let parsed = parse(&valid_value()).expect("canonical launch");
        assert_eq!(parsed.rerun().bind(), "127.0.0.1:9876".parse().unwrap());
        assert_eq!(
            parsed.rerun().diagnostics_url().to_string(),
            "rerun+http://127.0.0.1:9876/proxy"
        );

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
    fn state_paths_do_not_overlap_and_legacy_quota_fields_are_rejected() {
        let mut overlap = valid_value();
        overlap["storage"]["navigation_dataset_directory_relative_path"] = json!("maps");
        assert!(matches!(
            parse(&overlap),
            Err(NanoAgentLaunchParseError::OverlappingStoragePaths { .. })
        ));

        let mut stale_contract = valid_value();
        stale_contract["storage"]["maximum_navigation_record_bytes"] = json!(100);
        assert!(matches!(
            parse(&stale_contract),
            Err(NanoAgentLaunchParseError::LegacyStorageField {
                field: "storage.maximum_navigation_record_bytes"
            })
        ));

        for (field, qualified_field) in [
            (
                "maximum_map_snapshot_bytes",
                "storage.maximum_map_snapshot_bytes",
            ),
            (
                "minimum_free_bytes_after_map_save",
                "storage.minimum_free_bytes_after_map_save",
            ),
            (
                "maximum_navigation_dataset_bytes",
                "storage.maximum_navigation_dataset_bytes",
            ),
            (
                "maximum_navigation_dataset_files",
                "storage.maximum_navigation_dataset_files",
            ),
            (
                "maximum_navigation_ingress_records",
                "storage.maximum_navigation_ingress_records",
            ),
            (
                "minimum_free_bytes_after_navigation_dataset_write",
                "storage.minimum_free_bytes_after_navigation_dataset_write",
            ),
            (
                "navigation_dataset_terminal_reserve_bytes",
                "storage.navigation_dataset_terminal_reserve_bytes",
            ),
        ] {
            let mut missing = valid_value();
            missing["storage"]
                .as_object_mut()
                .expect("storage object")
                .remove(field);
            assert!(matches!(
                parse(&missing),
                Err(NanoAgentLaunchParseError::MissingStorageField {
                    field: actual
                }) if actual == qualified_field
            ));

            let mut zero = valid_value();
            zero["storage"][field] = json!(0);
            assert!(matches!(
                parse(&zero),
                Err(NanoAgentLaunchParseError::NumericOutOfRange { .. })
            ));
        }
    }

    #[test]
    fn navigation_dataset_limits_are_bounded_and_terminal_reserve_is_checked() {
        assert!(matches!(
            minimum_navigation_dataset_terminal_reserve(u64::MAX),
            Err(NanoAgentLaunchParseError::NavigationDatasetTerminalReserveArithmeticOverflow)
        ));

        let mut too_many_files = valid_value();
        too_many_files["storage"]["maximum_navigation_dataset_files"] =
            json!(MAX_NANO_NAVIGATION_DATASET_FILES + 1);
        assert!(matches!(
            parse(&too_many_files),
            Err(NanoAgentLaunchParseError::NumericOutOfRange {
                field: "storage.maximum_navigation_dataset_files",
                ..
            })
        ));

        let mut too_many_records = valid_value();
        too_many_records["storage"]["maximum_navigation_ingress_records"] =
            json!(u64::try_from(MAX_NAVIGATION_INGRESS_RECORDS).expect("bound fits u64") + 1);
        assert!(matches!(
            parse(&too_many_records),
            Err(NanoAgentLaunchParseError::NumericOutOfRange {
                field: "storage.maximum_navigation_ingress_records",
                ..
            })
        ));

        let mut reserve_reaches_maximum = valid_value();
        reserve_reaches_maximum["storage"]["navigation_dataset_terminal_reserve_bytes"] =
            reserve_reaches_maximum["storage"]["maximum_navigation_dataset_bytes"].clone();
        assert!(matches!(
            parse(&reserve_reaches_maximum),
            Err(NanoAgentLaunchParseError::NavigationDatasetTerminalReserveNotBelowMaximum { .. })
        ));

        let expected_minimum = 134_221_824;
        let mut one_byte_short = valid_value();
        one_byte_short["storage"]["navigation_dataset_terminal_reserve_bytes"] =
            json!(expected_minimum - 1);
        assert!(matches!(
            parse(&one_byte_short),
            Err(NanoAgentLaunchParseError::NavigationDatasetTerminalReserveTooSmall {
                reserve_bytes,
                minimum_bytes
            }) if reserve_bytes == expected_minimum - 1 && minimum_bytes == expected_minimum
        ));

        let mut exact_minimum = valid_value();
        exact_minimum["storage"]["navigation_dataset_terminal_reserve_bytes"] =
            json!(expected_minimum);
        let parsed = parse(&exact_minimum).expect("fragment-rounded minimum is admitted");
        assert_eq!(
            parsed
                .storage()
                .navigation_dataset_limits()
                .terminal_reserve_bytes(),
            expected_minimum
        );

        let mut unaligned_map = valid_value();
        unaligned_map["storage"]["maximum_map_snapshot_bytes"] = json!(67_108_865);
        unaligned_map["storage"]["navigation_dataset_terminal_reserve_bytes"] = json!(134_225_919);
        assert!(matches!(
            parse(&unaligned_map),
            Err(
                NanoAgentLaunchParseError::NavigationDatasetTerminalReserveTooSmall {
                    minimum_bytes: 134_225_920,
                    ..
                }
            )
        ));
    }

    #[test]
    fn actual_v1_storage_shape_is_a_typed_unsupported_schema() {
        let mut legacy = valid_value();
        legacy["schema_version"] = json!(1);
        let storage = legacy["storage"].as_object_mut().expect("storage object");
        storage.remove("navigation_dataset_directory_relative_path");
        storage.remove("maximum_map_snapshot_bytes");
        storage.remove("minimum_free_bytes_after_map_save");
        storage.remove("maximum_navigation_dataset_bytes");
        storage.remove("maximum_navigation_dataset_files");
        storage.remove("maximum_navigation_ingress_records");
        storage.remove("minimum_free_bytes_after_navigation_dataset_write");
        storage.remove("navigation_dataset_terminal_reserve_bytes");
        storage.insert(
            "navigation_records_relative_path".to_owned(),
            json!("records/navigation"),
        );
        storage.insert(
            "startup_evidence_relative_path".to_owned(),
            json!("evidence/startup"),
        );
        storage.insert("maximum_map_bytes".to_owned(), json!(67_108_864));
        storage.insert(
            "maximum_navigation_record_bytes".to_owned(),
            json!(536_870_912),
        );
        storage.insert(
            "maximum_startup_evidence_bytes".to_owned(),
            json!(16_777_216),
        );
        storage.insert("maximum_total_state_bytes".to_owned(), json!(1_073_741_824));
        storage.insert("minimum_free_bytes".to_owned(), json!(134_217_728));

        assert!(matches!(
            parse(&legacy),
            Err(NanoAgentLaunchParseError::UnsupportedSchema {
                actual: 1,
                supported: 2
            })
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
        assert!(matches!(
            binding.verify_exact_streaming(&directory),
            Err(NanoLaunchBoundAssetLoadError::ContentMismatch { .. })
        ));
        let exact = NanoLaunchAssetBinding {
            expected_sha256: Sha256::digest(b"actual").into(),
            ..binding
        };
        let identity = exact
            .verify_exact_streaming(&directory)
            .expect("streaming exact binding");
        assert_eq!(identity.byte_len(), 6);
        assert_eq!(identity.relative_path().as_str(), "config/policy.json");
        fs::remove_dir_all(directory).expect("remove test directory");
    }

    #[test]
    fn v3_face_bindings_retain_exact_bytes_and_deployment_relative_identity() {
        let requested_directory = unique_test_directory("face-assets");
        let model_directory = requested_directory.join("models/opencv");
        fs::create_dir_all(&model_directory).expect("create model directory");
        let directory =
            fs::canonicalize(requested_directory).expect("canonicalize test directory root");
        let frontal_bytes = b"frontal cascade fixture";
        let profile_bytes = b"profile cascade fixture";
        fs::write(directory.join("models/opencv/frontal.xml"), frontal_bytes)
            .expect("write frontal fixture");
        fs::write(directory.join("models/opencv/profile.xml"), profile_bytes)
            .expect("write profile fixture");

        let mut value = valid_v3_value();
        value["face_perception"]["frontal_face_cascade_asset"] = json!({
            "relative_path": "models/opencv/frontal.xml",
            "maximum_bytes": 64,
            "sha256_hex": format!("{:x}", Sha256::digest(frontal_bytes))
        });
        value["face_perception"]["profile_face_cascade_asset"] = json!({
            "relative_path": "models/opencv/profile.xml",
            "maximum_bytes": 64,
            "sha256_hex": format!("{:x}", Sha256::digest(profile_bytes))
        });
        let launch = parse_v3(&value).expect("parse bound V3 launch");
        let frontal = launch
            .face_perception()
            .frontal_face_cascade()
            .load_exact(&directory)
            .expect("load exact frontal cascade");
        let profile = launch
            .face_perception()
            .profile_face_cascade()
            .load_exact(&directory)
            .expect("load exact profile cascade");
        assert_eq!(frontal.bytes(), frontal_bytes);
        assert_eq!(profile.bytes(), profile_bytes);
        assert_eq!(
            frontal.relative_path().as_str(),
            "models/opencv/frontal.xml"
        );
        assert_eq!(
            profile.relative_path().as_str(),
            "models/opencv/profile.xml"
        );

        fs::write(
            directory.join("models/opencv/profile.xml"),
            b"different profile fixture",
        )
        .expect("replace profile fixture");
        assert!(matches!(
            launch
                .face_perception()
                .profile_face_cascade()
                .load_exact(&directory),
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
        let linked = load_nano_agent_launch_v2(
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
        let aliased = load_nano_agent_launch_v2(
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

    #[test]
    fn v3_launch_loader_rejects_face_asset_self_alias() {
        let requested_directory = unique_test_directory("launch-v3-loader");
        fs::create_dir_all(&requested_directory).expect("create test directory");
        let directory =
            fs::canonicalize(requested_directory).expect("canonicalize test directory root");
        let mut aliases = valid_v3_value();
        aliases["face_perception"]["profile_face_cascade_asset"]["relative_path"] =
            json!("launch-v3.json");
        fs::write(
            directory.join("launch-v3.json"),
            serde_json::to_vec(&aliases).expect("fixture serializes"),
        )
        .expect("write V3 launch");
        let aliased = load_nano_agent_launch_v3(
            &directory,
            ArtifactRelativePath::parse("launch-v3.json".to_owned()).expect("relative path"),
        );
        assert!(matches!(
            aliased,
            Err(NanoAgentLaunchLoadError::FaceInputAliasesLaunchDocument {
                role: NanoFaceCascadeAssetRole::ProfileFace,
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
