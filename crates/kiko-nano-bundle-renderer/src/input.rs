use std::collections::BTreeSet;
use std::fmt;
use std::path::{Component, Path, PathBuf};

use kiko_expression_core::{PositiveUnitAmount, UnitAmount};
use kiko_expression_runtime::{
    CameraToHeadGazeExtrinsics, CameraToHeadGazeExtrinsicsInput, MotionThresholds, SamplingGeometry,
};
use kiko_slam::navigation::{
    KIKO_REVIEWED_NATURAL_HEAD_MAXIMUM_TRAVEL_TICKS as REVIEWED_NATURAL_HEAD_MAXIMUM_TRAVEL_TICKS,
    KIKO_REVIEWED_NATURAL_HEAD_START_MAXIMUM_TICKS as REVIEWED_NATURAL_HEAD_START_MAXIMUM_TICKS,
    KIKO_REVIEWED_NATURAL_HEAD_START_MINIMUM_TICKS as REVIEWED_NATURAL_HEAD_START_MINIMUM_TICKS,
    KIKO_REVIEWED_NATURAL_HEAD_TARGET_TICKS as REVIEWED_NATURAL_HEAD_TARGET_TICKS,
    KIKO_REVIEWED_NATURAL_HEAD_TORQUE_LIMIT_PERMILLE as REVIEWED_NATURAL_HEAD_TORQUE_LIMIT_PERMILLE,
    MAX_INFERENCE_KEYPOINTS,
};
use robot_protocol::v2::ControllerCapabilities;
use serde::{Deserialize, Deserializer, Serialize};

pub const PRODUCTION_RENDER_INPUT_SCHEMA_VERSION: u32 = 2;
pub const WHEELS_OFF_QUALIFICATION_RENDER_INPUT_SCHEMA_VERSION: u32 = 4;
pub const PRODUCTION_PROFILE_SCHEMA_VERSION: u32 = 1;
pub const PRODUCTION_ADMISSION_SCOPE: &str =
    "production_motion_profile_after_physical_wheels_off_review_v1";
pub const CANDIDATE_FIRMWARE_ABI: u16 = 2;
pub const CANDIDATE_FIRMWARE_BUILD_ID: u32 = 135_169;
pub const CANDIDATE_FINGERPRINT_HEX: &str = "4b494b4f2d3450574d2d43414e443121";
pub const CANDIDATE_CAPABILITIES_BITS: u32 = 575;

const MAX_TEXT_BYTES: usize = 256;
const MAX_ABSOLUTE_PATH_BYTES: usize = 4_096;
const MAX_RELATIVE_PATH_BYTES: usize = 1_024;
const RENDERED_EYE_RESPONSE_TIMEOUT_MS: u64 = 20;
const RENDERED_EYE_WRITE_TIMEOUT_MS: u64 = 5;
const RENDERED_EYE_WRITE_ATTEMPTS: u64 = 2;
const MAX_RGB_FRAME_FRESHNESS_MS: u64 = 5_000;
const MAX_NANO_STATE_BYTES: u64 = 1_099_511_627_776;
const MAX_NAVIGATION_DATASET_FILES: u64 = 65_536;
const MAX_NAVIGATION_INGRESS_RECORDS: u64 = 1_048_576;
const NAVIGATION_DATASET_ADMISSION_FRAGMENT_BYTES: u64 = 4_096;
const MAX_NAVIGATION_DATASET_MANIFEST_BYTES: u64 = 64 * 1_024 * 1_024;
const MAX_WARM_START_SELECTION_BYTES: u64 = 4 * 1_024;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RenderInputDto {
    schema_version: u32,
    bundle: BundleSelectionDto,
    robot_id: String,
    discovery: DiscoveryDto,
    assets: AssetSetDto,
    native_libraries: Vec<NativeLibraryDto>,
    runtime: RuntimeEnvelopeDto,
    head_policy: HeadPolicyDto,
    rgb_expression_policy: RgbExpressionPolicyDto,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum BundleSelectionDto {
    WheelsOffQualification {
        qualification_executable_path: Option<PathBuf>,
    },
    Production {
        production_controller_profile_path: Option<PathBuf>,
    },
}

impl BundleSelectionDto {
    const fn kind_name(&self) -> &'static str {
        match self {
            Self::WheelsOffQualification { .. } => "wheels_off_qualification",
            Self::Production { .. } => "production",
        }
    }

    const fn render_input_schema_version(&self) -> u32 {
        match self {
            Self::WheelsOffQualification { .. } => {
                WHEELS_OFF_QUALIFICATION_RENDER_INPUT_SCHEMA_VERSION
            }
            Self::Production { .. } => PRODUCTION_RENDER_INPUT_SCHEMA_VERSION,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DiscoveryDto {
    oak: OakDiscoveryDto,
    stm32: Stm32DiscoveryDto,
    head: HeadDiscoveryDto,
    eye: EyeDiscoveryDto,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OakDiscoveryDto {
    mxid: String,
    compiled_depthai_header_sdk_version: String,
    compiled_depthai_header_sdk_commit: String,
    compiled_depthai_header_embedded_device_artifact_version: String,
    compiled_depthai_header_embedded_bootloader_artifact_version: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Stm32DiscoveryDto {
    serial_by_id_path: String,
    controller_uid_hex: String,
    firmware_abi: u16,
    firmware_build_id: u32,
    hardware_profile_fingerprint_hex: String,
    capabilities_bits: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HeadDiscoveryDto {
    adapter_serial_by_id_path: String,
    bow_servo_id: u8,
    curl_servo_id: u8,
    yaw_servo_id: u8,
    roll_servo_id: u8,
    baud_rate_bps: u32,
    dtr_asserted: bool,
    rts_asserted: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EyeDiscoveryDto {
    serial_by_id_path: String,
    kep_protocol_version: u8,
    device_uid_hex: String,
    firmware_build_id_hex: String,
    capabilities_bits: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AssetSetDto {
    calibration: ArtifactSourceDto,
    plant: ArtifactSourceDto,
    navigation_shadow_source_path: PathBuf,
    superpoint_model: FileSourceDto,
    lightglue_model: FileSourceDto,
    face_perception: Option<FacePerceptionAssetsDto>,
    #[serde(default)]
    head_gaze_policy_source_path: JsonFieldPresence<PathBuf>,
    #[serde(default)]
    head_gaze_review_evidence_source_path: JsonFieldPresence<PathBuf>,
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
struct ArtifactSourceDto {
    artifact_id: String,
    source_path: PathBuf,
    destination_relative_path: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileSourceDto {
    source_path: PathBuf,
    destination_relative_path: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FacePerceptionAssetsDto {
    frontal_face_cascade: FileSourceDto,
    profile_face_cascade: FileSourceDto,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum NativeLibraryRole {
    DepthaiCore,
    DynamicCalibration,
    Libusb1_0,
    Onnxruntime,
    OpencvCore,
    OpencvImgproc,
    OpencvObjdetect,
}

impl NativeLibraryRole {
    pub(crate) const ALL: [Self; 7] = [
        Self::DepthaiCore,
        Self::DynamicCalibration,
        Self::Libusb1_0,
        Self::Onnxruntime,
        Self::OpencvCore,
        Self::OpencvImgproc,
        Self::OpencvObjdetect,
    ];

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::DepthaiCore => "depthai_core",
            Self::DynamicCalibration => "dynamic_calibration",
            Self::Libusb1_0 => "libusb_1_0",
            Self::Onnxruntime => "onnxruntime",
            Self::OpencvCore => "opencv_core",
            Self::OpencvImgproc => "opencv_imgproc",
            Self::OpencvObjdetect => "opencv_objdetect",
        }
    }

    const fn exact_nano_soname(self) -> Option<&'static str> {
        match self {
            Self::OpencvCore => Some("libopencv_core.so.4.5d"),
            Self::OpencvImgproc => Some("libopencv_imgproc.so.4.5d"),
            Self::OpencvObjdetect => Some("libopencv_objdetect.so.4.5d"),
            Self::DepthaiCore | Self::DynamicCalibration | Self::Libusb1_0 | Self::Onnxruntime => {
                None
            }
        }
    }

    const fn qualification_exact_nano_soname(self) -> &'static str {
        match self {
            Self::DepthaiCore => "libdepthai-core.so",
            Self::DynamicCalibration => "libdynamic_calibration.so",
            Self::Libusb1_0 => "libusb-1.0.so",
            Self::Onnxruntime => "libonnxruntime.so.1",
            Self::OpencvCore => "libopencv_core.so.4.5d",
            Self::OpencvImgproc => "libopencv_imgproc.so.4.5d",
            Self::OpencvObjdetect => "libopencv_objdetect.so.4.5d",
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeLibraryDto {
    role: NativeLibraryRole,
    soname: String,
    source_path: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeEnvelopeDto {
    oak: OakGraphDto,
    occupancy: OccupancyDto,
    inference: InferenceDto,
    rerun: RerunDto,
    storage: StorageDto,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OakGraphDto {
    rgb_width_px: u32,
    rgb_height_px: u32,
    rgb_fps: u32,
    stereo_width_px: u32,
    stereo_height_px: u32,
    stereo_fps: u32,
    imu_rate_hz: u32,
    queue_size: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OccupancyDto {
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
struct InferenceDto {
    superpoint_backend: InferenceBackend,
    lightglue_backend: InferenceBackend,
    downscale_factor: u32,
    maximum_keypoints: u32,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceBackend {
    Auto,
    Cpu,
    Cuda,
    CudaCpuHybrid,
    #[serde(rename = "tensorrt")]
    TensorRt,
}

impl InferenceBackend {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::CudaCpuHybrid => "cuda_cpu_hybrid",
            Self::TensorRt => "tensorrt",
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RerunDto {
    decimation: u32,
    memory_limit_bytes: u64,
    flush_timeout_ms: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StorageDto {
    maximum_map_snapshot_bytes: u64,
    minimum_free_bytes_after_map_save: u64,
    maximum_navigation_dataset_bytes: u64,
    maximum_navigation_dataset_files: u64,
    maximum_navigation_ingress_records: u64,
    minimum_free_bytes_after_navigation_dataset_write: u64,
    navigation_dataset_terminal_reserve_bytes: u64,
    warm_start: WarmStartSelection,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum WarmStartSelection {
    None,
    DatasetReplay,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HeadPolicyDto {
    response_timeout_ms: u64,
    write_timeout_ms: u64,
    arming_freshness_ms: u64,
    write_attempts: u8,
    noise_budget_bytes: u16,
    redundant_read_tolerance_ticks: u16,
    readback_tolerance_ticks: u16,
    final_target_tolerance_ticks: u16,
    path_corridor_tolerance_ticks: u16,
    direction_regression_tolerance_ticks: u16,
    goal_speed_ticks_per_second: u16,
    torque_limit_permille: [u16; 4],
    minimum_start_ticks: [u16; 4],
    maximum_start_ticks: [u16; 4],
    reviewed_natural_target_ticks: [u16; 4],
    maximum_travel_ticks: [u16; 4],
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RgbExpressionPolicyDto {
    sampling_columns: u16,
    sampling_rows: u16,
    minimum_residual_luma: u16,
    minimum_active_fraction_basis_points: u16,
    frame_freshness_ms: u64,
    brightness_basis_points: u16,
    color_rgb: [u8; 3],
    blink: bool,
    head_origin_in_camera_m: [f64; 3],
    neutral_head_from_camera_quaternion_xyzw: [f64; 4],
}

#[derive(Debug)]
pub(crate) struct RenderInput {
    pub(crate) bundle: BundleSelection,
    pub(crate) robot_id: String,
    pub(crate) discovery: Discovery,
    pub(crate) assets: AssetSet,
    pub(crate) native_libraries: Vec<NativeLibrary>,
    pub(crate) runtime: RuntimeEnvelope,
    pub(crate) head_policy: HeadPolicy,
    pub(crate) rgb_expression_policy: RgbExpressionPolicy,
}

#[derive(Debug)]
pub(crate) enum BundleSelection {
    WheelsOffQualification {
        qualification_executable_path: AbsoluteSourcePath,
        face_perception: FacePerceptionAssets,
        head_gaze_policy_source_path: Option<AbsoluteSourcePath>,
    },
    Production {
        production_controller_profile_path: Option<AbsoluteSourcePath>,
        face_perception: FacePerceptionAssets,
        head_gaze_policy_source_path: AbsoluteSourcePath,
        head_gaze_review_evidence_source_path: AbsoluteSourcePath,
    },
}

impl BundleSelection {
    pub(crate) const fn kind_name(&self) -> &'static str {
        match self {
            Self::WheelsOffQualification { .. } => "wheels_off_qualification",
            Self::Production { .. } => "production",
        }
    }

    pub(crate) const fn launch_relative_path(&self) -> &'static str {
        match self {
            Self::WheelsOffQualification { .. } => "nano-wheels-off-qualification-launch-v4.json",
            Self::Production { .. } => "nano-agent-launch-v4.json",
        }
    }

    pub(crate) const fn render_input_evidence_path(&self) -> &'static str {
        match self {
            Self::WheelsOffQualification { .. } => "evidence/render-input-v4.json",
            Self::Production { .. } => "evidence/render-input-v2.json",
        }
    }

    pub(crate) const fn face_perception(&self) -> Option<&FacePerceptionAssets> {
        match self {
            Self::WheelsOffQualification {
                face_perception, ..
            }
            | Self::Production {
                face_perception, ..
            } => Some(face_perception),
        }
    }

    pub(crate) const fn head_gaze_policy_source_path(&self) -> Option<&AbsoluteSourcePath> {
        match self {
            Self::WheelsOffQualification {
                head_gaze_policy_source_path,
                ..
            } => head_gaze_policy_source_path.as_ref(),
            Self::Production {
                head_gaze_policy_source_path,
                ..
            } => Some(head_gaze_policy_source_path),
        }
    }

    pub(crate) const fn head_gaze_review_evidence_source_path(
        &self,
    ) -> Option<&AbsoluteSourcePath> {
        match self {
            Self::WheelsOffQualification { .. } => None,
            Self::Production {
                head_gaze_review_evidence_source_path,
                ..
            } => Some(head_gaze_review_evidence_source_path),
        }
    }
}

#[derive(Debug)]
pub(crate) struct Discovery {
    pub(crate) oak: OakDiscovery,
    pub(crate) stm32: Stm32Discovery,
    pub(crate) head: HeadDiscovery,
    pub(crate) eye: EyeDiscovery,
}

#[derive(Debug)]
pub(crate) struct OakDiscovery {
    pub(crate) mxid: String,
    pub(crate) sdk_version: String,
    pub(crate) sdk_commit: String,
    pub(crate) device_artifact_version: String,
    pub(crate) bootloader_artifact_version: String,
}

#[derive(Debug)]
pub(crate) struct Stm32Discovery {
    pub(crate) serial_by_id_path: String,
    pub(crate) controller_uid: [u8; 12],
    pub(crate) firmware_abi: u16,
    pub(crate) firmware_build_id: u32,
    pub(crate) hardware_profile_fingerprint: [u8; 16],
    pub(crate) capabilities_bits: u32,
}

#[derive(Debug)]
pub(crate) struct HeadDiscovery {
    pub(crate) serial_by_id_path: String,
    pub(crate) servo_ids: [u8; 4],
    pub(crate) baud_rate_bps: u32,
    pub(crate) dtr_asserted: bool,
    pub(crate) rts_asserted: bool,
}

#[derive(Debug)]
pub(crate) struct EyeDiscovery {
    pub(crate) serial_by_id_path: String,
    pub(crate) kep_protocol_version: u8,
    pub(crate) device_uid: [u8; 16],
    pub(crate) firmware_build_id: [u8; 32],
    pub(crate) capabilities_bits: u32,
}

#[derive(Debug)]
pub(crate) struct AssetSet {
    pub(crate) calibration: ArtifactSource,
    pub(crate) plant: ArtifactSource,
    pub(crate) navigation_shadow_source_path: AbsoluteSourcePath,
    pub(crate) superpoint_model: FileSource,
    pub(crate) lightglue_model: FileSource,
}

#[derive(Debug)]
pub(crate) struct ArtifactSource {
    pub(crate) artifact_id: String,
    pub(crate) source_path: AbsoluteSourcePath,
    pub(crate) destination_relative_path: RelativeBundlePath,
    pub(crate) artifact_relative_path: String,
}

#[derive(Debug)]
pub(crate) struct FileSource {
    pub(crate) source_path: AbsoluteSourcePath,
    pub(crate) destination_relative_path: RelativeBundlePath,
}

#[derive(Debug)]
pub(crate) struct FacePerceptionAssets {
    pub(crate) frontal_face_cascade: FileSource,
    pub(crate) profile_face_cascade: FileSource,
}

#[derive(Debug)]
pub(crate) struct NativeLibrary {
    pub(crate) role: NativeLibraryRole,
    pub(crate) soname: String,
    pub(crate) source_path: AbsoluteSourcePath,
    pub(crate) destination_relative_path: RelativeBundlePath,
}

#[derive(Debug)]
pub(crate) struct RuntimeEnvelope {
    pub(crate) oak: OakGraph,
    pub(crate) occupancy: Occupancy,
    pub(crate) inference: Inference,
    pub(crate) rerun: Rerun,
    pub(crate) storage: Storage,
}

#[derive(Debug)]
pub(crate) struct OakGraph {
    pub(crate) rgb_width_px: u32,
    pub(crate) rgb_height_px: u32,
    pub(crate) rgb_fps: u32,
    pub(crate) stereo_width_px: u32,
    pub(crate) stereo_height_px: u32,
    pub(crate) stereo_fps: u32,
    pub(crate) imu_rate_hz: u32,
    pub(crate) queue_size: u32,
}

#[derive(Debug)]
pub(crate) struct Occupancy {
    pub(crate) resolution_m: f64,
    pub(crate) lower_x_m: f64,
    pub(crate) lower_y_m: f64,
    pub(crate) width_cells: u32,
    pub(crate) height_cells: u32,
    pub(crate) maximum_cells: u64,
    pub(crate) maximum_keyframes: u64,
    pub(crate) snapshot_every_keyframes: u64,
}

#[derive(Debug)]
pub(crate) struct Inference {
    pub(crate) superpoint_backend: InferenceBackend,
    pub(crate) lightglue_backend: InferenceBackend,
    pub(crate) downscale_factor: u32,
    pub(crate) maximum_keypoints: u32,
}

#[derive(Debug)]
pub(crate) struct Rerun {
    pub(crate) decimation: u32,
    pub(crate) memory_limit_bytes: u64,
    pub(crate) flush_timeout_ms: u64,
}

#[derive(Debug)]
pub(crate) struct Storage {
    pub(crate) maximum_map_snapshot_bytes: u64,
    pub(crate) minimum_free_bytes_after_map_save: u64,
    pub(crate) navigation_dataset: NavigationDatasetStorageLimits,
    pub(crate) warm_start: WarmStartSelection,
}

#[derive(Debug)]
pub(crate) struct NavigationDatasetStorageLimits {
    pub(crate) maximum_bytes: u64,
    pub(crate) maximum_files: u64,
    pub(crate) maximum_ingress_records: u64,
    pub(crate) minimum_free_bytes_after_write: u64,
    pub(crate) terminal_reserve_bytes: u64,
}

#[derive(Debug)]
pub(crate) struct HeadPolicy {
    pub(crate) response_timeout_ms: u64,
    pub(crate) write_timeout_ms: u64,
    pub(crate) arming_freshness_ms: u64,
    pub(crate) write_attempts: u8,
    pub(crate) noise_budget_bytes: u16,
    pub(crate) redundant_read_tolerance_ticks: u16,
    pub(crate) readback_tolerance_ticks: u16,
    pub(crate) final_target_tolerance_ticks: u16,
    pub(crate) path_corridor_tolerance_ticks: u16,
    pub(crate) direction_regression_tolerance_ticks: u16,
    pub(crate) goal_speed_ticks_per_second: u16,
    pub(crate) torque_limit_permille: [u16; 4],
    pub(crate) minimum_start_ticks: [u16; 4],
    pub(crate) maximum_start_ticks: [u16; 4],
    pub(crate) reviewed_natural_target_ticks: [u16; 4],
    pub(crate) maximum_travel_ticks: [u16; 4],
}

#[derive(Debug)]
pub(crate) struct RgbExpressionPolicy {
    pub(crate) sampling_columns: u16,
    pub(crate) sampling_rows: u16,
    pub(crate) minimum_residual_luma: u16,
    pub(crate) minimum_active_fraction_basis_points: u16,
    pub(crate) frame_freshness_ms: u64,
    pub(crate) brightness_basis_points: u16,
    pub(crate) color_rgb: [u8; 3],
    pub(crate) blink: bool,
    pub(crate) head_origin_in_camera_m: [f64; 3],
    pub(crate) neutral_head_from_camera_quaternion_xyzw: [f64; 4],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AbsoluteSourcePath(PathBuf);

impl AbsoluteSourcePath {
    pub(crate) fn as_path(&self) -> &Path {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct RelativeBundlePath(String);

impl RelativeBundlePath {
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

impl RenderInput {
    pub(crate) fn parse(dto: RenderInputDto) -> Result<Self, InputError> {
        let expected_schema = dto.bundle.render_input_schema_version();
        if dto.schema_version != expected_schema {
            return Err(InputError::UnsupportedRenderInputSchema {
                actual: dto.schema_version,
                expected: expected_schema,
                bundle_kind: dto.bundle.kind_name(),
            });
        }

        let robot_id = parse_text("robot_id", dto.robot_id)?;
        let ParsedAssetSet {
            assets,
            face_perception,
            head_gaze_policy_source_path,
            head_gaze_review_evidence_source_path,
        } = AssetSet::parse(dto.assets)?;
        let bundle = match dto.bundle {
            BundleSelectionDto::WheelsOffQualification {
                qualification_executable_path,
            } => {
                if !matches!(
                    head_gaze_review_evidence_source_path,
                    JsonFieldPresence::Absent
                ) {
                    return Err(InputError::QualificationHeadGazeReviewEvidenceForbidden);
                }
                let head_gaze_policy_source_path = match head_gaze_policy_source_path {
                    JsonFieldPresence::Absent => None,
                    JsonFieldPresence::Null => {
                        return Err(InputError::QualificationHeadGazePolicyNullForbidden);
                    }
                    JsonFieldPresence::Value(path) => Some(parse_absolute_source_path(
                        "assets.head_gaze_policy_source_path",
                        path,
                    )?),
                };
                BundleSelection::WheelsOffQualification {
                    qualification_executable_path: parse_absolute_source_path(
                        "bundle.qualification_executable_path",
                        qualification_executable_path
                            .ok_or(InputError::QualificationExecutablePathRequired)?,
                    )?,
                    face_perception: face_perception
                        .ok_or(InputError::QualificationFacePerceptionAssetsRequired)?,
                    head_gaze_policy_source_path,
                }
            }
            BundleSelectionDto::Production {
                production_controller_profile_path,
            } => {
                let head_gaze_policy_source_path = parse_required_production_asset_path(
                    "assets.head_gaze_policy_source_path",
                    head_gaze_policy_source_path,
                    InputError::ProductionHeadGazePolicyRequired,
                )?;
                let head_gaze_review_evidence_source_path = parse_required_production_asset_path(
                    "assets.head_gaze_review_evidence_source_path",
                    head_gaze_review_evidence_source_path,
                    InputError::ProductionHeadGazeReviewEvidenceRequired,
                )?;
                BundleSelection::Production {
                    production_controller_profile_path: production_controller_profile_path
                        .map(|path| {
                            parse_absolute_source_path("production_controller_profile_path", path)
                        })
                        .transpose()?,
                    face_perception: face_perception
                        .ok_or(InputError::ProductionFacePerceptionAssetsRequired)?,
                    head_gaze_policy_source_path,
                    head_gaze_review_evidence_source_path,
                }
            }
        };
        let discovery = Discovery::parse(dto.discovery)?;
        if matches!(bundle, BundleSelection::WheelsOffQualification { .. }) {
            discovery.stm32.require_candidate_identity()?;
        }
        let qualification_bundle = matches!(bundle, BundleSelection::WheelsOffQualification { .. });
        let native_libraries = parse_native_libraries(dto.native_libraries, qualification_bundle)?;
        let runtime = RuntimeEnvelope::parse(dto.runtime)?;
        if matches!(bundle, BundleSelection::WheelsOffQualification { .. })
            && runtime.storage.warm_start == WarmStartSelection::DatasetReplay
        {
            return Err(InputError::WarmStartForbiddenInQualification);
        }
        let head_policy = HeadPolicy::parse(dto.head_policy)?;
        let rgb_expression_policy = RgbExpressionPolicy::parse(dto.rgb_expression_policy)?;

        Ok(Self {
            bundle,
            robot_id,
            discovery,
            assets,
            native_libraries,
            runtime,
            head_policy,
            rgb_expression_policy,
        })
    }
}

impl Discovery {
    fn parse(dto: DiscoveryDto) -> Result<Self, InputError> {
        let oak = OakDiscovery {
            mxid: parse_hex_text("discovery.oak.mxid", dto.oak.mxid, 1, 64)?,
            sdk_version: parse_text(
                "discovery.oak.compiled_depthai_header_sdk_version",
                dto.oak.compiled_depthai_header_sdk_version,
            )?,
            sdk_commit: parse_text(
                "discovery.oak.compiled_depthai_header_sdk_commit",
                dto.oak.compiled_depthai_header_sdk_commit,
            )?,
            device_artifact_version: parse_text(
                "discovery.oak.compiled_depthai_header_embedded_device_artifact_version",
                dto.oak
                    .compiled_depthai_header_embedded_device_artifact_version,
            )?,
            bootloader_artifact_version: parse_text(
                "discovery.oak.compiled_depthai_header_embedded_bootloader_artifact_version",
                dto.oak
                    .compiled_depthai_header_embedded_bootloader_artifact_version,
            )?,
        };
        let stm32 = Stm32Discovery {
            serial_by_id_path: parse_serial_by_id_path(
                "discovery.stm32.serial_by_id_path",
                dto.stm32.serial_by_id_path,
            )?,
            controller_uid: parse_hex_array(
                "discovery.stm32.controller_uid_hex",
                &dto.stm32.controller_uid_hex,
            )?,
            firmware_abi: require_nonzero_u16(
                "discovery.stm32.firmware_abi",
                dto.stm32.firmware_abi,
            )?,
            firmware_build_id: require_nonzero_u32(
                "discovery.stm32.firmware_build_id",
                dto.stm32.firmware_build_id,
            )?,
            hardware_profile_fingerprint: parse_hex_array(
                "discovery.stm32.hardware_profile_fingerprint_hex",
                &dto.stm32.hardware_profile_fingerprint_hex,
            )?,
            capabilities_bits: require_nonzero_u32(
                "discovery.stm32.capabilities_bits",
                dto.stm32.capabilities_bits,
            )?,
        };
        let servo_ids = [
            dto.head.bow_servo_id,
            dto.head.curl_servo_id,
            dto.head.yaw_servo_id,
            dto.head.roll_servo_id,
        ];
        if servo_ids.contains(&0) || servo_ids.iter().collect::<BTreeSet<_>>().len() != 4 {
            return Err(InputError::InvalidServoIds);
        }
        let head = HeadDiscovery {
            serial_by_id_path: parse_serial_by_id_path(
                "discovery.head.adapter_serial_by_id_path",
                dto.head.adapter_serial_by_id_path,
            )?,
            servo_ids,
            baud_rate_bps: require_nonzero_u32(
                "discovery.head.baud_rate_bps",
                dto.head.baud_rate_bps,
            )?,
            dtr_asserted: dto.head.dtr_asserted,
            rts_asserted: dto.head.rts_asserted,
        };
        let eye = EyeDiscovery {
            serial_by_id_path: parse_serial_by_id_path(
                "discovery.eye.serial_by_id_path",
                dto.eye.serial_by_id_path,
            )?,
            kep_protocol_version: require_nonzero_u8(
                "discovery.eye.kep_protocol_version",
                dto.eye.kep_protocol_version,
            )?,
            device_uid: parse_hex_array("discovery.eye.device_uid_hex", &dto.eye.device_uid_hex)?,
            firmware_build_id: parse_hex_array(
                "discovery.eye.firmware_build_id_hex",
                &dto.eye.firmware_build_id_hex,
            )?,
            capabilities_bits: require_nonzero_u32(
                "discovery.eye.capabilities_bits",
                dto.eye.capabilities_bits,
            )?,
        };
        let serial_paths = [
            stm32.serial_by_id_path.as_str(),
            head.serial_by_id_path.as_str(),
            eye.serial_by_id_path.as_str(),
        ];
        if serial_paths.into_iter().collect::<BTreeSet<_>>().len() != serial_paths.len() {
            return Err(InputError::DuplicateSerialByIdPath);
        }
        Ok(Self {
            oak,
            stm32,
            head,
            eye,
        })
    }
}

impl Stm32Discovery {
    fn require_candidate_identity(&self) -> Result<(), InputError> {
        if self.firmware_abi != CANDIDATE_FIRMWARE_ABI
            || self.firmware_build_id != CANDIDATE_FIRMWARE_BUILD_ID
            || encode_hex(&self.hardware_profile_fingerprint) != CANDIDATE_FINGERPRINT_HEX
            || self.capabilities_bits != CANDIDATE_CAPABILITIES_BITS
        {
            return Err(InputError::CandidateControllerIdentityMismatch);
        }
        Ok(())
    }
}

struct ParsedAssetSet {
    assets: AssetSet,
    face_perception: Option<FacePerceptionAssets>,
    head_gaze_policy_source_path: JsonFieldPresence<PathBuf>,
    head_gaze_review_evidence_source_path: JsonFieldPresence<PathBuf>,
}

impl AssetSet {
    fn parse(dto: AssetSetDto) -> Result<ParsedAssetSet, InputError> {
        let face_perception = dto
            .face_perception
            .map(|face| {
                Ok(FacePerceptionAssets {
                    frontal_face_cascade: parse_file_source(
                        "assets.face_perception.frontal_face_cascade",
                        face.frontal_face_cascade,
                    )?,
                    profile_face_cascade: parse_file_source(
                        "assets.face_perception.profile_face_cascade",
                        face.profile_face_cascade,
                    )?,
                })
            })
            .transpose()?;
        Ok(ParsedAssetSet {
            assets: Self {
                calibration: parse_artifact_source("assets.calibration", dto.calibration)?,
                plant: parse_artifact_source("assets.plant", dto.plant)?,
                navigation_shadow_source_path: parse_absolute_source_path(
                    "assets.navigation_shadow_source_path",
                    dto.navigation_shadow_source_path,
                )?,
                superpoint_model: parse_file_source(
                    "assets.superpoint_model",
                    dto.superpoint_model,
                )?,
                lightglue_model: parse_file_source("assets.lightglue_model", dto.lightglue_model)?,
            },
            face_perception,
            head_gaze_policy_source_path: dto.head_gaze_policy_source_path,
            head_gaze_review_evidence_source_path: dto.head_gaze_review_evidence_source_path,
        })
    }
}

fn parse_required_production_asset_path(
    field: &'static str,
    presence: JsonFieldPresence<PathBuf>,
    missing: InputError,
) -> Result<AbsoluteSourcePath, InputError> {
    match presence {
        JsonFieldPresence::Value(path) => parse_absolute_source_path(field, path),
        JsonFieldPresence::Absent | JsonFieldPresence::Null => Err(missing),
    }
}

fn parse_artifact_source(
    field: &'static str,
    dto: ArtifactSourceDto,
) -> Result<ArtifactSource, InputError> {
    let destination_relative_path =
        parse_relative_bundle_path(field, dto.destination_relative_path)?;
    let artifact_relative_path = destination_relative_path
        .as_str()
        .strip_prefix("artifacts/")
        .filter(|suffix| !suffix.is_empty())
        .ok_or(InputError::ArtifactOutsideArtifactRoot { field })?
        .to_owned();
    Ok(ArtifactSource {
        artifact_id: parse_text(field, dto.artifact_id)?,
        source_path: parse_absolute_source_path(field, dto.source_path)?,
        destination_relative_path,
        artifact_relative_path,
    })
}

fn parse_file_source(field: &'static str, dto: FileSourceDto) -> Result<FileSource, InputError> {
    Ok(FileSource {
        source_path: parse_absolute_source_path(field, dto.source_path)?,
        destination_relative_path: parse_relative_bundle_path(
            field,
            dto.destination_relative_path,
        )?,
    })
}

fn parse_native_libraries(
    dtos: Vec<NativeLibraryDto>,
    qualification_bundle: bool,
) -> Result<Vec<NativeLibrary>, InputError> {
    if dtos.len() != NativeLibraryRole::ALL.len() {
        return Err(InputError::IncompleteNativeLibrarySet);
    }
    let mut seen = BTreeSet::new();
    let mut libraries = Vec::with_capacity(dtos.len());
    for dto in dtos {
        if !seen.insert(dto.role) {
            return Err(InputError::DuplicateNativeLibraryRole { role: dto.role });
        }
        let soname = parse_soname(dto.soname)?;
        let exact_soname = if qualification_bundle {
            Some(dto.role.qualification_exact_nano_soname())
        } else {
            dto.role.exact_nano_soname()
        };
        if let Some(expected) = exact_soname
            && soname != expected
        {
            return Err(InputError::WrongNativeLibrarySoname {
                role: dto.role,
                expected,
                actual: soname,
            });
        }
        let expected_relative = format!("lib/{soname}");
        libraries.push(NativeLibrary {
            role: dto.role,
            soname,
            source_path: parse_absolute_source_path(
                "native_libraries.source_path",
                dto.source_path,
            )?,
            destination_relative_path: RelativeBundlePath(expected_relative),
        });
    }
    if NativeLibraryRole::ALL
        .iter()
        .any(|role| !seen.contains(role))
    {
        return Err(InputError::IncompleteNativeLibrarySet);
    }
    libraries.sort_by_key(|library| library.role);
    Ok(libraries)
}

impl RuntimeEnvelope {
    fn parse(dto: RuntimeEnvelopeDto) -> Result<Self, InputError> {
        let oak = OakGraph {
            rgb_width_px: bounded_nonzero_u32(
                "runtime.oak.rgb_width_px",
                dto.oak.rgb_width_px,
                4_096,
            )?,
            rgb_height_px: bounded_nonzero_u32(
                "runtime.oak.rgb_height_px",
                dto.oak.rgb_height_px,
                3_072,
            )?,
            rgb_fps: bounded_nonzero_u32("runtime.oak.rgb_fps", dto.oak.rgb_fps, 240)?,
            stereo_width_px: bounded_nonzero_u32(
                "runtime.oak.stereo_width_px",
                dto.oak.stereo_width_px,
                4_096,
            )?,
            stereo_height_px: bounded_nonzero_u32(
                "runtime.oak.stereo_height_px",
                dto.oak.stereo_height_px,
                3_072,
            )?,
            stereo_fps: bounded_nonzero_u32("runtime.oak.stereo_fps", dto.oak.stereo_fps, 240)?,
            imu_rate_hz: bounded_nonzero_u32(
                "runtime.oak.imu_rate_hz",
                dto.oak.imu_rate_hz,
                2_000,
            )?,
            queue_size: bounded_nonzero_u32("runtime.oak.queue_size", dto.oak.queue_size, 64)?,
        };
        let occupancy_cells = u64::from(dto.occupancy.width_cells)
            .checked_mul(u64::from(dto.occupancy.height_cells))
            .ok_or(InputError::NumericOutOfRange {
                field: "runtime.occupancy.width_cells * height_cells",
            })?;
        if !dto.occupancy.resolution_m.is_finite()
            || dto.occupancy.resolution_m < 0.001
            || dto.occupancy.resolution_m > 10.0
            || !dto.occupancy.lower_x_m.is_finite()
            || !dto.occupancy.lower_y_m.is_finite()
            || dto.occupancy.lower_x_m.abs() > 100_000.0
            || dto.occupancy.lower_y_m.abs() > 100_000.0
            || dto.occupancy.width_cells == 0
            || dto.occupancy.height_cells == 0
            || dto.occupancy.width_cells > 100_000
            || dto.occupancy.height_cells > 100_000
            || occupancy_cells > dto.occupancy.maximum_cells
            || dto.occupancy.maximum_cells == 0
            || dto.occupancy.maximum_cells > 16_000_000
            || dto.occupancy.maximum_keyframes == 0
            || dto.occupancy.maximum_keyframes > 1_000_000
            || dto.occupancy.snapshot_every_keyframes == 0
            || dto.occupancy.snapshot_every_keyframes > 1_000_000
        {
            return Err(InputError::NumericOutOfRange {
                field: "runtime.occupancy",
            });
        }
        let occupancy = Occupancy {
            resolution_m: dto.occupancy.resolution_m,
            lower_x_m: dto.occupancy.lower_x_m,
            lower_y_m: dto.occupancy.lower_y_m,
            width_cells: dto.occupancy.width_cells,
            height_cells: dto.occupancy.height_cells,
            maximum_cells: dto.occupancy.maximum_cells,
            maximum_keyframes: dto.occupancy.maximum_keyframes,
            snapshot_every_keyframes: dto.occupancy.snapshot_every_keyframes,
        };
        let inference = Inference {
            superpoint_backend: dto.inference.superpoint_backend,
            lightglue_backend: dto.inference.lightglue_backend,
            downscale_factor: bounded_nonzero_u32(
                "runtime.inference.downscale_factor",
                dto.inference.downscale_factor,
                16,
            )?,
            maximum_keypoints: bounded_nonzero_u32(
                "runtime.inference.maximum_keypoints",
                dto.inference.maximum_keypoints,
                MAX_INFERENCE_KEYPOINTS,
            )?,
        };
        let rerun = Rerun {
            decimation: bounded_nonzero_u32(
                "runtime.rerun.decimation",
                dto.rerun.decimation,
                10_000,
            )?,
            memory_limit_bytes: bounded_nonzero_u64(
                "runtime.rerun.memory_limit_bytes",
                dto.rerun.memory_limit_bytes,
                4 * 1_024 * 1_024 * 1_024,
            )?,
            flush_timeout_ms: bounded_nonzero_u64(
                "runtime.rerun.flush_timeout_ms",
                dto.rerun.flush_timeout_ms,
                120_000,
            )?,
        };
        if rerun.memory_limit_bytes < 1_048_576 {
            return Err(InputError::NumericOutOfRange {
                field: "runtime.rerun.memory_limit_bytes",
            });
        }
        let maximum_map_snapshot_bytes = bounded_nonzero_u64(
            "runtime.storage.maximum_map_snapshot_bytes",
            dto.storage.maximum_map_snapshot_bytes,
            MAX_NANO_STATE_BYTES,
        )?;
        let maximum_navigation_dataset_bytes = bounded_nonzero_u64(
            "runtime.storage.maximum_navigation_dataset_bytes",
            dto.storage.maximum_navigation_dataset_bytes,
            MAX_NANO_STATE_BYTES,
        )?;
        let navigation_dataset_terminal_reserve_bytes = bounded_nonzero_u64(
            "runtime.storage.navigation_dataset_terminal_reserve_bytes",
            dto.storage.navigation_dataset_terminal_reserve_bytes,
            MAX_NANO_STATE_BYTES,
        )?;
        if navigation_dataset_terminal_reserve_bytes >= maximum_navigation_dataset_bytes {
            return Err(
                InputError::NavigationDatasetTerminalReserveNotBelowMaximum {
                    reserve_bytes: navigation_dataset_terminal_reserve_bytes,
                    maximum_dataset_bytes: maximum_navigation_dataset_bytes,
                },
            );
        }
        let minimum_terminal_reserve =
            minimum_navigation_dataset_terminal_reserve(maximum_map_snapshot_bytes)?;
        if navigation_dataset_terminal_reserve_bytes < minimum_terminal_reserve {
            return Err(InputError::NavigationDatasetTerminalReserveTooSmall {
                reserve_bytes: navigation_dataset_terminal_reserve_bytes,
                minimum_bytes: minimum_terminal_reserve,
            });
        }
        let storage = Storage {
            maximum_map_snapshot_bytes,
            minimum_free_bytes_after_map_save: bounded_nonzero_u64(
                "runtime.storage.minimum_free_bytes_after_map_save",
                dto.storage.minimum_free_bytes_after_map_save,
                MAX_NANO_STATE_BYTES,
            )?,
            navigation_dataset: NavigationDatasetStorageLimits {
                maximum_bytes: maximum_navigation_dataset_bytes,
                maximum_files: bounded_nonzero_u64(
                    "runtime.storage.maximum_navigation_dataset_files",
                    dto.storage.maximum_navigation_dataset_files,
                    MAX_NAVIGATION_DATASET_FILES,
                )?,
                maximum_ingress_records: bounded_nonzero_u64(
                    "runtime.storage.maximum_navigation_ingress_records",
                    dto.storage.maximum_navigation_ingress_records,
                    MAX_NAVIGATION_INGRESS_RECORDS,
                )?,
                minimum_free_bytes_after_write: bounded_nonzero_u64(
                    "runtime.storage.minimum_free_bytes_after_navigation_dataset_write",
                    dto.storage
                        .minimum_free_bytes_after_navigation_dataset_write,
                    MAX_NANO_STATE_BYTES,
                )?,
                terminal_reserve_bytes: navigation_dataset_terminal_reserve_bytes,
            },
            warm_start: dto.storage.warm_start,
        };
        Ok(Self {
            oak,
            occupancy,
            inference,
            rerun,
            storage,
        })
    }
}

impl HeadPolicy {
    fn parse(dto: HeadPolicyDto) -> Result<Self, InputError> {
        if dto.response_timeout_ms == 0
            || dto.write_timeout_ms == 0
            || dto.arming_freshness_ms == 0
            || dto.write_attempts == 0
            || dto.goal_speed_ticks_per_second == 0
            || dto.torque_limit_permille.iter().any(|value| *value > 1_000)
        {
            return Err(InputError::NumericOutOfRange {
                field: "head_policy",
            });
        }
        for index in 0..4 {
            let minimum = dto.minimum_start_ticks[index];
            let maximum = dto.maximum_start_ticks[index];
            let target = dto.reviewed_natural_target_ticks[index];
            if minimum > maximum
                || target < minimum
                || target > maximum
                || dto.maximum_travel_ticks[index] == 0
            {
                return Err(InputError::InvalidHeadEnvelope { index });
            }
        }
        if dto.reviewed_natural_target_ticks != REVIEWED_NATURAL_HEAD_TARGET_TICKS
            || dto.minimum_start_ticks != REVIEWED_NATURAL_HEAD_START_MINIMUM_TICKS
            || dto.maximum_start_ticks != REVIEWED_NATURAL_HEAD_START_MAXIMUM_TICKS
            || dto.maximum_travel_ticks != REVIEWED_NATURAL_HEAD_MAXIMUM_TRAVEL_TICKS
            || dto.torque_limit_permille != REVIEWED_NATURAL_HEAD_TORQUE_LIMIT_PERMILLE
        {
            return Err(InputError::UnreviewedNaturalHeadPolicy);
        }
        Ok(Self {
            response_timeout_ms: dto.response_timeout_ms,
            write_timeout_ms: dto.write_timeout_ms,
            arming_freshness_ms: dto.arming_freshness_ms,
            write_attempts: dto.write_attempts,
            noise_budget_bytes: dto.noise_budget_bytes,
            redundant_read_tolerance_ticks: dto.redundant_read_tolerance_ticks,
            readback_tolerance_ticks: dto.readback_tolerance_ticks,
            final_target_tolerance_ticks: dto.final_target_tolerance_ticks,
            path_corridor_tolerance_ticks: dto.path_corridor_tolerance_ticks,
            direction_regression_tolerance_ticks: dto.direction_regression_tolerance_ticks,
            goal_speed_ticks_per_second: dto.goal_speed_ticks_per_second,
            torque_limit_permille: dto.torque_limit_permille,
            minimum_start_ticks: dto.minimum_start_ticks,
            maximum_start_ticks: dto.maximum_start_ticks,
            reviewed_natural_target_ticks: dto.reviewed_natural_target_ticks,
            maximum_travel_ticks: dto.maximum_travel_ticks,
        })
    }
}

impl RgbExpressionPolicy {
    fn parse(dto: RgbExpressionPolicyDto) -> Result<Self, InputError> {
        SamplingGeometry::try_new(dto.sampling_columns, dto.sampling_rows)
            .map_err(|_| InputError::InvalidRgbExpressionPolicy)?;
        let active_fraction =
            PositiveUnitAmount::try_from_basis_points(dto.minimum_active_fraction_basis_points)
                .map_err(|_| InputError::InvalidRgbExpressionPolicy)?;
        MotionThresholds::try_new(dto.minimum_residual_luma, active_fraction)
            .map_err(|_| InputError::InvalidRgbExpressionPolicy)?;
        UnitAmount::try_from_basis_points(dto.brightness_basis_points)
            .map_err(|_| InputError::InvalidRgbExpressionPolicy)?;
        CameraToHeadGazeExtrinsics::parse(CameraToHeadGazeExtrinsicsInput {
            head_origin_in_camera_m: dto.head_origin_in_camera_m,
            neutral_head_from_camera_quaternion_xyzw: dto.neutral_head_from_camera_quaternion_xyzw,
        })
        .map_err(|_| InputError::InvalidRgbExpressionPolicy)?;
        let eye_round_trip_ms = RENDERED_EYE_WRITE_TIMEOUT_MS
            .checked_mul(RENDERED_EYE_WRITE_ATTEMPTS)
            .and_then(|writes| writes.checked_add(RENDERED_EYE_RESPONSE_TIMEOUT_MS))
            .expect("rendered eye timing constants fit u64");
        if dto.frame_freshness_ms <= eye_round_trip_ms
            || dto.frame_freshness_ms > MAX_RGB_FRAME_FRESHNESS_MS
        {
            return Err(InputError::InvalidRgbExpressionPolicy);
        }
        Ok(Self {
            sampling_columns: dto.sampling_columns,
            sampling_rows: dto.sampling_rows,
            minimum_residual_luma: dto.minimum_residual_luma,
            minimum_active_fraction_basis_points: dto.minimum_active_fraction_basis_points,
            frame_freshness_ms: dto.frame_freshness_ms,
            brightness_basis_points: dto.brightness_basis_points,
            color_rgb: dto.color_rgb,
            blink: dto.blink,
            head_origin_in_camera_m: dto.head_origin_in_camera_m,
            neutral_head_from_camera_quaternion_xyzw: dto.neutral_head_from_camera_quaternion_xyzw,
        })
    }
}

fn parse_text(field: &'static str, value: String) -> Result<String, InputError> {
    if value.is_empty()
        || value.len() > MAX_TEXT_BYTES
        || value.contains("${")
        || value.chars().any(char::is_control)
    {
        return Err(InputError::InvalidText { field });
    }
    Ok(value)
}

fn parse_hex_text(
    field: &'static str,
    value: String,
    minimum_digits: usize,
    maximum_digits: usize,
) -> Result<String, InputError> {
    if value.len() < minimum_digits
        || value.len() > maximum_digits
        || !value.len().is_multiple_of(2)
        || !value.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(InputError::InvalidHex { field });
    }
    Ok(value.to_ascii_uppercase())
}

fn parse_canonical_sha256_content_id(
    field: &'static str,
    value: String,
) -> Result<String, InputError> {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return Err(InputError::InvalidHex { field });
    };
    if hex.len() != 64
        || !hex
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err(InputError::InvalidHex { field });
    }
    Ok(value)
}

pub(crate) fn parse_hex_array<const N: usize>(
    field: &'static str,
    value: &str,
) -> Result<[u8; N], InputError> {
    if value.len() != N * 2 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(InputError::InvalidHex { field });
    }
    let mut bytes = [0; N];
    for (index, byte) in bytes.iter_mut().enumerate() {
        let offset = index * 2;
        *byte = u8::from_str_radix(&value[offset..offset + 2], 16)
            .map_err(|_| InputError::InvalidHex { field })?;
    }
    if bytes.iter().all(|byte| *byte == 0) {
        return Err(InputError::ZeroIdentity { field });
    }
    Ok(bytes)
}

pub(crate) fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

fn parse_serial_by_id_path(field: &'static str, value: String) -> Result<String, InputError> {
    if value.len() > MAX_ABSOLUTE_PATH_BYTES || value.contains("${") {
        return Err(InputError::InvalidSerialByIdPath { field });
    }
    let path = Path::new(&value);
    let components = path.components().collect::<Vec<_>>();
    let valid = matches!(
        components.as_slice(),
        [
            Component::RootDir,
            Component::Normal(dev),
            Component::Normal(serial),
            Component::Normal(by_id),
            Component::Normal(_)
        ] if *dev == "dev" && *serial == "serial" && *by_id == "by-id"
    );
    if !valid {
        return Err(InputError::InvalidSerialByIdPath { field });
    }
    Ok(value)
}

fn parse_absolute_source_path(
    field: &'static str,
    path: PathBuf,
) -> Result<AbsoluteSourcePath, InputError> {
    if path.as_os_str().len() > MAX_ABSOLUTE_PATH_BYTES
        || !path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::RootDir | Component::Normal(_)))
    {
        return Err(InputError::InvalidAbsoluteSourcePath { field, path });
    }
    Ok(AbsoluteSourcePath(path))
}

fn parse_relative_bundle_path(
    field: &'static str,
    value: String,
) -> Result<RelativeBundlePath, InputError> {
    if value.is_empty()
        || value.len() > MAX_RELATIVE_PATH_BYTES
        || value.contains("${")
        || Path::new(&value)
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(InputError::InvalidRelativeBundlePath { field, value });
    }
    Ok(RelativeBundlePath(value))
}

fn parse_soname(value: String) -> Result<String, InputError> {
    if value.is_empty()
        || value.len() > MAX_TEXT_BYTES
        || value.contains("${")
        || value.contains('/')
        || value.contains('\\')
        || value == "."
        || value == ".."
        || value.chars().any(char::is_control)
    {
        return Err(InputError::InvalidSoname);
    }
    Ok(value)
}

fn require_nonzero_u8(field: &'static str, value: u8) -> Result<u8, InputError> {
    if value == 0 {
        Err(InputError::NumericOutOfRange { field })
    } else {
        Ok(value)
    }
}

fn require_nonzero_u16(field: &'static str, value: u16) -> Result<u16, InputError> {
    if value == 0 {
        Err(InputError::NumericOutOfRange { field })
    } else {
        Ok(value)
    }
}

fn require_nonzero_u32(field: &'static str, value: u32) -> Result<u32, InputError> {
    if value == 0 {
        Err(InputError::NumericOutOfRange { field })
    } else {
        Ok(value)
    }
}

fn bounded_nonzero_u32(field: &'static str, value: u32, maximum: u32) -> Result<u32, InputError> {
    if value == 0 || value > maximum {
        Err(InputError::NumericOutOfRange { field })
    } else {
        Ok(value)
    }
}

fn bounded_nonzero_u64(field: &'static str, value: u64, maximum: u64) -> Result<u64, InputError> {
    if value == 0 || value > maximum {
        Err(InputError::NumericOutOfRange { field })
    } else {
        Ok(value)
    }
}

fn minimum_navigation_dataset_terminal_reserve(
    maximum_map_snapshot_bytes: u64,
) -> Result<u64, InputError> {
    let unrounded = maximum_map_snapshot_bytes
        .checked_add(MAX_NAVIGATION_DATASET_MANIFEST_BYTES)
        .and_then(|bytes| bytes.checked_add(MAX_WARM_START_SELECTION_BYTES))
        .ok_or(InputError::NavigationDatasetTerminalReserveArithmeticOverflow)?;
    let remainder = unrounded % NAVIGATION_DATASET_ADMISSION_FRAGMENT_BYTES;
    if remainder == 0 {
        return Ok(unrounded);
    }
    unrounded
        .checked_add(NAVIGATION_DATASET_ADMISSION_FRAGMENT_BYTES - remainder)
        .ok_or(InputError::NavigationDatasetTerminalReserveArithmeticOverflow)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProductionControllerProfileDto {
    schema_version: u32,
    admission_scope: String,
    admission_id: String,
    reviewer_id: String,
    controller: ProductionControllerDto,
    actuation: ProductionActuationDto,
    live_mode_policy: ProductionLiveModePolicyDto,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProductionControllerDto {
    controller_uid_hex: String,
    firmware_abi: u16,
    firmware_build_id: u32,
    actuator_config_fingerprint_hex: String,
    hardware_profile_claim_id: String,
    controller_ready_timeout_ms: u16,
    heartbeat_period_ms: u16,
    maximum_heartbeat_age_ms: u16,
    maximum_host_command_rate_hz: u16,
    serial_transmit_timeout_ms: u16,
    serial_applied_ack_timeout_ms: u16,
    controller_clock_abs_error_ppm_bound: u32,
    deadline_quantization_margin_ms: u16,
    expected_max_abs_pwm_percent: u8,
    expected_pwm_frequency_hz: u32,
    expected_watchdog_nominal_timeout_ms: u16,
    expected_neutral_output: NeutralOutput,
    expected_physical_stop_semantics: VerifiedPhysicalStopSemantics,
    command_udp_port: u16,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum NeutralOutput {
    BothLow,
    BothHigh,
    HighImpedance,
}

impl NeutralOutput {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::BothLow => "both_low",
            Self::BothHigh => "both_high",
            Self::HighImpedance => "high_impedance",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum VerifiedPhysicalStopSemantics {
    CoastVerified,
    BrakeVerified,
}

impl VerifiedPhysicalStopSemantics {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::CoastVerified => "coast_verified",
            Self::BrakeVerified => "brake_verified",
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProductionActuationDto {
    plant_model_id: String,
    plant_model_version: u32,
    operator_claimed_physical_approval: PhysicalApprovalDto,
    apply_ack_budget_ns: u64,
    stop_ack_budget_ns: u64,
    scheduling_guard_ns: u64,
    controller_motion_lease_ms: u16,
    controller_deadline_tolerance_ns: u64,
    maximum_uncommanded_motion_ns: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PhysicalApprovalDto {
    approval_id: String,
    approver_id: String,
    plant_dataset_content_id: String,
    plant_identification_method_id: String,
    plant_sample_count: u64,
    plant_fit_residuals: PlantFitResidualsDto,
    imu_calibration_id: String,
    stereo_calibration_id: String,
    tracking_camera_to_base_calibration_id: String,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PlantFitResidualsDto {
    pub(crate) left_velocity_rmse_mps: f64,
    pub(crate) right_velocity_rmse_mps: f64,
    pub(crate) yaw_rate_rmse_rad_s: f64,
    pub(crate) max_abs_velocity_error_mps: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProductionLiveModePolicyDto {
    pub(crate) startup: ProductionStartupMode,
    pub(crate) manual: ManualModePolicyDto,
    pub(crate) point_goal: PointGoalPolicyDto,
    pub(crate) frontier_explore: FrontierExplorePolicyDto,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ProductionStartupMode {
    DisarmedMapOnly,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "permission", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum ManualModePolicyDto {
    Disabled,
    ControlApi {
        authority_lease_ms: u64,
        maximum_abs_forward_velocity_mps: f64,
        maximum_abs_yaw_rate_rad_s: f64,
        maximum_command_age_ms: u64,
        deadman_timeout_ms: u64,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "permission", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum PointGoalPolicyDto {
    Disabled,
    ControlApi {
        authority_lease_ms: u64,
        maximum_runtime_ms: u64,
        arrival_tolerance_m: f64,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "permission", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum FrontierExplorePolicyDto {
    Disabled,
    ControlApi {
        authority_lease_ms: u64,
        boundary_minimum_x_m: f64,
        boundary_minimum_y_m: f64,
        boundary_maximum_x_m: f64,
        boundary_maximum_y_m: f64,
        maximum_runtime_ms: u64,
        maximum_frontier_goals: u32,
        arrival_tolerance_m: f64,
        clearance_from_known_obstacles_m: f64,
        maximum_grid_cells: u32,
        maximum_expanded_cells: u32,
        maximum_open_set_entries: u32,
        maximum_abs_yaw_rate_rad_s: f64,
        yaw_travel_limit_exclusive_rad: f64,
        maximum_scan_origin_displacement_m: f64,
        maximum_scan_duration_ms: u64,
        yaw_turn_direction: YawTurnDirection,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum YawTurnDirection {
    CounterClockwise,
    Clockwise,
}

#[derive(Debug)]
pub(crate) struct ProductionControllerProfile {
    pub(crate) admission_id: String,
    pub(crate) reviewer_id: String,
    pub(crate) controller: ProductionController,
    pub(crate) actuation: ProductionActuation,
    pub(crate) live_mode_policy: ProductionLiveModePolicyDto,
}

#[derive(Debug)]
pub(crate) struct ProductionController {
    pub(crate) controller_uid: [u8; 12],
    pub(crate) firmware_abi: u16,
    pub(crate) firmware_build_id: u32,
    pub(crate) actuator_config_fingerprint: [u8; 16],
    pub(crate) hardware_profile_claim_id: String,
    pub(crate) controller_ready_timeout_ms: u16,
    pub(crate) heartbeat_period_ms: u16,
    pub(crate) maximum_heartbeat_age_ms: u16,
    pub(crate) maximum_host_command_rate_hz: u16,
    pub(crate) serial_transmit_timeout_ms: u16,
    pub(crate) serial_applied_ack_timeout_ms: u16,
    pub(crate) controller_clock_abs_error_ppm_bound: u32,
    pub(crate) deadline_quantization_margin_ms: u16,
    pub(crate) expected_max_abs_pwm_percent: u8,
    pub(crate) expected_pwm_frequency_hz: u32,
    pub(crate) expected_watchdog_nominal_timeout_ms: u16,
    pub(crate) expected_neutral_output: NeutralOutput,
    pub(crate) expected_physical_stop_semantics: VerifiedPhysicalStopSemantics,
    pub(crate) command_udp_port: u16,
}

#[derive(Debug)]
pub(crate) struct ProductionActuation {
    pub(crate) plant_model_id: String,
    pub(crate) plant_model_version: u32,
    pub(crate) approval: PhysicalApproval,
    pub(crate) apply_ack_budget_ns: u64,
    pub(crate) stop_ack_budget_ns: u64,
    pub(crate) scheduling_guard_ns: u64,
    pub(crate) controller_motion_lease_ms: u16,
    pub(crate) controller_deadline_tolerance_ns: u64,
    pub(crate) maximum_uncommanded_motion_ns: u64,
}

#[derive(Debug)]
pub(crate) struct PhysicalApproval {
    pub(crate) approval_id: String,
    pub(crate) approver_id: String,
    pub(crate) plant_dataset_content_id: String,
    pub(crate) plant_identification_method_id: String,
    pub(crate) plant_sample_count: u64,
    pub(crate) residuals: PlantFitResidualsDto,
    pub(crate) imu_calibration_id: String,
    pub(crate) stereo_calibration_id: String,
    pub(crate) tracking_camera_to_base_calibration_id: String,
}

impl ProductionControllerProfile {
    pub(crate) fn parse(
        dto: ProductionControllerProfileDto,
        discovered: &Stm32Discovery,
    ) -> Result<Self, InputError> {
        if dto.schema_version != PRODUCTION_PROFILE_SCHEMA_VERSION {
            return Err(InputError::UnsupportedProductionProfileSchema {
                actual: dto.schema_version,
            });
        }
        if dto.admission_scope != PRODUCTION_ADMISSION_SCOPE {
            return Err(InputError::InvalidProductionAdmissionScope);
        }
        let controller = ProductionController {
            controller_uid: parse_hex_array(
                "production.controller.controller_uid_hex",
                &dto.controller.controller_uid_hex,
            )?,
            firmware_abi: require_nonzero_u16(
                "production.controller.firmware_abi",
                dto.controller.firmware_abi,
            )?,
            firmware_build_id: require_nonzero_u32(
                "production.controller.firmware_build_id",
                dto.controller.firmware_build_id,
            )?,
            actuator_config_fingerprint: parse_hex_array(
                "production.controller.actuator_config_fingerprint_hex",
                &dto.controller.actuator_config_fingerprint_hex,
            )?,
            hardware_profile_claim_id: parse_text(
                "production.controller.hardware_profile_claim_id",
                dto.controller.hardware_profile_claim_id,
            )?,
            controller_ready_timeout_ms: require_nonzero_u16(
                "production.controller.controller_ready_timeout_ms",
                dto.controller.controller_ready_timeout_ms,
            )?,
            heartbeat_period_ms: require_nonzero_u16(
                "production.controller.heartbeat_period_ms",
                dto.controller.heartbeat_period_ms,
            )?,
            maximum_heartbeat_age_ms: require_nonzero_u16(
                "production.controller.maximum_heartbeat_age_ms",
                dto.controller.maximum_heartbeat_age_ms,
            )?,
            maximum_host_command_rate_hz: require_nonzero_u16(
                "production.controller.maximum_host_command_rate_hz",
                dto.controller.maximum_host_command_rate_hz,
            )?,
            serial_transmit_timeout_ms: require_nonzero_u16(
                "production.controller.serial_transmit_timeout_ms",
                dto.controller.serial_transmit_timeout_ms,
            )?,
            serial_applied_ack_timeout_ms: require_nonzero_u16(
                "production.controller.serial_applied_ack_timeout_ms",
                dto.controller.serial_applied_ack_timeout_ms,
            )?,
            controller_clock_abs_error_ppm_bound: require_nonzero_u32(
                "production.controller.controller_clock_abs_error_ppm_bound",
                dto.controller.controller_clock_abs_error_ppm_bound,
            )?,
            deadline_quantization_margin_ms: require_nonzero_u16(
                "production.controller.deadline_quantization_margin_ms",
                dto.controller.deadline_quantization_margin_ms,
            )?,
            expected_max_abs_pwm_percent: require_nonzero_u8(
                "production.controller.expected_max_abs_pwm_percent",
                dto.controller.expected_max_abs_pwm_percent,
            )?,
            expected_pwm_frequency_hz: require_nonzero_u32(
                "production.controller.expected_pwm_frequency_hz",
                dto.controller.expected_pwm_frequency_hz,
            )?,
            expected_watchdog_nominal_timeout_ms: require_nonzero_u16(
                "production.controller.expected_watchdog_nominal_timeout_ms",
                dto.controller.expected_watchdog_nominal_timeout_ms,
            )?,
            expected_neutral_output: dto.controller.expected_neutral_output,
            expected_physical_stop_semantics: dto.controller.expected_physical_stop_semantics,
            command_udp_port: require_nonzero_u16(
                "production.controller.command_udp_port",
                dto.controller.command_udp_port,
            )?,
        };
        if controller.expected_max_abs_pwm_percent > 100
            || controller.maximum_heartbeat_age_ms <= controller.heartbeat_period_ms
        {
            return Err(InputError::NumericOutOfRange {
                field: "production.controller",
            });
        }
        if controller.controller_uid != discovered.controller_uid
            || controller.firmware_abi != discovered.firmware_abi
            || controller.firmware_build_id != discovered.firmware_build_id
            || controller.actuator_config_fingerprint != discovered.hardware_profile_fingerprint
        {
            return Err(InputError::ProductionControllerDiscoveryMismatch);
        }
        if controller.firmware_build_id == CANDIDATE_FIRMWARE_BUILD_ID
            || encode_hex(&controller.actuator_config_fingerprint) == CANDIDATE_FINGERPRINT_HEX
        {
            return Err(InputError::CandidateControllerIdentityForbiddenInProduction);
        }
        let capabilities = ControllerCapabilities::try_from_bits(discovered.capabilities_bits)
            .map_err(|_| InputError::InvalidProductionControllerCapabilities)?;
        if controller.firmware_abi != CANDIDATE_FIRMWARE_ABI
            || !capabilities.supports_required_safety()
            || capabilities.supports_operator_supervised_four_pwm_candidate()
        {
            return Err(InputError::InvalidProductionControllerCapabilities);
        }

        let residuals = dto
            .actuation
            .operator_claimed_physical_approval
            .plant_fit_residuals;
        if [
            residuals.left_velocity_rmse_mps,
            residuals.right_velocity_rmse_mps,
            residuals.yaw_rate_rmse_rad_s,
            residuals.max_abs_velocity_error_mps,
        ]
        .into_iter()
        .any(|value| !value.is_finite() || value < 0.0)
        {
            return Err(InputError::NumericOutOfRange {
                field: "production.actuation.plant_fit_residuals",
            });
        }
        let approval_dto = dto.actuation.operator_claimed_physical_approval;
        let approval = PhysicalApproval {
            approval_id: parse_text("production.approval.approval_id", approval_dto.approval_id)?,
            approver_id: parse_text("production.approval.approver_id", approval_dto.approver_id)?,
            plant_dataset_content_id: parse_canonical_sha256_content_id(
                "production.approval.plant_dataset_content_id",
                approval_dto.plant_dataset_content_id,
            )?,
            plant_identification_method_id: parse_text(
                "production.approval.plant_identification_method_id",
                approval_dto.plant_identification_method_id,
            )?,
            plant_sample_count: bounded_nonzero_u64(
                "production.approval.plant_sample_count",
                approval_dto.plant_sample_count,
                u64::MAX,
            )?,
            residuals,
            imu_calibration_id: parse_text(
                "production.approval.imu_calibration_id",
                approval_dto.imu_calibration_id,
            )?,
            stereo_calibration_id: parse_text(
                "production.approval.stereo_calibration_id",
                approval_dto.stereo_calibration_id,
            )?,
            tracking_camera_to_base_calibration_id: parse_text(
                "production.approval.tracking_camera_to_base_calibration_id",
                approval_dto.tracking_camera_to_base_calibration_id,
            )?,
        };
        validate_live_modes(&dto.live_mode_policy)?;
        let actuation = ProductionActuation {
            plant_model_id: parse_text(
                "production.actuation.plant_model_id",
                dto.actuation.plant_model_id,
            )?,
            plant_model_version: require_nonzero_u32(
                "production.actuation.plant_model_version",
                dto.actuation.plant_model_version,
            )?,
            approval,
            apply_ack_budget_ns: bounded_nonzero_u64(
                "production.actuation.apply_ack_budget_ns",
                dto.actuation.apply_ack_budget_ns,
                10_000_000_000,
            )?,
            stop_ack_budget_ns: bounded_nonzero_u64(
                "production.actuation.stop_ack_budget_ns",
                dto.actuation.stop_ack_budget_ns,
                10_000_000_000,
            )?,
            scheduling_guard_ns: bounded_nonzero_u64(
                "production.actuation.scheduling_guard_ns",
                dto.actuation.scheduling_guard_ns,
                10_000_000_000,
            )?,
            controller_motion_lease_ms: require_nonzero_u16(
                "production.actuation.controller_motion_lease_ms",
                dto.actuation.controller_motion_lease_ms,
            )?,
            controller_deadline_tolerance_ns: bounded_nonzero_u64(
                "production.actuation.controller_deadline_tolerance_ns",
                dto.actuation.controller_deadline_tolerance_ns,
                10_000_000_000,
            )?,
            maximum_uncommanded_motion_ns: bounded_nonzero_u64(
                "production.actuation.maximum_uncommanded_motion_ns",
                dto.actuation.maximum_uncommanded_motion_ns,
                60_000_000_000,
            )?,
        };
        Ok(Self {
            admission_id: parse_text("production.admission_id", dto.admission_id)?,
            reviewer_id: parse_text("production.reviewer_id", dto.reviewer_id)?,
            controller,
            actuation,
            live_mode_policy: dto.live_mode_policy,
        })
    }
}

fn validate_live_modes(policy: &ProductionLiveModePolicyDto) -> Result<(), InputError> {
    if let ManualModePolicyDto::ControlApi {
        authority_lease_ms,
        maximum_abs_forward_velocity_mps,
        maximum_abs_yaw_rate_rad_s,
        maximum_command_age_ms,
        deadman_timeout_ms,
    } = policy.manual
        && (authority_lease_ms == 0
            || authority_lease_ms > 1_000
            || !maximum_abs_forward_velocity_mps.is_finite()
            || maximum_abs_forward_velocity_mps <= 0.0
            || !maximum_abs_yaw_rate_rad_s.is_finite()
            || maximum_abs_yaw_rate_rad_s <= 0.0
            || maximum_command_age_ms == 0
            || maximum_command_age_ms > 60_000
            || deadman_timeout_ms == 0
            || deadman_timeout_ms > 60_000
            || maximum_command_age_ms > deadman_timeout_ms)
    {
        return Err(InputError::NumericOutOfRange {
            field: "production.live_mode_policy.manual",
        });
    }
    if let PointGoalPolicyDto::ControlApi {
        authority_lease_ms,
        maximum_runtime_ms,
        arrival_tolerance_m,
    } = policy.point_goal
        && (authority_lease_ms == 0
            || authority_lease_ms > 1_000
            || maximum_runtime_ms == 0
            || maximum_runtime_ms > 86_400_000
            || !arrival_tolerance_m.is_finite()
            || arrival_tolerance_m <= 0.0)
    {
        return Err(InputError::NumericOutOfRange {
            field: "production.live_mode_policy.point_goal",
        });
    }
    if let FrontierExplorePolicyDto::ControlApi {
        authority_lease_ms,
        boundary_minimum_x_m,
        boundary_minimum_y_m,
        boundary_maximum_x_m,
        boundary_maximum_y_m,
        maximum_runtime_ms,
        maximum_frontier_goals,
        arrival_tolerance_m,
        clearance_from_known_obstacles_m,
        maximum_grid_cells,
        maximum_expanded_cells,
        maximum_open_set_entries,
        maximum_abs_yaw_rate_rad_s,
        yaw_travel_limit_exclusive_rad,
        maximum_scan_origin_displacement_m,
        maximum_scan_duration_ms,
        yaw_turn_direction: _,
    } = policy.frontier_explore
        && (authority_lease_ms == 0
            || authority_lease_ms > 1_000
            || maximum_runtime_ms == 0
            || maximum_runtime_ms > 86_400_000
            || maximum_frontier_goals == 0
            || maximum_frontier_goals > 10_000
            || maximum_grid_cells == 0
            || maximum_grid_cells > 16_000_000
            || maximum_expanded_cells == 0
            || maximum_expanded_cells > maximum_grid_cells
            || maximum_open_set_entries == 0
            || u64::from(maximum_open_set_entries)
                > u64::from(maximum_grid_cells).saturating_mul(8)
            || maximum_scan_duration_ms == 0
            || maximum_scan_duration_ms > 86_400_000
            || [
                boundary_minimum_x_m,
                boundary_minimum_y_m,
                boundary_maximum_x_m,
                boundary_maximum_y_m,
                arrival_tolerance_m,
                clearance_from_known_obstacles_m,
                maximum_abs_yaw_rate_rad_s,
                yaw_travel_limit_exclusive_rad,
                maximum_scan_origin_displacement_m,
            ]
            .into_iter()
            .any(|value| !value.is_finite())
            || boundary_minimum_x_m >= boundary_maximum_x_m
            || boundary_minimum_y_m >= boundary_maximum_y_m
            || arrival_tolerance_m <= 0.0
            || clearance_from_known_obstacles_m < 0.0
            || maximum_abs_yaw_rate_rad_s <= 0.0
            || yaw_travel_limit_exclusive_rad <= 0.0
            || maximum_scan_origin_displacement_m <= 0.0)
    {
        return Err(InputError::NumericOutOfRange {
            field: "production.live_mode_policy.frontier_explore",
        });
    }
    Ok(())
}

#[derive(Debug)]
pub enum InputError {
    UnsupportedRenderInputSchema {
        actual: u32,
        expected: u32,
        bundle_kind: &'static str,
    },
    UnsupportedProductionProfileSchema {
        actual: u32,
    },
    InvalidProductionAdmissionScope,
    InvalidText {
        field: &'static str,
    },
    InvalidHex {
        field: &'static str,
    },
    ZeroIdentity {
        field: &'static str,
    },
    InvalidSerialByIdPath {
        field: &'static str,
    },
    InvalidAbsoluteSourcePath {
        field: &'static str,
        path: PathBuf,
    },
    InvalidRelativeBundlePath {
        field: &'static str,
        value: String,
    },
    ArtifactOutsideArtifactRoot {
        field: &'static str,
    },
    InvalidSoname,
    InvalidServoIds,
    DuplicateSerialByIdPath,
    DuplicateNativeLibraryRole {
        role: NativeLibraryRole,
    },
    WrongNativeLibrarySoname {
        role: NativeLibraryRole,
        expected: &'static str,
        actual: String,
    },
    IncompleteNativeLibrarySet,
    NumericOutOfRange {
        field: &'static str,
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
    InvalidHeadEnvelope {
        index: usize,
    },
    UnreviewedNaturalHeadPolicy,
    InvalidRgbExpressionPolicy,
    CandidateControllerIdentityMismatch,
    ProductionControllerDiscoveryMismatch,
    CandidateControllerIdentityForbiddenInProduction,
    InvalidProductionControllerCapabilities,
    WarmStartForbiddenInQualification,
    ProductionFacePerceptionAssetsRequired,
    QualificationFacePerceptionAssetsRequired,
    QualificationHeadGazePolicyNullForbidden,
    QualificationHeadGazeReviewEvidenceForbidden,
    ProductionHeadGazePolicyRequired,
    ProductionHeadGazeReviewEvidenceRequired,
    QualificationExecutablePathRequired,
}

impl fmt::Display for InputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedRenderInputSchema {
                actual,
                expected,
                bundle_kind,
            } => write!(
                formatter,
                "unsupported {bundle_kind} render-input schema {actual}; expected {expected}"
            ),
            Self::UnsupportedProductionProfileSchema { actual } => write!(
                formatter,
                "unsupported production-controller profile schema {actual}; expected {PRODUCTION_PROFILE_SCHEMA_VERSION}"
            ),
            Self::InvalidProductionAdmissionScope => write!(
                formatter,
                "production-controller profile has the wrong admission scope"
            ),
            Self::InvalidText { field } => write!(
                formatter,
                "{field} is empty, unbounded, unresolved, or contains control bytes"
            ),
            Self::InvalidHex { field } => write!(
                formatter,
                "{field} is not the required hexadecimal identity"
            ),
            Self::ZeroIdentity { field } => {
                write!(formatter, "{field} cannot be the all-zero identity")
            }
            Self::InvalidSerialByIdPath { field } => write!(
                formatter,
                "{field} must be one stable /dev/serial/by-id/<identity> path"
            ),
            Self::InvalidAbsoluteSourcePath { field, path } => write!(
                formatter,
                "{field} is not a canonical absolute source path: {}",
                path.display()
            ),
            Self::InvalidRelativeBundlePath { field, value } => write!(
                formatter,
                "{field} is not a canonical bundle-relative path: {value}"
            ),
            Self::ArtifactOutsideArtifactRoot { field } => write!(
                formatter,
                "{field} destination must be strictly beneath artifacts/"
            ),
            Self::InvalidSoname => {
                write!(formatter, "native-library soname is not one safe filename")
            }
            Self::InvalidServoIds => write!(
                formatter,
                "head servo IDs must be four distinct nonzero IDs"
            ),
            Self::DuplicateSerialByIdPath => write!(
                formatter,
                "STM32, head, and eye must have three distinct serial-by-id identities"
            ),
            Self::DuplicateNativeLibraryRole { role } => write!(
                formatter,
                "native-library role {} appears more than once",
                role.as_str()
            ),
            Self::WrongNativeLibrarySoname {
                role,
                expected,
                actual,
            } => write!(
                formatter,
                "native-library role {} requires Nano SONAME {expected:?}, got {actual:?}",
                role.as_str()
            ),
            Self::IncompleteNativeLibrarySet => write!(
                formatter,
                "native libraries must contain the exact four legacy roles and three direct OpenCV roles"
            ),
            Self::NumericOutOfRange { field } => write!(
                formatter,
                "{field} is zero, non-finite, inconsistent, or outside its hard bound"
            ),
            Self::NavigationDatasetTerminalReserveNotBelowMaximum {
                reserve_bytes,
                maximum_dataset_bytes,
            } => write!(
                formatter,
                "runtime.storage.navigation_dataset_terminal_reserve_bytes ({reserve_bytes}) must be below runtime.storage.maximum_navigation_dataset_bytes ({maximum_dataset_bytes})"
            ),
            Self::NavigationDatasetTerminalReserveTooSmall {
                reserve_bytes,
                minimum_bytes,
            } => write!(
                formatter,
                "runtime.storage.navigation_dataset_terminal_reserve_bytes ({reserve_bytes}) is below the checked fragment-rounded terminal minimum ({minimum_bytes})"
            ),
            Self::NavigationDatasetTerminalReserveArithmeticOverflow => {
                formatter.write_str("runtime.storage terminal-reserve arithmetic overflowed")
            }
            Self::InvalidHeadEnvelope { index } => write!(
                formatter,
                "head_policy joint {index} has an inconsistent start/target/travel envelope"
            ),
            Self::UnreviewedNaturalHeadPolicy => write!(
                formatter,
                "head policy does not exactly match Kiko's operator-confirmed natural target and reviewed startup, travel, and torque policy"
            ),
            Self::InvalidRgbExpressionPolicy => write!(
                formatter,
                "RGB expression policy is outside the runtime sampling, threshold, timing, amount, or gaze-extrinsics domain"
            ),
            Self::CandidateControllerIdentityMismatch => write!(
                formatter,
                "wheels-off discovery does not match the canonical candidate firmware ABI/build/fingerprint/capabilities"
            ),
            Self::ProductionControllerDiscoveryMismatch => write!(
                formatter,
                "production-controller profile identity does not exactly match the discovery record"
            ),
            Self::CandidateControllerIdentityForbiddenInProduction => write!(
                formatter,
                "candidate STM32 firmware identity is forbidden in a production controller profile"
            ),
            Self::InvalidProductionControllerCapabilities => write!(
                formatter,
                "production STM32 discovery must declare protocol V2 and the complete external-interlock safety capability set"
            ),
            Self::WarmStartForbiddenInQualification => formatter
                .write_str("wheels-off qualification cannot replay persisted production map state"),
            Self::ProductionFacePerceptionAssetsRequired => formatter.write_str(
                "production rendering requires exact frontal and profile face-cascade sources",
            ),
            Self::QualificationFacePerceptionAssetsRequired => formatter.write_str(
                "wheels-off qualification requires exact frontal and profile face-cascade sources",
            ),
            Self::QualificationHeadGazePolicyNullForbidden => formatter.write_str(
                "wheels-off qualification head-gaze policy must be omitted to disable it; explicit null is forbidden",
            ),
            Self::QualificationHeadGazeReviewEvidenceForbidden => formatter.write_str(
                "wheels-off qualification cannot carry physical head-gaze review evidence",
            ),
            Self::ProductionHeadGazePolicyRequired => formatter.write_str(
                "production render-input V2 requires an exact physical head-gaze policy source",
            ),
            Self::ProductionHeadGazeReviewEvidenceRequired => formatter.write_str(
                "production render-input V2 requires exact attended head-gaze review evidence",
            ),
            Self::QualificationExecutablePathRequired => formatter.write_str(
                "wheels-off qualification requires bundle.qualification_executable_path",
            ),
        }
    }
}

impl std::error::Error for InputError {}
