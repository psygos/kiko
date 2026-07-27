//! Offline, fail-closed preparation of one Nano calibration artifact and its
//! exactly matching shadow-navigation configuration.
//!
//! This tool opens no device and makes no calibration-quality claim. Its input
//! must explicitly carry the physically established tracking-camera-to-base
//! transform, native-IMU-to-base rotation, and source provenance. Basalt's
//! calibration convention is converted once at this boundary:
//!
//! `Basalt: corrected = A * raw - b`
//! `Kiko:   corrected = A * (raw - A^-1 * b)`

#![forbid(unsafe_code)]

use std::error::Error;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use clap::Parser;
use kiko_slam::dense::occupancy::{DepthCameraModel, DepthToTrackingCamera};
use kiko_slam::navigation::{
    NanoCalibrationArtifactV1, RawImuCalibration, ShadowNavigationConfigV1,
};
use rustix::fs::{CWD, RenameFlags, renameat_with};
use rustix::io::Errno;
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

const PREPARATION_SCHEMA_V1: u32 = 1;
const MAX_PREPARATION_BYTES: usize = 256 * 1024;
const MAX_NAVIGATION_TEMPLATE_BYTES: usize = 256 * 1024;
const MAX_SOURCE_ID_BYTES: usize = 1_024;
const MAX_BASELINE_RELATIVE_DISCREPANCY: f64 = 0.02;
const SHA256_HEX_BYTES: usize = 64;

const INPUT_FILE: &str = "calibration-preparation-input-v1.json";
const ARTIFACT_FILE: &str = "calibration-artifact-v1.json";
const NAVIGATION_FILE: &str = "navigation-shadow-v1.json";
const MAX_STAGING_NAME_ATTEMPTS: u64 = 128;
const TEMPLATE_TOKEN_PREFIX: &[u8; 2] = b"${";
const TRACKING_CAMERA_TO_BASE_REPLACEMENT_MARKER: &str =
    "${CALIBRATION_PREPARER_REPLACES_TRACKING_CAMERA_TO_BASE}";
const RAW_IMU_CALIBRATION_REPLACEMENT_MARKER: &str =
    "${CALIBRATION_PREPARER_REPLACES_RAW_IMU_CALIBRATION}";

static NEXT_STAGING_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Parser)]
#[command(
    name = "kiko-nano-calibration-prepare",
    about = "Prepare matching offline Nano calibration/navigation JSON; performs no device I/O"
)]
struct Cli {
    /// Strict preparation schema V1 with explicit physical transforms.
    #[arg(long)]
    input: PathBuf,

    /// Complete navigation-shadow V1 JSON whose calibration fields are replaced.
    #[arg(long)]
    navigation_template: PathBuf,

    /// New output directory. Existing paths are never overwritten.
    #[arg(long)]
    output_dir: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PreparationInputV1 {
    schema_version: u32,
    oak_mxid: String,
    rectified_stereo: SourcedRectifiedStereo,
    basalt_imu_calibration: SourcedBasaltImuCalibration,
    native_imu_to_base: SourcedRotationF64,
    tracking_camera_to_base: SourcedPoseF32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SourceProvenance {
    source_id: String,
    source_sha256_hex: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SourcedRectifiedStereo {
    provenance: SourceProvenance,
    corroborating_baseline_provenance: SourceProvenance,
    corroborating_baseline_relationship: BaselineSourceRelationship,
    rectified: bool,
    left: CameraIntrinsics,
    right: CameraIntrinsics,
    baseline_m: f32,
    corroborating_baseline_m: f32,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum BaselineSourceRelationship {
    IndependentlyDerived,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CameraIntrinsics {
    fx_px: f32,
    fy_px: f32,
    cx_px: f32,
    cy_px: f32,
    width_px: u32,
    height_px: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SourcedBasaltImuCalibration {
    provenance: SourceProvenance,
    calib_accel_bias_units: BasaltAccelCalibrationUnits,
    calib_gyro_bias_units: BasaltGyroCalibrationUnits,
    /// Basalt `CalibAccelBias` parameters `[b0..b2, s0..s5]`.
    calib_accel_bias: [f64; 9],
    /// Basalt `CalibGyroBias` parameters `[b0..b2, s0..s8]`.
    calib_gyro_bias: [f64; 12],
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BasaltAccelCalibrationUnits {
    bias: AccelerationUnit,
    scale: ScaleUnit,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BasaltGyroCalibrationUnits {
    bias: AngularVelocityUnit,
    scale: ScaleUnit,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum AccelerationUnit {
    MetresPerSecondSquared,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum AngularVelocityUnit {
    RadiansPerSecond,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum ScaleUnit {
    Dimensionless,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SourcedRotationF64 {
    provenance: SourceProvenance,
    rotation: [[f64; 3]; 3],
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SourcedPoseF32 {
    provenance: SourceProvenance,
    rotation: [[f32; 3]; 3],
    translation_m: [f32; 3],
}

#[derive(Debug, Serialize)]
struct CalibrationArtifactV1 {
    schema_version: u32,
    oak_mxid: String,
    imu_calibration_id: String,
    stereo_calibration_id: String,
    tracking_camera_to_base_calibration_id: String,
    rectified_stereo: RectifiedStereoOutput,
    raw_imu_calibration: RawImuCalibrationOutput,
    tracking_camera_to_base: PoseOutput,
}

#[derive(Debug, Serialize)]
struct RectifiedStereoOutput {
    rectified: bool,
    left: CameraIntrinsics,
    right: CameraIntrinsics,
    baseline_m: f32,
}

#[derive(Clone, Debug, Serialize)]
struct RawImuCalibrationOutput {
    format_version: u32,
    source_id: String,
    content_id: String,
    gyro_affine: [[f64; 3]; 3],
    gyro_bias_native_rad_per_sec: [f64; 3],
    accel_affine: [[f64; 3]; 3],
    accel_bias_native_m_per_sec2: [f64; 3],
    native_imu_to_base_rotation: [[f64; 3]; 3],
}

#[derive(Clone, Copy, Debug, Serialize)]
struct PoseOutput {
    rotation: [[f32; 3]; 3],
    translation_m: [f32; 3],
}

struct PreparedCalibration {
    artifact_json: Vec<u8>,
    navigation_json: Vec<u8>,
}

struct DuplicateKeyRejectingJsonValue(Value);

impl<'de> Deserialize<'de> for DuplicateKeyRejectingJsonValue {
    fn deserialize<Deserializer>(deserializer: Deserializer) -> Result<Self, Deserializer::Error>
    where
        Deserializer: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(DuplicateKeyRejectingJsonVisitor)
    }
}

struct DuplicateKeyRejectingJsonVisitor;

impl<'de> Visitor<'de> for DuplicateKeyRejectingJsonVisitor {
    type Value = DuplicateKeyRejectingJsonValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value without duplicate object keys")
    }

    fn visit_bool<Error>(self, value: bool) -> Result<Self::Value, Error> {
        Ok(DuplicateKeyRejectingJsonValue(Value::Bool(value)))
    }

    fn visit_i64<Error>(self, value: i64) -> Result<Self::Value, Error> {
        Ok(DuplicateKeyRejectingJsonValue(Value::Number(value.into())))
    }

    fn visit_u64<Error>(self, value: u64) -> Result<Self::Value, Error> {
        Ok(DuplicateKeyRejectingJsonValue(Value::Number(value.into())))
    }

    fn visit_f64<Error>(self, value: f64) -> Result<Self::Value, Error>
    where
        Error: de::Error,
    {
        let number = serde_json::Number::from_f64(value)
            .ok_or_else(|| Error::custom("non-finite JSON number"))?;
        Ok(DuplicateKeyRejectingJsonValue(Value::Number(number)))
    }

    fn visit_str<Error>(self, value: &str) -> Result<Self::Value, Error> {
        Ok(DuplicateKeyRejectingJsonValue(Value::String(
            value.to_owned(),
        )))
    }

    fn visit_string<Error>(self, value: String) -> Result<Self::Value, Error> {
        Ok(DuplicateKeyRejectingJsonValue(Value::String(value)))
    }

    fn visit_unit<Error>(self) -> Result<Self::Value, Error> {
        Ok(DuplicateKeyRejectingJsonValue(Value::Null))
    }

    fn visit_seq<Access>(self, mut sequence: Access) -> Result<Self::Value, Access::Error>
    where
        Access: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(DuplicateKeyRejectingJsonValue(value)) = sequence.next_element()? {
            values.push(value);
        }
        Ok(DuplicateKeyRejectingJsonValue(Value::Array(values)))
    }

    fn visit_map<Access>(self, mut object: Access) -> Result<Self::Value, Access::Error>
    where
        Access: MapAccess<'de>,
    {
        let mut fields = serde_json::Map::new();
        while let Some(name) = object.next_key::<String>()? {
            if fields.contains_key(&name) {
                return Err(de::Error::custom(format_args!(
                    "duplicate object key {name:?}"
                )));
            }
            let DuplicateKeyRejectingJsonValue(value) = object.next_value()?;
            fields.insert(name, value);
        }
        Ok(DuplicateKeyRejectingJsonValue(Value::Object(fields)))
    }
}

#[derive(Debug)]
enum PrepareError {
    InputTooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    NavigationTemplateTooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    UnresolvedInputTemplateToken {
        byte_offset: usize,
    },
    UnresolvedNavigationTemplateToken {
        byte_offset: usize,
    },
    UnresolvedDecodedInputTemplateToken {
        field: &'static str,
    },
    InputJson(serde_json::Error),
    InputTrailingData(serde_json::Error),
    UnsupportedSchema {
        actual: u32,
        supported: u32,
    },
    EmptySourceId {
        field: &'static str,
    },
    SourceIdTooLong {
        field: &'static str,
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    InvalidSourceSha256 {
        field: &'static str,
    },
    CorroboratingBaselineReusesSourceId {
        source_id: String,
    },
    CorroboratingBaselineReusesContent {
        source_sha256_hex: String,
    },
    CombinedRawImuSourceIdTooLong {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    RectifiedStereoRequired,
    InvalidBaseline {
        field: &'static str,
        value: f32,
    },
    BaselineMismatch {
        baseline_m: f32,
        corroborating_baseline_m: f32,
        relative_discrepancy: f64,
        maximum_relative_discrepancy: f64,
    },
    NonFiniteBasaltParameter {
        field: &'static str,
        index: usize,
        value: f64,
    },
    UnstableBasaltAffine {
        field: &'static str,
        pivot: f64,
        minimum_pivot: f64,
    },
    NonFiniteConvertedBias {
        field: &'static str,
        axis: usize,
        value: f64,
    },
    ArtifactEncode(serde_json::Error),
    ArtifactParse(kiko_slam::navigation::NanoCalibrationArtifactParseError),
    NavigationJson(serde_json::Error),
    NavigationReplacementMarkerMismatch {
        path: &'static str,
        expected: &'static str,
    },
    UnresolvedDecodedNavigationTemplateToken,
    MissingNavigationField {
        path: &'static str,
    },
    NavigationEncode(serde_json::Error),
    NavigationParse(kiko_slam::navigation::ShadowNavigationConfigParseError),
    NavigationBinding(kiko_slam::navigation::NanoCalibrationBindingError),
}

impl fmt::Display for PrepareError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputTooLarge {
                actual_bytes,
                maximum_bytes,
            } => write!(
                formatter,
                "preparation input is {actual_bytes} bytes; maximum is {maximum_bytes}"
            ),
            Self::NavigationTemplateTooLarge {
                actual_bytes,
                maximum_bytes,
            } => write!(
                formatter,
                "navigation template is {actual_bytes} bytes; maximum is {maximum_bytes}"
            ),
            Self::UnresolvedInputTemplateToken { byte_offset } => write!(
                formatter,
                "preparation input contains an unresolved template token at byte {byte_offset}"
            ),
            Self::UnresolvedNavigationTemplateToken { byte_offset } => write!(
                formatter,
                "navigation template contains an unresolved non-replacement token at byte \
                 {byte_offset}"
            ),
            Self::UnresolvedDecodedInputTemplateToken { field } => write!(
                formatter,
                "preparation input field {field} decodes to an unresolved template token"
            ),
            Self::InputJson(source) => {
                write!(
                    formatter,
                    "preparation input is not strict schema-V1 JSON: {source}"
                )
            }
            Self::InputTrailingData(source) => {
                write!(formatter, "preparation input has trailing data: {source}")
            }
            Self::UnsupportedSchema { actual, supported } => write!(
                formatter,
                "preparation schema {actual} is unsupported; expected {supported}"
            ),
            Self::EmptySourceId { field } => {
                write!(formatter, "{field}.source_id is empty")
            }
            Self::SourceIdTooLong {
                field,
                actual_bytes,
                maximum_bytes,
            } => write!(
                formatter,
                "{field}.source_id is {actual_bytes} bytes; maximum is {maximum_bytes}"
            ),
            Self::InvalidSourceSha256 { field } => write!(
                formatter,
                "{field}.source_sha256_hex must be 64 lowercase hexadecimal characters"
            ),
            Self::CorroboratingBaselineReusesSourceId { source_id } => write!(
                formatter,
                "corroborating baseline source_id {source_id:?} is the live stereo source_id; \
                 independently derived evidence must have a distinct source"
            ),
            Self::CorroboratingBaselineReusesContent { source_sha256_hex } => write!(
                formatter,
                "corroborating baseline source SHA-256 {source_sha256_hex} is the live stereo \
                 source SHA-256; independently derived evidence must have distinct content"
            ),
            Self::CombinedRawImuSourceIdTooLong {
                actual_bytes,
                maximum_bytes,
            } => write!(
                formatter,
                "combined Basalt/native-IMU provenance is {actual_bytes} bytes; \
                 raw-IMU source_id maximum is {maximum_bytes}"
            ),
            Self::RectifiedStereoRequired => {
                write!(formatter, "rectified_stereo.rectified must be true")
            }
            Self::InvalidBaseline { field, value } => {
                write!(
                    formatter,
                    "{field} must be positive finite metres; got {value}"
                )
            }
            Self::BaselineMismatch {
                baseline_m,
                corroborating_baseline_m,
                relative_discrepancy,
                maximum_relative_discrepancy,
            } => write!(
                formatter,
                "live baseline {baseline_m} m and corroborating baseline \
                 {corroborating_baseline_m} m differ by {relative_discrepancy:.6}; \
                 maximum is {maximum_relative_discrepancy:.6}"
            ),
            Self::NonFiniteBasaltParameter {
                field,
                index,
                value,
            } => write!(formatter, "{field}[{index}] must be finite; got {value}"),
            Self::UnstableBasaltAffine {
                field,
                pivot,
                minimum_pivot,
            } => write!(
                formatter,
                "{field} produces an unstable affine pivot {pivot}; minimum is {minimum_pivot}"
            ),
            Self::NonFiniteConvertedBias { field, axis, value } => write!(
                formatter,
                "{field} converted bias axis {axis} is non-finite: {value}"
            ),
            Self::ArtifactEncode(source) => {
                write!(formatter, "cannot encode calibration artifact: {source}")
            }
            Self::ArtifactParse(source) => {
                write!(
                    formatter,
                    "generated calibration artifact was rejected: {source}"
                )
            }
            Self::NavigationJson(source) => {
                write!(formatter, "navigation template is not JSON: {source}")
            }
            Self::NavigationReplacementMarkerMismatch { path, expected } => write!(
                formatter,
                "navigation template field {path} must equal the exact replacement marker \
                 {expected:?} before calibration preparation"
            ),
            Self::UnresolvedDecodedNavigationTemplateToken => formatter.write_str(
                "navigation template contains an unresolved template token after JSON decoding",
            ),
            Self::MissingNavigationField { path } => {
                write!(
                    formatter,
                    "navigation template is missing required field {path}"
                )
            }
            Self::NavigationEncode(source) => {
                write!(formatter, "cannot encode navigation output: {source}")
            }
            Self::NavigationParse(source) => {
                write!(formatter, "generated navigation was rejected: {source}")
            }
            Self::NavigationBinding(source) => write!(
                formatter,
                "generated artifact and navigation do not match exactly: {source}"
            ),
        }
    }
}

impl Error for PrepareError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InputJson(source)
            | Self::InputTrailingData(source)
            | Self::ArtifactEncode(source)
            | Self::NavigationJson(source)
            | Self::NavigationEncode(source) => Some(source),
            Self::ArtifactParse(source) => Some(source),
            Self::NavigationParse(source) => Some(source),
            Self::NavigationBinding(source) => Some(source),
            _ => None,
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let input_bytes = fs::read(&cli.input)?;
    let navigation_template_bytes = fs::read(&cli.navigation_template)?;
    let prepared = prepare(&input_bytes, &navigation_template_bytes)?;
    write_new_output_directory(
        &cli.output_dir,
        &input_bytes,
        &prepared.artifact_json,
        &prepared.navigation_json,
    )?;
    println!("{}", cli.output_dir.display());
    Ok(())
}

fn prepare(
    input_bytes: &[u8],
    navigation_template_bytes: &[u8],
) -> Result<PreparedCalibration, PrepareError> {
    if input_bytes.len() > MAX_PREPARATION_BYTES {
        return Err(PrepareError::InputTooLarge {
            actual_bytes: input_bytes.len(),
            maximum_bytes: MAX_PREPARATION_BYTES,
        });
    }
    if navigation_template_bytes.len() > MAX_NAVIGATION_TEMPLATE_BYTES {
        return Err(PrepareError::NavigationTemplateTooLarge {
            actual_bytes: navigation_template_bytes.len(),
            maximum_bytes: MAX_NAVIGATION_TEMPLATE_BYTES,
        });
    }
    if let Some(byte_offset) = first_disallowed_template_token(input_bytes, &[]) {
        return Err(PrepareError::UnresolvedInputTemplateToken { byte_offset });
    }
    if let Some(byte_offset) = first_disallowed_template_token(
        navigation_template_bytes,
        &[
            TRACKING_CAMERA_TO_BASE_REPLACEMENT_MARKER.as_bytes(),
            RAW_IMU_CALIBRATION_REPLACEMENT_MARKER.as_bytes(),
        ],
    ) {
        return Err(PrepareError::UnresolvedNavigationTemplateToken { byte_offset });
    }

    let mut deserializer = serde_json::Deserializer::from_slice(input_bytes);
    let input =
        PreparationInputV1::deserialize(&mut deserializer).map_err(PrepareError::InputJson)?;
    deserializer
        .end()
        .map_err(PrepareError::InputTrailingData)?;
    if input.schema_version != PREPARATION_SCHEMA_V1 {
        return Err(PrepareError::UnsupportedSchema {
            actual: input.schema_version,
            supported: PREPARATION_SCHEMA_V1,
        });
    }
    require_no_decoded_input_template_token("oak_mxid", &input.oak_mxid)?;

    validate_provenance(
        "rectified_stereo.provenance",
        &input.rectified_stereo.provenance,
    )?;
    validate_provenance(
        "rectified_stereo.corroborating_baseline_provenance",
        &input.rectified_stereo.corroborating_baseline_provenance,
    )?;
    validate_provenance(
        "basalt_imu_calibration.provenance",
        &input.basalt_imu_calibration.provenance,
    )?;
    validate_provenance(
        "native_imu_to_base.provenance",
        &input.native_imu_to_base.provenance,
    )?;
    validate_provenance(
        "tracking_camera_to_base.provenance",
        &input.tracking_camera_to_base.provenance,
    )?;
    require_independent_baseline_sources(
        &input.rectified_stereo.provenance,
        &input.rectified_stereo.corroborating_baseline_provenance,
    )?;

    if !input.rectified_stereo.rectified {
        return Err(PrepareError::RectifiedStereoRequired);
    }
    validate_baseline(
        "rectified_stereo.baseline_m",
        input.rectified_stereo.baseline_m,
    )?;
    validate_baseline(
        "rectified_stereo.corroborating_baseline_m",
        input.rectified_stereo.corroborating_baseline_m,
    )?;
    require_consistent_baselines(
        input.rectified_stereo.baseline_m,
        input.rectified_stereo.corroborating_baseline_m,
    )?;

    let (accel_affine, accel_bias) =
        convert_basalt_accel(input.basalt_imu_calibration.calib_accel_bias)?;
    let (gyro_affine, gyro_bias) =
        convert_basalt_gyro(input.basalt_imu_calibration.calib_gyro_bias)?;

    let input_sha256 = Sha256::digest(input_bytes);
    let input_sha256_hex = lowercase_hex(&input_sha256);
    let imu_calibration_id = format!("imu@sha256:{input_sha256_hex}");
    let stereo_calibration_id = format!("stereo@sha256:{input_sha256_hex}");
    let tracking_camera_to_base_calibration_id = format!("camera-base@sha256:{input_sha256_hex}");
    let raw_imu_source_id = format!(
        "{}#sha256:{}+{}#sha256:{}",
        input.basalt_imu_calibration.provenance.source_id,
        input.basalt_imu_calibration.provenance.source_sha256_hex,
        input.native_imu_to_base.provenance.source_id,
        input.native_imu_to_base.provenance.source_sha256_hex,
    );
    if raw_imu_source_id.len() > RawImuCalibration::MAX_SOURCE_ID_BYTES {
        return Err(PrepareError::CombinedRawImuSourceIdTooLong {
            actual_bytes: raw_imu_source_id.len(),
            maximum_bytes: RawImuCalibration::MAX_SOURCE_ID_BYTES,
        });
    }

    let raw_imu = RawImuCalibrationOutput {
        format_version: 1,
        source_id: raw_imu_source_id,
        content_id: imu_calibration_id.clone(),
        gyro_affine,
        gyro_bias_native_rad_per_sec: gyro_bias,
        accel_affine,
        accel_bias_native_m_per_sec2: accel_bias,
        native_imu_to_base_rotation: input.native_imu_to_base.rotation,
    };
    let tracking_camera_to_base = PoseOutput {
        rotation: input.tracking_camera_to_base.rotation,
        translation_m: input.tracking_camera_to_base.translation_m,
    };
    let artifact = CalibrationArtifactV1 {
        schema_version: 1,
        oak_mxid: input.oak_mxid,
        imu_calibration_id,
        stereo_calibration_id,
        tracking_camera_to_base_calibration_id,
        rectified_stereo: RectifiedStereoOutput {
            rectified: true,
            left: input.rectified_stereo.left,
            right: input.rectified_stereo.right,
            baseline_m: input.rectified_stereo.baseline_m,
        },
        raw_imu_calibration: raw_imu.clone(),
        tracking_camera_to_base,
    };
    let mut artifact_json =
        serde_json::to_vec_pretty(&artifact).map_err(PrepareError::ArtifactEncode)?;
    artifact_json.push(b'\n');
    let parsed_artifact = NanoCalibrationArtifactV1::parse_json(&artifact_json)
        .map_err(PrepareError::ArtifactParse)?;

    let DuplicateKeyRejectingJsonValue(mut navigation) =
        serde_json::from_slice(navigation_template_bytes).map_err(PrepareError::NavigationJson)?;
    require_navigation_replacement_marker(
        &navigation,
        &["coordinate_frames", "tracking_camera_to_base"],
        "coordinate_frames.tracking_camera_to_base",
        TRACKING_CAMERA_TO_BASE_REPLACEMENT_MARKER,
    )?;
    require_navigation_replacement_marker(
        &navigation,
        &["odometry", "raw_imu_calibration"],
        "odometry.raw_imu_calibration",
        RAW_IMU_CALIBRATION_REPLACEMENT_MARKER,
    )?;
    replace_navigation_field(
        &mut navigation,
        &["coordinate_frames", "tracking_camera_to_base"],
        serde_json::to_value(tracking_camera_to_base).map_err(PrepareError::NavigationEncode)?,
    )?;
    replace_navigation_field(
        &mut navigation,
        &["odometry", "raw_imu_calibration"],
        serde_json::to_value(raw_imu).map_err(PrepareError::NavigationEncode)?,
    )?;
    if value_contains_template_token(&navigation) {
        return Err(PrepareError::UnresolvedDecodedNavigationTemplateToken);
    }
    let mut navigation_json =
        serde_json::to_vec_pretty(&navigation).map_err(PrepareError::NavigationEncode)?;
    navigation_json.push(b'\n');

    let stereo = parsed_artifact.rectified_stereo();
    let runtime_depth_camera = DepthCameraModel::new(
        stereo.left(),
        stereo.dimensions(),
        DepthToTrackingCamera::identity(),
    );
    let parsed_navigation =
        ShadowNavigationConfigV1::parse_json(&navigation_json, runtime_depth_camera)
            .map_err(PrepareError::NavigationParse)?;
    parsed_artifact
        .require_navigation(&parsed_navigation)
        .map_err(PrepareError::NavigationBinding)?;

    Ok(PreparedCalibration {
        artifact_json,
        navigation_json,
    })
}

fn first_disallowed_template_token(bytes: &[u8], allowed: &[&[u8]]) -> Option<usize> {
    let mut search_start = 0;
    while search_start < bytes.len() {
        let relative_offset = bytes[search_start..]
            .windows(TEMPLATE_TOKEN_PREFIX.len())
            .position(|window| window == TEMPLATE_TOKEN_PREFIX)?;
        let byte_offset = search_start + relative_offset;
        if let Some(marker) = allowed
            .iter()
            .find(|marker| bytes[byte_offset..].starts_with(marker))
        {
            search_start = byte_offset + marker.len();
        } else {
            return Some(byte_offset);
        }
    }
    None
}

fn require_no_decoded_input_template_token(
    field: &'static str,
    value: &str,
) -> Result<(), PrepareError> {
    if value.contains("${") {
        return Err(PrepareError::UnresolvedDecodedInputTemplateToken { field });
    }
    Ok(())
}

fn validate_provenance(
    field: &'static str,
    provenance: &SourceProvenance,
) -> Result<(), PrepareError> {
    require_no_decoded_input_template_token(field, &provenance.source_id)?;
    require_no_decoded_input_template_token(field, &provenance.source_sha256_hex)?;
    if provenance.source_id.trim().is_empty() {
        return Err(PrepareError::EmptySourceId { field });
    }
    if provenance.source_id.len() > MAX_SOURCE_ID_BYTES {
        return Err(PrepareError::SourceIdTooLong {
            field,
            actual_bytes: provenance.source_id.len(),
            maximum_bytes: MAX_SOURCE_ID_BYTES,
        });
    }
    if provenance.source_sha256_hex.len() != SHA256_HEX_BYTES
        || !provenance
            .source_sha256_hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(PrepareError::InvalidSourceSha256 { field });
    }
    Ok(())
}

fn validate_baseline(field: &'static str, value: f32) -> Result<(), PrepareError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(PrepareError::InvalidBaseline { field, value });
    }
    Ok(())
}

fn require_independent_baseline_sources(
    live: &SourceProvenance,
    corroborating: &SourceProvenance,
) -> Result<(), PrepareError> {
    if live.source_id == corroborating.source_id {
        return Err(PrepareError::CorroboratingBaselineReusesSourceId {
            source_id: live.source_id.clone(),
        });
    }
    if live.source_sha256_hex == corroborating.source_sha256_hex {
        return Err(PrepareError::CorroboratingBaselineReusesContent {
            source_sha256_hex: live.source_sha256_hex.clone(),
        });
    }
    Ok(())
}

fn require_consistent_baselines(
    baseline_m: f32,
    corroborating_baseline_m: f32,
) -> Result<(), PrepareError> {
    let left = f64::from(baseline_m);
    let right = f64::from(corroborating_baseline_m);
    let relative_discrepancy = (left - right).abs() / left.max(right);
    if relative_discrepancy > MAX_BASELINE_RELATIVE_DISCREPANCY {
        return Err(PrepareError::BaselineMismatch {
            baseline_m,
            corroborating_baseline_m,
            relative_discrepancy,
            maximum_relative_discrepancy: MAX_BASELINE_RELATIVE_DISCREPANCY,
        });
    }
    Ok(())
}

fn convert_basalt_accel(parameters: [f64; 9]) -> Result<([[f64; 3]; 3], [f64; 3]), PrepareError> {
    validate_basalt_parameters("calib_accel_bias", &parameters)?;
    let affine = [
        [1.0 + parameters[3], 0.0, 0.0],
        [parameters[4], 1.0 + parameters[6], 0.0],
        [parameters[5], parameters[7], 1.0 + parameters[8]],
    ];
    let bias = solve_affine_bias(
        "calib_accel_bias",
        affine,
        [parameters[0], parameters[1], parameters[2]],
    )?;
    Ok((affine, bias))
}

fn convert_basalt_gyro(parameters: [f64; 12]) -> Result<([[f64; 3]; 3], [f64; 3]), PrepareError> {
    validate_basalt_parameters("calib_gyro_bias", &parameters)?;
    let affine = [
        [1.0 + parameters[3], parameters[6], parameters[9]],
        [parameters[4], 1.0 + parameters[7], parameters[10]],
        [parameters[5], parameters[8], 1.0 + parameters[11]],
    ];
    let bias = solve_affine_bias(
        "calib_gyro_bias",
        affine,
        [parameters[0], parameters[1], parameters[2]],
    )?;
    Ok((affine, bias))
}

fn validate_basalt_parameters(field: &'static str, parameters: &[f64]) -> Result<(), PrepareError> {
    for (index, &value) in parameters.iter().enumerate() {
        if !value.is_finite() {
            return Err(PrepareError::NonFiniteBasaltParameter {
                field,
                index,
                value,
            });
        }
    }
    Ok(())
}

/// Solve `affine * canonical_bias = basalt_bias` with partial pivoting.
fn solve_affine_bias(
    field: &'static str,
    affine: [[f64; 3]; 3],
    basalt_bias: [f64; 3],
) -> Result<[f64; 3], PrepareError> {
    let mut augmented = [[0.0; 4]; 3];
    let scale = affine
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let minimum_pivot = f64::EPSILON * scale.max(1.0) * 64.0;
    for row in 0..3 {
        augmented[row][..3].copy_from_slice(&affine[row]);
        augmented[row][3] = basalt_bias[row];
    }

    for pivot_column in 0..3 {
        let pivot_row = (pivot_column..3)
            .max_by(|left, right| {
                augmented[*left][pivot_column]
                    .abs()
                    .total_cmp(&augmented[*right][pivot_column].abs())
            })
            .expect("nonempty fixed pivot range");
        let pivot = augmented[pivot_row][pivot_column].abs();
        if !pivot.is_finite() || pivot <= minimum_pivot {
            return Err(PrepareError::UnstableBasaltAffine {
                field,
                pivot,
                minimum_pivot,
            });
        }
        augmented.swap(pivot_column, pivot_row);
        let divisor = augmented[pivot_column][pivot_column];
        for value in augmented[pivot_column].iter_mut().skip(pivot_column) {
            *value /= divisor;
        }
        let normalized_pivot_row = augmented[pivot_column];
        for (row_index, row) in augmented.iter_mut().enumerate() {
            if row_index == pivot_column {
                continue;
            }
            let factor = row[pivot_column];
            for (value, pivot_value) in row.iter_mut().zip(normalized_pivot_row).skip(pivot_column)
            {
                *value -= factor * pivot_value;
            }
        }
    }
    let result = [augmented[0][3], augmented[1][3], augmented[2][3]];
    for (axis, &value) in result.iter().enumerate() {
        if !value.is_finite() {
            return Err(PrepareError::NonFiniteConvertedBias { field, axis, value });
        }
    }
    Ok(result)
}

fn replace_navigation_field(
    document: &mut Value,
    path: &[&'static str],
    replacement: Value,
) -> Result<(), PrepareError> {
    let mut current = document;
    for component in &path[..path.len() - 1] {
        current = current
            .as_object_mut()
            .and_then(|object| object.get_mut(*component))
            .ok_or(PrepareError::MissingNavigationField { path: path[0] })?;
    }
    let leaf = path[path.len() - 1];
    let object = current
        .as_object_mut()
        .ok_or(PrepareError::MissingNavigationField { path: path[0] })?;
    if !object.contains_key(leaf) {
        return Err(PrepareError::MissingNavigationField { path: leaf });
    }
    object.insert(leaf.to_owned(), replacement);
    Ok(())
}

fn require_navigation_replacement_marker(
    document: &Value,
    path: &[&str],
    display_path: &'static str,
    expected: &'static str,
) -> Result<(), PrepareError> {
    let mut current = document;
    for component in path {
        current = current
            .as_object()
            .and_then(|object| object.get(*component))
            .ok_or(PrepareError::NavigationReplacementMarkerMismatch {
                path: display_path,
                expected,
            })?;
    }
    if current.as_str() != Some(expected) {
        return Err(PrepareError::NavigationReplacementMarkerMismatch {
            path: display_path,
            expected,
        });
    }
    Ok(())
}

fn value_contains_template_token(value: &Value) -> bool {
    match value {
        Value::String(text) => text.contains("${"),
        Value::Array(values) => values.iter().any(value_contains_template_token),
        Value::Object(fields) => fields
            .iter()
            .any(|(name, value)| name.contains("${") || value_contains_template_token(value)),
        Value::Null | Value::Bool(_) | Value::Number(_) => false,
    }
}

fn lowercase_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PublicationPhase {
    InspectOutput,
    CreateStagingDirectory,
    CreateFile,
    WriteFile,
    SyncFile,
    SyncStagingDirectory,
    InspectOutputBeforePublish,
    AtomicRename,
    SyncParentDirectory,
}

impl fmt::Display for PublicationPhase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::InspectOutput => "inspect output path",
            Self::CreateStagingDirectory => "create sibling staging directory",
            Self::CreateFile => "create staging file",
            Self::WriteFile => "write complete staging file",
            Self::SyncFile => "sync staging file",
            Self::SyncStagingDirectory => "sync staging directory",
            Self::InspectOutputBeforePublish => "recheck output path before publication",
            Self::AtomicRename => "atomically rename staging directory",
            Self::SyncParentDirectory => "sync published directory entry",
        };
        formatter.write_str(text)
    }
}

#[derive(Debug)]
enum PublicationError {
    InvalidOutputPath {
        path: PathBuf,
    },
    OutputAlreadyExists {
        path: PathBuf,
    },
    StagingNameExhausted {
        output_path: PathBuf,
        attempts: u64,
    },
    StagingCleanup {
        primary: Box<Self>,
        path: PathBuf,
        source: io::Error,
    },
    Io {
        phase: PublicationPhase,
        path: PathBuf,
        source: io::Error,
        output_published: bool,
    },
}

impl fmt::Display for PublicationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOutputPath { path } => write!(
                formatter,
                "output path {} has no final path component",
                path.display()
            ),
            Self::OutputAlreadyExists { path } => {
                write!(formatter, "output path {} already exists", path.display())
            }
            Self::StagingNameExhausted {
                output_path,
                attempts,
            } => write!(
                formatter,
                "could not reserve a unique sibling staging directory for {} after {attempts} attempts",
                output_path.display()
            ),
            Self::StagingCleanup {
                primary,
                path,
                source,
            } => write!(
                formatter,
                "{primary}; additionally failed to clean staging directory {}: {source}",
                path.display()
            ),
            Self::Io {
                phase,
                path,
                source,
                output_published,
            } => {
                write!(
                    formatter,
                    "failed to {phase} at {}: {source}",
                    path.display()
                )?;
                if *output_published {
                    formatter.write_str(
                        "; the complete output is visible, but its directory entry durability is unconfirmed",
                    )?;
                }
                Ok(())
            }
        }
    }
}

impl Error for PublicationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::StagingCleanup { primary, .. } => Some(primary.as_ref()),
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

fn write_new_output_directory(
    output_dir: &Path,
    input_bytes: &[u8],
    artifact_json: &[u8],
    navigation_json: &[u8],
) -> Result<(), PublicationError> {
    publish_files_transactionally(
        output_dir,
        &[
            (INPUT_FILE, input_bytes),
            (ARTIFACT_FILE, artifact_json),
            (NAVIGATION_FILE, navigation_json),
        ],
    )
}

fn publish_files_transactionally(
    output_dir: &Path,
    files: &[(&str, &[u8])],
) -> Result<(), PublicationError> {
    publish_files_transactionally_with_hook(output_dir, files, || {})
}

fn publish_files_transactionally_with_hook(
    output_dir: &Path,
    files: &[(&str, &[u8])],
    before_atomic_publish: impl FnOnce(),
) -> Result<(), PublicationError> {
    require_output_absent(output_dir, PublicationPhase::InspectOutput)?;
    let (parent, output_name) = output_parent_and_name(output_dir)?;
    let staging_dir = create_unique_staging_directory(parent, output_name, output_dir)?;

    if let Err(failure) = write_and_sync_staging_directory(&staging_dir, files) {
        return Err(clean_staging_after_failure(failure, &staging_dir));
    }
    if let Err(failure) =
        require_output_absent(output_dir, PublicationPhase::InspectOutputBeforePublish)
    {
        return Err(clean_staging_after_failure(failure, &staging_dir));
    }
    before_atomic_publish();
    if let Err(source) = renameat_with(CWD, &staging_dir, CWD, output_dir, RenameFlags::NOREPLACE) {
        let failure = if source == Errno::EXIST {
            PublicationError::OutputAlreadyExists {
                path: output_dir.to_path_buf(),
            }
        } else {
            PublicationError::Io {
                phase: PublicationPhase::AtomicRename,
                path: output_dir.to_path_buf(),
                source: io::Error::from_raw_os_error(source.raw_os_error()),
                output_published: false,
            }
        };
        return Err(clean_staging_after_failure(failure, &staging_dir));
    }

    let parent_directory = File::open(parent).map_err(|source| PublicationError::Io {
        phase: PublicationPhase::SyncParentDirectory,
        path: parent.to_path_buf(),
        source,
        output_published: true,
    })?;
    parent_directory
        .sync_all()
        .map_err(|source| PublicationError::Io {
            phase: PublicationPhase::SyncParentDirectory,
            path: parent.to_path_buf(),
            source,
            output_published: true,
        })
}

fn output_parent_and_name(output_dir: &Path) -> Result<(&Path, &str), PublicationError> {
    let parent = output_dir
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let output_name = output_dir
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| PublicationError::InvalidOutputPath {
            path: output_dir.to_path_buf(),
        })?;
    Ok((parent, output_name))
}

fn require_output_absent(
    output_dir: &Path,
    phase: PublicationPhase,
) -> Result<(), PublicationError> {
    match fs::symlink_metadata(output_dir) {
        Ok(_) => Err(PublicationError::OutputAlreadyExists {
            path: output_dir.to_path_buf(),
        }),
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(PublicationError::Io {
            phase,
            path: output_dir.to_path_buf(),
            source,
            output_published: false,
        }),
    }
}

fn create_unique_staging_directory(
    parent: &Path,
    output_name: &str,
    output_dir: &Path,
) -> Result<PathBuf, PublicationError> {
    let process_id = std::process::id();
    for _ in 0..MAX_STAGING_NAME_ATTEMPTS {
        let staging_id = NEXT_STAGING_ID.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(".{output_name}.staging-{process_id}-{staging_id}"));
        match fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {}
            Err(source) => {
                return Err(PublicationError::Io {
                    phase: PublicationPhase::CreateStagingDirectory,
                    path: candidate,
                    source,
                    output_published: false,
                });
            }
        }
    }
    Err(PublicationError::StagingNameExhausted {
        output_path: output_dir.to_path_buf(),
        attempts: MAX_STAGING_NAME_ATTEMPTS,
    })
}

fn write_and_sync_staging_directory(
    staging_dir: &Path,
    files: &[(&str, &[u8])],
) -> Result<(), PublicationError> {
    for &(relative_path, bytes) in files {
        let path = staging_dir.join(relative_path);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|source| PublicationError::Io {
                phase: PublicationPhase::CreateFile,
                path: path.clone(),
                source,
                output_published: false,
            })?;
        file.write_all(bytes)
            .map_err(|source| PublicationError::Io {
                phase: PublicationPhase::WriteFile,
                path: path.clone(),
                source,
                output_published: false,
            })?;
        file.sync_all().map_err(|source| PublicationError::Io {
            phase: PublicationPhase::SyncFile,
            path,
            source,
            output_published: false,
        })?;
    }

    let staging_directory = File::open(staging_dir).map_err(|source| PublicationError::Io {
        phase: PublicationPhase::SyncStagingDirectory,
        path: staging_dir.to_path_buf(),
        source,
        output_published: false,
    })?;
    staging_directory
        .sync_all()
        .map_err(|source| PublicationError::Io {
            phase: PublicationPhase::SyncStagingDirectory,
            path: staging_dir.to_path_buf(),
            source,
            output_published: false,
        })
}

fn clean_staging_after_failure(primary: PublicationError, staging_dir: &Path) -> PublicationError {
    match fs::remove_dir_all(staging_dir) {
        Ok(()) => primary,
        Err(source) if source.kind() == io::ErrorKind::NotFound => primary,
        Err(source) => PublicationError::StagingCleanup {
            primary: Box::new(primary),
            path: staging_dir.to_path_buf(),
            source,
        },
    }
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};

    use serde_json::json;

    use super::*;

    const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const IDENTITY_F32: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    const IDENTITY_F64: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    static NEXT_TEST_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            for _ in 0..128 {
                let sequence = NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed);
                let path = std::env::temp_dir().join(format!(
                    "kiko-calibration-prepare-test-{}-{sequence}",
                    std::process::id()
                ));
                match fs::create_dir(&path) {
                    Ok(()) => return Self(path),
                    Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {}
                    Err(source) => panic!("create temporary test directory: {source}"),
                }
            }
            panic!("could not reserve temporary test directory");
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn template_with_unquoted_tokens_replaced_by_null(template: &str) -> Value {
        let input = template.as_bytes();
        let mut output = Vec::with_capacity(input.len());
        let mut index = 0;
        let mut in_string = false;
        let mut escaped = false;
        while index < input.len() {
            let byte = input[index];
            if !in_string && byte == b'$' && input.get(index + 1) == Some(&b'{') {
                let relative_end = input[index + 2..]
                    .iter()
                    .position(|candidate| *candidate == b'}')
                    .expect("template token must close");
                index += relative_end + 3;
                output.extend_from_slice(b"null");
                continue;
            }

            output.push(byte);
            if in_string {
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'"' {
                    in_string = false;
                }
            } else if byte == b'"' {
                in_string = true;
            }
            index += 1;
        }
        serde_json::from_slice(&output).expect("sentinelized template shape must be JSON")
    }

    fn fill_navigation_sentinels_from_fixture(template: &mut Value, fixture: &Value) {
        match template {
            Value::Null => *template = fixture.clone(),
            Value::String(value) if value.starts_with("${NAV_SHADOW_UNVALIDATED_") => {
                *template = fixture.clone();
            }
            Value::String(value) if value.starts_with("${CALIBRATION_PREPARER_REPLACES_") => {}
            Value::Object(template_fields) => {
                let fixture_fields = fixture
                    .as_object()
                    .expect("fixture field must match template object");
                assert_eq!(
                    template_fields.len(),
                    fixture_fields.len(),
                    "template and fixture object fields must be complete"
                );
                for (name, template_field) in template_fields {
                    let fixture_field = fixture_fields
                        .get(name)
                        .unwrap_or_else(|| panic!("fixture is missing template field {name}"));
                    fill_navigation_sentinels_from_fixture(template_field, fixture_field);
                }
            }
            Value::Array(template_items) => {
                let fixture_items = fixture
                    .as_array()
                    .expect("fixture field must match template array");
                assert_eq!(
                    template_items.len(),
                    fixture_items.len(),
                    "template and fixture array lengths must match"
                );
                for (template_item, fixture_item) in template_items.iter_mut().zip(fixture_items) {
                    fill_navigation_sentinels_from_fixture(template_item, fixture_item);
                }
            }
            fixed => assert_eq!(
                fixed, fixture,
                "fixed template value must agree with the domain-valid fixture"
            ),
        }
    }

    const CORROBORATING_SHA: &str =
        "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn provenance(source_id: &str, source_sha256_hex: &str) -> Value {
        json!({"source_id": source_id, "source_sha256_hex": source_sha256_hex})
    }

    fn valid_input() -> Value {
        json!({
            "schema_version": 1,
            "oak_mxid": "19443010F1B43A2E00",
            "rectified_stereo": {
                "provenance": provenance("nano-live-rectified-stereo", SHA),
                "corroborating_baseline_provenance":
                    provenance("independent-rectified-projection-baseline", CORROBORATING_SHA),
                "corroborating_baseline_relationship": "independently_derived",
                "rectified": true,
                "left": {
                    "fx_px": 400.0, "fy_px": 400.0,
                    "cx_px": 320.0, "cy_px": 200.0,
                    "width_px": 640, "height_px": 400
                },
                "right": {
                    "fx_px": 400.0, "fy_px": 400.0,
                    "cx_px": 320.0, "cy_px": 200.0,
                    "width_px": 640, "height_px": 400
                },
                "baseline_m": 0.075,
                "corroborating_baseline_m": 0.07503247
            },
            "basalt_imu_calibration": {
                "provenance": provenance("basalt-stable-first-pass", SHA),
                "calib_accel_bias_units": {
                    "bias": "metres_per_second_squared",
                    "scale": "dimensionless"
                },
                "calib_gyro_bias_units": {
                    "bias": "radians_per_second",
                    "scale": "dimensionless"
                },
                "calib_accel_bias": [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 2.0, 0.0, 3.0],
                "calib_gyro_bias": [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0]
            },
            "native_imu_to_base": {
                "provenance": provenance("measured-native-imu-to-base", SHA),
                "rotation": IDENTITY_F64
            },
            "tracking_camera_to_base": {
                "provenance": provenance("measured-camera-to-base", SHA),
                "rotation": IDENTITY_F32,
                "translation_m": [0.20, 0.0, 0.25]
            }
        })
    }

    /// Synthetic non-calibration parser fixture only; never a recommended
    /// deployable input. The two calibration leaves are converted to the exact
    /// preparation markers before use.
    fn synthetic_navigation_fixture() -> &'static [u8] {
        static FIXTURE: OnceLock<Vec<u8>> = OnceLock::new();
        FIXTURE
            .get_or_init(|| {
                let mut fixture: Value = serde_json::from_slice(include_bytes!(
                    "../../../../configs/navigation-shadow-v1.example.json"
                ))
                .expect("synthetic navigation fixture");
                fixture["coordinate_frames"]["tracking_camera_to_base"] =
                    json!(TRACKING_CAMERA_TO_BASE_REPLACEMENT_MARKER);
                fixture["odometry"]["raw_imu_calibration"] =
                    json!(RAW_IMU_CALIBRATION_REPLACEMENT_MARKER);
                serde_json::to_vec(&fixture).expect("preparation navigation fixture")
            })
            .as_slice()
    }

    #[test]
    fn prepares_domain_valid_bit_exact_artifact_and_navigation() {
        let input = serde_json::to_vec(&valid_input()).expect("fixture");
        let prepared = prepare(&input, synthetic_navigation_fixture()).expect("prepare");
        let artifact =
            NanoCalibrationArtifactV1::parse_json(&prepared.artifact_json).expect("artifact");
        let stereo = artifact.rectified_stereo();
        let camera = DepthCameraModel::new(
            stereo.left(),
            stereo.dimensions(),
            DepthToTrackingCamera::identity(),
        );
        let navigation = ShadowNavigationConfigV1::parse_json(&prepared.navigation_json, camera)
            .expect("navigation");
        artifact
            .require_navigation(&navigation)
            .expect("exact binding");

        let artifact_json: Value =
            serde_json::from_slice(&prepared.artifact_json).expect("artifact JSON");
        assert_eq!(
            artifact_json["raw_imu_calibration"]["accel_affine"],
            json!([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
        );
        assert_eq!(
            artifact_json["raw_imu_calibration"]["accel_bias_native_m_per_sec2"],
            json!([0.5, 2.0 / 3.0, 0.75])
        );
        assert_eq!(
            artifact_json["raw_imu_calibration"]["gyro_affine"],
            json!([[2.0, 2.0, 3.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        );
        assert_eq!(
            artifact_json["raw_imu_calibration"]["gyro_bias_native_rad_per_sec"],
            json!([-6.0, 2.0, 3.0])
        );
    }

    #[test]
    fn rejects_retained_tenfold_baseline_discrepancy() {
        let mut input = valid_input();
        input["rectified_stereo"]["corroborating_baseline_m"] = json!(0.0075033944182863015);
        let error = prepare(
            &serde_json::to_vec(&input).expect("fixture"),
            synthetic_navigation_fixture(),
        )
        .err()
        .expect("tenfold discrepancy must fail");
        assert!(matches!(
            error,
            PrepareError::BaselineMismatch {
                relative_discrepancy,
                ..
            } if relative_discrepancy > 0.89
        ));
    }

    #[test]
    fn requires_typed_independence_and_distinct_baseline_sources() {
        let mut missing_claim = valid_input();
        missing_claim["rectified_stereo"]
            .as_object_mut()
            .expect("rectified stereo object")
            .remove("corroborating_baseline_relationship");
        assert!(matches!(
            prepare(
                &serde_json::to_vec(&missing_claim).expect("fixture"),
                synthetic_navigation_fixture()
            ),
            Err(PrepareError::InputJson(_))
        ));

        let mut unsupported_claim = valid_input();
        unsupported_claim["rectified_stereo"]["corroborating_baseline_relationship"] =
            json!("derived_from_live_stereo");
        assert!(matches!(
            prepare(
                &serde_json::to_vec(&unsupported_claim).expect("fixture"),
                synthetic_navigation_fixture()
            ),
            Err(PrepareError::InputJson(_))
        ));

        let mut reused_id = valid_input();
        reused_id["rectified_stereo"]["corroborating_baseline_provenance"]["source_id"] =
            reused_id["rectified_stereo"]["provenance"]["source_id"].clone();
        assert!(matches!(
            prepare(
                &serde_json::to_vec(&reused_id).expect("fixture"),
                synthetic_navigation_fixture()
            ),
            Err(PrepareError::CorroboratingBaselineReusesSourceId { .. })
        ));

        let mut reused_content = valid_input();
        reused_content["rectified_stereo"]["corroborating_baseline_provenance"]["source_sha256_hex"] =
            reused_content["rectified_stereo"]["provenance"]["source_sha256_hex"].clone();
        assert!(matches!(
            prepare(
                &serde_json::to_vec(&reused_content).expect("fixture"),
                synthetic_navigation_fixture()
            ),
            Err(PrepareError::CorroboratingBaselineReusesContent { .. })
        ));
    }

    #[test]
    fn basalt_parameter_units_are_required_and_closed() {
        let mut missing = valid_input();
        missing["basalt_imu_calibration"]
            .as_object_mut()
            .expect("Basalt calibration object")
            .remove("calib_accel_bias_units");
        assert!(matches!(
            prepare(
                &serde_json::to_vec(&missing).expect("missing-unit fixture"),
                synthetic_navigation_fixture()
            ),
            Err(PrepareError::InputJson(_))
        ));

        let mut unsupported_bias = valid_input();
        unsupported_bias["basalt_imu_calibration"]["calib_accel_bias_units"]["bias"] =
            json!("feet_per_second_squared");
        assert!(matches!(
            prepare(
                &serde_json::to_vec(&unsupported_bias).expect("unsupported-unit fixture"),
                synthetic_navigation_fixture()
            ),
            Err(PrepareError::InputJson(_))
        ));

        let mut unsupported_scale = valid_input();
        unsupported_scale["basalt_imu_calibration"]["calib_gyro_bias_units"]["scale"] =
            json!("percent");
        assert!(matches!(
            prepare(
                &serde_json::to_vec(&unsupported_scale).expect("unsupported-scale fixture"),
                synthetic_navigation_fixture()
            ),
            Err(PrepareError::InputJson(_))
        ));
    }

    #[test]
    fn rejects_unknown_fields_missing_transforms_and_unstable_affines() {
        let mut unknown = valid_input();
        unknown["invented_default"] = json!(true);
        assert!(matches!(
            prepare(
                &serde_json::to_vec(&unknown).expect("fixture"),
                synthetic_navigation_fixture()
            ),
            Err(PrepareError::InputJson(_))
        ));

        let mut missing = valid_input();
        missing
            .as_object_mut()
            .expect("object")
            .remove("tracking_camera_to_base");
        assert!(matches!(
            prepare(
                &serde_json::to_vec(&missing).expect("fixture"),
                synthetic_navigation_fixture()
            ),
            Err(PrepareError::InputJson(_))
        ));

        let mut singular = valid_input();
        singular["basalt_imu_calibration"]["calib_accel_bias"][3] = json!(-1.0);
        assert!(matches!(
            prepare(
                &serde_json::to_vec(&singular).expect("fixture"),
                synthetic_navigation_fixture()
            ),
            Err(PrepareError::UnstableBasaltAffine {
                field: "calib_accel_bias",
                ..
            })
        ));
    }

    #[test]
    fn provenance_is_required_and_lowercase_content_addressed() {
        let mut upper = valid_input();
        upper["native_imu_to_base"]["provenance"]["source_sha256_hex"] =
            json!(SHA.to_ascii_uppercase());
        assert!(matches!(
            prepare(
                &serde_json::to_vec(&upper).expect("fixture"),
                synthetic_navigation_fixture()
            ),
            Err(PrepareError::InvalidSourceSha256 {
                field: "native_imu_to_base.provenance"
            })
        ));

        let mut empty = valid_input();
        empty["tracking_camera_to_base"]["provenance"]["source_id"] = json!(" ");
        assert!(matches!(
            prepare(
                &serde_json::to_vec(&empty).expect("fixture"),
                synthetic_navigation_fixture()
            ),
            Err(PrepareError::EmptySourceId {
                field: "tracking_camera_to_base.provenance"
            })
        ));

        let mut combined = valid_input();
        combined["basalt_imu_calibration"]["provenance"]["source_id"] = json!("b".repeat(1_000));
        combined["native_imu_to_base"]["provenance"]["source_id"] = json!("i".repeat(1_000));
        assert!(matches!(
            prepare(
                &serde_json::to_vec(&combined).expect("fixture"),
                synthetic_navigation_fixture()
            ),
            Err(PrepareError::CombinedRawImuSourceIdTooLong { .. })
        ));
    }

    #[test]
    fn rejects_literal_and_json_escaped_input_tokens_in_string_ids() {
        let mut literal = valid_input();
        literal["tracking_camera_to_base"]["provenance"]["source_id"] =
            json!("${UNRESOLVED_MEASUREMENT_SOURCE_ID}");
        let literal_bytes = serde_json::to_vec(&literal).expect("literal-token fixture");
        assert!(matches!(
            prepare(&literal_bytes, synthetic_navigation_fixture()),
            Err(PrepareError::UnresolvedInputTemplateToken { .. })
        ));

        let mut escaped = valid_input();
        escaped["oak_mxid"] = json!("${UNRESOLVED_OAK_MXID}");
        let escaped_json = serde_json::to_string(&escaped)
            .expect("escaped-token fixture")
            .replace("${UNRESOLVED_OAK_MXID}", "\\u0024{UNRESOLVED_OAK_MXID}");
        assert!(!escaped_json.contains("${"));
        assert!(matches!(
            prepare(escaped_json.as_bytes(), synthetic_navigation_fixture()),
            Err(PrepareError::UnresolvedDecodedInputTemplateToken { field: "oak_mxid" })
        ));
    }

    #[test]
    fn navigation_allows_only_exact_markers_at_exact_replacement_leaves() {
        let input = serde_json::to_vec(&valid_input()).expect("input fixture");

        let mut literal: Value =
            serde_json::from_slice(synthetic_navigation_fixture()).expect("navigation fixture");
        literal["plant_model"]["model_id"] = json!("${UNRESOLVED_PLANT_MODEL_ID}");
        assert!(matches!(
            prepare(
                &input,
                &serde_json::to_vec(&literal).expect("literal-token navigation")
            ),
            Err(PrepareError::UnresolvedNavigationTemplateToken { .. })
        ));

        let mut escaped: Value =
            serde_json::from_slice(synthetic_navigation_fixture()).expect("navigation fixture");
        escaped["plant_model"]["model_id"] = json!("${UNRESOLVED_PLANT_MODEL_ID}");
        let escaped_json = serde_json::to_string(&escaped)
            .expect("escaped-token navigation")
            .replace(
                "${UNRESOLVED_PLANT_MODEL_ID}",
                "\\u0024{UNRESOLVED_PLANT_MODEL_ID}",
            );
        assert!(!escaped_json.contains("${UNRESOLVED_PLANT_MODEL_ID}"));
        assert!(matches!(
            prepare(&input, escaped_json.as_bytes()),
            Err(PrepareError::UnresolvedDecodedNavigationTemplateToken)
        ));

        let mut misplaced_exact: Value =
            serde_json::from_slice(synthetic_navigation_fixture()).expect("navigation fixture");
        misplaced_exact["plant_model"]["model_id"] = json!(RAW_IMU_CALIBRATION_REPLACEMENT_MARKER);
        assert!(matches!(
            prepare(
                &input,
                &serde_json::to_vec(&misplaced_exact).expect("misplaced marker navigation")
            ),
            Err(PrepareError::UnresolvedDecodedNavigationTemplateToken)
        ));

        let mut altered_leaf: Value =
            serde_json::from_slice(synthetic_navigation_fixture()).expect("navigation fixture");
        altered_leaf["odometry"]["raw_imu_calibration"] = json!("already-filled");
        assert!(matches!(
            prepare(
                &input,
                &serde_json::to_vec(&altered_leaf).expect("altered marker navigation")
            ),
            Err(PrepareError::NavigationReplacementMarkerMismatch {
                path: "odometry.raw_imu_calibration",
                ..
            })
        ));
    }

    #[test]
    fn navigation_rejects_duplicate_top_level_object_keys() {
        let input = serde_json::to_vec(&valid_input()).expect("input fixture");
        let mut navigation =
            String::from_utf8(synthetic_navigation_fixture().to_vec()).expect("navigation UTF-8");
        navigation.insert_str(1, "\"schema_version\":1,");
        let error = prepare(&input, navigation.as_bytes())
            .err()
            .expect("duplicate top-level key must fail");
        assert!(matches!(
            error,
            PrepareError::NavigationJson(source)
                if source.to_string().contains("duplicate object key \"schema_version\"")
        ));
    }

    #[test]
    fn navigation_rejects_duplicate_nested_object_keys() {
        let input = serde_json::to_vec(&valid_input()).expect("input fixture");
        let mut navigation =
            String::from_utf8(synthetic_navigation_fixture().to_vec()).expect("navigation UTF-8");
        let plant_start = navigation
            .find("\"plant_model\":{")
            .expect("plant model object")
            + "\"plant_model\":{".len();
        navigation.insert_str(plant_start, "\"schema_version\":1,");
        let error = prepare(&input, navigation.as_bytes())
            .err()
            .expect("duplicate nested key must fail");
        assert!(matches!(
            error,
            PrepareError::NavigationJson(source)
                if source.to_string().contains("duplicate object key \"schema_version\"")
        ));
    }

    #[test]
    fn publication_exposes_only_the_complete_synced_directory() {
        let root = TestDirectory::new();
        let output = root.path().join("prepared");
        write_new_output_directory(&output, b"input", b"artifact", b"navigation")
            .expect("publish complete directory");

        assert_eq!(fs::read(output.join(INPUT_FILE)).expect("input"), b"input");
        assert_eq!(
            fs::read(output.join(ARTIFACT_FILE)).expect("artifact"),
            b"artifact"
        );
        assert_eq!(
            fs::read(output.join(NAVIGATION_FILE)).expect("navigation"),
            b"navigation"
        );
        let sibling_names: Vec<_> = fs::read_dir(root.path())
            .expect("read parent")
            .map(|entry| entry.expect("directory entry").file_name())
            .collect();
        assert_eq!(
            sibling_names,
            vec![output.file_name().expect("output name")]
        );
    }

    #[test]
    fn failed_publication_cleans_staging_and_preserves_existing_output() {
        let root = TestDirectory::new();
        let failed_output = root.path().join("failed");
        let failure =
            publish_files_transactionally(&failed_output, &[("missing/child", b"payload")])
                .expect_err("nested path must fail");
        assert!(matches!(
            failure,
            PublicationError::Io {
                phase: PublicationPhase::CreateFile,
                output_published: false,
                ..
            }
        ));
        assert!(!failed_output.exists());
        assert_eq!(
            fs::read_dir(root.path()).expect("read parent").count(),
            0,
            "failed staging directory must be removed"
        );

        let existing_output = root.path().join("existing");
        fs::create_dir(&existing_output).expect("existing output");
        fs::write(existing_output.join("sentinel"), b"retained").expect("sentinel");
        assert!(matches!(
            write_new_output_directory(&existing_output, b"input", b"artifact", b"navigation"),
            Err(PublicationError::OutputAlreadyExists { .. })
        ));
        assert_eq!(
            fs::read(existing_output.join("sentinel")).expect("sentinel"),
            b"retained"
        );
    }

    #[test]
    fn atomic_publish_never_replaces_destination_created_after_prechecks() {
        let root = TestDirectory::new();
        let output = root.path().join("concurrent-output");
        let failure = publish_files_transactionally_with_hook(
            &output,
            &[("complete", b"staged bytes")],
            || fs::create_dir(&output).expect("concurrent empty destination"),
        )
        .expect_err("atomic no-replace must reject concurrent destination");
        assert!(matches!(
            failure,
            PublicationError::OutputAlreadyExists { .. }
        ));
        assert!(output.is_dir());
        assert_eq!(
            fs::read_dir(&output)
                .expect("preserved destination")
                .count(),
            0,
            "the concurrently created empty destination must not be replaced"
        );
        let sibling_names: Vec<_> = fs::read_dir(root.path())
            .expect("read parent")
            .map(|entry| entry.expect("directory entry").file_name())
            .collect();
        assert_eq!(
            sibling_names,
            vec![output.file_name().expect("output name")],
            "failed staging directory must be removed"
        );
    }

    #[test]
    fn qualification_navigation_preparation_template_is_sentinelized_and_shadow_only() {
        let template = include_str!(
            "../../../../configs/nano-wheels-off-qualification-template/navigation-shadow-preparation-v1.json.template"
        );
        assert!(template.contains("${NAV_SHADOW_UNVALIDATED_"));
        assert!(template.contains("${CALIBRATION_PREPARER_REPLACES_TRACKING_CAMERA_TO_BASE}"));
        assert!(template.contains("${CALIBRATION_PREPARER_REPLACES_RAW_IMU_CALIBRATION}"));
        assert!(!template.contains("synthetic-host-shadow-example"));

        let fixture: Value =
            serde_json::from_slice(synthetic_navigation_fixture()).expect("synthetic fixture JSON");
        let mut rendered = template_with_unquoted_tokens_replaced_by_null(template);
        assert_eq!(
            rendered["plant_model"]["model_id"],
            "qualification-shadow-only-synthetic-unvalidated-v2"
        );
        assert_eq!(rendered["plant_model"]["model_version"], 2);
        assert_eq!(rendered["plant_model"]["sample_period_s"], 0.05);
        assert_eq!(
            rendered["plant_model"]["evidence"]["kind"],
            "synthetic_fixture"
        );
        fill_navigation_sentinels_from_fixture(&mut rendered, &fixture);
        let rendered_bytes = serde_json::to_vec(&rendered).expect("rendered template JSON");
        assert_eq!(
            String::from_utf8_lossy(&rendered_bytes)
                .matches("${")
                .count(),
            2,
            "only the two calibration-preparer replacement markers may remain"
        );

        let input = serde_json::to_vec(&valid_input()).expect("preparation fixture");
        prepare(&input, &rendered_bytes)
            .expect("fully rendered deployable template must pass production parsers");
    }
}
