//! Canonical, content-addressed calibration contract for one Kiko assembly.
//!
//! The retained artifact is parsed once before any motion authority is
//! acquired. It binds one exact OAK identity to one rectified stereo model,
//! raw OAK-IMU calibration, tracking-camera-to-base transform, and the three
//! calibration approval identifiers consumed by production actuation.
//! Structural parsing does not prove that a physical calibration procedure was
//! performed correctly; runtime admission requires bit-exact agreement with
//! the same-owner OAK observations and the parsed navigation configuration.

use std::fmt;

use serde::Deserialize;

use super::{
    NavigationActuationConfigV1, RawImuCalibration, RawImuCalibrationDto, RawImuCalibrationError,
    ShadowNavigationConfigV1, TrackingCameraToBase,
};
use crate::dataset::Calibration;
use crate::{
    FrameDimensions, FrameDimensionsError, IntrinsicsError, PinholeIntrinsics, Pose, PoseError,
    RectifiedStereo, RectifiedStereoCompatibilityError, RectifiedStereoError, StereoBaselineError,
    StereoCalibration,
};

pub const NANO_CALIBRATION_ARTIFACT_V1: u32 = 1;
pub const MAX_NANO_CALIBRATION_ARTIFACT_JSON_BYTES: usize = 64 * 1_024;

const MIN_OAK_MXID_BYTES: usize = 8;
const MAX_OAK_MXID_BYTES: usize = 64;
const MAX_CALIBRATION_ID_BYTES: usize = 128;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoCalibrationId(Box<str>);

impl NanoCalibrationId {
    fn parse(
        field: NanoCalibrationIdField,
        value: String,
    ) -> Result<Self, NanoCalibrationArtifactParseError> {
        if value.is_empty()
            || value.len() > MAX_CALIBRATION_ID_BYTES
            || value.bytes().all(|byte| byte == b'0')
            || !value.bytes().all(|byte| {
                byte.is_ascii_alphanumeric()
                    || matches!(byte, b'-' | b'_' | b'.' | b':' | b'/' | b'@' | b'+')
            })
        {
            return Err(NanoCalibrationArtifactParseError::InvalidCalibrationId {
                field,
                actual_bytes: value.len(),
                maximum_bytes: MAX_CALIBRATION_ID_BYTES,
            });
        }
        Ok(Self(value.into_boxed_str()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoCalibrationOakMxid(Box<str>);

impl NanoCalibrationOakMxid {
    fn parse(value: String) -> Result<Self, NanoCalibrationArtifactParseError> {
        if value.len() < MIN_OAK_MXID_BYTES
            || value.len() > MAX_OAK_MXID_BYTES
            || value.bytes().all(|byte| byte == b'0')
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || matches!(byte, b'A'..=b'F'))
        {
            return Err(NanoCalibrationArtifactParseError::InvalidOakMxid {
                actual_bytes: value.len(),
                minimum_bytes: MIN_OAK_MXID_BYTES,
                maximum_bytes: MAX_OAK_MXID_BYTES,
            });
        }
        Ok(Self(value.into_boxed_str()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug)]
pub struct NanoCalibrationArtifactV1 {
    oak_mxid: NanoCalibrationOakMxid,
    imu_calibration_id: NanoCalibrationId,
    stereo_calibration_id: NanoCalibrationId,
    tracking_camera_to_base_calibration_id: NanoCalibrationId,
    rectified_stereo: RectifiedStereo,
    raw_imu_calibration: RawImuCalibration,
    tracking_camera_to_base: TrackingCameraToBase,
}

impl NanoCalibrationArtifactV1 {
    pub fn parse_json(json: &[u8]) -> Result<Self, NanoCalibrationArtifactParseError> {
        if json.len() > MAX_NANO_CALIBRATION_ARTIFACT_JSON_BYTES {
            return Err(NanoCalibrationArtifactParseError::InputTooLarge {
                actual_bytes: json.len(),
                maximum_bytes: MAX_NANO_CALIBRATION_ARTIFACT_JSON_BYTES,
            });
        }
        let mut deserializer = serde_json::Deserializer::from_slice(json);
        let dto = NanoCalibrationArtifactV1Dto::deserialize(&mut deserializer)
            .map_err(NanoCalibrationArtifactParseError::JsonDecode)?;
        deserializer
            .end()
            .map_err(NanoCalibrationArtifactParseError::JsonTrailingData)?;
        if dto.schema_version != NANO_CALIBRATION_ARTIFACT_V1 {
            return Err(NanoCalibrationArtifactParseError::UnsupportedSchema {
                actual: dto.schema_version,
                supported: NANO_CALIBRATION_ARTIFACT_V1,
            });
        }

        let oak_mxid = NanoCalibrationOakMxid::parse(dto.oak_mxid)?;
        let imu_calibration_id =
            NanoCalibrationId::parse(NanoCalibrationIdField::Imu, dto.imu_calibration_id)?;
        let stereo_calibration_id =
            NanoCalibrationId::parse(NanoCalibrationIdField::Stereo, dto.stereo_calibration_id)?;
        let tracking_camera_to_base_calibration_id = NanoCalibrationId::parse(
            NanoCalibrationIdField::TrackingCameraToBase,
            dto.tracking_camera_to_base_calibration_id,
        )?;

        let rectified_stereo = parse_rectified_stereo(dto.rectified_stereo)?;
        let raw_imu_calibration = RawImuCalibration::parse(dto.raw_imu_calibration.into_domain())
            .map_err(NanoCalibrationArtifactParseError::RawImuCalibration)?;
        if raw_imu_calibration.provenance().content_id() != imu_calibration_id.as_str() {
            return Err(NanoCalibrationArtifactParseError::RawImuContentIdMismatch);
        }
        let tracking_camera_to_base = TrackingCameraToBase::new(
            Pose::try_from_rt(
                dto.tracking_camera_to_base.rotation,
                dto.tracking_camera_to_base.translation_m,
            )
            .map_err(NanoCalibrationArtifactParseError::TrackingCameraToBase)?,
        );

        Ok(Self {
            oak_mxid,
            imu_calibration_id,
            stereo_calibration_id,
            tracking_camera_to_base_calibration_id,
            rectified_stereo,
            raw_imu_calibration,
            tracking_camera_to_base,
        })
    }

    pub const fn oak_mxid(&self) -> &NanoCalibrationOakMxid {
        &self.oak_mxid
    }

    pub const fn imu_calibration_id(&self) -> &NanoCalibrationId {
        &self.imu_calibration_id
    }

    pub const fn stereo_calibration_id(&self) -> &NanoCalibrationId {
        &self.stereo_calibration_id
    }

    pub const fn tracking_camera_to_base_calibration_id(&self) -> &NanoCalibrationId {
        &self.tracking_camera_to_base_calibration_id
    }

    pub const fn rectified_stereo(&self) -> &RectifiedStereo {
        &self.rectified_stereo
    }

    pub const fn raw_imu_calibration(&self) -> &RawImuCalibration {
        &self.raw_imu_calibration
    }

    pub const fn tracking_camera_to_base(&self) -> TrackingCameraToBase {
        self.tracking_camera_to_base
    }

    pub fn require_manifest_oak_mxid(
        &self,
        manifest_mxid: &str,
    ) -> Result<(), NanoCalibrationBindingError> {
        if self.oak_mxid.as_str() != manifest_mxid {
            return Err(NanoCalibrationBindingError::ManifestOakMxidMismatch);
        }
        Ok(())
    }

    pub fn require_connected_oak_mxid(
        &self,
        connected_mxid: &str,
    ) -> Result<(), NanoCalibrationBindingError> {
        if self.oak_mxid.as_str() != connected_mxid {
            return Err(NanoCalibrationBindingError::ConnectedOakMxidMismatch);
        }
        Ok(())
    }

    pub fn require_observed_stereo(
        &self,
        observed: &Calibration,
    ) -> Result<(), NanoCalibrationBindingError> {
        let observed = RectifiedStereo::from_calibration(observed)
            .map_err(NanoCalibrationBindingError::ObservedStereoInvalid)?;
        if !self.rectified_stereo.exactly_matches(&observed) {
            return Err(NanoCalibrationBindingError::ObservedStereoMismatch);
        }
        Ok(())
    }

    pub fn require_navigation(
        &self,
        navigation: &ShadowNavigationConfigV1,
    ) -> Result<(), NanoCalibrationBindingError> {
        self.require_navigation_parts(
            navigation.odometry().raw_imu_calibration(),
            navigation.tracking_camera_to_base(),
        )
    }

    fn require_navigation_parts(
        &self,
        raw_imu_calibration: &RawImuCalibration,
        tracking_camera_to_base: TrackingCameraToBase,
    ) -> Result<(), NanoCalibrationBindingError> {
        if !self
            .raw_imu_calibration
            .exactly_matches(raw_imu_calibration)
        {
            return Err(NanoCalibrationBindingError::NavigationRawImuMismatch);
        }
        if !tracking_camera_to_base_exactly_matches(
            self.tracking_camera_to_base,
            tracking_camera_to_base,
        ) {
            return Err(NanoCalibrationBindingError::NavigationTrackingCameraToBaseMismatch);
        }
        Ok(())
    }

    pub fn require_actuation_approval(
        &self,
        actuation: &NavigationActuationConfigV1,
    ) -> Result<(), NanoCalibrationBindingError> {
        let approval = actuation.approval();
        self.require_actuation_approval_ids(
            approval.imu_calibration_id(),
            approval.stereo_calibration_id(),
            approval.tracking_camera_to_base_calibration_id(),
        )
    }

    fn require_actuation_approval_ids(
        &self,
        imu_calibration_id: &str,
        stereo_calibration_id: &str,
        tracking_camera_to_base_calibration_id: &str,
    ) -> Result<(), NanoCalibrationBindingError> {
        if self.imu_calibration_id.as_str() != imu_calibration_id {
            return Err(NanoCalibrationBindingError::ActuationImuCalibrationIdMismatch);
        }
        if self.stereo_calibration_id.as_str() != stereo_calibration_id {
            return Err(NanoCalibrationBindingError::ActuationStereoCalibrationIdMismatch);
        }
        if self.tracking_camera_to_base_calibration_id.as_str()
            != tracking_camera_to_base_calibration_id
        {
            return Err(
                NanoCalibrationBindingError::ActuationTrackingCameraToBaseCalibrationIdMismatch,
            );
        }
        Ok(())
    }
}

fn parse_rectified_stereo(
    dto: RectifiedStereoDto,
) -> Result<RectifiedStereo, NanoCalibrationArtifactParseError> {
    if !dto.rectified {
        return Err(NanoCalibrationArtifactParseError::RectifiedStereoRequired);
    }
    let left_dimensions =
        FrameDimensions::try_new(dto.left.width_px, dto.left.height_px).map_err(|source| {
            NanoCalibrationArtifactParseError::StereoDimensions {
                side: NanoCalibrationStereoSide::Left,
                source,
            }
        })?;
    let right_dimensions = FrameDimensions::try_new(dto.right.width_px, dto.right.height_px)
        .map_err(
            |source| NanoCalibrationArtifactParseError::StereoDimensions {
                side: NanoCalibrationStereoSide::Right,
                source,
            },
        )?;
    if left_dimensions != right_dimensions {
        return Err(NanoCalibrationArtifactParseError::StereoDimensionMismatch {
            left: left_dimensions,
            right: right_dimensions,
        });
    }
    let left = PinholeIntrinsics::try_new(
        dto.left.fx_px,
        dto.left.fy_px,
        dto.left.cx_px,
        dto.left.cy_px,
    )
    .map_err(
        |source| NanoCalibrationArtifactParseError::StereoIntrinsics {
            side: NanoCalibrationStereoSide::Left,
            source,
        },
    )?;
    let right = PinholeIntrinsics::try_new(
        dto.right.fx_px,
        dto.right.fy_px,
        dto.right.cx_px,
        dto.right.cy_px,
    )
    .map_err(
        |source| NanoCalibrationArtifactParseError::StereoIntrinsics {
            side: NanoCalibrationStereoSide::Right,
            source,
        },
    )?;
    let stereo =
        StereoCalibration::try_new(left, right, left_dimensions, dto.baseline_m, dto.rectified)
            .map_err(NanoCalibrationArtifactParseError::StereoBaseline)?;
    RectifiedStereo::from_stereo_calibration(&stereo)
        .map_err(NanoCalibrationArtifactParseError::RectifiedStereoCompatibility)
}

fn tracking_camera_to_base_exactly_matches(
    left: TrackingCameraToBase,
    right: TrackingCameraToBase,
) -> bool {
    let left = left.pose();
    let right = right.pose();
    float32_matrix_exact(left.rotation(), right.rotation())
        && float32_array_exact(left.translation(), right.translation())
}

fn float32_matrix_exact(left: [[f32; 3]; 3], right: [[f32; 3]; 3]) -> bool {
    left.iter()
        .flatten()
        .zip(right.iter().flatten())
        .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn float32_array_exact(left: [f32; 3], right: [f32; 3]) -> bool {
    left.into_iter()
        .zip(right)
        .all(|(left, right)| left.to_bits() == right.to_bits())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoCalibrationIdField {
    Imu,
    Stereo,
    TrackingCameraToBase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoCalibrationStereoSide {
    Left,
    Right,
}

#[derive(Debug)]
pub enum NanoCalibrationArtifactParseError {
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
    InvalidOakMxid {
        actual_bytes: usize,
        minimum_bytes: usize,
        maximum_bytes: usize,
    },
    InvalidCalibrationId {
        field: NanoCalibrationIdField,
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    RectifiedStereoRequired,
    StereoDimensions {
        side: NanoCalibrationStereoSide,
        source: FrameDimensionsError,
    },
    StereoDimensionMismatch {
        left: FrameDimensions,
        right: FrameDimensions,
    },
    StereoIntrinsics {
        side: NanoCalibrationStereoSide,
        source: IntrinsicsError,
    },
    StereoBaseline(StereoBaselineError),
    RectifiedStereoCompatibility(RectifiedStereoCompatibilityError),
    RawImuCalibration(RawImuCalibrationError),
    RawImuContentIdMismatch,
    TrackingCameraToBase(PoseError),
}

impl fmt::Display for NanoCalibrationArtifactParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid Nano calibration artifact: {self:?}")
    }
}

impl std::error::Error for NanoCalibrationArtifactParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::JsonDecode(source) | Self::JsonTrailingData(source) => Some(source),
            Self::StereoDimensions { source, .. } => Some(source),
            Self::StereoIntrinsics { source, .. } => Some(source),
            Self::StereoBaseline(source) => Some(source),
            Self::RectifiedStereoCompatibility(source) => Some(source),
            Self::RawImuCalibration(source) => Some(source),
            Self::TrackingCameraToBase(source) => Some(source),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum NanoCalibrationBindingError {
    ManifestOakMxidMismatch,
    ConnectedOakMxidMismatch,
    ObservedStereoInvalid(RectifiedStereoError),
    ObservedStereoMismatch,
    NavigationRawImuMismatch,
    NavigationTrackingCameraToBaseMismatch,
    ActuationImuCalibrationIdMismatch,
    ActuationStereoCalibrationIdMismatch,
    ActuationTrackingCameraToBaseCalibrationIdMismatch,
}

impl fmt::Display for NanoCalibrationBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Nano calibration binding rejected input: {self:?}"
        )
    }
}

impl std::error::Error for NanoCalibrationBindingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ObservedStereoInvalid(source) => Some(source),
            _ => None,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoCalibrationArtifactV1Dto {
    schema_version: u32,
    oak_mxid: String,
    imu_calibration_id: String,
    stereo_calibration_id: String,
    tracking_camera_to_base_calibration_id: String,
    rectified_stereo: RectifiedStereoDto,
    raw_imu_calibration: RawImuCalibrationJsonDto,
    tracking_camera_to_base: TrackingCameraToBaseDto,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RectifiedStereoDto {
    rectified: bool,
    left: CameraIntrinsicsDto,
    right: CameraIntrinsicsDto,
    baseline_m: f32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CameraIntrinsicsDto {
    fx_px: f32,
    fy_px: f32,
    cx_px: f32,
    cy_px: f32,
    width_px: u32,
    height_px: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawImuCalibrationJsonDto {
    format_version: u32,
    source_id: String,
    content_id: String,
    gyro_affine: [[f64; 3]; 3],
    gyro_bias_native_rad_per_sec: [f64; 3],
    accel_affine: [[f64; 3]; 3],
    accel_bias_native_m_per_sec2: [f64; 3],
    native_imu_to_base_rotation: [[f64; 3]; 3],
}

impl RawImuCalibrationJsonDto {
    fn into_domain(self) -> RawImuCalibrationDto {
        RawImuCalibrationDto {
            format_version: self.format_version,
            source_id: self.source_id,
            content_id: self.content_id,
            gyro_affine: self.gyro_affine,
            gyro_bias_native_rad_per_sec: self.gyro_bias_native_rad_per_sec,
            accel_affine: self.accel_affine,
            accel_bias_native_m_per_sec2: self.accel_bias_native_m_per_sec2,
            native_imu_to_base_rotation: self.native_imu_to_base_rotation,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TrackingCameraToBaseDto {
    rotation: [[f32; 3]; 3],
    translation_m: [f32; 3],
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;

    const IDENTITY_F32: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    const IDENTITY_F64: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    fn valid_value() -> Value {
        json!({
            "schema_version": 1,
            "oak_mxid": "19443010F1B43A2E00",
            "imu_calibration_id": "imu-v1",
            "stereo_calibration_id": "stereo-v1",
            "tracking_camera_to_base_calibration_id": "camera-base-v1",
            "rectified_stereo": {
                "rectified": true,
                "left": {
                    "fx_px": 400.0,
                    "fy_px": 400.0,
                    "cx_px": 320.0,
                    "cy_px": 200.0,
                    "width_px": 640,
                    "height_px": 400
                },
                "right": {
                    "fx_px": 400.0,
                    "fy_px": 400.0,
                    "cx_px": 320.0,
                    "cy_px": 200.0,
                    "width_px": 640,
                    "height_px": 400
                },
                "baseline_m": 0.075
            },
            "raw_imu_calibration": {
                "format_version": 1,
                "source_id": "fixture",
                "content_id": "imu-v1",
                "gyro_affine": IDENTITY_F64,
                "gyro_bias_native_rad_per_sec": [0.0, 0.0, 0.0],
                "accel_affine": IDENTITY_F64,
                "accel_bias_native_m_per_sec2": [0.0, 0.0, 0.0],
                "native_imu_to_base_rotation": IDENTITY_F64
            },
            "tracking_camera_to_base": {
                "rotation": IDENTITY_F32,
                "translation_m": [0.20, 0.0, -0.25]
            }
        })
    }

    fn parse(
        value: &Value,
    ) -> Result<NanoCalibrationArtifactV1, NanoCalibrationArtifactParseError> {
        NanoCalibrationArtifactV1::parse_json(
            &serde_json::to_vec(value).expect("fixture serialization"),
        )
    }

    fn observed_calibration() -> Calibration {
        Calibration {
            left: crate::dataset::CameraIntrinsics {
                fx: 400.0,
                fy: 400.0,
                cx: 320.0,
                cy: 200.0,
                width: 640,
                height: 400,
            },
            right: crate::dataset::CameraIntrinsics {
                fx: 400.0,
                fy: 400.0,
                cx: 320.0,
                cy: 200.0,
                width: 640,
                height: 400,
            },
            baseline_m: 0.075,
            rectified: true,
            oak_eeprom: None,
        }
    }

    #[test]
    fn parses_complete_contract_and_matches_exact_same_owner_stereo() {
        let artifact = parse(&valid_value()).expect("valid calibration artifact");
        assert_eq!(artifact.oak_mxid().as_str(), "19443010F1B43A2E00");
        assert_eq!(artifact.imu_calibration_id().as_str(), "imu-v1");
        artifact
            .require_manifest_oak_mxid("19443010F1B43A2E00")
            .expect("manifest MXID");
        artifact
            .require_connected_oak_mxid("19443010F1B43A2E00")
            .expect("connected MXID");
        artifact
            .require_observed_stereo(&observed_calibration())
            .expect("exact live stereo");
    }

    #[test]
    fn rejects_noncanonical_mxid_and_cross_unbound_imu_id() {
        let mut lowercase = valid_value();
        lowercase["oak_mxid"] = json!("19443010f1b43a2e00");
        assert!(matches!(
            parse(&lowercase),
            Err(NanoCalibrationArtifactParseError::InvalidOakMxid { .. })
        ));

        let mut unbound = valid_value();
        unbound["raw_imu_calibration"]["content_id"] = json!("another-imu");
        assert!(matches!(
            parse(&unbound),
            Err(NanoCalibrationArtifactParseError::RawImuContentIdMismatch)
        ));
    }

    #[test]
    fn stereo_comparison_is_bit_exact_not_approximately_equal() {
        let artifact = parse(&valid_value()).expect("valid calibration artifact");
        let mut changed = observed_calibration();
        changed.left.fx = f32::from_bits(changed.left.fx.to_bits() + 1);
        changed.right.fx = changed.left.fx;
        assert!(matches!(
            artifact.require_observed_stereo(&changed),
            Err(NanoCalibrationBindingError::ObservedStereoMismatch)
        ));
    }

    #[test]
    fn transform_and_raw_imu_comparisons_are_bit_exact() {
        let artifact = parse(&valid_value()).expect("valid calibration artifact");
        let mut raw = valid_value();
        raw["raw_imu_calibration"]["gyro_bias_native_rad_per_sec"][0] = json!(f64::from_bits(1));
        let changed = parse(&raw).expect("adjacent finite raw calibration");
        assert!(
            !artifact
                .raw_imu_calibration()
                .exactly_matches(changed.raw_imu_calibration())
        );

        let left = artifact.tracking_camera_to_base();
        let adjacent = f32::from_bits(left.pose().translation()[0].to_bits() + 1);
        let right = TrackingCameraToBase::new(
            Pose::try_from_rt(IDENTITY_F32, [adjacent, 0.0, -0.25]).expect("adjacent pose"),
        );
        assert!(!tracking_camera_to_base_exactly_matches(left, right));

        assert!(matches!(
            artifact.require_navigation_parts(changed.raw_imu_calibration(), left),
            Err(NanoCalibrationBindingError::NavigationRawImuMismatch)
        ));
        assert!(matches!(
            artifact.require_navigation_parts(artifact.raw_imu_calibration(), right),
            Err(NanoCalibrationBindingError::NavigationTrackingCameraToBaseMismatch)
        ));
    }

    #[test]
    fn production_approval_ids_are_cross_bound_individually() {
        let artifact = parse(&valid_value()).expect("valid calibration artifact");
        artifact
            .require_actuation_approval_ids("imu-v1", "stereo-v1", "camera-base-v1")
            .expect("exact approval IDs");
        assert!(matches!(
            artifact.require_actuation_approval_ids("other", "stereo-v1", "camera-base-v1"),
            Err(NanoCalibrationBindingError::ActuationImuCalibrationIdMismatch)
        ));
        assert!(matches!(
            artifact.require_actuation_approval_ids("imu-v1", "other", "camera-base-v1"),
            Err(NanoCalibrationBindingError::ActuationStereoCalibrationIdMismatch)
        ));
        assert!(matches!(
            artifact.require_actuation_approval_ids("imu-v1", "stereo-v1", "other"),
            Err(NanoCalibrationBindingError::ActuationTrackingCameraToBaseCalibrationIdMismatch)
        ));
    }
}
