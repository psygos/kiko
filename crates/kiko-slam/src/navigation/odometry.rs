//! Correction-safe, encoderless planar visual-inertial odometry.
//!
//! Visual increments are the only source of translational motion. Calibrated
//! raw IMU gyroscope samples extend yaw for a short, bounded prediction window;
//! acceleration calibration is retained and applied for observability, but
//! acceleration is deliberately not integrated into translation. Global SLAM
//! corrections update `map <- odom` and never jump the continuous odom pose.

use std::collections::VecDeque;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::sync::Arc;
use std::time::Duration;

use crate::dense::occupancy::WorldToOccupancy;
use crate::{
    DeviceSessionId, DeviceTimestamp, HostMonotonicTimestamp, ImuReport, InertialOrderOutcome,
    InertialOrderTracker, InertialOrderingError, InertialValueError, MapLocalization, MapSnapshot,
    OakImuAcceleration, OakImuAngularVelocity, Pose64, Pose64Error, SensorAccuracy, Timestamp,
    VisualFrameStamp, VisualIncrement, VisualIncrementBasis,
};

use super::{
    BaseFrame, BaseToOdom, MapFrame, MapToOdom, OdomToMap, PlanarTransform, PlanarTransformError,
    TrackingCameraToBase,
};

const ROTATION_TOLERANCE: f64 = 1.0e-6;
const MIN_AFFINE_DETERMINANT: f64 = 1.0e-12;
const NANOSECONDS_PER_SECOND: f64 = 1_000_000_000.0;
const RAW_IMU_CALIBRATION_FORMAT_V1: u32 = 1;
const MAX_IMU_CALIBRATION_SOURCE_ID_BYTES: usize = 2_048;
const MAX_IMU_CALIBRATION_CONTENT_ID_BYTES: usize = 256;
const MAX_CONFIG_GYRO_HISTORY_CAPACITY: usize = 65_536;
const MAX_CONFIG_POSE_HISTORY_CAPACITY: usize = 16_384;
const CONFIG_EULER_SINGULARITY_MARGIN_RAD: f64 = 0.1;

/// Explicit calibration document at an untrusted configuration boundary.
///
/// Corrected native-frame vectors are `affine * (raw - bias)`, then the proper
/// `native_imu_to_base_rotation` is applied. Identity values are permitted only
/// when supplied explicitly; this type has no default or implicit calibration.
/// `source_id` and `content_id` are bounded, caller-asserted opaque identifiers;
/// parsing does not verify that `content_id` hashes or otherwise authenticates
/// the calibration values.
#[derive(Clone, Debug)]
pub struct RawImuCalibrationDto {
    pub format_version: u32,
    pub source_id: String,
    pub content_id: String,
    pub gyro_affine: [[f64; 3]; 3],
    pub gyro_bias_native_rad_per_sec: [f64; 3],
    pub accel_affine: [[f64; 3]; 3],
    pub accel_bias_native_m_per_sec2: [f64; 3],
    pub native_imu_to_base_rotation: [[f64; 3]; 3],
}

#[derive(Clone, Debug)]
pub struct RawImuCalibration {
    provenance: ImuCalibrationProvenance,
    gyro_affine: [[f64; 3]; 3],
    gyro_bias_native_rad_per_sec: [f64; 3],
    accel_affine: [[f64; 3]; 3],
    accel_bias_native_m_per_sec2: [f64; 3],
    native_imu_to_base_rotation: [[f64; 3]; 3],
}

/// Bounded caller-asserted identifiers for the calibration used by odometry.
///
/// These values support diagnostics and record/replay matching. They are not a
/// signature or independently verified content identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImuCalibrationProvenance {
    format_version: NonZeroU32,
    source_id: Arc<str>,
    content_id: Arc<str>,
}

impl ImuCalibrationProvenance {
    pub fn format_version(&self) -> u32 {
        self.format_version.get()
    }

    pub fn source_id(&self) -> &str {
        &self.source_id
    }

    pub fn content_id(&self) -> &str {
        &self.content_id
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BaseAngularVelocity {
    rad_per_sec: [f64; 3],
}

impl BaseAngularVelocity {
    pub fn as_array(self) -> [f64; 3] {
        self.rad_per_sec
    }

    pub fn yaw_rate_rad_per_sec(self) -> f64 {
        self.rad_per_sec[2]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BaseAcceleration {
    metres_per_sec2: [f64; 3],
}

impl BaseAcceleration {
    pub fn as_array(self) -> [f64; 3] {
        self.metres_per_sec2
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibrationMatrix {
    GyroscopeAffine,
    AccelerometerAffine,
    NativeImuToBaseRotation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibrationVector {
    GyroscopeBias,
    AccelerometerBias,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RawImuCalibrationError {
    ZeroFormatVersion,
    UnsupportedFormatVersion {
        actual: u32,
        supported: u32,
    },
    EmptySourceId,
    EmptyContentId,
    SourceIdTooLong {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    ContentIdTooLong {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    NonFiniteMatrix {
        matrix: CalibrationMatrix,
        row: usize,
        column: usize,
        value: f64,
    },
    NonFiniteVector {
        vector: CalibrationVector,
        axis: usize,
        value: f64,
    },
    SingularAffine {
        matrix: CalibrationMatrix,
        determinant: f64,
    },
    NonOrthonormalRotation {
        max_error: f64,
    },
    ImproperRotation {
        determinant: f64,
    },
    CalibratedOutputNonFinite {
        quantity: CalibratedQuantity,
        axis: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibratedQuantity {
    AngularVelocity,
    Acceleration,
}

impl std::fmt::Display for RawImuCalibrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid raw-IMU calibration: {self:?}")
    }
}

impl std::error::Error for RawImuCalibrationError {}

impl RawImuCalibration {
    pub const FORMAT_VERSION: u32 = RAW_IMU_CALIBRATION_FORMAT_V1;
    pub const MAX_SOURCE_ID_BYTES: usize = MAX_IMU_CALIBRATION_SOURCE_ID_BYTES;
    pub const MAX_CONTENT_ID_BYTES: usize = MAX_IMU_CALIBRATION_CONTENT_ID_BYTES;

    pub fn parse(dto: RawImuCalibrationDto) -> Result<Self, RawImuCalibrationError> {
        let format_version =
            NonZeroU32::new(dto.format_version).ok_or(RawImuCalibrationError::ZeroFormatVersion)?;
        if dto.format_version != RAW_IMU_CALIBRATION_FORMAT_V1 {
            return Err(RawImuCalibrationError::UnsupportedFormatVersion {
                actual: dto.format_version,
                supported: RAW_IMU_CALIBRATION_FORMAT_V1,
            });
        }
        if dto.source_id.trim().is_empty() {
            return Err(RawImuCalibrationError::EmptySourceId);
        }
        if dto.content_id.trim().is_empty() {
            return Err(RawImuCalibrationError::EmptyContentId);
        }
        if dto.source_id.len() > MAX_IMU_CALIBRATION_SOURCE_ID_BYTES {
            return Err(RawImuCalibrationError::SourceIdTooLong {
                actual_bytes: dto.source_id.len(),
                maximum_bytes: MAX_IMU_CALIBRATION_SOURCE_ID_BYTES,
            });
        }
        if dto.content_id.len() > MAX_IMU_CALIBRATION_CONTENT_ID_BYTES {
            return Err(RawImuCalibrationError::ContentIdTooLong {
                actual_bytes: dto.content_id.len(),
                maximum_bytes: MAX_IMU_CALIBRATION_CONTENT_ID_BYTES,
            });
        }
        validate_matrix(CalibrationMatrix::GyroscopeAffine, dto.gyro_affine)?;
        validate_matrix(CalibrationMatrix::AccelerometerAffine, dto.accel_affine)?;
        validate_vector(
            CalibrationVector::GyroscopeBias,
            dto.gyro_bias_native_rad_per_sec,
        )?;
        validate_vector(
            CalibrationVector::AccelerometerBias,
            dto.accel_bias_native_m_per_sec2,
        )?;
        validate_affine(CalibrationMatrix::GyroscopeAffine, dto.gyro_affine)?;
        validate_affine(CalibrationMatrix::AccelerometerAffine, dto.accel_affine)?;
        validate_proper_rotation(dto.native_imu_to_base_rotation)?;

        Ok(Self {
            provenance: ImuCalibrationProvenance {
                format_version,
                source_id: Arc::from(dto.source_id),
                content_id: Arc::from(dto.content_id),
            },
            gyro_affine: dto.gyro_affine,
            gyro_bias_native_rad_per_sec: dto.gyro_bias_native_rad_per_sec,
            accel_affine: dto.accel_affine,
            accel_bias_native_m_per_sec2: dto.accel_bias_native_m_per_sec2,
            native_imu_to_base_rotation: dto.native_imu_to_base_rotation,
        })
    }

    pub fn provenance(&self) -> &ImuCalibrationProvenance {
        &self.provenance
    }

    pub fn calibrate_angular_velocity(
        &self,
        raw: OakImuAngularVelocity,
    ) -> Result<BaseAngularVelocity, RawImuCalibrationError> {
        let native = affine_bias_correct(
            self.gyro_affine,
            self.gyro_bias_native_rad_per_sec,
            raw.as_array(),
        );
        let base = matrix_vector(self.native_imu_to_base_rotation, native);
        validate_calibrated(CalibratedQuantity::AngularVelocity, base)?;
        Ok(BaseAngularVelocity { rad_per_sec: base })
    }

    pub fn calibrate_acceleration(
        &self,
        raw: OakImuAcceleration,
    ) -> Result<BaseAcceleration, RawImuCalibrationError> {
        let native = affine_bias_correct(
            self.accel_affine,
            self.accel_bias_native_m_per_sec2,
            raw.as_array(),
        );
        let base = matrix_vector(self.native_imu_to_base_rotation, native);
        validate_calibrated(CalibratedQuantity::Acceleration, base)?;
        Ok(BaseAcceleration {
            metres_per_sec2: base,
        })
    }
}

fn validate_matrix(
    matrix_name: CalibrationMatrix,
    matrix: [[f64; 3]; 3],
) -> Result<(), RawImuCalibrationError> {
    for (row, values) in matrix.iter().enumerate() {
        for (column, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(RawImuCalibrationError::NonFiniteMatrix {
                    matrix: matrix_name,
                    row,
                    column,
                    value,
                });
            }
        }
    }
    Ok(())
}

fn validate_vector(
    vector_name: CalibrationVector,
    vector: [f64; 3],
) -> Result<(), RawImuCalibrationError> {
    for (axis, value) in vector.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(RawImuCalibrationError::NonFiniteVector {
                vector: vector_name,
                axis,
                value,
            });
        }
    }
    Ok(())
}

fn validate_affine(
    matrix_name: CalibrationMatrix,
    matrix: [[f64; 3]; 3],
) -> Result<(), RawImuCalibrationError> {
    let determinant = determinant(matrix);
    if !determinant.is_finite() || determinant.abs() < MIN_AFFINE_DETERMINANT {
        return Err(RawImuCalibrationError::SingularAffine {
            matrix: matrix_name,
            determinant,
        });
    }
    Ok(())
}

fn validate_proper_rotation(rotation: [[f64; 3]; 3]) -> Result<(), RawImuCalibrationError> {
    validate_matrix(CalibrationMatrix::NativeImuToBaseRotation, rotation)?;
    let mut max_error = 0.0_f64;
    for row in 0..3 {
        for column in 0..3 {
            let dot = (0..3)
                .map(|index| rotation[index][row] * rotation[index][column])
                .sum::<f64>();
            let expected = if row == column { 1.0 } else { 0.0 };
            max_error = max_error.max((dot - expected).abs());
        }
    }
    if max_error > ROTATION_TOLERANCE {
        return Err(RawImuCalibrationError::NonOrthonormalRotation { max_error });
    }
    let determinant = determinant(rotation);
    if !determinant.is_finite() || (determinant - 1.0).abs() > ROTATION_TOLERANCE {
        return Err(RawImuCalibrationError::ImproperRotation { determinant });
    }
    Ok(())
}

fn determinant(matrix: [[f64; 3]; 3]) -> f64 {
    matrix[0][0]
        .mul_add(
            matrix[1][1].mul_add(matrix[2][2], -matrix[1][2] * matrix[2][1]),
            -matrix[0][1] * matrix[1][0].mul_add(matrix[2][2], -matrix[1][2] * matrix[2][0]),
        )
        .mul_add(
            1.0,
            matrix[0][2] * matrix[1][0].mul_add(matrix[2][1], -matrix[1][1] * matrix[2][0]),
        )
}

fn affine_bias_correct(affine: [[f64; 3]; 3], bias: [f64; 3], raw: [f64; 3]) -> [f64; 3] {
    matrix_vector(
        affine,
        [raw[0] - bias[0], raw[1] - bias[1], raw[2] - bias[2]],
    )
}

fn matrix_vector(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    matrix.map(|row| row[0].mul_add(vector[0], row[1].mul_add(vector[1], row[2] * vector[2])))
}

fn validate_calibrated(
    quantity: CalibratedQuantity,
    values: [f64; 3],
) -> Result<(), RawImuCalibrationError> {
    if let Some(axis) = values.iter().position(|value| !value.is_finite()) {
        return Err(RawImuCalibrationError::CalibratedOutputNonFinite { quantity, axis });
    }
    Ok(())
}

/// Weakly typed configuration parsed once into [`PlanarOdometryConfig`].
#[derive(Clone, Debug)]
pub struct PlanarOdometryConfigDto {
    pub raw_imu_calibration: RawImuCalibrationDto,
    pub tracking_camera_to_base: TrackingCameraToBase,
    pub world_to_occupancy: WorldToOccupancy,
    pub max_visual_interval: Duration,
    pub max_visual_linear_speed_m_per_sec: f64,
    pub max_visual_yaw_rate_rad_per_sec: f64,
    pub max_calibrated_yaw_rate_rad_per_sec: f64,
    pub minimum_gyro_accuracy: SensorAccuracy,
    pub max_vertical_increment_m: f64,
    pub max_relative_roll_pitch_increment_rad: f64,
    pub max_absolute_map_roll_pitch_rad: f64,
    pub max_imu_gap: Duration,
    pub max_prediction_age: Duration,
    pub max_host_observation_age: Duration,
    pub max_history_bracket_gap: Duration,
    pub gyro_history_capacity: usize,
    pub pose_history_capacity: usize,
}

#[derive(Clone, Debug)]
pub struct PlanarOdometryConfig {
    raw_imu_calibration: RawImuCalibration,
    tracking_camera_to_base: TrackingCameraToBase,
    world_to_occupancy: WorldToOccupancy,
    max_visual_interval_ns: u64,
    max_visual_linear_speed_m_per_sec: f64,
    max_visual_yaw_rate_rad_per_sec: f64,
    max_calibrated_yaw_rate_rad_per_sec: f64,
    minimum_gyro_accuracy: SensorAccuracy,
    max_vertical_increment_m: f64,
    max_relative_roll_pitch_increment_rad: f64,
    max_absolute_map_roll_pitch_rad: f64,
    max_imu_gap_ns: u64,
    max_prediction_age_ns: u64,
    max_host_observation_age_ns: u64,
    max_history_bracket_gap_ns: u64,
    gyro_history_capacity: NonZeroUsize,
    pose_history_capacity: NonZeroUsize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DurationParameter {
    MaximumVisualInterval,
    MaximumImuGap,
    MaximumPredictionAge,
    MaximumHostObservationAge,
    MaximumHistoryBracketGap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarParameter {
    MaximumVisualLinearSpeed,
    MaximumVisualYawRate,
    MaximumCalibratedYawRate,
    MaximumVerticalIncrement,
    MaximumRelativeRollPitchIncrement,
    MaximumAbsoluteMapRollPitch,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PlanarOdometryConfigError {
    RawImuCalibration(RawImuCalibrationError),
    ZeroDuration(DurationParameter),
    DurationNotRepresentable {
        parameter: DurationParameter,
        nanoseconds: u128,
    },
    InvalidPositiveScalar {
        parameter: ScalarParameter,
        value: f64,
    },
    InvalidNonnegativeScalar {
        parameter: ScalarParameter,
        value: f64,
    },
    RollPitchLimitOutOfRange {
        parameter: ScalarParameter,
        value: f64,
        maximum_exclusive: f64,
    },
    ZeroGyroHistoryCapacity,
    GyroHistoryCapacityTooLarge {
        actual: usize,
        maximum: usize,
    },
    PoseHistoryCapacityTooSmall {
        actual: usize,
        minimum: usize,
    },
    PoseHistoryCapacityTooLarge {
        actual: usize,
        maximum: usize,
    },
}

impl std::fmt::Display for PlanarOdometryConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid planar-odometry configuration: {self:?}")
    }
}

impl std::error::Error for PlanarOdometryConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::RawImuCalibration(error) => Some(error),
            _ => None,
        }
    }
}

impl PlanarOdometryConfig {
    pub const MAX_GYRO_HISTORY_CAPACITY: usize = MAX_CONFIG_GYRO_HISTORY_CAPACITY;
    pub const MAX_POSE_HISTORY_CAPACITY: usize = MAX_CONFIG_POSE_HISTORY_CAPACITY;
    pub const MIN_EULER_SINGULARITY_MARGIN_RAD: f64 = CONFIG_EULER_SINGULARITY_MARGIN_RAD;

    pub fn parse(dto: PlanarOdometryConfigDto) -> Result<Self, PlanarOdometryConfigError> {
        let raw_imu_calibration = RawImuCalibration::parse(dto.raw_imu_calibration)
            .map_err(PlanarOdometryConfigError::RawImuCalibration)?;
        let max_visual_interval_ns = duration_ns(
            DurationParameter::MaximumVisualInterval,
            dto.max_visual_interval,
        )?;
        let max_imu_gap_ns = duration_ns(DurationParameter::MaximumImuGap, dto.max_imu_gap)?;
        let max_prediction_age_ns = duration_ns(
            DurationParameter::MaximumPredictionAge,
            dto.max_prediction_age,
        )?;
        let max_host_observation_age_ns = duration_ns(
            DurationParameter::MaximumHostObservationAge,
            dto.max_host_observation_age,
        )?;
        let max_history_bracket_gap_ns = duration_ns(
            DurationParameter::MaximumHistoryBracketGap,
            dto.max_history_bracket_gap,
        )?;
        validate_positive(
            ScalarParameter::MaximumVisualLinearSpeed,
            dto.max_visual_linear_speed_m_per_sec,
        )?;
        validate_positive(
            ScalarParameter::MaximumVisualYawRate,
            dto.max_visual_yaw_rate_rad_per_sec,
        )?;
        validate_positive(
            ScalarParameter::MaximumCalibratedYawRate,
            dto.max_calibrated_yaw_rate_rad_per_sec,
        )?;
        validate_nonnegative(
            ScalarParameter::MaximumVerticalIncrement,
            dto.max_vertical_increment_m,
        )?;
        validate_roll_pitch_limit(
            ScalarParameter::MaximumRelativeRollPitchIncrement,
            dto.max_relative_roll_pitch_increment_rad,
        )?;
        validate_roll_pitch_limit(
            ScalarParameter::MaximumAbsoluteMapRollPitch,
            dto.max_absolute_map_roll_pitch_rad,
        )?;
        let gyro_history_capacity = NonZeroUsize::new(dto.gyro_history_capacity)
            .ok_or(PlanarOdometryConfigError::ZeroGyroHistoryCapacity)?;
        if gyro_history_capacity.get() > MAX_CONFIG_GYRO_HISTORY_CAPACITY {
            return Err(PlanarOdometryConfigError::GyroHistoryCapacityTooLarge {
                actual: gyro_history_capacity.get(),
                maximum: MAX_CONFIG_GYRO_HISTORY_CAPACITY,
            });
        }
        let pose_history_capacity = NonZeroUsize::new(dto.pose_history_capacity).ok_or(
            PlanarOdometryConfigError::PoseHistoryCapacityTooSmall {
                actual: 0,
                minimum: 2,
            },
        )?;
        if pose_history_capacity.get() < 2 {
            return Err(PlanarOdometryConfigError::PoseHistoryCapacityTooSmall {
                actual: pose_history_capacity.get(),
                minimum: 2,
            });
        }
        if pose_history_capacity.get() > MAX_CONFIG_POSE_HISTORY_CAPACITY {
            return Err(PlanarOdometryConfigError::PoseHistoryCapacityTooLarge {
                actual: pose_history_capacity.get(),
                maximum: MAX_CONFIG_POSE_HISTORY_CAPACITY,
            });
        }

        Ok(Self {
            raw_imu_calibration,
            tracking_camera_to_base: dto.tracking_camera_to_base,
            world_to_occupancy: dto.world_to_occupancy,
            max_visual_interval_ns,
            max_visual_linear_speed_m_per_sec: dto.max_visual_linear_speed_m_per_sec,
            max_visual_yaw_rate_rad_per_sec: dto.max_visual_yaw_rate_rad_per_sec,
            max_calibrated_yaw_rate_rad_per_sec: dto.max_calibrated_yaw_rate_rad_per_sec,
            minimum_gyro_accuracy: dto.minimum_gyro_accuracy,
            max_vertical_increment_m: dto.max_vertical_increment_m,
            max_relative_roll_pitch_increment_rad: dto.max_relative_roll_pitch_increment_rad,
            max_absolute_map_roll_pitch_rad: dto.max_absolute_map_roll_pitch_rad,
            max_imu_gap_ns,
            max_prediction_age_ns,
            max_host_observation_age_ns,
            max_history_bracket_gap_ns,
            gyro_history_capacity,
            pose_history_capacity,
        })
    }

    pub fn raw_imu_calibration(&self) -> &RawImuCalibration {
        &self.raw_imu_calibration
    }

    pub fn tracking_camera_to_base(&self) -> TrackingCameraToBase {
        self.tracking_camera_to_base
    }

    pub fn world_to_occupancy(&self) -> WorldToOccupancy {
        self.world_to_occupancy
    }
}

fn duration_ns(
    parameter: DurationParameter,
    duration: Duration,
) -> Result<u64, PlanarOdometryConfigError> {
    let nanoseconds = duration.as_nanos();
    if nanoseconds == 0 {
        return Err(PlanarOdometryConfigError::ZeroDuration(parameter));
    }
    u64::try_from(nanoseconds).map_err(|_| PlanarOdometryConfigError::DurationNotRepresentable {
        parameter,
        nanoseconds,
    })
}

fn validate_positive(
    parameter: ScalarParameter,
    value: f64,
) -> Result<(), PlanarOdometryConfigError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(PlanarOdometryConfigError::InvalidPositiveScalar { parameter, value });
    }
    Ok(())
}

fn validate_nonnegative(
    parameter: ScalarParameter,
    value: f64,
) -> Result<(), PlanarOdometryConfigError> {
    if !value.is_finite() || value < 0.0 {
        return Err(PlanarOdometryConfigError::InvalidNonnegativeScalar { parameter, value });
    }
    Ok(())
}

fn validate_roll_pitch_limit(
    parameter: ScalarParameter,
    value: f64,
) -> Result<(), PlanarOdometryConfigError> {
    validate_positive(parameter, value)?;
    let maximum_exclusive = std::f64::consts::FRAC_PI_2 - CONFIG_EULER_SINGULARITY_MARGIN_RAD;
    if value >= maximum_exclusive {
        return Err(PlanarOdometryConfigError::RollPitchLimitOutOfRange {
            parameter,
            value,
            maximum_exclusive,
        });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OdomSegmentId(NonZeroU64);

impl OdomSegmentId {
    pub fn try_new(raw: u64) -> Result<Self, OdomSegmentIdError> {
        NonZeroU64::new(raw).map(Self).ok_or(OdomSegmentIdError)
    }

    pub fn as_u64(self) -> u64 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OdomSegmentIdError;

impl std::fmt::Display for OdomSegmentIdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "odom segment ID must be nonzero")
    }
}

impl std::error::Error for OdomSegmentIdError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisualCaptureProvenance {
    session_id: DeviceSessionId,
    device_timestamp: DeviceTimestamp,
    host_arrival: HostMonotonicTimestamp,
    visual_stamp: VisualFrameStamp,
}

impl VisualCaptureProvenance {
    pub fn session_id(self) -> DeviceSessionId {
        self.session_id
    }

    pub fn device_timestamp(self) -> DeviceTimestamp {
        self.device_timestamp
    }

    pub fn host_arrival(self) -> HostMonotonicTimestamp {
        self.host_arrival
    }

    pub fn visual_stamp(self) -> VisualFrameStamp {
        self.visual_stamp
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OdomPlanarTwist {
    linear_x_in_odom_m_per_sec: f64,
    linear_y_in_odom_m_per_sec: f64,
    yaw_rate_rad_per_sec: f64,
}

impl OdomPlanarTwist {
    pub fn zero() -> Self {
        Self {
            linear_x_in_odom_m_per_sec: 0.0,
            linear_y_in_odom_m_per_sec: 0.0,
            yaw_rate_rad_per_sec: 0.0,
        }
    }

    fn try_new(x: f64, y: f64, yaw: f64) -> Result<Self, OdometryError> {
        let values = [x, y, yaw];
        if let Some(component) = values.iter().position(|value| !value.is_finite()) {
            return Err(OdometryError::NonFiniteTwist {
                component,
                value: values[component],
            });
        }
        Ok(Self {
            linear_x_in_odom_m_per_sec: x,
            linear_y_in_odom_m_per_sec: y,
            yaw_rate_rad_per_sec: yaw,
        })
    }

    pub fn linear_x_in_odom_m_per_sec(self) -> f64 {
        self.linear_x_in_odom_m_per_sec
    }

    pub fn linear_y_in_odom_m_per_sec(self) -> f64 {
        self.linear_y_in_odom_m_per_sec
    }

    pub fn yaw_rate_rad_per_sec(self) -> f64 {
        self.yaw_rate_rad_per_sec
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum OdometryQuality {
    Visual {
        basis: Option<VisualIncrementBasis>,
    },
    Predicted {
        visual_age_ns: u64,
        integration_from: DeviceTimestamp,
        integration_through: DeviceTimestamp,
        gyro_sample_from: DeviceTimestamp,
        gyro_sample_through: DeviceTimestamp,
        endpoint_gyro_yaw_rate_rad_per_sec: f64,
        minimum_gyro_accuracy: SensorAccuracy,
        calibration: ImuCalibrationProvenance,
        translation_model: PredictionTranslationModel,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredictionTranslationModel {
    BoundedConstantVisualVelocity,
}

#[derive(Clone, Debug)]
pub struct OdometryState {
    segment_id: OdomSegmentId,
    session_id: DeviceSessionId,
    timestamp: DeviceTimestamp,
    base_to_odom: BaseToOdom,
    odom_to_map: OdomToMap,
    map_snapshot: MapSnapshot,
    twist: OdomPlanarTwist,
    source_visual: VisualCaptureProvenance,
    quality: OdometryQuality,
}

impl OdometryState {
    pub fn segment_id(&self) -> OdomSegmentId {
        self.segment_id
    }

    pub fn session_id(&self) -> DeviceSessionId {
        self.session_id
    }

    pub fn timestamp(&self) -> DeviceTimestamp {
        self.timestamp
    }

    pub fn base_to_odom(&self) -> BaseToOdom {
        self.base_to_odom
    }

    pub fn odom_to_map(&self) -> OdomToMap {
        self.odom_to_map
    }

    pub fn map_to_odom(&self) -> Result<MapToOdom, PlanarTransformError> {
        self.odom_to_map.inverse()
    }

    pub fn map_snapshot(&self) -> MapSnapshot {
        self.map_snapshot
    }

    pub fn twist(&self) -> OdomPlanarTwist {
        self.twist
    }

    pub fn source_visual(&self) -> VisualCaptureProvenance {
        self.source_visual
    }

    pub fn quality(&self) -> &OdometryQuality {
        &self.quality
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReanchorReason {
    DeviceSessionChanged {
        expected: DeviceSessionId,
        actual: DeviceSessionId,
    },
    MapInstanceChanged,
    VisualStampDiscontinuity {
        expected: VisualFrameStamp,
        actual: VisualFrameStamp,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum OdometryError {
    NotAnchored,
    SegmentIdNotIncreasing {
        previous: OdomSegmentId,
        requested: OdomSegmentId,
    },
    ReanchorRequired(ReanchorReason),
    VisualTimestamp(InertialValueError),
    HostClockRegression {
        arrival: HostMonotonicTimestamp,
        now: HostMonotonicTimestamp,
    },
    ObservationTooOld {
        age_ns: u64,
        maximum_ns: u64,
    },
    VisualHostArrivalRegression {
        previous: HostMonotonicTimestamp,
        current: HostMonotonicTimestamp,
    },
    LocalizationStampMismatch {
        expected: VisualFrameStamp,
        actual: VisualFrameStamp,
    },
    BasisMapInstanceMismatch {
        basis: MapSnapshot,
        localization: MapSnapshot,
    },
    BasisMapRevisionAhead {
        basis: MapSnapshot,
        localization: MapSnapshot,
    },
    DivergentBasisMapRevision {
        basis: MapSnapshot,
        localization: MapSnapshot,
    },
    MapRevisionRegression,
    DivergentMapRevision,
    VisualIntervalTooLarge {
        interval_ns: u64,
        maximum_ns: u64,
    },
    VisualLinearSpeedTooLarge {
        speed_m_per_sec: f64,
        maximum_m_per_sec: f64,
    },
    VisualYawRateTooLarge {
        rate_rad_per_sec: f64,
        maximum_rad_per_sec: f64,
    },
    PlanarityViolation {
        component: PlanarityComponent,
        magnitude: f64,
        maximum: f64,
    },
    Pose64(Pose64Error),
    PlanarTransform(PlanarTransformError),
    RawImuCalibration(RawImuCalibrationError),
    CalibratedYawRateTooLarge {
        rate_rad_per_sec: f64,
        maximum_rad_per_sec: f64,
    },
    GyroAccuracyBelowMinimum {
        actual: SensorAccuracy,
        minimum: SensorAccuracy,
    },
    InertialOrdering(InertialOrderingError),
    NonFiniteTwist {
        component: usize,
        value: f64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanarityComponent {
    VerticalTranslation,
    Roll,
    Pitch,
    MapRoll,
    MapPitch,
}

impl std::fmt::Display for OdometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "planar odometry rejected observation: {self:?}")
    }
}

impl std::error::Error for OdometryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::VisualTimestamp(error) => Some(error),
            Self::Pose64(error) => Some(error),
            Self::PlanarTransform(error) => Some(error),
            Self::RawImuCalibration(error) => Some(error),
            Self::InertialOrdering(error) => Some(error),
            _ => None,
        }
    }
}

impl From<Pose64Error> for OdometryError {
    fn from(value: Pose64Error) -> Self {
        Self::Pose64(value)
    }
}

impl From<PlanarTransformError> for OdometryError {
    fn from(value: PlanarTransformError) -> Self {
        Self::PlanarTransform(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TranslationIntegration {
    DisabledNoEncoderNoAccelerationIntegration,
}

#[derive(Clone, Debug)]
pub struct ImuUpdate {
    pub order: InertialOrderOutcome,
    pub history_reset_for_gap: bool,
    pub calibrated_angular_velocity: BaseAngularVelocity,
    pub calibrated_acceleration: BaseAcceleration,
    pub gyro_accuracy: SensorAccuracy,
    pub accel_accuracy: SensorAccuracy,
    pub calibration: ImuCalibrationProvenance,
    pub translation_integration: TranslationIntegration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OdometryUnavailable {
    NotAnchored,
    SessionMismatch,
    QueryBeforeHistory,
    HistoryBracketTooWide { bracket_ns: u64, maximum_ns: u64 },
    PredictionExpired { age_ns: u64, maximum_ns: u64 },
    NoGyroCoverage,
    GyroGap { gap_ns: u64, maximum_ns: u64 },
    SupportingImuHostArrivalStale { age_ns: u64, maximum_ns: u64 },
    HostClockRegression,
}

#[derive(Clone, Debug)]
// Prediction runs on the control path; keep the available state inline rather
// than add one heap allocation to every estimate solely to equalize variants.
#[allow(clippy::large_enum_variant)]
pub enum OdometryEstimate {
    Available(OdometryState),
    Unavailable(OdometryUnavailable),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeAlignment {
    ExactVisual,
    InterpolatedVisual {
        before: DeviceTimestamp,
        after: DeviceTimestamp,
    },
    Predicted,
}

#[derive(Clone, Debug)]
pub struct TimeAlignedOdomPose {
    segment_id: OdomSegmentId,
    session_id: DeviceSessionId,
    timestamp: DeviceTimestamp,
    base_to_odom: BaseToOdom,
    alignment: TimeAlignment,
}

impl TimeAlignedOdomPose {
    pub fn segment_id(&self) -> OdomSegmentId {
        self.segment_id
    }

    pub fn session_id(&self) -> DeviceSessionId {
        self.session_id
    }

    pub fn timestamp(&self) -> DeviceTimestamp {
        self.timestamp
    }

    pub fn base_to_odom(&self) -> BaseToOdom {
        self.base_to_odom
    }

    pub fn alignment(&self) -> TimeAlignment {
        self.alignment
    }
}

#[derive(Clone, Debug)]
pub enum PoseHistoryQuery {
    Available(TimeAlignedOdomPose),
    Unavailable(OdometryUnavailable),
}

#[derive(Clone, Copy, Debug)]
struct GyroHistorySample {
    timestamp: DeviceTimestamp,
    host_arrival: HostMonotonicTimestamp,
    yaw_rate_rad_per_sec: f64,
    accuracy: SensorAccuracy,
}

#[derive(Clone, Copy, Debug)]
struct PoseHistoryEntry {
    timestamp: DeviceTimestamp,
    base_to_odom: BaseToOdom,
}

#[derive(Debug)]
struct ActiveSegment {
    current: OdometryState,
    inertial_order: InertialOrderTracker,
    gyro_history: VecDeque<GyroHistorySample>,
    pose_history: VecDeque<PoseHistoryEntry>,
}

impl ActiveSegment {
    fn segment_id(&self) -> OdomSegmentId {
        self.current.segment_id
    }

    fn session_id(&self) -> DeviceSessionId {
        self.current.session_id
    }

    fn last_visual_stamp(&self) -> VisualFrameStamp {
        self.current.source_visual.visual_stamp
    }

    fn last_visual_host_arrival(&self) -> HostMonotonicTimestamp {
        self.current.source_visual.host_arrival
    }
}

#[derive(Debug)]
pub struct PlanarOdometry {
    config: PlanarOdometryConfig,
    active: Option<ActiveSegment>,
    last_segment_id: Option<OdomSegmentId>,
}

impl PlanarOdometry {
    pub fn new(config: PlanarOdometryConfig) -> Self {
        Self {
            config,
            active: None,
            last_segment_id: None,
        }
    }

    pub fn config(&self) -> &PlanarOdometryConfig {
        &self.config
    }

    pub fn current(&self) -> Option<&OdometryState> {
        self.active.as_ref().map(|active| &active.current)
    }

    /// Explicitly start a new continuous odom segment at the supplied current
    /// map localization. The base origin is identity in the new odom frame.
    pub fn reanchor(
        &mut self,
        segment_id: OdomSegmentId,
        session_id: DeviceSessionId,
        localization: MapLocalization,
        host_arrival: HostMonotonicTimestamp,
        now: HostMonotonicTimestamp,
    ) -> Result<OdometryState, OdometryError> {
        if let Some(previous) = self.last_segment_id
            && segment_id <= previous
        {
            return Err(OdometryError::SegmentIdNotIncreasing {
                previous,
                requested: segment_id,
            });
        }
        validate_host_age(&self.config, host_arrival, now)?;
        let visual_stamp = localization.visual_stamp();
        let device_timestamp = parse_visual_timestamp(visual_stamp.timestamp())?;
        let base_to_odom = BaseToOdom::try_new(0.0, 0.0, 0.0)?;
        let base_to_map = self.base_to_map(localization)?;
        let odom_to_map = base_to_map.compose(base_to_odom.inverse()?)?;
        let source_visual = VisualCaptureProvenance {
            session_id,
            device_timestamp,
            host_arrival,
            visual_stamp,
        };
        let current = OdometryState {
            segment_id,
            session_id,
            timestamp: device_timestamp,
            base_to_odom,
            odom_to_map,
            map_snapshot: localization.map_snapshot(),
            twist: OdomPlanarTwist::zero(),
            source_visual,
            quality: OdometryQuality::Visual { basis: None },
        };
        let mut pose_history = VecDeque::with_capacity(self.config.pose_history_capacity.get());
        pose_history.push_back(PoseHistoryEntry {
            timestamp: device_timestamp,
            base_to_odom,
        });
        let active = ActiveSegment {
            current: current.clone(),
            inertial_order: InertialOrderTracker::with_session(session_id),
            gyro_history: VecDeque::with_capacity(self.config.gyro_history_capacity.get()),
            pose_history,
        };

        self.active = Some(active);
        self.last_segment_id = Some(segment_id);
        Ok(current)
    }

    /// Accept one correction-safe visual increment and its exact current map
    /// localization transactionally.
    pub fn observe_visual(
        &mut self,
        session_id: DeviceSessionId,
        increment: VisualIncrement,
        localization: MapLocalization,
        host_arrival: HostMonotonicTimestamp,
        now: HostMonotonicTimestamp,
    ) -> Result<OdometryState, OdometryError> {
        let active = self.active.as_ref().ok_or(OdometryError::NotAnchored)?;
        validate_session(active.session_id(), session_id)?;
        validate_host_age(&self.config, host_arrival, now)?;
        if host_arrival < active.last_visual_host_arrival() {
            return Err(OdometryError::VisualHostArrivalRegression {
                previous: active.last_visual_host_arrival(),
                current: host_arrival,
            });
        }
        if increment.from() != active.last_visual_stamp() {
            return Err(OdometryError::ReanchorRequired(
                ReanchorReason::VisualStampDiscontinuity {
                    expected: active.last_visual_stamp(),
                    actual: increment.from(),
                },
            ));
        }
        if localization.visual_stamp() != increment.to() {
            return Err(OdometryError::LocalizationStampMismatch {
                expected: increment.to(),
                actual: localization.visual_stamp(),
            });
        }
        validate_map_lineage(active.current.map_snapshot, localization.map_snapshot())?;
        validate_visual_basis(
            increment.basis().map_snapshot(),
            localization.map_snapshot(),
        )?;

        // Exact stamp equality above proves this weak tracker timestamp is the
        // boundary value already parsed into the active domain state.
        let from_timestamp = active.current.timestamp;
        let to_timestamp = parse_visual_timestamp(increment.to().timestamp())?;
        let interval_ns = to_timestamp
            .as_nanos()
            .checked_sub(from_timestamp.as_nanos())
            .ok_or(OdometryError::ReanchorRequired(
                ReanchorReason::VisualStampDiscontinuity {
                    expected: active.last_visual_stamp(),
                    actual: increment.to(),
                },
            ))?;
        if interval_ns == 0 || interval_ns > self.config.max_visual_interval_ns {
            return Err(OdometryError::VisualIntervalTooLarge {
                interval_ns,
                maximum_ns: self.config.max_visual_interval_ns,
            });
        }
        let delta = self.visual_base_delta(increment)?;
        let interval_sec = nanoseconds_to_seconds(interval_ns);
        let linear_speed = delta
            .source_origin_x_in_destination_m()
            .hypot(delta.source_origin_y_in_destination_m())
            / interval_sec;
        if !linear_speed.is_finite() || linear_speed > self.config.max_visual_linear_speed_m_per_sec
        {
            return Err(OdometryError::VisualLinearSpeedTooLarge {
                speed_m_per_sec: linear_speed,
                maximum_m_per_sec: self.config.max_visual_linear_speed_m_per_sec,
            });
        }
        let yaw_rate = delta.source_yaw_in_destination_rad().abs() / interval_sec;
        if !yaw_rate.is_finite() || yaw_rate > self.config.max_visual_yaw_rate_rad_per_sec {
            return Err(OdometryError::VisualYawRateTooLarge {
                rate_rad_per_sec: yaw_rate,
                maximum_rad_per_sec: self.config.max_visual_yaw_rate_rad_per_sec,
            });
        }

        let previous_pose = active.current.base_to_odom;
        let base_to_odom = previous_pose.compose(delta)?;
        let twist = OdomPlanarTwist::try_new(
            (base_to_odom.source_origin_x_in_destination_m()
                - previous_pose.source_origin_x_in_destination_m())
                / interval_sec,
            (base_to_odom.source_origin_y_in_destination_m()
                - previous_pose.source_origin_y_in_destination_m())
                / interval_sec,
            signed_yaw_delta(
                previous_pose.source_yaw_in_destination_rad(),
                base_to_odom.source_yaw_in_destination_rad(),
            ) / interval_sec,
        )?;
        let base_to_map = self.base_to_map(localization)?;
        let odom_to_map = base_to_map.compose(base_to_odom.inverse()?)?;
        let source_visual = VisualCaptureProvenance {
            session_id,
            device_timestamp: to_timestamp,
            host_arrival,
            visual_stamp: increment.to(),
        };
        let current = OdometryState {
            segment_id: active.segment_id(),
            session_id,
            timestamp: to_timestamp,
            base_to_odom,
            odom_to_map,
            map_snapshot: localization.map_snapshot(),
            twist,
            source_visual,
            quality: OdometryQuality::Visual {
                basis: Some(increment.basis()),
            },
        };

        let Some(active) = self.active.as_mut() else {
            return Err(OdometryError::NotAnchored);
        };
        active.current = current.clone();
        push_bounded(
            &mut active.pose_history,
            self.config.pose_history_capacity,
            PoseHistoryEntry {
                timestamp: to_timestamp,
                base_to_odom,
            },
        );
        Ok(current)
    }

    /// Update only `map <- odom` from a corrected localization of the exact
    /// current visual frame. The odom pose and twist are left byte-for-byte
    /// unchanged.
    pub fn observe_map_localization(
        &mut self,
        session_id: DeviceSessionId,
        localization: MapLocalization,
    ) -> Result<OdometryState, OdometryError> {
        let active = self.active.as_ref().ok_or(OdometryError::NotAnchored)?;
        validate_session(active.session_id(), session_id)?;
        if localization.visual_stamp() != active.last_visual_stamp() {
            return Err(OdometryError::LocalizationStampMismatch {
                expected: active.last_visual_stamp(),
                actual: localization.visual_stamp(),
            });
        }
        validate_map_lineage(active.current.map_snapshot, localization.map_snapshot())?;
        let base_to_map = self.base_to_map(localization)?;
        let odom_to_map = base_to_map.compose(active.current.base_to_odom.inverse()?)?;

        let Some(active) = self.active.as_mut() else {
            return Err(OdometryError::NotAnchored);
        };
        active.current.odom_to_map = odom_to_map;
        active.current.map_snapshot = localization.map_snapshot();
        Ok(active.current.clone())
    }

    /// Calibrate and retain one raw OAK IMU report. Missing dequeue reports or
    /// gyro device-time gaps exceeding the configured integration bound reset
    /// gyro history instead of integrating across unknown motion. A large
    /// accelerometer-only timestamp step does not discard continuous gyro
    /// coverage.
    pub fn observe_imu(
        &mut self,
        report: ImuReport,
        now: HostMonotonicTimestamp,
    ) -> Result<ImuUpdate, OdometryError> {
        let active = self.active.as_ref().ok_or(OdometryError::NotAnchored)?;
        validate_session(active.session_id(), report.session_id())?;
        validate_host_age(&self.config, report.host_arrival(), now)?;

        let mut candidate_order = active.inertial_order.clone();
        let order = candidate_order
            .observe(&report)
            .map_err(OdometryError::InertialOrdering)?;
        let gyro_accuracy = report.gyro().accuracy();
        if !accuracy_meets_minimum(gyro_accuracy, self.config.minimum_gyro_accuracy) {
            return Err(OdometryError::GyroAccuracyBelowMinimum {
                actual: gyro_accuracy,
                minimum: self.config.minimum_gyro_accuracy,
            });
        }
        let calibrated_angular_velocity = self
            .config
            .raw_imu_calibration
            .calibrate_angular_velocity(report.gyro().angular_velocity())
            .map_err(OdometryError::RawImuCalibration)?;
        let calibrated_acceleration = self
            .config
            .raw_imu_calibration
            .calibrate_acceleration(report.accel().acceleration())
            .map_err(OdometryError::RawImuCalibration)?;
        let gyro_timestamp = report.gyro().timestamp();
        let calibrated_yaw_rate = calibrated_angular_velocity.yaw_rate_rad_per_sec();
        if calibrated_yaw_rate.abs() > self.config.max_calibrated_yaw_rate_rad_per_sec {
            return Err(OdometryError::CalibratedYawRateTooLarge {
                rate_rad_per_sec: calibrated_yaw_rate,
                maximum_rad_per_sec: self.config.max_calibrated_yaw_rate_rad_per_sec,
            });
        }
        let gyro_gap = active
            .gyro_history
            .back()
            .map(|previous| gyro_timestamp.as_nanos() - previous.timestamp.as_nanos());
        let history_reset_for_gap = matches!(order, InertialOrderOutcome::Gap { .. })
            || gyro_gap.is_some_and(|gap| gap > self.config.max_imu_gap_ns);
        let sample = GyroHistorySample {
            timestamp: gyro_timestamp,
            host_arrival: report.host_arrival(),
            yaw_rate_rad_per_sec: calibrated_yaw_rate,
            accuracy: gyro_accuracy,
        };

        let Some(active) = self.active.as_mut() else {
            return Err(OdometryError::NotAnchored);
        };
        active.inertial_order = candidate_order;
        if history_reset_for_gap {
            active.gyro_history.clear();
        }
        push_bounded(
            &mut active.gyro_history,
            self.config.gyro_history_capacity,
            sample,
        );
        Ok(ImuUpdate {
            order,
            history_reset_for_gap,
            calibrated_angular_velocity,
            calibrated_acceleration,
            gyro_accuracy,
            accel_accuracy: report.accel().accuracy(),
            calibration: self.config.raw_imu_calibration.provenance().clone(),
            translation_integration:
                TranslationIntegration::DisabledNoEncoderNoAccelerationIntegration,
        })
    }

    pub fn estimate(
        &self,
        session_id: DeviceSessionId,
        timestamp: DeviceTimestamp,
        now: HostMonotonicTimestamp,
    ) -> Result<OdometryEstimate, OdometryError> {
        let Some(active) = self.active.as_ref() else {
            return Ok(OdometryEstimate::Unavailable(
                OdometryUnavailable::NotAnchored,
            ));
        };
        if session_id != active.session_id() {
            return Ok(OdometryEstimate::Unavailable(
                OdometryUnavailable::SessionMismatch,
            ));
        }
        if timestamp == active.current.timestamp {
            return Ok(OdometryEstimate::Available(active.current.clone()));
        }
        if timestamp < active.current.timestamp {
            return Ok(OdometryEstimate::Unavailable(
                OdometryUnavailable::QueryBeforeHistory,
            ));
        }
        self.predict(active, timestamp, now)
    }

    /// Time-align a depth capture to odom using exact/interpolated visual poses
    /// or the same explicitly bounded prediction policy as [`Self::estimate`].
    pub fn pose_at(
        &self,
        session_id: DeviceSessionId,
        timestamp: DeviceTimestamp,
        now: HostMonotonicTimestamp,
    ) -> Result<PoseHistoryQuery, OdometryError> {
        let Some(active) = self.active.as_ref() else {
            return Ok(PoseHistoryQuery::Unavailable(
                OdometryUnavailable::NotAnchored,
            ));
        };
        if session_id != active.session_id() {
            return Ok(PoseHistoryQuery::Unavailable(
                OdometryUnavailable::SessionMismatch,
            ));
        }
        if let Some(entry) = active
            .pose_history
            .iter()
            .find(|entry| entry.timestamp == timestamp)
        {
            return Ok(PoseHistoryQuery::Available(TimeAlignedOdomPose {
                segment_id: active.segment_id(),
                session_id,
                timestamp,
                base_to_odom: entry.base_to_odom,
                alignment: TimeAlignment::ExactVisual,
            }));
        }
        if let Some((before, after)) = bracket(&active.pose_history, timestamp) {
            let bracket_ns = after.timestamp.as_nanos() - before.timestamp.as_nanos();
            if bracket_ns > self.config.max_history_bracket_gap_ns {
                return Ok(PoseHistoryQuery::Unavailable(
                    OdometryUnavailable::HistoryBracketTooWide {
                        bracket_ns,
                        maximum_ns: self.config.max_history_bracket_gap_ns,
                    },
                ));
            }
            let offset_ns = timestamp.as_nanos() - before.timestamp.as_nanos();
            let fraction = offset_ns as f64 / bracket_ns as f64;
            let base_to_odom =
                interpolate_planar(before.base_to_odom, after.base_to_odom, fraction)?;
            return Ok(PoseHistoryQuery::Available(TimeAlignedOdomPose {
                segment_id: active.segment_id(),
                session_id,
                timestamp,
                base_to_odom,
                alignment: TimeAlignment::InterpolatedVisual {
                    before: before.timestamp,
                    after: after.timestamp,
                },
            }));
        }
        let Some(first) = active.pose_history.front() else {
            return Ok(PoseHistoryQuery::Unavailable(
                OdometryUnavailable::QueryBeforeHistory,
            ));
        };
        if timestamp < first.timestamp {
            return Ok(PoseHistoryQuery::Unavailable(
                OdometryUnavailable::QueryBeforeHistory,
            ));
        }
        match self.predict(active, timestamp, now)? {
            OdometryEstimate::Available(state) => {
                Ok(PoseHistoryQuery::Available(TimeAlignedOdomPose {
                    segment_id: state.segment_id,
                    session_id: state.session_id,
                    timestamp,
                    base_to_odom: state.base_to_odom,
                    alignment: TimeAlignment::Predicted,
                }))
            }
            OdometryEstimate::Unavailable(reason) => Ok(PoseHistoryQuery::Unavailable(reason)),
        }
    }

    fn predict(
        &self,
        active: &ActiveSegment,
        timestamp: DeviceTimestamp,
        now: HostMonotonicTimestamp,
    ) -> Result<OdometryEstimate, OdometryError> {
        let visual_age_ns = timestamp.as_nanos() - active.current.timestamp.as_nanos();
        if visual_age_ns > self.config.max_prediction_age_ns {
            return Ok(OdometryEstimate::Unavailable(
                OdometryUnavailable::PredictionExpired {
                    age_ns: visual_age_ns,
                    maximum_ns: self.config.max_prediction_age_ns,
                },
            ));
        }
        let integration = match integrate_yaw(
            &active.gyro_history,
            active.current.timestamp,
            timestamp,
            self.config.max_imu_gap_ns,
        ) {
            Ok(value) => value,
            Err(reason) => return Ok(OdometryEstimate::Unavailable(reason)),
        };
        let latest_host_age = match now
            .as_nanos()
            .checked_sub(integration.latest_sample_host_arrival.as_nanos())
        {
            Some(age) => age,
            None => {
                return Ok(OdometryEstimate::Unavailable(
                    OdometryUnavailable::HostClockRegression,
                ));
            }
        };
        if latest_host_age > self.config.max_host_observation_age_ns {
            return Ok(OdometryEstimate::Unavailable(
                OdometryUnavailable::SupportingImuHostArrivalStale {
                    age_ns: latest_host_age,
                    maximum_ns: self.config.max_host_observation_age_ns,
                },
            ));
        }
        let dt_sec = nanoseconds_to_seconds(visual_age_ns);
        let pose = active.current.base_to_odom;
        let base_to_odom = BaseToOdom::try_new(
            active
                .current
                .twist
                .linear_x_in_odom_m_per_sec
                .mul_add(dt_sec, pose.source_origin_x_in_destination_m()),
            active
                .current
                .twist
                .linear_y_in_odom_m_per_sec
                .mul_add(dt_sec, pose.source_origin_y_in_destination_m()),
            pose.source_yaw_in_destination_rad() + integration.delta_yaw_rad,
        )?;
        let twist = OdomPlanarTwist::try_new(
            active.current.twist.linear_x_in_odom_m_per_sec,
            active.current.twist.linear_y_in_odom_m_per_sec,
            integration.endpoint_yaw_rate_rad_per_sec,
        )?;
        Ok(OdometryEstimate::Available(OdometryState {
            segment_id: active.segment_id(),
            session_id: active.session_id(),
            timestamp,
            base_to_odom,
            odom_to_map: active.current.odom_to_map,
            map_snapshot: active.current.map_snapshot,
            twist,
            source_visual: active.current.source_visual,
            quality: OdometryQuality::Predicted {
                visual_age_ns,
                integration_from: integration.from,
                integration_through: integration.through,
                gyro_sample_from: integration.sample_from,
                gyro_sample_through: integration.sample_through,
                endpoint_gyro_yaw_rate_rad_per_sec: integration.endpoint_yaw_rate_rad_per_sec,
                minimum_gyro_accuracy: integration.minimum_accuracy,
                calibration: self.config.raw_imu_calibration.provenance().clone(),
                translation_model: PredictionTranslationModel::BoundedConstantVisualVelocity,
            },
        }))
    }

    fn visual_base_delta(
        &self,
        increment: VisualIncrement,
    ) -> Result<PlanarTransform<BaseFrame, BaseFrame>, OdometryError> {
        let tracking_to_base = Pose64::try_from_pose32(self.config.tracking_camera_to_base.pose())?;
        let base_to_tracking = tracking_to_base.try_inverse()?;
        let current_camera_to_previous_camera = increment
            .previous_camera_to_current_camera()
            .try_inverse()?;
        let current_base_to_previous_base = tracking_to_base
            .try_compose(current_camera_to_previous_camera)?
            .try_compose(base_to_tracking)?;
        self.checked_planar_increment(current_base_to_previous_base)
    }

    fn checked_planar_increment(
        &self,
        transform: Pose64,
    ) -> Result<PlanarTransform<BaseFrame, BaseFrame>, OdometryError> {
        let translation = transform.translation();
        let (roll, pitch, yaw) = rotation_to_roll_pitch_yaw(transform.rotation());
        for (component, magnitude, maximum) in [
            (
                PlanarityComponent::VerticalTranslation,
                translation[2].abs(),
                self.config.max_vertical_increment_m,
            ),
            (
                PlanarityComponent::Roll,
                roll.abs(),
                self.config.max_relative_roll_pitch_increment_rad,
            ),
            (
                PlanarityComponent::Pitch,
                pitch.abs(),
                self.config.max_relative_roll_pitch_increment_rad,
            ),
        ] {
            if !magnitude.is_finite() || magnitude > maximum {
                return Err(OdometryError::PlanarityViolation {
                    component,
                    magnitude,
                    maximum,
                });
            }
        }
        PlanarTransform::try_new(translation[0], translation[1], yaw).map_err(Into::into)
    }

    fn base_to_map(
        &self,
        localization: MapLocalization,
    ) -> Result<PlanarTransform<BaseFrame, MapFrame>, OdometryError> {
        let world_to_map = Pose64::from_rt(
            self.config.world_to_occupancy.rotation(),
            self.config.world_to_occupancy.translation_m(),
        )?;
        let world_to_camera =
            Pose64::try_from_pose32(localization.world_to_camera().into_legacy_pose())?;
        let camera_to_world = world_to_camera.try_inverse()?;
        let tracking_to_base = Pose64::try_from_pose32(self.config.tracking_camera_to_base.pose())?;
        let base_to_tracking = tracking_to_base.try_inverse()?;
        let base_to_map = world_to_map
            .try_compose(camera_to_world)?
            .try_compose(base_to_tracking)?;
        let translation = base_to_map.translation();
        let (roll, pitch, yaw) = rotation_to_roll_pitch_yaw(base_to_map.rotation());
        for (component, magnitude) in [
            (PlanarityComponent::MapRoll, roll.abs()),
            (PlanarityComponent::MapPitch, pitch.abs()),
        ] {
            if !magnitude.is_finite() || magnitude > self.config.max_absolute_map_roll_pitch_rad {
                return Err(OdometryError::PlanarityViolation {
                    component,
                    magnitude,
                    maximum: self.config.max_absolute_map_roll_pitch_rad,
                });
            }
        }
        PlanarTransform::try_new(translation[0], translation[1], yaw).map_err(Into::into)
    }
}

fn validate_session(
    expected: DeviceSessionId,
    actual: DeviceSessionId,
) -> Result<(), OdometryError> {
    if expected != actual {
        return Err(OdometryError::ReanchorRequired(
            ReanchorReason::DeviceSessionChanged { expected, actual },
        ));
    }
    Ok(())
}

fn accuracy_rank(accuracy: SensorAccuracy) -> u8 {
    match accuracy {
        SensorAccuracy::Unreliable => 0,
        SensorAccuracy::Low => 1,
        SensorAccuracy::Medium => 2,
        SensorAccuracy::High => 3,
    }
}

fn accuracy_meets_minimum(actual: SensorAccuracy, minimum: SensorAccuracy) -> bool {
    accuracy_rank(actual) >= accuracy_rank(minimum)
}

fn lower_accuracy(first: SensorAccuracy, second: SensorAccuracy) -> SensorAccuracy {
    if accuracy_rank(first) <= accuracy_rank(second) {
        first
    } else {
        second
    }
}

fn validate_host_age(
    config: &PlanarOdometryConfig,
    arrival: HostMonotonicTimestamp,
    now: HostMonotonicTimestamp,
) -> Result<(), OdometryError> {
    let age_ns = now
        .as_nanos()
        .checked_sub(arrival.as_nanos())
        .ok_or(OdometryError::HostClockRegression { arrival, now })?;
    if age_ns > config.max_host_observation_age_ns {
        return Err(OdometryError::ObservationTooOld {
            age_ns,
            maximum_ns: config.max_host_observation_age_ns,
        });
    }
    Ok(())
}

fn validate_map_lineage(previous: MapSnapshot, current: MapSnapshot) -> Result<(), OdometryError> {
    if previous.instance_id() != current.instance_id() {
        return Err(OdometryError::ReanchorRequired(
            ReanchorReason::MapInstanceChanged,
        ));
    }
    if current.generation() < previous.generation() {
        return Err(OdometryError::MapRevisionRegression);
    }
    if current != previous && !previous.shares_mutation_lineage_with(current) {
        return Err(OdometryError::DivergentMapRevision);
    }
    Ok(())
}

fn validate_visual_basis(
    basis: MapSnapshot,
    localization: MapSnapshot,
) -> Result<(), OdometryError> {
    if basis.instance_id() != localization.instance_id() {
        return Err(OdometryError::BasisMapInstanceMismatch {
            basis,
            localization,
        });
    }
    if basis.generation() > localization.generation() {
        return Err(OdometryError::BasisMapRevisionAhead {
            basis,
            localization,
        });
    }
    if basis != localization && !basis.shares_mutation_lineage_with(localization) {
        return Err(OdometryError::DivergentBasisMapRevision {
            basis,
            localization,
        });
    }
    Ok(())
}

fn parse_visual_timestamp(timestamp: Timestamp) -> Result<DeviceTimestamp, OdometryError> {
    DeviceTimestamp::try_from_nanos(timestamp.as_nanos()).map_err(OdometryError::VisualTimestamp)
}

fn rotation_to_roll_pitch_yaw(rotation: [[f64; 3]; 3]) -> (f64, f64, f64) {
    let pitch = (-rotation[2][0]).clamp(-1.0, 1.0).asin();
    let roll = rotation[2][1].atan2(rotation[2][2]);
    let yaw = rotation[1][0].atan2(rotation[0][0]);
    (roll, pitch, yaw)
}

fn signed_yaw_delta(from: f64, to: f64) -> f64 {
    let delta = (to - from).rem_euclid(std::f64::consts::TAU);
    if delta >= std::f64::consts::PI {
        delta - std::f64::consts::TAU
    } else {
        delta
    }
}

fn nanoseconds_to_seconds(nanoseconds: u64) -> f64 {
    nanoseconds as f64 / NANOSECONDS_PER_SECOND
}

fn push_bounded<T>(deque: &mut VecDeque<T>, capacity: NonZeroUsize, value: T) {
    if deque.len() == capacity.get() {
        deque.pop_front();
    }
    deque.push_back(value);
}

fn bracket(
    history: &VecDeque<PoseHistoryEntry>,
    timestamp: DeviceTimestamp,
) -> Option<(PoseHistoryEntry, PoseHistoryEntry)> {
    history
        .iter()
        .zip(history.iter().skip(1))
        .find(|(before, after)| before.timestamp < timestamp && timestamp < after.timestamp)
        .map(|(before, after)| (*before, *after))
}

fn interpolate_planar(
    before: BaseToOdom,
    after: BaseToOdom,
    fraction: f64,
) -> Result<BaseToOdom, OdometryError> {
    let inverse_fraction = 1.0 - fraction;
    let x = fraction.mul_add(
        after.source_origin_x_in_destination_m(),
        inverse_fraction * before.source_origin_x_in_destination_m(),
    );
    let y = fraction.mul_add(
        after.source_origin_y_in_destination_m(),
        inverse_fraction * before.source_origin_y_in_destination_m(),
    );
    let yaw = signed_yaw_delta(
        before.source_yaw_in_destination_rad(),
        after.source_yaw_in_destination_rad(),
    )
    .mul_add(fraction, before.source_yaw_in_destination_rad());
    BaseToOdom::try_new(x, y, yaw).map_err(Into::into)
}

#[derive(Clone, Copy, Debug)]
struct GyroIntegration {
    delta_yaw_rad: f64,
    from: DeviceTimestamp,
    through: DeviceTimestamp,
    sample_from: DeviceTimestamp,
    sample_through: DeviceTimestamp,
    latest_sample_host_arrival: HostMonotonicTimestamp,
    endpoint_yaw_rate_rad_per_sec: f64,
    minimum_accuracy: SensorAccuracy,
}

#[derive(Clone, Copy, Debug)]
struct GyroRateAt {
    yaw_rate_rad_per_sec: f64,
    sample_from: DeviceTimestamp,
    sample_through: DeviceTimestamp,
    latest_sample_host_arrival: HostMonotonicTimestamp,
    minimum_accuracy: SensorAccuracy,
}

fn integrate_yaw(
    history: &VecDeque<GyroHistorySample>,
    start: DeviceTimestamp,
    end: DeviceTimestamp,
    max_gap_ns: u64,
) -> Result<GyroIntegration, OdometryUnavailable> {
    if end <= start || history.is_empty() {
        return Err(OdometryUnavailable::NoGyroCoverage);
    }
    let start_rate = rate_at(history, start, max_gap_ns)?;
    let end_rate = rate_at(history, end, max_gap_ns)?;
    let mut previous_timestamp = start;
    let mut previous_rate = start_rate.yaw_rate_rad_per_sec;
    let mut sum = 0.0_f64;
    let mut compensation = 0.0_f64;
    let mut sample_from = start_rate.sample_from.min(end_rate.sample_from);
    let mut sample_through = start_rate.sample_through.max(end_rate.sample_through);
    let mut latest_sample_host_arrival = start_rate
        .latest_sample_host_arrival
        .max(end_rate.latest_sample_host_arrival);
    let mut minimum_accuracy =
        lower_accuracy(start_rate.minimum_accuracy, end_rate.minimum_accuracy);
    for sample in history
        .iter()
        .filter(|sample| start < sample.timestamp && sample.timestamp < end)
    {
        trapezoid_add(
            &mut sum,
            &mut compensation,
            previous_timestamp,
            sample.timestamp,
            previous_rate,
            sample.yaw_rate_rad_per_sec,
            max_gap_ns,
        )?;
        previous_timestamp = sample.timestamp;
        previous_rate = sample.yaw_rate_rad_per_sec;
        sample_from = sample_from.min(sample.timestamp);
        sample_through = sample_through.max(sample.timestamp);
        latest_sample_host_arrival = latest_sample_host_arrival.max(sample.host_arrival);
        minimum_accuracy = lower_accuracy(minimum_accuracy, sample.accuracy);
    }
    trapezoid_add(
        &mut sum,
        &mut compensation,
        previous_timestamp,
        end,
        previous_rate,
        end_rate.yaw_rate_rad_per_sec,
        max_gap_ns,
    )?;
    if !sum.is_finite() {
        return Err(OdometryUnavailable::NoGyroCoverage);
    }
    Ok(GyroIntegration {
        delta_yaw_rad: sum,
        from: start,
        through: end,
        sample_from,
        sample_through,
        latest_sample_host_arrival,
        endpoint_yaw_rate_rad_per_sec: end_rate.yaw_rate_rad_per_sec,
        minimum_accuracy,
    })
}

fn rate_at(
    history: &VecDeque<GyroHistorySample>,
    timestamp: DeviceTimestamp,
    max_gap_ns: u64,
) -> Result<GyroRateAt, OdometryUnavailable> {
    if let Some(exact) = history.iter().find(|sample| sample.timestamp == timestamp) {
        return Ok(GyroRateAt {
            yaw_rate_rad_per_sec: exact.yaw_rate_rad_per_sec,
            sample_from: exact.timestamp,
            sample_through: exact.timestamp,
            latest_sample_host_arrival: exact.host_arrival,
            minimum_accuracy: exact.accuracy,
        });
    }
    if let Some((before, after)) = history
        .iter()
        .zip(history.iter().skip(1))
        .find(|(before, after)| before.timestamp < timestamp && timestamp < after.timestamp)
    {
        let gap_ns = after.timestamp.as_nanos() - before.timestamp.as_nanos();
        if gap_ns > max_gap_ns {
            return Err(OdometryUnavailable::GyroGap {
                gap_ns,
                maximum_ns: max_gap_ns,
            });
        }
        let offset_ns = timestamp.as_nanos() - before.timestamp.as_nanos();
        let fraction = offset_ns as f64 / gap_ns as f64;
        return Ok(GyroRateAt {
            yaw_rate_rad_per_sec: fraction.mul_add(
                after.yaw_rate_rad_per_sec,
                (1.0 - fraction) * before.yaw_rate_rad_per_sec,
            ),
            sample_from: before.timestamp,
            sample_through: after.timestamp,
            latest_sample_host_arrival: after.host_arrival,
            minimum_accuracy: lower_accuracy(before.accuracy, after.accuracy),
        });
    }
    let first = history.front().ok_or(OdometryUnavailable::NoGyroCoverage)?;
    if timestamp < first.timestamp {
        return Err(OdometryUnavailable::NoGyroCoverage);
    }
    let last = history.back().ok_or(OdometryUnavailable::NoGyroCoverage)?;
    let gap_ns = timestamp.as_nanos() - last.timestamp.as_nanos();
    if gap_ns > max_gap_ns {
        return Err(OdometryUnavailable::GyroGap {
            gap_ns,
            maximum_ns: max_gap_ns,
        });
    }
    Ok(GyroRateAt {
        yaw_rate_rad_per_sec: last.yaw_rate_rad_per_sec,
        sample_from: last.timestamp,
        sample_through: last.timestamp,
        latest_sample_host_arrival: last.host_arrival,
        minimum_accuracy: last.accuracy,
    })
}

#[allow(clippy::too_many_arguments)]
fn trapezoid_add(
    sum: &mut f64,
    compensation: &mut f64,
    from: DeviceTimestamp,
    to: DeviceTimestamp,
    from_rate: f64,
    to_rate: f64,
    max_gap_ns: u64,
) -> Result<(), OdometryUnavailable> {
    let gap_ns = to.as_nanos() - from.as_nanos();
    if gap_ns > max_gap_ns {
        return Err(OdometryUnavailable::GyroGap {
            gap_ns,
            maximum_ns: max_gap_ns,
        });
    }
    // Average the endpoints before multiplying by time without first summing
    // them, so equal large finite rates do not overflow unnecessarily.
    let average_rate = from_rate.mul_add(0.5, to_rate * 0.5);
    let term = average_rate * nanoseconds_to_seconds(gap_ns);
    let corrected = term - *compensation;
    let next = *sum + corrected;
    *compensation = (next - *sum) - corrected;
    *sum = next;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::{ImageSize, SlamMap};
    use crate::{
        AccelSample, DequeueSequence, FrameId, GyroSample, Keypoint, Pose, SensorAccuracy,
        WorldToCamera,
    };

    const IDENTITY_3: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    fn session(raw: u64) -> DeviceSessionId {
        DeviceSessionId::try_new(raw).expect("nonzero test session")
    }

    fn segment(raw: u64) -> OdomSegmentId {
        OdomSegmentId::try_new(raw).expect("nonzero test segment")
    }

    fn device(nanos: u64) -> DeviceTimestamp {
        DeviceTimestamp::try_from_nanos(i64::try_from(nanos).expect("test timestamp fits i64"))
            .expect("nonnegative test timestamp")
    }

    fn host(nanos: u64) -> HostMonotonicTimestamp {
        HostMonotonicTimestamp::from_nanos(nanos)
    }

    fn stamp(frame: u64, nanos: i64) -> VisualFrameStamp {
        VisualFrameStamp::new(FrameId::new(frame), Timestamp::from_nanos(nanos))
    }

    fn explicit_calibration(rotation: [[f64; 3]; 3]) -> RawImuCalibrationDto {
        RawImuCalibrationDto {
            format_version: 1,
            source_id: "fixture://bench-imu".to_owned(),
            content_id: "sha256:fixture-calibration-v1".to_owned(),
            gyro_affine: IDENTITY_3,
            gyro_bias_native_rad_per_sec: [0.0; 3],
            accel_affine: IDENTITY_3,
            accel_bias_native_m_per_sec2: [0.0; 3],
            native_imu_to_base_rotation: rotation,
        }
    }

    fn config_dto_with_frames(
        tracking_to_base: Pose,
        world_to_occupancy: WorldToOccupancy,
    ) -> PlanarOdometryConfigDto {
        PlanarOdometryConfigDto {
            raw_imu_calibration: explicit_calibration(IDENTITY_3),
            tracking_camera_to_base: TrackingCameraToBase::new(tracking_to_base),
            world_to_occupancy,
            max_visual_interval: Duration::from_secs(2),
            max_visual_linear_speed_m_per_sec: 10.0,
            max_visual_yaw_rate_rad_per_sec: 5.0,
            max_calibrated_yaw_rate_rad_per_sec: 5.0,
            minimum_gyro_accuracy: SensorAccuracy::Low,
            max_vertical_increment_m: 0.1,
            max_relative_roll_pitch_increment_rad: 0.2,
            max_absolute_map_roll_pitch_rad: 0.2,
            max_imu_gap: Duration::from_millis(200),
            max_prediction_age: Duration::from_millis(500),
            max_host_observation_age: Duration::from_secs(1),
            max_history_bracket_gap: Duration::from_secs(2),
            gyro_history_capacity: 16,
            pose_history_capacity: 8,
        }
    }

    fn config_with_frames(
        tracking_to_base: Pose,
        world_to_occupancy: WorldToOccupancy,
    ) -> PlanarOdometryConfig {
        PlanarOdometryConfig::parse(config_dto_with_frames(tracking_to_base, world_to_occupancy))
            .expect("valid test odometry configuration")
    }

    fn config_dto() -> PlanarOdometryConfigDto {
        config_dto_with_frames(
            Pose::identity(),
            WorldToOccupancy::try_new(IDENTITY_3, [0.0; 3]).expect("identity occupancy frame"),
        )
    }

    fn config() -> PlanarOdometryConfig {
        config_with_frames(
            Pose::identity(),
            WorldToOccupancy::try_new(IDENTITY_3, [0.0; 3]).expect("identity occupancy frame"),
        )
    }

    fn pose64(rotation: [[f64; 3]; 3], translation: [f64; 3]) -> Pose64 {
        Pose64::from_rt(rotation, translation).expect("valid test rigid transform")
    }

    fn yaw_rotation(yaw: f64) -> [[f64; 3]; 3] {
        let (sin, cos) = yaw.sin_cos();
        [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]]
    }

    fn roll_rotation(roll: f64) -> [[f64; 3]; 3] {
        let (sin, cos) = roll.sin_cos();
        [[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]]
    }

    fn pitch_rotation(pitch: f64) -> [[f64; 3]; 3] {
        let (sin, cos) = pitch.sin_cos();
        [[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]]
    }

    fn world_to_camera(pose: Pose64) -> WorldToCamera {
        WorldToCamera::from_legacy_pose(pose.try_to_pose32().expect("test pose is representable"))
    }

    fn localization(
        stamp: VisualFrameStamp,
        snapshot: MapSnapshot,
        base_origin_in_map: [f64; 3],
        base_yaw_in_map: f64,
    ) -> MapLocalization {
        let base_to_map = pose64(yaw_rotation(base_yaw_in_map), base_origin_in_map);
        localization_from_base_to_map(stamp, snapshot, base_to_map)
    }

    fn localization_from_base_to_map(
        stamp: VisualFrameStamp,
        snapshot: MapSnapshot,
        base_to_map: Pose64,
    ) -> MapLocalization {
        MapLocalization::new(
            stamp,
            snapshot,
            world_to_camera(base_to_map.try_inverse().expect("invert test pose")),
        )
    }

    fn increment_from_base_delta(
        from: VisualFrameStamp,
        to: VisualFrameStamp,
        snapshot: MapSnapshot,
        delta: Pose64,
    ) -> VisualIncrement {
        VisualIncrement::try_from_world_to_camera_poses(
            from,
            to,
            WorldToCamera::identity(),
            world_to_camera(delta.try_inverse().expect("invert test delta")),
            VisualIncrementBasis::CoOptimized {
                map_snapshot: snapshot,
            },
        )
        .expect("valid test visual increment")
    }

    fn map_revision_pair() -> (MapSnapshot, MapSnapshot) {
        let mut map = SlamMap::new();
        let before = map.snapshot();
        add_test_keyframe(&mut map, 99, 0);
        (before, map.snapshot())
    }

    fn add_test_keyframe(map: &mut SlamMap, frame: u64, timestamp_ns: i64) {
        map.add_keyframe(
            FrameId::new(frame),
            Timestamp::from_nanos(timestamp_ns),
            WorldToCamera::identity(),
            ImageSize::try_new(2, 2).expect("nonzero image"),
            vec![Keypoint { x: 0.5, y: 0.5 }],
        )
        .expect("test map mutation");
    }

    fn imu_report(
        session_id: DeviceSessionId,
        sequence: u32,
        timestamp_ns: u64,
        host_ns: u64,
        gyro: [f64; 3],
        accel: [f64; 3],
    ) -> ImuReport {
        imu_report_with_metadata(
            session_id,
            sequence,
            timestamp_ns,
            timestamp_ns,
            host_ns,
            gyro,
            accel,
            SensorAccuracy::High,
            SensorAccuracy::High,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn imu_report_with_metadata(
        session_id: DeviceSessionId,
        sequence: u32,
        accel_timestamp_ns: u64,
        gyro_timestamp_ns: u64,
        host_ns: u64,
        gyro: [f64; 3],
        accel: [f64; 3],
        gyro_accuracy: SensorAccuracy,
        accel_accuracy: SensorAccuracy,
    ) -> ImuReport {
        ImuReport::new(
            session_id,
            DequeueSequence::new(sequence),
            host(host_ns),
            AccelSample::new(
                device(accel_timestamp_ns),
                OakImuAcceleration::try_new(accel[0], accel[1], accel[2])
                    .expect("finite test accel"),
                accel_accuracy,
            ),
            GyroSample::new(
                device(gyro_timestamp_ns),
                OakImuAngularVelocity::try_new(gyro[0], gyro[1], gyro[2])
                    .expect("finite test gyro"),
                gyro_accuracy,
            ),
        )
    }

    fn anchored(snapshot: MapSnapshot) -> PlanarOdometry {
        anchored_with_config(snapshot, config())
    }

    fn anchored_with_config(snapshot: MapSnapshot, config: PlanarOdometryConfig) -> PlanarOdometry {
        let mut odometry = PlanarOdometry::new(config);
        odometry
            .reanchor(
                segment(1),
                session(1),
                localization(stamp(1, 0), snapshot, [0.0; 3], 0.0),
                host(0),
                host(0),
            )
            .expect("anchor test odometry");
        odometry
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected:.12}, got {actual:.12}"
        );
    }

    #[test]
    fn raw_imu_calibration_is_explicit_versioned_and_axis_correct() {
        let rotation = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let mut dto = explicit_calibration(rotation);
        dto.gyro_affine = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]];
        dto.gyro_bias_native_rad_per_sec = [1.0, 0.0, 0.0];
        let calibration = RawImuCalibration::parse(dto).expect("valid explicit calibration");
        let calibrated = calibration
            .calibrate_angular_velocity(
                OakImuAngularVelocity::try_new(2.0, 2.0, 2.0).expect("finite raw gyro"),
            )
            .expect("finite calibrated gyro")
            .as_array();
        assert_eq!(calibrated, [-6.0, 2.0, 8.0]);
        assert_eq!(calibration.provenance().format_version(), 1);
        assert_eq!(calibration.provenance().source_id(), "fixture://bench-imu");
        assert_eq!(
            calibration.provenance().content_id(),
            "sha256:fixture-calibration-v1"
        );
    }

    #[test]
    fn calibration_rejects_unknown_version_missing_provenance_and_invalid_matrices() {
        let mut unknown_version = explicit_calibration(IDENTITY_3);
        unknown_version.format_version = 2;
        assert!(matches!(
            RawImuCalibration::parse(unknown_version),
            Err(RawImuCalibrationError::UnsupportedFormatVersion {
                actual: 2,
                supported: 1,
            })
        ));

        let mut empty_source = explicit_calibration(IDENTITY_3);
        empty_source.source_id = "  ".to_owned();
        assert!(matches!(
            RawImuCalibration::parse(empty_source),
            Err(RawImuCalibrationError::EmptySourceId)
        ));

        let mut singular = explicit_calibration(IDENTITY_3);
        singular.accel_affine[2] = [0.0; 3];
        assert!(matches!(
            RawImuCalibration::parse(singular),
            Err(RawImuCalibrationError::SingularAffine {
                matrix: CalibrationMatrix::AccelerometerAffine,
                ..
            })
        ));

        let reflection = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(matches!(
            RawImuCalibration::parse(explicit_calibration(reflection)),
            Err(RawImuCalibrationError::ImproperRotation { .. })
        ));
    }

    #[test]
    fn calibration_rejects_unbounded_asserted_provenance_ids() {
        let mut source_too_long = explicit_calibration(IDENTITY_3);
        source_too_long.source_id = "s".repeat(RawImuCalibration::MAX_SOURCE_ID_BYTES + 1);
        assert!(matches!(
            RawImuCalibration::parse(source_too_long),
            Err(RawImuCalibrationError::SourceIdTooLong {
                actual_bytes,
                maximum_bytes,
            }) if actual_bytes == RawImuCalibration::MAX_SOURCE_ID_BYTES + 1
                && maximum_bytes == RawImuCalibration::MAX_SOURCE_ID_BYTES
        ));

        let mut content_too_long = explicit_calibration(IDENTITY_3);
        content_too_long.content_id = "c".repeat(RawImuCalibration::MAX_CONTENT_ID_BYTES + 1);
        assert!(matches!(
            RawImuCalibration::parse(content_too_long),
            Err(RawImuCalibrationError::ContentIdTooLong {
                actual_bytes,
                maximum_bytes,
            }) if actual_bytes == RawImuCalibration::MAX_CONTENT_ID_BYTES + 1
                && maximum_bytes == RawImuCalibration::MAX_CONTENT_ID_BYTES
        ));
    }

    #[test]
    fn calibration_rejects_nonfinite_determinant_arithmetic() {
        let mut dto = explicit_calibration(IDENTITY_3);
        dto.gyro_affine = [[f64::MAX; 3]; 3];
        assert!(matches!(
            RawImuCalibration::parse(dto),
            Err(RawImuCalibrationError::SingularAffine {
                matrix: CalibrationMatrix::GyroscopeAffine,
                determinant,
            }) if !determinant.is_finite()
        ));
    }

    #[test]
    fn visual_transform_formula_has_correct_translation_and_yaw_signs() {
        let snapshot = SlamMap::new().snapshot();
        for (index, (x, y, yaw)) in [
            (1.0, 0.0, 0.0),
            (0.0, -0.5, 0.0),
            (0.2, 0.3, 0.7),
            (-0.4, 0.1, -0.9),
        ]
        .into_iter()
        .enumerate()
        {
            let mut odometry = anchored(snapshot);
            let from = stamp(1, 0);
            let to = stamp(2, 1_000_000_000);
            let delta = pose64(yaw_rotation(yaw), [x, y, 0.0]);
            let result = odometry
                .observe_visual(
                    session(1),
                    increment_from_base_delta(from, to, snapshot, delta),
                    localization(to, snapshot, [x, y, 0.0], yaw),
                    host(100 + index as u64),
                    host(100 + index as u64),
                )
                .expect("planar visual increment");
            assert_close(
                result.base_to_odom().source_origin_x_in_destination_m(),
                x,
                1.0e-6,
            );
            assert_close(
                result.base_to_odom().source_origin_y_in_destination_m(),
                y,
                1.0e-6,
            );
            assert_close(
                result.base_to_odom().source_yaw_in_destination_rad(),
                yaw,
                1.0e-6,
            );
        }
    }

    #[test]
    fn camera_to_base_extrinsic_converts_optical_forward_to_base_forward() {
        let optical_to_base = [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]];
        let tracking_to_base = pose64(optical_to_base, [0.0; 3]);
        let world_to_occupancy =
            WorldToOccupancy::try_new(optical_to_base, [0.0; 3]).expect("proper rotation");
        let mut odometry = PlanarOdometry::new(config_with_frames(
            tracking_to_base
                .try_to_pose32()
                .expect("extrinsic representable"),
            world_to_occupancy,
        ));
        let snapshot = SlamMap::new().snapshot();
        let from = stamp(1, 0);
        odometry
            .reanchor(
                segment(1),
                session(1),
                MapLocalization::new(from, snapshot, WorldToCamera::identity()),
                host(0),
                host(0),
            )
            .expect("anchor with optical extrinsic");

        let desired_base_delta = pose64(IDENTITY_3, [1.0, 0.0, 0.0]);
        let base_to_tracking = tracking_to_base.try_inverse().expect("invert extrinsic");
        let visual_d = base_to_tracking
            .try_compose(
                desired_base_delta
                    .try_inverse()
                    .expect("invert desired delta"),
            )
            .expect("compose camera delta")
            .try_compose(tracking_to_base)
            .expect("compose camera delta");
        let to = stamp(2, 1_000_000_000);
        let increment = VisualIncrement::try_from_world_to_camera_poses(
            from,
            to,
            WorldToCamera::identity(),
            world_to_camera(visual_d),
            VisualIncrementBasis::CoOptimized {
                map_snapshot: snapshot,
            },
        )
        .expect("camera-frame visual increment");
        let result = odometry
            .observe_visual(
                session(1),
                increment,
                MapLocalization::new(to, snapshot, world_to_camera(visual_d)),
                host(1),
                host(1),
            )
            .expect("extrinsic-adjusted visual update");
        assert_close(
            result.base_to_odom().source_origin_x_in_destination_m(),
            1.0,
            1.0e-6,
        );
        assert_close(
            result.base_to_odom().source_origin_y_in_destination_m(),
            0.0,
            1.0e-6,
        );
    }

    #[test]
    fn map_correction_changes_only_map_from_odom() {
        let (before, after) = map_revision_pair();
        let visual_stamp = stamp(1, 0);
        let mut odometry = PlanarOdometry::new(config());
        let initial = odometry
            .reanchor(
                segment(1),
                session(1),
                localization(visual_stamp, before, [2.0, 3.0, 0.0], 0.1),
                host(0),
                host(0),
            )
            .expect("initial map anchor");
        let corrected = odometry
            .observe_map_localization(
                session(1),
                localization(visual_stamp, after, [5.0, -1.0, 0.0], -0.2),
            )
            .expect("correction on same map instance");
        assert_eq!(corrected.base_to_odom(), initial.base_to_odom());
        assert_eq!(corrected.twist(), initial.twist());
        assert_ne!(corrected.odom_to_map(), initial.odom_to_map());
        assert_eq!(corrected.map_snapshot(), after);
    }

    #[test]
    fn higher_generation_from_divergent_map_branch_is_rejected_transactionally() {
        let root = SlamMap::new();
        let mut accepted_branch = root.clone();
        add_test_keyframe(&mut accepted_branch, 10, 0);
        let accepted_snapshot = accepted_branch.snapshot();

        let mut divergent_branch = root.clone();
        add_test_keyframe(&mut divergent_branch, 20, 0);
        add_test_keyframe(&mut divergent_branch, 21, 1);
        let divergent_snapshot = divergent_branch.snapshot();
        assert!(divergent_snapshot.generation() > accepted_snapshot.generation());

        let visual_stamp = stamp(1, 0);
        let mut odometry = PlanarOdometry::new(config());
        let initial = odometry
            .reanchor(
                segment(1),
                session(1),
                localization(visual_stamp, accepted_snapshot, [1.0, 2.0, 0.0], 0.1),
                host(0),
                host(0),
            )
            .expect("anchor accepted branch");
        assert!(matches!(
            odometry.observe_map_localization(
                session(1),
                localization(visual_stamp, divergent_snapshot, [5.0, 6.0, 0.0], -0.2),
            ),
            Err(OdometryError::DivergentMapRevision)
        ));
        let retained = odometry.current().expect("state retained after rejection");
        assert_eq!(retained.map_snapshot(), accepted_snapshot);
        assert_eq!(retained.base_to_odom(), initial.base_to_odom());
        assert_eq!(retained.odom_to_map(), initial.odom_to_map());
    }

    #[test]
    fn visual_increment_basis_must_be_an_accepted_ancestor_revision() {
        let root = SlamMap::new();
        let mut accepted_branch = root.clone();
        add_test_keyframe(&mut accepted_branch, 10, 0);
        let accepted_snapshot = accepted_branch.snapshot();

        let mut divergent_branch = root.clone();
        add_test_keyframe(&mut divergent_branch, 20, 0);
        let divergent_snapshot = divergent_branch.snapshot();
        assert_eq!(
            divergent_snapshot.generation(),
            accepted_snapshot.generation()
        );

        let mut future_branch = accepted_branch;
        add_test_keyframe(&mut future_branch, 11, 1);
        let future_snapshot = future_branch.snapshot();
        let unrelated_snapshot = SlamMap::new().snapshot();

        let mut odometry = anchored(accepted_snapshot);
        let before = odometry.current().expect("anchored state").clone();
        let from = stamp(1, 0);
        let to = stamp(2, 100_000_000);
        for (basis, expected) in [
            (divergent_snapshot, "divergent"),
            (future_snapshot, "future"),
            (unrelated_snapshot, "instance"),
        ] {
            let result = odometry.observe_visual(
                session(1),
                increment_from_base_delta(from, to, basis, pose64(IDENTITY_3, [0.01, 0.0, 0.0])),
                localization(to, accepted_snapshot, [0.01, 0.0, 0.0], 0.0),
                host(1),
                host(1),
            );
            assert!(
                matches!(
                    (&result, expected),
                    (
                        Err(OdometryError::DivergentBasisMapRevision { .. }),
                        "divergent"
                    ) | (Err(OdometryError::BasisMapRevisionAhead { .. }), "future")
                        | (
                            Err(OdometryError::BasisMapInstanceMismatch { .. }),
                            "instance"
                        )
                ),
                "unexpected {expected} basis result: {result:?}"
            );
            let retained = odometry.current().expect("state retained after rejection");
            assert_eq!(retained.map_snapshot(), before.map_snapshot());
            assert_eq!(retained.base_to_odom(), before.base_to_odom());
            assert_eq!(retained.odom_to_map(), before.odom_to_map());
        }

        let accepted = odometry
            .observe_visual(
                session(1),
                increment_from_base_delta(
                    from,
                    to,
                    accepted_snapshot,
                    pose64(IDENTITY_3, [0.01, 0.0, 0.0]),
                ),
                localization(to, future_snapshot, [0.01, 0.0, 0.0], 0.0),
                host(1),
                host(1),
            )
            .expect("older same-lineage basis remains valid after a map correction");
        assert_eq!(accepted.map_snapshot(), future_snapshot);
    }

    #[test]
    fn session_map_and_stamp_discontinuities_require_explicit_new_segment() {
        let first_map = SlamMap::new().snapshot();
        let second_map = SlamMap::new().snapshot();
        let mut odometry = anchored(first_map);
        let from = stamp(1, 0);
        let to = stamp(2, 1_000_000_000);
        let increment =
            increment_from_base_delta(from, to, first_map, pose64(IDENTITY_3, [0.1, 0.0, 0.0]));
        assert!(matches!(
            odometry.observe_visual(
                session(2),
                increment,
                localization(to, first_map, [0.1, 0.0, 0.0], 0.0),
                host(1),
                host(1),
            ),
            Err(OdometryError::ReanchorRequired(
                ReanchorReason::DeviceSessionChanged { .. }
            ))
        ));
        assert!(matches!(
            odometry.observe_visual(
                session(1),
                increment,
                localization(to, second_map, [0.1, 0.0, 0.0], 0.0),
                host(1),
                host(1),
            ),
            Err(OdometryError::ReanchorRequired(
                ReanchorReason::MapInstanceChanged
            ))
        ));

        let discontinuous = increment_from_base_delta(
            stamp(9, 0),
            stamp(10, 1_000_000_000),
            first_map,
            pose64(IDENTITY_3, [0.1, 0.0, 0.0]),
        );
        assert!(matches!(
            odometry.observe_visual(
                session(1),
                discontinuous,
                localization(stamp(10, 1_000_000_000), first_map, [0.1, 0.0, 0.0], 0.0,),
                host(1),
                host(1),
            ),
            Err(OdometryError::ReanchorRequired(
                ReanchorReason::VisualStampDiscontinuity { .. }
            ))
        ));

        let reanchored = odometry
            .reanchor(
                segment(2),
                session(2),
                localization(stamp(20, 0), second_map, [7.0, 0.0, 0.0], 0.0),
                host(2),
                host(2),
            )
            .expect("explicit new segment");
        assert_eq!(reanchored.segment_id(), segment(2));
        assert_close(
            reanchored.base_to_odom().source_origin_x_in_destination_m(),
            0.0,
            0.0,
        );
    }

    #[test]
    fn negative_visual_timestamp_is_rejected_before_state_mutation() {
        let snapshot = SlamMap::new().snapshot();
        let mut odometry = PlanarOdometry::new(config());
        let result = odometry.reanchor(
            segment(1),
            session(1),
            localization(stamp(1, -1), snapshot, [0.0; 3], 0.0),
            host(0),
            host(0),
        );
        assert!(matches!(
            result,
            Err(OdometryError::VisualTimestamp(
                InertialValueError::NegativeDeviceTimestamp { nanos: -1 }
            ))
        ));
        assert!(odometry.current().is_none());
    }

    #[test]
    fn planarity_rejection_is_transactional_and_retryable() {
        let snapshot = SlamMap::new().snapshot();
        let mut odometry = anchored(snapshot);
        let before = odometry.current().expect("anchored state").clone();
        let from = stamp(1, 0);
        let to = stamp(2, 1_000_000_000);
        let nonplanar =
            increment_from_base_delta(from, to, snapshot, pose64(IDENTITY_3, [0.1, 0.0, 0.2]));
        assert!(matches!(
            odometry.observe_visual(
                session(1),
                nonplanar,
                localization(to, snapshot, [0.1, 0.0, 0.0], 0.0),
                host(1),
                host(1),
            ),
            Err(OdometryError::PlanarityViolation {
                component: PlanarityComponent::VerticalTranslation,
                ..
            })
        ));
        assert_eq!(
            odometry.current().expect("state retained").base_to_odom(),
            before.base_to_odom()
        );

        let valid =
            increment_from_base_delta(from, to, snapshot, pose64(IDENTITY_3, [0.1, 0.0, 0.0]));
        let accepted = odometry
            .observe_visual(
                session(1),
                valid,
                localization(to, snapshot, [0.1, 0.0, 0.0], 0.0),
                host(1),
                host(1),
            )
            .expect("valid retry after rejection");
        assert_close(
            accepted.base_to_odom().source_origin_x_in_destination_m(),
            0.1,
            1.0e-6,
        );
    }

    #[test]
    fn roll_and_pitch_increments_are_rejected_independently() {
        let snapshot = SlamMap::new().snapshot();
        let from = stamp(1, 0);
        let to = stamp(2, 1_000_000_000);
        for (rotation, expected) in [
            (roll_rotation(0.3), PlanarityComponent::Roll),
            (pitch_rotation(0.3), PlanarityComponent::Pitch),
        ] {
            let mut odometry = anchored(snapshot);
            let delta = pose64(rotation, [0.0; 3]);
            assert!(matches!(
                odometry.observe_visual(
                    session(1),
                    increment_from_base_delta(from, to, snapshot, delta),
                    localization(to, snapshot, [0.0; 3], 0.0),
                    host(1),
                    host(1),
                ),
                Err(OdometryError::PlanarityViolation { component, .. }) if component == expected
            ));
        }
    }

    #[test]
    fn relative_increment_and_absolute_map_tilt_use_separate_limits() {
        let snapshot = SlamMap::new().snapshot();
        let visual_stamp = stamp(1, 0);

        let mut permissive_map_dto = config_dto();
        permissive_map_dto.max_relative_roll_pitch_increment_rad = 0.1;
        permissive_map_dto.max_absolute_map_roll_pitch_rad = 0.3;
        let permissive_map_config =
            PlanarOdometryConfig::parse(permissive_map_dto).expect("separate planarity limits");
        let mut map_tilt_odometry = PlanarOdometry::new(permissive_map_config.clone());
        map_tilt_odometry
            .reanchor(
                segment(1),
                session(1),
                localization_from_base_to_map(
                    visual_stamp,
                    snapshot,
                    pose64(roll_rotation(0.2), [0.0; 3]),
                ),
                host(0),
                host(0),
            )
            .expect("absolute map tilt below its independent limit");

        let mut increment_odometry = anchored_with_config(snapshot, permissive_map_config);
        assert!(matches!(
            increment_odometry.observe_visual(
                session(1),
                increment_from_base_delta(
                    visual_stamp,
                    stamp(2, 1_000_000_000),
                    snapshot,
                    pose64(roll_rotation(0.2), [0.0; 3]),
                ),
                localization(stamp(2, 1_000_000_000), snapshot, [0.0; 3], 0.0,),
                host(1),
                host(1),
            ),
            Err(OdometryError::PlanarityViolation {
                component: PlanarityComponent::Roll,
                maximum: 0.1,
                ..
            })
        ));

        let mut strict_map_dto = config_dto();
        strict_map_dto.max_relative_roll_pitch_increment_rad = 0.3;
        strict_map_dto.max_absolute_map_roll_pitch_rad = 0.1;
        let mut strict_map_odometry = PlanarOdometry::new(
            PlanarOdometryConfig::parse(strict_map_dto).expect("strict absolute map limit"),
        );
        assert!(matches!(
            strict_map_odometry.reanchor(
                segment(1),
                session(1),
                localization_from_base_to_map(
                    visual_stamp,
                    snapshot,
                    pose64(roll_rotation(0.2), [0.0; 3]),
                ),
                host(0),
                host(0),
            ),
            Err(OdometryError::PlanarityViolation {
                component: PlanarityComponent::MapRoll,
                maximum: 0.1,
                ..
            })
        ));
    }

    #[test]
    fn configuration_keeps_euler_planarity_limits_away_from_singularity() {
        let maximum_exclusive =
            std::f64::consts::FRAC_PI_2 - PlanarOdometryConfig::MIN_EULER_SINGULARITY_MARGIN_RAD;
        for parameter in [
            ScalarParameter::MaximumRelativeRollPitchIncrement,
            ScalarParameter::MaximumAbsoluteMapRollPitch,
        ] {
            let mut dto = config_dto();
            match parameter {
                ScalarParameter::MaximumRelativeRollPitchIncrement => {
                    dto.max_relative_roll_pitch_increment_rad = maximum_exclusive;
                }
                ScalarParameter::MaximumAbsoluteMapRollPitch => {
                    dto.max_absolute_map_roll_pitch_rad = maximum_exclusive;
                }
                _ => unreachable!("test iterates only roll/pitch parameters"),
            }
            assert!(matches!(
                PlanarOdometryConfig::parse(dto),
                Err(PlanarOdometryConfigError::RollPitchLimitOutOfRange {
                    parameter: actual,
                    value,
                    maximum_exclusive: maximum,
                }) if actual == parameter && value == maximum_exclusive
                    && maximum == maximum_exclusive
            ));
        }
    }

    #[test]
    fn calibrated_gyro_prediction_is_bounded_and_visibly_predicted() {
        let snapshot = SlamMap::new().snapshot();
        let mut odometry = anchored(snapshot);
        let to = stamp(2, 1_000_000_000);
        odometry
            .observe_visual(
                session(1),
                increment_from_base_delta(
                    stamp(1, 0),
                    to,
                    snapshot,
                    pose64(IDENTITY_3, [1.0, 0.0, 0.0]),
                ),
                localization(to, snapshot, [1.0, 0.0, 0.0], 0.0),
                host(10),
                host(10),
            )
            .expect("visual velocity source");
        odometry
            .observe_imu(
                imu_report(session(1), 0, 1_000_000_000, 10, [0.0, 0.0, 0.2], [0.0; 3]),
                host(10),
            )
            .expect("first gyro sample");
        odometry
            .observe_imu(
                imu_report(session(1), 1, 1_100_000_000, 20, [0.0, 0.0, 0.2], [0.0; 3]),
                host(20),
            )
            .expect("second gyro sample");
        let OdometryEstimate::Available(predicted) = odometry
            .estimate(session(1), device(1_100_000_000), host(20))
            .expect("prediction calculation")
        else {
            panic!("bounded prediction should be available")
        };
        assert_close(
            predicted.base_to_odom().source_origin_x_in_destination_m(),
            1.1,
            1.0e-9,
        );
        assert_close(
            predicted.base_to_odom().source_yaw_in_destination_rad(),
            0.02,
            1.0e-9,
        );
        assert!(matches!(
            predicted.quality(),
            OdometryQuality::Predicted {
                translation_model: PredictionTranslationModel::BoundedConstantVisualVelocity,
                ..
            }
        ));
    }

    #[test]
    fn trapezoidal_gyro_integration_is_exact_for_linear_rate_ramps() {
        let snapshot = SlamMap::new().snapshot();
        let mut odometry = anchored(snapshot);
        odometry
            .observe_imu(
                imu_report(session(1), 0, 0, 0, [0.0, 0.0, 0.0], [0.0; 3]),
                host(0),
            )
            .expect("ramp start");
        odometry
            .observe_imu(
                imu_report(session(1), 1, 100_000_000, 1, [0.0, 0.0, 1.0], [0.0; 3]),
                host(1),
            )
            .expect("ramp end");
        let OdometryEstimate::Available(predicted) = odometry
            .estimate(session(1), device(100_000_000), host(1))
            .expect("linear-ramp prediction")
        else {
            panic!("linear-ramp prediction should be available")
        };
        assert_close(
            predicted.base_to_odom().source_yaw_in_destination_rad(),
            0.05,
            1.0e-12,
        );
        assert_close(predicted.twist().yaw_rate_rad_per_sec(), 1.0, 1.0e-12);
        assert!(matches!(
            predicted.quality(),
            OdometryQuality::Predicted {
                integration_from,
                integration_through,
                gyro_sample_from,
                gyro_sample_through,
                endpoint_gyro_yaw_rate_rad_per_sec,
                minimum_gyro_accuracy: SensorAccuracy::High,
                ..
            } if *integration_from == device(0)
                && *integration_through == device(100_000_000)
                && *gyro_sample_from == device(0)
                && *gyro_sample_through == device(100_000_000)
                && (*endpoint_gyro_yaw_rate_rad_per_sec - 1.0).abs() <= 1.0e-12
        ));

        let OdometryEstimate::Available(extrapolated) = odometry
            .estimate(session(1), device(150_000_000), host(1))
            .expect("bounded held-rate prediction")
        else {
            panic!("held-rate prediction should be available")
        };
        assert_close(
            extrapolated.base_to_odom().source_yaw_in_destination_rad(),
            0.1,
            1.0e-12,
        );
        assert_close(extrapolated.twist().yaw_rate_rad_per_sec(), 1.0, 1.0e-12);
        assert!(matches!(
            extrapolated.quality(),
            OdometryQuality::Predicted {
                integration_through,
                gyro_sample_through,
                endpoint_gyro_yaw_rate_rad_per_sec,
                ..
            } if *integration_through == device(150_000_000)
                && *gyro_sample_through == device(100_000_000)
                && (*endpoint_gyro_yaw_rate_rad_per_sec - 1.0).abs() <= 1.0e-12
        ));
    }

    #[test]
    fn history_yaw_interpolation_always_takes_the_short_wrapped_arc() {
        let before =
            BaseToOdom::try_new(0.0, 0.0, std::f64::consts::PI - 0.1).expect("finite before pose");
        let after =
            BaseToOdom::try_new(0.0, 0.0, -std::f64::consts::PI + 0.1).expect("finite after pose");
        for (fraction, expected_magnitude_from_pi) in [(0.25, 0.05), (0.5, 0.0), (0.75, 0.05)] {
            let interpolated =
                interpolate_planar(before, after, fraction).expect("wrapped interpolation");
            let yaw = interpolated.source_yaw_in_destination_rad();
            assert_close(
                yaw.abs(),
                std::f64::consts::PI - expected_magnitude_from_pi,
                1.0e-12,
            );
        }
    }

    #[test]
    fn interpolation_and_gyro_quadrature_avoid_finite_intermediate_overflow() {
        let before = BaseToOdom::try_new(-f64::MAX, f64::MAX, 0.0).expect("finite before pose");
        let after = BaseToOdom::try_new(f64::MAX, -f64::MAX, 0.0).expect("finite after pose");
        let midpoint = interpolate_planar(before, after, 0.5).expect("finite convex midpoint");
        assert_eq!(midpoint.source_origin_in_destination_m(), [0.0, 0.0]);

        let history = VecDeque::from([
            GyroHistorySample {
                timestamp: device(0),
                host_arrival: host(0),
                yaw_rate_rad_per_sec: -f64::MAX,
                accuracy: SensorAccuracy::High,
            },
            GyroHistorySample {
                timestamp: device(2),
                host_arrival: host(2),
                yaw_rate_rad_per_sec: f64::MAX,
                accuracy: SensorAccuracy::High,
            },
        ]);
        let midpoint_rate = rate_at(&history, device(1), 2).expect("finite convex gyro rate");
        assert_eq!(midpoint_rate.yaw_rate_rad_per_sec, 0.0);

        let mut sum = 0.0;
        let mut compensation = 0.0;
        trapezoid_add(
            &mut sum,
            &mut compensation,
            device(0),
            device(1),
            f64::MAX,
            f64::MAX,
            1,
        )
        .expect("finite large-rate quadrature");
        assert!(sum.is_finite());
        assert_close(sum / f64::MAX, 1.0e-9, 1.0e-24);
    }

    #[test]
    fn prediction_expires_and_never_falls_back_to_latest_pose() {
        let snapshot = SlamMap::new().snapshot();
        let mut odometry = anchored(snapshot);
        odometry
            .observe_imu(imu_report(session(1), 0, 0, 0, [0.0; 3], [0.0; 3]), host(0))
            .expect("gyro at visual anchor");
        assert!(matches!(
            odometry
                .estimate(session(1), device(300_000_000), host(0))
                .expect("typed unavailable"),
            OdometryEstimate::Unavailable(OdometryUnavailable::GyroGap {
                gap_ns: 300_000_000,
                maximum_ns: 200_000_000,
            })
        ));
        assert!(matches!(
            odometry
                .pose_at(session(1), device(600_000_000), host(0))
                .expect("typed unavailable"),
            PoseHistoryQuery::Unavailable(OdometryUnavailable::PredictionExpired {
                age_ns: 600_000_000,
                maximum_ns: 500_000_000,
            })
        ));
    }

    #[test]
    fn imu_gap_resets_history_without_acceleration_translation() {
        let snapshot = SlamMap::new().snapshot();
        let mut odometry = anchored(snapshot);
        let first = odometry
            .observe_imu(
                imu_report(session(1), 0, 0, 0, [0.0, 0.0, 1.0], [2.0, 0.0, 0.0]),
                host(0),
            )
            .expect("first IMU report");
        assert!(!first.history_reset_for_gap);
        assert_eq!(
            first.translation_integration,
            TranslationIntegration::DisabledNoEncoderNoAccelerationIntegration
        );
        let after_gap = odometry
            .observe_imu(
                imu_report(
                    session(1),
                    1,
                    300_000_000,
                    1,
                    [0.0, 0.0, 1.0],
                    [2.0, 0.0, 0.0],
                ),
                host(1),
            )
            .expect("gap starts a new gyro history");
        assert!(after_gap.history_reset_for_gap);
        assert_eq!(
            odometry
                .current()
                .expect("visual state unchanged")
                .base_to_odom(),
            BaseToOdom::try_new(0.0, 0.0, 0.0).expect("zero odom pose")
        );
    }

    #[test]
    fn accelerometer_only_gap_does_not_clear_continuous_gyro_history() {
        let snapshot = SlamMap::new().snapshot();
        let mut odometry = anchored(snapshot);
        odometry
            .observe_imu(
                imu_report_with_metadata(
                    session(1),
                    0,
                    0,
                    0,
                    0,
                    [0.0, 0.0, 1.0],
                    [0.0; 3],
                    SensorAccuracy::High,
                    SensorAccuracy::High,
                ),
                host(0),
            )
            .expect("initial independent IMU timestamps");
        let update = odometry
            .observe_imu(
                imu_report_with_metadata(
                    session(1),
                    1,
                    300_000_000,
                    100_000_000,
                    1,
                    [0.0, 0.0, 1.0],
                    [0.0; 3],
                    SensorAccuracy::High,
                    SensorAccuracy::High,
                ),
                host(1),
            )
            .expect("accel gap must not poison gyro history");
        assert!(!update.history_reset_for_gap);
        let OdometryEstimate::Available(predicted) = odometry
            .estimate(session(1), device(100_000_000), host(1))
            .expect("continuous gyro prediction")
        else {
            panic!("continuous gyro history should remain available")
        };
        assert_close(
            predicted.base_to_odom().source_yaw_in_destination_rad(),
            0.1,
            1.0e-12,
        );
    }

    #[test]
    fn dequeue_sequence_gap_clears_gyro_history_even_with_bounded_device_gap() {
        let snapshot = SlamMap::new().snapshot();
        let mut odometry = anchored(snapshot);
        odometry
            .observe_imu(
                imu_report(session(1), 0, 0, 0, [0.0, 0.0, 1.0], [0.0; 3]),
                host(0),
            )
            .expect("initial gyro sample");
        let update = odometry
            .observe_imu(
                imu_report(session(1), 2, 100_000_000, 1, [0.0, 0.0, 1.0], [0.0; 3]),
                host(1),
            )
            .expect("sequence gap is accepted but resets integration history");
        assert!(matches!(
            update.order,
            InertialOrderOutcome::Gap { missing_reports: 1 }
        ));
        assert!(update.history_reset_for_gap);
        assert!(matches!(
            odometry
                .estimate(session(1), device(100_000_000), host(1))
                .expect("typed unavailable after reset"),
            OdometryEstimate::Unavailable(OdometryUnavailable::NoGyroCoverage)
        ));
    }

    #[test]
    fn gyro_accuracy_is_gated_preserved_and_transactional() {
        let snapshot = SlamMap::new().snapshot();
        let mut dto = config_dto();
        dto.minimum_gyro_accuracy = SensorAccuracy::Medium;
        let mut odometry = anchored_with_config(
            snapshot,
            PlanarOdometryConfig::parse(dto).expect("accuracy-gated config"),
        );
        let unreliable = imu_report_with_metadata(
            session(1),
            0,
            0,
            0,
            0,
            [0.0, 0.0, 0.2],
            [0.0; 3],
            SensorAccuracy::Unreliable,
            SensorAccuracy::Low,
        );
        assert!(matches!(
            odometry.observe_imu(unreliable, host(0)),
            Err(OdometryError::GyroAccuracyBelowMinimum {
                actual: SensorAccuracy::Unreliable,
                minimum: SensorAccuracy::Medium,
            })
        ));

        let accepted = odometry
            .observe_imu(
                imu_report_with_metadata(
                    session(1),
                    0,
                    0,
                    0,
                    0,
                    [0.0, 0.0, 0.2],
                    [0.0; 3],
                    SensorAccuracy::Medium,
                    SensorAccuracy::Low,
                ),
                host(0),
            )
            .expect("valid retry after rejected accuracy");
        assert_eq!(accepted.order, InertialOrderOutcome::FirstReport);
        assert_eq!(accepted.gyro_accuracy, SensorAccuracy::Medium);
        assert_eq!(accepted.accel_accuracy, SensorAccuracy::Low);
        odometry
            .observe_imu(
                imu_report_with_metadata(
                    session(1),
                    1,
                    100_000_000,
                    100_000_000,
                    1,
                    [0.0, 0.0, 0.2],
                    [0.0; 3],
                    SensorAccuracy::High,
                    SensorAccuracy::High,
                ),
                host(1),
            )
            .expect("second accepted accuracy");
        let OdometryEstimate::Available(predicted) = odometry
            .estimate(session(1), device(100_000_000), host(1))
            .expect("accuracy-provenanced prediction")
        else {
            panic!("prediction should be available")
        };
        assert!(matches!(
            predicted.quality(),
            OdometryQuality::Predicted {
                minimum_gyro_accuracy: SensorAccuracy::Medium,
                ..
            }
        ));
    }

    #[test]
    fn calibrated_yaw_rate_limit_rejects_before_order_or_history_mutation() {
        let snapshot = SlamMap::new().snapshot();
        let mut dto = config_dto();
        dto.max_calibrated_yaw_rate_rad_per_sec = 0.5;
        let mut odometry = anchored_with_config(
            snapshot,
            PlanarOdometryConfig::parse(dto).expect("bounded-rate config"),
        );
        assert!(matches!(
            odometry.observe_imu(
                imu_report(session(1), 0, 0, 0, [0.0, 0.0, 0.75], [0.0; 3]),
                host(0),
            ),
            Err(OdometryError::CalibratedYawRateTooLarge {
                rate_rad_per_sec: 0.75,
                maximum_rad_per_sec: 0.5,
            })
        ));
        let retry = odometry
            .observe_imu(
                imu_report(session(1), 0, 0, 0, [0.0, 0.0, 0.5], [0.0; 3]),
                host(0),
            )
            .expect("bounded retry after rate rejection");
        assert_eq!(retry.order, InertialOrderOutcome::FirstReport);
    }

    #[test]
    fn stale_imu_host_arrival_makes_prediction_unavailable() {
        let snapshot = SlamMap::new().snapshot();
        let mut odometry = anchored(snapshot);
        odometry
            .observe_imu(imu_report(session(1), 0, 0, 0, [0.0; 3], [0.0; 3]), host(0))
            .expect("fresh at ingestion");
        assert!(matches!(
            odometry
                .estimate(session(1), device(100_000_000), host(2_000_000_000))
                .expect("typed unavailable"),
            OdometryEstimate::Unavailable(OdometryUnavailable::SupportingImuHostArrivalStale {
                age_ns: 2_000_000_000,
                maximum_ns: 1_000_000_000,
            })
        ));
    }

    #[test]
    fn pose_history_interpolates_exact_capture_time_without_latest_fallback() {
        let snapshot = SlamMap::new().snapshot();
        let mut odometry = anchored(snapshot);
        let to = stamp(2, 1_000_000_000);
        odometry
            .observe_visual(
                session(1),
                increment_from_base_delta(
                    stamp(1, 0),
                    to,
                    snapshot,
                    pose64(yaw_rotation(0.4), [1.0, -0.5, 0.0]),
                ),
                localization(to, snapshot, [1.0, -0.5, 0.0], 0.4),
                host(1),
                host(1),
            )
            .expect("second visual pose");
        let PoseHistoryQuery::Available(aligned) = odometry
            .pose_at(session(1), device(500_000_000), host(1))
            .expect("history query")
        else {
            panic!("bracketed depth timestamp should interpolate")
        };
        assert!(matches!(
            aligned.alignment(),
            TimeAlignment::InterpolatedVisual { .. }
        ));
        assert_close(
            aligned.base_to_odom().source_origin_x_in_destination_m(),
            0.5,
            1.0e-6,
        );
        assert_close(
            aligned.base_to_odom().source_origin_y_in_destination_m(),
            -0.25,
            1.0e-6,
        );
        assert_close(
            aligned.base_to_odom().source_yaw_in_destination_rad(),
            0.2,
            1.0e-6,
        );
        assert!(matches!(
            odometry
                .pose_at(session(1), device(1_100_000_000), host(1))
                .expect("typed unavailable"),
            PoseHistoryQuery::Unavailable(OdometryUnavailable::NoGyroCoverage)
        ));
    }

    #[test]
    fn rejected_imu_order_does_not_poison_valid_retry() {
        let snapshot = SlamMap::new().snapshot();
        let mut odometry = anchored(snapshot);
        odometry
            .observe_imu(
                imu_report(session(1), 1, 10, 10, [0.0; 3], [0.0; 3]),
                host(10),
            )
            .expect("first accepted report");
        assert!(matches!(
            odometry.observe_imu(
                imu_report(session(1), 1, 20, 20, [0.0; 3], [0.0; 3]),
                host(20),
            ),
            Err(OdometryError::InertialOrdering(
                InertialOrderingError::DuplicateSequence { .. }
            ))
        ));
        let retry = odometry
            .observe_imu(
                imu_report(session(1), 2, 20, 20, [0.0; 3], [0.0; 3]),
                host(20),
            )
            .expect("valid retry after rejected duplicate");
        assert_eq!(retry.order, InertialOrderOutcome::Contiguous);
    }

    #[test]
    fn configuration_rejects_nonfinite_bounds_and_too_small_history() {
        let mut dto = config_dto();
        dto.max_visual_linear_speed_m_per_sec = f64::NAN;
        assert!(matches!(
            PlanarOdometryConfig::parse(dto.clone()),
            Err(PlanarOdometryConfigError::InvalidPositiveScalar {
                parameter: ScalarParameter::MaximumVisualLinearSpeed,
                value,
            }) if value.is_nan()
        ));
        dto.max_visual_linear_speed_m_per_sec = 1.0;
        dto.pose_history_capacity = 1;
        assert!(matches!(
            PlanarOdometryConfig::parse(dto),
            Err(PlanarOdometryConfigError::PoseHistoryCapacityTooSmall {
                actual: 1,
                minimum: 2,
            })
        ));
    }

    #[test]
    fn configuration_caps_history_before_any_allocation() {
        let mut gyro_too_large = config_dto();
        gyro_too_large.gyro_history_capacity = PlanarOdometryConfig::MAX_GYRO_HISTORY_CAPACITY + 1;
        assert!(matches!(
            PlanarOdometryConfig::parse(gyro_too_large),
            Err(PlanarOdometryConfigError::GyroHistoryCapacityTooLarge {
                actual,
                maximum,
            }) if actual == PlanarOdometryConfig::MAX_GYRO_HISTORY_CAPACITY + 1
                && maximum == PlanarOdometryConfig::MAX_GYRO_HISTORY_CAPACITY
        ));

        let mut pose_too_large = config_dto();
        pose_too_large.pose_history_capacity = PlanarOdometryConfig::MAX_POSE_HISTORY_CAPACITY + 1;
        assert!(matches!(
            PlanarOdometryConfig::parse(pose_too_large),
            Err(PlanarOdometryConfigError::PoseHistoryCapacityTooLarge {
                actual,
                maximum,
            }) if actual == PlanarOdometryConfig::MAX_POSE_HISTORY_CAPACITY + 1
                && maximum == PlanarOdometryConfig::MAX_POSE_HISTORY_CAPACITY
        ));
    }
}
