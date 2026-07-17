use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::io::{Seek, SeekFrom, Write};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use crate::inertial::{
    AccelSample, DequeueSequence, DeviceSessionId, DeviceTimestamp, GyroSample,
    HostMonotonicTimestamp, ImuReport, InertialAxis, InertialOrderTracker, InertialOrderingError,
    InertialSensorKind, InertialValueError, OakImuAcceleration, OakImuAngularVelocity,
    SensorAccuracy,
};
use crate::triangulation::{
    RectifiedStereo, RectifiedStereoConfig, RectifiedStereoError, StereoBaselineError,
    StereoBaselineMeters, StereoCalibration, StereoCameraSide,
};
use crate::{
    DepthImage, Frame, FrameDimensions, PairingWindowNs, PinholeIntrinsics, SensorId, StereoPair,
};

pub mod format {
    pub const FRAMES_DIR: &str = "frames";
    pub const META_FILE: &str = "meta.json";
    pub const CALIBRATION_FILE: &str = "calibration.json";
    pub const MANIFEST_FILE: &str = "manifest.json";
    pub const FRAME_SUFFIX: &str = ".raw";
    pub const IMU_STREAM_FILE: &str = "imu.v1.bin";

    pub(super) const IMU_STREAM_TEMP_FILE: &str = ".imu.v1.bin.tmp";

    pub fn frame_name(timestamp_ns: i64, sensor: &str) -> String {
        format!("{timestamp_ns}_{sensor}{FRAME_SUFFIX}")
    }

    pub fn parse_frame_filename(filename: &str) -> Option<(i64, String)> {
        let stem = filename.strip_suffix(FRAME_SUFFIX)?;
        let (timestamp_str, sensor) = stem.split_once('_')?;
        let timestamp_ns = timestamp_str.parse::<i64>().ok()?;
        Some((timestamp_ns, sensor.to_string()))
    }
}

mod reader;
pub use reader::{
    DatasetDepthCursor, DatasetImuCursor, DatasetReadTimings, DatasetReader, DatasetStats,
    TimedPair,
};

// Meta Structs
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Meta {
    pub created: String,
    pub device: String,
    pub mono: Option<MonoMeta>,
    #[serde(default)]
    pub depth: Option<DepthMeta>,
    pub imu: Option<ImuMeta>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MonoMeta {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImuMeta {
    /// Requested nominal rate for both raw OAK inertial sensors.
    /// Replay ordering and timing use each sample's device timestamp rather
    /// than deriving time from this nominal configuration value.
    pub rate_hz: u32,
}

/// Provenance for the transform from the OAK's native IMU frame.
///
/// Neither state implies alignment with Kiko's robot-base frame. A calibrated
/// value identifies only an independently supplied IMU-to-tracking-camera
/// calibration artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImuExtrinsicProvenance(ImuExtrinsicProvenanceKind);

#[derive(Clone, Debug, PartialEq, Eq)]
enum ImuExtrinsicProvenanceKind {
    UncalibratedUnknown,
    CalibratedToTrackingCamera { source: String },
}

impl ImuExtrinsicProvenance {
    pub fn uncalibrated_unknown() -> Self {
        Self(ImuExtrinsicProvenanceKind::UncalibratedUnknown)
    }

    pub fn try_calibrated_to_tracking_camera(
        source: impl Into<String>,
    ) -> Result<Self, ImuStreamMetadataError> {
        let source = source.into();
        if source.trim().is_empty() {
            return Err(ImuStreamMetadataError::EmptyCalibrationSource);
        }
        Ok(Self(
            ImuExtrinsicProvenanceKind::CalibratedToTrackingCamera { source },
        ))
    }
}

/// Complete metadata required before a new fixed-record OAK IMU stream can be written.
///
/// The remaining contract is intentionally fixed rather than caller-selectable:
/// acceleration is `m_s2`, angular velocity is `rad_s`, samples remain in the
/// DepthAI OAK native IMU axes, device timestamps belong to exactly one OAK
/// connection session, accelerometer and gyroscope timestamps are independent,
/// and host arrival uses one host process's monotonic nanosecond domain with an
/// intentionally unspecified epoch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImuStreamMetadata {
    device_session_id: DeviceSessionId,
    extrinsic: ImuExtrinsicProvenance,
}

impl ImuStreamMetadata {
    pub fn new(device_session_id: DeviceSessionId, extrinsic: ImuExtrinsicProvenance) -> Self {
        Self {
            device_session_id,
            extrinsic,
        }
    }

    pub fn try_from_raw_session_id(
        device_session_id: u64,
        extrinsic: ImuExtrinsicProvenance,
    ) -> Result<Self, ImuStreamMetadataError> {
        let device_session_id = DeviceSessionId::try_new(device_session_id)
            .map_err(|_| ImuStreamMetadataError::ZeroDeviceSessionId)?;
        Ok(Self::new(device_session_id, extrinsic))
    }

    pub fn device_session_id(&self) -> DeviceSessionId {
        self.device_session_id
    }

    pub fn extrinsic(&self) -> &ImuExtrinsicProvenance {
        &self.extrinsic
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImuStreamMetadataError {
    ZeroDeviceSessionId,
    EmptyCalibrationSource,
}

impl std::fmt::Display for ImuStreamMetadataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDeviceSessionId => f.write_str("IMU device session ID must be nonzero"),
            Self::EmptyCalibrationSource => {
                f.write_str("calibrated IMU extrinsic provenance must name its source")
            }
        }
    }
}

impl std::error::Error for ImuStreamMetadataError {}

const IMU_STREAM_MAGIC: [u8; 8] = *b"KIKOIMU\0";
const IMU_STREAM_FORMAT: &str = "fixed_le";
const IMU_STREAM_VERSION: u16 = 1;
const IMU_STREAM_HEADER_BYTES: u16 = 32;
const IMU_STREAM_RECORD_BYTES: u16 = 80;
const IMU_ACCELERATION_UNIT: &str = "m_s2";
const IMU_ANGULAR_VELOCITY_UNIT: &str = "rad_s";
const IMU_COORDINATE_FRAME: &str = "oak_imu";
const IMU_AXES: &str = "depthai_native";
const IMU_DEVICE_TIMEBASE: &str = "oak_device_clock_ns";
const IMU_DEVICE_SESSION_SEMANTICS: &str = "single_oak_connection_session";
const IMU_SAMPLE_TIMESTAMP_SEMANTICS: &str = "independent_accel_gyro";
const IMU_HOST_ARRIVAL_TIMEBASE: &str = "host_process_monotonic_ns";

#[derive(Clone, Debug)]
struct ImuStreamContract {
    nominal_rate_hz: NonZeroU32,
    metadata: ImuStreamMetadata,
}

impl ImuStreamContract {
    fn parse_for_write(
        meta: &Meta,
        metadata: Option<ImuStreamMetadata>,
    ) -> Result<Option<Self>, DatasetError> {
        match (meta.imu.as_ref(), metadata) {
            (None, None) => Ok(None),
            (Some(_), None) => Err(DatasetError::ImuMetadataRequiredForWrite),
            (None, Some(_)) => Err(DatasetError::ImuMetadataWithoutNominalStream),
            (Some(imu), Some(metadata)) => {
                let nominal_rate_hz = NonZeroU32::new(imu.rate_hz)
                    .ok_or(DatasetError::InvalidImuNominalRate { value: imu.rate_hz })?;
                Ok(Some(Self {
                    nominal_rate_hz,
                    metadata,
                }))
            }
        }
    }

    fn require_report(&self, report: &ImuReport) -> Result<(), DatasetWriteError> {
        let expected = self.metadata.device_session_id();
        let actual = report.session_id();
        if actual != expected {
            return Err(DatasetWriteError::ImuDeviceSessionMismatch {
                expected: expected.as_u64(),
                actual: actual.as_u64(),
            });
        }
        Ok(())
    }

    fn manifest(&self, written: RecordedImuStream) -> ManifestImuStream {
        let extrinsic = match &self.metadata.extrinsic.0 {
            ImuExtrinsicProvenanceKind::UncalibratedUnknown => {
                ManifestImuExtrinsic::UncalibratedUnknown
            }
            ImuExtrinsicProvenanceKind::CalibratedToTrackingCamera { source } => {
                ManifestImuExtrinsic::CalibratedToTrackingCamera {
                    source: source.clone(),
                }
            }
        };
        ManifestImuStream {
            path: format::IMU_STREAM_FILE.to_string(),
            format: IMU_STREAM_FORMAT.to_string(),
            version: IMU_STREAM_VERSION,
            header_bytes: IMU_STREAM_HEADER_BYTES,
            record_bytes: IMU_STREAM_RECORD_BYTES,
            record_count: written.record_count,
            byte_len: written.byte_len,
            nominal_rate_hz: self.nominal_rate_hz.get(),
            acceleration_unit: IMU_ACCELERATION_UNIT.to_string(),
            angular_velocity_unit: IMU_ANGULAR_VELOCITY_UNIT.to_string(),
            coordinate_frame: IMU_COORDINATE_FRAME.to_string(),
            axes: IMU_AXES.to_string(),
            extrinsic,
            device_session_id: self.metadata.device_session_id().as_u64(),
            device_timebase: IMU_DEVICE_TIMEBASE.to_string(),
            device_session_semantics: IMU_DEVICE_SESSION_SEMANTICS.to_string(),
            sample_timestamp_semantics: IMU_SAMPLE_TIMESTAMP_SEMANTICS.to_string(),
            host_arrival_timebase: IMU_HOST_ARRIVAL_TIMEBASE.to_string(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ImuWireRecord {
    sequence: u32,
    accel_accuracy: SensorAccuracy,
    gyro_accuracy: SensorAccuracy,
    host_arrival_ns: u64,
    accel_timestamp_ns: u64,
    gyro_timestamp_ns: u64,
    accel_m_s2: [f64; 3],
    gyro_rad_s: [f64; 3],
}

impl ImuWireRecord {
    fn from_report(report: ImuReport) -> Self {
        Self {
            sequence: report.sequence().as_u32(),
            accel_accuracy: report.accel().accuracy(),
            gyro_accuracy: report.gyro().accuracy(),
            host_arrival_ns: report.host_arrival().as_nanos(),
            accel_timestamp_ns: report.accel().timestamp().as_nanos(),
            gyro_timestamp_ns: report.gyro().timestamp().as_nanos(),
            accel_m_s2: report.accel().acceleration().as_array(),
            gyro_rad_s: report.gyro().angular_velocity().as_array(),
        }
    }

    fn into_report(
        self,
        session_id: DeviceSessionId,
        record_index: usize,
    ) -> Result<ImuReport, ImuStreamError> {
        let accel_timestamp =
            parse_device_timestamp(record_index, "accelerometer", self.accel_timestamp_ns)?;
        let gyro_timestamp =
            parse_device_timestamp(record_index, "gyroscope", self.gyro_timestamp_ns)?;
        let accel =
            OakImuAcceleration::try_new(self.accel_m_s2[0], self.accel_m_s2[1], self.accel_m_s2[2])
                .map_err(|source| {
                    map_vector_error(record_index, ["accel_x", "accel_y", "accel_z"], source)
                })?;
        let gyro = OakImuAngularVelocity::try_new(
            self.gyro_rad_s[0],
            self.gyro_rad_s[1],
            self.gyro_rad_s[2],
        )
        .map_err(|source| map_vector_error(record_index, ["gyro_x", "gyro_y", "gyro_z"], source))?;
        Ok(ImuReport::new(
            session_id,
            DequeueSequence::new(self.sequence),
            HostMonotonicTimestamp::from_nanos(self.host_arrival_ns),
            AccelSample::new(accel_timestamp, accel, self.accel_accuracy),
            GyroSample::new(gyro_timestamp, gyro, self.gyro_accuracy),
        ))
    }

    fn encode(self) -> [u8; IMU_STREAM_RECORD_BYTES as usize] {
        let mut bytes = [0_u8; IMU_STREAM_RECORD_BYTES as usize];
        bytes[0..4].copy_from_slice(&self.sequence.to_le_bytes());
        bytes[4] = encode_accuracy(self.accel_accuracy);
        bytes[5] = encode_accuracy(self.gyro_accuracy);
        // bytes 6..8 are reserved zeros.
        write_u64(&mut bytes, 8, self.host_arrival_ns);
        write_u64(&mut bytes, 16, self.accel_timestamp_ns);
        write_u64(&mut bytes, 24, self.gyro_timestamp_ns);
        for (index, value) in self.accel_m_s2.into_iter().enumerate() {
            write_f64(&mut bytes, 32 + index * 8, value);
        }
        for (index, value) in self.gyro_rad_s.into_iter().enumerate() {
            write_f64(&mut bytes, 56 + index * 8, value);
        }
        bytes
    }

    fn decode(
        bytes: &[u8; IMU_STREAM_RECORD_BYTES as usize],
        record_index: usize,
    ) -> Result<Self, ImuStreamError> {
        if bytes[6] != 0 || bytes[7] != 0 {
            return Err(ImuStreamError::NonZeroReservedRecord { record_index });
        }
        let accel_accuracy = decode_accuracy(bytes[4]).ok_or(ImuStreamError::UnknownAccuracy {
            record_index,
            sensor: "accelerometer",
            value: bytes[4],
        })?;
        let gyro_accuracy = decode_accuracy(bytes[5]).ok_or(ImuStreamError::UnknownAccuracy {
            record_index,
            sensor: "gyroscope",
            value: bytes[5],
        })?;
        let mut accel_m_s2 = [0.0; 3];
        let mut gyro_rad_s = [0.0; 3];
        for (index, value) in accel_m_s2.iter_mut().enumerate() {
            *value = read_f64(bytes, 32 + index * 8);
        }
        for (index, value) in gyro_rad_s.iter_mut().enumerate() {
            *value = read_f64(bytes, 56 + index * 8);
        }
        Ok(Self {
            sequence: read_u32(bytes, 0),
            accel_accuracy,
            gyro_accuracy,
            host_arrival_ns: read_u64(bytes, 8),
            accel_timestamp_ns: read_u64(bytes, 16),
            gyro_timestamp_ns: read_u64(bytes, 24),
            accel_m_s2,
            gyro_rad_s,
        })
    }
}

fn map_vector_error(
    record_index: usize,
    axis_fields: [&'static str; 3],
    source: InertialValueError,
) -> ImuStreamError {
    match source {
        InertialValueError::NonFiniteComponent { axis, value, .. } => {
            let field = axis_fields[match axis {
                InertialAxis::X => 0,
                InertialAxis::Y => 1,
                InertialAxis::Z => 2,
            }];
            ImuStreamError::NonFiniteSample {
                record_index,
                field,
                value,
            }
        }
        _ => unreachable!("three-axis construction can only reject non-finite components"),
    }
}

fn parse_device_timestamp(
    record_index: usize,
    sensor: &'static str,
    value_ns: u64,
) -> Result<DeviceTimestamp, ImuStreamError> {
    let signed = i64::try_from(value_ns).map_err(|_| ImuStreamError::InvalidDeviceTimestamp {
        record_index,
        sensor,
        value_ns,
    })?;
    DeviceTimestamp::try_from_nanos(signed).map_err(|_| ImuStreamError::InvalidDeviceTimestamp {
        record_index,
        sensor,
        value_ns,
    })
}

fn encode_accuracy(accuracy: SensorAccuracy) -> u8 {
    match accuracy {
        SensorAccuracy::Unreliable => 0,
        SensorAccuracy::Low => 1,
        SensorAccuracy::Medium => 2,
        SensorAccuracy::High => 3,
    }
}

fn decode_accuracy(value: u8) -> Option<SensorAccuracy> {
    match value {
        0 => Some(SensorAccuracy::Unreliable),
        1 => Some(SensorAccuracy::Low),
        2 => Some(SensorAccuracy::Medium),
        3 => Some(SensorAccuracy::High),
        _ => None,
    }
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn write_f64(bytes: &mut [u8], offset: usize, value: f64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ])
}

fn read_f64(bytes: &[u8], offset: usize) -> f64 {
    f64::from_bits(read_u64(bytes, offset))
}

fn encode_imu_header(record_count: u64) -> [u8; IMU_STREAM_HEADER_BYTES as usize] {
    let mut bytes = [0_u8; IMU_STREAM_HEADER_BYTES as usize];
    bytes[0..8].copy_from_slice(&IMU_STREAM_MAGIC);
    bytes[8..10].copy_from_slice(&IMU_STREAM_VERSION.to_le_bytes());
    bytes[10..12].copy_from_slice(&IMU_STREAM_HEADER_BYTES.to_le_bytes());
    bytes[12..14].copy_from_slice(&IMU_STREAM_RECORD_BYTES.to_le_bytes());
    // bytes 14..16 and 24..32 are reserved zeros.
    write_u64(&mut bytes, 16, record_count);
    bytes
}

fn decode_imu_header(
    bytes: &[u8; IMU_STREAM_HEADER_BYTES as usize],
) -> Result<u64, ImuStreamError> {
    if bytes[0..8] != IMU_STREAM_MAGIC {
        return Err(ImuStreamError::InvalidMagic);
    }
    let version = read_u16(bytes, 8);
    if version != IMU_STREAM_VERSION {
        return Err(ImuStreamError::UnsupportedVersion {
            expected: IMU_STREAM_VERSION,
            actual: version,
        });
    }
    let header_bytes = read_u16(bytes, 10);
    if header_bytes != IMU_STREAM_HEADER_BYTES {
        return Err(ImuStreamError::HeaderLengthMismatch {
            expected: IMU_STREAM_HEADER_BYTES,
            actual: header_bytes,
        });
    }
    let record_bytes = read_u16(bytes, 12);
    if record_bytes != IMU_STREAM_RECORD_BYTES {
        return Err(ImuStreamError::RecordLengthMismatch {
            expected: IMU_STREAM_RECORD_BYTES,
            actual: record_bytes,
        });
    }
    if bytes[14..16].iter().any(|&value| value != 0)
        || bytes[24..32].iter().any(|&value| value != 0)
    {
        return Err(ImuStreamError::NonZeroReservedHeader);
    }
    Ok(read_u64(bytes, 16))
}

fn imu_stream_byte_len(record_count: u64) -> Result<u64, ImuStreamError> {
    record_count
        .checked_mul(u64::from(IMU_STREAM_RECORD_BYTES))
        .and_then(|records| records.checked_add(u64::from(IMU_STREAM_HEADER_BYTES)))
        .ok_or(ImuStreamError::ByteLengthOverflow { record_count })
}

fn map_inertial_order_error(record_index: usize, error: InertialOrderingError) -> ImuStreamError {
    match error {
        InertialOrderingError::SessionMismatch { expected, actual } => {
            ImuStreamError::DeviceSessionMismatch {
                record_index,
                expected: expected.as_u64(),
                actual: actual.as_u64(),
            }
        }
        InertialOrderingError::DuplicateSequence { sequence, .. } => {
            ImuStreamError::SequenceNotNewer {
                record_index,
                previous: sequence.as_u32(),
                current: sequence.as_u32(),
            }
        }
        InertialOrderingError::SequenceRegression {
            previous, current, ..
        } => ImuStreamError::SequenceNotNewer {
            record_index,
            previous: previous.as_u32(),
            current: current.as_u32(),
        },
        InertialOrderingError::HostArrivalRegression {
            previous, current, ..
        } => ImuStreamError::HostArrivalRegressed {
            record_index,
            previous_ns: previous.as_nanos(),
            current_ns: current.as_nanos(),
        },
        InertialOrderingError::DuplicateDeviceTimestamp {
            sensor, timestamp, ..
        } => ImuStreamError::DeviceTimestampNotIncreasing {
            record_index,
            sensor: inertial_sensor_name(sensor),
            previous_ns: timestamp.as_nanos(),
            current_ns: timestamp.as_nanos(),
        },
        InertialOrderingError::DeviceTimestampRegression {
            sensor,
            previous,
            current,
            ..
        } => ImuStreamError::DeviceTimestampNotIncreasing {
            record_index,
            sensor: inertial_sensor_name(sensor),
            previous_ns: previous.as_nanos(),
            current_ns: current.as_nanos(),
        },
    }
}

fn inertial_sensor_name(sensor: InertialSensorKind) -> &'static str {
    match sensor {
        InertialSensorKind::Accelerometer => "accelerometer",
        InertialSensorKind::Gyroscope => "gyroscope",
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DepthMeta {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub encoding: String,
    /// Optical frame whose pixel grid and intrinsics define each depth sample.
    /// Legacy datasets omit this and cannot be used for geometric projection
    /// without an explicit caller-supplied assumption.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optical_frame: Option<DepthOpticalFrame>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DepthOpticalFrame {
    RectifiedLeft,
    RectifiedRight,
    Rgb,
}

/// Parsed geometric contract for projecting recorded depth samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthProjectionContract {
    dimensions: FrameDimensions,
    optical_frame: Option<DepthOpticalFrame>,
}

impl DepthProjectionContract {
    pub fn new(dimensions: FrameDimensions, optical_frame: DepthOpticalFrame) -> Self {
        Self {
            dimensions,
            optical_frame: Some(optical_frame),
        }
    }

    pub fn dimensions(self) -> FrameDimensions {
        self.dimensions
    }

    /// `None` identifies legacy metadata that did not record the optical frame.
    pub fn optical_frame(self) -> Option<DepthOpticalFrame> {
        self.optical_frame
    }
}

// Calibration structs
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Calibration {
    pub left: CameraIntrinsics,
    pub right: CameraIntrinsics,
    pub baseline_m: f32,
    #[serde(default)]
    pub rectified: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CameraIntrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
}

/// Parse the projection coefficients represented by a serialized camera.
///
/// Image dimensions are deliberately outside [`PinholeIntrinsics`]; callers
/// that own a complete image contract must parse those into a dimensions type
/// separately.
impl TryFrom<&CameraIntrinsics> for PinholeIntrinsics {
    type Error = crate::IntrinsicsError;

    fn try_from(value: &CameraIntrinsics) -> Result<Self, Self::Error> {
        Self::try_new(value.fx, value.fy, value.cx, value.cy)
    }
}

/// One camera parsed from the weak serialized calibration carrier.
#[derive(Clone, Copy, Debug)]
struct ParsedSerializedCamera {
    intrinsics: PinholeIntrinsics,
    dimensions: FrameDimensions,
}

impl ParsedSerializedCamera {
    fn parse(
        camera: StereoCameraSide,
        value: &CameraIntrinsics,
    ) -> Result<Self, RectifiedStereoError> {
        let dimensions = FrameDimensions::try_new(value.width, value.height)
            .map_err(|source| RectifiedStereoError::InvalidDimensions { camera, source })?;
        let intrinsics = PinholeIntrinsics::try_from(value)
            .map_err(|source| RectifiedStereoError::InvalidIntrinsics { camera, source })?;
        Ok(Self {
            intrinsics,
            dimensions,
        })
    }
}

impl RectifiedStereo {
    /// Parse a weak serialized calibration and apply the default rectification
    /// compatibility policy.
    pub fn from_calibration(calibration: &Calibration) -> Result<Self, RectifiedStereoError> {
        Self::from_calibration_with_config(calibration, RectifiedStereoConfig::default())
    }

    /// Parse a weak serialized calibration exactly once, then apply the given
    /// rectification compatibility policy.
    pub fn from_calibration_with_config(
        calibration: &Calibration,
        config: RectifiedStereoConfig,
    ) -> Result<Self, RectifiedStereoError> {
        let baseline =
            StereoBaselineMeters::try_new(calibration.baseline_m).map_err(
                |source| match source {
                    StereoBaselineError::NonFinite { baseline_m } => {
                        RectifiedStereoError::NonFiniteBaseline { baseline_m }
                    }
                    StereoBaselineError::NonPositive { baseline_m } => {
                        RectifiedStereoError::NonPositiveBaseline { baseline_m }
                    }
                },
            )?;

        let left = ParsedSerializedCamera::parse(StereoCameraSide::Left, &calibration.left)?;
        let right = ParsedSerializedCamera::parse(StereoCameraSide::Right, &calibration.right)?;

        if left.dimensions != right.dimensions {
            return Err(RectifiedStereoError::DimensionMismatch {
                left: left.dimensions,
                right: right.dimensions,
            });
        }

        let parsed = StereoCalibration::new(
            left.intrinsics,
            right.intrinsics,
            left.dimensions,
            baseline,
            calibration.rectified,
        );

        Self::from_stereo_calibration_with_config(&parsed, config).map_err(Into::into)
    }
}

#[derive(Clone, Copy, Debug)]
struct ParsedMonoContract {
    image: MonoImageContract,
    stereo: StereoCalibration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MonoImageContract {
    dimensions: FrameDimensions,
    nominal_fps_hz: NonZeroU32,
}

impl ParsedMonoContract {
    fn parse(meta: &Meta, calibration: &Calibration) -> Result<Self, DatasetError> {
        let mono = meta.mono.as_ref().ok_or(DatasetError::MissingMonoConfig)?;
        let dimensions = parse_image_dimensions("meta.mono", mono.width, mono.height)?;
        let nominal_fps_hz = NonZeroU32::new(mono.fps).ok_or(DatasetError::InvalidNominalFps {
            field: "meta.mono.fps",
            value: mono.fps,
        })?;
        let image = MonoImageContract {
            dimensions,
            nominal_fps_hz,
        };
        image.require_dimensions(
            "calibration.left",
            parse_image_dimensions(
                "calibration.left",
                calibration.left.width,
                calibration.left.height,
            )?,
        )?;
        image.require_dimensions(
            "calibration.right",
            parse_image_dimensions(
                "calibration.right",
                calibration.right.width,
                calibration.right.height,
            )?,
        )?;
        let left = PinholeIntrinsics::try_new(
            calibration.left.fx,
            calibration.left.fy,
            calibration.left.cx,
            calibration.left.cy,
        )
        .map_err(|source| DatasetError::InvalidCameraIntrinsics {
            field: "calibration.left",
            source,
        })?;
        let right = PinholeIntrinsics::try_new(
            calibration.right.fx,
            calibration.right.fy,
            calibration.right.cx,
            calibration.right.cy,
        )
        .map_err(|source| DatasetError::InvalidCameraIntrinsics {
            field: "calibration.right",
            source,
        })?;
        let baseline = StereoBaselineMeters::try_new(calibration.baseline_m)
            .map_err(|source| DatasetError::InvalidStereoBaseline { source })?;
        Ok(Self {
            image,
            stereo: StereoCalibration::new(
                left,
                right,
                dimensions,
                baseline,
                calibration.rectified,
            ),
        })
    }
}

impl MonoImageContract {
    fn dimensions(self) -> FrameDimensions {
        self.dimensions
    }

    fn nominal_fps_hz(self) -> NonZeroU32 {
        self.nominal_fps_hz
    }

    fn require_dimensions(
        self,
        field: &'static str,
        actual: FrameDimensions,
    ) -> Result<(), DatasetError> {
        if actual != self.dimensions {
            return Err(DatasetError::ImageDimensionsMismatch {
                expected_field: "meta.mono",
                expected: self.dimensions,
                actual_field: field,
                actual,
            });
        }
        Ok(())
    }

    fn require_frame(self, frame: &Frame) -> Result<(), DatasetWriteError> {
        let actual = frame.dimensions();
        if actual != self.dimensions {
            return Err(DatasetWriteError::FrameDimensionsMismatch {
                sensor: frame.sensor_id(),
                expected: self.dimensions,
                actual,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DepthPayloadFormat {
    F32MetersLe,
}

impl DepthPayloadFormat {
    fn parse(value: &str) -> Result<Self, DatasetError> {
        match value {
            "f32_meters_le" => Ok(Self::F32MetersLe),
            _ => Err(DatasetError::UnsupportedDepthEncoding {
                value: value.to_string(),
            }),
        }
    }

    fn bytes_per_sample(self) -> usize {
        match self {
            Self::F32MetersLe => std::mem::size_of::<f32>(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DepthImageContract {
    dimensions: FrameDimensions,
    optical_frame: Option<DepthOpticalFrame>,
    _nominal_fps_hz: NonZeroU32,
    payload_format: DepthPayloadFormat,
    expected_payload_len: u64,
}

impl DepthImageContract {
    fn parse(meta: &DepthMeta) -> Result<Self, DatasetError> {
        let dimensions = parse_image_dimensions("meta.depth", meta.width, meta.height)?;
        let nominal_fps_hz = NonZeroU32::new(meta.fps).ok_or(DatasetError::InvalidNominalFps {
            field: "meta.depth.fps",
            value: meta.fps,
        })?;
        let payload_format = DepthPayloadFormat::parse(&meta.encoding)?;
        let expected_payload_len = dimensions
            .area()
            .checked_mul(payload_format.bytes_per_sample())
            .and_then(|len| u64::try_from(len).ok())
            .ok_or(DatasetError::DepthPayloadSizeOverflow { dimensions })?;
        Ok(Self {
            dimensions,
            optical_frame: meta.optical_frame,
            _nominal_fps_hz: nominal_fps_hz,
            payload_format,
            expected_payload_len,
        })
    }

    fn require_image(self, depth: &DepthImage) -> Result<(), DatasetWriteError> {
        let actual = depth.dimensions();
        if actual != self.dimensions {
            return Err(DatasetWriteError::DepthDimensionsMismatch {
                expected: self.dimensions,
                actual,
            });
        }
        Ok(())
    }

    fn expected_payload_len(self) -> u64 {
        self.expected_payload_len
    }

    fn dimensions(self) -> FrameDimensions {
        self.dimensions
    }

    fn projection(self) -> DepthProjectionContract {
        DepthProjectionContract {
            dimensions: self.dimensions,
            optical_frame: self.optical_frame,
        }
    }

    fn decode(self, bytes: &[u8]) -> Vec<f32> {
        match self.payload_format {
            DepthPayloadFormat::F32MetersLe => bytes
                .chunks_exact(std::mem::size_of::<f32>())
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect(),
        }
    }
}

fn parse_image_dimensions(
    field: &'static str,
    width: u32,
    height: u32,
) -> Result<FrameDimensions, DatasetError> {
    FrameDimensions::try_new(width, height)
        .map_err(|source| DatasetError::InvalidFrameDimensions { field, source })
}

#[derive(Clone, Debug)]
struct WriterDatasetContract {
    created_at: String,
    device: String,
    mono: MonoImageContract,
    depth: Option<DepthImageContract>,
    imu: Option<ImuStreamContract>,
}

impl WriterDatasetContract {
    #[cfg(test)]
    fn parse(meta: &Meta, calibration: &Calibration) -> Result<Self, DatasetError> {
        Self::parse_with_imu(meta, calibration, None)
    }

    fn parse_with_imu(
        meta: &Meta,
        calibration: &Calibration,
        imu_metadata: Option<ImuStreamMetadata>,
    ) -> Result<Self, DatasetError> {
        let mono = ParsedMonoContract::parse(meta, calibration)?.image;
        if meta
            .depth
            .as_ref()
            .is_some_and(|depth| depth.optical_frame.is_none())
        {
            return Err(DatasetError::DepthOpticalFrameRequiredForWrite);
        }
        Ok(Self {
            created_at: meta.created.clone(),
            device: meta.device.clone(),
            mono,
            depth: meta
                .depth
                .as_ref()
                .map(DepthImageContract::parse)
                .transpose()?,
            imu: ImuStreamContract::parse_for_write(meta, imu_metadata)?,
        })
    }
}

#[derive(Debug)]
struct WriterSidecars {
    meta_json: Vec<u8>,
    calibration_json: Vec<u8>,
}

impl WriterSidecars {
    fn require_unchanged(&self, dataset_dir: &Path) -> Result<(), DatasetError> {
        self.require_file(dataset_dir.join(format::META_FILE), &self.meta_json)?;
        self.require_file(
            dataset_dir.join(format::CALIBRATION_FILE),
            &self.calibration_json,
        )
    }

    fn require_file(&self, path: PathBuf, expected: &[u8]) -> Result<(), DatasetError> {
        let actual = std::fs::read(&path).map_err(|source| DatasetError::ReadFile {
            path: path.clone(),
            source,
        })?;
        if actual != expected {
            return Err(DatasetError::SidecarChanged { path });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Backpressure {
    DropNewest,
    Block,
}

#[derive(Clone, Copy, Debug)]
pub struct DatasetWriterConfig {
    /// Maximum accepted logical payload units retained across the queue and in-flight batch.
    /// Mono, depth, and IMU reports count as one unit; a stereo pair counts as two.
    /// The `frames` name is retained for source compatibility.
    pub max_spool_frames: usize,
    /// Maximum accepted payload bytes retained across the queue and in-flight batch.
    pub max_spool_bytes: usize,
    /// Target logical payload units drained in one worker batch.
    /// The worker stops after reaching or crossing this value and never splits
    /// an atomic stereo pair. Units follow [`DatasetWriterConfig::max_spool_frames`].
    pub flush_batch_frames: usize,
    pub backpressure: Backpressure,
}

impl Default for DatasetWriterConfig {
    fn default() -> Self {
        Self {
            max_spool_frames: 64,
            max_spool_bytes: 32 * 1024 * 1024,
            flush_batch_frames: 16,
            backpressure: Backpressure::DropNewest,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteOutcome {
    Enqueued,
    Dropped,
    WriterFailed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DatasetWriteError {
    FrameDimensionsMismatch {
        sensor: SensorId,
        expected: FrameDimensions,
        actual: FrameDimensions,
    },
    DepthStreamNotConfigured,
    DepthDimensionsMismatch {
        expected: FrameDimensions,
        actual: FrameDimensions,
    },
    ImuStreamNotConfigured,
    ImuDeviceSessionMismatch {
        expected: u64,
        actual: u64,
    },
    PairOutsideWriterWindow {
        delta_ns: u64,
        max_delta_ns: u64,
    },
    /// One atomic item exceeds the logical-unit capacity. `frames` retains its
    /// compatibility name and follows [`WriterStats`] accounting semantics.
    SpoolFrameCapacityExceeded {
        item: &'static str,
        frames: usize,
        max_frames: usize,
    },
    SpoolByteCapacityExceeded {
        item: &'static str,
        bytes: usize,
        max_bytes: usize,
    },
}

impl std::fmt::Display for DatasetWriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FrameDimensionsMismatch {
                sensor,
                expected,
                actual,
            } => write!(
                f,
                "dataset {:?} frame dimensions must be {}x{}, got {}x{}",
                sensor,
                expected.width(),
                expected.height(),
                actual.width(),
                actual.height()
            ),
            Self::DepthStreamNotConfigured => {
                write!(f, "dataset metadata does not configure a depth stream")
            }
            Self::DepthDimensionsMismatch { expected, actual } => write!(
                f,
                "dataset depth image dimensions must be {}x{}, got {}x{}",
                expected.width(),
                expected.height(),
                actual.width(),
                actual.height()
            ),
            Self::ImuStreamNotConfigured => {
                f.write_str("dataset writer does not configure a replayable IMU stream")
            }
            Self::ImuDeviceSessionMismatch { expected, actual } => write!(
                f,
                "IMU report belongs to device session {actual}, but this dataset records session {expected}"
            ),
            Self::PairOutsideWriterWindow {
                delta_ns,
                max_delta_ns,
            } => write!(
                f,
                "stereo pair delta {delta_ns}ns exceeds the dataset writer window {max_delta_ns}ns"
            ),
            Self::SpoolFrameCapacityExceeded {
                item,
                frames,
                max_frames,
            } => write!(
                f,
                "dataset {item} requires {frames} logical spool units, but the configured maximum is {max_frames}"
            ),
            Self::SpoolByteCapacityExceeded {
                item,
                bytes,
                max_bytes,
            } => write!(
                f,
                "dataset {item} requires {bytes} spool bytes, but the configured maximum is {max_bytes}"
            ),
        }
    }
}

impl std::error::Error for DatasetWriteError {}

#[derive(Clone, Debug, PartialEq)]
pub enum ImuStreamError {
    InvalidMagic,
    UnsupportedVersion {
        expected: u16,
        actual: u16,
    },
    HeaderLengthMismatch {
        expected: u16,
        actual: u16,
    },
    RecordLengthMismatch {
        expected: u16,
        actual: u16,
    },
    NonZeroReservedHeader,
    NonZeroReservedRecord {
        record_index: usize,
    },
    ByteLengthOverflow {
        record_count: u64,
    },
    RecordIndexOutOfRange {
        record_count: u64,
    },
    ByteLengthMismatch {
        expected: u64,
        actual: u64,
    },
    ManifestMismatch {
        field: &'static str,
        declared: u64,
        encoded: u64,
    },
    UnknownAccuracy {
        record_index: usize,
        sensor: &'static str,
        value: u8,
    },
    NonFiniteSample {
        record_index: usize,
        field: &'static str,
        value: f64,
    },
    DeviceSessionMismatch {
        record_index: usize,
        expected: u64,
        actual: u64,
    },
    SequenceNotNewer {
        record_index: usize,
        previous: u32,
        current: u32,
    },
    HostArrivalRegressed {
        record_index: usize,
        previous_ns: u64,
        current_ns: u64,
    },
    DeviceTimestampNotIncreasing {
        record_index: usize,
        sensor: &'static str,
        previous_ns: u64,
        current_ns: u64,
    },
    InvalidDeviceTimestamp {
        record_index: usize,
        sensor: &'static str,
        value_ns: u64,
    },
}

impl std::fmt::Display for ImuStreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMagic => f.write_str("IMU stream magic does not identify Kiko IMU v1"),
            Self::UnsupportedVersion { expected, actual } => write!(
                f,
                "unsupported IMU stream version {actual}; expected {expected}"
            ),
            Self::HeaderLengthMismatch { expected, actual } => write!(
                f,
                "IMU stream header declares {actual} bytes; expected {expected}"
            ),
            Self::RecordLengthMismatch { expected, actual } => write!(
                f,
                "IMU stream record declares {actual} bytes; expected {expected}"
            ),
            Self::NonZeroReservedHeader => {
                f.write_str("IMU stream header reserved fields must be zero")
            }
            Self::NonZeroReservedRecord { record_index } => write!(
                f,
                "IMU stream record {record_index} has nonzero reserved fields"
            ),
            Self::ByteLengthOverflow { record_count } => write!(
                f,
                "IMU stream byte length overflows for {record_count} records"
            ),
            Self::RecordIndexOutOfRange { record_count } => write!(
                f,
                "IMU stream record count {record_count} exceeds the host index representation"
            ),
            Self::ByteLengthMismatch { expected, actual } => write!(
                f,
                "IMU stream must contain exactly {expected} bytes, got {actual}"
            ),
            Self::ManifestMismatch {
                field,
                declared,
                encoded,
            } => write!(
                f,
                "IMU manifest {field}={declared} disagrees with encoded value {encoded}"
            ),
            Self::UnknownAccuracy {
                record_index,
                sensor,
                value,
            } => write!(
                f,
                "IMU stream record {record_index} has unknown {sensor} accuracy code {value}"
            ),
            Self::NonFiniteSample {
                record_index,
                field,
                value,
            } => write!(
                f,
                "IMU stream record {record_index} has non-finite {field} value {value}"
            ),
            Self::DeviceSessionMismatch {
                record_index,
                expected,
                actual,
            } => write!(
                f,
                "IMU stream record {record_index} belongs to device session {actual}, expected {expected}"
            ),
            Self::SequenceNotNewer {
                record_index,
                previous,
                current,
            } => write!(
                f,
                "IMU stream record {record_index} dequeue sequence {current} is not newer than {previous}"
            ),
            Self::HostArrivalRegressed {
                record_index,
                previous_ns,
                current_ns,
            } => write!(
                f,
                "IMU stream record {record_index} host arrival regressed from {previous_ns}ns to {current_ns}ns"
            ),
            Self::DeviceTimestampNotIncreasing {
                record_index,
                sensor,
                previous_ns,
                current_ns,
            } => write!(
                f,
                "IMU stream record {record_index} {sensor} timestamp {current_ns}ns is not later than {previous_ns}ns"
            ),
            Self::InvalidDeviceTimestamp {
                record_index,
                sensor,
                value_ns,
            } => write!(
                f,
                "IMU stream record {record_index} has invalid {sensor} device timestamp {value_ns}ns"
            ),
        }
    }
}

impl std::error::Error for ImuStreamError {}

/// Writer accounting in logical payload units.
///
/// Fields retaining `frames` in their names count one unit for each mono frame,
/// depth image, or IMU report and two units for each stereo pair. The names are
/// retained for source compatibility; they do not mean that IMU reports are
/// camera frames.
#[derive(Clone, Copy, Debug)]
pub struct WriterStats {
    pub frames_enqueued: u64,
    pub frames_written: u64,
    pub frames_dropped: u64,
    pub bytes_enqueued: u64,
    pub bytes_written: u64,
    pub bytes_dropped: u64,
    /// Logical payload units belonging to accepted write transactions that failed.
    pub write_failed: u64,
    /// Bytes belonging to accepted write transactions that failed.
    pub bytes_write_failed: u64,
    /// Accepted logical payload units canceled after another write transaction failed.
    pub frames_canceled: u64,
    /// Accepted bytes that were canceled after another write transaction failed.
    pub bytes_canceled: u64,
    /// Accepted logical payload units still queued or being written.
    pub spool_frames: u64,
    /// Accepted payload bytes still queued or being written.
    pub spool_bytes: u64,
    pub spool_max_frames: u64,
    pub spool_max_bytes: u64,
    pub writer_failed: bool,
}

impl WriterStats {
    /// Accepted logical payload units not yet written, failed, or canceled.
    pub fn frames_pending(&self) -> u64 {
        self.frames_enqueued.saturating_sub(
            self.frames_written
                .saturating_add(self.write_failed)
                .saturating_add(self.frames_canceled),
        )
    }

    pub fn bytes_pending(&self) -> u64 {
        self.bytes_enqueued.saturating_sub(
            self.bytes_written
                .saturating_add(self.bytes_write_failed)
                .saturating_add(self.bytes_canceled),
        )
    }
}

#[derive(Debug)]
pub enum DatasetError {
    AlreadyExists {
        path: PathBuf,
    },
    CreateDirectory {
        path: PathBuf,
        source: std::io::Error,
    },
    ReadDirectory {
        path: PathBuf,
        source: std::io::Error,
    },
    ReadFile {
        path: PathBuf,
        source: std::io::Error,
    },
    InvalidConfig {
        msg: &'static str,
    },
    InvalidFramePath {
        path: String,
        reason: &'static str,
    },
    MissingMonoConfig,
    InvalidFrameDimensions {
        field: &'static str,
        source: crate::FrameDimensionsError,
    },
    InvalidNominalFps {
        field: &'static str,
        value: u32,
    },
    ImageDimensionsMismatch {
        expected_field: &'static str,
        expected: FrameDimensions,
        actual_field: &'static str,
        actual: FrameDimensions,
    },
    InvalidCameraIntrinsics {
        field: &'static str,
        source: crate::IntrinsicsError,
    },
    InvalidStereoBaseline {
        source: StereoBaselineError,
    },
    UnsupportedDepthEncoding {
        value: String,
    },
    DepthOpticalFrameRequiredForWrite,
    DepthPayloadSizeOverflow {
        dimensions: FrameDimensions,
    },
    DepthStreamNotConfigured,
    LegacyDepthManifestUnindexed,
    ManifestDepthEntriesWithoutMetadata {
        entry_count: usize,
    },
    DepthFrameIndexOutOfRange {
        index: usize,
    },
    DepthPayloadLengthOutOfRange {
        path: PathBuf,
        actual: usize,
    },
    DepthPayloadLengthChanged {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    ImuMetadataRequiredForWrite,
    ImuMetadataWithoutNominalStream,
    InvalidImuNominalRate {
        value: u32,
    },
    ImuNominalRateMismatch {
        metadata: u32,
        manifest: u32,
    },
    ImuStreamNotConfigured,
    LegacyImuManifestUnindexed,
    ManifestImuStreamWithoutMetadata,
    InvalidImuStream {
        path: PathBuf,
        source: ImuStreamError,
    },
    InvalidImuFileType {
        path: PathBuf,
    },
    SidecarChanged {
        path: PathBuf,
    },
    WriteContract {
        source: DatasetWriteError,
    },
    InvalidFrameFileType {
        path: PathBuf,
    },
    InvalidFrameLength {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    InvalidFrameData {
        path: PathBuf,
        source: crate::FrameError,
    },
    InvalidDepthData {
        path: PathBuf,
        source: crate::DepthImageError,
    },
    InvalidManifest {
        reason: &'static str,
    },
    UnsupportedManifestValue {
        field: &'static str,
        value: String,
    },
    NominalFpsMismatch {
        expected_field: &'static str,
        expected: u32,
        actual_field: &'static str,
        actual: u32,
    },
    ManifestMetadataMismatch {
        expected_field: &'static str,
        expected: String,
        actual_field: &'static str,
        actual: String,
    },
    ManifestPairingWindowOutOfRange {
        value: u64,
    },
    RecordedPairsDeclareOrphans {
        left_orphans: u64,
        right_orphans: u64,
    },
    RecordedPairsContainMissingRight {
        left_timestamp_ns: i64,
    },
    InvalidManifestStats {
        field: &'static str,
        declared: u64,
        derived: u64,
    },
    ManifestDeltaStatsPresenceMismatch {
        declared: bool,
        derived: bool,
    },
    ManifestFrameIdentityMismatch {
        declared: String,
        expected: PathBuf,
    },
    DuplicateManifestFrameRef {
        path: PathBuf,
    },
    NonMonotonicManifestEntries {
        previous_left_timestamp_ns: i64,
        current_left_timestamp_ns: i64,
    },
    NonMonotonicManifestDepthEntries {
        previous_depth_timestamp_ns: i64,
        current_depth_timestamp_ns: i64,
    },
    ManifestPairDeltaMismatch {
        left_timestamp_ns: i64,
        right_timestamp_ns: i64,
        declared_delta_ns: u64,
        derived_delta_ns: u64,
    },
    ManifestPairOutsideWindow {
        left_timestamp_ns: i64,
        right_timestamp_ns: i64,
        delta_ns: u64,
        max_delta_ns: u64,
    },
    ManifestCountOutOfRange {
        field: &'static str,
        value: u64,
    },
    PublishManifest {
        temporary_path: PathBuf,
        manifest_path: PathBuf,
        source: std::io::Error,
    },
    PublishImuStream {
        temporary_path: PathBuf,
        stream_path: PathBuf,
        source: std::io::Error,
    },
    ThreadSpawn {
        source: std::io::Error,
    },
    WriteFile {
        path: PathBuf,
        source: std::io::Error,
    },
    SerializeJson {
        document: &'static str,
        source: serde_json::Error,
    },
    DeserializeJson {
        path: PathBuf,
        source: serde_json::Error,
    },
    WorkerJoin {
        message: String,
    },
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetError::AlreadyExists { path } => {
                write!(f, "dataset path already exists: {}", path.display())
            }
            DatasetError::CreateDirectory { path, source } => {
                write!(
                    f,
                    "failed to create directory {}: {}",
                    path.display(),
                    source
                )
            }
            DatasetError::ReadDirectory { path, source } => {
                write!(f, "failed to read directory {}: {}", path.display(), source)
            }
            DatasetError::ReadFile { path, source } => {
                write!(f, "failed to read file {}: {}", path.display(), source)
            }
            DatasetError::InvalidConfig { msg } => {
                write!(f, "invalid dataset config: {msg}")
            }
            DatasetError::InvalidFramePath { path, reason } => {
                write!(f, "invalid dataset frame path {path:?}: {reason}")
            }
            DatasetError::MissingMonoConfig => {
                write!(
                    f,
                    "dataset metadata is missing the mono camera configuration"
                )
            }
            DatasetError::InvalidFrameDimensions { field, source } => {
                write!(f, "invalid dataset image dimensions in {field}: {source}")
            }
            DatasetError::InvalidNominalFps { field, value } => {
                write!(
                    f,
                    "dataset nominal frame rate {field} must be nonzero, got {value}"
                )
            }
            DatasetError::ImageDimensionsMismatch {
                expected_field,
                expected,
                actual_field,
                actual,
            } => write!(
                f,
                "dataset image dimensions disagree: {expected_field}={}x{}, {actual_field}={}x{}",
                expected.width(),
                expected.height(),
                actual.width(),
                actual.height()
            ),
            DatasetError::InvalidCameraIntrinsics { field, source } => {
                write!(f, "invalid dataset camera intrinsics in {field}: {source}")
            }
            DatasetError::InvalidStereoBaseline { source } => {
                write!(f, "invalid dataset stereo baseline: {source}")
            }
            DatasetError::UnsupportedDepthEncoding { value } => write!(
                f,
                "unsupported dataset depth encoding {value:?}; expected \"f32_meters_le\""
            ),
            DatasetError::DepthOpticalFrameRequiredForWrite => write!(
                f,
                "new dataset depth metadata must declare its optical frame; omission is reserved for reading legacy datasets"
            ),
            DatasetError::DepthPayloadSizeOverflow { dimensions } => write!(
                f,
                "dataset depth payload size overflows the host for {}x{} f32 samples",
                dimensions.width(),
                dimensions.height()
            ),
            DatasetError::DepthStreamNotConfigured => {
                write!(f, "dataset metadata does not configure a depth stream")
            }
            DatasetError::LegacyDepthManifestUnindexed => write!(
                f,
                "dataset depth stream is not indexed by this legacy manifest"
            ),
            DatasetError::ManifestDepthEntriesWithoutMetadata { entry_count } => write!(
                f,
                "dataset manifest declares {entry_count} depth entries without depth metadata"
            ),
            DatasetError::DepthFrameIndexOutOfRange { index } => write!(
                f,
                "dataset depth frame index {index} exceeds the frame-id representation"
            ),
            DatasetError::DepthPayloadLengthOutOfRange { path, actual } => write!(
                f,
                "dataset depth payload length for {} cannot be represented as u64: {actual} bytes",
                path.display()
            ),
            DatasetError::DepthPayloadLengthChanged {
                path,
                expected,
                actual,
            } => write!(
                f,
                "dataset depth payload length changed after open at {}: expected {expected} bytes, got {actual}",
                path.display()
            ),
            DatasetError::ImuMetadataRequiredForWrite => f.write_str(
                "new datasets with IMU metadata require an explicit replayable IMU stream contract",
            ),
            DatasetError::ImuMetadataWithoutNominalStream => {
                f.write_str("an IMU stream contract requires meta.imu nominal stream metadata")
            }
            DatasetError::InvalidImuNominalRate { value } => write!(
                f,
                "dataset nominal IMU rate must be nonzero, got {value} Hz"
            ),
            DatasetError::ImuNominalRateMismatch { metadata, manifest } => write!(
                f,
                "dataset nominal IMU rates disagree: meta.imu.rate_hz={metadata}Hz, manifest.imu_stream.nominal_rate_hz={manifest}Hz"
            ),
            DatasetError::ImuStreamNotConfigured => {
                f.write_str("dataset metadata does not configure an IMU stream")
            }
            DatasetError::LegacyImuManifestUnindexed => f.write_str(
                "legacy dataset IMU metadata does not identify a replayable fixed-record stream",
            ),
            DatasetError::ManifestImuStreamWithoutMetadata => {
                f.write_str("dataset manifest declares an IMU stream without meta.imu metadata")
            }
            DatasetError::InvalidImuStream { path, source } => write!(
                f,
                "invalid dataset IMU stream at {}: {source}",
                path.display()
            ),
            DatasetError::InvalidImuFileType { path } => write!(
                f,
                "dataset IMU stream path is not a regular file: {}",
                path.display()
            ),
            DatasetError::SidecarChanged { path } => write!(
                f,
                "dataset sidecar changed after writer creation: {}",
                path.display()
            ),
            DatasetError::WriteContract { source } => {
                write!(f, "dataset write contract violation: {source}")
            }
            DatasetError::InvalidFrameFileType { path } => write!(
                f,
                "dataset frame path is not a regular file: {}",
                path.display()
            ),
            DatasetError::InvalidFrameLength {
                path,
                expected,
                actual,
            } => write!(
                f,
                "invalid dataset frame length for {}: expected {expected} bytes, got {actual}",
                path.display()
            ),
            DatasetError::InvalidFrameData { path, source } => write!(
                f,
                "dataset frame changed after open at {}: {source}",
                path.display()
            ),
            DatasetError::InvalidDepthData { path, source } => write!(
                f,
                "dataset depth payload is invalid at {}: {source}",
                path.display()
            ),
            DatasetError::InvalidManifest { reason } => {
                write!(f, "invalid dataset manifest: {reason}")
            }
            DatasetError::UnsupportedManifestValue { field, value } => {
                write!(f, "unsupported dataset manifest value {field}={value:?}")
            }
            DatasetError::NominalFpsMismatch {
                expected_field,
                expected,
                actual_field,
                actual,
            } => write!(
                f,
                "dataset nominal frame rates disagree: {expected_field}={expected}Hz, {actual_field}={actual}Hz"
            ),
            DatasetError::ManifestMetadataMismatch {
                expected_field,
                expected,
                actual_field,
                actual,
            } => write!(
                f,
                "dataset metadata disagrees: {expected_field}={expected:?}, {actual_field}={actual:?}"
            ),
            DatasetError::ManifestPairingWindowOutOfRange { value } => write!(
                f,
                "dataset manifest pairing window {value}ns exceeds i64::MAX"
            ),
            DatasetError::RecordedPairsDeclareOrphans {
                left_orphans,
                right_orphans,
            } => write!(
                f,
                "recorded-pairs manifest cannot declare orphans: left={left_orphans}, right={right_orphans}"
            ),
            DatasetError::RecordedPairsContainMissingRight { left_timestamp_ns } => write!(
                f,
                "recorded-pairs manifest entry at left={left_timestamp_ns}ns is missing its recorded right frame"
            ),
            DatasetError::InvalidManifestStats {
                field,
                declared,
                derived,
            } => write!(
                f,
                "invalid dataset manifest statistic {field}: declared {declared}, derived {derived}"
            ),
            DatasetError::ManifestDeltaStatsPresenceMismatch { declared, derived } => write!(
                f,
                "invalid dataset manifest delta statistics presence: declared={declared}, derived={derived}"
            ),
            DatasetError::ManifestFrameIdentityMismatch { declared, expected } => write!(
                f,
                "dataset manifest frame identity {declared:?} does not match canonical path {}",
                expected.display()
            ),
            DatasetError::DuplicateManifestFrameRef { path } => write!(
                f,
                "dataset manifest references frame {} more than once",
                path.display()
            ),
            DatasetError::NonMonotonicManifestEntries {
                previous_left_timestamp_ns,
                current_left_timestamp_ns,
            } => write!(
                f,
                "dataset manifest left timestamps are not strictly increasing: previous={previous_left_timestamp_ns}ns, current={current_left_timestamp_ns}ns"
            ),
            DatasetError::NonMonotonicManifestDepthEntries {
                previous_depth_timestamp_ns,
                current_depth_timestamp_ns,
            } => write!(
                f,
                "dataset manifest depth timestamps are not strictly increasing: previous={previous_depth_timestamp_ns}ns, current={current_depth_timestamp_ns}ns"
            ),
            DatasetError::ManifestPairDeltaMismatch {
                left_timestamp_ns,
                right_timestamp_ns,
                declared_delta_ns,
                derived_delta_ns,
            } => write!(
                f,
                "dataset manifest pair delta is inconsistent for left={left_timestamp_ns}ns, right={right_timestamp_ns}ns: declared={declared_delta_ns}ns, derived={derived_delta_ns}ns"
            ),
            DatasetError::ManifestPairOutsideWindow {
                left_timestamp_ns,
                right_timestamp_ns,
                delta_ns,
                max_delta_ns,
            } => write!(
                f,
                "dataset manifest pair is outside its pairing window for left={left_timestamp_ns}ns, right={right_timestamp_ns}ns: delta={delta_ns}ns, max={max_delta_ns}ns"
            ),
            DatasetError::ManifestCountOutOfRange { field, value } => write!(
                f,
                "dataset manifest count {field}={value} exceeds the host address space"
            ),
            DatasetError::PublishManifest {
                temporary_path,
                manifest_path,
                source,
            } => write!(
                f,
                "failed to publish dataset manifest from {} to {}: {source}",
                temporary_path.display(),
                manifest_path.display()
            ),
            DatasetError::PublishImuStream {
                temporary_path,
                stream_path,
                source,
            } => write!(
                f,
                "failed to publish dataset IMU stream from {} to {}: {source}",
                temporary_path.display(),
                stream_path.display()
            ),
            DatasetError::ThreadSpawn { source } => {
                write!(f, "failed to spawn writer thread: {source}")
            }
            DatasetError::WriteFile { path, source } => {
                write!(f, "failed to write file {}: {}", path.display(), source)
            }
            DatasetError::SerializeJson { document, source } => {
                write!(f, "failed to serialize dataset {document}: {source}")
            }
            DatasetError::DeserializeJson { path, source } => {
                write!(
                    f,
                    "failed to parse dataset JSON {}: {source}",
                    path.display()
                )
            }
            DatasetError::WorkerJoin { message } => {
                write!(f, "writer thread panicked: {message}")
            }
        }
    }
}

impl std::error::Error for DatasetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DatasetError::CreateDirectory { source, .. }
            | DatasetError::ReadDirectory { source, .. }
            | DatasetError::ReadFile { source, .. }
            | DatasetError::ThreadSpawn { source }
            | DatasetError::WriteFile { source, .. }
            | DatasetError::PublishManifest { source, .. }
            | DatasetError::PublishImuStream { source, .. } => Some(source),
            DatasetError::SerializeJson { source, .. }
            | DatasetError::DeserializeJson { source, .. } => Some(source),
            DatasetError::InvalidFrameDimensions { source, .. } => Some(source),
            DatasetError::InvalidFrameData { source, .. } => Some(source),
            DatasetError::InvalidDepthData { source, .. } => Some(source),
            DatasetError::InvalidImuStream { source, .. } => Some(source),
            DatasetError::InvalidCameraIntrinsics { source, .. } => Some(source),
            DatasetError::InvalidStereoBaseline { source } => Some(source),
            DatasetError::WriteContract { source } => Some(source),
            DatasetError::AlreadyExists { .. }
            | DatasetError::InvalidConfig { .. }
            | DatasetError::InvalidFramePath { .. }
            | DatasetError::MissingMonoConfig
            | DatasetError::InvalidNominalFps { .. }
            | DatasetError::ImageDimensionsMismatch { .. }
            | DatasetError::UnsupportedDepthEncoding { .. }
            | DatasetError::DepthOpticalFrameRequiredForWrite
            | DatasetError::DepthPayloadSizeOverflow { .. }
            | DatasetError::DepthStreamNotConfigured
            | DatasetError::LegacyDepthManifestUnindexed
            | DatasetError::ManifestDepthEntriesWithoutMetadata { .. }
            | DatasetError::DepthFrameIndexOutOfRange { .. }
            | DatasetError::DepthPayloadLengthOutOfRange { .. }
            | DatasetError::DepthPayloadLengthChanged { .. }
            | DatasetError::ImuMetadataRequiredForWrite
            | DatasetError::ImuMetadataWithoutNominalStream
            | DatasetError::InvalidImuNominalRate { .. }
            | DatasetError::ImuNominalRateMismatch { .. }
            | DatasetError::ImuStreamNotConfigured
            | DatasetError::LegacyImuManifestUnindexed
            | DatasetError::ManifestImuStreamWithoutMetadata
            | DatasetError::InvalidImuFileType { .. }
            | DatasetError::SidecarChanged { .. }
            | DatasetError::InvalidFrameFileType { .. }
            | DatasetError::InvalidFrameLength { .. }
            | DatasetError::InvalidManifest { .. }
            | DatasetError::UnsupportedManifestValue { .. }
            | DatasetError::NominalFpsMismatch { .. }
            | DatasetError::ManifestMetadataMismatch { .. }
            | DatasetError::ManifestPairingWindowOutOfRange { .. }
            | DatasetError::RecordedPairsDeclareOrphans { .. }
            | DatasetError::RecordedPairsContainMissingRight { .. }
            | DatasetError::InvalidManifestStats { .. }
            | DatasetError::ManifestDeltaStatsPresenceMismatch { .. }
            | DatasetError::ManifestFrameIdentityMismatch { .. }
            | DatasetError::DuplicateManifestFrameRef { .. }
            | DatasetError::NonMonotonicManifestEntries { .. }
            | DatasetError::NonMonotonicManifestDepthEntries { .. }
            | DatasetError::ManifestPairDeltaMismatch { .. }
            | DatasetError::ManifestPairOutsideWindow { .. }
            | DatasetError::ManifestCountOutOfRange { .. }
            | DatasetError::WorkerJoin { .. } => None,
        }
    }
}

#[derive(Debug)]
pub struct DatasetWriter {
    config: DatasetWriterConfig,
    state: Arc<WriterState>,
}

/// A dataset writer whose mono payloads can only be enqueued as validated stereo pairs.
#[derive(Clone, Debug)]
pub struct PairedDatasetWriter {
    inner: DatasetWriter,
    pairing_window: PairingWindowNs,
}

#[derive(Debug)]
pub struct DatasetWriterHandle {
    handle: Option<thread::JoinHandle<()>>,
    state: Arc<WriterState>,
}

impl Clone for DatasetWriter {
    fn clone(&self) -> Self {
        self.state.open_writers.fetch_add(1, Ordering::Relaxed);
        Self {
            config: self.config,
            state: Arc::clone(&self.state),
        }
    }
}

impl Drop for DatasetWriter {
    fn drop(&mut self) {
        if self.state.open_writers.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.state.close_spool();
        }
    }
}

impl DatasetWriter {
    pub fn create(
        path: impl Into<PathBuf>,
        meta: &Meta,
        calibration: &Calibration,
    ) -> Result<(Self, DatasetWriterHandle), DatasetError> {
        Self::create_internal(
            path,
            meta,
            calibration,
            DatasetWriterConfig::default(),
            ManifestMode::InferPairs,
            None,
        )
    }

    pub fn create_with_config(
        path: impl Into<PathBuf>,
        meta: &Meta,
        calibration: &Calibration,
        config: DatasetWriterConfig,
    ) -> Result<(Self, DatasetWriterHandle), DatasetError> {
        Self::create_internal(
            path,
            meta,
            calibration,
            config,
            ManifestMode::InferPairs,
            None,
        )
    }

    /// Create a writer with one explicit replayable OAK IMU session.
    pub fn create_with_imu_config(
        path: impl Into<PathBuf>,
        meta: &Meta,
        calibration: &Calibration,
        imu_metadata: ImuStreamMetadata,
        config: DatasetWriterConfig,
    ) -> Result<(Self, DatasetWriterHandle), DatasetError> {
        Self::create_internal(
            path,
            meta,
            calibration,
            config,
            ManifestMode::InferPairs,
            Some(imu_metadata),
        )
    }

    pub fn create_paired(
        path: impl Into<PathBuf>,
        meta: &Meta,
        calibration: &Calibration,
        pairing_window: PairingWindowNs,
    ) -> Result<(PairedDatasetWriter, DatasetWriterHandle), DatasetError> {
        Self::create_paired_with_config(
            path,
            meta,
            calibration,
            pairing_window,
            DatasetWriterConfig::default(),
        )
    }

    pub fn create_paired_with_config(
        path: impl Into<PathBuf>,
        meta: &Meta,
        calibration: &Calibration,
        pairing_window: PairingWindowNs,
        config: DatasetWriterConfig,
    ) -> Result<(PairedDatasetWriter, DatasetWriterHandle), DatasetError> {
        if config.max_spool_frames < 2 {
            return Err(DatasetError::InvalidConfig {
                msg: "paired writer max_spool_frames must be >= 2",
            });
        }
        let (inner, handle) = Self::create_internal(
            path,
            meta,
            calibration,
            config,
            ManifestMode::PreservePairs { pairing_window },
            None,
        )?;
        Ok((
            PairedDatasetWriter {
                inner,
                pairing_window,
            },
            handle,
        ))
    }

    /// Create a paired writer with one explicit replayable OAK IMU session.
    pub fn create_paired_with_imu_config(
        path: impl Into<PathBuf>,
        meta: &Meta,
        calibration: &Calibration,
        pairing_window: PairingWindowNs,
        imu_metadata: ImuStreamMetadata,
        config: DatasetWriterConfig,
    ) -> Result<(PairedDatasetWriter, DatasetWriterHandle), DatasetError> {
        if config.max_spool_frames < 2 {
            return Err(DatasetError::InvalidConfig {
                msg: "paired writer max_spool_frames must be >= 2",
            });
        }
        let (inner, handle) = Self::create_internal(
            path,
            meta,
            calibration,
            config,
            ManifestMode::PreservePairs { pairing_window },
            Some(imu_metadata),
        )?;
        Ok((
            PairedDatasetWriter {
                inner,
                pairing_window,
            },
            handle,
        ))
    }

    fn create_internal(
        path: impl Into<PathBuf>,
        meta: &Meta,
        calibration: &Calibration,
        config: DatasetWriterConfig,
        manifest_mode: ManifestMode,
        imu_metadata: Option<ImuStreamMetadata>,
    ) -> Result<(Self, DatasetWriterHandle), DatasetError> {
        if config.max_spool_frames == 0 {
            return Err(DatasetError::InvalidConfig {
                msg: "max_spool_frames must be > 0",
            });
        }
        if config.max_spool_bytes == 0 {
            return Err(DatasetError::InvalidConfig {
                msg: "max_spool_bytes must be > 0",
            });
        }
        if config.flush_batch_frames == 0 {
            return Err(DatasetError::InvalidConfig {
                msg: "flush_batch_frames must be > 0",
            });
        }

        let dataset_contract =
            WriterDatasetContract::parse_with_imu(meta, calibration, imu_metadata)?;
        let meta_json =
            serde_json::to_vec_pretty(meta).map_err(|source| DatasetError::SerializeJson {
                document: "metadata",
                source,
            })?;
        let calibration_json = serde_json::to_vec_pretty(calibration).map_err(|source| {
            DatasetError::SerializeJson {
                document: "calibration",
                source,
            }
        })?;

        let path = path.into();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).map_err(|e| DatasetError::CreateDirectory {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }
        match std::fs::create_dir(&path) {
            Ok(()) => {}
            Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(DatasetError::AlreadyExists { path });
            }
            Err(source) => {
                return Err(DatasetError::CreateDirectory {
                    path: path.clone(),
                    source,
                });
            }
        }

        let frames_dir = path.join(format::FRAMES_DIR);
        std::fs::create_dir_all(&frames_dir).map_err(|e| DatasetError::CreateDirectory {
            path: frames_dir.clone(),
            source: e,
        })?;

        let calibration_path = path.join(format::CALIBRATION_FILE);
        write_new_file(&calibration_path, &calibration_json)?;
        let meta_path = path.join(format::META_FILE);
        write_new_file(&meta_path, &meta_json)?;

        let state = Arc::new(WriterState::new(
            config,
            path.clone(),
            frames_dir.clone(),
            manifest_mode,
            dataset_contract,
            WriterSidecars {
                meta_json,
                calibration_json,
            },
        ));
        let state_for_thread = state.clone();

        let handle = thread::Builder::new()
            .name("dataset-writer".to_string())
            .spawn(move || writer_loop(frames_dir, state_for_thread))
            .map_err(|e| DatasetError::ThreadSpawn { source: e })?;

        let writer = Self {
            config,
            state: state.clone(),
        };

        let handle = DatasetWriterHandle {
            handle: Some(handle),
            state,
        };

        Ok((writer, handle))
    }

    /// Enqueue one frame according to the configured backpressure policy.
    ///
    /// Mono frames written through this legacy boundary are paired by timestamp when the manifest
    /// is finalized. Use [`DatasetWriter::create_paired`] and
    /// [`PairedDatasetWriter::write_pair`] to preserve an existing validated pair identity.
    /// A returned [`DatasetWriteError`] closes the writer; finalization reports the same contract
    /// class and does not publish a completion manifest.
    pub fn write_frame(&self, frame: &Frame) -> Result<WriteOutcome, DatasetWriteError> {
        if !self.is_healthy() {
            return Ok(WriteOutcome::WriterFailed);
        }
        if let Err(error) = self.state.dataset_contract.mono.require_frame(frame) {
            return Err(self.reject_write(error));
        }
        self.write_item(SpoolItem::Mono(frame.clone()))
    }

    /// Enqueue a depth image according to the configured backpressure policy.
    ///
    /// A returned [`DatasetWriteError`] closes the writer and prevents manifest publication.
    pub fn write_depth(&self, depth: &DepthImage) -> Result<WriteOutcome, DatasetWriteError> {
        if !self.is_healthy() {
            return Ok(WriteOutcome::WriterFailed);
        }
        let Some(contract) = self.state.dataset_contract.depth else {
            return Err(self.reject_write(DatasetWriteError::DepthStreamNotConfigured));
        };
        if let Err(error) = contract.require_image(depth) {
            return Err(self.reject_write(error));
        }
        self.write_item(SpoolItem::Depth(depth.clone()))
    }

    /// Enqueue one independently timestamped accelerometer/gyroscope report.
    pub fn write_imu(&self, report: ImuReport) -> Result<WriteOutcome, DatasetWriteError> {
        if !self.is_healthy() {
            return Ok(WriteOutcome::WriterFailed);
        }
        let Some(contract) = self.state.dataset_contract.imu.as_ref() else {
            return Err(self.reject_write(DatasetWriteError::ImuStreamNotConfigured));
        };
        if let Err(error) = contract.require_report(&report) {
            return Err(self.reject_write(error));
        }
        self.write_item(SpoolItem::Imu(report))
    }

    fn write_item(&self, item: SpoolItem) -> Result<WriteOutcome, DatasetWriteError> {
        if self.state.failed.load(Ordering::Acquire) {
            return Ok(WriteOutcome::WriterFailed);
        }

        let frames = item.frame_count();
        let bytes = item.bytes_len();
        if frames > self.config.max_spool_frames {
            return Err(
                self.reject_write(DatasetWriteError::SpoolFrameCapacityExceeded {
                    item: item.kind(),
                    frames,
                    max_frames: self.config.max_spool_frames,
                }),
            );
        }
        if bytes > self.config.max_spool_bytes {
            return Err(
                self.reject_write(DatasetWriteError::SpoolByteCapacityExceeded {
                    item: item.kind(),
                    bytes,
                    max_bytes: self.config.max_spool_bytes,
                }),
            );
        }

        let mut spool = self
            .state
            .spool
            .lock()
            .unwrap_or_else(|err| err.into_inner());

        match self.config.backpressure {
            Backpressure::DropNewest => {
                if spool.closed || self.state.failed.load(Ordering::Acquire) {
                    return Ok(WriteOutcome::WriterFailed);
                }
                if !self.state.can_accept(&spool, frames, bytes) {
                    self.state
                        .dropped
                        .fetch_add(frames as u64, Ordering::Relaxed);
                    self.state
                        .bytes_dropped
                        .fetch_add(bytes as u64, Ordering::Relaxed);
                    return Ok(WriteOutcome::Dropped);
                }
            }
            Backpressure::Block => {
                while !self.state.can_accept(&spool, frames, bytes) {
                    if spool.closed || self.state.failed.load(Ordering::Acquire) {
                        return Ok(WriteOutcome::WriterFailed);
                    }
                    spool = self
                        .state
                        .spool_cvar
                        .wait(spool)
                        .unwrap_or_else(|err| err.into_inner());
                }
            }
        }

        if spool.closed || self.state.failed.load(Ordering::Acquire) {
            return Ok(WriteOutcome::WriterFailed);
        }

        spool.frames += frames;
        spool.bytes += bytes;
        spool.queue.push_back(item);

        self.state
            .enqueued
            .fetch_add(frames as u64, Ordering::Relaxed);
        self.state
            .bytes_enqueued
            .fetch_add(bytes as u64, Ordering::Relaxed);
        self.state
            .spool_frames
            .store(spool.frames as u64, Ordering::Relaxed);
        self.state
            .spool_bytes
            .store(spool.bytes as u64, Ordering::Relaxed);
        self.state.spool_cvar.notify_one();
        Ok(WriteOutcome::Enqueued)
    }

    fn reject_write(&self, error: DatasetWriteError) -> DatasetWriteError {
        self.state
            .fail(DatasetError::WriteContract { source: error });
        self.state.cancel_unwritten(std::iter::empty());
        error
    }

    pub fn stats(&self) -> WriterStats {
        self.state.stats()
    }

    pub fn is_healthy(&self) -> bool {
        !self.state.failed.load(Ordering::Acquire)
    }
}

impl PairedDatasetWriter {
    /// Enqueue both payloads as one spool item.
    ///
    /// A successful outcome means both frames were accepted together. The manifest pair is only
    /// published after both payload writes succeed. If the second payload write fails, the first
    /// payload can remain on disk unreferenced and `DatasetWriterHandle::finish` reports the typed
    /// I/O error. A returned [`DatasetWriteError`] closes the writer and prevents manifest
    /// publication.
    pub fn write_pair(&self, pair: StereoPair) -> Result<WriteOutcome, DatasetWriteError> {
        if !self.inner.is_healthy() {
            return Ok(WriteOutcome::WriterFailed);
        }
        let delta_ns = pair.timestamp_delta_ns();
        let max_delta_ns = self.pairing_window.as_u64();
        if delta_ns > max_delta_ns {
            return Err(self
                .inner
                .reject_write(DatasetWriteError::PairOutsideWriterWindow {
                    delta_ns,
                    max_delta_ns,
                }));
        }
        if let Err(error) = self
            .inner
            .state
            .dataset_contract
            .mono
            .require_frame(pair.left())
        {
            return Err(self.inner.reject_write(error));
        }
        self.inner.write_item(SpoolItem::Pair(pair))
    }

    pub fn write_depth(&self, depth: &DepthImage) -> Result<WriteOutcome, DatasetWriteError> {
        self.inner.write_depth(depth)
    }

    pub fn write_imu(&self, report: ImuReport) -> Result<WriteOutcome, DatasetWriteError> {
        self.inner.write_imu(report)
    }

    pub fn stats(&self) -> WriterStats {
        self.inner.stats()
    }

    pub fn is_healthy(&self) -> bool {
        self.inner.is_healthy()
    }
}

impl DatasetWriterHandle {
    /// Blocks until the writer thread exits; all writer clones must be dropped first.
    pub fn finish(mut self) -> Result<WriterStats, DatasetError> {
        let Some(handle) = self.handle.take() else {
            return Err(DatasetError::InvalidConfig {
                msg: "finish called twice",
            });
        };
        handle.join().map_err(|err| DatasetError::WorkerJoin {
            message: panic_message(err),
        })?;

        if let Some(err) = self.state.take_error() {
            return Err(err);
        }

        write_manifest(&self.state)?;
        Ok(self.state.stats())
    }

    pub fn stats(&self) -> WriterStats {
        self.state.stats()
    }
}

/// One atomic spool transaction.
///
/// Mono, depth, and IMU variants account for one logical payload unit; a pair
/// accounts for two. Payload bytes exclude container/header overhead.
#[derive(Debug)]
enum SpoolItem {
    Mono(Frame),
    Pair(StereoPair),
    Depth(DepthImage),
    Imu(ImuReport),
}

impl SpoolItem {
    fn frame_count(&self) -> usize {
        match self {
            SpoolItem::Mono(_) | SpoolItem::Depth(_) | SpoolItem::Imu(_) => 1,
            SpoolItem::Pair(_) => 2,
        }
    }

    fn bytes_len(&self) -> usize {
        match self {
            SpoolItem::Mono(frame) => frame.data().len(),
            SpoolItem::Pair(pair) => pair
                .left()
                .data()
                .len()
                .saturating_add(pair.right().data().len()),
            SpoolItem::Depth(depth) => depth
                .depth_m()
                .len()
                .saturating_mul(std::mem::size_of::<f32>()),
            SpoolItem::Imu(_) => usize::from(IMU_STREAM_RECORD_BYTES),
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            SpoolItem::Mono(_) => "mono frame",
            SpoolItem::Pair(_) => "stereo pair",
            SpoolItem::Depth(_) => "depth image",
            SpoolItem::Imu(_) => "IMU report",
        }
    }
}

#[derive(Debug)]
struct Spool {
    queue: VecDeque<SpoolItem>,
    frames: usize,
    bytes: usize,
    closed: bool,
}

impl Spool {
    fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            frames: 0,
            bytes: 0,
            closed: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum ManifestMode {
    InferPairs,
    PreservePairs { pairing_window: PairingWindowNs },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MonoPayloadFormat {
    RawGray8,
}

impl MonoPayloadFormat {
    fn as_str(self) -> &'static str {
        match self {
            Self::RawGray8 => "raw",
        }
    }

    fn parse(value: &str) -> Result<Self, DatasetError> {
        match value {
            "raw" => Ok(Self::RawGray8),
            _ => Err(DatasetError::UnsupportedManifestValue {
                field: "header.format",
                value: value.to_string(),
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DatasetTimebase {
    DeviceNs,
}

impl DatasetTimebase {
    fn as_str(self) -> &'static str {
        match self {
            Self::DeviceNs => "device_ns",
        }
    }

    fn parse(value: &str) -> Result<Self, DatasetError> {
        match value {
            "device_ns" => Ok(Self::DeviceNs),
            _ => Err(DatasetError::UnsupportedManifestValue {
                field: "header.timebase",
                value: value.to_string(),
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ManifestPairingPolicy {
    TimeSymmetric,
    RecordedPairs,
}

impl ManifestPairingPolicy {
    fn as_str(self) -> &'static str {
        match self {
            Self::TimeSymmetric => "time_symmetric",
            Self::RecordedPairs => "recorded_pairs",
        }
    }

    fn parse(value: &str) -> Result<Self, DatasetError> {
        match value {
            "time_symmetric" => Ok(Self::TimeSymmetric),
            "recorded_pairs" => Ok(Self::RecordedPairs),
            _ => Err(DatasetError::UnsupportedManifestValue {
                field: "header.pairing_policy",
                value: value.to_string(),
            }),
        }
    }
}

#[derive(Debug)]
struct WriterState {
    config: DatasetWriterConfig,
    manifest_mode: ManifestMode,
    dataset_contract: WriterDatasetContract,
    sidecars: WriterSidecars,
    dataset_dir: PathBuf,
    frames_dir: PathBuf,
    spool: Mutex<Spool>,
    spool_cvar: Condvar,
    enqueued: AtomicU64,
    written: AtomicU64,
    dropped: AtomicU64,
    bytes_enqueued: AtomicU64,
    bytes_written: AtomicU64,
    bytes_dropped: AtomicU64,
    write_failed: AtomicU64,
    bytes_write_failed: AtomicU64,
    frames_canceled: AtomicU64,
    bytes_canceled: AtomicU64,
    spool_frames: AtomicU64,
    spool_bytes: AtomicU64,
    open_writers: AtomicUsize,
    failed: AtomicBool,
    error: Mutex<Option<DatasetError>>,
    recorded_pairs: Mutex<Vec<RecordedPair>>,
    recorded_depth: Mutex<Vec<RecordedDepth>>,
    recorded_imu: Mutex<Option<RecordedImuStream>>,
}

impl WriterState {
    fn new(
        config: DatasetWriterConfig,
        dataset_dir: PathBuf,
        frames_dir: PathBuf,
        manifest_mode: ManifestMode,
        dataset_contract: WriterDatasetContract,
        sidecars: WriterSidecars,
    ) -> Self {
        Self {
            config,
            manifest_mode,
            dataset_contract,
            sidecars,
            dataset_dir,
            frames_dir,
            spool: Mutex::new(Spool::new()),
            spool_cvar: Condvar::new(),
            enqueued: AtomicU64::new(0),
            written: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            bytes_enqueued: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
            bytes_dropped: AtomicU64::new(0),
            write_failed: AtomicU64::new(0),
            bytes_write_failed: AtomicU64::new(0),
            frames_canceled: AtomicU64::new(0),
            bytes_canceled: AtomicU64::new(0),
            spool_frames: AtomicU64::new(0),
            spool_bytes: AtomicU64::new(0),
            open_writers: AtomicUsize::new(1),
            failed: AtomicBool::new(false),
            error: Mutex::new(None),
            recorded_pairs: Mutex::new(Vec::new()),
            recorded_depth: Mutex::new(Vec::new()),
            recorded_imu: Mutex::new(None),
        }
    }

    fn can_accept(&self, spool: &Spool, frames: usize, bytes: usize) -> bool {
        let next_frames = spool.frames.saturating_add(frames);
        let next_bytes = spool.bytes.saturating_add(bytes);
        next_frames <= self.config.max_spool_frames && next_bytes <= self.config.max_spool_bytes
    }

    fn close_spool(&self) {
        let mut spool = self.spool.lock().unwrap_or_else(|err| err.into_inner());
        spool.closed = true;
        self.spool_cvar.notify_all();
    }

    fn fail(&self, err: DatasetError) {
        self.failed.store(true, Ordering::Release);
        self.record_error(err);
        self.close_spool();
    }

    fn stats(&self) -> WriterStats {
        WriterStats {
            frames_enqueued: self.enqueued.load(Ordering::Relaxed),
            frames_written: self.written.load(Ordering::Relaxed),
            frames_dropped: self.dropped.load(Ordering::Relaxed),
            bytes_enqueued: self.bytes_enqueued.load(Ordering::Relaxed),
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
            bytes_dropped: self.bytes_dropped.load(Ordering::Relaxed),
            write_failed: self.write_failed.load(Ordering::Relaxed),
            bytes_write_failed: self.bytes_write_failed.load(Ordering::Relaxed),
            frames_canceled: self.frames_canceled.load(Ordering::Relaxed),
            bytes_canceled: self.bytes_canceled.load(Ordering::Relaxed),
            spool_frames: self.spool_frames.load(Ordering::Relaxed),
            spool_bytes: self.spool_bytes.load(Ordering::Relaxed),
            spool_max_frames: self.config.max_spool_frames as u64,
            spool_max_bytes: self.config.max_spool_bytes as u64,
            writer_failed: self.failed.load(Ordering::Acquire),
        }
    }

    fn record_error(&self, err: DatasetError) {
        let mut guard = self.error.lock().unwrap_or_else(|err| err.into_inner());
        if guard.is_none() {
            *guard = Some(err);
        }
    }

    fn take_error(&self) -> Option<DatasetError> {
        self.error
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .take()
    }

    fn record_pair(&self, pair: RecordedPair) {
        self.recorded_pairs
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .push(pair);
    }

    fn record_depth(&self, depth: RecordedDepth) {
        self.recorded_depth
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .push(depth);
    }

    fn record_imu(&self, imu: RecordedImuStream) {
        *self
            .recorded_imu
            .lock()
            .unwrap_or_else(|err| err.into_inner()) = Some(imu);
    }

    fn cancel_unwritten(&self, in_flight: impl IntoIterator<Item = SpoolItem>) {
        let mut canceled_frames = 0u64;
        let mut canceled_bytes = 0u64;
        for item in in_flight {
            canceled_frames = canceled_frames.saturating_add(item.frame_count() as u64);
            canceled_bytes = canceled_bytes.saturating_add(item.bytes_len() as u64);
        }

        let mut spool = self.spool.lock().unwrap_or_else(|err| err.into_inner());
        while let Some(item) = spool.queue.pop_front() {
            canceled_frames = canceled_frames.saturating_add(item.frame_count() as u64);
            canceled_bytes = canceled_bytes.saturating_add(item.bytes_len() as u64);
        }
        spool.frames = 0;
        spool.bytes = 0;
        self.spool_frames.store(0, Ordering::Relaxed);
        self.spool_bytes.store(0, Ordering::Relaxed);
        self.frames_canceled
            .fetch_add(canceled_frames, Ordering::Relaxed);
        self.bytes_canceled
            .fetch_add(canceled_bytes, Ordering::Relaxed);
        self.spool_cvar.notify_all();
    }

    fn record_written(&self, frames: usize, bytes: usize) {
        self.written.fetch_add(frames as u64, Ordering::Relaxed);
        self.bytes_written
            .fetch_add(bytes as u64, Ordering::Relaxed);

        let mut spool = self.spool.lock().unwrap_or_else(|err| err.into_inner());
        spool.frames = spool.frames.saturating_sub(frames);
        spool.bytes = spool.bytes.saturating_sub(bytes);
        self.spool_frames
            .store(spool.frames as u64, Ordering::Relaxed);
        self.spool_bytes
            .store(spool.bytes as u64, Ordering::Relaxed);
        self.spool_cvar.notify_all();
    }
}

struct ImuFileWriter {
    file: std::fs::File,
    temporary_path: PathBuf,
    stream_path: PathBuf,
    record_count: u64,
    order: InertialOrderTracker,
    published: bool,
}

impl ImuFileWriter {
    fn create(dataset_dir: &Path, session_id: DeviceSessionId) -> Result<Self, DatasetError> {
        let temporary_path = dataset_dir.join(format::IMU_STREAM_TEMP_FILE);
        let stream_path = dataset_dir.join(format::IMU_STREAM_FILE);
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&temporary_path)
            .map_err(|source| DatasetError::WriteFile {
                path: temporary_path.clone(),
                source,
            })?;
        if let Err(source) = file.write_all(&encode_imu_header(0)) {
            drop(file);
            let _ = std::fs::remove_file(&temporary_path);
            return Err(DatasetError::WriteFile {
                path: temporary_path,
                source,
            });
        }
        Ok(Self {
            file,
            temporary_path,
            stream_path,
            record_count: 0,
            order: InertialOrderTracker::with_session(session_id),
            published: false,
        })
    }

    fn write_report(&mut self, report: ImuReport) -> Result<(), DatasetError> {
        let record_index =
            usize::try_from(self.record_count).map_err(|_| DatasetError::InvalidImuStream {
                path: self.temporary_path.clone(),
                source: ImuStreamError::RecordIndexOutOfRange {
                    record_count: self.record_count,
                },
            })?;
        let next_record_count =
            self.record_count
                .checked_add(1)
                .ok_or_else(|| DatasetError::InvalidImuStream {
                    path: self.temporary_path.clone(),
                    source: ImuStreamError::ByteLengthOverflow {
                        record_count: self.record_count,
                    },
                })?;
        imu_stream_byte_len(next_record_count).map_err(|source| {
            DatasetError::InvalidImuStream {
                path: self.temporary_path.clone(),
                source,
            }
        })?;
        self.order
            .observe(&report)
            .map_err(|source| DatasetError::InvalidImuStream {
                path: self.temporary_path.clone(),
                source: map_inertial_order_error(record_index, source),
            })?;
        let bytes = ImuWireRecord::from_report(report).encode();
        self.file
            .write_all(&bytes)
            .map_err(|source| DatasetError::WriteFile {
                path: self.temporary_path.clone(),
                source,
            })?;
        self.record_count = next_record_count;
        Ok(())
    }

    fn finish(mut self) -> Result<RecordedImuStream, DatasetError> {
        let byte_len = imu_stream_byte_len(self.record_count).map_err(|source| {
            DatasetError::InvalidImuStream {
                path: self.temporary_path.clone(),
                source,
            }
        })?;
        self.file
            .seek(SeekFrom::Start(0))
            .and_then(|_| self.file.write_all(&encode_imu_header(self.record_count)))
            .and_then(|_| self.file.sync_all())
            .map_err(|source| DatasetError::WriteFile {
                path: self.temporary_path.clone(),
                source,
            })?;
        std::fs::rename(&self.temporary_path, &self.stream_path).map_err(|source| {
            DatasetError::PublishImuStream {
                temporary_path: self.temporary_path.clone(),
                stream_path: self.stream_path.clone(),
                source,
            }
        })?;
        self.published = true;
        Ok(RecordedImuStream {
            record_count: self.record_count,
            byte_len,
        })
    }
}

impl Drop for ImuFileWriter {
    fn drop(&mut self) {
        if !self.published {
            let _ = std::fs::remove_file(&self.temporary_path);
        }
    }
}

fn writer_loop(frames_dir: PathBuf, state: Arc<WriterState>) {
    let mut imu_writer = match state.dataset_contract.imu.as_ref() {
        Some(contract) => {
            match ImuFileWriter::create(&state.dataset_dir, contract.metadata.device_session_id) {
                Ok(writer) => Some(writer),
                Err(error) => {
                    state.fail(error);
                    state.cancel_unwritten(std::iter::empty());
                    return;
                }
            }
        }
        None => None,
    };
    writer_loop_with(Arc::clone(&state), |item| match item {
        SpoolItem::Imu(report) => {
            let Some(writer) = imu_writer.as_mut() else {
                return Err(DatasetError::InvalidConfig {
                    msg: "IMU spool item reached an unconfigured stream writer",
                });
            };
            writer.write_report(report)?;
            Ok(WrittenManifestItem::None)
        }
        item => write_item_to_dir(&frames_dir, item),
    });
    if state.failed.load(Ordering::Acquire) {
        return;
    }
    if let Some(writer) = imu_writer {
        match writer.finish() {
            Ok(recorded) => state.record_imu(recorded),
            Err(error) => state.fail(error),
        }
    }
}

fn writer_loop_with(
    state: Arc<WriterState>,
    mut write_item: impl FnMut(SpoolItem) -> Result<WrittenManifestItem, DatasetError>,
) {
    loop {
        let batch = {
            let mut spool = state.spool.lock().unwrap_or_else(|err| err.into_inner());
            while spool.queue.is_empty() && !spool.closed {
                spool = state
                    .spool_cvar
                    .wait(spool)
                    .unwrap_or_else(|err| err.into_inner());
            }

            if spool.queue.is_empty() && spool.closed {
                break;
            }

            let mut batch = Vec::new();
            let mut batch_frames = 0usize;
            while let Some(item) = spool.queue.pop_front() {
                let frames = item.frame_count();
                batch_frames = batch_frames.saturating_add(frames);
                batch.push(item);
                if batch_frames >= state.config.flush_batch_frames {
                    break;
                }
            }
            batch
        };

        let mut batch = batch.into_iter();
        while let Some(item) = batch.next() {
            if state.failed.load(Ordering::Acquire) {
                state.cancel_unwritten(std::iter::once(item).chain(batch));
                return;
            }
            let frames = item.frame_count();
            let bytes = item.bytes_len();
            let written_manifest_item = match write_item(item) {
                Ok(written_manifest_item) => written_manifest_item,
                Err(err) => {
                    state
                        .write_failed
                        .fetch_add(frames as u64, Ordering::Relaxed);
                    state
                        .bytes_write_failed
                        .fetch_add(bytes as u64, Ordering::Relaxed);
                    state.fail(err);
                    state.cancel_unwritten(batch);
                    return;
                }
            };
            match written_manifest_item {
                WrittenManifestItem::None => {}
                WrittenManifestItem::Pair(pair) => state.record_pair(pair),
                WrittenManifestItem::Depth(depth) => state.record_depth(depth),
            }
            state.record_written(frames, bytes);
        }
    }
}

fn write_item_to_dir(
    frames_dir: &Path,
    item: SpoolItem,
) -> Result<WrittenManifestItem, DatasetError> {
    match item {
        SpoolItem::Mono(frame) => {
            write_frame_to_dir(frames_dir, frame)?;
            Ok(WrittenManifestItem::None)
        }
        SpoolItem::Pair(pair) => write_pair_to_dir(frames_dir, pair).map(WrittenManifestItem::Pair),
        SpoolItem::Depth(depth) => {
            write_depth_to_dir(frames_dir, depth).map(WrittenManifestItem::Depth)
        }
        SpoolItem::Imu(_) => Err(DatasetError::InvalidConfig {
            msg: "IMU stream requires the transactional stream writer",
        }),
    }
}

fn write_pair_to_dir(frames_dir: &Path, pair: StereoPair) -> Result<RecordedPair, DatasetError> {
    let left_timestamp_ns = pair.left().timestamp().as_nanos();
    let right_timestamp_ns = pair.right().timestamp().as_nanos();
    let delta_ns = pair.timestamp_delta_ns();
    let (left_frame, right_frame) = pair.into_parts();
    write_frame_to_dir(frames_dir, left_frame)?;
    write_frame_to_dir(frames_dir, right_frame)?;
    Ok(RecordedPair {
        left_timestamp_ns,
        right_timestamp_ns,
        delta_ns,
    })
}

fn write_frame_to_dir(frames_dir: &Path, frame: Frame) -> Result<(), DatasetError> {
    let Frame {
        sensor_id,
        frame_id: _,
        timestamp,
        dimensions: _,
        data,
    } = frame;
    let filename = format::frame_name(timestamp.as_nanos(), sensor_to_str(sensor_id));
    let path = frames_dir.join(&filename);

    write_new_file(&path, data.as_ref())?;

    Ok(())
}

fn write_depth_to_dir(frames_dir: &Path, depth: DepthImage) -> Result<RecordedDepth, DatasetError> {
    let timestamp_ns = depth.timestamp().as_nanos();
    let filename = format::frame_name(timestamp_ns, "depth");
    let path = frames_dir.join(&filename);
    let mut bytes = Vec::with_capacity(
        depth
            .depth_m()
            .len()
            .saturating_mul(std::mem::size_of::<f32>()),
    );
    for value in depth.depth_m() {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    write_new_file(&path, &bytes)?;
    Ok(RecordedDepth { timestamp_ns })
}

fn write_new_file(path: &Path, bytes: &[u8]) -> Result<(), DatasetError> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|source| DatasetError::WriteFile {
            path: path.to_path_buf(),
            source,
        })?;
    file.write_all(bytes)
        .map_err(|source| DatasetError::WriteFile {
            path: path.to_path_buf(),
            source,
        })
}

fn validate_payload_file(path: &Path, expected: u64) -> Result<(), DatasetError> {
    let metadata = std::fs::metadata(path).map_err(|source| DatasetError::ReadFile {
        path: path.to_path_buf(),
        source,
    })?;
    if !metadata.is_file() {
        return Err(DatasetError::InvalidFrameFileType {
            path: path.to_path_buf(),
        });
    }
    let actual = metadata.len();
    if actual != expected {
        return Err(DatasetError::InvalidFrameLength {
            path: path.to_path_buf(),
            expected,
            actual,
        });
    }
    Ok(())
}

fn publish_manifest(dataset_dir: &Path, bytes: &[u8]) -> Result<(), DatasetError> {
    const TEMP_FILE: &str = ".manifest.json.tmp";

    let temporary_path = dataset_dir.join(TEMP_FILE);
    let manifest_path = dataset_dir.join(format::MANIFEST_FILE);
    let mut temporary_created = false;
    let result = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary_path)
            .map_err(|source| DatasetError::WriteFile {
                path: temporary_path.clone(),
                source,
            })?;
        temporary_created = true;
        file.write_all(bytes)
            .map_err(|source| DatasetError::WriteFile {
                path: temporary_path.clone(),
                source,
            })?;
        file.sync_all().map_err(|source| DatasetError::WriteFile {
            path: temporary_path.clone(),
            source,
        })?;
        drop(file);
        std::fs::rename(&temporary_path, &manifest_path).map_err(|source| {
            DatasetError::PublishManifest {
                temporary_path: temporary_path.clone(),
                manifest_path: manifest_path.clone(),
                source,
            }
        })
    })();

    if result.is_err() && temporary_created {
        let _ = std::fs::remove_file(&temporary_path);
    }
    result
}

fn sensor_to_str(id: SensorId) -> &'static str {
    match id {
        SensorId::StereoLeft => "mono_left",
        SensorId::StereoRight => "mono_right",
    }
}

fn panic_message(err: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = err.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = err.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic".to_string()
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Manifest {
    header: ManifestHeader,
    stats: ManifestStats,
    entries: Vec<ManifestEntry>,
    /// `None` is the legacy/unconfigured representation; `Some([])` records a configured stream
    /// that completed without writing any depth payloads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    depth_entries: Option<Vec<ManifestFrameRef>>,
    /// `None` preserves manifests written before fixed-record IMU replay existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    imu_stream: Option<ManifestImuStream>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ManifestImuStream {
    path: String,
    format: String,
    version: u16,
    header_bytes: u16,
    record_bytes: u16,
    record_count: u64,
    byte_len: u64,
    nominal_rate_hz: u32,
    acceleration_unit: String,
    angular_velocity_unit: String,
    coordinate_frame: String,
    axes: String,
    extrinsic: ManifestImuExtrinsic,
    device_session_id: u64,
    device_timebase: String,
    device_session_semantics: String,
    sample_timestamp_semantics: String,
    host_arrival_timebase: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum ManifestImuExtrinsic {
    UncalibratedUnknown,
    CalibratedToTrackingCamera { source: String },
}

#[derive(Debug, Serialize, Deserialize)]
struct ManifestHeader {
    dataset_id: String,
    created_at: String,
    device: String,
    format: String,
    width: u32,
    height: u32,
    fps: u32,
    timebase: String,
    pairing_policy: String,
    pairing_window_ns: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct ManifestStats {
    total_left: u64,
    total_right: u64,
    paired_count: u64,
    left_orphans: u64,
    right_orphans: u64,
    drops_by_reason: DropStats,
    delta_stats: Option<DeltaStats>,
}

#[derive(Debug, Serialize, Deserialize)]
/// Diagnostics from distinct pipeline stages, not a partition of mono manifest entries.
struct DropStats {
    /// Logical frames of any payload kind rejected by spool backpressure.
    spool_full: u64,
    /// Reserved write-failure diagnostic; successful publication requires this to be zero.
    write_fail: u64,
    /// Filesystem entries whose filenames could not be parsed canonically.
    parse_fail: u64,
    /// Canonically named payload files whose byte length disagreed with metadata.
    size_mismatch: u64,
    /// Explicit left entries that had no right frame inside the published window.
    outside_window: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct DeltaStats {
    min: u64,
    median: u64,
    p95: u64,
    p99: u64,
    max: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ManifestEntry {
    left: ManifestFrameRef,
    #[serde(flatten)]
    pairing: ManifestPairing,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ManifestFrameRef {
    timestamp_ns: i64,
    path: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "status", rename_all = "snake_case")]
enum ManifestPairing {
    Paired {
        right: ManifestFrameRef,
        delta_ns: u64,
    },
    MissingRight {
        #[serde(default = "default_missing_right_reason")]
        reason: PairReason,
    },
}

fn default_missing_right_reason() -> PairReason {
    PairReason::NoRightFrames
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum PairReason {
    OutsideWindow,
    NoRightFrames,
    RightExhausted,
}

#[derive(Debug, Clone)]
struct FrameInfo {
    timestamp_ns: i64,
    path: String,
}

#[derive(Debug, Clone)]
struct RecordedPair {
    left_timestamp_ns: i64,
    right_timestamp_ns: i64,
    delta_ns: u64,
}

#[derive(Clone, Copy, Debug)]
struct RecordedDepth {
    timestamp_ns: i64,
}

#[derive(Clone, Copy, Debug)]
struct RecordedImuStream {
    record_count: u64,
    byte_len: u64,
}

#[derive(Debug)]
enum WrittenManifestItem {
    None,
    Pair(RecordedPair),
    Depth(RecordedDepth),
}

#[derive(Debug)]
struct FrameSet {
    left: Vec<FrameInfo>,
    right: Vec<FrameInfo>,
    parse_fail: u64,
    size_mismatch: u64,
}

fn write_manifest(state: &WriterState) -> Result<(), DatasetError> {
    state.sidecars.require_unchanged(&state.dataset_dir)?;
    let contract = &state.dataset_contract;
    let mono = contract.mono;
    let dimensions = mono.dimensions();

    let FrameSet {
        mut left,
        mut right,
        parse_fail,
        size_mismatch,
    } = scan_frames_with_depth(&state.frames_dir, dimensions, contract.depth.as_ref())?;

    left.sort_by_key(|f| f.timestamp_ns);
    right.sort_by_key(|f| f.timestamp_ns);

    let topology = match state.manifest_mode {
        ManifestMode::InferPairs => inferred_manifest_topology(&left, &right),
        ManifestMode::PreservePairs { pairing_window } => {
            let recorded_pairs = state
                .recorded_pairs
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .clone();
            recorded_manifest_topology(
                &state.dataset_dir,
                &recorded_pairs,
                pairing_window,
                dimensions,
            )?
        }
    };

    let depth_entries = match contract.depth {
        Some(depth_contract) => {
            let recorded_depth = state
                .recorded_depth
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .clone();
            Some(recorded_depth_entries(
                &state.dataset_dir,
                &recorded_depth,
                depth_contract,
            )?)
        }
        None => None,
    };

    let imu_stream = match contract.imu.as_ref() {
        Some(imu_contract) => {
            let recorded = state
                .recorded_imu
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .ok_or(DatasetError::InvalidManifest {
                    reason: "configured IMU stream was not finalized",
                })?;
            Some(imu_contract.manifest(recorded))
        }
        None => None,
    };

    let manifest = Manifest {
        header: ManifestHeader {
            dataset_id: dataset_id(&state.dataset_dir),
            created_at: contract.created_at.clone(),
            device: contract.device.clone(),
            format: MonoPayloadFormat::RawGray8.as_str().to_string(),
            width: dimensions.width(),
            height: dimensions.height(),
            fps: mono.nominal_fps_hz().get(),
            timebase: DatasetTimebase::DeviceNs.as_str().to_string(),
            pairing_policy: topology.pairing_policy.as_str().to_string(),
            pairing_window_ns: topology.pairing_window_ns,
        },
        stats: ManifestStats {
            total_left: topology.total_left,
            total_right: topology.total_right,
            paired_count: topology.paired_count,
            left_orphans: topology.left_orphans,
            right_orphans: topology.right_orphans,
            drops_by_reason: DropStats {
                spool_full: state.dropped.load(Ordering::Relaxed),
                write_fail: state.write_failed.load(Ordering::Relaxed),
                parse_fail,
                size_mismatch,
                outside_window: topology.outside_window,
            },
            delta_stats: topology.delta_stats,
        },
        entries: topology.entries,
        depth_entries,
        imu_stream,
    };

    let bytes =
        serde_json::to_vec_pretty(&manifest).map_err(|source| DatasetError::SerializeJson {
            document: "manifest",
            source,
        })?;
    publish_manifest(&state.dataset_dir, &bytes)
}

fn recorded_depth_entries(
    dataset_dir: &Path,
    recorded_depth: &[RecordedDepth],
    contract: DepthImageContract,
) -> Result<Vec<ManifestFrameRef>, DatasetError> {
    let mut recorded_depth = recorded_depth.to_vec();
    recorded_depth.sort_by_key(|depth| depth.timestamp_ns);

    let mut entries = Vec::with_capacity(recorded_depth.len());
    for depth in recorded_depth {
        let frame_ref = manifest_frame_ref(depth.timestamp_ns, "depth");
        let path = dataset_dir.join(&frame_ref.path);
        if entries.last().is_some_and(|previous: &ManifestFrameRef| {
            previous.timestamp_ns == frame_ref.timestamp_ns
        }) {
            return Err(DatasetError::DuplicateManifestFrameRef { path });
        }
        validate_payload_file(&path, contract.expected_payload_len())?;
        entries.push(frame_ref);
    }
    Ok(entries)
}

struct ManifestTopology {
    pairing_policy: ManifestPairingPolicy,
    pairing_window_ns: u64,
    entries: Vec<ManifestEntry>,
    total_left: u64,
    total_right: u64,
    paired_count: u64,
    left_orphans: u64,
    right_orphans: u64,
    outside_window: u64,
    delta_stats: Option<DeltaStats>,
}

fn inferred_manifest_topology(left: &[FrameInfo], right: &[FrameInfo]) -> ManifestTopology {
    let left_period = compute_period_ns(left);
    let gate = left_period
        .map(|period| period / 4)
        .filter(|period| *period > 0);
    let mut estimator_deltas = collect_deltas(left, right, gate);
    let estimator_stats = build_delta_stats(&estimator_deltas);
    let pairing_window_ns =
        compute_pairing_window_ns(&estimator_deltas, estimator_stats.as_ref(), left_period);
    let (entries, paired_count, left_orphans, right_orphans, outside_window) =
        pair_entries(left, right, pairing_window_ns);
    estimator_deltas.clear();
    estimator_deltas.extend(entries.iter().filter_map(|entry| match &entry.pairing {
        ManifestPairing::Paired { delta_ns, .. } => Some(*delta_ns),
        ManifestPairing::MissingRight { .. } => None,
    }));
    let delta_stats = build_delta_stats(&estimator_deltas);
    ManifestTopology {
        pairing_policy: ManifestPairingPolicy::TimeSymmetric,
        pairing_window_ns,
        entries,
        total_left: left.len() as u64,
        total_right: right.len() as u64,
        paired_count,
        left_orphans,
        right_orphans,
        outside_window,
        delta_stats,
    }
}

fn recorded_manifest_topology(
    dataset_dir: &Path,
    recorded_pairs: &[RecordedPair],
    pairing_window: PairingWindowNs,
    dimensions: FrameDimensions,
) -> Result<ManifestTopology, DatasetError> {
    let expected_len = dimensions.area() as u64;
    for pair in recorded_pairs {
        for (timestamp_ns, sensor) in [
            (pair.left_timestamp_ns, "mono_left"),
            (pair.right_timestamp_ns, "mono_right"),
        ] {
            let path = dataset_dir.join(frame_path(timestamp_ns, sensor));
            validate_payload_file(&path, expected_len)?;
        }
    }

    let mut entries: Vec<ManifestEntry> = recorded_pairs
        .iter()
        .map(|pair| ManifestEntry {
            left: manifest_frame_ref(pair.left_timestamp_ns, "mono_left"),
            pairing: ManifestPairing::Paired {
                right: manifest_frame_ref(pair.right_timestamp_ns, "mono_right"),
                delta_ns: pair.delta_ns,
            },
        })
        .collect();
    entries.sort_by_key(|entry| entry.left.timestamp_ns);

    let deltas: Vec<u64> = recorded_pairs.iter().map(|pair| pair.delta_ns).collect();
    let pair_count = recorded_pairs.len() as u64;
    Ok(ManifestTopology {
        pairing_policy: ManifestPairingPolicy::RecordedPairs,
        pairing_window_ns: pairing_window.as_u64(),
        entries,
        total_left: pair_count,
        total_right: pair_count,
        paired_count: pair_count,
        left_orphans: 0,
        right_orphans: 0,
        outside_window: 0,
        delta_stats: build_delta_stats(&deltas),
    })
}

fn manifest_frame_ref(timestamp_ns: i64, sensor: &str) -> ManifestFrameRef {
    ManifestFrameRef {
        timestamp_ns,
        path: frame_path(timestamp_ns, sensor),
    }
}

fn frame_path(timestamp_ns: i64, sensor: &str) -> String {
    format!(
        "{}/{}",
        format::FRAMES_DIR,
        format::frame_name(timestamp_ns, sensor)
    )
}

fn read_meta(dataset_dir: &Path) -> Result<Meta, DatasetError> {
    let meta_path = dataset_dir.join(format::META_FILE);
    let meta_file = std::fs::File::open(&meta_path).map_err(|e| DatasetError::ReadFile {
        path: meta_path.clone(),
        source: e,
    })?;
    serde_json::from_reader(meta_file).map_err(|source| DatasetError::DeserializeJson {
        path: meta_path,
        source,
    })
}

fn read_manifest(dataset_dir: &Path) -> Result<Manifest, DatasetError> {
    let manifest_path = dataset_dir.join(format::MANIFEST_FILE);
    let manifest_file =
        std::fs::File::open(&manifest_path).map_err(|e| DatasetError::ReadFile {
            path: manifest_path.clone(),
            source: e,
        })?;
    serde_json::from_reader(manifest_file).map_err(|source| DatasetError::DeserializeJson {
        path: manifest_path,
        source,
    })
}

fn read_calibration(dataset_dir: &Path) -> Result<Calibration, DatasetError> {
    let calibration_path = dataset_dir.join(format::CALIBRATION_FILE);
    let calibration_file =
        std::fs::File::open(&calibration_path).map_err(|e| DatasetError::ReadFile {
            path: calibration_path.clone(),
            source: e,
        })?;
    serde_json::from_reader(calibration_file).map_err(|source| DatasetError::DeserializeJson {
        path: calibration_path,
        source,
    })
}

fn scan_frames_with_depth(
    frames_dir: &Path,
    mono_dimensions: FrameDimensions,
    depth: Option<&DepthImageContract>,
) -> Result<FrameSet, DatasetError> {
    let mut frames = FrameSet {
        left: Vec::new(),
        right: Vec::new(),
        parse_fail: 0,
        size_mismatch: 0,
    };
    let mono_expected_len = mono_dimensions.area() as u64;
    let depth_expected_len = depth.map(|contract| contract.expected_payload_len());

    let entries = std::fs::read_dir(frames_dir).map_err(|e| DatasetError::ReadDirectory {
        path: frames_dir.to_path_buf(),
        source: e,
    })?;

    for entry in entries {
        let entry = entry.map_err(|e| DatasetError::ReadDirectory {
            path: frames_dir.to_path_buf(),
            source: e,
        })?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let filename = match path.file_name().and_then(|f| f.to_str()) {
            Some(name) => name,
            None => {
                frames.parse_fail += 1;
                continue;
            }
        };

        let (timestamp_ns, sensor) = match format::parse_frame_filename(filename) {
            Some(info) => info,
            None => {
                frames.parse_fail += 1;
                continue;
            }
        };
        if filename != format::frame_name(timestamp_ns, &sensor) {
            frames.parse_fail = frames.parse_fail.saturating_add(1);
            continue;
        }

        let expected_len = match sensor.as_str() {
            "mono_left" | "mono_right" => mono_expected_len,
            "depth" => match depth_expected_len {
                Some(len) => len,
                None => continue,
            },
            _ => {
                frames.parse_fail = frames.parse_fail.saturating_add(1);
                continue;
            }
        };
        let metadata = entry.metadata().map_err(|e| DatasetError::ReadFile {
            path: path.clone(),
            source: e,
        })?;
        if metadata.len() != expected_len {
            frames.size_mismatch = frames.size_mismatch.saturating_add(1);
            continue;
        }
        if sensor == "depth" {
            continue;
        }
        let info = FrameInfo {
            timestamp_ns,
            path: frame_path(timestamp_ns, &sensor),
        };
        if sensor == "mono_left" {
            frames.left.push(info);
        } else {
            frames.right.push(info);
        }
    }

    Ok(frames)
}

fn compute_period_ns(frames: &[FrameInfo]) -> Option<u64> {
    if frames.len() < 2 {
        return None;
    }
    let mut deltas: Vec<u64> = frames
        .windows(2)
        .map(|pair| pair[1].timestamp_ns.abs_diff(pair[0].timestamp_ns))
        .collect();
    deltas.sort_unstable();
    Some(median_u64(&deltas))
}

fn collect_deltas(left: &[FrameInfo], right: &[FrameInfo], gate: Option<u64>) -> Vec<u64> {
    let mut deltas = Vec::new();
    if right.is_empty() {
        return deltas;
    }

    let mut right_idx = 0usize;
    for left_frame in left {
        while right_idx + 1 < right.len() && right[right_idx].timestamp_ns < left_frame.timestamp_ns
        {
            right_idx += 1;
        }

        let mut best: Option<u64> = None;
        let candidates = [Some(right_idx), right_idx.checked_sub(1)];

        for idx in candidates.into_iter().flatten() {
            if idx >= right.len() {
                continue;
            }
            let delta = right[idx].timestamp_ns.abs_diff(left_frame.timestamp_ns);
            if let Some(gate_ns) = gate
                && delta > gate_ns
            {
                continue;
            }
            if best.is_none_or(|b| delta < b) {
                best = Some(delta);
            }
        }

        if let Some(delta) = best {
            deltas.push(delta);
        }
    }
    deltas
}

fn build_delta_stats(deltas: &[u64]) -> Option<DeltaStats> {
    if deltas.is_empty() {
        return None;
    }
    let mut sorted = deltas.to_vec();
    sorted.sort_unstable();
    let min = sorted.first().copied().unwrap_or(0);
    let max = sorted.last().copied().unwrap_or(0);
    let median = median_u64(&sorted);
    let p95 = percentile_u64(&sorted, 0.95);
    let p99 = percentile_u64(&sorted, 0.99);
    Some(DeltaStats {
        min,
        median,
        p95,
        p99,
        max,
    })
}

fn compute_pairing_window_ns(
    deltas: &[u64],
    stats: Option<&DeltaStats>,
    left_period: Option<u64>,
) -> u64 {
    if deltas.is_empty() {
        return left_period.unwrap_or(0) / 4;
    }
    let mut sorted = deltas.to_vec();
    sorted.sort_unstable();
    let median = median_u64(&sorted);
    let mad = median_absolute_deviation(&sorted, median);
    let p99 = stats
        .map(|s| s.p99)
        .unwrap_or_else(|| sorted.last().copied().unwrap_or(0));
    let mut window = p99.max(median.saturating_add(6_u64.saturating_mul(mad)));
    if let Some(period) = left_period
        && period > 0
    {
        window = window.min(period / 4);
    }
    window.min(i64::MAX as u64)
}

fn pair_entries(
    left: &[FrameInfo],
    right: &[FrameInfo],
    window_ns: u64,
) -> (Vec<ManifestEntry>, u64, u64, u64, u64) {
    debug_assert!(
        left.windows(2)
            .all(|pair| pair[0].timestamp_ns < pair[1].timestamp_ns)
    );
    debug_assert!(
        right
            .windows(2)
            .all(|pair| pair[0].timestamp_ns < pair[1].timestamp_ns)
    );

    let mut entries = Vec::with_capacity(left.len());
    let mut paired_count = 0u64;
    let mut left_orphans = 0u64;
    let mut outside_window = 0u64;
    let has_right = !right.is_empty();

    // Rights below this floor were paired or superseded by a closer later match. Keeping the floor
    // monotone prevents inferred pairs from crossing while retaining nearest-eligible semantics.
    let mut right_floor = 0usize;
    let mut right_idx = 0usize;
    for left_frame in left {
        right_idx = right_idx.max(right_floor);
        while right_idx < right.len() && right[right_idx].timestamp_ns < left_frame.timestamp_ns {
            right_idx += 1;
        }

        let left_candidate = right_idx
            .checked_sub(1)
            .filter(|&candidate| candidate >= right_floor);
        let right_candidate = (right_idx < right.len()).then_some(right_idx);

        let mut best_idx = None;
        let mut best_delta = None;

        for idx in [left_candidate, right_candidate].into_iter().flatten() {
            let delta = right[idx].timestamp_ns.abs_diff(left_frame.timestamp_ns);
            if best_delta.is_none_or(|b| delta < b) {
                best_delta = Some(delta);
                best_idx = Some(idx);
            }
        }

        let entry = if let (Some(idx), Some(delta)) = (best_idx, best_delta) {
            if delta > window_ns {
                left_orphans += 1;
                outside_window += 1;
                ManifestEntry {
                    left: ManifestFrameRef {
                        timestamp_ns: left_frame.timestamp_ns,
                        path: left_frame.path.clone(),
                    },
                    pairing: ManifestPairing::MissingRight {
                        reason: PairReason::OutsideWindow,
                    },
                }
            } else {
                right_floor = idx + 1;
                paired_count += 1;
                ManifestEntry {
                    left: ManifestFrameRef {
                        timestamp_ns: left_frame.timestamp_ns,
                        path: left_frame.path.clone(),
                    },
                    pairing: ManifestPairing::Paired {
                        right: ManifestFrameRef {
                            timestamp_ns: right[idx].timestamp_ns,
                            path: right[idx].path.clone(),
                        },
                        delta_ns: delta,
                    },
                }
            }
        } else {
            left_orphans += 1;
            let reason = if has_right {
                // Earlier unpaired files can still exist below `right_floor`; exhausted means no
                // order-preserving right remains eligible for this or a later manifest entry.
                PairReason::RightExhausted
            } else {
                PairReason::NoRightFrames
            };
            ManifestEntry {
                left: ManifestFrameRef {
                    timestamp_ns: left_frame.timestamp_ns,
                    path: left_frame.path.clone(),
                },
                pairing: ManifestPairing::MissingRight { reason },
            }
        };

        entries.push(entry);
    }

    let total_right = u64::try_from(right.len()).expect("supported host usize fits u64 counters");
    let right_orphans = total_right
        .checked_sub(paired_count)
        .expect("one-to-one pairing cannot exceed the right input count");
    (
        entries,
        paired_count,
        left_orphans,
        right_orphans,
        outside_window,
    )
}

fn median_u64(sorted: &[u64]) -> u64 {
    let len = sorted.len();
    if len == 0 {
        return 0;
    }
    if len % 2 == 1 {
        sorted[len / 2]
    } else {
        let a = sorted[len / 2 - 1];
        let b = sorted[len / 2];
        a + (b - a) / 2
    }
}

fn percentile_u64(sorted: &[u64], pct: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * pct).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn median_absolute_deviation(sorted: &[u64], median: u64) -> u64 {
    let mut deviations: Vec<u64> = sorted.iter().map(|v| v.abs_diff(median)).collect();
    deviations.sort_unstable();
    median_u64(&deviations)
}

fn dataset_id(dataset_dir: &Path) -> String {
    dataset_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("dataset")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        FrameId, RectifiedStereo, RectifiedStereoCompatibilityError, RectifiedStereoConfig,
        Timestamp,
    };
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    static NEXT_PATH_ID: AtomicU64 = AtomicU64::new(0);

    #[derive(Debug, PartialEq, Eq)]
    enum PairEntrySignature {
        Paired {
            right_timestamp_ns: i64,
            delta_ns: u64,
        },
        OutsideWindow,
        NoRightFrames,
        RightExhausted,
    }

    fn pair_entry_signatures(entries: &[ManifestEntry]) -> Vec<PairEntrySignature> {
        entries
            .iter()
            .map(|entry| match &entry.pairing {
                ManifestPairing::Paired { right, delta_ns } => PairEntrySignature::Paired {
                    right_timestamp_ns: right.timestamp_ns,
                    delta_ns: *delta_ns,
                },
                ManifestPairing::MissingRight {
                    reason: PairReason::OutsideWindow,
                } => PairEntrySignature::OutsideWindow,
                ManifestPairing::MissingRight {
                    reason: PairReason::NoRightFrames,
                } => PairEntrySignature::NoRightFrames,
                ManifestPairing::MissingRight {
                    reason: PairReason::RightExhausted,
                } => PairEntrySignature::RightExhausted,
            })
            .collect()
    }

    fn reference_pair_entries(
        left: &[FrameInfo],
        right: &[FrameInfo],
        window_ns: u64,
    ) -> (Vec<PairEntrySignature>, u64, u64, u64, u64) {
        let mut signatures = Vec::with_capacity(left.len());
        let mut right_floor = 0usize;
        let mut paired_count = 0u64;
        let mut outside_window = 0u64;

        for left_frame in left {
            let mut best = None;
            for (index, right_frame) in right.iter().enumerate().skip(right_floor) {
                let delta = right_frame.timestamp_ns.abs_diff(left_frame.timestamp_ns);
                if best.is_none_or(|(_, best_delta)| delta < best_delta) {
                    best = Some((index, delta));
                }
            }

            match best {
                Some((index, delta_ns)) if delta_ns <= window_ns => {
                    right_floor = index + 1;
                    paired_count += 1;
                    signatures.push(PairEntrySignature::Paired {
                        right_timestamp_ns: right[index].timestamp_ns,
                        delta_ns,
                    });
                }
                Some(_) => {
                    outside_window += 1;
                    signatures.push(PairEntrySignature::OutsideWindow);
                }
                None if right.is_empty() => {
                    signatures.push(PairEntrySignature::NoRightFrames);
                }
                None => {
                    signatures.push(PairEntrySignature::RightExhausted);
                }
            }
        }

        let total_left = u64::try_from(left.len()).expect("small reference count");
        let total_right = u64::try_from(right.len()).expect("small reference count");
        (
            signatures,
            paired_count,
            total_left - paired_count,
            total_right - paired_count,
            outside_window,
        )
    }

    fn frame_infos(timestamps: &[i64], stream: &str) -> Vec<FrameInfo> {
        timestamps
            .iter()
            .map(|&timestamp_ns| FrameInfo {
                timestamp_ns,
                path: format!("{stream}-{timestamp_ns}"),
            })
            .collect()
    }

    fn unique_dataset_path(test_name: &str) -> PathBuf {
        let id = NEXT_PATH_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("kiko-slam-{test_name}-{}-{id}", std::process::id()))
    }

    fn meta() -> Meta {
        Meta {
            created: "2026-07-10T00:00:00Z".to_string(),
            device: "test-device".to_string(),
            mono: Some(MonoMeta {
                width: 2,
                height: 2,
                fps: 30,
            }),
            depth: None,
            imu: None,
        }
    }

    fn calibration() -> Calibration {
        let intrinsics = CameraIntrinsics {
            fx: 100.0,
            fy: 100.0,
            cx: 1.0,
            cy: 1.0,
            width: 2,
            height: 2,
        };
        Calibration {
            left: intrinsics.clone(),
            right: intrinsics,
            baseline_m: 0.1,
            rectified: true,
        }
    }

    fn meta_with_depth(width: u32, height: u32) -> Meta {
        let mut meta = meta();
        meta.depth = Some(DepthMeta {
            width,
            height,
            fps: 30,
            encoding: "f32_meters_le".to_string(),
            optical_frame: Some(DepthOpticalFrame::RectifiedLeft),
        });
        meta
    }

    fn meta_with_imu(rate_hz: u32) -> Meta {
        let mut meta = meta();
        meta.imu = Some(ImuMeta { rate_hz });
        meta
    }

    fn imu_stream_metadata(session_id: u64) -> ImuStreamMetadata {
        ImuStreamMetadata::new(
            DeviceSessionId::try_new(session_id).expect("valid test session"),
            ImuExtrinsicProvenance::uncalibrated_unknown(),
        )
    }

    fn imu_report(
        session_id: u64,
        sequence: u32,
        host_arrival_ns: u64,
        accel_timestamp_ns: i64,
        gyro_timestamp_ns: i64,
        offset: f64,
    ) -> ImuReport {
        let session_id = DeviceSessionId::try_new(session_id).expect("test session");
        ImuReport::new(
            session_id,
            DequeueSequence::new(sequence),
            HostMonotonicTimestamp::from_nanos(host_arrival_ns),
            AccelSample::new(
                DeviceTimestamp::try_from_nanos(accel_timestamp_ns).expect("accel timestamp"),
                OakImuAcceleration::try_new(1.0 + offset, 2.0 + offset, 3.0 + offset)
                    .expect("finite acceleration"),
                SensorAccuracy::High,
            ),
            GyroSample::new(
                DeviceTimestamp::try_from_nanos(gyro_timestamp_ns).expect("gyro timestamp"),
                OakImuAngularVelocity::try_new(4.0 + offset, 5.0 + offset, 6.0 + offset)
                    .expect("finite angular velocity"),
                SensorAccuracy::Medium,
            ),
        )
    }

    fn write_imu_fixture(test_name: &str, reports: &[ImuReport]) -> PathBuf {
        let path = unique_dataset_path(test_name);
        let (writer, handle) = DatasetWriter::create_with_imu_config(
            &path,
            &meta_with_imu(200),
            &calibration(),
            imu_stream_metadata(7),
            DatasetWriterConfig::default(),
        )
        .expect("create IMU dataset");
        for report in reports {
            assert_eq!(
                write_outcome(writer.write_imu(*report)),
                WriteOutcome::Enqueued
            );
        }
        drop(writer);
        handle.finish().expect("finish IMU dataset");
        path
    }

    fn corrupt_imu_fixture(test_name: &str, mutate: impl FnOnce(&mut Vec<u8>)) -> DatasetError {
        let path = write_imu_fixture(test_name, &[imu_report(7, 10, 100, 10, 40, 0.0)]);
        let stream_path = path.join(format::IMU_STREAM_FILE);
        let mut bytes = std::fs::read(&stream_path).expect("read IMU fixture");
        mutate(&mut bytes);
        std::fs::write(&stream_path, bytes).expect("write corrupted IMU fixture");
        let error = DatasetReader::open(&path).expect_err("corrupt IMU stream must fail open");
        std::fs::remove_dir_all(path).expect("remove corrupt IMU fixture");
        error
    }

    fn depth_image(width: u32, height: u32, timestamp_ns: i64) -> DepthImage {
        depth_image_filled(width, height, timestamp_ns, 1.0)
    }

    fn depth_image_filled(width: u32, height: u32, timestamp_ns: i64, value: f32) -> DepthImage {
        let len = usize::try_from(width)
            .expect("test width fits usize")
            .checked_mul(usize::try_from(height).expect("test height fits usize"))
            .expect("test depth area fits usize");
        DepthImage::new(
            FrameId::new(0),
            Timestamp::from_nanos(timestamp_ns),
            width,
            height,
            vec![value; len],
        )
        .expect("valid depth image")
    }

    fn writer_contract() -> WriterDatasetContract {
        WriterDatasetContract::parse(&meta(), &calibration()).expect("valid writer contract")
    }

    #[test]
    fn new_depth_writer_rejects_legacy_unspecified_optical_frame() {
        let mut metadata = meta_with_depth(2, 2);
        metadata
            .depth
            .as_mut()
            .expect("depth metadata")
            .optical_frame = None;

        assert!(matches!(
            WriterDatasetContract::parse(&metadata, &calibration()),
            Err(DatasetError::DepthOpticalFrameRequiredForWrite)
        ));
        assert!(
            DepthImageContract::parse(metadata.depth.as_ref().expect("depth metadata")).is_ok(),
            "reader parsing remains backward-compatible with legacy metadata"
        );
    }

    fn writer_sidecars() -> WriterSidecars {
        WriterSidecars {
            meta_json: serde_json::to_vec_pretty(&meta()).expect("serialize metadata"),
            calibration_json: serde_json::to_vec_pretty(&calibration())
                .expect("serialize calibration"),
        }
    }

    fn write_outcome(result: Result<WriteOutcome, DatasetWriteError>) -> WriteOutcome {
        result.expect("valid dataset write")
    }

    fn wait_for_written_frames(writer: &PairedDatasetWriter, expected: u64) {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let stats = writer.stats();
            if stats.frames_written >= expected {
                return;
            }
            assert!(!stats.writer_failed, "writer failed before test mutation");
            assert!(
                Instant::now() < deadline,
                "writer did not drain test payloads"
            );
            std::thread::yield_now();
        }
    }

    fn reject_before_filesystem_mutation(
        test_name: &str,
        meta: &Meta,
        calibration: &Calibration,
    ) -> DatasetError {
        let root = unique_dataset_path(test_name);
        let path = root.join("missing-parent").join("dataset");
        let error = DatasetWriter::create(&path, meta, calibration)
            .expect_err("invalid contract must reject writer creation");
        assert!(
            !root.exists(),
            "invalid boundary data must not create parent directories"
        );
        error
    }

    fn frame(sensor: SensorId, timestamp_ns: i64, value: u8) -> Frame {
        Frame::new(
            sensor,
            FrameId::new(0),
            Timestamp::from_nanos(timestamp_ns),
            2,
            2,
            vec![value; 4],
        )
        .expect("valid frame")
    }

    fn stereo_pair(
        left_timestamp_ns: i64,
        right_timestamp_ns: i64,
        validation_window: PairingWindowNs,
    ) -> StereoPair {
        StereoPair::try_new(
            frame(SensorId::StereoLeft, left_timestamp_ns, 1),
            frame(SensorId::StereoRight, right_timestamp_ns, 2),
            validation_window,
        )
        .expect("valid stereo pair")
    }

    fn write_exact_pairs(path: &Path, timestamps_ns: &[i64]) {
        let (writer, handle) =
            DatasetWriter::create(path, &meta(), &calibration()).expect("create dataset");
        for (index, &timestamp_ns) in timestamps_ns.iter().enumerate() {
            let left_value = u8::try_from(index.saturating_mul(2)).expect("small test fixture");
            let right_value = left_value.saturating_add(1);
            assert_eq!(
                write_outcome(writer.write_frame(&frame(
                    SensorId::StereoLeft,
                    timestamp_ns,
                    left_value,
                ))),
                WriteOutcome::Enqueued
            );
            assert_eq!(
                write_outcome(writer.write_frame(&frame(
                    SensorId::StereoRight,
                    timestamp_ns,
                    right_value,
                ))),
                WriteOutcome::Enqueued
            );
        }
        drop(writer);
        handle.finish().expect("finish dataset");
    }

    #[test]
    fn depth_optical_frame_round_trips_and_legacy_metadata_remains_readable() {
        let metadata = meta_with_depth(640, 480);
        let encoded = serde_json::to_value(&metadata).expect("serialize depth metadata");
        assert_eq!(
            encoded["depth"]["optical_frame"],
            serde_json::json!("rectified_left")
        );
        let decoded: Meta = serde_json::from_value(encoded.clone()).expect("parse metadata");
        assert_eq!(
            decoded.depth.and_then(|depth| depth.optical_frame),
            Some(DepthOpticalFrame::RectifiedLeft)
        );

        let mut legacy = encoded;
        legacy["depth"]
            .as_object_mut()
            .expect("depth object")
            .remove("optical_frame");
        let decoded: Meta = serde_json::from_value(legacy).expect("parse legacy metadata");
        assert_eq!(
            decoded.depth.and_then(|depth| depth.optical_frame),
            None,
            "absence is preserved so projection callers must make an explicit assumption"
        );
    }

    #[test]
    fn create_rejects_an_existing_dataset_path() {
        let path = unique_dataset_path("existing-path");
        std::fs::create_dir(&path).expect("create existing path");

        let result = DatasetWriter::create(&path, &meta(), &calibration());

        assert!(matches!(result, Err(DatasetError::AlreadyExists { .. })));
        std::fs::remove_dir_all(path).expect("remove test directory");
    }

    #[test]
    fn create_parses_the_complete_mono_contract_before_filesystem_mutation() {
        let mut missing_mono = meta();
        missing_mono.mono = None;
        assert!(matches!(
            reject_before_filesystem_mutation(
                "contract-missing-mono",
                &missing_mono,
                &calibration()
            ),
            DatasetError::MissingMonoConfig
        ));

        let mut zero_width = meta();
        zero_width.mono.as_mut().expect("mono metadata").width = 0;
        assert!(matches!(
            reject_before_filesystem_mutation("contract-zero-width", &zero_width, &calibration()),
            DatasetError::InvalidFrameDimensions {
                field: "meta.mono",
                source: crate::FrameDimensionsError::Zero {
                    width: 0,
                    height: 2
                }
            }
        ));

        let mut zero_fps = meta();
        zero_fps.mono.as_mut().expect("mono metadata").fps = 0;
        assert!(matches!(
            reject_before_filesystem_mutation("contract-zero-fps", &zero_fps, &calibration()),
            DatasetError::InvalidNominalFps {
                field: "meta.mono.fps",
                value: 0
            }
        ));

        for (test_name, side) in [
            ("contract-left-dimensions", "calibration.left"),
            ("contract-right-dimensions", "calibration.right"),
        ] {
            let mut mismatched = calibration();
            let intrinsics = if side == "calibration.left" {
                &mut mismatched.left
            } else {
                &mut mismatched.right
            };
            intrinsics.width = 1;
            intrinsics.height = 4;
            assert!(matches!(
                reject_before_filesystem_mutation(test_name, &meta(), &mismatched),
                DatasetError::ImageDimensionsMismatch {
                    expected_field: "meta.mono",
                    actual_field,
                    ..
                } if actual_field == side
            ));
        }

        let mut non_finite_left = calibration();
        non_finite_left.left.fx = f32::NAN;
        assert!(matches!(
            reject_before_filesystem_mutation(
                "contract-non-finite-left",
                &meta(),
                &non_finite_left
            ),
            DatasetError::InvalidCameraIntrinsics {
                field: "calibration.left",
                source: crate::IntrinsicsError::NonFinite { .. }
            }
        ));

        let mut non_finite_right = calibration();
        non_finite_right.right.cy = f32::INFINITY;
        assert!(matches!(
            reject_before_filesystem_mutation(
                "contract-non-finite-right",
                &meta(),
                &non_finite_right
            ),
            DatasetError::InvalidCameraIntrinsics {
                field: "calibration.right",
                source: crate::IntrinsicsError::NonFinite { .. }
            }
        ));

        for (test_name, baseline_m, non_finite) in [
            ("contract-nan-baseline", f32::NAN, true),
            ("contract-infinite-baseline", f32::INFINITY, true),
            ("contract-zero-baseline", 0.0, false),
            ("contract-negative-baseline", -1.0, false),
        ] {
            let mut invalid = calibration();
            invalid.baseline_m = baseline_m;
            let error = reject_before_filesystem_mutation(test_name, &meta(), &invalid);
            assert!(
                matches!(
                    (&error, non_finite),
                    (
                        DatasetError::InvalidStereoBaseline {
                            source: StereoBaselineError::NonFinite { baseline_m: actual }
                        },
                        true
                    ) if actual.to_bits() == baseline_m.to_bits()
                ) || matches!(
                    (&error, non_finite),
                    (
                        DatasetError::InvalidStereoBaseline {
                            source: StereoBaselineError::NonPositive { baseline_m: actual }
                        },
                        false
                    ) if actual.to_bits() == baseline_m.to_bits()
                )
            );
            assert!(
                std::error::Error::source(&error)
                    .and_then(|source| source.downcast_ref::<StereoBaselineError>())
                    .is_some(),
                "dataset baseline error must preserve its typed source"
            );
        }
    }

    #[test]
    fn inferred_delta_stats_describe_only_explicit_manifest_pairs() {
        let left = [
            FrameInfo {
                timestamp_ns: 0,
                path: "left-0".to_string(),
            },
            FrameInfo {
                timestamp_ns: 1,
                path: "left-1".to_string(),
            },
        ];
        let right = [
            FrameInfo {
                timestamp_ns: 0,
                path: "right-0".to_string(),
            },
            FrameInfo {
                timestamp_ns: 100,
                path: "right-100".to_string(),
            },
        ];

        let topology = inferred_manifest_topology(&left, &right);

        assert_eq!(topology.pairing_window_ns, 0);
        assert_eq!(topology.paired_count, 1);
        assert_eq!(topology.left_orphans, 1);
        assert_eq!(
            topology.delta_stats,
            Some(DeltaStats {
                min: 0,
                median: 0,
                p95: 0,
                p99: 0,
                max: 0,
            })
        );
    }

    #[test]
    fn inferred_pairing_supersedes_earlier_right_instead_of_crossing() {
        let left = frame_infos(&[6, 7], "left");
        let right = frame_infos(&[0, 7], "right");

        let (entries, paired, left_orphans, right_orphans, outside_window) =
            pair_entries(&left, &right, 7);

        assert_eq!(
            pair_entry_signatures(&entries),
            vec![
                PairEntrySignature::Paired {
                    right_timestamp_ns: 7,
                    delta_ns: 1,
                },
                PairEntrySignature::RightExhausted,
            ]
        );
        assert_eq!(
            (paired, left_orphans, right_orphans, outside_window),
            (1, 1, 1, 0)
        );
    }

    #[test]
    fn inferred_pairing_tie_keeps_the_successor_eligible() {
        let left = frame_infos(&[5, 6], "left");
        let right = frame_infos(&[4, 6], "right");

        let (entries, paired, left_orphans, right_orphans, outside_window) =
            pair_entries(&left, &right, 1);

        assert_eq!(
            pair_entry_signatures(&entries),
            vec![
                PairEntrySignature::Paired {
                    right_timestamp_ns: 4,
                    delta_ns: 1,
                },
                PairEntrySignature::Paired {
                    right_timestamp_ns: 6,
                    delta_ns: 0,
                },
            ]
        );
        assert_eq!(
            (paired, left_orphans, right_orphans, outside_window),
            (2, 0, 0, 0)
        );
    }

    #[test]
    fn inferred_pairing_outside_window_does_not_consume_a_future_match() {
        let left = frame_infos(&[0, 9], "left");
        let right = frame_infos(&[10], "right");

        let (entries, paired, left_orphans, right_orphans, outside_window) =
            pair_entries(&left, &right, 1);

        assert_eq!(
            pair_entry_signatures(&entries),
            vec![
                PairEntrySignature::OutsideWindow,
                PairEntrySignature::Paired {
                    right_timestamp_ns: 10,
                    delta_ns: 1,
                },
            ]
        );
        assert_eq!(
            (paired, left_orphans, right_orphans, outside_window),
            (1, 1, 0, 1)
        );
    }

    #[test]
    fn inferred_pairing_matches_exhaustive_monotone_floor_reference() {
        const VALUES: [i64; 5] = [-4, -1, 0, 1, 4];
        for left_mask in 0usize..1 << VALUES.len() {
            let left_timestamps: Vec<i64> = VALUES
                .iter()
                .enumerate()
                .filter_map(|(index, &value)| ((left_mask >> index) & 1 == 1).then_some(value))
                .collect();
            let left = frame_infos(&left_timestamps, "left");
            for right_mask in 0usize..1 << VALUES.len() {
                let right_timestamps: Vec<i64> = VALUES
                    .iter()
                    .enumerate()
                    .filter_map(|(index, &value)| ((right_mask >> index) & 1 == 1).then_some(value))
                    .collect();
                let right = frame_infos(&right_timestamps, "right");
                for window_ns in 0..=4 {
                    let actual = pair_entries(&left, &right, window_ns);
                    let expected = reference_pair_entries(&left, &right, window_ns);
                    assert_eq!(
                        (
                            pair_entry_signatures(&actual.0),
                            actual.1,
                            actual.2,
                            actual.3,
                            actual.4,
                        ),
                        expected,
                        "left={left_timestamps:?}, right={right_timestamps:?}, window={window_ns}"
                    );

                    let paired_rights: Vec<i64> = actual
                        .0
                        .iter()
                        .filter_map(|entry| match &entry.pairing {
                            ManifestPairing::Paired { right, .. } => Some(right.timestamp_ns),
                            ManifestPairing::MissingRight { .. } => None,
                        })
                        .collect();
                    assert!(
                        paired_rights.windows(2).all(|pair| pair[0] < pair[1]),
                        "paired right timestamps must be strict: {paired_rights:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn create_parses_the_depth_contract_before_filesystem_mutation() {
        let mut zero_dimensions = meta_with_depth(0, 2);
        assert!(matches!(
            reject_before_filesystem_mutation(
                "depth-contract-zero-dimensions",
                &zero_dimensions,
                &calibration()
            ),
            DatasetError::InvalidFrameDimensions {
                field: "meta.depth",
                ..
            }
        ));

        zero_dimensions
            .depth
            .as_mut()
            .expect("depth metadata")
            .width = 2;
        zero_dimensions.depth.as_mut().expect("depth metadata").fps = 0;
        assert!(matches!(
            reject_before_filesystem_mutation(
                "depth-contract-zero-fps",
                &zero_dimensions,
                &calibration()
            ),
            DatasetError::InvalidNominalFps {
                field: "meta.depth.fps",
                value: 0
            }
        ));

        let mut unsupported = meta_with_depth(2, 2);
        unsupported.depth.as_mut().expect("depth metadata").encoding =
            "u16_millimeters_le".to_string();
        assert!(matches!(
            reject_before_filesystem_mutation(
                "depth-contract-unsupported-encoding",
                &unsupported,
                &calibration()
            ),
            DatasetError::UnsupportedDepthEncoding { value }
                if value == "u16_millimeters_le"
        ));
    }

    #[test]
    fn depth_writes_require_a_configured_exact_shape() {
        let missing_path = unique_dataset_path("depth-contract-missing");
        let (writer, handle) = DatasetWriter::create(&missing_path, &meta(), &calibration())
            .expect("create dataset without depth");
        assert!(matches!(
            writer.write_depth(&depth_image(2, 2, 1)),
            Err(DatasetWriteError::DepthStreamNotConfigured)
        ));
        drop(writer);
        assert!(matches!(
            handle
                .finish()
                .expect_err("missing depth contract must fail"),
            DatasetError::WriteContract {
                source: DatasetWriteError::DepthStreamNotConfigured
            }
        ));
        assert!(!missing_path.join(format::MANIFEST_FILE).exists());
        std::fs::remove_dir_all(missing_path).expect("remove missing-depth dataset");

        let wrong_shape_path = unique_dataset_path("depth-contract-wrong-shape");
        let depth_meta = meta_with_depth(2, 6);
        let (writer, handle) =
            DatasetWriter::create(&wrong_shape_path, &depth_meta, &calibration())
                .expect("create depth dataset");
        assert!(matches!(
            writer.write_depth(&depth_image(3, 4, 2)),
            Err(DatasetWriteError::DepthDimensionsMismatch { expected, actual })
                if expected == FrameDimensions::new(2, 6)
                    && actual == FrameDimensions::new(3, 4)
        ));
        drop(writer);
        assert!(matches!(
            handle.finish().expect_err("wrong depth shape must fail"),
            DatasetError::WriteContract {
                source: DatasetWriteError::DepthDimensionsMismatch { .. }
            }
        ));
        assert!(!wrong_shape_path.join(format::MANIFEST_FILE).exists());
        std::fs::remove_dir_all(wrong_shape_path).expect("remove wrong-shape dataset");
    }

    #[test]
    fn configured_depth_payload_uses_declared_f32_little_endian_encoding() {
        let path = unique_dataset_path("depth-contract-valid");
        let dataset_meta = meta_with_depth(2, 2);
        let (writer, handle) = DatasetWriter::create(&path, &dataset_meta, &calibration())
            .expect("create depth dataset");
        assert_eq!(
            write_outcome(writer.write_depth(&depth_image(2, 2, 3))),
            WriteOutcome::Enqueued
        );
        drop(writer);
        let stats = handle.finish().expect("finish depth dataset");
        assert_eq!(stats.frames_written, 1);
        assert_eq!(stats.bytes_written, 16);
        let depth_path = path
            .join(format::FRAMES_DIR)
            .join(format::frame_name(3, "depth"));
        let payload = std::fs::read(depth_path).expect("read depth payload");
        assert_eq!(payload, 1.0_f32.to_le_bytes().repeat(4));
        let reader = DatasetReader::open(&path).expect("reader accepts parsed depth contract");
        let projection = reader
            .depth_projection_contract()
            .expect("parsed depth projection contract");
        assert_eq!(projection.dimensions(), FrameDimensions::new(2, 2));
        assert_eq!(
            projection.optical_frame(),
            Some(DepthOpticalFrame::RectifiedLeft)
        );
        let mut invalid_meta = dataset_meta;
        invalid_meta
            .depth
            .as_mut()
            .expect("depth metadata")
            .encoding = "u16_millimeters_le".to_string();
        std::fs::write(
            path.join(format::META_FILE),
            serde_json::to_vec_pretty(&invalid_meta).expect("serialize invalid depth metadata"),
        )
        .expect("replace depth metadata");
        assert!(matches!(
            DatasetReader::open(&path).expect_err("reader must reject unsupported depth encoding"),
            DatasetError::UnsupportedDepthEncoding { .. }
        ));
        std::fs::remove_dir_all(path).expect("remove valid depth dataset");
    }

    #[test]
    fn manifest_depth_entries_are_sorted_successful_writer_identities() {
        let path = unique_dataset_path("depth-manifest-identities");
        let dataset_meta = meta_with_depth(2, 2);
        let (writer, handle) = DatasetWriter::create(&path, &dataset_meta, &calibration())
            .expect("create depth dataset");
        for timestamp_ns in [30, 10, 20] {
            assert_eq!(
                write_outcome(writer.write_depth(&depth_image(2, 2, timestamp_ns))),
                WriteOutcome::Enqueued
            );
        }

        let unrecorded_path = path
            .join(format::FRAMES_DIR)
            .join(format::frame_name(15, "depth"));
        std::fs::write(&unrecorded_path, [0_u8; 16])
            .expect("write valid but unrecorded depth payload");

        drop(writer);
        handle.finish().expect("finish depth dataset");

        let manifest = read_manifest(&path).expect("read depth manifest");
        let depth_entries = manifest
            .depth_entries
            .expect("configured depth stream must be represented");
        let identities: Vec<(i64, String)> = depth_entries
            .into_iter()
            .map(|entry| (entry.timestamp_ns, entry.path))
            .collect();
        assert_eq!(
            identities,
            [
                (10, frame_path(10, "depth")),
                (20, frame_path(20, "depth")),
                (30, frame_path(30, "depth")),
            ]
        );
        assert!(unrecorded_path.exists());

        std::fs::remove_dir_all(path).expect("remove depth identity dataset");
    }

    #[test]
    fn depth_manifest_distinguishes_configured_empty_legacy_and_unconfigured_streams() {
        let configured_path = unique_dataset_path("depth-manifest-empty");
        let (writer, handle) =
            DatasetWriter::create(&configured_path, &meta_with_depth(2, 2), &calibration())
                .expect("create configured depth dataset");
        drop(writer);
        handle.finish().expect("finish empty depth dataset");

        let manifest_path = configured_path.join(format::MANIFEST_FILE);
        let mut serialized: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&manifest_path).expect("read configured depth manifest"),
        )
        .expect("parse configured depth manifest");
        assert_eq!(serialized["depth_entries"], serde_json::json!([]));
        assert!(
            read_manifest(&configured_path)
                .expect("read configured depth manifest")
                .depth_entries
                .is_some_and(|entries| entries.is_empty())
        );

        serialized
            .as_object_mut()
            .expect("manifest object")
            .remove("depth_entries");
        let legacy: Manifest =
            serde_json::from_value(serialized).expect("legacy manifest remains deserializable");
        assert!(legacy.depth_entries.is_none());
        std::fs::remove_dir_all(configured_path).expect("remove empty depth dataset");

        let unconfigured_path = unique_dataset_path("depth-manifest-unconfigured");
        let (writer, handle) = DatasetWriter::create(&unconfigured_path, &meta(), &calibration())
            .expect("create unconfigured dataset");
        drop(writer);
        handle.finish().expect("finish unconfigured dataset");
        let serialized: serde_json::Value = serde_json::from_slice(
            &std::fs::read(unconfigured_path.join(format::MANIFEST_FILE))
                .expect("read unconfigured manifest"),
        )
        .expect("parse unconfigured manifest");
        assert!(serialized.get("depth_entries").is_none());
        std::fs::remove_dir_all(unconfigured_path).expect("remove unconfigured dataset");
    }

    #[test]
    fn depth_cursor_is_manifest_owned_independent_and_handles_signed_timestamp_bounds() {
        let path = unique_dataset_path("depth-cursor-signed-bounds");
        let (writer, handle) = DatasetWriter::create(&path, &meta_with_depth(2, 2), &calibration())
            .expect("create depth dataset");
        assert_eq!(
            write_outcome(writer.write_frame(&frame(SensorId::StereoLeft, 0, 1))),
            WriteOutcome::Enqueued
        );
        assert_eq!(
            write_outcome(writer.write_frame(&frame(SensorId::StereoRight, 0, 2))),
            WriteOutcome::Enqueued
        );
        for depth in [
            depth_image_filled(2, 2, i64::MAX, 2.0),
            depth_image_filled(2, 2, i64::MIN, 1.0),
        ] {
            assert_eq!(
                write_outcome(writer.write_depth(&depth)),
                WriteOutcome::Enqueued
            );
        }
        drop(writer);
        handle.finish().expect("finish depth dataset");

        let unreferenced_path = path
            .join(format::FRAMES_DIR)
            .join(format::frame_name(0, "depth"));
        std::fs::write(&unreferenced_path, 3.0_f32.to_le_bytes().repeat(4))
            .expect("write unreferenced depth payload");
        let manifest = read_manifest(&path).expect("read manifest");
        let depth_entries = manifest.depth_entries.expect("indexed depth entries");
        assert_eq!(
            depth_entries
                .iter()
                .map(|entry| (entry.timestamp_ns, entry.path.as_str()))
                .collect::<Vec<_>>(),
            [
                (i64::MIN, frame_path(i64::MIN, "depth").as_str()),
                (i64::MAX, frame_path(i64::MAX, "depth").as_str()),
            ]
        );

        let mut reader = DatasetReader::open(&path).expect("open depth dataset");
        let mut first = reader.depth_cursor().expect("first depth cursor");
        let mut second = reader.depth_cursor().expect("second depth cursor");
        assert_eq!(first.len(), 2);
        assert!(!first.is_empty());
        assert_eq!(first.remaining(), 2);

        let earliest = first
            .next_at_or_before(Timestamp::from_nanos(i64::MIN))
            .expect("decode earliest depth")
            .expect("inclusive cutoff returns earliest depth");
        assert_eq!(earliest.frame_id(), FrameId::new(0));
        assert_eq!(earliest.timestamp(), Timestamp::from_nanos(i64::MIN));
        assert_eq!(earliest.depth_m(), &[1.0; 4]);
        assert_eq!(first.remaining(), 1);
        assert!(
            first
                .next_at_or_before(Timestamp::from_nanos(0))
                .expect("future depth remains pending")
                .is_none()
        );
        assert_eq!(first.remaining(), 1);

        let independently_earliest = second
            .next_at_or_before(Timestamp::from_nanos(i64::MIN))
            .expect("decode earliest depth through second cursor")
            .expect("second cursor starts independently");
        assert_eq!(independently_earliest.frame_id(), FrameId::new(0));

        let pair = reader
            .pairs()
            .next()
            .expect("one stereo pair")
            .expect("decode stereo pair while cursors remain alive");
        assert_eq!(pair.left().timestamp(), Timestamp::from_nanos(0));
        assert_eq!(pair.right().timestamp(), Timestamp::from_nanos(0));
        drop(reader);

        let latest = first
            .next_at_or_before(Timestamp::from_nanos(i64::MAX))
            .expect("decode latest depth")
            .expect("inclusive upper cutoff returns latest depth");
        assert_eq!(latest.frame_id(), FrameId::new(1));
        assert_eq!(latest.timestamp(), Timestamp::from_nanos(i64::MAX));
        assert_eq!(latest.depth_m(), &[2.0; 4]);
        assert_eq!(first.remaining(), 0);
        assert!(
            first
                .next_at_or_before(Timestamp::from_nanos(i64::MAX))
                .expect("exhausted depth cursor")
                .is_none()
        );

        std::fs::remove_dir_all(path).expect("remove depth cursor dataset");
    }

    #[test]
    fn depth_cursor_distinguishes_all_manifest_depth_states() {
        let indexed_path = unique_dataset_path("depth-cursor-indexed-empty");
        let (writer, handle) =
            DatasetWriter::create(&indexed_path, &meta_with_depth(2, 2), &calibration())
                .expect("create indexed depth dataset");
        drop(writer);
        handle.finish().expect("finish indexed depth dataset");
        let reader = DatasetReader::open(&indexed_path).expect("open indexed depth dataset");
        let mut cursor = reader.depth_cursor().expect("empty indexed depth cursor");
        assert_eq!(cursor.len(), 0);
        assert!(cursor.is_empty());
        assert_eq!(cursor.remaining(), 0);
        assert!(
            cursor
                .next_at_or_before(Timestamp::from_nanos(i64::MAX))
                .expect("empty indexed depth stream")
                .is_none()
        );
        std::fs::remove_dir_all(indexed_path).expect("remove indexed depth dataset");

        let unconfigured_path = unique_dataset_path("depth-cursor-unconfigured");
        let (writer, handle) = DatasetWriter::create(&unconfigured_path, &meta(), &calibration())
            .expect("create unconfigured dataset");
        drop(writer);
        handle.finish().expect("finish unconfigured dataset");
        let reader = DatasetReader::open(&unconfigured_path).expect("open unconfigured dataset");
        assert!(matches!(
            reader.depth_cursor(),
            Err(DatasetError::DepthStreamNotConfigured)
        ));
        std::fs::remove_dir_all(unconfigured_path).expect("remove unconfigured dataset");

        let legacy_path = unique_dataset_path("depth-cursor-legacy");
        let (writer, handle) =
            DatasetWriter::create(&legacy_path, &meta_with_depth(2, 2), &calibration())
                .expect("create depth dataset");
        assert_eq!(
            write_outcome(writer.write_frame(&frame(SensorId::StereoLeft, 5, 1))),
            WriteOutcome::Enqueued
        );
        assert_eq!(
            write_outcome(writer.write_frame(&frame(SensorId::StereoRight, 5, 2))),
            WriteOutcome::Enqueued
        );
        assert_eq!(
            write_outcome(writer.write_depth(&depth_image(2, 2, 5))),
            WriteOutcome::Enqueued
        );
        drop(writer);
        handle.finish().expect("finish legacy fixture");

        let manifest_path = legacy_path.join(format::MANIFEST_FILE);
        let mut manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&manifest_path).expect("read generated manifest"),
        )
        .expect("parse generated manifest");
        manifest
            .as_object_mut()
            .expect("manifest object")
            .remove("depth_entries");
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("serialize legacy manifest"),
        )
        .expect("write legacy manifest");

        let mut reader = DatasetReader::open(&legacy_path).expect("open legacy depth manifest");
        assert!(matches!(
            reader.depth_cursor(),
            Err(DatasetError::LegacyDepthManifestUnindexed)
        ));
        assert!(
            reader.pairs().next().expect("legacy stereo pair").is_ok(),
            "legacy depth manifests must retain stereo replay"
        );
        std::fs::remove_dir_all(legacy_path).expect("remove legacy dataset");
    }

    #[test]
    fn reader_rejects_incoherent_or_nonmonotonic_depth_manifest_state() {
        let unconfigured_path = unique_dataset_path("depth-entries-without-meta");
        let (writer, handle) = DatasetWriter::create(&unconfigured_path, &meta(), &calibration())
            .expect("create unconfigured dataset");
        drop(writer);
        handle.finish().expect("finish unconfigured dataset");
        let manifest_path = unconfigured_path.join(format::MANIFEST_FILE);
        let mut manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&manifest_path).expect("read generated manifest"),
        )
        .expect("parse generated manifest");
        manifest["depth_entries"] = serde_json::json!([]);
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("serialize incoherent manifest"),
        )
        .expect("write incoherent manifest");
        assert!(matches!(
            DatasetReader::open(&unconfigured_path)
                .expect_err("depth entries without metadata must fail"),
            DatasetError::ManifestDepthEntriesWithoutMetadata { entry_count: 0 }
        ));
        std::fs::remove_dir_all(unconfigured_path).expect("remove incoherent dataset");

        let reordered_path = unique_dataset_path("depth-entry-order");
        let (writer, handle) =
            DatasetWriter::create(&reordered_path, &meta_with_depth(2, 2), &calibration())
                .expect("create depth dataset");
        for timestamp_ns in [10, 20] {
            assert_eq!(
                write_outcome(writer.write_depth(&depth_image(2, 2, timestamp_ns))),
                WriteOutcome::Enqueued
            );
        }
        drop(writer);
        handle.finish().expect("finish depth dataset");
        let manifest_path = reordered_path.join(format::MANIFEST_FILE);
        let original: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&manifest_path).expect("read generated manifest"),
        )
        .expect("parse generated manifest");

        let mut duplicate = original.clone();
        duplicate["depth_entries"][1]["timestamp_ns"] = serde_json::json!(10);
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&duplicate).expect("serialize duplicate manifest"),
        )
        .expect("write duplicate manifest");
        assert!(matches!(
            DatasetReader::open(&reordered_path).expect_err("duplicate depth timestamps must fail"),
            DatasetError::NonMonotonicManifestDepthEntries {
                previous_depth_timestamp_ns: 10,
                current_depth_timestamp_ns: 10
            }
        ));

        let mut reordered = original;
        reordered["depth_entries"]
            .as_array_mut()
            .expect("depth entries array")
            .swap(0, 1);
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&reordered).expect("serialize reordered manifest"),
        )
        .expect("write reordered manifest");
        assert!(matches!(
            DatasetReader::open(&reordered_path)
                .expect_err("decreasing depth timestamps must fail"),
            DatasetError::NonMonotonicManifestDepthEntries {
                previous_depth_timestamp_ns: 20,
                current_depth_timestamp_ns: 10
            }
        ));
        std::fs::remove_dir_all(reordered_path).expect("remove reordered dataset");
    }

    #[test]
    fn reader_validates_depth_reference_topology_at_open() {
        let path = unique_dataset_path("depth-reference-topology");
        let (writer, handle) = DatasetWriter::create(&path, &meta_with_depth(2, 2), &calibration())
            .expect("create depth dataset");
        assert_eq!(
            write_outcome(writer.write_depth(&depth_image(2, 2, 7))),
            WriteOutcome::Enqueued
        );
        drop(writer);
        handle.finish().expect("finish depth dataset");

        let manifest_path = path.join(format::MANIFEST_FILE);
        let original: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&manifest_path).expect("read generated manifest"),
        )
        .expect("parse generated manifest");
        let frames = path.join(format::FRAMES_DIR);
        std::fs::write(
            frames.join(format::frame_name(8, "depth")),
            1.0_f32.to_le_bytes().repeat(4),
        )
        .expect("write alternate in-root payload");

        let mut wrong_identity = original.clone();
        wrong_identity["depth_entries"][0]["path"] =
            serde_json::Value::String(frame_path(8, "depth"));
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&wrong_identity).expect("serialize wrong identity"),
        )
        .expect("write wrong identity");
        assert!(matches!(
            DatasetReader::open(&path).expect_err("timestamp/path identity mismatch must fail"),
            DatasetError::ManifestFrameIdentityMismatch { .. }
        ));

        let mut escaping = original.clone();
        escaping["depth_entries"][0]["path"] =
            serde_json::Value::String("../outside.raw".to_string());
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&escaping).expect("serialize escaping identity"),
        )
        .expect("write escaping identity");
        assert!(matches!(
            DatasetReader::open(&path).expect_err("escaping depth identity must fail"),
            DatasetError::InvalidFramePath { .. }
        ));

        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&original).expect("serialize original manifest"),
        )
        .expect("restore original manifest");
        let payload_path = frames.join(format::frame_name(7, "depth"));
        std::fs::remove_file(&payload_path).expect("remove depth payload");
        std::fs::create_dir(&payload_path).expect("replace depth payload with directory");
        assert!(matches!(
            DatasetReader::open(&path).expect_err("non-regular depth identity must fail"),
            DatasetError::InvalidFrameFileType { .. }
        ));

        std::fs::remove_dir(&payload_path).expect("remove depth payload directory");
        std::fs::write(&payload_path, [0_u8; 12]).expect("write short depth payload");
        assert!(matches!(
            DatasetReader::open(&path).expect_err("wrong depth payload length must fail"),
            DatasetError::InvalidFrameLength {
                expected: 16,
                actual: 12,
                ..
            }
        ));

        std::fs::remove_dir_all(path).expect("remove topology dataset");
    }

    #[test]
    fn depth_cursor_retries_length_and_decode_failures_without_advancing() {
        let path = unique_dataset_path("depth-cursor-retry");
        let (writer, handle) = DatasetWriter::create(&path, &meta_with_depth(2, 2), &calibration())
            .expect("create depth dataset");
        assert_eq!(
            write_outcome(writer.write_depth(&depth_image(2, 2, 7))),
            WriteOutcome::Enqueued
        );
        drop(writer);
        handle.finish().expect("finish depth dataset");

        let reader = DatasetReader::open(&path).expect("open depth dataset");
        let mut cursor = reader.depth_cursor().expect("depth cursor");
        let payload_path = path
            .join(format::FRAMES_DIR)
            .join(format::frame_name(7, "depth"));
        std::fs::write(&payload_path, [0_u8; 12]).expect("truncate payload after open");
        assert!(matches!(
            cursor
                .next_at_or_before(Timestamp::from_nanos(7))
                .expect_err("post-open length change must fail"),
            DatasetError::DepthPayloadLengthChanged {
                expected: 16,
                actual: 12,
                ..
            }
        ));
        assert_eq!(cursor.remaining(), 1);

        let mut invalid_payload = Vec::with_capacity(16);
        for value in [f32::NAN, 0.0, 0.0, 0.0] {
            invalid_payload.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&payload_path, invalid_payload).expect("write invalid depth payload");
        let error = cursor
            .next_at_or_before(Timestamp::from_nanos(7))
            .expect_err("invalid sample domain must fail");
        assert!(matches!(
            &error,
            DatasetError::InvalidDepthData {
                source: crate::DepthImageError::InvalidDepthMeters { index: 0, value },
                ..
            } if value.is_nan()
        ));
        assert!(
            std::error::Error::source(&error)
                .is_some_and(|source| source.downcast_ref::<crate::DepthImageError>().is_some()),
            "dataset error must preserve the DepthImage validation source"
        );
        assert_eq!(cursor.remaining(), 1);

        std::fs::write(&payload_path, 4.0_f32.to_le_bytes().repeat(4))
            .expect("restore valid depth payload");
        let depth = cursor
            .next_at_or_before(Timestamp::from_nanos(7))
            .expect("retry valid depth payload")
            .expect("pending entry remains available after failures");
        assert_eq!(depth.frame_id(), FrameId::new(0));
        assert_eq!(depth.timestamp(), Timestamp::from_nanos(7));
        assert_eq!(depth.depth_m(), &[4.0; 4]);
        assert_eq!(cursor.remaining(), 0);
        std::fs::remove_dir_all(path).expect("remove retry dataset");
    }

    #[cfg(unix)]
    #[test]
    fn reader_tracks_resolved_frame_paths_across_stereo_and_depth_streams() {
        use std::os::unix::fs::symlink;

        let path = unique_dataset_path("shared-stereo-depth-path");
        let (writer, handle) = DatasetWriter::create(&path, &meta_with_depth(1, 1), &calibration())
            .expect("create depth dataset");
        assert_eq!(
            write_outcome(writer.write_frame(&frame(SensorId::StereoLeft, 5, 1))),
            WriteOutcome::Enqueued
        );
        assert_eq!(
            write_outcome(writer.write_frame(&frame(SensorId::StereoRight, 5, 2))),
            WriteOutcome::Enqueued
        );
        assert_eq!(
            write_outcome(writer.write_depth(&depth_image(1, 1, 5))),
            WriteOutcome::Enqueued
        );
        drop(writer);
        handle.finish().expect("finish depth dataset");

        let frames = path.join(format::FRAMES_DIR);
        let left_path = frames.join(format::frame_name(5, "mono_left"));
        let depth_path = frames.join(format::frame_name(5, "depth"));
        std::fs::remove_file(&depth_path).expect("remove original depth payload");
        symlink(&left_path, &depth_path).expect("alias depth identity to stereo payload");

        assert!(matches!(
            DatasetReader::open(&path).expect_err("cross-stream path alias must fail"),
            DatasetError::DuplicateManifestFrameRef { .. }
        ));
        std::fs::remove_dir_all(path).expect("remove aliased dataset");
    }

    #[test]
    fn changed_recorded_depth_payload_prevents_manifest_publication() {
        let path = unique_dataset_path("depth-manifest-mutated-payload");
        let (writer, handle) = DatasetWriter::create(&path, &meta_with_depth(2, 2), &calibration())
            .expect("create depth dataset");
        assert_eq!(
            write_outcome(writer.write_depth(&depth_image(2, 2, 7))),
            WriteOutcome::Enqueued
        );

        let deadline = Instant::now() + Duration::from_secs(2);
        while writer.stats().frames_written < 1 {
            assert!(writer.is_healthy(), "writer failed before payload mutation");
            assert!(
                Instant::now() < deadline,
                "writer did not persist the depth payload"
            );
            std::thread::yield_now();
        }
        let depth_path = path
            .join(format::FRAMES_DIR)
            .join(format::frame_name(7, "depth"));
        std::fs::write(&depth_path, [0_u8; 12]).expect("truncate recorded depth payload");

        drop(writer);
        assert!(matches!(
            handle
                .finish()
                .expect_err("changed recorded depth must fail finalization"),
            DatasetError::InvalidFrameLength {
                path: failed_path,
                expected: 16,
                actual: 12,
            } if failed_path == depth_path
        ));
        assert!(!path.join(format::MANIFEST_FILE).exists());
        std::fs::remove_dir_all(path).expect("remove mutated depth dataset");
    }

    #[test]
    fn failed_depth_write_never_records_a_manifest_identity() {
        let path = unique_dataset_path("depth-manifest-failed-write");
        let (writer, handle) = DatasetWriter::create(&path, &meta_with_depth(2, 2), &calibration())
            .expect("create depth dataset");
        let depth_path = path
            .join(format::FRAMES_DIR)
            .join(format::frame_name(9, "depth"));
        std::fs::write(&depth_path, [0_u8; 16]).expect("occupy depth payload path");
        assert_eq!(
            write_outcome(writer.write_depth(&depth_image(2, 2, 9))),
            WriteOutcome::Enqueued
        );
        drop(writer);
        let state = Arc::clone(&handle.state);

        assert!(matches!(
            handle.finish().expect_err("occupied depth path must fail"),
            DatasetError::WriteFile { path: failed_path, source }
                if failed_path == depth_path
                    && source.kind() == std::io::ErrorKind::AlreadyExists
        ));
        assert!(
            state
                .recorded_depth
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .is_empty()
        );
        assert!(!path.join(format::MANIFEST_FILE).exists());
        std::fs::remove_dir_all(path).expect("remove failed depth dataset");
    }

    #[test]
    fn mono_writers_reject_equal_area_wrong_shape_frames_without_a_manifest() {
        let mut dataset_meta = meta();
        let mono = dataset_meta.mono.as_mut().expect("mono metadata");
        mono.width = 2;
        mono.height = 6;
        let mut dataset_calibration = calibration();
        for intrinsics in [
            &mut dataset_calibration.left,
            &mut dataset_calibration.right,
        ] {
            intrinsics.width = 2;
            intrinsics.height = 6;
        }
        let wrong_left = Frame::new(
            SensorId::StereoLeft,
            FrameId::new(1),
            Timestamp::from_nanos(1),
            3,
            4,
            vec![0; 12],
        )
        .expect("valid equal-area frame");
        let wrong_right = Frame::new(
            SensorId::StereoRight,
            FrameId::new(2),
            Timestamp::from_nanos(1),
            3,
            4,
            vec![0; 12],
        )
        .expect("valid equal-area frame");

        let legacy_path = unique_dataset_path("legacy-wrong-shape");
        let (writer, handle) =
            DatasetWriter::create(&legacy_path, &dataset_meta, &dataset_calibration)
                .expect("create legacy writer");
        assert!(matches!(
            writer.write_frame(&wrong_left),
            Err(DatasetWriteError::FrameDimensionsMismatch {
                sensor: SensorId::StereoLeft,
                expected,
                actual,
            }) if (expected.width(), expected.height()) == (2, 6)
                && (actual.width(), actual.height()) == (3, 4)
        ));
        drop(writer);
        assert!(matches!(
            handle.finish().expect_err("invalid frame must fail finish"),
            DatasetError::WriteContract {
                source: DatasetWriteError::FrameDimensionsMismatch { .. }
            }
        ));
        assert!(!legacy_path.join(format::MANIFEST_FILE).exists());
        std::fs::remove_dir_all(legacy_path).expect("remove legacy dataset");

        let paired_path = unique_dataset_path("paired-wrong-shape");
        let window = PairingWindowNs::new(0).expect("valid exact window");
        let pair = StereoPair::try_new(wrong_left, wrong_right, window)
            .expect("valid same-shape stereo pair");
        let (writer, handle) =
            DatasetWriter::create_paired(&paired_path, &dataset_meta, &dataset_calibration, window)
                .expect("create paired writer");
        assert!(matches!(
            writer.write_pair(pair),
            Err(DatasetWriteError::FrameDimensionsMismatch { .. })
        ));
        drop(writer);
        assert!(matches!(
            handle.finish().expect_err("invalid pair must fail finish"),
            DatasetError::WriteContract {
                source: DatasetWriteError::FrameDimensionsMismatch { .. }
            }
        ));
        assert!(!paired_path.join(format::MANIFEST_FILE).exists());
        std::fs::remove_dir_all(paired_path).expect("remove paired dataset");
    }

    #[test]
    fn oversized_payload_is_a_typed_write_contract_error() {
        let path = unique_dataset_path("oversized-payload-contract");
        let config = DatasetWriterConfig {
            max_spool_frames: 1,
            max_spool_bytes: 3,
            flush_batch_frames: 1,
            backpressure: Backpressure::Block,
        };
        let (writer, handle) =
            DatasetWriter::create_with_config(&path, &meta(), &calibration(), config)
                .expect("create dataset");

        assert!(matches!(
            writer.write_frame(&frame(SensorId::StereoLeft, 1, 1)),
            Err(DatasetWriteError::SpoolByteCapacityExceeded {
                item: "mono frame",
                bytes: 4,
                max_bytes: 3
            })
        ));
        assert!(!writer.is_healthy());
        drop(writer);
        assert!(matches!(
            handle
                .finish()
                .expect_err("contract failure must fail finish"),
            DatasetError::WriteContract {
                source: DatasetWriteError::SpoolByteCapacityExceeded { .. }
            }
        ));
        assert!(!path.join(format::MANIFEST_FILE).exists());
        std::fs::remove_dir_all(path).expect("remove test directory");
    }

    #[test]
    fn duplicate_timestamp_does_not_overwrite_the_first_frame() {
        let path = unique_dataset_path("duplicate-timestamp");
        let (writer, handle) =
            DatasetWriter::create(&path, &meta(), &calibration()).expect("create dataset");
        let first = frame(SensorId::StereoLeft, 7, 1);
        let duplicate = frame(SensorId::StereoLeft, 7, 2);

        assert_eq!(
            write_outcome(writer.write_frame(&first)),
            WriteOutcome::Enqueued
        );
        assert_eq!(
            write_outcome(writer.write_frame(&duplicate)),
            WriteOutcome::Enqueued
        );
        drop(writer);

        let error = handle.finish().expect_err("duplicate filename must fail");
        assert!(matches!(error, DatasetError::WriteFile { .. }));
        let stored = std::fs::read(
            path.join(format::FRAMES_DIR)
                .join(format::frame_name(7, "mono_left")),
        )
        .expect("read first frame");
        assert_eq!(stored, vec![1; 4]);
        std::fs::remove_dir_all(path).expect("remove test directory");
    }

    #[test]
    fn exact_sync_dataset_round_trips_with_zero_pairing_window() {
        let path = unique_dataset_path("exact-sync");
        let (writer, handle) =
            DatasetWriter::create(&path, &meta(), &calibration()).expect("create dataset");

        assert_eq!(
            write_outcome(writer.write_frame(&frame(SensorId::StereoLeft, 11, 1))),
            WriteOutcome::Enqueued
        );
        assert_eq!(
            write_outcome(writer.write_frame(&frame(SensorId::StereoRight, 11, 2))),
            WriteOutcome::Enqueued
        );
        drop(writer);
        handle.finish().expect("finish dataset");

        let manifest = read_manifest(&path).expect("read manifest");
        assert_eq!(manifest.header.pairing_window_ns, 0);
        assert!(!path.join(".manifest.json.tmp").exists());
        let mut reader = DatasetReader::open(&path).expect("open exact-sync dataset");
        let pair = reader
            .pairs()
            .next()
            .expect("one pair")
            .expect("valid pair");
        assert_eq!(pair.timestamp_delta_ns(), 0);
        std::fs::remove_dir_all(path).expect("remove test directory");
    }

    #[test]
    fn finish_rejects_changed_writer_sidecars_without_reparsing_them() {
        let mut changed = meta();
        let mono = changed.mono.as_mut().expect("mono metadata");
        mono.width = 9;
        mono.height = 9;
        mono.fps = 99;
        let mut changed_calibration = calibration();
        changed_calibration.left.fx = 200.0;

        for (suffix, filename, replacement) in [
            (
                "metadata",
                format::META_FILE,
                serde_json::to_vec_pretty(&changed).expect("serialize changed metadata"),
            ),
            (
                "calibration",
                format::CALIBRATION_FILE,
                serde_json::to_vec_pretty(&changed_calibration)
                    .expect("serialize changed calibration"),
            ),
        ] {
            let path = unique_dataset_path(&format!("writer-contract-snapshot-{suffix}"));
            let (writer, handle) =
                DatasetWriter::create(&path, &meta(), &calibration()).expect("create dataset");
            let sidecar_path = path.join(filename);
            std::fs::write(&sidecar_path, replacement).expect("replace sidecar after creation");

            drop(writer);
            assert!(matches!(
                handle.finish().expect_err("changed sidecar must fail"),
                DatasetError::SidecarChanged { path: changed_path }
                    if changed_path == sidecar_path
            ));
            assert!(!path.join(format::MANIFEST_FILE).exists());
            std::fs::remove_dir_all(path).expect("remove test directory");
        }
    }

    #[test]
    fn manifest_is_not_visible_when_its_private_temporary_path_is_unavailable() {
        let path = unique_dataset_path("manifest-temp-collision");
        let (writer, handle) =
            DatasetWriter::create(&path, &meta(), &calibration()).expect("create dataset");
        let temporary_path = path.join(".manifest.json.tmp");
        let sentinel = b"not owned by the writer";
        std::fs::write(&temporary_path, sentinel).expect("occupy private temporary path");

        drop(writer);
        assert!(matches!(
            handle.finish().expect_err("temporary collision must fail"),
            DatasetError::WriteFile { path: failed_path, source }
                if failed_path == temporary_path
                    && source.kind() == std::io::ErrorKind::AlreadyExists
        ));
        assert!(!path.join(format::MANIFEST_FILE).exists());
        assert_eq!(
            std::fs::read(&temporary_path).expect("read untouched sentinel"),
            sentinel
        );
        std::fs::remove_dir_all(path).expect("remove test directory");
    }

    #[test]
    fn manifest_scan_ignores_noncanonical_filename_aliases() {
        let path = unique_dataset_path("noncanonical-frame-alias");
        let (writer, handle) =
            DatasetWriter::create(&path, &meta(), &calibration()).expect("create dataset");
        std::fs::write(
            path.join(format::FRAMES_DIR).join("01_mono_left.raw"),
            [0_u8; 4],
        )
        .expect("write noncanonical alias");

        drop(writer);
        handle.finish().expect("finish dataset");
        let manifest = read_manifest(&path).expect("read manifest");
        assert!(manifest.entries.is_empty());
        assert_eq!(manifest.stats.drops_by_reason.parse_fail, 1);
        std::fs::remove_dir_all(path).expect("remove test directory");
    }

    #[test]
    fn reader_requires_meta_calibration_and_manifest_dimensions_to_agree() {
        let calibration_path = unique_dataset_path("reader-calibration-dimensions");
        write_exact_pairs(&calibration_path, &[0]);
        let mut mismatched_calibration = calibration();
        mismatched_calibration.left.width = 1;
        mismatched_calibration.left.height = 4;
        std::fs::write(
            calibration_path.join(format::CALIBRATION_FILE),
            serde_json::to_vec_pretty(&mismatched_calibration)
                .expect("serialize calibration fixture"),
        )
        .expect("write calibration fixture");
        assert!(matches!(
            DatasetReader::open(&calibration_path)
                .expect_err("calibration dimensions must match metadata"),
            DatasetError::ImageDimensionsMismatch {
                expected_field: "meta.mono",
                actual_field: "calibration.left",
                ..
            }
        ));
        std::fs::remove_dir_all(calibration_path).expect("remove calibration dataset");

        let manifest_path = unique_dataset_path("reader-manifest-dimensions");
        write_exact_pairs(&manifest_path, &[0]);
        let completion_path = manifest_path.join(format::MANIFEST_FILE);
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&completion_path).expect("read manifest"))
                .expect("parse manifest");
        manifest["header"]["width"] = serde_json::json!(1);
        manifest["header"]["height"] = serde_json::json!(4);
        std::fs::write(
            &completion_path,
            serde_json::to_vec_pretty(&manifest).expect("serialize manifest fixture"),
        )
        .expect("write manifest fixture");
        assert!(matches!(
            DatasetReader::open(&manifest_path)
                .expect_err("manifest dimensions must match metadata"),
            DatasetError::ImageDimensionsMismatch {
                expected_field: "meta.mono",
                actual_field: "manifest.header",
                ..
            }
        ));
        std::fs::remove_dir_all(manifest_path).expect("remove manifest dataset");
    }

    #[test]
    fn reader_retains_exact_calibration_for_independent_rectification_policies() {
        let path = unique_dataset_path("reader-typed-calibration");
        write_exact_pairs(&path, &[0]);
        let smallest_positive = f32::from_bits(1);
        let mut calibration = calibration();
        calibration.left.fx = smallest_positive;
        calibration.right.fx = smallest_positive;
        calibration.left.fy = f32::MAX;
        calibration.right.fy = f32::MAX;
        calibration.left.cx = 1.0;
        calibration.right.cx = -smallest_positive;
        calibration.left.cy = -0.0;
        calibration.right.cy = 0.0;
        calibration.baseline_m = smallest_positive;
        std::fs::write(
            path.join(format::CALIBRATION_FILE),
            serde_json::to_vec_pretty(&calibration).expect("serialize calibration fixture"),
        )
        .expect("write calibration fixture");

        let reader = DatasetReader::open(&path).expect("open typed calibration");
        let strict = RectifiedStereoConfig::try_new(1.0, 0.0).expect("strict config");
        let loose = RectifiedStereoConfig::try_new(f32::from_bits(1.0_f32.to_bits() + 1), 0.0)
            .expect("loose config");

        for _ in 0..2 {
            assert!(matches!(
                RectifiedStereo::from_stereo_calibration_with_config(
                    reader.stereo_calibration(),
                    strict,
                ),
                Err(RectifiedStereoCompatibilityError::PrincipalPointMismatch {
                    left_cx: 1.0,
                    right_cx,
                    tolerance_px: 1.0,
                    ..
                }) if right_cx.to_bits() == (-smallest_positive).to_bits()
            ));

            let stereo = RectifiedStereo::from_stereo_calibration_with_config(
                reader.stereo_calibration(),
                loose,
            )
            .expect("loose policy accepts retained calibration");
            assert_eq!(stereo.left().fx().to_bits(), smallest_positive.to_bits());
            assert_eq!(stereo.right().fx().to_bits(), smallest_positive.to_bits());
            assert_eq!(stereo.left().fy().to_bits(), f32::MAX.to_bits());
            assert_eq!(stereo.right().fy().to_bits(), f32::MAX.to_bits());
            assert_eq!(stereo.left().cx().to_bits(), 1.0_f32.to_bits());
            assert_eq!(
                stereo.right().cx().to_bits(),
                (-smallest_positive).to_bits()
            );
            assert_eq!(stereo.left().cy().to_bits(), (-0.0_f32).to_bits());
            assert_eq!(stereo.right().cy().to_bits(), 0.0_f32.to_bits());
            assert_eq!(stereo.baseline_m().to_bits(), smallest_positive.to_bits());
            assert_eq!(stereo.dimensions(), FrameDimensions::new(2, 2));
        }

        std::fs::remove_dir_all(path).expect("remove typed calibration dataset");
    }

    #[test]
    fn reader_keeps_rectification_policy_out_of_dataset_open() {
        let path = unique_dataset_path("reader-unrectified-calibration");
        write_exact_pairs(&path, &[0]);
        let mut calibration = calibration();
        calibration.rectified = false;
        std::fs::write(
            path.join(format::CALIBRATION_FILE),
            serde_json::to_vec_pretty(&calibration).expect("serialize calibration fixture"),
        )
        .expect("write calibration fixture");

        let reader = DatasetReader::open(&path).expect("unrectified dataset remains readable");
        assert!(matches!(
            RectifiedStereo::from_stereo_calibration_with_config(
                reader.stereo_calibration(),
                RectifiedStereoConfig::default(),
            ),
            Err(RectifiedStereoCompatibilityError::NotRectified)
        ));

        std::fs::remove_dir_all(path).expect("remove unrectified dataset");
    }

    #[test]
    fn reader_parses_manifest_format_timebase_fps_and_window_once() {
        let path = unique_dataset_path("reader-header-contract");
        write_exact_pairs(&path, &[0]);
        let manifest_path = path.join(format::MANIFEST_FILE);
        let original: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).expect("read manifest"))
                .expect("parse manifest");

        for (field, value, expected_field) in [
            ("format", "jpeg", "header.format"),
            ("timebase", "host_ms", "header.timebase"),
            ("pairing_policy", "nearest", "header.pairing_policy"),
        ] {
            let mut invalid = original.clone();
            invalid["header"][field] = serde_json::Value::String(value.to_string());
            std::fs::write(
                &manifest_path,
                serde_json::to_vec_pretty(&invalid).expect("serialize invalid header"),
            )
            .expect("write invalid header");
            assert!(matches!(
                DatasetReader::open(&path).expect_err("unknown header value must fail"),
                DatasetError::UnsupportedManifestValue {
                    field: actual_field,
                    value: actual_value,
                } if actual_field == expected_field && actual_value == value
            ));
        }

        let mut zero_fps = original.clone();
        zero_fps["header"]["fps"] = serde_json::json!(0);
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&zero_fps).expect("serialize zero fps"),
        )
        .expect("write zero fps");
        assert!(matches!(
            DatasetReader::open(&path).expect_err("zero header fps must fail"),
            DatasetError::InvalidNominalFps {
                field: "manifest.header.fps",
                value: 0
            }
        ));

        let mut mismatched_fps = original.clone();
        mismatched_fps["header"]["fps"] = serde_json::json!(31);
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&mismatched_fps).expect("serialize mismatched fps"),
        )
        .expect("write mismatched fps");
        assert!(matches!(
            DatasetReader::open(&path).expect_err("header fps must match metadata"),
            DatasetError::NominalFpsMismatch {
                expected: 30,
                actual: 31,
                ..
            }
        ));

        let mut mismatched_device = original.clone();
        mismatched_device["header"]["device"] = serde_json::json!("other-device");
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&mismatched_device).expect("serialize mismatched device"),
        )
        .expect("write mismatched device");
        assert!(matches!(
            DatasetReader::open(&path).expect_err("header device must match metadata"),
            DatasetError::ManifestMetadataMismatch {
                expected_field: "meta.device",
                expected,
                actual_field: "manifest.header.device",
                actual,
            } if expected == "test-device" && actual == "other-device"
        ));

        let mut oversized_window = original;
        oversized_window["header"]["pairing_window_ns"] = serde_json::json!(u64::MAX);
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&oversized_window)
                .expect("serialize oversized pairing window"),
        )
        .expect("write oversized pairing window");
        assert!(matches!(
            DatasetReader::open(&path).expect_err("oversized pairing window must fail"),
            DatasetError::ManifestPairingWindowOutOfRange { value: u64::MAX }
        ));

        std::fs::remove_dir_all(path).expect("remove test directory");
    }

    #[test]
    fn recorded_pairs_policy_rejects_orphan_topology() {
        let path = unique_dataset_path("recorded-policy-orphan");
        let (writer, handle) =
            DatasetWriter::create(&path, &meta(), &calibration()).expect("create dataset");
        assert_eq!(
            write_outcome(writer.write_frame(&frame(SensorId::StereoLeft, 11, 1))),
            WriteOutcome::Enqueued
        );
        drop(writer);
        handle.finish().expect("finish orphan dataset");

        let manifest_path = path.join(format::MANIFEST_FILE);
        let original: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).expect("read manifest"))
                .expect("parse manifest");
        let mut declared_orphans = original.clone();
        declared_orphans["header"]["pairing_policy"] = serde_json::json!("recorded_pairs");
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&declared_orphans).expect("serialize orphan manifest"),
        )
        .expect("write orphan manifest");
        assert!(matches!(
            DatasetReader::open(&path).expect_err("recorded pairs cannot declare orphans"),
            DatasetError::RecordedPairsDeclareOrphans {
                left_orphans: 1,
                right_orphans: 0
            }
        ));

        let mut missing_right = original;
        missing_right["header"]["pairing_policy"] = serde_json::json!("recorded_pairs");
        missing_right["stats"]["left_orphans"] = serde_json::json!(0);
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&missing_right).expect("serialize missing-right manifest"),
        )
        .expect("write missing-right manifest");
        assert!(matches!(
            DatasetReader::open(&path).expect_err("recorded pair must contain both payloads"),
            DatasetError::RecordedPairsContainMissingRight {
                left_timestamp_ns: 11
            }
        ));

        std::fs::remove_dir_all(path).expect("remove test directory");
    }

    #[test]
    fn nominal_fps_is_not_compared_with_irregular_observed_timestamps() {
        let path = unique_dataset_path("irregular-observed-fps");
        write_exact_pairs(&path, &[0, 1, 2_000_000_000]);

        let reader = DatasetReader::open(&path).expect("nominal fps is metadata, not observation");
        assert_eq!(reader.stats().left_fps, Some(1.0));
        assert_eq!(reader.meta().mono.as_ref().expect("mono metadata").fps, 30);
        std::fs::remove_dir_all(path).expect("remove test directory");
    }

    #[test]
    fn paired_writer_preserves_capture_pair_identity_instead_of_repairing() {
        let path = unique_dataset_path("recorded-pair-identity");
        let window = PairingWindowNs::new(1).expect("valid pairing window");
        let (writer, handle) = DatasetWriter::create_paired(&path, &meta(), &calibration(), window)
            .expect("create paired dataset");

        for (left_timestamp_ns, right_timestamp_ns) in [(0, 0), (7, 6), (8, 7)] {
            assert_eq!(
                write_outcome(writer.write_pair(stereo_pair(
                    left_timestamp_ns,
                    right_timestamp_ns,
                    window,
                ))),
                WriteOutcome::Enqueued
            );
        }
        drop(writer);
        let stats = handle.finish().expect("finish paired dataset");
        assert_eq!(stats.frames_enqueued, 6);
        assert_eq!(stats.frames_written, 6);

        let manifest = read_manifest(&path).expect("read manifest");
        assert_eq!(manifest.header.pairing_policy, "recorded_pairs");
        assert_eq!(manifest.header.pairing_window_ns, 1);
        let manifest_pairs: Vec<(i64, i64)> = manifest
            .entries
            .iter()
            .map(|entry| {
                let ManifestPairing::Paired { right, .. } = &entry.pairing else {
                    panic!("recorded pair must remain paired");
                };
                (entry.left.timestamp_ns, right.timestamp_ns)
            })
            .collect();
        assert_eq!(manifest_pairs, [(0, 0), (7, 6), (8, 7)]);

        let mut reader = DatasetReader::open(&path).expect("open paired dataset");
        let replayed_pairs: Vec<(i64, i64)> = reader
            .pairs()
            .map(|pair| {
                let pair = pair.expect("read recorded pair");
                (
                    pair.left().timestamp().as_nanos(),
                    pair.right().timestamp().as_nanos(),
                )
            })
            .collect();
        assert_eq!(replayed_pairs, manifest_pairs);
        std::fs::remove_dir_all(path).expect("remove test directory");
    }

    #[test]
    fn paired_writer_rejects_a_pair_outside_its_declared_window() {
        let path = unique_dataset_path("recorded-pair-window");
        let exact_window = PairingWindowNs::new(0).expect("valid exact window");
        let wider_window = PairingWindowNs::new(1).expect("valid wider window");
        let (writer, handle) =
            DatasetWriter::create_paired(&path, &meta(), &calibration(), exact_window)
                .expect("create paired dataset");

        assert!(matches!(
            writer.write_pair(stereo_pair(0, 1, wider_window)),
            Err(DatasetWriteError::PairOutsideWriterWindow {
                delta_ns: 1,
                max_delta_ns: 0
            })
        ));
        assert!(!writer.is_healthy());
        drop(writer);
        assert!(matches!(
            handle
                .finish()
                .expect_err("writer contract must fail finish"),
            DatasetError::WriteContract {
                source: DatasetWriteError::PairOutsideWriterWindow {
                    delta_ns: 1,
                    max_delta_ns: 0
                }
            }
        ));
        assert!(!path.join(format::MANIFEST_FILE).exists());
        std::fs::remove_dir_all(path).expect("remove test directory");
    }

    #[test]
    fn paired_writer_rejects_a_queue_that_cannot_hold_an_atomic_pair() {
        let path = unique_dataset_path("recorded-pair-capacity");
        let window = PairingWindowNs::new(0).expect("valid exact window");
        let config = DatasetWriterConfig {
            max_spool_frames: 1,
            max_spool_bytes: 64,
            flush_batch_frames: 1,
            backpressure: Backpressure::Block,
        };
        let result = DatasetWriter::create_paired_with_config(
            &path,
            &meta(),
            &calibration(),
            window,
            config,
        );
        assert!(matches!(
            result.expect_err("undersized queue must fail before creating the dataset"),
            DatasetError::InvalidConfig {
                msg: "paired writer max_spool_frames must be >= 2"
            }
        ));
        assert!(!path.exists());
    }

    #[test]
    fn paired_writer_drop_newest_never_admits_only_one_side() {
        let window = PairingWindowNs::new(0).expect("valid exact window");
        for (max_spool_frames, max_spool_bytes) in [(2, 12), (4, 11)] {
            let config = DatasetWriterConfig {
                max_spool_frames,
                max_spool_bytes,
                flush_batch_frames: 1,
                backpressure: Backpressure::DropNewest,
            };
            let state = Arc::new(WriterState::new(
                config,
                PathBuf::new(),
                PathBuf::new(),
                ManifestMode::PreservePairs {
                    pairing_window: window,
                },
                writer_contract(),
                writer_sidecars(),
            ));
            {
                let mut spool = state.spool.lock().expect("lock spool");
                spool
                    .queue
                    .push_back(SpoolItem::Mono(frame(SensorId::StereoLeft, 10, 1)));
                spool.frames = 1;
                spool.bytes = 4;
            }
            state.enqueued.store(1, Ordering::Relaxed);
            state.bytes_enqueued.store(4, Ordering::Relaxed);
            state.spool_frames.store(1, Ordering::Relaxed);
            state.spool_bytes.store(4, Ordering::Relaxed);
            let writer = PairedDatasetWriter {
                inner: DatasetWriter {
                    config,
                    state: Arc::clone(&state),
                },
                pairing_window: window,
            };

            assert_eq!(
                write_outcome(writer.write_pair(stereo_pair(11, 11, window))),
                WriteOutcome::Dropped
            );
            let stats = writer.stats();
            assert_eq!(stats.frames_enqueued, 1);
            assert_eq!(stats.bytes_enqueued, 4);
            assert_eq!(stats.frames_dropped, 2);
            assert_eq!(stats.bytes_dropped, 8);
            let spool = state.spool.lock().expect("lock spool");
            assert_eq!(spool.queue.len(), 1);
            assert_eq!(spool.frames, 1);
            assert_eq!(spool.bytes, 4);
        }
    }

    #[test]
    fn spool_capacity_remains_reserved_while_a_batch_item_is_in_flight() {
        let config = DatasetWriterConfig {
            max_spool_frames: 1,
            max_spool_bytes: 4,
            flush_batch_frames: 1,
            backpressure: Backpressure::DropNewest,
        };
        let state = Arc::new(WriterState::new(
            config,
            PathBuf::new(),
            PathBuf::new(),
            ManifestMode::InferPairs,
            writer_contract(),
            writer_sidecars(),
        ));
        let writer = DatasetWriter {
            config,
            state: Arc::clone(&state),
        };
        assert_eq!(
            write_outcome(writer.write_frame(&frame(SensorId::StereoLeft, 1, 1))),
            WriteOutcome::Enqueued
        );

        let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(0);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let worker_state = Arc::clone(&state);
        let worker = std::thread::spawn(move || {
            writer_loop_with(worker_state, move |_item| {
                entered_tx.send(()).expect("signal in-flight item");
                release_rx.recv().expect("release in-flight item");
                Ok(WrittenManifestItem::None)
            });
        });
        entered_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("worker must enter controlled sink");

        let in_flight = writer.stats();
        assert_eq!(in_flight.spool_frames, 1);
        assert_eq!(in_flight.spool_bytes, 4);
        assert_eq!(
            write_outcome(writer.write_frame(&frame(SensorId::StereoRight, 2, 2))),
            WriteOutcome::Dropped
        );
        assert_eq!(writer.stats().spool_frames, 1);
        drop(writer);

        release_tx.send(()).expect("release controlled sink");
        worker.join().expect("join controlled writer");
        let completed = state.stats();
        assert_eq!(completed.frames_enqueued, 1);
        assert_eq!(completed.frames_written, 1);
        assert_eq!(completed.frames_dropped, 1);
        assert_eq!(completed.spool_frames, 0);
        assert_eq!(completed.spool_bytes, 0);
    }

    #[test]
    fn pending_writer_stats_exclude_completed_failures_but_not_rejected_work() {
        let stats = WriterStats {
            frames_enqueued: 8,
            frames_written: 2,
            frames_dropped: 20,
            bytes_enqueued: 32,
            bytes_written: 8,
            bytes_dropped: 80,
            write_failed: 2,
            bytes_write_failed: 8,
            frames_canceled: 2,
            bytes_canceled: 8,
            spool_frames: 2,
            spool_bytes: 8,
            spool_max_frames: 64,
            spool_max_bytes: 1024,
            writer_failed: true,
        };

        assert_eq!(stats.frames_pending(), 2);
        assert_eq!(stats.bytes_pending(), 8);
    }

    #[test]
    fn writer_failure_cancels_the_rest_of_its_batch_and_queue() {
        let path = unique_dataset_path("writer-failure-cancels-unwritten");
        let frames_dir = path.join(format::FRAMES_DIR);
        std::fs::create_dir_all(&frames_dir).expect("create frames directory");
        let failed_path = frames_dir.join(format::frame_name(11, "mono_left"));
        std::fs::write(&failed_path, [9_u8; 4]).expect("occupy first payload path");

        let config = DatasetWriterConfig {
            max_spool_frames: 3,
            max_spool_bytes: 12,
            flush_batch_frames: 2,
            backpressure: Backpressure::Block,
        };
        let state = Arc::new(WriterState::new(
            config,
            path.clone(),
            frames_dir.clone(),
            ManifestMode::InferPairs,
            writer_contract(),
            writer_sidecars(),
        ));
        {
            let mut spool = state.spool.lock().expect("lock spool");
            spool.queue.extend([
                SpoolItem::Mono(frame(SensorId::StereoLeft, 11, 1)),
                SpoolItem::Mono(frame(SensorId::StereoRight, 12, 2)),
                SpoolItem::Mono(frame(SensorId::StereoLeft, 13, 3)),
            ]);
            spool.frames = 3;
            spool.bytes = 12;
            spool.closed = true;
        }
        state.enqueued.store(3, Ordering::Relaxed);
        state.bytes_enqueued.store(12, Ordering::Relaxed);
        state.spool_frames.store(3, Ordering::Relaxed);
        state.spool_bytes.store(12, Ordering::Relaxed);

        writer_loop(frames_dir.clone(), Arc::clone(&state));

        let stats = state.stats();
        assert_eq!(stats.write_failed, 1);
        assert_eq!(stats.bytes_write_failed, 4);
        assert_eq!(stats.frames_canceled, 2);
        assert_eq!(stats.bytes_canceled, 8);
        assert_eq!(stats.frames_pending(), 0);
        assert_eq!(stats.bytes_pending(), 0);
        assert_eq!(stats.spool_frames, 0);
        assert_eq!(stats.spool_bytes, 0);
        assert!(
            !frames_dir
                .join(format::frame_name(12, "mono_right"))
                .exists()
        );
        assert!(
            !frames_dir
                .join(format::frame_name(13, "mono_left"))
                .exists()
        );

        std::fs::remove_dir_all(path).expect("remove test directory");
    }

    #[test]
    fn second_pair_payload_failure_never_publishes_a_half_pair() {
        let path = unique_dataset_path("recorded-pair-second-write-failure");
        let window = PairingWindowNs::new(0).expect("valid exact window");
        let (writer, handle) = DatasetWriter::create_paired(&path, &meta(), &calibration(), window)
            .expect("create paired dataset");
        let right_path = path
            .join(format::FRAMES_DIR)
            .join(format::frame_name(11, "mono_right"));
        std::fs::write(&right_path, [9_u8; 4]).expect("occupy right payload path");

        assert_eq!(
            write_outcome(writer.write_pair(stereo_pair(11, 11, window))),
            WriteOutcome::Enqueued
        );
        drop(writer);
        let state = Arc::clone(&handle.state);
        assert!(matches!(
            handle
                .finish()
                .expect_err("second create_new write must fail"),
            DatasetError::WriteFile { path: failed_path, .. } if failed_path == right_path
        ));
        let stats = state.stats();
        assert_eq!(stats.write_failed, 2);
        assert_eq!(stats.bytes_write_failed, 8);
        assert_eq!(stats.frames_canceled, 0);
        assert_eq!(stats.bytes_canceled, 0);
        assert_eq!(stats.frames_pending(), 0);
        assert_eq!(stats.bytes_pending(), 0);

        assert!(
            !path.join(format::MANIFEST_FILE).exists(),
            "a failed pair transaction must not publish a completion manifest"
        );
        assert!(
            path.join(format::FRAMES_DIR)
                .join(format::frame_name(11, "mono_left"))
                .exists(),
            "the first payload may remain as an explicitly unreferenced partial write"
        );
        std::fs::remove_dir_all(path).expect("remove test directory");
    }

    #[test]
    fn recorded_pair_finalization_preserves_payload_failure_causes() {
        let window = PairingWindowNs::new(0).expect("valid exact window");

        let missing_path = unique_dataset_path("recorded-payload-missing");
        let (writer, handle) =
            DatasetWriter::create_paired(&missing_path, &meta(), &calibration(), window)
                .expect("create paired dataset");
        assert_eq!(
            write_outcome(writer.write_pair(stereo_pair(21, 21, window))),
            WriteOutcome::Enqueued
        );
        wait_for_written_frames(&writer, 2);
        let missing_payload = missing_path
            .join(format::FRAMES_DIR)
            .join(format::frame_name(21, "mono_right"));
        std::fs::remove_file(&missing_payload).expect("remove recorded payload");
        drop(writer);
        assert!(matches!(
            handle.finish().expect_err("missing payload must fail finish"),
            DatasetError::ReadFile { path, source }
                if path == missing_payload && source.kind() == std::io::ErrorKind::NotFound
        ));
        assert!(!missing_path.join(format::MANIFEST_FILE).exists());
        std::fs::remove_dir_all(missing_path).expect("remove missing-payload dataset");

        let truncated_path = unique_dataset_path("recorded-payload-truncated");
        let (writer, handle) =
            DatasetWriter::create_paired(&truncated_path, &meta(), &calibration(), window)
                .expect("create paired dataset");
        assert_eq!(
            write_outcome(writer.write_pair(stereo_pair(22, 22, window))),
            WriteOutcome::Enqueued
        );
        wait_for_written_frames(&writer, 2);
        let truncated_payload = truncated_path
            .join(format::FRAMES_DIR)
            .join(format::frame_name(22, "mono_right"));
        std::fs::write(&truncated_payload, [0_u8; 3]).expect("truncate recorded payload");
        drop(writer);
        assert!(matches!(
            handle.finish().expect_err("truncated payload must fail finish"),
            DatasetError::InvalidFrameLength {
                path,
                expected: 4,
                actual: 3
            } if path == truncated_payload
        ));
        assert!(!truncated_path.join(format::MANIFEST_FILE).exists());
        std::fs::remove_dir_all(truncated_path).expect("remove truncated-payload dataset");
    }

    #[test]
    fn dataset_stats_follow_manifest_topology_and_sample_intervals() {
        let path = unique_dataset_path("manifest-stats");
        write_exact_pairs(&path, &[0, 1_000_000_000]);

        for sensor in ["mono_left", "mono_right"] {
            std::fs::write(
                path.join(format::FRAMES_DIR)
                    .join(format::frame_name(2_000_000_000, sensor)),
                [0_u8; 4],
            )
            .expect("write unreferenced frame");
        }

        let reader = DatasetReader::open(&path).expect("open dataset");
        let stats = reader.stats();
        assert_eq!(stats.left_count, 2);
        assert_eq!(stats.right_count, 2);
        assert_eq!(stats.paired_count, 2);
        assert_eq!(stats.left_orphan_count, 0);
        assert_eq!(stats.right_orphan_count, 0);
        assert_eq!(stats.left_fps, Some(1.0));
        assert_eq!(stats.right_fps, Some(1.0));
        assert_eq!(stats.paired_fps, Some(1.0));

        std::fs::remove_dir_all(path).expect("remove test directory");
    }

    #[test]
    fn dataset_stats_count_explicit_pairs_and_reject_inconsistent_aggregates() {
        let path = unique_dataset_path("manifest-orphans");
        write_exact_pairs(&path, &[0, 1_000_000_000]);

        let manifest_path = path.join(format::MANIFEST_FILE);
        let mut manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&manifest_path).expect("read generated manifest"),
        )
        .expect("parse generated manifest");
        let left = manifest["entries"][1]["left"].clone();
        manifest["entries"][1] = serde_json::json!({
            "left": left,
            "status": "missing_right",
            "reason": "outside_window"
        });
        manifest["stats"]["paired_count"] = serde_json::json!(1);
        manifest["stats"]["left_orphans"] = serde_json::json!(1);
        manifest["stats"]["right_orphans"] = serde_json::json!(1);
        manifest["stats"]["drops_by_reason"]["outside_window"] = serde_json::json!(1);
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("serialize modified manifest"),
        )
        .expect("write modified manifest");

        let mut reader = DatasetReader::open(&path).expect("open consistent orphan manifest");
        let stats = reader.stats();
        assert_eq!(stats.left_count, 2);
        assert_eq!(stats.right_count, 2);
        assert_eq!(stats.paired_count, 1);
        assert_eq!(stats.left_orphan_count, 1);
        assert_eq!(stats.right_orphan_count, 1);
        assert_eq!(stats.left_fps, Some(1.0));
        assert_eq!(stats.right_fps, None);
        assert_eq!(stats.paired_fps, None);
        assert_eq!(
            reader
                .pairs()
                .collect::<Result<Vec<_>, DatasetError>>()
                .expect("read manifest pairs")
                .len(),
            stats.paired_count
        );

        manifest["stats"]["paired_count"] = serde_json::json!(2);
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("serialize inconsistent manifest"),
        )
        .expect("write inconsistent manifest");
        let error = DatasetReader::open(&path).expect_err("inconsistent aggregate must fail");
        assert!(matches!(
            error,
            DatasetError::InvalidManifestStats {
                field: "paired_count",
                declared: 2,
                derived: 1
            }
        ));

        std::fs::remove_dir_all(path).expect("remove test directory");
    }

    #[test]
    fn dataset_open_rejects_diagnostics_that_disagree_with_explicit_entries() {
        let path = unique_dataset_path("manifest-diagnostics");
        write_exact_pairs(&path, &[0, 1_000_000_000]);
        let manifest_path = path.join(format::MANIFEST_FILE);
        let original: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&manifest_path).expect("read generated manifest"),
        )
        .expect("parse generated manifest");

        let mut wrong_delta = original.clone();
        wrong_delta["stats"]["delta_stats"]["max"] = serde_json::json!(1);
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&wrong_delta).expect("serialize wrong delta stats"),
        )
        .expect("write wrong delta stats");
        assert!(matches!(
            DatasetReader::open(&path).expect_err("wrong delta stats must fail"),
            DatasetError::InvalidManifestStats {
                field: "delta_stats.max",
                declared: 1,
                derived: 0
            }
        ));

        let mut missing_delta = original.clone();
        missing_delta["stats"]["delta_stats"] = serde_json::Value::Null;
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&missing_delta).expect("serialize absent delta stats"),
        )
        .expect("write absent delta stats");
        assert!(matches!(
            DatasetReader::open(&path).expect_err("absent delta stats must fail"),
            DatasetError::ManifestDeltaStatsPresenceMismatch {
                declared: false,
                derived: true
            }
        ));

        let mut wrong_outside_window = original;
        let left = wrong_outside_window["entries"][1]["left"].clone();
        wrong_outside_window["entries"][1] = serde_json::json!({
            "left": left,
            "status": "missing_right",
            "reason": "outside_window"
        });
        wrong_outside_window["stats"]["paired_count"] = serde_json::json!(1);
        wrong_outside_window["stats"]["left_orphans"] = serde_json::json!(1);
        wrong_outside_window["stats"]["right_orphans"] = serde_json::json!(1);
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&wrong_outside_window)
                .expect("serialize wrong outside-window count"),
        )
        .expect("write wrong outside-window count");
        assert!(matches!(
            DatasetReader::open(&path).expect_err("wrong outside-window count must fail"),
            DatasetError::InvalidManifestStats {
                field: "drops_by_reason.outside_window",
                declared: 0,
                derived: 1
            }
        ));

        std::fs::remove_dir_all(path).expect("remove test directory");
    }

    #[test]
    fn dataset_open_rejects_noncanonical_and_duplicate_frame_references() {
        let path = unique_dataset_path("manifest-frame-identity");
        write_exact_pairs(&path, &[0, 1_000_000_000]);
        let manifest_path = path.join(format::MANIFEST_FILE);
        let original: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&manifest_path).expect("read generated manifest"),
        )
        .expect("parse generated manifest");

        let mut wrong_side = original.clone();
        wrong_side["entries"][0]["left"]["path"] = original["entries"][0]["right"]["path"].clone();
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&wrong_side).expect("serialize wrong-side manifest"),
        )
        .expect("write wrong-side manifest");
        assert!(matches!(
            DatasetReader::open(&path).expect_err("left ref to right payload must fail"),
            DatasetError::ManifestFrameIdentityMismatch { .. }
        ));

        let mut wrong_timestamp = original.clone();
        wrong_timestamp["entries"][0]["left"]["timestamp_ns"] = serde_json::json!(1);
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&wrong_timestamp)
                .expect("serialize wrong-timestamp manifest"),
        )
        .expect("write wrong-timestamp manifest");
        assert!(matches!(
            DatasetReader::open(&path).expect_err("timestamp/path mismatch must fail"),
            DatasetError::ManifestFrameIdentityMismatch { .. }
        ));

        let mut duplicate_right = original;
        duplicate_right["entries"][1]["right"] = duplicate_right["entries"][0]["right"].clone();
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&duplicate_right).expect("serialize duplicate-ref manifest"),
        )
        .expect("write duplicate-ref manifest");
        assert!(matches!(
            DatasetReader::open(&path).expect_err("duplicate frame ref must fail"),
            DatasetError::DuplicateManifestFrameRef { .. }
        ));

        std::fs::remove_dir_all(path).expect("remove test directory");
    }

    #[test]
    fn dataset_open_rejects_nonmonotonic_left_entries() {
        let path = unique_dataset_path("manifest-entry-order");
        write_exact_pairs(&path, &[0, 1_000_000_000]);
        let manifest_path = path.join(format::MANIFEST_FILE);
        let mut manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&manifest_path).expect("read generated manifest"),
        )
        .expect("parse generated manifest");
        manifest["entries"]
            .as_array_mut()
            .expect("manifest entries array")
            .swap(0, 1);
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("serialize reordered manifest"),
        )
        .expect("write reordered manifest");

        assert!(matches!(
            DatasetReader::open(&path).expect_err("decreasing left timestamps must fail"),
            DatasetError::NonMonotonicManifestEntries {
                previous_left_timestamp_ns: 1_000_000_000,
                current_left_timestamp_ns: 0
            }
        ));

        std::fs::remove_dir_all(path).expect("remove test directory");
    }

    #[test]
    fn dataset_open_parses_declared_pair_delta_and_window_once() {
        let path = unique_dataset_path("manifest-pair-delta");
        write_exact_pairs(&path, &[0]);
        let manifest_path = path.join(format::MANIFEST_FILE);
        let original: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&manifest_path).expect("read generated manifest"),
        )
        .expect("parse generated manifest");

        let mut inconsistent_delta = original.clone();
        inconsistent_delta["entries"][0]["delta_ns"] = serde_json::json!(1);
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&inconsistent_delta)
                .expect("serialize inconsistent-delta manifest"),
        )
        .expect("write inconsistent-delta manifest");
        assert!(matches!(
            DatasetReader::open(&path).expect_err("declared pair delta must be exact"),
            DatasetError::ManifestPairDeltaMismatch {
                declared_delta_ns: 1,
                derived_delta_ns: 0,
                ..
            }
        ));

        let shifted_right_path = path
            .join(format::FRAMES_DIR)
            .join(format::frame_name(1, "mono_right"));
        std::fs::write(&shifted_right_path, [2_u8; 4]).expect("write shifted right fixture");
        let mut outside_window = original;
        outside_window["entries"][0]["right"]["timestamp_ns"] = serde_json::json!(1);
        outside_window["entries"][0]["right"]["path"] = serde_json::json!(format!(
            "{}/{}",
            format::FRAMES_DIR,
            format::frame_name(1, "mono_right")
        ));
        outside_window["entries"][0]["delta_ns"] = serde_json::json!(1);
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&outside_window).expect("serialize outside-window manifest"),
        )
        .expect("write outside-window manifest");
        assert!(matches!(
            DatasetReader::open(&path).expect_err("pair outside declared window must fail open"),
            DatasetError::ManifestPairOutsideWindow {
                delta_ns: 1,
                max_delta_ns: 0,
                ..
            }
        ));

        std::fs::remove_dir_all(path).expect("remove test directory");
    }

    #[test]
    fn dataset_open_rejects_invalid_and_post_open_frame_lengths() {
        let path = unique_dataset_path("frame-length");
        write_exact_pairs(&path, &[0]);
        let left_path = path
            .join(format::FRAMES_DIR)
            .join(format::frame_name(0, "mono_left"));
        let right_path = path
            .join(format::FRAMES_DIR)
            .join(format::frame_name(0, "mono_right"));

        std::fs::write(&left_path, [0_u8; 3]).expect("truncate referenced left frame");
        let error = DatasetReader::open(&path).expect_err("invalid frame length must fail open");
        assert!(matches!(
            error,
            DatasetError::InvalidFrameLength {
                expected: 4,
                actual: 3,
                ..
            }
        ));

        std::fs::write(&left_path, [0_u8; 4]).expect("restore referenced left frame");
        let mut reader = DatasetReader::open(&path).expect("open valid dataset");
        std::fs::write(&right_path, [0_u8; 3]).expect("truncate right frame after open");
        let error = reader
            .pairs()
            .next()
            .expect("one manifest pair")
            .expect_err("post-open frame mutation must fail");
        assert!(matches!(
            error,
            DatasetError::InvalidFrameData {
                source: crate::FrameError::DimensionMismatch {
                    expected: 4,
                    actual: 3
                },
                ..
            }
        ));

        std::fs::remove_dir_all(path).expect("remove test directory");
    }

    #[test]
    fn dataset_open_requires_complete_nonzero_mono_metadata() {
        let path = unique_dataset_path("mono-dimensions");
        write_exact_pairs(&path, &[0]);
        let meta_path = path.join(format::META_FILE);
        let original: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&meta_path).expect("read generated metadata"))
                .expect("parse generated metadata");

        let mut missing = original.clone();
        missing["mono"] = serde_json::Value::Null;
        std::fs::write(
            &meta_path,
            serde_json::to_vec_pretty(&missing).expect("serialize missing mono metadata"),
        )
        .expect("write missing mono metadata");
        assert!(matches!(
            DatasetReader::open(&path).expect_err("missing mono config must fail open"),
            DatasetError::MissingMonoConfig
        ));

        let mut zero_width = original.clone();
        zero_width["mono"]["width"] = serde_json::json!(0);
        std::fs::write(
            &meta_path,
            serde_json::to_vec_pretty(&zero_width).expect("serialize zero-width metadata"),
        )
        .expect("write zero-width metadata");
        assert!(matches!(
            DatasetReader::open(&path).expect_err("zero frame dimension must fail open"),
            DatasetError::InvalidFrameDimensions {
                source: crate::FrameDimensionsError::Zero {
                    width: 0,
                    height: 2
                },
                ..
            }
        ));

        let mut zero_fps = original;
        zero_fps["mono"]["fps"] = serde_json::json!(0);
        std::fs::write(
            &meta_path,
            serde_json::to_vec_pretty(&zero_fps).expect("serialize zero-fps metadata"),
        )
        .expect("write zero-fps metadata");
        assert!(matches!(
            DatasetReader::open(&path).expect_err("zero nominal fps must fail open"),
            DatasetError::InvalidNominalFps {
                field: "meta.mono.fps",
                value: 0
            }
        ));

        std::fs::remove_dir_all(path).expect("remove test directory");
    }

    #[test]
    fn manifest_frame_paths_are_confined_to_the_dataset_root() {
        let path = unique_dataset_path("confined-manifest-paths");
        let (writer, handle) =
            DatasetWriter::create(&path, &meta(), &calibration()).expect("create dataset");
        assert_eq!(
            write_outcome(writer.write_frame(&frame(SensorId::StereoLeft, 11, 1))),
            WriteOutcome::Enqueued
        );
        assert_eq!(
            write_outcome(writer.write_frame(&frame(SensorId::StereoRight, 11, 2))),
            WriteOutcome::Enqueued
        );
        drop(writer);
        handle.finish().expect("finish dataset");

        let manifest_path = path.join(format::MANIFEST_FILE);
        let original: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&manifest_path).expect("read generated manifest"),
        )
        .expect("parse generated manifest");

        for invalid in [
            "../outside.raw".to_string(),
            path.with_extension("outside.raw")
                .to_string_lossy()
                .into_owned(),
        ] {
            let mut manifest = original.clone();
            manifest["entries"][0]["left"]["path"] = serde_json::Value::String(invalid);
            std::fs::write(
                &manifest_path,
                serde_json::to_vec_pretty(&manifest).expect("serialize modified manifest"),
            )
            .expect("write modified manifest");

            let error = DatasetReader::open(&path)
                .expect_err("escaping manifest path must fail while opening the dataset");
            assert!(matches!(error, DatasetError::InvalidFramePath { .. }));
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;

            let outside = path.with_extension("outside.raw");
            std::fs::write(&outside, [0_u8; 4]).expect("write outside fixture");
            let link = path.join(format::FRAMES_DIR).join("outside-link.raw");
            symlink(&outside, &link).expect("create escaping symlink");
            let mut manifest = original.clone();
            manifest["entries"][0]["left"]["path"] =
                serde_json::Value::String("frames/outside-link.raw".to_string());
            std::fs::write(
                &manifest_path,
                serde_json::to_vec_pretty(&manifest).expect("serialize symlink manifest"),
            )
            .expect("write symlink manifest");

            let error = DatasetReader::open(&path)
                .expect_err("a normalized symlink must not escape the dataset root");
            assert!(matches!(error, DatasetError::InvalidFramePath { .. }));
            std::fs::remove_file(outside).expect("remove outside fixture");
        }

        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&original).expect("serialize original manifest"),
        )
        .expect("restore original manifest");
        DatasetReader::open(&path).expect("normalized in-root manifest path remains valid");

        std::fs::remove_dir_all(path).expect("remove test directory");
    }

    #[test]
    fn new_imu_writes_require_complete_metadata_before_filesystem_mutation() {
        let root = unique_dataset_path("imu-metadata-required");
        let path = root.join("missing-parent").join("dataset");
        assert!(matches!(
            DatasetWriter::create(&path, &meta_with_imu(200), &calibration())
                .expect_err("implicit IMU metadata must be rejected"),
            DatasetError::ImuMetadataRequiredForWrite
        ));
        assert!(!root.exists());

        let root = unique_dataset_path("imu-metadata-without-stream");
        let path = root.join("missing-parent").join("dataset");
        assert!(matches!(
            DatasetWriter::create_with_imu_config(
                &path,
                &meta(),
                &calibration(),
                imu_stream_metadata(7),
                DatasetWriterConfig::default(),
            )
            .expect_err("explicit metadata without meta.imu must be rejected"),
            DatasetError::ImuMetadataWithoutNominalStream
        ));
        assert!(!root.exists());

        let root = unique_dataset_path("imu-zero-rate");
        let path = root.join("missing-parent").join("dataset");
        assert!(matches!(
            DatasetWriter::create_with_imu_config(
                &path,
                &meta_with_imu(0),
                &calibration(),
                imu_stream_metadata(7),
                DatasetWriterConfig::default(),
            )
            .expect_err("zero nominal rate must be rejected"),
            DatasetError::InvalidImuNominalRate { value: 0 }
        ));
        assert!(!root.exists());

        assert!(matches!(
            ImuStreamMetadata::try_from_raw_session_id(
                0,
                ImuExtrinsicProvenance::uncalibrated_unknown()
            ),
            Err(ImuStreamMetadataError::ZeroDeviceSessionId)
        ));
        assert!(matches!(
            ImuExtrinsicProvenance::try_calibrated_to_tracking_camera("  "),
            Err(ImuStreamMetadataError::EmptyCalibrationSource)
        ));
    }

    #[test]
    fn imu_stream_round_trips_metadata_and_globally_merges_sensor_timestamps() {
        let reports = [
            imu_report(7, 10, 100, 10, 40, 0.0),
            imu_report(7, 11, 110, 20, 50, 10.0),
            imu_report(7, 12, 120, 40, 60, 20.0),
        ];
        let path = write_imu_fixture("imu-roundtrip", &reports);

        let stream = std::fs::read(path.join(format::IMU_STREAM_FILE))
            .expect("read fixed-record IMU stream");
        assert_eq!(
            stream.len(),
            usize::from(IMU_STREAM_HEADER_BYTES)
                + reports.len() * usize::from(IMU_STREAM_RECORD_BYTES)
        );
        assert_eq!(&stream[0..8], &IMU_STREAM_MAGIC);
        assert_eq!(read_u16(&stream, 8), IMU_STREAM_VERSION);
        assert_eq!(read_u16(&stream, 10), IMU_STREAM_HEADER_BYTES);
        assert_eq!(read_u16(&stream, 12), IMU_STREAM_RECORD_BYTES);
        assert_eq!(read_u64(&stream, 16), 3);
        assert_eq!(read_u32(&stream, usize::from(IMU_STREAM_HEADER_BYTES)), 10);

        let manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(path.join(format::MANIFEST_FILE)).expect("read manifest"),
        )
        .expect("parse manifest JSON");
        let imu = &manifest["imu_stream"];
        assert_eq!(imu["path"], format::IMU_STREAM_FILE);
        assert_eq!(imu["format"], IMU_STREAM_FORMAT);
        assert_eq!(imu["nominal_rate_hz"], 200);
        assert_eq!(imu["acceleration_unit"], IMU_ACCELERATION_UNIT);
        assert_eq!(imu["angular_velocity_unit"], IMU_ANGULAR_VELOCITY_UNIT);
        assert_eq!(imu["coordinate_frame"], IMU_COORDINATE_FRAME);
        assert_eq!(imu["axes"], IMU_AXES);
        assert_eq!(imu["extrinsic"]["status"], "uncalibrated_unknown");
        assert_eq!(imu["device_session_id"], 7);
        assert_eq!(imu["device_timebase"], IMU_DEVICE_TIMEBASE);
        assert_eq!(
            imu["device_session_semantics"],
            IMU_DEVICE_SESSION_SEMANTICS
        );
        assert_eq!(
            imu["sample_timestamp_semantics"],
            IMU_SAMPLE_TIMESTAMP_SEMANTICS
        );
        assert_eq!(imu["host_arrival_timebase"], IMU_HOST_ARRIVAL_TIMEBASE);

        let reader = DatasetReader::open(&path).expect("open IMU dataset");
        let mut cursor = reader.imu_cursor().expect("create IMU cursor");
        assert_eq!(cursor.len(), 6);
        assert_eq!(cursor.remaining(), 6);
        let events: Vec<_> = cursor
            .by_ref()
            .collect::<Result<_, _>>()
            .expect("replay validated IMU events");
        let observed: Vec<_> = events
            .iter()
            .map(|event| {
                (
                    event.device_timestamp().as_nanos(),
                    event.sensor(),
                    event.sequence().as_u32(),
                    event.host_arrival().as_nanos(),
                    event.session_id().as_u64(),
                )
            })
            .collect();
        assert_eq!(
            observed,
            vec![
                (10, InertialSensorKind::Accelerometer, 10, 100, 7),
                (20, InertialSensorKind::Accelerometer, 11, 110, 7),
                (40, InertialSensorKind::Accelerometer, 12, 120, 7),
                (40, InertialSensorKind::Gyroscope, 10, 100, 7),
                (50, InertialSensorKind::Gyroscope, 11, 110, 7),
                (60, InertialSensorKind::Gyroscope, 12, 120, 7),
            ]
        );
        assert_eq!(
            events[0]
                .as_accel()
                .expect("first event is acceleration")
                .acceleration()
                .as_array(),
            [1.0, 2.0, 3.0]
        );
        assert_eq!(
            events[3]
                .as_gyro()
                .expect("fourth event is angular velocity")
                .angular_velocity()
                .as_array(),
            [4.0, 5.0, 6.0]
        );
        assert_eq!(cursor.remaining(), 0);

        std::fs::remove_dir_all(path).expect("remove round-trip fixture");
    }

    #[test]
    fn configured_empty_imu_stream_is_distinct_from_legacy_and_unconfigured_streams() {
        let path = write_imu_fixture("imu-empty", &[]);
        assert_eq!(
            std::fs::metadata(path.join(format::IMU_STREAM_FILE))
                .expect("stat empty IMU stream")
                .len(),
            u64::from(IMU_STREAM_HEADER_BYTES)
        );
        let reader = DatasetReader::open(&path).expect("open configured empty stream");
        let cursor = reader.imu_cursor().expect("configured stream has a cursor");
        assert!(cursor.is_empty());
        assert_eq!(cursor.remaining(), 0);
        std::fs::remove_dir_all(path).expect("remove configured-empty fixture");

        let path = unique_dataset_path("imu-legacy");
        let (writer, handle) =
            DatasetWriter::create(&path, &meta(), &calibration()).expect("create legacy fixture");
        drop(writer);
        handle.finish().expect("finish legacy fixture");

        let reader = DatasetReader::open(&path).expect("open unconfigured dataset");
        assert!(matches!(
            reader.imu_cursor(),
            Err(DatasetError::ImuStreamNotConfigured)
        ));

        std::fs::write(
            path.join(format::META_FILE),
            serde_json::to_vec_pretty(&meta_with_imu(200)).expect("serialize legacy metadata"),
        )
        .expect("write legacy IMU metadata");
        let reader = DatasetReader::open(&path).expect("legacy IMU metadata remains readable");
        assert!(matches!(
            reader.imu_cursor(),
            Err(DatasetError::LegacyImuManifestUnindexed)
        ));
        std::fs::remove_dir_all(path).expect("remove legacy fixture");
    }

    #[test]
    fn imu_reader_rejects_truncation_reserved_fields_unknown_accuracy_and_nonfinite_values() {
        assert!(matches!(
            corrupt_imu_fixture("imu-truncated", |bytes| {
                bytes.pop();
            }),
            DatasetError::InvalidImuStream {
                source: ImuStreamError::ByteLengthMismatch { .. },
                ..
            }
        ));
        assert!(matches!(
            corrupt_imu_fixture("imu-version", |bytes| {
                bytes[8..10].copy_from_slice(&2_u16.to_le_bytes());
            }),
            DatasetError::InvalidImuStream {
                source: ImuStreamError::UnsupportedVersion {
                    expected: IMU_STREAM_VERSION,
                    actual: 2
                },
                ..
            }
        ));
        assert!(matches!(
            corrupt_imu_fixture("imu-reserved-header", |bytes| bytes[14] = 1),
            DatasetError::InvalidImuStream {
                source: ImuStreamError::NonZeroReservedHeader,
                ..
            }
        ));
        assert!(matches!(
            corrupt_imu_fixture("imu-reserved-record", |bytes| {
                bytes[usize::from(IMU_STREAM_HEADER_BYTES) + 6] = 1;
            }),
            DatasetError::InvalidImuStream {
                source: ImuStreamError::NonZeroReservedRecord { record_index: 0 },
                ..
            }
        ));
        assert!(matches!(
            corrupt_imu_fixture("imu-accuracy", |bytes| {
                bytes[usize::from(IMU_STREAM_HEADER_BYTES) + 4] = 9;
            }),
            DatasetError::InvalidImuStream {
                source: ImuStreamError::UnknownAccuracy {
                    record_index: 0,
                    sensor: "accelerometer",
                    value: 9
                },
                ..
            }
        ));
        assert!(matches!(
            corrupt_imu_fixture("imu-nonfinite", |bytes| {
                let offset = usize::from(IMU_STREAM_HEADER_BYTES) + 32;
                bytes[offset..offset + 8].copy_from_slice(&f64::NAN.to_le_bytes());
            }),
            DatasetError::InvalidImuStream {
                source: ImuStreamError::NonFiniteSample {
                    record_index: 0,
                    field: "accel_x",
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    fn imu_reader_rejects_corrupt_order_and_manifest_units() {
        let reports = [
            imu_report(7, 10, 100, 10, 40, 0.0),
            imu_report(7, 11, 110, 20, 50, 1.0),
        ];
        let path = write_imu_fixture("imu-corrupt-order", &reports);
        let stream_path = path.join(format::IMU_STREAM_FILE);
        let mut bytes = std::fs::read(&stream_path).expect("read order fixture");
        let second_record = usize::from(IMU_STREAM_HEADER_BYTES + IMU_STREAM_RECORD_BYTES);
        bytes[second_record..second_record + 4].copy_from_slice(&10_u32.to_le_bytes());
        std::fs::write(&stream_path, bytes).expect("write corrupt order fixture");
        assert!(matches!(
            DatasetReader::open(&path).expect_err("duplicate sequence must fail open"),
            DatasetError::InvalidImuStream {
                source: ImuStreamError::SequenceNotNewer {
                    record_index: 1,
                    previous: 10,
                    current: 10
                },
                ..
            }
        ));
        std::fs::remove_dir_all(path).expect("remove order fixture");

        let path = write_imu_fixture("imu-corrupt-units", &[imu_report(7, 10, 100, 10, 40, 0.0)]);
        let manifest_path = path.join(format::MANIFEST_FILE);
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).expect("read manifest fixture"))
                .expect("parse manifest fixture");
        manifest["imu_stream"]["acceleration_unit"] = serde_json::json!("g");
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("serialize corrupt manifest"),
        )
        .expect("write corrupt manifest");
        assert!(matches!(
            DatasetReader::open(&path).expect_err("ambiguous acceleration unit must fail open"),
            DatasetError::UnsupportedManifestValue {
                field: "imu_stream.acceleration_unit",
                value
            } if value == "g"
        ));
        std::fs::remove_dir_all(path).expect("remove unit fixture");
    }

    #[test]
    fn imu_cursor_reports_post_open_mutation_without_consuming_the_event() {
        let reports = [
            imu_report(7, 10, 100, 10, 40, 0.0),
            imu_report(7, 11, 110, 20, 50, 1.0),
        ];

        let path = write_imu_fixture("imu-post-open-truncate", &reports);
        let stream_path = path.join(format::IMU_STREAM_FILE);
        let original = std::fs::read(&stream_path).expect("read truncation fixture");
        let reader = DatasetReader::open(&path).expect("validate truncation fixture");
        let mut cursor = reader.imu_cursor().expect("open truncation cursor");
        let partial_first_record =
            usize::from(IMU_STREAM_HEADER_BYTES) + usize::from(IMU_STREAM_RECORD_BYTES) / 2;
        std::fs::write(&stream_path, &original[..partial_first_record])
            .expect("truncate current record after open");
        assert!(matches!(
            cursor
                .next_event()
                .expect_err("post-open truncation must fail replay"),
            DatasetError::InvalidImuStream {
                source: ImuStreamError::ByteLengthMismatch { .. },
                ..
            }
        ));
        assert_eq!(cursor.remaining(), 4);
        std::fs::write(&stream_path, &original).expect("restore truncated stream");
        let event = cursor
            .next_event()
            .expect("retry restored stream")
            .expect("first event remains available");
        assert_eq!(event.device_timestamp().as_nanos(), 10);
        assert_eq!(event.sensor(), InertialSensorKind::Accelerometer);
        assert_eq!(cursor.remaining(), 3);
        std::fs::remove_dir_all(path).expect("remove truncation fixture");

        let path = write_imu_fixture("imu-post-open-domain", &reports);
        let stream_path = path.join(format::IMU_STREAM_FILE);
        let original = std::fs::read(&stream_path).expect("read mutation fixture");
        let reader = DatasetReader::open(&path).expect("validate mutation fixture");
        let mut cursor = reader.imu_cursor().expect("open mutation cursor");
        let mut corrupted = original.clone();
        let accel_x = usize::from(IMU_STREAM_HEADER_BYTES) + 32;
        corrupted[accel_x..accel_x + 8].copy_from_slice(&f64::NAN.to_le_bytes());
        std::fs::write(&stream_path, corrupted).expect("mutate stream after open");
        assert!(matches!(
            cursor
                .next_event()
                .expect_err("post-open nonfinite sample must fail replay"),
            DatasetError::InvalidImuStream {
                source: ImuStreamError::NonFiniteSample {
                    record_index: 0,
                    field: "accel_x",
                    ..
                },
                ..
            }
        ));
        assert_eq!(cursor.remaining(), 4);
        std::fs::write(&stream_path, original).expect("restore mutated stream");
        let event = cursor
            .next_event()
            .expect("retry restored stream")
            .expect("first event remains available");
        assert_eq!(event.device_timestamp().as_nanos(), 10);
        assert_eq!(event.sensor(), InertialSensorKind::Accelerometer);
        assert_eq!(cursor.remaining(), 3);
        std::fs::remove_dir_all(path).expect("remove mutation fixture");
    }

    #[test]
    fn imu_cursor_checks_for_appended_data_once_at_logical_eof() {
        let path = write_imu_fixture(
            "imu-post-open-append",
            &[imu_report(7, 10, 100, 10, 40, 0.0)],
        );
        let stream_path = path.join(format::IMU_STREAM_FILE);
        let original = std::fs::read(&stream_path).expect("read append fixture");
        let reader = DatasetReader::open(&path).expect("validate append fixture");
        let mut cursor = reader.imu_cursor().expect("open append cursor");
        assert!(cursor.next_event().expect("read acceleration").is_some());
        assert!(cursor.next_event().expect("read gyroscope").is_some());
        assert_eq!(cursor.remaining(), 0);

        let mut appended = original.clone();
        appended.push(0);
        std::fs::write(&stream_path, appended).expect("append after cursor open");
        assert!(matches!(
            cursor
                .next_event()
                .expect_err("appended tail must fail at logical EOF"),
            DatasetError::InvalidImuStream {
                source: ImuStreamError::ByteLengthMismatch { .. },
                ..
            }
        ));
        assert_eq!(cursor.remaining(), 0);
        std::fs::write(&stream_path, original).expect("restore append fixture");
        assert!(
            cursor
                .next_event()
                .expect("retry restored logical EOF")
                .is_none()
        );
        std::fs::remove_dir_all(path).expect("remove append fixture");
    }

    #[test]
    fn imu_order_failure_does_not_publish_stream_or_completion_manifest() {
        let path = unique_dataset_path("imu-transactional-order");
        let (writer, handle) = DatasetWriter::create_with_imu_config(
            &path,
            &meta_with_imu(200),
            &calibration(),
            imu_stream_metadata(7),
            DatasetWriterConfig::default(),
        )
        .expect("create transactional IMU fixture");
        assert_eq!(
            write_outcome(writer.write_imu(imu_report(7, 10, 100, 10, 40, 0.0))),
            WriteOutcome::Enqueued
        );
        assert_eq!(
            write_outcome(writer.write_imu(imu_report(7, 10, 110, 20, 50, 1.0))),
            WriteOutcome::Enqueued
        );
        drop(writer);
        assert!(matches!(
            handle
                .finish()
                .expect_err("duplicate sequence must fail finalization"),
            DatasetError::InvalidImuStream {
                source: ImuStreamError::SequenceNotNewer {
                    record_index: 1,
                    previous: 10,
                    current: 10
                },
                ..
            }
        ));
        assert!(!path.join(format::IMU_STREAM_FILE).exists());
        assert!(!path.join(format::IMU_STREAM_TEMP_FILE).exists());
        assert!(!path.join(format::MANIFEST_FILE).exists());
        std::fs::remove_dir_all(path).expect("remove failed transaction fixture");
    }

    #[test]
    fn imu_write_contract_errors_reach_finalization_without_publication() {
        let path = unique_dataset_path("imu-session-mismatch");
        let (writer, handle) = DatasetWriter::create_with_imu_config(
            &path,
            &meta_with_imu(200),
            &calibration(),
            imu_stream_metadata(7),
            DatasetWriterConfig::default(),
        )
        .expect("create session-bound IMU writer");
        assert!(matches!(
            writer.write_imu(imu_report(8, 1, 10, 1, 2, 0.0)),
            Err(DatasetWriteError::ImuDeviceSessionMismatch {
                expected: 7,
                actual: 8
            })
        ));
        assert!(!writer.is_healthy());
        drop(writer);
        assert!(matches!(
            handle
                .finish()
                .expect_err("session mismatch must fail finalization"),
            DatasetError::WriteContract {
                source: DatasetWriteError::ImuDeviceSessionMismatch {
                    expected: 7,
                    actual: 8
                }
            }
        ));
        assert!(!path.join(format::IMU_STREAM_FILE).exists());
        assert!(!path.join(format::IMU_STREAM_TEMP_FILE).exists());
        assert!(!path.join(format::MANIFEST_FILE).exists());
        std::fs::remove_dir_all(path).expect("remove session mismatch fixture");

        let path = unique_dataset_path("imu-unconfigured-write");
        let (writer, handle) =
            DatasetWriter::create(&path, &meta(), &calibration()).expect("create plain writer");
        assert!(matches!(
            writer.write_imu(imu_report(7, 1, 10, 1, 2, 0.0)),
            Err(DatasetWriteError::ImuStreamNotConfigured)
        ));
        drop(writer);
        assert!(matches!(
            handle
                .finish()
                .expect_err("unconfigured IMU write must fail finalization"),
            DatasetError::WriteContract {
                source: DatasetWriteError::ImuStreamNotConfigured
            }
        ));
        assert!(!path.join(format::MANIFEST_FILE).exists());
        std::fs::remove_dir_all(path).expect("remove unconfigured write fixture");
    }

    #[test]
    fn imu_reports_share_the_bounded_writer_backpressure_accounting() {
        let config = DatasetWriterConfig {
            max_spool_frames: 1,
            max_spool_bytes: usize::from(IMU_STREAM_RECORD_BYTES),
            flush_batch_frames: 1,
            backpressure: Backpressure::DropNewest,
        };
        let metadata = meta_with_imu(200);
        let contract = WriterDatasetContract::parse_with_imu(
            &metadata,
            &calibration(),
            Some(imu_stream_metadata(7)),
        )
        .expect("valid IMU writer contract");
        let state = Arc::new(WriterState::new(
            config,
            PathBuf::new(),
            PathBuf::new(),
            ManifestMode::InferPairs,
            contract,
            WriterSidecars {
                meta_json: serde_json::to_vec_pretty(&metadata).expect("serialize IMU metadata"),
                calibration_json: serde_json::to_vec_pretty(&calibration())
                    .expect("serialize calibration"),
            },
        ));
        {
            let mut spool = state.spool.lock().expect("lock IMU spool");
            spool
                .queue
                .push_back(SpoolItem::Imu(imu_report(7, 1, 10, 1, 2, 0.0)));
            spool.frames = 1;
            spool.bytes = usize::from(IMU_STREAM_RECORD_BYTES);
        }
        state.enqueued.store(1, Ordering::Relaxed);
        state
            .bytes_enqueued
            .store(u64::from(IMU_STREAM_RECORD_BYTES), Ordering::Relaxed);
        state.spool_frames.store(1, Ordering::Relaxed);
        state
            .spool_bytes
            .store(u64::from(IMU_STREAM_RECORD_BYTES), Ordering::Relaxed);
        let writer = DatasetWriter {
            config,
            state: Arc::clone(&state),
        };

        assert_eq!(
            write_outcome(writer.write_imu(imu_report(7, 2, 20, 3, 4, 1.0))),
            WriteOutcome::Dropped
        );
        let stats = writer.stats();
        assert_eq!(stats.frames_enqueued, 1);
        assert_eq!(stats.bytes_enqueued, u64::from(IMU_STREAM_RECORD_BYTES));
        assert_eq!(stats.frames_dropped, 1);
        assert_eq!(stats.bytes_dropped, u64::from(IMU_STREAM_RECORD_BYTES));
        assert_eq!(stats.spool_frames, 1);
        assert_eq!(stats.spool_bytes, u64::from(IMU_STREAM_RECORD_BYTES));
        let spool = state.spool.lock().expect("lock IMU spool");
        assert_eq!(spool.queue.len(), 1);
        assert_eq!(spool.frames, 1);
        assert_eq!(spool.bytes, usize::from(IMU_STREAM_RECORD_BYTES));
    }
}
