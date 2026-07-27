//! oak-sys: FFI bindings to OAK-D camera via DepthAI
//!
//! Weak configuration and native payloads are parsed once before valid domain
//! frames, intrinsics, and measurements are exposed.

use thiserror::Error;

#[cfg(feature = "opencv-face-detector")]
mod opencv_face_ffi;
#[cfg(all(feature = "opencv-face-detector", any(test, not(oak_sys_check_only))))]
use opencv_face_ffi::ffi as face_ffi;

// ============================================================================
// FFI MODULE - Raw C++ interface (not re-exported publicly)
// ============================================================================

#[cxx::bridge(namespace = "kiko::oak")]
#[cfg_attr(oak_sys_check_only, allow(dead_code))]
mod ffi {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum DeviceState {
        Available,
        InUse,
        Bootloader,
        Unknown,
    }

    /// Raw DepthAI USB-speed readback. `Unknown` and `Unrecognized` are
    /// boundary values and are never admitted into the public domain.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum UsbSpeed {
        Unknown,
        Low,
        Full,
        High,
        Super,
        SuperPlus,
        Unrecognized,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum StreamId {
        Rgb,
        MonoLeft,
        MonoRight,
        Depth,
        Imu,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum FrameStatus {
        Ok,
        Timeout,
        QueueEmpty,
        Disconnected,
        Corrupt,
        StreamNotEnabled,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ImuStatus {
        Ok,
        Empty,
        Disconnected,
        Corrupt,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ImuAccuracy {
        Unreliable,
        Low,
        Medium,
        High,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum DepthAlignment {
        RectifiedLeft,
        RectifiedRight,
        Rgb,
    }

    /// Raw camera socket returned by the EEPROM stereo-role queries.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum CameraSocket {
        CameraA,
        CameraB,
        CameraC,
        Unrecognized,
    }

    #[derive(Debug, Clone)]
    pub struct DeviceConfig {
        pub maximum_usb_speed: UsbSpeed,

        pub rgb_enabled: bool,
        pub rgb_width: u32,
        pub rgb_height: u32,
        pub rgb_fps: u32,

        pub mono_enabled: bool,
        pub mono_width: u32,
        pub mono_height: u32,
        pub mono_fps: u32,
        pub mono_rectified: bool,

        pub depth_enabled: bool,
        pub depth_width: u32,
        pub depth_height: u32,
        pub depth_fps: u32,
        pub depth_alignment: DepthAlignment,

        pub imu_enabled: bool,
        pub imu_rate_hz: u32,

        pub queue_size: u32,
        pub queue_blocking: bool,
    }

    #[derive(Debug, Clone, Copy, Default)]
    pub struct Timestamp {
        pub device_ns: i64,
    }

    /// Projection intrinsics for the exact pixel grid delivered with a frame.
    #[derive(Debug, Clone, Copy)]
    pub struct Intrinsics {
        pub m00: f32,
        pub m01: f32,
        pub m02: f32,
        pub m10: f32,
        pub m11: f32,
        pub m12: f32,
        pub m20: f32,
        pub m21: f32,
        pub m22: f32,
        pub width: u32,
        pub height: u32,
    }

    #[derive(Debug, Clone)]
    pub struct ImageFrame {
        pub stream: StreamId,
        /// Native DepthAI capture sequence. Negative values are rejected in Rust.
        pub device_capture_sequence: i64,
        /// Synthetic host bridge dequeue sequence, independent of capture identity.
        pub host_delivery_sequence: u64,
        /// Device-clock timestamp requested at exposure midpoint.
        pub timestamp: Timestamp,
        /// DepthAI-reported exposure duration in microseconds. Zero means unavailable.
        pub exposure_time_us: i64,
        pub width: u32,
        pub height: u32,
        pub stride_bytes: u32,
        pub data: Vec<u8>,
        pub intrinsics: Intrinsics,
    }

    #[derive(Debug, Clone)]
    pub struct DepthFrame {
        /// Native DepthAI capture sequence. Negative values are rejected in Rust.
        pub device_capture_sequence: i64,
        /// Synthetic host bridge dequeue sequence, independent of capture identity.
        pub host_delivery_sequence: u64,
        /// Device-clock timestamp requested at exposure midpoint.
        pub timestamp: Timestamp,
        /// DepthAI-reported exposure duration in microseconds. Zero means unavailable.
        pub exposure_time_us: i64,
        pub width: u32,
        pub height: u32,
        /// Unsigned millimetres. Zero is the sole invalid-depth sentinel.
        pub data: Vec<u16>,
        pub intrinsics: Intrinsics,
    }

    #[derive(Debug, Clone, Copy)]
    pub struct ImuSample {
        pub accel_timestamp: Timestamp,
        pub gyro_timestamp: Timestamp,
        /// Strictly increasing host bridge dequeue sequence, not a device sequence.
        pub sequence: u32,
        pub accel_x: f32,
        pub accel_y: f32,
        pub accel_z: f32,
        pub accel_accuracy: ImuAccuracy,
        pub gyro_x: f32,
        pub gyro_y: f32,
        pub gyro_z: f32,
        pub gyro_accuracy: ImuAccuracy,
    }

    #[derive(Debug, Clone)]
    pub struct ImuBatch {
        pub samples: Vec<ImuSample>,
    }

    #[derive(Debug)]
    pub struct ImageFrameResult {
        pub status: FrameStatus,
        pub frame: ImageFrame,
    }

    #[derive(Debug)]
    pub struct DepthFrameResult {
        pub status: FrameStatus,
        pub frame: DepthFrame,
    }

    #[derive(Debug)]
    pub struct ImuBatchResult {
        pub status: ImuStatus,
        pub batch: ImuBatch,
    }

    #[derive(Debug, Clone)]
    pub struct DeviceInfo {
        pub device_id: String,
        pub name: String,
        pub state: DeviceState,
    }

    #[derive(Debug, Clone)]
    pub struct DepthAiBuildMetadata {
        pub sdk_version: String,
        pub sdk_commit: String,
        pub embedded_device_artifact_version: String,
        pub embedded_bootloader_artifact_version: String,
    }

    #[derive(Debug, Clone)]
    pub struct ConnectedDeviceIdentity {
        pub mxid: String,
        pub discovery_transport_name: String,
        pub eeprom_device_name: String,
        pub product_name: String,
    }

    /// Weak carrier for exact coefficients read from the already-opened
    /// device's EEPROM calibration through the pinned DepthAI API.
    #[derive(Debug, Clone)]
    pub struct EepromCalibrationEvidence {
        pub stereo_left_camera_socket: CameraSocket,
        pub stereo_right_camera_socket: CameraSocket,
        /// Row-major output of
        /// `getImuToCameraExtrinsics(CAM_B, false, METER)`.
        pub imu_to_camera_b_m: Vec<f32>,
        /// Exact row-major output of getStereoLeftRectificationRotation().
        pub stereo_left_rectification_rotation_raw: Vec<f32>,
    }

    #[cfg(not(oak_sys_check_only))]
    unsafe extern "C++" {
        include!("oak_device.hpp");

        type OakDevice;

        fn depthai_build_metadata() -> Result<DepthAiBuildMetadata>;
        fn list_devices() -> Result<Vec<DeviceInfo>>;
        fn create_device(selector: &str, config: &DeviceConfig) -> Result<UniquePtr<OakDevice>>;
        fn is_connected(self: &OakDevice) -> bool;
        fn get_usb_speed(self: &OakDevice) -> Result<UsbSpeed>;
        fn get_connected_device_identity(self: &OakDevice) -> Result<ConnectedDeviceIdentity>;
        fn try_get_rgb(self: Pin<&mut OakDevice>, timeout_ms: u32) -> Result<ImageFrameResult>;
        fn try_get_mono_left(
            self: Pin<&mut OakDevice>,
            timeout_ms: u32,
        ) -> Result<ImageFrameResult>;
        fn try_get_mono_right(
            self: Pin<&mut OakDevice>,
            timeout_ms: u32,
        ) -> Result<ImageFrameResult>;
        fn try_get_depth(self: Pin<&mut OakDevice>, timeout_ms: u32) -> Result<DepthFrameResult>;
        fn get_imu_batch(self: Pin<&mut OakDevice>) -> Result<ImuBatchResult>;
        fn get_stereo_baseline_m(self: &OakDevice) -> Result<f32>;
        fn get_eeprom_calibration_evidence(self: &OakDevice) -> Result<EepromCalibrationEvidence>;
        fn close(self: Pin<&mut OakDevice>) -> Result<()>;
    }
}

// ============================================================================
// PUBLIC RE-EXPORTS - Only types that are always valid
// ============================================================================

pub use ffi::{DeviceState, ImuAccuracy, StreamId};

// ============================================================================
// NEWTYPES - Lift validation into types
// ============================================================================

/// Nanosecond timestamp from device clock. Always valid once constructed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Timestamp(i64);

impl Timestamp {
    pub fn try_from_nanos(ns: i64) -> Result<Self, TimestampError> {
        if ns < 0 {
            return Err(TimestampError::Negative { value_ns: ns });
        }
        Ok(Self(ns))
    }

    pub fn as_nanos(self) -> i64 {
        self.0
    }

    pub fn as_secs_f64(self) -> f64 {
        self.0 as f64 / 1_000_000_000.0
    }

    pub fn as_millis(self) -> i64 {
        self.0 / 1_000_000
    }
}

impl TryFrom<ffi::Timestamp> for Timestamp {
    type Error = TimestampError;

    fn try_from(ts: ffi::Timestamp) -> Result<Self, Self::Error> {
        Self::try_from_nanos(ts.device_ns)
    }
}

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimestampError {
    #[error("device timestamp must be nonnegative, got {value_ns}ns")]
    Negative { value_ns: i64 },
}

/// Native DepthAI image/depth capture identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeviceFrameSequence(i64);

impl DeviceFrameSequence {
    pub fn try_from_i64(value: i64) -> Result<Self, FrameSequenceError> {
        if value < 0 {
            return Err(FrameSequenceError::Negative { value });
        }
        Ok(Self(value))
    }

    pub fn as_i64(self) -> i64 {
        self.0
    }

    pub fn as_u64(self) -> u64 {
        self.0 as u64
    }
}

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameSequenceError {
    #[error("native DepthAI frame sequence must be nonnegative, got {value}")]
    Negative { value: i64 },
}

/// Synthetic host-bridge delivery identity, distinct from device capture identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FrameDeliverySequence(u64);

impl FrameDeliverySequence {
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub fn as_u64(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraTimestampReference {
    ExposureMidpoint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExposureDuration {
    micros: u64,
}

impl ExposureDuration {
    pub fn as_micros(self) -> u64 {
        self.micros
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExposureTime {
    Unavailable,
    Known(ExposureDuration),
}

#[cfg(any(test, not(oak_sys_check_only)))]
fn parse_exposure_time(value_us: i64) -> Result<ExposureTime, ExposureTimeError> {
    match value_us {
        value if value < 0 => Err(ExposureTimeError::Negative { value_us: value }),
        0 => Ok(ExposureTime::Unavailable),
        value => Ok(ExposureTime::Known(ExposureDuration {
            micros: value as u64,
        })),
    }
}

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExposureTimeError {
    #[error("DepthAI exposure duration must be nonnegative, got {value_us}us")]
    Negative { value_us: i64 },
}

/// 3D vector for accelerometer/gyroscope readings
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn magnitude(self) -> f32 {
        self.x.hypot(self.y).hypot(self.z)
    }

    pub fn as_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
}

// ============================================================================
// CONFIGURATION - Parsed once at the Device::connect boundary
// ============================================================================

/// A known USB link speed in increasing transport-rate order.
///
/// The names deliberately match DepthAI's v3 `dai::UsbSpeed` values. The
/// native `UNKNOWN` readback is excluded: admitting it would turn missing
/// evidence into a transport claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UsbTransportSpeed {
    Low,
    Full,
    High,
    Super,
    SuperPlus,
}

impl UsbTransportSpeed {
    pub const fn as_depthai_name(self) -> &'static str {
        match self {
            Self::Low => "LOW",
            Self::Full => "FULL",
            Self::High => "HIGH",
            Self::Super => "SUPER",
            Self::SuperPlus => "SUPER_PLUS",
        }
    }

    /// Parse one exact DepthAI v3 USB-speed name.
    pub fn parse(value: &str) -> Result<Self, UsbTransportSpeedParseError> {
        match value {
            "LOW" => Ok(Self::Low),
            "FULL" => Ok(Self::Full),
            "HIGH" => Ok(Self::High),
            "SUPER" => Ok(Self::Super),
            "SUPER_PLUS" => Ok(Self::SuperPlus),
            _ => Err(UsbTransportSpeedParseError),
        }
    }

    #[cfg(not(oak_sys_check_only))]
    fn to_ffi(self) -> ffi::UsbSpeed {
        match self {
            Self::Low => ffi::UsbSpeed::Low,
            Self::Full => ffi::UsbSpeed::Full,
            Self::High => ffi::UsbSpeed::High,
            Self::Super => ffi::UsbSpeed::Super,
            Self::SuperPlus => ffi::UsbSpeed::SuperPlus,
        }
    }
}

impl std::fmt::Display for UsbTransportSpeed {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_depthai_name())
    }
}

#[derive(Error, Debug, Clone, PartialEq, Eq)]
#[error("OAK USB transport speed must be exactly one of LOW, FULL, HIGH, SUPER, SUPER_PLUS")]
pub struct UsbTransportSpeedParseError;

/// Requested maximum and admitted minimum USB link speed for one connection.
///
/// Construction makes `minimum <= maximum` unrepresentable. A HIGH/HIGH
/// policy is available for deliberately bandwidth-limited diagnostics; camera
/// production constructors use SUPER/SUPER.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UsbTransportPolicy {
    maximum: UsbTransportSpeed,
    minimum: UsbTransportSpeed,
}

impl UsbTransportPolicy {
    pub fn try_new(
        maximum: UsbTransportSpeed,
        minimum: UsbTransportSpeed,
    ) -> Result<Self, UsbTransportPolicyError> {
        if minimum > maximum {
            return Err(UsbTransportPolicyError::MinimumAboveMaximum { minimum, maximum });
        }
        Ok(Self { maximum, minimum })
    }

    /// Production camera policy: request and require USB 3 SuperSpeed.
    pub const fn super_speed_required() -> Self {
        Self {
            maximum: UsbTransportSpeed::Super,
            minimum: UsbTransportSpeed::Super,
        }
    }

    /// Explicit USB 2 HIGH diagnostic policy; never selected implicitly.
    pub const fn high_speed_diagnostic() -> Self {
        Self {
            maximum: UsbTransportSpeed::High,
            minimum: UsbTransportSpeed::High,
        }
    }

    pub const fn maximum(self) -> UsbTransportSpeed {
        self.maximum
    }

    pub const fn minimum(self) -> UsbTransportSpeed {
        self.minimum
    }
}

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsbTransportPolicyError {
    #[error("OAK USB minimum transport speed {minimum} exceeds requested maximum {maximum}")]
    MinimumAboveMaximum {
        minimum: UsbTransportSpeed,
        maximum: UsbTransportSpeed,
    },
}

/// Transport evidence captured from the already-selected device after its
/// pipeline started and admitted against the parsed policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UsbTransportAdmissionEvidence {
    requested_maximum: UsbTransportSpeed,
    required_minimum: UsbTransportSpeed,
    observed: UsbTransportSpeed,
}

impl UsbTransportAdmissionEvidence {
    pub const fn requested_maximum(self) -> UsbTransportSpeed {
        self.requested_maximum
    }

    pub const fn required_minimum(self) -> UsbTransportSpeed {
        self.required_minimum
    }

    pub const fn observed(self) -> UsbTransportSpeed {
        self.observed
    }
}

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsbTransportAdmissionError {
    #[error("DepthAI returned UNKNOWN for the connected OAK USB speed")]
    UnknownReadback,

    #[error("DepthAI returned an unrecognized OAK USB-speed value")]
    UnrecognizedReadback,

    #[error(
        "connected OAK USB speed {observed} is below required minimum {required_minimum} (requested maximum {requested_maximum})"
    )]
    BelowMinimum {
        observed: UsbTransportSpeed,
        required_minimum: UsbTransportSpeed,
        requested_maximum: UsbTransportSpeed,
    },

    #[error("connected OAK USB speed {observed} exceeds requested maximum {requested_maximum}")]
    AboveMaximum {
        observed: UsbTransportSpeed,
        requested_maximum: UsbTransportSpeed,
    },
}

#[cfg(any(test, not(oak_sys_check_only)))]
fn admit_usb_transport(
    policy: UsbTransportPolicy,
    raw: ffi::UsbSpeed,
) -> Result<UsbTransportAdmissionEvidence, UsbTransportAdmissionError> {
    let observed = match raw {
        ffi::UsbSpeed::Unknown => return Err(UsbTransportAdmissionError::UnknownReadback),
        ffi::UsbSpeed::Low => UsbTransportSpeed::Low,
        ffi::UsbSpeed::Full => UsbTransportSpeed::Full,
        ffi::UsbSpeed::High => UsbTransportSpeed::High,
        ffi::UsbSpeed::Super => UsbTransportSpeed::Super,
        ffi::UsbSpeed::SuperPlus => UsbTransportSpeed::SuperPlus,
        ffi::UsbSpeed::Unrecognized => {
            return Err(UsbTransportAdmissionError::UnrecognizedReadback);
        }
        _ => return Err(UsbTransportAdmissionError::UnrecognizedReadback),
    };
    if observed < policy.minimum() {
        return Err(UsbTransportAdmissionError::BelowMinimum {
            observed,
            required_minimum: policy.minimum(),
            requested_maximum: policy.maximum(),
        });
    }
    if observed > policy.maximum() {
        return Err(UsbTransportAdmissionError::AboveMaximum {
            observed,
            requested_maximum: policy.maximum(),
        });
    }
    Ok(UsbTransportAdmissionEvidence {
        requested_maximum: policy.maximum(),
        required_minimum: policy.minimum(),
        observed,
    })
}

/// RGB camera stream configuration
#[derive(Debug, Clone, Copy)]
pub struct RgbConfig {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

/// Mono camera pair configuration (left + right share settings)
#[derive(Debug, Clone, Copy)]
pub struct MonoConfig {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub rectified: bool,
}

/// Stereo depth stream configuration
#[derive(Debug, Clone, Copy)]
pub struct DepthConfig {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub alignment: DepthAlignment,
}

/// Optical frame whose pixel grid and intrinsics define the depth image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthAlignment {
    RectifiedLeft,
    RectifiedRight,
    Rgb,
}

impl From<DepthAlignment> for ffi::DepthAlignment {
    fn from(value: DepthAlignment) -> Self {
        match value {
            DepthAlignment::RectifiedLeft => Self::RectifiedLeft,
            DepthAlignment::RectifiedRight => Self::RectifiedRight,
            DepthAlignment::Rgb => Self::Rgb,
        }
    }
}

/// IMU sensor configuration
#[derive(Debug, Clone, Copy)]
pub struct ImuConfig {
    pub rate_hz: u32,
}

/// Queue behavior configuration
#[derive(Debug, Clone, Copy)]
pub struct QueueConfig {
    pub size: u32,
    pub blocking: bool,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            size: 4,
            blocking: false,
        }
    }
}

/// Weak device configuration parsed once by [`Device::connect`].
/// Disabled streams have no associated configuration.
#[derive(Debug, Clone)]
pub struct DeviceConfig {
    pub usb_transport: UsbTransportPolicy,
    pub rgb: Option<RgbConfig>,
    pub mono: Option<MonoConfig>,
    pub depth: Option<DepthConfig>,
    pub imu: Option<ImuConfig>,
    pub queue: QueueConfig,
}

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum DeviceConfigError {
    #[error("OAK device configuration must enable at least one stream")]
    NoStreamsEnabled,
    #[error("OAK {stream} {field} must be nonzero")]
    ZeroStreamField {
        stream: &'static str,
        field: &'static str,
    },
    #[error("OAK output queue size must be nonzero")]
    ZeroQueueSize,
    #[error(
        "OAK mono and depth share one stereo sensor timing/shape contract: mono={mono_width}x{mono_height}@{mono_fps}Hz, depth={depth_width}x{depth_height}@{depth_fps}Hz"
    )]
    ConflictingStereoContracts {
        mono_width: u32,
        mono_height: u32,
        mono_fps: u32,
        depth_width: u32,
        depth_height: u32,
        depth_fps: u32,
    },
    #[error("RGB-aligned depth requires an enabled RGB stream")]
    RgbAlignmentWithoutRgb,
}

impl DeviceConfig {
    /// All streams enabled at 640x480@30fps
    pub fn all_streams() -> Self {
        Self {
            usb_transport: UsbTransportPolicy::super_speed_required(),
            rgb: Some(RgbConfig {
                width: 640,
                height: 480,
                fps: 30,
            }),
            mono: Some(MonoConfig {
                width: 640,
                height: 480,
                fps: 30,
                rectified: true,
            }),
            depth: Some(DepthConfig {
                width: 640,
                height: 480,
                fps: 30,
                alignment: DepthAlignment::Rgb,
            }),
            imu: Some(ImuConfig { rate_hz: 400 }),
            queue: QueueConfig::default(),
        }
    }

    /// Only RGB stream
    pub fn rgb_only(width: u32, height: u32, fps: u32) -> Self {
        Self {
            usb_transport: UsbTransportPolicy::super_speed_required(),
            rgb: Some(RgbConfig { width, height, fps }),
            mono: None,
            depth: None,
            imu: None,
            queue: QueueConfig::default(),
        }
    }

    pub fn validate(&self) -> Result<(), DeviceConfigError> {
        if self.rgb.is_none() && self.mono.is_none() && self.depth.is_none() && self.imu.is_none() {
            return Err(DeviceConfigError::NoStreamsEnabled);
        }
        if self.queue.size == 0 {
            return Err(DeviceConfigError::ZeroQueueSize);
        }
        let require_nonzero = |stream: &'static str, width, height, fps| {
            for (field, value) in [("width", width), ("height", height), ("fps", fps)] {
                if value == 0 {
                    return Err(DeviceConfigError::ZeroStreamField { stream, field });
                }
            }
            Ok(())
        };
        if let Some(rgb) = self.rgb {
            require_nonzero("RGB", rgb.width, rgb.height, rgb.fps)?;
        }
        if let Some(mono) = self.mono {
            require_nonzero("mono", mono.width, mono.height, mono.fps)?;
        }
        if let Some(depth) = self.depth {
            require_nonzero("depth", depth.width, depth.height, depth.fps)?;
            if depth.alignment == DepthAlignment::Rgb && self.rgb.is_none() {
                return Err(DeviceConfigError::RgbAlignmentWithoutRgb);
            }
        }
        if let Some(imu) = self.imu {
            if imu.rate_hz == 0 {
                return Err(DeviceConfigError::ZeroStreamField {
                    stream: "IMU",
                    field: "rate_hz",
                });
            }
        }
        if let (Some(mono), Some(depth)) = (self.mono, self.depth) {
            if (mono.width, mono.height, mono.fps) != (depth.width, depth.height, depth.fps) {
                return Err(DeviceConfigError::ConflictingStereoContracts {
                    mono_width: mono.width,
                    mono_height: mono.height,
                    mono_fps: mono.fps,
                    depth_width: depth.width,
                    depth_height: depth.height,
                    depth_fps: depth.fps,
                });
            }
        }
        Ok(())
    }

    #[cfg(not(oak_sys_check_only))]
    fn to_ffi(&self) -> ffi::DeviceConfig {
        ffi::DeviceConfig {
            maximum_usb_speed: self.usb_transport.maximum().to_ffi(),
            rgb_enabled: self.rgb.is_some(),
            rgb_width: self.rgb.map(|c| c.width).unwrap_or(0),
            rgb_height: self.rgb.map(|c| c.height).unwrap_or(0),
            rgb_fps: self.rgb.map(|c| c.fps).unwrap_or(0),

            mono_enabled: self.mono.is_some(),
            mono_width: self.mono.map(|c| c.width).unwrap_or(0),
            mono_height: self.mono.map(|c| c.height).unwrap_or(0),
            mono_fps: self.mono.map(|c| c.fps).unwrap_or(0),
            mono_rectified: self.mono.map(|c| c.rectified).unwrap_or(false),

            depth_enabled: self.depth.is_some(),
            depth_width: self.depth.map(|c| c.width).unwrap_or(0),
            depth_height: self.depth.map(|c| c.height).unwrap_or(0),
            depth_fps: self.depth.map(|c| c.fps).unwrap_or(0),
            depth_alignment: self
                .depth
                .map(|config| config.alignment.into())
                .unwrap_or(ffi::DepthAlignment::RectifiedLeft),

            imu_enabled: self.imu.is_some(),
            imu_rate_hz: self.imu.map(|c| c.rate_hz).unwrap_or(0),

            queue_size: self.queue.size,
            queue_blocking: self.queue.blocking,
        }
    }
}

// ============================================================================
// ERROR TYPES - Domain-focused, not implementation-focused
// ============================================================================

/// Errors when acquiring image frames (RGB, mono)
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ImageError {
    #[error("frame acquisition timed out after {timeout_ms}ms")]
    Timeout { timeout_ms: u32 },

    #[error("frame queue is empty (non-blocking mode)")]
    QueueEmpty,

    #[error("device disconnected during frame acquisition")]
    Disconnected,

    #[error("received corrupt frame data")]
    Corrupt,

    #[error("attempted to get {stream:?} frame but stream is not enabled in config")]
    StreamNotEnabled { stream: StreamId },

    #[error("invalid frame projection intrinsics: {0}")]
    InvalidIntrinsics(#[from] IntrinsicsError),

    #[error("invalid image device timestamp: {0}")]
    InvalidTimestamp(#[from] TimestampError),

    #[error("invalid native image capture sequence: {0}")]
    InvalidDeviceSequence(#[from] FrameSequenceError),

    #[error("invalid image exposure metadata: {0}")]
    InvalidExposureTime(#[from] ExposureTimeError),

    #[error("DepthAI frame acquisition failed: {message}")]
    Native { message: String },
}

/// Errors when acquiring depth frames
#[derive(Error, Debug, Clone, PartialEq)]
pub enum DepthError {
    #[error("depth frame acquisition timed out after {timeout_ms}ms")]
    Timeout { timeout_ms: u32 },

    #[error("depth queue is empty (non-blocking mode)")]
    QueueEmpty,

    #[error("device disconnected during depth acquisition")]
    Disconnected,

    #[error("received corrupt depth data")]
    Corrupt,

    #[error("depth stream is not enabled in config")]
    StreamNotEnabled,

    #[error("invalid depth projection intrinsics: {0}")]
    InvalidIntrinsics(#[from] IntrinsicsError),

    #[error("invalid depth device timestamp: {0}")]
    InvalidTimestamp(#[from] TimestampError),

    #[error("invalid native depth capture sequence: {0}")]
    InvalidDeviceSequence(#[from] FrameSequenceError),

    #[error("invalid depth exposure metadata: {0}")]
    InvalidExposureTime(#[from] ExposureTimeError),

    #[error("DepthAI depth acquisition failed: {message}")]
    Native { message: String },
}

/// Errors when acquiring IMU data
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ImuError {
    #[error("no IMU samples available")]
    Empty,

    #[error("device disconnected")]
    Disconnected,

    #[error("received corrupt IMU data")]
    Corrupt,

    #[error("IMU stream is not enabled in config")]
    StreamNotEnabled,

    #[error("IMU sample {sample_index} contains a non-finite {field} value")]
    NonFiniteSample {
        sample_index: usize,
        field: &'static str,
    },

    #[error("IMU sample {sample_index} has an invalid {sensor} device timestamp: {source}")]
    InvalidTimestamp {
        sample_index: usize,
        sensor: &'static str,
        #[source]
        source: TimestampError,
    },

    #[error("DepthAI IMU acquisition failed: {message}")]
    Native { message: String },
}

/// Errors when enumerating devices.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
#[error("DepthAI device enumeration failed: {message}")]
pub struct DeviceDiscoveryError {
    message: String,
}

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum DepthAiBuildMetadataError {
    #[error("OAK native runtime is disabled by OAK_SYS_CHECK_ONLY=1")]
    CompileOnlyNativeDisabled,

    #[error("DepthAI SDK build-metadata query failed: {message}")]
    Native { message: String },

    #[error("required DepthAI SDK build-metadata field '{field}' is empty")]
    EmptyRequiredField { field: &'static str },
}

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ConnectedDeviceIdentityError {
    #[error("OAK native runtime is disabled by OAK_SYS_CHECK_ONLY=1")]
    CompileOnlyNativeDisabled,

    #[error("DepthAI connected-device identity query failed: {message}")]
    Native { message: String },

    #[error("the connected DepthAI device returned an empty MXID")]
    EmptyMxid,

    #[error(
        "DepthAI opened MXID '{actual_mxid}' instead of requested exact MXID '{requested_mxid}'"
    )]
    SelectorMismatch {
        requested_mxid: String,
        actual_mxid: String,
    },
}

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsbTransportEvidenceError {
    #[error("OAK native runtime is disabled by OAK_SYS_CHECK_ONLY=1")]
    CompileOnlyNativeDisabled,
}

/// Errors when connecting to a device
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ConnectionError {
    #[error("invalid OAK device configuration: {0}")]
    InvalidConfig(#[from] DeviceConfigError),

    #[error("OAK device selector must be a nonempty exact MXID")]
    EmptySelector,

    #[error("OAK native runtime is disabled by OAK_SYS_CHECK_ONLY=1")]
    CompileOnlyNativeDisabled,

    #[error("DepthAI failed to connect to OAK selector '{selector}': {message}")]
    Native { selector: String, message: String },

    #[error("DepthAI returned a null OAK device for selector '{selector}'")]
    NullDevice { selector: String },

    #[error("OAK pipeline for selector '{selector}' did not reach the connected state")]
    NotConnected { selector: String },

    #[error("DepthAI USB-speed readback failed for OAK selector '{selector}': {message}")]
    UsbTransportReadbackNative { selector: String, message: String },

    #[error("connected OAK USB transport was not admitted: {0}")]
    UsbTransportAdmission(#[from] UsbTransportAdmissionError),

    #[error("connected OAK identity is invalid: {0}")]
    InvalidConnectedIdentity(#[from] ConnectedDeviceIdentityError),
}

/// Errors when reading fixed stereo calibration.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum CalibrationError {
    #[error("DepthAI calibration query failed: {message}")]
    Native { message: String },

    #[error("DepthAI returned an invalid stereo baseline in metres: {value}")]
    InvalidBaseline { value: f32 },

    #[error("DepthAI returned {actual} coefficients for {matrix}; expected exactly {expected}")]
    InvalidMatrixLength {
        matrix: EepromCalibrationMatrix,
        expected: usize,
        actual: usize,
    },

    #[error("DepthAI returned a non-finite {matrix} coefficient at [{row}][{column}]: {value}")]
    NonFiniteMatrixCoefficient {
        matrix: EepromCalibrationMatrix,
        row: usize,
        column: usize,
        value: f32,
    },

    #[error("DepthAI EEPROM stereo-{side} camera socket is {actual}, expected {expected}")]
    UnexpectedStereoCameraSocket {
        side: EepromStereoSide,
        expected: OakCameraSocket,
        actual: OakCameraSocket,
    },
}

/// Exact EEPROM matrix whose weak native coefficients failed admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EepromCalibrationMatrix {
    ImuToCameraBMeters,
    StereoLeftRectificationRotationRaw,
}

impl std::fmt::Display for EepromCalibrationMatrix {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::ImuToCameraBMeters => "IMU-to-CAM_B matrix in metres",
            Self::StereoLeftRectificationRotationRaw => {
                "raw stereo-left rectification rotation matrix"
            }
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EepromStereoSide {
    Left,
    Right,
}

impl std::fmt::Display for EepromStereoSide {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Left => "left",
            Self::Right => "right",
        })
    }
}

/// Named OAK camera socket retained from EEPROM stereo-role evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OakCameraSocket {
    CameraA,
    CameraB,
    CameraC,
    Unrecognized,
}

impl std::fmt::Display for OakCameraSocket {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::CameraA => "CAM_A",
            Self::CameraB => "CAM_B",
            Self::CameraC => "CAM_C",
            Self::Unrecognized => "unrecognized",
        })
    }
}

impl From<ffi::CameraSocket> for OakCameraSocket {
    fn from(value: ffi::CameraSocket) -> Self {
        match value {
            ffi::CameraSocket::CameraA => Self::CameraA,
            ffi::CameraSocket::CameraB => Self::CameraB,
            ffi::CameraSocket::CameraC => Self::CameraC,
            ffi::CameraSocket::Unrecognized => Self::Unrecognized,
            _ => Self::Unrecognized,
        }
    }
}

/// Errors reported by an explicit device close.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
#[error("DepthAI device close failed: {message}")]
pub struct CloseError {
    message: String,
}

/// Errors while parsing a frame's projection matrix into pinhole intrinsics.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum IntrinsicsError {
    #[error("intrinsic pixel grid must have nonzero dimensions, got {width}x{height}")]
    ZeroDimensions { width: u32, height: u32 },

    #[error("intrinsic coefficient [{row}][{column}] is not finite: {value}")]
    NonFiniteCoefficient {
        row: usize,
        column: usize,
        value: f32,
    },

    #[error("intrinsic focal lengths must be positive, got fx={fx}, fy={fy}")]
    NonPositiveFocalLength { fx: f32, fy: f32 },

    #[error("projection matrix is not a supported canonical pinhole matrix: {matrix:?}")]
    UnsupportedProjectionMatrix { matrix: [[f32; 3]; 3] },

    #[error(
        "intrinsic pixel grid {intrinsic_width}x{intrinsic_height} does not match frame grid {frame_width}x{frame_height}"
    )]
    DimensionMismatch {
        intrinsic_width: u32,
        intrinsic_height: u32,
        frame_width: u32,
        frame_height: u32,
    },
}

// ============================================================================
// FRAME TYPES - Constructed only from valid FFI results
// ============================================================================

/// A valid RGB or mono image frame
#[derive(Debug, Clone, PartialEq)]
pub struct ImageFrame {
    pub stream: StreamId,
    /// Native sequence returned by `ImgFrame::getSequenceNum()`.
    pub device_capture_sequence: DeviceFrameSequence,
    /// Strictly increasing bridge dequeue sequence, independent of capture identity.
    pub host_delivery_sequence: FrameDeliverySequence,
    /// Device clock at exposure midpoint.
    pub timestamp: Timestamp,
    pub timestamp_reference: CameraTimestampReference,
    pub exposure_time: ExposureTime,
    pub width: u32,
    pub height: u32,
    pub stride_bytes: u32,
    data: Vec<u8>, // Private - use pixels() accessor
    intrinsics: Intrinsics,
}

impl ImageFrame {
    /// Tightly packed pixels. RGB is interleaved BGR888; mono is grayscale.
    /// Native row padding is removed at the bridge boundary.
    pub fn pixels(&self) -> &[u8] {
        &self.data
    }

    /// Projection intrinsics for this exact delivered pixel grid.
    pub fn intrinsics(&self) -> Intrinsics {
        self.intrinsics
    }

    /// Consume frame and take ownership of pixel data
    pub fn into_pixels(self) -> Vec<u8> {
        self.data
    }
}

// ============================================================================
// OPTIONAL HOST-SIDE OPENCV HAAR FACE DETECTOR
// ============================================================================

/// Hard upper bound for detections retained in one public result batch.
#[cfg(feature = "opencv-face-detector")]
pub const MAXIMUM_RETAINED_HAAR_FACE_DETECTIONS: u32 = 256;

/// Absolute upper bound for native candidates admitted across the CXX result
/// boundary before deterministic retained-output capping.
///
/// This bounds bridge materialization, not OpenCV's internal cascade-search
/// work or temporary allocations.
#[cfg(feature = "opencv-face-detector")]
pub const MAXIMUM_NATIVE_HAAR_FACE_DETECTIONS: u32 = 4096;

/// Per-axis bound for frames admitted to the host Haar detector.
#[cfg(feature = "opencv-face-detector")]
pub const MAXIMUM_HAAR_FACE_FRAME_DIMENSION: u32 = 4096;

/// Total-pixel bound for frames admitted to the host Haar detector. This
/// admits UHD 3840x2160 while bounding BGR and grayscale working sets.
#[cfg(feature = "opencv-face-detector")]
pub const MAXIMUM_HAAR_FACE_FRAME_PIXELS: u64 = 8_388_608;

/// Per-cascade byte bound, aligned with the Nano deployment asset cap.
#[cfg(feature = "opencv-face-detector")]
pub const MAXIMUM_OPENCV_HAAR_CASCADE_XML_BYTES: usize = 4 * 1_024 * 1_024;

/// Bounded OpenCV Haar search parameters.
///
/// Cascade identity and bytes are supplied separately to
/// [`OpenCvHaarFaceDetector::load`]. Keeping file paths out of this domain type
/// prevents a checked deployment asset from being reopened by name later.
#[cfg(feature = "opencv-face-detector")]
#[derive(Debug, Clone, PartialEq)]
pub struct OpenCvHaarFaceDetectorConfig {
    scale_factor: f64,
    frontal_minimum_neighbors: u32,
    profile_minimum_neighbors: u32,
    minimum_face_width: u32,
    minimum_face_height: u32,
    maximum_retained_detections: u32,
}

#[cfg(feature = "opencv-face-detector")]
impl OpenCvHaarFaceDetectorConfig {
    pub fn try_new(
        scale_factor: f64,
        frontal_minimum_neighbors: u32,
        profile_minimum_neighbors: u32,
        minimum_face_width: u32,
        minimum_face_height: u32,
        maximum_retained_detections: u32,
    ) -> Result<Self, OpenCvHaarFaceDetectorConfigError> {
        let config = Self {
            scale_factor,
            frontal_minimum_neighbors,
            profile_minimum_neighbors,
            minimum_face_width,
            minimum_face_height,
            maximum_retained_detections,
        };
        config.validate()?;
        Ok(config)
    }

    pub const fn scale_factor(&self) -> f64 {
        self.scale_factor
    }

    pub const fn frontal_minimum_neighbors(&self) -> u32 {
        self.frontal_minimum_neighbors
    }

    pub const fn profile_minimum_neighbors(&self) -> u32 {
        self.profile_minimum_neighbors
    }

    pub const fn minimum_face_width(&self) -> u32 {
        self.minimum_face_width
    }

    pub const fn minimum_face_height(&self) -> u32 {
        self.minimum_face_height
    }

    pub const fn maximum_retained_detections(&self) -> u32 {
        self.maximum_retained_detections
    }

    fn validate(&self) -> Result<(), OpenCvHaarFaceDetectorConfigError> {
        if !self.scale_factor.is_finite() || self.scale_factor <= 1.0 {
            return Err(OpenCvHaarFaceDetectorConfigError::InvalidScaleFactor {
                value: self.scale_factor,
            });
        }
        for (field, value) in [
            ("frontal_minimum_neighbors", self.frontal_minimum_neighbors),
            ("profile_minimum_neighbors", self.profile_minimum_neighbors),
        ] {
            if value == 0 {
                return Err(OpenCvHaarFaceDetectorConfigError::ZeroMinimumNeighbors { field });
            }
            if value > i32::MAX as u32 {
                return Err(
                    OpenCvHaarFaceDetectorConfigError::ParameterExceedsOpenCvInt { field, value },
                );
            }
        }
        for (field, value) in [
            ("minimum_face_width", self.minimum_face_width),
            ("minimum_face_height", self.minimum_face_height),
        ] {
            if value == 0 {
                return Err(OpenCvHaarFaceDetectorConfigError::ZeroMinimumFaceDimension { field });
            }
            if value > i32::MAX as u32 {
                return Err(
                    OpenCvHaarFaceDetectorConfigError::ParameterExceedsOpenCvInt { field, value },
                );
            }
        }
        if self.maximum_retained_detections == 0 {
            return Err(OpenCvHaarFaceDetectorConfigError::ZeroMaximumRetainedDetections);
        }
        if self.maximum_retained_detections > MAXIMUM_RETAINED_HAAR_FACE_DETECTIONS {
            return Err(
                OpenCvHaarFaceDetectorConfigError::MaximumRetainedDetectionsExceedsLimit {
                    value: self.maximum_retained_detections,
                    limit: MAXIMUM_RETAINED_HAAR_FACE_DETECTIONS,
                },
            );
        }
        Ok(())
    }

    #[cfg(not(oak_sys_check_only))]
    fn to_ffi(&self) -> face_ffi::RawOpenCvHaarFaceDetectorConfig {
        face_ffi::RawOpenCvHaarFaceDetectorConfig {
            scale_factor: self.scale_factor,
            frontal_minimum_neighbors: self.frontal_minimum_neighbors,
            profile_minimum_neighbors: self.profile_minimum_neighbors,
            minimum_face_width: self.minimum_face_width,
            minimum_face_height: self.minimum_face_height,
        }
    }
}

#[cfg(feature = "opencv-face-detector")]
fn validate_cascade_xml(
    cascade: &'static str,
    xml: &[u8],
) -> Result<(), OpenCvHaarFaceDetectorLoadError> {
    if xml.is_empty() {
        return Err(OpenCvHaarFaceDetectorLoadError::EmptyCascadeXml { cascade });
    }
    if xml.len() > MAXIMUM_OPENCV_HAAR_CASCADE_XML_BYTES {
        return Err(
            OpenCvHaarFaceDetectorLoadError::CascadeXmlExceedsByteLimit {
                cascade,
                actual_bytes: xml.len(),
                maximum_bytes: MAXIMUM_OPENCV_HAAR_CASCADE_XML_BYTES,
            },
        );
    }
    if xml.contains(&0) {
        return Err(OpenCvHaarFaceDetectorLoadError::CascadeXmlContainsNul { cascade });
    }
    Ok(())
}

#[cfg(feature = "opencv-face-detector")]
#[derive(Error, Debug, Clone, PartialEq)]
pub enum OpenCvHaarFaceDetectorConfigError {
    #[error("OpenCV Haar scale factor must be finite and greater than 1, got {value}")]
    InvalidScaleFactor { value: f64 },

    #[error("OpenCV Haar {field} count must be nonzero")]
    ZeroMinimumNeighbors { field: &'static str },

    #[error("OpenCV Haar {field} must be nonzero")]
    ZeroMinimumFaceDimension { field: &'static str },

    #[error("OpenCV Haar {field}={value} exceeds OpenCV's signed-int range")]
    ParameterExceedsOpenCvInt { field: &'static str, value: u32 },

    #[error("OpenCV Haar maximum retained-detection count must be nonzero")]
    ZeroMaximumRetainedDetections,

    #[error("OpenCV Haar maximum retained-detection count {value} exceeds hard limit {limit}")]
    MaximumRetainedDetectionsExceedsLimit { value: u32, limit: u32 },
}

/// Which exact cascade/search orientation produced a face rectangle.
#[cfg(feature = "opencv-face-detector")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HaarFaceDetectionSource {
    Frontal,
    Profile,
    MirroredProfile,
}

/// A validated rectangle in the source frame's pixel grid.
#[cfg(feature = "opencv-face-detector")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FacePixelRectangle {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

#[cfg(feature = "opencv-face-detector")]
impl FacePixelRectangle {
    pub const fn x(self) -> u32 {
        self.x
    }

    pub const fn y(self) -> u32 {
        self.y
    }

    pub const fn width(self) -> u32 {
        self.width
    }

    pub const fn height(self) -> u32 {
        self.height
    }

    pub fn right(self) -> u32 {
        self.x
            .checked_add(self.width)
            .expect("validated face rectangle right edge")
    }

    pub fn bottom(self) -> u32 {
        self.y
            .checked_add(self.height)
            .expect("validated face rectangle bottom edge")
    }

    pub const fn area(self) -> u64 {
        self.width as u64 * self.height as u64
    }
}

/// One validated OpenCV cascade detection.
///
/// `detector_level_weight` is OpenCV's finite, opaque cascade level weight. It
/// is not a calibrated confidence or probability, and retained-output
/// selection uses it only after rectangle width and height.
#[cfg(feature = "opencv-face-detector")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HaarFaceDetection {
    bounds: FacePixelRectangle,
    source: HaarFaceDetectionSource,
    detector_level_weight: f64,
}

#[cfg(feature = "opencv-face-detector")]
impl HaarFaceDetection {
    pub const fn bounds(self) -> FacePixelRectangle {
        self.bounds
    }

    pub const fn source(self) -> HaarFaceDetectionSource {
        self.source
    }

    pub const fn detector_level_weight(self) -> f64 {
        self.detector_level_weight
    }
}

/// A detector result bound to the exact borrowed RGB frame.
#[cfg(feature = "opencv-face-detector")]
#[derive(Debug, Clone, PartialEq)]
pub struct HaarFaceDetectionBatch {
    stream: StreamId,
    device_capture_sequence: DeviceFrameSequence,
    host_delivery_sequence: FrameDeliverySequence,
    timestamp: Timestamp,
    timestamp_reference: CameraTimestampReference,
    width: u32,
    height: u32,
    native_detection_count: usize,
    detections: Vec<HaarFaceDetection>,
}

#[cfg(feature = "opencv-face-detector")]
impl HaarFaceDetectionBatch {
    pub const fn stream(&self) -> StreamId {
        self.stream
    }

    pub const fn device_capture_sequence(&self) -> DeviceFrameSequence {
        self.device_capture_sequence
    }

    pub const fn host_delivery_sequence(&self) -> FrameDeliverySequence {
        self.host_delivery_sequence
    }

    pub const fn timestamp(&self) -> Timestamp {
        self.timestamp
    }

    pub const fn timestamp_reference(&self) -> CameraTimestampReference {
        self.timestamp_reference
    }

    pub const fn width(&self) -> u32 {
        self.width
    }

    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Candidate count returned by the first nonempty native cascade search,
    /// before the deterministic Rust-side output cap.
    pub const fn native_detection_count(&self) -> usize {
        self.native_detection_count
    }

    pub fn was_truncated(&self) -> bool {
        self.native_detection_count > self.detections.len()
    }

    /// Retained detections, ordered by descending rectangle width, then
    /// descending height, then descending opaque detector level weight, with
    /// deterministic coordinate/source tie-breaks.
    pub fn detections(&self) -> &[HaarFaceDetection] {
        &self.detections
    }

    pub fn into_detections(self) -> Vec<HaarFaceDetection> {
        self.detections
    }
}

#[cfg(feature = "opencv-face-detector")]
#[derive(Error, Debug, Clone, PartialEq)]
pub enum OpenCvHaarFaceDetectorLoadError {
    #[error("{cascade} OpenCV Haar cascade XML must not be empty")]
    EmptyCascadeXml { cascade: &'static str },

    #[error(
        "{cascade} OpenCV Haar cascade XML is {actual_bytes} bytes, exceeding hard limit {maximum_bytes}"
    )]
    CascadeXmlExceedsByteLimit {
        cascade: &'static str,
        actual_bytes: usize,
        maximum_bytes: usize,
    },

    #[error("{cascade} OpenCV Haar cascade XML must not contain an embedded NUL")]
    CascadeXmlContainsNul { cascade: &'static str },

    #[error("OpenCV Haar face detector is disabled by OAK_SYS_CHECK_ONLY=1")]
    CompileOnlyNativeDisabled,

    #[error("OpenCV failed to parse the supplied Haar cascade XML bytes: {message}")]
    Native { message: String },

    #[error("OpenCV returned a null Haar face detector")]
    NullDetector,
}

#[cfg(feature = "opencv-face-detector")]
#[derive(Error, Debug, Clone, PartialEq)]
pub enum HaarFaceDetectionError {
    #[error("OpenCV Haar face detection requires an RGB/BGR frame, got {stream:?}")]
    NonRgbFrame { stream: StreamId },

    #[error(
        "RGB frame is not tightly packed BGR888: {width}x{height}, stride={stride_bytes}, bytes={data_len}"
    )]
    InvalidBgrFrameLayout {
        width: u32,
        height: u32,
        stride_bytes: u32,
        data_len: usize,
    },

    #[error(
        "RGB frame {width}x{height} exceeds Haar detector limits: each axis <= {maximum_dimension}, total pixels <= {maximum_pixels}"
    )]
    FrameDimensionsExceedDetectorLimit {
        width: u32,
        height: u32,
        maximum_dimension: u32,
        maximum_pixels: u64,
    },

    #[error("OpenCV Haar face detector is disabled by OAK_SYS_CHECK_ONLY=1")]
    CompileOnlyNativeDisabled,

    #[error("OpenCV Haar face detection failed: {message}")]
    Native { message: String },

    #[error("native face detection {detection_index} has an unrecognized source kind")]
    InvalidSource { detection_index: usize },

    #[error(
        "native Haar result count {count} exceeds absolute bridge limit {limit}; OpenCV internal search work is not represented by this limit"
    )]
    NativeDetectionCountExceedsLimit { count: usize, limit: u32 },

    #[error(
        "native face detection {detection_index} changed source from {expected:?} to {actual:?} within one fallback batch"
    )]
    InconsistentSource {
        detection_index: usize,
        expected: HaarFaceDetectionSource,
        actual: HaarFaceDetectionSource,
    },

    #[error(
        "native face detection {detection_index} has a non-finite detector level weight {value}"
    )]
    NonFiniteDetectorLevelWeight { detection_index: usize, value: f64 },

    #[error(
        "native face detection {detection_index} has invalid rectangle ({x},{y},{width},{height}) for frame {frame_width}x{frame_height}"
    )]
    InvalidRectangle {
        detection_index: usize,
        x: i64,
        y: i64,
        width: i64,
        height: i64,
        frame_width: u32,
        frame_height: u32,
    },
}

/// Separately owned host detector. It owns no OAK device or DepthAI pipeline.
///
/// Successful compilation is not deployment-target runtime qualification of
/// OpenCV, either supplied cascade, or detector latency.
#[cfg(feature = "opencv-face-detector")]
pub struct OpenCvHaarFaceDetector {
    #[cfg(not(oak_sys_check_only))]
    inner: cxx::UniquePtr<face_ffi::OpenCvHaarFaceDetector>,
    config: OpenCvHaarFaceDetectorConfig,
}

#[cfg(feature = "opencv-face-detector")]
impl OpenCvHaarFaceDetector {
    /// Parse and load two exact, already-retained cascade XML payloads.
    ///
    /// Each payload is bounded and rejected when empty or when it contains an
    /// embedded NUL. The C++ boundary copies both borrowed slices before
    /// parsing them from memory; neither classifier nor this detector retains
    /// a Rust borrow. OpenCV performs the format-specific parse, so malformed
    /// or unsupported classifier XML is returned as a typed native load error.
    pub fn load(
        frontal_cascade_xml: &[u8],
        profile_cascade_xml: &[u8],
        config: OpenCvHaarFaceDetectorConfig,
    ) -> Result<Self, OpenCvHaarFaceDetectorLoadError> {
        validate_cascade_xml("frontal", frontal_cascade_xml)?;
        validate_cascade_xml("profile", profile_cascade_xml)?;

        #[cfg(oak_sys_check_only)]
        {
            let _ = (frontal_cascade_xml, profile_cascade_xml, config);
            Err(OpenCvHaarFaceDetectorLoadError::CompileOnlyNativeDisabled)
        }

        #[cfg(not(oak_sys_check_only))]
        {
            let raw_config = config.to_ffi();
            let inner = face_ffi::create_opencv_haar_face_detector(
                frontal_cascade_xml,
                profile_cascade_xml,
                &raw_config,
            )
            .map_err(|error| OpenCvHaarFaceDetectorLoadError::Native {
                message: error.what().to_owned(),
            })?;
            if inner.is_null() {
                return Err(OpenCvHaarFaceDetectorLoadError::NullDetector);
            }
            Ok(Self { inner, config })
        }
    }

    pub fn config(&self) -> &OpenCvHaarFaceDetectorConfig {
        &self.config
    }

    /// Detect faces by borrowing this frame's tightly packed BGR bytes.
    ///
    /// Native preprocessing converts BGR to grayscale and equalizes its
    /// histogram. Search then stops at the first nonempty source in this
    /// order: frontal, profile, then profile on a horizontally mirrored
    /// equalized image. Mirrored rectangles are mapped back into this frame's
    /// grid. Input dimensions and CXX/public boundary results have explicit
    /// limits; those result limits do not bound work OpenCV performs
    /// internally before returning candidates. Valid candidates are retained
    /// widest first, then tallest, before using the opaque finite level weight
    /// and deterministic coordinate/source tie-breaks.
    pub fn detect(
        &mut self,
        frame: &ImageFrame,
    ) -> Result<HaarFaceDetectionBatch, HaarFaceDetectionError> {
        validate_haar_detector_frame(frame)?;

        #[cfg(oak_sys_check_only)]
        {
            let _ = frame;
            Err(HaarFaceDetectionError::CompileOnlyNativeDisabled)
        }

        #[cfg(not(oak_sys_check_only))]
        {
            let raw = self
                .inner
                .pin_mut()
                .detect(
                    frame.pixels(),
                    frame.width,
                    frame.height,
                    frame.stride_bytes,
                )
                .map_err(|error| HaarFaceDetectionError::Native {
                    message: error.what().to_owned(),
                })?;
            parse_raw_haar_face_detection_batch(raw, frame, self.config.maximum_retained_detections)
        }
    }
}

#[cfg(feature = "opencv-face-detector")]
fn validate_haar_detector_frame(frame: &ImageFrame) -> Result<(), HaarFaceDetectionError> {
    if frame.stream != StreamId::Rgb {
        return Err(HaarFaceDetectionError::NonRgbFrame {
            stream: frame.stream,
        });
    }
    let frame_pixels = u64::from(frame.width) * u64::from(frame.height);
    if frame.width > MAXIMUM_HAAR_FACE_FRAME_DIMENSION
        || frame.height > MAXIMUM_HAAR_FACE_FRAME_DIMENSION
        || frame_pixels > MAXIMUM_HAAR_FACE_FRAME_PIXELS
    {
        return Err(HaarFaceDetectionError::FrameDimensionsExceedDetectorLimit {
            width: frame.width,
            height: frame.height,
            maximum_dimension: MAXIMUM_HAAR_FACE_FRAME_DIMENSION,
            maximum_pixels: MAXIMUM_HAAR_FACE_FRAME_PIXELS,
        });
    }
    let expected_stride = frame.width.checked_mul(3);
    let expected_len = expected_stride.and_then(|stride| {
        usize::try_from(stride).ok().and_then(|stride| {
            usize::try_from(frame.height)
                .ok()
                .and_then(|height| stride.checked_mul(height))
        })
    });
    if frame.width == 0
        || frame.height == 0
        || expected_stride != Some(frame.stride_bytes)
        || expected_len != Some(frame.data.len())
    {
        return Err(HaarFaceDetectionError::InvalidBgrFrameLayout {
            width: frame.width,
            height: frame.height,
            stride_bytes: frame.stride_bytes,
            data_len: frame.data.len(),
        });
    }
    Ok(())
}

#[cfg(all(feature = "opencv-face-detector", any(test, not(oak_sys_check_only))))]
fn parse_raw_haar_face_detection_batch(
    raw: face_ffi::RawHaarFaceDetectionBatch,
    frame: &ImageFrame,
    maximum_retained_detections: u32,
) -> Result<HaarFaceDetectionBatch, HaarFaceDetectionError> {
    let native_detection_count = raw.detections.len();
    if native_detection_count > MAXIMUM_NATIVE_HAAR_FACE_DETECTIONS as usize {
        return Err(HaarFaceDetectionError::NativeDetectionCountExceedsLimit {
            count: native_detection_count,
            limit: MAXIMUM_NATIVE_HAAR_FACE_DETECTIONS,
        });
    }
    let mut detections = Vec::with_capacity(native_detection_count);
    let mut batch_source = None;
    for (detection_index, raw) in raw.detections.into_iter().enumerate() {
        let source = match raw.source {
            face_ffi::RawHaarFaceDetectionSource::Frontal => HaarFaceDetectionSource::Frontal,
            face_ffi::RawHaarFaceDetectionSource::Profile => HaarFaceDetectionSource::Profile,
            face_ffi::RawHaarFaceDetectionSource::MirroredProfile => {
                HaarFaceDetectionSource::MirroredProfile
            }
            _ => {
                return Err(HaarFaceDetectionError::InvalidSource { detection_index });
            }
        };
        if let Some(expected) = batch_source {
            if source != expected {
                return Err(HaarFaceDetectionError::InconsistentSource {
                    detection_index,
                    expected,
                    actual: source,
                });
            }
        } else {
            batch_source = Some(source);
        }
        if !raw.detector_level_weight.is_finite() {
            return Err(HaarFaceDetectionError::NonFiniteDetectorLevelWeight {
                detection_index,
                value: raw.detector_level_weight,
            });
        }
        let rectangle_is_valid = raw.x >= 0
            && raw.y >= 0
            && raw.width > 0
            && raw.height > 0
            && u32::try_from(raw.x)
                .ok()
                .zip(u32::try_from(raw.width).ok())
                .and_then(|(x, width)| x.checked_add(width))
                .is_some_and(|right| right <= frame.width)
            && u32::try_from(raw.y)
                .ok()
                .zip(u32::try_from(raw.height).ok())
                .and_then(|(y, height)| y.checked_add(height))
                .is_some_and(|bottom| bottom <= frame.height);
        if !rectangle_is_valid {
            return Err(HaarFaceDetectionError::InvalidRectangle {
                detection_index,
                x: raw.x,
                y: raw.y,
                width: raw.width,
                height: raw.height,
                frame_width: frame.width,
                frame_height: frame.height,
            });
        }
        let bounds = FacePixelRectangle {
            x: u32::try_from(raw.x).expect("validated face x"),
            y: u32::try_from(raw.y).expect("validated face y"),
            width: u32::try_from(raw.width).expect("validated face width"),
            height: u32::try_from(raw.height).expect("validated face height"),
        };
        detections.push(HaarFaceDetection {
            bounds,
            source,
            detector_level_weight: raw.detector_level_weight,
        });
    }

    detections.sort_by(compare_haar_face_detections);
    detections.truncate(maximum_retained_detections as usize);
    Ok(HaarFaceDetectionBatch {
        stream: frame.stream,
        device_capture_sequence: frame.device_capture_sequence,
        host_delivery_sequence: frame.host_delivery_sequence,
        timestamp: frame.timestamp,
        timestamp_reference: frame.timestamp_reference,
        width: frame.width,
        height: frame.height,
        native_detection_count,
        detections,
    })
}

#[cfg(all(feature = "opencv-face-detector", any(test, not(oak_sys_check_only))))]
fn compare_haar_face_detections(
    left: &HaarFaceDetection,
    right: &HaarFaceDetection,
) -> std::cmp::Ordering {
    right
        .bounds
        .width
        .cmp(&left.bounds.width)
        .then_with(|| right.bounds.height.cmp(&left.bounds.height))
        .then_with(|| {
            right
                .detector_level_weight
                .total_cmp(&left.detector_level_weight)
        })
        .then_with(|| left.bounds.y.cmp(&right.bounds.y))
        .then_with(|| left.bounds.x.cmp(&right.bounds.x))
        .then_with(|| left.source.cmp(&right.source))
}

/// A valid depth frame
#[derive(Debug, Clone, PartialEq)]
pub struct DepthFrame {
    /// Native sequence returned by `ImgFrame::getSequenceNum()`.
    pub device_capture_sequence: DeviceFrameSequence,
    /// Strictly increasing bridge dequeue sequence, independent of capture identity.
    pub host_delivery_sequence: FrameDeliverySequence,
    /// Device clock at exposure midpoint.
    pub timestamp: Timestamp,
    pub timestamp_reference: CameraTimestampReference,
    pub exposure_time: ExposureTime,
    pub width: u32,
    pub height: u32,
    data: Vec<u16>, // Private - use depth_mm() or depth_m() accessors
    intrinsics: Intrinsics,
    connected_alignment: Option<DepthAlignment>,
}

impl DepthFrame {
    /// Raw depth values in millimeters. Invalid depths are 0.
    pub fn depth_mm(&self) -> &[u16] {
        &self.data
    }

    /// Projection intrinsics for this exact delivered depth grid.
    pub fn intrinsics(&self) -> Intrinsics {
        self.intrinsics
    }

    /// Alignment sealed by the connected device's validated configuration.
    pub fn connected_alignment(&self) -> Option<DepthAlignment> {
        self.connected_alignment
    }

    #[cfg(any(test, not(oak_sys_check_only)))]
    fn attach_connected_alignment(&mut self, alignment: DepthAlignment) {
        self.connected_alignment = Some(alignment);
    }

    /// Depth at pixel (x, y) in metres. Zero millimetres is invalid.
    pub fn depth_m_at(&self, x: u32, y: u32) -> Option<f32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let width = usize::try_from(self.width).ok()?;
        let x = usize::try_from(x).ok()?;
        let y = usize::try_from(y).ok()?;
        let index = y.checked_mul(width)?.checked_add(x)?;
        let raw = *self.data.get(index)?;
        if raw == 0 {
            return None;
        }
        Some(f32::from(raw) * 0.001)
    }

    /// Consume frame and take ownership of raw depth data
    pub fn into_depth_mm(self) -> Vec<u16> {
        self.data
    }
}

/// A single IMU measurement
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuSample {
    pub accel_timestamp: Timestamp,
    pub gyro_timestamp: Timestamp,
    /// Strictly increasing host-bridge dequeue sequence, not a device sequence;
    /// gaps are not evidence that the device dropped reports.
    pub sequence: u32,
    pub accel: Vec3,
    pub accel_accuracy: ImuAccuracy,
    pub gyro: Vec3,
    pub gyro_accuracy: ImuAccuracy,
}

/// Valid canonical pinhole intrinsics for one exact pixel grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Intrinsics {
    matrix: [[f32; 3]; 3],
    width: u32,
    height: u32,
}

impl Intrinsics {
    /// Parse a projection matrix for a specific pixel grid.
    pub fn try_from_projection_matrix(
        matrix: [[f32; 3]; 3],
        width: u32,
        height: u32,
    ) -> Result<Self, IntrinsicsError> {
        if width == 0 || height == 0 {
            return Err(IntrinsicsError::ZeroDimensions { width, height });
        }
        for (row, coefficients) in matrix.iter().enumerate() {
            for (column, value) in coefficients.iter().copied().enumerate() {
                if !value.is_finite() {
                    return Err(IntrinsicsError::NonFiniteCoefficient { row, column, value });
                }
            }
        }

        let fx = matrix[0][0];
        let fy = matrix[1][1];
        if fx <= 0.0 || fy <= 0.0 {
            return Err(IntrinsicsError::NonPositiveFocalLength { fx, fy });
        }

        // DepthAI defines this metadata as a pinhole K matrix. Permit only
        // round-off-sized deviations from its canonical zero/one entries.
        // The principal point is an independent translation term. It must not
        // loosen the contract for coefficients that are structurally zero.
        let coefficient_scale = 1.0_f32.max(fx.abs()).max(fy.abs());
        let zero_tolerance = 64.0 * f32::EPSILON * coefficient_scale;
        let one_tolerance = 64.0 * f32::EPSILON;
        if matrix[0][1].abs() > zero_tolerance
            || matrix[1][0].abs() > zero_tolerance
            || matrix[2][0].abs() > zero_tolerance
            || matrix[2][1].abs() > zero_tolerance
            || (matrix[2][2] - 1.0).abs() > one_tolerance
        {
            return Err(IntrinsicsError::UnsupportedProjectionMatrix { matrix });
        }

        Ok(Self {
            matrix,
            width,
            height,
        })
    }

    pub fn fx(self) -> f32 {
        self.matrix[0][0]
    }

    pub fn fy(self) -> f32 {
        self.matrix[1][1]
    }

    pub fn cx(self) -> f32 {
        self.matrix[0][2]
    }

    pub fn cy(self) -> f32 {
        self.matrix[1][2]
    }

    pub fn width(self) -> u32 {
        self.width
    }

    pub fn height(self) -> u32 {
        self.height
    }

    pub fn projection_matrix(self) -> [[f32; 3]; 3] {
        self.matrix
    }
}

/// Same-owner calibration evidence read from OAK EEPROM.
///
/// Coefficients are preserved exactly after fixed-shape and finite-value
/// admission. No matrix is inverted, orthonormalized, composed with another
/// transform, or relabelled as a Kiko robot-base calibration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OakEepromCalibrationEvidence {
    stereo_left_camera_socket: OakCameraSocket,
    stereo_right_camera_socket: OakCameraSocket,
    imu_to_camera_b_m: [[f32; 4]; 4],
    stereo_left_rectification_rotation_raw: [[f32; 3]; 3],
}

impl OakEepromCalibrationEvidence {
    /// Parse fixed-size matrix carriers into finite same-device calibration
    /// evidence without changing any coefficient.
    pub fn try_new(
        stereo_left_camera_socket: OakCameraSocket,
        stereo_right_camera_socket: OakCameraSocket,
        imu_to_camera_b_m: [[f32; 4]; 4],
        stereo_left_rectification_rotation_raw: [[f32; 3]; 3],
    ) -> Result<Self, CalibrationError> {
        require_stereo_camera_socket(
            EepromStereoSide::Left,
            OakCameraSocket::CameraB,
            stereo_left_camera_socket,
        )?;
        require_stereo_camera_socket(
            EepromStereoSide::Right,
            OakCameraSocket::CameraC,
            stereo_right_camera_socket,
        )?;
        require_finite_calibration_matrix(
            EepromCalibrationMatrix::ImuToCameraBMeters,
            &imu_to_camera_b_m,
        )?;
        require_finite_calibration_matrix(
            EepromCalibrationMatrix::StereoLeftRectificationRotationRaw,
            &stereo_left_rectification_rotation_raw,
        )?;
        Ok(Self {
            stereo_left_camera_socket,
            stereo_right_camera_socket,
            imu_to_camera_b_m,
            stereo_left_rectification_rotation_raw,
        })
    }

    pub const fn stereo_left_camera_socket(self) -> OakCameraSocket {
        self.stereo_left_camera_socket
    }

    pub const fn stereo_right_camera_socket(self) -> OakCameraSocket {
        self.stereo_right_camera_socket
    }

    /// Exact homogeneous matrix returned by DepthAI
    /// `getImuToCameraExtrinsics(CAM_B, false, METER)`.
    ///
    /// DepthAI documents the translation column in metres. This carrier makes
    /// no stronger camera-frame convention claim than that API call.
    pub const fn imu_to_camera_b_m(self) -> [[f32; 4]; 4] {
        self.imu_to_camera_b_m
    }

    /// Exact coefficient matrix returned by DepthAI
    /// `getStereoLeftRectificationRotation()`.
    ///
    /// The pinned API identifies the stereo-left source socket separately as
    /// CAM_B but does not document this matrix's transform direction. This
    /// raw output therefore carries no source-to-destination transform claim.
    pub const fn stereo_left_rectification_rotation_raw(self) -> [[f32; 3]; 3] {
        self.stereo_left_rectification_rotation_raw
    }
}

/// Information about an available device
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub device_id: String,
    pub name: String,
    pub state: DeviceState,
}

impl From<ffi::DeviceInfo> for DeviceInfo {
    fn from(d: ffi::DeviceInfo) -> Self {
        Self {
            device_id: d.device_id,
            name: d.name,
            state: d.state,
        }
    }
}

/// Provenance compiled into the native bridge from DepthAI's build header.
///
/// Embedded artifact versions describe inputs bundled with that SDK build;
/// they are not proof of firmware currently executing on a physical device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DepthAiBuildMetadata {
    sdk_version: String,
    sdk_commit: String,
    embedded_device_artifact_version: String,
    embedded_bootloader_artifact_version: String,
}

impl DepthAiBuildMetadata {
    pub fn sdk_version(&self) -> &str {
        &self.sdk_version
    }

    pub fn sdk_commit(&self) -> &str {
        &self.sdk_commit
    }

    pub fn embedded_device_artifact_version(&self) -> &str {
        &self.embedded_device_artifact_version
    }

    pub fn embedded_bootloader_artifact_version(&self) -> &str {
        &self.embedded_bootloader_artifact_version
    }
}

/// Exact, non-conflated identity fields observed for the opened OAK device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectedDeviceIdentity {
    mxid: String,
    discovery_transport_name: Option<String>,
    eeprom_device_name: Option<String>,
    product_name: Option<String>,
}

impl ConnectedDeviceIdentity {
    pub fn mxid(&self) -> &str {
        &self.mxid
    }

    pub fn discovery_transport_name(&self) -> Option<&str> {
        self.discovery_transport_name.as_deref()
    }

    pub fn eeprom_device_name(&self) -> Option<&str> {
        self.eeprom_device_name.as_deref()
    }

    pub fn product_name(&self) -> Option<&str> {
        self.product_name.as_deref()
    }
}

#[cfg(any(test, not(oak_sys_check_only)))]
fn parse_depthai_build_metadata(
    raw: ffi::DepthAiBuildMetadata,
) -> Result<DepthAiBuildMetadata, DepthAiBuildMetadataError> {
    for (field, value) in [
        ("sdk_version", raw.sdk_version.as_str()),
        ("sdk_commit", raw.sdk_commit.as_str()),
        (
            "embedded_device_artifact_version",
            raw.embedded_device_artifact_version.as_str(),
        ),
        (
            "embedded_bootloader_artifact_version",
            raw.embedded_bootloader_artifact_version.as_str(),
        ),
    ] {
        if value.is_empty() {
            return Err(DepthAiBuildMetadataError::EmptyRequiredField { field });
        }
    }

    Ok(DepthAiBuildMetadata {
        sdk_version: raw.sdk_version,
        sdk_commit: raw.sdk_commit,
        embedded_device_artifact_version: raw.embedded_device_artifact_version,
        embedded_bootloader_artifact_version: raw.embedded_bootloader_artifact_version,
    })
}

#[cfg(any(test, not(oak_sys_check_only)))]
fn parse_connected_device_identity(
    raw: ffi::ConnectedDeviceIdentity,
    requested_mxid: &str,
) -> Result<ConnectedDeviceIdentity, ConnectedDeviceIdentityError> {
    if raw.mxid.is_empty() {
        return Err(ConnectedDeviceIdentityError::EmptyMxid);
    }
    if raw.mxid != requested_mxid {
        return Err(ConnectedDeviceIdentityError::SelectorMismatch {
            requested_mxid: requested_mxid.to_owned(),
            actual_mxid: raw.mxid,
        });
    }

    let optional = |value: String| (!value.is_empty()).then_some(value);
    Ok(ConnectedDeviceIdentity {
        mxid: raw.mxid,
        discovery_transport_name: optional(raw.discovery_transport_name),
        eeprom_device_name: optional(raw.eeprom_device_name),
        product_name: optional(raw.product_name),
    })
}

#[cfg(not(oak_sys_check_only))]
pub fn depthai_build_metadata() -> Result<DepthAiBuildMetadata, DepthAiBuildMetadataError> {
    let raw = ffi::depthai_build_metadata().map_err(|error| DepthAiBuildMetadataError::Native {
        message: error.what().to_owned(),
    })?;
    parse_depthai_build_metadata(raw)
}

#[cfg(oak_sys_check_only)]
pub fn depthai_build_metadata() -> Result<DepthAiBuildMetadata, DepthAiBuildMetadataError> {
    Err(DepthAiBuildMetadataError::CompileOnlyNativeDisabled)
}

// ============================================================================
// DEVICE - Safe wrapper that enforces valid usage
// ============================================================================

/// An OAK-D pipeline that connected successfully when constructed.
///
/// A later physical disconnect is reported by acquisition methods.
#[cfg(not(oak_sys_check_only))]
pub struct Device {
    inner: cxx::UniquePtr<ffi::OakDevice>,
    config: DeviceConfig,
    connected_identity: ConnectedDeviceIdentity,
    usb_transport: UsbTransportAdmissionEvidence,
}

#[cfg(not(oak_sys_check_only))]
impl Device {
    /// List all visible OAK devices, including devices already in use.
    pub fn list() -> Result<Vec<DeviceInfo>, DeviceDiscoveryError> {
        Ok(ffi::list_devices()
            .map_err(|error| DeviceDiscoveryError {
                message: error.what().to_owned(),
            })?
            .into_iter()
            .map(DeviceInfo::from)
            .collect())
    }

    /// Connect to one exact, nonempty DepthAI MXID and verify what was opened.
    pub fn connect(selector: &str, config: DeviceConfig) -> Result<Self, ConnectionError> {
        if selector.is_empty() {
            return Err(ConnectionError::EmptySelector);
        }
        config.validate()?;
        let ffi_config = config.to_ffi();
        let inner =
            ffi::create_device(selector, &ffi_config).map_err(|error| ConnectionError::Native {
                selector: selector.to_owned(),
                message: error.what().to_owned(),
            })?;

        if inner.is_null() {
            return Err(ConnectionError::NullDevice {
                selector: selector.to_owned(),
            });
        }

        if !inner.is_connected() {
            return Err(ConnectionError::NotConnected {
                selector: selector.to_owned(),
            });
        }

        // This queries the exact device already owned by the started pipeline;
        // it does not enumerate or reopen a default device.
        let raw_usb_speed =
            inner
                .get_usb_speed()
                .map_err(|error| ConnectionError::UsbTransportReadbackNative {
                    selector: selector.to_owned(),
                    message: error.what().to_owned(),
                })?;
        let usb_transport = admit_usb_transport(config.usb_transport, raw_usb_speed)?;

        let raw_identity = inner.get_connected_device_identity().map_err(|error| {
            ConnectedDeviceIdentityError::Native {
                message: error.what().to_owned(),
            }
        })?;
        let connected_identity = parse_connected_device_identity(raw_identity, selector)?;

        Ok(Self {
            inner,
            config,
            connected_identity,
            usb_transport,
        })
    }

    pub fn connected_identity(
        &self,
    ) -> Result<&ConnectedDeviceIdentity, ConnectedDeviceIdentityError> {
        Ok(&self.connected_identity)
    }

    pub fn usb_transport_evidence(
        &self,
    ) -> Result<&UsbTransportAdmissionEvidence, UsbTransportEvidenceError> {
        Ok(&self.usb_transport)
    }

    /// Get RGB frame. Returns error if RGB not enabled in config.
    pub fn rgb(&mut self, timeout_ms: u32) -> Result<ImageFrame, ImageError> {
        if self.config.rgb.is_none() {
            return Err(ImageError::StreamNotEnabled {
                stream: StreamId::Rgb,
            });
        }
        let result = self
            .inner
            .pin_mut()
            .try_get_rgb(timeout_ms)
            .map_err(|error| ImageError::Native {
                message: error.what().to_owned(),
            })?;
        parse_image_result(result, timeout_ms)
    }

    /// Get left mono frame. Returns error if mono not enabled in config.
    pub fn mono_left(&mut self, timeout_ms: u32) -> Result<ImageFrame, ImageError> {
        if self.config.mono.is_none() {
            return Err(ImageError::StreamNotEnabled {
                stream: StreamId::MonoLeft,
            });
        }
        let result = self
            .inner
            .pin_mut()
            .try_get_mono_left(timeout_ms)
            .map_err(|error| ImageError::Native {
                message: error.what().to_owned(),
            })?;
        parse_image_result(result, timeout_ms)
    }

    /// Get right mono frame. Returns error if mono not enabled in config.
    pub fn mono_right(&mut self, timeout_ms: u32) -> Result<ImageFrame, ImageError> {
        if self.config.mono.is_none() {
            return Err(ImageError::StreamNotEnabled {
                stream: StreamId::MonoRight,
            });
        }
        let result = self
            .inner
            .pin_mut()
            .try_get_mono_right(timeout_ms)
            .map_err(|error| ImageError::Native {
                message: error.what().to_owned(),
            })?;
        parse_image_result(result, timeout_ms)
    }

    /// Get depth frame. Returns error if depth not enabled in config.
    pub fn depth(&mut self, timeout_ms: u32) -> Result<DepthFrame, DepthError> {
        if self.config.depth.is_none() {
            return Err(DepthError::StreamNotEnabled);
        }
        let result = self
            .inner
            .pin_mut()
            .try_get_depth(timeout_ms)
            .map_err(|error| DepthError::Native {
                message: error.what().to_owned(),
            })?;
        let mut frame = parse_depth_result(result, timeout_ms)?;
        let alignment = self
            .config
            .depth
            .expect("depth stream was checked as enabled")
            .alignment;
        frame.attach_connected_alignment(alignment);
        Ok(frame)
    }

    /// Get IMU samples. Returns error if IMU not enabled in config.
    pub fn imu(&mut self) -> Result<Vec<ImuSample>, ImuError> {
        if self.config.imu.is_none() {
            return Err(ImuError::StreamNotEnabled);
        }
        let result = self
            .inner
            .pin_mut()
            .get_imu_batch()
            .map_err(|error| ImuError::Native {
                message: error.what().to_owned(),
            })?;
        parse_imu_result(result)
    }

    /// Calibrated stereo baseline in metres.
    pub fn stereo_baseline_m(&self) -> Result<f32, CalibrationError> {
        let value =
            self.inner
                .get_stereo_baseline_m()
                .map_err(|error| CalibrationError::Native {
                    message: error.what().to_owned(),
                })?;
        parse_stereo_baseline_m(value)
    }

    /// Read exact EEPROM calibration evidence from this pipeline's
    /// already-opened device.
    pub fn eeprom_calibration_evidence(
        &self,
    ) -> Result<OakEepromCalibrationEvidence, CalibrationError> {
        let raw = self
            .inner
            .get_eeprom_calibration_evidence()
            .map_err(|error| CalibrationError::Native {
                message: error.what().to_owned(),
            })?;
        parse_eeprom_calibration_evidence(raw)
    }

    /// Gracefully disconnect and report any native shutdown failure.
    pub fn close(mut self) -> Result<(), CloseError> {
        self.inner.pin_mut().close().map_err(|error| CloseError {
            message: error.what().to_owned(),
        })
    }
}

/// Compile-only stand-in. It type-checks downstream contracts without
/// pretending that native discovery, capture, or shutdown occurred.
#[cfg(oak_sys_check_only)]
pub struct Device {
    _private: (),
}

#[cfg(oak_sys_check_only)]
impl Device {
    pub fn list() -> Result<Vec<DeviceInfo>, DeviceDiscoveryError> {
        Err(DeviceDiscoveryError {
            message: "native runtime disabled by OAK_SYS_CHECK_ONLY=1".to_owned(),
        })
    }

    pub fn connect(selector: &str, config: DeviceConfig) -> Result<Self, ConnectionError> {
        if selector.is_empty() {
            return Err(ConnectionError::EmptySelector);
        }
        config.validate()?;
        Err(ConnectionError::CompileOnlyNativeDisabled)
    }

    pub fn connected_identity(
        &self,
    ) -> Result<&ConnectedDeviceIdentity, ConnectedDeviceIdentityError> {
        Err(ConnectedDeviceIdentityError::CompileOnlyNativeDisabled)
    }

    pub fn usb_transport_evidence(
        &self,
    ) -> Result<&UsbTransportAdmissionEvidence, UsbTransportEvidenceError> {
        Err(UsbTransportEvidenceError::CompileOnlyNativeDisabled)
    }

    pub fn rgb(&mut self, _timeout_ms: u32) -> Result<ImageFrame, ImageError> {
        Err(ImageError::Native {
            message: "native runtime disabled by OAK_SYS_CHECK_ONLY=1".to_owned(),
        })
    }

    pub fn mono_left(&mut self, _timeout_ms: u32) -> Result<ImageFrame, ImageError> {
        Err(ImageError::Native {
            message: "native runtime disabled by OAK_SYS_CHECK_ONLY=1".to_owned(),
        })
    }

    pub fn mono_right(&mut self, _timeout_ms: u32) -> Result<ImageFrame, ImageError> {
        Err(ImageError::Native {
            message: "native runtime disabled by OAK_SYS_CHECK_ONLY=1".to_owned(),
        })
    }

    pub fn depth(&mut self, _timeout_ms: u32) -> Result<DepthFrame, DepthError> {
        Err(DepthError::Native {
            message: "native runtime disabled by OAK_SYS_CHECK_ONLY=1".to_owned(),
        })
    }

    pub fn imu(&mut self) -> Result<Vec<ImuSample>, ImuError> {
        Err(ImuError::Native {
            message: "native runtime disabled by OAK_SYS_CHECK_ONLY=1".to_owned(),
        })
    }

    pub fn stereo_baseline_m(&self) -> Result<f32, CalibrationError> {
        Err(CalibrationError::Native {
            message: "native runtime disabled by OAK_SYS_CHECK_ONLY=1".to_owned(),
        })
    }

    pub fn eeprom_calibration_evidence(
        &self,
    ) -> Result<OakEepromCalibrationEvidence, CalibrationError> {
        Err(CalibrationError::Native {
            message: "native runtime disabled by OAK_SYS_CHECK_ONLY=1".to_owned(),
        })
    }

    pub fn close(self) -> Result<(), CloseError> {
        Err(CloseError {
            message: "native runtime disabled by OAK_SYS_CHECK_ONLY=1".to_owned(),
        })
    }
}

// ============================================================================
// PARSING FUNCTIONS - Convert FFI results to proper Result types
// ============================================================================

#[cfg(any(test, not(oak_sys_check_only)))]
fn parse_stereo_baseline_m(value: f32) -> Result<f32, CalibrationError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(CalibrationError::InvalidBaseline { value });
    }
    Ok(value)
}

#[cfg(any(test, not(oak_sys_check_only)))]
fn parse_calibration_matrix_shape<const ROWS: usize, const COLUMNS: usize>(
    matrix: EepromCalibrationMatrix,
    coefficients: Vec<f32>,
) -> Result<[[f32; COLUMNS]; ROWS], CalibrationError> {
    let expected = ROWS * COLUMNS;
    if coefficients.len() != expected {
        return Err(CalibrationError::InvalidMatrixLength {
            matrix,
            expected,
            actual: coefficients.len(),
        });
    }
    Ok(std::array::from_fn(|row| {
        std::array::from_fn(|column| coefficients[row * COLUMNS + column])
    }))
}

fn require_stereo_camera_socket(
    side: EepromStereoSide,
    expected: OakCameraSocket,
    actual: OakCameraSocket,
) -> Result<(), CalibrationError> {
    if actual == expected {
        Ok(())
    } else {
        Err(CalibrationError::UnexpectedStereoCameraSocket {
            side,
            expected,
            actual,
        })
    }
}

fn require_finite_calibration_matrix<const ROWS: usize, const COLUMNS: usize>(
    matrix: EepromCalibrationMatrix,
    coefficients: &[[f32; COLUMNS]; ROWS],
) -> Result<(), CalibrationError> {
    for (row, values) in coefficients.iter().enumerate() {
        for (column, value) in values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(CalibrationError::NonFiniteMatrixCoefficient {
                    matrix,
                    row,
                    column,
                    value,
                });
            }
        }
    }
    Ok(())
}

#[cfg(any(test, not(oak_sys_check_only)))]
fn parse_eeprom_calibration_evidence(
    raw: ffi::EepromCalibrationEvidence,
) -> Result<OakEepromCalibrationEvidence, CalibrationError> {
    // Preserve deterministic boundary precedence: fixed shapes first, then
    // socket identities and coefficient domains in evidence order.
    let imu_to_camera_b_m = parse_calibration_matrix_shape(
        EepromCalibrationMatrix::ImuToCameraBMeters,
        raw.imu_to_camera_b_m,
    )?;
    let stereo_left_rectification_rotation_raw = parse_calibration_matrix_shape(
        EepromCalibrationMatrix::StereoLeftRectificationRotationRaw,
        raw.stereo_left_rectification_rotation_raw,
    )?;
    OakEepromCalibrationEvidence::try_new(
        raw.stereo_left_camera_socket.into(),
        raw.stereo_right_camera_socket.into(),
        imu_to_camera_b_m,
        stereo_left_rectification_rotation_raw,
    )
}

#[cfg(any(test, not(oak_sys_check_only)))]
fn parse_intrinsics(
    raw: ffi::Intrinsics,
    frame_width: u32,
    frame_height: u32,
) -> Result<Intrinsics, IntrinsicsError> {
    let intrinsics = Intrinsics::try_from_projection_matrix(
        [
            [raw.m00, raw.m01, raw.m02],
            [raw.m10, raw.m11, raw.m12],
            [raw.m20, raw.m21, raw.m22],
        ],
        raw.width,
        raw.height,
    )?;
    if intrinsics.width() != frame_width || intrinsics.height() != frame_height {
        return Err(IntrinsicsError::DimensionMismatch {
            intrinsic_width: intrinsics.width(),
            intrinsic_height: intrinsics.height(),
            frame_width,
            frame_height,
        });
    }
    Ok(intrinsics)
}

#[cfg(any(test, not(oak_sys_check_only)))]
fn parse_image_result(
    result: ffi::ImageFrameResult,
    timeout_ms: u32,
) -> Result<ImageFrame, ImageError> {
    match result.status {
        ffi::FrameStatus::Ok => {
            let channels = match result.frame.stream {
                StreamId::Rgb => 3_u32,
                StreamId::MonoLeft | StreamId::MonoRight => 1_u32,
                StreamId::Depth | StreamId::Imu => return Err(ImageError::Corrupt),
                _ => return Err(ImageError::Corrupt),
            };
            let expected_stride = result
                .frame
                .width
                .checked_mul(channels)
                .ok_or(ImageError::Corrupt)?;
            let expected_len = usize::try_from(expected_stride)
                .ok()
                .and_then(|stride| {
                    usize::try_from(result.frame.height)
                        .ok()
                        .and_then(|height| stride.checked_mul(height))
                })
                .ok_or(ImageError::Corrupt)?;
            if result.frame.width == 0
                || result.frame.height == 0
                || result.frame.stride_bytes != expected_stride
                || result.frame.data.len() != expected_len
            {
                return Err(ImageError::Corrupt);
            }
            let intrinsics = parse_intrinsics(
                result.frame.intrinsics,
                result.frame.width,
                result.frame.height,
            )?;
            let device_capture_sequence =
                DeviceFrameSequence::try_from_i64(result.frame.device_capture_sequence)?;
            let exposure_time = parse_exposure_time(result.frame.exposure_time_us)?;
            Ok(ImageFrame {
                stream: result.frame.stream,
                device_capture_sequence,
                host_delivery_sequence: FrameDeliverySequence::new(
                    result.frame.host_delivery_sequence,
                ),
                timestamp: Timestamp::try_from(result.frame.timestamp)?,
                timestamp_reference: CameraTimestampReference::ExposureMidpoint,
                exposure_time,
                width: result.frame.width,
                height: result.frame.height,
                stride_bytes: result.frame.stride_bytes,
                data: result.frame.data,
                intrinsics,
            })
        }
        ffi::FrameStatus::Timeout => Err(ImageError::Timeout { timeout_ms }),
        ffi::FrameStatus::QueueEmpty => Err(ImageError::QueueEmpty),
        ffi::FrameStatus::Disconnected => Err(ImageError::Disconnected),
        ffi::FrameStatus::Corrupt => Err(ImageError::Corrupt),
        ffi::FrameStatus::StreamNotEnabled => Err(ImageError::StreamNotEnabled {
            stream: result.frame.stream,
        }),
        _ => Err(ImageError::Corrupt),
    }
}

#[cfg(any(test, not(oak_sys_check_only)))]
fn parse_depth_result(
    result: ffi::DepthFrameResult,
    timeout_ms: u32,
) -> Result<DepthFrame, DepthError> {
    match result.status {
        ffi::FrameStatus::Ok => {
            let expected_len = usize::try_from(result.frame.width)
                .ok()
                .and_then(|width| {
                    usize::try_from(result.frame.height)
                        .ok()
                        .and_then(|height| width.checked_mul(height))
                })
                .ok_or(DepthError::Corrupt)?;
            if result.frame.width == 0
                || result.frame.height == 0
                || result.frame.data.len() != expected_len
            {
                return Err(DepthError::Corrupt);
            }
            let intrinsics = parse_intrinsics(
                result.frame.intrinsics,
                result.frame.width,
                result.frame.height,
            )?;
            let device_capture_sequence =
                DeviceFrameSequence::try_from_i64(result.frame.device_capture_sequence)?;
            let exposure_time = parse_exposure_time(result.frame.exposure_time_us)?;
            Ok(DepthFrame {
                device_capture_sequence,
                host_delivery_sequence: FrameDeliverySequence::new(
                    result.frame.host_delivery_sequence,
                ),
                timestamp: Timestamp::try_from(result.frame.timestamp)?,
                timestamp_reference: CameraTimestampReference::ExposureMidpoint,
                exposure_time,
                width: result.frame.width,
                height: result.frame.height,
                data: result.frame.data,
                intrinsics,
                connected_alignment: None,
            })
        }
        ffi::FrameStatus::Timeout => Err(DepthError::Timeout { timeout_ms }),
        ffi::FrameStatus::QueueEmpty => Err(DepthError::QueueEmpty),
        ffi::FrameStatus::Disconnected => Err(DepthError::Disconnected),
        ffi::FrameStatus::Corrupt => Err(DepthError::Corrupt),
        ffi::FrameStatus::StreamNotEnabled => Err(DepthError::StreamNotEnabled),
        _ => Err(DepthError::Corrupt),
    }
}

#[cfg(any(test, not(oak_sys_check_only)))]
fn parse_imu_accuracy(value: ffi::ImuAccuracy) -> Result<ImuAccuracy, ImuError> {
    match value {
        ffi::ImuAccuracy::Unreliable
        | ffi::ImuAccuracy::Low
        | ffi::ImuAccuracy::Medium
        | ffi::ImuAccuracy::High => Ok(value),
        _ => Err(ImuError::Corrupt),
    }
}

#[cfg(any(test, not(oak_sys_check_only)))]
fn parse_imu_result(result: ffi::ImuBatchResult) -> Result<Vec<ImuSample>, ImuError> {
    match result.status {
        ffi::ImuStatus::Ok => {
            if result.batch.samples.is_empty() {
                return Err(ImuError::Corrupt);
            }
            let samples = result
                .batch
                .samples
                .into_iter()
                .enumerate()
                .map(|(sample_index, s)| {
                    for (field, value) in [
                        ("accel_x", s.accel_x),
                        ("accel_y", s.accel_y),
                        ("accel_z", s.accel_z),
                        ("gyro_x", s.gyro_x),
                        ("gyro_y", s.gyro_y),
                        ("gyro_z", s.gyro_z),
                    ] {
                        if !value.is_finite() {
                            return Err(ImuError::NonFiniteSample {
                                sample_index,
                                field,
                            });
                        }
                    }
                    let accel_accuracy = parse_imu_accuracy(s.accel_accuracy)?;
                    let gyro_accuracy = parse_imu_accuracy(s.gyro_accuracy)?;
                    let accel_timestamp =
                        Timestamp::try_from(s.accel_timestamp).map_err(|source| {
                            ImuError::InvalidTimestamp {
                                sample_index,
                                sensor: "accelerometer",
                                source,
                            }
                        })?;
                    let gyro_timestamp =
                        Timestamp::try_from(s.gyro_timestamp).map_err(|source| {
                            ImuError::InvalidTimestamp {
                                sample_index,
                                sensor: "gyroscope",
                                source,
                            }
                        })?;
                    Ok(ImuSample {
                        accel_timestamp,
                        gyro_timestamp,
                        sequence: s.sequence,
                        accel: Vec3 {
                            x: s.accel_x,
                            y: s.accel_y,
                            z: s.accel_z,
                        },
                        accel_accuracy,
                        gyro: Vec3 {
                            x: s.gyro_x,
                            y: s.gyro_y,
                            z: s.gyro_z,
                        },
                        gyro_accuracy,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(samples)
        }
        ffi::ImuStatus::Empty => Err(ImuError::Empty),
        ffi::ImuStatus::Disconnected => Err(ImuError::Disconnected),
        ffi::ImuStatus::Corrupt => Err(ImuError::Corrupt),
        _ => Err(ImuError::Corrupt),
    }
}

#[cfg(not(oak_sys_check_only))]
impl Drop for Device {
    fn drop(&mut self) {
        // Destructors cannot report errors. Call `Device::close` explicitly if
        // shutdown success matters to the caller.
        if !self.inner.is_null() && self.inner.is_connected() {
            let _ = self.inner.pin_mut().close();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mono_config() -> MonoConfig {
        MonoConfig {
            width: 640,
            height: 480,
            fps: 30,
            rectified: true,
        }
    }

    fn base_config() -> DeviceConfig {
        DeviceConfig {
            usb_transport: UsbTransportPolicy::super_speed_required(),
            rgb: None,
            mono: Some(mono_config()),
            depth: None,
            imu: None,
            queue: QueueConfig::default(),
        }
    }

    fn raw_intrinsics(width: u32, height: u32) -> ffi::Intrinsics {
        ffi::Intrinsics {
            m00: 400.0,
            m01: 0.0,
            m02: width as f32 * 0.5,
            m10: 0.0,
            m11: 401.0,
            m12: height as f32 * 0.5,
            m20: 0.0,
            m21: 0.0,
            m22: 1.0,
            width,
            height,
        }
    }

    fn raw_image(
        stream: StreamId,
        width: u32,
        height: u32,
        stride_bytes: u32,
        data: Vec<u8>,
    ) -> ffi::ImageFrameResult {
        ffi::ImageFrameResult {
            status: ffi::FrameStatus::Ok,
            frame: ffi::ImageFrame {
                stream,
                device_capture_sequence: 41,
                host_delivery_sequence: 1,
                timestamp: ffi::Timestamp { device_ns: 2 },
                exposure_time_us: 3,
                width,
                height,
                stride_bytes,
                data,
                intrinsics: raw_intrinsics(width, height),
            },
        }
    }

    fn raw_depth(width: u32, height: u32, data: Vec<u16>) -> ffi::DepthFrameResult {
        ffi::DepthFrameResult {
            status: ffi::FrameStatus::Ok,
            frame: ffi::DepthFrame {
                device_capture_sequence: 41,
                host_delivery_sequence: 1,
                timestamp: ffi::Timestamp { device_ns: 2 },
                exposure_time_us: 3,
                width,
                height,
                data,
                intrinsics: raw_intrinsics(width, height),
            },
        }
    }

    fn raw_build_metadata() -> ffi::DepthAiBuildMetadata {
        ffi::DepthAiBuildMetadata {
            sdk_version: "3.6.1".to_owned(),
            sdk_commit: "commit-abc".to_owned(),
            embedded_device_artifact_version: "device-1".to_owned(),
            embedded_bootloader_artifact_version: "bootloader-1".to_owned(),
        }
    }

    fn raw_connected_identity() -> ffi::ConnectedDeviceIdentity {
        ffi::ConnectedDeviceIdentity {
            mxid: "mxid-123".to_owned(),
            discovery_transport_name: "usb".to_owned(),
            eeprom_device_name: "kiko-oak".to_owned(),
            product_name: "OAK-D".to_owned(),
        }
    }

    fn raw_eeprom_calibration_evidence() -> ffi::EepromCalibrationEvidence {
        ffi::EepromCalibrationEvidence {
            stereo_left_camera_socket: ffi::CameraSocket::CameraB,
            stereo_right_camera_socket: ffi::CameraSocket::CameraC,
            imu_to_camera_b_m: vec![
                1.0, 0.0, 0.0, 0.01, 0.0, 1.0, 0.0, -0.02, 0.0, 0.0, 1.0, 0.03, 0.0, 0.0, 0.0, 1.0,
            ],
            stereo_left_rectification_rotation_raw: vec![
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    #[test]
    fn build_metadata_parser_preserves_exact_fields_and_rejects_empty_required_values() {
        let parsed = parse_depthai_build_metadata(raw_build_metadata()).expect("valid metadata");
        assert_eq!(parsed.sdk_version(), "3.6.1");
        assert_eq!(parsed.sdk_commit(), "commit-abc");
        assert_eq!(parsed.embedded_device_artifact_version(), "device-1");
        assert_eq!(
            parsed.embedded_bootloader_artifact_version(),
            "bootloader-1"
        );

        for field in [
            "sdk_version",
            "sdk_commit",
            "embedded_device_artifact_version",
            "embedded_bootloader_artifact_version",
        ] {
            let mut raw = raw_build_metadata();
            match field {
                "sdk_version" => raw.sdk_version.clear(),
                "sdk_commit" => raw.sdk_commit.clear(),
                "embedded_device_artifact_version" => raw.embedded_device_artifact_version.clear(),
                "embedded_bootloader_artifact_version" => {
                    raw.embedded_bootloader_artifact_version.clear()
                }
                _ => unreachable!(),
            }
            assert_eq!(
                parse_depthai_build_metadata(raw),
                Err(DepthAiBuildMetadataError::EmptyRequiredField { field })
            );
        }
    }

    #[test]
    fn connected_identity_requires_and_verifies_the_exact_mxid() {
        let parsed = parse_connected_device_identity(raw_connected_identity(), "mxid-123")
            .expect("matching identity");
        assert_eq!(parsed.mxid(), "mxid-123");
        assert_eq!(parsed.discovery_transport_name(), Some("usb"));
        assert_eq!(parsed.eeprom_device_name(), Some("kiko-oak"));
        assert_eq!(parsed.product_name(), Some("OAK-D"));

        let mut empty = raw_connected_identity();
        empty.mxid.clear();
        assert_eq!(
            parse_connected_device_identity(empty, "mxid-123"),
            Err(ConnectedDeviceIdentityError::EmptyMxid)
        );
        assert_eq!(
            parse_connected_device_identity(raw_connected_identity(), "other"),
            Err(ConnectedDeviceIdentityError::SelectorMismatch {
                requested_mxid: "other".to_owned(),
                actual_mxid: "mxid-123".to_owned(),
            })
        );
    }

    #[test]
    fn usb_transport_speed_parser_is_exact_and_policy_orders_minimum_below_maximum() {
        for (name, expected) in [
            ("LOW", UsbTransportSpeed::Low),
            ("FULL", UsbTransportSpeed::Full),
            ("HIGH", UsbTransportSpeed::High),
            ("SUPER", UsbTransportSpeed::Super),
            ("SUPER_PLUS", UsbTransportSpeed::SuperPlus),
        ] {
            let parsed = UsbTransportSpeed::parse(name).expect("known exact DepthAI name");
            assert_eq!(parsed, expected);
            assert_eq!(parsed.as_depthai_name(), name);
        }
        for invalid in ["", "high", "SUPER+", "UNKNOWN"] {
            assert!(UsbTransportSpeed::parse(invalid).is_err());
        }
        assert!(UsbTransportSpeed::Low < UsbTransportSpeed::Full);
        assert!(UsbTransportSpeed::Full < UsbTransportSpeed::High);
        assert!(UsbTransportSpeed::High < UsbTransportSpeed::Super);
        assert!(UsbTransportSpeed::Super < UsbTransportSpeed::SuperPlus);

        assert_eq!(
            UsbTransportPolicy::try_new(UsbTransportSpeed::High, UsbTransportSpeed::Super),
            Err(UsbTransportPolicyError::MinimumAboveMaximum {
                minimum: UsbTransportSpeed::Super,
                maximum: UsbTransportSpeed::High,
            })
        );
        assert_eq!(
            UsbTransportPolicy::high_speed_diagnostic(),
            UsbTransportPolicy::try_new(UsbTransportSpeed::High, UsbTransportSpeed::High)
                .expect("HIGH diagnostic policy")
        );
    }

    #[test]
    fn usb_transport_admission_rejects_unknown_and_below_minimum_readbacks() {
        let policy =
            UsbTransportPolicy::try_new(UsbTransportSpeed::SuperPlus, UsbTransportSpeed::Super)
                .expect("ordered transport policy");

        assert_eq!(
            admit_usb_transport(policy, ffi::UsbSpeed::Unknown),
            Err(UsbTransportAdmissionError::UnknownReadback)
        );
        assert_eq!(
            admit_usb_transport(policy, ffi::UsbSpeed::Unrecognized),
            Err(UsbTransportAdmissionError::UnrecognizedReadback)
        );
        assert_eq!(
            admit_usb_transport(policy, ffi::UsbSpeed::High),
            Err(UsbTransportAdmissionError::BelowMinimum {
                observed: UsbTransportSpeed::High,
                required_minimum: UsbTransportSpeed::Super,
                requested_maximum: UsbTransportSpeed::SuperPlus,
            })
        );

        let capped = UsbTransportPolicy::super_speed_required();
        assert_eq!(
            admit_usb_transport(capped, ffi::UsbSpeed::SuperPlus),
            Err(UsbTransportAdmissionError::AboveMaximum {
                observed: UsbTransportSpeed::SuperPlus,
                requested_maximum: UsbTransportSpeed::Super,
            })
        );

        let evidence = admit_usb_transport(policy, ffi::UsbSpeed::Super)
            .expect("minimum transport speed is admissible");
        assert_eq!(
            (
                evidence.requested_maximum(),
                evidence.required_minimum(),
                evidence.observed(),
            ),
            (
                UsbTransportSpeed::SuperPlus,
                UsbTransportSpeed::Super,
                UsbTransportSpeed::Super,
            )
        );
    }

    #[test]
    fn production_camera_constructors_require_super_speed() {
        assert_eq!(
            DeviceConfig::all_streams().usb_transport,
            UsbTransportPolicy::super_speed_required()
        );
        assert_eq!(
            DeviceConfig::rgb_only(640, 480, 30).usb_transport,
            UsbTransportPolicy::super_speed_required()
        );
    }

    #[cfg(oak_sys_check_only)]
    #[test]
    fn compile_only_mode_never_claims_native_build_provenance() {
        assert_eq!(
            depthai_build_metadata(),
            Err(DepthAiBuildMetadataError::CompileOnlyNativeDisabled)
        );
        assert!(matches!(
            Device::connect("mxid-123", base_config()),
            Err(ConnectionError::CompileOnlyNativeDisabled)
        ));
    }

    #[test]
    fn device_connect_rejects_empty_selector_before_native_work() {
        assert!(matches!(
            Device::connect("", base_config()),
            Err(ConnectionError::EmptySelector)
        ));
    }

    #[test]
    fn device_config_rejects_no_streams_and_zero_queue() {
        assert_eq!(
            DeviceConfig {
                usb_transport: UsbTransportPolicy::super_speed_required(),
                rgb: None,
                mono: None,
                depth: None,
                imu: None,
                queue: QueueConfig::default(),
            }
            .validate(),
            Err(DeviceConfigError::NoStreamsEnabled)
        );

        let mut config = base_config();
        config.queue.size = 0;
        assert_eq!(config.validate(), Err(DeviceConfigError::ZeroQueueSize));
    }

    #[test]
    fn device_config_rejects_every_zero_stream_field() {
        let rgb_cases = [
            ("width", DeviceConfig::rgb_only(0, 480, 30)),
            ("height", DeviceConfig::rgb_only(640, 0, 30)),
            ("fps", DeviceConfig::rgb_only(640, 480, 0)),
        ];
        for (field, config) in rgb_cases {
            assert_eq!(
                config.validate(),
                Err(DeviceConfigError::ZeroStreamField {
                    stream: "RGB",
                    field,
                })
            );
        }

        for (field, mono) in [
            (
                "width",
                MonoConfig {
                    width: 0,
                    ..mono_config()
                },
            ),
            (
                "height",
                MonoConfig {
                    height: 0,
                    ..mono_config()
                },
            ),
            (
                "fps",
                MonoConfig {
                    fps: 0,
                    ..mono_config()
                },
            ),
        ] {
            let mut config = base_config();
            config.mono = Some(mono);
            assert_eq!(
                config.validate(),
                Err(DeviceConfigError::ZeroStreamField {
                    stream: "mono",
                    field,
                })
            );
        }

        for (field, width, height, fps) in [
            ("width", 0, 480, 30),
            ("height", 640, 0, 30),
            ("fps", 640, 480, 0),
        ] {
            let config = DeviceConfig {
                usb_transport: UsbTransportPolicy::super_speed_required(),
                rgb: None,
                mono: None,
                depth: Some(DepthConfig {
                    width,
                    height,
                    fps,
                    alignment: DepthAlignment::RectifiedLeft,
                }),
                imu: None,
                queue: QueueConfig::default(),
            };
            assert_eq!(
                config.validate(),
                Err(DeviceConfigError::ZeroStreamField {
                    stream: "depth",
                    field,
                })
            );
        }

        let config = DeviceConfig {
            usb_transport: UsbTransportPolicy::super_speed_required(),
            rgb: None,
            mono: None,
            depth: None,
            imu: Some(ImuConfig { rate_hz: 0 }),
            queue: QueueConfig::default(),
        };
        assert_eq!(
            config.validate(),
            Err(DeviceConfigError::ZeroStreamField {
                stream: "IMU",
                field: "rate_hz",
            })
        );
    }

    #[test]
    fn device_config_rejects_conflicting_stereo_and_missing_rgb_alignment() {
        let mut config = base_config();
        config.depth = Some(DepthConfig {
            width: 320,
            height: 240,
            fps: 15,
            alignment: DepthAlignment::RectifiedLeft,
        });
        assert!(matches!(
            config.validate(),
            Err(DeviceConfigError::ConflictingStereoContracts { .. })
        ));

        let mut config = base_config();
        config.depth = Some(DepthConfig {
            width: 640,
            height: 480,
            fps: 30,
            alignment: DepthAlignment::Rgb,
        });
        assert_eq!(
            config.validate(),
            Err(DeviceConfigError::RgbAlignmentWithoutRgb)
        );
    }

    #[test]
    fn intrinsics_parser_accepts_canonical_matrix_and_exposes_exact_values() {
        let matrix = [[400.0, 0.0, 320.0], [0.0, 401.0, 240.0], [0.0, 0.0, 1.0]];
        let intrinsics =
            Intrinsics::try_from_projection_matrix(matrix, 640, 480).expect("canonical intrinsics");
        assert_eq!(intrinsics.projection_matrix(), matrix);
        assert_eq!(intrinsics.fx(), 400.0);
        assert_eq!(intrinsics.fy(), 401.0);
        assert_eq!(intrinsics.cx(), 320.0);
        assert_eq!(intrinsics.cy(), 240.0);
        assert_eq!((intrinsics.width(), intrinsics.height()), (640, 480));
    }

    #[test]
    fn intrinsics_parser_rejects_each_nonfinite_coefficient() {
        let canonical = [[400.0, 0.0, 1.0], [0.0, 401.0, 1.0], [0.0, 0.0, 1.0]];
        for row in 0..3 {
            for column in 0..3 {
                let mut matrix = canonical;
                matrix[row][column] = f32::NAN;
                assert!(matches!(
                    Intrinsics::try_from_projection_matrix(matrix, 2, 2),
                    Err(IntrinsicsError::NonFiniteCoefficient {
                        row: error_row,
                        column: error_column,
                        ..
                    }) if error_row == row && error_column == column
                ));
            }
        }
    }

    #[test]
    fn intrinsics_parser_rejects_invalid_shape_focal_and_projection_terms() {
        let canonical = [[400.0, 0.0, 1.0], [0.0, 401.0, 1.0], [0.0, 0.0, 1.0]];
        assert!(matches!(
            Intrinsics::try_from_projection_matrix(canonical, 0, 2),
            Err(IntrinsicsError::ZeroDimensions { .. })
        ));

        let mut invalid = canonical;
        invalid[0][0] = 0.0;
        assert!(matches!(
            Intrinsics::try_from_projection_matrix(invalid, 2, 2),
            Err(IntrinsicsError::NonPositiveFocalLength { .. })
        ));

        for (row, column) in [(0, 1), (1, 0), (2, 0), (2, 1), (2, 2)] {
            let mut invalid = canonical;
            invalid[row][column] = if (row, column) == (2, 2) { 2.0 } else { 1.0 };
            assert!(matches!(
                Intrinsics::try_from_projection_matrix(invalid, 2, 2),
                Err(IntrinsicsError::UnsupportedProjectionMatrix { .. })
            ));
        }
    }

    #[test]
    fn principal_point_scale_cannot_hide_noncanonical_projection_terms() {
        let mut matrix = [[1.0, 0.0, 1.0e20], [0.0, 1.0, -1.0e20], [0.0, 0.0, 1.0]];
        matrix[0][1] = 1.0e10;

        assert!(matches!(
            Intrinsics::try_from_projection_matrix(matrix, 2, 2),
            Err(IntrinsicsError::UnsupportedProjectionMatrix { .. })
        ));
    }

    #[test]
    fn parsed_depth_frame_requires_exact_shape_and_matching_intrinsics() {
        let mut unknown_status = raw_depth(1, 1, vec![1]);
        unknown_status.status = ffi::FrameStatus { repr: u8::MAX };
        assert!(matches!(
            parse_depth_result(unknown_status, 0),
            Err(DepthError::Corrupt)
        ));
        assert!(matches!(
            parse_depth_result(raw_depth(0, 2, vec![]), 0),
            Err(DepthError::Corrupt)
        ));
        assert!(matches!(
            parse_depth_result(raw_depth(2, 2, vec![1, 2, 3]), 0),
            Err(DepthError::Corrupt)
        ));

        let mut mismatched = raw_depth(2, 2, vec![1, 2, 3, 4]);
        mismatched.frame.intrinsics.width = 3;
        assert!(matches!(
            parse_depth_result(mismatched, 0),
            Err(DepthError::InvalidIntrinsics(
                IntrinsicsError::DimensionMismatch { .. }
            ))
        ));
    }

    #[test]
    fn parsed_depth_preserves_exact_millimetres_and_zero_is_only_invalid_value() {
        let parsed = parse_depth_result(raw_depth(2, 2, vec![0, 1, 10_000, u16::MAX]), 0)
            .expect("valid exact depth payload");
        assert_eq!(parsed.depth_mm(), [0, 1, 10_000, u16::MAX]);
        assert_eq!(parsed.depth_m_at(0, 0), None);
        assert_eq!(parsed.depth_m_at(1, 0), Some(0.001));
        assert_eq!(parsed.depth_m_at(0, 1), Some(10.0));
        assert_eq!(parsed.depth_m_at(2, 0), None);
        assert_eq!(parsed.depth_m_at(0, 2), None);
        let max_metres = parsed.depth_m_at(1, 1).expect("u16::MAX is valid");
        assert!((max_metres - f32::from(u16::MAX) * 0.001).abs() <= f32::EPSILON);
    }

    #[test]
    fn camera_parser_preserves_native_and_host_capture_provenance_separately() {
        let mut image = raw_image(StreamId::MonoLeft, 1, 1, 1, vec![7]);
        image.frame.device_capture_sequence = 9001;
        image.frame.host_delivery_sequence = 17;
        image.frame.timestamp.device_ns = 123;
        image.frame.exposure_time_us = 456;
        let image = parse_image_result(image, 0).expect("valid image provenance");
        assert_eq!(image.device_capture_sequence.as_i64(), 9001);
        assert_eq!(image.host_delivery_sequence.as_u64(), 17);
        assert_eq!(image.timestamp.as_nanos(), 123);
        assert_eq!(
            image.timestamp_reference,
            CameraTimestampReference::ExposureMidpoint
        );
        assert_eq!(
            image.exposure_time,
            ExposureTime::Known(ExposureDuration { micros: 456 })
        );

        let mut depth = raw_depth(1, 1, vec![9]);
        depth.frame.device_capture_sequence = 9002;
        depth.frame.host_delivery_sequence = 18;
        depth.frame.exposure_time_us = 0;
        let mut depth = parse_depth_result(depth, 0).expect("valid depth provenance");
        assert_eq!(depth.device_capture_sequence.as_u64(), 9002);
        assert_eq!(depth.host_delivery_sequence.as_u64(), 18);
        assert_eq!(depth.exposure_time, ExposureTime::Unavailable);
        assert_eq!(depth.connected_alignment(), None);
        depth.attach_connected_alignment(DepthAlignment::RectifiedLeft);
        assert_eq!(
            depth.connected_alignment(),
            Some(DepthAlignment::RectifiedLeft)
        );
    }

    #[test]
    fn invalid_camera_capture_metadata_fails_closed() {
        let mut image = raw_image(StreamId::MonoLeft, 1, 1, 1, vec![7]);
        image.frame.device_capture_sequence = -1;
        assert_eq!(
            parse_image_result(image, 0),
            Err(ImageError::InvalidDeviceSequence(
                FrameSequenceError::Negative { value: -1 }
            ))
        );

        let mut depth = raw_depth(1, 1, vec![9]);
        depth.frame.timestamp.device_ns = -1;
        assert_eq!(
            parse_depth_result(depth, 0),
            Err(DepthError::InvalidTimestamp(TimestampError::Negative {
                value_ns: -1
            }))
        );

        let mut depth = raw_depth(1, 1, vec![9]);
        depth.frame.exposure_time_us = -1;
        assert_eq!(
            parse_depth_result(depth, 0),
            Err(DepthError::InvalidExposureTime(
                ExposureTimeError::Negative { value_us: -1 }
            ))
        );
    }

    #[test]
    fn parsed_images_require_exact_stream_stride_payload_and_intrinsics() {
        assert!(matches!(
            parse_image_result(raw_image(StreamId::MonoLeft, 0, 2, 0, vec![]), 0),
            Err(ImageError::Corrupt)
        ));
        assert!(matches!(
            parse_image_result(raw_image(StreamId::Rgb, u32::MAX, 1, 0, vec![]), 0),
            Err(ImageError::Corrupt)
        ));
        assert!(matches!(
            parse_image_result(raw_image(StreamId::MonoLeft, 2, 2, 2, vec![1, 2, 3]), 0),
            Err(ImageError::Corrupt)
        ));
        assert!(matches!(
            parse_image_result(
                raw_image(StreamId::MonoLeft, 2, 2, 3, vec![1, 2, 3, 4, 5, 6]),
                0
            ),
            Err(ImageError::Corrupt)
        ));
        assert!(matches!(
            parse_image_result(raw_image(StreamId::Depth, 2, 2, 2, vec![1, 2, 3, 4]), 0),
            Err(ImageError::Corrupt)
        ));

        let mut mismatched = raw_image(StreamId::MonoLeft, 2, 2, 2, vec![1, 2, 3, 4]);
        mismatched.frame.intrinsics.height = 3;
        assert!(matches!(
            parse_image_result(mismatched, 0),
            Err(ImageError::InvalidIntrinsics(
                IntrinsicsError::DimensionMismatch { .. }
            ))
        ));

        let frame = parse_image_result(raw_image(StreamId::MonoLeft, 2, 2, 2, vec![1, 2, 3, 4]), 0)
            .expect("valid exact mono payload");
        assert_eq!(frame.pixels(), [1, 2, 3, 4]);
        assert_eq!(
            (frame.intrinsics().width(), frame.intrinsics().height()),
            (2, 2)
        );

        let rgb = parse_image_result(raw_image(StreamId::Rgb, 2, 1, 6, vec![0; 6]), 0)
            .expect("valid exact BGR payload");
        assert_eq!(rgb.pixels().len(), 6);
    }

    #[test]
    fn frame_status_errors_preserve_timeout_and_stream_context() {
        let mut timeout = raw_image(StreamId::Rgb, 1, 1, 3, vec![0; 3]);
        timeout.status = ffi::FrameStatus::Timeout;
        assert_eq!(
            parse_image_result(timeout, 123),
            Err(ImageError::Timeout { timeout_ms: 123 })
        );

        let mut disabled = raw_image(StreamId::MonoRight, 1, 1, 1, vec![0]);
        disabled.status = ffi::FrameStatus::StreamNotEnabled;
        assert_eq!(
            parse_image_result(disabled, 0),
            Err(ImageError::StreamNotEnabled {
                stream: StreamId::MonoRight,
            })
        );

        let mut unknown = raw_image(StreamId::Rgb, 1, 1, 3, vec![0; 3]);
        unknown.status = ffi::FrameStatus { repr: u8::MAX };
        assert!(matches!(
            parse_image_result(unknown, 0),
            Err(ImageError::Corrupt)
        ));
    }

    fn raw_imu_sample() -> ffi::ImuSample {
        ffi::ImuSample {
            accel_timestamp: ffi::Timestamp { device_ns: 10 },
            gyro_timestamp: ffi::Timestamp { device_ns: 12 },
            sequence: 11,
            accel_x: 1.0,
            accel_y: 2.0,
            accel_z: 3.0,
            accel_accuracy: ImuAccuracy::High,
            gyro_x: 4.0,
            gyro_y: 5.0,
            gyro_z: 6.0,
            gyro_accuracy: ImuAccuracy::Medium,
        }
    }

    #[test]
    fn imu_parser_rejects_empty_ok_batch_and_nonfinite_components() {
        let corrupt = ffi::ImuBatchResult {
            status: ffi::ImuStatus::Corrupt,
            batch: ffi::ImuBatch { samples: vec![] },
        };
        assert_eq!(parse_imu_result(corrupt), Err(ImuError::Corrupt));

        let unknown_status = ffi::ImuBatchResult {
            status: ffi::ImuStatus { repr: u8::MAX },
            batch: ffi::ImuBatch { samples: vec![] },
        };
        assert_eq!(parse_imu_result(unknown_status), Err(ImuError::Corrupt));

        let empty = ffi::ImuBatchResult {
            status: ffi::ImuStatus::Ok,
            batch: ffi::ImuBatch { samples: vec![] },
        };
        assert_eq!(parse_imu_result(empty), Err(ImuError::Corrupt));

        let mut unknown_accuracy = raw_imu_sample();
        unknown_accuracy.gyro_accuracy = ffi::ImuAccuracy { repr: u8::MAX };
        let unknown_accuracy = ffi::ImuBatchResult {
            status: ffi::ImuStatus::Ok,
            batch: ffi::ImuBatch {
                samples: vec![unknown_accuracy],
            },
        };
        assert_eq!(parse_imu_result(unknown_accuracy), Err(ImuError::Corrupt));

        for (field, mutate) in [
            ("accel_x", 0),
            ("accel_y", 1),
            ("accel_z", 2),
            ("gyro_x", 3),
            ("gyro_y", 4),
            ("gyro_z", 5),
        ] {
            let mut sample = raw_imu_sample();
            match mutate {
                0 => sample.accel_x = f32::NAN,
                1 => sample.accel_y = f32::NAN,
                2 => sample.accel_z = f32::NAN,
                3 => sample.gyro_x = f32::NAN,
                4 => sample.gyro_y = f32::NAN,
                5 => sample.gyro_z = f32::NAN,
                _ => unreachable!(),
            }
            let result = ffi::ImuBatchResult {
                status: ffi::ImuStatus::Ok,
                batch: ffi::ImuBatch {
                    samples: vec![sample],
                },
            };
            assert_eq!(
                parse_imu_result(result),
                Err(ImuError::NonFiniteSample {
                    sample_index: 0,
                    field,
                })
            );
        }
    }

    #[test]
    fn imu_parser_constructs_finite_domain_sample() {
        let result = ffi::ImuBatchResult {
            status: ffi::ImuStatus::Ok,
            batch: ffi::ImuBatch {
                samples: vec![raw_imu_sample()],
            },
        };
        let samples = parse_imu_result(result).expect("finite IMU sample");
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].sequence, 11);
        assert_eq!(samples[0].accel.as_array(), [1.0, 2.0, 3.0]);
        assert_eq!(samples[0].gyro.as_array(), [4.0, 5.0, 6.0]);
    }

    #[test]
    fn baseline_parser_rejects_nonpositive_and_nonfinite_metres() {
        for value in [0.0, -0.1, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                parse_stereo_baseline_m(value),
                Err(CalibrationError::InvalidBaseline { .. })
            ));
        }
        assert_eq!(parse_stereo_baseline_m(0.075), Ok(0.075));
    }

    #[test]
    fn eeprom_calibration_parser_preserves_documented_direction_and_raw_output() {
        let raw = raw_eeprom_calibration_evidence();
        let expected_imu = raw.imu_to_camera_b_m.clone();
        let expected_rectification = raw.stereo_left_rectification_rotation_raw.clone();
        let parsed = parse_eeprom_calibration_evidence(raw).expect("valid EEPROM calibration");

        assert_eq!(parsed.stereo_left_camera_socket(), OakCameraSocket::CameraB);
        assert_eq!(
            parsed.stereo_right_camera_socket(),
            OakCameraSocket::CameraC
        );
        assert_eq!(
            parsed.imu_to_camera_b_m().concat(),
            expected_imu,
            "the parser must not invert or compose the EEPROM transform"
        );
        assert_eq!(
            parsed.stereo_left_rectification_rotation_raw().concat(),
            expected_rectification,
            "the parser must not alter or direction-label the raw EEPROM rectification output"
        );
    }

    #[test]
    fn eeprom_calibration_parser_rejects_wrong_shapes_before_coefficients() {
        let mut raw = raw_eeprom_calibration_evidence();
        raw.imu_to_camera_b_m.pop();
        raw.stereo_left_rectification_rotation_raw[0] = f32::NAN;

        assert_eq!(
            parse_eeprom_calibration_evidence(raw),
            Err(CalibrationError::InvalidMatrixLength {
                matrix: EepromCalibrationMatrix::ImuToCameraBMeters,
                expected: 16,
                actual: 15,
            })
        );
    }

    #[test]
    fn eeprom_calibration_parser_identifies_nonfinite_matrix_coefficient() {
        let mut raw = raw_eeprom_calibration_evidence();
        raw.stereo_left_rectification_rotation_raw[5] = f32::INFINITY;

        assert!(matches!(
            parse_eeprom_calibration_evidence(raw),
            Err(CalibrationError::NonFiniteMatrixCoefficient {
                matrix: EepromCalibrationMatrix::StereoLeftRectificationRotationRaw,
                row: 1,
                column: 2,
                value,
            }) if value == f32::INFINITY
        ));
    }

    #[test]
    fn eeprom_calibration_parser_requires_exact_stereo_socket_roles() {
        for (side, left, right, expected, actual) in [
            (
                EepromStereoSide::Left,
                ffi::CameraSocket::CameraA,
                ffi::CameraSocket::CameraC,
                OakCameraSocket::CameraB,
                OakCameraSocket::CameraA,
            ),
            (
                EepromStereoSide::Right,
                ffi::CameraSocket::CameraB,
                ffi::CameraSocket::CameraB,
                OakCameraSocket::CameraC,
                OakCameraSocket::CameraB,
            ),
        ] {
            let mut raw = raw_eeprom_calibration_evidence();
            raw.stereo_left_camera_socket = left;
            raw.stereo_right_camera_socket = right;
            assert_eq!(
                parse_eeprom_calibration_evidence(raw),
                Err(CalibrationError::UnexpectedStereoCameraSocket {
                    side,
                    expected,
                    actual,
                })
            );
        }
    }

    #[test]
    fn vector_magnitude_avoids_intermediate_square_overflow() {
        assert_eq!(
            Vec3 {
                x: 3.0,
                y: 4.0,
                z: 12.0,
            }
            .magnitude(),
            13.0
        );
        let component = f32::MAX / 4.0;
        assert!(Vec3 {
            x: component,
            y: component,
            z: component,
        }
        .magnitude()
        .is_finite());
    }

    #[cfg(feature = "opencv-face-detector")]
    fn valid_haar_detector_config(
        maximum_retained_detections: u32,
    ) -> OpenCvHaarFaceDetectorConfig {
        OpenCvHaarFaceDetectorConfig::try_new(1.15, 6, 4, 30, 30, maximum_retained_detections)
            .expect("valid Haar detector config")
    }

    #[cfg(feature = "opencv-face-detector")]
    fn detector_rgb_frame(width: u32, height: u32) -> ImageFrame {
        let stride = width.checked_mul(3).expect("test RGB stride");
        let length = usize::try_from(stride)
            .expect("test stride fits usize")
            .checked_mul(usize::try_from(height).expect("test height fits usize"))
            .expect("test RGB payload length");
        parse_image_result(
            raw_image(StreamId::Rgb, width, height, stride, vec![0; length]),
            0,
        )
        .expect("valid detector RGB frame")
    }

    #[cfg(feature = "opencv-face-detector")]
    fn raw_haar_detection(
        x: i64,
        y: i64,
        width: i64,
        height: i64,
        detector_level_weight: f64,
        source: face_ffi::RawHaarFaceDetectionSource,
    ) -> face_ffi::RawHaarFaceDetection {
        face_ffi::RawHaarFaceDetection {
            x,
            y,
            width,
            height,
            detector_level_weight,
            source,
        }
    }

    #[cfg(feature = "opencv-face-detector")]
    #[test]
    fn haar_detector_config_rejects_every_weak_search_parameter() {
        let valid = valid_haar_detector_config(8);
        assert_eq!(valid.scale_factor(), 1.15);
        assert_eq!(valid.frontal_minimum_neighbors(), 6);
        assert_eq!(valid.profile_minimum_neighbors(), 4);
        assert_eq!(
            (valid.minimum_face_width(), valid.minimum_face_height()),
            (30, 30)
        );
        assert_eq!(valid.maximum_retained_detections(), 8);

        let build = |scale_factor: f64,
                     frontal_neighbors: u32,
                     profile_neighbors: u32,
                     width: u32,
                     height: u32,
                     maximum: u32| {
            OpenCvHaarFaceDetectorConfig::try_new(
                scale_factor,
                frontal_neighbors,
                profile_neighbors,
                width,
                height,
                maximum,
            )
        };
        for scale_factor in [1.0, 0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                build(scale_factor, 6, 4, 30, 30, 1),
                Err(OpenCvHaarFaceDetectorConfigError::InvalidScaleFactor { .. })
            ));
        }
        for (frontal_neighbors, profile_neighbors, expected_field) in [
            (0, 4, "frontal_minimum_neighbors"),
            (6, 0, "profile_minimum_neighbors"),
        ] {
            assert!(matches!(
                build(1.1, frontal_neighbors, profile_neighbors, 30, 30, 1),
                Err(OpenCvHaarFaceDetectorConfigError::ZeroMinimumNeighbors {
                    field
                }) if field == expected_field
            ));
        }
        for (width, height, expected_field) in [
            (0, 30, "minimum_face_width"),
            (30, 0, "minimum_face_height"),
        ] {
            assert!(matches!(
                build(1.1, 6, 4, width, height, 1),
                Err(
                    OpenCvHaarFaceDetectorConfigError::ZeroMinimumFaceDimension {
                        field
                    }
                ) if field == expected_field
            ));
        }
        for (frontal_neighbors, profile_neighbors, width, height, expected_field) in [
            (i32::MAX as u32 + 1, 4, 30, 30, "frontal_minimum_neighbors"),
            (6, i32::MAX as u32 + 1, 30, 30, "profile_minimum_neighbors"),
            (6, 4, i32::MAX as u32 + 1, 30, "minimum_face_width"),
            (6, 4, 30, i32::MAX as u32 + 1, "minimum_face_height"),
        ] {
            assert!(matches!(
                build(
                    1.1,
                    frontal_neighbors,
                    profile_neighbors,
                    width,
                    height,
                    1,
                ),
                Err(
                    OpenCvHaarFaceDetectorConfigError::ParameterExceedsOpenCvInt {
                        field,
                        ..
                    }
                ) if field == expected_field
            ));
        }
        assert!(matches!(
            build(1.1, 6, 4, 30, 30, 0),
            Err(OpenCvHaarFaceDetectorConfigError::ZeroMaximumRetainedDetections)
        ));
        assert_eq!(
            build(1.1, 6, 4, 30, 30, MAXIMUM_RETAINED_HAAR_FACE_DETECTIONS + 1,),
            Err(
                OpenCvHaarFaceDetectorConfigError::MaximumRetainedDetectionsExceedsLimit {
                    value: MAXIMUM_RETAINED_HAAR_FACE_DETECTIONS + 1,
                    limit: MAXIMUM_RETAINED_HAAR_FACE_DETECTIONS,
                }
            )
        );
    }

    #[cfg(feature = "opencv-face-detector")]
    #[test]
    fn haar_cascade_xml_boundary_rejects_empty_nul_and_oversize_payloads() {
        assert_eq!(
            validate_cascade_xml("frontal", b""),
            Err(OpenCvHaarFaceDetectorLoadError::EmptyCascadeXml { cascade: "frontal" })
        );
        assert_eq!(
            validate_cascade_xml("profile", b"<xml>\0</xml>"),
            Err(OpenCvHaarFaceDetectorLoadError::CascadeXmlContainsNul { cascade: "profile" })
        );

        let at_limit = vec![b'x'; MAXIMUM_OPENCV_HAAR_CASCADE_XML_BYTES];
        assert_eq!(validate_cascade_xml("frontal", &at_limit), Ok(()));

        let above_limit = vec![b'x'; MAXIMUM_OPENCV_HAAR_CASCADE_XML_BYTES + 1];
        assert_eq!(
            validate_cascade_xml("profile", &above_limit),
            Err(
                OpenCvHaarFaceDetectorLoadError::CascadeXmlExceedsByteLimit {
                    cascade: "profile",
                    actual_bytes: MAXIMUM_OPENCV_HAAR_CASCADE_XML_BYTES + 1,
                    maximum_bytes: MAXIMUM_OPENCV_HAAR_CASCADE_XML_BYTES,
                }
            )
        );
    }

    #[cfg(feature = "opencv-face-detector")]
    #[test]
    fn raw_haar_results_are_fully_validated_sorted_capped_and_frame_stamped() {
        let mut frame = detector_rgb_frame(100, 80);
        frame.device_capture_sequence =
            DeviceFrameSequence::try_from_i64(9001).expect("test sequence");
        frame.host_delivery_sequence = FrameDeliverySequence::new(42);
        frame.timestamp = Timestamp::try_from_nanos(123_456).expect("test timestamp");
        let raw = face_ffi::RawHaarFaceDetectionBatch {
            detections: vec![
                raw_haar_detection(
                    20,
                    20,
                    10,
                    12,
                    -2.0,
                    face_ffi::RawHaarFaceDetectionSource::Frontal,
                ),
                raw_haar_detection(
                    10,
                    10,
                    10,
                    10,
                    4.0,
                    face_ffi::RawHaarFaceDetectionSource::Frontal,
                ),
                raw_haar_detection(
                    5,
                    5,
                    20,
                    10,
                    4.0,
                    face_ffi::RawHaarFaceDetectionSource::Frontal,
                ),
                raw_haar_detection(
                    1,
                    1,
                    5,
                    5,
                    8.0,
                    face_ffi::RawHaarFaceDetectionSource::Frontal,
                ),
            ],
        };
        let batch = parse_raw_haar_face_detection_batch(raw, &frame, 3).expect("valid raw batch");
        assert_eq!(batch.stream(), StreamId::Rgb);
        assert_eq!(batch.device_capture_sequence().as_i64(), 9001);
        assert_eq!(batch.host_delivery_sequence().as_u64(), 42);
        assert_eq!(batch.timestamp().as_nanos(), 123_456);
        assert_eq!(
            batch.timestamp_reference(),
            CameraTimestampReference::ExposureMidpoint
        );
        assert_eq!((batch.width(), batch.height()), (100, 80));
        assert_eq!(batch.native_detection_count(), 4);
        assert!(batch.was_truncated());
        assert_eq!(batch.detections().len(), 3);

        let ordered = batch
            .detections()
            .iter()
            .map(|detection| {
                (
                    detection.bounds().width(),
                    detection.bounds().height(),
                    detection.detector_level_weight(),
                    detection.source(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            ordered,
            [
                (20, 10, 4.0, HaarFaceDetectionSource::Frontal),
                (10, 12, -2.0, HaarFaceDetectionSource::Frontal),
                (10, 10, 4.0, HaarFaceDetectionSource::Frontal),
            ]
        );
    }

    #[cfg(feature = "opencv-face-detector")]
    #[test]
    fn widest_negative_level_weight_survives_retained_cap() {
        let frame = detector_rgb_frame(100, 80);
        let raw = face_ffi::RawHaarFaceDetectionBatch {
            detections: vec![
                raw_haar_detection(
                    5,
                    5,
                    30,
                    10,
                    -100.0,
                    face_ffi::RawHaarFaceDetectionSource::Frontal,
                ),
                raw_haar_detection(
                    10,
                    10,
                    29,
                    60,
                    1_000_000.0,
                    face_ffi::RawHaarFaceDetectionSource::Frontal,
                ),
            ],
        };

        let batch = parse_raw_haar_face_detection_batch(raw, &frame, 1)
            .expect("finite negative level weights are valid");
        assert!(batch.was_truncated());
        assert_eq!(batch.detections().len(), 1);
        assert_eq!(batch.detections()[0].bounds().width(), 30);
        assert_eq!(batch.detections()[0].detector_level_weight(), -100.0);
    }

    #[cfg(feature = "opencv-face-detector")]
    #[test]
    fn raw_haar_parser_validates_candidates_beyond_the_retained_cap() {
        let frame = detector_rgb_frame(40, 30);
        let valid = || {
            raw_haar_detection(
                1,
                1,
                10,
                10,
                1.0,
                face_ffi::RawHaarFaceDetectionSource::Frontal,
            )
        };
        let raw = face_ffi::RawHaarFaceDetectionBatch {
            detections: vec![
                valid(),
                valid(),
                raw_haar_detection(
                    1,
                    1,
                    10,
                    10,
                    f64::NAN,
                    face_ffi::RawHaarFaceDetectionSource::Frontal,
                ),
            ],
        };
        assert!(matches!(
            parse_raw_haar_face_detection_batch(raw, &frame, 1),
            Err(HaarFaceDetectionError::NonFiniteDetectorLevelWeight {
                detection_index: 2,
                ..
            })
        ));
    }

    #[cfg(feature = "opencv-face-detector")]
    #[test]
    fn raw_haar_parser_rejects_results_above_the_absolute_native_limit() {
        let frame = detector_rgb_frame(40, 30);
        let detection = raw_haar_detection(
            1,
            1,
            10,
            10,
            1.0,
            face_ffi::RawHaarFaceDetectionSource::Frontal,
        );
        let count = MAXIMUM_NATIVE_HAAR_FACE_DETECTIONS as usize + 1;
        let raw = face_ffi::RawHaarFaceDetectionBatch {
            detections: vec![detection; count],
        };
        assert_eq!(
            parse_raw_haar_face_detection_batch(raw, &frame, 1),
            Err(HaarFaceDetectionError::NativeDetectionCountExceedsLimit {
                count,
                limit: MAXIMUM_NATIVE_HAAR_FACE_DETECTIONS,
            })
        );
    }

    #[cfg(feature = "opencv-face-detector")]
    #[test]
    fn raw_haar_parser_rejects_nonfinite_weights_unknown_sources_and_bad_rectangles() {
        let frame = detector_rgb_frame(40, 30);
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let raw = face_ffi::RawHaarFaceDetectionBatch {
                detections: vec![raw_haar_detection(
                    1,
                    1,
                    10,
                    10,
                    value,
                    face_ffi::RawHaarFaceDetectionSource::Frontal,
                )],
            };
            assert!(matches!(
                parse_raw_haar_face_detection_batch(raw, &frame, 1),
                Err(HaarFaceDetectionError::NonFiniteDetectorLevelWeight {
                    detection_index: 0,
                    ..
                })
            ));
        }

        let raw = face_ffi::RawHaarFaceDetectionBatch {
            detections: vec![raw_haar_detection(
                1,
                1,
                10,
                10,
                1.0,
                face_ffi::RawHaarFaceDetectionSource { repr: u8::MAX },
            )],
        };
        assert_eq!(
            parse_raw_haar_face_detection_batch(raw, &frame, 1),
            Err(HaarFaceDetectionError::InvalidSource { detection_index: 0 })
        );

        for (raw_source, expected_source) in [
            (
                face_ffi::RawHaarFaceDetectionSource::Frontal,
                HaarFaceDetectionSource::Frontal,
            ),
            (
                face_ffi::RawHaarFaceDetectionSource::Profile,
                HaarFaceDetectionSource::Profile,
            ),
            (
                face_ffi::RawHaarFaceDetectionSource::MirroredProfile,
                HaarFaceDetectionSource::MirroredProfile,
            ),
        ] {
            let raw = face_ffi::RawHaarFaceDetectionBatch {
                detections: vec![raw_haar_detection(1, 1, 10, 10, 1.0, raw_source)],
            };
            let parsed =
                parse_raw_haar_face_detection_batch(raw, &frame, 1).expect("known source kind");
            assert_eq!(parsed.detections()[0].source(), expected_source);
        }

        let raw = face_ffi::RawHaarFaceDetectionBatch {
            detections: vec![
                raw_haar_detection(
                    1,
                    1,
                    10,
                    10,
                    1.0,
                    face_ffi::RawHaarFaceDetectionSource::Frontal,
                ),
                raw_haar_detection(
                    2,
                    2,
                    10,
                    10,
                    1.0,
                    face_ffi::RawHaarFaceDetectionSource::Profile,
                ),
            ],
        };
        assert_eq!(
            parse_raw_haar_face_detection_batch(raw, &frame, 2),
            Err(HaarFaceDetectionError::InconsistentSource {
                detection_index: 1,
                expected: HaarFaceDetectionSource::Frontal,
                actual: HaarFaceDetectionSource::Profile,
            })
        );

        for (x, y, width, height) in [
            (-1, 0, 1, 1),
            (0, -1, 1, 1),
            (0, 0, 0, 1),
            (0, 0, 1, 0),
            (39, 0, 2, 1),
            (0, 29, 1, 2),
            (i64::MAX, 0, 1, 1),
            (0, i64::MAX, 1, 1),
        ] {
            let raw = face_ffi::RawHaarFaceDetectionBatch {
                detections: vec![raw_haar_detection(
                    x,
                    y,
                    width,
                    height,
                    -100.0,
                    face_ffi::RawHaarFaceDetectionSource::Profile,
                )],
            };
            assert!(matches!(
                parse_raw_haar_face_detection_batch(raw, &frame, 1),
                Err(HaarFaceDetectionError::InvalidRectangle {
                    detection_index: 0,
                    ..
                })
            ));
        }
    }

    #[cfg(feature = "opencv-face-detector")]
    #[test]
    fn haar_detector_accepts_only_exact_tightly_packed_rgb_frames() {
        let mono = parse_image_result(raw_image(StreamId::MonoLeft, 2, 2, 2, vec![0; 4]), 0)
            .expect("valid mono frame");
        assert_eq!(
            validate_haar_detector_frame(&mono),
            Err(HaarFaceDetectionError::NonRgbFrame {
                stream: StreamId::MonoLeft,
            })
        );

        let mut rgb = detector_rgb_frame(2, 2);
        rgb.stride_bytes = 7;
        assert!(matches!(
            validate_haar_detector_frame(&rgb),
            Err(HaarFaceDetectionError::InvalidBgrFrameLayout { .. })
        ));
        rgb.stride_bytes = 6;
        rgb.data.pop();
        assert!(matches!(
            validate_haar_detector_frame(&rgb),
            Err(HaarFaceDetectionError::InvalidBgrFrameLayout { .. })
        ));

        let mut oversized = detector_rgb_frame(1, 1);
        oversized.width = MAXIMUM_HAAR_FACE_FRAME_DIMENSION + 1;
        assert_eq!(
            validate_haar_detector_frame(&oversized),
            Err(HaarFaceDetectionError::FrameDimensionsExceedDetectorLimit {
                width: MAXIMUM_HAAR_FACE_FRAME_DIMENSION + 1,
                height: 1,
                maximum_dimension: MAXIMUM_HAAR_FACE_FRAME_DIMENSION,
                maximum_pixels: MAXIMUM_HAAR_FACE_FRAME_PIXELS,
            })
        );

        oversized.width = MAXIMUM_HAAR_FACE_FRAME_DIMENSION;
        oversized.height = MAXIMUM_HAAR_FACE_FRAME_PIXELS
            .div_ceil(u64::from(MAXIMUM_HAAR_FACE_FRAME_DIMENSION))
            as u32
            + 1;
        assert!(matches!(
            validate_haar_detector_frame(&oversized),
            Err(HaarFaceDetectionError::FrameDimensionsExceedDetectorLimit { .. })
        ));

        assert_eq!(
            validate_haar_detector_frame(&detector_rgb_frame(640, 400)),
            Ok(())
        );
    }

    #[cfg(all(feature = "opencv-face-detector", not(oak_sys_check_only)))]
    #[test]
    #[ignore = "requires native OpenCV and explicit OAK_SYS_TEST_*_CASCADE_XML paths"]
    fn native_haar_detector_loads_cascades_and_detects_a_blank_frame() {
        let read_cascade = |variable: &'static str| {
            let path = std::env::var_os(variable)
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| panic!("{variable} must name an exact cascade XML file"));
            std::fs::read(&path).unwrap_or_else(|error| {
                panic!(
                    "{variable} cascade {} could not be read: {error}",
                    path.display()
                )
            })
        };
        let frontal = read_cascade("OAK_SYS_TEST_FRONTAL_CASCADE_XML");
        let profile = read_cascade("OAK_SYS_TEST_PROFILE_CASCADE_XML");
        let mut detector =
            OpenCvHaarFaceDetector::load(&frontal, &profile, valid_haar_detector_config(8))
                .expect("load native Haar cascades from exact retained bytes");

        let frame = detector_rgb_frame(160, 120);
        let batch = detector
            .detect(&frame)
            .expect("run native Haar detector on a valid blank BGR frame");
        assert_eq!(batch.native_detection_count(), 0);
        assert!(batch.detections().is_empty());
        assert!(!batch.was_truncated());
        assert_eq!((batch.width(), batch.height()), (160, 120));
    }

    #[cfg(all(feature = "opencv-face-detector", oak_sys_check_only))]
    #[test]
    fn compile_only_feature_requires_no_opencv_runtime_or_cascade_files() {
        assert!(matches!(
            OpenCvHaarFaceDetector::load(
                b"<opencv_storage><cascade/></opencv_storage>",
                b"<opencv_storage><cascade/></opencv_storage>",
                valid_haar_detector_config(8),
            ),
            Err(OpenCvHaarFaceDetectorLoadError::CompileOnlyNativeDisabled)
        ));
    }
}
