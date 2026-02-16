//! oak-sys: FFI bindings to OAK-D camera via DepthAI
//!
//! This crate follows "parse, don't validate" - invalid states are unrepresentable.

use thiserror::Error;

// ============================================================================
// FFI MODULE - Raw C++ interface (not re-exported publicly)
// ============================================================================

#[cxx::bridge(namespace = "kiko::oak")]
mod ffi {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum DeviceState {
        Available,
        InUse,
        Bootloader,
        Disconnected,
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
        QueueOverflow,
        Disconnected,
        Corrupt,
        StreamNotEnabled,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ImuStatus {
        Ok,
        Empty,
        Overflow,
        Disconnected,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ImuAccuracy {
        Unreliable,
        Low,
        Medium,
        High,
    }

    #[derive(Debug, Clone)]
    pub struct DeviceConfig {
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
        pub depth_align_to_rgb: bool,

        pub imu_enabled: bool,
        pub imu_rate_hz: u32,

        pub queue_size: u32,
        pub queue_blocking: bool,
    }

    #[derive(Debug, Clone, Copy, Default)]
    pub struct Timestamp {
        pub device_ns: i64,
    }

    #[derive(Debug, Clone)]
    pub struct ImageFrame {
        pub stream: StreamId,
        pub sequence: u64,
        pub timestamp: Timestamp,
        pub width: u32,
        pub height: u32,
        pub stride_bytes: u32,
        pub data: Vec<u8>,
    }

    #[derive(Debug, Clone)]
    pub struct DepthFrame {
        pub sequence: u64,
        pub timestamp: Timestamp,
        pub width: u32,
        pub height: u32,
        pub depth_scale: f32,
        pub min_depth_mm: u16,
        pub max_depth_mm: u16,
        pub data: Vec<u16>,
    }

    #[derive(Debug, Clone, Copy)]
    pub struct ImuSample {
        pub timestamp: Timestamp,
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
        pub dropped_count: u32,
    }

    #[derive(Debug, Clone, Copy)]
    pub struct Intrinsics {
        pub fx: f32,
        pub fy: f32,
        pub cx: f32,
        pub cy: f32,
        pub width: u32,
        pub height: u32,
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

    unsafe extern "C++" {
        include!("oak_device.hpp");

        type OakDevice;

        fn list_devices() -> Vec<DeviceInfo>;
        fn create_device(selector: &str, config: &DeviceConfig) -> UniquePtr<OakDevice>;
        fn is_connected(self: &OakDevice) -> bool;
        fn try_get_rgb(self: Pin<&mut OakDevice>, timeout_ms: u32) -> ImageFrameResult;
        fn try_get_mono_left(self: Pin<&mut OakDevice>, timeout_ms: u32) -> ImageFrameResult;
        fn try_get_mono_right(self: Pin<&mut OakDevice>, timeout_ms: u32) -> ImageFrameResult;
        fn try_get_depth(self: Pin<&mut OakDevice>, timeout_ms: u32) -> DepthFrameResult;
        fn get_imu_batch(self: Pin<&mut OakDevice>) -> ImuBatchResult;
        fn get_rgb_intrinsics(self: &OakDevice) -> Intrinsics;
        fn get_left_intrinsics(self: &OakDevice) -> Intrinsics;
        fn get_right_intrinsics(self: &OakDevice) -> Intrinsics;
        fn get_stereo_baseline_m(self: &OakDevice) -> f32;
        fn close(self: Pin<&mut OakDevice>);
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
    pub fn from_nanos(ns: i64) -> Self {
        Self(ns)
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

impl From<Timestamp> for std::time::Duration {
    fn from(ts: Timestamp) -> Self {
        std::time::Duration::from_nanos(ts.0 as u64)
    }
}

impl From<ffi::Timestamp> for Timestamp {
    fn from(ts: ffi::Timestamp) -> Self {
        Self(ts.device_ns)
    }
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
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }

    pub fn as_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
}

// ============================================================================
// CONFIGURATION - Use Option<T> to make invalid states unrepresentable
// ============================================================================

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
    pub align_to_rgb: bool,
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

/// Device configuration. Use Option<T> so disabled streams have no config.
#[derive(Debug, Clone)]
pub struct DeviceConfig {
    pub rgb: Option<RgbConfig>,
    pub mono: Option<MonoConfig>,
    pub depth: Option<DepthConfig>,
    pub imu: Option<ImuConfig>,
    pub queue: QueueConfig,
}

impl DeviceConfig {
    /// All streams enabled at 640x480@30fps
    pub fn all_streams() -> Self {
        Self {
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
                align_to_rgb: true,
            }),
            imu: Some(ImuConfig { rate_hz: 400 }),
            queue: QueueConfig::default(),
        }
    }

    /// Only RGB stream
    pub fn rgb_only(width: u32, height: u32, fps: u32) -> Self {
        Self {
            rgb: Some(RgbConfig { width, height, fps }),
            mono: None,
            depth: None,
            imu: None,
            queue: QueueConfig::default(),
        }
    }

    fn to_ffi(&self) -> ffi::DeviceConfig {
        ffi::DeviceConfig {
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
            depth_align_to_rgb: self.depth.map(|c| c.align_to_rgb).unwrap_or(false),

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
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ImageError {
    #[error("frame acquisition timed out after {timeout_ms}ms")]
    Timeout { timeout_ms: u32 },

    #[error("frame queue is empty (non-blocking mode)")]
    QueueEmpty,

    #[error("frame queue overflowed, frames were dropped")]
    QueueOverflow,

    #[error("device disconnected during frame acquisition")]
    Disconnected,

    #[error("received corrupt frame data")]
    Corrupt,

    #[error("attempted to get {stream:?} frame but stream is not enabled in config")]
    StreamNotEnabled { stream: StreamId },
}

/// Errors when acquiring depth frames
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum DepthError {
    #[error("depth frame acquisition timed out after {timeout_ms}ms")]
    Timeout { timeout_ms: u32 },

    #[error("depth queue is empty (non-blocking mode)")]
    QueueEmpty,

    #[error("depth queue overflowed, frames were dropped")]
    QueueOverflow,

    #[error("device disconnected during depth acquisition")]
    Disconnected,

    #[error("received corrupt depth data")]
    Corrupt,

    #[error("depth stream is not enabled in config")]
    StreamNotEnabled,
}

/// Errors when acquiring IMU data
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ImuError {
    #[error("no IMU samples available")]
    Empty,

    #[error("IMU queue overflowed, {dropped} samples were dropped")]
    Overflow { dropped: u32 },

    #[error("device disconnected")]
    Disconnected,
}

/// Errors when connecting to a device
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ConnectionError {
    #[error("no OAK device found matching selector '{selector}'")]
    NotFound { selector: String },

    #[error("failed to start camera pipeline")]
    PipelineFailed,
}

// ============================================================================
// FRAME TYPES - Constructed only from valid FFI results
// ============================================================================

/// A valid RGB or mono image frame
#[derive(Debug, Clone)]
pub struct ImageFrame {
    pub stream: StreamId,
    pub sequence: u64,
    pub timestamp: Timestamp,
    pub width: u32,
    pub height: u32,
    pub stride_bytes: u32,
    data: Vec<u8>, // Private - use pixels() accessor
}

impl ImageFrame {
    /// Raw pixel data. For RGB: BGR888 format. For Mono: Grayscale.
    pub fn pixels(&self) -> &[u8] {
        &self.data
    }

    /// Consume frame and take ownership of pixel data
    pub fn into_pixels(self) -> Vec<u8> {
        self.data
    }
}

/// A valid depth frame
#[derive(Debug, Clone)]
pub struct DepthFrame {
    pub sequence: u64,
    pub timestamp: Timestamp,
    pub width: u32,
    pub height: u32,
    depth_scale: f32,
    min_depth_mm: u16,
    max_depth_mm: u16,
    data: Vec<u16>, // Private - use depth_mm() or depth_m() accessors
}

impl DepthFrame {
    /// Raw depth values in millimeters. Invalid depths are 0.
    pub fn depth_mm(&self) -> &[u16] {
        &self.data
    }

    /// Depth at pixel (x, y) in meters, or None if invalid/out-of-range
    pub fn depth_m_at(&self, x: u32, y: u32) -> Option<f32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = (y * self.width + x) as usize;
        let raw = self.data[idx];
        if raw < self.min_depth_mm || raw > self.max_depth_mm {
            return None;
        }
        Some(raw as f32 * self.depth_scale)
    }

    /// Valid depth range in meters
    pub fn valid_range_m(&self) -> (f32, f32) {
        (
            self.min_depth_mm as f32 * self.depth_scale,
            self.max_depth_mm as f32 * self.depth_scale,
        )
    }

    /// Consume frame and take ownership of raw depth data
    pub fn into_depth_mm(self) -> Vec<u16> {
        self.data
    }
}

/// A single IMU measurement
#[derive(Debug, Clone, Copy)]
pub struct ImuSample {
    pub timestamp: Timestamp,
    pub sequence: u32,
    pub accel: Vec3,
    pub accel_accuracy: ImuAccuracy,
    pub gyro: Vec3,
    pub gyro_accuracy: ImuAccuracy,
}

/// Camera intrinsic parameters
#[derive(Debug, Clone, Copy)]
pub struct Intrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
}

impl From<ffi::Intrinsics> for Intrinsics {
    fn from(i: ffi::Intrinsics) -> Self {
        Self {
            fx: i.fx,
            fy: i.fy,
            cx: i.cx,
            cy: i.cy,
            width: i.width,
            height: i.height,
        }
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

// ============================================================================
// DEVICE - Safe wrapper that enforces valid usage
// ============================================================================

/// A connected OAK-D device. Existence of this type guarantees connection.
pub struct Device {
    inner: cxx::UniquePtr<ffi::OakDevice>,
    config: DeviceConfig,
}

impl Device {
    /// List all available OAK devices
    pub fn list() -> Vec<DeviceInfo> {
        ffi::list_devices()
            .into_iter()
            .map(DeviceInfo::from)
            .collect()
    }

    /// Connect to a device. Empty selector uses first available.
    pub fn connect(selector: &str, config: DeviceConfig) -> Result<Self, ConnectionError> {
        let ffi_config = config.to_ffi();
        let inner = ffi::create_device(selector, &ffi_config);

        // CXX returns null UniquePtr if device not found
        if inner.is_null() {
            return Err(ConnectionError::NotFound {
                selector: selector.to_string(),
            });
        }

        // Check if pipeline actually started
        if !inner.is_connected() {
            return Err(ConnectionError::PipelineFailed);
        }

        Ok(Self { inner, config })
    }

    /// Get RGB frame. Returns error if RGB not enabled in config.
    pub fn rgb(&mut self, timeout_ms: u32) -> Result<ImageFrame, ImageError> {
        if self.config.rgb.is_none() {
            return Err(ImageError::StreamNotEnabled {
                stream: StreamId::Rgb,
            });
        }
        let result = self.inner.pin_mut().try_get_rgb(timeout_ms);
        parse_image_result(result, timeout_ms)
    }

    /// Get left mono frame. Returns error if mono not enabled in config.
    pub fn mono_left(&mut self, timeout_ms: u32) -> Result<ImageFrame, ImageError> {
        if self.config.mono.is_none() {
            return Err(ImageError::StreamNotEnabled {
                stream: StreamId::MonoLeft,
            });
        }
        let result = self.inner.pin_mut().try_get_mono_left(timeout_ms);
        parse_image_result(result, timeout_ms)
    }

    /// Get right mono frame. Returns error if mono not enabled in config.
    pub fn mono_right(&mut self, timeout_ms: u32) -> Result<ImageFrame, ImageError> {
        if self.config.mono.is_none() {
            return Err(ImageError::StreamNotEnabled {
                stream: StreamId::MonoRight,
            });
        }
        let result = self.inner.pin_mut().try_get_mono_right(timeout_ms);
        parse_image_result(result, timeout_ms)
    }

    /// Get depth frame. Returns error if depth not enabled in config.
    pub fn depth(&mut self, timeout_ms: u32) -> Result<DepthFrame, DepthError> {
        if self.config.depth.is_none() {
            return Err(DepthError::StreamNotEnabled);
        }
        let result = self.inner.pin_mut().try_get_depth(timeout_ms);
        parse_depth_result(result, timeout_ms)
    }

    /// Get IMU samples. Returns error if IMU not enabled in config.
    pub fn imu(&mut self) -> Result<Vec<ImuSample>, ImuError> {
        if self.config.imu.is_none() {
            return Err(ImuError::Empty);
        }
        let result = self.inner.pin_mut().get_imu_batch();
        parse_imu_result(result)
    }

    /// RGB camera intrinsics. Panics if RGB not enabled (invariant violation).
    pub fn rgb_intrinsics(&self) -> Intrinsics {
        assert!(
            self.config.rgb.is_some(),
            "rgb_intrinsics called but RGB not enabled"
        );
        self.inner.get_rgb_intrinsics().into()
    }

    /// Left mono camera intrinsics. Panics if mono not enabled.
    pub fn left_intrinsics(&self) -> Intrinsics {
        assert!(
            self.config.mono.is_some(),
            "left_intrinsics called but mono not enabled"
        );
        self.inner.get_left_intrinsics().into()
    }

    /// Right mono camera intrinsics. Panics if mono not enabled.
    pub fn right_intrinsics(&self) -> Intrinsics {
        assert!(
            self.config.mono.is_some(),
            "right_intrinsics called but mono not enabled"
        );
        self.inner.get_right_intrinsics().into()
    }

    /// Stereo baseline in meters (~7.5cm for OAK-D)
    pub fn stereo_baseline_m(&self) -> f32 {
        self.inner.get_stereo_baseline_m()
    }

    /// Gracefully disconnect
    pub fn close(mut self) {
        self.inner.pin_mut().close();
    }
}

// ============================================================================
// PARSING FUNCTIONS - Convert FFI results to proper Result types
// ============================================================================

fn parse_image_result(
    result: ffi::ImageFrameResult,
    timeout_ms: u32,
) -> Result<ImageFrame, ImageError> {
    match result.status {
        ffi::FrameStatus::Ok => Ok(ImageFrame {
            stream: result.frame.stream,
            sequence: result.frame.sequence,
            timestamp: result.frame.timestamp.into(),
            width: result.frame.width,
            height: result.frame.height,
            stride_bytes: result.frame.stride_bytes,
            data: result.frame.data,
        }),
        ffi::FrameStatus::Timeout => Err(ImageError::Timeout { timeout_ms }),
        ffi::FrameStatus::QueueEmpty => Err(ImageError::QueueEmpty),
        ffi::FrameStatus::QueueOverflow => Err(ImageError::QueueOverflow),
        ffi::FrameStatus::Disconnected => Err(ImageError::Disconnected),
        ffi::FrameStatus::Corrupt => Err(ImageError::Corrupt),
        ffi::FrameStatus::StreamNotEnabled => Err(ImageError::StreamNotEnabled {
            stream: result.frame.stream,
        }),
        _ => Err(ImageError::Corrupt),
    }
}

fn parse_depth_result(
    result: ffi::DepthFrameResult,
    timeout_ms: u32,
) -> Result<DepthFrame, DepthError> {
    match result.status {
        ffi::FrameStatus::Ok => Ok(DepthFrame {
            sequence: result.frame.sequence,
            timestamp: result.frame.timestamp.into(),
            width: result.frame.width,
            height: result.frame.height,
            depth_scale: result.frame.depth_scale,
            min_depth_mm: result.frame.min_depth_mm,
            max_depth_mm: result.frame.max_depth_mm,
            data: result.frame.data,
        }),
        ffi::FrameStatus::Timeout => Err(DepthError::Timeout { timeout_ms }),
        ffi::FrameStatus::QueueEmpty => Err(DepthError::QueueEmpty),
        ffi::FrameStatus::QueueOverflow => Err(DepthError::QueueOverflow),
        ffi::FrameStatus::Disconnected => Err(DepthError::Disconnected),
        ffi::FrameStatus::Corrupt => Err(DepthError::Corrupt),
        ffi::FrameStatus::StreamNotEnabled => Err(DepthError::StreamNotEnabled),
        _ => Err(DepthError::Corrupt),
    }
}

fn parse_imu_result(result: ffi::ImuBatchResult) -> Result<Vec<ImuSample>, ImuError> {
    match result.status {
        ffi::ImuStatus::Ok => {
            let samples = result
                .batch
                .samples
                .into_iter()
                .map(|s| ImuSample {
                    timestamp: s.timestamp.into(),
                    sequence: s.sequence,
                    accel: Vec3 {
                        x: s.accel_x,
                        y: s.accel_y,
                        z: s.accel_z,
                    },
                    accel_accuracy: s.accel_accuracy,
                    gyro: Vec3 {
                        x: s.gyro_x,
                        y: s.gyro_y,
                        z: s.gyro_z,
                    },
                    gyro_accuracy: s.gyro_accuracy,
                })
                .collect();
            Ok(samples)
        }
        ffi::ImuStatus::Empty => Err(ImuError::Empty),
        ffi::ImuStatus::Overflow => Err(ImuError::Overflow {
            dropped: result.batch.dropped_count,
        }),
        ffi::ImuStatus::Disconnected => Err(ImuError::Disconnected),
        _ => Err(ImuError::Disconnected),
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        if self.inner.is_connected() {
            self.inner.pin_mut().close();
        }
    }
}
