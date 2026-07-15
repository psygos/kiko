use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use crate::{
    CalibrationBundleError, CaptureBundleError, CaptureIntervalError, DepthImage, Frame,
    FrameDimensions, FrameDimensionsError, FrameError, ImuBatch, ImuBatchError, ImuBatchSliceError,
    ImuSampleError, ImuTimestampShiftError, PairError, PairingConfigError, SensorId, StereoPair,
    Timestamp,
};

pub mod format {
    use std::num::ParseIntError;

    pub const FRAMES_DIR: &str = "frames";
    pub const META_FILE: &str = "meta.json";
    pub const CALIBRATION_FILE: &str = "calibration.json";
    pub const MANIFEST_FILE: &str = "manifest.json";
    pub const IMU_FILE: &str = "imu.bin";
    pub const FRAME_SUFFIX: &str = ".raw";

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum FrameKind {
        MonoLeft,
        MonoRight,
        Depth,
    }

    impl FrameKind {
        pub const fn as_str(self) -> &'static str {
            match self {
                Self::MonoLeft => "mono_left",
                Self::MonoRight => "mono_right",
                Self::Depth => "depth",
            }
        }

        fn parse(value: &str) -> Result<Self, FrameFilenameError> {
            match value {
                "mono_left" => Ok(Self::MonoLeft),
                "mono_right" => Ok(Self::MonoRight),
                "depth" => Ok(Self::Depth),
                _ => Err(FrameFilenameError::UnknownFrameKind {
                    value: value.to_string(),
                }),
            }
        }
    }

    impl std::fmt::Display for FrameKind {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(self.as_str())
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct FrameFilename {
        timestamp_ns: i64,
        kind: FrameKind,
    }

    impl FrameFilename {
        pub const fn timestamp_ns(self) -> i64 {
            self.timestamp_ns
        }

        pub const fn kind(self) -> FrameKind {
            self.kind
        }
    }

    #[derive(Debug)]
    pub enum FrameFilenameError {
        MissingRawSuffix,
        MissingTimestampSeparator,
        InvalidTimestamp { source: ParseIntError },
        UnknownFrameKind { value: String },
    }

    impl std::fmt::Display for FrameFilenameError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::MissingRawSuffix => write!(f, "frame filename must end with {FRAME_SUFFIX}"),
                Self::MissingTimestampSeparator => {
                    write!(f, "frame filename must contain a timestamp and frame kind")
                }
                Self::InvalidTimestamp { source } => {
                    write!(f, "frame filename has an invalid i64 timestamp: {source}")
                }
                Self::UnknownFrameKind { value } => {
                    write!(f, "frame filename has unknown frame kind {value:?}")
                }
            }
        }
    }

    impl std::error::Error for FrameFilenameError {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            match self {
                Self::InvalidTimestamp { source } => Some(source),
                Self::MissingRawSuffix
                | Self::MissingTimestampSeparator
                | Self::UnknownFrameKind { .. } => None,
            }
        }
    }

    pub fn frame_name(timestamp_ns: i64, kind: FrameKind) -> String {
        format!("{timestamp_ns}_{}{FRAME_SUFFIX}", kind.as_str())
    }

    pub fn parse_frame_filename(filename: &str) -> Result<FrameFilename, FrameFilenameError> {
        let stem = filename
            .strip_suffix(FRAME_SUFFIX)
            .ok_or(FrameFilenameError::MissingRawSuffix)?;
        let (timestamp_str, kind) = stem
            .split_once('_')
            .ok_or(FrameFilenameError::MissingTimestampSeparator)?;
        let timestamp_ns = timestamp_str
            .parse::<i64>()
            .map_err(|source| FrameFilenameError::InvalidTimestamp { source })?;
        Ok(FrameFilename {
            timestamp_ns,
            kind: FrameKind::parse(kind)?,
        })
    }
}

mod reader;
pub use reader::{DatasetReadTimings, DatasetReader, DatasetStats, TimedPair};

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
    pub rate_hz: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DepthMeta {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub encoding: String,
}

// Calibration structs
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Calibration {
    pub left: CameraIntrinsics,
    pub right: CameraIntrinsics,
    pub baseline_m: f32,
    #[serde(default)]
    pub rectified: bool,
    #[serde(default)]
    pub imu: Option<ImuCalibration>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImuCalibration {
    pub noise: ImuNoiseMeta,
    pub extrinsics: ImuExtrinsicsMeta,
    pub gravity_magnitude_mps2: f64,
    /// Factory-calibrated accelerometer bias [m/s²] (e.g. from Basalt/kalibr).
    #[serde(default)]
    pub initial_accel_bias: Option<[f64; 3]>,
    /// Factory-calibrated gyroscope bias [rad/s] (e.g. from Basalt/kalibr).
    #[serde(default)]
    pub initial_gyro_bias: Option<[f64; 3]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImuNoiseMeta {
    pub accel_noise_density: f64,
    pub gyro_noise_density: f64,
    pub accel_random_walk: f64,
    pub gyro_random_walk: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImuExtrinsicsMeta {
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
    #[serde(default)]
    pub time_offset_ns: i64,
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

#[derive(Clone, Copy, Debug)]
pub enum Backpressure {
    DropNewest,
    Block,
}

#[derive(Clone, Copy, Debug)]
pub struct DatasetWriterConfig {
    pub max_spool_frames: usize,
    pub max_spool_bytes: usize,
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

#[derive(Clone, Copy, Debug)]
pub struct WriterStats {
    pub frames_enqueued: u64,
    pub frames_written: u64,
    pub frames_dropped: u64,
    pub bytes_enqueued: u64,
    pub bytes_written: u64,
    pub bytes_dropped: u64,
    pub write_failed: u64,
    pub spool_frames: u64,
    pub spool_bytes: u64,
    pub spool_max_frames: u64,
    pub spool_max_bytes: u64,
    pub writer_failed: bool,
}

impl WriterStats {
    pub fn frames_pending(&self) -> u64 {
        self.frames_enqueued
            .saturating_sub(self.frames_written.saturating_add(self.frames_dropped))
    }

    pub fn bytes_pending(&self) -> u64 {
        self.bytes_enqueued
            .saturating_sub(self.bytes_written.saturating_add(self.bytes_dropped))
    }
}

#[derive(Debug)]
pub enum DatasetImuRecordError {
    Decode { source: std::io::Error },
    InvalidSample { source: ImuSampleError },
    TimestampShift { source: ImuTimestampShiftError },
}

impl std::fmt::Display for DatasetImuRecordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Decode { source } => write!(f, "could not decode fixed-width fields: {source}"),
            Self::InvalidSample { source } => write!(f, "invalid IMU sample: {source}"),
            Self::TimestampShift { source } => {
                write!(f, "invalid IMU timestamp shift: {source}")
            }
        }
    }
}

impl std::error::Error for DatasetImuRecordError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Decode { source } => Some(source),
            Self::InvalidSample { source } => Some(source),
            Self::TimestampShift { source } => Some(source),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DatasetFrameRole {
    Left,
    Right,
}

impl std::fmt::Display for DatasetFrameRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Left => f.write_str("left"),
            Self::Right => f.write_str("right"),
        }
    }
}

#[derive(Debug)]
pub enum DatasetFrameReferenceError {
    NonCanonicalPath {
        path: String,
    },
    WrongFrameKind {
        path: String,
        expected: format::FrameKind,
        actual: format::FrameKind,
    },
    TimestampMismatch {
        path: String,
        declared_ns: i64,
        filename_ns: i64,
    },
    MissingFromDataset {
        path: String,
    },
}

impl std::fmt::Display for DatasetFrameReferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonCanonicalPath { path } => write!(
                f,
                "frame path {path:?} must be exactly {}/<frame filename>",
                format::FRAMES_DIR
            ),
            Self::WrongFrameKind {
                path,
                expected,
                actual,
            } => write!(
                f,
                "frame path {path:?} has kind {actual}; expected {expected}"
            ),
            Self::TimestampMismatch {
                path,
                declared_ns,
                filename_ns,
            } => write!(
                f,
                "frame path {path:?} declares timestamp {declared_ns}, but its filename encodes {filename_ns}"
            ),
            Self::MissingFromDataset { path } => {
                write!(
                    f,
                    "frame path {path:?} is not present in the scanned dataset"
                )
            }
        }
    }
}

impl std::error::Error for DatasetFrameReferenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NonCanonicalPath { .. }
            | Self::WrongFrameKind { .. }
            | Self::TimestampMismatch { .. }
            | Self::MissingFromDataset { .. } => None,
        }
    }
}

#[derive(Debug)]
pub enum DatasetManifestError {
    MissingMonoConfig,
    HeaderTextMismatch {
        field: &'static str,
        expected: String,
        actual: String,
    },
    HeaderNumberMismatch {
        field: &'static str,
        expected: u64,
        actual: u64,
    },
    InvalidPairingWindow {
        source: PairingConfigError,
    },
    InvalidFrameReference {
        entry_index: usize,
        role: DatasetFrameRole,
        source: DatasetFrameReferenceError,
    },
    DuplicateFrameReference {
        entry_index: usize,
        role: DatasetFrameRole,
        path: String,
    },
    NonIncreasingTimestamp {
        entry_index: usize,
        role: DatasetFrameRole,
        previous_ns: i64,
        current_ns: i64,
    },
    PairDeltaMismatch {
        entry_index: usize,
        declared_ns: u64,
        actual_ns: u64,
    },
    PairOutsideWindow {
        entry_index: usize,
        delta_ns: u64,
        pairing_window_ns: u64,
    },
    PairedCountExceedsAvailableRightFrames {
        paired_count: usize,
        available_right: usize,
    },
    CountMismatch {
        field: &'static str,
        declared: u64,
        actual: u64,
    },
    CountOverflow {
        field: &'static str,
        value: usize,
    },
}

impl std::fmt::Display for DatasetManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingMonoConfig => write!(f, "manifest requires mono dataset metadata"),
            Self::HeaderTextMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "manifest header {field} must be {expected:?}, got {actual:?}"
            ),
            Self::HeaderNumberMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "manifest header {field} must be {expected}, got {actual}"
            ),
            Self::InvalidPairingWindow { source } => {
                write!(f, "manifest has an invalid pairing window: {source}")
            }
            Self::InvalidFrameReference {
                entry_index,
                role,
                source,
            } => write!(
                f,
                "manifest entry {entry_index} has an invalid {role} frame reference: {source}"
            ),
            Self::DuplicateFrameReference {
                entry_index,
                role,
                path,
            } => write!(
                f,
                "manifest entry {entry_index} repeats {role} frame path {path:?}"
            ),
            Self::NonIncreasingTimestamp {
                entry_index,
                role,
                previous_ns,
                current_ns,
            } => write!(
                f,
                "manifest entry {entry_index} has non-increasing {role} timestamp {current_ns} after {previous_ns}"
            ),
            Self::PairDeltaMismatch {
                entry_index,
                declared_ns,
                actual_ns,
            } => write!(
                f,
                "manifest entry {entry_index} declares pair delta {declared_ns}ns, but timestamps differ by {actual_ns}ns"
            ),
            Self::PairOutsideWindow {
                entry_index,
                delta_ns,
                pairing_window_ns,
            } => write!(
                f,
                "manifest entry {entry_index} pair delta {delta_ns}ns exceeds the {pairing_window_ns}ns pairing window"
            ),
            Self::PairedCountExceedsAvailableRightFrames {
                paired_count,
                available_right,
            } => write!(
                f,
                "validated manifest contains {paired_count} pairs but only {available_right} right frames are available"
            ),
            Self::CountMismatch {
                field,
                declared,
                actual,
            } => write!(
                f,
                "manifest count {field} declares {declared}, but validated data contains {actual}"
            ),
            Self::CountOverflow { field, value } => write!(
                f,
                "validated dataset count {field}={value} cannot be represented as u64"
            ),
        }
    }
}

impl std::error::Error for DatasetManifestError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidPairingWindow { source } => Some(source),
            Self::InvalidFrameReference { source, .. } => Some(source),
            Self::MissingMonoConfig
            | Self::HeaderTextMismatch { .. }
            | Self::HeaderNumberMismatch { .. }
            | Self::DuplicateFrameReference { .. }
            | Self::NonIncreasingTimestamp { .. }
            | Self::PairDeltaMismatch { .. }
            | Self::PairOutsideWindow { .. }
            | Self::PairedCountExceedsAvailableRightFrames { .. }
            | Self::CountMismatch { .. }
            | Self::CountOverflow { .. } => None,
        }
    }
}

#[derive(Debug)]
pub enum DatasetError {
    OutputAlreadyExists {
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
    NonUnicodeFrameFilename {
        path: PathBuf,
    },
    InvalidFrameFilename {
        path: PathBuf,
        source: format::FrameFilenameError,
    },
    UnexpectedFrameKind {
        path: PathBuf,
        kind: format::FrameKind,
    },
    InvalidFrameDimensions {
        stream: &'static str,
        source: FrameDimensionsError,
    },
    FrameByteLengthOverflow {
        stream: &'static str,
        width: u32,
        height: u32,
    },
    FrameSizeMismatch {
        path: PathBuf,
        expected_bytes: u64,
        actual_bytes: u64,
    },
    InvalidImuLength {
        path: PathBuf,
        byte_len: usize,
        record_bytes: usize,
    },
    InvalidImuRecord {
        path: PathBuf,
        record_index: usize,
        source: DatasetImuRecordError,
    },
    InvalidImuBatch {
        path: PathBuf,
        source: ImuBatchError,
    },
    InvalidImuSlice {
        start_ns: Option<i64>,
        end_ns: i64,
        source: ImuBatchSliceError,
    },
    InvalidConfig {
        msg: &'static str,
    },
    InvalidFrame {
        path: PathBuf,
        source: FrameError,
    },
    ThreadSpawn {
        source: std::io::Error,
    },
    WriteFile {
        path: PathBuf,
        source: std::io::Error,
    },
    SerializeJson {
        source: serde_json::Error,
    },
    DeserializeJson {
        source: serde_json::Error,
    },
    InvalidCalibration {
        path: PathBuf,
        source: CalibrationBundleError,
    },
    CalibrationDimensionsMismatch {
        metadata: FrameDimensions,
        calibration: FrameDimensions,
    },
    InvalidManifest {
        path: PathBuf,
        source: DatasetManifestError,
    },
    WorkerJoin {
        message: String,
    },
    PairingFailed {
        source: PairError,
    },
    InvalidCaptureInterval {
        source: CaptureIntervalError,
    },
    InvalidCaptureBundle {
        source: CaptureBundleError,
    },
    FrameSequenceExhausted {
        sensor: SensorId,
    },
    CaptureSequenceExhausted,
    MissingImuSamples {
        start_ns: Option<i64>,
        end_ns: i64,
    },
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetError::OutputAlreadyExists { path } => {
                write!(f, "dataset output path already exists: {}", path.display())
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
            DatasetError::NonUnicodeFrameFilename { path } => write!(
                f,
                "dataset frame filename is not valid Unicode: {}",
                path.display()
            ),
            DatasetError::InvalidFrameFilename { path, source } => write!(
                f,
                "invalid dataset frame filename {}: {source}",
                path.display()
            ),
            DatasetError::UnexpectedFrameKind { path, kind } => write!(
                f,
                "dataset contains an unconfigured {kind} frame: {}",
                path.display()
            ),
            DatasetError::InvalidFrameDimensions { stream, source } => {
                write!(f, "invalid {stream} dataset dimensions: {source}")
            }
            DatasetError::FrameByteLengthOverflow {
                stream,
                width,
                height,
            } => write!(
                f,
                "{stream} frame byte length overflows u64 for dimensions {width}x{height}"
            ),
            DatasetError::FrameSizeMismatch {
                path,
                expected_bytes,
                actual_bytes,
            } => write!(
                f,
                "dataset frame {} has {actual_bytes} bytes; expected {expected_bytes}",
                path.display()
            ),
            DatasetError::InvalidImuLength {
                path,
                byte_len,
                record_bytes,
            } => write!(
                f,
                "invalid IMU file {}: {byte_len} bytes is not a whole number of {record_bytes}-byte records",
                path.display()
            ),
            DatasetError::InvalidImuRecord {
                path,
                record_index,
                source,
            } => write!(
                f,
                "invalid IMU record {record_index} in {}: {source}",
                path.display()
            ),
            DatasetError::InvalidImuBatch { path, source } => write!(
                f,
                "invalid IMU sample sequence in {}: {source}",
                path.display()
            ),
            DatasetError::InvalidImuSlice {
                start_ns,
                end_ns,
                source,
            } => write!(
                f,
                "invalid dataset IMU slice for interval ({start_ns:?}, {end_ns}]: {source}"
            ),
            DatasetError::InvalidConfig { msg } => write!(f, "invalid dataset config: {msg}"),
            DatasetError::InvalidFrame { path, source } => {
                write!(f, "invalid dataset frame {}: {source}", path.display())
            }
            DatasetError::ThreadSpawn { source } => {
                write!(f, "failed to spawn writer thread: {source}")
            }
            DatasetError::WriteFile { path, source } => {
                write!(f, "failed to write file {}: {}", path.display(), source)
            }
            DatasetError::SerializeJson { source } => {
                write!(f, "failed to serialize JSON: {source}")
            }
            DatasetError::DeserializeJson { source } => {
                write!(f, "failed to deserialize JSON: {source}")
            }
            DatasetError::InvalidCalibration { path, source } => write!(
                f,
                "invalid dataset calibration {}: {source}",
                path.display()
            ),
            DatasetError::CalibrationDimensionsMismatch {
                metadata,
                calibration,
            } => write!(
                f,
                "dataset mono dimensions {}x{} do not match calibration dimensions {}x{}",
                metadata.width(),
                metadata.height(),
                calibration.width(),
                calibration.height()
            ),
            DatasetError::InvalidManifest { path, source } => {
                write!(f, "invalid dataset manifest {}: {source}", path.display())
            }
            DatasetError::WorkerJoin { message } => {
                write!(f, "writer thread panicked: {message}")
            }
            DatasetError::PairingFailed { source } => {
                write!(f, "dataset pairing failed: {source}")
            }
            DatasetError::InvalidCaptureInterval { source } => {
                write!(f, "invalid dataset capture interval: {source}")
            }
            DatasetError::InvalidCaptureBundle { source } => {
                write!(f, "invalid dataset capture bundle: {source}")
            }
            DatasetError::FrameSequenceExhausted { sensor } => {
                write!(f, "dataset {sensor:?} frame sequence exhausted u64 IDs")
            }
            DatasetError::CaptureSequenceExhausted => {
                write!(f, "dataset capture sequence exhausted u64 IDs")
            }
            DatasetError::MissingImuSamples { start_ns, end_ns } => {
                write!(
                    f,
                    "dataset missing imu samples for interval ({start_ns:?}, {end_ns}]"
                )
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
            | DatasetError::WriteFile { source, .. } => Some(source),
            DatasetError::SerializeJson { source } | DatasetError::DeserializeJson { source } => {
                Some(source)
            }
            DatasetError::InvalidCalibration { source, .. } => Some(source),
            DatasetError::InvalidManifest { source, .. } => Some(source),
            DatasetError::PairingFailed { source } => Some(source),
            DatasetError::InvalidCaptureInterval { source } => Some(source),
            DatasetError::InvalidCaptureBundle { source } => Some(source),
            DatasetError::InvalidFrame { source, .. } => Some(source),
            DatasetError::InvalidImuRecord { source, .. } => Some(source),
            DatasetError::InvalidImuBatch { source, .. } => Some(source),
            DatasetError::InvalidImuSlice { source, .. } => Some(source),
            DatasetError::InvalidFrameFilename { source, .. } => Some(source),
            DatasetError::InvalidFrameDimensions { source, .. } => Some(source),
            DatasetError::InvalidConfig { .. }
            | DatasetError::OutputAlreadyExists { .. }
            | DatasetError::NonUnicodeFrameFilename { .. }
            | DatasetError::UnexpectedFrameKind { .. }
            | DatasetError::FrameByteLengthOverflow { .. }
            | DatasetError::FrameSizeMismatch { .. }
            | DatasetError::InvalidImuLength { .. }
            | DatasetError::CalibrationDimensionsMismatch { .. }
            | DatasetError::FrameSequenceExhausted { .. }
            | DatasetError::CaptureSequenceExhausted
            | DatasetError::WorkerJoin { .. }
            | DatasetError::MissingImuSamples { .. } => None,
        }
    }
}

#[derive(Debug)]
pub struct DatasetWriter {
    config: DatasetWriterConfig,
    state: Arc<WriterState>,
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
        Self::create_with_config(path, meta, calibration, DatasetWriterConfig::default())
    }

    pub fn create_with_config(
        path: impl Into<PathBuf>,
        meta: &Meta,
        calibration: &Calibration,
        config: DatasetWriterConfig,
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

        let path = path.into();
        let validated_calibration = crate::CalibrationBundle::from_dataset_calibration(calibration)
            .map_err(|source| DatasetError::InvalidCalibration {
                path: path.join(format::CALIBRATION_FILE),
                source,
            })?;
        require_calibration_dimensions(meta, &validated_calibration)?;
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).map_err(|source| DatasetError::CreateDirectory {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        match std::fs::create_dir(&path) {
            Ok(()) => {}
            Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(DatasetError::OutputAlreadyExists { path });
            }
            Err(source) => {
                return Err(DatasetError::CreateDirectory {
                    path: path.clone(),
                    source,
                });
            }
        }

        let frames_dir = path.join(format::FRAMES_DIR);
        std::fs::create_dir(&frames_dir).map_err(|e| DatasetError::CreateDirectory {
            path: frames_dir.clone(),
            source: e,
        })?;

        let meta_path = path.join(format::META_FILE);
        let meta_file = std::fs::File::create(&meta_path).map_err(|e| DatasetError::WriteFile {
            path: meta_path.clone(),
            source: e,
        })?;

        let calibration_path = path.join(format::CALIBRATION_FILE);
        let calibration_file =
            std::fs::File::create(&calibration_path).map_err(|e| DatasetError::WriteFile {
                path: calibration_path.clone(),
                source: e,
            })?;

        serde_json::to_writer_pretty(calibration_file, calibration)
            .map_err(|e| DatasetError::SerializeJson { source: e })?;

        serde_json::to_writer_pretty(meta_file, meta)
            .map_err(|e| DatasetError::SerializeJson { source: e })?;

        let state = Arc::new(WriterState::new(config, path.clone(), frames_dir.clone()));
        let state_for_thread = state.clone();

        let handle = thread::Builder::new()
            .name("dataset-writer".to_string())
            .spawn(move || writer_loop(state_for_thread))
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

    /// Enqueue a frame according to the configured backpressure policy.
    pub fn write_frame(&self, frame: &Frame) -> WriteOutcome {
        self.write_item(
            SpoolItem::Mono(frame.clone()),
            frame.data().len(),
            1,
            "frame exceeds max_spool_bytes",
        )
    }

    /// Enqueue both halves of a validated stereo pair as one backpressure unit.
    pub fn write_stereo_pair(&self, pair: &StereoPair) -> WriteOutcome {
        let bytes = pair
            .left()
            .data()
            .len()
            .saturating_add(pair.right().data().len());
        self.write_item(
            SpoolItem::StereoPair(pair.left().clone(), pair.right().clone()),
            bytes,
            2,
            "stereo pair exceeds max_spool_bytes",
        )
    }

    /// Enqueue a depth image according to the configured backpressure policy.
    pub fn write_depth(&self, depth: &DepthImage) -> WriteOutcome {
        let bytes = depth
            .depth_m()
            .len()
            .saturating_mul(std::mem::size_of::<f32>());
        self.write_item(
            SpoolItem::Depth(depth.clone()),
            bytes,
            1,
            "depth image exceeds max_spool_bytes",
        )
    }

    /// Enqueue a validated IMU batch according to the configured backpressure policy.
    pub fn write_imu(&self, batch: &ImuBatch) -> WriteOutcome {
        let bytes = batch.len().saturating_mul(IMU_RECORD_BYTES);
        self.write_item(
            SpoolItem::Imu(batch.clone()),
            bytes,
            1,
            "imu batch exceeds max_spool_bytes",
        )
    }

    fn write_item(
        &self,
        item: SpoolItem,
        bytes: usize,
        frame_count: usize,
        oversize_msg: &'static str,
    ) -> WriteOutcome {
        if self.state.failed.load(Ordering::Acquire) {
            return WriteOutcome::WriterFailed;
        }

        if bytes > self.config.max_spool_bytes {
            self.state
                .fail(DatasetError::InvalidConfig { msg: oversize_msg });
            return WriteOutcome::WriterFailed;
        }

        let mut spool = self
            .state
            .spool
            .lock()
            .unwrap_or_else(|err| err.into_inner());

        match self.config.backpressure {
            Backpressure::DropNewest => {
                if spool.closed || self.state.failed.load(Ordering::Acquire) {
                    return WriteOutcome::WriterFailed;
                }
                if !self.state.can_accept(&spool, frame_count, bytes) {
                    self.state
                        .dropped
                        .fetch_add(frame_count as u64, Ordering::Relaxed);
                    self.state
                        .bytes_dropped
                        .fetch_add(bytes as u64, Ordering::Relaxed);
                    return WriteOutcome::Dropped;
                }
            }
            Backpressure::Block => {
                while !self.state.can_accept(&spool, frame_count, bytes) {
                    if spool.closed || self.state.failed.load(Ordering::Acquire) {
                        return WriteOutcome::WriterFailed;
                    }
                    spool = self
                        .state
                        .spool_cvar
                        .wait(spool)
                        .unwrap_or_else(|err| err.into_inner());
                }
            }
        }

        if spool.closed {
            return WriteOutcome::WriterFailed;
        }

        spool.frames += frame_count;
        spool.bytes += bytes;
        spool.queue.push_back(item);

        self.state
            .enqueued
            .fetch_add(frame_count as u64, Ordering::Relaxed);
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
        WriteOutcome::Enqueued
    }

    pub fn stats(&self) -> WriterStats {
        self.state.stats()
    }

    pub fn is_healthy(&self) -> bool {
        !self.state.failed.load(Ordering::Acquire)
    }
}

impl DatasetWriterHandle {
    /// Blocks until the writer thread exits; all DatasetWriter clones must be dropped first.
    pub fn finish(mut self) -> Result<WriterStats, DatasetError> {
        let Some(handle) = self.handle.take() else {
            return Err(DatasetError::InvalidConfig {
                msg: "finish called twice",
            });
        };
        handle.join().map_err(|err| DatasetError::WorkerJoin {
            message: panic_message(err),
        })?;

        let writer_error = self.state.take_error();
        if let Some(err) = writer_error {
            return Err(err);
        }

        write_manifest(&self.state)?;
        Ok(self.state.stats())
    }

    pub fn stats(&self) -> WriterStats {
        self.state.stats()
    }
}

#[derive(Debug)]
enum SpoolItem {
    Mono(Frame),
    StereoPair(Frame, Frame),
    Depth(DepthImage),
    Imu(ImuBatch),
}

impl SpoolItem {
    fn bytes_len(&self) -> usize {
        match self {
            SpoolItem::Mono(frame) => frame.data().len(),
            SpoolItem::StereoPair(left, right) => {
                left.data().len().saturating_add(right.data().len())
            }
            SpoolItem::Depth(depth) => depth
                .depth_m()
                .len()
                .saturating_mul(std::mem::size_of::<f32>()),
            SpoolItem::Imu(batch) => batch.len().saturating_mul(IMU_RECORD_BYTES),
        }
    }

    fn frame_count(&self) -> usize {
        match self {
            Self::StereoPair(_, _) => 2,
            Self::Mono(_) | Self::Depth(_) | Self::Imu(_) => 1,
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

#[derive(Debug)]
struct WriterState {
    config: DatasetWriterConfig,
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
    spool_frames: AtomicU64,
    spool_bytes: AtomicU64,
    open_writers: AtomicUsize,
    failed: AtomicBool,
    error: Mutex<Option<DatasetError>>,
    recorded_depth: Mutex<Vec<RecordedDepth>>,
}

impl WriterState {
    fn new(config: DatasetWriterConfig, dataset_dir: PathBuf, frames_dir: PathBuf) -> Self {
        Self {
            config,
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
            spool_frames: AtomicU64::new(0),
            spool_bytes: AtomicU64::new(0),
            open_writers: AtomicUsize::new(1),
            failed: AtomicBool::new(false),
            error: Mutex::new(None),
            recorded_depth: Mutex::new(Vec::new()),
        }
    }

    fn can_accept(&self, spool: &Spool, frame_count: usize, bytes: usize) -> bool {
        let next_frames = spool.frames.saturating_add(frame_count);
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

    fn record_depth(&self, depth: RecordedDepth) {
        self.recorded_depth
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .push(depth);
    }
}

pub(super) const IMU_RECORD_BYTES: usize =
    std::mem::size_of::<i64>() + 6 * std::mem::size_of::<f64>();

fn writer_loop(state: Arc<WriterState>) {
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
                let bytes = item.bytes_len();
                let frame_count = item.frame_count();
                spool.frames = spool.frames.saturating_sub(frame_count);
                spool.bytes = spool.bytes.saturating_sub(bytes);
                batch_frames = batch_frames.saturating_add(frame_count);
                batch.push(item);
                if batch_frames >= state.config.flush_batch_frames {
                    break;
                }
            }

            state
                .spool_frames
                .store(spool.frames as u64, Ordering::Relaxed);
            state
                .spool_bytes
                .store(spool.bytes as u64, Ordering::Relaxed);
            state.spool_cvar.notify_all();
            batch
        };

        for item in batch {
            let bytes = item.bytes_len() as u64;
            let frame_count = item.frame_count() as u64;
            match write_item_to_dir(&state.frames_dir, &state.dataset_dir, item) {
                Ok(Some(depth)) => state.record_depth(depth),
                Ok(None) => {}
                Err(err) => {
                    state.write_failed.fetch_add(frame_count, Ordering::Relaxed);
                    state.fail(err);
                    return;
                }
            }
            state.written.fetch_add(frame_count, Ordering::Relaxed);
            state.bytes_written.fetch_add(bytes, Ordering::Relaxed);
        }
    }
}

fn write_item_to_dir(
    frames_dir: &Path,
    dataset_dir: &Path,
    item: SpoolItem,
) -> Result<Option<RecordedDepth>, DatasetError> {
    match item {
        SpoolItem::Mono(frame) => {
            write_frame_to_dir(frames_dir, frame)?;
            Ok(None)
        }
        SpoolItem::StereoPair(left, right) => {
            write_frame_to_dir(frames_dir, left)?;
            write_frame_to_dir(frames_dir, right)?;
            Ok(None)
        }
        SpoolItem::Depth(depth) => write_depth_to_dir(frames_dir, depth).map(Some),
        SpoolItem::Imu(batch) => {
            write_imu_to_file(dataset_dir, &batch)?;
            Ok(None)
        }
    }
}

fn write_frame_to_dir(frames_dir: &Path, frame: Frame) -> Result<(), DatasetError> {
    let Frame {
        sensor_id,
        frame_id: _,
        timestamp,
        dimensions: _,
        data,
    } = frame;
    let filename = format::frame_name(timestamp.as_nanos(), frame_kind(sensor_id));
    let path = frames_dir.join(&filename);

    write_new_file(path, data.as_ref())
}

fn write_depth_to_dir(frames_dir: &Path, depth: DepthImage) -> Result<RecordedDepth, DatasetError> {
    let timestamp_ns = depth.timestamp().as_nanos();
    let filename = format::frame_name(timestamp_ns, format::FrameKind::Depth);
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
    write_new_file(path, &bytes)?;
    Ok(RecordedDepth { timestamp_ns })
}

fn write_new_file(path: PathBuf, bytes: &[u8]) -> Result<(), DatasetError> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|source| DatasetError::WriteFile {
            path: path.clone(),
            source,
        })?;
    std::io::Write::write_all(&mut file, bytes)
        .map_err(|source| DatasetError::WriteFile { path, source })
}

fn write_imu_to_file(dataset_dir: &Path, batch: &ImuBatch) -> Result<(), DatasetError> {
    let path = dataset_dir.join(format::IMU_FILE);
    let mut bytes = Vec::with_capacity(batch.len().saturating_mul(IMU_RECORD_BYTES));
    for sample in batch.samples() {
        bytes.extend_from_slice(&sample.timestamp().as_nanos().to_le_bytes());
        for value in sample.accel_mps2() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in sample.gyro_radps() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|source| DatasetError::WriteFile {
            path: path.clone(),
            source,
        })?;
    std::io::Write::write_all(&mut file, &bytes)
        .map_err(|source| DatasetError::WriteFile { path, source })
}

fn frame_kind(id: SensorId) -> format::FrameKind {
    match id {
        SensorId::StereoLeft => format::FrameKind::MonoLeft,
        SensorId::StereoRight => format::FrameKind::MonoRight,
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
    /// `None` is the legacy or unconfigured representation; `Some([])` records a configured
    /// stream that completed without writing any depth payloads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    depth_entries: Option<Vec<ManifestFrameRef>>,
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
struct DropStats {
    spool_full: u64,
    write_fail: u64,
    parse_fail: u64,
    size_mismatch: u64,
    outside_window: u64,
}

#[derive(Debug, Serialize, Deserialize)]
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

#[derive(Clone, Copy, Debug)]
struct RecordedDepth {
    timestamp_ns: i64,
}

#[derive(Debug)]
struct FrameSet {
    mono_dimensions: FrameDimensions,
    left: Vec<FrameInfo>,
    right: Vec<FrameInfo>,
}

#[derive(Debug)]
struct DatasetIndex {
    stats: DatasetStats,
    pairs: Box<[DatasetPairIndex]>,
}

#[derive(Debug)]
struct DatasetPairIndex {
    left: DatasetIndexFrameRef,
    right: DatasetIndexFrameRef,
}

#[derive(Debug)]
struct DatasetIndexFrameRef {
    timestamp: Timestamp,
    path: Arc<str>,
}

impl DatasetIndex {
    fn try_from_manifest(
        manifest: Manifest,
        meta: &Meta,
        frames: &FrameSet,
    ) -> Result<Self, DatasetManifestError> {
        let mono = meta
            .mono
            .as_ref()
            .ok_or(DatasetManifestError::MissingMonoConfig)?;
        require_manifest_text("format", &manifest.header.format, "raw")?;
        require_manifest_text("timebase", &manifest.header.timebase, "device_ns")?;
        require_manifest_text(
            "pairing_policy",
            &manifest.header.pairing_policy,
            "time_symmetric",
        )?;
        require_manifest_text("created_at", &manifest.header.created_at, &meta.created)?;
        require_manifest_text("device", &manifest.header.device, &meta.device)?;
        require_manifest_number("width", manifest.header.width, mono.width)?;
        require_manifest_number("height", manifest.header.height, mono.height)?;
        require_manifest_number("fps", manifest.header.fps, mono.fps)?;
        let pairing_window = crate::PairingWindowNs::try_from(manifest.header.pairing_window_ns)
            .map_err(|source| DatasetManifestError::InvalidPairingWindow { source })?;

        require_manifest_count(
            "total_left",
            manifest.stats.total_left,
            count_as_u64("scanned_left", frames.left.len())?,
        )?;
        require_manifest_count(
            "total_right",
            manifest.stats.total_right,
            count_as_u64("scanned_right", frames.right.len())?,
        )?;
        require_manifest_count(
            "entries",
            manifest.stats.total_left,
            count_as_u64("entries", manifest.entries.len())?,
        )?;

        let available_frames: HashMap<&str, (format::FrameKind, i64)> = frames
            .left
            .iter()
            .map(|frame| {
                (
                    frame.path.as_str(),
                    (format::FrameKind::MonoLeft, frame.timestamp_ns),
                )
            })
            .chain(frames.right.iter().map(|frame| {
                (
                    frame.path.as_str(),
                    (format::FrameKind::MonoRight, frame.timestamp_ns),
                )
            }))
            .collect();
        let mut seen_left = HashSet::with_capacity(manifest.entries.len());
        let mut seen_right = HashSet::with_capacity(manifest.entries.len());
        let mut pairs = Vec::with_capacity(manifest.entries.len().min(frames.right.len()));
        let mut previous_left = None;
        let mut previous_right = None;
        let mut left_orphans = 0_usize;
        let mut outside_window = 0_usize;

        for (entry_index, entry) in manifest.entries.into_iter().enumerate() {
            let left = validate_manifest_frame_ref(
                entry.left,
                format::FrameKind::MonoLeft,
                &available_frames,
                entry_index,
                DatasetFrameRole::Left,
            )?;
            insert_unique_frame_ref(&mut seen_left, &left, entry_index, DatasetFrameRole::Left)?;
            require_increasing_timestamp(
                &mut previous_left,
                left.timestamp,
                entry_index,
                DatasetFrameRole::Left,
            )?;

            match entry.pairing {
                ManifestPairing::Paired { right, delta_ns } => {
                    let right = validate_manifest_frame_ref(
                        right,
                        format::FrameKind::MonoRight,
                        &available_frames,
                        entry_index,
                        DatasetFrameRole::Right,
                    )?;
                    insert_unique_frame_ref(
                        &mut seen_right,
                        &right,
                        entry_index,
                        DatasetFrameRole::Right,
                    )?;
                    require_increasing_timestamp(
                        &mut previous_right,
                        right.timestamp,
                        entry_index,
                        DatasetFrameRole::Right,
                    )?;
                    let actual_delta = left
                        .timestamp
                        .as_nanos()
                        .abs_diff(right.timestamp.as_nanos());
                    if delta_ns != actual_delta {
                        return Err(DatasetManifestError::PairDeltaMismatch {
                            entry_index,
                            declared_ns: delta_ns,
                            actual_ns: actual_delta,
                        });
                    }
                    if actual_delta > pairing_window.as_ns() {
                        return Err(DatasetManifestError::PairOutsideWindow {
                            entry_index,
                            delta_ns: actual_delta,
                            pairing_window_ns: pairing_window.as_ns(),
                        });
                    }
                    pairs.push(DatasetPairIndex { left, right });
                }
                ManifestPairing::MissingRight { reason } => {
                    left_orphans += 1;
                    if matches!(reason, PairReason::OutsideWindow) {
                        outside_window += 1;
                    }
                }
            }
        }

        let paired_count = pairs.len();
        let right_orphans = frames.right.len().checked_sub(paired_count).ok_or(
            DatasetManifestError::PairedCountExceedsAvailableRightFrames {
                paired_count,
                available_right: frames.right.len(),
            },
        )?;
        require_manifest_count(
            "paired_count",
            manifest.stats.paired_count,
            count_as_u64("paired_count", paired_count)?,
        )?;
        require_manifest_count(
            "left_orphans",
            manifest.stats.left_orphans,
            count_as_u64("left_orphans", left_orphans)?,
        )?;
        require_manifest_count(
            "right_orphans",
            manifest.stats.right_orphans,
            count_as_u64("right_orphans", right_orphans)?,
        )?;
        require_manifest_count(
            "drops_by_reason.outside_window",
            manifest.stats.drops_by_reason.outside_window,
            count_as_u64("outside_window", outside_window)?,
        )?;

        let stats = DatasetStats::from_frames_and_pairs(frames, &pairs);
        Ok(Self {
            stats,
            pairs: pairs.into_boxed_slice(),
        })
    }
}

fn require_manifest_text(
    field: &'static str,
    actual: &str,
    expected: &str,
) -> Result<(), DatasetManifestError> {
    if actual == expected {
        return Ok(());
    }
    Err(DatasetManifestError::HeaderTextMismatch {
        field,
        expected: expected.to_string(),
        actual: actual.to_string(),
    })
}

fn require_manifest_number(
    field: &'static str,
    actual: u32,
    expected: u32,
) -> Result<(), DatasetManifestError> {
    if actual == expected {
        return Ok(());
    }
    Err(DatasetManifestError::HeaderNumberMismatch {
        field,
        expected: u64::from(expected),
        actual: u64::from(actual),
    })
}

fn count_as_u64(field: &'static str, value: usize) -> Result<u64, DatasetManifestError> {
    u64::try_from(value).map_err(|_| DatasetManifestError::CountOverflow { field, value })
}

fn require_manifest_count(
    field: &'static str,
    declared: u64,
    actual: u64,
) -> Result<(), DatasetManifestError> {
    if declared == actual {
        return Ok(());
    }
    Err(DatasetManifestError::CountMismatch {
        field,
        declared,
        actual,
    })
}

fn validate_manifest_frame_ref(
    frame_ref: ManifestFrameRef,
    expected_kind: format::FrameKind,
    available: &HashMap<&str, (format::FrameKind, i64)>,
    entry_index: usize,
    role: DatasetFrameRole,
) -> Result<DatasetIndexFrameRef, DatasetManifestError> {
    let invalid = |source| DatasetManifestError::InvalidFrameReference {
        entry_index,
        role,
        source,
    };
    let Some(filename) = Path::new(&frame_ref.path)
        .file_name()
        .and_then(|filename| filename.to_str())
    else {
        return Err(invalid(DatasetFrameReferenceError::NonCanonicalPath {
            path: frame_ref.path,
        }));
    };
    let canonical_path = format!("{}/{}", format::FRAMES_DIR, filename);
    if frame_ref.path != canonical_path {
        return Err(invalid(DatasetFrameReferenceError::NonCanonicalPath {
            path: frame_ref.path,
        }));
    }
    let Some(&(actual_kind, scanned_timestamp_ns)) = available.get(frame_ref.path.as_str()) else {
        return Err(invalid(DatasetFrameReferenceError::MissingFromDataset {
            path: frame_ref.path,
        }));
    };
    if actual_kind != expected_kind {
        return Err(invalid(DatasetFrameReferenceError::WrongFrameKind {
            path: frame_ref.path,
            expected: expected_kind,
            actual: actual_kind,
        }));
    }
    if scanned_timestamp_ns != frame_ref.timestamp_ns {
        return Err(invalid(DatasetFrameReferenceError::TimestampMismatch {
            path: frame_ref.path,
            declared_ns: frame_ref.timestamp_ns,
            filename_ns: scanned_timestamp_ns,
        }));
    }
    Ok(DatasetIndexFrameRef {
        timestamp: Timestamp::from_nanos(frame_ref.timestamp_ns),
        path: Arc::from(frame_ref.path),
    })
}

fn insert_unique_frame_ref(
    seen: &mut HashSet<Arc<str>>,
    frame_ref: &DatasetIndexFrameRef,
    entry_index: usize,
    role: DatasetFrameRole,
) -> Result<(), DatasetManifestError> {
    if seen.insert(Arc::clone(&frame_ref.path)) {
        return Ok(());
    }
    Err(DatasetManifestError::DuplicateFrameReference {
        entry_index,
        role,
        path: frame_ref.path.to_string(),
    })
}

fn require_increasing_timestamp(
    previous: &mut Option<Timestamp>,
    current: Timestamp,
    entry_index: usize,
    role: DatasetFrameRole,
) -> Result<(), DatasetManifestError> {
    if let Some(previous_timestamp) = *previous
        && current.as_nanos() <= previous_timestamp.as_nanos()
    {
        return Err(DatasetManifestError::NonIncreasingTimestamp {
            entry_index,
            role,
            previous_ns: previous_timestamp.as_nanos(),
            current_ns: current.as_nanos(),
        });
    }
    *previous = Some(current);
    Ok(())
}

fn write_manifest(state: &WriterState) -> Result<(), DatasetError> {
    let meta = read_meta(&state.dataset_dir)?;
    let mono = meta.mono.ok_or(DatasetError::InvalidConfig {
        msg: "meta.json missing mono config",
    })?;

    let FrameSet {
        mono_dimensions: _,
        left,
        right,
    } = scan_frames_with_depth(
        &state.frames_dir,
        mono.width,
        mono.height,
        meta.depth.as_ref(),
    )?;

    let left_period = compute_period_ns(&left);
    let gate = left_period.map(|p| p / 4).filter(|p| *p > 0);
    let deltas = collect_deltas(&left, &right, gate);
    let delta_stats = build_delta_stats(&deltas);
    let pairing_window_ns = compute_pairing_window_ns(&deltas, delta_stats.as_ref(), left_period);

    let (entries, paired_count, left_orphans, right_orphans, outside_window) =
        pair_entries(&left, &right, pairing_window_ns);

    let depth_entries = meta
        .depth
        .as_ref()
        .map(|depth| recorded_depth_entries(state, depth))
        .transpose()?;

    let manifest = Manifest {
        header: ManifestHeader {
            dataset_id: dataset_id(&state.dataset_dir),
            created_at: meta.created,
            device: meta.device,
            format: "raw".to_string(),
            width: mono.width,
            height: mono.height,
            fps: mono.fps,
            timebase: "device_ns".to_string(),
            pairing_policy: "time_symmetric".to_string(),
            pairing_window_ns,
        },
        stats: ManifestStats {
            total_left: left.len() as u64,
            total_right: right.len() as u64,
            paired_count,
            left_orphans,
            right_orphans,
            drops_by_reason: DropStats {
                spool_full: state.dropped.load(Ordering::Relaxed),
                write_fail: state.write_failed.load(Ordering::Relaxed),
                parse_fail: 0,
                size_mismatch: 0,
                outside_window,
            },
            delta_stats,
        },
        entries,
        depth_entries,
    };

    let manifest_path = state.dataset_dir.join(format::MANIFEST_FILE);
    let manifest_file =
        std::fs::File::create(&manifest_path).map_err(|e| DatasetError::WriteFile {
            path: manifest_path.clone(),
            source: e,
        })?;
    serde_json::to_writer_pretty(manifest_file, &manifest)
        .map_err(|e| DatasetError::SerializeJson { source: e })?;
    Ok(())
}

fn recorded_depth_entries(
    state: &WriterState,
    depth: &DepthMeta,
) -> Result<Vec<ManifestFrameRef>, DatasetError> {
    let (_, expected_bytes) = frame_layout(
        "depth",
        depth.width,
        depth.height,
        std::mem::size_of::<f32>() as u64,
    )?;
    let mut recorded = state
        .recorded_depth
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .clone();
    recorded.sort_unstable_by_key(|entry| entry.timestamp_ns);

    recorded
        .into_iter()
        .map(|entry| {
            let filename = format::frame_name(entry.timestamp_ns, format::FrameKind::Depth);
            let path = state.frames_dir.join(&filename);
            let actual_bytes = std::fs::metadata(&path)
                .map_err(|source| DatasetError::ReadFile {
                    path: path.clone(),
                    source,
                })?
                .len();
            if actual_bytes != expected_bytes {
                return Err(DatasetError::FrameSizeMismatch {
                    path,
                    expected_bytes,
                    actual_bytes,
                });
            }
            Ok(ManifestFrameRef {
                timestamp_ns: entry.timestamp_ns,
                path: format!("{}/{}", format::FRAMES_DIR, filename),
            })
        })
        .collect()
}

fn read_meta(dataset_dir: &Path) -> Result<Meta, DatasetError> {
    let meta_path = dataset_dir.join(format::META_FILE);
    let meta_file = std::fs::File::open(&meta_path).map_err(|e| DatasetError::ReadFile {
        path: meta_path.clone(),
        source: e,
    })?;
    serde_json::from_reader(meta_file).map_err(|e| DatasetError::DeserializeJson { source: e })
}

fn read_manifest(dataset_dir: &Path) -> Result<Manifest, DatasetError> {
    let manifest_path = dataset_dir.join(format::MANIFEST_FILE);
    let manifest_file =
        std::fs::File::open(&manifest_path).map_err(|e| DatasetError::ReadFile {
            path: manifest_path.clone(),
            source: e,
        })?;
    serde_json::from_reader(manifest_file).map_err(|e| DatasetError::DeserializeJson { source: e })
}

fn read_calibration_with_imu_override(
    dataset_dir: &Path,
    imu_override: Option<&ImuCalibration>,
) -> Result<Calibration, DatasetError> {
    let calibration_path = dataset_dir.join(format::CALIBRATION_FILE);
    let calibration_file =
        std::fs::File::open(&calibration_path).map_err(|e| DatasetError::ReadFile {
            path: calibration_path.clone(),
            source: e,
        })?;
    if let Some(imu_override) = imu_override {
        let mut calibration_value: serde_json::Value = serde_json::from_reader(calibration_file)
            .map_err(|e| DatasetError::DeserializeJson { source: e })?;
        let calibration_object =
            calibration_value
                .as_object_mut()
                .ok_or(DatasetError::InvalidConfig {
                    msg: "calibration.json must be a JSON object",
                })?;
        calibration_object.insert(
            "imu".to_string(),
            serde_json::to_value(imu_override)
                .map_err(|e| DatasetError::SerializeJson { source: e })?,
        );
        serde_json::from_value(calibration_value)
            .map_err(|e| DatasetError::DeserializeJson { source: e })
    } else {
        serde_json::from_reader(calibration_file)
            .map_err(|e| DatasetError::DeserializeJson { source: e })
    }
}

fn require_calibration_dimensions<'a>(
    meta: &'a Meta,
    calibration: &crate::CalibrationBundle,
) -> Result<&'a MonoMeta, DatasetError> {
    let mono = meta.mono.as_ref().ok_or(DatasetError::InvalidConfig {
        msg: "meta.json missing mono config",
    })?;
    let metadata = FrameDimensions::try_new(mono.width, mono.height).map_err(|source| {
        DatasetError::InvalidFrameDimensions {
            stream: "mono metadata",
            source,
        }
    })?;
    let calibrated = calibration.stereo().dimensions();
    if metadata != calibrated {
        return Err(DatasetError::CalibrationDimensionsMismatch {
            metadata,
            calibration: calibrated,
        });
    }
    Ok(mono)
}

fn scan_frames_with_depth(
    frames_dir: &Path,
    width: u32,
    height: u32,
    depth: Option<&DepthMeta>,
) -> Result<FrameSet, DatasetError> {
    let (mono_dimensions, mono_expected_len) = frame_layout("mono", width, height, 1)?;
    let mut frames = FrameSet {
        mono_dimensions,
        left: Vec::new(),
        right: Vec::new(),
    };
    let depth_expected_len = depth
        .map(|meta| {
            frame_layout(
                "depth",
                meta.width,
                meta.height,
                std::mem::size_of::<f32>() as u64,
            )
            .map(|(_, expected_bytes)| expected_bytes)
        })
        .transpose()?;

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
        let file_type = entry.file_type().map_err(|source| DatasetError::ReadFile {
            path: path.clone(),
            source,
        })?;
        if !file_type.is_file() {
            continue;
        }
        if path.extension() != Some(std::ffi::OsStr::new("raw")) {
            continue;
        }
        let filename = path
            .file_name()
            .and_then(|filename| filename.to_str())
            .ok_or_else(|| DatasetError::NonUnicodeFrameFilename { path: path.clone() })?;
        let parsed = format::parse_frame_filename(filename).map_err(|source| {
            DatasetError::InvalidFrameFilename {
                path: path.clone(),
                source,
            }
        })?;
        let expected_len = match parsed.kind() {
            format::FrameKind::MonoLeft | format::FrameKind::MonoRight => mono_expected_len,
            format::FrameKind::Depth => {
                depth_expected_len.ok_or_else(|| DatasetError::UnexpectedFrameKind {
                    path: path.clone(),
                    kind: parsed.kind(),
                })?
            }
        };
        let metadata = entry.metadata().map_err(|source| DatasetError::ReadFile {
            path: path.clone(),
            source,
        })?;
        if metadata.len() != expected_len {
            return Err(DatasetError::FrameSizeMismatch {
                path,
                expected_bytes: expected_len,
                actual_bytes: metadata.len(),
            });
        }

        let target = match parsed.kind() {
            format::FrameKind::MonoLeft => Some(&mut frames.left),
            format::FrameKind::MonoRight => Some(&mut frames.right),
            format::FrameKind::Depth => None,
        };
        if let Some(target) = target {
            target.push(FrameInfo {
                timestamp_ns: parsed.timestamp_ns(),
                path: format!("{}/{}", format::FRAMES_DIR, filename),
            });
        }
    }

    frames.left.sort_unstable_by_key(|frame| frame.timestamp_ns);
    frames
        .right
        .sort_unstable_by_key(|frame| frame.timestamp_ns);
    Ok(frames)
}

fn frame_layout(
    stream: &'static str,
    width: u32,
    height: u32,
    bytes_per_pixel: u64,
) -> Result<(FrameDimensions, u64), DatasetError> {
    let dimensions = FrameDimensions::try_new(width, height)
        .map_err(|source| DatasetError::InvalidFrameDimensions { stream, source })?;
    let pixels =
        u64::try_from(dimensions.area()).map_err(|_| DatasetError::FrameByteLengthOverflow {
            stream,
            width,
            height,
        })?;
    let expected_bytes =
        pixels
            .checked_mul(bytes_per_pixel)
            .ok_or(DatasetError::FrameByteLengthOverflow {
                stream,
                width,
                height,
            })?;
    Ok((dimensions, expected_bytes))
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
            if let Some(gate_ns) = gate {
                if delta > gate_ns {
                    continue;
                }
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
        return (left_period.unwrap_or(0) / 4).max(1);
    }
    let mut sorted = deltas.to_vec();
    sorted.sort_unstable();
    let median = median_u64(&sorted);
    let mad = median_absolute_deviation(&sorted, median);
    let p99 = stats
        .map(|s| s.p99)
        .unwrap_or_else(|| sorted.last().copied().unwrap_or(0));
    let mut window = p99.max(median.saturating_add(mad.saturating_mul(6)));
    if let Some(period) = left_period {
        if period > 0 {
            window = window.min(period / 4);
        }
    }
    window.max(1)
}

fn pair_entries(
    left: &[FrameInfo],
    right: &[FrameInfo],
    window_ns: u64,
) -> (Vec<ManifestEntry>, u64, u64, u64, u64) {
    let mut entries = Vec::with_capacity(left.len());
    let mut right_used = vec![false; right.len()];
    let mut paired_count = 0u64;
    let mut left_orphans = 0u64;
    let mut outside_window = 0u64;
    let has_right = !right.is_empty();

    let mut right_idx = 0usize;
    for left_frame in left {
        while right_idx + 1 < right.len() && right[right_idx].timestamp_ns < left_frame.timestamp_ns
        {
            right_idx += 1;
        }

        let mut left_candidate = right_idx as i64 - 1;
        while left_candidate >= 0 && right_used[left_candidate as usize] {
            left_candidate -= 1;
        }
        let left_candidate = if left_candidate >= 0 {
            Some(left_candidate as usize)
        } else {
            None
        };

        let mut right_candidate = right_idx;
        while right_candidate < right.len() && right_used[right_candidate] {
            right_candidate += 1;
        }
        let right_candidate = if right_candidate < right.len() {
            Some(right_candidate)
        } else {
            None
        };

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
            if window_ns > 0 && delta > window_ns {
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
                right_used[idx] = true;
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

    let right_orphans = right_used.iter().filter(|used| !**used).count() as u64;
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
        (a / 2)
            .saturating_add(b / 2)
            .saturating_add((a % 2 + b % 2) / 2)
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
    let mut deviations: Vec<u64> = sorted.iter().map(|value| value.abs_diff(median)).collect();
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
    use crate::{FrameId, PairingWindowNs, Timestamp};

    fn unique_temp_path(name: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        std::env::temp_dir().join(format!("kiko-{name}-{}-{nonce}", std::process::id()))
    }

    fn test_meta() -> Meta {
        Meta {
            created: "now".to_string(),
            device: "test".to_string(),
            mono: Some(MonoMeta {
                width: 2,
                height: 2,
                fps: 10,
            }),
            depth: None,
            imu: None,
        }
    }

    fn test_meta_with_depth() -> Meta {
        let mut meta = test_meta();
        meta.depth = Some(DepthMeta {
            width: 2,
            height: 2,
            fps: 10,
            encoding: "f32_m".to_string(),
        });
        meta
    }

    fn test_depth(timestamp_ns: i64) -> DepthImage {
        DepthImage::new(
            FrameId::new(timestamp_ns.unsigned_abs()),
            Timestamp::from_nanos(timestamp_ns),
            2,
            2,
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .expect("depth image")
    }

    fn test_calibration() -> Calibration {
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
            imu: None,
        }
    }

    fn manifest_validation_fixture() -> (Manifest, Meta, FrameSet) {
        let meta = test_meta();
        let frames = FrameSet {
            mono_dimensions: FrameDimensions::try_new(2, 2).expect("dimensions"),
            left: vec![
                FrameInfo {
                    timestamp_ns: 100,
                    path: "frames/100_mono_left.raw".to_string(),
                },
                FrameInfo {
                    timestamp_ns: 200,
                    path: "frames/200_mono_left.raw".to_string(),
                },
            ],
            right: vec![
                FrameInfo {
                    timestamp_ns: 104,
                    path: "frames/104_mono_right.raw".to_string(),
                },
                FrameInfo {
                    timestamp_ns: 204,
                    path: "frames/204_mono_right.raw".to_string(),
                },
            ],
        };
        let manifest = Manifest {
            header: ManifestHeader {
                dataset_id: "fixture".to_string(),
                created_at: meta.created.clone(),
                device: meta.device.clone(),
                format: "raw".to_string(),
                width: 2,
                height: 2,
                fps: 10,
                timebase: "device_ns".to_string(),
                pairing_policy: "time_symmetric".to_string(),
                pairing_window_ns: 10,
            },
            stats: ManifestStats {
                total_left: 2,
                total_right: 2,
                paired_count: 2,
                left_orphans: 0,
                right_orphans: 0,
                drops_by_reason: DropStats {
                    spool_full: 0,
                    write_fail: 0,
                    parse_fail: 0,
                    size_mismatch: 0,
                    outside_window: 0,
                },
                delta_stats: None,
            },
            entries: vec![
                ManifestEntry {
                    left: ManifestFrameRef {
                        timestamp_ns: 100,
                        path: "frames/100_mono_left.raw".to_string(),
                    },
                    pairing: ManifestPairing::Paired {
                        right: ManifestFrameRef {
                            timestamp_ns: 104,
                            path: "frames/104_mono_right.raw".to_string(),
                        },
                        delta_ns: 4,
                    },
                },
                ManifestEntry {
                    left: ManifestFrameRef {
                        timestamp_ns: 200,
                        path: "frames/200_mono_left.raw".to_string(),
                    },
                    pairing: ManifestPairing::Paired {
                        right: ManifestFrameRef {
                            timestamp_ns: 204,
                            path: "frames/204_mono_right.raw".to_string(),
                        },
                        delta_ns: 4,
                    },
                },
            ],
            depth_entries: None,
        };
        (manifest, meta, frames)
    }

    fn test_pair() -> StereoPair {
        let left = Frame::new(
            SensorId::StereoLeft,
            FrameId::new(1),
            Timestamp::from_nanos(100),
            2,
            2,
            vec![1; 4],
        )
        .expect("left frame");
        let right = Frame::new(
            SensorId::StereoRight,
            FrameId::new(2),
            Timestamp::from_nanos(104),
            2,
            2,
            vec![2; 4],
        )
        .expect("right frame");
        StereoPair::try_new(
            left,
            right,
            PairingWindowNs::new(10).expect("pairing window"),
        )
        .expect("stereo pair")
    }

    #[test]
    fn exact_sync_still_produces_a_readable_nonzero_pairing_window() {
        let deltas = [0_u64, 0, 0];
        let stats = build_delta_stats(&deltas).expect("delta stats");
        assert_eq!(compute_pairing_window_ns(&deltas, Some(&stats), None), 1);
    }

    #[test]
    fn median_does_not_overflow_at_u64_max() {
        assert_eq!(median_u64(&[u64::MAX, u64::MAX]), u64::MAX);
    }

    #[test]
    fn frame_filename_parser_returns_typed_kind_and_timestamp_source() {
        let parsed = format::parse_frame_filename("-42_mono_right.raw").expect("filename");
        assert_eq!(parsed.timestamp_ns(), -42);
        assert_eq!(parsed.kind(), format::FrameKind::MonoRight);

        let invalid_timestamp = format::parse_frame_filename("not-an-i64_mono_left.raw")
            .expect_err("timestamp must be parsed once at the filename boundary");
        assert!(matches!(
            invalid_timestamp,
            format::FrameFilenameError::InvalidTimestamp { .. }
        ));
        assert!(std::error::Error::source(&invalid_timestamp).is_some());

        assert!(matches!(
            format::parse_frame_filename("42_color.raw"),
            Err(format::FrameFilenameError::UnknownFrameKind { value }) if value == "color"
        ));
    }

    #[test]
    fn frame_scan_ignores_non_raw_files_but_rejects_malformed_raw_files() {
        let root = unique_temp_path("scan-malformed-raw");
        let frames_dir = root.join(format::FRAMES_DIR);
        std::fs::create_dir_all(&frames_dir).expect("frames directory");
        std::fs::write(frames_dir.join("notes.txt"), b"not dataset data").expect("unrelated file");
        let malformed_path = frames_dir.join("not-an-i64_mono_left.raw");
        std::fs::write(&malformed_path, [0_u8; 4]).expect("malformed frame");

        let error = scan_frames_with_depth(&frames_dir, 2, 2, None)
            .expect_err("malformed raw files must not disappear into counters");
        assert!(matches!(
            &error,
            DatasetError::InvalidFrameFilename { path, .. } if path == &malformed_path
        ));
        let filename = std::error::Error::source(&error).expect("filename source");
        assert!(
            filename.source().is_some(),
            "integer parse source must survive"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn frame_scan_rejects_wrong_size_and_unconfigured_depth() {
        let root = unique_temp_path("scan-invalid-frame");
        let frames_dir = root.join(format::FRAMES_DIR);
        std::fs::create_dir_all(&frames_dir).expect("frames directory");
        let short_path = frames_dir.join("1_mono_left.raw");
        std::fs::write(&short_path, [0_u8; 3]).expect("short frame");

        let size_error = scan_frames_with_depth(&frames_dir, 2, 2, None)
            .expect_err("wrong-sized raw frame must fail the scan");
        assert!(matches!(
            size_error,
            DatasetError::FrameSizeMismatch {
                path,
                expected_bytes: 4,
                actual_bytes: 3,
            } if path == short_path
        ));

        std::fs::remove_file(&short_path).expect("remove short frame");
        let depth_path = frames_dir.join("2_depth.raw");
        std::fs::write(&depth_path, [0_u8; 16]).expect("depth frame");
        let depth_error = scan_frames_with_depth(&frames_dir, 2, 2, None)
            .expect_err("depth frames require declared depth metadata");
        assert!(matches!(
            depth_error,
            DatasetError::UnexpectedFrameKind {
                path,
                kind: format::FrameKind::Depth,
            } if path == depth_path
        ));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn frame_scan_sorts_once_for_pairing_and_stats() {
        let root = unique_temp_path("scan-sorts");
        let frames_dir = root.join(format::FRAMES_DIR);
        std::fs::create_dir_all(&frames_dir).expect("frames directory");
        for filename in [
            "20_mono_left.raw",
            "10_mono_left.raw",
            "21_mono_right.raw",
            "11_mono_right.raw",
        ] {
            std::fs::write(frames_dir.join(filename), [0_u8; 4]).expect("frame");
        }

        let frames = scan_frames_with_depth(&frames_dir, 2, 2, None).expect("scan");
        assert_eq!(
            frames
                .left
                .iter()
                .map(|frame| frame.timestamp_ns)
                .collect::<Vec<_>>(),
            vec![10, 20]
        );
        assert_eq!(
            frames
                .right
                .iter()
                .map(|frame| frame.timestamp_ns)
                .collect::<Vec<_>>(),
            vec![11, 21]
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn manifest_parses_into_a_trustworthy_runtime_index() {
        let (manifest, meta, frames) = manifest_validation_fixture();
        let index = DatasetIndex::try_from_manifest(manifest, &meta, &frames).expect("index");

        assert_eq!(index.stats.left_count, 2);
        assert_eq!(index.stats.right_count, 2);
        assert_eq!(index.stats.paired_count, 2);
        assert_eq!(index.pairs.len(), 2);
        assert_eq!(index.pairs[0].left.timestamp.as_nanos(), 100);
        assert_eq!(
            index.pairs[0].left.path.as_ref(),
            "frames/100_mono_left.raw"
        );
        assert_eq!(index.pairs[0].right.timestamp.as_nanos(), 104);
        assert_eq!(
            index.pairs[0].right.path.as_ref(),
            "frames/104_mono_right.raw"
        );
    }

    #[test]
    fn manifest_orphans_are_counted_but_excluded_from_the_runtime_pair_index() {
        let (mut manifest, meta, frames) = manifest_validation_fixture();
        manifest.entries[1].pairing = ManifestPairing::MissingRight {
            reason: PairReason::RightExhausted,
        };
        manifest.stats.paired_count = 1;
        manifest.stats.left_orphans = 1;
        manifest.stats.right_orphans = 1;

        let index = DatasetIndex::try_from_manifest(manifest, &meta, &frames).expect("index");

        assert_eq!(index.stats.left_count, 2);
        assert_eq!(index.stats.right_count, 2);
        assert_eq!(index.stats.paired_count, 1);
        assert_eq!(index.pairs.len(), 1);
        assert_eq!(index.pairs[0].left.timestamp.as_nanos(), 100);
        assert_eq!(index.pairs[0].right.timestamp.as_nanos(), 104);
    }

    #[test]
    fn manifest_rejects_noncanonical_and_wrong_role_paths() {
        let (mut escaped, meta, frames) = manifest_validation_fixture();
        escaped.entries[0].left.path = "../100_mono_left.raw".to_string();
        assert!(matches!(
            DatasetIndex::try_from_manifest(escaped, &meta, &frames),
            Err(DatasetManifestError::InvalidFrameReference {
                entry_index: 0,
                role: DatasetFrameRole::Left,
                source: DatasetFrameReferenceError::NonCanonicalPath { .. },
            })
        ));

        let (mut wrong_role, meta, frames) = manifest_validation_fixture();
        wrong_role.entries[0].left = ManifestFrameRef {
            timestamp_ns: 104,
            path: "frames/104_mono_right.raw".to_string(),
        };
        assert!(matches!(
            DatasetIndex::try_from_manifest(wrong_role, &meta, &frames),
            Err(DatasetManifestError::InvalidFrameReference {
                entry_index: 0,
                role: DatasetFrameRole::Left,
                source: DatasetFrameReferenceError::WrongFrameKind {
                    expected: format::FrameKind::MonoLeft,
                    actual: format::FrameKind::MonoRight,
                    ..
                },
            })
        ));
    }

    #[test]
    fn manifest_rejects_references_absent_from_the_scanned_dataset() {
        let (mut manifest, meta, frames) = manifest_validation_fixture();
        manifest.entries[0].left.path = "frames/not-an-i64_mono_left.raw".to_string();

        assert!(matches!(
            DatasetIndex::try_from_manifest(manifest, &meta, &frames),
            Err(DatasetManifestError::InvalidFrameReference {
                entry_index: 0,
                role: DatasetFrameRole::Left,
                source: DatasetFrameReferenceError::MissingFromDataset { .. },
            })
        ));
    }

    #[test]
    fn manifest_rejects_timestamp_delta_window_and_order_mismatches() {
        let (mut timestamp, meta, frames) = manifest_validation_fixture();
        timestamp.entries[0].left.timestamp_ns = 99;
        assert!(matches!(
            DatasetIndex::try_from_manifest(timestamp, &meta, &frames),
            Err(DatasetManifestError::InvalidFrameReference {
                source: DatasetFrameReferenceError::TimestampMismatch { .. },
                ..
            })
        ));

        let (mut delta, meta, frames) = manifest_validation_fixture();
        let ManifestPairing::Paired { delta_ns, .. } = &mut delta.entries[0].pairing else {
            panic!("fixture pair");
        };
        *delta_ns = 5;
        assert!(matches!(
            DatasetIndex::try_from_manifest(delta, &meta, &frames),
            Err(DatasetManifestError::PairDeltaMismatch {
                entry_index: 0,
                declared_ns: 5,
                actual_ns: 4,
            })
        ));

        let (mut outside, meta, frames) = manifest_validation_fixture();
        outside.header.pairing_window_ns = 3;
        assert!(matches!(
            DatasetIndex::try_from_manifest(outside, &meta, &frames),
            Err(DatasetManifestError::PairOutsideWindow {
                entry_index: 0,
                delta_ns: 4,
                pairing_window_ns: 3,
            })
        ));

        let (mut unordered, meta, frames) = manifest_validation_fixture();
        unordered.entries.swap(0, 1);
        assert!(matches!(
            DatasetIndex::try_from_manifest(unordered, &meta, &frames),
            Err(DatasetManifestError::NonIncreasingTimestamp {
                entry_index: 1,
                role: DatasetFrameRole::Left,
                previous_ns: 200,
                current_ns: 100,
            })
        ));
    }

    #[test]
    fn manifest_rejects_duplicate_missing_and_false_count_claims() {
        let (mut duplicate, meta, frames) = manifest_validation_fixture();
        duplicate.entries[1].left = duplicate.entries[0].left.clone();
        assert!(matches!(
            DatasetIndex::try_from_manifest(duplicate, &meta, &frames),
            Err(DatasetManifestError::DuplicateFrameReference {
                entry_index: 1,
                role: DatasetFrameRole::Left,
                ..
            })
        ));

        let (mut missing, meta, frames) = manifest_validation_fixture();
        missing.entries[0].left = ManifestFrameRef {
            timestamp_ns: 300,
            path: "frames/300_mono_left.raw".to_string(),
        };
        assert!(matches!(
            DatasetIndex::try_from_manifest(missing, &meta, &frames),
            Err(DatasetManifestError::InvalidFrameReference {
                source: DatasetFrameReferenceError::MissingFromDataset { .. },
                ..
            })
        ));

        let (mut false_count, meta, frames) = manifest_validation_fixture();
        false_count.stats.paired_count = 1;
        assert!(matches!(
            DatasetIndex::try_from_manifest(false_count, &meta, &frames),
            Err(DatasetManifestError::CountMismatch {
                field: "paired_count",
                declared: 1,
                actual: 2,
            })
        ));
    }

    #[test]
    fn manifest_rejects_header_and_pairing_window_mismatches_with_source() {
        let (mut header, meta, frames) = manifest_validation_fixture();
        header.header.width = 3;
        assert!(matches!(
            DatasetIndex::try_from_manifest(header, &meta, &frames),
            Err(DatasetManifestError::HeaderNumberMismatch {
                field: "width",
                expected: 2,
                actual: 3,
            })
        ));

        let (mut window, meta, frames) = manifest_validation_fixture();
        window.header.pairing_window_ns = 0;
        let error = DatasetIndex::try_from_manifest(window, &meta, &frames)
            .expect_err("zero window must fail");
        assert!(matches!(
            error,
            DatasetManifestError::InvalidPairingWindow { .. }
        ));
        assert!(std::error::Error::source(&error).is_some());
    }

    #[test]
    fn dataset_writer_rejects_existing_output_path() {
        let path = unique_temp_path("writer-existing-output");
        std::fs::create_dir(&path).expect("create existing output");

        let error = DatasetWriter::create(&path, &test_meta(), &test_calibration())
            .expect_err("existing output must be rejected");

        assert!(matches!(
            error,
            DatasetError::OutputAlreadyExists { path: actual } if actual == path
        ));
    }

    #[test]
    fn dataset_writer_rejects_invalid_or_mismatched_calibration_before_side_effects() {
        let invalid_path = unique_temp_path("writer-invalid-calibration");
        let mut invalid = test_calibration();
        invalid.baseline_m = 0.0;
        let error = DatasetWriter::create(&invalid_path, &test_meta(), &invalid)
            .expect_err("invalid calibration must fail");
        assert!(matches!(
            &error,
            DatasetError::InvalidCalibration {
                path,
                source: CalibrationBundleError::InvalidStereo {
                    source: crate::RectifiedStereoError::InvalidBaseline { baseline_m: 0.0 },
                },
            } if path == &invalid_path.join(format::CALIBRATION_FILE)
        ));
        let calibration_source = std::error::Error::source(&error).expect("calibration source");
        assert!(
            calibration_source.source().is_some(),
            "stereo source must survive"
        );
        assert!(!invalid_path.exists());

        let mismatch_path = unique_temp_path("writer-calibration-dimensions");
        let mut mismatch = test_calibration();
        mismatch.left.width = 3;
        mismatch.right.width = 3;
        let error = DatasetWriter::create(&mismatch_path, &test_meta(), &mismatch)
            .expect_err("calibration dimensions must match metadata");
        assert!(matches!(
            error,
            DatasetError::CalibrationDimensionsMismatch {
                metadata,
                calibration,
            } if metadata.width() == 2
                && metadata.height() == 2
                && calibration.width() == 3
                && calibration.height() == 2
        ));
        assert!(!mismatch_path.exists());
    }

    #[test]
    fn manifest_depth_entries_are_sorted_successful_writer_identities() {
        let path = unique_temp_path("depth-manifest-identities");
        let (writer, handle) =
            DatasetWriter::create(&path, &test_meta_with_depth(), &test_calibration())
                .expect("create depth dataset");
        for timestamp_ns in [30, 10, 20] {
            assert_eq!(
                writer.write_depth(&test_depth(timestamp_ns)),
                WriteOutcome::Enqueued
            );
        }

        let unrecorded_path = path
            .join(format::FRAMES_DIR)
            .join(format::frame_name(15, format::FrameKind::Depth));
        std::fs::write(&unrecorded_path, [0_u8; 16])
            .expect("write valid but unrecorded depth payload");

        drop(writer);
        handle.finish().expect("finish depth dataset");

        let manifest = read_manifest(&path).expect("read depth manifest");
        let identities: Vec<(i64, String)> = manifest
            .depth_entries
            .expect("configured depth stream must be represented")
            .into_iter()
            .map(|entry| (entry.timestamp_ns, entry.path))
            .collect();
        assert_eq!(
            identities,
            [
                (10, "frames/10_depth.raw".to_string()),
                (20, "frames/20_depth.raw".to_string()),
                (30, "frames/30_depth.raw".to_string()),
            ]
        );
        assert!(unrecorded_path.exists());

        std::fs::remove_dir_all(path).expect("remove depth identity dataset");
    }

    #[test]
    fn depth_manifest_distinguishes_configured_empty_and_unconfigured_streams() {
        let configured_path = unique_temp_path("depth-manifest-empty");
        let (writer, handle) = DatasetWriter::create(
            &configured_path,
            &test_meta_with_depth(),
            &test_calibration(),
        )
        .expect("create configured depth dataset");
        drop(writer);
        handle.finish().expect("finish empty depth dataset");

        let configured: serde_json::Value = serde_json::from_slice(
            &std::fs::read(configured_path.join(format::MANIFEST_FILE))
                .expect("read configured depth manifest"),
        )
        .expect("parse configured depth manifest");
        assert_eq!(configured["depth_entries"], serde_json::json!([]));
        std::fs::remove_dir_all(configured_path).expect("remove configured depth dataset");

        let unconfigured_path = unique_temp_path("depth-manifest-unconfigured");
        let (writer, handle) =
            DatasetWriter::create(&unconfigured_path, &test_meta(), &test_calibration())
                .expect("create unconfigured dataset");
        drop(writer);
        handle.finish().expect("finish unconfigured dataset");
        let unconfigured: serde_json::Value = serde_json::from_slice(
            &std::fs::read(unconfigured_path.join(format::MANIFEST_FILE))
                .expect("read unconfigured manifest"),
        )
        .expect("parse unconfigured manifest");
        assert!(unconfigured.get("depth_entries").is_none());
        std::fs::remove_dir_all(unconfigured_path).expect("remove unconfigured depth dataset");
    }

    #[test]
    fn stereo_pair_is_dropped_as_one_backpressure_unit() {
        let path = unique_temp_path("writer-pair-backpressure");
        let config = DatasetWriterConfig {
            max_spool_frames: 1,
            max_spool_bytes: 1024,
            flush_batch_frames: 1,
            backpressure: Backpressure::DropNewest,
        };
        let (writer, handle) =
            DatasetWriter::create_with_config(&path, &test_meta(), &test_calibration(), config)
                .expect("writer");

        assert_eq!(
            writer.write_stereo_pair(&test_pair()),
            WriteOutcome::Dropped
        );
        let before_finish = writer.stats();
        assert_eq!(before_finish.frames_enqueued, 0);
        assert_eq!(before_finish.frames_dropped, 2);
        drop(writer);
        let stats = handle.finish().expect("finish writer");
        assert_eq!(stats.frames_written, 0);
    }

    #[test]
    fn stereo_pair_is_written_and_counted_as_two_frames() {
        let path = unique_temp_path("writer-pair-success");
        let (writer, handle) =
            DatasetWriter::create(&path, &test_meta(), &test_calibration()).expect("writer");
        let pair = test_pair();

        assert_eq!(writer.write_stereo_pair(&pair), WriteOutcome::Enqueued);
        drop(writer);
        let stats = handle.finish().expect("finish writer");

        assert_eq!(stats.frames_enqueued, 2);
        assert_eq!(stats.frames_written, 2);
        assert_eq!(stats.frames_dropped, 0);
        assert!(
            path.join(format::FRAMES_DIR)
                .join(format::frame_name(100, format::FrameKind::MonoLeft))
                .is_file()
        );
        assert!(
            path.join(format::FRAMES_DIR)
                .join(format::frame_name(104, format::FrameKind::MonoRight))
                .is_file()
        );
    }

    #[test]
    fn create_new_write_preserves_existing_frame_contents() {
        let path = unique_temp_path("writer-file-collision");
        write_new_file(path.clone(), b"first").expect("first write");

        let error = write_new_file(path.clone(), b"second")
            .expect_err("duplicate filename must be rejected");

        assert!(matches!(
            error,
            DatasetError::WriteFile { source, .. }
                if source.kind() == std::io::ErrorKind::AlreadyExists
        ));
        assert_eq!(std::fs::read(path).expect("read original"), b"first");
    }

    #[test]
    fn writer_reports_duplicate_timestamp_without_publishing_manifest() {
        let path = unique_temp_path("writer-duplicate-timestamp");
        let (writer, handle) =
            DatasetWriter::create(&path, &test_meta(), &test_calibration()).expect("writer");
        let first = Frame::new(
            SensorId::StereoLeft,
            FrameId::new(1),
            Timestamp::from_nanos(100),
            2,
            2,
            vec![1; 4],
        )
        .expect("first frame");
        let duplicate = Frame::new(
            SensorId::StereoLeft,
            FrameId::new(2),
            Timestamp::from_nanos(100),
            2,
            2,
            vec![2; 4],
        )
        .expect("duplicate frame");
        assert_eq!(writer.write_frame(&first), WriteOutcome::Enqueued);
        assert_eq!(writer.write_frame(&duplicate), WriteOutcome::Enqueued);
        drop(writer);

        let error = handle
            .finish()
            .expect_err("duplicate timestamp must fail the writer");

        assert!(matches!(
            error,
            DatasetError::WriteFile { source, .. }
                if source.kind() == std::io::ErrorKind::AlreadyExists
        ));
        assert!(!path.join(format::MANIFEST_FILE).exists());
        assert_eq!(
            std::fs::read(
                path.join(format::FRAMES_DIR)
                    .join(format::frame_name(100, format::FrameKind::MonoLeft))
            )
            .expect("original frame"),
            vec![1; 4]
        );
    }
}
