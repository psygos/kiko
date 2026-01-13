use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use crate::{Frame, SensorId};

pub mod format {
    pub const FRAMES_DIR: &str = "frames";
    pub const META_FILE: &str = "meta.json";
    pub const CALIBRATION_FILE: &str = "calibration.json";

    pub fn frame_name(timestamp_ns: i64, sensor: &str) -> String {
        format!("{}_{}.png", timestamp_ns, sensor)
    }

    pub fn parse_frame_filename(filename: &str) -> Option<(i64, String)> {
        let stem = filename.strip_suffix(".png")?;
        let (timestamp_str, sensor) = stem.split_once('_')?;
        let timestamp_ns = timestamp_str.parse::<i64>().ok()?;
        Some((timestamp_ns, sensor.to_string()))
    }
}

// Meta Structs
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Meta {
    pub created: String,
    pub device: String,
    pub mono: Option<MonoMeta>,
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

// Calibration structs
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Calibration {
    pub left: CameraIntrinsics,
    pub right: CameraIntrinsics,
    pub baseline_m: f32,
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
    pub queue_capacity: usize,
    pub backpressure: Backpressure,
}

impl Default for DatasetWriterConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 64,
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
    pub writer_failed: bool,
}

impl WriterStats {
    pub fn frames_pending(&self) -> u64 {
        self.frames_enqueued
            .saturating_sub(self.frames_written.saturating_add(self.frames_dropped))
    }
}

#[derive(Debug)]
pub enum DatasetError {
    CreateDirectory {
        path: PathBuf,
        source: std::io::Error,
    },
    InvalidConfig {
        msg: &'static str,
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
    WorkerJoin {
        message: String,
    },
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetError::CreateDirectory { path, source } => {
                write!(
                    f,
                    "failed to create directory {}: {}",
                    path.display(),
                    source
                )
            }
            DatasetError::InvalidConfig { msg } => {
                write!(f, "invalid dataset writer config: {msg}")
            }
            DatasetError::ThreadSpawn { source } => {
                write!(f, "failed to spawn writer thread: {source}")
            }
            DatasetError::WriteFile { path, source } => {
                write!(f, "failed to write file {}: {}", path.display(), source)
            }
            DatasetError::SerializeJson { source } => {
                write!(f, "failed to serialize JSON: {}", source)
            }
            DatasetError::WorkerJoin { message } => {
                write!(f, "writer thread panicked: {message}")
            }
        }
    }
}

impl std::error::Error for DatasetError {}

#[derive(Debug, Clone)]
pub struct DatasetWriter {
    tx: mpsc::SyncSender<Frame>,
    backpressure: Backpressure,
    state: Arc<WriterState>,
}

#[derive(Debug)]
pub struct DatasetWriterHandle {
    handle: Option<thread::JoinHandle<()>>,
    state: Arc<WriterState>,
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
        if config.queue_capacity == 0 {
            return Err(DatasetError::InvalidConfig {
                msg: "queue_capacity must be > 0",
            });
        }

        let path = path.into();
        std::fs::create_dir_all(&path).map_err(|e| DatasetError::CreateDirectory {
            path: path.clone(),
            source: e,
        })?;

        let frames_dir = path.join(format::FRAMES_DIR);
        std::fs::create_dir_all(&frames_dir).map_err(|e| DatasetError::CreateDirectory {
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

        let (tx, rx) = mpsc::sync_channel::<Frame>(config.queue_capacity);
        let state = Arc::new(WriterState::new());
        let state_for_thread = state.clone();

        let handle = thread::Builder::new()
            .name("dataset-writer".to_string())
            .spawn(move || writer_loop(frames_dir, rx, state_for_thread))
            .map_err(|e| DatasetError::ThreadSpawn { source: e })?;

        let writer = Self {
            tx,
            backpressure: config.backpressure,
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
        if self.state.failed.load(Ordering::Acquire) {
            return WriteOutcome::WriterFailed;
        }

        match self.backpressure {
            Backpressure::DropNewest => match self.tx.try_send(frame.clone()) {
                Ok(()) => {
                    self.state.enqueued.fetch_add(1, Ordering::Relaxed);
                    WriteOutcome::Enqueued
                }
                Err(mpsc::TrySendError::Full(_frame)) => {
                    self.state.dropped.fetch_add(1, Ordering::Relaxed);
                    WriteOutcome::Dropped
                }
                Err(mpsc::TrySendError::Disconnected(_frame)) => {
                    self.state.failed.store(true, Ordering::Release);
                    WriteOutcome::WriterFailed
                }
            },
            Backpressure::Block => match self.tx.send(frame.clone()) {
                Ok(()) => {
                    self.state.enqueued.fetch_add(1, Ordering::Relaxed);
                    WriteOutcome::Enqueued
                }
                Err(_err) => {
                    self.state.failed.store(true, Ordering::Release);
                    WriteOutcome::WriterFailed
                }
            },
        }
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
        let handle = self.handle.take().expect("finish called twice");
        handle.join().map_err(|err| DatasetError::WorkerJoin {
            message: panic_message(err),
        })?;

        if let Some(err) = self.state.take_error() {
            return Err(err);
        }

        Ok(self.state.stats())
    }

    pub fn stats(&self) -> WriterStats {
        self.state.stats()
    }
}

#[derive(Debug)]
struct WriterState {
    enqueued: AtomicU64,
    written: AtomicU64,
    dropped: AtomicU64,
    failed: AtomicBool,
    error: Mutex<Option<DatasetError>>,
}

impl WriterState {
    fn new() -> Self {
        Self {
            enqueued: AtomicU64::new(0),
            written: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            failed: AtomicBool::new(false),
            error: Mutex::new(None),
        }
    }

    fn stats(&self) -> WriterStats {
        WriterStats {
            frames_enqueued: self.enqueued.load(Ordering::Relaxed),
            frames_written: self.written.load(Ordering::Relaxed),
            frames_dropped: self.dropped.load(Ordering::Relaxed),
            writer_failed: self.failed.load(Ordering::Acquire),
        }
    }

    fn record_error(&self, err: DatasetError) {
        let mut guard = self
            .error
            .lock()
            .unwrap_or_else(|err| err.into_inner());
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
}

fn writer_loop(frames_dir: PathBuf, rx: mpsc::Receiver<Frame>, state: Arc<WriterState>) {
    for frame in rx {
        match write_frame_to_dir(&frames_dir, frame) {
            Ok(()) => {
                state.written.fetch_add(1, Ordering::Relaxed);
            }
            Err(err) => {
                state.failed.store(true, Ordering::Release);
                state.record_error(err);
                break;
            }
        }
    }
}

fn write_frame_to_dir(frames_dir: &Path, frame: Frame) -> Result<(), DatasetError> {
    let Frame {
        id,
        timestamp,
        width,
        height,
        data,
    } = frame;
    let filename = format::frame_name(timestamp.as_nanos(), sensor_to_str(id));
    let path = frames_dir.join(&filename);

    let img =
        image::GrayImage::from_raw(width, height, data.as_ref().to_vec()).ok_or_else(|| {
            DatasetError::WriteFile {
                path: path.clone(),
                source: std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "failed to create image from frame data",
                ),
            }
        })?;

    img.save(&path).map_err(|e| DatasetError::WriteFile {
        path,
        source: std::io::Error::new(std::io::ErrorKind::Other, e.to_string()),
    })?;

    Ok(())
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
