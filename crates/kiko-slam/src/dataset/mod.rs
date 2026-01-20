use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use crate::{Frame, SensorId};

pub mod format {
    pub const FRAMES_DIR: &str = "frames";
    pub const META_FILE: &str = "meta.json";
    pub const CALIBRATION_FILE: &str = "calibration.json";
    pub const FRAME_SUFFIX: &str = ".raw";

    pub fn frame_name(timestamp_ns: i64, sensor: &str) -> String {
        format!("{}_{}{}", timestamp_ns, sensor, FRAME_SUFFIX)
    }

    pub fn parse_frame_filename(filename: &str) -> Option<(i64, String)> {
        let stem = filename.strip_suffix(FRAME_SUFFIX)?;
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
        if self
            .state
            .open_writers
            .fetch_sub(1, Ordering::AcqRel)
            == 1
        {
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

        let state = Arc::new(WriterState::new(config));
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

    /// Enqueue a frame according to the configured backpressure policy.
    pub fn write_frame(&self, frame: &Frame) -> WriteOutcome {
        if self.state.failed.load(Ordering::Acquire) {
            return WriteOutcome::WriterFailed;
        }

        let bytes = frame.data().len();
        if bytes > self.config.max_spool_bytes {
            self.state.fail(DatasetError::InvalidConfig {
                msg: "frame exceeds max_spool_bytes",
            });
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
                if !self.state.can_accept(&spool, bytes) {
                    self.state.dropped.fetch_add(1, Ordering::Relaxed);
                    self.state
                        .bytes_dropped
                        .fetch_add(bytes as u64, Ordering::Relaxed);
                    return WriteOutcome::Dropped;
                }
            }
            Backpressure::Block => {
                while !self.state.can_accept(&spool, bytes) {
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

        spool.frames += 1;
        spool.bytes += bytes;
        spool.queue.push_back(frame.clone());

        self.state.enqueued.fetch_add(1, Ordering::Relaxed);
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
struct Spool {
    queue: VecDeque<Frame>,
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
    spool: Mutex<Spool>,
    spool_cvar: Condvar,
    enqueued: AtomicU64,
    written: AtomicU64,
    dropped: AtomicU64,
    bytes_enqueued: AtomicU64,
    bytes_written: AtomicU64,
    bytes_dropped: AtomicU64,
    spool_frames: AtomicU64,
    spool_bytes: AtomicU64,
    open_writers: AtomicUsize,
    failed: AtomicBool,
    error: Mutex<Option<DatasetError>>,
}

impl WriterState {
    fn new(config: DatasetWriterConfig) -> Self {
        Self {
            config,
            spool: Mutex::new(Spool::new()),
            spool_cvar: Condvar::new(),
            enqueued: AtomicU64::new(0),
            written: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            bytes_enqueued: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
            bytes_dropped: AtomicU64::new(0),
            spool_frames: AtomicU64::new(0),
            spool_bytes: AtomicU64::new(0),
            open_writers: AtomicUsize::new(1),
            failed: AtomicBool::new(false),
            error: Mutex::new(None),
        }
    }

    fn can_accept(&self, spool: &Spool, bytes: usize) -> bool {
        let next_frames = spool.frames.saturating_add(1);
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
            spool_frames: self.spool_frames.load(Ordering::Relaxed),
            spool_bytes: self.spool_bytes.load(Ordering::Relaxed),
            spool_max_frames: self.config.max_spool_frames as u64,
            spool_max_bytes: self.config.max_spool_bytes as u64,
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

fn writer_loop(frames_dir: PathBuf, state: Arc<WriterState>) {
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
            while let Some(frame) = spool.queue.pop_front() {
                let bytes = frame.data().len();
                spool.frames = spool.frames.saturating_sub(1);
                spool.bytes = spool.bytes.saturating_sub(bytes);
                batch.push(frame);
                if batch.len() >= state.config.flush_batch_frames {
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

        for frame in batch {
            let bytes = frame.data().len() as u64;
            if let Err(err) = write_frame_to_dir(&frames_dir, frame) {
                state.fail(err);
                return;
            }
            state.written.fetch_add(1, Ordering::Relaxed);
            state.bytes_written.fetch_add(bytes, Ordering::Relaxed);
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

    let expected_len = (width as usize).saturating_mul(height as usize);
    if data.len() != expected_len {
        return Err(DatasetError::WriteFile {
            path,
            source: std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "frame data len does not match width * height",
            ),
        });
    }

    std::fs::write(&path, data.as_ref()).map_err(|e| DatasetError::WriteFile {
        path,
        source: e,
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
