use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use crate::{DepthImage, Frame, PairError, SensorId};

pub mod format {
    pub const FRAMES_DIR: &str = "frames";
    pub const META_FILE: &str = "meta.json";
    pub const CALIBRATION_FILE: &str = "calibration.json";
    pub const MANIFEST_FILE: &str = "manifest.json";
    pub const FRAME_SUFFIX: &str = ".raw";

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
        source: crate::FrameError,
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
    InvalidManifest {
        reason: &'static str,
    },
    InvalidManifestStats {
        field: &'static str,
        declared: u64,
        derived: u64,
    },
    ManifestCountOutOfRange {
        field: &'static str,
        value: u64,
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
    WorkerJoin {
        message: String,
    },
    PairingFailed {
        source: PairError,
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
            DatasetError::InvalidFrameDimensions { source } => {
                write!(f, "invalid dataset frame dimensions: {source}")
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
            DatasetError::InvalidManifest { reason } => {
                write!(f, "invalid dataset manifest: {reason}")
            }
            DatasetError::InvalidManifestStats {
                field,
                declared,
                derived,
            } => write!(
                f,
                "invalid dataset manifest statistic {field}: declared {declared}, derived {derived}"
            ),
            DatasetError::ManifestCountOutOfRange { field, value } => write!(
                f,
                "dataset manifest count {field}={value} exceeds the host address space"
            ),
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
            DatasetError::WorkerJoin { message } => {
                write!(f, "writer thread panicked: {message}")
            }
            DatasetError::PairingFailed { source } => {
                write!(f, "dataset pairing failed: {source}")
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
            DatasetError::InvalidFrameDimensions { source }
            | DatasetError::InvalidFrameData { source, .. } => Some(source),
            DatasetError::PairingFailed { source } => Some(source),
            DatasetError::AlreadyExists { .. }
            | DatasetError::InvalidConfig { .. }
            | DatasetError::InvalidFramePath { .. }
            | DatasetError::MissingMonoConfig
            | DatasetError::InvalidFrameFileType { .. }
            | DatasetError::InvalidFrameLength { .. }
            | DatasetError::InvalidManifest { .. }
            | DatasetError::InvalidManifestStats { .. }
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
        self.write_item(
            SpoolItem::Mono(frame.clone()),
            frame.data().len(),
            "frame exceeds max_spool_bytes",
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
            "depth image exceeds max_spool_bytes",
        )
    }

    fn write_item(
        &self,
        item: SpoolItem,
        bytes: usize,
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
        spool.queue.push_back(item);

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
        let Some(handle) = self.handle.take() else {
            return Err(DatasetError::InvalidConfig {
                msg: "finish called twice",
            });
        };
        handle.join().map_err(|err| DatasetError::WorkerJoin {
            message: panic_message(err),
        })?;

        let writer_error = self.state.take_error();
        write_manifest(&self.state)?;

        if let Some(err) = writer_error {
            return Err(err);
        }

        Ok(self.state.stats())
    }

    pub fn stats(&self) -> WriterStats {
        self.state.stats()
    }
}

#[derive(Debug)]
enum SpoolItem {
    Mono(Frame),
    Depth(DepthImage),
}

impl SpoolItem {
    fn bytes_len(&self) -> usize {
        match self {
            SpoolItem::Mono(frame) => frame.data().len(),
            SpoolItem::Depth(depth) => depth
                .depth_m()
                .len()
                .saturating_mul(std::mem::size_of::<f32>()),
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
            while let Some(item) = spool.queue.pop_front() {
                let bytes = item.bytes_len();
                spool.frames = spool.frames.saturating_sub(1);
                spool.bytes = spool.bytes.saturating_sub(bytes);
                batch.push(item);
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

        for item in batch {
            let bytes = item.bytes_len() as u64;
            if let Err(err) = write_item_to_dir(&frames_dir, item) {
                state.write_failed.fetch_add(1, Ordering::Relaxed);
                state.fail(err);
                return;
            }
            state.written.fetch_add(1, Ordering::Relaxed);
            state.bytes_written.fetch_add(bytes, Ordering::Relaxed);
        }
    }
}

fn write_item_to_dir(frames_dir: &Path, item: SpoolItem) -> Result<(), DatasetError> {
    match item {
        SpoolItem::Mono(frame) => write_frame_to_dir(frames_dir, frame),
        SpoolItem::Depth(depth) => write_depth_to_dir(frames_dir, depth),
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
    let filename = format::frame_name(timestamp.as_nanos(), sensor_to_str(sensor_id));
    let path = frames_dir.join(&filename);

    write_new_file(&path, data.as_ref())?;

    Ok(())
}

fn write_depth_to_dir(frames_dir: &Path, depth: DepthImage) -> Result<(), DatasetError> {
    let filename = format::frame_name(depth.timestamp().as_nanos(), "depth");
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
    Ok(())
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

#[derive(Debug)]
struct FrameSet {
    left: Vec<FrameInfo>,
    right: Vec<FrameInfo>,
    depth: Vec<FrameInfo>,
    parse_fail: u64,
    size_mismatch: u64,
}

fn write_manifest(state: &WriterState) -> Result<(), DatasetError> {
    let meta = read_meta(&state.dataset_dir)?;
    let mono = meta.mono.ok_or(DatasetError::InvalidConfig {
        msg: "meta.json missing mono config",
    })?;

    let mut frames = scan_frames_with_depth(
        &state.frames_dir,
        mono.width,
        mono.height,
        meta.depth.as_ref(),
    )?;
    let parse_fail = frames.parse_fail;
    let size_mismatch = frames.size_mismatch;
    let mut left = std::mem::take(&mut frames.left);
    let mut right = std::mem::take(&mut frames.right);
    let mut depth_frames = std::mem::take(&mut frames.depth);

    left.sort_by_key(|f| f.timestamp_ns);
    right.sort_by_key(|f| f.timestamp_ns);
    depth_frames.sort_by_key(|f| f.timestamp_ns);

    let left_period = compute_period_ns(&left);
    let gate = left_period.map(|p| p / 4).filter(|p| *p > 0);
    let deltas = collect_deltas(&left, &right, gate);
    let delta_stats = build_delta_stats(&deltas);
    let pairing_window_ns = compute_pairing_window_ns(&deltas, delta_stats.as_ref(), left_period);

    let (entries, paired_count, left_orphans, right_orphans, outside_window) =
        pair_entries(&left, &right, pairing_window_ns);

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
                parse_fail,
                size_mismatch,
                outside_window,
            },
            delta_stats,
        },
        entries,
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

fn read_calibration(dataset_dir: &Path) -> Result<Calibration, DatasetError> {
    let calibration_path = dataset_dir.join(format::CALIBRATION_FILE);
    let calibration_file =
        std::fs::File::open(&calibration_path).map_err(|e| DatasetError::ReadFile {
            path: calibration_path.clone(),
            source: e,
        })?;
    serde_json::from_reader(calibration_file)
        .map_err(|e| DatasetError::DeserializeJson { source: e })
}

fn scan_frames_with_depth(
    frames_dir: &Path,
    width: u32,
    height: u32,
    depth: Option<&DepthMeta>,
) -> Result<FrameSet, DatasetError> {
    let mut frames = FrameSet {
        left: Vec::new(),
        right: Vec::new(),
        depth: Vec::new(),
        parse_fail: 0,
        size_mismatch: 0,
    };
    let mono_expected_len = (width as u64).saturating_mul(height as u64);
    let depth_expected_len = depth.map(|meta| {
        (meta.width as u64)
            .saturating_mul(meta.height as u64)
            .saturating_mul(std::mem::size_of::<f32>() as u64)
    });

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

        let rel_path = format!("{}/{}", format::FRAMES_DIR, filename);
        let info = FrameInfo {
            timestamp_ns,
            path: rel_path,
        };
        match sensor.as_str() {
            "mono_left" | "mono_right" | "depth" => {
                let metadata = entry.metadata().map_err(|e| DatasetError::ReadFile {
                    path: path.clone(),
                    source: e,
                })?;
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
                if metadata.len() != expected_len {
                    frames.size_mismatch += 1;
                    continue;
                }
                match sensor.as_str() {
                    "mono_left" => frames.left.push(info),
                    "mono_right" => frames.right.push(info),
                    "depth" => frames.depth.push(info),
                    _ => {
                        frames.parse_fail = frames.parse_fail.saturating_add(1);
                    }
                }
            }
            _ => {
                frames.parse_fail += 1;
            }
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
    use crate::{FrameId, Timestamp};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_PATH_ID: AtomicU64 = AtomicU64::new(0);

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

    fn write_exact_pairs(path: &Path, timestamps_ns: &[i64]) {
        let (writer, handle) =
            DatasetWriter::create(path, &meta(), &calibration()).expect("create dataset");
        for (index, &timestamp_ns) in timestamps_ns.iter().enumerate() {
            let left_value = u8::try_from(index.saturating_mul(2)).expect("small test fixture");
            let right_value = left_value.saturating_add(1);
            assert_eq!(
                writer.write_frame(&frame(SensorId::StereoLeft, timestamp_ns, left_value)),
                WriteOutcome::Enqueued
            );
            assert_eq!(
                writer.write_frame(&frame(SensorId::StereoRight, timestamp_ns, right_value)),
                WriteOutcome::Enqueued
            );
        }
        drop(writer);
        handle.finish().expect("finish dataset");
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
    fn duplicate_timestamp_does_not_overwrite_the_first_frame() {
        let path = unique_dataset_path("duplicate-timestamp");
        let (writer, handle) =
            DatasetWriter::create(&path, &meta(), &calibration()).expect("create dataset");
        let first = frame(SensorId::StereoLeft, 7, 1);
        let duplicate = frame(SensorId::StereoLeft, 7, 2);

        assert_eq!(writer.write_frame(&first), WriteOutcome::Enqueued);
        assert_eq!(writer.write_frame(&duplicate), WriteOutcome::Enqueued);
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
            writer.write_frame(&frame(SensorId::StereoLeft, 11, 1)),
            WriteOutcome::Enqueued
        );
        assert_eq!(
            writer.write_frame(&frame(SensorId::StereoRight, 11, 2)),
            WriteOutcome::Enqueued
        );
        drop(writer);
        handle.finish().expect("finish dataset");

        let manifest = read_manifest(&path).expect("read manifest");
        assert_eq!(manifest.header.pairing_window_ns, 0);
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
    fn dataset_open_requires_nonzero_mono_dimensions() {
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

        let mut zero_width = original;
        zero_width["mono"]["width"] = serde_json::json!(0);
        std::fs::write(
            &meta_path,
            serde_json::to_vec_pretty(&zero_width).expect("serialize zero-width metadata"),
        )
        .expect("write zero-width metadata");
        assert!(matches!(
            DatasetReader::open(&path).expect_err("zero frame dimension must fail open"),
            DatasetError::InvalidFrameDimensions {
                source: crate::FrameError::ZeroDimensions {
                    width: 0,
                    height: 2
                }
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
            writer.write_frame(&frame(SensorId::StereoLeft, 11, 1)),
            WriteOutcome::Enqueued
        );
        assert_eq!(
            writer.write_frame(&frame(SensorId::StereoRight, 11, 2)),
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
}
