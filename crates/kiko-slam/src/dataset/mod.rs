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
    pub const MANIFEST_FILE: &str = "manifest.json";
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
            DatasetError::ReadDirectory { path, source } => {
                write!(
                    f,
                    "failed to read directory {}: {}",
                    path.display(),
                    source
                )
            }
            DatasetError::ReadFile { path, source } => {
                write!(f, "failed to read file {}: {}", path.display(), source)
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
            DatasetError::DeserializeJson { source } => {
                write!(f, "failed to deserialize JSON: {}", source)
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

        let state = Arc::new(WriterState::new(
            config,
            path.clone(),
            frames_dir.clone(),
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

        let writer_error = self.state.take_error();
        if let Err(err) = write_manifest(&self.state) {
            return Err(err);
        }

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
                state.write_failed.fetch_add(1, Ordering::Relaxed);
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
        sensor_id,
        frame_id: _,
        timestamp,
        width,
        height,
        data,
    } = frame;
    let filename = format::frame_name(timestamp.as_nanos(), sensor_to_str(sensor_id));
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

#[derive(Debug, Serialize, Deserialize)]
struct ManifestEntry {
    left: ManifestFrameRef,
    right: Option<ManifestFrameRef>,
    delta_ns: Option<u64>,
    status: PairStatus,
    reason: Option<PairReason>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ManifestFrameRef {
    timestamp_ns: i64,
    path: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PairStatus {
    Paired,
    MissingRight,
}

#[derive(Debug, Serialize, Deserialize)]
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
    parse_fail: u64,
    size_mismatch: u64,
}

fn write_manifest(state: &WriterState) -> Result<(), DatasetError> {
    let meta = read_meta(&state.dataset_dir)?;
    let mono = meta.mono.ok_or(DatasetError::InvalidConfig {
        msg: "meta.json missing mono config",
    })?;

    let mut frames = scan_frames(&state.frames_dir, mono.width, mono.height)?;
    let parse_fail = frames.parse_fail;
    let size_mismatch = frames.size_mismatch;
    let mut left = std::mem::take(&mut frames.left);
    let mut right = std::mem::take(&mut frames.right);

    left.sort_by_key(|f| f.timestamp_ns);
    right.sort_by_key(|f| f.timestamp_ns);

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
    let manifest_file = std::fs::File::create(&manifest_path).map_err(|e| DatasetError::WriteFile {
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

fn scan_frames(frames_dir: &Path, width: u32, height: u32) -> Result<FrameSet, DatasetError> {
    let mut frames = FrameSet {
        left: Vec::new(),
        right: Vec::new(),
        parse_fail: 0,
        size_mismatch: 0,
    };
    let expected_len = (width as u64).saturating_mul(height as u64);

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

        let metadata = entry.metadata().map_err(|e| DatasetError::ReadFile {
            path: path.clone(),
            source: e,
        })?;
        if metadata.len() != expected_len {
            frames.size_mismatch += 1;
            continue;
        }

        let rel_path = format!("{}/{}", format::FRAMES_DIR, filename);
        let info = FrameInfo {
            timestamp_ns,
            path: rel_path,
        };
        match sensor.as_str() {
            "mono_left" => frames.left.push(info),
            "mono_right" => frames.right.push(info),
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
    let mut deltas: Vec<i64> = frames
        .windows(2)
        .map(|pair| pair[1].timestamp_ns - pair[0].timestamp_ns)
        .collect();
    deltas.sort_unstable();
    Some(median_i64(&deltas).abs() as u64)
}

fn collect_deltas(
    left: &[FrameInfo],
    right: &[FrameInfo],
    gate: Option<u64>,
) -> Vec<i64> {
    let mut deltas = Vec::new();
    if right.is_empty() {
        return deltas;
    }

    let mut right_idx = 0usize;
    for left_frame in left {
        while right_idx + 1 < right.len()
            && right[right_idx].timestamp_ns < left_frame.timestamp_ns
        {
            right_idx += 1;
        }

        let mut best: Option<i64> = None;
        let candidates = [
            Some(right_idx),
            right_idx.checked_sub(1),
        ];

        for idx in candidates.into_iter().flatten() {
            if idx >= right.len() {
                continue;
            }
            let delta = (right[idx].timestamp_ns - left_frame.timestamp_ns).abs();
            if let Some(gate_ns) = gate {
                if delta as u64 > gate_ns {
                    continue;
                }
            }
            if best.map_or(true, |b| delta < b) {
                best = Some(delta);
            }
        }

        if let Some(delta) = best {
            deltas.push(delta);
        }
    }
    deltas
}

fn build_delta_stats(deltas: &[i64]) -> Option<DeltaStats> {
    if deltas.is_empty() {
        return None;
    }
    let mut sorted = deltas.to_vec();
    sorted.sort_unstable();
    let min = *sorted.first().unwrap() as u64;
    let max = *sorted.last().unwrap() as u64;
    let median = median_i64(&sorted) as u64;
    let p95 = percentile_i64(&sorted, 0.95) as u64;
    let p99 = percentile_i64(&sorted, 0.99) as u64;
    Some(DeltaStats {
        min,
        median,
        p95,
        p99,
        max,
    })
}

fn compute_pairing_window_ns(
    deltas: &[i64],
    stats: Option<&DeltaStats>,
    left_period: Option<u64>,
) -> u64 {
    if deltas.is_empty() {
        return left_period.unwrap_or(0) / 4;
    }
    let mut sorted = deltas.to_vec();
    sorted.sort_unstable();
    let median = median_i64(&sorted);
    let mad = median_absolute_deviation(&sorted, median);
    let p99 = stats.map(|s| s.p99).unwrap_or(sorted.last().copied().unwrap() as u64);
    let mut window = p99.max((median + 6 * mad).max(0) as u64);
    if let Some(period) = left_period {
        if period > 0 {
            window = window.min(period / 4);
        }
    }
    window
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
        while right_idx + 1 < right.len()
            && right[right_idx].timestamp_ns < left_frame.timestamp_ns
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
            let delta = (right[idx].timestamp_ns - left_frame.timestamp_ns).abs() as u64;
            if best_delta.map_or(true, |b| delta < b) {
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
                    right: None,
                    delta_ns: None,
                    status: PairStatus::MissingRight,
                    reason: Some(PairReason::OutsideWindow),
                }
            } else {
                right_used[idx] = true;
                paired_count += 1;
                ManifestEntry {
                    left: ManifestFrameRef {
                        timestamp_ns: left_frame.timestamp_ns,
                        path: left_frame.path.clone(),
                    },
                    right: Some(ManifestFrameRef {
                        timestamp_ns: right[idx].timestamp_ns,
                        path: right[idx].path.clone(),
                    }),
                    delta_ns: Some(delta),
                    status: PairStatus::Paired,
                    reason: None,
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
                right: None,
                delta_ns: None,
                status: PairStatus::MissingRight,
                reason: Some(reason),
            }
        };

        entries.push(entry);
    }

    let right_orphans = right_used.iter().filter(|used| !**used).count() as u64;
    (entries, paired_count, left_orphans, right_orphans, outside_window)
}

fn median_i64(sorted: &[i64]) -> i64 {
    let len = sorted.len();
    if len == 0 {
        return 0;
    }
    if len % 2 == 1 {
        sorted[len / 2]
    } else {
        let a = sorted[len / 2 - 1];
        let b = sorted[len / 2];
        (a + b) / 2
    }
}

fn percentile_i64(sorted: &[i64], pct: f64) -> i64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * pct).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn median_absolute_deviation(sorted: &[i64], median: i64) -> i64 {
    let mut deviations: Vec<i64> = sorted.iter().map(|v| (v - median).abs()).collect();
    deviations.sort_unstable();
    median_i64(&deviations)
}

fn dataset_id(dataset_dir: &Path) -> String {
    dataset_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("dataset")
        .to_string()
}
