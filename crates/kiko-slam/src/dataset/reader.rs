use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

use crate::{Frame, FrameError, FrameId, PairingWindowNs, SensorId, StereoPair, Timestamp};

use super::{
    Calibration, DatasetError, FrameInfo, Manifest, ManifestFrameRef, ManifestPairing, format,
    read_calibration, read_manifest, read_meta, scan_frames,
};

#[derive(Clone, Debug)]
struct DatasetFrameRef {
    timestamp_ns: i64,
    path: PathBuf,
}

#[derive(Clone, Debug)]
struct DatasetEntry {
    left: DatasetFrameRef,
    right: Option<DatasetFrameRef>,
}

#[derive(Debug)]
pub struct DatasetReader {
    root: PathBuf,
    meta: super::Meta,
    calibration: Calibration,
    entries: Vec<DatasetEntry>,
    pairing_window: PairingWindowNs,
    left_seq: u64,
    right_seq: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct DatasetReadTimings {
    pub left_read: Duration,
    pub right_read: Duration,
    pub pairing: Duration,
    pub left_bytes: usize,
    pub right_bytes: usize,
}

#[derive(Debug)]
pub struct TimedPair {
    pub pair: StereoPair,
    pub timings: DatasetReadTimings,
}

impl DatasetReader {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, DatasetError> {
        let requested_root = path.into();
        let root =
            std::fs::canonicalize(&requested_root).map_err(|source| DatasetError::ReadFile {
                path: requested_root,
                source,
            })?;
        let meta = read_meta(&root)?;
        let calibration = read_calibration(&root)?;
        let manifest = read_manifest(&root)?;
        let pairing_window_ns = i64::try_from(manifest.header.pairing_window_ns).map_err(|_| {
            DatasetError::InvalidConfig {
                msg: "manifest pairing_window_ns exceeds i64::MAX",
            }
        })?;
        let pairing_window =
            PairingWindowNs::new(pairing_window_ns).map_err(|_| DatasetError::InvalidConfig {
                msg: "manifest pairing_window_ns must be non-negative",
            })?;
        let entries = parse_manifest_entries(&root, manifest)?;
        Ok(Self {
            root,
            meta,
            calibration,
            entries,
            pairing_window,
            left_seq: 0,
            right_seq: 0,
        })
    }

    pub fn meta(&self) -> &super::Meta {
        &self.meta
    }

    pub fn calibration(&self) -> &Calibration {
        &self.calibration
    }

    pub fn stats(&self) -> Result<DatasetStats, DatasetError> {
        let mono = self.meta.mono.as_ref().ok_or(DatasetError::InvalidConfig {
            msg: "meta.json missing mono config",
        })?;
        let frames = scan_frames(&self.root.join(format::FRAMES_DIR), mono.width, mono.height)?;
        Ok(DatasetStats::from_frames(&frames))
    }

    pub fn pairs(&mut self) -> DatasetPairs<'_> {
        DatasetPairs {
            reader: self,
            index: 0,
        }
    }

    pub fn timed_pairs(&mut self) -> DatasetTimedPairs<'_> {
        DatasetTimedPairs {
            reader: self,
            index: 0,
        }
    }

    fn next_left_id(&mut self) -> FrameId {
        let id = self.left_seq;
        self.left_seq = self.left_seq.saturating_add(1);
        FrameId::new(id)
    }

    fn next_right_id(&mut self) -> FrameId {
        let id = self.right_seq;
        self.right_seq = self.right_seq.saturating_add(1);
        FrameId::new(id)
    }
}

pub struct DatasetPairs<'a> {
    reader: &'a mut DatasetReader,
    index: usize,
}

pub struct DatasetTimedPairs<'a> {
    reader: &'a mut DatasetReader,
    index: usize,
}

impl<'a> Iterator for DatasetPairs<'a> {
    type Item = Result<StereoPair, DatasetError>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.reader.entries.len() {
            let entry = self.reader.entries[self.index].clone();
            self.index += 1;

            let right = match entry.right {
                Some(right) => right,
                None => continue,
            };

            let left_frame = match self.reader.read_frame(&entry.left, SensorId::StereoLeft) {
                Ok(frame) => frame,
                Err(err) => return Some(Err(err)),
            };
            let right_frame = match self.reader.read_frame(&right, SensorId::StereoRight) {
                Ok(frame) => frame,
                Err(err) => return Some(Err(err)),
            };

            match StereoPair::try_new(left_frame, right_frame, self.reader.pairing_window) {
                Ok(pair) => return Some(Ok(pair)),
                Err(err) => return Some(Err(DatasetError::PairingFailed { source: err })),
            }
        }

        None
    }
}

impl<'a> Iterator for DatasetTimedPairs<'a> {
    type Item = Result<TimedPair, DatasetError>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.reader.entries.len() {
            let entry = self.reader.entries[self.index].clone();
            self.index += 1;

            let right = match entry.right {
                Some(right) => right,
                None => continue,
            };

            let left_start = Instant::now();
            let left_frame = match self.reader.read_frame(&entry.left, SensorId::StereoLeft) {
                Ok(frame) => frame,
                Err(err) => return Some(Err(err)),
            };
            let left_time = left_start.elapsed();
            let left_bytes = left_frame.data().len();

            let right_start = Instant::now();
            let right_frame = match self.reader.read_frame(&right, SensorId::StereoRight) {
                Ok(frame) => frame,
                Err(err) => return Some(Err(err)),
            };
            let right_time = right_start.elapsed();
            let right_bytes = right_frame.data().len();

            let pair_start = Instant::now();
            let pair =
                match StereoPair::try_new(left_frame, right_frame, self.reader.pairing_window) {
                    Ok(pair) => pair,
                    Err(err) => return Some(Err(DatasetError::PairingFailed { source: err })),
                };
            let pairing = pair_start.elapsed();

            let timings = DatasetReadTimings {
                left_read: left_time,
                right_read: right_time,
                pairing,
                left_bytes,
                right_bytes,
            };

            return Some(Ok(TimedPair { pair, timings }));
        }

        None
    }
}

impl DatasetReader {
    fn read_frame(
        &mut self,
        frame_ref: &DatasetFrameRef,
        sensor: SensorId,
    ) -> Result<Frame, DatasetError> {
        let (width, height) = match self.meta.mono.as_ref() {
            Some(mono) => (mono.width, mono.height),
            None => {
                return Err(DatasetError::InvalidConfig {
                    msg: "meta.json missing mono config",
                });
            }
        };
        let path = frame_ref.path.clone();
        let data = std::fs::read(&path).map_err(|e| DatasetError::ReadFile {
            path: path.clone(),
            source: e,
        })?;
        Frame::new(
            sensor,
            match sensor {
                SensorId::StereoLeft => self.next_left_id(),
                SensorId::StereoRight => self.next_right_id(),
            },
            Timestamp::from_nanos(frame_ref.timestamp_ns),
            width,
            height,
            data,
        )
        .map_err(|e| DatasetError::InvalidConfig {
            msg: match e {
                FrameError::DimensionMismatch { .. } => "frame size mismatch",
                FrameError::ZeroDimensions { .. } => "frame dimensions must be nonzero",
            },
        })
    }
}

fn parse_manifest_entries(
    root: &Path,
    manifest: Manifest,
) -> Result<Vec<DatasetEntry>, DatasetError> {
    manifest
        .entries
        .into_iter()
        .map(|entry| {
            let left = parse_frame_ref(root, entry.left)?;
            let right = match entry.pairing {
                ManifestPairing::Paired { right, .. } => Some(parse_frame_ref(root, right)?),
                ManifestPairing::MissingRight { .. } => None,
            };
            Ok(DatasetEntry { left, right })
        })
        .collect()
}

fn parse_frame_ref(
    root: &Path,
    frame_ref: ManifestFrameRef,
) -> Result<DatasetFrameRef, DatasetError> {
    let relative = Path::new(&frame_ref.path);
    if relative.as_os_str().is_empty()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(DatasetError::InvalidFramePath {
            path: frame_ref.path,
            reason: "path must be a non-empty normalized relative path",
        });
    }

    let candidate = root.join(relative);
    let resolved = std::fs::canonicalize(&candidate).map_err(|source| DatasetError::ReadFile {
        path: candidate,
        source,
    })?;
    if !resolved.starts_with(root) {
        return Err(DatasetError::InvalidFramePath {
            path: frame_ref.path,
            reason: "resolved path escapes the dataset root",
        });
    }

    Ok(DatasetFrameRef {
        timestamp_ns: frame_ref.timestamp_ns,
        path: resolved,
    })
}

#[derive(Debug, Clone, Copy)]
pub struct DatasetStats {
    pub left_count: usize,
    pub right_count: usize,
    pub paired_count: usize,
    pub left_fps: Option<f64>,
    pub right_fps: Option<f64>,
    pub paired_fps: Option<f64>,
}

impl DatasetStats {
    fn from_frames(frames: &super::FrameSet) -> Self {
        let left_fps = fps_from_frames(&frames.left);
        let right_fps = fps_from_frames(&frames.right);
        let paired_fps = fps_from_pairs(&frames.left, &frames.right);
        Self {
            left_count: frames.left.len(),
            right_count: frames.right.len(),
            paired_count: frames.left.len().min(frames.right.len()),
            left_fps,
            right_fps,
            paired_fps,
        }
    }
}

fn fps_from_frames(frames: &[FrameInfo]) -> Option<f64> {
    if frames.len() < 2 {
        return None;
    }
    let (min_ts, max_ts) = min_max_ts(frames);
    let span_ns = max_ts.abs_diff(min_ts) as f64;
    if span_ns <= 0.0 {
        return None;
    }
    let span_s = span_ns / 1_000_000_000.0;
    Some(frames.len() as f64 / span_s)
}

fn fps_from_pairs(left: &[FrameInfo], right: &[FrameInfo]) -> Option<f64> {
    if left.is_empty() || right.is_empty() {
        return None;
    }
    let (left_min, left_max) = min_max_ts(left);
    let (right_min, right_max) = min_max_ts(right);
    let span_ns = left_max.max(right_max).abs_diff(left_min.min(right_min)) as f64;
    if span_ns <= 0.0 {
        return None;
    }
    let span_s = span_ns / 1_000_000_000.0;
    Some(left.len().min(right.len()) as f64 / span_s)
}

fn min_max_ts(frames: &[FrameInfo]) -> (i64, i64) {
    let mut min_ts = i64::MAX;
    let mut max_ts = i64::MIN;
    for frame in frames {
        min_ts = min_ts.min(frame.timestamp_ns);
        max_ts = max_ts.max(frame.timestamp_ns);
    }
    (min_ts, max_ts)
}
