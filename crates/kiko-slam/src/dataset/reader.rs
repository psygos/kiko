use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::{Frame, FrameError, FrameId, PairingWindowNs, SensorId, StereoPair, Timestamp};

use super::{
    Calibration, DatasetError, FrameInfo, Manifest, format, read_calibration, read_manifest,
    read_meta, scan_frames,
};

#[derive(Debug)]
pub struct DatasetReader {
    root: PathBuf,
    meta: super::Meta,
    calibration: Calibration,
    manifest: Manifest,
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
        let root = path.into();
        let meta = read_meta(&root)?;
        let calibration = read_calibration(&root)?;
        let manifest = read_manifest(&root)?;
        let pairing_window = PairingWindowNs::new(manifest.header.pairing_window_ns as i64)
            .map_err(|_| DatasetError::InvalidConfig {
                msg: "manifest pairing_window_ns must be > 0",
            })?;
        Ok(Self {
            root,
            meta,
            calibration,
            manifest,
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
        while self.index < self.reader.manifest.entries.len() {
            let entry = self.reader.manifest.entries[self.index].clone();
            self.index += 1;

            if !matches!(entry.status, super::PairStatus::Paired) {
                continue;
            }

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
        while self.index < self.reader.manifest.entries.len() {
            let entry = self.reader.manifest.entries[self.index].clone();
            self.index += 1;

            if !matches!(entry.status, super::PairStatus::Paired) {
                continue;
            }

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
        frame_ref: &super::ManifestFrameRef,
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
        let path = self.root.join(&frame_ref.path);
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
            },
        })
    }
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
    let span_ns = (max_ts - min_ts).abs() as f64;
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
    let span_ns = (left_max.max(right_max) - left_min.min(right_min)).abs() as f64;
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
