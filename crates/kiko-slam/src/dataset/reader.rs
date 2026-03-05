use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::{Frame, FrameError, FrameId, PairingWindowNs, SensorId, StereoPair, Timestamp};

use super::{
    format, read_calibration, read_manifest, read_meta, scan_frames, Calibration, DatasetError,
    FrameInfo, Manifest,
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
        Ok(DatasetStats::from_frames_and_manifest(
            &frames,
            &self.manifest,
        ))
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

            let right = match entry.pairing {
                super::ManifestPairing::Paired { right, .. } => right,
                super::ManifestPairing::MissingRight { .. } => continue,
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

            let right = match entry.pairing {
                super::ManifestPairing::Paired { right, .. } => right,
                super::ManifestPairing::MissingRight { .. } => continue,
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
    fn from_frames_and_manifest(frames: &super::FrameSet, manifest: &Manifest) -> Self {
        let left_fps = fps_from_frames(&frames.left);
        let right_fps = fps_from_frames(&frames.right);
        let paired_count = manifest
            .entries
            .iter()
            .filter(|entry| matches!(entry.pairing, super::ManifestPairing::Paired { .. }))
            .count();
        let paired_fps = fps_from_manifest_pairs(&manifest.entries);
        Self {
            left_count: frames.left.len(),
            right_count: frames.right.len(),
            paired_count,
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
    Some((frames.len().saturating_sub(1)) as f64 / span_s)
}

fn fps_from_manifest_pairs(entries: &[super::ManifestEntry]) -> Option<f64> {
    let paired: Vec<&super::ManifestEntry> = entries
        .iter()
        .filter(|entry| matches!(entry.pairing, super::ManifestPairing::Paired { .. }))
        .collect();
    if paired.len() < 2 {
        return None;
    }
    let mut min_ts = i64::MAX;
    let mut max_ts = i64::MIN;
    for entry in paired.iter().copied() {
        min_ts = min_ts.min(entry.left.timestamp_ns);
        max_ts = max_ts.max(entry.left.timestamp_ns);
        if let super::ManifestPairing::Paired { right, .. } = &entry.pairing {
            min_ts = min_ts.min(right.timestamp_ns);
            max_ts = max_ts.max(right.timestamp_ns);
        }
    }
    let span_ns = (max_ts - min_ts).abs() as f64;
    if span_ns <= 0.0 {
        return None;
    }
    let span_s = span_ns / 1_000_000_000.0;
    Some((paired.len().saturating_sub(1)) as f64 / span_s)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fps_from_frames_uses_intervals_not_sample_count() {
        let frames = vec![
            FrameInfo {
                timestamp_ns: 0,
                path: "a.raw".to_string(),
            },
            FrameInfo {
                timestamp_ns: 1_000_000_000,
                path: "b.raw".to_string(),
            },
            FrameInfo {
                timestamp_ns: 2_000_000_000,
                path: "c.raw".to_string(),
            },
        ];
        let fps = fps_from_frames(&frames).expect("fps");
        assert!((fps - 1.0).abs() < 1e-9);
    }

    #[test]
    fn paired_stats_count_only_manifest_pairs() {
        let frames = super::super::FrameSet {
            left: vec![
                FrameInfo {
                    timestamp_ns: 0,
                    path: "left0.raw".to_string(),
                },
                FrameInfo {
                    timestamp_ns: 1_000_000_000,
                    path: "left1.raw".to_string(),
                },
            ],
            right: vec![FrameInfo {
                timestamp_ns: 1_000_000_010,
                path: "right1.raw".to_string(),
            }],
            depth: Vec::new(),
            parse_fail: 0,
            size_mismatch: 0,
        };
        let manifest = Manifest {
            header: super::super::ManifestHeader {
                dataset_id: "dataset".to_string(),
                created_at: "now".to_string(),
                device: "oak".to_string(),
                format: "raw".to_string(),
                width: 640,
                height: 480,
                fps: 30,
                timebase: "ns".to_string(),
                pairing_policy: "nearest".to_string(),
                pairing_window_ns: 20_000_000,
            },
            stats: super::super::ManifestStats {
                total_left: 2,
                total_right: 1,
                paired_count: 1,
                left_orphans: 1,
                right_orphans: 0,
                drops_by_reason: super::super::DropStats {
                    spool_full: 0,
                    write_fail: 0,
                    parse_fail: 0,
                    size_mismatch: 0,
                    outside_window: 0,
                },
                delta_stats: None,
            },
            entries: vec![
                super::super::ManifestEntry {
                    left: super::super::ManifestFrameRef {
                        timestamp_ns: 0,
                        path: "left0.raw".to_string(),
                    },
                    pairing: super::super::ManifestPairing::MissingRight {
                        reason: super::super::PairReason::NoRightFrames,
                    },
                },
                super::super::ManifestEntry {
                    left: super::super::ManifestFrameRef {
                        timestamp_ns: 1_000_000_000,
                        path: "left1.raw".to_string(),
                    },
                    pairing: super::super::ManifestPairing::Paired {
                        right: super::super::ManifestFrameRef {
                            timestamp_ns: 1_000_000_010,
                            path: "right1.raw".to_string(),
                        },
                        delta_ns: 10,
                    },
                },
            ],
        };
        let stats = DatasetStats::from_frames_and_manifest(&frames, &manifest);
        assert_eq!(stats.left_count, 2);
        assert_eq!(stats.right_count, 1);
        assert_eq!(stats.paired_count, 1);
        assert_eq!(stats.paired_fps, None);
    }
}
