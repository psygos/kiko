use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::{
    CaptureBundle, CaptureId, CaptureImu, CaptureInterval, Frame, FrameError, FrameId, ImuBatch,
    ImuSample, PairingWindowNs, SensorId, StereoPair, Timestamp,
};

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
    imu_samples: Option<Box<[ImuSample]>>,
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
        let imu_time_offset_ns = calibration
            .imu
            .as_ref()
            .map(|imu| imu.extrinsics.time_offset_ns)
            .unwrap_or(0);
        let imu_samples = read_imu_samples(&root, &meta, imu_time_offset_ns)?;
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
            imu_samples,
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

    pub fn bundles(&mut self) -> DatasetBundles<'_> {
        DatasetBundles {
            reader: self,
            index: 0,
            capture_seq: 0,
            previous_capture_time: None,
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

    fn imu_for_interval(&self, interval: CaptureInterval) -> Result<CaptureImu, DatasetError> {
        let Some(samples) = self.imu_samples.as_deref() else {
            return Ok(CaptureImu::Absent);
        };
        let startup_interval = interval.start_exclusive().is_none();

        let start_idx = match interval.start_exclusive() {
            Some(start) => {
                samples.partition_point(|sample| sample.timestamp().as_nanos() <= start.as_nanos())
            }
            None => 0,
        };
        let end_idx = samples.partition_point(|sample| {
            sample.timestamp().as_nanos() <= interval.end_inclusive().as_nanos()
        });

        if start_idx == end_idx {
            if startup_interval {
                return Ok(CaptureImu::Absent);
            }
            return Err(DatasetError::MissingImuSamples {
                start_ns: interval
                    .start_exclusive()
                    .map(|timestamp| timestamp.as_nanos()),
                end_ns: interval.end_inclusive().as_nanos(),
            });
        }

        let batch = match ImuBatch::new(samples[start_idx..end_idx].to_vec()) {
            Ok(batch) => batch,
            Err(_) if startup_interval => return Ok(CaptureImu::Absent),
            Err(_) => {
                return Err(DatasetError::MissingImuSamples {
                    start_ns: interval
                        .start_exclusive()
                        .map(|timestamp| timestamp.as_nanos()),
                    end_ns: interval.end_inclusive().as_nanos(),
                });
            }
        };
        Ok(CaptureImu::present(batch))
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

pub struct DatasetBundles<'a> {
    reader: &'a mut DatasetReader,
    index: usize,
    capture_seq: u64,
    previous_capture_time: Option<Timestamp>,
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

impl<'a> Iterator for DatasetBundles<'a> {
    type Item = Result<CaptureBundle, DatasetError>;

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

            let pair =
                match StereoPair::try_new(left_frame, right_frame, self.reader.pairing_window) {
                    Ok(pair) => pair,
                    Err(err) => return Some(Err(DatasetError::PairingFailed { source: err })),
                };
            let interval =
                match CaptureInterval::new(self.previous_capture_time, pair.capture_time()) {
                    Ok(interval) => interval,
                    Err(_) => {
                        return Some(Err(DatasetError::InvalidConfig {
                            msg: "capture bundle interval must be strictly increasing",
                        }));
                    }
                };
            let imu = match self.reader.imu_for_interval(interval) {
                Ok(imu) => imu,
                Err(err) => return Some(Err(err)),
            };
            let capture_id = CaptureId::new(self.capture_seq);
            self.capture_seq = self.capture_seq.saturating_add(1);
            self.previous_capture_time = Some(interval.end_inclusive());
            return Some(
                CaptureBundle::new(capture_id, pair, interval, imu).map_err(|_| {
                    DatasetError::InvalidConfig {
                        msg: "capture bundle interval must match pair capture time",
                    }
                }),
            );
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

fn read_imu_samples(
    root: &Path,
    meta: &super::Meta,
    time_offset_ns: i64,
) -> Result<Option<Box<[ImuSample]>>, DatasetError> {
    if meta.imu.is_none() {
        return Ok(None);
    }
    let path = root.join(format::IMU_FILE);
    let bytes = std::fs::read(&path).map_err(|source| DatasetError::ReadFile {
        path: path.clone(),
        source,
    })?;
    if bytes.is_empty() {
        return Err(DatasetError::MissingImuSamples {
            start_ns: None,
            end_ns: 0,
        });
    }
    const RECORD_BYTES: usize = std::mem::size_of::<i64>() + 6 * std::mem::size_of::<f64>();
    let chunks = bytes.chunks_exact(RECORD_BYTES);
    if !chunks.remainder().is_empty() {
        return Err(DatasetError::ReadFile {
            path,
            source: std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "imu.bin length is not a whole number of imu records",
            ),
        });
    }
    let mut samples = Vec::with_capacity(bytes.len() / RECORD_BYTES);
    for chunk in chunks {
        let timestamp = i64::from_le_bytes(chunk[0..8].try_into().expect("timestamp bytes"));
        let mut offset = 8usize;
        let read_f64 = |chunk: &[u8], offset: &mut usize| -> f64 {
            let end = *offset + 8;
            let value = f64::from_le_bytes(chunk[*offset..end].try_into().expect("f64 bytes"));
            *offset = end;
            value
        };
        let accel = [
            read_f64(chunk, &mut offset),
            read_f64(chunk, &mut offset),
            read_f64(chunk, &mut offset),
        ];
        let gyro = [
            read_f64(chunk, &mut offset),
            read_f64(chunk, &mut offset),
            read_f64(chunk, &mut offset),
        ];
        let sample = ImuSample::new(Timestamp::from_nanos(timestamp), accel, gyro)
            .map_err(|source| DatasetError::ReadFile {
                path: path.clone(),
                source: std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("invalid imu sample: {source}"),
                ),
            })?
            .shifted_timestamp_ns(time_offset_ns)
            .map_err(|source| DatasetError::ReadFile {
                path: path.clone(),
                source: std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("invalid imu timestamp shift: {source}"),
                ),
            })?;
        samples.push(sample);
    }
    Ok(Some(samples.into_boxed_slice()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{
        Calibration, CameraIntrinsics, DatasetWriter, ImuCalibration, ImuExtrinsicsMeta, ImuMeta,
        ImuNoiseMeta, Meta, MonoMeta,
    };
    use crate::{ImuBatch, ImuSample, SensorId};
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

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

    fn unique_temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        std::env::temp_dir().join(format!("kiko-{name}-{}-{nanos}", std::process::id()))
    }

    fn mono_frame(sensor: SensorId, frame_id: u64, timestamp_ns: i64) -> Frame {
        Frame::new(
            sensor,
            FrameId::new(frame_id),
            Timestamp::from_nanos(timestamp_ns),
            2,
            2,
            vec![frame_id as u8; 4],
        )
        .expect("frame")
    }

    fn calibration() -> Calibration {
        Calibration {
            left: CameraIntrinsics {
                fx: 100.0,
                fy: 100.0,
                cx: 1.0,
                cy: 1.0,
                width: 2,
                height: 2,
            },
            right: CameraIntrinsics {
                fx: 100.0,
                fy: 100.0,
                cx: 1.0,
                cy: 1.0,
                width: 2,
                height: 2,
            },
            baseline_m: 0.1,
            rectified: true,
            imu: None,
        }
    }

    fn calibration_with_time_offset_ns(time_offset_ns: i64) -> Calibration {
        Calibration {
            imu: Some(ImuCalibration {
                noise: ImuNoiseMeta {
                    accel_noise_density: 0.1,
                    gyro_noise_density: 0.01,
                    accel_random_walk: 0.001,
                    gyro_random_walk: 0.0001,
                },
                extrinsics: ImuExtrinsicsMeta {
                    rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    translation: [0.0, 0.0, 0.0],
                    time_offset_ns,
                },
                gravity_magnitude_mps2: 9.81,
            }),
            ..calibration()
        }
    }

    fn meta_with_imu() -> Meta {
        Meta {
            created: "now".to_string(),
            device: "test".to_string(),
            mono: Some(MonoMeta {
                width: 2,
                height: 2,
                fps: 10,
            }),
            depth: None,
            imu: Some(ImuMeta { rate_hz: 200 }),
        }
    }

    #[test]
    fn bundles_round_trip_imu_batches_per_capture() {
        let dataset_dir = unique_temp_dir("reader-imu-round-trip");
        let (writer, handle) =
            DatasetWriter::create(&dataset_dir, &meta_with_imu(), &calibration()).expect("writer");

        writer.write_frame(&mono_frame(SensorId::StereoLeft, 0, 100));
        writer.write_frame(&mono_frame(SensorId::StereoRight, 1, 104));
        writer.write_imu(
            &ImuBatch::new(vec![
                ImuSample::new(Timestamp::from_nanos(90), [0.0; 3], [0.0; 3]).expect("imu 0"),
                ImuSample::new(Timestamp::from_nanos(102), [1.0; 3], [2.0; 3]).expect("imu 1"),
            ])
            .expect("imu batch 0"),
        );

        writer.write_frame(&mono_frame(SensorId::StereoLeft, 2, 200));
        writer.write_frame(&mono_frame(SensorId::StereoRight, 3, 204));
        writer.write_imu(
            &ImuBatch::new(vec![
                ImuSample::new(Timestamp::from_nanos(103), [3.0; 3], [4.0; 3]).expect("imu 2"),
                ImuSample::new(Timestamp::from_nanos(202), [5.0; 3], [6.0; 3]).expect("imu 3"),
            ])
            .expect("imu batch 1"),
        );

        drop(writer);
        handle.finish().expect("finish dataset");

        let mut reader = DatasetReader::open(&dataset_dir).expect("reader");
        let mut bundles = reader.bundles();

        let first = bundles.next().expect("first bundle").expect("first ok");
        assert_eq!(first.capture_id().as_u64(), 0);
        assert_eq!(first.capture_time().as_nanos(), 102);
        let first_imu = first
            .imu()
            .batch()
            .expect("first bundle should carry imu samples");
        assert_eq!(first_imu.len(), 2);
        assert_eq!(first_imu.start_time().as_nanos(), 90);
        assert_eq!(first_imu.end_time().as_nanos(), 102);

        let second = bundles.next().expect("second bundle").expect("second ok");
        assert_eq!(second.capture_id().as_u64(), 1);
        assert_eq!(second.capture_time().as_nanos(), 202);
        let second_imu = second
            .imu()
            .batch()
            .expect("second bundle should carry imu samples");
        assert_eq!(second_imu.len(), 2);
        assert_eq!(second_imu.start_time().as_nanos(), 103);
        assert_eq!(second_imu.end_time().as_nanos(), 202);

        assert!(bundles.next().is_none());

        let _ = std::fs::remove_dir_all(&dataset_dir);
    }

    #[test]
    fn bundles_fail_when_imu_interval_is_empty() {
        let dataset_dir = unique_temp_dir("reader-imu-gap");
        let (writer, handle) =
            DatasetWriter::create(&dataset_dir, &meta_with_imu(), &calibration()).expect("writer");

        writer.write_frame(&mono_frame(SensorId::StereoLeft, 0, 100));
        writer.write_frame(&mono_frame(SensorId::StereoRight, 1, 104));
        writer.write_imu(
            &ImuBatch::new(vec![
                ImuSample::new(Timestamp::from_nanos(101), [0.0; 3], [0.0; 3]).expect("imu 0"),
            ])
            .expect("imu batch"),
        );

        writer.write_frame(&mono_frame(SensorId::StereoLeft, 2, 200));
        writer.write_frame(&mono_frame(SensorId::StereoRight, 3, 204));

        drop(writer);
        handle.finish().expect("finish dataset");

        let mut reader = DatasetReader::open(&dataset_dir).expect("reader");
        let mut bundles = reader.bundles();
        let _first = bundles.next().expect("first bundle").expect("first ok");
        let err = bundles
            .next()
            .expect("second bundle result")
            .expect_err("second bundle should fail for imu gap");
        assert!(matches!(
            err,
            DatasetError::MissingImuSamples {
                start_ns: Some(102),
                end_ns: 202
            }
        ));

        let _ = std::fs::remove_dir_all(&dataset_dir);
    }

    #[test]
    fn bundles_apply_imu_time_offset_before_interval_selection() {
        let dataset_dir = unique_temp_dir("reader-imu-time-offset");
        let (writer, handle) = DatasetWriter::create(
            &dataset_dir,
            &meta_with_imu(),
            &calibration_with_time_offset_ns(10),
        )
        .expect("writer");

        writer.write_frame(&mono_frame(SensorId::StereoLeft, 0, 100));
        writer.write_frame(&mono_frame(SensorId::StereoRight, 1, 104));
        writer.write_imu(
            &ImuBatch::new(vec![
                ImuSample::new(Timestamp::from_nanos(80), [0.0; 3], [0.0; 3]).expect("imu 0"),
                ImuSample::new(Timestamp::from_nanos(92), [1.0; 3], [2.0; 3]).expect("imu 1"),
            ])
            .expect("imu batch"),
        );

        drop(writer);
        handle.finish().expect("finish dataset");

        let mut reader = DatasetReader::open(&dataset_dir).expect("reader");
        let bundle = reader
            .bundles()
            .next()
            .expect("bundle")
            .expect("bundle should apply time offset");
        let imu = bundle.imu().batch().expect("imu batch");
        assert_eq!(imu.start_time().as_nanos(), 90);
        assert_eq!(imu.end_time().as_nanos(), 102);

        let _ = std::fs::remove_dir_all(&dataset_dir);
    }

    #[test]
    fn startup_bundle_allows_missing_imu_until_first_sample() {
        let dataset_dir = unique_temp_dir("reader-imu-startup-gap");
        let (writer, handle) =
            DatasetWriter::create(&dataset_dir, &meta_with_imu(), &calibration()).expect("writer");

        writer.write_frame(&mono_frame(SensorId::StereoLeft, 0, 100));
        writer.write_frame(&mono_frame(SensorId::StereoRight, 1, 104));
        writer.write_imu(
            &ImuBatch::new(vec![
                ImuSample::new(Timestamp::from_nanos(150), [0.0; 3], [0.0; 3]).expect("imu 0"),
                ImuSample::new(Timestamp::from_nanos(180), [1.0; 3], [2.0; 3]).expect("imu 1"),
            ])
            .expect("imu batch"),
        );

        drop(writer);
        handle.finish().expect("finish dataset");

        let mut reader = DatasetReader::open(&dataset_dir).expect("reader");
        let bundle = reader
            .bundles()
            .next()
            .expect("bundle")
            .expect("startup bundle should succeed without imu");
        assert!(matches!(bundle.imu(), CaptureImu::Absent));

        let _ = std::fs::remove_dir_all(&dataset_dir);
    }
}
