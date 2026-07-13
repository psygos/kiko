use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::{
    CaptureBundle, CaptureId, CaptureImu, CaptureInterval, Frame, FrameId, ImuBatch, ImuSample,
    SensorId, StereoPair, Timestamp,
};

use super::{
    DatasetError, DatasetImuRecordError, DatasetIndex, DatasetIndexFrameRef, DatasetPairIndex,
    FrameInfo, IMU_RECORD_BYTES, format, read_calibration_with_imu_override, read_manifest,
    read_meta, require_calibration_dimensions, scan_frames_with_depth,
};

#[derive(Debug)]
pub struct DatasetReader {
    root: PathBuf,
    calibration: crate::CalibrationBundle,
    manifest: DatasetIndex,
    frame_dimensions: crate::FrameDimensions,
    left_seq: u64,
    right_seq: u64,
    imu_samples: Option<ImuBatch>,
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
        Self::open_with_imu_calibration_override(path, None)
    }

    pub fn open_with_imu_calibration_override(
        path: impl Into<PathBuf>,
        imu_override: Option<super::ImuCalibration>,
    ) -> Result<Self, DatasetError> {
        let root = path.into();
        let meta = read_meta(&root)?;
        if let Some(_) = imu_override.as_ref()
            && meta.imu.is_none()
        {
            return Err(DatasetError::InvalidConfig {
                msg: "runtime IMU override requires IMU data in dataset meta",
            });
        }
        let calibration_document =
            read_calibration_with_imu_override(&root, imu_override.as_ref())?;
        let calibration_path = root.join(format::CALIBRATION_FILE);
        let calibration = crate::CalibrationBundle::from_dataset_calibration(&calibration_document)
            .map_err(|source| DatasetError::InvalidCalibration {
                path: calibration_path,
                source,
            })?;
        let mono = require_calibration_dimensions(&meta, &calibration)?;
        let manifest_file = read_manifest(&root)?;
        let frames = scan_frames_with_depth(
            &root.join(format::FRAMES_DIR),
            mono.width,
            mono.height,
            meta.depth.as_ref(),
        )?;
        let frame_dimensions = frames.mono_dimensions;
        let manifest_path = root.join(format::MANIFEST_FILE);
        let manifest =
            DatasetIndex::try_from_manifest(manifest_file, &meta, &frames).map_err(|source| {
                DatasetError::InvalidManifest {
                    path: manifest_path,
                    source,
                }
            })?;
        let imu_time_offset_ns = calibration
            .inertial()
            .map(|inertial| inertial.extrinsics().time_offset_ns())
            .unwrap_or(0);
        let imu_samples = read_imu_samples(&root, &meta, imu_time_offset_ns)?;
        Ok(Self {
            root,
            calibration,
            manifest,
            frame_dimensions,
            left_seq: 0,
            right_seq: 0,
            imu_samples,
        })
    }

    pub fn calibration(&self) -> &crate::CalibrationBundle {
        &self.calibration
    }

    pub fn has_imu_data(&self) -> bool {
        self.imu_samples.is_some()
    }

    pub fn stats(&self) -> DatasetStats {
        self.manifest.stats
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

    fn imu_for_interval(&self, interval: CaptureInterval) -> Result<CaptureImu, DatasetError> {
        let Some(imu_samples) = self.imu_samples.as_ref() else {
            return Ok(CaptureImu::Absent);
        };
        let samples = imu_samples.samples();
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
        let before_first_imu_sample = end_idx == 0;

        if start_idx == end_idx {
            if startup_interval || before_first_imu_sample {
                return Ok(CaptureImu::Absent);
            }
            return Err(DatasetError::MissingImuSamples {
                start_ns: interval
                    .start_exclusive()
                    .map(|timestamp| timestamp.as_nanos()),
                end_ns: interval.end_inclusive().as_nanos(),
            });
        }

        let batch = imu_samples.slice(start_idx..end_idx).map_err(|source| {
            DatasetError::InvalidImuSlice {
                start_ns: interval
                    .start_exclusive()
                    .map(|timestamp| timestamp.as_nanos()),
                end_ns: interval.end_inclusive().as_nanos(),
                source,
            }
        })?;
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
        let pair = self.reader.manifest.pairs.get(self.index)?;
        self.index += 1;
        Some(read_indexed_pair(
            &self.reader.root,
            self.reader.frame_dimensions,
            &mut self.reader.left_seq,
            &mut self.reader.right_seq,
            pair,
        ))
    }
}

impl<'a> Iterator for DatasetTimedPairs<'a> {
    type Item = Result<TimedPair, DatasetError>;

    fn next(&mut self) -> Option<Self::Item> {
        let pair = self.reader.manifest.pairs.get(self.index)?;
        self.index += 1;

        let (left_frame_id, next_left_sequence) =
            match next_frame_id(self.reader.left_seq, SensorId::StereoLeft) {
                Ok(sequence) => sequence,
                Err(err) => return Some(Err(err)),
            };
        let (right_frame_id, next_right_sequence) =
            match next_frame_id(self.reader.right_seq, SensorId::StereoRight) {
                Ok(sequence) => sequence,
                Err(err) => return Some(Err(err)),
            };

        let left_start = Instant::now();
        let left_frame = match read_indexed_frame(
            &self.reader.root,
            self.reader.frame_dimensions,
            left_frame_id,
            &pair.left,
            SensorId::StereoLeft,
        ) {
            Ok(frame) => frame,
            Err(err) => return Some(Err(err)),
        };
        let left_time = left_start.elapsed();
        let left_bytes = left_frame.data().len();

        let right_start = Instant::now();
        let right_frame = match read_indexed_frame(
            &self.reader.root,
            self.reader.frame_dimensions,
            right_frame_id,
            &pair.right,
            SensorId::StereoRight,
        ) {
            Ok(frame) => frame,
            Err(err) => return Some(Err(err)),
        };
        let right_time = right_start.elapsed();
        let right_bytes = right_frame.data().len();

        let pair_start = Instant::now();
        let pair = StereoPair::from_parts(left_frame, right_frame);
        let pairing = pair_start.elapsed();
        self.reader.left_seq = next_left_sequence;
        self.reader.right_seq = next_right_sequence;

        let timings = DatasetReadTimings {
            left_read: left_time,
            right_read: right_time,
            pairing,
            left_bytes,
            right_bytes,
        };

        Some(Ok(TimedPair { pair, timings }))
    }
}

impl<'a> Iterator for DatasetBundles<'a> {
    type Item = Result<CaptureBundle, DatasetError>;

    fn next(&mut self) -> Option<Self::Item> {
        let pair_index = self.reader.manifest.pairs.get(self.index)?;
        self.index += 1;
        let (capture_id, next_capture_seq) = match next_capture_id(self.capture_seq) {
            Ok(sequence) => sequence,
            Err(err) => return Some(Err(err)),
        };
        let pair = match read_indexed_pair(
            &self.reader.root,
            self.reader.frame_dimensions,
            &mut self.reader.left_seq,
            &mut self.reader.right_seq,
            pair_index,
        ) {
            Ok(pair) => pair,
            Err(err) => return Some(Err(err)),
        };
        let interval = match CaptureInterval::new(self.previous_capture_time, pair.capture_time()) {
            Ok(interval) => interval,
            Err(source) => return Some(Err(DatasetError::InvalidCaptureInterval { source })),
        };
        let imu = match self.reader.imu_for_interval(interval) {
            Ok(imu) => imu,
            Err(err) => return Some(Err(err)),
        };
        match CaptureBundle::new(capture_id, pair, interval, imu) {
            Ok(bundle) => {
                self.capture_seq = next_capture_seq;
                self.previous_capture_time = Some(interval.end_inclusive());
                Some(Ok(bundle))
            }
            Err(source) => Some(Err(DatasetError::InvalidCaptureBundle { source })),
        }
    }
}

fn read_indexed_pair(
    root: &Path,
    dimensions: crate::FrameDimensions,
    left_sequence: &mut u64,
    right_sequence: &mut u64,
    pair: &DatasetPairIndex,
) -> Result<StereoPair, DatasetError> {
    let (left_frame_id, next_left_sequence) = next_frame_id(*left_sequence, SensorId::StereoLeft)?;
    let (right_frame_id, next_right_sequence) =
        next_frame_id(*right_sequence, SensorId::StereoRight)?;
    let left = read_indexed_frame(
        root,
        dimensions,
        left_frame_id,
        &pair.left,
        SensorId::StereoLeft,
    )?;
    let right = read_indexed_frame(
        root,
        dimensions,
        right_frame_id,
        &pair.right,
        SensorId::StereoRight,
    )?;
    *left_sequence = next_left_sequence;
    *right_sequence = next_right_sequence;
    Ok(StereoPair::from_parts(left, right))
}

fn next_frame_id(sequence: u64, sensor: SensorId) -> Result<(FrameId, u64), DatasetError> {
    let next_sequence = sequence
        .checked_add(1)
        .ok_or(DatasetError::FrameSequenceExhausted { sensor })?;
    Ok((FrameId::new(sequence), next_sequence))
}

fn next_capture_id(sequence: u64) -> Result<(CaptureId, u64), DatasetError> {
    let next_sequence = sequence
        .checked_add(1)
        .ok_or(DatasetError::CaptureSequenceExhausted)?;
    Ok((CaptureId::new(sequence), next_sequence))
}

fn read_indexed_frame(
    root: &Path,
    dimensions: crate::FrameDimensions,
    frame_id: FrameId,
    frame_ref: &DatasetIndexFrameRef,
    sensor: SensorId,
) -> Result<Frame, DatasetError> {
    let path = root.join(frame_ref.path.as_ref());
    let data = std::fs::read(&path).map_err(|source| DatasetError::ReadFile {
        path: path.clone(),
        source,
    })?;
    Frame::new(
        sensor,
        frame_id,
        frame_ref.timestamp,
        dimensions.width(),
        dimensions.height(),
        data,
    )
    .map_err(|source| DatasetError::InvalidFrame { path, source })
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
    pub(super) fn from_frames_and_pairs(
        frames: &super::FrameSet,
        pairs: &[DatasetPairIndex],
    ) -> Self {
        let left_fps = fps_from_frames(&frames.left);
        let right_fps = fps_from_frames(&frames.right);
        let paired_count = pairs.len();
        let paired_fps = fps_from_pairs(pairs);
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
    let span_ns = max_ts.abs_diff(min_ts) as f64;
    if span_ns <= 0.0 {
        return None;
    }
    let span_s = span_ns / 1_000_000_000.0;
    Some((frames.len().saturating_sub(1)) as f64 / span_s)
}

fn fps_from_pairs(pairs: &[DatasetPairIndex]) -> Option<f64> {
    if pairs.len() < 2 {
        return None;
    }
    let mut min_ts = i64::MAX;
    let mut max_ts = i64::MIN;
    for pair in pairs {
        min_ts = min_ts.min(pair.left.timestamp.as_nanos());
        max_ts = max_ts.max(pair.left.timestamp.as_nanos());
        min_ts = min_ts.min(pair.right.timestamp.as_nanos());
        max_ts = max_ts.max(pair.right.timestamp.as_nanos());
    }
    let span_ns = max_ts.abs_diff(min_ts) as f64;
    if span_ns <= 0.0 {
        return None;
    }
    let span_s = span_ns / 1_000_000_000.0;
    Some((pairs.len() - 1) as f64 / span_s)
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

#[derive(Clone, Copy, Debug)]
struct RawImuRecord {
    timestamp_ns: i64,
    accel_mps2: [f64; 3],
    gyro_radps: [f64; 3],
}

impl RawImuRecord {
    fn parse(mut bytes: &[u8]) -> Result<Self, std::io::Error> {
        fn read_array<const N: usize>(input: &mut &[u8]) -> Result<[u8; N], std::io::Error> {
            let mut bytes = [0_u8; N];
            std::io::Read::read_exact(input, &mut bytes)?;
            Ok(bytes)
        }

        fn read_f64(input: &mut &[u8]) -> Result<f64, std::io::Error> {
            read_array(input).map(f64::from_le_bytes)
        }

        let record = Self {
            timestamp_ns: i64::from_le_bytes(read_array(&mut bytes)?),
            accel_mps2: [
                read_f64(&mut bytes)?,
                read_f64(&mut bytes)?,
                read_f64(&mut bytes)?,
            ],
            gyro_radps: [
                read_f64(&mut bytes)?,
                read_f64(&mut bytes)?,
                read_f64(&mut bytes)?,
            ],
        };
        if !bytes.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "IMU record contains trailing bytes",
            ));
        }
        Ok(record)
    }

    fn into_sample(self, time_offset_ns: i64) -> Result<ImuSample, DatasetImuRecordError> {
        ImuSample::new(
            Timestamp::from_nanos(self.timestamp_ns),
            self.accel_mps2,
            self.gyro_radps,
        )
        .map_err(|source| DatasetImuRecordError::InvalidSample { source })?
        .shifted_timestamp_ns(time_offset_ns)
        .map_err(|source| DatasetImuRecordError::TimestampShift { source })
    }
}

fn read_imu_samples(
    root: &Path,
    meta: &super::Meta,
    time_offset_ns: i64,
) -> Result<Option<ImuBatch>, DatasetError> {
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
    let chunks = bytes.chunks_exact(IMU_RECORD_BYTES);
    if !chunks.remainder().is_empty() {
        return Err(DatasetError::InvalidImuLength {
            path,
            byte_len: bytes.len(),
            record_bytes: IMU_RECORD_BYTES,
        });
    }
    let mut samples = Vec::with_capacity(bytes.len() / IMU_RECORD_BYTES);
    for (record_index, chunk) in chunks.enumerate() {
        let record =
            RawImuRecord::parse(chunk).map_err(|source| DatasetError::InvalidImuRecord {
                path: path.clone(),
                record_index,
                source: DatasetImuRecordError::Decode { source },
            })?;
        let sample = record.into_sample(time_offset_ns).map_err(|source| {
            DatasetError::InvalidImuRecord {
                path: path.clone(),
                record_index,
                source,
            }
        })?;
        samples.push(sample);
    }
    ImuBatch::new(samples)
        .map(Some)
        .map_err(|source| DatasetError::InvalidImuBatch { path, source })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{
        Calibration, CameraIntrinsics, DatasetFrameReferenceError, DatasetManifestError,
        DatasetWriter, ImuCalibration, ImuExtrinsicsMeta, ImuMeta, ImuNoiseMeta, Manifest, Meta,
        MonoMeta,
    };
    use crate::{ImuBatch, ImuSample, SensorId};
    use std::error::Error as _;
    use std::path::PathBuf;
    use std::sync::Arc;
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
    fn raw_imu_record_parser_rejects_short_and_trailing_input_without_panicking() {
        let short = [0_u8; IMU_RECORD_BYTES - 1];
        assert_eq!(
            RawImuRecord::parse(&short)
                .expect_err("short record")
                .kind(),
            std::io::ErrorKind::UnexpectedEof
        );

        let trailing = [0_u8; IMU_RECORD_BYTES + 1];
        assert_eq!(
            RawImuRecord::parse(&trailing)
                .expect_err("trailing byte")
                .kind(),
            std::io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn imu_file_length_error_reports_the_format_boundary() {
        let dataset_dir = unique_temp_dir("invalid-imu-length");
        std::fs::create_dir_all(&dataset_dir).expect("dataset directory");
        let path = dataset_dir.join(crate::dataset::format::IMU_FILE);
        std::fs::write(&path, [0_u8; 3]).expect("malformed IMU file");

        let error = read_imu_samples(&dataset_dir, &meta_with_imu(), 0)
            .expect_err("partial record must fail");
        assert!(matches!(
            error,
            DatasetError::InvalidImuLength {
                path: actual_path,
                byte_len: 3,
                record_bytes: IMU_RECORD_BYTES,
            } if actual_path == path
        ));

        let _ = std::fs::remove_dir_all(&dataset_dir);
    }

    #[test]
    fn invalid_imu_sample_preserves_record_and_domain_source() {
        let dataset_dir = unique_temp_dir("invalid-imu-sample");
        std::fs::create_dir_all(&dataset_dir).expect("dataset directory");
        let path = dataset_dir.join(crate::dataset::format::IMU_FILE);
        let mut bytes = Vec::with_capacity(IMU_RECORD_BYTES);
        bytes.extend_from_slice(&7_i64.to_le_bytes());
        for value in [f64::NAN, 0.0, 0.0, 0.0, 0.0, 0.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&path, bytes).expect("invalid IMU file");

        let error = read_imu_samples(&dataset_dir, &meta_with_imu(), 0)
            .expect_err("nonfinite sample must fail");
        assert!(matches!(
            &error,
            DatasetError::InvalidImuRecord {
                path: actual_path,
                record_index: 0,
                source: DatasetImuRecordError::InvalidSample { .. },
            } if actual_path == &path
        ));
        let record = error.source().expect("record source");
        let sample = record.source().expect("sample source");
        assert!(sample.to_string().contains("imu accel axis 0"));

        let _ = std::fs::remove_dir_all(&dataset_dir);
    }

    #[test]
    fn imu_file_is_parsed_once_into_a_strictly_ordered_batch() {
        let dataset_dir = unique_temp_dir("invalid-imu-order");
        std::fs::create_dir_all(&dataset_dir).expect("dataset directory");
        let path = dataset_dir.join(crate::dataset::format::IMU_FILE);
        let mut bytes = Vec::with_capacity(IMU_RECORD_BYTES * 2);
        for timestamp_ns in [20_i64, 10_i64] {
            bytes.extend_from_slice(&timestamp_ns.to_le_bytes());
            for value in [0.0_f64; 6] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        std::fs::write(&path, bytes).expect("out-of-order IMU file");

        let error = read_imu_samples(&dataset_dir, &meta_with_imu(), 0)
            .expect_err("out-of-order timestamps must fail at file load");
        assert!(matches!(
            &error,
            DatasetError::InvalidImuBatch {
                path: actual_path,
                source: crate::ImuBatchError::NonIncreasingTimestamps {
                    previous,
                    current,
                },
            } if actual_path == &path
                && previous.as_nanos() == 20
                && current.as_nanos() == 10
        ));
        assert!(error.source().is_some(), "batch error must remain sourced");

        let _ = std::fs::remove_dir_all(&dataset_dir);
    }

    #[test]
    fn dataset_capture_errors_preserve_nested_domain_sources() {
        let timestamp = Timestamp::from_nanos(10);
        let interval_source = crate::CaptureIntervalError::NonIncreasing {
            start_exclusive: timestamp,
            end_inclusive: timestamp,
        };
        let interval_error = DatasetError::InvalidCaptureInterval {
            source: interval_source,
        };
        assert_eq!(
            interval_error
                .source()
                .expect("interval source")
                .to_string(),
            interval_source.to_string()
        );

        let bundle_error = DatasetError::InvalidCaptureBundle {
            source: crate::CaptureBundleError::InvalidInterval(interval_source),
        };
        let bundle_source = bundle_error.source().expect("bundle source");
        assert!(
            bundle_source.source().is_some(),
            "interval source must survive"
        );
    }

    #[test]
    fn paired_stats_count_only_manifest_pairs() {
        let frames = super::super::FrameSet {
            mono_dimensions: crate::FrameDimensions::try_new(640, 480).expect("dimensions"),
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
        };
        let pairs = vec![DatasetPairIndex {
            left: DatasetIndexFrameRef {
                timestamp: Timestamp::from_nanos(1_000_000_000),
                path: Arc::from("left1.raw"),
            },
            right: DatasetIndexFrameRef {
                timestamp: Timestamp::from_nanos(1_000_000_010),
                path: Arc::from("right1.raw"),
            },
        }];
        let stats = DatasetStats::from_frames_and_pairs(&frames, &pairs);
        assert_eq!(stats.left_count, 2);
        assert_eq!(stats.right_count, 1);
        assert_eq!(stats.paired_count, 1);
        assert_eq!(stats.paired_fps, None);
    }

    #[test]
    fn indexed_pair_sequences_advance_transactionally_and_never_saturate() {
        let dataset_dir = unique_temp_dir("reader-frame-sequence");
        std::fs::create_dir_all(&dataset_dir).expect("dataset directory");
        let pair_index = DatasetPairIndex {
            left: DatasetIndexFrameRef {
                timestamp: Timestamp::from_nanos(10),
                path: Arc::from("left.raw"),
            },
            right: DatasetIndexFrameRef {
                timestamp: Timestamp::from_nanos(12),
                path: Arc::from("right.raw"),
            },
        };
        let dimensions = crate::FrameDimensions::try_new(2, 2).expect("dimensions");
        let mut left_sequence = 7_u64;
        let mut right_sequence = 9_u64;
        std::fs::write(dataset_dir.join("left.raw"), [0_u8; 4]).expect("left frame");

        let missing_right = read_indexed_pair(
            &dataset_dir,
            dimensions,
            &mut left_sequence,
            &mut right_sequence,
            &pair_index,
        )
        .expect_err("missing right frame must fail");
        assert!(matches!(missing_right, DatasetError::ReadFile { .. }));
        assert_eq!(left_sequence, 7);
        assert_eq!(right_sequence, 9);

        std::fs::write(dataset_dir.join("right.raw"), [0_u8; 4]).expect("right frame");
        let pair = read_indexed_pair(
            &dataset_dir,
            dimensions,
            &mut left_sequence,
            &mut right_sequence,
            &pair_index,
        )
        .expect("pair");
        assert_eq!(pair.left().frame_id(), FrameId::new(7));
        assert_eq!(pair.right().frame_id(), FrameId::new(9));
        assert_eq!(left_sequence, 8);
        assert_eq!(right_sequence, 10);

        left_sequence = u64::MAX;
        let exhausted = read_indexed_pair(
            &dataset_dir,
            dimensions,
            &mut left_sequence,
            &mut right_sequence,
            &pair_index,
        )
        .expect_err("sequence exhaustion must be explicit");
        assert!(matches!(
            exhausted,
            DatasetError::FrameSequenceExhausted {
                sensor: SensorId::StereoLeft,
            }
        ));
        assert_eq!(left_sequence, u64::MAX);
        assert_eq!(right_sequence, 10);
        assert!(matches!(
            next_capture_id(u64::MAX),
            Err(DatasetError::CaptureSequenceExhausted)
        ));

        let _ = std::fs::remove_dir_all(&dataset_dir);
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
                initial_accel_bias: None,
                initial_gyro_bias: None,
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
    fn reader_rejects_invalid_or_mismatched_calibration_at_open() {
        let invalid_dir = unique_temp_dir("reader-invalid-calibration");
        let (writer, handle) =
            DatasetWriter::create(&invalid_dir, &meta_with_imu(), &calibration()).expect("writer");
        drop(writer);
        handle.finish().expect("finish dataset");

        let calibration_path = invalid_dir.join(crate::dataset::format::CALIBRATION_FILE);
        let mut invalid = calibration();
        invalid.baseline_m = 0.0;
        std::fs::write(
            &calibration_path,
            serde_json::to_vec_pretty(&invalid).expect("serialize invalid calibration"),
        )
        .expect("write invalid calibration");

        let error = DatasetReader::open(&invalid_dir).expect_err("invalid calibration must fail");
        assert!(matches!(
            &error,
            DatasetError::InvalidCalibration {
                path,
                source: crate::CalibrationBundleError::InvalidStereo {
                    source: crate::RectifiedStereoError::InvalidBaseline { baseline_m: 0.0 },
                },
            } if path == &calibration_path
        ));
        assert!(
            error.source().and_then(std::error::Error::source).is_some(),
            "stereo validation source must remain available"
        );
        let _ = std::fs::remove_dir_all(&invalid_dir);

        let mismatch_dir = unique_temp_dir("reader-calibration-dimensions");
        let (writer, handle) =
            DatasetWriter::create(&mismatch_dir, &meta_with_imu(), &calibration()).expect("writer");
        drop(writer);
        handle.finish().expect("finish dataset");

        let mut mismatch = calibration();
        mismatch.left.width = 3;
        mismatch.right.width = 3;
        std::fs::write(
            mismatch_dir.join(crate::dataset::format::CALIBRATION_FILE),
            serde_json::to_vec_pretty(&mismatch).expect("serialize mismatched calibration"),
        )
        .expect("write mismatched calibration");

        let error = DatasetReader::open(&mismatch_dir)
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
        let _ = std::fs::remove_dir_all(&mismatch_dir);
    }

    #[test]
    fn reader_rejects_unresolved_manifest_references_with_full_source_context() {
        let dataset_dir = unique_temp_dir("reader-invalid-manifest-reference");
        let (writer, handle) =
            DatasetWriter::create(&dataset_dir, &meta_with_imu(), &calibration()).expect("writer");
        writer.write_frame(&mono_frame(SensorId::StereoLeft, 0, 100));
        writer.write_frame(&mono_frame(SensorId::StereoRight, 1, 104));
        drop(writer);
        handle.finish().expect("finish dataset");

        let manifest_path = dataset_dir.join(crate::dataset::format::MANIFEST_FILE);
        let mut manifest: Manifest = serde_json::from_slice(
            &std::fs::read(&manifest_path).expect("read generated manifest"),
        )
        .expect("parse generated manifest");
        manifest.entries[0].left.path = "frames/100_missing.raw".to_string();
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("serialize altered manifest"),
        )
        .expect("write altered manifest");

        let error = DatasetReader::open(&dataset_dir)
            .expect_err("an unresolved manifest reference must fail open");
        assert!(matches!(
            &error,
            DatasetError::InvalidManifest {
                path,
                source: DatasetManifestError::InvalidFrameReference {
                    entry_index: 0,
                    role: super::super::DatasetFrameRole::Left,
                    source: DatasetFrameReferenceError::MissingFromDataset { .. },
                },
            } if path == &manifest_path
        ));
        let manifest_source = error.source().expect("manifest source");
        let frame_reference_source = manifest_source.source().expect("frame-reference source");
        assert!(
            frame_reference_source
                .to_string()
                .contains("not present in the scanned dataset")
        );

        let _ = std::fs::remove_dir_all(&dataset_dir);
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
        let source_imu_ptr = reader
            .imu_samples
            .as_ref()
            .expect("loaded IMU batch")
            .samples()
            .as_ptr();
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
        assert!(std::ptr::eq(first_imu.samples().as_ptr(), source_imu_ptr));

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
    fn runtime_imu_override_reshifts_loaded_samples_before_bundle_selection() {
        let dataset_dir = unique_temp_dir("reader-runtime-imu-override");
        let (writer, handle) = DatasetWriter::create(
            &dataset_dir,
            &meta_with_imu(),
            &calibration_with_time_offset_ns(0),
        )
        .expect("writer");

        writer.write_frame(&mono_frame(SensorId::StereoLeft, 0, 130));
        writer.write_frame(&mono_frame(SensorId::StereoRight, 1, 134));
        writer.write_imu(
            &ImuBatch::new(vec![
                ImuSample::new(Timestamp::from_nanos(100), [0.0; 3], [0.0; 3]).expect("imu 0"),
                ImuSample::new(Timestamp::from_nanos(112), [1.0; 3], [2.0; 3]).expect("imu 1"),
            ])
            .expect("imu batch"),
        );

        drop(writer);
        handle.finish().expect("finish dataset");

        let mut reader = DatasetReader::open_with_imu_calibration_override(
            &dataset_dir,
            Some(
                calibration_with_time_offset_ns(10)
                    .imu
                    .expect("imu calibration"),
            ),
        )
        .expect("reader");
        let bundle = reader
            .bundles()
            .next()
            .expect("bundle")
            .expect("bundle should reflect runtime override");
        let imu = bundle.imu().batch().expect("imu batch");
        assert_eq!(imu.start_time().as_nanos(), 110);
        assert_eq!(imu.end_time().as_nanos(), 122);
        assert_eq!(
            reader
                .calibration()
                .inertial()
                .expect("stored inertial calibration")
                .extrinsics()
                .time_offset_ns(),
            10
        );

        let _ = std::fs::remove_dir_all(&dataset_dir);
    }

    #[test]
    fn open_with_runtime_imu_override_applies_override_before_loading_samples() {
        let dataset_dir = unique_temp_dir("reader-runtime-imu-open-order");
        let (writer, handle) = DatasetWriter::create(
            &dataset_dir,
            &meta_with_imu(),
            &calibration_with_time_offset_ns(10),
        )
        .expect("writer");

        let near_max = i64::MAX - 5;
        writer.write_frame(&mono_frame(SensorId::StereoLeft, 0, near_max - 100));
        writer.write_frame(&mono_frame(SensorId::StereoRight, 1, near_max - 96));
        writer.write_imu(
            &ImuBatch::new(vec![
                ImuSample::new(Timestamp::from_nanos(near_max), [0.0; 3], [0.0; 3]).expect("imu 0"),
            ])
            .expect("imu batch"),
        );

        drop(writer);
        handle.finish().expect("finish dataset");

        let open_err =
            DatasetReader::open(&dataset_dir).expect_err("embedded offset should overflow");
        assert!(matches!(
            &open_err,
            DatasetError::InvalidImuRecord {
                record_index: 0,
                source: DatasetImuRecordError::TimestampShift { .. },
                ..
            }
        ));
        let record = open_err.source().expect("record source");
        let shift = record.source().expect("timestamp-shift source");
        assert!(shift.to_string().contains("overflowed i64 range"));

        let reader = DatasetReader::open_with_imu_calibration_override(
            &dataset_dir,
            Some(
                calibration_with_time_offset_ns(0)
                    .imu
                    .expect("imu calibration"),
            ),
        )
        .expect("runtime override should become authoritative before sample load");
        assert_eq!(
            reader
                .calibration()
                .inertial()
                .expect("stored inertial calibration")
                .extrinsics()
                .time_offset_ns(),
            0
        );

        let _ = std::fs::remove_dir_all(&dataset_dir);
    }

    #[test]
    fn open_with_runtime_imu_override_ignores_non_deserializable_embedded_imu_block() {
        let dataset_dir = unique_temp_dir("reader-runtime-imu-json-override");
        let (writer, handle) = DatasetWriter::create(
            &dataset_dir,
            &meta_with_imu(),
            &calibration_with_time_offset_ns(0),
        )
        .expect("writer");

        writer.write_frame(&mono_frame(SensorId::StereoLeft, 0, 130));
        writer.write_frame(&mono_frame(SensorId::StereoRight, 1, 134));
        writer.write_imu(
            &ImuBatch::new(vec![
                ImuSample::new(Timestamp::from_nanos(100), [0.0; 3], [0.0; 3]).expect("imu 0"),
                ImuSample::new(Timestamp::from_nanos(112), [1.0; 3], [2.0; 3]).expect("imu 1"),
            ])
            .expect("imu batch"),
        );

        drop(writer);
        handle.finish().expect("finish dataset");

        let calibration_path = dataset_dir.join(crate::dataset::format::CALIBRATION_FILE);
        std::fs::write(
            &calibration_path,
            serde_json::json!({
                "left": {
                    "fx": 100.0,
                    "fy": 100.0,
                    "cx": 1.0,
                    "cy": 1.0,
                    "width": 2,
                    "height": 2
                },
                "right": {
                    "fx": 100.0,
                    "fy": 100.0,
                    "cx": 1.0,
                    "cy": 1.0,
                    "width": 2,
                    "height": 2
                },
                "baseline_m": 0.1,
                "rectified": true,
                "imu": {
                    "noise": "invalid embedded imu block"
                }
            })
            .to_string(),
        )
        .expect("rewrite calibration");

        let open_err = DatasetReader::open(&dataset_dir)
            .expect_err("embedded imu block should fail without override");
        assert!(matches!(open_err, DatasetError::DeserializeJson { .. }));

        let mut reader = DatasetReader::open_with_imu_calibration_override(
            &dataset_dir,
            Some(
                calibration_with_time_offset_ns(10)
                    .imu
                    .expect("imu calibration"),
            ),
        )
        .expect("runtime override should replace embedded imu block before deserialization");
        let bundle = reader
            .bundles()
            .next()
            .expect("bundle")
            .expect("bundle should succeed");
        let imu = bundle.imu().batch().expect("imu batch");
        assert_eq!(imu.start_time().as_nanos(), 110);
        assert_eq!(imu.end_time().as_nanos(), 122);

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

    #[test]
    fn consecutive_startup_bundles_allow_missing_imu_until_first_sample() {
        let dataset_dir = unique_temp_dir("reader-imu-multi-startup-gap");
        let (writer, handle) =
            DatasetWriter::create(&dataset_dir, &meta_with_imu(), &calibration()).expect("writer");

        writer.write_frame(&mono_frame(SensorId::StereoLeft, 0, 100));
        writer.write_frame(&mono_frame(SensorId::StereoRight, 1, 104));
        writer.write_frame(&mono_frame(SensorId::StereoLeft, 2, 110));
        writer.write_frame(&mono_frame(SensorId::StereoRight, 3, 114));
        writer.write_frame(&mono_frame(SensorId::StereoLeft, 4, 150));
        writer.write_frame(&mono_frame(SensorId::StereoRight, 5, 154));
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
        let mut bundles = reader.bundles();

        let first = bundles.next().expect("first bundle").expect("first ok");
        assert!(matches!(first.imu(), CaptureImu::Absent));

        let second = bundles.next().expect("second bundle").expect("second ok");
        assert!(matches!(second.imu(), CaptureImu::Absent));

        let third = bundles.next().expect("third bundle").expect("third ok");
        let third_imu = third.imu().batch().expect("third bundle imu");
        assert_eq!(third_imu.start_time().as_nanos(), 150);
        assert_eq!(third_imu.end_time().as_nanos(), 150);

        assert!(bundles.next().is_none());

        let _ = std::fs::remove_dir_all(&dataset_dir);
    }
}
