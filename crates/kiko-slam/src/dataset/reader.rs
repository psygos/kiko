use std::collections::HashSet;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::{
    DepthImage, DeviceSessionId, Frame, FrameId, ImuEvent, ImuReport, InertialOrderTracker,
    PairingWindowNs, SensorId, StereoCalibration, StereoPair, Timestamp,
};

use super::{
    DatasetError, DatasetTimebase, DeltaStats, DepthImageContract, IMU_ACCELERATION_UNIT,
    IMU_ANGULAR_VELOCITY_UNIT, IMU_AXES, IMU_COORDINATE_FRAME, IMU_DEVICE_SESSION_SEMANTICS,
    IMU_DEVICE_TIMEBASE, IMU_HOST_ARRIVAL_TIMEBASE, IMU_SAMPLE_TIMESTAMP_SEMANTICS,
    IMU_STREAM_FORMAT, IMU_STREAM_HEADER_BYTES, IMU_STREAM_RECORD_BYTES, IMU_STREAM_VERSION,
    ImuStreamError, ImuWireRecord, Manifest, ManifestFrameRef, ManifestHeader,
    ManifestImuExtrinsic, ManifestImuStream, ManifestPairing, ManifestPairingPolicy, ManifestStats,
    MonoImageContract, MonoPayloadFormat, PairReason, ParsedMonoContract, build_delta_stats,
    decode_imu_header, format, imu_stream_byte_len, map_inertial_order_error,
    parse_image_dimensions, read_calibration, read_manifest, read_meta, sensor_to_str,
};

#[derive(Clone, Debug)]
struct DatasetFrameRef {
    timestamp: Timestamp,
    path: PathBuf,
}

#[derive(Clone, Copy, Debug)]
enum ManifestFrameKind {
    Mono(SensorId),
    Depth,
}

impl ManifestFrameKind {
    fn sensor_name(self) -> &'static str {
        match self {
            Self::Mono(sensor) => sensor_to_str(sensor),
            Self::Depth => "depth",
        }
    }
}

#[derive(Clone, Debug)]
struct DatasetEntry {
    left: DatasetFrameRef,
    right: Option<DatasetFrameRef>,
}

#[derive(Clone, Debug)]
struct DatasetDepthRef {
    frame_id: FrameId,
    frame_ref: DatasetFrameRef,
}

#[derive(Debug)]
enum ParsedDepthStream {
    Unconfigured,
    LegacyUnindexed,
    Indexed {
        contract: DepthImageContract,
        entries: Arc<[DatasetDepthRef]>,
    },
}

#[derive(Debug)]
enum ParsedImuStream {
    Unconfigured,
    LegacyUnindexed,
    Indexed(ImuStreamDescriptor),
}

#[derive(Clone, Debug)]
struct ImuStreamDescriptor {
    path: PathBuf,
    session_id: DeviceSessionId,
    record_count: u64,
    event_count: usize,
    byte_len: u64,
}

#[derive(Debug)]
struct ParsedManifest {
    entries: Vec<DatasetEntry>,
    depth: ParsedDepthStream,
    imu: ParsedImuStream,
    stats: DatasetStats,
}

#[derive(Clone, Copy, Debug)]
struct ParsedManifestContract {
    image: MonoImageContract,
    depth: Option<DepthImageContract>,
    imu_nominal_rate_hz: Option<u32>,
    pairing_policy: ManifestPairingPolicy,
    pairing_window: PairingWindowNs,
}

impl ParsedManifestContract {
    fn parse(
        header: &ManifestHeader,
        image: MonoImageContract,
        meta: &super::Meta,
    ) -> Result<Self, DatasetError> {
        MonoPayloadFormat::parse(&header.format)?;
        DatasetTimebase::parse(&header.timebase)?;
        let manifest_dimensions =
            parse_image_dimensions("manifest.header", header.width, header.height)?;
        image.require_dimensions("manifest.header", manifest_dimensions)?;
        if header.fps == 0 {
            return Err(DatasetError::InvalidNominalFps {
                field: "manifest.header.fps",
                value: header.fps,
            });
        }
        let expected_fps = image.nominal_fps_hz().get();
        if header.fps != expected_fps {
            return Err(DatasetError::NominalFpsMismatch {
                expected_field: "meta.mono.fps",
                expected: expected_fps,
                actual_field: "manifest.header.fps",
                actual: header.fps,
            });
        }
        for (expected_field, expected, actual_field, actual) in [
            (
                "meta.created",
                meta.created.as_str(),
                "manifest.header.created_at",
                header.created_at.as_str(),
            ),
            (
                "meta.device",
                meta.device.as_str(),
                "manifest.header.device",
                header.device.as_str(),
            ),
        ] {
            if actual != expected {
                return Err(DatasetError::ManifestMetadataMismatch {
                    expected_field,
                    expected: expected.to_string(),
                    actual_field,
                    actual: actual.to_string(),
                });
            }
        }
        let pairing_policy = ManifestPairingPolicy::parse(&header.pairing_policy)?;
        let pairing_window =
            PairingWindowNs::try_from_u64(header.pairing_window_ns).map_err(|_| {
                DatasetError::ManifestPairingWindowOutOfRange {
                    value: header.pairing_window_ns,
                }
            })?;
        let depth = meta
            .depth
            .as_ref()
            .map(DepthImageContract::parse)
            .transpose()?;
        let imu_nominal_rate_hz = meta.imu.as_ref().map(|imu| imu.rate_hz);
        Ok(Self {
            image,
            depth,
            imu_nominal_rate_hz,
            pairing_policy,
            pairing_window,
        })
    }
}

#[derive(Debug)]
pub struct DatasetReader {
    meta: super::Meta,
    stereo_calibration: StereoCalibration,
    entries: Vec<DatasetEntry>,
    depth: ParsedDepthStream,
    imu: ParsedImuStream,
    depth_projection: Option<super::DepthProjectionContract>,
    stats: DatasetStats,
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
        let parsed_mono = ParsedMonoContract::parse(&meta, &calibration)?;
        let image = parsed_mono.image;
        let manifest = read_manifest(&root)?;
        let contract = ParsedManifestContract::parse(&manifest.header, image, &meta)?;
        let depth_projection = contract.depth.map(DepthImageContract::projection);
        let parsed = parse_manifest(&root, manifest, contract)?;
        Ok(Self {
            meta,
            stereo_calibration: parsed_mono.stereo,
            entries: parsed.entries,
            depth: parsed.depth,
            imu: parsed.imu,
            depth_projection,
            stats: parsed.stats,
            left_seq: 0,
            right_seq: 0,
        })
    }

    pub fn meta(&self) -> &super::Meta {
        &self.meta
    }

    /// Structurally parsed stereo calibration retained from the dataset
    /// boundary. Rectification compatibility remains caller policy.
    pub fn stereo_calibration(&self) -> &StereoCalibration {
        &self.stereo_calibration
    }

    /// Return the depth projection contract parsed once from dataset metadata.
    pub fn depth_projection_contract(&self) -> Option<super::DepthProjectionContract> {
        self.depth_projection
    }

    pub fn stats(&self) -> DatasetStats {
        self.stats
    }

    /// Returns an independent cursor over the manifest-defined depth stream.
    ///
    /// Legacy manifests remain available for stereo replay, but depth replay requires explicit
    /// `depth_entries`; the reader never falls back to scanning the dataset directory.
    pub fn depth_cursor(&self) -> Result<DatasetDepthCursor, DatasetError> {
        match &self.depth {
            ParsedDepthStream::Unconfigured => Err(DatasetError::DepthStreamNotConfigured),
            ParsedDepthStream::LegacyUnindexed => Err(DatasetError::LegacyDepthManifestUnindexed),
            ParsedDepthStream::Indexed { contract, entries } => Ok(DatasetDepthCursor {
                contract: *contract,
                entries: Arc::clone(entries),
                next: 0,
            }),
        }
    }

    /// Returns a deterministic device-timestamp-ordered IMU event cursor.
    pub fn imu_cursor(&self) -> Result<DatasetImuCursor, DatasetError> {
        match &self.imu {
            ParsedImuStream::Unconfigured => Err(DatasetError::ImuStreamNotConfigured),
            ParsedImuStream::LegacyUnindexed => Err(DatasetError::LegacyImuManifestUnindexed),
            ParsedImuStream::Indexed(stream) => DatasetImuCursor::open(stream),
        }
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

/// Independent replay cursor over depth payloads named by the dataset manifest.
///
/// Replayed frame IDs are the zero-based positions of the entries in that manifest.
#[derive(Debug)]
pub struct DatasetDepthCursor {
    contract: DepthImageContract,
    entries: Arc<[DatasetDepthRef]>,
    next: usize,
}

impl DatasetDepthCursor {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn remaining(&self) -> usize {
        self.entries.len() - self.next
    }

    /// Decodes the next manifest entry when its timestamp is no later than `cutoff`.
    ///
    /// A future entry returns `Ok(None)` without consuming it. I/O, length, or sample-domain
    /// failures also leave the cursor unchanged so callers can report or retry the exact entry.
    pub fn next_at_or_before(
        &mut self,
        cutoff: Timestamp,
    ) -> Result<Option<DepthImage>, DatasetError> {
        let Some(entry) = self.entries.get(self.next) else {
            return Ok(None);
        };
        if entry.frame_ref.timestamp > cutoff {
            return Ok(None);
        }

        let depth = read_depth_image(entry, self.contract)?;
        self.next += 1;
        Ok(Some(depth))
    }
}

fn read_depth_image(
    entry: &DatasetDepthRef,
    contract: DepthImageContract,
) -> Result<DepthImage, DatasetError> {
    let path = &entry.frame_ref.path;
    let bytes = std::fs::read(path).map_err(|source| DatasetError::ReadFile {
        path: path.clone(),
        source,
    })?;
    let actual =
        u64::try_from(bytes.len()).map_err(|_| DatasetError::DepthPayloadLengthOutOfRange {
            path: path.clone(),
            actual: bytes.len(),
        })?;
    let expected = contract.expected_payload_len();
    if actual != expected {
        return Err(DatasetError::DepthPayloadLengthChanged {
            path: path.clone(),
            expected,
            actual,
        });
    }

    let dimensions = contract.dimensions();
    DepthImage::new(
        entry.frame_id,
        entry.frame_ref.timestamp,
        dimensions.width(),
        dimensions.height(),
        contract.decode(&bytes),
    )
    .map_err(|source| DatasetError::InvalidDepthData {
        path: path.clone(),
        source,
    })
}

/// Bounded-memory replay cursor that merges the individually monotonic
/// accelerometer and gyroscope streams by device timestamp.
///
/// Dataset open validates the stream in one sequential pass. The cursor then
/// owns two further sequential readers and retains at most one decoded report
/// per sensor so it can merge independent timestamps without retaining the
/// recording. Exact timestamp ties emit accelerometer before gyroscope;
/// sequence order is retained within each sensor stream. Host arrival and
/// dequeue sequence remain attached to both events from their originating
/// bridge report.
#[derive(Debug)]
pub struct DatasetImuCursor {
    accel: ImuReportFileCursor,
    gyro: ImuReportFileCursor,
    event_count: usize,
    emitted: usize,
}

impl DatasetImuCursor {
    fn open(stream: &ImuStreamDescriptor) -> Result<Self, DatasetError> {
        Ok(Self {
            accel: ImuReportFileCursor::open(stream)?,
            gyro: ImuReportFileCursor::open(stream)?,
            event_count: stream.event_count,
            emitted: 0,
        })
    }

    /// Total number of sensor events declared by the validated stream.
    pub fn len(&self) -> usize {
        self.event_count
    }

    pub fn is_empty(&self) -> bool {
        self.event_count == 0
    }

    pub fn remaining(&self) -> usize {
        self.event_count - self.emitted
    }

    /// Decode and return the next globally ordered event.
    ///
    /// Any post-open I/O, length, sample-domain, or ordering failure leaves the
    /// affected report unconsumed, so a caller can report the exact failure or
    /// retry after restoring the payload.
    pub fn next_event(&mut self) -> Result<Option<ImuEvent>, DatasetError> {
        let accel = self.accel.peek()?;
        let gyro = self.gyro.peek()?;
        let event = match (accel, gyro) {
            (None, None) => return Ok(None),
            (Some(_), None) => accel_event(self.accel.consume()),
            (None, Some(_)) => gyro_event(self.gyro.consume()),
            (Some(accel_report), Some(gyro_report))
                if accel_report.accel().timestamp() <= gyro_report.gyro().timestamp() =>
            {
                accel_event(self.accel.consume())
            }
            (Some(_), Some(_)) => gyro_event(self.gyro.consume()),
        };
        self.emitted += 1;
        Ok(Some(event))
    }
}

impl Iterator for DatasetImuCursor {
    type Item = Result<ImuEvent, DatasetError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_event() {
            Ok(Some(event)) => Some(Ok(event)),
            Ok(None) => None,
            Err(error) => Some(Err(error)),
        }
    }
}

#[derive(Debug)]
struct ImuReportFileCursor {
    path: PathBuf,
    reader: BufReader<std::fs::File>,
    session_id: DeviceSessionId,
    record_count: u64,
    expected_len: u64,
    next_record_index: u64,
    retry_requires_seek: bool,
    eof_validated: bool,
    order: InertialOrderTracker,
    lookahead: Option<ImuReport>,
}

impl ImuReportFileCursor {
    fn open(stream: &ImuStreamDescriptor) -> Result<Self, DatasetError> {
        let mut file =
            std::fs::File::open(&stream.path).map_err(|source| DatasetError::ReadFile {
                path: stream.path.clone(),
                source,
            })?;
        require_imu_file_len(&file, &stream.path, stream.byte_len)?;
        let mut header = [0_u8; IMU_STREAM_HEADER_BYTES as usize];
        file.read_exact(&mut header)
            .map_err(|source| DatasetError::ReadFile {
                path: stream.path.clone(),
                source,
            })?;
        let encoded_count =
            decode_imu_header(&header).map_err(|source| DatasetError::InvalidImuStream {
                path: stream.path.clone(),
                source,
            })?;
        if encoded_count != stream.record_count {
            return Err(DatasetError::InvalidImuStream {
                path: stream.path.clone(),
                source: ImuStreamError::ManifestMismatch {
                    field: "record_count",
                    declared: stream.record_count,
                    encoded: encoded_count,
                },
            });
        }
        Ok(Self {
            path: stream.path.clone(),
            reader: BufReader::with_capacity(usize::from(IMU_STREAM_RECORD_BYTES), file),
            session_id: stream.session_id,
            record_count: stream.record_count,
            expected_len: stream.byte_len,
            next_record_index: 0,
            retry_requires_seek: false,
            eof_validated: false,
            order: InertialOrderTracker::with_session(stream.session_id),
            lookahead: None,
        })
    }

    fn peek(&mut self) -> Result<Option<ImuReport>, DatasetError> {
        if let Some(report) = self.lookahead {
            return Ok(Some(report));
        }
        if self.next_record_index == self.record_count {
            if !self.eof_validated {
                require_imu_file_len(self.reader.get_ref(), &self.path, self.expected_len)?;
                self.eof_validated = true;
            }
            return Ok(None);
        }

        if self.retry_requires_seek {
            let offset = imu_stream_byte_len(self.next_record_index).map_err(|source| {
                DatasetError::InvalidImuStream {
                    path: self.path.clone(),
                    source,
                }
            })?;
            self.reader
                .seek(SeekFrom::Start(offset))
                .map_err(|source| DatasetError::ReadFile {
                    path: self.path.clone(),
                    source,
                })?;
            self.retry_requires_seek = false;
        }

        let mut bytes = [0_u8; IMU_STREAM_RECORD_BYTES as usize];
        if let Err(source) = self.reader.read_exact(&mut bytes) {
            self.retry_requires_seek = true;
            if let Ok(metadata) = self.reader.get_ref().metadata() {
                let actual = metadata.len();
                if actual != self.expected_len {
                    return Err(DatasetError::InvalidImuStream {
                        path: self.path.clone(),
                        source: ImuStreamError::ByteLengthMismatch {
                            expected: self.expected_len,
                            actual,
                        },
                    });
                }
            }
            return Err(DatasetError::ReadFile {
                path: self.path.clone(),
                source,
            });
        }
        let record_index = match usize::try_from(self.next_record_index) {
            Ok(record_index) => record_index,
            Err(_) => {
                self.retry_requires_seek = true;
                return Err(DatasetError::ManifestCountOutOfRange {
                    field: "imu_stream record index",
                    value: self.next_record_index,
                });
            }
        };
        let report = match ImuWireRecord::decode(&bytes, record_index)
            .and_then(|record| record.into_report(self.session_id, record_index))
        {
            Ok(report) => report,
            Err(source) => {
                self.retry_requires_seek = true;
                return Err(DatasetError::InvalidImuStream {
                    path: self.path.clone(),
                    source,
                });
            }
        };
        if let Err(source) = self.order.observe(&report) {
            self.retry_requires_seek = true;
            return Err(DatasetError::InvalidImuStream {
                path: self.path.clone(),
                source: map_inertial_order_error(record_index, source),
            });
        }
        self.next_record_index += 1;
        self.lookahead = Some(report);
        Ok(Some(report))
    }

    fn consume(&mut self) -> ImuReport {
        self.lookahead
            .take()
            .expect("consume follows a successful nonempty peek")
    }
}

fn require_imu_file_len(
    file: &std::fs::File,
    path: &Path,
    expected: u64,
) -> Result<(), DatasetError> {
    let actual = file
        .metadata()
        .map_err(|source| DatasetError::ReadFile {
            path: path.to_path_buf(),
            source,
        })?
        .len();
    if actual != expected {
        return Err(DatasetError::InvalidImuStream {
            path: path.to_path_buf(),
            source: ImuStreamError::ByteLengthMismatch { expected, actual },
        });
    }
    Ok(())
}

fn accel_event(report: ImuReport) -> ImuEvent {
    ImuEvent::Accel {
        session_id: report.session_id(),
        sequence: report.sequence(),
        host_arrival: report.host_arrival(),
        sample: report.accel(),
    }
}

fn gyro_event(report: ImuReport) -> ImuEvent {
    ImuEvent::Gyro {
        session_id: report.session_id(),
        sequence: report.sequence(),
        host_arrival: report.host_arrival(),
        sample: report.gyro(),
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

            return Some(Ok(StereoPair::from_parts(left_frame, right_frame)));
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
            let pair = StereoPair::from_parts(left_frame, right_frame);
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
        let path = frame_ref.path.clone();
        let data = std::fs::read(&path).map_err(|e| DatasetError::ReadFile {
            path: path.clone(),
            source: e,
        })?;
        let dimensions = self.stereo_calibration.dimensions();
        Frame::from_dimensions(
            sensor,
            match sensor {
                SensorId::StereoLeft => self.next_left_id(),
                SensorId::StereoRight => self.next_right_id(),
            },
            frame_ref.timestamp,
            dimensions,
            data,
        )
        .map_err(|source| DatasetError::InvalidFrameData { path, source })
    }
}

fn parse_manifest(
    root: &Path,
    manifest: Manifest,
    contract: ParsedManifestContract,
) -> Result<ParsedManifest, DatasetError> {
    let Manifest {
        stats,
        entries,
        depth_entries,
        imu_stream,
        ..
    } = manifest;
    if contract.pairing_policy == ManifestPairingPolicy::RecordedPairs
        && (stats.left_orphans != 0 || stats.right_orphans != 0)
    {
        return Err(DatasetError::RecordedPairsDeclareOrphans {
            left_orphans: stats.left_orphans,
            right_orphans: stats.right_orphans,
        });
    }
    let dimensions = contract.image.dimensions();
    let mono_payload_len = u64::from(dimensions.width()) * u64::from(dimensions.height());
    let max_delta_ns = contract.pairing_window.as_u64();
    let mut parsed_entries = Vec::with_capacity(entries.len());
    let mut paired_deltas = Vec::with_capacity(entries.len());
    let depth_entry_count = depth_entries.as_ref().map_or(0, Vec::len);
    let frame_ref_capacity = entries
        .len()
        .checked_mul(2)
        .and_then(|count| count.checked_add(depth_entry_count))
        .ok_or(DatasetError::InvalidManifest {
            reason: "manifest frame reference count exceeds the host address space",
        })?;
    let mut seen_paths = HashSet::with_capacity(frame_ref_capacity);
    let mut previous_left_timestamp_ns = None;
    let mut outside_window = 0u64;

    for entry in entries {
        let left_timestamp_ns = entry.left.timestamp_ns;
        if let Some(previous) = previous_left_timestamp_ns
            && left_timestamp_ns <= previous
        {
            return Err(DatasetError::NonMonotonicManifestEntries {
                previous_left_timestamp_ns: previous,
                current_left_timestamp_ns: left_timestamp_ns,
            });
        }

        let left = parse_manifest_frame_ref(
            root,
            entry.left,
            ManifestFrameKind::Mono(SensorId::StereoLeft),
            mono_payload_len,
        )?;
        insert_unique_frame_ref(&mut seen_paths, &left)?;
        let right = match entry.pairing {
            ManifestPairing::Paired { right, delta_ns } => {
                let right = parse_manifest_frame_ref(
                    root,
                    right,
                    ManifestFrameKind::Mono(SensorId::StereoRight),
                    mono_payload_len,
                )?;
                insert_unique_frame_ref(&mut seen_paths, &right)?;

                let left_timestamp_ns = left.timestamp.as_nanos();
                let right_timestamp_ns = right.timestamp.as_nanos();
                let derived_delta_ns = left_timestamp_ns.abs_diff(right_timestamp_ns);
                if delta_ns != derived_delta_ns {
                    return Err(DatasetError::ManifestPairDeltaMismatch {
                        left_timestamp_ns,
                        right_timestamp_ns,
                        declared_delta_ns: delta_ns,
                        derived_delta_ns,
                    });
                }
                if derived_delta_ns > max_delta_ns {
                    return Err(DatasetError::ManifestPairOutsideWindow {
                        left_timestamp_ns,
                        right_timestamp_ns,
                        delta_ns: derived_delta_ns,
                        max_delta_ns,
                    });
                }
                paired_deltas.push(derived_delta_ns);
                Some(right)
            }
            ManifestPairing::MissingRight { reason } => {
                if contract.pairing_policy == ManifestPairingPolicy::RecordedPairs {
                    return Err(DatasetError::RecordedPairsContainMissingRight {
                        left_timestamp_ns,
                    });
                }
                if matches!(reason, PairReason::OutsideWindow) {
                    outside_window = outside_window.saturating_add(1);
                }
                None
            }
        };
        parsed_entries.push(DatasetEntry { left, right });
        previous_left_timestamp_ns = Some(left_timestamp_ns);
    }

    let depth = parse_depth_stream(root, contract.depth, depth_entries, &mut seen_paths)?;
    let imu = parse_imu_stream(root, contract.imu_nominal_rate_hz, imu_stream)?;

    let derived_delta_stats = build_delta_stats(&paired_deltas);
    validate_manifest_delta_stats(stats.delta_stats.as_ref(), derived_delta_stats.as_ref())?;
    validate_manifest_stat(
        "drops_by_reason.outside_window",
        stats.drops_by_reason.outside_window,
        outside_window,
    )?;
    let stats = DatasetStats::from_manifest(&parsed_entries, stats)?;
    Ok(ParsedManifest {
        entries: parsed_entries,
        depth,
        imu,
        stats,
    })
}

fn parse_depth_stream(
    root: &Path,
    contract: Option<DepthImageContract>,
    entries: Option<Vec<ManifestFrameRef>>,
    seen_paths: &mut HashSet<PathBuf>,
) -> Result<ParsedDepthStream, DatasetError> {
    let (contract, entries) = match (contract, entries) {
        (None, None) => return Ok(ParsedDepthStream::Unconfigured),
        (Some(_), None) => return Ok(ParsedDepthStream::LegacyUnindexed),
        (None, Some(entries)) => {
            return Err(DatasetError::ManifestDepthEntriesWithoutMetadata {
                entry_count: entries.len(),
            });
        }
        (Some(contract), Some(entries)) => (contract, entries),
    };

    let mut parsed = Vec::with_capacity(entries.len());
    let mut previous_depth_timestamp_ns = None;
    let expected_payload_len = contract.expected_payload_len();
    for (index, frame_ref) in entries.into_iter().enumerate() {
        let timestamp_ns = frame_ref.timestamp_ns;
        if let Some(previous) = previous_depth_timestamp_ns
            && timestamp_ns <= previous
        {
            return Err(DatasetError::NonMonotonicManifestDepthEntries {
                previous_depth_timestamp_ns: previous,
                current_depth_timestamp_ns: timestamp_ns,
            });
        }

        let frame_ref = parse_manifest_frame_ref(
            root,
            frame_ref,
            ManifestFrameKind::Depth,
            expected_payload_len,
        )?;
        insert_unique_frame_ref(seen_paths, &frame_ref)?;
        let frame_id = u64::try_from(index)
            .map(FrameId::new)
            .map_err(|_| DatasetError::DepthFrameIndexOutOfRange { index })?;
        parsed.push(DatasetDepthRef {
            frame_id,
            frame_ref,
        });
        previous_depth_timestamp_ns = Some(timestamp_ns);
    }

    Ok(ParsedDepthStream::Indexed {
        contract,
        entries: Arc::from(parsed.into_boxed_slice()),
    })
}

fn parse_imu_stream(
    root: &Path,
    nominal_rate_hz: Option<u32>,
    manifest: Option<ManifestImuStream>,
) -> Result<ParsedImuStream, DatasetError> {
    let (nominal_rate_hz, manifest) = match (nominal_rate_hz, manifest) {
        (None, None) => return Ok(ParsedImuStream::Unconfigured),
        (Some(_), None) => return Ok(ParsedImuStream::LegacyUnindexed),
        (None, Some(_)) => return Err(DatasetError::ManifestImuStreamWithoutMetadata),
        (Some(rate), Some(manifest)) => (rate, manifest),
    };
    if nominal_rate_hz == 0 {
        return Err(DatasetError::InvalidImuNominalRate {
            value: nominal_rate_hz,
        });
    }
    if manifest.nominal_rate_hz != nominal_rate_hz {
        return Err(DatasetError::ImuNominalRateMismatch {
            metadata: nominal_rate_hz,
            manifest: manifest.nominal_rate_hz,
        });
    }

    require_imu_manifest_value("imu_stream.format", &manifest.format, IMU_STREAM_FORMAT)?;
    require_imu_manifest_value(
        "imu_stream.acceleration_unit",
        &manifest.acceleration_unit,
        IMU_ACCELERATION_UNIT,
    )?;
    require_imu_manifest_value(
        "imu_stream.angular_velocity_unit",
        &manifest.angular_velocity_unit,
        IMU_ANGULAR_VELOCITY_UNIT,
    )?;
    require_imu_manifest_value(
        "imu_stream.coordinate_frame",
        &manifest.coordinate_frame,
        IMU_COORDINATE_FRAME,
    )?;
    require_imu_manifest_value("imu_stream.axes", &manifest.axes, IMU_AXES)?;
    require_imu_manifest_value(
        "imu_stream.device_timebase",
        &manifest.device_timebase,
        IMU_DEVICE_TIMEBASE,
    )?;
    require_imu_manifest_value(
        "imu_stream.device_session_semantics",
        &manifest.device_session_semantics,
        IMU_DEVICE_SESSION_SEMANTICS,
    )?;
    require_imu_manifest_value(
        "imu_stream.sample_timestamp_semantics",
        &manifest.sample_timestamp_semantics,
        IMU_SAMPLE_TIMESTAMP_SEMANTICS,
    )?;
    require_imu_manifest_value(
        "imu_stream.host_arrival_timebase",
        &manifest.host_arrival_timebase,
        IMU_HOST_ARRIVAL_TIMEBASE,
    )?;
    if let ManifestImuExtrinsic::CalibratedToTrackingCamera { source } = &manifest.extrinsic
        && source.trim().is_empty()
    {
        return Err(DatasetError::InvalidManifest {
            reason: "calibrated IMU extrinsic provenance must name its source",
        });
    }

    let path = parse_imu_stream_path(root, &manifest.path)?;
    for (declared, expected, error) in [
        (
            manifest.version,
            IMU_STREAM_VERSION,
            ImuStreamError::UnsupportedVersion {
                expected: IMU_STREAM_VERSION,
                actual: manifest.version,
            },
        ),
        (
            manifest.header_bytes,
            IMU_STREAM_HEADER_BYTES,
            ImuStreamError::HeaderLengthMismatch {
                expected: IMU_STREAM_HEADER_BYTES,
                actual: manifest.header_bytes,
            },
        ),
        (
            manifest.record_bytes,
            IMU_STREAM_RECORD_BYTES,
            ImuStreamError::RecordLengthMismatch {
                expected: IMU_STREAM_RECORD_BYTES,
                actual: manifest.record_bytes,
            },
        ),
    ] {
        if declared != expected {
            return Err(DatasetError::InvalidImuStream {
                path: path.clone(),
                source: error,
            });
        }
    }

    let session_id = DeviceSessionId::try_new(manifest.device_session_id).map_err(|_| {
        DatasetError::InvalidManifest {
            reason: "IMU device_session_id must be nonzero",
        }
    })?;
    let expected_len = imu_stream_byte_len(manifest.record_count).map_err(|source| {
        DatasetError::InvalidImuStream {
            path: path.clone(),
            source,
        }
    })?;
    if manifest.byte_len != expected_len {
        return Err(DatasetError::InvalidImuStream {
            path: path.clone(),
            source: ImuStreamError::ManifestMismatch {
                field: "byte_len",
                declared: manifest.byte_len,
                encoded: expected_len,
            },
        });
    }
    let metadata = std::fs::metadata(&path).map_err(|source| DatasetError::ReadFile {
        path: path.clone(),
        source,
    })?;
    if !metadata.is_file() {
        return Err(DatasetError::InvalidImuFileType { path });
    }
    if metadata.len() != expected_len {
        return Err(DatasetError::InvalidImuStream {
            path,
            source: ImuStreamError::ByteLengthMismatch {
                expected: expected_len,
                actual: metadata.len(),
            },
        });
    }

    let file = std::fs::File::open(&path).map_err(|source| DatasetError::ReadFile {
        path: path.clone(),
        source,
    })?;
    require_imu_file_len(&file, &path, expected_len)?;
    let mut reader = BufReader::with_capacity(usize::from(IMU_STREAM_RECORD_BYTES), file);
    let mut header = [0_u8; IMU_STREAM_HEADER_BYTES as usize];
    reader
        .read_exact(&mut header)
        .map_err(|source| DatasetError::ReadFile {
            path: path.clone(),
            source,
        })?;
    let encoded_count =
        decode_imu_header(&header).map_err(|source| DatasetError::InvalidImuStream {
            path: path.clone(),
            source,
        })?;
    if encoded_count != manifest.record_count {
        return Err(DatasetError::InvalidImuStream {
            path: path.clone(),
            source: ImuStreamError::ManifestMismatch {
                field: "record_count",
                declared: manifest.record_count,
                encoded: encoded_count,
            },
        });
    }
    let record_count = usize::try_from(manifest.record_count).map_err(|_| {
        DatasetError::ManifestCountOutOfRange {
            field: "imu_stream.record_count",
            value: manifest.record_count,
        }
    })?;
    let event_count = record_count
        .checked_mul(2)
        .ok_or(DatasetError::InvalidManifest {
            reason: "IMU event count exceeds the host address space",
        })?;
    let mut order = InertialOrderTracker::with_session(session_id);
    let mut record_bytes = [0_u8; IMU_STREAM_RECORD_BYTES as usize];
    for record_index in 0..record_count {
        reader
            .read_exact(&mut record_bytes)
            .map_err(|source| DatasetError::ReadFile {
                path: path.clone(),
                source,
            })?;
        let record = ImuWireRecord::decode(&record_bytes, record_index).map_err(|source| {
            DatasetError::InvalidImuStream {
                path: path.clone(),
                source,
            }
        })?;
        let report = record
            .into_report(session_id, record_index)
            .map_err(|source| DatasetError::InvalidImuStream {
                path: path.clone(),
                source,
            })?;
        order
            .observe(&report)
            .map_err(|source| DatasetError::InvalidImuStream {
                path: path.clone(),
                source: map_inertial_order_error(record_index, source),
            })?;
    }
    Ok(ParsedImuStream::Indexed(ImuStreamDescriptor {
        path,
        session_id,
        record_count: manifest.record_count,
        event_count,
        byte_len: expected_len,
    }))
}

fn require_imu_manifest_value(
    field: &'static str,
    actual: &str,
    expected: &'static str,
) -> Result<(), DatasetError> {
    if actual != expected {
        return Err(DatasetError::UnsupportedManifestValue {
            field,
            value: actual.to_string(),
        });
    }
    Ok(())
}

fn parse_imu_stream_path(root: &Path, declared: &str) -> Result<PathBuf, DatasetError> {
    let relative = Path::new(declared);
    if relative != Path::new(format::IMU_STREAM_FILE) {
        return Err(DatasetError::InvalidFramePath {
            path: declared.to_string(),
            reason: "IMU stream path must be the canonical versioned root payload",
        });
    }
    let candidate = root.join(relative);
    let resolved = std::fs::canonicalize(&candidate).map_err(|source| DatasetError::ReadFile {
        path: candidate,
        source,
    })?;
    if !resolved.starts_with(root) {
        return Err(DatasetError::InvalidFramePath {
            path: declared.to_string(),
            reason: "resolved IMU stream path escapes the dataset root",
        });
    }
    Ok(resolved)
}

fn validate_manifest_delta_stats(
    declared: Option<&DeltaStats>,
    derived: Option<&DeltaStats>,
) -> Result<(), DatasetError> {
    let (Some(declared), Some(derived)) = (declared, derived) else {
        if declared.is_some() != derived.is_some() {
            return Err(DatasetError::ManifestDeltaStatsPresenceMismatch {
                declared: declared.is_some(),
                derived: derived.is_some(),
            });
        }
        return Ok(());
    };

    for (field, declared, derived) in [
        ("delta_stats.min", declared.min, derived.min),
        ("delta_stats.median", declared.median, derived.median),
        ("delta_stats.p95", declared.p95, derived.p95),
        ("delta_stats.p99", declared.p99, derived.p99),
        ("delta_stats.max", declared.max, derived.max),
    ] {
        validate_manifest_stat(field, declared, derived)?;
    }
    Ok(())
}

fn insert_unique_frame_ref(
    seen_paths: &mut HashSet<PathBuf>,
    frame_ref: &DatasetFrameRef,
) -> Result<(), DatasetError> {
    if !seen_paths.insert(frame_ref.path.clone()) {
        return Err(DatasetError::DuplicateManifestFrameRef {
            path: frame_ref.path.clone(),
        });
    }
    Ok(())
}

fn parse_manifest_frame_ref(
    root: &Path,
    frame_ref: ManifestFrameRef,
    kind: ManifestFrameKind,
    expected_payload_len: u64,
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

    let expected_relative = Path::new(format::FRAMES_DIR).join(format::frame_name(
        frame_ref.timestamp_ns,
        kind.sensor_name(),
    ));
    if relative != expected_relative {
        return Err(DatasetError::ManifestFrameIdentityMismatch {
            declared: frame_ref.path,
            expected: expected_relative,
        });
    }

    let metadata = std::fs::metadata(&resolved).map_err(|source| DatasetError::ReadFile {
        path: resolved.clone(),
        source,
    })?;
    if !metadata.is_file() {
        return Err(DatasetError::InvalidFrameFileType { path: resolved });
    }
    let actual = metadata.len();
    if actual != expected_payload_len {
        return Err(DatasetError::InvalidFrameLength {
            path: resolved,
            expected: expected_payload_len,
            actual,
        });
    }

    Ok(DatasetFrameRef {
        timestamp: Timestamp::from_nanos(frame_ref.timestamp_ns),
        path: resolved,
    })
}

/// Statistics for the manifest-defined replay stream.
///
/// `right_count` includes declared right orphans. Because the current manifest
/// does not record their timestamps, `right_fps` is unavailable whenever any
/// right orphan exists.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DatasetStats {
    pub left_count: usize,
    pub right_count: usize,
    pub paired_count: usize,
    pub left_orphan_count: usize,
    pub right_orphan_count: usize,
    pub left_fps: Option<f64>,
    pub right_fps: Option<f64>,
    pub paired_fps: Option<f64>,
}

impl DatasetStats {
    fn from_manifest(
        entries: &[DatasetEntry],
        declared: ManifestStats,
    ) -> Result<Self, DatasetError> {
        let left_count = entries.len();
        let paired_count = entries.iter().filter(|entry| entry.right.is_some()).count();
        let left_orphan_count = left_count - paired_count;
        let left_count_u64 =
            u64::try_from(left_count).map_err(|_| DatasetError::InvalidManifest {
                reason: "left entry count exceeds the manifest count representation",
            })?;
        let paired_count_u64 =
            u64::try_from(paired_count).map_err(|_| DatasetError::InvalidManifest {
                reason: "paired entry count exceeds the manifest count representation",
            })?;
        let left_orphan_count_u64 =
            u64::try_from(left_orphan_count).map_err(|_| DatasetError::InvalidManifest {
                reason: "left orphan count exceeds the manifest count representation",
            })?;

        validate_manifest_stat("total_left", declared.total_left, left_count_u64)?;
        validate_manifest_stat("paired_count", declared.paired_count, paired_count_u64)?;
        validate_manifest_stat("left_orphans", declared.left_orphans, left_orphan_count_u64)?;
        let derived_right_count = paired_count_u64.checked_add(declared.right_orphans).ok_or(
            DatasetError::InvalidManifest {
                reason: "paired and right-orphan counts overflow u64",
            },
        )?;
        validate_manifest_stat("total_right", declared.total_right, derived_right_count)?;

        let right_count = usize::try_from(declared.total_right).map_err(|_| {
            DatasetError::ManifestCountOutOfRange {
                field: "total_right",
                value: declared.total_right,
            }
        })?;
        let right_orphan_count = usize::try_from(declared.right_orphans).map_err(|_| {
            DatasetError::ManifestCountOutOfRange {
                field: "right_orphans",
                value: declared.right_orphans,
            }
        })?;

        let left_fps =
            fps_from_timestamps(entries.iter().map(|entry| entry.left.timestamp.as_nanos()));
        let paired_fps = fps_from_timestamps(
            entries
                .iter()
                .filter(|entry| entry.right.is_some())
                .map(|entry| entry.left.timestamp.as_nanos()),
        );
        let right_fps = if right_orphan_count == 0 {
            fps_from_timestamps(
                entries
                    .iter()
                    .filter_map(|entry| entry.right.as_ref())
                    .map(|right| right.timestamp.as_nanos()),
            )
        } else {
            None
        };

        Ok(Self {
            left_count,
            right_count,
            paired_count,
            left_orphan_count,
            right_orphan_count,
            left_fps,
            right_fps,
            paired_fps,
        })
    }
}

fn validate_manifest_stat(
    field: &'static str,
    declared: u64,
    derived: u64,
) -> Result<(), DatasetError> {
    if declared != derived {
        return Err(DatasetError::InvalidManifestStats {
            field,
            declared,
            derived,
        });
    }
    Ok(())
}

fn fps_from_timestamps(timestamps: impl IntoIterator<Item = i64>) -> Option<f64> {
    let mut count = 0usize;
    let mut min_ts = i64::MAX;
    let mut max_ts = i64::MIN;
    for timestamp_ns in timestamps {
        count = count.saturating_add(1);
        min_ts = min_ts.min(timestamp_ns);
        max_ts = max_ts.max(timestamp_ns);
    }
    if count < 2 {
        return None;
    }
    let span_ns = max_ts.abs_diff(min_ts);
    if span_ns == 0 {
        return None;
    }
    Some((count - 1) as f64 * 1_000_000_000.0 / span_ns as f64)
}
