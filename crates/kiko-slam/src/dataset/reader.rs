use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::{
    DepthImage, Frame, FrameId, PairingWindowNs, SensorId, StereoCalibration, StereoPair, Timestamp,
};

use super::{
    DatasetError, DatasetTimebase, DeltaStats, DepthImageContract, Manifest, ManifestFrameRef,
    ManifestHeader, ManifestPairing, ManifestPairingPolicy, ManifestStats, MonoImageContract,
    MonoPayloadFormat, PairReason, ParsedMonoContract, build_delta_stats, format,
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
struct ParsedManifest {
    entries: Vec<DatasetEntry>,
    depth: ParsedDepthStream,
    stats: DatasetStats,
}

#[derive(Clone, Copy, Debug)]
struct ParsedManifestContract {
    image: MonoImageContract,
    depth: Option<DepthImageContract>,
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
        Ok(Self {
            image,
            depth,
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
        let parsed = parse_manifest(&root, manifest, contract)?;
        Ok(Self {
            meta,
            stereo_calibration: parsed_mono.stereo,
            entries: parsed.entries,
            depth: parsed.depth,
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
