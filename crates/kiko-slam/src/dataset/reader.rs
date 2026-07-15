use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

use crate::{Frame, FrameDimensions, FrameId, PairingWindowNs, SensorId, StereoPair, Timestamp};

use super::{
    Calibration, DatasetError, Manifest, ManifestFrameRef, ManifestPairing, ManifestStats, format,
    read_calibration, read_manifest, read_meta, sensor_to_str,
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
struct ParsedManifest {
    entries: Vec<DatasetEntry>,
    stats: DatasetStats,
}

#[derive(Debug)]
pub struct DatasetReader {
    meta: super::Meta,
    calibration: Calibration,
    entries: Vec<DatasetEntry>,
    stats: DatasetStats,
    dimensions: FrameDimensions,
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
        let mono = meta.mono.as_ref().ok_or(DatasetError::MissingMonoConfig)?;
        let dimensions = FrameDimensions::try_new(mono.width, mono.height)
            .map_err(|source| DatasetError::InvalidFrameDimensions { source })?;
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
        let parsed = parse_manifest(&root, manifest, dimensions, pairing_window)?;
        Ok(Self {
            meta,
            calibration,
            entries: parsed.entries,
            stats: parsed.stats,
            dimensions,
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

    pub fn stats(&self) -> DatasetStats {
        self.stats
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
        Frame::from_dimensions(
            sensor,
            match sensor {
                SensorId::StereoLeft => self.next_left_id(),
                SensorId::StereoRight => self.next_right_id(),
            },
            Timestamp::from_nanos(frame_ref.timestamp_ns),
            self.dimensions,
            data,
        )
        .map_err(|source| DatasetError::InvalidFrameData { path, source })
    }
}

fn parse_manifest(
    root: &Path,
    manifest: Manifest,
    dimensions: FrameDimensions,
    pairing_window: PairingWindowNs,
) -> Result<ParsedManifest, DatasetError> {
    let Manifest { stats, entries, .. } = manifest;
    let max_delta_ns =
        u64::try_from(pairing_window.as_ns()).map_err(|_| DatasetError::InvalidManifest {
            reason: "parsed pairing window is negative",
        })?;
    let mut parsed_entries = Vec::with_capacity(entries.len());
    let mut seen_paths = HashSet::with_capacity(entries.len().saturating_mul(2));
    let mut previous_left_timestamp_ns = None;

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

        let left = parse_frame_ref(root, entry.left, SensorId::StereoLeft, dimensions)?;
        insert_unique_frame_ref(&mut seen_paths, &left)?;
        let right = match entry.pairing {
            ManifestPairing::Paired { right, delta_ns } => {
                let right = parse_frame_ref(root, right, SensorId::StereoRight, dimensions)?;
                insert_unique_frame_ref(&mut seen_paths, &right)?;

                let derived_delta_ns = left.timestamp_ns.abs_diff(right.timestamp_ns);
                if delta_ns != derived_delta_ns {
                    return Err(DatasetError::ManifestPairDeltaMismatch {
                        left_timestamp_ns: left.timestamp_ns,
                        right_timestamp_ns: right.timestamp_ns,
                        declared_delta_ns: delta_ns,
                        derived_delta_ns,
                    });
                }
                if derived_delta_ns > max_delta_ns {
                    return Err(DatasetError::ManifestPairOutsideWindow {
                        left_timestamp_ns: left.timestamp_ns,
                        right_timestamp_ns: right.timestamp_ns,
                        delta_ns: derived_delta_ns,
                        max_delta_ns,
                    });
                }
                Some(right)
            }
            ManifestPairing::MissingRight { .. } => None,
        };
        parsed_entries.push(DatasetEntry { left, right });
        previous_left_timestamp_ns = Some(left_timestamp_ns);
    }

    let stats = DatasetStats::from_manifest(&parsed_entries, stats)?;
    Ok(ParsedManifest {
        entries: parsed_entries,
        stats,
    })
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

fn parse_frame_ref(
    root: &Path,
    frame_ref: ManifestFrameRef,
    sensor: SensorId,
    dimensions: FrameDimensions,
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
        sensor_to_str(sensor),
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
    let expected = u64::from(dimensions.width()) * u64::from(dimensions.height());
    let actual = metadata.len();
    if actual != expected {
        return Err(DatasetError::InvalidFrameLength {
            path: resolved,
            expected,
            actual,
        });
    }

    Ok(DatasetFrameRef {
        timestamp_ns: frame_ref.timestamp_ns,
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

        let left_fps = fps_from_timestamps(entries.iter().map(|entry| entry.left.timestamp_ns));
        let paired_fps = fps_from_timestamps(
            entries
                .iter()
                .filter(|entry| entry.right.is_some())
                .map(|entry| entry.left.timestamp_ns),
        );
        let right_fps = if right_orphan_count == 0 {
            fps_from_timestamps(
                entries
                    .iter()
                    .filter_map(|entry| entry.right.as_ref())
                    .map(|right| right.timestamp_ns),
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
