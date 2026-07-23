//! Exact persisted-map replay and fresh-camera relocalization gating.
//!
//! A warm occupancy artifact is useful only when the same sparse tracker and
//! dense mapper that reconstructed it continue into the live runtime. This
//! module owns that transition. It never derives pose from occupancy and never
//! calls an ordinary successful tracking update "relocalized".

use std::fmt;
use std::io::Read;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::dataset::{
    DatasetDepthCursor, DatasetError, DatasetReader, DepthOpticalFrame, DepthProjectionContract,
};
use crate::dense::command_mapper::{
    DEPTH_ASSOCIATION_WINDOW, DenseCommandGeneration, DenseCommandGenerationError,
    DenseCommandMappingError, apply_pose_updates_command, map_output_to_dense_commands,
};
use crate::dense::occupancy::{DepthCameraModel, OccupancyError};
use crate::dense::occupancy_persistence::{ReplayOccupancyEvidence, ReplayOccupancyEvidenceError};
use crate::dense::occupancy_runtime::{
    OccupancyRuntime, OccupancyRuntimeConfig, OccupancyRuntimeError, TimedOccupancySnapshot,
};
use crate::dense::ring_buffer::{DepthRingBuffer, DepthRingBufferError};
use crate::map::{KeyframeId, MapSnapshot};
use crate::tracker::{DatasetReplayRelocalizationArmError, DatasetReplayTrackerQuiesceError};
use crate::{
    DepthImage, DiagnosticEvent, Frame, FrameDimensions, FrameId, MapLocalization,
    PinholeIntrinsics, Pose, SlamTracker, StereoCalibration, StereoPair, Timestamp, TrackerError,
    TrackerOutput, TrackingHealth,
};

use super::{
    NanoDatasetContentBindingStatus, NanoDatasetReplayRequired, NanoWarmSelectionError,
    NanoWarmStartReplayBindError, NavigationIngressCapacity, NavigationIngressCapacityError,
    NavigationIngressEvent, NavigationIngressReader, NavigationIngressStreamReadError,
    RecordedMapEpochId,
};

const REPLAY_FRAME_NAMESPACE_BIT: u64 = 1_u64 << 63;
const DEFAULT_REPLAY_DEPTH_RING_CAPACITY: usize = 8;
const DEFAULT_REPLAY_QUIESCENCE_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_REPLAY_QUIESCENCE_TIMEOUT: Duration = Duration::from_secs(60);

/// Bounded startup work policy parsed before replay begins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NanoWarmStartReplayConfig {
    depth_ring_capacity: NonZeroUsize,
    quiescence_timeout: Duration,
}

impl NanoWarmStartReplayConfig {
    pub fn try_new(
        depth_ring_capacity: usize,
        quiescence_timeout: Duration,
    ) -> Result<Self, NanoWarmStartReplayConfigError> {
        let depth_ring_capacity = NonZeroUsize::new(depth_ring_capacity)
            .ok_or(NanoWarmStartReplayConfigError::ZeroDepthRingCapacity)?;
        if quiescence_timeout.is_zero() || quiescence_timeout > MAX_REPLAY_QUIESCENCE_TIMEOUT {
            return Err(
                NanoWarmStartReplayConfigError::QuiescenceTimeoutOutOfRange {
                    actual: quiescence_timeout,
                    maximum: MAX_REPLAY_QUIESCENCE_TIMEOUT,
                },
            );
        }
        Ok(Self {
            depth_ring_capacity,
            quiescence_timeout,
        })
    }

    pub fn depth_ring_capacity(self) -> usize {
        self.depth_ring_capacity.get()
    }

    pub fn quiescence_timeout(self) -> Duration {
        self.quiescence_timeout
    }
}

impl Default for NanoWarmStartReplayConfig {
    fn default() -> Self {
        Self {
            depth_ring_capacity: NonZeroUsize::new(DEFAULT_REPLAY_DEPTH_RING_CAPACITY)
                .expect("default replay depth ring capacity is nonzero"),
            quiescence_timeout: DEFAULT_REPLAY_QUIESCENCE_TIMEOUT,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoWarmStartReplayConfigError {
    ZeroDepthRingCapacity,
    QuiescenceTimeoutOutOfRange { actual: Duration, maximum: Duration },
}

impl fmt::Display for NanoWarmStartReplayConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDepthRingCapacity => {
                formatter.write_str("warm-start replay depth ring capacity must be nonzero")
            }
            Self::QuiescenceTimeoutOutOfRange { actual, maximum } => write!(
                formatter,
                "warm-start replay quiescence timeout {actual:?} must be in (0, {maximum:?}]"
            ),
        }
    }
}

impl std::error::Error for NanoWarmStartReplayConfigError {}

/// Auditable replay completion facts. No field claims live localization.
#[derive(Debug)]
pub struct NanoWarmStartReplayReceipt {
    occupancy_snapshot_path: PathBuf,
    slam_dataset_directory_path: PathBuf,
    processed_stereo_pairs: u64,
    final_replay_map: MapSnapshot,
    replay_diagnostic_events: u64,
    replay_map_corrections: u64,
    selected_map_epoch_id: RecordedMapEpochId,
    selected_map_revision: u64,
    dataset_content_binding_status: NanoDatasetContentBindingStatus,
}

impl NanoWarmStartReplayReceipt {
    pub fn occupancy_snapshot_path(&self) -> &Path {
        &self.occupancy_snapshot_path
    }

    pub fn slam_dataset_directory_path(&self) -> &Path {
        &self.slam_dataset_directory_path
    }

    pub const fn processed_stereo_pairs(&self) -> u64 {
        self.processed_stereo_pairs
    }

    pub const fn final_replay_map(&self) -> MapSnapshot {
        self.final_replay_map
    }

    pub const fn replay_diagnostic_events(&self) -> u64 {
        self.replay_diagnostic_events
    }

    pub const fn replay_map_corrections(&self) -> u64 {
        self.replay_map_corrections
    }

    pub const fn selected_map_epoch_id(&self) -> RecordedMapEpochId {
        self.selected_map_epoch_id
    }

    pub const fn selected_map_revision(&self) -> u64 {
        self.selected_map_revision
    }

    pub const fn dataset_content_binding_status(&self) -> NanoDatasetContentBindingStatus {
        self.dataset_content_binding_status
    }
}

/// Owners that must be continued by the live inference, dense, and navigation
/// workers. Reconstructing fresh owners after this point discards warm start.
pub struct NanoWarmStartReplayReady {
    tracker: SlamTracker,
    occupancy: OccupancyRuntime,
    dense_generation: DenseCommandGeneration,
    initial_snapshot: TimedOccupancySnapshot,
    relocalization_gate: NanoWarmStartRelocalizationGate,
    receipt: NanoWarmStartReplayReceipt,
}

impl NanoWarmStartReplayReady {
    pub fn into_parts(self) -> NanoWarmStartReplayRuntimeParts {
        NanoWarmStartReplayRuntimeParts {
            tracker: self.tracker,
            occupancy: self.occupancy,
            dense_generation: self.dense_generation,
            initial_snapshot: self.initial_snapshot,
            relocalization_gate: self.relocalization_gate,
            receipt: self.receipt,
        }
    }
}

pub struct NanoWarmStartReplayRuntimeParts {
    pub tracker: SlamTracker,
    pub occupancy: OccupancyRuntime,
    pub dense_generation: DenseCommandGeneration,
    pub initial_snapshot: TimedOccupancySnapshot,
    pub relocalization_gate: NanoWarmStartRelocalizationGate,
    pub receipt: NanoWarmStartReplayReceipt,
}

/// Replay the exact path retained by map admission through the supplied live
/// tracker and occupancy configuration.
///
/// `tracker` must already be constructed from the live camera calibration and
/// inference assets. The replay dataset calibration is compared bit-for-bit
/// with that tracker before any frame is processed.
pub fn replay_nano_warm_start(
    mut required: NanoDatasetReplayRequired,
    mut tracker: SlamTracker,
    occupancy_config: OccupancyRuntimeConfig,
    config: NanoWarmStartReplayConfig,
) -> Result<NanoWarmStartReplayReady, NanoWarmStartReplayError> {
    let occupancy_snapshot_path = required.occupancy_snapshot_path().to_path_buf();
    let slam_dataset_directory_path = required.slam_dataset_directory_path().to_path_buf();
    let dataset_content_binding_status = required.dataset_content_binding_status();

    let mut selected_manifest = required
        .selected_manifest_reader()
        .map_err(NanoWarmStartReplayError::Selection)?;
    let mut reader = DatasetReader::open_with_manifest_reader(
        &slam_dataset_directory_path,
        &mut selected_manifest,
    )
    .map_err(NanoWarmStartReplayError::Dataset)?;
    selected_manifest
        .verify()
        .map_err(NanoWarmStartReplayError::Selection)?;
    let final_dataset_map = derive_final_dataset_map_identity(&reader)?;
    required
        .verify_selected_dataset_map_identity(
            final_dataset_map.map_epoch_id,
            final_dataset_map.map_revision,
        )
        .map_err(NanoWarmStartReplayError::Selection)?;
    if !tracker.exactly_matches_dataset_calibration(reader.stereo_calibration()) {
        return Err(NanoWarmStartReplayError::TrackerDatasetCalibrationMismatch);
    }
    validate_depth_projection(
        reader.depth_projection_contract(),
        reader.stereo_calibration(),
        occupancy_config.mapper().camera(),
    )?;
    let mut depth_cursor = reader
        .depth_cursor()
        .map_err(NanoWarmStartReplayError::Dataset)?;
    if depth_cursor.is_empty() {
        return Err(NanoWarmStartReplayError::EmptyDepthStream);
    }

    let mut occupancy =
        OccupancyRuntime::try_new(occupancy_config).map_err(NanoWarmStartReplayError::Occupancy)?;
    let mut depth_selector = ReplayDepthSelector::default();
    let mut depth_ring = DepthRingBuffer::try_new(config.depth_ring_capacity())
        .map_err(NanoWarmStartReplayError::DepthRing)?;
    let mut dense_generation = DenseCommandGeneration::default();
    let mut processed_stereo_pairs = 0_u64;
    let mut replay_diagnostic_events = 0_u64;
    let mut replay_map_corrections = 0_u64;
    let mut final_timestamp = None;
    let mut last_buffered_depth = None;

    for pair in reader.pairs() {
        let pair = pair.map_err(NanoWarmStartReplayError::Dataset)?;
        let timestamp = pair.left().timestamp();
        let selected_depth = depth_selector
            .select(timestamp, &mut depth_cursor)
            .map_err(NanoWarmStartReplayError::Dataset)?;
        if let Some(depth) = selected_depth
            && last_buffered_depth != Some(depth.frame_id())
        {
            last_buffered_depth = Some(depth.frame_id());
            depth_ring.push(depth);
        }

        let replay_pair = namespace_replay_pair(pair)?;
        let output = tracker
            .process(replay_pair)
            .map_err(NanoWarmStartReplayError::Tracker)?;
        replay_diagnostic_events =
            checked_replay_count(replay_diagnostic_events, output.events().len(), || {
                NanoWarmStartReplayError::ReplayDiagnosticEventCountExhausted
            })?;
        replay_map_corrections = checked_replay_count(
            replay_map_corrections,
            output.map_corrections().len(),
            || NanoWarmStartReplayError::ReplayMapCorrectionCountExhausted,
        )?;
        let pose_updates = tracker.take_pending_dense_pose_updates();
        let commands = map_output_to_dense_commands(
            &output,
            pose_updates,
            |keyframe_id| tracker.keyframe_pose(keyframe_id),
            &depth_ring,
            timestamp,
            &mut dense_generation,
        )
        .map_err(NanoWarmStartReplayError::DenseCommand)?;
        for command in commands {
            occupancy
                .process(command, false)
                .map_err(NanoWarmStartReplayError::OccupancyRuntime)?;
        }

        processed_stereo_pairs = processed_stereo_pairs
            .checked_add(1)
            .ok_or(NanoWarmStartReplayError::ProcessedStereoPairCountExhausted)?;
        final_timestamp = Some(timestamp);
    }

    let final_timestamp = final_timestamp.ok_or(NanoWarmStartReplayError::EmptyStereoStream)?;
    let quiesced = tracker
        .quiesce_dataset_replay(config.quiescence_timeout())
        .map_err(NanoWarmStartReplayError::TrackerQuiescence)?;
    let (final_replay_map, final_pose_updates, final_events, final_map_corrections) =
        quiesced.into_parts();
    replay_diagnostic_events =
        checked_replay_count(replay_diagnostic_events, final_events.len(), || {
            NanoWarmStartReplayError::ReplayDiagnosticEventCountExhausted
        })?;
    replay_map_corrections =
        checked_replay_count(replay_map_corrections, final_map_corrections.len(), || {
            NanoWarmStartReplayError::ReplayMapCorrectionCountExhausted
        })?;
    if let Some(command) =
        apply_pose_updates_command(final_pose_updates, final_timestamp, &mut dense_generation)
            .map_err(NanoWarmStartReplayError::DenseGeneration)?
    {
        occupancy
            .process(command, false)
            .map_err(NanoWarmStartReplayError::OccupancyRuntime)?;
    }

    let replay_occupancy = occupancy
        .mapper()
        .snapshot()
        .map_err(NanoWarmStartReplayError::Occupancy)?;
    let replay_evidence = ReplayOccupancyEvidence::try_new(final_replay_map, replay_occupancy)
        .map_err(NanoWarmStartReplayError::ReplayEvidence)?;
    let matched = required
        .verify_exact_replay(replay_evidence)
        .map_err(NanoWarmStartReplayError::ReplayBind)?;
    let matched_map = matched.replay_matched_map().sparse_map_snapshot();
    debug_assert_eq!(matched_map, final_replay_map);

    tracker
        .arm_live_relocalization_after_replay(matched_map)
        .map_err(NanoWarmStartReplayError::RelocalizationArm)?;
    let initial_snapshot =
        occupancy.continue_from_replay_match(final_timestamp, matched.into_replay_matched_map());
    let relocalization_gate = NanoWarmStartRelocalizationGate {
        replay_map: matched_map,
    };
    let receipt = NanoWarmStartReplayReceipt {
        occupancy_snapshot_path,
        slam_dataset_directory_path,
        processed_stereo_pairs,
        final_replay_map,
        replay_diagnostic_events,
        replay_map_corrections,
        selected_map_epoch_id: final_dataset_map.map_epoch_id,
        selected_map_revision: final_dataset_map.map_revision,
        dataset_content_binding_status,
    };

    Ok(NanoWarmStartReplayReady {
        tracker,
        occupancy,
        dense_generation,
        initial_snapshot,
        relocalization_gate,
        receipt,
    })
}

#[derive(Debug)]
pub enum NanoWarmStartReplayError {
    Dataset(DatasetError),
    NavigationIngressRecordCountOutOfRange {
        declared: u64,
    },
    NavigationIngressCapacity(NavigationIngressCapacityError),
    NavigationIngress(NavigationIngressStreamReadError),
    MissingFinalDatasetMapIdentity,
    Selection(NanoWarmSelectionError),
    EmptyStereoStream,
    EmptyDepthStream,
    TrackerDatasetCalibrationMismatch,
    MissingDepthProjectionContract,
    LegacyDepthOpticalFrame,
    UnsupportedDepthOpticalFrame {
        actual: DepthOpticalFrame,
    },
    DepthProjectionDimensionsMismatch {
        dataset: FrameDimensions,
        occupancy: FrameDimensions,
    },
    DepthProjectionIntrinsicsMismatch,
    DepthToTrackingCameraIsNotIdentity,
    ReplayFrameIdOutsideHistoricalNamespace {
        frame_id: FrameId,
    },
    ProcessedStereoPairCountExhausted,
    ReplayDiagnosticEventCountExhausted,
    ReplayMapCorrectionCountExhausted,
    DepthRing(DepthRingBufferError),
    Tracker(TrackerError),
    TrackerQuiescence(DatasetReplayTrackerQuiesceError),
    DenseCommand(DenseCommandMappingError),
    DenseGeneration(DenseCommandGenerationError),
    Occupancy(OccupancyError),
    OccupancyRuntime(OccupancyRuntimeError),
    ReplayEvidence(ReplayOccupancyEvidenceError),
    ReplayBind(NanoWarmStartReplayBindError),
    RelocalizationArm(DatasetReplayRelocalizationArmError),
}

impl fmt::Display for NanoWarmStartReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dataset(source) => write!(formatter, "warm-start dataset replay failed: {source}"),
            Self::NavigationIngressRecordCountOutOfRange { declared } => write!(
                formatter,
                "warm-start navigation journal declares {declared} records, which does not fit the host index domain",
            ),
            Self::NavigationIngressCapacity(source) => write!(
                formatter,
                "warm-start navigation journal has an invalid bounded capacity: {source}",
            ),
            Self::NavigationIngress(source) => write!(
                formatter,
                "warm-start navigation journal identity traversal failed: {source}",
            ),
            Self::MissingFinalDatasetMapIdentity => formatter.write_str(
                "warm-start navigation journal has no accepted global map in its final map epoch",
            ),
            Self::Selection(source) => write!(
                formatter,
                "warm-start selected map identity disagrees with its finalized navigation journal: {source}",
            ),
            Self::EmptyStereoStream => {
                formatter.write_str("warm-start dataset contains no replayable stereo pair")
            }
            Self::EmptyDepthStream => {
                formatter.write_str("warm-start dataset contains no indexed depth frame")
            }
            Self::TrackerDatasetCalibrationMismatch => formatter.write_str(
                "warm-start dataset stereo calibration differs bit-for-bit from the live tracker calibration",
            ),
            Self::MissingDepthProjectionContract => formatter.write_str(
                "warm-start dataset depth stream has no parsed projection contract",
            ),
            Self::LegacyDepthOpticalFrame => formatter.write_str(
                "warm-start dataset depth metadata does not identify its optical frame",
            ),
            Self::UnsupportedDepthOpticalFrame { actual } => write!(
                formatter,
                "warm-start dataset depth uses {actual:?}; occupancy replay requires rectified_left",
            ),
            Self::DepthProjectionDimensionsMismatch { dataset, occupancy } => write!(
                formatter,
                "warm-start dataset depth dimensions {dataset:?} differ from occupancy projection dimensions {occupancy:?}",
            ),
            Self::DepthProjectionIntrinsicsMismatch => formatter.write_str(
                "warm-start occupancy depth intrinsics differ bit-for-bit from the dataset rectified-left calibration",
            ),
            Self::DepthToTrackingCameraIsNotIdentity => formatter.write_str(
                "warm-start rectified-left depth replay requires an exact identity depth-to-tracking-camera transform",
            ),
            Self::ReplayFrameIdOutsideHistoricalNamespace { frame_id } => write!(
                formatter,
                "warm-start dataset frame ID {} already uses the reserved historical namespace bit",
                frame_id.as_u64()
            ),
            Self::ProcessedStereoPairCountExhausted => {
                formatter.write_str("warm-start processed stereo-pair counter exhausted")
            }
            Self::ReplayDiagnosticEventCountExhausted => {
                formatter.write_str("warm-start replay diagnostic-event counter exhausted")
            }
            Self::ReplayMapCorrectionCountExhausted => {
                formatter.write_str("warm-start replay map-correction counter exhausted")
            }
            Self::DepthRing(source) => source.fmt(formatter),
            Self::Tracker(source) => write!(formatter, "warm-start tracker replay failed: {source}"),
            Self::TrackerQuiescence(source) => source.fmt(formatter),
            Self::DenseCommand(source) => {
                write!(formatter, "warm-start dense command mapping failed: {source}")
            }
            Self::DenseGeneration(source) => source.fmt(formatter),
            Self::Occupancy(source) => {
                write!(formatter, "warm-start occupancy construction failed: {source}")
            }
            Self::OccupancyRuntime(source) => source.fmt(formatter),
            Self::ReplayEvidence(source) => source.fmt(formatter),
            Self::ReplayBind(source) => source.fmt(formatter),
            Self::RelocalizationArm(source) => source.fmt(formatter),
        }
    }
}

impl std::error::Error for NanoWarmStartReplayError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Dataset(source) => Some(source),
            Self::NavigationIngressCapacity(source) => Some(source),
            Self::NavigationIngress(source) => Some(source),
            Self::Selection(source) => Some(source),
            Self::DepthRing(source) => Some(source),
            Self::Tracker(source) => Some(source),
            Self::TrackerQuiescence(source) => Some(source),
            Self::DenseCommand(source) => Some(source),
            Self::DenseGeneration(source) => Some(source),
            Self::Occupancy(source) => Some(source),
            Self::OccupancyRuntime(source) => Some(source),
            Self::ReplayEvidence(source) => Some(source),
            Self::ReplayBind(source) => Some(source),
            Self::RelocalizationArm(source) => Some(source),
            Self::NavigationIngressRecordCountOutOfRange { .. }
            | Self::MissingFinalDatasetMapIdentity
            | Self::EmptyStereoStream
            | Self::EmptyDepthStream
            | Self::TrackerDatasetCalibrationMismatch
            | Self::MissingDepthProjectionContract
            | Self::LegacyDepthOpticalFrame
            | Self::UnsupportedDepthOpticalFrame { .. }
            | Self::DepthProjectionDimensionsMismatch { .. }
            | Self::DepthProjectionIntrinsicsMismatch
            | Self::DepthToTrackingCameraIsNotIdentity
            | Self::ReplayFrameIdOutsideHistoricalNamespace { .. }
            | Self::ProcessedStereoPairCountExhausted
            | Self::ReplayDiagnosticEventCountExhausted
            | Self::ReplayMapCorrectionCountExhausted => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FinalDatasetMapIdentity {
    map_epoch_id: RecordedMapEpochId,
    map_revision: u64,
}

/// Derive the final map identity from the manifest-bound, fixed-record
/// navigation journal without retaining the journal in memory.
///
/// A later `MapEpochStarted` invalidates an older accepted map: the selected
/// restart point must name a map accepted in the journal's final epoch.
fn derive_final_dataset_map_identity(
    dataset: &DatasetReader,
) -> Result<FinalDatasetMapIdentity, NanoWarmStartReplayError> {
    let descriptor = dataset
        .navigation_ingress_descriptor()
        .map_err(NanoWarmStartReplayError::Dataset)?;
    let declared = usize::try_from(descriptor.record_count()).map_err(|_| {
        NanoWarmStartReplayError::NavigationIngressRecordCountOutOfRange {
            declared: descriptor.record_count(),
        }
    })?;
    let capacity = NavigationIngressCapacity::try_new(declared.max(1))
        .map_err(NanoWarmStartReplayError::NavigationIngressCapacity)?;
    let mut journal = dataset
        .navigation_ingress_reader(capacity)
        .map_err(NanoWarmStartReplayError::Dataset)?;
    derive_final_map_identity_from_journal(&mut journal)
}

fn derive_final_map_identity_from_journal<R: Read>(
    journal: &mut NavigationIngressReader<R>,
) -> Result<FinalDatasetMapIdentity, NanoWarmStartReplayError> {
    let mut final_map = None;
    while let Some(record) = journal
        .next_record()
        .map_err(NanoWarmStartReplayError::NavigationIngress)?
    {
        match record.event() {
            NavigationIngressEvent::MapEpochStarted(_) => final_map = None,
            NavigationIngressEvent::AcceptedGlobalMap(map) => {
                final_map = Some(FinalDatasetMapIdentity {
                    map_epoch_id: map.map_epoch_id(),
                    map_revision: map.revision(),
                });
            }
            NavigationIngressEvent::VisualAttempt(_)
            | NavigationIngressEvent::ImuReport(_)
            | NavigationIngressEvent::AcceptedDepth(_)
            | NavigationIngressEvent::PointGoal(_)
            | NavigationIngressEvent::ControlTick(_)
            | NavigationIngressEvent::QualificationAppliedStep(_) => {}
        }
    }
    final_map.ok_or(NanoWarmStartReplayError::MissingFinalDatasetMapIdentity)
}

fn validate_depth_projection(
    projection: Option<DepthProjectionContract>,
    calibration: &StereoCalibration,
    occupancy: DepthCameraModel,
) -> Result<(), NanoWarmStartReplayError> {
    let projection = projection.ok_or(NanoWarmStartReplayError::MissingDepthProjectionContract)?;
    match projection.optical_frame() {
        Some(DepthOpticalFrame::RectifiedLeft) => {}
        None => return Err(NanoWarmStartReplayError::LegacyDepthOpticalFrame),
        Some(actual) => {
            return Err(NanoWarmStartReplayError::UnsupportedDepthOpticalFrame { actual });
        }
    }
    if projection.dimensions() != occupancy.dimensions() {
        return Err(
            NanoWarmStartReplayError::DepthProjectionDimensionsMismatch {
                dataset: projection.dimensions(),
                occupancy: occupancy.dimensions(),
            },
        );
    }
    if !intrinsics_match_exactly(calibration.left(), occupancy.intrinsics()) {
        return Err(NanoWarmStartReplayError::DepthProjectionIntrinsicsMismatch);
    }
    if !pose_matches_exactly(occupancy.depth_to_tracking().pose(), Pose::identity()) {
        return Err(NanoWarmStartReplayError::DepthToTrackingCameraIsNotIdentity);
    }
    Ok(())
}

fn intrinsics_match_exactly(left: PinholeIntrinsics, right: PinholeIntrinsics) -> bool {
    left.fx().to_bits() == right.fx().to_bits()
        && left.fy().to_bits() == right.fy().to_bits()
        && left.cx().to_bits() == right.cx().to_bits()
        && left.cy().to_bits() == right.cy().to_bits()
}

fn pose_matches_exactly(left: Pose, right: Pose) -> bool {
    left.rotation()
        .into_iter()
        .flatten()
        .zip(right.rotation().into_iter().flatten())
        .all(|(left, right)| left.to_bits() == right.to_bits())
        && left
            .translation()
            .into_iter()
            .zip(right.translation())
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn checked_replay_count(
    current: u64,
    additional: usize,
    exhausted: impl Fn() -> NanoWarmStartReplayError,
) -> Result<u64, NanoWarmStartReplayError> {
    let additional = u64::try_from(additional).map_err(|_| exhausted())?;
    current.checked_add(additional).ok_or_else(exhausted)
}

fn namespace_replay_pair(pair: StereoPair) -> Result<StereoPair, NanoWarmStartReplayError> {
    let (left, right) = pair.into_parts();
    Ok(StereoPair::from_parts(
        namespace_replay_frame(left)?,
        namespace_replay_frame(right)?,
    ))
}

fn namespace_replay_frame(frame: Frame) -> Result<Frame, NanoWarmStartReplayError> {
    let original = frame.frame_id();
    if original.as_u64() & REPLAY_FRAME_NAMESPACE_BIT != 0 {
        return Err(
            NanoWarmStartReplayError::ReplayFrameIdOutsideHistoricalNamespace {
                frame_id: original,
            },
        );
    }
    Ok(frame.with_frame_id(FrameId::new(original.as_u64() | REPLAY_FRAME_NAMESPACE_BIT)))
}

#[derive(Debug, Default)]
struct ReplayDepthSelector {
    previous: Option<DepthImage>,
    lookahead: Option<DepthImage>,
}

impl ReplayDepthSelector {
    fn select(
        &mut self,
        timestamp: Timestamp,
        cursor: &mut DatasetDepthCursor,
    ) -> Result<Option<DepthImage>, DatasetError> {
        if self
            .lookahead
            .as_ref()
            .is_some_and(|depth| depth.timestamp() <= timestamp)
        {
            self.previous = self.lookahead.take();
        }

        let cutoff_delta = i64::try_from(DEPTH_ASSOCIATION_WINDOW.as_nanos())
            .expect("the fixed 20 ms association window fits i64");
        let cutoff = Timestamp::from_nanos(
            timestamp
                .as_nanos()
                .checked_add(cutoff_delta)
                .unwrap_or(i64::MAX),
        );
        while self.lookahead.is_none() {
            let Some(depth) = cursor.next_at_or_before(cutoff)? else {
                break;
            };
            if depth.timestamp() <= timestamp {
                self.previous = Some(depth);
            } else {
                self.lookahead = Some(depth);
            }
        }

        let maximum_delta = DEPTH_ASSOCIATION_WINDOW.as_nanos();
        let candidate = match (&self.previous, &self.lookahead) {
            (Some(previous), Some(lookahead)) => {
                let previous_delta = previous
                    .timestamp()
                    .as_nanos()
                    .abs_diff(timestamp.as_nanos());
                let lookahead_delta = lookahead
                    .timestamp()
                    .as_nanos()
                    .abs_diff(timestamp.as_nanos());
                if previous_delta <= lookahead_delta {
                    Some((previous, previous_delta))
                } else {
                    Some((lookahead, lookahead_delta))
                }
            }
            (Some(previous), None) => Some((
                previous,
                previous
                    .timestamp()
                    .as_nanos()
                    .abs_diff(timestamp.as_nanos()),
            )),
            (None, Some(lookahead)) => Some((
                lookahead,
                lookahead
                    .timestamp()
                    .as_nanos()
                    .abs_diff(timestamp.as_nanos()),
            )),
            (None, None) => None,
        };
        Ok(candidate
            .filter(|(_, delta)| *delta <= maximum_delta)
            .map(|(depth, _)| depth.clone()))
    }
}

/// Linear gate proving that a fresh live frame passed the tracker's real
/// multi-frame relocalization verifier against the replayed sparse map.
#[derive(Debug)]
pub struct NanoWarmStartRelocalizationGate {
    replay_map: MapSnapshot,
}

impl NanoWarmStartRelocalizationGate {
    pub const fn replay_map(&self) -> MapSnapshot {
        self.replay_map
    }

    pub fn observe(
        self,
        output: &TrackerOutput,
    ) -> Result<NanoWarmStartRelocalizationTransition, NanoWarmStartRelocalizationError> {
        self.observe_parts(
            output.events(),
            output.current_map_localization(),
            output.health().tracking,
        )
    }

    fn observe_parts(
        self,
        events: &[DiagnosticEvent],
        localization: Option<MapLocalization>,
        tracking: TrackingHealth,
    ) -> Result<NanoWarmStartRelocalizationTransition, NanoWarmStartRelocalizationError> {
        if let Some(actual) = events.iter().find_map(|event| match event {
            DiagnosticEvent::MappingSessionReset { transition } => Some(transition.new_map()),
            _ => None,
        }) {
            return Err(
                NanoWarmStartRelocalizationError::ReplayMapReplacedBeforeLocalization {
                    expected: self.replay_map,
                    actual,
                },
            );
        }

        let mut successes = events.iter().filter_map(|event| match event {
            DiagnosticEvent::RelocalizationSucceeded { keyframe_id } => Some(*keyframe_id),
            _ => None,
        });
        let Some(candidate) = successes.next() else {
            return Ok(NanoWarmStartRelocalizationTransition::Awaiting(self));
        };
        if successes.next().is_some() {
            return Err(NanoWarmStartRelocalizationError::MultipleSuccessEvents);
        }
        if candidate.map_instance_id() != self.replay_map.instance_id() {
            return Err(
                NanoWarmStartRelocalizationError::CandidateMapInstanceMismatch {
                    expected: self.replay_map,
                    candidate,
                },
            );
        }
        let localization = localization
            .ok_or(NanoWarmStartRelocalizationError::SuccessWithoutCurrentLocalization)?;
        if localization.map_snapshot().instance_id() != self.replay_map.instance_id() {
            return Err(
                NanoWarmStartRelocalizationError::LocalizationMapInstanceMismatch {
                    expected: self.replay_map,
                    actual: localization.map_snapshot(),
                },
            );
        }
        if !localization
            .map_snapshot()
            .shares_mutation_lineage_with(self.replay_map)
        {
            return Err(
                NanoWarmStartRelocalizationError::LocalizationMapLineageMismatch {
                    expected: self.replay_map,
                    actual: localization.map_snapshot(),
                },
            );
        }
        if localization.map_snapshot().generation().as_u64()
            <= self.replay_map.generation().as_u64()
        {
            return Err(
                NanoWarmStartRelocalizationError::LocalizationMapDidNotAdvance {
                    replay: self.replay_map,
                    actual: localization.map_snapshot(),
                },
            );
        }
        if tracking != TrackingHealth::Good {
            return Err(
                NanoWarmStartRelocalizationError::SuccessWithoutGoodTracking { actual: tracking },
            );
        }

        Ok(NanoWarmStartRelocalizationTransition::Localized(
            NanoWarmStartLocalizedEvidence {
                replay_map: self.replay_map,
                candidate,
                localization,
            },
        ))
    }
}

#[derive(Debug)]
pub enum NanoWarmStartRelocalizationTransition {
    Awaiting(NanoWarmStartRelocalizationGate),
    Localized(NanoWarmStartLocalizedEvidence),
}

#[derive(Clone, Copy, Debug)]
pub struct NanoWarmStartLocalizedEvidence {
    replay_map: MapSnapshot,
    candidate: KeyframeId,
    localization: MapLocalization,
}

impl NanoWarmStartLocalizedEvidence {
    pub const fn replay_map(self) -> MapSnapshot {
        self.replay_map
    }

    pub const fn candidate(self) -> KeyframeId {
        self.candidate
    }

    pub const fn localization(self) -> MapLocalization {
        self.localization
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoWarmStartRelocalizationError {
    ReplayMapReplacedBeforeLocalization {
        expected: MapSnapshot,
        actual: crate::MapInstanceId,
    },
    MultipleSuccessEvents,
    CandidateMapInstanceMismatch {
        expected: MapSnapshot,
        candidate: KeyframeId,
    },
    SuccessWithoutCurrentLocalization,
    LocalizationMapInstanceMismatch {
        expected: MapSnapshot,
        actual: MapSnapshot,
    },
    LocalizationMapLineageMismatch {
        expected: MapSnapshot,
        actual: MapSnapshot,
    },
    LocalizationMapDidNotAdvance {
        replay: MapSnapshot,
        actual: MapSnapshot,
    },
    SuccessWithoutGoodTracking {
        actual: TrackingHealth,
    },
}

impl fmt::Display for NanoWarmStartRelocalizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "warm-start live relocalization evidence is invalid: {self:?}"
        )
    }
}

impl std::error::Error for NanoWarmStartRelocalizationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    use super::super::ingress::AcceptedGlobalMapIngress;
    use crate::HostMonotonicTimestamp;
    use crate::dense::occupancy::{
        DepthToTrackingCamera, OccupancyCell, OccupancyGridGeometry, OccupancyGridSnapshot,
    };
    use crate::map::{ImageSize, SlamMap};
    use crate::navigation::{
        NavigationClockEpoch, NavigationIngressLog, NavigationMapEpochCoordinator,
        NavigationRecordingId,
    };
    use crate::{Keypoint, SensorId, WorldToCamera};

    fn replay_map_with_live_attachment() -> (MapSnapshot, MapSnapshot, KeyframeId) {
        let mut map = SlamMap::new();
        let candidate = map
            .add_keyframe(
                FrameId::new(REPLAY_FRAME_NAMESPACE_BIT),
                Timestamp::from_nanos(1),
                WorldToCamera::identity(),
                ImageSize::try_new(1, 1).expect("image size"),
                vec![Keypoint { x: 0.0, y: 0.0 }],
            )
            .expect("test keyframe");
        let replay_map = map.snapshot();
        map.add_keyframe(
            FrameId::new(1),
            Timestamp::from_nanos(2),
            WorldToCamera::identity(),
            ImageSize::try_new(1, 1).expect("image size"),
            vec![Keypoint { x: 0.0, y: 0.0 }],
        )
        .expect("live relocalization attachment");
        (replay_map, map.snapshot(), candidate)
    }

    fn localization(map: MapSnapshot, frame_id: u64) -> MapLocalization {
        MapLocalization::new(
            crate::VisualFrameStamp::new(
                FrameId::new(frame_id),
                Timestamp::from_nanos(i64::try_from(frame_id).expect("test frame ID fits i64")),
            ),
            map,
            WorldToCamera::identity(),
        )
    }

    fn depth_projection_fixture() -> (DepthProjectionContract, StereoCalibration, DepthCameraModel)
    {
        let dimensions = FrameDimensions::try_new(640, 480).expect("dimensions");
        let intrinsics =
            PinholeIntrinsics::try_new(400.0, 401.0, 320.0, 240.0).expect("intrinsics");
        let calibration =
            StereoCalibration::try_new(intrinsics, intrinsics, dimensions, 0.075, true)
                .expect("calibration");
        let projection = DepthProjectionContract::new(dimensions, DepthOpticalFrame::RectifiedLeft);
        let camera =
            DepthCameraModel::new(intrinsics, dimensions, DepthToTrackingCamera::identity());
        (projection, calibration, camera)
    }

    fn occupancy_snapshot(map: &SlamMap, revision: u64) -> OccupancyGridSnapshot {
        let geometry =
            OccupancyGridGeometry::try_new(0.1, [-0.1, -0.1], 1, 1, 1).expect("geometry");
        OccupancyGridSnapshot::from_test_cells(
            geometry,
            &[OccupancyCell::Free],
            map.snapshot().instance_id(),
            revision,
        )
    }

    fn append_map_epoch(
        log: &mut NavigationIngressLog,
        coordinator: &mut NavigationMapEpochCoordinator,
        clock: NavigationClockEpoch,
        map: &SlamMap,
        revision: u64,
        time_ns: u64,
    ) {
        let transition = coordinator
            .start_epoch(
                clock,
                HostMonotonicTimestamp::from_nanos(time_ns),
                map.snapshot().instance_id(),
            )
            .expect("map epoch");
        log.push(NavigationIngressEvent::MapEpochStarted(transition.event()))
            .expect("record map epoch");
        let snapshot = occupancy_snapshot(map, revision);
        let accepted = AcceptedGlobalMapIngress::parse_snapshot(
            clock,
            HostMonotonicTimestamp::from_nanos(time_ns + 1),
            transition.binding(),
            Timestamp::from_nanos(i64::try_from(time_ns).expect("test timestamp")),
            &snapshot,
        )
        .expect("accepted map");
        log.push(NavigationIngressEvent::AcceptedGlobalMap(accepted))
            .expect("record accepted map");
    }

    #[test]
    fn final_dataset_map_identity_comes_from_the_last_journal_epoch() {
        let recording_id = NavigationRecordingId::try_new([7; 16]).expect("recording ID");
        let capacity = NavigationIngressCapacity::try_new(4).expect("capacity");
        let clock = NavigationClockEpoch::new(HostMonotonicTimestamp::from_nanos(0));
        let mut coordinator = NavigationMapEpochCoordinator::new();
        let mut log = NavigationIngressLog::new(recording_id, capacity);
        append_map_epoch(&mut log, &mut coordinator, clock, &SlamMap::new(), 3, 10);
        append_map_epoch(&mut log, &mut coordinator, clock, &SlamMap::new(), 9, 20);
        let bytes = log.encode().expect("encoded journal");
        let mut reader = NavigationIngressReader::new(Cursor::new(bytes), recording_id, capacity)
            .expect("bounded journal");

        let identity =
            derive_final_map_identity_from_journal(&mut reader).expect("final map identity");
        assert_eq!(identity.map_epoch_id.as_u64(), 2);
        assert_eq!(identity.map_revision, 9);
    }

    #[test]
    fn final_epoch_without_an_accepted_map_has_no_restart_identity() {
        let recording_id = NavigationRecordingId::try_new([8; 16]).expect("recording ID");
        let capacity = NavigationIngressCapacity::try_new(3).expect("capacity");
        let clock = NavigationClockEpoch::new(HostMonotonicTimestamp::from_nanos(0));
        let first_map = SlamMap::new();
        let second_map = SlamMap::new();
        let mut coordinator = NavigationMapEpochCoordinator::new();
        let mut log = NavigationIngressLog::new(recording_id, capacity);
        append_map_epoch(&mut log, &mut coordinator, clock, &first_map, 3, 10);
        let transition = coordinator
            .start_epoch(
                clock,
                HostMonotonicTimestamp::from_nanos(20),
                second_map.snapshot().instance_id(),
            )
            .expect("second epoch");
        log.push(NavigationIngressEvent::MapEpochStarted(transition.event()))
            .expect("record second epoch");
        let bytes = log.encode().expect("encoded journal");
        let mut reader = NavigationIngressReader::new(Cursor::new(bytes), recording_id, capacity)
            .expect("bounded journal");

        assert!(matches!(
            derive_final_map_identity_from_journal(&mut reader),
            Err(NanoWarmStartReplayError::MissingFinalDatasetMapIdentity)
        ));
    }

    #[test]
    fn replay_depth_projection_requires_exact_rectified_left_geometry() {
        let (projection, calibration, camera) = depth_projection_fixture();
        validate_depth_projection(Some(projection), &calibration, camera)
            .expect("exact projection");

        assert!(matches!(
            validate_depth_projection(None, &calibration, camera),
            Err(NanoWarmStartReplayError::MissingDepthProjectionContract)
        ));
        let rgb = DepthProjectionContract::new(projection.dimensions(), DepthOpticalFrame::Rgb);
        assert!(matches!(
            validate_depth_projection(Some(rgb), &calibration, camera),
            Err(NanoWarmStartReplayError::UnsupportedDepthOpticalFrame {
                actual: DepthOpticalFrame::Rgb
            })
        ));

        let translated =
            Pose::try_from_rt(Pose::identity().rotation(), [f32::from_bits(1), 0.0, 0.0])
                .expect("small rigid translation");
        let translated_camera = DepthCameraModel::new(
            camera.intrinsics(),
            camera.dimensions(),
            DepthToTrackingCamera::new(translated),
        );
        assert!(matches!(
            validate_depth_projection(Some(projection), &calibration, translated_camera),
            Err(NanoWarmStartReplayError::DepthToTrackingCameraIsNotIdentity)
        ));
    }

    #[test]
    fn replay_config_is_bounded_and_nonzero() {
        assert!(matches!(
            NanoWarmStartReplayConfig::try_new(0, Duration::from_secs(1)),
            Err(NanoWarmStartReplayConfigError::ZeroDepthRingCapacity)
        ));
        assert!(matches!(
            NanoWarmStartReplayConfig::try_new(1, Duration::ZERO),
            Err(NanoWarmStartReplayConfigError::QuiescenceTimeoutOutOfRange { .. })
        ));
        assert!(NanoWarmStartReplayConfig::default().depth_ring_capacity() > 0);
    }

    #[test]
    fn replay_namespace_is_zero_copy_and_disjoint_from_oak_sequence_domain() {
        let pixels = vec![7_u8; 4];
        let frame = Frame::new(
            SensorId::StereoLeft,
            FrameId::new(17),
            Timestamp::from_nanos(9),
            2,
            2,
            pixels,
        )
        .expect("frame");
        let original_pixels = frame.data().as_ptr();
        let replay = namespace_replay_frame(frame).expect("historical namespace");
        assert_eq!(replay.frame_id().as_u64(), REPLAY_FRAME_NAMESPACE_BIT | 17);
        assert_eq!(replay.data().as_ptr(), original_pixels);

        let already_historical = Frame::new(
            SensorId::StereoLeft,
            FrameId::new(REPLAY_FRAME_NAMESPACE_BIT),
            Timestamp::from_nanos(9),
            1,
            1,
            vec![0],
        )
        .expect("frame");
        assert!(matches!(
            namespace_replay_frame(already_historical),
            Err(NanoWarmStartReplayError::ReplayFrameIdOutsideHistoricalNamespace { .. })
        ));
    }

    #[test]
    fn gate_requires_exact_success_event_current_pose_and_replay_map() {
        let (replay_map, localized_map, candidate) = replay_map_with_live_attachment();
        let gate = NanoWarmStartRelocalizationGate { replay_map };
        let awaiting = gate
            .observe_parts(&[], Some(localization(replay_map, 4)), TrackingHealth::Good)
            .expect("ordinary tracking remains nonterminal");
        let NanoWarmStartRelocalizationTransition::Awaiting(gate) = awaiting else {
            panic!("ordinary tracking must not prove relocalization");
        };

        let transition = gate
            .observe_parts(
                &[DiagnosticEvent::RelocalizationSucceeded {
                    keyframe_id: candidate,
                }],
                Some(localization(localized_map, 5)),
                TrackingHealth::Good,
            )
            .expect("verified relocalization");
        let NanoWarmStartRelocalizationTransition::Localized(evidence) = transition else {
            panic!("success event must terminate gate");
        };
        assert_eq!(
            evidence.localization().map_snapshot().instance_id(),
            replay_map.instance_id()
        );
    }

    #[test]
    fn gate_fails_when_tracker_replaces_replay_map() {
        let (replay_map, _, _) = replay_map_with_live_attachment();
        let replacement = SlamMap::new().snapshot();
        let transition = crate::MappingSessionTransition::try_new(
            replay_map.instance_id(),
            replacement.instance_id(),
        )
        .expect("distinct maps");
        assert!(matches!(
            (NanoWarmStartRelocalizationGate { replay_map }).observe_parts(
                &[DiagnosticEvent::MappingSessionReset { transition }],
                None,
                TrackingHealth::Lost,
            ),
            Err(
                NanoWarmStartRelocalizationError::ReplayMapReplacedBeforeLocalization {
                    actual,
                    ..
                }
            ) if actual == replacement.instance_id()
        ));
    }

    #[test]
    fn gate_rejects_success_without_current_matching_localization() {
        let (replay_map, _, candidate) = replay_map_with_live_attachment();
        assert!(matches!(
            (NanoWarmStartRelocalizationGate { replay_map }).observe_parts(
                &[DiagnosticEvent::RelocalizationSucceeded {
                    keyframe_id: candidate,
                }],
                None,
                TrackingHealth::Good,
            ),
            Err(NanoWarmStartRelocalizationError::SuccessWithoutCurrentLocalization)
        ));
    }

    #[test]
    fn gate_rejects_success_without_a_live_map_mutation() {
        let (replay_map, _, candidate) = replay_map_with_live_attachment();
        assert!(matches!(
            (NanoWarmStartRelocalizationGate { replay_map }).observe_parts(
                &[DiagnosticEvent::RelocalizationSucceeded {
                    keyframe_id: candidate,
                }],
                Some(localization(replay_map, 5)),
                TrackingHealth::Good,
            ),
            Err(
                NanoWarmStartRelocalizationError::LocalizationMapDidNotAdvance {
                    replay,
                    actual,
                }
            ) if replay == replay_map && actual == replay_map
        ));
    }
}
