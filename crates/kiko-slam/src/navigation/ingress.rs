//! Versioned ordering journal for navigation coordinator ingress.
//!
//! This module records the exact admission order of already parsed navigation
//! inputs and the identities needed to find stereo and depth payloads in the
//! owning dataset. It does not serialize tracker-derived
//! [`crate::VisualIncrement`] or [`crate::MapLocalization`] values and therefore
//! does not promise deterministic SLAM or MPC replay. Host process clock values
//! and process-local map IDs never enter the wire format: live boundaries turn
//! them into recording-relative clock offsets and coordinator map epochs.
//! Legacy datasets without this sidecar remain explicitly timing-unavailable.

use std::io::{self, Read, Seek, SeekFrom, Write};
use std::num::{NonZeroU64, NonZeroUsize};

use crate::dense::occupancy::OccupancyGridSnapshot;
use crate::{
    AccelSample, DepthObservation, DequeueSequence, DeviceSessionId, DeviceTimestamp, FrameId,
    GyroSample, HostMonotonicTimestamp, ImuReport, InertialValueError, MapInstanceId,
    OakImuAcceleration, OakImuAngularVelocity, SensorAccuracy, StereoObservation,
};

use super::{GlobalPlanError, MapPoint, PointGoal};

pub const NAVIGATION_INGRESS_STREAM_FILE: &str = "navigation-ingress.v1.bin";
pub const NAVIGATION_INGRESS_FORMAT_VERSION: u16 = 1;
pub const MAX_NAVIGATION_INGRESS_RECORDS: usize = 1_048_576;

const MAGIC: [u8; 8] = *b"KIKONAV\0";
const HEADER_BYTES: usize = 48;
const RECORD_BYTES: usize = 112;
const RECORD_PAYLOAD_OFFSET: usize = 16;

const KIND_VISUAL_ATTEMPT: u8 = 1;
const KIND_IMU_REPORT: u8 = 2;
const KIND_ACCEPTED_DEPTH: u8 = 3;
const KIND_POINT_GOAL: u8 = 4;
const KIND_MAP_EPOCH_STARTED: u8 = 5;
const KIND_CONTROL_TICK: u8 = 6;

/// Opaque identity shared with the dataset manifest that owns this sidecar.
///
/// This identity prevents accidentally pairing a valid journal with another
/// recording. It does not provide integrity or origin guarantees.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NavigationRecordingId([u8; 16]);

impl NavigationRecordingId {
    pub fn try_new(bytes: [u8; 16]) -> Result<Self, NavigationRecordingIdError> {
        if bytes == [0; 16] {
            Err(NavigationRecordingIdError)
        } else {
            Ok(Self(bytes))
        }
    }

    pub fn into_bytes(self) -> [u8; 16] {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NavigationRecordingIdError;

impl std::fmt::Display for NavigationRecordingIdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("navigation recording ID must not be all zero")
    }
}

impl std::error::Error for NavigationRecordingIdError {}

/// Nanoseconds elapsed since one live navigation recording origin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NavigationClockOffset(u64);

impl NavigationClockOffset {
    pub fn as_nanos(self) -> u64 {
        self.0
    }
}

/// Live-only origin used to remove process clock epochs at the recording edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NavigationClockEpoch {
    origin: HostMonotonicTimestamp,
}

impl NavigationClockEpoch {
    pub fn new(origin: HostMonotonicTimestamp) -> Self {
        Self { origin }
    }

    pub fn origin(self) -> HostMonotonicTimestamp {
        self.origin
    }

    pub fn offset_at(
        self,
        timestamp: HostMonotonicTimestamp,
    ) -> Result<NavigationClockOffset, NavigationIngressBoundaryError> {
        timestamp
            .as_nanos()
            .checked_sub(self.origin.as_nanos())
            .map(NavigationClockOffset)
            .ok_or(NavigationIngressBoundaryError::HostTimeBeforeClockEpoch {
                origin_ns: self.origin.as_nanos(),
                timestamp_ns: timestamp.as_nanos(),
            })
    }
}

/// Explicit mapping from recorded offsets to a caller-selected replay clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NavigationReplayClock {
    virtual_origin: HostMonotonicTimestamp,
}

impl NavigationReplayClock {
    pub fn new(virtual_origin: HostMonotonicTimestamp) -> Self {
        Self { virtual_origin }
    }

    pub fn virtual_origin(self) -> HostMonotonicTimestamp {
        self.virtual_origin
    }

    pub fn resolve(
        self,
        offset: NavigationClockOffset,
    ) -> Result<HostMonotonicTimestamp, NavigationReplayClockError> {
        self.virtual_origin
            .as_nanos()
            .checked_add(offset.as_nanos())
            .map(HostMonotonicTimestamp::from_nanos)
            .ok_or(NavigationReplayClockError::TimestampOverflow {
                virtual_origin_ns: self.virtual_origin.as_nanos(),
                offset_ns: offset.as_nanos(),
            })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavigationReplayClockError {
    TimestampOverflow {
        virtual_origin_ns: u64,
        offset_ns: u64,
    },
}

impl std::fmt::Display for NavigationReplayClockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "navigation replay clock mapping failed: {self:?}")
    }
}

impl std::error::Error for NavigationReplayClockError {}

/// Caller-selected in-memory journal bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NavigationIngressCapacity(NonZeroUsize);

impl NavigationIngressCapacity {
    pub fn try_new(value: usize) -> Result<Self, NavigationIngressCapacityError> {
        let value = NonZeroUsize::new(value).ok_or(NavigationIngressCapacityError::Zero)?;
        if value.get() > MAX_NAVIGATION_INGRESS_RECORDS {
            return Err(NavigationIngressCapacityError::TooLarge {
                actual: value.get(),
                maximum: MAX_NAVIGATION_INGRESS_RECORDS,
            });
        }
        Ok(Self(value))
    }

    pub fn get(self) -> usize {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavigationIngressCapacityError {
    Zero,
    TooLarge { actual: usize, maximum: usize },
}

impl std::fmt::Display for NavigationIngressCapacityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid navigation ingress capacity: {self:?}")
    }
}

impl std::error::Error for NavigationIngressCapacityError {}

/// Contiguous, one-based order assigned by the sole coordinator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NavigationIngressSequence(NonZeroU64);

impl NavigationIngressSequence {
    pub fn as_u64(self) -> u64 {
        self.0.get()
    }
}

/// Outcome of processing exactly one stereo pair.
///
/// Every attempted pair must produce one variant. In particular, a tracker
/// failure cannot disappear and let odometry silently bridge its frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisualAttemptOutcome {
    /// A current localization and correction-safe increment were published.
    IncrementAndLocalization,
    /// A current localization was published without a preceding increment;
    /// navigation may use it only as an explicit anchor.
    LocalizationOnly,
    /// Tracking returned normally but no current localization was available.
    NoLocalization,
    /// A recoverable tracker error rejected this exact pair.
    RecoverableFailure,
    /// A pipeline-ending tracker error rejected this exact pair.
    FatalFailure,
}

impl VisualAttemptOutcome {
    fn wire_code(self) -> u8 {
        match self {
            Self::IncrementAndLocalization => 1,
            Self::LocalizationOnly => 2,
            Self::NoLocalization => 3,
            Self::RecoverableFailure => 4,
            Self::FatalFailure => 5,
        }
    }

    fn parse_wire(record_index: usize, value: u8) -> Result<Self, NavigationIngressParseError> {
        match value {
            1 => Ok(Self::IncrementAndLocalization),
            2 => Ok(Self::LocalizationOnly),
            3 => Ok(Self::NoLocalization),
            4 => Ok(Self::RecoverableFailure),
            5 => Ok(Self::FatalFailure),
            _ => Err(NavigationIngressParseError::UnknownVisualOutcome {
                record_index,
                value,
            }),
        }
    }
}

/// Stable stereo identity plus its recording-relative admission offset.
/// Image bytes remain in the dataset identified by these frame/timestamp pairs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisualAttemptIngress {
    session_id: DeviceSessionId,
    arrival_offset: NavigationClockOffset,
    left_frame_id: FrameId,
    left_timestamp: DeviceTimestamp,
    right_frame_id: FrameId,
    right_timestamp: DeviceTimestamp,
    outcome: VisualAttemptOutcome,
}

impl VisualAttemptIngress {
    pub fn from_observation(
        clock_epoch: NavigationClockEpoch,
        observation: &StereoObservation,
        outcome: VisualAttemptOutcome,
    ) -> Result<Self, NavigationIngressBoundaryError> {
        let pair = observation.pair();
        Ok(Self {
            session_id: observation.session_id(),
            arrival_offset: clock_epoch.offset_at(observation.host_arrival())?,
            left_frame_id: pair.left().frame_id(),
            left_timestamp: observation.left_device_timestamp(),
            right_frame_id: pair.right().frame_id(),
            right_timestamp: observation.right_device_timestamp(),
            outcome,
        })
    }

    pub fn session_id(self) -> DeviceSessionId {
        self.session_id
    }

    pub fn arrival_offset(self) -> NavigationClockOffset {
        self.arrival_offset
    }

    pub fn replay_host_arrival(
        self,
        clock: NavigationReplayClock,
    ) -> Result<HostMonotonicTimestamp, NavigationReplayClockError> {
        clock.resolve(self.arrival_offset)
    }

    pub fn left_frame_id(self) -> FrameId {
        self.left_frame_id
    }

    pub fn left_timestamp(self) -> DeviceTimestamp {
        self.left_timestamp
    }

    pub fn right_frame_id(self) -> FrameId {
        self.right_frame_id
    }

    pub fn right_timestamp(self) -> DeviceTimestamp {
        self.right_timestamp
    }

    pub fn outcome(self) -> VisualAttemptOutcome {
        self.outcome
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NavigationIngressBoundaryError {
    HostTimeBeforeClockEpoch {
        origin_ns: u64,
        timestamp_ns: u64,
    },
    MapEpochAlreadyCurrent {
        map_instance_id: MapInstanceId,
    },
    MapEpochIdExhausted,
    GoalMapMismatch {
        map_epoch_id: RecordedMapEpochId,
        bound_map_instance_id: MapInstanceId,
        goal_map_instance_id: MapInstanceId,
    },
}

impl std::fmt::Display for NavigationIngressBoundaryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid navigation ingress boundary: {self:?}")
    }
}

impl std::error::Error for NavigationIngressBoundaryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::HostTimeBeforeClockEpoch { .. }
            | Self::MapEpochAlreadyCurrent { .. }
            | Self::MapEpochIdExhausted
            | Self::GoalMapMismatch { .. } => None,
        }
    }
}

/// Provenance of a depth frame accepted by the navigation coordinator.
/// Metric pixels remain in the existing dataset depth payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AcceptedDepthIngress {
    session_id: DeviceSessionId,
    arrival_offset: NavigationClockOffset,
    frame_id: FrameId,
    device_timestamp: DeviceTimestamp,
}

impl AcceptedDepthIngress {
    pub fn parse(
        clock_epoch: NavigationClockEpoch,
        observation: &DepthObservation,
    ) -> Result<Self, NavigationIngressBoundaryError> {
        Ok(Self {
            session_id: observation.session_id(),
            arrival_offset: clock_epoch.offset_at(observation.host_arrival())?,
            frame_id: observation.frame_id(),
            device_timestamp: observation.device_timestamp(),
        })
    }

    pub fn session_id(self) -> DeviceSessionId {
        self.session_id
    }

    pub fn arrival_offset(self) -> NavigationClockOffset {
        self.arrival_offset
    }

    pub fn replay_host_arrival(
        self,
        clock: NavigationReplayClock,
    ) -> Result<HostMonotonicTimestamp, NavigationReplayClockError> {
        clock.resolve(self.arrival_offset)
    }

    pub fn frame_id(self) -> FrameId {
        self.frame_id
    }

    pub fn device_timestamp(self) -> DeviceTimestamp {
        self.device_timestamp
    }
}

/// Original inertial payload with its process clock replaced by a recording
/// offset. Replay must explicitly reconstruct an [`ImuReport`] with a
/// [`NavigationReplayClock`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecordedImuReport {
    session_id: DeviceSessionId,
    sequence: DequeueSequence,
    arrival_offset: NavigationClockOffset,
    accel: AccelSample,
    gyro: GyroSample,
}

impl RecordedImuReport {
    pub fn parse(
        clock_epoch: NavigationClockEpoch,
        report: ImuReport,
    ) -> Result<Self, NavigationIngressBoundaryError> {
        Ok(Self {
            session_id: report.session_id(),
            sequence: report.sequence(),
            arrival_offset: clock_epoch.offset_at(report.host_arrival())?,
            accel: report.accel(),
            gyro: report.gyro(),
        })
    }

    pub fn session_id(self) -> DeviceSessionId {
        self.session_id
    }

    pub fn sequence(self) -> DequeueSequence {
        self.sequence
    }

    pub fn arrival_offset(self) -> NavigationClockOffset {
        self.arrival_offset
    }

    pub fn accel(self) -> AccelSample {
        self.accel
    }

    pub fn gyro(self) -> GyroSample {
        self.gyro
    }

    pub fn replay(
        self,
        clock: NavigationReplayClock,
    ) -> Result<ImuReport, NavigationReplayClockError> {
        Ok(ImuReport::new(
            self.session_id,
            self.sequence,
            clock.resolve(self.arrival_offset)?,
            self.accel,
            self.gyro,
        ))
    }
}

/// Coordinator-assigned map epoch used by a recorded point-goal command.
///
/// This is deliberately not [`crate::map::MapInstanceId`], whose value is
/// process-local. The live coordinator advances this ordinal whenever its
/// process-local map instance changes. Replay binds the same ordinal to the
/// corresponding replay-process map instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RecordedMapEpochId(NonZeroU64);

impl RecordedMapEpochId {
    pub fn try_new(raw: u64) -> Result<Self, RecordedMapEpochIdError> {
        NonZeroU64::new(raw)
            .map(Self)
            .ok_or(RecordedMapEpochIdError)
    }

    pub fn as_u64(self) -> u64 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecordedMapEpochIdError;

impl std::fmt::Display for RecordedMapEpochIdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("recorded map epoch ID must be nonzero")
    }
}

impl std::error::Error for RecordedMapEpochIdError {}

/// Sole live allocator for map epochs in one navigation recording.
///
/// The binding returned by [`start_epoch`](Self::start_epoch) has no public
/// constructor. It is the typed proof that one recorded epoch currently maps
/// to one process-local map instance.
#[derive(Debug)]
pub struct NavigationMapEpochCoordinator {
    next_epoch_id: Option<NonZeroU64>,
    current_map_instance_id: Option<MapInstanceId>,
}

impl NavigationMapEpochCoordinator {
    pub fn new() -> Self {
        Self {
            next_epoch_id: NonZeroU64::new(1),
            current_map_instance_id: None,
        }
    }

    pub fn start_epoch(
        &mut self,
        clock_epoch: NavigationClockEpoch,
        host_time: HostMonotonicTimestamp,
        map_instance_id: MapInstanceId,
    ) -> Result<MapEpochTransition, NavigationIngressBoundaryError> {
        if self.current_map_instance_id == Some(map_instance_id) {
            return Err(NavigationIngressBoundaryError::MapEpochAlreadyCurrent { map_instance_id });
        }
        let offset = clock_epoch.offset_at(host_time)?;
        let raw = self
            .next_epoch_id
            .ok_or(NavigationIngressBoundaryError::MapEpochIdExhausted)?;
        let map_epoch_id = RecordedMapEpochId(raw);
        self.next_epoch_id = raw.get().checked_add(1).and_then(NonZeroU64::new);
        self.current_map_instance_id = Some(map_instance_id);
        Ok(MapEpochTransition {
            event: MapEpochStartedIngress {
                offset,
                map_epoch_id,
            },
            binding: CurrentMapEpochBinding {
                map_epoch_id,
                map_instance_id,
            },
        })
    }
}

impl Default for NavigationMapEpochCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

/// One coordinator-issued transition and its live map binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MapEpochTransition {
    event: MapEpochStartedIngress,
    binding: CurrentMapEpochBinding,
}

impl MapEpochTransition {
    pub fn event(self) -> MapEpochStartedIngress {
        self.event
    }

    pub fn binding(self) -> CurrentMapEpochBinding {
        self.binding
    }
}

/// Non-forgeable live relation between a wire-stable epoch and a process map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CurrentMapEpochBinding {
    map_epoch_id: RecordedMapEpochId,
    map_instance_id: MapInstanceId,
}

impl CurrentMapEpochBinding {
    pub fn map_epoch_id(self) -> RecordedMapEpochId {
        self.map_epoch_id
    }

    pub fn map_instance_id(self) -> MapInstanceId {
        self.map_instance_id
    }
}

/// Explicit wire event establishing the next coordinator map epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MapEpochStartedIngress {
    offset: NavigationClockOffset,
    map_epoch_id: RecordedMapEpochId,
}

impl MapEpochStartedIngress {
    pub fn offset(self) -> NavigationClockOffset {
        self.offset
    }

    pub fn map_epoch_id(self) -> RecordedMapEpochId {
        self.map_epoch_id
    }

    pub fn replay_host_time(
        self,
        clock: NavigationReplayClock,
    ) -> Result<HostMonotonicTimestamp, NavigationReplayClockError> {
        clock.resolve(self.offset)
    }

    pub fn bind_replay_snapshot(
        self,
        snapshot: &OccupancyGridSnapshot,
    ) -> Result<ReplayMapEpochBinding, NavigationGoalReplayError> {
        let map_instance_id = snapshot
            .map_instance_id()
            .ok_or(NavigationGoalReplayError::MapHasNoInstance)?;
        Ok(ReplayMapEpochBinding {
            map_epoch_id: self.map_epoch_id,
            map_instance_id,
        })
    }
}

/// Replay-process map bound to one parsed recorded epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplayMapEpochBinding {
    map_epoch_id: RecordedMapEpochId,
    map_instance_id: MapInstanceId,
}

impl ReplayMapEpochBinding {
    pub fn map_epoch_id(self) -> RecordedMapEpochId {
        self.map_epoch_id
    }

    pub fn map_instance_id(self) -> MapInstanceId {
        self.map_instance_id
    }
}

/// A fully typed point-goal command bound to the displayed map epoch on which
/// it was selected.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MapPointGoalIngress {
    arrival_offset: NavigationClockOffset,
    map_epoch_id: RecordedMapEpochId,
    selected_revision: u64,
    point: MapPoint,
}

impl MapPointGoalIngress {
    pub fn parse(
        clock_epoch: NavigationClockEpoch,
        host_arrival: HostMonotonicTimestamp,
        binding: CurrentMapEpochBinding,
        goal: PointGoal,
    ) -> Result<Self, NavigationIngressBoundaryError> {
        if goal.map_instance_id() != binding.map_instance_id {
            return Err(NavigationIngressBoundaryError::GoalMapMismatch {
                map_epoch_id: binding.map_epoch_id,
                bound_map_instance_id: binding.map_instance_id,
                goal_map_instance_id: goal.map_instance_id(),
            });
        }
        Ok(Self {
            arrival_offset: clock_epoch.offset_at(host_arrival)?,
            map_epoch_id: binding.map_epoch_id,
            selected_revision: goal.selected_revision(),
            point: goal.point(),
        })
    }

    pub fn arrival_offset(self) -> NavigationClockOffset {
        self.arrival_offset
    }

    pub fn replay_host_arrival(
        self,
        clock: NavigationReplayClock,
    ) -> Result<HostMonotonicTimestamp, NavigationReplayClockError> {
        clock.resolve(self.arrival_offset)
    }

    pub fn map_epoch_id(self) -> RecordedMapEpochId {
        self.map_epoch_id
    }

    pub fn selected_revision(self) -> u64 {
        self.selected_revision
    }

    pub fn point(self) -> MapPoint {
        self.point
    }

    pub fn replay(
        self,
        binding: ReplayMapEpochBinding,
        snapshot: &OccupancyGridSnapshot,
    ) -> Result<PointGoal, NavigationGoalReplayError> {
        if self.map_epoch_id != binding.map_epoch_id {
            return Err(NavigationGoalReplayError::MapEpochMismatch {
                goal_epoch_id: self.map_epoch_id,
                bound_epoch_id: binding.map_epoch_id,
            });
        }
        let snapshot_map_instance_id = snapshot
            .map_instance_id()
            .ok_or(NavigationGoalReplayError::MapHasNoInstance)?;
        if snapshot_map_instance_id != binding.map_instance_id {
            return Err(NavigationGoalReplayError::SnapshotMapMismatch {
                bound_map_instance_id: binding.map_instance_id,
                snapshot_map_instance_id,
            });
        }
        if snapshot.revision() != self.selected_revision {
            return Err(NavigationGoalReplayError::SnapshotRevisionMismatch {
                expected: self.selected_revision,
                actual: snapshot.revision(),
            });
        }
        PointGoal::for_snapshot(self.point, snapshot)
            .map_err(|source| NavigationGoalReplayError::GoalConstruction { source })
    }
}

#[derive(Debug, PartialEq)]
pub enum NavigationGoalReplayError {
    MapHasNoInstance,
    MapEpochMismatch {
        goal_epoch_id: RecordedMapEpochId,
        bound_epoch_id: RecordedMapEpochId,
    },
    SnapshotMapMismatch {
        bound_map_instance_id: MapInstanceId,
        snapshot_map_instance_id: MapInstanceId,
    },
    SnapshotRevisionMismatch {
        expected: u64,
        actual: u64,
    },
    GoalConstruction {
        source: GlobalPlanError,
    },
}

impl std::fmt::Display for NavigationGoalReplayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "navigation goal replay failed: {self:?}")
    }
}

impl std::error::Error for NavigationGoalReplayError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::GoalConstruction { source } => Some(source),
            Self::MapHasNoInstance
            | Self::MapEpochMismatch { .. }
            | Self::SnapshotMapMismatch { .. }
            | Self::SnapshotRevisionMismatch { .. } => None,
        }
    }
}

/// One explicit navigation control-loop scheduling instant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControlTickIngress {
    offset: NavigationClockOffset,
}

impl ControlTickIngress {
    pub fn parse(
        clock_epoch: NavigationClockEpoch,
        host_time: HostMonotonicTimestamp,
    ) -> Result<Self, NavigationIngressBoundaryError> {
        Ok(Self {
            offset: clock_epoch.offset_at(host_time)?,
        })
    }

    pub fn offset(self) -> NavigationClockOffset {
        self.offset
    }

    pub fn replay_host_time(
        self,
        clock: NavigationReplayClock,
    ) -> Result<HostMonotonicTimestamp, NavigationReplayClockError> {
        clock.resolve(self.offset)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NavigationIngressEvent {
    VisualAttempt(VisualAttemptIngress),
    ImuReport(RecordedImuReport),
    AcceptedDepth(AcceptedDepthIngress),
    MapEpochStarted(MapEpochStartedIngress),
    PointGoal(MapPointGoalIngress),
    ControlTick(ControlTickIngress),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NavigationIngressRecord {
    sequence: NavigationIngressSequence,
    event: NavigationIngressEvent,
}

impl NavigationIngressRecord {
    pub fn sequence(self) -> NavigationIngressSequence {
        self.sequence
    }

    pub fn event(self) -> NavigationIngressEvent {
        self.event
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct NavigationIngressOrderState {
    current_map_epoch_id: Option<RecordedMapEpochId>,
}

impl NavigationIngressOrderState {
    fn validate(self, event: NavigationIngressEvent) -> Result<(), IngressOrderViolation> {
        match event {
            NavigationIngressEvent::MapEpochStarted(started) => {
                let expected = match self.current_map_epoch_id {
                    Some(current) => current.as_u64().checked_add(1).ok_or(
                        IngressOrderViolation::MapEpochSequenceExhausted {
                            current: current.as_u64(),
                        },
                    )?,
                    None => 1,
                };
                let actual = started.map_epoch_id.as_u64();
                if actual != expected {
                    return Err(IngressOrderViolation::MapEpochSequenceMismatch {
                        expected,
                        actual,
                    });
                }
            }
            NavigationIngressEvent::PointGoal(goal) => match self.current_map_epoch_id {
                None => return Err(IngressOrderViolation::GoalBeforeMapEpoch),
                Some(current) if current != goal.map_epoch_id => {
                    return Err(IngressOrderViolation::GoalMapEpochMismatch {
                        current: current.as_u64(),
                        goal: goal.map_epoch_id.as_u64(),
                    });
                }
                Some(_) => {}
            },
            NavigationIngressEvent::VisualAttempt(_)
            | NavigationIngressEvent::ImuReport(_)
            | NavigationIngressEvent::AcceptedDepth(_)
            | NavigationIngressEvent::ControlTick(_) => {}
        }
        Ok(())
    }

    fn commit(&mut self, event: NavigationIngressEvent) {
        if let NavigationIngressEvent::MapEpochStarted(started) = event {
            self.current_map_epoch_id = Some(started.map_epoch_id);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IngressOrderViolation {
    MapEpochSequenceMismatch { expected: u64, actual: u64 },
    MapEpochSequenceExhausted { current: u64 },
    GoalBeforeMapEpoch,
    GoalMapEpochMismatch { current: u64, goal: u64 },
}

/// Ordered, bounded navigation ingress journal.
#[derive(Debug, PartialEq)]
pub struct NavigationIngressLog {
    recording_id: NavigationRecordingId,
    capacity: NavigationIngressCapacity,
    records: Vec<NavigationIngressRecord>,
    order_state: NavigationIngressOrderState,
}

impl NavigationIngressLog {
    pub fn new(recording_id: NavigationRecordingId, capacity: NavigationIngressCapacity) -> Self {
        Self {
            recording_id,
            capacity,
            records: Vec::new(),
            order_state: NavigationIngressOrderState::default(),
        }
    }

    pub fn recording_id(&self) -> NavigationRecordingId {
        self.recording_id
    }

    pub fn capacity(&self) -> NavigationIngressCapacity {
        self.capacity
    }

    pub fn records(&self) -> &[NavigationIngressRecord] {
        self.records.as_slice()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Assign the next contiguous coordinator sequence and retain the event.
    /// Failure leaves the log unchanged.
    pub fn push(
        &mut self,
        event: NavigationIngressEvent,
    ) -> Result<NavigationIngressRecord, NavigationIngressWriteError> {
        self.order_state
            .validate(event)
            .map_err(NavigationIngressWriteError::from_order_violation)?;
        if self.records.len() == self.capacity.get() {
            return Err(NavigationIngressWriteError::CapacityExhausted {
                capacity: self.capacity.get(),
            });
        }
        let raw = u64::try_from(self.records.len())
            .ok()
            .and_then(|value| value.checked_add(1))
            .and_then(NonZeroU64::new)
            .ok_or(NavigationIngressWriteError::SequenceExhausted)?;
        let requested_records = self.records.len().saturating_add(1);
        let requested_bytes = requested_record_storage_bytes(requested_records)
            .ok_or(NavigationIngressWriteError::ByteLengthOverflow)?;
        self.records
            .try_reserve(1)
            .map_err(|_| NavigationIngressWriteError::AllocationFailed { requested_bytes })?;
        let record = NavigationIngressRecord {
            sequence: NavigationIngressSequence(raw),
            event,
        };
        self.records.push(record);
        self.order_state.commit(event);
        Ok(record)
    }

    /// Encode the journal as fixed-size little-endian V1 records.
    pub fn encode(&self) -> Result<Vec<u8>, NavigationIngressWriteError> {
        let record_bytes = self
            .records
            .len()
            .checked_mul(RECORD_BYTES)
            .and_then(|bytes| bytes.checked_add(HEADER_BYTES))
            .ok_or(NavigationIngressWriteError::ByteLengthOverflow)?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(record_bytes).map_err(|_| {
            NavigationIngressWriteError::AllocationFailed {
                requested_bytes: record_bytes,
            }
        })?;
        bytes.resize(record_bytes, 0);
        encode_header(
            &mut bytes[..HEADER_BYTES],
            self.records.len(),
            self.recording_id,
        )?;
        for (index, record) in self.records.iter().copied().enumerate() {
            let start = HEADER_BYTES + index * RECORD_BYTES;
            encode_record(&mut bytes[start..start + RECORD_BYTES], record);
        }
        Ok(bytes)
    }

    /// Parse an optional sidecar once into domain records.
    ///
    /// `None` is the only representation of a legacy dataset without recorded
    /// coordinator timing. It is not replaced by device timestamps or nominal
    /// frame periods.
    pub fn parse(
        bytes: Option<&[u8]>,
        expected_recording_id: NavigationRecordingId,
        capacity: NavigationIngressCapacity,
    ) -> Result<Self, NavigationIngressParseError> {
        let bytes = bytes.ok_or(NavigationIngressParseError::TimingUnavailable)?;
        if bytes.len() < HEADER_BYTES {
            return Err(NavigationIngressParseError::Truncated {
                expected_at_least: HEADER_BYTES,
                actual: bytes.len(),
            });
        }
        let count = parse_header(&bytes[..HEADER_BYTES], expected_recording_id)?;
        if count > capacity.get() {
            return Err(NavigationIngressParseError::RecordLimitExceeded {
                declared: count,
                limit: capacity.get(),
            });
        }
        let expected = count
            .checked_mul(RECORD_BYTES)
            .and_then(|value| value.checked_add(HEADER_BYTES))
            .ok_or(NavigationIngressParseError::ByteLengthOverflow {
                record_count: count,
            })?;
        if bytes.len() < expected {
            return Err(NavigationIngressParseError::Truncated {
                expected_at_least: expected,
                actual: bytes.len(),
            });
        }
        if bytes.len() > expected {
            return Err(NavigationIngressParseError::TrailingBytes {
                expected,
                actual: bytes.len(),
            });
        }

        let requested_bytes = requested_record_storage_bytes(count).ok_or(
            NavigationIngressParseError::ByteLengthOverflow {
                record_count: count,
            },
        )?;
        let mut records = Vec::new();
        records
            .try_reserve_exact(count)
            .map_err(|_| NavigationIngressParseError::AllocationFailed { requested_bytes })?;
        let mut order_state = NavigationIngressOrderState::default();
        for index in 0..count {
            let start = HEADER_BYTES + index * RECORD_BYTES;
            let record = parse_record(index, &bytes[start..start + RECORD_BYTES])?;
            order_state.validate(record.event).map_err(|violation| {
                NavigationIngressParseError::from_order_violation(index, violation)
            })?;
            order_state.commit(record.event);
            records.push(record);
        }
        Ok(Self {
            recording_id: expected_recording_id,
            capacity,
            records,
            order_state,
        })
    }
}

fn requested_record_storage_bytes(record_count: usize) -> Option<usize> {
    record_count.checked_mul(std::mem::size_of::<NavigationIngressRecord>())
}

/// Constant-memory writer for a live navigation ingress journal.
///
/// Construction accepts only a stream proven empty at position zero. File
/// callers must create or truncate the sidecar before constructing this type.
/// The writer checks both the current position and the stream end before it
/// writes a zero-count header.
/// Each successful [`append`](Self::append) writes one fixed-size record and
/// retains no event payload. [`finish`](Self::finish) patches the header count,
/// restores the stream position to the end, and flushes the stream.
///
/// An I/O failure while appending may have written a record prefix, so it
/// permanently poisons the writer and prevents publishing a plausible count.
#[derive(Debug)]
pub struct NavigationIngressWriter<W> {
    inner: W,
    recording_id: NavigationRecordingId,
    capacity: NavigationIngressCapacity,
    record_count: usize,
    order_state: NavigationIngressOrderState,
    poisoned: bool,
}

impl<W: Write + Seek> NavigationIngressWriter<W> {
    pub fn new(
        mut inner: W,
        recording_id: NavigationRecordingId,
        capacity: NavigationIngressCapacity,
    ) -> Result<Self, NavigationIngressStreamWriteError> {
        let current_position =
            inner
                .stream_position()
                .map_err(|source| NavigationIngressStreamWriteError::Io {
                    stage: NavigationIngressWriteStage::InspectInitialPosition,
                    source,
                })?;
        let end_position = inner.seek(SeekFrom::End(0)).map_err(|source| {
            NavigationIngressStreamWriteError::Io {
                stage: NavigationIngressWriteStage::InspectSinkEnd,
                source,
            }
        })?;
        if current_position != end_position {
            if let Err(restore) = inner.seek(SeekFrom::Start(current_position)) {
                return Err(NavigationIngressStreamWriteError::SinkViolationAndRestore {
                    current_position,
                    end_position,
                    restore,
                });
            }
            return Err(NavigationIngressStreamWriteError::SinkHasSuffix {
                current_position,
                end_position,
            });
        }
        if end_position != 0 {
            return Err(NavigationIngressStreamWriteError::SinkNotEmpty {
                length: end_position,
            });
        }

        let mut header = [0; HEADER_BYTES];
        encode_header(&mut header, 0, recording_id)
            .map_err(NavigationIngressStreamWriteError::Write)?;
        inner
            .write_all(&header)
            .map_err(|source| NavigationIngressStreamWriteError::Io {
                stage: NavigationIngressWriteStage::WriteHeader,
                source,
            })?;
        Ok(Self {
            inner,
            recording_id,
            capacity,
            record_count: 0,
            order_state: NavigationIngressOrderState::default(),
            poisoned: false,
        })
    }

    pub fn recording_id(&self) -> NavigationRecordingId {
        self.recording_id
    }

    pub fn capacity(&self) -> NavigationIngressCapacity {
        self.capacity
    }

    pub fn record_count(&self) -> usize {
        self.record_count
    }

    /// Assign and write one contiguous coordinator sequence.
    pub fn append(
        &mut self,
        event: NavigationIngressEvent,
    ) -> Result<NavigationIngressRecord, NavigationIngressStreamWriteError> {
        if self.poisoned {
            return Err(NavigationIngressStreamWriteError::Poisoned);
        }
        self.order_state.validate(event).map_err(|violation| {
            NavigationIngressStreamWriteError::Write(
                NavigationIngressWriteError::from_order_violation(violation),
            )
        })?;
        if self.record_count == self.capacity.get() {
            return Err(NavigationIngressStreamWriteError::Write(
                NavigationIngressWriteError::CapacityExhausted {
                    capacity: self.capacity.get(),
                },
            ));
        }
        let raw = u64::try_from(self.record_count)
            .ok()
            .and_then(|value| value.checked_add(1))
            .and_then(NonZeroU64::new)
            .ok_or(NavigationIngressStreamWriteError::Write(
                NavigationIngressWriteError::SequenceExhausted,
            ))?;
        let record = NavigationIngressRecord {
            sequence: NavigationIngressSequence(raw),
            event,
        };
        let mut bytes = [0; RECORD_BYTES];
        encode_record(&mut bytes, record);
        if let Err(source) = self.inner.write_all(&bytes) {
            self.poisoned = true;
            return Err(NavigationIngressStreamWriteError::Io {
                stage: NavigationIngressWriteStage::WriteRecord,
                source,
            });
        }
        self.record_count += 1;
        self.order_state.commit(event);
        Ok(record)
    }

    /// Publish the final record count and return the underlying stream.
    pub fn finish(mut self) -> Result<W, NavigationIngressStreamWriteError> {
        if self.poisoned {
            return Err(NavigationIngressStreamWriteError::Poisoned);
        }
        self.inner
            .flush()
            .map_err(|source| NavigationIngressStreamWriteError::Io {
                stage: NavigationIngressWriteStage::FlushRecords,
                source,
            })?;
        let expected_end = self
            .record_count
            .checked_mul(RECORD_BYTES)
            .and_then(|bytes| bytes.checked_add(HEADER_BYTES))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(NavigationIngressStreamWriteError::Write(
                NavigationIngressWriteError::ByteLengthOverflow,
            ))?;
        let current_position = self.inner.stream_position().map_err(|source| {
            NavigationIngressStreamWriteError::Io {
                stage: NavigationIngressWriteStage::InspectEndPosition,
                source,
            }
        })?;
        let end_position = self.inner.seek(SeekFrom::End(0)).map_err(|source| {
            NavigationIngressStreamWriteError::Io {
                stage: NavigationIngressWriteStage::InspectSinkEnd,
                source,
            }
        })?;
        if current_position != expected_end || end_position != expected_end {
            if let Err(restore) = self.inner.seek(SeekFrom::Start(current_position)) {
                return Err(
                    NavigationIngressStreamWriteError::SinkLengthMismatchAndRestore {
                        expected_end,
                        current_position,
                        end_position,
                        restore,
                    },
                );
            }
            return Err(NavigationIngressStreamWriteError::SinkLengthMismatch {
                expected_end,
                current_position,
                end_position,
            });
        }
        let record_count = u64::try_from(self.record_count).map_err(|_| {
            NavigationIngressStreamWriteError::Write(
                NavigationIngressWriteError::RecordCountOutOfRange {
                    count: self.record_count,
                },
            )
        })?;

        let patch_result = self
            .inner
            .seek(SeekFrom::Start(16))
            .and_then(|_| self.inner.write_all(&record_count.to_le_bytes()))
            .and_then(|_| self.inner.flush());
        let restore_result = self.inner.seek(SeekFrom::Start(expected_end));
        match (patch_result, restore_result) {
            (Err(finalize), Err(restore)) => {
                return Err(NavigationIngressStreamWriteError::FinalizeAndRestore {
                    finalize,
                    restore,
                });
            }
            (Err(source), Ok(_)) => {
                return Err(NavigationIngressStreamWriteError::Io {
                    stage: NavigationIngressWriteStage::PatchRecordCount,
                    source,
                });
            }
            (Ok(_), Err(source)) => {
                return Err(NavigationIngressStreamWriteError::Io {
                    stage: NavigationIngressWriteStage::RestoreEndPosition,
                    source,
                });
            }
            (Ok(_), Ok(_)) => {}
        }
        Ok(self.inner)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavigationIngressWriteStage {
    InspectInitialPosition,
    InspectSinkEnd,
    WriteHeader,
    WriteRecord,
    FlushRecords,
    InspectEndPosition,
    PatchRecordCount,
    RestoreEndPosition,
}

#[derive(Debug)]
pub enum NavigationIngressStreamWriteError {
    Write(NavigationIngressWriteError),
    Io {
        stage: NavigationIngressWriteStage,
        source: io::Error,
    },
    FinalizeAndRestore {
        finalize: io::Error,
        restore: io::Error,
    },
    SinkHasSuffix {
        current_position: u64,
        end_position: u64,
    },
    SinkNotEmpty {
        length: u64,
    },
    SinkViolationAndRestore {
        current_position: u64,
        end_position: u64,
        restore: io::Error,
    },
    SinkLengthMismatch {
        expected_end: u64,
        current_position: u64,
        end_position: u64,
    },
    SinkLengthMismatchAndRestore {
        expected_end: u64,
        current_position: u64,
        end_position: u64,
        restore: io::Error,
    },
    Poisoned,
}

impl std::fmt::Display for NavigationIngressStreamWriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Write(source) => write!(f, "navigation ingress stream write failed: {source}"),
            Self::Io { stage, source } => {
                write!(
                    f,
                    "navigation ingress stream I/O failed at {stage:?}: {source}"
                )
            }
            Self::FinalizeAndRestore { finalize, restore } => write!(
                f,
                "navigation ingress count finalization failed ({finalize}); restoring the end position also failed ({restore})"
            ),
            Self::SinkHasSuffix {
                current_position,
                end_position,
            } => write!(
                f,
                "navigation ingress sink has a pre-existing suffix: position {current_position}, end {end_position}"
            ),
            Self::SinkNotEmpty { length } => write!(
                f,
                "navigation ingress sink must be truncated before recording, found {length} bytes"
            ),
            Self::SinkViolationAndRestore {
                current_position,
                end_position,
                restore,
            } => write!(
                f,
                "navigation ingress sink position {current_position} precedes end {end_position}, and restoring the original position failed: {restore}"
            ),
            Self::SinkLengthMismatch {
                expected_end,
                current_position,
                end_position,
            } => write!(
                f,
                "navigation ingress sink length changed: expected end {expected_end}, current position {current_position}, actual end {end_position}"
            ),
            Self::SinkLengthMismatchAndRestore {
                expected_end,
                current_position,
                end_position,
                restore,
            } => write!(
                f,
                "navigation ingress sink length changed (expected {expected_end}, position {current_position}, end {end_position}), and restoring the position failed: {restore}"
            ),
            Self::Poisoned => {
                f.write_str("navigation ingress stream is poisoned by a prior I/O failure")
            }
        }
    }
}

impl std::error::Error for NavigationIngressStreamWriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Write(source) => Some(source),
            Self::Io { source, .. }
            | Self::FinalizeAndRestore {
                finalize: source, ..
            }
            | Self::SinkViolationAndRestore {
                restore: source, ..
            }
            | Self::SinkLengthMismatchAndRestore {
                restore: source, ..
            } => Some(source),
            Self::SinkHasSuffix { .. }
            | Self::SinkNotEmpty { .. }
            | Self::SinkLengthMismatch { .. }
            | Self::Poisoned => None,
        }
    }
}

/// Constant-memory reader for a navigation ingress journal.
///
/// The declared count is parsed and checked against `capacity` before the
/// first record is exposed. [`read_to_log`](Self::read_to_log) is therefore
/// allocation-bounded; [`next_record`](Self::next_record) retains no records.
#[derive(Debug)]
pub struct NavigationIngressReader<R> {
    inner: R,
    recording_id: NavigationRecordingId,
    capacity: NavigationIngressCapacity,
    declared_count: usize,
    next_record_index: usize,
    order_state: NavigationIngressOrderState,
    state: NavigationIngressReaderState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NavigationIngressReaderState {
    Active,
    Complete,
    Failed,
}

impl<R: Read> NavigationIngressReader<R> {
    pub fn new(
        mut inner: R,
        expected_recording_id: NavigationRecordingId,
        capacity: NavigationIngressCapacity,
    ) -> Result<Self, NavigationIngressStreamReadError> {
        let mut header = [0; HEADER_BYTES];
        let actual = read_exact_counted(&mut inner, &mut header).map_err(|source| {
            NavigationIngressStreamReadError::Io {
                stage: NavigationIngressReadStage::ReadHeader,
                source,
            }
        })?;
        if actual != HEADER_BYTES {
            return Err(NavigationIngressStreamReadError::Parse(
                NavigationIngressParseError::Truncated {
                    expected_at_least: HEADER_BYTES,
                    actual,
                },
            ));
        }
        let declared_count = parse_header(&header, expected_recording_id)
            .map_err(NavigationIngressStreamReadError::Parse)?;
        if declared_count > capacity.get() {
            return Err(NavigationIngressStreamReadError::Parse(
                NavigationIngressParseError::RecordLimitExceeded {
                    declared: declared_count,
                    limit: capacity.get(),
                },
            ));
        }
        Ok(Self {
            inner,
            recording_id: expected_recording_id,
            capacity,
            declared_count,
            next_record_index: 0,
            order_state: NavigationIngressOrderState::default(),
            state: NavigationIngressReaderState::Active,
        })
    }

    /// Parse a possibly absent timing sidecar without inventing replacement
    /// timestamps for legacy datasets.
    pub fn from_optional(
        inner: Option<R>,
        expected_recording_id: NavigationRecordingId,
        capacity: NavigationIngressCapacity,
    ) -> Result<Self, NavigationIngressStreamReadError> {
        let inner = inner.ok_or(NavigationIngressStreamReadError::Parse(
            NavigationIngressParseError::TimingUnavailable,
        ))?;
        Self::new(inner, expected_recording_id, capacity)
    }

    pub fn recording_id(&self) -> NavigationRecordingId {
        self.recording_id
    }

    pub fn declared_count(&self) -> usize {
        self.declared_count
    }

    pub fn remaining(&self) -> usize {
        self.declared_count.saturating_sub(self.next_record_index)
    }

    /// Read and parse one fixed-size record.
    ///
    /// The first call after the declared records also verifies that the stream
    /// has no trailing data. Any parse or I/O failure terminates this reader.
    pub fn next_record(
        &mut self,
    ) -> Result<Option<NavigationIngressRecord>, NavigationIngressStreamReadError> {
        match self.state {
            NavigationIngressReaderState::Complete => return Ok(None),
            NavigationIngressReaderState::Failed => {
                return Err(NavigationIngressStreamReadError::Failed);
            }
            NavigationIngressReaderState::Active => {}
        }
        if self.next_record_index == self.declared_count {
            let mut trailing = [0; 1];
            loop {
                match self.inner.read(&mut trailing) {
                    Ok(0) => {
                        self.state = NavigationIngressReaderState::Complete;
                        return Ok(None);
                    }
                    Ok(_) => {
                        self.state = NavigationIngressReaderState::Failed;
                        return Err(NavigationIngressStreamReadError::TrailingData {
                            expected_bytes: HEADER_BYTES
                                + self.declared_count.saturating_mul(RECORD_BYTES),
                        });
                    }
                    Err(source) if source.kind() == io::ErrorKind::Interrupted => continue,
                    Err(source) => {
                        self.state = NavigationIngressReaderState::Failed;
                        return Err(NavigationIngressStreamReadError::Io {
                            stage: NavigationIngressReadStage::CheckTrailingData,
                            source,
                        });
                    }
                }
            }
        }

        let record_index = self.next_record_index;
        let mut bytes = [0; RECORD_BYTES];
        let actual = match read_exact_counted(&mut self.inner, &mut bytes) {
            Ok(actual) => actual,
            Err(source) => {
                self.state = NavigationIngressReaderState::Failed;
                return Err(NavigationIngressStreamReadError::Io {
                    stage: NavigationIngressReadStage::ReadRecord { record_index },
                    source,
                });
            }
        };
        if actual != RECORD_BYTES {
            self.state = NavigationIngressReaderState::Failed;
            return Err(NavigationIngressStreamReadError::TruncatedRecord {
                record_index,
                expected_record_bytes: RECORD_BYTES,
                actual_record_bytes: actual,
            });
        }
        match parse_record(record_index, &bytes) {
            Ok(record) => {
                if let Err(violation) = self.order_state.validate(record.event) {
                    self.state = NavigationIngressReaderState::Failed;
                    return Err(NavigationIngressStreamReadError::Parse(
                        NavigationIngressParseError::from_order_violation(record_index, violation),
                    ));
                }
                self.order_state.commit(record.event);
                self.next_record_index += 1;
                Ok(Some(record))
            }
            Err(source) => {
                self.state = NavigationIngressReaderState::Failed;
                Err(NavigationIngressStreamReadError::Parse(source))
            }
        }
    }

    /// Materialize a fully checked log after enforcing its declared bound.
    pub fn read_to_log(mut self) -> Result<NavigationIngressLog, NavigationIngressStreamReadError> {
        let requested_bytes = requested_record_storage_bytes(self.declared_count).ok_or(
            NavigationIngressStreamReadError::Parse(
                NavigationIngressParseError::ByteLengthOverflow {
                    record_count: self.declared_count,
                },
            ),
        )?;
        let mut records = Vec::new();
        records
            .try_reserve_exact(self.declared_count)
            .map_err(|_| {
                NavigationIngressStreamReadError::Parse(
                    NavigationIngressParseError::AllocationFailed { requested_bytes },
                )
            })?;
        while let Some(record) = self.next_record()? {
            records.push(record);
        }
        Ok(NavigationIngressLog {
            recording_id: self.recording_id,
            capacity: self.capacity,
            records,
            order_state: self.order_state,
        })
    }
}

fn read_exact_counted<R: Read>(reader: &mut R, mut bytes: &mut [u8]) -> io::Result<usize> {
    let expected = bytes.len();
    while !bytes.is_empty() {
        match reader.read(bytes) {
            Ok(0) => break,
            Ok(read) => bytes = &mut bytes[read..],
            Err(source) if source.kind() == io::ErrorKind::Interrupted => {}
            Err(source) => return Err(source),
        }
    }
    Ok(expected - bytes.len())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavigationIngressReadStage {
    ReadHeader,
    ReadRecord { record_index: usize },
    CheckTrailingData,
}

#[derive(Debug)]
pub enum NavigationIngressStreamReadError {
    Parse(NavigationIngressParseError),
    Io {
        stage: NavigationIngressReadStage,
        source: io::Error,
    },
    TruncatedRecord {
        record_index: usize,
        expected_record_bytes: usize,
        actual_record_bytes: usize,
    },
    TrailingData {
        expected_bytes: usize,
    },
    Failed,
}

impl std::fmt::Display for NavigationIngressStreamReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(source) => write!(f, "navigation ingress stream parse failed: {source}"),
            Self::Io { stage, source } => {
                write!(
                    f,
                    "navigation ingress stream I/O failed at {stage:?}: {source}"
                )
            }
            Self::TruncatedRecord {
                record_index,
                expected_record_bytes,
                actual_record_bytes,
            } => write!(
                f,
                "navigation ingress record {record_index} is truncated: expected {expected_record_bytes} bytes, got {actual_record_bytes}"
            ),
            Self::TrailingData { expected_bytes } => write!(
                f,
                "navigation ingress stream has data after its declared {expected_bytes} bytes"
            ),
            Self::Failed => {
                f.write_str("navigation ingress reader is terminated by a prior failure")
            }
        }
    }
}

impl std::error::Error for NavigationIngressStreamReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Parse(source) => Some(source),
            Self::Io { source, .. } => Some(source),
            Self::TruncatedRecord { .. } | Self::TrailingData { .. } | Self::Failed => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavigationIngressWriteError {
    CapacityExhausted { capacity: usize },
    SequenceExhausted,
    ByteLengthOverflow,
    RecordCountOutOfRange { count: usize },
    AllocationFailed { requested_bytes: usize },
    MapEpochSequenceMismatch { expected: u64, actual: u64 },
    MapEpochSequenceExhausted { current: u64 },
    GoalBeforeMapEpoch,
    GoalMapEpochMismatch { current: u64, goal: u64 },
}

impl NavigationIngressWriteError {
    fn from_order_violation(violation: IngressOrderViolation) -> Self {
        match violation {
            IngressOrderViolation::MapEpochSequenceMismatch { expected, actual } => {
                Self::MapEpochSequenceMismatch { expected, actual }
            }
            IngressOrderViolation::MapEpochSequenceExhausted { current } => {
                Self::MapEpochSequenceExhausted { current }
            }
            IngressOrderViolation::GoalBeforeMapEpoch => Self::GoalBeforeMapEpoch,
            IngressOrderViolation::GoalMapEpochMismatch { current, goal } => {
                Self::GoalMapEpochMismatch { current, goal }
            }
        }
    }
}

impl std::fmt::Display for NavigationIngressWriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "navigation ingress write failed: {self:?}")
    }
}

impl std::error::Error for NavigationIngressWriteError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NavigationIngressParseError {
    TimingUnavailable,
    Truncated {
        expected_at_least: usize,
        actual: usize,
    },
    InvalidMagic,
    UnsupportedVersion {
        expected: u16,
        actual: u16,
    },
    HeaderLengthMismatch {
        expected: u16,
        actual: u16,
    },
    RecordLengthMismatch {
        expected: u16,
        actual: u16,
    },
    InvalidRecordingId,
    RecordingIdMismatch {
        expected: NavigationRecordingId,
        actual: NavigationRecordingId,
    },
    NonZeroReservedHeader,
    NonZeroReservedRecord {
        record_index: usize,
    },
    RecordCountOutOfRange {
        count: u64,
    },
    RecordLimitExceeded {
        declared: usize,
        limit: usize,
    },
    ByteLengthOverflow {
        record_count: usize,
    },
    TrailingBytes {
        expected: usize,
        actual: usize,
    },
    AllocationFailed {
        requested_bytes: usize,
    },
    SequenceMismatch {
        record_index: usize,
        expected: u64,
        actual: u64,
    },
    UnknownEventKind {
        record_index: usize,
        value: u8,
    },
    UnknownVisualOutcome {
        record_index: usize,
        value: u8,
    },
    ZeroDeviceSessionId {
        record_index: usize,
    },
    DeviceTimestampOutsideDomain {
        record_index: usize,
        field: &'static str,
        value: u64,
    },
    InvalidInertialValue {
        record_index: usize,
        field: &'static str,
        source: InertialValueError,
    },
    UnknownSensorAccuracy {
        record_index: usize,
        field: &'static str,
        value: u8,
    },
    ZeroMapEpochId {
        record_index: usize,
    },
    MapEpochSequenceMismatch {
        record_index: usize,
        expected: u64,
        actual: u64,
    },
    MapEpochSequenceExhausted {
        record_index: usize,
        current: u64,
    },
    GoalBeforeMapEpoch {
        record_index: usize,
    },
    GoalMapEpochMismatch {
        record_index: usize,
        current: u64,
        goal: u64,
    },
    InvalidGoalPoint {
        record_index: usize,
        source: super::PlanarPointError,
    },
}

impl NavigationIngressParseError {
    fn from_order_violation(record_index: usize, violation: IngressOrderViolation) -> Self {
        match violation {
            IngressOrderViolation::MapEpochSequenceMismatch { expected, actual } => {
                Self::MapEpochSequenceMismatch {
                    record_index,
                    expected,
                    actual,
                }
            }
            IngressOrderViolation::MapEpochSequenceExhausted { current } => {
                Self::MapEpochSequenceExhausted {
                    record_index,
                    current,
                }
            }
            IngressOrderViolation::GoalBeforeMapEpoch => Self::GoalBeforeMapEpoch { record_index },
            IngressOrderViolation::GoalMapEpochMismatch { current, goal } => {
                Self::GoalMapEpochMismatch {
                    record_index,
                    current,
                    goal,
                }
            }
        }
    }
}

impl std::fmt::Display for NavigationIngressParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "navigation ingress parse failed: {self:?}")
    }
}

impl std::error::Error for NavigationIngressParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidInertialValue { source, .. } => Some(source),
            Self::InvalidGoalPoint { source, .. } => Some(source),
            _ => None,
        }
    }
}

fn encode_header(
    bytes: &mut [u8],
    record_count: usize,
    recording_id: NavigationRecordingId,
) -> Result<(), NavigationIngressWriteError> {
    let record_count = u64::try_from(record_count).map_err(|_| {
        NavigationIngressWriteError::RecordCountOutOfRange {
            count: record_count,
        }
    })?;
    bytes[0..8].copy_from_slice(&MAGIC);
    put_u16(bytes, 8, NAVIGATION_INGRESS_FORMAT_VERSION);
    put_u16(bytes, 10, HEADER_BYTES as u16);
    put_u16(bytes, 12, RECORD_BYTES as u16);
    put_u64(bytes, 16, record_count);
    bytes[24..40].copy_from_slice(&recording_id.0);
    Ok(())
}

fn parse_header(
    bytes: &[u8],
    expected_recording_id: NavigationRecordingId,
) -> Result<usize, NavigationIngressParseError> {
    if bytes[0..8] != MAGIC {
        return Err(NavigationIngressParseError::InvalidMagic);
    }
    let version = get_u16(bytes, 8);
    if version != NAVIGATION_INGRESS_FORMAT_VERSION {
        return Err(NavigationIngressParseError::UnsupportedVersion {
            expected: NAVIGATION_INGRESS_FORMAT_VERSION,
            actual: version,
        });
    }
    let header_bytes = get_u16(bytes, 10);
    if usize::from(header_bytes) != HEADER_BYTES {
        return Err(NavigationIngressParseError::HeaderLengthMismatch {
            expected: HEADER_BYTES as u16,
            actual: header_bytes,
        });
    }
    let record_bytes = get_u16(bytes, 12);
    if usize::from(record_bytes) != RECORD_BYTES {
        return Err(NavigationIngressParseError::RecordLengthMismatch {
            expected: RECORD_BYTES as u16,
            actual: record_bytes,
        });
    }
    if bytes[14..16]
        .iter()
        .chain(&bytes[40..48])
        .any(|byte| *byte != 0)
    {
        return Err(NavigationIngressParseError::NonZeroReservedHeader);
    }
    let actual_recording_id =
        NavigationRecordingId::try_new(bytes[24..40].try_into().expect("fixed navigation header"))
            .map_err(|_| NavigationIngressParseError::InvalidRecordingId)?;
    if actual_recording_id != expected_recording_id {
        return Err(NavigationIngressParseError::RecordingIdMismatch {
            expected: expected_recording_id,
            actual: actual_recording_id,
        });
    }
    let count = get_u64(bytes, 16);
    usize::try_from(count).map_err(|_| NavigationIngressParseError::RecordCountOutOfRange { count })
}

fn encode_record(bytes: &mut [u8], record: NavigationIngressRecord) {
    put_u64(bytes, 0, record.sequence.as_u64());
    let (record_header, payload) = bytes.split_at_mut(RECORD_PAYLOAD_OFFSET);
    match record.event {
        NavigationIngressEvent::VisualAttempt(event) => {
            record_header[8] = KIND_VISUAL_ATTEMPT;
            put_u64(payload, 0, event.session_id.as_u64());
            put_u64(payload, 8, event.arrival_offset.as_nanos());
            put_u64(payload, 16, event.left_frame_id.as_u64());
            put_u64(payload, 24, event.left_timestamp.as_nanos());
            put_u64(payload, 32, event.right_frame_id.as_u64());
            put_u64(payload, 40, event.right_timestamp.as_nanos());
            payload[48] = event.outcome.wire_code();
        }
        NavigationIngressEvent::ImuReport(report) => {
            record_header[8] = KIND_IMU_REPORT;
            put_u64(payload, 0, report.session_id().as_u64());
            put_u64(payload, 8, report.arrival_offset().as_nanos());
            put_u32(payload, 16, report.sequence().as_u32());
            payload[20] = encode_accuracy(report.accel().accuracy());
            payload[21] = encode_accuracy(report.gyro().accuracy());
            put_u64(payload, 24, report.accel().timestamp().as_nanos());
            put_u64(payload, 32, report.gyro().timestamp().as_nanos());
            for (index, value) in report
                .accel()
                .acceleration()
                .as_array()
                .into_iter()
                .chain(report.gyro().angular_velocity().as_array())
                .enumerate()
            {
                put_f64(payload, 40 + index * 8, value);
            }
        }
        NavigationIngressEvent::AcceptedDepth(event) => {
            record_header[8] = KIND_ACCEPTED_DEPTH;
            put_u64(payload, 0, event.session_id.as_u64());
            put_u64(payload, 8, event.arrival_offset.as_nanos());
            put_u64(payload, 16, event.frame_id.as_u64());
            put_u64(payload, 24, event.device_timestamp.as_nanos());
        }
        NavigationIngressEvent::PointGoal(event) => {
            record_header[8] = KIND_POINT_GOAL;
            put_u64(payload, 0, event.arrival_offset.as_nanos());
            put_u64(payload, 8, event.map_epoch_id.as_u64());
            put_u64(payload, 16, event.selected_revision);
            put_f64(payload, 24, event.point.x_m());
            put_f64(payload, 32, event.point.y_m());
        }
        NavigationIngressEvent::MapEpochStarted(event) => {
            record_header[8] = KIND_MAP_EPOCH_STARTED;
            put_u64(payload, 0, event.offset.as_nanos());
            put_u64(payload, 8, event.map_epoch_id.as_u64());
        }
        NavigationIngressEvent::ControlTick(event) => {
            record_header[8] = KIND_CONTROL_TICK;
            put_u64(payload, 0, event.offset.as_nanos());
        }
    }
}

fn parse_record(
    record_index: usize,
    bytes: &[u8],
) -> Result<NavigationIngressRecord, NavigationIngressParseError> {
    let expected = u64::try_from(record_index)
        .ok()
        .and_then(|index| index.checked_add(1))
        .ok_or(NavigationIngressParseError::RecordCountOutOfRange { count: u64::MAX })?;
    let actual = get_u64(bytes, 0);
    if actual != expected {
        return Err(NavigationIngressParseError::SequenceMismatch {
            record_index,
            expected,
            actual,
        });
    }
    if bytes[9..RECORD_PAYLOAD_OFFSET]
        .iter()
        .any(|byte| *byte != 0)
    {
        return Err(NavigationIngressParseError::NonZeroReservedRecord { record_index });
    }
    let sequence = NavigationIngressSequence(
        NonZeroU64::new(actual).expect("one-based expected sequence is nonzero"),
    );
    let payload = &bytes[RECORD_PAYLOAD_OFFSET..];
    let event = match bytes[8] {
        KIND_VISUAL_ATTEMPT => parse_visual(record_index, payload)?,
        KIND_IMU_REPORT => parse_imu(record_index, payload)?,
        KIND_ACCEPTED_DEPTH => parse_depth(record_index, payload)?,
        KIND_POINT_GOAL => parse_goal(record_index, payload)?,
        KIND_MAP_EPOCH_STARTED => parse_map_epoch_started(record_index, payload)?,
        KIND_CONTROL_TICK => parse_control_tick(record_index, payload)?,
        value => {
            return Err(NavigationIngressParseError::UnknownEventKind {
                record_index,
                value,
            });
        }
    };
    Ok(NavigationIngressRecord { sequence, event })
}

fn parse_visual(
    record_index: usize,
    payload: &[u8],
) -> Result<NavigationIngressEvent, NavigationIngressParseError> {
    require_zero(record_index, &payload[49..])?;
    Ok(NavigationIngressEvent::VisualAttempt(
        VisualAttemptIngress {
            session_id: parse_session(record_index, get_u64(payload, 0))?,
            arrival_offset: NavigationClockOffset(get_u64(payload, 8)),
            left_frame_id: FrameId::new(get_u64(payload, 16)),
            left_timestamp: parse_device_timestamp(
                record_index,
                "visual.left_timestamp_ns",
                get_u64(payload, 24),
            )?,
            right_frame_id: FrameId::new(get_u64(payload, 32)),
            right_timestamp: parse_device_timestamp(
                record_index,
                "visual.right_timestamp_ns",
                get_u64(payload, 40),
            )?,
            outcome: VisualAttemptOutcome::parse_wire(record_index, payload[48])?,
        },
    ))
}

fn parse_imu(
    record_index: usize,
    payload: &[u8],
) -> Result<NavigationIngressEvent, NavigationIngressParseError> {
    require_zero(record_index, &payload[22..24])?;
    require_zero(record_index, &payload[88..])?;
    let accel_accuracy = parse_accuracy(record_index, "imu.accel_accuracy", payload[20])?;
    let gyro_accuracy = parse_accuracy(record_index, "imu.gyro_accuracy", payload[21])?;
    let accel_timestamp =
        parse_device_timestamp(record_index, "imu.accel_timestamp_ns", get_u64(payload, 24))?;
    let gyro_timestamp =
        parse_device_timestamp(record_index, "imu.gyro_timestamp_ns", get_u64(payload, 32))?;
    let accel = OakImuAcceleration::try_new(
        get_f64(payload, 40),
        get_f64(payload, 48),
        get_f64(payload, 56),
    )
    .map_err(|source| NavigationIngressParseError::InvalidInertialValue {
        record_index,
        field: "imu.acceleration_m_s2",
        source,
    })?;
    let gyro = OakImuAngularVelocity::try_new(
        get_f64(payload, 64),
        get_f64(payload, 72),
        get_f64(payload, 80),
    )
    .map_err(|source| NavigationIngressParseError::InvalidInertialValue {
        record_index,
        field: "imu.angular_velocity_rad_s",
        source,
    })?;
    Ok(NavigationIngressEvent::ImuReport(RecordedImuReport {
        session_id: parse_session(record_index, get_u64(payload, 0))?,
        sequence: DequeueSequence::new(get_u32(payload, 16)),
        arrival_offset: NavigationClockOffset(get_u64(payload, 8)),
        accel: AccelSample::new(accel_timestamp, accel, accel_accuracy),
        gyro: GyroSample::new(gyro_timestamp, gyro, gyro_accuracy),
    }))
}

fn parse_depth(
    record_index: usize,
    payload: &[u8],
) -> Result<NavigationIngressEvent, NavigationIngressParseError> {
    require_zero(record_index, &payload[32..])?;
    Ok(NavigationIngressEvent::AcceptedDepth(
        AcceptedDepthIngress {
            session_id: parse_session(record_index, get_u64(payload, 0))?,
            arrival_offset: NavigationClockOffset(get_u64(payload, 8)),
            frame_id: FrameId::new(get_u64(payload, 16)),
            device_timestamp: parse_device_timestamp(
                record_index,
                "depth.device_timestamp_ns",
                get_u64(payload, 24),
            )?,
        },
    ))
}

fn parse_goal(
    record_index: usize,
    payload: &[u8],
) -> Result<NavigationIngressEvent, NavigationIngressParseError> {
    require_zero(record_index, &payload[40..])?;
    let map_epoch_id = RecordedMapEpochId::try_new(get_u64(payload, 8))
        .map_err(|_| NavigationIngressParseError::ZeroMapEpochId { record_index })?;
    let point =
        MapPoint::try_new(get_f64(payload, 24), get_f64(payload, 32)).map_err(|source| {
            NavigationIngressParseError::InvalidGoalPoint {
                record_index,
                source,
            }
        })?;
    Ok(NavigationIngressEvent::PointGoal(MapPointGoalIngress {
        arrival_offset: NavigationClockOffset(get_u64(payload, 0)),
        map_epoch_id,
        selected_revision: get_u64(payload, 16),
        point,
    }))
}

fn parse_map_epoch_started(
    record_index: usize,
    payload: &[u8],
) -> Result<NavigationIngressEvent, NavigationIngressParseError> {
    require_zero(record_index, &payload[16..])?;
    let map_epoch_id = RecordedMapEpochId::try_new(get_u64(payload, 8))
        .map_err(|_| NavigationIngressParseError::ZeroMapEpochId { record_index })?;
    Ok(NavigationIngressEvent::MapEpochStarted(
        MapEpochStartedIngress {
            offset: NavigationClockOffset(get_u64(payload, 0)),
            map_epoch_id,
        },
    ))
}

fn parse_control_tick(
    record_index: usize,
    payload: &[u8],
) -> Result<NavigationIngressEvent, NavigationIngressParseError> {
    require_zero(record_index, &payload[8..])?;
    Ok(NavigationIngressEvent::ControlTick(ControlTickIngress {
        offset: NavigationClockOffset(get_u64(payload, 0)),
    }))
}

fn parse_session(
    record_index: usize,
    raw: u64,
) -> Result<DeviceSessionId, NavigationIngressParseError> {
    DeviceSessionId::try_new(raw)
        .map_err(|_| NavigationIngressParseError::ZeroDeviceSessionId { record_index })
}

fn parse_device_timestamp(
    record_index: usize,
    field: &'static str,
    raw: u64,
) -> Result<DeviceTimestamp, NavigationIngressParseError> {
    let signed = i64::try_from(raw).map_err(|_| {
        NavigationIngressParseError::DeviceTimestampOutsideDomain {
            record_index,
            field,
            value: raw,
        }
    })?;
    DeviceTimestamp::try_from_nanos(signed).map_err(|source| {
        NavigationIngressParseError::InvalidInertialValue {
            record_index,
            field,
            source,
        }
    })
}

fn encode_accuracy(value: SensorAccuracy) -> u8 {
    match value {
        SensorAccuracy::Unreliable => 0,
        SensorAccuracy::Low => 1,
        SensorAccuracy::Medium => 2,
        SensorAccuracy::High => 3,
    }
}

fn parse_accuracy(
    record_index: usize,
    field: &'static str,
    value: u8,
) -> Result<SensorAccuracy, NavigationIngressParseError> {
    match value {
        0 => Ok(SensorAccuracy::Unreliable),
        1 => Ok(SensorAccuracy::Low),
        2 => Ok(SensorAccuracy::Medium),
        3 => Ok(SensorAccuracy::High),
        _ => Err(NavigationIngressParseError::UnknownSensorAccuracy {
            record_index,
            field,
            value,
        }),
    }
}

fn require_zero(record_index: usize, bytes: &[u8]) -> Result<(), NavigationIngressParseError> {
    if bytes.iter().any(|byte| *byte != 0) {
        Err(NavigationIngressParseError::NonZeroReservedRecord { record_index })
    } else {
        Ok(())
    }
}

fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn put_f64(bytes: &mut [u8], offset: usize, value: f64) {
    put_u64(bytes, offset, value.to_bits());
}

fn get_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn get_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("fixed record"))
}

fn get_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("fixed record"))
}

fn get_f64(bytes: &[u8], offset: usize) -> f64 {
    f64::from_bits(get_u64(bytes, offset))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DepthImage, Frame, PairingWindowNs, SensorId, StereoObservation, StereoPair, Timestamp,
        dense::occupancy::{OccupancyCell, OccupancyGridGeometry, OccupancyGridSnapshot},
        map::SlamMap,
    };
    use std::io::Cursor;

    #[derive(Debug)]
    struct ByteBudgetStream {
        inner: Cursor<Vec<u8>>,
        remaining_write_bytes: usize,
    }

    impl Write for ByteBudgetStream {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.remaining_write_bytes == 0 {
                return Err(io::Error::other("test byte budget exhausted"));
            }
            let allowed = bytes.len().min(self.remaining_write_bytes);
            let written = self.inner.write(&bytes[..allowed])?;
            self.remaining_write_bytes -= written;
            Ok(written)
        }

        fn flush(&mut self) -> io::Result<()> {
            self.inner.flush()
        }
    }

    impl Seek for ByteBudgetStream {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            self.inner.seek(position)
        }
    }

    #[derive(Debug)]
    struct FailingAbsoluteSeekStream {
        inner: Cursor<Vec<u8>>,
        fail_absolute_seeks: bool,
    }

    impl Write for FailingAbsoluteSeekStream {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.inner.write(bytes)
        }

        fn flush(&mut self) -> io::Result<()> {
            self.inner.flush()
        }
    }

    impl Seek for FailingAbsoluteSeekStream {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            if self.fail_absolute_seeks && matches!(position, SeekFrom::Start(_)) {
                Err(io::Error::other("test absolute seek failure"))
            } else {
                self.inner.seek(position)
            }
        }
    }

    fn capacity(value: usize) -> NavigationIngressCapacity {
        NavigationIngressCapacity::try_new(value).expect("valid capacity")
    }

    fn recording_id(seed: u8) -> NavigationRecordingId {
        NavigationRecordingId::try_new([seed; 16]).expect("nonzero recording ID")
    }

    fn clock() -> NavigationClockEpoch {
        NavigationClockEpoch::new(HostMonotonicTimestamp::from_nanos(700))
    }

    fn host_time(nanos: u64) -> HostMonotonicTimestamp {
        HostMonotonicTimestamp::from_nanos(nanos)
    }

    fn session() -> DeviceSessionId {
        DeviceSessionId::try_new(7).expect("valid session")
    }

    fn device_time(value: i64) -> DeviceTimestamp {
        DeviceTimestamp::try_from_nanos(value).expect("valid device time")
    }

    fn imu_report() -> ImuReport {
        ImuReport::new(
            session(),
            DequeueSequence::new(11),
            host_time(900),
            AccelSample::new(
                device_time(100),
                OakImuAcceleration::try_new(1.0, 2.0, 3.0).expect("acceleration"),
                SensorAccuracy::Medium,
            ),
            GyroSample::new(
                device_time(101),
                OakImuAngularVelocity::try_new(0.1, 0.2, 0.3).expect("angular velocity"),
                SensorAccuracy::High,
            ),
        )
    }

    fn recorded_imu() -> RecordedImuReport {
        RecordedImuReport::parse(clock(), imu_report()).expect("recorded IMU")
    }

    fn pair() -> StereoPair {
        let left = Frame::new(
            SensorId::StereoLeft,
            FrameId::new(21),
            Timestamp::from_nanos(200),
            1,
            1,
            vec![1],
        )
        .expect("left frame");
        let right = Frame::new(
            SensorId::StereoRight,
            FrameId::new(22),
            Timestamp::from_nanos(201),
            1,
            1,
            vec![2],
        )
        .expect("right frame");
        StereoPair::try_new(
            left,
            right,
            PairingWindowNs::try_from_u64(2).expect("window"),
        )
        .expect("pair")
    }

    fn stereo_observation() -> StereoObservation {
        StereoObservation::parse(session(), host_time(800), pair()).expect("stereo observation")
    }

    fn depth() -> DepthObservation {
        let image = DepthImage::new(
            FrameId::new(31),
            Timestamp::from_nanos(300),
            1,
            1,
            vec![1.5],
        )
        .expect("depth");
        DepthObservation::parse(session(), host_time(950), image).expect("depth observation")
    }

    fn geometry() -> OccupancyGridGeometry {
        OccupancyGridGeometry::try_new(1.0, [0.0, 0.0], 1, 1, 1).expect("geometry")
    }

    fn snapshot_for(map_instance_id: MapInstanceId, revision: u64) -> OccupancyGridSnapshot {
        OccupancyGridSnapshot::from_test_cells(
            geometry(),
            &[OccupancyCell::Free],
            map_instance_id,
            revision,
        )
    }

    fn snapshot(revision: u64) -> OccupancyGridSnapshot {
        snapshot_for(SlamMap::new().snapshot().instance_id(), revision)
    }

    fn point_goal(snapshot: &OccupancyGridSnapshot) -> PointGoal {
        PointGoal::for_snapshot(MapPoint::try_new(0.5, 0.5).expect("point"), snapshot)
            .expect("goal")
    }

    fn transition(
        coordinator: &mut NavigationMapEpochCoordinator,
        snapshot: &OccupancyGridSnapshot,
        at_ns: u64,
    ) -> MapEpochTransition {
        coordinator
            .start_epoch(
                clock(),
                host_time(at_ns),
                snapshot.map_instance_id().expect("map instance"),
            )
            .expect("map epoch")
    }

    fn mixed_events() -> [NavigationIngressEvent; 7] {
        let snapshot = snapshot(4);
        let mut coordinator = NavigationMapEpochCoordinator::new();
        let transition = transition(&mut coordinator, &snapshot, 970);
        [
            NavigationIngressEvent::VisualAttempt(
                VisualAttemptIngress::from_observation(
                    clock(),
                    &stereo_observation(),
                    VisualAttemptOutcome::IncrementAndLocalization,
                )
                .expect("visual ingress"),
            ),
            NavigationIngressEvent::ImuReport(recorded_imu()),
            NavigationIngressEvent::AcceptedDepth(
                AcceptedDepthIngress::parse(clock(), &depth()).expect("depth ingress"),
            ),
            NavigationIngressEvent::MapEpochStarted(transition.event()),
            NavigationIngressEvent::PointGoal(
                MapPointGoalIngress::parse(
                    clock(),
                    host_time(1_000),
                    transition.binding(),
                    point_goal(&snapshot),
                )
                .expect("goal ingress"),
            ),
            NavigationIngressEvent::ControlTick(
                ControlTickIngress::parse(clock(), host_time(1_010)).expect("control tick"),
            ),
            NavigationIngressEvent::ImuReport(recorded_imu()),
        ]
    }

    #[test]
    fn visual_ingress_uses_the_already_parsed_stereo_boundary() {
        let observation = stereo_observation();
        let ingress = VisualAttemptIngress::from_observation(
            clock(),
            &observation,
            VisualAttemptOutcome::LocalizationOnly,
        )
        .expect("ingress");
        assert_eq!(ingress.session_id(), observation.session_id());
        assert_eq!(
            ingress.left_timestamp(),
            observation.left_device_timestamp()
        );
        assert_eq!(
            ingress.right_timestamp(),
            observation.right_device_timestamp()
        );
        assert_eq!(
            ingress.left_frame_id(),
            observation.pair().left().frame_id()
        );
        assert_eq!(
            ingress.right_frame_id(),
            observation.pair().right().frame_id()
        );
    }

    #[test]
    fn mixed_records_round_trip_in_exact_admission_order() {
        let events = mixed_events();
        let mut log = NavigationIngressLog::new(recording_id(1), capacity(8));
        for event in events {
            log.push(event).expect("append");
        }
        let encoded = log.encode().expect("encode");
        let decoded = NavigationIngressLog::parse(Some(&encoded), recording_id(1), capacity(8))
            .expect("decode");
        assert_eq!(decoded.recording_id(), recording_id(1));
        assert_eq!(decoded.records(), log.records());
        assert_eq!(
            decoded
                .records()
                .iter()
                .map(|record| record.sequence().as_u64())
                .collect::<Vec<_>>(),
            (1..=events.len() as u64).collect::<Vec<_>>()
        );
    }

    #[test]
    fn wrong_recording_id_precedes_record_parsing() {
        let mut log = NavigationIngressLog::new(recording_id(1), capacity(1));
        log.push(NavigationIngressEvent::ImuReport(recorded_imu()))
            .expect("record");
        let mut bytes = log.encode().expect("encode");
        bytes[HEADER_BYTES + 8] = u8::MAX;
        assert_eq!(
            NavigationIngressLog::parse(Some(&bytes), recording_id(2), capacity(1)),
            Err(NavigationIngressParseError::RecordingIdMismatch {
                expected: recording_id(2),
                actual: recording_id(1),
            })
        );
    }

    #[test]
    fn absent_legacy_sidecar_reports_timing_unavailable() {
        assert_eq!(
            NavigationIngressLog::parse(None, recording_id(1), capacity(1)),
            Err(NavigationIngressParseError::TimingUnavailable)
        );
    }

    #[test]
    fn every_truncated_prefix_is_rejected_as_truncated() {
        let mut log = NavigationIngressLog::new(recording_id(1), capacity(2));
        log.push(NavigationIngressEvent::ImuReport(recorded_imu()))
            .expect("record");
        let bytes = log.encode().expect("encode");
        for length in 0..bytes.len() {
            assert!(matches!(
                NavigationIngressLog::parse(Some(&bytes[..length]), recording_id(1), capacity(2),),
                Err(NavigationIngressParseError::Truncated { .. })
            ));
        }
    }

    #[test]
    fn unsupported_version_is_distinct_from_truncation() {
        let log = NavigationIngressLog::new(recording_id(1), capacity(1));
        let mut bytes = log.encode().expect("encode");
        put_u16(&mut bytes, 8, NAVIGATION_INGRESS_FORMAT_VERSION + 1);
        assert_eq!(
            NavigationIngressLog::parse(Some(&bytes), recording_id(1), capacity(1)),
            Err(NavigationIngressParseError::UnsupportedVersion {
                expected: NAVIGATION_INGRESS_FORMAT_VERSION,
                actual: NAVIGATION_INGRESS_FORMAT_VERSION + 1,
            })
        );
    }

    #[test]
    fn record_limit_and_sequence_are_transactional() {
        let mut log = NavigationIngressLog::new(recording_id(1), capacity(1));
        log.push(NavigationIngressEvent::ImuReport(recorded_imu()))
            .expect("first");
        let before = log.records().to_vec();
        assert_eq!(
            log.push(NavigationIngressEvent::ImuReport(recorded_imu())),
            Err(NavigationIngressWriteError::CapacityExhausted { capacity: 1 })
        );
        assert_eq!(log.records(), before);

        let mut bytes = log.encode().expect("encode");
        put_u64(&mut bytes, HEADER_BYTES, 2);
        assert_eq!(
            NavigationIngressLog::parse(Some(&bytes), recording_id(1), capacity(1)),
            Err(NavigationIngressParseError::SequenceMismatch {
                record_index: 0,
                expected: 1,
                actual: 2,
            })
        );
    }

    #[test]
    fn declared_count_cannot_exceed_replay_bound() {
        let mut bytes = NavigationIngressLog::new(recording_id(1), capacity(2))
            .encode()
            .expect("empty log");
        put_u64(&mut bytes, 16, 2);
        assert_eq!(
            NavigationIngressLog::parse(Some(&bytes), recording_id(1), capacity(1)),
            Err(NavigationIngressParseError::RecordLimitExceeded {
                declared: 2,
                limit: 1,
            })
        );
    }

    #[test]
    fn shifted_replay_origin_reconstructs_imu_without_persisting_host_epoch() {
        let recorded = recorded_imu();
        assert_eq!(recorded.arrival_offset().as_nanos(), 200);
        let replay = recorded
            .replay(NavigationReplayClock::new(host_time(5_000)))
            .expect("replay");
        assert_eq!(replay.host_arrival(), host_time(5_200));
        assert_ne!(replay.host_arrival(), imu_report().host_arrival());
        assert!(matches!(
            NavigationReplayClock::new(host_time(u64::MAX)).resolve(NavigationClockOffset(1),),
            Err(NavigationReplayClockError::TimestampOverflow { .. })
        ));
    }

    #[test]
    fn live_clock_rejects_timestamp_before_recording_origin() {
        assert_eq!(
            clock().offset_at(host_time(699)),
            Err(NavigationIngressBoundaryError::HostTimeBeforeClockEpoch {
                origin_ns: 700,
                timestamp_ns: 699,
            })
        );
    }

    #[test]
    fn cross_map_goal_is_rejected_by_nonforgeable_binding() {
        let first = snapshot(4);
        let second = snapshot(4);
        let mut coordinator = NavigationMapEpochCoordinator::new();
        let transition = transition(&mut coordinator, &first, 800);
        assert!(matches!(
            MapPointGoalIngress::parse(
                clock(),
                host_time(900),
                transition.binding(),
                point_goal(&second),
            ),
            Err(NavigationIngressBoundaryError::GoalMapMismatch { .. })
        ));
    }

    #[test]
    fn replay_goal_requires_bound_map_and_exact_selected_revision() {
        let live_snapshot = snapshot(4);
        let mut coordinator = NavigationMapEpochCoordinator::new();
        let transition = transition(&mut coordinator, &live_snapshot, 800);
        let recorded = MapPointGoalIngress::parse(
            clock(),
            host_time(900),
            transition.binding(),
            point_goal(&live_snapshot),
        )
        .expect("recorded goal");

        let replay_map_id = SlamMap::new().snapshot().instance_id();
        let exact = snapshot_for(replay_map_id, 4);
        let binding = transition
            .event()
            .bind_replay_snapshot(&exact)
            .expect("replay binding");
        let replayed = recorded.replay(binding, &exact).expect("replayed goal");
        assert_eq!(replayed.map_instance_id(), replay_map_id);
        assert_eq!(replayed.selected_revision(), 4);

        let shifted_revision = snapshot_for(replay_map_id, 5);
        assert!(matches!(
            recorded.replay(binding, &shifted_revision),
            Err(NavigationGoalReplayError::SnapshotRevisionMismatch {
                expected: 4,
                actual: 5,
            })
        ));
        let other_map = snapshot(4);
        assert!(matches!(
            recorded.replay(binding, &other_map),
            Err(NavigationGoalReplayError::SnapshotMapMismatch { .. })
        ));
    }

    #[test]
    fn map_epochs_are_explicit_and_strictly_ordered() {
        let first = snapshot(1);
        let second = snapshot(1);
        let mut coordinator = NavigationMapEpochCoordinator::new();
        let first_transition = transition(&mut coordinator, &first, 800);
        let second_transition = transition(&mut coordinator, &second, 900);
        let mut log = NavigationIngressLog::new(recording_id(1), capacity(2));
        log.push(NavigationIngressEvent::MapEpochStarted(
            first_transition.event(),
        ))
        .expect("first epoch");
        log.push(NavigationIngressEvent::MapEpochStarted(
            second_transition.event(),
        ))
        .expect("second epoch");
        let mut bytes = log.encode().expect("encode");
        let second_payload = HEADER_BYTES + RECORD_BYTES + RECORD_PAYLOAD_OFFSET;
        put_u64(&mut bytes[second_payload..], 8, 3);
        assert_eq!(
            NavigationIngressLog::parse(Some(&bytes), recording_id(1), capacity(2)),
            Err(NavigationIngressParseError::MapEpochSequenceMismatch {
                record_index: 1,
                expected: 2,
                actual: 3,
            })
        );
    }

    #[test]
    fn goal_requires_a_preceding_current_map_epoch() {
        let snapshot = snapshot(1);
        let mut coordinator = NavigationMapEpochCoordinator::new();
        let transition = transition(&mut coordinator, &snapshot, 800);
        let goal = MapPointGoalIngress::parse(
            clock(),
            host_time(900),
            transition.binding(),
            point_goal(&snapshot),
        )
        .expect("goal");
        let mut log = NavigationIngressLog::new(recording_id(1), capacity(1));
        assert_eq!(
            log.push(NavigationIngressEvent::PointGoal(goal)),
            Err(NavigationIngressWriteError::GoalBeforeMapEpoch)
        );
    }

    #[test]
    fn map_wire_never_contains_process_local_map_identity() {
        let snapshot = loop {
            let candidate = snapshot(4);
            if candidate.map_instance_id().expect("map").as_u64() != 1 {
                break candidate;
            }
        };
        let process_id = snapshot.map_instance_id().expect("map").as_u64();
        let mut coordinator = NavigationMapEpochCoordinator::new();
        let transition = transition(&mut coordinator, &snapshot, 800);
        let goal = MapPointGoalIngress::parse(
            clock(),
            host_time(900),
            transition.binding(),
            point_goal(&snapshot),
        )
        .expect("goal");
        let mut log = NavigationIngressLog::new(recording_id(1), capacity(2));
        log.push(NavigationIngressEvent::MapEpochStarted(transition.event()))
            .expect("epoch");
        log.push(NavigationIngressEvent::PointGoal(goal))
            .expect("goal");
        let bytes = log.encode().expect("encode");
        let epoch_payload = &bytes[HEADER_BYTES + RECORD_PAYLOAD_OFFSET..];
        let goal_payload = &bytes[HEADER_BYTES + RECORD_BYTES + RECORD_PAYLOAD_OFFSET..];
        assert_eq!(get_u64(epoch_payload, 8), 1);
        assert_eq!(get_u64(goal_payload, 8), 1);
        assert_ne!(get_u64(goal_payload, 8), process_id);
    }

    #[test]
    fn nonfinite_goal_is_rejected_at_wire_boundary() {
        let snapshot = snapshot(4);
        let mut coordinator = NavigationMapEpochCoordinator::new();
        let transition = transition(&mut coordinator, &snapshot, 800);
        let goal = MapPointGoalIngress::parse(
            clock(),
            host_time(900),
            transition.binding(),
            point_goal(&snapshot),
        )
        .expect("goal");
        let mut log = NavigationIngressLog::new(recording_id(1), capacity(2));
        log.push(NavigationIngressEvent::MapEpochStarted(transition.event()))
            .expect("epoch");
        log.push(NavigationIngressEvent::PointGoal(goal))
            .expect("goal");
        let mut bytes = log.encode().expect("encode");
        let goal_payload = HEADER_BYTES + RECORD_BYTES + RECORD_PAYLOAD_OFFSET;
        put_f64(&mut bytes[goal_payload..], 24, f64::NAN);
        assert!(matches!(
            NavigationIngressLog::parse(Some(&bytes), recording_id(1), capacity(2)),
            Err(NavigationIngressParseError::InvalidGoalPoint {
                record_index: 1,
                ..
            })
        ));
    }

    #[test]
    fn streaming_writer_and_reader_round_trip_constant_memory_path() {
        let events = mixed_events();
        let mut writer =
            NavigationIngressWriter::new(Cursor::new(Vec::new()), recording_id(1), capacity(8))
                .expect("writer");
        for event in events {
            writer.append(event).expect("event");
        }
        assert_eq!(writer.recording_id(), recording_id(1));
        assert_eq!(writer.record_count(), events.len());
        let bytes = writer.finish().expect("finish").into_inner();
        assert_eq!(get_u64(&bytes, 16), events.len() as u64);
        assert_eq!(&bytes[24..40], &recording_id(1).into_bytes());
        assert_eq!(bytes.len(), HEADER_BYTES + events.len() * RECORD_BYTES);

        let reader = NavigationIngressReader::new(Cursor::new(bytes), recording_id(1), capacity(8))
            .expect("reader");
        assert_eq!(reader.recording_id(), recording_id(1));
        assert_eq!(reader.declared_count(), events.len());
        let decoded = reader.read_to_log().expect("decode");
        assert_eq!(
            decoded
                .records()
                .iter()
                .map(|record| record.event())
                .collect::<Vec<_>>(),
            events
        );
    }

    #[test]
    fn streaming_reader_enforces_identity_and_bound_from_header() {
        let mut bytes = NavigationIngressLog::new(recording_id(1), capacity(2))
            .encode()
            .expect("empty");
        put_u64(&mut bytes, 16, 2);
        assert!(matches!(
            NavigationIngressReader::new(Cursor::new(bytes.clone()), recording_id(2), capacity(2),),
            Err(NavigationIngressStreamReadError::Parse(
                NavigationIngressParseError::RecordingIdMismatch { .. }
            ))
        ));
        assert!(matches!(
            NavigationIngressReader::new(Cursor::new(bytes), recording_id(1), capacity(1),),
            Err(NavigationIngressStreamReadError::Parse(
                NavigationIngressParseError::RecordLimitExceeded {
                    declared: 2,
                    limit: 1,
                }
            ))
        ));
    }

    #[test]
    fn streaming_reader_reports_partial_record_exactly() {
        let mut writer =
            NavigationIngressWriter::new(Cursor::new(Vec::new()), recording_id(1), capacity(1))
                .expect("writer");
        writer
            .append(NavigationIngressEvent::ImuReport(recorded_imu()))
            .expect("record");
        let mut bytes = writer.finish().expect("finish").into_inner();
        bytes.truncate(bytes.len() - 7);
        let mut reader =
            NavigationIngressReader::new(Cursor::new(bytes), recording_id(1), capacity(1))
                .expect("reader");
        assert!(matches!(
            reader.next_record(),
            Err(NavigationIngressStreamReadError::TruncatedRecord {
                record_index: 0,
                expected_record_bytes: RECORD_BYTES,
                actual_record_bytes,
            }) if actual_record_bytes == RECORD_BYTES - 7
        ));
        assert!(matches!(
            reader.next_record(),
            Err(NavigationIngressStreamReadError::Failed)
        ));
    }

    #[test]
    fn optional_stream_does_not_invent_legacy_timing() {
        assert!(matches!(
            NavigationIngressReader::<Cursor<Vec<u8>>>::from_optional(
                None,
                recording_id(1),
                capacity(1),
            ),
            Err(NavigationIngressStreamReadError::Parse(
                NavigationIngressParseError::TimingUnavailable
            ))
        ));
    }

    #[test]
    fn writer_rejects_nonempty_sink_and_preexisting_suffix() {
        assert!(matches!(
            NavigationIngressWriter::new(Cursor::new(vec![1, 2, 3]), recording_id(1), capacity(1),),
            Err(NavigationIngressStreamWriteError::SinkHasSuffix {
                current_position: 0,
                end_position: 3,
            })
        ));

        let mut at_end = Cursor::new(vec![1, 2, 3]);
        at_end.set_position(3);
        assert!(matches!(
            NavigationIngressWriter::new(at_end, recording_id(1), capacity(1)),
            Err(NavigationIngressStreamWriteError::SinkNotEmpty { length: 3 })
        ));
    }

    #[test]
    fn writer_finish_rejects_suffix_added_during_recording() {
        let mut writer =
            NavigationIngressWriter::new(Cursor::new(Vec::new()), recording_id(1), capacity(1))
                .expect("writer");
        writer
            .append(NavigationIngressEvent::ImuReport(recorded_imu()))
            .expect("record");
        writer.inner.get_mut().extend_from_slice(&[9, 9]);
        assert!(matches!(
            writer.finish(),
            Err(NavigationIngressStreamWriteError::SinkLengthMismatch {
                expected_end,
                current_position,
                end_position,
            }) if expected_end == (HEADER_BYTES + RECORD_BYTES) as u64
                && current_position == expected_end
                && end_position == expected_end + 2
        ));
    }

    #[test]
    fn partial_record_write_poisons_without_publishing_count() {
        let stream = ByteBudgetStream {
            inner: Cursor::new(Vec::new()),
            remaining_write_bytes: HEADER_BYTES + 3,
        };
        let mut writer =
            NavigationIngressWriter::new(stream, recording_id(1), capacity(2)).expect("header");
        assert!(matches!(
            writer.append(NavigationIngressEvent::ImuReport(recorded_imu())),
            Err(NavigationIngressStreamWriteError::Io {
                stage: NavigationIngressWriteStage::WriteRecord,
                ..
            })
        ));
        assert_eq!(writer.record_count(), 0);
        assert!(matches!(
            writer.append(NavigationIngressEvent::ImuReport(recorded_imu())),
            Err(NavigationIngressStreamWriteError::Poisoned)
        ));
        assert!(matches!(
            writer.finish(),
            Err(NavigationIngressStreamWriteError::Poisoned)
        ));
    }

    #[test]
    fn finish_preserves_patch_and_position_restore_failures() {
        let stream = FailingAbsoluteSeekStream {
            inner: Cursor::new(Vec::new()),
            fail_absolute_seeks: false,
        };
        let mut writer =
            NavigationIngressWriter::new(stream, recording_id(1), capacity(1)).expect("header");
        writer
            .append(NavigationIngressEvent::ImuReport(recorded_imu()))
            .expect("record");
        writer.inner.fail_absolute_seeks = true;
        assert!(matches!(
            writer.finish(),
            Err(NavigationIngressStreamWriteError::FinalizeAndRestore { .. })
        ));
    }
}
