//! Transport-free host shadow-navigation coordination.
//!
//! This state machine is the sole owner of navigation admission order.  Its
//! callers may schedule typed inputs on any threads they choose, but the order
//! in which those inputs reach this object is the only order recorded here.
//! No deterministic cross-thread scheduling claim is made.

use std::fmt;
use std::io::{Seek, Write};
use std::num::NonZeroU64;

use crate::dense::occupancy::OccupancyGridSnapshot;
use crate::map::MapInstanceId;
use crate::{
    DeviceSessionId, DeviceTimestamp, HostMonotonicTimestamp, ImuReport, MapLocalization,
    Timestamp, VisualFrameStamp, VisualIncrement,
};

use super::frames::{BaseFrame, PlanarPoint, PlanarTransformError};
use super::frontier::FrontierGoal;
use super::global_planner::{
    GlobalPath, GlobalPlanError, GlobalPlanIdentity, GlobalPlanner, GlobalPlannerConfig, MapPoint,
    MapTraversalBoundary, PlanStart, PointGoal,
};
use super::goal_input::MapPointGoalSelection;
use super::ingress::{
    AcceptedDepthIngress, AcceptedGlobalMapIngress, ControlTickIngress, CurrentMapEpochBinding,
    MapPointGoalIngress, NavigationClockEpoch, NavigationIngressBoundaryError,
    NavigationIngressEvent, NavigationIngressLog, NavigationIngressRecord,
    NavigationIngressStreamWriteError, NavigationIngressWriteError, NavigationIngressWriter,
    NavigationMapEpochCoordinator, NavigationReplayClock, NavigationReplayClockError,
    RecordedImuReport, RecordedMapEpochId, VisualAttemptIngress, VisualAttemptOutcome,
};
use super::local_costmap::{
    DepthFrameKey, LocalCostmap, LocalCostmapClockRegression, LocalCostmapError,
    LocalCostmapUpdateOutcome, LocalDepthObservation, LocalDepthObservationError,
};
use super::manual_drive::{ManualDriveAcceptedStop, ManualDriveStopCause, ManualDriveStopped};
use super::manual_reference::{
    FrontierYawReferenceBuildError, FrontierYawReferenceBuilderV1, FrontierYawScanCommandV1,
    ManualMpcCommandV1, ManualReferenceBuildError, ManualReferenceBuilderV1,
    NumericAuthorityLeaseId,
};
use super::mpc::{
    HostMonotonicClock, MotionValueError, MpcConfigV1, NavigationEpochError, NavigationEpochV1,
    ODOM_MOTION_STATE_V1, OdomMotionStateV1, OdomMotionStateV1Dto, OdomPoseV1,
};
use super::odometry::{
    ImuUpdate, OdomSegmentId, OdometryError, OdometryEstimate, OdometryState, OdometryUnavailable,
    PlanarOdometry, PoseHistoryQuery,
};
use super::reference::{PathReferenceBuildError, PathReferenceBuilderV1};
use super::safety::{
    SafetyDecideError, SafetyDecision, SafetyNotReadyReason, SafetyReadyTick, SafetyTickInput,
    ShadowSafetySupervisor, SolverBudgetNs,
};

/// Append-only event sink used by live stream recording and bounded tests.
///
/// Implementations must either admit one complete record or return an error.
/// The coordinator permanently latches closed after any error because a
/// partially written stream cannot truthfully describe later decisions.
pub trait NavigationIngressSink {
    type Error;

    fn append_event(
        &mut self,
        event: NavigationIngressEvent,
    ) -> Result<NavigationIngressRecord, Self::Error>;
}

impl NavigationIngressSink for NavigationIngressLog {
    type Error = NavigationIngressWriteError;

    fn append_event(
        &mut self,
        event: NavigationIngressEvent,
    ) -> Result<NavigationIngressRecord, Self::Error> {
        self.push(event)
    }
}

impl<W: Write + Seek> NavigationIngressSink for NavigationIngressWriter<W> {
    type Error = NavigationIngressStreamWriteError;

    fn append_event(
        &mut self,
        event: NavigationIngressEvent,
    ) -> Result<NavigationIngressRecord, Self::Error> {
        self.append(event)
    }
}

/// Tracker result attached to the exact completed visual-attempt identity.
///
/// Construction checks the redundant wire outcome once.  The coordinator
/// therefore never accepts an `IncrementAndLocalization` record carrying a
/// failure payload, or vice versa.
#[derive(Clone, Copy, Debug)]
pub struct VisualAdmission {
    ingress: VisualAttemptIngress,
    payload: VisualAdmissionPayload,
}

#[derive(Clone, Copy, Debug)]
enum VisualAdmissionPayload {
    IncrementAndLocalization {
        increment: VisualIncrement,
        localization: MapLocalization,
    },
    LocalizationOnly {
        localization: MapLocalization,
    },
    NoLocalization,
    RecoverableFailure,
    FatalFailure,
}

impl VisualAdmission {
    pub fn increment_and_localization(
        ingress: VisualAttemptIngress,
        increment: VisualIncrement,
        localization: MapLocalization,
    ) -> Result<Self, VisualAdmissionError> {
        require_visual_outcome(ingress, VisualAttemptOutcome::IncrementAndLocalization)?;
        require_localization_identity(ingress, localization.visual_stamp())?;
        if increment.to() != localization.visual_stamp() {
            return Err(VisualAdmissionError::IncrementLocalizationMismatch {
                increment_to: increment.to(),
                localization: localization.visual_stamp(),
            });
        }
        Ok(Self {
            ingress,
            payload: VisualAdmissionPayload::IncrementAndLocalization {
                increment,
                localization,
            },
        })
    }

    pub fn localization_only(
        ingress: VisualAttemptIngress,
        localization: MapLocalization,
    ) -> Result<Self, VisualAdmissionError> {
        require_visual_outcome(ingress, VisualAttemptOutcome::LocalizationOnly)?;
        require_localization_identity(ingress, localization.visual_stamp())?;
        Ok(Self {
            ingress,
            payload: VisualAdmissionPayload::LocalizationOnly { localization },
        })
    }

    pub fn no_localization(ingress: VisualAttemptIngress) -> Result<Self, VisualAdmissionError> {
        require_visual_outcome(ingress, VisualAttemptOutcome::NoLocalization)?;
        Ok(Self {
            ingress,
            payload: VisualAdmissionPayload::NoLocalization,
        })
    }

    pub fn recoverable_failure(
        ingress: VisualAttemptIngress,
    ) -> Result<Self, VisualAdmissionError> {
        require_visual_outcome(ingress, VisualAttemptOutcome::RecoverableFailure)?;
        Ok(Self {
            ingress,
            payload: VisualAdmissionPayload::RecoverableFailure,
        })
    }

    pub fn fatal_failure(ingress: VisualAttemptIngress) -> Result<Self, VisualAdmissionError> {
        require_visual_outcome(ingress, VisualAttemptOutcome::FatalFailure)?;
        Ok(Self {
            ingress,
            payload: VisualAdmissionPayload::FatalFailure,
        })
    }

    pub fn ingress(self) -> VisualAttemptIngress {
        self.ingress
    }
}

fn require_visual_outcome(
    ingress: VisualAttemptIngress,
    expected: VisualAttemptOutcome,
) -> Result<(), VisualAdmissionError> {
    let actual = ingress.outcome();
    if actual == expected {
        Ok(())
    } else {
        Err(VisualAdmissionError::OutcomeMismatch { expected, actual })
    }
}

fn require_localization_identity(
    ingress: VisualAttemptIngress,
    localization: VisualFrameStamp,
) -> Result<(), VisualAdmissionError> {
    let timestamp_matches = u64::try_from(localization.timestamp().as_nanos())
        .is_ok_and(|timestamp| timestamp == ingress.left_timestamp().as_nanos());
    if localization.frame_id() == ingress.left_frame_id() && timestamp_matches {
        Ok(())
    } else {
        Err(VisualAdmissionError::LocalizationIdentityMismatch {
            ingress_frame_id: ingress.left_frame_id(),
            ingress_timestamp: ingress.left_timestamp(),
            localization,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VisualAdmissionError {
    OutcomeMismatch {
        expected: VisualAttemptOutcome,
        actual: VisualAttemptOutcome,
    },
    LocalizationIdentityMismatch {
        ingress_frame_id: crate::FrameId,
        ingress_timestamp: DeviceTimestamp,
        localization: VisualFrameStamp,
    },
    IncrementLocalizationMismatch {
        increment_to: VisualFrameStamp,
        localization: VisualFrameStamp,
    },
}

impl fmt::Display for VisualAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid typed visual admission: {self:?}")
    }
}

impl std::error::Error for VisualAdmissionError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavigationGoalState {
    Unavailable,
    PendingFirstMap,
    Bound {
        map_instance_id: MapInstanceId,
        selected_revision: u64,
    },
    InvalidatedByMapReset {
        previous_map_instance_id: MapInstanceId,
        replacement_map_instance_id: MapInstanceId,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoordinatorLatch {
    JournalFailure,
    ClockBoundary,
    MapEpochIdExhausted,
    GoalGenerationExhausted,
    MotionModeGenerationExhausted,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct BoundPointGoal {
    point_goal: PointGoal,
    traversal_boundary: Option<MapTraversalBoundary>,
}

impl BoundPointGoal {
    const fn unbounded(point_goal: PointGoal) -> Self {
        Self {
            point_goal,
            traversal_boundary: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum GoalBinding {
    Unavailable,
    Pending(MapPoint),
    Bound(BoundPointGoal),
    Invalidated {
        previous_map_instance_id: MapInstanceId,
        replacement_map_instance_id: MapInstanceId,
    },
}

/// Non-mutating proof that one map-point goal was valid for an exact immutable
/// displayed snapshot and the coordinator state observed during preparation.
///
/// The snapshot borrow is retained rather than reducing it to a revision
/// number. A caller can therefore perform a separately evidenced authority
/// handover and fresh-zero barrier between preparation and commit without
/// being able to substitute different occupancy content at commit. The proof
/// is deliberately neither `Copy` nor `Clone`; commit consumes it.
pub struct PreparedMapPointGoal<'snapshot> {
    displayed_snapshot: &'snapshot OccupancyGridSnapshot,
    current_map: CurrentOccupancyMap,
    goal_generation: u64,
    motion_mode_generation: u64,
    previous_goal: GoalBinding,
    previous_motion_mode: CoordinatorMotionModeV1,
    goal: PointGoal,
    traversal_boundary: Option<MapTraversalBoundary>,
}

impl fmt::Debug for PreparedMapPointGoal<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedMapPointGoal")
            .field("map_binding", &self.current_map.binding)
            .field("displayed_revision", &self.current_map.revision)
            .field("goal_generation", &self.goal_generation)
            .field("motion_mode_generation", &self.motion_mode_generation)
            .field("previous_goal", &self.previous_goal)
            .field("previous_motion_mode", &self.previous_motion_mode)
            .field("goal", &self.goal)
            .field("traversal_boundary", &self.traversal_boundary)
            .finish_non_exhaustive()
    }
}

impl PreparedMapPointGoal<'_> {
    pub fn map_binding(&self) -> CurrentMapEpochBinding {
        self.current_map.binding
    }

    pub fn displayed_revision(&self) -> u64 {
        self.current_map.revision
    }

    pub fn goal(&self) -> PointGoal {
        self.goal
    }

    pub fn traversal_boundary(&self) -> Option<MapTraversalBoundary> {
        self.traversal_boundary
    }
}

/// Mutually exclusive reference mode owned by this coordinator.
///
/// Entering a direct-control mode is only possible from `MappingOnly`, after
/// the caller has removed any point goal and fulfilled the supervisor's
/// separate fresh-zero handover obligation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoordinatorMotionModeV1 {
    MappingOnly,
    PointGoal,
    Manual { authority_lease_id: NonZeroU64 },
    FrontierInPlaceYaw { authority_lease_id: NonZeroU64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoordinatorMotionModeError {
    NotMappingOnly {
        actual: CoordinatorMotionModeV1,
    },
    NotDirectControl {
        actual: CoordinatorMotionModeV1,
    },
    AuthorityLeaseMismatch {
        bound: NonZeroU64,
        supplied: NonZeroU64,
    },
    GenerationExhausted,
}

impl fmt::Display for CoordinatorMotionModeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid coordinator motion-mode transition: {self:?}"
        )
    }
}

impl std::error::Error for CoordinatorMotionModeError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CurrentOccupancyMap {
    binding: CurrentMapEpochBinding,
    revision: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DirectLocalizationBinding {
    device_session_id: DeviceSessionId,
    odom_segment_id: OdomSegmentId,
    map_instance_id: MapInstanceId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VisualReadiness {
    AwaitingAnchor,
    Continuous,
    BrokenAttempt,
    RejectedUpdate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DepthReadiness {
    NoObservation,
    Current(DepthFrameKey),
    Unaligned {
        frame_id: crate::FrameId,
        timestamp: DeviceTimestamp,
        reason: OdometryUnavailable,
    },
    Rejected {
        frame_id: crate::FrameId,
        timestamp: DeviceTimestamp,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoredPlanFault {
    OdometryUnavailable(OdometryUnavailable),
    OdomMapMismatch,
    StartTransform,
    PlannerConstruction,
    Planning,
}

#[derive(Debug)]
pub enum PlanStartBuildError {
    OdometryUnavailable,
    OdomMapMismatch {
        odometry_map_instance_id: MapInstanceId,
        snapshot_map_instance_id: Option<MapInstanceId>,
    },
    Transform(PlanarTransformError),
    Plan(GlobalPlanError),
}

impl fmt::Display for PlanStartBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "cannot bind the current robot pose to a map snapshot: {self:?}"
        )
    }
}

impl std::error::Error for PlanStartBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Transform(source) => Some(source),
            Self::Plan(source) => Some(source),
            Self::OdometryUnavailable | Self::OdomMapMismatch { .. } => None,
        }
    }
}

/// Result of accepting one tracker attempt after its journal record committed.
#[derive(Clone, Debug)]
pub enum VisualAdmissionOutcome {
    Reanchored(OdometryState),
    Updated(OdometryState),
    ChainBroken(VisualAttemptOutcome),
    Rejected(OdometryError),
}

/// Result of accepting one inertial report after its journal record committed.
#[derive(Clone, Debug)]
pub enum ImuAdmissionOutcome {
    Updated(ImuUpdate),
    Rejected(OdometryError),
}

/// Result of accepting one depth frame after its journal record committed.
#[derive(Clone, Debug)]
pub enum DepthAdmissionOutcome {
    Updated(LocalCostmapUpdateOutcome),
    PoseUnavailable(OdometryUnavailable),
    OdometryRejected(OdometryError),
    AlignmentRejected(LocalDepthObservationError),
    CostmapRejected(LocalCostmapError),
}

#[derive(Debug)]
pub enum GlobalPlanningOutcome {
    Planned(GlobalPlanIdentity),
    Deferred(StoredPlanFault),
    Failed(GlobalPlanError),
}

#[derive(Debug)]
pub struct GlobalMapAdmissionOutcome {
    map_instance_id: MapInstanceId,
    revision: u64,
    started_new_epoch: bool,
    goal_state: NavigationGoalState,
    planning: GlobalPlanningOutcome,
}

#[derive(Debug)]
pub struct GoalSelectionOutcome {
    goal_state: NavigationGoalState,
    planning: GlobalPlanningOutcome,
}

impl GoalSelectionOutcome {
    pub fn goal_state(&self) -> NavigationGoalState {
        self.goal_state
    }

    pub fn planning(&self) -> &GlobalPlanningOutcome {
        &self.planning
    }
}

impl GlobalMapAdmissionOutcome {
    pub fn map_instance_id(&self) -> MapInstanceId {
        self.map_instance_id
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn started_new_epoch(&self) -> bool {
        self.started_new_epoch
    }

    pub fn goal_state(&self) -> NavigationGoalState {
        self.goal_state
    }

    pub fn planning(&self) -> &GlobalPlanningOutcome {
        &self.planning
    }
}

#[derive(Debug)]
pub enum CoordinatorAdmissionError<E> {
    Latched(CoordinatorLatch),
    Boundary(NavigationIngressBoundaryError),
    Journal(E),
    ReplayClock(NavigationReplayClockError),
    SegmentIdExhausted,
    Plan(GlobalPlanError),
    MapRevisionNotIncreasing {
        previous: u64,
        actual: u64,
    },
    NoCurrentMapForGoal,
    GoalMapEpochMismatch {
        displayed: RecordedMapEpochId,
        current: RecordedMapEpochId,
    },
    GoalDisplayedRevisionMismatch {
        displayed: u64,
        current: u64,
    },
    GoalSnapshotMapMismatch {
        expected: MapInstanceId,
        actual: Option<MapInstanceId>,
    },
    GoalSnapshotRevisionMismatch {
        displayed: u64,
        snapshot: u64,
    },
    FrontierGoalMapMismatch {
        expected: MapInstanceId,
        actual: MapInstanceId,
    },
    FrontierGoalRevisionMismatch {
        expected: u64,
        actual: u64,
    },
    FrontierGoalMissingTraversalBoundary,
    PointGoalPreparationStale {
        prepared_map_binding: CurrentMapEpochBinding,
        prepared_revision: u64,
        current_map_binding: Option<CurrentMapEpochBinding>,
        current_revision: Option<u64>,
        prepared_motion_mode: CoordinatorMotionModeV1,
        current_motion_mode: CoordinatorMotionModeV1,
        prepared_goal_generation: u64,
        current_goal_generation: u64,
        prepared_motion_mode_generation: u64,
        current_motion_mode_generation: u64,
    },
    PointGoalModeConflict {
        actual: CoordinatorMotionModeV1,
    },
}

impl<E: fmt::Debug> fmt::Display for CoordinatorAdmissionError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "shadow-navigation admission failed: {self:?}")
    }
}

impl<E: std::error::Error + 'static> std::error::Error for CoordinatorAdmissionError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Boundary(source) => Some(source),
            Self::Journal(source) => Some(source),
            Self::ReplayClock(source) => Some(source),
            Self::Plan(source) => Some(source),
            Self::Latched(_)
            | Self::SegmentIdExhausted
            | Self::MapRevisionNotIncreasing { .. }
            | Self::NoCurrentMapForGoal
            | Self::GoalMapEpochMismatch { .. }
            | Self::GoalDisplayedRevisionMismatch { .. }
            | Self::GoalSnapshotMapMismatch { .. }
            | Self::GoalSnapshotRevisionMismatch { .. }
            | Self::FrontierGoalMapMismatch { .. }
            | Self::FrontierGoalRevisionMismatch { .. }
            | Self::FrontierGoalMissingTraversalBoundary
            | Self::PointGoalPreparationStale { .. }
            | Self::PointGoalModeConflict { .. } => None,
        }
    }
}

/// Rich coordinator blocker retained alongside the safety layer's compact
/// semantic stop category.
#[derive(Debug)]
pub enum CoordinatorTickBlocker {
    JournalLatched,
    ControlTickBoundary(NavigationIngressBoundaryError),
    VisualOdometryUnavailable,
    VisualOdometryRejected,
    GoalUnavailable,
    GoalInvalidated,
    PathUnavailable(Option<StoredPlanFault>),
    PathInvalidated,
    DepthUnavailable,
    DepthUnaligned,
    LocalCostmapClock(LocalCostmapClockRegression),
    LocalCostmapExpired,
    OdometryUnavailable(OdometryUnavailable),
    Odometry(OdometryError),
    NavigationEpoch(NavigationEpochError),
    MotionState(MotionValueError),
    MapToOdom(PlanarTransformError),
    Reference(PathReferenceBuildError),
    MotionModeMismatch {
        expected: CoordinatorMotionModeV1,
        actual: CoordinatorMotionModeV1,
    },
    DirectAuthorityLeaseMismatch {
        bound: NonZeroU64,
        command: NonZeroU64,
    },
    ManualCommandSequenceRegression {
        previous: u64,
        command: u64,
    },
    ManualCommandIdentityConflict {
        sequence: u64,
    },
    ManualExplicitStop {
        authority_lease_id: NonZeroU64,
        sequence: u64,
    },
    ManualDriveStopped {
        authority_lease_id: NonZeroU64,
        cause: ManualDriveStopCause<NonZeroU64>,
    },
    FrontierCommandSequenceRegression {
        previous: u64,
        command: u64,
    },
    FrontierCommandIdentityConflict {
        sequence: u64,
    },
    DirectLocalizationEpochTransition,
    ManualReference(ManualReferenceBuildError),
    FrontierYawReference(FrontierYawReferenceBuildError),
    FrontierMapRevisionMismatch {
        expected_map_instance_id: MapInstanceId,
        expected_revision: u64,
        actual_map_instance_id: MapInstanceId,
        actual_revision: u64,
    },
}

impl CoordinatorTickBlocker {
    fn safety_reason(&self) -> SafetyNotReadyReason {
        match self {
            Self::JournalLatched | Self::ControlTickBoundary(_) => {
                SafetyNotReadyReason::SafetyJournalLatched
            }
            Self::VisualOdometryUnavailable => SafetyNotReadyReason::VisualOdometryUnavailable,
            Self::VisualOdometryRejected => SafetyNotReadyReason::VisualOdometryRejected,
            Self::GoalUnavailable => SafetyNotReadyReason::NavigationGoalUnavailable,
            Self::GoalInvalidated => SafetyNotReadyReason::NavigationGoalInvalidated,
            Self::PathUnavailable(_) => SafetyNotReadyReason::GlobalPathUnavailable,
            Self::PathInvalidated => SafetyNotReadyReason::GlobalPathInvalidated,
            Self::DepthUnavailable => SafetyNotReadyReason::DepthObservationUnavailable,
            Self::DepthUnaligned => SafetyNotReadyReason::DepthObservationUnaligned,
            Self::LocalCostmapClock(_) => SafetyNotReadyReason::LocalCostmapUnavailable,
            Self::LocalCostmapExpired => SafetyNotReadyReason::LocalCostmapExpired,
            Self::OdometryUnavailable(_) => SafetyNotReadyReason::VisualOdometryUnavailable,
            Self::Odometry(_) => SafetyNotReadyReason::VisualOdometryRejected,
            Self::NavigationEpoch(_) => SafetyNotReadyReason::NavigationEpochUnavailable,
            Self::MotionState(_) | Self::MapToOdom(_) => {
                SafetyNotReadyReason::MotionStateUnavailable
            }
            Self::Reference(_) | Self::ManualReference(_) | Self::FrontierYawReference(_) => {
                SafetyNotReadyReason::ReferenceUnavailable
            }
            Self::ManualExplicitStop { .. } | Self::ManualDriveStopped { .. } => {
                SafetyNotReadyReason::ManualDriveStopped
            }
            Self::MotionModeMismatch { .. }
            | Self::DirectAuthorityLeaseMismatch { .. }
            | Self::ManualCommandSequenceRegression { .. }
            | Self::ManualCommandIdentityConflict { .. }
            | Self::FrontierCommandSequenceRegression { .. }
            | Self::FrontierCommandIdentityConflict { .. }
            | Self::DirectLocalizationEpochTransition
            | Self::FrontierMapRevisionMismatch { .. } => {
                SafetyNotReadyReason::NavigationEpochTransition
            }
        }
    }
}

/// One periodic tick always contains the result of one safety-supervisor
/// decision unless command evidence itself could not be recorded.
pub struct CoordinatorTickOutcome<E> {
    decision: SafetyDecision,
    blocker: Option<CoordinatorTickBlocker>,
    control_tick_journaled: bool,
    journal_error: Option<E>,
}

impl<E> CoordinatorTickOutcome<E> {
    pub fn decision(&self) -> &SafetyDecision {
        &self.decision
    }

    pub fn blocker(&self) -> Option<&CoordinatorTickBlocker> {
        self.blocker.as_ref()
    }

    pub fn control_tick_journaled(&self) -> bool {
        self.control_tick_journaled
    }

    pub fn journal_error(&self) -> Option<&E> {
        self.journal_error.as_ref()
    }
}

#[derive(Debug)]
pub enum CoordinatorTickError {
    Safety(SafetyDecideError),
}

impl fmt::Display for CoordinatorTickError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "shadow-navigation safety evidence failed: {self:?}"
        )
    }
}

impl std::error::Error for CoordinatorTickError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Safety(source) => Some(source.as_ref()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ManualCommandEvidence {
    Velocity(ManualMpcCommandV1),
    ExplicitStop {
        authority_lease_id: NonZeroU64,
        sequence: u64,
    },
}

impl ManualCommandEvidence {
    fn sequence(self) -> u64 {
        match self {
            Self::Velocity(command) => command.sequence().get(),
            Self::ExplicitStop { sequence, .. } => sequence,
        }
    }
}

fn numeric_authority_lease<LeaseId: NumericAuthorityLeaseId>(lease: LeaseId) -> NonZeroU64 {
    NonZeroU64::new(lease.get()).expect("numeric authority lease IDs are sealed nonzero domains")
}

/// Pure host navigation owner.  It contains no transport handle, packet
/// encoder, callback, Rerun dependency, or encoder-derived state.
pub struct ShadowNavigationCoordinator<J: NavigationIngressSink> {
    clock_epoch: NavigationClockEpoch,
    journal: J,
    map_epochs: NavigationMapEpochCoordinator,
    current_map: Option<CurrentOccupancyMap>,
    goal: GoalBinding,
    goal_generation: u64,
    motion_mode_generation: u64,
    odometry: PlanarOdometry,
    next_segment_id: Option<NonZeroU64>,
    latest_device_time: Option<(DeviceSessionId, DeviceTimestamp)>,
    visual_readiness: VisualReadiness,
    local_costmap: LocalCostmap,
    depth_readiness: DepthReadiness,
    planner_config: GlobalPlannerConfig,
    path: Option<GlobalPath>,
    plan_fault: Option<StoredPlanFault>,
    reference_builder: PathReferenceBuilderV1,
    manual_reference_builder: ManualReferenceBuilderV1,
    frontier_yaw_reference_builder: FrontierYawReferenceBuilderV1,
    motion_mode: CoordinatorMotionModeV1,
    direct_localization_binding: Option<DirectLocalizationBinding>,
    last_manual_command: Option<ManualCommandEvidence>,
    last_frontier_yaw_command: Option<FrontierYawScanCommandV1>,
    mpc_config: MpcConfigV1,
    solver_budget: SolverBudgetNs,
    safety: ShadowSafetySupervisor,
    latch: Option<CoordinatorLatch>,
}

impl<J: NavigationIngressSink> ShadowNavigationCoordinator<J> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        clock_epoch: NavigationClockEpoch,
        journal: J,
        pending_goal: MapPoint,
        odometry: PlanarOdometry,
        local_costmap: LocalCostmap,
        planner_config: GlobalPlannerConfig,
        reference_builder: PathReferenceBuilderV1,
        mpc_config: MpcConfigV1,
        solver_budget: SolverBudgetNs,
        safety: ShadowSafetySupervisor,
    ) -> Self {
        Self::new_with_optional_goal(
            clock_epoch,
            journal,
            Some(pending_goal),
            odometry,
            local_costmap,
            planner_config,
            reference_builder,
            mpc_config,
            solver_budget,
            safety,
        )
    }

    /// Construct a mapping-only coordinator. Every control tick remains a
    /// typed stop until an exact map-revision-bound point is selected.
    #[allow(clippy::too_many_arguments)]
    pub fn new_without_goal(
        clock_epoch: NavigationClockEpoch,
        journal: J,
        odometry: PlanarOdometry,
        local_costmap: LocalCostmap,
        planner_config: GlobalPlannerConfig,
        reference_builder: PathReferenceBuilderV1,
        mpc_config: MpcConfigV1,
        solver_budget: SolverBudgetNs,
        safety: ShadowSafetySupervisor,
    ) -> Self {
        Self::new_with_optional_goal(
            clock_epoch,
            journal,
            None,
            odometry,
            local_costmap,
            planner_config,
            reference_builder,
            mpc_config,
            solver_budget,
            safety,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_optional_goal(
        clock_epoch: NavigationClockEpoch,
        journal: J,
        pending_goal: Option<MapPoint>,
        odometry: PlanarOdometry,
        local_costmap: LocalCostmap,
        planner_config: GlobalPlannerConfig,
        reference_builder: PathReferenceBuilderV1,
        mpc_config: MpcConfigV1,
        solver_budget: SolverBudgetNs,
        safety: ShadowSafetySupervisor,
    ) -> Self {
        let motion_mode = if pending_goal.is_some() {
            CoordinatorMotionModeV1::PointGoal
        } else {
            CoordinatorMotionModeV1::MappingOnly
        };
        Self {
            clock_epoch,
            journal,
            map_epochs: NavigationMapEpochCoordinator::new(),
            current_map: None,
            goal: pending_goal.map_or(GoalBinding::Unavailable, GoalBinding::Pending),
            goal_generation: 0,
            motion_mode_generation: 0,
            odometry,
            next_segment_id: NonZeroU64::new(1),
            latest_device_time: None,
            visual_readiness: VisualReadiness::AwaitingAnchor,
            local_costmap,
            depth_readiness: DepthReadiness::NoObservation,
            planner_config,
            path: None,
            plan_fault: None,
            reference_builder,
            manual_reference_builder: ManualReferenceBuilderV1,
            frontier_yaw_reference_builder: FrontierYawReferenceBuilderV1,
            motion_mode,
            direct_localization_binding: None,
            last_manual_command: None,
            last_frontier_yaw_command: None,
            mpc_config,
            solver_budget,
            safety,
            latch: None,
        }
    }

    pub fn journal(&self) -> &J {
        &self.journal
    }

    pub fn journal_mut(&mut self) -> &mut J {
        &mut self.journal
    }

    /// Consume the state machine so a live owner can finalize its streaming
    /// journal after all admitted ticks have drained.
    pub fn into_journal(self) -> J {
        self.journal
    }

    /// Consume the state machine while retaining both independently useful
    /// evidence owners.
    pub fn into_parts(self) -> (J, ShadowSafetySupervisor) {
        (self.journal, self.safety)
    }

    pub fn odometry(&self) -> &PlanarOdometry {
        &self.odometry
    }

    pub fn local_costmap(&self) -> &LocalCostmap {
        &self.local_costmap
    }

    pub fn global_path(&self) -> Option<&GlobalPath> {
        self.path.as_ref()
    }

    pub fn safety(&self) -> &ShadowSafetySupervisor {
        &self.safety
    }

    pub fn latch(&self) -> Option<CoordinatorLatch> {
        self.latch
    }

    pub fn goal_state(&self) -> NavigationGoalState {
        match self.goal {
            GoalBinding::Unavailable => NavigationGoalState::Unavailable,
            GoalBinding::Pending(_) => NavigationGoalState::PendingFirstMap,
            GoalBinding::Bound(goal) => NavigationGoalState::Bound {
                map_instance_id: goal.point_goal.map_instance_id(),
                selected_revision: goal.point_goal.selected_revision(),
            },
            GoalBinding::Invalidated {
                previous_map_instance_id,
                replacement_map_instance_id,
            } => NavigationGoalState::InvalidatedByMapReset {
                previous_map_instance_id,
                replacement_map_instance_id,
            },
        }
    }

    /// The exact currently bound point goal, if one exists.
    ///
    /// Pending CLI input and goals invalidated by a map reset are deliberately
    /// not exposed as actionable map goals.
    pub fn current_goal(&self) -> Option<PointGoal> {
        match self.goal {
            GoalBinding::Bound(goal) => Some(goal.point_goal),
            GoalBinding::Unavailable
            | GoalBinding::Pending(_)
            | GoalBinding::Invalidated { .. } => None,
        }
    }

    pub fn motion_mode(&self) -> CoordinatorMotionModeV1 {
        self.motion_mode
    }

    #[cfg(all(test, feature = "agent-runtime", feature = "actuation"))]
    pub(crate) fn exhaust_motion_mode_generation_for_test(&mut self) {
        self.motion_mode_generation = u64::MAX;
    }

    /// Enter manual reference mode after a separately evidenced zero handover.
    #[allow(
        dead_code,
        reason = "crate-private until the reviewed supervised live owner is wired"
    )]
    pub(crate) fn enter_manual_mode(
        &mut self,
        authority_lease_id: NonZeroU64,
    ) -> Result<(), CoordinatorMotionModeError> {
        if self.motion_mode != CoordinatorMotionModeV1::MappingOnly {
            return Err(CoordinatorMotionModeError::NotMappingOnly {
                actual: self.motion_mode,
            });
        }
        let next_motion_mode_generation = self
            .reserve_motion_mode_generation()
            .map_err(|_| CoordinatorMotionModeError::GenerationExhausted)?;
        self.motion_mode = CoordinatorMotionModeV1::Manual { authority_lease_id };
        self.motion_mode_generation = next_motion_mode_generation;
        self.direct_localization_binding = None;
        self.last_manual_command = None;
        self.last_frontier_yaw_command = None;
        Ok(())
    }

    /// Enter frontier-yaw reference mode after a separately evidenced zero
    /// handover. This is not a point goal and never creates a `GlobalPath`.
    #[allow(
        dead_code,
        reason = "crate-private until the reviewed supervised live owner is wired"
    )]
    pub(crate) fn enter_frontier_yaw_mode(
        &mut self,
        authority_lease_id: NonZeroU64,
    ) -> Result<(), CoordinatorMotionModeError> {
        if self.motion_mode != CoordinatorMotionModeV1::MappingOnly {
            return Err(CoordinatorMotionModeError::NotMappingOnly {
                actual: self.motion_mode,
            });
        }
        let next_motion_mode_generation = self
            .reserve_motion_mode_generation()
            .map_err(|_| CoordinatorMotionModeError::GenerationExhausted)?;
        self.motion_mode = CoordinatorMotionModeV1::FrontierInPlaceYaw { authority_lease_id };
        self.motion_mode_generation = next_motion_mode_generation;
        self.direct_localization_binding = None;
        self.last_manual_command = None;
        self.last_frontier_yaw_command = None;
        Ok(())
    }

    /// Leave a direct-control mode after checking the exact authority lease.
    ///
    /// This only changes reference ownership. It does not claim that zero was
    /// applied; the owner must make and retain an immediate stopped tick and
    /// satisfy the supervisor's independent fresh-zero obligation.
    #[allow(
        dead_code,
        reason = "crate-private until the reviewed supervised live owner is wired"
    )]
    pub(crate) fn leave_direct_mode(
        &mut self,
        authority_lease_id: NonZeroU64,
    ) -> Result<(), CoordinatorMotionModeError> {
        let bound = match self.motion_mode {
            CoordinatorMotionModeV1::Manual { authority_lease_id }
            | CoordinatorMotionModeV1::FrontierInPlaceYaw { authority_lease_id } => {
                authority_lease_id
            }
            actual => return Err(CoordinatorMotionModeError::NotDirectControl { actual }),
        };
        if bound != authority_lease_id {
            return Err(CoordinatorMotionModeError::AuthorityLeaseMismatch {
                bound,
                supplied: authority_lease_id,
            });
        }
        let next_motion_mode_generation = self
            .reserve_motion_mode_generation()
            .map_err(|_| CoordinatorMotionModeError::GenerationExhausted)?;
        self.motion_mode = CoordinatorMotionModeV1::MappingOnly;
        self.motion_mode_generation = next_motion_mode_generation;
        self.direct_localization_binding = None;
        self.last_manual_command = None;
        self.last_frontier_yaw_command = None;
        Ok(())
    }

    /// Remove every actionable point goal and path.
    ///
    /// A direct-control mode and its localization binding are deliberately
    /// preserved: this public goal operation cannot bypass the exact-lease
    /// check in `leave_direct_mode` or pretend a fresh hardware zero
    /// was applied. Proof-bearing in-crate supervisor actions own that
    /// handover. Mapping and point-goal modes return to mapping-only.
    pub fn clear_goal(&mut self) {
        let next_goal_generation = self.reserve_goal_generation().ok();
        let returns_to_mapping = !matches!(
            self.motion_mode,
            CoordinatorMotionModeV1::MappingOnly
                | CoordinatorMotionModeV1::Manual { .. }
                | CoordinatorMotionModeV1::FrontierInPlaceYaw { .. }
        );
        let next_motion_mode_generation = returns_to_mapping
            .then(|| self.reserve_motion_mode_generation().ok())
            .flatten();
        self.goal = GoalBinding::Unavailable;
        if let Some(next_goal_generation) = next_goal_generation {
            self.goal_generation = next_goal_generation;
        }
        self.path = None;
        self.plan_fault = None;
        if !matches!(
            self.motion_mode,
            CoordinatorMotionModeV1::Manual { .. }
                | CoordinatorMotionModeV1::FrontierInPlaceYaw { .. }
        ) {
            self.motion_mode = CoordinatorMotionModeV1::MappingOnly;
            if let Some(next_motion_mode_generation) = next_motion_mode_generation {
                self.motion_mode_generation = next_motion_mode_generation;
            }
            self.direct_localization_binding = None;
            self.last_manual_command = None;
            self.last_frontier_yaw_command = None;
        }
    }

    pub fn current_map_binding(&self) -> Option<CurrentMapEpochBinding> {
        self.current_map.map(|current| current.binding)
    }

    /// Evaluate the exact sensor-side prerequisites for granting new motion
    /// authority at `now`.
    ///
    /// This is deliberately stricter than checking whether a stream has ever
    /// produced a sample. It applies the configured odometry prediction/host
    /// age bounds and the configured local-depth age/alignment bound without
    /// mutating coordinator state. Goal, path, mode, supervisor, and
    /// controller checks remain separate typed gates owned by their existing
    /// layers.
    pub fn motion_start_readiness_at(
        &self,
        now: HostMonotonicTimestamp,
    ) -> Result<(), CoordinatorTickBlocker> {
        if self.latch.is_some() {
            return Err(CoordinatorTickBlocker::JournalLatched);
        }
        if let Some(blocker) = self.direct_preflight_blocker() {
            return Err(blocker);
        }
        self.estimate_state_at(now)?;
        let expected_depth = match self.depth_readiness {
            DepthReadiness::Current(frame) => frame,
            DepthReadiness::NoObservation => {
                return Err(CoordinatorTickBlocker::DepthUnavailable);
            }
            DepthReadiness::Unaligned { .. } | DepthReadiness::Rejected { .. } => {
                return Err(CoordinatorTickBlocker::DepthUnaligned);
            }
        };
        match self.local_costmap.view_at(now) {
            Ok(view)
                if view.freshness().is_current()
                    && view
                        .provenance()
                        .is_some_and(|provenance| provenance.frame() == expected_depth) =>
            {
                Ok(())
            }
            Ok(view) if view.freshness().is_current() => {
                Err(CoordinatorTickBlocker::DepthUnaligned)
            }
            Ok(_) => Err(CoordinatorTickBlocker::LocalCostmapExpired),
            Err(source) => Err(CoordinatorTickBlocker::LocalCostmapClock(source)),
        }
    }

    /// Bind the freshest admitted robot pose to this exact immutable global
    /// map. Frontier selection uses this instead of reconstructing transforms
    /// or borrowing a stale path start.
    pub fn plan_start_for_snapshot(
        &self,
        snapshot: &OccupancyGridSnapshot,
    ) -> Result<PlanStart, PlanStartBuildError> {
        let state = self
            .odometry
            .current()
            .ok_or(PlanStartBuildError::OdometryUnavailable)?;
        let odometry_map_instance_id = state.map_snapshot().instance_id();
        if snapshot.map_instance_id() != Some(odometry_map_instance_id) {
            return Err(PlanStartBuildError::OdomMapMismatch {
                odometry_map_instance_id,
                snapshot_map_instance_id: snapshot.map_instance_id(),
            });
        }
        let base_in_map = state
            .odom_to_map()
            .compose(state.base_to_odom())
            .and_then(|base_to_map| base_to_map.transform_point(PlanarPoint::<BaseFrame>::origin()))
            .map_err(PlanStartBuildError::Transform)?;
        PlanStart::for_snapshot(base_in_map, snapshot).map_err(PlanStartBuildError::Plan)
    }

    pub fn accept_visual(
        &mut self,
        admission: VisualAdmission,
        now: HostMonotonicTimestamp,
    ) -> Result<VisualAdmissionOutcome, CoordinatorAdmissionError<J::Error>> {
        self.ensure_open()?;
        let session_id = admission.ingress.session_id();
        let host_arrival = self
            .resolve_arrival(admission.ingress.arrival_offset())
            .map_err(CoordinatorAdmissionError::ReplayClock)?;
        self.append(NavigationIngressEvent::VisualAttempt(admission.ingress))?;

        if self
            .odometry
            .current()
            .is_some_and(|state| state.session_id() != session_id)
        {
            self.local_costmap.reset_session(session_id);
            self.depth_readiness = DepthReadiness::NoObservation;
            self.latest_device_time = None;
            self.visual_readiness = VisualReadiness::AwaitingAnchor;
        }

        match admission.payload {
            VisualAdmissionPayload::IncrementAndLocalization {
                increment,
                localization,
            } => {
                if self.must_reanchor(session_id, increment.from(), localization) {
                    self.reanchor(session_id, localization, host_arrival, now)
                } else {
                    match self.odometry.observe_visual(
                        session_id,
                        increment,
                        localization,
                        host_arrival,
                        now,
                    ) {
                        Ok(state) => {
                            self.accept_visual_state(&state);
                            Ok(VisualAdmissionOutcome::Updated(state))
                        }
                        Err(OdometryError::ReanchorRequired(_)) => {
                            self.reanchor(session_id, localization, host_arrival, now)
                        }
                        Err(source) => {
                            self.visual_readiness = VisualReadiness::RejectedUpdate;
                            Ok(VisualAdmissionOutcome::Rejected(source))
                        }
                    }
                }
            }
            VisualAdmissionPayload::LocalizationOnly { localization } => {
                let current_stamp = self
                    .odometry
                    .current()
                    .map(|state| state.source_visual().visual_stamp());
                let current_map = self
                    .odometry
                    .current()
                    .map(|state| state.map_snapshot().instance_id());
                if self.visual_readiness == VisualReadiness::Continuous
                    && current_stamp == Some(localization.visual_stamp())
                    && current_map == Some(localization.map_snapshot().instance_id())
                {
                    match self
                        .odometry
                        .observe_map_localization(session_id, localization)
                    {
                        Ok(state) => {
                            self.accept_visual_state(&state);
                            Ok(VisualAdmissionOutcome::Updated(state))
                        }
                        Err(source) => {
                            self.visual_readiness = VisualReadiness::RejectedUpdate;
                            Ok(VisualAdmissionOutcome::Rejected(source))
                        }
                    }
                } else {
                    self.reanchor(session_id, localization, host_arrival, now)
                }
            }
            VisualAdmissionPayload::NoLocalization
            | VisualAdmissionPayload::RecoverableFailure
            | VisualAdmissionPayload::FatalFailure => {
                self.visual_readiness = VisualReadiness::BrokenAttempt;
                Ok(VisualAdmissionOutcome::ChainBroken(
                    admission.ingress.outcome(),
                ))
            }
        }
    }

    pub fn accept_imu(
        &mut self,
        report: ImuReport,
        now: HostMonotonicTimestamp,
    ) -> Result<ImuAdmissionOutcome, CoordinatorAdmissionError<J::Error>> {
        self.ensure_open()?;
        let recorded = RecordedImuReport::parse(self.clock_epoch, report)
            .map_err(CoordinatorAdmissionError::Boundary)?;
        self.append(NavigationIngressEvent::ImuReport(recorded))?;
        match self.odometry.observe_imu(report, now) {
            Ok(update) => {
                let timestamp = report.gyro().timestamp();
                self.advance_latest_device_time(report.session_id(), timestamp);
                Ok(ImuAdmissionOutcome::Updated(update))
            }
            Err(source) => Ok(ImuAdmissionOutcome::Rejected(source)),
        }
    }

    pub fn accept_depth(
        &mut self,
        observation: crate::DepthObservation,
        now: HostMonotonicTimestamp,
    ) -> Result<DepthAdmissionOutcome, CoordinatorAdmissionError<J::Error>> {
        self.ensure_open()?;
        let ingress = AcceptedDepthIngress::parse(self.clock_epoch, &observation)
            .map_err(CoordinatorAdmissionError::Boundary)?;
        self.append(NavigationIngressEvent::AcceptedDepth(ingress))?;
        let frame_id = observation.frame_id();
        let timestamp = observation.device_timestamp();
        let pose = match self
            .odometry
            .pose_at(observation.session_id(), timestamp, now)
        {
            Ok(PoseHistoryQuery::Available(pose)) => pose,
            Ok(PoseHistoryQuery::Unavailable(reason)) => {
                self.depth_readiness = DepthReadiness::Unaligned {
                    frame_id,
                    timestamp,
                    reason,
                };
                return Ok(DepthAdmissionOutcome::PoseUnavailable(reason));
            }
            Err(source) => {
                self.depth_readiness = DepthReadiness::Rejected {
                    frame_id,
                    timestamp,
                };
                return Ok(DepthAdmissionOutcome::OdometryRejected(source));
            }
        };
        let local = match LocalDepthObservation::try_from_time_aligned(observation, pose) {
            Ok(local) => local,
            Err(source) => {
                self.depth_readiness = DepthReadiness::Rejected {
                    frame_id,
                    timestamp,
                };
                return Ok(DepthAdmissionOutcome::AlignmentRejected(source));
            }
        };
        match self.local_costmap.update(local) {
            Ok(outcome) => {
                let frame = match outcome {
                    LocalCostmapUpdateOutcome::Accepted { frame, .. }
                    | LocalCostmapUpdateOutcome::IgnoredDuplicate { frame } => frame,
                };
                self.depth_readiness = DepthReadiness::Current(frame);
                Ok(DepthAdmissionOutcome::Updated(outcome))
            }
            Err(source) => {
                self.depth_readiness = DepthReadiness::Rejected {
                    frame_id,
                    timestamp,
                };
                Ok(DepthAdmissionOutcome::CostmapRejected(source))
            }
        }
    }

    /// Accept and plan against an immutable snapshot by borrowing it.  The
    /// caller remains free to move that exact snapshot to visualization after
    /// this returns; the coordinator retains only the produced path.
    pub fn accept_global_map(
        &mut self,
        host_arrival: HostMonotonicTimestamp,
        source_capture_timestamp: Timestamp,
        snapshot: &OccupancyGridSnapshot,
    ) -> Result<GlobalMapAdmissionOutcome, CoordinatorAdmissionError<J::Error>> {
        self.ensure_open()?;
        let map_instance_id =
            snapshot
                .map_instance_id()
                .ok_or(CoordinatorAdmissionError::Boundary(
                    NavigationIngressBoundaryError::GlobalMapHasNoInstance,
                ))?;
        let started_new_epoch = self
            .current_map
            .is_none_or(|current| current.binding.map_instance_id() != map_instance_id);
        if !started_new_epoch
            && let Some(current) = self.current_map
            && snapshot.revision() <= current.revision
        {
            return Err(CoordinatorAdmissionError::MapRevisionNotIncreasing {
                previous: current.revision,
                actual: snapshot.revision(),
            });
        }
        let binding = if started_new_epoch {
            let transition =
                match self
                    .map_epochs
                    .start_epoch(self.clock_epoch, host_arrival, map_instance_id)
                {
                    Ok(transition) => transition,
                    Err(NavigationIngressBoundaryError::MapEpochIdExhausted) => {
                        self.latch = Some(CoordinatorLatch::MapEpochIdExhausted);
                        return Err(CoordinatorAdmissionError::Latched(
                            CoordinatorLatch::MapEpochIdExhausted,
                        ));
                    }
                    Err(source) => return Err(CoordinatorAdmissionError::Boundary(source)),
                };
            self.append(NavigationIngressEvent::MapEpochStarted(transition.event()))?;
            let binding = transition.binding();
            if let Some(previous) = self.current_map {
                self.invalidate_for_map_reset(
                    previous.binding.map_instance_id(),
                    binding.map_instance_id(),
                );
            }
            self.current_map = Some(CurrentOccupancyMap {
                binding,
                revision: snapshot.revision(),
            });
            binding
        } else {
            self.current_map
                .expect("same-map admission has a current binding")
                .binding
        };

        let accepted = AcceptedGlobalMapIngress::parse_snapshot(
            self.clock_epoch,
            host_arrival,
            binding,
            source_capture_timestamp,
            snapshot,
        )
        .map_err(CoordinatorAdmissionError::Boundary)?;
        self.append(NavigationIngressEvent::AcceptedGlobalMap(accepted))?;
        self.current_map = Some(CurrentOccupancyMap {
            binding,
            revision: snapshot.revision(),
        });
        self.path = None;
        self.plan_fault = None;

        if let GoalBinding::Pending(point) = self.goal {
            let goal = PointGoal::for_snapshot(point, snapshot)
                .map_err(CoordinatorAdmissionError::Plan)?;
            let next_goal_generation = self.reserve_goal_generation()?;
            let ingress = MapPointGoalIngress::parse(self.clock_epoch, host_arrival, binding, goal)
                .map_err(CoordinatorAdmissionError::Boundary)?;
            self.append(NavigationIngressEvent::PointGoal(ingress))?;
            self.goal = GoalBinding::Bound(BoundPointGoal::unbounded(goal));
            self.goal_generation = next_goal_generation;
        }

        let planning = self.plan_snapshot(snapshot);
        Ok(GlobalMapAdmissionOutcome {
            map_instance_id,
            revision: snapshot.revision(),
            started_new_epoch,
            goal_state: self.goal_state(),
            planning,
        })
    }

    /// Prepare a point-goal replacement without mutating coordinator state.
    ///
    /// The epoch, accepted revision, snapshot map instance, and snapshot
    /// revision must all agree. The returned proof retains a borrow of that
    /// exact snapshot so a later commit cannot substitute equal-looking
    /// metadata around different occupancy content.
    pub fn prepare_map_point_goal<'snapshot>(
        &self,
        selection: MapPointGoalSelection,
        displayed_snapshot: &'snapshot OccupancyGridSnapshot,
    ) -> Result<PreparedMapPointGoal<'snapshot>, CoordinatorAdmissionError<J::Error>> {
        self.ensure_open()?;
        if !matches!(
            self.motion_mode,
            CoordinatorMotionModeV1::MappingOnly | CoordinatorMotionModeV1::PointGoal
        ) {
            return Err(CoordinatorAdmissionError::PointGoalModeConflict {
                actual: self.motion_mode,
            });
        }
        let current = self
            .current_map
            .ok_or(CoordinatorAdmissionError::NoCurrentMapForGoal)?;
        if selection.map_epoch_id() != current.binding.map_epoch_id() {
            return Err(CoordinatorAdmissionError::GoalMapEpochMismatch {
                displayed: selection.map_epoch_id(),
                current: current.binding.map_epoch_id(),
            });
        }
        if selection.displayed_revision() != current.revision {
            return Err(CoordinatorAdmissionError::GoalDisplayedRevisionMismatch {
                displayed: selection.displayed_revision(),
                current: current.revision,
            });
        }
        if displayed_snapshot.map_instance_id() != Some(current.binding.map_instance_id()) {
            return Err(CoordinatorAdmissionError::GoalSnapshotMapMismatch {
                expected: current.binding.map_instance_id(),
                actual: displayed_snapshot.map_instance_id(),
            });
        }
        if displayed_snapshot.revision() != selection.displayed_revision() {
            return Err(CoordinatorAdmissionError::GoalSnapshotRevisionMismatch {
                displayed: selection.displayed_revision(),
                snapshot: displayed_snapshot.revision(),
            });
        }

        let goal = PointGoal::for_snapshot(selection.point(), displayed_snapshot)
            .map_err(CoordinatorAdmissionError::Plan)?;
        Ok(PreparedMapPointGoal {
            displayed_snapshot,
            current_map: current,
            goal_generation: self.goal_generation,
            motion_mode_generation: self.motion_mode_generation,
            previous_goal: self.goal,
            previous_motion_mode: self.motion_mode,
            goal,
            traversal_boundary: None,
        })
    }

    /// Prepare a Nano-bounded frontier goal without discarding the selector's
    /// non-forgeable traversal constraint.
    ///
    /// The returned proof is consumed by the same commit path as an operator
    /// click, but every plan and subsequent replan for this goal remains bound
    /// to the exact closed map-frame rectangle carried by `frontier`.
    pub fn prepare_frontier_goal<'snapshot>(
        &self,
        frontier: FrontierGoal,
        displayed_snapshot: &'snapshot OccupancyGridSnapshot,
    ) -> Result<PreparedMapPointGoal<'snapshot>, CoordinatorAdmissionError<J::Error>> {
        self.ensure_open()?;
        if !matches!(
            self.motion_mode,
            CoordinatorMotionModeV1::MappingOnly | CoordinatorMotionModeV1::PointGoal
        ) {
            return Err(CoordinatorAdmissionError::PointGoalModeConflict {
                actual: self.motion_mode,
            });
        }
        let current = self
            .current_map
            .ok_or(CoordinatorAdmissionError::NoCurrentMapForGoal)?;
        if displayed_snapshot.map_instance_id() != Some(current.binding.map_instance_id()) {
            return Err(CoordinatorAdmissionError::GoalSnapshotMapMismatch {
                expected: current.binding.map_instance_id(),
                actual: displayed_snapshot.map_instance_id(),
            });
        }
        if displayed_snapshot.revision() != current.revision {
            return Err(CoordinatorAdmissionError::GoalSnapshotRevisionMismatch {
                displayed: current.revision,
                snapshot: displayed_snapshot.revision(),
            });
        }
        if frontier.map_instance_id() != current.binding.map_instance_id() {
            return Err(CoordinatorAdmissionError::FrontierGoalMapMismatch {
                expected: current.binding.map_instance_id(),
                actual: frontier.map_instance_id(),
            });
        }
        if frontier.map_revision() != current.revision {
            return Err(CoordinatorAdmissionError::FrontierGoalRevisionMismatch {
                expected: current.revision,
                actual: frontier.map_revision(),
            });
        }
        let Some(traversal_boundary) = frontier.traversal_boundary() else {
            return Err(CoordinatorAdmissionError::FrontierGoalMissingTraversalBoundary);
        };
        let goal = frontier.point_goal();
        if !goal.was_selected_from(displayed_snapshot) {
            return Err(CoordinatorAdmissionError::FrontierGoalRevisionMismatch {
                expected: displayed_snapshot.revision(),
                actual: goal.selected_revision(),
            });
        }
        Ok(PreparedMapPointGoal {
            displayed_snapshot,
            current_map: current,
            goal_generation: self.goal_generation,
            motion_mode_generation: self.motion_mode_generation,
            previous_goal: self.goal,
            previous_motion_mode: self.motion_mode,
            goal,
            traversal_boundary: Some(traversal_boundary),
        })
    }

    /// Commit a previously prepared point goal after exact state rechecking.
    ///
    /// Any intervening map admission, map reset, goal change, clear, or motion
    /// mode transition makes the proof stale. Rejection is non-mutating and
    /// emits no ingress record.
    pub fn commit_prepared_map_point_goal(
        &mut self,
        host_arrival: HostMonotonicTimestamp,
        prepared: PreparedMapPointGoal<'_>,
    ) -> Result<GoalSelectionOutcome, CoordinatorAdmissionError<J::Error>> {
        self.ensure_open()?;
        let current_map = self.current_map;
        if current_map != Some(prepared.current_map)
            || self.goal != prepared.previous_goal
            || self.motion_mode != prepared.previous_motion_mode
            || self.goal_generation != prepared.goal_generation
            || self.motion_mode_generation != prepared.motion_mode_generation
        {
            return Err(CoordinatorAdmissionError::PointGoalPreparationStale {
                prepared_map_binding: prepared.current_map.binding,
                prepared_revision: prepared.current_map.revision,
                current_map_binding: current_map.map(|current| current.binding),
                current_revision: current_map.map(|current| current.revision),
                prepared_motion_mode: prepared.previous_motion_mode,
                current_motion_mode: self.motion_mode,
                prepared_goal_generation: prepared.goal_generation,
                current_goal_generation: self.goal_generation,
                prepared_motion_mode_generation: prepared.motion_mode_generation,
                current_motion_mode_generation: self.motion_mode_generation,
            });
        }
        if !matches!(
            self.motion_mode,
            CoordinatorMotionModeV1::MappingOnly | CoordinatorMotionModeV1::PointGoal
        ) {
            return Err(CoordinatorAdmissionError::PointGoalModeConflict {
                actual: self.motion_mode,
            });
        }
        if !prepared.goal.was_selected_from(prepared.displayed_snapshot) {
            return Err(CoordinatorAdmissionError::GoalSnapshotRevisionMismatch {
                displayed: prepared.goal.selected_revision(),
                snapshot: prepared.displayed_snapshot.revision(),
            });
        }
        let next_goal_generation = self.reserve_goal_generation()?;
        let next_motion_mode_generation = (self.motion_mode != CoordinatorMotionModeV1::PointGoal)
            .then(|| self.reserve_motion_mode_generation())
            .transpose()
            .map_err(CoordinatorAdmissionError::Latched)?;
        let ingress = MapPointGoalIngress::parse(
            self.clock_epoch,
            host_arrival,
            prepared.current_map.binding,
            prepared.goal,
        )
        .map_err(CoordinatorAdmissionError::Boundary)?;
        self.append(NavigationIngressEvent::PointGoal(ingress))?;

        self.goal = GoalBinding::Bound(BoundPointGoal {
            point_goal: prepared.goal,
            traversal_boundary: prepared.traversal_boundary,
        });
        self.goal_generation = next_goal_generation;
        self.motion_mode = CoordinatorMotionModeV1::PointGoal;
        if let Some(next_motion_mode_generation) = next_motion_mode_generation {
            self.motion_mode_generation = next_motion_mode_generation;
        }
        self.direct_localization_binding = None;
        self.path = None;
        self.plan_fault = None;
        let planning = self.plan_snapshot(prepared.displayed_snapshot);
        Ok(GoalSelectionOutcome {
            goal_state: self.goal_state(),
            planning,
        })
    }

    /// Replace the active point goal using the exact immutable map snapshot
    /// displayed by the control surface.
    ///
    /// This compatibility API performs prepare and commit adjacently. Live
    /// owners that need an authority/fresh-zero handover must use the explicit
    /// two-phase API above.
    pub fn select_map_point_goal(
        &mut self,
        host_arrival: HostMonotonicTimestamp,
        selection: MapPointGoalSelection,
        displayed_snapshot: &OccupancyGridSnapshot,
    ) -> Result<GoalSelectionOutcome, CoordinatorAdmissionError<J::Error>> {
        let prepared = self.prepare_map_point_goal(selection, displayed_snapshot)?;
        self.commit_prepared_map_point_goal(host_arrival, prepared)
    }

    /// Select a frontier while retaining its mandatory execution boundary.
    ///
    /// Production exploration owners needing an authority/fresh-zero handover
    /// should call [`Self::prepare_frontier_goal`] and commit separately.
    pub fn select_frontier_goal(
        &mut self,
        host_arrival: HostMonotonicTimestamp,
        frontier: FrontierGoal,
        displayed_snapshot: &OccupancyGridSnapshot,
    ) -> Result<GoalSelectionOutcome, CoordinatorAdmissionError<J::Error>> {
        let prepared = self.prepare_frontier_goal(frontier, displayed_snapshot)?;
        self.commit_prepared_map_point_goal(host_arrival, prepared)
    }

    pub fn tick<C: HostMonotonicClock>(
        &mut self,
        tick: HostMonotonicTimestamp,
        clock: &mut C,
    ) -> Result<CoordinatorTickOutcome<J::Error>, CoordinatorTickError> {
        let (control_tick_journaled, journal_error, journal_blocker) =
            self.journal_control_tick(tick);
        if let Some(blocker) = journal_blocker {
            return self.stop_tick(tick, blocker, control_tick_journaled, journal_error, clock);
        }
        if let Some(blocker) = self.preflight_blocker(tick) {
            return self.stop_tick(tick, blocker, control_tick_journaled, None, clock);
        }

        self.ready_tick(tick, control_tick_journaled, clock)
    }

    fn journal_control_tick(
        &mut self,
        tick: HostMonotonicTimestamp,
    ) -> (bool, Option<J::Error>, Option<CoordinatorTickBlocker>) {
        if self.latch.is_some() {
            return (false, None, Some(CoordinatorTickBlocker::JournalLatched));
        }
        match ControlTickIngress::parse(self.clock_epoch, tick) {
            Ok(event) => match self.append(NavigationIngressEvent::ControlTick(event)) {
                Ok(_) => (true, None, None),
                Err(CoordinatorAdmissionError::Journal(source)) => (
                    false,
                    Some(source),
                    Some(CoordinatorTickBlocker::JournalLatched),
                ),
                Err(_) => (false, None, Some(CoordinatorTickBlocker::JournalLatched)),
            },
            Err(source) => {
                self.latch = Some(CoordinatorLatch::ClockBoundary);
                (
                    false,
                    None,
                    Some(CoordinatorTickBlocker::ControlTickBoundary(source)),
                )
            }
        }
    }

    /// Run one manual command through the same MPC, immutable collision view,
    /// solver deadline, final revalidation, and shadow evidence path as a
    /// point goal. No manual value is ever converted directly to PWM.
    pub fn tick_manual<C: HostMonotonicClock>(
        &mut self,
        tick: HostMonotonicTimestamp,
        command: ManualMpcCommandV1,
        clock: &mut C,
    ) -> Result<CoordinatorTickOutcome<J::Error>, CoordinatorTickError> {
        let (control_tick_journaled, journal_error, journal_blocker) =
            self.journal_control_tick(tick);
        if let Some(blocker) = journal_blocker {
            return self.stop_tick(tick, blocker, control_tick_journaled, journal_error, clock);
        }
        let bound_lease = match self.motion_mode {
            CoordinatorMotionModeV1::Manual { authority_lease_id } => authority_lease_id,
            actual => {
                return self.stop_tick(
                    tick,
                    CoordinatorTickBlocker::MotionModeMismatch {
                        expected: CoordinatorMotionModeV1::Manual {
                            authority_lease_id: command.authority_lease_id(),
                        },
                        actual,
                    },
                    control_tick_journaled,
                    None,
                    clock,
                );
            }
        };
        if bound_lease != command.authority_lease_id() {
            return self.stop_tick(
                tick,
                CoordinatorTickBlocker::DirectAuthorityLeaseMismatch {
                    bound: bound_lease,
                    command: command.authority_lease_id(),
                },
                control_tick_journaled,
                None,
                clock,
            );
        }
        if let Some(blocker) =
            self.admit_manual_command_order(ManualCommandEvidence::Velocity(command))
        {
            return self.stop_tick(tick, blocker, control_tick_journaled, None, clock);
        }
        if let Some(blocker) = self.direct_preflight_blocker() {
            return self.stop_tick(tick, blocker, control_tick_journaled, None, clock);
        }
        let state = match self.estimate_state_at(tick) {
            Ok(state) => state,
            Err(blocker) => {
                return self.stop_tick(tick, blocker, control_tick_journaled, None, clock);
            }
        };
        if !self.admit_direct_localization_binding(&state) {
            return self.stop_tick(
                tick,
                CoordinatorTickBlocker::DirectLocalizationEpochTransition,
                control_tick_journaled,
                None,
                clock,
            );
        }
        let epoch = NavigationEpochV1::for_manual_body_twist(
            state.session_id(),
            state.segment_id(),
            state.map_snapshot(),
            command.identity(),
        );
        let motion_state = match motion_state_from_odometry(&state, epoch, tick) {
            Ok(state) => state,
            Err(source) => {
                return self.stop_tick(
                    tick,
                    CoordinatorTickBlocker::MotionState(source),
                    control_tick_journaled,
                    None,
                    clock,
                );
            }
        };
        let reference = match self.manual_reference_builder.build(
            epoch,
            command,
            motion_state.pose(),
            self.mpc_config,
            tick,
        ) {
            Ok(reference) => reference,
            Err(source) => {
                return self.stop_tick(
                    tick,
                    CoordinatorTickBlocker::ManualReference(source),
                    control_tick_journaled,
                    None,
                    clock,
                );
            }
        };
        let expected_depth = match self.depth_readiness {
            DepthReadiness::Current(frame) => frame,
            DepthReadiness::NoObservation => {
                return self.stop_tick(
                    tick,
                    CoordinatorTickBlocker::DepthUnavailable,
                    control_tick_journaled,
                    None,
                    clock,
                );
            }
            DepthReadiness::Unaligned { .. } | DepthReadiness::Rejected { .. } => {
                return self.stop_tick(
                    tick,
                    CoordinatorTickBlocker::DepthUnaligned,
                    control_tick_journaled,
                    None,
                    clock,
                );
            }
        };
        let local_view = match self.local_costmap.view_at(tick) {
            Ok(view) => view,
            Err(source) => {
                return self.stop_tick(
                    tick,
                    CoordinatorTickBlocker::LocalCostmapClock(source),
                    control_tick_journaled,
                    None,
                    clock,
                );
            }
        };
        let freshness_blocker = if !local_view.freshness().is_current() {
            Some(CoordinatorTickBlocker::LocalCostmapExpired)
        } else if local_view
            .provenance()
            .is_none_or(|provenance| provenance.frame() != expected_depth)
        {
            Some(CoordinatorTickBlocker::DepthUnaligned)
        } else {
            None
        };
        if let Some(blocker) = freshness_blocker {
            return self.stop_tick(tick, blocker, control_tick_journaled, None, clock);
        }
        let ready = SafetyReadyTick::new(
            epoch,
            motion_state,
            &reference,
            local_view,
            self.solver_budget,
        );
        let decision = self
            .safety
            .decide(tick, SafetyTickInput::Ready(ready), clock)
            .map_err(CoordinatorTickError::Safety)?;
        Ok(CoordinatorTickOutcome {
            decision,
            blocker: None,
            control_tick_journaled,
            journal_error: None,
        })
    }

    /// Record and decide one already-admitted ordered manual stop.
    ///
    /// This deliberately bypasses reference/MPC construction: a stop cannot
    /// be represented as a zero-valued velocity reference. It still journals
    /// the control tick and produces the same typed zero safety decision later
    /// consumed by the receipt-gated physical session.
    pub fn tick_manual_explicit_stop<C, LeaseId>(
        &mut self,
        tick: HostMonotonicTimestamp,
        stop: ManualDriveAcceptedStop<LeaseId>,
        clock: &mut C,
    ) -> Result<CoordinatorTickOutcome<J::Error>, CoordinatorTickError>
    where
        C: HostMonotonicClock,
        LeaseId: NumericAuthorityLeaseId,
    {
        let (control_tick_journaled, journal_error, journal_blocker) =
            self.journal_control_tick(tick);
        if let Some(blocker) = journal_blocker {
            return self.stop_tick(tick, blocker, control_tick_journaled, journal_error, clock);
        }
        let authority_lease_id = numeric_authority_lease(stop.authority_lease_id());
        let bound_lease = match self.motion_mode {
            CoordinatorMotionModeV1::Manual { authority_lease_id } => authority_lease_id,
            actual => {
                return self.stop_tick(
                    tick,
                    CoordinatorTickBlocker::MotionModeMismatch {
                        expected: CoordinatorMotionModeV1::Manual { authority_lease_id },
                        actual,
                    },
                    control_tick_journaled,
                    None,
                    clock,
                );
            }
        };
        if bound_lease != authority_lease_id {
            return self.stop_tick(
                tick,
                CoordinatorTickBlocker::DirectAuthorityLeaseMismatch {
                    bound: bound_lease,
                    command: authority_lease_id,
                },
                control_tick_journaled,
                None,
                clock,
            );
        }
        let evidence = ManualCommandEvidence::ExplicitStop {
            authority_lease_id,
            sequence: stop.sequence().get(),
        };
        if let Some(blocker) = self.admit_manual_command_order(evidence) {
            return self.stop_tick(tick, blocker, control_tick_journaled, None, clock);
        }
        self.stop_tick(
            tick,
            CoordinatorTickBlocker::ManualExplicitStop {
                authority_lease_id,
                sequence: stop.sequence().get(),
            },
            control_tick_journaled,
            None,
            clock,
        )
    }

    /// Record the exact fail-closed output of `ManualDriveCore` without
    /// relabelling it as a point-goal or motion-mode failure.
    pub fn tick_manual_stopped<C, LeaseId>(
        &mut self,
        tick: HostMonotonicTimestamp,
        stopped: ManualDriveStopped<LeaseId>,
        clock: &mut C,
    ) -> Result<CoordinatorTickOutcome<J::Error>, CoordinatorTickError>
    where
        C: HostMonotonicClock,
        LeaseId: NumericAuthorityLeaseId,
    {
        let (control_tick_journaled, journal_error, journal_blocker) =
            self.journal_control_tick(tick);
        if let Some(blocker) = journal_blocker {
            return self.stop_tick(tick, blocker, control_tick_journaled, journal_error, clock);
        }
        let authority_lease_id = numeric_authority_lease(stopped.bound_authority_lease_id());
        let bound_lease = match self.motion_mode {
            CoordinatorMotionModeV1::Manual { authority_lease_id } => authority_lease_id,
            actual => {
                return self.stop_tick(
                    tick,
                    CoordinatorTickBlocker::MotionModeMismatch {
                        expected: CoordinatorMotionModeV1::Manual { authority_lease_id },
                        actual,
                    },
                    control_tick_journaled,
                    None,
                    clock,
                );
            }
        };
        if bound_lease != authority_lease_id {
            return self.stop_tick(
                tick,
                CoordinatorTickBlocker::DirectAuthorityLeaseMismatch {
                    bound: bound_lease,
                    command: authority_lease_id,
                },
                control_tick_journaled,
                None,
                clock,
            );
        }
        let blocker = CoordinatorTickBlocker::ManualDriveStopped {
            authority_lease_id,
            cause: stopped.cause().map_authority_lease(numeric_authority_lease),
        };
        self.stop_tick(tick, blocker, control_tick_journaled, None, clock)
    }

    /// Run a map-revision-bound frontier yaw target without fabricating a
    /// global point goal or path.
    pub fn tick_frontier_yaw<C: HostMonotonicClock>(
        &mut self,
        tick: HostMonotonicTimestamp,
        command: FrontierYawScanCommandV1,
        clock: &mut C,
    ) -> Result<CoordinatorTickOutcome<J::Error>, CoordinatorTickError> {
        let (control_tick_journaled, journal_error, journal_blocker) =
            self.journal_control_tick(tick);
        if let Some(blocker) = journal_blocker {
            return self.stop_tick(tick, blocker, control_tick_journaled, journal_error, clock);
        }
        let bound_lease = match self.motion_mode {
            CoordinatorMotionModeV1::FrontierInPlaceYaw { authority_lease_id } => {
                authority_lease_id
            }
            actual => {
                return self.stop_tick(
                    tick,
                    CoordinatorTickBlocker::MotionModeMismatch {
                        expected: CoordinatorMotionModeV1::FrontierInPlaceYaw {
                            authority_lease_id: command.authority_lease_id(),
                        },
                        actual,
                    },
                    control_tick_journaled,
                    None,
                    clock,
                );
            }
        };
        if bound_lease != command.authority_lease_id() {
            return self.stop_tick(
                tick,
                CoordinatorTickBlocker::DirectAuthorityLeaseMismatch {
                    bound: bound_lease,
                    command: command.authority_lease_id(),
                },
                control_tick_journaled,
                None,
                clock,
            );
        }
        if let Some(blocker) = self.admit_frontier_command_order(command) {
            return self.stop_tick(tick, blocker, control_tick_journaled, None, clock);
        }
        let Some(current_map) = self.current_map else {
            return self.stop_tick(
                tick,
                CoordinatorTickBlocker::GoalUnavailable,
                control_tick_journaled,
                None,
                clock,
            );
        };
        if current_map.binding.map_instance_id() != command.scan().map_instance_id()
            || current_map.revision != command.scan().map_revision()
        {
            return self.stop_tick(
                tick,
                CoordinatorTickBlocker::FrontierMapRevisionMismatch {
                    expected_map_instance_id: current_map.binding.map_instance_id(),
                    expected_revision: current_map.revision,
                    actual_map_instance_id: command.scan().map_instance_id(),
                    actual_revision: command.scan().map_revision(),
                },
                control_tick_journaled,
                None,
                clock,
            );
        }
        if let Some(blocker) = self.direct_preflight_blocker() {
            return self.stop_tick(tick, blocker, control_tick_journaled, None, clock);
        }
        let state = match self.estimate_state_at(tick) {
            Ok(state) => state,
            Err(blocker) => {
                return self.stop_tick(tick, blocker, control_tick_journaled, None, clock);
            }
        };
        if !self.admit_direct_localization_binding(&state) {
            return self.stop_tick(
                tick,
                CoordinatorTickBlocker::DirectLocalizationEpochTransition,
                control_tick_journaled,
                None,
                clock,
            );
        }
        let base_to_odom = state.base_to_odom();
        let current_pose = match OdomPoseV1::try_new(
            base_to_odom.source_origin_x_in_destination_m(),
            base_to_odom.source_origin_y_in_destination_m(),
            base_to_odom.source_yaw_in_destination_rad(),
        ) {
            Ok(pose) => pose,
            Err(source) => {
                return self.stop_tick(
                    tick,
                    CoordinatorTickBlocker::MotionState(source),
                    control_tick_journaled,
                    None,
                    clock,
                );
            }
        };
        let map_to_odom = match state.map_to_odom() {
            Ok(transform) => transform,
            Err(source) => {
                return self.stop_tick(
                    tick,
                    CoordinatorTickBlocker::MapToOdom(source),
                    control_tick_journaled,
                    None,
                    clock,
                );
            }
        };
        let (epoch, reference) = match self.frontier_yaw_reference_builder.build(
            command,
            state.session_id(),
            state.segment_id(),
            state.map_snapshot(),
            map_to_odom,
            current_pose,
            self.mpc_config,
            tick,
        ) {
            Ok(reference) => reference,
            Err(source) => {
                return self.stop_tick(
                    tick,
                    CoordinatorTickBlocker::FrontierYawReference(source),
                    control_tick_journaled,
                    None,
                    clock,
                );
            }
        };
        let motion_state = match motion_state_from_odometry(&state, epoch, tick) {
            Ok(state) => state,
            Err(source) => {
                return self.stop_tick(
                    tick,
                    CoordinatorTickBlocker::MotionState(source),
                    control_tick_journaled,
                    None,
                    clock,
                );
            }
        };
        let expected_depth = match self.depth_readiness {
            DepthReadiness::Current(frame) => frame,
            DepthReadiness::NoObservation => {
                return self.stop_tick(
                    tick,
                    CoordinatorTickBlocker::DepthUnavailable,
                    control_tick_journaled,
                    None,
                    clock,
                );
            }
            DepthReadiness::Unaligned { .. } | DepthReadiness::Rejected { .. } => {
                return self.stop_tick(
                    tick,
                    CoordinatorTickBlocker::DepthUnaligned,
                    control_tick_journaled,
                    None,
                    clock,
                );
            }
        };
        let local_view = match self.local_costmap.view_at(tick) {
            Ok(view) => view,
            Err(source) => {
                return self.stop_tick(
                    tick,
                    CoordinatorTickBlocker::LocalCostmapClock(source),
                    control_tick_journaled,
                    None,
                    clock,
                );
            }
        };
        let freshness_blocker = if !local_view.freshness().is_current() {
            Some(CoordinatorTickBlocker::LocalCostmapExpired)
        } else if local_view
            .provenance()
            .is_none_or(|provenance| provenance.frame() != expected_depth)
        {
            Some(CoordinatorTickBlocker::DepthUnaligned)
        } else {
            None
        };
        if let Some(blocker) = freshness_blocker {
            return self.stop_tick(tick, blocker, control_tick_journaled, None, clock);
        }
        let ready = SafetyReadyTick::new(
            epoch,
            motion_state,
            &reference,
            local_view,
            self.solver_budget,
        );
        let decision = self
            .safety
            .decide(tick, SafetyTickInput::Ready(ready), clock)
            .map_err(CoordinatorTickError::Safety)?;
        Ok(CoordinatorTickOutcome {
            decision,
            blocker: None,
            control_tick_journaled,
            journal_error: None,
        })
    }

    fn ready_tick<C: HostMonotonicClock>(
        &mut self,
        tick: HostMonotonicTimestamp,
        control_tick_journaled: bool,
        clock: &mut C,
    ) -> Result<CoordinatorTickOutcome<J::Error>, CoordinatorTickError> {
        let (session_id, device_timestamp) = self
            .latest_device_time
            .expect("preflight proves a device query time");
        let state = match self.odometry.estimate(session_id, device_timestamp, tick) {
            Ok(OdometryEstimate::Available(state)) => state,
            Ok(OdometryEstimate::Unavailable(reason)) => {
                return self.stop_tick(
                    tick,
                    CoordinatorTickBlocker::OdometryUnavailable(reason),
                    control_tick_journaled,
                    None,
                    clock,
                );
            }
            Err(source) => {
                return self.stop_tick(
                    tick,
                    CoordinatorTickBlocker::Odometry(source),
                    control_tick_journaled,
                    None,
                    clock,
                );
            }
        };
        let path = self.path.as_ref().expect("preflight proves a path");
        let current_map = self.current_map.expect("preflight proves a current map");
        if path.map_instance_id() != current_map.binding.map_instance_id()
            || path.map_revision() != current_map.revision
            || state.map_snapshot().instance_id() != path.map_instance_id()
        {
            return self.stop_tick(
                tick,
                CoordinatorTickBlocker::PathInvalidated,
                control_tick_journaled,
                None,
                clock,
            );
        }
        let epoch = match NavigationEpochV1::from_runtime(
            state.session_id(),
            state.segment_id(),
            state.map_snapshot(),
            path,
        ) {
            Ok(epoch) => epoch,
            Err(source) => {
                return self.stop_tick(
                    tick,
                    CoordinatorTickBlocker::NavigationEpoch(source),
                    control_tick_journaled,
                    None,
                    clock,
                );
            }
        };
        let motion_state = match motion_state_from_odometry(&state, epoch, tick) {
            Ok(state) => state,
            Err(source) => {
                return self.stop_tick(
                    tick,
                    CoordinatorTickBlocker::MotionState(source),
                    control_tick_journaled,
                    None,
                    clock,
                );
            }
        };
        let map_to_odom = match state.map_to_odom() {
            Ok(transform) => transform,
            Err(source) => {
                return self.stop_tick(
                    tick,
                    CoordinatorTickBlocker::MapToOdom(source),
                    control_tick_journaled,
                    None,
                    clock,
                );
            }
        };
        let pose = motion_state.pose();
        let reference = match self.reference_builder.build(
            epoch,
            path,
            map_to_odom,
            pose,
            self.mpc_config,
            tick,
        ) {
            Ok(reference) => reference,
            Err(source) => {
                return self.stop_tick(
                    tick,
                    CoordinatorTickBlocker::Reference(source),
                    control_tick_journaled,
                    None,
                    clock,
                );
            }
        };
        let local_view = match self.local_costmap.view_at(tick) {
            Ok(view) => view,
            Err(source) => {
                return self.stop_tick(
                    tick,
                    CoordinatorTickBlocker::LocalCostmapClock(source),
                    control_tick_journaled,
                    None,
                    clock,
                );
            }
        };
        if !local_view.freshness().is_current() {
            return self.stop_tick(
                tick,
                CoordinatorTickBlocker::LocalCostmapExpired,
                control_tick_journaled,
                None,
                clock,
            );
        }
        let ready = SafetyReadyTick::new(
            epoch,
            motion_state,
            &reference,
            local_view,
            self.solver_budget,
        );
        let decision = self
            .safety
            .decide(tick, SafetyTickInput::Ready(ready), clock)
            .map_err(CoordinatorTickError::Safety)?;
        Ok(CoordinatorTickOutcome {
            decision,
            blocker: None,
            control_tick_journaled,
            journal_error: None,
        })
    }

    fn stop_tick<C: HostMonotonicClock>(
        &mut self,
        tick: HostMonotonicTimestamp,
        blocker: CoordinatorTickBlocker,
        control_tick_journaled: bool,
        journal_error: Option<J::Error>,
        clock: &mut C,
    ) -> Result<CoordinatorTickOutcome<J::Error>, CoordinatorTickError> {
        let reason = blocker.safety_reason();
        let decision = self
            .safety
            .decide(tick, SafetyTickInput::NotReady(reason), clock)
            .map_err(CoordinatorTickError::Safety)?;
        Ok(CoordinatorTickOutcome {
            decision,
            blocker: Some(blocker),
            control_tick_journaled,
            journal_error,
        })
    }

    fn preflight_blocker(&self, tick: HostMonotonicTimestamp) -> Option<CoordinatorTickBlocker> {
        if matches!(
            self.motion_mode,
            CoordinatorMotionModeV1::Manual { .. }
                | CoordinatorMotionModeV1::FrontierInPlaceYaw { .. }
        ) {
            return Some(CoordinatorTickBlocker::MotionModeMismatch {
                expected: CoordinatorMotionModeV1::PointGoal,
                actual: self.motion_mode,
            });
        }
        match self.visual_readiness {
            VisualReadiness::AwaitingAnchor => {
                return Some(CoordinatorTickBlocker::VisualOdometryUnavailable);
            }
            VisualReadiness::BrokenAttempt | VisualReadiness::RejectedUpdate => {
                return Some(CoordinatorTickBlocker::VisualOdometryRejected);
            }
            VisualReadiness::Continuous => {}
        }
        match self.goal {
            GoalBinding::Unavailable | GoalBinding::Pending(_) => {
                return Some(CoordinatorTickBlocker::GoalUnavailable);
            }
            GoalBinding::Invalidated { .. } => {
                return Some(CoordinatorTickBlocker::GoalInvalidated);
            }
            GoalBinding::Bound(_) => {}
        }
        if self.path.is_none() {
            return Some(CoordinatorTickBlocker::PathUnavailable(self.plan_fault));
        }
        let expected_depth = match self.depth_readiness {
            DepthReadiness::NoObservation => {
                return Some(CoordinatorTickBlocker::DepthUnavailable);
            }
            DepthReadiness::Unaligned { .. } | DepthReadiness::Rejected { .. } => {
                return Some(CoordinatorTickBlocker::DepthUnaligned);
            }
            DepthReadiness::Current(frame) => frame,
        };
        if self.latest_device_time.is_none() {
            return Some(CoordinatorTickBlocker::VisualOdometryUnavailable);
        }
        match self.local_costmap.view_at(tick) {
            Ok(view)
                if view.freshness().is_current()
                    && view
                        .provenance()
                        .is_some_and(|provenance| provenance.frame() == expected_depth) =>
            {
                None
            }
            Ok(view) if view.freshness().is_current() => {
                Some(CoordinatorTickBlocker::DepthUnaligned)
            }
            Ok(_) => Some(CoordinatorTickBlocker::LocalCostmapExpired),
            Err(source) => Some(CoordinatorTickBlocker::LocalCostmapClock(source)),
        }
    }

    fn direct_preflight_blocker(&self) -> Option<CoordinatorTickBlocker> {
        match self.visual_readiness {
            VisualReadiness::AwaitingAnchor => {
                return Some(CoordinatorTickBlocker::VisualOdometryUnavailable);
            }
            VisualReadiness::BrokenAttempt | VisualReadiness::RejectedUpdate => {
                return Some(CoordinatorTickBlocker::VisualOdometryRejected);
            }
            VisualReadiness::Continuous => {}
        }
        match self.depth_readiness {
            DepthReadiness::NoObservation => {
                return Some(CoordinatorTickBlocker::DepthUnavailable);
            }
            DepthReadiness::Unaligned { .. } | DepthReadiness::Rejected { .. } => {
                return Some(CoordinatorTickBlocker::DepthUnaligned);
            }
            DepthReadiness::Current(_) => {}
        }
        self.latest_device_time
            .is_none()
            .then_some(CoordinatorTickBlocker::VisualOdometryUnavailable)
    }

    fn estimate_state_at(
        &self,
        tick: HostMonotonicTimestamp,
    ) -> Result<OdometryState, CoordinatorTickBlocker> {
        let (session_id, device_timestamp) = self
            .latest_device_time
            .ok_or(CoordinatorTickBlocker::VisualOdometryUnavailable)?;
        match self.odometry.estimate(session_id, device_timestamp, tick) {
            Ok(OdometryEstimate::Available(state)) => Ok(state),
            Ok(OdometryEstimate::Unavailable(reason)) => {
                Err(CoordinatorTickBlocker::OdometryUnavailable(reason))
            }
            Err(source) => Err(CoordinatorTickBlocker::Odometry(source)),
        }
    }

    fn admit_direct_localization_binding(&mut self, state: &OdometryState) -> bool {
        let actual = DirectLocalizationBinding {
            device_session_id: state.session_id(),
            odom_segment_id: state.segment_id(),
            map_instance_id: state.map_snapshot().instance_id(),
        };
        match self.direct_localization_binding {
            None => {
                self.direct_localization_binding = Some(actual);
                true
            }
            Some(bound) => bound == actual,
        }
    }

    fn admit_manual_command_order(
        &mut self,
        command: ManualCommandEvidence,
    ) -> Option<CoordinatorTickBlocker> {
        let Some(previous) = self.last_manual_command else {
            self.last_manual_command = Some(command);
            return None;
        };
        if command == previous {
            return None;
        }
        let previous_sequence = previous.sequence();
        let command_sequence = command.sequence();
        if command_sequence < previous_sequence {
            return Some(CoordinatorTickBlocker::ManualCommandSequenceRegression {
                previous: previous_sequence,
                command: command_sequence,
            });
        }
        if command_sequence == previous_sequence {
            return Some(CoordinatorTickBlocker::ManualCommandIdentityConflict {
                sequence: command_sequence,
            });
        }
        self.last_manual_command = Some(command);
        None
    }

    fn admit_frontier_command_order(
        &mut self,
        command: FrontierYawScanCommandV1,
    ) -> Option<CoordinatorTickBlocker> {
        let Some(previous) = self.last_frontier_yaw_command else {
            self.last_frontier_yaw_command = Some(command);
            return None;
        };
        if command == previous {
            return None;
        }
        let previous_sequence = previous.scan_sequence();
        let command_sequence = command.scan_sequence();
        if command_sequence < previous_sequence {
            return Some(CoordinatorTickBlocker::FrontierCommandSequenceRegression {
                previous: previous_sequence,
                command: command_sequence,
            });
        }
        if command_sequence == previous_sequence {
            return Some(CoordinatorTickBlocker::FrontierCommandIdentityConflict {
                sequence: command_sequence,
            });
        }
        self.last_frontier_yaw_command = Some(command);
        None
    }

    fn plan_snapshot(&mut self, snapshot: &OccupancyGridSnapshot) -> GlobalPlanningOutcome {
        let Some(goal) = (match self.goal {
            GoalBinding::Bound(goal) => Some(goal),
            GoalBinding::Unavailable
            | GoalBinding::Pending(_)
            | GoalBinding::Invalidated { .. } => None,
        }) else {
            self.plan_fault = Some(StoredPlanFault::Planning);
            return GlobalPlanningOutcome::Deferred(StoredPlanFault::Planning);
        };
        let start = match self.plan_start_for_snapshot(snapshot) {
            Ok(start) => start,
            Err(PlanStartBuildError::OdometryUnavailable) => {
                let fault = StoredPlanFault::OdometryUnavailable(OdometryUnavailable::NotAnchored);
                self.plan_fault = Some(fault);
                return GlobalPlanningOutcome::Deferred(fault);
            }
            Err(PlanStartBuildError::OdomMapMismatch { .. }) => {
                self.plan_fault = Some(StoredPlanFault::OdomMapMismatch);
                return GlobalPlanningOutcome::Deferred(StoredPlanFault::OdomMapMismatch);
            }
            Err(PlanStartBuildError::Transform(_)) => {
                self.plan_fault = Some(StoredPlanFault::StartTransform);
                return GlobalPlanningOutcome::Deferred(StoredPlanFault::StartTransform);
            }
            Err(PlanStartBuildError::Plan(source)) => {
                self.plan_fault = Some(StoredPlanFault::Planning);
                return GlobalPlanningOutcome::Failed(source);
            }
        };
        let planner = match goal.traversal_boundary {
            Some(boundary) => {
                GlobalPlanner::try_new_bounded(snapshot, self.planner_config, boundary)
            }
            None => GlobalPlanner::try_new(snapshot, self.planner_config),
        };
        let mut planner = match planner {
            Ok(planner) => planner,
            Err(source) => {
                self.plan_fault = Some(StoredPlanFault::PlannerConstruction);
                return GlobalPlanningOutcome::Failed(source);
            }
        };
        match planner.plan(start, goal.point_goal) {
            Ok(path) => {
                let identity = path.identity();
                self.path = Some(path);
                self.plan_fault = None;
                GlobalPlanningOutcome::Planned(identity)
            }
            Err(source) => {
                self.plan_fault = Some(StoredPlanFault::Planning);
                GlobalPlanningOutcome::Failed(source)
            }
        }
    }

    fn must_reanchor(
        &self,
        session_id: DeviceSessionId,
        increment_from: VisualFrameStamp,
        localization: MapLocalization,
    ) -> bool {
        if self.visual_readiness != VisualReadiness::Continuous {
            return true;
        }
        let Some(current) = self.odometry.current() else {
            return true;
        };
        current.session_id() != session_id
            || current.source_visual().visual_stamp() != increment_from
            || current.map_snapshot().instance_id() != localization.map_snapshot().instance_id()
    }

    fn reanchor(
        &mut self,
        session_id: DeviceSessionId,
        localization: MapLocalization,
        host_arrival: HostMonotonicTimestamp,
        now: HostMonotonicTimestamp,
    ) -> Result<VisualAdmissionOutcome, CoordinatorAdmissionError<J::Error>> {
        let Some(raw) = self.next_segment_id else {
            self.visual_readiness = VisualReadiness::RejectedUpdate;
            return Err(CoordinatorAdmissionError::SegmentIdExhausted);
        };
        let segment_id =
            OdomSegmentId::try_new(raw.get()).expect("nonzero coordinator segment identity");
        match self
            .odometry
            .reanchor(segment_id, session_id, localization, host_arrival, now)
        {
            Ok(state) => {
                self.next_segment_id = raw.get().checked_add(1).and_then(NonZeroU64::new);
                self.local_costmap.reset_session(session_id);
                self.depth_readiness = DepthReadiness::NoObservation;
                self.accept_visual_state(&state);
                Ok(VisualAdmissionOutcome::Reanchored(state))
            }
            Err(source) => {
                self.visual_readiness = VisualReadiness::RejectedUpdate;
                Ok(VisualAdmissionOutcome::Rejected(source))
            }
        }
    }

    fn accept_visual_state(&mut self, state: &OdometryState) {
        self.visual_readiness = VisualReadiness::Continuous;
        self.advance_latest_device_time(state.session_id(), state.timestamp());
        if let Some(current) = self.current_map
            && current.binding.map_instance_id() != state.map_snapshot().instance_id()
        {
            self.invalidate_for_map_reset(
                current.binding.map_instance_id(),
                state.map_snapshot().instance_id(),
            );
        }
    }

    fn advance_latest_device_time(
        &mut self,
        session_id: DeviceSessionId,
        timestamp: DeviceTimestamp,
    ) {
        match self.latest_device_time {
            Some((current_session, current))
                if current_session == session_id && current >= timestamp => {}
            _ => self.latest_device_time = Some((session_id, timestamp)),
        }
    }

    fn invalidate_for_map_reset(
        &mut self,
        previous_map_instance_id: MapInstanceId,
        replacement_map_instance_id: MapInstanceId,
    ) {
        if let GoalBinding::Bound(_) = self.goal {
            let next_goal_generation = self.reserve_goal_generation().ok();
            self.goal = GoalBinding::Invalidated {
                previous_map_instance_id,
                replacement_map_instance_id,
            };
            if let Some(next_goal_generation) = next_goal_generation {
                self.goal_generation = next_goal_generation;
            }
        }
        self.path = None;
        self.plan_fault = None;
    }

    fn resolve_arrival(
        &self,
        offset: super::ingress::NavigationClockOffset,
    ) -> Result<HostMonotonicTimestamp, NavigationReplayClockError> {
        NavigationReplayClock::new(self.clock_epoch.origin()).resolve(offset)
    }

    fn ensure_open(&self) -> Result<(), CoordinatorAdmissionError<J::Error>> {
        match self.latch {
            Some(latch) => Err(CoordinatorAdmissionError::Latched(latch)),
            None => Ok(()),
        }
    }

    fn reserve_goal_generation(&mut self) -> Result<u64, CoordinatorAdmissionError<J::Error>> {
        match self.goal_generation.checked_add(1) {
            Some(next) => Ok(next),
            None => {
                if self.latch.is_none() {
                    self.latch = Some(CoordinatorLatch::GoalGenerationExhausted);
                }
                Err(CoordinatorAdmissionError::Latched(
                    CoordinatorLatch::GoalGenerationExhausted,
                ))
            }
        }
    }

    fn reserve_motion_mode_generation(&mut self) -> Result<u64, CoordinatorLatch> {
        match self.motion_mode_generation.checked_add(1) {
            Some(next) => Ok(next),
            None => {
                if self.latch.is_none() {
                    self.latch = Some(CoordinatorLatch::MotionModeGenerationExhausted);
                }
                Err(CoordinatorLatch::MotionModeGenerationExhausted)
            }
        }
    }

    fn append(
        &mut self,
        event: NavigationIngressEvent,
    ) -> Result<NavigationIngressRecord, CoordinatorAdmissionError<J::Error>> {
        self.ensure_open()?;
        match self.journal.append_event(event) {
            Ok(record) => Ok(record),
            Err(source) => {
                self.latch = Some(CoordinatorLatch::JournalFailure);
                Err(CoordinatorAdmissionError::Journal(source))
            }
        }
    }
}

fn motion_state_from_odometry(
    state: &OdometryState,
    epoch: NavigationEpochV1,
    observed_at: HostMonotonicTimestamp,
) -> Result<OdomMotionStateV1, MotionValueError> {
    let pose = state.base_to_odom();
    let twist = state.twist();
    OdomMotionStateV1::parse(
        OdomMotionStateV1Dto {
            schema_version: ODOM_MOTION_STATE_V1,
            observed_at_host_ns: observed_at.as_nanos(),
            x_m: pose.source_origin_x_in_destination_m(),
            y_m: pose.source_origin_y_in_destination_m(),
            yaw_rad: pose.source_yaw_in_destination_rad(),
            odom_velocity_x_mps: twist.linear_x_in_odom_m_per_sec(),
            odom_velocity_y_mps: twist.linear_y_in_odom_m_per_sec(),
            yaw_rate_rad_s: twist.yaw_rate_rad_per_sec(),
        },
        epoch,
    )
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use robot_protocol::CommandSequence;

    use super::*;
    use crate::dense::occupancy::{
        DepthCameraModel, DepthRangeMeters, DepthToTrackingCamera, HeightRangeMeters,
        OccupancyCell, OccupancyGridGeometry, WorldToOccupancy,
    };
    use crate::map::{MapSnapshot, SlamMap};
    use crate::{
        DepthImage, DepthObservation, Frame, FrameDimensions, FrameId, PairingWindowNs,
        PinholeIntrinsics, Pose, SensorAccuracy, SensorId, StereoObservation, StereoPair,
        WorldToCamera,
    };

    use super::super::frames::LocalCostmapFrame;
    use super::super::frontier::{
        FrontierExplorer, FrontierExplorerConfig, FrontierSearchOutcome, FrontierUnknownDirection,
    };
    use super::super::goal_input::MapPointGoalSelectionDto;
    use super::super::ingress::{
        NavigationIngressCapacity, NavigationRecordingId, PendingVisualAttemptIngress,
    };
    use super::super::local_costmap::{
        LocalCostmapCell, LocalCostmapConfig, LocalCostmapQuery, TrackingCameraToBase,
    };
    use super::super::manual_drive::{
        MANUAL_DRIVE_COMMAND_V1, ManualAuthoritySnapshot, ManualDriveCommandDto,
        ManualDriveCommandKindDto, ManualDriveConfigV1, ManualDriveConfigV1Dto, ManualDriveCore,
        ManualDriveOutput,
    };
    use super::super::manual_reference::{
        FrontierYawScanBudgetV1, FrontierYawScanCommandV1, FrontierYawTurnDirectionV1,
    };
    use super::super::mpc::{
        ClockFault, HostMonotonicClockFailure, HostMonotonicClockReadError, MPC_CONFIG_V1,
        MpcConfigV1Dto, MpcFailureKind, MpcSolveProgressV1, MpcSolver, PLANT_MODEL_V1,
        PlantEvidenceV1Dto, PlantModelV1, PlantModelV1Dto, PlantValidityEnvelopeV1Dto,
        WheelPlantV1Dto,
    };
    use super::super::odometry::{
        PlanarOdometryConfig, PlanarOdometryConfigDto, RawImuCalibrationDto,
    };
    use super::super::reference::{
        FORWARD_MOST_NEAREST_SEGMENT_V1, PATH_REFERENCE_CONFIG_V1, PathReferenceConfigV1,
        PathReferenceConfigV1Dto,
    };
    use super::super::safety::{SafetyDecisionOutcome, SafetyStopCause, ShadowSafetySupervisor};
    use super::super::shadow_command::{
        ShadowCommandConfig, ShadowCommandConfigDto, ShadowCommandDisposition,
    };
    #[cfg(all(feature = "agent-runtime", unix))]
    use super::super::{NanoBoundaryFrontierExplorer, NanoExploreBoundaryMeters};

    const IDENTITY_3: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    const SESSION_RAW: u64 = 7;
    const VISUAL_TIMESTAMP_NS: i64 = 100;
    const VISUAL_HOST_NS: u64 = 1_000;
    const DEPTH_HOST_NS: u64 = 1_100;

    fn host(nanos: u64) -> HostMonotonicTimestamp {
        HostMonotonicTimestamp::from_nanos(nanos)
    }

    fn session() -> DeviceSessionId {
        DeviceSessionId::try_new(SESSION_RAW).expect("nonzero test session")
    }

    fn clock_epoch() -> NavigationClockEpoch {
        NavigationClockEpoch::new(host(900))
    }

    fn journal() -> NavigationIngressLog {
        NavigationIngressLog::new(
            NavigationRecordingId::try_new([0x5a; 16]).expect("nonzero recording ID"),
            NavigationIngressCapacity::try_new(256).expect("bounded test journal"),
        )
    }

    fn odometry() -> PlanarOdometry {
        let config = PlanarOdometryConfig::parse(PlanarOdometryConfigDto {
            raw_imu_calibration: RawImuCalibrationDto {
                format_version: 1,
                source_id: "fixture://coordinator-imu".to_owned(),
                content_id: "fixture-content-v1".to_owned(),
                gyro_affine: IDENTITY_3,
                gyro_bias_native_rad_per_sec: [0.0; 3],
                accel_affine: IDENTITY_3,
                accel_bias_native_m_per_sec2: [0.0; 3],
                native_imu_to_base_rotation: IDENTITY_3,
            },
            tracking_camera_to_base: TrackingCameraToBase::new(Pose::identity()),
            world_to_occupancy: WorldToOccupancy::try_new(IDENTITY_3, [0.0; 3])
                .expect("identity world-to-map fixture"),
            max_visual_interval: Duration::from_secs(2),
            max_visual_linear_speed_m_per_sec: 10.0,
            max_visual_yaw_rate_rad_per_sec: 5.0,
            max_calibrated_yaw_rate_rad_per_sec: 5.0,
            minimum_gyro_accuracy: SensorAccuracy::Low,
            max_vertical_increment_m: 0.1,
            max_relative_roll_pitch_increment_rad: 0.2,
            max_absolute_map_roll_pitch_rad: 0.2,
            max_imu_gap: Duration::from_millis(200),
            max_prediction_age: Duration::from_millis(500),
            max_host_observation_age: Duration::from_secs(1),
            max_history_bracket_gap: Duration::from_secs(2),
            gyro_history_capacity: 16,
            pose_history_capacity: 8,
        })
        .expect("valid explicit odometry fixture");
        PlanarOdometry::new(config)
    }

    fn optical_to_tracking(camera_x_m: f32) -> DepthToTrackingCamera {
        DepthToTrackingCamera::new(
            Pose::try_from_rt(
                [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
                [camera_x_m, 0.0, 0.5],
            )
            .expect("proper optical-to-planar fixture"),
        )
    }

    fn local_costmap(max_age_ns: u64, camera_x_m: f32) -> LocalCostmap {
        let dimensions = FrameDimensions::try_new(9, 1).expect("nonzero depth dimensions");
        let camera = DepthCameraModel::new(
            PinholeIntrinsics::try_new(4.0, 4.0, 4.0, 0.0).expect("finite intrinsics"),
            dimensions,
            optical_to_tracking(camera_x_m),
        );
        let config = LocalCostmapConfig::try_new(
            OccupancyGridGeometry::try_new(0.25, [-2.0, -2.0], 20, 16, 320)
                .expect("bounded local grid"),
            camera,
            TrackingCameraToBase::new(Pose::identity()),
            HeightRangeMeters::try_new(0.1, 1.0).expect("obstacle height range"),
            DepthRangeMeters::try_new(0.1, 4.0).expect("metric depth range"),
            1,
            0.05,
            0.0,
            Duration::from_nanos(max_age_ns),
        )
        .expect("valid explicit local-costmap fixture");
        LocalCostmap::try_new(config, session()).expect("allocated local costmap")
    }

    fn mpc_config() -> MpcConfigV1 {
        MpcConfigV1::parse(MpcConfigV1Dto {
            schema_version: MPC_CONFIG_V1,
            horizon_steps: 4,
            step_period_s: 0.05,
            integration_substeps: 2,
            optimization_iterations: 1,
            candidates_per_wheel: 3,
            max_rollout_evaluations: 10_000,
            initial_search_radius_percent: 10,
            search_radius_decay_numerator: 1,
            search_radius_decay_denominator: 2,
            left_pwm_min_percent: -50,
            left_pwm_max_percent: 50,
            right_pwm_min_percent: -50,
            right_pwm_max_percent: 50,
            left_max_slew_percent_per_step: 200,
            right_max_slew_percent_per_step: 200,
            max_integration_tube_radius_m: 1.0,
            position_cost_per_m2: 1.0,
            heading_cost_per_rad2: 1.0,
            forward_velocity_cost_s2_per_m2: 0.0,
            yaw_rate_cost_s2_per_rad2: 0.0,
            pwm_cost_per_percent2: 0.001,
            slew_cost_per_percent2: 0.001,
            terminal_state_cost_multiplier: 2.0,
        })
        .expect("valid bounded MPC fixture")
    }

    fn plant() -> PlantModelV1 {
        PlantModelV1::parse(PlantModelV1Dto {
            schema_version: PLANT_MODEL_V1,
            model_id: "coordinator-test-plant".to_owned(),
            model_version: 1,
            sample_period_s: 0.05,
            wheelbase_m: 0.5,
            left: WheelPlantV1Dto {
                velocity_gain_mps_per_pwm_percent: 0.01,
                time_constant_s: 0.2,
            },
            right: WheelPlantV1Dto {
                velocity_gain_mps_per_pwm_percent: 0.01,
                time_constant_s: 0.2,
            },
            validity: PlantValidityEnvelopeV1Dto {
                left_pwm_min_percent: -100,
                left_pwm_max_percent: 100,
                right_pwm_min_percent: -100,
                right_pwm_max_percent: 100,
                left_velocity_min_mps: -2.0,
                left_velocity_max_mps: 2.0,
                right_velocity_min_mps: -2.0,
                right_velocity_max_mps: 2.0,
                max_abs_yaw_rate_rad_s: 2.0,
                max_abs_lateral_velocity_mps: 0.2,
            },
            evidence: PlantEvidenceV1Dto::SyntheticFixture {
                fixture_id: "coordinator-unit".to_owned(),
                generator_id: "explicit-hand-fixture".to_owned(),
            },
        })
        .expect("valid synthetic plant fixture")
    }

    fn safety(config: MpcConfigV1) -> ShadowSafetySupervisor {
        let solver = MpcSolver::new(plant(), config).expect("compatible solver fixture");
        let command_config = ShadowCommandConfig::parse(ShadowCommandConfigDto {
            lease_ms: 100,
            retained_records: 64,
            initial_sequence: 11,
        })
        .expect("bounded shadow-command fixture");
        ShadowSafetySupervisor::try_new(solver, command_config)
            .expect("reserved safety diagnostics")
    }

    fn reference_builder(maximum_projection_distance_m: f64) -> PathReferenceBuilderV1 {
        let config = PathReferenceConfigV1::parse(PathReferenceConfigV1Dto {
            schema_version: PATH_REFERENCE_CONFIG_V1,
            maximum_path_points: 256,
            minimum_segment_length_m: 1.0e-9,
            maximum_path_length_m: 100.0,
            maximum_projection_distance_m,
            target_forward_speed_mps: 0.5,
            goal_stop_distance_m: 0.25,
            maximum_abs_yaw_rate_rad_s: 2.0,
            nearest_segment_tie_policy: FORWARD_MOST_NEAREST_SEGMENT_V1,
        })
        .expect("bounded reference fixture");
        PathReferenceBuilderV1::new(config)
    }

    fn coordinator_with_sink<J: NavigationIngressSink>(
        sink: J,
        max_age_ns: u64,
        solver_budget_ns: u64,
        projection_distance_m: f64,
        camera_x_m: f32,
    ) -> ShadowNavigationCoordinator<J> {
        let config = mpc_config();
        ShadowNavigationCoordinator::new(
            clock_epoch(),
            sink,
            MapPoint::try_new(1.0, 0.0).expect("finite goal"),
            odometry(),
            local_costmap(max_age_ns, camera_x_m),
            GlobalPlannerConfig::try_new(0.05, super::super::UnknownSpacePolicy::Blocked)
                .expect("planner clearance"),
            reference_builder(projection_distance_m),
            config,
            SolverBudgetNs::try_new(solver_budget_ns).expect("nonzero solver budget"),
            safety(config),
        )
    }

    fn coordinator(
        max_age_ns: u64,
        solver_budget_ns: u64,
        projection_distance_m: f64,
    ) -> ShadowNavigationCoordinator<NavigationIngressLog> {
        coordinator_with_sink(
            journal(),
            max_age_ns,
            solver_budget_ns,
            projection_distance_m,
            0.0,
        )
    }

    fn coordinator_without_goal() -> ShadowNavigationCoordinator<NavigationIngressLog> {
        let config = mpc_config();
        ShadowNavigationCoordinator::new_without_goal(
            clock_epoch(),
            journal(),
            odometry(),
            local_costmap(1_000, 0.0),
            GlobalPlannerConfig::try_new(0.05, super::super::UnknownSpacePolicy::Blocked)
                .expect("planner clearance"),
            reference_builder(2.0),
            config,
            SolverBudgetNs::try_new(10).expect("nonzero solver budget"),
            safety(config),
        )
    }

    fn visual_ingress(
        frame_id: u64,
        timestamp_ns: i64,
        host_ns: u64,
        outcome: VisualAttemptOutcome,
    ) -> VisualAttemptIngress {
        let left = Frame::new(
            SensorId::StereoLeft,
            FrameId::new(frame_id),
            Timestamp::from_nanos(timestamp_ns),
            1,
            1,
            vec![0],
        )
        .expect("left fixture frame");
        let right = Frame::new(
            SensorId::StereoRight,
            FrameId::new(frame_id + 1_000),
            Timestamp::from_nanos(timestamp_ns),
            1,
            1,
            vec![0],
        )
        .expect("right fixture frame");
        let pair = StereoPair::try_new(
            left,
            right,
            PairingWindowNs::try_from_u64(1).expect("nonzero pairing window"),
        )
        .expect("exact-time stereo fixture");
        let observation =
            StereoObservation::parse(session(), host(host_ns), pair).expect("typed stereo fixture");
        PendingVisualAttemptIngress::from_observation(clock_epoch(), &observation)
            .expect("arrival after clock epoch")
            .complete(outcome)
    }

    fn localization(
        frame_id: u64,
        timestamp_ns: i64,
        map: MapSnapshot,
        world_to_camera: WorldToCamera,
    ) -> MapLocalization {
        MapLocalization::new(
            VisualFrameStamp::new(FrameId::new(frame_id), Timestamp::from_nanos(timestamp_ns)),
            map,
            world_to_camera,
        )
    }

    fn anchor<J: NavigationIngressSink>(
        coordinator: &mut ShadowNavigationCoordinator<J>,
        map: MapSnapshot,
    ) where
        J::Error: fmt::Debug,
    {
        let ingress = visual_ingress(
            1,
            VISUAL_TIMESTAMP_NS,
            VISUAL_HOST_NS,
            VisualAttemptOutcome::LocalizationOnly,
        );
        let admission = VisualAdmission::localization_only(
            ingress,
            localization(1, VISUAL_TIMESTAMP_NS, map, WorldToCamera::identity()),
        )
        .expect("matching typed anchor");
        assert!(matches!(
            coordinator
                .accept_visual(admission, host(VISUAL_HOST_NS))
                .expect("journaled anchor"),
            VisualAdmissionOutcome::Reanchored(_)
        ));
    }

    fn occupancy(map: MapSnapshot, revision: u64) -> OccupancyGridSnapshot {
        let geometry = OccupancyGridGeometry::try_new(0.25, [-2.0, -2.0], 20, 16, 320)
            .expect("bounded global grid");
        OccupancyGridSnapshot::from_test_cells(
            geometry,
            &vec![OccupancyCell::Free; geometry.cell_count()],
            map.instance_id(),
            revision,
        )
    }

    fn accept_map<J: NavigationIngressSink>(
        coordinator: &mut ShadowNavigationCoordinator<J>,
        map: MapSnapshot,
        revision: u64,
        host_ns: u64,
    ) -> GlobalMapAdmissionOutcome
    where
        J::Error: fmt::Debug,
    {
        let snapshot = occupancy(map, revision);
        coordinator
            .accept_global_map(
                host(host_ns),
                Timestamp::from_nanos(VISUAL_TIMESTAMP_NS),
                &snapshot,
            )
            .expect("accepted exact occupancy revision")
    }

    fn goal_selection(
        map_epoch_id: RecordedMapEpochId,
        revision: u64,
        x_m: f64,
        y_m: f64,
    ) -> MapPointGoalSelection {
        MapPointGoalSelection::parse(MapPointGoalSelectionDto {
            map_epoch_id: map_epoch_id.as_u64(),
            displayed_revision: revision,
            x_m,
            y_m,
        })
        .expect("typed goal selection fixture")
    }

    fn depth_observation(timestamp_ns: i64, host_ns: u64) -> DepthObservation {
        let image = DepthImage::new(
            FrameId::new(5_000 + u64::try_from(timestamp_ns).unwrap_or(0)),
            Timestamp::from_nanos(timestamp_ns),
            9,
            1,
            vec![2.5; 9],
        )
        .expect("metric depth fixture");
        DepthObservation::parse(session(), host(host_ns), image).expect("typed depth fixture")
    }

    fn accept_aligned_depth<J: NavigationIngressSink>(
        coordinator: &mut ShadowNavigationCoordinator<J>,
    ) where
        J::Error: fmt::Debug,
    {
        assert!(matches!(
            coordinator
                .accept_depth(
                    depth_observation(VISUAL_TIMESTAMP_NS, DEPTH_HOST_NS),
                    host(DEPTH_HOST_NS),
                )
                .expect("journaled depth"),
            DepthAdmissionOutcome::Updated(LocalCostmapUpdateOutcome::Accepted { .. })
        ));
    }

    struct FixedClock(HostMonotonicTimestamp);

    impl HostMonotonicClock for FixedClock {
        fn try_now(&mut self) -> Result<HostMonotonicTimestamp, HostMonotonicClockReadError> {
            Ok(self.0)
        }
    }

    struct ReadFailingClock(HostMonotonicClockReadError);

    impl HostMonotonicClock for ReadFailingClock {
        fn try_now(&mut self) -> Result<HostMonotonicTimestamp, HostMonotonicClockReadError> {
            Err(self.0)
        }
    }

    fn ready_fixture(
        max_age_ns: u64,
        budget_ns: u64,
        projection_distance_m: f64,
    ) -> ShadowNavigationCoordinator<NavigationIngressLog> {
        let mut coordinator = coordinator(max_age_ns, budget_ns, projection_distance_m);
        let map = SlamMap::new().snapshot();
        anchor(&mut coordinator, map);
        let outcome = accept_map(&mut coordinator, map, 1, 1_050);
        assert!(matches!(
            outcome.planning(),
            GlobalPlanningOutcome::Planned(_)
        ));
        accept_aligned_depth(&mut coordinator);
        coordinator
    }

    fn manual_command(
        lease: NonZeroU64,
        sequence: u64,
        forward_velocity_mps: f64,
        yaw_rate_rad_s: f64,
        received_at_ns: u64,
        deadman_timeout_ns: u64,
    ) -> ManualMpcCommandV1 {
        let config = ManualDriveConfigV1::parse(ManualDriveConfigV1Dto {
            schema_version: 1,
            maximum_abs_forward_velocity_mps: 1.0,
            maximum_abs_yaw_rate_rad_s: 2.0,
            maximum_command_age_ns: deadman_timeout_ns,
            deadman_timeout_ns,
        })
        .expect("manual fixture limits");
        let mut core = ManualDriveCore::new(config, lease, host(received_at_ns));
        let output = core.ingest(
            ManualDriveCommandDto {
                schema_version: MANUAL_DRIVE_COMMAND_V1,
                authority_lease_id: lease,
                sequence,
                command: ManualDriveCommandKindDto::Velocity {
                    forward_velocity_mps,
                    yaw_rate_rad_s,
                },
            },
            host(received_at_ns),
            host(received_at_ns),
            ManualAuthoritySnapshot::active_manual(lease, host(10_000)),
        );
        let ManualDriveOutput::Accepted(accepted) = output else {
            panic!("manual fixture must be admitted")
        };
        ManualMpcCommandV1::try_from_accepted(accepted).expect("typed manual MPC command")
    }

    fn manual_core(
        lease: NonZeroU64,
        created_at_ns: u64,
        deadman_timeout_ns: u64,
    ) -> ManualDriveCore<NonZeroU64> {
        let config = ManualDriveConfigV1::parse(ManualDriveConfigV1Dto {
            schema_version: 1,
            maximum_abs_forward_velocity_mps: 1.0,
            maximum_abs_yaw_rate_rad_s: 2.0,
            maximum_command_age_ns: deadman_timeout_ns,
            deadman_timeout_ns,
        })
        .expect("manual fixture limits");
        ManualDriveCore::new(config, lease, host(created_at_ns))
    }

    #[test]
    fn manual_mode_is_exclusive_and_uses_the_full_safety_pipeline() {
        let mut coordinator = ready_fixture(1_000, 10, 2.0);
        let lease = NonZeroU64::new(21).unwrap();
        assert!(matches!(
            coordinator.enter_manual_mode(lease),
            Err(CoordinatorMotionModeError::NotMappingOnly {
                actual: CoordinatorMotionModeV1::PointGoal
            })
        ));

        coordinator.clear_goal();
        coordinator.enter_manual_mode(lease).unwrap();
        assert!(coordinator.global_path().is_none());
        let command = manual_command(lease, 4, 0.2, 0.1, 1_110, 500);
        let tick = host(1_120);
        let outcome = coordinator
            .tick_manual(tick, command, &mut FixedClock(tick))
            .expect("manual attempt retains exactly one safety decision");
        assert!(outcome.control_tick_journaled());
        assert!(outcome.blocker().is_none(), "manual request reached safety");
        assert_eq!(outcome.decision().motor_packets_sent().get(), 0);

        let ordinary_tick = host(1_130);
        let ordinary = coordinator
            .tick(ordinary_tick, &mut FixedClock(ordinary_tick))
            .expect("point-path entry point fail-closes in manual mode");
        assert!(matches!(
            ordinary.blocker(),
            Some(CoordinatorTickBlocker::MotionModeMismatch {
                expected: CoordinatorMotionModeV1::PointGoal,
                actual: CoordinatorMotionModeV1::Manual { authority_lease_id }
            }) if *authority_lease_id == lease
        ));
        assert!(ordinary.decision().record().pwm().is_stop());
    }

    #[test]
    fn manual_tick_rejects_wrong_lease_expiry_and_stale_depth() {
        let lease = NonZeroU64::new(31).unwrap();
        let other = NonZeroU64::new(32).unwrap();

        let mut wrong_lease = ready_fixture(1_000, 10, 2.0);
        wrong_lease.clear_goal();
        wrong_lease.enter_manual_mode(lease).unwrap();
        let command = manual_command(other, 1, 0.2, 0.0, 1_110, 500);
        let tick = host(1_120);
        let outcome = wrong_lease
            .tick_manual(tick, command, &mut FixedClock(tick))
            .unwrap();
        assert!(matches!(
            outcome.blocker(),
            Some(CoordinatorTickBlocker::DirectAuthorityLeaseMismatch { bound, command })
                if *bound == lease && *command == other
        ));

        let mut expired = ready_fixture(1_000, 10, 2.0);
        expired.clear_goal();
        expired.enter_manual_mode(lease).unwrap();
        let command = manual_command(lease, 2, 0.2, 0.0, 1_110, 10);
        let tick = command.valid_through_exclusive();
        let outcome = expired
            .tick_manual(tick, command, &mut FixedClock(tick))
            .unwrap();
        assert!(matches!(
            outcome.blocker(),
            Some(CoordinatorTickBlocker::ManualReference(
                ManualReferenceBuildError::CommandExpired { .. }
            ))
        ));
        assert!(outcome.decision().record().pwm().is_stop());

        let mut stale = ready_fixture(20, 10, 2.0);
        stale.clear_goal();
        stale.enter_manual_mode(lease).unwrap();
        let command = manual_command(lease, 3, -0.2, 0.0, 1_110, 500);
        let tick = host(DEPTH_HOST_NS + 21);
        let outcome = stale
            .tick_manual(tick, command, &mut FixedClock(tick))
            .unwrap();
        assert!(matches!(
            outcome.blocker(),
            Some(CoordinatorTickBlocker::LocalCostmapExpired)
        ));
        assert!(outcome.decision().record().pwm().is_stop());
    }

    #[test]
    fn manual_command_order_is_monotonic_and_public_goal_clear_cannot_release_authority() {
        let mut coordinator = ready_fixture(1_000, 10, 2.0);
        coordinator.clear_goal();
        let lease = NonZeroU64::new(33).unwrap();
        coordinator.enter_manual_mode(lease).unwrap();
        let accepted = manual_command(lease, 5, 0.2, 0.0, 1_110, 1_000);

        for tick_ns in [1_120, 1_130] {
            let tick = host(tick_ns);
            let outcome = coordinator
                .tick_manual(tick, accepted, &mut FixedClock(tick))
                .unwrap();
            assert!(
                outcome.blocker().is_none(),
                "an exact periodic repeat is admissible"
            );
        }

        let regressed = manual_command(lease, 4, 0.2, 0.0, 1_110, 1_000);
        let tick = host(1_140);
        let outcome = coordinator
            .tick_manual(tick, regressed, &mut FixedClock(tick))
            .unwrap();
        assert!(matches!(
            outcome.blocker(),
            Some(CoordinatorTickBlocker::ManualCommandSequenceRegression {
                previous: 5,
                command: 4,
            })
        ));

        let conflicting = manual_command(lease, 5, 0.3, 0.0, 1_110, 1_000);
        let tick = host(1_150);
        let outcome = coordinator
            .tick_manual(tick, conflicting, &mut FixedClock(tick))
            .unwrap();
        assert!(matches!(
            outcome.blocker(),
            Some(CoordinatorTickBlocker::ManualCommandIdentityConflict { sequence: 5 })
        ));

        let advanced = manual_command(lease, 6, 0.3, 0.0, 1_110, 1_000);
        let tick = host(1_160);
        assert!(
            coordinator
                .tick_manual(tick, advanced, &mut FixedClock(tick))
                .unwrap()
                .blocker()
                .is_none()
        );

        coordinator.clear_goal();
        assert_eq!(
            coordinator.motion_mode(),
            CoordinatorMotionModeV1::Manual {
                authority_lease_id: lease,
            }
        );
        assert!(matches!(
            coordinator.leave_direct_mode(NonZeroU64::new(34).unwrap()),
            Err(CoordinatorMotionModeError::AuthorityLeaseMismatch { bound, supplied })
                if bound == lease && supplied.get() == 34
        ));
        coordinator.leave_direct_mode(lease).unwrap();
        assert_eq!(
            coordinator.motion_mode(),
            CoordinatorMotionModeV1::MappingOnly
        );
    }

    #[test]
    fn explicit_stop_and_deadman_keep_the_real_manual_reason_on_the_zero_decision() {
        let lease = NonZeroU64::new(35).unwrap();
        let mut coordinator = ready_fixture(1_000, 10, 2.0);
        coordinator.clear_goal();
        coordinator.enter_manual_mode(lease).unwrap();
        let mut core = manual_core(lease, 1_110, 100);
        let authority = ManualAuthoritySnapshot::active_manual(lease, host(10_000));

        let velocity = core.ingest(
            ManualDriveCommandDto {
                schema_version: MANUAL_DRIVE_COMMAND_V1,
                authority_lease_id: lease,
                sequence: 5,
                command: ManualDriveCommandKindDto::Velocity {
                    forward_velocity_mps: 0.2,
                    yaw_rate_rad_s: 0.0,
                },
            },
            host(1_110),
            host(1_110),
            authority,
        );
        let ManualDriveOutput::Accepted(velocity) = velocity else {
            panic!("velocity fixture")
        };
        let command = ManualMpcCommandV1::try_from_accepted(velocity).unwrap();
        let tick = host(1_120);
        assert!(
            coordinator
                .tick_manual(tick, command, &mut FixedClock(tick))
                .unwrap()
                .blocker()
                .is_none()
        );

        let stop = core.ingest(
            ManualDriveCommandDto {
                schema_version: MANUAL_DRIVE_COMMAND_V1,
                authority_lease_id: lease,
                sequence: 6,
                command: ManualDriveCommandKindDto::Stop,
            },
            host(1_130),
            host(1_130),
            authority,
        );
        let ManualDriveOutput::Accepted(stop) = stop else {
            panic!("stop fixture")
        };
        let stop = stop.into_explicit_stop().expect("explicit-stop domain");
        let tick = host(1_130);
        let outcome = coordinator
            .tick_manual_explicit_stop(tick, stop, &mut FixedClock(tick))
            .unwrap();
        assert!(matches!(
            outcome.blocker(),
            Some(CoordinatorTickBlocker::ManualExplicitStop {
                authority_lease_id,
                sequence: 6,
            }) if *authority_lease_id == lease
        ));
        assert!(outcome.decision().record().pwm().is_stop());
        assert!(matches!(
            outcome.decision().outcome(),
            SafetyDecisionOutcome::Stopped(stopped)
                if matches!(
                    stopped.cause(),
                    SafetyStopCause::NotReady(SafetyNotReadyReason::ManualDriveStopped)
                )
        ));

        let stopped = core.tick(host(1_230), authority);
        let ManualDriveOutput::Stopped(stopped) = stopped else {
            panic!("deadman equality must stop")
        };
        let tick = host(1_230);
        let outcome = coordinator
            .tick_manual_stopped(tick, stopped, &mut FixedClock(tick))
            .unwrap();
        assert!(matches!(
            outcome.blocker(),
            Some(CoordinatorTickBlocker::ManualDriveStopped {
                authority_lease_id,
                cause: ManualDriveStopCause::DeadmanExpired { sequence, .. },
            }) if *authority_lease_id == lease && sequence.get() == 6
        ));
        assert!(outcome.decision().record().pwm().is_stop());
    }

    #[test]
    fn manual_solver_deadline_cannot_outlive_exclusive_command_validity() {
        let mut coordinator = ready_fixture(1_000, 10, 2.0);
        coordinator.clear_goal();
        let lease = NonZeroU64::new(35).unwrap();
        coordinator.enter_manual_mode(lease).unwrap();
        let command = manual_command(lease, 1, 0.2, 0.0, 1_110, 15);
        let tick = host(1_120);
        let deadline = command.valid_through_exclusive();
        assert_eq!(deadline, host(1_125));

        let outcome = coordinator
            .tick_manual(tick, command, &mut FixedClock(deadline))
            .expect("authority deadline must produce one stopped decision");
        let SafetyDecisionOutcome::Stopped(stopped) = outcome.decision().outcome() else {
            panic!("a solve observed at command expiry cannot issue a controller decision")
        };
        let SafetyStopCause::Solver(source) = stopped.cause() else {
            panic!("the exact solver deadline failure must be retained")
        };
        assert!(matches!(
            source.kind(),
            MpcFailureKind::Clock(HostMonotonicClockFailure::Fault(
                ClockFault::DeadlineReached {
                    deadline: actual_deadline,
                    observed_at,
                }
            )) if *actual_deadline == deadline && *observed_at == deadline
        ));
        assert_eq!(source.request().deadline(), deadline);
        assert!(outcome.decision().record().pwm().is_stop());
    }

    #[test]
    fn manual_tick_stops_immediately_when_visual_localization_is_lost() {
        let mut coordinator = ready_fixture(1_000, 10, 2.0);
        coordinator.clear_goal();
        let lease = NonZeroU64::new(41).unwrap();
        coordinator.enter_manual_mode(lease).unwrap();
        let broken = VisualAdmission::no_localization(visual_ingress(
            2,
            200,
            1_115,
            VisualAttemptOutcome::NoLocalization,
        ))
        .unwrap();
        coordinator.accept_visual(broken, host(1_115)).unwrap();

        let command = manual_command(lease, 1, 0.2, 0.0, 1_110, 500);
        let tick = host(1_120);
        let outcome = coordinator
            .tick_manual(tick, command, &mut FixedClock(tick))
            .unwrap();
        assert!(matches!(
            outcome.blocker(),
            Some(CoordinatorTickBlocker::VisualOdometryRejected)
        ));
        assert!(outcome.decision().record().pwm().is_stop());
    }

    #[test]
    fn frontier_in_place_scan_uses_yaw_mpc_and_rejects_a_newer_map_revision() {
        let mut coordinator = coordinator_without_goal();
        let map = SlamMap::new().snapshot();
        anchor(&mut coordinator, map);
        let geometry = OccupancyGridGeometry::try_new(0.25, [-2.0, -2.0], 20, 16, 320)
            .expect("bounded frontier map");
        let mut cells = vec![OccupancyCell::Unknown; geometry.cell_count()];
        cells[8 * 20 + 8] = OccupancyCell::Free;
        let snapshot =
            OccupancyGridSnapshot::from_test_cells(geometry, &cells, map.instance_id(), 1);
        coordinator
            .accept_global_map(
                host(1_050),
                Timestamp::from_nanos(VISUAL_TIMESTAMP_NS),
                &snapshot,
            )
            .unwrap();
        accept_aligned_depth(&mut coordinator);
        let start = coordinator.plan_start_for_snapshot(&snapshot).unwrap();
        let mut explorer = FrontierExplorer::try_new(
            &snapshot,
            FrontierExplorerConfig::try_new(0.0, 320, 320, 2_560).unwrap(),
        )
        .unwrap();
        let FrontierSearchOutcome::InPlaceScanRequired(scan) = explorer.select(start).unwrap()
        else {
            panic!("single observed start cell must require a deliberate scan")
        };

        let lease = NonZeroU64::new(51).unwrap();
        coordinator.enter_frontier_yaw_mode(lease).unwrap();
        let budget =
            FrontierYawScanBudgetV1::try_new(1.0, std::f64::consts::PI, 0.0, 5_000_000_000)
                .unwrap();
        let command = FrontierYawScanCommandV1::try_new(
            lease,
            1,
            scan,
            FrontierUnknownDirection::PositiveMapY,
            FrontierYawTurnDirectionV1::CounterClockwise,
            host(1_110),
            host(10_000_000_000),
            budget,
        )
        .unwrap();
        let tick = host(1_120);
        let outcome = coordinator
            .tick_frontier_yaw(tick, command, &mut FixedClock(tick))
            .unwrap();
        assert!(outcome.blocker().is_none(), "yaw reference reached safety");
        assert_eq!(outcome.decision().motor_packets_sent().get(), 0);

        let newer = OccupancyGridSnapshot::from_test_cells(geometry, &cells, map.instance_id(), 2);
        coordinator
            .accept_global_map(
                host(1_130),
                Timestamp::from_nanos(VISUAL_TIMESTAMP_NS),
                &newer,
            )
            .unwrap();
        let tick = host(1_140);
        let outcome = coordinator
            .tick_frontier_yaw(tick, command, &mut FixedClock(tick))
            .unwrap();
        assert!(matches!(
            outcome.blocker(),
            Some(CoordinatorTickBlocker::FrontierMapRevisionMismatch {
                expected_revision: 2,
                actual_revision: 1,
                ..
            })
        ));
        assert!(outcome.decision().record().pwm().is_stop());
    }

    #[test]
    fn frontier_command_order_allows_exact_repeats_but_rejects_replay_and_conflict() {
        let mut coordinator = coordinator_without_goal();
        let map = SlamMap::new().snapshot();
        anchor(&mut coordinator, map);
        let geometry = OccupancyGridGeometry::try_new(0.25, [-2.0, -2.0], 20, 16, 320)
            .expect("bounded frontier map");
        let mut cells = vec![OccupancyCell::Unknown; geometry.cell_count()];
        cells[8 * 20 + 8] = OccupancyCell::Free;
        let snapshot =
            OccupancyGridSnapshot::from_test_cells(geometry, &cells, map.instance_id(), 1);
        coordinator
            .accept_global_map(
                host(1_050),
                Timestamp::from_nanos(VISUAL_TIMESTAMP_NS),
                &snapshot,
            )
            .unwrap();
        accept_aligned_depth(&mut coordinator);
        let start = coordinator.plan_start_for_snapshot(&snapshot).unwrap();
        let mut explorer = FrontierExplorer::try_new(
            &snapshot,
            FrontierExplorerConfig::try_new(0.0, 320, 320, 2_560).unwrap(),
        )
        .unwrap();
        let FrontierSearchOutcome::InPlaceScanRequired(scan) = explorer.select(start).unwrap()
        else {
            panic!("single observed start cell must require a deliberate scan")
        };
        let lease = NonZeroU64::new(52).unwrap();
        coordinator.enter_frontier_yaw_mode(lease).unwrap();
        let budget =
            FrontierYawScanBudgetV1::try_new(1.0, std::f64::consts::PI, 0.0, 5_000_000_000)
                .unwrap();
        let command = |sequence, turn_direction| {
            FrontierYawScanCommandV1::try_new(
                lease,
                sequence,
                scan,
                FrontierUnknownDirection::PositiveMapY,
                turn_direction,
                host(1_110),
                host(10_000_000_000),
                budget,
            )
            .unwrap()
        };
        let accepted = command(5, FrontierYawTurnDirectionV1::CounterClockwise);
        for tick_ns in [1_120, 1_130] {
            let tick = host(tick_ns);
            assert!(
                coordinator
                    .tick_frontier_yaw(tick, accepted, &mut FixedClock(tick))
                    .unwrap()
                    .blocker()
                    .is_none()
            );
        }

        let tick = host(1_140);
        let outcome = coordinator
            .tick_frontier_yaw(
                tick,
                command(4, FrontierYawTurnDirectionV1::CounterClockwise),
                &mut FixedClock(tick),
            )
            .unwrap();
        assert!(matches!(
            outcome.blocker(),
            Some(CoordinatorTickBlocker::FrontierCommandSequenceRegression {
                previous: 5,
                command: 4,
            })
        ));

        let tick = host(1_150);
        let outcome = coordinator
            .tick_frontier_yaw(
                tick,
                command(5, FrontierYawTurnDirectionV1::Clockwise),
                &mut FixedClock(tick),
            )
            .unwrap();
        assert!(matches!(
            outcome.blocker(),
            Some(CoordinatorTickBlocker::FrontierCommandIdentityConflict { sequence: 5 })
        ));

        let tick = host(1_160);
        assert!(
            coordinator
                .tick_frontier_yaw(
                    tick,
                    command(6, FrontierYawTurnDirectionV1::CounterClockwise),
                    &mut FixedClock(tick),
                )
                .unwrap()
                .blocker()
                .is_none()
        );
    }

    #[test]
    fn ready_tick_clock_read_failure_is_a_journaled_transport_free_stop() {
        let mut coordinator = ready_fixture(1_000, 10, 2.0);
        let injected = HostMonotonicClockReadError::ElapsedNanosecondsOutOfRange {
            elapsed_nanoseconds: 18_446_744_073_709_551_616,
        };
        let outcome = coordinator
            .tick(host(1_120), &mut ReadFailingClock(injected))
            .expect("clock read failure must remain an admitted STOP decision");

        let SafetyDecisionOutcome::Stopped(stopped) = outcome.decision().outcome() else {
            panic!("clock read failure cannot produce a controller decision")
        };
        let SafetyStopCause::Solver(source) = stopped.cause() else {
            panic!("coordinator must retain the exact solver clock failure")
        };
        assert_eq!(source.progress(), MpcSolveProgressV1::NotStarted);
        assert!(matches!(
            source.kind(),
            MpcFailureKind::Clock(HostMonotonicClockFailure::Read(actual))
                if *actual == injected
        ));
        assert_eq!(
            outcome.decision().record().disposition(),
            ShadowCommandDisposition::FailClosedStop
        );
        assert!(outcome.decision().record().pwm().is_stop());
        assert!(outcome.control_tick_journaled());
        assert!(coordinator.safety().last_success_trajectory().is_none());
        assert_eq!(outcome.decision().motor_packets_sent().get(), 0);
    }

    #[test]
    fn click_goal_reaches_mpc_but_forward_only_unknown_clearance_records_stop() {
        let mut coordinator = ready_fixture(1_000, 10, 2.0);
        let path = coordinator.global_path().expect("exact revision path");
        assert_eq!(path.map_revision(), 1);
        assert_eq!(
            path.points().last().copied(),
            Some(MapPoint::try_new(1.0, 0.0).unwrap())
        );

        let tick = host(1_120);
        let outcome = coordinator
            .tick(tick, &mut FixedClock(tick))
            .expect("one recorded safety decision");
        assert!(outcome.control_tick_journaled());
        assert!(
            outcome.blocker().is_none(),
            "the full request reached safety"
        );
        assert_eq!(outcome.decision().motor_packets_sent().get(), 0);
        assert_eq!(
            outcome.decision().record().disposition(),
            ShadowCommandDisposition::FailClosedStop
        );
        assert!(matches!(
            outcome.decision().outcome(),
            SafetyDecisionOutcome::Stopped(stopped)
                if matches!(stopped.cause(), SafetyStopCause::Solver(failure)
                    if matches!(failure.kind(), MpcFailureKind::CollisionBlocked {
                        horizon_step: 0,
                        integration_substep: 0,
                    }))
        ));
    }

    #[test]
    fn embodied_footprint_is_free_but_forward_only_unseen_clearance_stays_blocked() {
        let mut coordinator = coordinator(1_000, 10, 2.0);
        let map = SlamMap::new().snapshot();
        anchor(&mut coordinator, map);
        accept_aligned_depth(&mut coordinator);
        let view = coordinator
            .local_costmap()
            .view_at(host(DEPTH_HOST_NS))
            .expect("monotonic view");
        assert_eq!(
            view.cell_at_local(PlanarPoint::<LocalCostmapFrame>::origin()),
            LocalCostmapQuery::InBounds(LocalCostmapCell::Free)
        );
        let counts = view
            .class_ids()
            .iter()
            .fold([0_usize; 4], |mut counts, id| {
                counts[usize::from(*id)] += 1;
                counts
            });
        assert_eq!(counts, [224, 13, 76, 7]);
    }

    #[test]
    fn newest_unaligned_depth_forces_stop_instead_of_reusing_old_grid() {
        let mut coordinator = ready_fixture(1_000, 10, 2.0);
        let outcome = coordinator
            .accept_depth(depth_observation(99, 1_105), host(1_105))
            .expect("unaligned frame is still journaled once");
        assert!(matches!(
            outcome,
            DepthAdmissionOutcome::PoseUnavailable(OdometryUnavailable::QueryBeforeHistory)
        ));
        let tick = host(1_120);
        let outcome = coordinator
            .tick(tick, &mut FixedClock(tick))
            .expect("explicit stop decision");
        assert!(matches!(
            outcome.blocker(),
            Some(CoordinatorTickBlocker::DepthUnaligned)
        ));
        assert!(outcome.decision().record().pwm().is_stop());
    }

    #[test]
    fn expired_depth_forces_stop_before_solver_submission() {
        let mut coordinator = ready_fixture(20, 10, 2.0);
        let tick = host(DEPTH_HOST_NS + 21);
        let outcome = coordinator
            .tick(tick, &mut FixedClock(tick))
            .expect("explicit stale-depth stop");
        assert!(matches!(
            outcome.blocker(),
            Some(CoordinatorTickBlocker::LocalCostmapExpired)
        ));
        assert!(outcome.decision().record().pwm().is_stop());
    }

    #[test]
    fn map_reset_invalidates_goal_without_silent_rebinding() {
        let mut coordinator = coordinator(1_000, 10, 2.0);
        let first = SlamMap::new().snapshot();
        anchor(&mut coordinator, first);
        accept_map(&mut coordinator, first, 1, 1_050);

        let second = SlamMap::new().snapshot();
        let ingress = visual_ingress(2, 200, 1_200, VisualAttemptOutcome::LocalizationOnly);
        let admission = VisualAdmission::localization_only(
            ingress,
            localization(2, 200, second, WorldToCamera::identity()),
        )
        .expect("second-map anchor identity");
        coordinator
            .accept_visual(admission, host(1_200))
            .expect("journaled map-reset anchor");
        let outcome = accept_map(&mut coordinator, second, 0, 1_210);
        assert!(outcome.started_new_epoch());
        assert_eq!(
            coordinator.goal_state(),
            NavigationGoalState::InvalidatedByMapReset {
                previous_map_instance_id: first.instance_id(),
                replacement_map_instance_id: second.instance_id(),
            }
        );
        assert!(matches!(
            outcome.planning(),
            GlobalPlanningOutcome::Deferred(_)
        ));
        let goal_records = coordinator
            .journal()
            .records()
            .iter()
            .filter(|record| matches!(record.event(), NavigationIngressEvent::PointGoal(_)))
            .count();
        assert_eq!(
            goal_records, 1,
            "the old click must not bind to the reset map"
        );
    }

    #[test]
    fn exact_current_map_click_replaces_goal_and_is_journaled_before_planning() {
        let mut coordinator = coordinator(1_000, 10, 2.0);
        let map = SlamMap::new().snapshot();
        anchor(&mut coordinator, map);
        let snapshot = occupancy(map, 1);
        coordinator
            .accept_global_map(host(1_050), Timestamp::from_nanos(100), &snapshot)
            .expect("initial map");
        let binding = coordinator
            .current_map_binding()
            .expect("current map binding");

        let outcome = coordinator
            .select_map_point_goal(
                host(1_060),
                goal_selection(binding.map_epoch_id(), 1, 1.5, 0.5),
                &snapshot,
            )
            .expect("exact displayed-map click");
        assert_eq!(
            outcome.goal_state(),
            NavigationGoalState::Bound {
                map_instance_id: map.instance_id(),
                selected_revision: 1,
            }
        );
        assert!(matches!(
            outcome.planning(),
            GlobalPlanningOutcome::Planned(_)
        ));
        let goals: Vec<_> = coordinator
            .journal()
            .records()
            .iter()
            .filter_map(|record| match record.event() {
                NavigationIngressEvent::PointGoal(goal) => Some(goal),
                _ => None,
            })
            .collect();
        assert_eq!(goals.len(), 2);
        assert_eq!(goals[1].map_epoch_id(), binding.map_epoch_id());
        assert_eq!(goals[1].selected_revision(), 1);
        assert_eq!(goals[1].point().as_array(), [1.5, 0.5]);
    }

    #[test]
    fn point_goal_preparation_is_non_mutating_and_commit_consumes_exact_snapshot_proof() {
        let mut coordinator = coordinator_without_goal();
        let map = SlamMap::new().snapshot();
        anchor(&mut coordinator, map);
        let snapshot = occupancy(map, 1);
        coordinator
            .accept_global_map(host(1_050), Timestamp::from_nanos(100), &snapshot)
            .expect("mapping-only map");
        let binding = coordinator.current_map_binding().expect("map binding");
        let records_before_prepare = coordinator.journal().len();

        let prepared = coordinator
            .prepare_map_point_goal(
                goal_selection(binding.map_epoch_id(), 1, 1.5, 0.5),
                &snapshot,
            )
            .expect("prepared exact click");
        assert_eq!(prepared.map_binding(), binding);
        assert_eq!(prepared.displayed_revision(), 1);
        assert_eq!(prepared.goal().point().as_array(), [1.5, 0.5]);
        assert_eq!(coordinator.journal().len(), records_before_prepare);
        assert_eq!(coordinator.goal_state(), NavigationGoalState::Unavailable);
        assert_eq!(
            coordinator.motion_mode(),
            CoordinatorMotionModeV1::MappingOnly
        );
        assert!(coordinator.global_path().is_none());

        let committed = coordinator
            .commit_prepared_map_point_goal(host(1_060), prepared)
            .expect("commit exact prepared click");
        assert!(matches!(
            committed.planning(),
            GlobalPlanningOutcome::Planned(_)
        ));
        assert_eq!(
            coordinator.goal_state(),
            NavigationGoalState::Bound {
                map_instance_id: map.instance_id(),
                selected_revision: 1,
            }
        );
        assert_eq!(coordinator.journal().len(), records_before_prepare + 1);
    }

    #[test]
    fn prepared_point_goal_fails_closed_after_map_or_goal_state_changes() {
        let mut coordinator = coordinator_without_goal();
        let map = SlamMap::new().snapshot();
        anchor(&mut coordinator, map);
        let first_snapshot = occupancy(map, 1);
        coordinator
            .accept_global_map(host(1_050), Timestamp::from_nanos(100), &first_snapshot)
            .expect("first mapping snapshot");
        let binding = coordinator.current_map_binding().expect("map binding");
        let stale_by_map = coordinator
            .prepare_map_point_goal(
                goal_selection(binding.map_epoch_id(), 1, 1.5, 0.5),
                &first_snapshot,
            )
            .expect("first prepared goal");

        let second_snapshot = occupancy(map, 2);
        coordinator
            .accept_global_map(host(1_060), Timestamp::from_nanos(101), &second_snapshot)
            .expect("newer mapping snapshot");
        let records_before_stale_commit = coordinator.journal().len();
        assert!(matches!(
            coordinator.commit_prepared_map_point_goal(host(1_061), stale_by_map),
            Err(CoordinatorAdmissionError::PointGoalPreparationStale {
                prepared_revision: 1,
                current_revision: Some(2),
                ..
            })
        ));
        assert_eq!(coordinator.journal().len(), records_before_stale_commit);
        assert_eq!(coordinator.goal_state(), NavigationGoalState::Unavailable);

        let prepared_first = coordinator
            .prepare_map_point_goal(
                goal_selection(binding.map_epoch_id(), 2, 1.5, 0.5),
                &second_snapshot,
            )
            .expect("first concurrent preparation");
        let prepared_second = coordinator
            .prepare_map_point_goal(
                goal_selection(binding.map_epoch_id(), 2, 2.5, 0.5),
                &second_snapshot,
            )
            .expect("second concurrent preparation");
        coordinator
            .commit_prepared_map_point_goal(host(1_062), prepared_first)
            .expect("first preparation wins");
        let records_before_losing_commit = coordinator.journal().len();
        assert!(matches!(
            coordinator.commit_prepared_map_point_goal(host(1_063), prepared_second),
            Err(CoordinatorAdmissionError::PointGoalPreparationStale { .. })
        ));
        assert_eq!(coordinator.journal().len(), records_before_losing_commit);
        assert_eq!(
            coordinator
                .current_goal()
                .expect("winning goal remains")
                .point()
                .as_array(),
            [1.5, 0.5]
        );
    }

    #[test]
    fn prepared_point_goal_cannot_cross_a_direct_mode_transition() {
        let mut coordinator = coordinator_without_goal();
        let map = SlamMap::new().snapshot();
        anchor(&mut coordinator, map);
        let snapshot = occupancy(map, 1);
        coordinator
            .accept_global_map(host(1_050), Timestamp::from_nanos(100), &snapshot)
            .expect("mapping snapshot");
        let binding = coordinator.current_map_binding().expect("map binding");
        let prepared = coordinator
            .prepare_map_point_goal(
                goal_selection(binding.map_epoch_id(), 1, 1.5, 0.5),
                &snapshot,
            )
            .expect("prepared goal");
        coordinator
            .enter_manual_mode(NonZeroU64::new(17).expect("lease"))
            .expect("separately authorized direct-mode fixture");
        let records_before_commit = coordinator.journal().len();

        assert!(matches!(
            coordinator.commit_prepared_map_point_goal(host(1_060), prepared),
            Err(CoordinatorAdmissionError::PointGoalPreparationStale {
                prepared_motion_mode: CoordinatorMotionModeV1::MappingOnly,
                current_motion_mode: CoordinatorMotionModeV1::Manual { .. },
                ..
            })
        ));
        assert_eq!(coordinator.journal().len(), records_before_commit);
        assert_eq!(coordinator.goal_state(), NavigationGoalState::Unavailable);
        assert!(matches!(
            coordinator.motion_mode(),
            CoordinatorMotionModeV1::Manual { .. }
        ));
    }

    #[test]
    fn prepared_point_goal_detects_motion_mode_enter_leave_aba() {
        let mut coordinator = coordinator_without_goal();
        let map = SlamMap::new().snapshot();
        anchor(&mut coordinator, map);
        let snapshot = occupancy(map, 1);
        coordinator
            .accept_global_map(host(1_050), Timestamp::from_nanos(100), &snapshot)
            .expect("mapping snapshot");
        let binding = coordinator.current_map_binding().expect("map binding");
        let prepared = coordinator
            .prepare_map_point_goal(
                goal_selection(binding.map_epoch_id(), 1, 1.5, 0.5),
                &snapshot,
            )
            .expect("prepared goal");
        let lease = NonZeroU64::new(17).expect("lease");
        coordinator
            .enter_manual_mode(lease)
            .expect("enter direct-mode fixture");
        coordinator
            .leave_direct_mode(lease)
            .expect("leave direct-mode fixture");
        assert_eq!(
            coordinator.motion_mode(),
            CoordinatorMotionModeV1::MappingOnly,
            "visible motion mode returned to its prepared value"
        );
        let records_before_commit = coordinator.journal().len();

        assert!(matches!(
            coordinator.commit_prepared_map_point_goal(host(1_060), prepared),
            Err(CoordinatorAdmissionError::PointGoalPreparationStale {
                prepared_motion_mode_generation: 0,
                current_motion_mode_generation: 2,
                ..
            })
        ));
        assert_eq!(coordinator.journal().len(), records_before_commit);
        assert_eq!(coordinator.goal_state(), NavigationGoalState::Unavailable);
    }

    #[cfg(all(feature = "agent-runtime", unix))]
    #[test]
    fn bounded_frontier_constraint_survives_coordinator_commit_and_replan() {
        let planner_config =
            GlobalPlannerConfig::try_new(0.0, super::super::UnknownSpacePolicy::Traversable)
                .expect("point-robot planner");
        let mpc = mpc_config();
        let mut coordinator = ShadowNavigationCoordinator::new_without_goal(
            clock_epoch(),
            journal(),
            odometry(),
            local_costmap(1_000, 0.0),
            planner_config,
            reference_builder(2.0),
            mpc,
            SolverBudgetNs::try_new(10).expect("nonzero solver budget"),
            safety(mpc),
        );
        let map = SlamMap::new().snapshot();
        anchor(&mut coordinator, map);
        let geometry =
            OccupancyGridGeometry::try_new(1.0, [-3.0, -3.0], 7, 7, 49).expect("global grid");
        let mut cells = vec![OccupancyCell::Free; geometry.cell_count()];
        cells[3 * 7 + 5] = OccupancyCell::Unknown;
        let first_snapshot =
            OccupancyGridSnapshot::from_test_cells(geometry, &cells, map.instance_id(), 1);
        coordinator
            .accept_global_map(host(1_050), Timestamp::from_nanos(100), &first_snapshot)
            .expect("first global map");
        let start = coordinator
            .plan_start_for_snapshot(&first_snapshot)
            .expect("map-bound start");
        let boundary =
            NanoExploreBoundaryMeters::try_new(-0.5, -0.5, 2.5, 1.5).expect("operator boundary");
        let frontier_config =
            FrontierExplorerConfig::try_new(0.0, 49, 49, 392).expect("frontier resources");
        let mut explorer =
            NanoBoundaryFrontierExplorer::try_new(&first_snapshot, frontier_config, boundary)
                .expect("bounded explorer");
        let FrontierSearchOutcome::Selected(frontier) =
            explorer.select(start).expect("frontier selection")
        else {
            panic!("fixture must produce a positive-distance frontier")
        };
        let traversal_boundary = frontier
            .traversal_boundary()
            .expect("Nano frontier retains traversal boundary");
        coordinator
            .select_frontier_goal(host(1_060), frontier, &first_snapshot)
            .expect("commit bounded frontier");
        let first_path = coordinator.global_path().expect("first bounded path");
        assert_eq!(first_path.traversal_boundary(), Some(traversal_boundary));
        assert!(
            first_path
                .points()
                .iter()
                .copied()
                .all(|point| traversal_boundary.contains(point))
        );

        let second_snapshot =
            OccupancyGridSnapshot::from_test_cells(geometry, &cells, map.instance_id(), 2);
        coordinator
            .accept_global_map(host(1_070), Timestamp::from_nanos(101), &second_snapshot)
            .expect("newer global map replans");
        let replanned = coordinator.global_path().expect("replanned bounded path");
        assert_eq!(replanned.traversal_boundary(), Some(traversal_boundary));
        assert!(
            replanned
                .points()
                .iter()
                .copied()
                .all(|point| traversal_boundary.contains(point))
        );
    }

    #[test]
    fn prepared_point_goal_detects_goal_change_clear_aba_and_generation_exhaustion() {
        let mut coordinator = coordinator_without_goal();
        let map = SlamMap::new().snapshot();
        anchor(&mut coordinator, map);
        let snapshot = occupancy(map, 1);
        coordinator
            .accept_global_map(host(1_050), Timestamp::from_nanos(100), &snapshot)
            .expect("mapping snapshot");
        let binding = coordinator.current_map_binding().expect("map binding");
        let prepared_before_aba = coordinator
            .prepare_map_point_goal(
                goal_selection(binding.map_epoch_id(), 1, 1.5, 0.5),
                &snapshot,
            )
            .expect("prepared before ABA");
        coordinator
            .select_map_point_goal(
                host(1_060),
                goal_selection(binding.map_epoch_id(), 1, 2.5, 0.5),
                &snapshot,
            )
            .expect("intervening goal");
        coordinator.clear_goal();
        assert_eq!(
            coordinator.goal_state(),
            NavigationGoalState::Unavailable,
            "visible goal state returned to its prepared value"
        );
        let records_before_aba_commit = coordinator.journal().len();
        assert!(matches!(
            coordinator.commit_prepared_map_point_goal(host(1_061), prepared_before_aba),
            Err(CoordinatorAdmissionError::PointGoalPreparationStale {
                prepared_goal_generation: 0,
                current_goal_generation: 2,
                ..
            })
        ));
        assert_eq!(coordinator.journal().len(), records_before_aba_commit);

        let prepared_at_limit = coordinator
            .prepare_map_point_goal(
                goal_selection(binding.map_epoch_id(), 1, 1.5, 0.5),
                &snapshot,
            )
            .expect("prepared at finite generation");
        coordinator.goal_generation = u64::MAX;
        let records_before_exhausted_commit = coordinator.journal().len();
        assert!(matches!(
            coordinator.commit_prepared_map_point_goal(host(1_062), prepared_at_limit),
            Err(CoordinatorAdmissionError::PointGoalPreparationStale {
                current_goal_generation: u64::MAX,
                ..
            })
        ));
        assert_eq!(coordinator.journal().len(), records_before_exhausted_commit);

        let prepared_at_exact_limit = coordinator
            .prepare_map_point_goal(
                goal_selection(binding.map_epoch_id(), 1, 1.5, 0.5),
                &snapshot,
            )
            .expect("prepare observes exact generation limit");
        assert!(matches!(
            coordinator.commit_prepared_map_point_goal(host(1_063), prepared_at_exact_limit),
            Err(CoordinatorAdmissionError::Latched(
                CoordinatorLatch::GoalGenerationExhausted
            ))
        ));
        assert_eq!(
            coordinator.latch(),
            Some(CoordinatorLatch::GoalGenerationExhausted)
        );
        assert_eq!(coordinator.journal().len(), records_before_exhausted_commit);
    }

    #[test]
    fn mapping_without_a_goal_stays_stopped_until_an_exact_current_map_click() {
        let mut coordinator = coordinator_without_goal();
        assert_eq!(coordinator.goal_state(), NavigationGoalState::Unavailable);

        let map = SlamMap::new().snapshot();
        anchor(&mut coordinator, map);
        let snapshot = occupancy(map, 1);
        let admitted = coordinator
            .accept_global_map(host(1_050), Timestamp::from_nanos(100), &snapshot)
            .expect("mapping-only snapshot");
        assert_eq!(admitted.goal_state(), NavigationGoalState::Unavailable);
        assert!(matches!(
            admitted.planning(),
            GlobalPlanningOutcome::Deferred(StoredPlanFault::Planning)
        ));

        let start = coordinator
            .plan_start_for_snapshot(&snapshot)
            .expect("fresh current plan start");
        assert_eq!(start.map_instance_id(), map.instance_id());
        assert_eq!(start.map_revision(), 1);

        accept_aligned_depth(&mut coordinator);
        let tick = host(1_120);
        let stopped = coordinator
            .tick(tick, &mut FixedClock(tick))
            .expect("goal-free tick is a recorded stop");
        assert!(matches!(
            stopped.blocker(),
            Some(CoordinatorTickBlocker::GoalUnavailable)
        ));
        assert!(stopped.decision().record().pwm().is_stop());

        let binding = coordinator.current_map_binding().expect("map binding");
        let selected = coordinator
            .select_map_point_goal(
                host(1_121),
                goal_selection(binding.map_epoch_id(), 1, 1.5, 0.5),
                &snapshot,
            )
            .expect("exact click activates a goal");
        assert!(matches!(
            selected.planning(),
            GlobalPlanningOutcome::Planned(_)
        ));
    }

    #[test]
    fn clearing_a_bound_goal_removes_the_path_and_next_tick_records_stop() {
        let mut coordinator = coordinator(1_000, 10, 2.0);
        let map = SlamMap::new().snapshot();
        anchor(&mut coordinator, map);
        let snapshot = occupancy(map, 1);
        coordinator
            .accept_global_map(host(1_050), Timestamp::from_nanos(100), &snapshot)
            .expect("global map");
        accept_aligned_depth(&mut coordinator);
        assert!(coordinator.current_goal().is_some());
        assert!(coordinator.global_path().is_some());

        coordinator.clear_goal();
        assert_eq!(coordinator.goal_state(), NavigationGoalState::Unavailable);
        assert!(coordinator.current_goal().is_none());
        assert!(coordinator.global_path().is_none());

        let tick = host(1_120);
        let outcome = coordinator
            .tick(tick, &mut FixedClock(tick))
            .expect("goal-free tick records a stop");
        assert!(matches!(
            outcome.blocker(),
            Some(CoordinatorTickBlocker::GoalUnavailable)
        ));
        assert!(outcome.decision().record().pwm().is_stop());
    }

    #[test]
    fn stale_revision_or_epoch_click_cannot_change_goal_or_journal() {
        let mut coordinator = coordinator(1_000, 10, 2.0);
        let map = SlamMap::new().snapshot();
        anchor(&mut coordinator, map);
        let first_snapshot = occupancy(map, 1);
        coordinator
            .accept_global_map(host(1_050), Timestamp::from_nanos(100), &first_snapshot)
            .expect("first map");
        let binding = coordinator.current_map_binding().expect("binding");
        let second_snapshot = occupancy(map, 2);
        coordinator
            .accept_global_map(host(1_060), Timestamp::from_nanos(101), &second_snapshot)
            .expect("newer map");
        let before_state = coordinator.goal_state();
        let before_records = coordinator.journal().len();

        assert!(matches!(
            coordinator.select_map_point_goal(
                host(1_070),
                goal_selection(binding.map_epoch_id(), 1, 0.0, 0.0),
                &first_snapshot,
            ),
            Err(CoordinatorAdmissionError::GoalDisplayedRevisionMismatch {
                displayed: 1,
                current: 2,
            })
        ));
        let future_epoch = RecordedMapEpochId::try_new(binding.map_epoch_id().as_u64() + 1)
            .expect("next epoch fixture");
        assert!(matches!(
            coordinator.select_map_point_goal(
                host(1_071),
                goal_selection(future_epoch, 2, 0.0, 0.0),
                &second_snapshot,
            ),
            Err(CoordinatorAdmissionError::GoalMapEpochMismatch { .. })
        ));
        assert_eq!(coordinator.goal_state(), before_state);
        assert_eq!(coordinator.journal().len(), before_records);
    }

    #[test]
    fn mismatched_snapshot_identity_or_revision_cannot_back_a_click() {
        let mut coordinator = coordinator(1_000, 10, 2.0);
        let map = SlamMap::new().snapshot();
        anchor(&mut coordinator, map);
        let current_snapshot = occupancy(map, 4);
        coordinator
            .accept_global_map(host(1_050), Timestamp::from_nanos(100), &current_snapshot)
            .expect("current map");
        let binding = coordinator.current_map_binding().expect("binding");
        let selection = goal_selection(binding.map_epoch_id(), 4, 0.0, 0.0);
        let foreign = occupancy(SlamMap::new().snapshot(), 4);
        assert!(matches!(
            coordinator.select_map_point_goal(host(1_060), selection, &foreign),
            Err(CoordinatorAdmissionError::GoalSnapshotMapMismatch { .. })
        ));
        let wrong_revision = occupancy(map, 5);
        assert!(matches!(
            coordinator.select_map_point_goal(host(1_061), selection, &wrong_revision),
            Err(CoordinatorAdmissionError::GoalSnapshotRevisionMismatch {
                displayed: 4,
                snapshot: 5,
            })
        ));
    }

    #[test]
    fn click_before_any_accepted_map_is_rejected_without_a_fallback_binding() {
        let mut coordinator = coordinator(1_000, 10, 2.0);
        let snapshot = occupancy(SlamMap::new().snapshot(), 1);
        let selection = goal_selection(RecordedMapEpochId::try_new(1).expect("epoch"), 1, 0.0, 0.0);
        assert!(matches!(
            coordinator.select_map_point_goal(host(1_000), selection, &snapshot),
            Err(CoordinatorAdmissionError::NoCurrentMapForGoal)
        ));
        assert!(coordinator.journal().is_empty());
    }

    #[test]
    fn exact_path_revision_mismatch_is_rejected_at_tick() {
        let mut coordinator = ready_fixture(1_000, 10, 2.0);
        coordinator
            .current_map
            .as_mut()
            .expect("current map")
            .revision = 2;
        let tick = host(1_120);
        let outcome = coordinator
            .tick(tick, &mut FixedClock(tick))
            .expect("revision mismatch stops");
        assert!(matches!(
            outcome.blocker(),
            Some(CoordinatorTickBlocker::PathInvalidated)
        ));
        assert!(outcome.decision().record().pwm().is_stop());
    }

    #[test]
    fn far_map_correction_surfaces_reference_projection_failure() {
        let mut coordinator = ready_fixture(1_000, 10, 0.25);
        let map = coordinator
            .odometry()
            .current()
            .expect("anchored")
            .map_snapshot();
        let corrected_world_to_camera = WorldToCamera::from_legacy_pose(
            Pose::try_from_rt(Pose::identity().rotation(), [0.0, -2.0, 0.0])
                .expect("finite correction"),
        );
        let ingress = visual_ingress(
            1,
            VISUAL_TIMESTAMP_NS,
            1_110,
            VisualAttemptOutcome::LocalizationOnly,
        );
        let admission = VisualAdmission::localization_only(
            ingress,
            localization(1, VISUAL_TIMESTAMP_NS, map, corrected_world_to_camera),
        )
        .expect("same-stamp correction");
        coordinator
            .accept_visual(admission, host(1_110))
            .expect("journaled correction");
        let tick = host(1_120);
        let outcome = coordinator
            .tick(tick, &mut FixedClock(tick))
            .expect("reference failure stops");
        assert!(matches!(
            outcome.blocker(),
            Some(CoordinatorTickBlocker::Reference(
                PathReferenceBuildError::ProjectionDistanceExceeded { .. }
            ))
        ));
        assert!(outcome.decision().record().pwm().is_stop());
    }

    #[test]
    fn depth_validity_boundary_cannot_create_a_zero_duration_deadline() {
        let mut coordinator = ready_fixture(100, 100, 2.0);
        let tick = host(DEPTH_HOST_NS + 100);
        let outcome = coordinator
            .tick(tick, &mut FixedClock(tick))
            .expect("deadline rejection records stop");
        assert!(outcome.blocker().is_none());
        assert!(matches!(
            outcome.decision().outcome(),
            SafetyDecisionOutcome::Stopped(stopped)
                if matches!(stopped.cause(), SafetyStopCause::Request(
                    super::super::mpc::MpcRequestParseError::NonFutureDeadline { .. }
                ))
        ));
        assert!(outcome.decision().record().pwm().is_stop());
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct InjectedJournalFailure;

    impl fmt::Display for InjectedJournalFailure {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("injected journal failure")
        }
    }

    impl std::error::Error for InjectedJournalFailure {}

    struct FailingSink {
        inner: NavigationIngressLog,
        fail: bool,
    }

    impl NavigationIngressSink for FailingSink {
        type Error = InjectedJournalFailure;

        fn append_event(
            &mut self,
            event: NavigationIngressEvent,
        ) -> Result<NavigationIngressRecord, Self::Error> {
            if self.fail {
                Err(InjectedJournalFailure)
            } else {
                self.inner.push(event).map_err(|_| InjectedJournalFailure)
            }
        }
    }

    #[test]
    fn tick_journal_failure_latches_but_still_records_exactly_one_stop_per_tick() {
        let sink = FailingSink {
            inner: journal(),
            fail: false,
        };
        let mut coordinator = coordinator_with_sink(sink, 1_000, 10, 2.0, 0.0);
        let map = SlamMap::new().snapshot();
        anchor(&mut coordinator, map);
        accept_map(&mut coordinator, map, 1, 1_050);
        accept_aligned_depth(&mut coordinator);
        coordinator.journal_mut().fail = true;

        let first_tick = host(1_120);
        let first = coordinator
            .tick(first_tick, &mut FixedClock(first_tick))
            .expect("journal failure still yields stop evidence");
        assert!(!first.control_tick_journaled());
        assert_eq!(first.journal_error(), Some(&InjectedJournalFailure));
        assert!(matches!(
            first.blocker(),
            Some(CoordinatorTickBlocker::JournalLatched)
        ));
        assert!(first.decision().record().pwm().is_stop());

        let second_tick = host(1_130);
        let second = coordinator
            .tick(second_tick, &mut FixedClock(second_tick))
            .expect("latched tick still yields one stop");
        assert!(!second.control_tick_journaled());
        assert!(second.journal_error().is_none());
        assert!(second.decision().record().pwm().is_stop());
        assert_eq!(coordinator.safety().shadow_session().retained_len(), 2);
        assert_eq!(
            coordinator
                .safety()
                .shadow_session()
                .latest()
                .expect("second stop")
                .command()
                .sequence(),
            CommandSequence::new(12)
        );
    }

    #[test]
    fn tick_before_recording_epoch_retains_exact_boundary_error_and_stops() {
        let mut coordinator = coordinator(1_000, 10, 2.0);
        let tick = host(899);
        let outcome = coordinator
            .tick(tick, &mut FixedClock(tick))
            .expect("invalid scheduling time still records stop evidence");
        assert!(!outcome.control_tick_journaled());
        assert!(matches!(
            outcome.blocker(),
            Some(CoordinatorTickBlocker::ControlTickBoundary(
                NavigationIngressBoundaryError::HostTimeBeforeClockEpoch {
                    origin_ns: 900,
                    timestamp_ns: 899,
                }
            ))
        ));
        assert_eq!(coordinator.latch(), Some(CoordinatorLatch::ClockBoundary));
        assert!(outcome.decision().record().pwm().is_stop());
    }

    #[test]
    fn map_before_visual_anchor_is_explicitly_deferred_until_a_later_revision() {
        let mut coordinator = coordinator(1_000, 10, 2.0);
        let map = SlamMap::new().snapshot();
        let outcome = accept_map(&mut coordinator, map, 1, 1_050);
        assert!(matches!(
            outcome.planning(),
            GlobalPlanningOutcome::Deferred(StoredPlanFault::OdometryUnavailable(
                OdometryUnavailable::NotAnchored
            ))
        ));
        anchor(&mut coordinator, map);
        assert!(coordinator.global_path().is_none());
        let outcome = accept_map(&mut coordinator, map, 2, 1_060);
        assert!(matches!(
            outcome.planning(),
            GlobalPlanningOutcome::Planned(_)
        ));
    }

    #[test]
    fn consuming_coordinator_returns_stream_owner_for_finalization() {
        let coordinator = coordinator(1_000, 10, 2.0);
        let journal = coordinator.into_journal();
        assert!(journal.is_empty());
    }
}
