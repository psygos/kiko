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
use super::global_planner::{
    GlobalPath, GlobalPlanError, GlobalPlanIdentity, GlobalPlanner, GlobalPlannerConfig, MapPoint,
    PlanStart, PointGoal,
};
use super::ingress::{
    AcceptedDepthIngress, AcceptedGlobalMapIngress, ControlTickIngress, CurrentMapEpochBinding,
    MapPointGoalIngress, NavigationClockEpoch, NavigationIngressBoundaryError,
    NavigationIngressEvent, NavigationIngressLog, NavigationIngressRecord,
    NavigationIngressStreamWriteError, NavigationIngressWriteError, NavigationIngressWriter,
    NavigationMapEpochCoordinator, NavigationReplayClock, NavigationReplayClockError,
    RecordedImuReport, VisualAttemptIngress, VisualAttemptOutcome,
};
use super::local_costmap::{
    DepthFrameKey, LocalCostmap, LocalCostmapClockRegression, LocalCostmapError,
    LocalCostmapUpdateOutcome, LocalDepthObservation, LocalDepthObservationError,
};
use super::mpc::{
    HostMonotonicClock, MotionValueError, MpcConfigV1, NavigationEpochError, NavigationEpochV1,
    ODOM_MOTION_STATE_V1, OdomMotionStateV1, OdomMotionStateV1Dto,
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
}

#[derive(Debug)]
enum GoalBinding {
    Pending(MapPoint),
    Bound(PointGoal),
    Invalidated {
        previous_map_instance_id: MapInstanceId,
        replacement_map_instance_id: MapInstanceId,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CurrentOccupancyMap {
    binding: CurrentMapEpochBinding,
    revision: u64,
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
    MapRevisionNotIncreasing { previous: u64, actual: u64 },
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
            Self::Latched(_) | Self::SegmentIdExhausted | Self::MapRevisionNotIncreasing { .. } => {
                None
            }
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
            Self::Reference(_) => SafetyNotReadyReason::ReferenceUnavailable,
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

/// Pure host navigation owner.  It contains no transport handle, packet
/// encoder, callback, Rerun dependency, or encoder-derived state.
pub struct ShadowNavigationCoordinator<J: NavigationIngressSink> {
    clock_epoch: NavigationClockEpoch,
    journal: J,
    map_epochs: NavigationMapEpochCoordinator,
    current_map: Option<CurrentOccupancyMap>,
    goal: GoalBinding,
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
        Self {
            clock_epoch,
            journal,
            map_epochs: NavigationMapEpochCoordinator::new(),
            current_map: None,
            goal: GoalBinding::Pending(pending_goal),
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
            GoalBinding::Pending(_) => NavigationGoalState::PendingFirstMap,
            GoalBinding::Bound(goal) => NavigationGoalState::Bound {
                map_instance_id: goal.map_instance_id(),
                selected_revision: goal.selected_revision(),
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

    pub fn current_map_binding(&self) -> Option<CurrentMapEpochBinding> {
        self.current_map.map(|current| current.binding)
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
            let ingress = MapPointGoalIngress::parse(self.clock_epoch, host_arrival, binding, goal)
                .map_err(CoordinatorAdmissionError::Boundary)?;
            self.append(NavigationIngressEvent::PointGoal(ingress))?;
            self.goal = GoalBinding::Bound(goal);
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

    pub fn tick<C: HostMonotonicClock>(
        &mut self,
        tick: HostMonotonicTimestamp,
        clock: &mut C,
    ) -> Result<CoordinatorTickOutcome<J::Error>, CoordinatorTickError> {
        let mut control_tick_journaled = false;
        let mut journal_error = None;
        let journal_blocker = if self.latch.is_some() {
            Some(CoordinatorTickBlocker::JournalLatched)
        } else {
            match ControlTickIngress::parse(self.clock_epoch, tick) {
                Ok(event) => match self.append(NavigationIngressEvent::ControlTick(event)) {
                    Ok(_) => {
                        control_tick_journaled = true;
                        None
                    }
                    Err(CoordinatorAdmissionError::Journal(source)) => {
                        journal_error = Some(source);
                        Some(CoordinatorTickBlocker::JournalLatched)
                    }
                    Err(_) => Some(CoordinatorTickBlocker::JournalLatched),
                },
                Err(source) => {
                    self.latch = Some(CoordinatorLatch::ClockBoundary);
                    Some(CoordinatorTickBlocker::ControlTickBoundary(source))
                }
            }
        };
        if let Some(blocker) = journal_blocker {
            return self.stop_tick(tick, blocker, control_tick_journaled, journal_error, clock);
        }
        if let Some(blocker) = self.preflight_blocker(tick) {
            return self.stop_tick(tick, blocker, control_tick_journaled, None, clock);
        }

        self.ready_tick(tick, control_tick_journaled, clock)
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
            GoalBinding::Pending(_) => return Some(CoordinatorTickBlocker::GoalUnavailable),
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

    fn plan_snapshot(&mut self, snapshot: &OccupancyGridSnapshot) -> GlobalPlanningOutcome {
        let Some(goal) = (match self.goal {
            GoalBinding::Bound(goal) => Some(goal),
            GoalBinding::Pending(_) | GoalBinding::Invalidated { .. } => None,
        }) else {
            self.plan_fault = Some(StoredPlanFault::Planning);
            return GlobalPlanningOutcome::Deferred(StoredPlanFault::Planning);
        };
        let Some(state) = self.odometry.current() else {
            self.plan_fault = Some(StoredPlanFault::OdometryUnavailable(
                OdometryUnavailable::NotAnchored,
            ));
            return GlobalPlanningOutcome::Deferred(StoredPlanFault::OdometryUnavailable(
                OdometryUnavailable::NotAnchored,
            ));
        };
        if state.map_snapshot().instance_id() != snapshot.map_instance_id().expect("parsed map") {
            self.plan_fault = Some(StoredPlanFault::OdomMapMismatch);
            return GlobalPlanningOutcome::Deferred(StoredPlanFault::OdomMapMismatch);
        }
        let base_in_map = match state
            .odom_to_map()
            .compose(state.base_to_odom())
            .and_then(|base_to_map| base_to_map.transform_point(PlanarPoint::<BaseFrame>::origin()))
        {
            Ok(point) => point,
            Err(_) => {
                self.plan_fault = Some(StoredPlanFault::StartTransform);
                return GlobalPlanningOutcome::Deferred(StoredPlanFault::StartTransform);
            }
        };
        let start = match PlanStart::for_snapshot(base_in_map, snapshot) {
            Ok(start) => start,
            Err(source) => {
                self.plan_fault = Some(StoredPlanFault::Planning);
                return GlobalPlanningOutcome::Failed(source);
            }
        };
        let mut planner = match GlobalPlanner::try_new(snapshot, self.planner_config) {
            Ok(planner) => planner,
            Err(source) => {
                self.plan_fault = Some(StoredPlanFault::PlannerConstruction);
                return GlobalPlanningOutcome::Failed(source);
            }
        };
        match planner.plan(start, goal) {
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
            self.goal = GoalBinding::Invalidated {
                previous_map_instance_id,
                replacement_map_instance_id,
            };
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
    use super::super::ingress::{
        NavigationIngressCapacity, NavigationRecordingId, PendingVisualAttemptIngress,
    };
    use super::super::local_costmap::{
        LocalCostmapCell, LocalCostmapConfig, LocalCostmapQuery, TrackingCameraToBase,
    };
    use super::super::mpc::{
        HostMonotonicClockFailure, HostMonotonicClockReadError, MPC_CONFIG_V1, MpcConfigV1Dto,
        MpcFailureKind, MpcSolveProgressV1, MpcSolver, PLANT_MODEL_V1, PlantEvidenceV1Dto,
        PlantModelV1, PlantModelV1Dto, PlantValidityEnvelopeV1Dto, WheelPlantV1Dto,
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
