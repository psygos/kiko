use std::fmt;
use std::time::Duration;

use kiko_head_protocol::{
    ExactHeadTargetPose, FullTelemetry, HeadJoint, HeadPose, HeadPoseError, PositionAgreementError,
    PositionAgreementTicks, PositionTicks, PresentPosition, TelemetryParseError, TorqueSwitch,
    ValidatedPresentPosition, build_full_telemetry_read, build_goal_with_speed_write,
    build_natural_hold_frames, build_position_read, build_torque_switch_write,
};
use tokio::runtime::{Handle, TryCurrentError};
use tokio::sync::{mpsc, oneshot};
use tokio::task::{JoinError, JoinHandle};

use crate::config::{
    ConfiguredHeadPoseBounds, HeadPoseBoundsAdmissionError, HeadPoseWithinConfiguredBounds,
    HeadReturnPlan, HeadRuntimeConfig, ReturnToTargetConfig,
};
use crate::framing::{FrameReadError, read_response_frame};
use crate::motion::{
    FreshHeadTelemetrySet, HeadMotionError, HeadReturnAction, HeadReturnController,
    admit_stopped_return_start,
};
use crate::transport::{
    AsyncByteTransport, MonotonicClock, MonotonicTime, SerialConfigurationEvidence,
    SerialOpenError, SerialTransport, TokioClock, TransportFailure,
};

const ACTOR_MAILBOX_CAPACITY: usize = 1;

/// Deliberate opt-in required before the actor can enable servo torque.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PhysicalTorqueEnableConsent(());

impl PhysicalTorqueEnableConsent {
    /// Acknowledge that natural hold energises physical servos. No calibrated
    /// motion command is exposed by this crate.
    pub const fn explicitly_granted() -> Self {
        Self(())
    }
}

/// Separate deliberate opt-in for a command which changes physical pose.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PhysicalHeadMotionConsent(());

impl PhysicalHeadMotionConsent {
    pub const fn explicitly_granted() -> Self {
        Self(())
    }
}

/// Deliberate opt-in to omit the commissioning-only pre-observation
/// torque-disable transaction during a manifest-bound production handoff.
///
/// This consent makes no claim about the torque-switch register or physical
/// tension before this actor acquires the bus. It permits only a
/// tension-preserving takeover attempt: observe the present pose, admit it
/// against the parsed bounds and freshness budget, write that same pose as the
/// bounded hold goal, and enable or refresh torque.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProductionTensionPreservingTakeoverConsent(());

impl ProductionTensionPreservingTakeoverConsent {
    pub const fn explicitly_granted_for_manifest_bound_owner() -> Self {
        Self(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RuntimeStage {
    ObserveFirst,
    ObserveSecond,
    WriteObservedGoal,
    WriteTorqueLimit,
    EnableTorque,
    VerifyFirstStoppedPosition,
    VerifySecondStoppedPosition,
    HealthReadTelemetry,
    ReturnReadTelemetry,
    ReturnWriteWaypoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VerificationSample {
    First,
    Second,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArmingFreshnessCheck {
    BeforeConfigurationWrites,
    BeforeEnableWrite,
    AfterEnableWrite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WritePurpose {
    PositionReadRequest,
    TelemetryReadRequest,
    ObservedGoal,
    TorqueLimit,
    TorqueEnable,
    TorqueDisable,
    ReturnWaypoint,
}

/// Exact bounded retry history for one completed write.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WriteEvidence {
    attempts_used: u8,
    recovered_failures: Vec<TransportFailure>,
    completed_at: MonotonicTime,
}

impl WriteEvidence {
    pub const fn attempts_used(&self) -> u8 {
        self.attempts_used
    }

    pub fn recovered_failures(&self) -> impl Iterator<Item = &TransportFailure> {
        self.recovered_failures.iter()
    }

    pub const fn completed_at(&self) -> MonotonicTime {
        self.completed_at
    }
}

/// Typed response plus the request write and framing evidence that admitted it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResponseEvidence<T> {
    value: T,
    request_write: WriteEvidence,
    discarded_noise_bytes: u16,
    received_at: MonotonicTime,
}

impl<T> ResponseEvidence<T> {
    pub const fn value(&self) -> &T {
        &self.value
    }

    pub const fn request_write(&self) -> &WriteEvidence {
        &self.request_write
    }

    pub const fn discarded_noise_bytes(&self) -> u16 {
        self.discarded_noise_bytes
    }

    pub const fn received_at(&self) -> MonotonicTime {
        self.received_at
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PositionObservationEvidence {
    joint: HeadJoint,
    first: ResponseEvidence<PresentPosition>,
    second: ResponseEvidence<PresentPosition>,
    validated: ValidatedPresentPosition,
}

impl PositionObservationEvidence {
    pub const fn joint(&self) -> HeadJoint {
        self.joint
    }

    pub const fn first(&self) -> &ResponseEvidence<PresentPosition> {
        &self.first
    }

    pub const fn second(&self) -> &ResponseEvidence<PresentPosition> {
        &self.second
    }

    pub const fn validated(&self) -> ValidatedPresentPosition {
        self.validated
    }
}

/// Post-write full telemetry whose position agrees with the observed target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadbackEvidence {
    joint: HeadJoint,
    target: PositionTicks,
    first_target_difference_ticks: u16,
    second_target_difference_ticks: u16,
    stable_difference_ticks: u16,
    first: ResponseEvidence<FullTelemetry>,
    second: ResponseEvidence<FullTelemetry>,
}

impl ReadbackEvidence {
    pub const fn joint(&self) -> HeadJoint {
        self.joint
    }

    pub const fn target(&self) -> PositionTicks {
        self.target
    }

    pub const fn first_target_difference_ticks(&self) -> u16 {
        self.first_target_difference_ticks
    }

    pub const fn second_target_difference_ticks(&self) -> u16 {
        self.second_target_difference_ticks
    }

    pub const fn stable_difference_ticks(&self) -> u16 {
        self.stable_difference_ticks
    }

    pub const fn first(&self) -> &ResponseEvidence<FullTelemetry> {
        &self.first
    }

    pub const fn second(&self) -> &ResponseEvidence<FullTelemetry> {
        &self.second
    }
}

/// Exact startup torque-policy evidence.
///
/// `TensionPreservingTakeover` records only which transaction the host ran. It
/// deliberately does not claim that torque was enabled, disabled, or actively
/// holding before the actor acquired exclusive serial ownership.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeadStartupTorqueEvidence {
    CommissioningDisableFirst { report: Box<TorqueDisableReport> },
    TensionPreservingTakeover,
}

impl HeadStartupTorqueEvidence {
    pub fn commissioning_disable_report(&self) -> Option<&TorqueDisableReport> {
        match self {
            Self::CommissioningDisableFirst { report } => Some(report.as_ref()),
            Self::TensionPreservingTakeover => None,
        }
    }

    pub const fn is_tension_preserving_takeover(&self) -> bool {
        matches!(self, Self::TensionPreservingTakeover)
    }
}

/// Success is emitted only after two stopped post-write positions per joint
/// parse exactly, agree with each other, and agree with the observed targets.
/// Servo response level zero means the individual writes remain
/// write-completion evidence, not register readback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedNaturalHoldEvidence {
    started_at: MonotonicTime,
    completed_at: MonotonicTime,
    startup_torque: HeadStartupTorqueEvidence,
    observed_pose: HeadPose,
    configured_pose: HeadPoseWithinConfiguredBounds,
    observations: [PositionObservationEvidence; 4],
    observed_goal_writes: [WriteEvidence; 4],
    torque_limit_writes: [WriteEvidence; 4],
    torque_enable_writes: [WriteEvidence; 4],
    readbacks: [ReadbackEvidence; 4],
}

impl VerifiedNaturalHoldEvidence {
    pub const fn started_at(&self) -> MonotonicTime {
        self.started_at
    }

    pub const fn completed_at(&self) -> MonotonicTime {
        self.completed_at
    }

    /// The selected startup torque policy and its exact available evidence.
    pub const fn startup_torque(&self) -> &HeadStartupTorqueEvidence {
        &self.startup_torque
    }

    /// All four host writes which established a torque-disabled baseline
    /// before a commissioning actor attempted its first observation. A
    /// tension-preserving production takeover returns `None`.
    pub fn pre_observation_torque_disable(&self) -> Option<&TorqueDisableReport> {
        self.startup_torque.commissioning_disable_report()
    }

    pub const fn observed_pose(&self) -> HeadPose {
        self.observed_pose
    }

    /// Typed evidence that the observed pose passed inside the actor before
    /// any goal, torque-limit, or torque-enable write was attempted.
    pub const fn configured_pose(&self) -> HeadPoseWithinConfiguredBounds {
        self.configured_pose
    }

    pub const fn observations(&self) -> &[PositionObservationEvidence; 4] {
        &self.observations
    }

    pub const fn observed_goal_writes(&self) -> &[WriteEvidence; 4] {
        &self.observed_goal_writes
    }

    pub const fn torque_limit_writes(&self) -> &[WriteEvidence; 4] {
        &self.torque_limit_writes
    }

    pub const fn torque_enable_writes(&self) -> &[WriteEvidence; 4] {
        &self.torque_enable_writes
    }

    pub const fn readbacks(&self) -> &[ReadbackEvidence; 4] {
        &self.readbacks
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadWaypointEvidence {
    positions: [PositionTicks; 4],
    writes: [WriteEvidence; 4],
}

impl HeadWaypointEvidence {
    pub const fn positions(&self) -> [PositionTicks; 4] {
        self.positions
    }

    pub const fn writes(&self) -> &[WriteEvidence; 4] {
        &self.writes
    }
}

/// One complete status-zero, identity-ordered telemetry set together with the
/// monotonic receive times used for its bounded-span/freshness admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadTelemetrySetEvidence {
    samples: [FullTelemetry; 4],
    received_at: [MonotonicTime; 4],
    admitted_at: MonotonicTime,
}

impl HeadTelemetrySetEvidence {
    pub const fn samples(&self) -> &[FullTelemetry; 4] {
        &self.samples
    }

    pub const fn received_at(&self) -> &[MonotonicTime; 4] {
        &self.received_at
    }

    pub const fn admitted_at(&self) -> MonotonicTime {
        self.admitted_at
    }
}

impl From<FreshHeadTelemetrySet> for HeadTelemetrySetEvidence {
    fn from(set: FreshHeadTelemetrySet) -> Self {
        Self {
            samples: set.samples(),
            received_at: set.received_at(),
            admitted_at: set.admitted_at(),
        }
    }
}

/// One stopped, status-zero observation of an exact expected joint whose
/// position remains inside the actor's admitted natural-hold tolerance.
///
/// Values other than position and the moving flag remain deliberately raw in
/// [`FullTelemetry`]; this evidence does not invent physical units for them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadHealthJointEvidence {
    joint: HeadJoint,
    target: PositionTicks,
    absolute_difference_ticks: u16,
    response: ResponseEvidence<FullTelemetry>,
}

impl HeadHealthJointEvidence {
    pub const fn joint(&self) -> HeadJoint {
        self.joint
    }

    pub const fn target(&self) -> PositionTicks {
        self.target
    }

    pub const fn absolute_difference_ticks(&self) -> u16 {
        self.absolute_difference_ticks
    }

    /// Exact parsed telemetry, including raw speed, load, voltage,
    /// temperature, current, status, and otherwise unqualified registers.
    pub const fn telemetry(&self) -> &FullTelemetry {
        self.response.value()
    }

    pub const fn response(&self) -> &ResponseEvidence<FullTelemetry> {
        &self.response
    }
}

/// Exact provenance of the goal against which periodic head health is checked.
///
/// Startup can only hold a redundantly observed pose. A successful configured
/// return instead holds the exact reviewed target. Recoverable return faults
/// may leave a complete host-commanded position set active; that state remains
/// distinguishable from both observed and reviewed targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadHoldTarget {
    StartupObserved(HeadPose),
    ReviewedReturn(ExactHeadTargetPose),
    RecoverableReturnCommand([PositionTicks; 4]),
}

impl HeadHoldTarget {
    pub const fn position(self, joint: HeadJoint) -> PositionTicks {
        match self {
            Self::StartupObserved(pose) => pose.position(joint),
            Self::ReviewedReturn(target) => target.position(joint),
            Self::RecoverableReturnCommand(positions) => positions[joint as usize],
        }
    }

    pub const fn positions(self) -> [PositionTicks; 4] {
        match self {
            Self::StartupObserved(pose) => pose.positions(),
            Self::ReviewedReturn(target) => target.positions(),
            Self::RecoverableReturnCommand(positions) => positions,
        }
    }
}

/// A complete canonical bow/curl/yaw/roll health observation made while the
/// actor retained exclusive ownership of the servo bus.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedHeadHealthEvidence {
    started_at: MonotonicTime,
    completed_at: MonotonicTime,
    hold_target: HeadHoldTarget,
    tolerance: PositionAgreementTicks,
    joints: [HeadHealthJointEvidence; 4],
}

impl VerifiedHeadHealthEvidence {
    pub const fn started_at(&self) -> MonotonicTime {
        self.started_at
    }

    pub const fn completed_at(&self) -> MonotonicTime {
        self.completed_at
    }

    pub const fn hold_target(&self) -> HeadHoldTarget {
        self.hold_target
    }

    pub const fn tolerance(&self) -> PositionAgreementTicks {
        self.tolerance
    }

    /// Evidence in the exact canonical order defined by [`HeadJoint::ALL`].
    pub const fn joints(&self) -> &[HeadHealthJointEvidence; 4] {
        &self.joints
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HeadHealthClockBoundary {
    RequestWriteCompleted { joint: HeadJoint },
    ResponseReceived { joint: HeadJoint },
    CheckCompleted,
}

/// Exact reason a bounded natural-hold health observation was not admitted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeadHealthFailure {
    Cancelled {
        cause: CancellationCause,
        stage: RuntimeStage,
        joint: HeadJoint,
    },
    TelemetryRead {
        joint: HeadJoint,
        source: RequestError,
    },
    ClockRegression {
        boundary: HeadHealthClockBoundary,
        previous: MonotonicTime,
        observed: MonotonicTime,
        current_response: Option<Box<ResponseEvidence<FullTelemetry>>>,
    },
    DeviceStatus {
        joint: HeadJoint,
        raw: u8,
        response: ResponseEvidence<FullTelemetry>,
    },
    Moving {
        joint: HeadJoint,
        position: PositionTicks,
        response: ResponseEvidence<FullTelemetry>,
    },
    PositionMismatch {
        joint: HeadJoint,
        target: PositionTicks,
        actual: PositionTicks,
        absolute_difference_ticks: u16,
        tolerance: PositionAgreementTicks,
        response: ResponseEvidence<FullTelemetry>,
    },
}

impl fmt::Display for HeadHealthFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "natural-hold health observation failed: {self:?}"
        )
    }
}

impl std::error::Error for HeadHealthFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::TelemetryRead { source, .. } => Some(source),
            Self::Cancelled { .. }
            | Self::ClockRegression { .. }
            | Self::DeviceStatus { .. }
            | Self::Moving { .. }
            | Self::PositionMismatch { .. } => None,
        }
    }
}

/// Lossless accepted prefix plus the exact failure for one health check.
///
/// An observation which fails status, moving, or position admission is kept
/// in the corresponding [`HeadHealthFailure`] variant rather than being
/// mislabeled as part of the accepted prefix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadHealthObservationError {
    started_at: MonotonicTime,
    accepted_prefix: [Option<HeadHealthJointEvidence>; 4],
    failure: HeadHealthFailure,
}

impl HeadHealthObservationError {
    pub const fn started_at(&self) -> MonotonicTime {
        self.started_at
    }

    pub const fn accepted_prefix(&self) -> &[Option<HeadHealthJointEvidence>; 4] {
        &self.accepted_prefix
    }

    pub const fn failure(&self) -> &HeadHealthFailure {
        &self.failure
    }
}

impl fmt::Display for HeadHealthObservationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.failure.fmt(formatter)
    }
}

impl std::error::Error for HeadHealthObservationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.failure)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeadHealthCheckError {
    CommandBeforeStartup,
    CommandAlreadyInProgress,
    Observation(Box<HeadHealthObservationError>),
}

impl fmt::Display for HeadHealthCheckError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "head health check failed: {self:?}")
    }
}

impl std::error::Error for HeadHealthCheckError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Observation(source) => Some(source.as_ref()),
            Self::CommandBeforeStartup | Self::CommandAlreadyInProgress => None,
        }
    }
}

/// Lossless prefix evidence for a four-joint waypoint batch. Every `Some`
/// entry completed before `source`; later `None` entries were not attempted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadWaypointBatchWriteError {
    positions: [PositionTicks; 4],
    completed_writes: [Option<WriteEvidence>; 4],
    failure: HeadWaypointBatchFailure,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeadWaypointBatchFailure {
    Frame(FrameWriteError),
    Deadline {
        source: HeadMotionError,
        io: Option<FrameWriteError>,
    },
    Cancelled {
        cause: CancellationCause,
        stage: RuntimeStage,
        joint: HeadJoint,
    },
}

impl HeadWaypointBatchWriteError {
    pub const fn positions(&self) -> [PositionTicks; 4] {
        self.positions
    }

    pub const fn completed_writes(&self) -> &[Option<WriteEvidence>; 4] {
        &self.completed_writes
    }

    pub const fn failure(&self) -> &HeadWaypointBatchFailure {
        &self.failure
    }
}

impl fmt::Display for HeadWaypointBatchWriteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .write_str("four-joint waypoint batch failed after its recorded completed prefix: ")?;
        match &self.failure {
            HeadWaypointBatchFailure::Frame(source) => write!(formatter, "{source}"),
            HeadWaypointBatchFailure::Deadline { source, .. } => write!(formatter, "{source}"),
            HeadWaypointBatchFailure::Cancelled {
                cause,
                stage,
                joint,
            } => write!(
                formatter,
                "cancelled by {cause:?} at {stage:?} for {joint:?}"
            ),
        }
    }
}

impl std::error::Error for HeadWaypointBatchWriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.failure {
            HeadWaypointBatchFailure::Frame(source) => Some(source),
            HeadWaypointBatchFailure::Deadline { source, .. } => Some(source),
            HeadWaypointBatchFailure::Cancelled { .. } => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InterruptedTelemetryRead {
    joint: HeadJoint,
    source: Option<RequestError>,
}

impl InterruptedTelemetryRead {
    pub const fn joint(&self) -> HeadJoint {
        self.joint
    }

    pub const fn source_error(&self) -> Option<&RequestError> {
        self.source.as_ref()
    }
}

/// Evidence emitted only after the bounded waypoint stream commands the exact
/// reviewed target and two complete stopped telemetry samples lie inside the
/// configured final tolerance and agree with each other.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedHeadReturnEvidence {
    started_at: MonotonicTime,
    completed_at: MonotonicTime,
    start_pose: HeadPose,
    start_first: HeadTelemetrySetEvidence,
    start_second: HeadTelemetrySetEvidence,
    target: kiko_head_protocol::ExactHeadTargetPose,
    waypoint_writes: Vec<HeadWaypointEvidence>,
    first_stopped: HeadTelemetrySetEvidence,
    second_stopped: HeadTelemetrySetEvidence,
}

impl VerifiedHeadReturnEvidence {
    pub const fn started_at(&self) -> MonotonicTime {
        self.started_at
    }

    pub const fn completed_at(&self) -> MonotonicTime {
        self.completed_at
    }

    pub const fn start_pose(&self) -> HeadPose {
        self.start_pose
    }

    pub const fn target(&self) -> kiko_head_protocol::ExactHeadTargetPose {
        self.target
    }

    pub const fn start_first(&self) -> &HeadTelemetrySetEvidence {
        &self.start_first
    }

    pub const fn start_second(&self) -> &HeadTelemetrySetEvidence {
        &self.start_second
    }

    pub fn waypoint_writes(&self) -> &[HeadWaypointEvidence] {
        &self.waypoint_writes
    }

    pub const fn first_stopped(&self) -> &[FullTelemetry; 4] {
        self.first_stopped.samples()
    }

    pub const fn second_stopped(&self) -> &[FullTelemetry; 4] {
        self.second_stopped.samples()
    }

    pub const fn first_stopped_set(&self) -> &HeadTelemetrySetEvidence {
        &self.first_stopped
    }

    pub const fn second_stopped_set(&self) -> &HeadTelemetrySetEvidence {
        &self.second_stopped
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeadReturnError {
    CommandBeforeStartup,
    CommandAlreadyInProgress,
    CommandAlreadyAttempted,
    Cancelled {
        cause: CancellationCause,
        stage: RuntimeStage,
        joint: HeadJoint,
        waypoint_writes: Vec<HeadWaypointEvidence>,
    },
    TelemetryRead {
        joint: HeadJoint,
        source: RequestError,
        waypoint_writes: Vec<HeadWaypointEvidence>,
    },
    /// No new untrusted goal was written; the actor retains the last complete
    /// admitted per-joint goals and remains the serial owner.
    KinematicFaultExistingGoalRetained {
        source: HeadMotionError,
        commanded_positions: [PositionTicks; 4],
        interrupted_read: Option<Box<InterruptedTelemetryRead>>,
        interrupted_write: Option<Box<HeadWaypointBatchWriteError>>,
        waypoint_writes: Vec<HeadWaypointEvidence>,
    },
    KinematicFaultRecoveryWritten {
        source: HeadMotionError,
        held_positions: [PositionTicks; 4],
        hold_writes: Box<[WriteEvidence; 4]>,
        waypoint_writes: Vec<HeadWaypointEvidence>,
    },
    KinematicFaultRecoveryWriteFailed {
        source: HeadMotionError,
        recovery_write: Box<HeadWaypointBatchWriteError>,
        waypoint_writes: Vec<HeadWaypointEvidence>,
    },
    Motion {
        source: HeadMotionError,
        waypoint_writes: Vec<HeadWaypointEvidence>,
    },
    WaypointWrite {
        source: Box<HeadWaypointBatchWriteError>,
        waypoint_writes: Vec<HeadWaypointEvidence>,
    },
}

impl fmt::Display for HeadReturnError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Kiko return-to-target transaction failed: {self:?}"
        )
    }
}

impl std::error::Error for HeadReturnError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::TelemetryRead { source, .. } => Some(source),
            Self::KinematicFaultExistingGoalRetained { source, .. }
            | Self::KinematicFaultRecoveryWritten { source, .. }
            | Self::KinematicFaultRecoveryWriteFailed { source, .. }
            | Self::Motion { source, .. } => Some(source),
            Self::WaypointWrite { source, .. } => Some(source),
            Self::CommandBeforeStartup
            | Self::CommandAlreadyInProgress
            | Self::CommandAlreadyAttempted
            | Self::Cancelled { .. } => None,
        }
    }
}

impl HeadReturnError {
    /// True only for typed outcomes which explicitly preserve the actor and its
    /// previously admitted goal state. A successful recovery write remains host
    /// completion evidence, not register acknowledgement or stopped readback.
    pub const fn retains_owner_after_fault(&self) -> bool {
        matches!(
            self,
            Self::KinematicFaultExistingGoalRetained { .. }
                | Self::KinematicFaultRecoveryWritten { .. }
        )
    }

    pub fn waypoint_writes(&self) -> &[HeadWaypointEvidence] {
        match self {
            Self::Cancelled {
                waypoint_writes, ..
            }
            | Self::TelemetryRead {
                waypoint_writes, ..
            }
            | Self::KinematicFaultExistingGoalRetained {
                waypoint_writes, ..
            }
            | Self::KinematicFaultRecoveryWritten {
                waypoint_writes, ..
            }
            | Self::KinematicFaultRecoveryWriteFailed {
                waypoint_writes, ..
            }
            | Self::Motion {
                waypoint_writes, ..
            }
            | Self::WaypointWrite {
                waypoint_writes, ..
            } => waypoint_writes,
            Self::CommandBeforeStartup
            | Self::CommandAlreadyInProgress
            | Self::CommandAlreadyAttempted => &[],
        }
    }

    fn with_waypoint_writes(self, waypoint_writes: Vec<HeadWaypointEvidence>) -> Self {
        match self {
            Self::Cancelled {
                cause,
                stage,
                joint,
                ..
            } => Self::Cancelled {
                cause,
                stage,
                joint,
                waypoint_writes,
            },
            Self::TelemetryRead { joint, source, .. } => Self::TelemetryRead {
                joint,
                source,
                waypoint_writes,
            },
            Self::Motion { source, .. } => Self::Motion {
                source,
                waypoint_writes,
            },
            other => other,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameWriteError {
    pub joint: HeadJoint,
    pub purpose: WritePurpose,
    pub attempts_used: u8,
    pub recovered_failures: Vec<TransportFailure>,
    pub source: TransportFailure,
}

impl fmt::Display for FrameWriteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:?} write for {:?} failed on attempt {}: {}",
            self.purpose, self.joint, self.attempts_used, self.source
        )
    }
}

impl std::error::Error for FrameWriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RequestError {
    RequestWrite(FrameWriteError),
    ResponseFrame(FrameReadError),
    Telemetry(TelemetryParseError),
}

impl fmt::Display for RequestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "typed STS request/response failed: {self:?}")
    }
}

impl std::error::Error for RequestError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::RequestWrite(source) => Some(source),
            Self::ResponseFrame(source) => Some(source),
            Self::Telemetry(source) => Some(source),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CancellationCause {
    RequestedShutdown,
    RequestedHoldPreservingRelease,
    HandleDropped,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeadRuntimeError {
    PreObservationTorqueDisable {
        report: Box<TorqueDisableReport>,
    },
    Cancelled {
        cause: CancellationCause,
        stage: RuntimeStage,
        joint: HeadJoint,
    },
    PositionObservation {
        joint: HeadJoint,
        stage: RuntimeStage,
        source: RequestError,
    },
    PositionAgreement {
        joint: HeadJoint,
        source: PositionAgreementError,
    },
    PoseAdmission {
        source: HeadPoseError,
    },
    ConfiguredPoseAdmission {
        source: HeadPoseBoundsAdmissionError,
    },
    Write {
        stage: RuntimeStage,
        source: FrameWriteError,
    },
    VerificationRead {
        joint: HeadJoint,
        source: RequestError,
    },
    ReadbackMismatch {
        joint: HeadJoint,
        sample: VerificationSample,
        target: PositionTicks,
        actual: PositionTicks,
        absolute_difference_ticks: u16,
        tolerance: PositionAgreementTicks,
    },
    ReadbackMoving {
        joint: HeadJoint,
        sample: VerificationSample,
        position: PositionTicks,
    },
    ReadbackDeviceStatus {
        joint: HeadJoint,
        sample: VerificationSample,
        position: PositionTicks,
        raw: u8,
    },
    ReadbackUnstable {
        joint: HeadJoint,
        first: PositionTicks,
        second: PositionTicks,
        absolute_difference_ticks: u16,
        tolerance: PositionAgreementTicks,
    },
    ObservationClockRegression {
        oldest_observation_at: MonotonicTime,
        checked_at: MonotonicTime,
    },
    ObservationStaleBeforeArming {
        joint: HeadJoint,
        check: ArmingFreshnessCheck,
        oldest_observation_at: MonotonicTime,
        checked_at: MonotonicTime,
        age: Duration,
        maximum_age: Duration,
    },
    ObservationArmingWriteBudgetInsufficient {
        joint: HeadJoint,
        oldest_observation_at: MonotonicTime,
        checked_at: MonotonicTime,
        remaining_freshness: Duration,
        required_write_budget: Duration,
    },
}

impl fmt::Display for HeadRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Kiko natural-hold startup failed: {self:?}")
    }
}

impl std::error::Error for HeadRuntimeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::PreObservationTorqueDisable { report } => report
                .first_failure()
                .map(|source| source as &(dyn std::error::Error + 'static)),
            Self::PositionObservation { source, .. } | Self::VerificationRead { source, .. } => {
                Some(source)
            }
            Self::PositionAgreement { source, .. } => Some(source),
            Self::PoseAdmission { source } => Some(source),
            Self::ConfiguredPoseAdmission { source } => Some(source),
            Self::Write { source, .. } => Some(source),
            Self::Cancelled { .. }
            | Self::ReadbackMismatch { .. }
            | Self::ReadbackMoving { .. }
            | Self::ReadbackDeviceStatus { .. }
            | Self::ReadbackUnstable { .. }
            | Self::ObservationClockRegression { .. }
            | Self::ObservationStaleBeforeArming { .. }
            | Self::ObservationArmingWriteBudgetInsufficient { .. } => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TorqueDisableJointOutcome {
    joint: HeadJoint,
    result: Result<WriteEvidence, FrameWriteError>,
}

impl TorqueDisableJointOutcome {
    pub const fn joint(&self) -> HeadJoint {
        self.joint
    }

    pub const fn result(&self) -> &Result<WriteEvidence, FrameWriteError> {
        &self.result
    }
}

/// Every element is present because shutdown always attempts all four joints,
/// even after an earlier disable write fails.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TorqueDisableReport {
    started_at: MonotonicTime,
    completed_at: MonotonicTime,
    outcomes: [TorqueDisableJointOutcome; 4],
}

impl TorqueDisableReport {
    pub const fn started_at(&self) -> MonotonicTime {
        self.started_at
    }

    pub const fn completed_at(&self) -> MonotonicTime {
        self.completed_at
    }

    pub const fn outcomes(&self) -> &[TorqueDisableJointOutcome; 4] {
        &self.outcomes
    }

    pub fn all_writes_completed(&self) -> bool {
        self.outcomes.iter().all(|outcome| outcome.result.is_ok())
    }

    pub fn first_failure(&self) -> Option<&FrameWriteError> {
        self.outcomes
            .iter()
            .find_map(|outcome| outcome.result.as_ref().err())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActorTermination {
    RequestedShutdown,
    RequestedHoldPreservingRelease,
    HandleDropped,
    StartupFault,
    StartupFaultWithShutdownRequested,
    HeadReturnFault,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActorExit {
    startup: Result<VerifiedNaturalHoldEvidence, HeadRuntimeError>,
    head_return: Option<Result<VerifiedHeadReturnEvidence, HeadReturnError>>,
    termination: ActorTermination,
    torque_disable: TorqueDisableReport,
}

impl ActorExit {
    pub const fn startup(&self) -> &Result<VerifiedNaturalHoldEvidence, HeadRuntimeError> {
        &self.startup
    }

    pub const fn head_return(
        &self,
    ) -> Option<&Result<VerifiedHeadReturnEvidence, HeadReturnError>> {
        self.head_return.as_ref()
    }

    pub const fn termination(&self) -> &ActorTermination {
        &self.termination
    }

    pub const fn torque_disable(&self) -> &TorqueDisableReport {
        &self.torque_disable
    }
}

/// Evidence that the production actor closed its bus owner without issuing a
/// torque-switch write as part of cleanup.
///
/// This does not prove that servo torque was enabled before or after release.
/// Electrical power loss, another bus owner, or the servo itself can still
/// change physical holding state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HoldPreservingOwnershipReleaseEvidence {
    recorded_at: MonotonicTime,
}

impl HoldPreservingOwnershipReleaseEvidence {
    /// Host timestamp at which the actor selected the no-write cleanup path.
    /// It is not a measurement of the physical torque state.
    pub const fn recorded_at(&self) -> MonotonicTime {
        self.recorded_at
    }
}

/// Exit evidence for the production-only tension-preserving actor.
///
/// Unlike [`ActorExit`], this type cannot contain a torque-disable report: the
/// corresponding handle exposes no torque-release operation, and every exit
/// path closes ownership without a torque-switch write.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensionPreservingHeadActorExit {
    startup: Result<VerifiedNaturalHoldEvidence, HeadRuntimeError>,
    head_return: Option<Result<VerifiedHeadReturnEvidence, HeadReturnError>>,
    termination: ActorTermination,
    hold_preserving_release: HoldPreservingOwnershipReleaseEvidence,
}

impl TensionPreservingHeadActorExit {
    pub const fn startup(&self) -> &Result<VerifiedNaturalHoldEvidence, HeadRuntimeError> {
        &self.startup
    }

    pub const fn head_return(
        &self,
    ) -> Option<&Result<VerifiedHeadReturnEvidence, HeadReturnError>> {
        self.head_return.as_ref()
    }

    pub const fn termination(&self) -> &ActorTermination {
        &self.termination
    }

    pub const fn hold_preserving_release(&self) -> &HoldPreservingOwnershipReleaseEvidence {
        &self.hold_preserving_release
    }
}

enum HeadCommand {
    CheckHealth {
        response: oneshot::Sender<Result<VerifiedHeadHealthEvidence, HeadHealthCheckError>>,
    },
    ReturnToTarget {
        response: oneshot::Sender<Result<VerifiedHeadReturnEvidence, HeadReturnError>>,
    },
    Shutdown {
        response: oneshot::Sender<TorqueDisableReport>,
    },
    ReleaseOwnershipPreservingHold {
        response: oneshot::Sender<HoldPreservingOwnershipReleaseEvidence>,
    },
}

/// The only public command endpoint. It is intentionally not cloneable: one
/// caller owns shutdown authority, while the actor exclusively owns serial I/O.
pub struct HeadActorHandle {
    commands: mpsc::Sender<HeadCommand>,
}

impl HeadActorHandle {
    /// Observe all four joints without relinquishing the actor's exclusive bus
    /// ownership or changing any torque/goal register.
    pub async fn check_health(&self) -> Result<VerifiedHeadHealthEvidence, HeadHealthRequestError> {
        let (response, result) = oneshot::channel();
        self.commands
            .send(HeadCommand::CheckHealth { response })
            .await
            .map_err(|_| HeadHealthRequestError::ActorAlreadyStopped)?;
        result
            .await
            .map_err(|_| HeadHealthRequestError::ActorStoppedBeforeReporting)?
            .map_err(|source| HeadHealthRequestError::Check { source })
    }

    pub async fn shutdown(self) -> Result<TorqueDisableReport, ShutdownError> {
        let (response, result) = oneshot::channel();
        self.commands
            .send(HeadCommand::Shutdown { response })
            .await
            .map_err(|_| ShutdownError::ActorAlreadyStopped)?;
        result
            .await
            .map_err(|_| ShutdownError::ActorStoppedBeforeReporting)
    }
}

/// Motion endpoint produced only by a complete `ReturnToTargetConfig`. The
/// reviewed target, bounds, speed, torque, and tolerances are stored in the
/// actor; no command can cross-pair a plan from another configuration/device.
pub struct HeadReturnActorHandle {
    commands: mpsc::Sender<HeadCommand>,
}

impl HeadReturnActorHandle {
    pub async fn return_to_target(
        &self,
    ) -> Result<Result<VerifiedHeadReturnEvidence, HeadReturnError>, HeadCommandError> {
        let (response, result) = oneshot::channel();
        self.commands
            .send(HeadCommand::ReturnToTarget { response })
            .await
            .map_err(|_| HeadCommandError::ActorAlreadyStopped)?;
        result
            .await
            .map_err(|_| HeadCommandError::ActorStoppedBeforeReporting)
    }

    /// Observe the complete head against the currently active target.
    ///
    /// Before a return succeeds this is the startup-observed pose. Afterwards
    /// it is the exact reviewed return target. A recoverable return fault
    /// retains the complete host-commanded goal it reports.
    pub async fn check_health(&self) -> Result<VerifiedHeadHealthEvidence, HeadHealthRequestError> {
        let (response, result) = oneshot::channel();
        self.commands
            .send(HeadCommand::CheckHealth { response })
            .await
            .map_err(|_| HeadHealthRequestError::ActorAlreadyStopped)?;
        result
            .await
            .map_err(|_| HeadHealthRequestError::ActorStoppedBeforeReporting)?
            .map_err(|source| HeadHealthRequestError::Check { source })
    }

    pub async fn shutdown(self) -> Result<TorqueDisableReport, ShutdownError> {
        let (response, result) = oneshot::channel();
        self.commands
            .send(HeadCommand::Shutdown { response })
            .await
            .map_err(|_| ShutdownError::ActorAlreadyStopped)?;
        result
            .await
            .map_err(|_| ShutdownError::ActorStoppedBeforeReporting)
    }
}

/// Production-only motion endpoint for a tension-preserving handoff.
///
/// This handle intentionally has no torque-disable method. Releasing it,
/// including through ordinary process shutdown, closes exclusive serial
/// ownership without writing the torque-switch register.
pub struct TensionPreservingHeadReturnActorHandle {
    commands: mpsc::Sender<HeadCommand>,
}

impl TensionPreservingHeadReturnActorHandle {
    pub async fn return_to_target(
        &self,
    ) -> Result<Result<VerifiedHeadReturnEvidence, HeadReturnError>, HeadCommandError> {
        let (response, result) = oneshot::channel();
        self.commands
            .send(HeadCommand::ReturnToTarget { response })
            .await
            .map_err(|_| HeadCommandError::ActorAlreadyStopped)?;
        result
            .await
            .map_err(|_| HeadCommandError::ActorStoppedBeforeReporting)
    }

    pub async fn check_health(&self) -> Result<VerifiedHeadHealthEvidence, HeadHealthRequestError> {
        let (response, result) = oneshot::channel();
        self.commands
            .send(HeadCommand::CheckHealth { response })
            .await
            .map_err(|_| HeadHealthRequestError::ActorAlreadyStopped)?;
        result
            .await
            .map_err(|_| HeadHealthRequestError::ActorStoppedBeforeReporting)?
            .map_err(|source| HeadHealthRequestError::Check { source })
    }

    pub async fn release_ownership_preserving_hold(
        self,
    ) -> Result<HoldPreservingOwnershipReleaseEvidence, ShutdownError> {
        let (response, result) = oneshot::channel();
        self.commands
            .send(HeadCommand::ReleaseOwnershipPreservingHold { response })
            .await
            .map_err(|_| ShutdownError::ActorAlreadyStopped)?;
        result
            .await
            .map_err(|_| ShutdownError::ActorStoppedBeforeReporting)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadCommandError {
    ActorAlreadyStopped,
    ActorStoppedBeforeReporting,
}

impl fmt::Display for HeadCommandError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "head motion command failed: {self:?}")
    }
}

impl std::error::Error for HeadCommandError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeadHealthRequestError {
    ActorAlreadyStopped,
    ActorStoppedBeforeReporting,
    Check { source: HeadHealthCheckError },
}

impl fmt::Display for HeadHealthRequestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "head health request failed: {self:?}")
    }
}

impl std::error::Error for HeadHealthRequestError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Check { source } => Some(source),
            Self::ActorAlreadyStopped | Self::ActorStoppedBeforeReporting => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShutdownError {
    ActorAlreadyStopped,
    ActorStoppedBeforeReporting,
}

impl fmt::Display for ShutdownError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "head actor shutdown command failed: {self:?}")
    }
}

impl std::error::Error for ShutdownError {}

pub struct StartupReceipt {
    result: oneshot::Receiver<Result<VerifiedNaturalHoldEvidence, HeadRuntimeError>>,
}

impl StartupReceipt {
    pub async fn wait(
        self,
    ) -> Result<Result<VerifiedNaturalHoldEvidence, HeadRuntimeError>, StartupReceiptError> {
        self.result
            .await
            .map_err(|_| StartupReceiptError::ActorStoppedBeforeReporting)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StartupReceiptError {
    ActorStoppedBeforeReporting,
}

impl fmt::Display for StartupReceiptError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "head actor startup receipt failed: {self:?}")
    }
}

impl std::error::Error for StartupReceiptError {}

pub struct HeadActorTask {
    task: JoinHandle<ActorExit>,
}

impl HeadActorTask {
    pub async fn join(self) -> Result<ActorExit, JoinError> {
        self.task.await
    }
}

pub struct TensionPreservingHeadActorTask {
    task: JoinHandle<TensionPreservingHeadActorExit>,
}

impl TensionPreservingHeadActorTask {
    pub async fn join(self) -> Result<TensionPreservingHeadActorExit, JoinError> {
        self.task.await
    }
}

#[derive(Debug)]
pub enum HeadActorSpawnError {
    NoTokioRuntime { source: TryCurrentError },
}

impl fmt::Display for HeadActorSpawnError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "could not spawn Kiko head actor: {self:?}")
    }
}

impl std::error::Error for HeadActorSpawnError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NoTokioRuntime { source } => Some(source),
        }
    }
}

#[derive(Debug)]
pub enum HeadActorStartError {
    NoTokioRuntime { source: TryCurrentError },
    Serial { source: SerialOpenError },
}

impl fmt::Display for HeadActorStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "could not start Kiko head actor: {self:?}")
    }
}

impl std::error::Error for HeadActorStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NoTokioRuntime { source } => Some(source),
            Self::Serial { source } => Some(source),
        }
    }
}

/// Spawn the testable core with one transport owner and one injected clock.
pub fn spawn_head_actor<T, C>(
    transport: T,
    clock: C,
    config: HeadRuntimeConfig,
    configured_pose_bounds: ConfiguredHeadPoseBounds,
    consent: PhysicalTorqueEnableConsent,
) -> Result<(HeadActorHandle, StartupReceipt, HeadActorTask), HeadActorSpawnError>
where
    T: AsyncByteTransport,
    C: MonotonicClock,
{
    let runtime =
        Handle::try_current().map_err(|source| HeadActorSpawnError::NoTokioRuntime { source })?;
    let (commands, startup, task) = spawn_head_actor_on(
        &runtime,
        transport,
        clock,
        config,
        configured_pose_bounds,
        consent,
        None,
    );
    Ok((HeadActorHandle { commands }, startup, task))
}

/// Spawn the testable return owner from one inseparable parsed configuration.
pub fn spawn_head_return_actor<T, C>(
    transport: T,
    clock: C,
    config: ReturnToTargetConfig,
    torque_consent: PhysicalTorqueEnableConsent,
    _motion_consent: PhysicalHeadMotionConsent,
) -> Result<(HeadReturnActorHandle, StartupReceipt, HeadActorTask), HeadActorSpawnError>
where
    T: AsyncByteTransport,
    C: MonotonicClock,
{
    let runtime =
        Handle::try_current().map_err(|source| HeadActorSpawnError::NoTokioRuntime { source })?;
    let (runtime_config, start_bounds, plan) = config.into_actor_parts();
    let (commands, startup, task) = spawn_head_actor_on(
        &runtime,
        transport,
        clock,
        runtime_config,
        start_bounds,
        torque_consent,
        Some(plan),
    );
    Ok((HeadReturnActorHandle { commands }, startup, task))
}

/// Spawn the testable core for the manifest-bound production handoff without
/// first torque-disabling the neck.
///
/// Unlike [`spawn_head_return_actor`], this path does not establish or claim a
/// known prior torque state. Its first protocol traffic is the redundant
/// present-position observation used to adopt the current pose.
pub fn spawn_tension_preserving_head_return_actor<T, C>(
    transport: T,
    clock: C,
    config: ReturnToTargetConfig,
    torque_consent: PhysicalTorqueEnableConsent,
    _motion_consent: PhysicalHeadMotionConsent,
    _takeover_consent: ProductionTensionPreservingTakeoverConsent,
) -> Result<
    (
        TensionPreservingHeadReturnActorHandle,
        StartupReceipt,
        TensionPreservingHeadActorTask,
    ),
    HeadActorSpawnError,
>
where
    T: AsyncByteTransport,
    C: MonotonicClock,
{
    let runtime =
        Handle::try_current().map_err(|source| HeadActorSpawnError::NoTokioRuntime { source })?;
    let (runtime_config, start_bounds, plan) = config.into_actor_parts();
    let (commands, startup, task) = spawn_tension_preserving_head_actor_on(
        &runtime,
        transport,
        clock,
        runtime_config,
        start_bounds,
        torque_consent,
        Some(plan),
    );
    Ok((
        TensionPreservingHeadReturnActorHandle { commands },
        startup,
        task,
    ))
}

fn spawn_head_actor_on<T, C>(
    runtime: &Handle,
    transport: T,
    clock: C,
    config: HeadRuntimeConfig,
    configured_pose_bounds: ConfiguredHeadPoseBounds,
    _consent: PhysicalTorqueEnableConsent,
    return_plan: Option<HeadReturnPlan>,
) -> (mpsc::Sender<HeadCommand>, StartupReceipt, HeadActorTask)
where
    T: AsyncByteTransport,
    C: MonotonicClock,
{
    let (commands, receiver) = mpsc::channel(ACTOR_MAILBOX_CAPACITY);
    let (startup_sender, startup_result) = oneshot::channel();
    let actor = HeadActor {
        transport,
        clock,
        config,
        configured_pose_bounds,
        startup_torque_policy: StartupTorquePolicy::CommissioningDisableFirst,
        return_plan,
    };
    let task = runtime.spawn(async move {
        actor
            .run(receiver, startup_sender)
            .await
            .into_commissioning()
    });
    (
        commands,
        StartupReceipt {
            result: startup_result,
        },
        HeadActorTask { task },
    )
}

fn spawn_tension_preserving_head_actor_on<T, C>(
    runtime: &Handle,
    transport: T,
    clock: C,
    config: HeadRuntimeConfig,
    configured_pose_bounds: ConfiguredHeadPoseBounds,
    _consent: PhysicalTorqueEnableConsent,
    return_plan: Option<HeadReturnPlan>,
) -> (
    mpsc::Sender<HeadCommand>,
    StartupReceipt,
    TensionPreservingHeadActorTask,
)
where
    T: AsyncByteTransport,
    C: MonotonicClock,
{
    let (commands, receiver) = mpsc::channel(ACTOR_MAILBOX_CAPACITY);
    let (startup_sender, startup_result) = oneshot::channel();
    let actor = HeadActor {
        transport,
        clock,
        config,
        configured_pose_bounds,
        startup_torque_policy: StartupTorquePolicy::TensionPreservingTakeover,
        return_plan,
    };
    let task = runtime.spawn(async move {
        actor
            .run(receiver, startup_sender)
            .await
            .into_tension_preserving()
    });
    (
        commands,
        StartupReceipt {
            result: startup_result,
        },
        TensionPreservingHeadActorTask { task },
    )
}

/// Open, exclusively claim, configure, and then spawn the disable-first serial
/// actor used by commissioning. No protocol traffic occurs until every serial
/// setting succeeds.
pub fn start_serial_head_actor(
    config: HeadRuntimeConfig,
    configured_pose_bounds: ConfiguredHeadPoseBounds,
    consent: PhysicalTorqueEnableConsent,
) -> Result<
    (
        SerialConfigurationEvidence,
        HeadActorHandle,
        StartupReceipt,
        HeadActorTask,
    ),
    HeadActorStartError,
> {
    // Check the runtime before opening or changing any physical serial state.
    let runtime =
        Handle::try_current().map_err(|source| HeadActorStartError::NoTokioRuntime { source })?;
    let transport = SerialTransport::open(config.device())
        .map_err(|source| HeadActorStartError::Serial { source })?;
    let serial_evidence = transport.evidence().clone();
    let (handle, startup, task) = spawn_head_actor_on(
        &runtime,
        transport,
        TokioClock::new(),
        config,
        configured_pose_bounds,
        consent,
        None,
    );
    Ok((
        serial_evidence,
        HeadActorHandle { commands: handle },
        startup,
        task,
    ))
}

/// Open and exclusively own the configured bus for one config-bound return
/// transaction. No target/limits can be supplied later through the command API.
pub fn start_serial_head_return_actor(
    config: ReturnToTargetConfig,
    torque_consent: PhysicalTorqueEnableConsent,
    _motion_consent: PhysicalHeadMotionConsent,
) -> Result<
    (
        SerialConfigurationEvidence,
        HeadReturnActorHandle,
        StartupReceipt,
        HeadActorTask,
    ),
    HeadActorStartError,
> {
    let runtime =
        Handle::try_current().map_err(|source| HeadActorStartError::NoTokioRuntime { source })?;
    let (runtime_config, start_bounds, plan) = config.into_actor_parts();
    let transport = SerialTransport::open(runtime_config.device())
        .map_err(|source| HeadActorStartError::Serial { source })?;
    let serial_evidence = transport.evidence().clone();
    let (commands, startup, task) = spawn_head_actor_on(
        &runtime,
        transport,
        TokioClock::new(),
        runtime_config,
        start_bounds,
        torque_consent,
        Some(plan),
    );
    Ok((
        serial_evidence,
        HeadReturnActorHandle { commands },
        startup,
        task,
    ))
}

/// Open and exclusively own the configured production bus, then attempt a
/// tension-preserving adoption of the present pose before the reviewed return.
///
/// No torque-disable write is issued before observation. Serial-open failure
/// produces no protocol traffic, and startup evidence never claims a prior
/// torque-switch state.
pub fn start_serial_tension_preserving_head_return_actor(
    config: ReturnToTargetConfig,
    torque_consent: PhysicalTorqueEnableConsent,
    _motion_consent: PhysicalHeadMotionConsent,
    _takeover_consent: ProductionTensionPreservingTakeoverConsent,
) -> Result<
    (
        SerialConfigurationEvidence,
        TensionPreservingHeadReturnActorHandle,
        StartupReceipt,
        TensionPreservingHeadActorTask,
    ),
    HeadActorStartError,
> {
    let runtime =
        Handle::try_current().map_err(|source| HeadActorStartError::NoTokioRuntime { source })?;
    let (runtime_config, start_bounds, plan) = config.into_actor_parts();
    let transport = SerialTransport::open(runtime_config.device())
        .map_err(|source| HeadActorStartError::Serial { source })?;
    let serial_evidence = transport.evidence().clone();
    let (commands, startup, task) = spawn_tension_preserving_head_actor_on(
        &runtime,
        transport,
        TokioClock::new(),
        runtime_config,
        start_bounds,
        torque_consent,
        Some(plan),
    );
    Ok((
        serial_evidence,
        TensionPreservingHeadReturnActorHandle { commands },
        startup,
        task,
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StartupTorquePolicy {
    CommissioningDisableFirst,
    TensionPreservingTakeover,
}

struct HeadActor<T, C> {
    transport: T,
    clock: C,
    config: HeadRuntimeConfig,
    configured_pose_bounds: ConfiguredHeadPoseBounds,
    startup_torque_policy: StartupTorquePolicy,
    return_plan: Option<HeadReturnPlan>,
}

struct ControlState {
    termination: Option<ActorTermination>,
    shutdown_response: Option<HeadShutdownResponse>,
}

impl ControlState {
    const fn new() -> Self {
        Self {
            termination: None,
            shutdown_response: None,
        }
    }
}

enum HeadShutdownResponse {
    Disable(oneshot::Sender<TorqueDisableReport>),
    Preserve(oneshot::Sender<HoldPreservingOwnershipReleaseEvidence>),
}

enum HeadActorCleanup {
    TorqueDisable(Box<TorqueDisableReport>),
    HoldPreservingRelease(HoldPreservingOwnershipReleaseEvidence),
}

struct HeadActorRunExit {
    startup: Result<VerifiedNaturalHoldEvidence, HeadRuntimeError>,
    head_return: Option<Result<VerifiedHeadReturnEvidence, HeadReturnError>>,
    termination: ActorTermination,
    cleanup: HeadActorCleanup,
}

impl HeadActorRunExit {
    fn into_commissioning(self) -> ActorExit {
        let HeadActorCleanup::TorqueDisable(torque_disable) = self.cleanup else {
            unreachable!("commissioning actor always executes torque-disable cleanup");
        };
        ActorExit {
            startup: self.startup,
            head_return: self.head_return,
            termination: self.termination,
            torque_disable: *torque_disable,
        }
    }

    fn into_tension_preserving(self) -> TensionPreservingHeadActorExit {
        let HeadActorCleanup::HoldPreservingRelease(hold_preserving_release) = self.cleanup else {
            unreachable!("production actor never executes torque-disable cleanup");
        };
        TensionPreservingHeadActorExit {
            startup: self.startup,
            head_return: self.head_return,
            termination: self.termination,
            hold_preserving_release,
        }
    }
}

#[derive(Clone, Copy)]
enum ReturnOperationBudget<'a> {
    Initial {
        plan: HeadReturnPlan,
        started_at: MonotonicTime,
    },
    Moving(&'a HeadReturnController),
}

impl ReturnOperationBudget<'_> {
    fn remaining(self, now: MonotonicTime) -> Result<Duration, HeadMotionError> {
        match self {
            Self::Initial { plan, started_at } => {
                let elapsed = now.checked_duration_since(started_at).ok_or(
                    HeadMotionError::ClockRegression {
                        previous: started_at,
                        observed: now,
                    },
                )?;
                if elapsed >= plan.motion_timeout() {
                    return Err(HeadMotionError::MotionTimeout {
                        elapsed,
                        maximum: plan.motion_timeout(),
                    });
                }
                Ok(plan
                    .motion_timeout()
                    .checked_sub(elapsed)
                    .expect("elapsed is inside the initial return deadline"))
            }
            Self::Moving(controller) => controller.remaining_operation_budget(now),
        }
    }
}

enum ReturnFrameWriteFailure {
    Frame(FrameWriteError),
    Deadline {
        source: HeadMotionError,
        io: Option<FrameWriteError>,
    },
}

enum ReturnTelemetrySetFailure {
    Command(Box<HeadReturnError>),
    Read {
        joint: HeadJoint,
        source: RequestError,
    },
    Deadline {
        joint: HeadJoint,
        source: HeadMotionError,
        io: Option<RequestError>,
    },
    Admission(HeadMotionError),
}

impl<T, C> HeadActor<T, C>
where
    T: AsyncByteTransport,
    C: MonotonicClock,
{
    async fn run(
        mut self,
        mut commands: mpsc::Receiver<HeadCommand>,
        startup_sender: oneshot::Sender<Result<VerifiedNaturalHoldEvidence, HeadRuntimeError>>,
    ) -> HeadActorRunExit {
        let mut control = ControlState::new();
        let startup = self.startup(&mut commands, &mut control).await;
        let mut head_return = None;
        // Startup observation is optional to the actor's safety. If the caller
        // dropped its receipt, the complete result still remains in ActorExit.
        let _startup_receiver_present = startup_sender.send(startup.clone()).is_ok();

        let termination = if let Some(termination) = control.termination.clone() {
            termination
        } else {
            match startup.as_ref() {
                Err(_) => {
                    // Reject future commands, then drain a shutdown that was
                    // already queued while the failing operation was in flight.
                    commands.close();
                    match commands.try_recv() {
                        Ok(HeadCommand::Shutdown { response }) => {
                            control.shutdown_response =
                                Some(HeadShutdownResponse::Disable(response));
                            ActorTermination::StartupFaultWithShutdownRequested
                        }
                        Ok(HeadCommand::ReleaseOwnershipPreservingHold { response }) => {
                            control.shutdown_response =
                                Some(HeadShutdownResponse::Preserve(response));
                            ActorTermination::RequestedHoldPreservingRelease
                        }
                        Ok(HeadCommand::CheckHealth { response }) => {
                            let _requester_present = response
                                .send(Err(HeadHealthCheckError::CommandBeforeStartup))
                                .is_ok();
                            ActorTermination::StartupFault
                        }
                        Ok(HeadCommand::ReturnToTarget { response }) => {
                            let _requester_present = response
                                .send(Err(HeadReturnError::CommandBeforeStartup))
                                .is_ok();
                            ActorTermination::StartupFault
                        }
                        Err(
                            mpsc::error::TryRecvError::Empty
                            | mpsc::error::TryRecvError::Disconnected,
                        ) => ActorTermination::StartupFault,
                    }
                }
                Ok(evidence) => {
                    self.run_commands(
                        &mut commands,
                        &mut control,
                        evidence.observed_pose(),
                        &mut head_return,
                    )
                    .await
                }
            }
        };
        commands.close();

        let cleanup = match self.startup_torque_policy {
            StartupTorquePolicy::CommissioningDisableFirst => {
                HeadActorCleanup::TorqueDisable(Box::new(self.disable_all().await))
            }
            StartupTorquePolicy::TensionPreservingTakeover => {
                HeadActorCleanup::HoldPreservingRelease(HoldPreservingOwnershipReleaseEvidence {
                    recorded_at: self.clock.now(),
                })
            }
        };
        match (control.shutdown_response, &cleanup) {
            (
                Some(HeadShutdownResponse::Disable(response)),
                HeadActorCleanup::TorqueDisable(report),
            ) => {
                let _requester_present = response.send(report.as_ref().clone()).is_ok();
            }
            (
                Some(HeadShutdownResponse::Preserve(response)),
                HeadActorCleanup::HoldPreservingRelease(evidence),
            ) => {
                let _requester_present = response.send(evidence.clone()).is_ok();
            }
            (None, _) => {}
            (Some(_), _) => {
                unreachable!("private handle cleanup command always matches its actor policy")
            }
        }
        HeadActorRunExit {
            startup,
            head_return,
            termination,
            cleanup,
        }
    }

    async fn run_commands(
        &mut self,
        commands: &mut mpsc::Receiver<HeadCommand>,
        control: &mut ControlState,
        start_pose: HeadPose,
        head_return: &mut Option<Result<VerifiedHeadReturnEvidence, HeadReturnError>>,
    ) -> ActorTermination {
        let mut hold_target = HeadHoldTarget::StartupObserved(start_pose);
        loop {
            match commands.recv().await {
                Some(HeadCommand::Shutdown { response }) => {
                    control.shutdown_response = Some(HeadShutdownResponse::Disable(response));
                    return ActorTermination::RequestedShutdown;
                }
                Some(HeadCommand::ReleaseOwnershipPreservingHold { response }) => {
                    control.shutdown_response = Some(HeadShutdownResponse::Preserve(response));
                    return ActorTermination::RequestedHoldPreservingRelease;
                }
                Some(HeadCommand::CheckHealth { response }) => {
                    let result = self
                        .observe_natural_hold_health(hold_target, commands, control)
                        .await
                        .map_err(|source| HeadHealthCheckError::Observation(Box::new(source)));
                    let _requester_present = response.send(result).is_ok();
                    if let Some(termination) = control.termination.clone() {
                        return termination;
                    }
                }
                Some(HeadCommand::ReturnToTarget { response }) => {
                    if head_return.is_some() {
                        let _requester_present = response
                            .send(Err(HeadReturnError::CommandAlreadyAttempted))
                            .is_ok();
                        continue;
                    }
                    let plan = self
                        .return_plan
                        .expect("only a config-bound return handle can send this private command");
                    let result = self
                        .execute_return_to_target(start_pose, plan, commands, control)
                        .await;
                    let owner_retained_after_fault = result
                        .as_ref()
                        .is_err_and(HeadReturnError::retains_owner_after_fault);
                    hold_target = match &result {
                        Ok(evidence) => HeadHoldTarget::ReviewedReturn(evidence.target()),
                        Err(HeadReturnError::KinematicFaultExistingGoalRetained {
                            commanded_positions,
                            ..
                        }) => HeadHoldTarget::RecoverableReturnCommand(*commanded_positions),
                        Err(HeadReturnError::KinematicFaultRecoveryWritten {
                            held_positions,
                            ..
                        }) => HeadHoldTarget::RecoverableReturnCommand(*held_positions),
                        Err(_) => hold_target,
                    };
                    let _requester_present = response.send(result.clone()).is_ok();
                    *head_return = Some(result.clone());
                    if result.is_err() && !owner_retained_after_fault {
                        return control
                            .termination
                            .clone()
                            .unwrap_or(ActorTermination::HeadReturnFault);
                    }
                }
                None => return ActorTermination::HandleDropped,
            }
        }
    }

    async fn startup(
        &mut self,
        commands: &mut mpsc::Receiver<HeadCommand>,
        control: &mut ControlState,
    ) -> Result<VerifiedNaturalHoldEvidence, HeadRuntimeError> {
        let started_at = self.clock.now();
        let startup_torque = match self.startup_torque_policy {
            StartupTorquePolicy::CommissioningDisableFirst => {
                // Commissioning establishes a known host-commanded
                // torque-disabled baseline before any read request. All
                // joints are attempted even when one write fails.
                let report = self.disable_all().await;
                if !report.all_writes_completed() {
                    return Err(HeadRuntimeError::PreObservationTorqueDisable {
                        report: Box::new(report),
                    });
                }
                HeadStartupTorqueEvidence::CommissioningDisableFirst {
                    report: Box::new(report),
                }
            }
            StartupTorquePolicy::TensionPreservingTakeover => {
                // Production does not read or write the torque switch here and
                // therefore makes no prior-state claim. Its first traffic is
                // the position observation below.
                HeadStartupTorqueEvidence::TensionPreservingTakeover
            }
        };
        let bow = self
            .observe_joint(HeadJoint::Bow, commands, control)
            .await?;
        let curl = self
            .observe_joint(HeadJoint::Curl, commands, control)
            .await?;
        let yaw = self
            .observe_joint(HeadJoint::Yaw, commands, control)
            .await?;
        let roll = self
            .observe_joint(HeadJoint::Roll, commands, control)
            .await?;
        let observations = [bow, curl, yaw, roll];
        let observed_pose = HeadPose::try_from_validated([
            observations[0].validated(),
            observations[1].validated(),
            observations[2].validated(),
            observations[3].validated(),
        ])
        .map_err(|source| HeadRuntimeError::PoseAdmission { source })?;
        // This admission is deliberately inside the serial-owning actor and
        // before every configuration or torque-enable write. Callers receive
        // only the resulting evidence and cannot validate weak tick arrays a
        // second time after the head has already been energised.
        let configured_pose = self
            .configured_pose_bounds
            .admit(observed_pose)
            .map_err(|source| HeadRuntimeError::ConfiguredPoseAdmission { source })?;
        let oldest_observation_at = observations
            .iter()
            .flat_map(|observation| {
                [
                    observation.first().received_at(),
                    observation.second().received_at(),
                ]
            })
            .min()
            .expect("the exact head pose always has four observations");
        self.ensure_observation_freshness(
            oldest_observation_at,
            HeadJoint::Bow,
            ArmingFreshnessCheck::BeforeConfigurationWrites,
        )?;
        let frames = build_natural_hold_frames(
            observed_pose,
            self.config.torque_limits(),
            self.config.goal_speed(),
        );

        // Clamp torque before writing any goal, including the observed pose.
        let torque_limit_writes = self
            .write_stage(
                frames.torque_limit_writes(),
                RuntimeStage::WriteTorqueLimit,
                WritePurpose::TorqueLimit,
                commands,
                control,
            )
            .await?;
        let observed_goal_writes = self
            .write_stage(
                frames.goal_writes(),
                RuntimeStage::WriteObservedGoal,
                WritePurpose::ObservedGoal,
                commands,
                control,
            )
            .await?;
        let torque_enable_writes = self
            .write_enable_stage(
                frames.torque_enable_writes(),
                oldest_observation_at,
                commands,
                control,
            )
            .await?;

        let readbacks = [
            self.verify_joint(
                HeadJoint::Bow,
                observed_pose,
                &frames.verification_reads()[0],
                commands,
                control,
            )
            .await?,
            self.verify_joint(
                HeadJoint::Curl,
                observed_pose,
                &frames.verification_reads()[1],
                commands,
                control,
            )
            .await?,
            self.verify_joint(
                HeadJoint::Yaw,
                observed_pose,
                &frames.verification_reads()[2],
                commands,
                control,
            )
            .await?,
            self.verify_joint(
                HeadJoint::Roll,
                observed_pose,
                &frames.verification_reads()[3],
                commands,
                control,
            )
            .await?,
        ];

        Ok(VerifiedNaturalHoldEvidence {
            started_at,
            completed_at: self.clock.now(),
            startup_torque,
            observed_pose,
            configured_pose,
            observations,
            observed_goal_writes,
            torque_limit_writes,
            torque_enable_writes,
            readbacks,
        })
    }

    async fn execute_return_to_target(
        &mut self,
        startup_pose: HeadPose,
        plan: HeadReturnPlan,
        commands: &mut mpsc::Receiver<HeadCommand>,
        control: &mut ControlState,
    ) -> Result<VerifiedHeadReturnEvidence, HeadReturnError> {
        let started_at = self.clock.now();
        let mut commanded_positions = startup_pose.positions();
        let mut waypoint_writes = Vec::with_capacity(
            usize::try_from(plan.motion_timeout().as_millis() / plan.control_period().as_millis())
                .expect("fixed return cycle bound fits usize"),
        );
        let initial_budget = ReturnOperationBudget::Initial { plan, started_at };
        let start_first = match self
            .read_return_telemetry_set(commands, control, initial_budget, plan)
            .await
        {
            Ok(set) => set,
            Err(source) => {
                return Err(map_return_telemetry_failure(
                    source,
                    commanded_positions,
                    waypoint_writes,
                ));
            }
        };
        let start_second = match self
            .read_return_telemetry_set(commands, control, initial_budget, plan)
            .await
        {
            Ok(set) => set,
            Err(source) => {
                return Err(map_return_telemetry_failure(
                    source,
                    commanded_positions,
                    waypoint_writes,
                ));
            }
        };
        let start_pose = match admit_stopped_return_start(
            start_first,
            start_second,
            self.config.redundant_read_tolerance(),
        ) {
            Ok(pose) => pose,
            Err(source) => {
                return Err(retain_existing_goal_error(
                    source,
                    commanded_positions,
                    None,
                    None,
                    waypoint_writes,
                ));
            }
        };
        if let Err(source) = self.configured_pose_bounds.admit(start_pose) {
            return Err(retain_existing_goal_error(
                HeadMotionError::ReturnStartOutsideConfiguredBounds { source },
                commanded_positions,
                None,
                None,
                waypoint_writes,
            ));
        }
        let motion_started_at = self.clock.now();
        let mut controller =
            match HeadReturnController::try_new(plan, start_pose, started_at, motion_started_at) {
                Ok(controller) => controller,
                Err(source) => {
                    return Err(retain_existing_goal_error(
                        source,
                        commanded_positions,
                        None,
                        None,
                        waypoint_writes,
                    ));
                }
            };
        let mut interval = tokio::time::interval(plan.control_period());
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            interval.tick().await;
            if let Err(source) = self.check_return_control(
                commands,
                control,
                RuntimeStage::ReturnReadTelemetry,
                HeadJoint::Bow,
            ) {
                return Err(source.with_waypoint_writes(waypoint_writes));
            }
            if let Err(source) = controller.remaining_operation_budget(self.clock.now()) {
                return Err(retain_existing_goal_error(
                    source,
                    commanded_positions,
                    None,
                    None,
                    waypoint_writes,
                ));
            }
            let telemetry = match self
                .read_return_telemetry_set(
                    commands,
                    control,
                    ReturnOperationBudget::Moving(&controller),
                    plan,
                )
                .await
            {
                Ok(set) => set,
                Err(source) => {
                    return Err(map_return_telemetry_failure(
                        source,
                        commanded_positions,
                        waypoint_writes,
                    ));
                }
            };
            let action = match controller.advance(self.clock.now(), telemetry) {
                Ok(action) => action,
                Err(fault) if fault.recovery().is_some() => {
                    let positions = fault
                        .recovery()
                        .expect("guard admitted the recovery capability")
                        .positions();
                    match self
                        .write_return_waypoints(positions, commands, control, None)
                        .await
                    {
                        Ok(hold_writes) => {
                            return Err(HeadReturnError::KinematicFaultRecoveryWritten {
                                source: fault.into_source(),
                                held_positions: positions,
                                hold_writes: Box::new(hold_writes),
                                waypoint_writes,
                            });
                        }
                        Err(recovery_write) => {
                            return Err(HeadReturnError::KinematicFaultRecoveryWriteFailed {
                                source: fault.into_source(),
                                recovery_write: Box::new(recovery_write),
                                waypoint_writes,
                            });
                        }
                    }
                }
                Err(fault) if fault.source().permits_existing_goal_retention() => {
                    return Err(retain_existing_goal_error(
                        fault.into_source(),
                        commanded_positions,
                        None,
                        None,
                        waypoint_writes,
                    ));
                }
                Err(fault) => {
                    return Err(HeadReturnError::Motion {
                        source: fault.into_source(),
                        waypoint_writes,
                    });
                }
            };
            match action {
                HeadReturnAction::WriteWaypoints(positions) => {
                    let writes = match self
                        .write_return_waypoints(
                            positions,
                            commands,
                            control,
                            Some(ReturnOperationBudget::Moving(&controller)),
                        )
                        .await
                    {
                        Ok(writes) => writes,
                        Err(source) => {
                            apply_completed_waypoint_prefix(&mut commanded_positions, &source);
                            if let HeadWaypointBatchFailure::Deadline {
                                source: deadline, ..
                            } = source.failure()
                            {
                                return Err(retain_existing_goal_error(
                                    deadline.clone(),
                                    commanded_positions,
                                    None,
                                    Some(Box::new(source)),
                                    waypoint_writes,
                                ));
                            }
                            return Err(HeadReturnError::WaypointWrite {
                                source: Box::new(source),
                                waypoint_writes,
                            });
                        }
                    };
                    commanded_positions = positions;
                    controller.record_waypoint_written(positions);
                    waypoint_writes.push(HeadWaypointEvidence { positions, writes });
                }
                HeadReturnAction::AwaitSecondStoppedSample => {}
                HeadReturnAction::Complete(sets) => {
                    return Ok(VerifiedHeadReturnEvidence {
                        started_at,
                        completed_at: self.clock.now(),
                        start_pose,
                        start_first: start_first.into(),
                        start_second: start_second.into(),
                        target: plan.target(),
                        waypoint_writes,
                        first_stopped: sets[0].into(),
                        second_stopped: sets[1].into(),
                    });
                }
            }
        }
    }

    async fn read_return_telemetry_set(
        &mut self,
        commands: &mut mpsc::Receiver<HeadCommand>,
        control: &mut ControlState,
        budget: ReturnOperationBudget<'_>,
        plan: HeadReturnPlan,
    ) -> Result<FreshHeadTelemetrySet, ReturnTelemetrySetFailure> {
        let bow = self
            .read_return_joint(HeadJoint::Bow, commands, control, budget)
            .await?;
        let curl = self
            .read_return_joint(HeadJoint::Curl, commands, control, budget)
            .await?;
        let yaw = self
            .read_return_joint(HeadJoint::Yaw, commands, control, budget)
            .await?;
        let roll = self
            .read_return_joint(HeadJoint::Roll, commands, control, budget)
            .await?;
        let responses = [bow, curl, yaw, roll];
        let samples = std::array::from_fn(|index| *responses[index].value());
        let received_at = std::array::from_fn(|index| responses[index].received_at());
        FreshHeadTelemetrySet::try_new(
            samples,
            received_at,
            self.clock.now(),
            plan.telemetry_set_max_age(),
        )
        .map_err(ReturnTelemetrySetFailure::Admission)
    }

    async fn read_return_joint(
        &mut self,
        joint: HeadJoint,
        commands: &mut mpsc::Receiver<HeadCommand>,
        control: &mut ControlState,
        budget: ReturnOperationBudget<'_>,
    ) -> Result<ResponseEvidence<FullTelemetry>, ReturnTelemetrySetFailure> {
        self.check_return_control(commands, control, RuntimeStage::ReturnReadTelemetry, joint)
            .map_err(|source| ReturnTelemetrySetFailure::Command(Box::new(source)))?;
        let request = build_full_telemetry_read(joint.servo_id());
        let request_write = match self
            .write_frame_before_deadline(
                joint,
                WritePurpose::TelemetryReadRequest,
                request.as_bytes(),
                budget,
            )
            .await
        {
            Ok(evidence) => evidence,
            Err(ReturnFrameWriteFailure::Frame(source)) => {
                return Err(ReturnTelemetrySetFailure::Read {
                    joint,
                    source: RequestError::RequestWrite(source),
                });
            }
            Err(ReturnFrameWriteFailure::Deadline { source, io }) => {
                return Err(ReturnTelemetrySetFailure::Deadline {
                    joint,
                    source,
                    io: io.map(RequestError::RequestWrite),
                });
            }
        };
        let remaining = budget.remaining(self.clock.now()).map_err(|source| {
            ReturnTelemetrySetFailure::Deadline {
                joint,
                source,
                // The request write completed, but the operation budget expired
                // before a response read began. Do not fabricate an I/O failure.
                io: None,
            }
        })?;
        let frame = match read_response_frame(
            &mut self.transport,
            &self.clock,
            self.config.response_timeout().capped_by(remaining),
            self.config.noise_budget_bytes(),
        )
        .await
        {
            Ok(frame) => frame,
            Err(source) => {
                let deadline = budget.remaining(self.clock.now()).err();
                return Err(match deadline {
                    Some(deadline) => ReturnTelemetrySetFailure::Deadline {
                        joint,
                        source: deadline,
                        io: Some(RequestError::ResponseFrame(source)),
                    },
                    None => ReturnTelemetrySetFailure::Read {
                        joint,
                        source: RequestError::ResponseFrame(source),
                    },
                });
            }
        };
        let value = FullTelemetry::parse(frame.as_bytes(), joint.servo_id()).map_err(|source| {
            ReturnTelemetrySetFailure::Read {
                joint,
                source: RequestError::Telemetry(source),
            }
        })?;
        Ok(ResponseEvidence {
            value,
            request_write,
            discarded_noise_bytes: frame.discarded_noise_bytes(),
            received_at: self.clock.now(),
        })
    }

    async fn write_return_waypoints(
        &mut self,
        positions: [PositionTicks; 4],
        commands: &mut mpsc::Receiver<HeadCommand>,
        control: &mut ControlState,
        budget: Option<ReturnOperationBudget<'_>>,
    ) -> Result<[WriteEvidence; 4], HeadWaypointBatchWriteError> {
        let frames = HeadJoint::ALL.map(|joint| {
            build_goal_with_speed_write(
                joint.servo_id(),
                positions[joint as usize],
                self.config.goal_speed(),
            )
        });
        let mut completed_writes: [Option<WriteEvidence>; 4] = std::array::from_fn(|_| None);
        for (index, joint) in HeadJoint::ALL.into_iter().enumerate() {
            if let Err(error) = self.check_return_control(
                commands,
                control,
                RuntimeStage::ReturnWriteWaypoint,
                joint,
            ) {
                let HeadReturnError::Cancelled {
                    cause,
                    stage,
                    joint,
                    ..
                } = error
                else {
                    unreachable!("only cancellation can stop an in-flight private batch")
                };
                return Err(HeadWaypointBatchWriteError {
                    positions,
                    completed_writes,
                    failure: HeadWaypointBatchFailure::Cancelled {
                        cause,
                        stage,
                        joint,
                    },
                });
            }
            let result = match budget {
                Some(budget) => self
                    .write_frame_before_deadline(
                        joint,
                        WritePurpose::ReturnWaypoint,
                        frames[index].as_bytes(),
                        budget,
                    )
                    .await
                    .map_err(|source| match source {
                        ReturnFrameWriteFailure::Frame(source) => {
                            HeadWaypointBatchFailure::Frame(source)
                        }
                        ReturnFrameWriteFailure::Deadline { source, io } => {
                            HeadWaypointBatchFailure::Deadline { source, io }
                        }
                    }),
                None => self
                    .write_frame(
                        joint,
                        WritePurpose::ReturnWaypoint,
                        frames[index].as_bytes(),
                    )
                    .await
                    .map_err(HeadWaypointBatchFailure::Frame),
            };
            match result {
                Ok(evidence) => completed_writes[index] = Some(evidence),
                Err(failure) => {
                    return Err(HeadWaypointBatchWriteError {
                        positions,
                        completed_writes,
                        failure,
                    });
                }
            }
        }
        Ok(completed_writes
            .map(|write| write.expect("all four batch writes completed on the success path")))
    }

    async fn write_frame_before_deadline(
        &mut self,
        joint: HeadJoint,
        purpose: WritePurpose,
        bytes: &[u8],
        budget: ReturnOperationBudget<'_>,
    ) -> Result<WriteEvidence, ReturnFrameWriteFailure> {
        let mut recovered_failures = Vec::new();
        let maximum_attempts = self.config.write_attempts().get();
        let mut attempt = 1_u8;
        loop {
            let remaining = budget
                .remaining(self.clock.now())
                .map_err(|source| ReturnFrameWriteFailure::Deadline { source, io: None })?;
            let timeout = self.config.write_timeout().get().min(remaining);
            match self.transport.write_all(bytes, timeout).await {
                Ok(()) => {
                    return Ok(WriteEvidence {
                        attempts_used: attempt,
                        recovered_failures,
                        completed_at: self.clock.now(),
                    });
                }
                Err(source) => {
                    let frame_error = FrameWriteError {
                        joint,
                        purpose,
                        attempts_used: attempt,
                        recovered_failures: recovered_failures.clone(),
                        source: source.clone(),
                    };
                    if let Err(deadline) = budget.remaining(self.clock.now()) {
                        return Err(ReturnFrameWriteFailure::Deadline {
                            source: deadline,
                            io: Some(frame_error),
                        });
                    }
                    if attempt < maximum_attempts && source.is_retryable_without_progress() {
                        recovered_failures.push(source);
                        attempt += 1;
                        continue;
                    }
                    return Err(ReturnFrameWriteFailure::Frame(frame_error));
                }
            }
        }
    }

    fn check_return_control(
        &self,
        commands: &mut mpsc::Receiver<HeadCommand>,
        control: &mut ControlState,
        stage: RuntimeStage,
        joint: HeadJoint,
    ) -> Result<(), HeadReturnError> {
        match commands.try_recv() {
            Ok(HeadCommand::Shutdown { response }) => {
                control.termination = Some(ActorTermination::RequestedShutdown);
                control.shutdown_response = Some(HeadShutdownResponse::Disable(response));
                Err(HeadReturnError::Cancelled {
                    cause: CancellationCause::RequestedShutdown,
                    stage,
                    joint,
                    waypoint_writes: Vec::new(),
                })
            }
            Ok(HeadCommand::ReleaseOwnershipPreservingHold { response }) => {
                control.termination = Some(ActorTermination::RequestedHoldPreservingRelease);
                control.shutdown_response = Some(HeadShutdownResponse::Preserve(response));
                Err(HeadReturnError::Cancelled {
                    cause: CancellationCause::RequestedHoldPreservingRelease,
                    stage,
                    joint,
                    waypoint_writes: Vec::new(),
                })
            }
            Ok(HeadCommand::ReturnToTarget { response }) => {
                let _requester_present = response
                    .send(Err(HeadReturnError::CommandAlreadyInProgress))
                    .is_ok();
                Ok(())
            }
            Ok(HeadCommand::CheckHealth { response }) => {
                let _requester_present = response
                    .send(Err(HeadHealthCheckError::CommandAlreadyInProgress))
                    .is_ok();
                Ok(())
            }
            Err(mpsc::error::TryRecvError::Disconnected) => {
                control.termination = Some(ActorTermination::HandleDropped);
                Err(HeadReturnError::Cancelled {
                    cause: CancellationCause::HandleDropped,
                    stage,
                    joint,
                    waypoint_writes: Vec::new(),
                })
            }
            Err(mpsc::error::TryRecvError::Empty) => Ok(()),
        }
    }

    async fn observe_natural_hold_health(
        &mut self,
        hold_target: HeadHoldTarget,
        commands: &mut mpsc::Receiver<HeadCommand>,
        control: &mut ControlState,
    ) -> Result<VerifiedHeadHealthEvidence, HeadHealthObservationError> {
        let started_at = self.clock.now();
        let tolerance = self.config.readback_tolerance();
        let mut accepted_prefix: [Option<HeadHealthJointEvidence>; 4] =
            std::array::from_fn(|_| None);
        let mut previous_at = started_at;

        for (index, joint) in HeadJoint::ALL.into_iter().enumerate() {
            if let Err(failure) = self.check_health_control(commands, control, joint) {
                return Err(HeadHealthObservationError {
                    started_at,
                    accepted_prefix,
                    failure,
                });
            }

            let request = build_full_telemetry_read(joint.servo_id());
            let response = match self.read_telemetry(joint, &request).await {
                Ok(response) => response,
                Err(source) => {
                    return Err(HeadHealthObservationError {
                        started_at,
                        accepted_prefix,
                        failure: HeadHealthFailure::TelemetryRead { joint, source },
                    });
                }
            };

            let write_completed_at = response.request_write().completed_at();
            if write_completed_at < previous_at {
                return Err(HeadHealthObservationError {
                    started_at,
                    accepted_prefix,
                    failure: HeadHealthFailure::ClockRegression {
                        boundary: HeadHealthClockBoundary::RequestWriteCompleted { joint },
                        previous: previous_at,
                        observed: write_completed_at,
                        current_response: Some(Box::new(response)),
                    },
                });
            }
            let received_at = response.received_at();
            if received_at < write_completed_at {
                return Err(HeadHealthObservationError {
                    started_at,
                    accepted_prefix,
                    failure: HeadHealthFailure::ClockRegression {
                        boundary: HeadHealthClockBoundary::ResponseReceived { joint },
                        previous: write_completed_at,
                        observed: received_at,
                        current_response: Some(Box::new(response)),
                    },
                });
            }

            let telemetry = *response.value();
            if telemetry.device_status_raw() != 0 {
                return Err(HeadHealthObservationError {
                    started_at,
                    accepted_prefix,
                    failure: HeadHealthFailure::DeviceStatus {
                        joint,
                        raw: telemetry.device_status_raw(),
                        response,
                    },
                });
            }
            if telemetry.is_moving() {
                return Err(HeadHealthObservationError {
                    started_at,
                    accepted_prefix,
                    failure: HeadHealthFailure::Moving {
                        joint,
                        position: telemetry.position(),
                        response,
                    },
                });
            }

            let target = hold_target.position(joint);
            let absolute_difference_ticks = target.get().abs_diff(telemetry.position().get());
            if absolute_difference_ticks > tolerance.get() {
                return Err(HeadHealthObservationError {
                    started_at,
                    accepted_prefix,
                    failure: HeadHealthFailure::PositionMismatch {
                        joint,
                        target,
                        actual: telemetry.position(),
                        absolute_difference_ticks,
                        tolerance,
                        response,
                    },
                });
            }

            accepted_prefix[index] = Some(HeadHealthJointEvidence {
                joint,
                target,
                absolute_difference_ticks,
                response,
            });
            previous_at = received_at;
        }

        let completed_at = self.clock.now();
        if completed_at < previous_at {
            return Err(HeadHealthObservationError {
                started_at,
                accepted_prefix,
                failure: HeadHealthFailure::ClockRegression {
                    boundary: HeadHealthClockBoundary::CheckCompleted,
                    previous: previous_at,
                    observed: completed_at,
                    current_response: None,
                },
            });
        }

        let joints = accepted_prefix.map(|evidence| {
            evidence.expect("all four canonical joints were admitted on the success path")
        });
        Ok(VerifiedHeadHealthEvidence {
            started_at,
            completed_at,
            hold_target,
            tolerance,
            joints,
        })
    }

    fn check_health_control(
        &self,
        commands: &mut mpsc::Receiver<HeadCommand>,
        control: &mut ControlState,
        joint: HeadJoint,
    ) -> Result<(), HeadHealthFailure> {
        match commands.try_recv() {
            Ok(HeadCommand::Shutdown { response }) => {
                control.termination = Some(ActorTermination::RequestedShutdown);
                control.shutdown_response = Some(HeadShutdownResponse::Disable(response));
                Err(HeadHealthFailure::Cancelled {
                    cause: CancellationCause::RequestedShutdown,
                    stage: RuntimeStage::HealthReadTelemetry,
                    joint,
                })
            }
            Ok(HeadCommand::ReleaseOwnershipPreservingHold { response }) => {
                control.termination = Some(ActorTermination::RequestedHoldPreservingRelease);
                control.shutdown_response = Some(HeadShutdownResponse::Preserve(response));
                Err(HeadHealthFailure::Cancelled {
                    cause: CancellationCause::RequestedHoldPreservingRelease,
                    stage: RuntimeStage::HealthReadTelemetry,
                    joint,
                })
            }
            Ok(HeadCommand::CheckHealth { response }) => {
                let _requester_present = response
                    .send(Err(HeadHealthCheckError::CommandAlreadyInProgress))
                    .is_ok();
                Ok(())
            }
            Ok(HeadCommand::ReturnToTarget { response }) => {
                let _requester_present = response
                    .send(Err(HeadReturnError::CommandAlreadyInProgress))
                    .is_ok();
                Ok(())
            }
            Err(mpsc::error::TryRecvError::Disconnected) => {
                control.termination = Some(ActorTermination::HandleDropped);
                Err(HeadHealthFailure::Cancelled {
                    cause: CancellationCause::HandleDropped,
                    stage: RuntimeStage::HealthReadTelemetry,
                    joint,
                })
            }
            Err(mpsc::error::TryRecvError::Empty) => Ok(()),
        }
    }

    async fn observe_joint(
        &mut self,
        joint: HeadJoint,
        commands: &mut mpsc::Receiver<HeadCommand>,
        control: &mut ControlState,
    ) -> Result<PositionObservationEvidence, HeadRuntimeError> {
        self.check_control(commands, control, RuntimeStage::ObserveFirst, joint)?;
        let first = self.read_position(joint).await.map_err(|source| {
            HeadRuntimeError::PositionObservation {
                joint,
                stage: RuntimeStage::ObserveFirst,
                source,
            }
        })?;
        self.check_control(commands, control, RuntimeStage::ObserveSecond, joint)?;
        let second = self.read_position(joint).await.map_err(|source| {
            HeadRuntimeError::PositionObservation {
                joint,
                stage: RuntimeStage::ObserveSecond,
                source,
            }
        })?;
        let validated = ValidatedPresentPosition::try_from_pair(
            *first.value(),
            *second.value(),
            self.config.redundant_read_tolerance(),
        )
        .map_err(|source| HeadRuntimeError::PositionAgreement { joint, source })?;
        Ok(PositionObservationEvidence {
            joint,
            first,
            second,
            validated,
        })
    }

    async fn read_position(
        &mut self,
        joint: HeadJoint,
    ) -> Result<ResponseEvidence<PresentPosition>, RequestError> {
        let id = joint.servo_id();
        let request_write = self
            .write_frame(
                joint,
                WritePurpose::PositionReadRequest,
                build_position_read(id).as_bytes(),
            )
            .await
            .map_err(RequestError::RequestWrite)?;
        let frame = read_response_frame(
            &mut self.transport,
            &self.clock,
            self.config.response_timeout(),
            self.config.noise_budget_bytes(),
        )
        .await
        .map_err(RequestError::ResponseFrame)?;
        let value =
            PresentPosition::parse(frame.as_bytes(), id).map_err(RequestError::Telemetry)?;
        Ok(ResponseEvidence {
            value,
            request_write,
            discarded_noise_bytes: frame.discarded_noise_bytes(),
            received_at: self.clock.now(),
        })
    }

    async fn write_stage(
        &mut self,
        frames: &[kiko_head_protocol::CommandFrame; 4],
        stage: RuntimeStage,
        purpose: WritePurpose,
        commands: &mut mpsc::Receiver<HeadCommand>,
        control: &mut ControlState,
    ) -> Result<[WriteEvidence; 4], HeadRuntimeError> {
        let bow = self
            .write_controlled(
                HeadJoint::Bow,
                &frames[0],
                stage,
                purpose,
                commands,
                control,
            )
            .await?;
        let curl = self
            .write_controlled(
                HeadJoint::Curl,
                &frames[1],
                stage,
                purpose,
                commands,
                control,
            )
            .await?;
        let yaw = self
            .write_controlled(
                HeadJoint::Yaw,
                &frames[2],
                stage,
                purpose,
                commands,
                control,
            )
            .await?;
        let roll = self
            .write_controlled(
                HeadJoint::Roll,
                &frames[3],
                stage,
                purpose,
                commands,
                control,
            )
            .await?;
        Ok([bow, curl, yaw, roll])
    }

    async fn write_enable_stage(
        &mut self,
        frames: &[kiko_head_protocol::CommandFrame; 4],
        oldest_observation_at: MonotonicTime,
        commands: &mut mpsc::Receiver<HeadCommand>,
        control: &mut ControlState,
    ) -> Result<[WriteEvidence; 4], HeadRuntimeError> {
        let bow = self
            .write_enable_controlled(
                HeadJoint::Bow,
                &frames[0],
                oldest_observation_at,
                commands,
                control,
            )
            .await?;
        let curl = self
            .write_enable_controlled(
                HeadJoint::Curl,
                &frames[1],
                oldest_observation_at,
                commands,
                control,
            )
            .await?;
        let yaw = self
            .write_enable_controlled(
                HeadJoint::Yaw,
                &frames[2],
                oldest_observation_at,
                commands,
                control,
            )
            .await?;
        let roll = self
            .write_enable_controlled(
                HeadJoint::Roll,
                &frames[3],
                oldest_observation_at,
                commands,
                control,
            )
            .await?;
        Ok([bow, curl, yaw, roll])
    }

    async fn write_enable_controlled(
        &mut self,
        joint: HeadJoint,
        frame: &kiko_head_protocol::CommandFrame,
        oldest_observation_at: MonotonicTime,
        commands: &mut mpsc::Receiver<HeadCommand>,
        control: &mut ControlState,
    ) -> Result<WriteEvidence, HeadRuntimeError> {
        self.check_control(commands, control, RuntimeStage::EnableTorque, joint)?;
        self.ensure_arming_write_budget(oldest_observation_at, joint)?;
        let evidence = self
            .write_frame(joint, WritePurpose::TorqueEnable, frame.as_bytes())
            .await
            .map_err(|source| HeadRuntimeError::Write {
                stage: RuntimeStage::EnableTorque,
                source,
            })?;
        self.ensure_observation_freshness(
            oldest_observation_at,
            joint,
            ArmingFreshnessCheck::AfterEnableWrite,
        )?;
        Ok(evidence)
    }

    fn ensure_observation_freshness(
        &self,
        oldest_observation_at: MonotonicTime,
        joint: HeadJoint,
        check: ArmingFreshnessCheck,
    ) -> Result<(), HeadRuntimeError> {
        let (checked_at, age) = self.observation_age(oldest_observation_at)?;
        let maximum_age = self.config.arming_freshness().get();
        if age > maximum_age {
            return Err(HeadRuntimeError::ObservationStaleBeforeArming {
                joint,
                check,
                oldest_observation_at,
                checked_at,
                age,
                maximum_age,
            });
        }
        Ok(())
    }

    fn ensure_arming_write_budget(
        &self,
        oldest_observation_at: MonotonicTime,
        joint: HeadJoint,
    ) -> Result<(), HeadRuntimeError> {
        let (checked_at, age) = self.observation_age(oldest_observation_at)?;
        let maximum_age = self.config.arming_freshness().get();
        if age > maximum_age {
            return Err(HeadRuntimeError::ObservationStaleBeforeArming {
                joint,
                check: ArmingFreshnessCheck::BeforeEnableWrite,
                oldest_observation_at,
                checked_at,
                age,
                maximum_age,
            });
        }
        let remaining_freshness = maximum_age
            .checked_sub(age)
            .expect("age was admitted inside the freshness bound");
        let required_write_budget = self
            .config
            .write_timeout()
            .get()
            .checked_mul(u32::from(self.config.write_attempts().get()))
            .expect("parsed timeout and attempt bounds fit Duration");
        if remaining_freshness < required_write_budget {
            return Err(HeadRuntimeError::ObservationArmingWriteBudgetInsufficient {
                joint,
                oldest_observation_at,
                checked_at,
                remaining_freshness,
                required_write_budget,
            });
        }
        Ok(())
    }

    fn observation_age(
        &self,
        oldest_observation_at: MonotonicTime,
    ) -> Result<(MonotonicTime, Duration), HeadRuntimeError> {
        let checked_at = self.clock.now();
        let age = checked_at
            .checked_duration_since(oldest_observation_at)
            .ok_or(HeadRuntimeError::ObservationClockRegression {
                oldest_observation_at,
                checked_at,
            })?;
        Ok((checked_at, age))
    }

    async fn write_controlled(
        &mut self,
        joint: HeadJoint,
        frame: &kiko_head_protocol::CommandFrame,
        stage: RuntimeStage,
        purpose: WritePurpose,
        commands: &mut mpsc::Receiver<HeadCommand>,
        control: &mut ControlState,
    ) -> Result<WriteEvidence, HeadRuntimeError> {
        self.check_control(commands, control, stage, joint)?;
        self.write_frame(joint, purpose, frame.as_bytes())
            .await
            .map_err(|source| HeadRuntimeError::Write { stage, source })
    }

    async fn verify_joint(
        &mut self,
        joint: HeadJoint,
        pose: HeadPose,
        request: &kiko_head_protocol::CommandFrame,
        commands: &mut mpsc::Receiver<HeadCommand>,
        control: &mut ControlState,
    ) -> Result<ReadbackEvidence, HeadRuntimeError> {
        self.check_control(
            commands,
            control,
            RuntimeStage::VerifyFirstStoppedPosition,
            joint,
        )?;
        let target = pose.position(joint);
        let first = self
            .read_telemetry(joint, request)
            .await
            .map_err(|source| HeadRuntimeError::VerificationRead { joint, source })?;
        let first_target_difference_ticks =
            self.admit_stopped_readback(joint, VerificationSample::First, target, first.value())?;

        self.check_control(
            commands,
            control,
            RuntimeStage::VerifySecondStoppedPosition,
            joint,
        )?;
        let second = self
            .read_telemetry(joint, request)
            .await
            .map_err(|source| HeadRuntimeError::VerificationRead { joint, source })?;
        let second_target_difference_ticks =
            self.admit_stopped_readback(joint, VerificationSample::Second, target, second.value())?;
        let stable_difference_ticks = first
            .value()
            .position()
            .get()
            .abs_diff(second.value().position().get());
        if stable_difference_ticks > self.config.readback_tolerance().get() {
            return Err(HeadRuntimeError::ReadbackUnstable {
                joint,
                first: first.value().position(),
                second: second.value().position(),
                absolute_difference_ticks: stable_difference_ticks,
                tolerance: self.config.readback_tolerance(),
            });
        }

        Ok(ReadbackEvidence {
            joint,
            target,
            first_target_difference_ticks,
            second_target_difference_ticks,
            stable_difference_ticks,
            first,
            second,
        })
    }

    async fn read_telemetry(
        &mut self,
        joint: HeadJoint,
        request: &kiko_head_protocol::CommandFrame,
    ) -> Result<ResponseEvidence<FullTelemetry>, RequestError> {
        let id = joint.servo_id();
        let request_write = self
            .write_frame(
                joint,
                WritePurpose::TelemetryReadRequest,
                request.as_bytes(),
            )
            .await
            .map_err(RequestError::RequestWrite)?;
        let frame = read_response_frame(
            &mut self.transport,
            &self.clock,
            self.config.response_timeout(),
            self.config.noise_budget_bytes(),
        )
        .await
        .map_err(RequestError::ResponseFrame)?;
        let value = FullTelemetry::parse(frame.as_bytes(), id).map_err(RequestError::Telemetry)?;
        Ok(ResponseEvidence {
            value,
            request_write,
            discarded_noise_bytes: frame.discarded_noise_bytes(),
            received_at: self.clock.now(),
        })
    }

    fn admit_stopped_readback(
        &self,
        joint: HeadJoint,
        sample: VerificationSample,
        target: PositionTicks,
        telemetry: &FullTelemetry,
    ) -> Result<u16, HeadRuntimeError> {
        if telemetry.device_status_raw() != 0 {
            return Err(HeadRuntimeError::ReadbackDeviceStatus {
                joint,
                sample,
                position: telemetry.position(),
                raw: telemetry.device_status_raw(),
            });
        }
        if telemetry.is_moving() {
            return Err(HeadRuntimeError::ReadbackMoving {
                joint,
                sample,
                position: telemetry.position(),
            });
        }
        let absolute_difference_ticks = target.get().abs_diff(telemetry.position().get());
        if absolute_difference_ticks > self.config.readback_tolerance().get() {
            return Err(HeadRuntimeError::ReadbackMismatch {
                joint,
                sample,
                target,
                actual: telemetry.position(),
                absolute_difference_ticks,
                tolerance: self.config.readback_tolerance(),
            });
        }
        Ok(absolute_difference_ticks)
    }

    async fn write_frame(
        &mut self,
        joint: HeadJoint,
        purpose: WritePurpose,
        bytes: &[u8],
    ) -> Result<WriteEvidence, FrameWriteError> {
        // The common path does not allocate. Capacity grows only when an
        // explicitly configured, retryable zero-progress failure occurs.
        let mut recovered_failures = Vec::new();
        let maximum_attempts = self.config.write_attempts().get();
        let mut attempt = 1_u8;
        loop {
            match self
                .transport
                .write_all(bytes, self.config.write_timeout().get())
                .await
            {
                Ok(()) => {
                    return Ok(WriteEvidence {
                        attempts_used: attempt,
                        recovered_failures,
                        completed_at: self.clock.now(),
                    });
                }
                Err(source)
                    if attempt < maximum_attempts && source.is_retryable_without_progress() =>
                {
                    recovered_failures.push(source);
                    attempt += 1;
                }
                Err(source) => {
                    return Err(FrameWriteError {
                        joint,
                        purpose,
                        attempts_used: attempt,
                        recovered_failures,
                        source,
                    });
                }
            }
        }
    }

    fn check_control(
        &self,
        commands: &mut mpsc::Receiver<HeadCommand>,
        control: &mut ControlState,
        stage: RuntimeStage,
        joint: HeadJoint,
    ) -> Result<(), HeadRuntimeError> {
        match commands.try_recv() {
            Ok(HeadCommand::Shutdown { response }) => {
                control.termination = Some(ActorTermination::RequestedShutdown);
                control.shutdown_response = Some(HeadShutdownResponse::Disable(response));
                Err(HeadRuntimeError::Cancelled {
                    cause: CancellationCause::RequestedShutdown,
                    stage,
                    joint,
                })
            }
            Ok(HeadCommand::ReleaseOwnershipPreservingHold { response }) => {
                control.termination = Some(ActorTermination::RequestedHoldPreservingRelease);
                control.shutdown_response = Some(HeadShutdownResponse::Preserve(response));
                Err(HeadRuntimeError::Cancelled {
                    cause: CancellationCause::RequestedHoldPreservingRelease,
                    stage,
                    joint,
                })
            }
            Ok(HeadCommand::ReturnToTarget { response }) => {
                let _requester_present = response
                    .send(Err(HeadReturnError::CommandBeforeStartup))
                    .is_ok();
                Ok(())
            }
            Ok(HeadCommand::CheckHealth { response }) => {
                let _requester_present = response
                    .send(Err(HeadHealthCheckError::CommandBeforeStartup))
                    .is_ok();
                Ok(())
            }
            Err(mpsc::error::TryRecvError::Disconnected) => {
                control.termination = Some(ActorTermination::HandleDropped);
                Err(HeadRuntimeError::Cancelled {
                    cause: CancellationCause::HandleDropped,
                    stage,
                    joint,
                })
            }
            Err(mpsc::error::TryRecvError::Empty) => Ok(()),
        }
    }

    async fn disable_all(&mut self) -> TorqueDisableReport {
        let started_at = self.clock.now();
        let bow = self.disable_joint(HeadJoint::Bow).await;
        let curl = self.disable_joint(HeadJoint::Curl).await;
        let yaw = self.disable_joint(HeadJoint::Yaw).await;
        let roll = self.disable_joint(HeadJoint::Roll).await;
        TorqueDisableReport {
            started_at,
            completed_at: self.clock.now(),
            outcomes: [bow, curl, yaw, roll],
        }
    }

    async fn disable_joint(&mut self, joint: HeadJoint) -> TorqueDisableJointOutcome {
        let frame = build_torque_switch_write(joint.servo_id(), TorqueSwitch::Disabled);
        TorqueDisableJointOutcome {
            joint,
            result: self
                .write_frame(joint, WritePurpose::TorqueDisable, frame.as_bytes())
                .await,
        }
    }
}

fn retain_existing_goal_error(
    source: HeadMotionError,
    commanded_positions: [PositionTicks; 4],
    interrupted_read: Option<Box<InterruptedTelemetryRead>>,
    interrupted_write: Option<Box<HeadWaypointBatchWriteError>>,
    waypoint_writes: Vec<HeadWaypointEvidence>,
) -> HeadReturnError {
    HeadReturnError::KinematicFaultExistingGoalRetained {
        source,
        commanded_positions,
        interrupted_read,
        interrupted_write,
        waypoint_writes,
    }
}

fn map_return_telemetry_failure(
    failure: ReturnTelemetrySetFailure,
    commanded_positions: [PositionTicks; 4],
    waypoint_writes: Vec<HeadWaypointEvidence>,
) -> HeadReturnError {
    match failure {
        ReturnTelemetrySetFailure::Command(source) => {
            (*source).with_waypoint_writes(waypoint_writes)
        }
        ReturnTelemetrySetFailure::Deadline { joint, source, io } => retain_existing_goal_error(
            source,
            commanded_positions,
            Some(Box::new(InterruptedTelemetryRead { joint, source: io })),
            None,
            waypoint_writes,
        ),
        ReturnTelemetrySetFailure::Read { joint, source } => HeadReturnError::TelemetryRead {
            joint,
            source,
            waypoint_writes,
        },
        ReturnTelemetrySetFailure::Admission(source)
            if source.permits_existing_goal_retention() =>
        {
            retain_existing_goal_error(source, commanded_positions, None, None, waypoint_writes)
        }
        ReturnTelemetrySetFailure::Admission(source) => HeadReturnError::Motion {
            source,
            waypoint_writes,
        },
    }
}

fn apply_completed_waypoint_prefix(
    commanded_positions: &mut [PositionTicks; 4],
    error: &HeadWaypointBatchWriteError,
) {
    for (index, completed) in error.completed_writes().iter().enumerate() {
        if completed.is_some() {
            commanded_positions[index] = error.positions()[index];
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, VecDeque};
    use std::io;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use kiko_head_protocol::{ResponseParseError, ServoId, TelemetryParseError};

    use super::*;
    use crate::config::{
        HeadProbeConfig, HeadProbeConfigInput, HeadRuntimeConfigInput, ReturnToTargetConfigInput,
    };
    use crate::transport::{TransportFailureKind, TransportOperation};

    #[derive(Clone, Default)]
    struct TestClock {
        nanoseconds: Arc<AtomicU64>,
    }

    impl TestClock {
        fn advance_one_millisecond(&self) {
            self.nanoseconds.fetch_add(1_000_000, Ordering::Relaxed);
        }

        fn set_milliseconds(&self, milliseconds: u64) {
            self.nanoseconds
                .store(milliseconds * 1_000_000, Ordering::Relaxed);
        }
    }

    impl MonotonicClock for TestClock {
        fn now(&self) -> MonotonicTime {
            MonotonicTime::from_duration_since_origin(Duration::from_nanos(
                self.nanoseconds.load(Ordering::Relaxed),
            ))
        }
    }

    #[derive(Clone, Debug)]
    enum ReadAction {
        Bytes(Vec<u8>),
        SetClockAndBytes {
            milliseconds: u64,
            bytes: Vec<u8>,
        },
        Eof,
        Failure(TransportFailure),
        GatedFailure {
            entered: Arc<tokio::sync::Notify>,
            release: Arc<tokio::sync::Notify>,
            source: TransportFailure,
        },
    }

    #[derive(Default)]
    struct FakeShared {
        writes: Vec<Vec<u8>>,
        write_timeouts: Vec<Duration>,
        write_failures: BTreeMap<usize, TransportFailure>,
        read_calls: usize,
        read_timeouts: Vec<Duration>,
    }

    struct FakeTransport {
        clock: TestClock,
        reads: VecDeque<ReadAction>,
        pending: VecDeque<u8>,
        shared: Arc<Mutex<FakeShared>>,
    }

    impl FakeTransport {
        fn new(clock: TestClock, reads: Vec<ReadAction>) -> (Self, Arc<Mutex<FakeShared>>) {
            let shared = Arc::new(Mutex::new(FakeShared::default()));
            (
                Self {
                    clock,
                    reads: reads.into(),
                    pending: VecDeque::new(),
                    shared: Arc::clone(&shared),
                },
                shared,
            )
        }
    }

    impl AsyncByteTransport for FakeTransport {
        async fn write_all(
            &mut self,
            bytes: &[u8],
            timeout: Duration,
        ) -> Result<(), TransportFailure> {
            self.clock.advance_one_millisecond();
            let mut shared = self.shared.lock().expect("fake transport mutex");
            let call = shared.writes.len();
            shared.writes.push(bytes.to_vec());
            shared.write_timeouts.push(timeout);
            match shared.write_failures.remove(&call) {
                Some(source) => Err(source),
                None => Ok(()),
            }
        }

        async fn read_some(
            &mut self,
            bytes: &mut [u8],
            timeout: Duration,
        ) -> Result<usize, TransportFailure> {
            self.clock.advance_one_millisecond();
            {
                let mut shared = self.shared.lock().expect("fake transport mutex");
                shared.read_calls += 1;
                shared.read_timeouts.push(timeout);
            }
            loop {
                if !self.pending.is_empty() {
                    let read = bytes.len().min(self.pending.len());
                    for destination in &mut bytes[..read] {
                        *destination = self.pending.pop_front().expect("pending byte");
                    }
                    return Ok(read);
                }
                match self.reads.pop_front() {
                    Some(ReadAction::Bytes(chunk)) => self.pending.extend(chunk),
                    Some(ReadAction::SetClockAndBytes {
                        milliseconds,
                        bytes,
                    }) => {
                        self.clock.set_milliseconds(milliseconds);
                        self.pending.extend(bytes);
                    }
                    Some(ReadAction::Eof) | None => return Ok(0),
                    Some(ReadAction::Failure(source)) => return Err(source),
                    Some(ReadAction::GatedFailure {
                        entered,
                        release,
                        source,
                    }) => {
                        entered.notify_one();
                        release.notified().await;
                        return Err(source);
                    }
                }
            }
        }
    }

    fn valid_config(write_attempts: u8) -> HeadRuntimeConfig {
        config_with_freshness(write_attempts, 250)
    }

    fn config_with_freshness(write_attempts: u8, arming_freshness_ms: u64) -> HeadRuntimeConfig {
        HeadRuntimeConfig::parse(HeadRuntimeConfigInput {
            device_path: "/dev/serial/by-id/usb-Kiko_head_test".to_owned(),
            response_timeout_ms: 100,
            write_timeout_ms: 1,
            arming_freshness_ms,
            write_attempts,
            noise_budget_bytes: 16,
            redundant_read_tolerance_ticks: 10,
            readback_tolerance_ticks: 20,
            goal_speed_ticks_per_second: 100,
            torque_limit_permille: [600, 400, 400, 400],
        })
        .expect("test configuration")
    }

    fn valid_pose_bounds() -> ConfiguredHeadPoseBounds {
        ConfiguredHeadPoseBounds::try_new(
            [2_000, 2_450, 2_800, 2_800],
            [2_250, 2_700, 3_050, 3_050],
        )
        .expect("bounded test pose windows")
    }

    fn status(id: ServoId, parameters: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0xff, 0xff, id.get(), 0, 0];
        bytes[3] = u8::try_from(parameters.len() + 2).expect("test response length");
        bytes.extend_from_slice(parameters);
        let checksum = !bytes[2..]
            .iter()
            .fold(0_u8, |sum, byte| sum.wrapping_add(*byte));
        bytes.push(checksum);
        bytes
    }

    fn position_response(joint: HeadJoint, position: u16) -> Vec<u8> {
        status(joint.servo_id(), &position.to_le_bytes())
    }

    fn telemetry_response(joint: HeadJoint, position: u16) -> Vec<u8> {
        telemetry_response_with_moving(joint, position, false)
    }

    fn telemetry_response_with_moving(joint: HeadJoint, position: u16, moving: bool) -> Vec<u8> {
        telemetry_response_with_status(joint, position, moving, 0)
    }

    fn telemetry_response_with_status(
        joint: HeadJoint,
        position: u16,
        moving: bool,
        device_status_raw: u8,
    ) -> Vec<u8> {
        telemetry_response_with_raw(joint, position, moving, device_status_raw, [0; 13])
    }

    fn telemetry_response_with_raw(
        joint: HeadJoint,
        position: u16,
        moving: bool,
        device_status_raw: u8,
        remaining_raw: [u8; 13],
    ) -> Vec<u8> {
        let mut telemetry = [0_u8; 15];
        telemetry[..2].copy_from_slice(&position.to_le_bytes());
        telemetry[2..].copy_from_slice(&remaining_raw);
        telemetry[9] = device_status_raw;
        telemetry[10] = u8::from(moving);
        status(joint.servo_id(), &telemetry)
    }

    fn health_reads(positions: [u16; 4]) -> Vec<ReadAction> {
        HeadJoint::ALL
            .into_iter()
            .zip(positions)
            .map(|(joint, position)| ReadAction::Bytes(telemetry_response(joint, position)))
            .collect()
    }

    fn successful_reads() -> Vec<ReadAction> {
        let positions = [2_127_u16, 2_558, 2_925, 2_930];
        let mut reads = Vec::with_capacity(16);
        for (joint, position) in HeadJoint::ALL.into_iter().zip(positions) {
            reads.push(ReadAction::Bytes(position_response(joint, position - 2)));
            reads.push(ReadAction::Bytes(position_response(joint, position)));
        }
        for (joint, position) in HeadJoint::ALL.into_iter().zip(positions) {
            reads.push(ReadAction::Bytes(telemetry_response(joint, position)));
            reads.push(ReadAction::Bytes(telemetry_response(joint, position)));
        }
        reads
    }

    fn successful_reads_with_stationary_return() -> Vec<ReadAction> {
        let positions = [2_127_u16, 2_558, 2_925, 2_930];
        let mut reads = successful_reads();
        // Two fresh command-start sets, one set which causes the exact target
        // write, and two stopped completion sets.
        for _ in 0..5 {
            for (joint, position) in HeadJoint::ALL.into_iter().zip(positions) {
                reads.push(ReadAction::Bytes(telemetry_response(joint, position)));
            }
        }
        reads
    }

    fn valid_return_config(
        target: [u16; 4],
        maximum_travel_ticks: [u16; 4],
    ) -> ReturnToTargetConfig {
        return_config_with_start_bounds_and_freshness(
            target,
            target,
            target,
            maximum_travel_ticks,
            250,
        )
    }

    fn return_config_with_start_bounds_and_freshness(
        minimum_start_ticks: [u16; 4],
        maximum_start_ticks: [u16; 4],
        target: [u16; 4],
        maximum_travel_ticks: [u16; 4],
        arming_freshness_ms: u64,
    ) -> ReturnToTargetConfig {
        let probe = HeadProbeConfig::parse(HeadProbeConfigInput {
            device_path: "/dev/serial/by-id/usb-Kiko_head_test".to_owned(),
            response_timeout_ms: 100,
            request_timeout_ms: 100,
            noise_budget_bytes: 16,
        })
        .expect("test probe configuration");
        ReturnToTargetConfig::parse(
            &probe,
            ReturnToTargetConfigInput {
                write_timeout_ms: 1,
                arming_freshness_ms,
                write_attempts: 1,
                redundant_read_tolerance_ticks: 10,
                readback_tolerance_ticks: 20,
                final_target_tolerance_ticks: 0,
                path_corridor_tolerance_ticks: 0,
                direction_regression_tolerance_ticks: 0,
                goal_speed_ticks_per_second: 100,
                torque_limit_permille: [600, 400, 400, 400],
                minimum_start_ticks,
                maximum_start_ticks,
                target_ticks: target,
                maximum_travel_ticks,
            },
        )
        .expect("test return configuration")
    }

    fn spawn_return_fake(
        reads: Vec<ReadAction>,
        config: ReturnToTargetConfig,
    ) -> (
        HeadReturnActorHandle,
        StartupReceipt,
        HeadActorTask,
        Arc<Mutex<FakeShared>>,
    ) {
        let clock = TestClock::default();
        let (transport, shared) = FakeTransport::new(clock.clone(), reads);
        let (handle, startup, task) = spawn_head_return_actor(
            transport,
            clock,
            config,
            PhysicalTorqueEnableConsent::explicitly_granted(),
            PhysicalHeadMotionConsent::explicitly_granted(),
        )
        .expect("test runtime is active");
        (handle, startup, task, shared)
    }

    fn spawn_tension_preserving_return_fake(
        reads: Vec<ReadAction>,
        config: ReturnToTargetConfig,
    ) -> (
        TensionPreservingHeadReturnActorHandle,
        StartupReceipt,
        TensionPreservingHeadActorTask,
        Arc<Mutex<FakeShared>>,
    ) {
        let clock = TestClock::default();
        let (transport, shared) = FakeTransport::new(clock.clone(), reads);
        let (handle, startup, task) = spawn_tension_preserving_head_return_actor(
            transport,
            clock,
            config,
            PhysicalTorqueEnableConsent::explicitly_granted(),
            PhysicalHeadMotionConsent::explicitly_granted(),
            ProductionTensionPreservingTakeoverConsent::explicitly_granted_for_manifest_bound_owner(
            ),
        )
        .expect("test runtime is active");
        (handle, startup, task, shared)
    }

    fn spawn_fake(
        reads: Vec<ReadAction>,
        config: HeadRuntimeConfig,
    ) -> (
        HeadActorHandle,
        StartupReceipt,
        HeadActorTask,
        Arc<Mutex<FakeShared>>,
    ) {
        spawn_fake_with_bounds(reads, config, valid_pose_bounds())
    }

    fn spawn_fake_with_bounds(
        reads: Vec<ReadAction>,
        config: HeadRuntimeConfig,
        configured_pose_bounds: ConfiguredHeadPoseBounds,
    ) -> (
        HeadActorHandle,
        StartupReceipt,
        HeadActorTask,
        Arc<Mutex<FakeShared>>,
    ) {
        spawn_fake_with_bounds_and_write_failures(
            reads,
            config,
            configured_pose_bounds,
            BTreeMap::new(),
        )
    }

    fn spawn_fake_with_bounds_and_write_failures(
        reads: Vec<ReadAction>,
        config: HeadRuntimeConfig,
        configured_pose_bounds: ConfiguredHeadPoseBounds,
        write_failures: BTreeMap<usize, TransportFailure>,
    ) -> (
        HeadActorHandle,
        StartupReceipt,
        HeadActorTask,
        Arc<Mutex<FakeShared>>,
    ) {
        let clock = TestClock::default();
        let (transport, shared) = FakeTransport::new(clock.clone(), reads);
        shared.lock().expect("fake state").write_failures = write_failures;
        let (handle, startup, task) = spawn_head_actor(
            transport,
            clock,
            config,
            configured_pose_bounds,
            PhysicalTorqueEnableConsent::explicitly_granted(),
        )
        .expect("test runtime is active");
        (handle, startup, task, shared)
    }

    async fn run_startup_fault(
        reads: Vec<ReadAction>,
        config: HeadRuntimeConfig,
    ) -> (HeadRuntimeError, ActorExit, Arc<Mutex<FakeShared>>) {
        let (handle, receipt, task, shared) = spawn_fake(reads, config);
        let error = receipt
            .wait()
            .await
            .expect("startup channel")
            .expect_err("startup must fail");
        drop(handle);
        let exit = task.join().await.expect("actor task");
        (error, exit, shared)
    }

    fn health_observation_error(error: HeadHealthRequestError) -> Box<HeadHealthObservationError> {
        match error {
            HeadHealthRequestError::Check {
                source: HeadHealthCheckError::Observation(observation),
            } => observation,
            other => panic!("expected health observation error, got {other:#?}"),
        }
    }

    #[tokio::test]
    async fn startup_holds_only_observed_pose_and_shutdown_disables_every_joint() {
        let (handle, receipt, task, shared) = spawn_fake(successful_reads(), valid_config(1));
        let evidence = receipt
            .wait()
            .await
            .expect("startup channel")
            .expect("verified natural hold");
        assert_eq!(
            evidence.observed_pose().positions().map(PositionTicks::get),
            [2_127, 2_558, 2_925, 2_930]
        );
        assert_eq!(
            evidence
                .configured_pose()
                .observed_pose()
                .positions()
                .map(PositionTicks::get),
            [2_127, 2_558, 2_925, 2_930]
        );
        assert!(evidence.readbacks().iter().all(|readback| {
            readback.first_target_difference_ticks() == 0
                && readback.second_target_difference_ticks() == 0
                && readback.stable_difference_ticks() == 0
        }));
        assert!(
            evidence
                .pre_observation_torque_disable()
                .expect("commissioning startup disables before observation")
                .all_writes_completed()
        );

        let disable = handle.shutdown().await.expect("shutdown report");
        assert!(disable.all_writes_completed());
        let exit = task.join().await.expect("actor task");
        assert_eq!(exit.termination(), &ActorTermination::RequestedShutdown);
        assert_eq!(exit.torque_disable(), &disable);

        let shared = shared.lock().expect("fake state");
        assert_eq!(shared.writes.len(), 36);
        assert!(
            shared.writes[..4]
                .iter()
                .all(|write| write[5..=6] == [40, 0])
        );
        assert_eq!(
            shared.writes[..4]
                .iter()
                .map(|write| write[2])
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        for write in &shared.writes[4..12] {
            assert_eq!(&write[4..=6], &[2, 56, 2]);
        }
        assert_eq!(
            shared.writes[4..12]
                .iter()
                .map(|write| write[2])
                .collect::<Vec<_>>(),
            vec![1, 1, 2, 2, 3, 3, 4, 4]
        );
        assert!(shared.writes[12..16].iter().all(|write| write[5] == 48));
        for write in &shared.writes[16..20] {
            assert_eq!(write[5], 42);
            let id = usize::from(write[2] - 1);
            let target = u16::from_le_bytes([write[6], write[7]]);
            assert_eq!(target, [2_127, 2_558, 2_925, 2_930][id]);
            assert_ne!(u16::from_le_bytes([write[10], write[11]]), 0);
        }
        assert!(
            shared.writes[20..24]
                .iter()
                .all(|write| write[5..=6] == [40, 1])
        );
        assert!(
            shared.writes[24..32]
                .iter()
                .all(|write| write[4..=6] == [2, 56, 15])
        );
        assert!(
            shared.writes[32..]
                .iter()
                .all(|write| write[5..=6] == [40, 0])
        );
    }

    #[tokio::test]
    async fn production_takeover_observes_and_adopts_pose_without_a_torque_disable_gap() {
        let positions = [2_127, 2_558, 2_925, 2_930];
        let (handle, receipt, task, shared) = spawn_tension_preserving_return_fake(
            successful_reads(),
            valid_return_config(positions, [1; 4]),
        );
        let evidence = receipt
            .wait()
            .await
            .expect("startup channel")
            .expect("verified tension-preserving takeover");
        assert!(matches!(
            evidence.startup_torque(),
            HeadStartupTorqueEvidence::TensionPreservingTakeover
        ));
        assert_eq!(evidence.pre_observation_torque_disable(), None);
        assert_eq!(
            evidence.observed_pose().positions().map(PositionTicks::get),
            positions
        );

        let release = handle
            .release_ownership_preserving_hold()
            .await
            .expect("hold-preserving ownership release");
        let exit = task.join().await.expect("actor task");
        assert_eq!(exit.startup(), &Ok(evidence));
        assert_eq!(
            exit.termination(),
            &ActorTermination::RequestedHoldPreservingRelease
        );
        assert_eq!(exit.hold_preserving_release(), &release);

        let shared = shared.lock().expect("fake state");
        assert_eq!(shared.writes.len(), 28);
        assert!(
            shared.writes[..8]
                .iter()
                .all(|write| write[4..=6] == [2, 56, 2]),
            "the first protocol writes must be the redundant present-position reads"
        );
        assert!(
            shared
                .writes
                .iter()
                .all(|write| write.get(5..=6) != Some(&[40, 0]))
        );
        for write in &shared.writes[8..12] {
            assert_eq!(write[5], 48);
            let index = usize::from(write[2] - 1);
            assert_eq!(
                u16::from_le_bytes([write[6], write[7]]),
                [600, 400, 400, 400][index]
            );
        }
        for write in &shared.writes[12..16] {
            assert_eq!(write[5], 42);
            let index = usize::from(write[2] - 1);
            assert_eq!(u16::from_le_bytes([write[6], write[7]]), positions[index]);
            assert_eq!(
                u16::from_le_bytes([write[10], write[11]]),
                100,
                "the adopted goal carries the exact parsed bounded speed"
            );
        }
        assert!(
            shared.writes[16..20]
                .iter()
                .all(|write| write[5..=6] == [40, 1])
        );
        assert!(
            shared.writes[20..28]
                .iter()
                .all(|write| write[4..=6] == [2, 56, 15])
        );
    }

    #[tokio::test]
    async fn production_takeover_rejects_invalid_pose_evidence_before_motion_writes() {
        let positions = [2_127, 2_558, 2_925, 2_930];
        let mut invalid = successful_reads();
        invalid[1] = ReadAction::Bytes(position_response(HeadJoint::Bow, 2_200));
        let (handle, receipt, task, shared) =
            spawn_tension_preserving_return_fake(invalid, valid_return_config(positions, [1; 4]));
        let error = receipt
            .wait()
            .await
            .expect("startup channel")
            .expect_err("inconsistent present positions must fail");
        assert!(matches!(
            error,
            HeadRuntimeError::PositionAgreement {
                joint: HeadJoint::Bow,
                ..
            }
        ));
        drop(handle);
        let exit = task.join().await.expect("actor task");
        assert_eq!(exit.termination(), &ActorTermination::StartupFault);

        let shared = shared.lock().expect("fake state");
        assert_eq!(shared.writes.len(), 2);
        assert!(
            shared.writes[..2]
                .iter()
                .all(|write| write[4..=6] == [2, 56, 2])
        );
        assert!(
            shared.writes[..2]
                .iter()
                .all(|write| write.get(5..=6) != Some(&[40, 0])),
            "the production path must not pre-disable before rejected evidence"
        );
        assert!(shared.writes.iter().all(|write| {
            write.get(5) != Some(&48)
                && write.get(5) != Some(&42)
                && write.get(5..=6) != Some(&[40, 1])
        }));
    }

    #[tokio::test]
    async fn production_takeover_rejects_stale_or_out_of_window_pose_before_motion_writes() {
        let positions = [2_127, 2_558, 2_925, 2_930];
        let stale_config = return_config_with_start_bounds_and_freshness(
            positions, positions, positions, [1; 4], 1,
        );
        let (handle, receipt, task, stale_shared) =
            spawn_tension_preserving_return_fake(successful_reads(), stale_config);
        let stale = receipt
            .wait()
            .await
            .expect("startup channel")
            .expect_err("stale pose must fail");
        assert!(matches!(
            stale,
            HeadRuntimeError::ObservationStaleBeforeArming {
                check: ArmingFreshnessCheck::BeforeConfigurationWrites,
                ..
            }
        ));
        drop(handle);
        task.join().await.expect("actor task");

        let out_of_window_target = [2_000, 2_558, 2_925, 2_930];
        let (handle, receipt, task, bounds_shared) = spawn_tension_preserving_return_fake(
            successful_reads(),
            valid_return_config(out_of_window_target, [1; 4]),
        );
        let out_of_window = receipt
            .wait()
            .await
            .expect("startup channel")
            .expect_err("out-of-window pose must fail");
        assert!(matches!(
            out_of_window,
            HeadRuntimeError::ConfiguredPoseAdmission {
                source: HeadPoseBoundsAdmissionError::OutsideConfiguredWindow {
                    joint: HeadJoint::Bow,
                    ..
                }
            }
        ));
        drop(handle);
        task.join().await.expect("actor task");

        for shared in [stale_shared, bounds_shared] {
            let shared = shared.lock().expect("fake state");
            assert!(
                shared.writes.iter().all(|write| write.get(5) != Some(&48)
                    && write.get(5) != Some(&42)
                    && write.get(5..=6) != Some(&[40, 1])),
                "rejected boundary evidence must not produce torque-limit, goal, or enable writes"
            );
            assert!(
                shared.writes[..8]
                    .iter()
                    .all(|write| write[4..=6] == [2, 56, 2]),
                "boundary admission happens only after the complete redundant observation"
            );
            assert!(
                shared.writes[..8]
                    .iter()
                    .all(|write| write.get(5..=6) != Some(&[40, 0]))
            );
            assert_eq!(
                shared.writes.len(),
                8,
                "production startup failure adds no cleanup write"
            );
        }
    }

    #[tokio::test]
    async fn production_takeover_freshness_covers_both_reads_in_every_pair() {
        let positions = [2_127, 2_558, 2_925, 2_930];
        let mut reads = successful_reads();
        reads[1] = ReadAction::SetClockAndBytes {
            milliseconds: 100,
            bytes: position_response(HeadJoint::Bow, positions[0]),
        };
        let config = return_config_with_start_bounds_and_freshness(
            positions, positions, positions, [1; 4], 50,
        );
        let (handle, receipt, task, shared) = spawn_tension_preserving_return_fake(reads, config);
        let error = receipt
            .wait()
            .await
            .expect("startup channel")
            .expect_err("an old first read must make the redundant observation stale");
        assert!(matches!(
            error,
            HeadRuntimeError::ObservationStaleBeforeArming {
                check: ArmingFreshnessCheck::BeforeConfigurationWrites,
                age,
                maximum_age,
                ..
            } if age > maximum_age && maximum_age == Duration::from_millis(50)
        ));
        drop(handle);
        let exit = task.join().await.expect("actor task");
        assert_eq!(exit.termination(), &ActorTermination::StartupFault);

        let shared = shared.lock().expect("fake state");
        assert_eq!(shared.writes.len(), 8);
        assert!(shared.writes.iter().all(|write| write[4..=6] == [2, 56, 2]));
        assert!(
            shared
                .writes
                .iter()
                .all(|write| write.get(5..=6) != Some(&[40, 0]))
        );
    }

    #[tokio::test]
    async fn production_health_fault_and_ordinary_release_never_torque_disable() {
        let positions = [2_127, 2_558, 2_925, 2_930];
        let mut reads = successful_reads();
        reads.push(ReadAction::Failure(TransportFailure::timed_out(
            TransportOperation::Read,
            0,
        )));
        reads.extend(health_reads(positions));
        let (handle, receipt, task, shared) =
            spawn_tension_preserving_return_fake(reads, valid_return_config(positions, [1; 4]));
        receipt
            .wait()
            .await
            .expect("startup channel")
            .expect("verified tension-preserving takeover");

        assert!(matches!(
            handle.check_health().await,
            Err(HeadHealthRequestError::Check {
                source: HeadHealthCheckError::Observation(_),
            })
        ));
        let recovered = handle
            .check_health()
            .await
            .expect("a health fault does not tear down the production owner");
        assert_eq!(
            std::array::from_fn(|index| {
                recovered.joints()[index]
                    .response()
                    .value()
                    .position()
                    .get()
            }),
            positions
        );
        let release = handle
            .release_ownership_preserving_hold()
            .await
            .expect("hold-preserving release");
        let exit = task.join().await.expect("actor task");
        assert_eq!(exit.hold_preserving_release(), &release);
        assert!(
            shared
                .lock()
                .expect("fake state")
                .writes
                .iter()
                .all(|write| write.get(5..=6) != Some(&[40, 0]))
        );
    }

    #[tokio::test]
    async fn production_return_fault_exits_without_a_torque_disable_write() {
        let positions = [2_127, 2_558, 2_925, 2_930];
        let mut reads = successful_reads();
        reads.push(ReadAction::Eof);
        let (handle, receipt, task, shared) =
            spawn_tension_preserving_return_fake(reads, valid_return_config(positions, [1; 4]));
        receipt
            .wait()
            .await
            .expect("startup channel")
            .expect("verified tension-preserving takeover");
        assert!(matches!(
            handle.return_to_target().await,
            Ok(Err(HeadReturnError::TelemetryRead {
                joint: HeadJoint::Bow,
                ..
            }))
        ));
        drop(handle);
        let exit = task.join().await.expect("actor task");
        assert_eq!(exit.termination(), &ActorTermination::HeadReturnFault);
        assert!(matches!(
            exit.head_return(),
            Some(Err(HeadReturnError::TelemetryRead {
                joint: HeadJoint::Bow,
                ..
            }))
        ));
        assert!(
            shared
                .lock()
                .expect("fake state")
                .writes
                .iter()
                .all(|write| write.get(5..=6) != Some(&[40, 0]))
        );
    }

    #[tokio::test]
    async fn production_handle_loss_releases_ownership_without_torque_disable() {
        let positions = [2_127, 2_558, 2_925, 2_930];
        let (handle, receipt, task, shared) = spawn_tension_preserving_return_fake(
            successful_reads(),
            valid_return_config(positions, [1; 4]),
        );
        receipt
            .wait()
            .await
            .expect("startup channel")
            .expect("verified tension-preserving takeover");
        drop(handle);
        let exit = task.join().await.expect("actor task");
        assert_eq!(exit.termination(), &ActorTermination::HandleDropped);
        assert!(
            shared
                .lock()
                .expect("fake state")
                .writes
                .iter()
                .all(|write| write.get(5..=6) != Some(&[40, 0]))
        );
    }

    #[tokio::test]
    async fn health_check_retains_all_raw_values_identity_order_and_timing() {
        let positions = [2_127_u16, 2_558, 2_925, 2_930];
        let mut reads = successful_reads();
        for (index, (joint, position)) in HeadJoint::ALL.into_iter().zip(positions).enumerate() {
            let speed = 100_u16 + u16::try_from(index).expect("small test index");
            let load = 200_u16 + u16::try_from(index).expect("small test index");
            let registers = 300_u16 + u16::try_from(index).expect("small test index");
            let current = 400_u16 + u16::try_from(index).expect("small test index");
            let mut remaining_raw = [0_u8; 13];
            remaining_raw[0..2].copy_from_slice(&speed.to_le_bytes());
            remaining_raw[2..4].copy_from_slice(&load.to_le_bytes());
            remaining_raw[4] = 90 + u8::try_from(index).expect("small test index");
            remaining_raw[5] = 40 + u8::try_from(index).expect("small test index");
            remaining_raw[6] = 10 + u8::try_from(index).expect("small test index");
            remaining_raw[9..11].copy_from_slice(&registers.to_le_bytes());
            remaining_raw[11..13].copy_from_slice(&current.to_le_bytes());
            reads.push(ReadAction::Bytes(telemetry_response_with_raw(
                joint,
                position,
                false,
                0,
                remaining_raw,
            )));
        }
        let (handle, receipt, task, _) = spawn_fake(reads, valid_config(1));
        receipt
            .wait()
            .await
            .expect("startup channel")
            .expect("verified natural hold");

        let evidence = handle.check_health().await.expect("verified health");
        assert_eq!(
            evidence.hold_target().positions().map(PositionTicks::get),
            positions
        );
        assert!(matches!(
            evidence.hold_target(),
            HeadHoldTarget::StartupObserved(_)
        ));
        assert_eq!(evidence.tolerance().get(), 20);
        assert!(evidence.started_at() <= evidence.completed_at());
        for (index, (joint, sample)) in HeadJoint::ALL
            .into_iter()
            .zip(evidence.joints())
            .enumerate()
        {
            assert_eq!(sample.joint(), joint);
            assert_eq!(sample.telemetry().id(), joint.servo_id());
            assert_eq!(sample.target().get(), positions[index]);
            assert_eq!(sample.absolute_difference_ticks(), 0);
            assert!(!sample.telemetry().is_moving());
            assert_eq!(sample.telemetry().device_status_raw(), 0);
            assert_eq!(
                sample.telemetry().speed_raw(),
                100 + u16::try_from(index).expect("small test index")
            );
            assert_eq!(
                sample.telemetry().load_raw(),
                200 + u16::try_from(index).expect("small test index")
            );
            assert_eq!(
                sample.telemetry().voltage_raw(),
                90 + u8::try_from(index).expect("small test index")
            );
            assert_eq!(
                sample.telemetry().temperature_raw(),
                40 + u8::try_from(index).expect("small test index")
            );
            assert_eq!(
                sample.telemetry().async_write_flag_raw(),
                10 + u8::try_from(index).expect("small test index")
            );
            assert_eq!(
                sample.telemetry().registers_67_68_raw(),
                300 + u16::try_from(index).expect("small test index")
            );
            assert_eq!(
                sample.telemetry().current_raw(),
                400 + u16::try_from(index).expect("small test index")
            );
            assert!(
                sample.response().request_write().completed_at() <= sample.response().received_at()
            );
        }
        assert!(
            evidence
                .joints()
                .windows(2)
                .all(|pair| pair[0].response().received_at()
                    <= pair[1].response().request_write().completed_at())
        );
        assert!(
            evidence
                .joints()
                .last()
                .expect("four joints")
                .response()
                .received_at()
                <= evidence.completed_at()
        );

        handle.shutdown().await.expect("shutdown");
        task.join().await.expect("actor task");
    }

    #[tokio::test]
    async fn health_check_rejects_position_outside_natural_hold_tolerance() {
        let positions = [2_127_u16, 2_558, 2_925, 2_930];
        let mut reads = successful_reads();
        reads.push(ReadAction::Bytes(telemetry_response(
            HeadJoint::Bow,
            positions[0] + 21,
        )));
        reads.extend(health_reads(positions));
        let (handle, receipt, task, _) = spawn_fake(reads, valid_config(1));
        receipt
            .wait()
            .await
            .expect("startup channel")
            .expect("verified natural hold");

        let error = handle
            .check_health()
            .await
            .expect_err("21 ticks exceeds the admitted 20-tick tolerance");
        let HeadHealthRequestError::Check {
            source: HeadHealthCheckError::Observation(observation),
        } = error
        else {
            panic!("expected typed position-admission failure");
        };
        assert!(observation.accepted_prefix().iter().all(Option::is_none));
        assert!(matches!(
            observation.failure(),
            HeadHealthFailure::PositionMismatch {
                joint: HeadJoint::Bow,
                target,
                actual,
                absolute_difference_ticks: 21,
                tolerance,
                response,
            } if target.get() == positions[0]
                && actual.get() == positions[0] + 21
                && tolerance.get() == 20
                && response.value().position() == *actual
        ));

        handle
            .check_health()
            .await
            .expect("a rejected observation does not destroy the bus owner");
        handle.shutdown().await.expect("shutdown");
        task.join().await.expect("actor task");
    }

    #[tokio::test]
    async fn health_check_reports_moving_and_device_status_without_conflation() {
        let positions = [2_127_u16, 2_558, 2_925, 2_930];
        let mut moving_reads = successful_reads();
        moving_reads.push(ReadAction::Bytes(telemetry_response_with_moving(
            HeadJoint::Bow,
            positions[0],
            true,
        )));
        let (moving_handle, receipt, moving_task, _) = spawn_fake(moving_reads, valid_config(1));
        receipt
            .wait()
            .await
            .expect("startup channel")
            .expect("verified natural hold");
        let moving = moving_handle
            .check_health()
            .await
            .expect_err("moving sample is not a stopped hold");
        let moving = health_observation_error(moving);
        assert!(matches!(
            moving.failure(),
            HeadHealthFailure::Moving {
                joint: HeadJoint::Bow,
                position,
                ..
            } if position.get() == positions[0]
        ));
        moving_handle.shutdown().await.expect("shutdown");
        moving_task.join().await.expect("actor task");

        let mut status_reads = successful_reads();
        status_reads.push(ReadAction::Bytes(telemetry_response_with_status(
            HeadJoint::Bow,
            positions[0],
            false,
            7,
        )));
        let (status_handle, receipt, status_task, _) = spawn_fake(status_reads, valid_config(1));
        receipt
            .wait()
            .await
            .expect("startup channel")
            .expect("verified natural hold");
        let status = status_handle
            .check_health()
            .await
            .expect_err("nonzero device status is not healthy");
        let status = health_observation_error(status);
        assert!(matches!(
            status.failure(),
            HeadHealthFailure::DeviceStatus {
                joint: HeadJoint::Bow,
                raw: 7,
                response,
            } if response.value().position().get() == positions[0]
        ));
        status_handle.shutdown().await.expect("shutdown");
        status_task.join().await.expect("actor task");
    }

    #[tokio::test]
    async fn health_transport_failure_is_exact_and_the_actor_can_be_queried_again() {
        let positions = [2_127_u16, 2_558, 2_925, 2_930];
        let mut reads = successful_reads();
        reads.push(ReadAction::Failure(TransportFailure::timed_out(
            TransportOperation::Read,
            0,
        )));
        reads.extend(health_reads(positions));
        let (handle, receipt, task, _) = spawn_fake(reads, valid_config(1));
        receipt
            .wait()
            .await
            .expect("startup channel")
            .expect("verified natural hold");

        let failure = handle
            .check_health()
            .await
            .expect_err("scripted read timeout");
        let failure = health_observation_error(failure);
        assert!(matches!(
            failure.failure(),
            HeadHealthFailure::TelemetryRead {
                joint: HeadJoint::Bow,
                source: RequestError::ResponseFrame(FrameReadError::Transport {
                    source,
                    buffered_bytes: 0,
                    ..
                }),
            } if source.kind() == TransportFailureKind::TimedOut
                && source.operation() == TransportOperation::Read
        ));
        handle
            .check_health()
            .await
            .expect("transport failure is not swallowed and does not end ownership");

        handle.shutdown().await.expect("shutdown");
        task.join().await.expect("actor task");
    }

    #[tokio::test]
    async fn health_check_rejects_a_response_outside_canonical_identity_order() {
        let positions = [2_127_u16, 2_558, 2_925, 2_930];
        let mut reads = successful_reads();
        reads.push(ReadAction::Bytes(telemetry_response(
            HeadJoint::Curl,
            positions[1],
        )));
        let (handle, receipt, task, _) = spawn_fake(reads, valid_config(1));
        receipt
            .wait()
            .await
            .expect("startup channel")
            .expect("verified natural hold");

        let failure = handle
            .check_health()
            .await
            .expect_err("Curl response cannot satisfy the expected Bow request");
        let failure = health_observation_error(failure);
        assert!(matches!(
            failure.failure(),
            HeadHealthFailure::TelemetryRead {
                joint: HeadJoint::Bow,
                source: RequestError::Telemetry(
                    TelemetryParseError::Response(ResponseParseError::ServoIdMismatch {
                        expected,
                        actual,
                    })
                ),
            } if *expected == HeadJoint::Bow.servo_id()
                && *actual == HeadJoint::Curl.servo_id().get()
        ));

        handle.shutdown().await.expect("shutdown");
        task.join().await.expect("actor task");
    }

    #[tokio::test]
    async fn cancelled_health_receiver_does_not_disable_torque_or_end_the_actor() {
        let positions = [2_127_u16, 2_558, 2_925, 2_930];
        let mut reads = successful_reads();
        reads.extend(health_reads(positions));
        reads.extend(health_reads(positions));
        let (handle, receipt, task, shared) = spawn_fake(reads, valid_config(1));
        receipt
            .wait()
            .await
            .expect("startup channel")
            .expect("verified natural hold");

        let (response, cancelled_receiver) = oneshot::channel();
        handle
            .commands
            .send(HeadCommand::CheckHealth { response })
            .await
            .expect("queue health request");
        drop(cancelled_receiver);
        loop {
            if shared.lock().expect("fake state").writes.len() >= 36 {
                break;
            }
            tokio::task::yield_now().await;
        }

        handle
            .check_health()
            .await
            .expect("actor remains usable after the query receiver disappears");
        {
            let shared = shared.lock().expect("fake state");
            assert_eq!(shared.writes.len(), 40);
            assert!(
                shared.writes[32..40]
                    .iter()
                    .all(|write| write[4..=6] == [2, 56, 15]),
                "health requests are read-only and receiver cancellation adds no disable write"
            );
        }

        handle.shutdown().await.expect("shutdown");
        task.join().await.expect("actor task");
    }

    #[tokio::test]
    async fn repeated_health_checks_each_read_a_fresh_complete_set() {
        let positions = [2_127_u16, 2_558, 2_925, 2_930];
        let mut reads = successful_reads();
        reads.extend(health_reads(positions));
        reads.extend(health_reads(positions));
        let (handle, receipt, task, shared) = spawn_fake(reads, valid_config(1));
        receipt
            .wait()
            .await
            .expect("startup channel")
            .expect("verified natural hold");

        let first = handle.check_health().await.expect("first health set");
        let second = handle.check_health().await.expect("second health set");
        assert!(first.completed_at() <= second.started_at());
        assert!(
            first
                .joints()
                .iter()
                .chain(second.joints())
                .all(|joint| joint.telemetry().device_status_raw() == 0
                    && !joint.telemetry().is_moving())
        );
        assert_eq!(shared.lock().expect("fake state").writes.len(), 40);

        handle.shutdown().await.expect("shutdown");
        task.join().await.expect("actor task");
    }

    #[tokio::test]
    async fn health_clock_regression_is_a_typed_framing_failure() {
        let positions = [2_127_u16, 2_558, 2_925, 2_930];
        let mut reads = successful_reads();
        reads.push(ReadAction::SetClockAndBytes {
            milliseconds: 0,
            bytes: telemetry_response(HeadJoint::Bow, positions[0]),
        });
        let (handle, receipt, task, _) = spawn_fake(reads, valid_config(1));
        receipt
            .wait()
            .await
            .expect("startup channel")
            .expect("verified natural hold");

        let error = handle
            .check_health()
            .await
            .expect_err("scripted clock moved backwards after the response read");
        let observation = health_observation_error(error);
        assert!(
            matches!(
                observation.failure(),
                HeadHealthFailure::TelemetryRead {
                    joint: HeadJoint::Bow,
                    source: RequestError::ResponseFrame(FrameReadError::NonMonotonicClock {
                        previous,
                        actual,
                    }),
                } if actual < previous
            ),
            "unexpected health clock error: {observation:#?}"
        );

        handle.shutdown().await.expect("shutdown");
        task.join().await.expect("actor task");
    }

    #[tokio::test]
    async fn stationary_return_requires_two_samples_and_retains_exact_evidence() {
        let target = [2_127, 2_558, 2_925, 2_930];
        let mut reads = successful_reads_with_stationary_return();
        reads.extend(health_reads(target));
        let (handle, receipt, task, shared) =
            spawn_return_fake(reads, valid_return_config(target, [1; 4]));
        receipt
            .wait()
            .await
            .expect("startup channel")
            .expect("verified natural hold");

        let evidence = handle
            .return_to_target()
            .await
            .expect("actor command receipt")
            .expect("verified stationary return");
        assert_eq!(
            evidence.target().positions().map(PositionTicks::get),
            target
        );
        assert_eq!(evidence.waypoint_writes().len(), 1);
        assert_eq!(
            evidence.waypoint_writes()[0]
                .positions()
                .map(PositionTicks::get),
            target
        );
        assert!(
            evidence
                .first_stopped()
                .iter()
                .chain(evidence.second_stopped())
                .all(|sample| !sample.is_moving() && sample.device_status_raw() == 0)
        );
        let health = handle
            .check_health()
            .await
            .expect("post-return health uses the reviewed target");
        assert_eq!(
            health.hold_target(),
            HeadHoldTarget::ReviewedReturn(evidence.target())
        );
        assert_eq!(
            health.hold_target().positions().map(PositionTicks::get),
            target
        );
        assert!(
            health
                .joints()
                .iter()
                .all(|joint| joint.absolute_difference_ticks() == 0)
        );
        assert_eq!(
            handle
                .return_to_target()
                .await
                .expect("second request receives a typed actor response"),
            Err(HeadReturnError::CommandAlreadyAttempted)
        );

        let recorded = evidence.clone();
        let disable = handle.shutdown().await.expect("shutdown report");
        assert!(disable.all_writes_completed());
        let exit = task.join().await.expect("actor task");
        assert_eq!(exit.head_return(), Some(&Ok(recorded)));
        assert_eq!(exit.termination(), &ActorTermination::RequestedShutdown);

        let shared = shared.lock().expect("fake state");
        assert_eq!(shared.read_calls, 160);
        assert_eq!(shared.writes.len(), 64);
        assert_eq!(
            shared
                .writes
                .iter()
                .filter(|write| write.get(4) == Some(&3) && write.get(5) == Some(&42))
                .count(),
            8,
            "the return writes the exact four-joint target once before completion"
        );
    }

    #[tokio::test]
    async fn return_deadline_caps_io_and_never_fabricates_an_unstarted_response_read() {
        fn plan() -> HeadReturnPlan {
            let zero = PositionAgreementTicks::try_new(0).expect("zero tolerance");
            HeadReturnPlan::for_test(
                kiko_head_protocol::ExactHeadTargetPose::try_from_ticks([
                    2_127, 2_558, 2_925, 2_930,
                ])
                .expect("target"),
                [1; 4],
                zero,
                zero,
                zero,
                zero,
            )
        }

        let clock = TestClock::default();
        clock.set_milliseconds(19_999);
        let (transport, shared) = FakeTransport::new(
            clock.clone(),
            vec![ReadAction::Bytes(telemetry_response(HeadJoint::Bow, 2_127))],
        );
        let mut actor = HeadActor {
            transport,
            clock,
            config: valid_config(1),
            configured_pose_bounds: valid_pose_bounds(),
            startup_torque_policy: StartupTorquePolicy::CommissioningDisableFirst,
            return_plan: Some(plan()),
        };
        let (commands, mut receiver) = mpsc::channel(1);
        let mut control = ControlState::new();
        let failure = actor
            .read_return_joint(
                HeadJoint::Bow,
                &mut receiver,
                &mut control,
                ReturnOperationBudget::Initial {
                    plan: plan(),
                    started_at: MonotonicTime::from_duration_since_origin(Duration::ZERO),
                },
            )
            .await
            .expect_err("deadline expires immediately after the request write");
        drop(commands);

        assert!(matches!(
            failure,
            ReturnTelemetrySetFailure::Deadline {
                joint: HeadJoint::Bow,
                source: HeadMotionError::MotionTimeout { .. },
                io: None,
            }
        ));
        {
            let shared = shared.lock().expect("fake state");
            assert_eq!(shared.write_timeouts, [Duration::from_millis(1)]);
            assert_eq!(shared.read_calls, 0);
            assert!(shared.read_timeouts.is_empty());
        }

        let clock = TestClock::default();
        clock.set_milliseconds(19_998);
        let (transport, shared) = FakeTransport::new(
            clock.clone(),
            vec![ReadAction::Bytes(telemetry_response(HeadJoint::Bow, 2_127))],
        );
        let mut actor = HeadActor {
            transport,
            clock,
            config: valid_config(1),
            configured_pose_bounds: valid_pose_bounds(),
            startup_torque_policy: StartupTorquePolicy::CommissioningDisableFirst,
            return_plan: Some(plan()),
        };
        let (commands, mut receiver) = mpsc::channel(1);
        let mut control = ControlState::new();
        let failure = actor
            .read_return_joint(
                HeadJoint::Bow,
                &mut receiver,
                &mut control,
                ReturnOperationBudget::Initial {
                    plan: plan(),
                    started_at: MonotonicTime::from_duration_since_origin(Duration::ZERO),
                },
            )
            .await
            .expect_err("the response read is capped by the remaining motion deadline");
        drop(commands);

        assert!(matches!(
            failure,
            ReturnTelemetrySetFailure::Deadline {
                joint: HeadJoint::Bow,
                source: HeadMotionError::MotionTimeout { .. },
                io: Some(RequestError::ResponseFrame(FrameReadError::Transport {
                    source,
                    ..
                })),
            } if source.kind() == TransportFailureKind::TimedOut
        ));
        let shared = shared.lock().expect("fake state");
        assert_eq!(shared.write_timeouts, [Duration::from_millis(1)]);
        assert_eq!(shared.read_calls, 1);
        assert_eq!(shared.read_timeouts, [Duration::from_millis(1)]);
    }

    #[tokio::test]
    async fn waypoint_batch_failure_retains_the_exact_completed_prefix() {
        let clock = TestClock::default();
        let (transport, shared) = FakeTransport::new(clock.clone(), Vec::new());
        shared
            .lock()
            .expect("fake state")
            .write_failures
            .insert(1, TransportFailure::timed_out(TransportOperation::Write, 0));
        let mut actor = HeadActor {
            transport,
            clock,
            config: valid_config(1),
            configured_pose_bounds: valid_pose_bounds(),
            startup_torque_policy: StartupTorquePolicy::CommissioningDisableFirst,
            return_plan: None,
        };
        let positions =
            kiko_head_protocol::ExactHeadTargetPose::try_from_ticks([2_127, 2_558, 2_925, 2_930])
                .expect("target")
                .positions();
        let (commands, mut receiver) = mpsc::channel(1);
        let mut control = ControlState::new();

        let error = actor
            .write_return_waypoints(positions, &mut receiver, &mut control, None)
            .await
            .expect_err("second joint write fails");
        drop(commands);

        assert_eq!(error.positions(), positions);
        assert!(error.completed_writes()[0].is_some());
        assert!(error.completed_writes()[1..].iter().all(Option::is_none));
        assert!(matches!(
            error.failure(),
            HeadWaypointBatchFailure::Frame(FrameWriteError {
                joint: HeadJoint::Curl,
                attempts_used: 1,
                source,
                ..
            }) if source.kind() == TransportFailureKind::TimedOut
        ));
        let mut commanded = [PositionTicks::MIN; 4];
        apply_completed_waypoint_prefix(&mut commanded, &error);
        assert_eq!(commanded[0], positions[0]);
        assert_eq!(commanded[1..], [PositionTicks::MIN; 3]);
        assert_eq!(shared.lock().expect("fake state").writes.len(), 2);
    }

    #[tokio::test]
    async fn config_bound_return_rejects_fresh_start_drift_and_retains_existing_goal() {
        let target = [2_127, 2_558, 2_925, 2_930];
        let drifted = [2_129, 2_558, 2_925, 2_930];
        let mut reads = successful_reads();
        for _ in 0..2 {
            for (joint, position) in HeadJoint::ALL.into_iter().zip(drifted) {
                reads.push(ReadAction::Bytes(telemetry_response(joint, position)));
            }
        }
        let (handle, receipt, task, shared) =
            spawn_return_fake(reads, valid_return_config(target, [1; 4]));
        receipt
            .wait()
            .await
            .expect("startup channel")
            .expect("verified natural hold");

        let error = handle
            .return_to_target()
            .await
            .expect("actor command receipt")
            .expect_err("start-to-target travel must be admitted again by the actor");
        assert!(matches!(
            &error,
            HeadReturnError::KinematicFaultExistingGoalRetained {
                source: HeadMotionError::ReturnStartOutsideConfiguredBounds {
                    source: HeadPoseBoundsAdmissionError::OutsideConfiguredWindow {
                        joint: HeadJoint::Bow,
                        observed,
                        minimum,
                        maximum,
                    },
                },
                commanded_positions,
                waypoint_writes,
                ..
            } if observed.get() == 2_129
                && minimum.get() == 2_127
                && maximum.get() == 2_127
                && commanded_positions.map(PositionTicks::get) == target
                && waypoint_writes.is_empty()
        ));

        let disable = handle.shutdown().await.expect("shutdown report");
        assert!(disable.all_writes_completed());
        let exit = task.join().await.expect("actor task");
        assert_eq!(exit.termination(), &ActorTermination::RequestedShutdown);
        assert_eq!(exit.head_return(), Some(&Err(error)));
        assert!(exit.torque_disable().all_writes_completed());

        let shared = shared.lock().expect("fake state");
        assert_eq!(shared.read_calls, 96);
        assert_eq!(shared.writes.len(), 44);
        assert_eq!(
            shared
                .writes
                .iter()
                .filter(|write| write.get(4) == Some(&3) && write.get(5) == Some(&42))
                .count(),
            4,
            "only the startup observed-pose goals are permitted"
        );
    }

    #[tokio::test]
    async fn every_pre_observation_disable_failure_blocks_all_reads_and_arming_writes() {
        for failing_joint_index in 0..HeadJoint::ALL.len() {
            let mut write_failures = BTreeMap::new();
            let source = io::Error::from(io::ErrorKind::BrokenPipe);
            write_failures.insert(
                failing_joint_index,
                TransportFailure::from_io(TransportOperation::Write, &source, 0),
            );
            let (handle, receipt, task, shared) = spawn_fake_with_bounds_and_write_failures(
                successful_reads(),
                valid_config(1),
                valid_pose_bounds(),
                write_failures,
            );
            let error = receipt
                .wait()
                .await
                .expect("startup channel")
                .expect_err("every pre-observation disable failure must refuse startup");
            let HeadRuntimeError::PreObservationTorqueDisable { report } = &error else {
                panic!("unexpected startup error: {error:?}");
            };
            assert_eq!(report.outcomes().len(), HeadJoint::ALL.len());
            for (index, outcome) in report.outcomes().iter().enumerate() {
                assert_eq!(outcome.joint(), HeadJoint::ALL[index]);
                assert_eq!(outcome.result().is_err(), index == failing_joint_index);
            }
            assert!(report.first_failure().is_some());

            drop(handle);
            let exit = task.join().await.expect("actor task");
            assert_eq!(exit.startup(), &Err(error.clone()));
            assert_eq!(exit.termination(), &ActorTermination::StartupFault);
            assert!(exit.torque_disable().all_writes_completed());
            assert!(
                exit.torque_disable().started_at() >= report.completed_at(),
                "cleanup disable must follow the refused startup transaction"
            );

            let shared = shared.lock().expect("fake state");
            assert_eq!(shared.read_calls, 0);
            assert_eq!(shared.writes.len(), 8);
            assert!(
                shared.writes.iter().all(|write| write[5..=6] == [40, 0]),
                "only four pre-disable and four cleanup-disable frames are permitted"
            );
            assert_eq!(
                shared.writes[..4]
                    .iter()
                    .map(|write| write[2])
                    .collect::<Vec<_>>(),
                vec![1, 2, 3, 4]
            );
        }
    }

    #[tokio::test]
    async fn out_of_window_pose_prevents_every_goal_limit_and_enable_write() {
        let bounds = ConfiguredHeadPoseBounds::try_new(
            [1_900, 2_450, 2_800, 2_800],
            [2_000, 2_700, 3_050, 3_050],
        )
        .expect("bounded rejecting pose windows");
        let (handle, receipt, task, shared) =
            spawn_fake_with_bounds(successful_reads(), valid_config(1), bounds);
        let error = receipt
            .wait()
            .await
            .expect("startup channel")
            .expect_err("bow pose must be rejected before arming");
        assert!(matches!(
            error,
            HeadRuntimeError::ConfiguredPoseAdmission {
                source: HeadPoseBoundsAdmissionError::OutsideConfiguredWindow {
                    joint: HeadJoint::Bow,
                    observed,
                    minimum,
                    maximum,
                }
            } if observed.get() == 2_127 && minimum.get() == 1_900 && maximum.get() == 2_000
        ));
        drop(handle);
        let exit = task.join().await.expect("actor task");
        assert_eq!(exit.termination(), &ActorTermination::StartupFault);
        assert!(exit.torque_disable().all_writes_completed());

        let shared = shared.lock().expect("fake state");
        assert_eq!(shared.writes.len(), 16);
        assert!(
            shared.writes[..4]
                .iter()
                .all(|write| write[5..=6] == [40, 0])
        );
        assert!(
            shared.writes[4..12]
                .iter()
                .all(|write| write[4..=6] == [2, 56, 2])
        );
        assert!(
            shared.writes[12..]
                .iter()
                .all(|write| write[5..=6] == [40, 0])
        );
        assert!(shared.writes.iter().all(|write| {
            write.get(5) != Some(&48)
                && write.get(5) != Some(&42)
                && write.get(5..=6) != Some(&[40, 1])
        }));
    }

    #[tokio::test]
    async fn bounded_noise_is_resynchronised_and_reported() {
        let mut reads = successful_reads();
        let ReadAction::Bytes(first) = &mut reads[0] else {
            unreachable!("successful reads are byte chunks");
        };
        first.splice(0..0, [0x01, 0x7e]);
        let (handle, receipt, task, _) = spawn_fake(reads, valid_config(1));
        let evidence = receipt
            .wait()
            .await
            .expect("startup channel")
            .expect("noise within budget");
        assert_eq!(
            evidence.observations()[0].first().discarded_noise_bytes(),
            2
        );
        handle.shutdown().await.expect("shutdown");
        task.join().await.expect("actor task");
    }

    #[tokio::test]
    async fn noise_budget_and_declared_frame_bound_fail_closed() {
        let mut noisy = position_response(HeadJoint::Bow, 2_125);
        noisy.splice(0..0, [0x01; 17]);
        let (error, _, _) =
            run_startup_fault(vec![ReadAction::Bytes(noisy)], valid_config(1)).await;
        assert!(matches!(
            error,
            HeadRuntimeError::PositionObservation {
                source: RequestError::ResponseFrame(FrameReadError::NoiseBudgetExceeded {
                    budget_bytes: 16,
                    observed_noise_bytes: 17,
                }),
                ..
            }
        ));

        let oversized = vec![0xff, 0xff, HeadJoint::Bow.servo_id().get(), 0xff];
        let (error, _, _) =
            run_startup_fault(vec![ReadAction::Bytes(oversized)], valid_config(1)).await;
        assert!(matches!(
            error,
            HeadRuntimeError::PositionObservation {
                source: RequestError::ResponseFrame(FrameReadError::DeclaredLengthOutOfRange {
                    declared_bytes: 259,
                    maximum_bytes: 21,
                    ..
                }),
                ..
            }
        ));
    }

    #[tokio::test]
    async fn truncation_is_not_relabelled_as_a_protocol_error() {
        let complete = position_response(HeadJoint::Bow, 2_125);
        let reads = vec![ReadAction::Bytes(complete[..4].to_vec()), ReadAction::Eof];
        let (error, exit, _) = run_startup_fault(reads, valid_config(1)).await;
        assert!(matches!(
            error,
            HeadRuntimeError::PositionObservation {
                source: RequestError::ResponseFrame(FrameReadError::Truncated {
                    buffered_bytes: 4,
                    expected_bytes: Some(8),
                }),
                ..
            }
        ));
        assert!(exit.torque_disable().all_writes_completed());
    }

    #[tokio::test]
    async fn response_id_length_checksum_and_status_are_propagated_exactly() {
        let wrong_id = vec![ReadAction::Bytes(position_response(HeadJoint::Curl, 2_125))];
        let (error, _, _) = run_startup_fault(wrong_id, valid_config(1)).await;
        assert!(matches!(
            error,
            HeadRuntimeError::PositionObservation {
                source: RequestError::Telemetry(TelemetryParseError::Response(
                    ResponseParseError::ServoIdMismatch { actual: 2, .. }
                )),
                ..
            }
        ));

        let mut corrupt = position_response(HeadJoint::Bow, 2_125);
        let last = corrupt.len() - 1;
        corrupt[last] ^= 1;
        let (error, _, _) =
            run_startup_fault(vec![ReadAction::Bytes(corrupt)], valid_config(1)).await;
        assert!(matches!(
            error,
            HeadRuntimeError::PositionObservation {
                source: RequestError::Telemetry(TelemetryParseError::Response(
                    ResponseParseError::ChecksumMismatch { .. }
                )),
                ..
            }
        ));

        let wrong_parameter_count = status(HeadJoint::Bow.servo_id(), &[0x01]);
        let (error, _, _) = run_startup_fault(
            vec![ReadAction::Bytes(wrong_parameter_count)],
            valid_config(1),
        )
        .await;
        assert!(matches!(
            error,
            HeadRuntimeError::PositionObservation {
                source: RequestError::Telemetry(TelemetryParseError::Response(
                    ResponseParseError::ParameterCountMismatch {
                        expected: 2,
                        actual: 1,
                    }
                )),
                ..
            }
        ));

        let mut device_fault = position_response(HeadJoint::Bow, 2_125);
        device_fault[4] = 0x40;
        let last = device_fault.len() - 1;
        device_fault[last] = !device_fault[2..last]
            .iter()
            .fold(0_u8, |sum, byte| sum.wrapping_add(*byte));
        let (error, _, _) =
            run_startup_fault(vec![ReadAction::Bytes(device_fault)], valid_config(1)).await;
        assert!(matches!(
            error,
            HeadRuntimeError::PositionObservation {
                source: RequestError::Telemetry(TelemetryParseError::Response(
                    ResponseParseError::DeviceStatus(status)
                )),
                ..
            } if status.bits() == 0x40
        ));
    }

    #[tokio::test]
    async fn response_timeout_is_typed_and_shutdown_is_still_attempted() {
        let timeout = TransportFailure::timed_out(TransportOperation::Read, 0);
        let (error, exit, shared) =
            run_startup_fault(vec![ReadAction::Failure(timeout)], valid_config(1)).await;
        assert!(matches!(
            error,
            HeadRuntimeError::PositionObservation {
                source: RequestError::ResponseFrame(FrameReadError::Transport {
                    source,
                    buffered_bytes: 0,
                    ..
                }),
                ..
            } if source.kind() == TransportFailureKind::TimedOut
        ));
        assert!(exit.torque_disable().all_writes_completed());
        assert_eq!(shared.lock().expect("fake state").writes.len(), 9);
    }

    #[tokio::test]
    async fn partial_write_failure_is_never_retried() {
        let (handle, receipt, task, shared) = spawn_fake(successful_reads(), valid_config(8));
        shared
            .lock()
            .expect("fake state")
            .write_failures
            .insert(4, TransportFailure::timed_out(TransportOperation::Write, 3));
        let error = receipt
            .wait()
            .await
            .expect("startup channel")
            .expect_err("partial write must fail startup");
        assert!(matches!(
            error,
            HeadRuntimeError::PositionObservation {
                source: RequestError::RequestWrite(FrameWriteError {
                    attempts_used: 1,
                    source,
                    ..
                }),
                ..
            } if source.bytes_transferred() == 3
        ));
        drop(handle);
        let exit = task.join().await.expect("actor task");
        assert!(exit.torque_disable().all_writes_completed());
        assert_eq!(shared.lock().expect("fake state").writes.len(), 9);
    }

    #[tokio::test]
    async fn zero_progress_retry_is_explicit_in_success_evidence() {
        let (handle, receipt, task, shared) = spawn_fake(successful_reads(), valid_config(2));
        shared
            .lock()
            .expect("fake state")
            .write_failures
            .insert(4, TransportFailure::timed_out(TransportOperation::Write, 0));
        let evidence = receipt
            .wait()
            .await
            .expect("startup channel")
            .expect("bounded retry succeeds");
        let write = evidence.observations()[0].first().request_write();
        assert_eq!(write.attempts_used(), 2);
        assert_eq!(write.recovered_failures().count(), 1);
        handle.shutdown().await.expect("shutdown");
        task.join().await.expect("actor task");
    }

    #[tokio::test]
    async fn post_write_position_mismatch_prevents_success() {
        let mut reads = successful_reads();
        reads[8] = ReadAction::Bytes(telemetry_response(HeadJoint::Bow, 2_200));
        let (error, exit, _) = run_startup_fault(reads, valid_config(1)).await;
        assert!(matches!(
            error,
            HeadRuntimeError::ReadbackMismatch {
                joint: HeadJoint::Bow,
                sample: VerificationSample::First,
                target,
                actual,
                absolute_difference_ticks: 73,
                ..
            } if target.get() == 2_127 && actual.get() == 2_200
        ));
        assert!(exit.torque_disable().all_writes_completed());
    }

    #[tokio::test]
    async fn stale_observation_fails_before_any_arming_write() {
        let (error, exit, shared) =
            run_startup_fault(successful_reads(), config_with_freshness(1, 1)).await;
        assert!(matches!(
            error,
            HeadRuntimeError::ObservationStaleBeforeArming {
                joint: HeadJoint::Bow,
                check: ArmingFreshnessCheck::BeforeConfigurationWrites,
                age,
                maximum_age,
                ..
            } if age > maximum_age && maximum_age == Duration::from_millis(1)
        ));
        assert_eq!(exit.termination(), &ActorTermination::StartupFault);
        let shared = shared.lock().expect("fake state");
        assert_eq!(shared.writes.len(), 16);
        assert!(
            shared.writes[..4]
                .iter()
                .all(|write| write[5..=6] == [40, 0])
        );
        assert!(shared.writes[4..12].iter().all(|write| write[4] == 2));
        assert!(
            shared.writes[12..]
                .iter()
                .all(|write| write[5..=6] == [40, 0])
        );
    }

    #[tokio::test]
    async fn arming_requires_remaining_freshness_for_every_bounded_write_attempt() {
        let (error, exit, shared) =
            run_startup_fault(successful_reads(), config_with_freshness(8, 44)).await;
        assert!(matches!(
            error,
            HeadRuntimeError::ObservationArmingWriteBudgetInsufficient {
                joint: HeadJoint::Bow,
                remaining_freshness,
                required_write_budget,
                ..
            } if remaining_freshness < required_write_budget
                && required_write_budget == Duration::from_millis(8)
        ));
        assert_eq!(exit.termination(), &ActorTermination::StartupFault);
        let shared = shared.lock().expect("fake state");
        assert_eq!(shared.writes.len(), 24);
        assert!(
            shared.writes[20..]
                .iter()
                .all(|write| write[5..=6] == [40, 0])
        );
    }

    #[tokio::test]
    async fn moving_or_unstable_second_readback_never_claims_hold() {
        let mut moving = successful_reads();
        moving[9] = ReadAction::Bytes(telemetry_response_with_moving(HeadJoint::Bow, 2_127, true));
        let (error, _, _) = run_startup_fault(moving, valid_config(1)).await;
        assert!(matches!(
            error,
            HeadRuntimeError::ReadbackMoving {
                joint: HeadJoint::Bow,
                sample: VerificationSample::Second,
                position,
            } if position.get() == 2_127
        ));

        let mut unstable = successful_reads();
        unstable[8] = ReadAction::Bytes(telemetry_response(HeadJoint::Bow, 2_107));
        unstable[9] = ReadAction::Bytes(telemetry_response(HeadJoint::Bow, 2_147));
        let (error, _, _) = run_startup_fault(unstable, valid_config(1)).await;
        assert!(matches!(
            error,
            HeadRuntimeError::ReadbackUnstable {
                joint: HeadJoint::Bow,
                absolute_difference_ticks: 40,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn nonzero_startup_readback_device_status_never_claims_hold() {
        let mut reads = successful_reads();
        reads[8] = ReadAction::Bytes(telemetry_response_with_status(
            HeadJoint::Bow,
            2_127,
            false,
            7,
        ));
        let (error, exit, _) = run_startup_fault(reads, valid_config(1)).await;
        assert!(matches!(
            error,
            HeadRuntimeError::ReadbackDeviceStatus {
                joint: HeadJoint::Bow,
                sample: VerificationSample::First,
                position,
                raw: 7,
            } if position.get() == 2_127
        ));
        assert!(exit.torque_disable().all_writes_completed());
    }

    #[tokio::test]
    async fn dropping_handle_before_startup_cancels_then_disables_all() {
        let (handle, receipt, task, shared) = spawn_fake(Vec::new(), valid_config(1));
        drop(handle);
        let error = receipt
            .wait()
            .await
            .expect("startup channel")
            .expect_err("drop cancels startup");
        assert!(matches!(
            error,
            HeadRuntimeError::Cancelled {
                cause: CancellationCause::HandleDropped,
                stage: RuntimeStage::ObserveFirst,
                joint: HeadJoint::Bow,
            }
        ));
        let exit = task.join().await.expect("actor task");
        assert_eq!(exit.termination(), &ActorTermination::HandleDropped);
        assert!(exit.torque_disable().all_writes_completed());
        let shared = shared.lock().expect("fake state");
        assert_eq!(shared.writes.len(), 8);
        assert!(shared.writes.iter().all(|write| write[5..=6] == [40, 0]));
    }

    #[tokio::test]
    async fn disable_report_attempts_all_joints_after_individual_failures() {
        let (handle, receipt, task, shared) = spawn_fake(successful_reads(), valid_config(1));
        receipt
            .wait()
            .await
            .expect("startup channel")
            .expect("startup");
        {
            let mut state = shared.lock().expect("fake state");
            let error = io::Error::from(io::ErrorKind::BrokenPipe);
            state.write_failures.insert(
                32,
                TransportFailure::from_io(TransportOperation::Write, &error, 0),
            );
            state.write_failures.insert(
                33,
                TransportFailure::from_io(TransportOperation::Write, &error, 0),
            );
        }

        let report = handle.shutdown().await.expect("shutdown report");
        assert!(!report.all_writes_completed());
        assert!(report.outcomes()[0].result().is_err());
        assert!(report.outcomes()[1].result().is_err());
        assert!(report.outcomes()[2].result().is_ok());
        assert!(report.outcomes()[3].result().is_ok());
        let exit = task.join().await.expect("actor task");
        assert_eq!(exit.torque_disable().outcomes().len(), 4);
        assert_eq!(shared.lock().expect("fake state").writes.len(), 36);
    }

    #[tokio::test]
    async fn explicit_shutdown_during_startup_returns_disable_evidence() {
        let (handle, receipt, task, _) = spawn_fake(Vec::new(), valid_config(1));
        let (shutdown, startup) = tokio::join!(handle.shutdown(), receipt.wait());
        let report = shutdown.expect("shutdown report");
        assert!(report.all_writes_completed());
        assert!(matches!(
            startup.expect("startup channel"),
            Err(HeadRuntimeError::Cancelled {
                cause: CancellationCause::RequestedShutdown,
                ..
            })
        ));
        let exit = task.join().await.expect("actor task");
        assert_eq!(exit.termination(), &ActorTermination::RequestedShutdown);
        assert_eq!(exit.torque_disable(), &report);
    }

    #[tokio::test]
    async fn queued_shutdown_during_startup_fault_receives_exact_disable_report() {
        let entered = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let source = TransportFailure::timed_out(TransportOperation::Read, 0);
        let reads = vec![ReadAction::GatedFailure {
            entered: Arc::clone(&entered),
            release: Arc::clone(&release),
            source,
        }];
        let (handle, receipt, task, _) = spawn_fake(reads, valid_config(1));
        entered.notified().await;
        let (report_sender, report_receiver) = oneshot::channel();
        handle
            .commands
            .send(HeadCommand::Shutdown {
                response: report_sender,
            })
            .await
            .expect("queue shutdown while read is in flight");
        drop(handle);
        release.notify_one();

        let startup_error = receipt
            .wait()
            .await
            .expect("startup channel")
            .expect_err("in-flight read fault remains primary");
        assert!(matches!(
            startup_error,
            HeadRuntimeError::PositionObservation {
                source: RequestError::ResponseFrame(FrameReadError::Transport { .. }),
                ..
            }
        ));
        let report = report_receiver.await.expect("exact disable report");
        assert!(report.all_writes_completed());
        let exit = task.join().await.expect("actor task");
        assert_eq!(
            exit.termination(),
            &ActorTermination::StartupFaultWithShutdownRequested
        );
        assert_eq!(exit.torque_disable(), &report);
    }

    #[test]
    fn spawning_without_tokio_runtime_is_a_typed_error_before_serial_open() {
        let clock = TestClock::default();
        let (transport, _) = FakeTransport::new(clock.clone(), Vec::new());
        let spawn = spawn_head_actor(
            transport,
            clock,
            valid_config(1),
            valid_pose_bounds(),
            PhysicalTorqueEnableConsent::explicitly_granted(),
        );
        assert!(matches!(
            spawn,
            Err(HeadActorSpawnError::NoTokioRuntime { .. })
        ));

        let start = start_serial_head_actor(
            valid_config(1),
            valid_pose_bounds(),
            PhysicalTorqueEnableConsent::explicitly_granted(),
        );
        assert!(matches!(
            start,
            Err(HeadActorStartError::NoTokioRuntime { .. })
        ));
    }
}
