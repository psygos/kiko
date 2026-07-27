//! Transport-free head-gaze proposal admission and bounded command planning.
//!
//! This module deliberately owns no serial port, actuator protocol, camera,
//! calibration, or runtime task. Its output is a proposed pose in servo ticks,
//! not evidence that a head moved or that the tick values correspond to a
//! physical gaze direction. Production keeps its existing natural-hold
//! behavior until a separately qualified owner explicitly consumes this API.
//!
//! Motion is planned in discrete control-tick units. A velocity of one means
//! one servo position tick per serviced control tick; an acceleration of one
//! means that consecutive serviced-tick velocities may differ by one. A late
//! call still services at most one step and re-anchors the next deadline from
//! the observed call time. Elapsed wall time therefore cannot create catch-up
//! motion.
//!
//! A [`PreparedHeadGazeControlStep`] is only an uncommitted plan. Preparing a
//! step does not advance the controller. It must not be sent to hardware
//! outside the sole head actor. Physical integration requires an actor-local
//! ordered write, acknowledgement/readback, and explicit commit. An incomplete
//! or uncertain application must instead be aborted into an absorbing fault;
//! it must never be committed as though the complete target were applied.

use std::{
    fmt,
    num::{NonZeroI32, NonZeroU8, NonZeroU16, NonZeroU64},
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use kiko_head_protocol::{ExactHeadTargetPose, HeadJoint, PositionStepLimit, PositionTicks};

use crate::transport::MonotonicTime;

const JOINT_COUNT: usize = 4;
static NEXT_CONTROLLER_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

const fn joint_index(joint: HeadJoint) -> usize {
    match joint {
        HeadJoint::Bow => 0,
        HeadJoint::Curl => 1,
        HeadJoint::Yaw => 2,
        HeadJoint::Roll => 3,
    }
}

/// One signed discrete velocity in servo ticks per serviced control tick.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ServoVelocityTicksPerControlTick(i32);

impl ServoVelocityTicksPerControlTick {
    pub const ZERO: Self = Self(0);

    pub const fn get(self) -> i32 {
        self.0
    }

    const fn from_planned(ticks: i32) -> Self {
        Self(ticks)
    }
}

/// A positive absolute velocity bound in servo ticks per control tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ServoVelocityLimitTicksPerControlTick(NonZeroI32);

impl ServoVelocityLimitTicksPerControlTick {
    pub fn try_new(ticks: u32) -> Result<Self, PositiveServoTickLimitError> {
        positive_i32(ticks).map(Self)
    }

    pub const fn get(self) -> i32 {
        self.0.get()
    }
}

/// A positive acceleration bound in servo ticks per control tick squared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ServoAccelerationLimitTicksPerControlTickSquared(NonZeroI32);

impl ServoAccelerationLimitTicksPerControlTickSquared {
    pub fn try_new(ticks: u32) -> Result<Self, PositiveServoTickLimitError> {
        positive_i32(ticks).map(Self)
    }

    pub const fn get(self) -> i32 {
        self.0.get()
    }
}

fn positive_i32(value: u32) -> Result<NonZeroI32, PositiveServoTickLimitError> {
    if value == 0 {
        return Err(PositiveServoTickLimitError::Zero);
    }
    let converted =
        i32::try_from(value).map_err(|_| PositiveServoTickLimitError::TooLarge { value })?;
    Ok(NonZeroI32::new(converted).expect("positive value remains nonzero after conversion"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PositiveServoTickLimitError {
    Zero,
    TooLarge { value: u32 },
}

impl fmt::Display for PositiveServoTickLimitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid positive servo-tick limit: {self:?}")
    }
}

impl std::error::Error for PositiveServoTickLimitError {}

/// Named planned velocities for the four joints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadServoVelocity {
    velocities: [ServoVelocityTicksPerControlTick; JOINT_COUNT],
}

impl HeadServoVelocity {
    pub const ZERO: Self = Self {
        velocities: [ServoVelocityTicksPerControlTick::ZERO; JOINT_COUNT],
    };

    pub const fn velocity(self, joint: HeadJoint) -> ServoVelocityTicksPerControlTick {
        self.velocities[joint_index(joint)]
    }

    const fn from_velocities(velocities: [ServoVelocityTicksPerControlTick; JOINT_COUNT]) -> Self {
        Self { velocities }
    }
}

/// Position, velocity, acceleration, and per-step bounds for one joint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadJointMotionLimits {
    minimum: PositionTicks,
    maximum: PositionTicks,
    maximum_velocity: ServoVelocityLimitTicksPerControlTick,
    maximum_acceleration: ServoAccelerationLimitTicksPerControlTickSquared,
    maximum_position_step: PositionStepLimit,
}

impl HeadJointMotionLimits {
    pub fn try_new(
        minimum: PositionTicks,
        maximum: PositionTicks,
        maximum_velocity: ServoVelocityLimitTicksPerControlTick,
        maximum_acceleration: ServoAccelerationLimitTicksPerControlTickSquared,
        maximum_position_step: PositionStepLimit,
    ) -> Result<Self, HeadJointMotionLimitsError> {
        if minimum >= maximum {
            return Err(HeadJointMotionLimitsError::EmptyPositionRange { minimum, maximum });
        }
        Ok(Self {
            minimum,
            maximum,
            maximum_velocity,
            maximum_acceleration,
            maximum_position_step,
        })
    }

    pub const fn minimum(self) -> PositionTicks {
        self.minimum
    }

    pub const fn maximum(self) -> PositionTicks {
        self.maximum
    }

    pub const fn maximum_velocity(self) -> ServoVelocityLimitTicksPerControlTick {
        self.maximum_velocity
    }

    pub const fn maximum_acceleration(self) -> ServoAccelerationLimitTicksPerControlTickSquared {
        self.maximum_acceleration
    }

    pub const fn maximum_position_step(self) -> PositionStepLimit {
        self.maximum_position_step
    }

    fn contains(self, position: PositionTicks) -> bool {
        position >= self.minimum && position <= self.maximum
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadJointMotionLimitsError {
    EmptyPositionRange {
        minimum: PositionTicks,
        maximum: PositionTicks,
    },
}

impl fmt::Display for HeadJointMotionLimitsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid head-joint motion limits: {self:?}")
    }
}

impl std::error::Error for HeadJointMotionLimitsError {}

/// Named limits for all head joints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadMotionLimits {
    limits: [HeadJointMotionLimits; JOINT_COUNT],
}

impl HeadMotionLimits {
    pub const fn new(
        bow: HeadJointMotionLimits,
        curl: HeadJointMotionLimits,
        yaw: HeadJointMotionLimits,
        roll: HeadJointMotionLimits,
    ) -> Self {
        Self {
            limits: [bow, curl, yaw, roll],
        }
    }

    pub const fn joint(self, joint: HeadJoint) -> HeadJointMotionLimits {
        self.limits[joint_index(joint)]
    }
}

/// Positive scheduled period on the runtime's monotonic clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadControlPeriod(Duration);

impl HeadControlPeriod {
    pub fn try_new(duration: Duration) -> Result<Self, PositiveTimeValueError> {
        if duration.is_zero() {
            return Err(PositiveTimeValueError::Zero);
        }
        Ok(Self(duration))
    }

    pub const fn get(self) -> Duration {
        self.0
    }
}

/// Allowed lateness after a scheduled control tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadTickLateness(Duration);

impl HeadTickLateness {
    pub const fn new(duration: Duration) -> Self {
        Self(duration)
    }

    pub const fn get(self) -> Duration {
        self.0
    }
}

/// Exclusive proposal lifetime on the runtime's monotonic clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadProposalTtl(Duration);

impl HeadProposalTtl {
    pub fn try_new(duration: Duration) -> Result<Self, PositiveTimeValueError> {
        if duration.is_zero() {
            return Err(PositiveTimeValueError::Zero);
        }
        Ok(Self(duration))
    }

    pub const fn get(self) -> Duration {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PositiveTimeValueError {
    Zero,
}

impl fmt::Display for PositiveTimeValueError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("time value must be greater than zero")
    }
}

impl std::error::Error for PositiveTimeValueError {}

/// Number of distinct, ordered proposals required before tracking starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadAcquisitionProposalCount(NonZeroU8);

impl HeadAcquisitionProposalCount {
    pub fn try_new(proposals: u8) -> Result<Self, HeadAcquisitionProposalCountError> {
        NonZeroU8::new(proposals)
            .map(Self)
            .ok_or(HeadAcquisitionProposalCountError::Zero)
    }

    pub const fn get(self) -> u8 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadAcquisitionProposalCountError {
    Zero,
}

impl fmt::Display for HeadAcquisitionProposalCountError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("head acquisition proposal count must be greater than zero")
    }
}

impl std::error::Error for HeadAcquisitionProposalCountError {}

/// Timing policy for one controller instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadGazeTiming {
    control_period: HeadControlPeriod,
    maximum_tick_lateness: HeadTickLateness,
    proposal_ttl: HeadProposalTtl,
    acquisition_proposals: HeadAcquisitionProposalCount,
}

impl HeadGazeTiming {
    pub const fn new(
        control_period: HeadControlPeriod,
        maximum_tick_lateness: HeadTickLateness,
        proposal_ttl: HeadProposalTtl,
        acquisition_proposals: HeadAcquisitionProposalCount,
    ) -> Self {
        Self {
            control_period,
            maximum_tick_lateness,
            proposal_ttl,
            acquisition_proposals,
        }
    }

    pub const fn control_period(self) -> HeadControlPeriod {
        self.control_period
    }

    pub const fn maximum_tick_lateness(self) -> HeadTickLateness {
        self.maximum_tick_lateness
    }

    pub const fn proposal_ttl(self) -> HeadProposalTtl {
        self.proposal_ttl
    }

    pub const fn acquisition_proposals(self) -> HeadAcquisitionProposalCount {
        self.acquisition_proposals
    }
}

/// Maximum target error that enters settling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadDeadbandTicks(u16);

impl HeadDeadbandTicks {
    pub fn try_new(ticks: u16) -> Result<Self, HeadErrorBandValueError> {
        if ticks > PositionTicks::MAX.get() {
            return Err(HeadErrorBandValueError::TooLarge { ticks });
        }
        Ok(Self(ticks))
    }

    pub const fn get(self) -> u16 {
        self.0
    }
}

/// Target error that resumes tracking after settling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadResumeThresholdTicks(NonZeroU16);

impl HeadResumeThresholdTicks {
    pub fn try_new(ticks: u16) -> Result<Self, HeadErrorBandValueError> {
        if ticks == 0 {
            return Err(HeadErrorBandValueError::ZeroResumeThreshold);
        }
        if ticks > PositionTicks::MAX.get() {
            return Err(HeadErrorBandValueError::TooLarge { ticks });
        }
        Ok(Self(
            NonZeroU16::new(ticks).expect("positive threshold remains nonzero"),
        ))
    }

    pub const fn get(self) -> u16 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadErrorBandValueError {
    ZeroResumeThreshold,
    TooLarge { ticks: u16 },
}

impl fmt::Display for HeadErrorBandValueError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid head error-band value: {self:?}")
    }
}

impl std::error::Error for HeadErrorBandValueError {}

/// Deadband/resume hysteresis in absolute servo ticks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadGazeErrorBand {
    deadband: HeadDeadbandTicks,
    resume_threshold: HeadResumeThresholdTicks,
}

impl HeadGazeErrorBand {
    pub fn try_new(
        deadband: HeadDeadbandTicks,
        resume_threshold: HeadResumeThresholdTicks,
    ) -> Result<Self, HeadGazeErrorBandError> {
        if deadband.get() >= resume_threshold.get() {
            return Err(HeadGazeErrorBandError::NoHysteresis {
                deadband,
                resume_threshold,
            });
        }
        Ok(Self {
            deadband,
            resume_threshold,
        })
    }

    pub const fn deadband(self) -> HeadDeadbandTicks {
        self.deadband
    }

    pub const fn resume_threshold(self) -> HeadResumeThresholdTicks {
        self.resume_threshold
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadGazeErrorBandError {
    NoHysteresis {
        deadband: HeadDeadbandTicks,
        resume_threshold: HeadResumeThresholdTicks,
    },
}

impl fmt::Display for HeadGazeErrorBandError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid head-gaze hysteresis: {self:?}")
    }
}

impl std::error::Error for HeadGazeErrorBandError {}

/// Fully typed, transport-free control configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadGazeControlConfig {
    timing: HeadGazeTiming,
    natural_pose: ExactHeadTargetPose,
    motion_limits: HeadMotionLimits,
    error_band: HeadGazeErrorBand,
}

impl HeadGazeControlConfig {
    pub fn try_new(
        timing: HeadGazeTiming,
        natural_pose: ExactHeadTargetPose,
        motion_limits: HeadMotionLimits,
        error_band: HeadGazeErrorBand,
    ) -> Result<Self, HeadGazeControlConfigError> {
        for joint in HeadJoint::ALL {
            let position = natural_pose.position(joint);
            let limits = motion_limits.joint(joint);
            if !limits.contains(position) {
                return Err(HeadGazeControlConfigError::NaturalPoseOutOfRange {
                    joint,
                    position,
                    minimum: limits.minimum(),
                    maximum: limits.maximum(),
                });
            }
        }
        Ok(Self {
            timing,
            natural_pose,
            motion_limits,
            error_band,
        })
    }

    pub const fn timing(self) -> HeadGazeTiming {
        self.timing
    }

    pub const fn natural_pose(self) -> ExactHeadTargetPose {
        self.natural_pose
    }

    pub const fn motion_limits(self) -> HeadMotionLimits {
        self.motion_limits
    }

    pub const fn error_band(self) -> HeadGazeErrorBand {
        self.error_band
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadGazeControlConfigError {
    NaturalPoseOutOfRange {
        joint: HeadJoint,
        position: PositionTicks,
        minimum: PositionTicks,
        maximum: PositionTicks,
    },
}

impl fmt::Display for HeadGazeControlConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid head-gaze control configuration: {self:?}"
        )
    }
}

impl std::error::Error for HeadGazeControlConfigError {}

/// Strictly positive identity for one perception proposal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HeadGazeProposalId(NonZeroU64);

impl HeadGazeProposalId {
    pub fn try_new(raw: u64) -> Result<Self, HeadGazeProposalIdError> {
        NonZeroU64::new(raw)
            .map(Self)
            .ok_or(HeadGazeProposalIdError::Zero)
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadGazeProposalIdError {
    Zero,
}

impl fmt::Display for HeadGazeProposalIdError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("head-gaze proposal ID must be greater than zero")
    }
}

impl std::error::Error for HeadGazeProposalIdError {}

/// A pose proposal already expressed in the controller's servo-tick domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadGazeProposal {
    id: HeadGazeProposalId,
    observed_at: MonotonicTime,
    target: ExactHeadTargetPose,
}

impl HeadGazeProposal {
    pub const fn new(
        id: HeadGazeProposalId,
        observed_at: MonotonicTime,
        target: ExactHeadTargetPose,
    ) -> Self {
        Self {
            id,
            observed_at,
            target,
        }
    }

    pub const fn id(self) -> HeadGazeProposalId {
        self.id
    }

    pub const fn observed_at(self) -> MonotonicTime {
        self.observed_at
    }

    pub const fn target(self) -> ExactHeadTargetPose {
        self.target
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AdmittedHeadGazeProposal {
    proposal: HeadGazeProposal,
    valid_through_exclusive: MonotonicTime,
}

/// Result of writing the capacity-one pending proposal slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadGazeProposalAdmission {
    Stored,
    ReplacedPending { replaced: HeadGazeProposalId },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadGazeProposalAdmissionError {
    FaultHeld(HeadGazeFaultReason),
    FaultLatched(HeadGazeFaultReason),
    AdmissionClockRegression {
        latest_boundary_at: MonotonicTime,
        received_at: MonotonicTime,
    },
    ObservationInFuture {
        observed_at: MonotonicTime,
        received_at: MonotonicTime,
    },
    FreshnessDeadlineOverflow {
        observed_at: MonotonicTime,
        ttl: HeadProposalTtl,
    },
    Expired {
        id: HeadGazeProposalId,
        valid_through_exclusive: MonotonicTime,
        received_at: MonotonicTime,
    },
    SequenceNotIncreasing {
        previous: HeadGazeProposalId,
        actual: HeadGazeProposalId,
    },
    ObservationTimeNotIncreasing {
        previous: MonotonicTime,
        actual: MonotonicTime,
    },
    TargetOutOfRange {
        joint: HeadJoint,
        target: PositionTicks,
        minimum: PositionTicks,
        maximum: PositionTicks,
    },
}

impl fmt::Display for HeadGazeProposalAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "head-gaze proposal rejected: {self:?}")
    }
}

impl std::error::Error for HeadGazeProposalAdmissionError {}

/// Explicit lifecycle state for the transport-free controller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadGazeControlState {
    NaturalHold,
    Acquiring,
    Tracking,
    Settling,
    ReturningNatural,
    FaultHeld,
}

/// External reason that may irreversibly revoke this controller instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadGazeExternalFault {
    OwnerRevoked,
    EmergencyStop,
    TelemetryUnavailable,
    ActuatorWriteFailed,
    ActuatorReadbackFailed,
    ActuatorReadbackMismatch,
    ActuatorApplicationUncertain,
}

/// First fault latched by a controller. Fault state is absorbing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadGazeFaultReason {
    External(HeadGazeExternalFault),
    TickClockRegression {
        previous: MonotonicTime,
        actual: MonotonicTime,
    },
    TickDeadlineOverflow {
        scheduled_for: MonotonicTime,
        maximum_lateness: HeadTickLateness,
    },
    NextTickTimestampOverflow {
        serviced_at: MonotonicTime,
        control_period: HeadControlPeriod,
    },
    PlannerGenerationExhausted,
    MotionConstraintsInfeasible {
        joint: HeadJoint,
    },
    MotionArithmeticOverflow {
        joint: HeadJoint,
    },
}

/// Why one control step was serviced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadGazeTickDisposition {
    OnTime {
        deadline_inclusive: MonotonicTime,
    },
    LateTickRevoked {
        scheduled_for: MonotonicTime,
        deadline_inclusive: MonotonicTime,
    },
}

/// Process-unique identity of one controller instance.
///
/// The constructor allocates this identity internally. It exists so a step
/// prepared by one actor-local controller cannot be committed into another
/// controller that happens to be at the same planner generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HeadGazeControllerInstanceId(NonZeroU64);

impl HeadGazeControllerInstanceId {
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// Revision of all state on which a prepared step depends.
///
/// Successful proposal admission and successful step commit both advance this
/// value. A proposal admitted after preparation therefore makes the prepared
/// step stale instead of being silently overwritten by its later commit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HeadGazePlannerGeneration(u64);

impl HeadGazePlannerGeneration {
    pub const fn get(self) -> u64 {
        self.0
    }

    const fn next(self) -> Option<Self> {
        match self.0.checked_add(1) {
            Some(next) => Some(Self(next)),
            None => None,
        }
    }
}

/// Opaque binding between an uncommitted step and its source state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HeadGazePreparedStepToken {
    controller_instance: HeadGazeControllerInstanceId,
    based_on_generation: HeadGazePlannerGeneration,
}

impl HeadGazePreparedStepToken {
    pub const fn controller_instance(self) -> HeadGazeControllerInstanceId {
        self.controller_instance
    }

    pub const fn based_on_generation(self) -> HeadGazePlannerGeneration {
        self.based_on_generation
    }
}

/// One generation-bound, uncommitted pure-planner result.
///
/// This value is not a physical command and cannot be consumed directly by a
/// serial transport. Integration must keep it inside the sole head actor,
/// write joints in an actor-defined safe order, retain the exact completed
/// prefix after any partial write, obtain the required
/// acknowledgement/readback evidence, and only then pass the complete value
/// to [`HeadGazeController::commit_prepared`]. The value is intentionally not
/// `Clone` or `Copy`.
#[derive(Debug)]
pub struct PreparedHeadGazeControlStep {
    token: HeadGazePreparedStepToken,
    serviced_at: MonotonicTime,
    next_scheduled_for: MonotonicTime,
    state: HeadGazeControlState,
    planned_target: ExactHeadTargetPose,
    velocity: HeadServoVelocity,
    disposition: HeadGazeTickDisposition,
    candidate: HeadGazeController,
}

impl PreparedHeadGazeControlStep {
    pub const fn token(&self) -> HeadGazePreparedStepToken {
        self.token
    }

    pub const fn serviced_at(&self) -> MonotonicTime {
        self.serviced_at
    }

    pub const fn next_scheduled_for(&self) -> MonotonicTime {
        self.next_scheduled_for
    }

    pub const fn state(&self) -> HeadGazeControlState {
        self.state
    }

    /// Target produced by the pure planner, not a committed hardware pose.
    pub const fn planned_target(&self) -> ExactHeadTargetPose {
        self.planned_target
    }

    pub const fn velocity(&self) -> HeadServoVelocity {
        self.velocity
    }

    pub const fn disposition(&self) -> HeadGazeTickDisposition {
        self.disposition
    }
}

/// Evidence that one complete prepared planner step was committed.
///
/// This receipt is controller evidence only. It is not evidence that hardware
/// accepted or reached the target; the sole head actor must retain that
/// separate acknowledgement/readback evidence before requesting this commit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadGazeCommitReceipt {
    token: HeadGazePreparedStepToken,
    committed_generation: HeadGazePlannerGeneration,
    serviced_at: MonotonicTime,
    state: HeadGazeControlState,
    committed_target: ExactHeadTargetPose,
    velocity: HeadServoVelocity,
    next_scheduled_for: MonotonicTime,
    disposition: HeadGazeTickDisposition,
}

impl HeadGazeCommitReceipt {
    pub const fn token(self) -> HeadGazePreparedStepToken {
        self.token
    }

    pub const fn committed_generation(self) -> HeadGazePlannerGeneration {
        self.committed_generation
    }

    pub const fn serviced_at(self) -> MonotonicTime {
        self.serviced_at
    }

    pub const fn state(self) -> HeadGazeControlState {
        self.state
    }

    pub const fn committed_target(self) -> ExactHeadTargetPose {
        self.committed_target
    }

    pub const fn velocity(self) -> HeadServoVelocity {
        self.velocity
    }

    pub const fn next_scheduled_for(self) -> MonotonicTime {
        self.next_scheduled_for
    }

    pub const fn disposition(self) -> HeadGazeTickDisposition {
        self.disposition
    }
}

/// Why a prepared step could not be committed or fault-aborted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadGazePreparedStepError {
    WrongController {
        expected: HeadGazeControllerInstanceId,
        actual: HeadGazeControllerInstanceId,
    },
    StaleGeneration {
        current: HeadGazePlannerGeneration,
        prepared_from: HeadGazePlannerGeneration,
    },
    FutureGeneration {
        current: HeadGazePlannerGeneration,
        prepared_from: HeadGazePlannerGeneration,
    },
    FaultHeld(HeadGazeFaultReason),
}

impl fmt::Display for HeadGazePreparedStepError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "prepared head-gaze step rejected: {self:?}")
    }
}

impl std::error::Error for HeadGazePreparedStepError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadGazeTickError {
    BeforeScheduledTick {
        scheduled_for: MonotonicTime,
        observed_at: MonotonicTime,
    },
    FaultLatched(HeadGazeFaultReason),
    FaultHeld(HeadGazeFaultReason),
}

impl fmt::Display for HeadGazeTickError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "head-gaze tick failed: {self:?}")
    }
}

impl std::error::Error for HeadGazeTickError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PlannedHeadGazeControlStep {
    serviced_at: MonotonicTime,
    next_scheduled_for: MonotonicTime,
    state: HeadGazeControlState,
    planned_target: ExactHeadTargetPose,
    velocity: HeadServoVelocity,
    disposition: HeadGazeTickDisposition,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadGazeControllerInitError {
    ControllerInstanceIdentityExhausted,
    InitialCommittedTargetOutOfRange {
        joint: HeadJoint,
        position: PositionTicks,
        minimum: PositionTicks,
        maximum: PositionTicks,
    },
}

impl fmt::Display for HeadGazeControllerInitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid initial committed head target: {self:?}")
    }
}

impl std::error::Error for HeadGazeControllerInitError {}

/// Single-owner, capacity-one head-gaze controller.
///
/// This value is intentionally neither `Clone` nor transport-aware. Accepted
/// proposals replace only the pending slot; the controller consumes at most
/// one slot and prepares at most one bounded position step per serviced tick.
#[derive(Debug)]
pub struct HeadGazeController {
    instance_id: HeadGazeControllerInstanceId,
    generation: HeadGazePlannerGeneration,
    config: HeadGazeControlConfig,
    state: HeadGazeControlState,
    pose: ExactHeadTargetPose,
    velocity: HeadServoVelocity,
    pending: Option<AdmittedHeadGazeProposal>,
    active: Option<AdmittedHeadGazeProposal>,
    last_admitted_id: Option<HeadGazeProposalId>,
    last_admitted_observation: Option<MonotonicTime>,
    acquisition_count: u8,
    next_tick_due: MonotonicTime,
    latest_boundary_at: MonotonicTime,
    fault: Option<HeadGazeFaultReason>,
}

impl HeadGazeController {
    /// Construct from an exact target already committed by an external owner.
    ///
    /// The controller does not infer that hardware is at the configured
    /// natural pose. A non-natural committed target starts in
    /// [`HeadGazeControlState::ReturningNatural`] and can only approach natural
    /// through later bounded planner steps.
    pub fn try_new(
        config: HeadGazeControlConfig,
        initial_committed_target: ExactHeadTargetPose,
        started_at: MonotonicTime,
    ) -> Result<Self, HeadGazeControllerInitError> {
        let instance_id = allocate_controller_instance_id()
            .ok_or(HeadGazeControllerInitError::ControllerInstanceIdentityExhausted)?;
        for joint in HeadJoint::ALL {
            let position = initial_committed_target.position(joint);
            let limits = config.motion_limits().joint(joint);
            if !limits.contains(position) {
                return Err(
                    HeadGazeControllerInitError::InitialCommittedTargetOutOfRange {
                        joint,
                        position,
                        minimum: limits.minimum(),
                        maximum: limits.maximum(),
                    },
                );
            }
        }
        let state = if initial_committed_target == config.natural_pose() {
            HeadGazeControlState::NaturalHold
        } else {
            HeadGazeControlState::ReturningNatural
        };
        Ok(Self {
            instance_id,
            generation: HeadGazePlannerGeneration(0),
            config,
            state,
            pose: initial_committed_target,
            velocity: HeadServoVelocity::ZERO,
            pending: None,
            active: None,
            last_admitted_id: None,
            last_admitted_observation: None,
            acquisition_count: 0,
            next_tick_due: started_at,
            latest_boundary_at: started_at,
            fault: None,
        })
    }

    pub const fn config(&self) -> HeadGazeControlConfig {
        self.config
    }

    pub const fn instance_id(&self) -> HeadGazeControllerInstanceId {
        self.instance_id
    }

    pub const fn generation(&self) -> HeadGazePlannerGeneration {
        self.generation
    }

    pub const fn state(&self) -> HeadGazeControlState {
        self.state
    }

    /// Latest target committed to the planner, not observed hardware.
    pub const fn committed_target(&self) -> ExactHeadTargetPose {
        self.pose
    }

    pub const fn velocity(&self) -> HeadServoVelocity {
        self.velocity
    }

    pub const fn next_tick_due(&self) -> MonotonicTime {
        self.next_tick_due
    }

    pub const fn fault(&self) -> Option<HeadGazeFaultReason> {
        self.fault
    }

    pub fn admit_proposal(
        &mut self,
        proposal: HeadGazeProposal,
        received_at: MonotonicTime,
    ) -> Result<HeadGazeProposalAdmission, HeadGazeProposalAdmissionError> {
        if let Some(reason) = self.fault {
            return Err(HeadGazeProposalAdmissionError::FaultHeld(reason));
        }
        if received_at < self.latest_boundary_at {
            return Err(HeadGazeProposalAdmissionError::AdmissionClockRegression {
                latest_boundary_at: self.latest_boundary_at,
                received_at,
            });
        }
        if proposal.observed_at() > received_at {
            return Err(HeadGazeProposalAdmissionError::ObservationInFuture {
                observed_at: proposal.observed_at(),
                received_at,
            });
        }
        let ttl = self.config.timing().proposal_ttl();
        let valid_through_exclusive = checked_timestamp_add(proposal.observed_at(), ttl.get())
            .ok_or(HeadGazeProposalAdmissionError::FreshnessDeadlineOverflow {
                observed_at: proposal.observed_at(),
                ttl,
            })?;
        if received_at >= valid_through_exclusive {
            return Err(HeadGazeProposalAdmissionError::Expired {
                id: proposal.id(),
                valid_through_exclusive,
                received_at,
            });
        }
        if let Some(previous) = self.last_admitted_id
            && proposal.id() <= previous
        {
            return Err(HeadGazeProposalAdmissionError::SequenceNotIncreasing {
                previous,
                actual: proposal.id(),
            });
        }
        if let Some(previous) = self.last_admitted_observation
            && proposal.observed_at() <= previous
        {
            return Err(
                HeadGazeProposalAdmissionError::ObservationTimeNotIncreasing {
                    previous,
                    actual: proposal.observed_at(),
                },
            );
        }
        for joint in HeadJoint::ALL {
            let target = proposal.target().position(joint);
            let limits = self.config.motion_limits().joint(joint);
            if !limits.contains(target) {
                return Err(HeadGazeProposalAdmissionError::TargetOutOfRange {
                    joint,
                    target,
                    minimum: limits.minimum(),
                    maximum: limits.maximum(),
                });
            }
        }
        let Some(next_generation) = self.generation.next() else {
            let reason = HeadGazeFaultReason::PlannerGenerationExhausted;
            self.latch_fault(reason);
            return Err(HeadGazeProposalAdmissionError::FaultLatched(reason));
        };

        let admitted = AdmittedHeadGazeProposal {
            proposal,
            valid_through_exclusive,
        };
        let replaced = self.pending.replace(admitted);
        self.last_admitted_id = Some(proposal.id());
        self.last_admitted_observation = Some(proposal.observed_at());
        self.latest_boundary_at = received_at;
        self.generation = next_generation;
        Ok(match replaced {
            Some(previous) => HeadGazeProposalAdmission::ReplacedPending {
                replaced: previous.proposal.id(),
            },
            None => HeadGazeProposalAdmission::Stored,
        })
    }

    /// Irreversibly enter fault-held state without producing a motion command.
    pub fn latch_external_fault(&mut self, fault: HeadGazeExternalFault) {
        if self.fault.is_none() {
            self.latch_fault(HeadGazeFaultReason::External(fault));
        }
    }

    /// Prepare at most one bounded planner step without advancing committed
    /// planner state.
    ///
    /// A successful call snapshots all proposal, freshness, lifecycle, motion,
    /// and deadline changes into the returned value. Those changes become
    /// visible only through [`Self::commit_prepared`]. Planning failures that
    /// make continued use unsafe, including clock regression, still latch an
    /// absorbing fault immediately.
    pub fn prepare_tick(
        &mut self,
        now: MonotonicTime,
    ) -> Result<PreparedHeadGazeControlStep, HeadGazeTickError> {
        if let Some(reason) = self.fault {
            return Err(HeadGazeTickError::FaultHeld(reason));
        }
        let Some(next_generation) = self.generation.next() else {
            let reason = HeadGazeFaultReason::PlannerGenerationExhausted;
            self.latch_fault(reason);
            return Err(HeadGazeTickError::FaultLatched(reason));
        };

        let token = HeadGazePreparedStepToken {
            controller_instance: self.instance_id,
            based_on_generation: self.generation,
        };
        let mut candidate = self.planning_snapshot();
        let planned = match candidate.service_tick_in_place(now) {
            Ok(planned) => planned,
            Err(HeadGazeTickError::FaultLatched(reason)) => {
                self.latch_fault(reason);
                return Err(HeadGazeTickError::FaultLatched(reason));
            }
            Err(error) => return Err(error),
        };
        candidate.generation = next_generation;
        Ok(PreparedHeadGazeControlStep {
            token,
            serviced_at: planned.serviced_at,
            next_scheduled_for: planned.next_scheduled_for,
            state: planned.state,
            planned_target: planned.planned_target,
            velocity: planned.velocity,
            disposition: planned.disposition,
            candidate,
        })
    }

    /// Commit one fully applied and read-back-verified prepared step.
    ///
    /// The actor must not call this method after a partial, failed, or
    /// uncertain hardware application. Use [`Self::abort_prepared_with_fault`]
    /// in those cases.
    pub fn commit_prepared(
        &mut self,
        prepared: PreparedHeadGazeControlStep,
    ) -> Result<HeadGazeCommitReceipt, HeadGazePreparedStepError> {
        self.validate_prepared_token(prepared.token)?;
        let PreparedHeadGazeControlStep {
            token,
            serviced_at,
            next_scheduled_for,
            state,
            planned_target,
            velocity,
            disposition,
            candidate,
        } = prepared;
        debug_assert_eq!(candidate.instance_id, self.instance_id);
        debug_assert_eq!(Some(candidate.generation), self.generation.next());
        let committed_generation = candidate.generation;
        *self = candidate;
        Ok(HeadGazeCommitReceipt {
            token,
            committed_generation,
            serviced_at,
            state,
            committed_target: planned_target,
            velocity,
            next_scheduled_for,
            disposition,
        })
    }

    /// Reject a prepared step after failed or uncertain actor-local
    /// application and irreversibly latch the supplied external fault.
    ///
    /// The candidate planner pose is discarded. The controller retains its
    /// last committed target, clears motion velocity and proposal state, and
    /// will reject every later proposal, preparation, or commit.
    pub fn abort_prepared_with_fault(
        &mut self,
        prepared: PreparedHeadGazeControlStep,
        fault: HeadGazeExternalFault,
    ) -> Result<(), HeadGazePreparedStepError> {
        self.validate_prepared_token(prepared.token)?;
        self.latch_fault(HeadGazeFaultReason::External(fault));
        Ok(())
    }

    fn validate_prepared_token(
        &self,
        token: HeadGazePreparedStepToken,
    ) -> Result<(), HeadGazePreparedStepError> {
        if token.controller_instance != self.instance_id {
            return Err(HeadGazePreparedStepError::WrongController {
                expected: self.instance_id,
                actual: token.controller_instance,
            });
        }
        if let Some(reason) = self.fault {
            return Err(HeadGazePreparedStepError::FaultHeld(reason));
        }
        if token.based_on_generation < self.generation {
            return Err(HeadGazePreparedStepError::StaleGeneration {
                current: self.generation,
                prepared_from: token.based_on_generation,
            });
        }
        if token.based_on_generation > self.generation {
            return Err(HeadGazePreparedStepError::FutureGeneration {
                current: self.generation,
                prepared_from: token.based_on_generation,
            });
        }
        Ok(())
    }

    fn planning_snapshot(&self) -> Self {
        Self {
            instance_id: self.instance_id,
            generation: self.generation,
            config: self.config,
            state: self.state,
            pose: self.pose,
            velocity: self.velocity,
            pending: self.pending,
            active: self.active,
            last_admitted_id: self.last_admitted_id,
            last_admitted_observation: self.last_admitted_observation,
            acquisition_count: self.acquisition_count,
            next_tick_due: self.next_tick_due,
            latest_boundary_at: self.latest_boundary_at,
            fault: self.fault,
        }
    }

    fn service_tick_in_place(
        &mut self,
        now: MonotonicTime,
    ) -> Result<PlannedHeadGazeControlStep, HeadGazeTickError> {
        if now < self.latest_boundary_at {
            let reason = HeadGazeFaultReason::TickClockRegression {
                previous: self.latest_boundary_at,
                actual: now,
            };
            self.latch_fault(reason);
            return Err(HeadGazeTickError::FaultLatched(reason));
        }
        if now < self.next_tick_due {
            return Err(HeadGazeTickError::BeforeScheduledTick {
                scheduled_for: self.next_tick_due,
                observed_at: now,
            });
        }

        let scheduled_for = self.next_tick_due;
        let maximum_lateness = self.config.timing().maximum_tick_lateness();
        let Some(deadline_inclusive) = checked_timestamp_add(scheduled_for, maximum_lateness.get())
        else {
            let reason = HeadGazeFaultReason::TickDeadlineOverflow {
                scheduled_for,
                maximum_lateness,
            };
            self.latch_fault(reason);
            return Err(HeadGazeTickError::FaultLatched(reason));
        };
        let control_period = self.config.timing().control_period();
        let Some(next_scheduled_for) = checked_timestamp_add(now, control_period.get()) else {
            let reason = HeadGazeFaultReason::NextTickTimestampOverflow {
                serviced_at: now,
                control_period,
            };
            self.latch_fault(reason);
            return Err(HeadGazeTickError::FaultLatched(reason));
        };

        let late = now > deadline_inclusive;
        if late {
            self.revoke_for_return();
        } else {
            self.refresh_proposal_state(now);
        }

        let target = self.target_for_step();
        let (next_pose, next_velocity) = match plan_pose(
            self.pose,
            self.velocity,
            target,
            self.config.motion_limits(),
        ) {
            Ok(planned) => planned,
            Err(reason) => {
                self.latch_fault(reason);
                return Err(HeadGazeTickError::FaultLatched(reason));
            }
        };
        self.pose = next_pose;
        self.velocity = next_velocity;
        self.finish_state_after_step();
        self.latest_boundary_at = now;
        self.next_tick_due = next_scheduled_for;

        Ok(PlannedHeadGazeControlStep {
            serviced_at: now,
            next_scheduled_for,
            state: self.state,
            planned_target: self.pose,
            velocity: self.velocity,
            disposition: if late {
                HeadGazeTickDisposition::LateTickRevoked {
                    scheduled_for,
                    deadline_inclusive,
                }
            } else {
                HeadGazeTickDisposition::OnTime { deadline_inclusive }
            },
        })
    }

    fn refresh_proposal_state(&mut self, now: MonotonicTime) {
        if self.state == HeadGazeControlState::FaultHeld {
            return;
        }
        let active_expired = self
            .active
            .is_some_and(|active| now >= active.valid_through_exclusive);
        if active_expired {
            self.active = None;
            self.acquisition_count = 0;
            self.state = if self.is_natural_and_stopped() {
                HeadGazeControlState::NaturalHold
            } else {
                HeadGazeControlState::ReturningNatural
            };
        }
        if self
            .pending
            .is_some_and(|pending| now >= pending.valid_through_exclusive)
        {
            self.pending = None;
        }

        if let Some(pending) = self.pending.take() {
            self.active = Some(pending);
            match self.state {
                HeadGazeControlState::NaturalHold
                | HeadGazeControlState::ReturningNatural
                | HeadGazeControlState::Acquiring => {
                    let required = self.config.timing().acquisition_proposals().get();
                    if self.acquisition_count >= required - 1 {
                        self.state = HeadGazeControlState::Tracking;
                        self.acquisition_count = 0;
                    } else {
                        self.acquisition_count += 1;
                        self.state = HeadGazeControlState::Acquiring;
                    }
                }
                HeadGazeControlState::Tracking | HeadGazeControlState::Settling => {}
                HeadGazeControlState::FaultHeld => return,
            }
        }

        if let Some(active) = self.active {
            if self.state == HeadGazeControlState::Settling
                && maximum_pose_error(self.pose, active.proposal.target())
                    >= i64::from(self.config.error_band().resume_threshold().get())
            {
                self.state = HeadGazeControlState::Tracking;
            }
        } else {
            self.acquisition_count = 0;
            self.state = if self.is_natural_and_stopped() {
                HeadGazeControlState::NaturalHold
            } else {
                HeadGazeControlState::ReturningNatural
            };
        }
    }

    fn target_for_step(&self) -> ExactHeadTargetPose {
        match self.state {
            HeadGazeControlState::Tracking => self
                .active
                .expect("tracking state always has a fresh active proposal")
                .proposal
                .target(),
            HeadGazeControlState::Settling => self.pose,
            HeadGazeControlState::NaturalHold
            | HeadGazeControlState::Acquiring
            | HeadGazeControlState::ReturningNatural
            | HeadGazeControlState::FaultHeld => self.config.natural_pose(),
        }
    }

    fn finish_state_after_step(&mut self) {
        match self.state {
            HeadGazeControlState::Tracking => {
                let target = self
                    .active
                    .expect("tracking state always has an active proposal")
                    .proposal
                    .target();
                if maximum_pose_error(self.pose, target)
                    <= i64::from(self.config.error_band().deadband().get())
                {
                    self.state = HeadGazeControlState::Settling;
                }
            }
            HeadGazeControlState::ReturningNatural if self.is_natural_and_stopped() => {
                self.state = HeadGazeControlState::NaturalHold;
            }
            HeadGazeControlState::NaturalHold
            | HeadGazeControlState::Acquiring
            | HeadGazeControlState::Settling
            | HeadGazeControlState::ReturningNatural
            | HeadGazeControlState::FaultHeld => {}
        }
    }

    fn revoke_for_return(&mut self) {
        self.pending = None;
        self.active = None;
        self.acquisition_count = 0;
        self.state = if self.is_natural_and_stopped() {
            HeadGazeControlState::NaturalHold
        } else {
            HeadGazeControlState::ReturningNatural
        };
    }

    fn is_natural_and_stopped(&self) -> bool {
        self.pose == self.config.natural_pose() && self.velocity == HeadServoVelocity::ZERO
    }

    fn latch_fault(&mut self, reason: HeadGazeFaultReason) {
        if self.fault.is_none() {
            self.fault = Some(reason);
            if let Some(next_generation) = self.generation.next() {
                self.generation = next_generation;
            }
        }
        self.state = HeadGazeControlState::FaultHeld;
        self.pending = None;
        self.active = None;
        self.acquisition_count = 0;
        self.velocity = HeadServoVelocity::ZERO;
    }
}

fn allocate_controller_instance_id() -> Option<HeadGazeControllerInstanceId> {
    NEXT_CONTROLLER_INSTANCE_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .ok()
        .and_then(NonZeroU64::new)
        .map(HeadGazeControllerInstanceId)
}

fn checked_timestamp_add(timestamp: MonotonicTime, duration: Duration) -> Option<MonotonicTime> {
    timestamp
        .duration_since_origin()
        .checked_add(duration)
        .map(MonotonicTime::from_duration_since_origin)
}

fn maximum_pose_error(current: ExactHeadTargetPose, target: ExactHeadTargetPose) -> i64 {
    HeadJoint::ALL
        .into_iter()
        .map(|joint| {
            (i64::from(target.position(joint).get()) - i64::from(current.position(joint).get()))
                .abs()
        })
        .max()
        .unwrap_or(0)
}

fn plan_pose(
    current_pose: ExactHeadTargetPose,
    current_velocity: HeadServoVelocity,
    target: ExactHeadTargetPose,
    limits: HeadMotionLimits,
) -> Result<(ExactHeadTargetPose, HeadServoVelocity), HeadGazeFaultReason> {
    let mut positions = current_pose.positions();
    let mut velocities = current_velocity.velocities;
    for joint in HeadJoint::ALL {
        let (position, velocity) = plan_joint(
            joint,
            current_pose.position(joint),
            current_velocity.velocity(joint),
            target.position(joint),
            limits.joint(joint),
        )?;
        positions[joint_index(joint)] = position;
        velocities[joint_index(joint)] = velocity;
    }
    Ok((
        ExactHeadTargetPose::from_positions(positions[0], positions[1], positions[2], positions[3]),
        HeadServoVelocity::from_velocities(velocities),
    ))
}

fn plan_joint(
    joint: HeadJoint,
    current_position: PositionTicks,
    current_velocity: ServoVelocityTicksPerControlTick,
    target: PositionTicks,
    limits: HeadJointMotionLimits,
) -> Result<(PositionTicks, ServoVelocityTicksPerControlTick), HeadGazeFaultReason> {
    let position = i64::from(current_position.get());
    let velocity = i64::from(current_velocity.get());
    let target = i64::from(target.get());
    let minimum = i64::from(limits.minimum().get());
    let maximum = i64::from(limits.maximum().get());
    let velocity_limit = i64::from(limits.maximum_velocity().get());
    let acceleration_limit = i64::from(limits.maximum_acceleration().get());
    let step_limit = i64::from(limits.maximum_position_step().get());
    let effective_speed_limit = velocity_limit.min(step_limit);

    let error = target
        .checked_sub(position)
        .ok_or(HeadGazeFaultReason::MotionArithmeticOverflow { joint })?;
    let desired_velocity = preferred_velocity(error, acceleration_limit, effective_speed_limit);

    let acceleration_lower = velocity
        .checked_sub(acceleration_limit)
        .ok_or(HeadGazeFaultReason::MotionArithmeticOverflow { joint })?;
    let acceleration_upper = velocity
        .checked_add(acceleration_limit)
        .ok_or(HeadGazeFaultReason::MotionArithmeticOverflow { joint })?;
    let position_lower = minimum
        .checked_sub(position)
        .ok_or(HeadGazeFaultReason::MotionArithmeticOverflow { joint })?;
    let position_upper = maximum
        .checked_sub(position)
        .ok_or(HeadGazeFaultReason::MotionArithmeticOverflow { joint })?;
    let feasible_lower = (-velocity_limit)
        .max(-step_limit)
        .max(acceleration_lower)
        .max(position_lower);
    let feasible_upper = velocity_limit
        .min(step_limit)
        .min(acceleration_upper)
        .min(position_upper);
    if feasible_lower > feasible_upper {
        return Err(HeadGazeFaultReason::MotionConstraintsInfeasible { joint });
    }

    let planned_velocity = desired_velocity.clamp(feasible_lower, feasible_upper);
    let planned_position = position
        .checked_add(planned_velocity)
        .ok_or(HeadGazeFaultReason::MotionArithmeticOverflow { joint })?;
    let planned_position = u16::try_from(planned_position)
        .map_err(|_| HeadGazeFaultReason::MotionArithmeticOverflow { joint })?;
    let planned_velocity = i32::try_from(planned_velocity)
        .map_err(|_| HeadGazeFaultReason::MotionArithmeticOverflow { joint })?;
    Ok((
        PositionTicks::try_new(planned_position)
            .map_err(|_| HeadGazeFaultReason::MotionArithmeticOverflow { joint })?,
        ServoVelocityTicksPerControlTick::from_planned(planned_velocity),
    ))
}

fn preferred_velocity(error: i64, acceleration_limit: i64, speed_limit: i64) -> i64 {
    if error == 0 {
        return 0;
    }
    let distance = error.unsigned_abs();
    let acceleration = u64::try_from(acceleration_limit)
        .expect("validated positive i32 acceleration converts to u64");
    let speed = u64::try_from(speed_limit).expect("validated positive i32 speed converts to u64");
    let stoppable = maximum_stoppable_speed(distance, acceleration, speed);
    let magnitude = distance.min(speed).min(stoppable);
    let signed = i64::try_from(magnitude).expect("servo speed is bounded by i32::MAX");
    if error.is_negative() { -signed } else { signed }
}

/// Greatest discrete speed whose exact stepwise braking distance is no more
/// than `distance`.
fn maximum_stoppable_speed(distance: u64, acceleration: u64, upper: u64) -> u64 {
    let mut lower = 0_u64;
    let mut upper = upper;
    while lower < upper {
        let midpoint = lower + (upper - lower).div_ceil(2);
        if discrete_braking_distance(midpoint, acceleration) <= u128::from(distance) {
            lower = midpoint;
        } else {
            upper = midpoint - 1;
        }
    }
    lower
}

/// Exact distance traveled while braking a discrete velocity to zero.
///
/// For `speed = acceleration * q + r`, the serviced velocities are
/// `speed, speed - acceleration, ...` while positive, so their exact sum is
/// `acceleration*q*(q+1)/2 + r*(q+1)`. Every operand is widened before
/// multiplication.
fn discrete_braking_distance(speed: u64, acceleration: u64) -> u128 {
    debug_assert!(acceleration > 0);
    let acceleration = u128::from(acceleration);
    let speed = u128::from(speed);
    let quotient = speed / acceleration;
    let remainder = speed % acceleration;
    acceleration * quotient * (quotient + 1) / 2 + remainder * (quotient + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(nanoseconds: u64) -> MonotonicTime {
        MonotonicTime::from_duration_since_origin(Duration::from_nanos(nanoseconds))
    }

    fn before_duration_max(nanoseconds: u64) -> MonotonicTime {
        MonotonicTime::from_duration_since_origin(
            Duration::MAX
                .checked_sub(Duration::from_nanos(nanoseconds))
                .unwrap(),
        )
    }

    fn position(ticks: u16) -> PositionTicks {
        PositionTicks::try_new(ticks).unwrap()
    }

    fn pose(yaw_ticks: u16) -> ExactHeadTargetPose {
        ExactHeadTargetPose::from_positions(
            position(1_000),
            position(1_000),
            position(yaw_ticks),
            position(1_000),
        )
    }

    fn motion_limit() -> HeadJointMotionLimits {
        HeadJointMotionLimits::try_new(
            position(0),
            position(2_000),
            ServoVelocityLimitTicksPerControlTick::try_new(5).unwrap(),
            ServoAccelerationLimitTicksPerControlTickSquared::try_new(2).unwrap(),
            PositionStepLimit::try_new(4).unwrap(),
        )
        .unwrap()
    }

    fn config_with_ttl(ttl_ns: u64) -> HeadGazeControlConfig {
        let timing = HeadGazeTiming::new(
            HeadControlPeriod::try_new(Duration::from_nanos(10)).unwrap(),
            HeadTickLateness::new(Duration::from_nanos(5)),
            HeadProposalTtl::try_new(Duration::from_nanos(ttl_ns)).unwrap(),
            HeadAcquisitionProposalCount::try_new(2).unwrap(),
        );
        let limit = motion_limit();
        HeadGazeControlConfig::try_new(
            timing,
            pose(1_000),
            HeadMotionLimits::new(limit, limit, limit, limit),
            HeadGazeErrorBand::try_new(
                HeadDeadbandTicks::try_new(1).unwrap(),
                HeadResumeThresholdTicks::try_new(3).unwrap(),
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn proposal(id: u64, observed_at: u64, yaw_ticks: u16) -> HeadGazeProposal {
        HeadGazeProposal::new(
            HeadGazeProposalId::try_new(id).unwrap(),
            at(observed_at),
            pose(yaw_ticks),
        )
    }

    fn natural_controller(
        config: HeadGazeControlConfig,
        started_at: MonotonicTime,
    ) -> HeadGazeController {
        HeadGazeController::try_new(config, config.natural_pose(), started_at).unwrap()
    }

    trait CommitPreparedTickForTest {
        fn tick(&mut self, now: MonotonicTime) -> Result<HeadGazeCommitReceipt, HeadGazeTickError>;
    }

    impl CommitPreparedTickForTest for HeadGazeController {
        fn tick(&mut self, now: MonotonicTime) -> Result<HeadGazeCommitReceipt, HeadGazeTickError> {
            let prepared = self.prepare_tick(now)?;
            Ok(self
                .commit_prepared(prepared)
                .expect("freshly prepared step commits in test adapter"))
        }
    }

    #[test]
    fn preparing_a_step_does_not_advance_any_committed_planner_state() {
        let mut controller = natural_controller(config_with_ttl(1_000), at(0));
        controller
            .admit_proposal(proposal(1, 0, 1_100), at(0))
            .unwrap();
        controller.tick(at(0)).unwrap();
        controller
            .admit_proposal(proposal(2, 10, 1_100), at(10))
            .unwrap();

        let generation = controller.generation();
        let state = controller.state();
        let target = controller.committed_target();
        let velocity = controller.velocity();
        let next_tick_due = controller.next_tick_due();
        let prepared = controller.prepare_tick(at(10)).unwrap();

        assert_eq!(prepared.token().based_on_generation(), generation);
        assert_eq!(prepared.state(), HeadGazeControlState::Tracking);
        assert_ne!(prepared.planned_target(), target);
        assert_eq!(controller.generation(), generation);
        assert_eq!(controller.state(), state);
        assert_eq!(controller.committed_target(), target);
        assert_eq!(controller.velocity(), velocity);
        assert_eq!(controller.next_tick_due(), next_tick_due);

        let prepared_again = controller.prepare_tick(at(10)).unwrap();
        assert_eq!(prepared_again.token(), prepared.token());
        assert_eq!(prepared_again.planned_target(), prepared.planned_target());
        assert_eq!(controller.generation(), generation);
    }

    #[test]
    fn one_generation_commits_exactly_once() {
        let mut controller = natural_controller(config_with_ttl(100), at(0));
        let first_copy = controller.prepare_tick(at(0)).unwrap();
        let second_copy = controller.prepare_tick(at(0)).unwrap();
        let prepared_generation = first_copy.token().based_on_generation();

        let receipt = controller.commit_prepared(first_copy).unwrap();
        assert_eq!(
            receipt.committed_generation(),
            prepared_generation.next().unwrap()
        );
        assert_eq!(controller.generation(), receipt.committed_generation());
        assert_eq!(
            controller.commit_prepared(second_copy),
            Err(HeadGazePreparedStepError::StaleGeneration {
                current: receipt.committed_generation(),
                prepared_from: prepared_generation,
            })
        );
        assert_eq!(controller.generation(), receipt.committed_generation());
    }

    #[test]
    fn admission_after_prepare_invalidates_the_step_without_losing_the_proposal() {
        let mut controller = natural_controller(config_with_ttl(100), at(0));
        let prepared = controller.prepare_tick(at(0)).unwrap();
        let prepared_from = prepared.token().based_on_generation();
        controller
            .admit_proposal(proposal(1, 1, 1_100), at(1))
            .unwrap();
        let current = controller.generation();

        assert_eq!(
            controller.commit_prepared(prepared),
            Err(HeadGazePreparedStepError::StaleGeneration {
                current,
                prepared_from,
            })
        );
        assert_eq!(
            controller.tick(at(1)).unwrap().state(),
            HeadGazeControlState::Acquiring
        );
    }

    #[test]
    fn controller_rejects_a_step_prepared_by_another_instance() {
        let config = config_with_ttl(100);
        let mut source = natural_controller(config, at(0));
        let prepared = source.prepare_tick(at(0)).unwrap();
        let actual = prepared.token().controller_instance();
        let mut destination = natural_controller(config, at(0));
        let expected = destination.instance_id();

        assert_ne!(expected, actual);
        assert_eq!(
            destination.commit_prepared(prepared),
            Err(HeadGazePreparedStepError::WrongController { expected, actual })
        );
        assert_eq!(destination.generation(), HeadGazePlannerGeneration(0));
    }

    #[test]
    fn aborted_application_keeps_last_commit_and_latches_an_absorbing_fault() {
        let mut controller = natural_controller(config_with_ttl(1_000), at(0));
        controller
            .admit_proposal(proposal(1, 0, 1_100), at(0))
            .unwrap();
        controller.tick(at(0)).unwrap();
        controller
            .admit_proposal(proposal(2, 10, 1_100), at(10))
            .unwrap();
        let last_commit = controller.committed_target();
        let prepared = controller.prepare_tick(at(10)).unwrap();
        assert_ne!(prepared.planned_target(), last_commit);

        controller
            .abort_prepared_with_fault(prepared, HeadGazeExternalFault::ActuatorReadbackMismatch)
            .unwrap();
        let reason = HeadGazeFaultReason::External(HeadGazeExternalFault::ActuatorReadbackMismatch);
        assert_eq!(controller.fault(), Some(reason));
        assert_eq!(controller.state(), HeadGazeControlState::FaultHeld);
        assert_eq!(controller.committed_target(), last_commit);
        assert_eq!(controller.velocity(), HeadServoVelocity::ZERO);
        assert!(matches!(
            controller.prepare_tick(at(10)),
            Err(HeadGazeTickError::FaultHeld(actual)) if actual == reason
        ));
        assert_eq!(
            controller.admit_proposal(proposal(3, 11, 1_100), at(11)),
            Err(HeadGazeProposalAdmissionError::FaultHeld(reason))
        );
    }

    #[test]
    fn pending_slot_replaces_latest_and_rejects_stale_or_out_of_order_inputs() {
        let mut controller = natural_controller(config_with_ttl(100), at(0));
        assert_eq!(
            controller.admit_proposal(proposal(1, 0, 1_100), at(0)),
            Ok(HeadGazeProposalAdmission::Stored)
        );
        assert_eq!(
            controller.admit_proposal(proposal(2, 1, 1_200), at(1)),
            Ok(HeadGazeProposalAdmission::ReplacedPending {
                replaced: HeadGazeProposalId::try_new(1).unwrap()
            })
        );
        assert_eq!(
            controller.admit_proposal(proposal(1, 2, 1_300), at(2)),
            Err(HeadGazeProposalAdmissionError::SequenceNotIncreasing {
                previous: HeadGazeProposalId::try_new(2).unwrap(),
                actual: HeadGazeProposalId::try_new(1).unwrap(),
            })
        );
        assert_eq!(
            controller.admit_proposal(proposal(3, 0, 1_300), at(2)),
            Err(
                HeadGazeProposalAdmissionError::ObservationTimeNotIncreasing {
                    previous: at(1),
                    actual: at(0),
                }
            )
        );
        assert_eq!(
            controller.admit_proposal(proposal(4, 1, 1_300), at(2)),
            Err(
                HeadGazeProposalAdmissionError::ObservationTimeNotIncreasing {
                    previous: at(1),
                    actual: at(1),
                }
            )
        );
        assert_eq!(
            controller.tick(at(2)).unwrap().state(),
            HeadGazeControlState::Acquiring
        );
        assert_eq!(controller.committed_target(), pose(1_000));
    }

    #[test]
    fn proposal_ttl_is_exclusive_at_the_exact_boundary() {
        let mut before = natural_controller(config_with_ttl(50), at(0));
        assert!(before.admit_proposal(proposal(1, 0, 1_100), at(49)).is_ok());

        let mut boundary = natural_controller(config_with_ttl(50), at(0));
        assert_eq!(
            boundary.admit_proposal(proposal(1, 0, 1_100), at(50)),
            Err(HeadGazeProposalAdmissionError::Expired {
                id: HeadGazeProposalId::try_new(1).unwrap(),
                valid_through_exclusive: at(50),
                received_at: at(50),
            })
        );
    }

    #[test]
    fn expired_active_target_resets_acquisition_before_pending_is_consumed() {
        let mut controller = natural_controller(config_with_ttl(10), at(0));
        controller
            .admit_proposal(proposal(1, 0, 1_100), at(0))
            .unwrap();
        assert_eq!(
            controller.tick(at(0)).unwrap().state(),
            HeadGazeControlState::Acquiring
        );

        controller
            .admit_proposal(proposal(2, 10, 1_100), at(10))
            .unwrap();
        assert_eq!(
            controller.tick(at(10)).unwrap().state(),
            HeadGazeControlState::Acquiring
        );
    }

    #[test]
    fn maximum_typed_acquisition_count_reaches_tracking_without_counter_overflow() {
        let mut config = config_with_ttl(10_000);
        config.timing.acquisition_proposals =
            HeadAcquisitionProposalCount::try_new(u8::MAX).unwrap();
        let mut controller = natural_controller(config, at(0));

        for id in 1_u64..=u64::from(u8::MAX) {
            let now = (id - 1) * 10;
            controller
                .admit_proposal(proposal(id, now, 1_100), at(now))
                .unwrap();
            let state = controller.tick(at(now)).unwrap().state();
            if id < u64::from(u8::MAX) {
                assert_eq!(state, HeadGazeControlState::Acquiring);
            } else {
                assert_eq!(state, HeadGazeControlState::Tracking);
            }
        }
    }

    #[test]
    fn successful_receipt_advances_the_tick_causality_boundary() {
        let mut controller = natural_controller(config_with_ttl(200), at(0));
        controller
            .admit_proposal(proposal(1, 90, 1_100), at(100))
            .unwrap();
        assert_eq!(
            controller.tick(at(99)),
            Err(HeadGazeTickError::FaultLatched(
                HeadGazeFaultReason::TickClockRegression {
                    previous: at(100),
                    actual: at(99),
                }
            ))
        );
        assert_eq!(controller.state(), HeadGazeControlState::FaultHeld);

        let mut after_tick = natural_controller(config_with_ttl(200), at(0));
        after_tick.tick(at(10)).unwrap();
        assert_eq!(
            after_tick.admit_proposal(proposal(1, 9, 1_100), at(9)),
            Err(HeadGazeProposalAdmissionError::AdmissionClockRegression {
                latest_boundary_at: at(10),
                received_at: at(9),
            })
        );
    }

    #[test]
    fn controller_exposes_acquire_track_settle_resume_return_and_natural_states() {
        let mut controller = natural_controller(config_with_ttl(100), at(0));
        controller
            .admit_proposal(proposal(1, 0, 1_010), at(0))
            .unwrap();
        assert_eq!(
            controller.tick(at(0)).unwrap().state(),
            HeadGazeControlState::Acquiring
        );
        controller
            .admit_proposal(proposal(2, 10, 1_010), at(10))
            .unwrap();
        assert_eq!(
            controller.tick(at(10)).unwrap().state(),
            HeadGazeControlState::Tracking
        );

        let mut now = 20_u64;
        while controller.state() != HeadGazeControlState::Settling {
            controller.tick(at(now)).unwrap();
            now += 10;
            assert!(now < 100);
        }

        controller
            .admit_proposal(proposal(3, now, 900), at(now))
            .unwrap();
        assert_eq!(
            controller.tick(at(now)).unwrap().state(),
            HeadGazeControlState::Tracking
        );
        now += 100;
        let expired = controller.tick(at(now)).unwrap();
        assert_eq!(expired.state(), HeadGazeControlState::ReturningNatural);

        for _ in 0..100 {
            now += 10;
            controller.tick(at(now)).unwrap();
            if controller.state() == HeadGazeControlState::NaturalHold {
                break;
            }
        }
        assert_eq!(controller.state(), HeadGazeControlState::NaturalHold);
        assert_eq!(controller.committed_target(), pose(1_000));
        assert_eq!(controller.velocity(), HeadServoVelocity::ZERO);
    }

    #[test]
    fn late_tick_revokes_and_services_only_one_step_without_catch_up() {
        let mut controller = natural_controller(config_with_ttl(1_000), at(0));
        controller
            .admit_proposal(proposal(1, 0, 1_100), at(0))
            .unwrap();
        controller.tick(at(0)).unwrap();
        controller
            .admit_proposal(proposal(2, 10, 1_100), at(10))
            .unwrap();
        controller.tick(at(10)).unwrap();
        let before = controller.committed_target().position(HeadJoint::Yaw).get();

        let late = controller.tick(at(100)).unwrap();
        let after = late.committed_target().position(HeadJoint::Yaw).get();
        assert!(matches!(
            late.disposition(),
            HeadGazeTickDisposition::LateTickRevoked {
                scheduled_for,
                deadline_inclusive
            } if scheduled_for == at(20) && deadline_inclusive == at(25)
        ));
        assert_eq!(late.next_scheduled_for(), at(110));
        assert!(after.abs_diff(before) <= motion_limit().maximum_position_step().get());
        assert!(matches!(
            controller.tick(at(101)),
            Err(HeadGazeTickError::BeforeScheduledTick {
                scheduled_for,
                observed_at
            }) if scheduled_for == at(110) && observed_at == at(101)
        ));
    }

    #[test]
    fn varied_targets_preserve_position_velocity_step_and_acceleration_bounds() {
        let config = config_with_ttl(1_000);
        let limits = config.motion_limits().joint(HeadJoint::Yaw);
        for target in [0, 1, 250, 999, 1_001, 1_750, 1_999, 2_000] {
            let mut controller = natural_controller(config, at(0));
            let mut id = 1_u64;
            let mut now = 0_u64;
            let mut previous_position =
                controller.committed_target().position(HeadJoint::Yaw).get();
            let mut previous_velocity = 0_i32;
            for step in 0..80 {
                if step < 2 || step % 20 == 0 {
                    controller
                        .admit_proposal(proposal(id, now, target), at(now))
                        .unwrap();
                    id += 1;
                }
                let result = controller.tick(at(now)).unwrap();
                let position = result.committed_target().position(HeadJoint::Yaw).get();
                let velocity = result.velocity().velocity(HeadJoint::Yaw).get();
                assert!(position >= limits.minimum().get());
                assert!(position <= limits.maximum().get());
                assert_eq!(i32::from(position) - i32::from(previous_position), velocity);
                assert!(velocity.abs() <= limits.maximum_velocity().get());
                assert!(
                    position.abs_diff(previous_position) <= limits.maximum_position_step().get()
                );
                assert!(
                    (velocity - previous_velocity).abs() <= limits.maximum_acceleration().get()
                );
                previous_position = position;
                previous_velocity = velocity;
                now += 10;
            }
        }
    }

    #[test]
    fn exact_discrete_braking_distance_and_search_match_exhaustive_small_domains() {
        for acceleration in 1_u64..=16 {
            for speed in 0_u64..=128 {
                let mut simulated = 0_u128;
                let mut remaining_speed = speed;
                while remaining_speed > 0 {
                    simulated += u128::from(remaining_speed);
                    remaining_speed = remaining_speed.saturating_sub(acceleration);
                }
                assert_eq!(
                    discrete_braking_distance(speed, acceleration),
                    simulated,
                    "speed={speed}, acceleration={acceleration}"
                );
            }

            for distance in 0_u64..=1_024 {
                let upper = 128_u64;
                let selected = maximum_stoppable_speed(distance, acceleration, upper);
                assert!(discrete_braking_distance(selected, acceleration) <= u128::from(distance));
                if selected < upper {
                    assert!(
                        discrete_braking_distance(selected + 1, acceleration)
                            > u128::from(distance)
                    );
                }
            }
        }
    }

    #[test]
    fn planner_never_crosses_static_targets_in_exhaustive_small_domain() {
        for acceleration in 1_u32..=4 {
            for speed_limit in 1_u32..=6 {
                let limits = HeadJointMotionLimits::try_new(
                    position(0),
                    position(32),
                    ServoVelocityLimitTicksPerControlTick::try_new(speed_limit).unwrap(),
                    ServoAccelerationLimitTicksPerControlTickSquared::try_new(acceleration)
                        .unwrap(),
                    PositionStepLimit::try_new(u16::try_from(speed_limit).unwrap()).unwrap(),
                )
                .unwrap();
                for start in 0_u16..=20 {
                    for target in 0_u16..=20 {
                        let mut current = position(start);
                        let mut velocity = ServoVelocityTicksPerControlTick::ZERO;
                        let mut completed = false;
                        for _ in 0..128 {
                            let previous = current;
                            let previous_velocity = velocity;
                            (current, velocity) = plan_joint(
                                HeadJoint::Yaw,
                                current,
                                velocity,
                                position(target),
                                limits,
                            )
                            .unwrap();
                            if target >= start {
                                assert!(current >= previous);
                                assert!(current.get() <= target);
                            } else {
                                assert!(current <= previous);
                                assert!(current.get() >= target);
                            }
                            assert!(velocity.get().abs() <= limits.maximum_velocity().get());
                            assert!(
                                (velocity.get() - previous_velocity.get()).abs()
                                    <= limits.maximum_acceleration().get()
                            );
                            if current == position(target)
                                && velocity == ServoVelocityTicksPerControlTick::ZERO
                            {
                                completed = true;
                                break;
                            }
                        }
                        assert!(
                            completed,
                            "start={start}, target={target}, acceleration={acceleration}, speed_limit={speed_limit}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn zero_to_290_and_downward_return_do_not_cross_their_targets() {
        let limits = HeadJointMotionLimits::try_new(
            PositionTicks::MIN,
            PositionTicks::MAX,
            ServoVelocityLimitTicksPerControlTick::try_new(37).unwrap(),
            ServoAccelerationLimitTicksPerControlTickSquared::try_new(7).unwrap(),
            PositionStepLimit::try_new(37).unwrap(),
        )
        .unwrap();
        let mut current = PositionTicks::MIN;
        let mut velocity = ServoVelocityTicksPerControlTick::ZERO;

        for _ in 0..128 {
            let previous = current;
            (current, velocity) =
                plan_joint(HeadJoint::Yaw, current, velocity, position(290), limits).unwrap();
            assert!(current >= previous);
            assert!(current.get() <= 290);
            if current == position(290) && velocity == ServoVelocityTicksPerControlTick::ZERO {
                break;
            }
        }
        assert_eq!(current, position(290));
        assert_eq!(velocity, ServoVelocityTicksPerControlTick::ZERO);

        for _ in 0..128 {
            let previous = current;
            (current, velocity) = plan_joint(
                HeadJoint::Yaw,
                current,
                velocity,
                PositionTicks::MIN,
                limits,
            )
            .unwrap();
            assert!(current <= previous);
            if current == PositionTicks::MIN && velocity == ServoVelocityTicksPerControlTick::ZERO {
                break;
            }
        }
        assert_eq!(current, PositionTicks::MIN);
        assert_eq!(velocity, ServoVelocityTicksPerControlTick::ZERO);
    }

    #[test]
    fn timestamp_overflow_and_infeasible_motion_latch_absorbing_faults() {
        let freshness_observed_at = before_duration_max(2);
        let mut freshness_overflow =
            natural_controller(config_with_ttl(10), before_duration_max(20));
        assert!(matches!(
            freshness_overflow.admit_proposal(
                HeadGazeProposal::new(
                    HeadGazeProposalId::try_new(1).unwrap(),
                    freshness_observed_at,
                    pose(1_100),
                ),
                freshness_observed_at,
            ),
            Err(HeadGazeProposalAdmissionError::FreshnessDeadlineOverflow { .. })
        ));

        let near_max = before_duration_max(2);
        let mut deadline_overflow = natural_controller(config_with_ttl(10), near_max);
        assert!(matches!(
            deadline_overflow.tick(near_max),
            Err(HeadGazeTickError::FaultLatched(
                HeadGazeFaultReason::TickDeadlineOverflow { .. }
            ))
        ));

        let mut next_tick_overflow_config = config_with_ttl(10);
        next_tick_overflow_config.timing.maximum_tick_lateness =
            HeadTickLateness::new(Duration::ZERO);
        let mut schedule_overflow = natural_controller(next_tick_overflow_config, near_max);
        assert!(matches!(
            schedule_overflow.tick(near_max),
            Err(HeadGazeTickError::FaultLatched(
                HeadGazeFaultReason::NextTickTimestampOverflow { .. }
            ))
        ));
        let first_fault = schedule_overflow.fault().unwrap();
        schedule_overflow.latch_external_fault(HeadGazeExternalFault::EmergencyStop);
        assert_eq!(schedule_overflow.fault(), Some(first_fault));
        assert_eq!(schedule_overflow.state(), HeadGazeControlState::FaultHeld);
        assert!(matches!(
            schedule_overflow.tick(before_duration_max(1)),
            Err(HeadGazeTickError::FaultHeld(reason)) if reason == first_fault
        ));
        assert!(matches!(
            schedule_overflow.admit_proposal(
                HeadGazeProposal::new(
                    HeadGazeProposalId::try_new(1).unwrap(),
                    before_duration_max(1),
                    pose(1_000),
                ),
                before_duration_max(1),
            ),
            Err(HeadGazeProposalAdmissionError::FaultHeld(reason)) if reason == first_fault
        ));

        let infeasible_config = config_with_ttl(100);
        let mut infeasible =
            HeadGazeController::try_new(infeasible_config, pose(2_000), at(0)).unwrap();
        infeasible.velocity.velocities[joint_index(HeadJoint::Yaw)] =
            ServoVelocityTicksPerControlTick::from_planned(5);
        assert!(matches!(
            infeasible.tick(at(0)),
            Err(HeadGazeTickError::FaultLatched(
                HeadGazeFaultReason::MotionConstraintsInfeasible {
                    joint: HeadJoint::Yaw
                }
            ))
        ));
        assert_eq!(infeasible.state(), HeadGazeControlState::FaultHeld);
        assert_eq!(infeasible.velocity(), HeadServoVelocity::ZERO);
    }

    #[test]
    fn exact_lateness_deadline_is_serviced_but_clock_regression_faults() {
        let mut controller = natural_controller(config_with_ttl(100), at(10));
        let step = controller.tick(at(15)).unwrap();
        assert!(matches!(
            step.disposition(),
            HeadGazeTickDisposition::OnTime {
                deadline_inclusive
            } if deadline_inclusive == at(15)
        ));
        assert!(matches!(
            controller.tick(at(14)),
            Err(HeadGazeTickError::FaultLatched(
                HeadGazeFaultReason::TickClockRegression {
                    previous,
                    actual
                }
            )) if previous == at(15) && actual == at(14)
        ));
    }

    #[test]
    fn constructor_starts_from_the_exact_committed_target_without_inventing_natural() {
        let config = config_with_ttl(100);
        let mut controller = HeadGazeController::try_new(config, pose(1_100), at(0)).unwrap();
        assert_eq!(controller.state(), HeadGazeControlState::ReturningNatural);
        assert_eq!(controller.committed_target(), pose(1_100));
        assert_eq!(controller.velocity(), HeadServoVelocity::ZERO);

        let first = controller.tick(at(0)).unwrap();
        assert_eq!(first.state(), HeadGazeControlState::ReturningNatural);
        assert!(first.committed_target().position(HeadJoint::Yaw).get() < 1_100);
        assert!(first.committed_target().position(HeadJoint::Yaw).get() >= 1_000);

        assert!(matches!(
            HeadGazeController::try_new(config, pose(2_001), at(0)),
            Err(HeadGazeControllerInitError::InitialCommittedTargetOutOfRange {
                joint: HeadJoint::Yaw,
                position: actual,
                minimum,
                maximum,
            }) if actual == position(2_001)
                && minimum == position(0)
                && maximum == position(2_000)
        ));
    }

    #[test]
    fn configuration_rejects_natural_pose_outside_any_joint_range() {
        let timing = HeadGazeTiming::new(
            HeadControlPeriod::try_new(Duration::from_nanos(10)).unwrap(),
            HeadTickLateness::new(Duration::from_nanos(5)),
            HeadProposalTtl::try_new(Duration::from_nanos(50)).unwrap(),
            HeadAcquisitionProposalCount::try_new(2).unwrap(),
        );
        let limit = motion_limit();
        assert!(matches!(
            HeadGazeControlConfig::try_new(
                timing,
                pose(2_001),
                HeadMotionLimits::new(limit, limit, limit, limit),
                HeadGazeErrorBand::try_new(
                    HeadDeadbandTicks::try_new(1).unwrap(),
                    HeadResumeThresholdTicks::try_new(3).unwrap(),
                )
                .unwrap(),
            ),
            Err(HeadGazeControlConfigError::NaturalPoseOutOfRange {
                joint: HeadJoint::Yaw,
                ..
            })
        ));
    }
}
