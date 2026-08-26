//! Transport-free, four-axis compliant-hold planning.
//!
//! This controller lets an already torque-limited head yield to a deliberate
//! external displacement, briefly settle with it, and return to the pose that
//! was active when contact began. It never interprets the STS `load_raw` or
//! `current_raw` registers as force: their sign and physical units are not
//! qualified by Kiko's protocol contract. Contact evidence is instead a
//! repeatable encoder error against the sole owner's verified goal.
//!
//! The controller is transactional. Preparing a step advances only a private
//! candidate snapshot. The sole bus owner must apply and verify the complete
//! four-joint goal before committing it. A partial or uncertain write must be
//! fault-aborted, never committed.

use std::{
    fmt,
    num::{NonZeroU8, NonZeroU16, NonZeroU64},
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use kiko_head_protocol::{
    ExactHeadTargetPose, FullTelemetry, HeadJoint, HeadTorqueLimits, PositionTicks,
};

use crate::{HeadTelemetrySafetyLimits, HeadTelemetrySafetyViolation, MonotonicTime};

const JOINT_COUNT: usize = 4;
const INTERPOLATION_SCALE: u128 = 1_000_000;
static NEXT_COMPLIANT_CONTROLLER_ID: AtomicU64 = AtomicU64::new(1);

const fn joint_index(joint: HeadJoint) -> usize {
    joint as usize
}

/// Encoder-domain admission and travel policy for one joint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompliantJointPolicy {
    minimum: PositionTicks,
    maximum: PositionTicks,
    contact_entry_error_ticks: NonZeroU16,
    contact_release_error_ticks: u16,
    maximum_yield_ticks: NonZeroU16,
    maximum_command_step_ticks: NonZeroU16,
    maximum_observed_step_ticks: NonZeroU16,
}

impl CompliantJointPolicy {
    pub fn try_new(
        minimum: PositionTicks,
        maximum: PositionTicks,
        contact_entry_error_ticks: u16,
        contact_release_error_ticks: u16,
        maximum_yield_ticks: u16,
        maximum_command_step_ticks: u16,
        maximum_observed_step_ticks: u16,
    ) -> Result<Self, CompliantJointPolicyError> {
        if minimum >= maximum {
            return Err(CompliantJointPolicyError::EmptyEnvelope { minimum, maximum });
        }
        let contact_entry_error_ticks = NonZeroU16::new(contact_entry_error_ticks)
            .ok_or(CompliantJointPolicyError::ZeroContactEntryError)?;
        if contact_release_error_ticks >= contact_entry_error_ticks.get() {
            return Err(CompliantJointPolicyError::ReleaseNotInsideEntry {
                release_ticks: contact_release_error_ticks,
                entry_ticks: contact_entry_error_ticks.get(),
            });
        }
        let maximum_yield_ticks = NonZeroU16::new(maximum_yield_ticks)
            .ok_or(CompliantJointPolicyError::ZeroMaximumYield)?;
        if maximum_yield_ticks.get() < contact_entry_error_ticks.get() {
            return Err(CompliantJointPolicyError::YieldSmallerThanEntry {
                maximum_yield_ticks: maximum_yield_ticks.get(),
                entry_ticks: contact_entry_error_ticks.get(),
            });
        }
        let envelope_span = maximum.get() - minimum.get();
        if maximum_yield_ticks.get() > envelope_span {
            return Err(CompliantJointPolicyError::YieldExceedsEnvelopeSpan {
                maximum_yield_ticks: maximum_yield_ticks.get(),
                envelope_span_ticks: envelope_span,
            });
        }
        let maximum_command_step_ticks = NonZeroU16::new(maximum_command_step_ticks)
            .ok_or(CompliantJointPolicyError::ZeroMaximumCommandStep)?;
        let maximum_observed_step_ticks = NonZeroU16::new(maximum_observed_step_ticks)
            .ok_or(CompliantJointPolicyError::ZeroMaximumObservedStep)?;
        if maximum_observed_step_ticks.get() < maximum_command_step_ticks.get() {
            return Err(
                CompliantJointPolicyError::ObservedStepSmallerThanCommandStep {
                    maximum_observed_step_ticks: maximum_observed_step_ticks.get(),
                    maximum_command_step_ticks: maximum_command_step_ticks.get(),
                },
            );
        }
        if maximum_observed_step_ticks.get() < contact_entry_error_ticks.get() {
            return Err(CompliantJointPolicyError::ObservedStepSmallerThanEntry {
                maximum_observed_step_ticks: maximum_observed_step_ticks.get(),
                entry_ticks: contact_entry_error_ticks.get(),
            });
        }
        Ok(Self {
            minimum,
            maximum,
            contact_entry_error_ticks,
            contact_release_error_ticks,
            maximum_yield_ticks,
            maximum_command_step_ticks,
            maximum_observed_step_ticks,
        })
    }

    pub const fn minimum(self) -> PositionTicks {
        self.minimum
    }

    pub const fn maximum(self) -> PositionTicks {
        self.maximum
    }

    pub const fn contact_entry_error_ticks(self) -> u16 {
        self.contact_entry_error_ticks.get()
    }

    pub const fn contact_release_error_ticks(self) -> u16 {
        self.contact_release_error_ticks
    }

    pub const fn maximum_yield_ticks(self) -> u16 {
        self.maximum_yield_ticks.get()
    }

    pub const fn maximum_command_step_ticks(self) -> u16 {
        self.maximum_command_step_ticks.get()
    }

    pub const fn maximum_observed_step_ticks(self) -> u16 {
        self.maximum_observed_step_ticks.get()
    }

    const fn contains(self, value: PositionTicks) -> bool {
        value.get() >= self.minimum.get() && value.get() <= self.maximum.get()
    }

    /// Lowest physically observable position admitted while a safe command
    /// remains at the command envelope edge.
    ///
    /// Commands never use this wider range. A person can backdrive a joint by
    /// at most the separately reviewed yield distance beyond an edge, so
    /// rejecting that observation would turn the compliant controller's own
    /// permitted interaction into an absorbing fault.
    pub const fn observation_minimum(self) -> PositionTicks {
        let value = self
            .minimum
            .get()
            .saturating_sub(self.maximum_yield_ticks.get());
        match PositionTicks::try_new(value) {
            Ok(position) => position,
            Err(_) => panic!("a saturating subtraction stays inside the encoder domain"),
        }
    }

    /// Highest physically observable position admitted while a safe command
    /// remains at the command envelope edge.
    pub const fn observation_maximum(self) -> PositionTicks {
        let candidate = self
            .maximum
            .get()
            .saturating_add(self.maximum_yield_ticks.get());
        let value = if candidate > PositionTicks::MAX.get() {
            PositionTicks::MAX.get()
        } else {
            candidate
        };
        match PositionTicks::try_new(value) {
            Ok(position) => position,
            Err(_) => panic!("the observation maximum is capped to the encoder domain"),
        }
    }

    const fn contains_observation(self, value: PositionTicks) -> bool {
        value.get() >= self.observation_minimum().get()
            && value.get() <= self.observation_maximum().get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompliantJointPolicyError {
    EmptyEnvelope {
        minimum: PositionTicks,
        maximum: PositionTicks,
    },
    ZeroContactEntryError,
    ReleaseNotInsideEntry {
        release_ticks: u16,
        entry_ticks: u16,
    },
    ZeroMaximumYield,
    YieldSmallerThanEntry {
        maximum_yield_ticks: u16,
        entry_ticks: u16,
    },
    YieldExceedsEnvelopeSpan {
        maximum_yield_ticks: u16,
        envelope_span_ticks: u16,
    },
    ZeroMaximumCommandStep,
    ZeroMaximumObservedStep,
    ObservedStepSmallerThanCommandStep {
        maximum_observed_step_ticks: u16,
        maximum_command_step_ticks: u16,
    },
    ObservedStepSmallerThanEntry {
        maximum_observed_step_ticks: u16,
        entry_ticks: u16,
    },
}

impl fmt::Display for CompliantJointPolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid compliant joint policy: {self:?}")
    }
}

impl std::error::Error for CompliantJointPolicyError {}

/// A complete four-axis compliant-hold policy.
///
/// `holding_torque_limits` are evidence binding, not adaptive output. They
/// must exactly match the torque limits installed by the owning head runtime.
/// Physical commissioning remains responsible for finding the lowest limits
/// that safely support the assembly, especially bow and curl against gravity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadCompliantHoldConfig {
    joints: [CompliantJointPolicy; JOINT_COUNT],
    holding_torque_limits: HeadTorqueLimits,
    control_period: Duration,
    observation_transaction_timeout: Duration,
    maximum_observation_span: Duration,
    observation_ttl: Duration,
    contact_arm_dwell: Duration,
    contact_acquisition_samples: NonZeroU8,
    release_dwell: Duration,
    recovery_duration: Duration,
    follow_permille: NonZeroU16,
}

impl HeadCompliantHoldConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        bow: CompliantJointPolicy,
        curl: CompliantJointPolicy,
        yaw: CompliantJointPolicy,
        roll: CompliantJointPolicy,
        holding_torque_limits: HeadTorqueLimits,
        control_period: Duration,
        observation_transaction_timeout: Duration,
        maximum_observation_span: Duration,
        observation_ttl: Duration,
        contact_arm_dwell: Duration,
        contact_acquisition_samples: u8,
        release_dwell: Duration,
        recovery_duration: Duration,
        follow_permille: u16,
    ) -> Result<Self, HeadCompliantHoldConfigError> {
        if control_period.is_zero() {
            return Err(HeadCompliantHoldConfigError::ZeroControlPeriod);
        }
        if observation_transaction_timeout.is_zero() {
            return Err(HeadCompliantHoldConfigError::ZeroObservationTransactionTimeout);
        }
        if observation_transaction_timeout > control_period {
            return Err(
                HeadCompliantHoldConfigError::ObservationTransactionExceedsControlPeriod {
                    transaction_timeout: observation_transaction_timeout,
                    control_period,
                },
            );
        }
        if maximum_observation_span.is_zero() {
            return Err(HeadCompliantHoldConfigError::ZeroMaximumObservationSpan);
        }
        if maximum_observation_span > observation_transaction_timeout {
            return Err(
                HeadCompliantHoldConfigError::ObservationSpanExceedsTransaction {
                    maximum_span: maximum_observation_span,
                    transaction_timeout: observation_transaction_timeout,
                },
            );
        }
        if observation_ttl.is_zero() {
            return Err(HeadCompliantHoldConfigError::ZeroObservationTtl);
        }
        if maximum_observation_span >= observation_ttl {
            return Err(HeadCompliantHoldConfigError::ObservationSpanNotInsideTtl {
                maximum_span: maximum_observation_span,
                ttl: observation_ttl,
            });
        }
        if contact_arm_dwell.is_zero() {
            return Err(HeadCompliantHoldConfigError::ZeroContactArmDwell);
        }
        let contact_acquisition_samples = NonZeroU8::new(contact_acquisition_samples)
            .ok_or(HeadCompliantHoldConfigError::ZeroContactAcquisitionSamples)?;
        if release_dwell.is_zero() {
            return Err(HeadCompliantHoldConfigError::ZeroReleaseDwell);
        }
        if recovery_duration.is_zero() {
            return Err(HeadCompliantHoldConfigError::ZeroRecoveryDuration);
        }
        let follow_permille = NonZeroU16::new(follow_permille)
            .ok_or(HeadCompliantHoldConfigError::ZeroFollowPermille)?;
        if follow_permille.get() > 1_000 {
            return Err(HeadCompliantHoldConfigError::FollowPermilleOutOfRange {
                value: follow_permille.get(),
            });
        }
        Ok(Self {
            joints: [bow, curl, yaw, roll],
            holding_torque_limits,
            control_period,
            observation_transaction_timeout,
            maximum_observation_span,
            observation_ttl,
            contact_arm_dwell,
            contact_acquisition_samples,
            release_dwell,
            recovery_duration,
            follow_permille,
        })
    }

    pub const fn joint(self, joint: HeadJoint) -> CompliantJointPolicy {
        self.joints[joint_index(joint)]
    }

    pub const fn holding_torque_limits(self) -> HeadTorqueLimits {
        self.holding_torque_limits
    }

    pub const fn control_period(self) -> Duration {
        self.control_period
    }

    pub const fn observation_transaction_timeout(self) -> Duration {
        self.observation_transaction_timeout
    }

    pub const fn maximum_observation_span(self) -> Duration {
        self.maximum_observation_span
    }

    pub const fn observation_ttl(self) -> Duration {
        self.observation_ttl
    }

    pub const fn contact_arm_dwell(self) -> Duration {
        self.contact_arm_dwell
    }

    pub const fn contact_acquisition_samples(self) -> u8 {
        self.contact_acquisition_samples.get()
    }

    pub const fn release_dwell(self) -> Duration {
        self.release_dwell
    }

    pub const fn recovery_duration(self) -> Duration {
        self.recovery_duration
    }

    pub const fn follow_permille(self) -> u16 {
        self.follow_permille.get()
    }

    pub fn admit_runtime_torque_limits(
        self,
        actual: HeadTorqueLimits,
    ) -> Result<(), HeadCompliantTorqueBindingError> {
        for joint in HeadJoint::ALL {
            let required = self.holding_torque_limits.for_joint(joint);
            let actual = actual.for_joint(joint);
            if actual != required {
                return Err(HeadCompliantTorqueBindingError::Mismatch {
                    joint,
                    required,
                    actual,
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadCompliantHoldConfigError {
    ZeroControlPeriod,
    ZeroObservationTransactionTimeout,
    ObservationTransactionExceedsControlPeriod {
        transaction_timeout: Duration,
        control_period: Duration,
    },
    ZeroMaximumObservationSpan,
    ObservationSpanExceedsTransaction {
        maximum_span: Duration,
        transaction_timeout: Duration,
    },
    ZeroObservationTtl,
    ObservationSpanNotInsideTtl {
        maximum_span: Duration,
        ttl: Duration,
    },
    ZeroContactArmDwell,
    ZeroContactAcquisitionSamples,
    ZeroReleaseDwell,
    ZeroRecoveryDuration,
    ZeroFollowPermille,
    FollowPermilleOutOfRange {
        value: u16,
    },
}

impl fmt::Display for HeadCompliantHoldConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid compliant-hold configuration: {self:?}")
    }
}

impl std::error::Error for HeadCompliantHoldConfigError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadCompliantTorqueBindingError {
    Mismatch {
        joint: HeadJoint,
        required: kiko_head_protocol::TorqueLimitPermille,
        actual: kiko_head_protocol::TorqueLimitPermille,
    },
}

impl fmt::Display for HeadCompliantTorqueBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "compliant-hold torque binding failed: {self:?}")
    }
}

impl std::error::Error for HeadCompliantTorqueBindingError {}

/// One complete telemetry observation admitted for compliant control.
///
/// Raw load/current are retained for diagnostics and future calibration only.
/// They do not influence contact, yield, or recovery decisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompliantHeadObservation {
    observed_at: MonotonicTime,
    positions: [PositionTicks; JOINT_COUNT],
    moving: [bool; JOINT_COUNT],
    load_raw: [u16; JOINT_COUNT],
    current_raw: [u16; JOINT_COUNT],
}

impl CompliantHeadObservation {
    pub fn try_from_timed_telemetry(
        samples: [FullTelemetry; JOINT_COUNT],
        received_at: [MonotonicTime; JOINT_COUNT],
        admitted_at: MonotonicTime,
        safety: HeadTelemetrySafetyLimits,
        maximum_span: Duration,
        ttl: Duration,
    ) -> Result<Self, CompliantHeadObservationError> {
        let mut positions = [PositionTicks::MIN; JOINT_COUNT];
        let mut moving = [false; JOINT_COUNT];
        let mut load_raw = [0; JOINT_COUNT];
        let mut current_raw = [0; JOINT_COUNT];
        let mut first_temperature_violation = None;
        for index in 1..JOINT_COUNT {
            if received_at[index] < received_at[index - 1] {
                return Err(CompliantHeadObservationError::ClockRegression {
                    previous_joint: HeadJoint::ALL[index - 1],
                    previous: received_at[index - 1],
                    actual_joint: HeadJoint::ALL[index],
                    actual: received_at[index],
                });
            }
        }
        let span = received_at[JOINT_COUNT - 1]
            .checked_duration_since(received_at[0])
            .expect("ordered timestamps have a duration");
        if span > maximum_span {
            return Err(CompliantHeadObservationError::SetSpanExceeded {
                first: received_at[0],
                last: received_at[JOINT_COUNT - 1],
                span,
                maximum: maximum_span,
            });
        }
        let Some(age) = admitted_at.checked_duration_since(received_at[0]) else {
            return Err(CompliantHeadObservationError::AdmittedBeforeObservation {
                first: received_at[0],
                admitted_at,
            });
        };
        if age >= ttl {
            return Err(CompliantHeadObservationError::SetExpired {
                first: received_at[0],
                admitted_at,
                age,
                ttl,
            });
        }
        for (index, joint) in HeadJoint::ALL.into_iter().enumerate() {
            let sample = samples[index];
            if sample.id() != joint.servo_id() {
                return Err(CompliantHeadObservationError::ServoIdMismatch {
                    joint,
                    expected: joint.servo_id(),
                    actual: sample.id(),
                });
            }
            if sample.device_status_raw() != 0 {
                return Err(CompliantHeadObservationError::DeviceStatus {
                    joint,
                    raw: sample.device_status_raw(),
                });
            }
            match safety.admit_energized(sample.voltage_raw(), sample.temperature_raw()) {
                Ok(()) => {}
                Err(
                    source @ HeadTelemetrySafetyViolation::EnergizedTemperatureAtOrAboveExclusiveMaximum {
                        ..
                    },
                ) => {
                    first_temperature_violation.get_or_insert((joint, source));
                }
                Err(source) => {
                    return Err(CompliantHeadObservationError::TelemetrySafety {
                        joint,
                        source,
                    });
                }
            }
            positions[index] = sample.position();
            moving[index] = sample.is_moving();
            load_raw[index] = sample.load_raw();
            current_raw[index] = sample.current_raw();
        }
        if let Some((joint, source)) = first_temperature_violation {
            return Err(CompliantHeadObservationError::TelemetrySafety { joint, source });
        }
        Ok(Self {
            observed_at: received_at[JOINT_COUNT - 1],
            positions,
            moving,
            load_raw,
            current_raw,
        })
    }

    #[cfg(test)]
    fn from_parts(
        observed_at: MonotonicTime,
        positions: [u16; JOINT_COUNT],
        moving: [bool; JOINT_COUNT],
    ) -> Self {
        Self {
            observed_at,
            positions: positions.map(|value| PositionTicks::try_new(value).unwrap()),
            moving,
            load_raw: [0; JOINT_COUNT],
            current_raw: [0; JOINT_COUNT],
        }
    }

    pub const fn observed_at(self) -> MonotonicTime {
        self.observed_at
    }

    pub const fn position(self, joint: HeadJoint) -> PositionTicks {
        self.positions[joint_index(joint)]
    }

    pub const fn is_moving(self, joint: HeadJoint) -> bool {
        self.moving[joint_index(joint)]
    }

    pub const fn load_raw(self, joint: HeadJoint) -> u16 {
        self.load_raw[joint_index(joint)]
    }

    pub const fn current_raw(self, joint: HeadJoint) -> u16 {
        self.current_raw[joint_index(joint)]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompliantHeadObservationError {
    ClockRegression {
        previous_joint: HeadJoint,
        previous: MonotonicTime,
        actual_joint: HeadJoint,
        actual: MonotonicTime,
    },
    SetSpanExceeded {
        first: MonotonicTime,
        last: MonotonicTime,
        span: Duration,
        maximum: Duration,
    },
    AdmittedBeforeObservation {
        first: MonotonicTime,
        admitted_at: MonotonicTime,
    },
    SetExpired {
        first: MonotonicTime,
        admitted_at: MonotonicTime,
        age: Duration,
        ttl: Duration,
    },
    ServoIdMismatch {
        joint: HeadJoint,
        expected: kiko_head_protocol::ServoId,
        actual: kiko_head_protocol::ServoId,
    },
    DeviceStatus {
        joint: HeadJoint,
        raw: u8,
    },
    TelemetrySafety {
        joint: HeadJoint,
        source: HeadTelemetrySafetyViolation,
    },
}

impl fmt::Display for CompliantHeadObservationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "compliant head observation rejected: {self:?}")
    }
}

impl std::error::Error for CompliantHeadObservationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::TelemetrySafety { source, .. } => Some(source),
            Self::ClockRegression { .. }
            | Self::SetSpanExceeded { .. }
            | Self::AdmittedBeforeObservation { .. }
            | Self::SetExpired { .. }
            | Self::ServoIdMismatch { .. }
            | Self::DeviceStatus { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompliantHoldState {
    FollowingExpression,
    ConfirmingContact,
    Yielding,
    ReleaseDwell,
    Recovering,
    FaultHeld,
}

impl CompliantHoldState {
    pub const fn suppresses_expression_motion(self) -> bool {
        !matches!(self, Self::FollowingExpression)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompliantHoldFault {
    ObservationClockRegression {
        previous: MonotonicTime,
        actual: MonotonicTime,
    },
    ObservationOutsideEnvelope {
        joint: HeadJoint,
        observed: PositionTicks,
        minimum: PositionTicks,
        maximum: PositionTicks,
    },
    ObservationDiscontinuity {
        joint: HeadJoint,
        previous: PositionTicks,
        actual: PositionTicks,
        difference_ticks: u16,
        maximum_ticks: u16,
    },
    ApplicationUncertain,
    GenerationExhausted,
    NextServiceTimestampOverflow {
        serviced_at: MonotonicTime,
        control_period: Duration,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompliantHoldDisposition {
    FollowingExpression,
    ContactCandidate {
        consecutive_samples: u8,
    },
    Yielding {
        envelope_limited: [bool; JOINT_COUNT],
        command_step_limited: [bool; JOINT_COUNT],
    },
    ReleaseDwell,
    Recovering {
        progress_millionths: u32,
        command_step_limited: [bool; JOINT_COUNT],
    },
    ReturnedToExpression,
}

impl CompliantHoldDisposition {
    pub const fn suppresses_expression_motion(self) -> bool {
        !matches!(self, Self::FollowingExpression | Self::ReturnedToExpression)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ContactCandidate {
    return_target: ExactHeadTargetPose,
    directions: [i8; JOINT_COUNT],
    consecutive_samples: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ControllerPhase {
    FollowingExpression {
        quiescent_since: Option<MonotonicTime>,
        contact_armed: bool,
    },
    Confirming(ContactCandidate),
    Yielding {
        return_target: ExactHeadTargetPose,
        quiet_since: Option<MonotonicTime>,
    },
    Recovering {
        return_target: ExactHeadTargetPose,
        recovery_start: ExactHeadTargetPose,
        started_at: MonotonicTime,
        reacquisition: Option<ContactCandidate>,
    },
}

const fn following_expression_unarmed() -> ControllerPhase {
    ControllerPhase::FollowingExpression {
        quiescent_since: None,
        contact_armed: false,
    }
}

#[derive(Clone, Debug)]
pub struct HeadCompliantHoldController {
    instance_id: NonZeroU64,
    generation: u64,
    config: HeadCompliantHoldConfig,
    phase: ControllerPhase,
    committed_target: ExactHeadTargetPose,
    previous_observation: Option<CompliantHeadObservation>,
    next_service_due: MonotonicTime,
    latest_boundary_at: MonotonicTime,
    fault: Option<CompliantHoldFault>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CompliantHoldPreparedToken {
    controller_instance: NonZeroU64,
    generation: u64,
}

#[derive(Debug)]
pub struct PreparedCompliantHoldStep {
    token: CompliantHoldPreparedToken,
    state: CompliantHoldState,
    target: ExactHeadTargetPose,
    disposition: CompliantHoldDisposition,
    observation: CompliantHeadObservation,
    candidate: HeadCompliantHoldController,
}

impl PreparedCompliantHoldStep {
    pub const fn token(&self) -> CompliantHoldPreparedToken {
        self.token
    }

    pub const fn state(&self) -> CompliantHoldState {
        self.state
    }

    pub const fn target(&self) -> ExactHeadTargetPose {
        self.target
    }

    pub const fn disposition(&self) -> CompliantHoldDisposition {
        self.disposition
    }

    pub const fn observation(&self) -> CompliantHeadObservation {
        self.observation
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompliantHoldCommitReceipt {
    token: CompliantHoldPreparedToken,
    state: CompliantHoldState,
    committed_target: ExactHeadTargetPose,
    disposition: CompliantHoldDisposition,
    observation: CompliantHeadObservation,
}

impl CompliantHoldCommitReceipt {
    pub const fn state(self) -> CompliantHoldState {
        self.state
    }

    pub const fn committed_target(self) -> ExactHeadTargetPose {
        self.committed_target
    }

    pub const fn disposition(self) -> CompliantHoldDisposition {
        self.disposition
    }

    pub const fn observation(self) -> CompliantHeadObservation {
        self.observation
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompliantHoldPrepareError {
    FaultHeld(CompliantHoldFault),
    FaultLatched(CompliantHoldFault),
    ObservationInFuture {
        observed_at: MonotonicTime,
        serviced_at: MonotonicTime,
    },
    ObservationExpired {
        observed_at: MonotonicTime,
        serviced_at: MonotonicTime,
        age: Duration,
        ttl: Duration,
    },
    BeforeScheduledService {
        scheduled_for: MonotonicTime,
        observed_at: MonotonicTime,
    },
    ExpressionTargetOutsideEnvelope {
        joint: HeadJoint,
        target: PositionTicks,
        minimum: PositionTicks,
        maximum: PositionTicks,
    },
}

impl fmt::Display for CompliantHoldPrepareError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "compliant-hold preparation failed: {self:?}")
    }
}

impl std::error::Error for CompliantHoldPrepareError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompliantHoldCommitError {
    WrongController,
    StaleGeneration { current: u64, prepared_from: u64 },
    FutureGeneration { current: u64, prepared_from: u64 },
    FaultHeld(CompliantHoldFault),
}

impl fmt::Display for CompliantHoldCommitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "prepared compliant-hold step rejected: {self:?}")
    }
}

impl std::error::Error for CompliantHoldCommitError {}

impl HeadCompliantHoldController {
    pub fn try_new(
        config: HeadCompliantHoldConfig,
        initial_committed_target: ExactHeadTargetPose,
        started_at: MonotonicTime,
    ) -> Result<Self, CompliantHoldPrepareError> {
        admit_target(config, initial_committed_target)?;
        let raw = NEXT_COMPLIANT_CONTROLLER_ID.fetch_add(1, Ordering::Relaxed);
        let instance_id = NonZeroU64::new(raw).ok_or({
            CompliantHoldPrepareError::FaultLatched(CompliantHoldFault::GenerationExhausted)
        })?;
        Ok(Self {
            instance_id,
            generation: 0,
            config,
            phase: ControllerPhase::FollowingExpression {
                quiescent_since: None,
                contact_armed: false,
            },
            committed_target: initial_committed_target,
            previous_observation: None,
            next_service_due: started_at,
            latest_boundary_at: started_at,
            fault: None,
        })
    }

    pub const fn config(&self) -> HeadCompliantHoldConfig {
        self.config
    }

    pub const fn committed_target(&self) -> ExactHeadTargetPose {
        self.committed_target
    }

    pub const fn state(&self) -> CompliantHoldState {
        match (self.fault, self.phase) {
            (Some(_), _) => CompliantHoldState::FaultHeld,
            (None, ControllerPhase::FollowingExpression { .. }) => {
                CompliantHoldState::FollowingExpression
            }
            (None, ControllerPhase::Confirming(_)) => CompliantHoldState::ConfirmingContact,
            (
                None,
                ControllerPhase::Yielding {
                    quiet_since: None, ..
                },
            ) => CompliantHoldState::Yielding,
            (
                None,
                ControllerPhase::Yielding {
                    quiet_since: Some(_),
                    ..
                },
            ) => CompliantHoldState::ReleaseDwell,
            (None, ControllerPhase::Recovering { .. }) => CompliantHoldState::Recovering,
        }
    }

    pub const fn fault(&self) -> Option<CompliantHoldFault> {
        self.fault
    }

    pub const fn next_service_due(&self) -> MonotonicTime {
        self.next_service_due
    }

    /// Prepare one observation-driven command.
    ///
    /// `expression_quiet` must be true only when the lower-priority gaze or
    /// character planner has zero commanded velocity and has retained its
    /// target. This prevents normal commanded motion from being misclassified
    /// as touch. Once contact is acquired, the captured return target remains
    /// authoritative until recovery completes.
    pub fn prepare(
        &mut self,
        serviced_at: MonotonicTime,
        expression_target: ExactHeadTargetPose,
        expression_quiet: bool,
        observation: CompliantHeadObservation,
    ) -> Result<PreparedCompliantHoldStep, CompliantHoldPrepareError> {
        if let Some(fault) = self.fault {
            return Err(CompliantHoldPrepareError::FaultHeld(fault));
        }
        if serviced_at < self.next_service_due {
            return Err(CompliantHoldPrepareError::BeforeScheduledService {
                scheduled_for: self.next_service_due,
                observed_at: serviced_at,
            });
        }
        admit_target(self.config, expression_target)?;
        self.admit_observation_time(serviced_at, observation)?;

        let Some(next_generation) = self.generation.checked_add(1) else {
            let fault = CompliantHoldFault::GenerationExhausted;
            self.fault = Some(fault);
            return Err(CompliantHoldPrepareError::FaultLatched(fault));
        };
        let token = CompliantHoldPreparedToken {
            controller_instance: self.instance_id,
            generation: self.generation,
        };
        let mut candidate = self.clone();
        let disposition = candidate.advance(
            serviced_at,
            expression_target,
            expression_quiet,
            observation,
        )?;
        candidate.generation = next_generation;
        candidate.previous_observation = Some(observation);
        candidate.latest_boundary_at = serviced_at;
        candidate.next_service_due = checked_time_add(serviced_at, self.config.control_period())
            .ok_or_else(|| {
                let fault = CompliantHoldFault::NextServiceTimestampOverflow {
                    serviced_at,
                    control_period: self.config.control_period(),
                };
                self.fault = Some(fault);
                CompliantHoldPrepareError::FaultLatched(fault)
            })?;
        let target = candidate.committed_target;
        let state = candidate.state();
        Ok(PreparedCompliantHoldStep {
            token,
            state,
            target,
            disposition,
            observation,
            candidate,
        })
    }

    pub fn commit(
        &mut self,
        prepared: PreparedCompliantHoldStep,
    ) -> Result<CompliantHoldCommitReceipt, CompliantHoldCommitError> {
        self.validate_token(prepared.token)?;
        let PreparedCompliantHoldStep {
            token,
            state,
            target,
            disposition,
            observation,
            candidate,
        } = prepared;
        *self = candidate;
        Ok(CompliantHoldCommitReceipt {
            token,
            state,
            committed_target: target,
            disposition,
            observation,
        })
    }

    pub fn abort_with_application_uncertain(
        &mut self,
        prepared: PreparedCompliantHoldStep,
    ) -> Result<(), CompliantHoldCommitError> {
        self.validate_token(prepared.token)?;
        self.fault = Some(CompliantHoldFault::ApplicationUncertain);
        Ok(())
    }

    fn validate_token(
        &self,
        token: CompliantHoldPreparedToken,
    ) -> Result<(), CompliantHoldCommitError> {
        if token.controller_instance != self.instance_id {
            return Err(CompliantHoldCommitError::WrongController);
        }
        if let Some(fault) = self.fault {
            return Err(CompliantHoldCommitError::FaultHeld(fault));
        }
        if token.generation < self.generation {
            return Err(CompliantHoldCommitError::StaleGeneration {
                current: self.generation,
                prepared_from: token.generation,
            });
        }
        if token.generation > self.generation {
            return Err(CompliantHoldCommitError::FutureGeneration {
                current: self.generation,
                prepared_from: token.generation,
            });
        }
        Ok(())
    }

    fn admit_observation_time(
        &mut self,
        serviced_at: MonotonicTime,
        observation: CompliantHeadObservation,
    ) -> Result<(), CompliantHoldPrepareError> {
        if serviced_at < self.latest_boundary_at {
            let fault = CompliantHoldFault::ObservationClockRegression {
                previous: self.latest_boundary_at,
                actual: serviced_at,
            };
            self.fault = Some(fault);
            return Err(CompliantHoldPrepareError::FaultLatched(fault));
        }
        let Some(age) = serviced_at.checked_duration_since(observation.observed_at()) else {
            return Err(CompliantHoldPrepareError::ObservationInFuture {
                observed_at: observation.observed_at(),
                serviced_at,
            });
        };
        if age >= self.config.observation_ttl() {
            return Err(CompliantHoldPrepareError::ObservationExpired {
                observed_at: observation.observed_at(),
                serviced_at,
                age,
                ttl: self.config.observation_ttl(),
            });
        }
        for joint in HeadJoint::ALL {
            let policy = self.config.joint(joint);
            let observed = observation.position(joint);
            if !policy.contains_observation(observed) {
                let fault = CompliantHoldFault::ObservationOutsideEnvelope {
                    joint,
                    observed,
                    minimum: policy.observation_minimum(),
                    maximum: policy.observation_maximum(),
                };
                self.fault = Some(fault);
                return Err(CompliantHoldPrepareError::FaultLatched(fault));
            }
            if let Some(previous) = self.previous_observation {
                let previous = previous.position(joint);
                let difference_ticks = previous.get().abs_diff(observed.get());
                if difference_ticks > policy.maximum_observed_step_ticks() {
                    let fault = CompliantHoldFault::ObservationDiscontinuity {
                        joint,
                        previous,
                        actual: observed,
                        difference_ticks,
                        maximum_ticks: policy.maximum_observed_step_ticks(),
                    };
                    self.fault = Some(fault);
                    return Err(CompliantHoldPrepareError::FaultLatched(fault));
                }
            }
        }
        Ok(())
    }

    fn advance(
        &mut self,
        now: MonotonicTime,
        expression_target: ExactHeadTargetPose,
        expression_quiet: bool,
        observation: CompliantHeadObservation,
    ) -> Result<CompliantHoldDisposition, CompliantHoldPrepareError> {
        match self.phase {
            ControllerPhase::FollowingExpression {
                quiescent_since,
                contact_armed,
            } => {
                let expression_target_changed = expression_target != self.committed_target;
                self.committed_target = expression_target;
                if !expression_quiet || expression_target_changed {
                    self.phase = following_expression_unarmed();
                    return Ok(CompliantHoldDisposition::FollowingExpression);
                }
                if !contact_armed {
                    let settled =
                        inside_release_band(self.config, self.committed_target, observation)
                            && HeadJoint::ALL
                                .into_iter()
                                .all(|joint| !observation.is_moving(joint));
                    let quiescent_since = if settled {
                        quiescent_since.or(Some(now))
                    } else {
                        None
                    };
                    let contact_armed = quiescent_since.is_some_and(|started| {
                        now.checked_duration_since(started)
                            .is_some_and(|elapsed| elapsed >= self.config.contact_arm_dwell())
                    });
                    self.phase = ControllerPhase::FollowingExpression {
                        quiescent_since,
                        contact_armed,
                    };
                    return Ok(CompliantHoldDisposition::FollowingExpression);
                }
                let directions =
                    contact_directions(self.config, self.committed_target, observation);
                if directions == [0; JOINT_COUNT] {
                    return Ok(CompliantHoldDisposition::FollowingExpression);
                }
                let candidate = ContactCandidate {
                    return_target: self.committed_target,
                    directions,
                    consecutive_samples: 1,
                };
                if self.config.contact_acquisition_samples() == 1 {
                    return self.enter_yield(candidate.return_target, observation);
                }
                self.phase = ControllerPhase::Confirming(candidate);
                Ok(CompliantHoldDisposition::ContactCandidate {
                    consecutive_samples: 1,
                })
            }
            ControllerPhase::Confirming(candidate) => {
                if !expression_quiet || expression_target != candidate.return_target {
                    self.phase = following_expression_unarmed();
                    self.committed_target = expression_target;
                    return Ok(CompliantHoldDisposition::FollowingExpression);
                }
                let directions =
                    contact_directions(self.config, self.committed_target, observation);
                if !directions_continue(candidate.directions, directions) {
                    self.phase = following_expression_unarmed();
                    return Ok(CompliantHoldDisposition::FollowingExpression);
                }
                let consecutive_samples = candidate.consecutive_samples.saturating_add(1);
                if consecutive_samples >= self.config.contact_acquisition_samples() {
                    return self.enter_yield(candidate.return_target, observation);
                }
                self.phase = ControllerPhase::Confirming(ContactCandidate {
                    consecutive_samples,
                    ..candidate
                });
                Ok(CompliantHoldDisposition::ContactCandidate {
                    consecutive_samples,
                })
            }
            ControllerPhase::Yielding {
                return_target,
                quiet_since,
            } => {
                let released = inside_release_band(self.config, self.committed_target, observation)
                    && HeadJoint::ALL
                        .into_iter()
                        .all(|joint| !observation.is_moving(joint));
                let quiet_since = if released {
                    quiet_since.or(Some(now))
                } else {
                    None
                };
                if let Some(started) = quiet_since
                    && now
                        .checked_duration_since(started)
                        .is_some_and(|elapsed| elapsed >= self.config.release_dwell())
                {
                    self.phase = ControllerPhase::Recovering {
                        return_target,
                        recovery_start: self.committed_target,
                        started_at: now,
                        reacquisition: None,
                    };
                    return Ok(CompliantHoldDisposition::Recovering {
                        progress_millionths: 0,
                        command_step_limited: [false; JOINT_COUNT],
                    });
                }
                self.phase = ControllerPhase::Yielding {
                    return_target,
                    quiet_since,
                };
                if released {
                    Ok(CompliantHoldDisposition::ReleaseDwell)
                } else {
                    let (target, envelope_limited, command_step_limited) = yield_target(
                        self.config,
                        return_target,
                        self.committed_target,
                        observation,
                    );
                    self.committed_target = target;
                    Ok(CompliantHoldDisposition::Yielding {
                        envelope_limited,
                        command_step_limited,
                    })
                }
            }
            ControllerPhase::Recovering {
                return_target,
                recovery_start,
                started_at,
                reacquisition,
            } => {
                let directions =
                    contact_directions(self.config, self.committed_target, observation);
                let reacquisition = match (reacquisition, directions == [0; JOINT_COUNT]) {
                    (_, true) => None,
                    (Some(candidate), false)
                        if directions_continue(candidate.directions, directions) =>
                    {
                        Some(ContactCandidate {
                            consecutive_samples: candidate.consecutive_samples.saturating_add(1),
                            ..candidate
                        })
                    }
                    _ => Some(ContactCandidate {
                        return_target,
                        directions,
                        consecutive_samples: 1,
                    }),
                };
                if let Some(candidate) = reacquisition
                    && candidate.consecutive_samples >= self.config.contact_acquisition_samples()
                {
                    return self.enter_yield(return_target, observation);
                }
                let elapsed = now
                    .checked_duration_since(started_at)
                    .expect("observation time admission prevents clock regression");
                let progress = minimum_jerk_progress(elapsed, self.config.recovery_duration());
                let desired = interpolate_pose(recovery_start, return_target, progress);
                let (target, command_step_limited) =
                    command_step_target(self.config, self.committed_target, desired);
                self.committed_target = target;
                if elapsed >= self.config.recovery_duration() && target == return_target {
                    self.phase = following_expression_unarmed();
                    return Ok(CompliantHoldDisposition::ReturnedToExpression);
                }
                self.phase = ControllerPhase::Recovering {
                    return_target,
                    recovery_start,
                    started_at,
                    reacquisition,
                };
                Ok(CompliantHoldDisposition::Recovering {
                    progress_millionths: u32::try_from(progress)
                        .expect("fixed interpolation scale fits u32"),
                    command_step_limited,
                })
            }
        }
    }

    fn enter_yield(
        &mut self,
        return_target: ExactHeadTargetPose,
        observation: CompliantHeadObservation,
    ) -> Result<CompliantHoldDisposition, CompliantHoldPrepareError> {
        let (target, envelope_limited, command_step_limited) = yield_target(
            self.config,
            return_target,
            self.committed_target,
            observation,
        );
        self.committed_target = target;
        self.phase = ControllerPhase::Yielding {
            return_target,
            quiet_since: None,
        };
        Ok(CompliantHoldDisposition::Yielding {
            envelope_limited,
            command_step_limited,
        })
    }
}

fn admit_target(
    config: HeadCompliantHoldConfig,
    target: ExactHeadTargetPose,
) -> Result<(), CompliantHoldPrepareError> {
    for joint in HeadJoint::ALL {
        let policy = config.joint(joint);
        let position = target.position(joint);
        if !policy.contains(position) {
            return Err(CompliantHoldPrepareError::ExpressionTargetOutsideEnvelope {
                joint,
                target: position,
                minimum: policy.minimum(),
                maximum: policy.maximum(),
            });
        }
    }
    Ok(())
}

fn signed_error(actual: PositionTicks, reference: PositionTicks) -> i32 {
    i32::from(actual.get()) - i32::from(reference.get())
}

fn contact_directions(
    config: HeadCompliantHoldConfig,
    reference: ExactHeadTargetPose,
    observation: CompliantHeadObservation,
) -> [i8; JOINT_COUNT] {
    HeadJoint::ALL.map(|joint| {
        let error = signed_error(observation.position(joint), reference.position(joint));
        if error.unsigned_abs() >= u32::from(config.joint(joint).contact_entry_error_ticks()) {
            error.signum() as i8
        } else {
            0
        }
    })
}

fn directions_continue(previous: [i8; JOINT_COUNT], actual: [i8; JOINT_COUNT]) -> bool {
    previous
        .into_iter()
        .zip(actual)
        .any(|(previous, actual)| previous != 0 && previous == actual)
        && previous
            .into_iter()
            .zip(actual)
            .all(|(previous, actual)| previous == 0 || actual == 0 || previous == actual)
}

fn inside_release_band(
    config: HeadCompliantHoldConfig,
    command: ExactHeadTargetPose,
    observation: CompliantHeadObservation,
) -> bool {
    HeadJoint::ALL.into_iter().all(|joint| {
        observation
            .position(joint)
            .get()
            .abs_diff(command.position(joint).get())
            <= config.joint(joint).contact_release_error_ticks()
    })
}

fn yield_target(
    config: HeadCompliantHoldConfig,
    return_target: ExactHeadTargetPose,
    current_target: ExactHeadTargetPose,
    observation: CompliantHeadObservation,
) -> (
    ExactHeadTargetPose,
    [bool; JOINT_COUNT],
    [bool; JOINT_COUNT],
) {
    let mut positions = return_target.positions();
    let mut limited = [false; JOINT_COUNT];
    for joint in HeadJoint::ALL {
        let index = joint_index(joint);
        let policy = config.joint(joint);
        let displacement = signed_error(observation.position(joint), return_target.position(joint));
        let scaled = div_round_nearest(
            i64::from(displacement) * i64::from(config.follow_permille()),
            1_000,
        );
        let maximum_yield = i64::from(policy.maximum_yield_ticks());
        let yield_limited = scaled.clamp(-maximum_yield, maximum_yield);
        limited[index] |= yield_limited != scaled;
        let raw = i64::from(return_target.position(joint).get()) + yield_limited;
        let envelope_limited = raw.clamp(
            i64::from(policy.minimum().get()),
            i64::from(policy.maximum().get()),
        );
        limited[index] |= envelope_limited != raw;
        positions[index] = PositionTicks::try_new(
            u16::try_from(envelope_limited).expect("admitted encoder envelope fits u16"),
        )
        .expect("admitted encoder envelope is inside protocol range");
    }
    let desired =
        ExactHeadTargetPose::from_positions(positions[0], positions[1], positions[2], positions[3]);
    let (target, command_step_limited) = command_step_target(config, current_target, desired);
    (target, limited, command_step_limited)
}

fn command_step_target(
    config: HeadCompliantHoldConfig,
    current: ExactHeadTargetPose,
    desired: ExactHeadTargetPose,
) -> (ExactHeadTargetPose, [bool; JOINT_COUNT]) {
    let mut positions = current.positions();
    let mut limited = [false; JOINT_COUNT];
    for joint in HeadJoint::ALL {
        let index = joint_index(joint);
        let current = i64::from(current.position(joint).get());
        let desired = i64::from(desired.position(joint).get());
        let maximum = i64::from(config.joint(joint).maximum_command_step_ticks());
        let delta = desired - current;
        let bounded = delta.clamp(-maximum, maximum);
        limited[index] = bounded != delta;
        positions[index] = PositionTicks::try_new(
            u16::try_from(current + bounded)
                .expect("bounded step between admitted positions stays inside u16"),
        )
        .expect("bounded step between admitted positions stays in encoder domain");
    }
    (
        ExactHeadTargetPose::from_positions(positions[0], positions[1], positions[2], positions[3]),
        limited,
    )
}

fn div_round_nearest(numerator: i64, denominator: i64) -> i64 {
    debug_assert!(denominator > 0);
    if numerator >= 0 {
        (numerator + denominator / 2) / denominator
    } else {
        (numerator - denominator / 2) / denominator
    }
}

fn checked_time_add(time: MonotonicTime, duration: Duration) -> Option<MonotonicTime> {
    time.duration_since_origin()
        .checked_add(duration)
        .map(MonotonicTime::from_duration_since_origin)
}

/// Quintic minimum-jerk progress in fixed millionths.
fn minimum_jerk_progress(elapsed: Duration, total: Duration) -> u128 {
    if elapsed >= total {
        return INTERPOLATION_SCALE;
    }
    let elapsed_ns = elapsed.as_nanos();
    let total_ns = total.as_nanos();
    let u = elapsed_ns.saturating_mul(INTERPOLATION_SCALE) / total_ns;
    let u2 = u * u;
    let u3 = u2 * u;
    let u4 = u3 * u;
    let u5 = u4 * u;
    let scale2 = INTERPOLATION_SCALE * INTERPOLATION_SCALE;
    let scale4 = scale2 * scale2;
    (6 * u5 + 10 * u3 * scale2 - 15 * u4 * INTERPOLATION_SCALE) / scale4
}

fn interpolate_pose(
    start: ExactHeadTargetPose,
    end: ExactHeadTargetPose,
    progress_millionths: u128,
) -> ExactHeadTargetPose {
    let positions = HeadJoint::ALL.map(|joint| {
        let start = i64::from(start.position(joint).get());
        let delta = i64::from(end.position(joint).get()) - start;
        let scaled = div_round_nearest(
            delta * i64::try_from(progress_millionths).expect("progress fits i64"),
            i64::try_from(INTERPOLATION_SCALE).expect("scale fits i64"),
        );
        PositionTicks::try_new(
            u16::try_from(start + scaled).expect("interpolation stays between admitted endpoints"),
        )
        .expect("interpolation stays inside protocol encoder range")
    });
    ExactHeadTargetPose::from_positions(positions[0], positions[1], positions[2], positions[3])
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiko_head_protocol::{HeadTorqueLimits, TorqueLimitPermille};

    fn at(milliseconds: u64) -> MonotonicTime {
        MonotonicTime::from_duration_since_origin(Duration::from_millis(milliseconds))
    }

    fn pose(values: [u16; 4]) -> ExactHeadTargetPose {
        ExactHeadTargetPose::try_from_ticks(values).unwrap()
    }

    fn joint(minimum: u16, maximum: u16, maximum_yield: u16) -> CompliantJointPolicy {
        CompliantJointPolicy::try_new(
            PositionTicks::try_new(minimum).unwrap(),
            PositionTicks::try_new(maximum).unwrap(),
            20,
            6,
            maximum_yield,
            100,
            100,
        )
        .unwrap()
    }

    fn config(acquisition: u8) -> HeadCompliantHoldConfig {
        HeadCompliantHoldConfig::try_new(
            joint(2_064, 2_284, 80),
            joint(2_390, 2_750, 100),
            joint(1_157, 2_117, 180),
            joint(2_887, 3_207, 90),
            HeadTorqueLimits::new(
                TorqueLimitPermille::try_new(600).unwrap(),
                TorqueLimitPermille::try_new(400).unwrap(),
                TorqueLimitPermille::try_new(400).unwrap(),
                TorqueLimitPermille::try_new(400).unwrap(),
            ),
            Duration::from_millis(10),
            Duration::from_millis(10),
            Duration::from_millis(10),
            Duration::from_millis(30),
            Duration::from_millis(10),
            acquisition,
            Duration::from_millis(100),
            Duration::from_millis(1_000),
            800,
        )
        .unwrap()
    }

    fn observation(at_ms: u64, values: [u16; 4], moving: bool) -> CompliantHeadObservation {
        CompliantHeadObservation::from_parts(at(at_ms), values, [moving; 4])
    }

    fn prepare_commit(
        controller: &mut HeadCompliantHoldController,
        at_ms: u64,
        expression: ExactHeadTargetPose,
        quiet: bool,
        positions: [u16; 4],
        moving: bool,
    ) -> CompliantHoldCommitReceipt {
        let prepared = controller
            .prepare(
                at(at_ms),
                expression,
                quiet,
                observation(at_ms, positions, moving),
            )
            .unwrap();
        controller.commit(prepared).unwrap()
    }

    fn arm_at_natural(controller: &mut HeadCompliantHoldController, natural: ExactHeadTargetPose) {
        let positions = natural.positions().map(PositionTicks::get);
        let first = prepare_commit(controller, 0, natural, true, positions, false);
        assert_eq!(first.state(), CompliantHoldState::FollowingExpression);
        let armed = prepare_commit(controller, 10, natural, true, positions, false);
        assert_eq!(armed.state(), CompliantHoldState::FollowingExpression);
    }

    #[test]
    fn config_rejects_hysteresis_without_an_inner_release_band() {
        assert_eq!(
            CompliantJointPolicy::try_new(
                PositionTicks::try_new(1_000).unwrap(),
                PositionTicks::try_new(1_200).unwrap(),
                20,
                20,
                40,
                40,
                40,
            ),
            Err(CompliantJointPolicyError::ReleaseNotInsideEntry {
                release_ticks: 20,
                entry_ticks: 20,
            })
        );
    }

    #[test]
    fn touch_observation_may_cross_a_command_edge_only_by_reviewed_yield() {
        let natural = pose([2_174, 2_570, 1_637, 3_047]);
        let policy = config(1).joint(HeadJoint::Bow);
        assert_eq!(policy.minimum().get(), 2_064);
        assert_eq!(policy.observation_minimum().get(), 1_984);

        let mut exact_edge =
            HeadCompliantHoldController::try_new(config(1), natural, at(0)).unwrap();
        exact_edge
            .prepare(
                at(0),
                natural,
                true,
                observation(0, [1_984, 2_570, 1_637, 3_047], false),
            )
            .expect("the exact command edge plus reviewed yield is observable");

        let mut beyond = HeadCompliantHoldController::try_new(config(1), natural, at(0)).unwrap();
        assert_eq!(
            beyond
                .prepare(
                    at(0),
                    natural,
                    true,
                    observation(0, [1_983, 2_570, 1_637, 3_047], false),
                )
                .expect_err("one tick beyond reviewed yield must fault"),
            CompliantHoldPrepareError::FaultLatched(
                CompliantHoldFault::ObservationOutsideEnvelope {
                    joint: HeadJoint::Bow,
                    observed: PositionTicks::try_new(1_983).unwrap(),
                    minimum: PositionTicks::try_new(1_984).unwrap(),
                    maximum: PositionTicks::try_new(2_364).unwrap(),
                }
            )
        );
    }

    #[test]
    fn expression_command_cannot_use_the_observation_excursion() {
        let natural = pose([2_174, 2_570, 1_637, 3_047]);
        let unsafe_expression = pose([2_063, 2_570, 1_637, 3_047]);
        let mut controller =
            HeadCompliantHoldController::try_new(config(1), natural, at(0)).unwrap();
        assert_eq!(
            controller
                .prepare(
                    at(0),
                    unsafe_expression,
                    true,
                    observation(0, [2_063, 2_570, 1_637, 3_047], false),
                )
                .expect_err("physical observation latitude must never widen commands"),
            CompliantHoldPrepareError::ExpressionTargetOutsideEnvelope {
                joint: HeadJoint::Bow,
                target: PositionTicks::try_new(2_063).unwrap(),
                minimum: PositionTicks::try_new(2_064).unwrap(),
                maximum: PositionTicks::try_new(2_284).unwrap(),
            }
        );
    }

    #[test]
    fn two_consistent_samples_are_required_before_yield() {
        let natural = pose([2_174, 2_570, 1_637, 3_047]);
        let mut controller =
            HeadCompliantHoldController::try_new(config(2), natural, at(0)).unwrap();
        arm_at_natural(&mut controller, natural);
        let first = prepare_commit(
            &mut controller,
            20,
            natural,
            true,
            [2_204, 2_570, 1_637, 3_047],
            true,
        );
        assert_eq!(
            first.disposition(),
            CompliantHoldDisposition::ContactCandidate {
                consecutive_samples: 1
            }
        );
        assert_eq!(first.committed_target(), natural);

        let second = prepare_commit(
            &mut controller,
            30,
            natural,
            true,
            [2_214, 2_570, 1_637, 3_047],
            true,
        );
        assert_eq!(second.state(), CompliantHoldState::Yielding);
        assert_eq!(
            second.committed_target().position(HeadJoint::Bow).get(),
            2_206
        );
    }

    #[test]
    fn commanded_expression_motion_cannot_be_misclassified_as_contact() {
        let natural = pose([2_174, 2_570, 1_637, 3_047]);
        let expression = pose([2_200, 2_550, 1_600, 3_080]);
        let mut controller =
            HeadCompliantHoldController::try_new(config(2), natural, at(0)).unwrap();
        let receipt = prepare_commit(
            &mut controller,
            10,
            expression,
            false,
            natural.positions().map(PositionTicks::get),
            true,
        );
        assert_eq!(receipt.state(), CompliantHoldState::FollowingExpression);
        assert_eq!(receipt.committed_target(), expression);
    }

    #[test]
    fn contact_cannot_arm_until_the_head_is_stationary_and_inside_release_band() {
        let natural = pose([2_174, 2_570, 1_637, 3_047]);
        let mut controller =
            HeadCompliantHoldController::try_new(config(1), natural, at(0)).unwrap();
        for (time, moving) in [(0, true), (10, false)] {
            let receipt = prepare_commit(
                &mut controller,
                time,
                natural,
                true,
                [2_204, 2_570, 1_637, 3_047],
                moving,
            );
            assert_eq!(receipt.state(), CompliantHoldState::FollowingExpression);
            assert_eq!(receipt.committed_target(), natural);
        }

        let positions = natural.positions().map(PositionTicks::get);
        prepare_commit(&mut controller, 20, natural, true, positions, false);
        prepare_commit(&mut controller, 30, natural, true, positions, false);
        let contact = prepare_commit(
            &mut controller,
            40,
            natural,
            true,
            [2_204, 2_570, 1_637, 3_047],
            true,
        );
        assert_eq!(contact.state(), CompliantHoldState::Yielding);
    }

    #[test]
    fn yield_is_bounded_and_reports_limiting() {
        let natural = pose([2_174, 2_570, 1_637, 3_047]);
        let mut controller =
            HeadCompliantHoldController::try_new(config(1), natural, at(0)).unwrap();
        arm_at_natural(&mut controller, natural);
        let receipt = prepare_commit(
            &mut controller,
            20,
            natural,
            true,
            [2_274, 2_470, 1_737, 3_147],
            true,
        );
        assert_eq!(
            receipt.committed_target().positions(),
            [
                PositionTicks::try_new(2_254).unwrap(),
                PositionTicks::try_new(2_490).unwrap(),
                PositionTicks::try_new(1_717).unwrap(),
                PositionTicks::try_new(3_127).unwrap(),
            ]
        );
        assert_eq!(
            receipt.disposition(),
            CompliantHoldDisposition::Yielding {
                envelope_limited: [false; 4],
                command_step_limited: [false; 4],
            }
        );
    }

    #[test]
    fn stationary_release_dwells_then_recovers_with_exact_minimum_jerk_endpoints() {
        let natural = pose([2_174, 2_570, 1_637, 3_047]);
        let mut controller =
            HeadCompliantHoldController::try_new(config(1), natural, at(0)).unwrap();
        arm_at_natural(&mut controller, natural);
        let yielded = prepare_commit(
            &mut controller,
            20,
            natural,
            true,
            [2_224, 2_570, 1_637, 3_047],
            true,
        )
        .committed_target();
        assert_eq!(yielded.position(HeadJoint::Bow).get(), 2_214);

        let dwell = prepare_commit(
            &mut controller,
            30,
            natural,
            true,
            yielded.positions().map(PositionTicks::get),
            false,
        );
        assert_eq!(dwell.state(), CompliantHoldState::ReleaseDwell);
        let recovery_start = prepare_commit(
            &mut controller,
            130,
            natural,
            true,
            yielded.positions().map(PositionTicks::get),
            false,
        );
        assert_eq!(
            recovery_start.disposition(),
            CompliantHoldDisposition::Recovering {
                progress_millionths: 0,
                command_step_limited: [false; 4],
            }
        );
        assert_eq!(recovery_start.committed_target(), yielded);

        let midpoint = prepare_commit(
            &mut controller,
            630,
            natural,
            true,
            yielded.positions().map(PositionTicks::get),
            false,
        );
        assert_eq!(
            midpoint.disposition(),
            CompliantHoldDisposition::Recovering {
                progress_millionths: 500_000,
                command_step_limited: [false; 4],
            }
        );
        assert_eq!(
            midpoint.committed_target().position(HeadJoint::Bow).get(),
            2_194
        );

        let complete = prepare_commit(
            &mut controller,
            1_130,
            natural,
            true,
            midpoint
                .committed_target()
                .positions()
                .map(PositionTicks::get),
            false,
        );
        assert_eq!(
            complete.disposition(),
            CompliantHoldDisposition::ReturnedToExpression
        );
        assert_eq!(complete.committed_target(), natural);
    }

    #[test]
    fn continued_hand_resistance_reacquires_instead_of_fighting() {
        let natural = pose([2_174, 2_570, 1_637, 3_047]);
        let mut controller =
            HeadCompliantHoldController::try_new(config(1), natural, at(0)).unwrap();
        arm_at_natural(&mut controller, natural);
        let yielded = prepare_commit(
            &mut controller,
            20,
            natural,
            true,
            [2_224, 2_570, 1_637, 3_047],
            true,
        )
        .committed_target();
        prepare_commit(
            &mut controller,
            30,
            natural,
            true,
            yielded.positions().map(PositionTicks::get),
            false,
        );
        prepare_commit(
            &mut controller,
            130,
            natural,
            true,
            yielded.positions().map(PositionTicks::get),
            false,
        );
        let resisted = prepare_commit(
            &mut controller,
            230,
            natural,
            true,
            [2_234, 2_570, 1_637, 3_047],
            true,
        );
        assert_eq!(resisted.state(), CompliantHoldState::Yielding);
        assert_eq!(
            resisted.committed_target().position(HeadJoint::Bow).get(),
            2_222
        );
    }

    #[test]
    fn stale_observation_does_not_advance_transactional_state() {
        let natural = pose([2_174, 2_570, 1_637, 3_047]);
        let mut controller =
            HeadCompliantHoldController::try_new(config(2), natural, at(0)).unwrap();
        assert!(matches!(
            controller.prepare(
                at(40),
                natural,
                true,
                observation(10, [2_204, 2_570, 1_637, 3_047], true),
            ),
            Err(CompliantHoldPrepareError::ObservationExpired { .. })
        ));
        assert_eq!(controller.state(), CompliantHoldState::FollowingExpression);
        assert_eq!(controller.committed_target(), natural);
    }

    #[test]
    fn uncertain_application_is_absorbing() {
        let natural = pose([2_174, 2_570, 1_637, 3_047]);
        let mut controller =
            HeadCompliantHoldController::try_new(config(1), natural, at(0)).unwrap();
        arm_at_natural(&mut controller, natural);
        let prepared = controller
            .prepare(
                at(20),
                natural,
                true,
                observation(20, [2_204, 2_570, 1_637, 3_047], true),
            )
            .unwrap();
        controller
            .abort_with_application_uncertain(prepared)
            .unwrap();
        assert_eq!(controller.state(), CompliantHoldState::FaultHeld);
        assert_eq!(
            controller.fault(),
            Some(CompliantHoldFault::ApplicationUncertain)
        );
    }

    #[test]
    fn minimum_jerk_is_monotonic_symmetric_and_endpoint_exact() {
        let total = Duration::from_secs(2);
        assert_eq!(minimum_jerk_progress(Duration::ZERO, total), 0);
        assert_eq!(minimum_jerk_progress(total, total), INTERPOLATION_SCALE);
        let mut previous = 0;
        for millisecond in 0..=2_000 {
            let progress = minimum_jerk_progress(Duration::from_millis(millisecond), total);
            assert!(progress >= previous);
            previous = progress;
        }
        for millisecond in [1, 25, 100, 333, 750, 1_000] {
            let forward = minimum_jerk_progress(Duration::from_millis(millisecond), total);
            let reverse = minimum_jerk_progress(Duration::from_millis(2_000 - millisecond), total);
            assert!(forward.abs_diff(INTERPOLATION_SCALE - reverse) <= 1);
        }
    }

    #[test]
    fn interpolation_never_overshoots_either_endpoint() {
        let start = pose([2_250, 2_470, 1_800, 3_120]);
        let end = pose([2_174, 2_570, 1_637, 3_047]);
        for progress in (0..=1_000_000).step_by(997) {
            let result = interpolate_pose(start, end, progress);
            for joint in HeadJoint::ALL {
                let low = start.position(joint).min(end.position(joint));
                let high = start.position(joint).max(end.position(joint));
                assert!(result.position(joint) >= low);
                assert!(result.position(joint) <= high);
            }
        }
    }

    #[test]
    fn command_step_bound_is_monotonic_and_eventually_reaches_every_axis() {
        let config = config(1);
        let mut current = pose([2_174, 2_570, 1_637, 3_047]);
        let desired = pose([2_284, 2_390, 2_117, 3_207]);
        let mut iterations = 0;
        while current != desired {
            let previous = current;
            let (next, limited) = command_step_target(config, current, desired);
            for joint in HeadJoint::ALL {
                let step = next
                    .position(joint)
                    .get()
                    .abs_diff(previous.position(joint).get());
                assert!(step <= config.joint(joint).maximum_command_step_ticks());
                let before = previous
                    .position(joint)
                    .get()
                    .abs_diff(desired.position(joint).get());
                let after = next
                    .position(joint)
                    .get()
                    .abs_diff(desired.position(joint).get());
                assert!(after <= before);
                assert_eq!(
                    limited[joint_index(joint)],
                    before > config.joint(joint).maximum_command_step_ticks()
                );
            }
            current = next;
            iterations += 1;
            assert!(iterations <= 5, "bounded convergence must make progress");
        }
    }
}
