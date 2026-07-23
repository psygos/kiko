//! Pure motor-domain logic for the STM32 controller.
//!
//! This module deliberately does not touch a timer, GPIO, or interrupt.  Its
//! directives describe the ordering that a target-specific executor must
//! implement.  An `Applied` result therefore proves only the pure transition
//! contract; it is not evidence that a physical driver or wheel responded.

use core::num::{NonZeroU8, NonZeroU32};

use robot_protocol::{
    ControllerUptimeMsWrapping, PwmPercent, PwmPercentError,
    v2::{
        ATTENDED_WHEEL_ON_COMMISSIONING_MAX_COMMAND_STEP_PERCENT, ActuatorConfigFingerprint,
        ControllerDeadlineMsWrapping, DeadlineRelation,
        MAX_ATTENDED_WHEEL_ON_COMMISSIONING_PWM_PERCENT,
        MAX_OPERATOR_SUPERVISED_FOUR_PWM_PWM_PERCENT,
        OPERATOR_SUPERVISED_FOUR_PWM_MAX_COMMAND_STEP_PERCENT,
    },
};

pub const MAX_UNAMBIGUOUS_WRAPPING_TICKS: u32 = (1_u32 << 31) - 1;
pub const MAX_PWM_STEP_PERCENT: u8 = 200;
/// Per-command delta bound for the provisional four-PWM profile.
///
/// This is intentionally a command-step limit, not an experimentally measured
/// PWM-per-second, wheel-acceleration, or velocity bound.
pub const PROVISIONAL_FOUR_PWM_MAX_COMMAND_STEP_PERCENT: u8 =
    OPERATOR_SUPERVISED_FOUR_PWM_MAX_COMMAND_STEP_PERCENT;
pub const ATTENDED_WHEEL_ON_MAX_COMMAND_STEP_PERCENT: u8 =
    ATTENDED_WHEEL_ON_COMMISSIONING_MAX_COMMAND_STEP_PERCENT;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DurationMs(NonZeroU32);

impl DurationMs {
    pub fn try_new(value: u32) -> Result<Self, DurationMsError> {
        let value = NonZeroU32::new(value).ok_or(DurationMsError::Zero)?;
        if value.get() > MAX_UNAMBIGUOUS_WRAPPING_TICKS {
            return Err(DurationMsError::AboveUnambiguousHalfRange {
                value: value.get(),
                maximum: MAX_UNAMBIGUOUS_WRAPPING_TICKS,
            });
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DurationMsError {
    Zero,
    AboveUnambiguousHalfRange { value: u32, maximum: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WrappingDeadline {
    started_at: ControllerUptimeMsWrapping,
    duration: DurationMs,
    expires_at: ControllerDeadlineMsWrapping,
}

impl WrappingDeadline {
    pub const fn after(started_at: ControllerUptimeMsWrapping, duration: DurationMs) -> Self {
        Self {
            started_at,
            duration,
            expires_at: ControllerDeadlineMsWrapping::new(
                started_at.get().wrapping_add(duration.get()),
            ),
        }
    }

    pub const fn started_at(self) -> ControllerUptimeMsWrapping {
        self.started_at
    }

    pub const fn duration(self) -> DurationMs {
        self.duration
    }

    pub const fn expires_at(self) -> ControllerDeadlineMsWrapping {
        self.expires_at
    }

    pub const fn status_at(self, now: ControllerUptimeMsWrapping) -> DeadlineStatus {
        let elapsed = now.get().wrapping_sub(self.started_at.get());
        if elapsed > MAX_UNAMBIGUOUS_WRAPPING_TICKS {
            return DeadlineStatus::ObservationGap;
        }
        match self.expires_at.relation_to(now) {
            DeadlineRelation::Future { .. } => DeadlineStatus::Pending,
            DeadlineRelation::Expired => DeadlineStatus::Reached,
            DeadlineRelation::AmbiguousHalfRange => DeadlineStatus::ObservationGap,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeadlineStatus {
    Pending,
    Reached,
    /// The observation appears to precede the start or follows an unobserved
    /// half-range of the wrapping clock; elapsed time is not interpretable.
    ObservationGap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PwmPair {
    left: PwmPercent,
    right: PwmPercent,
}

impl PwmPair {
    pub const STOP: Self = Self {
        left: PwmPercent::ZERO,
        right: PwmPercent::ZERO,
    };

    pub const fn from_validated(left: PwmPercent, right: PwmPercent) -> Self {
        Self { left, right }
    }

    pub const fn left(self) -> PwmPercent {
        self.left
    }

    pub const fn right(self) -> PwmPercent {
        self.right
    }

    pub const fn is_stop(self) -> bool {
        self.left.get() == 0 && self.right.get() == 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelDrive {
    Disabled,
    Forward(NonZeroU8),
    Reverse(NonZeroU8),
}

impl WheelDrive {
    pub fn from_pwm(pwm: PwmPercent) -> Self {
        let value = pwm.get();
        let Some(magnitude) = NonZeroU8::new(value.unsigned_abs()) else {
            return Self::Disabled;
        };
        if value > 0 {
            Self::Forward(magnitude)
        } else {
            Self::Reverse(magnitude)
        }
    }

    pub const fn direction(self) -> WheelDirection {
        match self {
            Self::Disabled => WheelDirection::Disabled,
            Self::Forward(_) => WheelDirection::Forward,
            Self::Reverse(_) => WheelDirection::Reverse,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelDirection {
    Disabled,
    Forward,
    Reverse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DriveOutput {
    left: WheelDrive,
    right: WheelDrive,
}

impl DriveOutput {
    pub const SAFE_NEUTRAL: Self = Self {
        left: WheelDrive::Disabled,
        right: WheelDrive::Disabled,
    };

    pub fn from_pwm(pwm: PwmPair) -> Self {
        Self {
            left: WheelDrive::from_pwm(pwm.left()),
            right: WheelDrive::from_pwm(pwm.right()),
        }
    }

    pub const fn left(self) -> WheelDrive {
        self.left
    }

    pub const fn right(self) -> WheelDrive {
        self.right
    }

    pub const fn is_safe_neutral(self) -> bool {
        matches!(self.left, WheelDrive::Disabled) && matches!(self.right, WheelDrive::Disabled)
    }
}

/// Target-portable effects.  `WheelDrive` makes enabling both directions for
/// one wheel unrepresentable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MotorDirective {
    Hold,
    DisableAndZero,
    PreloadWhileDisabled(DriveOutput),
    UpdateEnabled(DriveOutput),
    EnablePreloaded(DriveOutput),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhysicalStopSemantics {
    /// Source has not established whether disabled/both-low means electrical
    /// coast, brake, or another driver-specific state.
    Unverified,
}

/// Compile-time contract for optional wheel-observation hardware.
///
/// `QuadratureEncoderTimers` means only that a reviewed target adapter
/// configured the two timer inputs. It is not evidence that encoders are
/// installed, calibrated, or physically responding. Kiko's canonical
/// encoderless profile uses [`Self::Absent`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservationalOdometryContract {
    Absent,
    QuadratureEncoderTimers,
}

impl ObservationalOdometryContract {
    pub const fn configures_quadrature_inputs(self) -> bool {
        matches!(self, Self::QuadratureEncoderTimers)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PwmRange {
    min: PwmPercent,
    max: PwmPercent,
}

impl PwmRange {
    pub fn try_new(min: PwmPercent, max: PwmPercent) -> Result<Self, PwmRangeError> {
        if min.get() > 0 || max.get() < 0 || min.get() > max.get() {
            return Err(PwmRangeError::MustBeOrderedAndContainZero {
                min: min.get(),
                max: max.get(),
            });
        }
        Ok(Self { min, max })
    }

    pub const fn contains(self, value: PwmPercent) -> bool {
        self.min.get() <= value.get() && value.get() <= self.max.get()
    }

    pub const fn min(self) -> PwmPercent {
        self.min
    }

    pub const fn max(self) -> PwmPercent {
        self.max
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PwmRangeError {
    MustBeOrderedAndContainZero { min: i8, max: i8 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PwmStepLimit {
    left_percent: NonZeroU8,
    right_percent: NonZeroU8,
}

impl PwmStepLimit {
    pub fn try_new(left_percent: u8, right_percent: u8) -> Result<Self, PwmStepLimitError> {
        let left_percent = NonZeroU8::new(left_percent).ok_or(PwmStepLimitError::ZeroLeft)?;
        let right_percent = NonZeroU8::new(right_percent).ok_or(PwmStepLimitError::ZeroRight)?;
        if left_percent.get() > MAX_PWM_STEP_PERCENT {
            return Err(PwmStepLimitError::AboveMaximumLeft {
                value: left_percent.get(),
                maximum: MAX_PWM_STEP_PERCENT,
            });
        }
        if right_percent.get() > MAX_PWM_STEP_PERCENT {
            return Err(PwmStepLimitError::AboveMaximumRight {
                value: right_percent.get(),
                maximum: MAX_PWM_STEP_PERCENT,
            });
        }
        Ok(Self {
            left_percent,
            right_percent,
        })
    }

    pub const fn left_percent(self) -> u8 {
        self.left_percent.get()
    }

    pub const fn right_percent(self) -> u8 {
        self.right_percent.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PwmStepLimitError {
    ZeroLeft,
    ZeroRight,
    AboveMaximumLeft { value: u8, maximum: u8 },
    AboveMaximumRight { value: u8, maximum: u8 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActuatorEnvelope {
    Unvalidated,
    Validated {
        fingerprint: ActuatorConfigFingerprint,
        left: PwmRange,
        right: PwmRange,
        maximum_step: PwmStepLimit,
    },
    OperatorSupervisedFourPwmCandidate {
        fingerprint: ActuatorConfigFingerprint,
        left: PwmRange,
        right: PwmRange,
        maximum_step: PwmStepLimit,
    },
    AttendedWheelOnCommissioning {
        fingerprint: ActuatorConfigFingerprint,
        left: PwmRange,
        right: PwmRange,
        maximum_step: PwmStepLimit,
    },
}

impl ActuatorEnvelope {
    pub const fn unvalidated() -> Self {
        Self::Unvalidated
    }

    pub const fn validated(
        fingerprint: ActuatorConfigFingerprint,
        left: PwmRange,
        right: PwmRange,
        maximum_step: PwmStepLimit,
    ) -> Self {
        Self::Validated {
            fingerprint,
            left,
            right,
            maximum_step,
        }
    }

    pub const fn physical_stop_semantics(self) -> PhysicalStopSemantics {
        PhysicalStopSemantics::Unverified
    }

    pub const fn fingerprint(self) -> Option<ActuatorConfigFingerprint> {
        match self {
            Self::Unvalidated => None,
            Self::Validated { fingerprint, .. }
            | Self::OperatorSupervisedFourPwmCandidate { fingerprint, .. }
            | Self::AttendedWheelOnCommissioning { fingerprint, .. } => Some(fingerprint),
        }
    }

    pub const fn is_operator_supervised_four_pwm_candidate(self) -> bool {
        matches!(self, Self::OperatorSupervisedFourPwmCandidate { .. })
    }

    pub const fn is_attended_wheel_on_commissioning(self) -> bool {
        matches!(self, Self::AttendedWheelOnCommissioning { .. })
    }

    /// Validate a requested transition without clamping.  An explicit stop
    /// bypasses validation and slew so every configuration can fail closed.
    pub fn validate_transition(
        self,
        currently_applied: PwmPair,
        requested: PwmPair,
    ) -> Result<(), MotionEnvelopeError> {
        if requested.is_stop() {
            return Ok(());
        }
        let (left, right, maximum_step) = match self {
            Self::Validated {
                left,
                right,
                maximum_step,
                ..
            }
            | Self::OperatorSupervisedFourPwmCandidate {
                left,
                right,
                maximum_step,
                ..
            }
            | Self::AttendedWheelOnCommissioning {
                left,
                right,
                maximum_step,
                ..
            } => (left, right, maximum_step),
            Self::Unvalidated => {
                return Err(MotionEnvelopeError::MotionDisabledUntilValidated);
            }
        };

        if !left.contains(requested.left()) {
            return Err(MotionEnvelopeError::LeftOutsideRange {
                value: requested.left().get(),
                min: left.min().get(),
                max: left.max().get(),
            });
        }
        if !right.contains(requested.right()) {
            return Err(MotionEnvelopeError::RightOutsideRange {
                value: requested.right().get(),
                min: right.min().get(),
                max: right.max().get(),
            });
        }

        let left_delta = absolute_pwm_delta(currently_applied.left(), requested.left());
        if left_delta > u16::from(maximum_step.left_percent()) {
            return Err(MotionEnvelopeError::LeftStepTooLarge {
                delta_percent: left_delta,
                maximum_percent: maximum_step.left_percent(),
            });
        }
        let right_delta = absolute_pwm_delta(currently_applied.right(), requested.right());
        if right_delta > u16::from(maximum_step.right_percent()) {
            return Err(MotionEnvelopeError::RightStepTooLarge {
                delta_percent: right_delta,
                maximum_percent: maximum_step.right_percent(),
            });
        }
        Ok(())
    }
}

/// Exact, bounded command envelope for the explicitly provisional PA0/PA1 +
/// PB4/PB5 profile.
///
/// The type proves only the software command domain: symmetric ±30% timer
/// duty and a 5 percentage-point maximum change per accepted nonzero command.
/// It makes no wheel sign, minimum useful duty, velocity, torque, or physical
/// stop claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProvisionalBoundedFourPwmEnvelope(ActuatorEnvelope);

impl ProvisionalBoundedFourPwmEnvelope {
    pub fn try_new(
        fingerprint: ActuatorConfigFingerprint,
    ) -> Result<Self, ProvisionalFourPwmEnvelopeError> {
        let maximum = PwmPercent::try_new(
            i8::try_from(MAX_OPERATOR_SUPERVISED_FOUR_PWM_PWM_PERCENT)
                .map_err(|_| ProvisionalFourPwmEnvelopeError::ProtocolCapNotRepresentable)?,
        )
        .map_err(ProvisionalFourPwmEnvelopeError::ProtocolCap)?;
        let minimum = PwmPercent::try_new(-maximum.get())
            .map_err(ProvisionalFourPwmEnvelopeError::ProtocolCap)?;
        let left =
            PwmRange::try_new(minimum, maximum).map_err(ProvisionalFourPwmEnvelopeError::Left)?;
        let right =
            PwmRange::try_new(minimum, maximum).map_err(ProvisionalFourPwmEnvelopeError::Right)?;
        let maximum_step = PwmStepLimit::try_new(
            PROVISIONAL_FOUR_PWM_MAX_COMMAND_STEP_PERCENT,
            PROVISIONAL_FOUR_PWM_MAX_COMMAND_STEP_PERCENT,
        )
        .map_err(ProvisionalFourPwmEnvelopeError::Step)?;
        Ok(Self(ActuatorEnvelope::OperatorSupervisedFourPwmCandidate {
            fingerprint,
            left,
            right,
            maximum_step,
        }))
    }

    pub const fn into_actuator_envelope(self) -> ActuatorEnvelope {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProvisionalFourPwmEnvelopeError {
    ProtocolCapNotRepresentable,
    ProtocolCap(PwmPercentError),
    Left(PwmRangeError),
    Right(PwmRangeError),
    Step(PwmStepLimitError),
}

/// Exact electrical command envelope for the separately identified attended
/// wheel-on commissioning image.
///
/// It shares the evidenced four-PWM timer topology with the wheels-off
/// candidate while narrowing the admitted range to ±20%. It does not claim a
/// wheel sign, useful duty, velocity, acceleration, or physical stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttendedWheelOnFourPwmEnvelope(ActuatorEnvelope);

impl AttendedWheelOnFourPwmEnvelope {
    pub fn try_new(
        fingerprint: ActuatorConfigFingerprint,
    ) -> Result<Self, ProvisionalFourPwmEnvelopeError> {
        let maximum = PwmPercent::try_new(
            i8::try_from(MAX_ATTENDED_WHEEL_ON_COMMISSIONING_PWM_PERCENT)
                .map_err(|_| ProvisionalFourPwmEnvelopeError::ProtocolCapNotRepresentable)?,
        )
        .map_err(ProvisionalFourPwmEnvelopeError::ProtocolCap)?;
        let minimum = PwmPercent::try_new(-maximum.get())
            .map_err(ProvisionalFourPwmEnvelopeError::ProtocolCap)?;
        let left =
            PwmRange::try_new(minimum, maximum).map_err(ProvisionalFourPwmEnvelopeError::Left)?;
        let right =
            PwmRange::try_new(minimum, maximum).map_err(ProvisionalFourPwmEnvelopeError::Right)?;
        let maximum_step = PwmStepLimit::try_new(
            ATTENDED_WHEEL_ON_MAX_COMMAND_STEP_PERCENT,
            ATTENDED_WHEEL_ON_MAX_COMMAND_STEP_PERCENT,
        )
        .map_err(ProvisionalFourPwmEnvelopeError::Step)?;
        Ok(Self(ActuatorEnvelope::AttendedWheelOnCommissioning {
            fingerprint,
            left,
            right,
            maximum_step,
        }))
    }

    pub const fn into_actuator_envelope(self) -> ActuatorEnvelope {
        self.0
    }
}

fn absolute_pwm_delta(left: PwmPercent, right: PwmPercent) -> u16 {
    i16::from(left.get()).abs_diff(i16::from(right.get()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MotionEnvelopeError {
    MotionDisabledUntilValidated,
    LeftOutsideRange {
        value: i8,
        min: i8,
        max: i8,
    },
    RightOutsideRange {
        value: i8,
        min: i8,
        max: i8,
    },
    LeftStepTooLarge {
        delta_percent: u16,
        maximum_percent: u8,
    },
    RightStepTooLarge {
        delta_percent: u16,
        maximum_percent: u8,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MotorTiming {
    neutral_hold: DurationMs,
    preload_latch: DurationMs,
}

impl MotorTiming {
    pub const fn new(neutral_hold: DurationMs, preload_latch: DurationMs) -> Self {
        Self {
            neutral_hold,
            preload_latch,
        }
    }

    pub const fn neutral_hold(self) -> DurationMs {
        self.neutral_hold
    }

    pub const fn preload_latch(self) -> DurationMs {
        self.preload_latch
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MotorTransitionPhase {
    Neutralizing,
    PreloadingDisabled,
    UpdatingEnabled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MotorTransition {
    target: PwmPair,
    phase: MotorTransitionPhase,
    phase_deadline: WrappingDeadline,
}

impl MotorTransition {
    pub fn start(
        currently_applied: PwmPair,
        target: PwmPair,
        now: ControllerUptimeMsWrapping,
        timing: MotorTiming,
    ) -> Result<(Self, MotorDirective), MotorTransitionError> {
        if target.is_stop() {
            return Err(MotorTransitionError::StopMustBypassTransition);
        }

        let current_output = DriveOutput::from_pwm(currently_applied);
        let target_output = DriveOutput::from_pwm(target);
        if currently_applied.is_stop() {
            return Ok((
                Self {
                    target,
                    phase: MotorTransitionPhase::PreloadingDisabled,
                    phase_deadline: WrappingDeadline::after(now, timing.preload_latch()),
                },
                MotorDirective::PreloadWhileDisabled(target_output),
            ));
        }

        if output_direction_changed(current_output, target_output) {
            Ok((
                Self {
                    target,
                    phase: MotorTransitionPhase::Neutralizing,
                    phase_deadline: WrappingDeadline::after(now, timing.neutral_hold()),
                },
                MotorDirective::DisableAndZero,
            ))
        } else {
            Ok((
                Self {
                    target,
                    phase: MotorTransitionPhase::UpdatingEnabled,
                    phase_deadline: WrappingDeadline::after(now, timing.preload_latch()),
                },
                MotorDirective::UpdateEnabled(target_output),
            ))
        }
    }

    pub const fn target(self) -> PwmPair {
        self.target
    }

    pub const fn phase(self) -> MotorTransitionPhase {
        self.phase
    }

    pub const fn phase_deadline(self) -> WrappingDeadline {
        self.phase_deadline
    }

    pub fn poll(
        &mut self,
        now: ControllerUptimeMsWrapping,
        timing: MotorTiming,
    ) -> Result<MotorPoll, MotorTransitionError> {
        match self.phase_deadline.status_at(now) {
            DeadlineStatus::Pending => Ok(MotorPoll::Pending(MotorDirective::Hold)),
            DeadlineStatus::ObservationGap => Err(MotorTransitionError::ClockObservationGap),
            DeadlineStatus::Reached => match self.phase {
                MotorTransitionPhase::Neutralizing => {
                    self.phase = MotorTransitionPhase::PreloadingDisabled;
                    self.phase_deadline = WrappingDeadline::after(now, timing.preload_latch());
                    Ok(MotorPoll::Pending(MotorDirective::PreloadWhileDisabled(
                        DriveOutput::from_pwm(self.target),
                    )))
                }
                MotorTransitionPhase::PreloadingDisabled => Ok(MotorPoll::Applied {
                    pwm: self.target,
                    output: DriveOutput::from_pwm(self.target),
                    directive: MotorDirective::EnablePreloaded(DriveOutput::from_pwm(self.target)),
                }),
                MotorTransitionPhase::UpdatingEnabled => Ok(MotorPoll::Applied {
                    pwm: self.target,
                    output: DriveOutput::from_pwm(self.target),
                    directive: MotorDirective::Hold,
                }),
            },
        }
    }
}

fn output_direction_changed(current: DriveOutput, target: DriveOutput) -> bool {
    current.left().direction() != target.left().direction()
        || current.right().direction() != target.right().direction()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MotorPoll {
    Pending(MotorDirective),
    Applied {
        pwm: PwmPair,
        output: DriveOutput,
        directive: MotorDirective,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MotorTransitionError {
    StopMustBypassTransition,
    ClockObservationGap,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pwm(value: i8) -> PwmPercent {
        PwmPercent::try_new(value).expect("test PWM is in the canonical domain")
    }

    fn pair(left: i8, right: i8) -> PwmPair {
        PwmPair::from_validated(pwm(left), pwm(right))
    }

    fn duration(value: u32) -> DurationMs {
        DurationMs::try_new(value).expect("test duration is valid")
    }

    fn timing() -> MotorTiming {
        MotorTiming::new(duration(5), duration(2))
    }

    fn validated_envelope(maximum_step: u8) -> ActuatorEnvelope {
        ActuatorEnvelope::validated(
            ActuatorConfigFingerprint::try_new([7; 16]).expect("nonzero fingerprint"),
            PwmRange::try_new(pwm(-50), pwm(50)).expect("valid left range"),
            PwmRange::try_new(pwm(-40), pwm(40)).expect("valid right range"),
            PwmStepLimit::try_new(maximum_step, maximum_step).expect("nonzero step"),
        )
    }

    #[test]
    fn wrapping_deadline_is_exact_across_wrap_and_rejects_ambiguous_observations() {
        assert_eq!(DurationMs::try_new(0), Err(DurationMsError::Zero));
        assert!(matches!(
            DurationMs::try_new(1_u32 << 31),
            Err(DurationMsError::AboveUnambiguousHalfRange { .. })
        ));

        let deadline =
            WrappingDeadline::after(ControllerUptimeMsWrapping::new(u32::MAX - 2), duration(5));
        assert_eq!(deadline.expires_at().get(), 2);
        assert_eq!(
            deadline.status_at(ControllerUptimeMsWrapping::new(1)),
            DeadlineStatus::Pending
        );
        assert_eq!(
            deadline.status_at(ControllerUptimeMsWrapping::new(2)),
            DeadlineStatus::Reached
        );
        assert_eq!(
            deadline.status_at(ControllerUptimeMsWrapping::new(u32::MAX - 3)),
            DeadlineStatus::ObservationGap
        );
    }

    #[test]
    fn wheel_drive_type_has_no_both_directions_state() {
        for raw in -100..=100 {
            let drive = WheelDrive::from_pwm(pwm(raw));
            match (raw.signum(), drive) {
                (0, WheelDrive::Disabled)
                | (1, WheelDrive::Forward(_))
                | (-1, WheelDrive::Reverse(_)) => {}
                _ => panic!("wheel direction does not match PWM sign"),
            }
        }
        assert!(DriveOutput::from_pwm(PwmPair::STOP).is_safe_neutral());
    }

    #[test]
    fn step_limit_rejects_values_outside_the_canonical_pwm_delta_domain() {
        assert_eq!(
            PwmStepLimit::try_new(0, 1),
            Err(PwmStepLimitError::ZeroLeft)
        );
        assert_eq!(
            PwmStepLimit::try_new(1, 0),
            Err(PwmStepLimitError::ZeroRight)
        );
        assert_eq!(
            PwmStepLimit::try_new(MAX_PWM_STEP_PERCENT + 1, 1),
            Err(PwmStepLimitError::AboveMaximumLeft {
                value: MAX_PWM_STEP_PERCENT + 1,
                maximum: MAX_PWM_STEP_PERCENT,
            })
        );
        assert_eq!(
            PwmStepLimit::try_new(1, MAX_PWM_STEP_PERCENT + 1),
            Err(PwmStepLimitError::AboveMaximumRight {
                value: MAX_PWM_STEP_PERCENT + 1,
                maximum: MAX_PWM_STEP_PERCENT,
            })
        );
        assert!(PwmStepLimit::try_new(MAX_PWM_STEP_PERCENT, MAX_PWM_STEP_PERCENT).is_ok());
    }

    #[test]
    fn motion_is_disabled_by_default_but_stop_always_bypasses_the_envelope() {
        let envelope = ActuatorEnvelope::unvalidated();
        assert_eq!(
            envelope.validate_transition(PwmPair::STOP, pair(1, 0)),
            Err(MotionEnvelopeError::MotionDisabledUntilValidated)
        );
        assert_eq!(
            envelope.validate_transition(pair(50, -50), PwmPair::STOP),
            Ok(())
        );
        assert_eq!(
            envelope.physical_stop_semantics(),
            PhysicalStopSemantics::Unverified
        );
        assert!(!ObservationalOdometryContract::Absent.configures_quadrature_inputs());
        assert!(
            ObservationalOdometryContract::QuadratureEncoderTimers.configures_quadrature_inputs()
        );
    }

    #[test]
    fn envelope_rejects_without_clamping_for_every_canonical_pwm_pair() {
        let envelope = validated_envelope(200);
        for left in -100..=100 {
            for right in -100..=100 {
                let requested = pair(left, right);
                let result = envelope.validate_transition(PwmPair::STOP, requested);
                let expected = requested.is_stop()
                    || ((-50..=50).contains(&left) && (-40..=40).contains(&right));
                assert_eq!(result.is_ok(), expected, "left={left}, right={right}");
            }
        }
    }

    #[test]
    fn envelope_enforces_per_wheel_step_and_zero_bypasses_it() {
        let envelope = validated_envelope(10);
        assert_eq!(
            envelope.validate_transition(pair(10, -10), pair(20, -20)),
            Ok(())
        );
        assert!(matches!(
            envelope.validate_transition(pair(10, 0), pair(21, 0)),
            Err(MotionEnvelopeError::LeftStepTooLarge { .. })
        ));
        assert_eq!(
            envelope.validate_transition(pair(50, 40), PwmPair::STOP),
            Ok(())
        );
    }

    #[test]
    fn provisional_four_pwm_envelope_exhaustively_enforces_cap_and_command_step() {
        let fingerprint =
            ActuatorConfigFingerprint::try_new(*b"KIKO-4PWM-CAND1!").expect("fingerprint");
        let envelope = ProvisionalBoundedFourPwmEnvelope::try_new(fingerprint)
            .expect("canonical constants form an envelope")
            .into_actuator_envelope();
        assert!(envelope.is_operator_supervised_four_pwm_candidate());
        assert_eq!(envelope.fingerprint(), Some(fingerprint));
        assert_eq!(
            envelope.physical_stop_semantics(),
            PhysicalStopSemantics::Unverified
        );

        let cap = i8::try_from(MAX_OPERATOR_SUPERVISED_FOUR_PWM_PWM_PERCENT)
            .expect("protocol cap fits i8");
        for current in -cap..=cap {
            assert_eq!(
                envelope.validate_transition(pair(current, -current), PwmPair::STOP),
                Ok(()),
                "zero must bypass cap and command-step checks"
            );
            for requested in -100..=100 {
                let left_result =
                    envelope.validate_transition(pair(current, 0), pair(requested, 0));
                let right_result =
                    envelope.validate_transition(pair(0, current), pair(0, requested));
                let expected = requested == 0
                    || ((-cap..=cap).contains(&requested)
                        && current.abs_diff(requested)
                            <= PROVISIONAL_FOUR_PWM_MAX_COMMAND_STEP_PERCENT);
                assert_eq!(
                    left_result.is_ok(),
                    expected,
                    "left current={current}, requested={requested}"
                );
                assert_eq!(
                    right_result.is_ok(),
                    expected,
                    "right current={current}, requested={requested}"
                );
            }
        }

        for requested_left in -100..=100 {
            for requested_right in -100..=100 {
                let requested = pair(requested_left, requested_right);
                let expected = requested.is_stop()
                    || ((-cap..=cap).contains(&requested_left)
                        && (-cap..=cap).contains(&requested_right)
                        && requested_left.unsigned_abs()
                            <= PROVISIONAL_FOUR_PWM_MAX_COMMAND_STEP_PERCENT
                        && requested_right.unsigned_abs()
                            <= PROVISIONAL_FOUR_PWM_MAX_COMMAND_STEP_PERCENT);
                assert_eq!(
                    envelope
                        .validate_transition(PwmPair::STOP, requested)
                        .is_ok(),
                    expected,
                    "stopped request=({requested_left},{requested_right})"
                );
            }
        }
    }

    #[test]
    fn stopped_start_preloads_while_disabled_before_enabling() {
        let (mut transition, directive) = MotorTransition::start(
            PwmPair::STOP,
            pair(10, -10),
            ControllerUptimeMsWrapping::new(100),
            timing(),
        )
        .expect("nonzero transition");
        assert!(matches!(directive, MotorDirective::PreloadWhileDisabled(_)));
        assert_eq!(
            transition
                .poll(ControllerUptimeMsWrapping::new(101), timing())
                .expect("unambiguous poll"),
            MotorPoll::Pending(MotorDirective::Hold)
        );
        assert!(matches!(
            transition
                .poll(ControllerUptimeMsWrapping::new(102), timing())
                .expect("deadline reached"),
            MotorPoll::Applied {
                directive: MotorDirective::EnablePreloaded(_),
                ..
            }
        ));
    }

    #[test]
    fn direction_change_has_neutral_then_preload_then_enable() {
        let (mut transition, directive) = MotorTransition::start(
            pair(10, 10),
            pair(-10, 10),
            ControllerUptimeMsWrapping::new(200),
            timing(),
        )
        .expect("nonzero transition");
        assert_eq!(directive, MotorDirective::DisableAndZero);
        assert_eq!(transition.phase(), MotorTransitionPhase::Neutralizing);
        assert_eq!(
            transition
                .poll(ControllerUptimeMsWrapping::new(204), timing())
                .expect("before deadline"),
            MotorPoll::Pending(MotorDirective::Hold)
        );
        assert!(matches!(
            transition
                .poll(ControllerUptimeMsWrapping::new(205), timing())
                .expect("neutral complete"),
            MotorPoll::Pending(MotorDirective::PreloadWhileDisabled(_))
        ));
        assert_eq!(transition.phase(), MotorTransitionPhase::PreloadingDisabled);
        assert!(matches!(
            transition
                .poll(ControllerUptimeMsWrapping::new(207), timing())
                .expect("preload complete"),
            MotorPoll::Applied {
                directive: MotorDirective::EnablePreloaded(_),
                ..
            }
        ));
    }

    #[test]
    fn same_direction_update_never_enables_an_opposite_channel() {
        let (mut transition, directive) = MotorTransition::start(
            pair(10, -10),
            pair(20, -20),
            ControllerUptimeMsWrapping::new(300),
            timing(),
        )
        .expect("nonzero transition");
        let MotorDirective::UpdateEnabled(output) = directive else {
            panic!("same-direction change should update enabled channels")
        };
        assert!(matches!(output.left(), WheelDrive::Forward(_)));
        assert!(matches!(output.right(), WheelDrive::Reverse(_)));
        assert!(matches!(
            transition
                .poll(ControllerUptimeMsWrapping::new(302), timing())
                .expect("update complete"),
            MotorPoll::Applied { .. }
        ));
    }

    #[test]
    fn stop_can_never_enter_the_delayed_transition_machine() {
        assert_eq!(
            MotorTransition::start(
                pair(10, 10),
                PwmPair::STOP,
                ControllerUptimeMsWrapping::new(0),
                timing(),
            ),
            Err(MotorTransitionError::StopMustBypassTransition)
        );
    }
}
