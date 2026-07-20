use std::fmt;
use std::num::{NonZeroU8, NonZeroU16, NonZeroU64};

use robot_protocol::{AppliedPwm, AppliedPwmError, PwmPercent};

use crate::BASE_IDENTIFICATION_V1;
use crate::identity::{BoundedId, IdentifierError};
use crate::time::MonotonicTimestampNs;

const STEPS_PER_CYCLE: u16 = 4;
const MAX_CYCLES: u8 = 16;
const MAX_EXCITATION_STEPS: u16 = MAX_CYCLES as u16 * STEPS_PER_CYCLE;

#[derive(Clone, Debug, PartialEq)]
pub struct CommissioningConfigV1Dto {
    pub schema_version: u32,
    pub expected_controller_session_id: String,
    pub expected_visual_velocity_source_id: String,
    pub expected_imu_calibration_id: String,
    pub symmetric_pwm_percent: u8,
    pub spin_pwm_percent: u8,
    pub max_abs_pwm_percent: u8,
    pub excitation_duration_ns: u64,
    pub zero_dwell_duration_ns: u64,
    pub application_timeout_ns: u64,
    pub max_visual_age_ns: u64,
    pub max_imu_age_ns: u64,
    pub max_controller_age_ns: u64,
    pub max_abs_stationary_forward_velocity_mps: f64,
    pub max_abs_stationary_yaw_rate_rad_s: f64,
    pub max_total_duration_ns: u64,
    pub cycles: u8,
    pub max_excitation_steps: u16,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CommissioningConfigV1 {
    expected_controller_session_id: BoundedId,
    expected_visual_velocity_source_id: BoundedId,
    expected_imu_calibration_id: BoundedId,
    symmetric_pwm_percent: NonZeroU8,
    spin_pwm_percent: NonZeroU8,
    max_abs_pwm_percent: NonZeroU8,
    excitation_duration_ns: NonZeroU64,
    zero_dwell_duration_ns: NonZeroU64,
    application_timeout_ns: NonZeroU64,
    max_visual_age_ns: NonZeroU64,
    max_imu_age_ns: NonZeroU64,
    max_controller_age_ns: NonZeroU64,
    max_abs_stationary_forward_velocity_mps: f64,
    max_abs_stationary_yaw_rate_rad_s: f64,
    max_total_duration_ns: NonZeroU64,
    cycles: NonZeroU8,
    max_excitation_steps: NonZeroU16,
    program_steps: NonZeroU16,
}

impl CommissioningConfigV1 {
    pub fn parse(dto: CommissioningConfigV1Dto) -> Result<Self, CommissioningConfigParseError> {
        if dto.schema_version != BASE_IDENTIFICATION_V1 {
            return Err(CommissioningConfigParseError::UnsupportedSchemaVersion(
                dto.schema_version,
            ));
        }
        let expected_controller_session_id = parse_config_id(
            "expected_controller_session_id",
            dto.expected_controller_session_id,
        )?;
        let expected_visual_velocity_source_id = parse_config_id(
            "expected_visual_velocity_source_id",
            dto.expected_visual_velocity_source_id,
        )?;
        let expected_imu_calibration_id = parse_config_id(
            "expected_imu_calibration_id",
            dto.expected_imu_calibration_id,
        )?;
        for (field, value) in [
            ("symmetric_pwm_percent", dto.symmetric_pwm_percent),
            ("spin_pwm_percent", dto.spin_pwm_percent),
            ("max_abs_pwm_percent", dto.max_abs_pwm_percent),
        ] {
            if !(1..=100).contains(&value) {
                return Err(CommissioningConfigParseError::IntegerOutOfRange {
                    field,
                    value: u64::from(value),
                    min: 1,
                    max: 100,
                });
            }
        }
        for (field, value) in [
            (
                "max_abs_stationary_forward_velocity_mps",
                dto.max_abs_stationary_forward_velocity_mps,
            ),
            (
                "max_abs_stationary_yaw_rate_rad_s",
                dto.max_abs_stationary_yaw_rate_rad_s,
            ),
        ] {
            if !value.is_finite() {
                return Err(CommissioningConfigParseError::NonFinite { field, value });
            }
            if value <= 0.0 {
                return Err(CommissioningConfigParseError::NotPositive { field, value });
            }
        }
        if dto.symmetric_pwm_percent > dto.max_abs_pwm_percent {
            return Err(CommissioningConfigParseError::ExcitationExceedsBound {
                field: "symmetric_pwm_percent",
                value: dto.symmetric_pwm_percent,
                maximum: dto.max_abs_pwm_percent,
            });
        }
        if dto.spin_pwm_percent > dto.max_abs_pwm_percent {
            return Err(CommissioningConfigParseError::ExcitationExceedsBound {
                field: "spin_pwm_percent",
                value: dto.spin_pwm_percent,
                maximum: dto.max_abs_pwm_percent,
            });
        }
        for (field, value) in [
            ("excitation_duration_ns", dto.excitation_duration_ns),
            ("zero_dwell_duration_ns", dto.zero_dwell_duration_ns),
            ("application_timeout_ns", dto.application_timeout_ns),
            ("max_visual_age_ns", dto.max_visual_age_ns),
            ("max_imu_age_ns", dto.max_imu_age_ns),
            ("max_controller_age_ns", dto.max_controller_age_ns),
            ("max_total_duration_ns", dto.max_total_duration_ns),
        ] {
            if value == 0 {
                return Err(CommissioningConfigParseError::IntegerOutOfRange {
                    field,
                    value,
                    min: 1,
                    max: u64::MAX,
                });
            }
        }
        if !(1..=MAX_CYCLES).contains(&dto.cycles) {
            return Err(CommissioningConfigParseError::IntegerOutOfRange {
                field: "cycles",
                value: u64::from(dto.cycles),
                min: 1,
                max: u64::from(MAX_CYCLES),
            });
        }
        let program_steps = u16::from(dto.cycles) * STEPS_PER_CYCLE;
        if dto.max_excitation_steps < program_steps
            || dto.max_excitation_steps > MAX_EXCITATION_STEPS
        {
            return Err(CommissioningConfigParseError::IntegerOutOfRange {
                field: "max_excitation_steps",
                value: u64::from(dto.max_excitation_steps),
                min: u64::from(program_steps),
                max: u64::from(MAX_EXCITATION_STEPS),
            });
        }
        let minimum_excitation_ns = u64::from(program_steps)
            .checked_mul(dto.excitation_duration_ns)
            .and_then(|duration| {
                u64::from(program_steps + 1)
                    .checked_mul(dto.zero_dwell_duration_ns)
                    .and_then(|zero_duration| duration.checked_add(zero_duration))
            })
            .ok_or(CommissioningConfigParseError::MinimumProgramDurationOverflow)?;
        let minimum_total_duration_ns = minimum_excitation_ns
            .checked_add(1)
            .ok_or(CommissioningConfigParseError::MinimumProgramDurationOverflow)?;
        if dto.max_total_duration_ns < minimum_total_duration_ns {
            return Err(
                CommissioningConfigParseError::TotalDurationCannotContainProgram {
                    configured_ns: dto.max_total_duration_ns,
                    minimum_ns: minimum_total_duration_ns,
                },
            );
        }
        Ok(Self {
            expected_controller_session_id,
            expected_visual_velocity_source_id,
            expected_imu_calibration_id,
            symmetric_pwm_percent: NonZeroU8::new(dto.symmetric_pwm_percent)
                .expect("validated nonzero"),
            spin_pwm_percent: NonZeroU8::new(dto.spin_pwm_percent).expect("validated nonzero"),
            max_abs_pwm_percent: NonZeroU8::new(dto.max_abs_pwm_percent)
                .expect("validated nonzero"),
            excitation_duration_ns: NonZeroU64::new(dto.excitation_duration_ns)
                .expect("validated nonzero"),
            zero_dwell_duration_ns: NonZeroU64::new(dto.zero_dwell_duration_ns)
                .expect("validated nonzero"),
            application_timeout_ns: NonZeroU64::new(dto.application_timeout_ns)
                .expect("validated nonzero"),
            max_visual_age_ns: NonZeroU64::new(dto.max_visual_age_ns).expect("validated nonzero"),
            max_imu_age_ns: NonZeroU64::new(dto.max_imu_age_ns).expect("validated nonzero"),
            max_controller_age_ns: NonZeroU64::new(dto.max_controller_age_ns)
                .expect("validated nonzero"),
            max_abs_stationary_forward_velocity_mps: dto.max_abs_stationary_forward_velocity_mps,
            max_abs_stationary_yaw_rate_rad_s: dto.max_abs_stationary_yaw_rate_rad_s,
            max_total_duration_ns: NonZeroU64::new(dto.max_total_duration_ns)
                .expect("validated nonzero"),
            cycles: NonZeroU8::new(dto.cycles).expect("validated nonzero"),
            max_excitation_steps: NonZeroU16::new(dto.max_excitation_steps)
                .expect("validated nonzero"),
            program_steps: NonZeroU16::new(program_steps).expect("cycles are nonzero"),
        })
    }

    pub fn program_steps(self) -> NonZeroU16 {
        self.program_steps
    }

    pub fn max_abs_pwm_percent(self) -> NonZeroU8 {
        self.max_abs_pwm_percent
    }

    pub fn cycles(self) -> NonZeroU8 {
        self.cycles
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CommissioningConfigParseError {
    UnsupportedSchemaVersion(u32),
    InvalidIdentifier(IdentifierError),
    IntegerOutOfRange {
        field: &'static str,
        value: u64,
        min: u64,
        max: u64,
    },
    ExcitationExceedsBound {
        field: &'static str,
        value: u8,
        maximum: u8,
    },
    NonFinite {
        field: &'static str,
        value: f64,
    },
    NotPositive {
        field: &'static str,
        value: f64,
    },
    MinimumProgramDurationOverflow,
    TotalDurationCannotContainProgram {
        configured_ns: u64,
        minimum_ns: u64,
    },
}

impl fmt::Display for CommissioningConfigParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid V1 commissioning configuration: {self:?}"
        )
    }
}

impl std::error::Error for CommissioningConfigParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidIdentifier(source) => Some(source),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CommissioningEvidenceV1Dto {
    pub controller_session_id: String,
    pub visual_velocity_source_id: String,
    pub imu_calibration_id: String,
    pub controller_observed_at_ns: u64,
    pub visual_observed_at_ns: u64,
    pub imu_observed_at_ns: u64,
    pub applied_command_sequence: u64,
    pub applied_left_pwm_percent: i8,
    pub applied_right_pwm_percent: i8,
    pub visual_forward_velocity_mps: f64,
    pub calibrated_imu_yaw_rate_rad_s: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CommissioningEvidence {
    controller_observed_at: MonotonicTimestampNs,
    visual_observed_at: MonotonicTimestampNs,
    imu_observed_at: MonotonicTimestampNs,
    applied_command_sequence: u64,
    applied_pwm: AppliedPwm,
    visual_forward_velocity_mps: f64,
    calibrated_imu_yaw_rate_rad_s: f64,
}

impl CommissioningEvidence {
    pub fn parse(
        dto: CommissioningEvidenceV1Dto,
        config: CommissioningConfigV1,
    ) -> Result<Self, CommissioningEvidenceParseError> {
        let controller_session_id =
            parse_evidence_id("controller_session_id", dto.controller_session_id)?;
        let visual_velocity_source_id =
            parse_evidence_id("visual_velocity_source_id", dto.visual_velocity_source_id)?;
        let imu_calibration_id = parse_evidence_id("imu_calibration_id", dto.imu_calibration_id)?;
        for (field, expected, actual) in [
            (
                "controller_session_id",
                config.expected_controller_session_id,
                controller_session_id,
            ),
            (
                "visual_velocity_source_id",
                config.expected_visual_velocity_source_id,
                visual_velocity_source_id,
            ),
            (
                "imu_calibration_id",
                config.expected_imu_calibration_id,
                imu_calibration_id,
            ),
        ] {
            if expected != actual {
                return Err(CommissioningEvidenceParseError::IdentityMismatch {
                    field,
                    expected: Box::new(expected),
                    actual: Box::new(actual),
                });
            }
        }
        let applied_pwm =
            AppliedPwm::try_new(dto.applied_left_pwm_percent, dto.applied_right_pwm_percent)
                .map_err(CommissioningEvidenceParseError::InvalidAppliedPwm)?;
        for (field, value) in [
            (
                "visual_forward_velocity_mps",
                dto.visual_forward_velocity_mps,
            ),
            (
                "calibrated_imu_yaw_rate_rad_s",
                dto.calibrated_imu_yaw_rate_rad_s,
            ),
        ] {
            if !value.is_finite() {
                return Err(CommissioningEvidenceParseError::NonFinite { field, value });
            }
        }
        Ok(Self {
            controller_observed_at: MonotonicTimestampNs::from_nanos(dto.controller_observed_at_ns),
            visual_observed_at: MonotonicTimestampNs::from_nanos(dto.visual_observed_at_ns),
            imu_observed_at: MonotonicTimestampNs::from_nanos(dto.imu_observed_at_ns),
            applied_command_sequence: dto.applied_command_sequence,
            applied_pwm,
            visual_forward_velocity_mps: dto.visual_forward_velocity_mps,
            calibrated_imu_yaw_rate_rad_s: dto.calibrated_imu_yaw_rate_rad_s,
        })
    }

    pub fn applied_pwm(self) -> AppliedPwm {
        self.applied_pwm
    }

    pub fn visual_forward_velocity_mps(self) -> f64 {
        self.visual_forward_velocity_mps
    }

    pub fn calibrated_imu_yaw_rate_rad_s(self) -> f64 {
        self.calibrated_imu_yaw_rate_rad_s
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum CommissioningEvidenceParseError {
    InvalidIdentifier(IdentifierError),
    IdentityMismatch {
        field: &'static str,
        expected: Box<BoundedId>,
        actual: Box<BoundedId>,
    },
    InvalidAppliedPwm(AppliedPwmError),
    NonFinite {
        field: &'static str,
        value: f64,
    },
}

impl fmt::Display for CommissioningEvidenceParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid commissioning evidence: {self:?}")
    }
}

impl std::error::Error for CommissioningEvidenceParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidIdentifier(source) => Some(source),
            Self::InvalidAppliedPwm(source) => Some(source),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cancellation {
    Continue,
    Requested,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExcitationKind {
    SymmetricForward,
    SymmetricReverse,
    PositiveYawSpin,
    NegativeYawSpin,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CanonicalPwmCommand {
    left: PwmPercent,
    right: PwmPercent,
}

impl CanonicalPwmCommand {
    const ZERO: Self = Self {
        left: PwmPercent::ZERO,
        right: PwmPercent::ZERO,
    };

    fn from_validated(left: PwmPercent, right: PwmPercent) -> Self {
        Self { left, right }
    }

    pub fn left(self) -> PwmPercent {
        self.left
    }

    pub fn right(self) -> PwmPercent {
        self.right
    }

    pub fn is_zero(self) -> bool {
        self == Self::ZERO
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommissioningStep {
    index: u16,
    kind: ExcitationKind,
    pwm: CanonicalPwmCommand,
}

impl CommissioningStep {
    pub fn index(self) -> u16 {
        self.index
    }

    pub fn kind(self) -> ExcitationKind {
        self.kind
    }

    pub fn pwm(self) -> CanonicalPwmCommand {
        self.pwm
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceKind {
    Controller,
    VisualVelocity,
    ImuYawRate,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CommissioningStopReason {
    Cancelled,
    ClockRegression {
        previous_ns: u64,
        current_ns: u64,
    },
    EvidenceFromFuture {
        kind: EvidenceKind,
        now_ns: u64,
        observed_at_ns: u64,
    },
    StaleEvidence {
        kind: EvidenceKind,
        age_ns: u64,
        maximum_age_ns: u64,
    },
    ControllerSequenceRegression {
        previous: u64,
        current: u64,
    },
    ChangedPwmForSameControllerSequence {
        sequence: u64,
        previous: AppliedPwm,
        current: AppliedPwm,
    },
    TotalDurationLimitReached {
        elapsed_ns: u64,
        maximum_ns: u64,
    },
    ExcitationStepLimitReached {
        issued: u16,
        maximum: u16,
    },
    ApplicationTimeout {
        step_index: u16,
        elapsed_ns: u64,
        maximum_ns: u64,
    },
    AppliedSequenceDidNotAdvance {
        step_index: u16,
        zero_sequence: u64,
        applied_sequence: u64,
    },
    UnexpectedAppliedPwm {
        step_index: u16,
        expected_command: CanonicalPwmCommand,
        actual: AppliedPwm,
    },
    ExcitationEndedEarly {
        step_index: u16,
        elapsed_ns: u64,
        required_ns: u64,
    },
    ZeroSequenceDidNotAdvance {
        step_index: u16,
        excitation_sequence: u64,
        zero_sequence: u64,
    },
    ZeroApplicationTimeout {
        completed_step_index: u16,
        elapsed_ns: u64,
        maximum_ns: u64,
    },
    MotionWhileZeroRequired {
        forward_velocity_mps: f64,
        maximum_abs_forward_velocity_mps: f64,
        yaw_rate_rad_s: f64,
        maximum_abs_yaw_rate_rad_s: f64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CommissioningState {
    AwaitingInitialZero,
    ZeroDwell { next_step_index: u16 },
    AwaitingApplication { step_index: u16 },
    Exciting { step_index: u16 },
    AwaitingInterstepZero { completed_step_index: u16 },
    Completed,
    Aborted(CommissioningStopReason),
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use = "commissioning decisions must be applied or replaced by an explicit stop"]
pub enum CommissioningAction {
    RequiredZero {
        state: CommissioningState,
    },
    Excitation {
        state: CommissioningState,
        step: CommissioningStep,
    },
}

impl CommissioningAction {
    pub fn state(self) -> CommissioningState {
        match self {
            Self::RequiredZero { state } | Self::Excitation { state, .. } => state,
        }
    }

    pub fn required_pwm(self) -> CanonicalPwmCommand {
        match self {
            Self::RequiredZero { .. } => CanonicalPwmCommand::ZERO,
            Self::Excitation { step, .. } => step.pwm,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Phase {
    AwaitingZero {
        next_step_index: u16,
        after_excitation: Option<CompletedExcitation>,
    },
    ZeroDwell {
        next_step_index: u16,
        zero_since: MonotonicTimestampNs,
        zero_sequence: u64,
    },
    AwaitingApplication {
        step: CommissioningStep,
        issued_at: MonotonicTimestampNs,
        zero_sequence: u64,
    },
    Exciting {
        step: CommissioningStep,
        applied_at: MonotonicTimestampNs,
        applied_sequence: u64,
    },
    Completed,
    Aborted(CommissioningStopReason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CompletedExcitation {
    pwm: CanonicalPwmCommand,
    applied_sequence: u64,
    zero_requested_at: MonotonicTimestampNs,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CommissioningController {
    config: CommissioningConfigV1,
    phase: Phase,
    started_at: Option<MonotonicTimestampNs>,
    last_now: Option<MonotonicTimestampNs>,
    last_controller_sequence: Option<u64>,
    last_applied_pwm: Option<AppliedPwm>,
    issued_steps: u16,
}

impl CommissioningController {
    pub fn new(config: CommissioningConfigV1) -> Self {
        Self {
            config,
            phase: Phase::AwaitingZero {
                next_step_index: 0,
                after_excitation: None,
            },
            started_at: None,
            last_now: None,
            last_controller_sequence: None,
            last_applied_pwm: None,
            issued_steps: 0,
        }
    }

    pub fn state(self) -> CommissioningState {
        state_from_phase(self.phase)
    }

    pub fn advance(
        &mut self,
        now: MonotonicTimestampNs,
        evidence: CommissioningEvidence,
        cancellation: Cancellation,
    ) -> CommissioningAction {
        if let Phase::Aborted(reason) = self.phase {
            return required_zero(CommissioningState::Aborted(reason));
        }
        if self.phase == Phase::Completed {
            return required_zero(CommissioningState::Completed);
        }
        if cancellation == Cancellation::Requested {
            return self.abort(CommissioningStopReason::Cancelled);
        }
        if let Some(previous) = self.last_now
            && now < previous
        {
            return self.abort(CommissioningStopReason::ClockRegression {
                previous_ns: previous.as_nanos(),
                current_ns: now.as_nanos(),
            });
        }
        self.last_now = Some(now);
        let started_at = *self.started_at.get_or_insert(now);
        let elapsed = now
            .checked_age_since(started_at)
            .expect("clock regression checked before elapsed time");
        if elapsed >= self.config.max_total_duration_ns.get() {
            return self.abort(CommissioningStopReason::TotalDurationLimitReached {
                elapsed_ns: elapsed,
                maximum_ns: self.config.max_total_duration_ns.get(),
            });
        }
        if let Some(reason) = freshness_failure(now, evidence, self.config) {
            return self.abort(reason);
        }
        if let Some(previous) = self.last_controller_sequence
            && evidence.applied_command_sequence < previous
        {
            return self.abort(CommissioningStopReason::ControllerSequenceRegression {
                previous,
                current: evidence.applied_command_sequence,
            });
        }
        if let (Some(previous_sequence), Some(previous_pwm)) =
            (self.last_controller_sequence, self.last_applied_pwm)
            && evidence.applied_command_sequence == previous_sequence
            && evidence.applied_pwm != previous_pwm
        {
            return self.abort(
                CommissioningStopReason::ChangedPwmForSameControllerSequence {
                    sequence: previous_sequence,
                    previous: previous_pwm,
                    current: evidence.applied_pwm,
                },
            );
        }
        self.last_controller_sequence = Some(evidence.applied_command_sequence);
        self.last_applied_pwm = Some(evidence.applied_pwm);

        match self.phase {
            Phase::AwaitingZero {
                next_step_index,
                after_excitation,
            } => self.advance_awaiting_zero(now, evidence, next_step_index, after_excitation),
            Phase::ZeroDwell {
                next_step_index,
                zero_since,
                zero_sequence,
            } => self.advance_zero_dwell(now, evidence, next_step_index, zero_since, zero_sequence),
            Phase::AwaitingApplication {
                step,
                issued_at,
                zero_sequence,
            } => self.advance_awaiting_application(now, evidence, step, issued_at, zero_sequence),
            Phase::Exciting {
                step,
                applied_at,
                applied_sequence,
            } => self.advance_exciting(now, evidence, step, applied_at, applied_sequence),
            Phase::Completed | Phase::Aborted(_) => unreachable!("terminal phases returned above"),
        }
    }

    fn advance_awaiting_zero(
        &mut self,
        now: MonotonicTimestampNs,
        evidence: CommissioningEvidence,
        next_step_index: u16,
        after_excitation: Option<CompletedExcitation>,
    ) -> CommissioningAction {
        if !is_zero_pwm(evidence.applied_pwm) {
            if let Some(completed) = after_excitation {
                if !applied_matches_command(evidence.applied_pwm, completed.pwm) {
                    return self.abort(CommissioningStopReason::UnexpectedAppliedPwm {
                        step_index: next_step_index - 1,
                        expected_command: zero_pwm(),
                        actual: evidence.applied_pwm,
                    });
                }
                let elapsed = now
                    .checked_age_since(completed.zero_requested_at)
                    .expect("monotonic clock checked before zero timeout");
                if elapsed >= self.config.application_timeout_ns.get() {
                    return self.abort(CommissioningStopReason::ZeroApplicationTimeout {
                        completed_step_index: next_step_index - 1,
                        elapsed_ns: elapsed,
                        maximum_ns: self.config.application_timeout_ns.get(),
                    });
                }
            }
            let state = if let Some(completed) = next_step_index.checked_sub(1) {
                CommissioningState::AwaitingInterstepZero {
                    completed_step_index: completed,
                }
            } else {
                CommissioningState::AwaitingInitialZero
            };
            return required_zero(state);
        }
        if let Some(reason) = stationarity_failure(evidence, self.config) {
            return self.abort(reason);
        }
        if let Some(completed) = after_excitation
            && evidence.applied_command_sequence <= completed.applied_sequence
        {
            return self.abort(CommissioningStopReason::ZeroSequenceDidNotAdvance {
                step_index: next_step_index - 1,
                excitation_sequence: completed.applied_sequence,
                zero_sequence: evidence.applied_command_sequence,
            });
        }
        self.phase = Phase::ZeroDwell {
            next_step_index,
            zero_since: now,
            zero_sequence: evidence.applied_command_sequence,
        };
        required_zero(CommissioningState::ZeroDwell { next_step_index })
    }

    fn advance_zero_dwell(
        &mut self,
        now: MonotonicTimestampNs,
        evidence: CommissioningEvidence,
        next_step_index: u16,
        zero_since: MonotonicTimestampNs,
        zero_sequence: u64,
    ) -> CommissioningAction {
        if !is_zero_pwm(evidence.applied_pwm) {
            let expected = zero_pwm();
            return self.abort(CommissioningStopReason::UnexpectedAppliedPwm {
                step_index: next_step_index,
                expected_command: expected,
                actual: evidence.applied_pwm,
            });
        }
        if let Some(reason) = stationarity_failure(evidence, self.config) {
            return self.abort(reason);
        }
        if evidence.applied_command_sequence < zero_sequence {
            return self.abort(CommissioningStopReason::ControllerSequenceRegression {
                previous: zero_sequence,
                current: evidence.applied_command_sequence,
            });
        }
        let elapsed = now
            .checked_age_since(zero_since)
            .expect("future evidence rejected by freshness check");
        if elapsed < self.config.zero_dwell_duration_ns.get() {
            return required_zero(CommissioningState::ZeroDwell { next_step_index });
        }
        if next_step_index >= self.config.program_steps.get() {
            self.phase = Phase::Completed;
            return required_zero(CommissioningState::Completed);
        }
        self.issue_step(now, evidence.applied_command_sequence, next_step_index)
    }

    fn issue_step(
        &mut self,
        now: MonotonicTimestampNs,
        zero_sequence: u64,
        step_index: u16,
    ) -> CommissioningAction {
        if self.issued_steps >= self.config.max_excitation_steps.get() {
            return self.abort(CommissioningStopReason::ExcitationStepLimitReached {
                issued: self.issued_steps,
                maximum: self.config.max_excitation_steps.get(),
            });
        }
        let step = step_for_index(self.config, step_index);
        self.issued_steps += 1;
        self.phase = Phase::AwaitingApplication {
            step,
            issued_at: now,
            zero_sequence,
        };
        CommissioningAction::Excitation {
            state: CommissioningState::AwaitingApplication { step_index },
            step,
        }
    }

    fn advance_awaiting_application(
        &mut self,
        now: MonotonicTimestampNs,
        evidence: CommissioningEvidence,
        step: CommissioningStep,
        issued_at: MonotonicTimestampNs,
        zero_sequence: u64,
    ) -> CommissioningAction {
        if applied_matches_command(evidence.applied_pwm, step.pwm) {
            if evidence.applied_command_sequence <= zero_sequence {
                return self.abort(CommissioningStopReason::AppliedSequenceDidNotAdvance {
                    step_index: step.index,
                    zero_sequence,
                    applied_sequence: evidence.applied_command_sequence,
                });
            }
            self.phase = Phase::Exciting {
                step,
                applied_at: now,
                applied_sequence: evidence.applied_command_sequence,
            };
            return CommissioningAction::Excitation {
                state: CommissioningState::Exciting {
                    step_index: step.index,
                },
                step,
            };
        }
        if !is_zero_pwm(evidence.applied_pwm) {
            return self.abort(CommissioningStopReason::UnexpectedAppliedPwm {
                step_index: step.index,
                expected_command: step.pwm,
                actual: evidence.applied_pwm,
            });
        }
        let elapsed = now
            .checked_age_since(issued_at)
            .expect("monotonic clock checked before application timeout");
        if elapsed >= self.config.application_timeout_ns.get() {
            return self.abort(CommissioningStopReason::ApplicationTimeout {
                step_index: step.index,
                elapsed_ns: elapsed,
                maximum_ns: self.config.application_timeout_ns.get(),
            });
        }
        CommissioningAction::Excitation {
            state: CommissioningState::AwaitingApplication {
                step_index: step.index,
            },
            step,
        }
    }

    fn advance_exciting(
        &mut self,
        now: MonotonicTimestampNs,
        evidence: CommissioningEvidence,
        step: CommissioningStep,
        applied_at: MonotonicTimestampNs,
        applied_sequence: u64,
    ) -> CommissioningAction {
        let elapsed = now
            .checked_age_since(applied_at)
            .expect("future controller evidence rejected by freshness check");
        if elapsed >= self.config.excitation_duration_ns.get() {
            self.phase = Phase::AwaitingZero {
                next_step_index: step.index + 1,
                after_excitation: Some(CompletedExcitation {
                    pwm: step.pwm,
                    applied_sequence: evidence.applied_command_sequence.max(applied_sequence),
                    zero_requested_at: now,
                }),
            };
            return required_zero(CommissioningState::AwaitingInterstepZero {
                completed_step_index: step.index,
            });
        }
        if !applied_matches_command(evidence.applied_pwm, step.pwm) {
            if is_zero_pwm(evidence.applied_pwm) {
                return self.abort(CommissioningStopReason::ExcitationEndedEarly {
                    step_index: step.index,
                    elapsed_ns: elapsed,
                    required_ns: self.config.excitation_duration_ns.get(),
                });
            }
            return self.abort(CommissioningStopReason::UnexpectedAppliedPwm {
                step_index: step.index,
                expected_command: step.pwm,
                actual: evidence.applied_pwm,
            });
        }
        CommissioningAction::Excitation {
            state: CommissioningState::Exciting {
                step_index: step.index,
            },
            step,
        }
    }

    fn abort(&mut self, reason: CommissioningStopReason) -> CommissioningAction {
        self.phase = Phase::Aborted(reason);
        required_zero(CommissioningState::Aborted(reason))
    }
}

fn state_from_phase(phase: Phase) -> CommissioningState {
    match phase {
        Phase::AwaitingZero {
            next_step_index: 0, ..
        } => CommissioningState::AwaitingInitialZero,
        Phase::AwaitingZero {
            next_step_index, ..
        } => CommissioningState::AwaitingInterstepZero {
            completed_step_index: next_step_index - 1,
        },
        Phase::ZeroDwell {
            next_step_index, ..
        } => CommissioningState::ZeroDwell { next_step_index },
        Phase::AwaitingApplication { step, .. } => CommissioningState::AwaitingApplication {
            step_index: step.index,
        },
        Phase::Exciting { step, .. } => CommissioningState::Exciting {
            step_index: step.index,
        },
        Phase::Completed => CommissioningState::Completed,
        Phase::Aborted(reason) => CommissioningState::Aborted(reason),
    }
}

fn required_zero(state: CommissioningState) -> CommissioningAction {
    CommissioningAction::RequiredZero { state }
}

fn zero_pwm() -> CanonicalPwmCommand {
    CanonicalPwmCommand::ZERO
}

fn is_zero_pwm(pwm: AppliedPwm) -> bool {
    pwm.left() == PwmPercent::ZERO && pwm.right() == PwmPercent::ZERO
}

fn applied_matches_command(applied: AppliedPwm, command: CanonicalPwmCommand) -> bool {
    applied.left() == command.left() && applied.right() == command.right()
}

fn freshness_failure(
    now: MonotonicTimestampNs,
    evidence: CommissioningEvidence,
    config: CommissioningConfigV1,
) -> Option<CommissioningStopReason> {
    for (kind, observed_at, maximum_age_ns) in [
        (
            EvidenceKind::Controller,
            evidence.controller_observed_at,
            config.max_controller_age_ns.get(),
        ),
        (
            EvidenceKind::VisualVelocity,
            evidence.visual_observed_at,
            config.max_visual_age_ns.get(),
        ),
        (
            EvidenceKind::ImuYawRate,
            evidence.imu_observed_at,
            config.max_imu_age_ns.get(),
        ),
    ] {
        let Some(age_ns) = now.checked_age_since(observed_at) else {
            return Some(CommissioningStopReason::EvidenceFromFuture {
                kind,
                now_ns: now.as_nanos(),
                observed_at_ns: observed_at.as_nanos(),
            });
        };
        if age_ns > maximum_age_ns {
            return Some(CommissioningStopReason::StaleEvidence {
                kind,
                age_ns,
                maximum_age_ns,
            });
        }
    }
    None
}

fn stationarity_failure(
    evidence: CommissioningEvidence,
    config: CommissioningConfigV1,
) -> Option<CommissioningStopReason> {
    if evidence.visual_forward_velocity_mps.abs() > config.max_abs_stationary_forward_velocity_mps
        || evidence.calibrated_imu_yaw_rate_rad_s.abs() > config.max_abs_stationary_yaw_rate_rad_s
    {
        Some(CommissioningStopReason::MotionWhileZeroRequired {
            forward_velocity_mps: evidence.visual_forward_velocity_mps,
            maximum_abs_forward_velocity_mps: config.max_abs_stationary_forward_velocity_mps,
            yaw_rate_rad_s: evidence.calibrated_imu_yaw_rate_rad_s,
            maximum_abs_yaw_rate_rad_s: config.max_abs_stationary_yaw_rate_rad_s,
        })
    } else {
        None
    }
}

fn step_for_index(config: CommissioningConfigV1, index: u16) -> CommissioningStep {
    let symmetric =
        i8::try_from(config.symmetric_pwm_percent.get()).expect("validated PWM is at most 100");
    let spin = i8::try_from(config.spin_pwm_percent.get()).expect("validated PWM is at most 100");
    let (kind, left, right) = match index % STEPS_PER_CYCLE {
        0 => (ExcitationKind::SymmetricForward, symmetric, symmetric),
        1 => (ExcitationKind::SymmetricReverse, -symmetric, -symmetric),
        2 => (ExcitationKind::PositiveYawSpin, -spin, spin),
        _ => (ExcitationKind::NegativeYawSpin, spin, -spin),
    };
    let pwm = CanonicalPwmCommand::from_validated(
        PwmPercent::try_new(left).expect("parsed config bounds left excitation PWM"),
        PwmPercent::try_new(right).expect("parsed config bounds right excitation PWM"),
    );
    debug_assert!(left.unsigned_abs() <= config.max_abs_pwm_percent.get());
    debug_assert!(right.unsigned_abs() <= config.max_abs_pwm_percent.get());
    CommissioningStep { index, kind, pwm }
}

fn parse_config_id(
    field: &'static str,
    value: String,
) -> Result<BoundedId, CommissioningConfigParseError> {
    BoundedId::parse(field, value).map_err(CommissioningConfigParseError::InvalidIdentifier)
}

fn parse_evidence_id(
    field: &'static str,
    value: String,
) -> Result<BoundedId, CommissioningEvidenceParseError> {
    BoundedId::parse(field, value).map_err(CommissioningEvidenceParseError::InvalidIdentifier)
}
