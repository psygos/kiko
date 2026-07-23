use std::collections::TryReserveError;
use std::fmt;

use crate::data::{IdentificationDatasetV1, PlantFitConfigV1};
use crate::identity::BoundedId;
use crate::{BASE_IDENTIFICATION_V1, IDENTIFICATION_METHOD_ID};

const LOG_TAU_GRID_INTERVALS: usize = 128;
const GOLDEN_SECTION_ITERATIONS: usize = 96;
const GOLDEN_RATIO_CONJUGATE: f64 = 0.618_033_988_749_894_9;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelSide {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoverageGate {
    TrainingTransitions,
    HoldoutTransitions,
    SymmetricTransitions,
    SpinTransitions,
    ZeroTransitions,
    LeftPositiveTransitions,
    LeftNegativeTransitions,
    RightPositiveTransitions,
    RightNegativeTransitions,
    CommandChanges,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericalStage {
    TransitionPeriod,
    TimeConstantSearch,
    GainNormalEquation,
    ResidualAccumulation,
    Conditioning,
    ValidityEnvelope,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidualMetric {
    LeftWheelVelocityRmse,
    RightWheelVelocityRmse,
    ForwardVelocityRmse,
    YawRateRmse,
    MaximumAbsoluteWheelVelocityError,
}

#[derive(Debug)]
pub enum FitError {
    SamplePeriodOutsideConfiguredRange {
        interval_start_index: usize,
        actual_s: f64,
        minimum_s: f64,
        maximum_s: f64,
    },
    SamplePeriodRatioExceeded {
        minimum_s: f64,
        maximum_s: f64,
        actual_ratio: f64,
        maximum_ratio: f64,
    },
    InsufficientCoverage {
        gate: CoverageGate,
        actual: u32,
        required: u32,
    },
    TransitionStorageAllocation {
        requested_transitions: usize,
        source: TryReserveError,
    },
    NumericalFailure {
        wheel: Option<WheelSide>,
        stage: NumericalStage,
    },
    TimeConstantAtSearchBoundary {
        wheel: WheelSide,
        fitted_s: f64,
        minimum_s: f64,
        maximum_s: f64,
        required_margin_fraction: f64,
    },
    VelocityGainOutsideConfiguredRange {
        wheel: WheelSide,
        fitted_mps_per_pwm_percent: f64,
        minimum_abs_mps_per_pwm_percent: f64,
        maximum_abs_mps_per_pwm_percent: f64,
    },
    NonPositiveCanonicalVelocityGain {
        wheel: WheelSide,
        fitted_mps_per_pwm_percent: f64,
    },
    TimeConstantSensitivityTooLow {
        wheel: WheelSide,
        actual_rms_mps: f64,
        minimum_rms_mps: f64,
    },
    IllConditionedNormalMatrix {
        wheel: WheelSide,
        actual_condition_number: f64,
        maximum_condition_number: f64,
    },
    HoldoutResidualExceeded {
        metric: ResidualMetric,
        actual: f64,
        maximum: f64,
    },
}

impl fmt::Display for FitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "encoderless first-order plant fit failed: {self:?}"
        )
    }
}

impl std::error::Error for FitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::TransitionStorageAllocation { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IdentifiedWheelPlantV1 {
    velocity_gain_mps_per_pwm_percent: f64,
    time_constant_s: f64,
    scaled_normal_condition_number: f64,
    log_time_constant_sensitivity_rms_mps: f64,
}

impl IdentifiedWheelPlantV1 {
    pub fn velocity_gain_mps_per_pwm_percent(self) -> f64 {
        self.velocity_gain_mps_per_pwm_percent
    }

    pub fn time_constant_s(self) -> f64 {
        self.time_constant_s
    }

    pub fn scaled_normal_condition_number(self) -> f64 {
        self.scaled_normal_condition_number
    }

    pub fn log_time_constant_sensitivity_rms_mps(self) -> f64 {
        self.log_time_constant_sensitivity_rms_mps
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LateralVelocityEvidence {
    /// This dataset has no independent lateral-velocity measurement. A caller
    /// must supply separate physical evidence before activating an MPC model
    /// that requires a lateral-slip envelope.
    Unidentified,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlantSupportV1 {
    pub left_pwm_min_percent: i8,
    pub left_pwm_max_percent: i8,
    pub right_pwm_min_percent: i8,
    pub right_pwm_max_percent: i8,
    pub left_velocity_min_mps: f64,
    pub left_velocity_max_mps: f64,
    pub right_velocity_min_mps: f64,
    pub right_velocity_max_mps: f64,
    pub max_abs_yaw_rate_rad_s: f64,
    pub lateral_velocity: LateralVelocityEvidence,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HoldoutResidualsV1 {
    pub left_velocity_rmse_mps: f64,
    pub right_velocity_rmse_mps: f64,
    pub forward_velocity_rmse_mps: f64,
    pub yaw_rate_rmse_rad_s: f64,
    pub max_abs_wheel_velocity_error_mps: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IdentifiedPlantV1 {
    schema_version: u32,
    dataset_content_id: BoundedId,
    robot_id: BoundedId,
    controller_session_id: BoundedId,
    visual_velocity_source_id: BoundedId,
    imu_calibration_id: BoundedId,
    wheelbase_calibration_id: BoundedId,
    identification_method_id: BoundedId,
    source_sample_count: u32,
    training_transition_count: u32,
    holdout_transition_count: u32,
    sample_period_s: f64,
    wheelbase_m: f64,
    left: IdentifiedWheelPlantV1,
    right: IdentifiedWheelPlantV1,
    support: PlantSupportV1,
    holdout_residuals: HoldoutResidualsV1,
}

impl IdentifiedPlantV1 {
    pub fn schema_version(self) -> u32 {
        self.schema_version
    }

    pub fn dataset_content_id(self) -> BoundedId {
        self.dataset_content_id
    }

    pub fn robot_id(self) -> BoundedId {
        self.robot_id
    }

    pub fn controller_session_id(self) -> BoundedId {
        self.controller_session_id
    }

    pub fn visual_velocity_source_id(self) -> BoundedId {
        self.visual_velocity_source_id
    }

    pub fn imu_calibration_id(self) -> BoundedId {
        self.imu_calibration_id
    }

    pub fn wheelbase_calibration_id(self) -> BoundedId {
        self.wheelbase_calibration_id
    }

    pub fn identification_method_id(self) -> BoundedId {
        self.identification_method_id
    }

    pub fn source_sample_count(self) -> u32 {
        self.source_sample_count
    }

    pub fn training_transition_count(self) -> u32 {
        self.training_transition_count
    }

    pub fn holdout_transition_count(self) -> u32 {
        self.holdout_transition_count
    }

    pub fn sample_period_s(self) -> f64 {
        self.sample_period_s
    }

    pub fn wheelbase_m(self) -> f64 {
        self.wheelbase_m
    }

    pub fn left(self) -> IdentifiedWheelPlantV1 {
        self.left
    }

    pub fn right(self) -> IdentifiedWheelPlantV1 {
        self.right
    }

    pub fn support(self) -> PlantSupportV1 {
        self.support
    }

    pub fn holdout_residuals(self) -> HoldoutResidualsV1 {
        self.holdout_residuals
    }
}

#[derive(Clone, Copy, Debug)]
struct Transition {
    dt_s: f64,
    left_pwm_percent: i8,
    right_pwm_percent: i8,
    left_velocity_0_mps: f64,
    left_velocity_1_mps: f64,
    right_velocity_0_mps: f64,
    right_velocity_1_mps: f64,
    holdout: bool,
}

#[derive(Clone, Copy, Debug)]
struct WheelFit {
    gain: f64,
    tau_s: f64,
    condition_number: f64,
    log_tau_sensitivity_rms_mps: f64,
}

#[derive(Clone, Copy, Debug)]
struct Objective {
    gain: f64,
    squared_error: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    fn add(&mut self, value: f64) {
        let next = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.correction += (self.sum - next) + value;
        } else {
            self.correction += (value - next) + self.sum;
        }
        self.sum = next;
    }

    fn total(self) -> f64 {
        self.sum + self.correction
    }
}

/// Fits the exact zero-order-hold discretization
/// `v1 = exp(-dt/tau) * v0 + gain * (1-exp(-dt/tau)) * pwm`.
///
/// `1-exp(-dt/tau)` is evaluated with `exp_m1` to avoid cancellation at small
/// sample periods. Gain is solved analytically for every log-time-constant
/// candidate; a fixed grid followed by fixed-iteration golden-section search
/// makes the result deterministic.
pub fn fit_first_order_plant(
    dataset: &IdentificationDatasetV1,
    config: PlantFitConfigV1,
) -> Result<IdentifiedPlantV1, FitError> {
    let PreparedTransitions {
        transitions,
        sample_period_s,
        training_count,
        holdout_count,
        support_counts,
    } = prepare_transitions(dataset, config)?;

    check_coverage(
        CoverageGate::TrainingTransitions,
        training_count,
        config.min_training_transitions.get(),
    )?;
    check_coverage(
        CoverageGate::HoldoutTransitions,
        holdout_count,
        config.min_holdout_transitions.get(),
    )?;
    for (gate, actual, required) in [
        (
            CoverageGate::SymmetricTransitions,
            support_counts.symmetric,
            config.min_symmetric_transitions,
        ),
        (
            CoverageGate::SpinTransitions,
            support_counts.spin,
            config.min_spin_transitions,
        ),
        (
            CoverageGate::ZeroTransitions,
            support_counts.zero,
            config.min_zero_transitions,
        ),
        (
            CoverageGate::LeftPositiveTransitions,
            support_counts.left_positive,
            config.min_positive_transitions_per_wheel,
        ),
        (
            CoverageGate::LeftNegativeTransitions,
            support_counts.left_negative,
            config.min_negative_transitions_per_wheel,
        ),
        (
            CoverageGate::RightPositiveTransitions,
            support_counts.right_positive,
            config.min_positive_transitions_per_wheel,
        ),
        (
            CoverageGate::RightNegativeTransitions,
            support_counts.right_negative,
            config.min_negative_transitions_per_wheel,
        ),
        (
            CoverageGate::CommandChanges,
            support_counts.command_changes,
            config.min_command_changes,
        ),
    ] {
        check_coverage(gate, actual, required)?;
    }

    let left = fit_wheel(WheelSide::Left, &transitions, config)?;
    let right = fit_wheel(WheelSide::Right, &transitions, config)?;
    let residuals = holdout_residuals(&transitions, left, right, config.wheelbase_m)?;
    check_residuals(residuals, config)?;
    let support = build_support(dataset, &transitions, left, right, config.wheelbase_m)?;

    Ok(IdentifiedPlantV1 {
        schema_version: BASE_IDENTIFICATION_V1,
        dataset_content_id: dataset.dataset_content_id,
        robot_id: dataset.robot_id,
        controller_session_id: dataset.controller_session_id,
        visual_velocity_source_id: dataset.visual_velocity_source_id,
        imu_calibration_id: dataset.imu_calibration_id,
        wheelbase_calibration_id: dataset.wheelbase_calibration_id,
        identification_method_id: BoundedId::parse_str(
            "identification_method_id",
            IDENTIFICATION_METHOD_ID,
        )
        .expect("static method identifier is valid"),
        source_sample_count: u32::try_from(dataset.samples.len())
            .expect("dataset parser bounds sample count to u32"),
        training_transition_count: training_count,
        holdout_transition_count: holdout_count,
        sample_period_s,
        wheelbase_m: config.wheelbase_m,
        left: IdentifiedWheelPlantV1 {
            velocity_gain_mps_per_pwm_percent: left.gain,
            time_constant_s: left.tau_s,
            scaled_normal_condition_number: left.condition_number,
            log_time_constant_sensitivity_rms_mps: left.log_tau_sensitivity_rms_mps,
        },
        right: IdentifiedWheelPlantV1 {
            velocity_gain_mps_per_pwm_percent: right.gain,
            time_constant_s: right.tau_s,
            scaled_normal_condition_number: right.condition_number,
            log_time_constant_sensitivity_rms_mps: right.log_tau_sensitivity_rms_mps,
        },
        support,
        holdout_residuals: residuals,
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct SupportCounts {
    symmetric: u32,
    spin: u32,
    zero: u32,
    left_positive: u32,
    left_negative: u32,
    right_positive: u32,
    right_negative: u32,
    command_changes: u32,
}

struct PreparedTransitions {
    transitions: Vec<Transition>,
    sample_period_s: f64,
    training_count: u32,
    holdout_count: u32,
    support_counts: SupportCounts,
}

fn prepare_transitions(
    dataset: &IdentificationDatasetV1,
    config: PlantFitConfigV1,
) -> Result<PreparedTransitions, FitError> {
    let transition_capacity = dataset.samples.len() - 1;
    let mut transitions = allocate_transition_storage(transition_capacity)?;
    let mut sample_period_sum = CompensatedSum::default();
    let mut minimum_sample_period_s = f64::INFINITY;
    let mut maximum_sample_period_s = 0.0_f64;
    let mut support = SupportCounts::default();
    let holdout_stride = usize::from(config.holdout_stride.get());
    let mut command_segment_ordinal = 0_usize;

    for (index, pair) in dataset.samples.windows(2).enumerate() {
        let previous = pair[0];
        let current = pair[1];
        let delta_ns = current
            .observed_at
            .as_nanos()
            .checked_sub(previous.observed_at.as_nanos())
            .ok_or(FitError::NumericalFailure {
                wheel: None,
                stage: NumericalStage::TransitionPeriod,
            })?;
        if delta_ns == 0 {
            return Err(FitError::NumericalFailure {
                wheel: None,
                stage: NumericalStage::TransitionPeriod,
            });
        }
        let dt_s = std::time::Duration::from_nanos(delta_ns).as_secs_f64();
        if delta_ns < config.min_sample_period_ns.get()
            || delta_ns > config.max_sample_period_ns.get()
        {
            return Err(FitError::SamplePeriodOutsideConfiguredRange {
                interval_start_index: index,
                actual_s: dt_s,
                minimum_s: config.min_sample_period_s,
                maximum_s: config.max_sample_period_s,
            });
        }
        sample_period_sum.add(dt_s);
        minimum_sample_period_s = minimum_sample_period_s.min(dt_s);
        maximum_sample_period_s = maximum_sample_period_s.max(dt_s);

        if current.applied_pwm != previous.applied_pwm {
            support.command_changes += 1;
            command_segment_ordinal += 1;
            continue;
        }
        let left = previous.applied_pwm.left().get();
        let right = previous.applied_pwm.right().get();
        let threshold = i16::from(config.min_abs_excitation_pwm_percent);
        let left_i16 = i16::from(left);
        let right_i16 = i16::from(right);
        if left == 0 && right == 0 {
            support.zero += 1;
        }
        if left == right && left_i16.abs() >= threshold {
            support.symmetric += 1;
        }
        if left_i16 == -right_i16 && left_i16.abs() >= threshold {
            support.spin += 1;
        }
        if left_i16 >= threshold {
            support.left_positive += 1;
        }
        if left_i16 <= -threshold {
            support.left_negative += 1;
        }
        if right_i16 >= threshold {
            support.right_positive += 1;
        }
        if right_i16 <= -threshold {
            support.right_negative += 1;
        }
        transitions.push(Transition {
            dt_s,
            left_pwm_percent: left,
            right_pwm_percent: right,
            left_velocity_0_mps: previous.left_velocity_mps,
            left_velocity_1_mps: current.left_velocity_mps,
            right_velocity_0_mps: previous.right_velocity_mps,
            right_velocity_1_mps: current.right_velocity_mps,
            holdout: (command_segment_ordinal + 1).is_multiple_of(holdout_stride),
        });
    }
    let period_count = dataset.samples.len() - 1;
    let period_count = u32::try_from(period_count).expect("sample count is bounded to u32");
    let sample_period_s = sample_period_sum.total() / f64::from(period_count);
    let ratio = maximum_sample_period_s / minimum_sample_period_s;
    if !sample_period_s.is_finite() || !ratio.is_finite() {
        return Err(FitError::NumericalFailure {
            wheel: None,
            stage: NumericalStage::TransitionPeriod,
        });
    }
    if ratio > config.max_sample_period_ratio {
        return Err(FitError::SamplePeriodRatioExceeded {
            minimum_s: minimum_sample_period_s,
            maximum_s: maximum_sample_period_s,
            actual_ratio: ratio,
            maximum_ratio: config.max_sample_period_ratio,
        });
    }
    let holdout_count = u32::try_from(transitions.iter().filter(|item| item.holdout).count())
        .expect("transition count is bounded by parsed u32 sample count");
    let transition_count =
        u32::try_from(transitions.len()).expect("transition count is bounded by u32 samples");
    let training_count = transition_count - holdout_count;
    Ok(PreparedTransitions {
        transitions,
        sample_period_s,
        training_count,
        holdout_count,
        support_counts: support,
    })
}

fn allocate_transition_storage(requested_transitions: usize) -> Result<Vec<Transition>, FitError> {
    let mut transitions = Vec::new();
    transitions
        .try_reserve_exact(requested_transitions)
        .map_err(|source| FitError::TransitionStorageAllocation {
            requested_transitions,
            source,
        })?;
    Ok(transitions)
}

fn check_coverage(gate: CoverageGate, actual: u32, required: u32) -> Result<(), FitError> {
    if actual < required {
        Err(FitError::InsufficientCoverage {
            gate,
            actual,
            required,
        })
    } else {
        Ok(())
    }
}

fn wheel_values(transition: Transition, wheel: WheelSide) -> (f64, f64, f64) {
    match wheel {
        WheelSide::Left => (
            f64::from(transition.left_pwm_percent),
            transition.left_velocity_0_mps,
            transition.left_velocity_1_mps,
        ),
        WheelSide::Right => (
            f64::from(transition.right_pwm_percent),
            transition.right_velocity_0_mps,
            transition.right_velocity_1_mps,
        ),
    }
}

fn objective_for_tau(
    wheel: WheelSide,
    transitions: &[Transition],
    tau_s: f64,
) -> Result<Objective, FitError> {
    if !tau_s.is_finite() || tau_s <= 0.0 {
        return Err(FitError::NumericalFailure {
            wheel: Some(wheel),
            stage: NumericalStage::TimeConstantSearch,
        });
    }
    let mut sum_xx = CompensatedSum::default();
    let mut sum_xy = CompensatedSum::default();
    for transition in transitions.iter().copied().filter(|item| !item.holdout) {
        let (pwm, velocity_0, velocity_1) = wheel_values(transition, wheel);
        let ratio = transition.dt_s / tau_s;
        let decay = (-ratio).exp();
        let response = -(-ratio).exp_m1();
        let x = response * pwm;
        let y = velocity_1 - decay * velocity_0;
        if !ratio.is_finite()
            || !decay.is_finite()
            || !response.is_finite()
            || !x.is_finite()
            || !y.is_finite()
        {
            return Err(FitError::NumericalFailure {
                wheel: Some(wheel),
                stage: NumericalStage::GainNormalEquation,
            });
        }
        sum_xx.add(x * x);
        sum_xy.add(x * y);
    }
    let denominator = sum_xx.total();
    let numerator = sum_xy.total();
    if !denominator.is_finite() || denominator <= 0.0 || !numerator.is_finite() {
        return Err(FitError::NumericalFailure {
            wheel: Some(wheel),
            stage: NumericalStage::GainNormalEquation,
        });
    }
    let gain = numerator / denominator;
    if !gain.is_finite() {
        return Err(FitError::NumericalFailure {
            wheel: Some(wheel),
            stage: NumericalStage::GainNormalEquation,
        });
    }
    let mut squared_error = CompensatedSum::default();
    for transition in transitions.iter().copied().filter(|item| !item.holdout) {
        let (pwm, velocity_0, velocity_1) = wheel_values(transition, wheel);
        let ratio = transition.dt_s / tau_s;
        let decay = (-ratio).exp();
        let response = -(-ratio).exp_m1();
        let residual = velocity_1 - (decay * velocity_0 + gain * response * pwm);
        squared_error.add(residual * residual);
    }
    let squared_error = squared_error.total();
    if !squared_error.is_finite() || squared_error < 0.0 {
        return Err(FitError::NumericalFailure {
            wheel: Some(wheel),
            stage: NumericalStage::ResidualAccumulation,
        });
    }
    Ok(Objective {
        gain,
        squared_error,
    })
}

fn fit_wheel(
    wheel: WheelSide,
    transitions: &[Transition],
    config: PlantFitConfigV1,
) -> Result<WheelFit, FitError> {
    let log_min = config.min_time_constant_s.ln();
    let log_max = config.max_time_constant_s.ln();
    let log_span = log_max - log_min;
    if !log_min.is_finite() || !log_max.is_finite() || !log_span.is_finite() || log_span <= 0.0 {
        return Err(FitError::NumericalFailure {
            wheel: Some(wheel),
            stage: NumericalStage::TimeConstantSearch,
        });
    }
    let grid_intervals =
        f64::from(u32::try_from(LOG_TAU_GRID_INTERVALS).expect("fixed grid fits u32"));
    let grid_step = log_span / grid_intervals;
    let mut best_index = 0_usize;
    let mut best = objective_for_tau(wheel, transitions, log_min.exp())?;
    for index in 1..=LOG_TAU_GRID_INTERVALS {
        let candidate_log = log_min
            + f64::from(u32::try_from(index).expect("fixed grid index fits u32")) * grid_step;
        let candidate = objective_for_tau(wheel, transitions, candidate_log.exp())?;
        if candidate.squared_error < best.squared_error {
            best = candidate;
            best_index = index;
        }
    }
    if best_index == 0 || best_index == LOG_TAU_GRID_INTERVALS {
        let fitted_s = (log_min
            + f64::from(u32::try_from(best_index).expect("fixed grid index fits u32")) * grid_step)
            .exp();
        return Err(FitError::TimeConstantAtSearchBoundary {
            wheel,
            fitted_s,
            minimum_s: config.min_time_constant_s,
            maximum_s: config.max_time_constant_s,
            required_margin_fraction: config.time_constant_bound_margin_fraction,
        });
    }
    let mut lower = log_min
        + f64::from(u32::try_from(best_index - 1).expect("fixed grid index fits u32")) * grid_step;
    let mut upper = log_min
        + f64::from(u32::try_from(best_index + 1).expect("fixed grid index fits u32")) * grid_step;
    let mut left_log = upper - GOLDEN_RATIO_CONJUGATE * (upper - lower);
    let mut right_log = lower + GOLDEN_RATIO_CONJUGATE * (upper - lower);
    let mut left_objective = objective_for_tau(wheel, transitions, left_log.exp())?;
    let mut right_objective = objective_for_tau(wheel, transitions, right_log.exp())?;
    for _ in 0..GOLDEN_SECTION_ITERATIONS {
        if left_objective.squared_error <= right_objective.squared_error {
            upper = right_log;
            right_log = left_log;
            right_objective = left_objective;
            left_log = upper - GOLDEN_RATIO_CONJUGATE * (upper - lower);
            left_objective = objective_for_tau(wheel, transitions, left_log.exp())?;
        } else {
            lower = left_log;
            left_log = right_log;
            left_objective = right_objective;
            right_log = lower + GOLDEN_RATIO_CONJUGATE * (upper - lower);
            right_objective = objective_for_tau(wheel, transitions, right_log.exp())?;
        }
    }
    let (fitted_log_tau, fitted_objective) =
        if left_objective.squared_error <= right_objective.squared_error {
            (left_log, left_objective)
        } else {
            (right_log, right_objective)
        };
    let tau_s = fitted_log_tau.exp();
    let relative_position = (fitted_log_tau - log_min) / log_span;
    let margin = config.time_constant_bound_margin_fraction;
    if !tau_s.is_finite()
        || !relative_position.is_finite()
        || relative_position < margin
        || relative_position > 1.0 - margin
    {
        return Err(FitError::TimeConstantAtSearchBoundary {
            wheel,
            fitted_s: tau_s,
            minimum_s: config.min_time_constant_s,
            maximum_s: config.max_time_constant_s,
            required_margin_fraction: margin,
        });
    }
    let gain = fitted_objective.gain;
    let absolute_gain = gain.abs();
    if absolute_gain < config.min_abs_velocity_gain_mps_per_pwm_percent
        || absolute_gain > config.max_abs_velocity_gain_mps_per_pwm_percent
    {
        return Err(FitError::VelocityGainOutsideConfiguredRange {
            wheel,
            fitted_mps_per_pwm_percent: gain,
            minimum_abs_mps_per_pwm_percent: config.min_abs_velocity_gain_mps_per_pwm_percent,
            maximum_abs_mps_per_pwm_percent: config.max_abs_velocity_gain_mps_per_pwm_percent,
        });
    }
    if config.require_positive_velocity_gain && gain <= 0.0 {
        return Err(FitError::NonPositiveCanonicalVelocityGain {
            wheel,
            fitted_mps_per_pwm_percent: gain,
        });
    }
    let (condition_number, sensitivity_rms) = fit_conditioning(wheel, transitions, gain, tau_s)?;
    if sensitivity_rms < config.min_log_time_constant_sensitivity_mps {
        return Err(FitError::TimeConstantSensitivityTooLow {
            wheel,
            actual_rms_mps: sensitivity_rms,
            minimum_rms_mps: config.min_log_time_constant_sensitivity_mps,
        });
    }
    if condition_number > config.max_normal_matrix_condition_number {
        return Err(FitError::IllConditionedNormalMatrix {
            wheel,
            actual_condition_number: condition_number,
            maximum_condition_number: config.max_normal_matrix_condition_number,
        });
    }
    Ok(WheelFit {
        gain,
        tau_s,
        condition_number,
        log_tau_sensitivity_rms_mps: sensitivity_rms,
    })
}

fn fit_conditioning(
    wheel: WheelSide,
    transitions: &[Transition],
    gain: f64,
    tau_s: f64,
) -> Result<(f64, f64), FitError> {
    let mut gain_energy = CompensatedSum::default();
    let mut tau_energy = CompensatedSum::default();
    let mut cross = CompensatedSum::default();
    let mut count = 0_u32;
    for transition in transitions.iter().copied().filter(|item| !item.holdout) {
        let (pwm, velocity_0, _) = wheel_values(transition, wheel);
        let ratio = transition.dt_s / tau_s;
        let decay = (-ratio).exp();
        let response = -(-ratio).exp_m1();
        let gain_jacobian = response * pwm;
        let log_tau_jacobian = decay * ratio * (velocity_0 - gain * pwm);
        if !gain_jacobian.is_finite() || !log_tau_jacobian.is_finite() {
            return Err(FitError::NumericalFailure {
                wheel: Some(wheel),
                stage: NumericalStage::Conditioning,
            });
        }
        gain_energy.add(gain_jacobian * gain_jacobian);
        tau_energy.add(log_tau_jacobian * log_tau_jacobian);
        cross.add(gain_jacobian * log_tau_jacobian);
        count += 1;
    }
    let gain_energy = gain_energy.total();
    let tau_energy = tau_energy.total();
    let cross = cross.total();
    if gain_energy <= 0.0
        || tau_energy <= 0.0
        || !gain_energy.is_finite()
        || !tau_energy.is_finite()
        || !cross.is_finite()
    {
        return Err(FitError::NumericalFailure {
            wheel: Some(wheel),
            stage: NumericalStage::Conditioning,
        });
    }
    let scale = gain_energy.sqrt() * tau_energy.sqrt();
    let correlation = (cross / scale).abs().min(1.0);
    let condition_number = if correlation >= 1.0 {
        f64::INFINITY
    } else {
        (1.0 + correlation) / (1.0 - correlation)
    };
    let sensitivity_rms = (tau_energy / f64::from(count)).sqrt();
    if !sensitivity_rms.is_finite() {
        return Err(FitError::NumericalFailure {
            wheel: Some(wheel),
            stage: NumericalStage::Conditioning,
        });
    }
    Ok((condition_number, sensitivity_rms))
}

fn predict(velocity_0: f64, pwm: f64, dt_s: f64, fit: WheelFit) -> f64 {
    let ratio = dt_s / fit.tau_s;
    let decay = (-ratio).exp();
    let response = -(-ratio).exp_m1();
    decay * velocity_0 + fit.gain * response * pwm
}

fn holdout_residuals(
    transitions: &[Transition],
    left: WheelFit,
    right: WheelFit,
    wheelbase_m: f64,
) -> Result<HoldoutResidualsV1, FitError> {
    let mut left_squared = CompensatedSum::default();
    let mut right_squared = CompensatedSum::default();
    let mut forward_squared = CompensatedSum::default();
    let mut yaw_squared = CompensatedSum::default();
    let mut max_abs_wheel_error = 0.0_f64;
    let mut count = 0_u32;
    for transition in transitions.iter().copied().filter(|item| item.holdout) {
        let predicted_left = predict(
            transition.left_velocity_0_mps,
            f64::from(transition.left_pwm_percent),
            transition.dt_s,
            left,
        );
        let predicted_right = predict(
            transition.right_velocity_0_mps,
            f64::from(transition.right_pwm_percent),
            transition.dt_s,
            right,
        );
        let left_error = transition.left_velocity_1_mps - predicted_left;
        let right_error = transition.right_velocity_1_mps - predicted_right;
        let forward_error = 0.5 * (left_error + right_error);
        let yaw_error = (right_error - left_error) / wheelbase_m;
        if !predicted_left.is_finite()
            || !predicted_right.is_finite()
            || !left_error.is_finite()
            || !right_error.is_finite()
            || !forward_error.is_finite()
            || !yaw_error.is_finite()
        {
            return Err(FitError::NumericalFailure {
                wheel: None,
                stage: NumericalStage::ResidualAccumulation,
            });
        }
        left_squared.add(left_error * left_error);
        right_squared.add(right_error * right_error);
        forward_squared.add(forward_error * forward_error);
        yaw_squared.add(yaw_error * yaw_error);
        max_abs_wheel_error = max_abs_wheel_error
            .max(left_error.abs())
            .max(right_error.abs());
        count += 1;
    }
    if count == 0 {
        return Err(FitError::InsufficientCoverage {
            gate: CoverageGate::HoldoutTransitions,
            actual: 0,
            required: 1,
        });
    }
    let denominator = f64::from(count);
    let residuals = HoldoutResidualsV1 {
        left_velocity_rmse_mps: (left_squared.total() / denominator).sqrt(),
        right_velocity_rmse_mps: (right_squared.total() / denominator).sqrt(),
        forward_velocity_rmse_mps: (forward_squared.total() / denominator).sqrt(),
        yaw_rate_rmse_rad_s: (yaw_squared.total() / denominator).sqrt(),
        max_abs_wheel_velocity_error_mps: max_abs_wheel_error,
    };
    if [
        residuals.left_velocity_rmse_mps,
        residuals.right_velocity_rmse_mps,
        residuals.forward_velocity_rmse_mps,
        residuals.yaw_rate_rmse_rad_s,
        residuals.max_abs_wheel_velocity_error_mps,
    ]
    .into_iter()
    .all(f64::is_finite)
    {
        Ok(residuals)
    } else {
        Err(FitError::NumericalFailure {
            wheel: None,
            stage: NumericalStage::ResidualAccumulation,
        })
    }
}

fn check_residuals(
    residuals: HoldoutResidualsV1,
    config: PlantFitConfigV1,
) -> Result<(), FitError> {
    for (metric, actual, maximum) in [
        (
            ResidualMetric::LeftWheelVelocityRmse,
            residuals.left_velocity_rmse_mps,
            config.max_holdout_wheel_velocity_rmse_mps,
        ),
        (
            ResidualMetric::RightWheelVelocityRmse,
            residuals.right_velocity_rmse_mps,
            config.max_holdout_wheel_velocity_rmse_mps,
        ),
        (
            ResidualMetric::ForwardVelocityRmse,
            residuals.forward_velocity_rmse_mps,
            config.max_holdout_forward_velocity_rmse_mps,
        ),
        (
            ResidualMetric::YawRateRmse,
            residuals.yaw_rate_rmse_rad_s,
            config.max_holdout_yaw_rate_rmse_rad_s,
        ),
        (
            ResidualMetric::MaximumAbsoluteWheelVelocityError,
            residuals.max_abs_wheel_velocity_error_mps,
            config.max_holdout_abs_wheel_velocity_error_mps,
        ),
    ] {
        if actual > maximum {
            return Err(FitError::HoldoutResidualExceeded {
                metric,
                actual,
                maximum,
            });
        }
    }
    Ok(())
}

fn build_support(
    dataset: &IdentificationDatasetV1,
    transitions: &[Transition],
    left: WheelFit,
    right: WheelFit,
    wheelbase_m: f64,
) -> Result<PlantSupportV1, FitError> {
    let mut left_pwm_min = i8::MAX;
    let mut left_pwm_max = i8::MIN;
    let mut right_pwm_min = i8::MAX;
    let mut right_pwm_max = i8::MIN;
    for transition in transitions {
        let left_pwm = transition.left_pwm_percent;
        let right_pwm = transition.right_pwm_percent;
        left_pwm_min = left_pwm_min.min(left_pwm);
        left_pwm_max = left_pwm_max.max(left_pwm);
        right_pwm_min = right_pwm_min.min(right_pwm);
        right_pwm_max = right_pwm_max.max(right_pwm);
    }
    let first_sample = dataset
        .samples
        .first()
        .expect("parsed identification datasets contain at least three samples");
    let mut left_velocity_min = first_sample.left_velocity_mps;
    let mut left_velocity_max = first_sample.left_velocity_mps;
    let mut right_velocity_min = first_sample.right_velocity_mps;
    let mut right_velocity_max = first_sample.right_velocity_mps;
    for sample in &dataset.samples[1..] {
        left_velocity_min = left_velocity_min.min(sample.left_velocity_mps);
        left_velocity_max = left_velocity_max.max(sample.left_velocity_mps);
        right_velocity_min = right_velocity_min.min(sample.right_velocity_mps);
        right_velocity_max = right_velocity_max.max(sample.right_velocity_mps);
    }
    expand_velocity_range(
        &mut left_velocity_min,
        &mut left_velocity_max,
        left.gain * f64::from(left_pwm_min),
    );
    expand_velocity_range(
        &mut left_velocity_min,
        &mut left_velocity_max,
        left.gain * f64::from(left_pwm_max),
    );
    expand_velocity_range(
        &mut right_velocity_min,
        &mut right_velocity_max,
        right.gain * f64::from(right_pwm_min),
    );
    expand_velocity_range(
        &mut right_velocity_min,
        &mut right_velocity_max,
        right.gain * f64::from(right_pwm_max),
    );
    let max_abs_yaw_rate_rad_s = [
        (right_velocity_min - left_velocity_min).abs(),
        (right_velocity_min - left_velocity_max).abs(),
        (right_velocity_max - left_velocity_min).abs(),
        (right_velocity_max - left_velocity_max).abs(),
    ]
    .into_iter()
    .fold(0.0_f64, f64::max)
        / wheelbase_m;
    if !left_velocity_min.is_finite()
        || !left_velocity_max.is_finite()
        || !right_velocity_min.is_finite()
        || !right_velocity_max.is_finite()
        || !max_abs_yaw_rate_rad_s.is_finite()
        || max_abs_yaw_rate_rad_s <= 0.0
    {
        return Err(FitError::NumericalFailure {
            wheel: None,
            stage: NumericalStage::ValidityEnvelope,
        });
    }
    Ok(PlantSupportV1 {
        left_pwm_min_percent: left_pwm_min,
        left_pwm_max_percent: left_pwm_max,
        right_pwm_min_percent: right_pwm_min,
        right_pwm_max_percent: right_pwm_max,
        left_velocity_min_mps: left_velocity_min,
        left_velocity_max_mps: left_velocity_max,
        right_velocity_min_mps: right_velocity_min,
        right_velocity_max_mps: right_velocity_max,
        max_abs_yaw_rate_rad_s,
        lateral_velocity: LateralVelocityEvidence::Unidentified,
    })
}

fn expand_velocity_range(minimum: &mut f64, maximum: &mut f64, value: f64) {
    *minimum = minimum.min(value);
    *maximum = maximum.max(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn impossible_transition_capacity_is_a_typed_allocation_error() {
        let error = allocate_transition_storage(usize::MAX).expect_err("capacity overflow");
        assert!(matches!(
            error,
            FitError::TransitionStorageAllocation {
                requested_transitions: usize::MAX,
                ..
            }
        ));
    }
}
