use std::collections::TryReserveError;
use std::fmt;
use std::num::{NonZeroU16, NonZeroU32};

use robot_protocol::{AppliedPwm, AppliedPwmError};

use crate::BASE_IDENTIFICATION_V1;
use crate::identity::{BoundedId, IdentifierError};
use crate::time::MonotonicTimestampNs;

const ABSOLUTE_MAX_SAMPLES: u32 = 1_000_000;

/// Weak configuration DTO. Call [`PlantFitConfigV1::parse`] exactly once at
/// the configuration boundary and retain the resulting domain value.
#[derive(Clone, Debug, PartialEq)]
pub struct PlantFitConfigV1Dto {
    pub schema_version: u32,
    pub expected_robot_id: String,
    pub expected_controller_session_id: String,
    pub expected_visual_velocity_source_id: String,
    pub expected_imu_calibration_id: String,
    pub wheelbase_calibration_id: String,
    pub wheelbase_m: f64,
    pub min_sample_period_s: f64,
    pub max_sample_period_s: f64,
    pub max_sample_period_ratio: f64,
    pub max_abs_observed_forward_velocity_mps: f64,
    pub max_abs_observed_yaw_rate_rad_s: f64,
    pub min_samples: u32,
    pub max_samples: u32,
    pub holdout_stride: u16,
    pub min_training_transitions: u32,
    pub min_holdout_transitions: u32,
    pub min_abs_excitation_pwm_percent: u8,
    pub min_symmetric_transitions: u32,
    pub min_spin_transitions: u32,
    pub min_zero_transitions: u32,
    pub min_positive_transitions_per_wheel: u32,
    pub min_negative_transitions_per_wheel: u32,
    pub min_command_changes: u32,
    pub min_time_constant_s: f64,
    pub max_time_constant_s: f64,
    pub time_constant_bound_margin_fraction: f64,
    pub min_abs_velocity_gain_mps_per_pwm_percent: f64,
    pub max_abs_velocity_gain_mps_per_pwm_percent: f64,
    pub require_positive_velocity_gain: bool,
    pub max_normal_matrix_condition_number: f64,
    pub min_log_time_constant_sensitivity_mps: f64,
    pub max_holdout_wheel_velocity_rmse_mps: f64,
    pub max_holdout_forward_velocity_rmse_mps: f64,
    pub max_holdout_yaw_rate_rmse_rad_s: f64,
    pub max_holdout_abs_wheel_velocity_error_mps: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlantFitConfigV1 {
    pub(crate) expected_robot_id: BoundedId,
    pub(crate) expected_controller_session_id: BoundedId,
    pub(crate) expected_visual_velocity_source_id: BoundedId,
    pub(crate) expected_imu_calibration_id: BoundedId,
    pub(crate) wheelbase_calibration_id: BoundedId,
    pub(crate) wheelbase_m: f64,
    pub(crate) min_sample_period_s: f64,
    pub(crate) max_sample_period_s: f64,
    pub(crate) max_sample_period_ratio: f64,
    pub(crate) max_abs_observed_forward_velocity_mps: f64,
    pub(crate) max_abs_observed_yaw_rate_rad_s: f64,
    pub(crate) min_samples: NonZeroU32,
    pub(crate) max_samples: NonZeroU32,
    pub(crate) holdout_stride: NonZeroU16,
    pub(crate) min_training_transitions: NonZeroU32,
    pub(crate) min_holdout_transitions: NonZeroU32,
    pub(crate) min_abs_excitation_pwm_percent: u8,
    pub(crate) min_symmetric_transitions: u32,
    pub(crate) min_spin_transitions: u32,
    pub(crate) min_zero_transitions: u32,
    pub(crate) min_positive_transitions_per_wheel: u32,
    pub(crate) min_negative_transitions_per_wheel: u32,
    pub(crate) min_command_changes: u32,
    pub(crate) min_time_constant_s: f64,
    pub(crate) max_time_constant_s: f64,
    pub(crate) time_constant_bound_margin_fraction: f64,
    pub(crate) min_abs_velocity_gain_mps_per_pwm_percent: f64,
    pub(crate) max_abs_velocity_gain_mps_per_pwm_percent: f64,
    pub(crate) require_positive_velocity_gain: bool,
    pub(crate) max_normal_matrix_condition_number: f64,
    pub(crate) min_log_time_constant_sensitivity_mps: f64,
    pub(crate) max_holdout_wheel_velocity_rmse_mps: f64,
    pub(crate) max_holdout_forward_velocity_rmse_mps: f64,
    pub(crate) max_holdout_yaw_rate_rmse_rad_s: f64,
    pub(crate) max_holdout_abs_wheel_velocity_error_mps: f64,
}

impl PlantFitConfigV1 {
    pub fn parse(dto: PlantFitConfigV1Dto) -> Result<Self, PlantFitConfigParseError> {
        if dto.schema_version != BASE_IDENTIFICATION_V1 {
            return Err(PlantFitConfigParseError::UnsupportedSchemaVersion(
                dto.schema_version,
            ));
        }
        let expected_robot_id = parse_id("expected_robot_id", dto.expected_robot_id)?;
        let expected_controller_session_id = parse_id(
            "expected_controller_session_id",
            dto.expected_controller_session_id,
        )?;
        let expected_visual_velocity_source_id = parse_id(
            "expected_visual_velocity_source_id",
            dto.expected_visual_velocity_source_id,
        )?;
        let expected_imu_calibration_id = parse_id(
            "expected_imu_calibration_id",
            dto.expected_imu_calibration_id,
        )?;
        let wheelbase_calibration_id =
            parse_id("wheelbase_calibration_id", dto.wheelbase_calibration_id)?;

        require_positive("wheelbase_m", dto.wheelbase_m)?;
        require_positive("min_sample_period_s", dto.min_sample_period_s)?;
        require_positive("max_sample_period_s", dto.max_sample_period_s)?;
        if dto.min_sample_period_s > dto.max_sample_period_s {
            return Err(PlantFitConfigParseError::InvertedRange {
                min_field: "min_sample_period_s",
                min: dto.min_sample_period_s,
                max_field: "max_sample_period_s",
                max: dto.max_sample_period_s,
            });
        }
        require_finite("max_sample_period_ratio", dto.max_sample_period_ratio)?;
        if dto.max_sample_period_ratio < 1.0 {
            return Err(PlantFitConfigParseError::LessThanOne {
                field: "max_sample_period_ratio",
                value: dto.max_sample_period_ratio,
            });
        }
        require_positive(
            "max_abs_observed_forward_velocity_mps",
            dto.max_abs_observed_forward_velocity_mps,
        )?;
        require_positive(
            "max_abs_observed_yaw_rate_rad_s",
            dto.max_abs_observed_yaw_rate_rad_s,
        )?;
        if !(3..=ABSOLUTE_MAX_SAMPLES).contains(&dto.min_samples) {
            return Err(PlantFitConfigParseError::IntegerOutOfRange {
                field: "min_samples",
                value: u64::from(dto.min_samples),
                min: 3,
                max: u64::from(ABSOLUTE_MAX_SAMPLES),
            });
        }
        if dto.max_samples < dto.min_samples || dto.max_samples > ABSOLUTE_MAX_SAMPLES {
            return Err(PlantFitConfigParseError::IntegerOutOfRange {
                field: "max_samples",
                value: u64::from(dto.max_samples),
                min: u64::from(dto.min_samples),
                max: u64::from(ABSOLUTE_MAX_SAMPLES),
            });
        }
        if dto.holdout_stride < 2 {
            return Err(PlantFitConfigParseError::IntegerOutOfRange {
                field: "holdout_stride",
                value: u64::from(dto.holdout_stride),
                min: 2,
                max: u64::from(u16::MAX),
            });
        }
        for (field, value) in [
            ("min_training_transitions", dto.min_training_transitions),
            ("min_holdout_transitions", dto.min_holdout_transitions),
        ] {
            if value == 0 || value >= dto.max_samples {
                return Err(PlantFitConfigParseError::IntegerOutOfRange {
                    field,
                    value: u64::from(value),
                    min: 1,
                    max: u64::from(dto.max_samples - 1),
                });
            }
        }
        if !(1..=100).contains(&dto.min_abs_excitation_pwm_percent) {
            return Err(PlantFitConfigParseError::IntegerOutOfRange {
                field: "min_abs_excitation_pwm_percent",
                value: u64::from(dto.min_abs_excitation_pwm_percent),
                min: 1,
                max: 100,
            });
        }
        for (field, value) in [
            ("min_symmetric_transitions", dto.min_symmetric_transitions),
            ("min_spin_transitions", dto.min_spin_transitions),
            ("min_zero_transitions", dto.min_zero_transitions),
            (
                "min_positive_transitions_per_wheel",
                dto.min_positive_transitions_per_wheel,
            ),
            (
                "min_negative_transitions_per_wheel",
                dto.min_negative_transitions_per_wheel,
            ),
            ("min_command_changes", dto.min_command_changes),
        ] {
            if value >= dto.max_samples {
                return Err(PlantFitConfigParseError::IntegerOutOfRange {
                    field,
                    value: u64::from(value),
                    min: 0,
                    max: u64::from(dto.max_samples - 1),
                });
            }
        }
        require_positive("min_time_constant_s", dto.min_time_constant_s)?;
        require_positive("max_time_constant_s", dto.max_time_constant_s)?;
        if dto.min_time_constant_s >= dto.max_time_constant_s {
            return Err(PlantFitConfigParseError::InvertedRange {
                min_field: "min_time_constant_s",
                min: dto.min_time_constant_s,
                max_field: "max_time_constant_s",
                max: dto.max_time_constant_s,
            });
        }
        require_finite(
            "time_constant_bound_margin_fraction",
            dto.time_constant_bound_margin_fraction,
        )?;
        if dto.time_constant_bound_margin_fraction <= 0.0
            || dto.time_constant_bound_margin_fraction >= 0.5
        {
            return Err(PlantFitConfigParseError::OutsideHalfOpenUnitHalf {
                field: "time_constant_bound_margin_fraction",
                value: dto.time_constant_bound_margin_fraction,
            });
        }
        require_positive(
            "min_abs_velocity_gain_mps_per_pwm_percent",
            dto.min_abs_velocity_gain_mps_per_pwm_percent,
        )?;
        require_positive(
            "max_abs_velocity_gain_mps_per_pwm_percent",
            dto.max_abs_velocity_gain_mps_per_pwm_percent,
        )?;
        if dto.min_abs_velocity_gain_mps_per_pwm_percent
            >= dto.max_abs_velocity_gain_mps_per_pwm_percent
        {
            return Err(PlantFitConfigParseError::InvertedRange {
                min_field: "min_abs_velocity_gain_mps_per_pwm_percent",
                min: dto.min_abs_velocity_gain_mps_per_pwm_percent,
                max_field: "max_abs_velocity_gain_mps_per_pwm_percent",
                max: dto.max_abs_velocity_gain_mps_per_pwm_percent,
            });
        }
        require_finite(
            "max_normal_matrix_condition_number",
            dto.max_normal_matrix_condition_number,
        )?;
        if dto.max_normal_matrix_condition_number < 1.0 {
            return Err(PlantFitConfigParseError::LessThanOne {
                field: "max_normal_matrix_condition_number",
                value: dto.max_normal_matrix_condition_number,
            });
        }
        for (field, value) in [
            (
                "min_log_time_constant_sensitivity_mps",
                dto.min_log_time_constant_sensitivity_mps,
            ),
            (
                "max_holdout_wheel_velocity_rmse_mps",
                dto.max_holdout_wheel_velocity_rmse_mps,
            ),
            (
                "max_holdout_forward_velocity_rmse_mps",
                dto.max_holdout_forward_velocity_rmse_mps,
            ),
            (
                "max_holdout_yaw_rate_rmse_rad_s",
                dto.max_holdout_yaw_rate_rmse_rad_s,
            ),
            (
                "max_holdout_abs_wheel_velocity_error_mps",
                dto.max_holdout_abs_wheel_velocity_error_mps,
            ),
        ] {
            require_positive(field, value)?;
        }

        Ok(Self {
            expected_robot_id,
            expected_controller_session_id,
            expected_visual_velocity_source_id,
            expected_imu_calibration_id,
            wheelbase_calibration_id,
            wheelbase_m: dto.wheelbase_m,
            min_sample_period_s: dto.min_sample_period_s,
            max_sample_period_s: dto.max_sample_period_s,
            max_sample_period_ratio: dto.max_sample_period_ratio,
            max_abs_observed_forward_velocity_mps: dto.max_abs_observed_forward_velocity_mps,
            max_abs_observed_yaw_rate_rad_s: dto.max_abs_observed_yaw_rate_rad_s,
            min_samples: NonZeroU32::new(dto.min_samples).expect("validated nonzero"),
            max_samples: NonZeroU32::new(dto.max_samples).expect("validated nonzero"),
            holdout_stride: NonZeroU16::new(dto.holdout_stride).expect("validated nonzero"),
            min_training_transitions: NonZeroU32::new(dto.min_training_transitions)
                .expect("validated nonzero"),
            min_holdout_transitions: NonZeroU32::new(dto.min_holdout_transitions)
                .expect("validated nonzero"),
            min_abs_excitation_pwm_percent: dto.min_abs_excitation_pwm_percent,
            min_symmetric_transitions: dto.min_symmetric_transitions,
            min_spin_transitions: dto.min_spin_transitions,
            min_zero_transitions: dto.min_zero_transitions,
            min_positive_transitions_per_wheel: dto.min_positive_transitions_per_wheel,
            min_negative_transitions_per_wheel: dto.min_negative_transitions_per_wheel,
            min_command_changes: dto.min_command_changes,
            min_time_constant_s: dto.min_time_constant_s,
            max_time_constant_s: dto.max_time_constant_s,
            time_constant_bound_margin_fraction: dto.time_constant_bound_margin_fraction,
            min_abs_velocity_gain_mps_per_pwm_percent: dto
                .min_abs_velocity_gain_mps_per_pwm_percent,
            max_abs_velocity_gain_mps_per_pwm_percent: dto
                .max_abs_velocity_gain_mps_per_pwm_percent,
            require_positive_velocity_gain: dto.require_positive_velocity_gain,
            max_normal_matrix_condition_number: dto.max_normal_matrix_condition_number,
            min_log_time_constant_sensitivity_mps: dto.min_log_time_constant_sensitivity_mps,
            max_holdout_wheel_velocity_rmse_mps: dto.max_holdout_wheel_velocity_rmse_mps,
            max_holdout_forward_velocity_rmse_mps: dto.max_holdout_forward_velocity_rmse_mps,
            max_holdout_yaw_rate_rmse_rad_s: dto.max_holdout_yaw_rate_rmse_rad_s,
            max_holdout_abs_wheel_velocity_error_mps: dto.max_holdout_abs_wheel_velocity_error_mps,
        })
    }

    pub fn wheelbase_m(self) -> f64 {
        self.wheelbase_m
    }

    pub fn expected_robot_id(self) -> BoundedId {
        self.expected_robot_id
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlantFitConfigParseError {
    UnsupportedSchemaVersion(u32),
    InvalidIdentifier(IdentifierError),
    NonFinite {
        field: &'static str,
        value: f64,
    },
    NotPositive {
        field: &'static str,
        value: f64,
    },
    LessThanOne {
        field: &'static str,
        value: f64,
    },
    OutsideHalfOpenUnitHalf {
        field: &'static str,
        value: f64,
    },
    InvertedRange {
        min_field: &'static str,
        min: f64,
        max_field: &'static str,
        max: f64,
    },
    IntegerOutOfRange {
        field: &'static str,
        value: u64,
        min: u64,
        max: u64,
    },
}

impl fmt::Display for PlantFitConfigParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid V1 plant-fit configuration: {self:?}")
    }
}

impl std::error::Error for PlantFitConfigParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidIdentifier(source) => Some(source),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct IdentificationDatasetV1Dto {
    pub schema_version: u32,
    pub dataset_content_id: String,
    pub robot_id: String,
    pub controller_session_id: String,
    pub visual_velocity_source_id: String,
    pub imu_calibration_id: String,
    pub wheelbase_calibration_id: String,
    pub samples: Vec<IdentificationSampleV1Dto>,
}

/// One already time-aligned observation. `applied_command_sequence` is the
/// controller result identity, not a locally requested command identity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IdentificationSampleV1Dto {
    pub observed_at_ns: u64,
    pub applied_command_sequence: u64,
    pub applied_left_pwm_percent: i8,
    pub applied_right_pwm_percent: i8,
    pub visual_forward_velocity_mps: f64,
    pub calibrated_imu_yaw_rate_rad_s: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct IdentificationSampleV1 {
    pub(crate) observed_at: MonotonicTimestampNs,
    pub(crate) applied_command_sequence: u64,
    pub(crate) applied_pwm: AppliedPwm,
    pub(crate) visual_forward_velocity_mps: f64,
    pub(crate) calibrated_imu_yaw_rate_rad_s: f64,
    pub(crate) left_velocity_mps: f64,
    pub(crate) right_velocity_mps: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IdentificationDatasetV1 {
    pub(crate) dataset_content_id: BoundedId,
    pub(crate) robot_id: BoundedId,
    pub(crate) controller_session_id: BoundedId,
    pub(crate) visual_velocity_source_id: BoundedId,
    pub(crate) imu_calibration_id: BoundedId,
    pub(crate) wheelbase_calibration_id: BoundedId,
    pub(crate) samples: Vec<IdentificationSampleV1>,
}

impl IdentificationDatasetV1 {
    pub fn parse(
        dto: IdentificationDatasetV1Dto,
        config: PlantFitConfigV1,
    ) -> Result<Self, DatasetParseError> {
        if dto.schema_version != BASE_IDENTIFICATION_V1 {
            return Err(DatasetParseError::UnsupportedSchemaVersion(
                dto.schema_version,
            ));
        }
        let dataset_content_id = parse_dataset_id("dataset_content_id", dto.dataset_content_id)?;
        let robot_id = parse_dataset_id("robot_id", dto.robot_id)?;
        let controller_session_id =
            parse_dataset_id("controller_session_id", dto.controller_session_id)?;
        let visual_velocity_source_id =
            parse_dataset_id("visual_velocity_source_id", dto.visual_velocity_source_id)?;
        let imu_calibration_id = parse_dataset_id("imu_calibration_id", dto.imu_calibration_id)?;
        let wheelbase_calibration_id =
            parse_dataset_id("wheelbase_calibration_id", dto.wheelbase_calibration_id)?;
        for (field, expected, actual) in [
            ("robot_id", config.expected_robot_id, robot_id),
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
            (
                "wheelbase_calibration_id",
                config.wheelbase_calibration_id,
                wheelbase_calibration_id,
            ),
        ] {
            if actual != expected {
                return Err(DatasetParseError::IdentityMismatch {
                    field,
                    expected: Box::new(expected),
                    actual: Box::new(actual),
                });
            }
        }
        let sample_count = dto.samples.len();
        if sample_count
            < usize::try_from(config.min_samples.get()).expect("u32 fits supported usize targets")
            || sample_count
                > usize::try_from(config.max_samples.get())
                    .expect("u32 fits supported usize targets")
        {
            return Err(DatasetParseError::SampleCountOutOfRange {
                actual: sample_count,
                min: config.min_samples.get(),
                max: config.max_samples.get(),
            });
        }

        let mut samples = allocate_sample_storage(sample_count)?;
        for (index, sample) in dto.samples.into_iter().enumerate() {
            let applied_pwm = AppliedPwm::try_new(
                sample.applied_left_pwm_percent,
                sample.applied_right_pwm_percent,
            )
            .map_err(|source| DatasetParseError::InvalidAppliedPwm { index, source })?;
            require_sample_finite(
                index,
                "visual_forward_velocity_mps",
                sample.visual_forward_velocity_mps,
            )?;
            require_sample_finite(
                index,
                "calibrated_imu_yaw_rate_rad_s",
                sample.calibrated_imu_yaw_rate_rad_s,
            )?;
            for (field, value, max_abs) in [
                (
                    "visual_forward_velocity_mps",
                    sample.visual_forward_velocity_mps,
                    config.max_abs_observed_forward_velocity_mps,
                ),
                (
                    "calibrated_imu_yaw_rate_rad_s",
                    sample.calibrated_imu_yaw_rate_rad_s,
                    config.max_abs_observed_yaw_rate_rad_s,
                ),
            ] {
                if value.abs() > max_abs {
                    return Err(DatasetParseError::ObservationOutsideConfiguredBound {
                        index,
                        field,
                        value,
                        max_abs,
                    });
                }
            }
            if let Some(previous) = samples.last() {
                if sample.observed_at_ns <= previous.observed_at.as_nanos() {
                    return Err(DatasetParseError::NonIncreasingTimestamp {
                        index,
                        previous_ns: previous.observed_at.as_nanos(),
                        current_ns: sample.observed_at_ns,
                    });
                }
                if sample.applied_command_sequence < previous.applied_command_sequence {
                    return Err(DatasetParseError::CommandSequenceRegression {
                        index,
                        previous: previous.applied_command_sequence,
                        current: sample.applied_command_sequence,
                    });
                }
                if sample.applied_command_sequence == previous.applied_command_sequence
                    && applied_pwm != previous.applied_pwm
                {
                    return Err(DatasetParseError::ChangedPwmForSameCommand {
                        index,
                        command_sequence: sample.applied_command_sequence,
                        previous: previous.applied_pwm,
                        current: applied_pwm,
                    });
                }
            }
            let yaw_component = 0.5 * config.wheelbase_m * sample.calibrated_imu_yaw_rate_rad_s;
            let left_velocity_mps = sample.visual_forward_velocity_mps - yaw_component;
            let right_velocity_mps = sample.visual_forward_velocity_mps + yaw_component;
            if !left_velocity_mps.is_finite() || !right_velocity_mps.is_finite() {
                return Err(DatasetParseError::DerivedWheelVelocityNonFinite {
                    index,
                    forward_velocity_mps: sample.visual_forward_velocity_mps,
                    yaw_rate_rad_s: sample.calibrated_imu_yaw_rate_rad_s,
                    wheelbase_m: config.wheelbase_m,
                });
            }
            samples.push(IdentificationSampleV1 {
                observed_at: MonotonicTimestampNs::from_nanos(sample.observed_at_ns),
                applied_command_sequence: sample.applied_command_sequence,
                applied_pwm,
                visual_forward_velocity_mps: sample.visual_forward_velocity_mps,
                calibrated_imu_yaw_rate_rad_s: sample.calibrated_imu_yaw_rate_rad_s,
                left_velocity_mps,
                right_velocity_mps,
            });
        }

        Ok(Self {
            dataset_content_id,
            robot_id,
            controller_session_id,
            visual_velocity_source_id,
            imu_calibration_id,
            wheelbase_calibration_id,
            samples,
        })
    }

    pub fn dataset_content_id(&self) -> BoundedId {
        self.dataset_content_id
    }

    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

#[derive(Debug)]
pub enum DatasetParseError {
    UnsupportedSchemaVersion(u32),
    InvalidIdentifier(IdentifierError),
    IdentityMismatch {
        field: &'static str,
        expected: Box<BoundedId>,
        actual: Box<BoundedId>,
    },
    SampleCountOutOfRange {
        actual: usize,
        min: u32,
        max: u32,
    },
    SampleStorageAllocation {
        requested_samples: usize,
        source: TryReserveError,
    },
    InvalidAppliedPwm {
        index: usize,
        source: AppliedPwmError,
    },
    NonFiniteSample {
        index: usize,
        field: &'static str,
        value: f64,
    },
    ObservationOutsideConfiguredBound {
        index: usize,
        field: &'static str,
        value: f64,
        max_abs: f64,
    },
    NonIncreasingTimestamp {
        index: usize,
        previous_ns: u64,
        current_ns: u64,
    },
    CommandSequenceRegression {
        index: usize,
        previous: u64,
        current: u64,
    },
    ChangedPwmForSameCommand {
        index: usize,
        command_sequence: u64,
        previous: AppliedPwm,
        current: AppliedPwm,
    },
    DerivedWheelVelocityNonFinite {
        index: usize,
        forward_velocity_mps: f64,
        yaw_rate_rad_s: f64,
        wheelbase_m: f64,
    },
}

impl fmt::Display for DatasetParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid V1 identification dataset: {self:?}")
    }
}

impl std::error::Error for DatasetParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidIdentifier(source) => Some(source),
            Self::InvalidAppliedPwm { source, .. } => Some(source),
            Self::SampleStorageAllocation { source, .. } => Some(source),
            _ => None,
        }
    }
}

fn allocate_sample_storage(
    requested_samples: usize,
) -> Result<Vec<IdentificationSampleV1>, DatasetParseError> {
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(requested_samples)
        .map_err(|source| DatasetParseError::SampleStorageAllocation {
            requested_samples,
            source,
        })?;
    Ok(samples)
}

fn parse_id(field: &'static str, value: String) -> Result<BoundedId, PlantFitConfigParseError> {
    BoundedId::parse(field, value).map_err(PlantFitConfigParseError::InvalidIdentifier)
}

fn parse_dataset_id(field: &'static str, value: String) -> Result<BoundedId, DatasetParseError> {
    BoundedId::parse(field, value).map_err(DatasetParseError::InvalidIdentifier)
}

fn require_finite(field: &'static str, value: f64) -> Result<(), PlantFitConfigParseError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(PlantFitConfigParseError::NonFinite { field, value })
    }
}

fn require_positive(field: &'static str, value: f64) -> Result<(), PlantFitConfigParseError> {
    require_finite(field, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(PlantFitConfigParseError::NotPositive { field, value })
    }
}

fn require_sample_finite(
    index: usize,
    field: &'static str,
    value: f64,
) -> Result<(), DatasetParseError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(DatasetParseError::NonFiniteSample {
            index,
            field,
            value,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn impossible_sample_capacity_is_a_typed_allocation_error() {
        let error = allocate_sample_storage(usize::MAX).expect_err("capacity overflow");
        assert!(matches!(
            error,
            DatasetParseError::SampleStorageAllocation {
                requested_samples: usize::MAX,
                ..
            }
        ));
    }
}
