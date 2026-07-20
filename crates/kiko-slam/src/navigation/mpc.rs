//! Fixed-order, hardware-independent shadow MPC for canonical wheel PWM.
//!
//! Weak DTOs are parsed once into versioned domain values. The solver has no
//! transport or actuator access. It predicts the continuous unequal-time-
//! constant wheel model and asks a caller-owned immutable collision snapshot
//! about conservative capsules around bounded midpoint integration segments.

use std::fmt;
use std::num::{NonZeroU32, NonZeroU64};

use robot_protocol::{PwmPercent, PwmPercentError};

use crate::{DeviceSessionId, HostMonotonicTimestamp, MapSnapshot};

use super::frames::{OdomFrame, OdomToLocalCostmap, PlanarPoint, PlanarTransformError};
use super::global_planner::{GlobalPath, GlobalPlanIdentity};
use super::local_costmap::{
    LocalCostmapCell, LocalCostmapFreshness, LocalCostmapProvenance, LocalCostmapView,
};
use super::odometry::OdomSegmentId;
use super::shadow_command::ShadowPwmPair;

pub const PLANT_MODEL_V1: u32 = 1;
pub const MPC_CONFIG_V1: u32 = 1;
pub const ODOM_MOTION_STATE_V1: u32 = 1;
pub const MPC_REFERENCE_V1: u32 = 1;
pub const MPC_REQUEST_V1: u32 = 1;
pub const COLLISION_SNAPSHOT_V1: u32 = 1;
pub const NAVIGATION_EPOCH_V1: u32 = 1;
pub const MAX_HORIZON_STEPS: u16 = 128;
pub const MAX_INTEGRATION_SUBSTEPS: u16 = 64;
pub const MAX_OPTIMIZATION_ITERATIONS: u8 = 16;
pub const MIN_CANDIDATES_PER_WHEEL: u8 = 3;
pub const MAX_CANDIDATES_PER_WHEEL: u8 = 9;
pub const MAX_ROLLOUT_EVALUATIONS: u64 = 250_000;
pub const MIN_STEP_PERIOD_S: f64 = 0.000_1;
pub const MAX_STEP_PERIOD_S: f64 = 1.0;
pub const MAX_SUBSTEP_TO_TAU_RATIO: f64 = 0.25;
pub const MAX_SUPPORTED_ABS_ODOM_COORDINATE_M: f64 = 1_000_000.0;
pub const MAX_SUPPORTED_ABS_INPUT_YAW_RAD: f64 = std::f64::consts::TAU;
pub const MAX_SUPPORTED_YAW_EXCURSION_PER_SUBSTEP_RAD: f64 = 0.25;
const MAX_ID_BYTES: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoundedId {
    bytes: [u8; MAX_ID_BYTES],
    len: u8,
}

impl BoundedId {
    fn parse(field: &'static str, value: String) -> Result<Self, IdentifierError> {
        if value.is_empty()
            || value.len() > MAX_ID_BYTES
            || !value.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':' | b'/')
            })
        {
            return Err(IdentifierError { field });
        }
        let mut bytes = [0; MAX_ID_BYTES];
        bytes[..value.len()].copy_from_slice(value.as_bytes());
        Ok(Self {
            bytes,
            len: value.len() as u8,
        })
    }

    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.bytes[..usize::from(self.len)])
            .expect("BoundedId contains checked ASCII")
    }
}

impl fmt::Debug for BoundedId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("BoundedId")
            .field(&self.as_str())
            .finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IdentifierError {
    pub field: &'static str,
}

impl fmt::Display for IdentifierError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid bounded identifier in {}", self.field)
    }
}
impl std::error::Error for IdentifierError {}

#[derive(Clone, Debug, PartialEq)]
pub struct PlantModelV1Dto {
    pub schema_version: u32,
    pub model_id: String,
    pub model_version: u32,
    pub sample_period_s: f64,
    pub wheelbase_m: f64,
    pub left: WheelPlantV1Dto,
    pub right: WheelPlantV1Dto,
    pub validity: PlantValidityEnvelopeV1Dto,
    pub evidence: PlantEvidenceV1Dto,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WheelPlantV1Dto {
    /// Signed steady velocity gain in m/s per canonical PWM percent.
    /// The sign captures motor wiring; no separate channel convention exists.
    pub velocity_gain_mps_per_pwm_percent: f64,
    pub time_constant_s: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlantValidityEnvelopeV1Dto {
    pub left_pwm_min_percent: i8,
    pub left_pwm_max_percent: i8,
    pub right_pwm_min_percent: i8,
    pub right_pwm_max_percent: i8,
    pub left_velocity_min_mps: f64,
    pub left_velocity_max_mps: f64,
    pub right_velocity_min_mps: f64,
    pub right_velocity_max_mps: f64,
    pub max_abs_yaw_rate_rad_s: f64,
    pub max_abs_lateral_velocity_mps: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitResidualsV1Dto {
    pub left_velocity_rmse_mps: f64,
    pub right_velocity_rmse_mps: f64,
    pub yaw_rate_rmse_rad_s: f64,
    pub max_abs_velocity_error_mps: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PlantEvidenceV1Dto {
    SyntheticFixture {
        fixture_id: String,
        generator_id: String,
    },
    /// Caller-asserted metadata. Parsing does not verify the dataset or robot.
    ClaimedPhysicalIdentification {
        dataset_content_id: String,
        identification_method_id: String,
        sample_count: u64,
        residuals: FitResidualsV1Dto,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitResidualsV1 {
    pub left_velocity_rmse_mps: f64,
    pub right_velocity_rmse_mps: f64,
    pub yaw_rate_rmse_rad_s: f64,
    pub max_abs_velocity_error_mps: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlantEvidenceV1 {
    SyntheticFixture {
        fixture_id: BoundedId,
        generator_id: BoundedId,
    },
    ClaimedPhysicalIdentification {
        dataset_content_id: BoundedId,
        identification_method_id: BoundedId,
        sample_count: NonZeroU64,
        residuals: FitResidualsV1,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct FiniteRange {
    min: f64,
    max: f64,
}

impl FiniteRange {
    fn contains(self, value: f64) -> bool {
        self.min <= value && value <= self.max
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PwmRange {
    min: PwmPercent,
    max: PwmPercent,
}

impl PwmRange {
    fn contains(self, value: PwmPercent) -> bool {
        self.min.get() <= value.get() && value.get() <= self.max.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct WheelPlantV1 {
    gain_mps_per_pwm_percent: f64,
    time_constant_s: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlantModelV1 {
    schema_version: u32,
    model_id: BoundedId,
    model_version: NonZeroU32,
    sample_period_s: f64,
    wheelbase_m: f64,
    left: WheelPlantV1,
    right: WheelPlantV1,
    left_pwm: PwmRange,
    right_pwm: PwmRange,
    left_velocity: FiniteRange,
    right_velocity: FiniteRange,
    max_abs_yaw_rate_rad_s: f64,
    max_abs_lateral_velocity_mps: f64,
    evidence: PlantEvidenceV1,
}

impl PlantModelV1 {
    pub fn parse(dto: PlantModelV1Dto) -> Result<Self, PlantModelParseError> {
        if dto.schema_version != PLANT_MODEL_V1 {
            return Err(PlantModelParseError::UnsupportedSchemaVersion(
                dto.schema_version,
            ));
        }
        let model_id = BoundedId::parse("model_id", dto.model_id)
            .map_err(PlantModelParseError::InvalidIdentifier)?;
        let model_version =
            NonZeroU32::new(dto.model_version).ok_or(PlantModelParseError::ZeroModelVersion)?;
        require_positive_plant("sample_period_s", dto.sample_period_s)?;
        require_positive_plant("wheelbase_m", dto.wheelbase_m)?;
        let left = parse_wheel(WheelSide::Left, dto.left)?;
        let right = parse_wheel(WheelSide::Right, dto.right)?;
        let left_pwm = parse_plant_pwm_range(
            WheelSide::Left,
            dto.validity.left_pwm_min_percent,
            dto.validity.left_pwm_max_percent,
        )?;
        let right_pwm = parse_plant_pwm_range(
            WheelSide::Right,
            dto.validity.right_pwm_min_percent,
            dto.validity.right_pwm_max_percent,
        )?;
        let left_velocity = parse_plant_finite_range(
            "left_velocity_min_mps",
            dto.validity.left_velocity_min_mps,
            "left_velocity_max_mps",
            dto.validity.left_velocity_max_mps,
        )?;
        let right_velocity = parse_plant_finite_range(
            "right_velocity_min_mps",
            dto.validity.right_velocity_min_mps,
            "right_velocity_max_mps",
            dto.validity.right_velocity_max_mps,
        )?;
        require_positive_plant(
            "max_abs_yaw_rate_rad_s",
            dto.validity.max_abs_yaw_rate_rad_s,
        )?;
        require_nonnegative_plant(
            "max_abs_lateral_velocity_mps",
            dto.validity.max_abs_lateral_velocity_mps,
        )?;
        for (side, range, wheel, velocity) in [
            (WheelSide::Left, left_pwm, left, left_velocity),
            (WheelSide::Right, right_pwm, right, right_velocity),
        ] {
            for command in [range.min, range.max] {
                let target = wheel.gain_mps_per_pwm_percent * f64::from(command.get());
                if !velocity.contains(target) {
                    return Err(PlantModelParseError::SteadyStateOutsideVelocityEnvelope {
                        wheel: side,
                        command,
                        target_mps: target,
                    });
                }
            }
        }
        let evidence = parse_evidence(dto.evidence)?;
        Ok(Self {
            schema_version: PLANT_MODEL_V1,
            model_id,
            model_version,
            sample_period_s: dto.sample_period_s,
            wheelbase_m: dto.wheelbase_m,
            left,
            right,
            left_pwm,
            right_pwm,
            left_velocity,
            right_velocity,
            max_abs_yaw_rate_rad_s: dto.validity.max_abs_yaw_rate_rad_s,
            max_abs_lateral_velocity_mps: dto.validity.max_abs_lateral_velocity_mps,
            evidence,
        })
    }

    pub fn model_id(self) -> BoundedId {
        self.model_id
    }
    pub fn model_version(self) -> NonZeroU32 {
        self.model_version
    }
    pub fn sample_period_s(self) -> f64 {
        self.sample_period_s
    }
    pub fn wheelbase_m(self) -> f64 {
        self.wheelbase_m
    }
    pub fn evidence(self) -> PlantEvidenceV1 {
        self.evidence
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelSide {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlantModelParseError {
    UnsupportedSchemaVersion(u32),
    InvalidIdentifier(IdentifierError),
    ZeroModelVersion,
    InvalidPwm {
        wheel: WheelSide,
        source: PwmPercentError,
    },
    InvalidPwmRange {
        wheel: WheelSide,
        min: i8,
        max: i8,
    },
    NonFinite {
        field: &'static str,
        value: f64,
    },
    NotPositive {
        field: &'static str,
        value: f64,
    },
    ZeroVelocityGain {
        wheel: WheelSide,
    },
    Negative {
        field: &'static str,
        value: f64,
    },
    InvalidRange {
        min_field: &'static str,
        max_field: &'static str,
        min: f64,
        max: f64,
    },
    ZeroPhysicalSampleCount,
    SteadyStateOutsideVelocityEnvelope {
        wheel: WheelSide,
        command: PwmPercent,
        target_mps: f64,
    },
}

impl fmt::Display for PlantModelParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid V1 plant model: {self:?}")
    }
}
impl std::error::Error for PlantModelParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidIdentifier(source) => Some(source),
            Self::InvalidPwm { source, .. } => Some(source),
            _ => None,
        }
    }
}

fn parse_wheel(
    wheel: WheelSide,
    dto: WheelPlantV1Dto,
) -> Result<WheelPlantV1, PlantModelParseError> {
    let (gain_field, tau_field) = match wheel {
        WheelSide::Left => (
            "left.velocity_gain_mps_per_pwm_percent",
            "left.time_constant_s",
        ),
        WheelSide::Right => (
            "right.velocity_gain_mps_per_pwm_percent",
            "right.time_constant_s",
        ),
    };
    require_finite_plant(gain_field, dto.velocity_gain_mps_per_pwm_percent)?;
    if dto.velocity_gain_mps_per_pwm_percent == 0.0 {
        return Err(PlantModelParseError::ZeroVelocityGain { wheel });
    }
    require_positive_plant(tau_field, dto.time_constant_s)?;
    Ok(WheelPlantV1 {
        gain_mps_per_pwm_percent: dto.velocity_gain_mps_per_pwm_percent,
        time_constant_s: dto.time_constant_s,
    })
}

fn parse_plant_pwm_range(
    wheel: WheelSide,
    min: i8,
    max: i8,
) -> Result<PwmRange, PlantModelParseError> {
    let min_value = PwmPercent::try_new(min)
        .map_err(|source| PlantModelParseError::InvalidPwm { wheel, source })?;
    let max_value = PwmPercent::try_new(max)
        .map_err(|source| PlantModelParseError::InvalidPwm { wheel, source })?;
    if min >= max || min > 0 || max < 0 {
        return Err(PlantModelParseError::InvalidPwmRange { wheel, min, max });
    }
    Ok(PwmRange {
        min: min_value,
        max: max_value,
    })
}

fn parse_plant_finite_range(
    min_field: &'static str,
    min: f64,
    max_field: &'static str,
    max: f64,
) -> Result<FiniteRange, PlantModelParseError> {
    require_finite_plant(min_field, min)?;
    require_finite_plant(max_field, max)?;
    if min >= max || min > 0.0 || max < 0.0 {
        return Err(PlantModelParseError::InvalidRange {
            min_field,
            max_field,
            min,
            max,
        });
    }
    Ok(FiniteRange { min, max })
}

fn parse_evidence(dto: PlantEvidenceV1Dto) -> Result<PlantEvidenceV1, PlantModelParseError> {
    match dto {
        PlantEvidenceV1Dto::SyntheticFixture {
            fixture_id,
            generator_id,
        } => Ok(PlantEvidenceV1::SyntheticFixture {
            fixture_id: BoundedId::parse("fixture_id", fixture_id)
                .map_err(PlantModelParseError::InvalidIdentifier)?,
            generator_id: BoundedId::parse("generator_id", generator_id)
                .map_err(PlantModelParseError::InvalidIdentifier)?,
        }),
        PlantEvidenceV1Dto::ClaimedPhysicalIdentification {
            dataset_content_id,
            identification_method_id,
            sample_count,
            residuals,
        } => {
            for (field, value) in [
                ("left_velocity_rmse_mps", residuals.left_velocity_rmse_mps),
                ("right_velocity_rmse_mps", residuals.right_velocity_rmse_mps),
                ("yaw_rate_rmse_rad_s", residuals.yaw_rate_rmse_rad_s),
                (
                    "max_abs_velocity_error_mps",
                    residuals.max_abs_velocity_error_mps,
                ),
            ] {
                require_nonnegative_plant(field, value)?;
            }
            Ok(PlantEvidenceV1::ClaimedPhysicalIdentification {
                dataset_content_id: BoundedId::parse("dataset_content_id", dataset_content_id)
                    .map_err(PlantModelParseError::InvalidIdentifier)?,
                identification_method_id: BoundedId::parse(
                    "identification_method_id",
                    identification_method_id,
                )
                .map_err(PlantModelParseError::InvalidIdentifier)?,
                sample_count: NonZeroU64::new(sample_count)
                    .ok_or(PlantModelParseError::ZeroPhysicalSampleCount)?,
                residuals: FitResidualsV1 {
                    left_velocity_rmse_mps: residuals.left_velocity_rmse_mps,
                    right_velocity_rmse_mps: residuals.right_velocity_rmse_mps,
                    yaw_rate_rmse_rad_s: residuals.yaw_rate_rmse_rad_s,
                    max_abs_velocity_error_mps: residuals.max_abs_velocity_error_mps,
                },
            })
        }
    }
}

fn require_finite_plant(field: &'static str, value: f64) -> Result<(), PlantModelParseError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(PlantModelParseError::NonFinite { field, value })
    }
}
fn require_positive_plant(field: &'static str, value: f64) -> Result<(), PlantModelParseError> {
    require_finite_plant(field, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(PlantModelParseError::NotPositive { field, value })
    }
}
fn require_nonnegative_plant(field: &'static str, value: f64) -> Result<(), PlantModelParseError> {
    require_finite_plant(field, value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(PlantModelParseError::Negative { field, value })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MpcConfigV1Dto {
    pub schema_version: u32,
    pub horizon_steps: u16,
    pub step_period_s: f64,
    pub integration_substeps: u16,
    pub optimization_iterations: u8,
    pub candidates_per_wheel: u8,
    pub max_rollout_evaluations: u64,
    pub initial_search_radius_percent: u8,
    pub search_radius_decay_numerator: u8,
    pub search_radius_decay_denominator: u8,
    pub left_pwm_min_percent: i8,
    pub left_pwm_max_percent: i8,
    pub right_pwm_min_percent: i8,
    pub right_pwm_max_percent: i8,
    pub left_max_slew_percent_per_step: u16,
    pub right_max_slew_percent_per_step: u16,
    pub max_integration_tube_radius_m: f64,
    pub position_cost_per_m2: f64,
    pub heading_cost_per_rad2: f64,
    pub forward_velocity_cost_s2_per_m2: f64,
    pub yaw_rate_cost_s2_per_rad2: f64,
    pub pwm_cost_per_percent2: f64,
    pub slew_cost_per_percent2: f64,
    pub terminal_state_cost_multiplier: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct CostWeightsV1 {
    position: f64,
    heading: f64,
    forward_velocity: f64,
    yaw_rate: f64,
    pwm: f64,
    slew: f64,
    terminal: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MpcConfigV1 {
    schema_version: u32,
    horizon: usize,
    dt_s: f64,
    integration_substeps: usize,
    iterations: u8,
    lattice: u8,
    evaluation_limit: u64,
    initial_search_radius_percent: u8,
    radius_decay_numerator: u8,
    radius_decay_denominator: u8,
    left_pwm: PwmRange,
    right_pwm: PwmRange,
    left_slew_percent_per_step: u16,
    right_slew_percent_per_step: u16,
    max_integration_tube_radius_m: f64,
    weights: CostWeightsV1,
}

impl MpcConfigV1 {
    pub fn parse(dto: MpcConfigV1Dto) -> Result<Self, MpcConfigParseError> {
        if dto.schema_version != MPC_CONFIG_V1 {
            return Err(MpcConfigParseError::UnsupportedSchemaVersion(
                dto.schema_version,
            ));
        }
        if !(1..=MAX_HORIZON_STEPS).contains(&dto.horizon_steps) {
            return Err(MpcConfigParseError::HorizonOutOfBounds(dto.horizon_steps));
        }
        if !(1..=MAX_INTEGRATION_SUBSTEPS).contains(&dto.integration_substeps) {
            return Err(MpcConfigParseError::IntegrationSubstepsOutOfBounds(
                dto.integration_substeps,
            ));
        }
        if !(1..=MAX_OPTIMIZATION_ITERATIONS).contains(&dto.optimization_iterations) {
            return Err(MpcConfigParseError::IterationsOutOfBounds(
                dto.optimization_iterations,
            ));
        }
        if !(MIN_CANDIDATES_PER_WHEEL..=MAX_CANDIDATES_PER_WHEEL)
            .contains(&dto.candidates_per_wheel)
            || dto.candidates_per_wheel.is_multiple_of(2)
        {
            return Err(MpcConfigParseError::CandidateLatticeOutOfBounds(
                dto.candidates_per_wheel,
            ));
        }
        if dto.max_rollout_evaluations == 0 || dto.max_rollout_evaluations > MAX_ROLLOUT_EVALUATIONS
        {
            return Err(MpcConfigParseError::EvaluationLimitOutOfBounds(
                dto.max_rollout_evaluations,
            ));
        }
        require_positive_config("step_period_s", dto.step_period_s)?;
        if !(MIN_STEP_PERIOD_S..=MAX_STEP_PERIOD_S).contains(&dto.step_period_s) {
            return Err(MpcConfigParseError::ScalarOutOfBounds {
                field: "step_period_s",
                value: dto.step_period_s,
            });
        }
        if dto.initial_search_radius_percent == 0 || dto.initial_search_radius_percent > 100 {
            return Err(MpcConfigParseError::SearchRadiusOutOfBounds(
                dto.initial_search_radius_percent,
            ));
        }
        if dto.search_radius_decay_numerator == 0
            || dto.search_radius_decay_denominator == 0
            || dto.search_radius_decay_numerator > dto.search_radius_decay_denominator
        {
            return Err(MpcConfigParseError::InvalidSearchRadiusDecay {
                numerator: dto.search_radius_decay_numerator,
                denominator: dto.search_radius_decay_denominator,
            });
        }
        let left_pwm = parse_config_pwm_range(
            WheelSide::Left,
            dto.left_pwm_min_percent,
            dto.left_pwm_max_percent,
        )?;
        let right_pwm = parse_config_pwm_range(
            WheelSide::Right,
            dto.right_pwm_min_percent,
            dto.right_pwm_max_percent,
        )?;
        if dto.left_max_slew_percent_per_step == 0 || dto.left_max_slew_percent_per_step > 200 {
            return Err(MpcConfigParseError::SlewOutOfBounds {
                wheel: WheelSide::Left,
                value: dto.left_max_slew_percent_per_step,
            });
        }
        if dto.right_max_slew_percent_per_step == 0 || dto.right_max_slew_percent_per_step > 200 {
            return Err(MpcConfigParseError::SlewOutOfBounds {
                wheel: WheelSide::Right,
                value: dto.right_max_slew_percent_per_step,
            });
        }
        require_positive_config(
            "max_integration_tube_radius_m",
            dto.max_integration_tube_radius_m,
        )?;
        let cost_fields = [
            ("position_cost_per_m2", dto.position_cost_per_m2),
            ("heading_cost_per_rad2", dto.heading_cost_per_rad2),
            (
                "forward_velocity_cost_s2_per_m2",
                dto.forward_velocity_cost_s2_per_m2,
            ),
            ("yaw_rate_cost_s2_per_rad2", dto.yaw_rate_cost_s2_per_rad2),
            ("pwm_cost_per_percent2", dto.pwm_cost_per_percent2),
            ("slew_cost_per_percent2", dto.slew_cost_per_percent2),
        ];
        for (field, value) in cost_fields {
            require_nonnegative_config(field, value)?;
        }
        if dto.position_cost_per_m2 == 0.0
            && dto.heading_cost_per_rad2 == 0.0
            && dto.forward_velocity_cost_s2_per_m2 == 0.0
            && dto.yaw_rate_cost_s2_per_rad2 == 0.0
        {
            return Err(MpcConfigParseError::NoTrackingCost);
        }
        require_positive_config(
            "terminal_state_cost_multiplier",
            dto.terminal_state_cost_multiplier,
        )?;
        for (field, weight) in cost_fields {
            let root = weight.sqrt();
            if !root.is_finite() {
                return Err(MpcConfigParseError::UnrepresentableWeightRoot {
                    field,
                    value: weight,
                });
            }
        }
        let terminal_weights = [
            ("position_cost_per_m2", dto.position_cost_per_m2),
            ("heading_cost_per_rad2", dto.heading_cost_per_rad2),
            (
                "forward_velocity_cost_s2_per_m2",
                dto.forward_velocity_cost_s2_per_m2,
            ),
            ("yaw_rate_cost_s2_per_rad2", dto.yaw_rate_cost_s2_per_rad2),
        ];
        for (field, weight) in terminal_weights {
            if weight != 0.0
                && (weight * dto.terminal_state_cost_multiplier)
                    .sqrt()
                    .is_infinite()
            {
                return Err(MpcConfigParseError::UnrepresentableTerminalWeight { field });
            }
        }
        Ok(Self {
            schema_version: MPC_CONFIG_V1,
            horizon: usize::from(dto.horizon_steps),
            dt_s: dto.step_period_s,
            integration_substeps: usize::from(dto.integration_substeps),
            iterations: dto.optimization_iterations,
            lattice: dto.candidates_per_wheel,
            evaluation_limit: dto.max_rollout_evaluations,
            initial_search_radius_percent: dto.initial_search_radius_percent,
            radius_decay_numerator: dto.search_radius_decay_numerator,
            radius_decay_denominator: dto.search_radius_decay_denominator,
            left_pwm,
            right_pwm,
            left_slew_percent_per_step: dto.left_max_slew_percent_per_step,
            right_slew_percent_per_step: dto.right_max_slew_percent_per_step,
            max_integration_tube_radius_m: dto.max_integration_tube_radius_m,
            weights: CostWeightsV1 {
                position: dto.position_cost_per_m2,
                heading: dto.heading_cost_per_rad2,
                forward_velocity: dto.forward_velocity_cost_s2_per_m2,
                yaw_rate: dto.yaw_rate_cost_s2_per_rad2,
                pwm: dto.pwm_cost_per_percent2,
                slew: dto.slew_cost_per_percent2,
                terminal: dto.terminal_state_cost_multiplier,
            },
        })
    }

    pub fn horizon_steps(self) -> usize {
        self.horizon
    }
    pub fn step_period_s(self) -> f64 {
        self.dt_s
    }
    pub fn integration_substeps(self) -> usize {
        self.integration_substeps
    }
    pub fn max_rollout_evaluations(self) -> u64 {
        self.evaluation_limit
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MpcConfigParseError {
    UnsupportedSchemaVersion(u32),
    HorizonOutOfBounds(u16),
    IntegrationSubstepsOutOfBounds(u16),
    IterationsOutOfBounds(u8),
    CandidateLatticeOutOfBounds(u8),
    EvaluationLimitOutOfBounds(u64),
    SearchRadiusOutOfBounds(u8),
    InvalidSearchRadiusDecay {
        numerator: u8,
        denominator: u8,
    },
    InvalidPwm {
        wheel: WheelSide,
        source: PwmPercentError,
    },
    InvalidPwmRange {
        wheel: WheelSide,
        min: i8,
        max: i8,
    },
    SlewOutOfBounds {
        wheel: WheelSide,
        value: u16,
    },
    NonFinite {
        field: &'static str,
        value: f64,
    },
    NotPositive {
        field: &'static str,
        value: f64,
    },
    Negative {
        field: &'static str,
        value: f64,
    },
    ScalarOutOfBounds {
        field: &'static str,
        value: f64,
    },
    NoTrackingCost,
    UnrepresentableWeightRoot {
        field: &'static str,
        value: f64,
    },
    UnrepresentableTerminalWeight {
        field: &'static str,
    },
}

impl fmt::Display for MpcConfigParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid V1 MPC config: {self:?}")
    }
}
impl std::error::Error for MpcConfigParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidPwm { source, .. } => Some(source),
            _ => None,
        }
    }
}

fn parse_config_pwm_range(
    wheel: WheelSide,
    min: i8,
    max: i8,
) -> Result<PwmRange, MpcConfigParseError> {
    let min_value = PwmPercent::try_new(min)
        .map_err(|source| MpcConfigParseError::InvalidPwm { wheel, source })?;
    let max_value = PwmPercent::try_new(max)
        .map_err(|source| MpcConfigParseError::InvalidPwm { wheel, source })?;
    if min >= max || min > 0 || max < 0 {
        return Err(MpcConfigParseError::InvalidPwmRange { wheel, min, max });
    }
    Ok(PwmRange {
        min: min_value,
        max: max_value,
    })
}

fn require_finite_config(field: &'static str, value: f64) -> Result<(), MpcConfigParseError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(MpcConfigParseError::NonFinite { field, value })
    }
}
fn require_positive_config(field: &'static str, value: f64) -> Result<(), MpcConfigParseError> {
    require_finite_config(field, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(MpcConfigParseError::NotPositive { field, value })
    }
}
fn require_nonnegative_config(field: &'static str, value: f64) -> Result<(), MpcConfigParseError> {
    require_finite_config(field, value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(MpcConfigParseError::Negative { field, value })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NavigationEpochV1 {
    schema_version: u32,
    device_session_id: DeviceSessionId,
    odom_segment_id: OdomSegmentId,
    map_snapshot: MapSnapshot,
    global_plan_identity: GlobalPlanIdentity,
}

impl NavigationEpochV1 {
    pub fn from_runtime(
        device_session_id: DeviceSessionId,
        odom_segment_id: OdomSegmentId,
        map_snapshot: MapSnapshot,
        global_path: &GlobalPath,
    ) -> Result<Self, NavigationEpochError> {
        let global_plan_identity = global_path.identity();
        if global_plan_identity.map_instance_id() != map_snapshot.instance_id() {
            return Err(NavigationEpochError::GlobalPathMapMismatch {
                map_snapshot,
                global_plan_identity: Box::new(global_plan_identity),
            });
        }
        Ok(Self {
            schema_version: NAVIGATION_EPOCH_V1,
            device_session_id,
            odom_segment_id,
            map_snapshot,
            global_plan_identity,
        })
    }

    pub fn device_session_id(self) -> DeviceSessionId {
        self.device_session_id
    }
    pub fn odom_segment_id(self) -> OdomSegmentId {
        self.odom_segment_id
    }
    pub fn map_snapshot(self) -> MapSnapshot {
        self.map_snapshot
    }
    pub fn global_plan_identity(self) -> GlobalPlanIdentity {
        self.global_plan_identity
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum NavigationEpochError {
    GlobalPathMapMismatch {
        map_snapshot: MapSnapshot,
        global_plan_identity: Box<GlobalPlanIdentity>,
    },
}

impl fmt::Display for NavigationEpochError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "cannot establish V1 live navigation epoch: {self:?}"
        )
    }
}
impl std::error::Error for NavigationEpochError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OdomPoseV1 {
    position: PlanarPoint<OdomFrame>,
    yaw_rad: f64,
}

impl OdomPoseV1 {
    pub fn try_new(x_m: f64, y_m: f64, yaw_rad: f64) -> Result<Self, MotionValueError> {
        require_motion_finite("pose.x_m", x_m)?;
        require_motion_finite("pose.y_m", y_m)?;
        require_motion_finite("pose.yaw_rad", yaw_rad)?;
        if x_m.abs() > MAX_SUPPORTED_ABS_ODOM_COORDINATE_M {
            return Err(MotionValueError::CoordinateOutsideSupportedDomain {
                axis: OdomAxisV1::X,
                value_m: x_m,
                maximum_abs_m: MAX_SUPPORTED_ABS_ODOM_COORDINATE_M,
            });
        }
        if y_m.abs() > MAX_SUPPORTED_ABS_ODOM_COORDINATE_M {
            return Err(MotionValueError::CoordinateOutsideSupportedDomain {
                axis: OdomAxisV1::Y,
                value_m: y_m,
                maximum_abs_m: MAX_SUPPORTED_ABS_ODOM_COORDINATE_M,
            });
        }
        if yaw_rad.abs() > MAX_SUPPORTED_ABS_INPUT_YAW_RAD {
            return Err(MotionValueError::YawOutsideSupportedDomain {
                value_rad: yaw_rad,
                maximum_abs_rad: MAX_SUPPORTED_ABS_INPUT_YAW_RAD,
            });
        }
        let position = PlanarPoint::try_new(x_m, y_m).map_err(|_| MotionValueError::NonFinite {
            field: "pose.position_m",
            value: if x_m.is_finite() { y_m } else { x_m },
        })?;
        Ok(Self {
            position,
            yaw_rad: normalize_angle(yaw_rad),
        })
    }

    pub fn position(self) -> PlanarPoint<OdomFrame> {
        self.position
    }
    pub fn yaw_rad(self) -> f64 {
        self.yaw_rad
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OdomMotionStateV1Dto {
    pub schema_version: u32,
    pub observed_at_host_ns: u64,
    pub x_m: f64,
    pub y_m: f64,
    pub yaw_rad: f64,
    pub odom_velocity_x_mps: f64,
    pub odom_velocity_y_mps: f64,
    pub yaw_rate_rad_s: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OdomMotionStateV1 {
    schema_version: u32,
    epoch: NavigationEpochV1,
    observed_at: HostMonotonicTimestamp,
    pose: OdomPoseV1,
    odom_velocity_x_mps: f64,
    odom_velocity_y_mps: f64,
    forward_velocity_mps: f64,
    lateral_velocity_mps: f64,
    yaw_rate_rad_s: f64,
}

impl OdomMotionStateV1 {
    pub fn parse(
        dto: OdomMotionStateV1Dto,
        epoch: NavigationEpochV1,
    ) -> Result<Self, MotionValueError> {
        if dto.schema_version != ODOM_MOTION_STATE_V1 {
            return Err(MotionValueError::UnsupportedStateSchemaVersion(
                dto.schema_version,
            ));
        }
        let pose = OdomPoseV1::try_new(dto.x_m, dto.y_m, dto.yaw_rad)?;
        require_motion_finite("odom_velocity_x_mps", dto.odom_velocity_x_mps)?;
        require_motion_finite("odom_velocity_y_mps", dto.odom_velocity_y_mps)?;
        require_motion_finite("yaw_rate_rad_s", dto.yaw_rate_rad_s)?;
        let (sin_yaw, cos_yaw) = pose.yaw_rad.sin_cos();
        let forward_velocity_mps =
            cos_yaw.mul_add(dto.odom_velocity_x_mps, sin_yaw * dto.odom_velocity_y_mps);
        let lateral_velocity_mps =
            (-sin_yaw).mul_add(dto.odom_velocity_x_mps, cos_yaw * dto.odom_velocity_y_mps);
        require_motion_finite("projected_forward_velocity_mps", forward_velocity_mps)?;
        require_motion_finite("projected_lateral_velocity_mps", lateral_velocity_mps)?;
        Ok(Self {
            schema_version: ODOM_MOTION_STATE_V1,
            epoch,
            observed_at: HostMonotonicTimestamp::from_nanos(dto.observed_at_host_ns),
            pose,
            odom_velocity_x_mps: dto.odom_velocity_x_mps,
            odom_velocity_y_mps: dto.odom_velocity_y_mps,
            forward_velocity_mps,
            lateral_velocity_mps,
            yaw_rate_rad_s: dto.yaw_rate_rad_s,
        })
    }

    pub fn epoch(self) -> NavigationEpochV1 {
        self.epoch
    }
    pub fn observed_at(self) -> HostMonotonicTimestamp {
        self.observed_at
    }
    pub fn pose(self) -> OdomPoseV1 {
        self.pose
    }
    pub fn forward_velocity_mps(self) -> f64 {
        self.forward_velocity_mps
    }
    pub fn lateral_velocity_mps(self) -> f64 {
        self.lateral_velocity_mps
    }
    pub fn yaw_rate_rad_s(self) -> f64 {
        self.yaw_rate_rad_s
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MotionValueError {
    UnsupportedStateSchemaVersion(u32),
    NonFinite {
        field: &'static str,
        value: f64,
    },
    CoordinateOutsideSupportedDomain {
        axis: OdomAxisV1,
        value_m: f64,
        maximum_abs_m: f64,
    },
    YawOutsideSupportedDomain {
        value_rad: f64,
        maximum_abs_rad: f64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OdomAxisV1 {
    X,
    Y,
}

impl fmt::Display for MotionValueError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid odom motion value: {self:?}")
    }
}
impl std::error::Error for MotionValueError {}

fn require_motion_finite(field: &'static str, value: f64) -> Result<(), MotionValueError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(MotionValueError::NonFinite { field, value })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OdomReferencePointV1Dto {
    pub x_m: f64,
    pub y_m: f64,
    pub yaw_rad: f64,
    pub forward_velocity_mps: f64,
    pub yaw_rate_rad_s: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MpcReferenceV1Dto {
    pub schema_version: u32,
    pub builder_revision: u32,
    pub created_at_host_ns: u64,
    pub step_period_s: f64,
    pub points: Vec<OdomReferencePointV1Dto>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ReferenceBuilderRevisionV1 {
    TimeParameterizedGlobalPathV1 = 1,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OdomReferencePointV1 {
    pose: OdomPoseV1,
    forward_velocity_mps: f64,
    yaw_rate_rad_s: f64,
}

impl OdomReferencePointV1 {
    pub fn pose(self) -> OdomPoseV1 {
        self.pose
    }
    pub fn forward_velocity_mps(self) -> f64 {
        self.forward_velocity_mps
    }
    pub fn yaw_rate_rad_s(self) -> f64 {
        self.yaw_rate_rad_s
    }
}

#[derive(Debug, PartialEq)]
pub struct MpcReferenceV1<'path> {
    schema_version: u32,
    builder_revision: ReferenceBuilderRevisionV1,
    epoch: NavigationEpochV1,
    global_plan_identity: GlobalPlanIdentity,
    source_path: &'path GlobalPath,
    created_at: HostMonotonicTimestamp,
    step_period_s: f64,
    points: Vec<OdomReferencePointV1>,
}

impl<'path> MpcReferenceV1<'path> {
    pub fn parse(
        dto: MpcReferenceV1Dto,
        config: MpcConfigV1,
        epoch: NavigationEpochV1,
        global_path: &'path GlobalPath,
    ) -> Result<Self, MpcReferenceParseError> {
        if dto.schema_version != MPC_REFERENCE_V1 {
            return Err(MpcReferenceParseError::UnsupportedSchemaVersion(
                dto.schema_version,
            ));
        }
        let builder_revision = match dto.builder_revision {
            1 => ReferenceBuilderRevisionV1::TimeParameterizedGlobalPathV1,
            value => return Err(MpcReferenceParseError::UnsupportedBuilderRevision(value)),
        };
        let global_plan_identity = global_path.identity();
        if global_plan_identity != epoch.global_plan_identity {
            return Err(MpcReferenceParseError::GlobalPathMismatch {
                expected: Box::new(epoch.global_plan_identity),
                actual: Box::new(global_plan_identity),
            });
        }
        if dto.step_period_s.to_bits() != config.dt_s.to_bits() {
            return Err(MpcReferenceParseError::StepPeriodMismatch {
                reference_s: dto.step_period_s,
                config_s: config.dt_s,
            });
        }
        if dto.points.len() != config.horizon {
            return Err(MpcReferenceParseError::PointCount {
                expected: config.horizon,
                actual: dto.points.len(),
            });
        }
        let mut points = Vec::new();
        points.try_reserve_exact(dto.points.len()).map_err(|_| {
            MpcReferenceParseError::Allocation {
                elements: dto.points.len(),
            }
        })?;
        for (index, point) in dto.points.into_iter().enumerate() {
            let pose = OdomPoseV1::try_new(point.x_m, point.y_m, point.yaw_rad)
                .map_err(|source| MpcReferenceParseError::InvalidPoint { index, source })?;
            require_motion_finite("reference.forward_velocity_mps", point.forward_velocity_mps)
                .map_err(|source| MpcReferenceParseError::InvalidPoint { index, source })?;
            require_motion_finite("reference.yaw_rate_rad_s", point.yaw_rate_rad_s)
                .map_err(|source| MpcReferenceParseError::InvalidPoint { index, source })?;
            points.push(OdomReferencePointV1 {
                pose,
                forward_velocity_mps: point.forward_velocity_mps,
                yaw_rate_rad_s: point.yaw_rate_rad_s,
            });
        }
        Ok(Self {
            schema_version: MPC_REFERENCE_V1,
            builder_revision,
            epoch,
            global_plan_identity,
            source_path: global_path,
            created_at: HostMonotonicTimestamp::from_nanos(dto.created_at_host_ns),
            step_period_s: dto.step_period_s,
            points,
        })
    }

    pub fn builder_revision(&self) -> ReferenceBuilderRevisionV1 {
        self.builder_revision
    }
    pub fn epoch(&self) -> NavigationEpochV1 {
        self.epoch
    }
    pub fn global_plan_identity(&self) -> GlobalPlanIdentity {
        self.global_plan_identity
    }
    pub fn source_path(&self) -> &'path GlobalPath {
        self.source_path
    }
    pub fn created_at(&self) -> HostMonotonicTimestamp {
        self.created_at
    }
    pub fn points(&self) -> &[OdomReferencePointV1] {
        &self.points
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum MpcReferenceParseError {
    UnsupportedSchemaVersion(u32),
    UnsupportedBuilderRevision(u32),
    GlobalPathMismatch {
        expected: Box<GlobalPlanIdentity>,
        actual: Box<GlobalPlanIdentity>,
    },
    StepPeriodMismatch {
        reference_s: f64,
        config_s: f64,
    },
    PointCount {
        expected: usize,
        actual: usize,
    },
    InvalidPoint {
        index: usize,
        source: MotionValueError,
    },
    Allocation {
        elements: usize,
    },
}

impl fmt::Display for MpcReferenceParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid V1 MPC reference: {self:?}")
    }
}
impl std::error::Error for MpcReferenceParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidPoint { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CollisionSnapshotProvenanceV1 {
    schema_version: u32,
    epoch: NavigationEpochV1,
    local_costmap: LocalCostmapProvenance,
    max_observation_age_ns: u64,
    valid_through: HostMonotonicTimestamp,
}

impl CollisionSnapshotProvenanceV1 {
    pub fn from_runtime(
        epoch: NavigationEpochV1,
        local_costmap: &LocalCostmapView<'_>,
    ) -> Result<Self, CollisionProvenanceError> {
        let provenance = local_costmap
            .provenance()
            .ok_or(CollisionProvenanceError::NoObservation)?;
        if !local_costmap.freshness().is_current() {
            return Err(CollisionProvenanceError::ViewNotCurrent(
                local_costmap.freshness(),
            ));
        }
        if provenance.session_id() != epoch.device_session_id {
            return Err(CollisionProvenanceError::DeviceSessionMismatch {
                navigation: epoch.device_session_id,
                local_costmap: provenance.session_id(),
            });
        }
        if provenance.odom_segment_id() != epoch.odom_segment_id {
            return Err(CollisionProvenanceError::OdomSegmentMismatch {
                navigation: epoch.odom_segment_id,
                local_costmap: provenance.odom_segment_id(),
            });
        }
        let max_observation_age_ns = local_costmap.max_observation_age_ns();
        let valid_through_ns = provenance
            .host_arrival()
            .as_nanos()
            .checked_add(max_observation_age_ns)
            .ok_or(CollisionProvenanceError::ValidityDeadlineOverflow {
                observed_at: provenance.host_arrival(),
                maximum_age_ns: max_observation_age_ns,
            })?;
        Ok(Self {
            schema_version: COLLISION_SNAPSHOT_V1,
            epoch,
            local_costmap: provenance,
            max_observation_age_ns,
            valid_through: HostMonotonicTimestamp::from_nanos(valid_through_ns),
        })
    }

    pub fn epoch(self) -> NavigationEpochV1 {
        self.epoch
    }
    pub fn local_costmap(self) -> LocalCostmapProvenance {
        self.local_costmap
    }
    pub fn observed_at(self) -> HostMonotonicTimestamp {
        self.local_costmap.host_arrival()
    }
    pub fn max_observation_age_ns(self) -> u64 {
        self.max_observation_age_ns
    }
    pub fn valid_through(self) -> HostMonotonicTimestamp {
        self.valid_through
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollisionProvenanceError {
    NoObservation,
    ViewNotCurrent(LocalCostmapFreshness),
    DeviceSessionMismatch {
        navigation: DeviceSessionId,
        local_costmap: DeviceSessionId,
    },
    OdomSegmentMismatch {
        navigation: OdomSegmentId,
        local_costmap: OdomSegmentId,
    },
    ValidityDeadlineOverflow {
        observed_at: HostMonotonicTimestamp,
        maximum_age_ns: u64,
    },
}

impl fmt::Display for CollisionProvenanceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "cannot establish V1 collision provenance: {self:?}"
        )
    }
}
impl std::error::Error for CollisionProvenanceError {}

#[derive(Clone, Debug, PartialEq)]
pub struct MpcRequestV1Dto {
    pub schema_version: u32,
    pub request_id: u64,
    pub submitted_at_host_ns: u64,
    pub deadline_host_ns: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MpcRequestV1<'reference> {
    schema_version: u32,
    request_id: NonZeroU64,
    submitted_at: HostMonotonicTimestamp,
    deadline: HostMonotonicTimestamp,
    state: OdomMotionStateV1,
    reference: &'reference MpcReferenceV1<'reference>,
    previous_pwm: ShadowPwmPair,
    collision_snapshot: CollisionSnapshotProvenanceV1,
}

impl<'reference> MpcRequestV1<'reference> {
    pub fn parse(
        dto: MpcRequestV1Dto,
        state: OdomMotionStateV1,
        reference: &'reference MpcReferenceV1<'reference>,
        previous_pwm: ShadowPwmPair,
        collision_snapshot: CollisionSnapshotProvenanceV1,
    ) -> Result<Self, MpcRequestParseError> {
        if dto.schema_version != MPC_REQUEST_V1 {
            return Err(MpcRequestParseError::UnsupportedSchemaVersion(
                dto.schema_version,
            ));
        }
        let request_id =
            NonZeroU64::new(dto.request_id).ok_or(MpcRequestParseError::ZeroRequestId)?;
        if reference.epoch != state.epoch {
            return Err(MpcRequestParseError::ReferenceEpochMismatch {
                state: Box::new(state.epoch),
                reference: Box::new(reference.epoch),
            });
        }
        if collision_snapshot.epoch != state.epoch {
            return Err(MpcRequestParseError::CollisionEpochMismatch {
                state: Box::new(state.epoch),
                collision: Box::new(collision_snapshot.epoch),
            });
        }
        let submitted_at = HostMonotonicTimestamp::from_nanos(dto.submitted_at_host_ns);
        let deadline = HostMonotonicTimestamp::from_nanos(dto.deadline_host_ns);
        if submitted_at < state.observed_at {
            return Err(MpcRequestParseError::StateAfterSubmission {
                state: state.observed_at,
                submitted_at,
            });
        }
        if submitted_at < reference.created_at {
            return Err(MpcRequestParseError::ReferenceAfterSubmission {
                reference: reference.created_at,
                submitted_at,
            });
        }
        if submitted_at < collision_snapshot.observed_at() {
            return Err(MpcRequestParseError::CollisionSnapshotAfterSubmission {
                snapshot: collision_snapshot.observed_at(),
                submitted_at,
            });
        }
        if deadline <= submitted_at {
            return Err(MpcRequestParseError::NonFutureDeadline {
                submitted_at,
                deadline,
            });
        }
        if deadline > collision_snapshot.valid_through {
            return Err(MpcRequestParseError::DeadlineExceedsCollisionValidity {
                deadline,
                collision_valid_through: collision_snapshot.valid_through,
            });
        }
        Ok(Self {
            schema_version: MPC_REQUEST_V1,
            request_id,
            submitted_at,
            deadline,
            state,
            reference,
            previous_pwm,
            collision_snapshot,
        })
    }

    pub fn request_id(self) -> NonZeroU64 {
        self.request_id
    }
    pub fn submitted_at(self) -> HostMonotonicTimestamp {
        self.submitted_at
    }
    pub fn deadline(self) -> HostMonotonicTimestamp {
        self.deadline
    }
    pub fn state(self) -> OdomMotionStateV1 {
        self.state
    }
    pub fn reference(self) -> &'reference MpcReferenceV1<'reference> {
        self.reference
    }
    pub fn previous_pwm(self) -> ShadowPwmPair {
        self.previous_pwm
    }
    pub fn collision_snapshot(self) -> CollisionSnapshotProvenanceV1 {
        self.collision_snapshot
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum MpcRequestParseError {
    UnsupportedSchemaVersion(u32),
    ZeroRequestId,
    ReferenceEpochMismatch {
        state: Box<NavigationEpochV1>,
        reference: Box<NavigationEpochV1>,
    },
    CollisionEpochMismatch {
        state: Box<NavigationEpochV1>,
        collision: Box<NavigationEpochV1>,
    },
    StateAfterSubmission {
        state: HostMonotonicTimestamp,
        submitted_at: HostMonotonicTimestamp,
    },
    ReferenceAfterSubmission {
        reference: HostMonotonicTimestamp,
        submitted_at: HostMonotonicTimestamp,
    },
    CollisionSnapshotAfterSubmission {
        snapshot: HostMonotonicTimestamp,
        submitted_at: HostMonotonicTimestamp,
    },
    NonFutureDeadline {
        submitted_at: HostMonotonicTimestamp,
        deadline: HostMonotonicTimestamp,
    },
    DeadlineExceedsCollisionValidity {
        deadline: HostMonotonicTimestamp,
        collision_valid_through: HostMonotonicTimestamp,
    },
}

impl fmt::Display for MpcRequestParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid V1 MPC request: {self:?}")
    }
}
impl std::error::Error for MpcRequestParseError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapsulePurposeV1 {
    StartOccupancy,
    PredictedMotion,
}

/// A closed odom-frame capsule around one bounded midpoint-model substep.
/// Collision snapshots are expected to be robot-footprint-inflated;
/// `extra_radius_m` combines the analytic integration bound and explicit
/// binary64 slack. The standard library's transcendental functions are not
/// formally interval-certified by this implementation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConservativeCapsuleSegmentV1 {
    purpose: CapsulePurposeV1,
    horizon_step: usize,
    integration_substep: usize,
    time_start_s: f64,
    time_end_s: f64,
    centerline_start: PlanarPoint<OdomFrame>,
    centerline_end: PlanarPoint<OdomFrame>,
    extra_radius_m: f64,
}

impl ConservativeCapsuleSegmentV1 {
    fn stationary_start(pose: OdomPoseV1) -> Self {
        Self {
            purpose: CapsulePurposeV1::StartOccupancy,
            horizon_step: 0,
            integration_substep: 0,
            time_start_s: 0.0,
            time_end_s: 0.0,
            centerline_start: pose.position,
            centerline_end: pose.position,
            extra_radius_m: 0.0,
        }
    }

    pub fn purpose(self) -> CapsulePurposeV1 {
        self.purpose
    }
    pub fn horizon_step(self) -> usize {
        self.horizon_step
    }
    pub fn integration_substep(self) -> usize {
        self.integration_substep
    }
    pub fn time_start_s(self) -> f64 {
        self.time_start_s
    }
    pub fn time_end_s(self) -> f64 {
        self.time_end_s
    }
    pub fn centerline_start(self) -> PlanarPoint<OdomFrame> {
        self.centerline_start
    }
    pub fn centerline_end(self) -> PlanarPoint<OdomFrame> {
        self.centerline_end
    }
    pub fn extra_radius_m(self) -> f64 {
        self.extra_radius_m
    }
}

pub trait CollisionQuery {
    type Error;

    /// Provenance of the immutable, footprint-inflated snapshot queried by
    /// every subsequent call during this solve.
    fn snapshot_provenance(&self) -> CollisionSnapshotProvenanceV1;

    /// Whether the entire closed capsule is traversable in that snapshot.
    fn is_capsule_traversable(
        &mut self,
        segment: ConservativeCapsuleSegmentV1,
    ) -> Result<bool, Self::Error>;
}

/// Allocation-free adapter for the canonical local-costmap view.
///
/// It supercovers each capsule by its axis-aligned rectangle in the frozen
/// local-costmap frame. This can reject extra corner cells, but cannot tunnel
/// through a non-free cell between sample points. Closed overlap means exact
/// tangency to a non-free cell is blocked.
pub struct LocalCostmapCapsuleQueryV1<'view> {
    view: LocalCostmapView<'view>,
    snapshot: CollisionSnapshotProvenanceV1,
    odom_to_local_costmap: OdomToLocalCostmap,
}

impl<'view> LocalCostmapCapsuleQueryV1<'view> {
    pub fn try_new(
        view: LocalCostmapView<'view>,
        snapshot: CollisionSnapshotProvenanceV1,
    ) -> Result<Self, LocalCostmapCapsuleAdapterError> {
        if !view.freshness().is_current() {
            return Err(LocalCostmapCapsuleAdapterError::ViewNotCurrent(
                view.freshness(),
            ));
        }
        let actual = view
            .provenance()
            .ok_or(LocalCostmapCapsuleAdapterError::NoObservation)?;
        if actual != snapshot.local_costmap {
            return Err(LocalCostmapCapsuleAdapterError::ProvenanceMismatch {
                expected: Box::new(snapshot.local_costmap),
                actual: Box::new(actual),
            });
        }
        if view.max_observation_age_ns() != snapshot.max_observation_age_ns {
            return Err(
                LocalCostmapCapsuleAdapterError::ObservationAgePolicyMismatch {
                    expected_ns: snapshot.max_observation_age_ns,
                    actual_ns: view.max_observation_age_ns(),
                },
            );
        }
        let odom_to_local_costmap = actual
            .local_costmap_to_odom()
            .inverse()
            .map_err(LocalCostmapCapsuleAdapterError::Transform)?;
        Ok(Self {
            view,
            snapshot,
            odom_to_local_costmap,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum LocalCostmapCapsuleAdapterError {
    NoObservation,
    ViewNotCurrent(LocalCostmapFreshness),
    ProvenanceMismatch {
        expected: Box<LocalCostmapProvenance>,
        actual: Box<LocalCostmapProvenance>,
    },
    ObservationAgePolicyMismatch {
        expected_ns: u64,
        actual_ns: u64,
    },
    Transform(PlanarTransformError),
}

impl fmt::Display for LocalCostmapCapsuleAdapterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "cannot construct local-costmap capsule adapter: {self:?}"
        )
    }
}
impl std::error::Error for LocalCostmapCapsuleAdapterError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Transform(source) => Some(source),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LocalCostmapCapsuleQueryError {
    Transform(PlanarTransformError),
    NumericalBounds,
}

impl fmt::Display for LocalCostmapCapsuleQueryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "local-costmap capsule query failed: {self:?}")
    }
}
impl std::error::Error for LocalCostmapCapsuleQueryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Transform(source) => Some(source),
            Self::NumericalBounds => None,
        }
    }
}

impl CollisionQuery for LocalCostmapCapsuleQueryV1<'_> {
    type Error = LocalCostmapCapsuleQueryError;

    fn snapshot_provenance(&self) -> CollisionSnapshotProvenanceV1 {
        self.snapshot
    }

    fn is_capsule_traversable(
        &mut self,
        segment: ConservativeCapsuleSegmentV1,
    ) -> Result<bool, Self::Error> {
        let start = self
            .odom_to_local_costmap
            .transform_point(segment.centerline_start)
            .map_err(LocalCostmapCapsuleQueryError::Transform)?;
        let end = self
            .odom_to_local_costmap
            .transform_point(segment.centerline_end)
            .map_err(LocalCostmapCapsuleQueryError::Transform)?;
        let radius = segment.extra_radius_m;
        if !radius.is_finite() || radius < 0.0 {
            return Err(LocalCostmapCapsuleQueryError::NumericalBounds);
        }
        let minimum_x = start.x_m().min(end.x_m()) - radius;
        let maximum_x = start.x_m().max(end.x_m()) + radius;
        let minimum_y = start.y_m().min(end.y_m()) - radius;
        let maximum_y = start.y_m().max(end.y_m()) + radius;
        if !minimum_x.is_finite()
            || !maximum_x.is_finite()
            || !minimum_y.is_finite()
            || !maximum_y.is_finite()
        {
            return Err(LocalCostmapCapsuleQueryError::NumericalBounds);
        }
        let lower = self.view.lower_bound_m();
        let resolution = self.view.resolution_m();
        let upper = [
            lower[0] + f64::from(self.view.width()) * resolution,
            lower[1] + f64::from(self.view.height()) * resolution,
        ];
        let inside = if radius == 0.0 {
            minimum_x >= lower[0]
                && minimum_y >= lower[1]
                && maximum_x < upper[0]
                && maximum_y < upper[1]
        } else {
            minimum_x > lower[0]
                && minimum_y > lower[1]
                && maximum_x < upper[0]
                && maximum_y < upper[1]
        };
        if !inside {
            return Ok(false);
        }
        let minimum_column = (((next_down(minimum_x) - lower[0]) / resolution).floor() as i64)
            .clamp(0, i64::from(self.view.width()) - 1) as u32;
        let maximum_column = (((next_up(maximum_x) - lower[0]) / resolution).floor() as i64)
            .clamp(0, i64::from(self.view.width()) - 1) as u32;
        let minimum_row = (((next_down(minimum_y) - lower[1]) / resolution).floor() as i64)
            .clamp(0, i64::from(self.view.height()) - 1) as u32;
        let maximum_row = (((next_up(maximum_y) - lower[1]) / resolution).floor() as i64)
            .clamp(0, i64::from(self.view.height()) - 1) as u32;
        for row in minimum_row..=maximum_row {
            for column in minimum_column..=maximum_column {
                if self.view.cell(column, row) != Some(LocalCostmapCell::Free) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
}

/// Failure to read a timestamp in the host navigation clock's `u64`
/// nanosecond domain.
///
/// The production clock is process-relative and based on [`std::time::Instant`].
/// `Duration::as_nanos` is `u128`, so the boundary conversion remains fallible
/// instead of fabricating a maximum timestamp when its target domain is
/// exhausted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostMonotonicClockReadError {
    ElapsedNanosecondsOutOfRange { elapsed_nanoseconds: u128 },
}

impl fmt::Display for HostMonotonicClockReadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ElapsedNanosecondsOutOfRange {
                elapsed_nanoseconds,
            } => write!(
                formatter,
                "host monotonic elapsed time {elapsed_nanoseconds} ns exceeds the u64 navigation timebase"
            ),
        }
    }
}

impl std::error::Error for HostMonotonicClockReadError {}

pub trait HostMonotonicClock {
    fn try_now(&mut self) -> Result<HostMonotonicTimestamp, HostMonotonicClockReadError>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SolveStatusV1 {
    started_at: HostMonotonicTimestamp,
    observed_at: HostMonotonicTimestamp,
    deadline: HostMonotonicTimestamp,
    completed_iterations: u8,
    active_iteration: Option<u8>,
    rollout_evaluations: u64,
    pre_final_collision_queries: u64,
    final_validation_queries: u64,
}

impl SolveStatusV1 {
    pub fn started_at(self) -> HostMonotonicTimestamp {
        self.started_at
    }
    pub fn observed_at(self) -> HostMonotonicTimestamp {
        self.observed_at
    }
    pub fn deadline(self) -> HostMonotonicTimestamp {
        self.deadline
    }
    pub fn completed_iterations(self) -> u8 {
        self.completed_iterations
    }
    /// Admitted rollout attempts charged against the configured solve budget.
    ///
    /// The count advances before the rollout body starts, so an in-progress
    /// failure can retain an attempt which did not produce a rollout outcome.
    pub fn rollout_evaluations(self) -> u64 {
        self.rollout_evaluations
    }
    /// Pre-final collision-query attempts whose query method was invoked.
    ///
    /// This counts invocations, not traversable results. A query error or the
    /// immediately following clock-read failure retains the invoked attempt.
    pub fn pre_final_collision_queries(self) -> u64 {
        self.pre_final_collision_queries
    }
    /// Final-revalidation collision-query attempts whose method was invoked.
    ///
    /// A successful solution proves these returned traversable; a failed
    /// solve can retain an invoked attempt without claiming that result.
    pub fn final_validation_queries(self) -> u64 {
        self.final_validation_queries
    }
}

/// Truthful progress retained by a failed solve.
///
/// `NotStarted` means the solver never obtained its first clock observation,
/// so no timestamp or work counter exists. `InProgress` begins after that
/// first successful observation and retains only the last successfully read
/// timestamp plus counters with the support documented by [`SolveStatusV1`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MpcSolveProgressV1 {
    NotStarted,
    InProgress(SolveStatusV1),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PredictedOdomPointV1 {
    time_from_start_s: f64,
    pose: OdomPoseV1,
    left_velocity_mps: f64,
    right_velocity_mps: f64,
    forward_velocity_mps: f64,
    yaw_rate_rad_s: f64,
    requested_pwm: ShadowPwmPair,
    integration_tube_radius_m: f64,
}

impl PredictedOdomPointV1 {
    pub fn time_from_start_s(self) -> f64 {
        self.time_from_start_s
    }
    pub fn pose(self) -> OdomPoseV1 {
        self.pose
    }
    pub fn left_velocity_mps(self) -> f64 {
        self.left_velocity_mps
    }
    pub fn right_velocity_mps(self) -> f64 {
        self.right_velocity_mps
    }
    pub fn forward_velocity_mps(self) -> f64 {
        self.forward_velocity_mps
    }
    pub fn yaw_rate_rad_s(self) -> f64 {
        self.yaw_rate_rad_s
    }
    pub fn requested_pwm(self) -> ShadowPwmPair {
        self.requested_pwm
    }
    pub fn integration_tube_radius_m(self) -> f64 {
        self.integration_tube_radius_m
    }
}

/// Private construction prevents callers from forging a claim that the
/// selected trajectory was revalidated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FinalTrajectoryValidationV1 {
    collision_snapshot: CollisionSnapshotProvenanceV1,
    segment_count: usize,
    validated_at: HostMonotonicTimestamp,
}

impl FinalTrajectoryValidationV1 {
    pub fn collision_snapshot(self) -> CollisionSnapshotProvenanceV1 {
        self.collision_snapshot
    }
    pub fn segment_count(self) -> usize {
        self.segment_count
    }
    pub fn validated_at(self) -> HostMonotonicTimestamp {
        self.validated_at
    }
}

#[derive(Debug)]
pub struct MpcSolution<'solver, 'reference> {
    model: PlantModelV1,
    config: MpcConfigV1,
    request: MpcRequestV1<'reference>,
    requested_pwm: ShadowPwmPair,
    command_sequence: &'solver [ShadowPwmPair],
    predicted_trajectory: &'solver [PredictedOdomPointV1],
    conservative_capsules: &'solver [ConservativeCapsuleSegmentV1],
    final_validation: FinalTrajectoryValidationV1,
    status: SolveStatusV1,
    objective_cost: f64,
}

impl<'solver, 'reference> MpcSolution<'solver, 'reference> {
    pub fn model(&self) -> PlantModelV1 {
        self.model
    }
    pub fn config(&self) -> MpcConfigV1 {
        self.config
    }
    pub fn request(&self) -> MpcRequestV1<'reference> {
        self.request
    }
    pub fn requested_pwm(&self) -> ShadowPwmPair {
        self.requested_pwm
    }
    pub fn command_sequence(&self) -> &[ShadowPwmPair] {
        self.command_sequence
    }
    pub fn predicted_trajectory(&self) -> &[PredictedOdomPointV1] {
        self.predicted_trajectory
    }
    pub fn conservative_capsules(&self) -> &[ConservativeCapsuleSegmentV1] {
        self.conservative_capsules
    }
    pub fn final_validation(&self) -> FinalTrajectoryValidationV1 {
        self.final_validation
    }
    pub fn status(&self) -> SolveStatusV1 {
        self.status
    }
    pub fn objective_cost(&self) -> f64 {
        self.objective_cost
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClockFault {
    BeforeRequestSubmission {
        submitted_at: HostMonotonicTimestamp,
        observed_at: HostMonotonicTimestamp,
    },
    Regression {
        previous: HostMonotonicTimestamp,
        observed_at: HostMonotonicTimestamp,
    },
    DeadlineReached {
        deadline: HostMonotonicTimestamp,
        observed_at: HostMonotonicTimestamp,
    },
}

impl fmt::Display for ClockFault {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "host monotonic clock fault: {self:?}")
    }
}
impl std::error::Error for ClockFault {}

/// A clock source read failure is distinct from a successfully read value
/// which violates the parsed request's monotonic/deadline contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostMonotonicClockFailure {
    Read(HostMonotonicClockReadError),
    Fault(ClockFault),
}

impl fmt::Display for HostMonotonicClockFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read(source) => source.fmt(formatter),
            Self::Fault(source) => source.fmt(formatter),
        }
    }
}

impl std::error::Error for HostMonotonicClockFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read(source) => Some(source),
            Self::Fault(source) => Some(source),
        }
    }
}

#[derive(Debug)]
pub enum CollisionObservationFailure<E> {
    Query(E),
    Clock(HostMonotonicClockFailure),
    QueryAndClock {
        query: E,
        clock: HostMonotonicClockFailure,
    },
}

impl<E: fmt::Debug> fmt::Display for CollisionObservationFailure<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "collision observation failure: {self:?}")
    }
}

impl<E: std::error::Error + 'static> std::error::Error for CollisionObservationFailure<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Query(source) | Self::QueryAndClock { query: source, .. } => Some(source),
            Self::Clock(source) => Some(source),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnvelopeLocationV1 {
    InitialState,
    Rollout {
        horizon_step: usize,
        integration_substep: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlantEnvelopeViolationV1 {
    LeftWheelVelocity {
        location: EnvelopeLocationV1,
        value_mps: f64,
        minimum_mps: f64,
        maximum_mps: f64,
    },
    RightWheelVelocity {
        location: EnvelopeLocationV1,
        value_mps: f64,
        minimum_mps: f64,
        maximum_mps: f64,
    },
    YawRate {
        location: EnvelopeLocationV1,
        maximum_observed_abs_rad_s: f64,
        allowed_abs_rad_s: f64,
    },
    LateralVelocity {
        value_mps: f64,
        allowed_abs_mps: f64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericalStageV1 {
    InitialWheelState,
    FirstOrderResponse,
    YawEnvelope,
    MidpointIntegration,
    IntegrationTube,
    CostEvaluation,
    PredictedTrajectory,
}

#[derive(Debug)]
pub enum MpcFailureKind<E> {
    Clock(HostMonotonicClockFailure),
    CollisionSnapshotMismatch {
        requested: Box<CollisionSnapshotProvenanceV1>,
        actual: Box<CollisionSnapshotProvenanceV1>,
    },
    OccupiedStart,
    PreviousPwmOutsideEnvelope {
        wheel: WheelSide,
        value: PwmPercent,
    },
    ReferenceDoesNotMatchConfig {
        expected_steps: usize,
        actual_steps: usize,
        expected_period_s: f64,
        actual_period_s: f64,
    },
    PlantEnvelope(PlantEnvelopeViolationV1),
    CollisionBlocked {
        horizon_step: usize,
        integration_substep: usize,
    },
    FinalTrajectoryBlocked {
        horizon_step: usize,
        integration_substep: usize,
    },
    IntegrationTubeExceeded {
        horizon_step: usize,
        integration_substep: usize,
        required_m: f64,
        allowed_m: f64,
    },
    CollisionObservation {
        horizon_step: Option<usize>,
        integration_substep: Option<usize>,
        final_revalidation: bool,
        source: CollisionObservationFailure<E>,
    },
    Numerical {
        stage: NumericalStageV1,
        horizon_step: Option<usize>,
        integration_substep: Option<usize>,
    },
    EvaluationLimit {
        /// Configured maximum admitted, budget-charged rollout attempts.
        configured: u64,
    },
}

#[derive(Debug)]
pub struct MpcFailure<'reference, E> {
    model: PlantModelV1,
    config: MpcConfigV1,
    request: MpcRequestV1<'reference>,
    progress: MpcSolveProgressV1,
    kind: MpcFailureKind<E>,
}

/// Failure ownership is boxed so successful control ticks do not carry the
/// full model, request, and provenance payload in their `Result` layout.
pub type MpcSolveError<'reference, E> = Box<MpcFailure<'reference, E>>;

impl<'reference, E> MpcFailure<'reference, E> {
    pub fn model(&self) -> PlantModelV1 {
        self.model
    }
    pub fn config(&self) -> MpcConfigV1 {
        self.config
    }
    pub fn request(&self) -> MpcRequestV1<'reference> {
        self.request
    }
    pub fn progress(&self) -> MpcSolveProgressV1 {
        self.progress
    }
    pub fn kind(&self) -> &MpcFailureKind<E> {
        &self.kind
    }
}

impl<E: fmt::Debug> fmt::Display for MpcFailure<'_, E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "V1 shadow MPC solve failed: {:?}", self.kind)
    }
}

impl<E: std::error::Error + 'static> std::error::Error for MpcFailure<'_, E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            MpcFailureKind::Clock(source) => Some(source),
            MpcFailureKind::CollisionObservation { source, .. } => source.source(),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MpcCreateError {
    StepPeriodDoesNotMatchModel {
        config_s: f64,
        model_s: f64,
    },
    ControllerPwmOutsideModelEnvelope {
        wheel: WheelSide,
    },
    UnsupportedTimeConstant {
        wheel: WheelSide,
        substep_s: f64,
        time_constant_s: f64,
        ratio: f64,
        maximum_ratio: f64,
    },
    UnsupportedYawExcursion {
        substep_s: f64,
        maximum_yaw_rate_rad_s: f64,
        excursion_rad: f64,
        maximum_excursion_rad: f64,
    },
    Allocation {
        buffer: &'static str,
        elements: usize,
    },
}

impl fmt::Display for MpcCreateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "cannot create V1 shadow MPC: {self:?}")
    }
}
impl std::error::Error for MpcCreateError {}

#[derive(Clone, Copy, Debug)]
struct SimState {
    x_m: f64,
    y_m: f64,
    yaw_rad: f64,
    left_velocity_mps: f64,
    right_velocity_mps: f64,
    integration_tube_radius_m: f64,
}

#[derive(Clone, Copy, Debug)]
struct WheelResponse {
    initial_velocity_mps: f64,
    target_velocity_mps: f64,
    time_constant_s: f64,
}

impl WheelResponse {
    fn velocity_at(self, time_s: f64) -> Option<f64> {
        let ratio = time_s / self.time_constant_s;
        if !ratio.is_finite() || ratio < 0.0 {
            return None;
        }
        let exponential_minus_one = (-ratio).exp_m1();
        let decay = 1.0 + exponential_minus_one;
        let response = -exponential_minus_one;
        stable_weighted_sum(
            decay,
            self.initial_velocity_mps,
            response,
            self.target_velocity_mps,
        )
    }

    fn distance_at(self, time_s: f64) -> Option<f64> {
        let ratio = time_s / self.time_constant_s;
        if !ratio.is_finite() || ratio < 0.0 {
            return None;
        }
        let response_area_s = if ratio < 1.0e-3 {
            let polynomial = 0.5
                + ratio
                    * (-1.0 / 6.0
                        + ratio
                            * (1.0 / 24.0
                                + ratio * (-1.0 / 120.0 + ratio * (1.0 / 720.0 - ratio / 5040.0))));
            time_s * ratio * polynomial
        } else {
            let alpha = -(-ratio).exp_m1();
            time_s - self.time_constant_s * alpha
        };
        if !response_area_s.is_finite() || response_area_s < 0.0 || response_area_s > time_s {
            return None;
        }
        stable_weighted_sum(
            time_s - response_area_s,
            self.initial_velocity_mps,
            response_area_s,
            self.target_velocity_mps,
        )
    }
}

/// Evaluate `left_weight * left + right_weight * right` without first taking
/// a potentially overflowing `right - left`. Scaling also avoids an
/// intermediate overflow when finite opposite-sign terms nearly cancel.
fn stable_weighted_sum(left_weight: f64, left: f64, right_weight: f64, right: f64) -> Option<f64> {
    if !left_weight.is_finite()
        || !right_weight.is_finite()
        || left_weight < 0.0
        || right_weight < 0.0
        || !left.is_finite()
        || !right.is_finite()
    {
        return None;
    }
    let scale = left.abs().max(right.abs());
    if scale == 0.0 {
        return Some(0.0);
    }
    let normalized = left_weight.mul_add(left / scale, right_weight * (right / scale));
    let result = scale * normalized;
    result
        .is_finite()
        .then_some(if result == 0.0 { 0.0 } else { result })
}

#[derive(Clone, Copy, Debug)]
enum RolloutRejection {
    PlantEnvelope(PlantEnvelopeViolationV1),
    CollisionBlocked {
        horizon_step: usize,
        integration_substep: usize,
    },
    IntegrationTubeExceeded {
        horizon_step: usize,
        integration_substep: usize,
        required_m: f64,
    },
}

#[derive(Clone, Copy, Debug)]
enum RolloutEvaluation {
    Feasible(f64),
    Rejected(RolloutRejection),
}

#[derive(Clone, Copy, Debug)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}

#[derive(Clone, Copy, Debug)]
struct CostRoots {
    position: f64,
    heading: f64,
    forward_velocity: f64,
    yaw_rate: f64,
    pwm: f64,
    slew: f64,
    terminal_position: f64,
    terminal_heading: f64,
    terminal_forward_velocity: f64,
    terminal_yaw_rate: f64,
}

impl CompensatedSum {
    fn new() -> Self {
        Self {
            sum: 0.0,
            correction: 0.0,
        }
    }

    fn add(&mut self, value: f64) -> bool {
        let next = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.correction += (self.sum - next) + value;
        } else {
            self.correction += (value - next) + self.sum;
        }
        self.sum = next;
        self.sum.is_finite() && self.correction.is_finite()
    }

    fn total(self) -> f64 {
        self.sum + self.correction
    }
}

pub struct MpcSolver {
    model: PlantModelV1,
    config: MpcConfigV1,
    commands: Vec<ShadowPwmPair>,
    trial_states: Vec<SimState>,
    trial_capsules: Vec<ConservativeCapsuleSegmentV1>,
    predicted_trajectory: Vec<PredictedOdomPointV1>,
    substep_s: f64,
    cost_roots: CostRoots,
}

impl MpcSolver {
    pub fn new(model: PlantModelV1, config: MpcConfigV1) -> Result<Self, MpcCreateError> {
        if model.sample_period_s.to_bits() != config.dt_s.to_bits() {
            return Err(MpcCreateError::StepPeriodDoesNotMatchModel {
                config_s: config.dt_s,
                model_s: model.sample_period_s,
            });
        }
        for (wheel, controller, plant) in [
            (WheelSide::Left, config.left_pwm, model.left_pwm),
            (WheelSide::Right, config.right_pwm, model.right_pwm),
        ] {
            if controller.min.get() < plant.min.get() || controller.max.get() > plant.max.get() {
                return Err(MpcCreateError::ControllerPwmOutsideModelEnvelope { wheel });
            }
        }
        let substep_s = config.dt_s / config.integration_substeps as f64;
        for (wheel, tau) in [
            (WheelSide::Left, model.left.time_constant_s),
            (WheelSide::Right, model.right.time_constant_s),
        ] {
            let ratio = substep_s / tau;
            if !substep_s.is_finite()
                || substep_s <= 0.0
                || !ratio.is_finite()
                || ratio <= 0.0
                || ratio > MAX_SUBSTEP_TO_TAU_RATIO
            {
                return Err(MpcCreateError::UnsupportedTimeConstant {
                    wheel,
                    substep_s,
                    time_constant_s: tau,
                    ratio,
                    maximum_ratio: MAX_SUBSTEP_TO_TAU_RATIO,
                });
            }
        }
        let yaw_excursion = next_up(model.max_abs_yaw_rate_rad_s * substep_s);
        if !yaw_excursion.is_finite() || yaw_excursion > MAX_SUPPORTED_YAW_EXCURSION_PER_SUBSTEP_RAD
        {
            return Err(MpcCreateError::UnsupportedYawExcursion {
                substep_s,
                maximum_yaw_rate_rad_s: model.max_abs_yaw_rate_rad_s,
                excursion_rad: yaw_excursion,
                maximum_excursion_rad: MAX_SUPPORTED_YAW_EXCURSION_PER_SUBSTEP_RAD,
            });
        }
        let commands = fallible_vec("commands", config.horizon, ShadowPwmPair::STOP)?;
        let zero_state = SimState {
            x_m: 0.0,
            y_m: 0.0,
            yaw_rad: 0.0,
            left_velocity_mps: 0.0,
            right_velocity_mps: 0.0,
            integration_tube_radius_m: 0.0,
        };
        let trial_states = fallible_vec("trial_states", config.horizon, zero_state)?;
        let zero_capsule = ConservativeCapsuleSegmentV1 {
            purpose: CapsulePurposeV1::PredictedMotion,
            horizon_step: 0,
            integration_substep: 0,
            time_start_s: 0.0,
            time_end_s: 0.0,
            centerline_start: PlanarPoint::origin(),
            centerline_end: PlanarPoint::origin(),
            extra_radius_m: 0.0,
        };
        let capsule_count = config
            .horizon
            .checked_mul(config.integration_substeps)
            .ok_or(MpcCreateError::Allocation {
                buffer: "trial_capsules",
                elements: usize::MAX,
            })?;
        let trial_capsules = fallible_vec("trial_capsules", capsule_count, zero_capsule)?;
        let zero_pose = OdomPoseV1 {
            position: PlanarPoint::origin(),
            yaw_rad: 0.0,
        };
        let zero_prediction = PredictedOdomPointV1 {
            time_from_start_s: 0.0,
            pose: zero_pose,
            left_velocity_mps: 0.0,
            right_velocity_mps: 0.0,
            forward_velocity_mps: 0.0,
            yaw_rate_rad_s: 0.0,
            requested_pwm: ShadowPwmPair::STOP,
            integration_tube_radius_m: 0.0,
        };
        let predicted_trajectory =
            fallible_vec("predicted_trajectory", config.horizon, zero_prediction)?;
        let weights = config.weights;
        let cost_roots = CostRoots {
            position: weights.position.sqrt(),
            heading: weights.heading.sqrt(),
            forward_velocity: weights.forward_velocity.sqrt(),
            yaw_rate: weights.yaw_rate.sqrt(),
            pwm: weights.pwm.sqrt(),
            slew: weights.slew.sqrt(),
            terminal_position: (weights.position * weights.terminal).sqrt(),
            terminal_heading: (weights.heading * weights.terminal).sqrt(),
            terminal_forward_velocity: (weights.forward_velocity * weights.terminal).sqrt(),
            terminal_yaw_rate: (weights.yaw_rate * weights.terminal).sqrt(),
        };
        Ok(Self {
            model,
            config,
            commands,
            trial_states,
            trial_capsules,
            predicted_trajectory,
            substep_s,
            cost_roots,
        })
    }

    pub fn model(&self) -> PlantModelV1 {
        self.model
    }
    pub fn config(&self) -> MpcConfigV1 {
        self.config
    }

    fn initial_sim_state(
        &self,
        state: OdomMotionStateV1,
    ) -> Result<SimState, PlantEnvelopeViolationV1> {
        if state.lateral_velocity_mps.abs() > self.model.max_abs_lateral_velocity_mps {
            return Err(PlantEnvelopeViolationV1::LateralVelocity {
                value_mps: state.lateral_velocity_mps,
                allowed_abs_mps: self.model.max_abs_lateral_velocity_mps,
            });
        }
        let half_turn = 0.5 * self.model.wheelbase_m * state.yaw_rate_rad_s;
        let left = state.forward_velocity_mps - half_turn;
        let right = state.forward_velocity_mps + half_turn;
        let location = EnvelopeLocationV1::InitialState;
        if !left.is_finite() || !right.is_finite() {
            return Err(PlantEnvelopeViolationV1::LeftWheelVelocity {
                location,
                value_mps: left,
                minimum_mps: self.model.left_velocity.min,
                maximum_mps: self.model.left_velocity.max,
            });
        }
        if !self.model.left_velocity.contains(left) {
            return Err(PlantEnvelopeViolationV1::LeftWheelVelocity {
                location,
                value_mps: left,
                minimum_mps: self.model.left_velocity.min,
                maximum_mps: self.model.left_velocity.max,
            });
        }
        if !self.model.right_velocity.contains(right) {
            return Err(PlantEnvelopeViolationV1::RightWheelVelocity {
                location,
                value_mps: right,
                minimum_mps: self.model.right_velocity.min,
                maximum_mps: self.model.right_velocity.max,
            });
        }
        if state.yaw_rate_rad_s.abs() > self.model.max_abs_yaw_rate_rad_s {
            return Err(PlantEnvelopeViolationV1::YawRate {
                location,
                maximum_observed_abs_rad_s: state.yaw_rate_rad_s.abs(),
                allowed_abs_rad_s: self.model.max_abs_yaw_rate_rad_s,
            });
        }
        Ok(SimState {
            x_m: state.pose.position.x_m(),
            y_m: state.pose.position.y_m(),
            yaw_rad: state.pose.yaw_rad,
            left_velocity_mps: left,
            right_velocity_mps: right,
            integration_tube_radius_m: 0.0,
        })
    }

    fn validate_previous_pwm(&self, previous: ShadowPwmPair) -> Result<(), WheelSide> {
        if !self.config.left_pwm.contains(previous.left())
            || !self.model.left_pwm.contains(previous.left())
        {
            return Err(WheelSide::Left);
        }
        if !self.config.right_pwm.contains(previous.right())
            || !self.model.right_pwm.contains(previous.right())
        {
            return Err(WheelSide::Right);
        }
        Ok(())
    }
}

fn fallible_vec<T: Clone>(
    buffer: &'static str,
    elements: usize,
    value: T,
) -> Result<Vec<T>, MpcCreateError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(elements)
        .map_err(|_| MpcCreateError::Allocation { buffer, elements })?;
    result.resize(elements, value);
    Ok(result)
}

fn normalize_angle(angle: f64) -> f64 {
    let positive = angle.rem_euclid(std::f64::consts::TAU);
    let result = if positive >= std::f64::consts::PI {
        positive - std::f64::consts::TAU
    } else {
        positive
    };
    if result == 0.0 { 0.0 } else { result }
}

fn next_up(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

fn next_down(value: f64) -> f64 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

fn outward_add_nonnegative(left: f64, right: f64) -> Option<f64> {
    if left < 0.0 || right < 0.0 {
        return None;
    }
    let value = next_up(left + right);
    value.is_finite().then_some(value)
}

fn outward_mul_nonnegative(left: f64, right: f64) -> Option<f64> {
    if left < 0.0 || right < 0.0 {
        return None;
    }
    let value = next_up(left * right);
    value.is_finite().then_some(value)
}

fn outward_abs_difference(left: f64, right: f64) -> Option<f64> {
    if !left.is_finite() || !right.is_finite() {
        return None;
    }
    let lower = next_down(left - right);
    let upper = next_up(left - right);
    if !lower.is_finite() || !upper.is_finite() {
        return None;
    }
    Some(next_up(lower.abs().max(upper.abs())))
}

fn outward_div_nonnegative(numerator: f64, denominator: f64) -> Option<f64> {
    if !numerator.is_finite() || numerator < 0.0 || !denominator.is_finite() || denominator <= 0.0 {
        return None;
    }
    let value = next_up(numerator / denominator);
    value.is_finite().then_some(value)
}

fn numeric_position_roundoff_slack(
    x_m: f64,
    y_m: f64,
    max_speed_mps: f64,
    duration_s: f64,
) -> Option<f64> {
    let travel = outward_mul_nonnegative(max_speed_mps, duration_s)?;
    let scale = 1.0 + x_m.abs() + y_m.abs() + travel;
    // Explicitly budgets 256 binary64 roundings per midpoint substep. This is
    // separate from the analytic Lipschitz truncation bound below.
    let gamma = 256.0 * f64::EPSILON / (1.0 - 256.0 * f64::EPSILON);
    outward_mul_nonnegative(gamma, scale)
}

#[derive(Clone, Copy, Debug)]
enum StepIntegrationFailure {
    Rejected(RolloutRejection),
    Numerical {
        stage: NumericalStageV1,
        substep: usize,
    },
}

impl MpcSolver {
    fn integrate_step(
        &mut self,
        initial: SimState,
        pwm: ShadowPwmPair,
        horizon_step: usize,
    ) -> Result<SimState, StepIntegrationFailure> {
        let left = WheelResponse {
            initial_velocity_mps: initial.left_velocity_mps,
            target_velocity_mps: self.model.left.gain_mps_per_pwm_percent
                * f64::from(pwm.left().get()),
            time_constant_s: self.model.left.time_constant_s,
        };
        let right = WheelResponse {
            initial_velocity_mps: initial.right_velocity_mps,
            target_velocity_mps: self.model.right.gain_mps_per_pwm_percent
                * f64::from(pwm.right().get()),
            time_constant_s: self.model.right.time_constant_s,
        };
        if !left.target_velocity_mps.is_finite() || !right.target_velocity_mps.is_finite() {
            return Err(StepIntegrationFailure::Numerical {
                stage: NumericalStageV1::FirstOrderResponse,
                substep: 0,
            });
        }
        let mut x_m = initial.x_m;
        let mut y_m = initial.y_m;
        let mut error_m = initial.integration_tube_radius_m;
        for substep in 0..self.config.integration_substeps {
            let t0 = substep as f64 * self.substep_s;
            let t1 = (substep + 1) as f64 * self.substep_s;
            let midpoint = t0 + 0.5 * self.substep_s;
            let Some((left0, left_mid, left1, right0, right_mid, right1)) = left
                .velocity_at(t0)
                .zip(left.velocity_at(midpoint))
                .zip(left.velocity_at(t1))
                .zip(right.velocity_at(t0))
                .zip(right.velocity_at(midpoint))
                .zip(right.velocity_at(t1))
                .map(|(((((l0, lm), l1), r0), rm), r1)| (l0, lm, l1, r0, rm, r1))
            else {
                return Err(StepIntegrationFailure::Numerical {
                    stage: NumericalStageV1::FirstOrderResponse,
                    substep,
                });
            };
            let location = EnvelopeLocationV1::Rollout {
                horizon_step,
                integration_substep: substep,
            };
            for value in [left0, left1] {
                if !self.model.left_velocity.contains(value) {
                    return Err(StepIntegrationFailure::Rejected(
                        RolloutRejection::PlantEnvelope(
                            PlantEnvelopeViolationV1::LeftWheelVelocity {
                                location,
                                value_mps: value,
                                minimum_mps: self.model.left_velocity.min,
                                maximum_mps: self.model.left_velocity.max,
                            },
                        ),
                    ));
                }
            }
            for value in [right0, right1] {
                if !self.model.right_velocity.contains(value) {
                    return Err(StepIntegrationFailure::Rejected(
                        RolloutRejection::PlantEnvelope(
                            PlantEnvelopeViolationV1::RightWheelVelocity {
                                location,
                                value_mps: value,
                                minimum_mps: self.model.right_velocity.min,
                                maximum_mps: self.model.right_velocity.max,
                            },
                        ),
                    ));
                }
            }
            let left_min = next_down(left0.min(left1));
            let left_max = next_up(left0.max(left1));
            let right_min = next_down(right0.min(right1));
            let right_max = next_up(right0.max(right1));
            let Some(max_abs_yaw_rate) =
                conservative_yaw_rate_bound(left0, left1, right0, right1, self.model.wheelbase_m)
            else {
                return Err(StepIntegrationFailure::Numerical {
                    stage: NumericalStageV1::YawEnvelope,
                    substep,
                });
            };
            if max_abs_yaw_rate > self.model.max_abs_yaw_rate_rad_s {
                return Err(StepIntegrationFailure::Rejected(
                    RolloutRejection::PlantEnvelope(PlantEnvelopeViolationV1::YawRate {
                        location,
                        maximum_observed_abs_rad_s: max_abs_yaw_rate,
                        allowed_abs_rad_s: self.model.max_abs_yaw_rate_rad_s,
                    }),
                ));
            }
            let Some(left_distance_mid) = left.distance_at(midpoint) else {
                return Err(StepIntegrationFailure::Numerical {
                    stage: NumericalStageV1::FirstOrderResponse,
                    substep,
                });
            };
            let Some(right_distance_mid) = right.distance_at(midpoint) else {
                return Err(StepIntegrationFailure::Numerical {
                    stage: NumericalStageV1::FirstOrderResponse,
                    substep,
                });
            };
            let yaw_mid =
                initial.yaw_rad + (right_distance_mid - left_distance_mid) / self.model.wheelbase_m;
            let center_velocity_mid = 0.5 * (left_mid + right_mid);
            if !yaw_mid.is_finite() || !center_velocity_mid.is_finite() {
                return Err(StepIntegrationFailure::Numerical {
                    stage: NumericalStageV1::MidpointIntegration,
                    substep,
                });
            }
            let (sin_yaw, cos_yaw) = yaw_mid.sin_cos();
            let delta_x = self.substep_s * center_velocity_mid * cos_yaw;
            let delta_y = self.substep_s * center_velocity_mid * sin_yaw;
            let next_x_m = x_m + delta_x;
            let next_y_m = y_m + delta_y;
            if !next_x_m.is_finite() || !next_y_m.is_finite() {
                return Err(StepIntegrationFailure::Numerical {
                    stage: NumericalStageV1::MidpointIntegration,
                    substep,
                });
            }
            let max_left_abs = left_min.abs().max(left_max.abs());
            let max_right_abs = right_min.abs().max(right_max.abs());
            let max_center_speed = next_up(0.5 * next_up(max_left_abs + max_right_abs));
            let max_left_derivative = outward_abs_difference(left.target_velocity_mps, left0)
                .and_then(|difference| outward_div_nonnegative(difference, left.time_constant_s))
                .ok_or(StepIntegrationFailure::Numerical {
                    stage: NumericalStageV1::IntegrationTube,
                    substep,
                })?;
            let max_right_derivative = outward_abs_difference(right.target_velocity_mps, right0)
                .and_then(|difference| outward_div_nonnegative(difference, right.time_constant_s))
                .ok_or(StepIntegrationFailure::Numerical {
                    stage: NumericalStageV1::IntegrationTube,
                    substep,
                })?;
            let max_center_acceleration =
                next_up(0.5 * next_up(max_left_derivative + max_right_derivative));
            let turn_component = outward_mul_nonnegative(max_center_speed, max_abs_yaw_rate)
                .ok_or(StepIntegrationFailure::Numerical {
                    stage: NumericalStageV1::IntegrationTube,
                    substep,
                })?;
            let lipschitz = outward_add_nonnegative(max_center_acceleration, turn_component)
                .ok_or(StepIntegrationFailure::Numerical {
                    stage: NumericalStageV1::IntegrationTube,
                    substep,
                })?;
            let h_squared = outward_mul_nonnegative(self.substep_s, self.substep_s).ok_or(
                StepIntegrationFailure::Numerical {
                    stage: NumericalStageV1::IntegrationTube,
                    substep,
                },
            )?;
            let truncation = outward_mul_nonnegative(lipschitz, 0.25 * h_squared).ok_or(
                StepIntegrationFailure::Numerical {
                    stage: NumericalStageV1::IntegrationTube,
                    substep,
                },
            )?;
            let rounding =
                numeric_position_roundoff_slack(x_m, y_m, max_center_speed, self.substep_s).ok_or(
                    StepIntegrationFailure::Numerical {
                        stage: NumericalStageV1::IntegrationTube,
                        substep,
                    },
                )?;
            let radius = outward_add_nonnegative(error_m, truncation)
                .and_then(|value| outward_add_nonnegative(value, rounding))
                .ok_or(StepIntegrationFailure::Numerical {
                    stage: NumericalStageV1::IntegrationTube,
                    substep,
                })?;
            if radius > self.config.max_integration_tube_radius_m {
                return Err(StepIntegrationFailure::Rejected(
                    RolloutRejection::IntegrationTubeExceeded {
                        horizon_step,
                        integration_substep: substep,
                        required_m: radius,
                    },
                ));
            }
            let centerline_start =
                PlanarPoint::try_new(x_m, y_m).map_err(|_| StepIntegrationFailure::Numerical {
                    stage: NumericalStageV1::IntegrationTube,
                    substep,
                })?;
            let centerline_end = PlanarPoint::try_new(next_x_m, next_y_m).map_err(|_| {
                StepIntegrationFailure::Numerical {
                    stage: NumericalStageV1::IntegrationTube,
                    substep,
                }
            })?;
            let global_start_s = horizon_step as f64 * self.config.dt_s + t0;
            let global_end_s = horizon_step as f64 * self.config.dt_s + t1;
            if !global_start_s.is_finite() || !global_end_s.is_finite() {
                return Err(StepIntegrationFailure::Numerical {
                    stage: NumericalStageV1::IntegrationTube,
                    substep,
                });
            }
            let capsule_index = horizon_step * self.config.integration_substeps + substep;
            self.trial_capsules[capsule_index] = ConservativeCapsuleSegmentV1 {
                purpose: CapsulePurposeV1::PredictedMotion,
                horizon_step,
                integration_substep: substep,
                time_start_s: global_start_s,
                time_end_s: global_end_s,
                centerline_start,
                centerline_end,
                extra_radius_m: radius,
            };
            x_m = next_x_m;
            y_m = next_y_m;
            error_m = radius;
        }
        let Some(left_final) = left.velocity_at(self.config.dt_s) else {
            return Err(StepIntegrationFailure::Numerical {
                stage: NumericalStageV1::FirstOrderResponse,
                substep: self.config.integration_substeps - 1,
            });
        };
        let Some(right_final) = right.velocity_at(self.config.dt_s) else {
            return Err(StepIntegrationFailure::Numerical {
                stage: NumericalStageV1::FirstOrderResponse,
                substep: self.config.integration_substeps - 1,
            });
        };
        let Some(left_distance) = left.distance_at(self.config.dt_s) else {
            return Err(StepIntegrationFailure::Numerical {
                stage: NumericalStageV1::FirstOrderResponse,
                substep: self.config.integration_substeps - 1,
            });
        };
        let Some(right_distance) = right.distance_at(self.config.dt_s) else {
            return Err(StepIntegrationFailure::Numerical {
                stage: NumericalStageV1::FirstOrderResponse,
                substep: self.config.integration_substeps - 1,
            });
        };
        let yaw_rad = normalize_angle(
            initial.yaw_rad + (right_distance - left_distance) / self.model.wheelbase_m,
        );
        if !yaw_rad.is_finite() {
            return Err(StepIntegrationFailure::Numerical {
                stage: NumericalStageV1::MidpointIntegration,
                substep: self.config.integration_substeps - 1,
            });
        }
        Ok(SimState {
            x_m,
            y_m,
            yaw_rad,
            left_velocity_mps: left_final,
            right_velocity_mps: right_final,
            integration_tube_radius_m: error_m,
        })
    }
}

impl MpcSolver {
    pub fn solve<'solver, 'reference, Q, C>(
        &'solver mut self,
        request: MpcRequestV1<'reference>,
        collision: &mut Q,
        clock: &mut C,
    ) -> Result<MpcSolution<'solver, 'reference>, MpcSolveError<'reference, Q::Error>>
    where
        Q: CollisionQuery,
        C: HostMonotonicClock,
    {
        let started_at = match clock.try_now() {
            Ok(started_at) => started_at,
            Err(source) => {
                return Err(self.failure_before_start(
                    request,
                    MpcFailureKind::Clock(HostMonotonicClockFailure::Read(source)),
                ));
            }
        };
        let mut status = SolveStatusV1 {
            started_at,
            observed_at: started_at,
            deadline: request.deadline,
            completed_iterations: 0,
            active_iteration: None,
            rollout_evaluations: 0,
            pre_final_collision_queries: 0,
            final_validation_queries: 0,
        };
        if let Some(fault) = clock_fault(request, started_at, started_at) {
            return Err(self.failure(
                request,
                status,
                MpcFailureKind::Clock(HostMonotonicClockFailure::Fault(fault)),
            ));
        }
        if request.reference.points.len() != self.config.horizon
            || request.reference.step_period_s.to_bits() != self.config.dt_s.to_bits()
        {
            return Err(self.failure(
                request,
                status,
                MpcFailureKind::ReferenceDoesNotMatchConfig {
                    expected_steps: self.config.horizon,
                    actual_steps: request.reference.points.len(),
                    expected_period_s: self.config.dt_s,
                    actual_period_s: request.reference.step_period_s,
                },
            ));
        }
        let actual_snapshot = collision.snapshot_provenance();
        if actual_snapshot != request.collision_snapshot {
            return Err(self.failure(
                request,
                status,
                MpcFailureKind::CollisionSnapshotMismatch {
                    requested: Box::new(request.collision_snapshot),
                    actual: Box::new(actual_snapshot),
                },
            ));
        }
        if let Err(wheel) = self.validate_previous_pwm(request.previous_pwm) {
            let value = match wheel {
                WheelSide::Left => request.previous_pwm.left(),
                WheelSide::Right => request.previous_pwm.right(),
            };
            return Err(self.failure(
                request,
                status,
                MpcFailureKind::PreviousPwmOutsideEnvelope { wheel, value },
            ));
        }
        let initial = match self.initial_sim_state(request.state) {
            Ok(state) => state,
            Err(violation) => {
                return Err(self.failure(
                    request,
                    status,
                    MpcFailureKind::PlantEnvelope(violation),
                ));
            }
        };
        self.check_clock(request, clock, &mut status)?;
        let start_capsule = ConservativeCapsuleSegmentV1::stationary_start(request.state.pose);
        match self.query_capsule(request, collision, clock, &mut status, start_capsule, false)? {
            true => {}
            false => return Err(self.failure(request, status, MpcFailureKind::OccupiedStart)),
        }
        self.commands.fill(request.previous_pwm);
        let mut radius = u16::from(self.config.initial_search_radius_percent);
        for iteration in 0..self.config.iterations {
            status.active_iteration = Some(iteration);
            for step in 0..self.config.horizon {
                self.check_clock(request, clock, &mut status)?;
                let center = self.commands[step];
                let mut best: Option<(f64, ShadowPwmPair)> = None;
                let mut rejection = None;
                for left_index in 0..self.config.lattice {
                    let left = lattice_pwm(
                        center.left(),
                        radius,
                        left_index,
                        self.config.lattice,
                        self.config.left_pwm,
                    );
                    if lattice_duplicate(
                        center.left(),
                        radius,
                        left_index,
                        self.config.lattice,
                        self.config.left_pwm,
                        left,
                    ) {
                        continue;
                    }
                    for right_index in 0..self.config.lattice {
                        let right = lattice_pwm(
                            center.right(),
                            radius,
                            right_index,
                            self.config.lattice,
                            self.config.right_pwm,
                        );
                        if lattice_duplicate(
                            center.right(),
                            radius,
                            right_index,
                            self.config.lattice,
                            self.config.right_pwm,
                            right,
                        ) {
                            continue;
                        }
                        let candidate = ShadowPwmPair::from_validated(left, right);
                        self.commands[step] = candidate;
                        if !self.slew_valid_at(step, request.previous_pwm) {
                            continue;
                        }
                        match self.evaluate_rollout(
                            initial,
                            request,
                            collision,
                            clock,
                            &mut status,
                        )? {
                            RolloutEvaluation::Feasible(cost) => {
                                let replace = match best {
                                    None => true,
                                    Some((best_cost, best_pwm)) => {
                                        cost.total_cmp(&best_cost).is_lt()
                                            || (cost.to_bits() == best_cost.to_bits()
                                                && pwm_lex_less(candidate, best_pwm))
                                    }
                                };
                                if replace {
                                    best = Some((cost, candidate));
                                }
                            }
                            RolloutEvaluation::Rejected(reason) => rejection = Some(reason),
                        }
                    }
                }
                if let Some((_, selected)) = best {
                    self.commands[step] = selected;
                } else {
                    self.commands[step] = center;
                    let reason = rejection.expect("the center lattice command is slew-valid");
                    return Err(self.rejection_failure(request, status, reason));
                }
            }
            status.completed_iterations = iteration + 1;
            status.active_iteration = None;
            radius = radius.saturating_mul(u16::from(self.config.radius_decay_numerator))
                / u16::from(self.config.radius_decay_denominator);
            if radius == 0 {
                break;
            }
        }
        self.begin_rollout(request, status)?;
        // Admission charges one bounded attempt before the body starts. A
        // body failure must retain that consumed budget rather than pretending
        // the attempt was never admitted.
        status.rollout_evaluations += 1;
        let objective_cost =
            match self.evaluate_rollout_body(initial, request, collision, clock, &mut status)? {
                RolloutEvaluation::Feasible(cost) => cost,
                RolloutEvaluation::Rejected(reason) => {
                    return Err(self.rejection_failure(request, status, reason));
                }
            };
        for step in 0..self.config.horizon {
            let state = self.trial_states[step];
            let pose = OdomPoseV1::try_new(state.x_m, state.y_m, state.yaw_rad).map_err(|_| {
                self.failure(
                    request,
                    status,
                    MpcFailureKind::Numerical {
                        stage: NumericalStageV1::PredictedTrajectory,
                        horizon_step: Some(step),
                        integration_substep: None,
                    },
                )
            })?;
            let forward_velocity_mps = 0.5 * (state.left_velocity_mps + state.right_velocity_mps);
            let yaw_rate_rad_s =
                (state.right_velocity_mps - state.left_velocity_mps) / self.model.wheelbase_m;
            if !forward_velocity_mps.is_finite() || !yaw_rate_rad_s.is_finite() {
                return Err(self.failure(
                    request,
                    status,
                    MpcFailureKind::Numerical {
                        stage: NumericalStageV1::PredictedTrajectory,
                        horizon_step: Some(step),
                        integration_substep: None,
                    },
                ));
            }
            self.predicted_trajectory[step] = PredictedOdomPointV1 {
                time_from_start_s: (step + 1) as f64 * self.config.dt_s,
                pose,
                left_velocity_mps: state.left_velocity_mps,
                right_velocity_mps: state.right_velocity_mps,
                forward_velocity_mps,
                yaw_rate_rad_s,
                requested_pwm: self.commands[step],
                integration_tube_radius_m: state.integration_tube_radius_m,
            };
        }
        // This deliberately re-queries every selected capsule after all
        // optimizer and selected-rollout queries have completed.
        let capsule_count = self.config.horizon * self.config.integration_substeps;
        for index in 0..capsule_count {
            self.check_clock(request, clock, &mut status)?;
            let capsule = self.trial_capsules[index];
            if !self.query_capsule(request, collision, clock, &mut status, capsule, true)? {
                return Err(self.failure(
                    request,
                    status,
                    MpcFailureKind::FinalTrajectoryBlocked {
                        horizon_step: capsule.horizon_step,
                        integration_substep: capsule.integration_substep,
                    },
                ));
            }
        }
        self.check_clock(request, clock, &mut status)?;
        let final_validation = FinalTrajectoryValidationV1 {
            collision_snapshot: request.collision_snapshot,
            segment_count: capsule_count,
            validated_at: status.observed_at,
        };
        Ok(MpcSolution {
            model: self.model,
            config: self.config,
            request,
            requested_pwm: self.commands[0],
            command_sequence: &self.commands[..self.config.horizon],
            predicted_trajectory: &self.predicted_trajectory[..self.config.horizon],
            conservative_capsules: &self.trial_capsules[..capsule_count],
            final_validation,
            status,
            objective_cost,
        })
    }

    fn begin_rollout<'reference, E>(
        &self,
        request: MpcRequestV1<'reference>,
        status: SolveStatusV1,
    ) -> Result<(), MpcSolveError<'reference, E>> {
        if status.rollout_evaluations >= self.config.evaluation_limit {
            Err(self.failure(
                request,
                status,
                MpcFailureKind::EvaluationLimit {
                    configured: self.config.evaluation_limit,
                },
            ))
        } else {
            Ok(())
        }
    }

    fn evaluate_rollout<'reference, Q, C>(
        &mut self,
        initial: SimState,
        request: MpcRequestV1<'reference>,
        collision: &mut Q,
        clock: &mut C,
        status: &mut SolveStatusV1,
    ) -> Result<RolloutEvaluation, MpcSolveError<'reference, Q::Error>>
    where
        Q: CollisionQuery,
        C: HostMonotonicClock,
    {
        self.begin_rollout(request, *status)?;
        // This is an admitted, budget-charged attempt, not a completion
        // counter. Increment before entering the fallible rollout body.
        status.rollout_evaluations += 1;
        self.evaluate_rollout_body(initial, request, collision, clock, status)
    }

    fn evaluate_rollout_body<'reference, Q, C>(
        &mut self,
        initial: SimState,
        request: MpcRequestV1<'reference>,
        collision: &mut Q,
        clock: &mut C,
        status: &mut SolveStatusV1,
    ) -> Result<RolloutEvaluation, MpcSolveError<'reference, Q::Error>>
    where
        Q: CollisionQuery,
        C: HostMonotonicClock,
    {
        let mut state = initial;
        let mut previous = request.previous_pwm;
        let mut cost = CompensatedSum::new();
        for step in 0..self.config.horizon {
            self.check_clock(request, clock, status)?;
            let command = self.commands[step];
            state = match self.integrate_step(state, command, step) {
                Ok(next) => next,
                Err(StepIntegrationFailure::Rejected(reason)) => {
                    return Ok(RolloutEvaluation::Rejected(reason));
                }
                Err(StepIntegrationFailure::Numerical { stage, substep }) => {
                    return Err(self.failure(
                        request,
                        *status,
                        MpcFailureKind::Numerical {
                            stage,
                            horizon_step: Some(step),
                            integration_substep: Some(substep),
                        },
                    ));
                }
            };
            let base = step * self.config.integration_substeps;
            for substep in 0..self.config.integration_substeps {
                let capsule = self.trial_capsules[base + substep];
                if !self.query_capsule(request, collision, clock, status, capsule, false)? {
                    return Ok(RolloutEvaluation::Rejected(
                        RolloutRejection::CollisionBlocked {
                            horizon_step: step,
                            integration_substep: substep,
                        },
                    ));
                }
            }
            if !self.add_step_cost(
                &mut cost,
                state,
                request.reference.points[step],
                command,
                previous,
                step,
            ) {
                return Err(self.failure(
                    request,
                    *status,
                    MpcFailureKind::Numerical {
                        stage: NumericalStageV1::CostEvaluation,
                        horizon_step: Some(step),
                        integration_substep: None,
                    },
                ));
            }
            self.trial_states[step] = state;
            previous = command;
        }
        let total = cost.total();
        if total.is_finite() {
            Ok(RolloutEvaluation::Feasible(total))
        } else {
            Err(self.failure(
                request,
                *status,
                MpcFailureKind::Numerical {
                    stage: NumericalStageV1::CostEvaluation,
                    horizon_step: None,
                    integration_substep: None,
                },
            ))
        }
    }

    fn add_step_cost(
        &self,
        sum: &mut CompensatedSum,
        state: SimState,
        reference: OdomReferencePointV1,
        command: ShadowPwmPair,
        previous: ShadowPwmPair,
        step: usize,
    ) -> bool {
        let terminal = step + 1 == self.config.horizon;
        let roots = self.cost_roots;
        let position_root = if terminal {
            roots.terminal_position
        } else {
            roots.position
        };
        if position_root != 0.0
            && (!add_scaled_square(
                sum,
                position_root,
                state.x_m - reference.pose.position.x_m(),
            ) || !add_scaled_square(
                sum,
                position_root,
                state.y_m - reference.pose.position.y_m(),
            ))
        {
            return false;
        }
        let heading_root = if terminal {
            roots.terminal_heading
        } else {
            roots.heading
        };
        if heading_root != 0.0 {
            let heading = normalize_angle(state.yaw_rad - reference.pose.yaw_rad);
            if !add_scaled_square(sum, heading_root, heading) {
                return false;
            }
        }
        let velocity_root = if terminal {
            roots.terminal_forward_velocity
        } else {
            roots.forward_velocity
        };
        if velocity_root != 0.0 {
            let forward = 0.5 * (state.left_velocity_mps + state.right_velocity_mps);
            if !add_scaled_square(sum, velocity_root, forward - reference.forward_velocity_mps) {
                return false;
            }
        }
        let yaw_root = if terminal {
            roots.terminal_yaw_rate
        } else {
            roots.yaw_rate
        };
        if yaw_root != 0.0 {
            let yaw_rate =
                (state.right_velocity_mps - state.left_velocity_mps) / self.model.wheelbase_m;
            if !add_scaled_square(sum, yaw_root, yaw_rate - reference.yaw_rate_rad_s) {
                return false;
            }
        }
        if roots.pwm != 0.0
            && (!add_scaled_square(sum, roots.pwm, f64::from(command.left().get()))
                || !add_scaled_square(sum, roots.pwm, f64::from(command.right().get())))
        {
            return false;
        }
        if roots.slew != 0.0 {
            let left_delta =
                f64::from(i16::from(command.left().get()) - i16::from(previous.left().get()));
            let right_delta =
                f64::from(i16::from(command.right().get()) - i16::from(previous.right().get()));
            if !add_scaled_square(sum, roots.slew, left_delta)
                || !add_scaled_square(sum, roots.slew, right_delta)
            {
                return false;
            }
        }
        true
    }

    fn slew_valid_at(&self, step: usize, previous: ShadowPwmPair) -> bool {
        let command = self.commands[step];
        let before = if step == 0 {
            previous
        } else {
            self.commands[step - 1]
        };
        if percent_delta(command.left(), before.left()) > self.config.left_slew_percent_per_step
            || percent_delta(command.right(), before.right())
                > self.config.right_slew_percent_per_step
        {
            return false;
        }
        if step + 1 == self.config.horizon {
            return true;
        }
        let after = self.commands[step + 1];
        percent_delta(after.left(), command.left()) <= self.config.left_slew_percent_per_step
            && percent_delta(after.right(), command.right())
                <= self.config.right_slew_percent_per_step
    }

    fn check_clock<'reference, E, C>(
        &self,
        request: MpcRequestV1<'reference>,
        clock: &mut C,
        status: &mut SolveStatusV1,
    ) -> Result<(), MpcSolveError<'reference, E>>
    where
        C: HostMonotonicClock,
    {
        let previous = status.observed_at;
        let observed = match clock.try_now() {
            Ok(observed) => observed,
            Err(source) => {
                return Err(self.failure(
                    request,
                    *status,
                    MpcFailureKind::Clock(HostMonotonicClockFailure::Read(source)),
                ));
            }
        };
        status.observed_at = observed;
        match clock_fault(request, previous, observed) {
            Some(fault) => Err(self.failure(
                request,
                *status,
                MpcFailureKind::Clock(HostMonotonicClockFailure::Fault(fault)),
            )),
            None => Ok(()),
        }
    }

    fn query_capsule<'reference, Q, C>(
        &self,
        request: MpcRequestV1<'reference>,
        collision: &mut Q,
        clock: &mut C,
        status: &mut SolveStatusV1,
        capsule: ConservativeCapsuleSegmentV1,
        final_revalidation: bool,
    ) -> Result<bool, MpcSolveError<'reference, Q::Error>>
    where
        Q: CollisionQuery,
        C: HostMonotonicClock,
    {
        // Count the invocation before calling the fallible query so a query
        // error or the immediately following clock-read failure retains that
        // the query method actually ran.
        if final_revalidation {
            status.final_validation_queries += 1;
        } else {
            status.pre_final_collision_queries += 1;
        }
        let query_result = collision.is_capsule_traversable(capsule);
        let previous = status.observed_at;
        let clock_result = match clock.try_now() {
            Ok(observed) => {
                status.observed_at = observed;
                clock_fault(request, previous, observed).map(HostMonotonicClockFailure::Fault)
            }
            Err(source) => Some(HostMonotonicClockFailure::Read(source)),
        };
        match (query_result, clock_result) {
            (Ok(value), None) => Ok(value),
            (Ok(_), Some(clock)) => Err(self.failure(
                request,
                *status,
                MpcFailureKind::CollisionObservation {
                    horizon_step: (capsule.purpose == CapsulePurposeV1::PredictedMotion)
                        .then_some(capsule.horizon_step),
                    integration_substep: (capsule.purpose == CapsulePurposeV1::PredictedMotion)
                        .then_some(capsule.integration_substep),
                    final_revalidation,
                    source: CollisionObservationFailure::Clock(clock),
                },
            )),
            (Err(query), None) => Err(self.failure(
                request,
                *status,
                MpcFailureKind::CollisionObservation {
                    horizon_step: (capsule.purpose == CapsulePurposeV1::PredictedMotion)
                        .then_some(capsule.horizon_step),
                    integration_substep: (capsule.purpose == CapsulePurposeV1::PredictedMotion)
                        .then_some(capsule.integration_substep),
                    final_revalidation,
                    source: CollisionObservationFailure::Query(query),
                },
            )),
            (Err(query), Some(clock)) => Err(self.failure(
                request,
                *status,
                MpcFailureKind::CollisionObservation {
                    horizon_step: (capsule.purpose == CapsulePurposeV1::PredictedMotion)
                        .then_some(capsule.horizon_step),
                    integration_substep: (capsule.purpose == CapsulePurposeV1::PredictedMotion)
                        .then_some(capsule.integration_substep),
                    final_revalidation,
                    source: CollisionObservationFailure::QueryAndClock { query, clock },
                },
            )),
        }
    }

    fn failure<'reference, E>(
        &self,
        request: MpcRequestV1<'reference>,
        status: SolveStatusV1,
        kind: MpcFailureKind<E>,
    ) -> MpcSolveError<'reference, E> {
        Box::new(MpcFailure {
            model: self.model,
            config: self.config,
            request,
            progress: MpcSolveProgressV1::InProgress(status),
            kind,
        })
    }

    fn failure_before_start<'reference, E>(
        &self,
        request: MpcRequestV1<'reference>,
        kind: MpcFailureKind<E>,
    ) -> MpcSolveError<'reference, E> {
        Box::new(MpcFailure {
            model: self.model,
            config: self.config,
            request,
            progress: MpcSolveProgressV1::NotStarted,
            kind,
        })
    }

    fn rejection_failure<'reference, E>(
        &self,
        request: MpcRequestV1<'reference>,
        status: SolveStatusV1,
        rejection: RolloutRejection,
    ) -> MpcSolveError<'reference, E> {
        let kind = match rejection {
            RolloutRejection::PlantEnvelope(value) => MpcFailureKind::PlantEnvelope(value),
            RolloutRejection::CollisionBlocked {
                horizon_step,
                integration_substep,
            } => MpcFailureKind::CollisionBlocked {
                horizon_step,
                integration_substep,
            },
            RolloutRejection::IntegrationTubeExceeded {
                horizon_step,
                integration_substep,
                required_m,
            } => MpcFailureKind::IntegrationTubeExceeded {
                horizon_step,
                integration_substep,
                required_m,
                allowed_m: self.config.max_integration_tube_radius_m,
            },
        };
        self.failure(request, status, kind)
    }
}

fn clock_fault(
    request: MpcRequestV1<'_>,
    previous: HostMonotonicTimestamp,
    observed: HostMonotonicTimestamp,
) -> Option<ClockFault> {
    if observed < previous {
        Some(ClockFault::Regression {
            previous,
            observed_at: observed,
        })
    } else if observed < request.submitted_at {
        Some(ClockFault::BeforeRequestSubmission {
            submitted_at: request.submitted_at,
            observed_at: observed,
        })
    } else if observed >= request.deadline {
        Some(ClockFault::DeadlineReached {
            deadline: request.deadline,
            observed_at: observed,
        })
    } else {
        None
    }
}

fn add_scaled_square(sum: &mut CompensatedSum, root_weight: f64, residual: f64) -> bool {
    let scaled = root_weight * residual;
    let square = scaled * scaled;
    scaled.is_finite() && square.is_finite() && sum.add(square)
}

fn conservative_yaw_rate_bound(
    left_start: f64,
    left_end: f64,
    right_start: f64,
    right_end: f64,
    wheelbase_m: f64,
) -> Option<f64> {
    if !left_start.is_finite()
        || !left_end.is_finite()
        || !right_start.is_finite()
        || !right_end.is_finite()
        || !wheelbase_m.is_finite()
        || wheelbase_m <= 0.0
    {
        return None;
    }
    let left_min = next_down(left_start.min(left_end));
    let left_max = next_up(left_start.max(left_end));
    let right_min = next_down(right_start.min(right_end));
    let right_max = next_up(right_start.max(right_end));
    let difference_min = next_down(right_min - left_max);
    let difference_max = next_up(right_max - left_min);
    let bound = next_up(difference_min.abs().max(difference_max.abs()) / wheelbase_m);
    bound.is_finite().then_some(bound)
}

fn percent_delta(left: PwmPercent, right: PwmPercent) -> u16 {
    i16::from(left.get()).abs_diff(i16::from(right.get()))
}

fn lattice_pwm(
    center: PwmPercent,
    radius: u16,
    index: u8,
    count: u8,
    range: PwmRange,
) -> PwmPercent {
    let half = i32::from(count / 2);
    let offset = i32::from(index) - half;
    let numerator = offset * i32::from(radius);
    let delta = if numerator >= 0 {
        (numerator + half / 2) / half
    } else {
        (numerator - half / 2) / half
    };
    let raw = (i32::from(center.get()) + delta)
        .clamp(i32::from(range.min.get()), i32::from(range.max.get()));
    PwmPercent::try_new(raw as i8).expect("configured PWM range is canonical")
}

fn lattice_duplicate(
    center: PwmPercent,
    radius: u16,
    index: u8,
    count: u8,
    range: PwmRange,
    value: PwmPercent,
) -> bool {
    (0..index).any(|earlier| lattice_pwm(center, radius, earlier, count, range) == value)
}

fn pwm_lex_less(left: ShadowPwmPair, right: ShadowPwmPair) -> bool {
    left.left().get() < right.left().get()
        || (left.left() == right.left() && left.right().get() < right.right().get())
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::convert::Infallible;
    use std::time::Duration;

    use super::super::{
        BaseToOdom, GlobalPlanner, GlobalPlannerConfig, LocalCostmap, LocalCostmapConfig,
        LocalDepthObservation, MapPoint, PlanStart, PointGoal, TimeAlignedOdomPose, TimeAlignment,
        TrackingCameraToBase, UnknownSpacePolicy,
    };
    use super::*;
    use crate::dense::occupancy::{
        DepthCameraModel, DepthRangeMeters, DepthToTrackingCamera, HeightRangeMeters,
        OccupancyConfig, OccupancyEvidenceModel, OccupancyGridGeometry, OccupancyMapper,
        WorldToOccupancy,
    };
    use crate::map::SlamMap;
    use crate::{
        DepthImage, DepthObservation, FrameDimensions, FrameId, PinholeIntrinsics, Pose, Timestamp,
    };

    fn pwm(left: i8, right: i8) -> ShadowPwmPair {
        ShadowPwmPair::from_validated(
            PwmPercent::try_new(left).expect("canonical test left PWM"),
            PwmPercent::try_new(right).expect("canonical test right PWM"),
        )
    }

    fn plant_dto(dt_s: f64, left_tau_s: f64, right_tau_s: f64) -> PlantModelV1Dto {
        PlantModelV1Dto {
            schema_version: PLANT_MODEL_V1,
            model_id: "test-plant".into(),
            model_version: 1,
            sample_period_s: dt_s,
            wheelbase_m: 0.5,
            left: WheelPlantV1Dto {
                velocity_gain_mps_per_pwm_percent: 0.01,
                time_constant_s: left_tau_s,
            },
            right: WheelPlantV1Dto {
                velocity_gain_mps_per_pwm_percent: 0.01,
                time_constant_s: right_tau_s,
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
                fixture_id: "unit".into(),
                generator_id: "hand".into(),
            },
        }
    }

    fn config_dto(dt_s: f64, horizon: u16, substeps: u16) -> MpcConfigV1Dto {
        MpcConfigV1Dto {
            schema_version: MPC_CONFIG_V1,
            horizon_steps: horizon,
            step_period_s: dt_s,
            integration_substeps: substeps,
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
        }
    }

    fn dimensions(width: u32, height: u32) -> FrameDimensions {
        FrameDimensions::try_new(width, height).expect("test dimensions")
    }

    fn camera(width: u32, height: u32) -> DepthCameraModel {
        DepthCameraModel::new(
            PinholeIntrinsics::try_new(
                4.0,
                4.0,
                width.saturating_sub(1) as f32 * 0.5,
                height.saturating_sub(1) as f32 * 0.5,
            )
            .expect("test intrinsics"),
            dimensions(width, height),
            DepthToTrackingCamera::identity(),
        )
    }

    fn global_path_fixture() -> (MapSnapshot, GlobalPath) {
        let map = SlamMap::new();
        let map_snapshot = map.snapshot();
        let config = OccupancyConfig::try_new(
            OccupancyGridGeometry::try_new(1.0, [-2.0, -2.0], 6, 6, 36).expect("geometry"),
            WorldToOccupancy::level_optical_world(1.0).expect("occupancy frame"),
            camera(9, 5),
            HeightRangeMeters::try_new(0.0, 2.0).expect("height"),
            DepthRangeMeters::try_new(0.1, 8.0).expect("depth"),
            1,
            OccupancyEvidenceModel::try_new(-1, 3, -1, 1).expect("evidence"),
            1,
        )
        .expect("occupancy config");
        let mut mapper = OccupancyMapper::try_new(config).expect("mapper");
        mapper
            .reset_to_map(map_snapshot.instance_id())
            .expect("map identity");
        let occupancy = mapper.snapshot().expect("snapshot");
        let mut planner = GlobalPlanner::try_new(
            &occupancy,
            GlobalPlannerConfig::try_new(0.0, UnknownSpacePolicy::Traversable)
                .expect("planner config"),
        )
        .expect("planner");
        let start_point = MapPoint::try_new(0.0, 0.0).expect("start");
        let goal_point = MapPoint::try_new(1.0, 0.0).expect("goal");
        let start = PlanStart::for_snapshot(start_point, &occupancy).expect("start provenance");
        let goal = PointGoal::for_snapshot(goal_point, &occupancy).expect("goal provenance");
        let path = planner.plan(start, goal).expect("global path");
        (map_snapshot, path)
    }

    fn optical_to_base() -> TrackingCameraToBase {
        TrackingCameraToBase::new(
            Pose::try_from_rt(
                [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
                [0.0, 0.0, 0.5],
            )
            .expect("proper optical-to-base transform"),
        )
    }

    fn local_costmap_fixture(
        session: DeviceSessionId,
        odom_segment_id: OdomSegmentId,
        host_ns: u64,
        max_age_ns: u64,
    ) -> LocalCostmap {
        let geometry =
            OccupancyGridGeometry::try_new(0.25, [-1.0, -1.0], 12, 8, 96).expect("local geometry");
        let config = LocalCostmapConfig::try_new(
            geometry,
            camera(9, 5),
            optical_to_base(),
            HeightRangeMeters::try_new(0.1, 1.5).expect("obstacle height"),
            DepthRangeMeters::try_new(0.1, 8.0).expect("local depth"),
            1,
            0.1,
            0.0,
            Duration::from_nanos(max_age_ns),
        )
        .expect("local config");
        let mut costmap = LocalCostmap::try_new(config, session).expect("local costmap");
        let image = DepthImage::new(
            FrameId::new(1),
            Timestamp::from_nanos(1),
            9,
            5,
            vec![2.0; 45],
        )
        .expect("depth image");
        let source =
            DepthObservation::parse(session, HostMonotonicTimestamp::from_nanos(host_ns), image)
                .expect("depth provenance");
        let pose = TimeAlignedOdomPose::from_validated_parts_for_test(
            odom_segment_id,
            session,
            source.device_timestamp(),
            BaseToOdom::try_new(0.0, 0.0, 0.0).expect("capture pose"),
            TimeAlignment::ExactVisual,
        );
        costmap
            .update(
                LocalDepthObservation::try_from_time_aligned(source, pose)
                    .expect("exact time-aligned depth observation"),
            )
            .expect("costmap update");
        costmap
    }

    struct RuntimeFixture {
        path: GlobalPath,
        epoch: NavigationEpochV1,
        collision: CollisionSnapshotProvenanceV1,
        config: MpcConfigV1,
        model: PlantModelV1,
        costmap: LocalCostmap,
    }

    fn runtime_fixture(horizon: u16, substeps: u16) -> RuntimeFixture {
        let session = DeviceSessionId::try_new(1).expect("session");
        let (map_snapshot, path) = global_path_fixture();
        let epoch = NavigationEpochV1::from_runtime(
            session,
            OdomSegmentId::try_new(1).expect("odom segment"),
            map_snapshot,
            &path,
        )
        .expect("epoch");
        let costmap = local_costmap_fixture(session, epoch.odom_segment_id(), 10, 1_000_000);
        let view = costmap
            .view_at(HostMonotonicTimestamp::from_nanos(20))
            .expect("current collision view");
        let collision = CollisionSnapshotProvenanceV1::from_runtime(epoch, &view)
            .expect("collision provenance");
        RuntimeFixture {
            path,
            epoch,
            collision,
            config: MpcConfigV1::parse(config_dto(0.1, horizon, substeps)).expect("config"),
            model: PlantModelV1::parse(plant_dto(0.1, 0.4, 0.5)).expect("model"),
            costmap,
        }
    }

    fn state(epoch: NavigationEpochV1) -> OdomMotionStateV1 {
        OdomMotionStateV1::parse(
            OdomMotionStateV1Dto {
                schema_version: ODOM_MOTION_STATE_V1,
                observed_at_host_ns: 11,
                x_m: 0.0,
                y_m: 0.0,
                yaw_rad: 0.0,
                odom_velocity_x_mps: 0.0,
                odom_velocity_y_mps: 0.0,
                yaw_rate_rad_s: 0.0,
            },
            epoch,
        )
        .expect("state")
    }

    fn reference<'path>(
        fixture: &'path RuntimeFixture,
        forward_velocity_mps: f64,
    ) -> MpcReferenceV1<'path> {
        let points = (0..fixture.config.horizon)
            .map(|index| OdomReferencePointV1Dto {
                x_m: (index + 1) as f64 * 0.01,
                y_m: 0.0,
                yaw_rad: 0.0,
                forward_velocity_mps,
                yaw_rate_rad_s: 0.0,
            })
            .collect();
        MpcReferenceV1::parse(
            MpcReferenceV1Dto {
                schema_version: MPC_REFERENCE_V1,
                builder_revision: ReferenceBuilderRevisionV1::TimeParameterizedGlobalPathV1 as u32,
                created_at_host_ns: 12,
                step_period_s: fixture.config.dt_s,
                points,
            },
            fixture.config,
            fixture.epoch,
            &fixture.path,
        )
        .expect("reference")
    }

    fn request<'reference>(
        fixture: &'reference RuntimeFixture,
        reference: &'reference MpcReferenceV1<'reference>,
    ) -> MpcRequestV1<'reference> {
        MpcRequestV1::parse(
            MpcRequestV1Dto {
                schema_version: MPC_REQUEST_V1,
                request_id: 1,
                submitted_at_host_ns: 20,
                deadline_host_ns: 1000,
            },
            state(fixture.epoch),
            reference,
            ShadowPwmPair::STOP,
            fixture.collision,
        )
        .expect("request")
    }

    struct ConstantClock(HostMonotonicTimestamp);
    impl HostMonotonicClock for ConstantClock {
        fn try_now(&mut self) -> Result<HostMonotonicTimestamp, HostMonotonicClockReadError> {
            Ok(self.0)
        }
    }

    struct ScriptedClock {
        times: VecDeque<HostMonotonicTimestamp>,
        last: HostMonotonicTimestamp,
    }
    impl ScriptedClock {
        fn new(values: &[u64]) -> Self {
            let times = values
                .iter()
                .copied()
                .map(HostMonotonicTimestamp::from_nanos)
                .collect();
            Self {
                times,
                last: HostMonotonicTimestamp::from_nanos(*values.last().expect("clock script")),
            }
        }
    }
    impl HostMonotonicClock for ScriptedClock {
        fn try_now(&mut self) -> Result<HostMonotonicTimestamp, HostMonotonicClockReadError> {
            Ok(self.times.pop_front().unwrap_or(self.last))
        }
    }

    const INJECTED_CLOCK_READ_ERROR: HostMonotonicClockReadError =
        HostMonotonicClockReadError::ElapsedNanosecondsOutOfRange {
            elapsed_nanoseconds: 18_446_744_073_709_551_616,
        };

    struct ReadFailingClock {
        now: HostMonotonicTimestamp,
        fail_at_call: usize,
        calls: usize,
    }

    impl ReadFailingClock {
        fn new(now: HostMonotonicTimestamp, fail_at_call: usize) -> Self {
            Self {
                now,
                fail_at_call,
                calls: 0,
            }
        }
    }

    impl HostMonotonicClock for ReadFailingClock {
        fn try_now(&mut self) -> Result<HostMonotonicTimestamp, HostMonotonicClockReadError> {
            self.calls += 1;
            if self.calls == self.fail_at_call {
                Err(INJECTED_CLOCK_READ_ERROR)
            } else {
                Ok(self.now)
            }
        }
    }

    fn in_progress<E>(failure: &MpcFailure<'_, E>) -> SolveStatusV1 {
        match failure.progress() {
            MpcSolveProgressV1::NotStarted => panic!("solve unexpectedly never started"),
            MpcSolveProgressV1::InProgress(status) => status,
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum QueryError {
        Injected,
    }

    struct ScriptedQuery {
        snapshot: CollisionSnapshotProvenanceV1,
        calls: usize,
        error_on: Option<usize>,
        block_on: Option<usize>,
    }
    impl ScriptedQuery {
        fn clear(snapshot: CollisionSnapshotProvenanceV1) -> Self {
            Self {
                snapshot,
                calls: 0,
                error_on: None,
                block_on: None,
            }
        }
    }
    impl CollisionQuery for ScriptedQuery {
        type Error = QueryError;
        fn snapshot_provenance(&self) -> CollisionSnapshotProvenanceV1 {
            self.snapshot
        }
        fn is_capsule_traversable(
            &mut self,
            _: ConservativeCapsuleSegmentV1,
        ) -> Result<bool, Self::Error> {
            self.calls += 1;
            if self.error_on == Some(self.calls) {
                Err(QueryError::Injected)
            } else {
                Ok(self.block_on != Some(self.calls))
            }
        }
    }

    #[test]
    fn initial_clock_read_failure_has_no_fabricated_progress_or_work() {
        let fixture = runtime_fixture(1, 1);
        let reference = reference(&fixture, 0.0);
        let request = request(&fixture, &reference);
        let mut solver = MpcSolver::new(fixture.model, fixture.config).expect("solver");
        let mut query = ScriptedQuery::clear(fixture.collision);
        let mut clock = ReadFailingClock::new(HostMonotonicTimestamp::from_nanos(30), 1);

        let failure = solver
            .solve(request, &mut query, &mut clock)
            .expect_err("initial read failure");

        assert_eq!(clock.calls, 1);
        assert_eq!(query.calls, 0);
        assert_eq!(failure.progress(), MpcSolveProgressV1::NotStarted);
        assert!(matches!(
            failure.kind(),
            MpcFailureKind::Clock(HostMonotonicClockFailure::Read(source))
                if *source == INJECTED_CLOCK_READ_ERROR
        ));
    }

    #[test]
    fn second_clock_read_failure_retains_first_observation_and_zero_work() {
        let fixture = runtime_fixture(1, 1);
        let reference = reference(&fixture, 0.0);
        let request = request(&fixture, &reference);
        let mut solver = MpcSolver::new(fixture.model, fixture.config).expect("solver");
        let mut query = ScriptedQuery::clear(fixture.collision);
        let observed = HostMonotonicTimestamp::from_nanos(30);
        let mut clock = ReadFailingClock::new(observed, 2);

        let failure = solver
            .solve(request, &mut query, &mut clock)
            .expect_err("second read failure");
        let status = in_progress(&failure);

        assert_eq!(clock.calls, 2);
        assert_eq!(query.calls, 0);
        assert_eq!(status.started_at(), observed);
        assert_eq!(status.observed_at(), observed);
        assert_eq!(status.completed_iterations(), 0);
        assert_eq!(status.rollout_evaluations(), 0);
        assert_eq!(status.pre_final_collision_queries(), 0);
        assert_eq!(status.final_validation_queries(), 0);
        assert!(matches!(
            failure.kind(),
            MpcFailureKind::Clock(HostMonotonicClockFailure::Read(source))
                if *source == INJECTED_CLOCK_READ_ERROR
        ));
    }

    #[test]
    fn query_and_following_clock_read_failure_retain_both_sources() {
        let fixture = runtime_fixture(1, 1);
        let reference = reference(&fixture, 0.0);
        let request = request(&fixture, &reference);
        let mut solver = MpcSolver::new(fixture.model, fixture.config).expect("solver");
        let mut query = ScriptedQuery {
            snapshot: fixture.collision,
            calls: 0,
            error_on: Some(1),
            block_on: None,
        };
        let observed = HostMonotonicTimestamp::from_nanos(30);
        let mut clock = ReadFailingClock::new(observed, 3);

        let failure = solver
            .solve(request, &mut query, &mut clock)
            .expect_err("combined query and read failure");
        let status = in_progress(&failure);

        assert_eq!(clock.calls, 3);
        assert_eq!(query.calls, 1);
        assert_eq!(status.observed_at(), observed);
        assert_eq!(status.rollout_evaluations(), 0);
        assert_eq!(status.pre_final_collision_queries(), 1);
        assert_eq!(status.final_validation_queries(), 0);
        assert!(matches!(
            failure.kind(),
            MpcFailureKind::CollisionObservation {
                horizon_step: None,
                integration_substep: None,
                final_revalidation: false,
                source: CollisionObservationFailure::QueryAndClock {
                    query: QueryError::Injected,
                    clock: HostMonotonicClockFailure::Read(source),
                },
            } if *source == INJECTED_CLOCK_READ_ERROR
        ));
    }

    #[test]
    fn admitted_rollout_is_charged_before_body_clock_read_failure() {
        let fixture = runtime_fixture(1, 1);
        let reference = reference(&fixture, 0.0);
        let request = request(&fixture, &reference);
        let mut solver = MpcSolver::new(fixture.model, fixture.config).expect("solver");
        let mut query = ScriptedQuery::clear(fixture.collision);
        let observed = HostMonotonicTimestamp::from_nanos(30);
        let mut clock = ReadFailingClock::new(observed, 5);

        let failure = solver
            .solve(request, &mut query, &mut clock)
            .expect_err("rollout body read failure");
        let status = in_progress(&failure);

        assert_eq!(clock.calls, 5);
        assert_eq!(query.calls, 1);
        assert_eq!(status.completed_iterations(), 0);
        assert_eq!(status.rollout_evaluations(), 1);
        assert!(status.rollout_evaluations() <= failure.config().max_rollout_evaluations());
        assert_eq!(status.pre_final_collision_queries(), 1);
        assert_eq!(status.final_validation_queries(), 0);
        assert!(matches!(
            failure.kind(),
            MpcFailureKind::Clock(HostMonotonicClockFailure::Read(source))
                if *source == INJECTED_CLOCK_READ_ERROR
        ));
    }

    #[test]
    fn final_clock_read_failure_retains_bounded_attempt_counters() {
        let fixture = runtime_fixture(1, 1);
        let reference = reference(&fixture, 0.0);
        let request = request(&fixture, &reference);
        let observed = HostMonotonicTimestamp::from_nanos(30);

        let mut successful_solver =
            MpcSolver::new(fixture.model, fixture.config).expect("successful solver");
        let mut successful_query = ScriptedQuery::clear(fixture.collision);
        let mut successful_clock = ReadFailingClock::new(observed, usize::MAX);
        let expected_status = successful_solver
            .solve(request, &mut successful_query, &mut successful_clock)
            .expect("deterministic successful solve")
            .status();
        let final_read_call = successful_clock.calls;

        let mut failing_solver =
            MpcSolver::new(fixture.model, fixture.config).expect("failing solver");
        let mut failing_query = ScriptedQuery::clear(fixture.collision);
        let mut failing_clock = ReadFailingClock::new(observed, final_read_call);
        let failure = failing_solver
            .solve(request, &mut failing_query, &mut failing_clock)
            .expect_err("last clock read must remain fallible");

        assert_eq!(failing_clock.calls, final_read_call);
        assert_eq!(
            failure.progress(),
            MpcSolveProgressV1::InProgress(expected_status)
        );
        let status = in_progress(&failure);
        assert!(status.rollout_evaluations() > 0);
        assert!(status.rollout_evaluations() <= failure.config().max_rollout_evaluations());
        assert!(status.pre_final_collision_queries() > 0);
        assert_eq!(status.final_validation_queries(), 1);
        assert!(matches!(
            failure.kind(),
            MpcFailureKind::Clock(HostMonotonicClockFailure::Read(source))
                if *source == INJECTED_CLOCK_READ_ERROR
        ));
    }

    #[test]
    fn convex_first_order_response_avoids_opposite_sign_overflow() {
        let response = WheelResponse {
            initial_velocity_mps: f64::MAX,
            target_velocity_mps: -f64::MAX,
            time_constant_s: 1.0,
        };
        let velocity = response.velocity_at(0.25).expect("finite convex velocity");
        let distance = response
            .distance_at(0.25)
            .expect("finite weighted distance");
        assert!(velocity.is_finite());
        assert!(distance.is_finite());
        assert!(velocity > 0.0);
        assert!(distance > 0.0);
    }

    #[test]
    fn first_order_response_stays_finite_for_extreme_finite_endpoints() {
        for (initial_velocity_mps, target_velocity_mps) in [
            (f64::MAX, f64::MAX),
            (f64::MAX, -f64::MAX),
            (-f64::MAX, f64::MAX),
            (-f64::MAX, -f64::MAX),
        ] {
            let response = WheelResponse {
                initial_velocity_mps,
                target_velocity_mps,
                time_constant_s: 1.0,
            };
            for time_s in [0.0, f64::MIN_POSITIVE, 1.0e-12, 0.25, 1.0, 16.0] {
                let velocity = response
                    .velocity_at(time_s)
                    .expect("finite convex response");
                assert!(velocity.is_finite());
            }
            for time_s in [0.0, f64::MIN_POSITIVE, 1.0e-12, 0.25, 1.0] {
                let distance = response
                    .distance_at(time_s)
                    .expect("finite bounded distance");
                assert!(distance.is_finite());
            }
        }
    }

    #[test]
    fn stable_weighted_sum_handles_subnormal_cancellation() {
        let tiny = f64::from_bits(1);
        assert_eq!(stable_weighted_sum(0.5, tiny, 0.5, -tiny), Some(0.0));
        assert!(outward_abs_difference(tiny, -tiny).expect("bound") >= 2.0 * tiny);
    }

    #[test]
    fn unsupported_subnormal_tau_is_rejected_at_construction() {
        let model = PlantModelV1::parse(plant_dto(0.1, f64::from_bits(1), 0.5))
            .expect("structurally valid model");
        let config = MpcConfigV1::parse(config_dto(0.1, 1, 4)).expect("config");
        assert!(matches!(
            MpcSolver::new(model, config),
            Err(MpcCreateError::UnsupportedTimeConstant {
                wheel: WheelSide::Left,
                ..
            })
        ));
    }

    #[test]
    fn outward_yaw_bound_contains_dense_unequal_tau_samples() {
        let mut seed = 0x4d595df4d0f33173_u64;
        for _case in 0..256 {
            let mut next = || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                (seed >> 11) as f64 / ((1_u64 << 53) as f64)
            };
            let left = WheelResponse {
                initial_velocity_mps: -2.0 + 4.0 * next(),
                target_velocity_mps: -2.0 + 4.0 * next(),
                time_constant_s: 0.05 + 0.95 * next(),
            };
            let right = WheelResponse {
                initial_velocity_mps: -2.0 + 4.0 * next(),
                target_velocity_mps: -2.0 + 4.0 * next(),
                time_constant_s: 0.05 + 0.95 * next(),
            };
            let start = 0.2 * next();
            let duration = 0.001 + 0.04 * next();
            let end = start + duration;
            let l0 = left.velocity_at(start).expect("left start");
            let l1 = left.velocity_at(end).expect("left end");
            let r0 = right.velocity_at(start).expect("right start");
            let r1 = right.velocity_at(end).expect("right end");
            let bound = conservative_yaw_rate_bound(l0, l1, r0, r1, 0.3).expect("yaw bound");
            for sample in 0..=128 {
                let time = start + duration * sample as f64 / 128.0;
                let actual = (right.velocity_at(time).expect("right")
                    - left.velocity_at(time).expect("left"))
                .abs()
                    / 0.3;
                assert!(actual <= bound, "actual {actual:e} > bound {bound:e}");
            }
        }
    }

    fn point_to_segment_distance(x: f64, y: f64, segment: ConservativeCapsuleSegmentV1) -> f64 {
        let ax = segment.centerline_start.x_m();
        let ay = segment.centerline_start.y_m();
        let bx = segment.centerline_end.x_m();
        let by = segment.centerline_end.y_m();
        let dx = bx - ax;
        let dy = by - ay;
        let denominator = dx.mul_add(dx, dy * dy);
        let fraction = if denominator == 0.0 {
            0.0
        } else {
            ((x - ax).mul_add(dx, (y - ay) * dy) / denominator).clamp(0.0, 1.0)
        };
        (x - (ax + fraction * dx)).hypot(y - (ay + fraction * dy))
    }

    #[test]
    fn unequal_tau_old_seven_centimetre_case_stays_inside_substep_tubes() {
        let mut model_dto = plant_dto(0.5, 0.08, 0.4);
        model_dto.wheelbase_m = 0.2;
        model_dto.left.velocity_gain_mps_per_pwm_percent = 0.02;
        model_dto.right.velocity_gain_mps_per_pwm_percent = 0.02;
        model_dto.validity.left_velocity_min_mps = -2.1;
        model_dto.validity.left_velocity_max_mps = 2.1;
        model_dto.validity.right_velocity_min_mps = -2.1;
        model_dto.validity.right_velocity_max_mps = 2.1;
        model_dto.validity.max_abs_yaw_rate_rad_s = 20.0;
        let model = PlantModelV1::parse(model_dto).expect("aggressive unequal-tau model");
        let mut config_dto = config_dto(0.5, 1, 64);
        config_dto.left_pwm_min_percent = -100;
        config_dto.left_pwm_max_percent = 100;
        config_dto.right_pwm_min_percent = -100;
        config_dto.right_pwm_max_percent = 100;
        let config = MpcConfigV1::parse(config_dto).expect("substepped config");
        let mut solver = MpcSolver::new(model, config).expect("supported numerical domain");
        let initial = SimState {
            x_m: 0.0,
            y_m: 0.0,
            yaw_rad: 0.0,
            left_velocity_mps: 1.5,
            right_velocity_mps: 1.5,
            integration_tube_radius_m: 0.0,
        };
        solver
            .integrate_step(initial, pwm(-100, 100), 0)
            .expect("integrated step");

        // Independent RK4 oracle integrates wheel ODEs and SE(2) kinematics;
        // it does not call WheelResponse or the production midpoint update.
        let target_left = -2.0;
        let target_right = 2.0;
        let mut oracle = [1.5_f64, 1.5, 0.0, 0.0, 0.0];
        let oracle_steps = 64 * 256;
        let oracle_dt = 0.5 / oracle_steps as f64;
        let derivative = |state: [f64; 5]| {
            let center = 0.5 * (state[0] + state[1]);
            [
                (target_left - state[0]) / 0.08,
                (target_right - state[1]) / 0.4,
                center * state[4].cos(),
                center * state[4].sin(),
                (state[1] - state[0]) / 0.2,
            ]
        };
        let left_response = WheelResponse {
            initial_velocity_mps: 1.5,
            target_velocity_mps: target_left,
            time_constant_s: 0.08,
        };
        let right_response = WheelResponse {
            initial_velocity_mps: 1.5,
            target_velocity_mps: target_right,
            time_constant_s: 0.4,
        };
        let left_distance = left_response.distance_at(0.5).expect("left distance");
        let right_distance = right_response.distance_at(0.5).expect("right distance");
        let old_center_distance = 0.5 * (left_distance + right_distance);
        let old_yaw_delta = (right_distance - left_distance) / 0.2;
        let mut maximum_old_surrogate_error = 0.0_f64;
        for oracle_index in 0..oracle_steps {
            let k1 = derivative(oracle);
            let midpoint1 = std::array::from_fn(|axis| oracle[axis] + 0.5 * oracle_dt * k1[axis]);
            let k2 = derivative(midpoint1);
            let midpoint2 = std::array::from_fn(|axis| oracle[axis] + 0.5 * oracle_dt * k2[axis]);
            let k3 = derivative(midpoint2);
            let endpoint = std::array::from_fn(|axis| oracle[axis] + oracle_dt * k3[axis]);
            let k4 = derivative(endpoint);
            oracle = std::array::from_fn(|axis| {
                oracle[axis]
                    + oracle_dt * (k1[axis] + 2.0 * k2[axis] + 2.0 * k3[axis] + k4[axis]) / 6.0
            });
            let elapsed = (oracle_index + 1) as f64 * oracle_dt;
            let substep = ((elapsed / solver.substep_s).ceil() as usize)
                .saturating_sub(1)
                .min(63);
            let tube = solver.trial_capsules[substep];
            let distance = point_to_segment_distance(oracle[2], oracle[3], tube);
            assert!(
                distance <= tube.extra_radius_m + 1.0e-10,
                "oracle distance {distance:e} exceeds tube {:e} at t={elapsed:e}",
                tube.extra_radius_m
            );

            let fraction = elapsed / 0.5;
            let distance_along_old_arc = old_center_distance * fraction;
            let yaw = old_yaw_delta * fraction;
            let (old_x, old_y) = if yaw.abs() < 1.0e-12 {
                (distance_along_old_arc, 0.0)
            } else {
                (
                    distance_along_old_arc * yaw.sin() / yaw,
                    distance_along_old_arc * (1.0 - yaw.cos()) / yaw,
                )
            };
            maximum_old_surrogate_error =
                maximum_old_surrogate_error.max((oracle[2] - old_x).hypot(oracle[3] - old_y));
        }
        assert!(
            maximum_old_surrogate_error > 0.0705,
            "fixture no longer exposes the old surrogate: {maximum_old_surrogate_error}"
        );
    }

    #[test]
    fn exact_runtime_provenance_rejects_path_and_deadline_substitution() {
        let fixture = runtime_fixture(1, 1);
        let session = fixture.epoch.device_session_id();
        let (other_map, other_path) = global_path_fixture();
        assert!(matches!(
            NavigationEpochV1::from_runtime(
                session,
                fixture.epoch.odom_segment_id(),
                other_map,
                &fixture.path
            ),
            Err(NavigationEpochError::GlobalPathMapMismatch { .. })
        ));
        let dto = MpcReferenceV1Dto {
            schema_version: MPC_REFERENCE_V1,
            builder_revision: 1,
            created_at_host_ns: 12,
            step_period_s: fixture.config.dt_s,
            points: vec![OdomReferencePointV1Dto {
                x_m: 0.0,
                y_m: 0.0,
                yaw_rad: 0.0,
                forward_velocity_mps: 0.0,
                yaw_rate_rad_s: 0.0,
            }],
        };
        assert!(matches!(
            MpcReferenceV1::parse(dto, fixture.config, fixture.epoch, &other_path),
            Err(MpcReferenceParseError::GlobalPathMismatch { .. })
        ));
        let reference = reference(&fixture, 0.0);
        assert!(matches!(
            MpcRequestV1::parse(
                MpcRequestV1Dto {
                    schema_version: 1,
                    request_id: 1,
                    submitted_at_host_ns: 20,
                    deadline_host_ns: fixture.collision.valid_through().as_nanos() + 1
                },
                state(fixture.epoch),
                &reference,
                ShadowPwmPair::STOP,
                fixture.collision,
            ),
            Err(MpcRequestParseError::DeadlineExceedsCollisionValidity { .. })
        ));
    }

    #[test]
    fn collision_provenance_rejects_session_and_odom_segment_substitution() {
        let fixture = runtime_fixture(1, 1);
        let navigation_session = fixture.epoch.device_session_id();
        let navigation_segment = fixture.epoch.odom_segment_id();

        let other_session = DeviceSessionId::try_new(2).expect("other session");
        let session_costmap =
            local_costmap_fixture(other_session, navigation_segment, 10, 1_000_000);
        let session_view = session_costmap
            .view_at(HostMonotonicTimestamp::from_nanos(20))
            .expect("current session-mismatch view");
        assert!(matches!(
            CollisionSnapshotProvenanceV1::from_runtime(fixture.epoch, &session_view),
            Err(CollisionProvenanceError::DeviceSessionMismatch {
                navigation,
                local_costmap,
            }) if navigation == navigation_session && local_costmap == other_session
        ));

        let other_segment = OdomSegmentId::try_new(2).expect("other odom segment");
        let segment_costmap =
            local_costmap_fixture(navigation_session, other_segment, 10, 1_000_000);
        let segment_view = segment_costmap
            .view_at(HostMonotonicTimestamp::from_nanos(20))
            .expect("current segment-mismatch view");
        assert!(matches!(
            CollisionSnapshotProvenanceV1::from_runtime(fixture.epoch, &segment_view),
            Err(CollisionProvenanceError::OdomSegmentMismatch {
                navigation,
                local_costmap,
            }) if navigation == navigation_segment && local_costmap == other_segment
        ));

        assert_eq!(fixture.collision.max_observation_age_ns(), 1_000_000);
        assert_eq!(
            fixture.collision.valid_through().as_nanos(),
            fixture.collision.observed_at().as_nanos() + fixture.collision.max_observation_age_ns()
        );

        let expired_view = fixture
            .costmap
            .view_at(HostMonotonicTimestamp::from_nanos(1_000_011))
            .expect("monotonic expired view");
        assert!(matches!(
            CollisionSnapshotProvenanceV1::from_runtime(fixture.epoch, &expired_view),
            Err(CollisionProvenanceError::ViewNotCurrent(
                LocalCostmapFreshness::Expired { .. }
            ))
        ));
    }

    #[test]
    fn solution_retains_exact_provenance_and_distinct_final_validation() {
        let fixture = runtime_fixture(1, 1);
        let reference = reference(&fixture, 0.0);
        let request = request(&fixture, &reference);
        let mut solver = MpcSolver::new(fixture.model, fixture.config).expect("solver");
        let mut query = ScriptedQuery::clear(fixture.collision);
        let mut clock = ConstantClock(HostMonotonicTimestamp::from_nanos(30));
        let solution = solver
            .solve(request, &mut query, &mut clock)
            .expect("solution");
        assert_eq!(solution.model(), fixture.model);
        assert_eq!(solution.config(), fixture.config);
        assert_eq!(solution.request(), request);
        assert_eq!(solution.request().reference().source_path(), &fixture.path);
        assert_eq!(
            solution.final_validation().collision_snapshot(),
            fixture.collision
        );
        assert_eq!(solution.final_validation().segment_count(), 1);
        assert_eq!(solution.status().final_validation_queries(), 1);
        assert_eq!(
            query.calls,
            solution.status().pre_final_collision_queries() as usize + 1
        );
    }

    #[test]
    fn collision_snapshot_mismatch_performs_no_query() {
        let fixture = runtime_fixture(1, 1);
        let other = runtime_fixture(1, 1);
        let reference = reference(&fixture, 0.0);
        let request = request(&fixture, &reference);
        let mut solver = MpcSolver::new(fixture.model, fixture.config).expect("solver");
        let mut query = ScriptedQuery::clear(other.collision);
        let mut clock = ConstantClock(HostMonotonicTimestamp::from_nanos(30));
        let failure = solver
            .solve(request, &mut query, &mut clock)
            .expect_err("mismatch");
        assert!(matches!(
            failure.kind(),
            MpcFailureKind::CollisionSnapshotMismatch { .. }
        ));
        assert_eq!(query.calls, 0);
    }

    #[test]
    fn query_error_and_post_call_deadline_are_preserved_together() {
        let fixture = runtime_fixture(1, 1);
        let reference = reference(&fixture, 0.0);
        let request = request(&fixture, &reference);
        let mut solver = MpcSolver::new(fixture.model, fixture.config).expect("solver");
        let mut query = ScriptedQuery {
            snapshot: fixture.collision,
            calls: 0,
            error_on: Some(1),
            block_on: None,
        };
        let mut clock = ScriptedClock::new(&[30, 30, 1000]);
        let failure = solver
            .solve(request, &mut query, &mut clock)
            .expect_err("combined error");
        assert!(matches!(
            failure.kind(),
            MpcFailureKind::CollisionObservation {
                final_revalidation: false,
                source: CollisionObservationFailure::QueryAndClock {
                    query: QueryError::Injected,
                    clock: HostMonotonicClockFailure::Fault(ClockFault::DeadlineReached { .. }),
                },
                ..
            }
        ));
    }

    #[test]
    fn final_trajectory_is_requeried_after_selected_rollout() {
        let fixture = runtime_fixture(1, 1);
        let reference = reference(&fixture, 0.0);
        let request = request(&fixture, &reference);
        let mut solver = MpcSolver::new(fixture.model, fixture.config).expect("solver");
        // start + nine lattice rollouts + selected rollout = 11 optimizer queries.
        let mut query = ScriptedQuery {
            snapshot: fixture.collision,
            calls: 0,
            error_on: None,
            block_on: Some(12),
        };
        let mut clock = ConstantClock(HostMonotonicTimestamp::from_nanos(30));
        let failure = solver
            .solve(request, &mut query, &mut clock)
            .expect_err("final revalidation blocks");
        assert!(matches!(
            failure.kind(),
            MpcFailureKind::FinalTrajectoryBlocked {
                horizon_step: 0,
                integration_substep: 0
            }
        ));
        assert_eq!(in_progress(&failure).pre_final_collision_queries(), 11);
        assert_eq!(in_progress(&failure).final_validation_queries(), 1);
    }

    #[test]
    fn disabled_weight_skips_overflowing_residual_arithmetic() {
        let mut fixture = runtime_fixture(1, 1);
        let mut dto = config_dto(0.1, 1, 1);
        dto.position_cost_per_m2 = 0.0;
        dto.heading_cost_per_rad2 = 1.0;
        dto.forward_velocity_cost_s2_per_m2 = 0.0;
        dto.yaw_rate_cost_s2_per_rad2 = 0.0;
        dto.pwm_cost_per_percent2 = 0.0;
        dto.slew_cost_per_percent2 = 0.0;
        fixture.config = MpcConfigV1::parse(dto).expect("disabled velocity config");
        {
            let reference = reference(&fixture, f64::MAX);
            let request = request(&fixture, &reference);
            let mut solver = MpcSolver::new(fixture.model, fixture.config).expect("solver");
            let mut query = ScriptedQuery::clear(fixture.collision);
            let mut clock = ConstantClock(HostMonotonicTimestamp::from_nanos(30));
            solver
                .solve(request, &mut query, &mut clock)
                .expect("disabled term is never evaluated");
        }
        dto.forward_velocity_cost_s2_per_m2 = 1.0;
        fixture.config = MpcConfigV1::parse(dto).expect("enabled velocity config");
        let reference = reference(&fixture, f64::MAX);
        let request = request(&fixture, &reference);
        let mut solver = MpcSolver::new(fixture.model, fixture.config).expect("solver");
        let mut query = ScriptedQuery::clear(fixture.collision);
        let mut clock = ConstantClock(HostMonotonicTimestamp::from_nanos(30));
        let failure = solver
            .solve(request, &mut query, &mut clock)
            .expect_err("enabled term overflows");
        assert!(matches!(
            failure.kind(),
            MpcFailureKind::Numerical {
                stage: NumericalStageV1::CostEvaluation,
                ..
            }
        ));
    }

    #[test]
    fn integer_lattice_uses_stable_lexicographic_tie_break() {
        let mut fixture = runtime_fixture(1, 1);
        fixture.model = PlantModelV1::parse(plant_dto(0.1, 0.4, 0.4)).expect("symmetric model");
        let mut dto = config_dto(0.1, 1, 1);
        dto.position_cost_per_m2 = 0.0;
        dto.heading_cost_per_rad2 = 1.0;
        dto.pwm_cost_per_percent2 = 0.0;
        dto.slew_cost_per_percent2 = 0.0;
        fixture.config = MpcConfigV1::parse(dto).expect("tie config");
        let reference = reference(&fixture, 0.0);
        let request = request(&fixture, &reference);
        let mut first = MpcSolver::new(fixture.model, fixture.config).expect("first solver");
        let mut first_query = ScriptedQuery::clear(fixture.collision);
        let mut first_clock = ConstantClock(HostMonotonicTimestamp::from_nanos(30));
        let first_pwm = first
            .solve(request, &mut first_query, &mut first_clock)
            .expect("first solution")
            .requested_pwm();
        let mut second = MpcSolver::new(fixture.model, fixture.config).expect("second solver");
        let mut second_query = ScriptedQuery::clear(fixture.collision);
        let mut second_clock = ConstantClock(HostMonotonicTimestamp::from_nanos(30));
        let second_pwm = second
            .solve(request, &mut second_query, &mut second_clock)
            .expect("second solution")
            .requested_pwm();
        assert_eq!(first_pwm, pwm(-10, -10));
        assert_eq!(second_pwm, first_pwm);
    }

    #[test]
    fn repeated_solves_reuse_preallocated_buffers_without_growth() {
        let fixture = runtime_fixture(2, 2);
        let reference = reference(&fixture, 0.0);
        let request = request(&fixture, &reference);
        let mut solver = MpcSolver::new(fixture.model, fixture.config).expect("solver");
        // This detects replacement or growth of every solver-owned heap
        // buffer. It is not a process-wide allocator measurement and makes no
        // claim about caller-owned clock or collision-query implementations.
        let pointers = (
            solver.commands.as_ptr(),
            solver.trial_states.as_ptr(),
            solver.trial_capsules.as_ptr(),
            solver.predicted_trajectory.as_ptr(),
        );
        let capacities = (
            solver.commands.capacity(),
            solver.trial_states.capacity(),
            solver.trial_capsules.capacity(),
            solver.predicted_trajectory.capacity(),
        );
        for _ in 0..2 {
            let mut query = ScriptedQuery::clear(fixture.collision);
            let mut clock = ConstantClock(HostMonotonicTimestamp::from_nanos(30));
            let solution = solver
                .solve(request, &mut query, &mut clock)
                .expect("solution");
            assert_eq!(solution.command_sequence().len(), 2);
        }
        assert_eq!(
            pointers,
            (
                solver.commands.as_ptr(),
                solver.trial_states.as_ptr(),
                solver.trial_capsules.as_ptr(),
                solver.predicted_trajectory.as_ptr()
            )
        );
        assert_eq!(
            capacities,
            (
                solver.commands.capacity(),
                solver.trial_states.capacity(),
                solver.trial_capsules.capacity(),
                solver.predicted_trajectory.capacity()
            )
        );
    }

    #[test]
    fn evaluation_limit_is_exact_and_fail_closed() {
        let mut fixture = runtime_fixture(1, 1);
        let mut dto = config_dto(0.1, 1, 1);
        dto.max_rollout_evaluations = 1;
        fixture.config = MpcConfigV1::parse(dto).expect("bounded config");
        let reference = reference(&fixture, 0.0);
        let request = request(&fixture, &reference);
        let mut solver = MpcSolver::new(fixture.model, fixture.config).expect("solver");
        let mut query = ScriptedQuery::clear(fixture.collision);
        let mut clock = ConstantClock(HostMonotonicTimestamp::from_nanos(30));
        let failure = solver
            .solve(request, &mut query, &mut clock)
            .expect_err("evaluation bound");
        assert!(matches!(
            failure.kind(),
            MpcFailureKind::EvaluationLimit { configured: 1 }
        ));
        assert_eq!(in_progress(&failure).rollout_evaluations(), 1);
    }

    #[test]
    fn every_selected_command_respects_canonical_adjacent_slew() {
        let mut fixture = runtime_fixture(4, 1);
        let mut dto = config_dto(0.1, 4, 1);
        dto.left_max_slew_percent_per_step = 5;
        dto.right_max_slew_percent_per_step = 7;
        fixture.config = MpcConfigV1::parse(dto).expect("slew config");
        let reference = reference(&fixture, 0.0);
        let request = request(&fixture, &reference);
        let mut solver = MpcSolver::new(fixture.model, fixture.config).expect("solver");
        let mut query = ScriptedQuery::clear(fixture.collision);
        let mut clock = ConstantClock(HostMonotonicTimestamp::from_nanos(30));
        let solution = solver
            .solve(request, &mut query, &mut clock)
            .expect("solution");
        let mut previous = ShadowPwmPair::STOP;
        for command in solution.command_sequence() {
            assert!(percent_delta(command.left(), previous.left()) <= 5);
            assert!(percent_delta(command.right(), previous.right()) <= 7);
            previous = *command;
        }
    }

    #[test]
    fn canonical_costmap_adapter_blocks_exact_tangent_to_nonfree_cell() {
        let fixture = runtime_fixture(1, 1);
        let view = fixture
            .costmap
            .view_at(HostMonotonicTimestamp::from_nanos(20))
            .expect("view clock");
        let mut boundary = None;
        for row in 0..view.height() {
            for column in 0..view.width() {
                if view.cell(column, row) != Some(LocalCostmapCell::Free) {
                    continue;
                }
                for (dc, dr) in [(1_i32, 0_i32), (-1, 0), (0, 1), (0, -1)] {
                    let Some(neighbor_column) = column.checked_add_signed(dc) else {
                        continue;
                    };
                    let Some(neighbor_row) = row.checked_add_signed(dr) else {
                        continue;
                    };
                    if neighbor_column < view.width()
                        && neighbor_row < view.height()
                        && view.cell(neighbor_column, neighbor_row) != Some(LocalCostmapCell::Free)
                    {
                        boundary = Some((column, row));
                        break;
                    }
                }
                if boundary.is_some() {
                    break;
                }
            }
            if boundary.is_some() {
                break;
            }
        }
        let (column, row) = boundary.expect("fixture contains a free/nonfree boundary");
        let center = view
            .cell_center_odom(column, row)
            .expect("centre transform")
            .expect("cell centre");
        let mut adapter =
            LocalCostmapCapsuleQueryV1::try_new(view, fixture.collision).expect("adapter");
        let point = ConservativeCapsuleSegmentV1 {
            purpose: CapsulePurposeV1::PredictedMotion,
            horizon_step: 0,
            integration_substep: 0,
            time_start_s: 0.0,
            time_end_s: 0.0,
            centerline_start: center,
            centerline_end: center,
            extra_radius_m: 0.0,
        };
        assert!(adapter.is_capsule_traversable(point).expect("point query"));
        let tangent = ConservativeCapsuleSegmentV1 {
            extra_radius_m: 0.5 * fixture.costmap.config().geometry().resolution_m(),
            ..point
        };
        assert!(
            !adapter
                .is_capsule_traversable(tangent)
                .expect("tangent query")
        );
    }

    #[test]
    fn canonical_costmap_adapter_blocks_unknown_and_out_of_bounds_space() {
        let fixture = runtime_fixture(1, 1);
        let view = fixture
            .costmap
            .view_at(HostMonotonicTimestamp::from_nanos(20))
            .expect("current view");
        let mut unknown = None;
        for row in 0..view.height() {
            for column in 0..view.width() {
                if view.cell(column, row) == Some(LocalCostmapCell::Unknown) {
                    unknown = view
                        .cell_center_odom(column, row)
                        .expect("finite cell transform");
                    break;
                }
            }
            if unknown.is_some() {
                break;
            }
        }
        let unknown = unknown.expect("fixture retains unknown cells");
        let mut adapter =
            LocalCostmapCapsuleQueryV1::try_new(view, fixture.collision).expect("adapter");
        let point_capsule = |point| ConservativeCapsuleSegmentV1 {
            purpose: CapsulePurposeV1::PredictedMotion,
            horizon_step: 0,
            integration_substep: 0,
            time_start_s: 0.0,
            time_end_s: 0.0,
            centerline_start: point,
            centerline_end: point,
            extra_radius_m: 0.0,
        };
        assert!(
            !adapter
                .is_capsule_traversable(point_capsule(unknown))
                .expect("unknown query")
        );
        let outside =
            PlanarPoint::<OdomFrame>::try_new(1_000.0, 1_000.0).expect("finite outside point");
        assert!(
            !adapter
                .is_capsule_traversable(point_capsule(outside))
                .expect("outside query")
        );
    }

    #[test]
    fn canonical_costmap_adapter_rejects_observation_age_policy_substitution() {
        let fixture = runtime_fixture(1, 1);
        let replacement = local_costmap_fixture(
            fixture.epoch.device_session_id(),
            fixture.epoch.odom_segment_id(),
            10,
            100,
        );
        let replacement_view = replacement
            .view_at(HostMonotonicTimestamp::from_nanos(20))
            .expect("current replacement view");
        let error = match LocalCostmapCapsuleQueryV1::try_new(replacement_view, fixture.collision) {
            Err(error) => error,
            Ok(_) => panic!("different freshness policy must not inherit a longer deadline"),
        };
        assert_eq!(
            error,
            LocalCostmapCapsuleAdapterError::ObservationAgePolicyMismatch {
                expected_ns: 1_000_000,
                actual_ns: 100,
            }
        );
    }

    #[test]
    fn no_transport_channel_identity_exists_in_live_mpc_contract() {
        let model = PlantModelV1::parse(plant_dto(0.1, 0.4, 0.5)).expect("model");
        assert_eq!(model.left.gain_mps_per_pwm_percent, 0.01);
        assert_eq!(pwm(-100, 100).left().get(), -100);
    }

    #[test]
    fn query_trait_accepts_infallible_pure_fixture() {
        struct Pure(CollisionSnapshotProvenanceV1);
        impl CollisionQuery for Pure {
            type Error = Infallible;
            fn snapshot_provenance(&self) -> CollisionSnapshotProvenanceV1 {
                self.0
            }
            fn is_capsule_traversable(
                &mut self,
                _: ConservativeCapsuleSegmentV1,
            ) -> Result<bool, Self::Error> {
                Ok(true)
            }
        }
        let fixture = runtime_fixture(1, 1);
        let mut pure = Pure(fixture.collision);
        assert!(
            pure.is_capsule_traversable(ConservativeCapsuleSegmentV1::stationary_start(
                state(fixture.epoch).pose()
            ))
            .expect("infallible")
        );
    }
}
