//! Strict JSON configuration boundary for live, transport-free navigation.
//!
//! JSON is admitted through [`ShadowNavigationConfigV1::parse_json`] only. The
//! input is size bounded, every object rejects unknown fields, and every field
//! is required. Weak scalar values are then consumed exactly once by the
//! existing domain constructors. In particular, the runtime depth camera is a
//! required argument rather than JSON: this module never guesses or overrides
//! its intrinsics, image dimensions, or depth-to-tracking extrinsic.
//!
//! The timing contract deliberately keeps sensor eligibility and gyro
//! integration gaps no longer than one command lease. A lease must cover one
//! control period and may cover at most two. Thus stale local-depth or
//! odometry state cannot remain eligible beyond two nominal ticks. The visual
//! interval is different: it validates the continuity between two individually
//! fresh visual observations, while host-age and prediction-age bounds decide
//! control eligibility between them. It may therefore exceed a lease without
//! extending stale control. These are host shadow-mode admission invariants;
//! they do not verify plant-identification evidence or authorize actuation.

use std::fmt;
use std::num::NonZeroU64;
use std::time::Duration;

use serde::Deserialize;

use crate::dense::occupancy::{
    DepthCameraModel, DepthRangeError, DepthRangeMeters, HeightRangeError, HeightRangeMeters,
    OccupancyGridGeometry, OccupancyGridGeometryError, WorldToOccupancy, WorldToOccupancyError,
};
use crate::{Pose, PoseError, SensorAccuracy};

use super::global_planner::{GlobalPlanError, GlobalPlannerConfig, UnknownSpacePolicy};
use super::ingress::{
    MAX_NAVIGATION_INGRESS_RECORDS, NavigationIngressCapacity, NavigationIngressCapacityError,
};
use super::local_costmap::{LocalCostmapConfig, LocalCostmapConfigError, TrackingCameraToBase};
use super::mpc::{
    FitResidualsV1Dto, MpcConfigParseError, MpcConfigV1, MpcConfigV1Dto, MpcCreateError, MpcSolver,
    PlantEvidenceV1Dto, PlantModelParseError, PlantModelV1, PlantModelV1Dto,
    PlantValidityEnvelopeV1Dto, WheelPlantV1Dto,
};
use super::odometry::{
    PlanarOdometryConfig, PlanarOdometryConfigDto, PlanarOdometryConfigError, RawImuCalibrationDto,
};
use super::reference::{
    PathReferenceConfigParseError, PathReferenceConfigV1, PathReferenceConfigV1Dto,
};
use super::safety::{SolverBudgetError, SolverBudgetNs};
use super::shadow_command::{
    ShadowCommandConfig, ShadowCommandConfigDto, ShadowCommandConfigError,
};

/// The only supported top-level live shadow-navigation configuration format.
pub const SHADOW_NAVIGATION_CONFIG_V1: u32 = 1;

/// Hard bound applied before JSON parsing or allocation from input-controlled
/// string lengths. The V1 fixture is under 5 KiB; this leaves ample room for
/// bounded provenance identifiers without accepting arbitrarily large files.
pub const MAX_SHADOW_NAVIGATION_CONFIG_JSON_BYTES: usize = 256 * 1024;

/// Maximum number of nominal control periods covered by a command lease.
pub const MAX_COMMAND_LEASE_CONTROL_PERIODS: u64 = 2;

/// A positive nominal control period represented exactly in nanoseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ControlPeriodNs(NonZeroU64);

impl ControlPeriodNs {
    pub const fn get(self) -> u64 {
        self.0.get()
    }

    pub const fn as_duration(self) -> Duration {
        Duration::from_nanos(self.get())
    }
}

/// Fully parsed runtime configuration. Invalid weak states cannot be
/// represented, and the retained MPC solver proves plant/controller
/// compatibility without repeating that validation in the live adapter.
pub struct ShadowNavigationConfigV1 {
    tracking_camera_to_base: TrackingCameraToBase,
    world_to_occupancy: WorldToOccupancy,
    odometry: PlanarOdometryConfig,
    local_costmap: LocalCostmapConfig,
    global_planner: GlobalPlannerConfig,
    mpc_solver: MpcSolver,
    path_reference: PathReferenceConfigV1,
    solver_budget: SolverBudgetNs,
    control_period: ControlPeriodNs,
    shadow_command: ShadowCommandConfig,
    ingress_capacity: NavigationIngressCapacity,
}

impl ShadowNavigationConfigV1 {
    pub const FORMAT_VERSION: u32 = SHADOW_NAVIGATION_CONFIG_V1;
    pub const MAX_JSON_BYTES: usize = MAX_SHADOW_NAVIGATION_CONFIG_JSON_BYTES;
    pub const MAX_INGRESS_RECORDS: usize = MAX_NAVIGATION_INGRESS_RECORDS;

    /// Parse one complete JSON document using the exact runtime depth-camera
    /// model. Caller-asserted plant evidence is retained but not authenticated,
    /// opened, hashed, measured, or otherwise independently verified here.
    pub fn parse_json(
        json: &[u8],
        runtime_depth_camera: DepthCameraModel,
    ) -> Result<Self, ShadowNavigationConfigParseError> {
        if json.len() > MAX_SHADOW_NAVIGATION_CONFIG_JSON_BYTES {
            return Err(ShadowNavigationConfigParseError::InputTooLarge {
                actual_bytes: json.len(),
                maximum_bytes: MAX_SHADOW_NAVIGATION_CONFIG_JSON_BYTES,
            });
        }
        let dto: ShadowNavigationConfigV1Dto =
            serde_json::from_slice(json).map_err(ShadowNavigationConfigParseError::Json)?;
        if dto.schema_version != SHADOW_NAVIGATION_CONFIG_V1 {
            return Err(ShadowNavigationConfigParseError::UnsupportedSchemaVersion {
                actual: dto.schema_version,
                supported: SHADOW_NAVIGATION_CONFIG_V1,
            });
        }

        let tracking_camera_to_base = TrackingCameraToBase::new(
            Pose::try_from_rt(
                dto.coordinate_frames.tracking_camera_to_base.rotation,
                dto.coordinate_frames.tracking_camera_to_base.translation_m,
            )
            .map_err(ShadowNavigationConfigParseError::TrackingCameraToBase)?,
        );
        let world_to_occupancy = WorldToOccupancy::try_new(
            dto.coordinate_frames.world_to_occupancy.rotation,
            dto.coordinate_frames.world_to_occupancy.translation_m,
        )
        .map_err(ShadowNavigationConfigParseError::WorldToOccupancy)?;

        let odometry_freshness = OdometryFreshness {
            maximum_imu_gap_ns: dto.odometry.maximum_imu_gap_ns,
            maximum_prediction_age_ns: dto.odometry.maximum_prediction_age_ns,
            maximum_host_observation_age_ns: dto.odometry.maximum_host_observation_age_ns,
            maximum_history_bracket_gap_ns: dto.odometry.maximum_history_bracket_gap_ns,
        };
        let odometry = PlanarOdometryConfig::parse(PlanarOdometryConfigDto {
            raw_imu_calibration: dto.odometry.raw_imu_calibration.into_domain_dto(),
            tracking_camera_to_base,
            world_to_occupancy,
            max_visual_interval: Duration::from_nanos(dto.odometry.maximum_visual_interval_ns),
            max_visual_linear_speed_m_per_sec: dto.odometry.maximum_visual_linear_speed_m_per_sec,
            max_visual_yaw_rate_rad_per_sec: dto.odometry.maximum_visual_yaw_rate_rad_per_sec,
            max_calibrated_yaw_rate_rad_per_sec: dto
                .odometry
                .maximum_calibrated_yaw_rate_rad_per_sec,
            minimum_gyro_accuracy: dto.odometry.minimum_gyro_accuracy.into_domain(),
            max_vertical_increment_m: dto.odometry.maximum_vertical_increment_m,
            max_relative_roll_pitch_increment_rad: dto
                .odometry
                .maximum_relative_roll_pitch_increment_rad,
            max_absolute_map_roll_pitch_rad: dto.odometry.maximum_absolute_map_roll_pitch_rad,
            max_imu_gap: Duration::from_nanos(dto.odometry.maximum_imu_gap_ns),
            max_prediction_age: Duration::from_nanos(dto.odometry.maximum_prediction_age_ns),
            max_host_observation_age: Duration::from_nanos(
                dto.odometry.maximum_host_observation_age_ns,
            ),
            max_history_bracket_gap: Duration::from_nanos(
                dto.odometry.maximum_history_bracket_gap_ns,
            ),
            gyro_history_capacity: dto.odometry.gyro_history_capacity,
            pose_history_capacity: dto.odometry.pose_history_capacity,
        })
        .map_err(ShadowNavigationConfigParseError::Odometry)?;

        let geometry = OccupancyGridGeometry::try_new(
            dto.local_costmap.resolution_m,
            dto.local_costmap.lower_bound_m,
            dto.local_costmap.width_cells,
            dto.local_costmap.height_cells,
            dto.local_costmap.maximum_cells,
        )
        .map_err(ShadowNavigationConfigParseError::LocalCostmapGeometry)?;
        let obstacle_height_range = HeightRangeMeters::try_new(
            dto.local_costmap.obstacle_height_minimum_m,
            dto.local_costmap.obstacle_height_maximum_m,
        )
        .map_err(ShadowNavigationConfigParseError::ObstacleHeightRange)?;
        let depth_range = DepthRangeMeters::try_new(
            dto.local_costmap.depth_minimum_m,
            dto.local_costmap.depth_maximum_m,
        )
        .map_err(ShadowNavigationConfigParseError::DepthRange)?;
        let local_observation_age_ns = dto.local_costmap.maximum_observation_age_ns;
        let local_costmap = LocalCostmapConfig::try_new(
            geometry,
            runtime_depth_camera,
            tracking_camera_to_base,
            obstacle_height_range,
            depth_range,
            dto.local_costmap.sampling_block_pixels,
            dto.local_costmap.footprint_radius_m,
            dto.local_costmap.clearance_m,
            Duration::from_nanos(local_observation_age_ns),
        )
        .map_err(ShadowNavigationConfigParseError::LocalCostmap)?;

        // The global planner and local collision grid deliberately share one
        // authoritative inflated footprint instead of accepting duplicate,
        // potentially contradictory radii at the JSON boundary.
        let global_planner = GlobalPlannerConfig::try_new(
            local_costmap.inflation_radius_m(),
            dto.global_planner.unknown_space_policy.into_domain(),
        )
        .map_err(ShadowNavigationConfigParseError::GlobalPlanner)?;

        let plant_model = PlantModelV1::parse(dto.plant_model.into_domain_dto())
            .map_err(ShadowNavigationConfigParseError::PlantModel)?;
        let mpc_step_period_s = dto.mpc.step_period_s;
        let mpc_config = MpcConfigV1::parse(dto.mpc.into_domain_dto())
            .map_err(ShadowNavigationConfigParseError::MpcConfig)?;
        let path_reference = PathReferenceConfigV1::parse(dto.path_reference.into_domain_dto())
            .map_err(ShadowNavigationConfigParseError::PathReference)?;

        let control_period = ControlPeriodNs(
            NonZeroU64::new(dto.control_loop.control_period_ns)
                .ok_or(ShadowNavigationConfigParseError::ZeroControlPeriod)?,
        );
        let mpc_step_period_ns = exact_period_nanoseconds(mpc_step_period_s)?;
        if mpc_step_period_ns != control_period.get() {
            return Err(ShadowNavigationConfigParseError::ControlPeriodMismatch {
                control_period_ns: control_period.get(),
                mpc_step_period_ns,
            });
        }
        let solver_budget = SolverBudgetNs::try_new(dto.control_loop.solver_budget_ns)
            .map_err(ShadowNavigationConfigParseError::SolverBudget)?;
        if solver_budget.get() >= control_period.get() {
            return Err(
                ShadowNavigationConfigParseError::SolverBudgetNotLessThanControlPeriod {
                    solver_budget_ns: solver_budget.get(),
                    control_period_ns: control_period.get(),
                },
            );
        }

        let shadow_command = ShadowCommandConfig::parse(ShadowCommandConfigDto {
            lease_ms: dto.shadow_command.lease_ms,
            retained_records: dto.shadow_command.retained_records,
            initial_sequence: dto.shadow_command.initial_sequence,
        })
        .map_err(ShadowNavigationConfigParseError::ShadowCommand)?;
        let ingress_capacity =
            NavigationIngressCapacity::try_new(dto.ingress_journal.maximum_ingress_records)
                .map_err(ShadowNavigationConfigParseError::IngressCapacity)?;
        let lease_ns = u64::from(shadow_command.lease().get()) * 1_000_000;
        if lease_ns < control_period.get() {
            return Err(
                ShadowNavigationConfigParseError::CommandLeaseDoesNotCoverControlPeriod {
                    lease_ns,
                    control_period_ns: control_period.get(),
                },
            );
        }
        let maximum_lease_ns = control_period
            .get()
            .saturating_mul(MAX_COMMAND_LEASE_CONTROL_PERIODS);
        if lease_ns > maximum_lease_ns {
            return Err(
                ShadowNavigationConfigParseError::CommandLeaseTooLongForControlPeriod {
                    lease_ns,
                    control_period_ns: control_period.get(),
                    maximum_control_periods: MAX_COMMAND_LEASE_CONTROL_PERIODS,
                },
            );
        }
        for (parameter, age_ns) in [
            (
                FreshnessParameter::LocalCostmapObservation,
                local_observation_age_ns,
            ),
            (
                FreshnessParameter::OdomGyroIntegrationGap,
                odometry_freshness.maximum_imu_gap_ns,
            ),
            (
                FreshnessParameter::OdomPrediction,
                odometry_freshness.maximum_prediction_age_ns,
            ),
            (
                FreshnessParameter::OdomHostObservation,
                odometry_freshness.maximum_host_observation_age_ns,
            ),
            (
                FreshnessParameter::OdomHistoryBracketGap,
                odometry_freshness.maximum_history_bracket_gap_ns,
            ),
        ] {
            if age_ns > lease_ns {
                return Err(
                    ShadowNavigationConfigParseError::FreshnessExceedsCommandLease {
                        parameter,
                        freshness_ns: age_ns,
                        lease_ns,
                    },
                );
            }
        }

        let mpc_solver = MpcSolver::new(plant_model, mpc_config)
            .map_err(ShadowNavigationConfigParseError::MpcSolver)?;
        Ok(Self {
            tracking_camera_to_base,
            world_to_occupancy,
            odometry,
            local_costmap,
            global_planner,
            mpc_solver,
            path_reference,
            solver_budget,
            control_period,
            shadow_command,
            ingress_capacity,
        })
    }

    pub fn odometry(&self) -> &PlanarOdometryConfig {
        &self.odometry
    }

    pub fn tracking_camera_to_base(&self) -> TrackingCameraToBase {
        self.tracking_camera_to_base
    }

    pub fn world_to_occupancy(&self) -> WorldToOccupancy {
        self.world_to_occupancy
    }

    pub fn local_costmap(&self) -> &LocalCostmapConfig {
        &self.local_costmap
    }

    pub fn global_planner(&self) -> GlobalPlannerConfig {
        self.global_planner
    }

    pub fn mpc_solver(&self) -> &MpcSolver {
        &self.mpc_solver
    }

    pub fn path_reference(&self) -> PathReferenceConfigV1 {
        self.path_reference
    }

    pub fn solver_budget(&self) -> SolverBudgetNs {
        self.solver_budget
    }

    pub fn control_period(&self) -> ControlPeriodNs {
        self.control_period
    }

    pub fn shadow_command(&self) -> ShadowCommandConfig {
        self.shadow_command
    }

    pub fn ingress_capacity(&self) -> NavigationIngressCapacity {
        self.ingress_capacity
    }

    pub fn into_runtime_parts(self) -> ShadowNavigationRuntimePartsV1 {
        ShadowNavigationRuntimePartsV1 {
            tracking_camera_to_base: self.tracking_camera_to_base,
            world_to_occupancy: self.world_to_occupancy,
            odometry: self.odometry,
            local_costmap: self.local_costmap,
            global_planner: self.global_planner,
            mpc_solver: self.mpc_solver,
            path_reference: self.path_reference,
            solver_budget: self.solver_budget,
            control_period: self.control_period,
            shadow_command: self.shadow_command,
            ingress_capacity: self.ingress_capacity,
        }
    }
}

/// Owned parts consumed by the pure coordinator and safety supervisor.
pub struct ShadowNavigationRuntimePartsV1 {
    pub tracking_camera_to_base: TrackingCameraToBase,
    pub world_to_occupancy: WorldToOccupancy,
    pub odometry: PlanarOdometryConfig,
    pub local_costmap: LocalCostmapConfig,
    pub global_planner: GlobalPlannerConfig,
    pub mpc_solver: MpcSolver,
    pub path_reference: PathReferenceConfigV1,
    pub solver_budget: SolverBudgetNs,
    pub control_period: ControlPeriodNs,
    pub shadow_command: ShadowCommandConfig,
    pub ingress_capacity: NavigationIngressCapacity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreshnessParameter {
    LocalCostmapObservation,
    OdomGyroIntegrationGap,
    OdomPrediction,
    OdomHostObservation,
    OdomHistoryBracketGap,
}

#[derive(Debug)]
pub enum ShadowNavigationConfigParseError {
    InputTooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    Json(serde_json::Error),
    UnsupportedSchemaVersion {
        actual: u32,
        supported: u32,
    },
    TrackingCameraToBase(PoseError),
    WorldToOccupancy(WorldToOccupancyError),
    Odometry(PlanarOdometryConfigError),
    LocalCostmapGeometry(OccupancyGridGeometryError),
    ObstacleHeightRange(HeightRangeError),
    DepthRange(DepthRangeError),
    LocalCostmap(LocalCostmapConfigError),
    GlobalPlanner(GlobalPlanError),
    PlantModel(PlantModelParseError),
    MpcConfig(MpcConfigParseError),
    PathReference(PathReferenceConfigParseError),
    ZeroControlPeriod,
    MpcStepPeriodNotIntegralNanoseconds {
        step_period_s: f64,
    },
    MpcStepPeriodNanosecondsOutOfRange {
        step_period_s: f64,
    },
    ControlPeriodMismatch {
        control_period_ns: u64,
        mpc_step_period_ns: u64,
    },
    SolverBudget(SolverBudgetError),
    SolverBudgetNotLessThanControlPeriod {
        solver_budget_ns: u64,
        control_period_ns: u64,
    },
    ShadowCommand(ShadowCommandConfigError),
    IngressCapacity(NavigationIngressCapacityError),
    CommandLeaseDoesNotCoverControlPeriod {
        lease_ns: u64,
        control_period_ns: u64,
    },
    CommandLeaseTooLongForControlPeriod {
        lease_ns: u64,
        control_period_ns: u64,
        maximum_control_periods: u64,
    },
    FreshnessExceedsCommandLease {
        parameter: FreshnessParameter,
        freshness_ns: u64,
        lease_ns: u64,
    },
    MpcSolver(MpcCreateError),
}

impl fmt::Display for ShadowNavigationConfigParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputTooLarge {
                actual_bytes,
                maximum_bytes,
            } => write!(
                formatter,
                "shadow navigation JSON is {actual_bytes} bytes; maximum is {maximum_bytes}"
            ),
            Self::Json(source) => write!(formatter, "invalid shadow navigation JSON: {source}"),
            Self::UnsupportedSchemaVersion { actual, supported } => write!(
                formatter,
                "unsupported shadow navigation schema version {actual}; supported version is {supported}"
            ),
            Self::TrackingCameraToBase(source) => {
                write!(formatter, "invalid tracking-camera-to-base pose: {source}")
            }
            Self::WorldToOccupancy(source) => {
                write!(formatter, "invalid world-to-occupancy transform: {source}")
            }
            Self::Odometry(source) => write!(formatter, "{source}"),
            Self::LocalCostmapGeometry(source) => {
                write!(formatter, "invalid local-costmap geometry: {source}")
            }
            Self::ObstacleHeightRange(source) => write!(formatter, "{source}"),
            Self::DepthRange(source) => write!(formatter, "{source}"),
            Self::LocalCostmap(source) => write!(formatter, "{source}"),
            Self::GlobalPlanner(source) => write!(formatter, "{source}"),
            Self::PlantModel(source) => write!(formatter, "{source}"),
            Self::MpcConfig(source) => write!(formatter, "{source}"),
            Self::PathReference(source) => write!(formatter, "{source}"),
            Self::ZeroControlPeriod => {
                formatter.write_str("control period must be nonzero nanoseconds")
            }
            Self::MpcStepPeriodNotIntegralNanoseconds { step_period_s } => write!(
                formatter,
                "MPC step period {step_period_s} s is not an exact whole number of nanoseconds"
            ),
            Self::MpcStepPeriodNanosecondsOutOfRange { step_period_s } => write!(
                formatter,
                "MPC step period {step_period_s} s is not representable as u64 nanoseconds"
            ),
            Self::ControlPeriodMismatch {
                control_period_ns,
                mpc_step_period_ns,
            } => write!(
                formatter,
                "control period {control_period_ns} ns does not equal MPC step period {mpc_step_period_ns} ns"
            ),
            Self::SolverBudget(source) => write!(formatter, "{source}"),
            Self::SolverBudgetNotLessThanControlPeriod {
                solver_budget_ns,
                control_period_ns,
            } => write!(
                formatter,
                "solver budget {solver_budget_ns} ns must be less than control period {control_period_ns} ns"
            ),
            Self::ShadowCommand(source) => write!(formatter, "{source}"),
            Self::IngressCapacity(source) => write!(formatter, "{source}"),
            Self::CommandLeaseDoesNotCoverControlPeriod {
                lease_ns,
                control_period_ns,
            } => write!(
                formatter,
                "command lease {lease_ns} ns does not cover one {control_period_ns} ns control period"
            ),
            Self::CommandLeaseTooLongForControlPeriod {
                lease_ns,
                control_period_ns,
                maximum_control_periods,
            } => write!(
                formatter,
                "command lease {lease_ns} ns exceeds {maximum_control_periods} control periods of {control_period_ns} ns"
            ),
            Self::FreshnessExceedsCommandLease {
                parameter,
                freshness_ns,
                lease_ns,
            } => write!(
                formatter,
                "{parameter:?} freshness {freshness_ns} ns exceeds command lease {lease_ns} ns"
            ),
            Self::MpcSolver(source) => write!(formatter, "{source}"),
        }
    }
}

impl std::error::Error for ShadowNavigationConfigParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(source) => Some(source),
            Self::TrackingCameraToBase(source) => Some(source),
            Self::WorldToOccupancy(source) => Some(source),
            Self::Odometry(source) => Some(source),
            Self::LocalCostmapGeometry(source) => Some(source),
            Self::ObstacleHeightRange(source) => Some(source),
            Self::DepthRange(source) => Some(source),
            Self::LocalCostmap(source) => Some(source),
            Self::GlobalPlanner(source) => Some(source),
            Self::PlantModel(source) => Some(source),
            Self::MpcConfig(source) => Some(source),
            Self::PathReference(source) => Some(source),
            Self::SolverBudget(source) => Some(source),
            Self::ShadowCommand(source) => Some(source),
            Self::IngressCapacity(source) => Some(source),
            Self::MpcSolver(source) => Some(source),
            Self::InputTooLarge { .. }
            | Self::UnsupportedSchemaVersion { .. }
            | Self::ZeroControlPeriod
            | Self::MpcStepPeriodNotIntegralNanoseconds { .. }
            | Self::MpcStepPeriodNanosecondsOutOfRange { .. }
            | Self::ControlPeriodMismatch { .. }
            | Self::SolverBudgetNotLessThanControlPeriod { .. }
            | Self::CommandLeaseDoesNotCoverControlPeriod { .. }
            | Self::CommandLeaseTooLongForControlPeriod { .. }
            | Self::FreshnessExceedsCommandLease { .. } => None,
        }
    }
}

fn exact_period_nanoseconds(step_period_s: f64) -> Result<u64, ShadowNavigationConfigParseError> {
    let nanoseconds = step_period_s * 1_000_000_000.0;
    if !nanoseconds.is_finite() || nanoseconds <= 0.0 || nanoseconds >= 2.0_f64.powi(64) {
        return Err(
            ShadowNavigationConfigParseError::MpcStepPeriodNanosecondsOutOfRange { step_period_s },
        );
    }
    if nanoseconds.fract() != 0.0 {
        return Err(
            ShadowNavigationConfigParseError::MpcStepPeriodNotIntegralNanoseconds { step_period_s },
        );
    }
    let parsed = nanoseconds as u64;
    if Duration::from_nanos(parsed).as_secs_f64().to_bits() != step_period_s.to_bits() {
        return Err(
            ShadowNavigationConfigParseError::MpcStepPeriodNotIntegralNanoseconds { step_period_s },
        );
    }
    Ok(parsed)
}

#[derive(Clone, Copy)]
struct OdometryFreshness {
    maximum_imu_gap_ns: u64,
    maximum_prediction_age_ns: u64,
    maximum_host_observation_age_ns: u64,
    maximum_history_bracket_gap_ns: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ShadowNavigationConfigV1Dto {
    schema_version: u32,
    coordinate_frames: CoordinateFramesV1Dto,
    odometry: OdometryV1Dto,
    local_costmap: LocalCostmapV1Dto,
    global_planner: GlobalPlannerV1Dto,
    plant_model: PlantModelJsonV1Dto,
    mpc: MpcJsonV1Dto,
    path_reference: PathReferenceJsonV1Dto,
    control_loop: ControlLoopV1Dto,
    shadow_command: ShadowCommandJsonV1Dto,
    ingress_journal: IngressJournalV1Dto,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CoordinateFramesV1Dto {
    tracking_camera_to_base: PoseV1Dto,
    world_to_occupancy: WorldToOccupancyV1Dto,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PoseV1Dto {
    rotation: [[f32; 3]; 3],
    translation_m: [f32; 3],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WorldToOccupancyV1Dto {
    rotation: [[f64; 3]; 3],
    translation_m: [f64; 3],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OdometryV1Dto {
    raw_imu_calibration: RawImuCalibrationJsonV1Dto,
    maximum_visual_interval_ns: u64,
    maximum_visual_linear_speed_m_per_sec: f64,
    maximum_visual_yaw_rate_rad_per_sec: f64,
    maximum_calibrated_yaw_rate_rad_per_sec: f64,
    minimum_gyro_accuracy: SensorAccuracyV1Dto,
    maximum_vertical_increment_m: f64,
    maximum_relative_roll_pitch_increment_rad: f64,
    maximum_absolute_map_roll_pitch_rad: f64,
    maximum_imu_gap_ns: u64,
    maximum_prediction_age_ns: u64,
    maximum_host_observation_age_ns: u64,
    maximum_history_bracket_gap_ns: u64,
    gyro_history_capacity: usize,
    pose_history_capacity: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawImuCalibrationJsonV1Dto {
    format_version: u32,
    source_id: String,
    content_id: String,
    gyro_affine: [[f64; 3]; 3],
    gyro_bias_native_rad_per_sec: [f64; 3],
    accel_affine: [[f64; 3]; 3],
    accel_bias_native_m_per_sec2: [f64; 3],
    native_imu_to_base_rotation: [[f64; 3]; 3],
}

impl RawImuCalibrationJsonV1Dto {
    fn into_domain_dto(self) -> RawImuCalibrationDto {
        RawImuCalibrationDto {
            format_version: self.format_version,
            source_id: self.source_id,
            content_id: self.content_id,
            gyro_affine: self.gyro_affine,
            gyro_bias_native_rad_per_sec: self.gyro_bias_native_rad_per_sec,
            accel_affine: self.accel_affine,
            accel_bias_native_m_per_sec2: self.accel_bias_native_m_per_sec2,
            native_imu_to_base_rotation: self.native_imu_to_base_rotation,
        }
    }
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SensorAccuracyV1Dto {
    Unreliable,
    Low,
    Medium,
    High,
}

impl SensorAccuracyV1Dto {
    const fn into_domain(self) -> SensorAccuracy {
        match self {
            Self::Unreliable => SensorAccuracy::Unreliable,
            Self::Low => SensorAccuracy::Low,
            Self::Medium => SensorAccuracy::Medium,
            Self::High => SensorAccuracy::High,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalCostmapV1Dto {
    resolution_m: f64,
    lower_bound_m: [f64; 2],
    width_cells: u32,
    height_cells: u32,
    maximum_cells: usize,
    obstacle_height_minimum_m: f64,
    obstacle_height_maximum_m: f64,
    depth_minimum_m: f64,
    depth_maximum_m: f64,
    sampling_block_pixels: u32,
    footprint_radius_m: f64,
    clearance_m: f64,
    maximum_observation_age_ns: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GlobalPlannerV1Dto {
    unknown_space_policy: UnknownSpacePolicyV1Dto,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum UnknownSpacePolicyV1Dto {
    Blocked,
    Traversable,
}

impl UnknownSpacePolicyV1Dto {
    const fn into_domain(self) -> UnknownSpacePolicy {
        match self {
            Self::Blocked => UnknownSpacePolicy::Blocked,
            Self::Traversable => UnknownSpacePolicy::Traversable,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PlantModelJsonV1Dto {
    schema_version: u32,
    model_id: String,
    model_version: u32,
    sample_period_s: f64,
    wheelbase_m: f64,
    left: WheelPlantJsonV1Dto,
    right: WheelPlantJsonV1Dto,
    validity: PlantValidityEnvelopeJsonV1Dto,
    evidence: PlantEvidenceJsonV1Dto,
}

impl PlantModelJsonV1Dto {
    fn into_domain_dto(self) -> PlantModelV1Dto {
        PlantModelV1Dto {
            schema_version: self.schema_version,
            model_id: self.model_id,
            model_version: self.model_version,
            sample_period_s: self.sample_period_s,
            wheelbase_m: self.wheelbase_m,
            left: self.left.into_domain_dto(),
            right: self.right.into_domain_dto(),
            validity: self.validity.into_domain_dto(),
            evidence: self.evidence.into_domain_dto(),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WheelPlantJsonV1Dto {
    velocity_gain_mps_per_pwm_percent: f64,
    time_constant_s: f64,
}

impl WheelPlantJsonV1Dto {
    fn into_domain_dto(self) -> WheelPlantV1Dto {
        WheelPlantV1Dto {
            velocity_gain_mps_per_pwm_percent: self.velocity_gain_mps_per_pwm_percent,
            time_constant_s: self.time_constant_s,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PlantValidityEnvelopeJsonV1Dto {
    left_pwm_min_percent: i8,
    left_pwm_max_percent: i8,
    right_pwm_min_percent: i8,
    right_pwm_max_percent: i8,
    left_velocity_min_mps: f64,
    left_velocity_max_mps: f64,
    right_velocity_min_mps: f64,
    right_velocity_max_mps: f64,
    maximum_absolute_yaw_rate_rad_per_sec: f64,
    maximum_absolute_lateral_velocity_m_per_sec: f64,
}

impl PlantValidityEnvelopeJsonV1Dto {
    fn into_domain_dto(self) -> PlantValidityEnvelopeV1Dto {
        PlantValidityEnvelopeV1Dto {
            left_pwm_min_percent: self.left_pwm_min_percent,
            left_pwm_max_percent: self.left_pwm_max_percent,
            right_pwm_min_percent: self.right_pwm_min_percent,
            right_pwm_max_percent: self.right_pwm_max_percent,
            left_velocity_min_mps: self.left_velocity_min_mps,
            left_velocity_max_mps: self.left_velocity_max_mps,
            right_velocity_min_mps: self.right_velocity_min_mps,
            right_velocity_max_mps: self.right_velocity_max_mps,
            max_abs_yaw_rate_rad_s: self.maximum_absolute_yaw_rate_rad_per_sec,
            max_abs_lateral_velocity_mps: self.maximum_absolute_lateral_velocity_m_per_sec,
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum PlantEvidenceJsonV1Dto {
    SyntheticFixture {
        fixture_id: String,
        generator_id: String,
    },
    ClaimedPhysicalIdentification {
        dataset_content_id: String,
        identification_method_id: String,
        sample_count: u64,
        residuals: FitResidualsJsonV1Dto,
    },
}

impl PlantEvidenceJsonV1Dto {
    fn into_domain_dto(self) -> PlantEvidenceV1Dto {
        match self {
            Self::SyntheticFixture {
                fixture_id,
                generator_id,
            } => PlantEvidenceV1Dto::SyntheticFixture {
                fixture_id,
                generator_id,
            },
            Self::ClaimedPhysicalIdentification {
                dataset_content_id,
                identification_method_id,
                sample_count,
                residuals,
            } => PlantEvidenceV1Dto::ClaimedPhysicalIdentification {
                dataset_content_id,
                identification_method_id,
                sample_count,
                residuals: residuals.into_domain_dto(),
            },
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FitResidualsJsonV1Dto {
    left_velocity_rmse_mps: f64,
    right_velocity_rmse_mps: f64,
    yaw_rate_rmse_rad_s: f64,
    maximum_absolute_velocity_error_mps: f64,
}

impl FitResidualsJsonV1Dto {
    fn into_domain_dto(self) -> FitResidualsV1Dto {
        FitResidualsV1Dto {
            left_velocity_rmse_mps: self.left_velocity_rmse_mps,
            right_velocity_rmse_mps: self.right_velocity_rmse_mps,
            yaw_rate_rmse_rad_s: self.yaw_rate_rmse_rad_s,
            max_abs_velocity_error_mps: self.maximum_absolute_velocity_error_mps,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MpcJsonV1Dto {
    schema_version: u32,
    horizon_steps: u16,
    step_period_s: f64,
    integration_substeps: u16,
    optimization_iterations: u8,
    candidates_per_wheel: u8,
    maximum_rollout_evaluations: u64,
    initial_search_radius_percent: u8,
    search_radius_decay_numerator: u8,
    search_radius_decay_denominator: u8,
    left_pwm_min_percent: i8,
    left_pwm_max_percent: i8,
    right_pwm_min_percent: i8,
    right_pwm_max_percent: i8,
    left_maximum_slew_percent_per_step: u16,
    right_maximum_slew_percent_per_step: u16,
    maximum_integration_tube_radius_m: f64,
    position_cost_per_m2: f64,
    heading_cost_per_rad2: f64,
    forward_velocity_cost_s2_per_m2: f64,
    yaw_rate_cost_s2_per_rad2: f64,
    pwm_cost_per_percent2: f64,
    slew_cost_per_percent2: f64,
    terminal_state_cost_multiplier: f64,
}

impl MpcJsonV1Dto {
    fn into_domain_dto(self) -> MpcConfigV1Dto {
        MpcConfigV1Dto {
            schema_version: self.schema_version,
            horizon_steps: self.horizon_steps,
            step_period_s: self.step_period_s,
            integration_substeps: self.integration_substeps,
            optimization_iterations: self.optimization_iterations,
            candidates_per_wheel: self.candidates_per_wheel,
            max_rollout_evaluations: self.maximum_rollout_evaluations,
            initial_search_radius_percent: self.initial_search_radius_percent,
            search_radius_decay_numerator: self.search_radius_decay_numerator,
            search_radius_decay_denominator: self.search_radius_decay_denominator,
            left_pwm_min_percent: self.left_pwm_min_percent,
            left_pwm_max_percent: self.left_pwm_max_percent,
            right_pwm_min_percent: self.right_pwm_min_percent,
            right_pwm_max_percent: self.right_pwm_max_percent,
            left_max_slew_percent_per_step: self.left_maximum_slew_percent_per_step,
            right_max_slew_percent_per_step: self.right_maximum_slew_percent_per_step,
            max_integration_tube_radius_m: self.maximum_integration_tube_radius_m,
            position_cost_per_m2: self.position_cost_per_m2,
            heading_cost_per_rad2: self.heading_cost_per_rad2,
            forward_velocity_cost_s2_per_m2: self.forward_velocity_cost_s2_per_m2,
            yaw_rate_cost_s2_per_rad2: self.yaw_rate_cost_s2_per_rad2,
            pwm_cost_per_percent2: self.pwm_cost_per_percent2,
            slew_cost_per_percent2: self.slew_cost_per_percent2,
            terminal_state_cost_multiplier: self.terminal_state_cost_multiplier,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PathReferenceJsonV1Dto {
    schema_version: u32,
    maximum_path_points: u32,
    minimum_segment_length_m: f64,
    maximum_path_length_m: f64,
    maximum_projection_distance_m: f64,
    target_forward_speed_m_per_sec: f64,
    goal_stop_distance_m: f64,
    maximum_absolute_yaw_rate_rad_per_sec: f64,
    nearest_segment_tie_policy: u32,
}

impl PathReferenceJsonV1Dto {
    fn into_domain_dto(self) -> PathReferenceConfigV1Dto {
        PathReferenceConfigV1Dto {
            schema_version: self.schema_version,
            maximum_path_points: self.maximum_path_points,
            minimum_segment_length_m: self.minimum_segment_length_m,
            maximum_path_length_m: self.maximum_path_length_m,
            maximum_projection_distance_m: self.maximum_projection_distance_m,
            target_forward_speed_mps: self.target_forward_speed_m_per_sec,
            goal_stop_distance_m: self.goal_stop_distance_m,
            maximum_abs_yaw_rate_rad_s: self.maximum_absolute_yaw_rate_rad_per_sec,
            nearest_segment_tie_policy: self.nearest_segment_tie_policy,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ControlLoopV1Dto {
    control_period_ns: u64,
    solver_budget_ns: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ShadowCommandJsonV1Dto {
    lease_ms: u16,
    retained_records: usize,
    initial_sequence: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IngressJournalV1Dto {
    maximum_ingress_records: usize,
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use crate::dense::occupancy::DepthToTrackingCamera;
    use crate::{FrameDimensions, PinholeIntrinsics};

    use super::super::mpc::PlantEvidenceV1;

    use super::*;

    fn camera() -> DepthCameraModel {
        let pose = Pose::try_from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.125, -0.25, 0.375],
        )
        .expect("valid camera extrinsic");
        DepthCameraModel::new(
            PinholeIntrinsics::try_new(411.0, 412.0, 319.5, 199.5).expect("valid intrinsics"),
            FrameDimensions::try_new(640, 400).expect("valid dimensions"),
            DepthToTrackingCamera::new(pose),
        )
    }

    fn fixture() -> Value {
        json!({
            "schema_version": 1,
            "coordinate_frames": {
                "tracking_camera_to_base": {
                    "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "translation_m": [0.0, 0.0, 0.0]
                },
                "world_to_occupancy": {
                    "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "translation_m": [0.0, 0.0, 0.0]
                }
            },
            "odometry": {
                "raw_imu_calibration": {
                    "format_version": 1,
                    "source_id": "fixture-calibration",
                    "content_id": "fixture-content-v1",
                    "gyro_affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "gyro_bias_native_rad_per_sec": [0.0, 0.0, 0.0],
                    "accel_affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "accel_bias_native_m_per_sec2": [0.0, 0.0, 0.0],
                    "native_imu_to_base_rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                },
                "maximum_visual_interval_ns": 500_000_000_u64,
                "maximum_visual_linear_speed_m_per_sec": 4.0,
                "maximum_visual_yaw_rate_rad_per_sec": 4.0,
                "maximum_calibrated_yaw_rate_rad_per_sec": 5.0,
                "minimum_gyro_accuracy": "low",
                "maximum_vertical_increment_m": 0.1,
                "maximum_relative_roll_pitch_increment_rad": 0.2,
                "maximum_absolute_map_roll_pitch_rad": 0.2,
                "maximum_imu_gap_ns": 100_000_000_u64,
                "maximum_prediction_age_ns": 150_000_000_u64,
                "maximum_host_observation_age_ns": 200_000_000_u64,
                "maximum_history_bracket_gap_ns": 150_000_000_u64,
                "gyro_history_capacity": 128,
                "pose_history_capacity": 128
            },
            "local_costmap": {
                "resolution_m": 0.1,
                "lower_bound_m": [-2.0, -2.0],
                "width_cells": 40,
                "height_cells": 40,
                "maximum_cells": 1600,
                "obstacle_height_minimum_m": -0.1,
                "obstacle_height_maximum_m": 1.5,
                "depth_minimum_m": 0.2,
                "depth_maximum_m": 5.0,
                "sampling_block_pixels": 1,
                "footprint_radius_m": 0.2,
                "clearance_m": 0.1,
                "maximum_observation_age_ns": 200_000_000_u64
            },
            "global_planner": {
                "unknown_space_policy": "blocked"
            },
            "plant_model": {
                "schema_version": 1,
                "model_id": "synthetic-differential-v1",
                "model_version": 1,
                "sample_period_s": 0.1,
                "wheelbase_m": 0.4,
                "left": {
                    "velocity_gain_mps_per_pwm_percent": 0.01,
                    "time_constant_s": 0.5
                },
                "right": {
                    "velocity_gain_mps_per_pwm_percent": 0.01,
                    "time_constant_s": 0.6
                },
                "validity": {
                    "left_pwm_min_percent": -50,
                    "left_pwm_max_percent": 50,
                    "right_pwm_min_percent": -50,
                    "right_pwm_max_percent": 50,
                    "left_velocity_min_mps": -0.5,
                    "left_velocity_max_mps": 0.5,
                    "right_velocity_min_mps": -0.5,
                    "right_velocity_max_mps": 0.5,
                    "maximum_absolute_yaw_rate_rad_per_sec": 2.0,
                    "maximum_absolute_lateral_velocity_m_per_sec": 0.1
                },
                "evidence": {
                    "kind": "synthetic_fixture",
                    "fixture_id": "unit-fixture",
                    "generator_id": "hand-authored-v1"
                }
            },
            "mpc": {
                "schema_version": 1,
                "horizon_steps": 10,
                "step_period_s": 0.1,
                "integration_substeps": 4,
                "optimization_iterations": 2,
                "candidates_per_wheel": 3,
                "maximum_rollout_evaluations": 1000,
                "initial_search_radius_percent": 20,
                "search_radius_decay_numerator": 1,
                "search_radius_decay_denominator": 2,
                "left_pwm_min_percent": -50,
                "left_pwm_max_percent": 50,
                "right_pwm_min_percent": -50,
                "right_pwm_max_percent": 50,
                "left_maximum_slew_percent_per_step": 10,
                "right_maximum_slew_percent_per_step": 10,
                "maximum_integration_tube_radius_m": 0.05,
                "position_cost_per_m2": 10.0,
                "heading_cost_per_rad2": 2.0,
                "forward_velocity_cost_s2_per_m2": 1.0,
                "yaw_rate_cost_s2_per_rad2": 1.0,
                "pwm_cost_per_percent2": 0.01,
                "slew_cost_per_percent2": 0.02,
                "terminal_state_cost_multiplier": 2.0
            },
            "path_reference": {
                "schema_version": 1,
                "maximum_path_points": 4096,
                "minimum_segment_length_m": 0.001,
                "maximum_path_length_m": 1000.0,
                "maximum_projection_distance_m": 2.0,
                "target_forward_speed_m_per_sec": 0.4,
                "goal_stop_distance_m": 0.3,
                "maximum_absolute_yaw_rate_rad_per_sec": 2.0,
                "nearest_segment_tie_policy": 1
            },
            "control_loop": {
                "control_period_ns": 100_000_000_u64,
                "solver_budget_ns": 50_000_000_u64
            },
            "shadow_command": {
                "lease_ms": 200,
                "retained_records": 1024,
                "initial_sequence": 0
            },
            "ingress_journal": {
                "maximum_ingress_records": 100_000
            }
        })
    }

    fn parse(value: &Value) -> Result<ShadowNavigationConfigV1, ShadowNavigationConfigParseError> {
        ShadowNavigationConfigV1::parse_json(
            &serde_json::to_vec(value).expect("serialize fixture"),
            camera(),
        )
    }

    #[test]
    fn parses_complete_fixture_and_derives_one_clearance_contract() {
        let parsed = parse(&fixture()).expect("parse fixture");
        assert_eq!(parsed.control_period().get(), 100_000_000);
        assert_eq!(parsed.solver_budget().get(), 50_000_000);
        assert_eq!(
            parsed.global_planner().unknown_space(),
            UnknownSpacePolicy::Blocked
        );
        assert_eq!(
            parsed.local_costmap().inflation_radius_m(),
            parsed.global_planner().clearance_radius_m()
        );
        assert_eq!(parsed.mpc_solver().config().step_period_s(), 0.1);
        assert_eq!(parsed.ingress_capacity().get(), 100_000);
        assert_eq!(
            parsed.world_to_occupancy(),
            parsed.odometry().world_to_occupancy()
        );
        assert_eq!(
            parsed.tracking_camera_to_base().pose().rotation(),
            parsed.local_costmap().tracking_to_base().pose().rotation()
        );
    }

    #[test]
    fn preserves_runtime_camera_without_json_extrinsic_inference() {
        let expected = camera();
        let bytes = serde_json::to_vec(&fixture()).expect("serialize fixture");
        let parsed = ShadowNavigationConfigV1::parse_json(&bytes, expected).expect("parse");
        let actual = parsed.local_costmap().camera();
        assert_eq!(actual.dimensions(), expected.dimensions());
        assert_eq!(actual.intrinsics().fx(), expected.intrinsics().fx());
        assert_eq!(actual.intrinsics().fy(), expected.intrinsics().fy());
        assert_eq!(actual.intrinsics().cx(), expected.intrinsics().cx());
        assert_eq!(actual.intrinsics().cy(), expected.intrinsics().cy());
        assert_eq!(
            actual.depth_to_tracking().pose().rotation(),
            expected.depth_to_tracking().pose().rotation()
        );
        assert_eq!(
            actual.depth_to_tracking().pose().translation(),
            expected.depth_to_tracking().pose().translation()
        );
    }

    #[test]
    fn rejects_unknown_and_missing_fields_at_nested_boundaries() {
        let mut unknown = fixture();
        unknown["odometry"]["mystery_timeout_ms"] = json!(1);
        assert!(matches!(
            parse(&unknown),
            Err(ShadowNavigationConfigParseError::Json(_))
        ));

        let mut missing = fixture();
        missing["control_loop"]
            .as_object_mut()
            .expect("object")
            .remove("solver_budget_ns");
        assert!(matches!(
            parse(&missing),
            Err(ShadowNavigationConfigParseError::Json(_))
        ));
    }

    #[test]
    fn rejects_ambiguous_unit_spelling_instead_of_defaulting() {
        let mut value = fixture();
        let loop_object = value["control_loop"].as_object_mut().expect("object");
        let period = loop_object.remove("control_period_ns").expect("period");
        loop_object.insert("control_period_ms".to_owned(), period);
        assert!(matches!(
            parse(&value),
            Err(ShadowNavigationConfigParseError::Json(_))
        ));
    }

    #[test]
    fn rejects_bad_top_version_and_unknown_enums() {
        let mut version = fixture();
        version["schema_version"] = json!(2);
        assert!(matches!(
            parse(&version),
            Err(ShadowNavigationConfigParseError::UnsupportedSchemaVersion {
                actual: 2,
                supported: 1
            })
        ));

        let mut policy = fixture();
        policy["global_planner"]["unknown_space_policy"] = json!("optimistic");
        assert!(matches!(
            parse(&policy),
            Err(ShadowNavigationConfigParseError::Json(_))
        ));
    }

    #[test]
    fn delegates_nested_versions_to_their_authoritative_domain_parsers() {
        let mut calibration = fixture();
        calibration["odometry"]["raw_imu_calibration"]["format_version"] = json!(2);
        assert!(matches!(
            parse(&calibration),
            Err(ShadowNavigationConfigParseError::Odometry(
                PlanarOdometryConfigError::RawImuCalibration(
                    super::super::odometry::RawImuCalibrationError::UnsupportedFormatVersion {
                        actual: 2,
                        supported: 1
                    }
                )
            ))
        ));

        let mut plant = fixture();
        plant["plant_model"]["schema_version"] = json!(2);
        assert!(matches!(
            parse(&plant),
            Err(ShadowNavigationConfigParseError::PlantModel(
                PlantModelParseError::UnsupportedSchemaVersion(2)
            ))
        ));

        let mut mpc = fixture();
        mpc["mpc"]["schema_version"] = json!(2);
        assert!(matches!(
            parse(&mpc),
            Err(ShadowNavigationConfigParseError::MpcConfig(
                MpcConfigParseError::UnsupportedSchemaVersion(2)
            ))
        ));

        let mut reference = fixture();
        reference["path_reference"]["schema_version"] = json!(2);
        assert!(matches!(
            parse(&reference),
            Err(ShadowNavigationConfigParseError::PathReference(
                PathReferenceConfigParseError::UnsupportedSchemaVersion(2)
            ))
        ));
    }

    #[test]
    fn reports_plant_controller_period_incompatibility_without_fallback() {
        let mut value = fixture();
        value["plant_model"]["sample_period_s"] = json!(0.2);
        assert!(matches!(
            parse(&value),
            Err(ShadowNavigationConfigParseError::MpcSolver(
                MpcCreateError::StepPeriodDoesNotMatchModel {
                    config_s: 0.1,
                    model_s: 0.2
                }
            ))
        ));
    }

    #[test]
    fn rejects_nonfinite_json_number_and_nonfinite_domain_transform() {
        let encoded = serde_json::to_string(&fixture()).expect("serialize fixture");
        let overflow = encoded.replacen("\"step_period_s\":0.1", "\"step_period_s\":1e400", 1);
        assert!(matches!(
            ShadowNavigationConfigV1::parse_json(overflow.as_bytes(), camera()),
            Err(ShadowNavigationConfigParseError::Json(_))
        ));

        let mut non_rigid = fixture();
        non_rigid["coordinate_frames"]["world_to_occupancy"]["rotation"][0][0] = json!(2.0);
        assert!(matches!(
            parse(&non_rigid),
            Err(ShadowNavigationConfigParseError::WorldToOccupancy(_))
        ));
    }

    #[test]
    fn requires_exact_integral_nanosecond_step_period_match() {
        let mut mismatch = fixture();
        mismatch["control_loop"]["control_period_ns"] = json!(100_000_001_u64);
        assert!(matches!(
            parse(&mismatch),
            Err(ShadowNavigationConfigParseError::ControlPeriodMismatch {
                control_period_ns: 100_000_001,
                mpc_step_period_ns: 100_000_000
            })
        ));

        let mut fractional = fixture();
        fractional["mpc"]["step_period_s"] = json!(0.1000000005);
        fractional["plant_model"]["sample_period_s"] = json!(0.1000000005);
        assert!(matches!(
            parse(&fractional),
            Err(ShadowNavigationConfigParseError::MpcStepPeriodNotIntegralNanoseconds { .. })
        ));
    }

    #[test]
    fn requires_solver_budget_strictly_below_tick() {
        let mut value = fixture();
        value["control_loop"]["solver_budget_ns"] = json!(100_000_000_u64);
        assert!(matches!(
            parse(&value),
            Err(ShadowNavigationConfigParseError::SolverBudgetNotLessThanControlPeriod { .. })
        ));
    }

    #[test]
    fn bounds_command_lease_to_one_or_two_ticks() {
        let mut short = fixture();
        short["shadow_command"]["lease_ms"] = json!(99);
        assert!(matches!(
            parse(&short),
            Err(ShadowNavigationConfigParseError::CommandLeaseDoesNotCoverControlPeriod { .. })
        ));

        let mut long = fixture();
        long["shadow_command"]["lease_ms"] = json!(201);
        assert!(matches!(
            parse(&long),
            Err(ShadowNavigationConfigParseError::CommandLeaseTooLongForControlPeriod { .. })
        ));
    }

    #[test]
    fn bounds_all_local_and_odom_freshness_by_command_lease() {
        for (section, field, expected) in [
            (
                "local_costmap",
                "maximum_observation_age_ns",
                FreshnessParameter::LocalCostmapObservation,
            ),
            (
                "odometry",
                "maximum_imu_gap_ns",
                FreshnessParameter::OdomGyroIntegrationGap,
            ),
            (
                "odometry",
                "maximum_prediction_age_ns",
                FreshnessParameter::OdomPrediction,
            ),
            (
                "odometry",
                "maximum_host_observation_age_ns",
                FreshnessParameter::OdomHostObservation,
            ),
            (
                "odometry",
                "maximum_history_bracket_gap_ns",
                FreshnessParameter::OdomHistoryBracketGap,
            ),
        ] {
            let mut value = fixture();
            value[section][field] = json!(200_000_001_u64);
            assert!(matches!(
                parse(&value),
                Err(ShadowNavigationConfigParseError::FreshnessExceedsCommandLease {
                    parameter,
                    ..
                }) if parameter == expected
            ));
        }
    }

    #[test]
    fn visual_interval_is_a_fresh_observation_continuity_bound_not_stale_eligibility() {
        let value = fixture();
        assert_eq!(
            value["odometry"]["maximum_visual_interval_ns"],
            json!(500_000_000_u64)
        );
        assert_eq!(value["shadow_command"]["lease_ms"], json!(200));
        parse(&value).expect(
            "a visual interval may exceed the lease because fresh host/prediction age gates still apply",
        );
    }

    #[test]
    fn preserves_synthetic_evidence_as_synthetic() {
        let parsed = parse(&fixture()).expect("parse fixture");
        match parsed.mpc_solver().model().evidence() {
            PlantEvidenceV1::SyntheticFixture {
                fixture_id,
                generator_id,
            } => {
                assert_eq!(fixture_id.as_str(), "unit-fixture");
                assert_eq!(generator_id.as_str(), "hand-authored-v1");
            }
            PlantEvidenceV1::ClaimedPhysicalIdentification { .. } => {
                panic!("synthetic evidence must not be promoted")
            }
        }
    }

    #[test]
    fn checked_in_shadow_example_stays_strict_and_synthetic() {
        let bytes = include_bytes!("../../../../configs/navigation-shadow-v1.example.json");
        let parsed = ShadowNavigationConfigV1::parse_json(bytes, camera())
            .expect("checked-in shadow configuration must satisfy the public parser");

        assert_eq!(
            parsed.global_planner().unknown_space(),
            UnknownSpacePolicy::Blocked
        );
        match parsed.mpc_solver().model().evidence() {
            PlantEvidenceV1::SyntheticFixture {
                fixture_id,
                generator_id,
            } => {
                assert_eq!(
                    fixture_id.as_str(),
                    "host-shadow-example-not-physically-validated"
                );
                assert_eq!(
                    generator_id.as_str(),
                    "hand-authored-from-shadow-config-v1-test-fixture"
                );
            }
            PlantEvidenceV1::ClaimedPhysicalIdentification { .. } => {
                panic!("the checked-in schema example must never imply physical evidence")
            }
        }
    }

    #[test]
    fn preserves_unverified_physical_evidence_as_a_claim() {
        let mut value = fixture();
        value["plant_model"]["evidence"] = json!({
            "kind": "claimed_physical_identification",
            "dataset_content_id": "caller-claimed-dataset",
            "identification_method_id": "caller-claimed-fit",
            "sample_count": 123,
            "residuals": {
                "left_velocity_rmse_mps": 0.01,
                "right_velocity_rmse_mps": 0.02,
                "yaw_rate_rmse_rad_s": 0.03,
                "maximum_absolute_velocity_error_mps": 0.04
            }
        });
        let parsed = parse(&value).expect("parse claimed evidence");
        match parsed.mpc_solver().model().evidence() {
            PlantEvidenceV1::ClaimedPhysicalIdentification {
                dataset_content_id,
                identification_method_id,
                sample_count,
                residuals,
            } => {
                assert_eq!(dataset_content_id.as_str(), "caller-claimed-dataset");
                assert_eq!(identification_method_id.as_str(), "caller-claimed-fit");
                assert_eq!(sample_count.get(), 123);
                assert_eq!(residuals.max_abs_velocity_error_mps, 0.04);
            }
            PlantEvidenceV1::SyntheticFixture { .. } => {
                panic!("claimed evidence must remain explicitly claimed")
            }
        }
    }

    #[test]
    fn rejects_input_before_json_parsing_when_size_bound_is_exceeded() {
        let oversized = vec![b' '; MAX_SHADOW_NAVIGATION_CONFIG_JSON_BYTES + 1];
        assert!(matches!(
            ShadowNavigationConfigV1::parse_json(&oversized, camera()),
            Err(ShadowNavigationConfigParseError::InputTooLarge {
                actual_bytes,
                maximum_bytes: MAX_SHADOW_NAVIGATION_CONFIG_JSON_BYTES
            }) if actual_bytes == MAX_SHADOW_NAVIGATION_CONFIG_JSON_BYTES + 1
        ));
    }

    #[test]
    fn requires_nonzero_bounded_ingress_session_capacity() {
        let mut zero = fixture();
        zero["ingress_journal"]["maximum_ingress_records"] = json!(0);
        assert!(matches!(
            parse(&zero),
            Err(ShadowNavigationConfigParseError::IngressCapacity(
                NavigationIngressCapacityError::Zero
            ))
        ));

        let mut too_large = fixture();
        too_large["ingress_journal"]["maximum_ingress_records"] =
            json!(ShadowNavigationConfigV1::MAX_INGRESS_RECORDS + 1);
        assert!(matches!(
            parse(&too_large),
            Err(ShadowNavigationConfigParseError::IngressCapacity(
                NavigationIngressCapacityError::TooLarge { .. }
            ))
        ));
    }

    #[test]
    fn nested_evidence_rejects_unknown_fields() {
        let mut value = fixture();
        value["plant_model"]["evidence"]["verified"] = json!(true);
        assert!(matches!(
            parse(&value),
            Err(ShadowNavigationConfigParseError::Json(_))
        ));
    }
}
