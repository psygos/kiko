//! Reusable preparation boundary for Kiko's live navigation runtime.
//!
//! This module deliberately stops before device and dataset activation. It
//! turns weak CLI-style options and bounded configuration files into typed
//! runtime parts that can be consumed by either `kiko-slam` or the production
//! Nano agent without duplicating parsing, validation, or assembly logic.

use std::fs::File;
use std::io::Read as _;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use crate::DeviceSessionId;
use crate::dense::occupancy::{
    DepthCameraModel, OccupancyConfig, OccupancyConfigError, OccupancyEvidenceModel,
    OccupancyGridGeometry, OccupancyGridGeometryError,
};
use crate::dense::occupancy_runtime::{
    OccupancyRuntimeConfig, OccupancySnapshotCadence, OccupancySnapshotCadenceError,
};
use crate::env::{EnvError, env_f64, env_u32, env_usize};
use crate::navigation::mpc::MpcConfigV1;
#[cfg(feature = "actuation")]
use crate::navigation::{
    ActuationConfigParseError, MAX_NAVIGATION_ACTUATION_CONFIG_JSON_BYTES,
    NavigationActuationConfigV1,
};
use crate::navigation::{
    ControlPeriodNs, GlobalPlannerConfig, LocalCostmap, LocalCostmapError, NavigationGoalArg,
    NavigationIngressCapacity, PathReferenceBuilderV1, PlanarOdometry, SafetySupervisorCreateError,
    ShadowNavigationConfigParseError, ShadowNavigationConfigV1, ShadowSafetySupervisor,
    SolverBudgetNs,
};

/// A coherent request to either omit or enable live navigation.
///
/// `Enabled` always carries both inputs that are required to bind a live
/// navigation dataset. Physical authority is represented separately so a
/// partially specified actuation request cannot cross this boundary.
#[derive(Clone, Debug, PartialEq)]
pub enum LiveNavigationRequest {
    Disabled,
    Enabled {
        config_path: PathBuf,
        goal: Option<NavigationGoalArg>,
        dataset_path: PathBuf,
        actuation: LiveActuationRequest,
    },
}

/// Requested actuation mode for one enabled live-navigation session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LiveActuationRequest {
    ShadowOnly,
    Physical {
        config_path: PathBuf,
        exact_robot_id: String,
    },
}

impl LiveNavigationRequest {
    /// Parse weak, independently optional boundary values into one coherent
    /// request. No file is opened by this step.
    pub fn parse(
        config_path: Option<PathBuf>,
        goal: Option<NavigationGoalArg>,
        dataset_path: Option<PathBuf>,
        actuation_config_path: Option<PathBuf>,
        exact_robot_id: Option<String>,
    ) -> Result<Self, LiveNavigationBoundaryError> {
        match (config_path, dataset_path) {
            (None, None) if goal.is_none() => {
                if actuation_config_path.is_some() || exact_robot_id.is_some() {
                    Err(LiveNavigationBoundaryError::ActuationRequiresNavigation)
                } else {
                    Ok(Self::Disabled)
                }
            }
            (Some(config_path), Some(dataset_path)) => {
                let actuation = match (actuation_config_path, exact_robot_id) {
                    (None, None) => LiveActuationRequest::ShadowOnly,
                    (Some(config_path), Some(exact_robot_id)) => LiveActuationRequest::Physical {
                        config_path,
                        exact_robot_id,
                    },
                    (config_path, exact_robot_id) => {
                        return Err(LiveNavigationBoundaryError::IncompleteActuation {
                            missing_config: config_path.is_none(),
                            missing_robot_id: exact_robot_id.is_none(),
                        });
                    }
                };
                Ok(Self::Enabled {
                    config_path,
                    goal,
                    dataset_path,
                    actuation,
                })
            }
            (config_path, dataset_path) => Err(LiveNavigationBoundaryError::IncompleteNavigation {
                missing_config: config_path.is_none(),
                missing_dataset: dataset_path.is_none(),
            }),
        }
    }

    pub fn is_enabled(&self) -> bool {
        matches!(self, Self::Enabled { .. })
    }

    /// Read every configuration document at most once with its format-specific
    /// byte bound. The returned type cannot represent an enabled request whose
    /// navigation policy has not been loaded.
    pub fn load(self) -> Result<LoadedLiveNavigationRequest, LiveNavigationLoadError> {
        let Self::Enabled {
            config_path,
            goal,
            dataset_path,
            actuation,
        } = self
        else {
            return Ok(LoadedLiveNavigationRequest {
                request: Self::Disabled,
                navigation_config_bytes: None,
                actuation_config_bytes: None,
                occupancy_host_policy: None,
            });
        };

        let navigation_config_bytes = read_live_config_bounded(
            &config_path,
            "navigation",
            ShadowNavigationConfigV1::MAX_JSON_BYTES,
        )?;
        let actuation_config_bytes = match &actuation {
            LiveActuationRequest::ShadowOnly => None,
            LiveActuationRequest::Physical { config_path, .. } => {
                #[cfg(feature = "actuation")]
                {
                    Some(read_live_config_bounded(
                        config_path,
                        "navigation actuation",
                        MAX_NAVIGATION_ACTUATION_CONFIG_JSON_BYTES,
                    )?)
                }
                #[cfg(not(feature = "actuation"))]
                {
                    let _ = config_path;
                    return Err(LiveNavigationLoadError::PhysicalActuationFeatureDisabled);
                }
            }
        };
        let occupancy_host_policy = LiveOccupancyHostPolicy::load_from_environment()?;
        Ok(LoadedLiveNavigationRequest {
            request: Self::Enabled {
                config_path,
                goal,
                dataset_path,
                actuation,
            },
            navigation_config_bytes: Some(navigation_config_bytes),
            actuation_config_bytes,
            occupancy_host_policy: Some(occupancy_host_policy),
        })
    }
}

/// A live-navigation request paired with the exact bounded bytes loaded for it.
///
/// Fields remain private so callers cannot construct mismatched request/bytes
/// combinations. The bytes are retained because physical authority binds the
/// exact navigation document by digest.
pub struct LoadedLiveNavigationRequest {
    request: LiveNavigationRequest,
    navigation_config_bytes: Option<Vec<u8>>,
    actuation_config_bytes: Option<Vec<u8>>,
    occupancy_host_policy: Option<LiveOccupancyHostPolicy>,
}

impl LoadedLiveNavigationRequest {
    pub fn request(&self) -> &LiveNavigationRequest {
        &self.request
    }

    pub fn is_enabled(&self) -> bool {
        self.request.is_enabled()
    }
}

/// Host-owned occupancy resource policy captured once at request load.
///
/// Every field is already a domain type, so a loaded enabled request cannot
/// retain syntactically valid but semantically invalid occupancy policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LiveOccupancyHostPolicy {
    geometry: OccupancyGridGeometry,
    evidence: OccupancyEvidenceModel,
    maximum_keyframes: NonZeroUsize,
    snapshot_cadence: OccupancySnapshotCadence,
}

impl LiveOccupancyHostPolicy {
    /// Parse the host-owned global-map resource envelope exactly once.
    ///
    /// Projection, camera, height/depth eligibility, and sampling remain in
    /// the exact shadow-navigation document. This type owns only global grid
    /// extent, retained evidence capacity, and publication cadence.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        resolution_m: f64,
        lower_x_m: f64,
        lower_y_m: f64,
        width_cells: u32,
        height_cells: u32,
        maximum_cells: usize,
        maximum_keyframes: usize,
        snapshot_every_keyframes: usize,
    ) -> Result<Self, LiveOccupancyHostPolicyError> {
        let geometry = OccupancyGridGeometry::try_new(
            resolution_m,
            [lower_x_m, lower_y_m],
            width_cells,
            height_cells,
            maximum_cells,
        )
        .map_err(LiveOccupancyHostPolicyError::Geometry)?;
        let evidence = OccupancyEvidenceModel::try_new(-1, 3, -2, 2)
            .expect("fixed live occupancy evidence model is valid");
        let maximum_keyframes = parse_live_occupancy_maximum_keyframes(
            maximum_keyframes,
            evidence,
        )
        .map_err(LiveOccupancyHostPolicyError::MaximumKeyframes)?;
        let snapshot_cadence =
            OccupancySnapshotCadence::try_new(snapshot_every_keyframes)
                .map_err(LiveOccupancyHostPolicyError::SnapshotCadence)?;
        Ok(Self {
            geometry,
            evidence,
            maximum_keyframes,
            snapshot_cadence,
        })
    }

    pub const fn geometry(self) -> OccupancyGridGeometry {
        self.geometry
    }

    pub const fn evidence(self) -> OccupancyEvidenceModel {
        self.evidence
    }

    pub const fn maximum_keyframes(self) -> NonZeroUsize {
        self.maximum_keyframes
    }

    pub const fn snapshot_cadence(self) -> OccupancySnapshotCadence {
        self.snapshot_cadence
    }

    fn load_from_environment() -> Result<Self, LiveNavigationLoadError> {
        let resolution_m = env_f64("KIKO_OCCUPANCY_RESOLUTION_M")?.unwrap_or(0.05);
        let lower_x_m = env_f64("KIKO_OCCUPANCY_LOWER_X_M")?.unwrap_or(-10.0);
        let lower_y_m = env_f64("KIKO_OCCUPANCY_LOWER_Y_M")?.unwrap_or(-5.0);
        let width = env_u32("KIKO_OCCUPANCY_WIDTH_CELLS")?.unwrap_or(400);
        let height = env_u32("KIKO_OCCUPANCY_HEIGHT_CELLS")?.unwrap_or(400);
        let maximum_cells = env_usize("KIKO_OCCUPANCY_MAX_CELLS")?.unwrap_or(4_000_000);
        let maximum_keyframes = env_usize("KIKO_OCCUPANCY_MAX_KEYFRAMES")?.unwrap_or(300);
        let snapshot_cadence = env_usize("KIKO_OCCUPANCY_RERUN_EVERY_KEYFRAMES")?.unwrap_or(5);

        Self::try_new(
            resolution_m,
            lower_x_m,
            lower_y_m,
            width,
            height,
            maximum_cells,
            maximum_keyframes,
            snapshot_cadence,
        )
        .map_err(LiveNavigationLoadError::OccupancyHostPolicy)
    }
}

#[derive(Debug)]
pub enum LiveOccupancyHostPolicyError {
    Geometry(OccupancyGridGeometryError),
    MaximumKeyframes(OccupancyConfigError),
    SnapshotCadence(OccupancySnapshotCadenceError),
}

impl std::fmt::Display for LiveOccupancyHostPolicyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Geometry(source) => source.fmt(formatter),
            Self::MaximumKeyframes(source) => source.fmt(formatter),
            Self::SnapshotCadence(source) => source.fmt(formatter),
        }
    }
}

impl std::error::Error for LiveOccupancyHostPolicyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Geometry(source) => Some(source),
            Self::MaximumKeyframes(source) => Some(source),
            Self::SnapshotCadence(source) => Some(source),
        }
    }
}

fn parse_live_occupancy_maximum_keyframes(
    maximum_keyframes: usize,
    evidence: OccupancyEvidenceModel,
) -> Result<NonZeroUsize, OccupancyConfigError> {
    let maximum_keyframes =
        NonZeroUsize::new(maximum_keyframes).ok_or(OccupancyConfigError::ZeroMaximumKeyframes)?;
    let count = u128::try_from(maximum_keyframes.get())
        .expect("usize occupancy keyframe bound always fits u128");
    let free_magnitude = count * u128::from(evidence.free_delta().unsigned_abs());
    let occupied_magnitude = count * u128::from(evidence.occupied_delta().unsigned_abs());
    if free_magnitude > u128::from(i32::MIN.unsigned_abs())
        || occupied_magnitude > u128::from(i32::MAX.unsigned_abs())
    {
        return Err(OccupancyConfigError::EvidenceAccumulatorMayOverflow {
            maximum_keyframes: maximum_keyframes.get(),
            free_delta: evidence.free_delta(),
            occupied_delta: evidence.occupied_delta(),
        });
    }
    Ok(maximum_keyframes)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiveNavigationBoundaryError {
    IncompleteNavigation {
        missing_config: bool,
        missing_dataset: bool,
    },
    IncompleteActuation {
        missing_config: bool,
        missing_robot_id: bool,
    },
    ActuationRequiresNavigation,
}

impl std::fmt::Display for LiveNavigationBoundaryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::IncompleteNavigation {
                missing_config,
                missing_dataset,
            } => {
                formatter.write_str(
                    "live navigation requires --navigation-config and --navigation-record together; --navigation-goal is optional; missing",
                )?;
                if missing_config {
                    formatter.write_str(" --navigation-config")?;
                }
                if missing_dataset {
                    formatter.write_str(" --navigation-record")?;
                }
                Ok(())
            }
            Self::IncompleteActuation {
                missing_config,
                missing_robot_id,
            } => {
                formatter.write_str(
                    "physical navigation actuation requires --navigation-actuation-config and --navigation-arm-robot together; missing",
                )?;
                if missing_config {
                    formatter.write_str(" --navigation-actuation-config")?;
                }
                if missing_robot_id {
                    formatter.write_str(" --navigation-arm-robot")?;
                }
                Ok(())
            }
            Self::ActuationRequiresNavigation => formatter
                .write_str("physical actuation options require a complete live navigation request"),
        }
    }
}

impl std::error::Error for LiveNavigationBoundaryError {}

#[derive(Debug)]
pub enum LiveNavigationLoadError {
    Config(LiveNavigationConfigReadError),
    Environment(EnvError),
    OccupancyHostPolicy(LiveOccupancyHostPolicyError),
    OccupancyCadence(OccupancySnapshotCadenceError),
    OccupancyGeometry(OccupancyGridGeometryError),
    OccupancyConfig(OccupancyConfigError),
    PhysicalActuationFeatureDisabled,
}

impl std::fmt::Display for LiveNavigationLoadError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(source) => source.fmt(formatter),
            Self::Environment(source) => source.fmt(formatter),
            Self::OccupancyHostPolicy(source) => source.fmt(formatter),
            Self::OccupancyCadence(source) => source.fmt(formatter),
            Self::OccupancyGeometry(source) => source.fmt(formatter),
            Self::OccupancyConfig(source) => source.fmt(formatter),
            Self::PhysicalActuationFeatureDisabled => formatter.write_str(
                "physical live navigation was requested without the actuation build feature",
            ),
        }
    }
}

impl std::error::Error for LiveNavigationLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Config(source) => Some(source),
            Self::Environment(source) => Some(source),
            Self::OccupancyHostPolicy(source) => Some(source),
            Self::OccupancyCadence(source) => Some(source),
            Self::OccupancyGeometry(source) => Some(source),
            Self::OccupancyConfig(source) => Some(source),
            Self::PhysicalActuationFeatureDisabled => None,
        }
    }
}

impl From<LiveNavigationConfigReadError> for LiveNavigationLoadError {
    fn from(source: LiveNavigationConfigReadError) -> Self {
        Self::Config(source)
    }
}

macro_rules! load_error_from {
    ($source:ty, $variant:ident) => {
        impl From<$source> for LiveNavigationLoadError {
            fn from(source: $source) -> Self {
                Self::$variant(source)
            }
        }
    };
}

load_error_from!(EnvError, Environment);
load_error_from!(OccupancySnapshotCadenceError, OccupancyCadence);
load_error_from!(OccupancyGridGeometryError, OccupancyGeometry);
load_error_from!(OccupancyConfigError, OccupancyConfig);

#[derive(Debug)]
pub enum LiveNavigationConfigReadError {
    Open {
        kind: &'static str,
        path: PathBuf,
        source: std::io::Error,
    },
    Read {
        kind: &'static str,
        path: PathBuf,
        source: std::io::Error,
    },
    InputTooLarge {
        kind: &'static str,
        path: PathBuf,
        actual_bytes_at_least: usize,
        maximum_bytes: usize,
    },
}

impl std::fmt::Display for LiveNavigationConfigReadError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Open { kind, path, source } => write!(
                formatter,
                "cannot open {kind} config {}: {source}",
                path.display()
            ),
            Self::Read { kind, path, source } => write!(
                formatter,
                "cannot read {kind} config {}: {source}",
                path.display()
            ),
            Self::InputTooLarge {
                kind,
                path,
                actual_bytes_at_least,
                maximum_bytes,
            } => write!(
                formatter,
                "{kind} config {} is at least {actual_bytes_at_least} bytes; maximum is {maximum_bytes} bytes",
                path.display()
            ),
        }
    }
}

impl std::error::Error for LiveNavigationConfigReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Open { source, .. } | Self::Read { source, .. } => Some(source),
            Self::InputTooLarge { .. } => None,
        }
    }
}

fn read_live_config_bounded(
    path: &Path,
    kind: &'static str,
    maximum_bytes: usize,
) -> Result<Vec<u8>, LiveNavigationConfigReadError> {
    let file = File::open(path).map_err(|source| LiveNavigationConfigReadError::Open {
        kind,
        path: path.to_path_buf(),
        source,
    })?;
    let read_bound = u64::try_from(maximum_bytes).expect("live config byte bound fits u64") + 1;
    let mut bytes = Vec::with_capacity(maximum_bytes.min(16 * 1024));
    file.take(read_bound)
        .read_to_end(&mut bytes)
        .map_err(|source| LiveNavigationConfigReadError::Read {
            kind,
            path: path.to_path_buf(),
            source,
        })?;
    if bytes.len() > maximum_bytes {
        return Err(LiveNavigationConfigReadError::InputTooLarge {
            kind,
            path: path.to_path_buf(),
            actual_bytes_at_least: bytes.len(),
            maximum_bytes,
        });
    }
    Ok(bytes)
}

/// Facts established by device/config assembly before live navigation may
/// activate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LiveNavigationPrerequisites {
    depth_enabled: bool,
    imu_enabled: bool,
    dense_occupancy_enabled: bool,
    rectified_stereo: bool,
}

impl LiveNavigationPrerequisites {
    pub const fn new(
        depth_enabled: bool,
        imu_enabled: bool,
        dense_occupancy_enabled: bool,
        rectified_stereo: bool,
    ) -> Self {
        Self {
            depth_enabled,
            imu_enabled,
            dense_occupancy_enabled,
            rectified_stereo,
        }
    }

    pub fn require_for(
        self,
        request: &LiveNavigationRequest,
    ) -> Result<(), LiveNavigationPrerequisiteError> {
        if !request.is_enabled() {
            return Ok(());
        }
        if !self.depth_enabled {
            return Err(LiveNavigationPrerequisiteError::DepthDisabled);
        }
        if !self.imu_enabled {
            return Err(LiveNavigationPrerequisiteError::ImuDisabled);
        }
        if !self.dense_occupancy_enabled {
            return Err(LiveNavigationPrerequisiteError::DenseOccupancyDisabled);
        }
        if !self.rectified_stereo {
            return Err(LiveNavigationPrerequisiteError::UnrectifiedStereo);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiveNavigationPrerequisiteError {
    DepthDisabled,
    ImuDisabled,
    DenseOccupancyDisabled,
    UnrectifiedStereo,
}

impl std::fmt::Display for LiveNavigationPrerequisiteError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::DepthDisabled => {
                "live navigation requires KIKO_LIVE_DEPTH=true and rectified-left metric depth"
            }
            Self::ImuDisabled => "live navigation requires --imu-rate-hz (or KIKO_IMU_RATE_HZ)",
            Self::DenseOccupancyDisabled => {
                "live navigation requires KIKO_DENSE=true for global occupancy snapshots"
            }
            Self::UnrectifiedStereo => {
                "live navigation requires --rectified=true for the rectified-left depth contract"
            }
        })
    }
}

impl std::error::Error for LiveNavigationPrerequisiteError {}

/// Fully parsed, allocated live-navigation components ready for dataset and
/// device-session activation.
pub struct PreparedLiveNavigationRuntime {
    goal: Option<NavigationGoalArg>,
    dataset_path: PathBuf,
    control_period: ControlPeriodNs,
    occupancy_config: Option<OccupancyRuntimeConfig>,
    ingress_capacity: NavigationIngressCapacity,
    odometry: PlanarOdometry,
    local_costmap: LocalCostmap,
    global_planner: GlobalPlannerConfig,
    reference_builder: PathReferenceBuilderV1,
    mpc_config: MpcConfigV1,
    solver_budget: SolverBudgetNs,
    safety: ShadowSafetySupervisor,
    #[cfg(feature = "actuation")]
    actuation: Option<NavigationActuationConfigV1>,
}

impl PreparedLiveNavigationRuntime {
    /// Move the authoritative global-occupancy configuration to its sole
    /// worker owner. A second call returns `None` instead of silently cloning a
    /// potentially large configuration.
    pub fn take_occupancy_config(&mut self) -> Option<OccupancyRuntimeConfig> {
        self.occupancy_config.take()
    }

    pub fn into_parts(self) -> PreparedLiveNavigationRuntimeParts {
        PreparedLiveNavigationRuntimeParts {
            goal: self.goal,
            dataset_path: self.dataset_path,
            control_period: self.control_period,
            ingress_capacity: self.ingress_capacity,
            odometry: self.odometry,
            local_costmap: self.local_costmap,
            global_planner: self.global_planner,
            reference_builder: self.reference_builder,
            mpc_config: self.mpc_config,
            solver_budget: self.solver_budget,
            safety: self.safety,
            #[cfg(feature = "actuation")]
            actuation: self.actuation,
        }
    }
}

/// Owned activation parts. This is the explicit seam between pure
/// configuration preparation and side-effecting dataset/device activation.
pub struct PreparedLiveNavigationRuntimeParts {
    pub goal: Option<NavigationGoalArg>,
    pub dataset_path: PathBuf,
    pub control_period: ControlPeriodNs,
    pub ingress_capacity: NavigationIngressCapacity,
    pub odometry: PlanarOdometry,
    pub local_costmap: LocalCostmap,
    pub global_planner: GlobalPlannerConfig,
    pub reference_builder: PathReferenceBuilderV1,
    pub mpc_config: MpcConfigV1,
    pub solver_budget: SolverBudgetNs,
    pub safety: ShadowSafetySupervisor,
    #[cfg(feature = "actuation")]
    pub actuation: Option<NavigationActuationConfigV1>,
}

#[derive(Debug)]
pub enum LiveNavigationPreparationError {
    NavigationConfig(ShadowNavigationConfigParseError),
    OccupancyConfig(OccupancyConfigError),
    LocalCostmap(LocalCostmapError),
    Safety(SafetySupervisorCreateError),
    #[cfg(feature = "actuation")]
    ActuationConfig(ActuationConfigParseError),
}

impl std::fmt::Display for LiveNavigationPreparationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NavigationConfig(source) => source.fmt(formatter),
            Self::OccupancyConfig(source) => source.fmt(formatter),
            Self::LocalCostmap(source) => source.fmt(formatter),
            Self::Safety(source) => source.fmt(formatter),
            #[cfg(feature = "actuation")]
            Self::ActuationConfig(source) => source.fmt(formatter),
        }
    }
}

impl std::error::Error for LiveNavigationPreparationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NavigationConfig(source) => Some(source),
            Self::OccupancyConfig(source) => Some(source),
            Self::LocalCostmap(source) => Some(source),
            Self::Safety(source) => Some(source),
            #[cfg(feature = "actuation")]
            Self::ActuationConfig(source) => Some(source),
        }
    }
}

macro_rules! preparation_error_from {
    ($source:ty, $variant:ident) => {
        impl From<$source> for LiveNavigationPreparationError {
            fn from(source: $source) -> Self {
                Self::$variant(source)
            }
        }
    };
}

preparation_error_from!(ShadowNavigationConfigParseError, NavigationConfig);
preparation_error_from!(OccupancyConfigError, OccupancyConfig);
preparation_error_from!(LocalCostmapError, LocalCostmap);
preparation_error_from!(SafetySupervisorCreateError, Safety);
#[cfg(feature = "actuation")]
preparation_error_from!(ActuationConfigParseError, ActuationConfig);

/// Parse and allocate every hardware-independent component exactly once using
/// the runtime depth-camera model admitted for this device session.
pub fn prepare_live_navigation_runtime(
    loaded: LoadedLiveNavigationRequest,
    runtime_depth_camera: DepthCameraModel,
    device_session: DeviceSessionId,
) -> Result<Option<PreparedLiveNavigationRuntime>, LiveNavigationPreparationError> {
    let LoadedLiveNavigationRequest {
        request,
        navigation_config_bytes,
        actuation_config_bytes,
        occupancy_host_policy,
    } = loaded;
    let LiveNavigationRequest::Enabled {
        goal,
        dataset_path,
        actuation,
        ..
    } = request
    else {
        debug_assert!(navigation_config_bytes.is_none());
        debug_assert!(actuation_config_bytes.is_none());
        debug_assert!(occupancy_host_policy.is_none());
        return Ok(None);
    };
    let bytes = navigation_config_bytes
        .expect("loaded enabled navigation always owns its bounded policy bytes");
    let occupancy_host_policy = occupancy_host_policy
        .expect("loaded enabled navigation always owns parsed occupancy host policy");
    let parsed = ShadowNavigationConfigV1::parse_json(&bytes, runtime_depth_camera)?;
    #[cfg(feature = "actuation")]
    let actuation = match actuation {
        LiveActuationRequest::ShadowOnly => None,
        LiveActuationRequest::Physical { exact_robot_id, .. } => {
            let actuation_bytes = actuation_config_bytes
                .expect("loaded physical navigation always owns bounded authority bytes");
            Some(NavigationActuationConfigV1::parse_and_authorize(
                &actuation_bytes,
                &exact_robot_id,
                &bytes,
                parsed.mpc_solver().model(),
                parsed.solver_budget(),
                parsed.control_period(),
            )?)
        }
    };
    #[cfg(not(feature = "actuation"))]
    debug_assert!(matches!(actuation, LiveActuationRequest::ShadowOnly));

    assemble_live_navigation_runtime(
        parsed,
        goal,
        dataset_path,
        occupancy_host_policy,
        device_session,
        #[cfg(feature = "actuation")]
        actuation,
    )
    .map(Some)
}

/// Allocate live navigation from a shadow policy which was already parsed
/// against this device session's exact depth-camera model.
///
/// This is the production seam: strict launch loading owns the bytes,
/// production admission owns the physical driver, and this function therefore
/// neither reopens a path nor reparses the navigation/actuation documents. The
/// returned runtime is shadow-only internally; the sole production motion
/// owner receives its already-admitted driver through a separate type.
pub fn prepare_live_navigation_runtime_from_parsed(
    parsed: ShadowNavigationConfigV1,
    goal: Option<NavigationGoalArg>,
    dataset_path: PathBuf,
    occupancy_host_policy: LiveOccupancyHostPolicy,
    device_session: DeviceSessionId,
) -> Result<PreparedLiveNavigationRuntime, LiveNavigationPreparationError> {
    assemble_live_navigation_runtime(
        parsed,
        goal,
        dataset_path,
        occupancy_host_policy,
        device_session,
        #[cfg(feature = "actuation")]
        None,
    )
}

fn assemble_live_navigation_runtime(
    parsed: ShadowNavigationConfigV1,
    goal: Option<NavigationGoalArg>,
    dataset_path: PathBuf,
    occupancy_host_policy: LiveOccupancyHostPolicy,
    device_session: DeviceSessionId,
    #[cfg(feature = "actuation")] actuation: Option<NavigationActuationConfigV1>,
) -> Result<PreparedLiveNavigationRuntime, LiveNavigationPreparationError> {
    let occupancy_config =
        build_navigation_occupancy_runtime_config(&parsed, occupancy_host_policy)?;
    let parts = parsed.into_runtime_parts();
    let mpc_config = parts.mpc_solver.config();
    let odometry = PlanarOdometry::new(parts.odometry);
    let local_costmap = LocalCostmap::try_new(parts.local_costmap, device_session)?;
    let reference_builder = PathReferenceBuilderV1::new(parts.path_reference);
    let safety = ShadowSafetySupervisor::try_new(parts.mpc_solver, parts.shadow_command)?;
    Ok(PreparedLiveNavigationRuntime {
        goal,
        dataset_path,
        control_period: parts.control_period,
        occupancy_config: Some(occupancy_config),
        ingress_capacity: parts.ingress_capacity,
        odometry,
        local_costmap,
        global_planner: parts.global_planner,
        reference_builder,
        mpc_config,
        solver_budget: parts.solver_budget,
        safety,
        #[cfg(feature = "actuation")]
        actuation,
    })
}

/// Build the global dense map from the already-parsed navigation contract.
///
/// Global extent, retained evidence, and snapshot cadence remain host resource
/// policy. Coordinate frame, camera, height/depth eligibility, and sampling
/// come only from the strict navigation document.
fn build_navigation_occupancy_runtime_config(
    navigation: &ShadowNavigationConfigV1,
    host_policy: LiveOccupancyHostPolicy,
) -> Result<OccupancyRuntimeConfig, LiveNavigationPreparationError> {
    let local = navigation.local_costmap();
    let world_to_occupancy = navigation.world_to_occupancy();
    let mapper = OccupancyConfig::try_new(
        host_policy.geometry,
        world_to_occupancy,
        local.camera(),
        local.obstacle_height_range(),
        local.depth_range(),
        local.sampling_block(),
        host_policy.evidence,
        host_policy.maximum_keyframes.get(),
    )?;
    let geometry = host_policy.geometry;
    eprintln!(
        "navigation occupancy: geometric=true learned=false grid={}x{} resolution_m={} lower_xy_m=[{},{}] max_keyframes={} snapshot_every_keyframes={} world_to_occupancy_rotation={:?} world_to_occupancy_translation_m={:?}",
        geometry.width(),
        geometry.height(),
        geometry.resolution_m(),
        geometry.lower_bound_m()[0],
        geometry.lower_bound_m()[1],
        host_policy.maximum_keyframes,
        host_policy.snapshot_cadence.get(),
        world_to_occupancy.rotation(),
        world_to_occupancy.translation_m(),
    );
    Ok(OccupancyRuntimeConfig::new(
        mapper,
        host_policy.snapshot_cadence,
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        LiveActuationRequest, LiveNavigationBoundaryError, LiveNavigationConfigReadError,
        LiveNavigationLoadError, LiveNavigationPrerequisiteError, LiveNavigationPrerequisites,
        LiveNavigationRequest, LiveOccupancyHostPolicy, LiveOccupancyHostPolicyError,
    };
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMPORARY_CONFIG: AtomicU64 = AtomicU64::new(0);

    struct TemporaryConfig(PathBuf);

    impl TemporaryConfig {
        fn new(bytes: &[u8]) -> Self {
            let sequence = NEXT_TEMPORARY_CONFIG.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "kiko-live-runtime-{}-{sequence}.json",
                std::process::id()
            ));
            std::fs::write(&path, bytes).expect("write unique temporary live config");
            Self(path)
        }

        fn path(&self) -> PathBuf {
            self.0.clone()
        }
    }

    impl Drop for TemporaryConfig {
        fn drop(&mut self) {
            if let Err(source) = std::fs::remove_file(&self.0) {
                eprintln!(
                    "could not remove temporary live config {}: {source}",
                    self.0.display()
                );
            }
        }
    }

    fn goal() -> crate::navigation::NavigationGoalArg {
        "1.25,-2.5"
            .parse()
            .expect("finite map-frame navigation goal")
    }

    #[test]
    fn disabled_request_requires_every_option_to_be_absent() {
        let request =
            LiveNavigationRequest::parse(None, None, None, None, None).expect("all options absent");
        assert_eq!(request, LiveNavigationRequest::Disabled);
        assert!(!request.is_enabled());
        let loaded = request.load().expect("disabled request loads no files");
        assert!(!loaded.is_enabled());
        assert_eq!(loaded.navigation_config_bytes, None);
        assert_eq!(loaded.actuation_config_bytes, None);
        assert!(loaded.occupancy_host_policy.is_none());
    }

    #[test]
    fn complete_request_preserves_mapping_and_physical_authority_inputs() {
        let request = LiveNavigationRequest::parse(
            Some(PathBuf::from("navigation.json")),
            Some(goal()),
            Some(PathBuf::from("capture")),
            Some(PathBuf::from("actuation.json")),
            Some("kiko-01".to_owned()),
        )
        .expect("complete physical request");
        assert_eq!(
            request,
            LiveNavigationRequest::Enabled {
                config_path: PathBuf::from("navigation.json"),
                goal: Some(goal()),
                dataset_path: PathBuf::from("capture"),
                actuation: LiveActuationRequest::Physical {
                    config_path: PathBuf::from("actuation.json"),
                    exact_robot_id: "kiko-01".to_owned(),
                },
            }
        );
    }

    #[test]
    fn occupancy_host_resources_parse_once_and_preserve_metric_geometry() {
        let policy =
            LiveOccupancyHostPolicy::try_new(0.05, -10.0, -5.0, 400, 300, 120_000, 300, 5)
                .expect("bounded host occupancy policy");

        let geometry = policy.geometry();
        assert_eq!(geometry.resolution_m(), 0.05);
        assert_eq!(geometry.lower_bound_m(), [-10.0, -5.0]);
        assert_eq!((geometry.width(), geometry.height()), (400, 300));
        assert_eq!(policy.maximum_keyframes().get(), 300);
        assert_eq!(policy.snapshot_cadence().get(), 5);
    }

    #[test]
    fn occupancy_host_resources_reject_each_unrepresentable_state() {
        assert!(matches!(
            LiveOccupancyHostPolicy::try_new(
                f64::NAN,
                -10.0,
                -5.0,
                400,
                300,
                120_000,
                300,
                5,
            ),
            Err(LiveOccupancyHostPolicyError::Geometry(_))
        ));
        assert!(matches!(
            LiveOccupancyHostPolicy::try_new(
                0.05, -10.0, -5.0, 400, 300, 120_000, 0, 5,
            ),
            Err(LiveOccupancyHostPolicyError::MaximumKeyframes(_))
        ));
        assert!(matches!(
            LiveOccupancyHostPolicy::try_new(
                0.05, -10.0, -5.0, 400, 300, 120_000, 300, 0,
            ),
            Err(LiveOccupancyHostPolicyError::SnapshotCadence(_))
        ));
    }

    #[test]
    fn partial_navigation_and_actuation_inputs_are_rejected_exactly() {
        for (present, expected, missing_flags) in [
            (
                0b001,
                LiveNavigationBoundaryError::IncompleteNavigation {
                    missing_config: false,
                    missing_dataset: true,
                },
                " --navigation-record",
            ),
            (
                0b010,
                LiveNavigationBoundaryError::IncompleteNavigation {
                    missing_config: true,
                    missing_dataset: true,
                },
                " --navigation-config --navigation-record",
            ),
            (
                0b100,
                LiveNavigationBoundaryError::IncompleteNavigation {
                    missing_config: true,
                    missing_dataset: false,
                },
                " --navigation-config",
            ),
            (
                0b011,
                LiveNavigationBoundaryError::IncompleteNavigation {
                    missing_config: false,
                    missing_dataset: true,
                },
                " --navigation-record",
            ),
            (
                0b110,
                LiveNavigationBoundaryError::IncompleteNavigation {
                    missing_config: true,
                    missing_dataset: false,
                },
                " --navigation-config",
            ),
        ] {
            let actual = LiveNavigationRequest::parse(
                (present & 0b001 != 0).then(|| PathBuf::from("navigation.json")),
                (present & 0b010 != 0).then_some(goal()),
                (present & 0b100 != 0).then(|| PathBuf::from("capture")),
                None,
                None,
            )
            .expect_err("every nonempty partial request must fail");
            assert_eq!(actual, expected, "present mask {present:#05b}");
            assert_eq!(
                actual.to_string(),
                format!(
                    "live navigation requires --navigation-config and --navigation-record together; --navigation-goal is optional; missing{missing_flags}"
                )
            );
        }

        for (config, robot, expected) in [
            (
                Some(PathBuf::from("actuation.json")),
                None,
                LiveNavigationBoundaryError::IncompleteActuation {
                    missing_config: false,
                    missing_robot_id: true,
                },
            ),
            (
                None,
                Some("kiko-01".to_owned()),
                LiveNavigationBoundaryError::IncompleteActuation {
                    missing_config: true,
                    missing_robot_id: false,
                },
            ),
        ] {
            assert_eq!(
                LiveNavigationRequest::parse(
                    Some(PathBuf::from("navigation.json")),
                    Some(goal()),
                    Some(PathBuf::from("capture")),
                    config,
                    robot,
                )
                .expect_err("partial physical authority must fail"),
                expected
            );
        }
        assert_eq!(
            LiveNavigationRequest::parse(
                None,
                None,
                None,
                Some(PathBuf::from("actuation.json")),
                Some("kiko-01".to_owned()),
            )
            .expect_err("actuation without navigation must fail"),
            LiveNavigationBoundaryError::ActuationRequiresNavigation
        );
    }

    #[test]
    fn load_retains_the_exact_bounded_navigation_document() {
        let document = br#"{"schema_version":1,"sentinel":"exact bytes"}"#;
        let config = TemporaryConfig::new(document);
        let request = LiveNavigationRequest::parse(
            Some(config.path()),
            None,
            Some(PathBuf::from("capture")),
            None,
            None,
        )
        .expect("complete mapping request");

        let loaded = request.load().expect("bounded config load");
        assert!(loaded.is_enabled());
        assert_eq!(
            loaded.navigation_config_bytes.as_deref(),
            Some(document.as_slice())
        );
        assert_eq!(loaded.actuation_config_bytes, None);
        assert!(loaded.occupancy_host_policy.is_some());
    }

    #[test]
    fn load_reads_only_one_byte_beyond_the_navigation_bound() {
        let maximum = crate::navigation::ShadowNavigationConfigV1::MAX_JSON_BYTES;
        let config = TemporaryConfig::new(&vec![b' '; maximum + 17]);
        let request = LiveNavigationRequest::parse(
            Some(config.path()),
            None,
            Some(PathBuf::from("capture")),
            None,
            None,
        )
        .expect("complete mapping request");

        assert!(matches!(
            request.load(),
            Err(LiveNavigationLoadError::Config(
                LiveNavigationConfigReadError::InputTooLarge {
                    actual_bytes_at_least,
                    maximum_bytes,
                    ..
                }
            )) if actual_bytes_at_least == maximum + 1 && maximum_bytes == maximum
        ));
    }

    #[test]
    fn prerequisites_are_ignored_when_navigation_is_disabled() {
        let prerequisites = LiveNavigationPrerequisites::new(false, false, false, false);
        assert_eq!(
            prerequisites.require_for(&LiveNavigationRequest::Disabled),
            Ok(())
        );
    }

    #[test]
    fn prerequisites_fail_in_authoritative_startup_order() {
        let request = LiveNavigationRequest::parse(
            Some(PathBuf::from("navigation.json")),
            None,
            Some(PathBuf::from("capture")),
            None,
            None,
        )
        .expect("mapping request");
        for (facts, expected) in [
            (
                LiveNavigationPrerequisites::new(false, false, false, false),
                LiveNavigationPrerequisiteError::DepthDisabled,
            ),
            (
                LiveNavigationPrerequisites::new(true, false, false, false),
                LiveNavigationPrerequisiteError::ImuDisabled,
            ),
            (
                LiveNavigationPrerequisites::new(true, true, false, false),
                LiveNavigationPrerequisiteError::DenseOccupancyDisabled,
            ),
            (
                LiveNavigationPrerequisites::new(true, true, true, false),
                LiveNavigationPrerequisiteError::UnrectifiedStereo,
            ),
        ] {
            assert_eq!(facts.require_for(&request), Err(expected));
        }
        assert_eq!(
            LiveNavigationPrerequisites::new(true, true, true, true).require_for(&request),
            Ok(())
        );
    }
}
