use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;

use crate::{
    Keypoint, Observation, ObservationError, PinholeIntrinsics, Point3, Pose,
    map::{KeyframeId, KeyframeKeypoint, MapError, MapPointId, SlamMap},
    math,
};

/// Maximum SE3 parameter step for convergence detection.
const STEP_CONVERGENCE_THRESHOLD: f32 = 1e-4;
/// Relative cost tolerance for LM convergence.
const RELATIVE_COST_TOLERANCE: f64 = 1e-6;
/// Floor for cost magnitude to avoid division-near-zero in relative convergence.
const COST_FLOOR: f64 = 1e-12;
/// Minimum camera-frame depth in metres for valid reprojection.
const MIN_PROJECTION_DEPTH_M: f32 = 1e-6;
/// Maximum relative deviation accepted from the ordinary f32 summation path.
/// This leaves compatibility headroom above well-conditioned affine-row roundoff while routing
/// materially degraded contributions through the coherent f64 factor path.
const F32_SUM_MAX_RELATIVE_ERROR: f64 = 128.0 * f32::EPSILON as f64;
/// Maximum fraction of a smaller summand that ordinary f32 addition may lose.
const F32_SUM_MAX_RELATIVE_CONTRIBUTION_LOSS: f64 = 1.0e-3;
/// Minimum landmark Schur complement damping.
const MIN_LANDMARK_DAMPING: f32 = 1e-6;
/// Minimum pose damping floor for simple BA (`optimize()`).
const MIN_POSE_DAMPING: f32 = 1e-9;
/// Pivot magnitude threshold below which the linear system is treated as singular.
const PIVOT_TOLERANCE: f32 = 1e-9;
/// Elimination factor threshold below which row reduction is skipped.
const ELIMINATION_TOLERANCE: f32 = 1e-12;
/// Minimum determinant magnitude for 3x3 matrix inversion.
const MIN_DETERMINANT: f64 = 1e-18;
/// Minimum number of poses required for bundle adjustment.
const MIN_BA_POSES: usize = 2;
/// Minimum observations per landmark for inclusion in full BA.
const MIN_LANDMARK_OBSERVATIONS: usize = 2;
/// Absolute minimum observation count for BA config (PnP geometric minimum).
const ABSOLUTE_MIN_OBSERVATIONS: usize = 4;
/// Minimum valid camera-to-landmark distance for the metric scale anchor.
const MIN_SCALE_ANCHOR_DISTANCE_M: f32 = 1e-6;

#[derive(Clone, Copy, Debug)]
pub struct LocalBaConfig {
    window: NonZeroUsize,
    pose_dimension: usize,
    normal_matrix_len: usize,
    max_iterations: NonZeroUsize,
    min_observations: NonZeroUsize,
    huber_delta_px: f32,
    lm: LmConfig,
}

#[derive(Clone, Copy, Debug)]
pub struct LmConfig {
    initial_lambda: f32,
    lambda_factor: f32,
    min_lambda: f32,
    max_lambda: f32,
    rho_accept: f32,
    rho_good: f32,
}

#[derive(Debug)]
pub enum LmConfigError {
    NonPositiveInitialLambda { value: f32 },
    NonPositiveLambdaFactor { value: f32 },
    LambdaFactorTooSmall { value: f32 },
    NonPositiveMinLambda { value: f32 },
    NonPositiveMaxLambda { value: f32 },
    MinLambdaExceedsMax { min: f32, max: f32 },
    InvalidRhoAccept { value: f32 },
    InvalidRhoGood { value: f32 },
    InvalidRhoOrdering { rho_accept: f32, rho_good: f32 },
}

impl std::fmt::Display for LmConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LmConfigError::NonPositiveInitialLambda { value } => {
                write!(f, "LM initial lambda must be > 0 (got {value})")
            }
            LmConfigError::NonPositiveLambdaFactor { value } => {
                write!(f, "LM lambda factor must be > 0 (got {value})")
            }
            LmConfigError::LambdaFactorTooSmall { value } => {
                write!(f, "LM lambda factor must be > 1 (got {value})")
            }
            LmConfigError::NonPositiveMinLambda { value } => {
                write!(f, "LM min lambda must be > 0 (got {value})")
            }
            LmConfigError::NonPositiveMaxLambda { value } => {
                write!(f, "LM max lambda must be > 0 (got {value})")
            }
            LmConfigError::MinLambdaExceedsMax { min, max } => {
                write!(
                    f,
                    "LM min lambda must be <= max lambda (min={min}, max={max})"
                )
            }
            LmConfigError::InvalidRhoAccept { value } => {
                write!(f, "LM rho_accept must be in (0, 1) (got {value})")
            }
            LmConfigError::InvalidRhoGood { value } => {
                write!(f, "LM rho_good must be in (0, 1) (got {value})")
            }
            LmConfigError::InvalidRhoOrdering {
                rho_accept,
                rho_good,
            } => write!(
                f,
                "LM requires rho_accept < rho_good (rho_accept={rho_accept}, rho_good={rho_good})"
            ),
        }
    }
}

impl std::error::Error for LmConfigError {}

impl LmConfig {
    pub fn new(
        initial_lambda: f32,
        lambda_factor: f32,
        min_lambda: f32,
        max_lambda: f32,
        rho_accept: f32,
        rho_good: f32,
    ) -> Result<Self, LmConfigError> {
        if !initial_lambda.is_finite() || initial_lambda <= 0.0 {
            return Err(LmConfigError::NonPositiveInitialLambda {
                value: initial_lambda,
            });
        }
        if !lambda_factor.is_finite() || lambda_factor <= 0.0 {
            return Err(LmConfigError::NonPositiveLambdaFactor {
                value: lambda_factor,
            });
        }
        if lambda_factor <= 1.0 {
            return Err(LmConfigError::LambdaFactorTooSmall {
                value: lambda_factor,
            });
        }
        if !min_lambda.is_finite() || min_lambda <= 0.0 {
            return Err(LmConfigError::NonPositiveMinLambda { value: min_lambda });
        }
        if !max_lambda.is_finite() || max_lambda <= 0.0 {
            return Err(LmConfigError::NonPositiveMaxLambda { value: max_lambda });
        }
        if min_lambda > max_lambda {
            return Err(LmConfigError::MinLambdaExceedsMax {
                min: min_lambda,
                max: max_lambda,
            });
        }
        if !rho_accept.is_finite() || rho_accept <= 0.0 || rho_accept >= 1.0 {
            return Err(LmConfigError::InvalidRhoAccept { value: rho_accept });
        }
        if !rho_good.is_finite() || rho_good <= 0.0 || rho_good >= 1.0 {
            return Err(LmConfigError::InvalidRhoGood { value: rho_good });
        }
        if rho_accept >= rho_good {
            return Err(LmConfigError::InvalidRhoOrdering {
                rho_accept,
                rho_good,
            });
        }
        Ok(Self {
            initial_lambda,
            lambda_factor,
            min_lambda,
            max_lambda,
            rho_accept,
            rho_good,
        })
    }

    pub fn initial_lambda(self) -> f32 {
        self.initial_lambda
    }

    pub fn lambda_factor(self) -> f32 {
        self.lambda_factor
    }

    pub fn min_lambda(self) -> f32 {
        self.min_lambda
    }

    pub fn max_lambda(self) -> f32 {
        self.max_lambda
    }

    pub fn rho_accept(self) -> f32 {
        self.rho_accept
    }

    pub fn rho_good(self) -> f32 {
        self.rho_good
    }
}

impl Default for LmConfig {
    fn default() -> Self {
        Self {
            initial_lambda: 1e-4,
            lambda_factor: 10.0,
            min_lambda: 1e-8,
            max_lambda: 1e4,
            rho_accept: 0.25,
            rho_good: 0.75,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LmAction {
    Accept,
    Reject,
}

#[derive(Clone, Copy, Debug)]
struct LmState {
    lambda: f32,
    prev_cost: f64,
}

impl LmState {
    fn new(config: LmConfig, prev_cost: f64) -> Self {
        Self {
            lambda: config.initial_lambda(),
            prev_cost,
        }
    }

    fn lambda(self) -> f32 {
        self.lambda
    }

    fn prev_cost(self) -> f64 {
        self.prev_cost
    }

    fn step(&mut self, cost: f64, predicted_decrease: f64, config: LmConfig) -> LmAction {
        if !cost.is_finite() || !predicted_decrease.is_finite() || predicted_decrease <= 0.0 {
            self.lambda = (self.lambda * config.lambda_factor()).min(config.max_lambda());
            return LmAction::Reject;
        }

        let rho = (self.prev_cost - cost) / predicted_decrease;
        if rho >= config.rho_accept() as f64 {
            self.prev_cost = cost;
            if rho > config.rho_good() as f64 {
                self.lambda = (self.lambda / config.lambda_factor()).max(config.min_lambda());
            }
            LmAction::Accept
        } else {
            self.lambda = (self.lambda * config.lambda_factor()).min(config.max_lambda());
            LmAction::Reject
        }
    }
}

#[derive(Debug)]
pub enum LocalBaConfigError {
    ZeroWindow,
    WindowTooLarge { value: usize },
    ZeroIterations,
    ZeroObservations,
    TooFewObservations { min: usize },
    NonPositiveHuber { value: f32 },
}

impl std::fmt::Display for LocalBaConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LocalBaConfigError::ZeroWindow => write!(f, "local BA window must be > 0"),
            LocalBaConfigError::WindowTooLarge { value } => write!(
                f,
                "local BA window {value} is too large for a 6-DoF normal matrix"
            ),
            LocalBaConfigError::ZeroIterations => write!(f, "local BA iterations must be > 0"),
            LocalBaConfigError::ZeroObservations => {
                write!(f, "local BA min observations must be > 0")
            }
            LocalBaConfigError::TooFewObservations { min } => {
                write!(f, "local BA min observations must be >= {min}")
            }
            LocalBaConfigError::NonPositiveHuber { value } => {
                write!(f, "local BA huber delta must be > 0 (got {value})")
            }
        }
    }
}

impl std::error::Error for LocalBaConfigError {}

impl LocalBaConfig {
    pub fn new(
        window: usize,
        max_iterations: usize,
        min_observations: usize,
        huber_delta_px: f32,
        lm: LmConfig,
    ) -> Result<Self, LocalBaConfigError> {
        let window = NonZeroUsize::new(window).ok_or(LocalBaConfigError::ZeroWindow)?;
        let pose_dimension =
            window
                .get()
                .checked_mul(6)
                .ok_or(LocalBaConfigError::WindowTooLarge {
                    value: window.get(),
                })?;
        let normal_matrix_len = pose_dimension.checked_mul(pose_dimension).ok_or(
            LocalBaConfigError::WindowTooLarge {
                value: window.get(),
            },
        )?;
        let max_iterations =
            NonZeroUsize::new(max_iterations).ok_or(LocalBaConfigError::ZeroIterations)?;
        let min_observations =
            NonZeroUsize::new(min_observations).ok_or(LocalBaConfigError::ZeroObservations)?;
        if min_observations.get() < ABSOLUTE_MIN_OBSERVATIONS {
            return Err(LocalBaConfigError::TooFewObservations {
                min: ABSOLUTE_MIN_OBSERVATIONS,
            });
        }
        if huber_delta_px <= 0.0 || !huber_delta_px.is_finite() {
            return Err(LocalBaConfigError::NonPositiveHuber {
                value: huber_delta_px,
            });
        }
        Ok(Self {
            window,
            pose_dimension,
            normal_matrix_len,
            max_iterations,
            min_observations,
            huber_delta_px,
            lm,
        })
    }

    pub fn window(&self) -> usize {
        self.window.get()
    }

    fn pose_dimension(&self) -> usize {
        self.pose_dimension
    }

    fn normal_matrix_len(&self) -> usize {
        self.normal_matrix_len
    }

    pub fn max_iterations(&self) -> usize {
        self.max_iterations.get()
    }

    pub fn min_observations(&self) -> usize {
        self.min_observations.get()
    }

    pub fn huber_delta_px(&self) -> f32 {
        self.huber_delta_px
    }

    pub fn lm(&self) -> LmConfig {
        self.lm
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum BaResult {
    Converged { iterations: usize, final_cost: f64 },
    MaxIterations { iterations: usize, final_cost: f64 },
    Degenerate { reason: DegenerateReason },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DegenerateReason {
    TooFewPoses { count: usize },
    TooFewLandmarks { count: usize },
    NoFactors,
    InvalidProjection,
    NumericalFailure,
    InvariantViolation,
}

#[derive(Debug)]
pub enum ObservationSetError {
    TooFew { required: usize, actual: usize },
}

impl std::fmt::Display for ObservationSetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ObservationSetError::TooFew { required, actual } => write!(
                f,
                "observation set requires at least {required} points, got {actual}"
            ),
        }
    }
}

impl std::error::Error for ObservationSetError {}

/// Exact failure returned by the incremental bundle-adjustment boundary.
#[derive(Debug)]
pub enum LocalBaError {
    ObservationSet(ObservationSetError),
    Map(MapError),
    Observation(ObservationError),
    Pose(crate::PoseError),
    MissingMapPointAssociation {
        keyframe_id: KeyframeId,
        keypoint_index: usize,
    },
    InsufficientUsableObservations {
        required: usize,
        actual: usize,
    },
    NumericalFailure {
        operation: &'static str,
    },
    EmptyOptimizedWindow,
}

impl std::fmt::Display for LocalBaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ObservationSet(err) => write!(f, "invalid BA observation set: {err}"),
            Self::Map(err) => write!(f, "BA map access failed: {err}"),
            Self::Observation(err) => write!(f, "BA observation parsing failed: {err}"),
            Self::Pose(err) => write!(f, "BA pose update failed: {err}"),
            Self::MissingMapPointAssociation {
                keyframe_id,
                keypoint_index,
            } => write!(
                f,
                "BA keyframe {keyframe_id:?} keypoint {keypoint_index} has no map-point association"
            ),
            Self::InsufficientUsableObservations { required, actual } => write!(
                f,
                "BA requires {required} usable projected observations, got {actual}"
            ),
            Self::NumericalFailure { operation } => {
                write!(f, "BA numerical failure while {operation}")
            }
            Self::EmptyOptimizedWindow => write!(f, "BA optimized frame window is empty"),
        }
    }
}

impl std::error::Error for LocalBaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ObservationSet(err) => Some(err),
            Self::Map(err) => Some(err),
            Self::Observation(err) => Some(err),
            Self::Pose(err) => Some(err),
            Self::MissingMapPointAssociation { .. }
            | Self::InsufficientUsableObservations { .. }
            | Self::NumericalFailure { .. }
            | Self::EmptyOptimizedWindow => None,
        }
    }
}

impl From<ObservationSetError> for LocalBaError {
    fn from(err: ObservationSetError) -> Self {
        Self::ObservationSet(err)
    }
}

impl From<MapError> for LocalBaError {
    fn from(err: MapError) -> Self {
        Self::Map(err)
    }
}

impl From<ObservationError> for LocalBaError {
    fn from(err: ObservationError) -> Self {
        Self::Observation(err)
    }
}

impl From<crate::PoseError> for LocalBaError {
    fn from(err: crate::PoseError) -> Self {
        Self::Pose(err)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MapObservation {
    keyframe_keypoint: KeyframeKeypoint,
    pixel: Keypoint,
}

impl MapObservation {
    pub fn new(keyframe_keypoint: KeyframeKeypoint, pixel: Keypoint) -> Self {
        Self {
            keyframe_keypoint,
            pixel,
        }
    }

    pub fn keyframe_keypoint(&self) -> KeyframeKeypoint {
        self.keyframe_keypoint
    }

    pub fn pixel(&self) -> Keypoint {
        self.pixel
    }
}

#[derive(Debug)]
pub struct ObservationSet {
    observations: Vec<MapObservation>,
}

impl ObservationSet {
    pub fn new(
        observations: Vec<MapObservation>,
        min_required: NonZeroUsize,
    ) -> Result<Self, ObservationSetError> {
        if observations.len() < min_required.get() {
            return Err(ObservationSetError::TooFew {
                required: min_required.get(),
                actual: observations.len(),
            });
        }
        Ok(Self { observations })
    }

    pub fn observations(&self) -> &[MapObservation] {
        &self.observations
    }

    fn resolve(
        &self,
        map: &SlamMap,
        min_required: NonZeroUsize,
    ) -> Result<ResolvedObservationSet, LocalBaError> {
        let mut resolved = Vec::with_capacity(self.observations.len());
        for obs in &self.observations {
            let keypoint_ref = obs.keyframe_keypoint();
            let point_id = map.map_point_for_keypoint(keypoint_ref)?.ok_or(
                LocalBaError::MissingMapPointAssociation {
                    keyframe_id: keypoint_ref.keyframe_id(),
                    keypoint_index: keypoint_ref.index(),
                },
            )?;
            let world = map
                .point(point_id)
                .ok_or(MapError::MapPointNotFound(point_id))?
                .position();
            resolved.push(Observation::try_new(world, obs.pixel())?);
        }
        if resolved.len() < min_required.get() {
            return Err(LocalBaError::InsufficientUsableObservations {
                required: min_required.get(),
                actual: resolved.len(),
            });
        }
        Ok(ResolvedObservationSet {
            observations: resolved,
        })
    }
}

#[derive(Debug)]
struct ResolvedObservationSet {
    observations: Vec<Observation>,
}

impl ResolvedObservationSet {
    fn observations(&self) -> &[Observation] {
        &self.observations
    }
}

#[derive(Debug)]
struct BaFrame {
    pose: Pose,
    observations: ObservationSet,
}

#[derive(Debug)]
enum FullBaBuildError {
    EmptyWindow,
    DuplicateKeyframe {
        keyframe_id: KeyframeId,
    },
    MissingKeyframe {
        keyframe_id: KeyframeId,
    },
    TooFewKeyframes {
        required: usize,
        actual: usize,
    },
    DuplicateLandmarkObservation {
        point_id: MapPointId,
        keyframe_id: KeyframeId,
    },
    InvalidLandmarkObservation {
        point_id: MapPointId,
        keyframe_id: KeyframeId,
        keypoint_index: usize,
    },
    InvalidScaleAnchor,
    NoLandmarks,
    PoseHasTooFewObservations {
        keyframe_id: KeyframeId,
        required: usize,
        actual: usize,
    },
    Pose(crate::PoseError),
}

impl std::fmt::Display for FullBaBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FullBaBuildError::EmptyWindow => write!(f, "full BA window is empty"),
            FullBaBuildError::DuplicateKeyframe { keyframe_id } => {
                write!(f, "full BA window has duplicate keyframe {keyframe_id:?}")
            }
            FullBaBuildError::MissingKeyframe { keyframe_id } => {
                write!(
                    f,
                    "full BA window references missing keyframe {keyframe_id:?}"
                )
            }
            FullBaBuildError::TooFewKeyframes { required, actual } => {
                write!(
                    f,
                    "full BA requires at least {required} keyframes, got {actual}"
                )
            }
            FullBaBuildError::DuplicateLandmarkObservation {
                point_id,
                keyframe_id,
            } => write!(
                f,
                "landmark {point_id:?} has duplicate observation in keyframe {keyframe_id:?}"
            ),
            FullBaBuildError::InvalidLandmarkObservation {
                point_id,
                keyframe_id,
                keypoint_index,
            } => write!(
                f,
                "landmark {point_id:?} has invalid observation at keyframe {keyframe_id:?} keypoint {keypoint_index}"
            ),
            FullBaBuildError::InvalidScaleAnchor => {
                write!(
                    f,
                    "full BA could not construct a finite metric scale anchor"
                )
            }
            FullBaBuildError::NoLandmarks => {
                write!(f, "full BA window has no optimizable landmarks")
            }
            FullBaBuildError::PoseHasTooFewObservations {
                keyframe_id,
                required,
                actual,
            } => write!(
                f,
                "keyframe {keyframe_id:?} has too few BA observations: required={required}, actual={actual}"
            ),
            FullBaBuildError::Pose(err) => write!(f, "full BA pose geometry failed: {err}"),
        }
    }
}

impl std::error::Error for FullBaBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Pose(err) => Some(err),
            _ => None,
        }
    }
}

impl From<crate::PoseError> for FullBaBuildError {
    fn from(err: crate::PoseError) -> Self {
        Self::Pose(err)
    }
}

fn degenerate_reason_from_build_error(err: &FullBaBuildError) -> DegenerateReason {
    match err {
        FullBaBuildError::EmptyWindow => DegenerateReason::TooFewPoses { count: 0 },
        FullBaBuildError::TooFewKeyframes { actual, .. } => {
            DegenerateReason::TooFewPoses { count: *actual }
        }
        FullBaBuildError::NoLandmarks => DegenerateReason::TooFewLandmarks { count: 0 },
        FullBaBuildError::DuplicateKeyframe { .. }
        | FullBaBuildError::MissingKeyframe { .. }
        | FullBaBuildError::DuplicateLandmarkObservation { .. }
        | FullBaBuildError::InvalidLandmarkObservation { .. }
        | FullBaBuildError::InvalidScaleAnchor
        | FullBaBuildError::Pose(_) => DegenerateReason::InvariantViolation,
        FullBaBuildError::PoseHasTooFewObservations { .. } => DegenerateReason::NoFactors,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PoseVarIndex(usize);

impl PoseVarIndex {
    fn as_usize(self) -> usize {
        self.0
    }

    fn offset6(self) -> usize {
        self.0 * 6
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct LandmarkVarIndex(usize);

impl LandmarkVarIndex {
    fn as_usize(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
struct PoseVariable {
    keyframe_id: KeyframeId,
    pose: Pose,
}

#[derive(Clone, Copy, Debug)]
struct LandmarkVariable {
    point_id: MapPointId,
    position: Point3,
}

#[derive(Clone, Copy, Debug)]
enum FactorPose {
    Variable(PoseVarIndex),
    Fixed(Pose),
}

#[derive(Clone, Copy, Debug)]
struct ReprojectionFactor {
    pose: FactorPose,
    landmark: LandmarkVarIndex,
    pixel: Keypoint,
}

#[derive(Clone, Copy, Debug)]
struct ScaleAnchor {
    landmark: LandmarkVarIndex,
    camera_center: Point3,
    reference_distance_m: f64,
}

#[derive(Debug)]
struct FullBaProblem {
    poses: Vec<PoseVariable>,
    landmarks: Vec<LandmarkVariable>,
    factors: Vec<ReprojectionFactor>,
    scale_anchor: Option<ScaleAnchor>,
}

impl FullBaProblem {
    fn try_from_map(
        map: &SlamMap,
        requested_window: &[KeyframeId],
        max_window: NonZeroUsize,
        min_observations: NonZeroUsize,
    ) -> Result<Self, FullBaBuildError> {
        if requested_window.is_empty() {
            return Err(FullBaBuildError::EmptyWindow);
        }

        let mut poses = Vec::new();
        let mut seen_keyframes = HashSet::new();
        for &keyframe_id in requested_window.iter().take(max_window.get()) {
            if !seen_keyframes.insert(keyframe_id) {
                return Err(FullBaBuildError::DuplicateKeyframe { keyframe_id });
            }
            let entry = map
                .keyframe(keyframe_id)
                .ok_or(FullBaBuildError::MissingKeyframe { keyframe_id })?;
            poses.push(PoseVariable {
                keyframe_id,
                pose: entry.pose().into_legacy_pose(),
            });
        }

        if poses.len() < MIN_BA_POSES {
            return Err(FullBaBuildError::TooFewKeyframes {
                required: MIN_BA_POSES,
                actual: poses.len(),
            });
        }

        let mut pose_lookup = HashMap::new();
        for (idx, pose) in poses.iter().enumerate() {
            pose_lookup.insert(pose.keyframe_id, PoseVarIndex(idx));
        }

        let mut landmarks = Vec::new();
        let mut factors = Vec::new();
        let mut pose_counts = vec![0_usize; poses.len()];

        for (point_id, point) in map.points() {
            let mut observations = Vec::new();
            let mut seen_observation_poses = HashSet::new();
            let mut has_local_observation = false;

            for &obs in point.observations() {
                if !seen_observation_poses.insert(obs.keyframe_id()) {
                    return Err(FullBaBuildError::DuplicateLandmarkObservation {
                        point_id,
                        keyframe_id: obs.keyframe_id(),
                    });
                }
                let pixel = map.keypoint(obs).map_err(|_| {
                    FullBaBuildError::InvalidLandmarkObservation {
                        point_id,
                        keyframe_id: obs.keyframe_id(),
                        keypoint_index: obs.index(),
                    }
                })?;
                let factor_pose = if let Some(&pose_idx) = pose_lookup.get(&obs.keyframe_id()) {
                    has_local_observation = true;
                    FactorPose::Variable(pose_idx)
                } else {
                    let entry = map.keyframe(obs.keyframe_id()).ok_or(
                        FullBaBuildError::MissingKeyframe {
                            keyframe_id: obs.keyframe_id(),
                        },
                    )?;
                    FactorPose::Fixed(entry.pose().into_legacy_pose())
                };
                observations.push((factor_pose, pixel));
            }

            if !has_local_observation || observations.len() < MIN_LANDMARK_OBSERVATIONS {
                continue;
            }

            let landmark_idx = LandmarkVarIndex(landmarks.len());
            landmarks.push(LandmarkVariable {
                point_id,
                position: point.position(),
            });

            for (factor_pose, pixel) in observations {
                if let FactorPose::Variable(pose_idx) = factor_pose {
                    pose_counts[pose_idx.as_usize()] += 1;
                }
                factors.push(ReprojectionFactor {
                    pose: factor_pose,
                    landmark: landmark_idx,
                    pixel,
                });
            }
        }

        if landmarks.is_empty() {
            return Err(FullBaBuildError::NoLandmarks);
        }

        for (idx, pose) in poses.iter().enumerate() {
            if pose_counts[idx] < min_observations.get() {
                return Err(FullBaBuildError::PoseHasTooFewObservations {
                    keyframe_id: pose.keyframe_id,
                    required: min_observations.get(),
                    actual: pose_counts[idx],
                });
            }
        }

        let camera_center = camera_center_world(poses[0].pose)?;
        let anchor_position = landmarks[0].position;
        let reference_distance_m = point_distance(anchor_position, camera_center);
        if !reference_distance_m.is_finite()
            || reference_distance_m <= f64::from(MIN_SCALE_ANCHOR_DISTANCE_M)
        {
            return Err(FullBaBuildError::InvalidScaleAnchor);
        }

        Ok(Self {
            poses,
            landmarks,
            factors,
            scale_anchor: Some(ScaleAnchor {
                landmark: LandmarkVarIndex(0),
                camera_center,
                reference_distance_m,
            }),
        })
    }

    fn write_back(self, map: &mut SlamMap) -> Result<(), MapError> {
        for pose in &self.poses {
            map.set_keyframe_pose(
                pose.keyframe_id,
                crate::WorldToCamera::from_legacy_pose(pose.pose),
            )?;
        }
        for landmark in &self.landmarks {
            map.set_map_point_position(landmark.point_id, landmark.position)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct PoseLandmarkCross {
    pose: PoseVarIndex,
    b: [[f32; 3]; 6],
}

#[derive(Debug, Default)]
struct LandmarkAccumulator {
    c: [[f32; 3]; 3],
    b: [f32; 3],
    links: Vec<PoseLandmarkCross>,
}

impl LandmarkAccumulator {
    fn add_link(&mut self, pose: PoseVarIndex, cross: [[f32; 3]; 6]) {
        if let Some(existing) = self.links.iter_mut().find(|link| link.pose == pose) {
            for (row, cross_row) in cross.iter().enumerate() {
                for (col, value) in cross_row.iter().enumerate() {
                    existing.b[row][col] += *value;
                }
            }
            return;
        }
        self.links.push(PoseLandmarkCross { pose, b: cross });
    }
}

#[derive(Debug)]
struct LandmarkSchur {
    inv_c: [[f32; 3]; 3],
    b: [f32; 3],
    links: Vec<PoseLandmarkCross>,
}

#[derive(Debug)]
pub struct LocalBundleAdjuster {
    config: LocalBaConfig,
    intrinsics: PinholeIntrinsics,
    frames: Vec<BaFrame>,
    a_buf: Vec<f32>,
    b_buf: Vec<f32>,
}

impl LocalBundleAdjuster {
    pub fn new(intrinsics: PinholeIntrinsics, config: LocalBaConfig) -> Self {
        let a_buf = vec![0.0_f32; config.normal_matrix_len()];
        let b_buf = vec![0.0_f32; config.pose_dimension()];
        Self {
            config,
            intrinsics,
            frames: Vec::new(),
            a_buf,
            b_buf,
        }
    }

    pub fn reset(&mut self) {
        self.frames.clear();
    }

    pub fn min_observations(&self) -> NonZeroUsize {
        self.config.min_observations
    }

    pub fn window_size(&self) -> NonZeroUsize {
        self.config.window
    }

    pub fn push_frame(
        &mut self,
        map: &SlamMap,
        pose: Pose,
        observations: ObservationSet,
    ) -> Result<Pose, LocalBaError> {
        self.frames.push(BaFrame { pose, observations });
        if self.frames.len() > self.config.window() {
            let excess = self.frames.len() - self.config.window();
            self.frames.drain(0..excess);
        }

        if let Err(err) = self.optimize(map) {
            self.reset();
            return Err(err);
        }
        self.frames
            .last()
            .map(|frame| frame.pose)
            .ok_or(LocalBaError::EmptyOptimizedWindow)
    }

    pub fn optimize_keyframe_window(
        &mut self,
        map: &mut SlamMap,
        window: &[KeyframeId],
    ) -> Result<BaResult, LocalBaError> {
        let mut problem = match FullBaProblem::try_from_map(
            map,
            window,
            self.config.window,
            self.config.min_observations,
        ) {
            Ok(problem) => problem,
            Err(FullBaBuildError::Pose(err)) => return Err(err.into()),
            Err(err) => {
                return Ok(BaResult::Degenerate {
                    reason: degenerate_reason_from_build_error(&err),
                });
            }
        };

        let result = self.optimize_full(&mut problem)?;
        if matches!(
            result,
            BaResult::Converged { .. } | BaResult::MaxIterations { .. }
        ) {
            let mut staged_map = map.clone();
            problem.write_back(&mut staged_map)?;
            *map = staged_map;
        }

        Ok(result)
    }

    fn optimize(&mut self, map: &SlamMap) -> Result<(), LocalBaError> {
        let frame_count = self.frames.len();
        if frame_count == 0 {
            return Err(LocalBaError::EmptyOptimizedWindow);
        }

        // The map is immutable for this complete optimization call. Resolve
        // map associations and parse finite correspondences once, rather than
        // repeating the same lookups and allocations at every iteration and
        // again during final-state validation.
        let resolved_frames: Vec<_> = self
            .frames
            .iter()
            .map(|frame| {
                frame
                    .observations
                    .resolve(map, self.config.min_observations)
            })
            .collect::<Result<_, _>>()?;

        let dim = frame_count * 6;
        let max_iters = self.config.max_iterations();
        let huber = self.config.huber_delta_px();
        let damping = self.config.lm().initial_lambda().max(MIN_POSE_DAMPING);

        for _ in 0..max_iters {
            let a = &mut self.a_buf[..dim * dim];
            let b = &mut self.b_buf[..dim];
            a.fill(0.0);
            b.fill(0.0);

            for (idx, (frame, resolved)) in self.frames.iter().zip(&resolved_frames).enumerate() {
                let base = idx * 6;
                let mut usable_observations = 0_usize;
                for observation in resolved.observations() {
                    let linearization = match reprojection_linearization(
                        frame.pose,
                        observation.world(),
                        observation.pixel(),
                        self.intrinsics,
                    ) {
                        Ok(linearization) => linearization,
                        Err(ProjectionLinearizationError::InvalidDepth) => continue,
                        Err(ProjectionLinearizationError::NumericalFailure) => {
                            return Err(LocalBaError::NumericalFailure {
                                operation: "linearizing an incremental reprojection factor",
                            });
                        }
                    };

                    usable_observations += 1;
                    let residual = linearization.residual;
                    let scale = linearization.huber_sqrt_weight(huber);
                    let r0 = residual[0] * scale;
                    let r1 = residual[1] * scale;
                    let j = linearization
                        .pose_jacobian
                        .map(|row| row.map(|value| value * scale));

                    for c in 0..6 {
                        let jr = j[0][c] * r0 + j[1][c] * r1;
                        b[base + c] -= jr;
                        for d in 0..6 {
                            let jt_j = j[0][c] * j[0][d] + j[1][c] * j[1][d];
                            a[(base + c) * dim + (base + d)] += jt_j;
                        }
                    }
                }
                if usable_observations < self.config.min_observations() {
                    return Err(LocalBaError::InsufficientUsableObservations {
                        required: self.config.min_observations(),
                        actual: usable_observations,
                    });
                }
            }

            for i in 0..dim {
                a[i * dim + i] += damping;
            }

            if !solve_linear_system(a, b, dim) {
                return Err(LocalBaError::NumericalFailure {
                    operation: "solving the incremental normal equations",
                });
            }

            let mut max_step = 0.0_f64;
            for i in 0..frame_count {
                let step = extract_se3_delta(b, i * 6);
                let Some(step_norm) = finite_norm(step) else {
                    return Err(LocalBaError::NumericalFailure {
                        operation: "computing the incremental pose step norm",
                    });
                };
                if step_norm > max_step {
                    max_step = step_norm;
                }
                let pose = self.frames[i].pose;
                let candidate = apply_se3_delta(pose, step)?;
                self.frames[i].pose = candidate;
            }

            if max_step < f64::from(STEP_CONVERGENCE_THRESHOLD) {
                break;
            }
        }

        // The last accepted step has not yet served as a linearization point. Re-check it before
        // exposing the pose so a final update cannot leave the window below its configured factor
        // support or defer a numerical failure until the next frame.
        self.validate_final_incremental_window_resolved(&resolved_frames)
    }

    fn validate_final_incremental_window_resolved(
        &self,
        resolved_frames: &[ResolvedObservationSet],
    ) -> Result<(), LocalBaError> {
        debug_assert_eq!(self.frames.len(), resolved_frames.len());
        for (frame, resolved) in self.frames.iter().zip(resolved_frames) {
            let mut usable_observations = 0_usize;
            for observation in resolved.observations() {
                match reprojection_linearization(
                    frame.pose,
                    observation.world(),
                    observation.pixel(),
                    self.intrinsics,
                ) {
                    Ok(_) => usable_observations += 1,
                    Err(ProjectionLinearizationError::InvalidDepth) => {}
                    Err(ProjectionLinearizationError::NumericalFailure) => {
                        return Err(LocalBaError::NumericalFailure {
                            operation: "validating final incremental reprojection factors",
                        });
                    }
                }
            }
            if usable_observations < self.config.min_observations() {
                return Err(LocalBaError::InsufficientUsableObservations {
                    required: self.config.min_observations(),
                    actual: usable_observations,
                });
            }
        }
        Ok(())
    }

    #[cfg(test)]
    fn validate_final_incremental_window(&self, map: &SlamMap) -> Result<(), LocalBaError> {
        let resolved_frames = self
            .frames
            .iter()
            .map(|frame| {
                frame
                    .observations
                    .resolve(map, self.config.min_observations)
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.validate_final_incremental_window_resolved(&resolved_frames)
    }

    fn optimize_full(&mut self, problem: &mut FullBaProblem) -> Result<BaResult, LocalBaError> {
        let pose_count = problem.poses.len();
        let landmark_count = problem.landmarks.len();
        if pose_count < MIN_BA_POSES {
            return Ok(BaResult::Degenerate {
                reason: DegenerateReason::TooFewPoses { count: pose_count },
            });
        }
        if landmark_count == 0 {
            return Ok(BaResult::Degenerate {
                reason: DegenerateReason::TooFewLandmarks {
                    count: landmark_count,
                },
            });
        }
        if problem.factors.is_empty() {
            return Ok(BaResult::Degenerate {
                reason: DegenerateReason::NoFactors,
            });
        }

        let pose_dim = pose_count * 6;
        let max_iters = self.config.max_iterations();
        let huber = self.config.huber_delta_px();
        let lm_config = self.config.lm();
        let initial_cost = match full_problem_cost(problem, self.intrinsics, huber) {
            Ok(cost) => cost,
            Err(error) => {
                return Ok(BaResult::Degenerate {
                    reason: error.degenerate_reason(),
                });
            }
        };
        let mut lm_state = LmState::new(lm_config, initial_cost);

        let mut pose_backup: Vec<Pose> = Vec::with_capacity(pose_count);
        let mut landmark_backup: Vec<Point3> = Vec::with_capacity(landmark_count);
        let mut landmark_accumulators: Vec<LandmarkAccumulator> =
            Vec::with_capacity(landmark_count);
        let mut full_pose_rhs: Vec<f32> = vec![0.0; pose_dim];

        for iter in 0..max_iters {
            pose_backup.clear();
            pose_backup.extend(problem.poses.iter().map(|pv| pv.pose));
            landmark_backup.clear();
            landmark_backup.extend(problem.landmarks.iter().map(|lv| lv.position));

            let s = &mut self.a_buf[..pose_dim * pose_dim];
            let rhs = &mut self.b_buf[..pose_dim];
            s.fill(0.0);
            rhs.fill(0.0);

            let pose_damping = lm_state.lambda().max(lm_config.min_lambda());
            let landmark_damping = pose_damping.max(MIN_LANDMARK_DAMPING);

            landmark_accumulators.clear();
            landmark_accumulators
                .extend((0..landmark_count).map(|_| LandmarkAccumulator::default()));

            for factor in &problem.factors {
                let landmark_idx = factor.landmark;
                let pose = match factor.pose {
                    FactorPose::Variable(pose_idx) => problem.poses[pose_idx.as_usize()].pose,
                    FactorPose::Fixed(pose) => pose,
                };
                let point = problem.landmarks[landmark_idx.as_usize()].position;

                let linearization =
                    match reprojection_linearization(pose, point, factor.pixel, self.intrinsics) {
                        Ok(linearization) => linearization,
                        Err(error) => {
                            return Ok(BaResult::Degenerate {
                                reason: error.degenerate_reason(),
                            });
                        }
                    };

                let residual = linearization.residual;
                let scale = linearization.huber_sqrt_weight(huber);

                let r_scaled = [residual[0] * scale, residual[1] * scale];
                let j_pose_scaled = linearization
                    .pose_jacobian
                    .map(|row| row.map(|value| value * scale));
                let j_landmark_scaled = linearization
                    .landmark_jacobian
                    .map(|row| row.map(|value| value * scale));

                let acc = &mut landmark_accumulators[landmark_idx.as_usize()];
                accumulate_landmark_hessian(&mut acc.c, j_landmark_scaled);
                accumulate_landmark_rhs(&mut acc.b, j_landmark_scaled, r_scaled);
                if let FactorPose::Variable(pose_idx) = factor.pose {
                    accumulate_pose_hessian(s, pose_dim, pose_idx, j_pose_scaled);
                    accumulate_pose_rhs(rhs, pose_idx, j_pose_scaled, r_scaled);
                    acc.add_link(
                        pose_idx,
                        pose_landmark_cross(j_pose_scaled, j_landmark_scaled),
                    );
                }
            }

            if let Some(anchor) = problem.scale_anchor {
                let point = problem.landmarks[anchor.landmark.as_usize()].position;
                let Some((residual, jacobian)) =
                    scale_anchor_residual_and_jacobian(anchor, point, self.intrinsics)
                else {
                    // Construction validates the anchor, and every accepted candidate passes the
                    // cost boundary below. Failure while assembling an accepted state is numeric.
                    return Ok(BaResult::Degenerate {
                        reason: DegenerateReason::NumericalFailure,
                    });
                };
                let acc = &mut landmark_accumulators[anchor.landmark.as_usize()];
                accumulate_scalar_landmark_factor(&mut acc.c, &mut acc.b, jacobian, residual);
            }

            full_pose_rhs.copy_from_slice(rhs);

            for i in 0..pose_dim {
                s[i * pose_dim + i] += pose_damping;
            }

            let mut schur_landmarks = Vec::with_capacity(landmark_count);
            for acc in landmark_accumulators.drain(..) {
                let mut c = acc.c;
                for (i, c_row) in c.iter_mut().enumerate() {
                    c_row[i] += landmark_damping;
                }
                let Some(inv_c) = invert_3x3(c) else {
                    return Ok(BaResult::Degenerate {
                        reason: DegenerateReason::NumericalFailure,
                    });
                };

                let inv_c_b = math::mat_mul_vec(inv_c, acc.b);
                for link_i in &acc.links {
                    let base_i = link_i.pose.offset6();
                    let rhs_contrib = mat63_mul_vec3(link_i.b, inv_c_b);
                    for row in 0..6 {
                        rhs[base_i + row] -= rhs_contrib[row];
                    }

                    for link_j in &acc.links {
                        let base_j = link_j.pose.offset6();
                        let block = schur_block(link_i.b, inv_c, link_j.b);
                        for row in 0..6 {
                            for col in 0..6 {
                                s[(base_i + row) * pose_dim + (base_j + col)] -= block[row][col];
                            }
                        }
                    }
                }

                schur_landmarks.push(LandmarkSchur {
                    inv_c,
                    b: acc.b,
                    links: acc.links,
                });
            }

            fix_pose_block(s, rhs, pose_dim, PoseVarIndex(0));

            if !solve_linear_system(s, rhs, pose_dim) {
                return Ok(BaResult::Degenerate {
                    reason: DegenerateReason::NumericalFailure,
                });
            }

            let mut predicted_decrease = 0.0_f64;
            let mut max_step = 0.0_f64;
            for (pose_i, pose_var) in problem.poses.iter_mut().enumerate() {
                let base = pose_i * 6;
                let delta = extract_se3_delta(rhs, base);
                let Some(step_norm) = finite_norm(delta) else {
                    return Ok(BaResult::Degenerate {
                        reason: DegenerateReason::NumericalFailure,
                    });
                };
                max_step = max_step.max(step_norm);
                for k in 0..6 {
                    let d = delta[k] as f64;
                    let gradient = full_pose_rhs[base + k] as f64;
                    predicted_decrease += 0.5 * d * ((pose_damping as f64) * d + gradient);
                }
                pose_var.pose = apply_se3_delta(pose_var.pose, delta)?;
            }

            for (landmark_i, landmark_var) in problem.landmarks.iter_mut().enumerate() {
                let schur = &schur_landmarks[landmark_i];
                let mut coupling = [0.0_f32; 3];
                for link in &schur.links {
                    let pose_delta = extract_se3_delta(rhs, link.pose.offset6());
                    for (row, pose_delta_value) in pose_delta.iter().enumerate() {
                        for (col, link_value) in link.b[row].iter().enumerate() {
                            coupling[col] += *link_value * *pose_delta_value;
                        }
                    }
                }

                let rhs_landmark = [
                    schur.b[0] - coupling[0],
                    schur.b[1] - coupling[1],
                    schur.b[2] - coupling[2],
                ];
                let delta_landmark = math::mat_mul_vec(schur.inv_c, rhs_landmark);
                let Some(step_norm) = finite_norm(delta_landmark) else {
                    return Ok(BaResult::Degenerate {
                        reason: DegenerateReason::NumericalFailure,
                    });
                };
                max_step = max_step.max(step_norm);
                for (axis, d) in delta_landmark.iter().enumerate() {
                    let d = *d as f64;
                    let gradient = schur.b[axis] as f64;
                    predicted_decrease += 0.5 * d * ((landmark_damping as f64) * d + gradient);
                }

                landmark_var.position.x += delta_landmark[0];
                landmark_var.position.y += delta_landmark[1];
                landmark_var.position.z += delta_landmark[2];
            }

            let candidate_cost = match full_problem_cost(problem, self.intrinsics, huber) {
                Ok(cost) => cost,
                // Invalid trial states are rejected by LM and restored from the iteration
                // backups below. The accepted state was already parsed by `initial_cost` or
                // the previous successful candidate-cost evaluation.
                Err(ProjectionLinearizationError::InvalidDepth)
                | Err(ProjectionLinearizationError::NumericalFailure) => f64::INFINITY,
            };
            let prev_cost = lm_state.prev_cost();
            if max_step == 0.0 && candidate_cost.is_finite() && candidate_cost == prev_cost {
                return Ok(BaResult::Converged {
                    iterations: iter + 1,
                    final_cost: candidate_cost,
                });
            }
            match lm_state.step(candidate_cost, predicted_decrease, lm_config) {
                LmAction::Accept => {
                    let threshold = RELATIVE_COST_TOLERANCE * prev_cost.abs().max(COST_FLOOR);
                    if max_step < f64::from(STEP_CONVERGENCE_THRESHOLD)
                        || (prev_cost - candidate_cost).abs() <= threshold
                    {
                        return Ok(BaResult::Converged {
                            iterations: iter + 1,
                            final_cost: candidate_cost,
                        });
                    }
                }
                LmAction::Reject => {
                    for (pose_var, pose) in
                        problem.poses.iter_mut().zip(pose_backup.iter().copied())
                    {
                        pose_var.pose = pose;
                    }
                    for (landmark_var, point) in problem
                        .landmarks
                        .iter_mut()
                        .zip(landmark_backup.iter().copied())
                    {
                        landmark_var.position = point;
                    }
                }
            }
        }

        let final_cost = lm_state.prev_cost();
        if !final_cost.is_finite() {
            return Ok(BaResult::Degenerate {
                reason: DegenerateReason::NumericalFailure,
            });
        }
        Ok(BaResult::MaxIterations {
            iterations: max_iters,
            final_cost,
        })
    }
}

fn full_problem_cost(
    problem: &FullBaProblem,
    intrinsics: PinholeIntrinsics,
    huber_delta_px: f32,
) -> Result<f64, ProjectionLinearizationError> {
    let mut cost = 0.0_f64;
    let huber = huber_delta_px as f64;

    for factor in &problem.factors {
        let pose = match factor.pose {
            FactorPose::Variable(pose_idx) => problem.poses[pose_idx.as_usize()].pose,
            FactorPose::Fixed(pose) => pose,
        };
        let point = problem.landmarks[factor.landmark.as_usize()].position;
        let linearization = reprojection_linearization(pose, point, factor.pixel, intrinsics)?;
        let r0 = f64::from(linearization.residual[0]);
        let r1 = f64::from(linearization.residual[1]);
        let r_norm = r0.hypot(r1);
        cost += if r_norm <= huber {
            0.5 * r_norm * r_norm
        } else {
            huber * (r_norm - 0.5 * huber)
        };
    }

    if let Some(anchor) = problem.scale_anchor {
        let point = problem.landmarks[anchor.landmark.as_usize()].position;
        let Some((residual, _)) = scale_anchor_residual_and_jacobian(anchor, point, intrinsics)
        else {
            return Err(ProjectionLinearizationError::NumericalFailure);
        };
        cost += 0.5 * (residual as f64) * (residual as f64);
    }

    cost.is_finite()
        .then_some(cost)
        .ok_or(ProjectionLinearizationError::NumericalFailure)
}

fn camera_center_world(pose: Pose) -> Result<Point3, crate::PoseError> {
    Ok(Point3::from_array(pose.try_inverse()?.translation()))
}

fn point_distance(a: Point3, b: Point3) -> f64 {
    let dx = f64::from(a.x) - f64::from(b.x);
    let dy = f64::from(a.y) - f64::from(b.y);
    let dz = f64::from(a.z) - f64::from(b.z);
    dx.hypot(dy).hypot(dz)
}

fn scale_anchor_residual_and_jacobian(
    anchor: ScaleAnchor,
    point: Point3,
    intrinsics: PinholeIntrinsics,
) -> Option<(f32, [f32; 3])> {
    if !anchor.reference_distance_m.is_finite()
        || anchor.reference_distance_m <= f64::from(MIN_SCALE_ANCHOR_DISTANCE_M)
    {
        return None;
    }
    let delta = [
        f64::from(point.x) - f64::from(anchor.camera_center.x),
        f64::from(point.y) - f64::from(anchor.camera_center.y),
        f64::from(point.z) - f64::from(anchor.camera_center.z),
    ];
    let distance = delta[0].hypot(delta[1]).hypot(delta[2]);
    if !distance.is_finite() || distance <= f64::from(MIN_SCALE_ANCHOR_DISTANCE_M) {
        return None;
    }

    let focal_weight = 0.5 * (f64::from(intrinsics.fx()) + f64::from(intrinsics.fy()));
    let residual =
        narrow_finite_f32(focal_weight * (distance / anchor.reference_distance_m - 1.0))?;
    let jacobian_scale = focal_weight / (anchor.reference_distance_m * distance);
    let mut jacobian = [0.0_f32; 3];
    for (axis, value) in jacobian.iter_mut().enumerate() {
        *value = narrow_finite_f32(delta[axis] * jacobian_scale)?;
    }
    Some((residual, jacobian))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProjectionLinearizationError {
    InvalidDepth,
    NumericalFailure,
}

impl ProjectionLinearizationError {
    fn degenerate_reason(self) -> DegenerateReason {
        match self {
            Self::InvalidDepth => DegenerateReason::InvalidProjection,
            Self::NumericalFailure => DegenerateReason::NumericalFailure,
        }
    }
}

/// A reprojection factor whose residual and both Jacobians are finite `f32` values.
#[derive(Clone, Copy, Debug)]
struct ReprojectionLinearization {
    residual: [f32; 2],
    pose_jacobian: [[f32; 6]; 2],
    landmark_jacobian: [[f32; 3]; 2],
}

impl ReprojectionLinearization {
    fn try_new(
        residual: [f32; 2],
        pose_jacobian: [[f32; 6]; 2],
        landmark_jacobian: [[f32; 3]; 2],
    ) -> Result<Self, ProjectionLinearizationError> {
        if residual
            .iter()
            .chain(pose_jacobian.iter().flatten())
            .chain(landmark_jacobian.iter().flatten())
            .any(|value| !value.is_finite())
        {
            return Err(ProjectionLinearizationError::NumericalFailure);
        }
        Ok(Self {
            residual,
            pose_jacobian,
            landmark_jacobian,
        })
    }

    fn try_from_widened(
        residual: [f64; 2],
        pose_jacobian: [[f64; 6]; 2],
        landmark_jacobian: [[f64; 3]; 2],
    ) -> Result<Self, ProjectionLinearizationError> {
        let mut narrowed_residual = [0.0_f32; 2];
        for (target, source) in narrowed_residual.iter_mut().zip(residual) {
            *target =
                narrow_finite_f32(source).ok_or(ProjectionLinearizationError::NumericalFailure)?;
        }

        let mut narrowed_pose_jacobian = [[0.0_f32; 6]; 2];
        for (target_row, source_row) in narrowed_pose_jacobian.iter_mut().zip(pose_jacobian) {
            for (target, source) in target_row.iter_mut().zip(source_row) {
                *target = narrow_finite_f32(source)
                    .ok_or(ProjectionLinearizationError::NumericalFailure)?;
            }
        }

        let mut narrowed_landmark_jacobian = [[0.0_f32; 3]; 2];
        for (target_row, source_row) in narrowed_landmark_jacobian.iter_mut().zip(landmark_jacobian)
        {
            for (target, source) in target_row.iter_mut().zip(source_row) {
                *target = narrow_finite_f32(source)
                    .ok_or(ProjectionLinearizationError::NumericalFailure)?;
            }
        }

        Ok(Self {
            residual: narrowed_residual,
            pose_jacobian: narrowed_pose_jacobian,
            landmark_jacobian: narrowed_landmark_jacobian,
        })
    }

    fn has_same_values(&self, other: &Self) -> bool {
        self.residual
            .iter()
            .chain(self.pose_jacobian.iter().flatten())
            .chain(self.landmark_jacobian.iter().flatten())
            .zip(
                other
                    .residual
                    .iter()
                    .chain(other.pose_jacobian.iter().flatten())
                    .chain(other.landmark_jacobian.iter().flatten()),
            )
            .all(|(left, right)| left == right)
    }

    #[cfg(test)]
    fn has_same_bits(&self, other: &Self) -> bool {
        self.residual
            .iter()
            .chain(self.pose_jacobian.iter().flatten())
            .chain(self.landmark_jacobian.iter().flatten())
            .zip(
                other
                    .residual
                    .iter()
                    .chain(other.pose_jacobian.iter().flatten())
                    .chain(other.landmark_jacobian.iter().flatten()),
            )
            .all(|(left, right)| left.to_bits() == right.to_bits())
    }

    fn huber_sqrt_weight(&self, delta: f32) -> f32 {
        let r_norm = f64::from(self.residual[0]).hypot(f64::from(self.residual[1]));
        if r_norm <= f64::from(delta) {
            1.0
        } else {
            (f64::from(delta) / r_norm).sqrt() as f32
        }
    }
}

fn reprojection_linearization(
    pose: Pose,
    world: Point3,
    pixel: Keypoint,
    intrinsics: PinholeIntrinsics,
) -> Result<ReprojectionLinearization, ProjectionLinearizationError> {
    let pw = [world.x, world.y, world.z];
    let rotation = pose.rotation();
    let translation = pose.translation();
    let (pc_f32, pc_widened, transform_precision_risk) =
        transform_point_f32_with_risk(rotation, translation, pw);
    let f32_camera_is_valid =
        pc_f32.into_iter().all(f32::is_finite) && pc_f32[2] > MIN_PROJECTION_DEPTH_M;
    let widened_camera_is_valid = pc_widened.into_iter().all(f64::is_finite)
        && pc_widened[2] > f64::from(MIN_PROJECTION_DEPTH_M);
    // Keep ordinary valid f32 transform semantics. Re-evaluate non-finite, invalid-depth, or
    // precision-sensitive results coherently before classifying the factor.
    if !f32_camera_is_valid
        || transform_precision_risk
        || f32_camera_is_valid != widened_camera_is_valid
    {
        return reprojection_linearization_widened(pc_widened, pixel, intrinsics, rotation);
    }
    let pc = pc_f32;
    let x = pc[0];
    let y = pc[1];
    let z = pc[2];

    let normalized_x = x / z;
    let normalized_y = y / z;
    let projected_x = intrinsics.fx() * normalized_x;
    let projected_y = intrinsics.fy() * normalized_y;
    let widened_normalized_x = pc_widened[0] / pc_widened[2];
    let widened_normalized_y = pc_widened[1] / pc_widened[2];
    let widened_projected_x = f64::from(intrinsics.fx()) * widened_normalized_x;
    let widened_projected_y = f64::from(intrinsics.fy()) * widened_normalized_y;
    let u_sum = f32_sum_with_precision_risk(
        F32SumTerm::new(projected_x, widened_projected_x),
        [F32SumTerm::from_f32(intrinsics.cx())],
    );
    let v_sum = f32_sum_with_precision_risk(
        F32SumTerm::new(projected_y, widened_projected_y),
        [F32SumTerm::from_f32(intrinsics.cy())],
    );
    let u = u_sum.rounded;
    let v = v_sum.rounded;
    let residual = [pixel.x - u, pixel.y - v];

    let inv_z = 1.0 / z;
    let inv_z2 = inv_z * inv_z;
    let du_dx = intrinsics.fx() * inv_z;
    let du_dy = 0.0;
    let du_dz = -intrinsics.fx() * x * inv_z2;
    let dv_dx = 0.0;
    let dv_dy = intrinsics.fy() * inv_z;
    let dv_dz = -intrinsics.fy() * y * inv_z2;

    let a1 = du_dx;
    let a2 = du_dy;
    let a3 = du_dz;
    let b1 = dv_dx;
    let b2 = dv_dy;
    let b3 = dv_dz;
    let widened_inv_z = 1.0 / pc_widened[2];
    let widened_inv_z2 = widened_inv_z * widened_inv_z;
    let widened_a1 = f64::from(intrinsics.fx()) * widened_inv_z;
    let widened_a2 = 0.0;
    let widened_a3 = -f64::from(intrinsics.fx()) * pc_widened[0] * widened_inv_z2;
    let widened_b1 = 0.0;
    let widened_b2 = f64::from(intrinsics.fy()) * widened_inv_z;
    let widened_b3 = -f64::from(intrinsics.fy()) * pc_widened[1] * widened_inv_z2;
    // A subnormal intermediate product can be re-amplified into a normal derivative. Compare
    // the completed base derivatives as well as checking their final f32 classifications.
    let projection_jacobian_precision_risk = [
        (a1, widened_a1),
        (a3, widened_a3),
        (b2, widened_b2),
        (b3, widened_b3),
    ]
    .into_iter()
    .any(|(rounded, widened)| f32_value_requires_widened(rounded, widened));

    let mut pose_jacobian = [[0.0_f32; 6]; 2];

    pose_jacobian[0][0] = a1;
    pose_jacobian[0][1] = a2;
    pose_jacobian[0][2] = a3;
    pose_jacobian[1][0] = b1;
    pose_jacobian[1][1] = b2;
    pose_jacobian[1][2] = b3;

    pose_jacobian[0][3] = -(a2 * z - a3 * y);
    pose_jacobian[0][4] = a1 * z - a3 * x;
    pose_jacobian[0][5] = -a1 * y + a2 * x;

    pose_jacobian[1][3] = -(b2 * z - b3 * y);
    pose_jacobian[1][4] = b1 * z - b3 * x;
    pose_jacobian[1][5] = -b1 * y + b2 * x;

    let mut landmark_jacobian = [[0.0_f32; 3]; 2];
    let mut landmark_precision_risk = false;
    for col in 0..3 {
        let u_sum = f32_sum_with_precision_risk(
            F32SumTerm::new(
                a1 * rotation[0][col],
                widened_a1 * f64::from(rotation[0][col]),
            ),
            [
                F32SumTerm::new(
                    a2 * rotation[1][col],
                    widened_a2 * f64::from(rotation[1][col]),
                ),
                F32SumTerm::new(
                    a3 * rotation[2][col],
                    widened_a3 * f64::from(rotation[2][col]),
                ),
            ],
        );
        let v_sum = f32_sum_with_precision_risk(
            F32SumTerm::new(
                b1 * rotation[0][col],
                widened_b1 * f64::from(rotation[0][col]),
            ),
            [
                F32SumTerm::new(
                    b2 * rotation[1][col],
                    widened_b2 * f64::from(rotation[1][col]),
                ),
                F32SumTerm::new(
                    b3 * rotation[2][col],
                    widened_b3 * f64::from(rotation[2][col]),
                ),
            ],
        );
        landmark_jacobian[0][col] = u_sum.rounded;
        landmark_jacobian[1][col] = v_sum.rounded;
        landmark_precision_risk |= u_sum.requires_widened_value() || v_sum.requires_widened_value();
    }

    // The Jacobian above is for projected pixel coordinates [u, v].
    // Residual is defined as [pixel.x - u, pixel.y - v], so dr/dx = -du/dx.
    for row in &mut pose_jacobian {
        for value in row {
            *value = -*value;
        }
    }
    for row in &mut landmark_jacobian {
        for value in row {
            *value = -*value;
        }
    }

    let needs_widened_fallback = underflow_sensitive(normalized_x, x != 0.0)
        || underflow_sensitive(normalized_y, y != 0.0)
        || underflow_sensitive(projected_x, x != 0.0)
        || underflow_sensitive(projected_y, y != 0.0)
        || u_sum.requires_widened_sum()
        || v_sum.requires_widened_sum()
        || projection_jacobian_precision_risk
        || underflow_sensitive(inv_z, true)
        || underflow_sensitive(inv_z2, x != 0.0 || y != 0.0)
        || underflow_sensitive(pose_jacobian[0][3], x != 0.0 && y != 0.0)
        || underflow_sensitive(pose_jacobian[0][4], true)
        || underflow_sensitive(pose_jacobian[0][5], y != 0.0)
        || underflow_sensitive(pose_jacobian[1][3], true)
        || underflow_sensitive(pose_jacobian[1][4], x != 0.0 && y != 0.0)
        || underflow_sensitive(pose_jacobian[1][5], x != 0.0)
        || landmark_precision_risk;
    let fast_path = ReprojectionLinearization::try_new(residual, pose_jacobian, landmark_jacobian);
    if let Ok(linearization) = fast_path {
        // Ordinary factors retain the historical f32 bits. A coherent f64 factor is used only
        // when f32 arithmetic may have erased information from the residual or Jacobians.
        if !needs_widened_fallback {
            return Ok(linearization);
        }
        let widened = reprojection_linearization_widened(pc_widened, pixel, intrinsics, rotation)?;
        return if linearization.has_same_values(&widened) {
            Ok(linearization)
        } else {
            Ok(widened)
        };
    }

    reprojection_linearization_widened(pc_widened, pixel, intrinsics, rotation)
}

fn underflow_sensitive(value: f32, mathematically_nonzero: bool) -> bool {
    mathematically_nonzero && (value == 0.0 || value.is_subnormal())
}

fn f32_value_requires_widened(rounded: f32, widened: f64) -> bool {
    underflow_sensitive(rounded, widened != 0.0)
        || f32_relative_error_exceeds_limit(rounded, widened)
}

#[derive(Clone, Copy)]
struct F32SumTerm {
    /// Value produced by the historical f32 operation sequence.
    rounded: f32,
    /// Same source operation evaluated before narrowing or stepwise f32 accumulation.
    widened: f64,
}

impl F32SumTerm {
    fn new(rounded: f32, widened: f64) -> Self {
        Self { rounded, widened }
    }

    fn product(left: f32, right: f32) -> Self {
        Self::new(left * right, f64::from(left) * f64::from(right))
    }

    fn from_f32(value: f32) -> Self {
        Self::new(value, f64::from(value))
    }

    fn has_nonzero_source(self) -> bool {
        self.widened != 0.0
    }
}

#[derive(Clone, Copy)]
struct F32SumAnalysis {
    rounded: f32,
    widened: f64,
    source_underflow: bool,
    material_addition_loss: bool,
    excessive_relative_error: bool,
}

impl F32SumAnalysis {
    fn requires_widened_value(self) -> bool {
        !self.rounded.is_finite()
            || !self.widened.is_finite()
            || self.source_underflow
            || self.excessive_relative_error
    }

    fn requires_widened_sum(self) -> bool {
        self.requires_widened_value() || self.material_addition_loss
    }
}

fn f32_sum_with_precision_risk<const N: usize>(
    first_term: F32SumTerm,
    remaining_terms: [F32SumTerm; N],
) -> F32SumAnalysis {
    let mut rounded_sum = first_term.rounded;
    let mut widened_sum = first_term.widened;
    let mut widened_compensation = 0.0_f64;
    let mut source_underflow =
        underflow_sensitive(first_term.rounded, first_term.has_nonzero_source());
    let mut material_addition_loss = false;
    let mut excessive_relative_error = f32_relative_error_exceeds_limit(rounded_sum, widened_sum);

    for term in remaining_terms {
        source_underflow |= underflow_sensitive(term.rounded, term.has_nonzero_source());

        let exact_sum_of_rounded_terms = f64::from(rounded_sum) + f64::from(term.rounded);
        let next = rounded_sum + term.rounded;
        let addition_error = (f64::from(next) - exact_sum_of_rounded_terms).abs();
        let smaller_contribution = f64::from(rounded_sum)
            .abs()
            .min(f64::from(term.rounded).abs());
        material_addition_loss |= (smaller_contribution > 0.0
            && addition_error > smaller_contribution * F32_SUM_MAX_RELATIVE_CONTRIBUTION_LOSS)
            || (term.has_nonzero_source() && term.rounded != 0.0 && next == rounded_sum)
            || (rounded_sum != 0.0 && next == term.rounded);
        rounded_sum = next;

        math::compensated_add(&mut widened_sum, &mut widened_compensation, term.widened);
        // Check every prefix because a later term can mask cancellation in an earlier one.
        excessive_relative_error |=
            f32_relative_error_exceeds_limit(rounded_sum, widened_sum + widened_compensation);
    }

    let widened = widened_sum + widened_compensation;
    // Exact cancellation to a true zero is not underflow.
    source_underflow |= underflow_sensitive(rounded_sum, widened != 0.0);

    F32SumAnalysis {
        rounded: rounded_sum,
        widened,
        source_underflow,
        material_addition_loss,
        excessive_relative_error,
    }
}

fn f32_relative_error_exceeds_limit(rounded: f32, widened: f64) -> bool {
    let absolute_error = (f64::from(rounded) - widened).abs();
    !rounded.is_finite()
        || !widened.is_finite()
        || if widened == 0.0 {
            absolute_error != 0.0
        } else {
            absolute_error > widened.abs() * F32_SUM_MAX_RELATIVE_ERROR
        }
}

fn transform_point_f32_with_risk(
    rotation: [[f32; 3]; 3],
    translation: [f32; 3],
    point: [f32; 3],
) -> ([f32; 3], [f64; 3], bool) {
    let mut transformed_f32 = [0.0_f32; 3];
    let mut transformed_widened = [0.0_f64; 3];
    let mut precision_risk = false;
    for row in 0..3 {
        let sum = f32_sum_with_precision_risk(
            F32SumTerm::product(rotation[row][0], point[0]),
            [
                F32SumTerm::product(rotation[row][1], point[1]),
                F32SumTerm::product(rotation[row][2], point[2]),
                F32SumTerm::from_f32(translation[row]),
            ],
        );
        transformed_f32[row] = sum.rounded;
        transformed_widened[row] = sum.widened;
        precision_risk |= sum.requires_widened_sum();
    }
    (transformed_f32, transformed_widened, precision_risk)
}

#[cfg(test)]
fn transform_point_widened(
    rotation: [[f32; 3]; 3],
    translation: [f32; 3],
    point: [f32; 3],
) -> Result<[f64; 3], ProjectionLinearizationError> {
    let mut transformed = [0.0_f64; 3];
    for (row, target) in transformed.iter_mut().enumerate() {
        *target = math::compensated_sum([
            f64::from(rotation[row][0]) * f64::from(point[0]),
            f64::from(rotation[row][1]) * f64::from(point[1]),
            f64::from(rotation[row][2]) * f64::from(point[2]),
            f64::from(translation[row]),
        ]);
    }
    if transformed.iter().any(|value| !value.is_finite()) {
        return Err(ProjectionLinearizationError::NumericalFailure);
    }
    Ok(transformed)
}

fn reprojection_linearization_widened(
    camera_point: [f64; 3],
    pixel: Keypoint,
    intrinsics: PinholeIntrinsics,
    rotation: [[f32; 3]; 3],
) -> Result<ReprojectionLinearization, ProjectionLinearizationError> {
    let [x, y, z] = camera_point;
    if !x.is_finite() || !y.is_finite() || !z.is_finite() {
        return Err(ProjectionLinearizationError::NumericalFailure);
    }
    if z <= f64::from(MIN_PROJECTION_DEPTH_M) {
        return Err(ProjectionLinearizationError::InvalidDepth);
    }

    let fx = f64::from(intrinsics.fx());
    let fy = f64::from(intrinsics.fy());
    let qx = x / z;
    let qy = y / z;
    let sx = fx / z;
    let sy = fy / z;
    let residual = [
        f64::from(pixel.x) - (fx * qx + f64::from(intrinsics.cx())),
        f64::from(pixel.y) - (fy * qy + f64::from(intrinsics.cy())),
    ];
    let pose_jacobian = [
        [
            -sx,
            0.0,
            sx * qx,
            fx * qx * qy,
            -fx * (1.0 + qx * qx),
            fx * qy,
        ],
        [
            0.0,
            -sy,
            sy * qy,
            fy * (1.0 + qy * qy),
            -fy * qx * qy,
            -fy * qx,
        ],
    ];

    let mut landmark_jacobian = [[0.0_f64; 3]; 2];
    for column in 0..3 {
        landmark_jacobian[0][column] =
            -sx * (f64::from(rotation[0][column]) - qx * f64::from(rotation[2][column]));
        landmark_jacobian[1][column] =
            -sy * (f64::from(rotation[1][column]) - qy * f64::from(rotation[2][column]));
    }

    ReprojectionLinearization::try_from_widened(residual, pose_jacobian, landmark_jacobian)
}

pub(crate) fn apply_se3_delta(pose: Pose, delta: [f32; 6]) -> Result<Pose, crate::PoseError> {
    let v = [delta[0], delta[1], delta[2]];
    let w = [delta[3], delta[4], delta[5]];
    let r_delta = math::so3_exp(w);
    Pose::try_from_rt(r_delta, v)?.try_compose(pose)
}

pub(crate) fn se3_delta_between(from: Pose, to: Pose) -> Result<[f32; 6], crate::PoseError> {
    let delta = to.try_compose(from.try_inverse()?)?;
    let translation = delta.translation();
    let rotation = math::so3_log(delta.rotation());
    Ok([
        translation[0],
        translation[1],
        translation[2],
        rotation[0],
        rotation[1],
        rotation[2],
    ])
}

fn accumulate_pose_hessian(
    hessian: &mut [f32],
    pose_dim: usize,
    pose_idx: PoseVarIndex,
    j_pose: [[f32; 6]; 2],
) {
    let base = pose_idx.offset6();
    for row in 0..6 {
        for col in 0..6 {
            let jt_j = j_pose[0][row] * j_pose[0][col] + j_pose[1][row] * j_pose[1][col];
            hessian[(base + row) * pose_dim + (base + col)] += jt_j;
        }
    }
}

fn accumulate_pose_rhs(
    rhs: &mut [f32],
    pose_idx: PoseVarIndex,
    j_pose: [[f32; 6]; 2],
    residual: [f32; 2],
) {
    let base = pose_idx.offset6();
    for col in 0..6 {
        rhs[base + col] -= j_pose[0][col] * residual[0] + j_pose[1][col] * residual[1];
    }
}

fn accumulate_landmark_hessian(c: &mut [[f32; 3]; 3], j_landmark: [[f32; 3]; 2]) {
    for (row, c_row) in c.iter_mut().enumerate().take(3) {
        for (col, c_value) in c_row.iter_mut().enumerate().take(3) {
            *c_value +=
                j_landmark[0][row] * j_landmark[0][col] + j_landmark[1][row] * j_landmark[1][col];
        }
    }
}

fn accumulate_landmark_rhs(b: &mut [f32; 3], j_landmark: [[f32; 3]; 2], residual: [f32; 2]) {
    for (col, b_value) in b.iter_mut().enumerate().take(3) {
        *b_value -= j_landmark[0][col] * residual[0] + j_landmark[1][col] * residual[1];
    }
}

fn accumulate_scalar_landmark_factor(
    hessian: &mut [[f32; 3]; 3],
    rhs: &mut [f32; 3],
    jacobian: [f32; 3],
    residual: f32,
) {
    for row in 0..3 {
        rhs[row] -= jacobian[row] * residual;
        for col in 0..3 {
            hessian[row][col] += jacobian[row] * jacobian[col];
        }
    }
}

fn pose_landmark_cross(j_pose: [[f32; 6]; 2], j_landmark: [[f32; 3]; 2]) -> [[f32; 3]; 6] {
    let mut cross = [[0.0_f32; 3]; 6];
    for (row, cross_row) in cross.iter_mut().enumerate() {
        for (col, value) in cross_row.iter_mut().enumerate() {
            *value = j_pose[0][row] * j_landmark[0][col] + j_pose[1][row] * j_landmark[1][col];
        }
    }
    cross
}

/// Extract a 6-element SE3 delta from a solution vector at the given offset.
fn extract_se3_delta(rhs: &[f32], base: usize) -> [f32; 6] {
    [
        rhs[base],
        rhs[base + 1],
        rhs[base + 2],
        rhs[base + 3],
        rhs[base + 4],
        rhs[base + 5],
    ]
}

fn narrow_finite_f32(value: f64) -> Option<f32> {
    let narrowed = value as f32;
    narrowed.is_finite().then_some(narrowed)
}

fn mat63_mul_vec3(m: [[f32; 3]; 6], v: [f32; 3]) -> [f32; 6] {
    let mut out = [0.0_f32; 6];
    for row in 0..6 {
        out[row] = m[row][0] * v[0] + m[row][1] * v[1] + m[row][2] * v[2];
    }
    out
}

fn schur_block(b_i: [[f32; 3]; 6], inv_c: [[f32; 3]; 3], b_j: [[f32; 3]; 6]) -> [[f32; 6]; 6] {
    // Precompute b_i_inv_c = b_i * inv_c (6x3 matrix) to avoid redundant inner-loop work.
    let mut b_i_inv_c = [[0.0_f32; 3]; 6];
    for (row, bi_row) in b_i.iter().enumerate() {
        for (l, val) in b_i_inv_c[row].iter_mut().enumerate() {
            *val = bi_row[0] * inv_c[0][l] + bi_row[1] * inv_c[1][l] + bi_row[2] * inv_c[2][l];
        }
    }
    // Now block[row][col] = dot(b_i_inv_c[row], b_j[col]).
    let mut block = [[0.0_f32; 6]; 6];
    for (row, block_row) in block.iter_mut().enumerate() {
        let bic = &b_i_inv_c[row];
        for (col, block_value) in block_row.iter_mut().enumerate() {
            *block_value = bic[0] * b_j[col][0] + bic[1] * b_j[col][1] + bic[2] * b_j[col][2];
        }
    }
    block
}

fn fix_pose_block(hessian: &mut [f32], rhs: &mut [f32], pose_dim: usize, pose_idx: PoseVarIndex) {
    let base = pose_idx.offset6();
    for row in 0..6 {
        let idx = base + row;
        for col in 0..pose_dim {
            hessian[idx * pose_dim + col] = 0.0;
            hessian[col * pose_dim + idx] = 0.0;
        }
        hessian[idx * pose_dim + idx] = 1.0;
        rhs[idx] = 0.0;
    }
}

fn invert_3x3(m: [[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let a = m[0][0] as f64;
    let b = m[0][1] as f64;
    let c = m[0][2] as f64;
    let d = m[1][0] as f64;
    let e = m[1][1] as f64;
    let f = m[1][2] as f64;
    let g = m[2][0] as f64;
    let h = m[2][1] as f64;
    let i = m[2][2] as f64;

    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if !det.is_finite() || det.abs() < MIN_DETERMINANT {
        return None;
    }
    let inv_det = 1.0 / det;
    let inv = [
        [
            (e * i - f * h) * inv_det,
            (c * h - b * i) * inv_det,
            (b * f - c * e) * inv_det,
        ],
        [
            (f * g - d * i) * inv_det,
            (a * i - c * g) * inv_det,
            (c * d - a * f) * inv_det,
        ],
        [
            (d * h - e * g) * inv_det,
            (b * g - a * h) * inv_det,
            (a * e - b * d) * inv_det,
        ],
    ];
    if inv
        .iter()
        .flat_map(|row| row.iter())
        .any(|value| !value.is_finite())
    {
        return None;
    }
    Some([
        [
            narrow_finite_f32(inv[0][0])?,
            narrow_finite_f32(inv[0][1])?,
            narrow_finite_f32(inv[0][2])?,
        ],
        [
            narrow_finite_f32(inv[1][0])?,
            narrow_finite_f32(inv[1][1])?,
            narrow_finite_f32(inv[1][2])?,
        ],
        [
            narrow_finite_f32(inv[2][0])?,
            narrow_finite_f32(inv[2][1])?,
            narrow_finite_f32(inv[2][2])?,
        ],
    ])
}

fn finite_norm<const N: usize>(values: [f32; N]) -> Option<f64> {
    let mut sum_squared = 0.0_f64;
    for value in values {
        if !value.is_finite() {
            return None;
        }
        sum_squared += f64::from(value) * f64::from(value);
    }
    let norm = sum_squared.sqrt();
    norm.is_finite().then_some(norm)
}

fn solve_linear_system(a: &mut [f32], b: &mut [f32], n: usize) -> bool {
    let Some(matrix_len) = n.checked_mul(n) else {
        return false;
    };
    let Some(a) = a.get_mut(..matrix_len) else {
        return false;
    };
    let Some(b) = b.get_mut(..n) else {
        return false;
    };
    if a.iter().chain(b.iter()).any(|value| !value.is_finite()) {
        return false;
    }

    for i in 0..n {
        let mut max_row = i;
        let mut max_val = a[i * n + i].abs();
        for r in (i + 1)..n {
            let val = a[r * n + i].abs();
            if val > max_val {
                max_val = val;
                max_row = r;
            }
        }

        if !max_val.is_finite() || max_val < PIVOT_TOLERANCE {
            return false;
        }

        if max_row != i {
            for c in i..n {
                a.swap(i * n + c, max_row * n + c);
            }
            b.swap(i, max_row);
        }

        let diag = a[i * n + i];
        for c in i..n {
            let Some(normalized) = narrow_finite_f32((a[i * n + c] as f64) / (diag as f64)) else {
                return false;
            };
            a[i * n + c] = normalized;
        }
        let Some(normalized_rhs) = narrow_finite_f32((b[i] as f64) / (diag as f64)) else {
            return false;
        };
        b[i] = normalized_rhs;

        for r in 0..n {
            if r == i {
                continue;
            }
            let factor = a[r * n + i];
            if factor.abs() < ELIMINATION_TOLERANCE {
                continue;
            }
            for c in i..n {
                let Some(reduced) = narrow_finite_f32(
                    (a[r * n + c] as f64) - (factor as f64) * (a[i * n + c] as f64),
                ) else {
                    return false;
                };
                a[r * n + c] = reduced;
            }
            let Some(reduced_rhs) =
                narrow_finite_f32((b[r] as f64) - (factor as f64) * (b[i] as f64))
            else {
                return false;
            };
            b[r] = reduced_rhs;
        }
    }

    a.iter().chain(b.iter()).all(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::{KeyframeId, MapPointId, SlamMap, assert_map_invariants};
    use crate::test_helpers::{
        axis_angle_pose, make_detections, make_pinhole_intrinsics, project_world_point,
    };
    use crate::{FrameId, Keypoint, Point3, SensorId, Timestamp};

    fn l2_3(a: [f32; 3], b: [f32; 3]) -> f32 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    fn lm(initial_lambda: f32) -> LmConfig {
        LmConfig::new(initial_lambda, 10.0, 1e-8, 1e4, 0.25, 0.75).expect("valid lm")
    }

    fn pose_close(a: Pose, b: Pose, tol: f32) -> bool {
        let mut rot_sq = 0.0_f32;
        let ra = a.rotation();
        let rb = b.rotation();
        for i in 0..3 {
            for j in 0..3 {
                let d = ra[i][j] - rb[i][j];
                rot_sq += d * d;
            }
        }
        rot_sq.sqrt() <= tol && l2_3(a.translation(), b.translation()) <= tol
    }

    fn projection_residual(
        pose: Pose,
        obs: &Observation,
        intrinsics: PinholeIntrinsics,
    ) -> [f32; 2] {
        reprojection_linearization(pose, obs.world(), obs.pixel(), intrinsics)
            .expect("valid reprojection")
            .residual
    }

    fn project_pixel(
        pose_world_to_camera: Pose,
        point_world: Point3,
        intr: PinholeIntrinsics,
    ) -> Keypoint {
        let pc = math::transform_point(
            pose_world_to_camera.rotation(),
            pose_world_to_camera.translation(),
            [point_world.x, point_world.y, point_world.z],
        );
        Keypoint {
            x: intr.fx() * (pc[0] / pc[2]) + intr.cx(),
            y: intr.fy() * (pc[1] / pc[2]) + intr.cy(),
        }
    }

    fn pose_distance(a: Pose, b: Pose) -> f32 {
        let mut rot_sq = 0.0_f32;
        let ra = a.rotation();
        let rb = b.rotation();
        for i in 0..3 {
            for j in 0..3 {
                let d = ra[i][j] - rb[i][j];
                rot_sq += d * d;
            }
        }
        rot_sq.sqrt() + l2_3(a.translation(), b.translation())
    }

    fn mean_landmark_error(map: &SlamMap, keyframe_id: KeyframeId, expected: &[Point3]) -> f32 {
        let mut sum = 0.0_f32;
        for (idx, target) in expected.iter().enumerate() {
            let kp = map
                .keyframe_keypoint(keyframe_id, idx)
                .expect("keypoint index in map");
            let point_id = map
                .map_point_for_keypoint(kp)
                .expect("keyframe lookup")
                .expect("point exists");
            let point = map.point(point_id).expect("point lookup").position();
            let dx = point.x - target.x;
            let dy = point.y - target.y;
            let dz = point.z - target.z;
            sum += (dx * dx + dy * dy + dz * dz).sqrt();
        }
        sum / expected.len() as f32
    }

    fn build_full_ba_fixture(
        noisy_pose_delta: [f32; 6],
    ) -> (
        SlamMap,
        PinholeIntrinsics,
        KeyframeId,
        KeyframeId,
        Pose,
        Vec<Point3>,
    ) {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let true_pose_0 = Pose::identity();
        let true_pose_1 = axis_angle_pose([0.20, -0.02, 0.03], [0.0, 0.03, -0.01]);
        let noisy_pose_1 =
            apply_se3_delta(true_pose_1, noisy_pose_delta).expect("finite test pose update");

        let points_true = vec![
            Point3 {
                x: -0.35,
                y: -0.25,
                z: 3.2,
            },
            Point3 {
                x: -0.10,
                y: -0.22,
                z: 3.5,
            },
            Point3 {
                x: 0.14,
                y: -0.20,
                z: 3.8,
            },
            Point3 {
                x: 0.32,
                y: -0.10,
                z: 3.4,
            },
            Point3 {
                x: -0.30,
                y: 0.10,
                z: 3.6,
            },
            Point3 {
                x: -0.08,
                y: 0.16,
                z: 4.0,
            },
            Point3 {
                x: 0.16,
                y: 0.12,
                z: 3.3,
            },
            Point3 {
                x: 0.34,
                y: 0.24,
                z: 3.9,
            },
        ];

        let mut keypoints_0 = Vec::with_capacity(points_true.len());
        let mut keypoints_1 = Vec::with_capacity(points_true.len());
        for &point in &points_true {
            keypoints_0.push(
                project_world_point(true_pose_0, point, intrinsics)
                    .expect("point visible in pose 0"),
            );
            keypoints_1.push(
                project_world_point(true_pose_1, point, intrinsics)
                    .expect("point visible in pose 1"),
            );
        }

        let detections_0 = make_detections(
            SensorId::StereoLeft,
            FrameId::new(500),
            640,
            480,
            keypoints_0,
        )
        .expect("detections 0");
        let detections_1 = make_detections(
            SensorId::StereoLeft,
            FrameId::new(501),
            640,
            480,
            keypoints_1,
        )
        .expect("detections 1");

        let mut map = SlamMap::new();
        let kf_0 = map
            .add_keyframe_from_detections(
                detections_0.as_ref(),
                Timestamp::from_nanos(1_000_000),
                crate::WorldToCamera::from_legacy_pose(true_pose_0),
            )
            .expect("insert keyframe 0");
        let kf_1 = map
            .add_keyframe_from_detections(
                detections_1.as_ref(),
                Timestamp::from_nanos(2_000_000),
                crate::WorldToCamera::from_legacy_pose(noisy_pose_1),
            )
            .expect("insert keyframe 1");

        for (idx, &point_true) in points_true.iter().enumerate() {
            let kp_0 = map.keyframe_keypoint(kf_0, idx).expect("kf0 keypoint");
            let kp_1 = map.keyframe_keypoint(kf_1, idx).expect("kf1 keypoint");
            let descriptor = detections_0.descriptors()[idx];
            let i = idx as f32;
            let noisy_point = Point3 {
                x: point_true.x + (i - 3.5) * 0.010,
                y: point_true.y - (i - 3.5) * 0.008,
                z: point_true.z + ((idx % 2) as f32 - 0.5) * 0.040,
            };
            let point_id = map
                .add_map_point(noisy_point, descriptor.quantize(), kp_0)
                .expect("insert map point");
            map.add_observation(point_id, kp_1)
                .expect("add shared observation");
        }

        (map, intrinsics, kf_0, kf_1, true_pose_1, points_true)
    }

    fn set_fixture_landmarks_exact(map: &mut SlamMap, keyframe_id: KeyframeId, points: &[Point3]) {
        for (index, &point) in points.iter().enumerate() {
            let keypoint = map
                .keyframe_keypoint(keyframe_id, index)
                .expect("keyframe keypoint");
            let point_id = map
                .map_point_for_keypoint(keypoint)
                .expect("map lookup")
                .expect("map point");
            map.set_map_point_position(point_id, point)
                .expect("set exact landmark");
        }
    }

    fn fixture_observation_set(
        map: &SlamMap,
        keyframe_id: KeyframeId,
        min_required: NonZeroUsize,
    ) -> ObservationSet {
        let observations = (0..map.keyframe(keyframe_id).expect("keyframe").len())
            .map(|index| {
                let keypoint = map
                    .keyframe_keypoint(keyframe_id, index)
                    .expect("keyframe keypoint");
                MapObservation::new(keypoint, map.keypoint(keypoint).expect("keypoint pixel"))
            })
            .collect();
        ObservationSet::new(observations, min_required).expect("observation set")
    }

    #[test]
    fn local_ba_config_rejects_invalid_values() {
        assert!(matches!(
            LocalBaConfig::new(0, 10, 4, 1.0, lm(1e-3)),
            Err(LocalBaConfigError::ZeroWindow)
        ));
        assert!(matches!(
            LocalBaConfig::new(usize::MAX, 10, 4, 1.0, lm(1e-3)),
            Err(LocalBaConfigError::WindowTooLarge { value: usize::MAX })
        ));
        assert!(matches!(
            LocalBaConfig::new(5, 0, 4, 1.0, lm(1e-3)),
            Err(LocalBaConfigError::ZeroIterations)
        ));
        assert!(matches!(
            LocalBaConfig::new(5, 10, 0, 1.0, lm(1e-3)),
            Err(LocalBaConfigError::ZeroObservations)
        ));
        assert!(matches!(
            LocalBaConfig::new(5, 10, 3, 1.0, lm(1e-3)),
            Err(LocalBaConfigError::TooFewObservations { .. })
        ));
        assert!(matches!(
            LocalBaConfig::new(5, 10, 4, 0.0, lm(1e-3)),
            Err(LocalBaConfigError::NonPositiveHuber { .. })
        ));
    }

    #[test]
    fn lm_config_rejects_invalid_values() {
        assert!(matches!(
            LmConfig::new(0.0, 10.0, 1e-8, 1e4, 0.25, 0.75),
            Err(LmConfigError::NonPositiveInitialLambda { .. })
        ));
        assert!(matches!(
            LmConfig::new(1e-4, 1.0, 1e-8, 1e4, 0.25, 0.75),
            Err(LmConfigError::LambdaFactorTooSmall { .. })
        ));
        assert!(matches!(
            LmConfig::new(1e-4, 10.0, 1e-2, 1e-3, 0.25, 0.75),
            Err(LmConfigError::MinLambdaExceedsMax { .. })
        ));
        assert!(matches!(
            LmConfig::new(1e-4, 10.0, 1e-8, 1e4, 0.8, 0.7),
            Err(LmConfigError::InvalidRhoOrdering { .. })
        ));
    }

    #[test]
    fn lm_state_good_rho_decreases_lambda() {
        let config = LmConfig::default();
        let mut state = LmState::new(config, 10.0);
        let action = state.step(8.0, 1.0, config);
        assert_eq!(action, LmAction::Accept);
        assert!(state.lambda() < config.initial_lambda());
        assert!((state.prev_cost() - 8.0).abs() < 1e-9);
    }

    #[test]
    fn lm_state_bad_rho_rejects_and_increases_lambda() {
        let config = LmConfig::default();
        let mut state = LmState::new(config, 10.0);
        let action = state.step(9.9, 10.0, config);
        assert_eq!(action, LmAction::Reject);
        assert!(state.lambda() > config.initial_lambda());
        assert!((state.prev_cost() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn observation_set_rejects_too_few_points() {
        let min_required = NonZeroUsize::new(4).expect("nonzero");
        let err = ObservationSet::new(Vec::new(), min_required).expect_err("must reject");
        match err {
            ObservationSetError::TooFew { required, actual } => {
                assert_eq!(required, 4);
                assert_eq!(actual, 0);
            }
        }
    }

    #[test]
    fn push_frame_rejects_unprojectable_observations_and_recovers() {
        let (mut map, intrinsics, keyframe_id, _, _, _) = build_full_ba_fixture([0.0; 6]);
        let config = LocalBaConfig::new(5, 5, 4, 2.0, lm(1e-3)).expect("valid BA config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);
        let min_required = ba.min_observations();
        let observation_count = map.keyframe(keyframe_id).expect("keyframe").len();
        let make_observation_set = |map: &SlamMap| {
            let observations = (0..observation_count)
                .map(|index| {
                    let keypoint = map
                        .keyframe_keypoint(keyframe_id, index)
                        .expect("keyframe keypoint");
                    let pixel = map.keypoint(keypoint).expect("keypoint pixel");
                    MapObservation::new(keypoint, pixel)
                })
                .collect();
            ObservationSet::new(observations, min_required).expect("observation set")
        };

        let behind_camera = Pose::from_rt(Pose::identity().rotation(), [0.0, 0.0, -10.0]);
        assert!(
            matches!(
                ba.push_frame(&map, behind_camera, make_observation_set(&map)),
                Err(LocalBaError::InsufficientUsableObservations { actual: 0, .. })
            ),
            "a frame with no usable reprojection factors must be rejected"
        );
        assert!(
            ba.frames.is_empty(),
            "a rejected frame must not remain in the optimization window"
        );

        let hidden_keypoint = map
            .keyframe_keypoint(keyframe_id, 0)
            .expect("first keypoint");
        let hidden_point = map
            .map_point_for_keypoint(hidden_keypoint)
            .expect("map lookup")
            .expect("associated point");
        let position = map.point(hidden_point).expect("point lookup").position();
        map.set_map_point_position(
            hidden_point,
            crate::WorldPoint3::new(position.x, position.y, -1.0),
        )
        .expect("finite hidden point");

        assert!(
            ba.push_frame(&map, Pose::identity(), make_observation_set(&map))
                .is_ok(),
            "a frame with seven usable factors must optimize after the failed window is reset"
        );

        for index in 1..5 {
            let keypoint = map.keyframe_keypoint(keyframe_id, index).expect("keypoint");
            let point_id = map
                .map_point_for_keypoint(keypoint)
                .expect("map lookup")
                .expect("associated point");
            let position = map.point(point_id).expect("point lookup").position();
            map.set_map_point_position(
                point_id,
                crate::WorldPoint3::new(position.x, position.y, -1.0),
            )
            .expect("finite hidden point");
        }

        assert!(matches!(
            ba.push_frame(&map, Pose::identity(), make_observation_set(&map)),
            Err(LocalBaError::InsufficientUsableObservations {
                required: 4,
                actual: 3,
            })
        ));
        assert!(
            ba.frames.is_empty(),
            "a below-minimum final window must not remain accepted"
        );
    }

    #[test]
    fn final_incremental_validation_rejects_a_staged_depth_threshold_crossing() {
        let (map, intrinsics, keyframe_id, _, _, _) = build_full_ba_fixture([0.0; 6]);
        let config = LocalBaConfig::new(5, 1, 4, 2.0, lm(1e-3)).expect("valid BA config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);
        let observations = (0..map.keyframe(keyframe_id).expect("keyframe").len())
            .map(|index| {
                let keypoint = map.keyframe_keypoint(keyframe_id, index).expect("keypoint");
                MapObservation::new(keypoint, map.keypoint(keypoint).expect("pixel"))
            })
            .collect();
        ba.frames.push(BaFrame {
            pose: Pose::identity(),
            observations: ObservationSet::new(observations, ba.min_observations())
                .expect("observation set"),
        });

        ba.validate_final_incremental_window(&map)
            .expect("initial pose has complete projection support");
        ba.frames[0].pose = Pose::from_rt(Pose::identity().rotation(), [0.0, 0.0, -10.0]);

        assert!(matches!(
            ba.validate_final_incremental_window(&map),
            Err(LocalBaError::InsufficientUsableObservations {
                required: 4,
                actual: 0,
            })
        ));
    }

    #[test]
    fn push_frame_preserves_missing_map_point_association_error() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let detections = make_detections(
            SensorId::StereoLeft,
            FrameId::new(700),
            640,
            480,
            vec![Keypoint { x: 10.0, y: 10.0 }; 4],
        )
        .expect("detections");
        let mut map = SlamMap::new();
        let keyframe_id = map
            .add_keyframe_from_detections(
                detections.as_ref(),
                Timestamp::from_nanos(1),
                crate::WorldToCamera::identity(),
            )
            .expect("keyframe");
        let observations = (0..4)
            .map(|index| {
                MapObservation::new(
                    map.keyframe_keypoint(keyframe_id, index)
                        .expect("keyframe keypoint"),
                    detections.keypoints()[index],
                )
            })
            .collect();
        let config = LocalBaConfig::new(5, 5, 4, 2.0, lm(1e-3)).expect("config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);
        let observation_set =
            ObservationSet::new(observations, ba.min_observations()).expect("observation set");

        assert!(matches!(
            ba.push_frame(&map, Pose::identity(), observation_set),
            Err(LocalBaError::MissingMapPointAssociation {
                keyframe_id: missing_keyframe,
                keypoint_index: 0,
            }) if missing_keyframe == keyframe_id
        ));
        assert!(ba.frames.is_empty());
    }

    #[test]
    fn push_frame_preserves_observation_error_and_resets_window() {
        let (map, intrinsics, keyframe_id, _, _, _) = build_full_ba_fixture([0.0; 6]);
        let config = LocalBaConfig::new(5, 5, 4, 2.0, lm(1e-3)).expect("config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);
        let observations = (0..ba.min_observations().get())
            .map(|index| {
                let keypoint = map
                    .keyframe_keypoint(keyframe_id, index)
                    .expect("keyframe keypoint");
                let mut pixel = map.keypoint(keypoint).expect("pixel");
                if index == 0 {
                    pixel.x = f32::NAN;
                }
                MapObservation::new(keypoint, pixel)
            })
            .collect();
        let observation_set =
            ObservationSet::new(observations, ba.min_observations()).expect("observation set");

        assert!(matches!(
            ba.push_frame(&map, Pose::identity(), observation_set),
            Err(LocalBaError::Observation(
                ObservationError::NonFinitePixel {
                    axis: 0,
                    value,
                }
            )) if value.is_nan()
        ));
        assert!(ba.frames.is_empty());
    }

    #[test]
    fn so3_exp_log_round_trip_for_small_rotation() {
        let w = [0.18, -0.06, 0.11];
        let r = math::so3_exp(w);
        let recovered = math::so3_log(r);
        assert!(
            l2_3(w, recovered) < 2e-4,
            "round-trip mismatch: {recovered:?}"
        );
    }

    #[test]
    fn so3_log_is_finite_near_pi() {
        let theta = std::f32::consts::PI - 1e-4;
        let w = [0.0, theta, 0.0];
        let r = math::so3_exp(w);
        let recovered = math::so3_log(r);
        assert!(recovered.iter().all(|v| v.is_finite()));

        let recovered_norm = (recovered[0] * recovered[0]
            + recovered[1] * recovered[1]
            + recovered[2] * recovered[2])
            .sqrt();
        assert!(
            (recovered_norm - theta).abs() < 3e-3,
            "theta mismatch: recovered={recovered_norm}, expected={theta}"
        );
    }

    #[test]
    fn apply_se3_delta_zero_is_fixpoint() {
        let pose = axis_angle_pose([0.3, -0.4, 0.5], [0.08, -0.05, 0.03]);
        let out = apply_se3_delta(pose, [0.0; 6]).expect("zero update stays finite");
        assert!(pose_close(pose, out, 1e-7));
    }

    #[test]
    fn apply_se3_delta_propagates_translation_overflow() {
        let pose = Pose::from_rt(Pose::identity().rotation(), [f32::MAX, 0.0, 0.0]);
        assert!(matches!(
            apply_se3_delta(pose, [f32::MAX, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Err(crate::PoseError::ComposeTranslationNotRepresentable {
                axis: 0,
                value,
            }) if value > f64::from(f32::MAX)
        ));
    }

    #[test]
    fn solve_linear_system_solves_identity_system() {
        let mut a = vec![1.0_f32, 0.0, 0.0, 1.0];
        let mut b = vec![2.5_f32, -3.0];
        assert!(solve_linear_system(&mut a, &mut b, 2));
        assert!((b[0] - 2.5).abs() < 1e-6);
        assert!((b[1] + 3.0).abs() < 1e-6);
    }

    #[test]
    fn solve_linear_system_reports_singular_matrix() {
        let mut a = vec![1.0_f32, 2.0, 2.0, 4.0];
        let mut b = vec![1.0_f32, 2.0];
        assert!(!solve_linear_system(&mut a, &mut b, 2));
    }

    #[test]
    fn solve_linear_system_rejects_non_finite_inputs() {
        for non_finite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut a = [non_finite];
            let mut b = [1.0];
            assert!(!solve_linear_system(&mut a, &mut b, 1));

            let mut a = [1.0];
            let mut b = [non_finite];
            assert!(!solve_linear_system(&mut a, &mut b, 1));
        }
    }

    #[test]
    fn solve_linear_system_rejects_non_finite_solution() {
        let mut a = [0.5];
        let mut b = [f32::MAX];
        assert!(!solve_linear_system(&mut a, &mut b, 1));
    }

    #[test]
    fn solve_linear_system_accepts_finite_extreme_scale() {
        let mut a = [f32::MAX];
        let mut b = [f32::MAX];
        assert!(solve_linear_system(&mut a, &mut b, 1));
        assert_eq!(b, [1.0]);
    }

    #[test]
    fn inverse_rejects_f64_result_that_cannot_narrow_to_f32() {
        let matrix = [[1e20, 0.0, 0.0], [0.0, 1e20, 0.0], [0.0, 0.0, 1e-40]];
        assert!(invert_3x3(matrix).is_none());
    }

    #[test]
    fn finite_norm_avoids_f32_intermediate_overflow() {
        let component = f32::MAX / 4.0;
        let norm = finite_norm([component; 6]).expect("finite f32 vector has an f64 norm");
        assert!(norm.is_finite());
        assert!(norm <= f64::from(f32::MAX));

        assert!(finite_norm([f32::NAN, 0.0, 0.0]).is_none());
        assert!(finite_norm([f32::INFINITY, 0.0, 0.0]).is_none());
    }

    #[test]
    fn optimize_full_reports_degenerate_variants() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let config = LocalBaConfig::new(5, 15, 4, 2.0, lm(1e-3)).expect("valid BA config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);

        let mut no_poses = FullBaProblem {
            poses: Vec::new(),
            landmarks: Vec::new(),
            factors: Vec::new(),
            scale_anchor: None,
        };
        assert!(matches!(
            ba.optimize_full(&mut no_poses).expect("typed BA result"),
            BaResult::Degenerate {
                reason: DegenerateReason::TooFewPoses { count: 0 }
            }
        ));

        let mut no_landmarks = FullBaProblem {
            poses: vec![
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: Pose::identity(),
                },
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: Pose::identity(),
                },
            ],
            landmarks: Vec::new(),
            factors: Vec::new(),
            scale_anchor: None,
        };
        assert!(matches!(
            ba.optimize_full(&mut no_landmarks)
                .expect("typed BA result"),
            BaResult::Degenerate {
                reason: DegenerateReason::TooFewLandmarks { count: 0 }
            }
        ));

        let mut no_factors = FullBaProblem {
            poses: vec![
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: Pose::identity(),
                },
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: Pose::identity(),
                },
            ],
            landmarks: vec![LandmarkVariable {
                point_id: MapPointId::default(),
                position: Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 2.0,
                },
            }],
            factors: Vec::new(),
            scale_anchor: None,
        };
        assert!(matches!(
            ba.optimize_full(&mut no_factors).expect("typed BA result"),
            BaResult::Degenerate {
                reason: DegenerateReason::NoFactors
            }
        ));
    }

    #[test]
    fn optimize_full_reports_numerical_failure_for_noninvertible_landmark_block() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, f32::MAX, 1.0, 0.0, 0.0).expect("finite intrinsics");
        let config = LocalBaConfig::new(5, 5, 4, 2.0, lm(1e-3)).expect("valid config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);
        let mut problem = FullBaProblem {
            poses: vec![
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: Pose::identity(),
                },
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: Pose::identity(),
                },
            ],
            landmarks: vec![LandmarkVariable {
                point_id: MapPointId::default(),
                position: Point3::new(0.0, 0.0, 1.0),
            }],
            factors: vec![ReprojectionFactor {
                pose: FactorPose::Variable(PoseVarIndex(1)),
                landmark: LandmarkVarIndex(0),
                pixel: Keypoint { x: 0.0, y: 0.0 },
            }],
            scale_anchor: None,
        };

        assert_eq!(
            ba.optimize_full(&mut problem).expect("typed BA result"),
            BaResult::Degenerate {
                reason: DegenerateReason::NumericalFailure
            }
        );
    }

    #[test]
    fn optimize_full_reports_numerical_failure_for_singular_pose_system() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let lm = LmConfig::new(1e-12, 10.0, 1e-12, 1e4, 0.25, 0.75).expect("valid LM");
        let config = LocalBaConfig::new(5, 5, 4, 2.0, lm).expect("valid config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);
        let point = Point3::new(0.0, 0.0, 2.0);
        let pixel = project_pixel(Pose::identity(), point, intrinsics);
        let mut problem = FullBaProblem {
            poses: vec![
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: Pose::identity(),
                },
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: Pose::identity(),
                },
            ],
            landmarks: vec![LandmarkVariable {
                point_id: MapPointId::default(),
                position: point,
            }],
            factors: vec![ReprojectionFactor {
                pose: FactorPose::Variable(PoseVarIndex(0)),
                landmark: LandmarkVarIndex(0),
                pixel,
            }],
            scale_anchor: None,
        };

        assert_eq!(
            ba.optimize_full(&mut problem).expect("typed BA result"),
            BaResult::Degenerate {
                reason: DegenerateReason::NumericalFailure
            }
        );
    }

    #[test]
    fn full_keyframe_ba_converges_at_exact_moving_solution() {
        let (mut map, intrinsics, kf_0, kf_1, _, points_true) = build_full_ba_fixture([0.0; 6]);
        set_fixture_landmarks_exact(&mut map, kf_0, &points_true);

        let config = LocalBaConfig::new(5, 5, 4, 2.0, lm(1e-3)).expect("valid config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);
        let mut problem = FullBaProblem::try_from_map(
            &map,
            &[kf_0, kf_1],
            ba.window_size(),
            ba.min_observations(),
        )
        .expect("full BA problem");
        assert_ne!(
            problem.poses[0].pose.translation(),
            problem.poses[1].pose.translation(),
            "the exact fixture must contain camera motion"
        );
        assert_eq!(
            full_problem_cost(&problem, intrinsics, config.huber_delta_px()),
            Ok(0.0)
        );

        assert_eq!(
            ba.optimize_full(&mut problem).expect("typed BA result"),
            BaResult::Converged {
                iterations: 1,
                final_cost: 0.0
            }
        );
    }

    #[test]
    fn incremental_ba_preserves_exact_moving_frame() {
        let (mut map, intrinsics, kf_0, kf_1, true_pose_1, points_true) =
            build_full_ba_fixture([0.0; 6]);
        set_fixture_landmarks_exact(&mut map, kf_0, &points_true);

        let config = LocalBaConfig::new(2, 1, 4, 2.0, lm(1e-3)).expect("config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);
        let min_required = ba.min_observations();
        ba.push_frame(
            &map,
            Pose::identity(),
            fixture_observation_set(&map, kf_0, min_required),
        )
        .expect("first exact frame");
        let refined = ba
            .push_frame(
                &map,
                true_pose_1,
                fixture_observation_set(&map, kf_1, min_required),
            )
            .expect("second exact frame");

        assert_eq!(refined.translation(), true_pose_1.translation());
        assert_eq!(refined.rotation(), true_pose_1.rotation());
    }

    #[test]
    fn invalid_landmark_observation_is_an_invariant_violation() {
        let error = FullBaBuildError::InvalidLandmarkObservation {
            point_id: MapPointId::default(),
            keyframe_id: KeyframeId::default(),
            keypoint_index: 7,
        };

        assert_eq!(
            degenerate_reason_from_build_error(&error),
            DegenerateReason::InvariantViolation
        );
    }

    #[test]
    fn optimize_full_returns_max_iterations_with_bad_init() {
        let (map, intrinsics, kf_0, kf_1, _, _) =
            build_full_ba_fixture([0.8, -0.3, 0.4, 0.2, -0.1, 0.15]);
        let config = LocalBaConfig::new(5, 1, 4, 2.0, lm(1e-3)).expect("valid BA config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);
        let mut problem = FullBaProblem::try_from_map(
            &map,
            &[kf_0, kf_1],
            ba.window_size(),
            ba.min_observations(),
        )
        .expect("full BA problem");
        match ba.optimize_full(&mut problem).expect("typed BA result") {
            BaResult::MaxIterations {
                iterations: 1,
                final_cost,
            } => assert!(final_cost.is_finite()),
            other => panic!("expected one finite-cost iteration, got {other:?}"),
        }
    }

    #[test]
    fn optimize_full_returns_converged_on_synthetic_scene() {
        let (map, intrinsics, kf_0, kf_1, _, _) =
            build_full_ba_fixture([0.08, -0.03, 0.04, 0.015, -0.01, 0.008]);
        let config = LocalBaConfig::new(5, 15, 4, 2.0, lm(1e-3)).expect("valid BA config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);
        let mut problem = FullBaProblem::try_from_map(
            &map,
            &[kf_0, kf_1],
            ba.window_size(),
            ba.min_observations(),
        )
        .expect("full BA problem");
        let result = ba.optimize_full(&mut problem).expect("typed BA result");
        match result {
            BaResult::Converged {
                iterations,
                final_cost,
            } => {
                assert!(iterations < config.max_iterations());
                assert!(final_cost.is_finite());
            }
            other => panic!("expected convergence, got {other:?}"),
        }
    }

    #[test]
    fn optimize_full_recovers_from_large_perturbation() {
        let (map, intrinsics, kf_0, kf_1, _, _) =
            build_full_ba_fixture([0.45, -0.20, 0.28, 0.15, -0.08, 0.10]);
        let config = LocalBaConfig::new(5, 30, 4, 2.0, lm(1e-3)).expect("valid BA config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);
        let mut problem = FullBaProblem::try_from_map(
            &map,
            &[kf_0, kf_1],
            ba.window_size(),
            ba.min_observations(),
        )
        .expect("full BA problem");
        let before = full_problem_cost(&problem, intrinsics, config.huber_delta_px())
            .expect("finite initial cost");
        let result = ba.optimize_full(&mut problem).expect("typed BA result");
        let after = full_problem_cost(&problem, intrinsics, config.huber_delta_px())
            .expect("finite final cost");
        assert!(
            after < before,
            "cost did not improve: before={before}, after={after}"
        );
        assert!(matches!(
            result,
            BaResult::Converged { .. } | BaResult::MaxIterations { .. }
        ));
    }

    #[test]
    fn optimize_full_final_cost_does_not_increase() {
        let (map, intrinsics, kf_0, kf_1, _, _) =
            build_full_ba_fixture([0.12, -0.05, 0.07, 0.03, -0.02, 0.01]);
        let config = LocalBaConfig::new(5, 20, 4, 2.0, lm(1e-3)).expect("valid BA config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);
        let mut problem = FullBaProblem::try_from_map(
            &map,
            &[kf_0, kf_1],
            ba.window_size(),
            ba.min_observations(),
        )
        .expect("full BA problem");
        let before = full_problem_cost(&problem, intrinsics, config.huber_delta_px())
            .expect("finite initial cost");
        let result = ba.optimize_full(&mut problem).expect("typed BA result");
        let final_cost = match result {
            BaResult::Converged { final_cost, .. } | BaResult::MaxIterations { final_cost, .. } => {
                final_cost
            }
            BaResult::Degenerate { reason } => panic!("unexpected degeneracy: {reason:?}"),
        };
        assert!(
            final_cost <= before + 1e-6,
            "final cost should not increase: before={before}, final={final_cost}"
        );
    }

    #[test]
    fn full_problem_cost_rejects_behind_camera_factor() {
        let (map, intrinsics, kf_0, kf_1, _, _) = build_full_ba_fixture([0.0; 6]);
        let config = LocalBaConfig::new(5, 5, 4, 2.0, lm(1e-3)).expect("valid config");
        let ba = LocalBundleAdjuster::new(intrinsics, config);
        let mut problem = FullBaProblem::try_from_map(
            &map,
            &[kf_0, kf_1],
            ba.window_size(),
            ba.min_observations(),
        )
        .expect("full BA problem");
        problem.landmarks[0].position = Point3 {
            x: 0.0,
            y: 0.0,
            z: -10.0,
        };

        assert_eq!(
            full_problem_cost(&problem, intrinsics, config.huber_delta_px()),
            Err(ProjectionLinearizationError::InvalidDepth)
        );
    }

    #[test]
    fn optimize_full_rejects_and_restores_an_invalid_depth_trial() {
        let intrinsics = make_pinhole_intrinsics(640, 480, 1.0, 1.0, 0.0, 0.0).expect("intrinsics");
        let config = LocalBaConfig::new(5, 1, 4, 2.0, lm(1e-3)).expect("valid config");
        let initial_pose = Pose::identity();
        let initial_landmark = Point3::new(1.0, 0.0, 1.0);
        let mut problem = FullBaProblem {
            poses: vec![
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: initial_pose,
                },
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: initial_pose,
                },
            ],
            landmarks: vec![LandmarkVariable {
                point_id: MapPointId::default(),
                position: initial_landmark,
            }],
            factors: vec![ReprojectionFactor {
                pose: FactorPose::Fixed(initial_pose),
                landmark: LandmarkVarIndex(0),
                pixel: Keypoint { x: 100.0, y: 0.0 },
            }],
            scale_anchor: None,
        };
        let initial_cost = full_problem_cost(&problem, intrinsics, config.huber_delta_px())
            .expect("finite initial cost");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);

        let result = ba.optimize_full(&mut problem).expect("typed BA result");

        assert!(matches!(
            result,
            BaResult::MaxIterations {
                iterations: 1,
                final_cost,
            } if final_cost.to_bits() == initial_cost.to_bits()
        ));
        for pose in &problem.poses {
            assert_eq!(pose.pose.rotation(), initial_pose.rotation());
            assert_eq!(pose.pose.translation(), initial_pose.translation());
        }
        assert_eq!(problem.landmarks[0].position, initial_landmark);
    }

    #[test]
    fn optimize_full_preserves_metric_gauge_anchor_distance() {
        let (map, intrinsics, kf_0, kf_1, _, _) =
            build_full_ba_fixture([0.08, -0.03, 0.04, 0.015, -0.01, 0.008]);
        let config = LocalBaConfig::new(5, 15, 4, 2.0, lm(1e-3)).expect("valid config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);
        let mut problem = FullBaProblem::try_from_map(
            &map,
            &[kf_0, kf_1],
            ba.window_size(),
            ba.min_observations(),
        )
        .expect("full BA problem");
        let anchor = problem.scale_anchor.expect("metric scale anchor");
        let anchor_before = problem.landmarks[anchor.landmark.as_usize()].position;
        let distance_before = point_distance(anchor_before, anchor.camera_center);

        let result = ba.optimize_full(&mut problem).expect("typed BA result");

        assert!(matches!(
            result,
            BaResult::Converged { .. } | BaResult::MaxIterations { .. }
        ));
        let anchor_after = problem.landmarks[anchor.landmark.as_usize()].position;
        let distance_after = point_distance(anchor_after, anchor.camera_center);
        assert!(
            (distance_after - distance_before).abs() < 1e-3,
            "metric anchor distance changed: before={distance_before}, after={distance_after}"
        );
        assert!(
            point_distance(anchor_after, anchor_before) > 1e-6,
            "rank-1 anchor should leave tangential landmark corrections available"
        );
    }

    #[test]
    fn full_ba_retains_observations_outside_local_window_as_fixed_factors() {
        let (mut map, intrinsics, kf_0, kf_1, _, points_true) = build_full_ba_fixture([0.0; 6]);
        let external_pose = axis_angle_pose([-0.25, 0.03, 0.04], [0.0, -0.02, 0.01]);
        let external_pixels: Vec<_> = points_true
            .iter()
            .map(|&point| {
                project_world_point(external_pose, point, intrinsics)
                    .expect("point visible in external keyframe")
            })
            .collect();
        let external_detections = make_detections(
            SensorId::StereoLeft,
            FrameId::new(502),
            640,
            480,
            external_pixels,
        )
        .expect("external detections");
        let external_kf = map
            .add_keyframe_from_detections(
                external_detections.as_ref(),
                Timestamp::from_nanos(3_000_000),
                crate::WorldToCamera::from_legacy_pose(external_pose),
            )
            .expect("insert external keyframe");
        for idx in 0..points_true.len() {
            let local_keypoint = map.keyframe_keypoint(kf_0, idx).expect("local keypoint");
            let point_id = map
                .map_point_for_keypoint(local_keypoint)
                .expect("map lookup")
                .expect("mapped point");
            let external_keypoint = map
                .keyframe_keypoint(external_kf, idx)
                .expect("external keypoint");
            map.add_observation(point_id, external_keypoint)
                .expect("external observation");
        }

        let config = LocalBaConfig::new(5, 5, 4, 2.0, lm(1e-3)).expect("valid config");
        let ba = LocalBundleAdjuster::new(intrinsics, config);
        let problem = FullBaProblem::try_from_map(
            &map,
            &[kf_0, kf_1],
            ba.window_size(),
            ba.min_observations(),
        )
        .expect("full BA problem");

        let fixed_factors = problem
            .factors
            .iter()
            .filter(|factor| matches!(factor.pose, FactorPose::Fixed(_)))
            .count();
        assert_eq!(fixed_factors, points_true.len());
    }

    #[test]
    fn reprojection_linearization_keeps_a_large_finite_robust_factor() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 1.0e30, 1.0, 0.0, 0.0).expect("finite intrinsics");
        let point = Point3::new(-1.0, 0.0, 1.0);
        let pixel = Keypoint { x: 0.0, y: 0.0 };
        let linearization = reprojection_linearization(Pose::identity(), point, pixel, intrinsics)
            .expect("large finite projection remains representable");

        assert_eq!(linearization.residual[0].to_bits(), 1.0e30_f32.to_bits());
        let scale = linearization.huber_sqrt_weight(2.0);
        assert!(scale.is_finite() && scale > 0.0);
        let scaled_residual = linearization.residual[0] * scale;
        let scaled_jacobian = linearization.pose_jacobian[0][0] * scale;
        let hessian_contribution = scaled_jacobian * scaled_jacobian;
        assert!(scaled_residual.is_finite() && (1.0e15..2.0e15).contains(&scaled_residual));
        assert!(hessian_contribution.is_finite() && hessian_contribution > 1.0e30);

        let problem = FullBaProblem {
            poses: vec![PoseVariable {
                keyframe_id: KeyframeId::default(),
                pose: Pose::identity(),
            }],
            landmarks: vec![LandmarkVariable {
                point_id: MapPointId::default(),
                position: point,
            }],
            factors: vec![ReprojectionFactor {
                pose: FactorPose::Variable(PoseVarIndex(0)),
                landmark: LandmarkVarIndex(0),
                pixel,
            }],
            scale_anchor: None,
        };
        let cost = full_problem_cost(&problem, intrinsics, 2.0).expect("large finite robust cost");
        assert!(cost.is_finite());
        assert!((cost / 2.0e30 - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn huber_sqrt_weight_remains_positive_at_f32_extremes() {
        let linearization =
            ReprojectionLinearization::try_new([f32::MAX, f32::MAX], [[0.0; 6]; 2], [[0.0; 3]; 2])
                .expect("finite residual");
        let smallest_positive_delta_px = f32::from_bits(1);
        let scale = linearization.huber_sqrt_weight(smallest_positive_delta_px);

        assert!(scale.is_finite());
        assert!(scale > 0.0);
        assert!(scale <= 1.0);
    }

    #[test]
    fn reprojection_linearization_distinguishes_invalid_depth_from_numerical_failure() {
        let ordinary =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 0.0, 0.0).expect("intrinsics");
        assert!(matches!(
            reprojection_linearization(
                Pose::identity(),
                Point3::new(0.0, 0.0, -1.0),
                Keypoint { x: 0.0, y: 0.0 },
                ordinary,
            ),
            Err(ProjectionLinearizationError::InvalidDepth)
        ));
        assert!(matches!(
            reprojection_linearization(
                Pose::identity(),
                Point3::new(0.0, 0.0, 0.5 * MIN_PROJECTION_DEPTH_M),
                Keypoint { x: 0.0, y: 0.0 },
                ordinary,
            ),
            Err(ProjectionLinearizationError::InvalidDepth)
        ));

        let extreme =
            make_pinhole_intrinsics(640, 480, f32::MAX, 1.0, 0.0, 0.0).expect("finite intrinsics");
        assert!(matches!(
            reprojection_linearization(
                Pose::identity(),
                Point3::new(2.0, 0.0, 1.0),
                Keypoint { x: 0.0, y: 0.0 },
                extreme,
            ),
            Err(ProjectionLinearizationError::NumericalFailure)
        ));
    }

    #[test]
    fn reprojection_linearization_preserves_ordinary_f32_bits() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let linearization = reprojection_linearization(
            Pose::identity(),
            Point3::new(0.25, -0.5, 2.0),
            Keypoint {
                x: 373.25,
                y: 134.75,
            },
            intrinsics,
        )
        .expect("ordinary factor");
        let expected = ReprojectionLinearization {
            residual: [0.75, -0.75],
            pose_jacobian: [
                [-210.0, -0.0, 26.25, -13.125, -426.5625, -105.0],
                [-0.0, -209.0, -52.25, 444.125, 13.0625, -52.25],
            ],
            landmark_jacobian: [[-210.0, -0.0, 26.25], [-0.0, -209.0, -52.25]],
        };

        assert!(
            linearization.has_same_bits(&expected),
            "ordinary f32 result changed: {linearization:?}"
        );
    }

    #[test]
    fn widened_reprojection_recovers_transient_depth_derivative_overflow() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 1.0e20, 1.0, 0.0, 0.0).expect("intrinsics");
        let linearization = reprojection_linearization(
            Pose::identity(),
            Point3::new(1.0e20, 0.0, 1.0e20),
            Keypoint { x: 1.0e20, y: 0.0 },
            intrinsics,
        )
        .expect("widened derivative");

        assert_eq!(linearization.residual, [0.0, 0.0]);
        assert_eq!(
            linearization.pose_jacobian[0][2].to_bits(),
            1.0_f32.to_bits()
        );
        assert!((linearization.pose_jacobian[0][4] / -2.0e20 - 1.0).abs() < 1.0e-6);
        assert_eq!(
            linearization.landmark_jacobian[0][2].to_bits(),
            1.0_f32.to_bits()
        );
    }

    #[test]
    fn widened_reprojection_recovers_squared_inverse_depth_underflow() {
        let intrinsics = make_pinhole_intrinsics(640, 480, 1.0, 1.0, 0.0, 0.0).expect("intrinsics");
        let depth_m = 1.0e23_f32;
        let linearization = reprojection_linearization(
            Pose::identity(),
            Point3::new(depth_m, 0.0, depth_m),
            Keypoint { x: 1.0, y: 0.0 },
            intrinsics,
        )
        .expect("widened inverse depth");
        let expected_depth_derivative =
            narrow_finite_f32(1.0_f64 / f64::from(depth_m)).expect("representable derivative");

        assert_eq!(linearization.residual, [0.0, 0.0]);
        assert_eq!(
            linearization.pose_jacobian[0][2].to_bits(),
            expected_depth_derivative.to_bits()
        );
        assert_eq!(
            linearization.pose_jacobian[0][4].to_bits(),
            (-2.0_f32).to_bits()
        );
        assert_eq!(
            linearization.landmark_jacobian[0][2].to_bits(),
            expected_depth_derivative.to_bits()
        );
    }

    #[test]
    fn widened_reprojection_recovers_rotation_after_focal_depth_underflow() {
        let focal_px = 1.0e-30_f32;
        let intrinsics =
            make_pinhole_intrinsics(640, 480, focal_px, 1.0, 0.0, 0.0).expect("intrinsics");
        let linearization = reprojection_linearization(
            Pose::identity(),
            Point3::new(0.0, 0.0, 1.0e30),
            Keypoint { x: 0.0, y: 0.0 },
            intrinsics,
        )
        .expect("widened focal/depth derivative");

        assert_eq!(linearization.pose_jacobian[0][0], 0.0);
        assert_eq!(
            linearization.pose_jacobian[0][4].to_bits(),
            (-focal_px).to_bits()
        );
    }

    #[test]
    fn widened_reprojection_recovers_summed_landmark_underflow() {
        let tiny = f32::from_bits(1);
        let rotation = [[1.0, tiny, 0.0], [0.0, 1.0, 0.0], [0.0, tiny, 1.0]];
        let pose = Pose::try_from_rt(rotation, [-1.0, 0.0, 1.0]).expect("valid pose");
        let intrinsics = make_pinhole_intrinsics(640, 480, 0.5, 1.0, 0.0, 0.0).expect("intrinsics");

        let linearization = reprojection_linearization(
            pose,
            Point3::new(0.0, 0.0, 0.0),
            Keypoint { x: 0.0, y: 0.0 },
            intrinsics,
        )
        .expect("widened landmark underflow");

        assert_eq!(
            linearization.landmark_jacobian[0][1].to_bits(),
            (-tiny).to_bits()
        );
    }

    #[test]
    fn widened_reprojection_removes_spurious_pose_jacobian_subnormal() {
        let focal_px = f32::from_bits(0x1e80_0bd3);
        let camera_y_m = 2.584_314e-26_f32;
        let camera_z_m = 0.500_059_4_f32;
        let legacy = (focal_px * (1.0 / camera_z_m)) * camera_y_m;
        assert_eq!(legacy.to_bits(), 1);
        let expected = narrow_finite_f32(
            f64::from(focal_px) * (f64::from(camera_y_m) / f64::from(camera_z_m)),
        )
        .expect("representable widened derivative");
        assert_eq!(expected, 0.0);
        let intrinsics =
            make_pinhole_intrinsics(640, 480, focal_px, 1.0, 0.0, 0.0).expect("intrinsics");

        let linearization = reprojection_linearization(
            Pose::identity(),
            Point3::new(0.0, camera_y_m, camera_z_m),
            Keypoint { x: 0.0, y: 0.0 },
            intrinsics,
        )
        .expect("widened pose Jacobian");

        assert_eq!(linearization.pose_jacobian[0][5].to_bits(), 0);
    }

    #[test]
    fn widened_reprojection_recovers_re_amplified_subnormal_jacobian_product() {
        let inverse_sqrt_2 = 1.0 / 2.0_f32.sqrt();
        let inverse_sqrt_3 = 1.0 / 3.0_f32.sqrt();
        let inverse_sqrt_6 = 1.0 / 6.0_f32.sqrt();
        let rotation = [
            [inverse_sqrt_3; 3],
            [inverse_sqrt_2, -inverse_sqrt_2, 0.0],
            [inverse_sqrt_6, inverse_sqrt_6, -2.0 * inverse_sqrt_6],
        ];
        let camera_x_m = 1.0e-14_f32;
        let camera_z_m = 1.1e-6_f32;
        let pose = Pose::try_from_rt(rotation, [camera_x_m, 0.0, camera_z_m])
            .expect("valid proper rotation");
        let focal_px = 1.0e-28_f32;
        let intrinsics =
            make_pinhole_intrinsics(640, 480, focal_px, 1.0, 0.0, 0.0).expect("intrinsics");

        let rounded_product = focal_px * camera_x_m;
        assert!(rounded_product.is_subnormal());
        let inverse_depth = 1.0 / camera_z_m;
        let legacy_pose_derivative = rounded_product * (inverse_depth * inverse_depth);
        assert!(legacy_pose_derivative.is_normal());
        let widened_inverse_depth = 1.0 / f64::from(camera_z_m);
        let expected_pose_derivative = narrow_finite_f32(
            (f64::from(focal_px) * widened_inverse_depth)
                * (f64::from(camera_x_m) * widened_inverse_depth),
        )
        .expect("representable widened pose derivative");
        let relative_error = ((f64::from(legacy_pose_derivative)
            - f64::from(expected_pose_derivative))
            / f64::from(expected_pose_derivative))
        .abs();
        assert!(relative_error > F32_SUM_MAX_RELATIVE_ERROR);

        let linearization = reprojection_linearization(
            pose,
            Point3::new(0.0, 0.0, 0.0),
            Keypoint { x: 0.0, y: 0.0 },
            intrinsics,
        )
        .expect("widened re-amplified Jacobian product");

        assert_eq!(
            linearization.pose_jacobian[0][2].to_bits(),
            expected_pose_derivative.to_bits()
        );
    }

    #[test]
    fn widened_reprojection_recovers_projection_ratio_overflow() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, f32::from_bits(1), f32::from_bits(1), 0.0, 0.0)
                .expect("subnormal focal lengths remain supported");
        let depth_m = 1.000001e-6_f32;
        assert!(depth_m > MIN_PROJECTION_DEPTH_M);
        let linearization = reprojection_linearization(
            Pose::identity(),
            Point3::new(1.0e34, 0.0, depth_m),
            Keypoint { x: 0.0, y: 0.0 },
            intrinsics,
        )
        .expect("widened projection ratio");

        assert!(linearization.residual[0].is_finite());
        assert!(linearization.residual[0] < 0.0);
        assert!(
            linearization
                .pose_jacobian
                .iter()
                .flatten()
                .all(|value| value.is_finite())
        );
        assert!(
            linearization
                .landmark_jacobian
                .iter()
                .flatten()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn widened_projection_recovers_principal_point_absorbed_by_projected_coordinate() {
        let principal_x_px = 1.0e-5_f32;
        let focal_px = 512.0_f32;
        let legacy_u = focal_px + principal_x_px;
        assert_eq!(legacy_u, focal_px);
        let expected_residual = narrow_finite_f32(
            f64::from(focal_px) - (f64::from(focal_px) + f64::from(principal_x_px)),
        )
        .expect("representable widened residual");
        let intrinsics = make_pinhole_intrinsics(640, 480, focal_px, 1.0, principal_x_px, 0.0)
            .expect("intrinsics");

        let linearization = reprojection_linearization(
            Pose::identity(),
            Point3::new(1.0, 0.0, 1.0),
            Keypoint {
                x: focal_px,
                y: 0.0,
            },
            intrinsics,
        )
        .expect("widened principal-point absorption recovery");

        assert_ne!(expected_residual, 0.0);
        assert_eq!(
            linearization.residual[0].to_bits(),
            expected_residual.to_bits()
        );
    }

    #[test]
    fn widened_projection_recovers_partially_absorbed_principal_point() {
        let principal_x_px = 4.0e-5_f32;
        let focal_px = 512.0_f32;
        let observed_x_px = focal_px + principal_x_px;
        assert_ne!(observed_x_px, focal_px);
        let expected_residual = narrow_finite_f32(
            f64::from(observed_x_px) - (f64::from(focal_px) + f64::from(principal_x_px)),
        )
        .expect("representable widened residual");
        let intrinsics = make_pinhole_intrinsics(640, 480, focal_px, 1.0, principal_x_px, 0.0)
            .expect("intrinsics");

        let linearization = reprojection_linearization(
            Pose::identity(),
            Point3::new(1.0, 0.0, 1.0),
            Keypoint {
                x: observed_x_px,
                y: 0.0,
            },
            intrinsics,
        )
        .expect("widened partial-absorption recovery");

        assert_ne!(expected_residual, 0.0);
        assert_eq!(
            linearization.residual[0].to_bits(),
            expected_residual.to_bits()
        );
    }

    #[test]
    fn widened_projection_recovers_product_rounding_hidden_by_principal_point_cancellation() {
        let camera_x = f32::from_bits(1.0_f32.to_bits() - 1);
        let focal_px = 1.0e8_f32;
        let projected_x = focal_px * camera_x;
        let principal_x_px = -projected_x;
        assert_eq!(projected_x + principal_x_px, 0.0);
        let expected_residual = narrow_finite_f32(
            -(f64::from(focal_px) * f64::from(camera_x) + f64::from(principal_x_px)),
        )
        .expect("representable widened residual");
        let intrinsics = make_pinhole_intrinsics(640, 480, focal_px, 1.0, principal_x_px, 0.0)
            .expect("intrinsics");

        let linearization = reprojection_linearization(
            Pose::identity(),
            Point3::new(camera_x, 0.0, 1.0),
            Keypoint { x: 0.0, y: 0.0 },
            intrinsics,
        )
        .expect("widened principal-point cancellation recovery");

        assert_ne!(expected_residual, 0.0);
        assert_eq!(
            linearization.residual[0].to_bits(),
            expected_residual.to_bits()
        );
    }

    #[test]
    fn widened_transform_recovers_finite_camera_point_after_f32_cancellation() {
        let cosine = std::f32::consts::FRAC_1_SQRT_2;
        let rotation = [
            [cosine, cosine, 0.0],
            [-cosine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let pose = Pose::try_from_rt(rotation, [-f32::MAX, 0.0, 0.0]).expect("valid pose");
        let point = Point3::new(f32::MAX, f32::MAX, 1.0);
        assert!(
            !math::transform_point(rotation, pose.translation(), point.to_array())[0].is_finite()
        );
        let intrinsics = make_pinhole_intrinsics(640, 480, f32::MIN_POSITIVE, 1.0, 0.0, 0.0)
            .expect("intrinsics");

        let linearization =
            reprojection_linearization(pose, point, Keypoint { x: 0.0, y: 0.0 }, intrinsics)
                .expect("widened transform");

        assert!(linearization.residual.iter().all(|value| value.is_finite()));
        assert!(
            linearization
                .pose_jacobian
                .iter()
                .flatten()
                .chain(linearization.landmark_jacobian.iter().flatten())
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn compensated_sum_keeps_a_small_term_across_large_cancellation() {
        let small = -f64::from(3.0_f32.sqrt() * 0.5);
        let large = f64::from(2.0_f32.powi(99));

        assert_eq!(large + small - large, 0.0);
        assert_eq!(math::compensated_sum([large, small, -large]), small);
    }

    #[test]
    fn widened_transform_recovers_lateral_coordinate_erased_by_f32_cancellation() {
        let cosine = std::f32::consts::FRAC_1_SQRT_2;
        let rotation = [
            [cosine, cosine, 0.0],
            [-cosine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let pose = Pose::try_from_rt(rotation, [-70_710_680.0, 0.0, 0.0]).expect("valid pose");
        let point = Point3::new(1.0e8, 0.0, 1.0);
        let f32_camera = math::transform_point(rotation, pose.translation(), point.to_array());
        assert_eq!(f32_camera[0], 0.0);

        let widened_camera =
            transform_point_widened(rotation, pose.translation(), point.to_array())
                .expect("finite widened camera point");
        assert_ne!(widened_camera[0], 0.0);
        let expected_residual = narrow_finite_f32(-widened_camera[0] / widened_camera[2])
            .expect("representable widened residual");
        let intrinsics = make_pinhole_intrinsics(640, 480, 1.0, 1.0, 0.0, 0.0).expect("intrinsics");

        let linearization =
            reprojection_linearization(pose, point, Keypoint { x: 0.0, y: 0.0 }, intrinsics)
                .expect("widened cancellation recovery");

        assert_ne!(expected_residual, 0.0);
        assert_eq!(
            linearization.residual[0].to_bits(),
            expected_residual.to_bits()
        );
    }

    #[test]
    fn widened_transform_recovers_nonzero_severe_f32_cancellation() {
        let cosine = std::f32::consts::FRAC_1_SQRT_2;
        let rotation = [
            [cosine, cosine, 0.0],
            [-cosine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let pose = Pose::try_from_rt(rotation, [-70_710_672.0, 0.0, 0.0]).expect("valid pose");
        let point = Point3::new(1.0e8, 0.0, 1.0);
        let f32_camera = math::transform_point(rotation, pose.translation(), point.to_array());
        assert_eq!(f32_camera[0], 8.0);

        let widened_camera =
            transform_point_widened(rotation, pose.translation(), point.to_array())
                .expect("finite widened camera point");
        let expected_residual = narrow_finite_f32(-widened_camera[0] / widened_camera[2])
            .expect("representable widened residual");
        let legacy_residual = -f32_camera[0] / f32_camera[2];
        let intrinsics = make_pinhole_intrinsics(640, 480, 1.0, 1.0, 0.0, 0.0).expect("intrinsics");

        let linearization =
            reprojection_linearization(pose, point, Keypoint { x: 0.0, y: 0.0 }, intrinsics)
                .expect("widened cancellation recovery");

        assert_ne!(expected_residual.to_bits(), legacy_residual.to_bits());
        assert_eq!(
            linearization.residual[0].to_bits(),
            expected_residual.to_bits()
        );
    }

    #[test]
    fn downstream_fallback_uses_camera_point_before_transform_product_rounding() {
        let sine_cosine = std::f32::consts::FRAC_1_SQRT_2;
        let rotation = [
            [sine_cosine, -sine_cosine, 0.0],
            [sine_cosine, sine_cosine, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let pose = Pose::try_from_rt(rotation, [0.0; 3]).expect("valid pose");
        let point = Point3::new(1.1, 0.0, 1.0);
        let (f32_camera, widened_camera, transform_precision_risk) =
            transform_point_f32_with_risk(rotation, pose.translation(), point.to_array());
        assert!(!transform_precision_risk);
        let principal_x_px = -f32_camera[0];
        assert_eq!(f32_camera[0] + principal_x_px, 0.0);
        let expected_residual = narrow_finite_f32(-(widened_camera[0] + f64::from(principal_x_px)))
            .expect("representable widened residual");
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 1.0, 1.0, principal_x_px, 0.0).expect("intrinsics");

        let linearization =
            reprojection_linearization(pose, point, Keypoint { x: 0.0, y: 0.0 }, intrinsics)
                .expect("coherent downstream fallback");

        assert_ne!(expected_residual, 0.0);
        assert_eq!(
            linearization.residual[0].to_bits(),
            expected_residual.to_bits()
        );
    }

    #[test]
    fn widened_transform_recovers_cancellation_masked_by_translation() {
        let cosine = 3.0_f32.sqrt() * 0.5;
        let rotation = [[cosine, -0.5, 0.0], [0.5, cosine, 0.0], [0.0, 0.0, 1.0]];
        let pose = Pose::try_from_rt(rotation, [1.0, 0.0, 0.0]).expect("valid pose");
        let point_x = 1.1_f32;
        let rounded_product = cosine * point_x;
        let point = Point3::new(point_x, 2.0 * rounded_product, 1.0);
        let (f32_camera, widened_camera, transform_precision_risk) =
            transform_point_f32_with_risk(rotation, pose.translation(), point.to_array());
        assert!(transform_precision_risk);
        assert_eq!(f32_camera[0], 1.0);
        let focal_px = 1.0e8_f32;
        let expected_residual = narrow_finite_f32(
            f64::from(focal_px) - f64::from(focal_px) * (widened_camera[0] / widened_camera[2]),
        )
        .expect("representable widened residual");
        let intrinsics =
            make_pinhole_intrinsics(640, 480, focal_px, 1.0, 0.0, 0.0).expect("intrinsics");

        let linearization = reprojection_linearization(
            pose,
            point,
            Keypoint {
                x: focal_px,
                y: 0.0,
            },
            intrinsics,
        )
        .expect("coherent masked-cancellation fallback");

        assert_ne!(expected_residual, 0.0);
        assert_eq!(
            linearization.residual[0].to_bits(),
            expected_residual.to_bits()
        );
    }

    #[test]
    fn exact_transform_cancellation_is_not_a_precision_risk() {
        let sine_cosine = std::f32::consts::FRAC_1_SQRT_2;
        let rotation = [
            [sine_cosine, -sine_cosine, 0.0],
            [sine_cosine, sine_cosine, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let pose = Pose::try_from_rt(rotation, [0.0; 3]).expect("valid pose");
        let point: Point3 = Point3::new(1.0, 1.0, 1.0);

        let (f32_camera, widened_camera, transform_precision_risk) =
            transform_point_f32_with_risk(rotation, pose.translation(), point.to_array());

        assert_eq!(f32_camera[0], 0.0);
        assert_eq!(widened_camera[0], 0.0);
        assert!(!transform_precision_risk);
    }

    #[test]
    fn widened_transform_recovers_translation_absorbed_by_large_coordinate() {
        let pose =
            Pose::try_from_rt(Pose::identity().rotation(), [1.0, 0.0, 0.0]).expect("valid pose");
        let point = Point3::new(1.0e8, 0.0, 1.0);
        let (f32_camera, widened_camera, transform_precision_risk) =
            transform_point_f32_with_risk(pose.rotation(), pose.translation(), point.to_array());
        assert_eq!(f32_camera[0], 1.0e8);
        assert_eq!(widened_camera[0], 100_000_001.0);
        assert!(transform_precision_risk);
        let intrinsics = make_pinhole_intrinsics(640, 480, 1.0, 1.0, 0.0, 0.0).expect("intrinsics");

        let linearization =
            reprojection_linearization(pose, point, Keypoint { x: 1.0e8, y: 0.0 }, intrinsics)
                .expect("widened absorbed-translation recovery");

        assert_eq!(linearization.residual[0].to_bits(), (-1.0_f32).to_bits());
    }

    #[test]
    fn widened_transform_rejects_material_error_above_relative_fast_path_contract() {
        let rotation = [
            [0.002_435_259_8, -0.743_477_7, 0.668_756_25],
            [0.999_997, 0.001_810_566_8, -0.001_628_600_1],
            [0.0, 0.668_758_2, 0.743_479_9],
        ];
        let pose = Pose::try_from_rt(rotation, [5.267_919e-7, 0.0, 0.0]).expect("valid pose");
        let point = Point3::new(-1.231_799_3e-5, 1.027_565_4e-5, 1.068_088_9e-5);
        let (f32_camera, widened_camera, transform_precision_risk) =
            transform_point_f32_with_risk(rotation, pose.translation(), point.to_array());
        assert!(transform_precision_risk);
        let relative_error =
            ((f64::from(f32_camera[0]) - widened_camera[0]) / widened_camera[0]).abs();
        assert!(relative_error > 0.07);

        let focal_px = 1.0e8_f32;
        let expected_residual =
            narrow_finite_f32(-f64::from(focal_px) * (widened_camera[0] / widened_camera[2]))
                .expect("representable widened residual");
        let intrinsics =
            make_pinhole_intrinsics(640, 480, focal_px, 1.0, 0.0, 0.0).expect("intrinsics");

        let linearization =
            reprojection_linearization(pose, point, Keypoint { x: 0.0, y: 0.0 }, intrinsics)
                .expect("widened relative-error recovery");

        assert_eq!(
            linearization.residual[0].to_bits(),
            expected_residual.to_bits()
        );
    }

    #[test]
    fn widened_reprojection_recovers_landmark_jacobian_cancellation() {
        let cosine = std::f32::consts::FRAC_1_SQRT_2;
        let rotation = [
            [cosine, 0.0, cosine],
            [0.0, 1.0, 0.0],
            [-cosine, 0.0, cosine],
        ];
        let camera_x = f32::from_bits(1.0_f32.to_bits() - 1);
        let pose = Pose::try_from_rt(rotation, [camera_x, 0.0, 1.0]).expect("valid pose");
        let focal_px = 1.0e8_f32;
        let intrinsics =
            make_pinhole_intrinsics(640, 480, focal_px, 1.0, 0.0, 0.0).expect("intrinsics");

        let a1 = focal_px;
        let a3 = -focal_px * camera_x;
        let legacy_landmark_derivative = -(a1 * cosine + a3 * cosine);
        assert_eq!(legacy_landmark_derivative, -8.0);
        let expected_landmark_derivative = narrow_finite_f32(
            -f64::from(focal_px) * f64::from(cosine) * (1.0 - f64::from(camera_x)),
        )
        .expect("representable widened derivative");

        let linearization = reprojection_linearization(
            pose,
            Point3::new(0.0, 0.0, 0.0),
            Keypoint { x: 0.0, y: 0.0 },
            intrinsics,
        )
        .expect("widened landmark cancellation recovery");

        assert_ne!(
            expected_landmark_derivative.to_bits(),
            legacy_landmark_derivative.to_bits()
        );
        assert_eq!(
            linearization.landmark_jacobian[0][2].to_bits(),
            expected_landmark_derivative.to_bits()
        );
    }

    #[test]
    fn widened_transform_recovers_product_underflow_hidden_by_normal_sum() {
        let tiny = f32::from_bits(1);
        let rotation = [[1.0, -tiny, 0.0], [tiny, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let pose = Pose::try_from_rt(rotation, [f32::MIN_POSITIVE, 0.0, 0.0]).expect("valid pose");
        let point = Point3::new(0.0, 0.25, 1.0);
        let f32_camera = math::transform_point(rotation, pose.translation(), point.to_array());
        assert_eq!(f32_camera[0].to_bits(), f32::MIN_POSITIVE.to_bits());

        let focal_px = f32::MAX;
        let principal_x_px = -(focal_px * f32::MIN_POSITIVE);
        let legacy_residual = -(focal_px * (f32_camera[0] / f32_camera[2]) + principal_x_px);
        assert_eq!(legacy_residual, 0.0);
        let widened_camera =
            transform_point_widened(rotation, pose.translation(), point.to_array())
                .expect("finite widened camera point");
        let expected_residual = narrow_finite_f32(
            -(f64::from(focal_px) * (widened_camera[0] / widened_camera[2])
                + f64::from(principal_x_px)),
        )
        .expect("representable widened residual");
        let intrinsics = make_pinhole_intrinsics(640, 480, focal_px, 1.0, principal_x_px, 0.0)
            .expect("intrinsics");

        let linearization =
            reprojection_linearization(pose, point, Keypoint { x: 0.0, y: 0.0 }, intrinsics)
                .expect("widened product-underflow recovery");

        assert_ne!(expected_residual, 0.0);
        assert_eq!(
            linearization.residual[0].to_bits(),
            expected_residual.to_bits()
        );
    }

    #[test]
    fn widened_transform_keeps_sub_f32_coordinate_amplified_by_focal_length() {
        let cosine = std::f32::consts::FRAC_1_SQRT_2;
        let rotation = [
            [0.5, 0.5, cosine],
            [-cosine, cosine, 0.0],
            [-0.5, -0.5, cosine],
        ];
        let pose = Pose::try_from_rt(rotation, [0.0, 1.0, 1.0]).expect("valid pose");
        let point = Point3::new(f32::from_bits(1), 0.0, 0.0);
        let f32_camera = math::transform_point(rotation, pose.translation(), point.to_array());
        assert_eq!(f32_camera[0], 0.0);

        let widened_camera =
            transform_point_widened(rotation, pose.translation(), point.to_array())
                .expect("finite widened camera point");
        assert!(widened_camera[0] > 0.0);
        assert_eq!(widened_camera[0] as f32, 0.0);
        let expected_residual =
            narrow_finite_f32(-f64::from(f32::MAX) * (widened_camera[0] / widened_camera[2]))
                .expect("representable widened residual");
        let intrinsics =
            make_pinhole_intrinsics(640, 480, f32::MAX, 1.0, 0.0, 0.0).expect("intrinsics");

        let linearization =
            reprojection_linearization(pose, point, Keypoint { x: 0.0, y: 0.0 }, intrinsics)
                .expect("widened sub-f32 coordinate recovery");

        assert_ne!(expected_residual, 0.0);
        assert_eq!(
            linearization.residual[0].to_bits(),
            expected_residual.to_bits()
        );
    }

    #[test]
    fn widened_transform_recovers_positive_depth_erased_by_f32_cancellation() {
        let sine_cosine = std::f32::consts::FRAC_1_SQRT_2;
        let rotation = [
            [sine_cosine, 0.0, sine_cosine],
            [0.0, 1.0, 0.0],
            [-sine_cosine, 0.0, sine_cosine],
        ];
        let pose = Pose::try_from_rt(rotation, [0.0, 0.0, 70_710_680.0]).expect("valid pose");
        let point = Point3::new(1.0e8, 0.0, 0.0);
        let f32_camera = math::transform_point(rotation, pose.translation(), point.to_array());
        assert!(f32_camera[2] <= MIN_PROJECTION_DEPTH_M);
        let widened_camera =
            transform_point_widened(rotation, pose.translation(), point.to_array())
                .expect("finite widened camera point");
        assert!(widened_camera[2] > f64::from(MIN_PROJECTION_DEPTH_M));
        let intrinsics = make_pinhole_intrinsics(640, 480, 1.0, 1.0, 0.0, 0.0).expect("intrinsics");

        let linearization =
            reprojection_linearization(pose, point, Keypoint { x: 0.0, y: 0.0 }, intrinsics)
                .expect("widened positive depth");

        assert!(linearization.residual.iter().all(|value| value.is_finite()));
        assert!(
            linearization
                .pose_jacobian
                .iter()
                .flatten()
                .chain(linearization.landmark_jacobian.iter().flatten())
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn widened_projection_keeps_camera_coordinates_wider_than_f32() {
        let sine_cosine = std::f32::consts::FRAC_1_SQRT_2;
        let rotation = [
            [sine_cosine, sine_cosine, 0.0],
            [-sine_cosine, sine_cosine, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let pose = Pose::try_from_rt(rotation, [0.0; 3]).expect("valid pose");
        let point = Point3::new(f32::MAX, f32::MAX, f32::MAX);
        let f32_camera = math::transform_point(rotation, pose.translation(), point.to_array());
        assert!(!f32_camera[0].is_finite());
        let widened_camera =
            transform_point_widened(rotation, pose.translation(), point.to_array())
                .expect("finite widened camera point");
        assert!(widened_camera[0] > f64::from(f32::MAX));
        let intrinsics = make_pinhole_intrinsics(640, 480, 1.0, 1.0, 0.0, 0.0).expect("intrinsics");

        let linearization =
            reprojection_linearization(pose, point, Keypoint { x: 0.0, y: 0.0 }, intrinsics)
                .expect("representable factor from widened camera point");

        assert!(linearization.residual.iter().all(|value| value.is_finite()));
        assert!(
            linearization
                .pose_jacobian
                .iter()
                .flatten()
                .chain(linearization.landmark_jacobian.iter().flatten())
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn reprojection_jacobian_matches_finite_difference() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let pose = axis_angle_pose([0.1, -0.05, 0.2], [0.06, -0.04, 0.03]);
        let point = Point3 {
            x: 0.4,
            y: -0.2,
            z: 3.8,
        };
        let mut pixel = project_pixel(pose, point, intrinsics);
        pixel.x += 1.7;
        pixel.y -= 0.9;

        let obs = Observation::try_new(point, pixel).expect("observation");
        let linearization = reprojection_linearization(pose, obs.world(), obs.pixel(), intrinsics)
            .expect("jacobian");
        let jac = linearization.pose_jacobian;

        let eps = 1e-3_f32;
        for col in 0..6 {
            let mut delta_pos = [0.0_f32; 6];
            delta_pos[col] = eps;
            let mut delta_neg = [0.0_f32; 6];
            delta_neg[col] = -eps;

            let r_plus = projection_residual(
                apply_se3_delta(pose, delta_pos).expect("finite positive perturbation"),
                &obs,
                intrinsics,
            );
            let r_minus = projection_residual(
                apply_se3_delta(pose, delta_neg).expect("finite negative perturbation"),
                &obs,
                intrinsics,
            );
            let numeric = [
                (r_plus[0] - r_minus[0]) / (2.0 * eps),
                (r_plus[1] - r_minus[1]) / (2.0 * eps),
            ];

            let err0 = (numeric[0] - jac[0][col]).abs();
            let err1 = (numeric[1] - jac[1][col]).abs();
            let tol0 = 4e-2_f32 + 3e-4_f32 * numeric[0].abs().max(jac[0][col].abs());
            let tol1 = 4e-2_f32 + 3e-4_f32 * numeric[1].abs().max(jac[1][col].abs());
            assert!(
                err0 < tol0 && err1 < tol1,
                "jacobian mismatch col={col}: analytic=({}, {}), numeric=({}, {}), err=({}, {}), tol=({}, {})",
                jac[0][col],
                jac[1][col],
                numeric[0],
                numeric[1],
                err0,
                err1,
                tol0,
                tol1
            );
        }
    }

    #[test]
    fn landmark_reprojection_jacobian_matches_finite_difference() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let pose = axis_angle_pose([0.1, -0.05, 0.2], [0.06, -0.04, 0.03]);
        let point = Point3::new(0.4, -0.2, 3.8);
        let mut pixel = project_pixel(pose, point, intrinsics);
        pixel.x += 1.7;
        pixel.y -= 0.9;
        let linearization =
            reprojection_linearization(pose, point, pixel, intrinsics).expect("landmark Jacobian");
        let jacobian = linearization.landmark_jacobian;

        let epsilon_m = 1e-3_f32;
        for column in 0..3 {
            let mut positive = point.to_array();
            positive[column] += epsilon_m;
            let mut negative = point.to_array();
            negative[column] -= epsilon_m;
            let positive_residual =
                reprojection_linearization(pose, Point3::from_array(positive), pixel, intrinsics)
                    .expect("positive landmark perturbation")
                    .residual;
            let negative_residual =
                reprojection_linearization(pose, Point3::from_array(negative), pixel, intrinsics)
                    .expect("negative landmark perturbation")
                    .residual;
            let numeric = [
                (positive_residual[0] - negative_residual[0]) / (2.0 * epsilon_m),
                (positive_residual[1] - negative_residual[1]) / (2.0 * epsilon_m),
            ];

            for row in 0..2 {
                let error = (numeric[row] - jacobian[row][column]).abs();
                let tolerance =
                    4e-2_f32 + 3e-4_f32 * numeric[row].abs().max(jacobian[row][column].abs());
                assert!(
                    error < tolerance,
                    "landmark Jacobian mismatch row={row}, column={column}: analytic={}, numeric={}, error={error}, tolerance={tolerance}",
                    jacobian[row][column],
                    numeric[row]
                );
            }
        }
    }

    #[test]
    fn optimize_keyframe_window_refines_pose_and_landmarks_with_schur() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let true_pose_0 = Pose::identity();
        let true_pose_1 = axis_angle_pose([0.20, -0.02, 0.03], [0.0, 0.03, -0.01]);
        let noisy_pose_1 = apply_se3_delta(true_pose_1, [0.08, -0.03, 0.04, 0.015, -0.01, 0.008])
            .expect("finite test pose update");

        let points_true = vec![
            Point3 {
                x: -0.35,
                y: -0.25,
                z: 3.2,
            },
            Point3 {
                x: -0.10,
                y: -0.22,
                z: 3.5,
            },
            Point3 {
                x: 0.14,
                y: -0.20,
                z: 3.8,
            },
            Point3 {
                x: 0.32,
                y: -0.10,
                z: 3.4,
            },
            Point3 {
                x: -0.30,
                y: 0.10,
                z: 3.6,
            },
            Point3 {
                x: -0.08,
                y: 0.16,
                z: 4.0,
            },
            Point3 {
                x: 0.16,
                y: 0.12,
                z: 3.3,
            },
            Point3 {
                x: 0.34,
                y: 0.24,
                z: 3.9,
            },
        ];

        let mut keypoints_0 = Vec::with_capacity(points_true.len());
        let mut keypoints_1 = Vec::with_capacity(points_true.len());
        for &point in &points_true {
            keypoints_0.push(
                project_world_point(true_pose_0, point, intrinsics)
                    .expect("point visible in pose 0"),
            );
            keypoints_1.push(
                project_world_point(true_pose_1, point, intrinsics)
                    .expect("point visible in pose 1"),
            );
        }

        let detections_0 = make_detections(
            SensorId::StereoLeft,
            FrameId::new(500),
            640,
            480,
            keypoints_0,
        )
        .expect("detections 0");
        let detections_1 = make_detections(
            SensorId::StereoLeft,
            FrameId::new(501),
            640,
            480,
            keypoints_1,
        )
        .expect("detections 1");

        let mut map = SlamMap::new();
        let kf_0 = map
            .add_keyframe_from_detections(
                detections_0.as_ref(),
                Timestamp::from_nanos(1_000_000),
                crate::WorldToCamera::from_legacy_pose(true_pose_0),
            )
            .expect("insert keyframe 0");
        let kf_1 = map
            .add_keyframe_from_detections(
                detections_1.as_ref(),
                Timestamp::from_nanos(2_000_000),
                crate::WorldToCamera::from_legacy_pose(noisy_pose_1),
            )
            .expect("insert keyframe 1");

        for (idx, &point_true) in points_true.iter().enumerate() {
            let kp_0 = map.keyframe_keypoint(kf_0, idx).expect("kf0 keypoint");
            let kp_1 = map.keyframe_keypoint(kf_1, idx).expect("kf1 keypoint");
            let descriptor = detections_0.descriptors()[idx];
            let i = idx as f32;
            let noisy_point = Point3 {
                x: point_true.x + (i - 3.5) * 0.010,
                y: point_true.y - (i - 3.5) * 0.008,
                z: point_true.z + ((idx % 2) as f32 - 0.5) * 0.040,
            };
            let point_id = map
                .add_map_point(noisy_point, descriptor.quantize(), kp_0)
                .expect("insert map point");
            map.add_observation(point_id, kp_1)
                .expect("add shared observation");
        }

        assert_map_invariants(&map).expect("map invariants before BA");
        let before_pose_err = pose_distance(
            map.keyframe(kf_1).expect("kf1").pose().into_legacy_pose(),
            true_pose_1,
        );
        let before_landmark_err = mean_landmark_error(&map, kf_0, &points_true);

        let config = LocalBaConfig::new(5, 15, 4, 2.0, lm(1e-3)).expect("valid BA config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);
        let result = ba
            .optimize_keyframe_window(&mut map, &[kf_0, kf_1])
            .expect("valid full BA input");
        assert!(
            matches!(
                result,
                BaResult::Converged { .. } | BaResult::MaxIterations { .. }
            ),
            "full local BA should succeed, got {result:?}"
        );
        assert_map_invariants(&map).expect("map invariants after BA");

        let after_pose_err = pose_distance(
            map.keyframe(kf_1).expect("kf1").pose().into_legacy_pose(),
            true_pose_1,
        );
        let after_landmark_err = mean_landmark_error(&map, kf_0, &points_true);

        assert!(
            after_pose_err < before_pose_err,
            "pose error did not improve: before={before_pose_err}, after={after_pose_err}"
        );
        assert!(
            after_landmark_err < before_landmark_err,
            "landmark error did not improve: before={before_landmark_err}, after={after_landmark_err}"
        );
    }

    #[test]
    fn optimize_keyframe_window_propagates_camera_center_overflow() {
        let (mut map, intrinsics, kf_0, kf_1, _, _) = build_full_ba_fixture([0.0; 6]);
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let pose = Pose::try_from_rt(
            [[s, -s, 0.0], [s, s, 0.0], [0.0, 0.0, 1.0]],
            [f32::MAX, f32::MAX, 0.0],
        )
        .expect("valid extreme pose");
        map.set_keyframe_pose(kf_0, crate::WorldToCamera::from_legacy_pose(pose))
            .expect("set pose");
        let before = map.snapshot();
        let config = LocalBaConfig::new(5, 5, 4, 2.0, lm(1e-3)).expect("config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);

        assert!(matches!(
            ba.optimize_keyframe_window(&mut map, &[kf_0, kf_1]),
            Err(LocalBaError::Pose(
                crate::PoseError::InverseTranslationNotRepresentable { axis: 0, value }
            )) if value < -f64::from(f32::MAX)
        ));
        assert_eq!(map.snapshot(), before);
    }
}
