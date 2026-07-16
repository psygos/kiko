use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;

use crate::{
    Keypoint, Observation, PinholeIntrinsics, Point3, Pose,
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
    motion_prior_weight: f32,
    motion_prior_weight_squared: f64,
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
    NegativeMotionWeight { value: f32 },
    NonFiniteMotionWeight { value: f32 },
    MotionWeightTooSmall { value: f32 },
    MotionWeightTooLarge { value: f32 },
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
            LocalBaConfigError::NegativeMotionWeight { value } => {
                write!(f, "local BA motion prior weight must be >= 0 (got {value})")
            }
            LocalBaConfigError::NonFiniteMotionWeight { value } => write!(
                f,
                "local BA motion prior weight must be finite (got {value})"
            ),
            LocalBaConfigError::MotionWeightTooSmall { value } => write!(
                f,
                "positive local BA motion prior weight must have a nonzero f32 square (got {value})"
            ),
            LocalBaConfigError::MotionWeightTooLarge { value } => write!(
                f,
                "local BA motion prior weight must have a finite f32 square (got {value})"
            ),
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
        motion_prior_weight: f32,
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
        if !motion_prior_weight.is_finite() {
            return Err(LocalBaConfigError::NonFiniteMotionWeight {
                value: motion_prior_weight,
            });
        }
        if motion_prior_weight < 0.0 {
            return Err(LocalBaConfigError::NegativeMotionWeight {
                value: motion_prior_weight,
            });
        }
        let motion_prior_weight_squared =
            (motion_prior_weight as f64) * (motion_prior_weight as f64);
        if motion_prior_weight > 0.0 && (motion_prior_weight_squared as f32) == 0.0 {
            return Err(LocalBaConfigError::MotionWeightTooSmall {
                value: motion_prior_weight,
            });
        }
        if motion_prior_weight_squared > f32::MAX as f64 {
            return Err(LocalBaConfigError::MotionWeightTooLarge {
                value: motion_prior_weight,
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
            motion_prior_weight,
            motion_prior_weight_squared,
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

    pub fn motion_prior_weight(&self) -> f32 {
        self.motion_prior_weight
    }

    fn motion_prior_weight_squared(&self) -> f64 {
        self.motion_prior_weight_squared
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
    Pnp(crate::PnpError),
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
            Self::Pnp(err) => write!(f, "BA observation parsing failed: {err}"),
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
            Self::Pnp(err) => Some(err),
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

impl From<crate::PnpError> for LocalBaError {
    fn from(err: crate::PnpError) -> Self {
        Self::Pnp(err)
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
        intrinsics: PinholeIntrinsics,
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
            let observation = Observation::try_new(world, obs.pixel(), intrinsics)?;
            resolved.push(observation);
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

        let dim = frame_count * 6;
        let max_iters = self.config.max_iterations();
        let huber = self.config.huber_delta_px();
        let damping = self.config.lm().initial_lambda().max(MIN_POSE_DAMPING);
        let motion_weight_squared = self.config.motion_prior_weight_squared();

        for _ in 0..max_iters {
            let a = &mut self.a_buf[..dim * dim];
            let b = &mut self.b_buf[..dim];
            a.fill(0.0);
            b.fill(0.0);

            for (idx, frame) in self.frames.iter().enumerate() {
                let base = idx * 6;
                let resolved = frame.observations.resolve(
                    map,
                    self.intrinsics,
                    self.config.min_observations,
                )?;
                let mut usable_observations = 0_usize;
                for obs in resolved.observations() {
                    let linearization = match reprojection_linearization(
                        frame.pose,
                        obs.world(),
                        obs.pixel(),
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

            if motion_weight_squared > 0.0 && frame_count >= 2 {
                for i in 1..frame_count {
                    if !accumulate_motion_prior(
                        a,
                        b,
                        dim,
                        self.frames[i - 1].pose,
                        self.frames[i].pose,
                        (i - 1) * 6,
                        i * 6,
                        motion_weight_squared,
                    ) {
                        return Err(LocalBaError::NumericalFailure {
                            operation: "accumulating the motion prior",
                        });
                    }
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

        Ok(())
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
        let motion_weight_squared = self.config.motion_prior_weight_squared();
        let lm_config = self.config.lm();
        let initial_cost =
            match full_problem_cost(problem, self.intrinsics, huber, motion_weight_squared) {
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

            if motion_weight_squared > 0.0 && pose_count >= 2 {
                for i in 1..pose_count {
                    if !accumulate_motion_prior(
                        s,
                        rhs,
                        pose_dim,
                        problem.poses[i - 1].pose,
                        problem.poses[i].pose,
                        (i - 1) * 6,
                        i * 6,
                        motion_weight_squared,
                    ) {
                        return Ok(BaResult::Degenerate {
                            reason: DegenerateReason::NumericalFailure,
                        });
                    }
                }
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

            let candidate_cost =
                match full_problem_cost(problem, self.intrinsics, huber, motion_weight_squared) {
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
    motion_weight_squared: f64,
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

    if motion_weight_squared > 0.0 {
        for i in 1..problem.poses.len() {
            let prev = pose_to_vec(problem.poses[i - 1].pose);
            let curr = pose_to_vec(problem.poses[i].pose);
            for k in 0..6 {
                let d = (curr[k] as f64) - (prev[k] as f64);
                cost += 0.5 * motion_weight_squared * d * d;
            }
        }
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
    let pc = math::transform_point(rotation, pose.translation(), pw);
    if pc.into_iter().any(|value| !value.is_finite()) {
        return Err(ProjectionLinearizationError::NumericalFailure);
    }
    let x = pc[0];
    let y = pc[1];
    let z = pc[2];
    if z <= MIN_PROJECTION_DEPTH_M {
        return Err(ProjectionLinearizationError::InvalidDepth);
    }

    let u = intrinsics.fx() * (x / z) + intrinsics.cx();
    let v = intrinsics.fy() * (y / z) + intrinsics.cy();
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
    for col in 0..3 {
        landmark_jacobian[0][col] =
            a1 * rotation[0][col] + a2 * rotation[1][col] + a3 * rotation[2][col];
        landmark_jacobian[1][col] =
            b1 * rotation[0][col] + b2 * rotation[1][col] + b3 * rotation[2][col];
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

    ReprojectionLinearization::try_new(residual, pose_jacobian, landmark_jacobian)
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

fn pose_to_vec(pose: Pose) -> [f32; 6] {
    let t = pose.translation();
    let w = math::so3_log(pose.rotation());
    [t[0], t[1], t[2], w[0], w[1], w[2]]
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

/// Accumulate motion-prior terms for a consecutive pair of poses into the
/// normal equations (Hessian `hessian` and right-hand side `rhs`).
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

fn add_finite_f32(target: &mut f32, delta: f64) -> bool {
    let Some(sum) = narrow_finite_f32((*target as f64) + delta) else {
        return false;
    };
    *target = sum;
    true
}

#[allow(clippy::too_many_arguments)]
fn accumulate_motion_prior(
    hessian: &mut [f32],
    rhs: &mut [f32],
    dim: usize,
    prev_pose: Pose,
    curr_pose: Pose,
    base_prev: usize,
    base_curr: usize,
    weight_squared: f64,
) -> bool {
    let r_prev = pose_to_vec(prev_pose);
    let r_curr = pose_to_vec(curr_pose);
    for k in 0..6 {
        let residual = (r_curr[k] as f64) - (r_prev[k] as f64);
        let weighted_residual = residual * weight_squared;
        let prev = base_prev + k;
        let curr = base_curr + k;
        if !add_finite_f32(&mut rhs[prev], weighted_residual)
            || !add_finite_f32(&mut rhs[curr], -weighted_residual)
            || !add_finite_f32(&mut hessian[prev * dim + prev], weight_squared)
            || !add_finite_f32(&mut hessian[curr * dim + curr], weight_squared)
            || !add_finite_f32(&mut hessian[prev * dim + curr], -weight_squared)
            || !add_finite_f32(&mut hessian[curr * dim + prev], -weight_squared)
        {
            return false;
        }
    }
    true
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

    #[test]
    fn local_ba_config_rejects_invalid_values() {
        assert!(matches!(
            LocalBaConfig::new(0, 10, 4, 1.0, lm(1e-3), 0.0),
            Err(LocalBaConfigError::ZeroWindow)
        ));
        assert!(matches!(
            LocalBaConfig::new(usize::MAX, 10, 4, 1.0, lm(1e-3), 0.0),
            Err(LocalBaConfigError::WindowTooLarge { value: usize::MAX })
        ));
        assert!(matches!(
            LocalBaConfig::new(5, 0, 4, 1.0, lm(1e-3), 0.0),
            Err(LocalBaConfigError::ZeroIterations)
        ));
        assert!(matches!(
            LocalBaConfig::new(5, 10, 0, 1.0, lm(1e-3), 0.0),
            Err(LocalBaConfigError::ZeroObservations)
        ));
        assert!(matches!(
            LocalBaConfig::new(5, 10, 3, 1.0, lm(1e-3), 0.0),
            Err(LocalBaConfigError::TooFewObservations { .. })
        ));
        assert!(matches!(
            LocalBaConfig::new(5, 10, 4, 0.0, lm(1e-3), 0.0),
            Err(LocalBaConfigError::NonPositiveHuber { .. })
        ));
        assert!(matches!(
            LocalBaConfig::new(5, 10, 4, 1.0, lm(1e-3), -1.0),
            Err(LocalBaConfigError::NegativeMotionWeight { .. })
        ));
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                LocalBaConfig::new(5, 10, 4, 1.0, lm(1e-3), value),
                Err(LocalBaConfigError::NonFiniteMotionWeight { .. })
            ));
        }
        assert!(matches!(
            LocalBaConfig::new(5, 10, 4, 1.0, lm(1e-3), f32::MIN_POSITIVE),
            Err(LocalBaConfigError::MotionWeightTooSmall { .. })
        ));
        assert!(matches!(
            LocalBaConfig::new(5, 10, 4, 1.0, lm(1e-3), f32::MAX),
            Err(LocalBaConfigError::MotionWeightTooLarge { .. })
        ));

        let extreme_finite_weight = f32::MAX.sqrt() / 2.0;
        let config = LocalBaConfig::new(5, 10, 4, 1.0, lm(1e-3), extreme_finite_weight)
            .expect("a motion weight with a representable square");
        assert!(config.motion_prior_weight_squared().is_finite());
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
        let (map, intrinsics, keyframe_id, _, _, _) = build_full_ba_fixture([0.0; 6]);
        let config = LocalBaConfig::new(5, 5, 4, 2.0, lm(1e-3), 0.0).expect("valid BA config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);
        let min_required = ba.min_observations();
        let observation_count = map.keyframe(keyframe_id).expect("keyframe").len();
        let make_observation_set = || {
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
                ba.push_frame(&map, behind_camera, make_observation_set()),
                Err(LocalBaError::InsufficientUsableObservations { actual: 0, .. })
            ),
            "a frame with no usable reprojection factors must be rejected"
        );
        assert!(
            ba.frames.is_empty(),
            "a rejected frame must not remain in the optimization window"
        );

        assert!(
            ba.push_frame(&map, Pose::identity(), make_observation_set())
                .is_ok(),
            "a valid frame must optimize after the failed window is reset"
        );
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
        let config = LocalBaConfig::new(5, 5, 4, 2.0, lm(1e-3), 0.0).expect("config");
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
        let config = LocalBaConfig::new(5, 15, 4, 2.0, lm(1e-3), 0.0).expect("valid BA config");
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
        let config = LocalBaConfig::new(5, 5, 4, 2.0, lm(1e-3), 0.0).expect("valid config");
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
        let config = LocalBaConfig::new(5, 5, 4, 2.0, lm, 0.0).expect("valid config");
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
    fn optimize_full_converges_at_exact_stationary_solution() {
        let (mut map, intrinsics, kf_0, kf_1, _, points_true) = build_full_ba_fixture([0.0; 6]);
        for (index, point) in points_true.into_iter().enumerate() {
            let keypoint = map
                .keyframe_keypoint(kf_0, index)
                .expect("keyframe keypoint");
            let point_id = map
                .map_point_for_keypoint(keypoint)
                .expect("map lookup")
                .expect("map point");
            map.set_map_point_position(point_id, point)
                .expect("set exact landmark");
        }

        let config = LocalBaConfig::new(5, 5, 4, 2.0, lm(1e-3), 0.0).expect("valid config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);
        let mut problem = FullBaProblem::try_from_map(
            &map,
            &[kf_0, kf_1],
            ba.window_size(),
            ba.min_observations(),
        )
        .expect("full BA problem");
        assert_eq!(
            full_problem_cost(&problem, intrinsics, config.huber_delta_px(), 0.0),
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
        let config = LocalBaConfig::new(5, 1, 4, 2.0, lm(1e-3), 0.0).expect("valid BA config");
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
        let config = LocalBaConfig::new(5, 15, 4, 2.0, lm(1e-3), 0.0).expect("valid BA config");
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
        let config = LocalBaConfig::new(5, 30, 4, 2.0, lm(1e-3), 0.0).expect("valid BA config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);
        let mut problem = FullBaProblem::try_from_map(
            &map,
            &[kf_0, kf_1],
            ba.window_size(),
            ba.min_observations(),
        )
        .expect("full BA problem");
        let before = full_problem_cost(&problem, intrinsics, config.huber_delta_px(), 0.0)
            .expect("finite initial cost");
        let result = ba.optimize_full(&mut problem).expect("typed BA result");
        let after = full_problem_cost(&problem, intrinsics, config.huber_delta_px(), 0.0)
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
        let config = LocalBaConfig::new(5, 20, 4, 2.0, lm(1e-3), 0.0).expect("valid BA config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);
        let mut problem = FullBaProblem::try_from_map(
            &map,
            &[kf_0, kf_1],
            ba.window_size(),
            ba.min_observations(),
        )
        .expect("full BA problem");
        let before = full_problem_cost(&problem, intrinsics, config.huber_delta_px(), 0.0)
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
        let config = LocalBaConfig::new(5, 5, 4, 2.0, lm(1e-3), 0.0).expect("valid config");
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
            full_problem_cost(&problem, intrinsics, config.huber_delta_px(), 0.0),
            Err(ProjectionLinearizationError::InvalidDepth)
        );
    }

    #[test]
    fn optimize_full_rejects_and_restores_an_invalid_depth_trial() {
        let intrinsics = make_pinhole_intrinsics(640, 480, 1.0, 1.0, 0.0, 0.0).expect("intrinsics");
        let config = LocalBaConfig::new(5, 1, 4, 2.0, lm(1e-3), 0.0).expect("valid config");
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
        let initial_cost = full_problem_cost(&problem, intrinsics, config.huber_delta_px(), 0.0)
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
    fn full_problem_cost_widens_extreme_supported_motion_weight() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let motion_weight = f32::MAX.sqrt() / 2.0;
        let problem = FullBaProblem {
            poses: vec![
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: Pose::identity(),
                },
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: Pose::from_rt(Pose::identity().rotation(), [4.0, 0.0, 0.0]),
                },
            ],
            landmarks: Vec::new(),
            factors: Vec::new(),
            scale_anchor: None,
        };

        let motion_weight_squared = (motion_weight as f64) * (motion_weight as f64);
        let cost = full_problem_cost(&problem, intrinsics, 2.0, motion_weight_squared)
            .expect("finite motion-prior cost");
        assert!(cost.is_finite());
        assert!(cost > f32::MAX as f64);
    }

    #[test]
    fn optimize_full_preserves_metric_gauge_anchor_distance() {
        let (map, intrinsics, kf_0, kf_1, _, _) =
            build_full_ba_fixture([0.08, -0.03, 0.04, 0.015, -0.01, 0.008]);
        let config = LocalBaConfig::new(5, 15, 4, 2.0, lm(1e-3), 0.0).expect("valid config");
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

        let config = LocalBaConfig::new(5, 5, 4, 2.0, lm(1e-3), 0.0).expect("valid config");
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
    fn motion_prior_rhs_scales_with_weight_squared() {
        let prev = Pose::identity();
        let curr = Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [1.0, -2.0, 0.5],
        );
        let mut hessian_unit = vec![0.0; 12 * 12];
        let mut rhs_unit = vec![0.0; 12];
        assert!(accumulate_motion_prior(
            &mut hessian_unit,
            &mut rhs_unit,
            12,
            prev,
            curr,
            0,
            6,
            1.0,
        ));
        let mut hessian_half = vec![0.0; 12 * 12];
        let mut rhs_half = vec![0.0; 12];
        assert!(accumulate_motion_prior(
            &mut hessian_half,
            &mut rhs_half,
            12,
            prev,
            curr,
            0,
            6,
            0.25,
        ));

        for i in 0..12 {
            assert!((rhs_half[i] - 0.25 * rhs_unit[i]).abs() < 1e-7);
        }
        for i in 0..12 * 12 {
            assert!((hessian_half[i] - 0.25 * hessian_unit[i]).abs() < 1e-7);
        }
    }

    #[test]
    fn motion_prior_accumulation_stays_finite_for_extreme_supported_weight() {
        let weight = f32::MAX.sqrt() / 2.0;
        let config = LocalBaConfig::new(5, 5, 4, 2.0, lm(1e-3), weight).expect("valid config");
        let prev = Pose::identity();
        let curr = Pose::from_rt(Pose::identity().rotation(), [1.0, -1.0, 0.5]);
        let mut hessian = vec![0.0; 12 * 12];
        let mut rhs = vec![0.0; 12];

        assert!(accumulate_motion_prior(
            &mut hessian,
            &mut rhs,
            12,
            prev,
            curr,
            0,
            6,
            config.motion_prior_weight_squared(),
        ));
        assert!(hessian.iter().all(|value| value.is_finite()));
        assert!(rhs.iter().all(|value| value.is_finite()));
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
        let cost =
            full_problem_cost(&problem, intrinsics, 2.0, 0.0).expect("large finite robust cost");
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

        let obs = Observation::try_new(point, pixel, intrinsics).expect("observation");
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

        let config = LocalBaConfig::new(5, 15, 4, 2.0, lm(1e-3), 0.0).expect("valid BA config");
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
        let config = LocalBaConfig::new(5, 5, 4, 2.0, lm(1e-3), 0.0).expect("config");
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
