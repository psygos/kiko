use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;

use crate::{
    Keypoint, Observation, PinholeIntrinsics, Point3, Pose,
    map::{KeyframeId, KeyframeKeypoint, MapPointId, SlamMap},
    math,
};

/// Maximum SE3 parameter step for convergence detection.
const STEP_CONVERGENCE_THRESHOLD: f32 = 1e-4;
/// Relative cost tolerance for LM convergence.
const RELATIVE_COST_TOLERANCE: f64 = 1e-6;
/// Floor for cost magnitude to avoid division-near-zero in relative convergence.
const COST_FLOOR: f64 = 1e-12;
/// Minimum projected depth for valid reprojection.
const MIN_PROJECTION_DEPTH: f32 = 1e-6;
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

#[derive(Clone, Copy, Debug)]
pub struct LocalBaConfig {
    window: NonZeroUsize,
    max_iterations: NonZeroUsize,
    min_observations: NonZeroUsize,
    huber_delta_px: f32,
    lm: LmConfig,
    motion_prior_weight: f32,
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
    InitialLambdaBelowMin { initial: f32, min: f32 },
    InitialLambdaAboveMax { initial: f32, max: f32 },
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
            LmConfigError::InitialLambdaBelowMin { initial, min } => write!(
                f,
                "LM initial lambda must be >= min lambda (initial={initial}, min={min})"
            ),
            LmConfigError::InitialLambdaAboveMax { initial, max } => write!(
                f,
                "LM initial lambda must be <= max lambda (initial={initial}, max={max})"
            ),
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
        if initial_lambda < min_lambda {
            return Err(LmConfigError::InitialLambdaBelowMin {
                initial: initial_lambda,
                min: min_lambda,
            });
        }
        if initial_lambda > max_lambda {
            return Err(LmConfigError::InitialLambdaAboveMax {
                initial: initial_lambda,
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

    fn reject(&mut self, config: LmConfig) {
        self.lambda = (self.lambda * config.lambda_factor()).min(config.max_lambda());
    }

    fn step(&mut self, cost: f64, predicted_decrease: f64, config: LmConfig) -> LmAction {
        if !cost.is_finite() || !predicted_decrease.is_finite() || predicted_decrease <= 0.0 {
            self.reject(config);
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
            self.reject(config);
            LmAction::Reject
        }
    }
}

#[derive(Debug)]
pub enum LocalBaConfigError {
    ZeroWindow,
    ZeroIterations,
    ZeroObservations,
    TooFewObservations { min: usize },
    NonPositiveHuber { value: f32 },
    NegativeMotionWeight { value: f32 },
}

impl std::fmt::Display for LocalBaConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LocalBaConfigError::ZeroWindow => write!(f, "local BA window must be > 0"),
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
        if motion_prior_weight < 0.0 || !motion_prior_weight.is_finite() {
            return Err(LocalBaConfigError::NegativeMotionWeight {
                value: motion_prior_weight,
            });
        }
        Ok(Self {
            window,
            max_iterations,
            min_observations,
            huber_delta_px,
            lm,
            motion_prior_weight,
        })
    }

    pub fn window(&self) -> usize {
        self.window.get()
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
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct BaCost(f64);

impl BaCost {
    pub fn new(value: f64) -> Result<Self, BaCostError> {
        if !value.is_finite() {
            return Err(BaCostError::NonFinite { value });
        }
        if value < 0.0 {
            return Err(BaCostError::Negative { value });
        }
        Ok(Self(value))
    }

    pub fn get(self) -> f64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BaCostError {
    NonFinite { value: f64 },
    Negative { value: f64 },
}

impl std::fmt::Display for BaCostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite { value } => write!(f, "BA cost must be finite, got {value}"),
            Self::Negative { value } => write!(f, "BA cost must be non-negative, got {value}"),
        }
    }
}

impl std::error::Error for BaCostError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaTermination {
    Converged { iterations: NonZeroUsize },
    IterationLimit { iterations: NonZeroUsize },
}

impl BaTermination {
    pub fn iterations(self) -> NonZeroUsize {
        match self {
            Self::Converged { iterations } | Self::IterationLimit { iterations } => iterations,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BaOptimization {
    termination: BaTermination,
    accepted_steps: NonZeroUsize,
    final_cost: BaCost,
}

impl BaOptimization {
    pub(crate) fn new(
        termination: BaTermination,
        accepted_steps: NonZeroUsize,
        final_cost: BaCost,
    ) -> Result<Self, BaOutcomeError> {
        let iterations = termination.iterations();
        if accepted_steps > iterations {
            return Err(BaOutcomeError::AcceptedStepsExceedIterations {
                accepted_steps,
                iterations,
            });
        }
        Ok(Self {
            termination,
            accepted_steps,
            final_cost,
        })
    }

    pub fn termination(self) -> BaTermination {
        self.termination
    }

    pub fn accepted_steps(self) -> NonZeroUsize {
        self.accepted_steps
    }

    pub fn final_cost(self) -> BaCost {
        self.final_cost
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BaStall {
    attempted_iterations: NonZeroUsize,
    retained_cost: BaCost,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BaStationary {
    detected_at_iteration: NonZeroUsize,
    retained_cost: BaCost,
}

impl BaStationary {
    pub(crate) fn new(detected_at_iteration: NonZeroUsize, retained_cost: BaCost) -> Self {
        Self {
            detected_at_iteration,
            retained_cost,
        }
    }

    pub fn detected_at_iteration(self) -> NonZeroUsize {
        self.detected_at_iteration
    }

    pub fn retained_cost(self) -> BaCost {
        self.retained_cost
    }
}

impl BaStall {
    pub(crate) fn new(attempted_iterations: NonZeroUsize, retained_cost: BaCost) -> Self {
        Self {
            attempted_iterations,
            retained_cost,
        }
    }

    pub fn attempted_iterations(self) -> NonZeroUsize {
        self.attempted_iterations
    }

    pub fn retained_cost(self) -> BaCost {
        self.retained_cost
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaOutcomeError {
    AcceptedStepsExceedIterations {
        accepted_steps: NonZeroUsize,
        iterations: NonZeroUsize,
    },
}

impl std::fmt::Display for BaOutcomeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AcceptedStepsExceedIterations {
                accepted_steps,
                iterations,
            } => write!(
                f,
                "BA accepted steps ({accepted_steps}) exceed attempted iterations ({iterations})"
            ),
        }
    }
}

impl std::error::Error for BaOutcomeError {}

#[derive(Clone, Debug, PartialEq)]
pub enum BaResult {
    Optimized(BaOptimization),
    Stationary(BaStationary),
    Stalled(BaStall),
    Degenerate { reason: DegenerateReason },
}

impl BaResult {
    pub fn optimization(&self) -> Option<BaOptimization> {
        match self {
            Self::Optimized(optimization) => Some(*optimization),
            Self::Stationary(_) | Self::Stalled(_) | Self::Degenerate { .. } => None,
        }
    }

    pub fn is_applicable(&self) -> bool {
        matches!(self, Self::Optimized(_))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DegenerateReason {
    TooFewPoses {
        count: usize,
    },
    TooFewLandmarks {
        count: usize,
    },
    TooFewObservations {
        keyframe_id: KeyframeId,
        required: usize,
        actual: usize,
    },
    NoFactors,
    NonProjectableFactors {
        count: usize,
    },
}

#[derive(Debug)]
pub enum BaExecutionError {
    DuplicateKeyframe {
        keyframe_id: KeyframeId,
    },
    MissingKeyframe {
        keyframe_id: KeyframeId,
    },
    DuplicateLandmarkObservation {
        point_id: MapPointId,
        keyframe_id: KeyframeId,
    },
    MapLookup {
        keypoint: KeyframeKeypoint,
        source: crate::map::MapError,
    },
    WriteBack {
        source: crate::map::MapError,
    },
    LandmarkLinearSystem {
        iteration: usize,
        landmark_index: usize,
        source: Matrix3InverseError,
    },
    PoseLinearSystem {
        iteration: usize,
        source: LinearSolveError,
    },
    InvalidCost {
        stage: &'static str,
        iteration: usize,
        source: BaCostError,
    },
    InvalidOutcome {
        source: BaOutcomeError,
    },
    InvariantViolation {
        message: &'static str,
    },
}

impl std::fmt::Display for BaExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateKeyframe { keyframe_id } => {
                write!(f, "full BA window has duplicate keyframe {keyframe_id:?}")
            }
            Self::MissingKeyframe { keyframe_id } => {
                write!(
                    f,
                    "full BA window references missing keyframe {keyframe_id:?}"
                )
            }
            Self::DuplicateLandmarkObservation {
                point_id,
                keyframe_id,
            } => write!(
                f,
                "landmark {point_id:?} has duplicate observation in keyframe {keyframe_id:?}"
            ),
            Self::MapLookup { keypoint, source } => write!(
                f,
                "full BA failed to read keypoint {:?}:{}: {source}",
                keypoint.keyframe_id(),
                keypoint.index()
            ),
            Self::WriteBack { source } => write!(f, "full BA writeback failed: {source}"),
            Self::LandmarkLinearSystem {
                iteration,
                landmark_index,
                source,
            } => write!(
                f,
                "full BA landmark system {landmark_index} failed at iteration {iteration}: {source}"
            ),
            Self::PoseLinearSystem { iteration, source } => write!(
                f,
                "full BA pose system failed at iteration {iteration}: {source}"
            ),
            Self::InvalidCost {
                stage,
                iteration,
                source,
            } => write!(
                f,
                "full BA {stage} cost is invalid at iteration {iteration}: {source}"
            ),
            Self::InvalidOutcome { source } => {
                write!(f, "full BA produced an invalid outcome: {source}")
            }
            Self::InvariantViolation { message } => {
                write!(f, "full BA invariant violation: {message}")
            }
        }
    }
}

impl std::error::Error for BaExecutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::MapLookup { source, .. } | Self::WriteBack { source } => Some(source),
            Self::LandmarkLinearSystem { source, .. } => Some(source),
            Self::PoseLinearSystem { source, .. } => Some(source),
            Self::InvalidCost { source, .. } => Some(source),
            Self::InvalidOutcome { source } => Some(source),
            Self::DuplicateKeyframe { .. }
            | Self::MissingKeyframe { .. }
            | Self::DuplicateLandmarkObservation { .. }
            | Self::InvariantViolation { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearSolveError {
    ZeroDimension,
    DimensionOverflow { dimension: usize },
    MatrixLengthMismatch { expected: usize, actual: usize },
    RhsLengthMismatch { expected: usize, actual: usize },
    NonFiniteMatrix { index: usize },
    NonFiniteRhs { index: usize },
    SingularPivot { column: usize },
    NonFiniteSolution { index: usize },
}

impl std::fmt::Display for LinearSolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "linear system dimension must be non-zero"),
            Self::DimensionOverflow { dimension } => write!(
                f,
                "linear system matrix size overflows usize for dimension {dimension}"
            ),
            Self::MatrixLengthMismatch { expected, actual } => write!(
                f,
                "linear system matrix length mismatch: expected {expected}, got {actual}"
            ),
            Self::RhsLengthMismatch { expected, actual } => write!(
                f,
                "linear system RHS length mismatch: expected {expected}, got {actual}"
            ),
            Self::NonFiniteMatrix { index } => {
                write!(f, "linear system matrix entry {index} is non-finite")
            }
            Self::NonFiniteRhs { index } => {
                write!(f, "linear system RHS entry {index} is non-finite")
            }
            Self::SingularPivot { column } => {
                write!(f, "linear system has a singular pivot in column {column}")
            }
            Self::NonFiniteSolution { index } => {
                write!(f, "linear system solution entry {index} is non-finite")
            }
        }
    }
}

impl std::error::Error for LinearSolveError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Matrix3InverseError {
    NonFiniteInput {
        row: usize,
        column: usize,
        value: f32,
    },
    NonFiniteDeterminant {
        value: f64,
    },
    Singular {
        determinant: f64,
    },
    NonFiniteOutput {
        row: usize,
        column: usize,
        value: f64,
    },
}

impl std::fmt::Display for Matrix3InverseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteInput { row, column, value } => write!(
                f,
                "3x3 matrix input [{row},{column}] is non-finite: {value}"
            ),
            Self::NonFiniteDeterminant { value } => {
                write!(f, "3x3 matrix determinant is non-finite: {value}")
            }
            Self::Singular { determinant } => {
                write!(f, "3x3 matrix is singular (determinant={determinant})")
            }
            Self::NonFiniteOutput { row, column, value } => write!(
                f,
                "3x3 inverse output [{row},{column}] is non-finite: {value}"
            ),
        }
    }
}

impl std::error::Error for Matrix3InverseError {}

#[derive(Clone, Debug, PartialEq)]
pub struct BaCorrection {
    pub pose_deltas: Vec<(KeyframeId, [f32; 6])>,
    pub landmark_deltas: Vec<(MapPointId, [f32; 3])>,
    pub result: BaResult,
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

#[derive(Debug)]
pub enum ObservationResolveError {
    Map { source: crate::map::MapError },
    MissingAssociation { keypoint: KeyframeKeypoint },
    MissingMapPoint { point_id: MapPointId },
    Pnp { source: crate::PnpError },
}

impl std::fmt::Display for ObservationResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Map { source } => write!(f, "BA observation map lookup failed: {source}"),
            Self::MissingAssociation { keypoint } => write!(
                f,
                "BA observation keypoint {:?}:{} has no map-point association",
                keypoint.keyframe_id(),
                keypoint.index()
            ),
            Self::MissingMapPoint { point_id } => {
                write!(
                    f,
                    "BA observation references missing map point {point_id:?}"
                )
            }
            Self::Pnp { source } => write!(f, "BA observation geometry is invalid: {source}"),
        }
    }
}

impl std::error::Error for ObservationResolveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Map { source } => Some(source),
            Self::Pnp { source } => Some(source),
            Self::MissingAssociation { .. } | Self::MissingMapPoint { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoseBaTermination {
    Converged { iterations: NonZeroUsize },
    IterationLimit { iterations: NonZeroUsize },
}

#[derive(Clone, Copy, Debug)]
pub struct PoseBaRefinement {
    pose: Pose,
    termination: PoseBaTermination,
}

impl PoseBaRefinement {
    pub fn pose(self) -> Pose {
        self.pose
    }

    pub fn termination(self) -> PoseBaTermination {
        self.termination
    }
}

#[derive(Clone, Copy, Debug)]
pub enum PoseBaOutcome {
    Refined(PoseBaRefinement),
    InsufficientSupport,
}

#[derive(Debug)]
pub enum PoseBaError {
    Observation {
        source: ObservationResolveError,
    },
    NoProjectableFactors {
        iteration: usize,
    },
    LinearSolve {
        iteration: usize,
        source: LinearSolveError,
    },
    InvalidPose {
        iteration: usize,
        frame_index: usize,
        source: crate::Pose64Error,
    },
    InvariantViolation {
        message: &'static str,
    },
}

impl std::fmt::Display for PoseBaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Observation { source } => {
                write!(f, "pose-only BA observation resolution failed: {source}")
            }
            Self::NoProjectableFactors { iteration } => write!(
                f,
                "pose-only BA has no projectable factors at iteration {iteration}"
            ),
            Self::LinearSolve { iteration, source } => write!(
                f,
                "pose-only BA linear solve failed at iteration {iteration}: {source}"
            ),
            Self::InvalidPose {
                iteration,
                frame_index,
                source,
            } => write!(
                f,
                "pose-only BA produced an invalid pose for frame {frame_index} at iteration {iteration}: {source}"
            ),
            Self::InvariantViolation { message } => {
                write!(f, "pose-only BA invariant violation: {message}")
            }
        }
    }
}

impl std::error::Error for PoseBaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Observation { source } => Some(source),
            Self::LinearSolve { source, .. } => Some(source),
            Self::InvalidPose { source, .. } => Some(source),
            Self::NoProjectableFactors { .. } | Self::InvariantViolation { .. } => None,
        }
    }
}

impl From<ObservationResolveError> for PoseBaError {
    fn from(source: ObservationResolveError) -> Self {
        Self::Observation { source }
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

#[derive(Clone, Debug)]
pub struct ObservationSet {
    observations: Vec<MapObservation>,
}

impl ObservationSet {
    pub fn new(
        observations: Vec<MapObservation>,
        min_required: NonZeroUsize,
    ) -> Result<Self, ObservationSetError> {
        let actual = observations.len();
        Self::when_sufficient(observations, min_required).ok_or(ObservationSetError::TooFew {
            required: min_required.get(),
            actual,
        })
    }

    pub fn when_sufficient(
        observations: Vec<MapObservation>,
        min_required: NonZeroUsize,
    ) -> Option<Self> {
        if observations.len() < min_required.get() {
            return None;
        }
        Some(Self { observations })
    }

    pub fn observations(&self) -> &[MapObservation] {
        &self.observations
    }

    fn resolve(
        &self,
        map: &SlamMap,
        intrinsics: PinholeIntrinsics,
        min_required: NonZeroUsize,
    ) -> Result<Option<ResolvedObservationSet>, ObservationResolveError> {
        let mut resolved = Vec::with_capacity(self.observations.len());
        for obs in &self.observations {
            let keypoint_ref = obs.keyframe_keypoint();
            let point_id = map
                .map_point_for_keypoint(keypoint_ref)
                .map_err(|source| ObservationResolveError::Map { source })?
                .ok_or(ObservationResolveError::MissingAssociation {
                    keypoint: keypoint_ref,
                })?;
            let world = map
                .point(point_id)
                .ok_or(ObservationResolveError::MissingMapPoint { point_id })?
                .position();
            let observation = Observation::try_new(world, obs.pixel(), intrinsics)
                .map_err(|source| ObservationResolveError::Pnp { source })?;
            resolved.push(observation);
        }
        if resolved.len() < min_required.get() {
            return Ok(None);
        }
        Ok(Some(ResolvedObservationSet {
            observations: resolved,
        }))
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

#[derive(Clone, Debug)]
struct BaFrame {
    pose: Pose,
    observations: ObservationSet,
}

// ---------------------------------------------------------------------------
// VIO window and solver types
//
// Layout guarantees:
// 1. All frames in a VIO window carry NavState (no mixing with visual-only)
// 2. Every frame except the first has a PreintegratedImu from its predecessor
//    (encoded structurally: VioAnchor has no preintegration, VioSuccessor has one)
// 3. Gravity is immutable for the window lifetime
// 4. Each frame contributes STATE_DIM (15) scalar unknowns
// ---------------------------------------------------------------------------

/// Dimension of the inertial state per frame:
/// [translation(3), rotation(3), velocity(3), accel_bias(3), gyro_bias(3)]
#[cfg(feature = "vio")]
const VIO_STATE_DIM: usize = 15;
#[cfg(feature = "vio")]
const VIO_REPROJECTION_TRANSLATION_FD_STEP_M: f64 = 1e-4;
#[cfg(feature = "vio")]
const VIO_REPROJECTION_ROTATION_FD_STEP_RAD: f64 = 1e-4;
#[cfg(feature = "vio")]
const VIO_REPROJECTION_POSE_FD_STEPS: [f64; 6] = [
    VIO_REPROJECTION_TRANSLATION_FD_STEP_M,
    VIO_REPROJECTION_TRANSLATION_FD_STEP_M,
    VIO_REPROJECTION_TRANSLATION_FD_STEP_M,
    VIO_REPROJECTION_ROTATION_FD_STEP_RAD,
    VIO_REPROJECTION_ROTATION_FD_STEP_RAD,
    VIO_REPROJECTION_ROTATION_FD_STEP_RAD,
];
#[cfg(feature = "vio")]
const VIO_TRANSLATION_CONVERGENCE_TOLERANCE_M: f64 = 1e-6;
#[cfg(feature = "vio")]
const VIO_ROTATION_CONVERGENCE_TOLERANCE_RAD: f64 = 1e-6;
#[cfg(feature = "vio")]
const VIO_VELOCITY_CONVERGENCE_TOLERANCE_MPS: f64 = 1e-6;
#[cfg(feature = "vio")]
const VIO_ACCEL_BIAS_CONVERGENCE_TOLERANCE_MPS2: f64 = 1e-6;
#[cfg(feature = "vio")]
const VIO_GYRO_BIAS_CONVERGENCE_TOLERANCE_RADPS: f64 = 1e-6;
#[cfg(feature = "vio")]
const VIO_STATE_CONVERGENCE_TOLERANCES: [f64; VIO_STATE_DIM] = [
    VIO_TRANSLATION_CONVERGENCE_TOLERANCE_M,
    VIO_TRANSLATION_CONVERGENCE_TOLERANCE_M,
    VIO_TRANSLATION_CONVERGENCE_TOLERANCE_M,
    VIO_ROTATION_CONVERGENCE_TOLERANCE_RAD,
    VIO_ROTATION_CONVERGENCE_TOLERANCE_RAD,
    VIO_ROTATION_CONVERGENCE_TOLERANCE_RAD,
    VIO_VELOCITY_CONVERGENCE_TOLERANCE_MPS,
    VIO_VELOCITY_CONVERGENCE_TOLERANCE_MPS,
    VIO_VELOCITY_CONVERGENCE_TOLERANCE_MPS,
    VIO_ACCEL_BIAS_CONVERGENCE_TOLERANCE_MPS2,
    VIO_ACCEL_BIAS_CONVERGENCE_TOLERANCE_MPS2,
    VIO_ACCEL_BIAS_CONVERGENCE_TOLERANCE_MPS2,
    VIO_GYRO_BIAS_CONVERGENCE_TOLERANCE_RADPS,
    VIO_GYRO_BIAS_CONVERGENCE_TOLERANCE_RADPS,
    VIO_GYRO_BIAS_CONVERGENCE_TOLERANCE_RADPS,
];
#[cfg(feature = "vio")]
const VIO_RELATIVE_COST_CONVERGENCE_TOLERANCE: f64 = 1e-10;
#[cfg(feature = "vio")]
const VIO_RELATIVE_COST_SCALE_FLOOR: f64 = 1e-12;

/// The first frame in a VIO window. Has no predecessor, hence no preintegration.
/// Carries the velocity-anchor reference. The optional calibrated bias prior
/// is solve configuration and applies to every frame.
#[cfg(feature = "vio")]
#[derive(Clone, Debug)]
pub(crate) struct VioAnchor {
    pub(crate) state: crate::NavState,
    /// Visual constraints tied to this anchor, when available.
    pub(crate) observations: Option<ObservationSet>,
    pub(crate) anchor_velocity_odom_mps: [f64; 3],
}

/// A successor frame in a VIO window. Always has a PreintegratedImu from
/// its immediate predecessor — this is structural, not optional.
#[cfg(feature = "vio")]
#[derive(Clone, Debug)]
pub(crate) struct VioSuccessor {
    pub(crate) state: crate::NavState,
    /// Visual constraints for this frame. Some frames legitimately contribute
    /// only inertial continuity and carry no lawful visual observations.
    pub(crate) observations: Option<ObservationSet>,
    pub(crate) preintegrated: crate::PreintegratedImu,
}

/// Validated diagonal prior on an IMU bias mean. Accelerometer and gyroscope
/// information use their distinct inverse-variance units.
#[cfg(feature = "vio")]
#[derive(Clone, Debug)]
pub struct VioBiasPrior {
    accel_information_s4_per_m2: f64,
    gyro_information_s2_per_rad2: f64,
    bias: crate::ImuBias,
}

#[cfg(feature = "vio")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VioBiasPriorInformationQuantity {
    AccelerometerBiasS4PerM2,
    GyroscopeBiasS2PerRad2,
}

#[cfg(feature = "vio")]
impl std::fmt::Display for VioBiasPriorInformationQuantity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::AccelerometerBiasS4PerM2 => "accelerometer-bias information (s^4/m^2)",
            Self::GyroscopeBiasS2PerRad2 => "gyroscope-bias information (s^2/rad^2)",
        })
    }
}

#[cfg(feature = "vio")]
impl VioBiasPrior {
    pub fn new(
        accel_information_s4_per_m2: f64,
        gyro_information_s2_per_rad2: f64,
        bias: crate::ImuBias,
    ) -> Result<Self, VioSolveConfigError> {
        for (quantity, value) in [
            (
                VioBiasPriorInformationQuantity::AccelerometerBiasS4PerM2,
                accel_information_s4_per_m2,
            ),
            (
                VioBiasPriorInformationQuantity::GyroscopeBiasS2PerRad2,
                gyro_information_s2_per_rad2,
            ),
        ] {
            if !value.is_finite() {
                return Err(VioSolveConfigError::NonFiniteBiasPriorInformation { quantity, value });
            }
            if value <= 0.0 {
                return Err(VioSolveConfigError::NonPositiveBiasPriorInformation {
                    quantity,
                    value,
                });
            }
        }
        Ok(Self {
            accel_information_s4_per_m2,
            gyro_information_s2_per_rad2,
            bias,
        })
    }

    pub fn accel_information_s4_per_m2(&self) -> f64 {
        self.accel_information_s4_per_m2
    }

    pub fn gyro_information_s2_per_rad2(&self) -> f64 {
        self.gyro_information_s2_per_rad2
    }

    pub fn bias(&self) -> &crate::ImuBias {
        &self.bias
    }
}

/// Immutable configuration shared by every evaluation of one VIO solve.
#[cfg(feature = "vio")]
#[derive(Clone, Debug)]
pub struct VioSolveConfig {
    gravity: crate::Gravity,
    camera_from_body: crate::Pose64,
    intrinsics: PinholeIntrinsics,
    lm: LmConfig,
    max_iterations: NonZeroUsize,
    huber_delta_px: f64,
    /// Diagonal information for the velocity anchor on frame 0, in s^2/m^2.
    anchor_velocity_information_s2_per_m2: f64,
    /// Optional prior mean from inertial calibration, applied to every frame.
    /// The information strengths are explicit inputs and are not inferred from
    /// calibration covariance.
    calibrated_bias_prior: Option<VioBiasPrior>,
}

#[cfg(feature = "vio")]
#[derive(Debug)]
pub enum VioSolveConfigError {
    NonFiniteVelocityAnchorInformation {
        value: f64,
    },
    NegativeVelocityAnchorInformation {
        value: f64,
    },
    NonFiniteBiasPriorInformation {
        quantity: VioBiasPriorInformationQuantity,
        value: f64,
    },
    NonPositiveBiasPriorInformation {
        quantity: VioBiasPriorInformationQuantity,
        value: f64,
    },
    NonFiniteHuberDeltaPx {
        value: f64,
    },
    NonPositiveHuberDeltaPx {
        value: f64,
    },
}

#[cfg(feature = "vio")]
impl std::fmt::Display for VioSolveConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteVelocityAnchorInformation { value } => write!(
                f,
                "VIO velocity-anchor information must be finite s^2/m^2, got {value}"
            ),
            Self::NegativeVelocityAnchorInformation { value } => write!(
                f,
                "VIO velocity-anchor information must be >= 0 s^2/m^2, got {value}"
            ),
            Self::NonFiniteBiasPriorInformation { quantity, value } => {
                write!(f, "VIO {quantity} must be finite, got {value}")
            }
            Self::NonPositiveBiasPriorInformation { quantity, value } => {
                write!(f, "VIO {quantity} must be > 0, got {value}")
            }
            Self::NonFiniteHuberDeltaPx { value } => {
                write!(f, "VIO Huber delta must be finite pixels, got {value}")
            }
            Self::NonPositiveHuberDeltaPx { value } => {
                write!(f, "VIO Huber delta must be > 0 pixels, got {value}")
            }
        }
    }
}

#[cfg(feature = "vio")]
impl std::error::Error for VioSolveConfigError {}

#[cfg(feature = "vio")]
impl VioSolveConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gravity: crate::Gravity,
        camera_from_body: crate::Pose64,
        intrinsics: PinholeIntrinsics,
        lm: LmConfig,
        max_iterations: NonZeroUsize,
        huber_delta_px: f64,
        anchor_velocity_information_s2_per_m2: f64,
        calibrated_bias_prior: Option<VioBiasPrior>,
    ) -> Result<Self, VioSolveConfigError> {
        if !anchor_velocity_information_s2_per_m2.is_finite() {
            return Err(VioSolveConfigError::NonFiniteVelocityAnchorInformation {
                value: anchor_velocity_information_s2_per_m2,
            });
        }
        if anchor_velocity_information_s2_per_m2 < 0.0 {
            return Err(VioSolveConfigError::NegativeVelocityAnchorInformation {
                value: anchor_velocity_information_s2_per_m2,
            });
        }
        if !huber_delta_px.is_finite() {
            return Err(VioSolveConfigError::NonFiniteHuberDeltaPx {
                value: huber_delta_px,
            });
        }
        if huber_delta_px <= 0.0 {
            return Err(VioSolveConfigError::NonPositiveHuberDeltaPx {
                value: huber_delta_px,
            });
        }
        Ok(Self {
            gravity,
            camera_from_body,
            intrinsics,
            lm,
            max_iterations,
            huber_delta_px,
            anchor_velocity_information_s2_per_m2,
            calibrated_bias_prior,
        })
    }

    pub fn gravity(&self) -> crate::Gravity {
        self.gravity
    }

    pub fn has_calibrated_bias_prior(&self) -> bool {
        self.calibrated_bias_prior.is_some()
    }
}

/// A structurally complete VIO optimization window.
///
/// Layout guarantees:
/// - At least one frame (the anchor)
/// - Every successor has a PreintegratedImu from its predecessor
/// - All frames carry NavState (no visual-only frames)
/// - Each frame contributes `VIO_STATE_DIM` scalar unknowns
#[cfg(feature = "vio")]
#[derive(Clone, Debug)]
pub(crate) struct VioWindow {
    pub(crate) anchor: VioAnchor,
    pub(crate) successors: Vec<VioSuccessor>,
}

#[cfg(feature = "vio")]
impl VioWindow {
    /// Total number of frames in the window (anchor + successors).
    pub(crate) fn len(&self) -> usize {
        1 + self.successors.len()
    }

    fn observations(&self, frame_idx: usize) -> Option<&ObservationSet> {
        if frame_idx == 0 {
            self.anchor.observations.as_ref()
        } else {
            self.successors[frame_idx - 1].observations.as_ref()
        }
    }

    /// Iterate all navigation states in window order.
    fn states(&self) -> impl Iterator<Item = &crate::NavState> {
        std::iter::once(&self.anchor.state).chain(self.successors.iter().map(|s| &s.state))
    }
}

/// Per-factor objective contributions from a VIO BA evaluation.
#[cfg(feature = "vio")]
#[derive(Clone, Debug, Default)]
pub struct VioCostBreakdown {
    pub reprojection_cost: f64,
    pub imu_cost: f64,
    pub bias_random_walk_cost: f64,
    pub velocity_anchor_cost: f64,
    pub bias_prior_cost: f64,
}

#[cfg(feature = "vio")]
impl VioCostBreakdown {
    pub fn total_cost(&self) -> f64 {
        self.reprojection_cost
            + self.imu_cost
            + self.bias_random_walk_cost
            + self.velocity_anchor_cost
            + self.bias_prior_cost
    }
}

/// Criterion that terminated a converged VIO solve.
#[cfg(feature = "vio")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VioConvergenceCriterion {
    ComponentwiseStepAndRelativeCost,
}

#[cfg(feature = "vio")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VioSolveTermination {
    NotRequired,
    Converged { criterion: VioConvergenceCriterion },
    IterationLimit,
    StalledNoCostImprovement,
}

#[cfg(feature = "vio")]
impl VioSolveTermination {
    pub fn is_converged(self) -> bool {
        matches!(self, Self::Converged { .. })
    }
}

#[cfg(feature = "vio")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VioEvaluationStage {
    Initial,
    Candidate,
}

#[cfg(feature = "vio")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VioLinearizationQuantity {
    Hessian,
    RightHandSide,
}

#[cfg(feature = "vio")]
impl std::fmt::Display for VioLinearizationQuantity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hessian => f.write_str("Hessian"),
            Self::RightHandSide => f.write_str("right-hand side"),
        }
    }
}

#[cfg(feature = "vio")]
impl std::fmt::Display for VioEvaluationStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Initial => f.write_str("initial"),
            Self::Candidate => f.write_str("candidate"),
        }
    }
}

#[cfg(feature = "vio")]
#[derive(Clone, Copy)]
struct VioEvaluation {
    stage: VioEvaluationStage,
    iteration: usize,
}

#[cfg(feature = "vio")]
#[derive(Debug)]
pub enum VioSolveError {
    Observation {
        source: ObservationResolveError,
    },
    LinearSolve {
        iteration: usize,
        source: crate::DenseSolveError,
    },
    StateRetraction {
        iteration: usize,
        frame_index: usize,
        source: crate::NavStateError,
    },
    ImuFactor {
        stage: VioEvaluationStage,
        iteration: usize,
        successor_index: usize,
        source: crate::VioFactorError,
    },
    ImuJacobian {
        stage: VioEvaluationStage,
        iteration: usize,
        successor_index: usize,
        source: crate::ImuJacobianError,
    },
    BiasRandomWalkFactor {
        stage: VioEvaluationStage,
        iteration: usize,
        successor_index: usize,
        source: crate::VioFactorError,
    },
    ReprojectionFactorUnavailable {
        stage: VioEvaluationStage,
        iteration: usize,
        frame_index: usize,
        observation_index: usize,
    },
    ReprojectionJacobianRetraction {
        stage: VioEvaluationStage,
        iteration: usize,
        frame_index: usize,
        observation_index: usize,
        tangent_axis: usize,
        side: crate::FiniteDifferenceSide,
        source: crate::NavStateError,
    },
    ReprojectionJacobianUnavailable {
        stage: VioEvaluationStage,
        iteration: usize,
        frame_index: usize,
        observation_index: usize,
        tangent_axis: usize,
        side: crate::FiniteDifferenceSide,
    },
    NonFiniteReprojectionJacobian {
        stage: VioEvaluationStage,
        iteration: usize,
        frame_index: usize,
        observation_index: usize,
        residual_axis: usize,
        tangent_axis: usize,
        value: f64,
    },
    NonFiniteLinearization {
        stage: VioEvaluationStage,
        iteration: usize,
        quantity: VioLinearizationQuantity,
        index: usize,
        value: f64,
    },
    NonFiniteCost {
        stage: VioEvaluationStage,
        iteration: usize,
        value: f64,
    },
}

#[cfg(feature = "vio")]
impl std::fmt::Display for VioSolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Observation { source } => {
                write!(f, "VIO observation resolution failed: {source}")
            }
            Self::LinearSolve { iteration, source } => {
                write!(
                    f,
                    "VIO linear solve failed at iteration {iteration}: {source}"
                )
            }
            Self::StateRetraction {
                iteration,
                frame_index,
                source,
            } => write!(
                f,
                "VIO state retraction failed at iteration {iteration}, frame {frame_index}: {source}"
            ),
            Self::ImuFactor {
                stage,
                iteration,
                successor_index,
                source,
            } => write!(
                f,
                "VIO {stage} IMU factor {successor_index} failed at iteration {iteration}: {source}"
            ),
            Self::ImuJacobian {
                stage,
                iteration,
                successor_index,
                source,
            } => write!(
                f,
                "VIO {stage} IMU Jacobian {successor_index} failed at iteration {iteration}: {source}"
            ),
            Self::BiasRandomWalkFactor {
                stage,
                iteration,
                successor_index,
                source,
            } => write!(
                f,
                "VIO {stage} bias random-walk factor {successor_index} failed at iteration {iteration}: {source}"
            ),
            Self::ReprojectionFactorUnavailable {
                stage,
                iteration,
                frame_index,
                observation_index,
            } => write!(
                f,
                "VIO {stage} visual factor is nonprojectable at iteration {iteration}, frame {frame_index}, observation {observation_index}"
            ),
            Self::ReprojectionJacobianRetraction {
                stage,
                iteration,
                frame_index,
                observation_index,
                tangent_axis,
                side,
                source,
            } => write!(
                f,
                "VIO {stage} reprojection Jacobian retraction failed at iteration {iteration}, frame {frame_index}, observation {observation_index}, tangent axis {tangent_axis}, {side} side: {source}"
            ),
            Self::ReprojectionJacobianUnavailable {
                stage,
                iteration,
                frame_index,
                observation_index,
                tangent_axis,
                side,
            } => write!(
                f,
                "VIO {stage} reprojection Jacobian is unavailable at iteration {iteration}, frame {frame_index}, observation {observation_index}, tangent axis {tangent_axis}, {side} side"
            ),
            Self::NonFiniteReprojectionJacobian {
                stage,
                iteration,
                frame_index,
                observation_index,
                residual_axis,
                tangent_axis,
                value,
            } => write!(
                f,
                "VIO {stage} reprojection Jacobian entry ({residual_axis}, {tangent_axis}) at iteration {iteration}, frame {frame_index}, observation {observation_index} must be finite, got {value}"
            ),
            Self::NonFiniteLinearization {
                stage,
                iteration,
                quantity,
                index,
                value,
            } => write!(
                f,
                "VIO {stage} linearization {quantity}[{index}] at iteration {iteration} must be finite, got {value}"
            ),
            Self::NonFiniteCost {
                stage,
                iteration,
                value,
            } => write!(
                f,
                "VIO {stage} cost at iteration {iteration} must be finite, got {value}"
            ),
        }
    }
}

#[cfg(feature = "vio")]
impl std::error::Error for VioSolveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Observation { source } => Some(source),
            Self::LinearSolve { source, .. } => Some(source),
            Self::StateRetraction { source, .. } => Some(source),
            Self::ImuFactor { source, .. } => Some(source),
            Self::ImuJacobian { source, .. } => Some(source),
            Self::BiasRandomWalkFactor { source, .. } => Some(source),
            Self::ReprojectionJacobianRetraction { source, .. } => Some(source),
            Self::ReprojectionFactorUnavailable { .. }
            | Self::ReprojectionJacobianUnavailable { .. }
            | Self::NonFiniteReprojectionJacobian { .. }
            | Self::NonFiniteLinearization { .. }
            | Self::NonFiniteCost { .. } => None,
        }
    }
}

#[cfg(feature = "vio")]
impl VioSolveError {
    fn is_rejected_candidate_nonprojectability(&self) -> bool {
        matches!(
            self,
            Self::ReprojectionFactorUnavailable {
                stage: VioEvaluationStage::Candidate,
                ..
            } | Self::ReprojectionJacobianUnavailable {
                stage: VioEvaluationStage::Candidate,
                ..
            }
        )
    }
}

#[cfg(feature = "vio")]
impl From<ObservationResolveError> for VioSolveError {
    fn from(source: ObservationResolveError) -> Self {
        Self::Observation { source }
    }
}

/// Diagnostics and termination outcome from a VIO BA solve.
#[cfg(feature = "vio")]
#[derive(Clone, Debug)]
pub struct VioSolveResult {
    pub termination: VioSolveTermination,
    pub iterations: usize,
    pub accepted_steps: usize,
    pub rejected_steps: usize,
    /// Candidate steps rejected because an initially active visual factor, or
    /// one of its finite-difference perturbations, became nonprojectable.
    pub rejected_nonprojectable_candidate_steps: usize,
    pub final_cost: f64,
    pub cost_breakdown: VioCostBreakdown,
    /// Visual factors that were projectable in the initial state and retained
    /// for every objective evaluation. Each factor contributes two residuals.
    pub last_frame_active_visual_factor_count: usize,
    /// Resolved visual observations excluded before optimization because they
    /// were nonprojectable in the initial state.
    pub initially_excluded_nonprojectable_visual_factor_count: usize,
    /// IMU factors whose mixed-unit residual covariance received the explicit
    /// block-unit diagonal regularization reported by their information type.
    pub regularized_imu_residual_factor_count: usize,
    /// Bias random-walk factors whose raw accelerometer-bias variance was
    /// raised to the documented floor.
    pub floored_accel_bias_random_walk_factor_count: usize,
    /// Bias random-walk factors whose raw gyroscope-bias variance was raised
    /// to the documented floor.
    pub floored_gyro_bias_random_walk_factor_count: usize,
}

#[cfg(feature = "vio")]
impl VioSolveResult {
    pub fn has_improved_estimate(&self) -> bool {
        self.accepted_steps > 0
    }
}

/// The refined output for a single frame after VIO BA.
#[cfg(feature = "vio")]
#[derive(Clone, Debug)]
pub struct VioFrameEstimate {
    pub pose: Pose,
    pub nav_state: crate::NavState,
}

/// Input for adding an inertial frame to the BA. The preintegration is
/// mandatory — the type enforces that you cannot add a VIO frame without
/// the IMU measurement connecting it to its predecessor.
#[cfg(feature = "vio")]
#[derive(Debug)]
pub struct InertialFrameInput {
    nav_state: crate::NavState,
    observations: ObservationSet,
    preintegrated: crate::PreintegratedImu,
}

#[cfg(feature = "vio")]
impl InertialFrameInput {
    pub fn new(
        nav_state: crate::NavState,
        observations: ObservationSet,
        preintegrated: crate::PreintegratedImu,
    ) -> Self {
        Self {
            nav_state,
            observations,
            preintegrated,
        }
    }

    pub fn nav_state(&self) -> &crate::NavState {
        &self.nav_state
    }

    pub fn observations(&self) -> &ObservationSet {
        &self.observations
    }

    pub fn preintegrated(&self) -> &crate::PreintegratedImu {
        &self.preintegrated
    }
}

/// Input for the anchor frame (first frame in VIO window). No preintegration.
#[cfg(feature = "vio")]
#[derive(Debug)]
pub struct AnchorFrameInput {
    nav_state: crate::NavState,
    observations: ObservationSet,
}

#[cfg(feature = "vio")]
impl AnchorFrameInput {
    pub fn new(nav_state: crate::NavState, observations: ObservationSet) -> Self {
        Self {
            nav_state,
            observations,
        }
    }

    pub fn nav_state(&self) -> &crate::NavState {
        &self.nav_state
    }

    pub fn observations(&self) -> &ObservationSet {
        &self.observations
    }
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
    MapLookup {
        keypoint: KeyframeKeypoint,
        source: crate::map::MapError,
    },
    NoLandmarks,
    PoseHasTooFewObservations {
        keyframe_id: KeyframeId,
        required: usize,
        actual: usize,
    },
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
            FullBaBuildError::MapLookup { keypoint, source } => write!(
                f,
                "full BA failed to read keypoint {:?}:{}: {source}",
                keypoint.keyframe_id(),
                keypoint.index()
            ),
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
        }
    }
}

impl std::error::Error for FullBaBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::MapLookup { source, .. } => Some(source),
            Self::EmptyWindow
            | Self::DuplicateKeyframe { .. }
            | Self::MissingKeyframe { .. }
            | Self::TooFewKeyframes { .. }
            | Self::DuplicateLandmarkObservation { .. }
            | Self::NoLandmarks
            | Self::PoseHasTooFewObservations { .. } => None,
        }
    }
}

fn classify_full_ba_build_error(
    err: FullBaBuildError,
) -> Result<DegenerateReason, BaExecutionError> {
    match err {
        FullBaBuildError::EmptyWindow => Ok(DegenerateReason::TooFewPoses { count: 0 }),
        FullBaBuildError::TooFewKeyframes { actual, .. } => {
            Ok(DegenerateReason::TooFewPoses { count: actual })
        }
        FullBaBuildError::NoLandmarks => Ok(DegenerateReason::TooFewLandmarks { count: 0 }),
        FullBaBuildError::PoseHasTooFewObservations {
            keyframe_id,
            required,
            actual,
        } => Ok(DegenerateReason::TooFewObservations {
            keyframe_id,
            required,
            actual,
        }),
        FullBaBuildError::DuplicateKeyframe { keyframe_id } => {
            Err(BaExecutionError::DuplicateKeyframe { keyframe_id })
        }
        FullBaBuildError::MissingKeyframe { keyframe_id } => {
            Err(BaExecutionError::MissingKeyframe { keyframe_id })
        }
        FullBaBuildError::DuplicateLandmarkObservation {
            point_id,
            keyframe_id,
        } => Err(BaExecutionError::DuplicateLandmarkObservation {
            point_id,
            keyframe_id,
        }),
        FullBaBuildError::MapLookup { keypoint, source } => {
            Err(BaExecutionError::MapLookup { keypoint, source })
        }
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
struct ReprojectionFactor {
    pose: PoseVarIndex,
    landmark: LandmarkVarIndex,
    pixel: Keypoint,
}

#[derive(Debug)]
struct FullBaProblem {
    poses: Vec<PoseVariable>,
    landmarks: Vec<LandmarkVariable>,
    factors: Vec<ReprojectionFactor>,
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
                pose: entry.pose(),
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
            let mut local_observations = Vec::new();
            let mut seen_local_poses = HashSet::new();

            for &obs in point.observations() {
                let Some(&pose_idx) = pose_lookup.get(&obs.keyframe_id()) else {
                    continue;
                };
                if !seen_local_poses.insert(pose_idx) {
                    return Err(FullBaBuildError::DuplicateLandmarkObservation {
                        point_id,
                        keyframe_id: obs.keyframe_id(),
                    });
                }
                let pixel = map
                    .keypoint(obs)
                    .map_err(|source| FullBaBuildError::MapLookup {
                        keypoint: obs,
                        source,
                    })?;
                local_observations.push((pose_idx, pixel));
            }

            if local_observations.len() < MIN_LANDMARK_OBSERVATIONS {
                continue;
            }

            let landmark_idx = LandmarkVarIndex(landmarks.len());
            landmarks.push(LandmarkVariable {
                point_id,
                position: point.position(),
            });

            for (pose_idx, pixel) in local_observations {
                pose_counts[pose_idx.as_usize()] += 1;
                factors.push(ReprojectionFactor {
                    pose: pose_idx,
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

        Ok(Self {
            poses,
            landmarks,
            factors,
        })
    }

    fn write_back(&self, map: &mut SlamMap) -> Result<(), crate::map::MapError> {
        let pose_updates = self
            .poses
            .iter()
            .map(|pose| (pose.keyframe_id, pose.pose))
            .collect::<Vec<_>>();
        let point_updates = self
            .landmarks
            .iter()
            .map(|landmark| (landmark.point_id, landmark.position))
            .collect::<Vec<_>>();
        map.apply_geometry_updates(&pose_updates, &point_updates)
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

#[derive(Clone, Debug)]
pub struct LocalBundleAdjuster {
    config: LocalBaConfig,
    intrinsics: PinholeIntrinsics,
    frames: Vec<BaFrame>,
    pose_backup_buf: Vec<Pose>,
    a_buf: Vec<f32>,
    b_buf: Vec<f32>,
}

impl LocalBundleAdjuster {
    pub fn new(intrinsics: PinholeIntrinsics, config: LocalBaConfig) -> Self {
        let dim = config.window().saturating_mul(6);
        let a_buf = vec![0.0_f32; dim * dim];
        let b_buf = vec![0.0_f32; dim];
        Self {
            config,
            intrinsics,
            frames: Vec::new(),
            pose_backup_buf: Vec::with_capacity(config.window()),
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
    ) -> Result<PoseBaOutcome, PoseBaError> {
        let retained_start = self
            .frames
            .len()
            .saturating_add(1)
            .saturating_sub(self.config.window());
        let mut resolved = Vec::with_capacity(
            self.frames
                .len()
                .saturating_add(1)
                .min(self.config.window()),
        );
        for frame in self.frames.iter().skip(retained_start) {
            let Some(frame_observations) =
                frame
                    .observations
                    .resolve(map, self.intrinsics, self.config.min_observations)?
            else {
                return Ok(PoseBaOutcome::InsufficientSupport);
            };
            resolved.push(frame_observations);
        }
        let Some(frame_observations) =
            observations.resolve(map, self.intrinsics, self.config.min_observations)?
        else {
            return Ok(PoseBaOutcome::InsufficientSupport);
        };
        resolved.push(frame_observations);

        debug_assert!(self.frames.len() <= self.config.window());
        let evicted = (self.frames.len() == self.config.window()).then(|| self.frames.remove(0));
        self.pose_backup_buf.clear();
        self.pose_backup_buf
            .extend(self.frames.iter().map(|frame| frame.pose));
        self.frames.push(BaFrame { pose, observations });

        let termination = match self.optimize(&resolved) {
            Ok(termination) => termination,
            Err(error) => {
                self.rollback_pose_push(evicted);
                return Err(error);
            }
        };
        let Some(refined_pose) = self.frames.last().map(|frame| frame.pose) else {
            self.rollback_pose_push(evicted);
            return Err(PoseBaError::InvariantViolation {
                message: "successful solve left an empty frame window",
            });
        };
        Ok(PoseBaOutcome::Refined(PoseBaRefinement {
            pose: refined_pose,
            termination,
        }))
    }

    fn rollback_pose_push(&mut self, evicted: Option<BaFrame>) {
        self.frames.pop();
        for (frame, pose) in self
            .frames
            .iter_mut()
            .zip(self.pose_backup_buf.iter().copied())
        {
            frame.pose = pose;
        }
        if let Some(frame) = evicted {
            self.frames.insert(0, frame);
        }
    }

    fn optimize(
        &mut self,
        resolved: &[ResolvedObservationSet],
    ) -> Result<PoseBaTermination, PoseBaError> {
        let frame_count = self.frames.len();
        if frame_count == 0 {
            return Err(PoseBaError::InvariantViolation {
                message: "optimizer called with an empty frame window",
            });
        }
        debug_assert_eq!(frame_count, resolved.len());

        let dim = frame_count * 6;
        let max_iters = self.config.max_iterations();
        let huber = self.config.huber_delta_px();
        let damping = self.config.lm().initial_lambda().max(MIN_POSE_DAMPING);
        let motion_weight = self.config.motion_prior_weight();

        for iter in 0..max_iters {
            let a = &mut self.a_buf[..dim * dim];
            let b = &mut self.b_buf[..dim];
            a.fill(0.0);
            b.fill(0.0);

            let mut projectable_factors = 0usize;
            for (idx, (frame, observations)) in self.frames.iter().zip(resolved).enumerate() {
                let base = idx * 6;
                for obs in observations.observations() {
                    if let Some((residual, jac)) =
                        reprojection_residual_and_jacobian(frame.pose, obs, self.intrinsics)
                    {
                        projectable_factors = projectable_factors.saturating_add(1);
                        let r_norm = (residual[0] * residual[0] + residual[1] * residual[1]).sqrt();
                        let weight = huber_weight(r_norm, huber);
                        let scale = weight.sqrt();
                        let r0 = residual[0] * scale;
                        let r1 = residual[1] * scale;
                        let j = jac.map(|row| row.map(|v| v * scale));

                        for c in 0..6 {
                            let jr = j[0][c] * r0 + j[1][c] * r1;
                            b[base + c] -= jr;
                            for d in 0..6 {
                                let jt_j = j[0][c] * j[0][d] + j[1][c] * j[1][d];
                                a[(base + c) * dim + (base + d)] += jt_j;
                            }
                        }
                    }
                }
            }
            if projectable_factors == 0 {
                return Err(PoseBaError::NoProjectableFactors {
                    iteration: iter + 1,
                });
            }

            if motion_weight > 0.0 && frame_count >= 2 {
                for i in 1..frame_count {
                    accumulate_motion_prior(
                        a,
                        b,
                        dim,
                        self.frames[i - 1].pose,
                        self.frames[i].pose,
                        (i - 1) * 6,
                        i * 6,
                        motion_weight,
                    );
                }
            }

            for i in 0..dim {
                a[i * dim + i] += damping;
            }

            solve_linear_system(a, b, dim).map_err(|source| PoseBaError::LinearSolve {
                iteration: iter + 1,
                source,
            })?;

            let mut max_step = 0.0_f32;
            for i in 0..frame_count {
                let step = extract_se3_delta(b, i * 6);
                let step_norm = norm6(step);
                if step_norm > max_step {
                    max_step = step_norm;
                }
                let pose = self.frames[i].pose;
                let updated = apply_se3_delta(pose, step);
                crate::Pose64::try_from_pose32(updated).map_err(|source| {
                    PoseBaError::InvalidPose {
                        iteration: iter + 1,
                        frame_index: i,
                        source,
                    }
                })?;
                self.frames[i].pose = updated;
            }

            if max_step < STEP_CONVERGENCE_THRESHOLD {
                return Ok(PoseBaTermination::Converged {
                    iterations: NonZeroUsize::new(iter + 1).ok_or(
                        PoseBaError::InvariantViolation {
                            message: "converged iteration count is zero",
                        },
                    )?,
                });
            }
        }

        Ok(PoseBaTermination::IterationLimit {
            iterations: self.config.max_iterations,
        })
    }

    pub fn optimize_keyframe_window(
        &mut self,
        map: &mut SlamMap,
        window: &[KeyframeId],
    ) -> Result<BaResult, BaExecutionError> {
        let mut problem = match FullBaProblem::try_from_map(
            map,
            window,
            self.config.window,
            self.config.min_observations,
        ) {
            Ok(problem) => problem,
            Err(err) => {
                return classify_full_ba_build_error(err)
                    .map(|reason| BaResult::Degenerate { reason });
            }
        };

        let result = self.optimize_full(&mut problem)?;
        if result.is_applicable() {
            problem
                .write_back(map)
                .map_err(|source| BaExecutionError::WriteBack { source })?;
        }

        Ok(result)
    }

    fn optimize_full(&mut self, problem: &mut FullBaProblem) -> Result<BaResult, BaExecutionError> {
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
        let motion_weight = self.config.motion_prior_weight();
        let lm_config = self.config.lm();
        let initial_cost = match full_problem_cost(problem, self.intrinsics, huber, motion_weight) {
            Ok(cost) => cost,
            Err(FullProblemCostError::NonProjectable { count }) => {
                return Ok(BaResult::Degenerate {
                    reason: DegenerateReason::NonProjectableFactors { count: count.get() },
                });
            }
            Err(FullProblemCostError::InvalidCost { source }) => {
                return Err(BaExecutionError::InvalidCost {
                    stage: "initial",
                    iteration: 0,
                    source,
                });
            }
        };
        let mut lm_state = LmState::new(lm_config, initial_cost.get());
        let mut accepted_steps = 0_usize;

        let mut pose_backup: Vec<Pose> = Vec::with_capacity(pose_count);
        let mut landmark_backup: Vec<Point3> = Vec::with_capacity(landmark_count);
        let mut landmark_accumulators: Vec<LandmarkAccumulator> =
            Vec::with_capacity(landmark_count);
        let mut pose_rhs_before_schur: Vec<f32> = vec![0.0; pose_dim];

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
                let pose_idx = factor.pose;
                let landmark_idx = factor.landmark;
                let pose = problem.poses[pose_idx.as_usize()].pose;
                let point = problem.landmarks[landmark_idx.as_usize()].position;

                let Some((residual, j_pose, j_landmark)) =
                    reprojection_residual_and_jacobians(pose, point, factor.pixel, self.intrinsics)
                else {
                    return Ok(BaResult::Degenerate {
                        reason: DegenerateReason::NonProjectableFactors { count: 1 },
                    });
                };

                let r_norm = (residual[0] * residual[0] + residual[1] * residual[1]).sqrt();
                let weight = huber_weight(r_norm, huber);
                let scale = weight.sqrt();

                let r_scaled = [residual[0] * scale, residual[1] * scale];
                let j_pose_scaled = j_pose.map(|row| row.map(|v| v * scale));
                let j_landmark_scaled = j_landmark.map(|row| row.map(|v| v * scale));

                accumulate_pose_hessian(s, pose_dim, pose_idx, j_pose_scaled);
                accumulate_pose_rhs(rhs, pose_idx, j_pose_scaled, r_scaled);

                let acc = &mut landmark_accumulators[landmark_idx.as_usize()];
                accumulate_landmark_hessian(&mut acc.c, j_landmark_scaled);
                accumulate_landmark_rhs(&mut acc.b, j_landmark_scaled, r_scaled);
                acc.add_link(
                    pose_idx,
                    pose_landmark_cross(j_pose_scaled, j_landmark_scaled),
                );
            }

            if motion_weight > 0.0 && pose_count >= 2 {
                for i in 1..pose_count {
                    accumulate_motion_prior(
                        s,
                        rhs,
                        pose_dim,
                        problem.poses[i - 1].pose,
                        problem.poses[i].pose,
                        (i - 1) * 6,
                        i * 6,
                        motion_weight,
                    );
                }
            }

            pose_rhs_before_schur.copy_from_slice(rhs);

            for i in 0..pose_dim {
                s[i * pose_dim + i] += pose_damping;
            }

            let mut schur_landmarks = Vec::with_capacity(landmark_count);
            for (landmark_index, acc) in landmark_accumulators.drain(..).enumerate() {
                let mut c = acc.c;
                for (i, c_row) in c.iter_mut().enumerate() {
                    c_row[i] += landmark_damping;
                }
                let inv_c =
                    invert_3x3(c).map_err(|source| BaExecutionError::LandmarkLinearSystem {
                        iteration: iter + 1,
                        landmark_index,
                        source,
                    })?;

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

            solve_linear_system(s, rhs, pose_dim).map_err(|source| {
                BaExecutionError::PoseLinearSystem {
                    iteration: iter + 1,
                    source,
                }
            })?;

            let mut predicted_decrease = 0.0_f64;
            let mut max_step = 0.0_f32;
            for (pose_i, pose_var) in problem.poses.iter_mut().enumerate() {
                let base = pose_i * 6;
                let delta = extract_se3_delta(rhs, base);
                max_step = max_step.max(norm6(delta));
                for k in 0..6 {
                    let d = delta[k] as f64;
                    let gradient = pose_rhs_before_schur[base + k] as f64;
                    predicted_decrease += 0.5 * d * ((pose_damping as f64) * d + gradient);
                }
                pose_var.pose = apply_se3_delta(pose_var.pose, delta);
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
                max_step = max_step.max(norm3(delta_landmark));
                for (axis, d) in delta_landmark.iter().enumerate() {
                    let d = *d as f64;
                    let gradient = schur.b[axis] as f64;
                    predicted_decrease += 0.5 * d * ((landmark_damping as f64) * d + gradient);
                }

                landmark_var.position.x += delta_landmark[0];
                landmark_var.position.y += delta_landmark[1];
                landmark_var.position.z += delta_landmark[2];
            }

            let prev_cost = lm_state.prev_cost();
            let (action, candidate_cost) =
                match full_problem_cost(problem, self.intrinsics, huber, motion_weight) {
                    Ok(cost) => (
                        lm_state.step(cost.get(), predicted_decrease, lm_config),
                        cost,
                    ),
                    Err(FullProblemCostError::NonProjectable { .. }) => {
                        lm_state.reject(lm_config);
                        (
                            LmAction::Reject,
                            BaCost::new(prev_cost).map_err(|source| {
                                BaExecutionError::InvalidCost {
                                    stage: "retained",
                                    iteration: iter + 1,
                                    source,
                                }
                            })?,
                        )
                    }
                    Err(FullProblemCostError::InvalidCost { source }) => {
                        restore_full_ba_candidate(
                            problem,
                            pose_backup.as_slice(),
                            landmark_backup.as_slice(),
                        );
                        return Err(BaExecutionError::InvalidCost {
                            stage: "candidate",
                            iteration: iter + 1,
                            source,
                        });
                    }
                };
            match action {
                LmAction::Accept => {
                    accepted_steps = accepted_steps.saturating_add(1);
                    let threshold = RELATIVE_COST_TOLERANCE * prev_cost.abs().max(COST_FLOOR);
                    if max_step < STEP_CONVERGENCE_THRESHOLD
                        || (prev_cost - candidate_cost.get()).abs() <= threshold
                    {
                        let iterations = nonzero_iteration_index(iter);
                        let accepted_steps = NonZeroUsize::new(accepted_steps).ok_or(
                            BaExecutionError::InvariantViolation {
                                message: "accepted LM action did not increment accepted-step count",
                            },
                        )?;
                        let optimization = BaOptimization::new(
                            BaTermination::Converged { iterations },
                            accepted_steps,
                            candidate_cost,
                        )
                        .map_err(|source| BaExecutionError::InvalidOutcome { source })?;
                        return Ok(BaResult::Optimized(optimization));
                    }
                }
                LmAction::Reject => {
                    restore_full_ba_candidate(
                        problem,
                        pose_backup.as_slice(),
                        landmark_backup.as_slice(),
                    );
                    if max_step <= f32::EPSILON {
                        let iterations = nonzero_iteration_index(iter);
                        let retained_cost = BaCost::new(prev_cost).map_err(|source| {
                            BaExecutionError::InvalidCost {
                                stage: "stationary",
                                iteration: iterations.get(),
                                source,
                            }
                        })?;
                        let Some(accepted_steps) = NonZeroUsize::new(accepted_steps) else {
                            return Ok(BaResult::Stationary(BaStationary::new(
                                iterations,
                                retained_cost,
                            )));
                        };
                        let optimization = BaOptimization::new(
                            BaTermination::Converged { iterations },
                            accepted_steps,
                            retained_cost,
                        )
                        .map_err(|source| BaExecutionError::InvalidOutcome { source })?;
                        return Ok(BaResult::Optimized(optimization));
                    }
                }
            }
        }

        let attempted_iterations = self.config.max_iterations;
        let final_cost =
            BaCost::new(lm_state.prev_cost()).map_err(|source| BaExecutionError::InvalidCost {
                stage: "final",
                iteration: max_iters,
                source,
            })?;
        let Some(accepted_steps) = NonZeroUsize::new(accepted_steps) else {
            return Ok(BaResult::Stalled(BaStall::new(
                attempted_iterations,
                final_cost,
            )));
        };
        let optimization = BaOptimization::new(
            BaTermination::IterationLimit {
                iterations: attempted_iterations,
            },
            accepted_steps,
            final_cost,
        )
        .map_err(|source| BaExecutionError::InvalidOutcome { source })?;
        Ok(BaResult::Optimized(optimization))
    }
}

fn nonzero_iteration_index(zero_based_iteration: usize) -> NonZeroUsize {
    NonZeroUsize::MIN.saturating_add(zero_based_iteration)
}

fn restore_full_ba_candidate(
    problem: &mut FullBaProblem,
    pose_backup: &[Pose],
    landmark_backup: &[Point3],
) {
    for (pose_var, pose) in problem.poses.iter_mut().zip(pose_backup.iter().copied()) {
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

// ---------------------------------------------------------------------------
// Tightly-coupled VIO optimizer
// ---------------------------------------------------------------------------

#[cfg(feature = "vio")]
fn vio_step_is_componentwise_small(delta: &[f64]) -> bool {
    let mut states = delta.chunks_exact(VIO_STATE_DIM);
    let within_tolerance = states.by_ref().all(|state_delta| {
        state_delta
            .iter()
            .zip(VIO_STATE_CONVERGENCE_TOLERANCES)
            .all(|(value, tolerance)| value.abs() < tolerance)
    });
    within_tolerance && states.remainder().is_empty()
}

/// Run Levenberg-Marquardt on a VIO window, jointly optimizing poses,
/// velocities, and IMU biases using reprojection + IMU + bias factors.
///
/// Mutates `window` in-place (states are retracted on accepted steps).
/// Returns the solve result.
///
/// Convention: we assemble gradient `g = J^T Ω r` and solve `H δ = -g`.
#[cfg(feature = "vio")]
pub fn optimize_vio(
    window: &mut VioWindow,
    config: &VioSolveConfig,
    map: &SlamMap,
    map_from_odom: &crate::MapFromOdom,
) -> Result<VioSolveResult, VioSolveError> {
    use crate::vio::solve::solve_dense_f64;

    let n_frames = window.len();
    let dim = n_frames * VIO_STATE_DIM;
    if n_frames < 2 {
        return Ok(VioSolveResult {
            termination: VioSolveTermination::NotRequired,
            iterations: 0,
            accepted_steps: 0,
            rejected_steps: 0,
            rejected_nonprojectable_candidate_steps: 0,
            final_cost: 0.0,
            cost_breakdown: VioCostBreakdown::default(),
            last_frame_active_visual_factor_count: 0,
            initially_excluded_nonprojectable_visual_factor_count: 0,
            regularized_imu_residual_factor_count: 0,
            floored_accel_bias_random_walk_factor_count: 0,
            floored_gyro_bias_random_walk_factor_count: 0,
        });
    }

    let max_iters = config.max_iterations.get();
    let mut lambda = f64::from(config.lm.initial_lambda);
    let mut states = window.states().cloned().collect::<Vec<_>>();
    let resolved_observations = (0..n_frames)
        .map(|frame_idx| match window.observations(frame_idx) {
            Some(observations) => observations.resolve(map, config.intrinsics, NonZeroUsize::MIN),
            None => Ok(None),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let visual_support = VisualFactorSupport::from_initial_states(
        &states,
        &resolved_observations,
        config,
        map_from_odom,
    );
    let mut linearization = linearize_vio_states(
        window,
        &states,
        &resolved_observations,
        &visual_support,
        config,
        map_from_odom,
        VioEvaluation {
            stage: VioEvaluationStage::Initial,
            iteration: 0,
        },
    )?;
    let mut current_cost = linearization.cost_breakdown.total_cost();
    if !current_cost.is_finite() {
        return Err(VioSolveError::NonFiniteCost {
            stage: VioEvaluationStage::Initial,
            iteration: 0,
            value: current_cost,
        });
    }
    let mut termination = None;
    let mut attempted_iterations = 0;
    let mut accepted_steps = 0;
    let mut rejected_steps = 0;
    let mut rejected_nonprojectable_candidate_steps = 0;

    for iteration in 0..max_iters {
        attempted_iterations = iteration + 1;

        let mut h_solve = linearization.hessian.clone();
        for i in 0..dim {
            h_solve[i * dim + i] += lambda;
        }
        let mut neg_g: Vec<f64> = linearization.rhs.iter().map(|v| -v).collect();
        solve_dense_f64(&mut h_solve, &mut neg_g, dim).map_err(|source| {
            VioSolveError::LinearSolve {
                iteration: attempted_iterations,
                source,
            }
        })?;
        let delta = neg_g;

        let mut candidate_states = Vec::with_capacity(n_frames);
        for (frame_idx, state) in states.iter().enumerate() {
            let base = frame_idx * VIO_STATE_DIM;
            let mut tangent = [0.0_f64; VIO_STATE_DIM];
            tangent.copy_from_slice(&delta[base..base + VIO_STATE_DIM]);
            let candidate_state =
                state
                    .retract(&tangent)
                    .map_err(|source| VioSolveError::StateRetraction {
                        iteration: attempted_iterations,
                        frame_index: frame_idx,
                        source,
                    })?;
            candidate_states.push(candidate_state);
        }

        let candidate_linearization = match linearize_vio_states(
            window,
            &candidate_states,
            &resolved_observations,
            &visual_support,
            config,
            map_from_odom,
            VioEvaluation {
                stage: VioEvaluationStage::Candidate,
                iteration: attempted_iterations,
            },
        ) {
            Ok(linearization) => linearization,
            Err(error) if error.is_rejected_candidate_nonprojectability() => {
                rejected_steps += 1;
                rejected_nonprojectable_candidate_steps += 1;
                lambda = (lambda * f64::from(config.lm.lambda_factor))
                    .min(f64::from(config.lm.max_lambda));
                continue;
            }
            Err(error) => return Err(error),
        };
        let candidate_cost = candidate_linearization.cost_breakdown.total_cost();
        if !candidate_cost.is_finite() {
            return Err(VioSolveError::NonFiniteCost {
                stage: VioEvaluationStage::Candidate,
                iteration: attempted_iterations,
                value: candidate_cost,
            });
        }
        if candidate_cost < current_cost {
            let cost_decrease = current_cost - candidate_cost;
            states = candidate_states;
            linearization = candidate_linearization;
            current_cost = candidate_cost;
            accepted_steps += 1;
            lambda =
                (lambda / f64::from(config.lm.lambda_factor)).max(f64::from(config.lm.min_lambda));
            let relative_cost_scale = current_cost.abs().max(VIO_RELATIVE_COST_SCALE_FLOOR);
            if vio_step_is_componentwise_small(&delta)
                && cost_decrease < VIO_RELATIVE_COST_CONVERGENCE_TOLERANCE * relative_cost_scale
            {
                termination = Some(VioSolveTermination::Converged {
                    criterion: VioConvergenceCriterion::ComponentwiseStepAndRelativeCost,
                });
                break;
            }
        } else {
            rejected_steps += 1;
            lambda =
                (lambda * f64::from(config.lm.lambda_factor)).min(f64::from(config.lm.max_lambda));
        }
    }

    window.anchor.state = states[0].clone();
    for (i, succ) in window.successors.iter_mut().enumerate() {
        succ.state = states[i + 1].clone();
    }

    let regularized_imu_residual_factor_count = window.successors.len();
    let floored_accel_bias_random_walk_factor_count = window
        .successors
        .iter()
        .filter(|successor| {
            successor
                .preintegrated
                .floored_bias_random_walk_information()
                .accel_variance_floor_applied()
        })
        .count();
    let floored_gyro_bias_random_walk_factor_count = window
        .successors
        .iter()
        .filter(|successor| {
            successor
                .preintegrated
                .floored_bias_random_walk_information()
                .gyro_variance_floor_applied()
        })
        .count();
    Ok(VioSolveResult {
        termination: termination.unwrap_or(if accepted_steps == 0 && rejected_steps > 0 {
            VioSolveTermination::StalledNoCostImprovement
        } else {
            VioSolveTermination::IterationLimit
        }),
        iterations: attempted_iterations,
        accepted_steps,
        rejected_steps,
        rejected_nonprojectable_candidate_steps,
        final_cost: current_cost,
        cost_breakdown: linearization.cost_breakdown,
        last_frame_active_visual_factor_count: visual_support.last_frame_factor_count(),
        initially_excluded_nonprojectable_visual_factor_count: visual_support
            .initially_excluded_nonprojectable_factor_count,
        regularized_imu_residual_factor_count,
        floored_accel_bias_random_walk_factor_count,
        floored_gyro_bias_random_walk_factor_count,
    })
}

#[cfg(feature = "vio")]
struct VioLinearization {
    hessian: Vec<f64>,
    rhs: Vec<f64>,
    cost_breakdown: VioCostBreakdown,
}

#[cfg(feature = "vio")]
struct VisualFactorSupport {
    observation_indices_by_frame: Vec<Vec<usize>>,
    initially_excluded_nonprojectable_factor_count: usize,
}

#[cfg(feature = "vio")]
impl VisualFactorSupport {
    fn from_initial_states(
        states: &[crate::NavState],
        resolved_observations: &[Option<ResolvedObservationSet>],
        config: &VioSolveConfig,
        map_from_odom: &crate::MapFromOdom,
    ) -> Self {
        debug_assert_eq!(states.len(), resolved_observations.len());
        let mut observation_indices_by_frame = Vec::with_capacity(states.len());
        let mut initially_excluded_nonprojectable_factor_count = 0;

        for (frame_index, state) in states.iter().enumerate() {
            let mut active_indices = Vec::new();
            if let Some(resolved) = &resolved_observations[frame_index] {
                active_indices.reserve(resolved.observations().len());
                let cam_from_map = vio_cam_from_map(state, config.camera_from_body, map_from_odom);
                for (observation_index, observation) in resolved.observations().iter().enumerate() {
                    if reprojection_residual_and_jacobians(
                        cam_from_map,
                        observation.world(),
                        observation.pixel(),
                        config.intrinsics,
                    )
                    .is_some()
                    {
                        active_indices.push(observation_index);
                    } else {
                        initially_excluded_nonprojectable_factor_count += 1;
                    }
                }
            }
            observation_indices_by_frame.push(active_indices);
        }

        Self {
            observation_indices_by_frame,
            initially_excluded_nonprojectable_factor_count,
        }
    }

    fn indices_for_frame(&self, frame_index: usize) -> &[usize] {
        &self.observation_indices_by_frame[frame_index]
    }

    fn last_frame_factor_count(&self) -> usize {
        self.observation_indices_by_frame.last().map_or(0, Vec::len)
    }
}

#[cfg(feature = "vio")]
fn linearize_vio_states(
    window: &VioWindow,
    states: &[crate::NavState],
    resolved_observations: &[Option<ResolvedObservationSet>],
    visual_support: &VisualFactorSupport,
    config: &VioSolveConfig,
    map_from_odom: &crate::MapFromOdom,
    evaluation: VioEvaluation,
) -> Result<VioLinearization, VioSolveError> {
    use crate::ImuFactor;
    use crate::vio::bias_random_walk_residual;
    use crate::vio::solve::{
        IMU_RESIDUAL_DIM, STATE_DIM, accumulate_cross_factor, accumulate_factor, imu_jacobians,
    };
    let VioEvaluation { stage, iteration } = evaluation;

    debug_assert_eq!(states.len(), window.len());
    let n_frames = window.len();
    debug_assert_eq!(resolved_observations.len(), n_frames);
    debug_assert_eq!(visual_support.observation_indices_by_frame.len(), n_frames);
    let dim = n_frames * STATE_DIM;
    let mut hessian = vec![0.0_f64; dim * dim];
    let mut rhs = vec![0.0_f64; dim];
    let mut cost_breakdown = VioCostBreakdown::default();

    for (frame_idx, state) in states.iter().enumerate() {
        let base = frame_idx * STATE_DIM;
        let cam_pose = vio_cam_from_map(state, config.camera_from_body, map_from_odom);
        if let Some(resolved) = &resolved_observations[frame_idx] {
            for &observation_index in visual_support.indices_for_frame(frame_idx) {
                let obs = &resolved.observations()[observation_index];
                let world = obs.world();
                let pixel = obs.pixel();
                let (residual, _, _) =
                    reprojection_residual_and_jacobians(cam_pose, world, pixel, config.intrinsics)
                        .ok_or(VioSolveError::ReprojectionFactorUnavailable {
                            stage,
                            iteration,
                            frame_index: frame_idx,
                            observation_index,
                        })?;
                let residual_f64 = [f64::from(residual[0]), f64::from(residual[1])];
                let r_norm = residual_f64[0].hypot(residual_f64[1]);
                let huber_delta = config.huber_delta_px;
                let (huber_weight, huber_cost) = if r_norm <= huber_delta {
                    (1.0, 0.5 * r_norm * r_norm)
                } else {
                    (
                        huber_delta / r_norm,
                        huber_delta * (r_norm - 0.5 * huber_delta),
                    )
                };
                cost_breakdown.reprojection_cost += huber_cost;
                let sqrt_w = huber_weight.sqrt();

                let mut j_15 = [[0.0_f64; STATE_DIM]; 2];
                for (axis, step) in VIO_REPROJECTION_POSE_FD_STEPS.into_iter().enumerate() {
                    let mut delta_plus = [0.0_f64; STATE_DIM];
                    let mut delta_minus = [0.0_f64; STATE_DIM];
                    delta_plus[axis] = step;
                    delta_minus[axis] = -step;
                    let s_plus = state.retract(&delta_plus).map_err(|source| {
                        VioSolveError::ReprojectionJacobianRetraction {
                            stage,
                            iteration,
                            frame_index: frame_idx,
                            observation_index,
                            tangent_axis: axis,
                            side: crate::FiniteDifferenceSide::Positive,
                            source,
                        }
                    })?;
                    let s_minus = state.retract(&delta_minus).map_err(|source| {
                        VioSolveError::ReprojectionJacobianRetraction {
                            stage,
                            iteration,
                            frame_index: frame_idx,
                            observation_index,
                            tangent_axis: axis,
                            side: crate::FiniteDifferenceSide::Negative,
                            source,
                        }
                    })?;
                    let p_plus = vio_cam_from_map(&s_plus, config.camera_from_body, map_from_odom);
                    let p_minus =
                        vio_cam_from_map(&s_minus, config.camera_from_body, map_from_odom);
                    let (r_plus, _, _) = reprojection_residual_and_jacobians(
                        p_plus,
                        world,
                        pixel,
                        config.intrinsics,
                    )
                    .ok_or(VioSolveError::ReprojectionJacobianUnavailable {
                        stage,
                        iteration,
                        frame_index: frame_idx,
                        observation_index,
                        tangent_axis: axis,
                        side: crate::FiniteDifferenceSide::Positive,
                    })?;
                    let (r_minus, _, _) = reprojection_residual_and_jacobians(
                        p_minus,
                        world,
                        pixel,
                        config.intrinsics,
                    )
                    .ok_or(VioSolveError::ReprojectionJacobianUnavailable {
                        stage,
                        iteration,
                        frame_index: frame_idx,
                        observation_index,
                        tangent_axis: axis,
                        side: crate::FiniteDifferenceSide::Negative,
                    })?;
                    for row in 0..2 {
                        let derivative =
                            f64::from(r_plus[row] - r_minus[row]) / (2.0 * step) * sqrt_w;
                        if !derivative.is_finite() {
                            return Err(VioSolveError::NonFiniteReprojectionJacobian {
                                stage,
                                iteration,
                                frame_index: frame_idx,
                                observation_index,
                                residual_axis: row,
                                tangent_axis: axis,
                                value: derivative,
                            });
                        }
                        j_15[row][axis] = derivative;
                    }
                }
                let r_f64 = [residual_f64[0] * sqrt_w, residual_f64[1] * sqrt_w];
                let identity_2 = [[1.0, 0.0], [0.0, 1.0]];
                accumulate_factor(
                    &mut hessian,
                    &mut rhs,
                    dim,
                    &j_15,
                    &identity_2,
                    &r_f64,
                    base,
                );
            }
        }
    }

    for succ_idx in 0..window.successors.len() {
        let prev_state = &states[succ_idx];
        let curr_state = &states[succ_idx + 1];
        let preint = &window.successors[succ_idx].preintegrated;
        let gravity = config.gravity();

        let residual =
            ImuFactor::residual(prev_state, curr_state, preint, &gravity).map_err(|source| {
                VioSolveError::ImuFactor {
                    stage,
                    iteration,
                    successor_index: succ_idx,
                    source,
                }
            })?;
        let info = preint.regularized_residual_information().matrix();
        let (j_prev, j_curr) =
            imu_jacobians(prev_state, curr_state, preint, gravity).map_err(|source| {
                VioSolveError::ImuJacobian {
                    stage,
                    iteration,
                    successor_index: succ_idx,
                    source,
                }
            })?;

        let mut imu_cost = 0.0;
        for i in 0..IMU_RESIDUAL_DIM {
            for j in 0..IMU_RESIDUAL_DIM {
                imu_cost += residual[i] * info[i][j] * residual[j];
            }
        }
        cost_breakdown.imu_cost += 0.5 * imu_cost;

        let base_prev = succ_idx * STATE_DIM;
        let base_curr = (succ_idx + 1) * STATE_DIM;
        accumulate_factor(
            &mut hessian,
            &mut rhs,
            dim,
            &j_prev,
            info,
            &residual,
            base_prev,
        );
        accumulate_factor(
            &mut hessian,
            &mut rhs,
            dim,
            &j_curr,
            info,
            &residual,
            base_curr,
        );
        accumulate_cross_factor(
            &mut hessian,
            dim,
            &j_prev,
            &j_curr,
            info,
            base_prev,
            base_curr,
        );
    }

    for succ_idx in 0..window.successors.len() {
        let prev_state = &states[succ_idx];
        let curr_state = &states[succ_idx + 1];
        let preint = &window.successors[succ_idx].preintegrated;
        let bias_residual =
            bias_random_walk_residual(prev_state, curr_state).map_err(|source| {
                VioSolveError::BiasRandomWalkFactor {
                    stage,
                    iteration,
                    successor_index: succ_idx,
                    source,
                }
            })?;
        let bias_information_diagonal = preint.floored_bias_random_walk_information().diagonal();

        let mut bias_cost = 0.0;
        let base_prev = succ_idx * STATE_DIM;
        let base_curr = (succ_idx + 1) * STATE_DIM;
        for (axis, (residual, information)) in bias_residual
            .into_iter()
            .zip(bias_information_diagonal)
            .enumerate()
        {
            let previous_index = base_prev + 9 + axis;
            let current_index = base_curr + 9 + axis;
            let weighted_residual = information * residual;
            bias_cost += residual * weighted_residual;
            hessian[previous_index * dim + previous_index] += information;
            hessian[current_index * dim + current_index] += information;
            hessian[previous_index * dim + current_index] -= information;
            hessian[current_index * dim + previous_index] -= information;
            rhs[previous_index] -= weighted_residual;
            rhs[current_index] += weighted_residual;
        }
        cost_breakdown.bias_random_walk_cost += 0.5 * bias_cost;
    }

    if config.anchor_velocity_information_s2_per_m2 > 0.0 {
        let anchor_state = &states[0];
        let anchor_velocity = anchor_state.velocity_odom_mps();
        let base = 0;
        for axis in 0..3 {
            let residual = anchor_velocity[axis] - window.anchor.anchor_velocity_odom_mps[axis];
            hessian[(base + 6 + axis) * dim + (base + 6 + axis)] +=
                config.anchor_velocity_information_s2_per_m2;
            rhs[base + 6 + axis] += config.anchor_velocity_information_s2_per_m2 * residual;
            cost_breakdown.velocity_anchor_cost +=
                0.5 * config.anchor_velocity_information_s2_per_m2 * residual * residual;
        }
    }

    if let Some(bias_prior) = config.calibrated_bias_prior.as_ref() {
        let accel_information_s4_per_m2 = bias_prior.accel_information_s4_per_m2();
        let gyro_information_s2_per_rad2 = bias_prior.gyro_information_s2_per_rad2();
        let calibrated_accel_mps2 = bias_prior.bias().accel_mps2();
        let calibrated_gyro_radps = bias_prior.bias().gyro_radps();
        for (frame_idx, state) in states.iter().enumerate() {
            let base = frame_idx * STATE_DIM;
            let bias = state.bias();
            let accel_mps2 = bias.accel_mps2();
            let gyro_radps = bias.gyro_radps();
            for axis in 0..3 {
                let accel_residual = accel_mps2[axis] - calibrated_accel_mps2[axis];
                hessian[(base + 9 + axis) * dim + (base + 9 + axis)] += accel_information_s4_per_m2;
                rhs[base + 9 + axis] += accel_information_s4_per_m2 * accel_residual;
                cost_breakdown.bias_prior_cost +=
                    0.5 * accel_information_s4_per_m2 * accel_residual * accel_residual;

                let gyro_residual = gyro_radps[axis] - calibrated_gyro_radps[axis];
                hessian[(base + 12 + axis) * dim + (base + 12 + axis)] +=
                    gyro_information_s2_per_rad2;
                rhs[base + 12 + axis] += gyro_information_s2_per_rad2 * gyro_residual;
                cost_breakdown.bias_prior_cost +=
                    0.5 * gyro_information_s2_per_rad2 * gyro_residual * gyro_residual;
            }
        }
    }

    let linearization = VioLinearization {
        hessian,
        rhs,
        cost_breakdown,
    };
    validate_vio_linearization_values(&linearization, evaluation)?;
    Ok(linearization)
}

#[cfg(feature = "vio")]
fn validate_vio_linearization_values(
    linearization: &VioLinearization,
    evaluation: VioEvaluation,
) -> Result<(), VioSolveError> {
    for (quantity, values) in [
        (VioLinearizationQuantity::Hessian, &linearization.hessian),
        (VioLinearizationQuantity::RightHandSide, &linearization.rhs),
    ] {
        for (index, value) in values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(VioSolveError::NonFiniteLinearization {
                    stage: evaluation.stage,
                    iteration: evaluation.iteration,
                    quantity,
                    index,
                    value,
                });
            }
        }
    }
    Ok(())
}

/// Compute camera-from-map pose from NavState (body-in-odom), extrinsics,
/// and the current map/odom bridge.
#[cfg(feature = "vio")]
fn vio_cam_from_map(
    state: &crate::NavState,
    camera_from_body: crate::Pose64,
    map_from_odom: &crate::MapFromOdom,
) -> Pose {
    let cam_from_odom = camera_from_body.compose(state.pose_odom_from_body().inverse());
    map_from_odom.odom_to_map(cam_from_odom).to_pose32()
}

#[derive(Debug)]
enum FullProblemCostError {
    NonProjectable { count: NonZeroUsize },
    InvalidCost { source: BaCostError },
}

fn full_problem_cost(
    problem: &FullBaProblem,
    intrinsics: PinholeIntrinsics,
    huber_delta_px: f32,
    motion_weight: f32,
) -> Result<BaCost, FullProblemCostError> {
    let mut cost = 0.0_f64;
    let huber = huber_delta_px as f64;
    let mut nonprojectable_count = 0_usize;

    for factor in &problem.factors {
        let pose = problem.poses[factor.pose.as_usize()].pose;
        let point = problem.landmarks[factor.landmark.as_usize()].position;
        let Some((residual, _, _)) =
            reprojection_residual_and_jacobians(pose, point, factor.pixel, intrinsics)
        else {
            nonprojectable_count = nonprojectable_count.saturating_add(1);
            continue;
        };
        let r0 = residual[0] as f64;
        let r1 = residual[1] as f64;
        let r_norm = (r0 * r0 + r1 * r1).sqrt();
        cost += if r_norm <= huber {
            0.5 * r_norm * r_norm
        } else {
            huber * (r_norm - 0.5 * huber)
        };
    }

    if motion_weight > 0.0 {
        let w2 = (motion_weight as f64) * (motion_weight as f64);
        for i in 1..problem.poses.len() {
            let delta = se3_delta_between(problem.poses[i - 1].pose, problem.poses[i].pose);
            for &value in &delta {
                let d = value as f64;
                cost += 0.5 * w2 * d * d;
            }
        }
    }

    match NonZeroUsize::new(nonprojectable_count) {
        Some(count) => Err(FullProblemCostError::NonProjectable { count }),
        None => BaCost::new(cost).map_err(|source| FullProblemCostError::InvalidCost { source }),
    }
}

pub(crate) fn reprojection_residual_and_jacobian(
    pose: Pose,
    obs: &Observation,
    intrinsics: PinholeIntrinsics,
) -> Option<([f32; 2], [[f32; 6]; 2])> {
    let (residual, jac_pose, _) =
        reprojection_residual_and_jacobians(pose, obs.world(), obs.pixel(), intrinsics)?;
    Some((residual, jac_pose))
}

type ReprojectionWithJacobians = ([f32; 2], [[f32; 6]; 2], [[f32; 3]; 2]);

fn reprojection_residual_and_jacobians(
    pose: Pose,
    world: Point3,
    pixel: Keypoint,
    intrinsics: PinholeIntrinsics,
) -> Option<ReprojectionWithJacobians> {
    let pw = [world.x, world.y, world.z];
    let rotation = pose.rotation();
    let pc = math::transform_point(rotation, pose.translation(), pw);
    let x = pc[0];
    let y = pc[1];
    let z = pc[2];
    if pc.iter().any(|value| !value.is_finite()) || z <= MIN_PROJECTION_DEPTH {
        return None;
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

    if residual.iter().any(|value| !value.is_finite())
        || [du_dx, du_dz, dv_dy, dv_dz]
            .iter()
            .any(|value| !value.is_finite())
    {
        return None;
    }

    let mut jac_pose = [[0.0_f32; 6]; 2];

    jac_pose[0][0] = a1;
    jac_pose[0][1] = a2;
    jac_pose[0][2] = a3;
    jac_pose[1][0] = b1;
    jac_pose[1][1] = b2;
    jac_pose[1][2] = b3;

    jac_pose[0][3] = -(a2 * z - a3 * y);
    jac_pose[0][4] = a1 * z - a3 * x;
    jac_pose[0][5] = -a1 * y + a2 * x;

    jac_pose[1][3] = -(b2 * z - b3 * y);
    jac_pose[1][4] = b1 * z - b3 * x;
    jac_pose[1][5] = -b1 * y + b2 * x;

    let mut jac_landmark = [[0.0_f32; 3]; 2];
    for col in 0..3 {
        jac_landmark[0][col] =
            a1 * rotation[0][col] + a2 * rotation[1][col] + a3 * rotation[2][col];
        jac_landmark[1][col] =
            b1 * rotation[0][col] + b2 * rotation[1][col] + b3 * rotation[2][col];
    }

    // The Jacobian above is for projected pixel coordinates [u, v].
    // Residual is defined as [pixel.x - u, pixel.y - v], so dr/dx = -du/dx.
    for row in &mut jac_pose {
        for value in row {
            *value = -*value;
        }
    }
    for row in &mut jac_landmark {
        for value in row {
            *value = -*value;
        }
    }

    if jac_pose
        .iter()
        .flat_map(|row| row.iter())
        .any(|value| !value.is_finite())
        || jac_landmark
            .iter()
            .flat_map(|row| row.iter())
            .any(|value| !value.is_finite())
    {
        return None;
    }

    Some((residual, jac_pose, jac_landmark))
}

pub(crate) fn apply_se3_delta(pose: Pose, delta: [f32; 6]) -> Pose {
    crate::math::se3_exp_f64(delta.map(f64::from))
        .compose(crate::Pose64::from_pose32(pose))
        .to_pose32()
}

pub(crate) fn se3_delta_between(from: Pose, to: Pose) -> [f32; 6] {
    crate::math::se3_log_f64(
        crate::Pose64::from_pose32(to).compose(crate::Pose64::from_pose32(from).inverse()),
    )
    .map(|value| value as f32)
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

/// Accumulate motion-prior terms for a consecutive pair of poses into the
/// normal equations (Hessian `hessian` and right-hand side `rhs`).
#[allow(clippy::too_many_arguments)]
fn accumulate_motion_prior(
    hessian: &mut [f32],
    rhs: &mut [f32],
    dim: usize,
    prev_pose: Pose,
    curr_pose: Pose,
    base_prev: usize,
    base_curr: usize,
    weight: f32,
) {
    let residual = se3_delta_between(prev_pose, curr_pose);
    let w2 = weight * weight;
    for k in 0..6 {
        let r = residual[k] * w2;
        rhs[base_prev + k] += r;
        rhs[base_curr + k] -= r;

        hessian[(base_prev + k) * dim + (base_prev + k)] += w2;
        hessian[(base_curr + k) * dim + (base_curr + k)] += w2;
        hessian[(base_prev + k) * dim + (base_curr + k)] -= w2;
        hessian[(base_curr + k) * dim + (base_prev + k)] -= w2;
    }
}

fn huber_weight(r_norm: f32, delta: f32) -> f32 {
    if r_norm <= delta { 1.0 } else { delta / r_norm }
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

fn invert_3x3(m: [[f32; 3]; 3]) -> Result<[[f32; 3]; 3], Matrix3InverseError> {
    for (row, values) in m.iter().enumerate() {
        for (column, value) in values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(Matrix3InverseError::NonFiniteInput { row, column, value });
            }
        }
    }
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
    if !det.is_finite() {
        return Err(Matrix3InverseError::NonFiniteDeterminant { value: det });
    }
    if det.abs() < MIN_DETERMINANT {
        return Err(Matrix3InverseError::Singular { determinant: det });
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
    for (row, values) in inv.iter().enumerate() {
        for (column, value) in values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(Matrix3InverseError::NonFiniteOutput { row, column, value });
            }
        }
    }
    Ok([
        [inv[0][0] as f32, inv[0][1] as f32, inv[0][2] as f32],
        [inv[1][0] as f32, inv[1][1] as f32, inv[1][2] as f32],
        [inv[2][0] as f32, inv[2][1] as f32, inv[2][2] as f32],
    ])
}

fn norm3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn norm6(v: [f32; 6]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3] + v[4] * v[4] + v[5] * v[5]).sqrt()
}

pub(crate) fn solve_linear_system(
    a: &mut [f32],
    b: &mut [f32],
    n: usize,
) -> Result<(), LinearSolveError> {
    if n == 0 {
        return Err(LinearSolveError::ZeroDimension);
    }
    let expected_matrix_len = n
        .checked_mul(n)
        .ok_or(LinearSolveError::DimensionOverflow { dimension: n })?;
    if a.len() != expected_matrix_len {
        return Err(LinearSolveError::MatrixLengthMismatch {
            expected: expected_matrix_len,
            actual: a.len(),
        });
    }
    if b.len() != n {
        return Err(LinearSolveError::RhsLengthMismatch {
            expected: n,
            actual: b.len(),
        });
    }
    if let Some(index) = a.iter().position(|value| !value.is_finite()) {
        return Err(LinearSolveError::NonFiniteMatrix { index });
    }
    if let Some(index) = b.iter().position(|value| !value.is_finite()) {
        return Err(LinearSolveError::NonFiniteRhs { index });
    }

    for i in 0..n {
        let mut max_row = i;
        let mut max_val = a[i * n + i].abs();
        for r in (i + 1)..n {
            let val = a[r * n + i].abs();
            if !val.is_finite() {
                return Err(LinearSolveError::NonFiniteMatrix { index: r * n + i });
            }
            if val > max_val {
                max_val = val;
                max_row = r;
            }
        }

        if !max_val.is_finite() {
            return Err(LinearSolveError::NonFiniteMatrix { index: i * n + i });
        }
        if max_val < PIVOT_TOLERANCE {
            return Err(LinearSolveError::SingularPivot { column: i });
        }

        if max_row != i {
            for c in i..n {
                a.swap(i * n + c, max_row * n + c);
            }
            b.swap(i, max_row);
        }

        let diag = a[i * n + i];
        for c in i..n {
            a[i * n + c] /= diag;
        }
        b[i] /= diag;

        for r in 0..n {
            if r == i {
                continue;
            }
            let factor = a[r * n + i];
            if factor.abs() < ELIMINATION_TOLERANCE {
                continue;
            }
            for c in i..n {
                a[r * n + c] -= factor * a[i * n + c];
            }
            b[r] -= factor * b[i];
        }
    }

    if let Some(index) = b.iter().position(|value| !value.is_finite()) {
        return Err(LinearSolveError::NonFiniteSolution { index });
    }
    Ok(())
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
        reprojection_residual_and_jacobian(pose, obs, intrinsics)
            .expect("valid reprojection")
            .0
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
        let noisy_pose_1 = apply_se3_delta(true_pose_1, noisy_pose_delta);

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
                true_pose_0,
            )
            .expect("insert keyframe 0");
        let kf_1 = map
            .add_keyframe_from_detections(
                detections_1.as_ref(),
                Timestamp::from_nanos(2_000_000),
                noisy_pose_1,
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
            LmConfig::new(1e-5, 10.0, 1e-4, 1e4, 0.25, 0.75),
            Err(LmConfigError::InitialLambdaBelowMin { .. })
        ));
        assert!(matches!(
            LmConfig::new(1e5, 10.0, 1e-8, 1e4, 0.25, 0.75),
            Err(LmConfigError::InitialLambdaAboveMax { .. })
        ));
        assert!(matches!(
            LmConfig::new(1e-4, 10.0, 1e-8, 1e4, 0.8, 0.7),
            Err(LmConfigError::InvalidRhoOrdering { .. })
        ));
    }

    #[test]
    fn ba_outcome_types_reject_forged_costs_and_step_counts() {
        assert!(matches!(
            BaCost::new(f64::NAN),
            Err(BaCostError::NonFinite { .. })
        ));
        assert!(matches!(
            BaCost::new(-1.0),
            Err(BaCostError::Negative { .. })
        ));
        let iterations = NonZeroUsize::new(2).expect("nonzero");
        let accepted_steps = NonZeroUsize::new(3).expect("nonzero");
        assert!(matches!(
            BaOptimization::new(
                BaTermination::IterationLimit { iterations },
                accepted_steps,
                BaCost::new(1.0).expect("valid cost"),
            ),
            Err(BaOutcomeError::AcceptedStepsExceedIterations { .. })
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
        assert!(ObservationSet::when_sufficient(Vec::new(), min_required).is_none());
        let err = ObservationSet::new(Vec::new(), min_required).expect_err("must reject");
        match err {
            ObservationSetError::TooFew { required, actual } => {
                assert_eq!(required, 4);
                assert_eq!(actual, 0);
            }
        }
    }

    #[test]
    fn observation_resolution_reports_removed_association() {
        let (mut map, intrinsics, keyframe_id, _, _, _) = build_full_ba_fixture([0.0; 6]);
        let keypoint = map
            .keyframe_keypoint(keyframe_id, 0)
            .expect("fixture keypoint");
        let point_id = map
            .map_point_for_keypoint(keypoint)
            .expect("fixture lookup")
            .expect("fixture association");
        let pixel = map.keypoint(keypoint).expect("fixture pixel");
        let observations = ObservationSet::new(
            vec![MapObservation::new(keypoint, pixel)],
            NonZeroUsize::MIN,
        )
        .expect("one observation");

        map.remove_map_point(point_id)
            .expect("remove fixture point");

        let error = observations
            .resolve(&map, intrinsics, NonZeroUsize::MIN)
            .expect_err("removed association must not look like insufficient support");
        assert!(matches!(
            error,
            ObservationResolveError::MissingAssociation { keypoint: actual }
                if actual == keypoint
        ));
    }

    #[test]
    fn observation_resolution_preserves_foreign_map_error_source() {
        let (map, intrinsics, keyframe_id, _, _, _) = build_full_ba_fixture([0.0; 6]);
        let keypoint = map
            .keyframe_keypoint(keyframe_id, 0)
            .expect("fixture keypoint");
        let pixel = map.keypoint(keypoint).expect("fixture pixel");
        let observations = ObservationSet::new(
            vec![MapObservation::new(keypoint, pixel)],
            NonZeroUsize::MIN,
        )
        .expect("one observation");

        let error = observations
            .resolve(&SlamMap::new(), intrinsics, NonZeroUsize::MIN)
            .expect_err("foreign keypoint must be rejected");
        assert!(matches!(error, ObservationResolveError::Map { .. }));
        assert!(std::error::Error::source(&error).is_some());
    }

    #[test]
    fn failed_observation_resolution_does_not_advance_ba_window() {
        let (mut map, intrinsics, keyframe_id, _, _, _) = build_full_ba_fixture([0.0; 6]);
        let observations = (0..4)
            .map(|index| {
                let keypoint = map
                    .keyframe_keypoint(keyframe_id, index)
                    .expect("fixture keypoint");
                let pixel = map.keypoint(keypoint).expect("fixture pixel");
                MapObservation::new(keypoint, pixel)
            })
            .collect();
        let observations =
            ObservationSet::new(observations, NonZeroUsize::new(4).expect("nonzero minimum"))
                .expect("fixture observations");
        let removed_keypoint = observations.observations()[0].keyframe_keypoint();
        let removed_point = map
            .map_point_for_keypoint(removed_keypoint)
            .expect("fixture lookup")
            .expect("fixture association");
        map.remove_map_point(removed_point)
            .expect("remove fixture point");
        let config = LocalBaConfig::new(5, 10, 4, 2.0, lm(1e-3), 0.0).expect("valid BA config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);

        let error = ba
            .push_frame(&map, Pose::identity(), observations)
            .expect_err("stale observation must fail");

        assert!(matches!(
            error,
            PoseBaError::Observation {
                source: ObservationResolveError::MissingAssociation { keypoint }
            }
                if keypoint == removed_keypoint
        ));
        assert!(ba.frames.is_empty(), "failed push must be transactional");
    }

    #[test]
    fn failed_pose_solve_restores_evicted_frame_and_pose() {
        let (mut map, intrinsics, keyframe_id, _, _, _) = build_full_ba_fixture([0.0; 6]);
        let mut point_ids = Vec::new();
        let observations = ObservationSet::new(
            (0..4)
                .map(|index| {
                    let keypoint = map.keyframe_keypoint(keyframe_id, index).expect("keypoint");
                    point_ids.push(
                        map.map_point_for_keypoint(keypoint)
                            .expect("association lookup")
                            .expect("map point"),
                    );
                    MapObservation::new(keypoint, map.keypoint(keypoint).expect("pixel"))
                })
                .collect(),
            NonZeroUsize::new(4).expect("literal is non-zero"),
        )
        .expect("observations");
        let config = LocalBaConfig::new(1, 3, 4, 2.0, lm(1e-3), 0.0).expect("BA config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);
        let original_pose = axis_angle_pose([0.1, -0.2, 0.3], [0.01, -0.02, 0.03]);
        let first = ba
            .push_frame(&map, original_pose, observations.clone())
            .expect("first solve");
        assert!(matches!(first, PoseBaOutcome::Refined(_)));
        let retained = ba.frames[0].clone();

        for point_id in point_ids {
            map.set_map_point_position(
                point_id,
                Point3 {
                    x: 1e13,
                    y: 0.0,
                    z: 1.0,
                },
            )
            .expect("finite extreme point");
        }

        let error = ba
            .push_frame(&map, Pose::identity(), observations)
            .expect_err("non-finite normal equations must fail");

        assert!(std::error::Error::source(&error).is_some());
        assert!(matches!(
            error,
            PoseBaError::LinearSolve {
                iteration: 1,
                source: LinearSolveError::NonFiniteMatrix { .. }
                    | LinearSolveError::NonFiniteRhs { .. },
            }
        ));
        assert_eq!(ba.frames.len(), 1);
        assert!(pose_close(ba.frames[0].pose, retained.pose, 0.0));
        assert_eq!(
            ba.frames[0].observations.observations().len(),
            retained.observations.observations().len()
        );
        for (actual, expected) in ba.frames[0]
            .observations
            .observations()
            .iter()
            .zip(retained.observations.observations())
        {
            assert_eq!(actual.keyframe_keypoint(), expected.keyframe_keypoint());
            assert_eq!(actual.pixel().x, expected.pixel().x);
            assert_eq!(actual.pixel().y, expected.pixel().y);
        }
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
        let out = apply_se3_delta(pose, [0.0; 6]);
        assert!(pose_close(pose, out, 1e-7));
    }

    #[test]
    fn apply_se3_delta_matches_pose64_reference_update() {
        let pose = axis_angle_pose([0.3, -0.4, 0.5], [0.08, -0.05, 0.03]);
        let delta = [0.12, -0.07, 0.05, 0.18, -0.09, 0.11];
        let actual = apply_se3_delta(pose, delta);
        let expected = crate::math::se3_exp_f64(delta.map(f64::from))
            .compose(crate::Pose64::from_pose32(pose))
            .to_pose32();
        assert!(pose_close(actual, expected, 1e-5));
    }

    #[test]
    fn se3_delta_between_matches_pose64_reference_log() {
        let from = axis_angle_pose([0.2, -0.3, 0.4], [0.05, -0.03, 0.02]);
        let to = axis_angle_pose([0.5, -0.1, 0.2], [0.18, -0.07, 0.11]);
        let actual = se3_delta_between(from, to);
        let expected = crate::math::se3_log_f64(
            crate::Pose64::from_pose32(to).compose(crate::Pose64::from_pose32(from).inverse()),
        )
        .map(|value| value as f32);
        for axis in 0..6 {
            assert!(
                (actual[axis] - expected[axis]).abs() < 1e-5,
                "delta mismatch on axis {axis}: actual={}, expected={}",
                actual[axis],
                expected[axis]
            );
        }
    }

    #[test]
    fn solve_linear_system_solves_identity_system() {
        let mut a = vec![1.0_f32, 0.0, 0.0, 1.0];
        let mut b = vec![2.5_f32, -3.0];
        solve_linear_system(&mut a, &mut b, 2).expect("identity system");
        assert!((b[0] - 2.5).abs() < 1e-6);
        assert!((b[1] + 3.0).abs() < 1e-6);
    }

    #[test]
    fn solve_linear_system_reports_singular_matrix() {
        let mut a = vec![1.0_f32, 2.0, 2.0, 4.0];
        let mut b = vec![1.0_f32, 2.0];
        assert!(matches!(
            solve_linear_system(&mut a, &mut b, 2),
            Err(LinearSolveError::SingularPivot { .. })
        ));
    }

    #[test]
    fn solve_linear_system_rejects_invalid_shapes_and_nonfinite_inputs() {
        let mut short_matrix = vec![1.0_f32; 3];
        let mut rhs = vec![1.0_f32; 2];
        assert_eq!(
            solve_linear_system(&mut short_matrix, &mut rhs, 2),
            Err(LinearSolveError::MatrixLengthMismatch {
                expected: 4,
                actual: 3,
            })
        );

        let mut matrix = vec![1.0_f32, 0.0, 0.0, 1.0];
        let mut short_rhs = vec![1.0_f32];
        assert_eq!(
            solve_linear_system(&mut matrix, &mut short_rhs, 2),
            Err(LinearSolveError::RhsLengthMismatch {
                expected: 2,
                actual: 1,
            })
        );

        let mut matrix = vec![1.0_f32, f32::NAN, 0.0, 1.0];
        let mut rhs = vec![1.0_f32, 1.0];
        assert_eq!(
            solve_linear_system(&mut matrix, &mut rhs, 2),
            Err(LinearSolveError::NonFiniteMatrix { index: 1 })
        );
    }

    #[test]
    fn ba_execution_error_preserves_linear_solve_source() {
        let error = BaExecutionError::PoseLinearSystem {
            iteration: 2,
            source: LinearSolveError::SingularPivot { column: 4 },
        };
        assert!(std::error::Error::source(&error).is_some());
    }

    #[test]
    fn motion_prior_rhs_uses_the_same_squared_weight_as_cost_and_hessian() {
        let previous = Pose::identity();
        let current = Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [1.0, 0.0, 0.0],
        );

        for (weight, expected) in [(0.5_f32, 0.25_f32), (2.0_f32, 4.0_f32)] {
            let mut hessian = vec![0.0_f32; 12 * 12];
            let mut rhs = vec![0.0_f32; 12];
            accumulate_motion_prior(&mut hessian, &mut rhs, 12, previous, current, 0, 6, weight);

            assert!((rhs[0] - expected).abs() < 1e-6);
            assert!((rhs[6] + expected).abs() < 1e-6);
            assert!((hessian[0] - expected).abs() < 1e-6);
            assert!((hessian[6 * 12 + 6] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn full_problem_cost_rejects_nonprojectable_factors() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
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
                position: Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: -1.0,
                },
            }],
            factors: vec![ReprojectionFactor {
                pose: PoseVarIndex(0),
                landmark: LandmarkVarIndex(0),
                pixel: Keypoint { x: 320.0, y: 240.0 },
            }],
        };

        let error = full_problem_cost(&problem, intrinsics, 2.0, 0.0)
            .expect_err("a behind-camera factor cannot disappear from cost");
        assert!(matches!(
            error,
            FullProblemCostError::NonProjectable { count } if count.get() == 1
        ));

        let config = LocalBaConfig::new(5, 2, 4, 2.0, lm(1e-3), 0.0).expect("BA config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);
        assert_eq!(
            ba.optimize_full(&mut problem).expect("degenerate outcome"),
            BaResult::Degenerate {
                reason: DegenerateReason::NonProjectableFactors { count: 1 }
            }
        );
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
        };
        assert!(matches!(
            ba.optimize_full(&mut no_poses),
            Ok(BaResult::Degenerate {
                reason: DegenerateReason::TooFewPoses { count: 0 }
            })
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
        };
        assert!(matches!(
            ba.optimize_full(&mut no_landmarks),
            Ok(BaResult::Degenerate {
                reason: DegenerateReason::TooFewLandmarks { count: 0 }
            })
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
        };
        assert!(matches!(
            ba.optimize_full(&mut no_factors),
            Ok(BaResult::Degenerate {
                reason: DegenerateReason::NoFactors
            })
        ));
    }

    #[test]
    fn optimize_full_reports_landmark_breakdown_without_mutating_problem() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let config = LocalBaConfig::new(5, 2, 4, 2.0, lm(1e-8), 0.0).expect("BA config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);
        let original_pose = axis_angle_pose([0.1, -0.2, 0.3], [0.01, -0.02, 0.03]);
        let original_point = Point3 {
            x: 0.0,
            y: 0.0,
            z: 1e20,
        };
        let mut problem = FullBaProblem {
            poses: vec![
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: Pose::identity(),
                },
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: original_pose,
                },
            ],
            landmarks: vec![LandmarkVariable {
                point_id: MapPointId::default(),
                position: original_point,
            }],
            factors: vec![ReprojectionFactor {
                pose: PoseVarIndex(1),
                landmark: LandmarkVarIndex(0),
                pixel: Keypoint { x: 320.0, y: 240.0 },
            }],
        };

        let error = ba
            .optimize_full(&mut problem)
            .expect_err("near-zero landmark information must be a solver error");

        assert!(std::error::Error::source(&error).is_some());
        assert!(matches!(
            error,
            BaExecutionError::LandmarkLinearSystem {
                iteration: 1,
                landmark_index: 0,
                source: Matrix3InverseError::Singular { .. },
            }
        ));
        assert!(pose_close(problem.poses[1].pose, original_pose, 0.0));
        let retained_point = problem.landmarks[0].position;
        assert_eq!(retained_point.x, original_point.x);
        assert_eq!(retained_point.y, original_point.y);
        assert_eq!(retained_point.z, original_point.z);
    }

    #[test]
    fn optimize_full_returns_iteration_limit_with_bad_init() {
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
        assert!(matches!(
            ba.optimize_full(&mut problem),
            Ok(BaResult::Optimized(optimization))
                if matches!(
                    optimization.termination(),
                    BaTermination::IterationLimit { iterations } if iterations.get() == 1
                )
        ));
    }

    #[test]
    fn optimize_full_reports_stationary_problem_without_applicable_correction() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let point = Point3 {
            x: 0.0,
            y: 0.0,
            z: 3.0,
        };
        let mut factors = Vec::new();
        for pose in [PoseVarIndex(0), PoseVarIndex(1)] {
            factors.push(ReprojectionFactor {
                pose,
                landmark: LandmarkVarIndex(0),
                pixel: Keypoint { x: 310.0, y: 240.0 },
            });
            factors.push(ReprojectionFactor {
                pose,
                landmark: LandmarkVarIndex(0),
                pixel: Keypoint { x: 330.0, y: 240.0 },
            });
        }
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
            factors,
        };
        let config = LocalBaConfig::new(5, 3, 4, 2.0, lm(1e-3), 0.0).expect("BA config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);

        let result = ba.optimize_full(&mut problem).expect("stationary solve");

        assert!(matches!(
            result,
            BaResult::Stationary(stationary)
                if stationary.detected_at_iteration() == NonZeroUsize::MIN
        ));
        assert!(!result.is_applicable());
        assert!(pose_close(problem.poses[1].pose, Pose::identity(), 0.0));
        assert_eq!(problem.landmarks[0].position.x, point.x);
        assert_eq!(problem.landmarks[0].position.y, point.y);
        assert_eq!(problem.landmarks[0].position.z, point.z);
    }

    #[test]
    fn stalled_ba_result_cannot_be_applied_as_a_correction() {
        let result = BaResult::Stalled(BaStall::new(
            NonZeroUsize::new(3).expect("nonzero"),
            BaCost::new(4.0).expect("valid cost"),
        ));
        assert!(!result.is_applicable());
        assert!(result.optimization().is_none());
    }

    #[test]
    fn optimize_full_reports_rejected_nonzero_proposals_as_stalled() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let original_point = Point3 {
            x: 1e-3,
            y: 0.0,
            z: 1e-3,
        };
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
                position: original_point,
            }],
            factors: vec![
                ReprojectionFactor {
                    pose: PoseVarIndex(0),
                    landmark: LandmarkVarIndex(0),
                    pixel: Keypoint {
                        x: 10_000.0,
                        y: 240.0,
                    },
                },
                ReprojectionFactor {
                    pose: PoseVarIndex(1),
                    landmark: LandmarkVarIndex(0),
                    pixel: Keypoint {
                        x: 10_000.0,
                        y: 240.0,
                    },
                },
            ],
        };
        let config = LocalBaConfig::new(5, 1, 4, 2.0, lm(1e4), 0.0).expect("BA config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);
        let original_poses: Vec<_> = problem.poses.iter().map(|pose| pose.pose).collect();

        let result = ba.optimize_full(&mut problem).expect("stalled solve");

        assert!(matches!(result, BaResult::Stalled(_)), "got {result:?}");
        assert!(!result.is_applicable());
        assert!(
            problem
                .poses
                .iter()
                .zip(original_poses)
                .all(|(actual, expected)| pose_close(actual.pose, expected, 0.0))
        );
        assert!(
            problem.landmarks[0].position.x == original_point.x
                && problem.landmarks[0].position.y == original_point.y
                && problem.landmarks[0].position.z == original_point.z
        );
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
        let result = ba.optimize_full(&mut problem).expect("full BA solve");
        match result {
            BaResult::Optimized(optimization) => {
                let BaTermination::Converged { iterations } = optimization.termination() else {
                    panic!("expected convergence, got {optimization:?}");
                };
                assert!(iterations.get() < config.max_iterations());
                assert!(optimization.final_cost().get().is_finite());
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
            .expect("initial factors are projectable")
            .get();
        let result = ba.optimize_full(&mut problem).expect("full BA solve");
        let after = full_problem_cost(&problem, intrinsics, config.huber_delta_px(), 0.0)
            .expect("optimized factors remain projectable")
            .get();
        assert!(
            after < before,
            "cost did not improve: before={before}, after={after}"
        );
        assert!(result.is_applicable());
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
            .expect("initial factors are projectable")
            .get();
        let result = ba.optimize_full(&mut problem).expect("full BA solve");
        let final_cost = match result {
            BaResult::Optimized(optimization) => optimization.final_cost().get(),
            other => panic!("unexpected BA outcome: {other:?}"),
        };
        assert!(
            final_cost <= before + 1e-6,
            "final cost should not increase: before={before}, final={final_cost}"
        );
    }

    #[test]
    fn local_bundle_adjuster_clone_supports_transactional_candidate_updates() {
        let (map, intrinsics, kf_0, _kf_1, _, _) =
            build_full_ba_fixture([0.12, -0.05, 0.07, 0.03, -0.02, 0.01]);
        let config = LocalBaConfig::new(5, 20, 4, 2.0, lm(1e-3), 0.0).expect("valid BA config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);
        let observations = ObservationSet::new(
            (0..4)
                .map(|idx| {
                    let keyframe_keypoint = map.keyframe_keypoint(kf_0, idx).expect("keypoint");
                    let pixel = map.keypoint(keyframe_keypoint).expect("pixel");
                    MapObservation::new(keyframe_keypoint, pixel)
                })
                .collect(),
            ba.min_observations(),
        )
        .expect("observations");

        let _ = ba.push_frame(&map, Pose::identity(), observations.clone());
        let original_len = ba.frames.len();

        let mut candidate_ba = ba.clone();
        let _ = candidate_ba.push_frame(&map, Pose::identity(), observations);

        assert_eq!(
            ba.frames.len(),
            original_len,
            "candidate BA updates must not mutate the authoritative window"
        );
        assert!(
            candidate_ba.frames.len() >= ba.frames.len(),
            "candidate BA should evolve independently from the original"
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

        let obs = Observation::try_new(point, pixel, intrinsics).expect("observation");
        let (_residual, jac) =
            reprojection_residual_and_jacobian(pose, &obs, intrinsics).expect("jacobian");

        let eps = 1e-3_f32;
        for col in 0..6 {
            let mut delta_pos = [0.0_f32; 6];
            delta_pos[col] = eps;
            let mut delta_neg = [0.0_f32; 6];
            delta_neg[col] = -eps;

            let r_plus = projection_residual(apply_se3_delta(pose, delta_pos), &obs, intrinsics);
            let r_minus = projection_residual(apply_se3_delta(pose, delta_neg), &obs, intrinsics);
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
        let noisy_pose_1 = apply_se3_delta(true_pose_1, [0.08, -0.03, 0.04, 0.015, -0.01, 0.008]);

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
                true_pose_0,
            )
            .expect("insert keyframe 0");
        let kf_1 = map
            .add_keyframe_from_detections(
                detections_1.as_ref(),
                Timestamp::from_nanos(2_000_000),
                noisy_pose_1,
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
        let before_pose_err = pose_distance(map.keyframe(kf_1).expect("kf1").pose(), true_pose_1);
        let before_landmark_err = mean_landmark_error(&map, kf_0, &points_true);

        let config = LocalBaConfig::new(5, 15, 4, 2.0, lm(1e-3), 0.0).expect("valid BA config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);
        let result = ba
            .optimize_keyframe_window(&mut map, &[kf_0, kf_1])
            .expect("full local BA execution");
        assert!(
            result.is_applicable(),
            "full local BA should succeed, got {result:?}"
        );
        assert_map_invariants(&map).expect("map invariants after BA");

        let after_pose_err = pose_distance(map.keyframe(kf_1).expect("kf1").pose(), true_pose_1);
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
    fn optimize_keyframe_window_reports_duplicate_ids_as_execution_error() {
        let (mut map, intrinsics, keyframe_id, _, _, _) = build_full_ba_fixture([0.0; 6]);
        let before = map.snapshot();
        let config = LocalBaConfig::new(5, 15, 4, 2.0, lm(1e-3), 0.0).expect("valid BA config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);

        let error = ba
            .optimize_keyframe_window(&mut map, &[keyframe_id, keyframe_id])
            .expect_err("duplicate keyframes are structural errors");

        assert!(matches!(
            error,
            BaExecutionError::DuplicateKeyframe { keyframe_id: actual }
                if actual == keyframe_id
        ));
        assert_eq!(map.snapshot(), before);
    }

    #[cfg(feature = "vio")]
    #[test]
    fn vio_linearization_rejects_nonfinite_hessian_and_rhs_before_success() {
        for (quantity, hessian, rhs) in [
            (
                VioLinearizationQuantity::Hessian,
                vec![f64::INFINITY],
                vec![0.0],
            ),
            (
                VioLinearizationQuantity::RightHandSide,
                vec![0.0],
                vec![f64::NAN],
            ),
        ] {
            let linearization = VioLinearization {
                hessian,
                rhs,
                cost_breakdown: VioCostBreakdown::default(),
            };
            let error = validate_vio_linearization_values(
                &linearization,
                VioEvaluation {
                    stage: VioEvaluationStage::Candidate,
                    iteration: 3,
                },
            )
            .expect_err("nonfinite linearization must fail");
            assert!(matches!(
                error,
                VioSolveError::NonFiniteLinearization {
                    stage: VioEvaluationStage::Candidate,
                    iteration: 3,
                    quantity: actual,
                    index: 0,
                    ..
                } if actual == quantity
            ));
        }
    }

    #[cfg(feature = "vio")]
    #[test]
    fn vio_convergence_uses_componentwise_physical_unit_tolerances() {
        assert!(vio_step_is_componentwise_small(&[0.0; VIO_STATE_DIM]));
        for (axis, tolerance) in VIO_STATE_CONVERGENCE_TOLERANCES.into_iter().enumerate() {
            let mut below = [0.0; VIO_STATE_DIM * 2];
            below[axis] = 0.5 * tolerance;
            below[VIO_STATE_DIM + axis] = -0.5 * tolerance;
            assert!(vio_step_is_componentwise_small(&below), "axis {axis}");

            let mut at_limit = [0.0; VIO_STATE_DIM];
            at_limit[axis] = tolerance;
            assert!(!vio_step_is_componentwise_small(&at_limit), "axis {axis}");
        }
        let mut nonfinite = [0.0; VIO_STATE_DIM];
        nonfinite[0] = f64::NAN;
        assert!(!vio_step_is_componentwise_small(&nonfinite));
        assert!(!vio_step_is_componentwise_small(&[0.0; VIO_STATE_DIM - 1]));
    }

    #[cfg(feature = "vio")]
    #[test]
    fn vio_config_preserves_bias_information_units_and_rejects_invalid_values() {
        let prior = VioBiasPrior::new(2.0, 3.0, crate::ImuBias::default()).expect("bias prior");
        assert_eq!(prior.accel_information_s4_per_m2(), 2.0);
        assert_eq!(prior.gyro_information_s2_per_rad2(), 3.0);

        for value in [0.0, -1.0] {
            for (quantity, result) in [
                (
                    VioBiasPriorInformationQuantity::AccelerometerBiasS4PerM2,
                    VioBiasPrior::new(value, 1.0, crate::ImuBias::default()),
                ),
                (
                    VioBiasPriorInformationQuantity::GyroscopeBiasS2PerRad2,
                    VioBiasPrior::new(1.0, value, crate::ImuBias::default()),
                ),
            ] {
                assert!(matches!(
                    result,
                    Err(VioSolveConfigError::NonPositiveBiasPriorInformation {
                        quantity: actual_quantity,
                        value: actual_value,
                    }) if actual_quantity == quantity && actual_value == value
                ));
            }
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for (quantity, result) in [
                (
                    VioBiasPriorInformationQuantity::AccelerometerBiasS4PerM2,
                    VioBiasPrior::new(value, 1.0, crate::ImuBias::default()),
                ),
                (
                    VioBiasPriorInformationQuantity::GyroscopeBiasS2PerRad2,
                    VioBiasPrior::new(1.0, value, crate::ImuBias::default()),
                ),
            ] {
                assert!(matches!(
                    result,
                    Err(VioSolveConfigError::NonFiniteBiasPriorInformation {
                        quantity: actual_quantity,
                        value: actual_value,
                    }) if actual_quantity == quantity && actual_value.to_bits() == value.to_bits()
                ));
            }
        }

        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                VioSolveConfig::new(
                    crate::Gravity::try_new([0.0, 9.81, 0.0]).expect("gravity"),
                    crate::Pose64::identity(),
                    intrinsics,
                    lm(1e-3),
                    NonZeroUsize::MIN,
                    2.0,
                    value,
                    None,
                ),
                Err(VioSolveConfigError::NonFiniteVelocityAnchorInformation {
                    value: actual,
                }) if actual.to_bits() == value.to_bits()
            ));
        }
        assert!(matches!(
            VioSolveConfig::new(
                crate::Gravity::try_new([0.0, 9.81, 0.0]).expect("gravity"),
                crate::Pose64::identity(),
                intrinsics,
                lm(1e-3),
                NonZeroUsize::MIN,
                2.0,
                -1.0,
                None,
            ),
            Err(VioSolveConfigError::NegativeVelocityAnchorInformation { value: -1.0 })
        ));
        for huber_delta_px in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                VioSolveConfig::new(
                    crate::Gravity::try_new([0.0, 9.81, 0.0]).expect("gravity"),
                    crate::Pose64::identity(),
                    intrinsics,
                    lm(1e-3),
                    NonZeroUsize::MIN,
                    huber_delta_px,
                    1.0,
                    None,
                ),
                Err(VioSolveConfigError::NonFiniteHuberDeltaPx { value })
                    if value.to_bits() == huber_delta_px.to_bits()
            ));
        }
        for huber_delta_px in [0.0, -1.0] {
            assert!(matches!(
                VioSolveConfig::new(
                    crate::Gravity::try_new([0.0, 9.81, 0.0]).expect("gravity"),
                    crate::Pose64::identity(),
                    intrinsics,
                    lm(1e-3),
                    NonZeroUsize::MIN,
                    huber_delta_px,
                    1.0,
                    None,
                ),
                Err(VioSolveConfigError::NonPositiveHuberDeltaPx { value })
                    if value == huber_delta_px
            ));
        }
    }

    #[cfg(feature = "vio")]
    #[test]
    fn vio_bias_prior_applies_distinct_accel_and_gyro_information() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let prior = VioBiasPrior::new(2.0, 3.0, crate::ImuBias::default()).expect("bias prior");
        let config = VioSolveConfig::new(
            crate::Gravity::try_new([0.0, 9.81, 0.0]).expect("gravity"),
            crate::Pose64::identity(),
            intrinsics,
            lm(1e-3),
            NonZeroUsize::MIN,
            2.0,
            0.0,
            Some(prior),
        )
        .expect("VIO config");
        let state = crate::NavState::try_new(
            crate::Pose64::identity(),
            [0.0; 3],
            crate::ImuBias::try_new([1.0, 0.0, 0.0], [2.0, 0.0, 0.0]).expect("bias"),
        )
        .expect("state");
        let states = vec![state.clone()];
        let resolved = vec![None];
        let map_from_odom = crate::MapFromOdom::identity();
        let support =
            VisualFactorSupport::from_initial_states(&states, &resolved, &config, &map_from_odom);
        let window = VioWindow {
            anchor: VioAnchor {
                state,
                observations: None,
                anchor_velocity_odom_mps: [0.0; 3],
            },
            successors: Vec::new(),
        };

        let linearization = linearize_vio_states(
            &window,
            &states,
            &resolved,
            &support,
            &config,
            &map_from_odom,
            VioEvaluation {
                stage: VioEvaluationStage::Initial,
                iteration: 0,
            },
        )
        .expect("finite linearization");
        let accel_index = 9;
        let gyro_index = 12;
        assert_eq!(
            linearization.hessian[accel_index * VIO_STATE_DIM + accel_index],
            2.0
        );
        assert_eq!(
            linearization.hessian[gyro_index * VIO_STATE_DIM + gyro_index],
            3.0
        );
        assert_eq!(linearization.rhs[accel_index], 2.0);
        assert_eq!(linearization.rhs[gyro_index], 6.0);
        assert_eq!(linearization.cost_breakdown.bias_prior_cost, 7.0);
    }

    #[cfg(feature = "vio")]
    #[test]
    fn vio_visual_factor_support_is_fixed_across_candidate_evaluations() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let config = VioSolveConfig::new(
            crate::Gravity::try_new([0.0, 9.81, 0.0]).expect("gravity"),
            crate::Pose64::identity(),
            intrinsics,
            lm(1e-3),
            NonZeroUsize::new(2).expect("iterations"),
            2.0,
            1.0,
            None,
        )
        .expect("VIO config");
        let identity_state = crate::NavState::try_new(
            crate::Pose64::identity(),
            [0.0; 3],
            crate::ImuBias::default(),
        )
        .expect("identity state");
        let initial_states = vec![identity_state.clone(), identity_state.clone()];
        let front = Observation::try_new(
            Point3 {
                x: 0.0,
                y: 0.0,
                z: 2.0,
            },
            Keypoint { x: 320.0, y: 240.0 },
            intrinsics,
        )
        .expect("front observation");
        let behind = Observation::try_new(
            Point3 {
                x: 0.0,
                y: 0.0,
                z: -2.0,
            },
            Keypoint { x: 320.0, y: 240.0 },
            intrinsics,
        )
        .expect("behind observation");
        let resolved = vec![
            None,
            Some(ResolvedObservationSet {
                observations: vec![front, behind],
            }),
        ];
        let map_from_odom = crate::MapFromOdom::identity();
        let support = VisualFactorSupport::from_initial_states(
            &initial_states,
            &resolved,
            &config,
            &map_from_odom,
        );
        assert_eq!(support.indices_for_frame(1), &[0]);
        assert_eq!(support.last_frame_factor_count(), 1);
        assert_eq!(support.initially_excluded_nonprojectable_factor_count, 1);

        let batch = crate::ImuBatch::new(vec![
            crate::ImuSample::new(Timestamp::from_nanos(0), [0.0; 3], [0.0; 3])
                .expect("first IMU sample"),
            crate::ImuSample::new(Timestamp::from_nanos(10_000_000), [0.0; 3], [0.0; 3])
                .expect("second IMU sample"),
        ])
        .expect("IMU batch");
        let preintegrated = crate::PreintegratedImu::integrate(
            &batch,
            &crate::ImuBias::default(),
            &crate::ImuNoiseModel::new(0.1, 0.01, 0.001, 0.0001).expect("noise"),
        )
        .expect("preintegration");
        let window = VioWindow {
            anchor: VioAnchor {
                state: initial_states[0].clone(),
                observations: None,
                anchor_velocity_odom_mps: [0.0; 3],
            },
            successors: vec![VioSuccessor {
                state: initial_states[1].clone(),
                observations: None,
                preintegrated,
            }],
        };
        let moved_past_point = crate::NavState::try_new(
            crate::Pose64::from_rt(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0, 0.0, 3.0],
            ),
            [0.0; 3],
            crate::ImuBias::default(),
        )
        .expect("translated state");
        let candidate_states = vec![initial_states[0].clone(), moved_past_point];

        let error = match linearize_vio_states(
            &window,
            &candidate_states,
            &resolved,
            &support,
            &config,
            &map_from_odom,
            VioEvaluation {
                stage: VioEvaluationStage::Candidate,
                iteration: 1,
            },
        ) {
            Ok(_) => panic!("candidate must not remove an initially active visual factor"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            VioSolveError::ReprojectionFactorUnavailable {
                stage: VioEvaluationStage::Candidate,
                iteration: 1,
                frame_index: 1,
                observation_index: 0,
            }
        ));
        assert!(error.is_rejected_candidate_nonprojectability());
    }

    #[cfg(feature = "vio")]
    #[test]
    fn vio_cam_from_map_respects_map_from_odom_bridge() {
        let state = crate::NavState::try_new(
            crate::Pose64::from_rt(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [1.0, -2.0, 3.0],
            ),
            [0.0; 3],
            crate::ImuBias::default(),
        )
        .expect("state");
        let camera_from_body = crate::Pose64::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            [0.1, 0.2, 0.3],
        );
        let mut bridge = crate::MapFromOdom::identity();
        bridge.set_pose_map_from_odom(crate::Pose64::from_rt(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [0.5, -0.25, 1.5],
        ));

        let cam_from_odom = camera_from_body.compose(state.pose_odom_from_body().inverse());
        let expected = bridge.odom_to_map(cam_from_odom).to_pose32();
        let actual = vio_cam_from_map(&state, camera_from_body, &bridge);

        assert_eq!(actual.translation(), expected.translation());
        assert_eq!(actual.rotation(), expected.rotation());
    }
}
