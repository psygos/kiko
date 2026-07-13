use std::collections::{HashMap, HashSet, TryReserveError};
use std::num::NonZeroUsize;

use crate::{
    Keypoint, Observation, PinholeIntrinsics, Point3, Pose,
    map::{KeyframeId, KeyframeKeypoint, MapPointId, SlamMap},
    math,
};

fn try_vec_with_capacity<T>(requested_elements: usize) -> Result<Vec<T>, TryReserveError> {
    let mut values = Vec::new();
    values.try_reserve_exact(requested_elements)?;
    Ok(values)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DenseWorkspaceShape {
    dimension: usize,
    matrix_elements: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DenseWorkspaceShapeError {
    DimensionOverflow { item_count: usize },
    MatrixElementCountOverflow { dimension: usize },
    MatrixByteLengthUnaddressable { element_count: usize },
}

impl DenseWorkspaceShape {
    fn try_new(
        item_count: usize,
        dimensions_per_item: usize,
        scalar_bytes: usize,
    ) -> Result<Self, DenseWorkspaceShapeError> {
        let dimension = item_count
            .checked_mul(dimensions_per_item)
            .ok_or(DenseWorkspaceShapeError::DimensionOverflow { item_count })?;
        let matrix_elements = dimension
            .checked_mul(dimension)
            .ok_or(DenseWorkspaceShapeError::MatrixElementCountOverflow { dimension })?;
        matrix_elements
            .checked_mul(scalar_bytes)
            .filter(|&byte_len| byte_len <= isize::MAX as usize)
            .ok_or(DenseWorkspaceShapeError::MatrixByteLengthUnaddressable {
                element_count: matrix_elements,
            })?;
        Ok(Self {
            dimension,
            matrix_elements,
        })
    }
}

/// Maximum camera-translation update in metres for convergence detection.
const POSE_TRANSLATION_STEP_CONVERGENCE_M: f32 = 1e-4;
/// Maximum camera-rotation update in radians for convergence detection.
const POSE_ROTATION_STEP_CONVERGENCE_RAD: f32 = 1e-4;
/// Maximum landmark-position update in metres for convergence detection.
const LANDMARK_STEP_CONVERGENCE_M: f32 = 1e-4;
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
/// Minimum separation from the fixed camera centre used to condition the
/// exact metric-scale anchor. Map positions and camera translations are metres.
const MIN_SCALE_ANCHOR_DISPLACEMENT_M: f32 = 1e-6;
/// Absolute minimum observation count for BA config (PnP geometric minimum).
const ABSOLUTE_MIN_OBSERVATIONS: usize = 4;
/// Number of tangent parameters in one SE(3) pose update.
const POSE_TANGENT_DIMENSION: usize = 6;

#[derive(Clone, Copy, Debug)]
pub struct LocalBaConfig {
    window: NonZeroUsize,
    workspace: DenseWorkspaceShape,
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
    PoseDimensionOverflow { window: usize },
    DenseMatrixElementCountOverflow { pose_dimension: usize },
    DenseMatrixByteLengthUnaddressable { element_count: usize },
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
            LocalBaConfigError::PoseDimensionOverflow { window } => write!(
                f,
                "local BA pose dimension overflows usize for a {window}-pose window"
            ),
            LocalBaConfigError::DenseMatrixElementCountOverflow { pose_dimension } => write!(
                f,
                "local BA dense matrix element count overflows usize for pose dimension {pose_dimension}"
            ),
            LocalBaConfigError::DenseMatrixByteLengthUnaddressable { element_count } => write!(
                f,
                "local BA dense matrix with {element_count} f32 elements exceeds the addressable vector byte length"
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
    ) -> Result<Self, LocalBaConfigError> {
        let window = NonZeroUsize::new(window).ok_or(LocalBaConfigError::ZeroWindow)?;
        let workspace = DenseWorkspaceShape::try_new(
            window.get(),
            POSE_TANGENT_DIMENSION,
            std::mem::size_of::<f32>(),
        )
        .map_err(|source| match source {
            DenseWorkspaceShapeError::DimensionOverflow { item_count } => {
                LocalBaConfigError::PoseDimensionOverflow { window: item_count }
            }
            DenseWorkspaceShapeError::MatrixElementCountOverflow { dimension } => {
                LocalBaConfigError::DenseMatrixElementCountOverflow {
                    pose_dimension: dimension,
                }
            }
            DenseWorkspaceShapeError::MatrixByteLengthUnaddressable { element_count } => {
                LocalBaConfigError::DenseMatrixByteLengthUnaddressable { element_count }
            }
        })?;
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
            workspace,
            max_iterations,
            min_observations,
            huber_delta_px,
            lm,
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
    DisconnectedFromFixedPose {
        disconnected_pose_count: NonZeroUsize,
    },
    UnobservableMetricScale,
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
    MissingObservationKeyframe {
        point_id: MapPointId,
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
            Self::MissingObservationKeyframe {
                point_id,
                keyframe_id,
            } => write!(
                f,
                "landmark {point_id:?} observation references missing keyframe {keyframe_id:?}"
            ),
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
            | Self::MissingObservationKeyframe { .. }
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
    Allocation {
        requested_observations: usize,
        source: TryReserveError,
    },
    Map {
        source: crate::map::MapError,
    },
    MissingAssociation {
        keypoint: KeyframeKeypoint,
    },
    MissingMapPoint {
        point_id: MapPointId,
    },
    Pnp {
        source: crate::PnpError,
    },
}

impl std::fmt::Display for ObservationResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Allocation {
                requested_observations,
                source,
            } => write!(
                f,
                "could not allocate BA buffer for {requested_observations} resolved observations: {source}"
            ),
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
            Self::Allocation { source, .. } => Some(source),
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
        let mut resolved = Vec::new();
        if !self.resolve_observations_into(map, intrinsics, min_required, &mut resolved)? {
            return Ok(None);
        }
        Ok(Some(ResolvedObservationSet {
            observations: resolved,
        }))
    }

    fn resolve_observations_into(
        &self,
        map: &SlamMap,
        intrinsics: PinholeIntrinsics,
        min_required: NonZeroUsize,
        resolved: &mut Vec<Observation>,
    ) -> Result<bool, ObservationResolveError> {
        resolved.clear();
        if resolved.capacity() < self.observations.len() {
            resolved
                .try_reserve_exact(self.observations.len())
                .map_err(|source| ObservationResolveError::Allocation {
                    requested_observations: self.observations.len(),
                    source,
                })?;
        }
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
        Ok(resolved.len() >= min_required.get())
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
const VIO_RELATIVE_OBJECTIVE_CONVERGENCE_TOLERANCE: f64 = 1e-10;
#[cfg(feature = "vio")]
const VIO_RELATIVE_OBJECTIVE_SCALE_FLOOR: f64 = 1e-12;
#[cfg(feature = "vio")]
const MIN_VIO_WINDOW_FRAMES: usize = 2;

/// Validated maximum number of frames in one VIO solve.
///
/// Construction proves that the corresponding dense `f64` normal matrix has
/// an addressable element and byte count. It does not promise that the host
/// has enough memory; workspace allocation remains fallible.
#[cfg(feature = "vio")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VioWindowCapacity {
    frames: NonZeroUsize,
    workspace: DenseWorkspaceShape,
}

#[cfg(feature = "vio")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VioWindowCapacityError {
    TooFewFrames { minimum: usize, actual: usize },
    StateDimensionOverflow { frames: usize },
    DenseMatrixElementCountOverflow { state_dimension: usize },
    DenseMatrixByteLengthUnaddressable { element_count: usize },
}

#[cfg(feature = "vio")]
impl std::fmt::Display for VioWindowCapacityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewFrames { minimum, actual } => write!(
                f,
                "VIO window capacity must be at least {minimum} frames, got {actual}"
            ),
            Self::StateDimensionOverflow { frames } => write!(
                f,
                "VIO state dimension overflows usize for a {frames}-frame window"
            ),
            Self::DenseMatrixElementCountOverflow { state_dimension } => write!(
                f,
                "VIO dense matrix element count overflows usize for state dimension {state_dimension}"
            ),
            Self::DenseMatrixByteLengthUnaddressable { element_count } => write!(
                f,
                "VIO dense matrix with {element_count} f64 elements exceeds the addressable vector byte length"
            ),
        }
    }
}

#[cfg(feature = "vio")]
impl std::error::Error for VioWindowCapacityError {}

#[cfg(feature = "vio")]
impl VioWindowCapacity {
    pub fn new(frames: usize) -> Result<Self, VioWindowCapacityError> {
        let actual_frames = frames;
        let frames = NonZeroUsize::new(frames).ok_or(VioWindowCapacityError::TooFewFrames {
            minimum: MIN_VIO_WINDOW_FRAMES,
            actual: actual_frames,
        })?;
        if frames.get() < MIN_VIO_WINDOW_FRAMES {
            return Err(VioWindowCapacityError::TooFewFrames {
                minimum: MIN_VIO_WINDOW_FRAMES,
                actual: actual_frames,
            });
        }
        let workspace =
            DenseWorkspaceShape::try_new(frames.get(), VIO_STATE_DIM, std::mem::size_of::<f64>())
                .map_err(|source| match source {
                DenseWorkspaceShapeError::DimensionOverflow { item_count } => {
                    VioWindowCapacityError::StateDimensionOverflow { frames: item_count }
                }
                DenseWorkspaceShapeError::MatrixElementCountOverflow { dimension } => {
                    VioWindowCapacityError::DenseMatrixElementCountOverflow {
                        state_dimension: dimension,
                    }
                }
                DenseWorkspaceShapeError::MatrixByteLengthUnaddressable { element_count } => {
                    VioWindowCapacityError::DenseMatrixByteLengthUnaddressable { element_count }
                }
            })?;
        Ok(Self { frames, workspace })
    }

    pub fn frames(self) -> NonZeroUsize {
        self.frames
    }
}

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
    window_capacity: VioWindowCapacity,
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
        window_capacity: VioWindowCapacity,
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
            window_capacity,
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

    pub fn window_capacity(&self) -> VioWindowCapacity {
        self.window_capacity
    }

    pub fn has_calibrated_bias_prior(&self) -> bool {
        self.calibrated_bias_prior.is_some()
    }
}

#[cfg(feature = "vio")]
#[derive(Debug)]
pub enum VioOptimizerWorkspaceError {
    Allocation {
        buffer: &'static str,
        requested_elements: usize,
        source: TryReserveError,
    },
}

#[cfg(feature = "vio")]
impl std::fmt::Display for VioOptimizerWorkspaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Allocation {
                buffer,
                requested_elements,
                source,
            } => write!(
                f,
                "could not allocate VIO optimizer {buffer} buffer with {requested_elements} elements: {source}"
            ),
        }
    }
}

#[cfg(feature = "vio")]
impl std::error::Error for VioOptimizerWorkspaceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Allocation { source, .. } => Some(source),
        }
    }
}

#[cfg(feature = "vio")]
fn try_vio_buffer<T>(
    buffer: &'static str,
    requested_elements: usize,
) -> Result<Vec<T>, VioOptimizerWorkspaceError> {
    try_vec_with_capacity(requested_elements).map_err(|source| {
        VioOptimizerWorkspaceError::Allocation {
            buffer,
            requested_elements,
            source,
        }
    })
}

#[cfg(feature = "vio")]
fn try_zeroed_vio_f64_buffer(
    buffer: &'static str,
    requested_elements: usize,
) -> Result<Vec<f64>, VioOptimizerWorkspaceError> {
    let mut values = try_vio_buffer(buffer, requested_elements)?;
    values.resize(requested_elements, 0.0);
    Ok(values)
}

#[cfg(feature = "vio")]
#[derive(Default)]
struct VioResolvedFrameObservations {
    available: bool,
    observations: Vec<Observation>,
}

#[cfg(feature = "vio")]
impl VioResolvedFrameObservations {
    fn available(&self) -> Option<&[Observation]> {
        self.available.then_some(self.observations.as_slice())
    }
}

#[cfg(feature = "vio")]
struct VioSolveWorkspace {
    states: Vec<crate::NavState>,
    candidate_states: Vec<crate::NavState>,
    linear_solver_row_scales: Vec<f64>,
    resolved_observations: Vec<VioResolvedFrameObservations>,
    visual_support: VisualFactorSupport,
    current_linearization: VioLinearization,
    scratch_linearization: VioLinearization,
}

#[cfg(feature = "vio")]
impl VioSolveWorkspace {
    fn try_new(capacity: VioWindowCapacity) -> Result<Self, VioOptimizerWorkspaceError> {
        let frame_capacity = capacity.frames.get();
        let shape = capacity.workspace;

        let mut resolved_observations =
            try_vio_buffer("resolved-observation frame", frame_capacity)?;
        resolved_observations.resize_with(frame_capacity, VioResolvedFrameObservations::default);

        Ok(Self {
            states: try_vio_buffer("current state", frame_capacity)?,
            candidate_states: try_vio_buffer("candidate state", frame_capacity)?,
            linear_solver_row_scales: try_zeroed_vio_f64_buffer(
                "dense solver row scales",
                shape.dimension,
            )?,
            resolved_observations,
            visual_support: VisualFactorSupport::try_new(frame_capacity)?,
            current_linearization: VioLinearization::try_new(shape)?,
            scratch_linearization: VioLinearization::try_new(shape)?,
        })
    }
}

/// Stateful VIO optimizer with fixed dense workspaces sized at construction.
///
/// The optimizer is intentionally not cloneable: its buffers are scratch
/// storage, not authoritative estimator state.
#[cfg(feature = "vio")]
pub(crate) struct VioOptimizer {
    config: VioSolveConfig,
    workspace: VioSolveWorkspace,
}

#[cfg(feature = "vio")]
impl VioOptimizer {
    pub(crate) fn try_new(config: VioSolveConfig) -> Result<Self, VioOptimizerWorkspaceError> {
        let workspace = VioSolveWorkspace::try_new(config.window_capacity)?;
        Ok(Self { config, workspace })
    }

    pub(crate) fn config(&self) -> &VioSolveConfig {
        &self.config
    }

    pub(crate) fn optimize(
        &mut self,
        window: &mut VioWindow,
        map: &SlamMap,
        map_from_odom: &crate::MapFromOdom,
    ) -> Result<VioSolveResult, VioSolveError> {
        optimize_vio_with_workspace(
            window,
            &self.config,
            &mut self.workspace,
            map,
            map_from_odom,
        )
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

/// Per-factor contributions to the VIO mixed objective.
///
/// The reprojection term is a robust pixel-squared quantity. The IMU,
/// bias-random-walk, velocity-anchor, and bias-prior terms are dimensionless
/// Mahalanobis quantities. Their sum is an optimization objective, not a
/// measurement carrying one physical unit.
///
/// External callers cannot forge component values without using `new`:
/// ```compile_fail
/// use kiko_slam::VioObjectiveBreakdown;
/// let _ = VioObjectiveBreakdown {
///     reprojection_robust_px2: 0.0,
///     imu_mahalanobis: 0.0,
///     bias_random_walk_mahalanobis: 0.0,
///     velocity_anchor_mahalanobis: 0.0,
///     bias_prior_mahalanobis: 0.0,
/// };
/// ```
#[cfg(feature = "vio")]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VioObjectiveBreakdown {
    reprojection_robust_px2: f64,
    imu_mahalanobis: f64,
    bias_random_walk_mahalanobis: f64,
    velocity_anchor_mahalanobis: f64,
    bias_prior_mahalanobis: f64,
}

#[cfg(feature = "vio")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VioObjectiveComponent {
    ReprojectionRobustPx2,
    ImuMahalanobis,
    BiasRandomWalkMahalanobis,
    VelocityAnchorMahalanobis,
    BiasPriorMahalanobis,
    MixedTotal,
}

#[cfg(feature = "vio")]
impl std::fmt::Display for VioObjectiveComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::ReprojectionRobustPx2 => "robust reprojection objective (px^2)",
            Self::ImuMahalanobis => "IMU Mahalanobis objective",
            Self::BiasRandomWalkMahalanobis => "bias random-walk Mahalanobis objective",
            Self::VelocityAnchorMahalanobis => "velocity-anchor Mahalanobis objective",
            Self::BiasPriorMahalanobis => "bias-prior Mahalanobis objective",
            Self::MixedTotal => "mixed VIO objective total",
        })
    }
}

#[cfg(feature = "vio")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VioObjectiveError {
    NonFinite {
        component: VioObjectiveComponent,
        value: f64,
    },
    Negative {
        component: VioObjectiveComponent,
        value: f64,
    },
}

#[cfg(feature = "vio")]
impl std::fmt::Display for VioObjectiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite { component, value } => {
                write!(f, "VIO {component} must be finite, got {value}")
            }
            Self::Negative { component, value } => {
                write!(f, "VIO {component} must be non-negative, got {value}")
            }
        }
    }
}

#[cfg(feature = "vio")]
impl std::error::Error for VioObjectiveError {}

#[cfg(feature = "vio")]
impl VioObjectiveBreakdown {
    pub fn new(
        reprojection_robust_px2: f64,
        imu_mahalanobis: f64,
        bias_random_walk_mahalanobis: f64,
        velocity_anchor_mahalanobis: f64,
        bias_prior_mahalanobis: f64,
    ) -> Result<Self, VioObjectiveError> {
        let objective = Self {
            reprojection_robust_px2,
            imu_mahalanobis,
            bias_random_walk_mahalanobis,
            velocity_anchor_mahalanobis,
            bias_prior_mahalanobis,
        };
        objective.validate()?;
        Ok(objective)
    }

    pub fn reprojection_robust_px2(self) -> f64 {
        self.reprojection_robust_px2
    }

    pub fn imu_mahalanobis(self) -> f64 {
        self.imu_mahalanobis
    }

    pub fn bias_random_walk_mahalanobis(self) -> f64 {
        self.bias_random_walk_mahalanobis
    }

    pub fn velocity_anchor_mahalanobis(self) -> f64 {
        self.velocity_anchor_mahalanobis
    }

    pub fn bias_prior_mahalanobis(self) -> f64 {
        self.bias_prior_mahalanobis
    }

    pub fn total_mixed_objective(self) -> f64 {
        self.reprojection_robust_px2
            + self.imu_mahalanobis
            + self.bias_random_walk_mahalanobis
            + self.velocity_anchor_mahalanobis
            + self.bias_prior_mahalanobis
    }

    fn validate(self) -> Result<(), VioObjectiveError> {
        for (component, value) in [
            (
                VioObjectiveComponent::ReprojectionRobustPx2,
                self.reprojection_robust_px2,
            ),
            (VioObjectiveComponent::ImuMahalanobis, self.imu_mahalanobis),
            (
                VioObjectiveComponent::BiasRandomWalkMahalanobis,
                self.bias_random_walk_mahalanobis,
            ),
            (
                VioObjectiveComponent::VelocityAnchorMahalanobis,
                self.velocity_anchor_mahalanobis,
            ),
            (
                VioObjectiveComponent::BiasPriorMahalanobis,
                self.bias_prior_mahalanobis,
            ),
        ] {
            if !value.is_finite() {
                return Err(VioObjectiveError::NonFinite { component, value });
            }
            if value < 0.0 {
                return Err(VioObjectiveError::Negative { component, value });
            }
        }
        let total = self.total_mixed_objective();
        if !total.is_finite() {
            return Err(VioObjectiveError::NonFinite {
                component: VioObjectiveComponent::MixedTotal,
                value: total,
            });
        }
        Ok(())
    }
}

/// Criterion that terminated a converged VIO solve.
#[cfg(feature = "vio")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VioConvergenceCriterion {
    ComponentwiseStepAndRelativeObjective,
}

#[cfg(feature = "vio")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VioSolveTermination {
    NotRequired,
    Converged { criterion: VioConvergenceCriterion },
    IterationLimit,
    StalledNoObjectiveImprovement,
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
    WindowExceedsConfiguredCapacity {
        actual_frames: usize,
        capacity_frames: usize,
    },
    WorkspaceGrowth {
        buffer: &'static str,
        frame_index: usize,
        requested_elements: usize,
        source: TryReserveError,
    },
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
    InvalidObjective {
        stage: VioEvaluationStage,
        iteration: usize,
        source: VioObjectiveError,
    },
    InvalidOutcome {
        source: VioSolveOutcomeError,
    },
}

#[cfg(feature = "vio")]
impl std::fmt::Display for VioSolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WindowExceedsConfiguredCapacity {
                actual_frames,
                capacity_frames,
            } => write!(
                f,
                "VIO window has {actual_frames} frames but optimizer capacity is {capacity_frames}"
            ),
            Self::WorkspaceGrowth {
                buffer,
                frame_index,
                requested_elements,
                source,
            } => write!(
                f,
                "could not grow VIO {buffer} buffer for frame {frame_index} to {requested_elements} elements: {source}"
            ),
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
            Self::InvalidObjective {
                stage,
                iteration,
                source,
            } => write!(
                f,
                "VIO {stage} objective is invalid at iteration {iteration}: {source}"
            ),
            Self::InvalidOutcome { source } => {
                write!(f, "VIO solver produced an invalid outcome: {source}")
            }
        }
    }
}

#[cfg(feature = "vio")]
impl std::error::Error for VioSolveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::WorkspaceGrowth { source, .. } => Some(source),
            Self::Observation { source } => Some(source),
            Self::LinearSolve { source, .. } => Some(source),
            Self::StateRetraction { source, .. } => Some(source),
            Self::ImuFactor { source, .. } => Some(source),
            Self::ImuJacobian { source, .. } => Some(source),
            Self::BiasRandomWalkFactor { source, .. } => Some(source),
            Self::ReprojectionJacobianRetraction { source, .. } => Some(source),
            Self::InvalidObjective { source, .. } => Some(source),
            Self::InvalidOutcome { source } => Some(source),
            Self::WindowExceedsConfiguredCapacity { .. }
            | Self::ReprojectionFactorUnavailable { .. }
            | Self::ReprojectionJacobianUnavailable { .. }
            | Self::NonFiniteReprojectionJacobian { .. }
            | Self::NonFiniteLinearization { .. } => None,
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

#[cfg(feature = "vio")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VioSolveOutcomeError {
    StepCountOverflow {
        accepted_steps: usize,
        rejected_steps: usize,
    },
    NonprojectableRejectionsExceedRejections {
        nonprojectable_rejections: usize,
        rejected_steps: usize,
    },
    TerminationIncompatibleWithSteps {
        termination: VioSolveTermination,
        accepted_steps: usize,
        rejected_steps: usize,
    },
}

#[cfg(feature = "vio")]
impl std::fmt::Display for VioSolveOutcomeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StepCountOverflow {
                accepted_steps,
                rejected_steps,
            } => write!(
                f,
                "VIO attempted-step count overflows usize: {accepted_steps} accepted + {rejected_steps} rejected"
            ),
            Self::NonprojectableRejectionsExceedRejections {
                nonprojectable_rejections,
                rejected_steps,
            } => write!(
                f,
                "VIO nonprojectable candidate rejections ({nonprojectable_rejections}) exceed all rejected steps ({rejected_steps})"
            ),
            Self::TerminationIncompatibleWithSteps {
                termination,
                accepted_steps,
                rejected_steps,
            } => write!(
                f,
                "VIO termination {termination:?} is incompatible with {accepted_steps} accepted and {rejected_steps} rejected steps"
            ),
        }
    }
}

#[cfg(feature = "vio")]
impl std::error::Error for VioSolveOutcomeError {}

#[cfg(feature = "vio")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct VioSolveSteps {
    attempted: usize,
    accepted: usize,
    rejected: usize,
    rejected_nonprojectable_candidates: usize,
}

#[cfg(feature = "vio")]
impl VioSolveSteps {
    fn try_evaluated(
        termination: VioSolveTermination,
        accepted: usize,
        rejected: usize,
        rejected_nonprojectable_candidates: usize,
    ) -> Result<Self, VioSolveOutcomeError> {
        let attempted =
            accepted
                .checked_add(rejected)
                .ok_or(VioSolveOutcomeError::StepCountOverflow {
                    accepted_steps: accepted,
                    rejected_steps: rejected,
                })?;
        if rejected_nonprojectable_candidates > rejected {
            return Err(
                VioSolveOutcomeError::NonprojectableRejectionsExceedRejections {
                    nonprojectable_rejections: rejected_nonprojectable_candidates,
                    rejected_steps: rejected,
                },
            );
        }
        let compatible = match termination {
            VioSolveTermination::NotRequired => false,
            VioSolveTermination::Converged { .. } | VioSolveTermination::IterationLimit => {
                accepted > 0
            }
            VioSolveTermination::StalledNoObjectiveImprovement => accepted == 0 && rejected > 0,
        };
        if !compatible {
            return Err(VioSolveOutcomeError::TerminationIncompatibleWithSteps {
                termination,
                accepted_steps: accepted,
                rejected_steps: rejected,
            });
        }
        Ok(Self {
            attempted,
            accepted,
            rejected,
            rejected_nonprojectable_candidates,
        })
    }
}

#[cfg(feature = "vio")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct VioFactorDiagnostics {
    pub(crate) last_frame_active_visual_factor_count: usize,
    pub(crate) initially_excluded_nonprojectable_visual_factor_count: usize,
    pub(crate) regularized_imu_residual_factor_count: usize,
    pub(crate) floored_accel_bias_random_walk_factor_count: usize,
    pub(crate) floored_gyro_bias_random_walk_factor_count: usize,
}

/// Diagnostics and termination outcome from a VIO BA solve.
///
/// Fields are private so callers cannot forge contradictory step counts,
/// termination states, or objective totals.
///
/// ```compile_fail
/// use kiko_slam::{VioSolveResult, VioSolveTermination};
/// let _ = VioSolveResult {
///     termination: VioSolveTermination::NotRequired,
/// };
/// ```
#[cfg(feature = "vio")]
#[derive(Clone, Debug, PartialEq)]
pub struct VioSolveResult {
    termination: VioSolveTermination,
    steps: VioSolveSteps,
    objective_breakdown: VioObjectiveBreakdown,
    factors: VioFactorDiagnostics,
}

#[cfg(feature = "vio")]
impl VioSolveResult {
    pub(crate) fn not_required() -> Self {
        Self {
            termination: VioSolveTermination::NotRequired,
            steps: VioSolveSteps::default(),
            objective_breakdown: VioObjectiveBreakdown::default(),
            factors: VioFactorDiagnostics::default(),
        }
    }

    pub(crate) fn try_evaluated(
        termination: VioSolveTermination,
        accepted_steps: usize,
        rejected_steps: usize,
        rejected_nonprojectable_candidate_steps: usize,
        objective_breakdown: VioObjectiveBreakdown,
        factors: VioFactorDiagnostics,
    ) -> Result<Self, VioSolveOutcomeError> {
        Ok(Self {
            termination,
            steps: VioSolveSteps::try_evaluated(
                termination,
                accepted_steps,
                rejected_steps,
                rejected_nonprojectable_candidate_steps,
            )?,
            objective_breakdown,
            factors,
        })
    }

    pub fn termination(&self) -> VioSolveTermination {
        self.termination
    }

    pub fn attempted_iterations(&self) -> usize {
        self.steps.attempted
    }

    pub fn accepted_steps(&self) -> usize {
        self.steps.accepted
    }

    pub fn rejected_steps(&self) -> usize {
        self.steps.rejected
    }

    /// Candidate steps rejected because an initially active visual factor, or
    /// one of its finite-difference perturbations, became nonprojectable.
    pub fn rejected_nonprojectable_candidate_steps(&self) -> usize {
        self.steps.rejected_nonprojectable_candidates
    }

    pub fn final_mixed_objective(&self) -> f64 {
        self.objective_breakdown.total_mixed_objective()
    }

    pub fn objective_breakdown(&self) -> VioObjectiveBreakdown {
        self.objective_breakdown
    }

    /// Visual factors that were projectable in the initial state and retained
    /// for every objective evaluation. Each factor contributes two residuals.
    pub fn last_frame_active_visual_factor_count(&self) -> usize {
        self.factors.last_frame_active_visual_factor_count
    }

    /// Resolved visual observations excluded before optimization because they
    /// were nonprojectable in the initial state.
    pub fn initially_excluded_nonprojectable_visual_factor_count(&self) -> usize {
        self.factors
            .initially_excluded_nonprojectable_visual_factor_count
    }

    /// IMU factors whose mixed-unit residual covariance received the explicit
    /// block-unit diagonal regularization reported by their information type.
    pub fn regularized_imu_residual_factor_count(&self) -> usize {
        self.factors.regularized_imu_residual_factor_count
    }

    /// Bias random-walk factors whose raw accelerometer-bias variance was
    /// raised to the documented floor.
    pub fn floored_accel_bias_random_walk_factor_count(&self) -> usize {
        self.factors.floored_accel_bias_random_walk_factor_count
    }

    /// Bias random-walk factors whose raw gyroscope-bias variance was raised
    /// to the documented floor.
    pub fn floored_gyro_bias_random_walk_factor_count(&self) -> usize {
        self.factors.floored_gyro_bias_random_walk_factor_count
    }

    pub fn has_improved_estimate(&self) -> bool {
        self.steps.accepted > 0
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
    MissingObservationKeyframe {
        point_id: MapPointId,
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
    DisconnectedFromFixedPose {
        disconnected_pose_count: NonZeroUsize,
    },
    UnobservableMetricScale {
        max_camera_displacement_m: f32,
        min_required_m: f32,
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
            FullBaBuildError::MissingObservationKeyframe {
                point_id,
                keyframe_id,
            } => write!(
                f,
                "landmark {point_id:?} observation references missing keyframe {keyframe_id:?}"
            ),
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
            FullBaBuildError::DisconnectedFromFixedPose {
                disconnected_pose_count,
            } => write!(
                f,
                "full BA has {disconnected_pose_count} variable pose(s) disconnected from the exactly fixed first pose"
            ),
            FullBaBuildError::UnobservableMetricScale {
                max_camera_displacement_m,
                min_required_m,
            } => write!(
                f,
                "full BA cannot condition its exact metric-scale anchor: maximum landmark displacement from the fixed camera centre is {max_camera_displacement_m}m, required >= {min_required_m}m"
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
            | Self::MissingObservationKeyframe { .. }
            | Self::TooFewKeyframes { .. }
            | Self::DuplicateLandmarkObservation { .. }
            | Self::NoLandmarks
            | Self::PoseHasTooFewObservations { .. }
            | Self::DisconnectedFromFixedPose { .. }
            | Self::UnobservableMetricScale { .. } => None,
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
        FullBaBuildError::DisconnectedFromFixedPose {
            disconnected_pose_count,
        } => Ok(DegenerateReason::DisconnectedFromFixedPose {
            disconnected_pose_count,
        }),
        FullBaBuildError::UnobservableMetricScale { .. } => {
            Ok(DegenerateReason::UnobservableMetricScale)
        }
        FullBaBuildError::DuplicateKeyframe { keyframe_id } => {
            Err(BaExecutionError::DuplicateKeyframe { keyframe_id })
        }
        FullBaBuildError::MissingKeyframe { keyframe_id } => {
            Err(BaExecutionError::MissingKeyframe { keyframe_id })
        }
        FullBaBuildError::MissingObservationKeyframe {
            point_id,
            keyframe_id,
        } => Err(BaExecutionError::MissingObservationKeyframe {
            point_id,
            keyframe_id,
        }),
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

    fn tangent_offset(self) -> usize {
        self.0 * POSE_TANGENT_DIMENSION
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct FixedPoseIndex(usize);

impl FixedPoseIndex {
    fn as_usize(self) -> usize {
        self.0
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
struct FixedPose {
    pose: Pose,
}

#[derive(Clone, Copy, Debug)]
struct LandmarkVariable {
    point_id: MapPointId,
    position: Point3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FactorPose {
    Variable(PoseVarIndex),
    Fixed(FixedPoseIndex),
}

#[derive(Clone, Copy, Debug)]
struct ReprojectionFactor {
    pose: FactorPose,
    landmark: LandmarkVarIndex,
    pixel: Keypoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LandmarkAxis {
    X,
    Y,
    Z,
}

impl LandmarkAxis {
    fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }

    fn coordinate_m(self, point: Point3) -> f32 {
        match self {
            Self::X => point.x,
            Self::Y => point.y,
            Self::Z => point.z,
        }
    }

    fn set_coordinate_m(self, point: &mut Point3, coordinate_m: f32) {
        match self {
            Self::X => point.x = coordinate_m,
            Self::Y => point.y = coordinate_m,
            Self::Z => point.z = coordinate_m,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct MetricScaleAnchor {
    landmark: LandmarkVarIndex,
    axis: LandmarkAxis,
    initial_coordinate_m: f32,
    absolute_camera_displacement_m: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct MetricScaleAnchorError {
    max_camera_displacement_m: f32,
    min_required_m: f32,
}

impl MetricScaleAnchor {
    fn select(
        fixed_world_to_camera: Pose,
        landmarks: &[LandmarkVariable],
    ) -> Result<Self, MetricScaleAnchorError> {
        let fixed_camera_center_m = fixed_world_to_camera.inverse().translation();
        let mut best: Option<Self> = None;

        for (landmark_index, landmark) in landmarks.iter().enumerate() {
            for axis in [LandmarkAxis::X, LandmarkAxis::Y, LandmarkAxis::Z] {
                let coordinate_m = axis.coordinate_m(landmark.position);
                let displacement_m = (coordinate_m - fixed_camera_center_m[axis.index()]).abs();
                if !displacement_m.is_finite()
                    || best.is_some_and(|anchor| {
                        displacement_m <= anchor.absolute_camera_displacement_m
                    })
                {
                    continue;
                }
                best = Some(Self {
                    landmark: LandmarkVarIndex(landmark_index),
                    axis,
                    initial_coordinate_m: coordinate_m,
                    absolute_camera_displacement_m: displacement_m,
                });
            }
        }

        let max_displacement_m = best
            .map(|anchor| anchor.absolute_camera_displacement_m)
            .unwrap_or(0.0);
        match best {
            Some(anchor)
                if anchor.absolute_camera_displacement_m >= MIN_SCALE_ANCHOR_DISPLACEMENT_M =>
            {
                Ok(anchor)
            }
            _ => Err(MetricScaleAnchorError {
                max_camera_displacement_m: max_displacement_m,
                min_required_m: MIN_SCALE_ANCHOR_DISPLACEMENT_M,
            }),
        }
    }
}

#[derive(Debug)]
struct FullBaProblem {
    variable_poses: Vec<PoseVariable>,
    fixed_poses: Vec<FixedPose>,
    landmarks: Vec<LandmarkVariable>,
    factors: Vec<ReprojectionFactor>,
    metric_scale_anchor: MetricScaleAnchor,
}

fn disconnected_variable_pose_count(
    pose_count: usize,
    landmark_count: usize,
    factors: &[ReprojectionFactor],
) -> Option<NonZeroUsize> {
    let mut reachable_poses = vec![false; pose_count];
    let fixed_pose_reachable = reachable_poses.first_mut()?;
    *fixed_pose_reachable = true;
    let mut reachable_landmarks = vec![false; landmark_count];

    loop {
        let mut changed = false;
        for factor in factors {
            let FactorPose::Variable(pose) = factor.pose else {
                continue;
            };
            let pose_index = pose.as_usize();
            let landmark_index = factor.landmark.as_usize();
            if reachable_poses[pose_index] && !reachable_landmarks[landmark_index] {
                reachable_landmarks[landmark_index] = true;
                changed = true;
            }
            if reachable_landmarks[landmark_index] && !reachable_poses[pose_index] {
                reachable_poses[pose_index] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    NonZeroUsize::new(
        reachable_poses
            .into_iter()
            .filter(|reachable| !reachable)
            .count(),
    )
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

        let variable_pose_capacity = requested_window.len().min(max_window.get());
        let mut variable_poses = Vec::with_capacity(variable_pose_capacity);
        let mut seen_keyframes = HashSet::with_capacity(variable_pose_capacity);
        for &keyframe_id in requested_window.iter().take(max_window.get()) {
            if !seen_keyframes.insert(keyframe_id) {
                return Err(FullBaBuildError::DuplicateKeyframe { keyframe_id });
            }
            let entry = map
                .keyframe(keyframe_id)
                .ok_or(FullBaBuildError::MissingKeyframe { keyframe_id })?;
            variable_poses.push(PoseVariable {
                keyframe_id,
                pose: entry.pose(),
            });
        }

        if variable_poses.len() < MIN_BA_POSES {
            return Err(FullBaBuildError::TooFewKeyframes {
                required: MIN_BA_POSES,
                actual: variable_poses.len(),
            });
        }

        let mut pose_lookup = HashMap::with_capacity(variable_poses.len());
        for (idx, pose) in variable_poses.iter().enumerate() {
            pose_lookup.insert(pose.keyframe_id, PoseVarIndex(idx));
        }

        let mut fixed_poses = Vec::new();
        let mut fixed_pose_lookup = HashMap::new();
        let mut landmarks = Vec::new();
        let mut factors = Vec::new();
        let mut pose_counts = vec![0_usize; variable_poses.len()];
        let mut seen_observation_keyframes = HashSet::new();

        for (point_id, point) in map.points() {
            seen_observation_keyframes.clear();
            let mut variable_observation_count = 0_usize;
            for &obs in point.observations() {
                if !seen_observation_keyframes.insert(obs.keyframe_id()) {
                    return Err(FullBaBuildError::DuplicateLandmarkObservation {
                        point_id,
                        keyframe_id: obs.keyframe_id(),
                    });
                }
                variable_observation_count = variable_observation_count
                    .saturating_add(usize::from(pose_lookup.contains_key(&obs.keyframe_id())));
            }

            if variable_observation_count == 0
                || point.observation_count() < MIN_LANDMARK_OBSERVATIONS
            {
                continue;
            }

            let landmark_idx = LandmarkVarIndex(landmarks.len());
            landmarks.push(LandmarkVariable {
                point_id,
                position: point.position(),
            });

            // Outside-window observations retain their measured pixels and use
            // immutable camera poses, so moving this landmark cannot silently
            // discard constraints owned by the rest of the map.
            for &obs in point.observations() {
                let pixel = map
                    .keypoint(obs)
                    .map_err(|source| FullBaBuildError::MapLookup {
                        keypoint: obs,
                        source,
                    })?;
                let pose = match pose_lookup.get(&obs.keyframe_id()).copied() {
                    Some(pose_idx) => {
                        let pose_count = &mut pose_counts[pose_idx.as_usize()];
                        *pose_count = pose_count.saturating_add(1);
                        FactorPose::Variable(pose_idx)
                    }
                    None => {
                        let fixed_pose_idx =
                            match fixed_pose_lookup.get(&obs.keyframe_id()).copied() {
                                Some(index) => index,
                                None => {
                                    let entry = map.keyframe(obs.keyframe_id()).ok_or(
                                        FullBaBuildError::MissingObservationKeyframe {
                                            point_id,
                                            keyframe_id: obs.keyframe_id(),
                                        },
                                    )?;
                                    let index = FixedPoseIndex(fixed_poses.len());
                                    fixed_poses.push(FixedPose { pose: entry.pose() });
                                    fixed_pose_lookup.insert(obs.keyframe_id(), index);
                                    index
                                }
                            };
                        FactorPose::Fixed(fixed_pose_idx)
                    }
                };
                factors.push(ReprojectionFactor {
                    pose,
                    landmark: landmark_idx,
                    pixel,
                });
            }
        }

        if landmarks.is_empty() {
            return Err(FullBaBuildError::NoLandmarks);
        }

        for (idx, pose) in variable_poses.iter().enumerate() {
            if pose_counts[idx] < min_observations.get() {
                return Err(FullBaBuildError::PoseHasTooFewObservations {
                    keyframe_id: pose.keyframe_id,
                    required: min_observations.get(),
                    actual: pose_counts[idx],
                });
            }
        }

        // One exact pose and one exact scale coordinate only remove a single
        // connected component's similarity gauge.
        if let Some(disconnected_pose_count) =
            disconnected_variable_pose_count(variable_poses.len(), landmarks.len(), &factors)
        {
            return Err(FullBaBuildError::DisconnectedFromFixedPose {
                disconnected_pose_count,
            });
        }

        let metric_scale_anchor = MetricScaleAnchor::select(variable_poses[0].pose, &landmarks)
            .map_err(|error| FullBaBuildError::UnobservableMetricScale {
                max_camera_displacement_m: error.max_camera_displacement_m,
                min_required_m: error.min_required_m,
            })?;

        Ok(Self {
            variable_poses,
            fixed_poses,
            landmarks,
            factors,
            metric_scale_anchor,
        })
    }

    fn factor_pose(&self, pose: FactorPose) -> Pose {
        match pose {
            FactorPose::Variable(index) => self.variable_poses[index.as_usize()].pose,
            FactorPose::Fixed(index) => self.fixed_poses[index.as_usize()].pose,
        }
    }

    fn write_back(&self, map: &mut SlamMap) -> Result<(), crate::map::MapError> {
        let pose_updates = self
            .variable_poses
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
    fn reset(&mut self) {
        self.c = [[0.0; 3]; 3];
        self.b = [0.0; 3];
        self.links.clear();
    }

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

    fn fix_delta_axis(&mut self, axis: LandmarkAxis) {
        let axis = axis.index();
        for index in 0..3 {
            self.c[axis][index] = 0.0;
            self.c[index][axis] = 0.0;
        }
        self.c[axis][axis] = 1.0;
        self.b[axis] = 0.0;
        for link in &mut self.links {
            for row in &mut link.b {
                row[axis] = 0.0;
            }
        }
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
    pose_backup_buf: Vec<Pose>,
    a_buf: Vec<f32>,
    b_buf: Vec<f32>,
}

#[derive(Debug)]
pub enum LocalBundleAdjusterWorkspaceError {
    Allocation {
        buffer: &'static str,
        requested_elements: usize,
        source: TryReserveError,
    },
    SourceExceedsCapacity {
        buffer: &'static str,
        source_elements: usize,
        requested_capacity: usize,
    },
}

impl std::fmt::Display for LocalBundleAdjusterWorkspaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Allocation {
                buffer,
                requested_elements,
                source,
            } => write!(
                f,
                "could not allocate local BA {buffer} buffer with {requested_elements} elements: {source}"
            ),
            Self::SourceExceedsCapacity {
                buffer,
                source_elements,
                requested_capacity,
            } => write!(
                f,
                "cannot fork local BA {buffer} buffer with {source_elements} elements into capacity {requested_capacity}"
            ),
        }
    }
}

impl std::error::Error for LocalBundleAdjusterWorkspaceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Allocation { source, .. } => Some(source),
            Self::SourceExceedsCapacity { .. } => None,
        }
    }
}

fn try_buffer_with_capacity<T>(
    buffer: &'static str,
    requested_elements: usize,
) -> Result<Vec<T>, LocalBundleAdjusterWorkspaceError> {
    try_vec_with_capacity(requested_elements).map_err(|source| {
        LocalBundleAdjusterWorkspaceError::Allocation {
            buffer,
            requested_elements,
            source,
        }
    })
}

fn try_zeroed_f32_buffer(
    buffer: &'static str,
    requested_elements: usize,
) -> Result<Vec<f32>, LocalBundleAdjusterWorkspaceError> {
    let mut values = try_buffer_with_capacity(buffer, requested_elements)?;
    values.resize(requested_elements, 0.0);
    Ok(values)
}

fn try_cloned_buffer<T: Clone>(
    buffer: &'static str,
    source: &[T],
    capacity: usize,
) -> Result<Vec<T>, LocalBundleAdjusterWorkspaceError> {
    if source.len() > capacity {
        return Err(LocalBundleAdjusterWorkspaceError::SourceExceedsCapacity {
            buffer,
            source_elements: source.len(),
            requested_capacity: capacity,
        });
    }
    let mut values = try_buffer_with_capacity(buffer, capacity)?;
    values.extend_from_slice(source);
    Ok(values)
}

impl LocalBundleAdjuster {
    pub fn try_new(
        intrinsics: PinholeIntrinsics,
        config: LocalBaConfig,
    ) -> Result<Self, LocalBundleAdjusterWorkspaceError> {
        let shape = config.workspace;
        let frames = try_buffer_with_capacity("frame window", config.window())?;
        let pose_backup_buf = try_buffer_with_capacity("pose backup", config.window())?;
        let a_buf = try_zeroed_f32_buffer("dense normal matrix", shape.matrix_elements)?;
        let b_buf = try_zeroed_f32_buffer("normal-equation RHS", shape.dimension)?;
        Ok(Self {
            config,
            intrinsics,
            frames,
            pose_backup_buf,
            a_buf,
            b_buf,
        })
    }

    /// Fork the authoritative frame window into a fresh solver workspace.
    /// Scratch buffers are reinitialized because they carry no state between solves.
    pub fn try_fork(&self) -> Result<Self, LocalBundleAdjusterWorkspaceError> {
        Ok(Self {
            config: self.config,
            intrinsics: self.intrinsics,
            frames: try_cloned_buffer("frame window", &self.frames, self.config.window())?,
            pose_backup_buf: try_buffer_with_capacity("pose backup", self.config.window())?,
            a_buf: try_zeroed_f32_buffer("dense normal matrix", self.a_buf.len())?,
            b_buf: try_zeroed_f32_buffer("normal-equation RHS", self.b_buf.len())?,
        })
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

        let dim = frame_count * POSE_TANGENT_DIMENSION;
        let max_iters = self.config.max_iterations();
        let huber = self.config.huber_delta_px();
        let damping = self.config.lm().initial_lambda().max(MIN_POSE_DAMPING);

        for iter in 0..max_iters {
            let a = &mut self.a_buf[..dim * dim];
            let b = &mut self.b_buf[..dim];
            a.fill(0.0);
            b.fill(0.0);

            let mut projectable_factors = 0usize;
            for (idx, (frame, observations)) in self.frames.iter().zip(resolved).enumerate() {
                let base = idx * POSE_TANGENT_DIMENSION;
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

            for i in 0..dim {
                a[i * dim + i] += damping;
            }

            solve_linear_system(a, b, dim).map_err(|source| PoseBaError::LinearSolve {
                iteration: iter + 1,
                source,
            })?;

            let mut all_steps_converged = true;
            for i in 0..frame_count {
                let step = extract_se3_delta(b, i * POSE_TANGENT_DIMENSION);
                all_steps_converged &= se3_step_is_converged(step);
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

            if all_steps_converged {
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
        let pose_count = problem.variable_poses.len();
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

        let pose_dim = pose_count * POSE_TANGENT_DIMENSION;
        let max_iters = self.config.max_iterations();
        let huber = self.config.huber_delta_px();
        let lm_config = self.config.lm();
        let initial_cost = match full_problem_cost(problem, self.intrinsics, huber) {
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
        let mut landmark_accumulators: Vec<LandmarkAccumulator> = (0..landmark_count)
            .map(|_| LandmarkAccumulator::default())
            .collect();
        let mut schur_landmarks: Vec<LandmarkSchur> = Vec::with_capacity(landmark_count);
        let mut pose_rhs_before_schur: Vec<f32> = vec![0.0; pose_dim];

        for iter in 0..max_iters {
            pose_backup.clear();
            pose_backup.extend(problem.variable_poses.iter().map(|pv| pv.pose));
            landmark_backup.clear();
            landmark_backup.extend(problem.landmarks.iter().map(|lv| lv.position));

            let s = &mut self.a_buf[..pose_dim * pose_dim];
            let rhs = &mut self.b_buf[..pose_dim];
            s.fill(0.0);
            rhs.fill(0.0);

            let pose_damping = lm_state.lambda().max(lm_config.min_lambda());
            let landmark_damping = pose_damping.max(MIN_LANDMARK_DAMPING);

            for accumulator in &mut landmark_accumulators {
                accumulator.reset();
            }

            for factor in &problem.factors {
                let landmark_idx = factor.landmark;
                let pose = problem.factor_pose(factor.pose);
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

            // Fixing the first camera removes the rigid gauge, but left-image
            // factors do not guarantee a conditioned metric-scale baseline.
            // Preserve the stereo-initialized scale by fixing the
            // best-conditioned landmark coordinate exactly, without a
            // unit-mixing weighted residual.
            let scale_anchor = problem.metric_scale_anchor;
            debug_assert!(
                scale_anchor.absolute_camera_displacement_m >= MIN_SCALE_ANCHOR_DISPLACEMENT_M
            );
            landmark_accumulators[scale_anchor.landmark.as_usize()]
                .fix_delta_axis(scale_anchor.axis);

            pose_rhs_before_schur.copy_from_slice(rhs);

            for i in 0..pose_dim {
                s[i * pose_dim + i] += pose_damping;
            }

            schur_landmarks.clear();
            for (landmark_index, acc) in landmark_accumulators.iter_mut().enumerate() {
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
                    let base_i = link_i.pose.tangent_offset();
                    let rhs_contrib = mat63_mul_vec3(link_i.b, inv_c_b);
                    for row in 0..6 {
                        rhs[base_i + row] -= rhs_contrib[row];
                    }

                    for link_j in &acc.links {
                        let base_j = link_j.pose.tangent_offset();
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
                    links: std::mem::take(&mut acc.links),
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
            let mut all_steps_converged = true;
            for (pose_i, pose_var) in problem.variable_poses.iter_mut().enumerate() {
                let base = pose_i * POSE_TANGENT_DIMENSION;
                let delta = extract_se3_delta(rhs, base);
                all_steps_converged &= se3_step_is_converged(delta);
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
                    let pose_delta = extract_se3_delta(rhs, link.pose.tangent_offset());
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
                all_steps_converged &= norm3(delta_landmark) < LANDMARK_STEP_CONVERGENCE_M;
                for (axis, d) in delta_landmark.iter().enumerate() {
                    let d = *d as f64;
                    let gradient = schur.b[axis] as f64;
                    predicted_decrease += 0.5 * d * ((landmark_damping as f64) * d + gradient);
                }

                landmark_var.position.x += delta_landmark[0];
                landmark_var.position.y += delta_landmark[1];
                landmark_var.position.z += delta_landmark[2];
                if landmark_i == scale_anchor.landmark.as_usize() {
                    scale_anchor.axis.set_coordinate_m(
                        &mut landmark_var.position,
                        scale_anchor.initial_coordinate_m,
                    );
                }
            }
            for (accumulator, schur) in landmark_accumulators.iter_mut().zip(&mut schur_landmarks) {
                accumulator.links = std::mem::take(&mut schur.links);
            }

            let prev_cost = lm_state.prev_cost();
            let (action, candidate_cost) = match full_problem_cost(problem, self.intrinsics, huber)
            {
                Ok(cost) => (
                    lm_state.step(cost.get(), predicted_decrease, lm_config),
                    cost,
                ),
                Err(FullProblemCostError::NonProjectable { .. }) => {
                    lm_state.reject(lm_config);
                    (
                        LmAction::Reject,
                        BaCost::new(prev_cost).map_err(|source| BaExecutionError::InvalidCost {
                            stage: "retained",
                            iteration: iter + 1,
                            source,
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
                    if all_steps_converged || (prev_cost - candidate_cost.get()).abs() <= threshold
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
                    if all_steps_converged {
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
    for (pose_var, pose) in problem
        .variable_poses
        .iter_mut()
        .zip(pose_backup.iter().copied())
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
fn optimize_vio_with_workspace(
    window: &mut VioWindow,
    config: &VioSolveConfig,
    workspace: &mut VioSolveWorkspace,
    map: &SlamMap,
    map_from_odom: &crate::MapFromOdom,
) -> Result<VioSolveResult, VioSolveError> {
    use crate::vio::solve::solve_dense_f64;

    let n_frames = window.len();
    let capacity_frames = config.window_capacity.frames.get();
    if n_frames > capacity_frames {
        return Err(VioSolveError::WindowExceedsConfiguredCapacity {
            actual_frames: n_frames,
            capacity_frames,
        });
    }
    // Capacity construction proved this product and its square addressable.
    let dim = n_frames * VIO_STATE_DIM;
    if n_frames < 2 {
        return Ok(VioSolveResult::not_required());
    }

    let VioSolveWorkspace {
        states,
        candidate_states,
        linear_solver_row_scales,
        resolved_observations,
        visual_support,
        current_linearization,
        scratch_linearization,
    } = workspace;
    let max_iters = config.max_iterations.get();
    let mut lambda = f64::from(config.lm.initial_lambda);
    states.clear();
    states.extend(window.states().cloned());
    for (frame_index, resolved) in resolved_observations[..n_frames].iter_mut().enumerate() {
        resolved.available = false;
        resolved.observations.clear();
        if let Some(observations) = window.observations(frame_index) {
            resolved.available = observations.resolve_observations_into(
                map,
                config.intrinsics,
                NonZeroUsize::MIN,
                &mut resolved.observations,
            )?;
        }
    }
    visual_support.update_from_initial_states(
        states,
        resolved_observations,
        config,
        map_from_odom,
    )?;
    linearize_vio_states(
        window,
        states,
        ResolvedVioVisualFactors {
            observations: resolved_observations,
            support: visual_support,
        },
        config,
        map_from_odom,
        VioEvaluation {
            stage: VioEvaluationStage::Initial,
            iteration: 0,
        },
        current_linearization,
    )?;
    let mut current_mixed_objective = current_linearization
        .objective_breakdown
        .total_mixed_objective();
    let mut termination = None;
    let mut accepted_steps = 0;
    let mut rejected_steps = 0;
    let mut rejected_nonprojectable_candidate_steps = 0;

    for iteration in 0..max_iters {
        let attempted_iteration = iteration + 1;

        let matrix_elements = dim * dim;
        scratch_linearization.hessian[..matrix_elements]
            .copy_from_slice(&current_linearization.hessian[..matrix_elements]);
        for (destination, source) in scratch_linearization.rhs[..dim]
            .iter_mut()
            .zip(&current_linearization.rhs[..dim])
        {
            *destination = -*source;
        }
        for i in 0..dim {
            scratch_linearization.hessian[i * dim + i] += lambda;
        }
        solve_dense_f64(
            &mut scratch_linearization.hessian[..matrix_elements],
            &mut scratch_linearization.rhs[..dim],
            dim,
            &mut linear_solver_row_scales[..dim],
        )
        .map_err(|source| VioSolveError::LinearSolve {
            iteration: attempted_iteration,
            source,
        })?;
        let step_is_componentwise_small =
            vio_step_is_componentwise_small(&scratch_linearization.rhs[..dim]);

        candidate_states.clear();
        for (frame_idx, state) in states.iter().enumerate() {
            let base = frame_idx * VIO_STATE_DIM;
            let mut tangent = [0.0_f64; VIO_STATE_DIM];
            tangent.copy_from_slice(&scratch_linearization.rhs[base..base + VIO_STATE_DIM]);
            let candidate_state =
                state
                    .retract(&tangent)
                    .map_err(|source| VioSolveError::StateRetraction {
                        iteration: attempted_iteration,
                        frame_index: frame_idx,
                        source,
                    })?;
            candidate_states.push(candidate_state);
        }

        match linearize_vio_states(
            window,
            candidate_states,
            ResolvedVioVisualFactors {
                observations: resolved_observations,
                support: visual_support,
            },
            config,
            map_from_odom,
            VioEvaluation {
                stage: VioEvaluationStage::Candidate,
                iteration: attempted_iteration,
            },
            scratch_linearization,
        ) {
            Ok(()) => {}
            Err(error) if error.is_rejected_candidate_nonprojectability() => {
                rejected_steps += 1;
                rejected_nonprojectable_candidate_steps += 1;
                lambda = (lambda * f64::from(config.lm.lambda_factor))
                    .min(f64::from(config.lm.max_lambda));
                continue;
            }
            Err(error) => return Err(error),
        }
        let candidate_mixed_objective = scratch_linearization
            .objective_breakdown
            .total_mixed_objective();
        if candidate_mixed_objective < current_mixed_objective {
            let objective_decrease = current_mixed_objective - candidate_mixed_objective;
            std::mem::swap(states, candidate_states);
            std::mem::swap(current_linearization, scratch_linearization);
            current_mixed_objective = candidate_mixed_objective;
            accepted_steps += 1;
            lambda =
                (lambda / f64::from(config.lm.lambda_factor)).max(f64::from(config.lm.min_lambda));
            let relative_objective_scale = current_mixed_objective
                .abs()
                .max(VIO_RELATIVE_OBJECTIVE_SCALE_FLOOR);
            if step_is_componentwise_small
                && objective_decrease
                    < VIO_RELATIVE_OBJECTIVE_CONVERGENCE_TOLERANCE * relative_objective_scale
            {
                termination = Some(VioSolveTermination::Converged {
                    criterion: VioConvergenceCriterion::ComponentwiseStepAndRelativeObjective,
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
    let termination = termination.unwrap_or(if accepted_steps == 0 && rejected_steps > 0 {
        VioSolveTermination::StalledNoObjectiveImprovement
    } else {
        VioSolveTermination::IterationLimit
    });
    VioSolveResult::try_evaluated(
        termination,
        accepted_steps,
        rejected_steps,
        rejected_nonprojectable_candidate_steps,
        current_linearization.objective_breakdown,
        VioFactorDiagnostics {
            last_frame_active_visual_factor_count: visual_support.last_frame_factor_count(),
            initially_excluded_nonprojectable_visual_factor_count: visual_support
                .initially_excluded_nonprojectable_factor_count,
            regularized_imu_residual_factor_count,
            floored_accel_bias_random_walk_factor_count,
            floored_gyro_bias_random_walk_factor_count,
        },
    )
    .map_err(|source| VioSolveError::InvalidOutcome { source })
}

#[cfg(feature = "vio")]
struct VioLinearization {
    hessian: Vec<f64>,
    rhs: Vec<f64>,
    objective_breakdown: VioObjectiveBreakdown,
}

#[cfg(feature = "vio")]
impl VioLinearization {
    fn try_new(shape: DenseWorkspaceShape) -> Result<Self, VioOptimizerWorkspaceError> {
        Ok(Self {
            hessian: try_zeroed_vio_f64_buffer("dense normal matrix", shape.matrix_elements)?,
            rhs: try_zeroed_vio_f64_buffer("normal-equation RHS", shape.dimension)?,
            objective_breakdown: VioObjectiveBreakdown::default(),
        })
    }

    fn reset(&mut self, state_dimension: usize) {
        self.hessian[..state_dimension * state_dimension].fill(0.0);
        self.rhs[..state_dimension].fill(0.0);
        self.objective_breakdown = VioObjectiveBreakdown::default();
    }
}

#[cfg(feature = "vio")]
struct VisualFactorSupport {
    observation_indices_by_frame: Vec<Vec<usize>>,
    active_frame_count: usize,
    initially_excluded_nonprojectable_factor_count: usize,
}

#[cfg(feature = "vio")]
struct ResolvedVioVisualFactors<'a> {
    observations: &'a [VioResolvedFrameObservations],
    support: &'a VisualFactorSupport,
}

#[cfg(feature = "vio")]
impl VisualFactorSupport {
    fn try_new(frame_capacity: usize) -> Result<Self, VioOptimizerWorkspaceError> {
        let mut observation_indices_by_frame =
            try_vio_buffer("visual-support frame", frame_capacity)?;
        observation_indices_by_frame.resize_with(frame_capacity, Vec::new);
        Ok(Self {
            observation_indices_by_frame,
            active_frame_count: 0,
            initially_excluded_nonprojectable_factor_count: 0,
        })
    }

    fn update_from_initial_states(
        &mut self,
        states: &[crate::NavState],
        resolved_observations: &[VioResolvedFrameObservations],
        config: &VioSolveConfig,
        map_from_odom: &crate::MapFromOdom,
    ) -> Result<(), VioSolveError> {
        debug_assert!(resolved_observations.len() >= states.len());
        debug_assert!(self.observation_indices_by_frame.len() >= states.len());
        self.active_frame_count = states.len();
        self.initially_excluded_nonprojectable_factor_count = 0;

        for (frame_index, state) in states.iter().enumerate() {
            let active_indices = &mut self.observation_indices_by_frame[frame_index];
            active_indices.clear();
            if let Some(resolved) = resolved_observations[frame_index].available() {
                if active_indices.capacity() < resolved.len() {
                    active_indices
                        .try_reserve_exact(resolved.len())
                        .map_err(|source| VioSolveError::WorkspaceGrowth {
                            buffer: "visual-factor support",
                            frame_index,
                            requested_elements: resolved.len(),
                            source,
                        })?;
                }
                let cam_from_map = vio_cam_from_map(state, config.camera_from_body, map_from_odom);
                for (observation_index, observation) in resolved.iter().enumerate() {
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
                        self.initially_excluded_nonprojectable_factor_count += 1;
                    }
                }
            }
        }
        Ok(())
    }

    fn indices_for_frame(&self, frame_index: usize) -> &[usize] {
        &self.observation_indices_by_frame[frame_index]
    }

    fn last_frame_factor_count(&self) -> usize {
        self.active_frame_count
            .checked_sub(1)
            .map_or(0, |index| self.observation_indices_by_frame[index].len())
    }
}

#[cfg(feature = "vio")]
fn linearize_vio_states(
    window: &VioWindow,
    states: &[crate::NavState],
    visual_factors: ResolvedVioVisualFactors<'_>,
    config: &VioSolveConfig,
    map_from_odom: &crate::MapFromOdom,
    evaluation: VioEvaluation,
    output: &mut VioLinearization,
) -> Result<(), VioSolveError> {
    use crate::ImuFactor;
    use crate::vio::bias_random_walk_residual;
    use crate::vio::solve::{
        STATE_DIM, accumulate_factor, accumulate_paired_factor, imu_jacobians,
    };
    let VioEvaluation { stage, iteration } = evaluation;
    let ResolvedVioVisualFactors {
        observations: resolved_observations,
        support: visual_support,
    } = visual_factors;

    debug_assert_eq!(states.len(), window.len());
    let n_frames = window.len();
    debug_assert!(resolved_observations.len() >= n_frames);
    debug_assert!(visual_support.observation_indices_by_frame.len() >= n_frames);
    let dim = n_frames * STATE_DIM;
    output.reset(dim);
    let VioLinearization {
        hessian,
        rhs,
        objective_breakdown,
    } = output;
    let hessian = &mut hessian[..dim * dim];
    let rhs = &mut rhs[..dim];

    for (frame_idx, state) in states.iter().enumerate() {
        let base = frame_idx * STATE_DIM;
        let cam_pose = vio_cam_from_map(state, config.camera_from_body, map_from_odom);
        if let Some(resolved) = resolved_observations[frame_idx].available() {
            for &observation_index in visual_support.indices_for_frame(frame_idx) {
                let obs = &resolved[observation_index];
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
                let residual_norm_px = residual_f64[0].hypot(residual_f64[1]);
                let huber_delta_px = config.huber_delta_px;
                let (huber_weight, huber_objective_px2) = if residual_norm_px <= huber_delta_px {
                    (1.0, 0.5 * residual_norm_px * residual_norm_px)
                } else {
                    (
                        huber_delta_px / residual_norm_px,
                        huber_delta_px * (residual_norm_px - 0.5 * huber_delta_px),
                    )
                };
                objective_breakdown.reprojection_robust_px2 += huber_objective_px2;
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
                accumulate_factor(hessian, rhs, dim, &j_15, &identity_2, &r_f64, base);
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

        let base_prev = succ_idx * STATE_DIM;
        let base_curr = (succ_idx + 1) * STATE_DIM;
        let imu_squared_mahalanobis = accumulate_paired_factor(
            hessian,
            rhs,
            dim,
            [&j_prev, &j_curr],
            info,
            &residual,
            [base_prev, base_curr],
        );
        objective_breakdown.imu_mahalanobis += 0.5 * imu_squared_mahalanobis;
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

        let mut bias_squared_mahalanobis = 0.0;
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
            bias_squared_mahalanobis += residual * weighted_residual;
            hessian[previous_index * dim + previous_index] += information;
            hessian[current_index * dim + current_index] += information;
            hessian[previous_index * dim + current_index] -= information;
            hessian[current_index * dim + previous_index] -= information;
            rhs[previous_index] -= weighted_residual;
            rhs[current_index] += weighted_residual;
        }
        objective_breakdown.bias_random_walk_mahalanobis += 0.5 * bias_squared_mahalanobis;
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
            objective_breakdown.velocity_anchor_mahalanobis +=
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
                objective_breakdown.bias_prior_mahalanobis +=
                    0.5 * accel_information_s4_per_m2 * accel_residual * accel_residual;

                let gyro_residual = gyro_radps[axis] - calibrated_gyro_radps[axis];
                hessian[(base + 12 + axis) * dim + (base + 12 + axis)] +=
                    gyro_information_s2_per_rad2;
                rhs[base + 12 + axis] += gyro_information_s2_per_rad2 * gyro_residual;
                objective_breakdown.bias_prior_mahalanobis +=
                    0.5 * gyro_information_s2_per_rad2 * gyro_residual * gyro_residual;
            }
        }
    }

    validate_vio_linearization_values(output, dim, evaluation)
}

#[cfg(feature = "vio")]
fn validate_vio_linearization_values(
    linearization: &VioLinearization,
    state_dimension: usize,
    evaluation: VioEvaluation,
) -> Result<(), VioSolveError> {
    for (quantity, values) in [
        (
            VioLinearizationQuantity::Hessian,
            &linearization.hessian[..state_dimension * state_dimension],
        ),
        (
            VioLinearizationQuantity::RightHandSide,
            &linearization.rhs[..state_dimension],
        ),
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
    linearization
        .objective_breakdown
        .validate()
        .map_err(|source| VioSolveError::InvalidObjective {
            stage: evaluation.stage,
            iteration: evaluation.iteration,
            source,
        })
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
) -> Result<BaCost, FullProblemCostError> {
    let mut cost = 0.0_f64;
    let huber = huber_delta_px as f64;
    let mut nonprojectable_count = 0_usize;

    for factor in &problem.factors {
        let pose = problem.factor_pose(factor.pose);
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
    let base = pose_idx.tangent_offset();
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
    let base = pose_idx.tangent_offset();
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
    let base = pose_idx.tangent_offset();
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

fn se3_step_is_converged(delta: [f32; 6]) -> bool {
    norm3([delta[0], delta[1], delta[2]]) < POSE_TRANSLATION_STEP_CONVERGENCE_M
        && norm3([delta[3], delta[4], delta[5]]) < POSE_ROTATION_STEP_CONVERGENCE_RAD
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

    fn make_bundle_adjuster(
        intrinsics: PinholeIntrinsics,
        config: LocalBaConfig,
    ) -> LocalBundleAdjuster {
        LocalBundleAdjuster::try_new(intrinsics, config).expect("allocate local BA workspace")
    }

    fn l2_3(a: [f32; 3], b: [f32; 3]) -> f32 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    fn lm(initial_lambda: f32) -> LmConfig {
        LmConfig::new(initial_lambda, 10.0, 1e-8, 1e4, 0.25, 0.75).expect("valid lm")
    }

    #[cfg(feature = "vio")]
    fn vio_window_capacity() -> VioWindowCapacity {
        VioWindowCapacity::new(5).expect("valid test VIO window capacity")
    }

    #[cfg(feature = "vio")]
    fn test_preintegrated_imu() -> crate::PreintegratedImu {
        let batch = crate::ImuBatch::new(vec![
            crate::ImuSample::new(Timestamp::from_nanos(0), [0.0; 3], [0.0; 3])
                .expect("first IMU sample"),
            crate::ImuSample::new(Timestamp::from_nanos(10_000_000), [0.0; 3], [0.0; 3])
                .expect("second IMU sample"),
        ])
        .expect("IMU batch");
        crate::PreintegratedImu::integrate(
            &batch,
            &crate::ImuBias::default(),
            &crate::ImuNoiseModel::new(0.1, 0.01, 0.001, 0.0001).expect("noise"),
        )
        .expect("preintegration")
    }

    #[cfg(feature = "vio")]
    fn test_vio_solve_config(window_frames: usize) -> VioSolveConfig {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        VioSolveConfig::new(
            VioWindowCapacity::new(window_frames).expect("window capacity"),
            crate::Gravity::try_new([0.0, 9.81, 0.0]).expect("gravity"),
            crate::Pose64::identity(),
            intrinsics,
            lm(1e-3),
            NonZeroUsize::new(2).expect("iterations"),
            2.0,
            1.0,
            None,
        )
        .expect("VIO solve config")
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

    fn add_fixed_observation_keyframe(
        map: &mut SlamMap,
        intrinsics: PinholeIntrinsics,
        source_keyframe: KeyframeId,
        points_world_m: &[Point3],
        world_to_camera: Pose,
        omitted_point_index: Option<usize>,
    ) -> KeyframeId {
        let keypoints = points_world_m
            .iter()
            .copied()
            .map(|point| {
                project_world_point(world_to_camera, point, intrinsics)
                    .expect("fixture point visible in fixed camera")
            })
            .collect();
        let detections =
            make_detections(SensorId::StereoLeft, FrameId::new(502), 640, 480, keypoints)
                .expect("fixed-camera detections");
        let fixed_keyframe = map
            .add_keyframe_from_detections(
                detections.as_ref(),
                Timestamp::from_nanos(3_000_000),
                world_to_camera,
            )
            .expect("insert fixed-camera keyframe");

        for index in 0..points_world_m.len() {
            if omitted_point_index == Some(index) {
                continue;
            }
            let source_keypoint = map
                .keyframe_keypoint(source_keyframe, index)
                .expect("source keypoint");
            let point_id = map
                .map_point_for_keypoint(source_keypoint)
                .expect("source association lookup")
                .expect("source map point");
            let fixed_keypoint = map
                .keyframe_keypoint(fixed_keyframe, index)
                .expect("fixed keypoint");
            map.add_observation(point_id, fixed_keypoint)
                .expect("add fixed-camera observation");
        }
        fixed_keyframe
    }

    fn make_full_ba_problem(
        variable_poses: Vec<PoseVariable>,
        fixed_poses: Vec<FixedPose>,
        landmarks: Vec<LandmarkVariable>,
        factors: Vec<ReprojectionFactor>,
    ) -> FullBaProblem {
        let metric_scale_anchor = MetricScaleAnchor::select(variable_poses[0].pose, &landmarks)
            .expect("test problem has an observable metric-scale anchor");
        FullBaProblem {
            variable_poses,
            fixed_poses,
            landmarks,
            factors,
            metric_scale_anchor,
        }
    }

    fn dummy_metric_scale_anchor() -> MetricScaleAnchor {
        MetricScaleAnchor {
            landmark: LandmarkVarIndex(0),
            axis: LandmarkAxis::X,
            initial_coordinate_m: 0.0,
            absolute_camera_displacement_m: MIN_SCALE_ANCHOR_DISPLACEMENT_M,
        }
    }

    fn scale_problem_about_fixed_camera(problem: &mut FullBaProblem, scale: f32) {
        let fixed_camera_center_m = problem.variable_poses[0].pose.inverse().translation();
        for pose in &mut problem.variable_poses {
            let camera_center_m = pose.pose.inverse().translation();
            let scaled_center_m = std::array::from_fn(|axis| {
                fixed_camera_center_m[axis]
                    + scale * (camera_center_m[axis] - fixed_camera_center_m[axis])
            });
            let rotation = pose.pose.rotation();
            let rotated_center = math::mat_mul_vec(rotation, scaled_center_m);
            pose.pose = Pose::from_rt(
                rotation,
                [-rotated_center[0], -rotated_center[1], -rotated_center[2]],
            );
        }
        for landmark in &mut problem.landmarks {
            landmark.position.x =
                fixed_camera_center_m[0] + scale * (landmark.position.x - fixed_camera_center_m[0]);
            landmark.position.y =
                fixed_camera_center_m[1] + scale * (landmark.position.y - fixed_camera_center_m[1]);
            landmark.position.z =
                fixed_camera_center_m[2] + scale * (landmark.position.z - fixed_camera_center_m[2]);
        }
    }

    #[test]
    fn local_ba_config_rejects_invalid_values() {
        assert!(matches!(
            LocalBaConfig::new(0, 10, 4, 1.0, lm(1e-3)),
            Err(LocalBaConfigError::ZeroWindow)
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
        assert!(matches!(
            LocalBaConfig::new(usize::MAX, 10, 4, 1.0, lm(1e-3)),
            Err(LocalBaConfigError::PoseDimensionOverflow { .. })
        ));

        let element_overflow_window = usize::MAX / POSE_TANGENT_DIMENSION;
        assert!(matches!(
            LocalBaConfig::new(element_overflow_window, 10, 4, 1.0, lm(1e-3)),
            Err(LocalBaConfigError::DenseMatrixElementCountOverflow { .. })
        ));

        let near_sqrt_usize_max = 1_usize << (usize::BITS / 2 - 1);
        let byte_overflow_window = near_sqrt_usize_max / POSE_TANGENT_DIMENSION;
        assert!(matches!(
            LocalBaConfig::new(byte_overflow_window, 10, 4, 1.0, lm(1e-3)),
            Err(LocalBaConfigError::DenseMatrixByteLengthUnaddressable { .. })
        ));
    }

    #[test]
    fn local_ba_workspace_is_checked_and_preallocated_once() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let config = LocalBaConfig::new(5, 10, 4, 1.0, lm(1e-3)).expect("valid BA config");

        let ba = LocalBundleAdjuster::try_new(intrinsics, config).expect("allocate BA workspace");

        assert_eq!(ba.a_buf.len(), 30 * 30);
        assert_eq!(ba.b_buf.len(), 30);
        assert!(ba.frames.capacity() >= 5);
        assert!(ba.pose_backup_buf.capacity() >= 5);
    }

    #[test]
    fn local_ba_allocation_error_preserves_allocator_source() {
        let error = try_buffer_with_capacity::<u8>("test", usize::MAX)
            .expect_err("an unaddressable vector capacity must fail");

        assert!(std::error::Error::source(&error).is_some());
    }

    #[test]
    fn local_ba_fork_rejects_source_larger_than_declared_capacity() {
        let error = try_cloned_buffer("test", &[1_u8, 2], 1)
            .expect_err("source larger than capacity must fail explicitly");

        assert!(matches!(
            &error,
            LocalBundleAdjusterWorkspaceError::SourceExceedsCapacity {
                source_elements: 2,
                requested_capacity: 1,
                ..
            }
        ));
        assert!(std::error::Error::source(&error).is_none());
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
        let config = LocalBaConfig::new(5, 10, 4, 2.0, lm(1e-3)).expect("valid BA config");
        let mut ba = make_bundle_adjuster(intrinsics, config);

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
        let config = LocalBaConfig::new(1, 3, 4, 2.0, lm(1e-3)).expect("BA config");
        let mut ba = make_bundle_adjuster(intrinsics, config);
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
    fn se3_convergence_checks_translation_and_rotation_in_their_own_units() {
        assert!(se3_step_is_converged([
            0.5 * POSE_TRANSLATION_STEP_CONVERGENCE_M,
            0.0,
            0.0,
            0.5 * POSE_ROTATION_STEP_CONVERGENCE_RAD,
            0.0,
            0.0,
        ]));
        assert!(!se3_step_is_converged([
            POSE_TRANSLATION_STEP_CONVERGENCE_M,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]));
        assert!(!se3_step_is_converged([
            0.0,
            0.0,
            0.0,
            POSE_ROTATION_STEP_CONVERGENCE_RAD,
            0.0,
            0.0,
        ]));
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
    fn full_problem_cost_rejects_nonprojectable_factors() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let mut problem = make_full_ba_problem(
            vec![
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: Pose::identity(),
                },
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: Pose::identity(),
                },
            ],
            Vec::new(),
            vec![LandmarkVariable {
                point_id: MapPointId::default(),
                position: Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: -1.0,
                },
            }],
            vec![ReprojectionFactor {
                pose: FactorPose::Variable(PoseVarIndex(0)),
                landmark: LandmarkVarIndex(0),
                pixel: Keypoint { x: 320.0, y: 240.0 },
            }],
        );

        let error = full_problem_cost(&problem, intrinsics, 2.0)
            .expect_err("a behind-camera factor cannot disappear from cost");
        assert!(matches!(
            error,
            FullProblemCostError::NonProjectable { count } if count.get() == 1
        ));

        let config = LocalBaConfig::new(5, 2, 4, 2.0, lm(1e-3)).expect("BA config");
        let mut ba = make_bundle_adjuster(intrinsics, config);
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
        let config = LocalBaConfig::new(5, 15, 4, 2.0, lm(1e-3)).expect("valid BA config");
        let mut ba = make_bundle_adjuster(intrinsics, config);

        let mut no_poses = FullBaProblem {
            variable_poses: Vec::new(),
            fixed_poses: Vec::new(),
            landmarks: Vec::new(),
            factors: Vec::new(),
            metric_scale_anchor: dummy_metric_scale_anchor(),
        };
        assert!(matches!(
            ba.optimize_full(&mut no_poses),
            Ok(BaResult::Degenerate {
                reason: DegenerateReason::TooFewPoses { count: 0 }
            })
        ));

        let mut no_landmarks = FullBaProblem {
            variable_poses: vec![
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: Pose::identity(),
                },
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: Pose::identity(),
                },
            ],
            fixed_poses: Vec::new(),
            landmarks: Vec::new(),
            factors: Vec::new(),
            metric_scale_anchor: dummy_metric_scale_anchor(),
        };
        assert!(matches!(
            ba.optimize_full(&mut no_landmarks),
            Ok(BaResult::Degenerate {
                reason: DegenerateReason::TooFewLandmarks { count: 0 }
            })
        ));

        let mut no_factors = FullBaProblem {
            variable_poses: vec![
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: Pose::identity(),
                },
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: Pose::identity(),
                },
            ],
            fixed_poses: Vec::new(),
            landmarks: vec![LandmarkVariable {
                point_id: MapPointId::default(),
                position: Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 2.0,
                },
            }],
            factors: Vec::new(),
            metric_scale_anchor: dummy_metric_scale_anchor(),
        };
        assert!(matches!(
            ba.optimize_full(&mut no_factors),
            Ok(BaResult::Degenerate {
                reason: DegenerateReason::NoFactors
            })
        ));
    }

    #[test]
    fn exact_metric_scale_anchor_removes_one_landmark_degree_of_freedom() {
        let mut accumulator = LandmarkAccumulator {
            c: [[4.0, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 6.0]],
            b: [7.0, 8.0, 9.0],
            links: vec![PoseLandmarkCross {
                pose: PoseVarIndex(1),
                b: [[1.0, 2.0, 3.0]; 6],
            }],
        };

        accumulator.fix_delta_axis(LandmarkAxis::Y);

        assert_eq!(accumulator.c[1], [0.0, 1.0, 0.0]);
        assert_eq!([accumulator.c[0][1], accumulator.c[2][1]], [0.0, 0.0]);
        assert_eq!(accumulator.b[1], 0.0);
        assert!(accumulator.links[0].b.iter().all(|row| row[1] == 0.0));

        let delta = math::mat_mul_vec(
            invert_3x3(accumulator.c).expect("anchored block is invertible"),
            accumulator.b,
        );
        assert_eq!(delta[1], 0.0);
    }

    #[test]
    fn optimize_full_returns_iteration_limit_with_bad_init() {
        let (map, intrinsics, kf_0, kf_1, _, _) =
            build_full_ba_fixture([0.8, -0.3, 0.4, 0.2, -0.1, 0.15]);
        let config = LocalBaConfig::new(5, 1, 4, 2.0, lm(1e-3)).expect("valid BA config");
        let mut ba = make_bundle_adjuster(intrinsics, config);
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
                pose: FactorPose::Variable(pose),
                landmark: LandmarkVarIndex(0),
                pixel: Keypoint { x: 310.0, y: 240.0 },
            });
            factors.push(ReprojectionFactor {
                pose: FactorPose::Variable(pose),
                landmark: LandmarkVarIndex(0),
                pixel: Keypoint { x: 330.0, y: 240.0 },
            });
        }
        let mut problem = make_full_ba_problem(
            vec![
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: Pose::identity(),
                },
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: Pose::identity(),
                },
            ],
            Vec::new(),
            vec![LandmarkVariable {
                point_id: MapPointId::default(),
                position: point,
            }],
            factors,
        );
        let config = LocalBaConfig::new(5, 3, 4, 2.0, lm(1e-3)).expect("BA config");
        let mut ba = make_bundle_adjuster(intrinsics, config);

        let result = ba.optimize_full(&mut problem).expect("stationary solve");

        assert!(matches!(
            result,
            BaResult::Stationary(stationary)
                if stationary.detected_at_iteration() == NonZeroUsize::MIN
        ));
        assert!(!result.is_applicable());
        assert!(pose_close(
            problem.variable_poses[1].pose,
            Pose::identity(),
            0.0
        ));
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
        let mut problem = make_full_ba_problem(
            vec![
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: Pose::identity(),
                },
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: Pose::identity(),
                },
            ],
            Vec::new(),
            vec![LandmarkVariable {
                point_id: MapPointId::default(),
                position: original_point,
            }],
            vec![
                ReprojectionFactor {
                    pose: FactorPose::Variable(PoseVarIndex(0)),
                    landmark: LandmarkVarIndex(0),
                    pixel: Keypoint {
                        x: 10_000.0,
                        y: 240.0,
                    },
                },
                ReprojectionFactor {
                    pose: FactorPose::Variable(PoseVarIndex(1)),
                    landmark: LandmarkVarIndex(0),
                    pixel: Keypoint {
                        x: 10_000.0,
                        y: 240.0,
                    },
                },
            ],
        );
        let config = LocalBaConfig::new(5, 1, 4, 2.0, lm(1e4)).expect("BA config");
        let mut ba = make_bundle_adjuster(intrinsics, config);
        let original_poses: Vec<_> = problem
            .variable_poses
            .iter()
            .map(|pose| pose.pose)
            .collect();

        let result = ba.optimize_full(&mut problem).expect("stalled solve");

        assert!(matches!(result, BaResult::Stalled(_)), "got {result:?}");
        assert!(!result.is_applicable());
        assert!(
            problem
                .variable_poses
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
        let config = LocalBaConfig::new(5, 15, 4, 2.0, lm(1e-3)).expect("valid BA config");
        let mut ba = make_bundle_adjuster(intrinsics, config);
        let mut problem = FullBaProblem::try_from_map(
            &map,
            &[kf_0, kf_1],
            ba.window_size(),
            ba.min_observations(),
        )
        .expect("full BA problem");
        let scale_anchor = problem.metric_scale_anchor;
        let anchored_coordinate_before = scale_anchor
            .axis
            .coordinate_m(problem.landmarks[scale_anchor.landmark.as_usize()].position);
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
        assert_eq!(
            scale_anchor
                .axis
                .coordinate_m(problem.landmarks[scale_anchor.landmark.as_usize()].position),
            anchored_coordinate_before
        );
    }

    #[test]
    fn optimize_full_recovers_from_large_perturbation() {
        let (map, intrinsics, kf_0, kf_1, _, _) =
            build_full_ba_fixture([0.45, -0.20, 0.28, 0.15, -0.08, 0.10]);
        let config = LocalBaConfig::new(5, 30, 4, 2.0, lm(1e-3)).expect("valid BA config");
        let mut ba = make_bundle_adjuster(intrinsics, config);
        let mut problem = FullBaProblem::try_from_map(
            &map,
            &[kf_0, kf_1],
            ba.window_size(),
            ba.min_observations(),
        )
        .expect("full BA problem");
        let before = full_problem_cost(&problem, intrinsics, config.huber_delta_px())
            .expect("initial factors are projectable")
            .get();
        let result = ba.optimize_full(&mut problem).expect("full BA solve");
        let after = full_problem_cost(&problem, intrinsics, config.huber_delta_px())
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
        let config = LocalBaConfig::new(5, 20, 4, 2.0, lm(1e-3)).expect("valid BA config");
        let mut ba = make_bundle_adjuster(intrinsics, config);
        let mut problem = FullBaProblem::try_from_map(
            &map,
            &[kf_0, kf_1],
            ba.window_size(),
            ba.min_observations(),
        )
        .expect("full BA problem");
        let before = full_problem_cost(&problem, intrinsics, config.huber_delta_px())
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
    fn local_bundle_adjuster_try_fork_supports_transactional_candidate_updates() {
        let (map, intrinsics, kf_0, _kf_1, _, _) =
            build_full_ba_fixture([0.12, -0.05, 0.07, 0.03, -0.02, 0.01]);
        let config = LocalBaConfig::new(5, 20, 4, 2.0, lm(1e-3)).expect("valid BA config");
        let mut ba = make_bundle_adjuster(intrinsics, config);
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

        let mut candidate_ba = ba.try_fork().expect("allocate candidate BA workspace");
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
    fn full_ba_retains_fixed_observations_for_selected_landmarks() {
        let (mut map, intrinsics, keyframe_0, keyframe_1, _, points_world_m) =
            build_full_ba_fixture([0.04, -0.01, 0.02, 0.01, -0.005, 0.004]);
        let variable_pose_1 = axis_angle_pose([0.36, 0.01, 0.02], [0.0, -0.02, 0.015]);
        let keyframe_2 = add_fixed_observation_keyframe(
            &mut map,
            intrinsics,
            keyframe_0,
            &points_world_m,
            variable_pose_1,
            Some(0),
        );
        let config = LocalBaConfig::new(5, 5, 4, 2.0, lm(1e-3)).expect("BA config");
        let ba = make_bundle_adjuster(intrinsics, config);
        let mut problem = FullBaProblem::try_from_map(
            &map,
            &[keyframe_0, keyframe_2],
            ba.window_size(),
            ba.min_observations(),
        )
        .expect("full BA problem with fixed observations");

        assert_eq!(problem.fixed_poses.len(), 1);
        let point_0 = map
            .map_point_for_keypoint(
                map.keyframe_keypoint(keyframe_0, 0)
                    .expect("keyframe-0 keypoint"),
            )
            .expect("point lookup")
            .expect("point association");
        let landmark_0 = problem
            .landmarks
            .iter()
            .position(|landmark| landmark.point_id == point_0)
            .map(LandmarkVarIndex)
            .expect("landmark with one variable and one fixed observation is retained");
        let landmark_0_factors: Vec<_> = problem
            .factors
            .iter()
            .filter(|factor| factor.landmark == landmark_0)
            .collect();
        assert_eq!(landmark_0_factors.len(), 2);
        assert_eq!(
            landmark_0_factors
                .iter()
                .filter(|factor| matches!(factor.pose, FactorPose::Variable(_)))
                .count(),
            1
        );
        assert_eq!(
            landmark_0_factors
                .iter()
                .filter(|factor| matches!(factor.pose, FactorPose::Fixed(_)))
                .count(),
            1
        );

        let original_fixed_pose = map.keyframe(keyframe_1).expect("fixed keyframe").pose();
        problem.variable_poses[1].pose = apply_se3_delta(
            problem.variable_poses[1].pose,
            [0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        problem
            .write_back(&mut map)
            .expect("transactional variable-only writeback");
        assert!(pose_close(
            map.keyframe(keyframe_1).expect("fixed keyframe").pose(),
            original_fixed_pose,
            0.0
        ));
    }

    #[test]
    fn fixed_observation_factor_contributes_to_full_ba_cost() {
        let (mut map, intrinsics, keyframe_0, keyframe_1, _, points_world_m) =
            build_full_ba_fixture([0.0; 6]);
        let keyframe_2 = add_fixed_observation_keyframe(
            &mut map,
            intrinsics,
            keyframe_0,
            &points_world_m,
            axis_angle_pose([0.32, -0.01, 0.02], [0.0, 0.015, -0.01]),
            None,
        );
        let config = LocalBaConfig::new(5, 5, 4, 2.0, lm(1e-3)).expect("BA config");
        let ba = make_bundle_adjuster(intrinsics, config);
        let mut problem = FullBaProblem::try_from_map(
            &map,
            &[keyframe_0, keyframe_2],
            ba.window_size(),
            ba.min_observations(),
        )
        .expect("full BA problem with fixed observations");
        assert_eq!(problem.fixed_poses.len(), 1);
        assert!(pose_close(
            problem.fixed_poses[0].pose,
            map.keyframe(keyframe_1).expect("fixed keyframe").pose(),
            0.0
        ));
        assert!(
            problem
                .factors
                .iter()
                .any(|factor| matches!(factor.pose, FactorPose::Fixed(_)))
        );

        let before = full_problem_cost(&problem, intrinsics, 2.0)
            .expect("initial cost")
            .get();
        let fixed_factor = problem
            .factors
            .iter_mut()
            .find(|factor| matches!(factor.pose, FactorPose::Fixed(_)))
            .expect("fixed factor");
        fixed_factor.pixel.x += 1_000.0;
        let after = full_problem_cost(&problem, intrinsics, 2.0)
            .expect("perturbed cost")
            .get();
        assert!(after > before, "fixed factor must affect cost");
    }

    #[test]
    fn full_ba_solves_with_fixed_factors_without_moving_fixed_keyframes() {
        let (mut map, intrinsics, keyframe_0, keyframe_1, _, points_world_m) =
            build_full_ba_fixture([0.0; 6]);
        let true_pose_2 = axis_angle_pose([0.34, 0.01, -0.02], [0.0, -0.018, 0.012]);
        let keyframe_2 = add_fixed_observation_keyframe(
            &mut map,
            intrinsics,
            keyframe_0,
            &points_world_m,
            true_pose_2,
            None,
        );
        map.set_keyframe_pose(
            keyframe_2,
            apply_se3_delta(true_pose_2, [0.06, -0.02, 0.03, 0.012, -0.008, 0.006]),
        )
        .expect("perturb variable keyframe pose");
        let fixed_pose_before = map.keyframe(keyframe_1).expect("fixed keyframe").pose();
        let config = LocalBaConfig::new(5, 15, 4, 2.0, lm(1e-3)).expect("BA config");
        let mut ba = make_bundle_adjuster(intrinsics, config);
        let before_problem = FullBaProblem::try_from_map(
            &map,
            &[keyframe_0, keyframe_2],
            ba.window_size(),
            ba.min_observations(),
        )
        .expect("full BA problem");
        let before_cost = full_problem_cost(&before_problem, intrinsics, 2.0)
            .expect("initial cost")
            .get();

        let result = ba
            .optimize_keyframe_window(&mut map, &[keyframe_0, keyframe_2])
            .expect("full BA with fixed factors");

        assert!(result.is_applicable(), "unexpected BA outcome: {result:?}");
        let after_problem = FullBaProblem::try_from_map(
            &map,
            &[keyframe_0, keyframe_2],
            ba.window_size(),
            ba.min_observations(),
        )
        .expect("updated full BA problem");
        let after_cost = full_problem_cost(&after_problem, intrinsics, 2.0)
            .expect("updated cost")
            .get();
        assert!(
            after_cost < before_cost,
            "fixed-factor solve did not improve cost: before={before_cost}, after={after_cost}"
        );
        assert!(pose_close(
            map.keyframe(keyframe_1).expect("fixed keyframe").pose(),
            fixed_pose_before,
            0.0
        ));
    }

    #[test]
    fn monocular_cost_has_scale_gauge_but_exact_anchor_preserves_metric_scale() {
        let (map, intrinsics, keyframe_0, keyframe_1, _, _) =
            build_full_ba_fixture([0.08, -0.03, 0.04, 0.015, -0.01, 0.008]);
        let config = LocalBaConfig::new(5, 5, 4, 2.0, lm(1e-3)).expect("BA config");
        let ba = make_bundle_adjuster(intrinsics, config);
        let mut problem = FullBaProblem::try_from_map(
            &map,
            &[keyframe_0, keyframe_1],
            ba.window_size(),
            ba.min_observations(),
        )
        .expect("full BA problem");
        assert!(problem.fixed_poses.is_empty());
        let anchor = problem.metric_scale_anchor;
        let anchored_coordinate_before = anchor
            .axis
            .coordinate_m(problem.landmarks[anchor.landmark.as_usize()].position);
        let cost_before = full_problem_cost(&problem, intrinsics, 2.0)
            .expect("initial cost")
            .get();

        scale_problem_about_fixed_camera(&mut problem, 1.75);

        let cost_after = full_problem_cost(&problem, intrinsics, 2.0)
            .expect("scaled cost")
            .get();
        let anchored_coordinate_after = anchor
            .axis
            .coordinate_m(problem.landmarks[anchor.landmark.as_usize()].position);
        assert!(
            (cost_after - cost_before).abs() <= 1e-3 * cost_before.abs().max(1.0),
            "monocular reprojection cost changed under a similarity scale: before={cost_before}, after={cost_after}"
        );
        assert_ne!(anchored_coordinate_after, anchored_coordinate_before);
    }

    #[test]
    fn metric_scale_anchor_rejects_near_zero_camera_displacement() {
        let landmarks = [LandmarkVariable {
            point_id: MapPointId::default(),
            position: Point3 {
                x: 0.25 * MIN_SCALE_ANCHOR_DISPLACEMENT_M,
                y: -0.5 * MIN_SCALE_ANCHOR_DISPLACEMENT_M,
                z: 0.75 * MIN_SCALE_ANCHOR_DISPLACEMENT_M,
            },
        }];

        let error = MetricScaleAnchor::select(Pose::identity(), &landmarks)
            .expect_err("near-zero displacement cannot condition a scale anchor");
        assert!(error.max_camera_displacement_m < error.min_required_m);
    }

    #[test]
    fn full_ba_connectivity_is_rooted_at_the_exactly_fixed_pose() {
        let factors = [
            ReprojectionFactor {
                pose: FactorPose::Variable(PoseVarIndex(0)),
                landmark: LandmarkVarIndex(0),
                pixel: Keypoint { x: 0.0, y: 0.0 },
            },
            ReprojectionFactor {
                pose: FactorPose::Variable(PoseVarIndex(1)),
                landmark: LandmarkVarIndex(0),
                pixel: Keypoint { x: 0.0, y: 0.0 },
            },
            ReprojectionFactor {
                pose: FactorPose::Variable(PoseVarIndex(2)),
                landmark: LandmarkVarIndex(1),
                pixel: Keypoint { x: 0.0, y: 0.0 },
            },
            ReprojectionFactor {
                pose: FactorPose::Fixed(FixedPoseIndex(0)),
                landmark: LandmarkVarIndex(1),
                pixel: Keypoint { x: 0.0, y: 0.0 },
            },
        ];

        assert_eq!(
            disconnected_variable_pose_count(3, 2, &factors),
            NonZeroUsize::new(1)
        );
        let reason = classify_full_ba_build_error(FullBaBuildError::DisconnectedFromFixedPose {
            disconnected_pose_count: NonZeroUsize::MIN,
        })
        .expect("disconnected graph is a mathematical degeneracy");
        assert_eq!(
            reason,
            DegenerateReason::DisconnectedFromFixedPose {
                disconnected_pose_count: NonZeroUsize::MIN
            }
        );
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

        let config = LocalBaConfig::new(5, 15, 4, 2.0, lm(1e-3)).expect("valid BA config");
        let mut ba = make_bundle_adjuster(intrinsics, config);
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
        let config = LocalBaConfig::new(5, 15, 4, 2.0, lm(1e-3)).expect("valid BA config");
        let mut ba = make_bundle_adjuster(intrinsics, config);

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
                objective_breakdown: VioObjectiveBreakdown::default(),
            };
            let error = validate_vio_linearization_values(
                &linearization,
                1,
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
    fn vio_objective_breakdown_rejects_each_invalid_component_and_total_overflow() {
        let construct = |values: [f64; 5]| {
            VioObjectiveBreakdown::new(values[0], values[1], values[2], values[3], values[4])
        };
        let components = [
            VioObjectiveComponent::ReprojectionRobustPx2,
            VioObjectiveComponent::ImuMahalanobis,
            VioObjectiveComponent::BiasRandomWalkMahalanobis,
            VioObjectiveComponent::VelocityAnchorMahalanobis,
            VioObjectiveComponent::BiasPriorMahalanobis,
        ];

        for (index, component) in components.into_iter().enumerate() {
            for value in [f64::NAN, f64::INFINITY] {
                let mut values = [0.0; 5];
                values[index] = value;
                assert!(matches!(
                    construct(values),
                    Err(VioObjectiveError::NonFinite {
                        component: actual,
                        ..
                    }) if actual == component
                ));
            }

            let mut values = [0.0; 5];
            values[index] = -f64::MIN_POSITIVE;
            assert!(matches!(
                construct(values),
                Err(VioObjectiveError::Negative {
                    component: actual,
                    ..
                }) if actual == component
            ));
        }

        assert!(matches!(
            construct([f64::MAX, f64::MAX, 0.0, 0.0, 0.0]),
            Err(VioObjectiveError::NonFinite {
                component: VioObjectiveComponent::MixedTotal,
                value: f64::INFINITY,
            })
        ));

        let objective = construct([1.0, 2.0, 3.0, 4.0, 5.0]).expect("valid objective");
        assert_eq!(objective.reprojection_robust_px2(), 1.0);
        assert_eq!(objective.imu_mahalanobis(), 2.0);
        assert_eq!(objective.bias_random_walk_mahalanobis(), 3.0);
        assert_eq!(objective.velocity_anchor_mahalanobis(), 4.0);
        assert_eq!(objective.bias_prior_mahalanobis(), 5.0);
        assert_eq!(objective.total_mixed_objective(), 15.0);
    }

    #[cfg(feature = "vio")]
    #[test]
    fn vio_linearization_reports_invalid_objective_with_source_context() {
        let linearization = VioLinearization {
            hessian: vec![0.0],
            rhs: vec![0.0],
            objective_breakdown: VioObjectiveBreakdown {
                reprojection_robust_px2: 0.0,
                imu_mahalanobis: f64::NAN,
                bias_random_walk_mahalanobis: 0.0,
                velocity_anchor_mahalanobis: 0.0,
                bias_prior_mahalanobis: 0.0,
            },
        };
        let error = validate_vio_linearization_values(
            &linearization,
            1,
            VioEvaluation {
                stage: VioEvaluationStage::Candidate,
                iteration: 7,
            },
        )
        .expect_err("invalid objective must fail");
        assert!(matches!(
            &error,
            VioSolveError::InvalidObjective {
                stage: VioEvaluationStage::Candidate,
                iteration: 7,
                source: VioObjectiveError::NonFinite {
                    component: VioObjectiveComponent::ImuMahalanobis,
                    ..
                },
            }
        ));
        assert!(
            std::error::Error::source(&error)
                .and_then(|source| source.downcast_ref::<VioObjectiveError>())
                .is_some()
        );
    }

    #[cfg(feature = "vio")]
    #[test]
    fn vio_solve_result_derives_counts_and_rejects_incompatible_outcomes() {
        let factors = VioFactorDiagnostics {
            last_frame_active_visual_factor_count: 11,
            initially_excluded_nonprojectable_visual_factor_count: 2,
            regularized_imu_residual_factor_count: 4,
            floored_accel_bias_random_walk_factor_count: 3,
            floored_gyro_bias_random_walk_factor_count: 1,
        };
        let objective =
            VioObjectiveBreakdown::new(1.0, 2.0, 3.0, 4.0, 5.0).expect("valid objective");
        let result = VioSolveResult::try_evaluated(
            VioSolveTermination::IterationLimit,
            2,
            3,
            1,
            objective,
            factors,
        )
        .expect("consistent result");
        assert_eq!(result.termination(), VioSolveTermination::IterationLimit);
        assert_eq!(result.attempted_iterations(), 5);
        assert_eq!(result.accepted_steps(), 2);
        assert_eq!(result.rejected_steps(), 3);
        assert_eq!(result.rejected_nonprojectable_candidate_steps(), 1);
        assert_eq!(result.final_mixed_objective(), 15.0);
        assert_eq!(result.objective_breakdown(), objective);
        assert_eq!(result.last_frame_active_visual_factor_count(), 11);
        assert_eq!(
            result.initially_excluded_nonprojectable_visual_factor_count(),
            2
        );
        assert_eq!(result.regularized_imu_residual_factor_count(), 4);
        assert_eq!(result.floored_accel_bias_random_walk_factor_count(), 3);
        assert_eq!(result.floored_gyro_bias_random_walk_factor_count(), 1);
        assert!(result.has_improved_estimate());

        let not_required = VioSolveResult::not_required();
        assert_eq!(not_required.termination(), VioSolveTermination::NotRequired);
        assert_eq!(not_required.attempted_iterations(), 0);
        assert_eq!(not_required.final_mixed_objective(), 0.0);
        assert!(!not_required.has_improved_estimate());

        for (termination, accepted, rejected) in [
            (VioSolveTermination::NotRequired, 1, 0),
            (
                VioSolveTermination::Converged {
                    criterion: VioConvergenceCriterion::ComponentwiseStepAndRelativeObjective,
                },
                0,
                1,
            ),
            (VioSolveTermination::IterationLimit, 0, 1),
            (VioSolveTermination::StalledNoObjectiveImprovement, 1, 0),
            (VioSolveTermination::StalledNoObjectiveImprovement, 0, 0),
        ] {
            assert!(matches!(
                VioSolveResult::try_evaluated(
                    termination,
                    accepted,
                    rejected,
                    0,
                    objective,
                    factors,
                ),
                Err(VioSolveOutcomeError::TerminationIncompatibleWithSteps {
                    termination: actual,
                    accepted_steps: actual_accepted,
                    rejected_steps: actual_rejected,
                }) if actual == termination
                    && actual_accepted == accepted
                    && actual_rejected == rejected
            ));
        }

        assert!(matches!(
            VioSolveResult::try_evaluated(
                VioSolveTermination::IterationLimit,
                usize::MAX,
                1,
                0,
                objective,
                factors,
            ),
            Err(VioSolveOutcomeError::StepCountOverflow { .. })
        ));
        assert!(matches!(
            VioSolveResult::try_evaluated(
                VioSolveTermination::IterationLimit,
                1,
                1,
                2,
                objective,
                factors,
            ),
            Err(
                VioSolveOutcomeError::NonprojectableRejectionsExceedRejections {
                    nonprojectable_rejections: 2,
                    rejected_steps: 1,
                }
            )
        ));

        let root = VioSolveOutcomeError::TerminationIncompatibleWithSteps {
            termination: VioSolveTermination::NotRequired,
            accepted_steps: 1,
            rejected_steps: 0,
        };
        let wrapped = VioSolveError::InvalidOutcome { source: root };
        assert!(
            std::error::Error::source(&wrapped)
                .and_then(|source| source.downcast_ref::<VioSolveOutcomeError>())
                .is_some()
        );
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
    fn vio_window_capacity_rejects_invalid_dense_workspace_shapes() {
        for actual in [0, 1] {
            assert!(matches!(
                VioWindowCapacity::new(actual),
                Err(VioWindowCapacityError::TooFewFrames {
                    minimum: 2,
                    actual: error_actual,
                }) if error_actual == actual
            ));
        }
        assert!(matches!(
            VioWindowCapacity::new(usize::MAX),
            Err(VioWindowCapacityError::StateDimensionOverflow { .. })
        ));

        let element_overflow_frames = usize::MAX / VIO_STATE_DIM;
        assert!(matches!(
            VioWindowCapacity::new(element_overflow_frames),
            Err(VioWindowCapacityError::DenseMatrixElementCountOverflow { .. })
        ));

        let byte_overflow_dimension = 1_usize << (usize::BITS / 2 - 2);
        let byte_overflow_frames = byte_overflow_dimension.div_ceil(VIO_STATE_DIM);
        assert!(matches!(
            VioWindowCapacity::new(byte_overflow_frames),
            Err(VioWindowCapacityError::DenseMatrixByteLengthUnaddressable { .. })
        ));

        let capacity = vio_window_capacity();
        assert_eq!(capacity.frames().get(), 5);
        assert_eq!(capacity.workspace.dimension, 5 * VIO_STATE_DIM);
        assert_eq!(
            capacity.workspace.matrix_elements,
            25 * VIO_STATE_DIM * VIO_STATE_DIM
        );
    }

    #[cfg(feature = "vio")]
    #[test]
    fn vio_optimizer_allocation_error_preserves_allocator_source() {
        let error = try_vio_buffer::<u8>("test", usize::MAX)
            .expect_err("an unaddressable vector capacity must fail");

        assert!(matches!(
            error,
            VioOptimizerWorkspaceError::Allocation {
                requested_elements: usize::MAX,
                ..
            }
        ));
        assert!(std::error::Error::source(&error).is_some());
    }

    #[cfg(feature = "vio")]
    #[test]
    fn vio_observation_allocation_error_preserves_full_source_chain() {
        let mut allocation_probe = Vec::<u8>::new();
        let source = allocation_probe
            .try_reserve_exact(usize::MAX)
            .expect_err("an unaddressable vector capacity must fail");
        let error = VioSolveError::from(ObservationResolveError::Allocation {
            requested_observations: usize::MAX,
            source,
        });

        let resolution = std::error::Error::source(&error).expect("observation resolution source");
        let allocator = resolution.source().expect("allocator source");
        assert!(!allocator.to_string().is_empty());
        assert!(allocator.source().is_none());
    }

    #[cfg(feature = "vio")]
    #[test]
    fn vio_optimizer_rejects_windows_beyond_preallocated_capacity() {
        let state = crate::NavState::try_new(
            crate::Pose64::identity(),
            [0.0; 3],
            crate::ImuBias::default(),
        )
        .expect("state");
        let preintegrated = test_preintegrated_imu();
        let mut window = VioWindow {
            anchor: VioAnchor {
                state: state.clone(),
                observations: None,
                anchor_velocity_odom_mps: [0.0; 3],
            },
            successors: vec![
                VioSuccessor {
                    state: state.clone(),
                    observations: None,
                    preintegrated: preintegrated.clone(),
                },
                VioSuccessor {
                    state,
                    observations: None,
                    preintegrated,
                },
            ],
        };
        let mut optimizer =
            VioOptimizer::try_new(test_vio_solve_config(2)).expect("optimizer workspace");

        let error = optimizer
            .optimize(
                &mut window,
                &SlamMap::new(),
                &crate::MapFromOdom::identity(),
            )
            .expect_err("three frames must not enter a two-frame workspace");
        assert!(matches!(
            error,
            VioSolveError::WindowExceedsConfiguredCapacity {
                actual_frames: 3,
                capacity_frames: 2,
            }
        ));
    }

    #[cfg(feature = "vio")]
    #[test]
    fn vio_optimizer_reuses_dense_and_state_buffers_across_solves() {
        fn sorted_pair(first: usize, second: usize) -> [usize; 2] {
            if first <= second {
                [first, second]
            } else {
                [second, first]
            }
        }

        fn allocation_addresses(
            optimizer: &VioOptimizer,
        ) -> ([usize; 2], [usize; 2], [usize; 2], usize, usize, usize) {
            let workspace = &optimizer.workspace;
            (
                sorted_pair(
                    workspace.current_linearization.hessian.as_ptr() as usize,
                    workspace.scratch_linearization.hessian.as_ptr() as usize,
                ),
                sorted_pair(
                    workspace.current_linearization.rhs.as_ptr() as usize,
                    workspace.scratch_linearization.rhs.as_ptr() as usize,
                ),
                sorted_pair(
                    workspace.states.as_ptr() as usize,
                    workspace.candidate_states.as_ptr() as usize,
                ),
                workspace.linear_solver_row_scales.as_ptr() as usize,
                workspace.resolved_observations.as_ptr() as usize,
                workspace
                    .visual_support
                    .observation_indices_by_frame
                    .as_ptr() as usize,
            )
        }

        let state = crate::NavState::try_new(
            crate::Pose64::identity(),
            [0.0; 3],
            crate::ImuBias::default(),
        )
        .expect("state");
        let mut window = VioWindow {
            anchor: VioAnchor {
                state: state.clone(),
                observations: None,
                anchor_velocity_odom_mps: [0.0; 3],
            },
            successors: vec![VioSuccessor {
                state,
                observations: None,
                preintegrated: test_preintegrated_imu(),
            }],
        };
        let mut optimizer =
            VioOptimizer::try_new(test_vio_solve_config(5)).expect("optimizer workspace");
        let before = allocation_addresses(&optimizer);

        for _ in 0..2 {
            optimizer
                .optimize(
                    &mut window,
                    &SlamMap::new(),
                    &crate::MapFromOdom::identity(),
                )
                .expect("bounded VIO solve");
            assert_eq!(allocation_addresses(&optimizer), before);
        }
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
                    vio_window_capacity(),
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
                vio_window_capacity(),
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
                    vio_window_capacity(),
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
                    vio_window_capacity(),
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
            vio_window_capacity(),
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
        let resolved = vec![VioResolvedFrameObservations::default()];
        let map_from_odom = crate::MapFromOdom::identity();
        let mut support = VisualFactorSupport::try_new(1).expect("visual support workspace");
        support
            .update_from_initial_states(&states, &resolved, &config, &map_from_odom)
            .expect("visual support");
        let window = VioWindow {
            anchor: VioAnchor {
                state,
                observations: None,
                anchor_velocity_odom_mps: [0.0; 3],
            },
            successors: Vec::new(),
        };

        let mut linearization =
            VioLinearization::try_new(vio_window_capacity().workspace).expect("linearization");
        linearize_vio_states(
            &window,
            &states,
            ResolvedVioVisualFactors {
                observations: &resolved,
                support: &support,
            },
            &config,
            &map_from_odom,
            VioEvaluation {
                stage: VioEvaluationStage::Initial,
                iteration: 0,
            },
            &mut linearization,
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
        assert_eq!(
            linearization.objective_breakdown.bias_prior_mahalanobis(),
            7.0
        );
    }

    #[cfg(feature = "vio")]
    #[test]
    fn vio_visual_factor_support_is_fixed_across_candidate_evaluations() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let config = VioSolveConfig::new(
            vio_window_capacity(),
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
            VioResolvedFrameObservations::default(),
            VioResolvedFrameObservations {
                available: true,
                observations: vec![front, behind],
            },
        ];
        let map_from_odom = crate::MapFromOdom::identity();
        let mut support = VisualFactorSupport::try_new(2).expect("visual support workspace");
        support
            .update_from_initial_states(&initial_states, &resolved, &config, &map_from_odom)
            .expect("visual support");
        assert_eq!(support.indices_for_frame(1), &[0]);
        assert_eq!(support.last_frame_factor_count(), 1);
        assert_eq!(support.initially_excluded_nonprojectable_factor_count, 1);
        let active_index_buffer = support.observation_indices_by_frame[1].as_ptr();
        support
            .update_from_initial_states(&initial_states, &resolved, &config, &map_from_odom)
            .expect("repeat visual support update");
        assert_eq!(
            support.observation_indices_by_frame[1].as_ptr(),
            active_index_buffer
        );

        let window = VioWindow {
            anchor: VioAnchor {
                state: initial_states[0].clone(),
                observations: None,
                anchor_velocity_odom_mps: [0.0; 3],
            },
            successors: vec![VioSuccessor {
                state: initial_states[1].clone(),
                observations: None,
                preintegrated: test_preintegrated_imu(),
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

        let mut candidate_linearization =
            VioLinearization::try_new(vio_window_capacity().workspace).expect("linearization");
        let error = match linearize_vio_states(
            &window,
            &candidate_states,
            ResolvedVioVisualFactors {
                observations: &resolved,
                support: &support,
            },
            &config,
            &map_from_odom,
            VioEvaluation {
                stage: VioEvaluationStage::Candidate,
                iteration: 1,
            },
            &mut candidate_linearization,
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
