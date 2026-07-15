/// Translation step in metres for numerical Jacobian central differences.
const NUMERICAL_DIFF_TRANSLATION_STEP_M: f64 = 1e-6;
/// Rotation step in radians for numerical Jacobian central differences.
const NUMERICAL_DIFF_ROTATION_STEP_RAD: f64 = 1e-6;
/// Maximum translation step norm in metres; larger steps are clamped.
const MAX_TRANSLATION_STEP_M: f64 = 1.0;
/// Maximum rotation step norm in radians; larger steps are clamped.
const MAX_ROTATION_STEP_RAD: f64 = 1.0;
/// Per-pose translation-step norm convergence threshold in metres.
const TRANSLATION_STEP_CONVERGENCE_M: f64 = 1e-6;
/// Per-pose rotation-step norm convergence threshold in radians.
const ROTATION_STEP_CONVERGENCE_RAD: f64 = 1e-6;
/// Near-zero normalized residual threshold in the Huber kernel.
const HUBER_NEAR_ZERO_NORMALIZED_RESIDUAL: f64 = 1e-12;

use crate::map::KeyframeId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcgStopReason {
    Converged,
    NearZeroDenominator,
    NonPositiveCurvature,
    NearZeroPreconditionedResidual,
    IterationLimit,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PoseGraphError {
    CsrIndexOutOfBounds {
        row: usize,
        col: usize,
        nrows: usize,
    },
    SpmvInputLength {
        expected: usize,
        actual: usize,
    },
    SpmvOutputLength {
        expected: usize,
        actual: usize,
    },
    PcgRhsLength {
        expected: usize,
        actual: usize,
    },
    PcgSolutionLength {
        expected: usize,
        actual: usize,
    },
    InvalidPcgTolerance {
        value: f64,
    },
    NonFinitePcgInput {
        input: &'static str,
        index: usize,
        value: f64,
    },
    NonFinitePcgScalar {
        scalar: &'static str,
        iteration: usize,
        value: f64,
    },
    InvalidPcgDiagonalBlock {
        block_index: usize,
    },
    NonFiniteCsrBlockValue {
        row: usize,
        col: usize,
        block_row: usize,
        block_col: usize,
        value: f64,
    },
    AsymmetricPcgMatrix {
        row: usize,
        col: usize,
        block_row: usize,
        block_col: usize,
        forward: f64,
        transpose: f64,
    },
    EdgeFromOutOfBounds {
        edge_index: usize,
        from: usize,
        pose_count: usize,
    },
    EdgeToOutOfBounds {
        edge_index: usize,
        to: usize,
        pose_count: usize,
    },
    EdgeConstruction {
        edge_index: usize,
        source: PoseGraphEdgeError,
    },
    UnregisteredEssentialEdgeEndpoint {
        edge_index: usize,
        endpoint: &'static str,
        keyframe_id: KeyframeId,
    },
    UnconstrainedPoseGraph {
        pose_count: usize,
    },
    DisconnectedPoseGraph {
        pose_count: usize,
        component_count: usize,
        anchor_component_size: usize,
    },
    InvalidRobustSquaredNorm {
        outer_iteration: usize,
        edge_index: usize,
        value: f64,
    },
    NonFiniteLinearSolveResidual {
        outer_iteration: usize,
        value: f64,
    },
    NonFiniteStep {
        outer_iteration: usize,
        pose_index: usize,
    },
    PcgDidNotConverge {
        outer_iteration: usize,
        pcg_iterations: usize,
        initial_residual_norm: f64,
        residual_norm: f64,
        target_residual_norm: f64,
        stop_reason: PcgStopReason,
    },
    NotConverged {
        outer_iterations: usize,
        last_linear_solve_residual_norm: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PoseGraphInformationError {
    NonFiniteEntry {
        row: usize,
        col: usize,
        value: f64,
    },
    NonFiniteNormalizedEntry {
        row: usize,
        col: usize,
        value: f64,
    },
    NonPositiveDiagonal {
        axis: usize,
        value: f64,
    },
    Asymmetric {
        row: usize,
        col: usize,
        upper: f64,
        lower: f64,
        relative_tolerance: f64,
    },
    NotPositiveDefinite {
        pivot: usize,
        normalized_schur_complement: f64,
        tolerance: f64,
    },
}

impl std::fmt::Display for PoseGraphInformationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteEntry { row, col, value } => write!(
                f,
                "pose graph information entry ({row}, {col}) must be finite, got {value}"
            ),
            Self::NonFiniteNormalizedEntry { row, col, value } => write!(
                f,
                "dimensionless pose graph information entry ({row}, {col}) became non-finite during diagonal normalization: {value}"
            ),
            Self::NonPositiveDiagonal { axis, value } => write!(
                f,
                "pose graph information diagonal {axis} must be finite and > 0, got {value}"
            ),
            Self::Asymmetric {
                row,
                col,
                upper,
                lower,
                relative_tolerance,
            } => write!(
                f,
                "pose graph information is asymmetric at ({row}, {col}): upper {upper}, lower {lower}, relative tolerance {relative_tolerance}"
            ),
            Self::NotPositiveDefinite {
                pivot,
                normalized_schur_complement,
                tolerance,
            } => write!(
                f,
                "dimensionless pose graph information is not numerically positive definite at pivot {pivot}: normalized Schur complement {normalized_schur_complement} must be finite and > {tolerance}"
            ),
        }
    }
}

impl std::error::Error for PoseGraphInformationError {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PoseGraphEdgeError {
    SelfEdge { pose_index: usize },
    Measurement { source: crate::Pose64Error },
    Information { source: PoseGraphInformationError },
}

impl std::fmt::Display for PoseGraphEdgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SelfEdge { pose_index } => {
                write!(
                    f,
                    "pose graph edge endpoints must differ, both were {pose_index}"
                )
            }
            Self::Measurement { source } => {
                write!(f, "invalid pose graph edge measurement: {source}")
            }
            Self::Information { source } => {
                write!(f, "invalid pose graph edge information: {source}")
            }
        }
    }
}

impl std::error::Error for PoseGraphEdgeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Measurement { source } => Some(source),
            Self::Information { source } => Some(source),
            Self::SelfEdge { .. } => None,
        }
    }
}

impl std::fmt::Display for PoseGraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoseGraphError::CsrIndexOutOfBounds { row, col, nrows } => {
                write!(
                    f,
                    "csr index out of bounds: row={row}, col={col}, nrows={nrows}"
                )
            }
            PoseGraphError::SpmvInputLength { expected, actual } => {
                write!(
                    f,
                    "spmv input length mismatch: expected {expected}, got {actual}"
                )
            }
            PoseGraphError::SpmvOutputLength { expected, actual } => {
                write!(
                    f,
                    "spmv output length mismatch: expected {expected}, got {actual}"
                )
            }
            PoseGraphError::PcgRhsLength { expected, actual } => {
                write!(
                    f,
                    "pcg rhs length mismatch: expected {expected}, got {actual}"
                )
            }
            PoseGraphError::PcgSolutionLength { expected, actual } => {
                write!(
                    f,
                    "pcg solution length mismatch: expected {expected}, got {actual}"
                )
            }
            PoseGraphError::InvalidPcgTolerance { value } => {
                write!(f, "pcg tolerance must be finite and in (0, 1], got {value}")
            }
            PoseGraphError::NonFinitePcgInput {
                input,
                index,
                value,
            } => write!(f, "pcg {input}[{index}] must be finite, got {value}"),
            PoseGraphError::NonFinitePcgScalar {
                scalar,
                iteration,
                value,
            } => write!(
                f,
                "pcg {scalar} became non-finite at iteration {iteration}: {value}"
            ),
            PoseGraphError::InvalidPcgDiagonalBlock { block_index } => write!(
                f,
                "pcg diagonal block {block_index} is not finite symmetric positive definite"
            ),
            PoseGraphError::NonFiniteCsrBlockValue {
                row,
                col,
                block_row,
                block_col,
                value,
            } => write!(
                f,
                "csr block ({row}, {col}) contains non-finite value at ({block_row}, {block_col}): {value}"
            ),
            PoseGraphError::AsymmetricPcgMatrix {
                row,
                col,
                block_row,
                block_col,
                forward,
                transpose,
            } => write!(
                f,
                "pcg matrix is asymmetric at block ({row}, {col}) element ({block_row}, {block_col}): forward={forward}, transpose={transpose}"
            ),
            PoseGraphError::EdgeFromOutOfBounds {
                edge_index,
                from,
                pose_count,
            } => {
                write!(
                    f,
                    "pose graph edge {edge_index} from-index is out of bounds: from={from}, pose_count={pose_count}"
                )
            }
            PoseGraphError::EdgeToOutOfBounds {
                edge_index,
                to,
                pose_count,
            } => {
                write!(
                    f,
                    "pose graph edge {edge_index} to-index is out of bounds: to={to}, pose_count={pose_count}"
                )
            }
            PoseGraphError::EdgeConstruction { edge_index, source } => {
                write!(f, "pose graph edge {edge_index} is invalid: {source}")
            }
            PoseGraphError::UnregisteredEssentialEdgeEndpoint {
                edge_index,
                endpoint,
                keyframe_id,
            } => write!(
                f,
                "essential graph edge {edge_index} endpoint {endpoint} references unregistered keyframe {keyframe_id:?}"
            ),
            PoseGraphError::UnconstrainedPoseGraph { pose_count } => write!(
                f,
                "pose graph has {pose_count} poses but no relative-pose constraints"
            ),
            PoseGraphError::DisconnectedPoseGraph {
                pose_count,
                component_count,
                anchor_component_size,
            } => write!(
                f,
                "pose graph is disconnected: {pose_count} poses in {component_count} components; {anchor_component_size} poses are connected to anchor pose 0"
            ),
            PoseGraphError::InvalidRobustSquaredNorm {
                outer_iteration,
                edge_index,
                value,
            } => write!(
                f,
                "pose graph edge {edge_index} Mahalanobis squared residual must be finite and >= 0 at outer iteration {outer_iteration}, got {value}"
            ),
            PoseGraphError::NonFiniteLinearSolveResidual {
                outer_iteration,
                value,
            } => write!(
                f,
                "pose graph PCG residual norm became non-finite at outer iteration {outer_iteration}: {value}"
            ),
            PoseGraphError::NonFiniteStep {
                outer_iteration,
                pose_index,
            } => write!(
                f,
                "pose graph step became non-finite at outer iteration {outer_iteration} for pose {pose_index}"
            ),
            PoseGraphError::PcgDidNotConverge {
                outer_iteration,
                pcg_iterations,
                initial_residual_norm,
                residual_norm,
                target_residual_norm,
                stop_reason,
            } => write!(
                f,
                "pose graph PCG stopped without convergence at outer iteration {outer_iteration} after {pcg_iterations} inner iterations ({stop_reason:?}, residual_norm={residual_norm:.3e}, initial_residual_norm={initial_residual_norm:.3e}, target_residual_norm={target_residual_norm:.3e})"
            ),
            PoseGraphError::NotConverged {
                outer_iterations,
                last_linear_solve_residual_norm,
            } => write!(
                f,
                "pose graph did not converge after {outer_iterations} outer iterations (last_linear_solve_residual_norm={last_linear_solve_residual_norm:.3e})"
            ),
        }
    }
}

impl std::error::Error for PoseGraphError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::EdgeConstruction { source, .. } => Some(source),
            _ => None,
        }
    }
}

pub(crate) fn scaled_identity6(scale: f64) -> [[f64; 6]; 6] {
    let mut out = [[0.0_f64; 6]; 6];
    for (i, row) in out.iter_mut().enumerate() {
        row[i] = scale;
    }
    out
}

mod essential;
mod optimizer;
mod solver;
mod sparse;

pub use essential::{
    EssentialEdge, EssentialEdgeError, EssentialEdgeKind, EssentialGraph, EssentialGraphError,
    EssentialGraphSnapshot, PoseGraphInput,
};
pub use optimizer::{
    PoseGraphConfig, PoseGraphConfigError, PoseGraphConvergenceCriterion, PoseGraphEdge,
    PoseGraphInformation, PoseGraphOptimizer, PoseGraphResult, PoseGraphTermination,
    compute_edge_error, compute_edge_jacobians,
};
pub use solver::{PcgResult, solve_pcg};
pub use sparse::BlockCsr6x6;

#[cfg(test)]
mod tests;
