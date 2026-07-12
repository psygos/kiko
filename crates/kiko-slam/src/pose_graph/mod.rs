/// Step size for numerical Jacobian computation via central differences.
const NUMERICAL_DIFF_EPS: f64 = 1e-6;
/// Anchor regularization weight to remove gauge freedom in pose graph optimization.
const ANCHOR_REGULARIZATION: f64 = 1e9;
/// Maximum SE3 step norm; larger steps are clamped for stability.
const MAX_STEP_NORM: f64 = 1.0;
/// Step convergence threshold for the pose graph optimizer.
const POSE_GRAPH_CONVERGENCE: f64 = 1e-6;
/// Near-zero threshold in Huber weight to avoid division by zero.
const HUBER_NEAR_ZERO: f64 = 1e-12;

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
        from: usize,
        pose_count: usize,
    },
    EdgeToOutOfBounds {
        to: usize,
        pose_count: usize,
    },
    InvalidEdgeSet {
        invalid_edges: usize,
        pose_count: usize,
    },
    NonFiniteResidual {
        iteration: usize,
    },
    NonFiniteStep {
        iteration: usize,
        pose_index: usize,
    },
    PcgDidNotConverge {
        outer_iteration: usize,
        iterations: usize,
        initial_residual_norm: f64,
        residual_norm: f64,
        target_residual_norm: f64,
        stop_reason: PcgStopReason,
    },
    NotConverged {
        iterations: usize,
        residual_norm: f64,
    },
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
            PoseGraphError::EdgeFromOutOfBounds { from, pose_count } => {
                write!(
                    f,
                    "pose graph edge.from out of bounds: from={from}, pose_count={pose_count}"
                )
            }
            PoseGraphError::EdgeToOutOfBounds { to, pose_count } => {
                write!(
                    f,
                    "pose graph edge.to out of bounds: to={to}, pose_count={pose_count}"
                )
            }
            PoseGraphError::InvalidEdgeSet {
                invalid_edges,
                pose_count,
            } => write!(
                f,
                "pose graph contains {invalid_edges} invalid edges for pose_count={pose_count}"
            ),
            PoseGraphError::NonFiniteResidual { iteration } => {
                write!(
                    f,
                    "pose graph residual became non-finite at iteration {iteration}"
                )
            }
            PoseGraphError::NonFiniteStep {
                iteration,
                pose_index,
            } => write!(
                f,
                "pose graph step became non-finite at iteration {iteration} for pose {pose_index}"
            ),
            PoseGraphError::PcgDidNotConverge {
                outer_iteration,
                iterations,
                initial_residual_norm,
                residual_norm,
                target_residual_norm,
                stop_reason,
            } => write!(
                f,
                "pose graph PCG stopped without convergence at outer iteration {outer_iteration} after {iterations} inner iterations ({stop_reason:?}, residual_norm={residual_norm:.3e}, initial_residual_norm={initial_residual_norm:.3e}, target_residual_norm={target_residual_norm:.3e})"
            ),
            PoseGraphError::NotConverged {
                iterations,
                residual_norm,
            } => write!(
                f,
                "pose graph did not converge after {iterations} iterations (residual_norm={residual_norm:.3e})"
            ),
        }
    }
}

impl std::error::Error for PoseGraphError {}

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
    EssentialEdge, EssentialEdgeKind, EssentialGraph, EssentialGraphError, EssentialGraphSnapshot,
    PoseGraphInput,
};
pub use optimizer::{
    PoseGraphConfig, PoseGraphEdge, PoseGraphOptimizer, PoseGraphResult, PoseGraphTermination,
    compute_edge_error, compute_edge_jacobians,
};
pub use solver::{PcgResult, solve_pcg};
pub use sparse::BlockCsr6x6;

#[cfg(test)]
mod tests;
