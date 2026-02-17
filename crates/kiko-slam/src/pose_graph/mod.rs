/// Near-zero threshold for PCG solver denominators and Gauss-Jordan pivot magnitudes.
const NEAR_ZERO: f64 = 1e-18;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    EdgeFromOutOfBounds {
        from: usize,
        pose_count: usize,
    },
    EdgeToOutOfBounds {
        to: usize,
        pose_count: usize,
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
    PoseGraphConfig, PoseGraphEdge, PoseGraphOptimizer, PoseGraphResult, compute_edge_error,
    compute_edge_jacobians,
};
pub use solver::{PcgResult, solve_pcg};
pub use sparse::BlockCsr6x6;

#[cfg(test)]
mod tests;
