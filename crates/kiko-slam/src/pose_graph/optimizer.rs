use std::num::NonZeroUsize;

use crate::Pose64;
use crate::math::{se3_exp_f64, se3_log_f64};

use super::{
    BlockCsr6x6, HUBER_NEAR_ZERO_NORMALIZED_RESIDUAL, MAX_ROTATION_STEP_RAD,
    MAX_TRANSLATION_STEP_M, NUMERICAL_DIFF_ROTATION_STEP_RAD, NUMERICAL_DIFF_TRANSLATION_STEP_M,
    PcgStopReason, PoseGraphEdgeError, PoseGraphError, PoseGraphInformationError,
    ROTATION_STEP_CONVERGENCE_RAD, TRANSLATION_STEP_CONVERGENCE_M, solve_pcg,
};

type Jacobian6 = [[f64; 6]; 6];
type EdgeJacobians = (Jacobian6, Jacobian6);
const INFORMATION_RELATIVE_SYMMETRY_TOLERANCE: f64 = 128.0 * f64::EPSILON;
const INFORMATION_NORMALIZED_CHOLESKY_TOLERANCE: f64 = 1e-12;

/// A non-self pose constraint with validated finite, symmetric-positive-
/// definite information. Endpoint bounds are resolved once when optimization
/// starts because they depend on that solve's pose array.
#[derive(Clone, Debug)]
pub struct PoseGraphEdge {
    from: usize,
    to: usize,
    measurement: Pose64,
    information: PoseGraphInformation,
}

/// Canonical pose-constraint information matrix.
///
/// Construction rejects non-finite, materially asymmetric, non-positive, and
/// numerically non-positive-definite matrices. Within-tolerance asymmetry is
/// replaced by the pairwise mean and exposed by `was_symmetrized`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoseGraphInformation {
    matrix: [[f64; 6]; 6],
    was_symmetrized: bool,
}

impl PoseGraphInformation {
    pub fn try_new(matrix: [[f64; 6]; 6]) -> Result<Self, PoseGraphInformationError> {
        let (matrix, was_symmetrized) = parse_information(matrix)?;
        Ok(Self {
            matrix,
            was_symmetrized,
        })
    }

    pub fn matrix(&self) -> &[[f64; 6]; 6] {
        &self.matrix
    }

    pub fn was_symmetrized(&self) -> bool {
        self.was_symmetrized
    }
}

impl PoseGraphEdge {
    pub fn try_new(
        from: usize,
        to: usize,
        measurement: Pose64,
        information: [[f64; 6]; 6],
    ) -> Result<Self, PoseGraphEdgeError> {
        validate_distinct_edge_endpoints(from, to)?;
        let measurement = Pose64::try_from_rt(measurement.rotation(), measurement.translation())
            .map_err(|source| PoseGraphEdgeError::Measurement { source })?;
        let information = PoseGraphInformation::try_new(information)
            .map_err(|source| PoseGraphEdgeError::Information { source })?;
        Ok(Self {
            from,
            to,
            measurement,
            information,
        })
    }

    pub(crate) fn try_from_validated_information(
        from: usize,
        to: usize,
        measurement: Pose64,
        information: PoseGraphInformation,
    ) -> Result<Self, PoseGraphEdgeError> {
        validate_distinct_edge_endpoints(from, to)?;
        Ok(Self {
            from,
            to,
            measurement,
            information,
        })
    }

    pub fn from(&self) -> usize {
        self.from
    }

    pub fn to(&self) -> usize {
        self.to
    }

    pub fn measurement(&self) -> Pose64 {
        self.measurement
    }

    pub fn information(&self) -> &PoseGraphInformation {
        &self.information
    }
}

fn validate_distinct_edge_endpoints(from: usize, to: usize) -> Result<(), PoseGraphEdgeError> {
    if from == to {
        Err(PoseGraphEdgeError::SelfEdge { pose_index: from })
    } else {
        Ok(())
    }
}

fn validate_edge_bounds(
    edge: &PoseGraphEdge,
    edge_index: usize,
    pose_count: usize,
) -> Result<(), PoseGraphError> {
    if edge.from >= pose_count {
        return Err(PoseGraphError::EdgeFromOutOfBounds {
            edge_index,
            from: edge.from,
            pose_count,
        });
    }
    if edge.to >= pose_count {
        return Err(PoseGraphError::EdgeToOutOfBounds {
            edge_index,
            to: edge.to,
            pose_count,
        });
    }
    Ok(())
}

fn parse_information(
    mut information: [[f64; 6]; 6],
) -> Result<([[f64; 6]; 6], bool), PoseGraphInformationError> {
    let mut sqrt_diagonal_information = [0.0_f64; 6];
    for (row_index, row) in information.iter().enumerate() {
        for (col_index, value) in row.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(PoseGraphInformationError::NonFiniteEntry {
                    row: row_index,
                    col: col_index,
                    value,
                });
            }
        }
        let diagonal = row[row_index];
        if diagonal <= 0.0 {
            return Err(PoseGraphInformationError::NonPositiveDiagonal {
                axis: row_index,
                value: diagonal,
            });
        }
        sqrt_diagonal_information[row_index] = diagonal.sqrt();
    }

    let mut information_was_symmetrized = false;
    for row in 0..6 {
        for col in (row + 1)..6 {
            let upper = information[row][col];
            let lower = information[col][row];
            let scale = upper.abs().max(lower.abs());
            if (upper - lower).abs() > INFORMATION_RELATIVE_SYMMETRY_TOLERANCE * scale {
                return Err(PoseGraphInformationError::Asymmetric {
                    row,
                    col,
                    upper,
                    lower,
                    relative_tolerance: INFORMATION_RELATIVE_SYMMETRY_TOLERANCE,
                });
            }
            if upper != lower {
                let symmetric = 0.5 * upper + 0.5 * lower;
                information[row][col] = symmetric;
                information[col][row] = symmetric;
                information_was_symmetrized = true;
            }
        }
    }

    let mut normalized = [[0.0_f64; 6]; 6];
    for row in 0..6 {
        for col in 0..6 {
            let value = (information[row][col] / sqrt_diagonal_information[row])
                / sqrt_diagonal_information[col];
            if !value.is_finite() {
                return Err(PoseGraphInformationError::NonFiniteNormalizedEntry {
                    row,
                    col,
                    value,
                });
            }
            normalized[row][col] = value;
        }
    }

    let mut lower = [[0.0_f64; 6]; 6];
    for row in 0..6 {
        for col in 0..=row {
            let mut schur_complement = normalized[row][col];
            for k in 0..col {
                schur_complement -= lower[row][k] * lower[col][k];
            }
            if row == col {
                if !schur_complement.is_finite()
                    || schur_complement <= INFORMATION_NORMALIZED_CHOLESKY_TOLERANCE
                {
                    return Err(PoseGraphInformationError::NotPositiveDefinite {
                        pivot: row,
                        normalized_schur_complement: schur_complement,
                        tolerance: INFORMATION_NORMALIZED_CHOLESKY_TOLERANCE,
                    });
                }
                lower[row][col] = schur_complement.sqrt();
            } else {
                let value = schur_complement / lower[col][col];
                if !value.is_finite() {
                    return Err(PoseGraphInformationError::NotPositiveDefinite {
                        pivot: col,
                        normalized_schur_complement: schur_complement,
                        tolerance: INFORMATION_NORMALIZED_CHOLESKY_TOLERANCE,
                    });
                }
                lower[row][col] = value;
            }
        }
    }
    Ok((information, information_was_symmetrized))
}

fn validate_optimizer_edges(
    edges: &[PoseGraphEdge],
    pose_count: usize,
) -> Result<usize, PoseGraphError> {
    let mut symmetrized_information_count = 0;
    for (edge_index, edge) in edges.iter().enumerate() {
        validate_edge_bounds(edge, edge_index, pose_count)?;
        symmetrized_information_count += usize::from(edge.information.was_symmetrized());
    }
    Ok(symmetrized_information_count)
}

fn edge_error_unchecked(from: usize, to: usize, measurement: Pose64, poses: &[Pose64]) -> [f64; 6] {
    let predicted = poses[from].inverse().compose(poses[to]);
    let residual_pose = predicted.compose(measurement.inverse());
    se3_log_f64(residual_pose)
}

pub fn compute_edge_error(
    edge: &PoseGraphEdge,
    poses: &[Pose64],
) -> Result<[f64; 6], PoseGraphError> {
    validate_edge_bounds(edge, 0, poses.len())?;
    Ok(edge_error_unchecked(
        edge.from,
        edge.to,
        edge.measurement,
        poses,
    ))
}

#[allow(clippy::too_many_arguments)]
fn numerical_diff_column(
    from: usize,
    to: usize,
    measurement: Pose64,
    poses: &mut [Pose64],
    pose_idx: usize,
    delta_plus: &[f64; 6],
    delta_minus: &[f64; 6],
    eps: f64,
    jacobian: &mut [[f64; 6]; 6],
    axis: usize,
) {
    let original = poses[pose_idx];
    poses[pose_idx] = se3_exp_f64(*delta_plus).compose(original);
    let err_plus = edge_error_unchecked(from, to, measurement, poses);
    poses[pose_idx] = se3_exp_f64(*delta_minus).compose(original);
    let err_minus = edge_error_unchecked(from, to, measurement, poses);
    poses[pose_idx] = original;
    for row in 0..6 {
        jacobian[row][axis] = (err_plus[row] - err_minus[row]) / (2.0 * eps);
    }
}

fn compute_edge_jacobians_with_buffer(
    from: usize,
    to: usize,
    measurement: Pose64,
    poses_perturbed: &mut [Pose64],
) -> EdgeJacobians {
    let mut j_from = [[0.0_f64; 6]; 6];
    let mut j_to = [[0.0_f64; 6]; 6];

    for axis in 0..6 {
        let eps = if axis < 3 {
            NUMERICAL_DIFF_TRANSLATION_STEP_M
        } else {
            NUMERICAL_DIFF_ROTATION_STEP_RAD
        };
        let delta_plus = perturb_axis(axis, eps);
        let delta_minus = perturb_axis(axis, -eps);

        numerical_diff_column(
            from,
            to,
            measurement,
            poses_perturbed,
            from,
            &delta_plus,
            &delta_minus,
            eps,
            &mut j_from,
            axis,
        );
        numerical_diff_column(
            from,
            to,
            measurement,
            poses_perturbed,
            to,
            &delta_plus,
            &delta_minus,
            eps,
            &mut j_to,
            axis,
        );
    }

    (j_from, j_to)
}

pub fn compute_edge_jacobians(
    edge: &PoseGraphEdge,
    poses: &[Pose64],
) -> Result<EdgeJacobians, PoseGraphError> {
    validate_edge_bounds(edge, 0, poses.len())?;
    let mut poses_perturbed = poses.to_vec();
    Ok(compute_edge_jacobians_with_buffer(
        edge.from,
        edge.to,
        edge.measurement,
        &mut poses_perturbed,
    ))
}

fn perturb_axis(axis: usize, magnitude: f64) -> [f64; 6] {
    let mut delta = [0.0_f64; 6];
    delta[axis] = magnitude;
    delta
}

#[derive(Clone, Copy, Debug)]
pub struct PoseGraphConfig {
    max_outer_iterations: NonZeroUsize,
    max_pcg_iterations: NonZeroUsize,
    pcg_tol: f64,
    huber_delta_normalized_residual: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PoseGraphConfigError {
    ZeroOuterIterations,
    ZeroPcgIterations,
    InvalidPcgTolerance { value: f64 },
    InvalidNormalizedResidualHuberDelta { value: f64 },
}

impl std::fmt::Display for PoseGraphConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroOuterIterations => {
                f.write_str("pose graph outer iteration limit must be > 0")
            }
            Self::ZeroPcgIterations => f.write_str("pose graph PCG iteration limit must be > 0"),
            Self::InvalidPcgTolerance { value } => write!(
                f,
                "pose graph PCG tolerance must be finite and in (0, 1], got {value}"
            ),
            Self::InvalidNormalizedResidualHuberDelta { value } => write!(
                f,
                "pose graph Huber delta for the dimensionless Mahalanobis residual must be positive and finite, got {value}"
            ),
        }
    }
}

impl std::error::Error for PoseGraphConfigError {}

impl PoseGraphConfig {
    pub fn try_new(
        max_outer_iterations: usize,
        max_pcg_iterations: usize,
        pcg_tol: f64,
        huber_delta_normalized_residual: f64,
    ) -> Result<Self, PoseGraphConfigError> {
        let max_outer_iterations = NonZeroUsize::new(max_outer_iterations)
            .ok_or(PoseGraphConfigError::ZeroOuterIterations)?;
        let max_pcg_iterations =
            NonZeroUsize::new(max_pcg_iterations).ok_or(PoseGraphConfigError::ZeroPcgIterations)?;
        if !pcg_tol.is_finite() || pcg_tol <= 0.0 || pcg_tol > 1.0 {
            return Err(PoseGraphConfigError::InvalidPcgTolerance { value: pcg_tol });
        }
        if !huber_delta_normalized_residual.is_finite() || huber_delta_normalized_residual <= 0.0 {
            return Err(PoseGraphConfigError::InvalidNormalizedResidualHuberDelta {
                value: huber_delta_normalized_residual,
            });
        }
        Ok(Self {
            max_outer_iterations,
            max_pcg_iterations,
            pcg_tol,
            huber_delta_normalized_residual,
        })
    }

    pub fn max_outer_iterations(self) -> usize {
        self.max_outer_iterations.get()
    }

    pub fn max_pcg_iterations(self) -> usize {
        self.max_pcg_iterations.get()
    }

    pub fn pcg_tol(self) -> f64 {
        self.pcg_tol
    }

    pub fn huber_delta_normalized_residual(self) -> f64 {
        self.huber_delta_normalized_residual
    }
}

impl Default for PoseGraphConfig {
    fn default() -> Self {
        const DEFAULT_OUTER_ITERATIONS: NonZeroUsize = NonZeroUsize::new(20).unwrap();
        const DEFAULT_PCG_ITERATIONS: NonZeroUsize = NonZeroUsize::new(100).unwrap();
        Self {
            max_outer_iterations: DEFAULT_OUTER_ITERATIONS,
            max_pcg_iterations: DEFAULT_PCG_ITERATIONS,
            pcg_tol: 1e-6,
            huber_delta_normalized_residual: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PoseGraphResult {
    pub corrected_poses: Vec<Pose64>,
    /// Completed outer Gauss-Newton iterations; zero only for `NoPoses` or
    /// `NoConstraints`.
    pub outer_iterations: usize,
    pub termination: PoseGraphTermination,
    /// Residual norm reported by the final inner PCG solve. This is not the
    /// pose-graph objective or an edge-residual norm; it is zero when no
    /// linear solve ran (`NoPoses` or `NoConstraints`).
    pub last_linear_solve_residual_norm: f64,
    /// Largest applied translation-step norm in the final outer iteration.
    pub last_max_translation_step_m: f64,
    /// Largest applied rotation-step norm in the final outer iteration.
    pub last_max_rotation_step_rad: f64,
    /// Number of per-pose translation steps clamped across all iterations.
    pub clamped_translation_step_count: usize,
    /// Number of per-pose rotation steps clamped across all iterations.
    pub clamped_rotation_step_count: usize,
    /// Edge information matrices whose within-tolerance asymmetry was
    /// explicitly replaced by the pairwise mean at the optimizer boundary.
    pub symmetrized_edge_information_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoseGraphConvergenceCriterion {
    TranslationAndRotationStepNorms,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoseGraphTermination {
    NoPoses,
    NoConstraints,
    Converged {
        criterion: PoseGraphConvergenceCriterion,
    },
    IterationLimit,
}

#[derive(Clone, Debug)]
pub struct PoseGraphOptimizer {
    config: PoseGraphConfig,
}

impl PoseGraphOptimizer {
    pub fn new(config: PoseGraphConfig) -> Self {
        Self { config }
    }

    pub fn optimize(
        &self,
        edges: &[PoseGraphEdge],
        initial_poses: &mut [Pose64],
    ) -> Result<PoseGraphResult, PoseGraphError> {
        let nposes = initial_poses.len();
        let symmetrized_edge_information_count = validate_optimizer_edges(edges, nposes)?;
        if nposes == 0 {
            return Ok(PoseGraphResult {
                corrected_poses: Vec::new(),
                outer_iterations: 0,
                termination: PoseGraphTermination::NoPoses,
                last_linear_solve_residual_norm: 0.0,
                last_max_translation_step_m: 0.0,
                last_max_rotation_step_rad: 0.0,
                clamped_translation_step_count: 0,
                clamped_rotation_step_count: 0,
                symmetrized_edge_information_count,
            });
        }

        let mut poses = initial_poses.to_vec();
        let mut termination = PoseGraphTermination::IterationLimit;
        let mut iters_run = 0;
        let mut last_linear_solve_residual_norm = 0.0_f64;
        let mut last_max_translation_step_m = 0.0_f64;
        let mut last_max_rotation_step_rad = 0.0_f64;
        let mut clamped_translation_step_count = 0_usize;
        let mut clamped_rotation_step_count = 0_usize;
        if edges.is_empty() {
            return Ok(PoseGraphResult {
                corrected_poses: poses,
                outer_iterations: 0,
                termination: PoseGraphTermination::NoConstraints,
                last_linear_solve_residual_norm,
                last_max_translation_step_m,
                last_max_rotation_step_rad,
                clamped_translation_step_count,
                clamped_rotation_step_count,
                symmetrized_edge_information_count,
            });
        }
        let mut numerical_diff_poses = poses.clone();

        for iter in 0..self.config.max_outer_iterations() {
            iters_run = iter + 1;
            let mut h = BlockCsr6x6::new(nposes);
            let mut b = vec![0.0_f64; nposes * 6];

            numerical_diff_poses.copy_from_slice(&poses);
            for (edge_index, edge) in edges.iter().enumerate() {
                let error = edge_error_unchecked(edge.from, edge.to, edge.measurement, &poses);
                let (j_from, j_to) = compute_edge_jacobians_with_buffer(
                    edge.from,
                    edge.to,
                    edge.measurement,
                    &mut numerical_diff_poses,
                );
                let robust_squared_norm = squared_mahalanobis_norm(error, edge.information.matrix);
                if !robust_squared_norm.is_finite() || robust_squared_norm < 0.0 {
                    return Err(PoseGraphError::InvalidRobustSquaredNorm {
                        outer_iteration: iter + 1,
                        edge_index,
                        value: robust_squared_norm,
                    });
                }
                let normalized_residual_norm = robust_squared_norm.sqrt();
                let weight = huber_weight_normalized_residual(
                    normalized_residual_norm,
                    self.config.huber_delta_normalized_residual(),
                );
                let mut information = edge.information.matrix;
                for row in &mut information {
                    for value in row {
                        *value *= weight;
                    }
                }

                let h_ff = jt_info_j(j_from, information, j_from);
                let h_ft = jt_info_j(j_from, information, j_to);
                let h_tf = jt_info_j(j_to, information, j_from);
                let h_tt = jt_info_j(j_to, information, j_to);
                h.add_to(edge.from, edge.from, h_ff)?;
                h.add_to(edge.from, edge.to, h_ft)?;
                h.add_to(edge.to, edge.from, h_tf)?;
                h.add_to(edge.to, edge.to, h_tt)?;

                let g_from = jt_info_vec(j_from, information, error);
                let g_to = jt_info_vec(j_to, information, error);
                for k in 0..6 {
                    b[edge.from * 6 + k] += g_from[k];
                    b[edge.to * 6 + k] += g_to[k];
                }
            }

            // Fix the first pose increment exactly to remove gauge freedom.
            h.fix_block_to_zero_increment(0)?;
            for v in b.iter_mut().take(6) {
                *v = 0.0;
            }

            let rhs: Vec<f64> = b.into_iter().map(|v| -v).collect();
            let mut delta = vec![0.0_f64; nposes * 6];
            let pcg = solve_pcg(
                &h,
                &rhs,
                &mut delta,
                self.config.max_pcg_iterations(),
                self.config.pcg_tol(),
            )?;
            last_linear_solve_residual_norm = pcg.residual_norm;
            if !pcg.residual_norm.is_finite() {
                return Err(PoseGraphError::NonFiniteLinearSolveResidual {
                    outer_iteration: iter + 1,
                    value: pcg.residual_norm,
                });
            }
            if pcg.stop_reason != PcgStopReason::Converged {
                return Err(PoseGraphError::PcgDidNotConverge {
                    outer_iteration: iter + 1,
                    pcg_iterations: pcg.iterations,
                    initial_residual_norm: pcg.initial_residual_norm,
                    residual_norm: pcg.residual_norm,
                    target_residual_norm: pcg.target_residual_norm,
                    stop_reason: pcg.stop_reason,
                });
            }

            last_max_translation_step_m = 0.0;
            last_max_rotation_step_rad = 0.0;
            for (pose_idx, (pose, xi_slice)) in poses
                .iter_mut()
                .zip(delta.chunks_exact(6))
                .enumerate()
                .skip(1)
            {
                let mut xi: [f64; 6] = [
                    xi_slice[0],
                    xi_slice[1],
                    xi_slice[2],
                    xi_slice[3],
                    xi_slice[4],
                    xi_slice[5],
                ];
                let mut translation_step_m = norm3([xi[0], xi[1], xi[2]]);
                let mut rotation_step_rad = norm3([xi[3], xi[4], xi[5]]);
                if !translation_step_m.is_finite() || !rotation_step_rad.is_finite() {
                    return Err(PoseGraphError::NonFiniteStep {
                        outer_iteration: iter + 1,
                        pose_index: pose_idx,
                    });
                }
                if translation_step_m > MAX_TRANSLATION_STEP_M {
                    let scale = MAX_TRANSLATION_STEP_M / translation_step_m;
                    for v in &mut xi[..3] {
                        *v *= scale;
                    }
                    translation_step_m = MAX_TRANSLATION_STEP_M;
                    clamped_translation_step_count += 1;
                }
                if rotation_step_rad > MAX_ROTATION_STEP_RAD {
                    let scale = MAX_ROTATION_STEP_RAD / rotation_step_rad;
                    for v in &mut xi[3..] {
                        *v *= scale;
                    }
                    rotation_step_rad = MAX_ROTATION_STEP_RAD;
                    clamped_rotation_step_count += 1;
                }
                last_max_translation_step_m = last_max_translation_step_m.max(translation_step_m);
                last_max_rotation_step_rad = last_max_rotation_step_rad.max(rotation_step_rad);
                *pose = se3_exp_f64(xi).compose(*pose);
            }

            if last_max_translation_step_m < TRANSLATION_STEP_CONVERGENCE_M
                && last_max_rotation_step_rad < ROTATION_STEP_CONVERGENCE_RAD
            {
                termination = PoseGraphTermination::Converged {
                    criterion: PoseGraphConvergenceCriterion::TranslationAndRotationStepNorms,
                };
                break;
            }
        }

        initial_poses.copy_from_slice(&poses);
        Ok(PoseGraphResult {
            corrected_poses: poses,
            outer_iterations: iters_run,
            termination,
            last_linear_solve_residual_norm,
            last_max_translation_step_m,
            last_max_rotation_step_rad,
            clamped_translation_step_count,
            clamped_rotation_step_count,
            symmetrized_edge_information_count,
        })
    }
}

fn norm3(vector: [f64; 3]) -> f64 {
    vector[0].hypot(vector[1]).hypot(vector[2])
}

fn squared_mahalanobis_norm(error: [f64; 6], information: [[f64; 6]; 6]) -> f64 {
    let information_error = mat6_vec6(information, error);
    error
        .iter()
        .zip(information_error)
        .map(|(error_value, weighted_value)| error_value * weighted_value)
        .sum()
}

fn huber_weight_normalized_residual(norm: f64, delta: f64) -> f64 {
    if norm <= delta || norm <= HUBER_NEAR_ZERO_NORMALIZED_RESIDUAL {
        1.0
    } else {
        delta / norm
    }
}

fn mat6_vec6(matrix: [[f64; 6]; 6], vector: [f64; 6]) -> [f64; 6] {
    std::array::from_fn(|row| {
        matrix[row]
            .iter()
            .zip(vector)
            .map(|(matrix_value, vector_value)| matrix_value * vector_value)
            .sum()
    })
}

fn jt_info_j(a: [[f64; 6]; 6], info: [[f64; 6]; 6], b: [[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut info_b = [[0.0_f64; 6]; 6];
    for row in 0..6 {
        for col in 0..6 {
            for (k, b_row) in b.iter().enumerate() {
                info_b[row][col] += info[row][k] * b_row[col];
            }
        }
    }

    let mut out = [[0.0_f64; 6]; 6];
    for (row, out_row) in out.iter_mut().enumerate() {
        for (col, out_value) in out_row.iter_mut().enumerate() {
            for (k, info_b_row) in info_b.iter().enumerate() {
                *out_value += a[k][row] * info_b_row[col];
            }
        }
    }
    out
}

fn jt_info_vec(j: [[f64; 6]; 6], info: [[f64; 6]; 6], e: [f64; 6]) -> [f64; 6] {
    let mut info_e = [0.0_f64; 6];
    for row in 0..6 {
        for (col, e_value) in e.iter().enumerate() {
            info_e[row] += info[row][col] * *e_value;
        }
    }
    let mut out = [0.0_f64; 6];
    for (row, out_value) in out.iter_mut().enumerate() {
        for (k, info_value) in info_e.iter().enumerate() {
            *out_value += j[k][row] * *info_value;
        }
    }
    out
}

#[cfg(test)]
mod robust_kernel_tests {
    use super::{huber_weight_normalized_residual, squared_mahalanobis_norm};

    #[test]
    fn mahalanobis_norm_combines_translation_and_rotation_through_information_units() {
        let error = [2.0, 0.0, 0.0, 0.5, 0.0, 0.0];
        let mut information = [[0.0_f64; 6]; 6];
        for (axis, row) in information.iter_mut().enumerate() {
            row[axis] = 1.0;
        }
        information[0][0] = 4.0;
        information[3][3] = 9.0;

        let squared = squared_mahalanobis_norm(error, information);
        assert!((squared - 18.25).abs() < f64::EPSILON);
        assert!(
            (huber_weight_normalized_residual(squared.sqrt(), 1.0) - 1.0 / squared.sqrt()).abs()
                < f64::EPSILON
        );
    }
}
