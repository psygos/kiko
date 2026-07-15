use crate::Pose64;
use crate::math::{se3_exp_f64, se3_log_f64};

use super::{
    BlockCsr6x6, HUBER_NEAR_ZERO, MAX_STEP_NORM, NUMERICAL_DIFF_EPS, POSE_GRAPH_CONVERGENCE,
    PoseGraphError, scaled_identity6, solve_pcg,
};

type Jacobian6 = [[f64; 6]; 6];
type EdgeJacobians = (Jacobian6, Jacobian6);

#[derive(Clone, Debug)]
pub struct PoseGraphEdge {
    from: usize,
    to: usize,
    /// Camera-`from` to camera-`to` transform for world-to-camera pose variables.
    measurement: Pose64,
    information: [[f64; 6]; 6],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoseGraphEdgeError {
    SelfEdge { index: usize },
    NonFiniteInformation { row: usize, col: usize },
    NonSymmetricInformation { row: usize, col: usize },
    NonPositiveDefiniteInformation { pivot: usize },
}

impl std::fmt::Display for PoseGraphEdgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SelfEdge { index } => {
                write!(
                    f,
                    "pose graph edge must connect distinct poses, got {index}"
                )
            }
            Self::NonFiniteInformation { row, col } => write!(
                f,
                "pose graph information matrix is non-finite at ({row}, {col})"
            ),
            Self::NonSymmetricInformation { row, col } => write!(
                f,
                "pose graph information matrix is not symmetric at ({row}, {col})"
            ),
            Self::NonPositiveDefiniteInformation { pivot } => write!(
                f,
                "pose graph information matrix is not positive definite at pivot {pivot}"
            ),
        }
    }
}

impl std::error::Error for PoseGraphEdgeError {}

impl PoseGraphEdge {
    pub fn try_new(
        from: usize,
        to: usize,
        measurement: Pose64,
        information: [[f64; 6]; 6],
    ) -> Result<Self, PoseGraphEdgeError> {
        if from == to {
            return Err(PoseGraphEdgeError::SelfEdge { index: from });
        }
        validate_information(information)?;
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

    pub fn information(&self) -> [[f64; 6]; 6] {
        self.information
    }
}

fn validate_information(information: [[f64; 6]; 6]) -> Result<(), PoseGraphEdgeError> {
    const SYMMETRY_TOLERANCE: f64 = 1e-12;

    for (row, values) in information.iter().enumerate() {
        for (col, value) in values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(PoseGraphEdgeError::NonFiniteInformation { row, col });
            }
            let scale = value.abs().max(information[col][row].abs()).max(1.0);
            if (value - information[col][row]).abs() > SYMMETRY_TOLERANCE * scale {
                return Err(PoseGraphEdgeError::NonSymmetricInformation { row, col });
            }
        }
    }

    let mut lower = [[0.0_f64; 6]; 6];
    for row in 0..6 {
        for col in 0..=row {
            let product_sum: f64 = (0..col).map(|k| lower[row][k] * lower[col][k]).sum();
            if row == col {
                let pivot = information[row][row] - product_sum;
                if !pivot.is_finite() || pivot <= 0.0 {
                    return Err(PoseGraphEdgeError::NonPositiveDefiniteInformation { pivot: row });
                }
                lower[row][col] = pivot.sqrt();
            } else {
                lower[row][col] = (information[row][col] - product_sum) / lower[col][col];
            }
        }
    }
    Ok(())
}

pub fn compute_edge_error(
    edge: &PoseGraphEdge,
    poses: &[Pose64],
) -> Result<[f64; 6], PoseGraphError> {
    let from_pose = poses
        .get(edge.from)
        .copied()
        .ok_or(PoseGraphError::EdgeFromOutOfBounds {
            from: edge.from,
            pose_count: poses.len(),
        })?;
    let to_pose = poses
        .get(edge.to)
        .copied()
        .ok_or(PoseGraphError::EdgeToOutOfBounds {
            to: edge.to,
            pose_count: poses.len(),
        })?;
    edge_error_from_endpoints(edge, from_pose, to_pose)
}

fn edge_error_from_endpoints(
    edge: &PoseGraphEdge,
    from_pose: Pose64,
    to_pose: Pose64,
) -> Result<[f64; 6], PoseGraphError> {
    let predicted = to_pose.try_compose(from_pose.try_inverse()?)?;
    let residual_pose = predicted.try_compose(edge.measurement.try_inverse()?)?;
    let residual = se3_log_f64(residual_pose);
    if let Some(component) = residual.iter().position(|value| !value.is_finite()) {
        return Err(PoseGraphError::NonFiniteEdgeResidual { component });
    }
    Ok(residual)
}

#[allow(clippy::too_many_arguments)]
fn numerical_diff_column(
    edge: &PoseGraphEdge,
    from_pose: Pose64,
    to_pose: Pose64,
    perturb_from: bool,
    delta_plus: &[f64; 6],
    delta_minus: &[f64; 6],
    eps: f64,
    jacobian: &mut [[f64; 6]; 6],
    axis: usize,
    pose_index: usize,
) -> Result<(), PoseGraphError> {
    let original = if perturb_from { from_pose } else { to_pose };
    let plus = se3_exp_f64(*delta_plus).try_compose(original)?;
    let minus = se3_exp_f64(*delta_minus).try_compose(original)?;
    let (from_plus, to_plus) = if perturb_from {
        (plus, to_pose)
    } else {
        (from_pose, plus)
    };
    let (from_minus, to_minus) = if perturb_from {
        (minus, to_pose)
    } else {
        (from_pose, minus)
    };
    let err_plus = edge_error_from_endpoints(edge, from_plus, to_plus)?;
    let err_minus = edge_error_from_endpoints(edge, from_minus, to_minus)?;
    for row in 0..6 {
        let value = (err_plus[row] - err_minus[row]) / (2.0 * eps);
        if !value.is_finite() {
            return Err(PoseGraphError::NonFiniteEdgeJacobian {
                pose_index,
                row,
                column: axis,
            });
        }
        jacobian[row][axis] = value;
    }
    Ok(())
}

pub fn compute_edge_jacobians(
    edge: &PoseGraphEdge,
    poses: &[Pose64],
) -> Result<EdgeJacobians, PoseGraphError> {
    let eps = NUMERICAL_DIFF_EPS;
    let mut j_from = [[0.0_f64; 6]; 6];
    let mut j_to = [[0.0_f64; 6]; 6];
    let from_pose = poses
        .get(edge.from)
        .copied()
        .ok_or(PoseGraphError::EdgeFromOutOfBounds {
            from: edge.from,
            pose_count: poses.len(),
        })?;
    let to_pose = poses
        .get(edge.to)
        .copied()
        .ok_or(PoseGraphError::EdgeToOutOfBounds {
            to: edge.to,
            pose_count: poses.len(),
        })?;

    for axis in 0..6 {
        let delta_plus = perturb_axis(axis, eps);
        let delta_minus = perturb_axis(axis, -eps);

        numerical_diff_column(
            edge,
            from_pose,
            to_pose,
            true,
            &delta_plus,
            &delta_minus,
            eps,
            &mut j_from,
            axis,
            edge.from,
        )?;
        numerical_diff_column(
            edge,
            from_pose,
            to_pose,
            false,
            &delta_plus,
            &delta_minus,
            eps,
            &mut j_to,
            axis,
            edge.to,
        )?;
    }

    Ok((j_from, j_to))
}

fn perturb_axis(axis: usize, magnitude: f64) -> [f64; 6] {
    let mut delta = [0.0_f64; 6];
    delta[axis] = magnitude;
    delta
}

pub(super) fn clamp_step(xi: &mut [f64; 6], pose_index: usize) -> Result<f64, PoseGraphError> {
    if xi.iter().any(|value| !value.is_finite()) {
        return Err(PoseGraphError::PcgNonFiniteStep { pose_index });
    }
    let step_norm = xi.iter().fold(0.0_f64, |norm, value| norm.hypot(*value));
    if step_norm <= MAX_STEP_NORM {
        return Ok(step_norm);
    }

    let max_component = xi
        .iter()
        .fold(0.0_f64, |max_value, value| max_value.max(value.abs()));
    let normalized_norm = xi
        .iter()
        .fold(0.0_f64, |norm, value| norm.hypot(*value / max_component));
    let scale = (MAX_STEP_NORM / max_component) / normalized_norm;
    for value in xi {
        *value *= scale;
    }
    Ok(MAX_STEP_NORM)
}

#[derive(Clone, Copy, Debug)]
pub struct PoseGraphConfig {
    max_iterations: usize,
    pcg_max_iters: usize,
    pcg_tol: f64,
    huber_delta: f64,
}

impl Default for PoseGraphConfig {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            pcg_max_iters: 100,
            pcg_tol: 1e-6,
            huber_delta: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PoseGraphConfigError {
    ZeroIterations { field: &'static str },
    InvalidPositiveFinite { field: &'static str, value: f64 },
}

impl std::fmt::Display for PoseGraphConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroIterations { field } => write!(f, "{field} must be greater than zero"),
            Self::InvalidPositiveFinite { field, value } => {
                write!(f, "{field} must be positive and finite, got {value}")
            }
        }
    }
}

impl std::error::Error for PoseGraphConfigError {}

impl PoseGraphConfig {
    pub fn try_new(
        max_iterations: usize,
        pcg_max_iters: usize,
        pcg_tol: f64,
        huber_delta: f64,
    ) -> Result<Self, PoseGraphConfigError> {
        if max_iterations == 0 {
            return Err(PoseGraphConfigError::ZeroIterations {
                field: "max_iterations",
            });
        }
        if pcg_max_iters == 0 {
            return Err(PoseGraphConfigError::ZeroIterations {
                field: "pcg_max_iters",
            });
        }
        for (field, value) in [("pcg_tol", pcg_tol), ("huber_delta", huber_delta)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(PoseGraphConfigError::InvalidPositiveFinite { field, value });
            }
        }
        Ok(Self {
            max_iterations,
            pcg_max_iters,
            pcg_tol,
            huber_delta,
        })
    }

    pub fn max_iterations(self) -> usize {
        self.max_iterations
    }

    pub fn pcg_max_iters(self) -> usize {
        self.pcg_max_iters
    }

    pub fn pcg_tol(self) -> f64 {
        self.pcg_tol
    }

    pub fn huber_delta(self) -> f64 {
        self.huber_delta
    }

    #[cfg(test)]
    pub(crate) fn new_unchecked_for_test(
        max_iterations: usize,
        pcg_max_iters: usize,
        pcg_tol: f64,
        huber_delta: f64,
    ) -> Self {
        Self {
            max_iterations,
            pcg_max_iters,
            pcg_tol,
            huber_delta,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PoseGraphResult {
    pub corrected_poses: Vec<Pose64>,
    pub iterations: usize,
    pub converged: bool,
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
        preflight_topology(edges, nposes)?;

        let mut poses = initial_poses.to_vec();
        if edges.is_empty() {
            return Ok(PoseGraphResult {
                corrected_poses: poses,
                iterations: 0,
                converged: true,
            });
        }

        let mut converged = false;
        let mut iters_run = 0;
        for iter in 0..self.config.max_iterations {
            iters_run = iter + 1;
            let mut h = BlockCsr6x6::new(nposes);
            let mut b = vec![0.0_f64; nposes * 6];

            for edge in edges {
                let error = compute_edge_error(edge, &poses)?;
                let (j_from, j_to) = compute_edge_jacobians(edge, &poses)?;
                let e_norm = error.iter().fold(0.0_f64, |norm, value| norm.hypot(*value));
                let weight = huber_weight(e_norm, self.config.huber_delta);
                let mut information = edge.information;
                for row in &mut information {
                    for value in row {
                        *value *= weight;
                    }
                }

                let from_is_anchor = edge.from == 0;
                let to_is_anchor = edge.to == 0;

                if !from_is_anchor {
                    h.add_to(edge.from, edge.from, jt_info_j(j_from, information, j_from))?;
                    let gradient = jt_info_vec(j_from, information, error);
                    for (value, contribution) in b[edge.from * 6..][..6].iter_mut().zip(gradient) {
                        *value += contribution;
                    }
                }
                if !to_is_anchor {
                    h.add_to(edge.to, edge.to, jt_info_j(j_to, information, j_to))?;
                    let gradient = jt_info_vec(j_to, information, error);
                    for (value, contribution) in b[edge.to * 6..][..6].iter_mut().zip(gradient) {
                        *value += contribution;
                    }
                }
                if !from_is_anchor && !to_is_anchor {
                    h.add_to(edge.from, edge.to, jt_info_j(j_from, information, j_to))?;
                    h.add_to(edge.to, edge.from, jt_info_j(j_to, information, j_from))?;
                }
            }

            // Eliminate the first pose from the normal equations. Its identity
            // block keeps the full-sized sparse system nonsingular while the
            // zero right-hand side makes its update exactly zero.
            h.insert(0, 0, scaled_identity6(1.0))?;

            let rhs: Vec<f64> = b.into_iter().map(|v| -v).collect();
            let mut delta = vec![0.0_f64; nposes * 6];
            let pcg = solve_pcg(
                &h,
                &rhs,
                &mut delta,
                self.config.pcg_max_iters,
                self.config.pcg_tol,
            )?;
            if !pcg.residual_norm.is_finite() {
                return Err(PoseGraphError::PcgNonFiniteResidual);
            }
            if !pcg.converged {
                return Err(PoseGraphError::PcgDidNotConverge {
                    iterations: pcg.iterations,
                });
            }

            let mut max_step = 0.0_f64;
            for (pose_idx, pose) in poses.iter_mut().enumerate().skip(1) {
                let base = pose_idx * 6;
                let Some(xi_slice) = delta.get(base..base + 6) else {
                    continue;
                };
                let mut xi: [f64; 6] = [
                    xi_slice[0],
                    xi_slice[1],
                    xi_slice[2],
                    xi_slice[3],
                    xi_slice[4],
                    xi_slice[5],
                ];
                let step_norm = clamp_step(&mut xi, pose_idx)?;
                max_step = max_step.max(step_norm);
                *pose = se3_exp_f64(xi).try_compose(*pose)?;
            }

            if max_step < POSE_GRAPH_CONVERGENCE {
                converged = true;
                break;
            }
        }

        initial_poses.copy_from_slice(&poses);
        Ok(PoseGraphResult {
            corrected_poses: poses,
            iterations: iters_run,
            converged,
        })
    }
}

fn preflight_topology(edges: &[PoseGraphEdge], pose_count: usize) -> Result<(), PoseGraphError> {
    for edge in edges {
        if edge.from >= pose_count {
            return Err(PoseGraphError::EdgeFromOutOfBounds {
                from: edge.from,
                pose_count,
            });
        }
        if edge.to >= pose_count {
            return Err(PoseGraphError::EdgeToOutOfBounds {
                to: edge.to,
                pose_count,
            });
        }
    }

    if pose_count <= 1 {
        return Ok(());
    }
    if edges.is_empty() {
        return Err(PoseGraphError::UnconstrainedPoseGraph { pose_count });
    }

    let mut parents: Vec<usize> = (0..pose_count).collect();
    for edge in edges {
        let from_root = find_root(&mut parents, edge.from);
        let to_root = find_root(&mut parents, edge.to);
        if from_root != to_root {
            parents[to_root] = from_root;
        }
    }

    let anchor_root = find_root(&mut parents, 0);
    let mut anchor_component_size = 0;
    let mut component_count = 0;
    for pose_index in 0..pose_count {
        let root = find_root(&mut parents, pose_index);
        component_count += usize::from(root == pose_index);
        anchor_component_size += usize::from(root == anchor_root);
    }
    if component_count > 1 {
        return Err(PoseGraphError::DisconnectedPoseGraph {
            pose_count,
            component_count,
            anchor_component_size,
        });
    }

    Ok(())
}

fn find_root(parents: &mut [usize], pose_index: usize) -> usize {
    let mut root = pose_index;
    while parents[root] != root {
        root = parents[root];
    }

    let mut current = pose_index;
    while parents[current] != current {
        let next = parents[current];
        parents[current] = root;
        current = next;
    }
    root
}

fn huber_weight(norm: f64, delta: f64) -> f64 {
    if norm <= delta || norm <= HUBER_NEAR_ZERO {
        1.0
    } else {
        delta / norm
    }
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
