use crate::Pose64;
use crate::math::{mat_mul_vec_f64, se3_exp_f64, se3_log_f64, so3_right_jacobian_f64};

use super::{
    ANCHOR_REGULARIZATION, BlockCsr6x6, HUBER_NEAR_ZERO, MAX_STEP_NORM, NUMERICAL_DIFF_EPS,
    POSE_GRAPH_CONVERGENCE, PoseGraphError, scaled_identity6, solve_pcg,
};

type Jacobian6 = [[f64; 6]; 6];
type EdgeJacobians = (Jacobian6, Jacobian6);

#[derive(Clone, Debug)]
pub struct PoseGraphEdge {
    pub from: usize,
    pub to: usize,
    pub measurement: Pose64,
    pub information: [[f64; 6]; 6],
}

pub fn compute_edge_error(
    edge: &PoseGraphEdge,
    poses: &[Pose64],
) -> Result<[f64; 6], PoseGraphError> {
    if edge.from >= poses.len() {
        return Err(PoseGraphError::EdgeFromOutOfBounds {
            from: edge.from,
            pose_count: poses.len(),
        });
    }
    if edge.to >= poses.len() {
        return Err(PoseGraphError::EdgeToOutOfBounds {
            to: edge.to,
            pose_count: poses.len(),
        });
    }
    let t_from_inv = poses[edge.from].inverse();
    let t_to = poses[edge.to];
    let predicted = t_from_inv.compose(t_to);
    let residual_pose = predicted.compose(edge.measurement.inverse());
    Ok(se3_log_f64(residual_pose))
}

#[allow(clippy::too_many_arguments)]
fn numerical_diff_column(
    edge: &PoseGraphEdge,
    poses: &mut [Pose64],
    pose_idx: usize,
    delta_plus: &[f64; 6],
    delta_minus: &[f64; 6],
    eps: f64,
    jacobian: &mut [[f64; 6]; 6],
    axis: usize,
) -> Result<(), PoseGraphError> {
    let original = poses[pose_idx];
    poses[pose_idx] = se3_exp_f64(*delta_plus).compose(original);
    let err_plus = compute_edge_error(edge, poses)?;
    poses[pose_idx] = se3_exp_f64(*delta_minus).compose(original);
    let err_minus = compute_edge_error(edge, poses)?;
    poses[pose_idx] = original;
    for row in 0..6 {
        jacobian[row][axis] = (err_plus[row] - err_minus[row]) / (2.0 * eps);
    }
    Ok(())
}

pub fn compute_edge_jacobians(
    edge: &PoseGraphEdge,
    poses: &[Pose64],
) -> Result<EdgeJacobians, PoseGraphError> {
    let base_error = compute_edge_error(edge, poses)?;
    let jr = so3_right_jacobian_f64([base_error[3], base_error[4], base_error[5]]);
    let eps = NUMERICAL_DIFF_EPS;
    let mut j_from = [[0.0_f64; 6]; 6];
    let mut j_to = [[0.0_f64; 6]; 6];
    let mut poses_perturbed = poses.to_vec();

    for axis in 0..6 {
        let delta_plus = perturb_axis(axis, eps, jr);
        let delta_minus = perturb_axis(axis, -eps, jr);

        numerical_diff_column(
            edge,
            &mut poses_perturbed,
            edge.from,
            &delta_plus,
            &delta_minus,
            eps,
            &mut j_from,
            axis,
        )?;
        numerical_diff_column(
            edge,
            &mut poses_perturbed,
            edge.to,
            &delta_plus,
            &delta_minus,
            eps,
            &mut j_to,
            axis,
        )?;
    }

    Ok((j_from, j_to))
}

fn perturb_axis(axis: usize, magnitude: f64, right_jacobian: [[f64; 3]; 3]) -> [f64; 6] {
    let mut delta = [0.0_f64; 6];
    if axis < 3 {
        delta[axis] = magnitude;
        return delta;
    }

    let mut unit_rot = [0.0_f64; 3];
    unit_rot[axis - 3] = magnitude;
    let rot = mat_mul_vec_f64(right_jacobian, unit_rot);
    delta[3] = rot[0];
    delta[4] = rot[1];
    delta[5] = rot[2];
    delta
}

#[derive(Clone, Copy, Debug)]
pub struct PoseGraphConfig {
    pub max_iterations: usize,
    pub pcg_max_iters: usize,
    pub pcg_tol: f64,
    pub huber_delta: f64,
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
        if nposes == 0 {
            return Ok(PoseGraphResult {
                corrected_poses: Vec::new(),
                iterations: 0,
                converged: true,
            });
        }

        let mut poses = initial_poses.to_vec();
        let mut converged = false;
        let mut iters_run = 0;
        let mut valid_edges = Vec::with_capacity(edges.len());
        let mut invalid_edges = 0usize;
        for edge in edges {
            if edge.from < nposes && edge.to < nposes {
                valid_edges.push(edge);
            } else {
                invalid_edges = invalid_edges.saturating_add(1);
            }
        }
        if invalid_edges > 0 {
            eprintln!("pose graph skipped {invalid_edges} invalid edges (nposes={nposes})");
        }
        if valid_edges.is_empty() {
            return Ok(PoseGraphResult {
                corrected_poses: poses,
                iterations: 0,
                converged: true,
            });
        }

        for iter in 0..self.config.max_iterations {
            iters_run = iter + 1;
            let mut h = BlockCsr6x6::new(nposes);
            let mut b = vec![0.0_f64; nposes * 6];

            for edge in &valid_edges {
                let error = compute_edge_error(edge, &poses)?;
                let (j_from, j_to) = compute_edge_jacobians(edge, &poses)?;
                let e_norm = error.iter().map(|v| v * v).sum::<f64>().sqrt();
                let weight = huber_weight(e_norm, self.config.huber_delta);
                let mut information = edge.information;
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

            // Anchor the first pose to remove gauge freedom.
            h.add_to(0, 0, scaled_identity6(ANCHOR_REGULARIZATION))?;
            for v in b.iter_mut().take(6) {
                *v = 0.0;
            }

            let rhs: Vec<f64> = b.into_iter().map(|v| -v).collect();
            let mut delta = vec![0.0_f64; nposes * 6];
            let pcg = solve_pcg(
                &h,
                &rhs,
                &mut delta,
                self.config.pcg_max_iters,
                self.config.pcg_tol,
            )?;
            if !pcg.converged && iter + 1 == self.config.max_iterations {
                eprintln!(
                    "pose graph PCG did not converge (iters={}, residual_norm={:.3e})",
                    pcg.iterations, pcg.residual_norm
                );
            }
            if !pcg.residual_norm.is_finite() {
                break;
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
                let mut step_norm = xi.iter().map(|v| v * v).sum::<f64>().sqrt();
                if !step_norm.is_finite() {
                    continue;
                }
                if step_norm > MAX_STEP_NORM {
                    let scale = MAX_STEP_NORM / step_norm;
                    for v in &mut xi {
                        *v *= scale;
                    }
                    step_norm = MAX_STEP_NORM;
                }
                max_step = max_step.max(step_norm);
                *pose = se3_exp_f64(xi).compose(*pose);
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
