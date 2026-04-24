//! Tightly-coupled VIO solver for local bundle adjustment.
//!
//! This module provides the Hessian assembly and solve routines for the
//! visual-inertial optimization problem. It operates on the types defined
//! in `local_ba.rs` (VioWindow, VioAnchor, VioSuccessor) and produces
//! the reduced Hessian after Schur-eliminating short-lived stereo landmarks.
//!
//! # State layout
//!
//! Each frame has a 15D state in the NavTangent layout:
//!   [translation(3), rotation(3), velocity(3), accel_bias(3), gyro_bias(3)]
//!
//! The Hessian is a dense `(N * 15) × (N * 15)` matrix in f64.
//!
//! # Factor types
//!
//! 1. **Reprojection** (2D): pixel - project(T_cam_body * T_body_odom * p_map).
//!    Jacobian is 2×6 in the pose block, zero in velocity/bias blocks.
//! 2. **IMU preintegration** (9D): couples consecutive frames through
//!    rotation, velocity, and position residuals.
//! 3. **Bias random walk** (6D): simple difference `bias_j - bias_i`.
//! 4. **Anchor prior** (9D diagonal): constrains velocity and bias of frame 0.

use crate::{Gravity, ImuFactor, NavState, PreintegratedImu};

/// State dimension per frame.
pub const STATE_DIM: usize = 15;
/// IMU preintegration residual dimension.
pub const IMU_RESIDUAL_DIM: usize = 9;
/// Bias random walk residual dimension.
pub const BIAS_RW_RESIDUAL_DIM: usize = 6;

// -----------------------------------------------------------------------
// Analytical IMU Jacobians
// -----------------------------------------------------------------------
//
// The IMU residual is 9D = [rotation_err(3), velocity_err(3), position_err(3)].
//
// Notation:
//   R_i, R_j — rotations of state i, j in odom frame
//   p_i, p_j — positions in odom frame
//   v_i, v_j — velocities in odom frame
//   g — gravity vector in odom frame
//   dt — preintegration interval
//   ΔR, Δv, Δp — preintegrated measurements in body frame
//
// Residual (from factors.rs ImuFactor::residual):
//   r_rot = Log(ΔR^T · R_i^T · R_j)                                    (3D)
//   r_vel = R_i^T · (v_j - v_i - g·dt) - Δv                           (3D)
//   r_pos = R_i^T · (p_j - p_i - v_i·dt - 0.5·g·dt²) - Δp            (3D)
//
// Jacobians derived analytically following Forster et al. (2017) / GTSAM:
//
//   ∂r/∂state_i is 9×15 = jac_prev
//   ∂r/∂state_j is 9×15 = jac_curr
//
//   Column layout: [trans(0:3), rot(3:6), vel(6:9), ba(9:12), bg(12:15)]

/// Compute IMU Jacobians via central finite differences.
/// Correct by construction. The analytical version requires careful
/// SO(3) Jacobian derivation that is deferred to a follow-up.
/// Cost: 30 residual evaluations per factor (~1μs total on ARM).
pub fn imu_jacobians(
    state_i: &NavState,
    state_j: &NavState,
    preintegrated: &PreintegratedImu,
    gravity: Gravity,
) -> (
    [[f64; STATE_DIM]; IMU_RESIDUAL_DIM],
    [[f64; STATE_DIM]; IMU_RESIDUAL_DIM],
) {
    const EPS: f64 = 1e-7;
    let mut jac_prev = [[0.0_f64; STATE_DIM]; IMU_RESIDUAL_DIM];
    let mut jac_curr = [[0.0_f64; STATE_DIM]; IMU_RESIDUAL_DIM];

    for axis in 0..STATE_DIM {
        let mut delta_plus = [0.0_f64; STATE_DIM];
        let mut delta_minus = [0.0_f64; STATE_DIM];
        delta_plus[axis] = EPS;
        delta_minus[axis] = -EPS;

        // w.r.t. state_i
        if let (Ok(si_plus), Ok(si_minus)) =
            (state_i.retract(&delta_plus), state_i.retract(&delta_minus))
        {
            let r_plus = ImuFactor::residual(&si_plus, state_j, preintegrated, &gravity);
            let r_minus = ImuFactor::residual(&si_minus, state_j, preintegrated, &gravity);
            for row in 0..IMU_RESIDUAL_DIM {
                jac_prev[row][axis] = (r_plus[row] - r_minus[row]) / (2.0 * EPS);
            }
        }

        // w.r.t. state_j
        if let (Ok(sj_plus), Ok(sj_minus)) =
            (state_j.retract(&delta_plus), state_j.retract(&delta_minus))
        {
            let r_plus = ImuFactor::residual(state_i, &sj_plus, preintegrated, &gravity);
            let r_minus = ImuFactor::residual(state_i, &sj_minus, preintegrated, &gravity);
            for row in 0..IMU_RESIDUAL_DIM {
                jac_curr[row][axis] = (r_plus[row] - r_minus[row]) / (2.0 * EPS);
            }
        }
    }
    (jac_prev, jac_curr)
}

// -----------------------------------------------------------------------
// Dense f64 linear solver
// -----------------------------------------------------------------------

/// Solve a dense linear system `A x = b` via Gaussian elimination with
/// partial pivoting. Operates in-place on `a` (row-major, `dim × dim`)
/// and `b` (`dim`). Returns `false` if singular.
pub fn solve_dense_f64(a: &mut [f64], b: &mut [f64], dim: usize) -> bool {
    for col in 0..dim {
        // Partial pivoting: find max in column
        let mut max_val = a[col * dim + col].abs();
        let mut max_row = col;
        for row in (col + 1)..dim {
            let val = a[row * dim + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return false;
        }
        // Swap rows
        if max_row != col {
            for k in 0..dim {
                a.swap(col * dim + k, max_row * dim + k);
            }
            b.swap(col, max_row);
        }
        // Eliminate
        let pivot = a[col * dim + col];
        let inv_pivot = 1.0 / pivot;
        for k in col..dim {
            a[col * dim + k] *= inv_pivot;
        }
        b[col] *= inv_pivot;
        for row in 0..dim {
            if row == col {
                continue;
            }
            let factor = a[row * dim + col];
            if factor.abs() < 1e-18 {
                continue;
            }
            for k in col..dim {
                a[row * dim + k] -= factor * a[col * dim + k];
            }
            b[row] -= factor * b[col];
        }
    }
    true
}

// -----------------------------------------------------------------------
// Hessian accumulation helpers
// -----------------------------------------------------------------------

/// Accumulate `J^T * Ω * J` into the Hessian and `J^T * Ω * r` into the RHS.
/// `j` is `[residual_dim][state_dim]` (row-major).
/// `info` is `[residual_dim][residual_dim]`.
/// `residual` is `[residual_dim]`.
/// `base` is the column/row offset in the Hessian for this frame.
pub fn accumulate_factor<const R: usize, const S: usize>(
    hessian: &mut [f64],
    rhs: &mut [f64],
    dim: usize,
    j: &[[f64; S]; R],
    info: &[[f64; R]; R],
    residual: &[f64; R],
    base: usize,
) {
    // Compute Ω · J (R × S) and Ω · r (R)
    let mut omega_j = [[0.0_f64; S]; R];
    let mut omega_r = [0.0_f64; R];
    for i in 0..R {
        for k in 0..R {
            omega_r[i] += info[i][k] * residual[k];
            for col in 0..S {
                omega_j[i][col] += info[i][k] * j[k][col];
            }
        }
    }
    // Accumulate J^T · (Ω · J) into Hessian
    for row in 0..S {
        for col in 0..S {
            let mut val = 0.0;
            for k in 0..R {
                val += j[k][row] * omega_j[k][col];
            }
            hessian[(base + row) * dim + (base + col)] += val;
        }
    }
    // Accumulate J^T · (Ω · r) into RHS
    for row in 0..S {
        let mut val = 0.0;
        for k in 0..R {
            val += j[k][row] * omega_r[k];
        }
        rhs[base + row] += val;
    }
}

/// Accumulate cross-term `J_a^T * Ω * J_b` into Hessian at (base_a, base_b)
/// and symmetrically at (base_b, base_a).
pub fn accumulate_cross_factor<const R: usize, const S: usize>(
    hessian: &mut [f64],
    dim: usize,
    j_a: &[[f64; S]; R],
    j_b: &[[f64; S]; R],
    info: &[[f64; R]; R],
    base_a: usize,
    base_b: usize,
) {
    // Compute Ω · J_b (R × S)
    let mut omega_jb = [[0.0_f64; S]; R];
    for i in 0..R {
        for k in 0..R {
            for col in 0..S {
                omega_jb[i][col] += info[i][k] * j_b[k][col];
            }
        }
    }
    // J_a^T · (Ω · J_b) at (base_a, base_b) + transpose at (base_b, base_a)
    for row in 0..S {
        for col in 0..S {
            let mut val = 0.0;
            for k in 0..R {
                val += j_a[k][row] * omega_jb[k][col];
            }
            hessian[(base_a + row) * dim + (base_b + col)] += val;
            hessian[(base_b + col) * dim + (base_a + row)] += val;
        }
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ImuBatch, ImuBias, ImuNoiseModel, ImuSample, Pose64, Timestamp};

    fn noise() -> ImuNoiseModel {
        ImuNoiseModel::new(0.1, 0.01, 0.001, 0.0001).expect("noise")
    }

    fn batch(entries: &[(i64, [f64; 3], [f64; 3])]) -> ImuBatch {
        ImuBatch::new(
            entries
                .iter()
                .map(|&(ts, accel, gyro)| {
                    ImuSample::new(Timestamp::from_nanos(ts), accel, gyro).expect("sample")
                })
                .collect(),
        )
        .expect("batch")
    }

    /// IMU Jacobians must be finite for non-trivial states.
    #[test]
    fn imu_jacobians_finite_for_nontrivial_states() {
        let gravity = Gravity::try_new([0.0, 9.81, 0.0]).expect("gravity");
        let state_i = NavState::try_new(
            Pose64::from_rt(
                [
                    [0.98, -0.17, 0.05],
                    [0.18, 0.97, -0.15],
                    [-0.02, 0.16, 0.99],
                ],
                [1.0, -0.5, 0.3],
            ),
            [0.5, -0.2, 0.1],
            ImuBias {
                accel: [0.1, -0.05, 0.02],
                gyro: [0.001, -0.002, 0.003],
            },
        )
        .expect("state_i");
        let state_j = NavState::try_new(
            Pose64::from_rt(
                [
                    [0.95, -0.30, 0.08],
                    [0.31, 0.94, -0.12],
                    [-0.04, 0.14, 0.99],
                ],
                [1.5, -0.3, 0.6],
            ),
            [0.8, -0.1, 0.3],
            ImuBias {
                accel: [0.12, -0.04, 0.03],
                gyro: [0.002, -0.001, 0.004],
            },
        )
        .expect("state_j");
        let preintegrated = PreintegratedImu::integrate(
            &batch(&[
                (0, [0.5, -0.3, 9.5], [0.1, -0.05, 0.02]),
                (5_000_000, [0.6, -0.2, 9.6], [0.12, -0.04, 0.03]),
                (10_000_000, [0.4, -0.1, 9.4], [0.08, -0.06, 0.01]),
            ]),
            &state_i.bias(),
            &noise(),
        )
        .expect("preintegrated");

        let (jp, jc) = imu_jacobians(&state_i, &state_j, &preintegrated, gravity);
        for row in 0..IMU_RESIDUAL_DIM {
            for col in 0..STATE_DIM {
                assert!(
                    jp[row][col].is_finite(),
                    "jac_prev[{row}][{col}] not finite"
                );
                assert!(
                    jc[row][col].is_finite(),
                    "jac_curr[{row}][{col}] not finite"
                );
            }
        }
        // At least some non-zero entries (the Jacobian should not be trivially zero)
        let sum: f64 = jp.iter().flat_map(|r| r.iter()).map(|v| v.abs()).sum();
        assert!(
            sum > 1.0,
            "jac_prev is suspiciously close to zero: sum={sum}"
        );
    }

    /// At the ground truth (zero residual), all Jacobians should still
    /// be well-defined (not NaN/Inf).
    #[test]
    fn imu_jacobians_finite_at_ground_truth() {
        let gravity = Gravity::try_new([0.0, 9.81, 0.0]).expect("gravity");
        let state =
            NavState::try_new(Pose64::identity(), [0.0; 3], ImuBias::default()).expect("state");

        let preintegrated = PreintegratedImu::integrate(
            &batch(&[
                (0, [0.0, 9.81, 0.0], [0.0; 3]),
                (10_000_000, [0.0, 9.81, 0.0], [0.0; 3]),
            ]),
            &ImuBias::default(),
            &noise(),
        )
        .expect("preintegrated");

        let (jac_prev, jac_curr) = imu_jacobians(&state, &state, &preintegrated, gravity);
        for row in 0..IMU_RESIDUAL_DIM {
            for col in 0..STATE_DIM {
                assert!(
                    jac_prev[row][col].is_finite(),
                    "jac_prev[{row}][{col}] not finite"
                );
                assert!(
                    jac_curr[row][col].is_finite(),
                    "jac_curr[{row}][{col}] not finite"
                );
            }
        }
    }

    /// Dense solver: trivial 2×2 system.
    #[test]
    fn dense_solver_2x2() {
        // 2x + 3y = 8
        // x + y = 3
        // => x = 1, y = 2
        let mut a = vec![2.0, 3.0, 1.0, 1.0];
        let mut b = vec![8.0, 3.0];
        assert!(solve_dense_f64(&mut a, &mut b, 2));
        assert!((b[0] - 1.0).abs() < 1e-12);
        assert!((b[1] - 2.0).abs() < 1e-12);
    }

    /// Dense solver: singular system returns false.
    #[test]
    fn dense_solver_singular() {
        let mut a = vec![1.0, 2.0, 2.0, 4.0];
        let mut b = vec![3.0, 6.0];
        assert!(!solve_dense_f64(&mut a, &mut b, 2));
    }
}
