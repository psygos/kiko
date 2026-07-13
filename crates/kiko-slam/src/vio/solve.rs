//! Numerical helpers for tightly-coupled VIO local bundle adjustment.
//!
//! The optimizer and window state live in `local_ba.rs`. This module provides
//! finite-difference IMU Jacobians, dense linear solves, and factor-accumulation
//! helpers used by that optimizer.
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
//! 1. **Reprojection** (2D): map points are reframed into odometry, then into
//!    the camera by `camera_from_body * body_from_odom` before projection.
//!    Jacobians affect the six pose tangent components only.
//! 2. **IMU preintegration** (9D): couples consecutive frames through
//!    rotation, velocity, and position residuals.
//! 3. **Bias random walk** (6D): simple difference `bias_j - bias_i`.
//! 4. **Velocity anchor** (3D diagonal): constrains velocity of frame 0.
//! 5. **Calibrated bias prior** (6D diagonal, optional): constrains each frame.

use crate::{Gravity, ImuFactor, NavState, NavStateError, PreintegratedImu, VioFactorError};

const DENSE_PIVOT_EPSILON_MULTIPLIER: f64 = 16.0;

/// State dimension per frame.
pub const STATE_DIM: usize = 15;
/// IMU preintegration residual dimension.
pub const IMU_RESIDUAL_DIM: usize = 9;
// -----------------------------------------------------------------------
// IMU residual and Jacobian layout
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
// The current implementation estimates these Jacobians by central differences:
//
//   ∂r/∂state_i is 9×15 = jac_prev
//   ∂r/∂state_j is 9×15 = jac_curr
//
//   Column layout: [trans(0:3), rot(3:6), vel(6:9), ba(9:12), bg(12:15)]

/// Compute IMU Jacobians via central finite differences.
/// This avoids relying on an unverified analytical SO(3) derivative, but
/// accuracy still depends on the finite-difference step and local conditioning.
/// The two endpoint Jacobians require 60 residual evaluations in total.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImuJacobianEndpoint {
    PreviousState,
    CurrentState,
}

impl std::fmt::Display for ImuJacobianEndpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::PreviousState => "previous state",
            Self::CurrentState => "current state",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FiniteDifferenceSide {
    Positive,
    Negative,
}

impl std::fmt::Display for FiniteDifferenceSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Positive => "positive",
            Self::Negative => "negative",
        })
    }
}

#[derive(Debug)]
pub enum ImuJacobianError {
    StateRetraction {
        endpoint: ImuJacobianEndpoint,
        side: FiniteDifferenceSide,
        tangent_axis: usize,
        source: NavStateError,
    },
    Residual {
        endpoint: ImuJacobianEndpoint,
        side: FiniteDifferenceSide,
        tangent_axis: usize,
        source: VioFactorError,
    },
    NonFiniteDerivative {
        endpoint: ImuJacobianEndpoint,
        residual_axis: usize,
        tangent_axis: usize,
        value: f64,
    },
}

impl std::fmt::Display for ImuJacobianError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StateRetraction {
                endpoint,
                side,
                tangent_axis,
                source,
            } => write!(
                f,
                "IMU Jacobian {endpoint} {side} retraction failed at tangent axis {tangent_axis}: {source}"
            ),
            Self::Residual {
                endpoint,
                side,
                tangent_axis,
                source,
            } => write!(
                f,
                "IMU Jacobian {endpoint} {side} residual failed at tangent axis {tangent_axis}: {source}"
            ),
            Self::NonFiniteDerivative {
                endpoint,
                residual_axis,
                tangent_axis,
                value,
            } => write!(
                f,
                "IMU Jacobian {endpoint} entry ({residual_axis}, {tangent_axis}) must be finite, got {value}"
            ),
        }
    }
}

impl std::error::Error for ImuJacobianError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::StateRetraction { source, .. } => Some(source),
            Self::Residual { source, .. } => Some(source),
            Self::NonFiniteDerivative { .. } => None,
        }
    }
}

pub type ImuStateJacobian = [[f64; STATE_DIM]; IMU_RESIDUAL_DIM];
pub type ImuJacobianPair = (ImuStateJacobian, ImuStateJacobian);

const IMU_JACOBIAN_TRANSLATION_STEP_M: f64 = 1e-7;
const IMU_JACOBIAN_ROTATION_STEP_RAD: f64 = 1e-7;
const IMU_JACOBIAN_VELOCITY_STEP_MPS: f64 = 1e-7;
const IMU_JACOBIAN_ACCEL_BIAS_STEP_MPS2: f64 = 1e-7;
const IMU_JACOBIAN_GYRO_BIAS_STEP_RADPS: f64 = 1e-7;
const IMU_JACOBIAN_STEPS: [f64; STATE_DIM] = [
    IMU_JACOBIAN_TRANSLATION_STEP_M,
    IMU_JACOBIAN_TRANSLATION_STEP_M,
    IMU_JACOBIAN_TRANSLATION_STEP_M,
    IMU_JACOBIAN_ROTATION_STEP_RAD,
    IMU_JACOBIAN_ROTATION_STEP_RAD,
    IMU_JACOBIAN_ROTATION_STEP_RAD,
    IMU_JACOBIAN_VELOCITY_STEP_MPS,
    IMU_JACOBIAN_VELOCITY_STEP_MPS,
    IMU_JACOBIAN_VELOCITY_STEP_MPS,
    IMU_JACOBIAN_ACCEL_BIAS_STEP_MPS2,
    IMU_JACOBIAN_ACCEL_BIAS_STEP_MPS2,
    IMU_JACOBIAN_ACCEL_BIAS_STEP_MPS2,
    IMU_JACOBIAN_GYRO_BIAS_STEP_RADPS,
    IMU_JACOBIAN_GYRO_BIAS_STEP_RADPS,
    IMU_JACOBIAN_GYRO_BIAS_STEP_RADPS,
];

pub fn imu_jacobians(
    state_i: &NavState,
    state_j: &NavState,
    preintegrated: &PreintegratedImu,
    gravity: Gravity,
) -> Result<ImuJacobianPair, ImuJacobianError> {
    let mut jac_prev = [[0.0_f64; STATE_DIM]; IMU_RESIDUAL_DIM];
    let mut jac_curr = [[0.0_f64; STATE_DIM]; IMU_RESIDUAL_DIM];

    for (axis, step) in IMU_JACOBIAN_STEPS.into_iter().enumerate() {
        let mut delta_plus = [0.0_f64; STATE_DIM];
        let mut delta_minus = [0.0_f64; STATE_DIM];
        delta_plus[axis] = step;
        delta_minus[axis] = -step;

        let si_plus =
            state_i
                .retract(&delta_plus)
                .map_err(|source| ImuJacobianError::StateRetraction {
                    endpoint: ImuJacobianEndpoint::PreviousState,
                    side: FiniteDifferenceSide::Positive,
                    tangent_axis: axis,
                    source,
                })?;
        let si_minus =
            state_i
                .retract(&delta_minus)
                .map_err(|source| ImuJacobianError::StateRetraction {
                    endpoint: ImuJacobianEndpoint::PreviousState,
                    side: FiniteDifferenceSide::Negative,
                    tangent_axis: axis,
                    source,
                })?;
        let r_plus =
            ImuFactor::residual(&si_plus, state_j, preintegrated, &gravity).map_err(|source| {
                ImuJacobianError::Residual {
                    endpoint: ImuJacobianEndpoint::PreviousState,
                    side: FiniteDifferenceSide::Positive,
                    tangent_axis: axis,
                    source,
                }
            })?;
        let r_minus =
            ImuFactor::residual(&si_minus, state_j, preintegrated, &gravity).map_err(|source| {
                ImuJacobianError::Residual {
                    endpoint: ImuJacobianEndpoint::PreviousState,
                    side: FiniteDifferenceSide::Negative,
                    tangent_axis: axis,
                    source,
                }
            })?;
        for row in 0..IMU_RESIDUAL_DIM {
            let derivative = (r_plus[row] - r_minus[row]) / (2.0 * step);
            if !derivative.is_finite() {
                return Err(ImuJacobianError::NonFiniteDerivative {
                    endpoint: ImuJacobianEndpoint::PreviousState,
                    residual_axis: row,
                    tangent_axis: axis,
                    value: derivative,
                });
            }
            jac_prev[row][axis] = derivative;
        }

        let sj_plus =
            state_j
                .retract(&delta_plus)
                .map_err(|source| ImuJacobianError::StateRetraction {
                    endpoint: ImuJacobianEndpoint::CurrentState,
                    side: FiniteDifferenceSide::Positive,
                    tangent_axis: axis,
                    source,
                })?;
        let sj_minus =
            state_j
                .retract(&delta_minus)
                .map_err(|source| ImuJacobianError::StateRetraction {
                    endpoint: ImuJacobianEndpoint::CurrentState,
                    side: FiniteDifferenceSide::Negative,
                    tangent_axis: axis,
                    source,
                })?;
        let r_plus =
            ImuFactor::residual(state_i, &sj_plus, preintegrated, &gravity).map_err(|source| {
                ImuJacobianError::Residual {
                    endpoint: ImuJacobianEndpoint::CurrentState,
                    side: FiniteDifferenceSide::Positive,
                    tangent_axis: axis,
                    source,
                }
            })?;
        let r_minus =
            ImuFactor::residual(state_i, &sj_minus, preintegrated, &gravity).map_err(|source| {
                ImuJacobianError::Residual {
                    endpoint: ImuJacobianEndpoint::CurrentState,
                    side: FiniteDifferenceSide::Negative,
                    tangent_axis: axis,
                    source,
                }
            })?;
        for row in 0..IMU_RESIDUAL_DIM {
            let derivative = (r_plus[row] - r_minus[row]) / (2.0 * step);
            if !derivative.is_finite() {
                return Err(ImuJacobianError::NonFiniteDerivative {
                    endpoint: ImuJacobianEndpoint::CurrentState,
                    residual_axis: row,
                    tangent_axis: axis,
                    value: derivative,
                });
            }
            jac_curr[row][axis] = derivative;
        }
    }
    Ok((jac_prev, jac_curr))
}

// -----------------------------------------------------------------------
// Dense f64 linear solver
// -----------------------------------------------------------------------

/// Dense-solver input whose finite-domain invariant was violated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseSolveInput {
    Matrix,
    RightHandSide,
}

impl std::fmt::Display for DenseSolveInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Matrix => "matrix",
            Self::RightHandSide => "right-hand side",
        })
    }
}

/// Solve a dense linear system `A x = b` via in-place Gauss-Jordan
/// elimination with scaled partial pivoting. `a` is row-major `dim × dim`;
/// `b` initially holds the right-hand side and receives the solution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DenseSolveError {
    DimensionOverflow {
        dim: usize,
    },
    DimensionMismatch {
        dim: usize,
        matrix_len: usize,
        rhs_len: usize,
    },
    NonFiniteInput {
        input: DenseSolveInput,
        index: usize,
        value: f64,
    },
    SingularOrIllConditioned {
        column: usize,
        pivot_magnitude: f64,
        row_scale: f64,
        scaled_pivot: f64,
        tolerance: f64,
    },
    NonFiniteEliminatedMatrix {
        index: usize,
        value: f64,
    },
    NonFiniteSolution {
        index: usize,
        value: f64,
    },
}

impl std::fmt::Display for DenseSolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionOverflow { dim } => {
                write!(f, "dense solver dimension {dim} overflows matrix size")
            }
            Self::DimensionMismatch {
                dim,
                matrix_len,
                rhs_len,
            } => write!(
                f,
                "dense solver expected a {dim}x{dim} matrix and RHS length {dim}, got matrix length {matrix_len} and RHS length {rhs_len}"
            ),
            Self::NonFiniteInput {
                input,
                index,
                value,
            } => write!(
                f,
                "dense solver {input}[{index}] must be finite, got {value}"
            ),
            Self::SingularOrIllConditioned {
                column,
                pivot_magnitude,
                row_scale,
                scaled_pivot,
                tolerance,
            } => write!(
                f,
                "dense solver is singular or ill-conditioned at column {column}: pivot magnitude {pivot_magnitude}, row scale {row_scale}, scaled pivot {scaled_pivot} <= tolerance {tolerance}"
            ),
            Self::NonFiniteEliminatedMatrix { index, value } => write!(
                f,
                "dense solver eliminated matrix[{index}] is non-finite: {value}"
            ),
            Self::NonFiniteSolution { index, value } => {
                write!(f, "dense solver solution[{index}] is non-finite: {value}")
            }
        }
    }
}

impl std::error::Error for DenseSolveError {}

pub fn solve_dense_f64(a: &mut [f64], b: &mut [f64], dim: usize) -> Result<(), DenseSolveError> {
    let matrix_len = dim
        .checked_mul(dim)
        .ok_or(DenseSolveError::DimensionOverflow { dim })?;
    if a.len() != matrix_len || b.len() != dim {
        return Err(DenseSolveError::DimensionMismatch {
            dim,
            matrix_len: a.len(),
            rhs_len: b.len(),
        });
    }
    for (index, &value) in a.iter().enumerate() {
        if !value.is_finite() {
            return Err(DenseSolveError::NonFiniteInput {
                input: DenseSolveInput::Matrix,
                index,
                value,
            });
        }
    }
    for (index, &value) in b.iter().enumerate() {
        if !value.is_finite() {
            return Err(DenseSolveError::NonFiniteInput {
                input: DenseSolveInput::RightHandSide,
                index,
                value,
            });
        }
    }

    let mut row_scales = (0..dim)
        .map(|row| {
            a[row * dim..(row + 1) * dim]
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max)
        })
        .collect::<Vec<_>>();
    let scaled_pivot_tolerance = DENSE_PIVOT_EPSILON_MULTIPLIER * f64::EPSILON * dim.max(1) as f64;

    for col in 0..dim {
        // Scaled partial pivoting makes the breakdown decision independent of
        // the physical units or a uniform rescaling of the linear system.
        let mut max_row = col;
        let mut max_scaled_pivot = if row_scales[col] > 0.0 {
            a[col * dim + col].abs() / row_scales[col]
        } else {
            0.0
        };
        for row in (col + 1)..dim {
            let candidate = if row_scales[row] > 0.0 {
                a[row * dim + col].abs() / row_scales[row]
            } else {
                0.0
            };
            if candidate > max_scaled_pivot {
                max_scaled_pivot = candidate;
                max_row = row;
            }
        }
        let pivot_magnitude = a[max_row * dim + col].abs();
        let row_scale = row_scales[max_row];
        if !max_scaled_pivot.is_finite() || max_scaled_pivot <= scaled_pivot_tolerance {
            return Err(DenseSolveError::SingularOrIllConditioned {
                column: col,
                pivot_magnitude,
                row_scale,
                scaled_pivot: max_scaled_pivot,
                tolerance: scaled_pivot_tolerance,
            });
        }
        if max_row != col {
            for k in 0..dim {
                a.swap(col * dim + k, max_row * dim + k);
            }
            b.swap(col, max_row);
            row_scales.swap(col, max_row);
        }
        let pivot = a[col * dim + col];
        for k in col..dim {
            a[col * dim + k] /= pivot;
        }
        b[col] /= pivot;
        for row in 0..dim {
            if row == col {
                continue;
            }
            let factor = a[row * dim + col];
            if factor == 0.0 {
                continue;
            }
            for k in col..dim {
                a[row * dim + k] -= factor * a[col * dim + k];
            }
            b[row] -= factor * b[col];
        }
    }
    for (index, &value) in a.iter().enumerate() {
        if !value.is_finite() {
            return Err(DenseSolveError::NonFiniteEliminatedMatrix { index, value });
        }
    }
    for (index, &value) in b.iter().enumerate() {
        if !value.is_finite() {
            return Err(DenseSolveError::NonFiniteSolution { index, value });
        }
    }
    Ok(())
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
    for (i, omega_row) in omega_jb.iter_mut().enumerate().take(R) {
        for (k, j_b_row) in j_b.iter().enumerate().take(R) {
            for col in 0..S {
                omega_row[col] += info[i][k] * j_b_row[col];
            }
        }
    }
    // J_a^T · (Ω · J_b) at (base_a, base_b) + transpose at (base_b, base_a)
    for row in 0..S {
        for col in 0..S {
            let mut val = 0.0;
            for (k, omega_row) in omega_jb.iter().enumerate().take(R) {
                val += j_a[k][row] * omega_row[col];
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
            ImuBias::try_new([0.1, -0.05, 0.02], [0.001, -0.002, 0.003]).expect("finite bias"),
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
            ImuBias::try_new([0.12, -0.04, 0.03], [0.002, -0.001, 0.004]).expect("finite bias"),
        )
        .expect("state_j");
        let preintegrated = PreintegratedImu::integrate(
            &batch(&[
                (0, [0.5, -0.3, 9.5], [0.1, -0.05, 0.02]),
                (5_000_000, [0.6, -0.2, 9.6], [0.12, -0.04, 0.03]),
                (10_000_000, [0.4, -0.1, 9.4], [0.08, -0.06, 0.01]),
            ]),
            state_i.bias(),
            &noise(),
        )
        .expect("preintegrated");

        let (jp, jc) = imu_jacobians(&state_i, &state_j, &preintegrated, gravity)
            .expect("finite IMU Jacobians");
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

        let (jac_prev, jac_curr) =
            imu_jacobians(&state, &state, &preintegrated, gravity).expect("finite IMU Jacobians");
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

    #[test]
    fn imu_jacobians_propagate_factor_failure_instead_of_zeroing_columns() {
        let stationary = batch(&[(0, [0.0; 3], [0.0; 3]), (10_000_000, [0.0; 3], [0.0; 3])]);
        let preintegrated = PreintegratedImu::integrate(&stationary, &ImuBias::default(), &noise())
            .expect("preintegrated");
        let huge_bias =
            ImuBias::try_new([0.0; 3], [f64::MAX, 0.0, 0.0]).expect("finite but extreme bias");
        let state_i =
            NavState::try_new(Pose64::identity(), [0.0; 3], huge_bias).expect("previous state");
        let state_j =
            NavState::try_new(Pose64::identity(), [0.0; 3], huge_bias).expect("current state");
        let gravity = Gravity::try_new([0.0, 0.0, -9.81]).expect("gravity");

        let error = imu_jacobians(&state_i, &state_j, &preintegrated, gravity)
            .expect_err("overflowed residual must fail the Jacobian");
        assert!(matches!(
            error,
            ImuJacobianError::Residual {
                endpoint: ImuJacobianEndpoint::PreviousState,
                side: FiniteDifferenceSide::Positive,
                tangent_axis: 0,
                source: VioFactorError::PreintegrationCorrection { .. },
            }
        ));
        let factor_source = std::error::Error::source(&error).expect("factor source");
        assert!(factor_source.source().is_some());
    }

    /// Dense solver: trivial 2×2 system.
    #[test]
    fn dense_solver_2x2() {
        // 2x + 3y = 8
        // x + y = 3
        // => x = 1, y = 2
        let mut a = vec![2.0, 3.0, 1.0, 1.0];
        let mut b = vec![8.0, 3.0];
        solve_dense_f64(&mut a, &mut b, 2).expect("nonsingular system");
        assert!((b[0] - 1.0).abs() < 1e-12);
        assert!((b[1] - 2.0).abs() < 1e-12);
    }

    /// Dense solver: singular system returns a truthful breakdown outcome.
    #[test]
    fn dense_solver_singular() {
        let mut a = vec![1.0, 2.0, 2.0, 4.0];
        let mut b = vec![3.0, 6.0];
        assert!(matches!(
            solve_dense_f64(&mut a, &mut b, 2),
            Err(DenseSolveError::SingularOrIllConditioned { column: 1, .. })
        ));
    }

    #[test]
    fn dense_solver_accepts_uniformly_tiny_well_conditioned_system() {
        let scale = 1e-300;
        let mut a = vec![2.0 * scale, 3.0 * scale, scale, scale];
        let mut b = vec![8.0 * scale, 3.0 * scale];
        solve_dense_f64(&mut a, &mut b, 2).expect("scaled nonsingular system");
        assert!((b[0] - 1.0).abs() < 1e-12);
        assert!((b[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn dense_solver_breakdown_is_invariant_to_uniform_scale() {
        for scale in [1e-200, 1.0, 1e200] {
            let mut a = vec![scale, 2.0 * scale, 2.0 * scale, 4.0 * scale];
            let mut b = vec![3.0 * scale, 6.0 * scale];
            assert!(matches!(
                solve_dense_f64(&mut a, &mut b, 2),
                Err(DenseSolveError::SingularOrIllConditioned { column: 1, .. })
            ));
        }
    }

    #[test]
    fn dense_solver_rejects_shape_and_nonfinite_inputs() {
        assert!(matches!(
            solve_dense_f64(&mut [1.0, 0.0, 1.0], &mut [1.0, 2.0], 2),
            Err(DenseSolveError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            solve_dense_f64(&mut [1.0, 0.0, 0.0, f64::NAN], &mut [1.0, 2.0], 2,),
            Err(DenseSolveError::NonFiniteInput {
                input: DenseSolveInput::Matrix,
                index: 3,
                ..
            })
        ));
    }
}
