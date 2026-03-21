use crate::math::{mat_mul_f64, mat_mul_vec_f64, so3_exp_f64, so3_log_f64};
use crate::{ImuBatch, ImuBias, ImuNoiseModel, ImuSample};

const BIAS_FD_EPS: f64 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PreintegrationError {
    TooFewSamples { len: usize },
    NonPositiveDeltaTime { dt_seconds: f64 },
}

impl std::fmt::Display for PreintegrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PreintegrationError::TooFewSamples { len } => {
                write!(
                    f,
                    "imu preintegration requires at least 2 samples, got {len}"
                )
            }
            PreintegrationError::NonPositiveDeltaTime { dt_seconds } => {
                write!(f, "imu sample dt must be > 0, got {dt_seconds}")
            }
        }
    }
}

impl std::error::Error for PreintegrationError {}

#[derive(Clone, Debug)]
pub struct CorrectedPreintegration {
    pub delta_rotation: [[f64; 3]; 3],
    pub delta_velocity: [f64; 3],
    pub delta_position: [f64; 3],
}

#[derive(Clone, Debug)]
pub struct PreintegratedImu {
    batch: ImuBatch,
    noise: ImuNoiseModel,
    bias_linearization: ImuBias,
    pub delta_rotation: [[f64; 3]; 3],
    pub delta_velocity: [f64; 3],
    pub delta_position: [f64; 3],
    pub dt_seconds: f64,
    pub covariance: [[f64; 9]; 9],
    pub d_rotation_d_gyro_bias: [[f64; 3]; 3],
    pub d_velocity_d_accel_bias: [[f64; 3]; 3],
    pub d_velocity_d_gyro_bias: [[f64; 3]; 3],
    pub d_position_d_accel_bias: [[f64; 3]; 3],
    pub d_position_d_gyro_bias: [[f64; 3]; 3],
}

impl PreintegratedImu {
    pub fn integrate(
        batch: &ImuBatch,
        bias: &ImuBias,
        noise: &ImuNoiseModel,
    ) -> Result<Self, PreintegrationError> {
        let core = integrate_core(batch.samples(), bias, noise)?;

        let d_rotation_d_gyro_bias = finite_difference_gyro_jacobian(batch, bias, noise, |core| {
            so3_log_f64(core.delta_rotation)
        })?;
        let d_velocity_d_gyro_bias =
            finite_difference_gyro_jacobian(batch, bias, noise, |core| core.delta_velocity)?;
        let d_position_d_gyro_bias =
            finite_difference_gyro_jacobian(batch, bias, noise, |core| core.delta_position)?;
        let d_velocity_d_accel_bias =
            finite_difference_accel_jacobian(batch, bias, noise, |core| core.delta_velocity)?;
        let d_position_d_accel_bias =
            finite_difference_accel_jacobian(batch, bias, noise, |core| core.delta_position)?;

        Ok(Self {
            batch: batch.clone(),
            noise: noise.clone(),
            bias_linearization: bias.clone(),
            delta_rotation: core.delta_rotation,
            delta_velocity: core.delta_velocity,
            delta_position: core.delta_position,
            dt_seconds: core.dt_seconds,
            covariance: core.covariance,
            d_rotation_d_gyro_bias,
            d_velocity_d_accel_bias,
            d_velocity_d_gyro_bias,
            d_position_d_accel_bias,
            d_position_d_gyro_bias,
        })
    }

    pub fn bias_linearization(&self) -> &ImuBias {
        &self.bias_linearization
    }

    pub fn corrected_first_order(&self, new_bias: &ImuBias) -> CorrectedPreintegration {
        let delta_accel = sub_vec3(new_bias.accel, self.bias_linearization.accel);
        let delta_gyro = sub_vec3(new_bias.gyro, self.bias_linearization.gyro);
        let rotation_delta = mat_mul_vec_f64(self.d_rotation_d_gyro_bias, delta_gyro);
        let rotation = mat_mul_f64(so3_exp_f64(rotation_delta), self.delta_rotation);
        let velocity = add_vec3(
            self.delta_velocity,
            add_vec3(
                mat_mul_vec_f64(self.d_velocity_d_accel_bias, delta_accel),
                mat_mul_vec_f64(self.d_velocity_d_gyro_bias, delta_gyro),
            ),
        );
        let position = add_vec3(
            self.delta_position,
            add_vec3(
                mat_mul_vec_f64(self.d_position_d_accel_bias, delta_accel),
                mat_mul_vec_f64(self.d_position_d_gyro_bias, delta_gyro),
            ),
        );
        CorrectedPreintegration {
            delta_rotation: rotation,
            delta_velocity: velocity,
            delta_position: position,
        }
    }

    pub fn reintegrate_exact(
        &self,
        new_bias: &ImuBias,
    ) -> Result<CorrectedPreintegration, PreintegrationError> {
        let core = integrate_core(self.batch.samples(), new_bias, &self.noise)?;
        Ok(CorrectedPreintegration {
            delta_rotation: core.delta_rotation,
            delta_velocity: core.delta_velocity,
            delta_position: core.delta_position,
        })
    }

    pub fn residual_information_diag(&self) -> [f64; 9] {
        let information = self.residual_information();
        let mut diagonal = [0.0_f64; 9];
        for axis in 0..9 {
            diagonal[axis] = information[axis][axis];
        }
        diagonal
    }

    pub fn residual_information(&self) -> [[f64; 9]; 9] {
        invert_stabilized_covariance::<9>(self.covariance, 1e-6)
    }

    pub fn bias_random_walk_information_diag(&self) -> [f64; 6] {
        let information = self.bias_random_walk_information();
        let mut diagonal = [0.0_f64; 6];
        for axis in 0..6 {
            diagonal[axis] = information[axis][axis];
        }
        diagonal
    }

    pub fn bias_random_walk_information(&self) -> [[f64; 6]; 6] {
        // Keep bias random-walk regularization comparable to the other VIO terms.
        // Without this floor, short IMU intervals produce enormous information that
        // effectively hard-locks the initial bias estimate.
        const MIN_VARIANCE: f64 = 1e-6;
        let dt = self.dt_seconds.max(1e-12);
        let accel_variance = self.noise.accel_random_walk() * self.noise.accel_random_walk() * dt;
        let gyro_variance = self.noise.gyro_random_walk() * self.noise.gyro_random_walk() * dt;
        let mut information = [[0.0_f64; 6]; 6];
        for axis in 0..3 {
            information[axis][axis] = 1.0 / accel_variance.max(MIN_VARIANCE);
            information[3 + axis][3 + axis] = 1.0 / gyro_variance.max(MIN_VARIANCE);
        }
        information
    }
}

struct CorePreintegration {
    delta_rotation: [[f64; 3]; 3],
    delta_velocity: [f64; 3],
    delta_position: [f64; 3],
    dt_seconds: f64,
    covariance: [[f64; 9]; 9],
}

fn integrate_core(
    samples: &[ImuSample],
    bias: &ImuBias,
    noise: &ImuNoiseModel,
) -> Result<CorePreintegration, PreintegrationError> {
    if samples.len() < 2 {
        return Err(PreintegrationError::TooFewSamples { len: samples.len() });
    }

    let mut delta_rotation = identity3();
    let mut delta_velocity = [0.0_f64; 3];
    let mut delta_position = [0.0_f64; 3];
    let mut dt_total = 0.0_f64;
    let mut covariance = [[0.0_f64; 9]; 9];

    for pair in samples.windows(2) {
        let a = &pair[0];
        let b = &pair[1];
        let dt = b.timestamp().seconds_since(a.timestamp());
        if !dt.is_finite() || dt <= 0.0 {
            return Err(PreintegrationError::NonPositiveDeltaTime { dt_seconds: dt });
        }

        let gyro_a = sub_vec3(a.gyro_radps(), bias.gyro);
        let gyro_b = sub_vec3(b.gyro_radps(), bias.gyro);
        let omega_mid = scale_vec3(add_vec3(gyro_a, gyro_b), 0.5);

        let accel_a = sub_vec3(a.accel_mps2(), bias.accel);
        let accel_b = sub_vec3(b.accel_mps2(), bias.accel);

        let delta_rotation_next =
            mat_mul_f64(delta_rotation, so3_exp_f64(scale_vec3(omega_mid, dt)));
        let accel_mid = scale_vec3(
            add_vec3(
                mat_mul_vec_f64(delta_rotation, accel_a),
                mat_mul_vec_f64(delta_rotation_next, accel_b),
            ),
            0.5,
        );

        delta_position = add_vec3(
            delta_position,
            add_vec3(
                scale_vec3(delta_velocity, dt),
                scale_vec3(accel_mid, 0.5 * dt * dt),
            ),
        );
        delta_velocity = add_vec3(delta_velocity, scale_vec3(accel_mid, dt));
        delta_rotation = delta_rotation_next;
        dt_total += dt;

        let f = error_state_transition(omega_mid, accel_mid, dt);
        let g = noise_injection(delta_rotation, dt);
        let mut q = [[0.0_f64; 6]; 6];
        let gyro_variance = noise.gyro_noise_density() * noise.gyro_noise_density();
        let accel_variance = noise.accel_noise_density() * noise.accel_noise_density();
        for axis in 0..3 {
            q[axis][axis] = gyro_variance;
            q[3 + axis][3 + axis] = accel_variance;
        }
        covariance = propagate_covariance(covariance, f, g, q);
    }

    Ok(CorePreintegration {
        delta_rotation,
        delta_velocity,
        delta_position,
        dt_seconds: dt_total,
        covariance,
    })
}

fn finite_difference_gyro_jacobian(
    batch: &ImuBatch,
    bias: &ImuBias,
    noise: &ImuNoiseModel,
    project: impl Fn(CorePreintegration) -> [f64; 3],
) -> Result<[[f64; 3]; 3], PreintegrationError> {
    let mut jacobian = [[0.0_f64; 3]; 3];
    for axis in 0..3 {
        let mut plus = bias.clone();
        let mut minus = bias.clone();
        plus.gyro[axis] += BIAS_FD_EPS;
        minus.gyro[axis] -= BIAS_FD_EPS;
        let plus_projected = project(integrate_core(batch.samples(), &plus, noise)?);
        let minus_projected = project(integrate_core(batch.samples(), &minus, noise)?);
        for row in 0..3 {
            jacobian[row][axis] =
                (plus_projected[row] - minus_projected[row]) / (2.0 * BIAS_FD_EPS);
        }
    }
    Ok(jacobian)
}

fn finite_difference_accel_jacobian(
    batch: &ImuBatch,
    bias: &ImuBias,
    noise: &ImuNoiseModel,
    project: impl Fn(CorePreintegration) -> [f64; 3],
) -> Result<[[f64; 3]; 3], PreintegrationError> {
    let mut jacobian = [[0.0_f64; 3]; 3];
    for axis in 0..3 {
        let mut plus = bias.clone();
        let mut minus = bias.clone();
        plus.accel[axis] += BIAS_FD_EPS;
        minus.accel[axis] -= BIAS_FD_EPS;
        let plus_projected = project(integrate_core(batch.samples(), &plus, noise)?);
        let minus_projected = project(integrate_core(batch.samples(), &minus, noise)?);
        for row in 0..3 {
            jacobian[row][axis] =
                (plus_projected[row] - minus_projected[row]) / (2.0 * BIAS_FD_EPS);
        }
    }
    Ok(jacobian)
}

fn identity3() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn add_vec3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub_vec3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale_vec3(v: [f64; 3], scale: f64) -> [f64; 3] {
    [v[0] * scale, v[1] * scale, v[2] * scale]
}

fn error_state_transition(omega_mid: [f64; 3], accel_mid: [f64; 3], dt: f64) -> [[f64; 9]; 9] {
    let mut f = identity9();
    let omega_hat = skew3(omega_mid);
    let accel_hat = skew3(accel_mid);
    for row in 0..3 {
        for col in 0..3 {
            f[row][col] -= omega_hat[row][col] * dt;
            f[3 + row][col] -= accel_hat[row][col] * dt;
            f[6 + row][col] -= accel_hat[row][col] * (0.5 * dt * dt);
        }
        f[6 + row][3 + row] = dt;
    }
    f
}

fn noise_injection(delta_rotation: [[f64; 3]; 3], dt: f64) -> [[f64; 6]; 9] {
    let mut g = [[0.0_f64; 6]; 9];
    for row in 0..3 {
        g[row][row] = dt;
    }
    for row in 0..3 {
        for col in 0..3 {
            g[3 + row][3 + col] = delta_rotation[row][col] * dt;
            g[6 + row][3 + col] = delta_rotation[row][col] * (0.5 * dt * dt);
        }
    }
    g
}

fn propagate_covariance(
    covariance: [[f64; 9]; 9],
    f: [[f64; 9]; 9],
    g: [[f64; 6]; 9],
    q: [[f64; 6]; 6],
) -> [[f64; 9]; 9] {
    let ft = transpose9(f);
    let gt = transpose96(g);
    let predicted = matmul9(matmul9(f, covariance), ft);
    let process = matmul9x6_6x9(matmul9x6_6x6(g, q), gt);
    symmetrize9(add9(predicted, process))
}

fn invert_stabilized_covariance<const N: usize>(
    mut covariance: [[f64; N]; N],
    min_variance: f64,
) -> [[f64; N]; N] {
    for axis in 0..N {
        covariance[axis][axis] = covariance[axis][axis].max(min_variance);
    }
    invert_square(covariance).unwrap_or_else(|| {
        let mut fallback = [[0.0_f64; N]; N];
        for axis in 0..N {
            fallback[axis][axis] = 1.0 / covariance[axis][axis].max(min_variance);
        }
        fallback
    })
}

fn invert_square<const N: usize>(matrix: [[f64; N]; N]) -> Option<[[f64; N]; N]> {
    let mut a = matrix;
    let mut inv = [[0.0_f64; N]; N];
    for idx in 0..N {
        inv[idx][idx] = 1.0;
    }
    for pivot in 0..N {
        let mut pivot_row = pivot;
        let mut pivot_abs = a[pivot][pivot].abs();
        for (row, a_row) in a.iter().enumerate().take(N).skip(pivot + 1) {
            let value = a_row[pivot].abs();
            if value > pivot_abs {
                pivot_abs = value;
                pivot_row = row;
            }
        }
        if !pivot_abs.is_finite() || pivot_abs < 1e-12 {
            return None;
        }
        if pivot_row != pivot {
            a.swap(pivot, pivot_row);
            inv.swap(pivot, pivot_row);
        }
        let diag = a[pivot][pivot];
        for col in 0..N {
            a[pivot][col] /= diag;
            inv[pivot][col] /= diag;
        }
        for row in 0..N {
            if row == pivot {
                continue;
            }
            let factor = a[row][pivot];
            if factor == 0.0 {
                continue;
            }
            for col in 0..N {
                a[row][col] -= factor * a[pivot][col];
                inv[row][col] -= factor * inv[pivot][col];
            }
        }
    }
    Some(inv)
}

fn add9(a: [[f64; 9]; 9], b: [[f64; 9]; 9]) -> [[f64; 9]; 9] {
    let mut out = [[0.0_f64; 9]; 9];
    for row in 0..9 {
        for col in 0..9 {
            out[row][col] = a[row][col] + b[row][col];
        }
    }
    out
}

fn identity9() -> [[f64; 9]; 9] {
    let mut out = [[0.0_f64; 9]; 9];
    for (idx, row) in out.iter_mut().enumerate() {
        row[idx] = 1.0;
    }
    out
}

fn matmul9(a: [[f64; 9]; 9], b: [[f64; 9]; 9]) -> [[f64; 9]; 9] {
    let mut out = [[0.0_f64; 9]; 9];
    for i in 0..9 {
        for j in 0..9 {
            let mut value = 0.0_f64;
            for (k, row) in b.iter().enumerate() {
                value += a[i][k] * row[j];
            }
            out[i][j] = value;
        }
    }
    out
}

fn matmul9x6_6x6(a: [[f64; 6]; 9], b: [[f64; 6]; 6]) -> [[f64; 6]; 9] {
    let mut out = [[0.0_f64; 6]; 9];
    for i in 0..9 {
        for j in 0..6 {
            let mut value = 0.0_f64;
            for (k, row) in b.iter().enumerate() {
                value += a[i][k] * row[j];
            }
            out[i][j] = value;
        }
    }
    out
}

fn matmul9x6_6x9(a: [[f64; 6]; 9], b: [[f64; 9]; 6]) -> [[f64; 9]; 9] {
    let mut out = [[0.0_f64; 9]; 9];
    for i in 0..9 {
        for j in 0..9 {
            let mut value = 0.0_f64;
            for (k, row) in b.iter().enumerate() {
                value += a[i][k] * row[j];
            }
            out[i][j] = value;
        }
    }
    out
}

fn skew3(v: [f64; 3]) -> [[f64; 3]; 3] {
    [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
}

fn symmetrize9(matrix: [[f64; 9]; 9]) -> [[f64; 9]; 9] {
    let mut out = matrix;
    for row in 0..9 {
        for col in (row + 1)..9 {
            let value = 0.5 * (matrix[row][col] + matrix[col][row]);
            out[row][col] = value;
            out[col][row] = value;
        }
    }
    out
}

fn transpose9(matrix: [[f64; 9]; 9]) -> [[f64; 9]; 9] {
    let mut out = [[0.0_f64; 9]; 9];
    for row in 0..9 {
        for col in 0..9 {
            out[col][row] = matrix[row][col];
        }
    }
    out
}

fn transpose96(matrix: [[f64; 6]; 9]) -> [[f64; 9]; 6] {
    let mut out = [[0.0_f64; 9]; 6];
    for row in 0..9 {
        for col in 0..6 {
            out[col][row] = matrix[row][col];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise() -> ImuNoiseModel {
        ImuNoiseModel::new(0.1, 0.01, 0.001, 0.0001).expect("noise")
    }

    fn batch(samples: &[(i64, [f64; 3], [f64; 3])]) -> ImuBatch {
        ImuBatch::new(
            samples
                .iter()
                .map(|(timestamp, accel, gyro)| {
                    ImuSample::new(crate::Timestamp::from_nanos(*timestamp), *accel, *gyro)
                        .expect("imu sample")
                })
                .collect(),
        )
        .expect("imu batch")
    }

    fn matrix_trace(matrix: [[f64; 3]; 3]) -> f64 {
        matrix[0][0] + matrix[1][1] + matrix[2][2]
    }

    #[test]
    fn integrate_rejects_too_few_samples() {
        let batch = batch(&[(0, [0.0; 3], [0.0; 3])]);
        let err = PreintegratedImu::integrate(&batch, &ImuBias::default(), &noise())
            .expect_err("single sample should fail");
        assert_eq!(err, PreintegrationError::TooFewSamples { len: 1 });
    }

    #[test]
    fn zero_specific_force_and_zero_gyro_stays_identity() {
        let batch = batch(&[
            (0, [0.0; 3], [0.0; 3]),
            (10_000_000, [0.0; 3], [0.0; 3]),
            (20_000_000, [0.0; 3], [0.0; 3]),
        ]);
        let preintegrated = PreintegratedImu::integrate(&batch, &ImuBias::default(), &noise())
            .expect("preintegrated");
        assert_eq!(preintegrated.delta_rotation, identity3());
        assert_eq!(preintegrated.delta_velocity, [0.0; 3]);
        assert_eq!(preintegrated.delta_position, [0.0; 3]);
        assert!((preintegrated.dt_seconds - 0.02).abs() < 1e-12);
    }

    #[test]
    fn constant_rotation_matches_so3_exponential() {
        let batch = batch(&[
            (0, [0.0; 3], [0.0, 0.0, 0.5]),
            (10_000_000, [0.0; 3], [0.0, 0.0, 0.5]),
            (20_000_000, [0.0; 3], [0.0, 0.0, 0.5]),
        ]);
        let preintegrated = PreintegratedImu::integrate(&batch, &ImuBias::default(), &noise())
            .expect("preintegrated");
        let expected = so3_exp_f64([0.0, 0.0, 0.01]);
        let error = so3_log_f64(mat_mul_f64(
            preintegrated.delta_rotation,
            mat_transpose(expected),
        ));
        let norm = (error[0] * error[0] + error[1] * error[1] + error[2] * error[2]).sqrt();
        assert!(norm < 1e-9, "rotation error={norm}");
    }

    #[test]
    fn covariance_is_symmetric_with_non_negative_diagonal() {
        let batch = batch(&[
            (0, [0.1, 0.0, 0.0], [0.0, 0.0, 0.1]),
            (10_000_000, [0.1, 0.0, 0.0], [0.0, 0.0, 0.1]),
            (20_000_000, [0.1, 0.0, 0.0], [0.0, 0.0, 0.1]),
        ]);
        let preintegrated = PreintegratedImu::integrate(&batch, &ImuBias::default(), &noise())
            .expect("preintegrated");
        for row in 0..9 {
            assert!(preintegrated.covariance[row][row] >= 0.0);
            for col in 0..9 {
                assert!(
                    (preintegrated.covariance[row][col] - preintegrated.covariance[col][row]).abs()
                        < 1e-12
                );
            }
        }
    }

    #[test]
    fn residual_information_diagonal_is_finite_and_positive() {
        let batch = batch(&[
            (0, [0.1, 0.0, 0.0], [0.0, 0.0, 0.1]),
            (10_000_000, [0.1, 0.0, 0.0], [0.0, 0.0, 0.1]),
            (20_000_000, [0.1, 0.0, 0.0], [0.0, 0.0, 0.1]),
        ]);
        let preintegrated = PreintegratedImu::integrate(&batch, &ImuBias::default(), &noise())
            .expect("preintegrated");
        for value in preintegrated.residual_information_diag() {
            assert!(value.is_finite() && value > 0.0);
        }
    }

    #[test]
    fn bias_random_walk_information_decreases_with_longer_dt() {
        let short = PreintegratedImu::integrate(
            &batch(&[(0, [0.0; 3], [0.0; 3]), (10_000_000, [0.0; 3], [0.0; 3])]),
            &ImuBias::default(),
            &noise(),
        )
        .expect("short");
        let long = PreintegratedImu::integrate(
            &batch(&[(0, [0.0; 3], [0.0; 3]), (20_000_000, [0.0; 3], [0.0; 3])]),
            &ImuBias::default(),
            &noise(),
        )
        .expect("long");
        let short_info = short.bias_random_walk_information_diag();
        let long_info = long.bias_random_walk_information_diag();
        for axis in 0..6 {
            assert!(short_info[axis] >= long_info[axis]);
        }
    }

    #[test]
    fn bias_random_walk_information_is_capped() {
        let preintegrated = PreintegratedImu::integrate(
            &batch(&[(0, [0.0; 3], [0.0; 3]), (10_000_000, [0.0; 3], [0.0; 3])]),
            &ImuBias::default(),
            &noise(),
        )
        .expect("preintegrated");
        for value in preintegrated.bias_random_walk_information_diag() {
            assert!(value <= 1.0e6);
            assert!(value.is_finite() && value > 0.0);
        }
    }

    #[test]
    fn corrected_first_order_stays_close_to_exact_reintegration() {
        let batch = batch(&[
            (0, [0.2, 0.1, 0.0], [0.0, 0.0, 0.5]),
            (10_000_000, [0.2, 0.1, 0.0], [0.0, 0.0, 0.5]),
            (20_000_000, [0.2, 0.1, 0.0], [0.0, 0.0, 0.5]),
        ]);
        let base_bias = ImuBias::default();
        let preintegrated =
            PreintegratedImu::integrate(&batch, &base_bias, &noise()).expect("preintegrated");
        let new_bias = ImuBias {
            accel: [1e-4, -2e-4, 3e-4],
            gyro: [-1e-4, 2e-4, -1e-4],
        };
        let corrected = preintegrated.corrected_first_order(&new_bias);
        let exact = preintegrated
            .reintegrate_exact(&new_bias)
            .expect("exact reintegration");

        let rot_error = so3_log_f64(mat_mul_f64(
            corrected.delta_rotation,
            mat_transpose(exact.delta_rotation),
        ));
        let rot_norm = (rot_error[0] * rot_error[0]
            + rot_error[1] * rot_error[1]
            + rot_error[2] * rot_error[2])
            .sqrt();
        let vel_error = sub_vec3(corrected.delta_velocity, exact.delta_velocity);
        let pos_error = sub_vec3(corrected.delta_position, exact.delta_position);
        let vel_norm = (vel_error[0] * vel_error[0]
            + vel_error[1] * vel_error[1]
            + vel_error[2] * vel_error[2])
            .sqrt();
        let pos_norm = (pos_error[0] * pos_error[0]
            + pos_error[1] * pos_error[1]
            + pos_error[2] * pos_error[2])
            .sqrt();
        assert!(rot_norm < 1e-5, "rotation first-order error={rot_norm}");
        assert!(vel_norm < 1e-6, "velocity first-order error={vel_norm}");
        assert!(pos_norm < 1e-6, "position first-order error={pos_norm}");
    }

    #[test]
    fn rotation_matrix_stays_on_so3() {
        let batch = batch(&[
            (0, [0.0; 3], [0.1, -0.2, 0.3]),
            (10_000_000, [0.0; 3], [0.1, -0.2, 0.3]),
            (20_000_000, [0.0; 3], [0.1, -0.2, 0.3]),
        ]);
        let preintegrated = PreintegratedImu::integrate(&batch, &ImuBias::default(), &noise())
            .expect("preintegrated");
        let rot = preintegrated.delta_rotation;
        let rt_r = mat_mul_f64(mat_transpose(rot), rot);
        let identity = identity3();
        for row in 0..3 {
            for col in 0..3 {
                assert!((rt_r[row][col] - identity[row][col]).abs() < 1e-10);
            }
        }
        let trace = matrix_trace(rot);
        assert!(trace.is_finite());
    }

    #[test]
    fn coupled_covariance_produces_cross_terms() {
        let batch = batch(&[
            (0, [0.2, 0.1, -0.05], [0.1, -0.2, 0.3]),
            (10_000_000, [0.25, 0.08, -0.04], [0.12, -0.18, 0.28]),
            (20_000_000, [0.3, 0.05, -0.02], [0.15, -0.15, 0.25]),
            (30_000_000, [0.28, 0.02, 0.0], [0.17, -0.1, 0.2]),
        ]);
        let preintegrated = PreintegratedImu::integrate(&batch, &ImuBias::default(), &noise())
            .expect("preintegrated");
        let mut max_off_diagonal = 0.0_f64;
        for row in 0..9 {
            for col in 0..9 {
                if row != col {
                    max_off_diagonal =
                        max_off_diagonal.max(preintegrated.covariance[row][col].abs());
                }
            }
        }
        assert!(max_off_diagonal > 0.0, "cross terms should be present");
    }

    #[test]
    fn residual_information_matrix_is_symmetric_and_positive_on_diagonal() {
        let batch = batch(&[
            (0, [0.1, -0.1, 0.2], [0.0, 0.05, -0.1]),
            (10_000_000, [0.1, -0.1, 0.2], [0.0, 0.05, -0.1]),
            (20_000_000, [0.1, -0.1, 0.2], [0.0, 0.05, -0.1]),
        ]);
        let preintegrated = PreintegratedImu::integrate(&batch, &ImuBias::default(), &noise())
            .expect("preintegrated");
        let information = preintegrated.residual_information();
        for row in 0..9 {
            assert!(information[row][row].is_finite() && information[row][row] > 0.0);
            for col in 0..9 {
                assert!(
                    (information[row][col] - information[col][row]).abs() < 1e-8,
                    "row={row} col={col}"
                );
            }
        }
    }

    fn mat_transpose(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
        [
            [matrix[0][0], matrix[1][0], matrix[2][0]],
            [matrix[0][1], matrix[1][1], matrix[2][1]],
            [matrix[0][2], matrix[1][2], matrix[2][2]],
        ]
    }
}
