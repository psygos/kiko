use crate::Pose;

#[derive(Clone, Copy, Debug)]
pub struct Pose64 {
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
}

impl Pose64 {
    pub fn identity() -> Self {
        Self {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
        }
    }

    pub fn compose(self, other: Self) -> Self {
        let rotation = mat_mul_f64(self.rotation, other.rotation);
        let rt = mat_mul_vec_f64(self.rotation, other.translation);
        let translation = [
            rt[0] + self.translation[0],
            rt[1] + self.translation[1],
            rt[2] + self.translation[2],
        ];
        Self {
            rotation,
            translation,
        }
    }

    pub fn inverse(self) -> Self {
        let r_t = mat_transpose_f64(self.rotation);
        let t = mat_mul_vec_f64(r_t, self.translation);
        Self {
            rotation: r_t,
            translation: [-t[0], -t[1], -t[2]],
        }
    }

    pub fn from_pose32(pose: Pose) -> Self {
        let mut rotation = [[0.0_f64; 3]; 3];
        for row in 0..3 {
            for col in 0..3 {
                rotation[row][col] = pose.rotation()[row][col] as f64;
            }
        }
        let t = pose.translation();
        Self {
            rotation,
            translation: [t[0] as f64, t[1] as f64, t[2] as f64],
        }
    }

    pub fn to_pose32(self) -> Pose {
        let mut rotation = [[0.0_f32; 3]; 3];
        for row in 0..3 {
            for col in 0..3 {
                rotation[row][col] = self.rotation[row][col] as f32;
            }
        }
        let translation = [
            self.translation[0] as f32,
            self.translation[1] as f32,
            self.translation[2] as f32,
        ];
        Pose::from_rt(rotation, translation)
    }
}

pub(crate) fn mat_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut r = [[0.0_f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    r
}

pub(crate) fn mat_mul_vec(r: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ]
}

pub(crate) fn mat_mul_f64(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut r = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    r
}

pub(crate) fn mat_mul_vec_f64(r: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ]
}

pub(crate) fn transform_point(r: [[f32; 3]; 3], t: [f32; 3], v: [f32; 3]) -> [f32; 3] {
    let rv = mat_mul_vec(r, v);
    [rv[0] + t[0], rv[1] + t[1], rv[2] + t[2]]
}

pub(crate) fn so3_exp_f64(omega: [f64; 3]) -> [[f64; 3]; 3] {
    let theta = (omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]).sqrt();
    let omega_hat = skew_f64(omega);
    let omega_hat2 = mat_mul_f64(omega_hat, omega_hat);
    let i = identity3_f64();

    if theta < 1e-12 {
        let mut r = i;
        for row in 0..3 {
            for col in 0..3 {
                r[row][col] += omega_hat[row][col] + 0.5 * omega_hat2[row][col];
            }
        }
        return r;
    }

    let a = theta.sin() / theta;
    let b = (1.0 - theta.cos()) / (theta * theta);
    let mut r = i;
    for row in 0..3 {
        for col in 0..3 {
            r[row][col] += a * omega_hat[row][col] + b * omega_hat2[row][col];
        }
    }
    r
}

pub(crate) fn so3_log_f64(r: [[f64; 3]; 3]) -> [f64; 3] {
    let trace = r[0][0] + r[1][1] + r[2][2];
    let cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();

    if theta < 1e-12 {
        return [
            0.5 * (r[2][1] - r[1][2]),
            0.5 * (r[0][2] - r[2][0]),
            0.5 * (r[1][0] - r[0][1]),
        ];
    }

    if (std::f64::consts::PI - theta).abs() < 1e-6 {
        let x = ((r[0][0] + 1.0) * 0.5).max(0.0).sqrt();
        let y = ((r[1][1] + 1.0) * 0.5).max(0.0).sqrt();
        let z = ((r[2][2] + 1.0) * 0.5).max(0.0).sqrt();
        let mut axis = [x, y, z];
        if r[2][1] - r[1][2] < 0.0 {
            axis[0] = -axis[0];
        }
        if r[0][2] - r[2][0] < 0.0 {
            axis[1] = -axis[1];
        }
        if r[1][0] - r[0][1] < 0.0 {
            axis[2] = -axis[2];
        }
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if norm > 1e-12 {
            return [
                theta * axis[0] / norm,
                theta * axis[1] / norm,
                theta * axis[2] / norm,
            ];
        }
    }

    let sin_theta = theta.sin();
    let factor = theta / (2.0 * sin_theta);
    [
        factor * (r[2][1] - r[1][2]),
        factor * (r[0][2] - r[2][0]),
        factor * (r[1][0] - r[0][1]),
    ]
}

pub(crate) fn so3_right_jacobian_f64(omega: [f64; 3]) -> [[f64; 3]; 3] {
    let theta = (omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]).sqrt();
    let omega_hat = skew_f64(omega);
    let omega_hat2 = mat_mul_f64(omega_hat, omega_hat);
    let mut jr = identity3_f64();

    if theta < 1e-9 {
        for row in 0..3 {
            for col in 0..3 {
                jr[row][col] += -0.5 * omega_hat[row][col] + (1.0 / 6.0) * omega_hat2[row][col];
            }
        }
        return jr;
    }

    let theta2 = theta * theta;
    let a = (1.0 - theta.cos()) / theta2;
    let b = (theta - theta.sin()) / (theta2 * theta);
    for row in 0..3 {
        for col in 0..3 {
            jr[row][col] += -a * omega_hat[row][col] + b * omega_hat2[row][col];
        }
    }
    jr
}

pub(crate) fn se3_exp_f64(xi: [f64; 6]) -> Pose64 {
    let rho = [xi[0], xi[1], xi[2]];
    let omega = [xi[3], xi[4], xi[5]];
    let rotation = so3_exp_f64(omega);
    let theta = (omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]).sqrt();
    let omega_hat = skew_f64(omega);
    let omega_hat2 = mat_mul_f64(omega_hat, omega_hat);
    let mut v = identity3_f64();
    if theta < 1e-9 {
        for row in 0..3 {
            for col in 0..3 {
                v[row][col] += 0.5 * omega_hat[row][col] + (1.0 / 6.0) * omega_hat2[row][col];
            }
        }
    } else {
        let theta2 = theta * theta;
        let b = (1.0 - theta.cos()) / theta2;
        let c = (theta - theta.sin()) / (theta2 * theta);
        for row in 0..3 {
            for col in 0..3 {
                v[row][col] += b * omega_hat[row][col] + c * omega_hat2[row][col];
            }
        }
    }
    let translation = mat_mul_vec_f64(v, rho);
    Pose64 {
        rotation,
        translation,
    }
}

pub(crate) fn se3_log_f64(pose: Pose64) -> [f64; 6] {
    let omega = so3_log_f64(pose.rotation);
    let theta = (omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]).sqrt();
    let omega_hat = skew_f64(omega);
    let omega_hat2 = mat_mul_f64(omega_hat, omega_hat);
    let mut v_inv = identity3_f64();
    if theta < 1e-9 {
        for row in 0..3 {
            for col in 0..3 {
                v_inv[row][col] += -0.5 * omega_hat[row][col] + (1.0 / 12.0) * omega_hat2[row][col];
            }
        }
    } else {
        let theta2 = theta * theta;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();
        let a = (1.0 / theta2) * (1.0 - (theta * sin_theta) / (2.0 * (1.0 - cos_theta)));
        for row in 0..3 {
            for col in 0..3 {
                v_inv[row][col] += -0.5 * omega_hat[row][col] + a * omega_hat2[row][col];
            }
        }
    }
    let rho = mat_mul_vec_f64(v_inv, pose.translation);
    [rho[0], rho[1], rho[2], omega[0], omega[1], omega[2]]
}

fn identity3_f64() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn skew_f64(v: [f64; 3]) -> [[f64; 3]; 3] {
    [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
}

fn mat_transpose_f64(r: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [r[0][0], r[1][0], r[2][0]],
        [r[0][1], r[1][1], r[2][1]],
        [r[0][2], r[1][2], r[2][2]],
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        mat_mul_f64, mat_mul_vec_f64, se3_exp_f64, se3_log_f64, so3_exp_f64, so3_log_f64,
        so3_right_jacobian_f64,
    };

    fn rot_diff_norm(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> f64 {
        let mut sum = 0.0;
        for row in 0..3 {
            for col in 0..3 {
                let d = a[row][col] - b[row][col];
                sum += d * d;
            }
        }
        sum.sqrt()
    }

    fn vec_norm(v: [f64; 3]) -> f64 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    #[test]
    fn so3_exp_log_round_trip_f64() {
        let omega = [0.2, -0.1, 0.07];
        let r = so3_exp_f64(omega);
        let recovered = so3_log_f64(r);
        let err = vec_norm([
            recovered[0] - omega[0],
            recovered[1] - omega[1],
            recovered[2] - omega[2],
        ]);
        assert!(err < 1e-9, "so3 round-trip error: {err}");
    }

    #[test]
    fn se3_exp_log_round_trip_f64() {
        let xi = [0.05, -0.03, 0.02, 0.1, -0.04, 0.03];
        let pose = se3_exp_f64(xi);
        let recovered = se3_log_f64(pose);
        let mut err = 0.0;
        for i in 0..6 {
            let d = recovered[i] - xi[i];
            err += d * d;
        }
        assert!(err.sqrt() < 1e-9, "se3 round-trip error: {}", err.sqrt());
    }

    #[test]
    fn so3_right_jacobian_matches_finite_diff() {
        let omega = [0.2, -0.05, 0.1];
        let delta = [1e-6, -2e-6, 1.5e-6];
        let jr_delta = mat_mul_vec_f64(so3_right_jacobian_f64(omega), delta);
        let r_fd = so3_exp_f64([omega[0] + delta[0], omega[1] + delta[1], omega[2] + delta[2]]);
        let r_pred = mat_mul_f64(so3_exp_f64(omega), so3_exp_f64(jr_delta));
        let err = rot_diff_norm(r_fd, r_pred);
        assert!(err < 1e-8, "right jacobian finite-diff error: {err}");
    }

    #[test]
    fn so3_log_is_finite_near_pi_f64() {
        let theta = std::f64::consts::PI - 1e-6;
        let r = so3_exp_f64([0.0, theta, 0.0]);
        let recovered = so3_log_f64(r);
        assert!(recovered.iter().all(|v| v.is_finite()));
        let recovered_theta = vec_norm(recovered);
        assert!(
            (recovered_theta - theta).abs() < 2e-4,
            "near-pi mismatch: recovered={recovered_theta}, expected={theta}"
        );
    }
}
