use crate::Pose;

/// Small-angle threshold for first-order Taylor expansion of SO(3) exponential/log maps.
const SO3_SMALL_ANGLE: f64 = 1e-12;
/// Threshold for detecting near-pi rotations in SO(3) log map.
const SO3_NEAR_PI: f64 = 1e-6;
/// Minimum dominant-axis component for stable near-pi axis extraction.
const SO3_AXIS_COMPONENT_MIN: f64 = 1e-12;
/// Minimum axis norm for valid axis extraction in near-pi log map.
const SO3_AXIS_NORM_MIN: f64 = 1e-12;
/// Small-angle threshold for f32 SO(3) operations.
#[cfg(test)]
const SO3_SMALL_ANGLE_F32: f32 = 1e-6;
/// Near-pi threshold for f32 SO(3) log map.
#[cfg(test)]
const SO3_NEAR_PI_F32: f32 = 1e-3;
/// Minimum axis component magnitude for f32 near-pi axis extraction.
#[cfg(test)]
const SO3_AXIS_COMPONENT_MIN_F32: f32 = 1e-6;
/// Minimum axis norm for valid f32 axis normalization.
#[cfg(test)]
const SO3_AXIS_NORM_MIN_F32: f32 = 1e-8;
/// Small-angle threshold below which closed-form Jacobian coefficients lose precision.
const JACOBIAN_SMALL_ANGLE: f64 = 1e-4;
const ROTATION_VALIDATION_TOLERANCE: f64 = 1e-6;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose64 {
    rotation: [[f64; 3]; 3],
    translation: [f64; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pose64Error {
    NonFiniteRotation { row: usize, col: usize, value: f64 },
    NonFiniteTranslation { axis: usize, value: f64 },
    RotationNotOrthonormal { max_error: f64 },
    ImproperRotation { determinant: f64 },
    TranslationOutOfF32Range { axis: usize, value: f64 },
}

impl std::fmt::Display for Pose64Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Pose64Error::NonFiniteRotation { row, col, value } => {
                write!(f, "rotation[{row}][{col}] must be finite, got {value}")
            }
            Pose64Error::NonFiniteTranslation { axis, value } => {
                write!(f, "translation axis {axis} must be finite, got {value}")
            }
            Pose64Error::RotationNotOrthonormal { max_error } => write!(
                f,
                "rotation must be orthonormal within {ROTATION_VALIDATION_TOLERANCE}, max error is {max_error}"
            ),
            Pose64Error::ImproperRotation { determinant } => write!(
                f,
                "rotation determinant must be +1 within {ROTATION_VALIDATION_TOLERANCE}, got {determinant}"
            ),
            Pose64Error::TranslationOutOfF32Range { axis, value } => write!(
                f,
                "translation axis {axis} cannot be represented as finite f32, got {value}"
            ),
        }
    }
}

impl std::error::Error for Pose64Error {}

impl Pose64 {
    pub fn try_from_rt(
        rotation: [[f64; 3]; 3],
        translation: [f64; 3],
    ) -> Result<Self, Pose64Error> {
        for (row_idx, row) in rotation.iter().enumerate() {
            for (col_idx, value) in row.iter().copied().enumerate() {
                if !value.is_finite() {
                    return Err(Pose64Error::NonFiniteRotation {
                        row: row_idx,
                        col: col_idx,
                        value,
                    });
                }
            }
        }
        for (axis, value) in translation.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(Pose64Error::NonFiniteTranslation { axis, value });
            }
        }

        let mut max_error = 0.0_f64;
        for row in 0..3 {
            for col in 0..3 {
                let dot = rotation[row][0] * rotation[col][0]
                    + rotation[row][1] * rotation[col][1]
                    + rotation[row][2] * rotation[col][2];
                let expected = if row == col { 1.0 } else { 0.0 };
                max_error = max_error.max((dot - expected).abs());
            }
        }
        if max_error > ROTATION_VALIDATION_TOLERANCE {
            return Err(Pose64Error::RotationNotOrthonormal { max_error });
        }
        let determinant = rotation[0][0]
            * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
            - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
            + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0]);
        if (determinant - 1.0).abs() > ROTATION_VALIDATION_TOLERANCE {
            return Err(Pose64Error::ImproperRotation { determinant });
        }
        Ok(Self {
            rotation,
            translation,
        })
    }

    #[cfg(test)]
    pub(crate) fn from_rt(rotation: [[f64; 3]; 3], translation: [f64; 3]) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    pub fn identity() -> Self {
        Self {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
        }
    }

    pub fn rotation(&self) -> [[f64; 3]; 3] {
        self.rotation
    }

    pub fn translation(&self) -> [f64; 3] {
        self.translation
    }

    /// Compose transforms as `self ∘ other`.
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

    pub fn try_from_pose32(pose: Pose) -> Result<Self, Pose64Error> {
        let converted = Self::from_pose32(pose);
        Self::try_from_rt(converted.rotation, converted.translation)
    }

    pub(crate) fn from_pose32(pose: Pose) -> Self {
        let pose_rotation = pose.rotation();
        let mut rotation = [[0.0_f64; 3]; 3];
        for (row_idx, row) in rotation.iter_mut().enumerate() {
            for (col_idx, value) in row.iter_mut().enumerate() {
                *value = pose_rotation[row_idx][col_idx] as f64;
            }
        }
        let t = pose.translation();
        Self {
            rotation,
            translation: [t[0] as f64, t[1] as f64, t[2] as f64],
        }
    }

    pub fn try_to_pose32(self) -> Result<Pose, Pose64Error> {
        for (axis, value) in self.translation.iter().copied().enumerate() {
            if !(value as f32).is_finite() {
                return Err(Pose64Error::TranslationOutOfF32Range { axis, value });
            }
        }
        let pose = self.to_pose32_unchecked();
        Self::try_from_pose32(pose)?;
        Ok(pose)
    }

    #[cfg(test)]
    pub(crate) fn to_pose32(self) -> Pose {
        self.to_pose32_unchecked()
    }

    fn to_pose32_unchecked(self) -> Pose {
        let mut rotation = [[0.0_f32; 3]; 3];
        for (row_idx, row) in rotation.iter_mut().enumerate() {
            for (col_idx, value) in row.iter_mut().enumerate() {
                *value = self.rotation[row_idx][col_idx] as f32;
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

pub(crate) fn mat_transpose(r: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [r[0][0], r[1][0], r[2][0]],
        [r[0][1], r[1][1], r[2][1]],
        [r[0][2], r[1][2], r[2][2]],
    ]
}

pub(crate) fn transform_point(r: [[f32; 3]; 3], t: [f32; 3], v: [f32; 3]) -> [f32; 3] {
    let rv = mat_mul_vec(r, v);
    [rv[0] + t[0], rv[1] + t[1], rv[2] + t[2]]
}

#[cfg(test)]
pub(crate) fn so3_exp(w: [f32; 3]) -> [[f32; 3]; 3] {
    let theta = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
    let mut r = [[0.0_f32; 3]; 3];
    if theta < SO3_SMALL_ANGLE_F32 {
        r[0][0] = 1.0;
        r[1][1] = 1.0;
        r[2][2] = 1.0;
        r[0][1] = -w[2];
        r[0][2] = w[1];
        r[1][0] = w[2];
        r[1][2] = -w[0];
        r[2][0] = -w[1];
        r[2][1] = w[0];
        return r;
    }

    let k = [w[0] / theta, w[1] / theta, w[2] / theta];
    let kx = [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]];

    let sin_t = theta.sin();
    let cos_t = theta.cos();
    let mut kx2 = [[0.0_f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            kx2[i][j] = kx[i][0] * kx[0][j] + kx[i][1] * kx[1][j] + kx[i][2] * kx[2][j];
        }
    }

    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = if i == j { 1.0 } else { 0.0 } + sin_t * kx[i][j] + (1.0 - cos_t) * kx2[i][j];
        }
    }
    r
}

#[cfg(test)]
pub(crate) fn so3_log(r: [[f32; 3]; 3]) -> [f32; 3] {
    let trace = r[0][0] + r[1][1] + r[2][2];
    let cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    if theta < SO3_SMALL_ANGLE_F32 {
        return [
            0.5 * (r[2][1] - r[1][2]),
            0.5 * (r[0][2] - r[2][0]),
            0.5 * (r[1][0] - r[0][1]),
        ];
    }

    // Near pi, theta/sin(theta) becomes numerically unstable. Recover the
    // axis from the diagonal of R (equivalently from R + I) and align the
    // sign with the skew-symmetric part.
    if std::f32::consts::PI - theta < SO3_NEAR_PI_F32 {
        let xx = ((r[0][0] + 1.0) * 0.5).max(0.0).sqrt();
        let yy = ((r[1][1] + 1.0) * 0.5).max(0.0).sqrt();
        let zz = ((r[2][2] + 1.0) * 0.5).max(0.0).sqrt();

        let mut axis = if xx >= yy && xx >= zz && xx > SO3_AXIS_COMPONENT_MIN_F32 {
            [
                xx,
                (r[0][1] + r[1][0]) / (4.0 * xx),
                (r[0][2] + r[2][0]) / (4.0 * xx),
            ]
        } else if yy >= zz && yy > SO3_AXIS_COMPONENT_MIN_F32 {
            [
                (r[0][1] + r[1][0]) / (4.0 * yy),
                yy,
                (r[1][2] + r[2][1]) / (4.0 * yy),
            ]
        } else if zz > SO3_AXIS_COMPONENT_MIN_F32 {
            [
                (r[0][2] + r[2][0]) / (4.0 * zz),
                (r[1][2] + r[2][1]) / (4.0 * zz),
                zz,
            ]
        } else {
            [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]]
        };

        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if norm > SO3_AXIS_NORM_MIN_F32 {
            axis = [axis[0] / norm, axis[1] / norm, axis[2] / norm];
        } else {
            axis = [1.0, 0.0, 0.0];
        }

        let skew = [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]];
        let sign = axis[0] * skew[0] + axis[1] * skew[1] + axis[2] * skew[2];
        if sign < 0.0 {
            axis = [-axis[0], -axis[1], -axis[2]];
        }
        return [axis[0] * theta, axis[1] * theta, axis[2] * theta];
    }

    let sin_theta = theta.sin();
    if sin_theta.abs() < SO3_SMALL_ANGLE_F32 {
        return [
            0.5 * (r[2][1] - r[1][2]),
            0.5 * (r[0][2] - r[2][0]),
            0.5 * (r[1][0] - r[0][1]),
        ];
    }
    let factor = theta / (2.0 * sin_theta);
    [
        factor * (r[2][1] - r[1][2]),
        factor * (r[0][2] - r[2][0]),
        factor * (r[1][0] - r[0][1]),
    ]
}

pub(crate) fn so3_exp_f64(omega: [f64; 3]) -> [[f64; 3]; 3] {
    let theta = (omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]).sqrt();
    let omega_hat = skew_f64(omega);
    let omega_hat2 = mat_mul_f64(omega_hat, omega_hat);
    let i = identity3_f64();

    if theta < SO3_SMALL_ANGLE {
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

    if theta < SO3_SMALL_ANGLE {
        return [
            0.5 * (r[2][1] - r[1][2]),
            0.5 * (r[0][2] - r[2][0]),
            0.5 * (r[1][0] - r[0][1]),
        ];
    }

    if (std::f64::consts::PI - theta).abs() < SO3_NEAR_PI {
        let xx = ((r[0][0] + 1.0) * 0.5).max(0.0).sqrt();
        let yy = ((r[1][1] + 1.0) * 0.5).max(0.0).sqrt();
        let zz = ((r[2][2] + 1.0) * 0.5).max(0.0).sqrt();
        let mut axis = if xx >= yy && xx >= zz && xx > SO3_AXIS_COMPONENT_MIN {
            [
                xx,
                (r[0][1] + r[1][0]) / (4.0 * xx),
                (r[0][2] + r[2][0]) / (4.0 * xx),
            ]
        } else if yy >= zz && yy > SO3_AXIS_COMPONENT_MIN {
            [
                (r[0][1] + r[1][0]) / (4.0 * yy),
                yy,
                (r[1][2] + r[2][1]) / (4.0 * yy),
            ]
        } else if zz > SO3_AXIS_COMPONENT_MIN {
            [
                (r[0][2] + r[2][0]) / (4.0 * zz),
                (r[1][2] + r[2][1]) / (4.0 * zz),
                zz,
            ]
        } else {
            [1.0, 0.0, 0.0]
        };
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if norm > SO3_AXIS_NORM_MIN {
            axis = [axis[0] / norm, axis[1] / norm, axis[2] / norm];
            let skew = [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]];
            if axis[0] * skew[0] + axis[1] * skew[1] + axis[2] * skew[2] < 0.0 {
                axis = [-axis[0], -axis[1], -axis[2]];
            }
            return [theta * axis[0], theta * axis[1], theta * axis[2]];
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

    if theta < JACOBIAN_SMALL_ANGLE {
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

#[allow(dead_code)]
pub(crate) fn so3_left_jacobian_f64(omega: [f64; 3]) -> [[f64; 3]; 3] {
    mat_mul_f64(so3_exp_f64(omega), so3_right_jacobian_f64(omega))
}

#[allow(dead_code)]
pub(crate) fn so3_right_jacobian_inv_f64(omega: [f64; 3]) -> [[f64; 3]; 3] {
    let theta = (omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]).sqrt();
    let omega_hat = skew_f64(omega);
    let omega_hat2 = mat_mul_f64(omega_hat, omega_hat);
    let mut jr_inv = identity3_f64();

    if theta < JACOBIAN_SMALL_ANGLE {
        for row in 0..3 {
            for col in 0..3 {
                jr_inv[row][col] += 0.5 * omega_hat[row][col] + (1.0 / 12.0) * omega_hat2[row][col];
            }
        }
        return jr_inv;
    }

    let theta2 = theta * theta;
    let sin_theta = theta.sin();
    let cos_theta = theta.cos();
    let a = (1.0 / theta2) - (1.0 + cos_theta) / (2.0 * theta * sin_theta);
    for row in 0..3 {
        for col in 0..3 {
            jr_inv[row][col] += 0.5 * omega_hat[row][col] + a * omega_hat2[row][col];
        }
    }
    jr_inv
}

#[allow(dead_code)]
pub(crate) fn so3_left_jacobian_inv_f64(omega: [f64; 3]) -> [[f64; 3]; 3] {
    let theta = (omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]).sqrt();
    let omega_hat = skew_f64(omega);
    let omega_hat2 = mat_mul_f64(omega_hat, omega_hat);
    let mut jl_inv = identity3_f64();

    if theta < JACOBIAN_SMALL_ANGLE {
        for row in 0..3 {
            for col in 0..3 {
                jl_inv[row][col] +=
                    -0.5 * omega_hat[row][col] + (1.0 / 12.0) * omega_hat2[row][col];
            }
        }
        return jl_inv;
    }

    let theta2 = theta * theta;
    let sin_theta = theta.sin();
    let cos_theta = theta.cos();
    let a = (1.0 / theta2) - (1.0 + cos_theta) / (2.0 * theta * sin_theta);
    for row in 0..3 {
        for col in 0..3 {
            jl_inv[row][col] += -0.5 * omega_hat[row][col] + a * omega_hat2[row][col];
        }
    }
    jl_inv
}

#[allow(dead_code)]
pub(crate) fn cholesky_6x6(mat: &[[f64; 6]; 6]) -> Option<[[f64; 6]; 6]> {
    let mut l = [[0.0_f64; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let mut sum = mat[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if !sum.is_finite() || sum <= 0.0 {
                    return None;
                }
                l[i][j] = sum.sqrt();
            } else {
                let denom = l[j][j];
                if !denom.is_finite() || denom <= 0.0 {
                    return None;
                }
                l[i][j] = sum / denom;
            }
        }
    }
    Some(l)
}

#[allow(dead_code)]
pub(crate) fn cholesky_solve_6x6(l: &[[f64; 6]; 6], b: &[f64; 6]) -> [f64; 6] {
    let mut y = [0.0_f64; 6];
    for i in 0..6 {
        let mut sum = b[i];
        for (k, y_value) in y.iter().enumerate().take(i) {
            sum -= l[i][k] * *y_value;
        }
        y[i] = sum / l[i][i];
    }

    let mut x = [0.0_f64; 6];
    for i in (0..6).rev() {
        let mut sum = y[i];
        for (k, x_value) in x.iter().enumerate().skip(i + 1) {
            sum -= l[k][i] * *x_value;
        }
        x[i] = sum / l[i][i];
    }
    x
}

#[allow(dead_code)]
pub(crate) fn symmetric_positive_definite_6x6(mat: &[[f64; 6]; 6]) -> bool {
    cholesky_6x6(mat).is_some()
}

pub(crate) fn se3_exp_f64(xi: [f64; 6]) -> Pose64 {
    let rho = [xi[0], xi[1], xi[2]];
    let omega = [xi[3], xi[4], xi[5]];
    let rotation = so3_exp_f64(omega);
    let theta = (omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]).sqrt();
    let omega_hat = skew_f64(omega);
    let omega_hat2 = mat_mul_f64(omega_hat, omega_hat);
    let mut v = identity3_f64();
    if theta < JACOBIAN_SMALL_ANGLE {
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
    if theta < JACOBIAN_SMALL_ANGLE {
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
        Pose64, Pose64Error, cholesky_6x6, cholesky_solve_6x6, mat_mul_f64, mat_mul_vec_f64,
        se3_exp_f64, se3_log_f64, so3_exp_f64, so3_left_jacobian_f64, so3_left_jacobian_inv_f64,
        so3_log_f64, so3_right_jacobian_f64, so3_right_jacobian_inv_f64,
        symmetric_positive_definite_6x6,
    };
    use crate::Pose;

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
    fn pose64_constructor_rejects_invalid_se3_values() {
        assert!(matches!(
            Pose64::try_from_rt(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [f64::NAN, 0.0, 0.0],
            ),
            Err(Pose64Error::NonFiniteTranslation { axis: 0, .. })
        ));
        assert!(matches!(
            Pose64::try_from_rt(
                [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0; 3],
            ),
            Err(Pose64Error::RotationNotOrthonormal { .. })
        ));
        assert!(matches!(
            Pose64::try_from_rt(
                [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0; 3],
            ),
            Err(Pose64Error::ImproperRotation { .. })
        ));

        let invalid_pose32 = Pose::from_rt(
            [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0; 3],
        );
        assert!(matches!(
            Pose64::try_from_pose32(invalid_pose32),
            Err(Pose64Error::RotationNotOrthonormal { .. })
        ));

        let large_translation = Pose64::try_from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [f64::MAX, 0.0, 0.0],
        )
        .expect("valid f64 pose");
        assert!(matches!(
            large_translation.try_to_pose32(),
            Err(Pose64Error::TranslationOutOfF32Range { axis: 0, .. })
        ));
    }

    #[test]
    fn pose32_and_pose64_compose_in_the_same_order() {
        let rotation64 = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let outer64 = Pose64::try_from_rt(rotation64, [1.0, 2.0, 0.0]).expect("outer pose");
        let inner64 = Pose64::try_from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [3.0, 0.0, 0.0],
        )
        .expect("inner pose");
        let composed64 = outer64.compose(inner64);

        let outer32 = Pose::from_rt(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [1.0, 2.0, 0.0],
        );
        let inner32 = Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [3.0, 0.0, 0.0],
        );
        let composed32 = Pose64::from_pose32(outer32.compose(inner32));
        assert_eq!(composed64.translation(), [1.0, 5.0, 0.0]);
        assert_eq!(composed32.translation(), composed64.translation());
        assert_eq!(composed32.rotation(), composed64.rotation());
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
    fn se3_exp_log_round_trip_is_stable_for_tiny_rotations() {
        for theta in [0.0, 1e-12, 2e-8, 1e-6, 5e-5] {
            let xi = [0.8, -0.4, 0.2, theta, -0.5 * theta, 0.25 * theta];
            let recovered = se3_log_f64(se3_exp_f64(xi));
            assert!(
                recovered.iter().all(|value| value.is_finite()),
                "non-finite SE(3) logarithm at theta={theta:e}: {recovered:?}"
            );
            for (axis, (actual, expected)) in recovered.iter().zip(xi).enumerate() {
                assert!(
                    (actual - expected).abs() < 1e-9,
                    "SE(3) round-trip mismatch at theta={theta:e}, axis={axis}: actual={actual:e}, expected={expected:e}"
                );
            }
        }
    }

    #[test]
    fn so3_right_jacobian_matches_finite_diff() {
        let omega = [0.2, -0.05, 0.1];
        let delta = [1e-6, -2e-6, 1.5e-6];
        let jr_delta = mat_mul_vec_f64(so3_right_jacobian_f64(omega), delta);
        let r_fd = so3_exp_f64([
            omega[0] + delta[0],
            omega[1] + delta[1],
            omega[2] + delta[2],
        ]);
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

    #[test]
    fn so3_log_reconstructs_exact_pi_rotation_with_mixed_axis_signs() {
        let inv_norm = 1.0_f64 / 14.0_f64.sqrt();
        let omega = [
            inv_norm * std::f64::consts::PI,
            -2.0 * inv_norm * std::f64::consts::PI,
            3.0 * inv_norm * std::f64::consts::PI,
        ];
        let rotation = so3_exp_f64(omega);
        let recovered = so3_log_f64(rotation);
        let reconstructed = so3_exp_f64(recovered);
        let error = rot_diff_norm(reconstructed, rotation);
        assert!(error < 1e-12, "exact-pi reconstruction error: {error:e}");
    }

    #[test]
    fn so3_left_jacobian_is_identity_at_zero() {
        let jl = so3_left_jacobian_f64([0.0, 0.0, 0.0]);
        assert_eq!(jl, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    }

    #[test]
    fn so3_right_jacobian_inverse_round_trip() {
        let omega = [0.17, -0.08, 0.05];
        let jr = so3_right_jacobian_f64(omega);
        let jr_inv = so3_right_jacobian_inv_f64(omega);
        let product = mat_mul_f64(jr, jr_inv);
        let err = rot_diff_norm(product, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        assert!(err < 1e-9, "right jacobian inverse error: {err}");
    }

    #[test]
    fn so3_left_and_right_jacobians_match_rotation_relation() {
        let omega = [0.13, -0.04, 0.09];
        let lhs = so3_left_jacobian_f64(omega);
        let rhs = mat_mul_f64(so3_exp_f64(omega), so3_right_jacobian_f64(omega));
        let err = rot_diff_norm(lhs, rhs);
        assert!(err < 1e-12, "left/right jacobian relation error: {err}");
    }

    #[test]
    fn so3_left_jacobian_inverse_round_trip() {
        let omega = [0.11, 0.06, -0.07];
        let jl = so3_left_jacobian_f64(omega);
        let jl_inv = so3_left_jacobian_inv_f64(omega);
        let product = mat_mul_f64(jl, jl_inv);
        let err = rot_diff_norm(product, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        assert!(err < 1e-9, "left jacobian inverse error: {err}");
    }

    #[test]
    fn cholesky_identity_returns_identity() {
        let l = cholesky_6x6(&[
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ])
        .expect("identity is SPD");
        assert_eq!(
            l,
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        );
    }

    #[test]
    fn cholesky_solve_known_system() {
        let a = [
            [4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 9.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 16.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 25.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 36.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 49.0],
        ];
        let l = cholesky_6x6(&a).expect("diagonal SPD");
        let x = cholesky_solve_6x6(&l, &[8.0, 18.0, 32.0, 50.0, 72.0, 98.0]);
        assert_eq!(x, [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn cholesky_rejects_non_positive_definite_matrix() {
        let matrix = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ];
        assert!(cholesky_6x6(&matrix).is_none());
        assert!(!symmetric_positive_definite_6x6(&matrix));
    }
}
