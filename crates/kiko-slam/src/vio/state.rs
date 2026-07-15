use crate::math::{se3_exp_f64, se3_log_f64};
use crate::{ImuBias, ImuBiasError, Pose64};

pub type NavTangent = [f64; 15];

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NavStateError {
    NonFinitePoseTranslation {
        axis: usize,
        value: f64,
    },
    NonFinitePoseRotation {
        row: usize,
        col: usize,
        value: f64,
    },
    NonFiniteVelocity {
        axis: usize,
        value: f64,
    },
    InvalidBias {
        source: ImuBiasError,
    },
    PoseComputation {
        operation: &'static str,
        source: crate::Pose64Error,
    },
}

impl std::fmt::Display for NavStateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NavStateError::NonFinitePoseTranslation { axis, value } => {
                write!(
                    f,
                    "pose translation axis {axis} must be finite, got {value}"
                )
            }
            NavStateError::NonFinitePoseRotation { row, col, value } => {
                write!(
                    f,
                    "pose rotation entry ({row}, {col}) must be finite, got {value}"
                )
            }
            NavStateError::NonFiniteVelocity { axis, value } => {
                write!(f, "velocity axis {axis} must be finite, got {value}")
            }
            NavStateError::InvalidBias { source } => {
                write!(f, "invalid navigation-state imu bias: {source}")
            }
            NavStateError::PoseComputation { operation, source } => {
                write!(f, "navigation-state {operation} failed: {source}")
            }
        }
    }
}

impl std::error::Error for NavStateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidBias { source } => Some(source),
            Self::PoseComputation { source, .. } => Some(source),
            Self::NonFinitePoseTranslation { .. }
            | Self::NonFinitePoseRotation { .. }
            | Self::NonFiniteVelocity { .. } => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NavState {
    pose_odom_from_body: Pose64,
    velocity_odom_mps: [f64; 3],
    bias: ImuBias,
}

impl NavState {
    pub fn try_new(
        pose_odom_from_body: Pose64,
        velocity_odom_mps: [f64; 3],
        bias: ImuBias,
    ) -> Result<Self, NavStateError> {
        let translation = pose_odom_from_body.translation();
        for (axis, value) in translation.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(NavStateError::NonFinitePoseTranslation { axis, value });
            }
        }
        let rotation = pose_odom_from_body.rotation();
        for (row_idx, row) in rotation.iter().enumerate() {
            for (col_idx, value) in row.iter().copied().enumerate() {
                if !value.is_finite() {
                    return Err(NavStateError::NonFinitePoseRotation {
                        row: row_idx,
                        col: col_idx,
                        value,
                    });
                }
            }
        }
        for (axis, value) in velocity_odom_mps.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(NavStateError::NonFiniteVelocity { axis, value });
            }
        }
        Ok(Self {
            pose_odom_from_body,
            velocity_odom_mps,
            bias,
        })
    }

    pub fn pose_odom_from_body(&self) -> Pose64 {
        self.pose_odom_from_body
    }

    pub fn velocity_odom_mps(&self) -> [f64; 3] {
        self.velocity_odom_mps
    }

    pub fn bias(&self) -> &ImuBias {
        &self.bias
    }

    pub fn retract(&self, delta: &NavTangent) -> Result<Self, NavStateError> {
        let mut pose_delta = [0.0_f64; 6];
        pose_delta.copy_from_slice(&delta[..6]);
        let pose_odom_from_body = se3_exp_f64(pose_delta)
            .try_compose(self.pose_odom_from_body)
            .map_err(|source| NavStateError::PoseComputation {
                operation: "retraction pose update",
                source,
            })?;
        let velocity_odom_mps = [
            self.velocity_odom_mps[0] + delta[6],
            self.velocity_odom_mps[1] + delta[7],
            self.velocity_odom_mps[2] + delta[8],
        ];
        let bias = self
            .bias
            .checked_add(
                [delta[9], delta[10], delta[11]],
                [delta[12], delta[13], delta[14]],
            )
            .map_err(|source| NavStateError::InvalidBias { source })?;
        Self::try_new(pose_odom_from_body, velocity_odom_mps, bias)
    }

    pub fn local_coordinates(&self, other: &Self) -> Result<NavTangent, NavStateError> {
        let mut tangent = [0.0_f64; 15];
        let self_inverse = self.pose_odom_from_body.try_inverse().map_err(|source| {
            NavStateError::PoseComputation {
                operation: "local-coordinate source-pose inversion",
                source,
            }
        })?;
        let relative = other
            .pose_odom_from_body
            .try_compose(self_inverse)
            .map_err(|source| NavStateError::PoseComputation {
                operation: "local-coordinate relative pose",
                source,
            })?;
        let pose_delta = se3_log_f64(relative);
        tangent[..6].copy_from_slice(&pose_delta);
        tangent[6] = other.velocity_odom_mps[0] - self.velocity_odom_mps[0];
        tangent[7] = other.velocity_odom_mps[1] - self.velocity_odom_mps[1];
        tangent[8] = other.velocity_odom_mps[2] - self.velocity_odom_mps[2];
        let self_accel_bias_mps2 = self.bias.accel_mps2();
        let other_accel_bias_mps2 = other.bias.accel_mps2();
        let self_gyro_bias_radps = self.bias.gyro_radps();
        let other_gyro_bias_radps = other.bias.gyro_radps();
        tangent[9] = other_accel_bias_mps2[0] - self_accel_bias_mps2[0];
        tangent[10] = other_accel_bias_mps2[1] - self_accel_bias_mps2[1];
        tangent[11] = other_accel_bias_mps2[2] - self_accel_bias_mps2[2];
        tangent[12] = other_gyro_bias_radps[0] - self_gyro_bias_radps[0];
        tangent[13] = other_gyro_bias_radps[1] - self_gyro_bias_radps[1];
        tangent[14] = other_gyro_bias_radps[2] - self_gyro_bias_radps[2];
        Ok(tangent)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GravityError {
    ZeroNorm,
    NonFinite { axis: usize, value: f64 },
}

impl std::fmt::Display for GravityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GravityError::ZeroNorm => write!(f, "gravity vector must have non-zero norm"),
            GravityError::NonFinite { axis, value } => {
                write!(f, "gravity axis {axis} must be finite, got {value}")
            }
        }
    }
}

impl std::error::Error for GravityError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gravity {
    vector_odom_mps2: [f64; 3],
}

impl Gravity {
    pub fn try_new(vector_odom_mps2: [f64; 3]) -> Result<Self, GravityError> {
        for (axis, value) in vector_odom_mps2.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(GravityError::NonFinite { axis, value });
            }
        }
        let norm_sq = vector_odom_mps2[0] * vector_odom_mps2[0]
            + vector_odom_mps2[1] * vector_odom_mps2[1]
            + vector_odom_mps2[2] * vector_odom_mps2[2];
        if norm_sq <= 0.0 {
            return Err(GravityError::ZeroNorm);
        }
        Ok(Self { vector_odom_mps2 })
    }

    pub fn vector_odom_mps2(&self) -> [f64; 3] {
        self.vector_odom_mps2
    }

    pub fn magnitude_mps2(&self) -> f64 {
        let v = self.vector_odom_mps2;
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nav_state_preserves_constructor_values() {
        let bias = ImuBias::try_new([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]).expect("finite bias");
        let state =
            NavState::try_new(Pose64::identity(), [1.0, 2.0, 3.0], bias).expect("nav state");
        assert_eq!(state.pose_odom_from_body().translation(), [0.0, 0.0, 0.0]);
        assert_eq!(state.velocity_odom_mps(), [1.0, 2.0, 3.0]);
        assert_eq!(state.bias(), &bias);
    }

    #[test]
    fn gravity_preserves_vector() {
        let gravity = Gravity::try_new([0.0, 0.0, -9.81]).expect("gravity");
        assert_eq!(gravity.vector_odom_mps2(), [0.0, 0.0, -9.81]);
        assert!((gravity.magnitude_mps2() - 9.81).abs() < 1e-12);
    }

    #[test]
    fn nav_state_retract_and_local_coordinates_round_trip() {
        let state = NavState::try_new(
            Pose64::identity(),
            [0.5, -0.2, 0.1],
            ImuBias::try_new([0.01, -0.02, 0.03], [0.001, -0.002, 0.003]).expect("finite bias"),
        )
        .expect("nav state");
        let delta: NavTangent = [
            0.01, -0.02, 0.03, 0.001, -0.002, 0.003, 0.1, -0.2, 0.3, 0.001, 0.002, -0.001, -0.0005,
            0.0007, -0.0009,
        ];
        let moved = state.retract(&delta).expect("retract");
        let recovered = state.local_coordinates(&moved).expect("local coordinates");
        for i in 0..15 {
            assert!((recovered[i] - delta[i]).abs() < 1e-9, "index {i}");
        }
    }

    #[test]
    fn gravity_rejects_zero_vector() {
        let err = Gravity::try_new([0.0, 0.0, 0.0]).expect_err("zero gravity should fail");
        assert_eq!(err, GravityError::ZeroNorm);
    }

    #[test]
    fn nav_state_rejects_non_finite_velocity() {
        let err = NavState::try_new(Pose64::identity(), [f64::NAN, 0.0, 0.0], ImuBias::default())
            .expect_err("nan velocity should fail");
        match err {
            NavStateError::NonFiniteVelocity { axis, value } => {
                assert_eq!(axis, 0);
                assert!(value.is_nan());
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn nav_state_rejects_non_finite_pose_translation() {
        let err = NavState::try_new(
            Pose64::from_rt(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [f64::INFINITY, 0.0, 0.0],
            ),
            [0.0; 3],
            ImuBias::default(),
        )
        .expect_err("non-finite translation should fail");
        match err {
            NavStateError::NonFinitePoseTranslation { axis, value } => {
                assert_eq!(axis, 0);
                assert!(!value.is_finite());
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn nav_state_retract_preserves_nonfinite_bias_source() {
        let state = NavState::try_new(Pose64::identity(), [0.0; 3], ImuBias::default())
            .expect("navigation state");
        let mut delta = [0.0; 15];
        delta[13] = f64::NAN;

        let error = state
            .retract(&delta)
            .expect_err("nonfinite bias update must fail");
        assert!(matches!(
            error,
            NavStateError::InvalidBias {
                source: ImuBiasError::NonFiniteGyroRadps { axis: 1, value },
            } if value.is_nan()
        ));
        assert!(std::error::Error::source(&error).is_some());
    }
}
