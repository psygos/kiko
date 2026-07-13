use crate::math::{mat_mul_f64, mat_mul_vec_f64, so3_log_f64};
use crate::{Gravity, NavState, PreintegratedImu, PreintegrationError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImuResidualQuantity {
    RotationRadians,
    VelocityMps,
    PositionM,
}

impl std::fmt::Display for ImuResidualQuantity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::RotationRadians => "rotation residual (rad)",
            Self::VelocityMps => "velocity residual (m/s)",
            Self::PositionM => "position residual (m)",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiasRandomWalkResidualQuantity {
    AccelerometerMps2,
    GyroscopeRadps,
}

impl std::fmt::Display for BiasRandomWalkResidualQuantity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::AccelerometerMps2 => "accelerometer-bias random-walk residual (m/s^2)",
            Self::GyroscopeRadps => "gyroscope-bias random-walk residual (rad/s)",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VioFactorError {
    PreintegrationCorrection {
        source: PreintegrationError,
    },
    NonFiniteImuResidual {
        quantity: ImuResidualQuantity,
        axis: usize,
        value: f64,
    },
    NonFiniteBiasRandomWalkResidual {
        quantity: BiasRandomWalkResidualQuantity,
        axis: usize,
        value: f64,
    },
}

impl std::fmt::Display for VioFactorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VioFactorError::PreintegrationCorrection { source } => {
                write!(f, "imu preintegration bias correction failed: {source}")
            }
            VioFactorError::NonFiniteImuResidual {
                quantity,
                axis,
                value,
            } => write!(f, "VIO {quantity} axis {axis} must be finite, got {value}"),
            VioFactorError::NonFiniteBiasRandomWalkResidual {
                quantity,
                axis,
                value,
            } => write!(f, "VIO {quantity} axis {axis} must be finite, got {value}"),
        }
    }
}

impl std::error::Error for VioFactorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::PreintegrationCorrection { source } => Some(source),
            Self::NonFiniteImuResidual { .. } | Self::NonFiniteBiasRandomWalkResidual { .. } => {
                None
            }
        }
    }
}

pub struct ImuFactor;

impl ImuFactor {
    pub fn residual(
        state_i: &NavState,
        state_j: &NavState,
        preintegrated: &PreintegratedImu,
        gravity: &Gravity,
    ) -> Result<[f64; 9], VioFactorError> {
        let corrected = preintegrated
            .corrected_first_order(state_i.bias())
            .map_err(|source| VioFactorError::PreintegrationCorrection { source })?;
        let pose_i = state_i.pose_odom_from_body();
        let pose_j = state_j.pose_odom_from_body();
        let r_i = pose_i.rotation();
        let p_i = pose_i.translation();
        let r_j = pose_j.rotation();
        let p_j = pose_j.translation();
        let v_i = state_i.velocity_odom_mps();
        let v_j = state_j.velocity_odom_mps();
        let g = gravity.vector_odom_mps2();
        let duration_seconds = preintegrated.duration_seconds();

        let r_i_t = transpose3(r_i);
        let rotation_error = so3_log_f64(mat_mul_f64(
            transpose3(corrected.delta_rotation()),
            mat_mul_f64(r_i_t, r_j),
        ));

        let delta_position_odom = [
            p_j[0]
                - p_i[0]
                - v_i[0] * duration_seconds
                - 0.5 * g[0] * duration_seconds * duration_seconds,
            p_j[1]
                - p_i[1]
                - v_i[1] * duration_seconds
                - 0.5 * g[1] * duration_seconds * duration_seconds,
            p_j[2]
                - p_i[2]
                - v_i[2] * duration_seconds
                - 0.5 * g[2] * duration_seconds * duration_seconds,
        ];
        let delta_velocity_odom = [
            v_j[0] - v_i[0] - g[0] * duration_seconds,
            v_j[1] - v_i[1] - g[1] * duration_seconds,
            v_j[2] - v_i[2] - g[2] * duration_seconds,
        ];

        let position_error = sub_vec3(
            mat_mul_vec_f64(r_i_t, delta_position_odom),
            corrected.delta_position_m(),
        );
        let velocity_error = sub_vec3(
            mat_mul_vec_f64(r_i_t, delta_velocity_odom),
            corrected.delta_velocity_mps(),
        );

        for (quantity, residual) in [
            (ImuResidualQuantity::RotationRadians, rotation_error),
            (ImuResidualQuantity::VelocityMps, velocity_error),
            (ImuResidualQuantity::PositionM, position_error),
        ] {
            for (axis, value) in residual.into_iter().enumerate() {
                if !value.is_finite() {
                    return Err(VioFactorError::NonFiniteImuResidual {
                        quantity,
                        axis,
                        value,
                    });
                }
            }
        }

        Ok([
            rotation_error[0],
            rotation_error[1],
            rotation_error[2],
            velocity_error[0],
            velocity_error[1],
            velocity_error[2],
            position_error[0],
            position_error[1],
            position_error[2],
        ])
    }
}

pub(crate) fn bias_random_walk_residual(
    state_i: &NavState,
    state_j: &NavState,
) -> Result<[f64; 6], VioFactorError> {
    let bias_i = state_i.bias();
    let bias_j = state_j.bias();
    let accel_i_mps2 = bias_i.accel_mps2();
    let accel_j_mps2 = bias_j.accel_mps2();
    let gyro_i_radps = bias_i.gyro_radps();
    let gyro_j_radps = bias_j.gyro_radps();
    let residual = [
        accel_j_mps2[0] - accel_i_mps2[0],
        accel_j_mps2[1] - accel_i_mps2[1],
        accel_j_mps2[2] - accel_i_mps2[2],
        gyro_j_radps[0] - gyro_i_radps[0],
        gyro_j_radps[1] - gyro_i_radps[1],
        gyro_j_radps[2] - gyro_i_radps[2],
    ];
    for (residual_axis, value) in residual.into_iter().enumerate() {
        if !value.is_finite() {
            let quantity = if residual_axis < 3 {
                BiasRandomWalkResidualQuantity::AccelerometerMps2
            } else {
                BiasRandomWalkResidualQuantity::GyroscopeRadps
            };
            return Err(VioFactorError::NonFiniteBiasRandomWalkResidual {
                quantity,
                axis: residual_axis % 3,
                value,
            });
        }
    }
    Ok(residual)
}

fn transpose3(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [matrix[0][0], matrix[1][0], matrix[2][0]],
        [matrix[0][1], matrix[1][1], matrix[2][1]],
        [matrix[0][2], matrix[1][2], matrix[2][2]],
    ]
}

fn sub_vec3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ImuBatch, ImuBias, ImuNoiseModel, ImuSample, Pose64, Timestamp};

    fn noise() -> ImuNoiseModel {
        ImuNoiseModel::new(0.1, 0.01, 0.001, 0.0001).expect("noise")
    }

    fn batch(samples: &[(i64, [f64; 3], [f64; 3])]) -> ImuBatch {
        ImuBatch::new(
            samples
                .iter()
                .map(|(timestamp, accel, gyro)| {
                    ImuSample::new(Timestamp::from_nanos(*timestamp), *accel, *gyro)
                        .expect("imu sample")
                })
                .collect(),
        )
        .expect("imu batch")
    }

    #[test]
    fn imu_factor_residual_is_zero_for_consistent_free_fall_motion() {
        let batch = batch(&[
            (0, [0.0; 3], [0.0; 3]),
            (10_000_000, [0.0; 3], [0.0; 3]),
            (20_000_000, [0.0; 3], [0.0; 3]),
        ]);
        let preintegrated = PreintegratedImu::integrate(&batch, &ImuBias::default(), &noise())
            .expect("preintegrated");
        let gravity = Gravity::try_new([0.0, 0.0, -9.81]).expect("gravity");
        let state_i =
            NavState::try_new(Pose64::identity(), [0.0; 3], ImuBias::default()).expect("state i");
        let state_j = NavState::try_new(
            Pose64::from_rt(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [
                    0.0,
                    0.0,
                    -0.5 * 9.81
                        * preintegrated.duration_seconds()
                        * preintegrated.duration_seconds(),
                ],
            ),
            [0.0, 0.0, -9.81 * preintegrated.duration_seconds()],
            ImuBias::default(),
        )
        .expect("state j");
        let residual = ImuFactor::residual(&state_i, &state_j, &preintegrated, &gravity)
            .expect("finite imu residual");
        let norm = residual
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        assert!(norm < 1e-9, "imu residual norm={norm}");
    }

    #[test]
    fn imu_factor_residual_is_zero_for_consistent_free_fall_with_rotation() {
        let batch = batch(&[
            (0, [0.0; 3], [0.0, 0.0, 0.5]),
            (10_000_000, [0.0; 3], [0.0, 0.0, 0.5]),
            (20_000_000, [0.0; 3], [0.0, 0.0, 0.5]),
        ]);
        let preintegrated = PreintegratedImu::integrate(&batch, &ImuBias::default(), &noise())
            .expect("preintegrated");
        let gravity = Gravity::try_new([0.0, 0.0, -9.81]).expect("gravity");
        let state_i =
            NavState::try_new(Pose64::identity(), [0.0; 3], ImuBias::default()).expect("state i");
        let state_j = NavState::try_new(
            Pose64::from_rt(
                preintegrated.delta_rotation(),
                [
                    0.0,
                    0.0,
                    -0.5 * 9.81
                        * preintegrated.duration_seconds()
                        * preintegrated.duration_seconds(),
                ],
            ),
            [0.0, 0.0, -9.81 * preintegrated.duration_seconds()],
            ImuBias::default(),
        )
        .expect("state j");
        let residual = ImuFactor::residual(&state_i, &state_j, &preintegrated, &gravity)
            .expect("finite imu residual");
        let norm = residual
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        assert!(norm < 1e-9, "imu residual norm={norm}");
    }

    #[test]
    fn bias_random_walk_residual_is_zero_for_equal_biases() {
        let bias = ImuBias::try_new([0.1, -0.2, 0.3], [0.01, -0.02, 0.03]).expect("finite bias");
        let state_i = NavState::try_new(Pose64::identity(), [0.0; 3], bias).expect("state i");
        let state_j = NavState::try_new(Pose64::identity(), [0.0; 3], bias).expect("state j");
        assert_eq!(
            bias_random_walk_residual(&state_i, &state_j).expect("finite residual"),
            [0.0; 6]
        );
    }

    #[test]
    fn bias_random_walk_residual_rejects_subtraction_overflow_at_factor_boundary() {
        let bias_i = ImuBias::try_new([-f64::MAX, 0.0, 0.0], [0.0; 3]).expect("finite bias i");
        let bias_j = ImuBias::try_new([f64::MAX, 0.0, 0.0], [0.0; 3]).expect("finite bias j");
        let state_i = NavState::try_new(Pose64::identity(), [0.0; 3], bias_i).expect("state i");
        let state_j = NavState::try_new(Pose64::identity(), [0.0; 3], bias_j).expect("state j");

        let error = bias_random_walk_residual(&state_i, &state_j)
            .expect_err("overflowed residual must fail at factor boundary");
        assert!(matches!(
            error,
            VioFactorError::NonFiniteBiasRandomWalkResidual {
                quantity: BiasRandomWalkResidualQuantity::AccelerometerMps2,
                axis: 0,
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn imu_factor_applies_first_order_bias_correction() {
        let bias =
            ImuBias::try_new([0.1, -0.05, 0.02], [0.01, -0.015, 0.005]).expect("finite bias");
        let gravity = Gravity::try_new([0.0, 0.0, -9.81]).expect("gravity");
        let accel_bias_mps2 = bias.accel_mps2();
        let gyro_bias_radps = bias.gyro_radps();
        let accel_measurement = [
            accel_bias_mps2[0],
            accel_bias_mps2[1],
            9.81 + accel_bias_mps2[2],
        ];
        let batch = batch(&[
            (0, accel_measurement, gyro_bias_radps),
            (10_000_000, accel_measurement, gyro_bias_radps),
            (20_000_000, accel_measurement, gyro_bias_radps),
        ]);
        let preintegrated = PreintegratedImu::integrate(&batch, &ImuBias::default(), &noise())
            .expect("preintegrated");
        let state_i = NavState::try_new(Pose64::identity(), [0.0; 3], bias).expect("state i");
        let state_j = NavState::try_new(Pose64::identity(), [0.0; 3], bias).expect("state j");
        let residual = ImuFactor::residual(&state_i, &state_j, &preintegrated, &gravity)
            .expect("finite imu residual");
        let norm = residual
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        assert!(norm < 1e-5, "bias-corrected imu residual norm={norm}");
    }

    #[test]
    fn imu_factor_preserves_preintegration_correction_failure() {
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

        let error = ImuFactor::residual(&state_i, &state_j, &preintegrated, &gravity)
            .expect_err("overflowed bias correction must fail");
        assert!(matches!(
            error,
            VioFactorError::PreintegrationCorrection {
                source: PreintegrationError::InvalidRotation {
                    quantity: crate::PreintegrationQuantity::DeltaRotation,
                    ..
                },
            }
        ));
        assert!(std::error::Error::source(&error).is_some());
    }
}
