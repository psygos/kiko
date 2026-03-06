use crate::math::{mat_mul_f64, mat_mul_vec_f64, so3_log_f64};
use crate::{Gravity, Keypoint, NavState, PinholeIntrinsics, Point3, Pose64, PreintegratedImu};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VioFactorError {
    PointBehindCamera { depth_m: f64 },
}

impl std::fmt::Display for VioFactorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VioFactorError::PointBehindCamera { depth_m } => {
                write!(f, "reprojection point lies behind camera (depth={depth_m})")
            }
        }
    }
}

impl std::error::Error for VioFactorError {}

pub struct ImuFactor;

impl ImuFactor {
    pub fn residual(
        state_i: &NavState,
        state_j: &NavState,
        preintegrated: &PreintegratedImu,
        gravity: &Gravity,
    ) -> [f64; 9] {
        let corrected = preintegrated.corrected_first_order(state_i.bias());
        let pose_i = state_i.pose_odom_from_body();
        let pose_j = state_j.pose_odom_from_body();
        let r_i = pose_i.rotation();
        let p_i = pose_i.translation();
        let r_j = pose_j.rotation();
        let p_j = pose_j.translation();
        let v_i = state_i.velocity_odom_mps();
        let v_j = state_j.velocity_odom_mps();
        let g = gravity.vector_odom_mps2();
        let dt = preintegrated.dt_seconds;

        let r_i_t = transpose3(r_i);
        let rotation_error =
            so3_log_f64(mat_mul_f64(corrected.delta_rotation, mat_mul_f64(r_i_t, r_j)));

        let delta_position_odom = [
            p_j[0] - p_i[0] - v_i[0] * dt - 0.5 * g[0] * dt * dt,
            p_j[1] - p_i[1] - v_i[1] * dt - 0.5 * g[1] * dt * dt,
            p_j[2] - p_i[2] - v_i[2] * dt - 0.5 * g[2] * dt * dt,
        ];
        let delta_velocity_odom = [
            v_j[0] - v_i[0] - g[0] * dt,
            v_j[1] - v_i[1] - g[1] * dt,
            v_j[2] - v_i[2] - g[2] * dt,
        ];

        let position_error = sub_vec3(
            mat_mul_vec_f64(r_i_t, delta_position_odom),
            corrected.delta_position,
        );
        let velocity_error = sub_vec3(
            mat_mul_vec_f64(r_i_t, delta_velocity_odom),
            corrected.delta_velocity,
        );

        [
            rotation_error[0],
            rotation_error[1],
            rotation_error[2],
            velocity_error[0],
            velocity_error[1],
            velocity_error[2],
            position_error[0],
            position_error[1],
            position_error[2],
        ]
    }
}

pub fn pose_prior_residual(state: &NavState, pose_measurement_odom: Pose64) -> [f64; 6] {
    so3_se3_residual(pose_measurement_odom, state.pose_odom_from_body())
}

pub fn bias_random_walk_residual(state_i: &NavState, state_j: &NavState) -> [f64; 6] {
    let bias_i = state_i.bias();
    let bias_j = state_j.bias();
    [
        bias_j.accel[0] - bias_i.accel[0],
        bias_j.accel[1] - bias_i.accel[1],
        bias_j.accel[2] - bias_i.accel[2],
        bias_j.gyro[0] - bias_i.gyro[0],
        bias_j.gyro[1] - bias_i.gyro[1],
        bias_j.gyro[2] - bias_i.gyro[2],
    ]
}

pub fn reprojection_residual(
    state: &NavState,
    camera_from_body: Pose64,
    point_odom: Point3,
    pixel: Keypoint,
    intrinsics: PinholeIntrinsics,
) -> Result<[f64; 2], VioFactorError> {
    let camera_from_odom = camera_from_body.compose(state.pose_odom_from_body().inverse());
    let point_cam = transform_point(
        camera_from_odom,
        [
            f64::from(point_odom.x),
            f64::from(point_odom.y),
            f64::from(point_odom.z),
        ],
    );
    let x = point_cam[0];
    let y = point_cam[1];
    let z = point_cam[2];
    if z <= 0.0 {
        return Err(VioFactorError::PointBehindCamera { depth_m: z });
    }
    let u = f64::from(intrinsics.fx()) * (x / z) + f64::from(intrinsics.cx());
    let v = f64::from(intrinsics.fy()) * (y / z) + f64::from(intrinsics.cy());
    Ok([f64::from(pixel.x) - u, f64::from(pixel.y) - v])
}

fn transform_point(transform: Pose64, point: [f64; 3]) -> [f64; 3] {
    let rotated = mat_mul_vec_f64(transform.rotation(), point);
    let translation = transform.translation();
    [
        rotated[0] + translation[0],
        rotated[1] + translation[1],
        rotated[2] + translation[2],
    ]
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

fn so3_se3_residual(target: Pose64, estimate: Pose64) -> [f64; 6] {
    let delta = target.compose(estimate.inverse());
    let rot = so3_log_f64(delta.rotation());
    let t = delta.translation();
    [t[0], t[1], t[2], rot[0], rot[1], rot[2]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::CameraIntrinsics;
    use crate::{ImuBatch, ImuBias, ImuNoiseModel, ImuSample, Timestamp};

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

    fn intrinsics() -> PinholeIntrinsics {
        PinholeIntrinsics::try_from(&CameraIntrinsics {
            fx: 100.0,
            fy: 100.0,
            cx: 50.0,
            cy: 40.0,
            width: 100,
            height: 80,
        })
        .expect("intrinsics")
    }

    #[test]
    fn imu_factor_residual_is_zero_for_consistent_free_fall_motion() {
        let batch = batch(&[
            (0, [0.0; 3], [0.0; 3]),
            (10_000_000, [0.0; 3], [0.0; 3]),
            (20_000_000, [0.0; 3], [0.0; 3]),
        ]);
        let preintegrated =
            PreintegratedImu::integrate(&batch, &ImuBias::default(), &noise()).expect("preintegrated");
        let gravity = Gravity::try_new([0.0, 0.0, -9.81]).expect("gravity");
        let state_i = NavState::try_new(Pose64::identity(), [0.0; 3], ImuBias::default())
            .expect("state i");
        let state_j = NavState::try_new(
            Pose64::from_rt(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0, 0.0, -0.5 * 9.81 * preintegrated.dt_seconds * preintegrated.dt_seconds],
            ),
            [0.0, 0.0, -9.81 * preintegrated.dt_seconds],
            ImuBias::default(),
        )
        .expect("state j");
        let residual = ImuFactor::residual(&state_i, &state_j, &preintegrated, &gravity);
        let norm = residual.iter().map(|value| value * value).sum::<f64>().sqrt();
        assert!(norm < 1e-9, "imu residual norm={norm}");
    }

    #[test]
    fn bias_random_walk_residual_is_zero_for_equal_biases() {
        let bias = ImuBias {
            accel: [0.1, -0.2, 0.3],
            gyro: [0.01, -0.02, 0.03],
        };
        let state_i = NavState::try_new(Pose64::identity(), [0.0; 3], bias.clone()).expect("state i");
        let state_j = NavState::try_new(Pose64::identity(), [0.0; 3], bias).expect("state j");
        assert_eq!(bias_random_walk_residual(&state_i, &state_j), [0.0; 6]);
    }

    #[test]
    fn imu_factor_applies_first_order_bias_correction() {
        let bias = ImuBias {
            accel: [0.1, -0.05, 0.02],
            gyro: [0.01, -0.015, 0.005],
        };
        let gravity = Gravity::try_new([0.0, 0.0, -9.81]).expect("gravity");
        let accel_measurement = [
            bias.accel[0],
            bias.accel[1],
            9.81 + bias.accel[2],
        ];
        let batch = batch(&[
            (0, accel_measurement, bias.gyro),
            (10_000_000, accel_measurement, bias.gyro),
            (20_000_000, accel_measurement, bias.gyro),
        ]);
        let preintegrated =
            PreintegratedImu::integrate(&batch, &ImuBias::default(), &noise()).expect("preintegrated");
        let state_i = NavState::try_new(Pose64::identity(), [0.0; 3], bias.clone())
            .expect("state i");
        let state_j = NavState::try_new(Pose64::identity(), [0.0; 3], bias).expect("state j");
        let residual = ImuFactor::residual(&state_i, &state_j, &preintegrated, &gravity);
        let norm = residual.iter().map(|value| value * value).sum::<f64>().sqrt();
        assert!(norm < 1e-5, "bias-corrected imu residual norm={norm}");
    }

    #[test]
    fn reprojection_residual_is_zero_for_consistent_geometry() {
        let state =
            NavState::try_new(Pose64::identity(), [0.0; 3], ImuBias::default()).expect("state");
        let residual = reprojection_residual(
            &state,
            Pose64::identity(),
            Point3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            Keypoint { x: 50.0, y: 40.0 },
            intrinsics(),
        )
        .expect("reprojection");
        assert!((residual[0]).abs() < 1e-12);
        assert!((residual[1]).abs() < 1e-12);
    }

    #[test]
    fn reprojection_residual_rejects_point_behind_camera() {
        let state =
            NavState::try_new(Pose64::identity(), [0.0; 3], ImuBias::default()).expect("state");
        let err = reprojection_residual(
            &state,
            Pose64::identity(),
            Point3 {
                x: 0.0,
                y: 0.0,
                z: -1.0,
            },
            Keypoint { x: 50.0, y: 40.0 },
            intrinsics(),
        )
        .expect_err("point behind camera should fail");
        assert_eq!(err, VioFactorError::PointBehindCamera { depth_m: -1.0 });
    }
}
