use std::collections::VecDeque;
use std::num::NonZeroUsize;

use crate::math::{mat_mul_vec_f64, mat_mul_f64};
use crate::map::KeyframeId;
use crate::{
    bias_random_walk_residual, pose_prior_residual, reprojection_residual, Gravity, ImuFactor,
    NavState, NavTangent, Observation, PinholeIntrinsics, Pose64, PreintegratedImu,
};

const STATE_DIM: usize = 15;
const IMU_RESIDUAL_DIM: usize = 9;
const POSE_PRIOR_RESIDUAL_DIM: usize = 6;
const REPROJECTION_RESIDUAL_DIM: usize = 2;
const BIAS_RW_RESIDUAL_DIM: usize = 6;

#[derive(Clone, Copy, Debug)]
pub struct VioConfig {
    window_size: NonZeroUsize,
    max_iterations: NonZeroUsize,
    pose_prior_weight: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VioConfigError {
    ZeroWindowSize,
    ZeroMaxIterations,
    NonFinitePosePriorWeight,
    NonPositivePosePriorWeight,
}

impl std::fmt::Display for VioConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VioConfigError::ZeroWindowSize => write!(f, "vio window size must be > 0"),
            VioConfigError::ZeroMaxIterations => write!(f, "vio max iterations must be > 0"),
            VioConfigError::NonFinitePosePriorWeight => {
                write!(f, "vio pose prior weight must be finite")
            }
            VioConfigError::NonPositivePosePriorWeight => {
                write!(f, "vio pose prior weight must be > 0")
            }
        }
    }
}

impl std::error::Error for VioConfigError {}

impl VioConfig {
    pub fn new(window_size: usize) -> Result<Self, VioConfigError> {
        let window_size = NonZeroUsize::new(window_size).ok_or(VioConfigError::ZeroWindowSize)?;
        Ok(Self {
            window_size,
            max_iterations: NonZeroUsize::new(4).expect("non-zero default"),
            pose_prior_weight: 100.0,
        })
    }

    pub fn window_size(self) -> usize {
        self.window_size.get()
    }

    pub fn max_iterations(self) -> usize {
        self.max_iterations.get()
    }

    pub fn pose_prior_weight(self) -> f64 {
        self.pose_prior_weight
    }

    pub fn with_max_iterations(mut self, max_iterations: usize) -> Result<Self, VioConfigError> {
        self.max_iterations =
            NonZeroUsize::new(max_iterations).ok_or(VioConfigError::ZeroMaxIterations)?;
        Ok(self)
    }

    pub fn with_pose_prior_weight(mut self, weight: f64) -> Result<Self, VioConfigError> {
        if !weight.is_finite() {
            return Err(VioConfigError::NonFinitePosePriorWeight);
        }
        if weight <= 0.0 {
            return Err(VioConfigError::NonPositivePosePriorWeight);
        }
        self.pose_prior_weight = weight;
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalVioError {
    AlreadyInitialized { keyframe_id: KeyframeId },
    NotInitialized,
    DuplicateKeyframe { keyframe_id: KeyframeId },
}

impl std::fmt::Display for LocalVioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LocalVioError::AlreadyInitialized { keyframe_id } => write!(
                f,
                "local vio already initialized at keyframe {keyframe_id:?}"
            ),
            LocalVioError::NotInitialized => write!(f, "local vio has not been initialized"),
            LocalVioError::DuplicateKeyframe { keyframe_id } => {
                write!(f, "local vio already contains keyframe {keyframe_id:?}")
            }
        }
    }
}

impl std::error::Error for LocalVioError {}

#[derive(Clone, Debug)]
pub struct VioEstimate {
    keyframe_id: KeyframeId,
    state: NavState,
}

impl VioEstimate {
    pub fn keyframe_id(&self) -> KeyframeId {
        self.keyframe_id
    }

    pub fn state(&self) -> &NavState {
        &self.state
    }
}

#[derive(Clone, Debug)]
pub struct VioOdometryConstraint {
    from: KeyframeId,
    to: KeyframeId,
    relative_pose: Pose64,
    information: [[f64; 6]; 6],
}

impl VioOdometryConstraint {
    pub fn from(&self) -> KeyframeId {
        self.from
    }

    pub fn to(&self) -> KeyframeId {
        self.to
    }

    pub fn relative_pose(&self) -> Pose64 {
        self.relative_pose
    }

    pub fn information(&self) -> [[f64; 6]; 6] {
        self.information
    }
}

#[derive(Clone, Debug)]
struct VioFrame {
    keyframe_id: KeyframeId,
    state: NavState,
    pose_measurement_odom: Pose64,
    visual_observations: Box<[Observation]>,
    #[allow(dead_code)]
    preintegrated_from_prev: Option<PreintegratedImu>,
}

pub struct LocalVio {
    config: VioConfig,
    gravity: Gravity,
    camera_from_body: Pose64,
    intrinsics: PinholeIntrinsics,
    frames: VecDeque<VioFrame>,
}

impl LocalVio {
    pub fn new(
        config: VioConfig,
        gravity: Gravity,
        camera_from_body: Pose64,
        intrinsics: PinholeIntrinsics,
    ) -> Self {
        Self {
            config,
            gravity,
            camera_from_body,
            intrinsics,
            frames: VecDeque::new(),
        }
    }

    pub fn initialize(
        &mut self,
        keyframe_id: KeyframeId,
        state: NavState,
        pose_measurement_odom: Pose64,
        visual_observations: Vec<Observation>,
    ) -> Result<(), LocalVioError> {
        if let Some(existing) = self.frames.front() {
            return Err(LocalVioError::AlreadyInitialized {
                keyframe_id: existing.keyframe_id,
            });
        }
        self.frames.push_back(VioFrame {
            keyframe_id,
            state,
            pose_measurement_odom,
            visual_observations: visual_observations.into_boxed_slice(),
            preintegrated_from_prev: None,
        });
        Ok(())
    }

    pub fn latest_estimate(&self) -> Option<VioEstimate> {
        self.frames.back().map(|frame| VioEstimate {
            keyframe_id: frame.keyframe_id,
            state: frame.state.clone(),
        })
    }

    pub fn estimate_for(&self, keyframe_id: KeyframeId) -> Option<VioEstimate> {
        self.frames
            .iter()
            .find(|frame| frame.keyframe_id == keyframe_id)
            .map(|frame| VioEstimate {
                keyframe_id: frame.keyframe_id,
                state: frame.state.clone(),
            })
    }

    pub fn predict_from_latest(
        &self,
        preintegrated: &PreintegratedImu,
    ) -> Result<VioEstimate, LocalVioError> {
        let previous = self.frames.back().ok_or(LocalVioError::NotInitialized)?;
        Ok(VioEstimate {
            keyframe_id: previous.keyframe_id,
            state: propagate_state(previous.state(), preintegrated, self.gravity),
        })
    }

    pub fn latest_odometry_constraint(&self) -> Option<VioOdometryConstraint> {
        let current = self.frames.back()?;
        let preintegrated = current.preintegrated_from_prev.as_ref()?;
        let previous = self
            .frames
            .iter()
            .rev()
            .nth(1)?;
        let relative_pose = previous
            .state()
            .pose_odom_from_body()
            .inverse()
            .compose(current.state().pose_odom_from_body());
        Some(VioOdometryConstraint {
            from: previous.keyframe_id,
            to: current.keyframe_id,
            relative_pose,
            information: pose_information_from_preintegration(preintegrated),
        })
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn push_preintegrated(
        &mut self,
        keyframe_id: KeyframeId,
        preintegrated: PreintegratedImu,
        pose_measurement_odom: Pose64,
        visual_observations: Vec<Observation>,
    ) -> Result<VioEstimate, LocalVioError> {
        let previous = self.frames.back().ok_or(LocalVioError::NotInitialized)?;
        if self.frames.iter().any(|frame| frame.keyframe_id == keyframe_id) {
            return Err(LocalVioError::DuplicateKeyframe { keyframe_id });
        }
        let propagated = propagate_state(previous.state(), &preintegrated, self.gravity);
        self.frames.push_back(VioFrame {
            keyframe_id,
            state: propagated.clone(),
            pose_measurement_odom,
            visual_observations: visual_observations.into_boxed_slice(),
            preintegrated_from_prev: Some(preintegrated),
        });
        if self.frames.len() >= 2 {
            self.optimize_window();
        }
        while self.frames.len() > self.config.window_size() {
            self.frames.pop_front();
        }
        let state = self
            .frames
            .back()
            .map(|frame| frame.state.clone())
            .unwrap_or(propagated);
        Ok(VioEstimate {
            keyframe_id,
            state,
        })
    }

    fn optimize_window(&mut self) {
        let frame_count = self.frames.len();
        if frame_count < 2 {
            return;
        }
        let dim = (frame_count - 1) * STATE_DIM;
        if dim == 0 {
            return;
        }

        for _ in 0..self.config.max_iterations() {
            let frames = self.frames.make_contiguous();
            let mut h = vec![0.0_f64; dim * dim];
            let mut b = vec![0.0_f64; dim];

            for frame_idx in 1..frame_count {
                let measurement = frames[frame_idx].pose_measurement_odom;
                let residual = pose_prior_residual(&frames[frame_idx].state, measurement);
                let jac = numerical_pose_prior_jacobian(frames, frame_idx, measurement);
                let pose_prior_weights = [self.config.pose_prior_weight(); POSE_PRIOR_RESIDUAL_DIM];
                accumulate_self(
                    &mut h,
                    &mut b,
                    dim,
                    frame_base(frame_idx),
                    &jac,
                    &residual,
                    &pose_prior_weights,
                );

                let Some(preintegrated) = frames[frame_idx].preintegrated_from_prev.as_ref() else {
                    continue;
                };
                let residual = ImuFactor::residual(
                    &frames[frame_idx - 1].state,
                    &frames[frame_idx].state,
                    preintegrated,
                    &self.gravity,
                );
                let (jac_prev, jac_curr) = numerical_imu_jacobians(
                    frames,
                    frame_idx - 1,
                    frame_idx,
                    preintegrated,
                    self.gravity,
                );
                let imu_weights = preintegrated.residual_information_diag();
                if frame_idx > 1 {
                    accumulate_self(
                        &mut h,
                        &mut b,
                        dim,
                        frame_base(frame_idx - 1),
                        &jac_prev,
                        &residual,
                        &imu_weights,
                    );
                }
                accumulate_self(
                    &mut h,
                    &mut b,
                    dim,
                    frame_base(frame_idx),
                    &jac_curr,
                    &residual,
                    &imu_weights,
                );
                if frame_idx > 1 {
                    accumulate_cross(
                        &mut h,
                        dim,
                        frame_base(frame_idx - 1),
                        frame_base(frame_idx),
                        &jac_prev,
                        &jac_curr,
                        &imu_weights,
                    );
                }

                let bias_residual =
                    bias_random_walk_residual(&frames[frame_idx - 1].state, &frames[frame_idx].state);
                let (bias_jac_prev, bias_jac_curr) =
                    numerical_bias_random_walk_jacobians(frames, frame_idx - 1, frame_idx);
                let bias_weights = preintegrated.bias_random_walk_information_diag();
                if frame_idx > 1 {
                    accumulate_self(
                        &mut h,
                        &mut b,
                        dim,
                        frame_base(frame_idx - 1),
                        &bias_jac_prev,
                        &bias_residual,
                        &bias_weights,
                    );
                }
                accumulate_self(
                    &mut h,
                    &mut b,
                    dim,
                    frame_base(frame_idx),
                    &bias_jac_curr,
                    &bias_residual,
                    &bias_weights,
                );
                if frame_idx > 1 {
                    accumulate_cross(
                        &mut h,
                        dim,
                        frame_base(frame_idx - 1),
                        frame_base(frame_idx),
                        &bias_jac_prev,
                        &bias_jac_curr,
                        &bias_weights,
                    );
                }

                for observation in frames[frame_idx].visual_observations.iter().copied() {
                    let residual = match reprojection_residual(
                        &frames[frame_idx].state,
                        self.camera_from_body,
                        observation.world(),
                        observation.pixel(),
                        self.intrinsics,
                    ) {
                        Ok(residual) => residual,
                        Err(_) => continue,
                    };
                    let jac = numerical_reprojection_jacobian(
                        frames,
                        frame_idx,
                        self.camera_from_body,
                        observation,
                        self.intrinsics,
                    );
                    accumulate_self(
                        &mut h,
                        &mut b,
                        dim,
                        frame_base(frame_idx),
                        &jac,
                        &residual,
                        &[1.0; REPROJECTION_RESIDUAL_DIM],
                    );
                }
            }

            for idx in 0..dim {
                h[idx * dim + idx] += 1e-4;
            }
            let Some(delta) = solve_dense(h, b) else {
                return;
            };
            let step_norm = delta.iter().map(|value| value * value).sum::<f64>().sqrt();
            let frames = self.frames.make_contiguous();
            for frame_idx in 1..frame_count {
                let base = frame_base(frame_idx);
                let mut tangent = [0.0_f64; 15];
                tangent.copy_from_slice(&delta[base..base + STATE_DIM]);
                frames[frame_idx].state = frames[frame_idx].state.retract(&tangent);
            }
            if step_norm < 1e-8 {
                break;
            }
        }
    }
}

impl VioFrame {
    fn state(&self) -> &NavState {
        &self.state
    }
}

fn propagate_state(state_i: &NavState, preintegrated: &PreintegratedImu, gravity: Gravity) -> NavState {
    let pose_i = state_i.pose_odom_from_body();
    let r_i = pose_i.rotation();
    let p_i = pose_i.translation();
    let v_i = state_i.velocity_odom_mps();
    let dt = preintegrated.dt_seconds;
    let g = gravity.vector_odom_mps2();

    let delta_velocity_odom = mat_mul_vec_f64(r_i, preintegrated.delta_velocity);
    let delta_position_odom = mat_mul_vec_f64(r_i, preintegrated.delta_position);

    let velocity_j = [
        v_i[0] + g[0] * dt + delta_velocity_odom[0],
        v_i[1] + g[1] * dt + delta_velocity_odom[1],
        v_i[2] + g[2] * dt + delta_velocity_odom[2],
    ];
    let position_j = [
        p_i[0] + v_i[0] * dt + 0.5 * g[0] * dt * dt + delta_position_odom[0],
        p_i[1] + v_i[1] * dt + 0.5 * g[1] * dt * dt + delta_position_odom[1],
        p_i[2] + v_i[2] * dt + 0.5 * g[2] * dt * dt + delta_position_odom[2],
    ];
    let rotation_j = mat_mul_f64(r_i, preintegrated.delta_rotation);
    NavState::try_new(
        Pose64::from_rt(rotation_j, position_j),
        velocity_j,
        state_i.bias().clone(),
    )
    .expect("propagated state must stay finite")
}

fn pose_information_from_preintegration(preintegrated: &PreintegratedImu) -> [[f64; 6]; 6] {
    let mut information = [[0.0_f64; 6]; 6];
    for axis in 0..3 {
        let pos_var = preintegrated.covariance[6 + axis][6 + axis].max(1e-12);
        let rot_var = preintegrated.covariance[axis][axis].max(1e-12);
        information[axis][axis] = 1.0 / pos_var;
        information[3 + axis][3 + axis] = 1.0 / rot_var;
    }
    information
}

fn frame_base(frame_idx: usize) -> usize {
    (frame_idx - 1) * STATE_DIM
}

fn numerical_pose_prior_jacobian(
    frames: &[VioFrame],
    frame_idx: usize,
    measurement: Pose64,
) -> [[f64; STATE_DIM]; POSE_PRIOR_RESIDUAL_DIM] {
    let mut jacobian = [[0.0_f64; STATE_DIM]; POSE_PRIOR_RESIDUAL_DIM];
    for axis in 0..STATE_DIM {
        let delta = tangent_axis(axis, 1e-6);
        let plus = pose_prior_residual(&frames[frame_idx].state.retract(&delta), measurement);
        let minus = pose_prior_residual(&frames[frame_idx].state.retract(&neg_tangent(delta)), measurement);
        for row in 0..POSE_PRIOR_RESIDUAL_DIM {
            jacobian[row][axis] = (plus[row] - minus[row]) / (2.0 * 1e-6);
        }
    }
    jacobian
}

fn numerical_imu_jacobians(
    frames: &[VioFrame],
    prev_idx: usize,
    curr_idx: usize,
    preintegrated: &PreintegratedImu,
    gravity: Gravity,
) -> ([[f64; STATE_DIM]; IMU_RESIDUAL_DIM], [[f64; STATE_DIM]; IMU_RESIDUAL_DIM]) {
    let mut jac_prev = [[0.0_f64; STATE_DIM]; IMU_RESIDUAL_DIM];
    let mut jac_curr = [[0.0_f64; STATE_DIM]; IMU_RESIDUAL_DIM];
    for axis in 0..STATE_DIM {
        let delta = tangent_axis(axis, 1e-6);
        let plus = ImuFactor::residual(
            &frames[prev_idx].state.retract(&delta),
            &frames[curr_idx].state,
            preintegrated,
            &gravity,
        );
        let minus = ImuFactor::residual(
            &frames[prev_idx].state.retract(&neg_tangent(delta)),
            &frames[curr_idx].state,
            preintegrated,
            &gravity,
        );
        for row in 0..IMU_RESIDUAL_DIM {
            jac_prev[row][axis] = (plus[row] - minus[row]) / (2.0 * 1e-6);
        }
    }
    for axis in 0..STATE_DIM {
        let delta = tangent_axis(axis, 1e-6);
        let plus = ImuFactor::residual(
            &frames[prev_idx].state,
            &frames[curr_idx].state.retract(&delta),
            preintegrated,
            &gravity,
        );
        let minus = ImuFactor::residual(
            &frames[prev_idx].state,
            &frames[curr_idx].state.retract(&neg_tangent(delta)),
            preintegrated,
            &gravity,
        );
        for row in 0..IMU_RESIDUAL_DIM {
            jac_curr[row][axis] = (plus[row] - minus[row]) / (2.0 * 1e-6);
        }
    }
    (jac_prev, jac_curr)
}

fn numerical_bias_random_walk_jacobians(
    frames: &[VioFrame],
    prev_idx: usize,
    curr_idx: usize,
) -> (
    [[f64; STATE_DIM]; BIAS_RW_RESIDUAL_DIM],
    [[f64; STATE_DIM]; BIAS_RW_RESIDUAL_DIM],
) {
    let mut jac_prev = [[0.0_f64; STATE_DIM]; BIAS_RW_RESIDUAL_DIM];
    let mut jac_curr = [[0.0_f64; STATE_DIM]; BIAS_RW_RESIDUAL_DIM];
    for axis in 0..STATE_DIM {
        let delta = tangent_axis(axis, 1e-6);
        let plus = bias_random_walk_residual(
            &frames[prev_idx].state.retract(&delta),
            &frames[curr_idx].state,
        );
        let minus = bias_random_walk_residual(
            &frames[prev_idx].state.retract(&neg_tangent(delta)),
            &frames[curr_idx].state,
        );
        for row in 0..BIAS_RW_RESIDUAL_DIM {
            jac_prev[row][axis] = (plus[row] - minus[row]) / (2.0 * 1e-6);
        }
    }
    for axis in 0..STATE_DIM {
        let delta = tangent_axis(axis, 1e-6);
        let plus = bias_random_walk_residual(
            &frames[prev_idx].state,
            &frames[curr_idx].state.retract(&delta),
        );
        let minus = bias_random_walk_residual(
            &frames[prev_idx].state,
            &frames[curr_idx].state.retract(&neg_tangent(delta)),
        );
        for row in 0..BIAS_RW_RESIDUAL_DIM {
            jac_curr[row][axis] = (plus[row] - minus[row]) / (2.0 * 1e-6);
        }
    }
    (jac_prev, jac_curr)
}

fn numerical_reprojection_jacobian(
    frames: &[VioFrame],
    frame_idx: usize,
    camera_from_body: Pose64,
    observation: Observation,
    intrinsics: PinholeIntrinsics,
) -> [[f64; STATE_DIM]; REPROJECTION_RESIDUAL_DIM] {
    let mut jacobian = [[0.0_f64; STATE_DIM]; REPROJECTION_RESIDUAL_DIM];
    for axis in 0..STATE_DIM {
        let delta = tangent_axis(axis, 1e-6);
        let plus = reprojection_residual(
            &frames[frame_idx].state.retract(&delta),
            camera_from_body,
            observation.world(),
            observation.pixel(),
            intrinsics,
        )
        .unwrap_or([0.0, 0.0]);
        let minus = reprojection_residual(
            &frames[frame_idx].state.retract(&neg_tangent(delta)),
            camera_from_body,
            observation.world(),
            observation.pixel(),
            intrinsics,
        )
        .unwrap_or([0.0, 0.0]);
        for row in 0..REPROJECTION_RESIDUAL_DIM {
            jacobian[row][axis] = (plus[row] - minus[row]) / (2.0 * 1e-6);
        }
    }
    jacobian
}

fn tangent_axis(axis: usize, magnitude: f64) -> NavTangent {
    let mut tangent = [0.0_f64; 15];
    tangent[axis] = magnitude;
    tangent
}

fn neg_tangent(mut tangent: NavTangent) -> NavTangent {
    for value in &mut tangent {
        *value = -*value;
    }
    tangent
}

fn accumulate_self(
    h: &mut [f64],
    b: &mut [f64],
    dim: usize,
    base: usize,
    jacobian: &[[f64; STATE_DIM]],
    residual: &[f64],
    row_weights: &[f64],
) {
    for i in 0..STATE_DIM {
        for j in 0..STATE_DIM {
            let mut value = 0.0_f64;
            for row in 0..residual.len() {
                value += jacobian[row][i] * jacobian[row][j] * row_weights[row];
            }
            h[(base + i) * dim + (base + j)] += value;
        }
        let mut rhs = 0.0_f64;
        for row in 0..residual.len() {
            rhs += jacobian[row][i] * residual[row] * row_weights[row];
        }
        b[base + i] += rhs;
    }
}

fn accumulate_cross(
    h: &mut [f64],
    dim: usize,
    base_a: usize,
    base_b: usize,
    jac_a: &[[f64; STATE_DIM]],
    jac_b: &[[f64; STATE_DIM]],
    row_weights: &[f64],
) {
    let rows = jac_a.len().min(jac_b.len());
    for i in 0..STATE_DIM {
        for j in 0..STATE_DIM {
            let mut value = 0.0_f64;
            for row in 0..rows {
                value += jac_a[row][i] * jac_b[row][j] * row_weights[row];
            }
            h[(base_a + i) * dim + (base_b + j)] += value;
            h[(base_b + j) * dim + (base_a + i)] += value;
        }
    }
}

fn solve_dense(mut h: Vec<f64>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for i in 0..n {
        let mut pivot = i;
        let mut pivot_abs = h[i * n + i].abs();
        for row in (i + 1)..n {
            let value = h[row * n + i].abs();
            if value > pivot_abs {
                pivot = row;
                pivot_abs = value;
            }
        }
        if !pivot_abs.is_finite() || pivot_abs < 1e-12 {
            return None;
        }
        if pivot != i {
            for col in 0..n {
                h.swap(i * n + col, pivot * n + col);
            }
            b.swap(i, pivot);
        }
        let diag = h[i * n + i];
        for row in (i + 1)..n {
            let factor = h[row * n + i] / diag;
            h[row * n + i] = 0.0;
            for col in (i + 1)..n {
                h[row * n + col] -= factor * h[i * n + col];
            }
            b[row] -= factor * b[i];
        }
    }
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for col in (i + 1)..n {
            sum -= h[i * n + col] * x[col];
        }
        x[i] = -sum / h[i * n + i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::SlamMap;
    use crate::{
        Detections, Descriptor, FrameId, ImuBatch, ImuBias, ImuNoiseModel, ImuSample, Keypoint,
        Observation, PinholeIntrinsics, Pose, SensorId, Timestamp,
    };

    fn noise() -> ImuNoiseModel {
        ImuNoiseModel::new(0.1, 0.01, 0.001, 0.0001).expect("noise")
    }

    fn intrinsics() -> PinholeIntrinsics {
        crate::test_helpers::make_pinhole_intrinsics(100, 80, 100.0, 100.0, 50.0, 40.0)
            .expect("intrinsics")
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

    fn keyframe_ids(count: usize) -> Vec<KeyframeId> {
        let mut map = SlamMap::new();
        let mut ids = Vec::with_capacity(count);
        for idx in 0..count {
            let detections = Detections::new(
                SensorId::StereoLeft,
                FrameId::new(idx as u64),
                2,
                2,
                vec![Keypoint { x: 0.0, y: 0.0 }],
                vec![1.0],
                vec![Descriptor([0.0; 256])],
            )
            .expect("detections");
            let keyframe_id = map
                .add_keyframe_from_detections(
                    &detections,
                    Timestamp::from_nanos(idx as i64),
                    Pose::identity(),
                )
                .expect("keyframe");
            ids.push(keyframe_id);
        }
        ids
    }

    #[test]
    fn config_rejects_zero_window() {
        let err = VioConfig::new(0).expect_err("zero window should fail");
        assert_eq!(err, VioConfigError::ZeroWindowSize);
    }

    #[test]
    fn local_vio_requires_initialization() {
        let mut vio = LocalVio::new(
            VioConfig::new(3).expect("config"),
            Gravity::try_new([0.0, 0.0, -9.81]).expect("gravity"),
            Pose64::identity(),
            intrinsics(),
        );
        let ids = keyframe_ids(2);
        let preintegrated = PreintegratedImu::integrate(
            &batch(&[
                (0, [0.0; 3], [0.0; 3]),
                (10_000_000, [0.0; 3], [0.0; 3]),
            ]),
            &ImuBias::default(),
            &noise(),
        )
        .expect("preintegrated");
        let err = vio
            .push_preintegrated(ids[1], preintegrated, Pose64::identity(), Vec::new())
            .expect_err("push without init should fail");
        assert_eq!(err, LocalVioError::NotInitialized);
    }

    #[test]
    fn local_vio_propagates_free_fall_state() {
        let gravity = Gravity::try_new([0.0, 0.0, -9.81]).expect("gravity");
        let mut vio =
            LocalVio::new(VioConfig::new(4).expect("config"), gravity, Pose64::identity(), intrinsics());
        let ids = keyframe_ids(2);
        vio.initialize(
            ids[0],
            NavState::try_new(Pose64::identity(), [0.0; 3], ImuBias::default()).expect("state"),
            Pose64::identity(),
            Vec::new(),
        )
        .expect("init");
        let preintegrated = PreintegratedImu::integrate(
            &batch(&[
                (0, [0.0; 3], [0.0; 3]),
                (10_000_000, [0.0; 3], [0.0; 3]),
                (20_000_000, [0.0; 3], [0.0; 3]),
            ]),
            &ImuBias::default(),
            &noise(),
        )
        .expect("preintegrated");
        let expected_pose = Pose64::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0, 0.0, -0.5 * 9.81 * 0.02 * 0.02],
        );
        let estimate = vio
            .push_preintegrated(ids[1], preintegrated, expected_pose, Vec::new())
            .expect("propagate");
        let position = estimate.state().pose_odom_from_body().translation();
        let velocity = estimate.state().velocity_odom_mps();
        assert!((position[2] + 0.5 * 9.81 * 0.02 * 0.02).abs() < 1e-9);
        assert!((velocity[2] + 9.81 * 0.02).abs() < 1e-9);
    }

    #[test]
    fn local_vio_caps_window_size() {
        let gravity = Gravity::try_new([0.0, 0.0, -9.81]).expect("gravity");
        let mut vio =
            LocalVio::new(VioConfig::new(2).expect("config"), gravity, Pose64::identity(), intrinsics());
        let ids = keyframe_ids(4);
        vio.initialize(
            ids[0],
            NavState::try_new(Pose64::identity(), [0.0; 3], ImuBias::default()).expect("state"),
            Pose64::identity(),
            Vec::new(),
        )
        .expect("init");
        for idx in 1..=3 {
            let preintegrated = PreintegratedImu::integrate(
                &batch(&[
                    (0, [0.0; 3], [0.0; 3]),
                    (10_000_000, [0.0; 3], [0.0; 3]),
                ]),
                &ImuBias::default(),
                &noise(),
            )
            .expect("preintegrated");
            vio.push_preintegrated(ids[idx], preintegrated, Pose64::identity(), Vec::new())
                .expect("push");
        }
        assert_eq!(vio.len(), 2);
        assert_eq!(vio.latest_estimate().expect("latest").keyframe_id(), ids[3]);
    }

    #[test]
    fn latest_odometry_constraint_matches_latest_pair() {
        let gravity = Gravity::try_new([0.0, 0.0, -9.81]).expect("gravity");
        let mut vio =
            LocalVio::new(VioConfig::new(4).expect("config"), gravity, Pose64::identity(), intrinsics());
        let ids = keyframe_ids(2);
        vio.initialize(
            ids[0],
            NavState::try_new(Pose64::identity(), [0.0; 3], ImuBias::default()).expect("state"),
            Pose64::identity(),
            Vec::new(),
        )
        .expect("init");
        let preintegrated = PreintegratedImu::integrate(
            &batch(&[
                (0, [0.0; 3], [0.0; 3]),
                (10_000_000, [0.0; 3], [0.0; 3]),
                (20_000_000, [0.0; 3], [0.0; 3]),
            ]),
            &ImuBias::default(),
            &noise(),
        )
        .expect("preintegrated");
        vio.push_preintegrated(ids[1], preintegrated, Pose64::identity(), Vec::new())
            .expect("push");
        let constraint = vio.latest_odometry_constraint().expect("constraint");
        assert_eq!(constraint.from(), ids[0]);
        assert_eq!(constraint.to(), ids[1]);
        assert!(constraint.information()[0][0].is_finite());
        assert!(constraint.information()[3][3].is_finite());
    }

    #[test]
    fn local_vio_solver_respects_pose_measurement() {
        let gravity = Gravity::try_new([0.0, 0.0, -9.81]).expect("gravity");
        let config = VioConfig::new(4)
            .expect("config")
            .with_pose_prior_weight(1.0e9)
            .expect("pose prior weight");
        let mut vio = LocalVio::new(config, gravity, Pose64::identity(), intrinsics());
        let ids = keyframe_ids(2);
        vio.initialize(
            ids[0],
            NavState::try_new(Pose64::identity(), [0.0; 3], ImuBias::default()).expect("state"),
            Pose64::identity(),
            Vec::new(),
        )
        .expect("init");
        let preintegrated = PreintegratedImu::integrate(
            &batch(&[
                (0, [0.0; 3], [0.0; 3]),
                (10_000_000, [0.0; 3], [0.0; 3]),
            ]),
            &ImuBias::default(),
            &noise(),
        )
        .expect("preintegrated");
        let measurement = Pose64::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [1.0, 0.0, 0.0],
        );
        let estimate = vio
            .push_preintegrated(ids[1], preintegrated, measurement, Vec::new())
            .expect("push");
        let x = estimate.state().pose_odom_from_body().translation()[0];
        assert!(x > 0.01, "optimized x should move toward measurement, got {x}");
    }

    #[test]
    fn local_vio_solver_uses_visual_observations() {
        let gravity = Gravity::try_new([0.0, 0.0, -9.81]).expect("gravity");
        let intrinsics = intrinsics();
        let mut vio =
            LocalVio::new(VioConfig::new(4).expect("config"), gravity, Pose64::identity(), intrinsics);
        let ids = keyframe_ids(2);
        vio.initialize(
            ids[0],
            NavState::try_new(Pose64::identity(), [0.0; 3], ImuBias::default()).expect("state"),
            Pose64::identity(),
            Vec::new(),
        )
        .expect("init");
        let preintegrated = PreintegratedImu::integrate(
            &batch(&[
                (0, [0.0; 3], [0.0; 3]),
                (10_000_000, [0.0; 3], [0.0; 3]),
            ]),
            &ImuBias::default(),
            &noise(),
        )
        .expect("preintegrated");
        let observation = Observation::try_new(
            crate::Point3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            Keypoint { x: 50.0, y: 40.0 },
            intrinsics,
        )
        .expect("observation");
        let estimate = vio
            .push_preintegrated(ids[1], preintegrated, Pose64::identity(), vec![observation])
            .expect("push");
        let z = estimate.state().pose_odom_from_body().translation()[2];
        assert!(z.abs() < 1e-3, "visual observation should keep translation near origin, got {z}");
    }

    #[test]
    fn local_vio_solver_preserves_bias_state_through_window_updates() {
        let gravity = Gravity::try_new([0.0, 0.0, -9.81]).expect("gravity");
        let mut vio =
            LocalVio::new(VioConfig::new(4).expect("config"), gravity, Pose64::identity(), intrinsics());
        let ids = keyframe_ids(3);
        let bias = ImuBias {
            accel: [0.02, -0.01, 0.005],
            gyro: [0.001, -0.002, 0.003],
        };
        vio.initialize(
            ids[0],
            NavState::try_new(Pose64::identity(), [0.0; 3], bias.clone()).expect("state"),
            Pose64::identity(),
            Vec::new(),
        )
        .expect("init");
        for keyframe_id in [ids[1], ids[2]] {
            let preintegrated = PreintegratedImu::integrate(
                &batch(&[
                    (0, bias.accel, bias.gyro),
                    (10_000_000, bias.accel, bias.gyro),
                    (20_000_000, bias.accel, bias.gyro),
                ]),
                &bias,
                &noise(),
            )
            .expect("preintegrated");
            vio.push_preintegrated(keyframe_id, preintegrated, Pose64::identity(), Vec::new())
                .expect("push");
        }
        let latest = vio.latest_estimate().expect("latest");
        let latest_bias = latest.state().bias();
        for axis in 0..3 {
            assert!(
                (latest_bias.accel[axis] - bias.accel[axis]).abs() < 1e-5,
                "accel bias axis {axis} drifted: {} vs {}",
                latest_bias.accel[axis],
                bias.accel[axis]
            );
            assert!(
                (latest_bias.gyro[axis] - bias.gyro[axis]).abs() < 1e-6,
                "gyro bias axis {axis} drifted: {} vs {}",
                latest_bias.gyro[axis],
                bias.gyro[axis]
            );
        }
    }
}
