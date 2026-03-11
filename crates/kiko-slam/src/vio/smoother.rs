use std::collections::VecDeque;
use std::num::NonZeroUsize;

use crate::map::KeyframeId;
use crate::math::{mat_mul_f64, mat_mul_vec_f64};
use crate::{
    bias_random_walk_residual, pose_prior_residual, reprojection_residual, Gravity, ImuFactor,
    MapFromOdom, NavState, NavTangent, PinholeIntrinsics, Pose64, PreintegratedImu, VioObservation,
};

const STATE_DIM: usize = 15;
const IMU_RESIDUAL_DIM: usize = 9;
const POSE_PRIOR_RESIDUAL_DIM: usize = 6;
const REPROJECTION_RESIDUAL_DIM: usize = 2;
const BIAS_RW_RESIDUAL_DIM: usize = 6;
type StateMatrix = [[f64; STATE_DIM]; STATE_DIM];
type BoundaryHessianBlocks = (StateMatrix, StateMatrix, StateMatrix, StateMatrix);

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
struct MarginalPrior {
    applies_to: KeyframeId,
    reference: NavState,
    information: [[f64; STATE_DIM]; STATE_DIM],
}

#[derive(Clone, Debug)]
struct VioFrame {
    keyframe_id: KeyframeId,
    state: NavState,
    pose_measurement_odom: Pose64,
    visual_observations: Box<[VioObservation]>,
    #[allow(dead_code)]
    preintegrated_from_prev: Option<PreintegratedImu>,
}

pub struct LocalVio {
    config: VioConfig,
    gravity: Gravity,
    camera_from_body: Pose64,
    intrinsics: PinholeIntrinsics,
    map_from_odom: MapFromOdom,
    anchor_prior: Option<MarginalPrior>,
    marginal_prior: Option<MarginalPrior>,
    pending_exported_odometry: VecDeque<VioOdometryConstraint>,
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
            map_from_odom: MapFromOdom::identity(),
            anchor_prior: None,
            marginal_prior: None,
            pending_exported_odometry: VecDeque::new(),
            frames: VecDeque::new(),
        }
    }

    pub fn set_map_from_odom(&mut self, map_from_odom: MapFromOdom) {
        self.map_from_odom = map_from_odom;
    }

    pub fn set_gravity(&mut self, gravity: Gravity) {
        self.gravity = gravity;
    }

    pub fn initialize(
        &mut self,
        keyframe_id: KeyframeId,
        state: NavState,
        pose_measurement_odom: Pose64,
        visual_observations: Vec<VioObservation>,
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
        self.anchor_prior = Some(MarginalPrior::anchor(
            keyframe_id,
            self.frames[0].state.clone(),
        ));
        self.marginal_prior = None;
        self.pending_exported_odometry.clear();
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

    pub fn correct_prediction(
        &self,
        preintegrated: &PreintegratedImu,
        pose_measurement_odom: Pose64,
        visual_observations: Vec<VioObservation>,
    ) -> Result<VioEstimate, LocalVioError> {
        let previous = self.frames.back().ok_or(LocalVioError::NotInitialized)?;
        let predicted = propagate_state(previous.state(), preintegrated, self.gravity);
        let corrected = self.optimize_predicted_state(
            previous,
            predicted,
            pose_measurement_odom,
            visual_observations.into_boxed_slice(),
            preintegrated,
        );
        Ok(VioEstimate {
            keyframe_id: previous.keyframe_id,
            state: corrected,
        })
    }

    pub fn drain_exported_odometry(&mut self) -> Vec<VioOdometryConstraint> {
        self.pending_exported_odometry.drain(..).collect()
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
        visual_observations: Vec<VioObservation>,
    ) -> Result<VioEstimate, LocalVioError> {
        let previous = self.frames.back().ok_or(LocalVioError::NotInitialized)?;
        if self
            .frames
            .iter()
            .any(|frame| frame.keyframe_id == keyframe_id)
        {
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
            self.rollover_oldest();
        }
        let state = self
            .frames
            .back()
            .map(|frame| frame.state.clone())
            .unwrap_or(propagated);
        Ok(VioEstimate { keyframe_id, state })
    }

    fn rollover_oldest(&mut self) {
        let (Some(oldest), Some(next)) =
            (self.frames.front().cloned(), self.frames.get(1).cloned())
        else {
            return;
        };
        let Some(preintegrated) = next.preintegrated_from_prev.as_ref() else {
            self.frames.pop_front();
            return;
        };
        let prior_for_oldest = self.prior_for_oldest().cloned();
        if let Some(exported) = exported_odometry_constraint(
            &oldest,
            &next,
            preintegrated,
            prior_for_oldest.as_ref(),
            self.gravity,
            self.camera_from_body,
            self.intrinsics,
            &self.map_from_odom,
        ) {
            self.pending_exported_odometry.push_back(exported);
        }
        self.marginal_prior = marginalize_oldest_to_next(
            &oldest,
            &next,
            preintegrated,
            prior_for_oldest.as_ref(),
            self.gravity,
            self.camera_from_body,
            self.intrinsics,
            &self.map_from_odom,
        );
        if self
            .anchor_prior
            .as_ref()
            .is_some_and(|prior| prior.applies_to == oldest.keyframe_id)
        {
            self.anchor_prior = None;
        }
        self.frames.pop_front();
        if let Some(new_oldest) = self.frames.front_mut() {
            new_oldest.preintegrated_from_prev = None;
        }
    }

    fn optimize_predicted_state(
        &self,
        previous: &VioFrame,
        mut state: NavState,
        pose_measurement_odom: Pose64,
        visual_observations: Box<[VioObservation]>,
        preintegrated: &PreintegratedImu,
    ) -> NavState {
        let pose_prior_weights = [self.config.pose_prior_weight(); POSE_PRIOR_RESIDUAL_DIM];
        let imu_weights = preintegrated.residual_information_diag();
        let bias_weights = preintegrated.bias_random_walk_information_diag();
        let reprojection_weights = [1.0; REPROJECTION_RESIDUAL_DIM];

        for _ in 0..self.config.max_iterations() {
            let frames = [
                previous.clone(),
                VioFrame {
                    keyframe_id: previous.keyframe_id,
                    state: state.clone(),
                    pose_measurement_odom,
                    visual_observations: visual_observations.clone(),
                    preintegrated_from_prev: Some(preintegrated.clone()),
                },
            ];
            let mut h = vec![0.0_f64; STATE_DIM * STATE_DIM];
            let mut b = vec![0.0_f64; STATE_DIM];

            let pose_residual = pose_prior_residual(&frames[1].state, pose_measurement_odom);
            let pose_jac = numerical_pose_prior_jacobian(&frames, 1, pose_measurement_odom);
            accumulate_self(
                &mut h,
                &mut b,
                STATE_DIM,
                0,
                &pose_jac,
                &pose_residual,
                &pose_prior_weights,
            );

            let imu_residual = ImuFactor::residual(
                &frames[0].state,
                &frames[1].state,
                preintegrated,
                &self.gravity,
            );
            let (_, imu_jac_curr) =
                numerical_imu_jacobians(&frames, 0, 1, preintegrated, self.gravity);
            accumulate_self(
                &mut h,
                &mut b,
                STATE_DIM,
                0,
                &imu_jac_curr,
                &imu_residual,
                &imu_weights,
            );

            let bias_residual = bias_random_walk_residual(&frames[0].state, &frames[1].state);
            let (_, bias_jac_curr) = numerical_bias_random_walk_jacobians(&frames, 0, 1);
            accumulate_self(
                &mut h,
                &mut b,
                STATE_DIM,
                0,
                &bias_jac_curr,
                &bias_residual,
                &bias_weights,
            );

            for observation in frames[1].visual_observations.iter().copied() {
                let residual = match reprojection_residual(
                    &frames[1].state,
                    self.camera_from_body,
                    &self.map_from_odom,
                    observation,
                    self.intrinsics,
                ) {
                    Ok(residual) => residual,
                    Err(_) => continue,
                };
                let jac = numerical_reprojection_jacobian(
                    &frames,
                    1,
                    self.camera_from_body,
                    &self.map_from_odom,
                    observation,
                    self.intrinsics,
                );
                accumulate_self(
                    &mut h,
                    &mut b,
                    STATE_DIM,
                    0,
                    &jac,
                    &residual,
                    &reprojection_weights,
                );
            }

            for idx in 0..STATE_DIM {
                h[idx * STATE_DIM + idx] += 1e-4;
            }
            let Some(delta) = solve_dense(h, b) else {
                return state;
            };
            let step_norm = delta.iter().map(|value| value * value).sum::<f64>().sqrt();
            let mut tangent = [0.0_f64; STATE_DIM];
            tangent.copy_from_slice(&delta[..STATE_DIM]);
            state = state.retract(&tangent);
            if step_norm < 1e-8 {
                break;
            }
        }
        state
    }

    fn optimize_window(&mut self) {
        let frame_count = self.frames.len();
        if frame_count < 2 {
            return;
        }
        let dim = frame_count * STATE_DIM;
        if dim == 0 {
            return;
        }

        for _ in 0..self.config.max_iterations() {
            let prior_for_oldest = self.prior_for_oldest().cloned();
            let frames = self.frames.make_contiguous();
            let mut h = vec![0.0_f64; dim * dim];
            let mut b = vec![0.0_f64; dim];
            if let Some(prior) = prior_for_oldest.as_ref() {
                let residual = prior.reference.local_coordinates(&frames[0].state);
                let jac = numerical_state_prior_jacobian(frames, 0, &prior.reference);
                accumulate_self_information(
                    &mut h,
                    &mut b,
                    dim,
                    frame_base(0),
                    &jac,
                    &residual,
                    &prior.information,
                );
            }

            for frame_idx in 0..frame_count {
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
                if frame_idx == 0 {
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

                let bias_residual = bias_random_walk_residual(
                    &frames[frame_idx - 1].state,
                    &frames[frame_idx].state,
                );
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
                        &self.map_from_odom,
                        observation,
                        self.intrinsics,
                    ) {
                        Ok(residual) => residual,
                        Err(_) => continue,
                    };
                    let jac = numerical_reprojection_jacobian(
                        frames,
                        frame_idx,
                        self.camera_from_body,
                        &self.map_from_odom,
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
            for (frame_idx, frame) in frames.iter_mut().enumerate().take(frame_count) {
                let base = frame_base(frame_idx);
                let mut tangent = [0.0_f64; 15];
                tangent.copy_from_slice(&delta[base..base + STATE_DIM]);
                frame.state = frame.state.retract(&tangent);
            }
            if step_norm < 1e-8 {
                break;
            }
        }
    }

    fn prior_for_oldest(&self) -> Option<&MarginalPrior> {
        let oldest = self.frames.front()?;
        if let Some(prior) = self
            .marginal_prior
            .as_ref()
            .filter(|prior| prior.applies_to == oldest.keyframe_id)
        {
            return Some(prior);
        }
        self.anchor_prior
            .as_ref()
            .filter(|prior| prior.applies_to == oldest.keyframe_id)
    }
}

impl VioFrame {
    fn state(&self) -> &NavState {
        &self.state
    }
}

impl MarginalPrior {
    fn anchor(applies_to: KeyframeId, reference: NavState) -> Self {
        let mut information = [[0.0_f64; STATE_DIM]; STATE_DIM];
        for (axis, row) in information.iter_mut().enumerate().take(STATE_DIM) {
            row[axis] = 1.0e6;
        }
        Self {
            applies_to,
            reference,
            information,
        }
    }
}

fn propagate_state(
    state_i: &NavState,
    preintegrated: &PreintegratedImu,
    gravity: Gravity,
) -> NavState {
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

fn relative_pose_jacobian_pair(previous: &NavState, current: &NavState) -> [[f64; 30]; 6] {
    let nominal = previous
        .pose_odom_from_body()
        .inverse()
        .compose(current.pose_odom_from_body());
    let mut jacobian = [[0.0_f64; 30]; 6];
    for axis in 0..30 {
        let delta = tangent_axis(axis % STATE_DIM, 1e-6);
        let (plus_prev, plus_curr) = if axis < STATE_DIM {
            (previous.retract(&delta), current.clone())
        } else {
            (previous.clone(), current.retract(&delta))
        };
        let (minus_prev, minus_curr) = if axis < STATE_DIM {
            (previous.retract(&neg_tangent(delta)), current.clone())
        } else {
            (previous.clone(), current.retract(&neg_tangent(delta)))
        };
        let plus_pose = plus_prev
            .pose_odom_from_body()
            .inverse()
            .compose(plus_curr.pose_odom_from_body());
        let minus_pose = minus_prev
            .pose_odom_from_body()
            .inverse()
            .compose(minus_curr.pose_odom_from_body());
        let plus = relative_pose_residual(nominal, plus_pose);
        let minus = relative_pose_residual(nominal, minus_pose);
        for row in 0..6 {
            jacobian[row][axis] = (plus[row] - minus[row]) / (2.0 * 1e-6);
        }
    }
    jacobian
}

fn relative_pose_residual(target: Pose64, estimate: Pose64) -> [f64; 6] {
    let delta = target.inverse().compose(estimate);
    let rot = crate::math::so3_log_f64(delta.rotation());
    let t = delta.translation();
    [t[0], t[1], t[2], rot[0], rot[1], rot[2]]
}

fn transpose_rect<const ROWS: usize, const COLS: usize>(
    matrix: &[[f64; COLS]; ROWS],
) -> [[f64; ROWS]; COLS] {
    let mut out = [[0.0_f64; ROWS]; COLS];
    for (row_idx, row) in matrix.iter().enumerate() {
        for (col_idx, value) in row.iter().copied().enumerate() {
            out[col_idx][row_idx] = value;
        }
    }
    out
}

fn matmul_rect<const M: usize, const N: usize, const K: usize>(
    a: &[[f64; N]; M],
    b: &[[f64; K]; N],
    c: &[[f64; M]; K],
) -> [[f64; M]; M] {
    let mut temp = [[0.0_f64; K]; M];
    for i in 0..M {
        for j in 0..K {
            let mut value = 0.0_f64;
            for (k, row) in b.iter().enumerate().take(N) {
                value += a[i][k] * row[j];
            }
            temp[i][j] = value;
        }
    }
    let mut out = [[0.0_f64; M]; M];
    for i in 0..M {
        for j in 0..M {
            let mut value = 0.0_f64;
            for (k, row) in c.iter().enumerate().take(K) {
                value += temp[i][k] * row[j];
            }
            out[i][j] = value;
        }
    }
    out
}

fn invert_dense_square(matrix: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    let mut a = matrix;
    let mut inv = vec![0.0_f64; n * n];
    for idx in 0..n {
        inv[idx * n + idx] = 1.0;
    }

    for pivot in 0..n {
        let mut pivot_row = pivot;
        let mut pivot_abs = a[pivot * n + pivot].abs();
        for row in (pivot + 1)..n {
            let value = a[row * n + pivot].abs();
            if value > pivot_abs {
                pivot_abs = value;
                pivot_row = row;
            }
        }
        if !pivot_abs.is_finite() || pivot_abs < 1e-12 {
            return None;
        }
        if pivot_row != pivot {
            for col in 0..n {
                a.swap(pivot * n + col, pivot_row * n + col);
                inv.swap(pivot * n + col, pivot_row * n + col);
            }
        }
        let diag = a[pivot * n + pivot];
        for col in 0..n {
            a[pivot * n + col] /= diag;
            inv[pivot * n + col] /= diag;
        }
        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = a[row * n + pivot];
            if factor == 0.0 {
                continue;
            }
            for col in 0..n {
                a[row * n + col] -= factor * a[pivot * n + col];
                inv[row * n + col] -= factor * inv[pivot * n + col];
            }
        }
    }
    Some(inv)
}

fn invert_dense_6x6(matrix: [[f64; 6]; 6]) -> Option<[[f64; 6]; 6]> {
    let flat = matrix
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    let inverse = invert_dense_square(flat, 6)?;
    let mut out = [[0.0_f64; 6]; 6];
    for row in 0..6 {
        for col in 0..6 {
            out[row][col] = inverse[row * 6 + col];
        }
    }
    Some(out)
}

fn invert_dense_15x15(matrix: Vec<f64>) -> Option<[[f64; STATE_DIM]; STATE_DIM]> {
    let inverse = invert_dense_square(matrix, STATE_DIM)?;
    let mut out = [[0.0_f64; STATE_DIM]; STATE_DIM];
    for row in 0..STATE_DIM {
        for col in 0..STATE_DIM {
            out[row][col] = inverse[row * STATE_DIM + col];
        }
    }
    Some(out)
}

fn invert_dense_30x30(matrix: Vec<f64>) -> Option<[[f64; 30]; 30]> {
    let inverse = invert_dense_square(matrix, 30)?;
    let mut out = [[0.0_f64; 30]; 30];
    for row in 0..30 {
        for col in 0..30 {
            out[row][col] = inverse[row * 30 + col];
        }
    }
    Some(out)
}

fn split_boundary_hessian(h: [[f64; 30]; 30]) -> BoundaryHessianBlocks {
    let mut h00 = [[0.0_f64; STATE_DIM]; STATE_DIM];
    let mut h01 = [[0.0_f64; STATE_DIM]; STATE_DIM];
    let mut h10 = [[0.0_f64; STATE_DIM]; STATE_DIM];
    let mut h11 = [[0.0_f64; STATE_DIM]; STATE_DIM];
    for row in 0..STATE_DIM {
        for col in 0..STATE_DIM {
            h00[row][col] = h[row][col];
            h01[row][col] = h[row][STATE_DIM + col];
            h10[row][col] = h[STATE_DIM + row][col];
            h11[row][col] = h[STATE_DIM + row][STATE_DIM + col];
        }
    }
    (h00, h01, h10, h11)
}

fn matmul_15x15(
    a: [[f64; STATE_DIM]; STATE_DIM],
    b: [[f64; STATE_DIM]; STATE_DIM],
) -> [[f64; STATE_DIM]; STATE_DIM] {
    let mut out = [[0.0_f64; STATE_DIM]; STATE_DIM];
    for i in 0..STATE_DIM {
        for j in 0..STATE_DIM {
            let mut value = 0.0_f64;
            for (k, row) in b.iter().enumerate().take(STATE_DIM) {
                value += a[i][k] * row[j];
            }
            out[i][j] = value;
        }
    }
    out
}

fn subtract_15x15(
    a: [[f64; STATE_DIM]; STATE_DIM],
    b: [[f64; STATE_DIM]; STATE_DIM],
) -> [[f64; STATE_DIM]; STATE_DIM] {
    let mut out = [[0.0_f64; STATE_DIM]; STATE_DIM];
    for i in 0..STATE_DIM {
        for j in 0..STATE_DIM {
            out[i][j] = a[i][j] - b[i][j];
        }
    }
    out
}

fn flatten_square<const N: usize>(matrix: [[f64; N]; N]) -> Vec<f64> {
    matrix
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>()
}

fn frame_base(frame_idx: usize) -> usize {
    frame_idx * STATE_DIM
}

fn numerical_state_prior_jacobian(
    frames: &[VioFrame],
    frame_idx: usize,
    reference: &NavState,
) -> [[f64; STATE_DIM]; STATE_DIM] {
    let mut jacobian = [[0.0_f64; STATE_DIM]; STATE_DIM];
    for axis in 0..STATE_DIM {
        let delta = tangent_axis(axis, 1e-6);
        let plus = reference.local_coordinates(&frames[frame_idx].state.retract(&delta));
        let minus =
            reference.local_coordinates(&frames[frame_idx].state.retract(&neg_tangent(delta)));
        for row in 0..STATE_DIM {
            jacobian[row][axis] = (plus[row] - minus[row]) / (2.0 * 1e-6);
        }
    }
    jacobian
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
        let minus = pose_prior_residual(
            &frames[frame_idx].state.retract(&neg_tangent(delta)),
            measurement,
        );
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
) -> (
    [[f64; STATE_DIM]; IMU_RESIDUAL_DIM],
    [[f64; STATE_DIM]; IMU_RESIDUAL_DIM],
) {
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
    map_from_odom: &MapFromOdom,
    observation: VioObservation,
    intrinsics: PinholeIntrinsics,
) -> [[f64; STATE_DIM]; REPROJECTION_RESIDUAL_DIM] {
    let mut jacobian = [[0.0_f64; STATE_DIM]; REPROJECTION_RESIDUAL_DIM];
    for axis in 0..STATE_DIM {
        let delta = tangent_axis(axis, 1e-6);
        let plus = reprojection_residual(
            &frames[frame_idx].state.retract(&delta),
            camera_from_body,
            map_from_odom,
            observation,
            intrinsics,
        )
        .unwrap_or([0.0, 0.0]);
        let minus = reprojection_residual(
            &frames[frame_idx].state.retract(&neg_tangent(delta)),
            camera_from_body,
            map_from_odom,
            observation,
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

fn accumulate_self_information<const RESIDUAL_DIM: usize>(
    h: &mut [f64],
    b: &mut [f64],
    dim: usize,
    base: usize,
    jacobian: &[[f64; STATE_DIM]; RESIDUAL_DIM],
    residual: &[f64; RESIDUAL_DIM],
    information: &[[f64; RESIDUAL_DIM]; RESIDUAL_DIM],
) {
    for i in 0..STATE_DIM {
        for j in 0..STATE_DIM {
            let mut value = 0.0_f64;
            for row in 0..RESIDUAL_DIM {
                for col in 0..RESIDUAL_DIM {
                    value += jacobian[row][i] * information[row][col] * jacobian[col][j];
                }
            }
            h[(base + i) * dim + (base + j)] += value;
        }
        let mut rhs = 0.0_f64;
        for row in 0..RESIDUAL_DIM {
            for (col, residual_value) in residual.iter().enumerate().take(RESIDUAL_DIM) {
                rhs += jacobian[row][i] * information[row][col] * *residual_value;
            }
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

#[allow(clippy::too_many_arguments)]
fn boundary_normal_equations(
    oldest: &VioFrame,
    next: &VioFrame,
    preintegrated: &PreintegratedImu,
    prior_on_oldest: Option<&MarginalPrior>,
    gravity: Gravity,
    camera_from_body: Pose64,
    intrinsics: PinholeIntrinsics,
    map_from_odom: &MapFromOdom,
) -> [[f64; 30]; 30] {
    let frames = [oldest.clone(), next.clone()];
    let dim = 2 * STATE_DIM;
    let mut h = vec![0.0_f64; dim * dim];
    let mut b = vec![0.0_f64; dim];

    if let Some(prior) = prior_on_oldest {
        let residual = prior.reference.local_coordinates(oldest.state());
        let jac = numerical_state_prior_jacobian(&frames, 0, &prior.reference);
        accumulate_self_information(
            &mut h,
            &mut b,
            dim,
            frame_base(0),
            &jac,
            &residual,
            &prior.information,
        );
    }

    let pose_prior_weights = [100.0; POSE_PRIOR_RESIDUAL_DIM];
    let oldest_pose_residual = pose_prior_residual(&frames[0].state, oldest.pose_measurement_odom);
    let oldest_pose_jac = numerical_pose_prior_jacobian(&frames, 0, oldest.pose_measurement_odom);
    accumulate_self(
        &mut h,
        &mut b,
        dim,
        frame_base(0),
        &oldest_pose_jac,
        &oldest_pose_residual,
        &pose_prior_weights,
    );

    for observation in oldest.visual_observations.iter().copied() {
        let residual = match reprojection_residual(
            &frames[0].state,
            camera_from_body,
            map_from_odom,
            observation,
            intrinsics,
        ) {
            Ok(residual) => residual,
            Err(_) => continue,
        };
        let jac = numerical_reprojection_jacobian(
            &frames,
            0,
            camera_from_body,
            map_from_odom,
            observation,
            intrinsics,
        );
        accumulate_self(
            &mut h,
            &mut b,
            dim,
            frame_base(0),
            &jac,
            &residual,
            &[1.0; REPROJECTION_RESIDUAL_DIM],
        );
    }

    let imu_residual =
        ImuFactor::residual(&frames[0].state, &frames[1].state, preintegrated, &gravity);
    let (imu_jac_prev, imu_jac_curr) =
        numerical_imu_jacobians(&frames, 0, 1, preintegrated, gravity);
    let imu_weights = preintegrated.residual_information_diag();
    accumulate_self(
        &mut h,
        &mut b,
        dim,
        frame_base(0),
        &imu_jac_prev,
        &imu_residual,
        &imu_weights,
    );
    accumulate_self(
        &mut h,
        &mut b,
        dim,
        frame_base(1),
        &imu_jac_curr,
        &imu_residual,
        &imu_weights,
    );
    accumulate_cross(
        &mut h,
        dim,
        frame_base(0),
        frame_base(1),
        &imu_jac_prev,
        &imu_jac_curr,
        &imu_weights,
    );

    let bias_residual = bias_random_walk_residual(&frames[0].state, &frames[1].state);
    let (bias_jac_prev, bias_jac_curr) = numerical_bias_random_walk_jacobians(&frames, 0, 1);
    let bias_weights = preintegrated.bias_random_walk_information_diag();
    accumulate_self(
        &mut h,
        &mut b,
        dim,
        frame_base(0),
        &bias_jac_prev,
        &bias_residual,
        &bias_weights,
    );
    accumulate_self(
        &mut h,
        &mut b,
        dim,
        frame_base(1),
        &bias_jac_curr,
        &bias_residual,
        &bias_weights,
    );
    accumulate_cross(
        &mut h,
        dim,
        frame_base(0),
        frame_base(1),
        &bias_jac_prev,
        &bias_jac_curr,
        &bias_weights,
    );

    let mut out = [[0.0_f64; 30]; 30];
    for row in 0..30 {
        for col in 0..30 {
            out[row][col] = h[row * 30 + col];
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn marginalize_oldest_to_next(
    oldest: &VioFrame,
    next: &VioFrame,
    preintegrated: &PreintegratedImu,
    prior_on_oldest: Option<&MarginalPrior>,
    gravity: Gravity,
    camera_from_body: Pose64,
    intrinsics: PinholeIntrinsics,
    map_from_odom: &MapFromOdom,
) -> Option<MarginalPrior> {
    let h = boundary_normal_equations(
        oldest,
        next,
        preintegrated,
        prior_on_oldest,
        gravity,
        camera_from_body,
        intrinsics,
        map_from_odom,
    );
    let (h00, h01, h10, h11) = split_boundary_hessian(h);
    let h00_inv = invert_dense_15x15(flatten_square::<STATE_DIM>(h00))?;
    let prior_information = subtract_15x15(h11, matmul_15x15(h10, matmul_15x15(h00_inv, h01)));
    Some(MarginalPrior {
        applies_to: next.keyframe_id,
        reference: next.state.clone(),
        information: prior_information,
    })
}

#[allow(clippy::too_many_arguments)]
fn exported_odometry_constraint(
    oldest: &VioFrame,
    next: &VioFrame,
    preintegrated: &PreintegratedImu,
    prior_on_oldest: Option<&MarginalPrior>,
    gravity: Gravity,
    camera_from_body: Pose64,
    intrinsics: PinholeIntrinsics,
    map_from_odom: &MapFromOdom,
) -> Option<VioOdometryConstraint> {
    let h = boundary_normal_equations(
        oldest,
        next,
        preintegrated,
        prior_on_oldest,
        gravity,
        camera_from_body,
        intrinsics,
        map_from_odom,
    );
    let covariance = invert_dense_30x30(flatten_square::<30>(h))?;
    let jacobian = relative_pose_jacobian_pair(oldest.state(), next.state());
    let covariance_pose = matmul_rect(&jacobian, &covariance, &transpose_rect::<6, 30>(&jacobian));
    Some(VioOdometryConstraint {
        from: oldest.keyframe_id,
        to: next.keyframe_id,
        relative_pose: oldest
            .state()
            .pose_odom_from_body()
            .inverse()
            .compose(next.state().pose_odom_from_body()),
        information: invert_dense_6x6(covariance_pose)
            .unwrap_or_else(|| pose_information_from_preintegration(preintegrated)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::SlamMap;
    use crate::{
        Descriptor, Detections, FrameId, ImuBatch, ImuBias, ImuNoiseModel, ImuSample, Keypoint,
        PinholeIntrinsics, Pose, SensorId, Timestamp,
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
            &batch(&[(0, [0.0; 3], [0.0; 3]), (10_000_000, [0.0; 3], [0.0; 3])]),
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
    fn correct_prediction_updates_pose_without_extending_window() {
        let gravity = Gravity::try_new([0.0, 0.0, -9.81]).expect("gravity");
        let config = VioConfig::new(4)
            .expect("config")
            .with_pose_prior_weight(1.0e12)
            .expect("pose prior weight");
        let mut vio = LocalVio::new(config, gravity, Pose64::identity(), intrinsics());
        let ids = keyframe_ids(1);
        vio.initialize(
            ids[0],
            NavState::try_new(Pose64::identity(), [0.0; 3], ImuBias::default()).expect("state"),
            Pose64::identity(),
            Vec::new(),
        )
        .expect("init");
        let preintegrated = PreintegratedImu::integrate(
            &batch(&[(0, [0.0; 3], [0.0; 3]), (10_000_000, [0.0; 3], [0.0; 3])]),
            &ImuBias::default(),
            &noise(),
        )
        .expect("preintegrated");
        let estimate = vio
            .correct_prediction(
                &preintegrated,
                Pose64::from_rt(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [1.0, 0.0, 0.0],
                ),
                Vec::new(),
            )
            .expect("correct prediction");
        assert_eq!(vio.len(), 1);
        assert!(
            estimate.state().pose_odom_from_body().translation()[0] > 0.95,
            "predicted state should follow strong pose prior"
        );
    }

    #[test]
    fn local_vio_propagates_free_fall_state() {
        let gravity = Gravity::try_new([0.0, 0.0, -9.81]).expect("gravity");
        let mut vio = LocalVio::new(
            VioConfig::new(4).expect("config"),
            gravity,
            Pose64::identity(),
            intrinsics(),
        );
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
        let mut vio = LocalVio::new(
            VioConfig::new(2).expect("config"),
            gravity,
            Pose64::identity(),
            intrinsics(),
        );
        let ids = keyframe_ids(4);
        vio.initialize(
            ids[0],
            NavState::try_new(Pose64::identity(), [0.0; 3], ImuBias::default()).expect("state"),
            Pose64::identity(),
            Vec::new(),
        )
        .expect("init");
        for keyframe_id in ids.iter().skip(1) {
            let preintegrated = PreintegratedImu::integrate(
                &batch(&[(0, [0.0; 3], [0.0; 3]), (10_000_000, [0.0; 3], [0.0; 3])]),
                &ImuBias::default(),
                &noise(),
            )
            .expect("preintegrated");
            vio.push_preintegrated(*keyframe_id, preintegrated, Pose64::identity(), Vec::new())
                .expect("push");
        }
        assert_eq!(vio.len(), 2);
        assert_eq!(vio.latest_estimate().expect("latest").keyframe_id(), ids[3]);
    }

    #[test]
    fn odometry_constraint_is_emitted_when_window_rolls() {
        let gravity = Gravity::try_new([0.0, 0.0, -9.81]).expect("gravity");
        let mut vio = LocalVio::new(
            VioConfig::new(2).expect("config"),
            gravity,
            Pose64::identity(),
            intrinsics(),
        );
        let ids = keyframe_ids(3);
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
        assert!(vio.drain_exported_odometry().is_empty());
        let preintegrated = PreintegratedImu::integrate(
            &batch(&[
                (20_000_000, [0.0; 3], [0.0; 3]),
                (30_000_000, [0.0; 3], [0.0; 3]),
                (40_000_000, [0.0; 3], [0.0; 3]),
            ]),
            &ImuBias::default(),
            &noise(),
        )
        .expect("preintegrated");
        vio.push_preintegrated(ids[2], preintegrated, Pose64::identity(), Vec::new())
            .expect("push");
        let exported = vio.drain_exported_odometry();
        assert_eq!(exported.len(), 1);
        let constraint = &exported[0];
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
            .with_pose_prior_weight(1.0e12)
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
            &batch(&[(0, [0.0; 3], [0.0; 3]), (10_000_000, [0.0; 3], [0.0; 3])]),
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
        assert!(
            x > 0.95,
            "optimized x should follow strong pose prior, got {x}"
        );
    }

    #[test]
    fn local_vio_solver_uses_visual_observations() {
        let gravity = Gravity::try_new([0.0, 0.0, -9.81]).expect("gravity");
        let intrinsics = intrinsics();
        let config = VioConfig::new(4)
            .expect("config")
            .with_pose_prior_weight(1.0e9)
            .expect("pose prior weight");
        let mut with_visual = LocalVio::new(config, gravity, Pose64::identity(), intrinsics);
        let mut without_visual = LocalVio::new(config, gravity, Pose64::identity(), intrinsics);
        let ids = keyframe_ids(2);
        with_visual
            .initialize(
                ids[0],
                NavState::try_new(Pose64::identity(), [0.0; 3], ImuBias::default()).expect("state"),
                Pose64::identity(),
                Vec::new(),
            )
            .expect("init");
        without_visual
            .initialize(
                ids[0],
                NavState::try_new(Pose64::identity(), [0.0; 3], ImuBias::default()).expect("state"),
                Pose64::identity(),
                Vec::new(),
            )
            .expect("init");
        let preintegrated = PreintegratedImu::integrate(
            &batch(&[(0, [0.0; 3], [0.0; 3]), (10_000_000, [0.0; 3], [0.0; 3])]),
            &ImuBias::default(),
            &noise(),
        )
        .expect("preintegrated");
        let measurement = Pose64::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0, 0.0, -0.25],
        );
        let observations = vec![
            VioObservation::new(
                crate::Point3 {
                    x: 0.2,
                    y: 0.0,
                    z: 1.0,
                },
                Keypoint { x: 70.0, y: 40.0 },
            )
            .expect("observation 0"),
            VioObservation::new(
                crate::Point3 {
                    x: -0.2,
                    y: 0.0,
                    z: 1.0,
                },
                Keypoint { x: 30.0, y: 40.0 },
            )
            .expect("observation 1"),
            VioObservation::new(
                crate::Point3 {
                    x: 0.0,
                    y: 0.2,
                    z: 1.0,
                },
                Keypoint { x: 50.0, y: 60.0 },
            )
            .expect("observation 2"),
        ];
        let with_visual_estimate = with_visual
            .push_preintegrated(ids[1], preintegrated.clone(), measurement, observations)
            .expect("push with visual");
        let without_visual_estimate = without_visual
            .push_preintegrated(ids[1], preintegrated, measurement, Vec::new())
            .expect("push without visual");
        let z_with_visual = with_visual_estimate
            .state()
            .pose_odom_from_body()
            .translation()[2];
        let z_without_visual = without_visual_estimate
            .state()
            .pose_odom_from_body()
            .translation()[2];
        assert!(
            z_with_visual.abs() < z_without_visual.abs(),
            "visual observations should pull depth toward reprojection-consistent geometry: with={z_with_visual} without={z_without_visual}"
        );
    }

    #[test]
    fn local_vio_solver_preserves_bias_state_through_window_updates() {
        let gravity = Gravity::try_new([0.0, 0.0, -9.81]).expect("gravity");
        let mut vio = LocalVio::new(
            VioConfig::new(4).expect("config"),
            gravity,
            Pose64::identity(),
            intrinsics(),
        );
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
