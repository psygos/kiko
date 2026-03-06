use std::collections::VecDeque;
use std::num::NonZeroUsize;

use crate::math::{mat_mul_vec_f64, mat_mul_f64};
use crate::map::KeyframeId;
use crate::{Gravity, NavState, Pose64, PreintegratedImu};

#[derive(Clone, Copy, Debug)]
pub struct VioConfig {
    window_size: NonZeroUsize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VioConfigError {
    ZeroWindowSize,
}

impl std::fmt::Display for VioConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VioConfigError::ZeroWindowSize => write!(f, "vio window size must be > 0"),
        }
    }
}

impl std::error::Error for VioConfigError {}

impl VioConfig {
    pub fn new(window_size: usize) -> Result<Self, VioConfigError> {
        let window_size = NonZeroUsize::new(window_size).ok_or(VioConfigError::ZeroWindowSize)?;
        Ok(Self { window_size })
    }

    pub fn window_size(self) -> usize {
        self.window_size.get()
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
struct VioFrame {
    keyframe_id: KeyframeId,
    state: NavState,
    #[allow(dead_code)]
    preintegrated_from_prev: Option<PreintegratedImu>,
}

pub struct LocalVio {
    config: VioConfig,
    gravity: Gravity,
    frames: VecDeque<VioFrame>,
}

impl LocalVio {
    pub fn new(config: VioConfig, gravity: Gravity) -> Self {
        Self {
            config,
            gravity,
            frames: VecDeque::new(),
        }
    }

    pub fn initialize(
        &mut self,
        keyframe_id: KeyframeId,
        state: NavState,
    ) -> Result<(), LocalVioError> {
        if let Some(existing) = self.frames.front() {
            return Err(LocalVioError::AlreadyInitialized {
                keyframe_id: existing.keyframe_id,
            });
        }
        self.frames.push_back(VioFrame {
            keyframe_id,
            state,
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
    ) -> Result<VioEstimate, LocalVioError> {
        let previous = self.frames.back().ok_or(LocalVioError::NotInitialized)?;
        if self.frames.iter().any(|frame| frame.keyframe_id == keyframe_id) {
            return Err(LocalVioError::DuplicateKeyframe { keyframe_id });
        }
        let propagated = propagate_state(previous.state(), &preintegrated, self.gravity);
        self.frames.push_back(VioFrame {
            keyframe_id,
            state: propagated.clone(),
            preintegrated_from_prev: Some(preintegrated),
        });
        while self.frames.len() > self.config.window_size() {
            self.frames.pop_front();
        }
        Ok(VioEstimate {
            keyframe_id,
            state: propagated,
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::SlamMap;
    use crate::{Detections, Descriptor, FrameId, ImuBatch, ImuBias, ImuNoiseModel, ImuSample, Keypoint, Pose, SensorId, Timestamp};

    fn noise() -> ImuNoiseModel {
        ImuNoiseModel {
            accel_noise_density: 0.1,
            gyro_noise_density: 0.01,
            accel_random_walk: 0.001,
            gyro_random_walk: 0.0001,
        }
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
            .push_preintegrated(ids[1], preintegrated)
            .expect_err("push without init should fail");
        assert_eq!(err, LocalVioError::NotInitialized);
    }

    #[test]
    fn local_vio_propagates_free_fall_state() {
        let gravity = Gravity::try_new([0.0, 0.0, -9.81]).expect("gravity");
        let mut vio = LocalVio::new(VioConfig::new(4).expect("config"), gravity);
        let ids = keyframe_ids(2);
        vio.initialize(
            ids[0],
            NavState::try_new(Pose64::identity(), [0.0; 3], ImuBias::default()).expect("state"),
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
        let estimate = vio
            .push_preintegrated(ids[1], preintegrated)
            .expect("propagate");
        let position = estimate.state().pose_odom_from_body().translation();
        let velocity = estimate.state().velocity_odom_mps();
        assert!((position[2] + 0.5 * 9.81 * 0.02 * 0.02).abs() < 1e-9);
        assert!((velocity[2] + 9.81 * 0.02).abs() < 1e-9);
    }

    #[test]
    fn local_vio_caps_window_size() {
        let gravity = Gravity::try_new([0.0, 0.0, -9.81]).expect("gravity");
        let mut vio = LocalVio::new(VioConfig::new(2).expect("config"), gravity);
        let ids = keyframe_ids(4);
        vio.initialize(
            ids[0],
            NavState::try_new(Pose64::identity(), [0.0; 3], ImuBias::default()).expect("state"),
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
            vio.push_preintegrated(ids[idx], preintegrated)
                .expect("push");
        }
        assert_eq!(vio.len(), 2);
        assert_eq!(vio.latest_estimate().expect("latest").keyframe_id(), ids[3]);
    }
}
