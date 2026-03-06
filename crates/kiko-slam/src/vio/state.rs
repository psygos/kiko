use crate::{ImuBias, Pose64};

#[derive(Clone, Debug)]
pub struct NavState {
    pose_odom_from_body: Pose64,
    velocity_odom_mps: [f64; 3],
    bias: ImuBias,
}

impl NavState {
    pub fn new(pose_odom_from_body: Pose64, velocity_odom_mps: [f64; 3], bias: ImuBias) -> Self {
        Self {
            pose_odom_from_body,
            velocity_odom_mps,
            bias,
        }
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
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gravity {
    vector_odom_mps2: [f64; 3],
}

impl Gravity {
    pub fn new(vector_odom_mps2: [f64; 3]) -> Self {
        Self { vector_odom_mps2 }
    }

    pub fn vector_odom_mps2(&self) -> [f64; 3] {
        self.vector_odom_mps2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nav_state_preserves_constructor_values() {
        let bias = ImuBias {
            accel: [0.1, 0.2, 0.3],
            gyro: [0.4, 0.5, 0.6],
        };
        let state = NavState::new(Pose64::identity(), [1.0, 2.0, 3.0], bias.clone());
        assert_eq!(state.pose_odom_from_body().translation(), [0.0, 0.0, 0.0]);
        assert_eq!(state.velocity_odom_mps(), [1.0, 2.0, 3.0]);
        assert_eq!(state.bias(), &bias);
    }

    #[test]
    fn gravity_preserves_vector() {
        let gravity = Gravity::new([0.0, 0.0, -9.81]);
        assert_eq!(gravity.vector_odom_mps2(), [0.0, 0.0, -9.81]);
    }
}
