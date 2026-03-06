use crate::Pose64;

#[derive(Clone, Debug)]
pub struct MapFromOdom {
    correction: Pose64,
}

impl MapFromOdom {
    pub fn identity() -> Self {
        Self {
            correction: Pose64::identity(),
        }
    }

    pub fn correction(&self) -> Pose64 {
        self.correction
    }

    pub fn set_correction(&mut self, correction: Pose64) {
        self.correction = correction;
    }

    pub fn apply_correction(&mut self, correction: Pose64) {
        self.correction = correction.compose(self.correction);
    }

    pub fn odom_to_map(&self, pose_cam_from_odom: Pose64) -> Pose64 {
        self.correction.compose(pose_cam_from_odom)
    }

    pub fn map_to_odom(&self, pose_cam_from_map: Pose64) -> Pose64 {
        self.correction.inverse().compose(pose_cam_from_map)
    }
}

impl Default for MapFromOdom {
    fn default() -> Self {
        Self::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pose;

    #[test]
    fn identity_correction_is_passthrough() {
        let bridge = MapFromOdom::identity();
        let pose = Pose64::from_pose32(Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            [1.0, 2.0, 3.0],
        ));
        assert_eq!(
            bridge.odom_to_map(pose).to_pose32().translation(),
            pose.to_pose32().translation()
        );
    }

    #[test]
    fn correction_updates_map_pose_without_losing_inverse() {
        let mut bridge = MapFromOdom::identity();
        let correction = Pose64::from_pose32(Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            [0.5, -1.0, 2.0],
        ));
        bridge.set_correction(correction);
        let odom_pose = Pose64::from_pose32(Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [4.0, 5.0, 6.0],
        ));
        let map_pose = bridge.odom_to_map(odom_pose);
        let recovered = bridge.map_to_odom(map_pose);
        let recovered_pose = recovered.to_pose32();
        let odom_pose32 = odom_pose.to_pose32();
        assert_eq!(recovered_pose.translation(), odom_pose32.translation());
        assert_eq!(recovered_pose.rotation(), odom_pose32.rotation());
    }
}
