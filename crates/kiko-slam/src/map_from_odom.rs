use crate::{Point3, Pose64};

#[derive(Clone, Debug)]
pub struct MapFromOdom {
    pose_map_from_odom: Pose64,
}

impl MapFromOdom {
    pub fn identity() -> Self {
        Self {
            pose_map_from_odom: Pose64::identity(),
        }
    }

    pub fn pose_map_from_odom(&self) -> Pose64 {
        self.pose_map_from_odom
    }

    pub fn set_pose_map_from_odom(&mut self, pose_map_from_odom: Pose64) {
        self.pose_map_from_odom = pose_map_from_odom;
    }

    pub fn apply_pose_map_correction(&mut self, pose_map_from_map: Pose64) {
        self.pose_map_from_odom = pose_map_from_map.compose(self.pose_map_from_odom);
    }

    pub fn odom_to_map(&self, pose_cam_from_odom: Pose64) -> Pose64 {
        pose_cam_from_odom.compose(self.pose_map_from_odom.inverse())
    }

    pub fn map_to_odom(&self, pose_cam_from_map: Pose64) -> Pose64 {
        pose_cam_from_map.compose(self.pose_map_from_odom)
    }

    pub fn align_to_pose(&mut self, pose_cam_from_map: Pose64, pose_cam_from_odom: Pose64) {
        self.pose_map_from_odom =
            Self::pose_map_from_odom_for(pose_cam_from_map, pose_cam_from_odom);
    }

    pub fn pose_map_from_odom_for(pose_cam_from_map: Pose64, pose_cam_from_odom: Pose64) -> Pose64 {
        pose_cam_from_map.inverse().compose(pose_cam_from_odom)
    }

    pub fn point_odom_to_map(&self, point_odom: Point3) -> Point3 {
        transform_point(self.pose_map_from_odom, point_odom)
    }

    pub fn point_map_to_odom(&self, point_map: Point3) -> Point3 {
        transform_point(self.pose_map_from_odom.inverse(), point_map)
    }
}

impl Default for MapFromOdom {
    fn default() -> Self {
        Self::identity()
    }
}

fn transform_point(transform: Pose64, point: Point3) -> Point3 {
    let rotated = crate::math::mat_mul_vec_f64(
        transform.rotation(),
        [f64::from(point.x), f64::from(point.y), f64::from(point.z)],
    );
    let translation = transform.translation();
    Point3 {
        x: (rotated[0] + translation[0]) as f32,
        y: (rotated[1] + translation[1]) as f32,
        z: (rotated[2] + translation[2]) as f32,
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
        let pose_map_from_odom = Pose64::from_pose32(Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            [0.5, -1.0, 2.0],
        ));
        bridge.set_pose_map_from_odom(pose_map_from_odom);
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

    #[test]
    fn alignment_from_pose_pair_round_trips_pose_and_points() {
        let pose_cam_from_odom = Pose64::from_pose32(Pose::from_rt(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [1.0, -0.5, 0.25],
        ));
        let pose_map_from_odom = Pose64::from_pose32(Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            [0.5, 1.0, -2.0],
        ));
        let pose_cam_from_map = pose_cam_from_odom.compose(pose_map_from_odom.inverse());
        let mut bridge = MapFromOdom::identity();
        bridge.align_to_pose(pose_cam_from_map, pose_cam_from_odom);

        let recovered_pose = bridge.odom_to_map(pose_cam_from_odom);
        assert_eq!(
            recovered_pose.to_pose32().translation(),
            pose_cam_from_map.to_pose32().translation()
        );
        assert_eq!(
            recovered_pose.to_pose32().rotation(),
            pose_cam_from_map.to_pose32().rotation()
        );

        let point_odom = Point3 {
            x: 0.2,
            y: -0.4,
            z: 1.5,
        };
        let point_map = bridge.point_odom_to_map(point_odom);
        let recovered_point = bridge.point_map_to_odom(point_map);
        assert!((recovered_point.x - point_odom.x).abs() < 1e-6);
        assert!((recovered_point.y - point_odom.y).abs() < 1e-6);
        assert!((recovered_point.z - point_odom.z).abs() < 1e-6);
    }

    #[test]
    fn alignment_from_consistent_pose_pairs_is_pose_invariant() {
        let pose_map_from_odom = Pose64::from_pose32(Pose::from_rt(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [0.25, -0.75, 1.5],
        ));
        let cam0_from_odom = Pose64::from_pose32(Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            [1.0, 2.0, 3.0],
        ));
        let cam1_from_odom = Pose64::from_pose32(Pose::from_rt(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [-2.0, 0.5, 4.0],
        ));
        let cam0_from_map = cam0_from_odom.compose(pose_map_from_odom.inverse());
        let cam1_from_map = cam1_from_odom.compose(pose_map_from_odom.inverse());

        let derived0 = MapFromOdom::pose_map_from_odom_for(cam0_from_map, cam0_from_odom);
        let derived1 = MapFromOdom::pose_map_from_odom_for(cam1_from_map, cam1_from_odom);

        let expected = pose_map_from_odom.to_pose32();
        assert_eq!(derived0.to_pose32().translation(), expected.translation());
        assert_eq!(derived0.to_pose32().rotation(), expected.rotation());
        assert_eq!(derived1.to_pose32().translation(), expected.translation());
        assert_eq!(derived1.to_pose32().rotation(), expected.rotation());
    }
}
