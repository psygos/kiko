use crate::{CamLFrame, GeometryError, MapFrame, OdomFrame, Pose64, Transform3d};

#[derive(Clone, Debug)]
pub struct MapFromOdom {
    odom_from_map: Transform3d<OdomFrame, MapFrame>,
}

impl MapFromOdom {
    pub fn identity() -> Self {
        Self {
            odom_from_map: Transform3d::identity_between_frames(),
        }
    }

    pub fn try_odom_to_map(&self, cam_from_odom: Pose64) -> Result<Pose64, GeometryError> {
        let cam_from_odom = Transform3d::<CamLFrame, OdomFrame>::from_pose64(cam_from_odom)?;
        Ok(cam_from_odom.try_compose(self.odom_from_map)?.into_pose64())
    }

    pub fn try_map_to_odom(&self, cam_from_map: Pose64) -> Result<Pose64, GeometryError> {
        let cam_from_map = Transform3d::<CamLFrame, MapFrame>::from_pose64(cam_from_map)?;
        let map_from_odom = self.odom_from_map.try_inverse()?;
        Ok(cam_from_map.try_compose(map_from_odom)?.into_pose64())
    }

    pub fn try_align_to_pose(
        &mut self,
        cam_from_map: Pose64,
        cam_from_odom: Pose64,
    ) -> Result<(), GeometryError> {
        self.odom_from_map = Self::try_odom_from_map_for(cam_from_map, cam_from_odom)?;
        Ok(())
    }

    fn try_odom_from_map_for(
        cam_from_map: Pose64,
        cam_from_odom: Pose64,
    ) -> Result<Transform3d<OdomFrame, MapFrame>, GeometryError> {
        let cam_from_map = Transform3d::<CamLFrame, MapFrame>::from_pose64(cam_from_map)?;
        let cam_from_odom = Transform3d::<CamLFrame, OdomFrame>::from_pose64(cam_from_odom)?;
        cam_from_odom.try_inverse()?.try_compose(cam_from_map)
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
            bridge
                .try_odom_to_map(pose)
                .expect("identity bridge")
                .to_pose32()
                .translation(),
            pose.to_pose32().translation()
        );
    }

    #[test]
    fn correction_updates_map_pose_without_losing_inverse() {
        let pose_map_from_odom = Pose64::from_pose32(Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            [0.5, -1.0, 2.0],
        ));
        let mut bridge = MapFromOdom::identity();
        bridge
            .try_align_to_pose(Pose64::identity(), pose_map_from_odom)
            .expect("map-from-odom bridge");
        let odom_pose = Pose64::from_pose32(Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [4.0, 5.0, 6.0],
        ));
        let map_pose = bridge.try_odom_to_map(odom_pose).expect("map pose");
        let recovered = bridge.try_map_to_odom(map_pose).expect("odom pose");
        let recovered_pose = recovered.to_pose32();
        let odom_pose32 = odom_pose.to_pose32();
        assert_eq!(recovered_pose.translation(), odom_pose32.translation());
        assert_eq!(recovered_pose.rotation(), odom_pose32.rotation());
    }

    #[test]
    fn alignment_from_pose_pair_round_trips_pose() {
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
        bridge
            .try_align_to_pose(pose_cam_from_map, pose_cam_from_odom)
            .expect("alignment");

        let recovered_pose = bridge
            .try_odom_to_map(pose_cam_from_odom)
            .expect("recovered pose");
        assert_eq!(
            recovered_pose.to_pose32().translation(),
            pose_cam_from_map.to_pose32().translation()
        );
        assert_eq!(
            recovered_pose.to_pose32().rotation(),
            pose_cam_from_map.to_pose32().rotation()
        );
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

        let derived0 = MapFromOdom::try_odom_from_map_for(cam0_from_map, cam0_from_odom)
            .expect("first bridge")
            .into_pose64();
        let derived1 = MapFromOdom::try_odom_from_map_for(cam1_from_map, cam1_from_odom)
            .expect("second bridge")
            .into_pose64();

        let expected = pose_map_from_odom.inverse().to_pose32();
        assert_eq!(derived0.to_pose32().translation(), expected.translation());
        assert_eq!(derived0.to_pose32().rotation(), expected.rotation());
        assert_eq!(derived1.to_pose32().translation(), expected.translation());
        assert_eq!(derived1.to_pose32().rotation(), expected.rotation());
    }

    #[test]
    fn failed_alignment_does_not_mutate_existing_bridge() {
        let half_sqrt_two = std::f64::consts::FRAC_1_SQRT_2;
        let cam_from_odom = Pose64::try_from_rt(
            [
                [half_sqrt_two, -half_sqrt_two, 0.0],
                [half_sqrt_two, half_sqrt_two, 0.0],
                [0.0, 0.0, 1.0],
            ],
            [f64::MAX, f64::MAX, 0.0],
        )
        .expect("finite odometry pose");
        let cam_from_map = Pose64::identity();
        let mut bridge = MapFromOdom::identity();

        assert!(
            bridge
                .try_align_to_pose(cam_from_map, cam_from_odom)
                .is_err()
        );
        assert_eq!(
            bridge
                .try_odom_to_map(Pose64::identity())
                .expect("unchanged bridge"),
            Pose64::identity()
        );
    }
}
