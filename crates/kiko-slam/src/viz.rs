use std::num::NonZeroUsize;

use crate::{
    CovisibilitySnapshot, DepthImage, Detections, Frame, FrameDiagnostics, ImuBatch, Keypoint,
    Point3, Pose, Raw, Timestamp, TrackingPose, VioTelemetry, VizPacket, env::env_f32,
    env::env_usize,
};

use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VizDecimation(NonZeroUsize);

impl VizDecimation {
    pub fn every_n(n: NonZeroUsize) -> Self {
        Self(n)
    }

    pub fn get(self) -> usize {
        self.0.get()
    }

    fn should_log(self, index: u64) -> bool {
        let n = self.0.get() as u64;
        index % n == 0
    }
}

impl Default for VizDecimation {
    fn default() -> Self {
        Self(NonZeroUsize::MIN)
    }
}

#[derive(Debug)]
pub enum VizDecimationError {
    Zero,
}

impl std::fmt::Display for VizDecimationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VizDecimationError::Zero => write!(f, "decimation must be > 0"),
        }
    }
}

impl std::error::Error for VizDecimationError {}

impl TryFrom<usize> for VizDecimation {
    type Error = VizDecimationError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        NonZeroUsize::new(value)
            .map(VizDecimation)
            .ok_or(VizDecimationError::Zero)
    }
}

impl std::fmt::Display for VizDecimation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.get())
    }
}

impl std::str::FromStr for VizDecimation {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let value: usize = s
            .trim()
            .parse()
            .map_err(|_| format!("invalid decimation: {s}"))?;
        Self::try_from(value).map_err(|e| e.to_string())
    }
}

#[derive(Debug)]
pub enum VizLogError {
    Rerun(rerun::RecordingStreamError),
}

impl std::fmt::Display for VizLogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VizLogError::Rerun(err) => write!(f, "rerun logging error: {err}"),
        }
    }
}

impl std::error::Error for VizLogError {}

impl From<rerun::RecordingStreamError> for VizLogError {
    fn from(err: rerun::RecordingStreamError) -> Self {
        VizLogError::Rerun(err)
    }
}

#[derive(Debug)]
pub struct RerunSink {
    rec: rerun::RecordingStream,
    decimation: VizDecimation,
    frame_index: u64,
    depth_index: u64,
    tracks: TrackState,
    odom_trajectory: TrajectoryLog,
    corrected_trajectory: TrajectoryLog,
    visual_measurement_trajectory: TrajectoryLog,
    logged_world: bool,
    surface_pose_quality_gate: SurfacePoseQualityGate,
    surface_map: crate::SurfaceBeliefMap,
}

impl RerunSink {
    pub fn new(rec: rerun::RecordingStream, decimation: VizDecimation) -> Self {
        Self {
            rec,
            decimation,
            frame_index: 0,
            depth_index: 0,
            tracks: TrackState::new(),
            odom_trajectory: TrajectoryLog::default(),
            corrected_trajectory: TrajectoryLog::default(),
            visual_measurement_trajectory: TrajectoryLog::default(),
            logged_world: false,
            surface_pose_quality_gate: SurfacePoseQualityGate::load(),
            surface_map: crate::SurfaceBeliefMap::new(crate::SurfaceMapConfig::from_env()),
        }
    }

    pub fn log(&mut self, packet: &VizPacket<Raw>) -> Result<(), VizLogError> {
        self.log_with_points(packet, None)
    }

    pub fn log_frames(&mut self, left: &Frame, right: &Frame) -> Result<(), VizLogError> {
        let index = self.frame_index;
        self.frame_index = self.frame_index.saturating_add(1);
        if !self.decimation.should_log(index) {
            return Ok(());
        }

        self.set_time(left.timestamp());

        let left_image = rerun::Image::from_color_model_and_bytes(
            left.data().to_vec(),
            [left.width(), left.height()],
            rerun::ColorModel::L,
            rerun::ChannelDatatype::U8,
        )
        .with_draw_order(0.0);
        self.rec.log("view/left", &left_image)?;

        let right_image = rerun::Image::from_color_model_and_bytes(
            right.data().to_vec(),
            [right.width(), right.height()],
            rerun::ColorModel::L,
            rerun::ChannelDatatype::U8,
        )
        .with_draw_order(0.0);
        self.rec.log("view/right", &right_image)?;

        Ok(())
    }

    pub fn log_depth(&mut self, depth: &DepthImage) -> Result<(), VizLogError> {
        let index = self.depth_index;
        self.depth_index = self.depth_index.saturating_add(1);
        if !self.decimation.should_log(index) {
            return Ok(());
        }

        self.set_time(depth.timestamp());
        // Normalize depth to 0-255 u8 for reliable rendering.
        let max_depth: f32 = 10.0;
        let mut pixels = Vec::with_capacity(depth.depth_m().len());
        for &sample in depth.depth_m() {
            let clamped = if sample.is_finite() && sample > 0.0 {
                ((sample / max_depth).min(1.0) * 255.0) as u8
            } else {
                0
            };
            pixels.push(clamped);
        }
        let depth_image = rerun::Image::from_color_model_and_bytes(
            pixels,
            [depth.width(), depth.height()],
            rerun::ColorModel::L,
            rerun::ChannelDatatype::U8,
        )
        .with_draw_order(0.0);
        self.rec.log("view/depth", &depth_image)?;
        Ok(())
    }

    pub fn log_with_points(
        &mut self,
        packet: &VizPacket<Raw>,
        points: Option<&[Point3]>,
    ) -> Result<(), VizLogError> {
        let index = self.frame_index;
        self.frame_index = self.frame_index.saturating_add(1);
        if !self.decimation.should_log(index) {
            return Ok(());
        }

        let left = packet.left();
        let right = packet.right();
        let track_ids = self.tracks.assign_tracks(packet.matches().source_a());

        self.set_time(left.timestamp());

        let left_image = rerun::Image::from_color_model_and_bytes(
            left.data().to_vec(),
            [left.width(), left.height()],
            rerun::ColorModel::L,
            rerun::ChannelDatatype::U8,
        )
        .with_draw_order(0.0);
        self.rec.log("view/left", &left_image)?;

        let right_image = rerun::Image::from_color_model_and_bytes(
            right.data().to_vec(),
            [right.width(), right.height()],
            rerun::ColorModel::L,
            rerun::ChannelDatatype::U8,
        )
        .with_draw_order(0.0);
        self.rec.log("view/right", &right_image)?;

        let (stitched, width, height) = stitch_luma(left, right);
        let matches_image = rerun::Image::from_color_model_and_bytes(
            stitched,
            [width, height],
            rerun::ColorModel::L,
            rerun::ChannelDatatype::U8,
        )
        .with_draw_order(0.0);
        self.rec.log("view/matches", &matches_image)?;

        log_matches(&self.rec, packet, left.width() as f32, &track_ids)?;

        if let Some(points) = points {
            if !points.is_empty() {
                let positions: Vec<[f32; 3]> = points.iter().map(|p| [p.x, p.y, p.z]).collect();
                let cloud = rerun::Points3D::new(positions);
                self.rec.log("world/points", &cloud)?;
            }
        }

        Ok(())
    }

    pub fn log_tracking_pose(
        &mut self,
        timestamp: Timestamp,
        pose: &TrackingPose,
    ) -> Result<(), VizLogError> {
        self.set_time(timestamp);
        self.ensure_world_logged()?;

        log_pose_variant(
            &self.rec,
            "odom",
            &pose.cam_from_odom_pose32(),
            rerun::Color::from_rgb(80, 180, 255),
            &mut self.odom_trajectory,
        )?;
        log_pose_variant(
            &self.rec,
            "map_corrected",
            &pose.cam_from_map_pose32(),
            rerun::Color::from_rgb(90, 220, 130),
            &mut self.corrected_trajectory,
        )?;
        match pose.cam_from_map_visual_measurement_pose32() {
            Some(measurement) => log_pose_variant(
                &self.rec,
                "visual_measurement",
                &measurement,
                rerun::Color::from_rgb(255, 170, 60),
                &mut self.visual_measurement_trajectory,
            )?,
            None => self.visual_measurement_trajectory.break_strip(),
        }

        Ok(())
    }

    /// Log stable surface observations, optionally fuse them into the surface
    /// belief map, and emit the low-resolution voxel/debug surface plus
    /// stability metrics.
    pub fn log_surface_observations(
        &mut self,
        timestamp: Timestamp,
        raw_frame_points: &[[f32; 3]],
        points: &[crate::StableSurfacePoint],
        stats: &crate::StableSurfaceStats,
        cam_from_map: Pose,
        diagnostics: &FrameDiagnostics,
        surface_integration_requested: bool,
        slam_keyframe: bool,
    ) -> Result<(), VizLogError> {
        self.set_time(timestamp);
        self.ensure_world_logged()?;
        self.rec.log(
            "diagnostics/surface/input_raw_observations",
            &rerun::Scalars::single(stats.input_samples as f64),
        )?;
        self.rec.log(
            "diagnostics/surface/accepted_raw_observations",
            &rerun::Scalars::single(stats.points_generated as f64),
        )?;
        self.rec.log(
            "diagnostics/surface/dropped_disparity",
            &rerun::Scalars::single(stats.dropped_disparity as f64),
        )?;
        self.rec.log(
            "diagnostics/surface/dropped_uncertainty",
            &rerun::Scalars::single(stats.dropped_uncertainty as f64),
        )?;
        self.rec.log(
            "diagnostics/surface/dropped_out_of_bounds",
            &rerun::Scalars::single(stats.dropped_out_of_bounds as f64),
        )?;
        self.rec.log(
            "diagnostics/surface/points_capped",
            &rerun::Scalars::single(if stats.points_capped { 1.0 } else { 0.0 }),
        )?;
        self.rec.log(
            "diagnostics/surface/frame_gate/integration_requested",
            &rerun::Scalars::single(if surface_integration_requested {
                1.0
            } else {
                0.0
            }),
        )?;
        self.rec.log(
            "diagnostics/surface/frame_gate/slam_keyframe",
            &rerun::Scalars::single(if slam_keyframe { 1.0 } else { 0.0 }),
        )?;
        if let Some(mean_sigma_m) = stats.mean_accepted_position_sigma_m {
            self.rec.log(
                "diagnostics/surface/accepted_mean_point_sigma_mm",
                &rerun::Scalars::single(mean_sigma_m * 1000.0),
            )?;
        }
        if let Some(max_sigma_m) = stats.max_accepted_position_sigma_m {
            self.rec.log(
                "diagnostics/surface/accepted_max_point_sigma_mm",
                &rerun::Scalars::single(max_sigma_m as f64 * 1000.0),
            )?;
        }
        if let Some(mean_rectified_row_mismatch_px) = stats.mean_accepted_rectified_row_mismatch_px
        {
            self.rec.log(
                "diagnostics/surface/retained_raw_observations/mean_rectified_row_mismatch_px",
                &rerun::Scalars::single(mean_rectified_row_mismatch_px.value_px() as f64),
            )?;
        }
        if let Some(max_rectified_row_mismatch_px) = stats.max_accepted_rectified_row_mismatch_px {
            self.rec.log(
                "diagnostics/surface/retained_raw_observations/max_rectified_row_mismatch_px",
                &rerun::Scalars::single(max_rectified_row_mismatch_px.value_px() as f64),
            )?;
        }

        let gate = self.surface_pose_quality_gate.decide(diagnostics);
        for (path, value) in surface_pose_quality_scalars(&gate) {
            self.rec.log(path, &rerun::Scalars::single(value))?;
        }
        let voxel_radius = (self.surface_map.config().voxel_size * 0.45).max(0.005);
        self.log_surface_raw_frame_observations(cam_from_map, raw_frame_points, voxel_radius)?;
        self.log_surface_frame_candidates(
            cam_from_map,
            points,
            gate.accepts_surface_integration(),
            voxel_radius,
        )?;

        let integration = if surface_integration_requested && gate.accepts_surface_integration() {
            self.surface_map.integrate(points, cam_from_map)
        } else {
            crate::surface_map::SurfaceBatchIntegrationSummary::default()
        };
        for (path, value) in surface_integration_scalars(&integration) {
            self.rec.log(path, &rerun::Scalars::single(value))?;
        }

        self.log_surface_map_state()
    }

    /// Log TSDF mesh as a triangle mesh with vertex colors.
    pub fn log_tsdf_mesh(&self, mesh: &crate::tsdf::MeshData) -> Result<(), VizLogError> {
        if mesh.positions.is_empty() {
            return Ok(());
        }
        let positions: Vec<rerun::Position3D> = mesh
            .positions
            .iter()
            .map(|p| rerun::Position3D::new(p[0], p[1], p[2]))
            .collect();
        let indices: Vec<rerun::TriangleIndices> = mesh
            .indices
            .iter()
            .map(|t| rerun::TriangleIndices::from([t[0], t[1], t[2]]))
            .collect();
        let colors: Vec<rerun::Color> = mesh
            .colors
            .iter()
            .map(|c| rerun::Color::from_rgb(c[0], c[1], c[2]))
            .collect();
        let mesh3d = rerun::Mesh3D::new(positions)
            .with_triangle_indices(indices)
            .with_vertex_colors(colors);
        self.rec.log_static("world/tsdf_mesh", &mesh3d)?;
        Ok(())
    }

    pub fn log_vio_telemetry(
        &self,
        timestamp: Timestamp,
        telemetry: &VioTelemetry,
    ) -> Result<(), VizLogError> {
        self.set_time(timestamp);
        let velocity = telemetry.velocity_odom_mps();
        let accel_bias = telemetry.accel_bias_mps2();
        let gyro_bias = telemetry.gyro_bias_radps();
        let speed =
            (velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2])
                .sqrt();

        self.rec
            .log("imu_state/velocity/x", &rerun::Scalars::single(velocity[0]))?;
        self.rec
            .log("imu_state/velocity/y", &rerun::Scalars::single(velocity[1]))?;
        self.rec
            .log("imu_state/velocity/z", &rerun::Scalars::single(velocity[2]))?;
        self.rec
            .log("imu_state/velocity/speed", &rerun::Scalars::single(speed))?;
        self.rec.log(
            "imu_state/bias/accel/x",
            &rerun::Scalars::single(accel_bias[0]),
        )?;
        self.rec.log(
            "imu_state/bias/accel/y",
            &rerun::Scalars::single(accel_bias[1]),
        )?;
        self.rec.log(
            "imu_state/bias/accel/z",
            &rerun::Scalars::single(accel_bias[2]),
        )?;
        self.rec.log(
            "imu_state/bias/gyro/x",
            &rerun::Scalars::single(gyro_bias[0]),
        )?;
        self.rec.log(
            "imu_state/bias/gyro/y",
            &rerun::Scalars::single(gyro_bias[1]),
        )?;
        self.rec.log(
            "imu_state/bias/gyro/z",
            &rerun::Scalars::single(gyro_bias[2]),
        )?;
        Ok(())
    }

    pub fn log_imu_batch(&self, batch: &ImuBatch) -> Result<(), VizLogError> {
        for sample in batch.samples() {
            self.set_time(sample.timestamp());
            let accel = sample.accel_mps2();
            let gyro = sample.gyro_radps();
            let accel_norm =
                (accel[0] * accel[0] + accel[1] * accel[1] + accel[2] * accel[2]).sqrt();
            let gyro_norm = (gyro[0] * gyro[0] + gyro[1] * gyro[1] + gyro[2] * gyro[2]).sqrt();

            self.rec
                .log("imu/raw/accel/x", &rerun::Scalars::single(accel[0]))?;
            self.rec
                .log("imu/raw/accel/y", &rerun::Scalars::single(accel[1]))?;
            self.rec
                .log("imu/raw/accel/z", &rerun::Scalars::single(accel[2]))?;
            self.rec
                .log("imu/raw/accel/norm", &rerun::Scalars::single(accel_norm))?;
            self.rec
                .log("imu/raw/gyro/x", &rerun::Scalars::single(gyro[0]))?;
            self.rec
                .log("imu/raw/gyro/y", &rerun::Scalars::single(gyro[1]))?;
            self.rec
                .log("imu/raw/gyro/z", &rerun::Scalars::single(gyro[2]))?;
            self.rec
                .log("imu/raw/gyro/norm", &rerun::Scalars::single(gyro_norm))?;
        }
        Ok(())
    }

    pub fn log_covisibility_graph(
        &mut self,
        timestamp: Timestamp,
        snapshot: &CovisibilitySnapshot,
    ) -> Result<(), VizLogError> {
        self.set_time(timestamp);

        let mut positions: HashMap<crate::map::KeyframeId, [f32; 3]> = HashMap::new();
        for node in &snapshot.nodes {
            let pos = pose_position(node.pose);
            positions.insert(node.id, pos);
        }

        if !positions.is_empty() {
            let points: Vec<[f32; 3]> = positions.values().copied().collect();
            self.rec
                .log("world/covisibility/nodes", &rerun::Points3D::new(points))?;
        }

        if !snapshot.edges.is_empty() {
            let mut strips: Vec<Vec<[f32; 3]>> = Vec::new();
            for edge in &snapshot.edges {
                let (Some(a), Some(b)) = (positions.get(&edge.a), positions.get(&edge.b)) else {
                    continue;
                };
                strips.push(vec![*a, *b]);
            }
            if !strips.is_empty() {
                self.rec.log(
                    "world/covisibility/edges",
                    &rerun::LineStrips3D::new(strips),
                )?;
            }
        }

        Ok(())
    }

    pub(crate) fn recording(&self) -> &rerun::RecordingStream {
        &self.rec
    }

    fn ensure_world_logged(&mut self) -> Result<(), VizLogError> {
        if !self.logged_world {
            let coords = rerun::archetypes::ViewCoordinates::RDF();
            self.rec.log("world", &coords)?;
            self.logged_world = true;
        }
        Ok(())
    }

    fn set_time(&self, timestamp: Timestamp) {
        self.rec.set_time(
            "capture_ns",
            rerun::TimeCell::from_duration_nanos(timestamp.as_nanos()),
        );
    }

    fn log_surface_points(
        &self,
        path: &str,
        points: &[([f32; 3], u8)],
        fallback_color: rerun::Color,
        voxel_radius: f32,
    ) -> Result<(), VizLogError> {
        let positions: Vec<[f32; 3]> = points.iter().map(|(position, _)| *position).collect();
        let colors: Vec<rerun::Color> = points
            .iter()
            .map(|(_, intensity)| rerun::Color::from_rgb(*intensity, *intensity, *intensity))
            .collect();
        let cloud = if colors.is_empty() {
            rerun::Points3D::new(positions)
                .with_colors([fallback_color])
                .with_radii([rerun::Radius::new_scene_units(voxel_radius)])
        } else {
            rerun::Points3D::new(positions)
                .with_colors(colors)
                .with_radii([rerun::Radius::new_scene_units(voxel_radius)])
        };
        self.rec.log(path, &cloud)?;
        Ok(())
    }

    fn log_surface_frame_candidates(
        &self,
        cam_from_map: Pose,
        points: &[crate::StableSurfacePoint],
        pose_gate_accepted: bool,
        voxel_radius: f32,
    ) -> Result<(), VizLogError> {
        let map_from_cam = cam_from_map.inverse();
        let rotation = map_from_cam.rotation();
        let translation = map_from_cam.translation();
        let positions: Vec<[f32; 3]> = points
            .iter()
            .map(|point| crate::math::transform_point(rotation, translation, point.position))
            .collect();
        let color = if pose_gate_accepted {
            rerun::Color::from_rgb(40, 220, 180)
        } else {
            rerun::Color::from_rgb(255, 140, 40)
        };
        let cloud = rerun::Points3D::new(positions)
            .with_colors([color])
            .with_radii([rerun::Radius::new_scene_units(
                (voxel_radius * 0.6).max(0.003),
            )]);
        self.rec
            .log("world/stable_surface_debug/frame_candidates", &cloud)?;
        Ok(())
    }

    fn log_surface_raw_frame_observations(
        &self,
        cam_from_map: Pose,
        points: &[[f32; 3]],
        voxel_radius: f32,
    ) -> Result<(), VizLogError> {
        let map_from_cam = cam_from_map.inverse();
        let rotation = map_from_cam.rotation();
        let translation = map_from_cam.translation();
        let positions: Vec<[f32; 3]> = points
            .iter()
            .map(|point| crate::math::transform_point(rotation, translation, *point))
            .collect();
        let cloud = rerun::Points3D::new(positions)
            .with_colors([rerun::Color::from_rgb(140, 180, 255)])
            .with_radii([rerun::Radius::new_scene_units(
                (voxel_radius * 0.45).max(0.0025),
            )]);
        self.rec
            .log("world/stable_surface_debug/frame_raw_observations", &cloud)?;
        Ok(())
    }

    fn log_surface_map_state(&mut self) -> Result<(), VizLogError> {
        let summary = self.surface_map.summary();
        let clouds = self.surface_map.extract_debug_clouds();
        for (path, value) in surface_summary_scalars(&summary, clouds.confirmed.len()) {
            self.rec.log(path, &rerun::Scalars::single(value))?;
        }
        let voxel_radius = (self.surface_map.config().voxel_size * 0.45).max(0.005);
        self.log_surface_points(
            "world/stable_surface_voxels",
            &clouds.confirmed,
            rerun::Color::from_rgb(220, 220, 220),
            voxel_radius,
        )?;
        self.log_surface_points(
            "world/stable_surface_debug/pending_support_voxels",
            &clouds.pending_support,
            rerun::Color::from_rgb(255, 210, 70),
            voxel_radius,
        )?;
        self.log_surface_points(
            "world/stable_surface_debug/rejected_consistency_voxels",
            &clouds.rejected_consistency,
            rerun::Color::from_rgb(255, 80, 80),
            voxel_radius,
        )?;
        self.log_surface_points(
            "world/stable_surface_debug/rejected_uncertainty_voxels",
            &clouds.rejected_uncertainty,
            rerun::Color::from_rgb(80, 170, 255),
            voxel_radius,
        )?;
        self.log_surface_points(
            "world/stable_surface_debug/rejected_consistency_and_uncertainty_voxels",
            &clouds.rejected_consistency_and_uncertainty,
            rerun::Color::from_rgb(220, 90, 220),
            voxel_radius,
        )?;

        eprintln!(
            "surface: total={} confirmed={} pending={} rejected_consistency={} rejected_uncertainty={} rejected_both={} rendered={} ratio={:.3} mean_σ={:.3}mm mean_views={:.2} mean_raw_obs={:.2} mean_consistency={:.3}",
            summary.total_voxels,
            summary.confirmed_voxels,
            summary.pending_support_voxels,
            summary.rejected_consistency_voxels,
            summary.rejected_uncertainty_voxels,
            summary.rejected_consistency_and_uncertainty_voxels,
            clouds.confirmed.len(),
            summary.confirmed_ratio,
            summary.mean_confirmed_std_dev_m * 1000.0,
            summary.mean_confirmed_support_views,
            summary.mean_confirmed_raw_observations,
            summary.mean_confirmed_consistency_score,
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct SurfacePoseQualityGate {
    min_accepted_inliers: crate::PnpAcceptedInlierCountMetric,
    max_accepted_inlier_reprojection_rmse_px: crate::PnpAcceptedInlierPixelResidualMetric,
}

impl Default for SurfacePoseQualityGate {
    fn default() -> Self {
        Self {
            min_accepted_inliers: crate::PnpAcceptedInlierCountMetric::new(8),
            max_accepted_inlier_reprojection_rmse_px:
                crate::PnpAcceptedInlierPixelResidualMetric::new(1.5)
                    .expect("default accepted-inlier reprojection RMSE gate must be lawful"),
        }
    }
}

impl SurfacePoseQualityGate {
    fn load() -> Self {
        let mut gate = Self::default();
        if let Some(count) = env_usize("KIKO_SURFACE_MIN_ACCEPTED_INLIERS")
            .or_else(|| env_usize("KIKO_SURFACE_MIN_PROJECTABLE_TRACKED_OBSERVATIONS"))
        {
            gate.min_accepted_inliers = crate::PnpAcceptedInlierCountMetric::new(count);
        }
        if let Some(value_px) = env_f32("KIKO_SURFACE_MAX_ACCEPTED_INLIER_REPROJECTION_RMSE_PX")
            .or_else(|| env_f32("KIKO_SURFACE_MAX_TRACKED_REPROJECTION_RMSE_PX"))
        {
            if let Ok(metric) = crate::PnpAcceptedInlierPixelResidualMetric::new(value_px) {
                gate.max_accepted_inlier_reprojection_rmse_px = metric;
            }
        }
        gate
    }

    fn decide(self, diagnostics: &FrameDiagnostics) -> SurfacePoseQualityDecision {
        let Some(accepted_inliers) = diagnostics.pnp_accepted_inliers else {
            return SurfacePoseQualityDecision::RejectMissingAcceptedInliers {
                min_required_accepted_inliers: self.min_accepted_inliers,
                max_allowed_accepted_inlier_reprojection_rmse_px: self
                    .max_accepted_inlier_reprojection_rmse_px,
            };
        };
        if accepted_inliers.count() < self.min_accepted_inliers.count() {
            return SurfacePoseQualityDecision::RejectLowAcceptedInliers {
                accepted_inliers,
                min_required_accepted_inliers: self.min_accepted_inliers,
                max_allowed_accepted_inlier_reprojection_rmse_px: self
                    .max_accepted_inlier_reprojection_rmse_px,
            };
        }
        if let Some(crate::BaResult::Degenerate { reason }) = diagnostics.ba_result.as_ref() {
            return SurfacePoseQualityDecision::RejectDegenerateBundleAdjustment {
                accepted_inliers,
                min_required_accepted_inliers: self.min_accepted_inliers,
                degenerate_reason: *reason,
                accepted_inlier_reprojection_rmse_px: diagnostics.pnp_inlier_reprojection_rmse_px,
                max_allowed_accepted_inlier_reprojection_rmse_px: self
                    .max_accepted_inlier_reprojection_rmse_px,
            };
        }
        match diagnostics.pnp_inlier_reprojection_rmse_px {
            Some(accepted_inlier_reprojection_rmse_px)
                if accepted_inlier_reprojection_rmse_px.value_px()
                    <= self.max_accepted_inlier_reprojection_rmse_px.value_px() =>
            {
                SurfacePoseQualityDecision::Accept {
                    accepted_inliers,
                    min_required_accepted_inliers: self.min_accepted_inliers,
                    accepted_inlier_reprojection_rmse_px,
                    max_allowed_accepted_inlier_reprojection_rmse_px: self
                        .max_accepted_inlier_reprojection_rmse_px,
                }
            }
            Some(accepted_inlier_reprojection_rmse_px) => {
                SurfacePoseQualityDecision::RejectHighAcceptedInlierReprojectionRmse {
                    accepted_inliers,
                    min_required_accepted_inliers: self.min_accepted_inliers,
                    accepted_inlier_reprojection_rmse_px,
                    max_allowed_accepted_inlier_reprojection_rmse_px: self
                        .max_accepted_inlier_reprojection_rmse_px,
                }
            }
            None => SurfacePoseQualityDecision::RejectMissingAcceptedInlierReprojectionRmse {
                accepted_inliers,
                min_required_accepted_inliers: self.min_accepted_inliers,
                max_allowed_accepted_inlier_reprojection_rmse_px: self
                    .max_accepted_inlier_reprojection_rmse_px,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum SurfacePoseQualityDecision {
    Accept {
        accepted_inliers: crate::PnpAcceptedInlierCountMetric,
        min_required_accepted_inliers: crate::PnpAcceptedInlierCountMetric,
        accepted_inlier_reprojection_rmse_px: crate::PnpAcceptedInlierPixelResidualMetric,
        max_allowed_accepted_inlier_reprojection_rmse_px:
            crate::PnpAcceptedInlierPixelResidualMetric,
    },
    RejectLowAcceptedInliers {
        accepted_inliers: crate::PnpAcceptedInlierCountMetric,
        min_required_accepted_inliers: crate::PnpAcceptedInlierCountMetric,
        max_allowed_accepted_inlier_reprojection_rmse_px:
            crate::PnpAcceptedInlierPixelResidualMetric,
    },
    RejectDegenerateBundleAdjustment {
        accepted_inliers: crate::PnpAcceptedInlierCountMetric,
        min_required_accepted_inliers: crate::PnpAcceptedInlierCountMetric,
        degenerate_reason: crate::DegenerateReason,
        accepted_inlier_reprojection_rmse_px: Option<crate::PnpAcceptedInlierPixelResidualMetric>,
        max_allowed_accepted_inlier_reprojection_rmse_px:
            crate::PnpAcceptedInlierPixelResidualMetric,
    },
    RejectHighAcceptedInlierReprojectionRmse {
        accepted_inliers: crate::PnpAcceptedInlierCountMetric,
        min_required_accepted_inliers: crate::PnpAcceptedInlierCountMetric,
        accepted_inlier_reprojection_rmse_px: crate::PnpAcceptedInlierPixelResidualMetric,
        max_allowed_accepted_inlier_reprojection_rmse_px:
            crate::PnpAcceptedInlierPixelResidualMetric,
    },
    RejectMissingAcceptedInlierReprojectionRmse {
        accepted_inliers: crate::PnpAcceptedInlierCountMetric,
        min_required_accepted_inliers: crate::PnpAcceptedInlierCountMetric,
        max_allowed_accepted_inlier_reprojection_rmse_px:
            crate::PnpAcceptedInlierPixelResidualMetric,
    },
    RejectMissingAcceptedInliers {
        min_required_accepted_inliers: crate::PnpAcceptedInlierCountMetric,
        max_allowed_accepted_inlier_reprojection_rmse_px:
            crate::PnpAcceptedInlierPixelResidualMetric,
    },
}

impl SurfacePoseQualityDecision {
    fn accepts_surface_integration(self) -> bool {
        matches!(self, Self::Accept { .. })
    }
}

fn surface_pose_quality_scalars(decision: &SurfacePoseQualityDecision) -> Vec<(&'static str, f64)> {
    let mut scalars = Vec::with_capacity(12);
    let (
        accepted,
        rejected_low_count,
        rejected_missing_count,
        rejected_missing_rmse,
        rejected_degenerate_ba_result,
        rejected_high_rmse,
        ba_degenerate_too_few_poses,
        ba_degenerate_too_few_landmarks,
        ba_degenerate_no_factors,
        accepted_inliers,
        min_required_accepted_inliers,
        rmse_px,
        max_allowed_rmse_px,
    ) = match *decision {
        SurfacePoseQualityDecision::Accept {
            accepted_inliers,
            min_required_accepted_inliers,
            accepted_inlier_reprojection_rmse_px,
            max_allowed_accepted_inlier_reprojection_rmse_px,
        } => (
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            Some(accepted_inliers.count() as f64),
            Some(min_required_accepted_inliers.count() as f64),
            Some(accepted_inlier_reprojection_rmse_px.value_px() as f64),
            Some(max_allowed_accepted_inlier_reprojection_rmse_px.value_px() as f64),
        ),
        SurfacePoseQualityDecision::RejectLowAcceptedInliers {
            accepted_inliers,
            min_required_accepted_inliers,
            max_allowed_accepted_inlier_reprojection_rmse_px,
        } => (
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            Some(accepted_inliers.count() as f64),
            Some(min_required_accepted_inliers.count() as f64),
            None,
            Some(max_allowed_accepted_inlier_reprojection_rmse_px.value_px() as f64),
        ),
        SurfacePoseQualityDecision::RejectDegenerateBundleAdjustment {
            accepted_inliers,
            min_required_accepted_inliers,
            degenerate_reason,
            accepted_inlier_reprojection_rmse_px,
            max_allowed_accepted_inlier_reprojection_rmse_px,
        } => (
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            match degenerate_reason {
                crate::DegenerateReason::TooFewPoses { .. } => 1.0,
                crate::DegenerateReason::TooFewLandmarks { .. } => 0.0,
                crate::DegenerateReason::NoFactors => 0.0,
            },
            match degenerate_reason {
                crate::DegenerateReason::TooFewPoses { .. } => 0.0,
                crate::DegenerateReason::TooFewLandmarks { .. } => 1.0,
                crate::DegenerateReason::NoFactors => 0.0,
            },
            match degenerate_reason {
                crate::DegenerateReason::TooFewPoses { .. } => 0.0,
                crate::DegenerateReason::TooFewLandmarks { .. } => 0.0,
                crate::DegenerateReason::NoFactors => 1.0,
            },
            Some(accepted_inliers.count() as f64),
            Some(min_required_accepted_inliers.count() as f64),
            accepted_inlier_reprojection_rmse_px.map(|value| value.value_px() as f64),
            Some(max_allowed_accepted_inlier_reprojection_rmse_px.value_px() as f64),
        ),
        SurfacePoseQualityDecision::RejectHighAcceptedInlierReprojectionRmse {
            accepted_inliers,
            min_required_accepted_inliers,
            accepted_inlier_reprojection_rmse_px,
            max_allowed_accepted_inlier_reprojection_rmse_px,
        } => (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            Some(accepted_inliers.count() as f64),
            Some(min_required_accepted_inliers.count() as f64),
            Some(accepted_inlier_reprojection_rmse_px.value_px() as f64),
            Some(max_allowed_accepted_inlier_reprojection_rmse_px.value_px() as f64),
        ),
        SurfacePoseQualityDecision::RejectMissingAcceptedInlierReprojectionRmse {
            accepted_inliers,
            min_required_accepted_inliers,
            max_allowed_accepted_inlier_reprojection_rmse_px,
        } => (
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            Some(accepted_inliers.count() as f64),
            Some(min_required_accepted_inliers.count() as f64),
            None,
            Some(max_allowed_accepted_inlier_reprojection_rmse_px.value_px() as f64),
        ),
        SurfacePoseQualityDecision::RejectMissingAcceptedInliers {
            min_required_accepted_inliers,
            max_allowed_accepted_inlier_reprojection_rmse_px,
        } => (
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            None,
            Some(min_required_accepted_inliers.count() as f64),
            None,
            Some(max_allowed_accepted_inlier_reprojection_rmse_px.value_px() as f64),
        ),
    };
    scalars.push(("diagnostics/surface/pose_gate/accepted", accepted));
    scalars.push((
        "diagnostics/surface/pose_gate/rejected_low_accepted_inliers",
        rejected_low_count,
    ));
    scalars.push((
        "diagnostics/surface/pose_gate/rejected_missing_accepted_inliers",
        rejected_missing_count,
    ));
    scalars.push((
        "diagnostics/surface/pose_gate/rejected_missing_accepted_inlier_reprojection_rmse",
        rejected_missing_rmse,
    ));
    scalars.push((
        "diagnostics/surface/pose_gate/rejected_degenerate_ba_result",
        rejected_degenerate_ba_result,
    ));
    scalars.push((
        "diagnostics/surface/pose_gate/rejected_high_accepted_inlier_reprojection_rmse",
        rejected_high_rmse,
    ));
    scalars.push((
        "diagnostics/surface/pose_gate/ba_degenerate_too_few_poses",
        ba_degenerate_too_few_poses,
    ));
    scalars.push((
        "diagnostics/surface/pose_gate/ba_degenerate_too_few_landmarks",
        ba_degenerate_too_few_landmarks,
    ));
    scalars.push((
        "diagnostics/surface/pose_gate/ba_degenerate_no_factors",
        ba_degenerate_no_factors,
    ));
    if let Some(value) = rmse_px {
        scalars.push((
            "diagnostics/surface/pose_gate/accepted_inlier_reprojection_rmse_px",
            value,
        ));
    }
    if let Some(value) = accepted_inliers {
        scalars.push(("diagnostics/surface/pose_gate/accepted_inliers", value));
    }
    if let Some(value) = min_required_accepted_inliers {
        scalars.push((
            "diagnostics/surface/pose_gate/min_required_accepted_inliers",
            value,
        ));
    }
    if let Some(value) = max_allowed_rmse_px {
        scalars.push((
            "diagnostics/surface/pose_gate/max_allowed_accepted_inlier_reprojection_rmse_px",
            value,
        ));
    }
    scalars
}

fn surface_integration_scalars(
    integration: &crate::surface_map::SurfaceBatchIntegrationSummary,
) -> Vec<(&'static str, f64)> {
    // `integrated_raw_observations` counts accepted raw surface samples before
    // grouped-view novelty filtering. The remaining counters are grouped-view
    // quantities, so these values are intentionally not additive.
    let mut scalars = Vec::with_capacity(6);
    scalars.push((
        "diagnostics/surface/integrated_raw_observations",
        integration.raw_observations_integrated as f64,
    ));
    scalars.push((
        "diagnostics/surface/integrated_support_views",
        integration.support_views_integrated as f64,
    ));
    scalars.push((
        "diagnostics/surface/redundant_grouped_views_ignored",
        integration.redundant_grouped_views_ignored as f64,
    ));
    scalars.push((
        "diagnostics/surface/predictive_grouped_views_rejected",
        integration.predictive_grouped_views_rejected as f64,
    ));
    if let Some(value) = integration.mean_rejected_predictive_consistency_score {
        scalars.push((
            "diagnostics/surface/rejected_predictive_grouped_views_mean_consistency_score",
            value,
        ));
    }
    if let Some(value) = integration.max_rejected_predictive_consistency_score {
        scalars.push((
            "diagnostics/surface/rejected_predictive_grouped_views_max_consistency_score",
            value,
        ));
    }
    scalars
}

fn surface_summary_scalars(
    summary: &crate::surface_map::SurfaceMapSummary,
    rendered_confirmed_voxels: usize,
) -> [(&'static str, f64); 13] {
    [
        (
            "diagnostics/surface/total_voxels",
            summary.total_voxels as f64,
        ),
        (
            "diagnostics/surface/confirmed_voxels",
            summary.confirmed_voxels as f64,
        ),
        (
            "diagnostics/surface/pending_support_voxels",
            summary.pending_support_voxels as f64,
        ),
        (
            "diagnostics/surface/rejected_consistency_voxels",
            summary.rejected_consistency_voxels as f64,
        ),
        (
            "diagnostics/surface/rejected_uncertainty_voxels",
            summary.rejected_uncertainty_voxels as f64,
        ),
        (
            "diagnostics/surface/rejected_consistency_and_uncertainty_voxels",
            summary.rejected_consistency_and_uncertainty_voxels as f64,
        ),
        (
            "diagnostics/surface/confirmed_ratio",
            summary.confirmed_ratio,
        ),
        (
            "diagnostics/surface/mean_confirmed_std_dev_mm",
            summary.mean_confirmed_std_dev_m * 1000.0,
        ),
        (
            "diagnostics/surface/mean_confirmed_support_views",
            summary.mean_confirmed_support_views,
        ),
        (
            "diagnostics/surface/mean_confirmed_raw_observations",
            summary.mean_confirmed_raw_observations,
        ),
        (
            "diagnostics/surface/mean_confirmed_consistency_score",
            summary.mean_confirmed_consistency_score,
        ),
        (
            "diagnostics/surface/max_confirmed_consistency_score",
            summary.max_confirmed_consistency_score,
        ),
        (
            "diagnostics/surface/rendered_confirmed_voxels",
            rendered_confirmed_voxels as f64,
        ),
    ]
}

fn pose_position(pose: Pose) -> [f32; 3] {
    let camera_pose = pose.inverse();
    camera_pose.translation()
}

#[derive(Debug, Default)]
struct TrajectoryLog {
    strips: Vec<Vec<[f32; 3]>>,
}

impl TrajectoryLog {
    fn push(&mut self, position: [f32; 3]) {
        if self.strips.last().is_none_or(Vec::is_empty) {
            self.strips.push(Vec::new());
        }
        self.strips
            .last_mut()
            .expect("trajectory strip exists")
            .push(position);
    }

    fn break_strip(&mut self) {
        if self.strips.last().is_some_and(|strip| !strip.is_empty()) {
            self.strips.push(Vec::new());
        }
    }

    fn strips(&self) -> Vec<Vec<[f32; 3]>> {
        self.strips
            .iter()
            .filter(|strip| !strip.is_empty())
            .cloned()
            .collect()
    }
}

fn log_pose_variant(
    rec: &rerun::RecordingStream,
    name: &str,
    pose: &Pose,
    color: rerun::Color,
    trajectory: &mut TrajectoryLog,
) -> Result<(), VizLogError> {
    let camera_pose = pose.inverse();
    let position = camera_pose.translation();
    let rotation = camera_pose.rotation();
    let quat = quat_from_rotation(rotation);

    let transform = rerun::Transform3D::update_fields()
        .with_translation(position)
        .with_quaternion(rerun::Quaternion::from_xyzw(quat));
    rec.log(format!("world/camera/{name}"), &transform)?;

    rec.log(
        format!("world/pose/{name}"),
        &rerun::Points3D::new([position])
            .with_colors([color])
            .with_radii([rerun::Radius::new_ui_points(5.0)]),
    )?;

    trajectory.push(position);
    let strips = trajectory.strips();
    if !strips.is_empty() {
        rec.log(
            format!("world/trajectory/{name}"),
            &rerun::LineStrips3D::new(strips).with_colors([color]),
        )?;
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct TrackConfig {
    max_distance_px: f32,
    min_similarity: f32,
}

impl TrackConfig {
    fn load() -> Self {
        let max_distance_px = env_f32("KIKO_TRACK_MAX_DIST").unwrap_or(24.0);
        let min_similarity = env_f32("KIKO_TRACK_MIN_SIM").unwrap_or(0.8);
        Self {
            max_distance_px,
            min_similarity,
        }
    }
}

#[derive(Debug)]
struct TrackState {
    config: TrackConfig,
    prev_left: Option<Detections>,
    prev_track_ids: Vec<u64>,
    next_track_id: u64,
}

impl TrackState {
    fn new() -> Self {
        Self {
            config: TrackConfig::load(),
            prev_left: None,
            prev_track_ids: Vec::new(),
            next_track_id: 0,
        }
    }

    fn assign_tracks(&mut self, left: &Detections) -> Vec<u64> {
        let count = left.len();
        let mut track_ids = vec![0u64; count];

        let (prev_left, prev_ids) = match self.prev_left.as_ref() {
            Some(prev) if self.prev_track_ids.len() == prev.len() => {
                (Some(prev), &self.prev_track_ids)
            }
            _ => (None, &self.prev_track_ids),
        };

        if let Some(prev) = prev_left {
            let mut used_prev = vec![false; prev.len()];
            let max_dist_sq = self.config.max_distance_px * self.config.max_distance_px;

            for (i, desc) in left.descriptors().iter().enumerate() {
                let kp = left.keypoints()[i];
                let mut best_idx = None;
                let mut best_sim = self.config.min_similarity;

                for (j, prev_desc) in prev.descriptors().iter().enumerate() {
                    if used_prev[j] {
                        continue;
                    }
                    let prev_kp = prev.keypoints()[j];
                    if distance_sq(kp, prev_kp) > max_dist_sq {
                        continue;
                    }
                    let sim = dot(desc.0.as_slice(), prev_desc.0.as_slice());
                    if sim > best_sim {
                        best_sim = sim;
                        best_idx = Some(j);
                    }
                }

                if let Some(j) = best_idx {
                    track_ids[i] = prev_ids[j];
                    used_prev[j] = true;
                } else {
                    track_ids[i] = self.next_track_id;
                    self.next_track_id = self.next_track_id.saturating_add(1);
                }
            }
        } else {
            for id in &mut track_ids {
                *id = self.next_track_id;
                self.next_track_id = self.next_track_id.saturating_add(1);
            }
        }

        self.prev_left = Some(left.clone());
        self.prev_track_ids = track_ids.clone();

        track_ids
    }
}

fn quat_from_rotation(r: [[f32; 3]; 3]) -> [f32; 4] {
    let trace = r[0][0] + r[1][1] + r[2][2];
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        let w = 0.25 * s;
        let x = (r[2][1] - r[1][2]) / s;
        let y = (r[0][2] - r[2][0]) / s;
        let z = (r[1][0] - r[0][1]) / s;
        [x, y, z, w]
    } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = (1.0 + r[0][0] - r[1][1] - r[2][2]).sqrt() * 2.0;
        let w = (r[2][1] - r[1][2]) / s;
        let x = 0.25 * s;
        let y = (r[0][1] + r[1][0]) / s;
        let z = (r[0][2] + r[2][0]) / s;
        [x, y, z, w]
    } else if r[1][1] > r[2][2] {
        let s = (1.0 + r[1][1] - r[0][0] - r[2][2]).sqrt() * 2.0;
        let w = (r[0][2] - r[2][0]) / s;
        let x = (r[0][1] + r[1][0]) / s;
        let y = 0.25 * s;
        let z = (r[1][2] + r[2][1]) / s;
        [x, y, z, w]
    } else {
        let s = (1.0 + r[2][2] - r[0][0] - r[1][1]).sqrt() * 2.0;
        let w = (r[1][0] - r[0][1]) / s;
        let x = (r[0][2] + r[2][0]) / s;
        let y = (r[1][2] + r[2][1]) / s;
        let z = 0.25 * s;
        [x, y, z, w]
    }
}

fn log_matches(
    rec: &rerun::RecordingStream,
    packet: &VizPacket<Raw>,
    x_offset: f32,
    track_ids: &[u64],
) -> Result<(), rerun::RecordingStreamError> {
    let matches = packet.matches();
    if matches.is_empty() {
        return Ok(());
    }

    let keypoints_left = matches.source_a().keypoints();
    let keypoints_right = matches.source_b().keypoints();
    let mut strips = Vec::with_capacity(matches.len());

    for &(idx_left, idx_right) in matches.indices() {
        if let (Some(kp_left), Some(kp_right)) =
            (keypoints_left.get(idx_left), keypoints_right.get(idx_right))
        {
            strips.push(vec![
                [kp_left.x, kp_left.y],
                [kp_right.x + x_offset, kp_right.y],
            ]);
        }
    }

    if strips.is_empty() {
        return Ok(());
    }

    let palette = [
        rerun::Color::from_rgb(230, 57, 70),
        rerun::Color::from_rgb(241, 250, 238),
        rerun::Color::from_rgb(168, 218, 220),
        rerun::Color::from_rgb(69, 123, 157),
        rerun::Color::from_rgb(29, 53, 87),
        rerun::Color::from_rgb(255, 183, 3),
        rerun::Color::from_rgb(251, 86, 7),
        rerun::Color::from_rgb(131, 56, 236),
    ];
    let colors: Vec<rerun::Color> = matches
        .indices()
        .iter()
        .map(|(idx_left, _)| {
            let track_id = track_ids.get(*idx_left).copied().unwrap_or(0);
            palette[(track_id as usize) % palette.len()]
        })
        .collect();

    rec.log(
        "view/matches",
        &rerun::LineStrips2D::new(strips)
            .with_colors(colors)
            .with_radii([rerun::Radius::new_ui_points(1.5)])
            .with_draw_order(10.0),
    )?;

    Ok(())
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn distance_sq(a: Keypoint, b: Keypoint) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    dx * dx + dy * dy
}

fn stitch_luma(left: &Frame, right: &Frame) -> (Vec<u8>, u32, u32) {
    let left_width = left.width();
    let left_height = left.height();
    let right_width = right.width();
    let right_height = right.height();

    let out_width = left_width + right_width;
    let out_height = left_height.max(right_height);

    let out_width_usize = out_width as usize;
    let out_height_usize = out_height as usize;
    let left_width_usize = left_width as usize;
    let left_height_usize = left_height as usize;
    let right_width_usize = right_width as usize;
    let right_height_usize = right_height as usize;

    let mut out = vec![0u8; out_width_usize * out_height_usize];

    let left_data = left.data();
    let right_data = right.data();

    for y in 0..out_height_usize {
        let out_row = y * out_width_usize;
        if y < left_height_usize {
            let left_row = y * left_width_usize;
            out[out_row..out_row + left_width_usize]
                .copy_from_slice(&left_data[left_row..left_row + left_width_usize]);
        }
        if y < right_height_usize {
            let right_row = y * right_width_usize;
            let out_right = out_row + left_width_usize;
            out[out_right..out_right + right_width_usize]
                .copy_from_slice(&right_data[right_row..right_row + right_width_usize]);
        }
    }

    (out, out_width, out_height)
}

#[cfg(test)]
mod tests {
    use super::{
        RerunSink, SurfacePoseQualityDecision, SurfacePoseQualityGate, VizDecimation,
        surface_integration_scalars, surface_pose_quality_scalars, surface_summary_scalars,
    };
    use crate::{
        Frame, FrameDiagnostics, FrameId, Pose, RectifiedRowMismatchPx, SensorId,
        StableSurfacePoint, StableSurfaceStats, Timestamp,
        surface_map::{SurfaceBatchIntegrationSummary, SurfaceMapSummary},
    };

    #[test]
    fn log_frames_emits_left_and_right_view_entities() {
        let (rec, storage) = rerun::RecordingStreamBuilder::new("kiko-slam-viz-test")
            .memory()
            .expect("in-memory rerun stream");
        let mut sink = RerunSink::new(rec, VizDecimation::default());
        let left = Frame::new(
            SensorId::StereoLeft,
            FrameId::new(1),
            Timestamp::from_nanos(1),
            2,
            2,
            vec![0, 16, 32, 48],
        )
        .expect("left frame");
        let right = Frame::new(
            SensorId::StereoRight,
            FrameId::new(2),
            Timestamp::from_nanos(1),
            2,
            2,
            vec![255, 192, 128, 64],
        )
        .expect("right frame");

        sink.log_frames(&left, &right).expect("frame logging");

        let entity_paths = recorded_entity_paths(storage);
        assert!(entity_paths.iter().any(|path| path == "/view/left"));
        assert!(entity_paths.iter().any(|path| path == "/view/right"));
    }

    fn recorded_entity_paths(storage: rerun::sink::MemorySinkStorage) -> Vec<String> {
        storage
            .take()
            .into_iter()
            .filter_map(|msg| match msg {
                rerun::external::re_log_types::LogMsg::ArrowMsg(_, arrow_msg) => Some(
                    rerun::log::Chunk::from_arrow_msg(&arrow_msg)
                        .expect("valid arrow chunk")
                        .entity_path()
                        .to_string(),
                ),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn surface_integration_scalars_export_honest_support_accounting() {
        let integration = SurfaceBatchIntegrationSummary {
            raw_observations_integrated: 11,
            support_views_integrated: 3,
            redundant_grouped_views_ignored: 2,
            predictive_grouped_views_rejected: 1,
            mean_rejected_predictive_consistency_score: Some(14.5),
            max_rejected_predictive_consistency_score: Some(14.5),
        };

        let scalars = surface_integration_scalars(&integration);
        assert!(scalars.contains(&("diagnostics/surface/integrated_raw_observations", 11.0)));
        assert!(scalars.contains(&("diagnostics/surface/integrated_support_views", 3.0)));
        assert!(scalars.contains(&("diagnostics/surface/redundant_grouped_views_ignored", 2.0)));
        assert!(scalars.contains(&("diagnostics/surface/predictive_grouped_views_rejected", 1.0)));
        assert!(scalars.contains(&(
            "diagnostics/surface/rejected_predictive_grouped_views_mean_consistency_score",
            14.5
        )));
        assert!(scalars.contains(&(
            "diagnostics/surface/rejected_predictive_grouped_views_max_consistency_score",
            14.5
        )));
    }

    #[test]
    fn surface_summary_scalars_export_confirmed_consistency_metrics() {
        let summary = SurfaceMapSummary {
            total_voxels: 10,
            confirmed_voxels: 4,
            pending_support_voxels: 3,
            rejected_consistency_voxels: 2,
            rejected_uncertainty_voxels: 1,
            rejected_consistency_and_uncertainty_voxels: 0,
            confirmed_ratio: 0.4,
            mean_confirmed_std_dev_m: 0.012,
            mean_confirmed_support_views: 3.5,
            mean_confirmed_raw_observations: 6.0,
            mean_confirmed_consistency_score: 1.25,
            max_confirmed_consistency_score: 2.75,
        };

        let scalars = surface_summary_scalars(&summary, 3);
        assert!(scalars.contains(&("diagnostics/surface/total_voxels", 10.0)));
        assert!(scalars.contains(&("diagnostics/surface/confirmed_voxels", 4.0)));
        assert!(scalars.contains(&("diagnostics/surface/pending_support_voxels", 3.0)));
        assert!(scalars.contains(&("diagnostics/surface/rejected_consistency_voxels", 2.0)));
        assert!(scalars.contains(&("diagnostics/surface/rejected_uncertainty_voxels", 1.0)));
        assert!(scalars.contains(&(
            "diagnostics/surface/rejected_consistency_and_uncertainty_voxels",
            0.0
        )));
        assert!(scalars.contains(&("diagnostics/surface/confirmed_ratio", 0.4)));
        assert!(scalars.contains(&("diagnostics/surface/mean_confirmed_std_dev_mm", 12.0)));
        assert!(scalars.contains(&("diagnostics/surface/mean_confirmed_support_views", 3.5)));
        assert!(scalars.contains(&("diagnostics/surface/mean_confirmed_raw_observations", 6.0)));
        assert!(scalars.contains(&("diagnostics/surface/mean_confirmed_consistency_score", 1.25)));
        assert!(scalars.contains(&("diagnostics/surface/max_confirmed_consistency_score", 2.75)));
        assert!(scalars.contains(&("diagnostics/surface/rendered_confirmed_voxels", 3.0)));
    }

    #[test]
    fn surface_pose_quality_gate_accepts_low_tracked_reprojection_rmse() {
        let gate = SurfacePoseQualityGate::default();
        let mut diagnostics = FrameDiagnostics::empty(0, 0);
        diagnostics.pnp_accepted_inliers = Some(crate::PnpAcceptedInlierCountMetric::new(12));
        diagnostics.pnp_inlier_reprojection_rmse_px =
            Some(crate::PnpAcceptedInlierPixelResidualMetric::new(1.0).expect("rmse"));

        assert!(matches!(
            gate.decide(&diagnostics),
            SurfacePoseQualityDecision::Accept { .. }
        ));
    }

    #[test]
    fn surface_pose_quality_gate_rejects_missing_accepted_inlier_reprojection_rmse() {
        let gate = SurfacePoseQualityGate::default();
        let mut diagnostics = FrameDiagnostics::empty(0, 0);
        diagnostics.pnp_accepted_inliers = Some(crate::PnpAcceptedInlierCountMetric::new(12));

        assert!(matches!(
            gate.decide(&diagnostics),
            SurfacePoseQualityDecision::RejectMissingAcceptedInlierReprojectionRmse { .. }
        ));
    }

    #[test]
    fn surface_pose_quality_gate_rejects_missing_accepted_inliers() {
        let gate = SurfacePoseQualityGate::default();
        let diagnostics = FrameDiagnostics::empty(0, 0);

        assert!(matches!(
            gate.decide(&diagnostics),
            SurfacePoseQualityDecision::RejectMissingAcceptedInliers { .. }
        ));
    }

    #[test]
    fn surface_pose_quality_gate_rejects_low_accepted_inliers() {
        let gate = SurfacePoseQualityGate::default();
        let mut diagnostics = FrameDiagnostics::empty(0, 0);
        diagnostics.pnp_accepted_inliers = Some(crate::PnpAcceptedInlierCountMetric::new(4));

        assert!(matches!(
            gate.decide(&diagnostics),
            SurfacePoseQualityDecision::RejectLowAcceptedInliers { .. }
        ));
    }

    #[test]
    fn surface_pose_quality_gate_rejects_high_accepted_inlier_reprojection_rmse() {
        let gate = SurfacePoseQualityGate::default();
        let mut diagnostics = FrameDiagnostics::empty(0, 0);
        diagnostics.pnp_accepted_inliers = Some(crate::PnpAcceptedInlierCountMetric::new(12));
        diagnostics.pnp_inlier_reprojection_rmse_px =
            Some(crate::PnpAcceptedInlierPixelResidualMetric::new(2.0).expect("rmse"));

        assert!(matches!(
            gate.decide(&diagnostics),
            SurfacePoseQualityDecision::RejectHighAcceptedInlierReprojectionRmse { .. }
        ));
    }

    #[test]
    fn surface_pose_quality_gate_rejects_degenerate_ba_result() {
        let gate = SurfacePoseQualityGate::default();
        let mut diagnostics = FrameDiagnostics::empty(0, 0);
        diagnostics.pnp_accepted_inliers = Some(crate::PnpAcceptedInlierCountMetric::new(12));
        diagnostics.ba_result = Some(crate::BaResult::Degenerate {
            reason: crate::DegenerateReason::TooFewLandmarks { count: 3 },
        });

        assert!(matches!(
            gate.decide(&diagnostics),
            SurfacePoseQualityDecision::RejectDegenerateBundleAdjustment {
                degenerate_reason: crate::DegenerateReason::TooFewLandmarks { count: 3 },
                accepted_inlier_reprojection_rmse_px: None,
                ..
            }
        ));
    }

    #[test]
    fn surface_pose_quality_scalars_export_decision_and_threshold() {
        let decision =
            SurfacePoseQualityDecision::RejectHighAcceptedInlierReprojectionRmse {
                accepted_inliers: crate::PnpAcceptedInlierCountMetric::new(12),
                min_required_accepted_inliers: crate::PnpAcceptedInlierCountMetric::new(8),
                accepted_inlier_reprojection_rmse_px:
                    crate::PnpAcceptedInlierPixelResidualMetric::new(2.0).expect("rmse"),
                max_allowed_accepted_inlier_reprojection_rmse_px:
                    crate::PnpAcceptedInlierPixelResidualMetric::new(1.5).expect("rmse"),
            };

        let scalars = surface_pose_quality_scalars(&decision);
        assert!(scalars.contains(&("diagnostics/surface/pose_gate/accepted", 0.0)));
        assert!(scalars.contains(&(
            "diagnostics/surface/pose_gate/rejected_low_accepted_inliers",
            0.0
        )));
        assert!(scalars.contains(&(
            "diagnostics/surface/pose_gate/rejected_missing_accepted_inliers",
            0.0
        )));
        assert!(scalars.contains(&(
            "diagnostics/surface/pose_gate/rejected_missing_accepted_inlier_reprojection_rmse",
            0.0
        )));
        assert!(scalars.contains(&(
            "diagnostics/surface/pose_gate/rejected_degenerate_ba_result",
            0.0
        )));
        assert!(scalars.contains(&(
            "diagnostics/surface/pose_gate/rejected_high_accepted_inlier_reprojection_rmse",
            1.0
        )));
        assert!(scalars.contains(&(
            "diagnostics/surface/pose_gate/ba_degenerate_too_few_poses",
            0.0
        )));
        assert!(scalars.contains(&(
            "diagnostics/surface/pose_gate/ba_degenerate_too_few_landmarks",
            0.0
        )));
        assert!(scalars.contains(&(
            "diagnostics/surface/pose_gate/ba_degenerate_no_factors",
            0.0
        )));
        assert!(scalars.contains(&(
            "diagnostics/surface/pose_gate/accepted_inlier_reprojection_rmse_px",
            2.0
        )));
        assert!(scalars.contains(&("diagnostics/surface/pose_gate/accepted_inliers", 12.0)));
        assert!(scalars.contains(&(
            "diagnostics/surface/pose_gate/min_required_accepted_inliers",
            8.0
        )));
        assert!(scalars.contains(&(
            "diagnostics/surface/pose_gate/max_allowed_accepted_inlier_reprojection_rmse_px",
            1.5
        )));
    }

    #[test]
    fn surface_pose_quality_scalars_export_ba_degenerate_reason() {
        let decision = SurfacePoseQualityDecision::RejectDegenerateBundleAdjustment {
            accepted_inliers: crate::PnpAcceptedInlierCountMetric::new(12),
            min_required_accepted_inliers: crate::PnpAcceptedInlierCountMetric::new(8),
            degenerate_reason: crate::DegenerateReason::NoFactors,
            accepted_inlier_reprojection_rmse_px: Some(
                crate::PnpAcceptedInlierPixelResidualMetric::new(1.25).expect("rmse"),
            ),
            max_allowed_accepted_inlier_reprojection_rmse_px:
                crate::PnpAcceptedInlierPixelResidualMetric::new(1.5).expect("rmse"),
        };

        let scalars = surface_pose_quality_scalars(&decision);
        assert!(scalars.contains(&(
            "diagnostics/surface/pose_gate/rejected_degenerate_ba_result",
            1.0
        )));
        assert!(scalars.contains(&(
            "diagnostics/surface/pose_gate/ba_degenerate_too_few_poses",
            0.0
        )));
        assert!(scalars.contains(&(
            "diagnostics/surface/pose_gate/ba_degenerate_too_few_landmarks",
            0.0
        )));
        assert!(scalars.contains(&(
            "diagnostics/surface/pose_gate/ba_degenerate_no_factors",
            1.0
        )));
        assert!(scalars.contains(&(
            "diagnostics/surface/pose_gate/accepted_inlier_reprojection_rmse_px",
            1.25
        )));
    }

    #[test]
    fn log_surface_observations_reject_path_leaves_surface_map_unchanged() {
        let (rec, storage) = rerun::RecordingStreamBuilder::new("kiko-slam-viz-test")
            .memory()
            .expect("in-memory rerun stream");
        let mut sink = RerunSink::new(rec, VizDecimation::default());
        let diagnostics = FrameDiagnostics::empty(0, 0);
        let point = StableSurfacePoint {
            position: [0.0, 0.0, 2.0],
            intensity: 180,
            position_variance: 0.0025,
            rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.0).expect("row mismatch"),
        };

        sink.log_surface_observations(
            Timestamp::from_nanos(1),
            &[[0.0, 0.0, 2.0]],
            &[point],
            &StableSurfaceStats {
                input_samples: 1,
                points_generated: 1,
                ..StableSurfaceStats::default()
            },
            Pose::identity(),
            &diagnostics,
            true,
            false,
        )
        .expect("surface logging");

        assert_eq!(sink.surface_map.num_voxels(), 0);
        assert!(storage.num_msgs() > 0);
    }

    #[test]
    fn log_surface_observations_degenerate_ba_path_leaves_surface_map_unchanged() {
        let (rec, storage) = rerun::RecordingStreamBuilder::new("kiko-slam-viz-test")
            .memory()
            .expect("in-memory rerun stream");
        let mut sink = RerunSink::new(rec, VizDecimation::default());
        let mut diagnostics = FrameDiagnostics::empty(0, 0);
        diagnostics.pnp_accepted_inliers = Some(crate::PnpAcceptedInlierCountMetric::new(12));
        diagnostics.pnp_inlier_reprojection_rmse_px =
            Some(crate::PnpAcceptedInlierPixelResidualMetric::new(1.0).expect("rmse"));
        diagnostics.ba_result = Some(crate::BaResult::Degenerate {
            reason: crate::DegenerateReason::NoFactors,
        });
        let point = StableSurfacePoint {
            position: [0.0, 0.0, 2.0],
            intensity: 180,
            position_variance: 0.0025,
            rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.0).expect("row mismatch"),
        };

        sink.log_surface_observations(
            Timestamp::from_nanos(1),
            &[[0.0, 0.0, 2.0]],
            &[point],
            &StableSurfaceStats {
                input_samples: 1,
                points_generated: 1,
                ..StableSurfaceStats::default()
            },
            Pose::identity(),
            &diagnostics,
            true,
            false,
        )
        .expect("surface logging");

        assert_eq!(sink.surface_map.num_voxels(), 0);
        assert!(storage.num_msgs() > 0);
    }

    #[test]
    fn log_surface_observations_accept_path_mutates_surface_map() {
        let (rec, storage) = rerun::RecordingStreamBuilder::new("kiko-slam-viz-test")
            .memory()
            .expect("in-memory rerun stream");
        let mut sink = RerunSink::new(rec, VizDecimation::default());
        let mut diagnostics = FrameDiagnostics::empty(0, 0);
        diagnostics.pnp_accepted_inliers = Some(crate::PnpAcceptedInlierCountMetric::new(12));
        diagnostics.pnp_inlier_reprojection_rmse_px =
            Some(crate::PnpAcceptedInlierPixelResidualMetric::new(1.0).expect("rmse"));
        let point = StableSurfacePoint {
            position: [0.0, 0.0, 2.0],
            intensity: 180,
            position_variance: 0.0025,
            rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.0).expect("row mismatch"),
        };

        sink.log_surface_observations(
            Timestamp::from_nanos(1),
            &[[0.0, 0.0, 2.0]],
            &[point],
            &StableSurfaceStats {
                input_samples: 1,
                points_generated: 1,
                ..StableSurfaceStats::default()
            },
            Pose::identity(),
            &diagnostics,
            true,
            true,
        )
        .expect("surface logging");

        assert_eq!(sink.surface_map.num_voxels(), 1);
        assert!(storage.num_msgs() > 0);
    }

    #[test]
    fn log_surface_observations_support_frame_mutates_surface_map_without_slam_keyframe() {
        let (rec, _storage) = rerun::RecordingStreamBuilder::new("kiko-slam-viz-test")
            .memory()
            .expect("in-memory rerun stream");
        let mut sink = RerunSink::new(rec, VizDecimation::default());
        let mut diagnostics = FrameDiagnostics::empty(0, 0);
        diagnostics.pnp_accepted_inliers = Some(crate::PnpAcceptedInlierCountMetric::new(12));
        diagnostics.pnp_inlier_reprojection_rmse_px =
            Some(crate::PnpAcceptedInlierPixelResidualMetric::new(1.0).expect("rmse"));
        let point = StableSurfacePoint {
            position: [0.0, 0.0, 2.0],
            intensity: 180,
            position_variance: 0.0025,
            rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.0).expect("row mismatch"),
        };

        sink.log_surface_observations(
            Timestamp::from_nanos(1),
            &[[0.0, 0.0, 2.0]],
            &[point],
            &StableSurfaceStats {
                input_samples: 1,
                points_generated: 1,
                ..StableSurfaceStats::default()
            },
            Pose::identity(),
            &diagnostics,
            true,
            false,
        )
        .expect("surface logging");

        assert_eq!(
            sink.surface_map.num_voxels(),
            1,
            "support-frame surface integration must not depend on SLAM keyframe creation"
        );
    }

    #[test]
    fn log_surface_observations_logs_debug_entities_before_confirmation() {
        let (rec, storage) = rerun::RecordingStreamBuilder::new("kiko-slam-viz-test")
            .memory()
            .expect("in-memory rerun stream");
        let mut sink = RerunSink::new(rec, VizDecimation::default());
        let mut diagnostics = FrameDiagnostics::empty(0, 0);
        diagnostics.pnp_accepted_inliers = Some(crate::PnpAcceptedInlierCountMetric::new(12));
        diagnostics.pnp_inlier_reprojection_rmse_px =
            Some(crate::PnpAcceptedInlierPixelResidualMetric::new(1.0).expect("rmse"));
        let point = StableSurfacePoint {
            position: [0.0, 0.0, 2.0],
            intensity: 180,
            position_variance: 0.0025,
            rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.0).expect("row mismatch"),
        };

        sink.log_surface_observations(
            Timestamp::from_nanos(1),
            &[[0.0, 0.0, 2.0]],
            &[point],
            &StableSurfaceStats {
                input_samples: 1,
                points_generated: 1,
                ..StableSurfaceStats::default()
            },
            Pose::identity(),
            &diagnostics,
            true,
            false,
        )
        .expect("surface logging");

        let entity_paths = recorded_entity_paths(storage);
        assert!(
            entity_paths
                .iter()
                .any(|path| path == "/world/stable_surface_debug/frame_raw_observations")
        );
        assert!(
            entity_paths
                .iter()
                .any(|path| path == "/world/stable_surface_debug/frame_candidates")
        );
        assert!(
            entity_paths
                .iter()
                .any(|path| path == "/world/stable_surface_debug/pending_support_voxels")
        );
    }

    #[test]
    fn log_surface_observations_logs_raw_frame_observations_without_retained_candidates() {
        let (rec, storage) = rerun::RecordingStreamBuilder::new("kiko-slam-viz-test")
            .memory()
            .expect("in-memory rerun stream");
        let mut sink = RerunSink::new(rec, VizDecimation::default());
        let mut diagnostics = FrameDiagnostics::empty(0, 0);
        diagnostics.pnp_accepted_inliers = Some(crate::PnpAcceptedInlierCountMetric::new(12));
        diagnostics.pnp_inlier_reprojection_rmse_px =
            Some(crate::PnpAcceptedInlierPixelResidualMetric::new(1.0).expect("rmse"));

        sink.log_surface_observations(
            Timestamp::from_nanos(1),
            &[[0.0, 0.0, 2.0]],
            &[],
            &StableSurfaceStats {
                input_samples: 1,
                points_generated: 0,
                ..StableSurfaceStats::default()
            },
            Pose::identity(),
            &diagnostics,
            false,
            false,
        )
        .expect("surface logging");

        let entity_paths = recorded_entity_paths(storage);
        assert!(
            entity_paths
                .iter()
                .any(|path| path == "/world/stable_surface_debug/frame_raw_observations")
        );
    }

    #[test]
    fn log_surface_observations_visual_only_path_logs_candidates_without_mutating_map() {
        let (rec, storage) = rerun::RecordingStreamBuilder::new("kiko-slam-viz-test")
            .memory()
            .expect("in-memory rerun stream");
        let mut sink = RerunSink::new(rec, VizDecimation::default());
        let mut diagnostics = FrameDiagnostics::empty(0, 0);
        diagnostics.pnp_accepted_inliers = Some(crate::PnpAcceptedInlierCountMetric::new(12));
        diagnostics.pnp_inlier_reprojection_rmse_px =
            Some(crate::PnpAcceptedInlierPixelResidualMetric::new(1.0).expect("rmse"));
        let point = StableSurfacePoint {
            position: [0.0, 0.0, 2.0],
            intensity: 180,
            position_variance: 0.0025,
            rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.0).expect("row mismatch"),
        };

        sink.log_surface_observations(
            Timestamp::from_nanos(1),
            &[[0.0, 0.0, 2.0]],
            &[point],
            &StableSurfaceStats {
                input_samples: 1,
                points_generated: 1,
                ..StableSurfaceStats::default()
            },
            Pose::identity(),
            &diagnostics,
            false,
            false,
        )
        .expect("surface logging");

        assert_eq!(sink.surface_map.num_voxels(), 0);
        let entity_paths = recorded_entity_paths(storage);
        assert!(
            entity_paths
                .iter()
                .any(|path| path == "/world/stable_surface_debug/frame_candidates")
        );
    }
}
