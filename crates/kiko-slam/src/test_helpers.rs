#![allow(dead_code)]

use std::sync::Arc;

use crate::dataset::{Calibration, CameraIntrinsics};
use crate::{
    Descriptor, DetectionError, Detections, FrameId, IntrinsicsError, Keypoint, MatchError,
    Matches, Observation, PinholeIntrinsics, PnpError, Point3, Pose, Raw, RectifiedStereo,
    RectifiedStereoConfig, RectifiedStereoError, SensorId, Timestamp, math,
};

#[derive(Debug)]
pub(crate) enum TestHelperError {
    InvalidIntrinsics(IntrinsicsError),
    InvalidRectifiedStereo(RectifiedStereoError),
    InvalidDetections(DetectionError),
    InvalidMatches(MatchError),
    Observation(PnpError),
    NonPositiveSpacing { spacing: f32 },
    NonPositiveDepth { depth: f32 },
}

impl std::fmt::Display for TestHelperError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TestHelperError::InvalidIntrinsics(err) => write!(f, "{err}"),
            TestHelperError::InvalidRectifiedStereo(err) => write!(f, "{err}"),
            TestHelperError::InvalidDetections(err) => write!(f, "{err:?}"),
            TestHelperError::InvalidMatches(err) => write!(f, "{err:?}"),
            TestHelperError::Observation(err) => write!(f, "{err}"),
            TestHelperError::NonPositiveSpacing { spacing } => {
                write!(f, "grid spacing must be > 0, got {spacing}")
            }
            TestHelperError::NonPositiveDepth { depth } => {
                write!(f, "grid depth must be > 0, got {depth}")
            }
        }
    }
}

impl std::error::Error for TestHelperError {}

impl From<IntrinsicsError> for TestHelperError {
    fn from(value: IntrinsicsError) -> Self {
        Self::InvalidIntrinsics(value)
    }
}

impl From<RectifiedStereoError> for TestHelperError {
    fn from(value: RectifiedStereoError) -> Self {
        Self::InvalidRectifiedStereo(value)
    }
}

impl From<DetectionError> for TestHelperError {
    fn from(value: DetectionError) -> Self {
        Self::InvalidDetections(value)
    }
}

impl From<MatchError> for TestHelperError {
    fn from(value: MatchError) -> Self {
        Self::InvalidMatches(value)
    }
}

impl From<PnpError> for TestHelperError {
    fn from(value: PnpError) -> Self {
        Self::Observation(value)
    }
}

pub(crate) fn make_camera_intrinsics(
    width: u32,
    height: u32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
) -> CameraIntrinsics {
    CameraIntrinsics {
        fx,
        fy,
        cx,
        cy,
        width,
        height,
    }
}

pub(crate) fn make_pinhole_intrinsics(
    width: u32,
    height: u32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
) -> Result<PinholeIntrinsics, TestHelperError> {
    let intrinsics = make_camera_intrinsics(width, height, fx, fy, cx, cy);
    Ok(PinholeIntrinsics::try_from(&intrinsics)?)
}

pub(crate) fn make_rectified_stereo(
    width: u32,
    height: u32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    baseline_m: f32,
) -> Result<RectifiedStereo, TestHelperError> {
    let left = make_camera_intrinsics(width, height, fx, fy, cx, cy);
    let right = make_camera_intrinsics(width, height, fx, fy, cx, cy);
    let calibration = Calibration {
        left,
        right,
        baseline_m,
        rectified: true,
    };
    Ok(RectifiedStereo::from_calibration_with_config(
        &calibration,
        RectifiedStereoConfig::default(),
    )?)
}

pub(crate) fn grid_points_xy(
    rows: usize,
    cols: usize,
    spacing_m: f32,
    depth_m: f32,
) -> Result<Vec<Point3>, TestHelperError> {
    if !spacing_m.is_finite() || spacing_m <= 0.0 {
        return Err(TestHelperError::NonPositiveSpacing { spacing: spacing_m });
    }
    if !depth_m.is_finite() || depth_m <= 0.0 {
        return Err(TestHelperError::NonPositiveDepth { depth: depth_m });
    }
    let mut points = Vec::with_capacity(rows.saturating_mul(cols));
    let x0 = -(cols as f32 - 1.0) * spacing_m * 0.5;
    let y0 = -(rows as f32 - 1.0) * spacing_m * 0.5;
    for r in 0..rows {
        for c in 0..cols {
            points.push(Point3 {
                x: x0 + c as f32 * spacing_m,
                y: y0 + r as f32 * spacing_m,
                z: depth_m,
            });
        }
    }
    Ok(points)
}

pub(crate) fn project_world_point(
    pose_world_to_camera: Pose,
    point_world: Point3,
    intrinsics: PinholeIntrinsics,
) -> Option<Keypoint> {
    let pc = math::transform_point(
        pose_world_to_camera.rotation(),
        pose_world_to_camera.translation(),
        [point_world.x, point_world.y, point_world.z],
    );
    if pc[2] <= 1e-6 {
        return None;
    }
    Some(Keypoint {
        x: intrinsics.fx() * (pc[0] / pc[2]) + intrinsics.cx(),
        y: intrinsics.fy() * (pc[1] / pc[2]) + intrinsics.cy(),
    })
}

pub(crate) fn project_world_points(
    pose_world_to_camera: Pose,
    points_world: &[Point3],
    intrinsics: PinholeIntrinsics,
) -> Vec<(usize, Keypoint)> {
    let mut out = Vec::new();
    for (idx, &point) in points_world.iter().enumerate() {
        if let Some(pixel) = project_world_point(pose_world_to_camera, point, intrinsics) {
            out.push((idx, pixel));
        }
    }
    out
}

pub(crate) fn observations_from_projection(
    pose_world_to_camera: Pose,
    points_world: &[Point3],
    intrinsics: PinholeIntrinsics,
) -> Result<Vec<Observation>, TestHelperError> {
    let mut observations = Vec::new();
    for point in points_world {
        if let Some(pixel) = project_world_point(pose_world_to_camera, *point, intrinsics) {
            observations.push(Observation::try_new(*point, pixel, intrinsics)?);
        }
    }
    Ok(observations)
}

pub(crate) fn rectified_stereo_keypoints_from_points(
    points_left_camera: &[Point3],
    intrinsics: PinholeIntrinsics,
    baseline_m: f32,
) -> Vec<(usize, Keypoint, Keypoint)> {
    let mut out = Vec::new();
    for (idx, &point) in points_left_camera.iter().enumerate() {
        if point.z <= 1e-6 {
            continue;
        }
        let left_x = intrinsics.fx() * (point.x / point.z) + intrinsics.cx();
        let left_y = intrinsics.fy() * (point.y / point.z) + intrinsics.cy();
        let disparity = intrinsics.fx() * baseline_m / point.z;
        let right_x = left_x - disparity;
        out.push((
            idx,
            Keypoint {
                x: left_x,
                y: left_y,
            },
            Keypoint {
                x: right_x,
                y: left_y,
            },
        ));
    }
    out
}

pub(crate) fn make_detections(
    sensor_id: SensorId,
    frame_id: FrameId,
    width: u32,
    height: u32,
    keypoints: Vec<Keypoint>,
) -> Result<Arc<Detections>, TestHelperError> {
    let mut scores = Vec::with_capacity(keypoints.len());
    let mut descriptors = Vec::with_capacity(keypoints.len());

    for i in 0..keypoints.len() {
        scores.push(1.0 - (i as f32 * 1e-4));
        descriptors.push(indexed_descriptor(i));
    }

    Ok(Arc::new(Detections::new(
        sensor_id,
        frame_id,
        width,
        height,
        keypoints,
        scores,
        descriptors,
    )?))
}

pub(crate) fn make_raw_matches(
    left: Arc<Detections>,
    right: Arc<Detections>,
    pairs: Vec<(usize, usize)>,
) -> Result<Matches<Raw>, TestHelperError> {
    let scores = vec![1.0_f32; pairs.len()];
    Ok(Matches::new(left, right, pairs, scores)?)
}

pub(crate) fn axis_angle_pose(translation: [f32; 3], axis_angle: [f32; 3]) -> Pose {
    Pose::from_rt(so3_exp(axis_angle), translation)
}

fn so3_exp(w: [f32; 3]) -> [[f32; 3]; 3] {
    let theta = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
    if theta < 1e-6 {
        return [[1.0, -w[2], w[1]], [w[2], 1.0, -w[0]], [-w[1], w[0], 1.0]];
    }
    let k = [w[0] / theta, w[1] / theta, w[2] / theta];
    let kx = [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]];
    let sin_t = theta.sin();
    let cos_t = theta.cos();
    let mut kx2 = [[0.0_f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            kx2[i][j] = kx[i][0] * kx[0][j] + kx[i][1] * kx[1][j] + kx[i][2] * kx[2][j];
        }
    }
    let mut r = [[0.0_f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = if i == j { 1.0 } else { 0.0 } + sin_t * kx[i][j] + (1.0 - cos_t) * kx2[i][j];
        }
    }
    r
}

pub(crate) fn make_depth_image(
    frame_id: FrameId,
    timestamp: Timestamp,
    width: u32,
    height: u32,
    fill_value: f32,
) -> crate::DepthImage {
    let pixels = (width as usize) * (height as usize);
    let data = vec![fill_value; pixels];
    crate::DepthImage::new(frame_id, timestamp, width, height, data)
        .expect("make_depth_image: valid dimensions")
}

fn indexed_descriptor(index: usize) -> Descriptor {
    let mut data = [0.0_f32; 256];
    data[index % 256] = 1.0;
    Descriptor(data)
}
