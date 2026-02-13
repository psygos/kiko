use std::sync::Arc;

use crate::dataset::{Calibration, CameraIntrinsics};
use crate::{Detections, FrameId, Keypoint, Matches, Raw, SensorId};

#[derive(Clone, Copy, Debug)]
pub struct RectifiedStereoConfig {
    pub max_principal_delta_px: Option<f32>,
    pub allow_unrectified: bool,
}

impl Default for RectifiedStereoConfig {
    fn default() -> Self {
        Self {
            max_principal_delta_px: None,
            allow_unrectified: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RectifiedStereo {
    left: CameraIntrinsics,
    right: CameraIntrinsics,
    baseline_m: f32,
}

#[derive(Debug)]
pub enum RectifiedStereoError {
    NonPositiveBaseline { baseline_m: f32 },
    DimensionMismatch {
        left: (u32, u32),
        right: (u32, u32),
    },
    InvalidFocal {
        fx: f32,
        fy: f32,
    },
    NotRectified,
    PrincipalPointMismatch {
        delta_cx: f32,
        delta_cy: f32,
        tolerance: f32,
    },
}

impl std::fmt::Display for RectifiedStereoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RectifiedStereoError::NonPositiveBaseline { baseline_m } => {
                write!(f, "baseline must be > 0, got {baseline_m}")
            }
            RectifiedStereoError::DimensionMismatch { left, right } => {
                write!(
                    f,
                    "rectified stereo requires same dimensions: left={}x{}, right={}x{}",
                    left.0, left.1, right.0, right.1
                )
            }
            RectifiedStereoError::InvalidFocal { fx, fy } => {
                write!(f, "rectified stereo requires positive focal lengths: fx={fx}, fy={fy}")
            }
            RectifiedStereoError::NotRectified => {
                write!(f, "calibration is not marked rectified")
            }
            RectifiedStereoError::PrincipalPointMismatch {
                delta_cx,
                delta_cy,
                tolerance,
            } => {
                write!(
                    f,
                    "principal points differ too much: delta_cx={delta_cx}, delta_cy={delta_cy}, tolerance={tolerance}"
                )
            }
        }
    }
}

impl std::error::Error for RectifiedStereoError {}

impl RectifiedStereo {
    pub fn from_calibration(
        calibration: &Calibration,
    ) -> Result<Self, RectifiedStereoError> {
        Self::from_calibration_with_config(calibration, RectifiedStereoConfig::default())
    }

    pub fn from_calibration_with_config(
        calibration: &Calibration,
        config: RectifiedStereoConfig,
    ) -> Result<Self, RectifiedStereoError> {
        let left = calibration.left.clone();
        let right = calibration.right.clone();

        if calibration.baseline_m <= 0.0 {
            return Err(RectifiedStereoError::NonPositiveBaseline {
                baseline_m: calibration.baseline_m,
            });
        }

        if left.width != right.width || left.height != right.height {
            return Err(RectifiedStereoError::DimensionMismatch {
                left: (left.width, left.height),
                right: (right.width, right.height),
            });
        }

        if left.fx <= 0.0 || left.fy <= 0.0 {
            return Err(RectifiedStereoError::InvalidFocal {
                fx: left.fx,
                fy: left.fy,
            });
        }

        if !calibration.rectified && !config.allow_unrectified {
            return Err(RectifiedStereoError::NotRectified);
        }

        if let Some(tolerance) = config.max_principal_delta_px {
            let delta_cx = (left.cx - right.cx).abs();
            let delta_cy = (left.cy - right.cy).abs();
            if delta_cx > tolerance || delta_cy > tolerance {
                return Err(RectifiedStereoError::PrincipalPointMismatch {
                    delta_cx,
                    delta_cy,
                    tolerance,
                });
            }
        }

        Ok(Self {
            left,
            right,
            baseline_m: calibration.baseline_m,
        })
    }

    pub fn left(&self) -> &CameraIntrinsics {
        &self.left
    }

    pub fn right(&self) -> &CameraIntrinsics {
        &self.right
    }

    pub fn baseline_m(&self) -> f32 {
        self.baseline_m
    }

    pub fn width(&self) -> u32 {
        self.left.width
    }

    pub fn height(&self) -> u32 {
        self.left.height
    }

    pub fn fx(&self) -> f32 {
        self.left.fx
    }

    pub fn fy(&self) -> f32 {
        self.left.fy
    }

    pub fn cx(&self) -> f32 {
        self.left.cx
    }

    pub fn cy(&self) -> f32 {
        self.left.cy
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TriangulationConfig {
    pub min_disparity_px: f32,
    pub max_depth_m: Option<f32>,
}

impl Default for TriangulationConfig {
    fn default() -> Self {
        Self {
            min_disparity_px: 1.0,
            max_depth_m: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TriangulationStats {
    pub candidate_matches: usize,
    pub kept: usize,
    pub dropped_disparity: usize,
    pub dropped_out_of_bounds: usize,
    pub dropped_depth: usize,
    pub dropped_duplicate: usize,
}

#[derive(Debug)]
pub enum TriangulationError {
    SensorMismatch {
        left: SensorId,
        right: SensorId,
    },
    IndexOutOfBounds {
        left_len: usize,
        right_len: usize,
        left_index: usize,
        right_index: usize,
    },
    NoLandmarks {
        stats: TriangulationStats,
    },
    InvalidDetections {
        message: String,
    },
}

impl std::fmt::Display for TriangulationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TriangulationError::SensorMismatch { left, right } => {
                write!(
                    f,
                    "triangulation requires stereo left/right detections, got left={left:?}, right={right:?}"
                )
            }
            TriangulationError::IndexOutOfBounds {
                left_len,
                right_len,
                left_index,
                right_index,
            } => {
                write!(
                    f,
                    "match index out of bounds: left_index={left_index} (len={left_len}), right_index={right_index} (len={right_len})"
                )
            }
            TriangulationError::NoLandmarks { stats } => {
                write!(
                    f,
                    "triangulation produced no landmarks (candidates={}, dropped_disparity={}, dropped_out_of_bounds={}, dropped_depth={}, dropped_duplicate={})",
                    stats.candidate_matches,
                    stats.dropped_disparity,
                    stats.dropped_out_of_bounds,
                    stats.dropped_depth,
                    stats.dropped_duplicate
                )
            }
            TriangulationError::InvalidDetections { message } => {
                write!(f, "failed to build detections: {message}")
            }
        }
    }
}

impl std::error::Error for TriangulationError {}

#[derive(Clone, Copy, Debug)]
pub struct Point3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Debug)]
pub struct Keyframe {
    frame_id: FrameId,
    detections: Arc<Detections>,
    landmarks: Vec<Point3>,
    landmark_indices: Vec<usize>,
    index_to_landmark: Vec<Option<usize>>,
}

#[derive(Debug)]
pub enum KeyframeError {
    Empty,
    LenMismatch {
        detections: usize,
        landmarks: usize,
    },
    LandmarkIndexOutOfBounds {
        detections: usize,
        index: usize,
    },
    DuplicateLandmarkIndex {
        index: usize,
    },
    SensorMismatch {
        expected: SensorId,
        actual: SensorId,
    },
}

impl std::fmt::Display for KeyframeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KeyframeError::Empty => write!(f, "keyframe must contain at least one landmark"),
            KeyframeError::LenMismatch {
                detections,
                landmarks,
            } => write!(
                f,
                "keyframe landmarks/indices length mismatch: detections={detections}, landmarks={landmarks}"
            ),
            KeyframeError::LandmarkIndexOutOfBounds { detections, index } => write!(
                f,
                "keyframe landmark index out of bounds: index={index} (detections={detections})"
            ),
            KeyframeError::DuplicateLandmarkIndex { index } => {
                write!(f, "keyframe landmark index used twice: index={index}")
            }
            KeyframeError::SensorMismatch { expected, actual } => {
                write!(
                    f,
                    "keyframe detections must be from {expected:?}, got {actual:?}"
                )
            }
        }
    }
}

impl std::error::Error for KeyframeError {}

impl Keyframe {
    pub fn new(
        detections: Detections,
        landmarks: Vec<Point3>,
        landmark_indices: Vec<usize>,
    ) -> Result<Self, KeyframeError> {
        Self::from_arc(Arc::new(detections), landmarks, landmark_indices)
    }

    pub fn from_arc(
        detections: Arc<Detections>,
        landmarks: Vec<Point3>,
        landmark_indices: Vec<usize>,
    ) -> Result<Self, KeyframeError> {
        if detections.is_empty() || landmarks.is_empty() || landmark_indices.is_empty() {
            return Err(KeyframeError::Empty);
        }
        if landmarks.len() != landmark_indices.len() {
            return Err(KeyframeError::LenMismatch {
                detections: detections.len(),
                landmarks: landmarks.len(),
            });
        }
        if detections.sensor_id() != SensorId::StereoLeft {
            return Err(KeyframeError::SensorMismatch {
                expected: SensorId::StereoLeft,
                actual: detections.sensor_id(),
            });
        }

        let mut index_to_landmark = vec![None; detections.len()];
        for (landmark_idx, &det_idx) in landmark_indices.iter().enumerate() {
            if det_idx >= detections.len() {
                return Err(KeyframeError::LandmarkIndexOutOfBounds {
                    detections: detections.len(),
                    index: det_idx,
                });
            }
            if index_to_landmark[det_idx].is_some() {
                return Err(KeyframeError::DuplicateLandmarkIndex { index: det_idx });
            }
            index_to_landmark[det_idx] = Some(landmark_idx);
        }

        Ok(Self {
            frame_id: detections.frame_id(),
            detections,
            landmarks,
            landmark_indices,
            index_to_landmark,
        })
    }

    pub fn frame_id(&self) -> FrameId {
        self.frame_id
    }

    pub fn detections(&self) -> &Arc<Detections> {
        &self.detections
    }

    pub fn landmarks(&self) -> &[Point3] {
        &self.landmarks
    }

    pub fn landmark_indices(&self) -> &[usize] {
        &self.landmark_indices
    }

    pub fn landmark_for_detection(&self, index: usize) -> Option<Point3> {
        let landmark_idx = *self.index_to_landmark.get(index)?;
        landmark_idx.map(|idx| self.landmarks[idx])
    }
}

#[derive(Debug)]
pub struct TriangulationResult {
    pub keyframe: Keyframe,
    pub stats: TriangulationStats,
}

#[derive(Debug)]
pub struct Triangulator {
    stereo: RectifiedStereo,
    config: TriangulationConfig,
}

impl Triangulator {
    pub fn new(stereo: RectifiedStereo, config: TriangulationConfig) -> Self {
        Self { stereo, config }
    }

    pub fn triangulate(
        &self,
        matches: &Matches<Raw>,
    ) -> Result<TriangulationResult, TriangulationError> {
        let left = matches.source_a_arc();
        let right = matches.source_b();

        if left.sensor_id() != SensorId::StereoLeft || right.sensor_id() != SensorId::StereoRight {
            return Err(TriangulationError::SensorMismatch {
                left: left.sensor_id(),
                right: right.sensor_id(),
            });
        }

        let left_len = left.len();
        let right_len = right.len();
        let mut stats = TriangulationStats::default();
        stats.candidate_matches = matches.len();

        let mut best: Vec<Option<(usize, f32)>> = vec![None; left_len];
        for (&(li, ri), &score) in matches.indices().iter().zip(matches.scores()) {
            if li >= left_len || ri >= right_len {
                return Err(TriangulationError::IndexOutOfBounds {
                    left_len,
                    right_len,
                    left_index: li,
                    right_index: ri,
                });
            }
            match best[li] {
                Some((_, best_score)) if best_score >= score => {
                    stats.dropped_duplicate += 1;
                }
                Some(_) => {
                    stats.dropped_duplicate += 1;
                    best[li] = Some((ri, score));
                }
                None => {
                    best[li] = Some((ri, score));
                }
            }
        }

        let width = self.stereo.width() as f32;
        let height = self.stereo.height() as f32;
        let fx = self.stereo.fx();
        let fy = self.stereo.fy();
        let cx = self.stereo.cx();
        let cy = self.stereo.cy();
        let baseline = self.stereo.baseline_m();

        let mut landmarks = Vec::new();
        let mut landmark_indices = Vec::new();

        for (li, candidate) in best.into_iter().enumerate() {
            let Some((ri, _match_score)) = candidate else {
                continue;
            };

        let left_kp = left.keypoints()[li];
            let right_kp = right.keypoints()[ri];

            if !in_bounds(left_kp, width, height) || !in_bounds(right_kp, width, height) {
                stats.dropped_out_of_bounds += 1;
                continue;
            }

            let disparity = left_kp.x - right_kp.x;
            if disparity <= self.config.min_disparity_px {
                stats.dropped_disparity += 1;
                continue;
            }

            let z = fx * baseline / disparity;
            if let Some(max_depth) = self.config.max_depth_m {
                if z > max_depth {
                    stats.dropped_depth += 1;
                    continue;
                }
            }

            let x = (left_kp.x - cx) * z / fx;
            let y = (left_kp.y - cy) * z / fy;

            landmarks.push(Point3 { x, y, z });
            landmark_indices.push(li);
            stats.kept += 1;
        }

        if landmarks.is_empty() {
            return Err(TriangulationError::NoLandmarks { stats });
        }

        let keyframe = Keyframe::from_arc(left, landmarks, landmark_indices)
            .map_err(|err| TriangulationError::InvalidDetections {
                message: err.to_string(),
            })?;

        Ok(TriangulationResult { keyframe, stats })
    }
}

fn in_bounds(kp: Keypoint, width: f32, height: f32) -> bool {
    kp.x >= 0.0 && kp.y >= 0.0 && kp.x < width && kp.y < height
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{
        make_detections, make_pinhole_intrinsics, make_rectified_stereo,
        rectified_stereo_keypoints_from_points,
    };
    use crate::{FrameId, Matches};

    fn assert_stats_accounting(stats: TriangulationStats) {
        assert_eq!(
            stats.kept
                + stats.dropped_disparity
                + stats.dropped_out_of_bounds
                + stats.dropped_depth
                + stats.dropped_duplicate,
            stats.candidate_matches
        );
    }

    #[test]
    fn triangulate_recovers_known_depth_for_rectified_pairs() {
        let intrinsics = make_pinhole_intrinsics(640, 480, 400.0, 402.0, 320.0, 240.0)
            .expect("intrinsics");
        let stereo = make_rectified_stereo(640, 480, 400.0, 402.0, 320.0, 240.0, 0.075)
            .expect("stereo");
        let triangulator = Triangulator::new(stereo, TriangulationConfig::default());

        let points = vec![
            Point3 {
                x: -0.2,
                y: -0.1,
                z: 2.5,
            },
            Point3 {
                x: 0.1,
                y: 0.15,
                z: 3.2,
            },
            Point3 {
                x: 0.3,
                y: -0.05,
                z: 4.1,
            },
            Point3 {
                x: -0.35,
                y: 0.2,
                z: 5.4,
            },
        ];

        let kps = rectified_stereo_keypoints_from_points(&points, intrinsics, 0.075);
        let mut left_kps = Vec::new();
        let mut right_kps = Vec::new();
        let mut pairs = Vec::new();
        for (idx, (src_idx, left, right)) in kps.into_iter().enumerate() {
            left_kps.push(left);
            right_kps.push(right);
            pairs.push((idx, idx));
            assert_eq!(src_idx, idx);
        }

        let left = make_detections(SensorId::StereoLeft, FrameId::new(10), 640, 480, left_kps)
            .expect("left detections");
        let right =
            make_detections(SensorId::StereoRight, FrameId::new(11), 640, 480, right_kps)
                .expect("right detections");
        let matches = Matches::new(left, right, pairs, vec![1.0; points.len()]).expect("matches");

        let result = triangulator.triangulate(&matches).expect("triangulation");
        assert_stats_accounting(result.stats);

        let keyframe = result.keyframe;
        assert_eq!(keyframe.landmarks().len(), points.len());
        for (landmark, &det_idx) in keyframe.landmarks().iter().zip(keyframe.landmark_indices()) {
            let expected = points[det_idx];
            assert!((landmark.x - expected.x).abs() < 1e-4);
            assert!((landmark.y - expected.y).abs() < 1e-4);
            assert!((landmark.z - expected.z).abs() < 1e-4);
        }
    }

    #[test]
    fn triangulate_rejects_points_below_min_disparity() {
        let intrinsics = make_pinhole_intrinsics(640, 480, 400.0, 400.0, 320.0, 240.0)
            .expect("intrinsics");
        let stereo = make_rectified_stereo(640, 480, 400.0, 400.0, 320.0, 240.0, 0.075)
            .expect("stereo");
        let triangulator = Triangulator::new(
            stereo,
            TriangulationConfig {
                min_disparity_px: 1.0,
                max_depth_m: None,
            },
        );

        let far_points = vec![Point3 {
            x: 0.0,
            y: 0.0,
            z: 90.0,
        }];
        let kps = rectified_stereo_keypoints_from_points(&far_points, intrinsics, 0.075);
        let (_, left_kp, right_kp) = kps[0];

        let left = make_detections(
            SensorId::StereoLeft,
            FrameId::new(20),
            640,
            480,
            vec![left_kp],
        )
        .expect("left");
        let right = make_detections(
            SensorId::StereoRight,
            FrameId::new(21),
            640,
            480,
            vec![right_kp],
        )
        .expect("right");
        let matches = Matches::new(left, right, vec![(0, 0)], vec![1.0]).expect("matches");

        let err = triangulator
            .triangulate(&matches)
            .expect_err("should reject low disparity");
        match err {
            TriangulationError::NoLandmarks { stats } => {
                assert_eq!(stats.candidate_matches, 1);
                assert_eq!(stats.kept, 0);
                assert_eq!(stats.dropped_disparity, 1);
                assert_stats_accounting(stats);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn triangulate_returns_index_out_of_bounds_for_bad_match_indices() {
        let stereo = make_rectified_stereo(640, 480, 400.0, 400.0, 320.0, 240.0, 0.075)
            .expect("stereo");
        let triangulator = Triangulator::new(stereo, TriangulationConfig::default());

        let left = make_detections(
            SensorId::StereoLeft,
            FrameId::new(30),
            640,
            480,
            vec![Keypoint { x: 320.0, y: 240.0 }],
        )
        .expect("left");
        let right = make_detections(
            SensorId::StereoRight,
            FrameId::new(31),
            640,
            480,
            vec![Keypoint { x: 300.0, y: 240.0 }],
        )
        .expect("right");
        let matches = Matches::new(left, right, vec![(0, 2)], vec![1.0]).expect("matches");

        let err = triangulator
            .triangulate(&matches)
            .expect_err("index error expected");
        match err {
            TriangulationError::IndexOutOfBounds {
                left_len,
                right_len,
                left_index,
                right_index,
            } => {
                assert_eq!(left_len, 1);
                assert_eq!(right_len, 1);
                assert_eq!(left_index, 0);
                assert_eq!(right_index, 2);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn triangulate_uses_best_score_for_duplicate_left_matches() {
        let stereo = make_rectified_stereo(640, 480, 400.0, 400.0, 320.0, 240.0, 0.075)
            .expect("stereo");
        let triangulator = Triangulator::new(stereo, TriangulationConfig::default());

        let left = make_detections(
            SensorId::StereoLeft,
            FrameId::new(40),
            640,
            480,
            vec![Keypoint { x: 360.0, y: 240.0 }],
        )
        .expect("left");
        let right = make_detections(
            SensorId::StereoRight,
            FrameId::new(41),
            640,
            480,
            vec![Keypoint { x: 335.0, y: 240.0 }, Keypoint { x: 345.0, y: 240.0 }],
        )
        .expect("right");

        let matches = Matches::new(
            left,
            right,
            vec![(0, 0), (0, 1)],
            vec![0.1, 0.9], // winner should be (0,1): smaller disparity => larger depth
        )
        .expect("matches");

        let result = triangulator.triangulate(&matches).expect("triangulation");
        assert_eq!(result.stats.candidate_matches, 2);
        assert_eq!(result.stats.dropped_duplicate, 1);
        assert_eq!(result.stats.kept, 1);
        assert_stats_accounting(result.stats);

        let z = result.keyframe.landmarks()[0].z;
        let expected_disparity = 360.0 - 345.0;
        let expected_z = 400.0 * 0.075 / expected_disparity;
        assert!((z - expected_z).abs() < 1e-4);
    }

    #[test]
    fn triangulate_rejects_sensor_mismatch() {
        let stereo = make_rectified_stereo(640, 480, 400.0, 400.0, 320.0, 240.0, 0.075)
            .expect("stereo");
        let triangulator = Triangulator::new(stereo, TriangulationConfig::default());

        let left = make_detections(
            SensorId::StereoLeft,
            FrameId::new(50),
            640,
            480,
            vec![Keypoint { x: 360.0, y: 240.0 }],
        )
        .expect("left");
        let right_wrong = make_detections(
            SensorId::StereoLeft,
            FrameId::new(51),
            640,
            480,
            vec![Keypoint { x: 330.0, y: 240.0 }],
        )
        .expect("right_wrong");
        let matches = Matches::new(left, right_wrong, vec![(0, 0)], vec![1.0]).expect("matches");

        let err = triangulator
            .triangulate(&matches)
            .expect_err("sensor mismatch expected");
        match err {
            TriangulationError::SensorMismatch { left, right } => {
                assert_eq!(left, SensorId::StereoLeft);
                assert_eq!(right, SensorId::StereoLeft);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
