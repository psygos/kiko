use std::sync::Arc;

use crate::dataset::{Calibration, CameraIntrinsics};
use crate::{Detections, FrameDimensions, FrameId, Keypoint, Matches, Raw, SensorId};

#[derive(Clone, Copy, Debug)]
pub struct RectifiedStereoConfig {
    max_principal_delta_px: f32,
    max_focal_delta_px: f32,
}

impl Default for RectifiedStereoConfig {
    fn default() -> Self {
        Self {
            max_principal_delta_px: 1e-3,
            max_focal_delta_px: 1e-3,
        }
    }
}

impl RectifiedStereoConfig {
    pub fn try_new(
        max_principal_delta_px: f32,
        max_focal_delta_px: f32,
    ) -> Result<Self, RectifiedStereoError> {
        for (name, value) in [
            ("principal-point", max_principal_delta_px),
            ("focal-length", max_focal_delta_px),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(RectifiedStereoError::InvalidTolerance { name, value });
            }
        }
        Ok(Self {
            max_principal_delta_px,
            max_focal_delta_px,
        })
    }

    pub fn max_principal_delta_px(self) -> f32 {
        self.max_principal_delta_px
    }

    pub fn max_focal_delta_px(self) -> f32 {
        self.max_focal_delta_px
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
    NonPositiveBaseline {
        baseline_m: f32,
    },
    ZeroDimensions {
        camera: &'static str,
        width: u32,
        height: u32,
    },
    DimensionMismatch {
        left: FrameDimensions,
        right: FrameDimensions,
    },
    InvalidFocal {
        fx: f32,
        fy: f32,
    },
    NonFiniteIntrinsics {
        camera: &'static str,
    },
    NotRectified,
    InvalidTolerance {
        name: &'static str,
        value: f32,
    },
    FocalMismatch {
        delta_fx: f32,
        delta_fy: f32,
        tolerance: f32,
    },
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
                write!(f, "baseline must be positive and finite, got {baseline_m}")
            }
            RectifiedStereoError::ZeroDimensions {
                camera,
                width,
                height,
            } => {
                write!(
                    f,
                    "{camera} camera dimensions must be nonzero, got {width}x{height}"
                )
            }
            RectifiedStereoError::DimensionMismatch { left, right } => {
                write!(
                    f,
                    "rectified stereo requires same dimensions: left={}x{}, right={}x{}",
                    left.width(),
                    left.height(),
                    right.width(),
                    right.height()
                )
            }
            RectifiedStereoError::InvalidFocal { fx, fy } => {
                write!(
                    f,
                    "rectified stereo requires positive focal lengths: fx={fx}, fy={fy}"
                )
            }
            RectifiedStereoError::NonFiniteIntrinsics { camera } => {
                write!(f, "{camera} camera intrinsics must be finite")
            }
            RectifiedStereoError::NotRectified => {
                write!(f, "calibration is not marked rectified")
            }
            RectifiedStereoError::InvalidTolerance { name, value } => {
                write!(
                    f,
                    "{name} tolerance must be finite and nonnegative, got {value}"
                )
            }
            RectifiedStereoError::FocalMismatch {
                delta_fx,
                delta_fy,
                tolerance,
            } => write!(
                f,
                "rectified focal lengths differ: delta_fx={delta_fx}, delta_fy={delta_fy}, tolerance={tolerance}"
            ),
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
    pub fn from_calibration(calibration: &Calibration) -> Result<Self, RectifiedStereoError> {
        Self::from_calibration_with_config(calibration, RectifiedStereoConfig::default())
    }

    pub fn from_calibration_with_config(
        calibration: &Calibration,
        config: RectifiedStereoConfig,
    ) -> Result<Self, RectifiedStereoError> {
        let left = calibration.left.clone();
        let right = calibration.right.clone();

        if !calibration.baseline_m.is_finite() || calibration.baseline_m <= 0.0 {
            return Err(RectifiedStereoError::NonPositiveBaseline {
                baseline_m: calibration.baseline_m,
            });
        }

        for (camera, intrinsics) in [("left", &left), ("right", &right)] {
            if intrinsics.width == 0 || intrinsics.height == 0 {
                return Err(RectifiedStereoError::ZeroDimensions {
                    camera,
                    width: intrinsics.width,
                    height: intrinsics.height,
                });
            }
            if !intrinsics.fx.is_finite()
                || !intrinsics.fy.is_finite()
                || !intrinsics.cx.is_finite()
                || !intrinsics.cy.is_finite()
            {
                return Err(RectifiedStereoError::NonFiniteIntrinsics { camera });
            }
            if intrinsics.fx <= 0.0 || intrinsics.fy <= 0.0 {
                return Err(RectifiedStereoError::InvalidFocal {
                    fx: intrinsics.fx,
                    fy: intrinsics.fy,
                });
            }
        }

        if left.width != right.width || left.height != right.height {
            return Err(RectifiedStereoError::DimensionMismatch {
                left: FrameDimensions::new(left.width, left.height),
                right: FrameDimensions::new(right.width, right.height),
            });
        }

        if !calibration.rectified {
            return Err(RectifiedStereoError::NotRectified);
        }

        let delta_fx = (left.fx - right.fx).abs();
        let delta_fy = (left.fy - right.fy).abs();
        if delta_fx > config.max_focal_delta_px || delta_fy > config.max_focal_delta_px {
            return Err(RectifiedStereoError::FocalMismatch {
                delta_fx,
                delta_fy,
                tolerance: config.max_focal_delta_px,
            });
        }

        let delta_cx = (left.cx - right.cx).abs();
        let delta_cy = (left.cy - right.cy).abs();
        if delta_cx > config.max_principal_delta_px || delta_cy > config.max_principal_delta_px {
            return Err(RectifiedStereoError::PrincipalPointMismatch {
                delta_cx,
                delta_cy,
                tolerance: config.max_principal_delta_px,
            });
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
    min_disparity_px: f32,
    max_depth_m: Option<f32>,
}

impl Default for TriangulationConfig {
    fn default() -> Self {
        Self {
            min_disparity_px: 1.0,
            max_depth_m: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TriangulationConfigError {
    InvalidMinDisparity { value: f32 },
    InvalidMaxDepth { value: f32 },
}

impl std::fmt::Display for TriangulationConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMinDisparity { value } => {
                write!(
                    f,
                    "minimum disparity must be positive and finite, got {value}"
                )
            }
            Self::InvalidMaxDepth { value } => {
                write!(f, "maximum depth must be positive and finite, got {value}")
            }
        }
    }
}

impl std::error::Error for TriangulationConfigError {}

impl TriangulationConfig {
    pub fn try_new(
        min_disparity_px: f32,
        max_depth_m: Option<f32>,
    ) -> Result<Self, TriangulationConfigError> {
        if !min_disparity_px.is_finite() || min_disparity_px <= 0.0 {
            return Err(TriangulationConfigError::InvalidMinDisparity {
                value: min_disparity_px,
            });
        }
        if let Some(max_depth_m) = max_depth_m
            && (!max_depth_m.is_finite() || max_depth_m <= 0.0)
        {
            return Err(TriangulationConfigError::InvalidMaxDepth { value: max_depth_m });
        }
        Ok(Self {
            min_disparity_px,
            max_depth_m,
        })
    }

    pub fn min_disparity_px(self) -> f32 {
        self.min_disparity_px
    }

    pub fn max_depth_m(self) -> Option<f32> {
        self.max_depth_m
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TriangulationStats {
    pub candidate_matches: usize,
    pub kept: usize,
    pub dropped_disparity: usize,
    pub dropped_depth: usize,
}

#[derive(Debug)]
pub enum TriangulationError {
    SensorMismatch {
        left: SensorId,
        right: SensorId,
    },
    DetectionDimensionMismatch {
        expected: FrameDimensions,
        left: FrameDimensions,
        right: FrameDimensions,
    },
    NoLandmarks {
        stats: TriangulationStats,
    },
    InvalidKeyframe(KeyframeError),
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
            TriangulationError::DetectionDimensionMismatch {
                expected,
                left,
                right,
            } => write!(
                f,
                "detection dimensions do not match calibration: expected={}x{}, left={}x{}, right={}x{}",
                expected.width(),
                expected.height(),
                left.width(),
                left.height(),
                right.width(),
                right.height()
            ),
            TriangulationError::NoLandmarks { stats } => {
                write!(
                    f,
                    "triangulation produced no landmarks (candidates={}, dropped_disparity={}, dropped_depth={})",
                    stats.candidate_matches, stats.dropped_disparity, stats.dropped_depth
                )
            }
            TriangulationError::InvalidKeyframe(err) => {
                write!(f, "failed to build triangulated keyframe: {err}")
            }
        }
    }
}

impl std::error::Error for TriangulationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidKeyframe(err) => Some(err),
            Self::SensorMismatch { .. }
            | Self::DetectionDimensionMismatch { .. }
            | Self::NoLandmarks { .. } => None,
        }
    }
}

#[derive(Debug)]
pub struct Keyframe {
    frame_id: FrameId,
    detections: Arc<Detections>,
    landmarks: Vec<crate::CameraPoint3>,
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
        landmarks: Vec<crate::CameraPoint3>,
        landmark_indices: Vec<usize>,
    ) -> Result<Self, KeyframeError> {
        Self::from_arc(Arc::new(detections), landmarks, landmark_indices)
    }

    pub fn from_arc(
        detections: Arc<Detections>,
        landmarks: Vec<crate::CameraPoint3>,
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

    pub fn landmarks(&self) -> &[crate::CameraPoint3] {
        &self.landmarks
    }

    pub fn landmark_indices(&self) -> &[usize] {
        &self.landmark_indices
    }

    pub fn landmark_for_detection(&self, index: usize) -> Option<crate::CameraPoint3> {
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

        let expected = FrameDimensions::new(self.stereo.width(), self.stereo.height());
        let left_dimensions = FrameDimensions::new(left.width(), left.height());
        let right_dimensions = FrameDimensions::new(right.width(), right.height());
        if left_dimensions != expected || right_dimensions != expected {
            return Err(TriangulationError::DetectionDimensionMismatch {
                expected,
                left: left_dimensions,
                right: right_dimensions,
            });
        }

        let mut stats = TriangulationStats {
            candidate_matches: matches.len(),
            ..TriangulationStats::default()
        };

        let width = self.stereo.width() as f32;
        let height = self.stereo.height() as f32;
        let fx = self.stereo.fx();
        let fy = self.stereo.fy();
        let cx = self.stereo.cx();
        let cy = self.stereo.cy();
        let baseline = self.stereo.baseline_m();

        let mut landmarks = Vec::new();
        let mut landmark_indices = Vec::new();

        for &(li, ri) in matches.indices() {
            let left_kp = left.keypoints()[li];
            let right_kp = right.keypoints()[ri];
            debug_assert!(in_bounds(left_kp, width, height));
            debug_assert!(in_bounds(right_kp, width, height));

            let disparity = left_kp.x - right_kp.x;
            if disparity <= self.config.min_disparity_px {
                stats.dropped_disparity += 1;
                continue;
            }

            let z = fx * baseline / disparity;
            if let Some(max_depth) = self.config.max_depth_m
                && z > max_depth
            {
                stats.dropped_depth += 1;
                continue;
            }

            let x = (left_kp.x - cx) * z / fx;
            let y = (left_kp.y - cy) * z / fy;

            landmarks.push(crate::CameraPoint3 { x, y, z });
            landmark_indices.push(li);
            stats.kept += 1;
        }

        if landmarks.is_empty() {
            return Err(TriangulationError::NoLandmarks { stats });
        }

        let keyframe = Keyframe::from_arc(left, landmarks, landmark_indices)
            .map_err(TriangulationError::InvalidKeyframe)?;

        Ok(TriangulationResult { keyframe, stats })
    }
}

fn in_bounds(kp: Keypoint, width: f32, height: f32) -> bool {
    kp.x >= 0.0 && kp.y >= 0.0 && kp.x < width && kp.y < height
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CameraPoint3;
    use crate::test_helpers::{
        make_camera_intrinsics, make_detections, make_pinhole_intrinsics, make_rectified_stereo,
        rectified_stereo_keypoints_from_points,
    };
    use crate::{FrameId, MatchError, Matches};

    fn assert_stats_accounting(stats: TriangulationStats) {
        assert_eq!(
            stats.kept + stats.dropped_disparity + stats.dropped_depth,
            stats.candidate_matches
        );
    }

    #[test]
    fn rectified_stereo_rejects_unrectified_metadata() {
        let intrinsics = make_camera_intrinsics(640, 480, 400.0, 400.0, 320.0, 240.0);
        let calibration = Calibration {
            left: intrinsics.clone(),
            right: intrinsics,
            baseline_m: 0.075,
            rectified: false,
        };

        assert!(matches!(
            RectifiedStereo::from_calibration(&calibration),
            Err(RectifiedStereoError::NotRectified)
        ));
    }

    #[test]
    fn rectified_stereo_rejects_nonfinite_and_incompatible_intrinsics() {
        let left = make_camera_intrinsics(640, 480, 400.0, 400.0, 320.0, 240.0);
        let mut calibration = Calibration {
            left: left.clone(),
            right: left,
            baseline_m: f32::NAN,
            rectified: true,
        };
        assert!(matches!(
            RectifiedStereo::from_calibration(&calibration),
            Err(RectifiedStereoError::NonPositiveBaseline { .. })
        ));

        calibration.baseline_m = 0.075;
        calibration.right.fx = 401.0;
        assert!(matches!(
            RectifiedStereo::from_calibration(&calibration),
            Err(RectifiedStereoError::FocalMismatch { .. })
        ));

        calibration.right.fx = calibration.left.fx;
        calibration.right.cx = f32::NAN;
        assert!(matches!(
            RectifiedStereo::from_calibration(&calibration),
            Err(RectifiedStereoError::NonFiniteIntrinsics { camera: "right" })
        ));
    }

    #[test]
    fn stereo_and_triangulation_configs_reject_invalid_scalars() {
        for value in [-1.0, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                RectifiedStereoConfig::try_new(value, 1e-3),
                Err(RectifiedStereoError::InvalidTolerance { .. })
            ));
        }
        for value in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                TriangulationConfig::try_new(value, None),
                Err(TriangulationConfigError::InvalidMinDisparity { .. })
            ));
            assert!(matches!(
                TriangulationConfig::try_new(1.0, Some(value)),
                Err(TriangulationConfigError::InvalidMaxDepth { .. })
            ));
        }
    }

    #[test]
    fn triangulate_rejects_detection_dimensions_that_do_not_match_calibration() {
        let stereo =
            make_rectified_stereo(640, 480, 400.0, 400.0, 320.0, 240.0, 0.075).expect("stereo");
        let triangulator = Triangulator::new(stereo, TriangulationConfig::default());
        let left = make_detections(
            SensorId::StereoLeft,
            FrameId::new(1),
            320,
            240,
            vec![Keypoint { x: 100.0, y: 100.0 }],
        )
        .expect("left");
        let right = make_detections(
            SensorId::StereoRight,
            FrameId::new(2),
            640,
            480,
            vec![Keypoint { x: 90.0, y: 100.0 }],
        )
        .expect("right");
        let matches = Matches::new(left, right, vec![(0, 0)], vec![1.0]).expect("matches");

        assert!(matches!(
            triangulator.triangulate(&matches),
            Err(TriangulationError::DetectionDimensionMismatch { .. })
        ));
    }

    #[test]
    fn triangulate_recovers_known_depth_for_rectified_pairs() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 400.0, 402.0, 320.0, 240.0).expect("intrinsics");
        let stereo =
            make_rectified_stereo(640, 480, 400.0, 402.0, 320.0, 240.0, 0.075).expect("stereo");
        let triangulator = Triangulator::new(stereo, TriangulationConfig::default());

        let points = vec![
            CameraPoint3 {
                x: -0.2,
                y: -0.1,
                z: 2.5,
            },
            CameraPoint3 {
                x: 0.1,
                y: 0.15,
                z: 3.2,
            },
            CameraPoint3 {
                x: 0.3,
                y: -0.05,
                z: 4.1,
            },
            CameraPoint3 {
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
        let right = make_detections(SensorId::StereoRight, FrameId::new(11), 640, 480, right_kps)
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
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 400.0, 400.0, 320.0, 240.0).expect("intrinsics");
        let stereo =
            make_rectified_stereo(640, 480, 400.0, 400.0, 320.0, 240.0, 0.075).expect("stereo");
        let triangulator = Triangulator::new(
            stereo,
            TriangulationConfig::try_new(1.0, None).expect("triangulation config"),
        );

        let far_points = vec![CameraPoint3 {
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
    fn matches_reject_out_of_bounds_indices_before_triangulation() {
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
        let err = Matches::new(left, right, vec![(0, 2)], vec![1.0])
            .expect_err("match construction must reject invalid indices");
        assert!(matches!(
            err,
            MatchError::IndexOutOfBounds {
                source_a_len: 1,
                source_b_len: 1,
                source_a_index: 0,
                source_b_index: 2,
                ..
            }
        ));
    }

    #[test]
    fn matches_reject_duplicate_left_indices_before_triangulation() {
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
            vec![
                Keypoint { x: 335.0, y: 240.0 },
                Keypoint { x: 345.0, y: 240.0 },
            ],
        )
        .expect("right");

        let err = Matches::new(left, right, vec![(0, 0), (0, 1)], vec![0.1, 0.9])
            .expect_err("match construction must reject reused detections");
        assert!(matches!(
            err,
            MatchError::DuplicateIndex {
                source: "a",
                index: 0
            }
        ));
    }

    #[test]
    fn triangulate_rejects_sensor_mismatch() {
        let stereo =
            make_rectified_stereo(640, 480, 400.0, 400.0, 320.0, 240.0, 0.075).expect("stereo");
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
