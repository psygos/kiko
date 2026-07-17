use std::sync::Arc;

use crate::dataset::{Calibration, CameraIntrinsics};
use crate::{
    DetectionError, Detections, FrameDimensions, FrameDimensionsError, FrameId, Keypoint, Matches,
    Raw, SensorId,
};

#[derive(Clone, Debug)]
pub struct RectifiedStereo {
    left: CameraIntrinsics,
    right: CameraIntrinsics,
    dimensions: FrameDimensions,
    baseline_m: f32,
    arithmetic: RectifiedStereoArithmetic,
}

/// Exact f32-to-f64 widening of the validated stereo calibration, parsed once.
#[derive(Clone, Debug)]
struct RectifiedStereoArithmetic {
    left: CameraIntrinsics64,
    right: CameraIntrinsics64,
    baseline_m: f64,
}

#[derive(Clone, Copy, Debug)]
struct CameraIntrinsics64 {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
}

impl CameraIntrinsics64 {
    fn from_finite(intrinsics: &CameraIntrinsics) -> Self {
        Self {
            fx: f64::from(intrinsics.fx),
            fy: f64::from(intrinsics.fy),
            cx: f64::from(intrinsics.cx),
            cy: f64::from(intrinsics.cy),
        }
    }
}

impl RectifiedStereoArithmetic {
    fn new(left: &CameraIntrinsics, right: &CameraIntrinsics, baseline_m: f32) -> Self {
        Self {
            left: CameraIntrinsics64::from_finite(left),
            right: CameraIntrinsics64::from_finite(right),
            baseline_m: f64::from(baseline_m),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum RectifiedStereoError {
    InvalidLeftDimensions(FrameDimensionsError),
    InvalidRightDimensions(FrameDimensionsError),
    InvalidBaseline {
        baseline_m: f32,
    },
    DimensionMismatch {
        left: FrameDimensions,
        right: FrameDimensions,
    },
    InvalidFocal {
        camera: &'static str,
        fx: f32,
        fy: f32,
    },
    InvalidPrincipalPoint {
        camera: &'static str,
        cx: f32,
        cy: f32,
    },
    NotRectified,
}

impl std::fmt::Display for RectifiedStereoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RectifiedStereoError::InvalidLeftDimensions(source) => {
                write!(f, "invalid left camera dimensions: {source}")
            }
            RectifiedStereoError::InvalidRightDimensions(source) => {
                write!(f, "invalid right camera dimensions: {source}")
            }
            RectifiedStereoError::InvalidBaseline { baseline_m } => {
                write!(f, "baseline must be positive and finite, got {baseline_m}")
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
            RectifiedStereoError::InvalidFocal { camera, fx, fy } => {
                write!(
                    f,
                    "rectified stereo requires positive finite {camera} focal lengths: fx={fx}, fy={fy}"
                )
            }
            RectifiedStereoError::InvalidPrincipalPoint { camera, cx, cy } => write!(
                f,
                "rectified stereo requires finite {camera} principal points: cx={cx}, cy={cy}"
            ),
            RectifiedStereoError::NotRectified => {
                write!(f, "calibration is not marked rectified")
            }
        }
    }
}

impl std::error::Error for RectifiedStereoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RectifiedStereoError::InvalidLeftDimensions(source)
            | RectifiedStereoError::InvalidRightDimensions(source) => Some(source),
            RectifiedStereoError::InvalidBaseline { .. }
            | RectifiedStereoError::DimensionMismatch { .. }
            | RectifiedStereoError::InvalidFocal { .. }
            | RectifiedStereoError::InvalidPrincipalPoint { .. }
            | RectifiedStereoError::NotRectified => None,
        }
    }
}

impl RectifiedStereo {
    pub fn from_calibration(calibration: &Calibration) -> Result<Self, RectifiedStereoError> {
        let left = calibration.left.clone();
        let right = calibration.right.clone();
        let left_dimensions = FrameDimensions::try_new(left.width, left.height)
            .map_err(RectifiedStereoError::InvalidLeftDimensions)?;
        let right_dimensions = FrameDimensions::try_new(right.width, right.height)
            .map_err(RectifiedStereoError::InvalidRightDimensions)?;

        if !calibration.baseline_m.is_finite() || calibration.baseline_m <= 0.0 {
            return Err(RectifiedStereoError::InvalidBaseline {
                baseline_m: calibration.baseline_m,
            });
        }

        if left.width != right.width || left.height != right.height {
            return Err(RectifiedStereoError::DimensionMismatch {
                left: left_dimensions,
                right: right_dimensions,
            });
        }

        for (camera, intrinsics) in [("left", &left), ("right", &right)] {
            if !intrinsics.fx.is_finite()
                || !intrinsics.fy.is_finite()
                || intrinsics.fx <= 0.0
                || intrinsics.fy <= 0.0
            {
                return Err(RectifiedStereoError::InvalidFocal {
                    camera,
                    fx: intrinsics.fx,
                    fy: intrinsics.fy,
                });
            }
            if !intrinsics.cx.is_finite() || !intrinsics.cy.is_finite() {
                return Err(RectifiedStereoError::InvalidPrincipalPoint {
                    camera,
                    cx: intrinsics.cx,
                    cy: intrinsics.cy,
                });
            }
        }

        if !calibration.rectified {
            return Err(RectifiedStereoError::NotRectified);
        }

        let arithmetic = RectifiedStereoArithmetic::new(&left, &right, calibration.baseline_m);

        Ok(Self {
            left,
            right,
            dimensions: left_dimensions,
            baseline_m: calibration.baseline_m,
            arithmetic,
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
        self.dimensions.width()
    }

    pub fn height(&self) -> u32 {
        self.dimensions.height()
    }

    pub fn dimensions(&self) -> FrameDimensions {
        self.dimensions
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
    max_vertical_disparity_px: RectifiedRowMismatchPx,
}

// Selected from Kiko's full-resolution lab sweep. A 3 px gate rejected only
// 0.70% of stereo candidates, preserved baseline tracking support, and still
// rejects gross rectified-row outliers; 2 px measurably reduced map support.
const DEFAULT_MAX_VERTICAL_DISPARITY_PX: RectifiedRowMismatchPx = RectifiedRowMismatchPx(3.0);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TriangulationConfigError {
    InvalidMinDisparityPx { value: f32 },
    InvalidMaxDepthM { value: f32 },
    InvalidMaxVerticalDisparityPx { source: RectifiedRowMismatchError },
}

impl std::fmt::Display for TriangulationConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TriangulationConfigError::InvalidMinDisparityPx { value } => write!(
                f,
                "minimum disparity must be non-negative and finite, got {value}"
            ),
            TriangulationConfigError::InvalidMaxDepthM { value } => {
                write!(f, "maximum depth must be positive and finite, got {value}")
            }
            TriangulationConfigError::InvalidMaxVerticalDisparityPx { source } => {
                write!(f, "invalid maximum vertical disparity: {source}")
            }
        }
    }
}

impl std::error::Error for TriangulationConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidMaxVerticalDisparityPx { source } => Some(source),
            Self::InvalidMinDisparityPx { .. } | Self::InvalidMaxDepthM { .. } => None,
        }
    }
}

impl TriangulationConfig {
    pub fn new(
        min_disparity_px: f32,
        max_depth_m: Option<f32>,
    ) -> Result<Self, TriangulationConfigError> {
        Self::new_with_vertical_disparity(
            min_disparity_px,
            max_depth_m,
            DEFAULT_MAX_VERTICAL_DISPARITY_PX.value_px(),
        )
    }

    /// Construct a triangulation policy with an explicit rectified-row gate.
    ///
    /// The acceptable mismatch depends on calibration residuals and feature
    /// localization uncertainty. Deployments should supply a measured value
    /// instead of treating the conservative fallback as a universal constant.
    pub fn new_with_vertical_disparity(
        min_disparity_px: f32,
        max_depth_m: Option<f32>,
        max_vertical_disparity_px: f32,
    ) -> Result<Self, TriangulationConfigError> {
        if !min_disparity_px.is_finite() || min_disparity_px < 0.0 {
            return Err(TriangulationConfigError::InvalidMinDisparityPx {
                value: min_disparity_px,
            });
        }
        if let Some(value) = max_depth_m {
            if !value.is_finite() || value <= 0.0 {
                return Err(TriangulationConfigError::InvalidMaxDepthM { value });
            }
        }
        let max_vertical_disparity_px = RectifiedRowMismatchPx::new(max_vertical_disparity_px)
            .map_err(|source| TriangulationConfigError::InvalidMaxVerticalDisparityPx { source })?;
        Ok(Self {
            min_disparity_px,
            max_depth_m,
            max_vertical_disparity_px,
        })
    }

    pub fn min_disparity_px(self) -> f32 {
        self.min_disparity_px
    }

    pub fn max_depth_m(self) -> Option<f32> {
        self.max_depth_m
    }

    pub fn max_vertical_disparity_px(self) -> RectifiedRowMismatchPx {
        self.max_vertical_disparity_px
    }
}

impl Default for TriangulationConfig {
    fn default() -> Self {
        Self {
            min_disparity_px: 1.0,
            max_depth_m: None,
            max_vertical_disparity_px: DEFAULT_MAX_VERTICAL_DISPARITY_PX,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TriangulationStats {
    pub candidate_matches: usize,
    pub kept: usize,
    pub dropped_disparity: usize,
    pub dropped_epipolar: usize,
    pub dropped_depth: usize,
    pub dropped_numerical: usize,
    /// Geometrically valid f64 results that cannot be represented as a finite
    /// positive-depth f32 landmark or sparse sample.
    pub dropped_unrepresentable: usize,
    pub dropped_duplicate: usize,
}

#[derive(Debug)]
pub enum TriangulationError {
    SensorMismatch {
        left: SensorId,
        right: SensorId,
    },
    DetectionDimensionsMismatch {
        expected: FrameDimensions,
        left: FrameDimensions,
        right: FrameDimensions,
    },
    NoLandmarks {
        stats: TriangulationStats,
    },
    InvalidKeyframe {
        source: KeyframeError,
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
            TriangulationError::DetectionDimensionsMismatch {
                expected,
                left,
                right,
            } => write!(
                f,
                "triangulation detection dimensions must match calibration {}x{}: left={}x{}, right={}x{}",
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
                    "triangulation produced no landmarks (candidates={}, dropped_disparity={}, dropped_epipolar={}, dropped_depth={}, dropped_numerical={}, dropped_unrepresentable={}, dropped_duplicate={})",
                    stats.candidate_matches,
                    stats.dropped_disparity,
                    stats.dropped_epipolar,
                    stats.dropped_depth,
                    stats.dropped_numerical,
                    stats.dropped_unrepresentable,
                    stats.dropped_duplicate
                )
            }
            TriangulationError::InvalidKeyframe { source } => {
                write!(f, "failed to build triangulation keyframe: {source}")
            }
        }
    }
}

impl std::error::Error for TriangulationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TriangulationError::InvalidKeyframe { source } => Some(source),
            TriangulationError::SensorMismatch { .. }
            | TriangulationError::DetectionDimensionsMismatch { .. }
            | TriangulationError::NoLandmarks { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RectifiedRowMismatchError {
    NonFinite { value: f32 },
    Negative { value: f32 },
}

impl std::fmt::Display for RectifiedRowMismatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RectifiedRowMismatchError::NonFinite { value } => {
                write!(f, "rectified row mismatch must be finite, got {value}")
            }
            RectifiedRowMismatchError::Negative { value } => {
                write!(f, "rectified row mismatch must be >= 0, got {value}")
            }
        }
    }
}

impl std::error::Error for RectifiedRowMismatchError {}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct RectifiedRowMismatchPx(f32);

impl RectifiedRowMismatchPx {
    pub fn new(value_px: f32) -> Result<Self, RectifiedRowMismatchError> {
        if !value_px.is_finite() {
            return Err(RectifiedRowMismatchError::NonFinite { value: value_px });
        }
        if value_px < 0.0 {
            return Err(RectifiedRowMismatchError::Negative { value: value_px });
        }
        Ok(Self(value_px))
    }

    pub fn value_px(self) -> f32 {
        self.0
    }
}

/// A validated sparse stereo sample: left pixel coordinate + disparity.
/// Produced by the same deduplication and filtering as triangulation.
#[derive(Clone, Copy, Debug)]
pub struct SparseStereoSample {
    pub(crate) u: f32,
    pub(crate) v: f32,
    pub(crate) right_u: f32,
    pub(crate) right_v: f32,
    pub(crate) disparity: f32,
    pub(crate) depth_m: f32,
    /// Absolute vertical row mismatch on the rectified stereo pair, in pixels.
    pub(crate) rectified_row_mismatch_px: RectifiedRowMismatchPx,
}

impl SparseStereoSample {
    pub fn left_pixel_px(self) -> [f32; 2] {
        [self.u, self.v]
    }

    pub fn right_pixel_px(self) -> [f32; 2] {
        [self.right_u, self.right_v]
    }

    pub fn disparity_px(self) -> f32 {
        self.disparity
    }

    pub fn depth_m(self) -> f32 {
        self.depth_m
    }

    pub fn rectified_row_mismatch_px(self) -> RectifiedRowMismatchPx {
        self.rectified_row_mismatch_px
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Point3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Why a camera-frame landmark cannot cross the public `Keyframe` boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyframeLandmarkError {
    NonFiniteX,
    NonFiniteY,
    NonFiniteDepth,
    NonPositiveDepth,
}

impl std::fmt::Display for KeyframeLandmarkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteX => write!(f, "x coordinate is not finite"),
            Self::NonFiniteY => write!(f, "y coordinate is not finite"),
            Self::NonFiniteDepth => write!(f, "z coordinate is not finite"),
            Self::NonPositiveDepth => write!(f, "z coordinate is not positive"),
        }
    }
}

impl std::error::Error for KeyframeLandmarkError {}

#[derive(Debug)]
pub struct Keyframe {
    frame_id: FrameId,
    detections: Arc<Detections>,
    tracking_detections: Arc<Detections>,
    tracking_detection_indices: Vec<usize>,
    landmarks: Vec<Point3>,
    landmark_indices: Vec<usize>,
    index_to_landmark: Vec<Option<usize>>,
}

#[derive(Debug)]
pub enum KeyframeError {
    Empty,
    LenMismatch {
        landmarks: usize,
        landmark_indices: usize,
    },
    InvalidLandmark {
        index: usize,
        cause: KeyframeLandmarkError,
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
    InvalidTrackingDetections {
        source: DetectionError,
    },
}

impl std::fmt::Display for KeyframeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KeyframeError::Empty => write!(f, "keyframe must contain at least one landmark"),
            KeyframeError::LenMismatch {
                landmarks,
                landmark_indices,
            } => write!(
                f,
                "keyframe landmarks/indices length mismatch: landmarks={landmarks}, landmark_indices={landmark_indices}"
            ),
            KeyframeError::InvalidLandmark { index, cause } => {
                write!(f, "keyframe landmark {index} is invalid: {cause}")
            }
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
            KeyframeError::InvalidTrackingDetections { source } => {
                write!(f, "failed to build tracking detections: {source}")
            }
        }
    }
}

impl std::error::Error for KeyframeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            KeyframeError::InvalidTrackingDetections { source } => Some(source),
            KeyframeError::InvalidLandmark { cause, .. } => Some(cause),
            KeyframeError::Empty
            | KeyframeError::LenMismatch { .. }
            | KeyframeError::LandmarkIndexOutOfBounds { .. }
            | KeyframeError::DuplicateLandmarkIndex { .. }
            | KeyframeError::SensorMismatch { .. } => None,
        }
    }
}

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
        Self::validate_shape_and_sensor(&detections, &landmarks, &landmark_indices)?;
        for (index, &landmark) in landmarks.iter().enumerate() {
            validate_camera_landmark(landmark)
                .map_err(|cause| KeyframeError::InvalidLandmark { index, cause })?;
        }
        Self::build(detections, landmarks, landmark_indices)
    }

    fn from_arc_with_validated_landmarks(
        detections: Arc<Detections>,
        landmarks: Vec<Point3>,
        landmark_indices: Vec<usize>,
    ) -> Result<Self, KeyframeError> {
        Self::validate_shape_and_sensor(&detections, &landmarks, &landmark_indices)?;
        Self::build(detections, landmarks, landmark_indices)
    }

    fn validate_shape_and_sensor(
        detections: &Detections,
        landmarks: &[Point3],
        landmark_indices: &[usize],
    ) -> Result<(), KeyframeError> {
        if detections.is_empty() || landmarks.is_empty() || landmark_indices.is_empty() {
            return Err(KeyframeError::Empty);
        }
        if landmarks.len() != landmark_indices.len() {
            return Err(KeyframeError::LenMismatch {
                landmarks: landmarks.len(),
                landmark_indices: landmark_indices.len(),
            });
        }
        if detections.sensor_id() != SensorId::StereoLeft {
            return Err(KeyframeError::SensorMismatch {
                expected: SensorId::StereoLeft,
                actual: detections.sensor_id(),
            });
        }
        Ok(())
    }

    fn build(
        detections: Arc<Detections>,
        landmarks: Vec<Point3>,
        landmark_indices: Vec<usize>,
    ) -> Result<Self, KeyframeError> {
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

        let tracking_detections = Arc::new(
            detections
                .select(&landmark_indices)
                .map_err(|source| KeyframeError::InvalidTrackingDetections { source })?,
        );

        Ok(Self {
            frame_id: detections.frame_id(),
            detections,
            tracking_detections,
            tracking_detection_indices: landmark_indices.clone(),
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

    pub fn tracking_detections(&self) -> &Arc<Detections> {
        &self.tracking_detections
    }

    pub fn remap_tracking_matches(
        &self,
        matches: &Matches<Raw>,
    ) -> Result<Matches<Raw>, crate::MatchError> {
        matches.require_source_b(&self.tracking_detections, "remap tracking matches")?;
        let mut indices = Vec::with_capacity(matches.len());
        let mut scores = Vec::with_capacity(matches.len());
        for (match_idx, &(current_idx, tracking_idx)) in matches.indices().iter().enumerate() {
            let Some(&keyframe_idx) = self.tracking_detection_indices.get(tracking_idx) else {
                continue;
            };
            indices.push((current_idx, keyframe_idx));
            scores.push(matches.scores()[match_idx]);
        }
        Matches::new(
            matches.source_a_arc(),
            Arc::clone(&self.detections),
            indices,
            scores,
        )
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

fn validate_camera_landmark(point: Point3) -> Result<(), KeyframeLandmarkError> {
    if !point.x.is_finite() {
        return Err(KeyframeLandmarkError::NonFiniteX);
    }
    if !point.y.is_finite() {
        return Err(KeyframeLandmarkError::NonFiniteY);
    }
    if !point.z.is_finite() {
        return Err(KeyframeLandmarkError::NonFiniteDepth);
    }
    if point.z <= 0.0 {
        return Err(KeyframeLandmarkError::NonPositiveDepth);
    }
    Ok(())
}

fn narrow_camera_landmark(x: f64, y: f64, z: f64) -> Result<Point3, KeyframeLandmarkError> {
    let point = Point3 {
        x: x as f32,
        y: y as f32,
        z: z as f32,
    };
    validate_camera_landmark(point)?;
    Ok(point)
}

#[derive(Debug)]
pub struct TriangulationResult {
    pub keyframe: Keyframe,
    pub stats: TriangulationStats,
}

/// Deduplicated stereo geometry bound to the exact detections and rectified rig
/// that produced it. Public callers can inspect but cannot forge the batch.
#[derive(Debug)]
pub struct SparseStereoSamples {
    samples: Vec<SparseStereoSample>,
    stats: TriangulationStats,
    left_frame_id: FrameId,
    right_frame_id: FrameId,
    stereo: RectifiedStereo,
}

impl SparseStereoSamples {
    pub fn samples(&self) -> &[SparseStereoSample] {
        &self.samples
    }

    pub fn stats(&self) -> TriangulationStats {
        self.stats
    }

    pub fn left_frame_id(&self) -> FrameId {
        self.left_frame_id
    }

    pub fn right_frame_id(&self) -> FrameId {
        self.right_frame_id
    }

    pub fn stereo(&self) -> &RectifiedStereo {
        &self.stereo
    }

    #[cfg(test)]
    pub(crate) fn for_test(
        samples: Vec<SparseStereoSample>,
        left_frame_id: FrameId,
        right_frame_id: FrameId,
        stereo: RectifiedStereo,
    ) -> Self {
        let count = samples.len();
        Self {
            samples,
            stats: TriangulationStats {
                candidate_matches: count,
                kept: count,
                ..TriangulationStats::default()
            },
            left_frame_id,
            right_frame_id,
            stereo,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct StereoGeometry {
    point: Point3,
    sample: SparseStereoSample,
}

#[derive(Clone, Copy, Debug)]
enum StereoGeometryRejection {
    Disparity,
    Epipolar,
    Depth,
    Numerical,
    Unrepresentable,
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

    pub fn stereo(&self) -> &RectifiedStereo {
        &self.stereo
    }

    fn validate_sources(&self, matches: &Matches<Raw>) -> Result<(), TriangulationError> {
        let left = matches.source_a();
        let right = matches.source_b();
        if left.sensor_id() != SensorId::StereoLeft || right.sensor_id() != SensorId::StereoRight {
            return Err(TriangulationError::SensorMismatch {
                left: left.sensor_id(),
                right: right.sensor_id(),
            });
        }
        let expected = self.stereo.dimensions();
        if left.dimensions() != expected || right.dimensions() != expected {
            return Err(TriangulationError::DetectionDimensionsMismatch {
                expected,
                left: left.dimensions(),
                right: right.dimensions(),
            });
        }
        Ok(())
    }

    fn deduplicate_matches(matches: &Matches<Raw>) -> (Vec<Option<usize>>, usize) {
        let mut best: Vec<Option<(usize, f32)>> = vec![None; matches.source_a().len()];
        let mut dropped = 0;
        for (&(li, ri), &score) in matches.indices().iter().zip(matches.scores()) {
            match best[li] {
                Some((_, best_score)) if best_score >= score => dropped += 1,
                Some(_) => {
                    dropped += 1;
                    best[li] = Some((ri, score));
                }
                None => best[li] = Some((ri, score)),
            }
        }
        (
            best.into_iter()
                .map(|candidate| candidate.map(|(index, _)| index))
                .collect(),
            dropped,
        )
    }

    fn reconstruct(
        &self,
        left_keypoint: Keypoint,
        right_keypoint: Keypoint,
    ) -> Result<StereoGeometry, StereoGeometryRejection> {
        let arithmetic = &self.stereo.arithmetic;
        let left_x = (f64::from(left_keypoint.x) - arithmetic.left.cx) / arithmetic.left.fx;
        let right_x = (f64::from(right_keypoint.x) - arithmetic.right.cx) / arithmetic.right.fx;
        let normalized_disparity = left_x - right_x;
        let disparity_px = normalized_disparity * arithmetic.left.fx;
        if !left_x.is_finite()
            || !right_x.is_finite()
            || !normalized_disparity.is_finite()
            || !disparity_px.is_finite()
        {
            return Err(StereoGeometryRejection::Numerical);
        }
        if disparity_px <= f64::from(self.config.min_disparity_px()) {
            return Err(StereoGeometryRejection::Disparity);
        }

        let depth_m = arithmetic.baseline_m / normalized_disparity;
        if let Some(max_depth_m) = self.config.max_depth_m() {
            if depth_m > f64::from(max_depth_m) {
                return Err(StereoGeometryRejection::Depth);
            }
        }
        let left_y_px = f64::from(left_keypoint.y) - arithmetic.left.cy;
        let left_y = left_y_px / arithmetic.left.fy;
        let right_y_in_left_pixels = (f64::from(right_keypoint.y) - arithmetic.right.cy)
            * (arithmetic.left.fy / arithmetic.right.fy);
        let x_m = left_x * depth_m;
        let y_m = left_y * depth_m;
        let row_mismatch_px = (left_y_px - right_y_in_left_pixels).abs();
        if ![depth_m, x_m, y_m, right_y_in_left_pixels, row_mismatch_px]
            .into_iter()
            .all(f64::is_finite)
        {
            return Err(StereoGeometryRejection::Numerical);
        }
        if row_mismatch_px > f64::from(self.config.max_vertical_disparity_px().value_px()) {
            return Err(StereoGeometryRejection::Epipolar);
        }

        let disparity = disparity_px as f32;
        let row_mismatch_px = row_mismatch_px as f32;
        if !disparity.is_finite()
            || disparity <= 0.0
            || !row_mismatch_px.is_finite()
            || row_mismatch_px < 0.0
        {
            return Err(StereoGeometryRejection::Unrepresentable);
        }
        let point = narrow_camera_landmark(x_m, y_m, depth_m)
            .map_err(|_| StereoGeometryRejection::Unrepresentable)?;
        let rectified_row_mismatch_px = RectifiedRowMismatchPx::new(row_mismatch_px)
            .map_err(|_| StereoGeometryRejection::Unrepresentable)?;
        Ok(StereoGeometry {
            point,
            sample: SparseStereoSample {
                u: left_keypoint.x,
                v: left_keypoint.y,
                right_u: right_keypoint.x,
                right_v: right_keypoint.y,
                disparity,
                depth_m: point.z,
                rectified_row_mismatch_px,
            },
        })
    }

    fn reconstruct_matches(
        &self,
        matches: &Matches<Raw>,
    ) -> Result<(Vec<(usize, StereoGeometry)>, TriangulationStats), TriangulationError> {
        self.validate_sources(matches)?;
        let left = matches.source_a();
        let right = matches.source_b();
        let (best, dropped_duplicate) = Self::deduplicate_matches(matches);
        let mut stats = TriangulationStats {
            candidate_matches: matches.len(),
            dropped_duplicate,
            ..TriangulationStats::default()
        };
        let mut geometry = Vec::with_capacity(matches.len().saturating_sub(dropped_duplicate));
        for (left_index, right_index) in best.into_iter().enumerate() {
            let Some(right_index) = right_index else {
                continue;
            };
            match self.reconstruct(left.keypoints()[left_index], right.keypoints()[right_index]) {
                Ok(point) => {
                    geometry.push((left_index, point));
                    stats.kept += 1;
                }
                Err(StereoGeometryRejection::Disparity) => stats.dropped_disparity += 1,
                Err(StereoGeometryRejection::Epipolar) => stats.dropped_epipolar += 1,
                Err(StereoGeometryRejection::Depth) => stats.dropped_depth += 1,
                Err(StereoGeometryRejection::Numerical) => stats.dropped_numerical += 1,
                Err(StereoGeometryRejection::Unrepresentable) => stats.dropped_unrepresentable += 1,
            }
        }
        Ok((geometry, stats))
    }

    /// Extract deduplicated, filtered stereo samples from matches.
    pub fn extract_stereo_samples(
        &self,
        matches: &Matches<Raw>,
    ) -> Result<SparseStereoSamples, TriangulationError> {
        let (geometry, stats) = self.reconstruct_matches(matches)?;
        Ok(SparseStereoSamples {
            samples: geometry
                .into_iter()
                .map(|(_, point)| point.sample)
                .collect(),
            stats,
            left_frame_id: matches.source_a().frame_id(),
            right_frame_id: matches.source_b().frame_id(),
            stereo: self.stereo.clone(),
        })
    }

    pub fn triangulate(
        &self,
        matches: &Matches<Raw>,
    ) -> Result<TriangulationResult, TriangulationError> {
        let left = matches.source_a_arc();
        let (geometry, stats) = self.reconstruct_matches(matches)?;
        let mut landmarks = Vec::with_capacity(geometry.len());
        let mut landmark_indices = Vec::with_capacity(geometry.len());
        for (left_index, point) in geometry {
            landmarks.push(point.point);
            landmark_indices.push(left_index);
        }

        if landmarks.is_empty() {
            return Err(TriangulationError::NoLandmarks { stats });
        }

        let keyframe =
            Keyframe::from_arc_with_validated_landmarks(left, landmarks, landmark_indices)
                .map_err(|source| TriangulationError::InvalidKeyframe { source })?;

        Ok(TriangulationResult { keyframe, stats })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{Calibration, CameraIntrinsics};
    use crate::test_helpers::{
        make_detections, make_pinhole_intrinsics, make_rectified_stereo,
        rectified_stereo_keypoints_from_points,
    };
    use crate::{FrameId, Matches};

    fn valid_calibration() -> Calibration {
        Calibration {
            left: CameraIntrinsics {
                fx: 400.0,
                fy: 400.0,
                cx: 320.0,
                cy: 240.0,
                width: 640,
                height: 480,
            },
            right: CameraIntrinsics {
                fx: 400.0,
                fy: 400.0,
                cx: 320.0,
                cy: 240.0,
                width: 640,
                height: 480,
            },
            baseline_m: 0.075,
            rectified: true,
            imu: None,
        }
    }

    fn assert_stats_accounting(stats: TriangulationStats) {
        assert_eq!(
            stats.kept
                + stats.dropped_disparity
                + stats.dropped_epipolar
                + stats.dropped_depth
                + stats.dropped_numerical
                + stats.dropped_unrepresentable
                + stats.dropped_duplicate,
            stats.candidate_matches
        );
    }

    #[test]
    fn rectified_row_mismatch_rejects_invalid_values() {
        assert!(matches!(
            RectifiedRowMismatchPx::new(-0.1),
            Err(RectifiedRowMismatchError::Negative { .. })
        ));
        assert!(matches!(
            RectifiedRowMismatchPx::new(f32::NAN),
            Err(RectifiedRowMismatchError::NonFinite { .. })
        ));
        let mismatch = RectifiedRowMismatchPx::new(0.25).expect("row mismatch");
        assert_eq!(mismatch.value_px(), 0.25);
    }

    #[test]
    fn rectified_stereo_rejects_malformed_calibration() {
        let mut calibration = valid_calibration();
        calibration.baseline_m = f32::NAN;
        assert!(matches!(
            RectifiedStereo::from_calibration(&calibration),
            Err(RectifiedStereoError::InvalidBaseline { .. })
        ));

        let mut calibration = valid_calibration();
        calibration.right.fx = 0.0;
        assert!(matches!(
            RectifiedStereo::from_calibration(&calibration),
            Err(RectifiedStereoError::InvalidFocal {
                camera: "right",
                ..
            })
        ));

        let mut calibration = valid_calibration();
        calibration.left.cx = f32::INFINITY;
        assert!(matches!(
            RectifiedStereo::from_calibration(&calibration),
            Err(RectifiedStereoError::InvalidPrincipalPoint { camera: "left", .. })
        ));

        let mut calibration = valid_calibration();
        calibration.rectified = false;
        assert!(matches!(
            RectifiedStereo::from_calibration(&calibration),
            Err(RectifiedStereoError::NotRectified)
        ));
    }

    #[test]
    fn triangulation_config_rejects_invalid_thresholds() {
        let default_policy =
            TriangulationConfig::new(1.0, None).expect("default triangulation policy");
        assert_eq!(default_policy.max_vertical_disparity_px().value_px(), 3.0);
        let gated_policy = TriangulationConfig::new_with_vertical_disparity(1.0, None, 1.0)
            .expect("explicit row-mismatch gate");
        assert_eq!(gated_policy.max_vertical_disparity_px().value_px(), 1.0);
        assert!(matches!(
            TriangulationConfig::new(f32::NAN, None),
            Err(TriangulationConfigError::InvalidMinDisparityPx { .. })
        ));
        assert!(matches!(
            TriangulationConfig::new(1.0, Some(0.0)),
            Err(TriangulationConfigError::InvalidMaxDepthM { .. })
        ));
        for value in [-1.0, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                TriangulationConfig::new_with_vertical_disparity(1.0, None, value),
                Err(TriangulationConfigError::InvalidMaxVerticalDisparityPx { .. })
            ));
        }
    }

    #[test]
    fn triangulation_error_preserves_keyframe_source() {
        let err = TriangulationError::InvalidKeyframe {
            source: KeyframeError::Empty,
        };
        let source = std::error::Error::source(&err).expect("nested keyframe error");
        assert_eq!(source.to_string(), KeyframeError::Empty.to_string());
    }

    #[test]
    fn triangulate_recovers_known_depth_for_rectified_pairs() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 400.0, 402.0, 320.0, 240.0).expect("intrinsics");
        let stereo =
            make_rectified_stereo(640, 480, 400.0, 402.0, 320.0, 240.0, 0.075).expect("stereo");
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
    fn triangulate_uses_both_cameras_intrinsics_in_normalized_geometry() {
        let stereo = RectifiedStereo::from_calibration(&Calibration {
            left: CameraIntrinsics {
                fx: 400.0,
                fy: 400.0,
                cx: 320.0,
                cy: 240.0,
                width: 640,
                height: 480,
            },
            right: CameraIntrinsics {
                fx: 420.0,
                fy: 410.0,
                cx: 316.0,
                cy: 238.0,
                width: 640,
                height: 480,
            },
            baseline_m: 0.075,
            rectified: true,
            imu: None,
        })
        .expect("stereo");
        let triangulator = Triangulator::new(stereo, TriangulationConfig::default());

        let point = Point3 {
            x: 0.15,
            y: -0.05,
            z: 2.8,
        };
        let left_kp = Keypoint {
            x: 400.0 * point.x / point.z + 320.0,
            y: 400.0 * point.y / point.z + 240.0,
        };
        let right_x = 420.0 * (point.x - 0.075) / point.z + 316.0;
        let right_kp = Keypoint {
            x: right_x,
            y: 410.0 * point.y / point.z + 238.0,
        };

        let left = make_detections(
            SensorId::StereoLeft,
            FrameId::new(12),
            640,
            480,
            vec![left_kp],
        )
        .expect("left detections");
        let right = make_detections(
            SensorId::StereoRight,
            FrameId::new(13),
            640,
            480,
            vec![right_kp],
        )
        .expect("right detections");
        let matches = Matches::new(left, right, vec![(0, 0)], vec![1.0]).expect("matches");

        let result = triangulator.triangulate(&matches).expect("triangulation");
        let recovered = result.keyframe.landmarks()[0];
        assert!((recovered.x - point.x).abs() < 1e-4);
        assert!((recovered.y - point.y).abs() < 1e-4);
        assert!((recovered.z - point.z).abs() < 1e-4);
        let samples = triangulator
            .extract_stereo_samples(&matches)
            .expect("stereo samples");
        assert_eq!(samples.left_frame_id(), FrameId::new(12));
        assert_eq!(samples.right_frame_id(), FrameId::new(13));
        assert_eq!(samples.stereo().right().fx, 420.0);
        assert_eq!(samples.stats().kept, 1);
        let sample = samples.samples()[0];
        assert_eq!(sample.left_pixel_px(), [left_kp.x, left_kp.y]);
        assert_eq!(sample.right_pixel_px(), [right_kp.x, right_kp.y]);
        assert!(sample.disparity_px() > 0.0);
        assert!((sample.depth_m() - recovered.z).abs() < 1e-4);
        assert!(sample.rectified_row_mismatch_px().value_px() < 1e-4);
    }

    #[test]
    fn triangulation_requires_detection_dimensions_to_match_calibration() {
        let stereo = RectifiedStereo::from_calibration(&valid_calibration()).expect("stereo");
        let triangulator = Triangulator::new(stereo, TriangulationConfig::default());
        let left = make_detections(
            SensorId::StereoLeft,
            FrameId::new(14),
            320,
            240,
            vec![Keypoint { x: 100.0, y: 100.0 }],
        )
        .expect("left detections");
        let right = make_detections(
            SensorId::StereoRight,
            FrameId::new(15),
            320,
            240,
            vec![Keypoint { x: 90.0, y: 100.0 }],
        )
        .expect("right detections");
        let matches = Matches::new(left, right, vec![(0, 0)], vec![1.0]).expect("matches");

        assert!(matches!(
            triangulator.triangulate(&matches),
            Err(TriangulationError::DetectionDimensionsMismatch { .. })
        ));
        assert!(matches!(
            triangulator.extract_stereo_samples(&matches),
            Err(TriangulationError::DetectionDimensionsMismatch { .. })
        ));
    }

    #[test]
    fn triangulation_rejects_vertical_epipolar_mismatch() {
        let stereo =
            make_rectified_stereo(640, 480, 400.0, 400.0, 320.0, 240.0, 0.075).expect("stereo");
        let triangulator = Triangulator::new(stereo, TriangulationConfig::default());
        let left = make_detections(
            SensorId::StereoLeft,
            FrameId::new(14),
            640,
            480,
            vec![Keypoint { x: 320.0, y: 0.0 }],
        )
        .expect("left detections");
        let right = make_detections(
            SensorId::StereoRight,
            FrameId::new(15),
            640,
            480,
            vec![Keypoint { x: 310.0, y: 479.0 }],
        )
        .expect("right detections");
        let matches = Matches::new(left, right, vec![(0, 0)], vec![1.0]).expect("matches");

        let error = triangulator
            .triangulate(&matches)
            .expect_err("skew rectified rays cannot define a landmark");
        let TriangulationError::NoLandmarks { stats } = error else {
            panic!("unexpected triangulation error: {error:?}");
        };
        assert_eq!(stats.dropped_epipolar, 1);
        assert_stats_accounting(stats);
    }

    #[test]
    fn triangulation_keeps_geometry_at_the_explicit_row_mismatch_limit() {
        let stereo =
            make_rectified_stereo(640, 480, 400.0, 400.0, 320.0, 240.0, 0.075).expect("stereo");
        let config = TriangulationConfig::new_with_vertical_disparity(1.0, None, 1.0)
            .expect("explicit row-mismatch gate");
        let triangulator = Triangulator::new(stereo, config);
        let left = make_detections(
            SensorId::StereoLeft,
            FrameId::new(14),
            640,
            480,
            vec![Keypoint { x: 320.0, y: 100.0 }],
        )
        .expect("left detections");
        let right = make_detections(
            SensorId::StereoRight,
            FrameId::new(15),
            640,
            480,
            vec![Keypoint { x: 310.0, y: 101.0 }],
        )
        .expect("right detections");
        let matches = Matches::new(left, right, vec![(0, 0)], vec![1.0]).expect("matches");

        let result = triangulator
            .triangulate(&matches)
            .expect("the explicit maximum is inclusive");
        assert_eq!(result.stats.kept, 1);
        assert_eq!(result.stats.dropped_epipolar, 0);
        assert_stats_accounting(result.stats);
    }

    #[test]
    fn permissive_rectified_row_gate_preserves_measured_mismatch() {
        let stereo =
            make_rectified_stereo(640, 480, 400.0, 400.0, 320.0, 240.0, 0.075).expect("stereo");
        let config = TriangulationConfig::new_with_vertical_disparity(1.0, None, 3.0)
            .expect("measured row-mismatch gate");
        let triangulator = Triangulator::new(stereo, config);
        let left = make_detections(
            SensorId::StereoLeft,
            FrameId::new(14),
            640,
            480,
            vec![Keypoint { x: 320.0, y: 100.0 }],
        )
        .expect("left detections");
        let right = make_detections(
            SensorId::StereoRight,
            FrameId::new(15),
            640,
            480,
            vec![Keypoint { x: 310.0, y: 102.0 }],
        )
        .expect("right detections");
        let matches = Matches::new(left, right, vec![(0, 0)], vec![1.0]).expect("matches");

        let result = triangulator
            .triangulate(&matches)
            .expect("geometry lies within the measured row-mismatch gate");
        assert_eq!(result.stats.candidate_matches, 1);
        assert_eq!(result.stats.kept, 1);
        assert_eq!(result.stats.dropped_epipolar, 0);
        assert_stats_accounting(result.stats);

        let samples = triangulator
            .extract_stereo_samples(&matches)
            .expect("same geometry yields a sparse sample");
        assert_eq!(samples.samples().len(), 1);
        assert_eq!(
            samples.samples()[0].rectified_row_mismatch_px().value_px(),
            2.0
        );
    }

    #[test]
    fn triangulation_preserves_positive_f32_subnormal_depth() {
        let mut calibration = valid_calibration();
        calibration.left.fx = f32::MIN_POSITIVE;
        calibration.left.cx = 0.0;
        calibration.right.fx = 1.0;
        calibration.right.cx = 0.0;
        let stereo = RectifiedStereo::from_calibration(&calibration).expect("stereo");
        let triangulator = Triangulator::new(stereo, TriangulationConfig::default());
        let left = make_detections(
            SensorId::StereoLeft,
            FrameId::new(16),
            640,
            480,
            vec![Keypoint { x: 100.0, y: 100.0 }],
        )
        .expect("left detections");
        let right = make_detections(
            SensorId::StereoRight,
            FrameId::new(17),
            640,
            480,
            vec![Keypoint { x: 0.0, y: 100.0 }],
        )
        .expect("right detections");
        let matches = Matches::new(left, right, vec![(0, 0)], vec![1.0]).expect("matches");

        let result = triangulator
            .triangulate(&matches)
            .expect("a finite positive subnormal depth remains a valid landmark");
        assert_stats_accounting(result.stats);
        assert_eq!(result.stats.kept, 1);
        let landmark = result.keyframe.landmarks()[0];
        assert!(landmark.x.is_finite());
        assert!(landmark.y.is_finite());
        assert!(landmark.z.is_finite());
        assert!(landmark.z > 0.0);
    }

    #[test]
    fn triangulate_rejects_points_below_min_disparity() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 400.0, 400.0, 320.0, 240.0).expect("intrinsics");
        let stereo =
            make_rectified_stereo(640, 480, 400.0, 400.0, 320.0, 240.0, 0.075).expect("stereo");
        let triangulator = Triangulator::new(
            stereo,
            TriangulationConfig::new(1.0, None).expect("triangulation config"),
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
    fn matches_reject_out_of_bounds_before_triangulation() {
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
        assert!(matches!(
            Matches::new(left, right, vec![(0, 2)], vec![1.0]),
            Err(crate::MatchError::SourceBIndexOutOfBounds {
                match_index: 0,
                detection_index: 2,
                detection_count: 1,
            })
        ));
    }

    #[test]
    fn triangulate_uses_best_score_for_duplicate_left_matches() {
        let stereo =
            make_rectified_stereo(640, 480, 400.0, 400.0, 320.0, 240.0, 0.075).expect("stereo");
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
            vec![
                Keypoint { x: 335.0, y: 240.0 },
                Keypoint { x: 345.0, y: 240.0 },
            ],
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

    #[test]
    fn triangulation_widens_tiny_focal_arithmetic_before_computing_landmarks() {
        let tiny_focal = f32::from_bits(1);
        let stereo = make_rectified_stereo(2, 2, tiny_focal, tiny_focal, 0.0, 0.0, 1.0)
            .expect("tiny but positive focal lengths are representable");
        let triangulator = Triangulator::new(
            stereo,
            TriangulationConfig::new(0.5, None).expect("triangulation config"),
        );
        let left = make_detections(
            SensorId::StereoLeft,
            FrameId::new(62),
            2,
            2,
            vec![Keypoint { x: 1.0, y: 1.0 }],
        )
        .expect("left");
        let right = make_detections(
            SensorId::StereoRight,
            FrameId::new(63),
            2,
            2,
            vec![Keypoint { x: 0.0, y: 1.0 }],
        )
        .expect("right");
        let matches = Matches::new(left, right, vec![(0, 0)], vec![1.0]).expect("matches");

        let result = triangulator
            .triangulate(&matches)
            .expect("f64 rays avoid intermediate overflow");

        assert_stats_accounting(result.stats);
        assert_eq!(result.stats.kept, 1);
        let point = result.keyframe.landmarks()[0];
        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 1.0);
        assert_eq!(point.z, tiny_focal);
    }

    #[test]
    fn triangulation_counts_finite_f64_points_that_do_not_fit_f32() {
        let stereo =
            make_rectified_stereo(3, 2, 1.0, 1.0, 0.0, 0.0, f32::MAX).expect("finite calibration");
        let triangulator = Triangulator::new(
            stereo,
            TriangulationConfig::new(0.5, None).expect("triangulation config"),
        );
        let left = make_detections(
            SensorId::StereoLeft,
            FrameId::new(64),
            3,
            2,
            vec![Keypoint { x: 2.0, y: 0.0 }],
        )
        .expect("left");
        let right = make_detections(
            SensorId::StereoRight,
            FrameId::new(65),
            3,
            2,
            vec![Keypoint { x: 1.0, y: 0.0 }],
        )
        .expect("right");
        let matches = Matches::new(left, right, vec![(0, 0)], vec![1.0]).expect("matches");

        let error = triangulator
            .triangulate(&matches)
            .expect_err("overflowing f32 x must not enter a keyframe");
        let TriangulationError::NoLandmarks { stats } = error else {
            panic!("unexpected error: {error:?}");
        };
        assert_eq!(stats.dropped_unrepresentable, 1);
        assert_stats_accounting(stats);
    }

    #[test]
    fn keyframe_rejects_invalid_camera_landmarks_with_typed_causes() {
        let cases = [
            (
                Point3 {
                    x: f32::NAN,
                    y: 0.0,
                    z: 1.0,
                },
                KeyframeLandmarkError::NonFiniteX,
            ),
            (
                Point3 {
                    x: 0.0,
                    y: f32::INFINITY,
                    z: 1.0,
                },
                KeyframeLandmarkError::NonFiniteY,
            ),
            (
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: f32::NAN,
                },
                KeyframeLandmarkError::NonFiniteDepth,
            ),
            (
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                KeyframeLandmarkError::NonPositiveDepth,
            ),
            (
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: -1.0,
                },
                KeyframeLandmarkError::NonPositiveDepth,
            ),
        ];

        for (landmark, expected_cause) in cases {
            let detections = make_detections(
                SensorId::StereoLeft,
                FrameId::new(66),
                2,
                2,
                vec![Keypoint { x: 1.0, y: 1.0 }],
            )
            .expect("detections");
            let error = Keyframe::from_arc(detections, vec![landmark], vec![0])
                .expect_err("invalid camera landmark must be rejected");
            assert!(matches!(
                error,
                KeyframeError::InvalidLandmark { index: 0, cause }
                    if cause == expected_cause
            ));
        }
    }

    #[test]
    fn keyframe_length_mismatch_reports_the_compared_lengths() {
        let detections = make_detections(
            SensorId::StereoLeft,
            FrameId::new(67),
            4,
            2,
            vec![
                Keypoint { x: 1.0, y: 1.0 },
                Keypoint { x: 2.0, y: 1.0 },
                Keypoint { x: 3.0, y: 1.0 },
            ],
        )
        .expect("detections");
        let error = Keyframe::from_arc(
            detections,
            vec![
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                Point3 {
                    x: 1.0,
                    y: 0.0,
                    z: 1.0,
                },
            ],
            vec![0],
        )
        .expect_err("landmark/index lengths differ");

        assert!(matches!(
            error,
            KeyframeError::LenMismatch {
                landmarks: 2,
                landmark_indices: 1
            }
        ));
    }

    #[test]
    fn keyframe_tracking_detections_only_keep_landmark_entries() {
        let detections = make_detections(
            SensorId::StereoLeft,
            FrameId::new(60),
            640,
            480,
            vec![
                Keypoint { x: 10.0, y: 20.0 },
                Keypoint { x: 30.0, y: 40.0 },
                Keypoint { x: 50.0, y: 60.0 },
            ],
        )
        .expect("detections");
        let keyframe = Keyframe::new(
            (*detections).clone(),
            vec![
                Point3 {
                    x: 0.1,
                    y: 0.0,
                    z: 1.0,
                },
                Point3 {
                    x: 0.2,
                    y: 0.0,
                    z: 1.5,
                },
            ],
            vec![0, 2],
        )
        .expect("keyframe");
        let current = make_detections(
            SensorId::StereoLeft,
            FrameId::new(61),
            640,
            480,
            vec![Keypoint { x: 11.0, y: 21.0 }, Keypoint { x: 31.0, y: 41.0 }],
        )
        .expect("current");

        let tracking_matches = Matches::new(
            current.clone(),
            keyframe.tracking_detections().clone(),
            vec![(0, 0), (1, 1)],
            vec![0.9, 0.8],
        )
        .expect("tracking matches");

        let remapped = keyframe
            .remap_tracking_matches(&tracking_matches)
            .expect("remapped matches");

        assert_eq!(keyframe.tracking_detections().len(), 2);
        assert_eq!(remapped.indices(), &[(0, 0), (1, 2)]);
        assert_eq!(remapped.scores(), &[0.9, 0.8]);
        assert_eq!(remapped.source_b().len(), 3);

        let copied_tracking_batch = std::sync::Arc::new((**keyframe.tracking_detections()).clone());
        let wrong_source = Matches::new(current, copied_tracking_batch, vec![(0, 0)], vec![0.9])
            .expect("raw matches");
        assert!(matches!(
            keyframe.remap_tracking_matches(&wrong_source),
            Err(crate::MatchError::SourceBatchMismatch { .. })
        ));
    }
}
