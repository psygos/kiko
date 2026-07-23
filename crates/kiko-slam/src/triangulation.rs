use std::sync::Arc;

use crate::{
    Detections, FrameDimensions, FrameDimensionsError, FrameId, IntrinsicsError, Matches,
    PinholeIntrinsics, Raw, SensorId, math,
};

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
    ) -> Result<Self, RectifiedStereoConfigError> {
        for (kind, value_px) in [
            (StereoToleranceKind::PrincipalPoint, max_principal_delta_px),
            (StereoToleranceKind::FocalLength, max_focal_delta_px),
        ] {
            if !value_px.is_finite() || value_px < 0.0 {
                return Err(RectifiedStereoConfigError::InvalidTolerance { kind, value_px });
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RectifiedStereoConfigError {
    InvalidTolerance {
        kind: StereoToleranceKind,
        value_px: f32,
    },
}

impl std::fmt::Display for RectifiedStereoConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTolerance { kind, value_px } => write!(
                f,
                "{kind} tolerance must be finite and nonnegative, got {value_px} px"
            ),
        }
    }
}

impl std::error::Error for RectifiedStereoConfigError {}

/// Camera side named by a rectified-stereo calibration error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StereoCameraSide {
    Left,
    Right,
}

impl std::fmt::Display for StereoCameraSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Left => f.write_str("left"),
            Self::Right => f.write_str("right"),
        }
    }
}

/// Rectified-stereo compatibility threshold named by a configuration error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StereoToleranceKind {
    PrincipalPoint,
    FocalLength,
}

impl std::fmt::Display for StereoToleranceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PrincipalPoint => f.write_str("principal-point"),
            Self::FocalLength => f.write_str("focal-length"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RectifiedStereo {
    left: PinholeIntrinsics,
    right: PinholeIntrinsics,
    dimensions: FrameDimensions,
    baseline_m: f32,
    arithmetic: RectifiedStereoArithmetic,
}

/// Positive finite stereo baseline, expressed in metres.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StereoBaselineMeters(f32);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StereoBaselineError {
    NonFinite { baseline_m: f32 },
    NonPositive { baseline_m: f32 },
}

impl StereoBaselineMeters {
    pub(crate) fn try_new(baseline_m: f32) -> Result<Self, StereoBaselineError> {
        if !baseline_m.is_finite() {
            return Err(StereoBaselineError::NonFinite { baseline_m });
        }
        if baseline_m <= 0.0 {
            return Err(StereoBaselineError::NonPositive { baseline_m });
        }
        Ok(Self(baseline_m))
    }

    fn get(self) -> f32 {
        self.0
    }
}

impl std::fmt::Display for StereoBaselineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite { baseline_m } => {
                write!(f, "stereo baseline must be finite, got {baseline_m} m")
            }
            Self::NonPositive { baseline_m } => {
                write!(f, "stereo baseline must be positive, got {baseline_m} m")
            }
        }
    }
}

impl std::error::Error for StereoBaselineError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StereoRectification {
    Rectified,
    Unrectified,
}

/// Policy-neutral stereo calibration whose structural facts are already typed.
///
/// Both cameras have finite projection coefficients with positive focal
/// lengths, share one nonzero image size, and have a positive finite metric
/// baseline. Rectification compatibility remains an explicit caller policy.
#[derive(Clone, Copy, Debug)]
pub struct StereoCalibration {
    left: PinholeIntrinsics,
    right: PinholeIntrinsics,
    dimensions: FrameDimensions,
    baseline: StereoBaselineMeters,
    rectification: StereoRectification,
}

impl StereoCalibration {
    pub(crate) fn new(
        left: PinholeIntrinsics,
        right: PinholeIntrinsics,
        dimensions: FrameDimensions,
        baseline: StereoBaselineMeters,
        rectified: bool,
    ) -> Self {
        Self {
            left,
            right,
            dimensions,
            baseline,
            rectification: if rectified {
                StereoRectification::Rectified
            } else {
                StereoRectification::Unrectified
            },
        }
    }

    pub fn try_new(
        left: PinholeIntrinsics,
        right: PinholeIntrinsics,
        dimensions: FrameDimensions,
        baseline_m: f32,
        rectified: bool,
    ) -> Result<Self, StereoBaselineError> {
        Ok(Self::new(
            left,
            right,
            dimensions,
            StereoBaselineMeters::try_new(baseline_m)?,
            rectified,
        ))
    }

    pub fn left(&self) -> PinholeIntrinsics {
        self.left
    }

    pub fn right(&self) -> PinholeIntrinsics {
        self.right
    }

    pub fn dimensions(&self) -> FrameDimensions {
        self.dimensions
    }

    pub fn baseline_m(&self) -> f32 {
        self.baseline.get()
    }

    pub fn is_rectified(&self) -> bool {
        self.rectification == StereoRectification::Rectified
    }
}

/// Exact f32-to-f64 widening of the validated stereo calibration.
///
/// Keeping this alongside the parsed f32 calibration makes the precision used
/// by triangulation an invariant of `RectifiedStereo`, rather than something
/// reconstructed independently for every match.
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

#[derive(Debug)]
pub enum RectifiedStereoError {
    NonFiniteBaseline {
        baseline_m: f32,
    },
    NonPositiveBaseline {
        baseline_m: f32,
    },
    InvalidDimensions {
        camera: StereoCameraSide,
        source: FrameDimensionsError,
    },
    InvalidIntrinsics {
        camera: StereoCameraSide,
        source: IntrinsicsError,
    },
    DimensionMismatch {
        left: FrameDimensions,
        right: FrameDimensions,
    },
    NotRectified,
    FocalMismatch {
        left_fx: f32,
        right_fx: f32,
        left_fy: f32,
        right_fy: f32,
        tolerance_px: f32,
    },
    PrincipalPointMismatch {
        left_cx: f32,
        right_cx: f32,
        left_cy: f32,
        right_cy: f32,
        tolerance_px: f32,
    },
}

/// Compatibility failures that remain possible after structural calibration
/// parsing has succeeded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RectifiedStereoCompatibilityError {
    NotRectified,
    FocalMismatch {
        left_fx: f32,
        right_fx: f32,
        left_fy: f32,
        right_fy: f32,
        tolerance_px: f32,
    },
    PrincipalPointMismatch {
        left_cx: f32,
        right_cx: f32,
        left_cy: f32,
        right_cy: f32,
        tolerance_px: f32,
    },
}

impl std::fmt::Display for RectifiedStereoCompatibilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotRectified => f.write_str("calibration is not marked rectified"),
            Self::FocalMismatch {
                left_fx,
                right_fx,
                left_fy,
                right_fy,
                tolerance_px,
            } => write!(
                f,
                "rectified focal lengths differ by more than {tolerance_px} px: left_fx={left_fx}, right_fx={right_fx}, left_fy={left_fy}, right_fy={right_fy}"
            ),
            Self::PrincipalPointMismatch {
                left_cx,
                right_cx,
                left_cy,
                right_cy,
                tolerance_px,
            } => write!(
                f,
                "principal points differ by more than {tolerance_px} px: left_cx={left_cx}, right_cx={right_cx}, left_cy={left_cy}, right_cy={right_cy}"
            ),
        }
    }
}

impl std::error::Error for RectifiedStereoCompatibilityError {}

impl From<RectifiedStereoCompatibilityError> for RectifiedStereoError {
    fn from(source: RectifiedStereoCompatibilityError) -> Self {
        match source {
            RectifiedStereoCompatibilityError::NotRectified => Self::NotRectified,
            RectifiedStereoCompatibilityError::FocalMismatch {
                left_fx,
                right_fx,
                left_fy,
                right_fy,
                tolerance_px,
            } => Self::FocalMismatch {
                left_fx,
                right_fx,
                left_fy,
                right_fy,
                tolerance_px,
            },
            RectifiedStereoCompatibilityError::PrincipalPointMismatch {
                left_cx,
                right_cx,
                left_cy,
                right_cy,
                tolerance_px,
            } => Self::PrincipalPointMismatch {
                left_cx,
                right_cx,
                left_cy,
                right_cy,
                tolerance_px,
            },
        }
    }
}

impl std::fmt::Display for RectifiedStereoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RectifiedStereoError::NonFiniteBaseline { baseline_m } => {
                write!(f, "baseline must be finite, got {baseline_m} m")
            }
            RectifiedStereoError::NonPositiveBaseline { baseline_m } => {
                write!(f, "baseline must be positive, got {baseline_m} m")
            }
            RectifiedStereoError::InvalidDimensions { camera, source } => {
                write!(f, "invalid {camera} camera dimensions: {source}")
            }
            RectifiedStereoError::InvalidIntrinsics { camera, source } => {
                write!(f, "invalid {camera} camera intrinsics: {source}")
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
            RectifiedStereoError::NotRectified => {
                write!(f, "calibration is not marked rectified")
            }
            RectifiedStereoError::FocalMismatch {
                left_fx,
                right_fx,
                left_fy,
                right_fy,
                tolerance_px,
            } => write!(
                f,
                "rectified focal lengths differ by more than {tolerance_px} px: left_fx={left_fx}, right_fx={right_fx}, left_fy={left_fy}, right_fy={right_fy}"
            ),
            RectifiedStereoError::PrincipalPointMismatch {
                left_cx,
                right_cx,
                left_cy,
                right_cy,
                tolerance_px,
            } => {
                write!(
                    f,
                    "principal points differ by more than {tolerance_px} px: left_cx={left_cx}, right_cx={right_cx}, left_cy={left_cy}, right_cy={right_cy}"
                )
            }
        }
    }
}

impl std::error::Error for RectifiedStereoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidDimensions { source, .. } => Some(source),
            Self::InvalidIntrinsics { source, .. } => Some(source),
            Self::NonFiniteBaseline { .. }
            | Self::NonPositiveBaseline { .. }
            | Self::DimensionMismatch { .. }
            | Self::NotRectified
            | Self::FocalMismatch { .. }
            | Self::PrincipalPointMismatch { .. } => None,
        }
    }
}

/// Compare an f32-derived absolute difference to an f32 tolerance without
/// rounding away a low limb before the comparison.
fn exact_difference_exceeds_tolerance(left: f32, right: f32, tolerance: f32) -> bool {
    debug_assert!(left.is_finite());
    debug_assert!(right.is_finite());
    debug_assert!(tolerance.is_finite() && tolerance >= 0.0);

    let left = f64::from(left);
    let right = f64::from(right);
    let (upper, lower) = if left >= right {
        (left, right)
    } else {
        (right, left)
    };
    math::expansion_sum([upper, -lower, -f64::from(tolerance)]) > 0.0
}

impl RectifiedStereo {
    pub fn from_stereo_calibration(
        calibration: &StereoCalibration,
    ) -> Result<Self, RectifiedStereoCompatibilityError> {
        Self::from_stereo_calibration_with_config(calibration, RectifiedStereoConfig::default())
    }

    /// Apply rectification policy to an already-typed stereo calibration.
    pub fn from_stereo_calibration_with_config(
        calibration: &StereoCalibration,
        config: RectifiedStereoConfig,
    ) -> Result<Self, RectifiedStereoCompatibilityError> {
        let left = calibration.left();
        let right = calibration.right();

        if !calibration.is_rectified() {
            return Err(RectifiedStereoCompatibilityError::NotRectified);
        }

        if exact_difference_exceeds_tolerance(left.fx(), right.fx(), config.max_focal_delta_px)
            || exact_difference_exceeds_tolerance(left.fy(), right.fy(), config.max_focal_delta_px)
        {
            return Err(RectifiedStereoCompatibilityError::FocalMismatch {
                left_fx: left.fx(),
                right_fx: right.fx(),
                left_fy: left.fy(),
                right_fy: right.fy(),
                tolerance_px: config.max_focal_delta_px,
            });
        }

        if exact_difference_exceeds_tolerance(left.cx(), right.cx(), config.max_principal_delta_px)
            || exact_difference_exceeds_tolerance(
                left.cy(),
                right.cy(),
                config.max_principal_delta_px,
            )
        {
            return Err(RectifiedStereoCompatibilityError::PrincipalPointMismatch {
                left_cx: left.cx(),
                right_cx: right.cx(),
                left_cy: left.cy(),
                right_cy: right.cy(),
                tolerance_px: config.max_principal_delta_px,
            });
        }

        let baseline_m = calibration.baseline_m();
        let arithmetic = RectifiedStereoArithmetic::new(left, right, baseline_m);

        Ok(Self {
            left,
            right,
            dimensions: calibration.dimensions(),
            baseline_m,
            arithmetic,
        })
    }

    /// Parsed left-camera projection coefficients in pixels.
    pub fn left(&self) -> PinholeIntrinsics {
        self.left
    }

    /// Parsed right-camera projection coefficients in pixels.
    pub fn right(&self) -> PinholeIntrinsics {
        self.right
    }

    /// Positive stereo baseline in metres.
    pub fn baseline_m(&self) -> f32 {
        self.baseline_m
    }

    pub fn width(&self) -> u32 {
        self.dimensions.width()
    }

    pub fn height(&self) -> u32 {
        self.dimensions.height()
    }

    /// Shared, nonzero left/right image dimensions.
    pub fn dimensions(&self) -> FrameDimensions {
        self.dimensions
    }

    /// Exact calibration equality used when one tracker must replay historical
    /// frames and then continue with a live camera. Approximate rectification
    /// compatibility is insufficient here: accepting different projection
    /// coefficients or baseline would silently change the reconstructed map.
    #[cfg(any(feature = "nano-agent", test))]
    pub(crate) fn exactly_matches_calibration(&self, calibration: &StereoCalibration) -> bool {
        fn intrinsics_match(left: PinholeIntrinsics, right: PinholeIntrinsics) -> bool {
            left.fx().to_bits() == right.fx().to_bits()
                && left.fy().to_bits() == right.fy().to_bits()
                && left.cx().to_bits() == right.cx().to_bits()
                && left.cy().to_bits() == right.cy().to_bits()
        }

        calibration.is_rectified()
            && self.dimensions == calibration.dimensions()
            && self.baseline_m.to_bits() == calibration.baseline_m().to_bits()
            && intrinsics_match(self.left, calibration.left())
            && intrinsics_match(self.right, calibration.right())
    }

    /// Bit-exact equality for immutable live-calibration admission.
    ///
    /// Both values already represent structurally valid rectified stereo
    /// models. Exact matching prevents a reconnect or launch artifact from
    /// silently changing the projection used by an existing map.
    pub fn exactly_matches(&self, other: &Self) -> bool {
        fn intrinsics(left: PinholeIntrinsics, right: PinholeIntrinsics) -> bool {
            left.fx().to_bits() == right.fx().to_bits()
                && left.fy().to_bits() == right.fy().to_bits()
                && left.cx().to_bits() == right.cx().to_bits()
                && left.cy().to_bits() == right.cy().to_bits()
        }

        self.dimensions == other.dimensions
            && self.baseline_m.to_bits() == other.baseline_m.to_bits()
            && intrinsics(self.left, other.left)
            && intrinsics(self.right, other.right)
    }
}

impl RectifiedStereoArithmetic {
    fn new(left: PinholeIntrinsics, right: PinholeIntrinsics, baseline_m: f32) -> Self {
        let left = CameraIntrinsics64::from(left);
        let right = CameraIntrinsics64::from(right);
        let baseline_m = f64::from(baseline_m);

        Self {
            left,
            right,
            baseline_m,
        }
    }
}

impl From<PinholeIntrinsics> for CameraIntrinsics64 {
    fn from(intrinsics: PinholeIntrinsics) -> Self {
        Self {
            fx: f64::from(intrinsics.fx()),
            fy: f64::from(intrinsics.fy()),
            cx: f64::from(intrinsics.cx()),
            cy: f64::from(intrinsics.cy()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TriangulationConfig {
    min_disparity_px: f32,
    max_depth_m: Option<f32>,
    max_vertical_disparity_px: f32,
}

const DEFAULT_MAX_VERTICAL_DISPARITY_PX: f32 = 1.0;

impl Default for TriangulationConfig {
    fn default() -> Self {
        Self {
            min_disparity_px: 1.0,
            max_depth_m: None,
            max_vertical_disparity_px: DEFAULT_MAX_VERTICAL_DISPARITY_PX,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TriangulationConfigError {
    InvalidMinDisparity { value: f32 },
    InvalidMaxDepth { value: f32 },
    InvalidMaxVerticalDisparity { value: f32 },
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
            Self::InvalidMaxVerticalDisparity { value } => write!(
                f,
                "maximum vertical disparity must be finite and nonnegative pixels, got {value}"
            ),
        }
    }
}

impl std::error::Error for TriangulationConfigError {}

impl TriangulationConfig {
    pub fn try_new(
        min_disparity_px: f32,
        max_depth_m: Option<f32>,
    ) -> Result<Self, TriangulationConfigError> {
        Self::try_new_with_vertical_disparity(
            min_disparity_px,
            max_depth_m,
            DEFAULT_MAX_VERTICAL_DISPARITY_PX,
        )
    }

    pub fn try_new_with_vertical_disparity(
        min_disparity_px: f32,
        max_depth_m: Option<f32>,
        max_vertical_disparity_px: f32,
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
        if !max_vertical_disparity_px.is_finite() || max_vertical_disparity_px < 0.0 {
            return Err(TriangulationConfigError::InvalidMaxVerticalDisparity {
                value: max_vertical_disparity_px,
            });
        }
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

    pub fn max_vertical_disparity_px(self) -> f32 {
        self.max_vertical_disparity_px
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TriangulationStats {
    pub candidate_matches: usize,
    pub kept: usize,
    pub dropped_disparity: usize,
    pub dropped_epipolar: usize,
    pub dropped_depth: usize,
    /// Geometrically admissible matches whose camera point cannot be
    /// represented as finite f32 coordinates with strictly positive depth.
    pub dropped_unrepresentable: usize,
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
                    "triangulation produced no landmarks (candidates={}, dropped_disparity={}, dropped_epipolar={}, dropped_depth={}, dropped_unrepresentable={})",
                    stats.candidate_matches,
                    stats.dropped_disparity,
                    stats.dropped_epipolar,
                    stats.dropped_depth,
                    stats.dropped_unrepresentable
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
        }
    }
}

impl std::error::Error for KeyframeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidLandmark { cause, .. } => Some(cause),
            Self::Empty
            | Self::LenMismatch { .. }
            | Self::LandmarkIndexOutOfBounds { .. }
            | Self::DuplicateLandmarkIndex { .. }
            | Self::SensorMismatch { .. } => None,
        }
    }
}

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
        Self::validate_shape_and_sensor(&detections, &landmarks, &landmark_indices)?;
        for (index, &landmark) in landmarks.iter().enumerate() {
            validate_camera_landmark(landmark)
                .map_err(|cause| KeyframeError::InvalidLandmark { index, cause })?;
        }
        Self::build(detections, landmarks, landmark_indices)
    }

    fn from_arc_with_validated_landmarks(
        detections: Arc<Detections>,
        landmarks: Vec<crate::CameraPoint3>,
        landmark_indices: Vec<usize>,
    ) -> Result<Self, KeyframeError> {
        Self::validate_shape_and_sensor(&detections, &landmarks, &landmark_indices)?;
        Self::build(detections, landmarks, landmark_indices)
    }

    fn validate_shape_and_sensor(
        detections: &Detections,
        landmarks: &[crate::CameraPoint3],
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
        landmarks: Vec<crate::CameraPoint3>,
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

fn validate_camera_landmark(point: crate::CameraPoint3) -> Result<(), KeyframeLandmarkError> {
    validate_camera_coordinates(point.x, point.y, point.z)
}

fn validate_camera_coordinates(x: f32, y: f32, z: f32) -> Result<(), KeyframeLandmarkError> {
    if !x.is_finite() {
        return Err(KeyframeLandmarkError::NonFiniteX);
    }
    if !y.is_finite() {
        return Err(KeyframeLandmarkError::NonFiniteY);
    }
    if !z.is_finite() {
        return Err(KeyframeLandmarkError::NonFiniteDepth);
    }
    if z <= 0.0 {
        return Err(KeyframeLandmarkError::NonPositiveDepth);
    }
    Ok(())
}

fn narrow_camera_landmark(
    x: f64,
    y: f64,
    z: f64,
) -> Result<crate::CameraPoint3, KeyframeLandmarkError> {
    let x = x as f32;
    let y = y as f32;
    let z = z as f32;
    validate_camera_coordinates(x, y, z)?;
    Ok(crate::CameraPoint3::new(x, y, z))
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

    #[cfg(feature = "nano-agent")]
    pub(crate) fn exactly_matches_calibration(&self, calibration: &StereoCalibration) -> bool {
        self.stereo.exactly_matches_calibration(calibration)
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

        let expected = self.stereo.dimensions();
        let left_dimensions = left.dimensions();
        let right_dimensions = right.dimensions();
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

        let arithmetic = &self.stereo.arithmetic;
        let min_disparity_px = f64::from(self.config.min_disparity_px);
        let max_vertical_disparity_px = f64::from(self.config.max_vertical_disparity_px);
        let max_depth_m = self.config.max_depth_m.map(f64::from);

        let mut landmarks = Vec::new();
        let mut landmark_indices = Vec::new();

        for &(li, ri) in matches.indices() {
            let left_kp = left.keypoints()[li];
            let right_kp = right.keypoints()[ri];

            let left_x = (f64::from(left_kp.x) - arithmetic.left.cx) / arithmetic.left.fx;
            let right_x = (f64::from(right_kp.x) - arithmetic.right.cx) / arithmetic.right.fx;
            let normalized_disparity = left_x - right_x;
            let disparity_left_px = normalized_disparity * arithmetic.left.fx;
            if !left_x.is_finite()
                || !right_x.is_finite()
                || !normalized_disparity.is_finite()
                || !disparity_left_px.is_finite()
            {
                stats.dropped_unrepresentable += 1;
                continue;
            }
            if disparity_left_px <= min_disparity_px {
                stats.dropped_disparity += 1;
                continue;
            }

            let left_y = (f64::from(left_kp.y) - arithmetic.left.cy) / arithmetic.left.fy;
            let right_y = (f64::from(right_kp.y) - arithmetic.right.cy) / arithmetic.right.fy;
            let vertical_disparity_left_px = (left_y - right_y).abs() * arithmetic.left.fy;
            if !left_y.is_finite()
                || !right_y.is_finite()
                || !vertical_disparity_left_px.is_finite()
            {
                stats.dropped_unrepresentable += 1;
                continue;
            }
            if vertical_disparity_left_px > max_vertical_disparity_px {
                stats.dropped_epipolar += 1;
                continue;
            }

            let z = arithmetic.baseline_m / normalized_disparity;
            if !z.is_finite() || z <= 0.0 {
                stats.dropped_unrepresentable += 1;
                continue;
            }
            if let Some(max_depth) = max_depth_m
                && z > max_depth
            {
                stats.dropped_depth += 1;
                continue;
            }

            let x = left_x * z;
            let y = left_y * z;

            let Ok(landmark) = narrow_camera_landmark(x, y, z) else {
                stats.dropped_unrepresentable += 1;
                continue;
            };

            landmarks.push(landmark);
            landmark_indices.push(li);
            stats.kept += 1;
        }

        if landmarks.is_empty() {
            return Err(TriangulationError::NoLandmarks { stats });
        }

        // Every point entered `landmarks` only after `narrow_camera_landmark`
        // established the same invariant enforced by the public constructor.
        let keyframe =
            Keyframe::from_arc_with_validated_landmarks(left, landmarks, landmark_indices)
                .map_err(TriangulationError::InvalidKeyframe)?;

        Ok(TriangulationResult { keyframe, stats })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CameraPoint3;
    use crate::dataset::Calibration;
    use crate::test_helpers::{
        make_camera_intrinsics, make_detections, make_pinhole_intrinsics, make_rectified_stereo,
        rectified_stereo_keypoints_from_points,
    };
    use crate::{FrameId, Keypoint, MatchError, Matches};

    fn assert_stats_accounting(stats: TriangulationStats) {
        assert_eq!(
            stats.kept
                + stats.dropped_disparity
                + stats.dropped_epipolar
                + stats.dropped_depth
                + stats.dropped_unrepresentable,
            stats.candidate_matches
        );
    }

    fn assert_landmarks_are_finite_and_in_front(keyframe: &Keyframe) {
        for point in keyframe.landmarks() {
            assert!(point.x.is_finite(), "nonfinite landmark x: {point:?}");
            assert!(point.y.is_finite(), "nonfinite landmark y: {point:?}");
            assert!(point.z.is_finite(), "nonfinite landmark z: {point:?}");
            assert!(point.z > 0.0, "nonpositive landmark depth: {point:?}");
        }
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
    fn rectified_stereo_preserves_both_camera_bits_and_exact_widening() {
        let left =
            make_camera_intrinsics(5, 7, f32::from_bits(1), f32::MAX, -0.0, f32::from_bits(1));
        let right =
            make_camera_intrinsics(5, 7, f32::from_bits(2), f32::MAX, 0.0, -f32::from_bits(1));
        let baseline_m = f32::from_bits(1);
        let calibration = Calibration {
            left: left.clone(),
            right: right.clone(),
            baseline_m,
            rectified: true,
        };
        let stereo = RectifiedStereo::from_calibration(&calibration).expect("rectified stereo");

        for (parsed, raw) in [(stereo.left(), left), (stereo.right(), right)] {
            assert_eq!(parsed.fx().to_bits(), raw.fx.to_bits());
            assert_eq!(parsed.fy().to_bits(), raw.fy.to_bits());
            assert_eq!(parsed.cx().to_bits(), raw.cx.to_bits());
            assert_eq!(parsed.cy().to_bits(), raw.cy.to_bits());
        }
        assert_eq!(stereo.dimensions(), FrameDimensions::new(5, 7));
        assert_eq!(stereo.baseline_m().to_bits(), baseline_m.to_bits());
        for (parsed, raw) in [
            (&stereo.arithmetic.left, &calibration.left),
            (&stereo.arithmetic.right, &calibration.right),
        ] {
            assert_eq!(parsed.fx.to_bits(), f64::from(raw.fx).to_bits());
            assert_eq!(parsed.fy.to_bits(), f64::from(raw.fy).to_bits());
            assert_eq!(parsed.cx.to_bits(), f64::from(raw.cx).to_bits());
            assert_eq!(parsed.cy.to_bits(), f64::from(raw.cy).to_bits());
        }
        assert_eq!(
            stereo.arithmetic.baseline_m.to_bits(),
            f64::from(baseline_m).to_bits()
        );
    }

    #[test]
    fn replay_calibration_match_is_bit_exact_and_requires_rectification() {
        let dimensions = FrameDimensions::try_new(640, 480).expect("test dimensions");
        let left = PinholeIntrinsics::try_new(400.0, 401.0, 320.0, 240.0).expect("left intrinsics");
        let right =
            PinholeIntrinsics::try_new(400.0, 401.0, 320.0, 240.0).expect("right intrinsics");
        let baseline_m = 0.075_f32;
        let calibration = StereoCalibration::try_new(left, right, dimensions, baseline_m, true)
            .expect("test calibration");
        let stereo =
            RectifiedStereo::from_stereo_calibration(&calibration).expect("rectified stereo");
        assert!(stereo.exactly_matches_calibration(&calibration));

        let adjacent_baseline = f32::from_bits(
            baseline_m
                .to_bits()
                .checked_add(1)
                .expect("finite adjacent bit"),
        );
        let changed_baseline =
            StereoCalibration::try_new(left, right, dimensions, adjacent_baseline, true)
                .expect("adjacent baseline");
        assert!(!stereo.exactly_matches_calibration(&changed_baseline));

        let unrectified = StereoCalibration::try_new(left, right, dimensions, baseline_m, false)
            .expect("structurally valid unrectified calibration");
        assert!(!stereo.exactly_matches_calibration(&unrectified));
    }

    #[test]
    fn rectified_stereo_rejects_nonfinite_and_incompatible_intrinsics() {
        let left = make_camera_intrinsics(640, 480, 400.0, 400.0, 320.0, 240.0);
        let mut invalid_right = left.clone();
        invalid_right.fx = 0.0;
        let mut calibration = Calibration {
            left,
            right: invalid_right,
            baseline_m: f32::NAN,
            rectified: true,
        };
        assert!(matches!(
            RectifiedStereo::from_calibration(&calibration),
            Err(RectifiedStereoError::NonFiniteBaseline { baseline_m })
                if baseline_m.is_nan()
        ));

        calibration.baseline_m = 0.0;
        assert!(matches!(
            RectifiedStereo::from_calibration(&calibration),
            Err(RectifiedStereoError::NonPositiveBaseline { baseline_m: 0.0 })
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
            Err(RectifiedStereoError::InvalidIntrinsics {
                camera: StereoCameraSide::Right,
                source: IntrinsicsError::NonFinite { cx, .. },
            }) if cx.is_nan()
        ));

        calibration.right.cx = calibration.left.cx;
        calibration.right.width = 0;
        assert!(matches!(
            RectifiedStereo::from_calibration(&calibration),
            Err(RectifiedStereoError::InvalidDimensions {
                camera: StereoCameraSide::Right,
                source: FrameDimensionsError::Zero {
                    width: 0,
                    height: 480,
                },
            })
        ));

        calibration.right.width = calibration.left.width;
        calibration.right.fx = 0.0;
        assert!(matches!(
            RectifiedStereo::from_calibration(&calibration),
            Err(RectifiedStereoError::InvalidIntrinsics {
                camera: StereoCameraSide::Right,
                source: IntrinsicsError::NonPositiveFocal { fx: 0.0, .. },
            })
        ));

        calibration.right.fx = calibration.left.fx;
        calibration.right.width = calibration.left.width / 2;
        assert!(matches!(
            RectifiedStereo::from_calibration(&calibration),
            Err(RectifiedStereoError::DimensionMismatch { left, right })
                if left == FrameDimensions::new(640, 480)
                    && right == FrameDimensions::new(320, 480)
        ));
    }

    #[test]
    fn rectified_stereo_exposes_typed_boundary_error_sources() {
        let left = make_camera_intrinsics(640, 480, 400.0, 400.0, 320.0, 240.0);
        let mut calibration = Calibration {
            left: left.clone(),
            right: left,
            baseline_m: 0.075,
            rectified: true,
        };

        calibration.right.width = 0;
        let dimensions_error =
            RectifiedStereo::from_calibration(&calibration).expect_err("invalid dimensions");
        assert!(matches!(
            std::error::Error::source(&dimensions_error)
                .and_then(|source| source.downcast_ref::<FrameDimensionsError>()),
            Some(FrameDimensionsError::Zero {
                width: 0,
                height: 480,
            })
        ));

        calibration.right.width = calibration.left.width;
        calibration.right.fx = f32::NAN;
        let intrinsics_error =
            RectifiedStereo::from_calibration(&calibration).expect_err("invalid intrinsics");
        assert!(matches!(
            std::error::Error::source(&intrinsics_error)
                .and_then(|source| source.downcast_ref::<IntrinsicsError>()),
            Some(IntrinsicsError::NonFinite { fx, .. }) if fx.is_nan()
        ));
    }

    #[test]
    fn rectified_stereo_reports_extreme_principal_mismatch_with_raw_evidence() {
        let left = make_camera_intrinsics(640, 480, 400.0, 400.0, f32::MAX, 240.0);
        let mut right = left.clone();
        right.cx = -f32::MAX;
        let calibration = Calibration {
            left: left.clone(),
            right,
            baseline_m: 0.075,
            rectified: true,
        };
        assert!(matches!(
            RectifiedStereo::from_calibration(&calibration),
            Err(RectifiedStereoError::PrincipalPointMismatch {
                left_cx,
                right_cx,
                left_cy: 240.0,
                right_cy: 240.0,
                tolerance_px,
            }) if left_cx.to_bits() == f32::MAX.to_bits()
                && right_cx.to_bits() == (-f32::MAX).to_bits()
                && tolerance_px.to_bits() == RectifiedStereoConfig::default()
                    .max_principal_delta_px()
                    .to_bits()
        ));
    }

    #[test]
    fn rectified_stereo_compares_exact_principal_delta_to_tolerance() {
        let left = make_camera_intrinsics(640, 480, 400.0, 400.0, 1.0, 240.0);
        let mut right = left.clone();
        right.cx = -f32::from_bits(1); // -2^-149
        let rounded_f64_delta = (f64::from(left.cx) - f64::from(right.cx)).abs();
        assert_eq!(rounded_f64_delta, 1.0);
        let calibration = Calibration {
            left,
            right,
            baseline_m: 0.075,
            rectified: true,
        };
        let config = RectifiedStereoConfig::try_new(1.0, 0.0).expect("config");

        assert!(matches!(
            RectifiedStereo::from_calibration_with_config(&calibration, config),
            Err(RectifiedStereoError::PrincipalPointMismatch {
                left_cx: 1.0,
                right_cx,
                tolerance_px: 1.0,
                ..
            }) if right_cx.to_bits() == (-f32::from_bits(1)).to_bits()
        ));
    }

    #[test]
    fn stereo_and_triangulation_configs_reject_invalid_scalars() {
        for value in [-1.0, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                RectifiedStereoConfig::try_new(value, 1e-3),
                Err(RectifiedStereoConfigError::InvalidTolerance { .. })
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
        for value in [-1.0, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                TriangulationConfig::try_new_with_vertical_disparity(1.0, None, value),
                Err(TriangulationConfigError::InvalidMaxVerticalDisparity { .. })
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
        assert_landmarks_are_finite_and_in_front(&keyframe);
        assert_eq!(keyframe.landmarks().len(), points.len());
        for (landmark, &det_idx) in keyframe.landmarks().iter().zip(keyframe.landmark_indices()) {
            let expected = points[det_idx];
            assert!((landmark.x - expected.x).abs() < 1e-4);
            assert!((landmark.y - expected.y).abs() < 1e-4);
            assert!((landmark.z - expected.z).abs() < 1e-4);
        }
    }

    #[test]
    fn triangulation_uses_both_cameras_calibrated_coordinates() {
        let calibration = Calibration {
            left: make_camera_intrinsics(640, 480, 400.0, 400.0, 320.0, 240.0),
            right: make_camera_intrinsics(640, 480, 420.0, 410.0, 300.0, 238.0),
            baseline_m: 0.075,
            rectified: true,
        };
        let stereo = RectifiedStereo::from_calibration_with_config(
            &calibration,
            RectifiedStereoConfig::try_new(20.0, 20.0).expect("tolerance config"),
        )
        .expect("compatible calibrated stereo pair");
        let triangulator = Triangulator::new(stereo, TriangulationConfig::default());

        // X=Y=0, Z=3 m. The unequal right focal length and principal
        // point make raw pixel subtraction produce the wrong depth.
        let left = make_detections(
            SensorId::StereoLeft,
            FrameId::new(12),
            640,
            480,
            vec![Keypoint { x: 320.0, y: 240.0 }],
        )
        .expect("left detections");
        let right = make_detections(
            SensorId::StereoRight,
            FrameId::new(13),
            640,
            480,
            vec![Keypoint { x: 289.5, y: 238.0 }],
        )
        .expect("right detections");
        let matches = Matches::new(left, right, vec![(0, 0)], vec![1.0]).expect("matches");

        let result = triangulator
            .triangulate(&matches)
            .expect("calibrated triangulation");
        let point = result.keyframe.landmarks()[0];
        assert_landmarks_are_finite_and_in_front(&result.keyframe);
        assert!(point.x.abs() < 1e-6, "x={}", point.x);
        assert!(point.y.abs() < 1e-6, "y={}", point.y);
        assert!((point.z - 3.0).abs() < 1e-5, "z={}", point.z);
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
    fn triangulation_widens_tiny_focal_arithmetic_before_computing_landmarks() {
        let tiny_focal = f32::from_bits(1);
        let stereo = make_rectified_stereo(2, 2, tiny_focal, tiny_focal, 0.0, 0.0, 1.0)
            .expect("tiny but positive focal lengths are representable");
        let triangulator = Triangulator::new(
            stereo,
            TriangulationConfig::try_new(0.5, None).expect("triangulation config"),
        );
        let left = make_detections(
            SensorId::StereoLeft,
            FrameId::new(22),
            2,
            2,
            vec![Keypoint { x: 1.0, y: 1.0 }],
        )
        .expect("left");
        let right = make_detections(
            SensorId::StereoRight,
            FrameId::new(23),
            2,
            2,
            vec![Keypoint { x: 0.0, y: 1.0 }],
        )
        .expect("right");
        let matches = Matches::new(left, right, vec![(0, 0)], vec![1.0]).expect("matches");

        let result = triangulator
            .triangulate(&matches)
            .expect("f64 rays avoid the former inf-times-zero NaN");

        assert_stats_accounting(result.stats);
        assert_eq!(result.stats.kept, 1);
        assert_eq!(result.stats.dropped_unrepresentable, 0);
        assert_landmarks_are_finite_and_in_front(&result.keyframe);
        let point = result.keyframe.landmarks()[0];
        assert_eq!(point, CameraPoint3::new(1.0, 1.0, tiny_focal));
    }

    #[test]
    fn triangulation_counts_finite_f64_points_that_do_not_fit_f32() {
        let stereo =
            make_rectified_stereo(3, 2, 1.0, 1.0, 0.0, 0.0, f32::MAX).expect("finite calibration");
        let triangulator = Triangulator::new(
            stereo,
            TriangulationConfig::try_new(0.5, None).expect("triangulation config"),
        );
        let left = make_detections(
            SensorId::StereoLeft,
            FrameId::new(24),
            3,
            2,
            vec![Keypoint { x: 2.0, y: 0.0 }],
        )
        .expect("left");
        let right = make_detections(
            SensorId::StereoRight,
            FrameId::new(25),
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
    fn every_successful_boundary_scale_triangulation_has_valid_landmarks() {
        let focal_lengths = [f32::from_bits(1), f32::MIN_POSITIVE, 1.0, 400.0, f32::MAX];
        let baselines = [f32::from_bits(1), f32::MIN_POSITIVE, 1.0, f32::MAX];
        let mut successful_cases = 0;

        for focal in focal_lengths {
            for baseline in baselines {
                let stereo = make_rectified_stereo(3, 2, focal, focal, 0.0, 0.0, baseline)
                    .expect("finite positive boundary calibration");
                let triangulator = Triangulator::new(
                    stereo,
                    TriangulationConfig::try_new(0.5, None).expect("triangulation config"),
                );
                let left = make_detections(
                    SensorId::StereoLeft,
                    FrameId::new(28),
                    3,
                    2,
                    vec![Keypoint { x: 2.0, y: 1.0 }],
                )
                .expect("left");
                let right = make_detections(
                    SensorId::StereoRight,
                    FrameId::new(29),
                    3,
                    2,
                    vec![Keypoint { x: 1.0, y: 1.0 }],
                )
                .expect("right");
                let matches = Matches::new(left, right, vec![(0, 0)], vec![1.0]).expect("matches");

                match triangulator.triangulate(&matches) {
                    Ok(result) => {
                        successful_cases += 1;
                        assert_stats_accounting(result.stats);
                        assert_landmarks_are_finite_and_in_front(&result.keyframe);
                    }
                    Err(TriangulationError::NoLandmarks { stats }) => {
                        assert_stats_accounting(stats);
                        assert_eq!(stats.dropped_unrepresentable, 1);
                    }
                    Err(error) => panic!("unexpected boundary-scale error: {error:?}"),
                }
            }
        }

        assert!(successful_cases > 0);
    }

    #[test]
    fn keyframe_rejects_invalid_camera_landmarks_with_typed_causes() {
        let cases = [
            (
                CameraPoint3::new(f32::NAN, 0.0, 1.0),
                KeyframeLandmarkError::NonFiniteX,
            ),
            (
                CameraPoint3::new(0.0, f32::INFINITY, 1.0),
                KeyframeLandmarkError::NonFiniteY,
            ),
            (
                CameraPoint3::new(0.0, 0.0, f32::NAN),
                KeyframeLandmarkError::NonFiniteDepth,
            ),
            (
                CameraPoint3::new(0.0, 0.0, f32::INFINITY),
                KeyframeLandmarkError::NonFiniteDepth,
            ),
            (
                CameraPoint3::new(0.0, 0.0, 0.0),
                KeyframeLandmarkError::NonPositiveDepth,
            ),
            (
                CameraPoint3::new(0.0, 0.0, -1.0),
                KeyframeLandmarkError::NonPositiveDepth,
            ),
        ];

        for (landmark, expected_cause) in cases {
            let detections = make_detections(
                SensorId::StereoLeft,
                FrameId::new(26),
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
            FrameId::new(27),
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
                CameraPoint3::new(0.0, 0.0, 1.0),
                CameraPoint3::new(1.0, 0.0, 1.0),
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
