use std::collections::TryReserveError;
use std::num::{NonZeroU8, NonZeroU64, NonZeroUsize};

use crate::dataset::CameraIntrinsics;
use crate::{Keypoint, Point3, math};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PinholeIntrinsics {
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IntrinsicsError {
    NonFinite { field: &'static str, value: f32 },
    NonPositive { field: &'static str, value: f32 },
}

impl std::fmt::Display for IntrinsicsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IntrinsicsError::NonFinite { field, value } => {
                write!(f, "pinhole intrinsic {field} must be finite, got {value}")
            }
            IntrinsicsError::NonPositive { field, value } => {
                write!(f, "pinhole intrinsic {field} must be positive, got {value}")
            }
        }
    }
}

impl std::error::Error for IntrinsicsError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraFrameAxis {
    X,
    Y,
    Z,
}

impl std::fmt::Display for CameraFrameAxis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::X => "x",
            Self::Y => "y",
            Self::Z => "z",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImagePlaneAxis {
    U,
    V,
}

impl std::fmt::Display for ImagePlaneAxis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::U => "u",
            Self::V => "v",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PinholeProjectionError {
    NonFiniteCameraPointMeters { axis: CameraFrameAxis, value: f64 },
    NonFinitePixelCoordinatePx { axis: ImagePlaneAxis, value: f64 },
    PixelCoordinateOutsideF32Domain { axis: ImagePlaneAxis, value: f64 },
}

impl std::fmt::Display for PinholeProjectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteCameraPointMeters { axis, value } => write!(
                f,
                "pinhole projection camera-frame {axis} coordinate must be finite in meters, got {value}"
            ),
            Self::NonFinitePixelCoordinatePx { axis, value } => write!(
                f,
                "pinhole projection image-plane {axis} coordinate must be finite in pixels, got {value}"
            ),
            Self::PixelCoordinateOutsideF32Domain { axis, value } => write!(
                f,
                "pinhole projection image-plane {axis} coordinate is outside the finite f32 pixel domain: {value} px"
            ),
        }
    }
}

impl std::error::Error for PinholeProjectionError {}

impl TryFrom<&CameraIntrinsics> for PinholeIntrinsics {
    type Error = IntrinsicsError;

    fn try_from(value: &CameraIntrinsics) -> Result<Self, Self::Error> {
        Self::try_new(value.fx, value.fy, value.cx, value.cy)
    }
}

impl PinholeIntrinsics {
    pub fn try_new(fx: f32, fy: f32, cx: f32, cy: f32) -> Result<Self, IntrinsicsError> {
        for (field, value) in [("fx", fx), ("fy", fy), ("cx", cx), ("cy", cy)] {
            if !value.is_finite() {
                return Err(IntrinsicsError::NonFinite { field, value });
            }
        }
        for (field, value) in [("fx", fx), ("fy", fy)] {
            if value <= 0.0 {
                return Err(IntrinsicsError::NonPositive { field, value });
            }
        }
        Ok(Self { fx, fy, cx, cy })
    }

    pub(crate) fn from_rectified_stereo(stereo: &crate::RectifiedStereo) -> Self {
        Self {
            fx: stereo.fx(),
            fy: stereo.fy(),
            cx: stereo.cx(),
            cy: stereo.cy(),
        }
    }

    pub fn fx(&self) -> f32 {
        self.fx
    }

    pub fn fy(&self) -> f32 {
        self.fy
    }

    pub fn cx(&self) -> f32 {
        self.cx
    }

    pub fn cy(&self) -> f32 {
        self.cy
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Observation {
    world: Point3,
    pixel: Keypoint,
    bearing: [f32; 3],
}

impl Observation {
    pub fn world(&self) -> Point3 {
        self.world
    }

    pub fn pixel(&self) -> Keypoint {
        self.pixel
    }

    pub fn bearing(&self) -> [f32; 3] {
        self.bearing
    }

    pub fn try_new(
        world: Point3,
        pixel: Keypoint,
        intrinsics: PinholeIntrinsics,
    ) -> Result<Self, PnpError> {
        for (field, value) in [
            ("world.x", world.x),
            ("world.y", world.y),
            ("world.z", world.z),
            ("pixel.x", pixel.x),
            ("pixel.y", pixel.y),
        ] {
            if !value.is_finite() {
                return Err(PnpError::NonFiniteObservation { field, value });
            }
        }
        let bearing = normalize_bearing(pixel, intrinsics)?;
        Ok(Self {
            world,
            pixel,
            bearing,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Pose {
    rotation: [[f32; 3]; 3],
    translation: [f32; 3],
}

impl Pose {
    pub fn identity() -> Self {
        Self {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
        }
    }

    pub(crate) fn from_rt(rotation: [[f32; 3]; 3], translation: [f32; 3]) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    pub fn rotation(&self) -> [[f32; 3]; 3] {
        self.rotation
    }

    pub fn translation(&self) -> [f32; 3] {
        self.translation
    }

    pub fn inverse(&self) -> Pose {
        let r_t = math::mat_transpose(self.rotation);
        let t = self.translation;
        let t_inv = [
            -(r_t[0][0] * t[0] + r_t[0][1] * t[1] + r_t[0][2] * t[2]),
            -(r_t[1][0] * t[0] + r_t[1][1] * t[1] + r_t[1][2] * t[2]),
            -(r_t[2][0] * t[0] + r_t[2][1] * t[1] + r_t[2][2] * t[2]),
        ];
        Pose {
            rotation: r_t,
            translation: t_inv,
        }
    }

    /// Compose transforms as `self ∘ other`.
    pub fn compose(self, other: Pose) -> Pose {
        let r = math::mat_mul(self.rotation, other.rotation);
        let t = math::mat_mul_vec(self.rotation, other.translation);
        Pose {
            rotation: r,
            translation: [
                t[0] + self.translation[0],
                t[1] + self.translation[1],
                t[2] + self.translation[2],
            ],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum PinholeProjection {
    Projected { u_px: f32, v_px: f32 },
    NonPositiveCameraDepth { depth_m: f64 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum PinholeProjectionF64 {
    Projected { u_px: f64, v_px: f64 },
    NonPositiveCameraDepth { depth_m: f64 },
}

fn project_world_point_f64_px(
    pose_world_to_camera: Pose,
    point_world_m: Point3,
    intrinsics: PinholeIntrinsics,
) -> Result<PinholeProjectionF64, PinholeProjectionError> {
    let rotation = pose_world_to_camera.rotation();
    let translation = pose_world_to_camera.translation();
    let point_world_m = [
        f64::from(point_world_m.x),
        f64::from(point_world_m.y),
        f64::from(point_world_m.z),
    ];
    let mut point_camera_m = [0.0_f64; 3];
    for axis in 0..3 {
        point_camera_m[axis] = f64::from(rotation[axis][0]).mul_add(
            point_world_m[0],
            f64::from(rotation[axis][1]).mul_add(
                point_world_m[1],
                f64::from(rotation[axis][2])
                    .mul_add(point_world_m[2], f64::from(translation[axis])),
            ),
        );
    }
    for (axis, value) in [
        (CameraFrameAxis::X, point_camera_m[0]),
        (CameraFrameAxis::Y, point_camera_m[1]),
        (CameraFrameAxis::Z, point_camera_m[2]),
    ] {
        if !value.is_finite() {
            return Err(PinholeProjectionError::NonFiniteCameraPointMeters { axis, value });
        }
    }
    let depth_m = point_camera_m[2];
    if depth_m <= 0.0 {
        return Ok(PinholeProjectionF64::NonPositiveCameraDepth { depth_m });
    }
    let u_px =
        f64::from(intrinsics.fx()).mul_add(point_camera_m[0] / depth_m, f64::from(intrinsics.cx()));
    let v_px =
        f64::from(intrinsics.fy()).mul_add(point_camera_m[1] / depth_m, f64::from(intrinsics.cy()));
    for (axis, value) in [(ImagePlaneAxis::U, u_px), (ImagePlaneAxis::V, v_px)] {
        if !value.is_finite() {
            return Err(PinholeProjectionError::NonFinitePixelCoordinatePx { axis, value });
        }
    }
    Ok(PinholeProjectionF64::Projected { u_px, v_px })
}

pub(crate) fn project_world_point_px(
    pose_world_to_camera: Pose,
    point_world_m: Point3,
    intrinsics: PinholeIntrinsics,
) -> Result<PinholeProjection, PinholeProjectionError> {
    let (u_px, v_px) =
        match project_world_point_f64_px(pose_world_to_camera, point_world_m, intrinsics)? {
            PinholeProjectionF64::Projected { u_px, v_px } => (u_px, v_px),
            PinholeProjectionF64::NonPositiveCameraDepth { depth_m } => {
                return Ok(PinholeProjection::NonPositiveCameraDepth { depth_m });
            }
        };
    let narrow_px = |axis, value: f64| {
        if value < -f64::from(f32::MAX) || value > f64::from(f32::MAX) {
            Err(PinholeProjectionError::PixelCoordinateOutsideF32Domain { axis, value })
        } else {
            Ok(value as f32)
        }
    };
    Ok(PinholeProjection::Projected {
        u_px: narrow_px(ImagePlaneAxis::U, u_px)?,
        v_px: narrow_px(ImagePlaneAxis::V, v_px)?,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct RansacConfig {
    max_iterations: NonZeroUsize,
    reprojection_threshold_px: f32,
    min_inliers: usize,
    seed: NonZeroU64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RansacConfigError {
    ZeroMaxIterations,
    ZeroSeed,
    InvalidReprojectionThresholdPx { value: f32 },
    TooFewMinInliers { value: usize, minimum: usize },
}

impl std::fmt::Display for RansacConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RansacConfigError::ZeroMaxIterations => {
                write!(f, "RANSAC max iterations must be greater than zero")
            }
            RansacConfigError::ZeroSeed => write!(f, "RANSAC seed must be nonzero"),
            RansacConfigError::InvalidReprojectionThresholdPx { value } => write!(
                f,
                "RANSAC reprojection threshold must be positive and finite, got {value}"
            ),
            RansacConfigError::TooFewMinInliers { value, minimum } => write!(
                f,
                "RANSAC minimum inliers must be at least {minimum}, got {value}"
            ),
        }
    }
}

impl std::error::Error for RansacConfigError {}

impl RansacConfig {
    pub fn new(
        max_iterations: usize,
        reprojection_threshold_px: f32,
        min_inliers: usize,
        seed: u64,
    ) -> Result<Self, RansacConfigError> {
        let max_iterations =
            NonZeroUsize::new(max_iterations).ok_or(RansacConfigError::ZeroMaxIterations)?;
        let seed = NonZeroU64::new(seed).ok_or(RansacConfigError::ZeroSeed)?;
        if !reprojection_threshold_px.is_finite() || reprojection_threshold_px <= 0.0 {
            return Err(RansacConfigError::InvalidReprojectionThresholdPx {
                value: reprojection_threshold_px,
            });
        }
        if min_inliers < MIN_PNP_POINTS {
            return Err(RansacConfigError::TooFewMinInliers {
                value: min_inliers,
                minimum: MIN_PNP_POINTS,
            });
        }
        Ok(Self {
            max_iterations,
            reprojection_threshold_px,
            min_inliers,
            seed,
        })
    }

    pub fn max_iterations(self) -> usize {
        self.max_iterations.get()
    }

    pub fn reprojection_threshold_px(self) -> f32 {
        self.reprojection_threshold_px
    }

    pub fn min_inliers(self) -> usize {
        self.min_inliers
    }

    pub fn seed(self) -> u64 {
        self.seed.get()
    }

    pub fn try_with_min_inliers(self, min_inliers: usize) -> Result<Self, RansacConfigError> {
        Self::new(
            self.max_iterations(),
            self.reprojection_threshold_px,
            min_inliers,
            self.seed.get(),
        )
    }
}

impl Default for RansacConfig {
    fn default() -> Self {
        Self {
            max_iterations: NonZeroUsize::MIN.saturating_add(199),
            reprojection_threshold_px: 2.0,
            min_inliers: 20,
            seed: NonZeroU64::new(0x5EED_u64).expect("default RANSAC seed is nonzero"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PnpInlierBuffer {
    BestConsensus,
    CandidateScratch,
}

impl std::fmt::Display for PnpInlierBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::BestConsensus => "best-consensus inlier indices",
            Self::CandidateScratch => "candidate inlier scratch indices",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PnpRansacRejectionKind {
    MinimalSample,
    CandidateProjection,
}

impl std::fmt::Display for PnpRansacRejectionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::MinimalSample => "minimal-sample",
            Self::CandidateProjection => "candidate-projection",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PnpP3pBuffer {
    RealPolynomialRoots,
    DistanceRatioRoots,
    PoseCandidates,
}

impl std::fmt::Display for PnpP3pBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::RealPolynomialRoots => "real polynomial roots",
            Self::DistanceRatioRoots => "P3P distance-ratio roots",
            Self::PoseCandidates => "P3P pose candidates",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PnpWorldTriangleSide {
    Point2ToPoint3,
    Point1ToPoint3,
    Point1ToPoint2,
}

impl std::fmt::Display for PnpWorldTriangleSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Point2ToPoint3 => "point 2 to point 3",
            Self::Point1ToPoint3 => "point 1 to point 3",
            Self::Point1ToPoint2 => "point 1 to point 2",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PnpMinimalSampleRejectionReason {
    NonFiniteWorldTriangleSideMeters {
        side: PnpWorldTriangleSide,
        value: f64,
    },
    DegenerateWorldTriangle,
    NonFiniteQuarticCoefficient {
        coefficient_index: usize,
        value: f64,
    },
    DegenerateQuartic,
    RootSolverUnsupportedDegree {
        degree: usize,
        maximum_supported_degree: usize,
    },
    RootSolverBreakdown {
        root_iteration: NonZeroU8,
        root_index: u8,
        denominator_magnitude: f64,
    },
    RootSolverNonFiniteDenominator {
        root_iteration: NonZeroU8,
        root_index: u8,
        real: f64,
        imaginary: f64,
    },
    RootSolverNonFinitePolynomialValue {
        root_iteration: NonZeroU8,
        root_index: u8,
        real: f64,
        imaginary: f64,
    },
    RootSolverDivisionFailed {
        root_iteration: NonZeroU8,
        root_index: u8,
        polynomial_magnitude: f64,
        denominator_magnitude: f64,
    },
    RootSolverNonFiniteUpdatedRoot {
        root_iteration: NonZeroU8,
        root_index: u8,
        updated_real: f64,
        updated_imaginary: f64,
    },
    RootSolverNonFiniteCorrectionMagnitude {
        root_iteration: NonZeroU8,
        root_index: u8,
        correction_real: f64,
        correction_imaginary: f64,
    },
    RootSolverIterationLimit {
        iterations: NonZeroU8,
    },
    NoAdmissibleDistanceRatioRoots,
    NoGeometricallyAdmissiblePoseCandidates,
}

impl std::fmt::Display for PnpMinimalSampleRejectionReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteWorldTriangleSideMeters { side, value } => write!(
                f,
                "sampled world-triangle side {side} became nonfinite during f64 distance evaluation: {value} m"
            ),
            Self::DegenerateWorldTriangle => {
                write!(
                    f,
                    "the sampled world points do not form a nondegenerate triangle"
                )
            }
            Self::NonFiniteQuarticCoefficient {
                coefficient_index,
                value,
            } => write!(
                f,
                "P3P quartic coefficient {coefficient_index} is nonfinite: {value}"
            ),
            Self::DegenerateQuartic => {
                write!(f, "the normalized P3P quartic has no nonconstant term")
            }
            Self::RootSolverUnsupportedDegree {
                degree,
                maximum_supported_degree,
            } => write!(
                f,
                "P3P root solver received degree {degree}, above its fixed maximum of {maximum_supported_degree}"
            ),
            Self::RootSolverBreakdown {
                root_iteration,
                root_index,
                denominator_magnitude,
            } => write!(
                f,
                "P3P root solve broke down at iteration {root_iteration}, root {root_index}: Durand-Kerner denominator magnitude {denominator_magnitude} is at or below {ROOT_DENOMINATOR_TOLERANCE}"
            ),
            Self::RootSolverNonFiniteDenominator {
                root_iteration,
                root_index,
                real,
                imaginary,
            } => write!(
                f,
                "P3P root solve produced a nonfinite Durand-Kerner denominator at iteration {root_iteration}, root {root_index}: ({real}, {imaginary})"
            ),
            Self::RootSolverNonFinitePolynomialValue {
                root_iteration,
                root_index,
                real,
                imaginary,
            } => write!(
                f,
                "P3P root solve produced a nonfinite polynomial value at iteration {root_iteration}, root {root_index}: ({real}, {imaginary})"
            ),
            Self::RootSolverDivisionFailed {
                root_iteration,
                root_index,
                polynomial_magnitude,
                denominator_magnitude,
            } => write!(
                f,
                "P3P root solve could not divide polynomial magnitude {polynomial_magnitude} by denominator magnitude {denominator_magnitude} at iteration {root_iteration}, root {root_index} without leaving the finite f64 domain"
            ),
            Self::RootSolverNonFiniteUpdatedRoot {
                root_iteration,
                root_index,
                updated_real,
                updated_imaginary,
            } => write!(
                f,
                "P3P root solve produced a nonfinite updated root at iteration {root_iteration}, root {root_index}: ({updated_real}, {updated_imaginary})"
            ),
            Self::RootSolverNonFiniteCorrectionMagnitude {
                root_iteration,
                root_index,
                correction_real,
                correction_imaginary,
            } => write!(
                f,
                "P3P root solve correction magnitude left the finite f64 domain at iteration {root_iteration}, root {root_index}: correction=({correction_real}, {correction_imaginary})"
            ),
            Self::RootSolverIterationLimit { iterations } => write!(
                f,
                "P3P root solve reached its {iterations}-iteration limit without convergence"
            ),
            Self::NoAdmissibleDistanceRatioRoots => write!(
                f,
                "the P3P quartic produced no positive, equation-consistent distance-ratio roots"
            ),
            Self::NoGeometricallyAdmissiblePoseCandidates => write!(
                f,
                "the P3P distance-ratio roots produced no geometrically admissible rigid pose"
            ),
        }
    }
}

impl std::error::Error for PnpMinimalSampleRejectionReason {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PnpMinimalSampleRejection {
    ransac_iteration: NonZeroUsize,
    reason: PnpMinimalSampleRejectionReason,
}

impl PnpMinimalSampleRejection {
    pub fn ransac_iteration(self) -> NonZeroUsize {
        self.ransac_iteration
    }

    pub fn reason(self) -> PnpMinimalSampleRejectionReason {
        self.reason
    }
}

impl std::fmt::Display for PnpMinimalSampleRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RANSAC minimal sample was rejected at iteration {}: {}",
            self.ransac_iteration, self.reason
        )
    }
}

impl std::error::Error for PnpMinimalSampleRejection {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.reason)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PnpRansacRejectionSummary<T> {
    count: NonZeroUsize,
    first: T,
}

impl<T: Copy> PnpRansacRejectionSummary<T> {
    fn first(rejection: T) -> Self {
        Self {
            count: NonZeroUsize::MIN,
            first: rejection,
        }
    }

    fn try_record_another(&mut self, kind: PnpRansacRejectionKind) -> Result<(), PnpError> {
        self.count = self
            .count
            .get()
            .checked_add(1)
            .and_then(NonZeroUsize::new)
            .ok_or(PnpError::RansacRejectionCountOverflow { kind })?;
        Ok(())
    }

    pub fn count(self) -> NonZeroUsize {
        self.count
    }

    pub fn first_rejection(self) -> T {
        self.first
    }
}

pub type PnpMinimalSampleRejections = PnpRansacRejectionSummary<PnpMinimalSampleRejection>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PnpCandidateProjectionRejection {
    ransac_iteration: NonZeroUsize,
    observation_index: usize,
    source: PinholeProjectionError,
}

impl PnpCandidateProjectionRejection {
    pub fn ransac_iteration(self) -> NonZeroUsize {
        self.ransac_iteration
    }

    pub fn observation_index(self) -> usize {
        self.observation_index
    }

    pub fn projection_error(self) -> PinholeProjectionError {
        self.source
    }
}

impl std::fmt::Display for PnpCandidateProjectionRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RANSAC candidate projection failed at iteration {} for observation {}: {}",
            self.ransac_iteration, self.observation_index, self.source
        )
    }
}

impl std::error::Error for PnpCandidateProjectionRejection {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

pub type PnpCandidateProjectionRejections =
    PnpRansacRejectionSummary<PnpCandidateProjectionRejection>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PnpRansacRejections {
    inner: PnpRansacRejectionsInner,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum PnpRansacRejectionsInner {
    MinimalSamples(PnpMinimalSampleRejections),
    CandidateProjections(PnpCandidateProjectionRejections),
    Both {
        minimal_samples: PnpMinimalSampleRejections,
        candidate_projections: PnpCandidateProjectionRejections,
    },
}

impl PnpRansacRejections {
    fn try_record_minimal_sample(
        summary: &mut Option<Self>,
        rejection: PnpMinimalSampleRejection,
    ) -> Result<(), PnpError> {
        match summary {
            Some(summary) => match &mut summary.inner {
                PnpRansacRejectionsInner::MinimalSamples(rejections)
                | PnpRansacRejectionsInner::Both {
                    minimal_samples: rejections,
                    ..
                } => rejections.try_record_another(PnpRansacRejectionKind::MinimalSample),
                PnpRansacRejectionsInner::CandidateProjections(candidate_projections) => {
                    summary.inner = PnpRansacRejectionsInner::Both {
                        minimal_samples: PnpMinimalSampleRejections::first(rejection),
                        candidate_projections: *candidate_projections,
                    };
                    Ok(())
                }
            },
            None => {
                *summary = Some(Self {
                    inner: PnpRansacRejectionsInner::MinimalSamples(
                        PnpMinimalSampleRejections::first(rejection),
                    ),
                });
                Ok(())
            }
        }
    }

    fn try_record_candidate_projection(
        summary: &mut Option<Self>,
        rejection: PnpCandidateProjectionRejection,
    ) -> Result<(), PnpError> {
        match summary {
            Some(summary) => match &mut summary.inner {
                PnpRansacRejectionsInner::CandidateProjections(rejections)
                | PnpRansacRejectionsInner::Both {
                    candidate_projections: rejections,
                    ..
                } => rejections.try_record_another(PnpRansacRejectionKind::CandidateProjection),
                PnpRansacRejectionsInner::MinimalSamples(minimal_samples) => {
                    summary.inner = PnpRansacRejectionsInner::Both {
                        minimal_samples: *minimal_samples,
                        candidate_projections: PnpCandidateProjectionRejections::first(rejection),
                    };
                    Ok(())
                }
            },
            None => {
                *summary = Some(Self {
                    inner: PnpRansacRejectionsInner::CandidateProjections(
                        PnpCandidateProjectionRejections::first(rejection),
                    ),
                });
                Ok(())
            }
        }
    }

    pub fn minimal_sample_rejections(self) -> Option<PnpMinimalSampleRejections> {
        match self.inner {
            PnpRansacRejectionsInner::MinimalSamples(rejections)
            | PnpRansacRejectionsInner::Both {
                minimal_samples: rejections,
                ..
            } => Some(rejections),
            PnpRansacRejectionsInner::CandidateProjections(_) => None,
        }
    }

    pub fn candidate_projection_rejections(self) -> Option<PnpCandidateProjectionRejections> {
        match self.inner {
            PnpRansacRejectionsInner::CandidateProjections(rejections)
            | PnpRansacRejectionsInner::Both {
                candidate_projections: rejections,
                ..
            } => Some(rejections),
            PnpRansacRejectionsInner::MinimalSamples(_) => None,
        }
    }

    fn first_rejection_error(&self) -> &(dyn std::error::Error + 'static) {
        match &self.inner {
            PnpRansacRejectionsInner::MinimalSamples(samples) => &samples.first,
            PnpRansacRejectionsInner::CandidateProjections(projections) => &projections.first,
            PnpRansacRejectionsInner::Both {
                minimal_samples,
                candidate_projections,
            } => {
                if minimal_samples.first.ransac_iteration
                    <= candidate_projections.first.ransac_iteration
                {
                    &minimal_samples.first
                } else {
                    &candidate_projections.first
                }
            }
        }
    }
}

impl std::fmt::Display for PnpRansacRejections {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let minimal_samples = self
            .minimal_sample_rejections()
            .map_or(0, |value| value.count.get());
        let candidate_projections = self
            .candidate_projection_rejections()
            .map_or(0, |value| value.count.get());
        write!(
            f,
            "{minimal_samples} rejected minimal samples and {candidate_projections} rejected candidate projections"
        )
    }
}

impl std::error::Error for PnpRansacRejections {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.first_rejection_error())
    }
}

#[derive(Debug)]
pub struct PnpResult {
    pose: Pose,
    inliers: Vec<usize>,
    iterations: NonZeroUsize,
    refinement: PnpRefinementStatus,
    ransac_rejections: Option<PnpRansacRejections>,
}

impl PnpResult {
    pub fn pose(&self) -> Pose {
        self.pose
    }

    pub fn inliers(&self) -> &[usize] {
        &self.inliers
    }

    pub fn iterations(&self) -> NonZeroUsize {
        self.iterations
    }

    pub fn refinement(&self) -> &PnpRefinementStatus {
        &self.refinement
    }

    pub fn candidate_projection_rejections(&self) -> Option<PnpCandidateProjectionRejections> {
        self.ransac_rejections
            .and_then(PnpRansacRejections::candidate_projection_rejections)
    }

    pub fn minimal_sample_rejections(&self) -> Option<PnpMinimalSampleRejections> {
        self.ransac_rejections
            .and_then(PnpRansacRejections::minimal_sample_rejections)
    }

    pub fn ransac_rejections(&self) -> Option<PnpRansacRejections> {
        self.ransac_rejections
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PnpRefinementTermination {
    Converged {
        iterations: NonZeroUsize,
    },
    IterationLimit {
        iterations: NonZeroUsize,
    },
    Stalled {
        iterations: NonZeroUsize,
        current_cost: PnpRefinementCost,
        candidate_cost: PnpRefinementCost,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct PnpRefinementCost(f64);

impl PnpRefinementCost {
    fn try_new(
        iteration: NonZeroUsize,
        stage: PnpRefinementObjectiveStage,
        value: f64,
    ) -> Result<Self, PnpRefinementFallback> {
        if !value.is_finite() || value < 0.0 {
            return Err(PnpRefinementFallback::InvalidObjective {
                iteration,
                stage,
                value,
            });
        }
        Ok(Self(value))
    }

    pub fn value(self) -> f64 {
        self.0
    }
}

impl std::fmt::Display for PnpRefinementCost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PnpRefinementObjectiveStage {
    Current,
    Candidate,
}

impl std::fmt::Display for PnpRefinementObjectiveStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Current => write!(f, "current"),
            Self::Candidate => write!(f, "candidate"),
        }
    }
}

impl PnpRefinementTermination {
    pub fn iterations(self) -> NonZeroUsize {
        match self {
            Self::Converged { iterations }
            | Self::IterationLimit { iterations }
            | Self::Stalled { iterations, .. } => iterations,
        }
    }
}

#[derive(Clone, Debug)]
pub enum PnpRefinementStatus {
    Applied {
        termination: PnpRefinementTermination,
    },
    RetainedRansacPose {
        reason: PnpRefinementFallback,
    },
}

impl PnpRefinementStatus {
    pub fn applied(&self) -> bool {
        matches!(self, Self::Applied { .. })
    }

    pub fn iterations(&self) -> Option<NonZeroUsize> {
        match self {
            Self::Applied { termination } => Some(termination.iterations()),
            Self::RetainedRansacPose { reason } => reason.iterations(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum PnpRefinementFallback {
    EmptyInlierSet,
    InvalidInlierIndex {
        iteration: NonZeroUsize,
        index: usize,
        observation_count: usize,
    },
    NonProjectableInliers {
        iteration: NonZeroUsize,
        count: NonZeroUsize,
    },
    LinearSolve {
        iteration: NonZeroUsize,
        source: crate::LinearSolveError,
    },
    InvalidPoseUpdate {
        iteration: NonZeroUsize,
        source: crate::Se3TangentError,
    },
    CandidateProjection {
        iteration: NonZeroUsize,
        observation_index: usize,
        source: PinholeProjectionError,
    },
    PostRefinementConsensusProjection {
        termination: PnpRefinementTermination,
        observation_index: usize,
        source: PinholeProjectionError,
    },
    Stationary {
        iteration: NonZeroUsize,
    },
    NoImprovement {
        iteration: NonZeroUsize,
        current_cost: PnpRefinementCost,
        candidate_cost: PnpRefinementCost,
    },
    InvalidObjective {
        iteration: NonZeroUsize,
        stage: PnpRefinementObjectiveStage,
        value: f64,
    },
    LostConsensus {
        termination: PnpRefinementTermination,
        candidate_inliers: usize,
        required_inliers: usize,
    },
}

impl PnpRefinementFallback {
    pub fn iterations(&self) -> Option<NonZeroUsize> {
        match self {
            Self::EmptyInlierSet => None,
            Self::InvalidInlierIndex { iteration, .. }
            | Self::NonProjectableInliers { iteration, .. }
            | Self::LinearSolve { iteration, .. }
            | Self::InvalidPoseUpdate { iteration, .. }
            | Self::CandidateProjection { iteration, .. }
            | Self::Stationary { iteration }
            | Self::NoImprovement { iteration, .. }
            | Self::InvalidObjective { iteration, .. } => Some(*iteration),
            Self::PostRefinementConsensusProjection { termination, .. }
            | Self::LostConsensus { termination, .. } => Some(termination.iterations()),
        }
    }
}

impl std::fmt::Display for PnpRefinementFallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInlierSet => write!(f, "PnP refinement received no inliers"),
            Self::InvalidInlierIndex {
                iteration,
                index,
                observation_count,
            } => write!(
                f,
                "PnP refinement inlier index {index} is out of bounds for {observation_count} observations at iteration {iteration}"
            ),
            Self::NonProjectableInliers { iteration, count } => write!(
                f,
                "PnP refinement has {count} nonprojectable inliers at iteration {iteration}"
            ),
            Self::LinearSolve { iteration, source } => write!(
                f,
                "PnP refinement linear solve failed at iteration {iteration}: {source}"
            ),
            Self::InvalidPoseUpdate { iteration, source } => write!(
                f,
                "PnP refinement pose update is invalid at iteration {iteration}: {source}"
            ),
            Self::CandidateProjection {
                iteration,
                observation_index,
                source,
            } => write!(
                f,
                "PnP refinement candidate projection failed at iteration {iteration} for observation {observation_index}: {source}"
            ),
            Self::PostRefinementConsensusProjection {
                termination,
                observation_index,
                source,
            } => write!(
                f,
                "PnP post-refinement consensus projection failed after {termination:?} for observation {observation_index}: {source}"
            ),
            Self::Stationary { iteration } => write!(
                f,
                "PnP refinement was already stationary at iteration {iteration}"
            ),
            Self::NoImprovement {
                iteration,
                current_cost,
                candidate_cost,
            } => write!(
                f,
                "PnP refinement proposal did not improve the fixed-inlier objective at iteration {iteration} (current={current_cost}, candidate={candidate_cost})"
            ),
            Self::InvalidObjective {
                iteration,
                stage,
                value,
            } => write!(
                f,
                "PnP refinement {stage} objective must be finite and nonnegative at iteration {iteration}, got {value}"
            ),
            Self::LostConsensus {
                termination,
                candidate_inliers,
                required_inliers,
            } => write!(
                f,
                "PnP refinement ended with {termination:?} but retained only {candidate_inliers} inliers (required {required_inliers})"
            ),
        }
    }
}

impl std::error::Error for PnpRefinementFallback {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::LinearSolve { source, .. } => Some(source),
            Self::InvalidPoseUpdate { source, .. } => Some(source),
            Self::CandidateProjection { source, .. }
            | Self::PostRefinementConsensusProjection { source, .. } => Some(source),
            Self::EmptyInlierSet
            | Self::InvalidInlierIndex { .. }
            | Self::NonProjectableInliers { .. }
            | Self::Stationary { .. }
            | Self::NoImprovement { .. }
            | Self::InvalidObjective { .. }
            | Self::LostConsensus { .. } => None,
        }
    }
}

#[derive(Debug)]
pub enum PnpError {
    Rejected(PnpRejection),
    NonFiniteObservation {
        field: &'static str,
        value: f32,
    },
    ObservationBearingForwardComponentOutsideF32Domain {
        value: f64,
    },
    InlierBufferAllocation {
        buffer: PnpInlierBuffer,
        observation_count: usize,
        source: TryReserveError,
    },
    P3pBufferCapacityExceeded {
        ransac_iteration: NonZeroUsize,
        buffer: PnpP3pBuffer,
        capacity: usize,
    },
    RansacRejectionCountOverflow {
        kind: PnpRansacRejectionKind,
    },
}

impl std::fmt::Display for PnpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PnpError::Rejected(rejection) => write!(f, "pnp rejected: {rejection}"),
            PnpError::NonFiniteObservation { field, value } => {
                write!(f, "pnp observation {field} must be finite, got {value}")
            }
            PnpError::ObservationBearingForwardComponentOutsideF32Domain { value } => write!(
                f,
                "pnp observation unit bearing has positive forward component {value}, which is not representable in f32"
            ),
            PnpError::InlierBufferAllocation {
                buffer,
                observation_count,
                source,
            } => write!(
                f,
                "failed to allocate {buffer} for {observation_count} PnP observations: {source}"
            ),
            PnpError::P3pBufferCapacityExceeded {
                ransac_iteration,
                buffer,
                capacity,
            } => write!(
                f,
                "P3P exceeded the fixed {buffer} capacity of {capacity} at RANSAC iteration {ransac_iteration}"
            ),
            PnpError::RansacRejectionCountOverflow { kind } => {
                write!(f, "PnP {kind} rejection count exceeded the usize domain")
            }
        }
    }
}

impl std::error::Error for PnpError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Rejected(rejection) => Some(rejection),
            Self::InlierBufferAllocation { source, .. } => Some(source),
            Self::NonFiniteObservation { .. }
            | Self::ObservationBearingForwardComponentOutsideF32Domain { .. }
            | Self::P3pBufferCapacityExceeded { .. }
            | Self::RansacRejectionCountOverflow { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PnpRejection {
    NotEnoughPoints {
        required: usize,
        actual: usize,
    },
    InsufficientObservationsForRequiredInliers {
        required_inliers: usize,
        observations: usize,
    },
    NoUsableRansacCandidate {
        rejections: PnpRansacRejections,
    },
    NoConsensusAfterRansacRejections {
        rejections: PnpRansacRejections,
    },
    NoSolution,
}

impl std::fmt::Display for PnpRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotEnoughPoints { required, actual } => {
                write!(f, "requires at least {required} points, got {actual}")
            }
            Self::InsufficientObservationsForRequiredInliers {
                required_inliers,
                observations,
            } => write!(
                f,
                "requires {required_inliers} inliers but received only {observations} observations"
            ),
            Self::NoUsableRansacCandidate { rejections } => {
                write!(f, "no usable RANSAC candidate after {rejections}")
            }
            Self::NoConsensusAfterRansacRejections { rejections } => {
                write!(f, "no sufficient consensus after {rejections}")
            }
            Self::NoSolution => write!(f, "no valid pose solution"),
        }
    }
}

impl std::error::Error for PnpRejection {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NoUsableRansacCandidate { rejections }
            | Self::NoConsensusAfterRansacRejections { rejections } => Some(rejections),
            Self::NotEnoughPoints { .. }
            | Self::InsufficientObservationsForRequiredInliers { .. }
            | Self::NoSolution => None,
        }
    }
}

impl PnpRejection {
    pub fn ransac_rejections(self) -> Option<PnpRansacRejections> {
        match self {
            Self::NoUsableRansacCandidate { rejections }
            | Self::NoConsensusAfterRansacRejections { rejections } => Some(rejections),
            Self::NotEnoughPoints { .. }
            | Self::InsufficientObservationsForRequiredInliers { .. }
            | Self::NoSolution => None,
        }
    }
}

impl PnpError {
    pub fn rejection(&self) -> Option<PnpRejection> {
        match self {
            Self::Rejected(rejection) => Some(*rejection),
            Self::NonFiniteObservation { .. }
            | Self::ObservationBearingForwardComponentOutsideF32Domain { .. }
            | Self::InlierBufferAllocation { .. }
            | Self::P3pBufferCapacityExceeded { .. }
            | Self::RansacRejectionCountOverflow { .. } => None,
        }
    }
}

impl From<PnpRejection> for PnpError {
    fn from(value: PnpRejection) -> Self {
        Self::Rejected(value)
    }
}

pub fn solve_pnp_ransac(
    observations: &[Observation],
    intrinsics: PinholeIntrinsics,
    config: RansacConfig,
) -> Result<PnpResult, PnpError> {
    if observations.len() < MIN_PNP_POINTS {
        return Err(PnpRejection::NotEnoughPoints {
            required: MIN_PNP_POINTS,
            actual: observations.len(),
        }
        .into());
    }
    if observations.len() < config.min_inliers() {
        return Err(PnpRejection::InsufficientObservationsForRequiredInliers {
            required_inliers: config.min_inliers(),
            observations: observations.len(),
        }
        .into());
    }

    let mut rng = XorShift64::new(config.seed);
    let mut best_pose = None;
    let mut best_inliers = try_inlier_buffer(PnpInlierBuffer::BestConsensus, observations.len())?;
    let mut candidate_inliers =
        try_inlier_buffer(PnpInlierBuffer::CandidateScratch, observations.len())?;
    let mut p3p_candidates = FixedBuffer::new();
    let mut ransac_rejections: Option<PnpRansacRejections> = None;
    let mut saw_generated_candidate = false;
    let mut saw_projection_complete_candidate = false;

    let mut ransac_iteration = NonZeroUsize::MIN;
    let total = observations.len();
    let mut target_iterations = config.max_iterations();

    let threshold_px = f64::from(config.reprojection_threshold_px());
    let threshold_sq_px2 = threshold_px * threshold_px;
    loop {
        let [a, b, c] = sample_three(&mut rng, total).ok_or_else(|| {
            PnpError::from(PnpRejection::NotEnoughPoints {
                required: 3,
                actual: total,
            })
        })?;

        let obs = [&observations[a], &observations[b], &observations[c]];
        match p3p_solutions(obs, ransac_iteration, &mut p3p_candidates)? {
            Some(reason) => {
                PnpRansacRejections::try_record_minimal_sample(
                    &mut ransac_rejections,
                    PnpMinimalSampleRejection {
                        ransac_iteration,
                        reason,
                    },
                )?;
            }
            None => {
                saw_generated_candidate = true;
                'candidate: for pose in p3p_candidates.iter() {
                    if let Err(failure) = collect_inliers_into(
                        &mut candidate_inliers,
                        pose,
                        observations,
                        intrinsics,
                        threshold_sq_px2,
                    ) {
                        PnpRansacRejections::try_record_candidate_projection(
                            &mut ransac_rejections,
                            PnpCandidateProjectionRejection {
                                ransac_iteration,
                                observation_index: failure.observation_index,
                                source: failure.source,
                            },
                        )?;
                        continue 'candidate;
                    }
                    saw_projection_complete_candidate = true;

                    if candidate_inliers.len() > best_inliers.len() {
                        best_inliers.clear();
                        best_inliers.extend(candidate_inliers.iter().copied());
                        best_pose = Some(pose);
                        target_iterations = target_iterations.min(adaptive_ransac_iterations(
                            best_inliers.len(),
                            total,
                            RANSAC_CONFIDENCE,
                        ));
                        if best_inliers.len() == total {
                            break;
                        }
                    }
                }
            }
        }
        if ransac_iteration.get() >= target_iterations {
            break;
        }
        ransac_iteration = ransac_iteration.saturating_add(1);
    }

    let pose = match best_pose {
        Some(pose) => pose,
        None if !saw_generated_candidate || !saw_projection_complete_candidate => {
            if let Some(rejections) = ransac_rejections {
                return Err(PnpRejection::NoUsableRansacCandidate { rejections }.into());
            }
            return Err(PnpRejection::NoSolution.into());
        }
        None => return Err(no_consensus_error(ransac_rejections)),
    };
    if best_inliers.len() < config.min_inliers() {
        return Err(no_consensus_error(ransac_rejections));
    }

    let (pose, inliers, refinement) =
        match refine_pose_on_inliers(pose, observations, intrinsics, &best_inliers) {
            Ok((refined_pose, termination)) => match collect_inliers_into(
                &mut candidate_inliers,
                refined_pose,
                observations,
                intrinsics,
                threshold_sq_px2,
            ) {
                Ok(()) if candidate_inliers.len() >= config.min_inliers() => (
                    refined_pose,
                    candidate_inliers,
                    PnpRefinementStatus::Applied { termination },
                ),
                Ok(()) => {
                    let refined_inliers = candidate_inliers.len();
                    (
                        pose,
                        best_inliers,
                        PnpRefinementStatus::RetainedRansacPose {
                            reason: PnpRefinementFallback::LostConsensus {
                                termination,
                                candidate_inliers: refined_inliers,
                                required_inliers: config.min_inliers(),
                            },
                        },
                    )
                }
                Err(failure) => (
                    pose,
                    best_inliers,
                    PnpRefinementStatus::RetainedRansacPose {
                        reason: PnpRefinementFallback::PostRefinementConsensusProjection {
                            termination,
                            observation_index: failure.observation_index,
                            source: failure.source,
                        },
                    },
                ),
            },
            Err(reason) => (
                pose,
                best_inliers,
                PnpRefinementStatus::RetainedRansacPose { reason },
            ),
        };

    Ok(PnpResult {
        pose,
        inliers,
        iterations: ransac_iteration,
        refinement,
        ransac_rejections,
    })
}

fn no_consensus_error(ransac_rejections: Option<PnpRansacRejections>) -> PnpError {
    match ransac_rejections {
        Some(rejections) => PnpRejection::NoConsensusAfterRansacRejections { rejections }.into(),
        None => PnpRejection::NoSolution.into(),
    }
}

fn try_inlier_buffer(
    buffer: PnpInlierBuffer,
    observation_count: usize,
) -> Result<Vec<usize>, PnpError> {
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(observation_count)
        .map_err(|source| PnpError::InlierBufferAllocation {
            buffer,
            observation_count,
            source,
        })?;
    Ok(indices)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct IndexedProjectionFailure {
    observation_index: usize,
    source: PinholeProjectionError,
}

fn collect_inliers_into(
    inliers: &mut Vec<usize>,
    pose: Pose,
    observations: &[Observation],
    intrinsics: PinholeIntrinsics,
    threshold_sq_px2: f64,
) -> Result<(), IndexedProjectionFailure> {
    inliers.clear();
    for (observation_index, observation) in observations.iter().enumerate() {
        match evaluate_reprojection_residual(pose, observation, intrinsics).map_err(|source| {
            IndexedProjectionFailure {
                observation_index,
                source,
            }
        })? {
            ReprojectionResidual::Projectable { residual_sq_px2 }
                if residual_sq_px2 <= threshold_sq_px2 =>
            {
                inliers.push(observation_index);
            }
            ReprojectionResidual::Projectable { .. }
            | ReprojectionResidual::NonPositiveCameraDepth { .. } => {}
        }
    }
    Ok(())
}

fn adaptive_ransac_iterations(inlier_count: usize, total: usize, confidence: f64) -> usize {
    if inlier_count == 0 || total == 0 {
        return usize::MAX;
    }
    let inlier_ratio = (inlier_count as f64 / total as f64).clamp(0.0, 1.0);
    let sample_success = inlier_ratio.powi(3);
    if sample_success >= 1.0 {
        return 1;
    }
    if sample_success <= 0.0 {
        return usize::MAX;
    }
    let numerator = (1.0 - confidence).ln();
    let denominator = (1.0 - sample_success).ln();
    if !numerator.is_finite() || !denominator.is_finite() || denominator >= 0.0 {
        return usize::MAX;
    }
    (numerator / denominator).ceil().max(1.0) as usize
}

fn refine_pose_on_inliers(
    initial_pose: Pose,
    observations: &[Observation],
    intrinsics: PinholeIntrinsics,
    inlier_indices: &[usize],
) -> Result<(Pose, PnpRefinementTermination), PnpRefinementFallback> {
    if inlier_indices.is_empty() {
        return Err(PnpRefinementFallback::EmptyInlierSet);
    }
    let mut pose = initial_pose;
    let mut hessian = [0.0_f32; 36];
    let mut rhs = [0.0_f32; 6];
    let mut accepted_any = false;

    for iter in 0..PNP_REFINEMENT_ITERS {
        let iteration = NonZeroUsize::MIN.saturating_add(iter);
        hessian.fill(0.0);
        rhs.fill(0.0);
        let mut nonprojectable = 0usize;
        let mut current_cost = 0.0_f64;

        for &idx in inlier_indices {
            let obs = observations
                .get(idx)
                .ok_or(PnpRefinementFallback::InvalidInlierIndex {
                    iteration,
                    index: idx,
                    observation_count: observations.len(),
                })?;
            let Some((residual, jacobian)) =
                crate::local_ba::reprojection_residual_and_jacobian(pose, obs, intrinsics)
            else {
                nonprojectable = nonprojectable.saturating_add(1);
                continue;
            };

            let residual_x = f64::from(residual[0]);
            let residual_y = f64::from(residual[1]);
            current_cost =
                residual_x.mul_add(residual_x, residual_y.mul_add(residual_y, current_cost));

            for row in 0..6 {
                rhs[row] -= jacobian[0][row] * residual[0] + jacobian[1][row] * residual[1];
                for col in 0..6 {
                    hessian[row * 6 + col] +=
                        jacobian[0][row] * jacobian[0][col] + jacobian[1][row] * jacobian[1][col];
                }
            }
        }
        if let Some(count) = NonZeroUsize::new(nonprojectable) {
            return Err(PnpRefinementFallback::NonProjectableInliers { iteration, count });
        }

        for axis in 0..6 {
            hessian[axis * 6 + axis] += PNP_REFINEMENT_DAMPING;
        }

        crate::local_ba::solve_linear_system(&mut hessian, &mut rhs, 6)
            .map_err(|source| PnpRefinementFallback::LinearSolve { iteration, source })?;

        let step_m_rad = rhs;
        let tangent = crate::Se3Tangent64::try_from_meters_radians(step_m_rad.map(f64::from))
            .map_err(|source| PnpRefinementFallback::InvalidPoseUpdate { iteration, source })?;
        let normalized_step_norm = pnp_refinement_normalized_step_norm(tangent)
            .map_err(|source| PnpRefinementFallback::InvalidPoseUpdate { iteration, source })?;
        let candidate = tangent
            .try_apply_left_to_metric_pose(pose)
            .map_err(|source| PnpRefinementFallback::InvalidPoseUpdate { iteration, source })?;
        let mut candidate_nonprojectable = 0usize;
        let mut candidate_cost = 0.0_f64;
        for &idx in inlier_indices {
            let observation =
                observations
                    .get(idx)
                    .ok_or(PnpRefinementFallback::InvalidInlierIndex {
                        iteration,
                        index: idx,
                        observation_count: observations.len(),
                    })?;
            match evaluate_reprojection_residual(candidate, observation, intrinsics).map_err(
                |source| PnpRefinementFallback::CandidateProjection {
                    iteration,
                    observation_index: idx,
                    source,
                },
            )? {
                ReprojectionResidual::Projectable { residual_sq_px2 } => {
                    candidate_cost += residual_sq_px2;
                }
                ReprojectionResidual::NonPositiveCameraDepth { .. } => {
                    candidate_nonprojectable = candidate_nonprojectable.saturating_add(1);
                }
            };
        }
        if let Some(count) = NonZeroUsize::new(candidate_nonprojectable) {
            return Err(PnpRefinementFallback::NonProjectableInliers { iteration, count });
        }
        match decide_refinement_step(
            iteration,
            accepted_any,
            normalized_step_norm,
            current_cost,
            candidate_cost,
        )? {
            RefinementStepDecision::Accept => {
                pose = candidate;
                accepted_any = true;
            }
            RefinementStepDecision::Finish(termination) => {
                return Ok((pose, termination));
            }
        }
    }

    Ok((
        pose,
        PnpRefinementTermination::IterationLimit {
            iterations: NonZeroUsize::MIN.saturating_add(PNP_REFINEMENT_ITERS - 1),
        },
    ))
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum RefinementStepDecision {
    Accept,
    Finish(PnpRefinementTermination),
}

fn pnp_refinement_normalized_step_norm(
    tangent: crate::Se3Tangent64,
) -> Result<f64, crate::Se3TangentError> {
    let translation_step_norm_m = tangent.try_translation_tangent_norm_m()?;
    let rotation_step_norm_rad = tangent.try_rotation_vector_norm_rad()?;
    Ok(
        (translation_step_norm_m / PNP_REFINEMENT_TRANSLATION_CONVERGENCE_M)
            .hypot(rotation_step_norm_rad / PNP_REFINEMENT_ROTATION_CONVERGENCE_RAD),
    )
}

fn decide_refinement_step(
    iteration: NonZeroUsize,
    accepted_any: bool,
    normalized_step_norm: f64,
    current_cost: f64,
    candidate_cost: f64,
) -> Result<RefinementStepDecision, PnpRefinementFallback> {
    let current_cost = PnpRefinementCost::try_new(
        iteration,
        PnpRefinementObjectiveStage::Current,
        current_cost,
    )?;
    let candidate_cost = PnpRefinementCost::try_new(
        iteration,
        PnpRefinementObjectiveStage::Candidate,
        candidate_cost,
    )?;

    if normalized_step_norm < 1.0 {
        return if accepted_any {
            Ok(RefinementStepDecision::Finish(
                PnpRefinementTermination::Converged {
                    iterations: iteration,
                },
            ))
        } else {
            Err(PnpRefinementFallback::Stationary { iteration })
        };
    }

    if candidate_cost < current_cost {
        return Ok(RefinementStepDecision::Accept);
    }

    if accepted_any {
        Ok(RefinementStepDecision::Finish(
            PnpRefinementTermination::Stalled {
                iterations: iteration,
                current_cost,
                candidate_cost,
            },
        ))
    } else {
        Err(PnpRefinementFallback::NoImprovement {
            iteration,
            current_cost,
            candidate_cost,
        })
    }
}

fn normalize_bearing(pixel: Keypoint, intrinsics: PinholeIntrinsics) -> Result<[f32; 3], PnpError> {
    let x = (f64::from(pixel.x) - f64::from(intrinsics.cx())) / f64::from(intrinsics.fx());
    let y = (f64::from(pixel.y) - f64::from(intrinsics.cy())) / f64::from(intrinsics.fy());
    let norm = x.hypot(y).hypot(1.0);
    let forward = 1.0 / norm;
    if forward < f64::from(f32::from_bits(1)) {
        return Err(
            PnpError::ObservationBearingForwardComponentOutsideF32Domain { value: forward },
        );
    }
    Ok([(x / norm) as f32, (y / norm) as f32, forward as f32])
}

const MAX_P3P_REAL_ROOTS: usize = 4;
const MAX_P3P_DISTANCE_RATIO_ROOTS: usize = MAX_P3P_REAL_ROOTS * 2;
const MAX_P3P_POSE_CANDIDATES: usize = MAX_P3P_DISTANCE_RATIO_ROOTS;

#[derive(Clone, Copy, Debug, PartialEq)]
struct FixedBuffer<T: Copy, const CAPACITY: usize> {
    entries: [Option<T>; CAPACITY],
    len: usize,
}

impl<T: Copy, const CAPACITY: usize> FixedBuffer<T, CAPACITY> {
    fn new() -> Self {
        Self {
            entries: [None; CAPACITY],
            len: 0,
        }
    }

    fn try_push(&mut self, value: T) -> Result<(), usize> {
        let next_len = self.len.checked_add(1).ok_or(CAPACITY)?;
        let Some(slot) = self.entries.get_mut(self.len) else {
            return Err(CAPACITY);
        };
        *slot = Some(value);
        self.len = next_len;
        Ok(())
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn clear(&mut self) {
        self.len = 0;
    }

    fn iter(&self) -> impl Iterator<Item = T> + '_ {
        self.entries
            .iter()
            .take(self.len)
            .filter_map(|entry| *entry)
    }
}

fn p3p_capacity_error(
    ransac_iteration: NonZeroUsize,
    buffer: PnpP3pBuffer,
    capacity: usize,
) -> PnpError {
    PnpError::P3pBufferCapacityExceeded {
        ransac_iteration,
        buffer,
        capacity,
    }
}

fn p3p_solutions(
    obs: [&Observation; 3],
    ransac_iteration: NonZeroUsize,
    solutions: &mut FixedBuffer<Pose, MAX_P3P_POSE_CANDIDATES>,
) -> Result<Option<PnpMinimalSampleRejectionReason>, PnpError> {
    solutions.clear();
    let p1 = vec3_from_point(obs[0].world);
    let p2 = vec3_from_point(obs[1].world);
    let p3 = vec3_from_point(obs[2].world);
    let f1 = vec3_from_bearing(obs[0].bearing);
    let f2 = vec3_from_bearing(obs[1].bearing);
    let f3 = vec3_from_bearing(obs[2].bearing);

    let a = norm(sub(p2, p3));
    let b = norm(sub(p1, p3));
    let c = norm(sub(p1, p2));

    for (side, value) in [
        (PnpWorldTriangleSide::Point2ToPoint3, a),
        (PnpWorldTriangleSide::Point1ToPoint3, b),
        (PnpWorldTriangleSide::Point1ToPoint2, c),
    ] {
        if !value.is_finite() {
            return Ok(Some(
                PnpMinimalSampleRejectionReason::NonFiniteWorldTriangleSideMeters { side, value },
            ));
        }
    }

    let scene_scale = a.max(b).max(c);
    if scene_scale <= 0.0 || a <= 0.0 || b <= 0.0 || c <= 0.0 {
        return Ok(Some(
            PnpMinimalSampleRejectionReason::DegenerateWorldTriangle,
        ));
    }
    let normalized_a = a / scene_scale;
    let normalized_b = b / scene_scale;
    let normalized_c = c / scene_scale;

    let cos_alpha = dot(f2, f3).clamp(-1.0, 1.0);
    let cos_beta = dot(f1, f3).clamp(-1.0, 1.0);
    let cos_gamma = dot(f1, f2).clamp(-1.0, 1.0);

    let roots = match find_roots(
        cos_alpha,
        cos_beta,
        cos_gamma,
        normalized_a,
        normalized_b,
        normalized_c,
        ransac_iteration,
    )? {
        P3pRootGeneration::Roots(roots) => roots,
        P3pRootGeneration::Rejected(reason) => return Ok(Some(reason)),
    };

    for (x, y) in roots.iter() {
        let denom = 1.0 + x * x - 2.0 * x * cos_gamma;
        if !denom.is_finite() || denom <= 0.0 {
            continue;
        }
        let d1 = c / denom.sqrt();
        let d2 = x * d1;
        let d3 = y * d1;
        if !d1.is_finite()
            || !d2.is_finite()
            || !d3.is_finite()
            || d1 <= 0.0
            || d2 <= 0.0
            || d3 <= 0.0
        {
            continue;
        }

        let c1 = mul(f1, d1);
        let c2 = mul(f2, d2);
        let c3 = mul(f3, d3);

        if let Some(pose) = pose_from_points(p1, p2, p3, c1, c2, c3) {
            solutions.try_push(pose).map_err(|capacity| {
                p3p_capacity_error(ransac_iteration, PnpP3pBuffer::PoseCandidates, capacity)
            })?;
        }
    }

    if solutions.is_empty() {
        Ok(Some(
            PnpMinimalSampleRejectionReason::NoGeometricallyAdmissiblePoseCandidates,
        ))
    } else {
        Ok(None)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum P3pRootGeneration {
    Roots(FixedBuffer<(f64, f64), MAX_P3P_DISTANCE_RATIO_ROOTS>),
    Rejected(PnpMinimalSampleRejectionReason),
}

fn find_roots(
    cos_alpha: f64,
    cos_beta: f64,
    cos_gamma: f64,
    a: f64,
    b: f64,
    c: f64,
    ransac_iteration: NonZeroUsize,
) -> Result<P3pRootGeneration, PnpError> {
    let coeffs_meta = P3pCoeffs {
        cos_alpha,
        cos_beta,
        cos_gamma,
        a,
        b,
        c,
    };
    let coeffs = quartic_coeffs(cos_alpha, cos_beta, cos_gamma, a, b, c);
    let xs = match solve_real_roots(coeffs, ransac_iteration)? {
        RealRootGeneration::Roots(roots) => roots,
        RealRootGeneration::Rejected(reason) => return Ok(P3pRootGeneration::Rejected(reason)),
    };

    let mut roots = FixedBuffer::new();
    for x in xs.iter() {
        if !x.is_finite() || x <= 0.0 {
            continue;
        }
        for sign in [-1.0_f64, 1.0_f64] {
            let Some(y) = y_from_x(x, sign, cos_beta, cos_gamma, b, c) else {
                continue;
            };
            if !y.is_finite() || y <= 0.0 {
                continue;
            }
            let Some(fx) = f_equation(x, sign, &coeffs_meta) else {
                continue;
            };
            if fx.is_finite() && fx.abs() <= P3P_ROOT_TOLERANCE {
                push_unique_root(&mut roots, (x, y), ransac_iteration)?;
            }
        }
    }

    if roots.is_empty() {
        Ok(P3pRootGeneration::Rejected(
            PnpMinimalSampleRejectionReason::NoAdmissibleDistanceRatioRoots,
        ))
    } else {
        Ok(P3pRootGeneration::Roots(roots))
    }
}

struct P3pCoeffs {
    cos_alpha: f64,
    cos_beta: f64,
    cos_gamma: f64,
    a: f64,
    b: f64,
    c: f64,
}

fn f_equation(x: f64, sign: f64, coeffs: &P3pCoeffs) -> Option<f64> {
    let denom = 1.0 + x * x - 2.0 * x * coeffs.cos_gamma;
    if !denom.is_finite() || denom <= 0.0 {
        return None;
    }
    let k = (coeffs.b * coeffs.b / (coeffs.c * coeffs.c)) * denom;
    let disc = k + coeffs.cos_beta * coeffs.cos_beta - 1.0;
    if !disc.is_finite() || disc < 0.0 {
        return None;
    }
    let y = coeffs.cos_beta + sign * disc.sqrt();
    let num = x * x + y * y - 2.0 * x * y * coeffs.cos_alpha;
    let residual = coeffs.a * coeffs.a - (coeffs.c * coeffs.c) * (num / denom);
    residual.is_finite().then_some(residual)
}

fn y_from_x(x: f64, sign: f64, cos_beta: f64, cos_gamma: f64, b: f64, c: f64) -> Option<f64> {
    let denom = 1.0 + x * x - 2.0 * x * cos_gamma;
    if !denom.is_finite() || denom <= 0.0 {
        return None;
    }
    let k = (b * b / (c * c)) * denom;
    let disc = k + cos_beta * cos_beta - 1.0;
    if !disc.is_finite() || disc < 0.0 {
        return None;
    }
    let y = cos_beta + sign * disc.sqrt();
    y.is_finite().then_some(y)
}

fn quartic_coeffs(
    cos_alpha: f64,
    cos_beta: f64,
    cos_gamma: f64,
    a: f64,
    b: f64,
    c: f64,
) -> [f64; 5] {
    let a2 = a * a;
    let b2 = b * b;
    let c2 = c * c;

    let ca = cos_alpha;
    let cb = cos_beta;
    let cg = cos_gamma;

    let n0 = a2 - b2 + c2;
    let n1 = -2.0 * (a2 - b2) * cg;
    let n2 = a2 - b2 - c2;
    let d0 = 2.0 * c2 * cb;
    let d1 = -2.0 * c2 * ca;

    let k_scale = b2 / c2;
    let k0 = 1.0 - k_scale;
    let k1 = 2.0 * k_scale * cg;
    let k2 = -k_scale;

    let n_squared = [
        n0 * n0,
        2.0 * n0 * n1,
        2.0 * n0 * n2 + n1 * n1,
        2.0 * n1 * n2,
        n2 * n2,
    ];
    let n_times_d = [n0 * d0, n0 * d1 + n1 * d0, n1 * d1 + n2 * d0, n2 * d1];
    let d_squared = [d0 * d0, 2.0 * d0 * d1, d1 * d1];
    let k_times_d_squared = [
        k0 * d_squared[0],
        k0 * d_squared[1] + k1 * d_squared[0],
        k0 * d_squared[2] + k1 * d_squared[1] + k2 * d_squared[0],
        k1 * d_squared[2] + k2 * d_squared[1],
        k2 * d_squared[2],
    ];

    [
        n_squared[0] - 2.0 * cb * n_times_d[0] + k_times_d_squared[0],
        n_squared[1] - 2.0 * cb * n_times_d[1] + k_times_d_squared[1],
        n_squared[2] - 2.0 * cb * n_times_d[2] + k_times_d_squared[2],
        n_squared[3] - 2.0 * cb * n_times_d[3] + k_times_d_squared[3],
        n_squared[4] + k_times_d_squared[4],
    ]
}

/// Minimum number of point correspondences required for PnP solving (geometric minimum for P3P).
pub(crate) const MIN_PNP_POINTS: usize = 4;

/// Relative tolerance for treating normalized polynomial coefficients as zero.
const POLY_RELATIVE_COEFFICIENT_TOLERANCE: f64 = 1e-12;
/// Maximum imaginary component for a root to be considered real.
const IMAGINARY_TOLERANCE: f64 = 1e-6;
/// Convergence threshold for the Durand-Kerner root-finding iterations.
const ROOT_CONVERGENCE_THRESHOLD: f64 = 1e-10;
/// Denominator magnitude at or below which Durand-Kerner reports root-estimate breakdown.
const ROOT_DENOMINATOR_TOLERANCE: f64 = 1e-12;
/// Maximum iterations for the Durand-Kerner root-finding algorithm.
const MAX_ROOT_ITERATIONS: NonZeroU8 = NonZeroU8::MIN.saturating_add(63);
/// Tolerance for detecting duplicate P3P root solutions.
const ROOT_UNIQUENESS_TOLERANCE: f64 = 1e-3;
/// Tolerance for accepting the dimensionless P3P equation evaluation as a valid root.
const P3P_ROOT_TOLERANCE: f64 = 1e-3;
/// Target confidence used to adaptively shorten RANSAC once a strong model exists.
const RANSAC_CONFIDENCE: f64 = 0.99;
/// Number of nonlinear pose-only refinement steps on the best inlier set.
const PNP_REFINEMENT_ITERS: usize = 8;
/// Damping added to the normal equations during PnP pose refinement.
const PNP_REFINEMENT_DAMPING: f32 = 1e-4;
/// Translation-tangent convergence threshold for PnP pose refinement, in meters.
const PNP_REFINEMENT_TRANSLATION_CONVERGENCE_M: f64 = 1e-5;
/// Rotation-vector convergence threshold for PnP pose refinement, in radians.
const PNP_REFINEMENT_ROTATION_CONVERGENCE_RAD: f64 = 1e-5;

#[derive(Clone, Copy, Debug, PartialEq)]
enum RealRootGeneration {
    Roots(FixedBuffer<f64, MAX_P3P_REAL_ROOTS>),
    Rejected(PnpMinimalSampleRejectionReason),
}

fn solve_real_roots(
    coeffs: [f64; 5],
    ransac_iteration: NonZeroUsize,
) -> Result<RealRootGeneration, PnpError> {
    if let Some((coefficient_index, value)) = coeffs
        .iter()
        .copied()
        .enumerate()
        .find(|(_, coefficient)| !coefficient.is_finite())
    {
        return Ok(RealRootGeneration::Rejected(
            PnpMinimalSampleRejectionReason::NonFiniteQuarticCoefficient {
                coefficient_index,
                value,
            },
        ));
    }
    let scale = coeffs
        .iter()
        .map(|coefficient| coefficient.abs())
        .fold(0.0, f64::max);
    if scale == 0.0 {
        return Ok(RealRootGeneration::Rejected(
            PnpMinimalSampleRejectionReason::DegenerateQuartic,
        ));
    }
    let mut coeffs = coeffs.map(|coefficient| coefficient / scale);
    let degree = coeffs
        .iter()
        .rposition(|coefficient| coefficient.abs() > POLY_RELATIVE_COEFFICIENT_TOLERANCE)
        .unwrap_or(0);
    if degree == 0 {
        return Ok(RealRootGeneration::Rejected(
            PnpMinimalSampleRejectionReason::DegenerateQuartic,
        ));
    }

    let mut real_roots = FixedBuffer::new();
    if degree == 1 {
        let c1 = coeffs[1];
        if c1.abs() <= POLY_RELATIVE_COEFFICIENT_TOLERANCE {
            return Ok(RealRootGeneration::Rejected(
                PnpMinimalSampleRejectionReason::DegenerateQuartic,
            ));
        }
        real_roots.try_push(-coeffs[0] / c1).map_err(|capacity| {
            p3p_capacity_error(
                ransac_iteration,
                PnpP3pBuffer::RealPolynomialRoots,
                capacity,
            )
        })?;
        return Ok(RealRootGeneration::Roots(real_roots));
    }

    let Some(&lead) = coeffs.get(degree) else {
        return Ok(RealRootGeneration::Rejected(
            PnpMinimalSampleRejectionReason::DegenerateQuartic,
        ));
    };
    if lead.abs() <= POLY_RELATIVE_COEFFICIENT_TOLERANCE {
        return Ok(RealRootGeneration::Rejected(
            PnpMinimalSampleRejectionReason::DegenerateQuartic,
        ));
    }
    for c in coeffs.iter_mut().take(degree + 1) {
        *c /= lead;
    }

    let Some(active_coeffs) = coeffs.get(..=degree) else {
        return Ok(RealRootGeneration::Rejected(
            PnpMinimalSampleRejectionReason::DegenerateQuartic,
        ));
    };
    let roots = match durand_kerner(active_coeffs, MAX_ROOT_ITERATIONS) {
        Ok(roots) => roots,
        Err(reason) => return Ok(RealRootGeneration::Rejected(reason)),
    };
    for r in roots.iter().take(degree) {
        if r.im.abs() < IMAGINARY_TOLERANCE {
            real_roots.try_push(r.re).map_err(|capacity| {
                p3p_capacity_error(
                    ransac_iteration,
                    PnpP3pBuffer::RealPolynomialRoots,
                    capacity,
                )
            })?;
        }
    }
    Ok(RealRootGeneration::Roots(real_roots))
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    fn from_polar(r: f64, theta: f64) -> Self {
        Self::new(r * theta.cos(), r * theta.sin())
    }

    fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    fn try_div(self, rhs: Self) -> Option<Self> {
        let scale = rhs.re.abs().max(rhs.im.abs());
        if !scale.is_finite() || scale == 0.0 {
            return None;
        }
        let rhs_re = rhs.re / scale;
        let rhs_im = rhs.im / scale;
        let denominator = rhs_re.mul_add(rhs_re, rhs_im * rhs_im);
        let lhs_re = self.re / scale;
        let lhs_im = self.im / scale;
        let quotient = Self::new(
            lhs_re.mul_add(rhs_re, lhs_im * rhs_im) / denominator,
            lhs_im.mul_add(rhs_re, -(lhs_re * rhs_im)) / denominator,
        );
        quotient.is_finite().then_some(quotient)
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

fn poly_eval(coeffs: &[f64], x: Complex) -> Complex {
    let mut acc = Complex::new(0.0, 0.0);
    for &c in coeffs.iter().rev() {
        acc = acc * x + Complex::new(c, 0.0);
    }
    acc
}

fn durand_kerner(
    coeffs: &[f64],
    max_iterations: NonZeroU8,
) -> Result<[Complex; MAX_P3P_REAL_ROOTS], PnpMinimalSampleRejectionReason> {
    let degree = coeffs.len().saturating_sub(1);
    if degree == 0 || degree > MAX_P3P_REAL_ROOTS {
        return Err(
            PnpMinimalSampleRejectionReason::RootSolverUnsupportedDegree {
                degree,
                maximum_supported_degree: MAX_P3P_REAL_ROOTS,
            },
        );
    }

    let radius = 1.0_f64;
    let mut roots = [Complex::new(0.0, 0.0); MAX_P3P_REAL_ROOTS];
    for i in 0..degree {
        let theta = (2.0 * std::f64::consts::PI * i as f64) / degree as f64;
        let Some(root) = roots.get_mut(i) else {
            return Err(
                PnpMinimalSampleRejectionReason::RootSolverUnsupportedDegree {
                    degree,
                    maximum_supported_degree: MAX_P3P_REAL_ROOTS,
                },
            );
        };
        *root = Complex::from_polar(radius, theta);
    }

    durand_kerner_from_roots(coeffs, degree, roots, max_iterations)
}

fn durand_kerner_from_roots(
    coeffs: &[f64],
    degree: usize,
    mut roots: [Complex; MAX_P3P_REAL_ROOTS],
    max_iterations: NonZeroU8,
) -> Result<[Complex; MAX_P3P_REAL_ROOTS], PnpMinimalSampleRejectionReason> {
    if degree == 0 || degree > MAX_P3P_REAL_ROOTS || coeffs.len() != degree + 1 {
        return Err(
            PnpMinimalSampleRejectionReason::RootSolverUnsupportedDegree {
                degree,
                maximum_supported_degree: MAX_P3P_REAL_ROOTS,
            },
        );
    }

    for root_iteration_index in 0..max_iterations.get() {
        let root_iteration = NonZeroU8::MIN.saturating_add(root_iteration_index);
        let mut max_delta = 0.0_f64;
        for root_index in 0..degree {
            let root_index_u8 = u8::try_from(root_index).map_err(|_| {
                PnpMinimalSampleRejectionReason::RootSolverUnsupportedDegree {
                    degree,
                    maximum_supported_degree: MAX_P3P_REAL_ROOTS,
                }
            })?;
            let Some(&root) = roots.get(root_index) else {
                return Err(
                    PnpMinimalSampleRejectionReason::RootSolverUnsupportedDegree {
                        degree,
                        maximum_supported_degree: MAX_P3P_REAL_ROOTS,
                    },
                );
            };
            let mut denom = Complex::new(1.0, 0.0);
            for (other_index, &other_root) in roots.iter().take(degree).enumerate() {
                if root_index != other_index {
                    denom = denom * (root - other_root);
                }
            }
            if !denom.is_finite() {
                return Err(
                    PnpMinimalSampleRejectionReason::RootSolverNonFiniteDenominator {
                        root_iteration,
                        root_index: root_index_u8,
                        real: denom.re,
                        imaginary: denom.im,
                    },
                );
            }
            let denominator_magnitude = denom.abs();
            if denominator_magnitude <= ROOT_DENOMINATOR_TOLERANCE {
                return Err(PnpMinimalSampleRejectionReason::RootSolverBreakdown {
                    root_iteration,
                    root_index: root_index_u8,
                    denominator_magnitude,
                });
            }
            let p = poly_eval(coeffs, root);
            if !p.is_finite() {
                return Err(
                    PnpMinimalSampleRejectionReason::RootSolverNonFinitePolynomialValue {
                        root_iteration,
                        root_index: root_index_u8,
                        real: p.re,
                        imaginary: p.im,
                    },
                );
            }
            let Some(delta) = p.try_div(denom) else {
                return Err(PnpMinimalSampleRejectionReason::RootSolverDivisionFailed {
                    root_iteration,
                    root_index: root_index_u8,
                    polynomial_magnitude: p.abs(),
                    denominator_magnitude,
                });
            };
            let delta_magnitude =
                finite_root_correction_magnitude(delta, root_iteration, root_index_u8)?;
            let updated_root = root - delta;
            if !updated_root.is_finite() {
                return Err(
                    PnpMinimalSampleRejectionReason::RootSolverNonFiniteUpdatedRoot {
                        root_iteration,
                        root_index: root_index_u8,
                        updated_real: updated_root.re,
                        updated_imaginary: updated_root.im,
                    },
                );
            }
            let Some(root_slot) = roots.get_mut(root_index) else {
                return Err(
                    PnpMinimalSampleRejectionReason::RootSolverUnsupportedDegree {
                        degree,
                        maximum_supported_degree: MAX_P3P_REAL_ROOTS,
                    },
                );
            };
            *root_slot = updated_root;
            max_delta = max_delta.max(delta_magnitude);
        }
        if max_delta < ROOT_CONVERGENCE_THRESHOLD {
            return Ok(roots);
        }
    }

    Err(PnpMinimalSampleRejectionReason::RootSolverIterationLimit {
        iterations: max_iterations,
    })
}

fn finite_root_correction_magnitude(
    correction: Complex,
    root_iteration: NonZeroU8,
    root_index: u8,
) -> Result<f64, PnpMinimalSampleRejectionReason> {
    let magnitude = correction.abs();
    if !magnitude.is_finite() {
        return Err(
            PnpMinimalSampleRejectionReason::RootSolverNonFiniteCorrectionMagnitude {
                root_iteration,
                root_index,
                correction_real: correction.re,
                correction_imaginary: correction.im,
            },
        );
    }
    Ok(magnitude)
}

fn push_unique_root(
    roots: &mut FixedBuffer<(f64, f64), MAX_P3P_DISTANCE_RATIO_ROOTS>,
    candidate: (f64, f64),
    ransac_iteration: NonZeroUsize,
) -> Result<(), PnpError> {
    let (x, y) = candidate;
    let tol = ROOT_UNIQUENESS_TOLERANCE;
    if roots
        .iter()
        .any(|(rx, ry)| (rx - x).abs() < tol && (ry - y).abs() < tol)
    {
        return Ok(());
    }
    roots.try_push(candidate).map_err(|capacity| {
        p3p_capacity_error(ransac_iteration, PnpP3pBuffer::DistanceRatioRoots, capacity)
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ReprojectionResidual {
    Projectable { residual_sq_px2: f64 },
    NonPositiveCameraDepth { depth_m: f64 },
}

fn evaluate_reprojection_residual(
    pose: Pose,
    observation: &Observation,
    intrinsics: PinholeIntrinsics,
) -> Result<ReprojectionResidual, PinholeProjectionError> {
    let (u_px, v_px) = match project_world_point_f64_px(pose, observation.world, intrinsics)? {
        PinholeProjectionF64::Projected { u_px, v_px } => (u_px, v_px),
        PinholeProjectionF64::NonPositiveCameraDepth { depth_m } => {
            return Ok(ReprojectionResidual::NonPositiveCameraDepth { depth_m });
        }
    };
    let residual_u_px = u_px - f64::from(observation.pixel.x);
    let residual_v_px = v_px - f64::from(observation.pixel.y);
    Ok(ReprojectionResidual::Projectable {
        residual_sq_px2: residual_u_px.mul_add(residual_u_px, residual_v_px * residual_v_px),
    })
}

#[derive(Debug)]
pub enum ReprojectionEvaluationError {
    Allocation {
        observation_count: usize,
        source: TryReserveError,
    },
    Projection {
        observation_index: usize,
        source: PinholeProjectionError,
    },
    MetricOutsideF32PixelDomain {
        metric: ReprojectionMetric,
        value_px: f64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReprojectionMetric {
    Rmse,
    Maximum,
}

impl std::fmt::Display for ReprojectionMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Rmse => "RMSE",
            Self::Maximum => "maximum",
        })
    }
}

impl std::fmt::Display for ReprojectionEvaluationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Allocation {
                observation_count,
                source,
            } => write!(
                f,
                "failed to allocate reprojection residuals for {observation_count} observations: {source}"
            ),
            Self::Projection {
                observation_index,
                source,
            } => write!(
                f,
                "failed to project reprojection observation {observation_index}: {source}"
            ),
            Self::MetricOutsideF32PixelDomain { metric, value_px } => write!(
                f,
                "reprojection {metric} is outside the finite f32 pixel domain: {value_px} px"
            ),
        }
    }
}

impl std::error::Error for ReprojectionEvaluationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Allocation { source, .. } => Some(source),
            Self::Projection { source, .. } => Some(source),
            Self::MetricOutsideF32PixelDomain { .. } => None,
        }
    }
}

pub(crate) fn reprojection_residuals_px(
    pose: &Pose,
    observations: &[Observation],
    intrinsics: PinholeIntrinsics,
) -> Result<Vec<Option<f64>>, ReprojectionEvaluationError> {
    let mut residuals_px = Vec::new();
    residuals_px
        .try_reserve_exact(observations.len())
        .map_err(|source| ReprojectionEvaluationError::Allocation {
            observation_count: observations.len(),
            source,
        })?;
    for (observation_index, observation) in observations.iter().enumerate() {
        let residual_px = match evaluate_reprojection_residual(*pose, observation, intrinsics)
            .map_err(|source| ReprojectionEvaluationError::Projection {
                observation_index,
                source,
            })? {
            ReprojectionResidual::Projectable { residual_sq_px2 } => Some(residual_sq_px2.sqrt()),
            ReprojectionResidual::NonPositiveCameraDepth { .. } => None,
        };
        residuals_px.push(residual_px);
    }
    Ok(residuals_px)
}

pub(crate) fn reprojection_rmse_px(
    residuals_px: &[Option<f64>],
) -> Result<Option<f32>, ReprojectionEvaluationError> {
    let mut scale = 0.0_f64;
    let mut scaled_sum_sq = 1.0_f64;
    let mut count = 0usize;
    for &residual_px in residuals_px.iter().flatten() {
        if residual_px != 0.0 {
            if scale < residual_px {
                let ratio = scale / residual_px;
                scaled_sum_sq = 1.0 + scaled_sum_sq * ratio * ratio;
                scale = residual_px;
            } else {
                let ratio = residual_px / scale;
                scaled_sum_sq += ratio * ratio;
            }
        }
        count += 1;
    }
    if count == 0 {
        return Ok(None);
    }
    let rmse_px = if scale == 0.0 {
        0.0
    } else {
        scale * (scaled_sum_sq / count as f64).sqrt()
    };
    narrow_reprojection_metric(rmse_px, ReprojectionMetric::Rmse).map(Some)
}

pub(crate) fn reprojection_max_px(
    residuals_px: &[Option<f64>],
) -> Result<Option<f32>, ReprojectionEvaluationError> {
    residuals_px
        .iter()
        .flatten()
        .copied()
        .reduce(f64::max)
        .map(|maximum| narrow_reprojection_metric(maximum, ReprojectionMetric::Maximum))
        .transpose()
}

fn narrow_reprojection_metric(
    value_px: f64,
    metric: ReprojectionMetric,
) -> Result<f32, ReprojectionEvaluationError> {
    let narrowed = value_px as f32;
    if !value_px.is_finite() || value_px < 0.0 || !narrowed.is_finite() {
        return Err(ReprojectionEvaluationError::MetricOutsideF32PixelDomain { metric, value_px });
    }
    Ok(narrowed)
}

pub(crate) fn reprojection_mse_per_axis_px2(residuals_px: &[Option<f64>]) -> Option<f64> {
    let mut sum_sq = 0.0_f64;
    let mut count = 0usize;
    for &residual_px in residuals_px.iter().flatten() {
        sum_sq = residual_px.mul_add(residual_px, sum_sq);
        count += 1;
    }
    if count == 0 {
        return None;
    }
    Some(sum_sq / (2.0 * count as f64))
}

fn pose_from_points(
    w1: [f64; 3],
    w2: [f64; 3],
    w3: [f64; 3],
    c1: [f64; 3],
    c2: [f64; 3],
    c3: [f64; 3],
) -> Option<Pose> {
    let xw = normalize(sub(w2, w1))?;
    let zw = normalize(cross(xw, sub(w3, w1)))?;
    let yw = cross(zw, xw);

    let xc = normalize(sub(c2, c1))?;
    let zc = normalize(cross(xc, sub(c3, c1)))?;
    let yc = cross(zc, xc);

    let mut r = mat_from_cols(xc, yc, zc, xw, yw, zw);
    if det(r) < 0.0 {
        let zc_flipped = [-zc[0], -zc[1], -zc[2]];
        r = mat_from_cols(xc, yc, zc_flipped, xw, yw, zw);
    }

    let t = sub(c1, mat_mul_vec_f64(r, w1));
    narrow_pose(r, t)
}

fn mat_from_cols(
    xc: [f64; 3],
    yc: [f64; 3],
    zc: [f64; 3],
    xw: [f64; 3],
    yw: [f64; 3],
    zw: [f64; 3],
) -> [[f64; 3]; 3] {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        r[i][0] = xc[i] * xw[0] + yc[i] * yw[0] + zc[i] * zw[0];
        r[i][1] = xc[i] * xw[1] + yc[i] * yw[1] + zc[i] * zw[1];
        r[i][2] = xc[i] * xw[2] + yc[i] * yw[2] + zc[i] * zw[2];
    }
    r
}

fn det(r: [[f64; 3]; 3]) -> f64 {
    r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
}

fn narrow_pose(rotation: [[f64; 3]; 3], translation: [f64; 3]) -> Option<Pose> {
    let mut rotation_f32 = [[0.0_f32; 3]; 3];
    for (source_row, destination_row) in rotation.iter().zip(&mut rotation_f32) {
        for (&source, destination) in source_row.iter().zip(destination_row) {
            *destination = narrow_finite_f32(source)?;
        }
    }
    let translation = [
        narrow_finite_f32(translation[0])?,
        narrow_finite_f32(translation[1])?,
        narrow_finite_f32(translation[2])?,
    ];
    Some(Pose::from_rt(rotation_f32, translation))
}

fn narrow_finite_f32(value: f64) -> Option<f32> {
    let narrowed = value as f32;
    narrowed.is_finite().then_some(narrowed)
}

fn vec3_from_point(p: Point3) -> [f64; 3] {
    [f64::from(p.x), f64::from(p.y), f64::from(p.z)]
}

fn vec3_from_bearing(bearing: [f32; 3]) -> [f64; 3] {
    bearing.map(f64::from)
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    a[0].hypot(a[1]).hypot(a[2])
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn mul(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f64; 3]) -> Option<[f64; 3]> {
    let n = norm(v);
    if !n.is_finite() || n <= 0.0 {
        return None;
    }
    Some([v[0] / n, v[1] / n, v[2] / n])
}

fn mat_mul_vec_f64(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    [
        matrix[0][0].mul_add(
            vector[0],
            matrix[0][1].mul_add(vector[1], matrix[0][2] * vector[2]),
        ),
        matrix[1][0].mul_add(
            vector[0],
            matrix[1][1].mul_add(vector[1], matrix[1][2] * vector[2]),
        ),
        matrix[2][0].mul_add(
            vector[0],
            matrix[2][1].mul_add(vector[1], matrix[2][2] * vector[2]),
        ),
    ]
}

#[derive(Debug)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: NonZeroU64) -> Self {
        Self { state: seed.get() }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.next_u64() as usize) % max
    }
}

fn sample_three(rng: &mut XorShift64, max: usize) -> Option<[usize; 3]> {
    if max < 3 {
        return None;
    }
    let a = rng.next_usize(max);
    let mut b = rng.next_usize(max - 1);
    if b >= a {
        b += 1;
    }

    let (min_ab, max_ab) = if a < b { (a, b) } else { (b, a) };
    let mut c = rng.next_usize(max - 2);
    if c >= min_ab {
        c += 1;
    }
    if c >= max_ab {
        c += 1;
    }

    Some([a, b, c])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{
        axis_angle_pose, make_pinhole_intrinsics, observations_from_projection,
    };

    fn synthetic_world_points() -> Vec<Point3> {
        let mut points = Vec::new();
        for yi in -2..=2 {
            for xi in -2..=2 {
                let x = xi as f32 * 0.25;
                let y = yi as f32 * 0.20;
                let z = 3.0 + 0.08 * ((xi * xi + yi * yi) as f32);
                points.push(Point3 { x, y, z });
            }
        }
        points
    }

    fn rot_frob_norm(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> f32 {
        let mut sum = 0.0_f32;
        for i in 0..3 {
            for j in 0..3 {
                let d = a[i][j] - b[i][j];
                sum += d * d;
            }
        }
        sum.sqrt()
    }

    #[test]
    fn ransac_config_rejects_invalid_solver_inputs() {
        assert!(matches!(
            RansacConfig::new(0, 1.0, MIN_PNP_POINTS, 1),
            Err(RansacConfigError::ZeroMaxIterations)
        ));
        assert!(matches!(
            RansacConfig::new(1, 1.0, MIN_PNP_POINTS, 0),
            Err(RansacConfigError::ZeroSeed)
        ));
        assert!(matches!(
            RansacConfig::new(1, f32::NAN, MIN_PNP_POINTS, 1),
            Err(RansacConfigError::InvalidReprojectionThresholdPx { .. })
        ));
        assert!(matches!(
            RansacConfig::new(1, 1.0, MIN_PNP_POINTS - 1, 1),
            Err(RansacConfigError::TooFewMinInliers {
                value,
                minimum,
            }) if value == MIN_PNP_POINTS - 1 && minimum == MIN_PNP_POINTS
        ));
    }

    #[test]
    fn pinhole_intrinsics_reject_non_finite_and_non_positive_parameters() {
        for (field, values) in [
            ("fx", [f32::NAN, 400.0, 320.0, 240.0]),
            ("fy", [400.0, f32::INFINITY, 320.0, 240.0]),
            ("cx", [400.0, 400.0, f32::NEG_INFINITY, 240.0]),
            ("cy", [400.0, 400.0, 320.0, f32::NAN]),
        ] {
            assert!(matches!(
                PinholeIntrinsics::try_new(values[0], values[1], values[2], values[3]),
                Err(IntrinsicsError::NonFinite { field: actual, .. }) if actual == field
            ));
        }
        for (field, fx, fy) in [("fx", 0.0, 400.0), ("fy", 400.0, -1.0)] {
            assert!(matches!(
                PinholeIntrinsics::try_new(fx, fy, 320.0, 240.0),
                Err(IntrinsicsError::NonPositive { field: actual, .. }) if actual == field
            ));
        }
    }

    #[test]
    fn observation_rejects_non_finite_world_and_pixel_values() {
        let intrinsics =
            PinholeIntrinsics::try_new(400.0, 400.0, 320.0, 240.0).expect("intrinsics");
        let cases = [
            (
                "world.x",
                Point3 {
                    x: f32::NAN,
                    y: 0.0,
                    z: 1.0,
                },
                Keypoint { x: 1.0, y: 2.0 },
            ),
            (
                "world.y",
                Point3 {
                    x: 0.0,
                    y: f32::INFINITY,
                    z: 1.0,
                },
                Keypoint { x: 1.0, y: 2.0 },
            ),
            (
                "world.z",
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: f32::NEG_INFINITY,
                },
                Keypoint { x: 1.0, y: 2.0 },
            ),
            (
                "pixel.x",
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                Keypoint {
                    x: f32::NAN,
                    y: 2.0,
                },
            ),
            (
                "pixel.y",
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                Keypoint {
                    x: 1.0,
                    y: f32::INFINITY,
                },
            ),
        ];

        for (field, world, pixel) in cases {
            assert!(matches!(
                Observation::try_new(world, pixel, intrinsics),
                Err(PnpError::NonFiniteObservation { field: actual, .. }) if actual == field
            ));
        }
    }

    fn l2(a: [f32; 3], b: [f32; 3]) -> f32 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    #[test]
    fn normalize_bearing_has_unit_norm() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 400.0, 400.0, 320.0, 240.0).expect("intrinsics");
        let pixel = Keypoint { x: 369.0, y: 211.0 };
        let b = normalize_bearing(pixel, intrinsics).expect("bearing");
        let n = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
        assert!((n - 1.0).abs() < 1e-6, "bearing norm must be 1, got {n}");
    }

    #[test]
    fn normalize_bearing_avoids_f32_pixel_subtraction_overflow() {
        let intrinsics = PinholeIntrinsics::try_new(f32::MAX, f32::MAX, -f32::MAX, 0.0)
            .expect("finite positive intrinsics");
        let bearing = normalize_bearing(
            Keypoint {
                x: f32::MAX,
                y: 0.0,
            },
            intrinsics,
        )
        .expect("the finite normalized coordinate is representable");

        assert!(bearing.into_iter().all(f32::is_finite));
        assert!(bearing[0] > 0.0);
        assert!(bearing[2] > 0.0);
        let norm = bearing
            .into_iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() <= f32::EPSILON);
    }

    #[test]
    fn normalize_bearing_rejects_forward_component_that_underflows_f32() {
        let intrinsics = PinholeIntrinsics::try_new(f32::from_bits(1), 1.0, -f32::MAX, 0.0)
            .expect("finite positive intrinsics");
        let error = normalize_bearing(
            Keypoint {
                x: f32::MAX,
                y: 0.0,
            },
            intrinsics,
        )
        .expect_err("the positive f64 forward component rounds to zero in f32");

        assert!(matches!(
            error,
            PnpError::ObservationBearingForwardComponentOutsideF32Domain { value }
                if value.is_finite() && value > 0.0 && value < f64::from(f32::from_bits(1))
        ));
        assert!(error.rejection().is_none());
    }

    #[test]
    fn polynomial_roots_are_invariant_to_coefficient_scale() {
        let base = [-6.0, 11.0, -6.0, 1.0, 0.0];
        for scale in [1e-18, 1.0, 1e18] {
            let RealRootGeneration::Roots(roots) = solve_real_roots(
                base.map(|coefficient| coefficient * scale),
                NonZeroUsize::MIN,
            )
            .expect("fixed root capacity") else {
                panic!("scaled polynomial must produce real roots")
            };
            let mut roots: Vec<_> = roots.iter().collect();
            roots.sort_by(f64::total_cmp);
            assert_eq!(roots.len(), 3, "scale={scale:e}, roots={roots:?}");
            for (actual, expected) in roots.into_iter().zip([1.0, 2.0, 3.0]) {
                assert!(
                    (actual - expected).abs() < 1e-8,
                    "scale={scale:e}, actual={actual:e}, expected={expected:e}"
                );
            }
        }
    }

    #[test]
    fn stack_quartic_coefficients_match_the_convolution_reference() {
        fn multiply(lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
            let mut product = vec![0.0; lhs.len() + rhs.len() - 1];
            for (lhs_index, lhs_value) in lhs.iter().copied().enumerate() {
                for (rhs_index, rhs_value) in rhs.iter().copied().enumerate() {
                    product[lhs_index + rhs_index] += lhs_value * rhs_value;
                }
            }
            product
        }

        let mut state = 0x5EED_CAFE_D00D_F00D_u64;
        for _ in 0..256 {
            let mut next_unit = || {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                f64::from((state >> 32) as u32) / f64::from(u32::MAX)
            };
            let cos_alpha = next_unit() * 1.8 - 0.9;
            let cos_beta = next_unit() * 1.8 - 0.9;
            let cos_gamma = next_unit() * 1.8 - 0.9;
            let a = 0.1 + next_unit() * 0.9;
            let b = 0.1 + next_unit() * 0.9;
            let c = 0.1 + next_unit() * 0.9;

            let actual = quartic_coeffs(cos_alpha, cos_beta, cos_gamma, a, b, c);
            let a2 = a.powi(2);
            let b2 = b.powi(2);
            let c2 = c.powi(2);
            let ca = cos_alpha;
            let cb = cos_beta;
            let cg = cos_gamma;
            let n = [a2 - b2 + c2, -2.0 * (a2 - b2) * cg, a2 - b2 - c2];
            let d = [2.0 * c2 * cb, -2.0 * c2 * ca];
            let scale = b2 / c2;
            let k = [1.0 - scale, 2.0 * scale * cg, -scale];
            let n_squared = multiply(&n, &n);
            let n_times_d = multiply(&n, &d);
            let k_times_d_squared = multiply(&k, &multiply(&d, &d));
            let mut expected = [0.0; 5];
            for (index, expected_coefficient) in expected.iter_mut().enumerate() {
                *expected_coefficient = n_squared.get(index).copied().unwrap_or(0.0)
                    - 2.0 * cb * n_times_d.get(index).copied().unwrap_or(0.0)
                    + k_times_d_squared.get(index).copied().unwrap_or(0.0);
            }

            for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
                let scale = actual.abs().max(expected.abs()).max(1.0);
                assert!(
                    (actual - expected).abs() <= 8.0 * f64::EPSILON * scale,
                    "coefficient {index}: actual={actual:e}, expected={expected:e}"
                );
            }
        }
    }

    #[test]
    fn p3p_is_invariant_to_scene_scale() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let base_world = [
            Point3 {
                x: -0.4,
                y: -0.3,
                z: 4.0,
            },
            Point3 {
                x: 0.7,
                y: -0.2,
                z: 4.5,
            },
            Point3 {
                x: 0.1,
                y: 0.8,
                z: 5.2,
            },
        ];
        let base_translation = [0.2, -0.1, 0.3];
        let axis_angle = [0.05, -0.03, 0.02];

        for scale in [1e-23_f32, 1e-3, 1.0, 1e3, 1e20] {
            let world = base_world.map(|point| Point3 {
                x: point.x * scale,
                y: point.y * scale,
                z: point.z * scale,
            });
            let expected = axis_angle_pose(
                base_translation.map(|component| component * scale),
                axis_angle,
            );
            let observations: Vec<_> = world
                .iter()
                .map(|&point| {
                    let PinholeProjectionF64::Projected { u_px, v_px } =
                        project_world_point_f64_px(expected, point, intrinsics)
                            .expect("finite projection")
                    else {
                        panic!("scaled point must remain in front of the camera");
                    };
                    Observation::try_new(
                        point,
                        Keypoint {
                            x: u_px as f32,
                            y: v_px as f32,
                        },
                        intrinsics,
                    )
                    .expect("finite observation")
                })
                .collect();
            assert_eq!(observations.len(), 3);
            let mut solutions = FixedBuffer::new();
            let rejection = p3p_solutions(
                [&observations[0], &observations[1], &observations[2]],
                NonZeroUsize::MIN,
                &mut solutions,
            )
            .expect("fixed P3P capacity");
            assert!(rejection.is_none(), "sample rejection: {rejection:?}");
            let matched = solutions.iter().any(|solution| {
                let actual_translation = solution.translation().map(f64::from);
                let expected_translation = expected.translation().map(f64::from);
                let translation_error = (actual_translation[0] - expected_translation[0])
                    .abs()
                    .hypot((actual_translation[1] - expected_translation[1]).abs())
                    .hypot((actual_translation[2] - expected_translation[2]).abs());
                rot_frob_norm(solution.rotation(), expected.rotation()) < 2e-3
                    && translation_error / f64::from(scale) < 2e-2
            });
            assert!(
                matched,
                "no correct P3P solution at scene scale {scale:e}; candidates={solutions:?}"
            );
        }
    }

    #[test]
    fn p3p_world_triangle_distances_do_not_overflow_in_f32_domain() {
        let observation = |world| Observation {
            world,
            pixel: Keypoint { x: 0.0, y: 0.0 },
            bearing: [0.0, 0.0, 1.0],
        };
        let observations = [
            observation(Point3 {
                x: f32::MAX,
                y: 0.0,
                z: 1.0,
            }),
            observation(Point3 {
                x: -f32::MAX,
                y: 0.0,
                z: 1.0,
            }),
            observation(Point3 {
                x: 0.0,
                y: 1.0,
                z: 1.0,
            }),
        ];
        let mut candidates = FixedBuffer::new();

        let rejection = p3p_solutions(
            [&observations[0], &observations[1], &observations[2]],
            NonZeroUsize::MIN,
            &mut candidates,
        )
        .expect("fixed P3P capacity")
        .expect("parallel bearings cannot produce an admissible pose");
        assert!(!matches!(
            rejection,
            PnpMinimalSampleRejectionReason::NonFiniteWorldTriangleSideMeters { .. }
        ));
        assert!(candidates.is_empty());
    }

    #[test]
    fn durand_kerner_iteration_limit_is_not_reported_as_convergence() {
        let one_iteration = NonZeroU8::MIN;
        let error = durand_kerner(&[-6.0, 11.0, -6.0, 1.0], one_iteration)
            .expect_err("a cubic with roots 1, 2, and 3 cannot converge in one update sweep");
        assert_eq!(
            error,
            PnpMinimalSampleRejectionReason::RootSolverIterationLimit {
                iterations: one_iteration,
            }
        );
    }

    #[test]
    fn durand_kerner_coincident_estimates_are_breakdown_not_convergence() {
        let coincident = Complex::new(1.0, 0.0);
        let roots = [
            coincident,
            coincident,
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let error = durand_kerner_from_roots(&[-1.0, 0.0, 1.0], 2, roots, NonZeroU8::MIN)
            .expect_err("coincident root estimates have a zero Durand-Kerner denominator");
        assert_eq!(
            error,
            PnpMinimalSampleRejectionReason::RootSolverBreakdown {
                root_iteration: NonZeroU8::MIN,
                root_index: 0,
                denominator_magnitude: 0.0,
            }
        );
    }

    #[test]
    fn complex_division_scales_finite_large_operands_before_squaring() {
        let component = f64::MAX / 4.0;
        let quotient = Complex::new(component, component)
            .try_div(Complex::new(component, component))
            .expect("finite equal operands must have a finite quotient");
        assert!((quotient.re - 1.0).abs() <= f64::EPSILON);
        assert!(quotient.im.abs() <= f64::EPSILON);
    }

    #[test]
    fn finite_root_correction_with_unrepresentable_norm_is_rejected() {
        let component = f64::MAX * 0.75;
        let error =
            finite_root_correction_magnitude(Complex::new(component, component), NonZeroU8::MIN, 2)
                .expect_err("the correction components are finite but their norm exceeds f64");
        assert_eq!(
            error,
            PnpMinimalSampleRejectionReason::RootSolverNonFiniteCorrectionMagnitude {
                root_iteration: NonZeroU8::MIN,
                root_index: 2,
                correction_real: component,
                correction_imaginary: component,
            }
        );
    }

    #[test]
    fn fixed_buffer_reports_capacity_without_allocating_or_panicking() {
        let mut buffer = FixedBuffer::<u8, 1>::new();
        buffer.try_push(7).expect("first slot");
        assert_eq!(buffer.try_push(8), Err(1));
        assert_eq!(buffer.iter().collect::<Vec<_>>(), vec![7]);
    }

    #[test]
    fn sample_three_returns_distinct_indices() {
        let mut rng =
            XorShift64::new(NonZeroU64::new(0xDEADBEEF).expect("test RANSAC seed is nonzero"));
        for _ in 0..500 {
            let sample = sample_three(&mut rng, 17).expect("sample");
            assert!(sample[0] < 17 && sample[1] < 17 && sample[2] < 17);
            assert_ne!(sample[0], sample[1]);
            assert_ne!(sample[0], sample[2]);
            assert_ne!(sample[1], sample[2]);
        }
    }

    #[test]
    fn pose_inverse_is_involution() {
        let pose = axis_angle_pose([0.3, -0.2, 0.7], [0.1, -0.05, 0.08]);
        let recovered = pose.inverse().inverse();
        assert!(rot_frob_norm(pose.rotation(), recovered.rotation()) < 1e-5);
        assert!(l2(pose.translation(), recovered.translation()) < 1e-5);
    }

    #[test]
    fn solve_pnp_ransac_recovers_pose_on_synthetic_scene() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let world = synthetic_world_points();
        let pose_gt = axis_angle_pose([0.2, -0.1, 0.35], [0.08, -0.06, 0.04]);

        let observations =
            observations_from_projection(pose_gt, &world, intrinsics).expect("observations");
        assert!(observations.len() >= 20);

        let config = RansacConfig::new(700, 1.0, 20, 0xBAD5EED).expect("RANSAC config");
        let result = solve_pnp_ransac(&observations, intrinsics, config).expect("pnp");
        assert!(result.inliers().len() >= 20, "insufficient inliers");
        assert!(result.refinement().applied());
        assert!(result.candidate_projection_rejections().is_none());

        let rot_err = rot_frob_norm(result.pose().rotation(), pose_gt.rotation());
        let trans_err = l2(result.pose().translation(), pose_gt.translation());
        assert!(rot_err < 0.03, "rotation error too high: {rot_err}");
        assert!(trans_err < 0.08, "translation error too high: {trans_err}");
    }

    #[test]
    fn successful_pnp_retains_earlier_minimal_sample_rejection() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let good = observations_from_projection(
            Pose::identity(),
            &synthetic_world_points()[..4],
            intrinsics,
        )
        .expect("observations");
        let mut observations = vec![good[0]; 4];
        observations.extend_from_slice(&good);

        let config = RansacConfig::new(2, 1.0, 4, 192).expect("RANSAC config");
        let result = solve_pnp_ransac(&observations, intrinsics, config)
            .expect("the first sampled triple is degenerate and the second is valid");
        let rejections = result
            .minimal_sample_rejections()
            .expect("successful solve must retain its earlier degradation");

        assert_eq!(rejections.count().get(), 1);
        assert_eq!(
            rejections.first_rejection().reason(),
            PnpMinimalSampleRejectionReason::DegenerateWorldTriangle
        );
    }

    #[test]
    fn solve_pnp_ransac_handles_outliers() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let world = synthetic_world_points();
        let pose_gt = axis_angle_pose([0.2, -0.1, 0.35], [0.08, -0.06, 0.04]);

        let clean =
            observations_from_projection(pose_gt, &world, intrinsics).expect("observations");
        let mut with_outliers = Vec::with_capacity(clean.len());
        for (idx, obs) in clean.iter().enumerate() {
            let mut pixel = obs.pixel();
            if idx % 6 == 0 {
                pixel.x += 120.0;
                pixel.y -= 85.0;
            }
            with_outliers
                .push(Observation::try_new(obs.world(), pixel, intrinsics).expect("observation"));
        }

        let config = RansacConfig::new(1000, 2.0, 14, 0x1337).expect("RANSAC config");
        let result = solve_pnp_ransac(&with_outliers, intrinsics, config).expect("pnp");
        assert!(
            result.inliers().len() >= 14,
            "expected robust inliers, got {}",
            result.inliers().len()
        );

        let rot_err = rot_frob_norm(result.pose().rotation(), pose_gt.rotation());
        let trans_err = l2(result.pose().translation(), pose_gt.translation());
        assert!(
            rot_err < 0.08,
            "rotation error too high with outliers: {rot_err}"
        );
        assert!(
            trans_err < 0.18,
            "translation error too high with outliers: {trans_err}"
        );
    }

    #[test]
    fn ransac_f32_max_pixel_threshold_is_squared_in_f64_without_overflow() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let pose = axis_angle_pose([0.2, -0.1, 0.35], [0.08, -0.06, 0.04]);
        let observations =
            observations_from_projection(pose, &synthetic_world_points(), intrinsics)
                .expect("observations");
        let threshold_sq_px2 = f64::from(f32::MAX).powi(2);
        assert!(threshold_sq_px2.is_finite());

        let config =
            RansacConfig::new(100, f32::MAX, 20, 0xA11CE).expect("finite maximum f32 threshold");
        let result = solve_pnp_ransac(&observations, intrinsics, config)
            .expect("large finite threshold must remain representable");
        assert_eq!(result.inliers().len(), observations.len());
    }

    #[test]
    fn finite_extreme_projection_is_an_outlier_not_a_numerical_failure() {
        let intrinsics =
            PinholeIntrinsics::try_new(1.0, 1.0, 0.0, 0.0).expect("finite unit intrinsics");
        let observation = Observation {
            world: Point3 {
                x: f32::MAX,
                y: f32::MAX,
                z: f32::MIN_POSITIVE,
            },
            pixel: Keypoint { x: 0.0, y: 0.0 },
            bearing: [0.0, 0.0, 1.0],
        };
        let mut inliers = try_inlier_buffer(PnpInlierBuffer::CandidateScratch, 1)
            .expect("single-index scratch buffer");

        collect_inliers_into(
            &mut inliers,
            Pose::identity(),
            &[observation],
            intrinsics,
            1.0,
        )
        .expect("finite f32 inputs must remain numerically evaluable in f64");
        assert!(inliers.is_empty());
    }

    #[test]
    fn candidate_projection_rejection_summary_is_nonzero_and_source_chained() {
        let intrinsics =
            PinholeIntrinsics::try_new(1.0, 1.0, 0.0, 0.0).expect("finite unit intrinsics");
        let observation = Observation::try_new(
            Point3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            Keypoint { x: 0.0, y: 0.0 },
            intrinsics,
        )
        .expect("finite observation");
        let mut inliers = try_inlier_buffer(PnpInlierBuffer::CandidateScratch, 1)
            .expect("single-index scratch buffer");
        let failure = collect_inliers_into(
            &mut inliers,
            Pose::from_rt(Pose::identity().rotation(), [f32::NAN, 0.0, 0.0]),
            &[observation],
            intrinsics,
            1.0,
        )
        .expect_err("nonfinite candidate geometry must reject the candidate");
        let rejection = PnpCandidateProjectionRejection {
            ransac_iteration: NonZeroUsize::MIN,
            observation_index: failure.observation_index,
            source: failure.source,
        };
        let mut rejections = PnpCandidateProjectionRejections::first(rejection);
        rejections
            .try_record_another(PnpRansacRejectionKind::CandidateProjection)
            .expect("second rejection count");

        assert_eq!(rejections.count().get(), 2);
        assert_eq!(rejections.first_rejection().ransac_iteration().get(), 1);
        assert_eq!(rejections.first_rejection().observation_index(), 0);
        assert!(matches!(
            rejections.first_rejection().projection_error(),
            PinholeProjectionError::NonFiniteCameraPointMeters {
                axis: CameraFrameAxis::X,
                value,
            } if value.is_nan()
        ));

        let summary = PnpRansacRejections {
            inner: PnpRansacRejectionsInner::CandidateProjections(rejections),
        };
        let error = PnpError::from(PnpRejection::NoUsableRansacCandidate {
            rejections: summary,
        });
        let rejection_source = std::error::Error::source(&error).expect("rejection source");
        let summary_source = rejection_source.source().expect("rejection summary source");
        let rejection_source = summary_source.source().expect("indexed rejection source");
        assert!(
            error
                .to_string()
                .contains("2 rejected candidate projections")
        );
        assert!(rejection_source.to_string().contains("observation 0"));
        let projection_source = rejection_source.source().expect("projection source");
        assert!(projection_source.to_string().contains("camera-frame x"));
        let recoverable = error.rejection().expect("recoverable no-candidate outcome");
        assert!(
            recoverable
                .to_string()
                .contains("2 rejected candidate projections")
        );
        assert!(std::error::Error::source(&recoverable).is_some());

        let no_consensus = no_consensus_error(Some(summary));
        assert!(matches!(
            &no_consensus,
            PnpError::Rejected(PnpRejection::NoConsensusAfterRansacRejections { .. })
        ));
        assert!(std::error::Error::source(&no_consensus).is_some());
    }

    #[test]
    fn candidate_projection_rejection_count_overflow_is_explicit() {
        let first = PnpCandidateProjectionRejection {
            ransac_iteration: NonZeroUsize::MIN,
            observation_index: 0,
            source: PinholeProjectionError::NonFiniteCameraPointMeters {
                axis: CameraFrameAxis::X,
                value: f64::NAN,
            },
        };
        let mut rejections = PnpCandidateProjectionRejections {
            count: NonZeroUsize::MAX,
            first,
        };

        assert!(matches!(
            rejections.try_record_another(PnpRansacRejectionKind::CandidateProjection),
            Err(PnpError::RansacRejectionCountOverflow {
                kind: PnpRansacRejectionKind::CandidateProjection,
            })
        ));
    }

    #[test]
    fn minimal_sample_rejection_count_overflow_is_explicit() {
        let first = PnpMinimalSampleRejection {
            ransac_iteration: NonZeroUsize::MIN,
            reason: PnpMinimalSampleRejectionReason::DegenerateWorldTriangle,
        };
        let mut rejections = PnpMinimalSampleRejections {
            count: NonZeroUsize::MAX,
            first,
        };

        assert!(matches!(
            rejections.try_record_another(PnpRansacRejectionKind::MinimalSample),
            Err(PnpError::RansacRejectionCountOverflow {
                kind: PnpRansacRejectionKind::MinimalSample,
            })
        ));
    }

    #[test]
    fn pnp_inlier_buffer_allocation_error_preserves_buffer_and_source() {
        let error = try_inlier_buffer(PnpInlierBuffer::CandidateScratch, usize::MAX)
            .expect_err("unaddressable inlier buffer must fail");
        assert!(matches!(
            &error,
            PnpError::InlierBufferAllocation {
                buffer: PnpInlierBuffer::CandidateScratch,
                observation_count: usize::MAX,
                ..
            }
        ));
        let source = std::error::Error::source(&error).expect("allocation source");
        assert!(!source.to_string().is_empty());
        assert!(error.rejection().is_none());
    }

    #[test]
    fn refine_pose_on_inliers_reduces_reprojection_rmse_px() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let world = synthetic_world_points();
        let pose_gt = axis_angle_pose([0.2, -0.1, 0.35], [0.08, -0.06, 0.04]);
        let observations =
            observations_from_projection(pose_gt, &world, intrinsics).expect("observations");
        let inlier_indices: Vec<usize> = (0..observations.len()).collect();

        let initial = axis_angle_pose([0.23, -0.08, 0.33], [0.10, -0.04, 0.02]);
        let initial_residuals_px = reprojection_residuals_px(&initial, &observations, intrinsics)
            .expect("initial reprojection residuals");
        let (refined, termination) =
            refine_pose_on_inliers(initial, &observations, intrinsics, &inlier_indices)
                .expect("refinement");
        let refined_residuals_px = reprojection_residuals_px(&refined, &observations, intrinsics)
            .expect("refined reprojection residuals");

        let initial_rmse = reprojection_rmse_px(&initial_residuals_px)
            .expect("representable initial rmse")
            .expect("initial rmse");
        let refined_rmse = reprojection_rmse_px(&refined_residuals_px)
            .expect("representable refined rmse")
            .expect("refined rmse");
        assert!(
            refined_rmse < initial_rmse,
            "expected refinement to improve reprojection error: initial={initial_rmse}, refined={refined_rmse}"
        );
        assert!(termination.iterations().get() <= PNP_REFINEMENT_ITERS);
    }

    #[test]
    fn refinement_convergence_norm_normalizes_meter_and_radian_domains() {
        let translation_only = crate::Se3Tangent64::try_from_meters_radians([
            PNP_REFINEMENT_TRANSLATION_CONVERGENCE_M,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ])
        .expect("finite translation tangent");
        let rotation_only = crate::Se3Tangent64::try_from_meters_radians([
            0.0,
            0.0,
            0.0,
            PNP_REFINEMENT_ROTATION_CONVERGENCE_RAD,
            0.0,
            0.0,
        ])
        .expect("finite rotation vector");
        let both = crate::Se3Tangent64::try_from_meters_radians([
            PNP_REFINEMENT_TRANSLATION_CONVERGENCE_M,
            0.0,
            0.0,
            PNP_REFINEMENT_ROTATION_CONVERGENCE_RAD,
            0.0,
            0.0,
        ])
        .expect("finite mixed-domain tangent");

        assert_eq!(
            pnp_refinement_normalized_step_norm(translation_only).expect("finite norm"),
            1.0
        );
        assert_eq!(
            pnp_refinement_normalized_step_norm(rotation_only).expect("finite norm"),
            1.0
        );
        assert!(
            (pnp_refinement_normalized_step_norm(both).expect("finite norm") - 2.0_f64.sqrt())
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn refinement_objective_gate_never_accepts_a_non_improving_step() {
        let iteration = NonZeroUsize::MIN;
        assert_eq!(
            decide_refinement_step(iteration, false, 1.0, 10.0, 9.0).expect("improving step"),
            RefinementStepDecision::Accept
        );
        assert!(matches!(
            decide_refinement_step(iteration, false, 1.0, 10.0, 10.0),
            Err(PnpRefinementFallback::NoImprovement { .. })
        ));
        assert!(matches!(
            decide_refinement_step(iteration, false, 0.0, 10.0, 10.0),
            Err(PnpRefinementFallback::Stationary { .. })
        ));
        assert!(matches!(
            decide_refinement_step(iteration, true, 1.0, 10.0, 11.0),
            Ok(RefinementStepDecision::Finish(
                PnpRefinementTermination::Stalled { .. }
            ))
        ));
        assert!(matches!(
            decide_refinement_step(iteration, false, 1.0, f64::NAN, 9.0),
            Err(PnpRefinementFallback::InvalidObjective {
                stage: PnpRefinementObjectiveStage::Current,
                ..
            })
        ));
    }

    #[test]
    fn refine_pose_never_increases_the_fixed_inlier_objective() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let world = synthetic_world_points();
        let pose_gt = axis_angle_pose([0.2, -0.1, 0.35], [0.08, -0.06, 0.04]);
        let observations =
            observations_from_projection(pose_gt, &world, intrinsics).expect("observations");
        let inlier_indices: Vec<usize> = (0..observations.len()).collect();

        for initial in [
            axis_angle_pose([0.23, -0.08, 0.33], [0.10, -0.04, 0.02]),
            axis_angle_pose([0.15, -0.15, 0.40], [0.02, -0.10, 0.08]),
            axis_angle_pose([0.30, 0.00, 0.25], [0.15, 0.02, -0.03]),
        ] {
            let initial_cost: f64 = inlier_indices
                .iter()
                .map(|&index| {
                    match evaluate_reprojection_residual(initial, &observations[index], intrinsics)
                        .expect("finite initial projection")
                    {
                        ReprojectionResidual::Projectable { residual_sq_px2 } => residual_sq_px2,
                        ReprojectionResidual::NonPositiveCameraDepth { .. } => {
                            panic!("initial inlier must be projectable")
                        }
                    }
                })
                .sum();
            if let Ok((refined, _)) =
                refine_pose_on_inliers(initial, &observations, intrinsics, &inlier_indices)
            {
                let refined_cost: f64 = inlier_indices
                    .iter()
                    .map(|&index| {
                        match evaluate_reprojection_residual(
                            refined,
                            &observations[index],
                            intrinsics,
                        )
                        .expect("finite refined projection")
                        {
                            ReprojectionResidual::Projectable { residual_sq_px2 } => {
                                residual_sq_px2
                            }
                            ReprojectionResidual::NonPositiveCameraDepth { .. } => {
                                panic!("refined inlier must be projectable")
                            }
                        }
                    })
                    .sum();
                assert!(
                    refined_cost <= initial_cost,
                    "refinement increased fixed-inlier cost: initial={initial_cost}, refined={refined_cost}"
                );
            }
        }
    }

    #[test]
    fn refine_pose_reports_invalid_inlier_provenance() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let observations =
            observations_from_projection(Pose::identity(), &synthetic_world_points(), intrinsics)
                .expect("observations");
        let invalid_index = observations.len();

        let error = refine_pose_on_inliers(
            Pose::identity(),
            &observations,
            intrinsics,
            &[0, invalid_index],
        )
        .expect_err("invalid inlier index must not be skipped");

        assert!(matches!(
            error,
            PnpRefinementFallback::InvalidInlierIndex {
                index,
                observation_count,
                ..
            } if index == invalid_index && observation_count == observations.len()
        ));
    }

    #[test]
    fn refine_pose_rejects_nonprojectable_inliers_instead_of_dropping_them() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let observation = Observation::try_new(
            Point3 {
                x: 0.0,
                y: 0.0,
                z: -1.0,
            },
            Keypoint { x: 320.0, y: 240.0 },
            intrinsics,
        )
        .expect("finite observation");

        let error = refine_pose_on_inliers(Pose::identity(), &[observation], intrinsics, &[0])
            .expect_err("behind-camera inlier must fail refinement");

        assert!(matches!(
            error,
            PnpRefinementFallback::NonProjectableInliers {
                iteration,
                count,
            } if iteration.get() == 1 && count.get() == 1
        ));
    }

    #[test]
    fn pnp_refinement_fallback_preserves_nested_solver_source() {
        let error = PnpRefinementFallback::LinearSolve {
            iteration: NonZeroUsize::MIN,
            source: crate::LinearSolveError::SingularPivot { column: 2 },
        };
        assert!(std::error::Error::source(&error).is_some());

        let projection = PnpRefinementFallback::CandidateProjection {
            iteration: NonZeroUsize::MIN,
            observation_index: 3,
            source: PinholeProjectionError::NonFiniteCameraPointMeters {
                axis: CameraFrameAxis::Z,
                value: f64::NAN,
            },
        };
        let source = std::error::Error::source(&projection).expect("projection source");
        assert!(projection.to_string().contains("observation 3"));
        assert!(source.to_string().contains("camera-frame z"));

        let post_refinement = PnpRefinementFallback::PostRefinementConsensusProjection {
            termination: PnpRefinementTermination::Converged {
                iterations: NonZeroUsize::new(2).expect("nonzero literal"),
            },
            observation_index: 4,
            source: PinholeProjectionError::NonFinitePixelCoordinatePx {
                axis: ImagePlaneAxis::U,
                value: f64::INFINITY,
            },
        };
        assert_eq!(post_refinement.iterations().map(NonZeroUsize::get), Some(2));
        assert!(post_refinement.to_string().contains("post-refinement"));
        assert!(post_refinement.to_string().contains("observation 4"));
        assert!(std::error::Error::source(&post_refinement).is_some());
    }

    #[test]
    fn solve_pnp_ransac_rejects_too_few_points() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 400.0, 400.0, 320.0, 240.0).expect("intrinsics");
        let obs = vec![
            Observation::try_new(
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 3.0,
                },
                Keypoint { x: 320.0, y: 240.0 },
                intrinsics,
            )
            .expect("obs"),
            Observation::try_new(
                Point3 {
                    x: 0.2,
                    y: 0.1,
                    z: 3.5,
                },
                Keypoint { x: 342.0, y: 252.0 },
                intrinsics,
            )
            .expect("obs"),
            Observation::try_new(
                Point3 {
                    x: -0.2,
                    y: 0.2,
                    z: 2.9,
                },
                Keypoint { x: 290.0, y: 266.0 },
                intrinsics,
            )
            .expect("obs"),
        ];

        let err =
            solve_pnp_ransac(&obs, intrinsics, RansacConfig::default()).expect_err("should reject");
        match err {
            PnpError::Rejected(PnpRejection::NotEnoughPoints { required, actual }) => {
                assert_eq!(required, 4);
                assert_eq!(actual, 3);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn solve_pnp_rejects_unreachable_required_inlier_count_before_sampling() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 400.0, 400.0, 320.0, 240.0).expect("intrinsics");
        let world = synthetic_world_points();
        let observations =
            observations_from_projection(Pose::identity(), &world[..MIN_PNP_POINTS], intrinsics)
                .expect("four observations");
        let required_inliers = observations.len() + 1;
        let config =
            RansacConfig::new(10, 2.0, required_inliers, 7).expect("globally valid RANSAC config");

        let error = solve_pnp_ransac(&observations, intrinsics, config)
            .expect_err("required consensus larger than input must fail before sampling");
        assert!(matches!(
            error,
            PnpError::Rejected(PnpRejection::InsufficientObservationsForRequiredInliers {
                required_inliers: actual_required,
                observations: actual_observations,
            }) if actual_required == required_inliers && actual_observations == observations.len()
        ));
    }

    #[test]
    fn solve_pnp_reports_every_rejected_degenerate_minimal_sample() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 400.0, 400.0, 320.0, 240.0).expect("intrinsics");
        let world = Point3 {
            x: 0.0,
            y: 0.0,
            z: 4.0,
        };
        let observations: Vec<_> = [
            Keypoint { x: 300.0, y: 220.0 },
            Keypoint { x: 340.0, y: 220.0 },
            Keypoint { x: 300.0, y: 260.0 },
            Keypoint { x: 340.0, y: 260.0 },
        ]
        .into_iter()
        .map(|pixel| Observation::try_new(world, pixel, intrinsics).expect("finite observation"))
        .collect();
        let max_iterations = 7;
        let config = RansacConfig::new(max_iterations, 2.0, 4, 17).expect("RANSAC configuration");

        let error = solve_pnp_ransac(&observations, intrinsics, config)
            .expect_err("coincident world points cannot produce a P3P candidate");
        let PnpError::Rejected(PnpRejection::NoUsableRansacCandidate { rejections }) = &error
        else {
            panic!("unexpected error: {error:?}");
        };
        let minimal_samples = rejections
            .minimal_sample_rejections()
            .expect("minimal-sample rejection summary");
        assert_eq!(minimal_samples.count().get(), max_iterations);
        assert_eq!(
            minimal_samples.first_rejection().ransac_iteration().get(),
            1
        );
        assert_eq!(
            minimal_samples.first_rejection().reason(),
            PnpMinimalSampleRejectionReason::DegenerateWorldTriangle
        );
        assert!(rejections.candidate_projection_rejections().is_none());

        let rejection_source = std::error::Error::source(&error).expect("rejection source");
        let summary_source = rejection_source.source().expect("summary source");
        let sample_source = summary_source.source().expect("indexed sample source");
        let reason_source = sample_source.source().expect("sample reason source");
        assert!(sample_source.to_string().contains("iteration 1"));
        assert!(reason_source.to_string().contains("nondegenerate triangle"));
    }

    #[test]
    fn reprojection_error_is_zero_for_exact_projection() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 400.0, 400.0, 320.0, 240.0).expect("intrinsics");
        let pose = axis_angle_pose([0.0, 0.0, 0.0], [0.05, -0.03, 0.01]);
        let point = Point3 {
            x: 0.3,
            y: -0.1,
            z: 4.2,
        };
        let pixel = project_pixel_from_pose(pose, point, intrinsics);
        let obs = Observation::try_new(point, pixel, intrinsics).expect("obs");
        let ReprojectionResidual::Projectable { residual_sq_px2 } =
            evaluate_reprojection_residual(pose, &obs, intrinsics)
                .expect("finite projectable residual")
        else {
            panic!("positive-depth observation must be projectable");
        };
        assert!(
            residual_sq_px2 < 1e-8,
            "expected exact reprojection, got {residual_sq_px2} px^2"
        );
    }

    #[test]
    fn reprojection_residuals_are_zero_for_exact_projection() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let pose = Pose::identity();
        let world = synthetic_world_points();
        let observations =
            observations_from_projection(pose, &world, intrinsics).expect("synthetic observations");
        let residuals_px = reprojection_residuals_px(&pose, &observations, intrinsics)
            .expect("reprojection residuals");
        assert_eq!(residuals_px.len(), observations.len());
        assert!(
            residuals_px
                .iter()
                .all(|residual_px| residual_px.is_some_and(|value_px| value_px < 1e-4))
        );
    }

    #[test]
    fn reprojection_residuals_mark_nonpositive_camera_depth_unprojectable() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let observation = Observation::try_new(
            Point3 {
                x: 0.0,
                y: 0.0,
                z: 2.0,
            },
            Keypoint { x: 320.0, y: 240.0 },
            intrinsics,
        )
        .expect("observation");
        let behind_pose = Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0, 0.0, -3.0],
        );
        let residuals_px = reprojection_residuals_px(&behind_pose, &[observation], intrinsics)
            .expect("finite nonprojectable observation");
        assert_eq!(residuals_px, vec![None]);
    }

    #[test]
    fn reprojection_residual_retains_f64_until_metric_boundary() {
        let intrinsics =
            PinholeIntrinsics::try_new(1.0, 1.0, 0.0, 0.0).expect("finite unit intrinsics");
        let observation = Observation {
            world: Point3 {
                x: f32::MAX,
                y: f32::MAX,
                z: 1.0,
            },
            pixel: Keypoint {
                x: -f32::MAX,
                y: -f32::MAX,
            },
            bearing: [0.0, 0.0, 1.0],
        };

        let ReprojectionResidual::Projectable {
            residual_sq_px2: squared_px2,
        } = evaluate_reprojection_residual(Pose::identity(), &observation, intrinsics)
            .expect("finite projection")
        else {
            panic!("positive camera depth must be projectable");
        };
        assert!(squared_px2.is_finite());
        assert!(squared_px2 > f64::from(f32::MAX).powi(2));

        let residuals_px = reprojection_residuals_px(&Pose::identity(), &[observation], intrinsics)
            .expect("finite f64 residual");
        let value_px = residuals_px[0].expect("projectable residual");
        assert!(value_px.is_finite() && value_px > f64::from(f32::MAX));
        let error = reprojection_rmse_px(&residuals_px)
            .expect_err("unrepresentable f32 metric must remain explicit");
        assert!(matches!(
            error,
            ReprojectionEvaluationError::MetricOutsideF32PixelDomain {
                metric: ReprojectionMetric::Rmse,
                value_px,
            } if value_px.is_finite() && value_px > f64::from(f32::MAX)
        ));
    }

    #[test]
    fn reprojection_evaluation_allocation_error_preserves_source() {
        let mut allocation_probe = Vec::<u8>::new();
        let source = allocation_probe
            .try_reserve_exact(usize::MAX)
            .expect_err("unaddressable allocation must fail");
        let error = ReprojectionEvaluationError::Allocation {
            observation_count: usize::MAX,
            source,
        };

        let allocation_source = std::error::Error::source(&error).expect("allocation source");
        assert!(error.to_string().contains(&usize::MAX.to_string()));
        assert!(!allocation_source.to_string().is_empty());
        assert!(allocation_source.source().is_none());
    }

    #[test]
    fn reprojection_rmse_matches_manual() {
        let residuals_px = vec![Some(3.0), None, Some(4.0)];
        let rmse = reprojection_rmse_px(&residuals_px)
            .expect("representable rmse")
            .expect("rmse");
        let expected = ((3.0_f32 * 3.0 + 4.0 * 4.0) / 2.0).sqrt();
        assert!((rmse - expected).abs() < 1e-6);
    }

    #[test]
    fn reprojection_statistics_do_not_overflow_on_finite_f32_residuals() {
        let residuals_px = [Some(f64::from(f32::MAX)), Some(f64::from(f32::MAX))];

        let rmse = reprojection_rmse_px(&residuals_px)
            .expect("representable RMSE")
            .expect("finite RMSE");
        assert_eq!(rmse, f32::MAX);

        let mse = reprojection_mse_per_axis_px2(&residuals_px).expect("finite MSE");
        assert!(mse.is_finite());
        assert_eq!(mse, f64::from(f32::MAX).powi(2) / 2.0);
    }

    #[test]
    fn reprojection_mse_per_axis_px2_matches_manual() {
        let residuals_px = vec![Some(3.0), None, Some(4.0)];
        let mse = reprojection_mse_per_axis_px2(&residuals_px).expect("mse");
        let expected = ((3.0_f64 * 3.0 + 4.0 * 4.0) / 2.0) / 2.0;
        assert!((mse - expected).abs() < 1e-12);
    }

    #[test]
    fn reprojection_residuals_match_known_pixel_perturbation() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let pose = Pose::identity();
        let world = [
            Point3 {
                x: -0.2,
                y: 0.0,
                z: 3.0,
            },
            Point3 {
                x: 0.2,
                y: 0.1,
                z: 3.5,
            },
            Point3 {
                x: 0.0,
                y: -0.2,
                z: 4.0,
            },
        ];
        let observations: Vec<_> = world
            .iter()
            .map(|&point| {
                let mut pixel = project_pixel_from_pose(pose, point, intrinsics);
                pixel.x += 2.0;
                Observation::try_new(point, pixel, intrinsics).expect("observation")
            })
            .collect();
        let residuals_px = reprojection_residuals_px(&pose, &observations, intrinsics)
            .expect("reprojection residuals");
        let rmse = reprojection_rmse_px(&residuals_px)
            .expect("representable rmse")
            .expect("rmse");
        assert!((1.5..=2.5).contains(&rmse), "rmse={rmse}");
        let max = reprojection_max_px(&residuals_px)
            .expect("representable maximum")
            .expect("max");
        assert!((1.5..=2.5).contains(&max), "max={max}");
    }

    #[test]
    fn reprojection_residual_count_matches_observation_count() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let pose = Pose::identity();
        let observations = vec![
            Observation::try_new(
                Point3 {
                    x: -0.1,
                    y: 0.0,
                    z: 2.5,
                },
                Keypoint { x: 300.0, y: 240.0 },
                intrinsics,
            )
            .expect("observation"),
            Observation::try_new(
                Point3 {
                    x: 0.1,
                    y: 0.0,
                    z: 3.0,
                },
                Keypoint { x: 340.0, y: 240.0 },
                intrinsics,
            )
            .expect("observation"),
        ];
        let residuals_px = reprojection_residuals_px(&pose, &observations, intrinsics)
            .expect("reprojection residuals");
        assert_eq!(residuals_px.len(), observations.len());
    }

    fn project_pixel_from_pose(
        pose_world_to_camera: Pose,
        point_world: Point3,
        intrinsics: PinholeIntrinsics,
    ) -> Keypoint {
        let pc = math::transform_point(
            pose_world_to_camera.rotation(),
            pose_world_to_camera.translation(),
            [point_world.x, point_world.y, point_world.z],
        );
        Keypoint {
            x: intrinsics.fx() * (pc[0] / pc[2]) + intrinsics.cx(),
            y: intrinsics.fy() * (pc[1] / pc[2]) + intrinsics.cy(),
        }
    }
}
