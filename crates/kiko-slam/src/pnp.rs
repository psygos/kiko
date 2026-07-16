use crate::dataset::CameraIntrinsics;
use std::marker::PhantomData;
use std::num::NonZeroU64;

use crate::{
    CameraFrame, CoordinateFrame, Keyframe, Keypoint, Matches, Verified, WorldFrame, WorldPoint3,
    math,
};

const POSE_ROTATION_VALIDATION_TOLERANCE: f64 = 1e-6;

/// Finite pinhole projection coefficients, expressed in pixels.
///
/// Camera coordinates follow the image convention used throughout Kiko: `+x`
/// points right, `+y` points down, and `+z` points forward.
#[derive(Clone, Copy, Debug)]
pub struct PinholeIntrinsics {
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
}

#[derive(Debug)]
pub enum IntrinsicsError {
    NonFinite { fx: f32, fy: f32, cx: f32, cy: f32 },
    NonPositiveFocal { fx: f32, fy: f32 },
}

impl std::fmt::Display for IntrinsicsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IntrinsicsError::NonFinite { fx, fy, cx, cy } => write!(
                f,
                "pinhole intrinsics must be finite (fx={fx}, fy={fy}, cx={cx}, cy={cy})"
            ),
            IntrinsicsError::NonPositiveFocal { fx, fy } => {
                write!(
                    f,
                    "pinhole intrinsics require fx, fy > 0 (fx={fx}, fy={fy})"
                )
            }
        }
    }
}

impl std::error::Error for IntrinsicsError {}

/// Parse the projection coefficients represented by a serialized camera.
///
/// Image dimensions are deliberately outside [`PinholeIntrinsics`]; callers
/// that own a complete image contract must parse those into a dimensions type
/// separately.
impl TryFrom<&CameraIntrinsics> for PinholeIntrinsics {
    type Error = IntrinsicsError;

    fn try_from(value: &CameraIntrinsics) -> Result<Self, Self::Error> {
        Self::try_new(value.fx, value.fy, value.cx, value.cy)
    }
}

impl PinholeIntrinsics {
    /// Parse finite projection coefficients expressed in pixels.
    pub fn try_new(fx: f32, fy: f32, cx: f32, cy: f32) -> Result<Self, IntrinsicsError> {
        if !fx.is_finite() || !fy.is_finite() || !cx.is_finite() || !cy.is_finite() {
            return Err(IntrinsicsError::NonFinite { fx, fy, cx, cy });
        }
        if fx <= 0.0 || fy <= 0.0 {
            return Err(IntrinsicsError::NonPositiveFocal { fx, fy });
        }
        Ok(Self { fx, fy, cx, cy })
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

/// A finite correspondence between a world-frame point and an image pixel.
///
/// This type is deliberately calibration-neutral. A PnP solve applies one
/// authoritative [`PinholeIntrinsics`] value to every observation, so a ray
/// cannot be generated with one camera model and scored with another.
#[derive(Clone, Copy, Debug)]
pub struct Observation {
    world: WorldPoint3,
    pixel: Keypoint,
}

/// Failure to parse weakly typed coordinates into an [`Observation`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ObservationError {
    InvalidWorldPoint(crate::Point3Error),
    NonFinitePixel { axis: usize, value: f32 },
}

impl std::fmt::Display for ObservationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidWorldPoint(err) => write!(f, "invalid observation world point: {err}"),
            Self::NonFinitePixel { axis, value } => write!(
                f,
                "observation pixel coordinate on axis {axis} must be finite, got {value}"
            ),
        }
    }
}

impl std::error::Error for ObservationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidWorldPoint(err) => Some(err),
            Self::NonFinitePixel { .. } => None,
        }
    }
}

impl Observation {
    pub fn world(&self) -> WorldPoint3 {
        self.world
    }

    pub fn pixel(&self) -> Keypoint {
        self.pixel
    }

    pub fn try_new(world: WorldPoint3, pixel: Keypoint) -> Result<Self, ObservationError> {
        let world = world
            .validate()
            .map_err(ObservationError::InvalidWorldPoint)?;
        for (axis, value) in [pixel.x, pixel.y].into_iter().enumerate() {
            if !value.is_finite() {
                return Err(ObservationError::NonFinitePixel { axis, value });
            }
        }
        Ok(Self { world, pixel })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Pose {
    rotation: [[f32; 3]; 3],
    translation: [f32; 3],
}

/// Failure to construct or apply a finite rigid-body pose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PoseError {
    NonFiniteRotation {
        row: usize,
        column: usize,
        value: f32,
    },
    NonFiniteTranslation {
        axis: usize,
        value: f32,
    },
    NonOrthonormalRotation {
        max_error: f64,
    },
    ImproperRotation {
        determinant: f64,
    },
    ComposeRotationNotRepresentable {
        row: usize,
        column: usize,
        value: f64,
    },
    ComposeTranslationNotRepresentable {
        axis: usize,
        value: f64,
    },
    InverseTranslationNotRepresentable {
        axis: usize,
        value: f64,
    },
}

impl std::fmt::Display for PoseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteRotation { row, column, value } => write!(
                f,
                "pose rotation[{row}][{column}] must be finite, got {value}"
            ),
            Self::NonFiniteTranslation { axis, value } => {
                write!(f, "pose translation[{axis}] must be finite, got {value}")
            }
            Self::NonOrthonormalRotation { max_error } => write!(
                f,
                "pose rotation must be orthonormal (maximum error {max_error})"
            ),
            Self::ImproperRotation { determinant } => write!(
                f,
                "pose rotation determinant must be +1 (got {determinant})"
            ),
            Self::ComposeRotationNotRepresentable { row, column, value } => write!(
                f,
                "pose composition rotation[{row}][{column}] is not representable as a finite f32: {value}"
            ),
            Self::ComposeTranslationNotRepresentable { axis, value } => write!(
                f,
                "pose composition translation[{axis}] is not representable as a finite f32: {value}"
            ),
            Self::InverseTranslationNotRepresentable { axis, value } => write!(
                f,
                "pose inversion translation[{axis}] is not representable as a finite f32: {value}"
            ),
        }
    }
}

impl std::error::Error for PoseError {}

/// Failure to apply a frame-typed transform to a point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TransformError {
    InvalidInput(crate::geometry::Point3Error),
    OutputNotRepresentable { axis: usize, value: f64 },
}

impl std::fmt::Display for TransformError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(err) => write!(f, "invalid transform input: {err}"),
            Self::OutputNotRepresentable { axis, value } => write!(
                f,
                "transformed point coordinate on axis {axis} is not representable as a finite f32: {value}"
            ),
        }
    }
}

impl std::error::Error for TransformError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidInput(err) => Some(err),
            Self::OutputNotRepresentable { .. } => None,
        }
    }
}

impl Pose {
    pub fn identity() -> Self {
        Self {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
        }
    }

    pub fn try_from_rt(rotation: [[f32; 3]; 3], translation: [f32; 3]) -> Result<Self, PoseError> {
        for (row, values) in rotation.iter().enumerate() {
            for (column, &value) in values.iter().enumerate() {
                if !value.is_finite() {
                    return Err(PoseError::NonFiniteRotation { row, column, value });
                }
            }
        }
        if let Some(axis) = translation.iter().position(|value| !value.is_finite()) {
            return Err(PoseError::NonFiniteTranslation {
                axis,
                value: translation[axis],
            });
        }

        let rotation_f64 = rotation.map(|row| row.map(f64::from));
        let mut max_error = 0.0_f64;
        for row in 0..3 {
            for column in 0..3 {
                let dot = (0..3)
                    .map(|index| rotation_f64[index][row] * rotation_f64[index][column])
                    .sum::<f64>();
                let expected = if row == column { 1.0 } else { 0.0 };
                max_error = max_error.max((dot - expected).abs());
            }
        }
        if max_error > POSE_ROTATION_VALIDATION_TOLERANCE {
            return Err(PoseError::NonOrthonormalRotation { max_error });
        }

        let determinant = rotation_determinant(rotation_f64);
        if (determinant - 1.0).abs() > POSE_ROTATION_VALIDATION_TOLERANCE {
            return Err(PoseError::ImproperRotation { determinant });
        }

        Ok(Self {
            rotation,
            translation,
        })
    }

    /// Construct a deliberately unchecked pose for malformed-input tests.
    #[cfg(test)]
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

    pub fn try_inverse(self) -> Result<Pose, PoseError> {
        let r_t = math::mat_transpose(self.rotation);
        let rotated = mat3_mul_vec_f64(
            r_t.map(|row| row.map(f64::from)),
            self.translation.map(f64::from),
        );
        let mut translation = [0.0_f32; 3];
        for (axis, output) in translation.iter_mut().enumerate() {
            let value = -rotated[axis];
            *output = narrow_pose_component(value)
                .ok_or(PoseError::InverseTranslationNotRepresentable { axis, value })?;
        }
        Self::try_from_rt(r_t, translation)
    }

    /// Compose two poses: `self ∘ other`.
    pub fn try_compose(self, other: Pose) -> Result<Pose, PoseError> {
        let rotation_f64 = mat3_mul_f64(
            self.rotation.map(|row| row.map(f64::from)),
            other.rotation.map(|row| row.map(f64::from)),
        );
        let mut rotation = [[0.0_f32; 3]; 3];
        for (row, values) in rotation.iter_mut().enumerate() {
            for (column, output) in values.iter_mut().enumerate() {
                let value = rotation_f64[row][column];
                *output = narrow_pose_component(value)
                    .ok_or(PoseError::ComposeRotationNotRepresentable { row, column, value })?;
            }
        }

        let rotated_translation = mat3_mul_vec_f64(
            self.rotation.map(|row| row.map(f64::from)),
            other.translation.map(f64::from),
        );
        let mut translation = [0.0_f32; 3];
        for (axis, output) in translation.iter_mut().enumerate() {
            let value = rotated_translation[axis] + f64::from(self.translation[axis]);
            *output = narrow_pose_component(value)
                .ok_or(PoseError::ComposeTranslationNotRepresentable { axis, value })?;
        }
        Self::try_from_rt(rotation, translation)
    }
}

fn narrow_pose_component(value: f64) -> Option<f32> {
    let narrowed = value as f32;
    (value.is_finite() && narrowed.is_finite()).then_some(narrowed)
}

fn mat3_mul_vec_f64(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    matrix.map(|row| row[0].mul_add(vector[0], row[1].mul_add(vector[1], row[2] * vector[2])))
}

fn mat3_mul_f64(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let right_t = [
        [right[0][0], right[1][0], right[2][0]],
        [right[0][1], right[1][1], right[2][1]],
        [right[0][2], right[1][2], right[2][2]],
    ];
    left.map(|row| {
        right_t
            .map(|column| row[0].mul_add(column[0], row[1].mul_add(column[1], row[2] * column[2])))
    })
}

fn rotation_determinant(rotation: [[f64; 3]; 3]) -> f64 {
    rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
}

/// A rigid transform whose source and destination frames are checked at compile time.
#[derive(Clone, Copy, Debug)]
pub struct Transform<From: CoordinateFrame, To: CoordinateFrame> {
    pose: Pose,
    frames: PhantomData<fn(From) -> To>,
}

impl<From, To> Transform<From, To>
where
    From: CoordinateFrame<Scalar = f32>,
    To: CoordinateFrame<Scalar = f32>,
{
    /// Wrap an untyped pose at a legacy subsystem boundary.
    pub(crate) fn from_legacy_pose(pose: Pose) -> Self {
        Self {
            pose,
            frames: PhantomData,
        }
    }

    /// Unwrap this transform for a subsystem that has not yet adopted frame types.
    pub fn into_legacy_pose(self) -> Pose {
        self.pose
    }

    pub fn rotation(self) -> [[f32; 3]; 3] {
        self.pose.rotation()
    }

    pub fn translation(self) -> [f32; 3] {
        self.pose.translation()
    }

    pub fn try_transform_point(
        self,
        point: crate::Point3<From>,
    ) -> Result<crate::Point3<To>, TransformError> {
        let point = point.validate().map_err(TransformError::InvalidInput)?;
        let transformed = mat3_mul_vec_f64(
            self.pose.rotation.map(|row| row.map(f64::from)),
            point.to_array().map(f64::from),
        );
        let translation = self.pose.translation.map(f64::from);
        let mut output = [0.0_f32; 3];
        for (axis, destination) in output.iter_mut().enumerate() {
            let value = transformed[axis] + translation[axis];
            *destination = narrow_pose_component(value)
                .ok_or(TransformError::OutputNotRepresentable { axis, value })?;
        }
        Ok(crate::Point3::from_array(output))
    }

    pub fn try_inverse(self) -> Result<Transform<To, From>, PoseError> {
        self.pose.try_inverse().map(Transform::from_legacy_pose)
    }

    /// Compose `self` after `other`, preserving both endpoint frames.
    pub fn try_compose<Source>(
        self,
        other: Transform<Source, From>,
    ) -> Result<Transform<Source, To>, PoseError>
    where
        Source: CoordinateFrame<Scalar = f32>,
    {
        self.pose
            .try_compose(other.pose)
            .map(Transform::from_legacy_pose)
    }
}

impl<Frame> Transform<Frame, Frame>
where
    Frame: CoordinateFrame<Scalar = f32>,
{
    pub fn identity() -> Self {
        Self::from_legacy_pose(Pose::identity())
    }
}

pub type WorldToCamera = Transform<WorldFrame, CameraFrame>;
pub type CameraToWorld = Transform<CameraFrame, WorldFrame>;

impl WorldToCamera {
    /// The coincident world-to-camera transform used for the initial keyframe.
    pub fn identity() -> Self {
        Self::from_legacy_pose(Pose::identity())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RansacConfig {
    max_iterations: usize,
    reprojection_threshold_px: f32,
    min_inliers: usize,
    seed: NonZeroU64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RansacConfigError {
    ZeroIterations,
    ZeroSeed,
    InvalidReprojectionThreshold { value: f32 },
    TooFewInliers { value: usize, minimum: usize },
}

impl std::fmt::Display for RansacConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroIterations => write!(f, "RANSAC iterations must be greater than zero"),
            Self::ZeroSeed => write!(f, "RANSAC seed must be nonzero"),
            Self::InvalidReprojectionThreshold { value } => write!(
                f,
                "RANSAC reprojection threshold must be positive and finite, got {value}"
            ),
            Self::TooFewInliers { value, minimum } => write!(
                f,
                "RANSAC minimum inliers must be at least {minimum}, got {value}"
            ),
        }
    }
}

impl std::error::Error for RansacConfigError {}

impl RansacConfig {
    pub fn try_new(
        max_iterations: usize,
        reprojection_threshold_px: f32,
        min_inliers: usize,
        seed: u64,
    ) -> Result<Self, RansacConfigError> {
        if max_iterations == 0 {
            return Err(RansacConfigError::ZeroIterations);
        }
        let seed = NonZeroU64::new(seed).ok_or(RansacConfigError::ZeroSeed)?;
        if !reprojection_threshold_px.is_finite() || reprojection_threshold_px <= 0.0 {
            return Err(RansacConfigError::InvalidReprojectionThreshold {
                value: reprojection_threshold_px,
            });
        }
        if min_inliers < MIN_PNP_POINTS {
            return Err(RansacConfigError::TooFewInliers {
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
        self.max_iterations
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

    pub(crate) fn with_min_inliers(mut self, min_inliers: usize) -> Self {
        assert!(min_inliers >= MIN_PNP_POINTS);
        self.min_inliers = min_inliers;
        self
    }
}

impl Default for RansacConfig {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            reprojection_threshold_px: 2.0,
            min_inliers: 20,
            seed: NonZeroU64::new(0x5EED_u64).expect("default RANSAC seed is nonzero"),
        }
    }
}

#[derive(Debug)]
pub struct PnpResult {
    /// Estimated transform from the observations' world frame into the camera frame.
    /// Translation uses the same length unit as the supplied world points.
    pub pose: WorldToCamera,
    /// Zero-based indices into the exact observation slice passed to
    /// [`solve_pnp_ransac`].
    pub inliers: Vec<usize>,
    /// Three-point RANSAC sampling iterations performed. The current
    /// fixed-budget solver always reports [`RansacConfig::max_iterations`].
    pub iterations: usize,
}

#[derive(Debug)]
pub enum PnpError {
    NotEnoughPoints {
        required: usize,
        actual: usize,
    },
    IndexOutOfBounds {
        current_len: usize,
        keyframe_len: usize,
        current_index: usize,
        keyframe_index: usize,
    },
    MissingLandmark {
        keyframe_index: usize,
    },
    Observation(ObservationError),
    Numerical {
        operation: &'static str,
        value: f64,
    },
    Map(crate::map::MapError),
    Transform(TransformError),
    NoSolution,
}

impl std::fmt::Display for PnpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PnpError::NotEnoughPoints { required, actual } => {
                write!(f, "pnp requires at least {required} points, got {actual}")
            }
            PnpError::IndexOutOfBounds {
                current_len,
                keyframe_len,
                current_index,
                keyframe_index,
            } => write!(
                f,
                "pnp match index out of bounds: current_index={current_index} (len={current_len}), keyframe_index={keyframe_index} (len={keyframe_len})"
            ),
            PnpError::MissingLandmark { keyframe_index } => write!(
                f,
                "pnp match references missing landmark for keyframe index {keyframe_index}"
            ),
            PnpError::Observation(err) => write!(f, "invalid pnp observation: {err}"),
            PnpError::Numerical { operation, value } => write!(
                f,
                "pnp numerical failure while {operation}: {value} is not representable as a finite f32"
            ),
            PnpError::Map(err) => write!(f, "pnp map observation lookup failed: {err}"),
            PnpError::Transform(err) => write!(f, "pnp coordinate transform failed: {err}"),
            PnpError::NoSolution => write!(f, "pnp failed to find a valid pose"),
        }
    }
}

impl std::error::Error for PnpError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Observation(err) => Some(err),
            Self::Map(err) => Some(err),
            Self::Transform(err) => Some(err),
            _ => None,
        }
    }
}

impl From<ObservationError> for PnpError {
    fn from(error: ObservationError) -> Self {
        Self::Observation(error)
    }
}

impl From<crate::map::MapError> for PnpError {
    fn from(error: crate::map::MapError) -> Self {
        Self::Map(error)
    }
}

impl From<TransformError> for PnpError {
    fn from(error: TransformError) -> Self {
        Self::Transform(error)
    }
}

/// Build finite, calibration-neutral correspondences in verified-match order.
pub fn build_observations(
    keyframe: &Keyframe,
    matches: &Matches<Verified>,
    keyframe_to_world: CameraToWorld,
) -> Result<Vec<Observation>, PnpError> {
    let current = matches.source_a();
    if current.is_empty() || keyframe.landmarks().is_empty() {
        return Err(PnpError::NotEnoughPoints {
            required: 4,
            actual: 0,
        });
    }

    if matches.len() < MIN_PNP_POINTS {
        return Err(PnpError::NotEnoughPoints {
            required: MIN_PNP_POINTS,
            actual: matches.len(),
        });
    }

    let current_len = current.len();
    let keyframe_len = keyframe.detections().len();

    let mut observations = Vec::with_capacity(matches.len());
    for &(ci, ki) in matches.indices() {
        if ci >= current_len || ki >= keyframe_len {
            return Err(PnpError::IndexOutOfBounds {
                current_len,
                keyframe_len,
                current_index: ci,
                keyframe_index: ki,
            });
        }

        let pixel = current.keypoints()[ci];
        let camera = keyframe
            .landmark_for_detection(ki)
            .ok_or(PnpError::MissingLandmark { keyframe_index: ki })?;
        let world = keyframe_to_world.try_transform_point(camera)?;
        observations.push(Observation::try_new(world, pixel)?);
    }

    Ok(observations)
}

/// Estimate one world-to-camera pose from finite 2D-3D correspondences.
///
/// `intrinsics` is the sole camera model used both to construct P3P rays and
/// to score pixel reprojection residuals. Returned inlier indices refer to
/// `observations`.
pub fn solve_pnp_ransac(
    observations: &[Observation],
    intrinsics: PinholeIntrinsics,
    config: RansacConfig,
) -> Result<PnpResult, PnpError> {
    if observations.len() < MIN_PNP_POINTS {
        return Err(PnpError::NotEnoughPoints {
            required: MIN_PNP_POINTS,
            actual: observations.len(),
        });
    }
    if config.min_inliers > observations.len() {
        return Err(PnpError::NotEnoughPoints {
            required: config.min_inliers,
            actual: observations.len(),
        });
    }

    let problem = CalibratedPnpProblem::parse(observations, intrinsics);

    let mut rng = XorShift64::new(config.seed);
    let mut best_pose = None;
    // Grow lazily so degenerate inputs do not reserve two full observation-sized buffers before
    // P3P has produced a candidate. Capacities are still reused across every candidate.
    let mut best_inliers = Vec::new();
    let mut candidate_inliers = Vec::new();

    let mut iterations = 0usize;
    let total = observations.len();

    let threshold = ReprojectionThresholdPx::from_config(config);
    while iterations < config.max_iterations {
        iterations += 1;
        let sample = sample_three(&mut rng, total);
        let Some([a, b, c]) = sample else { continue };

        let candidates = p3p_solutions(problem.sample([a, b, c]));
        if candidates.is_empty() {
            continue;
        }

        for pose in candidates {
            candidate_inliers.clear();
            for idx in 0..problem.len() {
                if let ReprojectionOutcome::Projected(residual) =
                    problem.reprojection_residual_px(pose, idx)
                    && residual.is_within_threshold(threshold)
                {
                    candidate_inliers.push(idx);
                }
            }

            if candidate_inliers.len() > best_inliers.len() {
                std::mem::swap(&mut best_inliers, &mut candidate_inliers);
                best_pose = Some(pose);
            }
        }
    }

    let pose = best_pose.ok_or(PnpError::NoSolution)?;
    if best_inliers.len() < config.min_inliers {
        return Err(PnpError::NoSolution);
    }

    Ok(PnpResult {
        pose: WorldToCamera::from_legacy_pose(pose),
        inliers: best_inliers,
        iterations,
    })
}

/// Build correspondences from verified matches and estimate their pose.
pub fn solve_pnp(
    keyframe: &Keyframe,
    matches: &Matches<Verified>,
    keyframe_to_world: CameraToWorld,
    intrinsics: PinholeIntrinsics,
    config: RansacConfig,
) -> Result<PnpResult, PnpError> {
    let observations = build_observations(keyframe, matches, keyframe_to_world)?;
    solve_pnp_ransac(&observations, intrinsics, config)
}

#[derive(Clone, Copy, Debug)]
struct UnitBearing([f64; 3]);

impl UnitBearing {
    fn from_pixel(pixel: Keypoint, intrinsics: PinholeIntrinsics) -> Self {
        let x = (f64::from(pixel.x) - f64::from(intrinsics.cx())) / f64::from(intrinsics.fx());
        let y = (f64::from(pixel.y) - f64::from(intrinsics.cy())) / f64::from(intrinsics.fy());
        let norm = x.hypot(y).hypot(1.0);

        // Pixel coordinates and intrinsics have already been parsed as finite
        // f32 values and both focal lengths are positive. Their largest
        // possible quotient is below 2^279, so `norm` is finite and at least
        // one; the normalized forward component is therefore strictly positive.
        debug_assert!(norm.is_finite() && norm >= 1.0);
        let bearing = [x / norm, y / norm, 1.0 / norm];
        debug_assert!(bearing.into_iter().all(f64::is_finite));
        debug_assert!(bearing[2] > 0.0);
        Self(bearing)
    }

    fn into_array(self) -> [f64; 3] {
        self.0
    }
}

struct CalibratedPnpProblem<'a> {
    observations: &'a [Observation],
    bearings: Vec<UnitBearing>,
    intrinsics: PinholeIntrinsics,
}

impl<'a> CalibratedPnpProblem<'a> {
    fn parse(observations: &'a [Observation], intrinsics: PinholeIntrinsics) -> Self {
        let bearings = observations
            .iter()
            .map(|observation| UnitBearing::from_pixel(observation.pixel, intrinsics))
            .collect();
        Self {
            observations,
            bearings,
            intrinsics,
        }
    }

    fn len(&self) -> usize {
        self.observations.len()
    }

    fn sample(&self, indices: [usize; 3]) -> [CalibratedPnpObservation<'_>; 3] {
        indices.map(|index| CalibratedPnpObservation {
            observation: &self.observations[index],
            bearing: self.bearings[index],
        })
    }

    fn reprojection_residual_px(&self, pose: Pose, index: usize) -> ReprojectionOutcome {
        reprojection_residual_px(pose, &self.observations[index], self.intrinsics)
    }
}

#[derive(Clone, Copy)]
struct CalibratedPnpObservation<'a> {
    observation: &'a Observation,
    bearing: UnitBearing,
}

#[cfg(test)]
fn normalize_bearing(pixel: Keypoint, intrinsics: PinholeIntrinsics) -> [f64; 3] {
    UnitBearing::from_pixel(pixel, intrinsics).into_array()
}

fn p3p_solutions(obs: [CalibratedPnpObservation<'_>; 3]) -> Vec<Pose> {
    let p1 = vec3_from_world_point(obs[0].observation.world);
    let p2 = vec3_from_world_point(obs[1].observation.world);
    let p3 = vec3_from_world_point(obs[2].observation.world);
    let f1 = obs[0].bearing.into_array();
    let f2 = obs[1].bearing.into_array();
    let f3 = obs[2].bearing.into_array();

    // Compute the scene scale in f64 before normalization. Squaring f32 world
    // coordinates here used to overflow above roughly 1e19 and underflow below
    // roughly 1e-22, even though both scenes remain representable in f32.
    let Some(a) = norm(sub(p2, p3)) else {
        return Vec::new();
    };
    let Some(b) = norm(sub(p1, p3)) else {
        return Vec::new();
    };
    let Some(c) = norm(sub(p1, p2)) else {
        return Vec::new();
    };

    // Bearings are parsed once from finite pixels and normalized in f64.
    // Clamp their dot products to absorb only the final f64 rounding.
    let cos_alpha = dot(f2, f3).clamp(-1.0, 1.0);
    let cos_beta = dot(f1, f3).clamp(-1.0, 1.0);
    let cos_gamma = dot(f1, f2).clamp(-1.0, 1.0);

    let mut solutions = Vec::new();
    let mut roots = Vec::new();
    find_roots(cos_alpha, cos_beta, cos_gamma, a, b, c, &mut roots);

    for (x, y) in roots {
        let denom = 1.0 + x * x - 2.0 * x * cos_gamma;
        if !denom.is_finite() || denom <= 0.0 {
            continue;
        }
        let d1 = c / denom.sqrt();
        let d2 = x * d1;
        let d3 = y * d1;
        if [d1, d2, d3]
            .into_iter()
            .any(|distance| !distance.is_finite() || distance <= 0.0)
        {
            continue;
        }

        let c1 = mul(f1, d1);
        let c2 = mul(f2, d2);
        let c3 = mul(f3, d3);

        if let Some(pose) = pose_from_points(p1, p2, p3, c1, c2, c3) {
            solutions.push(pose);
        }
    }

    solutions
}

fn find_roots(
    cos_alpha: f64,
    cos_beta: f64,
    cos_gamma: f64,
    a: f64,
    b: f64,
    c: f64,
    roots: &mut Vec<(f64, f64)>,
) {
    let scene_scale = a.max(b).max(c);
    if !scene_scale.is_finite() || scene_scale <= 0.0 {
        return;
    }
    // P3P depends only on ratios between the three world-space distances.
    // Normalize them so polynomial trimming and root validation are unitless.
    let a = a / scene_scale;
    let b = b / scene_scale;
    let c = c / scene_scale;
    let coeffs_meta = P3pCoeffs {
        cos_alpha,
        cos_beta,
        cos_gamma,
        a,
        b,
        c,
    };
    let coeffs = quartic_coeffs(cos_alpha, cos_beta, cos_gamma, a, b, c);
    let Some(coeffs) = coeffs else {
        return;
    };

    let xs = solve_real_roots(coeffs);
    for x in xs {
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
                push_unique_root(roots, (x, y));
            }
        }
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
) -> Option<[f64; 5]> {
    if [cos_alpha, cos_beta, cos_gamma, a, b, c]
        .into_iter()
        .any(|value| !value.is_finite())
        || a <= 0.0
        || b <= 0.0
        || c <= 0.0
    {
        return None;
    }
    let a2 = a * a;
    let b2 = b * b;
    let c2 = c * c;
    if !a2.is_finite() || !b2.is_finite() || !c2.is_finite() || c2 <= 0.0 {
        return None;
    }

    let ca = cos_alpha;
    let cb = cos_beta;
    let cg = cos_gamma;

    let n0 = a2 - b2 + c2;
    let n1 = -2.0 * (a2 - b2) * cg;
    let n2 = a2 - b2 - c2;
    let n = [n0, n1, n2];

    let d0 = 2.0 * c2 * cb;
    let d1 = -2.0 * c2 * ca;
    let d = [d0, d1];

    let k_scale = b2 / c2;
    let k0 = 1.0 - k_scale;
    let k1 = 2.0 * k_scale * cg;
    let k2 = -k_scale;
    let k = [k0, k1, k2];

    let n2_poly = poly_mul(&n, &n);
    let nd_poly = poly_mul(&n, &d);
    let d2_poly = poly_mul(&d, &d);
    let kd2_poly = poly_mul(&k, &d2_poly);

    let mut p = vec![0.0_f64; 5];
    add_scaled(&mut p, &n2_poly, 1.0);
    add_scaled(&mut p, &nd_poly, -2.0 * cb);
    add_scaled(&mut p, &kd2_poly, 1.0);

    Some([p[0], p[1], p[2], p[3], p[4]])
}

fn poly_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
    out
}

fn add_scaled(dst: &mut [f64], src: &[f64], scale: f64) {
    for (i, &v) in src.iter().enumerate() {
        dst[i] += scale * v;
    }
}

/// Minimum number of point correspondences required for PnP solving (geometric minimum for P3P).
pub(crate) const MIN_PNP_POINTS: usize = 4;

/// Relative tolerance for treating polynomial coefficients as zero during trimming.
const POLY_RELATIVE_COEFFICIENT_TOLERANCE: f64 = 1e-12;
/// Maximum imaginary component for a root to be considered real.
const IMAGINARY_TOLERANCE: f64 = 1e-6;
/// Convergence threshold for the Durand-Kerner root-finding iterations.
const ROOT_CONVERGENCE_THRESHOLD: f64 = 1e-10;
/// Denominator magnitude below which a Durand-Kerner correction is skipped.
const ROOT_DENOMINATOR_TOLERANCE: f64 = 1e-12;
/// Maximum iterations for the Durand-Kerner root-finding algorithm.
const MAX_ROOT_ITERATIONS: usize = 64;
/// Tolerance for detecting duplicate P3P root solutions.
const ROOT_UNIQUENESS_TOLERANCE: f64 = 1e-3;
/// Dimensionless tolerance for accepting a normalized P3P equation root.
const P3P_ROOT_TOLERANCE: f64 = 1e-3;

fn solve_real_roots(coeffs: [f64; 5]) -> Vec<f64> {
    let mut coeffs: Vec<f64> = coeffs.into();
    let coefficient_scale = coeffs.iter().copied().map(f64::abs).fold(0.0, f64::max);
    if !coefficient_scale.is_finite() || coefficient_scale == 0.0 {
        return Vec::new();
    }
    let trim_tolerance = POLY_RELATIVE_COEFFICIENT_TOLERANCE * coefficient_scale;
    while coeffs.len() > 1 && coeffs.last().copied().unwrap_or(0.0).abs() <= trim_tolerance {
        coeffs.pop();
    }
    let degree = coeffs.len().saturating_sub(1);
    if degree == 0 {
        return Vec::new();
    }
    if degree == 1 {
        let c1 = coeffs[1];
        if !c1.is_finite() || c1 == 0.0 {
            return Vec::new();
        }
        return vec![-coeffs[0] / c1];
    }

    let Some(&lead) = coeffs.last() else {
        return Vec::new();
    };
    if !lead.is_finite() || lead == 0.0 {
        return Vec::new();
    }
    for c in &mut coeffs {
        *c /= lead;
    }

    let roots = durand_kerner(&coeffs);
    let mut real = Vec::new();
    for r in roots {
        if r.im.abs() < IMAGINARY_TOLERANCE {
            real.push(r.re);
        }
    }
    real
}

#[derive(Clone, Copy, Debug)]
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

impl std::ops::Div for Complex {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        let denom = rhs.re * rhs.re + rhs.im * rhs.im;
        Self::new(
            (self.re * rhs.re + self.im * rhs.im) / denom,
            (self.im * rhs.re - self.re * rhs.im) / denom,
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

fn durand_kerner(coeffs: &[f64]) -> Vec<Complex> {
    let degree = coeffs.len().saturating_sub(1);
    if degree == 0 {
        return Vec::new();
    }

    let radius = 1.0_f64;
    let mut roots = Vec::with_capacity(degree);
    for i in 0..degree {
        let theta = (2.0 * std::f64::consts::PI * i as f64) / degree as f64;
        roots.push(Complex::from_polar(radius, theta));
    }

    for _ in 0..MAX_ROOT_ITERATIONS {
        let mut max_delta = 0.0_f64;
        for i in 0..degree {
            let mut denom = Complex::new(1.0, 0.0);
            for j in 0..degree {
                if i != j {
                    denom = denom * (roots[i] - roots[j]);
                }
            }
            if denom.abs() < ROOT_DENOMINATOR_TOLERANCE {
                continue;
            }
            let p = poly_eval(coeffs, roots[i]);
            let delta = p / denom;
            roots[i] = roots[i] - delta;
            max_delta = max_delta.max(delta.abs());
        }
        if max_delta < ROOT_CONVERGENCE_THRESHOLD {
            break;
        }
    }

    roots
}

fn push_unique_root(roots: &mut Vec<(f64, f64)>, candidate: (f64, f64)) {
    let (x, y) = candidate;
    let tol = ROOT_UNIQUENESS_TOLERANCE;
    if roots
        .iter()
        .any(|(rx, ry)| (rx - x).abs() < tol && (ry - y).abs() < tol)
    {
        return;
    }
    roots.push(candidate);
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ReprojectionOutcome {
    /// Camera-space depth is zero or negative, so pinhole projection is undefined.
    NotInFront,
    Projected(ReprojectionResidualPx),
}

/// A parsed positive finite pixel threshold and its sole derived squared value.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ReprojectionThresholdPx {
    value: f64,
    squared: f64,
}

impl ReprojectionThresholdPx {
    fn from_config(config: RansacConfig) -> Self {
        let value = f64::from(config.reprojection_threshold_px());
        let squared = value * value;
        debug_assert!(value.is_finite() && value > 0.0 && squared.is_finite());
        Self { value, squared }
    }
}

/// Finite signed pixel residuals (`projected - observed`) from parsed f32 inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ReprojectionResidualPx {
    dx_px: f64,
    dy_px: f64,
}

impl ReprojectionResidualPx {
    fn magnitude_px(self) -> f64 {
        self.dx_px.hypot(self.dy_px)
    }

    fn is_within_threshold(self, threshold: ReprojectionThresholdPx) -> bool {
        let abs_dx = self.dx_px.abs();
        let abs_dy = self.dy_px.abs();

        // A component outside the positive threshold cannot be an inlier. Guarding first bounds
        // every remaining square by f32::MAX^2, far below f64 overflow.
        if abs_dx > threshold.value || abs_dy > threshold.value {
            return false;
        }

        // A nonzero orthogonal component makes a residual strictly outside the circle when the
        // other component lies exactly on its boundary, even if that contribution is too small
        // to change a rounded f64 square sum.
        if abs_dx == threshold.value {
            return abs_dy == 0.0;
        }
        if abs_dy == threshold.value {
            return abs_dx == 0.0;
        }

        let dx_squared = abs_dx * abs_dx;
        let dy_squared = abs_dy * abs_dy;
        let threshold_roundoff = threshold.value.mul_add(threshold.value, -threshold.squared);
        let squared_difference = math::compensated_sum([
            dx_squared,
            abs_dx.mul_add(abs_dx, -dx_squared),
            dy_squared,
            abs_dy.mul_add(abs_dy, -dy_squared),
            -threshold.squared,
            -threshold_roundoff,
        ]);
        squared_difference <= 0.0
    }
}

fn compensated_projection_residual_px(
    focal_px: f32,
    camera_axis: f64,
    principal_px: f32,
    observed_px: f32,
    depth: f64,
) -> f64 {
    let focal = f64::from(focal_px);
    let principal = f64::from(principal_px);
    let observed = f64::from(observed_px);
    let offset = principal - observed;
    let observed_virtual = principal - offset;
    let principal_virtual = offset + observed_virtual;
    let offset_roundoff = (principal - principal_virtual) + (observed_virtual - observed);
    let focal_coordinate = focal * camera_axis;
    let offset_depth = offset * depth;
    let offset_roundoff_depth = offset_roundoff * depth;

    // Cancel in the homogeneous numerator before the sole division. The exact two-limb f32
    // principal/observation difference avoids manufacturing two large products, while an
    // expansion preserves every product limb across staged cancellation.
    math::expansion_sum([
        focal_coordinate,
        focal.mul_add(camera_axis, -focal_coordinate),
        offset_depth,
        offset.mul_add(depth, -offset_depth),
        offset_roundoff_depth,
        offset_roundoff.mul_add(depth, -offset_roundoff_depth),
    ]) / depth
}

fn reprojection_residual_px(
    pose: Pose,
    observation: &Observation,
    intrinsics: PinholeIntrinsics,
) -> ReprojectionOutcome {
    let camera = transform_point(
        mat3_from_f32(pose.rotation()),
        vec3_from_f32(pose.translation()),
        vec3_from_world_point(observation.world),
    );
    // PnP cheirality is the geometric z > 0 condition. Local BA deliberately applies its
    // separate 1e-6 m conditioning floor when differentiating a projection.
    if camera[2] <= 0.0 {
        return ReprojectionOutcome::NotInFront;
    }

    let dx_px = compensated_projection_residual_px(
        intrinsics.fx(),
        camera[0],
        intrinsics.cx(),
        observation.pixel.x,
        camera[2],
    );
    let dy_px = compensated_projection_residual_px(
        intrinsics.fy(),
        camera[1],
        intrinsics.cy(),
        observation.pixel.y,
        camera[2],
    );

    // Parsed poses, observations, and intrinsics contain only finite f32 values. A validated
    // rotation bounds each coefficient, and the smallest nonzero exact f32 product is about
    // 2^-298; therefore the complete transform, projection, and residual remain finite in f64.
    debug_assert!(camera.into_iter().all(f64::is_finite));
    debug_assert!(dx_px.is_finite() && dy_px.is_finite());
    ReprojectionOutcome::Projected(ReprojectionResidualPx { dx_px, dy_px })
}

/// One-pass diagnostics over the projected subset of the supplied observations.
///
/// Counts retain nonpositive-depth outcomes instead of disguising them as missing arithmetic.
/// RMSE uses the number of projected observations as its denominator, and both metrics remain in
/// f64 until their public f32 diagnostic boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ReprojectionMetrics {
    values_px: Option<ReprojectionMetricValuesPx>,
    projected_count: usize,
    not_in_front_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ReprojectionMetricValuesPx {
    rmse_px: f64,
    max_px: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CompleteReprojectionMetricsPx {
    rmse_px: f32,
    max_px: f32,
}

impl CompleteReprojectionMetricsPx {
    pub(crate) fn rmse_px(self) -> f32 {
        self.rmse_px
    }

    pub(crate) fn max_px(self) -> f32 {
        self.max_px
    }
}

impl ReprojectionMetrics {
    #[cfg(test)]
    fn rmse_px(self) -> Option<f64> {
        self.values_px.map(|values| values.rmse_px)
    }

    #[cfg(test)]
    fn max_px(self) -> Option<f64> {
        self.values_px.map(|values| values.max_px)
    }

    pub(crate) fn projected_count(self) -> usize {
        self.projected_count
    }

    pub(crate) fn not_in_front_count(self) -> usize {
        self.not_in_front_count
    }

    pub(crate) fn complete_px(self) -> Result<Option<CompleteReprojectionMetricsPx>, PnpError> {
        if self.not_in_front_count != 0 {
            return Ok(None);
        }
        let Some(values) = self.values_px else {
            return Ok(None);
        };

        Ok(Some(CompleteReprojectionMetricsPx {
            rmse_px: narrow_reprojection_metric(values.rmse_px, "computing reprojection RMSE")?,
            max_px: narrow_reprojection_metric(
                values.max_px,
                "computing maximum reprojection error",
            )?,
        }))
    }
}

pub(crate) fn reprojection_metrics<'a>(
    pose: &Pose,
    observations: impl IntoIterator<Item = &'a Observation>,
    intrinsics: PinholeIntrinsics,
) -> ReprojectionMetrics {
    // Scaled sum-of-squares avoids overflow and underflow across both residual components.
    let mut scale = 0.0_f64;
    let mut scaled_sum_sq = 1.0_f64;
    let mut maximum = 0.0_f64;
    let mut projected_count = 0usize;
    let mut not_in_front_count = 0usize;

    for observation in observations {
        let ReprojectionOutcome::Projected(residual) =
            reprojection_residual_px(*pose, observation, intrinsics)
        else {
            not_in_front_count += 1;
            continue;
        };

        for component in [residual.dx_px, residual.dy_px] {
            let component = component.abs();
            if component != 0.0 {
                if scale < component {
                    let ratio = scale / component;
                    scaled_sum_sq = 1.0 + scaled_sum_sq * ratio * ratio;
                    scale = component;
                } else {
                    let ratio = component / scale;
                    scaled_sum_sq += ratio * ratio;
                }
            }
        }
        maximum = maximum.max(residual.magnitude_px());
        projected_count += 1;
    }

    let values_px = if projected_count == 0 {
        None
    } else {
        let rmse = if scale == 0.0 {
            0.0
        } else {
            scale * (scaled_sum_sq / projected_count as f64).sqrt()
        };
        Some(ReprojectionMetricValuesPx {
            rmse_px: rmse,
            max_px: maximum,
        })
    };

    ReprojectionMetrics {
        values_px,
        projected_count,
        not_in_front_count,
    }
}

fn narrow_reprojection_metric(value: f64, operation: &'static str) -> Result<f32, PnpError> {
    let narrowed = value as f32;
    if !value.is_finite()
        || value < 0.0
        || value > f64::from(f32::MAX)
        || !narrowed.is_finite()
        || (value > 0.0 && narrowed == 0.0)
    {
        return Err(PnpError::Numerical { operation, value });
    }
    Ok(narrowed)
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

    let t = sub(c1, mat_mul_vec(r, w1));
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
    Pose::try_from_rt(rotation_f32, translation).ok()
}

fn narrow_finite_f32(value: f64) -> Option<f32> {
    let narrowed = value as f32;
    narrowed.is_finite().then_some(narrowed)
}

fn vec3_from_world_point(point: WorldPoint3) -> [f64; 3] {
    [f64::from(point.x), f64::from(point.y), f64::from(point.z)]
}

fn vec3_from_f32(vector: [f32; 3]) -> [f64; 3] {
    vector.map(f64::from)
}

fn mat3_from_f32(matrix: [[f32; 3]; 3]) -> [[f64; 3]; 3] {
    matrix.map(|row| row.map(f64::from))
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0].mul_add(b[0], a[1].mul_add(b[1], a[2] * b[2]))
}

fn norm(a: [f64; 3]) -> Option<f64> {
    if a.into_iter().any(|component| !component.is_finite()) {
        return None;
    }
    let norm = a[0].hypot(a[1]).hypot(a[2]);
    (norm.is_finite() && norm > 0.0).then_some(norm)
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
    let norm = norm(v)?;
    let normalized = [v[0] / norm, v[1] / norm, v[2] / norm];
    normalized
        .into_iter()
        .all(f64::is_finite)
        .then_some(normalized)
}

fn mat_mul_vec(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    [
        dot(matrix[0], vector),
        dot(matrix[1], vector),
        dot(matrix[2], vector),
    ]
}

fn transform_point(rotation: [[f64; 3]; 3], translation: [f64; 3], point: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| {
        math::compensated_sum([
            rotation[axis][0] * point[0],
            rotation[axis][1] * point[1],
            rotation[axis][2] * point[2],
            translation[axis],
        ])
    })
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
        axis_angle_pose, make_detections, make_pinhole_intrinsics, make_raw_matches,
        observations_from_projection,
    };
    use crate::{CameraPoint3, FrameId, Point3, SensorId};

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

    fn rot_frob_norm(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> f64 {
        let mut sum = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                let d = f64::from(a[i][j]) - f64::from(b[i][j]);
                sum += d * d;
            }
        }
        sum.sqrt()
    }

    fn l2(a: [f32; 3], b: [f32; 3]) -> f64 {
        let dx = f64::from(a[0]) - f64::from(b[0]);
        let dy = f64::from(a[1]) - f64::from(b[1]);
        let dz = f64::from(a[2]) - f64::from(b[2]);
        dx.hypot(dy).hypot(dz)
    }

    #[test]
    fn pinhole_conversion_parses_only_represented_projection_coefficients() {
        let raw = CameraIntrinsics {
            fx: 400.0,
            fy: 401.0,
            cx: 320.0,
            cy: 240.0,
            width: 0,
            height: 0,
        };

        let parsed = PinholeIntrinsics::try_from(&raw).expect("projection coefficients");
        assert_eq!(parsed.fx().to_bits(), raw.fx.to_bits());
        assert_eq!(parsed.fy().to_bits(), raw.fy.to_bits());
        assert_eq!(parsed.cx().to_bits(), raw.cx.to_bits());
        assert_eq!(parsed.cy().to_bits(), raw.cy.to_bits());
    }

    #[test]
    fn normalize_bearing_has_unit_norm() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 400.0, 400.0, 320.0, 240.0).expect("intrinsics");
        let pixel = Keypoint { x: 369.0, y: 211.0 };
        let b = normalize_bearing(pixel, intrinsics);
        let n = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
        assert!(
            (n - 1.0).abs() <= 4.0 * f64::EPSILON,
            "bearing norm must be 1, got {n}"
        );
    }

    #[test]
    fn normalize_bearing_stays_finite_for_extreme_finite_inputs() {
        let smallest_positive_f32 = f32::from_bits(1);
        let intrinsics = make_pinhole_intrinsics(
            1,
            1,
            smallest_positive_f32,
            smallest_positive_f32,
            -f32::MAX,
            f32::MAX,
        )
        .expect("intrinsics");
        let pixel = Keypoint {
            x: f32::MAX,
            y: -f32::MAX,
        };

        let bearing = normalize_bearing(pixel, intrinsics);
        assert!(bearing.into_iter().all(f64::is_finite));
        let norm = bearing[0].hypot(bearing[1]).hypot(bearing[2]);
        assert!(
            (norm - 1.0).abs() <= 4.0 * f64::EPSILON,
            "bearing norm must remain one, got {norm}"
        );
        assert!(bearing[0].is_sign_positive());
        assert!(bearing[1].is_sign_negative());
        assert!(bearing[2] > 0.0);
    }

    #[test]
    fn normalize_bearing_preserves_center_and_image_axis_signs() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 100.0, 100.0, 320.0, 240.0).expect("intrinsics");

        let center = normalize_bearing(Keypoint { x: 320.0, y: 240.0 }, intrinsics);
        assert_eq!(center, [0.0, 0.0, 1.0]);

        let left_and_down = normalize_bearing(Keypoint { x: 220.0, y: 340.0 }, intrinsics);
        assert!(left_and_down[0].is_sign_negative());
        assert!(left_and_down[1].is_sign_positive());
        assert!(left_and_down[2].is_sign_positive());
    }

    #[test]
    fn normalize_bearing_preserves_distinct_extreme_finite_rays() {
        let tiny = f32::from_bits(1);
        let intrinsics =
            make_pinhole_intrinsics(5, 5, f32::MAX, tiny, 0.0, 0.0).expect("intrinsics");
        let first = normalize_bearing(Keypoint { x: tiny, y: 4.0 }, intrinsics);
        let second = normalize_bearing(
            Keypoint {
                x: 2.0 * tiny,
                y: 4.0,
            },
            intrinsics,
        );

        assert!(first.into_iter().all(f64::is_finite));
        assert!(second.into_iter().all(f64::is_finite));
        assert!(first[0] > 0.0);
        assert!(second[0] > first[0]);
        assert!(first[2] > 0.0 && second[2] > 0.0);
        assert_ne!(first, second);
    }

    #[test]
    fn normalize_bearing_is_finite_unit_and_forward_over_f32_extremes() {
        let focal_values = [f32::from_bits(1), f32::MIN_POSITIVE, 1.0, f32::MAX];
        let coordinate_values = [-f32::MAX, -0.0, 0.0, f32::from_bits(1), f32::MAX];

        for fx in focal_values {
            for fy in focal_values {
                for cx in coordinate_values {
                    for cy in coordinate_values {
                        let intrinsics = make_pinhole_intrinsics(1, 1, fx, fy, cx, cy)
                            .expect("finite positive intrinsics");
                        for x in coordinate_values {
                            for y in coordinate_values {
                                let bearing = normalize_bearing(Keypoint { x, y }, intrinsics);
                                assert!(bearing.into_iter().all(f64::is_finite));
                                assert!(bearing[2] > 0.0);
                                let norm = bearing[0].hypot(bearing[1]).hypot(bearing[2]);
                                assert!(
                                    (norm - 1.0).abs() <= 4.0 * f64::EPSILON,
                                    "fx={fx} fy={fy} cx={cx} cy={cy} x={x} y={y} norm={norm}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn observation_reports_the_exact_invalid_coordinate() {
        let valid_world = WorldPoint3::new(1.0, 2.0, 3.0);
        for (axis, value) in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY]
            .into_iter()
            .enumerate()
        {
            let mut coordinates = valid_world.to_array();
            coordinates[axis] = value;
            assert!(matches!(
                Observation::try_new(
                    WorldPoint3::from_array(coordinates),
                    Keypoint { x: 4.0, y: 5.0 },
                ),
                Err(ObservationError::InvalidWorldPoint(
                    crate::Point3Error::NonFinite {
                        axis: error_axis,
                        value: error_value,
                    }
                )) if error_axis == axis && error_value.to_bits() == value.to_bits()
            ));
        }

        for (axis, value) in [f32::NAN, f32::INFINITY].into_iter().enumerate() {
            let mut pixel = Keypoint { x: 4.0, y: 5.0 };
            if axis == 0 {
                pixel.x = value;
            } else {
                pixel.y = value;
            }
            assert!(matches!(
                Observation::try_new(valid_world, pixel),
                Err(ObservationError::NonFinitePixel {
                    axis: error_axis,
                    value: error_value,
                }) if error_axis == axis && error_value.to_bits() == value.to_bits()
            ));
        }
    }

    #[test]
    fn calibrated_problem_binds_bearing_and_scoring_to_one_camera_model() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 200.0, 200.0, 320.0, 240.0).expect("intrinsics");
        let observation = Observation::try_new(
            WorldPoint3::new(1.0, 0.0, 2.0),
            Keypoint { x: 420.0, y: 240.0 },
        )
        .expect("finite observation");
        let observations = [observation];
        let problem = CalibratedPnpProblem::parse(&observations, intrinsics);
        let bearing = problem.bearings[0].into_array();

        assert!((bearing[0] - 0.5_f64 / 1.25_f64.sqrt()).abs() < 4.0 * f64::EPSILON);
        assert_eq!(bearing[1].to_bits(), 0.0_f64.to_bits());
        assert!((bearing[2] - 1.0_f64 / 1.25_f64.sqrt()).abs() < 4.0 * f64::EPSILON);
        let ReprojectionOutcome::Projected(residual) =
            problem.reprojection_residual_px(Pose::identity(), 0)
        else {
            panic!("positive-depth observation must project");
        };
        assert_eq!(residual.dx_px.to_bits(), 0.0_f64.to_bits());
        assert_eq!(residual.dy_px.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn normalize_rejects_zero_and_nonfinite_vectors() {
        for vector in [
            [0.0, 0.0, 0.0],
            [f64::NAN, 1.0, 0.0],
            [1.0, f64::INFINITY, 0.0],
            [1.0, 0.0, f64::NEG_INFINITY],
        ] {
            assert!(normalize(vector).is_none(), "vector={vector:?}");
        }
    }

    #[test]
    fn build_observations_rejects_nonfinite_transformed_landmarks() {
        let keypoints = vec![
            Keypoint { x: 10.0, y: 10.0 },
            Keypoint { x: 20.0, y: 20.0 },
            Keypoint { x: 30.0, y: 30.0 },
            Keypoint { x: 40.0, y: 40.0 },
        ];
        let current = make_detections(
            SensorId::StereoLeft,
            FrameId::new(1),
            640,
            480,
            keypoints.clone(),
        )
        .expect("current detections");
        let keyframe_detections =
            make_detections(SensorId::StereoLeft, FrameId::new(2), 640, 480, keypoints)
                .expect("keyframe detections");
        let keyframe = Keyframe::from_arc(
            keyframe_detections.clone(),
            vec![
                CameraPoint3 {
                    x: f32::MAX,
                    y: 0.0,
                    z: 1.0,
                };
                MIN_PNP_POINTS
            ],
            (0..MIN_PNP_POINTS).collect(),
        )
        .expect("keyframe");
        let raw_matches = make_raw_matches(
            current,
            keyframe_detections,
            (0..MIN_PNP_POINTS).map(|index| (index, index)).collect(),
        )
        .expect("raw matches");
        let matches = raw_matches
            .with_landmarks(&keyframe)
            .expect("verified matches");
        let keyframe_to_world = CameraToWorld::from_legacy_pose(Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [f32::MAX, 0.0, 0.0],
        ));

        assert!(matches!(
            build_observations(&keyframe, &matches, keyframe_to_world),
            Err(PnpError::Transform(
                TransformError::OutputNotRepresentable { axis: 0, value }
            )) if value > f64::from(f32::MAX)
        ));
    }

    #[test]
    fn sample_three_returns_distinct_indices() {
        let mut rng = XorShift64::new(NonZeroU64::new(0xDEADBEEF).expect("nonzero test seed"));
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
        let recovered = pose
            .try_inverse()
            .and_then(Pose::try_inverse)
            .expect("finite pose remains invertible");
        assert!(rot_frob_norm(pose.rotation(), recovered.rotation()) < 1e-5);
        assert!(l2(pose.translation(), recovered.translation()) < 1e-5);
    }

    #[test]
    fn pose_compose_matches_pose64_ordering() {
        let first = axis_angle_pose([0.3, -0.2, 0.7], [0.1, -0.05, 0.08]);
        let second = axis_angle_pose([-0.1, 0.4, 0.2], [-0.03, 0.09, 0.02]);
        let composed = first
            .try_compose(second)
            .expect("finite poses remain composable");
        let composed64 = crate::Pose64::try_from_pose32(first)
            .expect("first pose should be valid")
            .try_compose(
                crate::Pose64::try_from_pose32(second).expect("second pose should be valid"),
            )
            .expect("composition should remain finite")
            .try_to_pose32()
            .expect("test pose should fit in f32");

        assert!(rot_frob_norm(composed.rotation(), composed64.rotation()) < 1e-6);
        assert!(l2(composed.translation(), composed64.translation()) < 1e-6);

        let point = [1.2, -0.4, 3.5];
        let after_second =
            crate::math::transform_point(second.rotation(), second.translation(), point);
        let expected =
            crate::math::transform_point(first.rotation(), first.translation(), after_second);
        let actual =
            crate::math::transform_point(composed.rotation(), composed.translation(), point);
        assert!(l2(actual, expected) < 1e-5);
    }

    #[test]
    fn pose_constructor_rejects_nonfinite_and_non_so3_inputs() {
        let identity = Pose::identity().rotation();
        assert!(matches!(
            Pose::try_from_rt(identity, [f32::NAN, 0.0, 0.0]),
            Err(PoseError::NonFiniteTranslation { axis: 0, value }) if value.is_nan()
        ));

        let mut nonfinite_rotation = identity;
        nonfinite_rotation[1][2] = f32::INFINITY;
        assert!(matches!(
            Pose::try_from_rt(nonfinite_rotation, [0.0; 3]),
            Err(PoseError::NonFiniteRotation {
                row: 1,
                column: 2,
                value: f32::INFINITY,
            })
        ));

        let scaled = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(matches!(
            Pose::try_from_rt(scaled, [0.0; 3]),
            Err(PoseError::NonOrthonormalRotation { .. })
        ));

        let reflection = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(matches!(
            Pose::try_from_rt(reflection, [0.0; 3]),
            Err(PoseError::ImproperRotation { determinant }) if determinant == -1.0
        ));
    }

    #[test]
    fn pose_operations_report_finite_f32_overflow() {
        let maximum_translation =
            Pose::try_from_rt(Pose::identity().rotation(), [f32::MAX, 0.0, 0.0])
                .expect("maximum finite translation is representable");
        assert!(matches!(
            maximum_translation.try_compose(maximum_translation),
            Err(PoseError::ComposeTranslationNotRepresentable { axis: 0, value })
                if value > f64::from(f32::MAX)
        ));

        let s = std::f32::consts::FRAC_1_SQRT_2;
        let rotated = Pose::try_from_rt(
            [[s, -s, 0.0], [s, s, 0.0], [0.0, 0.0, 1.0]],
            [f32::MAX, f32::MAX, 0.0],
        )
        .expect("valid rotated maximum translation");
        assert!(matches!(
            rotated.try_inverse(),
            Err(PoseError::InverseTranslationNotRepresentable { axis: 0, value })
                if value < -f64::from(f32::MAX)
        ));
    }

    #[test]
    fn typed_transform_reports_point_overflow_without_losing_frames() {
        let pose = Pose::try_from_rt(Pose::identity().rotation(), [f32::MAX, 0.0, 0.0])
            .expect("finite pose");
        let world_to_camera = WorldToCamera::from_legacy_pose(pose);
        assert!(matches!(
            world_to_camera.try_transform_point(WorldPoint3::new(f32::MAX, 0.0, 1.0)),
            Err(TransformError::OutputNotRepresentable { axis: 0, value })
                if value > f64::from(f32::MAX)
        ));

        let finite = WorldPoint3::new(1.0, -2.0, 3.0);
        let camera: CameraPoint3 = WorldToCamera::identity()
            .try_transform_point(finite)
            .expect("identity transform");
        assert_eq!(camera.to_array(), finite.to_array());
    }

    #[test]
    fn checked_pose_and_typed_transform_round_trips_are_deterministic() {
        for index in -32_i32..=32 {
            let scale = index as f32 / 32.0;
            let pose = Pose::try_from_rt(
                math::so3_exp([0.17 * scale, -0.11 * scale, 0.07 * scale]),
                [3.0 * scale, -2.0 * scale, 0.5 * scale],
            )
            .expect("bounded deterministic pose");
            let inverse = pose.try_inverse().expect("bounded pose inverse");
            let identity = pose.try_compose(inverse).expect("bounded pose closure");
            assert!(rot_frob_norm(identity.rotation(), Pose::identity().rotation()) < 2e-5);
            assert!(l2(identity.translation(), [0.0; 3]) < 2e-5);

            let world_to_camera = WorldToCamera::from_legacy_pose(pose);
            let point = WorldPoint3::new(0.25 + scale, -0.75 * scale, 2.0 - 0.5 * scale);
            let camera = world_to_camera
                .try_transform_point(point)
                .expect("bounded forward transform");
            let recovered = world_to_camera
                .try_inverse()
                .expect("bounded typed inverse")
                .try_transform_point(camera)
                .expect("bounded inverse transform");
            assert!(l2(recovered.to_array(), point.to_array()) < 2e-5);

            let recomposed: WorldToCamera = world_to_camera
                .try_compose(Transform::<WorldFrame, WorldFrame>::identity())
                .expect("typed composition closure");
            assert!(rot_frob_norm(recomposed.rotation(), pose.rotation()) < 2e-5);
        }
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

        let config = RansacConfig::try_new(700, 1.0, 20, 0xBAD5EED).expect("config");
        let result = solve_pnp_ransac(&observations, intrinsics, config).expect("pnp");
        assert!(result.inliers.len() >= 20, "insufficient inliers");

        let rot_err = rot_frob_norm(result.pose.rotation(), pose_gt.rotation());
        let trans_err = l2(result.pose.translation(), pose_gt.translation());
        assert!(rot_err < 0.03, "rotation error too high: {rot_err}");
        assert!(trans_err < 0.08, "translation error too high: {trans_err}");
    }

    #[test]
    fn p3p_solver_is_invariant_to_scene_scale() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let base_world = synthetic_world_points();
        let base_translation = [0.2, -0.1, 0.35];
        let rotation_pose = axis_angle_pose([0.0; 3], [0.08, -0.06, 0.04]);
        let config = RansacConfig::try_new(700, 1.0, 20, 0x51CA1E).expect("config");

        for scale in [1e-23_f32, 1e-3, 1.0, 1e3, 1e20] {
            let world: Vec<_> = base_world
                .iter()
                .map(|point| Point3 {
                    x: point.x * scale,
                    y: point.y * scale,
                    z: point.z * scale,
                })
                .collect();
            let pose = Pose::from_rt(
                rotation_pose.rotation(),
                [
                    base_translation[0] * scale,
                    base_translation[1] * scale,
                    base_translation[2] * scale,
                ],
            );
            let observations: Vec<_> = world
                .iter()
                .map(|&point| {
                    Observation::try_new(point, project_pixel_from_pose(pose, point, intrinsics))
                        .expect("finite scale-invariance observation")
                })
                .collect();
            let result = solve_pnp_ransac(&observations, intrinsics, config)
                .unwrap_or_else(|err| panic!("P3P failed at scene scale {scale}: {err}"));

            assert!(result.inliers.len() >= 20, "scale={scale}");
            assert!(
                rot_frob_norm(result.pose.rotation(), pose.rotation()) < 0.03,
                "rotation error at scale={scale}"
            );
            assert!(
                l2(result.pose.translation(), pose.translation()) < 0.08 * f64::from(scale),
                "translation error at scale={scale}"
            );
        }
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
            with_outliers.push(Observation::try_new(obs.world(), pixel).expect("observation"));
        }

        let config = RansacConfig::try_new(1000, 2.0, 14, 0x1337).expect("config");
        let result = solve_pnp_ransac(&with_outliers, intrinsics, config).expect("pnp");
        assert!(
            result.inliers.len() >= 14,
            "expected robust inliers, got {}",
            result.inliers.len()
        );

        let rot_err = rot_frob_norm(result.pose.rotation(), pose_gt.rotation());
        let trans_err = l2(result.pose.translation(), pose_gt.translation());
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
            )
            .expect("obs"),
            Observation::try_new(
                Point3 {
                    x: 0.2,
                    y: 0.1,
                    z: 3.5,
                },
                Keypoint { x: 342.0, y: 252.0 },
            )
            .expect("obs"),
            Observation::try_new(
                Point3 {
                    x: -0.2,
                    y: 0.2,
                    z: 2.9,
                },
                Keypoint { x: 290.0, y: 266.0 },
            )
            .expect("obs"),
        ];

        let err =
            solve_pnp_ransac(&obs, intrinsics, RansacConfig::default()).expect_err("should reject");
        match err {
            PnpError::NotEnoughPoints { required, actual } => {
                assert_eq!(required, 4);
                assert_eq!(actual, 3);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn ransac_config_rejects_invalid_values() {
        assert!(matches!(
            RansacConfig::try_new(0, 1.0, 4, 1),
            Err(RansacConfigError::ZeroIterations)
        ));
        assert!(matches!(
            RansacConfig::try_new(10, 1.0, 4, 0),
            Err(RansacConfigError::ZeroSeed)
        ));
        for threshold in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                RansacConfig::try_new(10, threshold, 4, 1),
                Err(RansacConfigError::InvalidReprojectionThreshold { .. })
            ));
        }
        assert!(matches!(
            RansacConfig::try_new(10, 1.0, 3, 1),
            Err(RansacConfigError::TooFewInliers {
                value: 3,
                minimum: 4
            })
        ));

        let extreme = RansacConfig::try_new(10, f32::MAX, 4, 1)
            .expect("every positive finite f32 threshold is accepted");
        let threshold_px = f64::from(extreme.reprojection_threshold_px());
        assert!(threshold_px.powi(2).is_finite());
        assert!(threshold_px.powi(2) > f64::from(f32::MAX));
    }

    #[test]
    fn ransac_reports_configured_inlier_requirement_before_solving() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let pose = axis_angle_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let world = synthetic_world_points();
        let observations =
            observations_from_projection(pose, &world[..4], intrinsics).expect("observations");
        let config = RansacConfig::try_new(20, 1.0, 5, 1).expect("config");

        let error = solve_pnp_ransac(&observations, intrinsics, config)
            .expect_err("five configured inliers require at least five observations");
        assert!(matches!(
            error,
            PnpError::NotEnoughPoints {
                required: 5,
                actual: 4
            }
        ));
    }

    fn expect_projected_residual(
        pose: Pose,
        observation: &Observation,
        intrinsics: PinholeIntrinsics,
    ) -> ReprojectionResidualPx {
        match reprojection_residual_px(pose, observation, intrinsics) {
            ReprojectionOutcome::Projected(residual) => residual,
            ReprojectionOutcome::NotInFront => panic!("expected point in front of the camera"),
        }
    }

    fn parsed_reprojection_threshold(value: f32) -> ReprojectionThresholdPx {
        ReprojectionThresholdPx::from_config(
            RansacConfig::try_new(1, value, MIN_PNP_POINTS, 1).expect("valid threshold"),
        )
    }

    #[test]
    fn reprojection_residual_is_zero_for_exact_projection() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 400.0, 400.0, 320.0, 240.0).expect("intrinsics");
        let pose = axis_angle_pose([0.0, 0.0, 0.0], [0.05, -0.03, 0.01]);
        let point = Point3 {
            x: 0.3,
            y: -0.1,
            z: 4.2,
        };
        let pixel = project_pixel_from_pose(pose, point, intrinsics);
        let observation = Observation::try_new(point, pixel).expect("observation");
        let residual = expect_projected_residual(pose, &observation, intrinsics);
        let error_sq = residual
            .dx_px
            .mul_add(residual.dx_px, residual.dy_px * residual.dy_px);
        assert!(
            error_sq < 1e-8,
            "expected exact reprojection, got {residual:?}"
        );
    }

    #[test]
    fn reprojection_metrics_are_zero_for_exact_projection() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let pose = Pose::identity();
        let world = synthetic_world_points();
        let observations =
            observations_from_projection(pose, &world, intrinsics).expect("synthetic observations");
        let metrics = reprojection_metrics(&pose, &observations, intrinsics);

        assert_eq!(metrics.projected_count(), observations.len());
        assert_eq!(metrics.not_in_front_count(), 0);
        assert!(metrics.rmse_px().is_some_and(|value| value < 1e-4));
        assert!(metrics.max_px().is_some_and(|value| value < 1e-4));
    }

    #[test]
    fn reprojection_metrics_count_nonpositive_depth_separately() {
        let intrinsics = make_pinhole_intrinsics(640, 480, 1.0, 1.0, 0.0, 0.0).expect("intrinsics");
        let observations = [
            Observation::try_new(Point3::new(0.0, 0.0, 1.0), Keypoint { x: -3.0, y: 0.0 })
                .expect("three-pixel observation"),
            Observation::try_new(Point3::new(0.0, 0.0, -1.0), Keypoint { x: 0.0, y: 0.0 })
                .expect("nonpositive-depth observation"),
            Observation::try_new(Point3::new(0.0, 0.0, 1.0), Keypoint { x: -4.0, y: 0.0 })
                .expect("four-pixel observation"),
        ];

        assert_eq!(
            reprojection_residual_px(Pose::identity(), &observations[1], intrinsics),
            ReprojectionOutcome::NotInFront
        );
        let metrics = reprojection_metrics(&Pose::identity(), &observations, intrinsics);
        let expected_rmse = ((3.0_f64 * 3.0 + 4.0 * 4.0) / 2.0).sqrt();

        assert_eq!(metrics.projected_count(), 2);
        assert_eq!(metrics.not_in_front_count(), 1);
        assert!(
            metrics
                .rmse_px()
                .is_some_and(|value| (value - expected_rmse).abs() < 1e-6)
        );
        assert_eq!(metrics.max_px(), Some(4.0));
    }

    #[test]
    fn reprojection_metrics_define_empty_and_all_not_in_front_batches() {
        let intrinsics = make_pinhole_intrinsics(1, 1, 1.0, 1.0, 0.0, 0.0).expect("intrinsics");
        let empty: [Observation; 0] = [];
        let empty_metrics = reprojection_metrics(&Pose::identity(), &empty, intrinsics);
        assert_eq!(empty_metrics.rmse_px(), None);
        assert_eq!(empty_metrics.max_px(), None);
        assert_eq!(empty_metrics.projected_count(), 0);
        assert_eq!(empty_metrics.not_in_front_count(), 0);

        let not_in_front =
            [
                Observation::try_new(Point3::new(0.0, 0.0, 0.0), Keypoint { x: 0.0, y: 0.0 })
                    .expect("zero-depth observation"),
            ];
        let hidden_metrics = reprojection_metrics(&Pose::identity(), &not_in_front, intrinsics);
        assert_eq!(hidden_metrics.rmse_px(), None);
        assert_eq!(hidden_metrics.max_px(), None);
        assert_eq!(hidden_metrics.projected_count(), 0);
        assert_eq!(hidden_metrics.not_in_front_count(), 1);
    }

    #[test]
    fn reprojection_metrics_are_order_and_power_of_two_scale_stable() {
        let intrinsics = make_pinhole_intrinsics(1, 1, 1.0, 1.0, 0.0, 0.0).expect("intrinsics");
        let make_observation = |error_px: f32| {
            Observation::try_new(
                Point3::new(0.0, 0.0, 1.0),
                Keypoint {
                    x: -error_px,
                    y: 0.0,
                },
            )
            .expect("finite error observation")
        };
        let observations = [
            make_observation(3.0),
            make_observation(4.0),
            make_observation(12.0),
        ];
        let reversed = [observations[2], observations[1], observations[0]];
        let metrics = reprojection_metrics(&Pose::identity(), &observations, intrinsics);
        let reversed_metrics = reprojection_metrics(&Pose::identity(), &reversed, intrinsics);
        let rmse = metrics.rmse_px().expect("rmse");

        assert_eq!(metrics, reversed_metrics);
        assert!((3.0..=12.0).contains(&rmse));
        assert_eq!(metrics.max_px(), Some(12.0));

        let scale = 2.0_f32.powi(20);
        let scaled = [
            make_observation(3.0 * scale),
            make_observation(4.0 * scale),
            make_observation(12.0 * scale),
        ];
        let scaled_metrics = reprojection_metrics(&Pose::identity(), &scaled, intrinsics);
        assert_eq!(scaled_metrics.rmse_px(), Some(rmse * f64::from(scale)));
        assert_eq!(scaled_metrics.max_px(), Some(12.0 * f64::from(scale)));
    }

    #[test]
    fn reprojection_metrics_keep_large_finite_residuals_finite() {
        let intrinsics = make_pinhole_intrinsics(1, 1, 1.0, 1.0, 0.0, 0.0).expect("intrinsics");
        let observations = [
            Observation::try_new(
                Point3::new(0.0, 0.0, 1.0),
                Keypoint {
                    x: -f32::MAX,
                    y: 0.0,
                },
            )
            .expect("first extreme observation"),
            Observation::try_new(
                Point3::new(0.0, 0.0, 1.0),
                Keypoint {
                    x: -f32::MAX,
                    y: 0.0,
                },
            )
            .expect("second extreme observation"),
        ];
        let metrics = reprojection_metrics(&Pose::identity(), &observations, intrinsics);

        assert_eq!(metrics.rmse_px(), Some(f64::from(f32::MAX)));
        assert_eq!(metrics.max_px(), Some(f64::from(f32::MAX)));
        let complete = metrics
            .complete_px()
            .expect("representable metrics")
            .expect("nonempty complete metrics");
        assert_eq!(complete.rmse_px(), f32::MAX);
        assert_eq!(complete.max_px(), f32::MAX);
    }

    #[test]
    fn reprojection_metric_narrowing_rejects_range_overflow_and_underflow() {
        let residual = 2.0 * f64::from(f32::MAX);
        assert!(matches!(
            narrow_reprojection_metric(residual, "computing reprojection RMSE"),
            Err(PnpError::Numerical {
                operation: "computing reprojection RMSE",
                value,
            }) if value == residual
        ));

        let rounds_down_to_max = f64::from(f32::MAX) + 2.0_f64.powi(102);
        assert!(matches!(
            narrow_reprojection_metric(rounds_down_to_max, "computing reprojection RMSE"),
            Err(PnpError::Numerical {
                operation: "computing reprojection RMSE",
                value,
            }) if value == rounds_down_to_max
        ));

        let rounds_to_zero = 0.25 * f64::from(f32::from_bits(1));
        assert!(matches!(
            narrow_reprojection_metric(rounds_to_zero, "computing maximum reprojection error"),
            Err(PnpError::Numerical {
                operation: "computing maximum reprojection error",
                value,
            }) if value == rounds_to_zero
        ));
    }

    #[test]
    fn reprojection_metrics_retain_large_finite_residuals_until_metric_boundary() {
        let intrinsics =
            make_pinhole_intrinsics(1, 1, f32::MAX, 1.0, 0.0, 0.0).expect("intrinsics");
        let observation = Observation::try_new(
            Point3 {
                x: f32::MAX,
                y: 0.0,
                z: f32::from_bits(1),
            },
            Keypoint { x: 0.0, y: 0.0 },
        )
        .expect("finite observation");

        let residual = expect_projected_residual(Pose::identity(), &observation, intrinsics);
        let magnitude = residual.magnitude_px();
        assert!(magnitude.is_finite());
        assert!(magnitude > f64::from(f32::MAX));
        let metrics = reprojection_metrics(&Pose::identity(), [&observation], intrinsics);
        assert_eq!(metrics.rmse_px(), Some(magnitude));
        assert_eq!(metrics.max_px(), Some(magnitude));
        assert!(matches!(
            metrics.complete_px(),
            Err(PnpError::Numerical {
                operation: "computing reprojection RMSE",
                value,
            }) if value == magnitude
        ));
    }

    #[test]
    fn reprojection_metrics_match_known_perturbation() {
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
                Observation::try_new(point, pixel).expect("observation")
            })
            .collect();
        let metrics = reprojection_metrics(&pose, &observations, intrinsics);
        let rmse = metrics.rmse_px().expect("rmse");
        assert!((1.5..=2.5).contains(&rmse), "rmse={rmse}");
        let max = metrics.max_px().expect("max");
        assert!((1.5..=2.5).contains(&max), "max={max}");
    }

    #[test]
    fn reprojection_metric_counts_match_input_cardinality() {
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
            )
            .expect("observation"),
            Observation::try_new(
                Point3 {
                    x: 0.1,
                    y: 0.0,
                    z: 3.0,
                },
                Keypoint { x: 340.0, y: 240.0 },
            )
            .expect("observation"),
        ];
        let metrics = reprojection_metrics(&pose, &observations, intrinsics);
        assert_eq!(
            metrics.projected_count() + metrics.not_in_front_count(),
            observations.len()
        );
    }

    #[test]
    fn compensated_transform_recovers_pixel_residual_erased_by_f64_cancellation() {
        let sine_cosine = std::f32::consts::FRAC_1_SQRT_2;
        let rotation = [
            [0.5, 0.5, sine_cosine],
            [0.5, 0.5, -sine_cosine],
            [-sine_cosine, sine_cosine, 0.0],
        ];
        let pose = Pose::try_from_rt(rotation, [0.0; 3]).expect("valid rotation");
        let intrinsics =
            make_pinhole_intrinsics(1, 1, f32::MAX, f32::MAX, 0.0, 0.0).expect("intrinsics");
        let observation = Observation::try_new(
            Point3::new(-f32::MAX, f32::MAX, 1.0),
            Keypoint { x: 0.0, y: 0.0 },
        )
        .expect("observation");

        let residual = expect_projected_residual(pose, &observation, intrinsics);

        assert!((residual.dx_px - 0.5).abs() < 1e-15);
        assert!((residual.dy_px + 0.5).abs() < 1e-15);
        assert!((residual.magnitude_px() - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-15);
        assert!(!residual.is_within_threshold(parsed_reprojection_threshold(0.25)));
    }

    #[test]
    fn compensated_transform_recovers_positive_depth_erased_by_f64_cancellation() {
        let inverse_sqrt_2 = 1.0 / 2.0_f32.sqrt();
        let inverse_sqrt_6 = 1.0 / 6.0_f32.sqrt();
        let inverse_sqrt_3 = 1.0 / 3.0_f32.sqrt();
        let rotation = [
            [inverse_sqrt_2, -inverse_sqrt_2, 0.0],
            [inverse_sqrt_6, inverse_sqrt_6, -2.0 * inverse_sqrt_6],
            [inverse_sqrt_3; 3],
        ];
        let pose = Pose::try_from_rt(rotation, [0.0; 3]).expect("valid rotation");
        let point = Point3::new(f32::MAX, -f32::MAX, 1.0);
        let camera = transform_point(
            mat3_from_f32(rotation),
            [0.0; 3],
            vec3_from_world_point(point),
        );
        let intrinsics = make_pinhole_intrinsics(1, 1, 1.0, 1.0, 0.0, 0.0).expect("intrinsics");
        let observation =
            Observation::try_new(point, Keypoint { x: 0.0, y: 0.0 }).expect("observation");

        assert!((camera[2] - f64::from(inverse_sqrt_3)).abs() < f64::EPSILON);
        assert!(matches!(
            reprojection_residual_px(pose, &observation, intrinsics),
            ReprojectionOutcome::Projected(_)
        ));
    }

    #[test]
    fn direct_projection_division_avoids_reciprocal_rounding_residual() {
        let focal_px = f32::from_bits(0x640e_6030);
        let camera_x_m = f32::from_bits(0xbb3f_0274);
        let camera_z_m = f32::from_bits(0x3cfe_adf0);
        let observed_x_px = f32::from_bits(0xe255_9048);
        let intrinsics =
            make_pinhole_intrinsics(1, 1, focal_px, 1.0, 0.0, 0.0).expect("intrinsics");
        let observation = Observation::try_new(
            Point3::new(camera_x_m, 0.0, camera_z_m),
            Keypoint {
                x: observed_x_px,
                y: 0.0,
            },
        )
        .expect("observation");

        let reciprocal_first = (f64::from(focal_px) * f64::from(camera_x_m))
            * (1.0 / f64::from(camera_z_m))
            - f64::from(observed_x_px);
        let residual = expect_projected_residual(Pose::identity(), &observation, intrinsics);

        assert_eq!(reciprocal_first, -131_072.0);
        assert_eq!(residual.dx_px.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn homogeneous_projection_cancels_before_division_rounding() {
        let focal_px = f32::from_bits(0x4103_95df);
        let camera_x_m = f32::from_bits(0xe22a_24de);
        let camera_z_m = f32::from_bits(0x47d1_b958);
        let principal_x_px = f32::from_bits(0x3ff7_7e81);
        let observed_x_px = f32::from_bits(0xdb55_8111);
        let intrinsics =
            make_pinhole_intrinsics(1, 1, focal_px, 1.0, principal_x_px, 0.0).expect("intrinsics");
        let observation = Observation::try_new(
            Point3::new(camera_x_m, 0.0, camera_z_m),
            Keypoint {
                x: observed_x_px,
                y: 0.0,
            },
        )
        .expect("observation");

        let quotient_first = math::compensated_sum([
            (f64::from(focal_px) * f64::from(camera_x_m)) / f64::from(camera_z_m),
            f64::from(principal_x_px),
            -f64::from(observed_x_px),
        ]);
        let residual = expect_projected_residual(Pose::identity(), &observation, intrinsics);
        let exact_source_residual = 6_251.671_327_761_643_f64;

        assert!((quotient_first - exact_source_residual).abs() > 1.7);
        assert!((residual.dx_px - exact_source_residual).abs() < 1e-9);
    }

    #[test]
    fn compensated_residual_retains_absorbed_principal_point() {
        let camera_x_m = 1.0e20_f32;
        let intrinsics = make_pinhole_intrinsics(1, 1, 1.0, 1.0, 1.0, 0.0).expect("intrinsics");
        let observation = Observation::try_new(
            Point3::new(camera_x_m, 0.0, 1.0),
            Keypoint {
                x: camera_x_m,
                y: 0.0,
            },
        )
        .expect("observation");

        let residual = expect_projected_residual(Pose::identity(), &observation, intrinsics);

        assert_eq!(residual.dx_px, 1.0);
    }

    #[test]
    fn expansion_residual_retains_term_below_canceling_product_roundoff() {
        let tiny = f32::from_bits(1);
        let cosine = std::f32::consts::FRAC_1_SQRT_2;
        let rotation = [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -cosine],
            [0.0, cosine, cosine],
        ];
        let pose = Pose::try_from_rt(rotation, [0.0; 3]).expect("valid rotation");
        let point = Point3::new(tiny, 1.0, 0.1);
        let intrinsics =
            make_pinhole_intrinsics(1, 1, tiny, 1.0, f32::MAX, 0.0).expect("intrinsics");
        let observation = Observation::try_new(
            point,
            Keypoint {
                x: f32::MAX,
                y: 0.0,
            },
        )
        .expect("observation");
        let camera = transform_point(
            mat3_from_f32(rotation),
            [0.0; 3],
            vec3_from_world_point(point),
        );

        let residual = expect_projected_residual(pose, &observation, intrinsics);
        let expected = (f64::from(tiny) * f64::from(tiny)) / camera[2];

        assert!(expected > 0.0);
        assert_eq!(residual.dx_px.to_bits(), expected.to_bits());
    }

    #[test]
    fn finite_reprojection_norm_survives_squared_overflow() {
        let tiny = f32::from_bits(1);
        let rotation = [[1.0, 0.0, 0.0], [0.0, tiny, -1.0], [0.0, 1.0, tiny]];
        let pose = Pose::try_from_rt(rotation, [0.0; 3]).expect("valid rotation");
        let intrinsics =
            make_pinhole_intrinsics(1, 1, f32::MAX, tiny, 0.0, 0.0).expect("intrinsics");
        let observation = Observation::try_new(
            Point3::new(f32::MAX, 0.0, tiny),
            Keypoint { x: 0.0, y: 0.0 },
        )
        .expect("observation");

        let residual = expect_projected_residual(pose, &observation, intrinsics);
        let magnitude = residual.magnitude_px();
        let threshold = parsed_reprojection_threshold(f32::MAX);

        assert!(
            residual
                .dx_px
                .mul_add(residual.dx_px, residual.dy_px * residual.dy_px)
                .is_infinite()
        );
        assert!(magnitude.is_finite());
        assert!(magnitude > threshold.value);
        assert!(!residual.is_within_threshold(threshold));
        let metrics = reprojection_metrics(&pose, [&observation], intrinsics);
        assert_eq!(metrics.rmse_px(), Some(magnitude));
        assert!(matches!(
            metrics.complete_px(),
            Err(PnpError::Numerical {
                operation: "computing reprojection RMSE",
                value,
            }) if value == magnitude
        ));
    }

    #[test]
    fn reprojection_threshold_is_inclusive_without_unbounded_squaring() {
        let residual = ReprojectionResidualPx {
            dx_px: 3.0,
            dy_px: 4.0,
        };
        let threshold = parsed_reprojection_threshold(5.0);

        assert!(residual.is_within_threshold(threshold));
        let below_threshold = f32::from_bits(5.0_f32.to_bits() - 1);
        assert!(!residual.is_within_threshold(parsed_reprojection_threshold(below_threshold)));
    }

    #[test]
    fn reprojection_threshold_rejects_nonzero_orthogonal_boundary_component() {
        let tiny = f32::from_bits(1);
        let intrinsics = make_pinhole_intrinsics(1, 1, 1.0, tiny, 1.0, 0.0).expect("intrinsics");
        let observation = Observation::try_new(
            Point3::new(0.0, tiny, f32::MAX),
            Keypoint { x: 0.0, y: 0.0 },
        )
        .expect("observation");
        let residual = expect_projected_residual(Pose::identity(), &observation, intrinsics);
        let threshold = parsed_reprojection_threshold(1.0);

        assert_eq!(residual.dx_px, threshold.value);
        assert!(residual.dy_px > 0.0);
        assert_eq!(residual.magnitude_px(), threshold.value);
        assert!(!residual.is_within_threshold(threshold));
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
