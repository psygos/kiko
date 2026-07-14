use std::collections::TryReserveError;
use std::num::NonZeroUsize;

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
    NonFiniteCameraPointMeters { axis: CameraFrameAxis, value: f32 },
    NonFinitePixelCoordinatePx { axis: ImagePlaneAxis, value: f32 },
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
    NonPositiveCameraDepth { depth_m: f32 },
}

pub(crate) fn project_world_point_px(
    pose_world_to_camera: Pose,
    point_world_m: Point3,
    intrinsics: PinholeIntrinsics,
) -> Result<PinholeProjection, PinholeProjectionError> {
    let point_camera_m = math::transform_point(
        pose_world_to_camera.rotation(),
        pose_world_to_camera.translation(),
        [point_world_m.x, point_world_m.y, point_world_m.z],
    );
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
        return Ok(PinholeProjection::NonPositiveCameraDepth { depth_m });
    }
    let u_px = intrinsics.fx() * (point_camera_m[0] / depth_m) + intrinsics.cx();
    let v_px = intrinsics.fy() * (point_camera_m[1] / depth_m) + intrinsics.cy();
    for (axis, value) in [(ImagePlaneAxis::U, u_px), (ImagePlaneAxis::V, v_px)] {
        if !value.is_finite() {
            return Err(PinholeProjectionError::NonFinitePixelCoordinatePx { axis, value });
        }
    }
    Ok(PinholeProjection::Projected { u_px, v_px })
}

#[derive(Clone, Copy, Debug)]
pub struct RansacConfig {
    max_iterations: NonZeroUsize,
    reprojection_threshold_px: f32,
    min_inliers: usize,
    seed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RansacConfigError {
    ZeroMaxIterations,
    InvalidReprojectionThresholdPx { value: f32 },
    TooFewMinInliers { value: usize, minimum: usize },
}

impl std::fmt::Display for RansacConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RansacConfigError::ZeroMaxIterations => {
                write!(f, "RANSAC max iterations must be greater than zero")
            }
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
        self.seed
    }

    pub fn try_with_min_inliers(self, min_inliers: usize) -> Result<Self, RansacConfigError> {
        Self::new(
            self.max_iterations(),
            self.reprojection_threshold_px,
            min_inliers,
            self.seed,
        )
    }
}

impl Default for RansacConfig {
    fn default() -> Self {
        Self {
            max_iterations: NonZeroUsize::MIN.saturating_add(199),
            reprojection_threshold_px: 2.0,
            min_inliers: 20,
            seed: 0x5EED_u64,
        }
    }
}

#[derive(Debug)]
pub struct PnpResult {
    pose: Pose,
    inliers: Vec<usize>,
    iterations: NonZeroUsize,
    refinement: PnpRefinementStatus,
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
            | Self::Stationary { iteration }
            | Self::NoImprovement { iteration, .. }
            | Self::InvalidObjective { iteration, .. } => Some(*iteration),
            Self::LostConsensus { termination, .. } => Some(termination.iterations()),
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
    NotEnoughPoints { required: usize, actual: usize },
    NonFiniteObservation { field: &'static str, value: f32 },
    Degenerate { message: &'static str },
    NoSolution,
}

impl std::fmt::Display for PnpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PnpError::NotEnoughPoints { required, actual } => {
                write!(f, "pnp requires at least {required} points, got {actual}")
            }
            PnpError::NonFiniteObservation { field, value } => {
                write!(f, "pnp observation {field} must be finite, got {value}")
            }
            PnpError::Degenerate { message } => write!(f, "pnp degenerate input: {message}"),
            PnpError::NoSolution => write!(f, "pnp failed to find a valid pose"),
        }
    }
}

impl std::error::Error for PnpError {}

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

    let mut rng = XorShift64::new(config.seed());
    let mut best_pose = None;
    let mut best_inliers: Vec<usize> = Vec::new();
    let mut candidate_inliers = Vec::with_capacity(observations.len());

    let mut iterations = 0usize;
    let total = observations.len();
    let mut target_iterations = config.max_iterations();

    let threshold_px = config.reprojection_threshold_px();
    let threshold_sq_px2 = threshold_px * threshold_px;
    while iterations < target_iterations {
        iterations += 1;
        let sample = sample_three(&mut rng, total);
        let Some([a, b, c]) = sample else { continue };

        let obs = [&observations[a], &observations[b], &observations[c]];
        let candidates = p3p_solutions(obs);
        if candidates.is_empty() {
            continue;
        }

        for pose in candidates {
            candidate_inliers.clear();
            for (idx, obs) in observations.iter().enumerate() {
                if let Some(residual_sq_px2) = reprojection_error_sq_px2(pose, obs, intrinsics) {
                    if residual_sq_px2 <= threshold_sq_px2 {
                        candidate_inliers.push(idx);
                    }
                }
            }

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

    let pose = best_pose.ok_or(PnpError::NoSolution)?;
    if best_inliers.len() < config.min_inliers() {
        return Err(PnpError::NoSolution);
    }

    let (pose, inliers, refinement) =
        match refine_pose_on_inliers(pose, observations, intrinsics, &best_inliers) {
            Ok((refined_pose, termination)) => {
                let refined_inliers =
                    collect_inliers(refined_pose, observations, intrinsics, threshold_sq_px2);
                if refined_inliers.len() >= config.min_inliers() {
                    (
                        refined_pose,
                        refined_inliers,
                        PnpRefinementStatus::Applied { termination },
                    )
                } else {
                    let candidate_inliers = refined_inliers.len();
                    (
                        pose,
                        best_inliers,
                        PnpRefinementStatus::RetainedRansacPose {
                            reason: PnpRefinementFallback::LostConsensus {
                                termination,
                                candidate_inliers,
                                required_inliers: config.min_inliers(),
                            },
                        },
                    )
                }
            }
            Err(reason) => (
                pose,
                best_inliers,
                PnpRefinementStatus::RetainedRansacPose { reason },
            ),
        };

    Ok(PnpResult {
        pose,
        inliers,
        iterations: NonZeroUsize::new(iterations).ok_or(PnpError::NoSolution)?,
        refinement,
    })
}

fn adaptive_ransac_iterations(inlier_count: usize, total: usize, confidence: f32) -> usize {
    if inlier_count == 0 || total == 0 {
        return usize::MAX;
    }
    let inlier_ratio = (inlier_count as f32 / total as f32).clamp(0.0, 1.0);
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

fn collect_inliers(
    pose: Pose,
    observations: &[Observation],
    intrinsics: PinholeIntrinsics,
    threshold_sq_px2: f32,
) -> Vec<usize> {
    let mut inliers = Vec::with_capacity(observations.len());
    for (idx, obs) in observations.iter().enumerate() {
        if let Some(residual_sq_px2) = reprojection_error_sq_px2(pose, obs, intrinsics) {
            if residual_sq_px2 <= threshold_sq_px2 {
                inliers.push(idx);
            }
        }
    }
    inliers
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
            let Some(residual_sq_px2) = observations
                .get(idx)
                .and_then(|obs| reprojection_error_sq_px2(candidate, obs, intrinsics))
            else {
                candidate_nonprojectable = candidate_nonprojectable.saturating_add(1);
                continue;
            };
            candidate_cost += f64::from(residual_sq_px2);
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
    let x = (pixel.x - intrinsics.cx()) / intrinsics.fx();
    let y = (pixel.y - intrinsics.cy()) / intrinsics.fy();
    let v = [x, y, 1.0];
    let n = norm(v);
    if !n.is_finite() || n <= 0.0 {
        return Err(PnpError::Degenerate {
            message: "non-finite or zero-length bearing",
        });
    }
    Ok([v[0] / n, v[1] / n, v[2] / n])
}

fn p3p_solutions(obs: [&Observation; 3]) -> Vec<Pose> {
    let p1 = vec3_from_point(obs[0].world);
    let p2 = vec3_from_point(obs[1].world);
    let p3 = vec3_from_point(obs[2].world);
    let f1 = obs[0].bearing;
    let f2 = obs[1].bearing;
    let f3 = obs[2].bearing;

    let a = norm(sub(p2, p3));
    let b = norm(sub(p1, p3));
    let c = norm(sub(p1, p2));

    let scene_scale = a.max(b).max(c);
    if !scene_scale.is_finite() || scene_scale <= 0.0 {
        return Vec::new();
    }
    let normalized_a = a / scene_scale;
    let normalized_b = b / scene_scale;
    let normalized_c = c / scene_scale;

    let cos_alpha = dot(f2, f3);
    let cos_beta = dot(f1, f3);
    let cos_gamma = dot(f1, f2);

    let mut solutions = Vec::new();
    let mut roots = Vec::new();
    find_roots(
        cos_alpha,
        cos_beta,
        cos_gamma,
        normalized_a,
        normalized_b,
        normalized_c,
        &mut roots,
    );

    for (x, y) in roots {
        let denom = 1.0 + x * x - 2.0 * x * cos_gamma;
        if denom <= 0.0 {
            continue;
        }
        let d1 = c / denom.sqrt();
        let d2 = x * d1;
        let d3 = y * d1;
        if d1 <= 0.0 || d2 <= 0.0 || d3 <= 0.0 {
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
    cos_alpha: f32,
    cos_beta: f32,
    cos_gamma: f32,
    a: f32,
    b: f32,
    c: f32,
    roots: &mut Vec<(f32, f32)>,
) {
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
        let xf = x as f32;
        for sign in [-1.0_f32, 1.0_f32] {
            let Some(y) = y_from_x(xf, sign, cos_beta, cos_gamma, b, c) else {
                continue;
            };
            if y <= 0.0 {
                continue;
            }
            let Some(fx) = f_equation(xf, sign, &coeffs_meta) else {
                continue;
            };
            if fx.abs() < P3P_ROOT_TOLERANCE {
                push_unique_root(roots, (xf, y));
            }
        }
    }
}

struct P3pCoeffs {
    cos_alpha: f32,
    cos_beta: f32,
    cos_gamma: f32,
    a: f32,
    b: f32,
    c: f32,
}

fn f_equation(x: f32, sign: f32, coeffs: &P3pCoeffs) -> Option<f32> {
    let denom = 1.0 + x * x - 2.0 * x * coeffs.cos_gamma;
    if denom <= 0.0 {
        return None;
    }
    let k = (coeffs.b * coeffs.b / (coeffs.c * coeffs.c)) * denom;
    let disc = k + coeffs.cos_beta * coeffs.cos_beta - 1.0;
    if disc < 0.0 {
        return None;
    }
    let y = coeffs.cos_beta + sign * disc.sqrt();
    let num = x * x + y * y - 2.0 * x * y * coeffs.cos_alpha;
    Some(coeffs.a * coeffs.a - (coeffs.c * coeffs.c) * (num / denom))
}

fn y_from_x(x: f32, sign: f32, cos_beta: f32, cos_gamma: f32, b: f32, c: f32) -> Option<f32> {
    let denom = 1.0 + x * x - 2.0 * x * cos_gamma;
    if denom <= 0.0 {
        return None;
    }
    let k = (b * b / (c * c)) * denom;
    let disc = k + cos_beta * cos_beta - 1.0;
    if disc < 0.0 {
        return None;
    }
    Some(cos_beta + sign * disc.sqrt())
}

fn quartic_coeffs(
    cos_alpha: f32,
    cos_beta: f32,
    cos_gamma: f32,
    a: f32,
    b: f32,
    c: f32,
) -> Option<[f64; 5]> {
    if a <= 0.0 || b <= 0.0 || c <= 0.0 {
        return None;
    }
    let a2 = (a as f64) * (a as f64);
    let b2 = (b as f64) * (b as f64);
    let c2 = (c as f64) * (c as f64);
    if !a2.is_finite() || !b2.is_finite() || !c2.is_finite() || c2 <= 0.0 {
        return None;
    }

    let ca = cos_alpha as f64;
    let cb = cos_beta as f64;
    let cg = cos_gamma as f64;

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

/// Relative tolerance for treating normalized polynomial coefficients as zero.
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
const ROOT_UNIQUENESS_TOLERANCE: f32 = 1e-3;
/// Tolerance for accepting the dimensionless P3P equation evaluation as a valid root.
const P3P_ROOT_TOLERANCE: f32 = 1e-3;
/// Target confidence used to adaptively shorten RANSAC once a strong model exists.
const RANSAC_CONFIDENCE: f32 = 0.99;
/// Number of nonlinear pose-only refinement steps on the best inlier set.
const PNP_REFINEMENT_ITERS: usize = 8;
/// Damping added to the normal equations during PnP pose refinement.
const PNP_REFINEMENT_DAMPING: f32 = 1e-4;
/// Translation-tangent convergence threshold for PnP pose refinement, in meters.
const PNP_REFINEMENT_TRANSLATION_CONVERGENCE_M: f64 = 1e-5;
/// Rotation-vector convergence threshold for PnP pose refinement, in radians.
const PNP_REFINEMENT_ROTATION_CONVERGENCE_RAD: f64 = 1e-5;

fn solve_real_roots(coeffs: [f64; 5]) -> Vec<f64> {
    if coeffs.iter().any(|coefficient| !coefficient.is_finite()) {
        return Vec::new();
    }
    let scale = coeffs
        .iter()
        .map(|coefficient| coefficient.abs())
        .fold(0.0, f64::max);
    if scale == 0.0 {
        return Vec::new();
    }
    let mut coeffs: Vec<f64> = coeffs
        .into_iter()
        .map(|coefficient| coefficient / scale)
        .collect();
    while coeffs.len() > 1
        && coeffs.last().copied().unwrap_or(0.0).abs() <= POLY_RELATIVE_COEFFICIENT_TOLERANCE
    {
        coeffs.pop();
    }
    let degree = coeffs.len().saturating_sub(1);
    if degree == 0 {
        return Vec::new();
    }
    if degree == 1 {
        let c1 = coeffs[1];
        if c1.abs() <= POLY_RELATIVE_COEFFICIENT_TOLERANCE {
            return Vec::new();
        }
        return vec![-coeffs[0] / c1];
    }

    let Some(&lead) = coeffs.last() else {
        return Vec::new();
    };
    if lead.abs() <= POLY_RELATIVE_COEFFICIENT_TOLERANCE {
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
        (self.re * self.re + self.im * self.im).sqrt()
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

fn push_unique_root(roots: &mut Vec<(f32, f32)>, candidate: (f32, f32)) {
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

fn reprojection_error_sq_px2(
    pose: Pose,
    obs: &Observation,
    intrinsics: PinholeIntrinsics,
) -> Option<f32> {
    let pc = math::transform_point(
        pose.rotation(),
        pose.translation(),
        vec3_from_point(obs.world),
    );
    if pc.iter().any(|value| !value.is_finite()) || pc[2] <= 0.0 {
        return None;
    }
    let u = intrinsics.fx() * (pc[0] / pc[2]) + intrinsics.cx();
    let v = intrinsics.fy() * (pc[1] / pc[2]) + intrinsics.cy();
    let dx = u - obs.pixel.x;
    let dy = v - obs.pixel.y;
    let residual_sq_px2 = dx * dx + dy * dy;
    residual_sq_px2.is_finite().then_some(residual_sq_px2)
}

fn try_reprojection_error_sq_px2(
    pose: Pose,
    observation: &Observation,
    intrinsics: PinholeIntrinsics,
) -> Result<Option<f64>, PinholeProjectionError> {
    let PinholeProjection::Projected { u_px, v_px } =
        project_world_point_px(pose, observation.world, intrinsics)?
    else {
        return Ok(None);
    };
    let residual_u_px = f64::from(u_px) - f64::from(observation.pixel.x);
    let residual_v_px = f64::from(v_px) - f64::from(observation.pixel.y);
    Ok(Some(
        residual_u_px.mul_add(residual_u_px, residual_v_px * residual_v_px),
    ))
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
    ResidualOutsideF32PixelDomain {
        observation_index: usize,
        value_px: f64,
    },
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
            Self::ResidualOutsideF32PixelDomain {
                observation_index,
                value_px,
            } => write!(
                f,
                "reprojection residual magnitude at observation {observation_index} is outside the finite f32 pixel domain: {value_px} px"
            ),
        }
    }
}

impl std::error::Error for ReprojectionEvaluationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Allocation { source, .. } => Some(source),
            Self::Projection { source, .. } => Some(source),
            Self::ResidualOutsideF32PixelDomain { .. } => None,
        }
    }
}

pub(crate) fn reprojection_residuals_px(
    pose: &Pose,
    observations: &[Observation],
    intrinsics: PinholeIntrinsics,
) -> Result<Vec<Option<f32>>, ReprojectionEvaluationError> {
    let mut residuals_px = Vec::new();
    residuals_px
        .try_reserve_exact(observations.len())
        .map_err(|source| ReprojectionEvaluationError::Allocation {
            observation_count: observations.len(),
            source,
        })?;
    for (observation_index, observation) in observations.iter().enumerate() {
        let residual_px = match try_reprojection_error_sq_px2(*pose, observation, intrinsics)
            .map_err(|source| ReprojectionEvaluationError::Projection {
                observation_index,
                source,
            })? {
            Some(error_sq_px2) => {
                let value_px = error_sq_px2.sqrt();
                if value_px > f64::from(f32::MAX) {
                    return Err(ReprojectionEvaluationError::ResidualOutsideF32PixelDomain {
                        observation_index,
                        value_px,
                    });
                }
                Some(value_px as f32)
            }
            None => None,
        };
        residuals_px.push(residual_px);
    }
    Ok(residuals_px)
}

pub(crate) fn reprojection_rmse_px(residuals_px: &[Option<f32>]) -> Option<f32> {
    // Accumulating in f64 keeps the square and sum finite for any addressable
    // slice of finite f32 residuals; their RMS still fits the f32 input domain.
    let mut sum_sq = 0.0_f64;
    let mut count = 0usize;
    for &residual_px in residuals_px.iter().flatten() {
        let residual_px = f64::from(residual_px);
        sum_sq = residual_px.mul_add(residual_px, sum_sq);
        count += 1;
    }
    if count == 0 {
        return None;
    }
    Some((sum_sq / count as f64).sqrt() as f32)
}

pub(crate) fn reprojection_max_px(residuals_px: &[Option<f32>]) -> Option<f32> {
    residuals_px.iter().flatten().copied().reduce(f32::max)
}

pub(crate) fn reprojection_mse_per_axis_px2(residuals_px: &[Option<f32>]) -> Option<f64> {
    let mut sum_sq = 0.0_f64;
    let mut count = 0usize;
    for &residual_px in residuals_px.iter().flatten() {
        let residual_px = f64::from(residual_px);
        sum_sq = residual_px.mul_add(residual_px, sum_sq);
        count += 1;
    }
    if count == 0 {
        return None;
    }
    Some(sum_sq / (2.0 * count as f64))
}

fn pose_from_points(
    w1: [f32; 3],
    w2: [f32; 3],
    w3: [f32; 3],
    c1: [f32; 3],
    c2: [f32; 3],
    c3: [f32; 3],
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

    let t = sub(c1, math::mat_mul_vec(r, w1));
    Some(Pose {
        rotation: r,
        translation: t,
    })
}

fn mat_from_cols(
    xc: [f32; 3],
    yc: [f32; 3],
    zc: [f32; 3],
    xw: [f32; 3],
    yw: [f32; 3],
    zw: [f32; 3],
) -> [[f32; 3]; 3] {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        r[i][0] = xc[i] * xw[0] + yc[i] * yw[0] + zc[i] * zw[0];
        r[i][1] = xc[i] * xw[1] + yc[i] * yw[1] + zc[i] * zw[1];
        r[i][2] = xc[i] * xw[2] + yc[i] * yw[2] + zc[i] * zw[2];
    }
    r
}

fn det(r: [[f32; 3]; 3]) -> f32 {
    r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
}

fn vec3_from_point(p: Point3) -> [f32; 3] {
    [p.x, p.y, p.z]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn mul(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> Option<[f32; 3]> {
    let n = norm(v);
    if n <= 0.0 {
        return None;
    }
    Some([v[0] / n, v[1] / n, v[2] / n])
}

#[derive(Debug)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
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
    fn polynomial_roots_are_invariant_to_coefficient_scale() {
        let base = [-6.0, 11.0, -6.0, 1.0, 0.0];
        for scale in [1e-18, 1.0, 1e18] {
            let mut roots = solve_real_roots(base.map(|coefficient| coefficient * scale));
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

        for scale in [1e-3_f32, 1.0, 1e3] {
            let world = base_world.map(|point| Point3 {
                x: point.x * scale,
                y: point.y * scale,
                z: point.z * scale,
            });
            let expected = axis_angle_pose(
                base_translation.map(|component| component * scale),
                axis_angle,
            );
            let observations =
                observations_from_projection(expected, &world, intrinsics).expect("observations");
            assert_eq!(observations.len(), 3);
            let solutions = p3p_solutions([&observations[0], &observations[1], &observations[2]]);
            let matched = solutions.iter().any(|solution| {
                rot_frob_norm(solution.rotation(), expected.rotation()) < 2e-3
                    && l2(solution.translation(), expected.translation()) / scale < 2e-2
            });
            assert!(
                matched,
                "no correct P3P solution at scene scale {scale:e}; candidates={solutions:?}"
            );
        }
    }

    #[test]
    fn sample_three_returns_distinct_indices() {
        let mut rng = XorShift64::new(0xDEADBEEF);
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

        let rot_err = rot_frob_norm(result.pose().rotation(), pose_gt.rotation());
        let trans_err = l2(result.pose().translation(), pose_gt.translation());
        assert!(rot_err < 0.03, "rotation error too high: {rot_err}");
        assert!(trans_err < 0.08, "translation error too high: {trans_err}");
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

        let initial_rmse = reprojection_rmse_px(&initial_residuals_px).expect("initial rmse");
        let refined_rmse = reprojection_rmse_px(&refined_residuals_px).expect("refined rmse");
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
                    f64::from(
                        reprojection_error_sq_px2(initial, &observations[index], intrinsics)
                            .expect("initial projection"),
                    )
                })
                .sum();
            if let Ok((refined, _)) =
                refine_pose_on_inliers(initial, &observations, intrinsics, &inlier_indices)
            {
                let refined_cost: f64 = inlier_indices
                    .iter()
                    .map(|&index| {
                        f64::from(
                            reprojection_error_sq_px2(refined, &observations[index], intrinsics)
                                .expect("refined projection"),
                        )
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
            PnpError::NotEnoughPoints { required, actual } => {
                assert_eq!(required, 4);
                assert_eq!(actual, 3);
            }
            other => panic!("unexpected error: {other:?}"),
        }
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
        let residual_sq_px2 =
            reprojection_error_sq_px2(pose, &obs, intrinsics).expect("projectable residual");
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
    fn reprojection_residual_rejects_values_outside_f32_pixel_domain() {
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

        let squared_px2 = try_reprojection_error_sq_px2(Pose::identity(), &observation, intrinsics)
            .expect("finite projection")
            .expect("positive camera depth");
        assert!(squared_px2.is_finite());
        assert!(squared_px2 > f64::from(f32::MAX).powi(2));

        let error = reprojection_residuals_px(&Pose::identity(), &[observation], intrinsics)
            .expect_err("f64 residual outside the f32 diagnostic domain must be explicit");
        assert!(matches!(
            error,
            ReprojectionEvaluationError::ResidualOutsideF32PixelDomain {
                observation_index: 0,
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
        let rmse = reprojection_rmse_px(&residuals_px).expect("rmse");
        let expected = ((3.0_f32 * 3.0 + 4.0 * 4.0) / 2.0).sqrt();
        assert!((rmse - expected).abs() < 1e-6);
    }

    #[test]
    fn reprojection_statistics_do_not_overflow_on_finite_f32_residuals() {
        let residuals_px = [Some(f32::MAX), Some(f32::MAX)];

        let rmse = reprojection_rmse_px(&residuals_px).expect("finite RMSE");
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
        let rmse = reprojection_rmse_px(&residuals_px).expect("rmse");
        assert!((1.5..=2.5).contains(&rmse), "rmse={rmse}");
        let max = reprojection_max_px(&residuals_px).expect("max");
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
