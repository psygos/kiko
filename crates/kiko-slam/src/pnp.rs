use crate::dataset::CameraIntrinsics;
use std::marker::PhantomData;
use std::num::NonZeroU64;

use crate::{
    CameraFrame, CoordinateFrame, Keyframe, Keypoint, Matches, Verified, WorldFrame, WorldPoint3,
    math,
};

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
    ZeroDimensions { width: u32, height: u32 },
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
            IntrinsicsError::ZeroDimensions { width, height } => write!(
                f,
                "pinhole image dimensions must be nonzero (width={width}, height={height})"
            ),
        }
    }
}

impl std::error::Error for IntrinsicsError {}

impl TryFrom<&CameraIntrinsics> for PinholeIntrinsics {
    type Error = IntrinsicsError;

    fn try_from(value: &CameraIntrinsics) -> Result<Self, Self::Error> {
        if !value.fx.is_finite()
            || !value.fy.is_finite()
            || !value.cx.is_finite()
            || !value.cy.is_finite()
        {
            return Err(IntrinsicsError::NonFinite {
                fx: value.fx,
                fy: value.fy,
                cx: value.cx,
                cy: value.cy,
            });
        }
        if value.fx <= 0.0 || value.fy <= 0.0 {
            return Err(IntrinsicsError::NonPositiveFocal {
                fx: value.fx,
                fy: value.fy,
            });
        }
        if value.width == 0 || value.height == 0 {
            return Err(IntrinsicsError::ZeroDimensions {
                width: value.width,
                height: value.height,
            });
        }
        Ok(Self {
            fx: value.fx,
            fy: value.fy,
            cx: value.cx,
            cy: value.cy,
        })
    }
}

impl PinholeIntrinsics {
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
    world: WorldPoint3,
    pixel: Keypoint,
    bearing: [f32; 3],
}

impl Observation {
    pub fn world(&self) -> WorldPoint3 {
        self.world
    }

    pub fn pixel(&self) -> Keypoint {
        self.pixel
    }

    pub fn bearing(&self) -> [f32; 3] {
        self.bearing
    }

    pub fn try_new(
        world: WorldPoint3,
        pixel: Keypoint,
        intrinsics: PinholeIntrinsics,
    ) -> Result<Self, PnpError> {
        if !world.x.is_finite()
            || !world.y.is_finite()
            || !world.z.is_finite()
            || !pixel.x.is_finite()
            || !pixel.y.is_finite()
        {
            return Err(PnpError::Degenerate {
                message: "observation coordinates must be finite",
            });
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

    /// Compose two poses: `self ∘ other`.
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

    pub fn transform_point(self, point: crate::Point3<From>) -> crate::Point3<To> {
        crate::Point3::from_array(math::transform_point(
            self.pose.rotation(),
            self.pose.translation(),
            point.to_array(),
        ))
    }

    pub fn inverse(self) -> Transform<To, From> {
        Transform::from_legacy_pose(self.pose.inverse())
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
    reprojection_threshold_sq_px2: f64,
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
            reprojection_threshold_sq_px2: f64::from(reprojection_threshold_px).powi(2),
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

    fn reprojection_threshold_sq_px2(self) -> f64 {
        self.reprojection_threshold_sq_px2
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
            reprojection_threshold_sq_px2: 4.0,
            min_inliers: 20,
            seed: NonZeroU64::new(0x5EED_u64).expect("default RANSAC seed is nonzero"),
        }
    }
}

#[derive(Debug)]
pub struct PnpResult {
    pub pose: WorldToCamera,
    pub inliers: Vec<usize>,
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
    Degenerate {
        message: &'static str,
    },
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
            PnpError::Degenerate { message } => write!(f, "pnp degenerate input: {message}"),
            PnpError::NoSolution => write!(f, "pnp failed to find a valid pose"),
        }
    }
}

impl std::error::Error for PnpError {}

pub fn build_observations(
    keyframe: &Keyframe,
    matches: &Matches<Verified>,
    keyframe_to_world: CameraToWorld,
    intrinsics: PinholeIntrinsics,
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
        let bearing = normalize_bearing(pixel, intrinsics)?;
        let camera = keyframe
            .landmark_for_detection(ki)
            .ok_or(PnpError::MissingLandmark { keyframe_index: ki })?;
        let world = keyframe_to_world.transform_point(camera);
        observations.push(Observation {
            world,
            pixel,
            bearing,
        });
    }

    Ok(observations)
}

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

    let mut rng = XorShift64::new(config.seed);
    let mut best_pose = None;
    let mut best_inliers: Vec<usize> = Vec::new();

    let mut iterations = 0usize;
    let total = observations.len();

    let threshold_sq = config.reprojection_threshold_sq_px2();
    while iterations < config.max_iterations {
        iterations += 1;
        let sample = sample_three(&mut rng, total);
        let Some([a, b, c]) = sample else { continue };

        let obs = [&observations[a], &observations[b], &observations[c]];
        let candidates = p3p_solutions(obs);
        if candidates.is_empty() {
            continue;
        }

        for pose in candidates {
            let mut inliers = Vec::new();
            for (idx, obs) in observations.iter().enumerate() {
                if let Some(err_sq) = reprojection_error_sq_px(pose, obs, intrinsics)
                    && err_sq <= threshold_sq
                {
                    inliers.push(idx);
                }
            }

            if inliers.len() > best_inliers.len() {
                best_inliers = inliers;
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

pub fn solve_pnp(
    keyframe: &Keyframe,
    matches: &Matches<Verified>,
    keyframe_to_world: CameraToWorld,
    intrinsics: PinholeIntrinsics,
    config: RansacConfig,
) -> Result<PnpResult, PnpError> {
    let observations = build_observations(keyframe, matches, keyframe_to_world, intrinsics)?;
    solve_pnp_ransac(&observations, intrinsics, config)
}

fn normalize_bearing(pixel: Keypoint, intrinsics: PinholeIntrinsics) -> Result<[f32; 3], PnpError> {
    let x = (f64::from(pixel.x) - f64::from(intrinsics.cx())) / f64::from(intrinsics.fx());
    let y = (f64::from(pixel.y) - f64::from(intrinsics.cy())) / f64::from(intrinsics.fy());
    let norm = x.hypot(y).hypot(1.0);
    if !norm.is_finite() || norm <= 0.0 {
        return Err(PnpError::Degenerate {
            message: "bearing coordinates must be finite",
        });
    }
    Ok([(x / norm) as f32, (y / norm) as f32, (1.0 / norm) as f32])
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

    if !a.is_finite() || !b.is_finite() || !c.is_finite() || a <= 0.0 || b <= 0.0 || c <= 0.0 {
        return Vec::new();
    }

    let cos_alpha = dot(f2, f3);
    let cos_beta = dot(f1, f3);
    let cos_gamma = dot(f1, f2);

    let mut solutions = Vec::new();
    let mut roots = Vec::new();
    find_roots(cos_alpha, cos_beta, cos_gamma, a, b, c, &mut roots);

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
            if fx.abs() <= P3P_ROOT_TOLERANCE {
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
const ROOT_UNIQUENESS_TOLERANCE: f32 = 1e-3;
/// Dimensionless tolerance for accepting a normalized P3P equation root.
const P3P_ROOT_TOLERANCE: f32 = 1e-3;

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

fn reprojection_error_sq_px(
    pose: Pose,
    obs: &Observation,
    intrinsics: PinholeIntrinsics,
) -> Option<f64> {
    let pc = math::transform_point(
        pose.rotation(),
        pose.translation(),
        vec3_from_point(obs.world),
    );
    if pc.iter().any(|value| !value.is_finite()) || pc[2] <= 0.0 {
        return None;
    }
    let inverse_depth = 1.0 / f64::from(pc[2]);
    let u =
        f64::from(intrinsics.fx()) * f64::from(pc[0]) * inverse_depth + f64::from(intrinsics.cx());
    let v =
        f64::from(intrinsics.fy()) * f64::from(pc[1]) * inverse_depth + f64::from(intrinsics.cy());
    let dx = u - f64::from(obs.pixel.x);
    let dy = v - f64::from(obs.pixel.y);
    let error_sq = dx.mul_add(dx, dy * dy);
    error_sq.is_finite().then_some(error_sq)
}

pub(crate) fn reprojection_errors(
    pose: &Pose,
    observations: &[Observation],
    intrinsics: PinholeIntrinsics,
) -> Vec<Option<f32>> {
    observations
        .iter()
        .map(|obs| {
            reprojection_error_sq_px(*pose, obs, intrinsics).map(|error_sq| error_sq.sqrt() as f32)
        })
        .collect()
}

pub(crate) fn reprojection_rmse(errors: &[Option<f32>]) -> Option<f32> {
    let mut sum_sq = 0.0_f32;
    let mut count = 0usize;
    for &error in errors.iter().flatten() {
        sum_sq += error * error;
        count = count.saturating_add(1);
    }
    if count == 0 {
        return None;
    }
    Some((sum_sq / count as f32).sqrt())
}

pub(crate) fn reprojection_max(errors: &[Option<f32>]) -> Option<f32> {
    errors.iter().flatten().copied().reduce(f32::max)
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

fn vec3_from_point(p: WorldPoint3) -> [f32; 3] {
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
    use crate::Point3;
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

        let bearing = normalize_bearing(pixel, intrinsics).expect("finite bearing");
        assert!(bearing.into_iter().all(f32::is_finite));
        let norm = f64::from(bearing[0])
            .hypot(f64::from(bearing[1]))
            .hypot(f64::from(bearing[2]));
        assert!(
            (norm - 1.0).abs() <= 2.0 * f64::from(f32::EPSILON),
            "narrowed bearing norm must remain one, got {norm}"
        );
        assert!(bearing[0].is_sign_positive());
        assert!(bearing[1].is_sign_negative());
    }

    #[test]
    fn normalize_bearing_preserves_center_and_image_axis_signs() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 100.0, 100.0, 320.0, 240.0).expect("intrinsics");

        let center =
            normalize_bearing(Keypoint { x: 320.0, y: 240.0 }, intrinsics).expect("center bearing");
        assert_eq!(center, [0.0, 0.0, 1.0]);

        let left_and_down = normalize_bearing(Keypoint { x: 220.0, y: 340.0 }, intrinsics)
            .expect("off-center bearing");
        assert!(left_and_down[0].is_sign_negative());
        assert!(left_and_down[1].is_sign_positive());
        assert!(left_and_down[2].is_sign_positive());
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
        let recovered = pose.inverse().inverse();
        assert!(rot_frob_norm(pose.rotation(), recovered.rotation()) < 1e-5);
        assert!(l2(pose.translation(), recovered.translation()) < 1e-5);
    }

    #[test]
    fn pose_compose_matches_pose64_ordering() {
        let first = axis_angle_pose([0.3, -0.2, 0.7], [0.1, -0.05, 0.08]);
        let second = axis_angle_pose([-0.1, 0.4, 0.2], [-0.03, 0.09, 0.02]);
        let composed = first.compose(second);
        let composed64 = crate::Pose64::from_pose32(first)
            .compose(crate::Pose64::from_pose32(second))
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

        for scale in [1e-3_f32, 1.0, 1e3] {
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
            let observations =
                observations_from_projection(pose, &world, intrinsics).expect("observations");
            let result = solve_pnp_ransac(&observations, intrinsics, config)
                .unwrap_or_else(|err| panic!("P3P failed at scene scale {scale}: {err}"));

            assert!(result.inliers.len() >= 20, "scale={scale}");
            assert!(
                rot_frob_norm(result.pose.rotation(), pose.rotation()) < 0.03,
                "rotation error at scale={scale}"
            );
            assert!(
                l2(result.pose.translation(), pose.translation()) < 0.08 * scale,
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
            with_outliers
                .push(Observation::try_new(obs.world(), pixel, intrinsics).expect("observation"));
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
            .expect("every positive finite f32 threshold has a finite f64 square");
        assert!(extreme.reprojection_threshold_sq_px2().is_finite());
        assert!(extreme.reprojection_threshold_sq_px2() > f64::from(f32::MAX));
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
        let err_sq = reprojection_error_sq_px(pose, &obs, intrinsics).expect("error");
        assert!(err_sq < 1e-8, "expected exact reprojection, got {err_sq}");
    }

    #[test]
    fn reprojection_errors_zero_for_exact_projection() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let pose = Pose::identity();
        let world = synthetic_world_points();
        let observations =
            observations_from_projection(pose, &world, intrinsics).expect("synthetic observations");
        let errors = reprojection_errors(&pose, &observations, intrinsics);
        assert_eq!(errors.len(), observations.len());
        assert!(errors.iter().all(|e| e.is_some_and(|v| v < 1e-4)));
    }

    #[test]
    fn reprojection_errors_none_for_behind_camera() {
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
        let errors = reprojection_errors(&behind_pose, &[observation], intrinsics);
        assert_eq!(errors, vec![None]);
    }

    #[test]
    fn reprojection_rmse_matches_manual() {
        let errors = vec![Some(3.0), None, Some(4.0)];
        let rmse = reprojection_rmse(&errors).expect("rmse");
        let expected = ((3.0_f32 * 3.0 + 4.0 * 4.0) / 2.0).sqrt();
        assert!((rmse - expected).abs() < 1e-6);
    }

    #[test]
    fn reprojection_errors_with_known_perturbation() {
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
        let errors = reprojection_errors(&pose, &observations, intrinsics);
        let rmse = reprojection_rmse(&errors).expect("rmse");
        assert!((1.5..=2.5).contains(&rmse), "rmse={rmse}");
        let max = reprojection_max(&errors).expect("max");
        assert!((1.5..=2.5).contains(&max), "max={max}");
    }

    #[test]
    fn reprojection_errors_len_matches_input() {
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
        let errors = reprojection_errors(&pose, &observations, intrinsics);
        assert_eq!(errors.len(), observations.len());
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
