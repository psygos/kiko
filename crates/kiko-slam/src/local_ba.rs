use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;

use crate::{
    map::{KeyframeId, KeyframeKeypoint, MapPointId, SlamMap},
    math, Keypoint, Observation, PinholeIntrinsics, Point3, Pose,
};

#[derive(Clone, Copy, Debug)]
pub struct LocalBaConfig {
    window: NonZeroUsize,
    max_iterations: NonZeroUsize,
    min_observations: NonZeroUsize,
    huber_delta_px: f32,
    damping: f32,
    motion_prior_weight: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct LmConfig {
    initial_lambda: f32,
    lambda_factor: f32,
    min_lambda: f32,
    max_lambda: f32,
    rho_accept: f32,
    rho_good: f32,
}

#[derive(Debug)]
pub enum LmConfigError {
    NonPositiveInitialLambda { value: f32 },
    NonPositiveLambdaFactor { value: f32 },
    LambdaFactorTooSmall { value: f32 },
    NonPositiveMinLambda { value: f32 },
    NonPositiveMaxLambda { value: f32 },
    MinLambdaExceedsMax { min: f32, max: f32 },
    InvalidRhoAccept { value: f32 },
    InvalidRhoGood { value: f32 },
    InvalidRhoOrdering { rho_accept: f32, rho_good: f32 },
}

impl std::fmt::Display for LmConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LmConfigError::NonPositiveInitialLambda { value } => {
                write!(f, "LM initial lambda must be > 0 (got {value})")
            }
            LmConfigError::NonPositiveLambdaFactor { value } => {
                write!(f, "LM lambda factor must be > 0 (got {value})")
            }
            LmConfigError::LambdaFactorTooSmall { value } => {
                write!(f, "LM lambda factor must be > 1 (got {value})")
            }
            LmConfigError::NonPositiveMinLambda { value } => {
                write!(f, "LM min lambda must be > 0 (got {value})")
            }
            LmConfigError::NonPositiveMaxLambda { value } => {
                write!(f, "LM max lambda must be > 0 (got {value})")
            }
            LmConfigError::MinLambdaExceedsMax { min, max } => {
                write!(f, "LM min lambda must be <= max lambda (min={min}, max={max})")
            }
            LmConfigError::InvalidRhoAccept { value } => {
                write!(f, "LM rho_accept must be in (0, 1) (got {value})")
            }
            LmConfigError::InvalidRhoGood { value } => {
                write!(f, "LM rho_good must be in (0, 1) (got {value})")
            }
            LmConfigError::InvalidRhoOrdering {
                rho_accept,
                rho_good,
            } => write!(
                f,
                "LM requires rho_accept < rho_good (rho_accept={rho_accept}, rho_good={rho_good})"
            ),
        }
    }
}

impl std::error::Error for LmConfigError {}

impl LmConfig {
    pub fn new(
        initial_lambda: f32,
        lambda_factor: f32,
        min_lambda: f32,
        max_lambda: f32,
        rho_accept: f32,
        rho_good: f32,
    ) -> Result<Self, LmConfigError> {
        if !initial_lambda.is_finite() || initial_lambda <= 0.0 {
            return Err(LmConfigError::NonPositiveInitialLambda {
                value: initial_lambda,
            });
        }
        if !lambda_factor.is_finite() || lambda_factor <= 0.0 {
            return Err(LmConfigError::NonPositiveLambdaFactor {
                value: lambda_factor,
            });
        }
        if lambda_factor <= 1.0 {
            return Err(LmConfigError::LambdaFactorTooSmall {
                value: lambda_factor,
            });
        }
        if !min_lambda.is_finite() || min_lambda <= 0.0 {
            return Err(LmConfigError::NonPositiveMinLambda { value: min_lambda });
        }
        if !max_lambda.is_finite() || max_lambda <= 0.0 {
            return Err(LmConfigError::NonPositiveMaxLambda { value: max_lambda });
        }
        if min_lambda > max_lambda {
            return Err(LmConfigError::MinLambdaExceedsMax {
                min: min_lambda,
                max: max_lambda,
            });
        }
        if !rho_accept.is_finite() || rho_accept <= 0.0 || rho_accept >= 1.0 {
            return Err(LmConfigError::InvalidRhoAccept { value: rho_accept });
        }
        if !rho_good.is_finite() || rho_good <= 0.0 || rho_good >= 1.0 {
            return Err(LmConfigError::InvalidRhoGood { value: rho_good });
        }
        if rho_accept >= rho_good {
            return Err(LmConfigError::InvalidRhoOrdering {
                rho_accept,
                rho_good,
            });
        }
        Ok(Self {
            initial_lambda,
            lambda_factor,
            min_lambda,
            max_lambda,
            rho_accept,
            rho_good,
        })
    }

    pub fn initial_lambda(self) -> f32 {
        self.initial_lambda
    }

    pub fn lambda_factor(self) -> f32 {
        self.lambda_factor
    }

    pub fn min_lambda(self) -> f32 {
        self.min_lambda
    }

    pub fn max_lambda(self) -> f32 {
        self.max_lambda
    }

    pub fn rho_accept(self) -> f32 {
        self.rho_accept
    }

    pub fn rho_good(self) -> f32 {
        self.rho_good
    }
}

impl Default for LmConfig {
    fn default() -> Self {
        Self {
            initial_lambda: 1e-4,
            lambda_factor: 10.0,
            min_lambda: 1e-8,
            max_lambda: 1e4,
            rho_accept: 0.25,
            rho_good: 0.75,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LmAction {
    Accept,
    Reject,
}

#[derive(Clone, Copy, Debug)]
struct LmState {
    lambda: f32,
    prev_cost: f64,
}

impl LmState {
    fn new(config: LmConfig, prev_cost: f64) -> Self {
        Self {
            lambda: config.initial_lambda(),
            prev_cost,
        }
    }

    fn lambda(self) -> f32 {
        self.lambda
    }

    fn prev_cost(self) -> f64 {
        self.prev_cost
    }

    fn step(&mut self, cost: f64, predicted_decrease: f64, config: LmConfig) -> LmAction {
        if !cost.is_finite() || !predicted_decrease.is_finite() || predicted_decrease <= 0.0 {
            self.lambda = (self.lambda * config.lambda_factor()).min(config.max_lambda());
            return LmAction::Reject;
        }

        let rho = (self.prev_cost - cost) / predicted_decrease;
        if rho >= config.rho_accept() as f64 {
            self.prev_cost = cost;
            if rho > config.rho_good() as f64 {
                self.lambda = (self.lambda / config.lambda_factor()).max(config.min_lambda());
            }
            LmAction::Accept
        } else {
            self.lambda = (self.lambda * config.lambda_factor()).min(config.max_lambda());
            LmAction::Reject
        }
    }
}

#[derive(Debug)]
pub enum LocalBaConfigError {
    ZeroWindow,
    ZeroIterations,
    ZeroObservations,
    TooFewObservations { min: usize },
    NonPositiveHuber { value: f32 },
    NegativeDamping { value: f32 },
    NegativeMotionWeight { value: f32 },
}

impl std::fmt::Display for LocalBaConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LocalBaConfigError::ZeroWindow => write!(f, "local BA window must be > 0"),
            LocalBaConfigError::ZeroIterations => write!(f, "local BA iterations must be > 0"),
            LocalBaConfigError::ZeroObservations => {
                write!(f, "local BA min observations must be > 0")
            }
            LocalBaConfigError::TooFewObservations { min } => {
                write!(f, "local BA min observations must be >= {min}")
            }
            LocalBaConfigError::NonPositiveHuber { value } => {
                write!(f, "local BA huber delta must be > 0 (got {value})")
            }
            LocalBaConfigError::NegativeDamping { value } => {
                write!(f, "local BA damping must be >= 0 (got {value})")
            }
            LocalBaConfigError::NegativeMotionWeight { value } => {
                write!(f, "local BA motion prior weight must be >= 0 (got {value})")
            }
        }
    }
}

impl std::error::Error for LocalBaConfigError {}

impl LocalBaConfig {
    pub fn new(
        window: usize,
        max_iterations: usize,
        min_observations: usize,
        huber_delta_px: f32,
        damping: f32,
        motion_prior_weight: f32,
    ) -> Result<Self, LocalBaConfigError> {
        let window = NonZeroUsize::new(window).ok_or(LocalBaConfigError::ZeroWindow)?;
        let max_iterations =
            NonZeroUsize::new(max_iterations).ok_or(LocalBaConfigError::ZeroIterations)?;
        let min_observations =
            NonZeroUsize::new(min_observations).ok_or(LocalBaConfigError::ZeroObservations)?;
        if min_observations.get() < 4 {
            return Err(LocalBaConfigError::TooFewObservations { min: 4 });
        }
        if huber_delta_px <= 0.0 || !huber_delta_px.is_finite() {
            return Err(LocalBaConfigError::NonPositiveHuber {
                value: huber_delta_px,
            });
        }
        if damping < 0.0 || !damping.is_finite() {
            return Err(LocalBaConfigError::NegativeDamping { value: damping });
        }
        if motion_prior_weight < 0.0 || !motion_prior_weight.is_finite() {
            return Err(LocalBaConfigError::NegativeMotionWeight {
                value: motion_prior_weight,
            });
        }
        Ok(Self {
            window,
            max_iterations,
            min_observations,
            huber_delta_px,
            damping,
            motion_prior_weight,
        })
    }

    pub fn window(&self) -> usize {
        self.window.get()
    }

    pub fn max_iterations(&self) -> usize {
        self.max_iterations.get()
    }

    pub fn min_observations(&self) -> usize {
        self.min_observations.get()
    }

    pub fn huber_delta_px(&self) -> f32 {
        self.huber_delta_px
    }

    pub fn damping(&self) -> f32 {
        self.damping
    }

    pub fn motion_prior_weight(&self) -> f32 {
        self.motion_prior_weight
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum BaResult {
    Converged { iterations: usize, final_cost: f64 },
    MaxIterations { iterations: usize, final_cost: f64 },
    Degenerate { reason: DegenerateReason },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DegenerateReason {
    TooFewPoses { count: usize },
    TooFewLandmarks { count: usize },
    NoFactors,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BaCorrection {
    pub pose_deltas: Vec<(KeyframeId, [f32; 6])>,
    pub landmark_deltas: Vec<(MapPointId, [f32; 3])>,
    pub result: BaResult,
}

#[derive(Debug)]
pub enum ObservationSetError {
    TooFew { required: usize, actual: usize },
}

impl std::fmt::Display for ObservationSetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ObservationSetError::TooFew { required, actual } => write!(
                f,
                "observation set requires at least {required} points, got {actual}"
            ),
        }
    }
}

impl std::error::Error for ObservationSetError {}

#[derive(Debug, Clone, Copy)]
pub struct MapObservation {
    keyframe_keypoint: KeyframeKeypoint,
    pixel: Keypoint,
}

impl MapObservation {
    pub fn new(keyframe_keypoint: KeyframeKeypoint, pixel: Keypoint) -> Self {
        Self {
            keyframe_keypoint,
            pixel,
        }
    }

    pub fn keyframe_keypoint(&self) -> KeyframeKeypoint {
        self.keyframe_keypoint
    }

    pub fn pixel(&self) -> Keypoint {
        self.pixel
    }
}

#[derive(Debug)]
pub struct ObservationSet {
    observations: Vec<MapObservation>,
}

impl ObservationSet {
    pub fn new(
        observations: Vec<MapObservation>,
        min_required: NonZeroUsize,
    ) -> Result<Self, ObservationSetError> {
        if observations.len() < min_required.get() {
            return Err(ObservationSetError::TooFew {
                required: min_required.get(),
                actual: observations.len(),
            });
        }
        Ok(Self { observations })
    }

    pub fn observations(&self) -> &[MapObservation] {
        &self.observations
    }

    fn resolve(
        &self,
        map: &SlamMap,
        intrinsics: PinholeIntrinsics,
        min_required: NonZeroUsize,
    ) -> Option<ResolvedObservationSet> {
        let mut resolved = Vec::with_capacity(self.observations.len());
        for obs in &self.observations {
            let keypoint_ref = obs.keyframe_keypoint();
            let point_id = map.map_point_for_keypoint(keypoint_ref).ok().flatten()?;
            let world = map.point(point_id)?.position();
            let observation = Observation::try_new(world, obs.pixel(), intrinsics).ok()?;
            resolved.push(observation);
        }
        if resolved.len() < min_required.get() {
            return None;
        }
        Some(ResolvedObservationSet {
            observations: resolved,
        })
    }
}

#[derive(Debug)]
struct ResolvedObservationSet {
    observations: Vec<Observation>,
}

impl ResolvedObservationSet {
    fn observations(&self) -> &[Observation] {
        &self.observations
    }
}

#[derive(Debug)]
struct BaFrame {
    pose: Pose,
    observations: ObservationSet,
}

#[derive(Debug)]
enum FullBaBuildError {
    EmptyWindow,
    DuplicateKeyframe {
        keyframe_id: KeyframeId,
    },
    MissingKeyframe {
        keyframe_id: KeyframeId,
    },
    TooFewKeyframes {
        required: usize,
        actual: usize,
    },
    DuplicateLandmarkObservation {
        point_id: MapPointId,
        keyframe_id: KeyframeId,
    },
    NoLandmarks,
    PoseHasTooFewObservations {
        keyframe_id: KeyframeId,
        required: usize,
        actual: usize,
    },
}

impl std::fmt::Display for FullBaBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FullBaBuildError::EmptyWindow => write!(f, "full BA window is empty"),
            FullBaBuildError::DuplicateKeyframe { keyframe_id } => {
                write!(f, "full BA window has duplicate keyframe {keyframe_id:?}")
            }
            FullBaBuildError::MissingKeyframe { keyframe_id } => {
                write!(
                    f,
                    "full BA window references missing keyframe {keyframe_id:?}"
                )
            }
            FullBaBuildError::TooFewKeyframes { required, actual } => {
                write!(
                    f,
                    "full BA requires at least {required} keyframes, got {actual}"
                )
            }
            FullBaBuildError::DuplicateLandmarkObservation {
                point_id,
                keyframe_id,
            } => write!(
                f,
                "landmark {point_id:?} has duplicate observation in keyframe {keyframe_id:?}"
            ),
            FullBaBuildError::NoLandmarks => {
                write!(f, "full BA window has no optimizable landmarks")
            }
            FullBaBuildError::PoseHasTooFewObservations {
                keyframe_id,
                required,
                actual,
            } => write!(
                f,
                "keyframe {keyframe_id:?} has too few BA observations: required={required}, actual={actual}"
            ),
        }
    }
}

impl std::error::Error for FullBaBuildError {}

fn degenerate_reason_from_build_error(err: &FullBaBuildError) -> DegenerateReason {
    match err {
        FullBaBuildError::EmptyWindow => DegenerateReason::TooFewPoses { count: 0 },
        FullBaBuildError::TooFewKeyframes { actual, .. } => {
            DegenerateReason::TooFewPoses { count: *actual }
        }
        FullBaBuildError::NoLandmarks => DegenerateReason::TooFewLandmarks { count: 0 },
        FullBaBuildError::DuplicateKeyframe { .. }
        | FullBaBuildError::MissingKeyframe { .. }
        | FullBaBuildError::DuplicateLandmarkObservation { .. }
        | FullBaBuildError::PoseHasTooFewObservations { .. } => DegenerateReason::NoFactors,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PoseVarIndex(usize);

impl PoseVarIndex {
    fn as_usize(self) -> usize {
        self.0
    }

    fn offset6(self) -> usize {
        self.0 * 6
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct LandmarkVarIndex(usize);

impl LandmarkVarIndex {
    fn as_usize(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
struct PoseVariable {
    keyframe_id: KeyframeId,
    pose: Pose,
}

#[derive(Clone, Copy, Debug)]
struct LandmarkVariable {
    point_id: MapPointId,
    position: Point3,
}

#[derive(Clone, Copy, Debug)]
struct ReprojectionFactor {
    pose: PoseVarIndex,
    landmark: LandmarkVarIndex,
    pixel: Keypoint,
}

#[derive(Debug)]
struct FullBaProblem {
    poses: Vec<PoseVariable>,
    landmarks: Vec<LandmarkVariable>,
    factors: Vec<ReprojectionFactor>,
}

impl FullBaProblem {
    fn try_from_map(
        map: &SlamMap,
        requested_window: &[KeyframeId],
        max_window: NonZeroUsize,
        min_observations: NonZeroUsize,
    ) -> Result<Self, FullBaBuildError> {
        if requested_window.is_empty() {
            return Err(FullBaBuildError::EmptyWindow);
        }

        let mut poses = Vec::new();
        let mut seen_keyframes = HashSet::new();
        for &keyframe_id in requested_window.iter().take(max_window.get()) {
            if !seen_keyframes.insert(keyframe_id) {
                return Err(FullBaBuildError::DuplicateKeyframe { keyframe_id });
            }
            let entry = map
                .keyframe(keyframe_id)
                .ok_or(FullBaBuildError::MissingKeyframe { keyframe_id })?;
            poses.push(PoseVariable {
                keyframe_id,
                pose: entry.pose(),
            });
        }

        if poses.len() < 2 {
            return Err(FullBaBuildError::TooFewKeyframes {
                required: 2,
                actual: poses.len(),
            });
        }

        let mut pose_lookup = HashMap::new();
        for (idx, pose) in poses.iter().enumerate() {
            pose_lookup.insert(pose.keyframe_id, PoseVarIndex(idx));
        }

        let mut landmarks = Vec::new();
        let mut factors = Vec::new();
        let mut pose_counts = vec![0_usize; poses.len()];

        for (point_id, point) in map.points() {
            let mut local_observations = Vec::new();
            let mut seen_local_poses = HashSet::new();

            for &obs in point.observations() {
                let Some(&pose_idx) = pose_lookup.get(&obs.keyframe_id()) else {
                    continue;
                };
                if !seen_local_poses.insert(pose_idx) {
                    return Err(FullBaBuildError::DuplicateLandmarkObservation {
                        point_id,
                        keyframe_id: obs.keyframe_id(),
                    });
                }
                let Ok(pixel) = map.keypoint(obs) else {
                    continue;
                };
                local_observations.push((pose_idx, pixel));
            }

            if local_observations.len() < 2 {
                continue;
            }

            let landmark_idx = LandmarkVarIndex(landmarks.len());
            landmarks.push(LandmarkVariable {
                point_id,
                position: point.position(),
            });

            for (pose_idx, pixel) in local_observations {
                pose_counts[pose_idx.as_usize()] += 1;
                factors.push(ReprojectionFactor {
                    pose: pose_idx,
                    landmark: landmark_idx,
                    pixel,
                });
            }
        }

        if landmarks.is_empty() {
            return Err(FullBaBuildError::NoLandmarks);
        }

        for (idx, pose) in poses.iter().enumerate() {
            if pose_counts[idx] < min_observations.get() {
                return Err(FullBaBuildError::PoseHasTooFewObservations {
                    keyframe_id: pose.keyframe_id,
                    required: min_observations.get(),
                    actual: pose_counts[idx],
                });
            }
        }

        Ok(Self {
            poses,
            landmarks,
            factors,
        })
    }

    fn write_back(self, map: &mut SlamMap) -> bool {
        for pose in &self.poses {
            if map.set_keyframe_pose(pose.keyframe_id, pose.pose).is_err() {
                return false;
            }
        }
        for landmark in &self.landmarks {
            if map
                .set_map_point_position(landmark.point_id, landmark.position)
                .is_err()
            {
                return false;
            }
        }
        true
    }
}

#[derive(Clone, Copy, Debug)]
struct PoseLandmarkCross {
    pose: PoseVarIndex,
    b: [[f32; 3]; 6],
}

#[derive(Debug, Default)]
struct LandmarkAccumulator {
    c: [[f32; 3]; 3],
    b: [f32; 3],
    links: Vec<PoseLandmarkCross>,
}

impl LandmarkAccumulator {
    fn add_link(&mut self, pose: PoseVarIndex, cross: [[f32; 3]; 6]) {
        if let Some(existing) = self.links.iter_mut().find(|link| link.pose == pose) {
            for row in 0..6 {
                for col in 0..3 {
                    existing.b[row][col] += cross[row][col];
                }
            }
            return;
        }
        self.links.push(PoseLandmarkCross { pose, b: cross });
    }
}

#[derive(Debug)]
struct LandmarkSchur {
    inv_c: [[f32; 3]; 3],
    b: [f32; 3],
    links: Vec<PoseLandmarkCross>,
}

#[derive(Debug)]
pub struct LocalBundleAdjuster {
    config: LocalBaConfig,
    intrinsics: PinholeIntrinsics,
    frames: Vec<BaFrame>,
    a_buf: Vec<f32>,
    b_buf: Vec<f32>,
}

impl LocalBundleAdjuster {
    pub fn new(intrinsics: PinholeIntrinsics, config: LocalBaConfig) -> Self {
        let dim = config.window().saturating_mul(6);
        let a_buf = vec![0.0_f32; dim * dim];
        let b_buf = vec![0.0_f32; dim];
        Self {
            config,
            intrinsics,
            frames: Vec::new(),
            a_buf,
            b_buf,
        }
    }

    pub fn reset(&mut self) {
        self.frames.clear();
    }

    pub fn min_observations(&self) -> NonZeroUsize {
        self.config.min_observations
    }

    pub fn window_size(&self) -> NonZeroUsize {
        self.config.window
    }

    pub fn push_frame(
        &mut self,
        map: &SlamMap,
        pose: Pose,
        observations: ObservationSet,
    ) -> Option<Pose> {
        self.frames.push(BaFrame { pose, observations });
        if self.frames.len() > self.config.window() {
            let excess = self.frames.len() - self.config.window();
            self.frames.drain(0..excess);
        }

        if !self.optimize(map) {
            return None;
        }
        self.frames.last().map(|frame| frame.pose)
    }

    pub fn optimize_keyframe_window(&mut self, map: &mut SlamMap, window: &[KeyframeId]) -> BaResult {
        let mut problem = match FullBaProblem::try_from_map(
            map,
            window,
            self.config.window,
            self.config.min_observations,
        ) {
            Ok(problem) => problem,
            Err(err) => {
                return BaResult::Degenerate {
                    reason: degenerate_reason_from_build_error(&err),
                };
            }
        };

        let result = self.optimize_full(&mut problem);
        if matches!(
            result,
            BaResult::Converged { .. } | BaResult::MaxIterations { .. }
        ) && !problem.write_back(map)
        {
            return BaResult::Degenerate {
                reason: DegenerateReason::NoFactors,
            };
        }

        result
    }

    fn optimize(&mut self, map: &SlamMap) -> bool {
        let frame_count = self.frames.len();
        if frame_count == 0 {
            return false;
        }

        let dim = frame_count * 6;
        let max_iters = self.config.max_iterations();
        let huber = self.config.huber_delta_px();
        let damping = self.config.damping();
        let motion_weight = self.config.motion_prior_weight();

        for _ in 0..max_iters {
            let a = &mut self.a_buf[..dim * dim];
            let b = &mut self.b_buf[..dim];
            a.fill(0.0);
            b.fill(0.0);

            for (idx, frame) in self.frames.iter().enumerate() {
                let base = idx * 6;
                let resolved = match frame.observations.resolve(
                    map,
                    self.intrinsics,
                    self.config.min_observations,
                ) {
                    Some(set) => set,
                    None => return false,
                };
                for obs in resolved.observations() {
                    if let Some((residual, jac)) =
                        reprojection_residual_and_jacobian(frame.pose, obs, self.intrinsics)
                    {
                        let r_norm = (residual[0] * residual[0] + residual[1] * residual[1]).sqrt();
                        let weight = if r_norm <= huber { 1.0 } else { huber / r_norm };
                        let scale = weight.sqrt();
                        let r0 = residual[0] * scale;
                        let r1 = residual[1] * scale;
                        let mut j = [[0.0_f32; 6]; 2];
                        for c in 0..6 {
                            j[0][c] = jac[0][c] * scale;
                            j[1][c] = jac[1][c] * scale;
                        }

                        for c in 0..6 {
                            let jr = j[0][c] * r0 + j[1][c] * r1;
                            b[base + c] -= jr;
                            for d in 0..6 {
                                let jt_j = j[0][c] * j[0][d] + j[1][c] * j[1][d];
                                a[(base + c) * dim + (base + d)] += jt_j;
                            }
                        }
                    }
                }
            }

            if motion_weight > 0.0 && frame_count >= 2 {
                let weight = motion_weight;
                for i in 1..frame_count {
                    let prev = &self.frames[i - 1].pose;
                    let curr = &self.frames[i].pose;
                    let r_prev = pose_to_vec(*prev);
                    let r_curr = pose_to_vec(*curr);
                    let mut residual = [0.0_f32; 6];
                    for k in 0..6 {
                        residual[k] = r_curr[k] - r_prev[k];
                    }
                    let base_prev = (i - 1) * 6;
                    let base_curr = i * 6;

                    for k in 0..6 {
                        let r = residual[k] * weight;
                        b[base_prev + k] += r;
                        b[base_curr + k] -= r;

                        let w = weight * weight;
                        a[(base_prev + k) * dim + (base_prev + k)] += w;
                        a[(base_curr + k) * dim + (base_curr + k)] += w;
                        a[(base_prev + k) * dim + (base_curr + k)] -= w;
                        a[(base_curr + k) * dim + (base_prev + k)] -= w;
                    }
                }
            }

            for i in 0..dim {
                a[i * dim + i] += damping;
            }

            if !solve_linear_system(a, b, dim) {
                return false;
            }

            let mut max_step = 0.0_f32;
            for i in 0..frame_count {
                let base = i * 6;
                let step = [
                    b[base],
                    b[base + 1],
                    b[base + 2],
                    b[base + 3],
                    b[base + 4],
                    b[base + 5],
                ];
                let step_norm = (step.iter().map(|v| v * v).sum::<f32>()).sqrt();
                if step_norm > max_step {
                    max_step = step_norm;
                }
                let pose = self.frames[i].pose;
                self.frames[i].pose = apply_se3_delta(pose, step);
            }

            if max_step < 1e-4 {
                break;
            }
        }

        true
    }

    fn optimize_full(&mut self, problem: &mut FullBaProblem) -> BaResult {
        let pose_count = problem.poses.len();
        let landmark_count = problem.landmarks.len();
        if pose_count < 2 {
            return BaResult::Degenerate {
                reason: DegenerateReason::TooFewPoses { count: pose_count },
            };
        }
        if landmark_count == 0 {
            return BaResult::Degenerate {
                reason: DegenerateReason::TooFewLandmarks {
                    count: landmark_count,
                },
            };
        }
        if problem.factors.is_empty() {
            return BaResult::Degenerate {
                reason: DegenerateReason::NoFactors,
            };
        }

        let pose_dim = pose_count * 6;
        let max_iters = self.config.max_iterations();
        let huber = self.config.huber_delta_px();
        let pose_damping = self.config.damping().max(1e-9);
        let landmark_damping = self.config.damping().max(1e-6);
        let motion_weight = self.config.motion_prior_weight();
        let mut final_cost = full_problem_cost(problem, self.intrinsics, huber, motion_weight);

        for iter in 0..max_iters {
            let s = &mut self.a_buf[..pose_dim * pose_dim];
            let rhs = &mut self.b_buf[..pose_dim];
            s.fill(0.0);
            rhs.fill(0.0);

            let mut landmark_accumulators = (0..landmark_count)
                .map(|_| LandmarkAccumulator::default())
                .collect::<Vec<_>>();

            for factor in &problem.factors {
                let pose_idx = factor.pose;
                let landmark_idx = factor.landmark;
                let pose = problem.poses[pose_idx.as_usize()].pose;
                let point = problem.landmarks[landmark_idx.as_usize()].position;

                let Some((residual, j_pose, j_landmark)) =
                    reprojection_residual_and_jacobians(pose, point, factor.pixel, self.intrinsics)
                else {
                    continue;
                };

                let r_norm = (residual[0] * residual[0] + residual[1] * residual[1]).sqrt();
                let weight = if r_norm <= huber { 1.0 } else { huber / r_norm };
                let scale = weight.sqrt();

                let r_scaled = [residual[0] * scale, residual[1] * scale];
                let j_pose_scaled = [
                    [
                        j_pose[0][0] * scale,
                        j_pose[0][1] * scale,
                        j_pose[0][2] * scale,
                        j_pose[0][3] * scale,
                        j_pose[0][4] * scale,
                        j_pose[0][5] * scale,
                    ],
                    [
                        j_pose[1][0] * scale,
                        j_pose[1][1] * scale,
                        j_pose[1][2] * scale,
                        j_pose[1][3] * scale,
                        j_pose[1][4] * scale,
                        j_pose[1][5] * scale,
                    ],
                ];
                let j_landmark_scaled = [
                    [
                        j_landmark[0][0] * scale,
                        j_landmark[0][1] * scale,
                        j_landmark[0][2] * scale,
                    ],
                    [
                        j_landmark[1][0] * scale,
                        j_landmark[1][1] * scale,
                        j_landmark[1][2] * scale,
                    ],
                ];

                accumulate_pose_hessian(s, pose_dim, pose_idx, j_pose_scaled);
                accumulate_pose_rhs(rhs, pose_idx, j_pose_scaled, r_scaled);

                let acc = &mut landmark_accumulators[landmark_idx.as_usize()];
                accumulate_landmark_hessian(&mut acc.c, j_landmark_scaled);
                accumulate_landmark_rhs(&mut acc.b, j_landmark_scaled, r_scaled);
                acc.add_link(
                    pose_idx,
                    pose_landmark_cross(j_pose_scaled, j_landmark_scaled),
                );
            }

            if motion_weight > 0.0 && pose_count >= 2 {
                for i in 1..pose_count {
                    let prev = &problem.poses[i - 1].pose;
                    let curr = &problem.poses[i].pose;
                    let r_prev = pose_to_vec(*prev);
                    let r_curr = pose_to_vec(*curr);
                    let mut residual = [0.0_f32; 6];
                    for k in 0..6 {
                        residual[k] = r_curr[k] - r_prev[k];
                    }
                    let base_prev = (i - 1) * 6;
                    let base_curr = i * 6;

                    for k in 0..6 {
                        let r = residual[k] * motion_weight;
                        rhs[base_prev + k] += r;
                        rhs[base_curr + k] -= r;

                        let w = motion_weight * motion_weight;
                        s[(base_prev + k) * pose_dim + (base_prev + k)] += w;
                        s[(base_curr + k) * pose_dim + (base_curr + k)] += w;
                        s[(base_prev + k) * pose_dim + (base_curr + k)] -= w;
                        s[(base_curr + k) * pose_dim + (base_prev + k)] -= w;
                    }
                }
            }

            for i in 0..pose_dim {
                s[i * pose_dim + i] += pose_damping;
            }

            let mut schur_landmarks = Vec::with_capacity(landmark_count);
            for acc in landmark_accumulators.into_iter() {
                let mut c = acc.c;
                for i in 0..3 {
                    c[i][i] += landmark_damping;
                }
                let Some(inv_c) = invert_3x3(c) else {
                    return BaResult::MaxIterations {
                        iterations: iter + 1,
                        final_cost,
                    };
                };

                let inv_c_b = mat3_mul_vec3(inv_c, acc.b);
                for link_i in &acc.links {
                    let base_i = link_i.pose.offset6();
                    let rhs_contrib = mat63_mul_vec3(link_i.b, inv_c_b);
                    for row in 0..6 {
                        rhs[base_i + row] -= rhs_contrib[row];
                    }

                    for link_j in &acc.links {
                        let base_j = link_j.pose.offset6();
                        let block = schur_block(link_i.b, inv_c, link_j.b);
                        for row in 0..6 {
                            for col in 0..6 {
                                s[(base_i + row) * pose_dim + (base_j + col)] -= block[row][col];
                            }
                        }
                    }
                }

                schur_landmarks.push(LandmarkSchur {
                    inv_c,
                    b: acc.b,
                    links: acc.links,
                });
            }

            fix_pose_block(s, rhs, pose_dim, PoseVarIndex(0));

            if !solve_linear_system(s, rhs, pose_dim) {
                return BaResult::MaxIterations {
                    iterations: iter + 1,
                    final_cost,
                };
            }

            let mut max_step = 0.0_f32;
            for (pose_i, pose_var) in problem.poses.iter_mut().enumerate() {
                let base = pose_i * 6;
                let delta = [
                    rhs[base],
                    rhs[base + 1],
                    rhs[base + 2],
                    rhs[base + 3],
                    rhs[base + 4],
                    rhs[base + 5],
                ];
                max_step = max_step.max(norm6(delta));
                pose_var.pose = apply_se3_delta(pose_var.pose, delta);
            }

            for (landmark_i, landmark_var) in problem.landmarks.iter_mut().enumerate() {
                let schur = &schur_landmarks[landmark_i];
                let mut coupling = [0.0_f32; 3];
                for link in &schur.links {
                    let base = link.pose.offset6();
                    let pose_delta = [
                        rhs[base],
                        rhs[base + 1],
                        rhs[base + 2],
                        rhs[base + 3],
                        rhs[base + 4],
                        rhs[base + 5],
                    ];
                    for col in 0..3 {
                        for row in 0..6 {
                            coupling[col] += link.b[row][col] * pose_delta[row];
                        }
                    }
                }

                let rhs_landmark = [
                    schur.b[0] - coupling[0],
                    schur.b[1] - coupling[1],
                    schur.b[2] - coupling[2],
                ];
                let delta_landmark = mat3_mul_vec3(schur.inv_c, rhs_landmark);
                max_step = max_step.max(norm3(delta_landmark));

                landmark_var.position.x += delta_landmark[0];
                landmark_var.position.y += delta_landmark[1];
                landmark_var.position.z += delta_landmark[2];
            }

            final_cost = full_problem_cost(problem, self.intrinsics, huber, motion_weight);
            if max_step < 1e-4 {
                return BaResult::Converged {
                    iterations: iter + 1,
                    final_cost,
                };
            }
        }

        BaResult::MaxIterations {
            iterations: max_iters,
            final_cost,
        }
    }
}

fn full_problem_cost(
    problem: &FullBaProblem,
    intrinsics: PinholeIntrinsics,
    huber_delta_px: f32,
    motion_weight: f32,
) -> f64 {
    let mut cost = 0.0_f64;
    let huber = huber_delta_px as f64;

    for factor in &problem.factors {
        let pose = problem.poses[factor.pose.as_usize()].pose;
        let point = problem.landmarks[factor.landmark.as_usize()].position;
        let Some((residual, _, _)) =
            reprojection_residual_and_jacobians(pose, point, factor.pixel, intrinsics)
        else {
            continue;
        };
        let r0 = residual[0] as f64;
        let r1 = residual[1] as f64;
        let r_norm = (r0 * r0 + r1 * r1).sqrt();
        cost += if r_norm <= huber {
            0.5 * r_norm * r_norm
        } else {
            huber * (r_norm - 0.5 * huber)
        };
    }

    if motion_weight > 0.0 {
        let w2 = (motion_weight as f64) * (motion_weight as f64);
        for i in 1..problem.poses.len() {
            let prev = pose_to_vec(problem.poses[i - 1].pose);
            let curr = pose_to_vec(problem.poses[i].pose);
            for k in 0..6 {
                let d = (curr[k] - prev[k]) as f64;
                cost += 0.5 * w2 * d * d;
            }
        }
    }

    cost
}

fn reprojection_residual_and_jacobian(
    pose: Pose,
    obs: &Observation,
    intrinsics: PinholeIntrinsics,
) -> Option<([f32; 2], [[f32; 6]; 2])> {
    let (residual, jac_pose, _) =
        reprojection_residual_and_jacobians(pose, obs.world(), obs.pixel(), intrinsics)?;
    Some((residual, jac_pose))
}

fn reprojection_residual_and_jacobians(
    pose: Pose,
    world: Point3,
    pixel: Keypoint,
    intrinsics: PinholeIntrinsics,
) -> Option<([f32; 2], [[f32; 6]; 2], [[f32; 3]; 2])> {
    let pw = [world.x, world.y, world.z];
    let rotation = pose.rotation();
    let pc = math::transform_point(rotation, pose.translation(), pw);
    let x = pc[0];
    let y = pc[1];
    let z = pc[2];
    if z <= 1e-6 {
        return None;
    }

    let u = intrinsics.fx() * (x / z) + intrinsics.cx();
    let v = intrinsics.fy() * (y / z) + intrinsics.cy();
    let residual = [pixel.x - u, pixel.y - v];

    let inv_z = 1.0 / z;
    let inv_z2 = inv_z * inv_z;
    let du_dx = intrinsics.fx() * inv_z;
    let du_dy = 0.0;
    let du_dz = -intrinsics.fx() * x * inv_z2;
    let dv_dx = 0.0;
    let dv_dy = intrinsics.fy() * inv_z;
    let dv_dz = -intrinsics.fy() * y * inv_z2;

    let a1 = du_dx;
    let a2 = du_dy;
    let a3 = du_dz;
    let b1 = dv_dx;
    let b2 = dv_dy;
    let b3 = dv_dz;

    let mut jac_pose = [[0.0_f32; 6]; 2];

    jac_pose[0][0] = a1;
    jac_pose[0][1] = a2;
    jac_pose[0][2] = a3;
    jac_pose[1][0] = b1;
    jac_pose[1][1] = b2;
    jac_pose[1][2] = b3;

    jac_pose[0][3] = -(a2 * z - a3 * y);
    jac_pose[0][4] = a1 * z - a3 * x;
    jac_pose[0][5] = -a1 * y + a2 * x;

    jac_pose[1][3] = -(b2 * z - b3 * y);
    jac_pose[1][4] = b1 * z - b3 * x;
    jac_pose[1][5] = -b1 * y + b2 * x;

    let mut jac_landmark = [[0.0_f32; 3]; 2];
    for col in 0..3 {
        jac_landmark[0][col] =
            a1 * rotation[0][col] + a2 * rotation[1][col] + a3 * rotation[2][col];
        jac_landmark[1][col] =
            b1 * rotation[0][col] + b2 * rotation[1][col] + b3 * rotation[2][col];
    }

    // The Jacobian above is for projected pixel coordinates [u, v].
    // Residual is defined as [pixel.x - u, pixel.y - v], so dr/dx = -du/dx.
    for row in &mut jac_pose {
        for value in row {
            *value = -*value;
        }
    }
    for row in &mut jac_landmark {
        for value in row {
            *value = -*value;
        }
    }

    Some((residual, jac_pose, jac_landmark))
}

pub(crate) fn apply_se3_delta(pose: Pose, delta: [f32; 6]) -> Pose {
    let v = [delta[0], delta[1], delta[2]];
    let w = [delta[3], delta[4], delta[5]];
    let r_delta = so3_exp(w);
    let r = math::mat_mul(r_delta, pose.rotation());
    let t = math::mat_mul_vec(r_delta, pose.translation());
    Pose::from_rt(r, [t[0] + v[0], t[1] + v[1], t[2] + v[2]])
}

pub(crate) fn se3_delta_between(from: Pose, to: Pose) -> [f32; 6] {
    let from_rot = from.rotation();
    let mut from_rot_t = [[0.0_f32; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            from_rot_t[row][col] = from_rot[col][row];
        }
    }

    let r_delta = math::mat_mul(to.rotation(), from_rot_t);
    let w = so3_log(r_delta);
    let rotated_from_t = math::mat_mul_vec(r_delta, from.translation());
    let to_t = to.translation();
    let v = [
        to_t[0] - rotated_from_t[0],
        to_t[1] - rotated_from_t[1],
        to_t[2] - rotated_from_t[2],
    ];
    [v[0], v[1], v[2], w[0], w[1], w[2]]
}

fn pose_to_vec(pose: Pose) -> [f32; 6] {
    let t = pose.translation();
    let w = so3_log(pose.rotation());
    [t[0], t[1], t[2], w[0], w[1], w[2]]
}

fn so3_exp(w: [f32; 3]) -> [[f32; 3]; 3] {
    let theta = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
    let mut r = [[0.0_f32; 3]; 3];
    if theta < 1e-6 {
        r[0][0] = 1.0;
        r[1][1] = 1.0;
        r[2][2] = 1.0;
        r[0][1] = -w[2];
        r[0][2] = w[1];
        r[1][0] = w[2];
        r[1][2] = -w[0];
        r[2][0] = -w[1];
        r[2][1] = w[0];
        return r;
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

    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = if i == j { 1.0 } else { 0.0 } + sin_t * kx[i][j] + (1.0 - cos_t) * kx2[i][j];
        }
    }
    r
}

fn so3_log(r: [[f32; 3]; 3]) -> [f32; 3] {
    let trace = r[0][0] + r[1][1] + r[2][2];
    let mut cos_theta = (trace - 1.0) * 0.5;
    if cos_theta > 1.0 {
        cos_theta = 1.0;
    } else if cos_theta < -1.0 {
        cos_theta = -1.0;
    }
    let theta = cos_theta.acos();
    if theta < 1e-6 {
        return [
            0.5 * (r[2][1] - r[1][2]),
            0.5 * (r[0][2] - r[2][0]),
            0.5 * (r[1][0] - r[0][1]),
        ];
    }

    // Near pi, theta/sin(theta) becomes numerically unstable. Recover the
    // axis from the diagonal of R (equivalently from R + I) and align the
    // sign with the skew-symmetric part.
    if std::f32::consts::PI - theta < 1e-3 {
        let xx = ((r[0][0] + 1.0) * 0.5).max(0.0).sqrt();
        let yy = ((r[1][1] + 1.0) * 0.5).max(0.0).sqrt();
        let zz = ((r[2][2] + 1.0) * 0.5).max(0.0).sqrt();

        let mut axis = if xx >= yy && xx >= zz && xx > 1e-6 {
            [
                xx,
                (r[0][1] + r[1][0]) / (4.0 * xx),
                (r[0][2] + r[2][0]) / (4.0 * xx),
            ]
        } else if yy >= zz && yy > 1e-6 {
            [
                (r[0][1] + r[1][0]) / (4.0 * yy),
                yy,
                (r[1][2] + r[2][1]) / (4.0 * yy),
            ]
        } else if zz > 1e-6 {
            [
                (r[0][2] + r[2][0]) / (4.0 * zz),
                (r[1][2] + r[2][1]) / (4.0 * zz),
                zz,
            ]
        } else {
            [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]]
        };

        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if norm > 1e-8 {
            axis = [axis[0] / norm, axis[1] / norm, axis[2] / norm];
        } else {
            axis = [1.0, 0.0, 0.0];
        }

        let skew = [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]];
        let sign = axis[0] * skew[0] + axis[1] * skew[1] + axis[2] * skew[2];
        if sign < 0.0 {
            axis = [-axis[0], -axis[1], -axis[2]];
        }
        return [axis[0] * theta, axis[1] * theta, axis[2] * theta];
    }

    let sin_theta = theta.sin();
    if sin_theta.abs() < 1e-6 {
        return [
            0.5 * (r[2][1] - r[1][2]),
            0.5 * (r[0][2] - r[2][0]),
            0.5 * (r[1][0] - r[0][1]),
        ];
    }
    let factor = theta / (2.0 * sin_theta);
    [
        factor * (r[2][1] - r[1][2]),
        factor * (r[0][2] - r[2][0]),
        factor * (r[1][0] - r[0][1]),
    ]
}

fn accumulate_pose_hessian(
    hessian: &mut [f32],
    pose_dim: usize,
    pose_idx: PoseVarIndex,
    j_pose: [[f32; 6]; 2],
) {
    let base = pose_idx.offset6();
    for row in 0..6 {
        for col in 0..6 {
            let jt_j = j_pose[0][row] * j_pose[0][col] + j_pose[1][row] * j_pose[1][col];
            hessian[(base + row) * pose_dim + (base + col)] += jt_j;
        }
    }
}

fn accumulate_pose_rhs(
    rhs: &mut [f32],
    pose_idx: PoseVarIndex,
    j_pose: [[f32; 6]; 2],
    residual: [f32; 2],
) {
    let base = pose_idx.offset6();
    for col in 0..6 {
        rhs[base + col] -= j_pose[0][col] * residual[0] + j_pose[1][col] * residual[1];
    }
}

fn accumulate_landmark_hessian(c: &mut [[f32; 3]; 3], j_landmark: [[f32; 3]; 2]) {
    for row in 0..3 {
        for col in 0..3 {
            c[row][col] +=
                j_landmark[0][row] * j_landmark[0][col] + j_landmark[1][row] * j_landmark[1][col];
        }
    }
}

fn accumulate_landmark_rhs(b: &mut [f32; 3], j_landmark: [[f32; 3]; 2], residual: [f32; 2]) {
    for col in 0..3 {
        b[col] -= j_landmark[0][col] * residual[0] + j_landmark[1][col] * residual[1];
    }
}

fn pose_landmark_cross(j_pose: [[f32; 6]; 2], j_landmark: [[f32; 3]; 2]) -> [[f32; 3]; 6] {
    let mut cross = [[0.0_f32; 3]; 6];
    for row in 0..6 {
        for col in 0..3 {
            cross[row][col] =
                j_pose[0][row] * j_landmark[0][col] + j_pose[1][row] * j_landmark[1][col];
        }
    }
    cross
}

fn mat3_mul_vec3(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn mat63_mul_vec3(m: [[f32; 3]; 6], v: [f32; 3]) -> [f32; 6] {
    let mut out = [0.0_f32; 6];
    for row in 0..6 {
        out[row] = m[row][0] * v[0] + m[row][1] * v[1] + m[row][2] * v[2];
    }
    out
}

fn schur_block(b_i: [[f32; 3]; 6], inv_c: [[f32; 3]; 3], b_j: [[f32; 3]; 6]) -> [[f32; 6]; 6] {
    let mut block = [[0.0_f32; 6]; 6];
    for row in 0..6 {
        for col in 0..6 {
            let mut sum = 0.0_f32;
            for k in 0..3 {
                for l in 0..3 {
                    sum += b_i[row][k] * inv_c[k][l] * b_j[col][l];
                }
            }
            block[row][col] = sum;
        }
    }
    block
}

fn fix_pose_block(hessian: &mut [f32], rhs: &mut [f32], pose_dim: usize, pose_idx: PoseVarIndex) {
    let base = pose_idx.offset6();
    for row in 0..6 {
        let idx = base + row;
        for col in 0..pose_dim {
            hessian[idx * pose_dim + col] = 0.0;
            hessian[col * pose_dim + idx] = 0.0;
        }
        hessian[idx * pose_dim + idx] = 1.0;
        rhs[idx] = 0.0;
    }
}

fn invert_3x3(m: [[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let a = m[0][0] as f64;
    let b = m[0][1] as f64;
    let c = m[0][2] as f64;
    let d = m[1][0] as f64;
    let e = m[1][1] as f64;
    let f = m[1][2] as f64;
    let g = m[2][0] as f64;
    let h = m[2][1] as f64;
    let i = m[2][2] as f64;

    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if !det.is_finite() || det.abs() < 1e-18 {
        return None;
    }
    let inv_det = 1.0 / det;
    let inv = [
        [
            (e * i - f * h) * inv_det,
            (c * h - b * i) * inv_det,
            (b * f - c * e) * inv_det,
        ],
        [
            (f * g - d * i) * inv_det,
            (a * i - c * g) * inv_det,
            (c * d - a * f) * inv_det,
        ],
        [
            (d * h - e * g) * inv_det,
            (b * g - a * h) * inv_det,
            (a * e - b * d) * inv_det,
        ],
    ];
    if inv
        .iter()
        .flat_map(|row| row.iter())
        .any(|value| !value.is_finite())
    {
        return None;
    }
    Some([
        [inv[0][0] as f32, inv[0][1] as f32, inv[0][2] as f32],
        [inv[1][0] as f32, inv[1][1] as f32, inv[1][2] as f32],
        [inv[2][0] as f32, inv[2][1] as f32, inv[2][2] as f32],
    ])
}

fn norm3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn norm6(v: [f32; 6]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3] + v[4] * v[4] + v[5] * v[5]).sqrt()
}

fn solve_linear_system(a: &mut [f32], b: &mut [f32], n: usize) -> bool {
    for i in 0..n {
        let mut max_row = i;
        let mut max_val = a[i * n + i].abs();
        for r in (i + 1)..n {
            let val = a[r * n + i].abs();
            if val > max_val {
                max_val = val;
                max_row = r;
            }
        }

        if max_val < 1e-9 {
            return false;
        }

        if max_row != i {
            for c in i..n {
                a.swap(i * n + c, max_row * n + c);
            }
            b.swap(i, max_row);
        }

        let diag = a[i * n + i];
        for c in i..n {
            a[i * n + c] /= diag;
        }
        b[i] /= diag;

        for r in 0..n {
            if r == i {
                continue;
            }
            let factor = a[r * n + i];
            if factor.abs() < 1e-12 {
                continue;
            }
            for c in i..n {
                a[r * n + c] -= factor * a[i * n + c];
            }
            b[r] -= factor * b[i];
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::{assert_map_invariants, KeyframeId, MapPointId, SlamMap};
    use crate::test_helpers::{
        axis_angle_pose, make_detections, make_pinhole_intrinsics, project_world_point,
    };
    use crate::{FrameId, Keypoint, Point3, SensorId, Timestamp};

    fn l2_3(a: [f32; 3], b: [f32; 3]) -> f32 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    fn pose_close(a: Pose, b: Pose, tol: f32) -> bool {
        let mut rot_sq = 0.0_f32;
        let ra = a.rotation();
        let rb = b.rotation();
        for i in 0..3 {
            for j in 0..3 {
                let d = ra[i][j] - rb[i][j];
                rot_sq += d * d;
            }
        }
        rot_sq.sqrt() <= tol && l2_3(a.translation(), b.translation()) <= tol
    }

    fn projection_residual(
        pose: Pose,
        obs: &Observation,
        intrinsics: PinholeIntrinsics,
    ) -> [f32; 2] {
        reprojection_residual_and_jacobian(pose, obs, intrinsics)
            .expect("valid reprojection")
            .0
    }

    fn project_pixel(
        pose_world_to_camera: Pose,
        point_world: Point3,
        intr: PinholeIntrinsics,
    ) -> Keypoint {
        let pc = math::transform_point(
            pose_world_to_camera.rotation(),
            pose_world_to_camera.translation(),
            [point_world.x, point_world.y, point_world.z],
        );
        Keypoint {
            x: intr.fx() * (pc[0] / pc[2]) + intr.cx(),
            y: intr.fy() * (pc[1] / pc[2]) + intr.cy(),
        }
    }

    fn pose_distance(a: Pose, b: Pose) -> f32 {
        let mut rot_sq = 0.0_f32;
        let ra = a.rotation();
        let rb = b.rotation();
        for i in 0..3 {
            for j in 0..3 {
                let d = ra[i][j] - rb[i][j];
                rot_sq += d * d;
            }
        }
        rot_sq.sqrt() + l2_3(a.translation(), b.translation())
    }

    fn mean_landmark_error(map: &SlamMap, keyframe_id: KeyframeId, expected: &[Point3]) -> f32 {
        let mut sum = 0.0_f32;
        for (idx, target) in expected.iter().enumerate() {
            let kp = map
                .keyframe_keypoint(keyframe_id, idx)
                .expect("keypoint index in map");
            let point_id = map
                .map_point_for_keypoint(kp)
                .expect("keyframe lookup")
                .expect("point exists");
            let point = map.point(point_id).expect("point lookup").position();
            let dx = point.x - target.x;
            let dy = point.y - target.y;
            let dz = point.z - target.z;
            sum += (dx * dx + dy * dy + dz * dz).sqrt();
        }
        sum / expected.len() as f32
    }

    fn build_full_ba_fixture(
        noisy_pose_delta: [f32; 6],
    ) -> (
        SlamMap,
        PinholeIntrinsics,
        KeyframeId,
        KeyframeId,
        Pose,
        Vec<Point3>,
    ) {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let true_pose_0 = Pose::identity();
        let true_pose_1 = axis_angle_pose([0.20, -0.02, 0.03], [0.0, 0.03, -0.01]);
        let noisy_pose_1 = apply_se3_delta(true_pose_1, noisy_pose_delta);

        let points_true = vec![
            Point3 {
                x: -0.35,
                y: -0.25,
                z: 3.2,
            },
            Point3 {
                x: -0.10,
                y: -0.22,
                z: 3.5,
            },
            Point3 {
                x: 0.14,
                y: -0.20,
                z: 3.8,
            },
            Point3 {
                x: 0.32,
                y: -0.10,
                z: 3.4,
            },
            Point3 {
                x: -0.30,
                y: 0.10,
                z: 3.6,
            },
            Point3 {
                x: -0.08,
                y: 0.16,
                z: 4.0,
            },
            Point3 {
                x: 0.16,
                y: 0.12,
                z: 3.3,
            },
            Point3 {
                x: 0.34,
                y: 0.24,
                z: 3.9,
            },
        ];

        let mut keypoints_0 = Vec::with_capacity(points_true.len());
        let mut keypoints_1 = Vec::with_capacity(points_true.len());
        for &point in &points_true {
            keypoints_0.push(
                project_world_point(true_pose_0, point, intrinsics)
                    .expect("point visible in pose 0"),
            );
            keypoints_1.push(
                project_world_point(true_pose_1, point, intrinsics)
                    .expect("point visible in pose 1"),
            );
        }

        let detections_0 = make_detections(
            SensorId::StereoLeft,
            FrameId::new(500),
            640,
            480,
            keypoints_0,
        )
        .expect("detections 0");
        let detections_1 = make_detections(
            SensorId::StereoLeft,
            FrameId::new(501),
            640,
            480,
            keypoints_1,
        )
        .expect("detections 1");

        let mut map = SlamMap::new();
        let kf_0 = map
            .add_keyframe_from_detections(
                detections_0.as_ref(),
                Timestamp::from_nanos(1_000_000),
                true_pose_0,
            )
            .expect("insert keyframe 0");
        let kf_1 = map
            .add_keyframe_from_detections(
                detections_1.as_ref(),
                Timestamp::from_nanos(2_000_000),
                noisy_pose_1,
            )
            .expect("insert keyframe 1");

        for (idx, &point_true) in points_true.iter().enumerate() {
            let kp_0 = map.keyframe_keypoint(kf_0, idx).expect("kf0 keypoint");
            let kp_1 = map.keyframe_keypoint(kf_1, idx).expect("kf1 keypoint");
            let descriptor = detections_0.descriptors()[idx];
            let i = idx as f32;
            let noisy_point = Point3 {
                x: point_true.x + (i - 3.5) * 0.010,
                y: point_true.y - (i - 3.5) * 0.008,
                z: point_true.z + ((idx % 2) as f32 - 0.5) * 0.040,
            };
            let point_id = map
                .add_map_point(noisy_point, descriptor.quantize(), kp_0)
                .expect("insert map point");
            map.add_observation(point_id, kp_1)
                .expect("add shared observation");
        }

        (map, intrinsics, kf_0, kf_1, true_pose_1, points_true)
    }

    #[test]
    fn local_ba_config_rejects_invalid_values() {
        assert!(matches!(
            LocalBaConfig::new(0, 10, 4, 1.0, 0.0, 0.0),
            Err(LocalBaConfigError::ZeroWindow)
        ));
        assert!(matches!(
            LocalBaConfig::new(5, 0, 4, 1.0, 0.0, 0.0),
            Err(LocalBaConfigError::ZeroIterations)
        ));
        assert!(matches!(
            LocalBaConfig::new(5, 10, 0, 1.0, 0.0, 0.0),
            Err(LocalBaConfigError::ZeroObservations)
        ));
        assert!(matches!(
            LocalBaConfig::new(5, 10, 3, 1.0, 0.0, 0.0),
            Err(LocalBaConfigError::TooFewObservations { .. })
        ));
        assert!(matches!(
            LocalBaConfig::new(5, 10, 4, 0.0, 0.0, 0.0),
            Err(LocalBaConfigError::NonPositiveHuber { .. })
        ));
        assert!(matches!(
            LocalBaConfig::new(5, 10, 4, 1.0, -1.0, 0.0),
            Err(LocalBaConfigError::NegativeDamping { .. })
        ));
        assert!(matches!(
            LocalBaConfig::new(5, 10, 4, 1.0, 0.0, -1.0),
            Err(LocalBaConfigError::NegativeMotionWeight { .. })
        ));
    }

    #[test]
    fn lm_config_rejects_invalid_values() {
        assert!(matches!(
            LmConfig::new(0.0, 10.0, 1e-8, 1e4, 0.25, 0.75),
            Err(LmConfigError::NonPositiveInitialLambda { .. })
        ));
        assert!(matches!(
            LmConfig::new(1e-4, 1.0, 1e-8, 1e4, 0.25, 0.75),
            Err(LmConfigError::LambdaFactorTooSmall { .. })
        ));
        assert!(matches!(
            LmConfig::new(1e-4, 10.0, 1e-2, 1e-3, 0.25, 0.75),
            Err(LmConfigError::MinLambdaExceedsMax { .. })
        ));
        assert!(matches!(
            LmConfig::new(1e-4, 10.0, 1e-8, 1e4, 0.8, 0.7),
            Err(LmConfigError::InvalidRhoOrdering { .. })
        ));
    }

    #[test]
    fn lm_state_good_rho_decreases_lambda() {
        let config = LmConfig::default();
        let mut state = LmState::new(config, 10.0);
        let action = state.step(8.0, 1.0, config);
        assert_eq!(action, LmAction::Accept);
        assert!(state.lambda() < config.initial_lambda());
        assert!((state.prev_cost() - 8.0).abs() < 1e-9);
    }

    #[test]
    fn lm_state_bad_rho_rejects_and_increases_lambda() {
        let config = LmConfig::default();
        let mut state = LmState::new(config, 10.0);
        let action = state.step(9.9, 10.0, config);
        assert_eq!(action, LmAction::Reject);
        assert!(state.lambda() > config.initial_lambda());
        assert!((state.prev_cost() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn observation_set_rejects_too_few_points() {
        let min_required = NonZeroUsize::new(4).expect("nonzero");
        let err = ObservationSet::new(Vec::new(), min_required).expect_err("must reject");
        match err {
            ObservationSetError::TooFew { required, actual } => {
                assert_eq!(required, 4);
                assert_eq!(actual, 0);
            }
        }
    }

    #[test]
    fn so3_exp_log_round_trip_for_small_rotation() {
        let w = [0.18, -0.06, 0.11];
        let r = so3_exp(w);
        let recovered = so3_log(r);
        assert!(
            l2_3(w, recovered) < 2e-4,
            "round-trip mismatch: {recovered:?}"
        );
    }

    #[test]
    fn so3_log_is_finite_near_pi() {
        let theta = std::f32::consts::PI - 1e-4;
        let w = [0.0, theta, 0.0];
        let r = so3_exp(w);
        let recovered = so3_log(r);
        assert!(recovered.iter().all(|v| v.is_finite()));

        let recovered_norm = (recovered[0] * recovered[0]
            + recovered[1] * recovered[1]
            + recovered[2] * recovered[2])
            .sqrt();
        assert!(
            (recovered_norm - theta).abs() < 3e-3,
            "theta mismatch: recovered={recovered_norm}, expected={theta}"
        );
    }

    #[test]
    fn apply_se3_delta_zero_is_fixpoint() {
        let pose = axis_angle_pose([0.3, -0.4, 0.5], [0.08, -0.05, 0.03]);
        let out = apply_se3_delta(pose, [0.0; 6]);
        assert!(pose_close(pose, out, 1e-7));
    }

    #[test]
    fn solve_linear_system_solves_identity_system() {
        let mut a = vec![1.0_f32, 0.0, 0.0, 1.0];
        let mut b = vec![2.5_f32, -3.0];
        assert!(solve_linear_system(&mut a, &mut b, 2));
        assert!((b[0] - 2.5).abs() < 1e-6);
        assert!((b[1] + 3.0).abs() < 1e-6);
    }

    #[test]
    fn solve_linear_system_reports_singular_matrix() {
        let mut a = vec![1.0_f32, 2.0, 2.0, 4.0];
        let mut b = vec![1.0_f32, 2.0];
        assert!(!solve_linear_system(&mut a, &mut b, 2));
    }

    #[test]
    fn optimize_full_reports_degenerate_variants() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let config = LocalBaConfig::new(5, 15, 4, 2.0, 1e-3, 0.0).expect("valid BA config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);

        let mut no_poses = FullBaProblem {
            poses: Vec::new(),
            landmarks: Vec::new(),
            factors: Vec::new(),
        };
        assert!(matches!(
            ba.optimize_full(&mut no_poses),
            BaResult::Degenerate {
                reason: DegenerateReason::TooFewPoses { count: 0 }
            }
        ));

        let mut no_landmarks = FullBaProblem {
            poses: vec![
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: Pose::identity(),
                },
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: Pose::identity(),
                },
            ],
            landmarks: Vec::new(),
            factors: Vec::new(),
        };
        assert!(matches!(
            ba.optimize_full(&mut no_landmarks),
            BaResult::Degenerate {
                reason: DegenerateReason::TooFewLandmarks { count: 0 }
            }
        ));

        let mut no_factors = FullBaProblem {
            poses: vec![
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: Pose::identity(),
                },
                PoseVariable {
                    keyframe_id: KeyframeId::default(),
                    pose: Pose::identity(),
                },
            ],
            landmarks: vec![LandmarkVariable {
                point_id: MapPointId::default(),
                position: Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 2.0,
                },
            }],
            factors: Vec::new(),
        };
        assert!(matches!(
            ba.optimize_full(&mut no_factors),
            BaResult::Degenerate {
                reason: DegenerateReason::NoFactors
            }
        ));
    }

    #[test]
    fn optimize_full_returns_max_iterations_with_bad_init() {
        let (map, intrinsics, kf_0, kf_1, _, _) =
            build_full_ba_fixture([0.8, -0.3, 0.4, 0.2, -0.1, 0.15]);
        let config = LocalBaConfig::new(5, 1, 4, 2.0, 1e-3, 0.0).expect("valid BA config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);
        let mut problem = FullBaProblem::try_from_map(
            &map,
            &[kf_0, kf_1],
            ba.window_size(),
            ba.min_observations(),
        )
        .expect("full BA problem");
        assert!(matches!(
            ba.optimize_full(&mut problem),
            BaResult::MaxIterations { iterations: 1, .. }
        ));
    }

    #[test]
    fn optimize_full_returns_converged_on_synthetic_scene() {
        let (map, intrinsics, kf_0, kf_1, _, _) =
            build_full_ba_fixture([0.08, -0.03, 0.04, 0.015, -0.01, 0.008]);
        let config = LocalBaConfig::new(5, 15, 4, 2.0, 1e-3, 0.0).expect("valid BA config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);
        let mut problem = FullBaProblem::try_from_map(
            &map,
            &[kf_0, kf_1],
            ba.window_size(),
            ba.min_observations(),
        )
        .expect("full BA problem");
        assert!(matches!(
            ba.optimize_full(&mut problem),
            BaResult::Converged { .. }
        ));
    }

    #[test]
    fn reprojection_jacobian_matches_finite_difference() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let pose = axis_angle_pose([0.1, -0.05, 0.2], [0.06, -0.04, 0.03]);
        let point = Point3 {
            x: 0.4,
            y: -0.2,
            z: 3.8,
        };
        let mut pixel = project_pixel(pose, point, intrinsics);
        pixel.x += 1.7;
        pixel.y -= 0.9;

        let obs = Observation::try_new(point, pixel, intrinsics).expect("observation");
        let (_residual, jac) =
            reprojection_residual_and_jacobian(pose, &obs, intrinsics).expect("jacobian");

        let eps = 1e-3_f32;
        for col in 0..6 {
            let mut delta_pos = [0.0_f32; 6];
            delta_pos[col] = eps;
            let mut delta_neg = [0.0_f32; 6];
            delta_neg[col] = -eps;

            let r_plus = projection_residual(apply_se3_delta(pose, delta_pos), &obs, intrinsics);
            let r_minus = projection_residual(apply_se3_delta(pose, delta_neg), &obs, intrinsics);
            let numeric = [
                (r_plus[0] - r_minus[0]) / (2.0 * eps),
                (r_plus[1] - r_minus[1]) / (2.0 * eps),
            ];

            let err0 = (numeric[0] - jac[0][col]).abs();
            let err1 = (numeric[1] - jac[1][col]).abs();
            let tol0 = 4e-2_f32 + 3e-4_f32 * numeric[0].abs().max(jac[0][col].abs());
            let tol1 = 4e-2_f32 + 3e-4_f32 * numeric[1].abs().max(jac[1][col].abs());
            assert!(
                err0 < tol0 && err1 < tol1,
                "jacobian mismatch col={col}: analytic=({}, {}), numeric=({}, {}), err=({}, {}), tol=({}, {})",
                jac[0][col],
                jac[1][col],
                numeric[0],
                numeric[1],
                err0,
                err1,
                tol0,
                tol1
            );
        }
    }

    #[test]
    fn optimize_keyframe_window_refines_pose_and_landmarks_with_schur() {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let true_pose_0 = Pose::identity();
        let true_pose_1 = axis_angle_pose([0.20, -0.02, 0.03], [0.0, 0.03, -0.01]);
        let noisy_pose_1 = apply_se3_delta(true_pose_1, [0.08, -0.03, 0.04, 0.015, -0.01, 0.008]);

        let points_true = vec![
            Point3 {
                x: -0.35,
                y: -0.25,
                z: 3.2,
            },
            Point3 {
                x: -0.10,
                y: -0.22,
                z: 3.5,
            },
            Point3 {
                x: 0.14,
                y: -0.20,
                z: 3.8,
            },
            Point3 {
                x: 0.32,
                y: -0.10,
                z: 3.4,
            },
            Point3 {
                x: -0.30,
                y: 0.10,
                z: 3.6,
            },
            Point3 {
                x: -0.08,
                y: 0.16,
                z: 4.0,
            },
            Point3 {
                x: 0.16,
                y: 0.12,
                z: 3.3,
            },
            Point3 {
                x: 0.34,
                y: 0.24,
                z: 3.9,
            },
        ];

        let mut keypoints_0 = Vec::with_capacity(points_true.len());
        let mut keypoints_1 = Vec::with_capacity(points_true.len());
        for &point in &points_true {
            keypoints_0.push(
                project_world_point(true_pose_0, point, intrinsics)
                    .expect("point visible in pose 0"),
            );
            keypoints_1.push(
                project_world_point(true_pose_1, point, intrinsics)
                    .expect("point visible in pose 1"),
            );
        }

        let detections_0 = make_detections(
            SensorId::StereoLeft,
            FrameId::new(500),
            640,
            480,
            keypoints_0,
        )
        .expect("detections 0");
        let detections_1 = make_detections(
            SensorId::StereoLeft,
            FrameId::new(501),
            640,
            480,
            keypoints_1,
        )
        .expect("detections 1");

        let mut map = SlamMap::new();
        let kf_0 = map
            .add_keyframe_from_detections(
                detections_0.as_ref(),
                Timestamp::from_nanos(1_000_000),
                true_pose_0,
            )
            .expect("insert keyframe 0");
        let kf_1 = map
            .add_keyframe_from_detections(
                detections_1.as_ref(),
                Timestamp::from_nanos(2_000_000),
                noisy_pose_1,
            )
            .expect("insert keyframe 1");

        for (idx, &point_true) in points_true.iter().enumerate() {
            let kp_0 = map.keyframe_keypoint(kf_0, idx).expect("kf0 keypoint");
            let kp_1 = map.keyframe_keypoint(kf_1, idx).expect("kf1 keypoint");
            let descriptor = detections_0.descriptors()[idx];
            let i = idx as f32;
            let noisy_point = Point3 {
                x: point_true.x + (i - 3.5) * 0.010,
                y: point_true.y - (i - 3.5) * 0.008,
                z: point_true.z + ((idx % 2) as f32 - 0.5) * 0.040,
            };
            let point_id = map
                .add_map_point(noisy_point, descriptor.quantize(), kp_0)
                .expect("insert map point");
            map.add_observation(point_id, kp_1)
                .expect("add shared observation");
        }

        assert_map_invariants(&map).expect("map invariants before BA");
        let before_pose_err = pose_distance(map.keyframe(kf_1).expect("kf1").pose(), true_pose_1);
        let before_landmark_err = mean_landmark_error(&map, kf_0, &points_true);

        let config = LocalBaConfig::new(5, 15, 4, 2.0, 1e-3, 0.0).expect("valid BA config");
        let mut ba = LocalBundleAdjuster::new(intrinsics, config);
        let result = ba.optimize_keyframe_window(&mut map, &[kf_0, kf_1]);
        assert!(
            matches!(
                result,
                BaResult::Converged { .. } | BaResult::MaxIterations { .. }
            ),
            "full local BA should succeed, got {result:?}"
        );
        assert_map_invariants(&map).expect("map invariants after BA");

        let after_pose_err = pose_distance(map.keyframe(kf_1).expect("kf1").pose(), true_pose_1);
        let after_landmark_err = mean_landmark_error(&map, kf_0, &points_true);

        assert!(
            after_pose_err < before_pose_err,
            "pose error did not improve: before={before_pose_err}, after={after_pose_err}"
        );
        assert!(
            after_landmark_err < before_landmark_err,
            "landmark error did not improve: before={before_landmark_err}, after={after_landmark_err}"
        );
    }
}
