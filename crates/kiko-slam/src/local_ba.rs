use std::num::NonZeroUsize;

use crate::{
    math, Observation, PinholeIntrinsics, Pose, Keypoint,
    map::{KeyframeId, KeyframeKeypoint, SlamMap},
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
            LocalBaConfigError::TooFewObservations { min } => write!(
                f,
                "local BA min observations must be >= {min}"
            ),
            LocalBaConfigError::NonPositiveHuber { value } => write!(
                f,
                "local BA huber delta must be > 0 (got {value})"
            ),
            LocalBaConfigError::NegativeDamping { value } => write!(
                f,
                "local BA damping must be >= 0 (got {value})"
            ),
            LocalBaConfigError::NegativeMotionWeight { value } => write!(
                f,
                "local BA motion prior weight must be >= 0 (got {value})"
            ),
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

#[derive(Debug)]
pub enum ObservationSetError {
    TooFew {
        required: usize,
        actual: usize,
    },
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
        Some(ResolvedObservationSet { observations: resolved })
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
        self.frames.push(BaFrame {
            pose,
            observations,
        });
        if self.frames.len() > self.config.window() {
            let excess = self.frames.len() - self.config.window();
            self.frames.drain(0..excess);
        }

        if !self.optimize(map) {
            return None;
        }
        self.frames.last().map(|frame| frame.pose)
    }

    pub fn optimize_keyframe_window(
        &mut self,
        map: &mut SlamMap,
        window: &[KeyframeId],
    ) -> bool {
        let Some((&seed, rest)) = window.split_first() else {
            return false;
        };

        let seed_obs = match build_keyframe_observations(map, seed, self.min_observations()) {
            Some(obs) => obs,
            None => return false,
        };

        self.frames.clear();
        let mut selected = Vec::new();
        if let Some(entry) = map.keyframe(seed) {
            self.frames.push(BaFrame {
                pose: entry.pose(),
                observations: seed_obs,
            });
            selected.push(seed);
        } else {
            return false;
        }

        for &kf_id in rest {
            let Some(entry) = map.keyframe(kf_id) else {
                continue;
            };
            let Some(obs) = build_keyframe_observations(map, kf_id, self.min_observations()) else {
                continue;
            };
            self.frames.push(BaFrame {
                pose: entry.pose(),
                observations: obs,
            });
            selected.push(kf_id);
            if self.frames.len() >= self.config.window() {
                break;
            }
        }

        if self.frames.len() < 2 {
            return false;
        }

        if !self.optimize(map) {
            return false;
        }

        for (kf_id, frame) in selected.iter().zip(self.frames.iter()) {
            let _ = map.set_keyframe_pose(*kf_id, frame.pose);
        }
        true
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
                        let weight = if r_norm <= huber {
                            1.0
                        } else {
                            huber / r_norm
                        };
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
                            b[base + c] += jr;
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
                        b[base_prev + k] -= r;
                        b[base_curr + k] += r;

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
}

fn build_keyframe_observations(
    map: &SlamMap,
    keyframe_id: KeyframeId,
    min_required: NonZeroUsize,
) -> Option<ObservationSet> {
    let pairs = map.keyframe_observation_pixels(keyframe_id).ok()?;
    let observations: Vec<MapObservation> = pairs
        .into_iter()
        .map(|(kp_ref, pixel)| MapObservation::new(kp_ref, pixel))
        .collect();
    ObservationSet::new(observations, min_required).ok()
}

fn reprojection_residual_and_jacobian(
    pose: Pose,
    obs: &Observation,
    intrinsics: PinholeIntrinsics,
) -> Option<([f32; 2], [[f32; 6]; 2])> {
    let world = obs.world();
    let pw = [world.x, world.y, world.z];
    let pc = math::transform_point(pose.rotation(), pose.translation(), pw);
    let x = pc[0];
    let y = pc[1];
    let z = pc[2];
    if z <= 1e-6 {
        return None;
    }

    let u = intrinsics.fx() * (x / z) + intrinsics.cx();
    let v = intrinsics.fy() * (y / z) + intrinsics.cy();
    let pixel = obs.pixel();
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

    let mut jac = [[0.0_f32; 6]; 2];

    jac[0][0] = a1;
    jac[0][1] = a2;
    jac[0][2] = a3;
    jac[1][0] = b1;
    jac[1][1] = b2;
    jac[1][2] = b3;

    jac[0][3] = -(a2 * z - a3 * y);
    jac[0][4] = a1 * z - a3 * x;
    jac[0][5] = -a1 * y + a2 * x;

    jac[1][3] = -(b2 * z - b3 * y);
    jac[1][4] = b1 * z - b3 * x;
    jac[1][5] = -b1 * y + b2 * x;

    // The Jacobian above is for projected pixel coordinates [u, v].
    // Residual is defined as [pixel.x - u, pixel.y - v], so dr/dx = -du/dx.
    for row in &mut jac {
        for value in row {
            *value = -*value;
        }
    }

    Some((residual, jac))
}

fn apply_se3_delta(pose: Pose, delta: [f32; 6]) -> Pose {
    let v = [delta[0], delta[1], delta[2]];
    let w = [delta[3], delta[4], delta[5]];
    let r_delta = so3_exp(w);
    let r = math::mat_mul(r_delta, pose.rotation());
    let t = math::mat_mul_vec(r_delta, pose.translation());
    Pose::from_rt(r, [t[0] + v[0], t[1] + v[1], t[2] + v[2]])
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
    let kx = [
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ];

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
            r[i][j] = if i == j { 1.0 } else { 0.0 }
                + sin_t * kx[i][j]
                + (1.0 - cos_t) * kx2[i][j];
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
            [xx, (r[0][1] + r[1][0]) / (4.0 * xx), (r[0][2] + r[2][0]) / (4.0 * xx)]
        } else if yy >= zz && yy > 1e-6 {
            [(r[0][1] + r[1][0]) / (4.0 * yy), yy, (r[1][2] + r[2][1]) / (4.0 * yy)]
        } else if zz > 1e-6 {
            [(r[0][2] + r[2][0]) / (4.0 * zz), (r[1][2] + r[2][1]) / (4.0 * zz), zz]
        } else {
            [
                r[2][1] - r[1][2],
                r[0][2] - r[2][0],
                r[1][0] - r[0][1],
            ]
        };

        let norm =
            (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if norm > 1e-8 {
            axis = [axis[0] / norm, axis[1] / norm, axis[2] / norm];
        } else {
            axis = [1.0, 0.0, 0.0];
        }

        let skew = [
            r[2][1] - r[1][2],
            r[0][2] - r[2][0],
            r[1][0] - r[0][1],
        ];
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
    use crate::test_helpers::{axis_angle_pose, make_pinhole_intrinsics};
    use crate::{Keypoint, Point3};

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

    fn project_pixel(pose_world_to_camera: Pose, point_world: Point3, intr: PinholeIntrinsics) -> Keypoint {
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
        assert!(l2_3(w, recovered) < 2e-4, "round-trip mismatch: {recovered:?}");
    }

    #[test]
    fn so3_log_is_finite_near_pi() {
        let theta = std::f32::consts::PI - 1e-4;
        let w = [0.0, theta, 0.0];
        let r = so3_exp(w);
        let recovered = so3_log(r);
        assert!(recovered.iter().all(|v| v.is_finite()));

        let recovered_norm =
            (recovered[0] * recovered[0] + recovered[1] * recovered[1] + recovered[2] * recovered[2]).sqrt();
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
    fn reprojection_jacobian_matches_finite_difference() {
        let intrinsics = make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0)
            .expect("intrinsics");
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
}
