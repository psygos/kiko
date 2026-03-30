//! Information-weighted surface belief map.
//!
//! Replaces blind point cloud accumulation with statistically correct
//! fusion. Multiple observations of the same surface merge via inverse-
//! variance weighting, naturally filtering noise and building confidence.
//!
//! Key invariants:
//! - Every observation carries measurement uncertainty (depth_variance)
//! - Fusion uses information-weighted mean (optimal for Gaussian noise)
//! - Consistency tracked via chi-squared innovation statistics
//! - Only confirmed voxels (count ≥ 3, consistent) are rendered

use std::collections::HashMap;
use crate::dense_cloud::DensePoint;
use crate::math;
use crate::Pose;

/// Voxel key: discretized 3D position at the given resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct VoxelKey {
    x: i32,
    y: i32,
    z: i32,
}

impl VoxelKey {
    fn from_position(pos: [f32; 3], inv_voxel_size: f32) -> Self {
        Self {
            x: (pos[0] * inv_voxel_size).floor() as i32,
            y: (pos[1] * inv_voxel_size).floor() as i32,
            z: (pos[2] * inv_voxel_size).floor() as i32,
        }
    }
}

/// A surface belief: the posterior estimate of a voxel's position and quality.
///
/// Uses Welford's online algorithm for numerically stable mean and variance
/// tracking, with inverse-variance (information) weighting for fusion.
#[derive(Clone, Debug)]
struct SurfaceBelief {
    /// Information-weighted position sum: Σ(p_i / σ_i²)
    info_weighted_sum: [f64; 3],
    /// Total information: Σ(1 / σ_i²)
    total_information: f64,
    /// Number of observations merged.
    count: u32,
    /// Sum of weighted squared residuals for chi-squared consistency.
    sum_weighted_sq_residual: f64,
    /// Running color mean.
    color_sum: f64,
}

impl SurfaceBelief {
    fn new() -> Self {
        Self {
            info_weighted_sum: [0.0; 3],
            total_information: 0.0,
            count: 0,
            sum_weighted_sq_residual: 0.0,
            color_sum: 0.0,
        }
    }

    /// The information-weighted mean position.
    fn position(&self) -> [f64; 3] {
        if self.total_information < 1e-15 {
            return [0.0; 3];
        }
        let inv = 1.0 / self.total_information;
        [
            self.info_weighted_sum[0] * inv,
            self.info_weighted_sum[1] * inv,
            self.info_weighted_sum[2] * inv,
        ]
    }

    /// Fused standard deviation: σ = 1/√(total_information).
    /// Decreases as more consistent observations merge.
    fn std_dev(&self) -> f64 {
        if self.total_information < 1e-15 {
            return f64::MAX;
        }
        (1.0 / self.total_information).sqrt()
    }

    /// Chi-squared per DOF. Should be ≈ 1 for consistent observations.
    /// >> 1 means observations disagree (outlier, moving object).
    fn chi_squared_per_dof(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.sum_weighted_sq_residual / (self.count - 1) as f64
    }

    /// Mean color.
    fn color(&self) -> u8 {
        if self.count == 0 {
            return 128;
        }
        (self.color_sum / self.count as f64).clamp(0.0, 255.0) as u8
    }

    /// Integrate a new observation with its measurement uncertainty.
    fn integrate(&mut self, pos: [f64; 3], depth_variance: f64, intensity: u8) {
        let info = 1.0 / depth_variance.max(1e-12);

        // Track consistency before updating the mean
        if self.count > 0 {
            let current = self.position();
            let sq_dist = (pos[0] - current[0]).powi(2)
                + (pos[1] - current[1]).powi(2)
                + (pos[2] - current[2]).powi(2);
            self.sum_weighted_sq_residual += sq_dist * info;
        }

        // Information-weighted accumulation
        for i in 0..3 {
            self.info_weighted_sum[i] += pos[i] * info;
        }
        self.total_information += info;
        self.count += 1;
        self.color_sum += intensity as f64;
    }

    /// Is this belief confirmed? Enough observations AND consistent.
    fn is_confirmed(&self) -> bool {
        self.count >= 3 && self.chi_squared_per_dof() < 10.0
    }
}

/// Configuration for the surface belief map.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceMapConfig {
    /// Voxel size in meters. Points within the same voxel merge.
    pub voxel_size: f32,
    /// Minimum observation count to render.
    pub min_observations: u32,
    /// Maximum chi-squared per DOF for a voxel to be considered consistent.
    pub max_chi_squared: f64,
    /// Maximum total points to render (prevents Rerun overload).
    pub max_render_points: usize,
}

impl Default for SurfaceMapConfig {
    fn default() -> Self {
        Self {
            voxel_size: 0.02,
            min_observations: 3,
            max_chi_squared: 10.0,
            max_render_points: 500_000,
        }
    }
}

/// The surface belief map: information-weighted spatial hash.
#[derive(Debug)]
pub struct SurfaceBeliefMap {
    voxels: HashMap<VoxelKey, SurfaceBelief>,
    config: SurfaceMapConfig,
    inv_voxel_size: f32,
}

impl SurfaceBeliefMap {
    pub fn new(config: SurfaceMapConfig) -> Self {
        Self {
            voxels: HashMap::new(),
            inv_voxel_size: 1.0 / config.voxel_size,
            config,
        }
    }

    /// Integrate a batch of dense points (camera frame) with a pose.
    pub fn integrate(&mut self, points: &[DensePoint], cam_from_map: Pose) {
        let map_from_cam = cam_from_map.inverse();
        let r = map_from_cam.rotation();
        let t = map_from_cam.translation();

        for p in points {
            if p.depth_variance <= 0.0 || !p.depth_variance.is_finite() {
                continue;
            }
            // Transform to map frame
            let world = math::transform_point(r, t, p.position);
            let key = VoxelKey::from_position(world, self.inv_voxel_size);

            let belief = self.voxels.entry(key).or_insert_with(SurfaceBelief::new);
            belief.integrate(
                [world[0] as f64, world[1] as f64, world[2] as f64],
                p.depth_variance as f64,
                p.intensity,
            );
        }
    }

    /// Number of voxels in the map.
    pub fn num_voxels(&self) -> usize {
        self.voxels.len()
    }

    /// Number of confirmed voxels.
    pub fn num_confirmed(&self) -> usize {
        self.voxels
            .values()
            .filter(|v| v.count >= self.config.min_observations && v.chi_squared_per_dof() < self.config.max_chi_squared)
            .count()
    }

    /// Extract confirmed surface points for rendering.
    /// Returns (position, color) tuples in map frame.
    pub fn extract_confirmed(&self) -> Vec<([f32; 3], u8)> {
        let mut points: Vec<([f32; 3], u8)> = self
            .voxels
            .values()
            .filter(|v| v.count >= self.config.min_observations && v.chi_squared_per_dof() < self.config.max_chi_squared)
            .map(|v| {
                let pos = v.position();
                ([pos[0] as f32, pos[1] as f32, pos[2] as f32], v.color())
            })
            .collect();

        // Cap output to prevent Rerun overload
        if points.len() > self.config.max_render_points {
            // Keep every Nth point
            let stride = points.len() / self.config.max_render_points + 1;
            points = points.into_iter().step_by(stride).collect();
        }
        points
    }

    /// Diagnostic summary.
    pub fn summary(&self) -> SurfaceMapSummary {
        let total = self.voxels.len();
        let confirmed = self.num_confirmed();
        let mean_std_dev: f64 = if confirmed > 0 {
            self.voxels
                .values()
                .filter(|v| v.count >= self.config.min_observations)
                .map(|v| v.std_dev())
                .sum::<f64>()
                / confirmed as f64
        } else {
            0.0
        };
        SurfaceMapSummary {
            total_voxels: total,
            confirmed_voxels: confirmed,
            mean_confirmed_std_dev_m: mean_std_dev,
        }
    }
}

/// Diagnostic summary for the surface map.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceMapSummary {
    pub total_voxels: usize,
    pub confirmed_voxels: usize,
    pub mean_confirmed_std_dev_m: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_observation_not_confirmed() {
        let mut map = SurfaceBeliefMap::new(SurfaceMapConfig::default());
        let points = vec![DensePoint {
            position: [0.0, 0.0, 1.0],
            intensity: 128,
            depth_variance: 0.001,
        }];
        map.integrate(&points, Pose::identity());
        assert_eq!(map.num_voxels(), 1);
        assert_eq!(map.num_confirmed(), 0); // needs 3+ observations
    }

    #[test]
    fn three_consistent_observations_confirmed() {
        let mut map = SurfaceBeliefMap::new(SurfaceMapConfig::default());
        let p = DensePoint {
            position: [0.5, 0.5, 2.0],
            intensity: 200,
            depth_variance: 0.01, // σ_z = 0.1m
        };
        // Same point 3 times — lands in same voxel, consistent
        for _ in 0..3 {
            map.integrate(&[p], Pose::identity());
        }
        assert_eq!(map.num_confirmed(), 1);
        let points = map.extract_confirmed();
        assert_eq!(points.len(), 1);
    }

    #[test]
    fn fused_std_dev_decreases_with_observations() {
        let mut belief = SurfaceBelief::new();
        let sigma_single = 0.01_f64; // σ² = 0.01 → σ = 0.1m
        for _ in 0..10 {
            belief.integrate([1.0, 2.0, 3.0], sigma_single, 128);
        }
        // After 10 observations with σ²=0.01 each:
        // total_info = 10/0.01 = 1000
        // σ_fused = 1/√1000 ≈ 0.032m
        let sigma_fused = belief.std_dev();
        assert!(sigma_fused < 0.04, "σ_fused={sigma_fused} should be < 0.04");
        assert!(sigma_fused > 0.02, "σ_fused={sigma_fused} should be > 0.02");
    }

    #[test]
    fn inconsistent_observations_flagged() {
        let mut belief = SurfaceBelief::new();
        // 5 observations at z=1.0
        for _ in 0..5 {
            belief.integrate([0.0, 0.0, 1.0], 0.001, 128);
        }
        // 1 outlier at z=2.0 — very different
        belief.integrate([0.0, 0.0, 2.0], 0.001, 128);
        // chi-squared should be high
        assert!(
            belief.chi_squared_per_dof() > 10.0,
            "chi²/dof={} should indicate inconsistency",
            belief.chi_squared_per_dof()
        );
    }

    #[test]
    fn noise_rejection_via_inverse_variance() {
        let mut belief = SurfaceBelief::new();
        // 5 precise observations at z=1.0 (small σ²)
        for _ in 0..5 {
            belief.integrate([0.0, 0.0, 1.0], 0.0001, 128); // σ=1cm
        }
        // 1 noisy observation at z=1.5 (large σ²)
        belief.integrate([0.0, 0.0, 1.5], 0.1, 128); // σ=31cm

        let pos = belief.position();
        // Position should be very close to 1.0, barely affected by the noisy obs
        assert!(
            (pos[2] - 1.0).abs() < 0.01,
            "z={} should be near 1.0 despite outlier",
            pos[2]
        );
    }
}
