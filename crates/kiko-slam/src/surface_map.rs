//! Information-weighted surface belief map.
//!
//! Replaces blind point cloud accumulation with conservative fusion. Multiple
//! observations of the same surface merge via inverse-variance weighting, but
//! raw sample count is never treated as independent evidence count.
//!
//! Key invariants:
//! - Every observation carries conservative positional uncertainty (`position_variance`)
//! - Within a single integration batch, observations landing in the same voxel are
//!   grouped into one correlated support view before entering the persistent map
//! - Fusion across support views uses information-weighted means
//! - Consistency is tracked via an explicitly named residual energy score
//! - Only confirmed voxels (enough support views, low residual inconsistency) are rendered

use crate::Pose;
use crate::dense_cloud::StableSurfacePoint;
use crate::math;
use std::collections::HashMap;

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

/// Correlated evidence contributed by one integration batch for one voxel.
///
/// Multiple observations within the same batch share pose uncertainty and must not
/// be counted as independent support. We conservatively collapse them into a single
/// representative evidence item whose variance includes both observation variance
/// and within-voxel positional spread.
#[derive(Clone, Debug)]
struct BatchVoxelEvidence {
    position: [f64; 3],
    position_variance: f64,
    intensity: u8,
    raw_observations: u32,
}

#[derive(Clone, Debug)]
struct BatchVoxelAccumulator {
    weight_sum: f64,
    weighted_position_sum: [f64; 3],
    weighted_sq_norm_sum: f64,
    weighted_variance_sum: f64,
    weighted_intensity_sum: f64,
    raw_observations: u32,
}

impl BatchVoxelAccumulator {
    fn new() -> Self {
        Self {
            weight_sum: 0.0,
            weighted_position_sum: [0.0; 3],
            weighted_sq_norm_sum: 0.0,
            weighted_variance_sum: 0.0,
            weighted_intensity_sum: 0.0,
            raw_observations: 0,
        }
    }

    fn add(&mut self, pos: [f64; 3], position_variance: f64, intensity: u8) {
        let weight = 1.0 / position_variance.max(1e-12);
        self.weight_sum += weight;
        self.weighted_position_sum[0] += pos[0] * weight;
        self.weighted_position_sum[1] += pos[1] * weight;
        self.weighted_position_sum[2] += pos[2] * weight;
        self.weighted_sq_norm_sum += weight * (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]);
        self.weighted_variance_sum += weight * position_variance;
        self.weighted_intensity_sum += weight * intensity as f64;
        self.raw_observations = self.raw_observations.saturating_add(1);
    }

    fn finalize(&self) -> Option<BatchVoxelEvidence> {
        if self.weight_sum <= 0.0 || !self.weight_sum.is_finite() {
            return None;
        }

        let inv_weight_sum = 1.0 / self.weight_sum;
        let position = [
            self.weighted_position_sum[0] * inv_weight_sum,
            self.weighted_position_sum[1] * inv_weight_sum,
            self.weighted_position_sum[2] * inv_weight_sum,
        ];
        let position_sq_norm =
            position[0] * position[0] + position[1] * position[1] + position[2] * position[2];
        let spread_variance =
            (self.weighted_sq_norm_sum * inv_weight_sum - position_sq_norm).max(0.0);
        let mean_measurement_variance = self.weighted_variance_sum * inv_weight_sum;
        let position_variance = mean_measurement_variance + spread_variance;
        if !position_variance.is_finite() || position_variance <= 0.0 {
            return None;
        }

        Some(BatchVoxelEvidence {
            position,
            position_variance,
            intensity: (self.weighted_intensity_sum * inv_weight_sum).clamp(0.0, 255.0) as u8,
            raw_observations: self.raw_observations,
        })
    }
}

/// A surface belief: the posterior estimate of a voxel's position and quality.
#[derive(Clone, Debug)]
struct SurfaceBelief {
    /// Information-weighted position sum: Σ(p_i / σ_i²)
    info_weighted_sum: [f64; 3],
    /// Total information: Σ(1 / σ_i²)
    total_information: f64,
    /// Number of correlated support views merged into this belief.
    support_views: u32,
    /// Number of raw stereo observations collapsed into the support views.
    raw_observations: u32,
    /// Sum of weighted squared residuals for consistency scoring.
    residual_energy: f64,
    /// Running color mean.
    color_sum: f64,
}

impl SurfaceBelief {
    fn new() -> Self {
        Self {
            info_weighted_sum: [0.0; 3],
            total_information: 0.0,
            support_views: 0,
            raw_observations: 0,
            residual_energy: 0.0,
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

    /// Mean residual energy per excess support view.
    ///
    /// This is intentionally not named chi-squared: the current map path does not
    /// yet model the full innovation covariance, so this value is an honest
    /// consistency score rather than a calibrated statistical test.
    fn consistency_score(&self) -> f64 {
        if self.support_views < 2 {
            return 0.0;
        }
        self.residual_energy / (self.support_views - 1) as f64
    }

    /// Mean color.
    fn color(&self) -> u8 {
        if self.support_views == 0 {
            return 128;
        }
        (self.color_sum / self.support_views as f64).clamp(0.0, 255.0) as u8
    }

    fn support_views(&self) -> u32 {
        self.support_views
    }

    fn raw_observations(&self) -> u32 {
        self.raw_observations
    }

    /// Integrate one correlated support view with conservative positional uncertainty.
    fn integrate_support_view(&mut self, evidence: &BatchVoxelEvidence) {
        let info = 1.0 / evidence.position_variance.max(1e-12);

        // Track consistency with a weighted online second-moment update. This is
        // order-stable for weighted samples and measures residual spread around the
        // eventual fused mean rather than the pre-update mean.
        if self.total_information > 0.0 {
            let current = self.position();
            let sq_dist = (evidence.position[0] - current[0]).powi(2)
                + (evidence.position[1] - current[1]).powi(2)
                + (evidence.position[2] - current[2]).powi(2);
            let combined_information = self.total_information + info;
            let weighted_delta_scale = info * self.total_information / combined_information;
            self.residual_energy += weighted_delta_scale * sq_dist;
        }

        // Information-weighted accumulation across support views.
        for i in 0..3 {
            self.info_weighted_sum[i] += evidence.position[i] * info;
        }
        self.total_information += info;
        self.support_views = self.support_views.saturating_add(1);
        self.raw_observations = self
            .raw_observations
            .saturating_add(evidence.raw_observations);
        self.color_sum += evidence.intensity as f64;
    }
}

/// Configuration for the surface belief map.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceMapConfig {
    /// Voxel size in meters. Points within the same voxel merge.
    pub voxel_size: f32,
    /// Minimum number of distinct support views required to render.
    pub min_support_views: u32,
    /// Maximum allowed residual consistency score for a confirmed voxel.
    pub max_consistency_score: f64,
    /// Maximum total points to render (prevents Rerun overload).
    pub max_render_points: usize,
}

impl Default for SurfaceMapConfig {
    fn default() -> Self {
        Self {
            voxel_size: 0.05,
            min_support_views: 3,
            max_consistency_score: 8.0,
            max_render_points: 250_000,
        }
    }
}

impl SurfaceMapConfig {
    pub fn from_env() -> Self {
        let mut config = Self::default();
        if let Some(v) = crate::env::env_f32("KIKO_SURFACE_VOXEL_SIZE_M") {
            if v.is_finite() && v > 0.0 {
                config.voxel_size = v;
            }
        }
        if let Some(v) = crate::env::env_usize("KIKO_SURFACE_MIN_OBS") {
            config.min_support_views = v.max(1) as u32;
        }
        if let Some(v) = crate::env::env_usize("KIKO_SURFACE_MIN_SUPPORT_VIEWS") {
            config.min_support_views = v.max(1) as u32;
        }
        if let Some(v) = crate::env::env_f32("KIKO_SURFACE_MAX_CHI2") {
            if v.is_finite() && v > 0.0 {
                config.max_consistency_score = v as f64;
            }
        }
        if let Some(v) = crate::env::env_f32("KIKO_SURFACE_MAX_CONSISTENCY_SCORE") {
            if v.is_finite() && v > 0.0 {
                config.max_consistency_score = v as f64;
            }
        }
        if let Some(v) = crate::env::env_usize("KIKO_SURFACE_MAX_RENDER_POINTS") {
            config.max_render_points = v.max(1);
        }
        config
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

    pub fn config(&self) -> SurfaceMapConfig {
        self.config
    }

    fn is_confirmed_belief(&self, belief: &SurfaceBelief) -> bool {
        belief.support_views() >= self.config.min_support_views
            && belief.consistency_score() <= self.config.max_consistency_score
    }

    /// Integrate a batch of surface observations (camera frame) with a pose.
    pub fn integrate(
        &mut self,
        points: &[StableSurfacePoint],
        cam_from_map: Pose,
    ) -> SurfaceBatchIntegrationSummary {
        let map_from_cam = cam_from_map.inverse();
        let r = map_from_cam.rotation();
        let t = map_from_cam.translation();
        let mut batch = HashMap::<VoxelKey, BatchVoxelAccumulator>::new();
        let mut raw_observations_integrated = 0usize;

        for p in points {
            if p.position_variance <= 0.0 || !p.position_variance.is_finite() {
                continue;
            }
            // Transform to map frame
            let world = math::transform_point(r, t, p.position);
            let key = VoxelKey::from_position(world, self.inv_voxel_size);
            batch
                .entry(key)
                .or_insert_with(BatchVoxelAccumulator::new)
                .add(
                    [world[0] as f64, world[1] as f64, world[2] as f64],
                    p.position_variance as f64,
                    p.intensity,
                );
            raw_observations_integrated = raw_observations_integrated.saturating_add(1);
        }

        let mut support_views_integrated = 0usize;
        for (key, grouped) in batch {
            let Some(evidence) = grouped.finalize() else {
                continue;
            };
            let belief = self.voxels.entry(key).or_insert_with(SurfaceBelief::new);
            belief.integrate_support_view(&evidence);
            support_views_integrated = support_views_integrated.saturating_add(1);
        }

        SurfaceBatchIntegrationSummary {
            raw_observations_integrated,
            support_views_integrated,
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
            .filter(|v| self.is_confirmed_belief(v))
            .count()
    }

    /// Extract confirmed surface points for rendering.
    /// Returns (position, color) tuples in map frame.
    pub fn extract_confirmed(&self) -> Vec<([f32; 3], u8)> {
        let mut points: Vec<([f32; 3], u8)> = self
            .voxels
            .values()
            .filter(|v| self.is_confirmed_belief(v))
            .map(|v| {
                let pos = v.position();
                ([pos[0] as f32, pos[1] as f32, pos[2] as f32], v.color())
            })
            .collect();

        // Keep extraction deterministic before capping.
        points.sort_by(|(a, _), (b, _)| {
            a[0].total_cmp(&b[0])
                .then(a[1].total_cmp(&b[1]))
                .then(a[2].total_cmp(&b[2]))
        });

        // Cap output to prevent Rerun overload.
        if points.len() > self.config.max_render_points {
            let stride = points.len() / self.config.max_render_points + 1;
            points = points.into_iter().step_by(stride).collect();
        }
        points
    }

    /// Diagnostic summary.
    pub fn summary(&self) -> SurfaceMapSummary {
        let total = self.voxels.len();
        let confirmed = self.num_confirmed();
        let confirmed_beliefs: Vec<&SurfaceBelief> = self
            .voxels
            .values()
            .filter(|v| self.is_confirmed_belief(v))
            .collect();
        let mean_std_dev: f64 = if confirmed > 0 {
            confirmed_beliefs.iter().map(|v| v.std_dev()).sum::<f64>() / confirmed as f64
        } else {
            0.0
        };
        let mean_support_views = if confirmed > 0 {
            confirmed_beliefs
                .iter()
                .map(|v| v.support_views() as f64)
                .sum::<f64>()
                / confirmed as f64
        } else {
            0.0
        };
        let mean_raw_observations = if confirmed > 0 {
            confirmed_beliefs
                .iter()
                .map(|v| v.raw_observations() as f64)
                .sum::<f64>()
                / confirmed as f64
        } else {
            0.0
        };
        SurfaceMapSummary {
            total_voxels: total,
            confirmed_voxels: confirmed,
            confirmed_ratio: if total > 0 {
                confirmed as f64 / total as f64
            } else {
                0.0
            },
            mean_confirmed_std_dev_m: mean_std_dev,
            mean_confirmed_support_views: mean_support_views,
            mean_confirmed_raw_observations: mean_raw_observations,
        }
    }
}

/// Diagnostic summary for the surface map.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceMapSummary {
    pub total_voxels: usize,
    pub confirmed_voxels: usize,
    pub confirmed_ratio: f64,
    pub mean_confirmed_std_dev_m: f64,
    pub mean_confirmed_support_views: f64,
    pub mean_confirmed_raw_observations: f64,
}

/// Summary of one integration batch after correlated observations are grouped.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SurfaceBatchIntegrationSummary {
    pub raw_observations_integrated: usize,
    pub support_views_integrated: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RectifiedRowMismatchPx;

    #[test]
    fn single_observation_not_confirmed() {
        let mut map = SurfaceBeliefMap::new(SurfaceMapConfig::default());
        let points = vec![StableSurfacePoint {
            position: [0.0, 0.0, 1.0],
            intensity: 128,
            position_variance: 0.001,
            rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.0).expect("row mismatch"),
        }];
        let summary = map.integrate(&points, Pose::identity());
        assert_eq!(
            summary,
            SurfaceBatchIntegrationSummary {
                raw_observations_integrated: 1,
                support_views_integrated: 1,
            }
        );
        assert_eq!(map.num_voxels(), 1);
        assert_eq!(map.num_confirmed(), 0); // needs 3+ support views
    }

    #[test]
    fn single_batch_duplicate_points_do_not_count_as_multiple_support_views() {
        let mut map = SurfaceBeliefMap::new(SurfaceMapConfig::default());
        let points = vec![
            StableSurfacePoint {
                position: [0.01, 0.01, 2.0],
                intensity: 200,
                position_variance: 0.01,
                rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.0).expect("row mismatch"),
            },
            StableSurfacePoint {
                position: [0.015, 0.012, 2.0],
                intensity: 190,
                position_variance: 0.01,
                rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.0).expect("row mismatch"),
            },
            StableSurfacePoint {
                position: [0.012, 0.017, 2.0],
                intensity: 210,
                position_variance: 0.01,
                rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.0).expect("row mismatch"),
            },
        ];
        let summary = map.integrate(&points, Pose::identity());
        assert_eq!(
            summary,
            SurfaceBatchIntegrationSummary {
                raw_observations_integrated: 3,
                support_views_integrated: 1,
            }
        );
        assert_eq!(map.num_voxels(), 1);
        assert_eq!(map.num_confirmed(), 0);
    }

    #[test]
    fn three_consistent_support_views_confirmed() {
        let mut map = SurfaceBeliefMap::new(SurfaceMapConfig::default());
        let p = StableSurfacePoint {
            position: [0.5, 0.5, 2.0],
            intensity: 200,
            position_variance: 0.01, // σ = 0.1m equivalent positional variance
            rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.0).expect("row mismatch"),
        };
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
            belief.integrate_support_view(&BatchVoxelEvidence {
                position: [1.0, 2.0, 3.0],
                position_variance: sigma_single,
                intensity: 128,
                raw_observations: 1,
            });
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
            belief.integrate_support_view(&BatchVoxelEvidence {
                position: [0.0, 0.0, 1.0],
                position_variance: 0.001,
                intensity: 128,
                raw_observations: 1,
            });
        }
        // 1 outlier at z=2.0 — very different
        belief.integrate_support_view(&BatchVoxelEvidence {
            position: [0.0, 0.0, 2.0],
            position_variance: 0.001,
            intensity: 128,
            raw_observations: 1,
        });
        // consistency score should be high
        assert!(
            belief.consistency_score() > 10.0,
            "consistency_score={} should indicate inconsistency",
            belief.consistency_score()
        );
    }

    #[test]
    fn noise_rejection_via_inverse_variance() {
        let mut belief = SurfaceBelief::new();
        // 5 precise observations at z=1.0 (small σ²)
        for _ in 0..5 {
            belief.integrate_support_view(&BatchVoxelEvidence {
                position: [0.0, 0.0, 1.0],
                position_variance: 0.0001,
                intensity: 128,
                raw_observations: 1,
            }); // σ=1cm
        }
        // 1 noisy observation at z=1.5 (large σ²)
        belief.integrate_support_view(&BatchVoxelEvidence {
            position: [0.0, 0.0, 1.5],
            position_variance: 0.1,
            intensity: 128,
            raw_observations: 1,
        }); // σ=31cm

        let pos = belief.position();
        // Position should be very close to 1.0, barely affected by the noisy obs
        assert!(
            (pos[2] - 1.0).abs() < 0.01,
            "z={} should be near 1.0 despite outlier",
            pos[2]
        );
    }
}
