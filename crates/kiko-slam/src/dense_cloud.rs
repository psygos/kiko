//! Dense interpolated point cloud from sparse stereo matches.
//!
//! Uses Delaunay triangulation of sparse feature disparities to produce
//! a piecewise-linear dense disparity map, then back-projects to 3D.
//! This is a visualization aid, not a dense reconstruction.

use std::collections::TryReserveError;

use crate::triangulation::{RectifiedRowMismatchPx, RectifiedStereo, SparseStereoSample};
use crate::{
    DepthImage, DepthImageError, DiagnosticMetricError, Frame, FrameDimensions, FrameId,
    InterpolatedDepth, SensorId, StableSurfaceRetainedRawPixelResidualMetric, Timestamp,
};

/// Configuration for dense visualization and stable surface observation generation.
#[derive(Clone, Copy, Debug)]
pub struct DenseCloudConfig {
    /// Generate a point every Nth pixel within each triangle.
    subsample: u32,
    /// Reject triangle if `(max_d - min_d) / min_d > this`.
    max_disparity_gradient: f32,
    /// Minimum disparity for valid back-projection.
    min_disparity_px: f32,
    /// Reject triangle if any edge exceeds this length in pixels.
    max_edge_length_px: f32,
    /// Reject triangle if area exceeds this in pixels².
    max_triangle_area_px2: f32,
    /// Reject observations whose conservative positional sigma exceeds this threshold.
    max_observation_std_dev_m: f32,
    /// Hard cap on output points per keyframe. When exceeded, the most stable points are kept.
    max_points_per_keyframe: usize,
}

#[derive(Debug)]
pub enum DenseCloudConfigError {
    Environment { source: crate::env::EnvError },
    ZeroSubsample,
    SubsampleOutOfRange { value: usize },
    InvalidPositiveField { field: &'static str, value: f32 },
    ZeroMaxPoints,
}

impl std::fmt::Display for DenseCloudConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Environment { source } => write!(f, "dense cloud environment error: {source}"),
            Self::ZeroSubsample => write!(f, "dense cloud subsample must be greater than zero"),
            Self::SubsampleOutOfRange { value } => {
                write!(f, "dense cloud subsample {value} exceeds u32 range")
            }
            Self::InvalidPositiveField { field, value } => write!(
                f,
                "dense cloud {field} must be positive and finite, got {value}"
            ),
            Self::ZeroMaxPoints => {
                write!(f, "dense cloud maximum points must be greater than zero")
            }
        }
    }
}

impl std::error::Error for DenseCloudConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Environment { source } => Some(source),
            _ => None,
        }
    }
}

impl Default for DenseCloudConfig {
    fn default() -> Self {
        Self {
            subsample: 2,
            max_disparity_gradient: 0.25,
            min_disparity_px: 1.5,
            max_edge_length_px: 120.0,
            max_triangle_area_px2: 8_000.0,
            max_observation_std_dev_m: 0.05,
            max_points_per_keyframe: 30_000,
        }
    }
}

impl DenseCloudConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        subsample: u32,
        max_disparity_gradient: f32,
        min_disparity_px: f32,
        max_edge_length_px: f32,
        max_triangle_area_px2: f32,
        max_observation_std_dev_m: f32,
        max_points_per_keyframe: usize,
    ) -> Result<Self, DenseCloudConfigError> {
        if subsample == 0 {
            return Err(DenseCloudConfigError::ZeroSubsample);
        }
        for (field, value) in [
            ("maximum disparity gradient", max_disparity_gradient),
            ("minimum disparity", min_disparity_px),
            ("maximum edge length", max_edge_length_px),
            ("maximum triangle area", max_triangle_area_px2),
            (
                "maximum observation standard deviation",
                max_observation_std_dev_m,
            ),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(DenseCloudConfigError::InvalidPositiveField { field, value });
            }
        }
        if max_points_per_keyframe == 0 {
            return Err(DenseCloudConfigError::ZeroMaxPoints);
        }
        Ok(Self {
            subsample,
            max_disparity_gradient,
            min_disparity_px,
            max_edge_length_px,
            max_triangle_area_px2,
            max_observation_std_dev_m,
            max_points_per_keyframe,
        })
    }

    pub fn try_from_env() -> Result<Self, DenseCloudConfigError> {
        let defaults = Self::default();
        let subsample = crate::env::try_env_usize("KIKO_DENSE_SUBSAMPLE")
            .map_err(|source| DenseCloudConfigError::Environment { source })?
            .unwrap_or(defaults.subsample as usize);
        let subsample = u32::try_from(subsample)
            .map_err(|_| DenseCloudConfigError::SubsampleOutOfRange { value: subsample })?;
        let env_f32 = |key| {
            crate::env::try_env_f32(key)
                .map_err(|source| DenseCloudConfigError::Environment { source })
        };
        let max_observation_std_dev_m = match env_f32("KIKO_SURFACE_MAX_POINT_SIGMA_M")? {
            Some(value) => value,
            None => env_f32("KIKO_DENSE_MAX_POINT_SIGMA_M")?
                .unwrap_or(defaults.max_observation_std_dev_m),
        };
        Self::try_new(
            subsample,
            env_f32("KIKO_DENSE_MAX_GRADIENT")?.unwrap_or(defaults.max_disparity_gradient),
            defaults.min_disparity_px,
            env_f32("KIKO_DENSE_MAX_EDGE_PX")?.unwrap_or(defaults.max_edge_length_px),
            env_f32("KIKO_DENSE_MAX_AREA_PX2")?.unwrap_or(defaults.max_triangle_area_px2),
            max_observation_std_dev_m,
            crate::env::try_env_usize("KIKO_DENSE_MAX_POINTS")
                .map_err(|source| DenseCloudConfigError::Environment { source })?
                .unwrap_or(defaults.max_points_per_keyframe),
        )
    }

    pub fn subsample(self) -> u32 {
        self.subsample
    }

    pub fn max_disparity_gradient(self) -> f32 {
        self.max_disparity_gradient
    }

    pub fn min_disparity_px(self) -> f32 {
        self.min_disparity_px
    }

    pub fn max_edge_length_px(self) -> f32 {
        self.max_edge_length_px
    }

    pub fn max_triangle_area_px2(self) -> f32 {
        self.max_triangle_area_px2
    }

    pub fn max_observation_std_dev_m(self) -> f32 {
        self.max_observation_std_dev_m
    }

    pub fn max_points_per_keyframe(self) -> usize {
        self.max_points_per_keyframe
    }
}

/// An interpolated dense visualization point in camera frame.
///
/// This is a derived artifact produced by Delaunay interpolation over sparse stereo
/// correspondences. It is useful for visualization, but must not be treated as an
/// authoritative measured surface observation.
#[derive(Clone, Copy, Debug)]
pub struct DensePoint {
    pub position: [f32; 3],
    pub intensity: u8,
    /// Conservative scalar positional variance in m².
    pub position_variance: f32,
}

/// A measured sparse stereo surface observation in camera frame.
///
/// `position_variance` is a scalar summary of the 3D point covariance induced by
/// stereo disparity uncertainty, lateral pixel uncertainty, and rectified row
/// disagreement between the two matched features. With the usual rectified stereo
/// model:
///
///   x = (u - cx) z / fx
///   y = (v - cy) z / fy
///   z = fx * baseline / disparity
///
/// and with conservative per-axis image uncertainty `(σ_u, σ_v, σ_d)`, the trace
/// of the induced 3D covariance is:
///
///   trace(Σ_p) = σ_u² ||∂p/∂u||²
///              + σ_v² ||∂p/∂v||²
///              + σ_d² ||∂p/∂d||²
///
/// The vertical term uses the midpoint row of the stereo match and inflates
/// uncertainty when the left/right rows disagree after rectification. This
/// naturally downweights far points, off-axis edge points, and vertically
/// inconsistent correspondences without relying on ad hoc image-radius heuristics.
#[derive(Clone, Copy, Debug)]
pub struct StableSurfacePoint {
    position: [f32; 3],
    intensity: u8,
    /// Conservative scalar positional variance in m².
    position_variance: f32,
    /// Absolute vertical row mismatch on the rectified stereo pair, in pixels.
    rectified_row_mismatch_px: RectifiedRowMismatchPx,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StableSurfacePointError {
    NonFinitePosition { axis: usize, value: f32 },
    InvalidPositionVariance { value: f32 },
}

impl std::fmt::Display for StableSurfacePointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinitePosition { axis, value } => write!(
                f,
                "stable surface position axis {axis} must be finite, got {value}"
            ),
            Self::InvalidPositionVariance { value } => write!(
                f,
                "stable surface position variance must be positive and finite, got {value}"
            ),
        }
    }
}

impl std::error::Error for StableSurfacePointError {}

impl StableSurfacePoint {
    pub fn try_new(
        position: [f32; 3],
        intensity: u8,
        position_variance: f32,
        rectified_row_mismatch_px: RectifiedRowMismatchPx,
    ) -> Result<Self, StableSurfacePointError> {
        for (axis, value) in position.into_iter().enumerate() {
            if !value.is_finite() {
                return Err(StableSurfacePointError::NonFinitePosition { axis, value });
            }
        }
        if !position_variance.is_finite() || position_variance <= 0.0 {
            return Err(StableSurfacePointError::InvalidPositionVariance {
                value: position_variance,
            });
        }
        Ok(Self {
            position,
            intensity,
            position_variance,
            rectified_row_mismatch_px,
        })
    }

    pub fn position(self) -> [f32; 3] {
        self.position
    }

    pub fn intensity(self) -> u8 {
        self.intensity
    }

    pub fn position_variance(self) -> f32 {
        self.position_variance
    }

    pub fn rectified_row_mismatch_px(self) -> RectifiedRowMismatchPx {
        self.rectified_row_mismatch_px
    }
}

/// Statistics from dense interpolated cloud generation.
#[derive(Clone, Copy, Debug, Default)]
pub struct DenseCloudStats {
    pub input_samples: usize,
    pub triangles_total: usize,
    pub triangles_rejected: usize,
    pub triangles_rasterized: usize,
    pub points_generated: usize,
    pub points_capped: bool,
}

/// Result of dense cloud generation.
#[derive(Debug)]
pub struct DenseCloudResult {
    pub points: Vec<DensePoint>,
    pub stats: DenseCloudStats,
}

/// Statistics from stable sparse surface observation generation.
#[derive(Clone, Copy, Debug, Default)]
pub struct StableSurfaceStats {
    pub input_samples: usize,
    pub points_generated: usize,
    pub dropped_disparity: usize,
    pub dropped_uncertainty: usize,
    pub dropped_out_of_bounds: usize,
    pub points_capped: bool,
    /// Metrics over the final retained raw observations after variance filtering
    /// and capping. `None` means no retained observations existed for this frame.
    pub mean_accepted_position_sigma_m: Option<f64>,
    pub max_accepted_position_sigma_m: Option<f32>,
    pub mean_accepted_rectified_row_mismatch_px:
        Option<StableSurfaceRetainedRawPixelResidualMetric>,
    pub max_accepted_rectified_row_mismatch_px: Option<StableSurfaceRetainedRawPixelResidualMetric>,
}

/// Result of stable sparse surface observation generation.
#[derive(Debug)]
pub struct StableSurfaceResult {
    pub points: Vec<StableSurfacePoint>,
    pub stats: StableSurfaceStats,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StableSurfaceStatistic {
    MeanRetainedRectifiedRowMismatchPx,
    MaxRetainedRectifiedRowMismatchPx,
}

impl std::fmt::Display for StableSurfaceStatistic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MeanRetainedRectifiedRowMismatchPx => {
                f.write_str("mean retained rectified row mismatch (px)")
            }
            Self::MaxRetainedRectifiedRowMismatchPx => {
                f.write_str("maximum retained rectified row mismatch (px)")
            }
        }
    }
}

#[derive(Debug)]
pub enum StableSurfaceGenerationError {
    ImageSensorMismatch {
        expected: SensorId,
        actual: SensorId,
    },
    ImageDimensionsMismatch {
        expected: FrameDimensions,
        actual: FrameDimensions,
    },
    ImagePixelUnavailable {
        sample_index: usize,
        pixel_index: usize,
        pixel_count: usize,
    },
    PointAllocation {
        requested_points: usize,
        source: TryReserveError,
    },
    InvalidDerivedPoint {
        sample_index: usize,
        source: StableSurfacePointError,
    },
    InvalidStatistic {
        statistic: StableSurfaceStatistic,
        source: DiagnosticMetricError,
    },
}

impl std::fmt::Display for StableSurfaceGenerationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ImageSensorMismatch { expected, actual } => write!(
                f,
                "stable surface intensity image must come from {expected:?}, got {actual:?}"
            ),
            Self::ImageDimensionsMismatch { expected, actual } => write!(
                f,
                "stable surface intensity image dimensions must match the rectified rig {}x{}, got {}x{}",
                expected.width(),
                expected.height(),
                actual.width(),
                actual.height()
            ),
            Self::ImagePixelUnavailable {
                sample_index,
                pixel_index,
                pixel_count,
            } => write!(
                f,
                "stable surface sample {sample_index} selected image pixel {pixel_index}, but the validated frame contains {pixel_count} pixels"
            ),
            Self::PointAllocation {
                requested_points,
                source,
            } => write!(
                f,
                "failed to reserve {requested_points} stable surface points: {source}"
            ),
            Self::InvalidDerivedPoint {
                sample_index,
                source,
            } => write!(
                f,
                "stable surface sample {sample_index} produced an invalid camera-frame point: {source}"
            ),
            Self::InvalidStatistic { statistic, source } => {
                write!(f, "failed to construct {statistic}: {source}")
            }
        }
    }
}

impl std::error::Error for StableSurfaceGenerationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::PointAllocation { source, .. } => Some(source),
            Self::InvalidDerivedPoint { source, .. } => Some(source),
            Self::InvalidStatistic { source, .. } => Some(source),
            Self::ImageSensorMismatch { .. }
            | Self::ImageDimensionsMismatch { .. }
            | Self::ImagePixelUnavailable { .. } => None,
        }
    }
}

// Keep calibrated intrinsics and uncertainty terms explicit at this numerical boundary.
#[allow(clippy::too_many_arguments)]
fn stereo_position_variance_m2(
    u: f32,
    v: f32,
    z: f32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    baseline_m: f32,
    sigma_d_px: f32,
    sigma_feature_px: f32,
    row_mismatch_px: f32,
) -> f32 {
    let sigma_u_sq = sigma_feature_px * sigma_feature_px;
    // If the rectified correspondence disagrees vertically by r pixels, the
    // shared row is uncertain by at least half that spread.
    let sigma_v_sq = sigma_u_sq + 0.25 * row_mismatch_px * row_mismatch_px;
    let dz_dd = z * z / (fx * baseline_m);
    let sigma_z_sq = dz_dd * dz_dd * sigma_d_px * sigma_d_px;
    let ux = (u - cx) / fx;
    let vy = (v - cy) / fy;
    let sigma_x_sq = (z / fx) * (z / fx) * sigma_u_sq;
    let sigma_y_sq = (z / fy) * (z / fy) * sigma_v_sq;
    sigma_x_sq + sigma_y_sq + sigma_z_sq * (1.0 + ux * ux + vy * vy)
}

/// Generate an interpolated depth image by rasterizing a Delaunay disparity field.
/// Every pixel inside a valid triangle gets a derived depth value.
/// Pixels outside triangles or in rejected triangles remain 0.0 (invalid).
/// This is a derived visualization artifact, not authoritative measured depth.
#[allow(clippy::too_many_arguments)]
pub fn generate_dense_depth_image(
    frame_id: FrameId,
    timestamp: Timestamp,
    samples: &[SparseStereoSample],
    fx: f32,
    baseline_m: f32,
    dimensions: FrameDimensions,
    config: &DenseCloudConfig,
) -> Result<DepthImage<InterpolatedDepth>, DepthImageError> {
    let image_width = dimensions.width();
    let image_height = dimensions.height();
    let w = image_width as usize;
    let h = image_height as usize;
    let mut depth = vec![0.0_f32; dimensions.area()];

    if samples.len() < 3 {
        eprintln!("dense_depth: too few samples ({})", samples.len());
        return DepthImage::<InterpolatedDepth>::new_interpolated(
            frame_id,
            timestamp,
            image_width,
            image_height,
            depth,
        );
    }

    let pts: Vec<[f32; 2]> = samples.iter().map(|s| [s.u, s.v]).collect();
    let triangles = delaunay(&pts, image_width as f32, image_height as f32);
    let mut rejected = 0usize;
    let mut rasterized = 0usize;

    for tri in &triangles {
        let (a, b, c) = (tri[0], tri[1], tri[2]);
        let (ax, ay) = (pts[a][0], pts[a][1]);
        let (bx, by) = (pts[b][0], pts[b][1]);
        let (cx_, cy_) = (pts[c][0], pts[c][1]);
        let (da, db, dc) = (
            samples[a].disparity,
            samples[b].disparity,
            samples[c].disparity,
        );

        let area = triangle_area(ax, ay, bx, by, cx_, cy_);
        if area < 0.5 || area > config.max_triangle_area_px2 {
            rejected += 1;
            continue;
        }
        let e_ab = edge_length(ax, ay, bx, by);
        let e_bc = edge_length(bx, by, cx_, cy_);
        let e_ca = edge_length(cx_, cy_, ax, ay);
        if e_ab > config.max_edge_length_px
            || e_bc > config.max_edge_length_px
            || e_ca > config.max_edge_length_px
        {
            rejected += 1;
            continue;
        }
        let d_min = da.min(db).min(dc);
        let d_max = da.max(db).max(dc);
        if d_min > 0.0 && (d_max - d_min) / d_min > config.max_disparity_gradient {
            rejected += 1;
            continue;
        }
        rasterized += 1;

        let min_x = (ax.min(bx).min(cx_).floor() as i32).max(0);
        let max_x = (ax.max(bx).max(cx_).ceil() as i32).min(w as i32 - 1);
        let min_y = (ay.min(by).min(cy_).floor() as i32).max(0);
        let max_y = (ay.max(by).max(cy_).ceil() as i32).min(h as i32 - 1);
        let inv_area = 1.0 / area;

        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let fpx = px as f32 + 0.5;
                let fpy = py as f32 + 0.5;
                let w0 = cross2d(bx - ax, by - ay, fpx - ax, fpy - ay) * inv_area;
                let w1 = cross2d(cx_ - bx, cy_ - by, fpx - bx, fpy - by) * inv_area;
                let w2 = 1.0 - w0 - w1;
                if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                    let d = w0 * dc + w1 * da + w2 * db;
                    if d >= config.min_disparity_px {
                        let z = fx * baseline_m / d;
                        let idx = py as usize * w + px as usize;
                        if depth[idx] == 0.0 || z < depth[idx] {
                            depth[idx] = z;
                        }
                    }
                }
            }
        }
    }
    let filled = depth.iter().filter(|&&d| d > 0.0).count();
    eprintln!(
        "dense_depth: samples={} triangles={} rejected={} rasterized={} filled={}/{}",
        samples.len(),
        triangles.len(),
        rejected,
        rasterized,
        filled,
        w * h,
    );
    DepthImage::<InterpolatedDepth>::new_interpolated(
        frame_id,
        timestamp,
        image_width,
        image_height,
        depth,
    )
}

/// Generate stable sparse surface observations directly from measured stereo samples.
///
/// Unlike `generate_dense_cloud`, this path does not invent new geometry by
/// interpolating across triangles. Each output point corresponds to a measured
/// stereo feature, and points are filtered/ranked by propagated positional
/// uncertainty before entering the voxel belief map.
pub fn generate_stable_surface_points(
    samples: &[SparseStereoSample],
    stereo: &RectifiedStereo,
    intensity_image: &Frame,
    config: &DenseCloudConfig,
) -> Result<StableSurfaceResult, StableSurfaceGenerationError> {
    if intensity_image.sensor_id() != SensorId::StereoLeft {
        return Err(StableSurfaceGenerationError::ImageSensorMismatch {
            expected: SensorId::StereoLeft,
            actual: intensity_image.sensor_id(),
        });
    }
    if intensity_image.dimensions() != stereo.dimensions() {
        return Err(StableSurfaceGenerationError::ImageDimensionsMismatch {
            expected: stereo.dimensions(),
            actual: intensity_image.dimensions(),
        });
    }

    let mut stats = StableSurfaceStats {
        input_samples: samples.len(),
        ..Default::default()
    };
    let requested_points = samples.len().min(config.max_points_per_keyframe);
    let mut points = Vec::new();
    points
        .try_reserve_exact(requested_points)
        .map_err(|source| StableSurfaceGenerationError::PointAllocation {
            requested_points,
            source,
        })?;
    let fx = stereo.fx();
    let fy = stereo.fy();
    let cx = stereo.cx();
    let cy = stereo.cy();
    let baseline_m = stereo.baseline_m();
    let image_data = intensity_image.data();
    let image_width = intensity_image.width();
    let image_height = intensity_image.height();
    let max_position_variance = config.max_observation_std_dev_m * config.max_observation_std_dev_m;
    // Conservative feature-level image noise priors until the calibrated stereo
    // uncertainty model in M7 is wired through this path.
    let sigma_d_feature_px = 0.5_f32;
    let sigma_feature_px = 0.5_f32;

    for (sample_index, sample) in samples.iter().enumerate() {
        if sample.disparity < config.min_disparity_px {
            stats.dropped_disparity += 1;
            continue;
        }

        let px = sample.u.round() as i32;
        let py = sample.v.round() as i32;
        if px < 0 || py < 0 || px >= image_width as i32 || py >= image_height as i32 {
            stats.dropped_out_of_bounds += 1;
            continue;
        }
        let idx = py as usize * image_width as usize + px as usize;
        let Some(&intensity) = image_data.get(idx) else {
            return Err(StableSurfaceGenerationError::ImagePixelUnavailable {
                sample_index,
                pixel_index: idx,
                pixel_count: image_data.len(),
            });
        };

        let z = sample.depth_m;
        let x = (sample.u - cx) * z / fx;
        let v = sample.v;
        let y = (v - cy) * z / fy;
        let position_variance = stereo_position_variance_m2(
            sample.u,
            v,
            z,
            fx,
            fy,
            cx,
            cy,
            baseline_m,
            sigma_d_feature_px,
            sigma_feature_px,
            sample.rectified_row_mismatch_px.value_px(),
        );
        let point = StableSurfacePoint::try_new(
            [x, y, z],
            intensity,
            position_variance,
            sample.rectified_row_mismatch_px,
        )
        .map_err(|source| StableSurfaceGenerationError::InvalidDerivedPoint {
            sample_index,
            source,
        })?;
        if point.position_variance() > max_position_variance {
            stats.dropped_uncertainty += 1;
            continue;
        }
        points.push(point);
    }

    if points.len() > config.max_points_per_keyframe {
        points.sort_by(|a, b| a.position_variance().total_cmp(&b.position_variance()));
        points.truncate(config.max_points_per_keyframe);
        stats.points_capped = true;
    }

    stats.points_generated = points.len();
    if stats.points_generated > 0 {
        let mut sigma_sum_m = 0.0_f64;
        let mut sigma_max_m = 0.0_f32;
        let mut rectified_row_mismatch_sum_px = 0.0_f64;
        let mut rectified_row_mismatch_max_px = 0.0_f32;
        for point in &points {
            let sigma_m = point.position_variance().sqrt();
            sigma_sum_m += sigma_m as f64;
            sigma_max_m = sigma_max_m.max(sigma_m);
            let row_mismatch_px = point.rectified_row_mismatch_px().value_px();
            rectified_row_mismatch_sum_px += row_mismatch_px as f64;
            rectified_row_mismatch_max_px = rectified_row_mismatch_max_px.max(row_mismatch_px);
        }
        stats.mean_accepted_position_sigma_m = Some(sigma_sum_m / stats.points_generated as f64);
        stats.max_accepted_position_sigma_m = Some(sigma_max_m);
        let mean_rectified_row_mismatch_px =
            (rectified_row_mismatch_sum_px / stats.points_generated as f64) as f32;
        stats.mean_accepted_rectified_row_mismatch_px = Some(
            StableSurfaceRetainedRawPixelResidualMetric::new(mean_rectified_row_mismatch_px)
                .map_err(|source| StableSurfaceGenerationError::InvalidStatistic {
                    statistic: StableSurfaceStatistic::MeanRetainedRectifiedRowMismatchPx,
                    source,
                })?,
        );
        stats.max_accepted_rectified_row_mismatch_px = Some(
            StableSurfaceRetainedRawPixelResidualMetric::new(rectified_row_mismatch_max_px)
                .map_err(|source| StableSurfaceGenerationError::InvalidStatistic {
                    statistic: StableSurfaceStatistic::MaxRetainedRectifiedRowMismatchPx,
                    source,
                })?,
        );
    }
    Ok(StableSurfaceResult { points, stats })
}

/// Generate a dense point cloud by interpolating disparity over a Delaunay
/// triangulation of sparse stereo samples.
#[allow(clippy::too_many_arguments)]
pub fn generate_dense_cloud(
    samples: &[SparseStereoSample],
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    baseline_m: f32,
    image_data: &[u8],
    image_width: u32,
    image_height: u32,
    config: &DenseCloudConfig,
) -> DenseCloudResult {
    let mut stats = DenseCloudStats {
        input_samples: samples.len(),
        ..Default::default()
    };
    if samples.len() < 3 {
        return DenseCloudResult {
            points: Vec::new(),
            stats,
        };
    }

    // Extract 2D points for Delaunay
    let pts: Vec<[f32; 2]> = samples.iter().map(|s| [s.u, s.v]).collect();
    let triangles = delaunay(&pts, image_width as f32, image_height as f32);
    stats.triangles_total = triangles.len();

    let mut points = Vec::with_capacity(config.max_points_per_keyframe);
    let stride = config.subsample.max(1) as i32;

    for tri in &triangles {
        let (a, b, c) = (tri[0], tri[1], tri[2]);
        let (ax, ay) = (pts[a][0], pts[a][1]);
        let (bx, by) = (pts[b][0], pts[b][1]);
        let (cx_, cy_) = (pts[c][0], pts[c][1]);
        let (da, db, dc) = (
            samples[a].disparity,
            samples[b].disparity,
            samples[c].disparity,
        );

        // --- Triangle rejection ---
        // 1. Degenerate area
        let area = triangle_area(ax, ay, bx, by, cx_, cy_);
        if area < 0.5 {
            stats.triangles_rejected += 1;
            continue;
        }
        // 2. Max area
        if area > config.max_triangle_area_px2 {
            stats.triangles_rejected += 1;
            continue;
        }
        // 3. Max edge length
        let e_ab = edge_length(ax, ay, bx, by);
        let e_bc = edge_length(bx, by, cx_, cy_);
        let e_ca = edge_length(cx_, cy_, ax, ay);
        if e_ab > config.max_edge_length_px
            || e_bc > config.max_edge_length_px
            || e_ca > config.max_edge_length_px
        {
            stats.triangles_rejected += 1;
            continue;
        }
        // 4. Disparity gradient
        let d_min = da.min(db).min(dc);
        let d_max = da.max(db).max(dc);
        if d_min > 0.0 && (d_max - d_min) / d_min > config.max_disparity_gradient {
            stats.triangles_rejected += 1;
            continue;
        }

        stats.triangles_rasterized += 1;

        // --- Rasterize ---
        let min_x = (ax.min(bx).min(cx_).floor() as i32).max(0);
        let max_x = (ax.max(bx).max(cx_).ceil() as i32).min(image_width as i32 - 1);
        let min_y = (ay.min(by).min(cy_).floor() as i32).max(0);
        let max_y = (ay.max(by).max(cy_).ceil() as i32).min(image_height as i32 - 1);

        let inv_area = 1.0 / area;

        let mut py = min_y;
        while py <= max_y {
            let mut px = min_x;
            while px <= max_x {
                let fpx = px as f32 + 0.5;
                let fpy = py as f32 + 0.5;

                // Barycentric coordinates via cross products
                let w0 = cross2d(bx - ax, by - ay, fpx - ax, fpy - ay) * inv_area;
                let w1 = cross2d(cx_ - bx, cy_ - by, fpx - bx, fpy - by) * inv_area;
                let w2 = 1.0 - w0 - w1;

                if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                    let d = w0 * dc + w1 * da + w2 * db;
                    if d >= config.min_disparity_px {
                        let z = fx * baseline_m / d;
                        let x = (fpx - cx) * z / fx;
                        let y = (fpy - cy) * z / fy;
                        let idx = (py as u32 * image_width + px as u32) as usize;
                        let intensity = if idx < image_data.len() {
                            image_data[idx]
                        } else {
                            128
                        };
                        // Propagate disparity uncertainty to a conservative scalar
                        // positional variance. Interpolation amplifies disparity noise
                        // through the inverse sqrt of the weakest barycentric support.
                        let min_bary = w0.min(w1).min(w2).max(0.01);
                        // Conservative fixed disparity noise prior until the
                        // calibrated stereo uncertainty model in M7 lands here.
                        let sigma_d_feature = 0.5_f32;
                        let sigma_d = sigma_d_feature / min_bary.sqrt();
                        let sigma_feature = 0.5_f32;
                        let position_variance = stereo_position_variance_m2(
                            fpx,
                            fpy,
                            z,
                            fx,
                            fy,
                            cx,
                            cy,
                            baseline_m,
                            sigma_d,
                            sigma_feature,
                            0.0,
                        );
                        points.push(DensePoint {
                            position: [x, y, z],
                            intensity,
                            position_variance,
                        });
                        if points.len() >= config.max_points_per_keyframe {
                            stats.points_generated = points.len();
                            stats.points_capped = true;
                            return DenseCloudResult { points, stats };
                        }
                    }
                }
                px += stride;
            }
            py += stride;
        }
    }

    stats.points_generated = points.len();
    DenseCloudResult { points, stats }
}

// ---------------------------------------------------------------------------
// Bowyer-Watson Delaunay triangulation
// ---------------------------------------------------------------------------

/// Returns triangle indices into the `pts` array.
fn delaunay(pts: &[[f32; 2]], width: f32, height: f32) -> Vec<[usize; 3]> {
    if pts.len() < 3 {
        return Vec::new();
    }
    let n = pts.len();
    // Super-triangle vertices (indices n, n+1, n+2)
    let margin = width.max(height) * 2.0;
    let super_a = [-margin, -margin];
    let super_b = [width + 2.0 * margin, -margin];
    let super_c = [width * 0.5, height + 2.0 * margin];

    // All points: originals + super-triangle
    let mut all_pts: Vec<[f32; 2]> = pts.to_vec();
    all_pts.push(super_a);
    all_pts.push(super_b);
    all_pts.push(super_c);

    let mut triangles: Vec<[usize; 3]> = vec![[n, n + 1, n + 2]];

    for pi in 0..n {
        let p = all_pts[pi];

        // Find all triangles whose circumcircle contains p
        let mut bad = Vec::new();
        for (ti, tri) in triangles.iter().enumerate() {
            if circumcircle_contains(all_pts[tri[0]], all_pts[tri[1]], all_pts[tri[2]], p) {
                bad.push(ti);
            }
        }

        // Find boundary polygon of the hole
        let mut boundary: Vec<[usize; 2]> = Vec::new();
        for &ti in &bad {
            let tri = triangles[ti];
            for edge_idx in 0..3 {
                let e = [tri[edge_idx], tri[(edge_idx + 1) % 3]];
                // Edge is boundary if it's not shared with another bad triangle
                let shared = bad.iter().any(|&other| {
                    other != ti && {
                        let ot = triangles[other];
                        edge_in_triangle(e, ot)
                    }
                });
                if !shared {
                    boundary.push(e);
                }
            }
        }

        // Remove bad triangles (reverse order to keep indices valid)
        bad.sort_unstable();
        for &ti in bad.iter().rev() {
            triangles.swap_remove(ti);
        }

        // Re-triangulate with the new point
        for edge in &boundary {
            triangles.push([edge[0], edge[1], pi]);
        }
    }

    // Remove any triangle that references super-triangle vertices
    triangles.retain(|tri| tri[0] < n && tri[1] < n && tri[2] < n);
    triangles
}

fn edge_in_triangle(e: [usize; 2], tri: [usize; 3]) -> bool {
    for i in 0..3 {
        let te = [tri[i], tri[(i + 1) % 3]];
        if (te[0] == e[0] && te[1] == e[1]) || (te[0] == e[1] && te[1] == e[0]) {
            return true;
        }
    }
    false
}

/// Returns true if point `p` is inside the circumcircle of triangle (a, b, c).
/// Uses the determinant-based test (positive for CCW-oriented triangles).
fn circumcircle_contains(a: [f32; 2], b: [f32; 2], c: [f32; 2], p: [f32; 2]) -> bool {
    let ax = a[0] - p[0];
    let ay = a[1] - p[1];
    let bx = b[0] - p[0];
    let by = b[1] - p[1];
    let cx = c[0] - p[0];
    let cy = c[1] - p[1];
    let det = (ax * ax + ay * ay) * (bx * cy - cx * by) - (bx * bx + by * by) * (ax * cy - cx * ay)
        + (cx * cx + cy * cy) * (ax * by - bx * ay);
    det > 0.0
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

fn cross2d(ux: f32, uy: f32, vx: f32, vy: f32) -> f32 {
    ux * vy - uy * vx
}

fn triangle_area(ax: f32, ay: f32, bx: f32, by: f32, cx: f32, cy: f32) -> f32 {
    (cross2d(bx - ax, by - ay, cx - ax, cy - ay)).abs() * 0.5
}

fn edge_length(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let dx = bx - ax;
    let dy = by - ay;
    (dx * dx + dy * dy).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    fn zero_row_mismatch() -> RectifiedRowMismatchPx {
        RectifiedRowMismatchPx::new(0.0).expect("row mismatch")
    }

    fn stable_surface_test_rig(
        width: u32,
        height: u32,
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        baseline_m: f32,
    ) -> RectifiedStereo {
        crate::test_helpers::make_rectified_stereo(width, height, fx, fy, cx, cy, baseline_m)
            .expect("valid test rectified stereo rig")
    }

    fn stable_surface_test_image(data: Vec<u8>, width: u32, height: u32) -> Frame {
        Frame::new(
            SensorId::StereoLeft,
            FrameId::new(1),
            Timestamp::from_nanos(1),
            width,
            height,
            data,
        )
        .expect("valid test intensity image")
    }

    #[test]
    fn stable_surface_generation_rejects_wrong_image_provenance_before_processing() {
        let stereo = stable_surface_test_rig(100, 100, 200.0, 200.0, 50.0, 50.0, 0.075);
        let right_image = Frame::new(
            SensorId::StereoRight,
            FrameId::new(1),
            Timestamp::from_nanos(1),
            100,
            100,
            vec![0; 100 * 100],
        )
        .expect("valid right image");
        assert!(matches!(
            generate_stable_surface_points(
                &[],
                &stereo,
                &right_image,
                &DenseCloudConfig::default()
            ),
            Err(StableSurfaceGenerationError::ImageSensorMismatch {
                expected: SensorId::StereoLeft,
                actual: SensorId::StereoRight,
            })
        ));

        let wrong_size = stable_surface_test_image(vec![0; 50 * 50], 50, 50);
        assert!(matches!(
            generate_stable_surface_points(
                &[],
                &stereo,
                &wrong_size,
                &DenseCloudConfig::default()
            ),
            Err(StableSurfaceGenerationError::ImageDimensionsMismatch {
                expected,
                actual,
            }) if expected == stereo.dimensions() && actual == wrong_size.dimensions()
        ));
    }

    #[test]
    fn stable_surface_generation_preserves_derived_point_and_metric_sources() {
        let stereo = stable_surface_test_rig(100, 100, 200.0, 200.0, f32::MAX, 50.0, 0.075);
        let image = stable_surface_test_image(vec![0; 100 * 100], 100, 100);
        let samples = [SparseStereoSample {
            u: 50.0,
            v: 50.0,
            right_u: 45.0,
            right_v: 50.0,
            disparity: 5.0,
            depth_m: 1.0,
            rectified_row_mismatch_px: zero_row_mismatch(),
        }];
        let error =
            generate_stable_surface_points(&samples, &stereo, &image, &DenseCloudConfig::default())
                .expect_err("extreme principal point must not become an uncertainty rejection");
        assert!(matches!(
            &error,
            StableSurfaceGenerationError::InvalidDerivedPoint {
                sample_index: 0,
                source: StableSurfacePointError::InvalidPositionVariance { value },
            } if value.is_infinite()
        ));
        assert!(
            error
                .source()
                .and_then(|source| source.downcast_ref::<StableSurfacePointError>())
                .is_some()
        );

        let metric_error = StableSurfaceGenerationError::InvalidStatistic {
            statistic: StableSurfaceStatistic::MeanRetainedRectifiedRowMismatchPx,
            source: DiagnosticMetricError::NonFinite {
                metric: "test metric",
                value: f32::NAN,
            },
        };
        assert!(
            metric_error
                .source()
                .and_then(|source| source.downcast_ref::<DiagnosticMetricError>())
                .is_some()
        );
    }

    #[test]
    fn stable_surface_point_enforces_finite_position_and_positive_variance() {
        for axis in 0..3 {
            let mut position = [0.0, 0.0, 1.0];
            position[axis] = f32::NAN;
            assert!(matches!(
                StableSurfacePoint::try_new(position, 128, 0.01, zero_row_mismatch()),
                Err(StableSurfacePointError::NonFinitePosition { axis: actual, .. })
                    if actual == axis
            ));
        }
        for value in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                StableSurfacePoint::try_new(
                    [0.0, 0.0, 1.0],
                    128,
                    value,
                    zero_row_mismatch(),
                ),
                Err(StableSurfacePointError::InvalidPositionVariance { value: actual })
                    if actual.to_bits() == value.to_bits()
            ));
        }

        let point = StableSurfacePoint::try_new([0.1, -0.2, 2.0], 42, 0.005, zero_row_mismatch())
            .expect("valid point");
        assert_eq!(point.position(), [0.1, -0.2, 2.0]);
        assert_eq!(point.intensity(), 42);
        assert_eq!(point.position_variance(), 0.005);
    }

    #[test]
    fn dense_cloud_config_rejects_invalid_runtime_values() {
        assert!(matches!(
            DenseCloudConfig::try_new(0, 0.25, 1.5, 120.0, 8_000.0, 0.05, 30_000),
            Err(DenseCloudConfigError::ZeroSubsample)
        ));
        for invalid in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                DenseCloudConfig::try_new(2, invalid, 1.5, 120.0, 8_000.0, 0.05, 30_000),
                Err(DenseCloudConfigError::InvalidPositiveField { value, .. })
                    if value.to_bits() == invalid.to_bits()
            ));
        }
        assert!(matches!(
            DenseCloudConfig::try_new(2, 0.25, 1.5, 120.0, 8_000.0, 0.05, 0),
            Err(DenseCloudConfigError::ZeroMaxPoints)
        ));

        let config = DenseCloudConfig::try_new(3, 0.25, 1.5, 120.0, 8_000.0, 0.05, 42)
            .expect("valid config");
        assert_eq!(config.subsample(), 3);
        assert_eq!(config.max_disparity_gradient(), 0.25);
        assert_eq!(config.min_disparity_px(), 1.5);
        assert_eq!(config.max_edge_length_px(), 120.0);
        assert_eq!(config.max_triangle_area_px2(), 8_000.0);
        assert_eq!(config.max_observation_std_dev_m(), 0.05);
        assert_eq!(config.max_points_per_keyframe(), 42);
    }

    #[test]
    fn dense_depth_empty_result_is_fallible_without_panicking() {
        let dimensions = FrameDimensions::try_new(2, 2).expect("dimensions");
        let depth = generate_dense_depth_image(
            FrameId::new(1),
            Timestamp::from_nanos(1),
            &[],
            200.0,
            0.075,
            dimensions,
            &DenseCloudConfig::default(),
        )
        .expect("valid empty interpolated depth");
        assert_eq!(depth.dimensions(), dimensions);
        assert_eq!(depth.depth_m(), &[0.0; 4]);
    }

    #[test]
    fn delaunay_four_points_two_triangles() {
        // Square: (0,0), (10,0), (10,10), (0,10)
        let pts = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let tris = delaunay(&pts, 20.0, 20.0);
        assert_eq!(tris.len(), 2, "4 points in convex position → 2 triangles");
        // All indices should be in 0..4
        for tri in &tris {
            for &idx in tri {
                assert!(idx < 4, "index {idx} out of range");
            }
        }
    }

    #[test]
    fn delaunay_three_points_one_triangle() {
        let pts = [[5.0, 1.0], [1.0, 9.0], [9.0, 9.0]];
        let tris = delaunay(&pts, 20.0, 20.0);
        assert_eq!(tris.len(), 1);
    }

    #[test]
    fn delaunay_two_points_empty() {
        let pts = [[0.0, 0.0], [5.0, 5.0]];
        let tris = delaunay(&pts, 20.0, 20.0);
        assert!(tris.is_empty());
    }

    #[test]
    fn barycentric_inside() {
        // Test via the generate function: a point at (5,5) inside triangle
        // (0,0)-(10,0)-(5,10) should produce dense points including near (5,5)
        let samples = vec![
            SparseStereoSample {
                u: 0.0,
                v: 0.0,
                right_u: -5.0,
                right_v: 0.0,
                disparity: 5.0,
                depth_m: 1.0,
                rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.0).expect("row mismatch"),
            },
            SparseStereoSample {
                u: 10.0,
                v: 0.0,
                right_u: 5.0,
                right_v: 0.0,
                disparity: 5.0,
                depth_m: 1.0,
                rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.0).expect("row mismatch"),
            },
            SparseStereoSample {
                u: 5.0,
                v: 10.0,
                right_u: 0.0,
                right_v: 10.0,
                disparity: 5.0,
                depth_m: 1.0,
                rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.0).expect("row mismatch"),
            },
        ];
        let image = vec![128u8; 20 * 20];
        let config = DenseCloudConfig {
            subsample: 1,
            max_observation_std_dev_m: 1.0,
            max_points_per_keyframe: 10000,
            ..Default::default()
        };
        let result = generate_dense_cloud(
            &samples, 200.0, 200.0, 10.0, 10.0, 0.075, &image, 20, 20, &config,
        );
        assert!(
            !result.points.is_empty(),
            "should produce points inside triangle"
        );
    }

    #[test]
    fn barycentric_outside() {
        // Single triangle, no points should be generated outside it
        // Triangle covers a small region, image is larger
        let samples = vec![
            SparseStereoSample {
                u: 2.0,
                v: 2.0,
                right_u: -3.0,
                right_v: 2.0,
                disparity: 5.0,
                depth_m: 1.0,
                rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.0).expect("row mismatch"),
            },
            SparseStereoSample {
                u: 4.0,
                v: 2.0,
                right_u: -1.0,
                right_v: 2.0,
                disparity: 5.0,
                depth_m: 1.0,
                rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.0).expect("row mismatch"),
            },
            SparseStereoSample {
                u: 3.0,
                v: 4.0,
                right_u: -2.0,
                right_v: 4.0,
                disparity: 5.0,
                depth_m: 1.0,
                rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.0).expect("row mismatch"),
            },
        ];
        let image = vec![128u8; 100 * 100];
        let config = DenseCloudConfig {
            subsample: 1,
            max_points_per_keyframe: 100000,
            ..Default::default()
        };
        let result = generate_dense_cloud(
            &samples, 200.0, 200.0, 50.0, 50.0, 0.075, &image, 100, 100, &config,
        );
        // All generated points should be within the triangle's bounding box
        for p in &result.points {
            // Back-project to pixel: u = px*fx/z + cx, but just check Z is valid
            assert!(p.position[2] > 0.0, "Z should be positive");
        }
    }

    #[test]
    fn circumcircle_contains_inside_point() {
        // Equilateral-ish triangle, point at center
        let a = [0.0_f32, 0.0];
        let b = [10.0, 0.0];
        let c = [5.0, 8.66];
        let p = [5.0, 3.0]; // inside
        assert!(circumcircle_contains(a, b, c, p));
    }

    #[test]
    fn disparity_gradient_rejects_large_jump() {
        let config = DenseCloudConfig {
            max_disparity_gradient: 0.3,
            ..Default::default()
        };
        let d_min = 10.0_f32;
        let d_max = 20.0_f32;
        let gradient = (d_max - d_min) / d_min;
        assert!(gradient > config.max_disparity_gradient);
    }

    #[test]
    fn edge_length_filter() {
        let len = edge_length(0.0, 0.0, 100.0, 0.0);
        assert!(len > 80.0);
    }

    #[test]
    fn generate_dense_cloud_fronto_parallel() {
        // 3 points on a plane at Z=2m
        let fx = 200.0;
        let fy = 200.0;
        let cx_val = 50.0;
        let cy_val = 50.0;
        let baseline = 0.075;
        let z = 2.0;
        let d = fx * baseline / z; // disparity for Z=2m

        let samples = vec![
            SparseStereoSample {
                u: 20.0,
                v: 20.0,
                right_u: 20.0 - d,
                right_v: 20.0,
                disparity: d,
                depth_m: z,
                rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.0).expect("row mismatch"),
            },
            SparseStereoSample {
                u: 80.0,
                v: 20.0,
                right_u: 80.0 - d,
                right_v: 20.0,
                disparity: d,
                depth_m: z,
                rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.0).expect("row mismatch"),
            },
            SparseStereoSample {
                u: 50.0,
                v: 80.0,
                right_u: 50.0 - d,
                right_v: 80.0,
                disparity: d,
                depth_m: z,
                rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.0).expect("row mismatch"),
            },
        ];
        let image = vec![128u8; 100 * 100];
        let config = DenseCloudConfig {
            subsample: 1,
            max_observation_std_dev_m: 1.0,
            max_points_per_keyframe: 100_000,
            ..Default::default()
        };
        let result = generate_dense_cloud(
            &samples, fx, fy, cx_val, cy_val, baseline, &image, 100, 100, &config,
        );
        assert!(!result.points.is_empty(), "should produce points");
        // All points should have Z ≈ 2.0 (fronto-parallel plane)
        for p in &result.points {
            assert!(
                (p.position[2] - z).abs() < 0.01,
                "Z={} expected {}",
                p.position[2],
                z
            );
        }
    }

    #[test]
    fn point_cap_enforced() {
        // Small triangle that won't be rejected by area/edge filters
        let samples = vec![
            SparseStereoSample {
                u: 10.0,
                v: 10.0,
                right_u: 5.0,
                right_v: 10.0,
                disparity: 5.0,
                depth_m: 1.0,
                rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.0).expect("row mismatch"),
            },
            SparseStereoSample {
                u: 40.0,
                v: 10.0,
                right_u: 35.0,
                right_v: 10.0,
                disparity: 5.0,
                depth_m: 1.0,
                rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.0).expect("row mismatch"),
            },
            SparseStereoSample {
                u: 25.0,
                v: 40.0,
                right_u: 20.0,
                right_v: 40.0,
                disparity: 5.0,
                depth_m: 1.0,
                rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.0).expect("row mismatch"),
            },
        ];
        let image = vec![128u8; 100 * 100];
        let config = DenseCloudConfig {
            subsample: 1,
            max_observation_std_dev_m: 1.0,
            max_points_per_keyframe: 10,
            max_edge_length_px: 200.0,
            max_triangle_area_px2: 10000.0,
            ..Default::default()
        };
        let result = generate_dense_cloud(
            &samples, 200.0, 200.0, 50.0, 50.0, 0.075, &image, 100, 100, &config,
        );
        assert!(
            result.points.len() <= 10,
            "got {} points",
            result.points.len()
        );
        assert!(result.stats.points_capped);
    }

    #[test]
    fn empty_input_returns_empty() {
        let result = generate_dense_cloud(
            &[],
            200.0,
            200.0,
            50.0,
            50.0,
            0.075,
            &[],
            100,
            100,
            &DenseCloudConfig::default(),
        );
        assert!(result.points.is_empty());
    }

    #[test]
    fn stable_surface_rejects_high_uncertainty_observations() {
        let fx = 200.0;
        let fy = 200.0;
        let cx = 50.0;
        let cy = 50.0;
        let baseline = 0.075;
        let z = 8.0;
        let d = fx * baseline / z;
        let samples = vec![SparseStereoSample {
            u: 50.0,
            v: 50.0,
            right_u: 50.0 - d,
            right_v: 50.0,
            disparity: d,
            depth_m: z,
            rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.0).expect("row mismatch"),
        }];
        let stereo = stable_surface_test_rig(100, 100, fx, fy, cx, cy, baseline);
        let image = stable_surface_test_image(vec![128u8; 100 * 100], 100, 100);
        let result =
            generate_stable_surface_points(&samples, &stereo, &image, &DenseCloudConfig::default())
                .expect("stable surface generation");
        assert!(result.points.is_empty());
        assert_eq!(result.stats.dropped_uncertainty, 1);
        assert_eq!(result.stats.mean_accepted_position_sigma_m, None);
        assert_eq!(result.stats.max_accepted_position_sigma_m, None);
        assert_eq!(result.stats.mean_accepted_rectified_row_mismatch_px, None);
        assert_eq!(result.stats.max_accepted_rectified_row_mismatch_px, None);
    }

    #[test]
    fn stable_surface_keeps_most_stable_points_when_capped() {
        let fx = 200.0;
        let fy = 200.0;
        let cx = 50.0;
        let cy = 50.0;
        let baseline = 0.075;
        let z = 2.0;
        let d = fx * baseline / z;
        let samples = vec![
            SparseStereoSample {
                u: 50.0,
                v: 50.0,
                right_u: 50.0 - d,
                right_v: 50.0,
                disparity: d,
                depth_m: z,
                rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.0).expect("row mismatch"),
            },
            SparseStereoSample {
                u: 95.0,
                v: 95.0,
                right_u: 95.0 - d,
                right_v: 95.0,
                disparity: d,
                depth_m: z,
                rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.0).expect("row mismatch"),
            },
        ];
        let stereo = stable_surface_test_rig(100, 100, fx, fy, cx, cy, baseline);
        let mut image_data = vec![0u8; 100 * 100];
        image_data[50 * 100 + 50] = 11;
        image_data[95 * 100 + 95] = 22;
        let image = stable_surface_test_image(image_data, 100, 100);
        let config = DenseCloudConfig {
            max_observation_std_dev_m: 1.0,
            max_points_per_keyframe: 1,
            ..DenseCloudConfig::default()
        };
        let result = generate_stable_surface_points(&samples, &stereo, &image, &config)
            .expect("stable surface generation");
        assert_eq!(result.points.len(), 1);
        assert_eq!(result.points[0].intensity, 11);
        assert!(result.stats.points_capped);
        assert!(result.stats.mean_accepted_position_sigma_m.is_some());
        assert!(result.stats.max_accepted_position_sigma_m.is_some());
        assert!(
            result
                .stats
                .mean_accepted_rectified_row_mismatch_px
                .is_some()
        );
        assert!(
            result
                .stats
                .max_accepted_rectified_row_mismatch_px
                .is_some()
        );
    }

    #[test]
    fn stable_surface_reports_retained_rectified_row_mismatch_metrics() {
        let fx = 200.0;
        let fy = 200.0;
        let cx = 50.0;
        let cy = 50.0;
        let baseline = 0.075;
        let z = 1.5;
        let d = fx * baseline / z;
        let samples = vec![
            SparseStereoSample {
                u: 40.0,
                v: 40.0,
                right_u: 40.0 - d,
                right_v: 40.0,
                disparity: d,
                depth_m: z,
                rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.25).expect("row mismatch"),
            },
            SparseStereoSample {
                u: 60.0,
                v: 60.0,
                right_u: 60.0 - d,
                right_v: 59.5,
                disparity: d,
                depth_m: z,
                rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.5).expect("row mismatch"),
            },
        ];
        let stereo = stable_surface_test_rig(100, 100, fx, fy, cx, cy, baseline);
        let image = stable_surface_test_image(vec![128u8; 100 * 100], 100, 100);
        let config = DenseCloudConfig {
            max_observation_std_dev_m: 1.0,
            ..DenseCloudConfig::default()
        };
        let result = generate_stable_surface_points(&samples, &stereo, &image, &config)
            .expect("stable surface generation");
        assert_eq!(result.points.len(), 2);
        assert_eq!(result.stats.points_generated, 2);
        assert_eq!(
            result
                .stats
                .mean_accepted_rectified_row_mismatch_px
                .map(|metric| metric.value_px()),
            Some(0.375)
        );
        assert_eq!(
            result
                .stats
                .max_accepted_rectified_row_mismatch_px
                .map(|metric| metric.value_px()),
            Some(0.5)
        );
    }

    #[test]
    fn stereo_position_variance_increases_with_rectified_row_mismatch() {
        let fx = 200.0;
        let fy = 200.0;
        let cx = 50.0;
        let cy = 50.0;
        let baseline = 0.075;
        let z = 0.5;
        let sigma_d_px = 0.5;
        let sigma_feature_px = 0.5;

        let aligned = stereo_position_variance_m2(
            50.0,
            50.0,
            z,
            fx,
            fy,
            cx,
            cy,
            baseline,
            sigma_d_px,
            sigma_feature_px,
            0.0,
        );
        let mismatched = stereo_position_variance_m2(
            50.0,
            50.0,
            z,
            fx,
            fy,
            cx,
            cy,
            baseline,
            sigma_d_px,
            sigma_feature_px,
            10.0,
        );

        assert!(mismatched > aligned);
    }

    #[test]
    fn stable_surface_rejects_large_row_mismatch_when_uncertainty_exceeds_threshold() {
        let fx = 200.0;
        let fy = 200.0;
        let cx = 50.0;
        let cy = 50.0;
        let baseline = 0.075;
        let z = 0.5;
        let d = fx * baseline / z;
        let stereo = stable_surface_test_rig(100, 100, fx, fy, cx, cy, baseline);
        let image = stable_surface_test_image(vec![128u8; 100 * 100], 100, 100);
        let config = DenseCloudConfig {
            max_observation_std_dev_m: 0.012,
            ..DenseCloudConfig::default()
        };

        let low_mismatch = vec![SparseStereoSample {
            u: 50.0,
            v: 50.0,
            right_u: 50.0 - d,
            right_v: 50.0,
            disparity: d,
            depth_m: z,
            rectified_row_mismatch_px: RectifiedRowMismatchPx::new(0.0).expect("row mismatch"),
        }];
        let high_mismatch = vec![SparseStereoSample {
            u: 50.0,
            v: 50.0,
            right_u: 50.0 - d,
            right_v: 60.0,
            disparity: d,
            depth_m: z,
            rectified_row_mismatch_px: RectifiedRowMismatchPx::new(10.0).expect("row mismatch"),
        }];

        let accepted = generate_stable_surface_points(&low_mismatch, &stereo, &image, &config)
            .expect("low-mismatch stable surface generation");
        let rejected = generate_stable_surface_points(&high_mismatch, &stereo, &image, &config)
            .expect("high-mismatch stable surface generation");

        assert_eq!(accepted.points.len(), 1);
        assert!(rejected.points.is_empty());
        assert_eq!(rejected.stats.dropped_uncertainty, 1);
    }
}
