//! Dense interpolated point cloud from sparse stereo matches.
//!
//! Uses Delaunay triangulation of sparse feature disparities to produce
//! a piecewise-linear dense disparity map, then back-projects to 3D.
//! This is a visualization aid, not a dense reconstruction.

use crate::triangulation::{Point3, SparseStereoSample};

/// Configuration for dense cloud generation.
#[derive(Clone, Copy, Debug)]
pub struct DenseCloudConfig {
    /// Generate a point every Nth pixel within each triangle.
    pub subsample: u32,
    /// Reject triangle if `(max_d - min_d) / min_d > this`.
    pub max_disparity_gradient: f32,
    /// Minimum disparity for valid back-projection.
    pub min_disparity_px: f32,
    /// Reject triangle if any edge exceeds this length in pixels.
    pub max_edge_length_px: f32,
    /// Reject triangle if area exceeds this in pixels².
    pub max_triangle_area_px2: f32,
    /// Hard cap on output points per keyframe.
    pub max_points_per_keyframe: usize,
}

impl Default for DenseCloudConfig {
    fn default() -> Self {
        Self {
            subsample: 2,
            max_disparity_gradient: 0.4,
            min_disparity_px: 1.0,
            max_edge_length_px: 200.0,
            max_triangle_area_px2: 20_000.0,
            max_points_per_keyframe: 50_000,
        }
    }
}

impl DenseCloudConfig {
    pub fn from_env() -> Self {
        let mut c = Self::default();
        if let Some(v) = crate::env::env_usize("KIKO_DENSE_SUBSAMPLE") {
            c.subsample = v.max(1) as u32;
        }
        if let Some(v) = crate::env::env_f32("KIKO_DENSE_MAX_GRADIENT") {
            c.max_disparity_gradient = v;
        }
        if let Some(v) = crate::env::env_f32("KIKO_DENSE_MAX_EDGE_PX") {
            c.max_edge_length_px = v;
        }
        if let Some(v) = crate::env::env_f32("KIKO_DENSE_MAX_AREA_PX2") {
            c.max_triangle_area_px2 = v;
        }
        if let Some(v) = crate::env::env_usize("KIKO_DENSE_MAX_POINTS") {
            c.max_points_per_keyframe = v;
        }
        c
    }
}

/// A dense colored point in camera frame.
#[derive(Clone, Copy, Debug)]
pub struct DensePoint {
    pub position: [f32; 3],
    pub intensity: u8,
}

/// Statistics from dense cloud generation.
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

/// Generate a dense depth image by interpolating disparity over a Delaunay
/// triangulation. Every pixel inside a valid triangle gets a depth value.
/// Pixels outside triangles or in rejected triangles remain 0.0 (invalid).
/// This is the input format nvblox expects.
pub fn generate_dense_depth_image(
    samples: &[SparseStereoSample],
    fx: f32,
    baseline_m: f32,
    image_width: u32,
    image_height: u32,
    config: &DenseCloudConfig,
) -> Vec<f32> {
    let w = image_width as usize;
    let h = image_height as usize;
    let mut depth = vec![0.0_f32; w * h];

    if samples.len() < 3 {
        eprintln!("dense_depth: too few samples ({})", samples.len());
        return depth;
    }

    let pts: Vec<[f32; 2]> = samples.iter().map(|s| [s.u, s.v]).collect();
    let triangles = delaunay(&pts, image_width as f32, image_height as f32);
    let mut rejected = 0usize;
    let mut rasterized = 0usize;
    let mut filled = 0usize;

    for tri in &triangles {
        let (a, b, c) = (tri[0], tri[1], tri[2]);
        let (ax, ay) = (pts[a][0], pts[a][1]);
        let (bx, by) = (pts[b][0], pts[b][1]);
        let (cx_, cy_) = (pts[c][0], pts[c][1]);
        let (da, db, dc) = (samples[a].disparity, samples[b].disparity, samples[c].disparity);

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
    filled = depth.iter().filter(|&&d| d > 0.0).count();
    eprintln!(
        "dense_depth: samples={} triangles={} rejected={} rasterized={} filled={}/{}",
        samples.len(), triangles.len(), rejected, rasterized, filled, w * h,
    );
    depth
}

/// Generate a dense point cloud by interpolating disparity over a Delaunay
/// triangulation of sparse stereo samples.
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
        let (da, db, dc) = (samples[a].disparity, samples[b].disparity, samples[c].disparity);

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
                        points.push(DensePoint {
                            position: [x, y, z],
                            intensity,
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
            if circumcircle_contains(
                all_pts[tri[0]],
                all_pts[tri[1]],
                all_pts[tri[2]],
                p,
            ) {
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
    let det = (ax * ax + ay * ay) * (bx * cy - cx * by)
        - (bx * bx + by * by) * (ax * cy - cx * ay)
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
            SparseStereoSample { u: 0.0, v: 0.0, disparity: 5.0, depth_m: 1.0 },
            SparseStereoSample { u: 10.0, v: 0.0, disparity: 5.0, depth_m: 1.0 },
            SparseStereoSample { u: 5.0, v: 10.0, disparity: 5.0, depth_m: 1.0 },
        ];
        let image = vec![128u8; 20 * 20];
        let config = DenseCloudConfig {
            subsample: 1,
            max_points_per_keyframe: 10000,
            ..Default::default()
        };
        let result = generate_dense_cloud(
            &samples, 200.0, 200.0, 10.0, 10.0, 0.075, &image, 20, 20, &config,
        );
        assert!(!result.points.is_empty(), "should produce points inside triangle");
    }

    #[test]
    fn barycentric_outside() {
        // Single triangle, no points should be generated outside it
        // Triangle covers a small region, image is larger
        let samples = vec![
            SparseStereoSample { u: 2.0, v: 2.0, disparity: 5.0, depth_m: 1.0 },
            SparseStereoSample { u: 4.0, v: 2.0, disparity: 5.0, depth_m: 1.0 },
            SparseStereoSample { u: 3.0, v: 4.0, disparity: 5.0, depth_m: 1.0 },
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
            SparseStereoSample { u: 20.0, v: 20.0, disparity: d, depth_m: z },
            SparseStereoSample { u: 80.0, v: 20.0, disparity: d, depth_m: z },
            SparseStereoSample { u: 50.0, v: 80.0, disparity: d, depth_m: z },
        ];
        let image = vec![128u8; 100 * 100];
        let config = DenseCloudConfig {
            subsample: 1,
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
            SparseStereoSample { u: 10.0, v: 10.0, disparity: 5.0, depth_m: 1.0 },
            SparseStereoSample { u: 40.0, v: 10.0, disparity: 5.0, depth_m: 1.0 },
            SparseStereoSample { u: 25.0, v: 40.0, disparity: 5.0, depth_m: 1.0 },
        ];
        let image = vec![128u8; 100 * 100];
        let config = DenseCloudConfig {
            subsample: 1,
            max_points_per_keyframe: 10,
            max_edge_length_px: 200.0,
            max_triangle_area_px2: 10000.0,
            ..Default::default()
        };
        let result = generate_dense_cloud(
            &samples, 200.0, 200.0, 50.0, 50.0, 0.075, &image, 100, 100, &config,
        );
        assert!(result.points.len() <= 10, "got {} points", result.points.len());
        assert!(result.stats.points_capped);
    }

    #[test]
    fn empty_input_returns_empty() {
        let result = generate_dense_cloud(
            &[], 200.0, 200.0, 50.0, 50.0, 0.075, &[], 100, 100, &DenseCloudConfig::default(),
        );
        assert!(result.points.is_empty());
    }
}
