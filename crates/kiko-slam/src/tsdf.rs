//! TSDF voxel grid with nvblox-style block hashing.
//!
//! Each voxel stores a truncated signed distance and observation weight.
//! Multiple observations of the same surface merge via weighted averaging,
//! which naturally filters noise: a mature voxel (weight=20) is barely
//! moved by a single outlier.

use std::collections::HashMap;

use crate::dense_cloud::DensePoint;
use crate::math;
use crate::Pose;

/// Voxel size in meters.
const DEFAULT_VOXEL_SIZE: f32 = 0.02; // 2cm
/// Block is 8³ voxels.
const BLOCK_SIDE: usize = 8;
const BLOCK_VOLUME: usize = BLOCK_SIDE * BLOCK_SIDE * BLOCK_SIDE; // 512
/// Truncation distance = 3 × voxel_size.
const TRUNCATION_FACTOR: f32 = 3.0;
/// Maximum weight per voxel. After this many observations, new data
/// has minimal influence — this is the noise suppression mechanism.
const MAX_WEIGHT: f32 = 20.0;
/// Minimum weight to consider a voxel as "observed" for surface extraction.
const MIN_SURFACE_WEIGHT: f32 = 2.0;
/// Surface is where TSDF crosses zero. Extract voxels with |distance| < threshold.
const SURFACE_DISTANCE_THRESHOLD_FACTOR: f32 = 0.75; // × voxel_size

// ---------------------------------------------------------------------------
// Core types — parse-don't-validate, illegal states irrepresentable
// ---------------------------------------------------------------------------

/// A single TSDF voxel. Default state is unobserved (weight = 0).
#[derive(Clone, Copy, Debug)]
struct TsdfVoxel {
    /// Signed distance to nearest surface (positive = outside, negative = inside).
    distance: f32,
    /// Observation count / confidence. Zero = unobserved.
    weight: f32,
    /// Grayscale intensity from the observation.
    color: u8,
}

impl Default for TsdfVoxel {
    fn default() -> Self {
        Self {
            distance: 0.0,
            weight: 0.0,
            color: 0,
        }
    }
}

impl TsdfVoxel {
    /// Returns true if this voxel has been observed at least once.
    fn is_observed(&self) -> bool {
        self.weight > 0.0
    }

    /// Integrate a new distance observation using weighted running average.
    fn integrate(&mut self, new_distance: f32, new_color: u8, max_weight: f32) {
        if self.weight == 0.0 {
            self.distance = new_distance;
            self.weight = 1.0;
            self.color = new_color;
        } else {
            let w = self.weight;
            self.distance = (self.distance * w + new_distance) / (w + 1.0);
            self.color = ((self.color as f32 * w + new_color as f32) / (w + 1.0)) as u8;
            self.weight = (w + 1.0).min(max_weight);
        }
    }
}

/// An 8³ block of voxels. Heap-allocated to avoid stack overflow.
/// This is the allocation unit — individual voxels are never allocated alone.
struct VoxelBlock {
    voxels: [TsdfVoxel; BLOCK_VOLUME],
}

impl VoxelBlock {
    fn new() -> Box<Self> {
        Box::new(Self {
            voxels: [TsdfVoxel::default(); BLOCK_VOLUME],
        })
    }

    /// Linear index from (x, y, z) within the block. Each in [0, 8).
    fn index(x: usize, y: usize, z: usize) -> usize {
        debug_assert!(x < BLOCK_SIDE && y < BLOCK_SIDE && z < BLOCK_SIDE);
        x + y * BLOCK_SIDE + z * BLOCK_SIDE * BLOCK_SIDE
    }

    fn get(&self, x: usize, y: usize, z: usize) -> &TsdfVoxel {
        &self.voxels[Self::index(x, y, z)]
    }

    fn get_mut(&mut self, x: usize, y: usize, z: usize) -> &mut TsdfVoxel {
        &mut self.voxels[Self::index(x, y, z)]
    }
}

/// Block index in the spatial hash. Computed as floor(position / block_edge).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct BlockIndex {
    x: i32,
    y: i32,
    z: i32,
}

impl BlockIndex {
    fn from_point(px: f32, py: f32, pz: f32, block_edge: f32) -> Self {
        Self {
            x: (px / block_edge).floor() as i32,
            y: (py / block_edge).floor() as i32,
            z: (pz / block_edge).floor() as i32,
        }
    }
}

/// Configuration for the TSDF layer.
#[derive(Clone, Copy, Debug)]
pub struct TsdfConfig {
    pub voxel_size: f32,
    pub max_weight: f32,
    pub truncation_distance: f32,
}

impl Default for TsdfConfig {
    fn default() -> Self {
        let voxel_size = DEFAULT_VOXEL_SIZE;
        Self {
            voxel_size,
            max_weight: MAX_WEIGHT,
            truncation_distance: voxel_size * TRUNCATION_FACTOR,
        }
    }
}

// ---------------------------------------------------------------------------
// TSDF Layer — the spatial hash of voxel blocks
// ---------------------------------------------------------------------------

/// A TSDF voxel layer using spatial hashing of 8³ blocks.
pub struct TsdfLayer {
    blocks: HashMap<BlockIndex, Box<VoxelBlock>>,
    config: TsdfConfig,
    block_edge: f32, // voxel_size * BLOCK_SIDE
}

impl TsdfLayer {
    pub fn new(config: TsdfConfig) -> Self {
        Self {
            blocks: HashMap::new(),
            block_edge: config.voxel_size * BLOCK_SIDE as f32,
            config,
        }
    }

    /// Number of allocated blocks.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Integrate a batch of dense points (in world/map frame) into the TSDF.
    ///
    /// Each point is treated as a surface observation. The voxel at the
    /// point's position gets distance ≈ 0, and nearby voxels along the
    /// viewing ray get positive/negative distances.
    ///
    /// For simplicity (and speed), we use projective TSDF: each point
    /// updates only the single voxel it falls into, with distance = 0
    /// (it IS the surface). This is equivalent to a point-based TSDF
    /// with zero truncation on the surface.
    pub fn integrate_points(&mut self, points: &[DensePoint], cam_from_map: Pose) {
        let map_from_cam = cam_from_map.inverse();
        let r = map_from_cam.rotation();
        let t = map_from_cam.translation();
        let cam_pos_world = t; // camera origin in world frame
        let voxel_size = self.config.voxel_size;
        let trunc = self.config.truncation_distance;
        let max_w = self.config.max_weight;
        let block_edge = self.block_edge;

        for point in points {
            // Transform point from camera to world frame
            let p_world = math::transform_point(r, t, point.position);

            // Viewing ray direction (world frame): from camera to point
            let ray = [
                p_world[0] - cam_pos_world[0],
                p_world[1] - cam_pos_world[1],
                p_world[2] - cam_pos_world[2],
            ];
            let ray_len = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();
            if ray_len < 1e-6 {
                continue;
            }
            let ray_dir = [ray[0] / ray_len, ray[1] / ray_len, ray[2] / ray_len];

            // Update voxels along the ray near the surface.
            // Sample from (point - trunc) to (point + trunc) along ray.
            let n_steps = ((2.0 * trunc / voxel_size).ceil() as usize).max(1);
            let step = 2.0 * trunc / n_steps as f32;

            for i in 0..=n_steps {
                let offset = -trunc + i as f32 * step;
                let vx = p_world[0] + ray_dir[0] * offset;
                let vy = p_world[1] + ray_dir[1] * offset;
                let vz = p_world[2] + ray_dir[2] * offset;

                // Signed distance: positive = in front of surface, negative = behind
                let sdf = -offset; // distance from this voxel to surface along ray

                // Block and local voxel indices
                let bi = BlockIndex::from_point(vx, vy, vz, block_edge);
                let lx = ((vx / voxel_size).floor() as i32).rem_euclid(BLOCK_SIDE as i32) as usize;
                let ly = ((vy / voxel_size).floor() as i32).rem_euclid(BLOCK_SIDE as i32) as usize;
                let lz = ((vz / voxel_size).floor() as i32).rem_euclid(BLOCK_SIDE as i32) as usize;

                let block = self.blocks.entry(bi).or_insert_with(VoxelBlock::new);
                block.get_mut(lx, ly, lz).integrate(sdf, point.intensity, max_w);
            }
        }
    }

    /// Extract surface points: voxels near the zero-crossing with sufficient weight.
    pub fn extract_surface(&self) -> Vec<([f32; 3], u8)> {
        let threshold = self.config.voxel_size * SURFACE_DISTANCE_THRESHOLD_FACTOR;
        let mut surface = Vec::new();

        for (bi, block) in &self.blocks {
            for lz in 0..BLOCK_SIDE {
                for ly in 0..BLOCK_SIDE {
                    for lx in 0..BLOCK_SIDE {
                        let voxel = block.get(lx, ly, lz);
                        if voxel.weight < MIN_SURFACE_WEIGHT {
                            continue;
                        }
                        if voxel.distance.abs() > threshold {
                            continue;
                        }
                        // Reconstruct world position from block + local index
                        let gx = bi.x as f32 * self.block_edge
                            + (lx as f32 + 0.5) * self.config.voxel_size;
                        let gy = bi.y as f32 * self.block_edge
                            + (ly as f32 + 0.5) * self.config.voxel_size;
                        let gz = bi.z as f32 * self.block_edge
                            + (lz as f32 + 0.5) * self.config.voxel_size;
                        surface.push(([gx, gy, gz], voxel.color));
                    }
                }
            }
        }
        surface
    }
}

// ---------------------------------------------------------------------------
// Async TSDF worker
// ---------------------------------------------------------------------------

use crossbeam_channel::{Receiver, Sender, TrySendError};

/// Message sent to the TSDF worker.
pub struct TsdfIntegrateMsg {
    pub points: Vec<DensePoint>,
    pub cam_from_map: Pose,
}

/// The TSDF worker handle. Send integrate messages, receive surface updates.
pub struct TsdfWorker {
    tx: Sender<TsdfIntegrateMsg>,
    surface_rx: Receiver<Vec<([f32; 3], u8)>>,
}

impl TsdfWorker {
    /// Spawn the TSDF worker thread. Returns the handle.
    pub fn spawn(config: TsdfConfig, queue_depth: usize) -> Self {
        let (tx, rx) = crossbeam_channel::bounded::<TsdfIntegrateMsg>(queue_depth);
        let (surface_tx, surface_rx) = crossbeam_channel::bounded::<Vec<([f32; 3], u8)>>(2);

        std::thread::Builder::new()
            .name("kiko-tsdf".into())
            .spawn(move || {
                let mut layer = TsdfLayer::new(config);
                let mut frame_count = 0u64;
                while let Ok(msg) = rx.recv() {
                    layer.integrate_points(&msg.points, msg.cam_from_map);
                    frame_count += 1;
                    // Extract surface every 5 keyframes to avoid over-logging
                    if frame_count % 5 == 0 {
                        let surface = layer.extract_surface();
                        let _ = surface_tx.try_send(surface);
                    }
                }
            })
            .expect("failed to spawn tsdf worker");

        Self { tx, surface_rx }
    }

    /// Try to send an integrate message. Returns false if queue is full.
    pub fn try_integrate(&self, msg: TsdfIntegrateMsg) -> bool {
        match self.tx.try_send(msg) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) => false,
            Err(TrySendError::Disconnected(_)) => false,
        }
    }

    /// Try to receive the latest surface extraction.
    pub fn try_recv_surface(&self) -> Option<Vec<([f32; 3], u8)>> {
        self.surface_rx.try_recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voxel_integrate_converges() {
        let mut v = TsdfVoxel::default();
        assert!(!v.is_observed());
        v.integrate(0.01, 128, 20.0);
        assert!(v.is_observed());
        assert!((v.distance - 0.01).abs() < 1e-6);

        // Integrate 19 more at distance 0.0 — should converge near 0
        for _ in 0..19 {
            v.integrate(0.0, 128, 20.0);
        }
        assert!(v.distance.abs() < 0.001, "d={}", v.distance);
        assert!((v.weight - 20.0).abs() < 1e-6);
    }

    #[test]
    fn voxel_noise_rejection() {
        let mut v = TsdfVoxel::default();
        // 20 good observations at d=0
        for _ in 0..20 {
            v.integrate(0.0, 128, 20.0);
        }
        // One noisy outlier at d=0.1
        v.integrate(0.1, 128, 20.0);
        // Should barely move (weight capped at 20)
        assert!(v.distance.abs() < 0.006, "d={} — noise leaked", v.distance);
    }

    #[test]
    fn block_index_roundtrip() {
        let block_edge = 0.02 * 8.0; // 0.16m
        let bi = BlockIndex::from_point(0.5, -0.3, 1.2, block_edge);
        assert_eq!(bi.x, 3);  // 0.5 / 0.16 = 3.125 → 3
        assert_eq!(bi.y, -2); // -0.3 / 0.16 = -1.875 → -2
        assert_eq!(bi.z, 7);  // 1.2 / 0.16 = 7.5 → 7
    }

    #[test]
    fn integrate_single_point_creates_block() {
        let mut layer = TsdfLayer::new(TsdfConfig::default());
        assert_eq!(layer.num_blocks(), 0);
        let points = vec![DensePoint {
            position: [0.0, 0.0, 1.0],
            intensity: 200,
        }];
        let pose = Pose::identity();
        layer.integrate_points(&points, pose);
        assert!(layer.num_blocks() > 0);
    }

    #[test]
    fn surface_extraction_finds_observed_voxels() {
        let mut layer = TsdfLayer::new(TsdfConfig::default());
        let pose = Pose::identity();
        // Integrate same point many times to build weight
        let points = vec![DensePoint {
            position: [0.5, 0.5, 2.0],
            intensity: 180,
        }];
        for _ in 0..5 {
            layer.integrate_points(&points, pose);
        }
        let surface = layer.extract_surface();
        assert!(!surface.is_empty(), "should find surface voxels");
    }

    #[test]
    fn empty_layer_has_no_surface() {
        let layer = TsdfLayer::new(TsdfConfig::default());
        assert!(layer.extract_surface().is_empty());
    }
}
