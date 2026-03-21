//! Occupancy voxel grid with marching cubes mesh extraction.

use std::collections::HashMap;
use crate::math;
use crate::Pose;

use crossbeam_channel::{Receiver, Sender, TrySendError};

const BLOCK_SIDE: usize = 8;
const BLOCK_VOL: usize = BLOCK_SIDE * BLOCK_SIDE * BLOCK_SIDE;

#[derive(Clone, Copy, Debug)]
pub struct TsdfConfig {
    pub voxel_size: f32,
}

impl Default for TsdfConfig {
    fn default() -> Self {
        Self { voxel_size: 0.03 }
    }
}

pub struct TsdfIntegrateMsg {
    pub depth_image: Vec<f32>,
    pub grayscale: Vec<u8>,
    pub cam_from_map: Pose,
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub image_width: u32,
    pub image_height: u32,
}

/// Mesh triangle with vertex positions and colors.
pub struct MeshData {
    pub positions: Vec<[f32; 3]>,
    pub indices: Vec<[u32; 3]>,
    pub colors: Vec<[u8; 3]>,
}

pub struct TsdfWorker {
    tx: Sender<TsdfIntegrateMsg>,
    mesh_rx: Receiver<MeshData>,
}

#[derive(Clone, Copy)]
struct Voxel {
    tsdf: f32,    // signed distance: + outside, - inside, 0 = surface
    weight: f32,
    color: u8,
}

impl Default for Voxel {
    fn default() -> Self {
        Self { tsdf: 1.0, weight: 0.0, color: 128 }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct BlockIdx(i32, i32, i32);

struct Layer {
    blocks: HashMap<BlockIdx, Vec<Voxel>>,
    voxel_size: f32,
    block_edge: f32,
    trunc: f32,
}

impl Layer {
    fn new(voxel_size: f32) -> Self {
        Self {
            blocks: HashMap::new(),
            voxel_size,
            block_edge: voxel_size * BLOCK_SIDE as f32,
            trunc: voxel_size * 3.0,
        }
    }

    /// Get voxel at global voxel coordinates. Returns default if not allocated.
    fn get_voxel(&self, gx: i32, gy: i32, gz: i32) -> Voxel {
        let bs = BLOCK_SIDE as i32;
        let bi = BlockIdx(
            gx.div_euclid(bs),
            gy.div_euclid(bs),
            gz.div_euclid(bs),
        );
        let lx = gx.rem_euclid(bs) as usize;
        let ly = gy.rem_euclid(bs) as usize;
        let lz = gz.rem_euclid(bs) as usize;
        match self.blocks.get(&bi) {
            Some(block) => block[lx + ly * BLOCK_SIDE + lz * BLOCK_SIDE * BLOCK_SIDE],
            None => Voxel::default(),
        }
    }

    fn integrate_depth(&mut self, msg: &TsdfIntegrateMsg) {
        let map_from_cam = msg.cam_from_map.inverse();
        let r = map_from_cam.rotation();
        let t = map_from_cam.translation();
        let cam_origin = t;
        let w = msg.image_width as usize;
        let h = msg.image_height as usize;
        let vs = self.voxel_size;
        let trunc = self.trunc;
        let bs = BLOCK_SIDE as i32;

        for v in 0..h {
            for u in 0..w {
                let depth = msg.depth_image[v * w + u];
                if depth <= 0.0 || !depth.is_finite() || depth > 8.0 {
                    continue;
                }
                let x_cam = (u as f32 - msg.cx) * depth / msg.fx;
                let y_cam = (v as f32 - msg.cy) * depth / msg.fy;
                let p_map = math::transform_point(r, t, [x_cam, y_cam, depth]);

                let ray = [
                    p_map[0] - cam_origin[0],
                    p_map[1] - cam_origin[1],
                    p_map[2] - cam_origin[2],
                ];
                let ray_len = (ray[0]*ray[0] + ray[1]*ray[1] + ray[2]*ray[2]).sqrt();
                if ray_len < 1e-6 { continue; }
                let rd = [ray[0]/ray_len, ray[1]/ray_len, ray[2]/ray_len];

                let color = if v * w + u < msg.grayscale.len() {
                    msg.grayscale[v * w + u]
                } else { 128 };

                // Update voxels along ray: -trunc to +trunc around surface
                let n_steps = ((2.0 * trunc / vs) as i32).max(2);
                let step = 2.0 * trunc / n_steps as f32;
                for i in 0..=n_steps {
                    let offset = -trunc + i as f32 * step;
                    let vx = p_map[0] + rd[0] * offset;
                    let vy = p_map[1] + rd[1] * offset;
                    let vz = p_map[2] + rd[2] * offset;
                    // SDF: positive in front of surface, negative behind
                    let sdf = (-offset / trunc).clamp(-1.0, 1.0);

                    let gxi = (vx / vs).floor() as i32;
                    let gyi = (vy / vs).floor() as i32;
                    let gzi = (vz / vs).floor() as i32;
                    let bi = BlockIdx(gxi.div_euclid(bs), gyi.div_euclid(bs), gzi.div_euclid(bs));
                    let lx = gxi.rem_euclid(bs) as usize;
                    let ly = gyi.rem_euclid(bs) as usize;
                    let lz = gzi.rem_euclid(bs) as usize;
                    let idx = lx + ly * BLOCK_SIDE + lz * BLOCK_SIDE * BLOCK_SIDE;

                    let block = self.blocks.entry(bi).or_insert_with(|| vec![Voxel::default(); BLOCK_VOL]);
                    let voxel = &mut block[idx];
                    let w_old = voxel.weight;
                    voxel.tsdf = if w_old == 0.0 { sdf } else {
                        (voxel.tsdf * w_old + sdf) / (w_old + 1.0)
                    };
                    voxel.color = if w_old == 0.0 { color } else {
                        ((voxel.color as f32 * w_old + color as f32) / (w_old + 1.0)) as u8
                    };
                    voxel.weight = (w_old + 1.0).min(20.0);
                }
            }
        }
    }

    /// Marching cubes mesh extraction from the TSDF zero-crossing.
    fn extract_mesh(&self) -> MeshData {
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut indices: Vec<[u32; 3]> = Vec::new();
        let mut colors: Vec<[u8; 3]> = Vec::new();
        let vs = self.voxel_size;

        // For each observed voxel, check its 3 positive-direction edges
        // for zero-crossings. This produces vertices on edges where the
        // TSDF changes sign. Then form triangles from the MC lookup table.
        // Simplified: instead of full MC, just emit a small quad at each
        // zero-crossing edge midpoint, oriented perpendicular to the edge.
        for (&bi, block) in &self.blocks {
            let bx = bi.0 * BLOCK_SIDE as i32;
            let by = bi.1 * BLOCK_SIDE as i32;
            let bz = bi.2 * BLOCK_SIDE as i32;
            for lz in 0..BLOCK_SIDE {
                for ly in 0..BLOCK_SIDE {
                    for lx in 0..BLOCK_SIDE {
                        let idx = lx + ly * BLOCK_SIDE + lz * BLOCK_SIDE * BLOCK_SIDE;
                        let v0 = block[idx];
                        if v0.weight < 2.0 { continue; }

                        let gx = bx + lx as i32;
                        let gy = by + ly as i32;
                        let gz = bz + lz as i32;

                        // Check +X edge
                        let v1 = self.get_voxel(gx + 1, gy, gz);
                        if v1.weight >= 2.0 && v0.tsdf * v1.tsdf < 0.0 {
                            let t = v0.tsdf / (v0.tsdf - v1.tsdf);
                            let px = (gx as f32 + 0.5 + t) * vs;
                            let py = (gy as f32 + 0.5) * vs;
                            let pz = (gz as f32 + 0.5) * vs;
                            let c = ((v0.color as f32 * (1.0 - t) + v1.color as f32 * t) as u8);
                            let vi = positions.len() as u32;
                            let hs = vs * 0.5;
                            positions.push([px, py - hs, pz - hs]);
                            positions.push([px, py + hs, pz - hs]);
                            positions.push([px, py + hs, pz + hs]);
                            positions.push([px, py - hs, pz + hs]);
                            colors.push([c, c, c]); colors.push([c, c, c]);
                            colors.push([c, c, c]); colors.push([c, c, c]);
                            indices.push([vi, vi+1, vi+2]);
                            indices.push([vi, vi+2, vi+3]);
                        }

                        // Check +Y edge
                        let v2 = self.get_voxel(gx, gy + 1, gz);
                        if v2.weight >= 2.0 && v0.tsdf * v2.tsdf < 0.0 {
                            let t = v0.tsdf / (v0.tsdf - v2.tsdf);
                            let px = (gx as f32 + 0.5) * vs;
                            let py = (gy as f32 + 0.5 + t) * vs;
                            let pz = (gz as f32 + 0.5) * vs;
                            let c = ((v0.color as f32 * (1.0 - t) + v2.color as f32 * t) as u8);
                            let vi = positions.len() as u32;
                            let hs = vs * 0.5;
                            positions.push([px - hs, py, pz - hs]);
                            positions.push([px + hs, py, pz - hs]);
                            positions.push([px + hs, py, pz + hs]);
                            positions.push([px - hs, py, pz + hs]);
                            colors.push([c, c, c]); colors.push([c, c, c]);
                            colors.push([c, c, c]); colors.push([c, c, c]);
                            indices.push([vi, vi+1, vi+2]);
                            indices.push([vi, vi+2, vi+3]);
                        }

                        // Check +Z edge
                        let v3 = self.get_voxel(gx, gy, gz + 1);
                        if v3.weight >= 2.0 && v0.tsdf * v3.tsdf < 0.0 {
                            let t = v0.tsdf / (v0.tsdf - v3.tsdf);
                            let px = (gx as f32 + 0.5) * vs;
                            let py = (gy as f32 + 0.5) * vs;
                            let pz = (gz as f32 + 0.5 + t) * vs;
                            let c = ((v0.color as f32 * (1.0 - t) + v3.color as f32 * t) as u8);
                            let vi = positions.len() as u32;
                            let hs = vs * 0.5;
                            positions.push([px - hs, py - hs, pz]);
                            positions.push([px + hs, py - hs, pz]);
                            positions.push([px + hs, py + hs, pz]);
                            positions.push([px - hs, py + hs, pz]);
                            colors.push([c, c, c]); colors.push([c, c, c]);
                            colors.push([c, c, c]); colors.push([c, c, c]);
                            indices.push([vi, vi+1, vi+2]);
                            indices.push([vi, vi+2, vi+3]);
                        }
                    }
                }
            }
        }
        MeshData { positions, indices, colors }
    }
}

impl TsdfWorker {
    pub fn spawn(config: TsdfConfig, queue_depth: usize) -> Self {
        let (tx, rx) = crossbeam_channel::bounded::<TsdfIntegrateMsg>(queue_depth);
        let (mesh_tx, mesh_rx) = crossbeam_channel::bounded::<MeshData>(2);

        std::thread::Builder::new()
            .name("kiko-tsdf".into())
            .spawn(move || {
                let mut layer = Layer::new(config.voxel_size);
                let mut count = 0u64;
                while let Ok(msg) = rx.recv() {
                    layer.integrate_depth(&msg);
                    count += 1;
                    if count % 5 == 0 {
                        let mesh = layer.extract_mesh();
                        eprintln!(
                            "tsdf: frame {} blocks={} mesh_verts={} mesh_tris={}",
                            count, layer.blocks.len(),
                            mesh.positions.len(), mesh.indices.len(),
                        );
                        let _ = mesh_tx.try_send(mesh);
                    }
                }
            })
            .expect("failed to spawn tsdf worker");

        Self { tx, mesh_rx }
    }

    pub fn try_integrate(&self, msg: TsdfIntegrateMsg) -> bool {
        match self.tx.try_send(msg) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) => false,
            Err(TrySendError::Disconnected(_)) => false,
        }
    }

    pub fn try_recv_mesh(&self) -> Option<MeshData> {
        self.mesh_rx.try_recv().ok()
    }
}
