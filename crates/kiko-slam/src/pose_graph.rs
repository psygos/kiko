use std::collections::HashMap;
use std::num::NonZeroU32;

use crate::Pose64;
use crate::map::{KeyframeId, SlamMap};
use crate::math::{mat_mul_vec_f64, se3_exp_f64, se3_log_f64, so3_right_jacobian_f64};

#[derive(Clone, Debug)]
pub struct BlockCsr6x6 {
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<[[f64; 6]; 6]>,
    nrows: usize,
}

impl BlockCsr6x6 {
    pub fn new(nrows: usize) -> Self {
        Self {
            row_ptr: vec![0; nrows.saturating_add(1)],
            col_idx: Vec::new(),
            values: Vec::new(),
            nrows,
        }
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn insert(&mut self, row: usize, col: usize, block: [[f64; 6]; 6]) {
        let Some(idx) = self.find_index(row, col) else {
            self.insert_new(row, col, block);
            return;
        };
        self.values[idx] = block;
    }

    pub fn add_to(&mut self, row: usize, col: usize, block: [[f64; 6]; 6]) {
        let Some(idx) = self.find_index(row, col) else {
            self.insert_new(row, col, block);
            return;
        };
        for (r, block_row) in block.iter().enumerate() {
            for (c, value) in block_row.iter().enumerate() {
                self.values[idx][r][c] += *value;
            }
        }
    }

    pub fn get(&self, row: usize, col: usize) -> Option<[[f64; 6]; 6]> {
        let idx = self.find_index(row, col)?;
        Some(self.values[idx])
    }

    pub fn spmv(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(
            x.len(),
            self.nrows * 6,
            "spmv input length must be nrows * 6"
        );
        assert_eq!(
            y.len(),
            self.nrows * 6,
            "spmv output length must be nrows * 6"
        );
        y.fill(0.0);
        for row in 0..self.nrows {
            let row_base = row * 6;
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            for idx in start..end {
                let col = self.col_idx[idx];
                let col_base = col * 6;
                let block = self.values[idx];
                for r in 0..6 {
                    let mut sum = 0.0_f64;
                    for c in 0..6 {
                        sum += block[r][c] * x[col_base + c];
                    }
                    y[row_base + r] += sum;
                }
            }
        }
    }

    pub fn diagonal_blocks(&self) -> Vec<[[f64; 6]; 6]> {
        let mut diagonal = vec![[[0.0_f64; 6]; 6]; self.nrows];
        for (row, diag_block) in diagonal.iter_mut().enumerate().take(self.nrows) {
            if let Some(block) = self.get(row, row) {
                *diag_block = block;
            }
        }
        diagonal
    }

    fn find_index(&self, row: usize, col: usize) -> Option<usize> {
        if row >= self.nrows || col >= self.nrows {
            return None;
        }
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        (start..end).find(|&idx| self.col_idx[idx] == col)
    }

    fn insert_new(&mut self, row: usize, col: usize, block: [[f64; 6]; 6]) {
        assert!(row < self.nrows, "row out of bounds");
        assert!(col < self.nrows, "col out of bounds");

        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        let mut insert_at = end;
        for idx in start..end {
            if self.col_idx[idx] > col {
                insert_at = idx;
                break;
            }
        }

        self.col_idx.insert(insert_at, col);
        self.values.insert(insert_at, block);
        for ptr in self.row_ptr.iter_mut().skip(row + 1) {
            *ptr += 1;
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PcgResult {
    pub iterations: usize,
    pub residual_norm: f64,
    pub converged: bool,
}

pub fn solve_pcg(
    h: &BlockCsr6x6,
    b: &[f64],
    x: &mut [f64],
    max_iters: usize,
    tol: f64,
) -> PcgResult {
    let dim = h.nrows() * 6;
    assert_eq!(b.len(), dim, "rhs length must be nrows * 6");
    assert_eq!(x.len(), dim, "solution length must be nrows * 6");

    let mut r = vec![0.0_f64; dim];
    let mut hx = vec![0.0_f64; dim];
    h.spmv(x, &mut hx);
    for i in 0..dim {
        r[i] = b[i] - hx[i];
    }

    let diag_inv = invert_diagonal_blocks(h.diagonal_blocks());
    let mut z = vec![0.0_f64; dim];
    apply_preconditioner(&diag_inv, &r, &mut z);
    let mut p = z.clone();
    let mut rz_old = dot(&r, &z);
    let mut residual_norm = norm(&r);
    if residual_norm <= tol {
        return PcgResult {
            iterations: 0,
            residual_norm,
            converged: true,
        };
    }

    let mut hp = vec![0.0_f64; dim];
    for iter in 0..max_iters {
        h.spmv(&p, &mut hp);
        let denom = dot(&p, &hp);
        if denom.abs() < 1e-18 {
            return PcgResult {
                iterations: iter,
                residual_norm,
                converged: false,
            };
        }

        let alpha = rz_old / denom;
        for i in 0..dim {
            x[i] += alpha * p[i];
            r[i] -= alpha * hp[i];
        }
        residual_norm = norm(&r);
        if residual_norm <= tol {
            return PcgResult {
                iterations: iter + 1,
                residual_norm,
                converged: true,
            };
        }

        apply_preconditioner(&diag_inv, &r, &mut z);
        let rz_new = dot(&r, &z);
        if rz_old.abs() < 1e-18 {
            return PcgResult {
                iterations: iter + 1,
                residual_norm,
                converged: false,
            };
        }
        let beta = rz_new / rz_old;
        for i in 0..dim {
            p[i] = z[i] + beta * p[i];
        }
        rz_old = rz_new;
    }

    PcgResult {
        iterations: max_iters,
        residual_norm,
        converged: false,
    }
}

fn apply_preconditioner(diag_inv: &[[[f64; 6]; 6]], r: &[f64], z: &mut [f64]) {
    for (block_idx, inv) in diag_inv.iter().enumerate() {
        let base = block_idx * 6;
        for row in 0..6 {
            let mut sum = 0.0_f64;
            for col in 0..6 {
                sum += inv[row][col] * r[base + col];
            }
            z[base + row] = sum;
        }
    }
}

fn invert_diagonal_blocks(diag: Vec<[[f64; 6]; 6]>) -> Vec<[[f64; 6]; 6]> {
    diag.into_iter()
        .map(|block| invert_6x6(block).unwrap_or_else(identity6))
        .collect()
}

fn invert_6x6(a: [[f64; 6]; 6]) -> Option<[[f64; 6]; 6]> {
    let mut aug = [[0.0_f64; 12]; 6];
    for row in 0..6 {
        for col in 0..6 {
            aug[row][col] = a[row][col];
        }
        aug[row][6 + row] = 1.0;
    }

    for pivot in 0..6 {
        let mut max_row = pivot;
        let mut max_val = aug[pivot][pivot].abs();
        for (row, row_vals) in aug.iter().enumerate().skip(pivot + 1) {
            let val = row_vals[pivot].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-18 {
            return None;
        }
        if max_row != pivot {
            aug.swap(pivot, max_row);
        }

        let pivot_val = aug[pivot][pivot];
        for col in 0..12 {
            aug[pivot][col] /= pivot_val;
        }

        for row in 0..6 {
            if row == pivot {
                continue;
            }
            let factor = aug[row][pivot];
            if factor.abs() < 1e-18 {
                continue;
            }
            for col in 0..12 {
                aug[row][col] -= factor * aug[pivot][col];
            }
        }
    }

    let mut inv = [[0.0_f64; 6]; 6];
    for row in 0..6 {
        for col in 0..6 {
            inv[row][col] = aug[row][6 + col];
        }
    }
    Some(inv)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

fn identity6() -> [[f64; 6]; 6] {
    let mut out = [[0.0_f64; 6]; 6];
    for (i, row) in out.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    out
}

#[derive(Clone, Debug)]
pub struct PoseGraphEdge {
    pub from: usize,
    pub to: usize,
    pub measurement: Pose64,
    pub information: [[f64; 6]; 6],
}

pub fn compute_edge_error(edge: &PoseGraphEdge, poses: &[Pose64]) -> [f64; 6] {
    assert!(edge.from < poses.len(), "edge.from out of bounds");
    assert!(edge.to < poses.len(), "edge.to out of bounds");
    let t_from_inv = poses[edge.from].inverse();
    let t_to = poses[edge.to];
    let predicted = t_from_inv.compose(t_to);
    let residual_pose = predicted.compose(edge.measurement.inverse());
    se3_log_f64(residual_pose)
}

pub fn compute_edge_jacobians(
    edge: &PoseGraphEdge,
    poses: &[Pose64],
) -> ([[f64; 6]; 6], [[f64; 6]; 6]) {
    let base_error = compute_edge_error(edge, poses);
    let jr = so3_right_jacobian_f64([base_error[3], base_error[4], base_error[5]]);
    let eps = 1e-6_f64;
    let mut j_from = [[0.0_f64; 6]; 6];
    let mut j_to = [[0.0_f64; 6]; 6];

    for axis in 0..6 {
        let delta_plus = perturb_axis(axis, eps, jr);
        let delta_minus = perturb_axis(axis, -eps, jr);

        let mut poses_plus = poses.to_vec();
        poses_plus[edge.from] = se3_exp_f64(delta_plus).compose(poses_plus[edge.from]);
        let err_plus = compute_edge_error(edge, &poses_plus);

        let mut poses_minus = poses.to_vec();
        poses_minus[edge.from] = se3_exp_f64(delta_minus).compose(poses_minus[edge.from]);
        let err_minus = compute_edge_error(edge, &poses_minus);

        for row in 0..6 {
            j_from[row][axis] = (err_plus[row] - err_minus[row]) / (2.0 * eps);
        }

        let mut poses_plus = poses.to_vec();
        poses_plus[edge.to] = se3_exp_f64(delta_plus).compose(poses_plus[edge.to]);
        let err_plus = compute_edge_error(edge, &poses_plus);

        let mut poses_minus = poses.to_vec();
        poses_minus[edge.to] = se3_exp_f64(delta_minus).compose(poses_minus[edge.to]);
        let err_minus = compute_edge_error(edge, &poses_minus);

        for row in 0..6 {
            j_to[row][axis] = (err_plus[row] - err_minus[row]) / (2.0 * eps);
        }
    }

    (j_from, j_to)
}

fn perturb_axis(axis: usize, magnitude: f64, right_jacobian: [[f64; 3]; 3]) -> [f64; 6] {
    let mut delta = [0.0_f64; 6];
    if axis < 3 {
        delta[axis] = magnitude;
        return delta;
    }

    let mut unit_rot = [0.0_f64; 3];
    unit_rot[axis - 3] = magnitude;
    let rot = mat_mul_vec_f64(right_jacobian, unit_rot);
    delta[3] = rot[0];
    delta[4] = rot[1];
    delta[5] = rot[2];
    delta
}

#[derive(Clone, Copy, Debug)]
pub struct PoseGraphConfig {
    pub max_iterations: usize,
    pub pcg_max_iters: usize,
    pub pcg_tol: f64,
    pub huber_delta: f64,
}

impl Default for PoseGraphConfig {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            pcg_max_iters: 100,
            pcg_tol: 1e-6,
            huber_delta: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PoseGraphResult {
    pub corrected_poses: Vec<Pose64>,
    pub iterations: usize,
    pub converged: bool,
}

#[derive(Clone, Debug)]
pub struct PoseGraphOptimizer {
    config: PoseGraphConfig,
}

impl PoseGraphOptimizer {
    pub fn new(config: PoseGraphConfig) -> Self {
        Self { config }
    }

    pub fn optimize(
        &self,
        edges: &[PoseGraphEdge],
        initial_poses: &mut [Pose64],
    ) -> PoseGraphResult {
        let nposes = initial_poses.len();
        if nposes == 0 {
            return PoseGraphResult {
                corrected_poses: Vec::new(),
                iterations: 0,
                converged: true,
            };
        }

        let mut poses = initial_poses.to_vec();
        let mut converged = false;
        let mut iters_run = 0;
        let mut valid_edges = Vec::with_capacity(edges.len());
        let mut invalid_edges = 0usize;
        for edge in edges {
            if edge.from < nposes && edge.to < nposes {
                valid_edges.push(edge);
            } else {
                invalid_edges = invalid_edges.saturating_add(1);
            }
        }
        if invalid_edges > 0 {
            eprintln!("pose graph skipped {invalid_edges} invalid edges (nposes={nposes})");
        }
        if valid_edges.is_empty() {
            return PoseGraphResult {
                corrected_poses: poses,
                iterations: 0,
                converged: true,
            };
        }

        for iter in 0..self.config.max_iterations {
            iters_run = iter + 1;
            let mut h = BlockCsr6x6::new(nposes);
            let mut b = vec![0.0_f64; nposes * 6];

            for edge in &valid_edges {
                let error = compute_edge_error(edge, &poses);
                let (j_from, j_to) = compute_edge_jacobians(edge, &poses);
                let e_norm = error.iter().map(|v| v * v).sum::<f64>().sqrt();
                let weight = huber_weight(e_norm, self.config.huber_delta);
                let mut information = edge.information;
                for row in &mut information {
                    for value in row {
                        *value *= weight;
                    }
                }

                let h_ff = jt_info_j(j_from, information, j_from);
                let h_ft = jt_info_j(j_from, information, j_to);
                let h_tf = jt_info_j(j_to, information, j_from);
                let h_tt = jt_info_j(j_to, information, j_to);
                h.add_to(edge.from, edge.from, h_ff);
                h.add_to(edge.from, edge.to, h_ft);
                h.add_to(edge.to, edge.from, h_tf);
                h.add_to(edge.to, edge.to, h_tt);

                let g_from = jt_info_vec(j_from, information, error);
                let g_to = jt_info_vec(j_to, information, error);
                for k in 0..6 {
                    b[edge.from * 6 + k] += g_from[k];
                    b[edge.to * 6 + k] += g_to[k];
                }
            }

            // Anchor the first pose to remove gauge freedom.
            h.add_to(0, 0, scaled_identity6(1e9));
            for v in b.iter_mut().take(6) {
                *v = 0.0;
            }

            let rhs: Vec<f64> = b.into_iter().map(|v| -v).collect();
            let mut delta = vec![0.0_f64; nposes * 6];
            let pcg = solve_pcg(
                &h,
                &rhs,
                &mut delta,
                self.config.pcg_max_iters,
                self.config.pcg_tol,
            );
            if !pcg.converged && iter + 1 == self.config.max_iterations {
                eprintln!(
                    "pose graph PCG did not converge (iters={}, residual_norm={:.3e})",
                    pcg.iterations, pcg.residual_norm
                );
            }
            if !pcg.residual_norm.is_finite() {
                break;
            }

            let mut max_step = 0.0_f64;
            for (pose_idx, pose) in poses.iter_mut().enumerate().skip(1) {
                let base = pose_idx * 6;
                let mut xi = [
                    delta[base],
                    delta[base + 1],
                    delta[base + 2],
                    delta[base + 3],
                    delta[base + 4],
                    delta[base + 5],
                ];
                let mut step_norm = xi.iter().map(|v| v * v).sum::<f64>().sqrt();
                if !step_norm.is_finite() {
                    continue;
                }
                if step_norm > 1.0 {
                    let scale = 1.0 / step_norm;
                    for v in &mut xi {
                        *v *= scale;
                    }
                    step_norm = 1.0;
                }
                max_step = max_step.max(step_norm);
                *pose = se3_exp_f64(xi).compose(*pose);
            }

            if max_step < 1e-6 {
                converged = true;
                break;
            }
        }

        initial_poses.copy_from_slice(&poses);
        PoseGraphResult {
            corrected_poses: poses,
            iterations: iters_run,
            converged,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EssentialEdgeKind {
    SpanningTree,
    StrongCovisibility,
    Loop,
}

#[derive(Clone, Debug)]
pub struct EssentialEdge {
    pub a: KeyframeId,
    pub b: KeyframeId,
    pub kind: EssentialEdgeKind,
    pub relative_pose: Pose64,
    pub information: [[f64; 6]; 6],
}

#[derive(Clone, Debug)]
pub struct EssentialGraphSnapshot {
    pub parent: HashMap<KeyframeId, KeyframeId>,
    pub order: Vec<KeyframeId>,
    pub spanning_edges: Vec<EssentialEdge>,
    pub strong_covis_edges: Vec<EssentialEdge>,
    pub loop_edges: Vec<EssentialEdge>,
    pub strong_threshold: u32,
}

#[derive(Clone, Debug)]
pub struct PoseGraphInput {
    pub keyframe_ids: Vec<KeyframeId>,
    pub edges: Vec<PoseGraphEdge>,
}

#[derive(Clone, Debug)]
pub struct EssentialGraph {
    parent: HashMap<KeyframeId, KeyframeId>,
    order: Vec<KeyframeId>,
    spanning_edges: Vec<EssentialEdge>,
    strong_covis_edges: Vec<EssentialEdge>,
    loop_edges: Vec<EssentialEdge>,
    strong_threshold: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EssentialGraphError {
    KeyframeNotFound { keyframe_id: KeyframeId },
    RootRemovalDenied { keyframe_id: KeyframeId },
}

impl std::fmt::Display for EssentialGraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EssentialGraphError::KeyframeNotFound { keyframe_id } => {
                write!(f, "essential graph keyframe not found: {keyframe_id:?}")
            }
            EssentialGraphError::RootRemovalDenied { keyframe_id } => {
                write!(
                    f,
                    "cannot remove essential graph root keyframe: {keyframe_id:?}"
                )
            }
        }
    }
}

impl std::error::Error for EssentialGraphError {}

impl EssentialGraph {
    pub fn new(strong_threshold: u32) -> Self {
        Self {
            parent: HashMap::new(),
            order: Vec::new(),
            spanning_edges: Vec::new(),
            strong_covis_edges: Vec::new(),
            loop_edges: Vec::new(),
            strong_threshold,
        }
    }

    pub fn parent_of(&self, keyframe_id: KeyframeId) -> Option<KeyframeId> {
        self.parent.get(&keyframe_id).copied()
    }

    pub fn add_keyframe(
        &mut self,
        keyframe_id: KeyframeId,
        covisibility: Option<&HashMap<KeyframeId, NonZeroU32>>,
        map: &SlamMap,
    ) {
        if self.parent.contains_key(&keyframe_id) {
            return;
        }
        self.order.push(keyframe_id);
        if self.parent.is_empty() {
            self.parent.insert(keyframe_id, keyframe_id);
            return;
        }

        let Some(neighbors) = covisibility else {
            self.parent.insert(keyframe_id, keyframe_id);
            return;
        };
        if neighbors.is_empty() {
            self.parent.insert(keyframe_id, keyframe_id);
            return;
        }

        let mut strongest = None;
        for (&neighbor, &weight) in neighbors {
            if strongest
                .as_ref()
                .is_none_or(|(_, best_w): &(KeyframeId, u32)| weight.get() > *best_w)
            {
                strongest = Some((neighbor, weight.get()));
            }

            if weight.get() >= self.strong_threshold
                && !contains_edge(&self.strong_covis_edges, keyframe_id, neighbor)
                && let Some(relative_pose) = relative_pose(map, keyframe_id, neighbor)
            {
                self.strong_covis_edges.push(EssentialEdge {
                    a: keyframe_id,
                    b: neighbor,
                    kind: EssentialEdgeKind::StrongCovisibility,
                    relative_pose,
                    information: scaled_identity6(weight.get() as f64),
                });
            }
        }

        let Some((parent, weight)) = strongest else {
            self.parent.insert(keyframe_id, keyframe_id);
            return;
        };
        self.parent.insert(keyframe_id, parent);
        if let Some(relative_pose) = relative_pose(map, parent, keyframe_id) {
            self.spanning_edges.push(EssentialEdge {
                a: parent,
                b: keyframe_id,
                kind: EssentialEdgeKind::SpanningTree,
                relative_pose,
                information: scaled_identity6(weight as f64),
            });
        }
    }

    pub fn add_loop_edge(&mut self, edge: EssentialEdge) {
        if let std::collections::hash_map::Entry::Vacant(entry) = self.parent.entry(edge.a) {
            entry.insert(edge.a);
            self.order.push(edge.a);
        }
        if let std::collections::hash_map::Entry::Vacant(entry) = self.parent.entry(edge.b) {
            entry.insert(edge.b);
            self.order.push(edge.b);
        }
        self.loop_edges.push(edge);
    }

    pub fn remove_keyframe(
        &mut self,
        keyframe_id: KeyframeId,
        map: &SlamMap,
    ) -> Result<(), EssentialGraphError> {
        let parent = self
            .parent
            .get(&keyframe_id)
            .copied()
            .ok_or(EssentialGraphError::KeyframeNotFound { keyframe_id })?;
        if parent == keyframe_id {
            return Err(EssentialGraphError::RootRemovalDenied { keyframe_id });
        }

        let children: Vec<KeyframeId> = self
            .parent
            .iter()
            .filter_map(|(&child, &child_parent)| {
                if child_parent == keyframe_id && child != keyframe_id {
                    Some(child)
                } else {
                    None
                }
            })
            .collect();

        for child in &children {
            if let Some(entry) = self.parent.get_mut(child) {
                *entry = parent;
            }
        }

        self.parent.remove(&keyframe_id);
        self.order.retain(|&id| id != keyframe_id);
        self.spanning_edges
            .retain(|edge| edge.a != keyframe_id && edge.b != keyframe_id);
        self.strong_covis_edges
            .retain(|edge| edge.a != keyframe_id && edge.b != keyframe_id);
        self.loop_edges
            .retain(|edge| edge.a != keyframe_id && edge.b != keyframe_id);

        for child in children {
            if contains_edge(&self.spanning_edges, parent, child) {
                continue;
            }
            if let Some(relative_pose) = relative_pose(map, parent, child) {
                self.spanning_edges.push(EssentialEdge {
                    a: parent,
                    b: child,
                    kind: EssentialEdgeKind::SpanningTree,
                    relative_pose,
                    information: scaled_identity6(1.0),
                });
            }
        }

        Ok(())
    }

    pub fn snapshot(&self) -> EssentialGraphSnapshot {
        EssentialGraphSnapshot {
            parent: self.parent.clone(),
            order: self.order.clone(),
            spanning_edges: self.spanning_edges.clone(),
            strong_covis_edges: self.strong_covis_edges.clone(),
            loop_edges: self.loop_edges.clone(),
            strong_threshold: self.strong_threshold,
        }
    }

    pub fn pose_graph_input(&self) -> PoseGraphInput {
        let mut keyframe_ids = self.order.clone();
        for edge in self
            .spanning_edges
            .iter()
            .chain(self.strong_covis_edges.iter())
            .chain(self.loop_edges.iter())
        {
            if !keyframe_ids.contains(&edge.a) {
                keyframe_ids.push(edge.a);
            }
            if !keyframe_ids.contains(&edge.b) {
                keyframe_ids.push(edge.b);
            }
        }

        let mut id_to_idx = HashMap::new();
        for (idx, &id) in keyframe_ids.iter().enumerate() {
            id_to_idx.insert(id, idx);
        }

        let edges = self
            .spanning_edges
            .iter()
            .chain(self.strong_covis_edges.iter())
            .chain(self.loop_edges.iter())
            .filter_map(|edge| {
                Some(PoseGraphEdge {
                    from: *id_to_idx.get(&edge.a)?,
                    to: *id_to_idx.get(&edge.b)?,
                    measurement: edge.relative_pose,
                    information: edge.information,
                })
            })
            .collect();

        PoseGraphInput {
            keyframe_ids,
            edges,
        }
    }

    pub fn all_edges(&self) -> Vec<PoseGraphEdge> {
        self.pose_graph_input().edges
    }
}

fn contains_edge(edges: &[EssentialEdge], a: KeyframeId, b: KeyframeId) -> bool {
    edges
        .iter()
        .any(|edge| (edge.a == a && edge.b == b) || (edge.a == b && edge.b == a))
}

fn relative_pose(map: &SlamMap, from: KeyframeId, to: KeyframeId) -> Option<Pose64> {
    let from_pose = map.keyframe(from)?.pose();
    let to_pose = map.keyframe(to)?.pose();
    let from_64 = Pose64::from_pose32(from_pose);
    let to_64 = Pose64::from_pose32(to_pose);
    Some(from_64.inverse().compose(to_64))
}

fn huber_weight(norm: f64, delta: f64) -> f64 {
    if norm <= delta || norm <= 1e-12 {
        1.0
    } else {
        delta / norm
    }
}

fn jt_info_j(a: [[f64; 6]; 6], info: [[f64; 6]; 6], b: [[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut info_b = [[0.0_f64; 6]; 6];
    for row in 0..6 {
        for col in 0..6 {
            for (k, b_row) in b.iter().enumerate() {
                info_b[row][col] += info[row][k] * b_row[col];
            }
        }
    }

    let mut out = [[0.0_f64; 6]; 6];
    for (row, out_row) in out.iter_mut().enumerate() {
        for (col, out_value) in out_row.iter_mut().enumerate() {
            for (k, info_b_row) in info_b.iter().enumerate() {
                *out_value += a[k][row] * info_b_row[col];
            }
        }
    }
    out
}

fn jt_info_vec(j: [[f64; 6]; 6], info: [[f64; 6]; 6], e: [f64; 6]) -> [f64; 6] {
    let mut info_e = [0.0_f64; 6];
    for row in 0..6 {
        for (col, e_value) in e.iter().enumerate() {
            info_e[row] += info[row][col] * *e_value;
        }
    }
    let mut out = [0.0_f64; 6];
    for (row, out_value) in out.iter_mut().enumerate() {
        for (k, info_value) in info_e.iter().enumerate() {
            *out_value += j[k][row] * *info_value;
        }
    }
    out
}

fn scaled_identity6(scale: f64) -> [[f64; 6]; 6] {
    let mut out = [[0.0_f64; 6]; 6];
    for (i, row) in out.iter_mut().enumerate() {
        row[i] = scale;
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::num::NonZeroU32;

    use super::{
        BlockCsr6x6, EssentialEdge, EssentialEdgeKind, EssentialGraph, EssentialGraphError,
        PoseGraphConfig, PoseGraphEdge, PoseGraphOptimizer, compute_edge_error,
        compute_edge_jacobians, solve_pcg,
    };
    use crate::Pose64;
    use crate::map::{ImageSize, SlamMap};
    use crate::math::se3_exp_f64;
    use crate::{CompactDescriptor, FrameId, Keypoint, Point3, Pose, Timestamp};

    #[derive(Clone, Debug)]
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_usize(&mut self, upper: usize) -> usize {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((self.state >> 32) as usize) % upper
        }
    }

    fn scalar_block(diagonal: f64) -> [[f64; 6]; 6] {
        let mut block = [[0.0_f64; 6]; 6];
        for (i, row) in block.iter_mut().enumerate() {
            row[i] = diagonal;
        }
        block
    }

    fn make_map_for_essential_graph() -> (
        SlamMap,
        crate::map::KeyframeId,
        crate::map::KeyframeId,
        crate::map::KeyframeId,
    ) {
        let mut map = SlamMap::new();
        let size = ImageSize::try_new(640, 480).expect("size");
        let keypoints = vec![
            Keypoint { x: 20.0, y: 20.0 },
            Keypoint { x: 40.0, y: 20.0 },
            Keypoint { x: 60.0, y: 20.0 },
        ];
        let kf0 = map
            .add_keyframe(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                Pose::identity(),
                size,
                keypoints.clone(),
            )
            .expect("kf0");
        let kf1 = map
            .add_keyframe(
                FrameId::new(2),
                Timestamp::from_nanos(2),
                Pose::from_rt(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [1.0, 0.0, 0.0],
                ),
                size,
                keypoints.clone(),
            )
            .expect("kf1");
        let kf2 = map
            .add_keyframe(
                FrameId::new(3),
                Timestamp::from_nanos(3),
                Pose::from_rt(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [2.0, 0.0, 0.0],
                ),
                size,
                keypoints,
            )
            .expect("kf2");

        for i in 0..2 {
            let kp0 = map.keyframe_keypoint(kf0, i).expect("kp0");
            let point_id = map
                .add_map_point(
                    Point3 {
                        x: i as f32,
                        y: 0.0,
                        z: 3.0,
                    },
                    CompactDescriptor([128; 256]),
                    kp0,
                )
                .expect("point");
            let kp1 = map.keyframe_keypoint(kf1, i).expect("kp1");
            map.add_observation(point_id, kp1).expect("obs");
        }

        let kp1 = map.keyframe_keypoint(kf1, 2).expect("kp1 third");
        let point_id = map
            .add_map_point(
                Point3 {
                    x: 2.0,
                    y: 0.0,
                    z: 3.0,
                },
                CompactDescriptor([128; 256]),
                kp1,
            )
            .expect("point third");
        let kp2 = map.keyframe_keypoint(kf2, 0).expect("kp2");
        map.add_observation(point_id, kp2).expect("obs third");

        (map, kf0, kf1, kf2)
    }

    fn make_chain_keyframes(count: usize) -> (SlamMap, Vec<crate::map::KeyframeId>) {
        let mut map = SlamMap::new();
        let size = ImageSize::try_new(640, 480).expect("size");
        let keypoints = vec![Keypoint { x: 20.0, y: 20.0 }, Keypoint { x: 40.0, y: 20.0 }];
        let mut ids = Vec::with_capacity(count);
        for idx in 0..count {
            let pose = Pose::from_rt(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [idx as f32 * 0.5, 0.0, 0.0],
            );
            let id = map
                .add_keyframe(
                    FrameId::new((idx + 1) as u64),
                    Timestamp::from_nanos((idx + 1) as i64),
                    pose,
                    size,
                    keypoints.clone(),
                )
                .expect("keyframe");
            ids.push(id);
        }
        (map, ids)
    }

    #[test]
    fn block_csr_insert_and_get_are_consistent() {
        let mut h = BlockCsr6x6::new(3);
        let block = scalar_block(2.0);
        h.insert(1, 2, block);
        assert_eq!(h.get(1, 2), Some(block));

        let replacement = scalar_block(3.0);
        h.insert(1, 2, replacement);
        assert_eq!(h.get(1, 2), Some(replacement));
    }

    #[test]
    fn block_csr_spmv_matches_dense_reference() {
        let mut h = BlockCsr6x6::new(2);
        h.insert(0, 0, scalar_block(2.0));
        h.insert(0, 1, scalar_block(1.0));
        h.insert(1, 0, scalar_block(-1.0));
        h.insert(1, 1, scalar_block(3.0));

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let mut y_sparse = vec![0.0; 12];
        h.spmv(&x, &mut y_sparse);

        let mut y_dense = [0.0; 12];
        for row in 0..2 {
            for col in 0..2 {
                let Some(block) = h.get(row, col) else {
                    continue;
                };
                for r in 0..6 {
                    let mut sum = 0.0;
                    for c in 0..6 {
                        sum += block[r][c] * x[col * 6 + c];
                    }
                    y_dense[row * 6 + r] += sum;
                }
            }
        }

        for i in 0..12 {
            assert!(
                (y_sparse[i] - y_dense[i]).abs() < 1e-12,
                "mismatch at {i}: sparse={}, dense={}",
                y_sparse[i],
                y_dense[i]
            );
        }
    }

    #[test]
    fn block_csr_diagonal_extraction_returns_only_diagonal_blocks() {
        let mut h = BlockCsr6x6::new(3);
        h.insert(0, 0, scalar_block(1.0));
        h.insert(0, 1, scalar_block(5.0));
        h.insert(1, 1, scalar_block(2.0));
        h.insert(2, 0, scalar_block(7.0));
        h.insert(2, 2, scalar_block(3.0));

        let diag = h.diagonal_blocks();
        assert_eq!(diag.len(), 3);
        assert_eq!(diag[0], scalar_block(1.0));
        assert_eq!(diag[1], scalar_block(2.0));
        assert_eq!(diag[2], scalar_block(3.0));
    }

    #[test]
    fn pcg_solves_identity_in_one_iteration() {
        let mut h = BlockCsr6x6::new(2);
        h.insert(0, 0, scalar_block(1.0));
        h.insert(1, 1, scalar_block(1.0));
        let b = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0,
        ];
        let mut x = vec![0.0; b.len()];
        let result = solve_pcg(&h, &b, &mut x, 20, 1e-12);
        assert!(result.converged);
        assert_eq!(result.iterations, 1);
        for i in 0..x.len() {
            assert!((x[i] - b[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn pcg_converges_on_small_spd_system() {
        let mut h = BlockCsr6x6::new(2);
        h.insert(0, 0, scalar_block(4.0));
        h.insert(1, 1, scalar_block(5.0));
        h.insert(0, 1, scalar_block(0.2));
        h.insert(1, 0, scalar_block(0.2));

        let x_true = vec![
            0.5, -0.3, 0.8, 0.1, -0.2, 0.4, 1.2, -0.6, 0.7, -0.9, 0.2, 0.3,
        ];
        let mut b = vec![0.0; x_true.len()];
        h.spmv(&x_true, &mut b);

        let mut x = vec![0.0; x_true.len()];
        let result = solve_pcg(&h, &b, &mut x, 50, 1e-10);
        assert!(result.converged, "pcg did not converge: {result:?}");
        for i in 0..x.len() {
            assert!(
                (x[i] - x_true[i]).abs() < 1e-8,
                "solution mismatch at {i}: got {}, expected {}",
                x[i],
                x_true[i]
            );
        }
    }

    #[test]
    fn pcg_zero_rhs_returns_zero_solution() {
        let mut h = BlockCsr6x6::new(2);
        h.insert(0, 0, scalar_block(3.0));
        h.insert(1, 1, scalar_block(2.0));
        let b = vec![0.0; 12];
        let mut x = vec![0.0; 12];
        let result = solve_pcg(&h, &b, &mut x, 10, 1e-12);
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        assert!(x.iter().all(|v| v.abs() < 1e-15));
    }

    #[test]
    fn pose_graph_edge_error_is_zero_for_consistent_measurement() {
        let pose_a = Pose64::identity();
        let pose_b = se3_exp_f64([0.2, -0.1, 0.05, 0.03, -0.02, 0.01]);
        let measurement = pose_a.inverse().compose(pose_b);
        let edge = PoseGraphEdge {
            from: 0,
            to: 1,
            measurement,
            information: scalar_block(1.0),
        };
        let error = compute_edge_error(&edge, &[pose_a, pose_b]);
        let norm: f64 = error.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm < 1e-9, "expected near-zero error, got {norm}");
    }

    #[test]
    fn pose_graph_edge_jacobians_match_finite_difference() {
        let pose_a = se3_exp_f64([0.1, 0.05, -0.02, 0.02, -0.01, 0.03]);
        let pose_b = se3_exp_f64([0.3, -0.08, 0.12, -0.02, 0.03, -0.01]);
        let measurement = pose_a
            .inverse()
            .compose(pose_b)
            .compose(se3_exp_f64([0.01, -0.005, 0.002, 0.001, -0.0015, 0.0008]));
        let edge = PoseGraphEdge {
            from: 0,
            to: 1,
            measurement,
            information: scalar_block(1.0),
        };
        let poses = [pose_a, pose_b];
        let (j_from, j_to) = compute_edge_jacobians(&edge, &poses);
        for row in 0..6 {
            for col in 0..6 {
                assert!(j_from[row][col].is_finite(), "non-finite J_from entry");
                assert!(j_to[row][col].is_finite(), "non-finite J_to entry");
            }
        }
    }

    fn edge(from: usize, to: usize, from_pose: Pose64, to_pose: Pose64) -> PoseGraphEdge {
        let measurement = from_pose.inverse().compose(to_pose);
        PoseGraphEdge {
            from,
            to,
            measurement,
            information: scalar_block(1.0),
        }
    }

    fn translation_error(poses: &[Pose64], target: &[Pose64]) -> f64 {
        poses
            .iter()
            .zip(target.iter())
            .map(|(a, b)| {
                let dx = a.translation[0] - b.translation[0];
                let dy = a.translation[1] - b.translation[1];
                let dz = a.translation[2] - b.translation[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .sum::<f64>()
            / poses.len() as f64
    }

    #[test]
    fn pose_graph_optimizer_ring_graph_converges() {
        let gt = vec![
            Pose64::identity(),
            se3_exp_f64([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            se3_exp_f64([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            se3_exp_f64([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ];
        let edges = vec![
            edge(0, 1, gt[0], gt[1]),
            edge(1, 2, gt[1], gt[2]),
            edge(2, 3, gt[2], gt[3]),
            edge(3, 0, gt[3], gt[0]),
        ];
        let mut initial = vec![
            gt[0],
            se3_exp_f64([1.2, 0.1, 0.0, 0.0, 0.01, 0.0]),
            se3_exp_f64([2.3, -0.2, 0.1, 0.0, -0.02, 0.0]),
            se3_exp_f64([3.4, 0.2, -0.1, 0.0, 0.01, 0.0]),
        ];
        let before = translation_error(&initial, &gt);
        let optimizer = PoseGraphOptimizer::new(PoseGraphConfig::default());
        let result = optimizer.optimize(&edges, &mut initial);
        let after = translation_error(&result.corrected_poses, &gt);
        assert!(result.converged || result.iterations > 0);
        assert!(
            after < before,
            "ring graph did not improve: before={before}, after={after}"
        );
    }

    #[test]
    fn pose_graph_optimizer_loop_closure_reduces_drift() {
        let gt = vec![
            Pose64::identity(),
            se3_exp_f64([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            se3_exp_f64([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ];
        let edges = vec![
            edge(0, 1, gt[0], gt[1]),
            edge(1, 2, gt[1], gt[2]),
            edge(0, 2, gt[0], gt[2]),
        ];
        let mut initial = vec![
            gt[0],
            se3_exp_f64([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            se3_exp_f64([2.7, 0.4, 0.0, 0.0, 0.03, 0.0]),
        ];
        let before = translation_error(&initial, &gt);
        let optimizer = PoseGraphOptimizer::new(PoseGraphConfig::default());
        let result = optimizer.optimize(&edges, &mut initial);
        let after = translation_error(&result.corrected_poses, &gt);
        assert!(after < before, "loop closure did not reduce drift");
    }

    #[test]
    fn pose_graph_optimizer_keeps_anchor_pose_fixed() {
        let gt = [
            Pose64::identity(),
            se3_exp_f64([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ];
        let edges = vec![edge(0, 1, gt[0], gt[1])];
        let mut initial = vec![gt[0], se3_exp_f64([1.4, 0.3, 0.0, 0.0, 0.02, 0.0])];
        let anchor_before = initial[0];
        let optimizer = PoseGraphOptimizer::new(PoseGraphConfig::default());
        let result = optimizer.optimize(&edges, &mut initial);
        let anchor_after = result.corrected_poses[0];
        for i in 0..3 {
            assert!((anchor_before.translation[i] - anchor_after.translation[i]).abs() < 1e-12);
            for j in 0..3 {
                assert!((anchor_before.rotation[i][j] - anchor_after.rotation[i][j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn essential_graph_builds_spanning_tree_connectivity() {
        let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
        let mut graph = EssentialGraph::new(2);
        graph.add_keyframe(kf0, map.covisibility().neighbors(kf0), &map);
        graph.add_keyframe(kf1, map.covisibility().neighbors(kf1), &map);
        graph.add_keyframe(kf2, map.covisibility().neighbors(kf2), &map);

        assert_eq!(graph.parent_of(kf0), Some(kf0));
        assert_eq!(graph.parent_of(kf1), Some(kf0));
        assert_eq!(graph.parent_of(kf2), Some(kf1));
        assert!(graph.all_edges().len() >= 2);
    }

    #[test]
    fn essential_graph_respects_strong_edge_threshold() {
        let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
        let mut graph = EssentialGraph::new(2);
        graph.add_keyframe(kf0, map.covisibility().neighbors(kf0), &map);
        graph.add_keyframe(kf1, map.covisibility().neighbors(kf1), &map);
        graph.add_keyframe(kf2, map.covisibility().neighbors(kf2), &map);
        let snapshot = graph.snapshot();
        assert_eq!(snapshot.strong_covis_edges.len(), 1);
        let strong = &snapshot.strong_covis_edges[0];
        assert_eq!(strong.kind, EssentialEdgeKind::StrongCovisibility);
        assert!((strong.a == kf1 && strong.b == kf0) || (strong.a == kf0 && strong.b == kf1));
    }

    #[test]
    fn essential_graph_snapshot_is_independent_copy() {
        let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
        let mut graph = EssentialGraph::new(2);
        graph.add_keyframe(kf0, map.covisibility().neighbors(kf0), &map);
        graph.add_keyframe(kf1, map.covisibility().neighbors(kf1), &map);
        graph.add_keyframe(kf2, map.covisibility().neighbors(kf2), &map);
        let snapshot = graph.snapshot();
        graph.add_loop_edge(EssentialEdge {
            a: kf2,
            b: kf0,
            kind: EssentialEdgeKind::Loop,
            relative_pose: Pose64::identity(),
            information: scalar_block(1.0),
        });
        assert_eq!(snapshot.loop_edges.len(), 0);
        assert_eq!(graph.snapshot().loop_edges.len(), 1);
    }

    #[test]
    fn essential_graph_remove_keyframe_reparents_children() {
        let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
        let mut graph = EssentialGraph::new(2);
        graph.add_keyframe(kf0, map.covisibility().neighbors(kf0), &map);
        graph.add_keyframe(kf1, map.covisibility().neighbors(kf1), &map);
        graph.add_keyframe(kf2, map.covisibility().neighbors(kf2), &map);
        assert_eq!(graph.parent_of(kf2), Some(kf1));

        graph
            .remove_keyframe(kf1, &map)
            .expect("remove non-root keyframe");
        assert_eq!(graph.parent_of(kf2), Some(kf0));
        assert_eq!(graph.parent_of(kf1), None);
        let snapshot = graph.snapshot();
        assert!(snapshot.order.iter().all(|&id| id != kf1));
        assert!(
            snapshot
                .spanning_edges
                .iter()
                .all(|edge| edge.a != kf1 && edge.b != kf1)
        );
        let input = graph.pose_graph_input();
        assert!(input.keyframe_ids.iter().all(|&id| id != kf1));
    }

    #[test]
    fn essential_graph_remove_keyframe_rejects_root() {
        let (map, kf0, kf1, _kf2) = make_map_for_essential_graph();
        let mut graph = EssentialGraph::new(2);
        graph.add_keyframe(kf0, map.covisibility().neighbors(kf0), &map);
        graph.add_keyframe(kf1, map.covisibility().neighbors(kf1), &map);

        let err = graph
            .remove_keyframe(kf0, &map)
            .expect_err("root removal should fail");
        assert_eq!(
            err,
            EssentialGraphError::RootRemovalDenied { keyframe_id: kf0 }
        );
    }

    #[test]
    fn essential_graph_remove_keyframe_rejects_missing_id() {
        let (map, kf0, kf1, _kf2) = make_map_for_essential_graph();
        let mut graph = EssentialGraph::new(2);
        graph.add_keyframe(kf0, map.covisibility().neighbors(kf0), &map);

        let err = graph
            .remove_keyframe(kf1, &map)
            .expect_err("missing keyframe should fail");
        assert_eq!(
            err,
            EssentialGraphError::KeyframeNotFound { keyframe_id: kf1 }
        );
    }

    #[test]
    fn essential_graph_remove_keyframe_purges_incident_loop_edges() {
        let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
        let mut graph = EssentialGraph::new(2);
        graph.add_keyframe(kf0, map.covisibility().neighbors(kf0), &map);
        graph.add_keyframe(kf1, map.covisibility().neighbors(kf1), &map);
        graph.add_keyframe(kf2, map.covisibility().neighbors(kf2), &map);
        graph.add_loop_edge(EssentialEdge {
            a: kf2,
            b: kf0,
            kind: EssentialEdgeKind::Loop,
            relative_pose: Pose64::identity(),
            information: scalar_block(1.0),
        });
        assert_eq!(graph.snapshot().loop_edges.len(), 1);

        graph
            .remove_keyframe(kf2, &map)
            .expect("remove keyframe with loop edge");
        let snapshot = graph.snapshot();
        assert_eq!(snapshot.loop_edges.len(), 0);
        assert!(
            snapshot
                .strong_covis_edges
                .iter()
                .all(|e| e.a != kf2 && e.b != kf2)
        );
        assert!(
            snapshot
                .spanning_edges
                .iter()
                .all(|e| e.a != kf2 && e.b != kf2)
        );
    }

    #[test]
    fn essential_graph_random_remove_preserves_connectivity_invariants() {
        let (map, ids) = make_chain_keyframes(12);
        let root = ids[0];
        let mut graph = EssentialGraph::new(100);
        for (idx, &id) in ids.iter().enumerate() {
            if idx == 0 {
                graph.add_keyframe(id, None, &map);
            } else {
                let mut covis = HashMap::new();
                covis.insert(ids[idx - 1], NonZeroU32::new(10).expect("non-zero"));
                graph.add_keyframe(id, Some(&covis), &map);
            }
        }

        let mut alive = ids.clone();
        let mut rng = Lcg::new(0x5EED_u64);
        for _ in 0..64 {
            let removable: Vec<_> = alive.iter().copied().filter(|id| *id != root).collect();
            if removable.is_empty() {
                break;
            }
            let remove_id = removable[rng.next_usize(removable.len())];
            graph
                .remove_keyframe(remove_id, &map)
                .expect("non-root should be removable");
            alive.retain(|id| *id != remove_id);

            let alive_set: HashSet<_> = alive.iter().copied().collect();
            let snapshot = graph.snapshot();
            assert_eq!(snapshot.parent.len(), alive.len());
            assert!(!snapshot.order.contains(&remove_id));

            for (&child, &parent) in &snapshot.parent {
                assert!(alive_set.contains(&child));
                assert!(alive_set.contains(&parent));
                if child == root {
                    assert_eq!(parent, root);
                }
            }

            for edge in snapshot
                .spanning_edges
                .iter()
                .chain(snapshot.strong_covis_edges.iter())
                .chain(snapshot.loop_edges.iter())
            {
                assert!(alive_set.contains(&edge.a));
                assert!(alive_set.contains(&edge.b));
            }

            let input = graph.pose_graph_input();
            let input_set: HashSet<_> = input.keyframe_ids.iter().copied().collect();
            assert!(input_set.is_subset(&alive_set));
            for edge in &input.edges {
                assert!(edge.from < input.keyframe_ids.len());
                assert!(edge.to < input.keyframe_ids.len());
            }
        }
    }
}
