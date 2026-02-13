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
        for r in 0..6 {
            for c in 0..6 {
                self.values[idx][r][c] += block[r][c];
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

#[cfg(test)]
mod tests {
    use super::{solve_pcg, BlockCsr6x6};

    fn scalar_block(diagonal: f64) -> [[f64; 6]; 6] {
        let mut block = [[0.0_f64; 6]; 6];
        for (i, row) in block.iter_mut().enumerate() {
            row[i] = diagonal;
        }
        block
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

        let mut y_dense = vec![0.0; 12];
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
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0];
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

        let x_true = vec![0.5, -0.3, 0.8, 0.1, -0.2, 0.4, 1.2, -0.6, 0.7, -0.9, 0.2, 0.3];
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
}
