use super::{BlockCsr6x6, NEAR_ZERO, PoseGraphError};

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
) -> Result<PcgResult, PoseGraphError> {
    let dim = h.nrows() * 6;
    if b.len() != dim {
        return Err(PoseGraphError::PcgRhsLength {
            expected: dim,
            actual: b.len(),
        });
    }
    if x.len() != dim {
        return Err(PoseGraphError::PcgSolutionLength {
            expected: dim,
            actual: x.len(),
        });
    }

    let mut r = vec![0.0_f64; dim];
    let mut hx = vec![0.0_f64; dim];
    h.spmv(x, &mut hx)?;
    for ((ri, bi), hi) in r.iter_mut().zip(b.iter()).zip(hx.iter()) {
        *ri = *bi - *hi;
    }

    let diag_inv = invert_diagonal_blocks(h.diagonal_blocks());
    let mut z = vec![0.0_f64; dim];
    apply_preconditioner(&diag_inv, &r, &mut z);
    let mut p = z.clone();
    let mut rz_old = dot(&r, &z);
    let mut residual_norm = norm(&r);
    if residual_norm <= tol {
        return Ok(PcgResult {
            iterations: 0,
            residual_norm,
            converged: true,
        });
    }

    let mut hp = vec![0.0_f64; dim];
    for iter in 0..max_iters {
        h.spmv(&p, &mut hp)?;
        let denom = dot(&p, &hp);
        if denom.abs() < NEAR_ZERO {
            return Ok(PcgResult {
                iterations: iter,
                residual_norm,
                converged: false,
            });
        }

        let alpha = rz_old / denom;
        for (xi, pi) in x.iter_mut().zip(p.iter()) {
            *xi += alpha * *pi;
        }
        for (ri, hi) in r.iter_mut().zip(hp.iter()) {
            *ri -= alpha * *hi;
        }
        residual_norm = norm(&r);
        if residual_norm <= tol {
            return Ok(PcgResult {
                iterations: iter + 1,
                residual_norm,
                converged: true,
            });
        }

        apply_preconditioner(&diag_inv, &r, &mut z);
        let rz_new = dot(&r, &z);
        if rz_old.abs() < NEAR_ZERO {
            return Ok(PcgResult {
                iterations: iter + 1,
                residual_norm,
                converged: false,
            });
        }
        let beta = rz_new / rz_old;
        for (pi, zi) in p.iter_mut().zip(z.iter()) {
            *pi = *zi + beta * *pi;
        }
        rz_old = rz_new;
    }

    Ok(PcgResult {
        iterations: max_iters,
        residual_norm,
        converged: false,
    })
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
        if max_val < NEAR_ZERO {
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
            if factor.abs() < NEAR_ZERO {
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
