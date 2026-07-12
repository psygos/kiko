use crate::math::{cholesky_6x6, cholesky_solve_6x6};

use super::{BlockCsr6x6, PcgStopReason, PoseGraphError};

const DOT_BREAKDOWN_EPSILON_SCALE: f64 = 64.0;

#[derive(Clone, Copy, Debug)]
pub struct PcgResult {
    pub iterations: usize,
    pub initial_residual_norm: f64,
    pub residual_norm: f64,
    pub target_residual_norm: f64,
    pub stop_reason: PcgStopReason,
}

impl PcgResult {
    pub fn converged(self) -> bool {
        matches!(self.stop_reason, PcgStopReason::Converged)
    }
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
    if !tol.is_finite() || tol <= 0.0 || tol > 1.0 {
        return Err(PoseGraphError::InvalidPcgTolerance { value: tol });
    }
    validate_finite_input("rhs", b)?;
    validate_finite_input("initial_solution", x)?;
    h.validate_symmetric()?;

    let mut r = vec![0.0_f64; dim];
    let mut hx = vec![0.0_f64; dim];
    h.spmv(x, &mut hx)?;
    for ((ri, bi), hi) in r.iter_mut().zip(b.iter()).zip(hx.iter()) {
        *ri = *bi - *hi;
    }

    let mut residual_norm = norm(&r);
    ensure_finite_scalar("residual_norm", 0, residual_norm)?;
    let initial_residual_norm = residual_norm;
    let target_residual_norm = tol * initial_residual_norm.max(1.0);
    if residual_norm <= target_residual_norm {
        return Ok(PcgResult {
            iterations: 0,
            initial_residual_norm,
            residual_norm,
            target_residual_norm,
            stop_reason: PcgStopReason::Converged,
        });
    }

    let diag_inv = invert_diagonal_blocks(h.diagonal_blocks())?;
    let mut z = vec![0.0_f64; dim];
    apply_preconditioner(&diag_inv, &r, &mut z);
    let mut p = z.clone();
    let mut rz_old = dot(&r, &z);
    ensure_finite_scalar("preconditioned residual inner product", 0, rz_old)?;
    if rz_old <= 0.0 {
        return Ok(PcgResult {
            iterations: 0,
            initial_residual_norm,
            residual_norm,
            target_residual_norm,
            stop_reason: PcgStopReason::NonPositiveCurvature,
        });
    }
    if numerically_zero_dot(rz_old, &r, &z, 0)? {
        return Ok(PcgResult {
            iterations: 0,
            initial_residual_norm,
            residual_norm,
            target_residual_norm,
            stop_reason: PcgStopReason::NearZeroPreconditionedResidual,
        });
    }

    let mut hp = vec![0.0_f64; dim];
    for iter in 0..max_iters {
        h.spmv(&p, &mut hp)?;
        let denom = dot(&p, &hp);
        ensure_finite_scalar("search curvature", iter, denom)?;
        if denom <= 0.0 {
            return Ok(PcgResult {
                iterations: iter,
                initial_residual_norm,
                residual_norm,
                target_residual_norm,
                stop_reason: PcgStopReason::NonPositiveCurvature,
            });
        }
        if numerically_zero_dot(denom, &p, &hp, iter)? {
            return Ok(PcgResult {
                iterations: iter,
                initial_residual_norm,
                residual_norm,
                target_residual_norm,
                stop_reason: PcgStopReason::NearZeroDenominator,
            });
        }

        let alpha = rz_old / denom;
        ensure_finite_scalar("step length", iter, alpha)?;
        for (xi, pi) in x.iter_mut().zip(p.iter()) {
            *xi += alpha * *pi;
        }
        for (ri, hi) in r.iter_mut().zip(hp.iter()) {
            *ri -= alpha * *hi;
        }
        residual_norm = norm(&r);
        ensure_finite_scalar("residual_norm", iter + 1, residual_norm)?;
        if residual_norm <= target_residual_norm {
            return Ok(PcgResult {
                iterations: iter + 1,
                initial_residual_norm,
                residual_norm,
                target_residual_norm,
                stop_reason: PcgStopReason::Converged,
            });
        }

        apply_preconditioner(&diag_inv, &r, &mut z);
        let rz_new = dot(&r, &z);
        ensure_finite_scalar("preconditioned residual inner product", iter + 1, rz_new)?;
        if rz_new <= 0.0 {
            return Ok(PcgResult {
                iterations: iter + 1,
                initial_residual_norm,
                residual_norm,
                target_residual_norm,
                stop_reason: PcgStopReason::NonPositiveCurvature,
            });
        }
        if numerically_zero_dot(rz_new, &r, &z, iter + 1)? {
            return Ok(PcgResult {
                iterations: iter + 1,
                initial_residual_norm,
                residual_norm,
                target_residual_norm,
                stop_reason: PcgStopReason::NearZeroPreconditionedResidual,
            });
        }
        let beta = rz_new / rz_old;
        ensure_finite_scalar("search direction scale", iter + 1, beta)?;
        for (pi, zi) in p.iter_mut().zip(z.iter()) {
            *pi = *zi + beta * *pi;
        }
        rz_old = rz_new;
    }

    Ok(PcgResult {
        iterations: max_iters,
        initial_residual_norm,
        residual_norm,
        target_residual_norm,
        stop_reason: PcgStopReason::IterationLimit,
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

fn invert_diagonal_blocks(diag: Vec<[[f64; 6]; 6]>) -> Result<Vec<[[f64; 6]; 6]>, PoseGraphError> {
    diag.into_iter()
        .enumerate()
        .map(|(block_index, block)| invert_spd_6x6(block, block_index))
        .collect()
}

fn invert_spd_6x6(a: [[f64; 6]; 6], block_index: usize) -> Result<[[f64; 6]; 6], PoseGraphError> {
    let l = cholesky_6x6(&a).ok_or(PoseGraphError::InvalidPcgDiagonalBlock { block_index })?;
    let mut inv = [[0.0_f64; 6]; 6];
    for col in 0..6 {
        let mut basis = [0.0_f64; 6];
        basis[col] = 1.0;
        let solution = cholesky_solve_6x6(&l, &basis);
        for row in 0..6 {
            let value = solution[row];
            if !value.is_finite() {
                return Err(PoseGraphError::InvalidPcgDiagonalBlock { block_index });
            }
            inv[row][col] = value;
        }
    }
    Ok(inv)
}

fn validate_finite_input(input: &'static str, values: &[f64]) -> Result<(), PoseGraphError> {
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(PoseGraphError::NonFinitePcgInput {
                input,
                index,
                value,
            });
        }
    }
    Ok(())
}

fn ensure_finite_scalar(
    scalar: &'static str,
    iteration: usize,
    value: f64,
) -> Result<(), PoseGraphError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(PoseGraphError::NonFinitePcgScalar {
            scalar,
            iteration,
            value,
        })
    }
}

fn numerically_zero_dot(
    dot_value: f64,
    a: &[f64],
    b: &[f64],
    iteration: usize,
) -> Result<bool, PoseGraphError> {
    let scale = norm(a) * norm(b);
    ensure_finite_scalar("dot-product scale", iteration, scale)?;
    Ok(scale == 0.0 || dot_value.abs() <= f64::EPSILON * DOT_BREAKDOWN_EPSILON_SCALE * scale)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f64]) -> f64 {
    v.iter()
        .fold(0.0_f64, |magnitude, value| magnitude.hypot(*value))
}
