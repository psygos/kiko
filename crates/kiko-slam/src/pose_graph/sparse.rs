use super::PoseGraphError;

const SYMMETRY_EPSILON_SCALE: f64 = 128.0;

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

    pub fn insert(
        &mut self,
        row: usize,
        col: usize,
        block: [[f64; 6]; 6],
    ) -> Result<(), PoseGraphError> {
        self.validate_index(row, col)?;
        Self::validate_block(row, col, &block)?;
        let Some(idx) = self.find_index(row, col) else {
            self.insert_new(row, col, block);
            return Ok(());
        };
        self.values[idx] = block;
        Ok(())
    }

    pub fn add_to(
        &mut self,
        row: usize,
        col: usize,
        block: [[f64; 6]; 6],
    ) -> Result<(), PoseGraphError> {
        self.validate_index(row, col)?;
        Self::validate_block(row, col, &block)?;
        let Some(idx) = self.find_index(row, col) else {
            self.insert_new(row, col, block);
            return Ok(());
        };
        let mut updated = self.values[idx];
        for (r, block_row) in block.iter().enumerate() {
            for (c, value) in block_row.iter().enumerate() {
                updated[r][c] += *value;
                if !updated[r][c].is_finite() {
                    return Err(PoseGraphError::NonFiniteCsrBlockValue {
                        row,
                        col,
                        block_row: r,
                        block_col: c,
                        value: updated[r][c],
                    });
                }
            }
        }
        self.values[idx] = updated;
        Ok(())
    }

    pub fn get(&self, row: usize, col: usize) -> Option<[[f64; 6]; 6]> {
        let idx = self.find_index(row, col)?;
        Some(self.values[idx])
    }

    pub fn spmv(&self, x: &[f64], y: &mut [f64]) -> Result<(), PoseGraphError> {
        let expected = self.nrows * 6;
        if x.len() != expected {
            return Err(PoseGraphError::SpmvInputLength {
                expected,
                actual: x.len(),
            });
        }
        if y.len() != expected {
            return Err(PoseGraphError::SpmvOutputLength {
                expected,
                actual: y.len(),
            });
        }
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
        Ok(())
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

    pub(super) fn validate_symmetric(&self) -> Result<(), PoseGraphError> {
        for row in 0..self.nrows {
            for idx in self.row_ptr[row]..self.row_ptr[row + 1] {
                let col = self.col_idx[idx];
                let forward = &self.values[idx];
                let transpose = self.get(col, row).unwrap_or([[0.0; 6]; 6]);
                for (block_row, values) in forward.iter().enumerate() {
                    for (block_col, value) in values.iter().copied().enumerate() {
                        let transposed_value = transpose[block_col][block_row];
                        let scale = value.abs().max(transposed_value.abs()).max(1.0);
                        if (value - transposed_value).abs()
                            > f64::EPSILON * SYMMETRY_EPSILON_SCALE * scale
                        {
                            return Err(PoseGraphError::AsymmetricPcgMatrix {
                                row,
                                col,
                                block_row,
                                block_col,
                                forward: value,
                                transpose: transposed_value,
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn find_index(&self, row: usize, col: usize) -> Option<usize> {
        if row >= self.nrows || col >= self.nrows {
            return None;
        }
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        self.col_idx[start..end]
            .binary_search(&col)
            .ok()
            .map(|offset| start + offset)
    }

    fn insert_new(&mut self, row: usize, col: usize, block: [[f64; 6]; 6]) {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        let insert_at = start + self.col_idx[start..end].partition_point(|&c| c <= col);

        self.col_idx.insert(insert_at, col);
        self.values.insert(insert_at, block);
        for ptr in self.row_ptr.iter_mut().skip(row + 1) {
            *ptr += 1;
        }
    }

    fn validate_index(&self, row: usize, col: usize) -> Result<(), PoseGraphError> {
        if row < self.nrows && col < self.nrows {
            Ok(())
        } else {
            Err(PoseGraphError::CsrIndexOutOfBounds {
                row,
                col,
                nrows: self.nrows,
            })
        }
    }

    fn validate_block(row: usize, col: usize, block: &[[f64; 6]; 6]) -> Result<(), PoseGraphError> {
        for (block_row, values) in block.iter().enumerate() {
            for (block_col, value) in values.iter().copied().enumerate() {
                if !value.is_finite() {
                    return Err(PoseGraphError::NonFiniteCsrBlockValue {
                        row,
                        col,
                        block_row,
                        block_col,
                        value,
                    });
                }
            }
        }
        Ok(())
    }
}
