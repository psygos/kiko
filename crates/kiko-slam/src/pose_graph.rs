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

#[cfg(test)]
mod tests {
    use super::BlockCsr6x6;

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
}
