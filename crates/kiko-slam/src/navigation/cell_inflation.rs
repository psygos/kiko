//! Reusable exact cell-square inflation for bounded row-major grids.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CellInflationError {
    InvalidInput,
    AllocationFailed {
        context: &'static str,
        requested: usize,
    },
}

/// Allocation-free-after-construction conservative grid inflation.
///
/// A one-cell Chebyshev expansion converts source-cell/candidate-cell square
/// separation into centre separation. A separable exact squared Euclidean
/// distance transform then applies the requested continuous radius in O(N).
/// This accounts for both cells' spatial extent without a centre-only or
/// half-diagonal approximation.
pub(super) struct CellSquareInflation {
    width: usize,
    height: usize,
    cell_count: usize,
    expanded_sources: Vec<bool>,
    vertical_distances: Vec<u64>,
    sites: Vec<usize>,
    starts: Vec<i128>,
}

impl CellSquareInflation {
    pub(super) fn try_new(width: usize, height: usize) -> Result<Self, CellInflationError> {
        let cell_count = width
            .checked_mul(height)
            .filter(|_| {
                width > 0 && height > 0 && width <= u32::MAX as usize && height <= u32::MAX as usize
            })
            .ok_or(CellInflationError::InvalidInput)?;
        let expanded_sources = bool_buffer(cell_count, "expanded source mask")?;
        let mut vertical_distances = Vec::new();
        reserve(
            &mut vertical_distances,
            cell_count,
            "vertical distance transform",
        )?;
        vertical_distances.resize(cell_count, u64::MAX);
        let mut sites = Vec::new();
        reserve(&mut sites, width, "distance-transform sites")?;
        let mut starts = Vec::new();
        reserve(&mut starts, width, "distance-transform boundaries")?;
        Ok(Self {
            width,
            height,
            cell_count,
            expanded_sources,
            vertical_distances,
            sites,
            starts,
        })
    }

    pub(super) fn inflate(
        &mut self,
        sources: &[bool],
        output: &mut [bool],
        resolution_m: f64,
        radius_m: f64,
        block_exterior: bool,
    ) -> Result<(), CellInflationError> {
        if sources.len() != self.cell_count
            || output.len() != self.cell_count
            || !resolution_m.is_finite()
            || resolution_m <= 0.0
            || !radius_m.is_finite()
            || radius_m < 0.0
        {
            return Err(CellInflationError::InvalidInput);
        }
        if radius_m == 0.0 {
            output.clone_from_slice(sources);
            return Ok(());
        }

        let limit_squared = conservative_squared_cell_limit(radius_m, resolution_m);
        let maximum_boundary_gap = ((self.width - 1) / 2).min((self.height - 1) / 2);
        if block_exterior && square_usize(maximum_boundary_gap) <= limit_squared {
            output.fill(true);
            return Ok(());
        }

        self.expanded_sources.fill(false);
        for source_row in 0..self.height {
            for source_column in 0..self.width {
                if !sources[source_row * self.width + source_column] {
                    continue;
                }
                let minimum_row = source_row.saturating_sub(1);
                let maximum_row = source_row.saturating_add(1).min(self.height - 1);
                let minimum_column = source_column.saturating_sub(1);
                let maximum_column = source_column.saturating_add(1).min(self.width - 1);
                for row in minimum_row..=maximum_row {
                    for column in minimum_column..=maximum_column {
                        self.expanded_sources[row * self.width + column] = true;
                    }
                }
            }
        }

        output.fill(false);
        if !self.expanded_sources.iter().any(|source| *source) {
            if block_exterior {
                for row in 0..self.height {
                    for column in 0..self.width {
                        output[row * self.width + column] =
                            boundary_gap_squared(self.width, self.height, column, row)
                                <= limit_squared;
                    }
                }
            }
            return Ok(());
        }

        // The vertical pass has binary input, so two nearest-source sweeps are
        // an exact one-dimensional squared distance transform.
        self.vertical_distances.fill(u64::MAX);
        for column in 0..self.width {
            let mut nearest_source = None;
            for row in 0..self.height {
                let index = row * self.width + column;
                if self.expanded_sources[index] {
                    nearest_source = Some(row);
                    self.vertical_distances[index] = 0;
                } else if let Some(source_row) = nearest_source {
                    self.vertical_distances[index] = square_axis_delta(row, source_row)?;
                }
            }
            nearest_source = None;
            for row in (0..self.height).rev() {
                let index = row * self.width + column;
                if self.expanded_sources[index] {
                    nearest_source = Some(row);
                } else if let Some(source_row) = nearest_source {
                    self.vertical_distances[index] =
                        self.vertical_distances[index].min(square_axis_delta(row, source_row)?);
                }
            }
        }

        // Each row is the lower envelope of integer parabolas. Integer start
        // positions avoid floating-point tie differences.
        for row in 0..self.height {
            self.sites.clear();
            self.starts.clear();
            for column in 0..self.width {
                let value = self.vertical_distances[row * self.width + column];
                if value == u64::MAX {
                    continue;
                }
                if self.sites.is_empty() {
                    self.sites.push(column);
                    self.starts.push(i128::MIN);
                    continue;
                }
                loop {
                    let Some(&previous) = self.sites.last() else {
                        return Err(CellInflationError::InvalidInput);
                    };
                    let previous_value = self.vertical_distances[row * self.width + previous];
                    let start = parabola_start(previous, previous_value, column, value);
                    let Some(&previous_start) = self.starts.last() else {
                        return Err(CellInflationError::InvalidInput);
                    };
                    if start > previous_start {
                        self.sites.push(column);
                        self.starts.push(start);
                        break;
                    }
                    self.sites.pop();
                    self.starts.pop();
                    if self.sites.is_empty() {
                        self.sites.push(column);
                        self.starts.push(i128::MIN);
                        break;
                    }
                }
            }

            let mut envelope_index = 0;
            for column in 0..self.width {
                while envelope_index + 1 < self.sites.len()
                    && self.starts[envelope_index + 1] <= column as i128
                {
                    envelope_index += 1;
                }
                let Some(&site) = self.sites.get(envelope_index) else {
                    return Err(CellInflationError::InvalidInput);
                };
                let source_distance_squared = square_usize(column.abs_diff(site))
                    + u128::from(self.vertical_distances[row * self.width + site]);
                output[row * self.width + column] = source_distance_squared <= limit_squared
                    || (block_exterior
                        && boundary_gap_squared(self.width, self.height, column, row)
                            <= limit_squared);
            }
        }
        Ok(())
    }
}

fn bool_buffer(length: usize, context: &'static str) -> Result<Vec<bool>, CellInflationError> {
    let mut values = Vec::new();
    reserve(&mut values, length, context)?;
    values.resize(length, false);
    Ok(values)
}

fn reserve<T>(
    values: &mut Vec<T>,
    additional: usize,
    context: &'static str,
) -> Result<(), CellInflationError> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| CellInflationError::AllocationFailed {
            context,
            requested: additional,
        })
}

fn conservative_squared_cell_limit(radius_m: f64, resolution_m: f64) -> u128 {
    let radius_cells = radius_m / resolution_m;
    if !radius_cells.is_finite() {
        return u128::MAX;
    }
    // Both operations can round toward zero. Move the positive ratio outward
    // before squaring, then move the square outward again so a tangent cell
    // cannot become traversable at a representable threshold.
    let outward_radius_cells = next_up_positive(radius_cells);
    let squared = outward_radius_cells * outward_radius_cells;
    if !squared.is_finite() {
        return u128::MAX;
    }
    let outward = f64::from_bits(squared.to_bits().saturating_add(1));
    if outward >= u128::MAX as f64 {
        u128::MAX
    } else {
        outward.floor() as u128
    }
}

fn next_up_positive(value: f64) -> f64 {
    debug_assert!(value.is_finite());
    debug_assert!(value >= 0.0);
    f64::from_bits(value.to_bits().saturating_add(1))
}

fn square_usize(value: usize) -> u128 {
    let value = value as u128;
    value * value
}

fn square_axis_delta(left: usize, right: usize) -> Result<u64, CellInflationError> {
    u64::try_from(square_usize(left.abs_diff(right))).map_err(|_| CellInflationError::InvalidInput)
}

fn boundary_gap_squared(width: usize, height: usize, column: usize, row: usize) -> u128 {
    let horizontal_gap = column.min(width - 1 - column);
    let vertical_gap = row.min(height - 1 - row);
    square_usize(horizontal_gap.min(vertical_gap))
}

fn parabola_start(
    left_index: usize,
    left_value: u64,
    right_index: usize,
    right_value: u64,
) -> i128 {
    debug_assert!(left_index < right_index);
    debug_assert_ne!(left_value, u64::MAX);
    debug_assert_ne!(right_value, u64::MAX);
    let left_index = left_index as i128;
    let right_index = right_index as i128;
    let numerator = right_index * right_index + i128::from(right_value)
        - left_index * left_index
        - i128::from(left_value);
    let denominator = 2 * (right_index - left_index);
    let quotient = numerator.div_euclid(denominator);
    if numerator.rem_euclid(denominator) == 0 {
        quotient
    } else {
        quotient + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next_down_positive(value: f64) -> f64 {
        debug_assert!(value > 0.0 && value.is_finite());
        f64::from_bits(value.to_bits() - 1)
    }

    #[test]
    fn representable_tangent_thresholds_are_never_rounded_traversable() {
        let width = 17;
        let height = 5;
        let source_column = 1;
        let source_row = 2;
        let mut sources = vec![false; width * height];
        sources[source_row * width + source_column] = true;
        let mut result = vec![false; sources.len()];
        let mut inflation = CellSquareInflation::try_new(width, height).expect("inflation");

        for resolution_m in [0.01, 0.03, 0.1, 0.2, 1.0, 3.0] {
            for square_gap_cells in 1..=8_usize {
                let candidate_column = source_column + square_gap_cells + 1;
                let tangent_radius_m = square_gap_cells as f64 * resolution_m;
                for radius_m in [
                    next_down_positive(tangent_radius_m),
                    tangent_radius_m,
                    next_up_positive(tangent_radius_m),
                ] {
                    inflation
                        .inflate(&sources, &mut result, resolution_m, radius_m, false)
                        .expect("valid transform");
                    if radius_m >= tangent_radius_m {
                        assert!(
                            result[source_row * width + candidate_column],
                            "tangent/overlapping cell became traversable: resolution={resolution_m:e}, gap={square_gap_cells}, radius={radius_m:e}, tangent={tangent_radius_m:e}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn distance_transform_is_a_conservative_cell_square_oracle() {
        for width in 1..=8_usize {
            for height in 1..=7_usize {
                let cell_count = width * height;
                let mut inflation =
                    CellSquareInflation::try_new(width, height).expect("inflation scratch");
                for pattern in 0..32_u64 {
                    let sources = (0..cell_count)
                        .map(|index| {
                            let mixed = (index as u64)
                                .wrapping_mul(0x9e37_79b9)
                                .wrapping_add(pattern.wrapping_mul(0x85eb_ca6b));
                            mixed.rotate_left((index % 31) as u32) & 7 == 0
                        })
                        .collect::<Vec<_>>();
                    let mut result = vec![false; cell_count];
                    for (resolution_m, radius_m) in [
                        (0.05, 0.01),
                        (0.05, 0.05),
                        (0.2, 0.31),
                        (1.0, std::f64::consts::SQRT_2),
                    ] {
                        inflation
                            .inflate(&sources, &mut result, resolution_m, radius_m, false)
                            .expect("valid inflation");
                        for row in 0..height {
                            for column in 0..width {
                                let oracle_blocked =
                                    sources.iter().enumerate().any(|(source_index, source)| {
                                        if !source {
                                            return false;
                                        }
                                        let source_column = source_index % width;
                                        let source_row = source_index / width;
                                        let gap_x =
                                            column.abs_diff(source_column).saturating_sub(1);
                                        let gap_y = row.abs_diff(source_row).saturating_sub(1);
                                        let distance_m =
                                            (gap_x as f64).hypot(gap_y as f64) * resolution_m;
                                        distance_m <= radius_m
                                    });
                                assert!(
                                    !oracle_blocked || result[row * width + column],
                                    "false-negative inflation at {width}x{height}, pattern={pattern}, cell=({column},{row}), resolution={resolution_m}, radius={radius_m}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
