//! Deterministic host-side 2D occupancy mapping.
//!
//! This module deliberately contains no learned model and no hardware backend.
//! It projects calibrated depth observations into an explicitly supplied
//! occupancy frame, fuses bounded fixed-point evidence, and retains enough
//! compact source data to remove or rebuild every accepted keyframe exactly.

use std::collections::{HashMap, HashSet, VecDeque};
use std::num::{NonZeroU32, NonZeroUsize};
use std::sync::Arc;

use crate::map::{KeyframeId, MapInstanceId};
use crate::{DepthImage, FrameDimensions, PinholeIntrinsics, Pose, WorldToCamera};

const ROTATION_VALIDATION_TOLERANCE: f64 = 1.0e-6;

/// A validated rigid transform from Kiko's visual world frame into an
/// occupancy frame whose coordinates are `[grid_x, grid_y, height]`, in metres.
///
/// There is intentionally no default: visual SLAM does not establish gravity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldToOccupancy {
    rotation: [[f64; 3]; 3],
    translation_m: [f64; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WorldToOccupancyError {
    NonFiniteRotation {
        row: usize,
        column: usize,
        value: f64,
    },
    NonFiniteTranslation {
        axis: usize,
        value: f64,
    },
    NonOrthonormalRotation {
        max_error: f64,
    },
    ImproperRotation {
        determinant: f64,
    },
    InvalidLevelCameraHeight {
        camera_height_m: f64,
    },
}

impl std::fmt::Display for WorldToOccupancyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteRotation { row, column, value } => write!(
                f,
                "world-to-occupancy rotation[{row}][{column}] must be finite, got {value}"
            ),
            Self::NonFiniteTranslation { axis, value } => write!(
                f,
                "world-to-occupancy translation[{axis}] must be finite metres, got {value}"
            ),
            Self::NonOrthonormalRotation { max_error } => write!(
                f,
                "world-to-occupancy rotation must be orthonormal (maximum error {max_error})"
            ),
            Self::ImproperRotation { determinant } => write!(
                f,
                "world-to-occupancy rotation determinant must be +1, got {determinant}"
            ),
            Self::InvalidLevelCameraHeight { camera_height_m } => write!(
                f,
                "level optical-world camera height must be finite and nonnegative, got {camera_height_m} m"
            ),
        }
    }
}

impl std::error::Error for WorldToOccupancyError {}

impl WorldToOccupancy {
    pub fn try_new(
        rotation: [[f64; 3]; 3],
        translation_m: [f64; 3],
    ) -> Result<Self, WorldToOccupancyError> {
        for (row_index, row) in rotation.iter().enumerate() {
            for (column, &value) in row.iter().enumerate() {
                if !value.is_finite() {
                    return Err(WorldToOccupancyError::NonFiniteRotation {
                        row: row_index,
                        column,
                        value,
                    });
                }
            }
        }
        if let Some(axis) = translation_m.iter().position(|value| !value.is_finite()) {
            return Err(WorldToOccupancyError::NonFiniteTranslation {
                axis,
                value: translation_m[axis],
            });
        }

        let mut max_error = 0.0_f64;
        for row in 0..3 {
            for column in 0..3 {
                let dot = (0..3)
                    .map(|index| rotation[index][row] * rotation[index][column])
                    .sum::<f64>();
                let expected = if row == column { 1.0 } else { 0.0 };
                max_error = max_error.max((dot - expected).abs());
            }
        }
        if max_error > ROTATION_VALIDATION_TOLERANCE {
            return Err(WorldToOccupancyError::NonOrthonormalRotation { max_error });
        }
        let determinant = determinant(rotation);
        if (determinant - 1.0).abs() > ROTATION_VALIDATION_TOLERANCE {
            return Err(WorldToOccupancyError::ImproperRotation { determinant });
        }

        Ok(Self {
            rotation,
            translation_m,
        })
    }

    /// Explicit convenience for the assumption that the visual world is a
    /// level optical frame (`+x` right, `+y` down, `+z` forward) whose camera
    /// centre is `camera_height_m` above the floor.
    ///
    /// The resulting mapping is
    /// `[occ_x, occ_y, height] = [world_x, world_z, camera_height_m - world_y]`.
    pub fn level_optical_world(camera_height_m: f64) -> Result<Self, WorldToOccupancyError> {
        if !camera_height_m.is_finite() || camera_height_m < 0.0 {
            return Err(WorldToOccupancyError::InvalidLevelCameraHeight { camera_height_m });
        }
        Self::try_new(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            [0.0, 0.0, camera_height_m],
        )
    }

    pub fn rotation(self) -> [[f64; 3]; 3] {
        self.rotation
    }

    pub fn translation_m(self) -> [f64; 3] {
        self.translation_m
    }

    pub fn try_transform_world_point(self, world_m: [f64; 3]) -> Result<[f64; 3], OccupancyError> {
        transform_point(self.rotation, self.translation_m, world_m)
    }
}

/// Explicit rigid transform from the depth optical camera into the tracking
/// camera whose world-to-camera pose is supplied with each keyframe.
#[derive(Clone, Copy, Debug)]
pub struct DepthToTrackingCamera(Pose);

impl DepthToTrackingCamera {
    pub fn new(pose: Pose) -> Self {
        Self(pose)
    }

    pub fn identity() -> Self {
        Self(Pose::identity())
    }

    pub fn pose(self) -> Pose {
        self.0
    }
}

/// Calibration and exact image-shape contract for depth samples.
#[derive(Clone, Copy, Debug)]
pub struct DepthCameraModel {
    intrinsics: PinholeIntrinsics,
    dimensions: FrameDimensions,
    depth_to_tracking: DepthToTrackingCamera,
}

impl DepthCameraModel {
    pub fn new(
        intrinsics: PinholeIntrinsics,
        dimensions: FrameDimensions,
        depth_to_tracking: DepthToTrackingCamera,
    ) -> Self {
        Self {
            intrinsics,
            dimensions,
            depth_to_tracking,
        }
    }

    pub fn intrinsics(self) -> PinholeIntrinsics {
        self.intrinsics
    }

    pub fn dimensions(self) -> FrameDimensions {
        self.dimensions
    }

    pub fn depth_to_tracking(self) -> DepthToTrackingCamera {
        self.depth_to_tracking
    }
}

/// Fixed, bounded row-major occupancy-grid geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OccupancyGridGeometry {
    resolution_m: f64,
    lower_bound_m: [f64; 2],
    upper_bound_m: [f64; 2],
    width: NonZeroU32,
    height: NonZeroU32,
    cell_count: usize,
    max_cells: NonZeroUsize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OccupancyGridGeometryError {
    InvalidResolution {
        resolution_m: f64,
    },
    NonFiniteLowerBound {
        axis: usize,
        value: f64,
    },
    ZeroDimensions {
        width: u32,
        height: u32,
    },
    ZeroMaximumCells,
    CellCountNotAddressable {
        width: u32,
        height: u32,
    },
    TooManyCells {
        cells: usize,
        maximum: usize,
    },
    NonFiniteUpperBound {
        axis: usize,
        value: f64,
    },
    IndistinguishableBounds {
        axis: usize,
        lower: f64,
        upper: f64,
    },
    IndistinguishableCellBoundary {
        axis: usize,
        boundary_index: u32,
        coordinate_m: f64,
        adjacent_coordinate_m: f64,
    },
}

impl std::fmt::Display for OccupancyGridGeometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidResolution { resolution_m } => write!(
                f,
                "occupancy resolution must be finite and positive, got {resolution_m} m/cell"
            ),
            Self::NonFiniteLowerBound { axis, value } => write!(
                f,
                "occupancy lower bound on axis {axis} must be finite metres, got {value}"
            ),
            Self::ZeroDimensions { width, height } => {
                write!(
                    f,
                    "occupancy dimensions must be nonzero, got {width}x{height}"
                )
            }
            Self::ZeroMaximumCells => write!(f, "occupancy maximum cell count must be nonzero"),
            Self::CellCountNotAddressable { width, height } => write!(
                f,
                "occupancy dimensions {width}x{height} exceed addressable memory"
            ),
            Self::TooManyCells { cells, maximum } => write!(
                f,
                "occupancy grid has {cells} cells, exceeding configured maximum {maximum}"
            ),
            Self::NonFiniteUpperBound { axis, value } => write!(
                f,
                "occupancy upper bound on axis {axis} is not finite: {value}"
            ),
            Self::IndistinguishableBounds { axis, lower, upper } => write!(
                f,
                "occupancy bounds on axis {axis} are not distinguishable in f64: [{lower}, {upper})"
            ),
            Self::IndistinguishableCellBoundary {
                axis,
                boundary_index,
                coordinate_m,
                adjacent_coordinate_m,
            } => write!(
                f,
                "occupancy cell boundary {boundary_index} on axis {axis} is numerically indistinguishable from its adjacent boundary: {coordinate_m} m versus {adjacent_coordinate_m} m"
            ),
        }
    }
}

impl std::error::Error for OccupancyGridGeometryError {}

impl OccupancyGridGeometry {
    pub fn try_new(
        resolution_m: f64,
        lower_bound_m: [f64; 2],
        width: u32,
        height: u32,
        max_cells: usize,
    ) -> Result<Self, OccupancyGridGeometryError> {
        if !resolution_m.is_finite() || resolution_m <= 0.0 {
            return Err(OccupancyGridGeometryError::InvalidResolution { resolution_m });
        }
        if let Some(axis) = lower_bound_m.iter().position(|value| !value.is_finite()) {
            return Err(OccupancyGridGeometryError::NonFiniteLowerBound {
                axis,
                value: lower_bound_m[axis],
            });
        }
        let Some(width_nonzero) = NonZeroU32::new(width) else {
            return Err(OccupancyGridGeometryError::ZeroDimensions { width, height });
        };
        let Some(height_nonzero) = NonZeroU32::new(height) else {
            return Err(OccupancyGridGeometryError::ZeroDimensions { width, height });
        };
        let Some(max_cells) = NonZeroUsize::new(max_cells) else {
            return Err(OccupancyGridGeometryError::ZeroMaximumCells);
        };
        let cell_count = usize::try_from(width)
            .ok()
            .and_then(|width| {
                usize::try_from(height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .ok_or(OccupancyGridGeometryError::CellCountNotAddressable { width, height })?;
        if cell_count > max_cells.get() {
            return Err(OccupancyGridGeometryError::TooManyCells {
                cells: cell_count,
                maximum: max_cells.get(),
            });
        }

        let extents = [
            resolution_m * f64::from(width),
            resolution_m * f64::from(height),
        ];
        let upper_bound_m = [lower_bound_m[0] + extents[0], lower_bound_m[1] + extents[1]];
        for axis in 0..2 {
            if !upper_bound_m[axis].is_finite() {
                return Err(OccupancyGridGeometryError::NonFiniteUpperBound {
                    axis,
                    value: upper_bound_m[axis],
                });
            }
            if upper_bound_m[axis] <= lower_bound_m[axis] {
                return Err(OccupancyGridGeometryError::IndistinguishableBounds {
                    axis,
                    lower: lower_bound_m[axis],
                    upper: upper_bound_m[axis],
                });
            }

            let axis_cell_count = [width, height][axis];
            let mut previous_boundary = lower_bound_m[axis];
            for boundary_index in 1..=axis_cell_count {
                let boundary = if boundary_index == axis_cell_count {
                    upper_bound_m[axis]
                } else {
                    lower_bound_m[axis] + resolution_m * f64::from(boundary_index)
                };
                if boundary <= previous_boundary {
                    return Err(OccupancyGridGeometryError::IndistinguishableCellBoundary {
                        axis,
                        boundary_index,
                        coordinate_m: boundary,
                        adjacent_coordinate_m: previous_boundary,
                    });
                }
                previous_boundary = boundary;
            }
        }

        Ok(Self {
            resolution_m,
            lower_bound_m,
            upper_bound_m,
            width: width_nonzero,
            height: height_nonzero,
            cell_count,
            max_cells,
        })
    }

    pub fn resolution_m(self) -> f64 {
        self.resolution_m
    }

    pub fn lower_bound_m(self) -> [f64; 2] {
        self.lower_bound_m
    }

    pub fn upper_bound_m(self) -> [f64; 2] {
        self.upper_bound_m
    }

    pub fn width(self) -> u32 {
        self.width.get()
    }

    pub fn height(self) -> u32 {
        self.height.get()
    }

    pub fn cell_count(self) -> usize {
        self.cell_count
    }

    pub fn max_cells(self) -> usize {
        self.max_cells.get()
    }

    fn contains_xy(self, point: [f64; 2]) -> bool {
        point[0] >= self.lower_bound_m[0]
            && point[0] < self.upper_bound_m[0]
            && point[1] >= self.lower_bound_m[1]
            && point[1] < self.upper_bound_m[1]
    }

    fn point_index(self, point: [f64; 2]) -> Option<usize> {
        if !self.contains_xy(point) {
            return None;
        }
        let width = self.width.get() as usize;
        let column = self.axis_cell(0, point[0]);
        let row = self.axis_cell(1, point[1]);
        Some(row * width + column)
    }

    /// Returns the positive-side cell at a generated internal boundary.
    ///
    /// Grid construction proves that the exact boundary sequence used here is
    /// strictly increasing. Division supplies the fast-path estimate, then
    /// exact comparisons against that sequence either confirm it or bound a
    /// binary correction. This avoids both a full search on every ray and the
    /// inconsistent rounding of an unchecked quotient.
    fn axis_cell(self, axis: usize, coordinate_m: f64) -> usize {
        let cell_count = [self.width.get(), self.height.get()][axis] as usize;
        let quotient = ((coordinate_m - self.lower_bound_m[axis]) / self.resolution_m).floor();
        debug_assert!(quotient.is_finite());
        let estimate = (quotient as usize).min(cell_count - 1);
        let estimate_lower_m = self.axis_boundary_m(axis, estimate);
        let next_boundary_m =
            (estimate + 1 < cell_count).then(|| self.axis_boundary_m(axis, estimate + 1));

        let (mut lower_index, mut upper_index) = if estimate_lower_m > coordinate_m {
            (0, estimate.saturating_sub(1))
        } else if next_boundary_m.is_some_and(|boundary_m| boundary_m <= coordinate_m) {
            (estimate + 1, cell_count - 1)
        } else {
            return estimate;
        };
        while lower_index < upper_index {
            let distance = upper_index - lower_index;
            let midpoint = lower_index + distance.div_ceil(2);
            let boundary_m = self.axis_boundary_m(axis, midpoint);
            if boundary_m <= coordinate_m {
                lower_index = midpoint;
            } else {
                upper_index = midpoint - 1;
            }
        }
        lower_index
    }

    fn axis_boundary_m(self, axis: usize, boundary_index: usize) -> f64 {
        self.lower_bound_m[axis] + self.resolution_m * boundary_index as f64
    }

    fn traversal_cell(self, point: [f64; 2]) -> Option<(usize, usize)> {
        if point.iter().any(|value| !value.is_finite())
            || point[0] < self.lower_bound_m[0]
            || point[0] > self.upper_bound_m[0]
            || point[1] < self.lower_bound_m[1]
            || point[1] > self.upper_bound_m[1]
        {
            return None;
        }
        Some((self.axis_cell(0, point[0]), self.axis_cell(1, point[1])))
    }

    fn clamp_to_closed_bounds(self, point: [f64; 2]) -> [f64; 2] {
        [
            point[0].clamp(self.lower_bound_m[0], self.upper_bound_m[0]),
            point[1].clamp(self.lower_bound_m[1], self.upper_bound_m[1]),
        ]
    }
}

macro_rules! metric_range {
    ($name:ident, $error:ident, $description:literal, $positive:expr) => {
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $name {
            minimum_m: f64,
            maximum_m: f64,
        }

        #[derive(Clone, Copy, Debug, PartialEq)]
        pub enum $error {
            NonFinite { minimum_m: f64, maximum_m: f64 },
            InvalidOrder { minimum_m: f64, maximum_m: f64 },
            NonPositiveMinimum { minimum_m: f64 },
        }

        impl std::fmt::Display for $error {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    Self::NonFinite {
                        minimum_m,
                        maximum_m,
                    } => write!(
                        f,
                        concat!($description, " bounds must be finite metres, got [{}, {}]"),
                        minimum_m, maximum_m
                    ),
                    Self::InvalidOrder {
                        minimum_m,
                        maximum_m,
                    } => write!(
                        f,
                        concat!($description, " requires minimum < maximum, got [{}, {}]"),
                        minimum_m, maximum_m
                    ),
                    Self::NonPositiveMinimum { minimum_m } => write!(
                        f,
                        concat!($description, " minimum must be positive, got {} m"),
                        minimum_m
                    ),
                }
            }
        }

        impl std::error::Error for $error {}

        impl $name {
            pub fn try_new(minimum_m: f64, maximum_m: f64) -> Result<Self, $error> {
                if !minimum_m.is_finite() || !maximum_m.is_finite() {
                    return Err($error::NonFinite {
                        minimum_m,
                        maximum_m,
                    });
                }
                if $positive && minimum_m <= 0.0 {
                    return Err($error::NonPositiveMinimum { minimum_m });
                }
                if minimum_m >= maximum_m {
                    return Err($error::InvalidOrder {
                        minimum_m,
                        maximum_m,
                    });
                }
                Ok(Self {
                    minimum_m,
                    maximum_m,
                })
            }

            pub fn minimum_m(self) -> f64 {
                self.minimum_m
            }

            pub fn maximum_m(self) -> f64 {
                self.maximum_m
            }

            fn contains(self, value_m: f64) -> bool {
                value_m >= self.minimum_m && value_m <= self.maximum_m
            }
        }
    };
}

metric_range!(
    HeightRangeMeters,
    HeightRangeError,
    "occupancy height range",
    false
);
metric_range!(
    DepthRangeMeters,
    DepthRangeError,
    "occupancy depth range",
    true
);

/// Fixed-point evidence parameters. Evidence is summed exactly and is never
/// clamped during integration, so integration order and exact removal commute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OccupancyEvidenceModel {
    free_delta: i32,
    occupied_delta: i32,
    free_threshold: i32,
    occupied_threshold: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OccupancyEvidenceModelError {
    NonNegativeFreeDelta { value: i32 },
    NonPositiveOccupiedDelta { value: i32 },
    NonNegativeFreeThreshold { value: i32 },
    NonPositiveOccupiedThreshold { value: i32 },
}

impl std::fmt::Display for OccupancyEvidenceModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonNegativeFreeDelta { value } => {
                write!(f, "free evidence delta must be negative, got {value}")
            }
            Self::NonPositiveOccupiedDelta { value } => {
                write!(f, "occupied evidence delta must be positive, got {value}")
            }
            Self::NonNegativeFreeThreshold { value } => {
                write!(f, "free evidence threshold must be negative, got {value}")
            }
            Self::NonPositiveOccupiedThreshold { value } => {
                write!(
                    f,
                    "occupied evidence threshold must be positive, got {value}"
                )
            }
        }
    }
}

impl std::error::Error for OccupancyEvidenceModelError {}

impl OccupancyEvidenceModel {
    pub fn try_new(
        free_delta: i32,
        occupied_delta: i32,
        free_threshold: i32,
        occupied_threshold: i32,
    ) -> Result<Self, OccupancyEvidenceModelError> {
        if free_delta >= 0 {
            return Err(OccupancyEvidenceModelError::NonNegativeFreeDelta { value: free_delta });
        }
        if occupied_delta <= 0 {
            return Err(OccupancyEvidenceModelError::NonPositiveOccupiedDelta {
                value: occupied_delta,
            });
        }
        if free_threshold >= 0 {
            return Err(OccupancyEvidenceModelError::NonNegativeFreeThreshold {
                value: free_threshold,
            });
        }
        if occupied_threshold <= 0 {
            return Err(OccupancyEvidenceModelError::NonPositiveOccupiedThreshold {
                value: occupied_threshold,
            });
        }
        Ok(Self {
            free_delta,
            occupied_delta,
            free_threshold,
            occupied_threshold,
        })
    }

    pub fn free_delta(self) -> i32 {
        self.free_delta
    }

    pub fn occupied_delta(self) -> i32 {
        self.occupied_delta
    }

    pub fn free_threshold(self) -> i32 {
        self.free_threshold
    }

    pub fn occupied_threshold(self) -> i32 {
        self.occupied_threshold
    }

    fn classify(self, evidence: i32) -> OccupancyCell {
        if evidence >= self.occupied_threshold {
            OccupancyCell::Occupied
        } else if evidence <= self.free_threshold {
            OccupancyCell::Free
        } else {
            OccupancyCell::Unknown
        }
    }
}

#[derive(Clone, Debug)]
pub struct OccupancyConfig {
    geometry: OccupancyGridGeometry,
    world_to_occupancy: WorldToOccupancy,
    camera: DepthCameraModel,
    height_range: HeightRangeMeters,
    depth_range: DepthRangeMeters,
    sampling_block: NonZeroU32,
    evidence: OccupancyEvidenceModel,
    max_keyframes: NonZeroUsize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OccupancyConfigError {
    ZeroSamplingBlock,
    ZeroMaximumKeyframes,
    EvidenceAccumulatorMayOverflow {
        maximum_keyframes: usize,
        free_delta: i32,
        occupied_delta: i32,
    },
}

impl std::fmt::Display for OccupancyConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroSamplingBlock => {
                write!(f, "occupancy nearest-depth sampling block must be nonzero")
            }
            Self::ZeroMaximumKeyframes => {
                write!(f, "occupancy maximum retained keyframes must be nonzero")
            }
            Self::EvidenceAccumulatorMayOverflow {
                maximum_keyframes,
                free_delta,
                occupied_delta,
            } => write!(
                f,
                "occupancy evidence bound does not fit i32: {maximum_keyframes} keyframes with deltas {free_delta} (free) and +{occupied_delta} (occupied)"
            ),
        }
    }
}

impl std::error::Error for OccupancyConfigError {}

impl OccupancyConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        geometry: OccupancyGridGeometry,
        world_to_occupancy: WorldToOccupancy,
        camera: DepthCameraModel,
        height_range: HeightRangeMeters,
        depth_range: DepthRangeMeters,
        sampling_block: u32,
        evidence: OccupancyEvidenceModel,
        max_keyframes: usize,
    ) -> Result<Self, OccupancyConfigError> {
        let sampling_block =
            NonZeroU32::new(sampling_block).ok_or(OccupancyConfigError::ZeroSamplingBlock)?;
        let max_keyframes =
            NonZeroUsize::new(max_keyframes).ok_or(OccupancyConfigError::ZeroMaximumKeyframes)?;
        let bounded = u128::try_from(max_keyframes.get())
            .ok()
            .and_then(|count| {
                let free_magnitude =
                    count.checked_mul(u128::from(evidence.free_delta.unsigned_abs()))?;
                let occupied_magnitude =
                    count.checked_mul(u128::from(evidence.occupied_delta.unsigned_abs()))?;
                Some(
                    free_magnitude <= u128::from(i32::MIN.unsigned_abs())
                        && occupied_magnitude <= u128::from(i32::MAX.unsigned_abs()),
                )
            })
            .unwrap_or(false);
        if !bounded {
            return Err(OccupancyConfigError::EvidenceAccumulatorMayOverflow {
                maximum_keyframes: max_keyframes.get(),
                free_delta: evidence.free_delta,
                occupied_delta: evidence.occupied_delta,
            });
        }
        Ok(Self {
            geometry,
            world_to_occupancy,
            camera,
            height_range,
            depth_range,
            sampling_block,
            evidence,
            max_keyframes,
        })
    }

    pub fn geometry(&self) -> OccupancyGridGeometry {
        self.geometry
    }

    pub fn world_to_occupancy(&self) -> WorldToOccupancy {
        self.world_to_occupancy
    }

    pub fn camera(&self) -> DepthCameraModel {
        self.camera
    }

    pub fn height_range(&self) -> HeightRangeMeters {
        self.height_range
    }

    pub fn depth_range(&self) -> DepthRangeMeters {
        self.depth_range
    }

    pub fn sampling_block(&self) -> u32 {
        self.sampling_block.get()
    }

    pub fn evidence(&self) -> OccupancyEvidenceModel {
        self.evidence
    }

    pub fn max_keyframes(&self) -> usize {
        self.max_keyframes.get()
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OccupancyCell {
    Unknown = 0,
    Free = 1,
    Occupied = 2,
}

impl OccupancyCell {
    pub const fn class_id(self) -> u8 {
        self as u8
    }

    fn from_class_id(class_id: u8) -> Self {
        match class_id {
            0 => Self::Unknown,
            1 => Self::Free,
            2 => Self::Occupied,
            _ => unreachable!("occupancy snapshots contain only mapper-produced class IDs"),
        }
    }
}

/// Row zero corresponds to the grid's lower `y` bound. Increasing row indices
/// move in the positive occupancy-frame `y` direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OccupancyRowOrder {
    IncreasingOccupancyY,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OccupancyGridMetadata {
    width: u32,
    height: u32,
    resolution_m: f64,
    lower_bound_m: [f64; 2],
    world_to_occupancy: WorldToOccupancy,
    height_range: HeightRangeMeters,
    row_order: OccupancyRowOrder,
    map_instance_id: Option<MapInstanceId>,
    revision: u64,
}

impl OccupancyGridMetadata {
    pub fn width(self) -> u32 {
        self.width
    }

    pub fn height(self) -> u32 {
        self.height
    }

    pub fn resolution_m(self) -> f64 {
        self.resolution_m
    }

    pub fn lower_bound_m(self) -> [f64; 2] {
        self.lower_bound_m
    }

    pub fn world_to_occupancy(self) -> WorldToOccupancy {
        self.world_to_occupancy
    }

    pub fn height_range(self) -> HeightRangeMeters {
        self.height_range
    }

    pub fn row_order(self) -> OccupancyRowOrder {
        self.row_order
    }

    pub fn map_instance_id(self) -> Option<MapInstanceId> {
        self.map_instance_id
    }

    pub fn revision(self) -> u64 {
        self.revision
    }
}

/// Immutable, self-describing row-major occupancy result for visualization or
/// export adapters. Class IDs are `0=unknown`, `1=free`, `2=occupied`.
///
/// A Rerun adapter can consume this value with [`Self::into_parts`] and move the
/// returned byte vector directly into a segmentation image without another
/// full-grid copy.
#[derive(Debug)]
pub struct OccupancyGridSnapshot {
    class_ids: Vec<u8>,
    metadata: OccupancyGridMetadata,
}

impl OccupancyGridSnapshot {
    pub fn class_ids(&self) -> &[u8] {
        self.class_ids.as_slice()
    }

    pub fn metadata(&self) -> OccupancyGridMetadata {
        self.metadata
    }

    pub fn into_parts(self) -> (OccupancyGridMetadata, Vec<u8>) {
        (self.metadata, self.class_ids)
    }

    pub fn width(&self) -> u32 {
        self.metadata.width()
    }

    pub fn height(&self) -> u32 {
        self.metadata.height()
    }

    pub fn resolution_m(&self) -> f64 {
        self.metadata.resolution_m()
    }

    pub fn lower_bound_m(&self) -> [f64; 2] {
        self.metadata.lower_bound_m()
    }

    pub fn world_to_occupancy(&self) -> WorldToOccupancy {
        self.metadata.world_to_occupancy()
    }

    pub fn height_range(&self) -> HeightRangeMeters {
        self.metadata.height_range()
    }

    pub fn row_order(&self) -> OccupancyRowOrder {
        self.metadata.row_order()
    }

    pub fn map_instance_id(&self) -> Option<MapInstanceId> {
        self.metadata.map_instance_id()
    }

    pub fn revision(&self) -> u64 {
        self.metadata.revision()
    }

    pub fn cell(&self, column: u32, row: u32) -> Option<OccupancyCell> {
        if column >= self.width() || row >= self.height() {
            return None;
        }
        let index = row as usize * self.width() as usize + column as usize;
        self.class_ids
            .get(index)
            .copied()
            .map(OccupancyCell::from_class_id)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OccupancyIntegrateOutcome {
    pub evicted_keyframe: Option<KeyframeId>,
    pub retained_keyframes: usize,
    pub sampled_blocks: usize,
    pub free_cells_touched: usize,
    pub occupied_cells_touched: usize,
    pub revision: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OccupancyRemoveOutcome {
    NotStored,
    Removed {
        retained_keyframes: usize,
        revision: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OccupancyRebuildOutcome {
    pub retained_keyframes: usize,
    pub revision: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OccupancyPoseUpdateOutcome {
    pub updated_keyframes: usize,
    pub not_stored_keyframes: usize,
    pub revision: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OccupancyResetOutcome {
    pub cleared_keyframes: usize,
    pub active_map: MapInstanceId,
    pub revision: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OccupancyError {
    AllocationFailed {
        context: &'static str,
        requested: usize,
    },
    DepthDimensionsMismatch {
        expected: FrameDimensions,
        actual: FrameDimensions,
    },
    DuplicateKeyframe {
        keyframe_id: KeyframeId,
    },
    MapMismatch {
        active: MapInstanceId,
        received: MapInstanceId,
    },
    MixedMapBatch {
        expected: MapInstanceId,
        received: MapInstanceId,
    },
    DuplicateRebuildPose {
        keyframe_id: KeyframeId,
    },
    DuplicatePoseUpdate {
        keyframe_id: KeyframeId,
    },
    MissingRebuildPose {
        keyframe_id: KeyframeId,
    },
    RevisionExhausted,
    NonFiniteTransform {
        stage: &'static str,
        axis: usize,
        value: f64,
    },
    NonFiniteProjectedPoint {
        axis: usize,
        value: f64,
    },
    EvidenceOverflow {
        cell_index: usize,
        current: i32,
        delta: i64,
    },
    RayTraversalInvariant {
        start_column: usize,
        start_row: usize,
        end_column: usize,
        end_row: usize,
    },
    ClippedRayEndpointOutsideGrid {
        endpoint: &'static str,
        x_m: f64,
        y_m: f64,
    },
}

impl std::fmt::Display for OccupancyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AllocationFailed { context, requested } => write!(
                f,
                "occupancy allocation failed for {context} ({requested} elements)"
            ),
            Self::DepthDimensionsMismatch { expected, actual } => write!(
                f,
                "depth dimensions do not match occupancy camera model: expected {}x{}, got {}x{}",
                expected.width(),
                expected.height(),
                actual.width(),
                actual.height()
            ),
            Self::DuplicateKeyframe { keyframe_id } => {
                write!(
                    f,
                    "occupancy keyframe {keyframe_id:?} was already integrated"
                )
            }
            Self::MapMismatch { active, received } => write!(
                f,
                "occupancy command belongs to map {}, but active map is {}",
                received.as_u64(),
                active.as_u64()
            ),
            Self::MixedMapBatch { expected, received } => write!(
                f,
                "occupancy batch combines map {} with map {}",
                expected.as_u64(),
                received.as_u64()
            ),
            Self::DuplicateRebuildPose { keyframe_id } => write!(
                f,
                "occupancy rebuild contains duplicate pose for {keyframe_id:?}"
            ),
            Self::DuplicatePoseUpdate { keyframe_id } => write!(
                f,
                "occupancy pose-update batch contains duplicate pose for {keyframe_id:?}"
            ),
            Self::MissingRebuildPose { keyframe_id } => write!(
                f,
                "occupancy rebuild is missing retained keyframe {keyframe_id:?}"
            ),
            Self::RevisionExhausted => write!(f, "occupancy revision space is exhausted"),
            Self::NonFiniteTransform { stage, axis, value } => write!(
                f,
                "occupancy {stage} transform produced non-finite component {axis}: {value}"
            ),
            Self::NonFiniteProjectedPoint { axis, value } => write!(
                f,
                "occupancy projection produced non-finite coordinate {axis}: {value}"
            ),
            Self::EvidenceOverflow {
                cell_index,
                current,
                delta,
            } => write!(
                f,
                "occupancy evidence overflow at cell {cell_index}: {current} + {delta}"
            ),
            Self::RayTraversalInvariant {
                start_column,
                start_row,
                end_column,
                end_row,
            } => write!(
                f,
                "bounded occupancy ray traversal did not reach ({end_column},{end_row}) from ({start_column},{start_row})"
            ),
            Self::ClippedRayEndpointOutsideGrid { endpoint, x_m, y_m } => write!(
                f,
                "occupancy clipping produced an unaddressable {endpoint} endpoint at [{x_m}, {y_m}] m"
            ),
        }
    }
}

impl std::error::Error for OccupancyError {}

#[derive(Clone, Copy, Debug)]
struct DepthSample {
    column: u32,
    row: u32,
    depth_m: f32,
}

#[derive(Clone, Debug)]
struct SampledDepthSource {
    samples: Arc<Vec<DepthSample>>,
}

impl SampledDepthSource {
    fn parse(depth: &DepthImage, config: &OccupancyConfig) -> Result<Self, OccupancyError> {
        let actual = depth.dimensions();
        let expected = config.camera.dimensions();
        if actual != expected {
            return Err(OccupancyError::DepthDimensionsMismatch { expected, actual });
        }

        let width = actual.width();
        let height = actual.height();
        let block = config.sampling_block.get();
        let block_columns = width.div_ceil(block);
        let block_rows = height.div_ceil(block);
        let capacity = usize::try_from(block_columns)
            .ok()
            .and_then(|columns| {
                usize::try_from(block_rows)
                    .ok()
                    .and_then(|rows| columns.checked_mul(rows))
            })
            .ok_or(OccupancyError::AllocationFailed {
                context: "sampled depth blocks",
                requested: usize::MAX,
            })?;
        let mut samples = Vec::new();
        try_reserve(&mut samples, capacity, "sampled depth blocks")?;
        let width_usize = width as usize;
        let values = depth.depth_m();

        let mut block_row = 0_u32;
        while block_row < height {
            let row_end = block_row.saturating_add(block).min(height);
            let mut block_column = 0_u32;
            while block_column < width {
                let column_end = block_column.saturating_add(block).min(width);
                let mut nearest: Option<DepthSample> = None;
                for row in block_row..row_end {
                    let row_offset = row as usize * width_usize;
                    for column in block_column..column_end {
                        let depth_m = values[row_offset + column as usize];
                        if depth_m == 0.0 || !config.depth_range.contains(f64::from(depth_m)) {
                            continue;
                        }
                        if nearest.is_none_or(|current| depth_m < current.depth_m) {
                            nearest = Some(DepthSample {
                                column,
                                row,
                                depth_m,
                            });
                        }
                    }
                }
                if let Some(sample) = nearest {
                    samples.push(sample);
                }
                block_column = block_column.saturating_add(block);
            }
            block_row = block_row.saturating_add(block);
        }

        Ok(Self {
            // Keep the already-filled sample vector shared for transactional
            // pose updates and rebuilds.
            samples: Arc::new(samples),
        })
    }

    fn samples(&self) -> &[DepthSample] {
        self.samples.as_slice()
    }
}

#[derive(Clone, Debug)]
struct StoredKeyframe {
    pose: WorldToCamera,
    source: SampledDepthSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CellUpdate {
    index: usize,
    delta: i32,
}

const SCRATCH_UNSEEN: u8 = 0;
const SCRATCH_FREE: u8 = 1;
const SCRATCH_OCCUPIED: u8 = 2;

#[derive(Debug)]
struct KeyframeContribution {
    updates: Vec<CellUpdate>,
    free_cells: usize,
    occupied_cells: usize,
}

/// A deterministic, fixed-bounds occupancy mapper.
pub struct OccupancyMapper {
    config: OccupancyConfig,
    evidence: Vec<i32>,
    scratch: Vec<u8>,
    touched: Vec<usize>,
    column_rays: Vec<f64>,
    row_rays: Vec<f64>,
    stored: HashMap<KeyframeId, StoredKeyframe>,
    order: VecDeque<KeyframeId>,
    map_instance_id: Option<MapInstanceId>,
    revision: u64,
}

impl OccupancyMapper {
    pub fn try_new(config: OccupancyConfig) -> Result<Self, OccupancyError> {
        let cell_count = config.geometry.cell_count();
        let mut evidence = Vec::new();
        try_reserve(&mut evidence, cell_count, "evidence cells")?;
        evidence.resize(cell_count, 0_i32);
        let mut scratch = Vec::new();
        try_reserve(&mut scratch, cell_count, "cell classification scratch")?;
        scratch.resize(cell_count, SCRATCH_UNSEEN);

        let dimensions = config.camera.dimensions();
        let width = dimensions.width() as usize;
        let height = dimensions.height() as usize;
        let intrinsics = config.camera.intrinsics();
        let mut column_rays = Vec::new();
        try_reserve(&mut column_rays, width, "depth column rays")?;
        for column in 0..dimensions.width() {
            column_rays.push(
                (f64::from(column) - f64::from(intrinsics.cx())) / f64::from(intrinsics.fx()),
            );
        }
        let mut row_rays = Vec::new();
        try_reserve(&mut row_rays, height, "depth row rays")?;
        for row in 0..dimensions.height() {
            row_rays
                .push((f64::from(row) - f64::from(intrinsics.cy())) / f64::from(intrinsics.fy()));
        }

        let mut stored = HashMap::new();
        stored.try_reserve(config.max_keyframes()).map_err(|_| {
            OccupancyError::AllocationFailed {
                context: "retained keyframes",
                requested: config.max_keyframes(),
            }
        })?;
        let mut order = VecDeque::new();
        order.try_reserve(config.max_keyframes()).map_err(|_| {
            OccupancyError::AllocationFailed {
                context: "retained keyframe order",
                requested: config.max_keyframes(),
            }
        })?;

        let mut touched = Vec::new();
        try_reserve(&mut touched, cell_count, "touched-cell indices")?;

        Ok(Self {
            config,
            evidence,
            scratch,
            touched,
            column_rays,
            row_rays,
            stored,
            order,
            map_instance_id: None,
            revision: 0,
        })
    }

    pub fn config(&self) -> &OccupancyConfig {
        &self.config
    }

    pub fn retained_keyframes(&self) -> usize {
        self.stored.len()
    }

    pub fn map_instance_id(&self) -> Option<MapInstanceId> {
        self.map_instance_id
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn integrate(
        &mut self,
        keyframe_id: KeyframeId,
        pose: WorldToCamera,
        depth: &DepthImage,
    ) -> Result<OccupancyIntegrateOutcome, OccupancyError> {
        if self.stored.contains_key(&keyframe_id) {
            return Err(OccupancyError::DuplicateKeyframe { keyframe_id });
        }
        let received_map = keyframe_id.map_instance_id();
        if let Some(active) = self.map_instance_id
            && active != received_map
        {
            return Err(OccupancyError::MapMismatch {
                active,
                received: received_map,
            });
        }
        let next_revision = self.next_revision()?;
        let source = SampledDepthSource::parse(depth, &self.config)?;
        let contribution = self.contribution(pose, &source)?;

        let evicted = if self.stored.len() == self.config.max_keyframes() {
            self.order.front().copied()
        } else {
            None
        };
        let evicted_contribution = if let Some(evicted_id) = evicted {
            let old = self
                .stored
                .get(&evicted_id)
                .expect("retained keyframe order must reference stored data")
                .clone();
            Some(self.contribution(old.pose, &old.source)?)
        } else {
            None
        };
        let staged = stage_evidence_changes(
            &self.evidence,
            evicted_contribution.as_ref(),
            Some(&contribution),
        )?;

        apply_staged_changes(&mut self.evidence, &staged);
        if let Some(evicted_id) = evicted {
            let front = self
                .order
                .pop_front()
                .expect("capacity eviction requires an oldest keyframe");
            debug_assert_eq!(front, evicted_id);
            self.stored
                .remove(&evicted_id)
                .expect("capacity eviction keyframe must be stored");
        }
        let sampled_blocks = source.samples().len();
        let replaced = self
            .stored
            .insert(keyframe_id, StoredKeyframe { pose, source });
        debug_assert!(replaced.is_none(), "duplicate integration was preflighted");
        self.order.push_back(keyframe_id);
        self.map_instance_id = Some(received_map);
        self.revision = next_revision;

        Ok(OccupancyIntegrateOutcome {
            evicted_keyframe: evicted,
            retained_keyframes: self.stored.len(),
            sampled_blocks,
            free_cells_touched: contribution.free_cells,
            occupied_cells_touched: contribution.occupied_cells,
            revision: next_revision,
        })
    }

    pub fn remove(
        &mut self,
        keyframe_id: KeyframeId,
    ) -> Result<OccupancyRemoveOutcome, OccupancyError> {
        let received_map = keyframe_id.map_instance_id();
        if let Some(active) = self.map_instance_id
            && active != received_map
        {
            return Err(OccupancyError::MapMismatch {
                active,
                received: received_map,
            });
        }
        let Some(stored) = self.stored.get(&keyframe_id).cloned() else {
            return Ok(OccupancyRemoveOutcome::NotStored);
        };
        let next_revision = self.next_revision()?;
        let contribution = self.contribution(stored.pose, &stored.source)?;
        let staged = stage_evidence_changes(&self.evidence, Some(&contribution), None)?;

        apply_staged_changes(&mut self.evidence, &staged);
        self.stored
            .remove(&keyframe_id)
            .expect("stored keyframe existence was preflighted");
        self.order.retain(|candidate| *candidate != keyframe_id);
        self.revision = next_revision;
        Ok(OccupancyRemoveOutcome::Removed {
            retained_keyframes: self.stored.len(),
            revision: next_revision,
        })
    }

    /// Transactionally replace poses for retained keyframes after a partial BA
    /// correction. Unknown IDs are counted as no-ops because their integration
    /// may legitimately have been dropped upstream.
    pub fn update_poses(
        &mut self,
        corrected_poses: &[(KeyframeId, WorldToCamera)],
    ) -> Result<OccupancyPoseUpdateOutcome, OccupancyError> {
        let batch_map = corrected_poses
            .first()
            .map(|(keyframe_id, _)| keyframe_id.map_instance_id());

        let mut unique = HashSet::new();
        unique.try_reserve(corrected_poses.len()).map_err(|_| {
            OccupancyError::AllocationFailed {
                context: "partial occupancy pose updates",
                requested: corrected_poses.len(),
            }
        })?;
        for &(keyframe_id, _) in corrected_poses {
            if let Some(expected) = batch_map
                && keyframe_id.map_instance_id() != expected
            {
                return Err(OccupancyError::MixedMapBatch {
                    expected,
                    received: keyframe_id.map_instance_id(),
                });
            }
            if !unique.insert(keyframe_id) {
                return Err(OccupancyError::DuplicatePoseUpdate { keyframe_id });
            }
        }
        if let (Some(active), Some(received)) = (self.map_instance_id, batch_map)
            && active != received
        {
            return Err(OccupancyError::MapMismatch { active, received });
        }

        let mut retained = Vec::new();
        try_reserve(
            &mut retained,
            corrected_poses.len().min(self.stored.len()),
            "retained occupancy pose updates",
        )?;
        for &(keyframe_id, new_pose) in corrected_poses {
            if let Some(old) = self.stored.get(&keyframe_id) {
                retained.push((keyframe_id, old.pose, new_pose, old.source.clone()));
            }
        }
        let not_stored_keyframes = corrected_poses.len().saturating_sub(retained.len());
        if retained.is_empty() {
            return Ok(OccupancyPoseUpdateOutcome {
                updated_keyframes: 0,
                not_stored_keyframes,
                revision: self.revision,
            });
        }

        let next_revision = self.next_revision()?;
        let mut staged_evidence = Vec::new();
        try_reserve(
            &mut staged_evidence,
            self.evidence.len(),
            "partial pose-update evidence",
        )?;
        staged_evidence.extend_from_slice(&self.evidence);
        for (_, old_pose, new_pose, source) in &retained {
            let old = self.contribution(*old_pose, source)?;
            let new = self.contribution(*new_pose, source)?;
            let changes = stage_evidence_changes(&staged_evidence, Some(&old), Some(&new))?;
            apply_staged_changes(&mut staged_evidence, &changes);
        }

        for (keyframe_id, _, new_pose, _) in &retained {
            self.stored
                .get_mut(keyframe_id)
                .expect("retained pose-update keyframe existence was preflighted")
                .pose = *new_pose;
        }
        self.evidence = staged_evidence;
        self.revision = next_revision;
        Ok(OccupancyPoseUpdateOutcome {
            updated_keyframes: retained.len(),
            not_stored_keyframes,
            revision: next_revision,
        })
    }

    /// Rebuild from a complete pose set for every retained keyframe.
    /// Additional poses are accepted only when they belong to the same map.
    pub fn rebuild(
        &mut self,
        corrected_poses: &[(KeyframeId, WorldToCamera)],
    ) -> Result<OccupancyRebuildOutcome, OccupancyError> {
        let next_revision = self.next_revision()?;
        let batch_map = corrected_poses
            .first()
            .map(|(keyframe_id, _)| keyframe_id.map_instance_id());
        let mut poses = HashMap::new();
        poses
            .try_reserve(corrected_poses.len())
            .map_err(|_| OccupancyError::AllocationFailed {
                context: "corrected occupancy poses",
                requested: corrected_poses.len(),
            })?;
        for &(keyframe_id, pose) in corrected_poses {
            if let Some(expected) = batch_map
                && keyframe_id.map_instance_id() != expected
            {
                return Err(OccupancyError::MixedMapBatch {
                    expected,
                    received: keyframe_id.map_instance_id(),
                });
            }
            if poses.insert(keyframe_id, pose).is_some() {
                return Err(OccupancyError::DuplicateRebuildPose { keyframe_id });
            }
        }
        if let (Some(active), Some(received)) = (self.map_instance_id, batch_map)
            && active != received
        {
            return Err(OccupancyError::MapMismatch { active, received });
        }
        let inferred_map = self.map_instance_id.or(batch_map);

        let mut staged_sources = Vec::new();
        try_reserve(
            &mut staged_sources,
            self.stored.len(),
            "occupancy rebuild sources",
        )?;
        for &keyframe_id in &self.order {
            let pose = poses
                .get(&keyframe_id)
                .copied()
                .ok_or(OccupancyError::MissingRebuildPose { keyframe_id })?;
            let source = self
                .stored
                .get(&keyframe_id)
                .expect("retained keyframe order must reference stored data")
                .source
                .clone();
            staged_sources.push((keyframe_id, pose, source));
        }

        let mut rebuilt = Vec::new();
        try_reserve(
            &mut rebuilt,
            self.evidence.len(),
            "rebuilt occupancy evidence",
        )?;
        rebuilt.resize(self.evidence.len(), 0_i32);
        for (_, pose, source) in &staged_sources {
            let contribution = self.contribution(*pose, source)?;
            apply_contribution_checked(&mut rebuilt, &contribution)?;
        }

        for (keyframe_id, pose, _) in staged_sources {
            self.stored
                .get_mut(&keyframe_id)
                .expect("rebuild keyframe existence was preflighted")
                .pose = pose;
        }
        self.evidence = rebuilt;
        self.map_instance_id = inferred_map;
        self.revision = next_revision;
        Ok(OccupancyRebuildOutcome {
            retained_keyframes: self.stored.len(),
            revision: next_revision,
        })
    }

    /// Clear all evidence and establish the new mapping-session identity.
    /// Late commands from the previous map remain ineligible even while this
    /// new map is still empty.
    pub fn reset_to_map(
        &mut self,
        new_map: MapInstanceId,
    ) -> Result<OccupancyResetOutcome, OccupancyError> {
        let next_revision = self.next_revision()?;
        let cleared_keyframes = self.stored.len();
        self.evidence.fill(0);
        self.clear_scratch();
        self.stored.clear();
        self.order.clear();
        self.map_instance_id = Some(new_map);
        self.revision = next_revision;
        Ok(OccupancyResetOutcome {
            cleared_keyframes,
            active_map: new_map,
            revision: next_revision,
        })
    }

    pub fn snapshot(&self) -> Result<OccupancyGridSnapshot, OccupancyError> {
        let mut class_ids = Vec::new();
        try_reserve(
            &mut class_ids,
            self.evidence.len(),
            "occupancy snapshot cells",
        )?;
        class_ids.extend(
            self.evidence
                .iter()
                .copied()
                .map(|value| self.config.evidence.classify(value).class_id()),
        );
        let geometry = self.config.geometry;
        Ok(OccupancyGridSnapshot {
            class_ids,
            metadata: OccupancyGridMetadata {
                width: geometry.width(),
                height: geometry.height(),
                resolution_m: geometry.resolution_m(),
                lower_bound_m: geometry.lower_bound_m(),
                world_to_occupancy: self.config.world_to_occupancy,
                height_range: self.config.height_range,
                row_order: OccupancyRowOrder::IncreasingOccupancyY,
                map_instance_id: self.map_instance_id,
                revision: self.revision,
            },
        })
    }

    fn next_revision(&self) -> Result<u64, OccupancyError> {
        self.revision
            .checked_add(1)
            .ok_or(OccupancyError::RevisionExhausted)
    }

    fn contribution(
        &mut self,
        pose: WorldToCamera,
        source: &SampledDepthSource,
    ) -> Result<KeyframeContribution, OccupancyError> {
        self.clear_scratch();
        let transform = depth_to_occupancy_transform(&self.config, pose)?;
        let origin = transform.translation;
        let classify_result = (|| {
            for sample in source.samples() {
                let depth_m = f64::from(sample.depth_m);
                let camera_point = [
                    self.column_rays[sample.column as usize] * depth_m,
                    self.row_rays[sample.row as usize] * depth_m,
                    depth_m,
                ];
                let endpoint = transform.transform_point(camera_point)?;
                self.mark_ray(origin, endpoint)?;
            }
            Ok(())
        })();
        if let Err(error) = classify_result {
            self.clear_scratch();
            return Err(error);
        }
        self.finish_contribution()
    }

    fn finish_contribution(&mut self) -> Result<KeyframeContribution, OccupancyError> {
        let free_cells = self
            .touched
            .iter()
            .filter(|&&index| self.scratch[index] == SCRATCH_FREE)
            .count();
        let occupied_cells = self.touched.len() - free_cells;
        let mut updates = Vec::new();
        if let Err(error) = try_reserve(
            &mut updates,
            self.touched.len(),
            "keyframe cell contribution",
        ) {
            self.clear_scratch();
            return Err(error);
        }
        for &index in &self.touched {
            let delta = match self.scratch[index] {
                SCRATCH_FREE => self.config.evidence.free_delta,
                SCRATCH_OCCUPIED => self.config.evidence.occupied_delta,
                _ => unreachable!("touched occupancy cell must be classified"),
            };
            updates.push(CellUpdate { index, delta });
        }
        updates.sort_unstable_by_key(|update| update.index);
        self.clear_scratch();
        Ok(KeyframeContribution {
            updates,
            free_cells,
            occupied_cells,
        })
    }

    fn clear_scratch(&mut self) {
        for index in self.touched.drain(..) {
            self.scratch[index] = SCRATCH_UNSEEN;
        }
    }

    fn mark_free(&mut self, index: usize) {
        if self.scratch[index] == SCRATCH_UNSEEN {
            self.scratch[index] = SCRATCH_FREE;
            self.touched.push(index);
        }
    }

    fn mark_occupied(&mut self, index: usize) {
        if self.scratch[index] == SCRATCH_UNSEEN {
            self.touched.push(index);
        }
        self.scratch[index] = SCRATCH_OCCUPIED;
    }

    fn mark_ray(&mut self, origin: [f64; 3], endpoint: [f64; 3]) -> Result<(), OccupancyError> {
        let geometry = self.config.geometry;
        let height = self.config.height_range;
        let direction = checked_difference(endpoint, origin)?;
        let mut interval = [0.0_f64, 1.0_f64];
        if clip_axis(
            origin[2],
            direction[2],
            height.minimum_m,
            height.maximum_m,
            &mut interval,
        ) && clip_axis(
            origin[0],
            direction[0],
            geometry.lower_bound_m[0],
            geometry.upper_bound_m[0],
            &mut interval,
        ) && clip_axis(
            origin[1],
            direction[1],
            geometry.lower_bound_m[1],
            geometry.upper_bound_m[1],
            &mut interval,
        ) {
            let start = geometry.clamp_to_closed_bounds([
                origin[0] + direction[0] * interval[0],
                origin[1] + direction[1] * interval[0],
            ]);
            let end = geometry.clamp_to_closed_bounds([
                origin[0] + direction[0] * interval[1],
                origin[1] + direction[1] * interval[1],
            ]);
            self.traverse_free_segment(start, end)?;
        }

        if height.contains(endpoint[2])
            && let Some(index) = geometry.point_index([endpoint[0], endpoint[1]])
        {
            self.mark_occupied(index);
        }
        Ok(())
    }

    fn traverse_free_segment(
        &mut self,
        start: [f64; 2],
        end: [f64; 2],
    ) -> Result<(), OccupancyError> {
        let geometry = self.config.geometry;
        let (mut column, mut row) = geometry.traversal_cell(start).ok_or(
            OccupancyError::ClippedRayEndpointOutsideGrid {
                endpoint: "start",
                x_m: start[0],
                y_m: start[1],
            },
        )?;
        let (end_column, end_row) =
            geometry
                .traversal_cell(end)
                .ok_or(OccupancyError::ClippedRayEndpointOutsideGrid {
                    endpoint: "end",
                    x_m: end[0],
                    y_m: end[1],
                })?;
        let start_column = column;
        let start_row = row;
        let width = geometry.width() as usize;
        self.mark_free(row * width + column);
        if column == end_column && row == end_row {
            return Ok(());
        }

        let delta_x = end[0] - start[0];
        let delta_y = end[1] - start[1];
        let (column_step, mut next_x, delta_t_x) = if delta_x > 0.0 {
            let boundary = geometry.lower_bound_m[0] + (column + 1) as f64 * geometry.resolution_m;
            (
                1_isize,
                (boundary - start[0]) / delta_x,
                geometry.resolution_m / delta_x,
            )
        } else if delta_x < 0.0 {
            let boundary = geometry.lower_bound_m[0] + column as f64 * geometry.resolution_m;
            (
                -1_isize,
                (boundary - start[0]) / delta_x,
                -geometry.resolution_m / delta_x,
            )
        } else {
            (0_isize, f64::INFINITY, f64::INFINITY)
        };
        let (row_step, mut next_y, delta_t_y) = if delta_y > 0.0 {
            let boundary = geometry.lower_bound_m[1] + (row + 1) as f64 * geometry.resolution_m;
            (
                1_isize,
                (boundary - start[1]) / delta_y,
                geometry.resolution_m / delta_y,
            )
        } else if delta_y < 0.0 {
            let boundary = geometry.lower_bound_m[1] + row as f64 * geometry.resolution_m;
            (
                -1_isize,
                (boundary - start[1]) / delta_y,
                -geometry.resolution_m / delta_y,
            )
        } else {
            (0_isize, f64::INFINITY, f64::INFINITY)
        };

        let maximum_steps = (geometry.width() as usize)
            .saturating_add(geometry.height() as usize)
            .saturating_add(1);
        for _ in 0..maximum_steps {
            match crossing_time_order(next_x, next_y) {
                std::cmp::Ordering::Less => {
                    column = column.saturating_add_signed(column_step);
                    next_x += delta_t_x;
                }
                std::cmp::Ordering::Greater => {
                    row = row.saturating_add_signed(row_step);
                    next_y += delta_t_y;
                }
                std::cmp::Ordering::Equal => {
                    // A corner has two side-adjacent candidates. Select one by
                    // its direction-independent row-major index, then visit the
                    // diagonal cell on the following iteration. This keeps the
                    // traversed set reversible even when the two crossing times
                    // differ only by floating-point rounding.
                    let x_candidate = column.saturating_add_signed(column_step);
                    let y_candidate = row.saturating_add_signed(row_step);
                    let x_index = row.saturating_mul(width).saturating_add(x_candidate);
                    let y_index = y_candidate.saturating_mul(width).saturating_add(column);
                    if x_index <= y_index {
                        column = x_candidate;
                        next_x += delta_t_x;
                    } else {
                        row = y_candidate;
                        next_y += delta_t_y;
                    }
                }
            }
            if column >= geometry.width() as usize || row >= geometry.height() as usize {
                break;
            }
            self.mark_free(row * width + column);
            if column == end_column && row == end_row {
                return Ok(());
            }
        }
        Err(OccupancyError::RayTraversalInvariant {
            start_column,
            start_row,
            end_column,
            end_row,
        })
    }
}

fn try_reserve<T>(
    values: &mut Vec<T>,
    additional: usize,
    context: &'static str,
) -> Result<(), OccupancyError> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| OccupancyError::AllocationFailed {
            context,
            requested: additional,
        })
}

fn stage_evidence_changes(
    current: &[i32],
    removed: Option<&KeyframeContribution>,
    added: Option<&KeyframeContribution>,
) -> Result<Vec<(usize, i32)>, OccupancyError> {
    let removed = removed.map_or(&[][..], |value| value.updates.as_slice());
    let added = added.map_or(&[][..], |value| value.updates.as_slice());
    let mut staged = Vec::new();
    try_reserve(
        &mut staged,
        removed.len().saturating_add(added.len()),
        "staged evidence changes",
    )?;
    let mut removed_index = 0;
    let mut added_index = 0;
    while removed_index < removed.len() || added_index < added.len() {
        let next_removed = removed.get(removed_index);
        let next_added = added.get(added_index);
        let cell_index = match (next_removed, next_added) {
            (Some(left), Some(right)) => left.index.min(right.index),
            (Some(left), None) => left.index,
            (None, Some(right)) => right.index,
            (None, None) => break,
        };
        let mut delta = 0_i64;
        if next_removed.is_some_and(|update| update.index == cell_index) {
            delta -= i64::from(next_removed.expect("removed update exists").delta);
            removed_index += 1;
        }
        if next_added.is_some_and(|update| update.index == cell_index) {
            delta += i64::from(next_added.expect("added update exists").delta);
            added_index += 1;
        }
        let value = i64::from(current[cell_index]) + delta;
        let value = i32::try_from(value).map_err(|_| OccupancyError::EvidenceOverflow {
            cell_index,
            current: current[cell_index],
            delta,
        })?;
        staged.push((cell_index, value));
    }
    Ok(staged)
}

fn apply_staged_changes(evidence: &mut [i32], staged: &[(usize, i32)]) {
    for &(index, value) in staged {
        evidence[index] = value;
    }
}

fn apply_contribution_checked(
    evidence: &mut [i32],
    contribution: &KeyframeContribution,
) -> Result<(), OccupancyError> {
    for update in &contribution.updates {
        evidence[update.index] = evidence[update.index].checked_add(update.delta).ok_or(
            OccupancyError::EvidenceOverflow {
                cell_index: update.index,
                current: evidence[update.index],
                delta: i64::from(update.delta),
            },
        )?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct RigidTransform64 {
    rotation: [[f64; 3]; 3],
    translation: [f64; 3],
}

impl RigidTransform64 {
    fn from_pose(pose: Pose) -> Self {
        Self {
            rotation: pose.rotation().map(|row| row.map(f64::from)),
            translation: pose.translation().map(f64::from),
        }
    }

    fn from_world_to_occupancy(transform: WorldToOccupancy) -> Self {
        Self {
            rotation: transform.rotation,
            translation: transform.translation_m,
        }
    }

    fn try_inverse(self, stage: &'static str) -> Result<Self, OccupancyError> {
        let rotation = transpose(self.rotation);
        let rotated = multiply_vector(rotation, self.translation);
        let translation = [-rotated[0], -rotated[1], -rotated[2]];
        validate_transform(stage, rotation, translation)?;
        Ok(Self {
            rotation,
            translation,
        })
    }

    /// Compose `self` after `other`.
    fn try_compose(self, other: Self, stage: &'static str) -> Result<Self, OccupancyError> {
        let rotation = multiply_matrix(self.rotation, other.rotation);
        let rotated_translation = multiply_vector(self.rotation, other.translation);
        let translation = [
            rotated_translation[0] + self.translation[0],
            rotated_translation[1] + self.translation[1],
            rotated_translation[2] + self.translation[2],
        ];
        validate_transform(stage, rotation, translation)?;
        Ok(Self {
            rotation,
            translation,
        })
    }

    fn transform_point(self, point: [f64; 3]) -> Result<[f64; 3], OccupancyError> {
        transform_point(self.rotation, self.translation, point)
    }
}

fn depth_to_occupancy_transform(
    config: &OccupancyConfig,
    world_to_tracking: WorldToCamera,
) -> Result<RigidTransform64, OccupancyError> {
    let tracking_to_world = RigidTransform64::from_pose(world_to_tracking.into_legacy_pose())
        .try_inverse("tracking-to-world")?;
    let depth_to_tracking = RigidTransform64::from_pose(config.camera.depth_to_tracking.pose());
    let depth_to_world = tracking_to_world.try_compose(depth_to_tracking, "depth-to-world")?;
    RigidTransform64::from_world_to_occupancy(config.world_to_occupancy)
        .try_compose(depth_to_world, "depth-to-occupancy")
}

fn validate_transform(
    stage: &'static str,
    rotation: [[f64; 3]; 3],
    translation: [f64; 3],
) -> Result<(), OccupancyError> {
    for (index, value) in rotation.into_iter().flatten().enumerate() {
        if !value.is_finite() {
            return Err(OccupancyError::NonFiniteTransform {
                stage,
                axis: index,
                value,
            });
        }
    }
    if let Some(axis) = translation.iter().position(|value| !value.is_finite()) {
        return Err(OccupancyError::NonFiniteTransform {
            stage,
            axis: 9 + axis,
            value: translation[axis],
        });
    }
    Ok(())
}

fn transform_point(
    rotation: [[f64; 3]; 3],
    translation: [f64; 3],
    point: [f64; 3],
) -> Result<[f64; 3], OccupancyError> {
    let rotated = multiply_vector(rotation, point);
    let result = [
        rotated[0] + translation[0],
        rotated[1] + translation[1],
        rotated[2] + translation[2],
    ];
    if let Some(axis) = result.iter().position(|value| !value.is_finite()) {
        return Err(OccupancyError::NonFiniteProjectedPoint {
            axis,
            value: result[axis],
        });
    }
    Ok(result)
}

fn checked_difference(end: [f64; 3], start: [f64; 3]) -> Result<[f64; 3], OccupancyError> {
    let difference = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
    if let Some(axis) = difference.iter().position(|value| !value.is_finite()) {
        return Err(OccupancyError::NonFiniteProjectedPoint {
            axis,
            value: difference[axis],
        });
    }
    Ok(difference)
}

fn clip_axis(
    origin: f64,
    direction: f64,
    minimum: f64,
    maximum: f64,
    interval: &mut [f64; 2],
) -> bool {
    if direction == 0.0 {
        return origin >= minimum && origin <= maximum;
    }
    let first = (minimum - origin) / direction;
    let second = (maximum - origin) / direction;
    let entry = first.min(second);
    let exit = first.max(second);
    interval[0] = interval[0].max(entry);
    interval[1] = interval[1].min(exit);
    interval[0] <= interval[1]
}

fn crossing_time_order(left: f64, right: f64) -> std::cmp::Ordering {
    const MAX_ULP_DISTANCE: u64 = 16;

    let ordering = left.total_cmp(&right);
    if ordering == std::cmp::Ordering::Equal || !left.is_finite() || !right.is_finite() {
        return ordering;
    }
    let left_key = ordered_f64_key(left);
    let right_key = ordered_f64_key(right);
    if left_key.abs_diff(right_key) <= MAX_ULP_DISTANCE {
        std::cmp::Ordering::Equal
    } else {
        ordering
    }
}

fn ordered_f64_key(value: f64) -> u64 {
    const SIGN: u64 = 1_u64 << 63;
    let bits = value.to_bits();
    if bits & SIGN == 0 { bits | SIGN } else { !bits }
}

fn determinant(rotation: [[f64; 3]; 3]) -> f64 {
    rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
}

fn transpose(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [matrix[0][0], matrix[1][0], matrix[2][0]],
        [matrix[0][1], matrix[1][1], matrix[2][1]],
        [matrix[0][2], matrix[1][2], matrix[2][2]],
    ]
}

fn multiply_matrix(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let right = transpose(right);
    left.map(|row| {
        right.map(|column| row[0].mul_add(column[0], row[1].mul_add(column[1], row[2] * column[2])))
    })
}

fn multiply_vector(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    matrix.map(|row| row[0].mul_add(vector[0], row[1].mul_add(vector[1], row[2] * vector[2])))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::{ImageSize, SlamMap};
    use crate::{FrameId, Keypoint, Timestamp};

    fn dimensions(width: u32, height: u32) -> FrameDimensions {
        FrameDimensions::try_new(width, height).expect("test dimensions")
    }

    fn test_config_with(
        width: u32,
        height: u32,
        sampling_block: u32,
        max_keyframes: usize,
    ) -> OccupancyConfig {
        let camera = DepthCameraModel::new(
            PinholeIntrinsics::try_new(100.0, 100.0, 0.0, 0.0).expect("test intrinsics"),
            dimensions(width, height),
            DepthToTrackingCamera::identity(),
        );
        OccupancyConfig::try_new(
            OccupancyGridGeometry::try_new(1.0, [-2.0, 0.0], 6, 6, 36).expect("test geometry"),
            WorldToOccupancy::level_optical_world(1.0).expect("test frame"),
            camera,
            HeightRangeMeters::try_new(0.0, 2.0).expect("test height range"),
            DepthRangeMeters::try_new(0.1, 10.0).expect("test depth range"),
            sampling_block,
            OccupancyEvidenceModel::try_new(-1, 3, -1, 1).expect("test evidence"),
            max_keyframes,
        )
        .expect("test config")
    }

    fn test_mapper(width: u32, height: u32) -> OccupancyMapper {
        OccupancyMapper::try_new(test_config_with(width, height, 1, 8)).expect("test mapper")
    }

    fn depth(width: u32, height: u32, values: Vec<f32>) -> DepthImage {
        DepthImage::new(
            FrameId::new(1),
            Timestamp::from_nanos(1),
            width,
            height,
            values,
        )
        .expect("test depth")
    }

    fn keyframe(index: usize) -> KeyframeId {
        KeyframeId::for_test(index)
    }

    fn mapped_keyframe(frame: u64) -> KeyframeId {
        let mut map = SlamMap::new();
        map.add_keyframe(
            FrameId::new(frame),
            Timestamp::from_nanos(i64::try_from(frame).expect("test frame fits i64")),
            WorldToCamera::identity(),
            ImageSize::try_new(1, 1).expect("test image size"),
            vec![Keypoint { x: 0.0, y: 0.0 }],
        )
        .expect("test mapped keyframe")
    }

    fn camera_at_world_x(world_x: f32) -> WorldToCamera {
        let pose = Pose::try_from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [-world_x, 0.0, 0.0],
        )
        .expect("test pose");
        WorldToCamera::from_legacy_pose(pose)
    }

    fn cells(mapper: &OccupancyMapper) -> Vec<u8> {
        mapper.snapshot().expect("snapshot").class_ids().to_vec()
    }

    #[test]
    fn level_optical_world_has_explicit_documented_axes_and_signs() {
        let frame = WorldToOccupancy::level_optical_world(1.5).expect("level frame");
        assert_eq!(
            frame
                .try_transform_world_point([2.0, 0.25, 4.0])
                .expect("finite transform"),
            [2.0, 4.0, 1.25]
        );
        assert!(matches!(
            WorldToOccupancy::level_optical_world(-0.1),
            Err(WorldToOccupancyError::InvalidLevelCameraHeight { .. })
        ));
    }

    #[test]
    fn world_to_camera_translation_is_inverted_before_projection() {
        let config = test_config_with(1, 1, 1, 2);
        let transform = depth_to_occupancy_transform(&config, camera_at_world_x(1.0))
            .expect("depth-to-occupancy transform");
        assert_eq!(transform.translation, [1.0, 0.0, 1.0]);
        let endpoint = transform
            .transform_point([0.0, 0.0, 2.0])
            .expect("endpoint");
        assert_eq!(endpoint, [1.0, 2.0, 1.0]);
    }

    #[test]
    fn depth_to_tracking_extrinsic_is_applied_before_world_transform() {
        let camera = DepthCameraModel::new(
            PinholeIntrinsics::try_new(100.0, 100.0, 0.0, 0.0).expect("intrinsics"),
            dimensions(1, 1),
            DepthToTrackingCamera::new(
                Pose::try_from_rt(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [0.5, 0.0, 0.0],
                )
                .expect("extrinsic"),
            ),
        );
        let mut config = test_config_with(1, 1, 1, 2);
        config.camera = camera;
        let transform =
            depth_to_occupancy_transform(&config, WorldToCamera::identity()).expect("transform");
        assert_eq!(transform.translation, [0.5, 0.0, 1.0]);
    }

    #[test]
    fn grid_bounds_are_lower_inclusive_upper_exclusive() {
        let geometry =
            OccupancyGridGeometry::try_new(1.0, [-1.0, -2.0], 2, 3, 6).expect("geometry");
        assert_eq!(geometry.point_index([-1.0, -2.0]), Some(0));
        assert_eq!(geometry.point_index([0.999_999, 0.999_999]), Some(5));
        assert_eq!(geometry.point_index([-1.000_001, -2.0]), None);
        assert_eq!(geometry.point_index([1.0, -2.0]), None);
        assert_eq!(geometry.point_index([-1.0, 1.0]), None);

        let default_geometry =
            OccupancyGridGeometry::try_new(0.05, [-10.0, -5.0], 400, 400, 160_000)
                .expect("default-like geometry");
        assert_eq!(
            default_geometry.point_index([9.999_999_999_999_998, 0.0]),
            Some(100 * 400 + 399),
            "a contained point whose quotient rounds upward belongs to the final cell"
        );
    }

    #[test]
    fn generated_internal_boundaries_select_the_positive_side_cell() {
        let resolution_m = 0.000_205_170_327_817_259_1;
        let lower_x_m = -510.824_772_569_703_04;
        let width = 3_000_u32;
        let geometry = OccupancyGridGeometry::try_new(
            resolution_m,
            [lower_x_m, 0.0],
            width,
            1,
            width as usize,
        )
        .expect("geometry with distinguishable generated boundaries");
        let row_coordinate_m = resolution_m * 0.5;

        for boundary_index in 1..width {
            let boundary_m = lower_x_m + resolution_m * f64::from(boundary_index);
            assert_eq!(
                geometry.point_index([boundary_m, row_coordinate_m]),
                Some(boundary_index as usize),
                "generated boundary {boundary_index} must belong to its positive-side cell"
            );
            assert_eq!(
                geometry.traversal_cell([boundary_m, row_coordinate_m]),
                Some((boundary_index as usize, 0)),
                "DDA traversal must classify generated boundary {boundary_index} identically"
            );
        }

        let counterexample_index = 2_698_u32;
        let counterexample_boundary_m = lower_x_m + resolution_m * f64::from(counterexample_index);
        assert!(
            (counterexample_boundary_m - lower_x_m) / resolution_m
                < f64::from(counterexample_index),
            "regression fixture must exercise downward-rounded division"
        );
    }

    #[test]
    fn invalid_geometry_and_accumulator_bound_are_rejected() {
        assert!(matches!(
            OccupancyGridGeometry::try_new(0.0, [0.0, 0.0], 1, 1, 1),
            Err(OccupancyGridGeometryError::InvalidResolution { .. })
        ));
        assert!(matches!(
            OccupancyGridGeometry::try_new(1.0, [0.0, 0.0], 2, 2, 3),
            Err(OccupancyGridGeometryError::TooManyCells { .. })
        ));
        assert!(matches!(
            OccupancyGridGeometry::try_new(1.0, [9_007_199_254_740_992.0, 0.0], 2, 1, 2,),
            Err(OccupancyGridGeometryError::IndistinguishableCellBoundary { axis: 0, .. })
        ));
        assert!(matches!(
            OccupancyGridGeometry::try_new(
                5.421_010_862_427_522e-20,
                [0.000_976_562_499_999_999_9, 0.0],
                5,
                1,
                5,
            ),
            Err(OccupancyGridGeometryError::IndistinguishableCellBoundary {
                axis: 0,
                boundary_index: 2,
                ..
            })
        ));

        let mut config = test_config_with(1, 1, 1, 1);
        config.evidence = OccupancyEvidenceModel::try_new(i32::MIN, 1, -1, 1)
            .expect("extreme valid evidence delta");
        let exact_minimum = OccupancyConfig::try_new(
            config.geometry,
            config.world_to_occupancy,
            config.camera,
            config.height_range,
            config.depth_range,
            1,
            config.evidence,
            1,
        )
        .expect("one minimum i32 contribution fits exactly");
        let mut exact_mapper = OccupancyMapper::try_new(exact_minimum).expect("exact mapper");
        exact_mapper
            .integrate(
                keyframe(0),
                WorldToCamera::identity(),
                &depth(1, 1, vec![2.0]),
            )
            .expect("exact minimum integration");
        assert!(exact_mapper.evidence.contains(&i32::MIN));
        exact_mapper.remove(keyframe(0)).expect("exact removal");
        assert!(exact_mapper.evidence.iter().all(|value| *value == 0));

        assert!(matches!(
            OccupancyConfig::try_new(
                config.geometry,
                config.world_to_occupancy,
                config.camera,
                config.height_range,
                config.depth_range,
                1,
                config.evidence,
                2,
            ),
            Err(OccupancyConfigError::EvidenceAccumulatorMayOverflow { .. })
        ));
    }

    #[test]
    fn zero_depth_produces_no_evidence_but_is_retained_truthfully() {
        let mut mapper = test_mapper(1, 1);
        let result = mapper
            .integrate(
                keyframe(0),
                WorldToCamera::identity(),
                &depth(1, 1, vec![0.0]),
            )
            .expect("zero-depth integration");
        assert_eq!(result.sampled_blocks, 0);
        assert_eq!(result.retained_keyframes, 1);
        assert!(cells(&mapper).iter().all(|class| *class == 0));
    }

    #[test]
    fn central_ray_marks_free_cells_and_occupied_endpoint() {
        let mut mapper = test_mapper(1, 1);
        mapper
            .integrate(
                keyframe(0),
                WorldToCamera::identity(),
                &depth(1, 1, vec![2.0]),
            )
            .expect("integration");
        let snapshot = mapper.snapshot().expect("snapshot");
        assert_eq!(snapshot.cell(2, 0), Some(OccupancyCell::Free));
        assert_eq!(snapshot.cell(2, 1), Some(OccupancyCell::Free));
        assert_eq!(snapshot.cell(2, 2), Some(OccupancyCell::Occupied));
        assert_eq!(snapshot.cell(1, 1), Some(OccupancyCell::Unknown));
    }

    #[test]
    fn per_keyframe_dedup_and_occupied_precedence_are_exact() {
        let mut mapper = test_mapper(2, 1);
        mapper
            .integrate(
                keyframe(0),
                WorldToCamera::identity(),
                &depth(2, 1, vec![1.0, 2.0]),
            )
            .expect("integration");
        let geometry = mapper.config.geometry;
        let near = geometry.point_index([0.0, 1.0]).expect("near cell");
        let far = geometry.point_index([0.02, 2.0]).expect("far cell");
        assert_eq!(mapper.evidence[near], mapper.config.evidence.occupied_delta);
        assert_eq!(mapper.evidence[far], mapper.config.evidence.occupied_delta);
    }

    #[test]
    fn nearest_valid_sample_is_selected_once_per_block() {
        let config = test_config_with(2, 2, 2, 2);
        let image = depth(2, 2, vec![0.0, 3.0, 2.0, 1.0]);
        let sampled = SampledDepthSource::parse(&image, &config).expect("sampled source");
        assert_eq!(sampled.samples().len(), 1);
        let sample = sampled.samples()[0];
        assert_eq!((sample.column, sample.row, sample.depth_m), (1, 1, 1.0));
    }

    #[test]
    fn equal_block_depth_uses_first_row_major_pixel_deterministically() {
        let config = test_config_with(2, 2, 2, 2);
        let image = depth(2, 2, vec![2.0, 1.0, 1.0, 3.0]);
        let sampled = SampledDepthSource::parse(&image, &config).expect("sampled source");
        let sample = sampled.samples()[0];
        assert_eq!((sample.column, sample.row), (1, 0));
    }

    #[test]
    fn dimension_mismatch_is_transactional() {
        let mut mapper = test_mapper(1, 1);
        let before = cells(&mapper);
        let error = mapper
            .integrate(
                keyframe(0),
                WorldToCamera::identity(),
                &depth(2, 1, vec![1.0, 1.0]),
            )
            .expect_err("dimension mismatch");
        assert!(matches!(
            error,
            OccupancyError::DepthDimensionsMismatch { .. }
        ));
        assert_eq!(mapper.revision(), 0);
        assert_eq!(mapper.retained_keyframes(), 0);
        assert_eq!(cells(&mapper), before);
    }

    #[test]
    fn duplicate_integration_is_rejected_without_mutation() {
        let mut mapper = test_mapper(1, 1);
        let id = keyframe(0);
        let image = depth(1, 1, vec![2.0]);
        mapper
            .integrate(id, WorldToCamera::identity(), &image)
            .expect("first integration");
        let before = cells(&mapper);
        let revision = mapper.revision();
        assert!(matches!(
            mapper.integrate(id, camera_at_world_x(1.0), &image),
            Err(OccupancyError::DuplicateKeyframe { .. })
        ));
        assert_eq!(mapper.revision(), revision);
        assert_eq!(cells(&mapper), before);
    }

    #[test]
    fn integrate_then_remove_restores_exact_evidence() {
        for depth_m in [0.5, 1.0, 2.0, 4.5] {
            let mut mapper = test_mapper(1, 1);
            let before = mapper.evidence.clone();
            let id = keyframe(0);
            mapper
                .integrate(id, WorldToCamera::identity(), &depth(1, 1, vec![depth_m]))
                .expect("integration");
            assert!(matches!(
                mapper.remove(id).expect("remove"),
                OccupancyRemoveOutcome::Removed { .. }
            ));
            assert_eq!(mapper.evidence, before, "depth={depth_m}");
        }
    }

    #[test]
    fn unknown_removal_is_truthful_noop() {
        let mut mapper = test_mapper(1, 1);
        let result = mapper.remove(keyframe(3)).expect("unknown removal");
        assert_eq!(result, OccupancyRemoveOutcome::NotStored);
        assert_eq!(mapper.revision(), 0);
    }

    #[test]
    fn capacity_eviction_subtracts_old_contribution_exactly() {
        let config = test_config_with(1, 1, 1, 1);
        let mut evicting = OccupancyMapper::try_new(config.clone()).expect("mapper");
        let first = keyframe(0);
        let second = keyframe(1);
        evicting
            .integrate(first, WorldToCamera::identity(), &depth(1, 1, vec![1.0]))
            .expect("first");
        let outcome = evicting
            .integrate(second, camera_at_world_x(1.0), &depth(1, 1, vec![2.0]))
            .expect("second");
        assert_eq!(outcome.evicted_keyframe, Some(first));

        let mut reference = OccupancyMapper::try_new(config).expect("reference");
        reference
            .integrate(second, camera_at_world_x(1.0), &depth(1, 1, vec![2.0]))
            .expect("reference integration");
        assert_eq!(evicting.evidence, reference.evidence);
    }

    #[test]
    fn integration_order_is_exactly_independent() {
        let image_a = depth(1, 1, vec![1.0]);
        let image_b = depth(1, 1, vec![2.0]);
        let a = keyframe(0);
        let b = keyframe(1);
        let mut first = test_mapper(1, 1);
        first
            .integrate(a, WorldToCamera::identity(), &image_a)
            .expect("a");
        first
            .integrate(b, camera_at_world_x(1.0), &image_b)
            .expect("b");
        let mut second = test_mapper(1, 1);
        second
            .integrate(b, camera_at_world_x(1.0), &image_b)
            .expect("b");
        second
            .integrate(a, WorldToCamera::identity(), &image_a)
            .expect("a");
        assert_eq!(first.evidence, second.evidence);
    }

    #[test]
    fn partial_pose_update_moves_obstacle_without_ghost() {
        let id = keyframe(0);
        let image = depth(1, 1, vec![2.0]);
        let mut mapper = test_mapper(1, 1);
        mapper
            .integrate(id, WorldToCamera::identity(), &image)
            .expect("integration");
        let old_index = mapper
            .config
            .geometry
            .point_index([0.0, 2.0])
            .expect("old index");
        let new_index = mapper
            .config
            .geometry
            .point_index([1.0, 2.0])
            .expect("new index");
        let outcome = mapper
            .update_poses(&[
                (id, camera_at_world_x(1.0)),
                (keyframe(3), WorldToCamera::identity()),
            ])
            .expect("partial update");
        assert_eq!(outcome.updated_keyframes, 1);
        assert_eq!(outcome.not_stored_keyframes, 1);
        assert_eq!(mapper.evidence[old_index], 0);
        assert_eq!(
            mapper.evidence[new_index],
            mapper.config.evidence.occupied_delta
        );
    }

    #[test]
    fn empty_partial_pose_update_is_noop_without_revision() {
        let mut mapper = test_mapper(1, 1);
        let outcome = mapper
            .update_poses(&[(keyframe(2), WorldToCamera::identity())])
            .expect("unknown update");
        assert_eq!(outcome.updated_keyframes, 0);
        assert_eq!(outcome.not_stored_keyframes, 1);
        assert_eq!(outcome.revision, 0);
    }

    #[test]
    fn pose_batches_reject_duplicates_and_mixed_maps_without_mutation() {
        let mut mapper = test_mapper(1, 1);
        let duplicate = keyframe(0);
        assert_eq!(
            mapper.update_poses(&[
                (duplicate, WorldToCamera::identity()),
                (duplicate, camera_at_world_x(1.0)),
            ]),
            Err(OccupancyError::DuplicatePoseUpdate {
                keyframe_id: duplicate,
            })
        );

        let first = mapped_keyframe(10);
        let second = mapped_keyframe(11);
        assert_ne!(first.map_instance_id(), second.map_instance_id());
        assert_eq!(
            mapper.update_poses(&[
                (first, WorldToCamera::identity()),
                (second, WorldToCamera::identity()),
            ]),
            Err(OccupancyError::MixedMapBatch {
                expected: first.map_instance_id(),
                received: second.map_instance_id(),
            })
        );
        assert_eq!(
            mapper.rebuild(&[
                (first, WorldToCamera::identity()),
                (second, WorldToCamera::identity()),
            ]),
            Err(OccupancyError::MixedMapBatch {
                expected: first.map_instance_id(),
                received: second.map_instance_id(),
            })
        );
        assert_eq!(mapper.revision(), 0);
        assert_eq!(mapper.retained_keyframes(), 0);
        assert!(mapper.evidence.iter().all(|value| *value == 0));
    }

    #[test]
    fn rebuild_requires_complete_unique_poses_transactionally() {
        let a = keyframe(0);
        let b = keyframe(1);
        let mut mapper = test_mapper(1, 1);
        mapper
            .integrate(a, WorldToCamera::identity(), &depth(1, 1, vec![1.0]))
            .expect("a");
        mapper
            .integrate(b, WorldToCamera::identity(), &depth(1, 1, vec![2.0]))
            .expect("b");
        let before = mapper.evidence.clone();
        let revision = mapper.revision();
        assert!(matches!(
            mapper.rebuild(&[(a, camera_at_world_x(1.0))]),
            Err(OccupancyError::MissingRebuildPose { keyframe_id }) if keyframe_id == b
        ));
        assert_eq!(mapper.evidence, before);
        assert_eq!(mapper.revision(), revision);
        assert!(matches!(
            mapper.rebuild(&[
                (a, WorldToCamera::identity()),
                (a, WorldToCamera::identity())
            ]),
            Err(OccupancyError::DuplicateRebuildPose { .. })
        ));
        assert_eq!(mapper.evidence, before);
    }

    #[test]
    fn rebuild_input_order_and_same_map_extras_do_not_change_result() {
        let a = keyframe(0);
        let b = keyframe(1);
        let extra = keyframe(3);
        let image_a = depth(1, 1, vec![1.0]);
        let image_b = depth(1, 1, vec![2.0]);
        let mut first = test_mapper(1, 1);
        first
            .integrate(a, WorldToCamera::identity(), &image_a)
            .expect("a");
        first
            .integrate(b, WorldToCamera::identity(), &image_b)
            .expect("b");
        let mut second = test_mapper(1, 1);
        second
            .integrate(a, WorldToCamera::identity(), &image_a)
            .expect("a");
        second
            .integrate(b, WorldToCamera::identity(), &image_b)
            .expect("b");
        first
            .rebuild(&[
                (a, camera_at_world_x(1.0)),
                (b, camera_at_world_x(-1.0)),
                (extra, WorldToCamera::identity()),
            ])
            .expect("first rebuild");
        second
            .rebuild(&[
                (extra, WorldToCamera::identity()),
                (b, camera_at_world_x(-1.0)),
                (a, camera_at_world_x(1.0)),
            ])
            .expect("second rebuild");
        assert_eq!(first.evidence, second.evidence);
    }

    #[test]
    fn height_slab_clips_free_ray_before_grid_traversal() {
        let mut mapper = test_mapper(1, 1);
        mapper.config.height_range = HeightRangeMeters::try_new(0.0, 1.0).expect("height");
        mapper
            .mark_ray([0.5, 0.5, 2.0], [3.5, 0.5, 0.5])
            .expect("ray");
        let contribution = mapper.finish_contribution().expect("contribution");
        let touched: Vec<usize> = contribution
            .updates
            .iter()
            .map(|update| update.index)
            .collect();
        let geometry = mapper.config.geometry;
        assert!(!touched.contains(&geometry.point_index([0.5, 0.5]).expect("cell")));
        assert!(touched.contains(&geometry.point_index([2.5, 0.5]).expect("cell")));
        assert!(touched.contains(&geometry.point_index([3.5, 0.5]).expect("cell")));
    }

    #[test]
    fn clipping_clamps_roundoff_back_to_the_closed_traversal_bounds() {
        let mut config = test_config_with(1, 1, 1, 8);
        config.geometry = OccupancyGridGeometry::try_new(1.0, [-1.0, -5.0], 2, 20, 40)
            .expect("regression geometry");
        let mut mapper = OccupancyMapper::try_new(config).expect("regression mapper");
        mapper
            .mark_ray(
                [0.0, -0.300_000_011_920_928_96, 1.0],
                [0.0, -9.600_000_381_469_727, 1.0],
            )
            .expect("clipped ray");
        let contribution = mapper.finish_contribution().expect("contribution");
        let boundary_cell = mapper
            .config
            .geometry
            .point_index([0.0, -4.5])
            .expect("first in-grid cell");
        assert!(
            contribution
                .updates
                .iter()
                .any(|update| update.index == boundary_cell),
            "the valid in-grid segment must not be discarded by a one-ULP clipping overshoot"
        );
    }

    #[test]
    fn diagonal_corner_ties_choose_a_direction_independent_side_cell() {
        let mut mapper = test_mapper(1, 1);
        mapper
            .traverse_free_segment([-1.5, 0.5], [0.5, 2.5])
            .expect("diagonal traversal");
        let contribution = mapper.finish_contribution().expect("contribution");
        let geometry = mapper.config.geometry;
        let expected = [
            geometry.point_index([-1.5, 0.5]).expect("cell 0"),
            geometry.point_index([-0.5, 0.5]).expect("cell 1"),
            geometry.point_index([-0.5, 1.5]).expect("cell 2"),
            geometry.point_index([0.5, 1.5]).expect("cell 3"),
            geometry.point_index([0.5, 2.5]).expect("cell 4"),
        ];
        let actual: Vec<usize> = contribution
            .updates
            .iter()
            .map(|update| update.index)
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn grid_dda_is_reversible_for_translated_nonbinary_geometry() {
        let mut config = test_config_with(1, 1, 1, 8);
        config.geometry =
            OccupancyGridGeometry::try_new(0.07, [0.1, -0.3], 3, 3, 9).expect("nonbinary geometry");
        let mut mapper = OccupancyMapper::try_new(config).expect("nonbinary mapper");
        let geometry = mapper.config.geometry;
        let center = |column: u32, row: u32| {
            [
                geometry.lower_bound_m()[0] + (f64::from(column) + 0.5) * geometry.resolution_m(),
                geometry.lower_bound_m()[1] + (f64::from(row) + 0.5) * geometry.resolution_m(),
            ]
        };

        for start_row in 0..geometry.height() {
            for start_column in 0..geometry.width() {
                for end_row in 0..geometry.height() {
                    for end_column in 0..geometry.width() {
                        let start = center(start_column, start_row);
                        let end = center(end_column, end_row);
                        mapper
                            .traverse_free_segment(start, end)
                            .expect("forward traversal");
                        let forward: Vec<_> = mapper
                            .finish_contribution()
                            .expect("forward contribution")
                            .updates
                            .into_iter()
                            .map(|update| update.index)
                            .collect();
                        mapper
                            .traverse_free_segment(end, start)
                            .expect("reverse traversal");
                        let reverse: Vec<_> = mapper
                            .finish_contribution()
                            .expect("reverse contribution")
                            .updates
                            .into_iter()
                            .map(|update| update.index)
                            .collect();
                        assert_eq!(
                            forward, reverse,
                            "translated nonbinary path ({start_column},{start_row}) -> ({end_column},{end_row})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn grid_dda_is_reversible_for_every_cell_center_pair() {
        let mut mapper = test_mapper(1, 1);
        let geometry = mapper.config.geometry;
        let center = |column: u32, row: u32| {
            [
                geometry.lower_bound_m()[0] + (f64::from(column) + 0.5) * geometry.resolution_m(),
                geometry.lower_bound_m()[1] + (f64::from(row) + 0.5) * geometry.resolution_m(),
            ]
        };

        for start_row in 0..geometry.height() {
            for start_column in 0..geometry.width() {
                for end_row in 0..geometry.height() {
                    for end_column in 0..geometry.width() {
                        let start = center(start_column, start_row);
                        let end = center(end_column, end_row);
                        mapper
                            .traverse_free_segment(start, end)
                            .expect("forward traversal");
                        let forward: Vec<usize> = mapper
                            .finish_contribution()
                            .expect("forward contribution")
                            .updates
                            .into_iter()
                            .map(|update| update.index)
                            .collect();
                        mapper
                            .traverse_free_segment(end, start)
                            .expect("reverse traversal");
                        let reverse: Vec<usize> = mapper
                            .finish_contribution()
                            .expect("reverse contribution")
                            .updates
                            .into_iter()
                            .map(|update| update.index)
                            .collect();
                        assert_eq!(
                            forward, reverse,
                            "cell path ({start_column},{start_row}) -> ({end_column},{end_row})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn reset_establishes_new_map_and_rejects_late_old_map_commands() {
        let mut mapper = test_mapper(1, 1);
        mapper
            .integrate(
                keyframe(0),
                WorldToCamera::identity(),
                &depth(1, 1, vec![2.0]),
            )
            .expect("integration");
        let new_map = crate::map::SlamMap::new().snapshot().instance_id();
        let outcome = mapper.reset_to_map(new_map).expect("reset");
        assert_eq!(outcome.cleared_keyframes, 1);
        assert_eq!(outcome.active_map, new_map);
        assert_eq!(mapper.retained_keyframes(), 0);
        assert_eq!(mapper.map_instance_id(), Some(new_map));
        assert!(mapper.evidence.iter().all(|value| *value == 0));
        assert_eq!(outcome.revision, mapper.revision());

        let old_map = keyframe(1).map_instance_id();
        assert_eq!(
            mapper.integrate(
                keyframe(1),
                WorldToCamera::identity(),
                &depth(1, 1, vec![2.0]),
            ),
            Err(OccupancyError::MapMismatch {
                active: new_map,
                received: old_map,
            })
        );
        assert_eq!(mapper.retained_keyframes(), 0);
        assert!(mapper.evidence.iter().all(|value| *value == 0));
        assert_eq!(mapper.revision(), outcome.revision);
        assert_eq!(
            mapper.remove(keyframe(1)),
            Err(OccupancyError::MapMismatch {
                active: new_map,
                received: old_map,
            })
        );
    }

    #[test]
    fn snapshot_is_unambiguous_and_consumes_without_reclassification() {
        let mut mapper = test_mapper(1, 1);
        mapper
            .integrate(
                keyframe(0),
                WorldToCamera::identity(),
                &depth(1, 1, vec![2.0]),
            )
            .expect("integration");
        let snapshot = mapper.snapshot().expect("snapshot");
        assert_eq!(snapshot.width(), 6);
        assert_eq!(snapshot.height(), 6);
        assert_eq!(
            snapshot.row_order(),
            OccupancyRowOrder::IncreasingOccupancyY
        );
        let expected = snapshot.class_ids().to_vec();
        let (metadata, owned) = snapshot.into_parts();
        assert_eq!(metadata.width(), 6);
        assert_eq!(metadata.height(), 6);
        assert_eq!(owned, expected);
    }

    #[test]
    fn revision_exhaustion_prevents_mutation() {
        let mut mapper = test_mapper(1, 1);
        mapper.revision = u64::MAX;
        let before = mapper.evidence.clone();
        assert_eq!(
            mapper.integrate(
                keyframe(0),
                WorldToCamera::identity(),
                &depth(1, 1, vec![2.0])
            ),
            Err(OccupancyError::RevisionExhausted)
        );
        assert_eq!(mapper.evidence, before);
        assert_eq!(mapper.retained_keyframes(), 0);
    }
}
