//! Deterministic, transport-free frontier selection over one occupancy snapshot.
//!
//! A frontier is a known-free cell with at least one cardinally adjacent
//! unknown cell. Search never enters unknown space. Known occupied cells and
//! the map exterior are inflated with the same exact cell-square distance
//! transform used by the global planner. Unknown cells are deliberately not
//! inflation sources: inflating unknown space would eliminate every frontier
//! for any positive clearance. The online local safety layer must still guard
//! against obstacles that have not yet entered the global map.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::num::{NonZeroU8, NonZeroUsize};

use crate::dense::occupancy::{OccupancyCell, OccupancyGridGeometry, OccupancyGridSnapshot};
use crate::map::MapInstanceId;

#[cfg(all(feature = "agent-runtime", unix))]
use super::NanoExploreBoundaryMeters;
use super::cell_inflation::{CellInflationError, CellSquareInflation};
use super::global_planner::{MapPoint, MapTraversalBoundary, PlanStart, PointGoal};

const CARDINAL_COST_DECICELLS: u64 = 10;
const DIAGONAL_COST_DECICELLS: u64 = 14;
const MAX_NEIGHBORS_PER_CELL: usize = 8;

/// Parsed resource and footprint contract for one frontier explorer.
///
/// `maximum_grid_cells` bounds every grid-sized allocation and the O(N)
/// inflation pass. `maximum_expanded_cells` bounds settled search cells, and
/// `maximum_open_set_entries` bounds the retained priority queue. A search
/// that cannot be completed within these limits returns a typed error; it
/// never returns the best partial result as though it were globally selected.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrontierExplorerConfig {
    clearance_from_known_obstacles_m: f64,
    maximum_grid_cells: NonZeroUsize,
    maximum_expanded_cells: NonZeroUsize,
    maximum_open_set_entries: NonZeroUsize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FrontierExplorerConfigError {
    InvalidClearance {
        clearance_m: f64,
    },
    ZeroMaximumGridCells,
    ZeroMaximumExpandedCells,
    ExpandedCellsExceedGridLimit {
        maximum_expanded_cells: usize,
        maximum_grid_cells: usize,
    },
    ZeroMaximumOpenSetEntries,
    OpenSetBoundOverflow {
        maximum_grid_cells: usize,
    },
    OpenSetEntriesExceedEdgeBound {
        maximum_open_set_entries: usize,
        maximum_edge_entries: usize,
    },
}

impl std::fmt::Display for FrontierExplorerConfigError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidClearance { clearance_m } => write!(
                formatter,
                "frontier clearance from known obstacles must be finite and nonnegative, got {clearance_m} m"
            ),
            Self::ZeroMaximumGridCells => {
                write!(formatter, "frontier maximum grid cells must be nonzero")
            }
            Self::ZeroMaximumExpandedCells => {
                write!(formatter, "frontier maximum expanded cells must be nonzero")
            }
            Self::ExpandedCellsExceedGridLimit {
                maximum_expanded_cells,
                maximum_grid_cells,
            } => write!(
                formatter,
                "frontier expanded-cell limit {maximum_expanded_cells} exceeds grid-cell limit {maximum_grid_cells}"
            ),
            Self::ZeroMaximumOpenSetEntries => {
                write!(
                    formatter,
                    "frontier maximum open-set entries must be nonzero"
                )
            }
            Self::OpenSetBoundOverflow { maximum_grid_cells } => write!(
                formatter,
                "frontier open-set edge bound overflows for {maximum_grid_cells} grid cells"
            ),
            Self::OpenSetEntriesExceedEdgeBound {
                maximum_open_set_entries,
                maximum_edge_entries,
            } => write!(
                formatter,
                "frontier open-set limit {maximum_open_set_entries} exceeds the {maximum_edge_entries}-entry grid edge bound"
            ),
        }
    }
}

impl std::error::Error for FrontierExplorerConfigError {}

impl FrontierExplorerConfig {
    pub fn try_new(
        clearance_from_known_obstacles_m: f64,
        maximum_grid_cells: usize,
        maximum_expanded_cells: usize,
        maximum_open_set_entries: usize,
    ) -> Result<Self, FrontierExplorerConfigError> {
        if !clearance_from_known_obstacles_m.is_finite() || clearance_from_known_obstacles_m < 0.0 {
            return Err(FrontierExplorerConfigError::InvalidClearance {
                clearance_m: clearance_from_known_obstacles_m,
            });
        }
        let maximum_grid_cells = NonZeroUsize::new(maximum_grid_cells)
            .ok_or(FrontierExplorerConfigError::ZeroMaximumGridCells)?;
        let maximum_expanded_cells = NonZeroUsize::new(maximum_expanded_cells)
            .ok_or(FrontierExplorerConfigError::ZeroMaximumExpandedCells)?;
        if maximum_expanded_cells > maximum_grid_cells {
            return Err(FrontierExplorerConfigError::ExpandedCellsExceedGridLimit {
                maximum_expanded_cells: maximum_expanded_cells.get(),
                maximum_grid_cells: maximum_grid_cells.get(),
            });
        }
        let maximum_open_set_entries = NonZeroUsize::new(maximum_open_set_entries)
            .ok_or(FrontierExplorerConfigError::ZeroMaximumOpenSetEntries)?;
        let maximum_edge_entries = maximum_grid_cells
            .get()
            .checked_mul(MAX_NEIGHBORS_PER_CELL)
            .ok_or(FrontierExplorerConfigError::OpenSetBoundOverflow {
                maximum_grid_cells: maximum_grid_cells.get(),
            })?;
        if maximum_open_set_entries.get() > maximum_edge_entries {
            return Err(FrontierExplorerConfigError::OpenSetEntriesExceedEdgeBound {
                maximum_open_set_entries: maximum_open_set_entries.get(),
                maximum_edge_entries,
            });
        }

        Ok(Self {
            clearance_from_known_obstacles_m: if clearance_from_known_obstacles_m == 0.0 {
                0.0
            } else {
                clearance_from_known_obstacles_m
            },
            maximum_grid_cells,
            maximum_expanded_cells,
            maximum_open_set_entries,
        })
    }

    pub fn clearance_from_known_obstacles_m(self) -> f64 {
        self.clearance_from_known_obstacles_m
    }

    pub fn maximum_grid_cells(self) -> usize {
        self.maximum_grid_cells.get()
    }

    pub fn maximum_expanded_cells(self) -> usize {
        self.maximum_expanded_cells.get()
    }

    pub fn maximum_open_set_entries(self) -> usize {
        self.maximum_open_set_entries.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrontierBuildError {
    MapHasNoInstance,
    MapTooLarge {
        cells: usize,
        maximum: usize,
    },
    GeometryNotAddressable,
    SnapshotInvariant,
    AllocationFailed {
        context: &'static str,
        requested: usize,
    },
}

impl std::fmt::Display for FrontierBuildError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MapHasNoInstance => write!(
                formatter,
                "cannot explore an occupancy snapshot without a live map instance"
            ),
            Self::MapTooLarge { cells, maximum } => write!(
                formatter,
                "occupancy snapshot has {cells} cells, exceeding the frontier limit {maximum}"
            ),
            Self::GeometryNotAddressable => write!(
                formatter,
                "occupancy geometry dimensions are not addressable on this host"
            ),
            Self::SnapshotInvariant => write!(
                formatter,
                "occupancy snapshot metadata and payload violate their parsed invariant"
            ),
            Self::AllocationFailed { context, requested } => write!(
                formatter,
                "failed to allocate {requested} entries for frontier {context}"
            ),
        }
    }
}

impl std::error::Error for FrontierBuildError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FrontierSearchError {
    StartMapMismatch {
        explorer_map_instance_id: MapInstanceId,
        explorer_revision: u64,
        start_map_instance_id: MapInstanceId,
        start_revision: u64,
    },
    StartOutsideMap {
        point: MapPoint,
    },
    StartOutsideExplorationBoundary {
        point: MapPoint,
    },
    StartCellCenterOutsideExplorationBoundary {
        point: MapPoint,
        cell_center: MapPoint,
    },
    StartBlocked {
        point: MapPoint,
    },
    ExpandedCellLimitExceeded {
        maximum: usize,
    },
    OpenSetLimitExceeded {
        maximum: usize,
    },
    SearchCostOverflow,
    MetricConversionOverflow,
    SearchInvariant,
}

impl std::fmt::Display for FrontierSearchError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StartMapMismatch {
                explorer_map_instance_id,
                explorer_revision,
                start_map_instance_id,
                start_revision,
            } => write!(
                formatter,
                "frontier start belongs to map {} revision {}, but the explorer is bound to map {} revision {}",
                start_map_instance_id.as_u64(),
                start_revision,
                explorer_map_instance_id.as_u64(),
                explorer_revision
            ),
            Self::StartOutsideMap { point } => write!(
                formatter,
                "frontier start [{}, {}] m is outside the occupancy map",
                point.x_m(),
                point.y_m()
            ),
            Self::StartOutsideExplorationBoundary { point } => write!(
                formatter,
                "frontier start [{}, {}] m is outside the operator exploration boundary",
                point.x_m(),
                point.y_m()
            ),
            Self::StartCellCenterOutsideExplorationBoundary { point, cell_center } => write!(
                formatter,
                "frontier start [{}, {}] m belongs to cell center [{}, {}] m outside the operator exploration boundary",
                point.x_m(),
                point.y_m(),
                cell_center.x_m(),
                cell_center.y_m()
            ),
            Self::StartBlocked { point } => write!(
                formatter,
                "frontier start [{}, {}] m is not known-free and traversable after inflation",
                point.x_m(),
                point.y_m()
            ),
            Self::ExpandedCellLimitExceeded { maximum } => write!(
                formatter,
                "frontier search requires more than {maximum} settled cells"
            ),
            Self::OpenSetLimitExceeded { maximum } => write!(
                formatter,
                "frontier search requires more than {maximum} retained open-set entries"
            ),
            Self::SearchCostOverflow => write!(formatter, "frontier search cost overflowed"),
            Self::MetricConversionOverflow => write!(
                formatter,
                "frontier planner cost cannot be represented as finite metres"
            ),
            Self::SearchInvariant => {
                write!(
                    formatter,
                    "frontier search violated its bounded grid invariant"
                )
            }
        }
    }
}

impl std::error::Error for FrontierSearchError {}

/// Deterministic selection score.
///
/// Candidates are ordered by shortest planner-compatible travel cost, then by
/// greatest cardinally adjacent unknown-cell count, then by lowest row-major
/// cell index. `travel_cost_m` uses the global planner's integer octile metric:
/// a cardinal step is one resolution and a diagonal step is 1.4 resolutions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrontierScore {
    travel_cost_m: f64,
    adjacent_unknown_cells: u8,
}

impl FrontierScore {
    pub fn travel_cost_m(self) -> f64 {
        self.travel_cost_m
    }

    pub fn adjacent_unknown_cells(self) -> u8 {
        self.adjacent_unknown_cells
    }
}

/// A frontier destination bound to the exact map instance and revision searched.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrontierGoal {
    point_goal: PointGoal,
    traversal_boundary: Option<MapTraversalBoundary>,
    column: u32,
    row: u32,
    score: FrontierScore,
    settled_cells: usize,
}

impl FrontierGoal {
    pub fn point_goal(self) -> PointGoal {
        self.point_goal
    }

    /// The exact robot-centre traversal boundary enforced while selecting this
    /// goal. `Some` is retained by Nano-bounded selection and must be carried
    /// into every planner invocation which may execute this goal.
    pub fn traversal_boundary(self) -> Option<MapTraversalBoundary> {
        self.traversal_boundary
    }

    pub fn point(self) -> MapPoint {
        self.point_goal.point()
    }

    pub fn map_instance_id(self) -> MapInstanceId {
        self.point_goal.map_instance_id()
    }

    pub fn map_revision(self) -> u64 {
        self.point_goal.selected_revision()
    }

    pub fn column(self) -> u32 {
        self.column
    }

    pub fn row(self) -> u32 {
        self.row
    }

    pub fn score(self) -> FrontierScore {
        self.score
    }

    pub fn settled_cells(self) -> usize {
        self.settled_cells
    }
}

/// One cardinal direction from a known-free frontier cell into unknown space.
///
/// Directions are expressed in the occupancy map frame, not the robot body
/// frame. Turning them into a yaw target therefore requires the current typed
/// map-to-base transform, which this transport-free selector does not own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FrontierUnknownDirection {
    NegativeMapY,
    NegativeMapX,
    PositiveMapX,
    PositiveMapY,
}

impl FrontierUnknownDirection {
    const fn bit(self) -> u8 {
        match self {
            Self::NegativeMapY => 1 << 0,
            Self::NegativeMapX => 1 << 1,
            Self::PositiveMapX => 1 << 2,
            Self::PositiveMapY => 1 << 3,
        }
    }
}

const FRONTIER_DIRECTIONS: [FrontierUnknownDirection; 4] = [
    FrontierUnknownDirection::NegativeMapY,
    FrontierUnknownDirection::NegativeMapX,
    FrontierUnknownDirection::PositiveMapX,
    FrontierUnknownDirection::PositiveMapY,
];

/// Non-empty cardinal evidence that a known-free cell borders unknown space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrontierUnknownDirections(NonZeroU8);

impl FrontierUnknownDirections {
    pub fn contains(self, direction: FrontierUnknownDirection) -> bool {
        self.0.get() & direction.bit() != 0
    }

    pub fn count(self) -> u8 {
        self.0.get().count_ones() as u8
    }

    pub fn iter(self) -> impl Iterator<Item = FrontierUnknownDirection> {
        FRONTIER_DIRECTIONS
            .into_iter()
            .filter(move |direction| self.contains(*direction))
    }

    fn from_bits(bits: u8) -> Option<Self> {
        NonZeroU8::new(bits).map(Self)
    }
}

/// Evidence that exploration cannot produce a positive-distance goal while
/// the robot's current cell is itself a frontier.
///
/// The owner should deliberately scan in place using a separately time-aligned
/// robot orientation, then wait for a newer occupancy revision before asking
/// this immutable explorer again. Repeating selection against the same start
/// and map revision intentionally reproduces the same evidence; it does not
/// fabricate motion progress.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrontierInPlaceScan {
    map_instance_id: MapInstanceId,
    map_revision: u64,
    robot_point: MapPoint,
    column: u32,
    row: u32,
    unknown_directions: FrontierUnknownDirections,
    settled_cells: usize,
}

impl FrontierInPlaceScan {
    pub fn map_instance_id(self) -> MapInstanceId {
        self.map_instance_id
    }

    pub fn map_revision(self) -> u64 {
        self.map_revision
    }

    pub fn robot_point(self) -> MapPoint {
        self.robot_point
    }

    pub fn column(self) -> u32 {
        self.column
    }

    pub fn row(self) -> u32 {
        self.row
    }

    pub fn unknown_directions(self) -> FrontierUnknownDirections {
        self.unknown_directions
    }

    pub fn settled_cells(self) -> usize {
        self.settled_cells
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FrontierSearchOutcome {
    /// A map-bound navigation destination with strictly positive travel cost.
    Selected(FrontierGoal),
    /// No positive-distance frontier exists, but the current cell needs a
    /// deliberate map-frame-aware scan before exploration can be declared done.
    InPlaceScanRequired(FrontierInPlaceScan),
    NoReachableFrontier {
        settled_cells: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SearchNode {
    index: usize,
    cost_decicells: u64,
}

impl Ord for SearchNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap, so reverse both priorities.
        other
            .cost_decicells
            .cmp(&self.cost_decicells)
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl PartialOrd for SearchNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Candidate {
    index: usize,
    cost_decicells: u64,
    adjacent_unknown_cells: u8,
}

/// Closed metric rectangle reduced once to an exact set of cell centers.
///
/// Every search edge joins two admitted centers inside the same convex
/// rectangle, so the whole straight edge remains inside the operator
/// boundary. Unknown cells outside this range cannot create a frontier.
#[derive(Clone, Copy, Debug, PartialEq)]
struct FrontierTraversalBoundary {
    minimum_x_m: f64,
    minimum_y_m: f64,
    maximum_x_m: f64,
    maximum_y_m: f64,
    minimum_column: usize,
    maximum_column_exclusive: usize,
    minimum_row: usize,
    maximum_row_exclusive: usize,
    map_boundary: Option<MapTraversalBoundary>,
}

impl FrontierTraversalBoundary {
    fn unbounded(width: usize, height: usize) -> Self {
        Self {
            minimum_x_m: f64::NEG_INFINITY,
            minimum_y_m: f64::NEG_INFINITY,
            maximum_x_m: f64::INFINITY,
            maximum_y_m: f64::INFINITY,
            minimum_column: 0,
            maximum_column_exclusive: width,
            minimum_row: 0,
            maximum_row_exclusive: height,
            map_boundary: None,
        }
    }

    #[cfg(all(feature = "agent-runtime", unix))]
    fn for_nano(
        geometry: OccupancyGridGeometry,
        width: usize,
        height: usize,
        boundary: NanoExploreBoundaryMeters,
    ) -> Self {
        let lower = geometry.lower_bound_m();
        let resolution_m = geometry.resolution_m();
        let map_boundary = MapTraversalBoundary::try_new(
            boundary.minimum_x_m(),
            boundary.minimum_y_m(),
            boundary.maximum_x_m(),
            boundary.maximum_y_m(),
        )
        .expect("parsed Nano exploration boundary is a finite ordered rectangle");
        Self {
            minimum_x_m: boundary.minimum_x_m(),
            minimum_y_m: boundary.minimum_y_m(),
            maximum_x_m: boundary.maximum_x_m(),
            maximum_y_m: boundary.maximum_y_m(),
            minimum_column: first_center_not_less_than(
                width,
                lower[0],
                resolution_m,
                boundary.minimum_x_m(),
            ),
            maximum_column_exclusive: first_center_greater_than(
                width,
                lower[0],
                resolution_m,
                boundary.maximum_x_m(),
            ),
            minimum_row: first_center_not_less_than(
                height,
                lower[1],
                resolution_m,
                boundary.minimum_y_m(),
            ),
            maximum_row_exclusive: first_center_greater_than(
                height,
                lower[1],
                resolution_m,
                boundary.maximum_y_m(),
            ),
            map_boundary: Some(map_boundary),
        }
    }

    fn contains_point(self, point: MapPoint) -> bool {
        point.x_m() >= self.minimum_x_m
            && point.x_m() <= self.maximum_x_m
            && point.y_m() >= self.minimum_y_m
            && point.y_m() <= self.maximum_y_m
    }

    fn contains_cell(self, column: usize, row: usize) -> bool {
        column >= self.minimum_column
            && column < self.maximum_column_exclusive
            && row >= self.minimum_row
            && row < self.maximum_row_exclusive
    }
}

/// Reusable frontier selector bound to one immutable occupancy-map revision.
pub struct FrontierExplorer<'map> {
    snapshot: &'map OccupancyGridSnapshot,
    config: FrontierExplorerConfig,
    map_instance_id: MapInstanceId,
    map_revision: u64,
    geometry: OccupancyGridGeometry,
    width: usize,
    height: usize,
    traversable: Vec<bool>,
    distances_decicells: Vec<u64>,
    settled: Vec<bool>,
    open: BinaryHeap<SearchNode>,
}

/// Nano frontier selector whose operator boundary is mandatory and immutable.
///
/// Metric bounds are converted to cell-center ranges once at construction;
/// every subsequent `select` enforces that same parsed contract.
#[cfg(all(feature = "agent-runtime", unix))]
pub struct NanoBoundaryFrontierExplorer<'map> {
    inner: FrontierExplorer<'map>,
    boundary: NanoExploreBoundaryMeters,
    traversal_boundary: FrontierTraversalBoundary,
}

#[cfg(all(feature = "agent-runtime", unix))]
impl<'map> NanoBoundaryFrontierExplorer<'map> {
    pub fn try_new(
        snapshot: &'map OccupancyGridSnapshot,
        config: FrontierExplorerConfig,
        boundary: NanoExploreBoundaryMeters,
    ) -> Result<Self, FrontierBuildError> {
        let inner = FrontierExplorer::try_new(snapshot, config)?;
        let traversal_boundary = FrontierTraversalBoundary::for_nano(
            inner.geometry,
            inner.width,
            inner.height,
            boundary,
        );
        Ok(Self {
            inner,
            boundary,
            traversal_boundary,
        })
    }

    pub fn map_instance_id(&self) -> MapInstanceId {
        self.inner.map_instance_id()
    }

    pub fn map_revision(&self) -> u64 {
        self.inner.map_revision()
    }

    pub fn config(&self) -> FrontierExplorerConfig {
        self.inner.config()
    }

    pub fn boundary(&self) -> NanoExploreBoundaryMeters {
        self.boundary
    }

    pub fn is_current_for(&self, snapshot: &OccupancyGridSnapshot) -> bool {
        self.inner.is_current_for(snapshot)
    }

    pub fn select(
        &mut self,
        start: PlanStart,
    ) -> Result<FrontierSearchOutcome, FrontierSearchError> {
        self.inner
            .select_in_boundary(start, self.traversal_boundary)
    }
}

impl<'map> FrontierExplorer<'map> {
    pub fn try_new(
        snapshot: &'map OccupancyGridSnapshot,
        config: FrontierExplorerConfig,
    ) -> Result<Self, FrontierBuildError> {
        let map_instance_id = snapshot
            .map_instance_id()
            .ok_or(FrontierBuildError::MapHasNoInstance)?;
        let geometry = snapshot.geometry();
        let cell_count = geometry.cell_count();
        if cell_count > config.maximum_grid_cells() {
            return Err(FrontierBuildError::MapTooLarge {
                cells: cell_count,
                maximum: config.maximum_grid_cells(),
            });
        }
        if snapshot.class_ids().len() != cell_count {
            return Err(FrontierBuildError::SnapshotInvariant);
        }
        let width = usize::try_from(geometry.width())
            .map_err(|_| FrontierBuildError::GeometryNotAddressable)?;
        let height = usize::try_from(geometry.height())
            .map_err(|_| FrontierBuildError::GeometryNotAddressable)?;
        if width
            .checked_mul(height)
            .filter(|count| *count == cell_count)
            .is_none()
        {
            return Err(FrontierBuildError::SnapshotInvariant);
        }

        let mut occupied_sources = try_bool_buffer(cell_count, "occupied-cell mask")?;
        for (index, source) in occupied_sources.iter_mut().enumerate() {
            *source = snapshot.class_ids()[index] == OccupancyCell::Occupied.class_id();
        }
        let mut traversable = try_bool_buffer(cell_count, "inflated traversability mask")?;
        let mut inflation =
            CellSquareInflation::try_new(width, height).map_err(map_inflation_build_error)?;
        inflation
            .inflate(
                &occupied_sources,
                &mut traversable,
                geometry.resolution_m(),
                config.clearance_from_known_obstacles_m(),
                true,
            )
            .map_err(map_inflation_build_error)?;
        for (index, blocked) in traversable.iter_mut().enumerate() {
            *blocked = snapshot.class_ids()[index] == OccupancyCell::Free.class_id() && !*blocked;
        }

        let distances_decicells = try_u64_buffer(cell_count, "distance grid")?;
        let settled = try_bool_buffer(cell_count, "settled-cell mask")?;
        let mut open = BinaryHeap::new();
        open.try_reserve_exact(config.maximum_open_set_entries())
            .map_err(|_| FrontierBuildError::AllocationFailed {
                context: "open set",
                requested: config.maximum_open_set_entries(),
            })?;

        Ok(Self {
            snapshot,
            config,
            map_instance_id,
            map_revision: snapshot.revision(),
            geometry,
            width,
            height,
            traversable,
            distances_decicells,
            settled,
            open,
        })
    }

    pub fn map_instance_id(&self) -> MapInstanceId {
        self.map_instance_id
    }

    pub fn map_revision(&self) -> u64 {
        self.map_revision
    }

    pub fn config(&self) -> FrontierExplorerConfig {
        self.config
    }

    pub fn is_current_for(&self, snapshot: &OccupancyGridSnapshot) -> bool {
        snapshot.map_instance_id() == Some(self.map_instance_id)
            && snapshot.revision() == self.map_revision
    }

    pub fn select(
        &mut self,
        start: PlanStart,
    ) -> Result<FrontierSearchOutcome, FrontierSearchError> {
        let boundary = FrontierTraversalBoundary::unbounded(self.width, self.height);
        self.select_in_boundary(start, boundary)
    }

    fn select_in_boundary(
        &mut self,
        start: PlanStart,
        boundary: FrontierTraversalBoundary,
    ) -> Result<FrontierSearchOutcome, FrontierSearchError> {
        if start.map_instance_id() != self.map_instance_id
            || start.map_revision() != self.map_revision
        {
            return Err(FrontierSearchError::StartMapMismatch {
                explorer_map_instance_id: self.map_instance_id,
                explorer_revision: self.map_revision,
                start_map_instance_id: start.map_instance_id(),
                start_revision: start.map_revision(),
            });
        }
        let start_point = start.point();
        let start_index = self
            .geometry
            .point_index(start_point.as_array())
            .ok_or(FrontierSearchError::StartOutsideMap { point: start_point })?;
        if !boundary.contains_point(start_point) {
            return Err(FrontierSearchError::StartOutsideExplorationBoundary {
                point: start_point,
            });
        }
        let start_column = start_index % self.width;
        let start_row = start_index / self.width;
        if !boundary.contains_cell(start_column, start_row) {
            return Err(
                FrontierSearchError::StartCellCenterOutsideExplorationBoundary {
                    point: start_point,
                    cell_center: self.cell_center(start_index)?,
                },
            );
        }
        if !self.traversable.get(start_index).copied().unwrap_or(false) {
            return Err(FrontierSearchError::StartBlocked { point: start_point });
        }

        self.distances_decicells.fill(u64::MAX);
        self.settled.fill(false);
        self.open.clear();
        self.distances_decicells[start_index] = 0;
        self.push_open(SearchNode {
            index: start_index,
            cost_decicells: 0,
        })?;

        let mut settled_cells = 0_usize;
        let mut best = None;
        let mut start_frontier = None;
        while let Some(current) = self.open.pop() {
            if self.settled[current.index]
                || self.distances_decicells[current.index] != current.cost_decicells
            {
                continue;
            }
            if best.is_some_and(|candidate: Candidate| {
                current.cost_decicells > candidate.cost_decicells
            }) {
                break;
            }
            if settled_cells == self.config.maximum_expanded_cells() {
                return Err(FrontierSearchError::ExpandedCellLimitExceeded {
                    maximum: self.config.maximum_expanded_cells(),
                });
            }
            self.settled[current.index] = true;
            settled_cells += 1;

            let unknown_directions = self.adjacent_unknown_directions(current.index, boundary)?;
            if current.index == start_index {
                start_frontier = unknown_directions;
            } else if let Some(unknown_directions) = unknown_directions {
                let candidate = Candidate {
                    index: current.index,
                    cost_decicells: current.cost_decicells,
                    adjacent_unknown_cells: unknown_directions.count(),
                };
                if best.is_none_or(|incumbent| candidate_is_better(candidate, incumbent)) {
                    best = Some(candidate);
                }
            }

            // Once the minimum candidate distance is known, positive edge
            // costs mean neighbors cannot tie it. Retain already-open cells at
            // this distance solely to apply the secondary and stable ties.
            if best.is_some() {
                continue;
            }
            self.expand(current, boundary)?;
        }

        let Some(best) = best else {
            if let Some(unknown_directions) = start_frontier {
                let column = u32::try_from(start_index % self.width)
                    .map_err(|_| FrontierSearchError::SearchInvariant)?;
                let row = u32::try_from(start_index / self.width)
                    .map_err(|_| FrontierSearchError::SearchInvariant)?;
                return Ok(FrontierSearchOutcome::InPlaceScanRequired(
                    FrontierInPlaceScan {
                        map_instance_id: self.map_instance_id,
                        map_revision: self.map_revision,
                        robot_point: start_point,
                        column,
                        row,
                        unknown_directions,
                        settled_cells,
                    },
                ));
            }
            return Ok(FrontierSearchOutcome::NoReachableFrontier { settled_cells });
        };
        if best.cost_decicells == 0 {
            return Err(FrontierSearchError::SearchInvariant);
        }
        let point = self.cell_center(best.index)?;
        let point_goal = PointGoal::for_snapshot(point, self.snapshot)
            .map_err(|_| FrontierSearchError::SearchInvariant)?;
        let travel_cost_m = best.cost_decicells as f64
            * (self.geometry.resolution_m() / CARDINAL_COST_DECICELLS as f64);
        if !travel_cost_m.is_finite() {
            return Err(FrontierSearchError::MetricConversionOverflow);
        }
        let column = u32::try_from(best.index % self.width)
            .map_err(|_| FrontierSearchError::SearchInvariant)?;
        let row = u32::try_from(best.index / self.width)
            .map_err(|_| FrontierSearchError::SearchInvariant)?;
        Ok(FrontierSearchOutcome::Selected(FrontierGoal {
            point_goal,
            traversal_boundary: boundary.map_boundary,
            column,
            row,
            score: FrontierScore {
                travel_cost_m,
                adjacent_unknown_cells: best.adjacent_unknown_cells,
            },
            settled_cells,
        }))
    }

    fn push_open(&mut self, node: SearchNode) -> Result<(), FrontierSearchError> {
        if self.open.len() == self.config.maximum_open_set_entries() {
            return Err(FrontierSearchError::OpenSetLimitExceeded {
                maximum: self.config.maximum_open_set_entries(),
            });
        }
        self.open.push(node);
        Ok(())
    }

    fn expand(
        &mut self,
        current: SearchNode,
        boundary: FrontierTraversalBoundary,
    ) -> Result<(), FrontierSearchError> {
        let column = current.index % self.width;
        let row = current.index / self.width;
        for (dx, dy, step_cost) in search_neighbors() {
            let (Some(next_column), Some(next_row)) =
                (column.checked_add_signed(dx), row.checked_add_signed(dy))
            else {
                continue;
            };
            if next_column >= self.width || next_row >= self.height {
                continue;
            }
            if !boundary.contains_cell(next_column, next_row) {
                continue;
            }
            let next = self.index(next_column, next_row)?;
            if !self.traversable[next] || self.settled[next] {
                continue;
            }
            if dx != 0 && dy != 0 {
                let horizontal = self.index(next_column, row)?;
                let vertical = self.index(column, next_row)?;
                if !boundary.contains_cell(next_column, row)
                    || !boundary.contains_cell(column, next_row)
                    || !self.traversable[horizontal]
                    || !self.traversable[vertical]
                {
                    continue;
                }
            }
            let tentative = current
                .cost_decicells
                .checked_add(step_cost)
                .ok_or(FrontierSearchError::SearchCostOverflow)?;
            if tentative < self.distances_decicells[next] {
                self.distances_decicells[next] = tentative;
                self.push_open(SearchNode {
                    index: next,
                    cost_decicells: tentative,
                })?;
            }
        }
        Ok(())
    }

    fn adjacent_unknown_directions(
        &self,
        index: usize,
        boundary: FrontierTraversalBoundary,
    ) -> Result<Option<FrontierUnknownDirections>, FrontierSearchError> {
        let column = index % self.width;
        let row = index / self.width;
        let mut bits = 0_u8;
        for (direction, dx, dy) in [
            (FrontierUnknownDirection::NegativeMapY, 0_isize, -1_isize),
            (FrontierUnknownDirection::NegativeMapX, -1, 0),
            (FrontierUnknownDirection::PositiveMapX, 1, 0),
            (FrontierUnknownDirection::PositiveMapY, 0, 1),
        ] {
            let (Some(neighbor_column), Some(neighbor_row)) =
                (column.checked_add_signed(dx), row.checked_add_signed(dy))
            else {
                continue;
            };
            if neighbor_column >= self.width || neighbor_row >= self.height {
                continue;
            }
            if !boundary.contains_cell(neighbor_column, neighbor_row) {
                continue;
            }
            let neighbor = self.index(neighbor_column, neighbor_row)?;
            if self.snapshot.class_ids()[neighbor] == OccupancyCell::Unknown.class_id() {
                bits |= direction.bit();
            }
        }
        Ok(FrontierUnknownDirections::from_bits(bits))
    }

    fn index(&self, column: usize, row: usize) -> Result<usize, FrontierSearchError> {
        row.checked_mul(self.width)
            .and_then(|base| base.checked_add(column))
            .filter(|index| *index < self.traversable.len())
            .ok_or(FrontierSearchError::SearchInvariant)
    }

    fn cell_center(&self, index: usize) -> Result<MapPoint, FrontierSearchError> {
        let column = index % self.width;
        let row = index / self.width;
        let lower = self.geometry.lower_bound_m();
        let resolution_m = self.geometry.resolution_m();
        let x_m = lower[0] + (column as f64 + 0.5) * resolution_m;
        let y_m = lower[1] + (row as f64 + 0.5) * resolution_m;
        MapPoint::try_new(x_m, y_m).map_err(|_| FrontierSearchError::MetricConversionOverflow)
    }
}

#[cfg(all(feature = "agent-runtime", unix))]
fn first_center_not_less_than(
    count: usize,
    lower_m: f64,
    resolution_m: f64,
    boundary_m: f64,
) -> usize {
    first_center_matching(count, |index| {
        cell_center_axis(lower_m, resolution_m, index) >= boundary_m
    })
}

#[cfg(all(feature = "agent-runtime", unix))]
fn first_center_greater_than(
    count: usize,
    lower_m: f64,
    resolution_m: f64,
    boundary_m: f64,
) -> usize {
    first_center_matching(count, |index| {
        cell_center_axis(lower_m, resolution_m, index) > boundary_m
    })
}

#[cfg(all(feature = "agent-runtime", unix))]
fn first_center_matching(count: usize, predicate: impl Fn(usize) -> bool) -> usize {
    let mut lower = 0_usize;
    let mut upper = count;
    while lower < upper {
        let midpoint = lower + (upper - lower) / 2;
        if predicate(midpoint) {
            upper = midpoint;
        } else {
            lower = midpoint + 1;
        }
    }
    lower
}

#[cfg(all(feature = "agent-runtime", unix))]
fn cell_center_axis(lower_m: f64, resolution_m: f64, index: usize) -> f64 {
    lower_m + (index as f64 + 0.5) * resolution_m
}

fn search_neighbors() -> [(isize, isize, u64); MAX_NEIGHBORS_PER_CELL] {
    [
        (0, -1, CARDINAL_COST_DECICELLS),
        (-1, 0, CARDINAL_COST_DECICELLS),
        (1, 0, CARDINAL_COST_DECICELLS),
        (0, 1, CARDINAL_COST_DECICELLS),
        (-1, -1, DIAGONAL_COST_DECICELLS),
        (1, -1, DIAGONAL_COST_DECICELLS),
        (-1, 1, DIAGONAL_COST_DECICELLS),
        (1, 1, DIAGONAL_COST_DECICELLS),
    ]
}

fn candidate_is_better(candidate: Candidate, incumbent: Candidate) -> bool {
    candidate.cost_decicells < incumbent.cost_decicells
        || (candidate.cost_decicells == incumbent.cost_decicells
            && (candidate.adjacent_unknown_cells > incumbent.adjacent_unknown_cells
                || (candidate.adjacent_unknown_cells == incumbent.adjacent_unknown_cells
                    && candidate.index < incumbent.index)))
}

fn try_bool_buffer(length: usize, context: &'static str) -> Result<Vec<bool>, FrontierBuildError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| FrontierBuildError::AllocationFailed {
            context,
            requested: length,
        })?;
    values.resize(length, false);
    Ok(values)
}

fn try_u64_buffer(length: usize, context: &'static str) -> Result<Vec<u64>, FrontierBuildError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| FrontierBuildError::AllocationFailed {
            context,
            requested: length,
        })?;
    values.resize(length, u64::MAX);
    Ok(values)
}

fn map_inflation_build_error(error: CellInflationError) -> FrontierBuildError {
    match error {
        CellInflationError::InvalidInput => FrontierBuildError::SnapshotInvariant,
        CellInflationError::AllocationFailed { context, requested } => {
            FrontierBuildError::AllocationFailed { context, requested }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::SlamMap;

    fn map_instance_id() -> MapInstanceId {
        SlamMap::new().snapshot().instance_id()
    }

    fn geometry(
        resolution_m: f64,
        lower_bound_m: [f64; 2],
        width: u32,
        height: u32,
    ) -> OccupancyGridGeometry {
        OccupancyGridGeometry::try_new(
            resolution_m,
            lower_bound_m,
            width,
            height,
            width as usize * height as usize,
        )
        .expect("test geometry")
    }

    fn snapshot(
        geometry: OccupancyGridGeometry,
        cells: &[OccupancyCell],
        map_instance_id: MapInstanceId,
        revision: u64,
    ) -> OccupancyGridSnapshot {
        OccupancyGridSnapshot::from_test_cells(geometry, cells, map_instance_id, revision)
    }

    fn config(cell_count: usize) -> FrontierExplorerConfig {
        FrontierExplorerConfig::try_new(0.0, cell_count, cell_count, cell_count * 8)
            .expect("test config")
    }

    fn point(x_m: f64, y_m: f64) -> MapPoint {
        MapPoint::try_new(x_m, y_m).expect("test point")
    }

    #[cfg(all(feature = "agent-runtime", unix))]
    fn boundary(
        minimum_x_m: f64,
        minimum_y_m: f64,
        maximum_x_m: f64,
        maximum_y_m: f64,
    ) -> NanoExploreBoundaryMeters {
        NanoExploreBoundaryMeters::try_new(minimum_x_m, minimum_y_m, maximum_x_m, maximum_y_m)
            .expect("test exploration boundary")
    }

    #[cfg(all(feature = "agent-runtime", unix))]
    fn bounded_explorer<'map>(
        snapshot: &'map OccupancyGridSnapshot,
        boundary: NanoExploreBoundaryMeters,
    ) -> NanoBoundaryFrontierExplorer<'map> {
        NanoBoundaryFrontierExplorer::try_new(
            snapshot,
            config(snapshot.geometry().cell_count()),
            boundary,
        )
        .expect("bounded Nano explorer")
    }

    fn start_for(snapshot: &OccupancyGridSnapshot, column: usize, row: usize) -> PlanStart {
        let geometry = snapshot.geometry();
        let lower = geometry.lower_bound_m();
        let resolution_m = geometry.resolution_m();
        PlanStart::for_snapshot(
            point(
                lower[0] + (column as f64 + 0.5) * resolution_m,
                lower[1] + (row as f64 + 0.5) * resolution_m,
            ),
            snapshot,
        )
        .expect("test start")
    }

    fn selected(outcome: FrontierSearchOutcome) -> FrontierGoal {
        match outcome {
            FrontierSearchOutcome::Selected(goal) => goal,
            FrontierSearchOutcome::InPlaceScanRequired(_) => {
                panic!("expected positive-distance selected frontier")
            }
            FrontierSearchOutcome::NoReachableFrontier { .. } => {
                panic!("expected selected frontier")
            }
        }
    }

    #[test]
    fn config_parses_once_and_canonicalizes_zero() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -f64::EPSILON] {
            assert!(matches!(
                FrontierExplorerConfig::try_new(invalid, 9, 9, 72),
                Err(FrontierExplorerConfigError::InvalidClearance { .. })
            ));
        }
        assert_eq!(
            FrontierExplorerConfig::try_new(0.0, 0, 1, 1),
            Err(FrontierExplorerConfigError::ZeroMaximumGridCells)
        );
        assert_eq!(
            FrontierExplorerConfig::try_new(0.0, 9, 0, 1),
            Err(FrontierExplorerConfigError::ZeroMaximumExpandedCells)
        );
        assert!(matches!(
            FrontierExplorerConfig::try_new(0.0, 9, 10, 1),
            Err(FrontierExplorerConfigError::ExpandedCellsExceedGridLimit { .. })
        ));
        assert_eq!(
            FrontierExplorerConfig::try_new(0.0, 9, 9, 0),
            Err(FrontierExplorerConfigError::ZeroMaximumOpenSetEntries)
        );
        assert!(matches!(
            FrontierExplorerConfig::try_new(0.0, 9, 9, 73),
            Err(FrontierExplorerConfigError::OpenSetEntriesExceedEdgeBound { .. })
        ));
        let parsed = FrontierExplorerConfig::try_new(-0.0, 9, 9, 72).expect("negative zero");
        assert_eq!(
            parsed.clearance_from_known_obstacles_m().to_bits(),
            0.0_f64.to_bits()
        );
    }

    #[test]
    fn selected_goal_uses_map_frame_metric_cell_center_and_exact_identity() {
        let geometry = geometry(0.5, [-2.0, 3.0], 4, 3);
        let mut cells = vec![OccupancyCell::Free; geometry.cell_count()];
        cells[2 * 4 + 3] = OccupancyCell::Unknown;
        let map_instance_id = map_instance_id();
        let snapshot = snapshot(geometry, &cells, map_instance_id, 17);
        let start = start_for(&snapshot, 0, 1);
        let mut explorer = FrontierExplorer::try_new(&snapshot, config(12)).expect("explorer");

        let goal = selected(explorer.select(start).expect("frontier search"));
        assert_eq!((goal.column(), goal.row()), (2, 2));
        assert_eq!(goal.point(), point(-0.75, 4.25));
        assert_eq!(goal.map_instance_id(), map_instance_id);
        assert_eq!(goal.map_revision(), 17);
        assert!(goal.point_goal().was_selected_from(&snapshot));
        assert_eq!(goal.score().adjacent_unknown_cells(), 1);
        assert!((goal.score().travel_cost_m() - 1.2).abs() < 1.0e-12);
    }

    #[test]
    fn rejects_wrong_map_revision_instance_outside_and_blocked_starts() {
        let geometry = geometry(1.0, [0.0, 0.0], 3, 3);
        let mut cells = vec![OccupancyCell::Free; 9];
        cells[8] = OccupancyCell::Unknown;
        cells[0] = OccupancyCell::Occupied;
        let id = map_instance_id();
        let same_map_new_revision = snapshot(geometry, &cells, id, 5);
        let other_map = snapshot(geometry, &cells, map_instance_id(), 4);
        let map_snapshot = snapshot(geometry, &cells, id, 4);
        let mut explorer = FrontierExplorer::try_new(&map_snapshot, config(9)).expect("explorer");

        assert!(matches!(
            explorer.select(start_for(&same_map_new_revision, 1, 1)),
            Err(FrontierSearchError::StartMapMismatch {
                explorer_revision: 4,
                start_revision: 5,
                ..
            })
        ));
        assert!(matches!(
            explorer.select(start_for(&other_map, 1, 1)),
            Err(FrontierSearchError::StartMapMismatch { .. })
        ));
        let outside =
            PlanStart::for_snapshot(point(3.0, 1.5), &map_snapshot).expect("bound outside");
        assert!(matches!(
            explorer.select(outside),
            Err(FrontierSearchError::StartOutsideMap { .. })
        ));
        assert!(matches!(
            explorer.select(start_for(&map_snapshot, 0, 0)),
            Err(FrontierSearchError::StartBlocked { .. })
        ));
    }

    #[test]
    fn diagonal_corner_cut_does_not_make_frontier_reachable() {
        let geometry = geometry(1.0, [0.0, 0.0], 3, 3);
        let mut cells = vec![OccupancyCell::Occupied; 9];
        cells[0] = OccupancyCell::Free;
        cells[4] = OccupancyCell::Free;
        cells[5] = OccupancyCell::Unknown;
        let snapshot = snapshot(geometry, &cells, map_instance_id(), 1);
        let start = start_for(&snapshot, 0, 0);
        let mut explorer = FrontierExplorer::try_new(&snapshot, config(9)).expect("explorer");

        assert_eq!(
            explorer.select(start).expect("bounded search"),
            FrontierSearchOutcome::NoReachableFrontier { settled_cells: 1 }
        );
    }

    #[test]
    fn known_obstacle_clearance_keeps_safe_interior_frontiers_usable() {
        let map_geometry = geometry(1.0, [0.0, 0.0], 9, 9);
        let mut cells = vec![OccupancyCell::Free; 81];
        cells[4 * 9 + 6] = OccupancyCell::Unknown;
        let map_snapshot = snapshot(map_geometry, &cells, map_instance_id(), 1);
        let start = start_for(&map_snapshot, 2, 4);
        let clearance =
            FrontierExplorerConfig::try_new(0.1, 81, 81, 648).expect("clearance config");
        let mut explorer = FrontierExplorer::try_new(&map_snapshot, clearance).expect("explorer");

        let goal = selected(explorer.select(start).expect("frontier search"));
        assert_eq!((goal.column(), goal.row()), (5, 4));
        assert_eq!(goal.score().travel_cost_m(), 3.0);
    }

    #[test]
    fn fully_observed_reachable_region_reports_no_frontier() {
        let map_geometry = geometry(1.0, [0.0, 0.0], 3, 3);
        let cells = vec![OccupancyCell::Free; 9];
        let map_snapshot = snapshot(map_geometry, &cells, map_instance_id(), 1);
        let start = start_for(&map_snapshot, 1, 1);
        let mut explorer = FrontierExplorer::try_new(&map_snapshot, config(9)).expect("explorer");

        assert_eq!(
            explorer.select(start).expect("bounded search"),
            FrontierSearchOutcome::NoReachableFrontier { settled_cells: 9 }
        );
    }

    #[cfg(all(feature = "agent-runtime", unix))]
    #[test]
    fn closed_exploration_boundary_excludes_outside_unknown_evidence_during_search() {
        let map_geometry = geometry(1.0, [0.0, 0.0], 5, 3);
        let mut cells = vec![OccupancyCell::Free; 15];
        cells[3 + 5] = OccupancyCell::Unknown;
        let map_snapshot = snapshot(map_geometry, &cells, map_instance_id(), 1);
        let start = start_for(&map_snapshot, 0, 1);
        let first_boundary = boundary(0.0, 1.0, 3.0, 2.0);
        let mut explorer = bounded_explorer(&map_snapshot, first_boundary);
        assert_eq!(explorer.boundary(), first_boundary);

        assert_eq!(
            explorer.select(start).expect("bounded search"),
            FrontierSearchOutcome::NoReachableFrontier { settled_cells: 3 },
            "an unknown cell whose center is outside the boundary is not frontier evidence"
        );

        let mut closed_edge_explorer =
            bounded_explorer(&map_snapshot, boundary(0.0, 1.0, 3.5, 2.0));
        let goal = selected(
            closed_edge_explorer
                .select(start)
                .expect("closed-edge search"),
        );
        assert_eq!((goal.column(), goal.row()), (2, 1));
        assert_eq!(goal.point(), point(2.5, 1.5));
    }

    #[cfg(all(feature = "agent-runtime", unix))]
    #[test]
    fn frontier_search_cannot_detour_outside_boundary_and_reenter() {
        let map_geometry = geometry(1.0, [0.0, 0.0], 7, 5);
        let mut cells = vec![OccupancyCell::Occupied; 35];
        cells[1..=5].fill(OccupancyCell::Free);
        for row in cells.chunks_exact_mut(7).take(3) {
            row[1] = OccupancyCell::Free;
            row[5] = OccupancyCell::Free;
        }
        let third_row = cells.chunks_exact_mut(7).nth(2).expect("fixture row");
        third_row[1..=2].fill(OccupancyCell::Free);
        third_row[4..=5].fill(OccupancyCell::Free);
        third_row[6] = OccupancyCell::Unknown;
        let map_snapshot = snapshot(map_geometry, &cells, map_instance_id(), 2);
        let start = start_for(&map_snapshot, 1, 2);
        let mut explorer = FrontierExplorer::try_new(&map_snapshot, config(35)).expect("explorer");

        let unbounded = selected(explorer.select(start).expect("unbounded detour"));
        assert_eq!((unbounded.column(), unbounded.row()), (5, 2));

        let mut bounded_explorer = bounded_explorer(&map_snapshot, boundary(0.0, 1.0, 7.0, 4.0));
        assert!(matches!(
            bounded_explorer.select(start).expect("bounded traversal"),
            FrontierSearchOutcome::NoReachableFrontier { .. }
        ));
    }

    #[cfg(all(feature = "agent-runtime", unix))]
    #[test]
    fn boundary_rejects_outside_start_and_cell_center_conservatively() {
        let map_geometry = geometry(1.0, [0.0, 0.0], 3, 3);
        let cells = vec![OccupancyCell::Free; 9];
        let map_snapshot = snapshot(map_geometry, &cells, map_instance_id(), 1);
        let mut explorer = bounded_explorer(&map_snapshot, boundary(0.6, 1.0, 1.4, 2.0));
        let outside = start_for(&map_snapshot, 0, 1);
        assert!(matches!(
            explorer.select(outside),
            Err(FrontierSearchError::StartOutsideExplorationBoundary { .. })
        ));

        let center_outside = PlanStart::for_snapshot(point(0.6, 1.5), &map_snapshot)
            .expect("start inside map and metric boundary");
        assert!(matches!(
            explorer.select(center_outside),
            Err(
                FrontierSearchError::StartCellCenterOutsideExplorationBoundary {
                    cell_center,
                    ..
                }
            ) if cell_center == point(0.5, 1.5)
        ));
    }

    #[test]
    fn initial_single_free_cell_requires_repeatable_in_place_scan() {
        let map_geometry = geometry(0.5, [-0.75, -0.75], 3, 3);
        let mut cells = vec![OccupancyCell::Unknown; 9];
        cells[4] = OccupancyCell::Free;
        let id = map_instance_id();
        let map_snapshot = snapshot(map_geometry, &cells, id, 23);
        let start = start_for(&map_snapshot, 1, 1);
        let mut explorer = FrontierExplorer::try_new(&map_snapshot, config(9)).expect("explorer");

        let first = explorer.select(start).expect("initial selection");
        let second = explorer.select(start).expect("repeated selection");
        assert_eq!(first, second);
        let FrontierSearchOutcome::InPlaceScanRequired(scan) = first else {
            panic!("a zero-distance frontier must request an explicit scan")
        };
        assert_eq!(scan.map_instance_id(), id);
        assert_eq!(scan.map_revision(), 23);
        assert_eq!(scan.robot_point(), start.point());
        assert_eq!((scan.column(), scan.row()), (1, 1));
        assert_eq!(scan.settled_cells(), 1);
        assert_eq!(scan.unknown_directions().count(), 4);
        assert_eq!(
            scan.unknown_directions().iter().collect::<Vec<_>>(),
            FRONTIER_DIRECTIONS
        );
    }

    #[test]
    fn positive_clearance_rejects_known_obstacle_and_exterior_inflation() {
        let map_geometry = geometry(1.0, [0.0, 0.0], 9, 9);
        let mut cells = vec![OccupancyCell::Occupied; 81];
        for column in 2..=5 {
            cells[4 * 9 + column] = OccupancyCell::Free;
        }
        cells[4 * 9 + 6] = OccupancyCell::Unknown;
        let map_snapshot = snapshot(map_geometry, &cells, map_instance_id(), 1);
        let start = start_for(&map_snapshot, 2, 4);
        let clearance =
            FrontierExplorerConfig::try_new(0.1, 81, 81, 648).expect("clearance config");
        let mut explorer = FrontierExplorer::try_new(&map_snapshot, clearance).expect("explorer");

        // The only frontier cell touches occupied cells above and below, so
        // conservative cell-square inflation rejects it.
        assert_eq!(
            explorer.select(start),
            Err(FrontierSearchError::StartBlocked {
                point: start.point()
            })
        );

        let edge_geometry = geometry(1.0, [0.0, 0.0], 5, 5);
        let mut edge_cells = vec![OccupancyCell::Free; 25];
        edge_cells[1] = OccupancyCell::Unknown;
        let edge_snapshot = snapshot(edge_geometry, &edge_cells, map_instance_id(), 1);
        let edge_start = start_for(&edge_snapshot, 0, 2);
        let mut edge_explorer =
            FrontierExplorer::try_new(&edge_snapshot, clearance).expect("edge explorer");
        assert!(matches!(
            edge_explorer.select(edge_start),
            Err(FrontierSearchError::StartBlocked { .. })
        ));
    }

    #[test]
    fn candidate_order_has_stable_information_and_row_major_ties() {
        let farther = Candidate {
            index: 0,
            cost_decicells: 11,
            adjacent_unknown_cells: 4,
        };
        let nearer = Candidate {
            index: 8,
            cost_decicells: 10,
            adjacent_unknown_cells: 1,
        };
        assert!(candidate_is_better(nearer, farther));

        let more_information = Candidate {
            index: 8,
            cost_decicells: 10,
            adjacent_unknown_cells: 2,
        };
        assert!(candidate_is_better(more_information, nearer));

        let lower_index = Candidate {
            index: 7,
            cost_decicells: 10,
            adjacent_unknown_cells: 2,
        };
        assert!(candidate_is_better(lower_index, more_information));
        assert!(!candidate_is_better(more_information, lower_index));
    }

    #[test]
    fn grid_expansion_and_open_set_limits_fail_closed() {
        let geometry = geometry(1.0, [0.0, 0.0], 5, 5);
        let mut cells = vec![OccupancyCell::Free; 25];
        cells[24] = OccupancyCell::Unknown;
        let snapshot = snapshot(geometry, &cells, map_instance_id(), 1);
        assert_eq!(
            FrontierExplorer::try_new(
                &snapshot,
                FrontierExplorerConfig::try_new(0.0, 24, 24, 192).expect("small config")
            )
            .err(),
            Some(FrontierBuildError::MapTooLarge {
                cells: 25,
                maximum: 24
            })
        );

        let start = start_for(&snapshot, 0, 0);
        let mut expansion_limited = FrontierExplorer::try_new(
            &snapshot,
            FrontierExplorerConfig::try_new(0.0, 25, 1, 8).expect("expansion config"),
        )
        .expect("expansion explorer");
        assert_eq!(
            expansion_limited.select(start),
            Err(FrontierSearchError::ExpandedCellLimitExceeded { maximum: 1 })
        );

        let mut open_limited = FrontierExplorer::try_new(
            &snapshot,
            FrontierExplorerConfig::try_new(0.0, 25, 25, 1).expect("open config"),
        )
        .expect("open explorer");
        assert_eq!(
            open_limited.select(start),
            Err(FrontierSearchError::OpenSetLimitExceeded { maximum: 1 })
        );
    }

    #[test]
    fn selection_is_repeatable_and_matches_small_grid_reference() {
        let geometry = geometry(0.25, [-0.5, -0.5], 3, 3);
        let id = map_instance_id();
        let mut state = 0xA5A5_1234_u64;
        for revision in 1..=512_u64 {
            let mut cells = vec![OccupancyCell::Free; 9];
            for (index, cell) in cells.iter_mut().enumerate() {
                if index == 4 {
                    continue;
                }
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                *cell = match state % 3 {
                    0 => OccupancyCell::Unknown,
                    1 => OccupancyCell::Free,
                    _ => OccupancyCell::Occupied,
                };
            }
            let snapshot = snapshot(geometry, &cells, id, revision);
            let start = start_for(&snapshot, 1, 1);
            let mut explorer = FrontierExplorer::try_new(&snapshot, config(9)).expect("explorer");
            let expected = reference_best(&cells, 3, 3, 4);
            let start_unknown_directions = reference_unknown_directions(&cells, 3, 3, 4);
            let first = explorer.select(start).expect("first selection");
            let second = explorer.select(start).expect("repeat selection");
            assert_eq!(first, second, "revision {revision}");
            match (expected, first) {
                (Some(expected), FrontierSearchOutcome::Selected(actual)) => {
                    assert_eq!(
                        actual.row() as usize * 3 + actual.column() as usize,
                        expected.index
                    );
                    assert_eq!(
                        actual.score().adjacent_unknown_cells(),
                        expected.adjacent_unknown_cells
                    );
                    let expected_m = expected.cost_decicells as f64 * 0.025;
                    assert!((actual.score().travel_cost_m() - expected_m).abs() < 1.0e-12);
                }
                (None, FrontierSearchOutcome::InPlaceScanRequired(scan)) => {
                    assert_eq!(Some(scan.unknown_directions()), start_unknown_directions);
                }
                (None, FrontierSearchOutcome::NoReachableFrontier { .. }) => {
                    assert_eq!(start_unknown_directions, None);
                }
                (expected, actual) => panic!(
                    "reference mismatch at revision {revision}: expected {expected:?}, got {actual:?}"
                ),
            }
        }
    }

    #[cfg(all(feature = "agent-runtime", unix))]
    #[test]
    fn bounded_selection_matches_reference_across_cells_and_closed_edges() {
        let map_geometry = geometry(1.0, [-1.5, -1.5], 3, 3);
        let id = map_instance_id();
        let boundaries = [
            boundary(-0.1, -0.1, 0.1, 0.1),
            boundary(-1.0, -1.0, 0.0, 1.0),
            boundary(0.0, -1.0, 1.0, 1.0),
            boundary(-1.0, -1.0, 1.0, 1.0),
        ];
        let mut state = 0x791F_02D4_u64;
        for revision in 1..=256_u64 {
            let mut cells = vec![OccupancyCell::Free; 9];
            for (index, cell) in cells.iter_mut().enumerate() {
                if index == 4 {
                    continue;
                }
                state = state
                    .wrapping_mul(2_862_933_555_777_941_757)
                    .wrapping_add(3_037_000_493);
                *cell = match state % 3 {
                    0 => OccupancyCell::Unknown,
                    1 => OccupancyCell::Free,
                    _ => OccupancyCell::Occupied,
                };
            }
            let map_snapshot = snapshot(map_geometry, &cells, id, revision);
            let start = start_for(&map_snapshot, 1, 1);
            for boundary in boundaries {
                let allowed = (0..9)
                    .map(|index| {
                        let column = index % 3;
                        let row = index / 3;
                        let center = point(column as f64 - 1.0, row as f64 - 1.0);
                        center.x_m() >= boundary.minimum_x_m()
                            && center.x_m() <= boundary.maximum_x_m()
                            && center.y_m() >= boundary.minimum_y_m()
                            && center.y_m() <= boundary.maximum_y_m()
                    })
                    .collect::<Vec<_>>();
                let expected = reference_best_with_allowed(&cells, 3, 3, 4, &allowed);
                let start_unknown_directions =
                    reference_unknown_directions_with_allowed(&cells, 3, 3, 4, &allowed);
                let actual = bounded_explorer(&map_snapshot, boundary)
                    .select(start)
                    .expect("bounded reference search");
                match (expected, actual) {
                    (Some(expected), FrontierSearchOutcome::Selected(actual)) => {
                        assert_eq!(
                            actual.row() as usize * 3 + actual.column() as usize,
                            expected.index,
                            "revision {revision}, boundary {boundary:?}"
                        );
                        assert!(allowed[expected.index]);
                        assert_eq!(
                            actual.score().adjacent_unknown_cells(),
                            expected.adjacent_unknown_cells
                        );
                    }
                    (None, FrontierSearchOutcome::InPlaceScanRequired(scan)) => {
                        assert_eq!(
                            Some(scan.unknown_directions()),
                            start_unknown_directions,
                            "revision {revision}, boundary {boundary:?}"
                        );
                    }
                    (None, FrontierSearchOutcome::NoReachableFrontier { .. }) => {
                        assert_eq!(start_unknown_directions, None);
                    }
                    (expected, actual) => panic!(
                        "bounded reference mismatch at revision {revision}, boundary {boundary:?}: expected {expected:?}, got {actual:?}"
                    ),
                }
            }
        }
    }

    fn reference_best(
        cells: &[OccupancyCell],
        width: usize,
        height: usize,
        start: usize,
    ) -> Option<Candidate> {
        reference_best_with_allowed(cells, width, height, start, &vec![true; cells.len()])
    }

    fn reference_best_with_allowed(
        cells: &[OccupancyCell],
        width: usize,
        height: usize,
        start: usize,
        allowed: &[bool],
    ) -> Option<Candidate> {
        let mut distances = vec![u64::MAX; cells.len()];
        let mut settled = vec![false; cells.len()];
        distances[start] = 0;
        let mut best = None;
        for _ in 0..cells.len() {
            let Some(current) = (0..cells.len())
                .filter(|index| {
                    allowed[*index] && cells[*index] == OccupancyCell::Free && !settled[*index]
                })
                .min_by_key(|index| (distances[*index], *index))
            else {
                break;
            };
            if distances[current] == u64::MAX {
                break;
            }
            settled[current] = true;
            let column = current % width;
            let row = current / width;
            let unknown_count = [(0_isize, -1_isize), (-1, 0), (1, 0), (0, 1)]
                .into_iter()
                .filter_map(|(dx, dy)| {
                    Some((column.checked_add_signed(dx)?, row.checked_add_signed(dy)?))
                })
                .filter(|(column, row)| *column < width && *row < height)
                .filter(|(column, row)| {
                    let index = row * width + column;
                    allowed[index] && cells[index] == OccupancyCell::Unknown
                })
                .count() as u8;
            if current != start && unknown_count > 0 {
                let candidate = Candidate {
                    index: current,
                    cost_decicells: distances[current],
                    adjacent_unknown_cells: unknown_count,
                };
                if best.is_none_or(|incumbent| candidate_is_better(candidate, incumbent)) {
                    best = Some(candidate);
                }
            }
            for (dx, dy, step_cost) in search_neighbors() {
                let (Some(next_column), Some(next_row)) =
                    (column.checked_add_signed(dx), row.checked_add_signed(dy))
                else {
                    continue;
                };
                if next_column >= width || next_row >= height {
                    continue;
                }
                let next = next_row * width + next_column;
                if !allowed[next] || cells[next] != OccupancyCell::Free || settled[next] {
                    continue;
                }
                if dx != 0
                    && dy != 0
                    && (!allowed[row * width + next_column]
                        || !allowed[next_row * width + column]
                        || cells[row * width + next_column] != OccupancyCell::Free
                        || cells[next_row * width + column] != OccupancyCell::Free)
                {
                    continue;
                }
                distances[next] = distances[next].min(distances[current] + step_cost);
            }
        }
        best
    }

    fn reference_unknown_directions(
        cells: &[OccupancyCell],
        width: usize,
        height: usize,
        index: usize,
    ) -> Option<FrontierUnknownDirections> {
        reference_unknown_directions_with_allowed(
            cells,
            width,
            height,
            index,
            &vec![true; cells.len()],
        )
    }

    fn reference_unknown_directions_with_allowed(
        cells: &[OccupancyCell],
        width: usize,
        height: usize,
        index: usize,
        allowed: &[bool],
    ) -> Option<FrontierUnknownDirections> {
        let column = index % width;
        let row = index / width;
        let mut bits = 0_u8;
        for (direction, dx, dy) in [
            (FrontierUnknownDirection::NegativeMapY, 0_isize, -1_isize),
            (FrontierUnknownDirection::NegativeMapX, -1, 0),
            (FrontierUnknownDirection::PositiveMapX, 1, 0),
            (FrontierUnknownDirection::PositiveMapY, 0, 1),
        ] {
            let (Some(neighbor_column), Some(neighbor_row)) =
                (column.checked_add_signed(dx), row.checked_add_signed(dy))
            else {
                continue;
            };
            if neighbor_column < width
                && neighbor_row < height
                && allowed[neighbor_row * width + neighbor_column]
                && cells[neighbor_row * width + neighbor_column] == OccupancyCell::Unknown
            {
                bits |= direction.bit();
            }
        }
        FrontierUnknownDirections::from_bits(bits)
    }
}
