use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::dense::occupancy::{OccupancyCell, OccupancyGridGeometry, OccupancyGridSnapshot};
use crate::map::MapInstanceId;

use super::cell_inflation::{CellInflationError, CellSquareInflation};
use super::frames::{MapFrame, PlanarPoint};

const CARDINAL_COST: u64 = 10;
const DIAGONAL_COST: u64 = 14;

/// A finite metric point in the displayed occupancy-map frame.
pub type MapPoint = PlanarPoint<MapFrame>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnknownSpacePolicy {
    Blocked,
    Traversable,
}

/// Parsed global-planning policy in SI units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobalPlannerConfig {
    clearance_radius_m: f64,
    unknown_space: UnknownSpacePolicy,
}

impl GlobalPlannerConfig {
    /// Parse a circular footprint-clearance policy.
    ///
    /// For a positive radius, occupied cells and the map exterior are closed
    /// sets: a footprint exactly tangent to either is conservatively blocked.
    /// A zero radius retains point-cell semantics and blocks only source cells.
    pub fn try_new(
        clearance_radius_m: f64,
        unknown_space: UnknownSpacePolicy,
    ) -> Result<Self, GlobalPlanError> {
        if !clearance_radius_m.is_finite() || clearance_radius_m < 0.0 {
            return Err(GlobalPlanError::InvalidClearanceRadius { clearance_radius_m });
        }
        Ok(Self {
            clearance_radius_m,
            unknown_space,
        })
    }

    pub fn clearance_radius_m(self) -> f64 {
        self.clearance_radius_m
    }

    pub fn unknown_space(self) -> UnknownSpacePolicy {
        self.unknown_space
    }
}

#[derive(Debug, PartialEq)]
pub enum GlobalPlanError {
    MapHasNoInstance,
    InvalidClearanceRadius {
        clearance_radius_m: f64,
    },
    AllocationFailed {
        context: &'static str,
        requested: usize,
    },
    GoalMapMismatch {
        planner_map_instance_id: MapInstanceId,
        planner_revision: u64,
        goal_map_instance_id: MapInstanceId,
        goal_selected_revision: u64,
    },
    StartMapMismatch {
        planner_map_instance_id: MapInstanceId,
        planner_revision: u64,
        start_map_instance_id: MapInstanceId,
        start_revision: u64,
    },
    StartOutsideMap {
        point: MapPoint,
    },
    GoalOutsideMap {
        point: MapPoint,
    },
    StartBlocked {
        point: MapPoint,
    },
    GoalBlocked {
        point: MapPoint,
    },
    NoPath,
    SearchCostOverflow,
    SearchInvariant,
}

impl std::fmt::Display for GlobalPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MapHasNoInstance => write!(
                f,
                "cannot plan against an occupancy map without a map instance"
            ),
            Self::InvalidClearanceRadius { clearance_radius_m } => write!(
                f,
                "planning clearance radius must be finite and nonnegative, got {clearance_radius_m} m"
            ),
            Self::AllocationFailed { context, requested } => write!(
                f,
                "failed to allocate {requested} elements for global planner {context}"
            ),
            Self::GoalMapMismatch {
                planner_map_instance_id,
                planner_revision,
                goal_map_instance_id,
                goal_selected_revision,
            } => write!(
                f,
                "navigation goal was selected on map {} revision {}, but the planner is bound to map {} revision {}",
                goal_map_instance_id.as_u64(),
                goal_selected_revision,
                planner_map_instance_id.as_u64(),
                planner_revision
            ),
            Self::StartMapMismatch {
                planner_map_instance_id,
                planner_revision,
                start_map_instance_id,
                start_revision,
            } => write!(
                f,
                "robot start belongs to map {} revision {}, but the planner is bound to map {} revision {}",
                start_map_instance_id.as_u64(),
                start_revision,
                planner_map_instance_id.as_u64(),
                planner_revision
            ),
            Self::StartOutsideMap { point } => write!(
                f,
                "robot start [{}, {}] m is outside the occupancy map",
                point.x_m(),
                point.y_m()
            ),
            Self::GoalOutsideMap { point } => write!(
                f,
                "navigation goal [{}, {}] m is outside the occupancy map",
                point.x_m(),
                point.y_m()
            ),
            Self::StartBlocked { point } => write!(
                f,
                "robot start [{}, {}] m is not traversable after footprint inflation",
                point.x_m(),
                point.y_m()
            ),
            Self::GoalBlocked { point } => write!(
                f,
                "navigation goal [{}, {}] m is not traversable after footprint inflation",
                point.x_m(),
                point.y_m()
            ),
            Self::NoPath => write!(
                f,
                "no traversable path connects the robot to the requested goal"
            ),
            Self::SearchCostOverflow => write!(f, "global planner path cost overflowed"),
            Self::SearchInvariant => {
                write!(f, "global planner violated its bounded search invariant")
            }
        }
    }
}

impl std::error::Error for GlobalPlanError {}

/// A global path valid only for one exact occupancy-map instance and revision.
#[derive(Clone, Debug, PartialEq)]
pub struct GlobalPath {
    map_instance_id: MapInstanceId,
    map_revision: u64,
    safety_profile: GlobalPlannerConfig,
    points: Vec<MapPoint>,
}

/// The robot position used for planning, bound to the occupancy snapshot in
/// whose frame it was expressed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanStart {
    map_instance_id: MapInstanceId,
    map_revision: u64,
    point: MapPoint,
}

/// A user-selected destination bound to one map-frame epoch.
///
/// `selected_revision` is retained for audit and visualization provenance.
/// Occupancy-content revisions within the same map instance do not redefine
/// map coordinates, so each current planner re-evaluates this point against
/// its own exact revision. A reset changes the map instance and invalidates it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointGoal {
    map_instance_id: MapInstanceId,
    selected_revision: u64,
    point: MapPoint,
}

impl PointGoal {
    pub fn for_snapshot(
        point: MapPoint,
        snapshot: &OccupancyGridSnapshot,
    ) -> Result<Self, GlobalPlanError> {
        let map_instance_id = snapshot
            .map_instance_id()
            .ok_or(GlobalPlanError::MapHasNoInstance)?;
        Ok(Self {
            map_instance_id,
            selected_revision: snapshot.revision(),
            point,
        })
    }

    pub fn map_instance_id(self) -> MapInstanceId {
        self.map_instance_id
    }

    pub fn selected_revision(self) -> u64 {
        self.selected_revision
    }

    pub fn point(self) -> MapPoint {
        self.point
    }

    pub fn is_for_map(self, snapshot: &OccupancyGridSnapshot) -> bool {
        snapshot.map_instance_id() == Some(self.map_instance_id)
    }

    pub fn was_selected_from(self, snapshot: &OccupancyGridSnapshot) -> bool {
        self.is_for_map(snapshot) && snapshot.revision() == self.selected_revision
    }
}

impl PlanStart {
    pub fn for_snapshot(
        point: MapPoint,
        snapshot: &OccupancyGridSnapshot,
    ) -> Result<Self, GlobalPlanError> {
        let map_instance_id = snapshot
            .map_instance_id()
            .ok_or(GlobalPlanError::MapHasNoInstance)?;
        Ok(Self {
            map_instance_id,
            map_revision: snapshot.revision(),
            point,
        })
    }

    pub fn map_instance_id(self) -> MapInstanceId {
        self.map_instance_id
    }

    pub fn map_revision(self) -> u64 {
        self.map_revision
    }

    pub fn point(self) -> MapPoint {
        self.point
    }

    pub fn is_current_for(self, snapshot: &OccupancyGridSnapshot) -> bool {
        snapshot.map_instance_id() == Some(self.map_instance_id)
            && snapshot.revision() == self.map_revision
    }
}

impl GlobalPath {
    pub fn map_instance_id(&self) -> MapInstanceId {
        self.map_instance_id
    }

    pub fn map_revision(&self) -> u64 {
        self.map_revision
    }

    /// Safety assumptions under which this path was computed.
    pub fn safety_profile(&self) -> GlobalPlannerConfig {
        self.safety_profile
    }

    pub fn points(&self) -> &[MapPoint] {
        self.points.as_slice()
    }

    pub fn is_current_for(&self, snapshot: &OccupancyGridSnapshot) -> bool {
        snapshot.map_instance_id() == Some(self.map_instance_id)
            && snapshot.revision() == self.map_revision
    }
}

/// Cached inflated traversability for one exact occupancy-map revision.
pub struct GlobalPlanner {
    map_instance_id: MapInstanceId,
    map_revision: u64,
    safety_profile: GlobalPlannerConfig,
    geometry: OccupancyGridGeometry,
    width: usize,
    height: usize,
    blocked: Vec<bool>,
}

impl GlobalPlanner {
    pub fn try_new(
        snapshot: &OccupancyGridSnapshot,
        config: GlobalPlannerConfig,
    ) -> Result<Self, GlobalPlanError> {
        let map_instance_id = snapshot
            .map_instance_id()
            .ok_or(GlobalPlanError::MapHasNoInstance)?;
        let geometry = snapshot.geometry();
        let width = geometry.width() as usize;
        let height = geometry.height() as usize;
        let cell_count = geometry.cell_count();
        debug_assert_eq!(snapshot.class_ids().len(), cell_count);

        let mut sources = try_bool_grid(cell_count, "blocked-cell mask")?;
        for row in 0..height {
            for column in 0..width {
                let cell = snapshot
                    .cell(column as u32, row as u32)
                    .expect("parsed occupancy snapshot dimensions match its payload");
                sources[row * width + column] = match cell {
                    OccupancyCell::Occupied => true,
                    OccupancyCell::Free => false,
                    OccupancyCell::Unknown => config.unknown_space == UnknownSpacePolicy::Blocked,
                };
            }
        }
        let blocked = inflate_blocked_cells(
            sources,
            width,
            height,
            geometry.resolution_m(),
            config.clearance_radius_m,
        )?;

        Ok(Self {
            map_instance_id,
            map_revision: snapshot.revision(),
            safety_profile: config,
            geometry,
            width,
            height,
            blocked,
        })
    }

    pub fn map_instance_id(&self) -> MapInstanceId {
        self.map_instance_id
    }

    pub fn map_revision(&self) -> u64 {
        self.map_revision
    }

    pub fn safety_profile(&self) -> GlobalPlannerConfig {
        self.safety_profile
    }

    pub fn is_current_for(&self, snapshot: &OccupancyGridSnapshot) -> bool {
        snapshot.map_instance_id() == Some(self.map_instance_id)
            && snapshot.revision() == self.map_revision
    }

    pub fn plan(&self, start: PlanStart, goal: PointGoal) -> Result<GlobalPath, GlobalPlanError> {
        if start.map_instance_id != self.map_instance_id || start.map_revision != self.map_revision
        {
            return Err(GlobalPlanError::StartMapMismatch {
                planner_map_instance_id: self.map_instance_id,
                planner_revision: self.map_revision,
                start_map_instance_id: start.map_instance_id,
                start_revision: start.map_revision,
            });
        }
        if goal.map_instance_id != self.map_instance_id {
            return Err(GlobalPlanError::GoalMapMismatch {
                planner_map_instance_id: self.map_instance_id,
                planner_revision: self.map_revision,
                goal_map_instance_id: goal.map_instance_id,
                goal_selected_revision: goal.selected_revision,
            });
        }
        let start = start.point;
        let goal = goal.point;
        let start_index = self
            .point_index(start)
            .ok_or(GlobalPlanError::StartOutsideMap { point: start })?;
        let goal_index = self
            .point_index(goal)
            .ok_or(GlobalPlanError::GoalOutsideMap { point: goal })?;
        if self.blocked[start_index] {
            return Err(GlobalPlanError::StartBlocked { point: start });
        }
        if self.blocked[goal_index] {
            return Err(GlobalPlanError::GoalBlocked { point: goal });
        }

        if start_index == goal_index {
            let point_count = if start == goal { 1 } else { 2 };
            let mut points = Vec::new();
            points.try_reserve_exact(point_count).map_err(|_| {
                GlobalPlanError::AllocationFailed {
                    context: "same-cell path points",
                    requested: point_count,
                }
            })?;
            if start != goal {
                points.push(start);
            }
            points.push(goal);
            return Ok(GlobalPath {
                map_instance_id: self.map_instance_id,
                map_revision: self.map_revision,
                safety_profile: self.safety_profile,
                points,
            });
        }

        let indices = search_grid(
            self.width,
            self.height,
            &self.blocked,
            start_index,
            goal_index,
        )?;
        let mut points = Vec::new();
        points
            .try_reserve(indices.len().saturating_add(2))
            .map_err(|_| GlobalPlanError::AllocationFailed {
                context: "path points",
                requested: indices.len().saturating_add(2),
            })?;
        points.push(start);
        for index in indices
            .iter()
            .copied()
            .skip(1)
            .take(indices.len().saturating_sub(2))
        {
            points.push(self.cell_center(index));
        }
        points.push(goal);

        Ok(GlobalPath {
            map_instance_id: self.map_instance_id,
            map_revision: self.map_revision,
            safety_profile: self.safety_profile,
            points,
        })
    }

    fn point_index(&self, point: MapPoint) -> Option<usize> {
        self.geometry.point_index(point.as_array())
    }

    fn cell_center(&self, index: usize) -> MapPoint {
        let column = index % self.width;
        let row = index / self.width;
        let lower_bound_m = self.geometry.lower_bound_m();
        let resolution_m = self.geometry.resolution_m();
        MapPoint::try_new(
            lower_bound_m[0] + (column as f64 + 0.5) * resolution_m,
            lower_bound_m[1] + (row as f64 + 0.5) * resolution_m,
        )
        .expect("parsed occupancy geometry produces finite cell centres")
    }
}

fn try_bool_grid(cell_count: usize, context: &'static str) -> Result<Vec<bool>, GlobalPlanError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(cell_count)
        .map_err(|_| GlobalPlanError::AllocationFailed {
            context,
            requested: cell_count,
        })?;
    values.resize(cell_count, false);
    Ok(values)
}

/// Inflate source cells for a circular footprint while keeping every point in
/// an unblocked result cell safe. For a positive radius, a one-cell Chebyshev
/// dilation converts cell-square separation into centre separation; an exact
/// squared Euclidean distance transform then applies the remaining radius.
/// For a positive radius the map exterior is treated as blocked independently
/// of unknown-cell policy, and equality at the clearance threshold is blocked.
/// Zero radius preserves the grid's lower-inclusive point classification.
fn inflate_blocked_cells(
    sources: Vec<bool>,
    width: usize,
    height: usize,
    resolution_m: f64,
    clearance_radius_m: f64,
) -> Result<Vec<bool>, GlobalPlanError> {
    let Some(cell_count) = width.checked_mul(height) else {
        return Err(GlobalPlanError::SearchInvariant);
    };
    if width == 0
        || height == 0
        || sources.len() != cell_count
        || !resolution_m.is_finite()
        || resolution_m <= 0.0
        || !clearance_radius_m.is_finite()
        || clearance_radius_m < 0.0
    {
        return Err(GlobalPlanError::SearchInvariant);
    }
    let mut output = try_bool_grid(cell_count, "inflated blocked-cell mask")?;
    let mut inflation = CellSquareInflation::try_new(width, height).map_err(map_inflation_error)?;
    inflation
        .inflate(
            &sources,
            &mut output,
            resolution_m,
            clearance_radius_m,
            true,
        )
        .map_err(map_inflation_error)?;
    Ok(output)
}

fn map_inflation_error(error: CellInflationError) -> GlobalPlanError {
    match error {
        CellInflationError::InvalidInput => GlobalPlanError::SearchInvariant,
        CellInflationError::AllocationFailed { context, requested } => {
            GlobalPlanError::AllocationFailed { context, requested }
        }
    }
}

#[cfg(test)]
fn conservative_squared_cell_limit(clearance_radius_m: f64, resolution_m: f64) -> u128 {
    let radius_cells = clearance_radius_m / resolution_m;
    if !radius_cells.is_finite() {
        return u128::MAX;
    }
    let outward_radius_cells = f64::from_bits(radius_cells.to_bits().saturating_add(1));
    let squared = outward_radius_cells * outward_radius_cells;
    if !squared.is_finite() {
        return u128::MAX;
    }
    // Round the threshold outward by one representable value so a tangent cell
    // cannot become traversable solely through floating-point roundoff.
    let outward = f64::from_bits(squared.to_bits().saturating_add(1));
    if outward >= u128::MAX as f64 {
        u128::MAX
    } else {
        outward.floor() as u128
    }
}

#[cfg(test)]
fn square_usize(value: usize) -> u128 {
    let value = value as u128;
    value * value
}

#[cfg(test)]
fn boundary_gap_squared(width: usize, height: usize, column: usize, row: usize) -> u128 {
    let horizontal_gap = column.min(width - 1 - column);
    let vertical_gap = row.min(height - 1 - row);
    square_usize(horizontal_gap.min(vertical_gap))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OpenNode {
    index: usize,
    path_cost: u64,
    heuristic: u64,
}

impl OpenNode {
    fn estimated_total(self) -> u128 {
        u128::from(self.path_cost) + u128::from(self.heuristic)
    }
}

impl Ord for OpenNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap. Reverse every priority comparison so the
        // smallest total, then heuristic, then index is popped first.
        other
            .estimated_total()
            .cmp(&self.estimated_total())
            .then_with(|| other.heuristic.cmp(&self.heuristic))
            .then_with(|| other.index.cmp(&self.index))
            .then_with(|| other.path_cost.cmp(&self.path_cost))
    }
}

impl PartialOrd for OpenNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn search_grid(
    width: usize,
    height: usize,
    blocked: &[bool],
    start: usize,
    goal: usize,
) -> Result<Vec<usize>, GlobalPlanError> {
    let cell_count = width
        .checked_mul(height)
        .ok_or(GlobalPlanError::SearchInvariant)?;
    if width == 0
        || height == 0
        || blocked.len() != cell_count
        || start >= cell_count
        || goal >= cell_count
        || blocked[start]
        || blocked[goal]
    {
        return Err(GlobalPlanError::SearchInvariant);
    }

    let mut costs = Vec::new();
    costs
        .try_reserve_exact(cell_count)
        .map_err(|_| GlobalPlanError::AllocationFailed {
            context: "path costs",
            requested: cell_count,
        })?;
    costs.resize(cell_count, u64::MAX);
    let mut parents = Vec::new();
    parents
        .try_reserve_exact(cell_count)
        .map_err(|_| GlobalPlanError::AllocationFailed {
            context: "path parents",
            requested: cell_count,
        })?;
    parents.resize(cell_count, usize::MAX);
    let mut closed = Vec::new();
    closed
        .try_reserve_exact(cell_count)
        .map_err(|_| GlobalPlanError::AllocationFailed {
            context: "closed cells",
            requested: cell_count,
        })?;
    closed.resize(cell_count, false);

    costs[start] = 0;
    let mut open = BinaryHeap::new();
    open.try_reserve(cell_count.min(4096))
        .map_err(|_| GlobalPlanError::AllocationFailed {
            context: "open set",
            requested: cell_count.min(4096),
        })?;
    open.push(OpenNode {
        index: start,
        path_cost: 0,
        heuristic: octile_distance(width, start, goal)?,
    });

    while let Some(current) = open.pop() {
        if closed[current.index] || current.path_cost != costs[current.index] {
            continue;
        }
        if current.index == goal {
            return reconstruct_path(&parents, start, goal);
        }
        closed[current.index] = true;

        let column = current.index % width;
        let row = current.index / width;
        for (dx, dy, step_cost) in [
            (0_isize, -1_isize, CARDINAL_COST),
            (-1, 0, CARDINAL_COST),
            (1, 0, CARDINAL_COST),
            (0, 1, CARDINAL_COST),
            (-1, -1, DIAGONAL_COST),
            (1, -1, DIAGONAL_COST),
            (-1, 1, DIAGONAL_COST),
            (1, 1, DIAGONAL_COST),
        ] {
            let Some(next_column) = column.checked_add_signed(dx) else {
                continue;
            };
            let Some(next_row) = row.checked_add_signed(dy) else {
                continue;
            };
            if next_column >= width || next_row >= height {
                continue;
            }
            let next = next_row * width + next_column;
            if blocked[next] || closed[next] {
                continue;
            }
            if dx != 0 && dy != 0 {
                let horizontal = row * width + next_column;
                let vertical = next_row * width + column;
                if blocked[horizontal] || blocked[vertical] {
                    continue;
                }
            }

            let tentative = current
                .path_cost
                .checked_add(step_cost)
                .ok_or(GlobalPlanError::SearchCostOverflow)?;
            if tentative < costs[next] {
                costs[next] = tentative;
                parents[next] = current.index;
                open.try_reserve(1)
                    .map_err(|_| GlobalPlanError::AllocationFailed {
                        context: "open-set growth",
                        requested: open.len().saturating_add(1),
                    })?;
                open.push(OpenNode {
                    index: next,
                    path_cost: tentative,
                    heuristic: octile_distance(width, next, goal)?,
                });
            }
        }
    }

    Err(GlobalPlanError::NoPath)
}

fn octile_distance(width: usize, from: usize, to: usize) -> Result<u64, GlobalPlanError> {
    let from_column = from % width;
    let from_row = from / width;
    let to_column = to % width;
    let to_row = to / width;
    let dx = from_column.abs_diff(to_column) as u64;
    let dy = from_row.abs_diff(to_row) as u64;
    let diagonal = dx.min(dy);
    let cardinal = dx.max(dy) - diagonal;
    diagonal
        .checked_mul(DIAGONAL_COST)
        .and_then(|value| {
            cardinal
                .checked_mul(CARDINAL_COST)
                .and_then(|rest| value.checked_add(rest))
        })
        .ok_or(GlobalPlanError::SearchCostOverflow)
}

fn reconstruct_path(
    parents: &[usize],
    start: usize,
    goal: usize,
) -> Result<Vec<usize>, GlobalPlanError> {
    let mut reversed = Vec::new();
    reversed.try_reserve(parents.len().min(1024)).map_err(|_| {
        GlobalPlanError::AllocationFailed {
            context: "path reconstruction",
            requested: parents.len().min(1024),
        }
    })?;
    let mut current = goal;
    for _ in 0..parents.len() {
        reversed
            .try_reserve(1)
            .map_err(|_| GlobalPlanError::AllocationFailed {
                context: "path-reconstruction growth",
                requested: reversed.len().saturating_add(1),
            })?;
        reversed.push(current);
        if current == start {
            reversed.reverse();
            return Ok(reversed);
        }
        current = *parents
            .get(current)
            .filter(|parent| **parent != usize::MAX)
            .ok_or(GlobalPlanError::SearchInvariant)?;
    }
    Err(GlobalPlanError::SearchInvariant)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::SlamMap;

    fn index(width: usize, column: usize, row: usize) -> usize {
        row * width + column
    }

    fn point(x_m: f64, y_m: f64) -> MapPoint {
        MapPoint::try_new(x_m, y_m).expect("finite test point")
    }

    fn snapshot(
        geometry: OccupancyGridGeometry,
        cells: &[OccupancyCell],
        map_instance_id: MapInstanceId,
        revision: u64,
    ) -> OccupancyGridSnapshot {
        OccupancyGridSnapshot::from_test_cells(geometry, cells, map_instance_id, revision)
    }

    fn new_map_instance_id() -> MapInstanceId {
        SlamMap::new().snapshot().instance_id()
    }

    fn brute_force_inflation(
        sources: &[bool],
        width: usize,
        height: usize,
        resolution_m: f64,
        clearance_radius_m: f64,
    ) -> Vec<bool> {
        if clearance_radius_m == 0.0 {
            return sources.to_vec();
        }
        let limit = conservative_squared_cell_limit(clearance_radius_m, resolution_m);
        let mut expected = vec![false; sources.len()];
        for row in 0..height {
            for column in 0..width {
                let mut blocked = boundary_gap_squared(width, height, column, row) <= limit;
                for source_row in 0..height {
                    for source_column in 0..width {
                        if !sources[index(width, source_column, source_row)] {
                            continue;
                        }
                        let dx = column.abs_diff(source_column).saturating_sub(1);
                        let dy = row.abs_diff(source_row).saturating_sub(1);
                        if square_usize(dx) + square_usize(dy) <= limit {
                            blocked = true;
                        }
                    }
                }
                expected[index(width, column, row)] = blocked;
            }
        }
        expected
    }

    fn path_cost(width: usize, path: &[usize]) -> u64 {
        path.windows(2)
            .map(|edge| {
                let left_column = edge[0] % width;
                let left_row = edge[0] / width;
                let right_column = edge[1] % width;
                let right_row = edge[1] / width;
                if left_column != right_column && left_row != right_row {
                    DIAGONAL_COST
                } else {
                    CARDINAL_COST
                }
            })
            .sum()
    }

    fn reference_shortest_cost(
        width: usize,
        height: usize,
        blocked: &[bool],
        start: usize,
        goal: usize,
    ) -> Option<u64> {
        let cell_count = width * height;
        let mut costs = vec![u64::MAX; cell_count];
        let mut visited = vec![false; cell_count];
        costs[start] = 0;
        for _ in 0..cell_count {
            let current = (0..cell_count)
                .filter(|candidate| !visited[*candidate] && costs[*candidate] != u64::MAX)
                .min_by_key(|candidate| (costs[*candidate], *candidate))?;
            if current == goal {
                return Some(costs[current]);
            }
            visited[current] = true;
            let column = current % width;
            let row = current / width;
            for (dx, dy, step_cost) in [
                (0_isize, -1_isize, CARDINAL_COST),
                (-1, 0, CARDINAL_COST),
                (1, 0, CARDINAL_COST),
                (0, 1, CARDINAL_COST),
                (-1, -1, DIAGONAL_COST),
                (1, -1, DIAGONAL_COST),
                (-1, 1, DIAGONAL_COST),
                (1, 1, DIAGONAL_COST),
            ] {
                let (Some(next_column), Some(next_row)) =
                    (column.checked_add_signed(dx), row.checked_add_signed(dy))
                else {
                    continue;
                };
                if next_column >= width || next_row >= height {
                    continue;
                }
                let next = index(width, next_column, next_row);
                if blocked[next] || visited[next] {
                    continue;
                }
                if dx != 0
                    && dy != 0
                    && (blocked[index(width, next_column, row)]
                        || blocked[index(width, column, next_row)])
                {
                    continue;
                }
                costs[next] = costs[next].min(costs[current] + step_cost);
            }
        }
        None
    }

    #[test]
    fn planner_config_rejects_invalid_clearance() {
        for invalid in [f64::NAN, f64::INFINITY, -f64::EPSILON] {
            assert!(matches!(
                GlobalPlannerConfig::try_new(invalid, UnknownSpacePolicy::Blocked),
                Err(GlobalPlanError::InvalidClearanceRadius { .. })
            ));
        }
    }

    #[test]
    fn planner_reuses_the_canonical_internal_boundary_classifier() {
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
        .expect("boundary-regression geometry");
        let boundary_index = 2_698_u32;
        let boundary_x_m = lower_x_m + resolution_m * f64::from(boundary_index);
        assert!(
            (boundary_x_m - lower_x_m) / resolution_m < f64::from(boundary_index),
            "fixture must round the naive quotient downward"
        );

        let mut cells = vec![OccupancyCell::Free; width as usize];
        cells[boundary_index as usize] = OccupancyCell::Occupied;
        let map_instance_id = new_map_instance_id();
        let snapshot = snapshot(geometry, &cells, map_instance_id, 7);
        let planner = GlobalPlanner::try_new(
            &snapshot,
            GlobalPlannerConfig::try_new(0.0, UnknownSpacePolicy::Traversable)
                .expect("planner config"),
        )
        .expect("planner");
        let start_point = point(boundary_x_m, resolution_m * 0.5);
        assert_eq!(
            planner.point_index(start_point),
            Some(boundary_index as usize)
        );
        let start = PlanStart::for_snapshot(start_point, &snapshot).expect("bound start");
        let goal = PointGoal::for_snapshot(
            point(
                lower_x_m + resolution_m * (f64::from(boundary_index) + 1.5),
                resolution_m * 0.5,
            ),
            &snapshot,
        )
        .expect("bound goal");
        assert_eq!(
            planner.plan(start, goal),
            Err(GlobalPlanError::StartBlocked { point: start_point })
        );
    }

    #[test]
    fn contained_final_cell_point_is_not_rejected_by_rounded_quotient() {
        let geometry = OccupancyGridGeometry::try_new(0.05, [-10.0, 0.0], 400, 1, 400)
            .expect("default-like geometry");
        let cells = vec![OccupancyCell::Free; geometry.cell_count()];
        let snapshot = snapshot(geometry, &cells, new_map_instance_id(), 1);
        let planner = GlobalPlanner::try_new(
            &snapshot,
            GlobalPlannerConfig::try_new(0.0, UnknownSpacePolicy::Blocked)
                .expect("explicit point-robot config"),
        )
        .expect("planner");
        let contained = point(9.999_999_999_999_998, 0.025);
        assert_eq!(planner.point_index(contained), Some(399));
    }

    #[test]
    fn positive_clearance_blocks_every_position_in_an_obstacle_adjacent_cell() {
        let geometry =
            OccupancyGridGeometry::try_new(1.0, [0.0, 0.0], 5, 5, 25).expect("test geometry");
        let mut cells = vec![OccupancyCell::Free; geometry.cell_count()];
        cells[index(5, 2, 2)] = OccupancyCell::Occupied;
        let snapshot = snapshot(geometry, &cells, new_map_instance_id(), 1);
        let config = GlobalPlannerConfig::try_new(0.1, UnknownSpacePolicy::Traversable)
            .expect("planner config");
        let planner = GlobalPlanner::try_new(&snapshot, config).expect("planner");
        let unsafe_start_point = point(3.01, 2.5);
        let start = PlanStart::for_snapshot(unsafe_start_point, &snapshot).expect("bound start");
        let goal = PointGoal::for_snapshot(point(3.2, 2.5), &snapshot).expect("bound goal");
        assert_eq!(
            planner.plan(start, goal),
            Err(GlobalPlanError::StartBlocked {
                point: unsafe_start_point
            })
        );
    }

    #[test]
    fn exact_footprint_tangency_is_blocked() {
        let geometry =
            OccupancyGridGeometry::try_new(1.0, [0.0, 0.0], 7, 5, 35).expect("test geometry");
        let mut cells = vec![OccupancyCell::Free; geometry.cell_count()];
        cells[index(7, 2, 2)] = OccupancyCell::Occupied;
        let snapshot = snapshot(geometry, &cells, new_map_instance_id(), 1);

        let below_tangent = f64::from_bits(1.0_f64.to_bits() - 1);
        let below = GlobalPlanner::try_new(
            &snapshot,
            GlobalPlannerConfig::try_new(below_tangent, UnknownSpacePolicy::Traversable)
                .expect("below-tangent config"),
        )
        .expect("below-tangent planner");
        let tangent = GlobalPlanner::try_new(
            &snapshot,
            GlobalPlannerConfig::try_new(1.0, UnknownSpacePolicy::Traversable)
                .expect("tangent config"),
        )
        .expect("tangent planner");

        // The threshold is rounded outward through both division and
        // squaring. One-ULP-below may therefore remain blocked, which is an
        // intentional conservative false positive rather than a false-safe.
        assert!(below.blocked[index(7, 4, 2)]);
        assert!(tangent.blocked[index(7, 4, 2)]);
    }

    #[test]
    fn positive_clearance_erodes_map_bounds_but_zero_radius_does_not() {
        let geometry =
            OccupancyGridGeometry::try_new(1.0, [0.0, 0.0], 3, 3, 9).expect("test geometry");
        let cells = vec![OccupancyCell::Free; geometry.cell_count()];
        let map_instance_id = new_map_instance_id();
        let snapshot = snapshot(geometry, &cells, map_instance_id, 1);
        let edge_point = point(0.5, 1.5);
        let edge_start = PlanStart::for_snapshot(edge_point, &snapshot).expect("bound start");
        let edge_goal = PointGoal::for_snapshot(edge_point, &snapshot).expect("bound goal");

        let point_planner = GlobalPlanner::try_new(
            &snapshot,
            GlobalPlannerConfig::try_new(0.0, UnknownSpacePolicy::Traversable)
                .expect("point planner config"),
        )
        .expect("point planner");
        assert!(point_planner.plan(edge_start, edge_goal).is_ok());

        let footprint_planner = GlobalPlanner::try_new(
            &snapshot,
            GlobalPlannerConfig::try_new(0.01, UnknownSpacePolicy::Traversable)
                .expect("footprint planner config"),
        )
        .expect("footprint planner");
        assert_eq!(
            footprint_planner.plan(edge_start, edge_goal),
            Err(GlobalPlanError::StartBlocked { point: edge_point })
        );
    }

    #[test]
    fn exact_squared_distance_transform_matches_small_grid_oracle() {
        let width = 3;
        let height = 3;
        for mask in 0_u16..(1 << (width * height)) {
            let sources = (0..width * height)
                .map(|bit| mask & (1 << bit) != 0)
                .collect::<Vec<_>>();
            for clearance_radius_m in [0.0, 0.01, 0.999, 1.0, 1.5] {
                let expected =
                    brute_force_inflation(&sources, width, height, 1.0, clearance_radius_m);
                let actual =
                    inflate_blocked_cells(sources.clone(), width, height, 1.0, clearance_radius_m)
                        .expect("bounded transform");
                assert_eq!(
                    actual, expected,
                    "mask={mask:#011b}, radius={clearance_radius_m}"
                );
            }
        }

        let width = 7;
        let height = 6;
        for seed in 0..32_usize {
            let sources = (0..width * height)
                .map(|cell| (cell * 17 + seed * 13) % 23 == 0)
                .collect::<Vec<_>>();
            for clearance_radius_m in [0.01, 0.75, 1.0, 1.25, 1.9] {
                let expected =
                    brute_force_inflation(&sources, width, height, 1.0, clearance_radius_m);
                let actual =
                    inflate_blocked_cells(sources.clone(), width, height, 1.0, clearance_radius_m)
                        .expect("bounded transform");
                assert_eq!(actual, expected, "seed={seed}, radius={clearance_radius_m}");
            }
        }
    }

    #[test]
    fn path_stamps_safety_profile_rejects_stale_start_and_reuses_same_map_goal() {
        let geometry =
            OccupancyGridGeometry::try_new(1.0, [0.0, 0.0], 5, 5, 25).expect("test geometry");
        let cells = vec![OccupancyCell::Free; geometry.cell_count()];
        let map_instance_id = new_map_instance_id();
        let stale_snapshot = snapshot(geometry, &cells, map_instance_id, 4);
        let current_snapshot = snapshot(geometry, &cells, map_instance_id, 5);
        let config = GlobalPlannerConfig::try_new(0.1, UnknownSpacePolicy::Traversable)
            .expect("planner config");
        let planner = GlobalPlanner::try_new(&current_snapshot, config).expect("planner");
        let stale_start =
            PlanStart::for_snapshot(point(2.5, 2.5), &stale_snapshot).expect("stale bound start");
        let selected_goal =
            PointGoal::for_snapshot(point(2.5, 2.5), &stale_snapshot).expect("selected goal");
        assert_eq!(selected_goal.selected_revision(), 4);
        assert!(selected_goal.is_for_map(&current_snapshot));
        assert!(!selected_goal.was_selected_from(&current_snapshot));
        assert!(matches!(
            planner.plan(stale_start, selected_goal),
            Err(GlobalPlanError::StartMapMismatch {
                planner_revision: 5,
                start_revision: 4,
                ..
            })
        ));

        let current_start = PlanStart::for_snapshot(point(2.5, 2.5), &current_snapshot)
            .expect("current bound start");
        let path = planner
            .plan(current_start, selected_goal)
            .expect("same-map goal is re-evaluated on the current revision");
        assert_eq!(path.safety_profile(), config);
        assert!(path.is_current_for(&current_snapshot));
        assert!(!path.is_current_for(&stale_snapshot));

        let foreign_snapshot = snapshot(geometry, &cells, new_map_instance_id(), 1);
        let foreign_goal =
            PointGoal::for_snapshot(point(2.5, 2.5), &foreign_snapshot).expect("foreign goal");
        assert!(matches!(
            planner.plan(current_start, foreign_goal),
            Err(GlobalPlanError::GoalMapMismatch {
                goal_selected_revision: 1,
                ..
            })
        ));
    }

    #[test]
    fn astar_is_deterministic_and_routes_through_the_only_gap() {
        let width = 7;
        let height = 5;
        let mut blocked = vec![false; width * height];
        for row in 0..height {
            if row != 3 {
                blocked[index(width, 3, row)] = true;
            }
        }
        let start = index(width, 1, 1);
        let goal = index(width, 5, 1);
        let first = search_grid(width, height, &blocked, start, goal).expect("reachable path");
        let second = search_grid(width, height, &blocked, start, goal).expect("same path");
        assert_eq!(first, second);
        assert!(first.contains(&index(width, 3, 3)));
        assert_eq!(first.first(), Some(&start));
        assert_eq!(first.last(), Some(&goal));
    }

    #[test]
    fn diagonal_move_cannot_cut_a_blocked_corner() {
        let width = 2;
        let height = 2;
        let blocked = vec![false, true, true, false];
        assert_eq!(
            search_grid(width, height, &blocked, 0, 3),
            Err(GlobalPlanError::NoPath)
        );
    }

    #[test]
    fn diagonal_move_cannot_cut_past_either_single_corner_blocker() {
        let width = 2;
        let height = 2;
        for blocked in [
            vec![false, true, false, false],
            vec![false, false, true, false],
        ] {
            let path = search_grid(width, height, &blocked, 0, 3).expect("cardinal detour");
            assert_eq!(path.len(), 3);
            assert_eq!(path_cost(width, &path), 2 * CARDINAL_COST);
        }
    }

    #[test]
    fn astar_matches_reference_shortest_cost_on_every_three_by_three_grid() {
        let width = 3;
        let height = 3;
        let cell_count = width * height;
        for mask in 0_u16..(1 << cell_count) {
            let blocked = (0..cell_count)
                .map(|bit| mask & (1 << bit) != 0)
                .collect::<Vec<_>>();
            for start in 0..cell_count {
                if blocked[start] {
                    continue;
                }
                for goal in 0..cell_count {
                    if blocked[goal] {
                        continue;
                    }
                    let expected = reference_shortest_cost(width, height, &blocked, start, goal);
                    let actual = search_grid(width, height, &blocked, start, goal);
                    match (expected, actual) {
                        (Some(expected), Ok(path)) => {
                            assert_eq!(path.first(), Some(&start));
                            assert_eq!(path.last(), Some(&goal));
                            assert_eq!(path_cost(width, &path), expected);
                        }
                        (None, Err(GlobalPlanError::NoPath)) => {}
                        (expected, actual) => panic!(
                            "A* disagreed with reference: mask={mask:#011b}, start={start}, goal={goal}, expected={expected:?}, actual={actual:?}"
                        ),
                    }
                }
            }
        }
    }

    #[test]
    fn astar_handles_single_cell_and_single_axis_maps() {
        assert_eq!(search_grid(1, 1, &[false], 0, 0), Ok(vec![0]));
        assert_eq!(
            search_grid(1, 5, &[false; 5], 0, 4).map(|path| path_cost(1, &path)),
            Ok(4 * CARDINAL_COST)
        );
        assert_eq!(
            search_grid(5, 1, &[false; 5], 0, 4).map(|path| path_cost(5, &path)),
            Ok(4 * CARDINAL_COST)
        );
    }

    #[test]
    fn unreachable_goal_is_a_typed_error() {
        let width = 3;
        let height = 3;
        let blocked = vec![false, true, false, false, true, false, false, true, false];
        assert_eq!(
            search_grid(width, height, &blocked, 0, 2),
            Err(GlobalPlanError::NoPath)
        );
    }

    #[test]
    fn octile_heuristic_matches_cardinal_and_diagonal_costs() {
        let width = 10;
        let origin = index(width, 1, 1);
        assert_eq!(
            octile_distance(width, origin, index(width, 4, 1)),
            Ok(3 * CARDINAL_COST)
        );
        assert_eq!(
            octile_distance(width, origin, index(width, 4, 4)),
            Ok(3 * DIAGONAL_COST)
        );
        assert_eq!(
            octile_distance(width, origin, index(width, 5, 3)),
            Ok(2 * DIAGONAL_COST + 2 * CARDINAL_COST)
        );
    }
}
