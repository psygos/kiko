use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use crate::dense::occupancy::{OccupancyCell, OccupancyGridGeometry, OccupancyGridSnapshot};
use crate::map::MapInstanceId;

use super::cell_inflation::{CellInflationError, CellSquareInflation};
use super::frames::{MapFrame, PlanarPoint};

const CARDINAL_COST: u64 = 10;
const DIAGONAL_COST: u64 = 14;
static NEXT_GLOBAL_PLANNER_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

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

/// A closed, finite rectangle constraining robot-center path geometry in the
/// displayed occupancy-map frame.
///
/// The private fields make invalid or reversed bounds unrepresentable. A
/// bounded planner admits only start/goal points and grid-cell centres inside
/// this rectangle; because the rectangle is convex, every straight path
/// segment between admitted points also remains inside it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MapTraversalBoundary {
    minimum_x_m: f64,
    minimum_y_m: f64,
    maximum_x_m: f64,
    maximum_y_m: f64,
}

impl MapTraversalBoundary {
    pub fn try_new(
        minimum_x_m: f64,
        minimum_y_m: f64,
        maximum_x_m: f64,
        maximum_y_m: f64,
    ) -> Result<Self, MapTraversalBoundaryError> {
        for (component, value) in [
            (MapTraversalBoundaryComponent::MinimumX, minimum_x_m),
            (MapTraversalBoundaryComponent::MinimumY, minimum_y_m),
            (MapTraversalBoundaryComponent::MaximumX, maximum_x_m),
            (MapTraversalBoundaryComponent::MaximumY, maximum_y_m),
        ] {
            if !value.is_finite() {
                return Err(MapTraversalBoundaryError::NonFinite { component, value });
            }
        }
        if minimum_x_m >= maximum_x_m {
            return Err(MapTraversalBoundaryError::EmptyOrReversedX {
                minimum_x_m,
                maximum_x_m,
            });
        }
        if minimum_y_m >= maximum_y_m {
            return Err(MapTraversalBoundaryError::EmptyOrReversedY {
                minimum_y_m,
                maximum_y_m,
            });
        }
        Ok(Self {
            minimum_x_m,
            minimum_y_m,
            maximum_x_m,
            maximum_y_m,
        })
    }

    pub const fn minimum_x_m(self) -> f64 {
        self.minimum_x_m
    }

    pub const fn minimum_y_m(self) -> f64 {
        self.minimum_y_m
    }

    pub const fn maximum_x_m(self) -> f64 {
        self.maximum_x_m
    }

    pub const fn maximum_y_m(self) -> f64 {
        self.maximum_y_m
    }

    pub fn contains(self, point: MapPoint) -> bool {
        point.x_m() >= self.minimum_x_m
            && point.x_m() <= self.maximum_x_m
            && point.y_m() >= self.minimum_y_m
            && point.y_m() <= self.maximum_y_m
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapTraversalBoundaryComponent {
    MinimumX,
    MinimumY,
    MaximumX,
    MaximumY,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MapTraversalBoundaryError {
    NonFinite {
        component: MapTraversalBoundaryComponent,
        value: f64,
    },
    EmptyOrReversedX {
        minimum_x_m: f64,
        maximum_x_m: f64,
    },
    EmptyOrReversedY {
        minimum_y_m: f64,
        maximum_y_m: f64,
    },
}

impl std::fmt::Display for MapTraversalBoundaryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "invalid map traversal boundary: {self:?}")
    }
}

impl std::error::Error for MapTraversalBoundaryError {}

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
            clearance_radius_m: if clearance_radius_m == 0.0 {
                0.0
            } else {
                clearance_radius_m
            },
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
    StartOutsideTraversalBoundary {
        point: MapPoint,
        boundary: MapTraversalBoundary,
    },
    GoalOutsideTraversalBoundary {
        point: MapPoint,
        boundary: MapTraversalBoundary,
    },
    StartCellCenterOutsideTraversalBoundary {
        point: MapPoint,
        cell_center: MapPoint,
        boundary: MapTraversalBoundary,
    },
    GoalCellCenterOutsideTraversalBoundary {
        point: MapPoint,
        cell_center: MapPoint,
        boundary: MapTraversalBoundary,
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
    ZeroPlannerInstanceId,
    PlannerInstanceIdExhausted,
    PlannerInvocationIdExhausted,
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
            Self::StartOutsideTraversalBoundary { point, boundary } => write!(
                f,
                "robot start [{}, {}] m is outside traversal boundary [{}, {}]..=[{}, {}] m",
                point.x_m(),
                point.y_m(),
                boundary.minimum_x_m(),
                boundary.minimum_y_m(),
                boundary.maximum_x_m(),
                boundary.maximum_y_m()
            ),
            Self::GoalOutsideTraversalBoundary { point, boundary } => write!(
                f,
                "navigation goal [{}, {}] m is outside traversal boundary [{}, {}]..=[{}, {}] m",
                point.x_m(),
                point.y_m(),
                boundary.minimum_x_m(),
                boundary.minimum_y_m(),
                boundary.maximum_x_m(),
                boundary.maximum_y_m()
            ),
            Self::StartCellCenterOutsideTraversalBoundary {
                point, cell_center, ..
            } => write!(
                f,
                "robot start [{}, {}] m belongs to cell center [{}, {}] m outside the traversal boundary",
                point.x_m(),
                point.y_m(),
                cell_center.x_m(),
                cell_center.y_m()
            ),
            Self::GoalCellCenterOutsideTraversalBoundary {
                point, cell_center, ..
            } => write!(
                f,
                "navigation goal [{}, {}] m belongs to cell center [{}, {}] m outside the traversal boundary",
                point.x_m(),
                point.y_m(),
                cell_center.x_m(),
                cell_center.y_m()
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
            Self::ZeroPlannerInstanceId => {
                write!(f, "global planner instance ID must be nonzero")
            }
            Self::PlannerInstanceIdExhausted => {
                write!(f, "global planner instance ID space is exhausted")
            }
            Self::PlannerInvocationIdExhausted => {
                write!(f, "global planner invocation ID space is exhausted")
            }
        }
    }
}

impl std::error::Error for GlobalPlanError {}

/// Process-local identity of one constructed planner and its immutable map/config contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GlobalPlannerInstanceId(NonZeroU64);

impl GlobalPlannerInstanceId {
    /// Parse a recorded planner-instance ID for deterministic replay.
    pub fn try_new(raw: u64) -> Result<Self, GlobalPlanError> {
        NonZeroU64::new(raw)
            .map(Self)
            .ok_or(GlobalPlanError::ZeroPlannerInstanceId)
    }

    pub fn as_u64(self) -> u64 {
        self.0.get()
    }

    fn allocate() -> Result<Self, GlobalPlanError> {
        let raw = NEXT_GLOBAL_PLANNER_INSTANCE_ID
            .fetch_update(
                AtomicOrdering::Relaxed,
                AtomicOrdering::Relaxed,
                |current| current.checked_add(1),
            )
            .map_err(|_| GlobalPlanError::PlannerInstanceIdExhausted)?;
        let value = NonZeroU64::new(raw)
            .expect("global planner instance allocation starts at one and never wraps");
        Ok(Self(value))
    }
}

/// Monotonic identity of one planning attempt within a planner instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GlobalPlannerInvocationId(NonZeroU64);

impl GlobalPlannerInvocationId {
    pub fn as_u64(self) -> u64 {
        self.0.get()
    }
}

/// Exact deterministic algorithm contract used to produce a global path.
///
/// Increment this revision whenever neighbor ordering, costs, tie-breaking,
/// inflation semantics, or path reconstruction changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u32)]
pub enum GlobalPlannerRevision {
    OctileAStarV1 = 1,
}

impl GlobalPlannerRevision {
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// Collision-free, content-bearing identity of one produced live plan.
///
/// This is deliberately not a hash. It retains the exact parsed boundary
/// values and the unique process-local invocation that produced the path. An
/// exact replay intentionally reconstructs an equal identity from its recorded
/// planner-instance ID and the same ordered planning attempts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobalPlanIdentity {
    planner_instance_id: GlobalPlannerInstanceId,
    planner_revision: GlobalPlannerRevision,
    invocation_id: GlobalPlannerInvocationId,
    map_instance_id: MapInstanceId,
    map_revision: u64,
    start: PlanStart,
    goal: PointGoal,
    safety_profile: GlobalPlannerConfig,
    traversal_boundary: Option<MapTraversalBoundary>,
}

impl GlobalPlanIdentity {
    pub fn planner_instance_id(self) -> GlobalPlannerInstanceId {
        self.planner_instance_id
    }

    pub fn planner_revision(self) -> GlobalPlannerRevision {
        self.planner_revision
    }

    pub fn invocation_id(self) -> GlobalPlannerInvocationId {
        self.invocation_id
    }

    pub fn map_instance_id(self) -> MapInstanceId {
        self.map_instance_id
    }

    pub fn map_revision(self) -> u64 {
        self.map_revision
    }

    pub fn start(self) -> PlanStart {
        self.start
    }

    pub fn goal(self) -> PointGoal {
        self.goal
    }

    pub fn safety_profile(self) -> GlobalPlannerConfig {
        self.safety_profile
    }

    pub fn traversal_boundary(self) -> Option<MapTraversalBoundary> {
        self.traversal_boundary
    }
}

/// A global path valid only for the exact contract in its identity.
#[derive(Clone, Debug, PartialEq)]
pub struct GlobalPath {
    identity: GlobalPlanIdentity,
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
    pub fn identity(&self) -> GlobalPlanIdentity {
        self.identity
    }

    pub fn map_instance_id(&self) -> MapInstanceId {
        self.identity.map_instance_id
    }

    pub fn map_revision(&self) -> u64 {
        self.identity.map_revision
    }

    /// Safety assumptions under which this path was computed.
    pub fn safety_profile(&self) -> GlobalPlannerConfig {
        self.identity.safety_profile
    }

    pub fn traversal_boundary(&self) -> Option<MapTraversalBoundary> {
        self.identity.traversal_boundary
    }

    pub fn points(&self) -> &[MapPoint] {
        self.points.as_slice()
    }

    pub fn is_current_for(&self, snapshot: &OccupancyGridSnapshot) -> bool {
        snapshot.map_instance_id() == Some(self.identity.map_instance_id)
            && snapshot.revision() == self.identity.map_revision
    }
}

/// Cached inflated traversability for one exact occupancy-map revision.
pub struct GlobalPlanner {
    instance_id: GlobalPlannerInstanceId,
    next_invocation_id: Option<NonZeroU64>,
    map_instance_id: MapInstanceId,
    map_revision: u64,
    safety_profile: GlobalPlannerConfig,
    traversal_boundary: Option<MapTraversalBoundary>,
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
        Self::try_new_with_optional_boundary_and_instance_id(
            snapshot,
            config,
            None,
            GlobalPlannerInstanceId::allocate()?,
        )
    }

    /// Construct a planner whose actual output path cannot leave `boundary`.
    pub fn try_new_bounded(
        snapshot: &OccupancyGridSnapshot,
        config: GlobalPlannerConfig,
        boundary: MapTraversalBoundary,
    ) -> Result<Self, GlobalPlanError> {
        Self::try_new_with_optional_boundary_and_instance_id(
            snapshot,
            config,
            Some(boundary),
            GlobalPlannerInstanceId::allocate()?,
        )
    }

    /// Reconstruct a planner under an exact recorded identity for replay.
    ///
    /// Reusing an ID is authority to reproduce that recorded planner. Ordinary
    /// live construction must use [`Self::try_new`] so unrelated planners
    /// receive distinct instance IDs.
    pub fn try_new_with_instance_id(
        snapshot: &OccupancyGridSnapshot,
        config: GlobalPlannerConfig,
        instance_id: GlobalPlannerInstanceId,
    ) -> Result<Self, GlobalPlanError> {
        Self::try_new_with_optional_boundary_and_instance_id(snapshot, config, None, instance_id)
    }

    /// Reconstruct a boundary-constrained planner under an exact recorded
    /// identity for replay.
    pub fn try_new_bounded_with_instance_id(
        snapshot: &OccupancyGridSnapshot,
        config: GlobalPlannerConfig,
        boundary: MapTraversalBoundary,
        instance_id: GlobalPlannerInstanceId,
    ) -> Result<Self, GlobalPlanError> {
        Self::try_new_with_optional_boundary_and_instance_id(
            snapshot,
            config,
            Some(boundary),
            instance_id,
        )
    }

    fn try_new_with_optional_boundary_and_instance_id(
        snapshot: &OccupancyGridSnapshot,
        config: GlobalPlannerConfig,
        traversal_boundary: Option<MapTraversalBoundary>,
        instance_id: GlobalPlannerInstanceId,
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
        let mut blocked = inflate_blocked_cells(
            sources,
            width,
            height,
            geometry.resolution_m(),
            config.clearance_radius_m,
        )?;
        if let Some(boundary) = traversal_boundary {
            for (index, cell_blocked) in blocked.iter_mut().enumerate() {
                if !boundary.contains(cell_center_for_geometry(geometry, width, index)) {
                    *cell_blocked = true;
                }
            }
        }

        Ok(Self {
            instance_id,
            next_invocation_id: Some(NonZeroU64::MIN),
            map_instance_id,
            map_revision: snapshot.revision(),
            safety_profile: config,
            traversal_boundary,
            geometry,
            width,
            height,
            blocked,
        })
    }

    pub fn map_instance_id(&self) -> MapInstanceId {
        self.map_instance_id
    }

    pub fn instance_id(&self) -> GlobalPlannerInstanceId {
        self.instance_id
    }

    pub fn revision(&self) -> GlobalPlannerRevision {
        GlobalPlannerRevision::OctileAStarV1
    }

    pub fn map_revision(&self) -> u64 {
        self.map_revision
    }

    pub fn safety_profile(&self) -> GlobalPlannerConfig {
        self.safety_profile
    }

    pub fn traversal_boundary(&self) -> Option<MapTraversalBoundary> {
        self.traversal_boundary
    }

    pub fn is_current_for(&self, snapshot: &OccupancyGridSnapshot) -> bool {
        snapshot.map_instance_id() == Some(self.map_instance_id)
            && snapshot.revision() == self.map_revision
    }

    pub fn plan(
        &mut self,
        start: PlanStart,
        goal: PointGoal,
    ) -> Result<GlobalPath, GlobalPlanError> {
        let invocation_id = self.next_invocation_id()?;
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
        let identity = self.plan_identity(invocation_id, start, goal);
        let start = start.point;
        let goal = goal.point;
        if let Some(boundary) = self.traversal_boundary {
            if !boundary.contains(start) {
                return Err(GlobalPlanError::StartOutsideTraversalBoundary {
                    point: start,
                    boundary,
                });
            }
            if !boundary.contains(goal) {
                return Err(GlobalPlanError::GoalOutsideTraversalBoundary {
                    point: goal,
                    boundary,
                });
            }
        }
        let start_index = self
            .point_index(start)
            .ok_or(GlobalPlanError::StartOutsideMap { point: start })?;
        let goal_index = self
            .point_index(goal)
            .ok_or(GlobalPlanError::GoalOutsideMap { point: goal })?;
        if let Some(boundary) = self.traversal_boundary {
            let start_cell_center = self.cell_center(start_index);
            if !boundary.contains(start_cell_center) {
                return Err(GlobalPlanError::StartCellCenterOutsideTraversalBoundary {
                    point: start,
                    cell_center: start_cell_center,
                    boundary,
                });
            }
            let goal_cell_center = self.cell_center(goal_index);
            if !boundary.contains(goal_cell_center) {
                return Err(GlobalPlanError::GoalCellCenterOutsideTraversalBoundary {
                    point: goal,
                    cell_center: goal_cell_center,
                    boundary,
                });
            }
        }
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
            return Ok(GlobalPath { identity, points });
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

        Ok(GlobalPath { identity, points })
    }

    fn next_invocation_id(&mut self) -> Result<GlobalPlannerInvocationId, GlobalPlanError> {
        let current = self
            .next_invocation_id
            .take()
            .ok_or(GlobalPlanError::PlannerInvocationIdExhausted)?;
        self.next_invocation_id = current.get().checked_add(1).and_then(NonZeroU64::new);
        Ok(GlobalPlannerInvocationId(current))
    }

    fn plan_identity(
        &self,
        invocation_id: GlobalPlannerInvocationId,
        start: PlanStart,
        goal: PointGoal,
    ) -> GlobalPlanIdentity {
        debug_assert_eq!(start.map_instance_id, self.map_instance_id);
        debug_assert_eq!(start.map_revision, self.map_revision);
        debug_assert_eq!(goal.map_instance_id, self.map_instance_id);
        GlobalPlanIdentity {
            planner_instance_id: self.instance_id,
            planner_revision: self.revision(),
            invocation_id,
            map_instance_id: self.map_instance_id,
            map_revision: self.map_revision,
            start,
            goal,
            safety_profile: self.safety_profile,
            traversal_boundary: self.traversal_boundary,
        }
    }

    fn point_index(&self, point: MapPoint) -> Option<usize> {
        self.geometry.point_index(point.as_array())
    }

    fn cell_center(&self, index: usize) -> MapPoint {
        cell_center_for_geometry(self.geometry, self.width, index)
    }
}

fn cell_center_for_geometry(
    geometry: OccupancyGridGeometry,
    width: usize,
    index: usize,
) -> MapPoint {
    let column = index % width;
    let row = index / width;
    let lower_bound_m = geometry.lower_bound_m();
    let resolution_m = geometry.resolution_m();
    MapPoint::try_new(
        lower_bound_m[0] + (column as f64 + 0.5) * resolution_m,
        lower_bound_m[1] + (row as f64 + 0.5) * resolution_m,
    )
    .expect("parsed occupancy geometry produces finite cell centres")
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
        let negative_zero = GlobalPlannerConfig::try_new(-0.0, UnknownSpacePolicy::Blocked)
            .expect("negative zero has canonical zero-radius semantics");
        assert_eq!(
            negative_zero.clearance_radius_m().to_bits(),
            0.0_f64.to_bits()
        );
    }

    #[test]
    fn bounded_planner_retains_constraint_and_cannot_take_shorter_outside_route() {
        let geometry =
            OccupancyGridGeometry::try_new(1.0, [0.0, 0.0], 7, 5, 35).expect("test geometry");
        let mut cells = vec![OccupancyCell::Occupied; 35];
        for row in [0, 3] {
            cells.chunks_exact_mut(7).nth(row).expect("fixture row")[1..=5]
                .fill(OccupancyCell::Free);
        }
        for row in cells.chunks_exact_mut(7).skip(1).take(2) {
            row[1] = OccupancyCell::Free;
            row[5] = OccupancyCell::Free;
        }
        let snapshot = snapshot(geometry, &cells, new_map_instance_id(), 1);
        let config = GlobalPlannerConfig::try_new(0.0, UnknownSpacePolicy::Blocked)
            .expect("point-robot planner");
        let start = PlanStart::for_snapshot(point(1.5, 1.5), &snapshot).expect("map-bound start");
        let goal = PointGoal::for_snapshot(point(5.5, 1.5), &snapshot).expect("map-bound goal");

        let mut unbounded = GlobalPlanner::try_new(&snapshot, config).expect("unbounded planner");
        let unbounded_path = unbounded.plan(start, goal).expect("outside route");
        assert!(
            unbounded_path
                .points()
                .iter()
                .any(|point| point.y_m() < 1.0),
            "fixture must expose the shorter route outside the future boundary"
        );

        let boundary = MapTraversalBoundary::try_new(0.0, 1.0, 7.0, 4.0).expect("closed boundary");
        let mut bounded =
            GlobalPlanner::try_new_bounded(&snapshot, config, boundary).expect("bounded planner");
        let bounded_path = bounded.plan(start, goal).expect("inside route");
        assert_eq!(bounded_path.traversal_boundary(), Some(boundary));
        assert_eq!(bounded_path.identity().traversal_boundary(), Some(boundary));
        assert!(
            bounded_path
                .points()
                .iter()
                .copied()
                .all(|point| boundary.contains(point))
        );
        assert!(
            bounded_path.points().iter().any(|point| point.y_m() > 3.0),
            "bounded planner must take the available inside route"
        );
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
        let mut planner = GlobalPlanner::try_new(
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
        let mut planner = GlobalPlanner::try_new(&snapshot, config).expect("planner");
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

        let mut point_planner = GlobalPlanner::try_new(
            &snapshot,
            GlobalPlannerConfig::try_new(0.0, UnknownSpacePolicy::Traversable)
                .expect("point planner config"),
        )
        .expect("point planner");
        assert!(point_planner.plan(edge_start, edge_goal).is_ok());

        let mut footprint_planner = GlobalPlanner::try_new(
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
        let mut planner = GlobalPlanner::try_new(&current_snapshot, config).expect("planner");
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
    fn exact_plan_identity_binds_contract_and_distinguishes_every_produced_plan() {
        let geometry =
            OccupancyGridGeometry::try_new(1.0, [0.0, 0.0], 5, 5, 25).expect("test geometry");
        let cells = vec![OccupancyCell::Free; geometry.cell_count()];
        let snapshot = snapshot(geometry, &cells, new_map_instance_id(), 9);
        let config = GlobalPlannerConfig::try_new(0.25, UnknownSpacePolicy::Traversable)
            .expect("planner config");
        let start = PlanStart::for_snapshot(point(1.5, 1.5), &snapshot).expect("bound start");
        let goal = PointGoal::for_snapshot(point(3.5, 3.5), &snapshot).expect("bound goal");

        let mut first_planner = GlobalPlanner::try_new(&snapshot, config).expect("first planner");
        let first = first_planner.plan(start, goal).expect("first plan");
        let repeated = first_planner.plan(start, goal).expect("repeated plan");
        assert_eq!(first.points(), repeated.points());

        let first_identity = first.identity();
        assert_eq!(
            first_identity.planner_instance_id(),
            first_planner.instance_id()
        );
        assert_eq!(
            first_identity.planner_revision(),
            GlobalPlannerRevision::OctileAStarV1
        );
        assert_eq!(first_identity.planner_revision().as_u32(), 1);
        assert_eq!(first_identity.invocation_id().as_u64(), 1);
        assert_eq!(repeated.identity().invocation_id().as_u64(), 2);
        assert_ne!(first.identity(), repeated.identity());
        assert_eq!(
            first_identity.map_instance_id(),
            snapshot
                .map_instance_id()
                .expect("test snapshot has an instance")
        );
        assert_eq!(first_identity.map_revision(), snapshot.revision());
        assert_eq!(first_identity.start(), start);
        assert_eq!(first_identity.goal(), goal);
        assert_eq!(first_identity.safety_profile(), config);

        let mut second_planner = GlobalPlanner::try_new(&snapshot, config).expect("second planner");
        let equivalent = second_planner
            .plan(start, goal)
            .expect("equivalent plan from a distinct planner");
        assert_eq!(first.points(), equivalent.points());
        assert_ne!(
            first.identity().planner_instance_id(),
            equivalent.identity().planner_instance_id()
        );
        assert_ne!(first.identity(), equivalent.identity());
    }

    #[test]
    fn recorded_planner_instance_makes_replay_identity_allocator_independent() {
        assert_eq!(
            GlobalPlannerInstanceId::try_new(0),
            Err(GlobalPlanError::ZeroPlannerInstanceId)
        );
        let recorded_instance =
            GlobalPlannerInstanceId::try_new(42_424).expect("nonzero recorded planner instance");
        let geometry =
            OccupancyGridGeometry::try_new(1.0, [0.0, 0.0], 3, 3, 9).expect("test geometry");
        let cells = vec![OccupancyCell::Free; geometry.cell_count()];
        let snapshot = snapshot(geometry, &cells, new_map_instance_id(), 3);
        let config = GlobalPlannerConfig::try_new(0.0, UnknownSpacePolicy::Traversable)
            .expect("planner config");
        let start = PlanStart::for_snapshot(point(0.5, 0.5), &snapshot).expect("bound start");
        let goal = PointGoal::for_snapshot(point(2.5, 2.5), &snapshot).expect("bound goal");

        let mut first =
            GlobalPlanner::try_new_with_instance_id(&snapshot, config, recorded_instance)
                .expect("first replay planner");
        let expected = first
            .plan(start, goal)
            .expect("first replay plan")
            .identity();

        let _unrelated = GlobalPlanner::try_new(&snapshot, config).expect("unrelated live planner");

        let mut replayed =
            GlobalPlanner::try_new_with_instance_id(&snapshot, config, recorded_instance)
                .expect("reconstructed replay planner");
        let actual = replayed
            .plan(start, goal)
            .expect("reconstructed replay plan")
            .identity();
        assert_eq!(actual, expected);
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
