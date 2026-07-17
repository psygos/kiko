//! Deterministic, hardware-independent dynamic obstacle costs from dense depth.
//!
//! The grid is expressed in [`LocalCostmapFrame`] coordinates frozen at one
//! depth frame's capture time. Its axes use base convention `+x` forward and
//! `+y` left. Depth samples
//! use Kiko's optical convention (`+x` right, `+y` down, `+z` forward) and are
//! projected in metres through an explicit calibrated transform into the base
//! frame. No camera-to-base alignment is assumed by this module.
//!
//! Each accepted frame replaces the previous grid. This gives the capacity-one
//! navigation depth route an exact newest-frame meaning: only ray segments
//! visible through the configured obstacle-height slab carry free evidence;
//! unseen cells become unknown. Unknown, expired, inflated, and out-of-bounds
//! locations are all non-traversable.

use std::num::NonZeroU32;
use std::time::Duration;

use crate::dense::occupancy::{
    DepthCameraModel, DepthRangeMeters, HeightRangeMeters, OccupancyGridGeometry,
};
use crate::{
    DepthImage, DepthObservation, DeviceSessionId, DeviceTimestamp, FrameDimensions, FrameId,
    HostMonotonicTimestamp, Pose,
};

use super::cell_inflation::{CellInflationError, CellSquareInflation};
use super::{
    LocalCostmapFrame, LocalCostmapToOdom, OdomFrame, PlanarPoint, PlanarPointError,
    PlanarTransformError,
};

const CLASS_UNKNOWN: u8 = LocalCostmapCell::Unknown as u8;
const CLASS_FREE: u8 = LocalCostmapCell::Free as u8;
const CLASS_INFLATED: u8 = LocalCostmapCell::Inflated as u8;
const CLASS_OCCUPIED: u8 = LocalCostmapCell::Occupied as u8;

/// Explicit calibrated rigid transform from Kiko's tracking camera into the
/// robot base frame (`+x` forward, `+y` left, `+z` up).
///
/// [`DepthCameraModel`] separately carries the depth-optical-to-tracking
/// extrinsic. The two transforms are composed once when the costmap is built.
#[derive(Clone, Copy, Debug)]
pub struct TrackingCameraToBase(Pose);

impl TrackingCameraToBase {
    pub fn new(pose: Pose) -> Self {
        Self(pose)
    }

    pub fn pose(self) -> Pose {
        self.0
    }
}

/// Configuration parsed into bounded, finite domain values.
#[derive(Clone, Debug)]
pub struct LocalCostmapConfig {
    geometry: OccupancyGridGeometry,
    camera: DepthCameraModel,
    tracking_to_base: TrackingCameraToBase,
    obstacle_height_range: HeightRangeMeters,
    depth_range: DepthRangeMeters,
    sampling_block: NonZeroU32,
    footprint_radius_m: f64,
    clearance_m: f64,
    inflation_radius_m: f64,
    max_observation_age_ns: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LocalCostmapConfigError {
    ZeroSamplingBlock,
    UnsafeSamplingBlock {
        sampling_block: u32,
    },
    InvalidFootprintRadius {
        radius_m: f64,
    },
    InvalidClearance {
        clearance_m: f64,
    },
    InflationRadiusNotFinite {
        radius_m: f64,
        clearance_m: f64,
    },
    ZeroMaximumObservationAge,
    MaximumObservationAgeNotRepresentable {
        nanoseconds: u128,
    },
    InflatedFootprintOutsideGrid {
        axis: usize,
        lower_bound_m: f64,
        upper_bound_m: f64,
        required_radius_m: f64,
    },
}

impl std::fmt::Display for LocalCostmapConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroSamplingBlock => {
                write!(
                    f,
                    "local-costmap nearest-depth sampling block must be nonzero"
                )
            }
            Self::UnsafeSamplingBlock { sampling_block } => write!(
                f,
                "local-costmap safety requires sampling every depth pixel; got a {sampling_block}x{sampling_block} nearest-depth block"
            ),
            Self::InvalidFootprintRadius { radius_m } => write!(
                f,
                "robot footprint radius must be finite and positive, got {radius_m} m"
            ),
            Self::InvalidClearance { clearance_m } => write!(
                f,
                "robot obstacle clearance must be finite and nonnegative, got {clearance_m} m"
            ),
            Self::InflationRadiusNotFinite {
                radius_m,
                clearance_m,
            } => write!(
                f,
                "robot footprint radius plus clearance is not finite: {radius_m} m + {clearance_m} m"
            ),
            Self::ZeroMaximumObservationAge => {
                write!(f, "local-costmap maximum observation age must be positive")
            }
            Self::MaximumObservationAgeNotRepresentable { nanoseconds } => write!(
                f,
                "local-costmap maximum observation age {nanoseconds} ns does not fit u64"
            ),
            Self::InflatedFootprintOutsideGrid {
                axis,
                lower_bound_m,
                upper_bound_m,
                required_radius_m,
            } => write!(
                f,
                "local-costmap base-frame bounds on axis {axis}, [{lower_bound_m}, {upper_bound_m}) m, do not contain the inflated robot footprint [-{required_radius_m}, {required_radius_m}] m"
            ),
        }
    }
}

impl std::error::Error for LocalCostmapConfigError {}

impl LocalCostmapConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        geometry: OccupancyGridGeometry,
        camera: DepthCameraModel,
        tracking_to_base: TrackingCameraToBase,
        obstacle_height_range: HeightRangeMeters,
        depth_range: DepthRangeMeters,
        sampling_block: u32,
        footprint_radius_m: f64,
        clearance_m: f64,
        max_observation_age: Duration,
    ) -> Result<Self, LocalCostmapConfigError> {
        let sampling_block =
            NonZeroU32::new(sampling_block).ok_or(LocalCostmapConfigError::ZeroSamplingBlock)?;
        if sampling_block.get() != 1 {
            return Err(LocalCostmapConfigError::UnsafeSamplingBlock {
                sampling_block: sampling_block.get(),
            });
        }
        if !footprint_radius_m.is_finite() || footprint_radius_m <= 0.0 {
            return Err(LocalCostmapConfigError::InvalidFootprintRadius {
                radius_m: footprint_radius_m,
            });
        }
        if !clearance_m.is_finite() || clearance_m < 0.0 {
            return Err(LocalCostmapConfigError::InvalidClearance { clearance_m });
        }
        let inflation_radius_m = footprint_radius_m + clearance_m;
        if !inflation_radius_m.is_finite() {
            return Err(LocalCostmapConfigError::InflationRadiusNotFinite {
                radius_m: footprint_radius_m,
                clearance_m,
            });
        }
        let maximum_age_ns = max_observation_age.as_nanos();
        if maximum_age_ns == 0 {
            return Err(LocalCostmapConfigError::ZeroMaximumObservationAge);
        }
        let max_observation_age_ns = u64::try_from(maximum_age_ns).map_err(|_| {
            LocalCostmapConfigError::MaximumObservationAgeNotRepresentable {
                nanoseconds: maximum_age_ns,
            }
        })?;

        let lower = geometry.lower_bound_m();
        let upper = geometry.upper_bound_m();
        for axis in 0..2 {
            if lower[axis] > -inflation_radius_m || upper[axis] < inflation_radius_m {
                return Err(LocalCostmapConfigError::InflatedFootprintOutsideGrid {
                    axis,
                    lower_bound_m: lower[axis],
                    upper_bound_m: upper[axis],
                    required_radius_m: inflation_radius_m,
                });
            }
        }

        Ok(Self {
            geometry,
            camera,
            tracking_to_base,
            obstacle_height_range,
            depth_range,
            sampling_block,
            footprint_radius_m,
            clearance_m,
            inflation_radius_m,
            max_observation_age_ns,
        })
    }

    pub fn geometry(&self) -> OccupancyGridGeometry {
        self.geometry
    }

    pub fn camera(&self) -> DepthCameraModel {
        self.camera
    }

    pub fn tracking_to_base(&self) -> TrackingCameraToBase {
        self.tracking_to_base
    }

    pub fn obstacle_height_range(&self) -> HeightRangeMeters {
        self.obstacle_height_range
    }

    pub fn depth_range(&self) -> DepthRangeMeters {
        self.depth_range
    }

    pub fn sampling_block(&self) -> u32 {
        self.sampling_block.get()
    }

    pub fn footprint_radius_m(&self) -> f64 {
        self.footprint_radius_m
    }

    pub fn clearance_m(&self) -> f64 {
        self.clearance_m
    }

    pub fn inflation_radius_m(&self) -> f64 {
        self.inflation_radius_m
    }

    pub fn max_observation_age_ns(&self) -> u64 {
        self.max_observation_age_ns
    }
}

/// One depth frame plus the process-clock and planar-pose provenance used to
/// build its local grid.
#[derive(Clone, Debug)]
pub struct LocalDepthObservation {
    source: DepthObservation,
    local_costmap_to_odom: LocalCostmapToOdom,
}

impl LocalDepthObservation {
    pub fn new(source: DepthObservation, local_costmap_to_odom: LocalCostmapToOdom) -> Self {
        Self {
            source,
            local_costmap_to_odom,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DepthFrameKey {
    sensor_timestamp: DeviceTimestamp,
    frame_id: FrameId,
}

impl DepthFrameKey {
    pub fn sensor_timestamp(self) -> DeviceTimestamp {
        self.sensor_timestamp
    }

    pub fn frame_id(self) -> FrameId {
        self.frame_id
    }
}

/// Exact source and transform attached to the current grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalCostmapProvenance {
    session_id: DeviceSessionId,
    frame: DepthFrameKey,
    host_arrival: HostMonotonicTimestamp,
    local_costmap_to_odom: LocalCostmapToOdom,
}

impl LocalCostmapProvenance {
    pub fn session_id(self) -> DeviceSessionId {
        self.session_id
    }

    pub fn frame(self) -> DepthFrameKey {
        self.frame
    }

    pub fn host_arrival(self) -> HostMonotonicTimestamp {
        self.host_arrival
    }

    pub fn local_costmap_to_odom(self) -> LocalCostmapToOdom {
        self.local_costmap_to_odom
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalCostmapUpdateOutcome {
    Accepted {
        frame: DepthFrameKey,
        sampled_pixels: usize,
        usable_depth_samples: usize,
        obstacle_endpoints: usize,
    },
    IgnoredDuplicate {
        frame: DepthFrameKey,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LocalCostmapError {
    SessionMismatch {
        expected: DeviceSessionId,
        actual: DeviceSessionId,
    },
    HostArrivalRegression {
        previous: HostMonotonicTimestamp,
        current: HostMonotonicTimestamp,
    },
    DeviceTimestampRegression {
        previous: DeviceTimestamp,
        current: DeviceTimestamp,
    },
    EqualDeviceTimestampConflict {
        timestamp: DeviceTimestamp,
        previous_frame_id: FrameId,
        current_frame_id: FrameId,
    },
    DuplicateFrameId {
        frame_id: FrameId,
        previous_timestamp: DeviceTimestamp,
        current_timestamp: DeviceTimestamp,
    },
    FrameIdRegression {
        previous: FrameId,
        current: FrameId,
    },
    DepthDimensionsMismatch {
        expected: FrameDimensions,
        actual: FrameDimensions,
    },
    AllocationFailed {
        context: &'static str,
        requested: usize,
    },
    NonFiniteProjection {
        axis: usize,
        value: f64,
    },
    NonFiniteRayArithmetic {
        stage: &'static str,
        axis: usize,
        value: f64,
    },
    RayTraversalInvariant {
        start_column: usize,
        start_row: usize,
        end_column: usize,
        end_row: usize,
    },
    InflationInvariant,
}

impl std::fmt::Display for LocalCostmapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SessionMismatch { expected, actual } => write!(
                f,
                "local-costmap depth session mismatch: expected {}, got {}",
                expected.as_u64(),
                actual.as_u64()
            ),
            Self::HostArrivalRegression { previous, current } => write!(
                f,
                "local-costmap host arrival regressed from {} ns to {} ns",
                previous.as_nanos(),
                current.as_nanos()
            ),
            Self::DeviceTimestampRegression { previous, current } => write!(
                f,
                "local-costmap depth timestamp regressed from {} ns to {} ns",
                previous.as_nanos(),
                current.as_nanos()
            ),
            Self::EqualDeviceTimestampConflict {
                timestamp,
                previous_frame_id,
                current_frame_id,
            } => write!(
                f,
                "local-costmap device timestamp {} ns was reused by distinct depth frames {} and {}",
                timestamp.as_nanos(),
                previous_frame_id.as_u64(),
                current_frame_id.as_u64()
            ),
            Self::DuplicateFrameId {
                frame_id,
                previous_timestamp,
                current_timestamp,
            } => write!(
                f,
                "local-costmap depth frame ID {} was reused with timestamps {} ns and {} ns",
                frame_id.as_u64(),
                previous_timestamp.as_nanos(),
                current_timestamp.as_nanos()
            ),
            Self::FrameIdRegression { previous, current } => write!(
                f,
                "local-costmap depth frame ID regressed from {} to {}",
                previous.as_u64(),
                current.as_u64()
            ),
            Self::DepthDimensionsMismatch { expected, actual } => write!(
                f,
                "local-costmap depth dimensions differ from the parsed camera model: expected {}x{}, got {}x{}",
                expected.width(),
                expected.height(),
                actual.width(),
                actual.height()
            ),
            Self::AllocationFailed { context, requested } => write!(
                f,
                "local-costmap allocation failed for {context} ({requested} elements)"
            ),
            Self::NonFiniteProjection { axis, value } => write!(
                f,
                "local-costmap depth projection produced nonfinite base coordinate {axis}: {value}"
            ),
            Self::NonFiniteRayArithmetic { stage, axis, value } => write!(
                f,
                "local-costmap {stage} produced nonfinite ray component {axis}: {value}"
            ),
            Self::RayTraversalInvariant {
                start_column,
                start_row,
                end_column,
                end_row,
            } => write!(
                f,
                "bounded local-costmap ray traversal did not reach ({end_column},{end_row}) from ({start_column},{start_row})"
            ),
            Self::InflationInvariant => write!(
                f,
                "local-costmap cell-square distance transform rejected its parsed grid contract"
            ),
        }
    }
}

impl std::error::Error for LocalCostmapError {}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalCostmapCell {
    Unknown = 0,
    /// Depth-visible free evidence in the capture-stamped local frame.
    Free = 1,
    Inflated = 2,
    Occupied = 3,
}

impl LocalCostmapCell {
    pub const fn class_id(self) -> u8 {
        self as u8
    }

    pub const fn is_traversable(self) -> bool {
        matches!(self, Self::Free)
    }

    fn from_class_id(value: u8) -> Self {
        match value {
            CLASS_UNKNOWN => Self::Unknown,
            CLASS_FREE => Self::Free,
            CLASS_INFLATED => Self::Inflated,
            CLASS_OCCUPIED => Self::Occupied,
            _ => unreachable!("local-costmap buffers contain only module class IDs"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalCostmapQuery {
    InBounds(LocalCostmapCell),
    OutOfBounds,
}

impl LocalCostmapQuery {
    pub const fn is_traversable(self) -> bool {
        matches!(self, Self::InBounds(LocalCostmapCell::Free))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalCostmapFreshness {
    NoObservation,
    Current { age_ns: u64 },
    Expired { age_ns: u64, maximum_age_ns: u64 },
}

impl LocalCostmapFreshness {
    pub const fn is_current(self) -> bool {
        matches!(self, Self::Current { .. })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocalCostmapClockRegression {
    pub observation_host_arrival: HostMonotonicTimestamp,
    pub requested_now: HostMonotonicTimestamp,
}

impl std::fmt::Display for LocalCostmapClockRegression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "local-costmap view time {} ns precedes observation arrival {} ns",
            self.requested_now.as_nanos(),
            self.observation_host_arrival.as_nanos()
        )
    }
}

impl std::error::Error for LocalCostmapClockRegression {}

/// Allocation-free read view. Class IDs are stable for direct segmentation
/// logging: `0=unknown`, `1=free`, `2=inflated`, `3=occupied`.
pub struct LocalCostmapView<'a> {
    class_ids: &'a [u8],
    geometry: OccupancyGridGeometry,
    freshness: LocalCostmapFreshness,
    provenance: Option<LocalCostmapProvenance>,
}

impl<'a> LocalCostmapView<'a> {
    pub fn class_ids(&self) -> &'a [u8] {
        self.class_ids
    }

    pub fn geometry(&self) -> OccupancyGridGeometry {
        self.geometry
    }

    pub fn width(&self) -> u32 {
        self.geometry.width()
    }

    pub fn height(&self) -> u32 {
        self.geometry.height()
    }

    pub fn resolution_m(&self) -> f64 {
        self.geometry.resolution_m()
    }

    pub fn lower_bound_m(&self) -> [f64; 2] {
        self.geometry.lower_bound_m()
    }

    pub fn freshness(&self) -> LocalCostmapFreshness {
        self.freshness
    }

    pub fn provenance(&self) -> Option<LocalCostmapProvenance> {
        self.provenance
    }

    pub fn cell(&self, column: u32, row: u32) -> Option<LocalCostmapCell> {
        if column >= self.width() || row >= self.height() {
            return None;
        }
        let index = row as usize * self.width() as usize + column as usize;
        self.class_ids
            .get(index)
            .copied()
            .map(LocalCostmapCell::from_class_id)
    }

    pub fn cell_at_local(&self, point: PlanarPoint<LocalCostmapFrame>) -> LocalCostmapQuery {
        self.geometry
            .point_index(point.as_array())
            .and_then(|index| self.class_ids.get(index).copied())
            .map(LocalCostmapCell::from_class_id)
            .map(LocalCostmapQuery::InBounds)
            .unwrap_or(LocalCostmapQuery::OutOfBounds)
    }

    pub fn is_traversable_at_local(&self, point: PlanarPoint<LocalCostmapFrame>) -> bool {
        self.cell_at_local(point).is_traversable()
    }

    pub fn cell_center_local(
        &self,
        column: u32,
        row: u32,
    ) -> Result<Option<PlanarPoint<LocalCostmapFrame>>, PlanarPointError> {
        if column >= self.width() || row >= self.height() {
            return Ok(None);
        }
        let lower = self.lower_bound_m();
        let resolution = self.resolution_m();
        PlanarPoint::try_new(
            lower[0] + (f64::from(column) + 0.5) * resolution,
            lower[1] + (f64::from(row) + 0.5) * resolution,
        )
        .map(Some)
    }

    pub fn cell_center_odom(
        &self,
        column: u32,
        row: u32,
    ) -> Result<Option<PlanarPoint<OdomFrame>>, LocalCostmapCoordinateError> {
        let Some(local) = self
            .cell_center_local(column, row)
            .map_err(LocalCostmapCoordinateError::Point)?
        else {
            return Ok(None);
        };
        let provenance = self
            .provenance
            .ok_or(LocalCostmapCoordinateError::NoObservation)?;
        provenance
            .local_costmap_to_odom
            .transform_point(local)
            .map(Some)
            .map_err(LocalCostmapCoordinateError::Transform)
    }

    /// Query an odom-frame point by applying the exact inverse capture pose.
    pub fn cell_at_odom(
        &self,
        point: PlanarPoint<OdomFrame>,
    ) -> Result<LocalCostmapQuery, LocalCostmapCoordinateError> {
        let provenance = self
            .provenance
            .ok_or(LocalCostmapCoordinateError::NoObservation)?;
        let odom_to_local = provenance
            .local_costmap_to_odom
            .inverse()
            .map_err(LocalCostmapCoordinateError::Transform)?;
        let local = odom_to_local
            .transform_point(point)
            .map_err(LocalCostmapCoordinateError::Transform)?;
        Ok(self.cell_at_local(local))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LocalCostmapCoordinateError {
    NoObservation,
    Point(PlanarPointError),
    Transform(PlanarTransformError),
}

impl std::fmt::Display for LocalCostmapCoordinateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoObservation => write!(f, "local costmap has no depth-frame pose provenance"),
            Self::Point(source) => write!(f, "invalid local-costmap cell centre: {source}"),
            Self::Transform(source) => {
                write!(f, "failed to transform local-costmap cell centre: {source}")
            }
        }
    }
}

impl std::error::Error for LocalCostmapCoordinateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NoObservation => None,
            Self::Point(source) => Some(source),
            Self::Transform(source) => Some(source),
        }
    }
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

    /// Compose `self` after `before`.
    fn compose(self, before: Self) -> Self {
        let rotation = multiply_matrix(self.rotation, before.rotation);
        let before_translation = multiply_vector(self.rotation, before.translation);
        Self {
            rotation,
            translation: [
                before_translation[0] + self.translation[0],
                before_translation[1] + self.translation[1],
                before_translation[2] + self.translation[2],
            ],
        }
    }

    fn transform_point(self, point: [f64; 3]) -> [f64; 3] {
        let rotated = multiply_vector(self.rotation, point);
        [
            rotated[0] + self.translation[0],
            rotated[1] + self.translation[1],
            rotated[2] + self.translation[2],
        ]
    }
}

/// Reusable newest-frame local costmap. Updating and viewing perform no heap
/// allocation after successful construction.
pub struct LocalCostmap {
    config: LocalCostmapConfig,
    session_id: DeviceSessionId,
    depth_to_base: RigidTransform64,
    column_rays: Vec<f64>,
    row_rays: Vec<f64>,
    current: Vec<u8>,
    staging: Vec<u8>,
    all_unknown: Vec<u8>,
    inflation_sources: Vec<bool>,
    inflation_result: Vec<bool>,
    inflation: CellSquareInflation,
    provenance: Option<LocalCostmapProvenance>,
    last_observed_host_arrival: Option<HostMonotonicTimestamp>,
}

impl LocalCostmap {
    pub fn try_new(
        config: LocalCostmapConfig,
        session_id: DeviceSessionId,
    ) -> Result<Self, LocalCostmapError> {
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

        let cell_count = config.geometry.cell_count();
        let current = unknown_buffer(cell_count, "current cells")?;
        let staging = unknown_buffer(cell_count, "staging cells")?;
        let all_unknown = unknown_buffer(cell_count, "expired cells")?;
        let inflation_sources = bool_buffer(cell_count, "inflation sources")?;
        let inflation_result = bool_buffer(cell_count, "inflation result")?;
        let inflation = CellSquareInflation::try_new(
            config.geometry.width() as usize,
            config.geometry.height() as usize,
        )
        .map_err(map_inflation_construction_error)?;

        let depth_to_tracking =
            RigidTransform64::from_pose(config.camera.depth_to_tracking().pose());
        let tracking_to_base = RigidTransform64::from_pose(config.tracking_to_base.pose());
        let depth_to_base = tracking_to_base.compose(depth_to_tracking);

        Ok(Self {
            config,
            session_id,
            depth_to_base,
            column_rays,
            row_rays,
            current,
            staging,
            all_unknown,
            inflation_sources,
            inflation_result,
            inflation,
            provenance: None,
            last_observed_host_arrival: None,
        })
    }

    pub fn config(&self) -> &LocalCostmapConfig {
        &self.config
    }

    pub fn session_id(&self) -> DeviceSessionId {
        self.session_id
    }

    pub fn provenance(&self) -> Option<LocalCostmapProvenance> {
        self.provenance
    }

    /// Establish a hard device-clock boundary and fail closed until a frame
    /// from the new session is accepted.
    pub fn reset_session(&mut self, session_id: DeviceSessionId) {
        self.session_id = session_id;
        self.current.fill(CLASS_UNKNOWN);
        self.staging.fill(CLASS_UNKNOWN);
        self.inflation_sources.fill(false);
        self.inflation_result.fill(false);
        self.provenance = None;
        self.last_observed_host_arrival = None;
    }

    pub fn update(
        &mut self,
        observation: LocalDepthObservation,
    ) -> Result<LocalCostmapUpdateOutcome, LocalCostmapError> {
        let session_id = observation.source.session_id();
        let host_arrival = observation.source.host_arrival();
        let sensor_timestamp = observation.source.device_timestamp();
        let depth = observation.source.depth();
        if session_id != self.session_id {
            return Err(LocalCostmapError::SessionMismatch {
                expected: self.session_id,
                actual: session_id,
            });
        }
        if let Some(previous) = self.last_observed_host_arrival
            && host_arrival < previous
        {
            return Err(LocalCostmapError::HostArrivalRegression {
                previous,
                current: host_arrival,
            });
        }
        let frame = DepthFrameKey {
            sensor_timestamp,
            frame_id: observation.source.frame_id(),
        };

        if let Some(previous) = self.provenance {
            if frame == previous.frame {
                self.last_observed_host_arrival = Some(host_arrival);
                return Ok(LocalCostmapUpdateOutcome::IgnoredDuplicate { frame });
            }
            if sensor_timestamp < previous.frame.sensor_timestamp {
                return Err(LocalCostmapError::DeviceTimestampRegression {
                    previous: previous.frame.sensor_timestamp,
                    current: sensor_timestamp,
                });
            }
            if sensor_timestamp == previous.frame.sensor_timestamp {
                return Err(LocalCostmapError::EqualDeviceTimestampConflict {
                    timestamp: sensor_timestamp,
                    previous_frame_id: previous.frame.frame_id,
                    current_frame_id: frame.frame_id,
                });
            }
            if frame.frame_id < previous.frame.frame_id {
                return Err(LocalCostmapError::FrameIdRegression {
                    previous: previous.frame.frame_id,
                    current: frame.frame_id,
                });
            }
            if frame.frame_id == previous.frame.frame_id {
                return Err(LocalCostmapError::DuplicateFrameId {
                    frame_id: frame.frame_id,
                    previous_timestamp: previous.frame.sensor_timestamp,
                    current_timestamp: sensor_timestamp,
                });
            }
        }

        let expected = self.config.camera.dimensions();
        let actual = depth.dimensions();
        if actual != expected {
            return Err(LocalCostmapError::DepthDimensionsMismatch { expected, actual });
        }

        self.staging.fill(CLASS_UNKNOWN);
        let integration = self.integrate_depth(depth);
        let (sampled_pixels, usable_depth_samples, obstacle_endpoints) = match integration {
            Ok(value) => value,
            Err(error) => {
                self.staging.fill(CLASS_UNKNOWN);
                return Err(error);
            }
        };
        self.inflate_nonfree()?;
        std::mem::swap(&mut self.current, &mut self.staging);
        self.provenance = Some(LocalCostmapProvenance {
            session_id,
            frame,
            host_arrival,
            local_costmap_to_odom: observation.local_costmap_to_odom,
        });
        self.last_observed_host_arrival = Some(host_arrival);

        Ok(LocalCostmapUpdateOutcome::Accepted {
            frame,
            sampled_pixels,
            usable_depth_samples,
            obstacle_endpoints,
        })
    }

    /// Return a fail-closed view at a timestamp from the same host monotonic
    /// clock as [`LocalDepthObservation::new`]. Expired grids expose an
    /// all-unknown buffer while retaining source provenance for diagnostics.
    pub fn view_at(
        &self,
        now: HostMonotonicTimestamp,
    ) -> Result<LocalCostmapView<'_>, LocalCostmapClockRegression> {
        let Some(provenance) = self.provenance else {
            return Ok(LocalCostmapView {
                class_ids: &self.all_unknown,
                geometry: self.config.geometry,
                freshness: LocalCostmapFreshness::NoObservation,
                provenance: None,
            });
        };
        let age_ns = now
            .as_nanos()
            .checked_sub(provenance.host_arrival.as_nanos())
            .ok_or(LocalCostmapClockRegression {
                observation_host_arrival: provenance.host_arrival,
                requested_now: now,
            })?;
        let (class_ids, freshness) = if age_ns <= self.config.max_observation_age_ns {
            (
                self.current.as_slice(),
                LocalCostmapFreshness::Current { age_ns },
            )
        } else {
            (
                self.all_unknown.as_slice(),
                LocalCostmapFreshness::Expired {
                    age_ns,
                    maximum_age_ns: self.config.max_observation_age_ns,
                },
            )
        };
        Ok(LocalCostmapView {
            class_ids,
            geometry: self.config.geometry,
            freshness,
            provenance: Some(provenance),
        })
    }

    fn integrate_depth(
        &mut self,
        depth: &DepthImage,
    ) -> Result<(usize, usize, usize), LocalCostmapError> {
        let dimensions = depth.dimensions();
        let width = dimensions.width();
        let height = dimensions.height();
        let values = depth.depth_m();
        let width_usize = width as usize;
        let origin = self.depth_to_base.translation;
        validate_projected_point(origin)?;

        let sampled_pixels = dimensions.area();
        let mut usable_depth_samples = 0_usize;
        let mut obstacle_endpoints = 0_usize;
        for row in 0..height {
            let offset = row as usize * width_usize;
            for column in 0..width {
                let depth_m = values[offset + column as usize];
                if depth_m == 0.0
                    || f64::from(depth_m) < self.config.depth_range.minimum_m()
                    || f64::from(depth_m) > self.config.depth_range.maximum_m()
                {
                    continue;
                }
                usable_depth_samples = usable_depth_samples.saturating_add(1);
                let depth_m = f64::from(depth_m);
                let optical = [
                    self.column_rays[column as usize] * depth_m,
                    self.row_rays[row as usize] * depth_m,
                    depth_m,
                ];
                let endpoint = self.depth_to_base.transform_point(optical);
                validate_projected_point(endpoint)?;
                self.mark_free_ray(origin, endpoint)?;
                if endpoint[2] >= self.config.obstacle_height_range.minimum_m()
                    && endpoint[2] <= self.config.obstacle_height_range.maximum_m()
                    && let Some(index) =
                        self.config.geometry.point_index([endpoint[0], endpoint[1]])
                {
                    self.mark_occupied(index);
                    obstacle_endpoints = obstacle_endpoints.saturating_add(1);
                }
            }
        }

        Ok((sampled_pixels, usable_depth_samples, obstacle_endpoints))
    }

    fn mark_free(&mut self, index: usize) {
        if self.staging[index] == CLASS_UNKNOWN {
            self.staging[index] = CLASS_FREE;
        }
    }

    fn mark_occupied(&mut self, index: usize) {
        self.staging[index] = CLASS_OCCUPIED;
    }

    fn mark_free_ray(
        &mut self,
        origin: [f64; 3],
        endpoint: [f64; 3],
    ) -> Result<(), LocalCostmapError> {
        let Some(clipped) = clip_segment_to_obstacle_slab(
            self.config.geometry,
            self.config.obstacle_height_range,
            origin,
            endpoint,
        )?
        else {
            return Ok(());
        };
        let start = clipped.start;
        let end = clipped.end;
        let Some((mut column, mut row)) = closed_cell(self.config.geometry, start) else {
            return Ok(());
        };
        let Some((end_column, end_row)) = closed_cell(self.config.geometry, end) else {
            return Ok(());
        };
        let start_column = column;
        let start_row = row;
        let width = self.config.geometry.width() as usize;
        self.mark_free(row * width + column);
        if column == end_column && row == end_row {
            return Ok(());
        }

        let delta_x = end[0] - start[0];
        let delta_y = end[1] - start[1];
        let resolution = self.config.geometry.resolution_m();
        let lower = self.config.geometry.lower_bound_m();
        let (step_x, mut next_x, delta_t_x) = if delta_x > 0.0 {
            let boundary = lower[0] + (column + 1) as f64 * resolution;
            (
                1_isize,
                (boundary - start[0]) / delta_x,
                resolution / delta_x,
            )
        } else if delta_x < 0.0 {
            let boundary = lower[0] + column as f64 * resolution;
            (
                -1_isize,
                (boundary - start[0]) / delta_x,
                -resolution / delta_x,
            )
        } else {
            (0_isize, f64::INFINITY, f64::INFINITY)
        };
        let (step_y, mut next_y, delta_t_y) = if delta_y > 0.0 {
            let boundary = lower[1] + (row + 1) as f64 * resolution;
            (
                1_isize,
                (boundary - start[1]) / delta_y,
                resolution / delta_y,
            )
        } else if delta_y < 0.0 {
            let boundary = lower[1] + row as f64 * resolution;
            (
                -1_isize,
                (boundary - start[1]) / delta_y,
                -resolution / delta_y,
            )
        } else {
            (0_isize, f64::INFINITY, f64::INFINITY)
        };

        let maximum_steps =
            self.config.geometry.width() as usize + self.config.geometry.height() as usize + 2;
        for _ in 0..maximum_steps {
            if column == end_column {
                next_x = f64::INFINITY;
            }
            if row == end_row {
                next_y = f64::INFINITY;
            }
            if next_x < next_y {
                column = column.saturating_add_signed(step_x);
                next_x += delta_t_x;
            } else if next_y < next_x {
                row = row.saturating_add_signed(step_y);
                next_y += delta_t_y;
            } else {
                column = column.saturating_add_signed(step_x);
                row = row.saturating_add_signed(step_y);
                next_x += delta_t_x;
                next_y += delta_t_y;
            }
            if column >= self.config.geometry.width() as usize
                || row >= self.config.geometry.height() as usize
            {
                break;
            }
            self.mark_free(row * width + column);
            if column == end_column && row == end_row {
                return Ok(());
            }
        }

        Err(LocalCostmapError::RayTraversalInvariant {
            start_column,
            start_row,
            end_column,
            end_row,
        })
    }

    fn inflate_nonfree(&mut self) -> Result<(), LocalCostmapError> {
        for (source, &class_id) in self.inflation_sources.iter_mut().zip(&self.staging) {
            *source = class_id != CLASS_FREE;
        }
        self.inflation
            .inflate(
                &self.inflation_sources,
                &mut self.inflation_result,
                self.config.geometry.resolution_m(),
                self.config.inflation_radius_m,
                true,
            )
            .map_err(|_| LocalCostmapError::InflationInvariant)?;
        for (class_id, &blocked) in self.staging.iter_mut().zip(&self.inflation_result) {
            if blocked && *class_id == CLASS_FREE {
                *class_id = CLASS_INFLATED;
            }
        }
        Ok(())
    }
}

fn unknown_buffer(length: usize, context: &'static str) -> Result<Vec<u8>, LocalCostmapError> {
    let mut values = Vec::new();
    try_reserve(&mut values, length, context)?;
    values.resize(length, CLASS_UNKNOWN);
    Ok(values)
}

fn bool_buffer(length: usize, context: &'static str) -> Result<Vec<bool>, LocalCostmapError> {
    let mut values = Vec::new();
    try_reserve(&mut values, length, context)?;
    values.resize(length, false);
    Ok(values)
}

fn map_inflation_construction_error(error: CellInflationError) -> LocalCostmapError {
    match error {
        CellInflationError::InvalidInput => LocalCostmapError::InflationInvariant,
        CellInflationError::AllocationFailed { context, requested } => {
            LocalCostmapError::AllocationFailed { context, requested }
        }
    }
}

fn try_reserve<T>(
    values: &mut Vec<T>,
    additional: usize,
    context: &'static str,
) -> Result<(), LocalCostmapError> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| LocalCostmapError::AllocationFailed {
            context,
            requested: additional,
        })
}

fn validate_projected_point(point: [f64; 3]) -> Result<(), LocalCostmapError> {
    if let Some(axis) = point.iter().position(|value| !value.is_finite()) {
        return Err(LocalCostmapError::NonFiniteProjection {
            axis,
            value: point[axis],
        });
    }
    Ok(())
}

fn multiply_vector(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    matrix.map(|row| row[0].mul_add(vector[0], row[1].mul_add(vector[1], row[2] * vector[2])))
}

fn multiply_matrix(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut product = [[0.0_f64; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            product[row][column] = left[row][0].mul_add(
                right[0][column],
                left[row][1].mul_add(right[1][column], left[row][2] * right[2][column]),
            );
        }
    }
    product
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ClippedPlanarSegment {
    start: [f64; 2],
    end: [f64; 2],
}

fn clip_segment_to_obstacle_slab(
    geometry: OccupancyGridGeometry,
    height_range: HeightRangeMeters,
    origin: [f64; 3],
    endpoint: [f64; 3],
) -> Result<Option<ClippedPlanarSegment>, LocalCostmapError> {
    let direction = [
        endpoint[0] - origin[0],
        endpoint[1] - origin[1],
        endpoint[2] - origin[2],
    ];
    if let Some(axis) = direction.iter().position(|value| !value.is_finite()) {
        return Err(LocalCostmapError::NonFiniteRayArithmetic {
            stage: "ray-direction subtraction",
            axis,
            value: direction[axis],
        });
    }
    let lower = geometry.lower_bound_m();
    let upper = geometry.upper_bound_m();
    let mut interval = [0.0_f64, 1.0_f64];
    if !clip_axis(
        origin[2],
        direction[2],
        height_range.minimum_m(),
        height_range.maximum_m(),
        2,
        &mut interval,
    )? {
        return Ok(None);
    }
    for axis in 0..2 {
        if !clip_axis(
            origin[axis],
            direction[axis],
            lower[axis],
            upper[axis],
            axis,
            &mut interval,
        )? {
            return Ok(None);
        }
    }
    let mut start = [0.0_f64; 2];
    let mut end = [0.0_f64; 2];
    for axis in 0..2 {
        let start_value = direction[axis].mul_add(interval[0], origin[axis]);
        if !start_value.is_finite() {
            return Err(LocalCostmapError::NonFiniteRayArithmetic {
                stage: "clipped start-point interpolation",
                axis,
                value: start_value,
            });
        }
        let end_value = direction[axis].mul_add(interval[1], origin[axis]);
        if !end_value.is_finite() {
            return Err(LocalCostmapError::NonFiniteRayArithmetic {
                stage: "clipped end-point interpolation",
                axis,
                value: end_value,
            });
        }
        start[axis] = start_value.clamp(lower[axis], upper[axis]);
        end[axis] = end_value.clamp(lower[axis], upper[axis]);
    }
    Ok(Some(ClippedPlanarSegment { start, end }))
}

fn clip_axis(
    origin: f64,
    direction: f64,
    lower: f64,
    upper: f64,
    axis: usize,
    interval: &mut [f64; 2],
) -> Result<bool, LocalCostmapError> {
    if direction == 0.0 {
        return Ok(origin >= lower && origin <= upper);
    }
    let first = (lower - origin) / direction;
    let second = (upper - origin) / direction;
    for (stage, value) in [
        ("lower ray-slab intersection", first),
        ("upper ray-slab intersection", second),
    ] {
        if !value.is_finite() {
            return Err(LocalCostmapError::NonFiniteRayArithmetic { stage, axis, value });
        }
    }
    let entry = first.min(second);
    let exit = first.max(second);
    interval[0] = interval[0].max(entry);
    interval[1] = interval[1].min(exit);
    if let Some(component) = interval.iter().position(|value| !value.is_finite()) {
        return Err(LocalCostmapError::NonFiniteRayArithmetic {
            stage: "ray-slab interval update",
            axis: component,
            value: interval[component],
        });
    }
    Ok(interval[0] <= interval[1])
}

fn closed_cell(geometry: OccupancyGridGeometry, mut point: [f64; 2]) -> Option<(usize, usize)> {
    let lower = geometry.lower_bound_m();
    let upper = geometry.upper_bound_m();
    if point[0] < lower[0] || point[0] > upper[0] || point[1] < lower[1] || point[1] > upper[1] {
        return None;
    }
    for axis in 0..2 {
        if point[axis] == upper[axis] {
            point[axis] = next_down(point[axis]);
        }
    }
    let index = geometry.point_index(point)?;
    let width = geometry.width() as usize;
    Some((index % width, index / width))
}

fn next_down(value: f64) -> f64 {
    debug_assert!(value.is_finite());
    if value == 0.0 {
        return -f64::from_bits(1);
    }
    if value > 0.0 {
        f64::from_bits(value.to_bits() - 1)
    } else {
        f64::from_bits(value.to_bits() + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::occupancy::{DepthToTrackingCamera, OccupancyGridGeometryError};
    use crate::{PinholeIntrinsics, Timestamp};

    fn session(raw: u64) -> DeviceSessionId {
        DeviceSessionId::try_new(raw).expect("nonzero session")
    }

    fn dimensions(width: u32, height: u32) -> FrameDimensions {
        FrameDimensions::try_new(width, height).expect("test dimensions")
    }

    fn optical_to_base(camera_height_m: f32) -> TrackingCameraToBase {
        // Optical [right, down, forward] -> base [forward, left, up].
        TrackingCameraToBase::new(
            Pose::try_from_rt(
                [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
                [0.0, 0.0, camera_height_m],
            )
            .expect("proper optical-to-base rotation"),
        )
    }

    fn config_with(
        image_width: u32,
        image_height: u32,
        sampling_block: u32,
        footprint_radius_m: f64,
        clearance_m: f64,
        max_age_ns: u64,
    ) -> LocalCostmapConfig {
        let geometry =
            OccupancyGridGeometry::try_new(0.25, [-1.0, -2.0], 20, 16, 320).expect("test grid");
        let camera = DepthCameraModel::new(
            PinholeIntrinsics::try_new(
                4.0,
                4.0,
                (image_width.saturating_sub(1) as f32) * 0.5,
                (image_height.saturating_sub(1) as f32) * 0.5,
            )
            .expect("test intrinsics"),
            dimensions(image_width, image_height),
            DepthToTrackingCamera::identity(),
        );
        LocalCostmapConfig::try_new(
            geometry,
            camera,
            optical_to_base(0.5),
            HeightRangeMeters::try_new(0.1, 1.5).expect("height range"),
            DepthRangeMeters::try_new(0.1, 8.0).expect("depth range"),
            sampling_block,
            footprint_radius_m,
            clearance_m,
            Duration::from_nanos(max_age_ns),
        )
        .expect("test costmap config")
    }

    fn depth(
        frame: u64,
        timestamp_ns: i64,
        width: u32,
        height: u32,
        values: Vec<f32>,
    ) -> DepthImage {
        DepthImage::new(
            FrameId::new(frame),
            Timestamp::from_nanos(timestamp_ns),
            width,
            height,
            values,
        )
        .expect("test depth")
    }

    fn depth_source(
        session_id: DeviceSessionId,
        image: &DepthImage,
        host_ns: u64,
    ) -> DepthObservation {
        DepthObservation::parse(
            session_id,
            HostMonotonicTimestamp::from_nanos(host_ns),
            image.clone(),
        )
        .expect("valid typed depth observation")
    }

    fn observation(image: &DepthImage, host_ns: u64) -> LocalDepthObservation {
        LocalDepthObservation::new(
            depth_source(session(1), image, host_ns),
            LocalCostmapToOdom::try_new(0.0, 0.0, 0.0).expect("identity capture pose"),
        )
    }

    fn point(x_m: f64, y_m: f64) -> PlanarPoint<LocalCostmapFrame> {
        PlanarPoint::try_new(x_m, y_m).expect("finite base point")
    }

    #[test]
    fn config_rejects_invalid_si_values_and_unusable_bounds() {
        let valid = config_with(1, 1, 1, 0.1, 0.0, 1);
        for radius in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                LocalCostmapConfig::try_new(
                    valid.geometry,
                    valid.camera,
                    valid.tracking_to_base,
                    valid.obstacle_height_range,
                    valid.depth_range,
                    1,
                    radius,
                    0.0,
                    Duration::from_nanos(1),
                ),
                Err(LocalCostmapConfigError::InvalidFootprintRadius { .. })
            ));
        }
        for clearance in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                LocalCostmapConfig::try_new(
                    valid.geometry,
                    valid.camera,
                    valid.tracking_to_base,
                    valid.obstacle_height_range,
                    valid.depth_range,
                    1,
                    0.1,
                    clearance,
                    Duration::from_nanos(1),
                ),
                Err(LocalCostmapConfigError::InvalidClearance { .. })
            ));
        }
        assert!(matches!(
            LocalCostmapConfig::try_new(
                valid.geometry,
                valid.camera,
                valid.tracking_to_base,
                valid.obstacle_height_range,
                valid.depth_range,
                0,
                0.1,
                0.0,
                Duration::from_nanos(1),
            ),
            Err(LocalCostmapConfigError::ZeroSamplingBlock)
        ));
        assert!(matches!(
            LocalCostmapConfig::try_new(
                valid.geometry,
                valid.camera,
                valid.tracking_to_base,
                valid.obstacle_height_range,
                valid.depth_range,
                2,
                0.1,
                0.0,
                Duration::from_nanos(1),
            ),
            Err(LocalCostmapConfigError::UnsafeSamplingBlock { sampling_block: 2 })
        ));
        assert!(matches!(
            LocalCostmapConfig::try_new(
                valid.geometry,
                valid.camera,
                valid.tracking_to_base,
                valid.obstacle_height_range,
                valid.depth_range,
                1,
                0.1,
                0.0,
                Duration::ZERO,
            ),
            Err(LocalCostmapConfigError::ZeroMaximumObservationAge)
        ));
        assert!(matches!(
            LocalCostmapConfig::try_new(
                valid.geometry,
                valid.camera,
                valid.tracking_to_base,
                valid.obstacle_height_range,
                valid.depth_range,
                1,
                0.1,
                0.0,
                Duration::new(u64::MAX, 999_999_999),
            ),
            Err(LocalCostmapConfigError::MaximumObservationAgeNotRepresentable { .. })
        ));

        let too_narrow = OccupancyGridGeometry::try_new(0.1, [-0.1, -1.0], 20, 20, 400)
            .expect("narrow test grid");
        assert!(matches!(
            LocalCostmapConfig::try_new(
                too_narrow,
                valid.camera,
                valid.tracking_to_base,
                valid.obstacle_height_range,
                valid.depth_range,
                1,
                0.2,
                0.0,
                Duration::from_nanos(1),
            ),
            Err(LocalCostmapConfigError::InflatedFootprintOutsideGrid { axis: 0, .. })
        ));
        assert!(matches!(
            OccupancyGridGeometry::try_new(0.0, [0.0; 2], 1, 1, 1),
            Err(OccupancyGridGeometryError::InvalidResolution { .. })
        ));
    }

    #[test]
    fn optical_projection_uses_documented_axes_and_extrinsic_chain() {
        let config = config_with(3, 1, 1, 0.1, 0.0, 100);
        let mut map = LocalCostmap::try_new(config, session(1)).expect("costmap");
        // Left image pixel has negative optical x, which maps to positive base y.
        let image = depth(1, 10, 3, 1, vec![2.0, 2.0, 2.0]);
        map.update(observation(&image, 20)).expect("update");
        let view = map
            .view_at(HostMonotonicTimestamp::from_nanos(20))
            .expect("view");
        for location in [point(2.0, 0.5), point(2.0, 0.0), point(2.0, -0.5)] {
            assert_eq!(
                view.cell_at_local(location),
                LocalCostmapQuery::InBounds(LocalCostmapCell::Occupied)
            );
        }
    }

    #[test]
    fn depth_to_tracking_is_composed_before_tracking_to_base() {
        let mut config = config_with(1, 1, 1, 0.1, 0.0, 100);
        config.camera = DepthCameraModel::new(
            config.camera.intrinsics(),
            config.camera.dimensions(),
            DepthToTrackingCamera::new(
                Pose::try_from_rt(Pose::identity().rotation(), [0.5, 0.0, 0.0])
                    .expect("depth-to-tracking"),
            ),
        );
        let mut map = LocalCostmap::try_new(config, session(1)).expect("costmap");
        let image = depth(1, 10, 1, 1, vec![2.0]);
        map.update(observation(&image, 20)).expect("update");
        let view = map
            .view_at(HostMonotonicTimestamp::from_nanos(20))
            .expect("view");
        assert_eq!(
            view.cell_at_local(point(2.0, -0.5)),
            LocalCostmapQuery::InBounds(LocalCostmapCell::Occupied),
            "positive tracking x must become negative base y after composition"
        );
    }

    #[test]
    fn cell_centres_transform_through_capture_pose_property_samples() {
        let config = config_with(1, 1, 1, 0.1, 0.0, 100);
        for yaw in [
            -std::f64::consts::PI,
            -0.7,
            0.0,
            0.9,
            std::f64::consts::FRAC_PI_2,
        ] {
            let mut map = LocalCostmap::try_new(config.clone(), session(1)).expect("costmap");
            let image = depth(1, 10, 1, 1, vec![1.0]);
            let transform = LocalCostmapToOdom::try_new(10.0, -2.0, yaw).expect("capture pose");
            map.update(LocalDepthObservation::new(
                depth_source(session(1), &image, 20),
                transform,
            ))
            .expect("update");
            let view = map
                .view_at(HostMonotonicTimestamp::from_nanos(20))
                .expect("view");
            for (column, row) in [(0, 0), (4, 7), (19, 15)] {
                let base = view
                    .cell_center_local(column, row)
                    .expect("finite centre")
                    .expect("in bounds");
                let odom = view
                    .cell_center_odom(column, row)
                    .expect("finite transformed centre")
                    .expect("in bounds");
                let expected = transform.transform_point(base).expect("expected transform");
                assert!((odom.x_m() - expected.x_m()).abs() <= 16.0 * f64::EPSILON);
                assert!((odom.y_m() - expected.y_m()).abs() <= 16.0 * f64::EPSILON);
            }
        }
    }

    #[test]
    fn grid_bounds_are_lower_inclusive_upper_exclusive_and_fail_closed() {
        let config = config_with(1, 1, 1, 0.1, 0.0, 100);
        let map = LocalCostmap::try_new(config, session(1)).expect("costmap");
        let view = map
            .view_at(HostMonotonicTimestamp::from_nanos(0))
            .expect("empty view");
        let lower = view.geometry().lower_bound_m();
        let upper = view.geometry().upper_bound_m();
        assert_eq!(
            view.cell_at_local(point(lower[0], lower[1])),
            LocalCostmapQuery::InBounds(LocalCostmapCell::Unknown)
        );
        for outside in [
            point(upper[0], 0.0),
            point(0.0, upper[1]),
            point(lower[0] - f64::EPSILON, 0.0),
        ] {
            assert_eq!(view.cell_at_local(outside), LocalCostmapQuery::OutOfBounds);
            assert!(!view.is_traversable_at_local(outside));
        }
        assert!(!view.is_traversable_at_local(point(0.0, 0.0)));
    }

    #[test]
    fn invalid_or_missing_depth_never_reuses_stale_free_space() {
        assert!(
            DepthImage::new(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                1,
                1,
                vec![f32::NAN]
            )
            .is_err()
        );

        let config = config_with(1, 1, 1, 0.1, 0.0, 100);
        let mut map = LocalCostmap::try_new(config, session(1)).expect("costmap");
        let valid = depth(1, 10, 1, 1, vec![1.0]);
        map.update(observation(&valid, 20)).expect("valid update");
        let mismatch = depth(2, 11, 2, 1, vec![1.0, 1.0]);
        assert!(matches!(
            map.update(observation(&mismatch, 21)),
            Err(LocalCostmapError::DepthDimensionsMismatch { .. })
        ));
        assert_eq!(
            map.provenance().expect("old provenance").frame().frame_id(),
            FrameId::new(1),
            "rejected frames must not mutate provenance"
        );

        let missing = depth(3, 12, 1, 1, vec![0.0]);
        let outcome = map
            .update(observation(&missing, 22))
            .expect("missing update");
        assert!(matches!(
            outcome,
            LocalCostmapUpdateOutcome::Accepted {
                usable_depth_samples: 0,
                obstacle_endpoints: 0,
                ..
            }
        ));
        let view = map
            .view_at(HostMonotonicTimestamp::from_nanos(22))
            .expect("view");
        assert!(
            view.class_ids()
                .iter()
                .all(|class_id| *class_id == CLASS_UNKNOWN)
        );
    }

    #[test]
    fn newer_depth_inserts_and_clears_obstacles() {
        let config = config_with(1, 1, 1, 0.1, 0.0, 100);
        let mut map = LocalCostmap::try_new(config, session(1)).expect("costmap");
        let near = depth(1, 10, 1, 1, vec![1.0]);
        map.update(observation(&near, 20)).expect("near update");
        let near_point = point(1.0, 0.0);
        assert_eq!(
            map.view_at(HostMonotonicTimestamp::from_nanos(20))
                .expect("near view")
                .cell_at_local(near_point),
            LocalCostmapQuery::InBounds(LocalCostmapCell::Occupied)
        );

        let far = depth(2, 11, 1, 1, vec![3.0]);
        map.update(observation(&far, 21)).expect("far update");
        let view = map
            .view_at(HostMonotonicTimestamp::from_nanos(21))
            .expect("far view");
        assert_eq!(
            view.cell_at_local(near_point),
            LocalCostmapQuery::InBounds(LocalCostmapCell::Inflated),
            "a longer observation clears the old occupied endpoint, but its thin observed ray remains conservatively inflated against adjacent unknown space"
        );
        assert_eq!(
            view.cell_at_local(point(3.0, 0.0)),
            LocalCostmapQuery::InBounds(LocalCostmapCell::Occupied)
        );
    }

    #[test]
    fn footprint_and_clearance_inflate_obstacles_conservatively() {
        let config = config_with(1, 1, 1, 0.2, 0.1, 100);
        let mut map = LocalCostmap::try_new(config, session(1)).expect("costmap");
        map.staging.fill(CLASS_FREE);
        let obstacle = map
            .config
            .geometry
            .point_index([2.0, 0.0])
            .expect("obstacle cell");
        map.mark_occupied(obstacle);
        map.inflate_nonfree().expect("inflation");
        let width = map.config.geometry.width() as usize;
        assert_eq!(map.staging[obstacle], CLASS_OCCUPIED);
        assert_eq!(map.staging[obstacle + 1], CLASS_INFLATED);
        assert_eq!(map.staging[obstacle + width], CLASS_INFLATED);
        assert_eq!(map.staging[obstacle + 3 * width], CLASS_FREE);
    }

    #[test]
    fn rays_outside_obstacle_height_slab_do_not_create_free_evidence() {
        let mut config = config_with(1, 1, 1, 0.1, 0.0, 100);
        config.tracking_to_base = optical_to_base(2.0);
        let mut map = LocalCostmap::try_new(config, session(1)).expect("costmap");
        let overhead = depth(1, 10, 1, 1, vec![3.0]);
        let outcome = map
            .update(observation(&overhead, 20))
            .expect("overhead frame");
        assert!(matches!(
            outcome,
            LocalCostmapUpdateOutcome::Accepted {
                usable_depth_samples: 1,
                obstacle_endpoints: 0,
                ..
            }
        ));
        let view = map
            .view_at(HostMonotonicTimestamp::from_nanos(20))
            .expect("view");
        assert!(
            view.class_ids()
                .iter()
                .all(|class_id| *class_id == CLASS_UNKNOWN)
        );
    }

    #[test]
    fn slab_clipping_rejects_nonfinite_intermediate_arithmetic() {
        let geometry = config_with(1, 1, 1, 0.1, 0.0, 100).geometry;
        let height = HeightRangeMeters::try_new(0.1, 1.5).expect("height range");
        assert!(matches!(
            clip_segment_to_obstacle_slab(
                geometry,
                height,
                [f64::MAX, 0.0, 0.5],
                [-f64::MAX, 0.0, 0.5],
            ),
            Err(LocalCostmapError::NonFiniteRayArithmetic {
                stage: "ray-direction subtraction",
                axis: 0,
                ..
            })
        ));

        let clipped =
            clip_segment_to_obstacle_slab(geometry, height, [0.0, 0.0, 2.0], [3.0, 0.0, 0.0])
                .expect("finite clipping")
                .expect("ray crosses obstacle slab");
        assert!((clipped.start[0] - 0.75).abs() <= 8.0 * f64::EPSILON);
        assert!((clipped.end[0] - 2.85).abs() <= 32.0 * f64::EPSILON);
    }

    #[test]
    fn observations_expire_to_unknown_at_a_strict_age_boundary() {
        let config = config_with(1, 1, 1, 0.1, 0.0, 10);
        let mut map = LocalCostmap::try_new(config, session(1)).expect("costmap");
        let image = depth(1, 10, 1, 1, vec![1.0]);
        map.update(observation(&image, 100)).expect("update");
        let boundary = map
            .view_at(HostMonotonicTimestamp::from_nanos(110))
            .expect("boundary view");
        assert_eq!(
            boundary.freshness(),
            LocalCostmapFreshness::Current { age_ns: 10 }
        );
        assert!(boundary.class_ids().contains(&CLASS_OCCUPIED));

        let expired = map
            .view_at(HostMonotonicTimestamp::from_nanos(111))
            .expect("expired view");
        assert_eq!(
            expired.freshness(),
            LocalCostmapFreshness::Expired {
                age_ns: 11,
                maximum_age_ns: 10
            }
        );
        assert!(
            expired
                .class_ids()
                .iter()
                .all(|class_id| *class_id == CLASS_UNKNOWN)
        );
        assert!(expired.provenance().is_some());
    }

    #[test]
    fn newest_frame_ordering_and_host_regression_are_transactional() {
        let config = config_with(1, 1, 1, 0.1, 0.0, 100);
        let mut map = LocalCostmap::try_new(config, session(1)).expect("costmap");
        let newest = depth(5, 10, 1, 1, vec![1.0]);
        map.update(observation(&newest, 20)).expect("newest update");
        let before = map
            .view_at(HostMonotonicTimestamp::from_nanos(20))
            .expect("before")
            .class_ids()
            .to_vec();

        let stale = depth(100, 9, 1, 1, vec![2.0]);
        assert!(matches!(
            map.update(observation(&stale, 21)),
            Err(LocalCostmapError::DeviceTimestampRegression { .. })
        ));
        let duplicate = depth(5, 10, 1, 1, vec![2.0]);
        assert!(matches!(
            map.update(observation(&duplicate, 22)),
            Ok(LocalCostmapUpdateOutcome::IgnoredDuplicate { .. })
        ));
        let next = depth(6, 11, 1, 1, vec![2.0]);
        assert!(matches!(
            map.update(observation(&next, 19)),
            Err(LocalCostmapError::HostArrivalRegression { .. })
        ));
        assert_eq!(
            map.view_at(HostMonotonicTimestamp::from_nanos(20))
                .expect("after rejects")
                .class_ids(),
            before
        );

        let same_time_newer_id = depth(6, 10, 1, 1, vec![2.0]);
        assert!(matches!(
            map.update(observation(&same_time_newer_id, 23)),
            Err(LocalCostmapError::EqualDeviceTimestampConflict { .. })
        ));
        let same_id_newer_time = depth(5, 11, 1, 1, vec![2.0]);
        assert!(matches!(
            map.update(observation(&same_id_newer_time, 23)),
            Err(LocalCostmapError::DuplicateFrameId { .. })
        ));
        let frame_regression = depth(4, 11, 1, 1, vec![2.0]);
        assert!(matches!(
            map.update(observation(&frame_regression, 23)),
            Err(LocalCostmapError::FrameIdRegression { .. })
        ));
        let accepted = depth(6, 11, 1, 1, vec![2.0]);
        assert!(matches!(
            map.update(observation(&accepted, 23)),
            Ok(LocalCostmapUpdateOutcome::Accepted { .. })
        ));
        assert!(matches!(
            map.update(observation(&accepted, 25)),
            Ok(LocalCostmapUpdateOutcome::IgnoredDuplicate { .. })
        ));
        let host_after_duplicate = depth(7, 12, 1, 1, vec![2.0]);
        assert!(matches!(
            map.update(observation(&host_after_duplicate, 24)),
            Err(LocalCostmapError::HostArrivalRegression { .. })
        ));
    }

    #[test]
    fn deterministic_output_holds_across_repeated_mixed_depth_patterns() {
        let config = config_with(4, 3, 1, 0.15, 0.05, 1_000);
        let mut first = LocalCostmap::try_new(config.clone(), session(1)).expect("first map");
        let mut second = LocalCostmap::try_new(config, session(1)).expect("second map");
        for frame in 0..32_u64 {
            let values = (0..12_u64)
                .map(|index| {
                    if (frame + index) % 5 == 0 {
                        0.0
                    } else {
                        0.5 + ((frame * 7 + index * 3) % 20) as f32 * 0.1
                    }
                })
                .collect::<Vec<_>>();
            let image = depth(frame, frame as i64, 4, 3, values);
            let host = 100 + frame;
            let first_outcome = first
                .update(observation(&image, host))
                .expect("first update");
            let second_outcome = second
                .update(observation(&image, host))
                .expect("second update");
            assert_eq!(first_outcome, second_outcome);
            assert_eq!(
                first
                    .view_at(HostMonotonicTimestamp::from_nanos(host))
                    .expect("first view")
                    .class_ids(),
                second
                    .view_at(HostMonotonicTimestamp::from_nanos(host))
                    .expect("second view")
                    .class_ids(),
                "frame {frame}"
            );
        }
    }

    #[test]
    fn session_and_clock_boundaries_fail_closed() {
        let config = config_with(1, 1, 1, 0.1, 0.0, 100);
        let mut map = LocalCostmap::try_new(config, session(1)).expect("costmap");
        let image = depth(1, 10, 1, 1, vec![1.0]);
        assert!(matches!(
            map.update(LocalDepthObservation::new(
                depth_source(session(2), &image, 20),
                LocalCostmapToOdom::try_new(0.0, 0.0, 0.0).expect("identity capture pose"),
            )),
            Err(LocalCostmapError::SessionMismatch { .. })
        ));
        map.update(observation(&image, 20)).expect("valid update");
        assert!(map.view_at(HostMonotonicTimestamp::from_nanos(19)).is_err());

        map.reset_session(session(2));
        let view = map
            .view_at(HostMonotonicTimestamp::from_nanos(0))
            .expect("reset view");
        assert_eq!(view.freshness(), LocalCostmapFreshness::NoObservation);
        assert!(
            view.class_ids()
                .iter()
                .all(|class_id| *class_id == CLASS_UNKNOWN)
        );
    }
}
