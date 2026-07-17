use std::{num::NonZeroUsize, time::Duration};

use crate::{
    CameraPoint3, CovisibilitySnapshot, DepthImage, Detections, Frame, Keypoint, Pose, Raw,
    Timestamp, VizPacket, WorldToCamera,
    dense::occupancy::{OccupancyGridMetadata, OccupancyGridSnapshot, WorldToOccupancy},
    env::{EnvError, env_f32},
};

use std::collections::HashMap;

const TIMELINE_CAPTURE_NS: &str = "capture_ns";
const WORLD_PATH: &str = "world";
const OCCUPANCY_ROOT_PATH: &str = "world/map2d";
const OCCUPANCY_GRID_PATH: &str = "world/map2d/grid";
const OCCUPANCY_METADATA_PATH: &str = "world/map2d/metadata";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VizDecimation(NonZeroUsize);

impl VizDecimation {
    pub fn every_n(n: NonZeroUsize) -> Self {
        Self(n)
    }

    pub fn get(self) -> usize {
        self.0.get()
    }

    fn should_log(self, index: u64) -> bool {
        let n = self.0.get() as u64;
        index.is_multiple_of(n)
    }
}

impl Default for VizDecimation {
    fn default() -> Self {
        Self(NonZeroUsize::MIN)
    }
}

#[derive(Debug)]
pub enum VizDecimationError {
    Zero,
}

impl std::fmt::Display for VizDecimationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VizDecimationError::Zero => write!(f, "decimation must be > 0"),
        }
    }
}

impl std::error::Error for VizDecimationError {}

#[derive(Debug)]
pub enum VizConfigError {
    Environment(EnvError),
    InvalidTrackMaxDistance { value: f32 },
    InvalidTrackMinSimilarity { value: f32 },
}

impl std::fmt::Display for VizConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Environment(err) => write!(f, "invalid visualization environment: {err}"),
            Self::InvalidTrackMaxDistance { value } => write!(
                f,
                "KIKO_TRACK_MAX_DIST must be finite and nonnegative, got {value}"
            ),
            Self::InvalidTrackMinSimilarity { value } => {
                write!(f, "KIKO_TRACK_MIN_SIM must be in [-1, 1], got {value}")
            }
        }
    }
}

impl std::error::Error for VizConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Environment(err) => Some(err),
            Self::InvalidTrackMaxDistance { .. } | Self::InvalidTrackMinSimilarity { .. } => None,
        }
    }
}

impl From<EnvError> for VizConfigError {
    fn from(err: EnvError) -> Self {
        Self::Environment(err)
    }
}

impl TryFrom<usize> for VizDecimation {
    type Error = VizDecimationError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        NonZeroUsize::new(value)
            .map(VizDecimation)
            .ok_or(VizDecimationError::Zero)
    }
}

#[derive(Debug)]
pub enum VizLogError {
    Rerun(rerun::RecordingStreamError),
    TimestampUnrepresentable {
        timestamp_ns: i64,
        encoded_ns: i64,
    },
    OccupancyCoordinateUnrepresentable {
        component: &'static str,
        value_m: f64,
    },
    OccupancyRotationUnrepresentable {
        component: &'static str,
        value: f64,
    },
    Pose(crate::PoseError),
    Point(crate::Point3Error),
}

impl std::fmt::Display for VizLogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VizLogError::Rerun(err) => write!(f, "rerun logging error: {err}"),
            VizLogError::TimestampUnrepresentable {
                timestamp_ns,
                encoded_ns,
            } => write!(
                f,
                "capture timestamp {timestamp_ns}ns is not exactly representable on the Rerun duration timeline (the SDK would encode {encoded_ns}ns)"
            ),
            VizLogError::OccupancyCoordinateUnrepresentable { component, value_m } => write!(
                f,
                "occupancy {component} value {value_m}m is not representable by Rerun's finite f32 transform domain"
            ),
            VizLogError::OccupancyRotationUnrepresentable { component, value } => write!(
                f,
                "occupancy {component} dimensionless value {value} is not representable by Rerun's finite f32 transform domain"
            ),
            VizLogError::Pose(err) => write!(f, "visualization pose error: {err}"),
            VizLogError::Point(err) => write!(f, "visualization point error: {err}"),
        }
    }
}

impl std::error::Error for VizLogError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Rerun(err) => Some(err),
            Self::TimestampUnrepresentable { .. }
            | Self::OccupancyCoordinateUnrepresentable { .. }
            | Self::OccupancyRotationUnrepresentable { .. } => None,
            Self::Pose(err) => Some(err),
            Self::Point(err) => Some(err),
        }
    }
}

impl From<rerun::RecordingStreamError> for VizLogError {
    fn from(err: rerun::RecordingStreamError) -> Self {
        VizLogError::Rerun(err)
    }
}

impl From<crate::PoseError> for VizLogError {
    fn from(err: crate::PoseError) -> Self {
        Self::Pose(err)
    }
}

impl From<crate::Point3Error> for VizLogError {
    fn from(err: crate::Point3Error) -> Self {
        Self::Point(err)
    }
}

/// A device-clock nanosecond timestamp proven to survive Rerun's duration encoding exactly.
///
/// Rerun reserves `i64::MIN` for static data, while Kiko's timestamp domain includes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RerunCaptureTime(rerun::TimeCell);

impl TryFrom<Timestamp> for RerunCaptureTime {
    type Error = VizLogError;

    fn try_from(timestamp: Timestamp) -> Result<Self, Self::Error> {
        let timestamp_ns = timestamp.as_nanos();
        let cell = rerun::TimeCell::from_duration_nanos(timestamp_ns);
        let encoded_ns = cell.as_i64();
        if encoded_ns != timestamp_ns {
            return Err(VizLogError::TimestampUnrepresentable {
                timestamp_ns,
                encoded_ns,
            });
        }
        Ok(Self(cell))
    }
}

impl RerunCaptureTime {
    fn set_on(self, rec: &rerun::RecordingStream) {
        rec.set_time(TIMELINE_CAPTURE_NS, self.0);
    }
}

#[derive(Debug)]
pub enum VizFlushError {
    Rerun(rerun::sink::SinkFlushError),
}

impl std::fmt::Display for VizFlushError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rerun(err) => write!(f, "rerun flush error: {err}"),
        }
    }
}

impl std::error::Error for VizFlushError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Rerun(err) => Some(err),
        }
    }
}

impl From<rerun::sink::SinkFlushError> for VizFlushError {
    fn from(err: rerun::sink::SinkFlushError) -> Self {
        Self::Rerun(err)
    }
}

#[derive(Debug)]
pub struct RerunSink {
    rec: rerun::RecordingStream,
    decimation: VizDecimation,
    frame_index: u64,
    depth_index: u64,
    tracks: TrackState,
    trajectory: Vec<[f32; 3]>,
    logged_world: bool,
    logged_occupancy_context: bool,
}

/// Fully parsed visualization settings needed to construct a [`RerunSink`].
#[derive(Clone, Copy, Debug)]
pub struct RerunSinkConfig {
    track: TrackConfig,
}

impl RerunSinkConfig {
    /// Parses the visualization environment once without creating a Rerun recording.
    pub fn from_environment() -> Result<Self, VizConfigError> {
        TrackConfig::load().map(|track| Self { track })
    }
}

impl RerunSink {
    pub fn new(
        rec: rerun::RecordingStream,
        decimation: VizDecimation,
    ) -> Result<Self, VizConfigError> {
        let config = RerunSinkConfig::from_environment()?;
        Ok(Self::from_config(rec, decimation, config))
    }

    /// Constructs a sink infallibly from settings that were already parsed.
    pub fn from_config(
        rec: rerun::RecordingStream,
        decimation: VizDecimation,
        config: RerunSinkConfig,
    ) -> Self {
        Self {
            rec,
            decimation,
            frame_index: 0,
            depth_index: 0,
            tracks: TrackState::new(config.track),
            trajectory: Vec::new(),
            logged_world: false,
            logged_occupancy_context: false,
        }
    }

    pub fn log(&mut self, packet: &VizPacket<Raw>) -> Result<(), VizLogError> {
        self.log_with_points(packet, None)
    }

    /// Flushes this sink and consumes it so no later visualization data can be logged through it.
    ///
    /// Rerun guarantees that operations previously issued by the calling thread have reached the
    /// underlying sink when this returns successfully. Operations issued through recording-stream
    /// clones on other threads can still be in flight. For a disabled recording, the Rerun SDK
    /// defines flushing as a successful no-op.
    pub fn finish_with_timeout(self, timeout: Duration) -> Result<(), VizFlushError> {
        self.rec.flush_with_timeout(timeout).map_err(Into::into)
    }

    pub fn log_frames(&mut self, left: &Frame, right: &Frame) -> Result<(), VizLogError> {
        let index = self.frame_index;
        self.frame_index = self.frame_index.saturating_add(1);
        if !self.decimation.should_log(index) {
            return Ok(());
        }

        self.set_time(left.timestamp())?;

        let left_image = rerun::Image::from_color_model_and_bytes(
            left.data().to_vec(),
            [left.width(), left.height()],
            rerun::ColorModel::L,
            rerun::ChannelDatatype::U8,
        )
        .with_draw_order(0.0);
        self.rec.log("view/left", &left_image)?;

        let right_image = rerun::Image::from_color_model_and_bytes(
            right.data().to_vec(),
            [right.width(), right.height()],
            rerun::ColorModel::L,
            rerun::ChannelDatatype::U8,
        )
        .with_draw_order(0.0);
        self.rec.log("view/right", &right_image)?;

        Ok(())
    }

    pub fn log_depth(&mut self, depth: &DepthImage) -> Result<(), VizLogError> {
        let index = self.depth_index;
        self.depth_index = self.depth_index.saturating_add(1);
        if !self.decimation.should_log(index) {
            return Ok(());
        }

        self.set_time(depth.timestamp())?;
        let mut bytes = Vec::with_capacity(depth.depth_m().len().saturating_mul(4));
        for sample in depth.depth_m() {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        let depth_image = rerun::Image::from_color_model_and_bytes(
            bytes,
            [depth.width(), depth.height()],
            rerun::ColorModel::L,
            rerun::ChannelDatatype::F32,
        )
        .with_draw_order(0.0);
        self.rec.log("view/depth", &depth_image)?;
        Ok(())
    }

    /// Logs a complete metric 2D occupancy snapshot without applying image decimation.
    ///
    /// The snapshot's canonical row-major storage is preserved: columns increase along the
    /// occupancy `x` axis and rows increase along occupancy `y`. Temporal transforms register the
    /// occupancy frame under Kiko's visual world and map pixels into metric occupancy coordinates,
    /// including snapshots whose fixed grid origin changes.
    pub fn log_occupancy(
        &mut self,
        timestamp: Timestamp,
        snapshot: OccupancyGridSnapshot,
    ) -> Result<(), VizLogError> {
        let capture_time = RerunCaptureTime::try_from(timestamp)?;
        let (metadata, class_ids) = snapshot.into_parts();
        let world_placement = occupancy_to_world_placement(metadata.world_to_occupancy())?;
        let grid_placement = occupancy_grid_placement(metadata)?;
        let grid_outline = OccupancyGridOutline::from_placement(metadata, grid_placement);
        let image = occupancy_segmentation_image(metadata, class_ids);

        self.ensure_world_context()?;
        if !self.logged_occupancy_context {
            self.rec
                .log_static(OCCUPANCY_ROOT_PATH, &occupancy_annotation_context())?;
            self.logged_occupancy_context = true;
        }

        capture_time.set_on(&self.rec);
        self.rec.log(
            OCCUPANCY_ROOT_PATH,
            &world_placement.as_rerun_parent_from_child(),
        )?;
        // The lightweight metric outline makes the occupancy root an indicated
        // 2D entity. Rerun can therefore auto-root the dedicated map view here
        // and apply the child image's pixel-to-metre transform.
        self.rec
            .log(OCCUPANCY_ROOT_PATH, &grid_outline.as_rerun())?;
        self.rec.log(
            OCCUPANCY_GRID_PATH,
            &rerun::Transform3D::from_translation_scale(
                grid_placement.translation,
                grid_placement.scale,
            )
            .with_relation(rerun::components::TransformRelation::ParentFromChild),
        )?;
        self.rec.log(OCCUPANCY_GRID_PATH, &image)?;
        self.rec.log(
            OCCUPANCY_METADATA_PATH,
            &rerun::TextLog::new(occupancy_metadata_text(metadata)),
        )?;
        Ok(())
    }

    pub fn log_with_points(
        &mut self,
        packet: &VizPacket<Raw>,
        points: Option<&[CameraPoint3]>,
    ) -> Result<(), VizLogError> {
        let index = self.frame_index;
        self.frame_index = self.frame_index.saturating_add(1);
        if !self.decimation.should_log(index) {
            return Ok(());
        }

        let left = packet.left();
        let right = packet.right();

        self.set_time(left.timestamp())?;
        let track_ids = self.tracks.assign_tracks(packet.matches().source_a_arc());

        let left_image = rerun::Image::from_color_model_and_bytes(
            left.data().to_vec(),
            [left.width(), left.height()],
            rerun::ColorModel::L,
            rerun::ChannelDatatype::U8,
        )
        .with_draw_order(0.0);
        self.rec.log("view/left", &left_image)?;

        let right_image = rerun::Image::from_color_model_and_bytes(
            right.data().to_vec(),
            [right.width(), right.height()],
            rerun::ColorModel::L,
            rerun::ChannelDatatype::U8,
        )
        .with_draw_order(0.0);
        self.rec.log("view/right", &right_image)?;

        let (stitched, width, height) = stitch_luma(left, right);
        let matches_image = rerun::Image::from_color_model_and_bytes(
            stitched,
            [width, height],
            rerun::ColorModel::L,
            rerun::ChannelDatatype::U8,
        )
        .with_draw_order(0.0);
        self.rec.log("view/matches", &matches_image)?;

        log_matches(&self.rec, packet, left.width() as f32, &track_ids)?;

        if let Some(points) = points
            && !points.is_empty()
        {
            let positions: Vec<[f32; 3]> = points
                .iter()
                .copied()
                .map(|point| point.validate().map(|point| point.to_array()))
                .collect::<Result<_, _>>()?;
            let cloud = rerun::Points3D::new(positions);
            self.rec.log("world/camera/points", &cloud)?;
        }

        Ok(())
    }

    pub fn log_pose(
        &mut self,
        timestamp: Timestamp,
        pose: &WorldToCamera,
    ) -> Result<(), VizLogError> {
        self.set_time(timestamp)?;
        self.ensure_world_context()?;

        let camera_pose = pose.try_inverse()?;
        let position = camera_pose.translation();
        let rotation = camera_pose.rotation();
        let quat = quat_from_rotation(rotation);

        let transform = rerun::Transform3D::update_fields()
            .with_translation(position)
            .with_quaternion(rerun::Quaternion::from_xyzw(quat));
        self.rec.log("world/camera", &transform)?;

        self.trajectory.push(position);
        let strips = vec![self.trajectory.clone()];
        self.rec
            .log("world/trajectory", &rerun::LineStrips3D::new(strips))?;

        Ok(())
    }

    pub fn log_covisibility_graph(
        &mut self,
        timestamp: Timestamp,
        snapshot: &CovisibilitySnapshot,
    ) -> Result<(), VizLogError> {
        self.set_time(timestamp)?;

        let mut positions: HashMap<crate::map::KeyframeId, [f32; 3]> = HashMap::new();
        for node in &snapshot.nodes {
            let pos = pose_position(node.pose.into_legacy_pose())?;
            positions.insert(node.id, pos);
        }

        if !positions.is_empty() {
            let points: Vec<[f32; 3]> = positions.values().copied().collect();
            self.rec
                .log("world/covisibility/nodes", &rerun::Points3D::new(points))?;
        }

        if !snapshot.edges.is_empty() {
            let mut strips: Vec<Vec<[f32; 3]>> = Vec::new();
            for edge in &snapshot.edges {
                let (Some(a), Some(b)) = (positions.get(&edge.a), positions.get(&edge.b)) else {
                    continue;
                };
                strips.push(vec![*a, *b]);
            }
            if !strips.is_empty() {
                self.rec.log(
                    "world/covisibility/edges",
                    &rerun::LineStrips3D::new(strips),
                )?;
            }
        }

        Ok(())
    }

    pub(crate) fn recording(&self) -> &rerun::RecordingStream {
        &self.rec
    }

    pub(crate) fn set_time(&self, timestamp: Timestamp) -> Result<(), VizLogError> {
        RerunCaptureTime::try_from(timestamp)?.set_on(&self.rec);
        Ok(())
    }

    fn ensure_world_context(&mut self) -> Result<(), VizLogError> {
        if !self.logged_world {
            let coords = rerun::archetypes::ViewCoordinates::RDF();
            self.rec.log_static(WORLD_PATH, &coords)?;
            self.logged_world = true;
        }
        Ok(())
    }
}

/// Rerun's parent-from-child transform from metric occupancy coordinates into Kiko world.
#[derive(Clone, Copy, Debug, PartialEq)]
struct OccupancyToWorldPlacement {
    translation_m: [f32; 3],
    /// Column-major: each entry is one occupancy basis vector expressed in world coordinates.
    rotation_columns: [[f32; 3]; 3],
}

impl OccupancyToWorldPlacement {
    fn as_rerun_parent_from_child(self) -> rerun::Transform3D {
        rerun::Transform3D::from_translation_mat3x3(self.translation_m, self.rotation_columns)
            .with_relation(rerun::components::TransformRelation::ParentFromChild)
    }
}

fn occupancy_to_world_placement(
    world_to_occupancy: WorldToOccupancy,
) -> Result<OccupancyToWorldPlacement, VizLogError> {
    const ROTATION_COMPONENTS: [[&str; 3]; 3] = [
        [
            "occupancy-to-world rotation column 0 row 0",
            "occupancy-to-world rotation column 0 row 1",
            "occupancy-to-world rotation column 0 row 2",
        ],
        [
            "occupancy-to-world rotation column 1 row 0",
            "occupancy-to-world rotation column 1 row 1",
            "occupancy-to-world rotation column 1 row 2",
        ],
        [
            "occupancy-to-world rotation column 2 row 0",
            "occupancy-to-world rotation column 2 row 1",
            "occupancy-to-world rotation column 2 row 2",
        ],
    ];
    const TRANSLATION_COMPONENTS: [&str; 3] = [
        "occupancy-to-world translation x",
        "occupancy-to-world translation y",
        "occupancy-to-world translation z",
    ];

    // If occupancy = R * world + t, then world = R^T * occupancy - R^T * t.
    // Rerun's nested arrays are columns, so row `axis` of R is column `axis` of R^T.
    let rotation = world_to_occupancy.rotation();
    let translation = world_to_occupancy.translation_m();
    let mut rotation_columns = [[0.0_f32; 3]; 3];
    for occupancy_axis in 0..3 {
        for world_axis in 0..3 {
            rotation_columns[occupancy_axis][world_axis] = rerun_rotation_scalar(
                ROTATION_COMPONENTS[occupancy_axis][world_axis],
                rotation[occupancy_axis][world_axis],
            )?;
        }
    }

    let mut translation_m = [0.0_f32; 3];
    for world_axis in 0..3 {
        let inverse_translation = (0..3).fold(0.0, |sum, occupancy_axis| {
            (-rotation[occupancy_axis][world_axis]).mul_add(translation[occupancy_axis], sum)
        });
        translation_m[world_axis] =
            rerun_metric_scalar(TRANSLATION_COMPONENTS[world_axis], inverse_translation)?;
    }

    Ok(OccupancyToWorldPlacement {
        translation_m,
        rotation_columns,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct OccupancyGridPlacement {
    translation: [f32; 3],
    scale: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct OccupancyGridOutline {
    points: [[f32; 2]; 5],
}

impl OccupancyGridOutline {
    fn from_placement(metadata: OccupancyGridMetadata, placement: OccupancyGridPlacement) -> Self {
        let lower = [placement.translation[0], placement.translation[1]];
        let upper = [
            lower[0] + placement.scale[0] * metadata.width() as f32,
            lower[1] + placement.scale[1] * metadata.height() as f32,
        ];
        Self {
            points: [
                lower,
                [upper[0], lower[1]],
                upper,
                [lower[0], upper[1]],
                lower,
            ],
        }
    }

    fn as_rerun(self) -> rerun::LineStrips2D {
        rerun::LineStrips2D::new([self.points])
            .with_radii([rerun::Radius::new_ui_points(1.0)])
            .with_colors([rerun::Color::from_rgb(48, 48, 48)])
    }
}

fn occupancy_grid_placement(
    metadata: OccupancyGridMetadata,
) -> Result<OccupancyGridPlacement, VizLogError> {
    let lower_bound_m = metadata.lower_bound_m();
    let resolution_m = metadata.resolution_m();
    let lower_u = rerun_metric_scalar("lower u bound", lower_bound_m[0])?;
    let lower_v = rerun_metric_scalar("lower v bound", lower_bound_m[1])?;
    let resolution = rerun_metric_scalar("resolution", resolution_m)?;

    for (axis, lower, lower_m, cells) in [
        ("u", lower_u, lower_bound_m[0], metadata.width()),
        ("v", lower_v, lower_bound_m[1], metadata.height()),
    ] {
        let mut previous = lower;
        for boundary_index in 1..=cells {
            let boundary = lower + resolution * boundary_index as f32;
            if !boundary.is_finite() || boundary <= previous {
                return Err(VizLogError::OccupancyCoordinateUnrepresentable {
                    component: match (axis, boundary_index) {
                        ("u", 1) => "u first cell boundary",
                        ("v", 1) => "v first cell boundary",
                        ("u", index) if index == cells => "u upper bound",
                        ("v", index) if index == cells => "v upper bound",
                        ("u", _) => "u interior cell boundary",
                        ("v", _) => "v interior cell boundary",
                        _ => unreachable!("the occupancy axes are u and v"),
                    },
                    value_m: resolution_m.mul_add(f64::from(boundary_index), lower_m),
                });
            }
            previous = boundary;
        }
    }

    Ok(OccupancyGridPlacement {
        translation: [lower_u, lower_v, 0.0],
        scale: [resolution, resolution, 1.0],
    })
}

fn rerun_metric_scalar(component: &'static str, value_m: f64) -> Result<f32, VizLogError> {
    let narrowed = value_m as f32;
    if !narrowed.is_finite() || (value_m != 0.0 && narrowed == 0.0) {
        return Err(VizLogError::OccupancyCoordinateUnrepresentable { component, value_m });
    }
    Ok(narrowed)
}

fn rerun_rotation_scalar(component: &'static str, value: f64) -> Result<f32, VizLogError> {
    let narrowed = value as f32;
    if !narrowed.is_finite() || (value != 0.0 && narrowed == 0.0) {
        return Err(VizLogError::OccupancyRotationUnrepresentable { component, value });
    }
    Ok(narrowed)
}

fn occupancy_segmentation_image(
    metadata: OccupancyGridMetadata,
    class_ids: Vec<u8>,
) -> rerun::SegmentationImage {
    rerun::SegmentationImage::new(
        class_ids,
        rerun::components::ImageFormat::segmentation(
            [metadata.width(), metadata.height()],
            rerun::ChannelDatatype::U8,
        ),
    )
    .with_opacity(1.0)
    .with_draw_order(0.0)
}

fn occupancy_annotation_context() -> rerun::AnnotationContext {
    rerun::AnnotationContext::new([
        (0_u16, "unknown", rerun::Rgba32::from_rgb(96, 96, 96)),
        (1_u16, "free", rerun::Rgba32::from_rgb(238, 238, 238)),
        (2_u16, "occupied", rerun::Rgba32::from_rgb(230, 57, 70)),
    ])
}

fn occupancy_metadata_text(metadata: OccupancyGridMetadata) -> String {
    let map_instance = metadata
        .map_instance_id()
        .map_or_else(|| "none".to_owned(), |id| id.as_u64().to_string());
    let lower = metadata.lower_bound_m();
    let height = metadata.height_range();
    let world_to_occupancy = metadata.world_to_occupancy();
    format!(
        "revision={} map_instance_id={} grid={}x{} resolution_m={} lower_xy_m=[{},{}] height_range_m=[{},{}] row_order=increasing_occupancy_y classes=0:unknown,1:free,2:occupied world_to_occupancy_rotation={:?} world_to_occupancy_translation_m={:?}",
        metadata.revision(),
        map_instance,
        metadata.width(),
        metadata.height(),
        metadata.resolution_m(),
        lower[0],
        lower[1],
        height.minimum_m(),
        height.maximum_m(),
        world_to_occupancy.rotation(),
        world_to_occupancy.translation_m(),
    )
}

fn pose_position(pose: Pose) -> Result<[f32; 3], crate::PoseError> {
    pose.try_inverse()
        .map(|camera_pose| camera_pose.translation())
}

#[derive(Debug, Clone, Copy)]
struct TrackConfig {
    max_distance_sq: f64,
    min_similarity: f64,
}

impl TrackConfig {
    const DEFAULT_MAX_DISTANCE_PX: f32 = 24.0;
    const DEFAULT_MIN_SIMILARITY: f32 = 0.8;

    fn load() -> Result<Self, VizConfigError> {
        Self::from_parsed(
            env_f32("KIKO_TRACK_MAX_DIST")?,
            env_f32("KIKO_TRACK_MIN_SIM")?,
        )
    }

    fn from_parsed(
        max_distance_px: Option<f32>,
        min_similarity: Option<f32>,
    ) -> Result<Self, VizConfigError> {
        Self::try_new(
            max_distance_px.unwrap_or(Self::DEFAULT_MAX_DISTANCE_PX),
            min_similarity.unwrap_or(Self::DEFAULT_MIN_SIMILARITY),
        )
    }

    fn try_new(max_distance_px: f32, min_similarity: f32) -> Result<Self, VizConfigError> {
        if !max_distance_px.is_finite() || max_distance_px < 0.0 {
            return Err(VizConfigError::InvalidTrackMaxDistance {
                value: max_distance_px,
            });
        }
        if !min_similarity.is_finite() || !(-1.0..=1.0).contains(&min_similarity) {
            return Err(VizConfigError::InvalidTrackMinSimilarity {
                value: min_similarity,
            });
        }

        let max_distance_px = f64::from(max_distance_px);
        Ok(Self {
            max_distance_sq: max_distance_px * max_distance_px,
            min_similarity: f64::from(min_similarity),
        })
    }
}

#[derive(Debug)]
struct TrackState {
    config: TrackConfig,
    prev_left: Option<std::sync::Arc<Detections>>,
    prev_track_ids: Vec<u64>,
    next_track_id: u64,
}

impl TrackState {
    fn new(config: TrackConfig) -> Self {
        Self {
            config,
            prev_left: None,
            prev_track_ids: Vec::new(),
            next_track_id: 0,
        }
    }

    fn assign_tracks(&mut self, left: std::sync::Arc<Detections>) -> Vec<u64> {
        let count = left.len();
        let mut track_ids = vec![0u64; count];

        let (prev_left, prev_ids) = match self.prev_left.as_ref() {
            Some(prev) if self.prev_track_ids.len() == prev.len() => {
                (Some(prev), &self.prev_track_ids)
            }
            _ => (None, &self.prev_track_ids),
        };

        if let Some(prev) = prev_left {
            let mut used_prev = vec![false; prev.len()];

            for (i, desc) in left.descriptors().iter().enumerate() {
                let kp = left.keypoints()[i];
                let mut best: Option<(usize, f64)> = None;

                for (j, prev_desc) in prev.descriptors().iter().enumerate() {
                    if used_prev[j] {
                        continue;
                    }
                    let prev_kp = prev.keypoints()[j];
                    if distance_sq(kp, prev_kp) > self.config.max_distance_sq {
                        continue;
                    }
                    let Some(similarity) =
                        cosine_similarity(desc.0.as_slice(), prev_desc.0.as_slice())
                    else {
                        continue;
                    };
                    if similarity < self.config.min_similarity {
                        continue;
                    }
                    if best.is_none_or(|(_, best_similarity)| similarity > best_similarity) {
                        best = Some((j, similarity));
                    }
                }

                if let Some((j, _)) = best {
                    track_ids[i] = prev_ids[j];
                    used_prev[j] = true;
                } else {
                    track_ids[i] = self.next_track_id;
                    self.next_track_id = self.next_track_id.saturating_add(1);
                }
            }
        } else {
            for id in &mut track_ids {
                *id = self.next_track_id;
                self.next_track_id = self.next_track_id.saturating_add(1);
            }
        }

        self.prev_left = Some(left);
        self.prev_track_ids = track_ids.clone();

        track_ids
    }
}

fn quat_from_rotation(r: [[f32; 3]; 3]) -> [f32; 4] {
    let trace = r[0][0] + r[1][1] + r[2][2];
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        let w = 0.25 * s;
        let x = (r[2][1] - r[1][2]) / s;
        let y = (r[0][2] - r[2][0]) / s;
        let z = (r[1][0] - r[0][1]) / s;
        [x, y, z, w]
    } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = (1.0 + r[0][0] - r[1][1] - r[2][2]).sqrt() * 2.0;
        let w = (r[2][1] - r[1][2]) / s;
        let x = 0.25 * s;
        let y = (r[0][1] + r[1][0]) / s;
        let z = (r[0][2] + r[2][0]) / s;
        [x, y, z, w]
    } else if r[1][1] > r[2][2] {
        let s = (1.0 + r[1][1] - r[0][0] - r[2][2]).sqrt() * 2.0;
        let w = (r[0][2] - r[2][0]) / s;
        let x = (r[0][1] + r[1][0]) / s;
        let y = 0.25 * s;
        let z = (r[1][2] + r[2][1]) / s;
        [x, y, z, w]
    } else {
        let s = (1.0 + r[2][2] - r[0][0] - r[1][1]).sqrt() * 2.0;
        let w = (r[1][0] - r[0][1]) / s;
        let x = (r[0][2] + r[2][0]) / s;
        let y = (r[1][2] + r[2][1]) / s;
        let z = 0.25 * s;
        [x, y, z, w]
    }
}

fn log_matches(
    rec: &rerun::RecordingStream,
    packet: &VizPacket<Raw>,
    x_offset: f32,
    track_ids: &[u64],
) -> Result<(), rerun::RecordingStreamError> {
    let matches = packet.matches();
    if matches.is_empty() {
        return Ok(());
    }

    let keypoints_left = matches.source_a().keypoints();
    let keypoints_right = matches.source_b().keypoints();
    let mut strips = Vec::with_capacity(matches.len());

    for &(idx_left, idx_right) in matches.indices() {
        if let (Some(kp_left), Some(kp_right)) =
            (keypoints_left.get(idx_left), keypoints_right.get(idx_right))
        {
            strips.push(vec![
                [kp_left.x, kp_left.y],
                [kp_right.x + x_offset, kp_right.y],
            ]);
        }
    }

    if strips.is_empty() {
        return Ok(());
    }

    let palette = [
        rerun::Color::from_rgb(230, 57, 70),
        rerun::Color::from_rgb(241, 250, 238),
        rerun::Color::from_rgb(168, 218, 220),
        rerun::Color::from_rgb(69, 123, 157),
        rerun::Color::from_rgb(29, 53, 87),
        rerun::Color::from_rgb(255, 183, 3),
        rerun::Color::from_rgb(251, 86, 7),
        rerun::Color::from_rgb(131, 56, 236),
    ];
    let colors: Vec<rerun::Color> = matches
        .indices()
        .iter()
        .map(|(idx_left, _)| {
            let track_id = track_ids.get(*idx_left).copied().unwrap_or(0);
            palette[(track_id as usize) % palette.len()]
        })
        .collect();

    rec.log(
        "view/matches",
        &rerun::LineStrips2D::new(strips)
            .with_colors(colors)
            .with_radii([rerun::Radius::new_ui_points(1.5)])
            .with_draw_order(10.0),
    )?;

    Ok(())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let (dot, norm_a_sq, norm_b_sq) = a.iter().zip(b).fold(
        (0.0_f64, 0.0_f64, 0.0_f64),
        |(dot, norm_a_sq, norm_b_sq), (&a, &b)| {
            let a = f64::from(a);
            let b = f64::from(b);
            (dot + a * b, norm_a_sq + a * a, norm_b_sq + b * b)
        },
    );
    if norm_a_sq == 0.0 || norm_b_sq == 0.0 {
        return None;
    }
    Some((dot / (norm_a_sq.sqrt() * norm_b_sq.sqrt())).clamp(-1.0, 1.0))
}

fn distance_sq(a: Keypoint, b: Keypoint) -> f64 {
    let dx = f64::from(a.x) - f64::from(b.x);
    let dy = f64::from(a.y) - f64::from(b.y);
    dx * dx + dy * dy
}

fn stitch_luma(left: &Frame, right: &Frame) -> (Vec<u8>, u32, u32) {
    let left_width = left.width();
    let left_height = left.height();
    let right_width = right.width();
    let right_height = right.height();

    let out_width = left_width + right_width;
    let out_height = left_height.max(right_height);

    let out_width_usize = out_width as usize;
    let out_height_usize = out_height as usize;
    let left_width_usize = left_width as usize;
    let left_height_usize = left_height as usize;
    let right_width_usize = right_width as usize;
    let right_height_usize = right_height as usize;

    let mut out = vec![0u8; out_width_usize * out_height_usize];

    let left_data = left.data();
    let right_data = right.data();

    for y in 0..out_height_usize {
        let out_row = y * out_width_usize;
        if y < left_height_usize {
            let left_row = y * left_width_usize;
            out[out_row..out_row + left_width_usize]
                .copy_from_slice(&left_data[left_row..left_row + left_width_usize]);
        }
        if y < right_height_usize {
            let right_row = y * right_width_usize;
            let out_right = out_row + left_width_usize;
            out[out_right..out_right + right_width_usize]
                .copy_from_slice(&right_data[right_row..right_row + right_width_usize]);
        }
    }

    (out, out_width, out_height)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Descriptor, FrameDimensions, FrameId, PinholeIntrinsics, SensorId,
        dense::occupancy::{
            DepthCameraModel, DepthRangeMeters, DepthToTrackingCamera, HeightRangeMeters,
            OccupancyConfig, OccupancyEvidenceModel, OccupancyGridGeometry, OccupancyMapper,
            WorldToOccupancy,
        },
    };
    use rerun::Loggable as _;

    #[derive(Clone, Copy)]
    enum TestFlushResult {
        Failed(&'static str),
        Timeout,
    }

    struct TestLogSink {
        flush_result: TestFlushResult,
    }

    impl rerun::sink::LogSink for TestLogSink {
        fn send(&self, _msg: rerun::log::LogMsg) {}

        fn flush_blocking(&self, _timeout: Duration) -> Result<(), rerun::sink::SinkFlushError> {
            match self.flush_result {
                TestFlushResult::Failed(message) => {
                    Err(rerun::sink::SinkFlushError::failed(message))
                }
                TestFlushResult::Timeout => Err(rerun::sink::SinkFlushError::Timeout),
            }
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn rerun_sink_with_flush_result(flush_result: TestFlushResult) -> RerunSink {
        let (enabled, store_info, recording_info, batcher_config, batcher_hooks) =
            rerun::RecordingStreamBuilder::new("kiko_viz_flush_test")
                .enabled(true)
                .send_properties(false)
                .into_args();
        assert!(enabled);
        let rec = rerun::RecordingStream::new(
            store_info,
            recording_info,
            batcher_config,
            batcher_hooks,
            Box::new(TestLogSink { flush_result }),
        )
        .expect("test recording stream");

        RerunSink {
            rec,
            decimation: VizDecimation::default(),
            frame_index: 0,
            depth_index: 0,
            tracks: TrackState::new(
                TrackConfig::try_new(24.0, 0.8).expect("test track configuration"),
            ),
            trajectory: Vec::new(),
            logged_world: false,
            logged_occupancy_context: false,
        }
    }

    fn test_occupancy_snapshot() -> OccupancyGridSnapshot {
        let geometry = OccupancyGridGeometry::try_new(0.25, [-2.0, 1.5], 3, 2, 6)
            .expect("test occupancy geometry");
        test_occupancy_snapshot_with_geometry(geometry)
    }

    fn test_occupancy_snapshot_with_geometry(
        geometry: OccupancyGridGeometry,
    ) -> OccupancyGridSnapshot {
        let camera = DepthCameraModel::new(
            PinholeIntrinsics::try_new(1.0, 1.0, 0.0, 0.0).expect("test intrinsics"),
            FrameDimensions::try_new(1, 1).expect("test frame dimensions"),
            DepthToTrackingCamera::identity(),
        );
        let config = OccupancyConfig::try_new(
            geometry,
            WorldToOccupancy::level_optical_world(1.0).expect("test occupancy frame"),
            camera,
            HeightRangeMeters::try_new(-1.0, 2.0).expect("test height range"),
            DepthRangeMeters::try_new(0.1, 10.0).expect("test depth range"),
            1,
            OccupancyEvidenceModel::try_new(-1, 1, -1, 1).expect("test evidence model"),
            1,
        )
        .expect("test occupancy configuration");
        OccupancyMapper::try_new(config)
            .expect("test occupancy mapper")
            .snapshot()
            .expect("test occupancy snapshot")
    }

    #[test]
    fn rerun_capture_time_preserves_the_complete_supported_boundary_classes() {
        for timestamp_ns in [i64::MIN + 1, -1, 0, 1, i64::MAX] {
            let parsed = RerunCaptureTime::try_from(Timestamp::from_nanos(timestamp_ns))
                .expect("Rerun-supported duration timestamp");
            assert_eq!(parsed.0.as_i64(), timestamp_ns);
            assert_eq!(parsed.0.typ(), rerun::time::TimeType::DurationNs);
        }
    }

    #[test]
    fn rerun_capture_time_rejects_minimum_instead_of_aliasing_it() {
        let sdk_cell = rerun::TimeCell::from_duration_nanos(i64::MIN);
        assert_eq!(sdk_cell.as_i64(), i64::MIN + 1);

        let error = RerunCaptureTime::try_from(Timestamp::from_nanos(i64::MIN))
            .expect_err("Rerun reserves the minimum i64 time value");
        assert!(matches!(
            &error,
            VizLogError::TimestampUnrepresentable {
                timestamp_ns,
                encoded_ns,
            } if *timestamp_ns == i64::MIN && *encoded_ns == i64::MIN + 1
        ));
        assert!(std::error::Error::source(&error).is_none());
    }

    #[test]
    fn occupancy_grid_placement_maps_canonical_axes_without_flipping_rows() {
        let metadata = test_occupancy_snapshot().metadata();
        let placement = occupancy_grid_placement(metadata).expect("representable placement");

        assert_eq!(placement.translation, [-2.0, 1.5, 0.0]);
        assert_eq!(placement.scale, [0.25, 0.25, 1.0]);

        let last_cell_center = [
            placement.translation[0] + (metadata.width() as f32 - 0.5) * placement.scale[0],
            placement.translation[1] + (metadata.height() as f32 - 0.5) * placement.scale[1],
        ];
        assert_eq!(last_cell_center, [-1.375, 1.875]);
    }

    #[test]
    fn occupancy_outline_marks_the_metric_map_root_with_exact_bounds() {
        let metadata = test_occupancy_snapshot().metadata();
        let placement = occupancy_grid_placement(metadata).expect("representable placement");
        let outline = OccupancyGridOutline::from_placement(metadata, placement);

        assert_eq!(
            outline.points,
            [
                [-2.0, 1.5],
                [-1.25, 1.5],
                [-1.25, 2.0],
                [-2.0, 2.0],
                [-2.0, 1.5],
            ]
        );
        assert!(outline.as_rerun().strips.is_some());
    }

    #[test]
    fn occupancy_paths_are_nested_under_the_existing_world_spatial_graph() {
        assert_eq!(WORLD_PATH, "world");
        assert_eq!(OCCUPANCY_ROOT_PATH, "world/map2d");
        assert_eq!(OCCUPANCY_GRID_PATH, "world/map2d/grid");
        assert_eq!(OCCUPANCY_METADATA_PATH, "world/map2d/metadata");
        assert_eq!(OCCUPANCY_ROOT_PATH.strip_prefix(WORLD_PATH), Some("/map2d"));
        assert_eq!(
            OCCUPANCY_GRID_PATH.strip_prefix(OCCUPANCY_ROOT_PATH),
            Some("/grid")
        );
    }

    #[test]
    fn occupancy_to_world_is_the_exact_inverse_frame_transform_before_f32_narrowing() {
        let world_to_occupancy =
            WorldToOccupancy::level_optical_world(1.25).expect("test occupancy frame");
        let placement =
            occupancy_to_world_placement(world_to_occupancy).expect("representable inverse");

        assert_eq!(placement.translation_m, [0.0, 1.25, 0.0]);
        assert_eq!(
            placement.rotation_columns,
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
        );

        for world_m in [[0.0, 0.0, 0.0], [2.0, -0.25, 4.0], [-3.0, 1.5, 0.5]] {
            let occupancy_m = world_to_occupancy
                .try_transform_world_point(world_m)
                .expect("finite world point");
            let recovered_world_m = std::array::from_fn(|world_axis| {
                (0..3).fold(
                    f64::from(placement.translation_m[world_axis]),
                    |sum, axis| {
                        f64::from(placement.rotation_columns[axis][world_axis])
                            .mul_add(occupancy_m[axis], sum)
                    },
                )
            });
            assert_eq!(recovered_world_m, world_m);
        }
    }

    #[test]
    fn occupancy_to_world_rerun_transform_explicitly_uses_parent_from_child() {
        let placement = occupancy_to_world_placement(
            WorldToOccupancy::level_optical_world(1.0).expect("test occupancy frame"),
        )
        .expect("representable inverse");
        let transform = placement.as_rerun_parent_from_child();
        let relations = rerun::components::TransformRelation::from_arrow(
            transform
                .relation
                .as_ref()
                .expect("explicit transform relation")
                .array
                .as_ref(),
        )
        .expect("decode transform relation");

        assert_eq!(
            relations,
            [rerun::components::TransformRelation::ParentFromChild]
        );
    }

    #[test]
    fn occupancy_to_world_rejects_unrepresentable_rerun_translation() {
        let world_to_occupancy = WorldToOccupancy::try_new(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [f64::MAX, 0.0, 0.0],
        )
        .expect("the mapper accepts finite f64 metric translations");

        let error = occupancy_to_world_placement(world_to_occupancy)
            .expect_err("Rerun cannot represent the inverse translation as f32");
        assert!(matches!(
            error,
            VizLogError::OccupancyCoordinateUnrepresentable {
                component: "occupancy-to-world translation x",
                value_m,
            } if value_m == -f64::MAX
        ));
    }

    #[test]
    fn occupancy_to_world_reports_dimensionless_rotation_narrowing_separately() {
        let tiny = f64::MIN_POSITIVE;
        let world_to_occupancy = WorldToOccupancy::try_new(
            [[1.0, -tiny, 0.0], [tiny, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0; 3],
        )
        .expect("the transform remains orthonormal within the mapper's f64 tolerance");

        let error = occupancy_to_world_placement(world_to_occupancy)
            .expect_err("a nonzero f64 rotation coefficient must not silently collapse to zero");
        assert!(matches!(
            error,
            VizLogError::OccupancyRotationUnrepresentable {
                component: "occupancy-to-world rotation column 0 row 1",
                value,
            } if value == -tiny
        ));
    }

    #[test]
    fn occupancy_grid_placement_rejects_cells_collapsed_by_rerun_f32_coordinates() {
        let geometry = OccupancyGridGeometry::try_new(1.0, [1.0e9, 0.0], 400, 2, 800)
            .expect("geometry remains distinct in the mapper's f64 domain");
        let metadata = test_occupancy_snapshot_with_geometry(geometry).metadata();

        let error = occupancy_grid_placement(metadata)
            .expect_err("Rerun f32 placement must not collapse adjacent cells");
        assert!(matches!(
            error,
            VizLogError::OccupancyCoordinateUnrepresentable {
                component: "u first cell boundary",
                ..
            }
        ));
    }

    #[test]
    fn occupancy_grid_placement_rejects_an_interior_f32_boundary_collapse() {
        let geometry = OccupancyGridGeometry::try_new(
            5.937_181_302_151_373e-11,
            [0.000_976_562_5, 0.0],
            3,
            2,
            6,
        )
        .expect("geometry remains distinct in the mapper's f64 domain");
        let metadata = test_occupancy_snapshot_with_geometry(geometry).metadata();

        let error = occupancy_grid_placement(metadata)
            .expect_err("every adjacent Rerun f32 cell boundary must remain distinct");
        assert!(matches!(
            error,
            VizLogError::OccupancyCoordinateUnrepresentable {
                component: "u interior cell boundary",
                ..
            }
        ));
    }

    #[test]
    fn occupancy_metadata_is_exact_and_self_describing() {
        let metadata = test_occupancy_snapshot().metadata();
        let text = occupancy_metadata_text(metadata);

        assert!(text.contains("revision=0 map_instance_id=none"));
        assert!(text.contains("grid=3x2 resolution_m=0.25 lower_xy_m=[-2,1.5]"));
        assert!(text.contains("height_range_m=[-1,2]"));
        assert!(text.contains("row_order=increasing_occupancy_y"));
        assert!(text.contains("classes=0:unknown,1:free,2:occupied"));
        assert!(text.contains("world_to_occupancy_rotation="));
        assert!(text.contains("world_to_occupancy_translation_m="));
    }

    #[test]
    fn occupancy_segmentation_preserves_owned_row_major_class_ids_and_u8_shape() {
        let metadata = test_occupancy_snapshot().metadata();
        let expected = vec![0_u8, 1, 2, 2, 1, 0];
        let image = occupancy_segmentation_image(metadata, expected.clone());

        let bytes = rerun::datatypes::Blob::serialized_blob_as_slice(
            image.buffer.as_ref().expect("segmentation buffer"),
        )
        .expect("serialized segmentation bytes");
        assert_eq!(bytes, expected);

        let formats = rerun::components::ImageFormat::from_arrow(
            image
                .format
                .as_ref()
                .expect("segmentation format")
                .array
                .as_ref(),
        )
        .expect("decode segmentation format");
        assert_eq!(formats.len(), 1);
        assert_eq!(formats[0].0.width, 3);
        assert_eq!(formats[0].0.height, 2);
        assert_eq!(formats[0].datatype(), rerun::ChannelDatatype::U8);
        assert!(formats[0].0.color_model.is_none());
    }

    #[test]
    fn occupancy_annotation_context_names_every_class_including_unknown() {
        let annotation = occupancy_annotation_context();
        let contexts = rerun::components::AnnotationContext::from_arrow(
            annotation
                .context
                .as_ref()
                .expect("annotation context")
                .array
                .as_ref(),
        )
        .expect("decode annotation context");
        let classes = &contexts.first().expect("one annotation context").0;
        let ids_and_labels: Vec<_> = classes
            .iter()
            .map(|class| {
                (
                    class.class_id.0,
                    class
                        .class_description
                        .info
                        .label
                        .as_ref()
                        .expect("class label")
                        .0
                        .as_str(),
                )
            })
            .collect();
        assert_eq!(
            ids_and_labels,
            [(0, "unknown"), (1, "free"), (2, "occupied")]
        );
    }

    #[test]
    fn occupancy_logging_is_not_decimated_and_initializes_context_once() {
        let rec = rerun::RecordingStream::disabled();
        let config = RerunSinkConfig {
            track: TrackConfig::try_new(24.0, 0.8).expect("test track configuration"),
        };
        let mut sink = RerunSink::from_config(
            rec,
            VizDecimation::every_n(NonZeroUsize::new(100).expect("nonzero")),
            config,
        );

        sink.log_occupancy(Timestamp::from_nanos(1), test_occupancy_snapshot())
            .expect("first occupancy snapshot");
        assert!(sink.logged_world);
        assert!(sink.logged_occupancy_context);
        assert_eq!(sink.frame_index, 0);
        assert_eq!(sink.depth_index, 0);

        sink.log_occupancy(Timestamp::from_nanos(2), test_occupancy_snapshot())
            .expect("second occupancy snapshot");
        assert!(sink.logged_occupancy_context);
        assert_eq!(sink.frame_index, 0);
        assert_eq!(sink.depth_index, 0);
    }

    #[test]
    fn occupancy_logging_parses_timestamp_before_static_side_effects() {
        let rec = rerun::RecordingStream::disabled();
        let config = RerunSinkConfig {
            track: TrackConfig::try_new(24.0, 0.8).expect("test track configuration"),
        };
        let mut sink = RerunSink::from_config(rec, VizDecimation::default(), config);

        let error = sink
            .log_occupancy(Timestamp::from_nanos(i64::MIN), test_occupancy_snapshot())
            .expect_err("reserved timestamp must fail before logging");
        assert!(matches!(
            error,
            VizLogError::TimestampUnrepresentable { .. }
        ));
        assert!(!sink.logged_world);
        assert!(!sink.logged_occupancy_context);
    }

    #[test]
    fn finish_preserves_rerun_sink_failure_as_typed_source() {
        let error = rerun_sink_with_flush_result(TestFlushResult::Failed("test delivery failure"))
            .finish_with_timeout(Duration::from_secs(1))
            .expect_err("sink failure must be reported");

        assert!(matches!(
            &error,
            VizFlushError::Rerun(rerun::sink::SinkFlushError::Failed { message })
                if message == "test delivery failure"
        ));
        assert!(
            std::error::Error::source(&error)
                .and_then(|source| source.downcast_ref::<rerun::sink::SinkFlushError>())
                .is_some()
        );
    }

    #[test]
    fn finish_preserves_rerun_timeout() {
        let error = rerun_sink_with_flush_result(TestFlushResult::Timeout)
            .finish_with_timeout(Duration::from_secs(1))
            .expect_err("sink timeout must be reported");

        assert!(matches!(
            error,
            VizFlushError::Rerun(rerun::sink::SinkFlushError::Timeout)
        ));
    }

    #[test]
    fn finish_treats_a_disabled_recording_as_the_sdk_noop() {
        let mut sink = rerun_sink_with_flush_result(TestFlushResult::Timeout);
        sink.rec = rerun::RecordingStream::disabled();

        sink.finish_with_timeout(Duration::ZERO)
            .expect("Rerun defines a disabled recording flush as a no-op");
    }

    #[test]
    fn track_config_rejects_negative_max_distance() {
        let error = TrackConfig::try_new(-0.25, 0.8).expect_err("negative distance");
        assert!(matches!(
            error,
            VizConfigError::InvalidTrackMaxDistance { value } if value == -0.25
        ));
    }

    #[test]
    fn track_config_rejects_nonfinite_max_distance() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                TrackConfig::try_new(value, 0.8),
                Err(VizConfigError::InvalidTrackMaxDistance { value: actual })
                    if actual.to_bits() == value.to_bits()
            ));
        }
    }

    #[test]
    fn track_config_rejects_similarity_outside_cosine_range() {
        for value in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -1.000_001,
            1.000_001,
        ] {
            assert!(matches!(
                TrackConfig::try_new(24.0, value),
                Err(VizConfigError::InvalidTrackMinSimilarity { value: actual })
                    if actual.to_bits() == value.to_bits()
            ));
        }
    }

    #[test]
    fn track_config_accepts_zero_max_distance() {
        let config = TrackConfig::try_new(0.0, 0.8).expect("zero distance is exact matching");
        assert_eq!(config.max_distance_sq, 0.0);
    }

    #[test]
    fn large_finite_distance_is_squared_without_f32_overflow() {
        let config = TrackConfig::try_new(f32::MAX, 0.8).expect("finite distance");
        let max_distance = f64::from(f32::MAX);
        assert_eq!(config.max_distance_sq, max_distance * max_distance);
        assert!(config.max_distance_sq.is_finite());

        let separated = distance_sq(
            Keypoint {
                x: f32::MAX,
                y: f32::MAX,
            },
            Keypoint {
                x: -f32::MAX,
                y: -f32::MAX,
            },
        );
        assert!(separated.is_finite());
        assert!(separated > config.max_distance_sq);
    }

    #[test]
    fn pose_position_propagates_rotated_inverse_overflow() {
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let pose = Pose::from_rt(
            [[s, -s, 0.0], [s, s, 0.0], [0.0, 0.0, 1.0]],
            [f32::MAX, f32::MAX, 0.0],
        );
        assert!(matches!(
            pose_position(pose),
            Err(crate::PoseError::InverseTranslationNotRepresentable {
                axis: 0,
                value,
            }) if value < -f64::from(f32::MAX)
        ));
    }

    #[test]
    fn visualization_point_error_preserves_the_boundary_cause() {
        let point_error = CameraPoint3::new(0.0, f32::NAN, 1.0)
            .validate()
            .expect_err("non-finite point");
        let error = VizLogError::from(point_error);
        assert!(matches!(
            error,
            VizLogError::Point(crate::Point3Error::NonFinite {
                axis: 1,
                value,
            }) if value.is_nan()
        ));
    }

    #[test]
    fn track_config_uses_defaults_only_for_absent_values() {
        let config = TrackConfig::from_parsed(None, None).expect("default track configuration");
        assert_eq!(
            config.max_distance_sq,
            f64::from(TrackConfig::DEFAULT_MAX_DISTANCE_PX).powi(2)
        );
        assert_eq!(
            config.min_similarity,
            f64::from(TrackConfig::DEFAULT_MIN_SIMILARITY)
        );
    }

    #[test]
    fn cosine_similarity_is_stable_for_extreme_finite_descriptors() {
        let positive = [f32::MAX; crate::DESCRIPTOR_DIM];
        let negative = [-f32::MAX; crate::DESCRIPTOR_DIM];
        let zero = [0.0; crate::DESCRIPTOR_DIM];

        assert_eq!(cosine_similarity(&positive, &positive), Some(1.0));
        assert_eq!(cosine_similarity(&positive, &negative), Some(-1.0));
        assert_eq!(cosine_similarity(&positive, &zero), None);
    }

    #[test]
    fn minimum_similarity_threshold_is_inclusive() {
        fn detections(frame_id: u64) -> Detections {
            let mut descriptor = [0.0; crate::DESCRIPTOR_DIM];
            descriptor[0] = 1.0;
            Detections::new(
                SensorId::StereoLeft,
                FrameId::new(frame_id),
                8,
                8,
                vec![Keypoint { x: 1.0, y: 1.0 }],
                vec![1.0],
                vec![Descriptor(descriptor)],
            )
            .expect("valid test detection")
        }

        let mut tracks = TrackState::new(
            TrackConfig::try_new(0.0, 1.0).expect("exact-match track configuration"),
        );
        assert_eq!(
            tracks.assign_tracks(std::sync::Arc::new(detections(1))),
            vec![0]
        );
        assert_eq!(
            tracks.assign_tracks(std::sync::Arc::new(detections(2))),
            vec![0]
        );
    }

    #[test]
    fn environment_error_is_preserved_as_typed_source() {
        let source = EnvError::InvalidF32 {
            key: "KIKO_TRACK_MAX_DIST".to_owned(),
            value: "twenty-four".to_owned(),
            source: "twenty-four"
                .parse::<f32>()
                .expect_err("invalid test float"),
        };
        let error = VizConfigError::from(source);
        assert!(matches!(error, VizConfigError::Environment(_)));
        assert!(
            std::error::Error::source(&error)
                .and_then(|source| source.downcast_ref::<EnvError>())
                .is_some()
        );
    }
}
