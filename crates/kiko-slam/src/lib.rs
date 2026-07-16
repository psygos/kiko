use std::marker::PhantomData;
use std::num::NonZeroU32;
use std::sync::Arc;

pub use inference::{
    EigenPlaces, InferenceBackend, InferenceError, LightGlue, PlaceDescriptorExtractor, SuperPoint,
    WatchdogConfigError,
};
mod channel;
pub mod dataset;
pub mod dense;
mod depth;
mod diagnostics;
pub mod env;
mod geometry;
mod inference;
mod local_ba;
pub mod loop_closure;
pub mod map;
mod math;
#[cfg(feature = "record")]
mod oak;
mod observability;
mod pairing;
mod pipeline;
mod pnp;
pub mod pose_graph;
mod preprocess;
#[cfg(test)]
pub(crate) mod test_helpers;
mod tracker;
mod triangulation;
mod viz;
pub use channel::{
    ChannelCapacity, ChannelCapacityError, ChannelStats, ChannelStatsHandle, DropPolicy,
    DropReceiver, DropSender, SendOutcome, bounded_channel,
};
pub use dense::backend::{
    Mesh, TsdfBackend, TsdfBackendFactory, TsdfConfig, TsdfConfigError, TsdfError,
};
pub use dense::{
    DenseCommand, DenseCommandChannelError, DenseCommandQueueStats, DenseCommandQueueStatsHandle,
    DenseCommandReceiver, DenseCommandSendOutcome, DenseCommandSender, DenseConfig,
    DenseConfigError, DenseStats, ReconState, dense_command_channel,
};
pub use depth::{DepthImage, DepthImageError};
pub use diagnostics::{
    DiagnosticEvent, FrameDiagnostics, KeyframeRemovalReason, LoopClosureRejectReason,
    MappingSessionTransition, MappingSessionTransitionError,
};
pub use env::{EnvError, env_bool, env_f32, env_u32, env_u64, env_usize};
pub use geometry::{
    CameraFrame, CameraPoint3, CoordinateFrame, Point3, Point3Error, WorldFrame, WorldPoint3,
};
pub use local_ba::{
    BaResult, DegenerateReason, LmConfig, LmConfigError, LocalBaConfig, LocalBaConfigError,
    LocalBaError, LocalBundleAdjuster, MapObservation, ObservationSet, ObservationSetError,
};
pub use loop_closure::{
    DescriptorSource, GlobalDescriptor, GlobalDescriptorError, KeyframeDatabase,
    KeyframeDatabaseError, LoopApplyError, LoopClosureConfig, LoopClosureConfigError,
    LoopClosureConfigInput, LoopDetectError, LoopVerificationError, PlaceMatch,
    RelocalizationConfig, RelocalizationConfigError, RelocalizationConfigInput,
    RelocalizationMatch, VerifiedLoop, VerifiedRelocalization, aggregate_global_descriptor,
    try_match_descriptors_for_loop,
};
pub use map::{
    CovisibilityEdge, CovisibilityNode, CovisibilitySnapshot, MapGeneration, MapInstanceId,
    MapSnapshot,
};
pub use math::{Pose64, Pose64Error, PoseNarrowingError};
#[cfg(feature = "record")]
pub use oak::{oak_to_depth_image, oak_to_frame};
pub use pairing::{
    PairingConfigError, PairingInputError, PairingStats, PairingWindowNs, StereoPairer,
};
pub use pipeline::{
    InferencePipeline, KeypointLimit, KeypointLimitError, PipelineError, PipelineTimingError,
    PipelineTimings, PipelineWallBreakdown,
};
pub use pnp::{
    CameraToWorld, IntrinsicsError, Observation, ObservationError, PinholeIntrinsics, PnpError,
    PnpResult, Pose, PoseError, RansacConfig, RansacConfigError, Transform, TransformError,
    WorldToCamera, build_observations, solve_pnp, solve_pnp_ransac,
};
pub use tracker::{
    BackendConfig, BackendConfigError, BackendStats, ComponentHealth, CovisibilityRatio,
    DegradationLevel, DescriptorStats, GlobalDescriptorConfig, GlobalDescriptorConfigError,
    KeyframePolicy, KeyframePolicyError, LoopSubsystemConfig, ParallaxPx, PoseStatus,
    RedundancyPolicy, RedundancyPolicyError, SlamTracker, SystemHealth, TrackerConfig,
    TrackerError, TrackerInitError, TrackerOutput, TrackingHealth,
};
pub use triangulation::{
    Keyframe, KeyframeError, KeyframeLandmarkError, RectifiedStereo, RectifiedStereoConfig,
    RectifiedStereoError, StereoCameraSide, StereoToleranceKind, TriangulationConfig,
    TriangulationConfigError, TriangulationError, TriangulationResult, TriangulationStats,
    Triangulator,
};
pub use viz::{
    RerunSink, RerunSinkConfig, VizConfigError, VizDecimation, VizDecimationError, VizFlushError,
    VizLogError,
};

pub fn panic_payload_to_string(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(msg) = payload.downcast_ref::<&'static str>() {
        return (*msg).to_string();
    }
    if let Some(msg) = payload.downcast_ref::<String>() {
        return msg.clone();
    }
    "unknown panic payload".to_string()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SensorId {
    StereoLeft,
    StereoRight,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord, Hash)]
pub struct FrameId(u64);

impl FrameId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub struct Timestamp(i64);

impl Timestamp {
    pub fn from_nanos(ns: i64) -> Self {
        Self(ns)
    }

    pub fn as_nanos(&self) -> i64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameDimensions {
    width: NonZeroU32,
    height: NonZeroU32,
    area: usize,
}

impl FrameDimensions {
    pub fn try_new(width: u32, height: u32) -> Result<Self, FrameDimensionsError> {
        let Some(width_value) = NonZeroU32::new(width) else {
            return Err(FrameDimensionsError::Zero { width, height });
        };
        let Some(height_value) = NonZeroU32::new(height) else {
            return Err(FrameDimensionsError::Zero { width, height });
        };
        let area = usize::try_from(width)
            .ok()
            .and_then(|width| {
                usize::try_from(height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .ok_or(FrameDimensionsError::TooLarge { width, height })?;
        Ok(Self {
            width: width_value,
            height: height_value,
            area,
        })
    }

    pub(crate) fn new(width: u32, height: u32) -> Self {
        Self::try_new(width, height).expect("internally validated frame dimensions")
    }

    pub fn width(self) -> u32 {
        self.width.get()
    }

    pub fn height(self) -> u32 {
        self.height.get()
    }

    pub fn area(self) -> usize {
        self.area
    }
}

#[derive(Debug)]
pub enum FrameDimensionsError {
    Zero { width: u32, height: u32 },
    TooLarge { width: u32, height: u32 },
}

impl std::fmt::Display for FrameDimensionsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Zero { width, height } => {
                write!(f, "frame dimensions must be nonzero, got {width}x{height}")
            }
            Self::TooLarge { width, height } => {
                write!(
                    f,
                    "frame dimensions {width}x{height} exceed addressable memory"
                )
            }
        }
    }
}

impl std::error::Error for FrameDimensionsError {}

#[derive(Debug)]
pub enum FrameError {
    InvalidDimensions { source: FrameDimensionsError },
    DimensionMismatch { expected: usize, actual: usize },
}
impl std::fmt::Display for FrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameError::InvalidDimensions { source } => {
                write!(f, "invalid frame dimensions: {source}")
            }
            FrameError::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for FrameError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidDimensions { source } => Some(source),
            Self::DimensionMismatch { .. } => None,
        }
    }
}

impl From<FrameDimensionsError> for FrameError {
    fn from(source: FrameDimensionsError) -> Self {
        Self::InvalidDimensions { source }
    }
}

#[derive(Debug)]
pub enum PairError {
    DimensionMismatch {
        left: FrameDimensions,
        right: FrameDimensions,
    },
    TimestampDelta {
        delta_ns: u64,
        max_delta_ns: u64,
    },
    SensorMismatch {
        left: SensorId,
        right: SensorId,
    },
}

impl std::fmt::Display for PairError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PairError::DimensionMismatch { left, right } => {
                write!(
                    f,
                    "stereo dimension mismatch: left={}x{}, right={}x{}",
                    left.width(),
                    left.height(),
                    right.width(),
                    right.height()
                )
            }
            PairError::TimestampDelta {
                delta_ns,
                max_delta_ns,
            } => {
                write!(
                    f,
                    "stereo delta {delta_ns}ns exceeds window {max_delta_ns}ns"
                )
            }
            PairError::SensorMismatch { left, right } => {
                write!(f, "stereo sensor mismatch: left={left:?}, right={right:?}")
            }
        }
    }
}

impl std::error::Error for PairError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DownscaleFactor(NonZeroU32);

impl DownscaleFactor {
    pub fn new(value: NonZeroU32) -> Self {
        Self(value)
    }

    pub fn get(self) -> usize {
        self.0.get() as usize
    }

    pub fn get_u32(self) -> u32 {
        self.0.get()
    }

    pub fn identity() -> Self {
        Self(NonZeroU32::MIN)
    }
}

#[derive(Debug)]
pub enum DownscaleError {
    Zero,
    TooLarge {
        value: usize,
    },
    NonDivisible {
        width: u32,
        height: u32,
        factor: usize,
    },
}

impl std::fmt::Display for DownscaleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DownscaleError::Zero => write!(f, "downscale factor must be > 0"),
            DownscaleError::TooLarge { value } => {
                write!(f, "downscale factor {value} exceeds u32::MAX")
            }
            DownscaleError::NonDivisible {
                width,
                height,
                factor,
            } => write!(
                f,
                "downscale factor {factor} does not divide frame {width}x{height}"
            ),
        }
    }
}

impl std::error::Error for DownscaleError {}

impl TryFrom<usize> for DownscaleFactor {
    type Error = DownscaleError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        let value = u32::try_from(value).map_err(|_| DownscaleError::TooLarge { value })?;
        NonZeroU32::new(value)
            .map(DownscaleFactor)
            .ok_or(DownscaleError::Zero)
    }
}

#[derive(Clone, Debug)]
pub struct Frame {
    sensor_id: SensorId,
    frame_id: FrameId,
    timestamp: Timestamp,
    dimensions: FrameDimensions,
    data: Arc<[u8]>,
}

impl Frame {
    pub fn new(
        sensor_id: SensorId,
        frame_id: FrameId,
        timestamp: Timestamp,
        width: u32,
        height: u32,
        data: Vec<u8>,
    ) -> Result<Self, FrameError> {
        let dimensions = FrameDimensions::try_new(width, height)?;
        Self::from_dimensions(sensor_id, frame_id, timestamp, dimensions, data)
    }

    pub(crate) fn from_dimensions(
        sensor_id: SensorId,
        frame_id: FrameId,
        timestamp: Timestamp,
        dimensions: FrameDimensions,
        data: Vec<u8>,
    ) -> Result<Self, FrameError> {
        let size = dimensions.area();

        if data.len() != size {
            return Err(FrameError::DimensionMismatch {
                expected: size,
                actual: data.len(),
            });
        }

        Ok(Self {
            sensor_id,
            frame_id,
            timestamp,
            dimensions,
            data: Arc::from(data.into_boxed_slice()),
        })
    }

    pub fn width(&self) -> u32 {
        self.dimensions.width()
    }
    pub fn height(&self) -> u32 {
        self.dimensions.height()
    }

    pub fn dimensions(&self) -> FrameDimensions {
        self.dimensions
    }
    pub fn data(&self) -> &[u8] {
        self.data.as_ref()
    }

    pub fn sensor_id(&self) -> SensorId {
        self.sensor_id
    }

    pub fn frame_id(&self) -> FrameId {
        self.frame_id
    }

    pub fn timestamp(&self) -> Timestamp {
        self.timestamp
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
}

pub const DESCRIPTOR_DIM: usize = 256;
const SIGNED_DESCRIPTOR_SCALE: f32 = 127.0;
const COMPACT_DESCRIPTOR_ZERO: i16 = 128;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Descriptor(pub [f32; DESCRIPTOR_DIM]);

impl Descriptor {
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    /// Quantizes each finite descriptor component into a centered unsigned byte.
    ///
    /// Components are clamped to `[-1, 1]`, scaled symmetrically to
    /// `[-127, 127]`, and biased by 128. Consequently, zero is encoded as 128,
    /// while -1 and 1 are encoded as 1 and 255 respectively.
    pub fn quantize(&self) -> CompactDescriptor {
        let mut out = [0_u8; DESCRIPTOR_DIM];
        for (&value, encoded) in self.0.iter().zip(out.iter_mut()) {
            let signed = (value.clamp(-1.0, 1.0) * SIGNED_DESCRIPTOR_SCALE).round() as i16;
            *encoded = u8::try_from(signed + COMPACT_DESCRIPTOR_ZERO)
                .expect("clamped descriptor component must fit centered-u8 encoding");
        }
        CompactDescriptor(out)
    }
}

/// A signed local descriptor stored in centered-u8 form.
///
/// Each byte decodes to the signed component `code - 128`. Quantization emits
/// codes in `1..=255`; code 0 remains representable for compatibility with the
/// public byte-array representation and decodes to -128.
#[repr(transparent)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactDescriptor(pub [u8; DESCRIPTOR_DIM]);

impl CompactDescriptor {
    /// Computes cosine similarity after decoding the centered signed values.
    ///
    /// The fixed descriptor dimension and byte range allow all dot products and
    /// squared norms to be accumulated exactly in wide integers. An encoded zero
    /// vector (all components equal to 128) has no defined cosine; this API uses
    /// `0.0` as the finite, no-similarity convention when either input is zero.
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let (dot, norm_a, norm_b) = self.0.iter().zip(other.0.iter()).fold(
            (0_i64, 0_u64, 0_u64),
            |(dot, na, nb), (&a, &b)| {
                let a = i64::from(a) - i64::from(COMPACT_DESCRIPTOR_ZERO);
                let b = i64::from(b) - i64::from(COMPACT_DESCRIPTOR_ZERO);
                let a_magnitude = a.unsigned_abs();
                let b_magnitude = b.unsigned_abs();
                (
                    dot + a * b,
                    na + a_magnitude * a_magnitude,
                    nb + b_magnitude * b_magnitude,
                )
            },
        );
        if norm_a == 0 || norm_b == 0 {
            return 0.0;
        }

        let denominator = ((norm_a as f64) * (norm_b as f64)).sqrt();
        ((dot as f64) / denominator).clamp(-1.0, 1.0) as f32
    }
}

#[derive(Debug)]
pub enum DetectionError {
    ShapeMismatch {
        keypoints_len: usize,
        scores_len: usize,
        descriptors_len: usize,
    },
    ZeroDimensions {
        width: u32,
        height: u32,
    },
    NonFiniteKeypoint {
        index: usize,
        x: f32,
        y: f32,
    },
    KeypointOutOfBounds {
        index: usize,
        x: f32,
        y: f32,
        width: u32,
        height: u32,
    },
    NonFiniteScore {
        index: usize,
        score: f32,
    },
    NonFiniteDescriptor {
        detection_index: usize,
        component_index: usize,
        value: f32,
    },
}

impl std::fmt::Display for DetectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DetectionError::ShapeMismatch {
                keypoints_len,
                scores_len,
                descriptors_len,
            } => write!(
                f,
                "detections shape mismatch: keypoints={keypoints_len}, scores={scores_len}, descriptors={descriptors_len}"
            ),
            DetectionError::ZeroDimensions { width, height } => {
                write!(
                    f,
                    "detection dimensions must be nonzero, got {width}x{height}"
                )
            }
            DetectionError::NonFiniteKeypoint { index, x, y } => {
                write!(f, "detection keypoint {index} is nonfinite: ({x}, {y})")
            }
            DetectionError::KeypointOutOfBounds {
                index,
                x,
                y,
                width,
                height,
            } => write!(
                f,
                "detection keypoint {index} ({x}, {y}) is outside {width}x{height}"
            ),
            DetectionError::NonFiniteScore { index, score } => {
                write!(f, "detection score {index} is nonfinite: {score}")
            }
            DetectionError::NonFiniteDescriptor {
                detection_index,
                component_index,
                value,
            } => write!(
                f,
                "descriptor {detection_index} component {component_index} is nonfinite: {value}"
            ),
        }
    }
}

impl std::error::Error for DetectionError {}

#[derive(Debug, Clone)]
pub struct Detections {
    sensor_id: SensorId,
    frame_id: FrameId,
    width: u32,
    height: u32,
    keypoints: Vec<Keypoint>,
    scores: Vec<f32>,
    descriptors: Vec<Descriptor>,
}

impl Detections {
    pub fn sensor_id(&self) -> SensorId {
        self.sensor_id
    }

    pub fn frame_id(&self) -> FrameId {
        self.frame_id
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn keypoints(&self) -> &[Keypoint] {
        &self.keypoints
    }

    pub fn keypoints_flat(&self) -> &[f32] {
        bytemuck::cast_slice(self.keypoints.as_slice())
    }

    pub fn scores(&self) -> &[f32] {
        &self.scores
    }

    pub fn descriptors(&self) -> &[Descriptor] {
        &self.descriptors
    }

    pub fn descriptors_flat(&self) -> &[f32] {
        bytemuck::cast_slice(self.descriptors.as_slice())
    }

    pub fn len(&self) -> usize {
        self.descriptors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }

    pub fn new(
        sensor_id: SensorId,
        frame_id: FrameId,
        width: u32,
        height: u32,
        keypoints: Vec<Keypoint>,
        scores: Vec<f32>,
        descriptors: Vec<Descriptor>,
    ) -> Result<Self, DetectionError> {
        if keypoints.len() != descriptors.len() || descriptors.len() != scores.len() {
            return Err(DetectionError::ShapeMismatch {
                keypoints_len: keypoints.len(),
                scores_len: scores.len(),
                descriptors_len: descriptors.len(),
            });
        }
        if width == 0 || height == 0 {
            return Err(DetectionError::ZeroDimensions { width, height });
        }
        // Compare in f64 so u32 dimensions above f32's exact-integer range do
        // not round inward and incorrectly reject their last representable
        // pixel coordinate.
        let width_f = f64::from(width);
        let height_f = f64::from(height);
        for (index, point) in keypoints.iter().enumerate() {
            if !point.x.is_finite() || !point.y.is_finite() {
                return Err(DetectionError::NonFiniteKeypoint {
                    index,
                    x: point.x,
                    y: point.y,
                });
            }
            if point.x < 0.0
                || point.y < 0.0
                || f64::from(point.x) >= width_f
                || f64::from(point.y) >= height_f
            {
                return Err(DetectionError::KeypointOutOfBounds {
                    index,
                    x: point.x,
                    y: point.y,
                    width,
                    height,
                });
            }
        }
        for (index, &score) in scores.iter().enumerate() {
            if !score.is_finite() {
                return Err(DetectionError::NonFiniteScore { index, score });
            }
        }
        for (detection_index, descriptor) in descriptors.iter().enumerate() {
            if let Some((component_index, &value)) = descriptor
                .as_slice()
                .iter()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
            {
                return Err(DetectionError::NonFiniteDescriptor {
                    detection_index,
                    component_index,
                    value,
                });
            }
        }

        Ok(Self {
            sensor_id,
            frame_id,
            width,
            height,
            keypoints,
            scores,
            descriptors,
        })
    }

    pub fn top_k(self, max: usize) -> Self {
        if self.descriptors.len() <= max {
            return self;
        }

        let Detections {
            sensor_id,
            frame_id,
            width,
            height,
            keypoints,
            scores,
            descriptors,
        } = self;

        let mut order: Vec<usize> = (0..descriptors.len()).collect();
        let kth = max.saturating_sub(1);
        let cmp_desc = |&a: &usize, &b: &usize| scores[b].total_cmp(&scores[a]);
        order.select_nth_unstable_by(kth, cmp_desc);
        order.truncate(max);
        order.sort_unstable_by(cmp_desc);

        let mut new_keypoints = Vec::with_capacity(order.len());
        let mut new_scores = Vec::with_capacity(order.len());
        let mut new_descriptors = Vec::with_capacity(order.len());

        for &idx in &order {
            new_keypoints.push(keypoints[idx]);
            new_scores.push(scores[idx]);
            new_descriptors.push(descriptors[idx]);
        }

        Self {
            sensor_id,
            frame_id,
            width,
            height,
            keypoints: new_keypoints,
            scores: new_scores,
            descriptors: new_descriptors,
        }
    }
}

#[derive(Debug)]
pub struct StereoPair {
    left: Frame,
    right: Frame,
}

impl StereoPair {
    pub fn try_new(left: Frame, right: Frame, window: PairingWindowNs) -> Result<Self, PairError> {
        if left.sensor_id() != SensorId::StereoLeft || right.sensor_id() != SensorId::StereoRight {
            return Err(PairError::SensorMismatch {
                left: left.sensor_id(),
                right: right.sensor_id(),
            });
        }

        if left.width() != right.width() || left.height() != right.height() {
            return Err(PairError::DimensionMismatch {
                left: left.dimensions(),
                right: right.dimensions(),
            });
        }

        let delta = left
            .timestamp()
            .as_nanos()
            .abs_diff(right.timestamp().as_nanos());
        if delta > window.as_u64() {
            return Err(PairError::TimestampDelta {
                delta_ns: delta,
                max_delta_ns: window.as_u64(),
            });
        }

        Ok(Self { left, right })
    }

    /// Construct a pair without validation. Use when frames are known to be
    /// correctly paired (e.g. from a pre-validated dataset manifest).
    pub(crate) fn from_parts(left: Frame, right: Frame) -> Self {
        Self { left, right }
    }

    pub fn left(&self) -> &Frame {
        &self.left
    }

    pub fn right(&self) -> &Frame {
        &self.right
    }

    pub fn into_parts(self) -> (Frame, Frame) {
        (self.left, self.right)
    }

    pub fn timestamp_delta_ns(&self) -> u64 {
        self.left
            .timestamp()
            .as_nanos()
            .abs_diff(self.right.timestamp().as_nanos())
    }
}

// Typestates for Matches
#[derive(Debug)]
pub struct Raw;
#[derive(Debug)]
pub struct Verified;

#[derive(Debug)]
pub enum MatchError {
    Mismatch {
        score_len: usize,
        indices_len: usize,
    },
    IndexOutOfBounds {
        match_index: usize,
        source_a_index: usize,
        source_b_index: usize,
        source_a_len: usize,
        source_b_len: usize,
    },
    DuplicateIndex {
        source: &'static str,
        index: usize,
    },
    NonFiniteScore {
        match_index: usize,
        score: f32,
    },
    KeyframeSourceMismatch {
        matches_frame: FrameId,
        keyframe_frame: FrameId,
    },
}

impl std::fmt::Display for MatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatchError::Mismatch {
                score_len,
                indices_len,
            } => write!(
                f,
                "match shape mismatch: scores={score_len}, indices={indices_len}"
            ),
            MatchError::IndexOutOfBounds {
                match_index,
                source_a_index,
                source_b_index,
                source_a_len,
                source_b_len,
            } => write!(
                f,
                "match {match_index} index out of bounds: ({source_a_index}, {source_b_index}) for lengths ({source_a_len}, {source_b_len})"
            ),
            MatchError::DuplicateIndex { source, index } => {
                write!(f, "match source {source} reuses detection index {index}")
            }
            MatchError::NonFiniteScore { match_index, score } => {
                write!(f, "match {match_index} has nonfinite score {score}")
            }
            MatchError::KeyframeSourceMismatch {
                matches_frame,
                keyframe_frame,
            } => write!(
                f,
                "verified match source frame {} does not match keyframe frame {}",
                matches_frame.as_u64(),
                keyframe_frame.as_u64()
            ),
        }
    }
}

impl std::error::Error for MatchError {}

#[derive(Debug)]
pub struct Matches<State> {
    source_a: Arc<Detections>,
    source_b: Arc<Detections>,
    indices: Vec<(usize, usize)>,
    scores: Vec<f32>,
    _state: PhantomData<State>,
}

impl Matches<Raw> {
    pub fn new(
        source_a: Arc<Detections>,
        source_b: Arc<Detections>,
        indices: Vec<(usize, usize)>,
        scores: Vec<f32>,
    ) -> Result<Self, MatchError> {
        Matches::<Raw>::from_parts(source_a, source_b, indices, scores)
    }

    pub(crate) fn with_landmarks(
        &self,
        keyframe: &Keyframe,
    ) -> Result<Matches<Verified>, MatchError> {
        if self.source_b.frame_id() != keyframe.frame_id()
            || !Arc::ptr_eq(&self.source_b, keyframe.detections())
        {
            return Err(MatchError::KeyframeSourceMismatch {
                matches_frame: self.source_b.frame_id(),
                keyframe_frame: keyframe.frame_id(),
            });
        }
        let mut indices = Vec::new();
        let mut scores = Vec::new();
        for (idx, &(a, b)) in self.indices.iter().enumerate() {
            if keyframe.landmark_for_detection(b).is_some() {
                indices.push((a, b));
                scores.push(self.scores[idx]);
            }
        }

        Matches::new_verified(self.source_a_arc(), self.source_b_arc(), indices, scores)
    }
}

impl Matches<Verified> {
    fn new_verified(
        source_a: Arc<Detections>,
        source_b: Arc<Detections>,
        indices: Vec<(usize, usize)>,
        scores: Vec<f32>,
    ) -> Result<Self, MatchError> {
        Matches::<Verified>::from_parts(source_a, source_b, indices, scores)
    }
}

impl<State> Matches<State> {
    fn from_parts(
        source_a: Arc<Detections>,
        source_b: Arc<Detections>,
        indices: Vec<(usize, usize)>,
        scores: Vec<f32>,
    ) -> Result<Self, MatchError> {
        if indices.len() != scores.len() {
            return Err(MatchError::Mismatch {
                score_len: scores.len(),
                indices_len: indices.len(),
            });
        }
        let mut seen_a = vec![false; source_a.len()];
        let mut seen_b = vec![false; source_b.len()];
        for (match_index, (&(a, b), &score)) in indices.iter().zip(&scores).enumerate() {
            if a >= source_a.len() || b >= source_b.len() {
                return Err(MatchError::IndexOutOfBounds {
                    match_index,
                    source_a_index: a,
                    source_b_index: b,
                    source_a_len: source_a.len(),
                    source_b_len: source_b.len(),
                });
            }
            if seen_a[a] {
                return Err(MatchError::DuplicateIndex {
                    source: "a",
                    index: a,
                });
            }
            if seen_b[b] {
                return Err(MatchError::DuplicateIndex {
                    source: "b",
                    index: b,
                });
            }
            if !score.is_finite() {
                return Err(MatchError::NonFiniteScore { match_index, score });
            }
            seen_a[a] = true;
            seen_b[b] = true;
        }
        Ok(Self {
            source_a,
            source_b,
            indices,
            scores,
            _state: PhantomData,
        })
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn source_a(&self) -> &Detections {
        &self.source_a
    }

    pub fn source_b(&self) -> &Detections {
        &self.source_b
    }

    pub fn source_a_arc(&self) -> Arc<Detections> {
        Arc::clone(&self.source_a)
    }

    pub fn source_b_arc(&self) -> Arc<Detections> {
        Arc::clone(&self.source_b)
    }

    pub fn indices(&self) -> &[(usize, usize)] {
        &self.indices
    }

    pub fn scores(&self) -> &[f32] {
        &self.scores
    }
}

#[derive(Debug)]
pub enum VizError {
    FrameMismatch {
        left: FrameId,
        right: FrameId,
        matches_left: FrameId,
        matches_right: FrameId,
    },
    SensorMismatch {
        left: SensorId,
        right: SensorId,
        matches_left: SensorId,
        matches_right: SensorId,
    },
}

impl std::fmt::Display for VizError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VizError::FrameMismatch {
                left,
                right,
                matches_left,
                matches_right,
            } => write!(
                f,
                "viz packet frame mismatch: left={}, right={}, matches_left={}, matches_right={}",
                left.as_u64(),
                right.as_u64(),
                matches_left.as_u64(),
                matches_right.as_u64()
            ),
            VizError::SensorMismatch {
                left,
                right,
                matches_left,
                matches_right,
            } => write!(
                f,
                "viz packet sensor mismatch: left={left:?}, right={right:?}, matches_left={matches_left:?}, matches_right={matches_right:?}"
            ),
        }
    }
}

impl std::error::Error for VizError {}

#[derive(Debug)]
pub struct VizPacket<State> {
    left: Frame,
    right: Frame,
    matches: Matches<State>,
}

impl<State> VizPacket<State> {
    pub fn try_new(left: Frame, right: Frame, matches: Matches<State>) -> Result<Self, VizError> {
        let matches_left = matches.source_a().frame_id();
        let matches_right = matches.source_b().frame_id();
        let matches_left_sensor = matches.source_a().sensor_id();
        let matches_right_sensor = matches.source_b().sensor_id();
        if left.frame_id() != matches_left || right.frame_id() != matches_right {
            return Err(VizError::FrameMismatch {
                left: left.frame_id(),
                right: right.frame_id(),
                matches_left,
                matches_right,
            });
        }
        if left.sensor_id() != matches_left_sensor || right.sensor_id() != matches_right_sensor {
            return Err(VizError::SensorMismatch {
                left: left.sensor_id(),
                right: right.sensor_id(),
                matches_left: matches_left_sensor,
                matches_right: matches_right_sensor,
            });
        }

        Ok(Self {
            left,
            right,
            matches,
        })
    }

    pub fn left(&self) -> &Frame {
        &self.left
    }

    pub fn right(&self) -> &Frame {
        &self.right
    }

    pub fn matches(&self) -> &Matches<State> {
        &self.matches
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CompactDescriptor, DESCRIPTOR_DIM, Descriptor, DetectionError, Detections, Frame,
        FrameDimensionsError, FrameError, FrameId, Keypoint, SensorId, Timestamp,
    };

    #[test]
    fn frame_stores_the_dimensions_parsed_at_construction() {
        let frame = Frame::new(
            SensorId::StereoLeft,
            FrameId::new(1),
            Timestamp::from_nanos(2),
            2,
            3,
            vec![0; 6],
        )
        .expect("valid frame");

        assert_eq!(frame.dimensions().width(), 2);
        assert_eq!(frame.dimensions().height(), 3);
        assert_eq!(frame.data().len(), frame.dimensions().area());
    }

    #[test]
    fn frame_preserves_the_typed_dimension_error() {
        let error = Frame::new(
            SensorId::StereoLeft,
            FrameId::new(1),
            Timestamp::from_nanos(2),
            0,
            3,
            Vec::new(),
        )
        .expect_err("zero width is outside the frame-dimension domain");

        assert!(matches!(
            error,
            FrameError::InvalidDimensions {
                source: FrameDimensionsError::Zero {
                    width: 0,
                    height: 3
                }
            }
        ));
    }

    #[test]
    fn detection_bounds_preserve_large_u32_dimensions() {
        const WIDTH: u32 = 16_777_217;
        const LAST_REPRESENTABLE_COLUMN: f32 = 16_777_216.0;
        const NEXT_REPRESENTABLE_COLUMN: f32 = 16_777_218.0;

        Detections::new(
            SensorId::StereoLeft,
            FrameId::new(1),
            WIDTH,
            1,
            vec![Keypoint {
                x: LAST_REPRESENTABLE_COLUMN,
                y: 0.0,
            }],
            vec![1.0],
            vec![Descriptor([0.0; DESCRIPTOR_DIM])],
        )
        .expect("the last representable column below width remains in bounds");

        let error = Detections::new(
            SensorId::StereoLeft,
            FrameId::new(2),
            WIDTH,
            1,
            vec![Keypoint {
                x: NEXT_REPRESENTABLE_COLUMN,
                y: 0.0,
            }],
            vec![1.0],
            vec![Descriptor([0.0; DESCRIPTOR_DIM])],
        )
        .expect_err("the next representable coordinate is outside the image");

        assert!(matches!(
            error,
            DetectionError::KeypointOutOfBounds {
                index: 0,
                x: NEXT_REPRESENTABLE_COLUMN,
                y: 0.0,
                width: WIDTH,
                height: 1,
            }
        ));
    }

    fn cosine_f32(a: &Descriptor, b: &Descriptor) -> f32 {
        let mut dot = 0.0_f32;
        let mut norm_a = 0.0_f32;
        let mut norm_b = 0.0_f32;
        for i in 0..DESCRIPTOR_DIM {
            let x = a.0[i];
            let y = b.0[i];
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }
        if norm_a <= 0.0 || norm_b <= 0.0 {
            return 0.0;
        }
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }

    #[test]
    fn quantize_preserves_signed_similarity_ordering() {
        let mut base = [0.0_f32; DESCRIPTOR_DIM];
        let mut close = [0.0_f32; DESCRIPTOR_DIM];
        let mut far = [0.0_f32; DESCRIPTOR_DIM];
        for i in 0..DESCRIPTOR_DIM {
            let t = (i as f32 / 127.5) - 1.0;
            base[i] = t;
            close[i] = (t + if i % 2 == 0 { 0.01 } else { -0.01 }).clamp(-1.0, 1.0);
            far[i] = -t;
        }
        let base = Descriptor(base);
        let close = Descriptor(close);
        let far = Descriptor(far);

        let float_close = cosine_f32(&base, &close);
        let float_far = cosine_f32(&base, &far);
        assert!(float_close > float_far);

        let q_base = base.quantize();
        let q_close = close.quantize();
        let q_far = far.quantize();
        let quant_close = q_base.cosine_similarity(&q_close);
        let quant_far = q_base.cosine_similarity(&q_far);
        assert!(quant_close > quant_far);
    }

    #[test]
    fn quantize_uses_symmetric_centered_u8_codes() {
        let mut descriptor = Descriptor([0.0; DESCRIPTOR_DIM]);
        descriptor.0[..5].copy_from_slice(&[-1.0, -0.5, 0.0, 0.5, 1.0]);

        let compact = descriptor.quantize();

        assert_eq!(&compact.0[..5], &[1, 64, 128, 192, 255]);
        assert!(compact.0[5..].iter().all(|&value| value == 128));
    }

    #[test]
    fn compact_descriptor_negative_components_are_self_similar() {
        let descriptor = Descriptor(std::array::from_fn(|index| {
            -((index % 127 + 1) as f32 / 127.0)
        }));
        let compact = descriptor.quantize();

        assert!(compact.0.iter().all(|&value| value < 128));
        let sim = compact.cosine_similarity(&compact);
        assert!((sim - 1.0).abs() < 1e-6, "sim={sim}");
    }

    #[test]
    fn compact_descriptor_antipodal_signed_vectors_are_negative_one() {
        let descriptor = Descriptor(std::array::from_fn(|index| {
            ((index % 127 + 1) as f32 / 127.0) * if index % 2 == 0 { 1.0 } else { -1.0 }
        }));
        let antipode = Descriptor(descriptor.0.map(|value| -value));

        let sim = descriptor
            .quantize()
            .cosine_similarity(&antipode.quantize());

        assert!((sim + 1.0).abs() < 1e-6, "sim={sim}");
    }

    #[test]
    fn compact_descriptor_orthogonal_signed_vectors_are_zero() {
        let mut a = [128_u8; DESCRIPTOR_DIM];
        let mut b = [128_u8; DESCRIPTOR_DIM];
        a[0] = 255;
        b[1] = 1;

        let sim = CompactDescriptor(a).cosine_similarity(&CompactDescriptor(b));
        assert!(sim.abs() < 1e-6, "sim={sim}");
    }

    #[test]
    fn compact_descriptor_zero_vector_has_no_similarity() {
        let zero = CompactDescriptor([128; DESCRIPTOR_DIM]);
        let mut nonzero = [128; DESCRIPTOR_DIM];
        nonzero[0] = 255;
        let nonzero = CompactDescriptor(nonzero);

        assert_eq!(zero.cosine_similarity(&zero), 0.0);
        assert_eq!(zero.cosine_similarity(&nonzero), 0.0);
        assert_eq!(nonzero.cosine_similarity(&zero), 0.0);
    }

    #[test]
    fn compact_descriptor_cosine_is_finite_symmetric_and_bounded() {
        let mut state = 0x9e37_79b9_u32;
        for _ in 0..1_024 {
            let a = CompactDescriptor(std::array::from_fn(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as u8
            }));
            let b = CompactDescriptor(std::array::from_fn(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as u8
            }));

            let ab = a.cosine_similarity(&b);
            let ba = b.cosine_similarity(&a);
            assert!(ab.is_finite(), "similarity must be finite: {ab}");
            assert!((-1.0..=1.0).contains(&ab), "similarity out of bounds: {ab}");
            assert_eq!(ab, ba, "cosine must be symmetric");
            assert!((a.cosine_similarity(&a) - 1.0).abs() < 1e-6);
            assert!((b.cosine_similarity(&b) - 1.0).abs() < 1e-6);
        }
    }
}
