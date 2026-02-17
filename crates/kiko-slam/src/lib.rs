use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::sync::Arc;

pub use inference::{
    EigenPlaces, InferenceBackend, LightGlue, PlaceDescriptorExtractor, SuperPoint,
};
mod channel;
pub mod dataset;
pub mod dense;
mod depth;
mod diagnostics;
pub mod env;
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
pub use dense::{DenseCommand, DenseConfig, DenseStats, ReconState};
pub use depth::{DepthImage, DepthImageError};
pub use diagnostics::{
    DiagnosticEvent, FrameDiagnostics, KeyframeRemovalReason, LoopClosureRejectReason,
};
pub use env::{env_bool, env_f32, env_usize};
pub use local_ba::{
    BaCorrection, BaResult, DegenerateReason, LmConfig, LmConfigError, LocalBaConfig,
    LocalBaConfigError, LocalBundleAdjuster, MapObservation, ObservationSet, ObservationSetError,
};
pub use loop_closure::{
    DescriptorSource, GlobalDescriptor, GlobalDescriptorError, KeyframeDatabase, LoopApplyError,
    LoopCandidate, LoopClosureConfig, LoopClosureConfigError, LoopClosureConfigInput,
    LoopDetectError, LoopVerificationError, PlaceMatch, RelocalizationCandidate,
    RelocalizationConfig, RelocalizationConfigError, RelocalizationConfigInput,
    RelocalizationMatch, VerifiedLoop, VerifiedRelocalization, aggregate_global_descriptor,
    match_descriptors_for_loop,
};
pub use map::{CovisibilityEdge, CovisibilityNode, CovisibilitySnapshot};
pub use math::Pose64;
#[cfg(feature = "record")]
pub use oak::{oak_to_depth_image, oak_to_frame};
pub use pairing::{PairingConfigError, PairingStats, PairingWindowNs, StereoPairer};
pub use pipeline::{
    InferencePipeline, KeypointLimit, KeypointLimitError, PipelineError, PipelineTimings,
};
pub use pnp::{
    IntrinsicsError, Observation, PinholeIntrinsics, PnpError, PnpResult, Pose, RansacConfig,
    build_observations, solve_pnp, solve_pnp_ransac,
};
pub use tracker::{
    BackendConfig, BackendConfigError, BackendStats, ComponentHealth, CovisibilityRatio,
    DegradationLevel, DescriptorStats, GlobalDescriptorConfig, GlobalDescriptorConfigError,
    KeyframePolicy, KeyframePolicyError, LoopSubsystemConfig, ParallaxPx, RedundancyPolicy,
    RedundancyPolicyError, SlamTracker, SystemHealth, TrackerConfig, TrackerError, TrackerOutput,
    TrackingHealth,
};
pub use triangulation::{
    Keyframe, KeyframeError, Point3, RectificationMode, RectifiedStereo, RectifiedStereoConfig,
    RectifiedStereoError, TriangulationConfig, TriangulationError, TriangulationResult,
    TriangulationStats, Triangulator,
};
pub use viz::{RerunSink, VizDecimation, VizDecimationError, VizLogError};

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
    width: u32,
    height: u32,
}

impl FrameDimensions {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub fn width(self) -> u32 {
        self.width
    }

    pub fn height(self) -> u32 {
        self.height
    }

    pub fn area(self) -> usize {
        (self.width as usize).saturating_mul(self.height as usize)
    }
}

// Define these much more concretely
#[derive(Debug)]
pub enum FrameError {
    DimensionMismatch { expected: usize, actual: usize },
}
impl std::fmt::Display for FrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameError::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for FrameError {}

#[derive(Debug)]
pub enum PairError {
    DimensionMismatch {
        left: FrameDimensions,
        right: FrameDimensions,
    },
    TimestampDelta {
        delta_ns: i64,
        max_delta_ns: i64,
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
pub struct DownscaleFactor(NonZeroUsize);

impl DownscaleFactor {
    pub fn new(value: NonZeroUsize) -> Self {
        Self(value)
    }

    pub fn get(self) -> usize {
        self.0.get()
    }

    pub fn identity() -> Self {
        Self(NonZeroUsize::MIN)
    }
}

#[derive(Debug)]
pub enum DownscaleError {
    Zero,
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
        NonZeroUsize::new(value)
            .map(DownscaleFactor)
            .ok_or(DownscaleError::Zero)
    }
}

#[derive(Clone, Debug)]
pub struct Frame {
    sensor_id: SensorId,
    frame_id: FrameId,
    timestamp: Timestamp,
    width: u32,
    height: u32,
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
        let size = (width as usize) * (height as usize);

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
            width,
            height,
            data: Arc::from(data.into_boxed_slice()),
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn dimensions(&self) -> FrameDimensions {
        FrameDimensions::new(self.width, self.height)
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
const U8_SCALE: f32 = 255.0;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Descriptor(pub [f32; DESCRIPTOR_DIM]);

impl Descriptor {
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    pub fn quantize(&self) -> CompactDescriptor {
        let mut out = [0_u8; DESCRIPTOR_DIM];
        for (idx, value) in self.0.iter().enumerate() {
            let clamped = value.clamp(0.0, 1.0);
            out[idx] = (clamped * U8_SCALE).round() as u8;
        }
        CompactDescriptor(out)
    }
}

#[repr(transparent)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactDescriptor(pub [u8; DESCRIPTOR_DIM]);

impl CompactDescriptor {
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let (dot, norm_a, norm_b) = self.0.iter().zip(other.0.iter()).fold(
            (0_u32, 0_u32, 0_u32),
            |(dot, na, nb), (&a, &b)| {
                let a = a as u32;
                let b = b as u32;
                (
                    dot.saturating_add(a.saturating_mul(b)),
                    na.saturating_add(a.saturating_mul(a)),
                    nb.saturating_add(b.saturating_mul(b)),
                )
            },
        );
        if norm_a == 0 || norm_b == 0 {
            return 0.0;
        }
        (dot as f32) / ((norm_a as f32).sqrt() * (norm_b as f32).sqrt())
    }
}

#[derive(Debug)]
pub enum DetectionError {
    ShapeMismatch {
        keypoints_len: usize,
        scores_len: usize,
        descriptors_len: usize,
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

pub trait FrameSource {
    fn next_frame(&mut self) -> Option<Frame>;

    fn frames(self) -> Frames<Self>
    where
        Self: Sized,
    {
        Frames::new(self)
    }
}

pub struct Frames<S> {
    source: S,
}

impl<S> Frames<S> {
    pub fn new(source: S) -> Self {
        Self { source }
    }
}

impl<S: FrameSource> Iterator for Frames<S> {
    type Item = Frame;

    fn next(&mut self) -> Option<Self::Item> {
        self.source.next_frame()
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

        let delta = (left.timestamp().as_nanos() - right.timestamp().as_nanos()).abs();
        if delta > window.as_ns() {
            return Err(PairError::TimestampDelta {
                delta_ns: delta,
                max_delta_ns: window.as_ns(),
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

    pub fn timestamp_delta_ns(&self) -> i64 {
        (self.left.timestamp().as_nanos() - self.right.timestamp().as_nanos()).abs()
    }
}

pub trait StereoSource {
    fn left(&mut self) -> Option<Frame>;
    fn right(&mut self) -> Option<Frame>;

    fn stereo_pair(&mut self) -> Option<StereoPair> {
        Some(StereoPair::from_parts(self.left()?, self.right()?))
    }

    fn stereo_pairs(self) -> StereoPairs<Self>
    where
        Self: Sized,
    {
        StereoPairs::new(self)
    }

    fn left_frames(self) -> LeftFrames<Self>
    where
        Self: Sized,
    {
        LeftFrames::new(self)
    }

    fn right_frames(self) -> RightFrames<Self>
    where
        Self: Sized,
    {
        RightFrames::new(self)
    }
}

pub struct StereoPairs<S> {
    source: S,
}

impl<S> StereoPairs<S> {
    pub fn new(source: S) -> Self {
        Self { source }
    }
}

impl<S: StereoSource> Iterator for StereoPairs<S> {
    type Item = StereoPair;

    fn next(&mut self) -> Option<Self::Item> {
        self.source.stereo_pair()
    }
}

pub struct LeftFrames<S> {
    source: S,
}

impl<S> LeftFrames<S> {
    pub fn new(source: S) -> Self {
        Self { source }
    }
}

impl<S: StereoSource> Iterator for LeftFrames<S> {
    type Item = Frame;

    fn next(&mut self) -> Option<Self::Item> {
        self.source.left()
    }
}

pub struct RightFrames<S> {
    source: S,
}

impl<S> RightFrames<S> {
    pub fn new(source: S) -> Self {
        Self { source }
    }
}

impl<S: StereoSource> Iterator for RightFrames<S> {
    type Item = Frame;

    fn next(&mut self) -> Option<Self::Item> {
        self.source.right()
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

    pub fn with_landmarks(&self, keyframe: &Keyframe) -> Result<Matches<Verified>, MatchError> {
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
    pub fn new_verified(
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
    use super::{CompactDescriptor, DESCRIPTOR_DIM, Descriptor, U8_SCALE};

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
    fn quantize_preserves_similarity_ordering() {
        let mut base = [0.0_f32; DESCRIPTOR_DIM];
        let mut close = [0.0_f32; DESCRIPTOR_DIM];
        let mut far = [0.0_f32; DESCRIPTOR_DIM];
        for i in 0..DESCRIPTOR_DIM {
            let t = i as f32 / U8_SCALE;
            base[i] = t;
            close[i] = (t + 0.02).clamp(0.0, 1.0);
            far[i] = if i < 128 { 1.0 } else { 0.0 };
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
    fn compact_descriptor_cosine_identical_is_one() {
        let mut data = [0_u8; DESCRIPTOR_DIM];
        for (idx, value) in data.iter_mut().enumerate() {
            *value = ((idx * 7) % 251) as u8;
        }
        let a = CompactDescriptor(data);
        let b = CompactDescriptor(data);
        let sim = a.cosine_similarity(&b);
        assert!((sim - 1.0).abs() < 1e-6, "sim={sim}");
    }

    #[test]
    fn compact_descriptor_cosine_orthogonal_is_zeroish() {
        let mut a = [0_u8; DESCRIPTOR_DIM];
        let mut b = [0_u8; DESCRIPTOR_DIM];
        for value in a.iter_mut().take(128) {
            *value = 255;
        }
        for value in b.iter_mut().skip(128) {
            *value = 255;
        }
        let sim = CompactDescriptor(a).cosine_similarity(&CompactDescriptor(b));
        assert!(sim.abs() < 1e-6, "sim={sim}");
    }
}
