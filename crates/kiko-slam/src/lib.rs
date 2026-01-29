#![warn(clippy::all)]
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::sync::Arc;

pub use inference::{InferenceBackend, LightGlue, SuperPoint};
pub mod dataset;
mod inference;
mod preprocess;
mod pairing;
mod pipeline;
mod viz;
mod channel;
#[cfg(feature = "record")]
mod oak;
pub use pairing::{PairingConfigError, PairingStats, PairingWindowNs, StereoPairer};
pub use pipeline::{
    InferencePipeline, KeypointLimit, KeypointLimitError, PipelineError, PipelineTimings,
};
pub use viz::{RerunSink, VizDecimation, VizDecimationError, VizLogError};
pub use channel::{
    bounded_channel, ChannelCapacity, ChannelCapacityError, ChannelStats, ChannelStatsHandle,
    DropPolicy, DropReceiver, DropSender, SendOutcome,
};
#[cfg(feature = "record")]
pub use oak::oak_to_frame;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SensorId {
    StereoLeft,
    StereoRight,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
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
        left: (u32, u32),
        right: (u32, u32),
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
                    left.0, left.1, right.0, right.1
                )
            }
            PairError::TimestampDelta {
                delta_ns,
                max_delta_ns,
            } => {
                write!(
                    f,
                    "stereo delta {}ns exceeds window {}ns",
                    delta_ns, max_delta_ns
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
        Self(NonZeroUsize::new(1).expect("nonzero"))
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
                "downscale factor {factor} does not divide frame {}x{}",
                width, height
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
#[derive(Debug, Clone, Copy)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
}

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct Descriptor(pub [f32; 256]);

impl Descriptor {
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }
}

#[derive(Debug)]
pub enum DetectionError {
    ShapeMismatch { msg: String },
}

// i need to create invariant of descriptors.len() and keypoint.len() remain the same
#[derive(Debug, Clone)]
pub struct Detections {
    sensor_id: SensorId,
    frame_id: FrameId,
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

    pub fn keypoints(&self) -> &[Keypoint] {
        &self.keypoints
    }

    pub fn keypoints_flat(&self) -> &[f32] {
        let len = self.keypoints.len() * 2;
        let ptr = self.keypoints.as_ptr() as *const f32;
        // Safety: Keypoint is #[repr(C)] over two f32 values, so this is a
        // contiguous [f32; 2] slice with no padding.
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    pub fn scores(&self) -> &[f32] {
        &self.scores
    }

    pub fn descriptors(&self) -> &[Descriptor] {
        &self.descriptors
    }

    pub fn descriptors_flat(&self) -> &[f32] {
        let len = self.descriptors.len() * 256;
        let ptr = self.descriptors.as_ptr() as *const f32;
        // Safety: Descriptor is #[repr(transparent)] over [f32; 256].
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    pub fn len(&self) -> usize {
        self.descriptors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.descriptors.len() == 0
    }

    pub fn new(
        sensor_id: SensorId,
        frame_id: FrameId,
        keypoints: Vec<Keypoint>,
        scores: Vec<f32>,
        descriptors: Vec<Descriptor>,
    ) -> Result<Self, DetectionError> {
        if keypoints.len() != descriptors.len() || descriptors.len() != scores.len() {
            return Err(DetectionError::ShapeMismatch {
                msg: "There is a mismatch between sizes of descriptors, scores and keypoints"
                    .to_string(),
            });
        }

        Ok(Self {
            sensor_id,
            frame_id,
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
            keypoints,
            scores,
            descriptors,
        } = self;

        let mut order: Vec<usize> = (0..descriptors.len()).collect();
        let kth = max.saturating_sub(1);
        order.select_nth_unstable_by(kth, |&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        order.truncate(max);
        order.sort_unstable_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut new_keypoints = Vec::with_capacity(order.len());
        let mut new_scores = Vec::with_capacity(order.len());
        let mut new_descriptors = Vec::with_capacity(order.len());

        for &idx in &order {
            new_keypoints.push(keypoints[idx]);
            new_scores.push(scores[idx]);
            new_descriptors.push(descriptors[idx].clone());
        }

        Self {
            sensor_id,
            frame_id,
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
    pub left: Frame,
    pub right: Frame,
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
                left: (left.width(), left.height()),
                right: (right.width(), right.height()),
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

    pub fn timestamp_delta_ns(&self) -> i64 {
        (self.left.timestamp().as_nanos() - self.right.timestamp().as_nanos()).abs()
    }
}

pub trait StereoSource {
    fn left(&mut self) -> Option<Frame>;
    fn right(&mut self) -> Option<Frame>;

    fn stereo_pair(&mut self) -> Option<StereoPair> {
        Some(StereoPair {
            left: self.left()?,
            right: self.right()?,
        })
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

// Typesstates for Matches
pub struct Raw;
pub struct Verified;

#[derive(Debug)]
pub enum MatchError {
    MissMatch {
        score_len: usize,
        indices_len: usize,
    },
}

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
        if indices.len() != scores.len() {
            return Err(MatchError::MissMatch {
                score_len: (scores.len()),
                indices_len: (indices.len()),
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
}

impl<State> Matches<State> {
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
