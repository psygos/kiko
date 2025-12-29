#![warn(clippy::all)]
pub use inference::SuperPoint;
mod inference;
mod preprocess;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SensorId {
    StereoLeft,
    StereoRight,
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

#[derive(Clone, Debug)]
pub struct Frame {
    id: SensorId,
    timestamp: Timestamp,
    width: u32,
    height: u32,
    data: Vec<u8>,
}

impl Frame {
    pub fn new(
        id: SensorId,
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
            id,
            timestamp,
            width,
            height,
            data,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn id(&self) -> SensorId {
        self.id
    }

    pub fn timestamp(&self) -> Timestamp {
        self.timestamp
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
}

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
    keypoints: Vec<Keypoint>,
    scores: Vec<f32>,
    descriptors: Vec<Descriptor>,
}

impl Detections {
    pub fn keypoints(&self) -> &[Keypoint] {
        &self.keypoints
    }

    pub fn scores(&self) -> &[f32] {
        &self.scores
    }

    pub fn descriptors(&self) -> &[Descriptor] {
        &self.descriptors
    }

    pub fn len(&self) -> usize {
        self.descriptors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.descriptors.len() == 0
    }

    pub fn new(
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
            keypoints,
            scores,
            descriptors,
        })
    }
}

pub trait FrameSource {
    fn next_frame(&mut self) -> Option<Frame>;
}
