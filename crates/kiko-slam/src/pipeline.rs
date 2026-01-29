use std::num::NonZeroUsize;
use std::sync::Arc;

use crate::{
    inference::InferenceError, LightGlue, Matches, Raw, StereoPair, SuperPoint, VizError,
    VizPacket,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeypointLimit(NonZeroUsize);

impl KeypointLimit {
    pub fn new(limit: NonZeroUsize) -> Self {
        Self(limit)
    }

    pub fn get(self) -> usize {
        self.0.get()
    }
}

#[derive(Debug)]
pub enum KeypointLimitError {
    Zero,
}

impl std::fmt::Display for KeypointLimitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KeypointLimitError::Zero => write!(f, "keypoint limit must be > 0"),
        }
    }
}

impl std::error::Error for KeypointLimitError {}

impl TryFrom<usize> for KeypointLimit {
    type Error = KeypointLimitError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        NonZeroUsize::new(value)
            .map(KeypointLimit)
            .ok_or(KeypointLimitError::Zero)
    }
}

#[derive(Debug)]
pub enum PipelineError {
    Inference(InferenceError),
    Viz(VizError),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::Inference(err) => write!(f, "inference error: {err}"),
            PipelineError::Viz(err) => write!(f, "viz error: {err}"),
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<InferenceError> for PipelineError {
    fn from(err: InferenceError) -> Self {
        PipelineError::Inference(err)
    }
}

impl From<VizError> for PipelineError {
    fn from(err: VizError) -> Self {
        PipelineError::Viz(err)
    }
}

pub struct InferencePipeline {
    superpoint: SuperPoint,
    lightglue: LightGlue,
    max_keypoints: KeypointLimit,
}

impl InferencePipeline {
    pub fn new(superpoint: SuperPoint, lightglue: LightGlue, max_keypoints: KeypointLimit) -> Self {
        Self {
            superpoint,
            lightglue,
            max_keypoints,
        }
    }

    pub fn max_keypoints(&self) -> KeypointLimit {
        self.max_keypoints
    }

    pub fn process_pair(&mut self, pair: StereoPair) -> Result<VizPacket<Raw>, PipelineError> {
        let left = self
            .superpoint
            .detect(&pair.left)?
            .top_k(self.max_keypoints.get());
        let right = self
            .superpoint
            .detect(&pair.right)?
            .top_k(self.max_keypoints.get());

        let left = Arc::new(left);
        let right = Arc::new(right);

        let matches = if left.is_empty() || right.is_empty() {
            Matches::new(left.clone(), right.clone(), Vec::new(), Vec::new())
                .map_err(|e| InferenceError::Domain(format!("{e:?}")))?
        } else {
            self.lightglue.match_these(left.clone(), right.clone())?
        };

        Ok(VizPacket::try_new(pair.left, pair.right, matches)?)
    }
}
