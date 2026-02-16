use std::num::NonZeroUsize;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::{
    DownscaleFactor, LightGlue, Matches, Raw, StereoPair, SuperPoint, VizError, VizPacket,
    inference::InferenceError,
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
    superpoint_left: SuperPoint,
    superpoint_right: SuperPoint,
    lightglue: LightGlue,
    max_keypoints: KeypointLimit,
    downscale: DownscaleFactor,
}

impl InferencePipeline {
    pub fn new(
        superpoint_left: SuperPoint,
        superpoint_right: SuperPoint,
        lightglue: LightGlue,
        max_keypoints: KeypointLimit,
    ) -> Self {
        Self {
            superpoint_left,
            superpoint_right,
            lightglue,
            max_keypoints,
            downscale: DownscaleFactor::identity(),
        }
    }

    pub fn max_keypoints(&self) -> KeypointLimit {
        self.max_keypoints
    }

    pub fn downscale(&self) -> DownscaleFactor {
        self.downscale
    }

    pub fn with_downscale(mut self, downscale: DownscaleFactor) -> Self {
        self.downscale = downscale;
        self
    }

    pub fn process_pair(&mut self, pair: StereoPair) -> Result<VizPacket<Raw>, PipelineError> {
        let (packet, _) = self.process_pair_timed(pair)?;
        Ok(packet)
    }

    pub fn process_pair_timed(
        &mut self,
        pair: StereoPair,
    ) -> Result<(VizPacket<Raw>, PipelineTimings), PipelineError> {
        let total_start = Instant::now();
        let StereoPair { left, right } = pair;
        let left_frame = left;
        let right_frame = right;
        let downscale = self.downscale;
        let max_keypoints = self.max_keypoints.get();

        let (left_result, right_result) = thread::scope(|scope| {
            let left_sp = &mut self.superpoint_left;
            let right_sp = &mut self.superpoint_right;
            let left_ref = &left_frame;
            let right_ref = &right_frame;

            let left_handle = scope.spawn(move || {
                let start = Instant::now();
                let det = left_sp
                    .detect_with_downscale(left_ref, downscale)?
                    .top_k(max_keypoints);
                Ok::<_, InferenceError>((det, start.elapsed()))
            });

            let right_handle = scope.spawn(move || {
                let start = Instant::now();
                let det = right_sp
                    .detect_with_downscale(right_ref, downscale)?
                    .top_k(max_keypoints);
                Ok::<_, InferenceError>((det, start.elapsed()))
            });

            (left_handle.join(), right_handle.join())
        });

        let (left_det, left_time) = left_result
            .map_err(|_| InferenceError::Domain("left superpoint thread panicked".to_string()))??;
        let (right_det, right_time) = right_result.map_err(|_| {
            InferenceError::Domain("right superpoint thread panicked".to_string())
        })??;

        let left = Arc::new(left_det);
        let right = Arc::new(right_det);

        let match_start = Instant::now();
        let matches = if left.is_empty() || right.is_empty() {
            Matches::new(left.clone(), right.clone(), Vec::new(), Vec::new())
                .map_err(|e| InferenceError::Domain(format!("{e:?}")))?
        } else {
            self.lightglue.match_these(left.clone(), right.clone())?
        };
        let match_time = match_start.elapsed();

        let packet = VizPacket::try_new(left_frame, right_frame, matches)?;
        let total = total_start.elapsed();

        let timings = PipelineTimings {
            superpoint_left: left_time,
            superpoint_right: right_time,
            lightglue: match_time,
            total,
        };

        Ok((packet, timings))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PipelineTimings {
    pub superpoint_left: Duration,
    pub superpoint_right: Duration,
    pub lightglue: Duration,
    pub total: Duration,
}
