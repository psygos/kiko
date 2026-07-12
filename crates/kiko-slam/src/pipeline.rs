use std::num::NonZeroUsize;
use std::sync::Arc;
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

    pub fn min() -> Self {
        Self(NonZeroUsize::MIN)
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

impl std::fmt::Display for KeypointLimit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.get())
    }
}

impl std::str::FromStr for KeypointLimit {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let value: usize = s
            .trim()
            .parse()
            .map_err(|_| format!("invalid keypoint limit: {s}"))?;
        Self::try_from(value).map_err(|e| e.to_string())
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

impl std::error::Error for PipelineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PipelineError::Inference(source) => Some(source),
            PipelineError::Viz(source) => Some(source),
        }
    }
}

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
    superpoint_right: Option<SuperPoint>,
    lightglue: LightGlue,
    max_keypoints: KeypointLimit,
    downscale: DownscaleFactor,
}

impl InferencePipeline {
    pub fn new(superpoint: SuperPoint, lightglue: LightGlue, max_keypoints: KeypointLimit) -> Self {
        Self {
            superpoint_left: superpoint,
            superpoint_right: None,
            lightglue,
            max_keypoints,
            downscale: DownscaleFactor::identity(),
        }
    }

    pub fn with_stereo_superpoint(mut self, right: SuperPoint) -> Self {
        self.superpoint_right = Some(right);
        self
    }

    pub fn with_stereo_superpoint_opt(mut self, right: Option<SuperPoint>) -> Self {
        self.superpoint_right = right;
        self
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
        let (left_frame, right_frame) = pair.into_parts();
        let downscale = self.downscale;
        let max_keypoints = self.max_keypoints.get();

        let (left_det, left_time, right_det, right_time) =
            if let Some(sp_right) = &mut self.superpoint_right {
                // Parallel SP: run left and right on separate threads
                let sp_left = &mut self.superpoint_left;
                std::thread::scope(|s| {
                    let left_handle = s.spawn(|| {
                        let start = Instant::now();
                        let det = sp_left
                            .detect_with_downscale_limited(&left_frame, downscale, max_keypoints)
                            .map(|d| d.top_k(max_keypoints));
                        (det, start.elapsed())
                    });
                    let right_handle = s.spawn(|| {
                        let start = Instant::now();
                        let det = sp_right
                            .detect_with_downscale_limited(&right_frame, downscale, max_keypoints)
                            .map(|d| d.top_k(max_keypoints));
                        (det, start.elapsed())
                    });
                    let (left_result, lt) = left_handle
                        .join()
                        .map_err(|_| InferenceError::ThreadPanic { stage: "sp_left" })?;
                    let (right_result, rt) = right_handle
                        .join()
                        .map_err(|_| InferenceError::ThreadPanic { stage: "sp_right" })?;
                    Ok::<_, PipelineError>((left_result?, lt, right_result?, rt))
                })?
            } else {
                // Sequential SP fallback
                let left_start = Instant::now();
                let left_det = self
                    .superpoint_left
                    .detect_with_downscale_limited(&left_frame, downscale, max_keypoints)?
                    .top_k(max_keypoints);
                let left_time = left_start.elapsed();

                let right_start = Instant::now();
                let right_det = self
                    .superpoint_left
                    .detect_with_downscale_limited(&right_frame, downscale, max_keypoints)?
                    .top_k(max_keypoints);
                let right_time = right_start.elapsed();

                (left_det, left_time, right_det, right_time)
            };

        let left = Arc::new(left_det);
        let right = Arc::new(right_det);

        let match_start = Instant::now();
        let matches = if left.is_empty() || right.is_empty() {
            Matches::new(left.clone(), right.clone(), Vec::new(), Vec::new())
                .map_err(InferenceError::Match)?
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

#[cfg(test)]
mod tests {
    use super::PipelineError;
    use crate::{FrameError, inference::InferenceError};
    use std::error::Error as _;

    #[test]
    fn pipeline_error_exposes_inference_and_domain_sources() {
        let error =
            PipelineError::Inference(InferenceError::Frame(FrameError::DimensionMismatch {
                expected: 4,
                actual: 3,
            }));

        let inference = error.source().expect("inference source");
        assert!(inference.to_string().contains("frame error"));
        let frame = inference.source().expect("frame source");
        assert_eq!(frame.to_string(), "dimension mismatch: expected 4, got 3");
        assert!(frame.source().is_none());
    }
}
