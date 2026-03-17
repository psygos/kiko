use super::{InferenceBackend, InferenceError, build_session};
use crate::{Descriptor, Detections, Frame, Keypoint, Matches, Raw};
use ort::session::Session;
use ort::value::TensorRef;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::DESCRIPTOR_DIM;

/// End-to-end pipeline model that takes two grayscale images (batch=2)
/// and outputs keypoints, matches, and match scores in a single call.
pub struct End2EndPipeline {
    session: Session,
    backend: InferenceBackend,
    scratch: Vec<f32>,
}

pub struct End2EndTimings {
    pub total: Duration,
}

impl End2EndPipeline {
    pub fn new(path: impl AsRef<Path>, backend: InferenceBackend) -> Result<Self, InferenceError> {
        let path = path.as_ref();
        let (session, selected) = build_session(path, backend)?;
        Ok(Self {
            session,
            backend: selected,
            scratch: Vec::new(),
        })
    }

    pub fn backend(&self) -> InferenceBackend {
        self.backend
    }

    /// Run the full SP+LG pipeline on a stereo pair.
    /// Both images are batched as [2, 1, H, W] in a single session.run().
    pub fn match_pair(
        &mut self,
        left: &Frame,
        right: &Frame,
        max_keypoints: usize,
    ) -> Result<(Matches<Raw>, End2EndTimings), InferenceError> {
        let w = left.width() as usize;
        let h = left.height() as usize;
        let pixels = w * h;

        // Prepare batch input: [2, 1, H, W] — both images concatenated
        self.scratch.resize(2 * pixels, 0.0);
        crate::preprocess::normalise_into(left.data(), &mut self.scratch[..pixels])?;
        crate::preprocess::normalise_into(right.data(), &mut self.scratch[pixels..])?;

        let input_tensor = TensorRef::from_array_view(([2, 1, h, w], self.scratch.as_slice()))?;

        let start = Instant::now();
        let outputs = super::run_with_watchdog("pipeline", || {
            self.session
                .run(ort::inputs!["images" => input_tensor])
                .map_err(InferenceError::Execution)
        })?;
        let total = start.elapsed();

        // Parse keypoints: [batch_size, max_keypoints, 2] dtype=i64
        let kpts_value =
            outputs
                .get("keypoints")
                .ok_or_else(|| InferenceError::UnexpectedOutput {
                    name: "keypoints".to_string(),
                    expected: "named output tensor".to_string(),
                    actual: "missing output".to_string(),
                })?;
        let kpts_data = kpts_value.try_extract_tensor::<i64>()?.1.to_vec();

        // Parse matches: [N, 3] dtype=i64 — (batch_idx, left_idx, right_idx)
        let matches_value =
            outputs
                .get("matches")
                .ok_or_else(|| InferenceError::UnexpectedOutput {
                    name: "matches".to_string(),
                    expected: "named output tensor".to_string(),
                    actual: "missing output".to_string(),
                })?;
        let matches_data = matches_value.try_extract_tensor::<i64>()?.1.to_vec();

        // Parse scores: [N] dtype=f32
        let scores_value =
            outputs
                .get("mscores")
                .ok_or_else(|| InferenceError::UnexpectedOutput {
                    name: "mscores".to_string(),
                    expected: "named output tensor".to_string(),
                    actual: "missing output".to_string(),
                })?;
        let scores_data = scores_value.try_extract_tensor::<f32>()?.1.to_vec();

        // Build keypoints for left (batch=0) and right (batch=1)
        let max_kp = max_keypoints.min(kpts_data.len() / 4); // [2, N, 2] flattened
        let kpts_per_image = kpts_data.len() / 4; // total / (batch=2 * xy=2)

        let mut left_kps = Vec::with_capacity(kpts_per_image);
        let mut right_kps = Vec::with_capacity(kpts_per_image);
        for i in 0..kpts_per_image.min(max_keypoints) {
            left_kps.push(Keypoint {
                x: kpts_data[i * 2] as f32,
                y: kpts_data[i * 2 + 1] as f32,
            });
        }
        let right_offset = kpts_per_image * 2;
        for i in 0..kpts_per_image.min(max_keypoints) {
            right_kps.push(Keypoint {
                x: kpts_data[right_offset + i * 2] as f32,
                y: kpts_data[right_offset + i * 2 + 1] as f32,
            });
        }

        // Build detections with empty descriptors — matching is done internally by the pipeline
        let left_scores = vec![1.0_f32; left_kps.len()];
        let left_descs = vec![Descriptor([0.0; DESCRIPTOR_DIM]); left_kps.len()];
        let left_det = Detections::new(
            left.sensor_id(),
            left.frame_id(),
            left.width(),
            left.height(),
            left_kps,
            left_scores,
            left_descs,
        )
        .map_err(InferenceError::Detection)?;

        let right_scores = vec![1.0_f32; right_kps.len()];
        let right_descs = vec![Descriptor([0.0; DESCRIPTOR_DIM]); right_kps.len()];
        let right_det = Detections::new(
            right.sensor_id(),
            right.frame_id(),
            right.width(),
            right.height(),
            right_kps,
            right_scores,
            right_descs,
        )
        .map_err(InferenceError::Detection)?;

        let left_det = Arc::new(left_det);
        let right_det = Arc::new(right_det);

        // Parse match pairs: [N, 3] = (batch_idx, left_idx, right_idx)
        // For stereo (batch=0 matched against batch=1), batch_idx is 0
        let num_matches = matches_data.len() / 3;
        let mut indices = Vec::with_capacity(num_matches);
        let mut match_scores = Vec::with_capacity(num_matches);
        for i in 0..num_matches {
            let left_idx = matches_data[i * 3 + 1] as usize;
            let right_idx = matches_data[i * 3 + 2] as usize;
            let score = scores_data.get(i).copied().unwrap_or(1.0);
            indices.push((left_idx, right_idx));
            match_scores.push(score);
        }

        let matches = Matches::new(left_det, right_det, indices, match_scores)
            .map_err(InferenceError::Match)?;

        Ok((matches, End2EndTimings { total }))
    }
}
