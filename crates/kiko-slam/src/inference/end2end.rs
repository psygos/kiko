use super::{InferenceBackend, InferenceError, InferenceRunDiagnostics, build_session};
use crate::{Descriptor, Detections, Frame, FrameDimensions, Keypoint, Matches, Raw};
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
    diagnostics: InferenceRunDiagnostics,
    scratch: Vec<f32>,
}

pub struct End2EndTimings {
    pub total: Duration,
}

impl End2EndPipeline {
    pub fn new(path: impl AsRef<Path>, backend: InferenceBackend) -> Result<Self, InferenceError> {
        let path = path.as_ref();
        let (session, selected, diagnostics) = build_session(path, backend)?;
        Ok(Self {
            session,
            backend: selected,
            diagnostics,
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
        let dimensions = left.dimensions();
        let batch_elements = stereo_batch_element_count(dimensions, right.dimensions())?;
        let w = dimensions.width() as usize;
        let h = dimensions.height() as usize;
        let pixels = dimensions.area();

        // Prepare batch input: [2, 1, H, W] — both images concatenated
        self.scratch.resize(batch_elements, 0.0);
        crate::preprocess::normalise_into(left.data(), &mut self.scratch[..pixels])?;
        crate::preprocess::normalise_into(right.data(), &mut self.scratch[pixels..])?;

        let input_tensor = TensorRef::from_array_view(([2, 1, h, w], self.scratch.as_slice()))
            .map_err(|source| InferenceError::InputTensor {
                name: "images",
                source,
            })?;

        let start = Instant::now();
        let outputs = super::run_with_slow_call_diagnostics(self.diagnostics, "pipeline", || {
            self.session
                .run(ort::inputs!["images" => input_tensor])
                .map_err(|source| InferenceError::SessionRun {
                    model: "end-to-end-superpoint-lightglue",
                    source,
                })
        })?;
        let total = start.elapsed();

        let parsed = if outputs.get("keypoints0").is_some() {
            parse_stereo_fused_outputs(&outputs, max_keypoints)?
        } else {
            parse_legacy_pipeline_outputs(&outputs, max_keypoints)?
        };

        // Build detections with empty descriptors — matching is done internally by the pipeline
        let left_scores = vec![1.0_f32; parsed.left_keypoints.len()];
        let left_descs = vec![Descriptor([0.0; DESCRIPTOR_DIM]); parsed.left_keypoints.len()];
        let left_det = Detections::new(
            left.sensor_id(),
            left.frame_id(),
            left.width(),
            left.height(),
            parsed.left_keypoints,
            left_scores,
            left_descs,
        )
        .map_err(InferenceError::Detection)?;

        let right_scores = vec![1.0_f32; parsed.right_keypoints.len()];
        let right_descs = vec![Descriptor([0.0; DESCRIPTOR_DIM]); parsed.right_keypoints.len()];
        let right_det = Detections::new(
            right.sensor_id(),
            right.frame_id(),
            right.width(),
            right.height(),
            parsed.right_keypoints,
            right_scores,
            right_descs,
        )
        .map_err(InferenceError::Detection)?;

        let left_det = Arc::new(left_det);
        let right_det = Arc::new(right_det);

        let matches = Matches::new(
            left_det,
            right_det,
            parsed.match_indices,
            parsed.match_scores,
        )
        .map_err(InferenceError::Match)?;

        Ok((matches, End2EndTimings { total }))
    }
}

fn stereo_batch_element_count(
    left: FrameDimensions,
    right: FrameDimensions,
) -> Result<usize, InferenceError> {
    if left != right {
        return Err(InferenceError::StereoInputDimensionsMismatch { left, right });
    }
    left.area()
        .checked_mul(2)
        .ok_or(InferenceError::InputBatchSizeOverflow {
            dimensions: left,
            batch_size: 2,
        })
}

struct ParsedPipelineOutputs {
    left_keypoints: Vec<Keypoint>,
    right_keypoints: Vec<Keypoint>,
    match_indices: Vec<(usize, usize)>,
    match_scores: Vec<f32>,
}

fn parse_stereo_fused_outputs(
    outputs: &ort::session::SessionOutputs<'_>,
    max_keypoints: usize,
) -> Result<ParsedPipelineOutputs, InferenceError> {
    let left_value = outputs
        .get("keypoints0")
        .ok_or_else(|| missing_output("keypoints0"))?;
    let right_value = outputs
        .get("keypoints1")
        .ok_or_else(|| missing_output("keypoints1"))?;
    let matches_value = outputs
        .get("matches0")
        .ok_or_else(|| missing_output("matches0"))?;
    let scores_value = outputs
        .get("mscores0")
        .ok_or_else(|| missing_output("mscores0"))?;
    let left_data = super::extract_tensor::<i64>(left_value, "keypoints0")?.1;
    let right_data = super::extract_tensor::<i64>(right_value, "keypoints1")?.1;
    let matches_data = super::extract_tensor::<i64>(matches_value, "matches0")?.1;
    let scores_data = super::extract_tensor::<f32>(scores_value, "mscores0")?.1;

    let left_keypoints = parse_keypoints_i64_xy(left_data, max_keypoints, "keypoints0")?;
    let right_keypoints = parse_keypoints_i64_xy(right_data, max_keypoints, "keypoints1")?;
    let left_output_count = left_data.len() / 2;
    let right_output_count = right_data.len() / 2;
    let match_count = super::output_record_count("matches0", matches_data.len(), 2)?;
    super::require_output_elements("mscores0", scores_data.len(), match_count)?;
    let mut match_indices = Vec::with_capacity(match_count);
    let mut match_scores = Vec::with_capacity(match_count);
    for i in 0..match_count {
        let left_idx =
            usize::try_from(matches_data[2 * i]).map_err(|_| negative_index("matches0"))?;
        let right_idx =
            usize::try_from(matches_data[2 * i + 1]).map_err(|_| negative_index("matches0"))?;
        require_index("matches0", left_idx, left_output_count)?;
        require_index("matches0", right_idx, right_output_count)?;
        if left_idx < left_keypoints.len() && right_idx < right_keypoints.len() {
            match_indices.push((left_idx, right_idx));
            match_scores.push(scores_data[i]);
        }
    }

    Ok(ParsedPipelineOutputs {
        left_keypoints,
        right_keypoints,
        match_indices,
        match_scores,
    })
}

fn parse_legacy_pipeline_outputs(
    outputs: &ort::session::SessionOutputs<'_>,
    max_keypoints: usize,
) -> Result<ParsedPipelineOutputs, InferenceError> {
    let keypoints_value = outputs
        .get("keypoints")
        .ok_or_else(|| missing_output("keypoints"))?;
    let matches_value = outputs
        .get("matches")
        .ok_or_else(|| missing_output("matches"))?;
    let scores_value = outputs
        .get("mscores")
        .ok_or_else(|| missing_output("mscores"))?;
    let kpts_data = super::extract_tensor::<i64>(keypoints_value, "keypoints")?.1;
    let matches_data = super::extract_tensor::<i64>(matches_value, "matches")?.1;
    let scores_data = super::extract_tensor::<f32>(scores_value, "mscores")?.1;

    let kpts_per_image = super::output_record_count("keypoints", kpts_data.len(), 4)?;
    let left_data = &kpts_data[..kpts_per_image * 2];
    let right_data = &kpts_data[kpts_per_image * 2..];
    let left_keypoints = parse_keypoints_i64_xy(left_data, max_keypoints, "keypoints")?;
    let right_keypoints = parse_keypoints_i64_xy(right_data, max_keypoints, "keypoints")?;

    let match_count = super::output_record_count("matches", matches_data.len(), 3)?;
    super::require_output_elements("mscores", scores_data.len(), match_count)?;
    let mut match_indices = Vec::with_capacity(match_count);
    let mut match_scores = Vec::with_capacity(match_count);
    for i in 0..match_count {
        let left_idx =
            usize::try_from(matches_data[i * 3 + 1]).map_err(|_| negative_index("matches"))?;
        let right_idx =
            usize::try_from(matches_data[i * 3 + 2]).map_err(|_| negative_index("matches"))?;
        require_index("matches", left_idx, kpts_per_image)?;
        require_index("matches", right_idx, kpts_per_image)?;
        if left_idx < left_keypoints.len() && right_idx < right_keypoints.len() {
            match_indices.push((left_idx, right_idx));
            match_scores.push(scores_data[i]);
        }
    }

    Ok(ParsedPipelineOutputs {
        left_keypoints,
        right_keypoints,
        match_indices,
        match_scores,
    })
}

fn parse_keypoints_i64_xy(
    data: &[i64],
    max_keypoints: usize,
    output_name: &str,
) -> Result<Vec<Keypoint>, InferenceError> {
    if data.len() % 2 != 0 {
        return Err(InferenceError::UnexpectedOutput {
            name: output_name.to_string(),
            expected: "xy keypoint pairs".to_string(),
            actual: format!("{} scalar values", data.len()),
        });
    }
    let count = (data.len() / 2).min(max_keypoints);
    let mut keypoints = Vec::with_capacity(count);
    for i in 0..count {
        keypoints.push(Keypoint {
            x: super::exact_i64_output_f32(output_name, 2 * i, data[2 * i])?,
            y: super::exact_i64_output_f32(output_name, 2 * i + 1, data[2 * i + 1])?,
        });
    }
    Ok(keypoints)
}

fn missing_output(name: &str) -> InferenceError {
    InferenceError::UnexpectedOutput {
        name: name.to_string(),
        expected: "named output tensor".to_string(),
        actual: "missing output".to_string(),
    }
}

fn negative_index(name: &str) -> InferenceError {
    InferenceError::UnexpectedOutput {
        name: name.to_string(),
        expected: "non-negative index".to_string(),
        actual: "negative index".to_string(),
    }
}

fn require_index(name: &str, index: usize, available: usize) -> Result<(), InferenceError> {
    if index >= available {
        return Err(InferenceError::UnexpectedOutput {
            name: name.to_string(),
            expected: format!("indices below {available}"),
            actual: format!("index {index}"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::stereo_batch_element_count;
    use crate::{FrameDimensions, InferenceError};

    #[test]
    fn stereo_batch_layout_rejects_dimension_mismatch_before_buffer_mutation() {
        let left = FrameDimensions::try_new(640, 480).expect("left dimensions");
        let right = FrameDimensions::try_new(320, 240).expect("right dimensions");
        assert!(matches!(
            stereo_batch_element_count(left, right),
            Err(InferenceError::StereoInputDimensionsMismatch {
                left: actual_left,
                right: actual_right,
            }) if actual_left == left && actual_right == right
        ));
    }

    #[test]
    fn stereo_batch_layout_uses_checked_capacity_arithmetic() {
        let dimensions = FrameDimensions::try_new(640, 480).expect("dimensions");
        assert_eq!(
            stereo_batch_element_count(dimensions, dimensions).expect("batch elements"),
            2 * 640 * 480
        );

        #[cfg(target_pointer_width = "64")]
        {
            let huge = FrameDimensions::try_new(u32::MAX, u32::MAX).expect("64-bit dimensions");
            assert!(matches!(
                stereo_batch_element_count(huge, huge),
                Err(InferenceError::InputBatchSizeOverflow {
                    dimensions,
                    batch_size: 2,
                }) if dimensions == huge
            ));
        }
    }
}
