use super::{InferenceBackend, InferenceError, InferenceRunDiagnostics, build_session};
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
        let w = left.width() as usize;
        let h = left.height() as usize;
        let pixels = w * h;

        // Prepare batch input: [2, 1, H, W] — both images concatenated
        self.scratch.resize(2 * pixels, 0.0);
        crate::preprocess::normalise_into(left.data(), &mut self.scratch[..pixels])?;
        crate::preprocess::normalise_into(right.data(), &mut self.scratch[pixels..])?;

        let input_tensor = TensorRef::from_array_view(([2, 1, h, w], self.scratch.as_slice()))?;

        let start = Instant::now();
        let outputs = super::run_with_slow_call_diagnostics(self.diagnostics, "pipeline", || {
            self.session
                .run(ort::inputs!["images" => input_tensor])
                .map_err(InferenceError::Execution)
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
    let left_data = outputs
        .get("keypoints0")
        .ok_or_else(|| missing_output("keypoints0"))?
        .try_extract_tensor::<i64>()?
        .1
        .to_vec();
    let right_data = outputs
        .get("keypoints1")
        .ok_or_else(|| missing_output("keypoints1"))?
        .try_extract_tensor::<i64>()?
        .1
        .to_vec();
    let matches_data = outputs
        .get("matches0")
        .ok_or_else(|| missing_output("matches0"))?
        .try_extract_tensor::<i64>()?
        .1
        .to_vec();
    let scores_data = outputs
        .get("mscores0")
        .ok_or_else(|| missing_output("mscores0"))?
        .try_extract_tensor::<f32>()?
        .1
        .to_vec();

    let left_keypoints = parse_keypoints_i64_xy(&left_data, max_keypoints, "keypoints0")?;
    let right_keypoints = parse_keypoints_i64_xy(&right_data, max_keypoints, "keypoints1")?;
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
    let kpts_data = outputs
        .get("keypoints")
        .ok_or_else(|| missing_output("keypoints"))?
        .try_extract_tensor::<i64>()?
        .1
        .to_vec();
    let matches_data = outputs
        .get("matches")
        .ok_or_else(|| missing_output("matches"))?
        .try_extract_tensor::<i64>()?
        .1
        .to_vec();
    let scores_data = outputs
        .get("mscores")
        .ok_or_else(|| missing_output("mscores"))?
        .try_extract_tensor::<f32>()?
        .1
        .to_vec();

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
            x: data[2 * i] as f32,
            y: data[2 * i + 1] as f32,
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
