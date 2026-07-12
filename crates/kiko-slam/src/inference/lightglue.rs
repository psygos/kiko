use super::{InferenceBackend, InferenceError, build_session};
use crate::DESCRIPTOR_DIM;
use crate::Detections;
use crate::Matches;
use crate::Raw;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::Path;
use std::sync::Arc;

pub struct LightGlue {
    session: Session,
    backend: InferenceBackend,
    keypoints_0: Vec<f32>,
    keypoints_1: Vec<f32>,
}

impl LightGlue {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, InferenceError> {
        Self::new_with_backend(path, InferenceBackend::auto())
    }

    pub fn new_with_backend(
        path: impl AsRef<Path>,
        backend: InferenceBackend,
    ) -> Result<Self, InferenceError> {
        let path = path.as_ref();
        let (session, selected) = build_session(path, backend)?;

        Ok(Self {
            session,
            backend: selected,
            keypoints_0: Vec::new(),
            keypoints_1: Vec::new(),
        })
    }

    pub fn backend(&self) -> InferenceBackend {
        self.backend
    }

    pub fn match_these(
        &mut self,
        dec_1: Arc<Detections>,
        dec_2: Arc<Detections>,
    ) -> Result<Matches<Raw>, InferenceError> {
        normalize_keypoints_into(&dec_1, &mut self.keypoints_0);
        normalize_keypoints_into(&dec_2, &mut self.keypoints_1);
        let desc_0 = dec_1.descriptors_flat();
        let desc_1 = dec_2.descriptors_flat();

        let kpts_0_tensor =
            TensorRef::from_array_view(([1, dec_1.len(), 2], self.keypoints_0.as_slice()))?;
        let kpts_1_tensor =
            TensorRef::from_array_view(([1, dec_2.len(), 2], self.keypoints_1.as_slice()))?;
        let desc_0_tensor = TensorRef::from_array_view(([1, dec_1.len(), DESCRIPTOR_DIM], desc_0))?;
        let desc_1_tensor = TensorRef::from_array_view(([1, dec_2.len(), DESCRIPTOR_DIM], desc_1))?;

        let mut indices = Vec::new();
        let mut scores = Vec::new();
        {
            let outputs = super::run_with_slow_call_diagnostics("lightglue", || {
                self.session
                    .run(ort::inputs!["kpts0" => kpts_0_tensor, "kpts1" => kpts_1_tensor, "desc0" => desc_0_tensor, "desc1" => desc_1_tensor])
                    .map_err(InferenceError::Execution)
            })?;
            let matches_raw = outputs
                .get("matches0")
                .ok_or_else(|| InferenceError::UnexpectedOutput {
                    name: "matches0".to_string(),
                    expected: "named output tensor".to_string(),
                    actual: "missing output".to_string(),
                })?
                .try_extract_tensor::<i64>()?;
            let scores_raw = outputs
                .get("mscores0")
                .ok_or_else(|| InferenceError::UnexpectedOutput {
                    name: "mscores0".to_string(),
                    expected: "named output tensor".to_string(),
                    actual: "missing output".to_string(),
                })?
                .try_extract_tensor::<f32>()?;
            let matches_shape = matches_raw.0;
            let matches_data = matches_raw.1;
            let scores_data = scores_raw.1;

            let is_pair_format =
                matches_shape.len() >= 2 && matches_shape[matches_shape.len() - 1] == 2;

            if is_pair_format {
                // Fused/TRT format: matches0 is [N, 2] with (left_idx, right_idx) pairs
                let num_pairs = super::output_record_count("matches0", matches_data.len(), 2)?;
                super::require_output_elements("mscores0", scores_data.len(), num_pairs)?;
                for i in 0..num_pairs {
                    let left_idx = usize::try_from(matches_data[2 * i]).map_err(|_| {
                        InferenceError::UnexpectedOutput {
                            name: "matches0".to_string(),
                            expected: "non-negative index".to_string(),
                            actual: format!("index {}", matches_data[2 * i]),
                        }
                    })?;
                    let right_idx = usize::try_from(matches_data[2 * i + 1]).map_err(|_| {
                        InferenceError::UnexpectedOutput {
                            name: "matches0".to_string(),
                            expected: "non-negative index".to_string(),
                            actual: format!("index {}", matches_data[2 * i + 1]),
                        }
                    })?;
                    indices.push((left_idx, right_idx));
                    scores.push(scores_data[i]);
                }
            } else {
                // Standard format: matches0 is [1, num_keypoints] with per-keypoint match index
                super::require_output_elements("mscores0", scores_data.len(), matches_data.len())?;
                for (i, &match_idx) in matches_data.iter().enumerate() {
                    if match_idx < 0 {
                        continue;
                    }
                    let right_idx = usize::try_from(match_idx).map_err(|_| {
                        InferenceError::UnexpectedOutput {
                            name: "matches0".to_string(),
                            expected: "non-negative match indices".to_string(),
                            actual: format!("index {match_idx}"),
                        }
                    })?;
                    indices.push((i, right_idx));
                    scores.push(scores_data[i]);
                }
            }
        }

        Matches::new(dec_1, dec_2, indices, scores).map_err(InferenceError::Match)
    }
}

fn normalize_keypoints_into(detections: &Detections, out: &mut Vec<f32>) {
    let width = detections.width() as f32;
    let height = detections.height() as f32;
    let scale = 0.5 * width.max(height);
    let cx = width * 0.5;
    let cy = height * 0.5;

    out.clear();
    out.reserve(detections.len() * 2);
    for kp in detections.keypoints() {
        out.push((kp.x - cx) / scale);
        out.push((kp.y - cy) / scale);
    }
}
