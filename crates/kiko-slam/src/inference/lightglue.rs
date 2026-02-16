use super::{InferenceBackend, InferenceError, build_session};
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
        let kpts_0 = normalize_keypoints(&dec_1);
        let kpts_1 = normalize_keypoints(&dec_2);
        let desc_0 = dec_1.descriptors_flat();
        let desc_1 = dec_2.descriptors_flat();

        let kpts_0_tensor = TensorRef::from_array_view(([1, dec_1.len(), 2], kpts_0.as_slice()))?;
        let kpts_1_tensor = TensorRef::from_array_view(([1, dec_2.len(), 2], kpts_1.as_slice()))?;
        let desc_0_tensor = TensorRef::from_array_view(([1, dec_1.len(), 256], desc_0))?;
        let desc_1_tensor = TensorRef::from_array_view(([1, dec_2.len(), 256], desc_1))?;

        let outputs = self.session.run(ort::inputs!["kpts0" => kpts_0_tensor, "kpts1" => kpts_1_tensor, "desc0" => desc_0_tensor, "desc1" => desc_1_tensor])?;
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
        let matches_data = matches_raw.1;
        let scores_data = scores_raw.1;

        let mut indices = Vec::new();
        let mut scores = Vec::new();

        for (i, &match_idx) in matches_data.iter().enumerate() {
            if match_idx < 0 {
                continue;
            }
            let Some(&score) = scores_data.get(i) else {
                return Err(InferenceError::UnexpectedOutput {
                    name: "mscores0".to_string(),
                    expected: format!("at least {} elements", matches_data.len()),
                    actual: format!("{} elements", scores_data.len()),
                });
            };
            let right_idx =
                usize::try_from(match_idx).map_err(|_| InferenceError::UnexpectedOutput {
                    name: "matches0".to_string(),
                    expected: "non-negative match indices".to_string(),
                    actual: format!("index {match_idx}"),
                })?;
            indices.push((i, right_idx));
            scores.push(score);
        }

        Matches::new(dec_1, dec_2, indices, scores)
            .map_err(|e| InferenceError::Domain(format!("{e:?}")))
    }
}

fn normalize_keypoints(detections: &Detections) -> Vec<f32> {
    let width = detections.width() as f32;
    let height = detections.height() as f32;
    let scale = 0.5 * width.max(height);
    let cx = width * 0.5;
    let cy = height * 0.5;

    let mut out = Vec::with_capacity(detections.len() * 2);
    for kp in detections.keypoints() {
        out.push((kp.x - cx) / scale);
        out.push((kp.y - cy) / scale);
    }
    out
}
