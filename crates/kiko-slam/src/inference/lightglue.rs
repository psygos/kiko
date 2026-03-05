use super::{build_session, InferenceBackend, InferenceError};
use crate::Detections;
use crate::Matches;
use crate::Raw;
use crate::DESCRIPTOR_DIM;
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

        let outputs = super::run_with_watchdog("lightglue", || {
            self.session
                .run(ort::inputs!["kpts0" => kpts_0_tensor, "kpts1" => kpts_1_tensor, "desc0" => desc_0_tensor, "desc1" => desc_1_tensor])
                .map_err(InferenceError::Execution)
        })?;
        let parsed = parse_match_outputs(&outputs)?;

        let mut indices = Vec::new();
        let mut scores = Vec::new();

        for (i, &match_idx) in parsed.matches0.iter().enumerate() {
            if match_idx < 0 {
                continue;
            }
            let Some(&score) = parsed.mscores0.get(i) else {
                return Err(InferenceError::UnexpectedOutput {
                    name: "mscores0".to_string(),
                    expected: format!("at least {} elements", parsed.matches0.len()),
                    actual: format!("{} elements", parsed.mscores0.len()),
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

        Matches::new(dec_1, dec_2, indices, scores).map_err(InferenceError::Match)
    }
}

struct ParsedMatchOutputs<'a> {
    matches0: &'a [i64],
    mscores0: &'a [f32],
}

fn parse_match_outputs<'a>(
    outputs: &'a ort::session::SessionOutputs<'a>,
) -> Result<ParsedMatchOutputs<'a>, InferenceError> {
    let matches0 = outputs
        .get("matches0")
        .ok_or_else(|| InferenceError::UnexpectedOutput {
            name: "matches0".to_string(),
            expected: "named output tensor".to_string(),
            actual: "missing output".to_string(),
        })?
        .try_extract_tensor::<i64>()?
        .1;
    let mscores0 = outputs
        .get("mscores0")
        .ok_or_else(|| InferenceError::UnexpectedOutput {
            name: "mscores0".to_string(),
            expected: "named output tensor".to_string(),
            actual: "missing output".to_string(),
        })?
        .try_extract_tensor::<f32>()?
        .1;
    Ok(ParsedMatchOutputs { matches0, mscores0 })
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
