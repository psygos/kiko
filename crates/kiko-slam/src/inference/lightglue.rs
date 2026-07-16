use super::{InferenceBackend, InferenceError, ManagedSession, build_session};
use crate::DESCRIPTOR_DIM;
use crate::Detections;
use crate::Matches;
use crate::Raw;
use ort::session::RunOptions;
use ort::value::Tensor;
use std::path::Path;
use std::sync::Arc;

pub struct LightGlue {
    session: ManagedSession,
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
        let desc_0 = dec_1.descriptors_flat().to_vec();
        let desc_1 = dec_2.descriptors_flat().to_vec();

        let len_0 = dec_1.len();
        let len_1 = dec_2.len();
        let kpts_0_tensor = Tensor::from_array(([1, len_0, 2], kpts_0))?;
        let kpts_1_tensor = Tensor::from_array(([1, len_1, 2], kpts_1))?;
        let desc_0_tensor = Tensor::from_array(([1, len_0, DESCRIPTOR_DIM], desc_0))?;
        let desc_1_tensor = Tensor::from_array(([1, len_1, DESCRIPTOR_DIM], desc_1))?;

        self.session.run("lightglue", |session| {
            let run_options = RunOptions::new().map_err(InferenceError::Execution)?;
            let outputs = super::run_with_watchdog("lightglue", || {
                session.run_async(
                    ort::inputs!["kpts0" => kpts_0_tensor, "kpts1" => kpts_1_tensor, "desc0" => desc_0_tensor, "desc1" => desc_1_tensor],
                    &run_options,
                )
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

            drop(outputs);
            Matches::new(dec_1, dec_2, indices, scores).map_err(InferenceError::Match)
        })
    }
}

fn normalize_keypoints(detections: &Detections) -> Vec<f32> {
    let dimensions = detections.dimensions();
    let width = f64::from(dimensions.width());
    let height = f64::from(dimensions.height());
    let scale = 0.5 * width.max(height);
    let cx = width * 0.5;
    let cy = height * 0.5;

    let mut out = Vec::with_capacity(detections.len() * 2);
    for kp in detections.keypoints() {
        // Exact normalized coordinates are in [-1, 1). Final f32 rounding can
        // reach 1.0, but the model inputs remain finite and within [-1, 1].
        out.push(((f64::from(kp.x) - cx) / scale) as f32);
        out.push(((f64::from(kp.y) - cy) / scale) as f32);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DESCRIPTOR_DIM, Descriptor, FrameId, Keypoint, SensorId};

    #[test]
    fn normalization_does_not_round_large_dimensions_before_arithmetic() {
        const WIDTH: u32 = 16_777_217;
        const X: f32 = 16_777_216.0;
        let detections = Detections::new(
            SensorId::StereoLeft,
            FrameId::new(1),
            WIDTH,
            1,
            vec![Keypoint { x: X, y: 0.0 }],
            vec![1.0],
            vec![Descriptor([0.0; DESCRIPTOR_DIM])],
        )
        .expect("large exact detection domain");

        let normalized = normalize_keypoints(&detections);
        let rounded_width = WIDTH as f32;
        let legacy_x = (X - rounded_width * 0.5) / (rounded_width * 0.5);

        assert_eq!(legacy_x, 1.0);
        assert_eq!(normalized[0].to_bits(), 0x3f7f_fffe);
        assert!(normalized[0] < 1.0);
    }

    #[test]
    fn normalization_documents_final_rounding_at_positive_endpoint() {
        const WIDTH: u32 = 67_108_865;
        const X: f32 = 67_108_864.0;
        let detections = Detections::new(
            SensorId::StereoLeft,
            FrameId::new(1),
            WIDTH,
            1,
            vec![Keypoint { x: X, y: 0.0 }],
            vec![1.0],
            vec![Descriptor([0.0; DESCRIPTOR_DIM])],
        )
        .expect("large exact detection domain");

        let exact = (f64::from(X) - f64::from(WIDTH) * 0.5) / (f64::from(WIDTH) * 0.5);
        let normalized = normalize_keypoints(&detections);

        assert!(exact < 1.0);
        assert_eq!(normalized[0], 1.0);
        assert!(normalized[0].is_finite());
    }
}
