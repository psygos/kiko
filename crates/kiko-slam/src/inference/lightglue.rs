use super::{InferenceBackend, InferenceError, ManagedSession, build_session};
use crate::DESCRIPTOR_DIM;
use crate::Detections;
use crate::Matches;
use crate::Raw;
use ort::session::RunOptions;
use ort::value::{Shape, Tensor};
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
                .try_extract_tensor::<i64>()
                .map_err(|source| InferenceError::OutputDecode {
                    name: "matches0",
                    source,
                })?;
            let scores_raw = outputs
                .get("mscores0")
                .ok_or_else(|| InferenceError::UnexpectedOutput {
                    name: "mscores0".to_string(),
                    expected: "named output tensor".to_string(),
                    actual: "missing output".to_string(),
                })?
                .try_extract_tensor::<f32>()
                .map_err(|source| InferenceError::OutputDecode {
                    name: "mscores0",
                    source,
                })?;
            let parsed = parse_outputs(
                matches_raw.0,
                matches_raw.1,
                scores_raw.0,
                scores_raw.1,
                len_0,
            )?;

            drop(outputs);
            Matches::new(dec_1, dec_2, parsed.indices, parsed.scores)
                .map_err(InferenceError::Match)
        })
    }
}

#[derive(Debug, PartialEq)]
struct ParsedLightGlueOutput {
    indices: Vec<(usize, usize)>,
    scores: Vec<f32>,
}

fn parse_outputs(
    matches_shape: &Shape,
    matches_data: &[i64],
    scores_shape: &Shape,
    scores_data: &[f32],
    source_len: usize,
) -> Result<ParsedLightGlueOutput, InferenceError> {
    require_aligned_output("matches0", matches_shape, matches_data.len(), source_len)?;
    require_aligned_output("mscores0", scores_shape, scores_data.len(), source_len)?;

    let mut indices = Vec::new();
    for (source_index, &match_index) in matches_data.iter().enumerate() {
        if match_index == -1 {
            continue;
        }
        if match_index < -1 {
            return Err(InferenceError::UnexpectedOutput {
                name: "matches0".to_string(),
                expected: "-1 for unmatched detections or a non-negative match index".to_string(),
                actual: format!("index {source_index} contains {match_index}"),
            });
        }
        let matched_index =
            usize::try_from(match_index).map_err(|_| InferenceError::UnexpectedOutput {
                name: "matches0".to_string(),
                expected: "match indices representable by the host".to_string(),
                actual: format!("index {source_index} contains {match_index}"),
            })?;
        indices.push((source_index, matched_index));
    }

    for (source_index, &score) in scores_data.iter().enumerate() {
        if !score.is_finite() || score < 0.0 {
            return Err(InferenceError::UnexpectedOutput {
                name: "mscores0".to_string(),
                expected: "finite non-negative scores for every source detection".to_string(),
                actual: format!("index {source_index} contains {score}"),
            });
        }
    }
    let scores = indices
        .iter()
        .map(|&(source_index, _)| scores_data[source_index])
        .collect();
    Ok(ParsedLightGlueOutput { indices, scores })
}

fn require_aligned_output(
    name: &str,
    shape: &Shape,
    data_len: usize,
    source_len: usize,
) -> Result<(), InferenceError> {
    let aligned_shape = match &shape[..] {
        [1, count] => usize::try_from(*count).ok() == Some(source_len),
        _ => false,
    };
    if aligned_shape && data_len == source_len {
        return Ok(());
    }

    Err(InferenceError::UnexpectedOutput {
        name: name.to_string(),
        expected: format!("shape [1, {source_len}] with {source_len} elements"),
        actual: format!("shape {shape} with {data_len} elements"),
    })
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
    fn output_parser_requires_the_aligned_model_contract() {
        let matches_shape = Shape::new([1_i64, 4]);
        let scores_shape = Shape::new([1_i64, 4]);
        let parsed = parse_outputs(
            &matches_shape,
            &[-1, 2, 0, -1],
            &scores_shape,
            &[0.0, 0.75, 0.5, 0.0],
            4,
        )
        .expect("exact LightGlue output contract");

        assert_eq!(parsed.indices, vec![(1, 2), (2, 0)]);
        assert_eq!(parsed.scores, vec![0.75, 0.5]);
    }

    #[test]
    fn output_parser_rejects_wrong_rank_and_extra_sentinels() {
        let scores_shape = Shape::new([1_i64, 2]);
        for (shape, data) in [
            (Shape::new([2_i64]), vec![-1, -1]),
            (Shape::new([1_i64, 3]), vec![-1, -1, -1]),
        ] {
            let error = parse_outputs(&shape, &data, &scores_shape, &[0.0, 0.0], 2)
                .expect_err("malformed matches0 shape");
            assert!(matches!(
                error,
                InferenceError::UnexpectedOutput { ref name, .. } if name == "matches0"
            ));
        }
    }

    #[test]
    fn output_parser_requires_one_score_per_source_even_if_all_are_unmatched() {
        let matches_shape = Shape::new([1_i64, 2]);
        let scores_shape = Shape::new([1_i64, 0]);
        let error = parse_outputs(&matches_shape, &[-1, -1], &scores_shape, &[], 2)
            .expect_err("missing aligned scores");

        assert!(matches!(
            error,
            InferenceError::UnexpectedOutput { ref name, .. } if name == "mscores0"
        ));
    }

    #[test]
    fn output_parser_rejects_undocumented_negative_match_sentinels() {
        let shape = Shape::new([1_i64, 1]);
        for sentinel in [-2, i64::MIN] {
            let error = parse_outputs(&shape, &[sentinel], &shape, &[0.0], 1)
                .expect_err("only -1 is the unmatched sentinel");

            assert!(matches!(
                error,
                InferenceError::UnexpectedOutput {
                    ref name,
                    ref actual,
                    ..
                } if name == "matches0" && actual.contains(&sentinel.to_string())
            ));
        }
    }

    #[test]
    fn output_parser_rejects_invalid_unmatched_scores() {
        let shape = Shape::new([1_i64, 1]);
        for score in [f32::NAN, -f32::from_bits(1)] {
            let error = parse_outputs(&shape, &[-1], &shape, &[score], 1)
                .expect_err("every model score must be finite and non-negative");

            assert!(matches!(
                error,
                InferenceError::UnexpectedOutput { ref name, .. } if name == "mscores0"
            ));
        }
    }

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
