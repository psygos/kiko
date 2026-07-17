use super::{
    InferenceBackend, InferenceError, InferenceRunDiagnostics, build_run_options, build_session,
};
use crate::{DESCRIPTOR_DIM, Detections, Matches, Raw};
use ort::session::{RunOptions, Session};
use ort::value::{Outlet, Shape, TensorElementType, TensorRef, ValueType};
use std::path::Path;
use std::sync::Arc;

pub struct LightGlue {
    session: Session,
    run_options: RunOptions,
    backend: InferenceBackend,
    diagnostics: InferenceRunDiagnostics,
    output_contract: LightGlueOutputContract,
    keypoints_0: Vec<f32>,
    keypoints_1: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LightGlueOutputContract {
    /// Standard LightGlue emits one match index and score per source keypoint.
    StandardAligned,
    /// Kiko's Jetson export emits compact `(source, target)` pairs and one score per pair.
    FusedPairs,
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
        let (session, selected, diagnostics) = build_session(path, backend)?;
        let output_contract = parse_lightglue_output_contract(session.outputs())?;
        let run_options = build_run_options(selected)?;

        Ok(Self {
            session,
            run_options,
            backend: selected,
            diagnostics,
            output_contract,
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
            TensorRef::from_array_view(([1, dec_1.len(), 2], self.keypoints_0.as_slice()))
                .map_err(|source| InferenceError::InputTensor {
                    name: "kpts0",
                    source,
                })?;
        let kpts_1_tensor =
            TensorRef::from_array_view(([1, dec_2.len(), 2], self.keypoints_1.as_slice()))
                .map_err(|source| InferenceError::InputTensor {
                    name: "kpts1",
                    source,
                })?;
        let desc_0_tensor = TensorRef::from_array_view(([1, dec_1.len(), DESCRIPTOR_DIM], desc_0))
            .map_err(|source| InferenceError::InputTensor {
                name: "desc0",
                source,
            })?;
        let desc_1_tensor = TensorRef::from_array_view(([1, dec_2.len(), DESCRIPTOR_DIM], desc_1))
            .map_err(|source| InferenceError::InputTensor {
                name: "desc1",
                source,
            })?;

        let parsed = {
            let outputs = super::run_with_slow_call_diagnostics(
                self.diagnostics,
                "lightglue",
                || {
                    self.session
                    .run_with_options(ort::inputs!["kpts0" => kpts_0_tensor, "kpts1" => kpts_1_tensor, "desc0" => desc_0_tensor, "desc1" => desc_1_tensor], &self.run_options)
                    .map_err(|source| InferenceError::SessionRun {
                        model: "lightglue",
                        source,
                    })
                },
            )?;
            let matches_value =
                outputs
                    .get("matches0")
                    .ok_or_else(|| InferenceError::UnexpectedOutput {
                        name: "matches0".to_string(),
                        expected: "named output tensor".to_string(),
                        actual: "missing output".to_string(),
                    })?;
            let scores_value =
                outputs
                    .get("mscores0")
                    .ok_or_else(|| InferenceError::UnexpectedOutput {
                        name: "mscores0".to_string(),
                        expected: "named output tensor".to_string(),
                        actual: "missing output".to_string(),
                    })?;
            let (matches_shape, matches_data) =
                super::extract_tensor::<i64>(matches_value, "matches0")?;
            let (scores_shape, scores_data) =
                super::extract_tensor::<f32>(scores_value, "mscores0")?;

            parse_lightglue_outputs(
                self.output_contract,
                matches_shape,
                matches_data,
                scores_shape,
                scores_data,
                dec_1.len(),
                dec_2.len(),
            )?
        };

        Matches::new(dec_1, dec_2, parsed.indices, parsed.scores).map_err(InferenceError::Match)
    }
}

const LIGHTGLUE_INTERFACE_DESCRIPTION: &str = "matches0:i64 and mscores0:f32 using either the standard aligned contract [1, N]/[1, N] or the fused-pair contract [N, 2]/[N]";

fn parse_lightglue_output_contract(
    outputs: &[Outlet],
) -> Result<LightGlueOutputContract, InferenceError> {
    let Some((matches_type, matches_shape)) = tensor_output_metadata(outputs, "matches0") else {
        return Err(unsupported_lightglue_interface(outputs));
    };
    let Some((scores_type, scores_shape)) = tensor_output_metadata(outputs, "mscores0") else {
        return Err(unsupported_lightglue_interface(outputs));
    };
    if matches_type != TensorElementType::Int64 || scores_type != TensorElementType::Float32 {
        return Err(unsupported_lightglue_interface(outputs));
    }

    if matches_shape.len() == 2
        && matches_shape[1] == 2
        && scores_shape.len() == 1
        && metadata_count_dimensions_compatible(matches_shape[0], scores_shape[0])
    {
        return Ok(LightGlueOutputContract::FusedPairs);
    }

    if matches_shape.len() == 2
        && scores_shape.len() == 2
        && matches!(matches_shape[0], -1 | 1)
        && matches!(scores_shape[0], -1 | 1)
        && metadata_count_dimensions_compatible(matches_shape[0], scores_shape[0])
        && metadata_count_dimensions_compatible(matches_shape[1], scores_shape[1])
    {
        return Ok(LightGlueOutputContract::StandardAligned);
    }

    Err(unsupported_lightglue_interface(outputs))
}

fn tensor_output_metadata<'a>(
    outputs: &'a [Outlet],
    name: &str,
) -> Option<(TensorElementType, &'a Shape)> {
    match outputs.iter().find(|output| output.name() == name)?.dtype() {
        ValueType::Tensor { ty, shape, .. } => Some((*ty, shape)),
        _ => None,
    }
}

fn metadata_count_dimensions_compatible(left: i64, right: i64) -> bool {
    left >= -1 && right >= -1 && (left == -1 || right == -1 || left == right)
}

fn unsupported_lightglue_interface(outputs: &[Outlet]) -> InferenceError {
    let actual = if outputs.is_empty() {
        "none".to_string()
    } else {
        outputs
            .iter()
            .map(|output| format!("{}:{}", output.name(), output.dtype()))
            .collect::<Vec<_>>()
            .join(", ")
    };
    InferenceError::UnsupportedModelInterface {
        model: "LightGlue",
        expected: LIGHTGLUE_INTERFACE_DESCRIPTION,
        actual: format!("outputs [{actual}]"),
    }
}

#[derive(Debug, PartialEq)]
struct ParsedLightGlueOutput {
    indices: Vec<(usize, usize)>,
    scores: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn parse_lightglue_outputs(
    contract: LightGlueOutputContract,
    matches_shape: &Shape,
    matches_data: &[i64],
    scores_shape: &Shape,
    scores_data: &[f32],
    source_len: usize,
    target_len: usize,
) -> Result<ParsedLightGlueOutput, InferenceError> {
    match contract {
        LightGlueOutputContract::StandardAligned => parse_standard_outputs(
            matches_shape,
            matches_data,
            scores_shape,
            scores_data,
            source_len,
            target_len,
        ),
        LightGlueOutputContract::FusedPairs => parse_fused_pair_outputs(
            matches_shape,
            matches_data,
            scores_shape,
            scores_data,
            source_len,
            target_len,
        ),
    }
}

fn parse_standard_outputs(
    matches_shape: &Shape,
    matches_data: &[i64],
    scores_shape: &Shape,
    scores_data: &[f32],
    source_len: usize,
    target_len: usize,
) -> Result<ParsedLightGlueOutput, InferenceError> {
    require_aligned_output("matches0", matches_shape, matches_data.len(), source_len)?;
    require_aligned_output("mscores0", scores_shape, scores_data.len(), source_len)?;

    let mut indices = Vec::new();
    let mut scores = Vec::new();
    for (source_index, (&match_index, &score)) in matches_data.iter().zip(scores_data).enumerate() {
        require_valid_score(source_index, score)?;
        if match_index == -1 {
            continue;
        }
        if match_index < -1 {
            return Err(InferenceError::UnexpectedOutput {
                name: "matches0".to_string(),
                expected: "-1 for unmatched detections or a non-negative target index".to_string(),
                actual: format!("source index {source_index} contains {match_index}"),
            });
        }
        let target_index =
            usize::try_from(match_index).map_err(|_| InferenceError::UnexpectedOutput {
                name: "matches0".to_string(),
                expected: "target indices representable by the host".to_string(),
                actual: format!("source index {source_index} contains {match_index}"),
            })?;
        if target_index >= target_len {
            return Err(InferenceError::UnexpectedOutput {
                name: "matches0".to_string(),
                expected: format!("target indices below {target_len}"),
                actual: format!("source index {source_index} contains {target_index}"),
            });
        }
        indices.push((source_index, target_index));
        scores.push(score);
    }
    Ok(ParsedLightGlueOutput { indices, scores })
}

fn require_aligned_output(
    name: &str,
    shape: &Shape,
    data_len: usize,
    source_len: usize,
) -> Result<(), InferenceError> {
    let exact_shape =
        shape.len() == 2 && shape[0] == 1 && usize::try_from(shape[1]).ok() == Some(source_len);
    if exact_shape && data_len == source_len {
        return Ok(());
    }
    Err(InferenceError::UnexpectedOutput {
        name: name.to_string(),
        expected: format!("shape [1, {source_len}] with {source_len} elements"),
        actual: format!("shape {shape} with {data_len} elements"),
    })
}

fn parse_fused_pair_outputs(
    matches_shape: &Shape,
    matches_data: &[i64],
    scores_shape: &Shape,
    scores_data: &[f32],
    source_len: usize,
    target_len: usize,
) -> Result<ParsedLightGlueOutput, InferenceError> {
    let pair_count = require_fused_matches(matches_shape, matches_data.len())?;
    require_fused_scores(scores_shape, scores_data.len(), pair_count)?;

    let mut indices = Vec::with_capacity(pair_count);
    let mut scores = Vec::with_capacity(pair_count);
    for (pair_index, (pair, &score)) in matches_data.chunks_exact(2).zip(scores_data).enumerate() {
        require_valid_score(pair_index, score)?;
        let source_index = require_fused_index(pair_index, "source", pair[0], source_len)?;
        let target_index = require_fused_index(pair_index, "target", pair[1], target_len)?;
        indices.push((source_index, target_index));
        scores.push(score);
    }
    Ok(ParsedLightGlueOutput { indices, scores })
}

fn require_fused_matches(shape: &Shape, data_len: usize) -> Result<usize, InferenceError> {
    let pair_count = if shape.len() == 2 && shape[1] == 2 {
        usize::try_from(shape[0]).ok()
    } else {
        None
    };
    if let Some(pair_count) = pair_count
        && pair_count.checked_mul(2) == Some(data_len)
    {
        return Ok(pair_count);
    }
    Err(InferenceError::UnexpectedOutput {
        name: "matches0".to_string(),
        expected: "shape [N, 2] with exactly 2*N elements".to_string(),
        actual: format!("shape {shape} with {data_len} elements"),
    })
}

fn require_fused_scores(
    shape: &Shape,
    data_len: usize,
    pair_count: usize,
) -> Result<(), InferenceError> {
    let exact_shape = shape.len() == 1
        && usize::try_from(shape[0]).ok() == Some(pair_count)
        && data_len == pair_count;
    if exact_shape {
        return Ok(());
    }
    Err(InferenceError::UnexpectedOutput {
        name: "mscores0".to_string(),
        expected: format!("shape [{pair_count}] with {pair_count} elements"),
        actual: format!("shape {shape} with {data_len} elements"),
    })
}

fn require_fused_index(
    pair_index: usize,
    side: &'static str,
    value: i64,
    bound: usize,
) -> Result<usize, InferenceError> {
    let index = usize::try_from(value).map_err(|_| InferenceError::UnexpectedOutput {
        name: "matches0".to_string(),
        expected: "fused pairs containing no sentinels and only non-negative indices".to_string(),
        actual: format!("pair {pair_index} has {side} index {value}"),
    })?;
    if index >= bound {
        return Err(InferenceError::UnexpectedOutput {
            name: "matches0".to_string(),
            expected: format!("fused {side} indices below {bound}"),
            actual: format!("pair {pair_index} has {side} index {index}"),
        });
    }
    Ok(index)
}

fn require_valid_score(index: usize, score: f32) -> Result<(), InferenceError> {
    if score.is_finite() && score >= 0.0 {
        return Ok(());
    }
    Err(InferenceError::UnexpectedOutput {
        name: "mscores0".to_string(),
        expected: "finite non-negative scores for every output slot".to_string(),
        actual: format!("index {index} contains {score}"),
    })
}

fn normalize_keypoints_into(detections: &Detections, out: &mut Vec<f32>) {
    let dimensions = detections.dimensions();
    let width = f64::from(dimensions.width());
    let height = f64::from(dimensions.height());
    let scale = 0.5 * width.max(height);
    let cx = width * 0.5;
    let cy = height * 0.5;

    out.clear();
    out.reserve(detections.len() * 2);
    for keypoint in detections.keypoints() {
        out.push(((f64::from(keypoint.x) - cx) / scale) as f32);
        out.push(((f64::from(keypoint.y) - cy) / scale) as f32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Descriptor, FrameId, Keypoint, SensorId};
    use ort::value::SymbolicDimensions;

    fn tensor_outlet<const N: usize>(name: &str, ty: TensorElementType, shape: [i64; N]) -> Outlet {
        Outlet::new(
            name,
            ValueType::Tensor {
                ty,
                shape: Shape::new(shape),
                dimension_symbols: SymbolicDimensions::new((0..N).map(|_| String::new())),
            },
        )
    }

    #[test]
    fn model_interface_selects_standard_and_fused_contracts_once() {
        let standard = [
            tensor_outlet("matches0", TensorElementType::Int64, [-1, -1]),
            tensor_outlet("mscores0", TensorElementType::Float32, [-1, -1]),
        ];
        let fused = [
            tensor_outlet("matches0", TensorElementType::Int64, [-1, 2]),
            tensor_outlet("mscores0", TensorElementType::Float32, [-1]),
        ];

        assert_eq!(
            parse_lightglue_output_contract(&standard).expect("standard interface"),
            LightGlueOutputContract::StandardAligned
        );
        assert_eq!(
            parse_lightglue_output_contract(&fused).expect("fused interface"),
            LightGlueOutputContract::FusedPairs
        );
    }

    #[test]
    fn model_interface_rejects_ambiguous_shapes_and_wrong_types() {
        let ambiguous = [
            tensor_outlet("matches0", TensorElementType::Int64, [-1, -1]),
            tensor_outlet("mscores0", TensorElementType::Float32, [-1]),
        ];
        let wrong_type = [
            tensor_outlet("matches0", TensorElementType::Int32, [-1, 2]),
            tensor_outlet("mscores0", TensorElementType::Float32, [-1]),
        ];

        for outputs in [&ambiguous[..], &wrong_type[..]] {
            assert!(matches!(
                parse_lightglue_output_contract(outputs),
                Err(InferenceError::UnsupportedModelInterface {
                    model: "LightGlue",
                    ..
                })
            ));
        }
    }

    #[test]
    fn standard_parser_enforces_aligned_shapes_sentinel_bounds_and_scores() {
        let shape = Shape::new([1_i64, 4]);
        let parsed = parse_lightglue_outputs(
            LightGlueOutputContract::StandardAligned,
            &shape,
            &[-1, 2, 0, -1],
            &shape,
            &[0.0, 0.75, 0.5, -0.0],
            4,
            3,
        )
        .expect("exact standard output contract");
        assert_eq!(parsed.indices, vec![(1, 2), (2, 0)]);
        assert_eq!(parsed.scores, vec![0.75, 0.5]);

        let wrong_shape = Shape::new([4_i64]);
        assert!(
            parse_lightglue_outputs(
                LightGlueOutputContract::StandardAligned,
                &wrong_shape,
                &[-1; 4],
                &shape,
                &[0.0; 4],
                4,
                3,
            )
            .is_err()
        );
        assert!(
            parse_lightglue_outputs(
                LightGlueOutputContract::StandardAligned,
                &shape,
                &[-1; 3],
                &shape,
                &[0.0; 4],
                4,
                3,
            )
            .is_err()
        );
        assert!(
            parse_lightglue_outputs(
                LightGlueOutputContract::StandardAligned,
                &shape,
                &[-1; 4],
                &shape,
                &[0.0; 3],
                4,
                3,
            )
            .is_err()
        );

        for (match_index, score) in [
            (-2, 0.0),
            (i64::MIN, 0.0),
            (0, f32::NAN),
            (0, -f32::from_bits(1)),
        ] {
            let one = Shape::new([1_i64, 1]);
            assert!(
                parse_lightglue_outputs(
                    LightGlueOutputContract::StandardAligned,
                    &one,
                    &[match_index],
                    &one,
                    &[score],
                    1,
                    1,
                )
                .is_err()
            );
        }
        let one = Shape::new([1_i64, 1]);
        assert!(
            parse_lightglue_outputs(
                LightGlueOutputContract::StandardAligned,
                &one,
                &[1],
                &one,
                &[0.0],
                1,
                1,
            )
            .is_err()
        );
    }

    #[test]
    fn fused_parser_enforces_pair_shapes_indices_and_scores() {
        let matches_shape = Shape::new([2_i64, 2]);
        let scores_shape = Shape::new([2_i64]);
        let parsed = parse_lightglue_outputs(
            LightGlueOutputContract::FusedPairs,
            &matches_shape,
            &[0, 1, 2, 0],
            &scores_shape,
            &[0.75, 0.5],
            3,
            2,
        )
        .expect("exact fused-pair output contract");
        assert_eq!(parsed.indices, vec![(0, 1), (2, 0)]);
        assert_eq!(parsed.scores, vec![0.75, 0.5]);

        for (indices, score) in [
            ([-1, 0], 0.0),
            ([0, -1], 0.0),
            ([3, 0], 0.0),
            ([0, 2], 0.0),
            ([0, 0], f32::INFINITY),
        ] {
            let one_match = Shape::new([1_i64, 2]);
            let one_score = Shape::new([1_i64]);
            assert!(
                parse_lightglue_outputs(
                    LightGlueOutputContract::FusedPairs,
                    &one_match,
                    &indices,
                    &one_score,
                    &[score],
                    3,
                    2,
                )
                .is_err()
            );
        }

        let wrong_scores_shape = Shape::new([1_i64, 2]);
        assert!(
            parse_lightglue_outputs(
                LightGlueOutputContract::FusedPairs,
                &matches_shape,
                &[0, 1, 2, 0],
                &wrong_scores_shape,
                &[0.75, 0.5],
                3,
                2,
            )
            .is_err()
        );
        assert!(
            parse_lightglue_outputs(
                LightGlueOutputContract::FusedPairs,
                &matches_shape,
                &[0, 1, 2],
                &scores_shape,
                &[0.75, 0.5],
                3,
                2,
            )
            .is_err()
        );
        assert!(
            parse_lightglue_outputs(
                LightGlueOutputContract::FusedPairs,
                &matches_shape,
                &[0, 1, 2, 0],
                &scores_shape,
                &[0.75],
                3,
                2,
            )
            .is_err()
        );
    }

    #[test]
    fn normalization_uses_exact_dimensions_until_the_f32_model_boundary() {
        const WIDTH: u32 = 16_777_217;
        const X: f32 = 16_777_216.0;
        let detections = Detections::new(
            SensorId::StereoLeft,
            FrameId::new(1),
            WIDTH,
            1,
            vec![Keypoint { x: X, y: 0.0 }],
            vec![1.0],
            vec![Descriptor::ZERO],
        )
        .expect("large exact detection domain");
        let mut normalized = Vec::new();
        normalize_keypoints_into(&detections, &mut normalized);

        let rounded_width = WIDTH as f32;
        let legacy_x = (X - rounded_width * 0.5) / (rounded_width * 0.5);
        assert_eq!(legacy_x, 1.0);
        assert_eq!(normalized[0].to_bits(), 0x3f7f_fffe);
        assert!(normalized.iter().all(|value| value.is_finite()));
    }
}
