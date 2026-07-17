use super::{
    InferenceBackend, InferenceError, InferenceRunDiagnostics, build_run_options, build_session,
    inference_env,
};
use crate::{Descriptor, Detections, DownscaleFactor, Frame, FrameDimensions, Keypoint};
use ort::session::{RunOptions, Session};
use ort::value::Outlet;
use ort::value::PrimitiveTensorElementType;
use ort::value::TensorElementType;
use ort::value::TensorRef;
use ort::value::ValueType;
use std::num::NonZeroUsize;
use std::path::Path;

use crate::DESCRIPTOR_DIM;

// The sparse adapter implements the declared profile exported by the tracked
// sp_topk2048.onnx artifact. A matching session interface selects this adapter,
// but cannot attest the model's internal semantics; callers declare the profile.
const MODEL_STRIDE: u32 = 8;
const EXCLUDED_BORDER: u32 = 4;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SuperPointSparseProfile {
    /// The tracked f16 sparse exports, including the active 2048-row Jetson model.
    #[default]
    CanonicalFp16,
    /// The archived f32 `superpoint_512.onnx` export.
    LegacyFp32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SparseOutputContract {
    capacity: usize,
    score_threshold: f32,
}

impl SuperPointSparseProfile {
    const fn contract(self) -> SparseOutputContract {
        match self {
            Self::CanonicalFp16 => SparseOutputContract {
                capacity: 2048,
                // Exact f32 value of the graph's exported fp16 threshold (0x1019).
                score_threshold: f32::from_bits(0x3a03_2000),
            },
            Self::LegacyFp32 => SparseOutputContract {
                capacity: 512,
                score_threshold: f32::from_bits(0x3a03_126f),
            },
        }
    }
}

pub struct SuperPoint {
    session: Session,
    run_options: RunOptions,
    backend: InferenceBackend,
    diagnostics: InferenceRunDiagnostics,
    kind: SuperPointModelKind,
    input_kind: SuperPointInputKind,
    scratch: Vec<f32>,
    scratch_u8: Vec<u8>,
    candidates: Vec<DenseCandidate>,
    dense_score_map: Vec<f32>,
    dense_candidate_cap: Option<NonZeroUsize>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum SuperPointModelKind {
    SparseOutputs(SparseOutputContract),
    DenseHeads,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SuperPointInputKind {
    Float32,
    Uint8,
}

#[derive(Clone, Copy, Debug)]
struct DenseCandidate {
    x: u32,
    y: u32,
    score: f32,
}

#[derive(Clone, Copy, Debug)]
struct SuperPointInputDomain {
    scale: u32,
    effective_width: u32,
    effective_height: u32,
}

impl SuperPointInputDomain {
    fn try_new(
        model_dimensions: FrameDimensions,
        frame_dimensions: FrameDimensions,
        downscale: Option<DownscaleFactor>,
    ) -> Result<Self, InferenceError> {
        let width = model_dimensions.width();
        let height = model_dimensions.height();
        if width < MODEL_STRIDE || height < MODEL_STRIDE {
            return Err(InferenceError::InputDimensionsTooSmall {
                model: "superpoint",
                width,
                height,
                minimum: MODEL_STRIDE,
            });
        }

        let scale = downscale.map_or(1, DownscaleFactor::as_u32);
        if width.checked_mul(scale) != Some(frame_dimensions.width())
            || height.checked_mul(scale) != Some(frame_dimensions.height())
        {
            return Err(InferenceError::InvariantViolation {
                context: "SuperPoint model dimensions do not map to the frame dimensions",
            });
        }

        Ok(Self {
            scale,
            effective_width: width / MODEL_STRIDE * MODEL_STRIDE,
            effective_height: height / MODEL_STRIDE * MODEL_STRIDE,
        })
    }
}

impl SuperPoint {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, InferenceError> {
        Self::new_with_backend_and_sparse_profile(
            path,
            InferenceBackend::auto(),
            SuperPointSparseProfile::default(),
        )
    }

    pub fn new_with_sparse_profile(
        path: impl AsRef<Path>,
        sparse_profile: SuperPointSparseProfile,
    ) -> Result<Self, InferenceError> {
        Self::new_with_backend_and_sparse_profile(path, InferenceBackend::auto(), sparse_profile)
    }

    pub fn new_with_backend(
        path: impl AsRef<Path>,
        backend: InferenceBackend,
    ) -> Result<Self, InferenceError> {
        Self::new_with_backend_and_sparse_profile(path, backend, SuperPointSparseProfile::default())
    }

    pub fn new_with_backend_and_sparse_profile(
        path: impl AsRef<Path>,
        backend: InferenceBackend,
        sparse_profile: SuperPointSparseProfile,
    ) -> Result<Self, InferenceError> {
        let path = path.as_ref();
        let (session, selected, diagnostics) = build_session(path, backend)?;
        let run_options = build_run_options(selected)?;
        let (kind, input_kind) =
            parse_superpoint_model_interface(session.inputs(), session.outputs(), sparse_profile)?;
        let dense_candidate_cap =
            inference_env(crate::env::try_env_usize("KIKO_SUPERPOINT_DENSE_CAP"))?
                .map(|value| {
                    NonZeroUsize::new(value).ok_or_else(|| InferenceError::InvalidSetting {
                        key: "KIKO_SUPERPOINT_DENSE_CAP",
                        value: value.to_string(),
                        expected: "an integer greater than zero",
                    })
                })
                .transpose()?;
        Ok(Self {
            session,
            run_options,
            backend: selected,
            diagnostics,
            kind,
            input_kind,
            scratch: Vec::new(),
            scratch_u8: Vec::new(),
            candidates: Vec::new(),
            dense_score_map: Vec::new(),
            dense_candidate_cap,
        })
    }

    pub fn backend(&self) -> InferenceBackend {
        self.backend
    }

    pub fn detect(&mut self, frame: &Frame) -> Result<Detections, InferenceError> {
        self.detect_limited(frame, usize::MAX)
    }

    pub fn detect_limited(
        &mut self,
        frame: &Frame,
        max_keypoints: usize,
    ) -> Result<Detections, InferenceError> {
        match self.input_kind {
            SuperPointInputKind::Float32 => {
                crate::preprocess::normalise_frame_into(frame, &mut self.scratch);

                let input_tensor = TensorRef::from_array_view((
                    [1, 1, frame.height() as usize, frame.width() as usize],
                    self.scratch.as_slice(),
                ))
                .map_err(|source| InferenceError::InputTensor {
                    name: "image",
                    source,
                })?;
                run_with_tensor(
                    self.kind,
                    &mut self.session,
                    &self.run_options,
                    frame,
                    input_tensor,
                    frame.dimensions(),
                    None,
                    &mut self.candidates,
                    &mut self.dense_score_map,
                    max_keypoints,
                    self.dense_candidate_cap,
                    self.diagnostics,
                )
            }
            SuperPointInputKind::Uint8 => {
                let input_tensor = TensorRef::from_array_view((
                    [1, 1, frame.height() as usize, frame.width() as usize],
                    frame.data(),
                ))
                .map_err(|source| InferenceError::InputTensor {
                    name: "image",
                    source,
                })?;
                run_with_tensor(
                    self.kind,
                    &mut self.session,
                    &self.run_options,
                    frame,
                    input_tensor,
                    frame.dimensions(),
                    None,
                    &mut self.candidates,
                    &mut self.dense_score_map,
                    max_keypoints,
                    self.dense_candidate_cap,
                    self.diagnostics,
                )
            }
        }
    }

    pub fn detect_with_downscale(
        &mut self,
        frame: &Frame,
        downscale: DownscaleFactor,
    ) -> Result<Detections, InferenceError> {
        self.detect_with_downscale_limited(frame, downscale, usize::MAX)
    }

    pub fn detect_with_downscale_limited(
        &mut self,
        frame: &Frame,
        downscale: DownscaleFactor,
        max_keypoints: usize,
    ) -> Result<Detections, InferenceError> {
        if downscale.get() == 1 {
            return self.detect_limited(frame, max_keypoints);
        }

        match self.input_kind {
            SuperPointInputKind::Float32 => {
                let dimensions = crate::preprocess::normalise_downscale_into(
                    frame,
                    downscale,
                    &mut self.scratch,
                )
                .map_err(InferenceError::from)?;

                let input_tensor = TensorRef::from_array_view((
                    [
                        1,
                        1,
                        dimensions.height() as usize,
                        dimensions.width() as usize,
                    ],
                    self.scratch.as_slice(),
                ))
                .map_err(|source| InferenceError::InputTensor {
                    name: "image",
                    source,
                })?;
                run_with_tensor(
                    self.kind,
                    &mut self.session,
                    &self.run_options,
                    frame,
                    input_tensor,
                    dimensions,
                    Some(downscale),
                    &mut self.candidates,
                    &mut self.dense_score_map,
                    max_keypoints,
                    self.dense_candidate_cap,
                    self.diagnostics,
                )
            }
            SuperPointInputKind::Uint8 => {
                let dimensions =
                    crate::preprocess::downscale_u8_into(frame, downscale, &mut self.scratch_u8)
                        .map_err(InferenceError::from)?;

                let input_tensor = TensorRef::from_array_view((
                    [
                        1,
                        1,
                        dimensions.height() as usize,
                        dimensions.width() as usize,
                    ],
                    self.scratch_u8.as_slice(),
                ))
                .map_err(|source| InferenceError::InputTensor {
                    name: "image",
                    source,
                })?;
                run_with_tensor(
                    self.kind,
                    &mut self.session,
                    &self.run_options,
                    frame,
                    input_tensor,
                    dimensions,
                    Some(downscale),
                    &mut self.candidates,
                    &mut self.dense_score_map,
                    max_keypoints,
                    self.dense_candidate_cap,
                    self.diagnostics,
                )
            }
        }
    }
}

const SUPERPOINT_INTERFACE_DESCRIPTION: &str = "one image tensor input f32|u8 [1, 1, height, width] and either the canonical sparse outputs keypoints:i64 [1, N, 2], scores:f32 [1, N], descriptors:f32 rank-3, or the dense-head outputs dense_scores:f32 and dense_descriptors:f32";

fn parse_superpoint_model_interface(
    inputs: &[Outlet],
    outputs: &[Outlet],
    sparse_profile: SuperPointSparseProfile,
) -> Result<(SuperPointModelKind, SuperPointInputKind), InferenceError> {
    if inputs.len() != 1 || inputs[0].name() != "image" {
        return Err(unsupported_superpoint_interface(inputs, outputs));
    }
    let input_kind = match inputs[0].dtype() {
        ValueType::Tensor {
            ty: TensorElementType::Float32,
            shape,
            ..
        } if superpoint_input_shape_supported(shape) => SuperPointInputKind::Float32,
        ValueType::Tensor {
            ty: TensorElementType::Uint8,
            shape,
            ..
        } if superpoint_input_shape_supported(shape) => SuperPointInputKind::Uint8,
        _ => return Err(unsupported_superpoint_interface(inputs, outputs)),
    };

    let kind = if outputs.len() == 2
        && outlet_tensor_element_type(outputs, "dense_scores") == Some(TensorElementType::Float32)
        && outlet_tensor_element_type(outputs, "dense_descriptors")
            == Some(TensorElementType::Float32)
    {
        SuperPointModelKind::DenseHeads
    } else if canonical_sparse_interface(outputs) {
        SuperPointModelKind::SparseOutputs(sparse_profile.contract())
    } else {
        return Err(unsupported_superpoint_interface(inputs, outputs));
    };

    Ok((kind, input_kind))
}

fn superpoint_input_shape_supported(shape: &ort::value::Shape) -> bool {
    shape.len() == 4
        && shape[0] == 1
        && shape[1] == 1
        && (shape[2] == -1 || shape[2] > 0)
        && (shape[3] == -1 || shape[3] > 0)
}

fn canonical_sparse_interface(outputs: &[Outlet]) -> bool {
    outputs.len() == 3
        && outlet_tensor_shape(outputs, "keypoints", TensorElementType::Int64)
            .is_some_and(|shape| {
                shape.len() == 3
                    && shape[0] == 1
                    && (shape[1] == -1 || shape[1] >= 0)
                    && shape[2] == 2
            })
        && outlet_tensor_shape(outputs, "scores", TensorElementType::Float32).is_some_and(
            |shape| {
                shape.len() == 2
                    && shape[0] == 1
                    && (shape[1] == -1 || shape[1] >= 0)
            },
        )
        // The tracked exports have inaccurate symbolic descriptor dimensions;
        // rank and type are construction-time facts, exact [1, N, 256] is a
        // concrete runtime requirement.
        && outlet_tensor_shape(outputs, "descriptors", TensorElementType::Float32)
            .is_some_and(|shape| shape.len() == 3)
}

fn outlet_tensor_shape<'a>(
    outlets: &'a [Outlet],
    name: &str,
    expected_type: TensorElementType,
) -> Option<&'a ort::value::Shape> {
    match outlets.iter().find(|outlet| outlet.name() == name)?.dtype() {
        ValueType::Tensor { ty, shape, .. } if *ty == expected_type => Some(shape),
        _ => None,
    }
}

fn outlet_tensor_element_type(outlets: &[Outlet], name: &str) -> Option<TensorElementType> {
    match outlets.iter().find(|outlet| outlet.name() == name)?.dtype() {
        ValueType::Tensor { ty, .. } => Some(*ty),
        _ => None,
    }
}

fn unsupported_superpoint_interface(inputs: &[Outlet], outputs: &[Outlet]) -> InferenceError {
    fn describe(outlets: &[Outlet]) -> String {
        if outlets.is_empty() {
            return "none".to_string();
        }
        outlets
            .iter()
            .map(|outlet| format!("{}:{}", outlet.name(), outlet.dtype()))
            .collect::<Vec<_>>()
            .join(", ")
    }

    InferenceError::UnsupportedModelInterface {
        model: "SuperPoint",
        expected: SUPERPOINT_INTERFACE_DESCRIPTION,
        actual: format!(
            "inputs [{}], outputs [{}]",
            describe(inputs),
            describe(outputs)
        ),
    }
}

// Keep the session, frame metadata, tensor, and reusable scratch buffers explicit.
#[allow(clippy::too_many_arguments)]
fn run_with_tensor<T>(
    kind: SuperPointModelKind,
    session: &mut Session,
    run_options: &RunOptions,
    frame: &Frame,
    input_tensor: TensorRef<'_, T>,
    input_dimensions: FrameDimensions,
    downscale: Option<DownscaleFactor>,
    candidates: &mut Vec<DenseCandidate>,
    dense_score_map: &mut Vec<f32>,
    max_keypoints: usize,
    dense_candidate_cap: Option<NonZeroUsize>,
    diagnostics: InferenceRunDiagnostics,
) -> Result<Detections, InferenceError>
where
    T: PrimitiveTensorElementType + std::fmt::Debug,
{
    match kind {
        SuperPointModelKind::SparseOutputs(contract) => run_sparse_inference(
            session,
            run_options,
            frame,
            input_tensor,
            input_dimensions,
            downscale,
            max_keypoints,
            contract,
            diagnostics,
        ),
        SuperPointModelKind::DenseHeads => run_dense_inference(
            session,
            run_options,
            frame,
            input_tensor,
            input_dimensions,
            downscale,
            candidates,
            dense_score_map,
            max_keypoints,
            dense_candidate_cap,
            diagnostics,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_sparse_inference<T>(
    session: &mut Session,
    run_options: &RunOptions,
    frame: &Frame,
    input_tensor: TensorRef<'_, T>,
    input_dimensions: FrameDimensions,
    downscale: Option<DownscaleFactor>,
    max_keypoints: usize,
    contract: SparseOutputContract,
    diagnostics: InferenceRunDiagnostics,
) -> Result<Detections, InferenceError>
where
    T: PrimitiveTensorElementType + std::fmt::Debug,
{
    let input_domain =
        SuperPointInputDomain::try_new(input_dimensions, frame.dimensions(), downscale)?;
    let outputs = super::run_with_slow_call_diagnostics(diagnostics, "superpoint", || {
        session
            .run_with_options(ort::inputs!["image" => input_tensor], run_options)
            .map_err(|source| InferenceError::SessionRun {
                model: "superpoint-sparse",
                source,
            })
    })?;

    let keypoints_value =
        outputs
            .get("keypoints")
            .ok_or_else(|| InferenceError::UnexpectedOutput {
                name: "keypoints".to_string(),
                expected: "named output tensor".to_string(),
                actual: "missing output".to_string(),
            })?;
    let scores_value = outputs
        .get("scores")
        .ok_or_else(|| InferenceError::UnexpectedOutput {
            name: "scores".to_string(),
            expected: "named output tensor".to_string(),
            actual: "missing output".to_string(),
        })?;
    let descriptors_value =
        outputs
            .get("descriptors")
            .ok_or_else(|| InferenceError::UnexpectedOutput {
                name: "descriptors".to_string(),
                expected: "named output tensor".to_string(),
                actual: "missing output".to_string(),
            })?;

    let keypoints =
        extract_tensor_allow_empty::<i64>(keypoints_value, "keypoints", TensorElementType::Int64)?;
    let scores =
        extract_tensor_allow_empty::<f32>(scores_value, "scores", TensorElementType::Float32)?;
    let descriptors = extract_tensor_allow_empty::<f32>(
        descriptors_value,
        "descriptors",
        TensorElementType::Float32,
    )?;
    let parsed = parse_sparse_outputs(
        keypoints,
        scores,
        descriptors,
        input_domain,
        max_keypoints,
        contract,
    )?;

    Ok(Detections::from_parsed_components(
        frame.sensor_id(),
        frame.frame_id(),
        frame.dimensions(),
        parsed.keypoints,
        parsed.scores,
        parsed.descriptors,
    ))
}

fn extract_tensor_allow_empty<'value, T: PrimitiveTensorElementType>(
    value: &'value ort::value::DynValue,
    name: &str,
    expected_type: TensorElementType,
) -> Result<(&'value ort::value::Shape, &'value [T]), InferenceError> {
    if let ValueType::Tensor { ty, shape, .. } = value.dtype()
        && *ty == expected_type
        && shape.num_elements() == 0
    {
        // ONNX Runtime may represent an empty tensor with a null backing pointer.
        // The concrete shape and construction-time type still define an empty slice.
        return Ok((shape, &[]));
    }
    super::extract_tensor::<T>(value, name)
}

#[derive(Debug)]
struct ParsedSuperPointOutput {
    keypoints: Vec<Keypoint>,
    scores: Vec<f32>,
    descriptors: Vec<Descriptor>,
}

fn parse_sparse_outputs(
    keypoints: (&ort::value::Shape, &[i64]),
    scores: (&ort::value::Shape, &[f32]),
    descriptors: (&ort::value::Shape, &[f32]),
    input_domain: SuperPointInputDomain,
    max_keypoints: usize,
    contract: SparseOutputContract,
) -> Result<ParsedSuperPointOutput, InferenceError> {
    let (keypoint_shape, keypoint_data) = keypoints;
    let count_i64 = match &keypoint_shape[..] {
        [1, count, 2] if *count >= 0 => *count,
        _ => {
            return Err(InferenceError::UnexpectedOutput {
                name: "keypoints".to_string(),
                expected: format!(
                    "i64 tensor shape [1, N, 2] with 0 <= N <= {}",
                    contract.capacity
                ),
                actual: format!("tensor shape {keypoint_shape}"),
            });
        }
    };
    let count = usize::try_from(count_i64).map_err(|_| InferenceError::UnexpectedOutput {
        name: "keypoints".to_string(),
        expected: "keypoint count representable by the host".to_string(),
        actual: format!("keypoint count {count_i64}"),
    })?;
    if count > contract.capacity {
        return Err(InferenceError::UnexpectedOutput {
            name: "keypoints".to_string(),
            expected: format!("at most {} keypoints", contract.capacity),
            actual: format!("{count} keypoints"),
        });
    }

    let keypoint_elements =
        count
            .checked_mul(2)
            .ok_or_else(|| InferenceError::UnexpectedOutput {
                name: "keypoints".to_string(),
                expected: "keypoint tensor size representable by the host".to_string(),
                actual: format!("keypoint count {count}"),
            })?;
    require_shape_and_length(
        "keypoints",
        keypoint_shape,
        keypoint_data.len(),
        &[1, count_i64, 2],
        keypoint_elements,
    )?;
    require_shape_and_length("scores", scores.0, scores.1.len(), &[1, count_i64], count)?;
    let descriptor_elements =
        count
            .checked_mul(DESCRIPTOR_DIM)
            .ok_or_else(|| InferenceError::UnexpectedOutput {
                name: "descriptors".to_string(),
                expected: "descriptor tensor size representable by the host".to_string(),
                actual: format!("descriptor count {count}"),
            })?;
    require_shape_and_length(
        "descriptors",
        descriptors.0,
        descriptors.1.len(),
        &[1, count_i64, DESCRIPTOR_DIM as i64],
        descriptor_elements,
    )?;

    let mut previous_score = f32::INFINITY;
    for (index, &score) in scores.1.iter().enumerate() {
        if !score.is_finite() || score <= contract.score_threshold || score > 1.0 {
            return Err(InferenceError::UnexpectedOutput {
                name: "scores".to_string(),
                expected: format!(
                    "finite confidence scores within ({}, 1] in non-increasing order",
                    contract.score_threshold
                ),
                actual: format!("index {index} contains {score}"),
            });
        }
        if score > previous_score {
            return Err(InferenceError::UnexpectedOutput {
                name: "scores".to_string(),
                expected: "confidence scores in non-increasing TopK order".to_string(),
                actual: format!(
                    "index {index} contains {score} after {}",
                    scores.1[index - 1]
                ),
            });
        }
        previous_score = score;
    }

    let selected_count = count.min(max_keypoints);
    let mut parsed_keypoints = Vec::with_capacity(selected_count);
    for (index, pair) in keypoint_data.chunks_exact(2).enumerate() {
        let x = parse_canonical_coordinate(
            pair[0],
            input_domain.effective_width,
            input_domain.scale,
            index,
            "x",
        )?;
        let y = parse_canonical_coordinate(
            pair[1],
            input_domain.effective_height,
            input_domain.scale,
            index,
            "y",
        )?;
        if index < selected_count {
            parsed_keypoints.push(Keypoint { x, y });
        }
    }

    let mut parsed_descriptors = Vec::with_capacity(selected_count);
    for (descriptor_index, chunk) in descriptors.1.chunks_exact(DESCRIPTOR_DIM).enumerate() {
        Descriptor::validate_components(chunk).map_err(|source| {
            InferenceError::DescriptorOutput {
                name: "descriptors".to_string(),
                descriptor_index,
                source,
            }
        })?;
        if descriptor_index < selected_count {
            let mut descriptor = [0.0_f32; DESCRIPTOR_DIM];
            descriptor.copy_from_slice(chunk);
            parsed_descriptors.push(Descriptor::from_validated_components(descriptor));
        }
    }

    Ok(ParsedSuperPointOutput {
        keypoints: parsed_keypoints,
        scores: scores.1[..selected_count].to_vec(),
        descriptors: parsed_descriptors,
    })
}

fn require_shape_and_length(
    name: &str,
    shape: &ort::value::Shape,
    actual_length: usize,
    expected_shape: &[i64],
    expected_length: usize,
) -> Result<(), InferenceError> {
    if &shape[..] == expected_shape && actual_length == expected_length {
        return Ok(());
    }

    Err(InferenceError::UnexpectedOutput {
        name: name.to_string(),
        expected: format!("shape {expected_shape:?} with {expected_length} elements"),
        actual: format!("shape {shape} with {actual_length} elements"),
    })
}

fn parse_canonical_coordinate(
    value: i64,
    effective_extent: u32,
    scale: u32,
    keypoint_index: usize,
    axis: &'static str,
) -> Result<f32, InferenceError> {
    let upper_bound = effective_extent - EXCLUDED_BORDER;
    let coordinate = u32::try_from(value)
        .ok()
        .filter(|&coordinate| coordinate >= EXCLUDED_BORDER && coordinate < upper_bound)
        .ok_or_else(|| InferenceError::UnexpectedOutput {
            name: "keypoints".to_string(),
            expected: format!(
                "absolute integer {axis} coordinates within [{EXCLUDED_BORDER}, {upper_bound})"
            ),
            actual: format!("keypoint {keypoint_index} contains {axis}={value}"),
        })?;
    let scaled = coordinate
        .checked_mul(scale)
        .ok_or(InferenceError::InvariantViolation {
            context: "validated SuperPoint coordinate scaling overflowed",
        })?;
    let narrowed = scaled as f32;
    if f64::from(narrowed) != f64::from(scaled) {
        return Err(InferenceError::KeypointCoordinateUnrepresentable {
            model: "superpoint",
            index: keypoint_index,
            axis,
            coordinate: scaled,
        });
    }
    Ok(narrowed)
}

// Dense decoding mutates two independent scratch buffers in addition to inference inputs.
#[allow(clippy::too_many_arguments)]
fn run_dense_inference<T>(
    session: &mut Session,
    run_options: &RunOptions,
    frame: &Frame,
    input_tensor: TensorRef<'_, T>,
    input_dimensions: FrameDimensions,
    downscale: Option<DownscaleFactor>,
    candidates: &mut Vec<DenseCandidate>,
    dense_score_map: &mut Vec<f32>,
    max_keypoints: usize,
    dense_candidate_cap: Option<NonZeroUsize>,
    diagnostics: InferenceRunDiagnostics,
) -> Result<Detections, InferenceError>
where
    T: PrimitiveTensorElementType + std::fmt::Debug,
{
    let outputs = super::run_with_slow_call_diagnostics(diagnostics, "superpoint", || {
        session
            .run_with_options(ort::inputs!["image" => input_tensor], run_options)
            .map_err(|source| InferenceError::SessionRun {
                model: "superpoint-dense",
                source,
            })
    })?;

    let scores_value =
        outputs
            .get("dense_scores")
            .ok_or_else(|| InferenceError::UnexpectedOutput {
                name: "dense_scores".to_string(),
                expected: "named output tensor".to_string(),
                actual: "missing output".to_string(),
            })?;
    let descriptors_value =
        outputs
            .get("dense_descriptors")
            .ok_or_else(|| InferenceError::UnexpectedOutput {
                name: "dense_descriptors".to_string(),
                expected: "named output tensor".to_string(),
                actual: "missing output".to_string(),
            })?;

    let (scores_shape, scores_data) = super::extract_tensor::<f32>(scores_value, "dense_scores")?;
    let (desc_shape, desc_data) =
        super::extract_tensor::<f32>(descriptors_value, "dense_descriptors")?;
    let (grid_h, grid_w) = dense_scores_shape(scores_shape, scores_data.len())?;
    dense_descriptors_shape(desc_shape, desc_data.len(), grid_h, grid_w)?;

    collect_dense_candidates(
        scores_data,
        grid_h,
        grid_w,
        input_dimensions.width(),
        input_dimensions.height(),
        dense_score_map,
        candidates,
    );
    let default_cap = if max_keypoints == usize::MAX {
        4096
    } else {
        max_keypoints
    };
    let cap = dense_candidate_cap
        .map(NonZeroUsize::get)
        .unwrap_or(default_cap)
        .min(default_cap);
    sort_and_cap_candidates(candidates, cap);

    let mut keypoints = Vec::with_capacity(candidates.len());
    let mut scores = Vec::with_capacity(candidates.len());
    let mut descriptors = Vec::with_capacity(candidates.len());
    for (descriptor_index, candidate) in candidates.iter().copied().enumerate() {
        let mut x = candidate.x as f32;
        let mut y = candidate.y as f32;
        if let Some(scale) = downscale {
            let factor = scale.get() as f32;
            x *= factor;
            y *= factor;
        }
        keypoints.push(Keypoint { x, y });
        scores.push(candidate.score);
        let descriptor = sample_dense_descriptor(
            desc_data,
            grid_h,
            grid_w,
            input_dimensions.width(),
            input_dimensions.height(),
            candidate.x as f32,
            candidate.y as f32,
        )
        .map_err(|source| InferenceError::DescriptorOutput {
            name: "dense_descriptors".to_string(),
            descriptor_index,
            source,
        })?;
        descriptors.push(descriptor);
    }

    Detections::from_dimensions(
        frame.sensor_id(),
        frame.frame_id(),
        frame.dimensions(),
        keypoints,
        scores,
        descriptors,
    )
    .map_err(InferenceError::Detection)
}

fn dense_scores_shape(
    shape: &ort::value::Shape,
    len: usize,
) -> Result<(usize, usize), InferenceError> {
    let dims = &shape[..];
    if dims.len() != 4 || dims[0] != 1 || dims[1] != 65 {
        return Err(InferenceError::UnexpectedOutput {
            name: "dense_scores".to_string(),
            expected: "shape [1, 65, grid_h, grid_w]".to_string(),
            actual: format!("{shape}"),
        });
    }
    let grid_h = usize::try_from(dims[2]).map_err(|_| InferenceError::UnexpectedOutput {
        name: "dense_scores".to_string(),
        expected: "positive grid height".to_string(),
        actual: format!("{shape}"),
    })?;
    let grid_w = usize::try_from(dims[3]).map_err(|_| InferenceError::UnexpectedOutput {
        name: "dense_scores".to_string(),
        expected: "positive grid width".to_string(),
        actual: format!("{shape}"),
    })?;
    if grid_h == 0 || grid_w == 0 || len != 65 * grid_h * grid_w {
        return Err(InferenceError::UnexpectedOutput {
            name: "dense_scores".to_string(),
            expected: "contiguous score tensor".to_string(),
            actual: format!("shape {shape} with {len} elements"),
        });
    }
    Ok((grid_h, grid_w))
}

fn dense_descriptors_shape(
    shape: &ort::value::Shape,
    len: usize,
    grid_h: usize,
    grid_w: usize,
) -> Result<(), InferenceError> {
    let dims = &shape[..];
    let channels = dims
        .get(1)
        .and_then(|value| usize::try_from(*value).ok())
        .unwrap_or(0);
    if dims.len() != 4 || dims[0] != 1 || channels != DESCRIPTOR_DIM {
        return Err(InferenceError::UnexpectedOutput {
            name: "dense_descriptors".to_string(),
            expected: format!("shape [1, {DESCRIPTOR_DIM}, grid_h, grid_w]"),
            actual: format!("{shape}"),
        });
    }
    let desc_grid_h = usize::try_from(dims[2]).map_err(|_| InferenceError::UnexpectedOutput {
        name: "dense_descriptors".to_string(),
        expected: "positive grid height".to_string(),
        actual: format!("{shape}"),
    })?;
    let desc_grid_w = usize::try_from(dims[3]).map_err(|_| InferenceError::UnexpectedOutput {
        name: "dense_descriptors".to_string(),
        expected: "positive grid width".to_string(),
        actual: format!("{shape}"),
    })?;
    if desc_grid_h != grid_h || desc_grid_w != grid_w || len != DESCRIPTOR_DIM * grid_h * grid_w {
        return Err(InferenceError::UnexpectedOutput {
            name: "dense_descriptors".to_string(),
            expected: "descriptor tensor matching dense_scores grid".to_string(),
            actual: format!("shape {shape} with {len} elements"),
        });
    }
    Ok(())
}

fn collect_dense_candidates(
    scores: &[f32],
    grid_h: usize,
    grid_w: usize,
    width: u32,
    height: u32,
    dense_score_map: &mut Vec<f32>,
    candidates: &mut Vec<DenseCandidate>,
) {
    const BORDER: usize = 4;
    const NMS_RADIUS: usize = 4;
    const MIN_SCORE: f32 = 0.0005;

    candidates.clear();
    let width = width as usize;
    let height = height as usize;
    if width <= BORDER * 2 || height <= BORDER * 2 {
        return;
    }
    build_dense_score_map(scores, grid_h, grid_w, width, height, dense_score_map);
    candidates.reserve((width * height / 32).max(512));
    for y in BORDER..(height - BORDER) {
        let row = y * width;
        for x in BORDER..(width - BORDER) {
            let score = dense_score_map[row + x];
            if score <= MIN_SCORE {
                continue;
            }
            let mut is_max = true;
            'neighborhood: for ny in (y - NMS_RADIUS)..=(y + NMS_RADIUS) {
                let neighbor_row = ny * width;
                for nx in (x - NMS_RADIUS)..=(x + NMS_RADIUS) {
                    if nx == x && ny == y {
                        continue;
                    }
                    if dense_score_map[neighbor_row + nx] > score {
                        is_max = false;
                        break 'neighborhood;
                    }
                }
            }
            if is_max {
                candidates.push(DenseCandidate {
                    x: x as u32,
                    y: y as u32,
                    score,
                });
            }
        }
    }
}

fn build_dense_score_map(
    scores: &[f32],
    grid_h: usize,
    grid_w: usize,
    width: usize,
    height: usize,
    dense_score_map: &mut Vec<f32>,
) {
    dense_score_map.resize(width * height, f32::NEG_INFINITY);
    for cell_y in 0..grid_h {
        let base_y = cell_y * 8;
        for cell_x in 0..grid_w {
            let base_x = cell_x * 8;
            for offset_y in 0..8 {
                let y = base_y + offset_y;
                if y >= height {
                    break;
                }
                let row = y * width;
                for offset_x in 0..8 {
                    let x = base_x + offset_x;
                    if x >= width {
                        break;
                    }
                    let channel = offset_y * 8 + offset_x;
                    dense_score_map[row + x] =
                        scores[(channel * grid_h + cell_y) * grid_w + cell_x];
                }
            }
        }
    }
}

fn sort_and_cap_candidates(candidates: &mut Vec<DenseCandidate>, cap: usize) {
    if cap == 0 {
        candidates.clear();
        return;
    }
    let cmp_desc = |a: &DenseCandidate, b: &DenseCandidate| b.score.total_cmp(&a.score);
    if candidates.len() > cap {
        candidates.select_nth_unstable_by(cap - 1, cmp_desc);
        candidates.truncate(cap);
    }
    candidates.sort_unstable_by(cmp_desc);
}

fn sample_dense_descriptor(
    desc: &[f32],
    grid_h: usize,
    grid_w: usize,
    width: u32,
    height: u32,
    x: f32,
    y: f32,
) -> Result<Descriptor, crate::DescriptorError> {
    let gx = ((x - 3.5) / (width as f32 - 4.5)) * (grid_w.saturating_sub(1) as f32);
    let gy = ((y - 3.5) / (height as f32 - 4.5)) * (grid_h.saturating_sub(1) as f32);
    let x0 = gx.floor().clamp(0.0, grid_w.saturating_sub(1) as f32) as usize;
    let y0 = gy.floor().clamp(0.0, grid_h.saturating_sub(1) as f32) as usize;
    let x1 = (x0 + 1).min(grid_w.saturating_sub(1));
    let y1 = (y0 + 1).min(grid_h.saturating_sub(1));
    let wx = (gx - x0 as f32).clamp(0.0, 1.0);
    let wy = (gy - y0 as f32).clamp(0.0, 1.0);
    let w00 = (1.0 - wx) * (1.0 - wy);
    let w10 = wx * (1.0 - wy);
    let w01 = (1.0 - wx) * wy;
    let w11 = wx * wy;

    // Scaled sum-of-squares avoids overflowing on finite model output while
    // retaining the existing 1e-6 norm floor.
    let mut out = [0.0_f32; DESCRIPTOR_DIM];
    let mut scale = 0.0_f32;
    let mut scaled_sum_squares = 1.0_f32;
    for (channel, value) in out.iter_mut().enumerate() {
        let base = channel * grid_h * grid_w;
        let sample = w00 * desc[base + y0 * grid_w + x0]
            + w10 * desc[base + y0 * grid_w + x1]
            + w01 * desc[base + y1 * grid_w + x0]
            + w11 * desc[base + y1 * grid_w + x1];
        if !sample.is_finite() {
            return Err(crate::DescriptorError::NonFiniteComponent {
                index: channel,
                value: sample,
            });
        }
        *value = sample;
        let magnitude = sample.abs();
        if magnitude == 0.0 {
            continue;
        }
        if scale < magnitude {
            let ratio = scale / magnitude;
            scaled_sum_squares = 1.0 + scaled_sum_squares * ratio * ratio;
            scale = magnitude;
        } else {
            let ratio = magnitude / scale;
            scaled_sum_squares += ratio * ratio;
        }
    }
    let scaled_norm = scaled_sum_squares.sqrt();
    if scale == 0.0 || scale <= 1e-6 / scaled_norm {
        for value in &mut out {
            *value *= 1e6;
        }
    } else {
        for value in &mut out {
            *value = (*value / scale) / scaled_norm;
        }
    }
    Descriptor::try_new(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CANONICAL_CONTRACT: SparseOutputContract =
        SuperPointSparseProfile::CanonicalFp16.contract();
    const LEGACY_CONTRACT: SparseOutputContract = SuperPointSparseProfile::LegacyFp32.contract();

    fn tensor_outlet_with_shape<const N: usize>(
        name: &str,
        ty: TensorElementType,
        shape: [i64; N],
    ) -> Outlet {
        Outlet::new(
            name,
            ValueType::Tensor {
                ty,
                shape: ort::value::Shape::new(shape),
                dimension_symbols: ort::value::SymbolicDimensions::new(
                    (0..N).map(|_| String::new()),
                ),
            },
        )
    }

    fn image_outlet(ty: TensorElementType) -> Outlet {
        tensor_outlet_with_shape("image", ty, [1_i64, 1, -1, -1])
    }

    fn canonical_sparse_outlets(keypoint_type: TensorElementType) -> [Outlet; 3] {
        [
            tensor_outlet_with_shape("keypoints", keypoint_type, [1_i64, -1, 2]),
            tensor_outlet_with_shape("scores", TensorElementType::Float32, [1_i64, -1]),
            // This deliberately mirrors the tracked artifact's inaccurate
            // symbolic descriptor dimensions. Runtime parsing is exact.
            tensor_outlet_with_shape("descriptors", TensorElementType::Float32, [-1_i64, -1, -1]),
        ]
    }

    #[allow(clippy::too_many_arguments)]
    fn parse_fixture(
        keypoint_shape: &[i64],
        keypoint_data: &[i64],
        score_shape: &[i64],
        score_data: &[f32],
        descriptor_shape: &[i64],
        descriptor_data: &[f32],
        model_dimensions: FrameDimensions,
        frame_dimensions: FrameDimensions,
        downscale: Option<DownscaleFactor>,
        max_keypoints: usize,
    ) -> Result<ParsedSuperPointOutput, InferenceError> {
        parse_fixture_with_contract(
            keypoint_shape,
            keypoint_data,
            score_shape,
            score_data,
            descriptor_shape,
            descriptor_data,
            model_dimensions,
            frame_dimensions,
            downscale,
            max_keypoints,
            CANONICAL_CONTRACT,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn parse_fixture_with_contract(
        keypoint_shape: &[i64],
        keypoint_data: &[i64],
        score_shape: &[i64],
        score_data: &[f32],
        descriptor_shape: &[i64],
        descriptor_data: &[f32],
        model_dimensions: FrameDimensions,
        frame_dimensions: FrameDimensions,
        downscale: Option<DownscaleFactor>,
        max_keypoints: usize,
        contract: SparseOutputContract,
    ) -> Result<ParsedSuperPointOutput, InferenceError> {
        let keypoint_shape = ort::value::Shape::new(keypoint_shape.iter().copied());
        let score_shape = ort::value::Shape::new(score_shape.iter().copied());
        let descriptor_shape = ort::value::Shape::new(descriptor_shape.iter().copied());
        let input_domain =
            SuperPointInputDomain::try_new(model_dimensions, frame_dimensions, downscale)?;
        parse_sparse_outputs(
            (&keypoint_shape, keypoint_data),
            (&score_shape, score_data),
            (&descriptor_shape, descriptor_data),
            input_domain,
            max_keypoints,
            contract,
        )
    }

    fn parse_original_scale(
        keypoint_shape: &[i64],
        keypoint_data: &[i64],
        score_shape: &[i64],
        score_data: &[f32],
        descriptor_shape: &[i64],
        descriptor_data: &[f32],
    ) -> Result<ParsedSuperPointOutput, InferenceError> {
        let dimensions = FrameDimensions::try_new(640, 480).expect("valid dimensions");
        parse_fixture(
            keypoint_shape,
            keypoint_data,
            score_shape,
            score_data,
            descriptor_shape,
            descriptor_data,
            dimensions,
            dimensions,
            None,
            usize::MAX,
        )
    }

    fn assert_output_error(
        result: Result<ParsedSuperPointOutput, InferenceError>,
        expected_name: &str,
    ) {
        assert!(matches!(
            result.expect_err("malformed model output"),
            InferenceError::UnexpectedOutput { name, .. } if name == expected_name
        ));
    }

    #[test]
    fn superpoint_interface_is_parsed_once_into_supported_kinds() {
        let float_input = [image_outlet(TensorElementType::Float32)];
        let uint8_input = [image_outlet(TensorElementType::Uint8)];
        let sparse_i64 = canonical_sparse_outlets(TensorElementType::Int64);
        let dense = [
            tensor_outlet_with_shape(
                "dense_scores",
                TensorElementType::Float32,
                [1_i64, 65, -1, -1],
            ),
            tensor_outlet_with_shape(
                "dense_descriptors",
                TensorElementType::Float32,
                [1_i64, DESCRIPTOR_DIM as i64, -1, -1],
            ),
        ];

        assert_eq!(
            parse_superpoint_model_interface(
                &float_input,
                &sparse_i64,
                SuperPointSparseProfile::CanonicalFp16,
            )
            .expect("sparse i64"),
            (
                SuperPointModelKind::SparseOutputs(CANONICAL_CONTRACT),
                SuperPointInputKind::Float32
            )
        );
        assert_eq!(
            parse_superpoint_model_interface(
                &uint8_input,
                &sparse_i64,
                SuperPointSparseProfile::LegacyFp32,
            )
            .expect("sparse u8"),
            (
                SuperPointModelKind::SparseOutputs(LEGACY_CONTRACT),
                SuperPointInputKind::Uint8
            )
        );
        assert_eq!(
            parse_superpoint_model_interface(
                &float_input,
                &dense,
                SuperPointSparseProfile::LegacyFp32,
            )
            .expect("dense"),
            (
                SuperPointModelKind::DenseHeads,
                SuperPointInputKind::Float32
            )
        );
    }

    #[test]
    fn every_tracked_superpoint_artifact_selects_an_explicit_profile() {
        let model_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
        for name in [
            "sp.onnx",
            "sp_topk1024.onnx",
            "sp_topk1536.onnx",
            "sp_topk2048.onnx",
        ] {
            let model = SuperPoint::new_with_backend(model_dir.join(name), InferenceBackend::Cpu)
                .unwrap_or_else(|error| panic!("{name} must select the sparse profile: {error}"));
            assert_eq!(
                model.kind,
                SuperPointModelKind::SparseOutputs(CANONICAL_CONTRACT)
            );
            assert_eq!(model.input_kind, SuperPointInputKind::Float32);
        }

        let model = SuperPoint::new_with_backend_and_sparse_profile(
            model_dir.join("superpoint_512.onnx"),
            InferenceBackend::Cpu,
            SuperPointSparseProfile::LegacyFp32,
        )
        .expect("tracked legacy f32 sparse profile");
        assert_eq!(
            model.kind,
            SuperPointModelKind::SparseOutputs(LEGACY_CONTRACT)
        );
        assert_eq!(model.input_kind, SuperPointInputKind::Float32);

        let model = SuperPoint::new_with_backend(
            model_dir.join("sp_topk2048_u8.onnx"),
            InferenceBackend::Cpu,
        )
        .expect("tracked u8 sparse profile");
        assert_eq!(
            model.kind,
            SuperPointModelKind::SparseOutputs(CANONICAL_CONTRACT)
        );
        assert_eq!(model.input_kind, SuperPointInputKind::Uint8);

        let model =
            SuperPoint::new_with_backend(model_dir.join("sp_heads.onnx"), InferenceBackend::Cpu)
                .expect("tracked dense-head profile");
        assert_eq!(model.kind, SuperPointModelKind::DenseHeads);
        assert_eq!(model.input_kind, SuperPointInputKind::Float32);
    }

    #[test]
    fn active_and_legacy_artifacts_execute_through_their_declared_runtime_contracts() {
        let model_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
        let data = (0..64 * 64)
            .map(|index| ((index * 37 + index / 64 * 11) % 256) as u8)
            .collect();
        let frame = Frame::new(
            crate::SensorId::StereoLeft,
            crate::FrameId::new(1),
            crate::Timestamp::from_nanos(1),
            64,
            64,
            data,
        )
        .expect("synthetic frame");

        let mut active =
            SuperPoint::new_with_backend(model_dir.join("sp_topk2048.onnx"), InferenceBackend::Cpu)
                .expect("active sparse model");
        let detections = active
            .detect_limited(&frame, 16)
            .expect("active runtime tensors satisfy the declared contract");
        assert!(detections.len() <= 16);

        let mut legacy = SuperPoint::new_with_backend_and_sparse_profile(
            model_dir.join("superpoint_512.onnx"),
            InferenceBackend::Cpu,
            SuperPointSparseProfile::LegacyFp32,
        )
        .expect("legacy sparse model");
        let detections = legacy
            .detect_limited(&frame, 16)
            .expect("legacy runtime tensors satisfy the declared contract");
        assert!(detections.len() <= 16);
    }

    #[test]
    fn superpoint_interface_rejects_missing_partial_and_wrong_typed_models() {
        let float_input = [image_outlet(TensorElementType::Float32)];
        let wrong_input = [image_outlet(TensorElementType::Int32)];
        let wrong_shape_input = [tensor_outlet_with_shape(
            "image",
            TensorElementType::Float32,
            [1_i64, 3, -1, -1],
        )];
        let sparse = canonical_sparse_outlets(TensorElementType::Int64);
        let partial_dense = [tensor_outlet_with_shape(
            "dense_scores",
            TensorElementType::Float32,
            [1_i64, 65, -1, -1],
        )];
        let wrong_sparse = canonical_sparse_outlets(TensorElementType::Float32);
        let wrong_score_type = [
            tensor_outlet_with_shape("keypoints", TensorElementType::Int64, [1_i64, -1, 2]),
            tensor_outlet_with_shape("scores", TensorElementType::Float64, [1_i64, -1]),
            tensor_outlet_with_shape("descriptors", TensorElementType::Float32, [-1_i64, -1, -1]),
        ];
        let wrong_descriptor_type = [
            tensor_outlet_with_shape("keypoints", TensorElementType::Int64, [1_i64, -1, 2]),
            tensor_outlet_with_shape("scores", TensorElementType::Float32, [1_i64, -1]),
            tensor_outlet_with_shape("descriptors", TensorElementType::Float64, [-1_i64, -1, -1]),
        ];
        let wrong_layout = [
            tensor_outlet_with_shape("keypoints", TensorElementType::Int64, [1_i64, 2, -1]),
            tensor_outlet_with_shape("scores", TensorElementType::Float32, [1_i64, -1]),
            tensor_outlet_with_shape("descriptors", TensorElementType::Float32, [-1_i64, -1, -1]),
        ];

        for error in [
            parse_superpoint_model_interface(&[], &sparse, SuperPointSparseProfile::CanonicalFp16)
                .expect_err("missing input"),
            parse_superpoint_model_interface(
                &wrong_input,
                &sparse,
                SuperPointSparseProfile::CanonicalFp16,
            )
            .expect_err("wrong input"),
            parse_superpoint_model_interface(
                &wrong_shape_input,
                &sparse,
                SuperPointSparseProfile::CanonicalFp16,
            )
            .expect_err("wrong input shape"),
            parse_superpoint_model_interface(
                &float_input,
                &partial_dense,
                SuperPointSparseProfile::CanonicalFp16,
            )
            .expect_err("partial dense outputs"),
            parse_superpoint_model_interface(
                &float_input,
                &wrong_sparse,
                SuperPointSparseProfile::CanonicalFp16,
            )
            .expect_err("wrong sparse keypoints"),
            parse_superpoint_model_interface(
                &float_input,
                &wrong_score_type,
                SuperPointSparseProfile::CanonicalFp16,
            )
            .expect_err("wrong sparse score type"),
            parse_superpoint_model_interface(
                &float_input,
                &wrong_descriptor_type,
                SuperPointSparseProfile::CanonicalFp16,
            )
            .expect_err("wrong sparse descriptor type"),
            parse_superpoint_model_interface(
                &float_input,
                &wrong_layout,
                SuperPointSparseProfile::CanonicalFp16,
            )
            .expect_err("wrong sparse layout"),
        ] {
            assert!(matches!(
                error,
                InferenceError::UnsupportedModelInterface {
                    model: "SuperPoint",
                    ..
                }
            ));
        }
    }

    #[test]
    fn input_domain_preserves_typed_dimensions_and_model_minimum() {
        for (width, height) in [(7, 8), (8, 7)] {
            let dimensions = FrameDimensions::try_new(width, height).expect("nonzero dimensions");
            assert!(matches!(
                SuperPointInputDomain::try_new(dimensions, dimensions, None),
                Err(InferenceError::InputDimensionsTooSmall {
                    model: "superpoint",
                    width: actual_width,
                    height: actual_height,
                    minimum: MODEL_STRIDE,
                }) if actual_width == width && actual_height == height
            ));
        }

        for (width, height) in [(8, 8), (15, 15), (27, 19)] {
            let dimensions = FrameDimensions::try_new(width, height).expect("valid dimensions");
            SuperPointInputDomain::try_new(dimensions, dimensions, None)
                .expect("supported nonzero feature grid");
        }
    }

    #[test]
    fn canonical_outputs_preserve_integer_xy_and_row_alignment() {
        let mut descriptor_data = vec![0.125; DESCRIPTOR_DIM];
        descriptor_data.extend(std::iter::repeat_n(-0.5, DESCRIPTOR_DIM));
        let parsed = parse_original_scale(
            &[1, 2, 2],
            &[4, 4, 17, 9],
            &[1, 2],
            &[1.0, 0.5],
            &[1, 2, DESCRIPTOR_DIM as i64],
            &descriptor_data,
        )
        .expect("canonical sparse output");

        assert_eq!((parsed.keypoints[0].x, parsed.keypoints[0].y), (4.0, 4.0));
        assert_eq!((parsed.keypoints[1].x, parsed.keypoints[1].y), (17.0, 9.0));
        assert_eq!(parsed.scores, [1.0, 0.5]);
        assert_eq!(parsed.descriptors[0].as_slice(), &[0.125; DESCRIPTOR_DIM]);
        assert_eq!(parsed.descriptors[1].as_slice(), &[-0.5; DESCRIPTOR_DIM]);
    }

    #[test]
    fn canonical_empty_output_requires_exact_empty_shapes() {
        let parsed = parse_original_scale(
            &[1, 0, 2],
            &[],
            &[1, 0],
            &[],
            &[1, 0, DESCRIPTOR_DIM as i64],
            &[],
        )
        .expect("canonical empty output");
        assert!(parsed.keypoints.is_empty());
        assert!(parsed.scores.is_empty());
        assert!(parsed.descriptors.is_empty());

        assert_output_error(
            parse_original_scale(
                &[0, 1, 2],
                &[],
                &[1, 0],
                &[],
                &[1, 0, DESCRIPTOR_DIM as i64],
                &[],
            ),
            "keypoints",
        );
    }

    #[test]
    fn output_parser_requires_exact_aligned_shapes_and_lengths() {
        let descriptors = vec![0.0; DESCRIPTOR_DIM];
        for shape in [&[1, 2][..], &[2, 1, 2], &[1, 1, 2, 1], &[1, 1, 3]] {
            assert_output_error(
                parse_original_scale(
                    shape,
                    &[4, 4],
                    &[1, 1],
                    &[0.5],
                    &[1, 1, DESCRIPTOR_DIM as i64],
                    &descriptors,
                ),
                "keypoints",
            );
        }
        assert_output_error(
            parse_original_scale(
                &[1, 1, 2],
                &[4],
                &[1, 1],
                &[0.5],
                &[1, 1, DESCRIPTOR_DIM as i64],
                &descriptors,
            ),
            "keypoints",
        );

        for (shape, data) in [
            (&[1, 1, 1][..], &[0.5][..]),
            (&[1, 2][..], &[0.5, 0.25][..]),
            (&[1, 1][..], &[][..]),
        ] {
            assert_output_error(
                parse_original_scale(
                    &[1, 1, 2],
                    &[4, 4],
                    shape,
                    data,
                    &[1, 1, DESCRIPTOR_DIM as i64],
                    &descriptors,
                ),
                "scores",
            );
        }

        for (shape, data) in [
            (&[1, DESCRIPTOR_DIM as i64, 1][..], descriptors.as_slice()),
            (
                &[1, 1, (DESCRIPTOR_DIM - 1) as i64][..],
                descriptors.as_slice(),
            ),
            (&[1, DESCRIPTOR_DIM as i64][..], descriptors.as_slice()),
            (
                &[1, 1, DESCRIPTOR_DIM as i64][..],
                &descriptors[..DESCRIPTOR_DIM - 1],
            ),
        ] {
            assert_output_error(
                parse_original_scale(&[1, 1, 2], &[4, 4], &[1, 1], &[0.5], shape, data),
                "descriptors",
            );
        }
    }

    #[test]
    fn output_keypoint_count_accepts_2048_and_rejects_2049_before_copying() {
        let count = CANONICAL_CONTRACT.capacity;
        let keypoints: Vec<i64> = std::iter::repeat_n([4_i64, 4], count).flatten().collect();
        let scores = vec![1.0; count];
        let descriptors = vec![0.0; count * DESCRIPTOR_DIM];
        let parsed = parse_original_scale(
            &[1, count as i64, 2],
            &keypoints,
            &[1, count as i64],
            &scores,
            &[1, count as i64, DESCRIPTOR_DIM as i64],
            &descriptors,
        )
        .expect("active 2048-row capacity");
        assert_eq!(parsed.keypoints.len(), CANONICAL_CONTRACT.capacity);

        let rejected = (CANONICAL_CONTRACT.capacity + 1) as i64;
        assert_output_error(
            parse_original_scale(
                &[1, rejected, 2],
                &[],
                &[1, rejected],
                &[],
                &[1, rejected, DESCRIPTOR_DIM as i64],
                &[],
            ),
            "keypoints",
        );
    }

    #[test]
    fn limited_parser_validates_all_rows_but_constructs_only_sorted_prefix() {
        let dimensions = FrameDimensions::try_new(640, 480).expect("valid dimensions");
        let descriptors = vec![0.0; 2 * DESCRIPTOR_DIM];
        let parsed = parse_fixture(
            &[1, 2, 2],
            &[4, 4, 17, 9],
            &[1, 2],
            &[0.75, 0.5],
            &[1, 2, DESCRIPTOR_DIM as i64],
            &descriptors,
            dimensions,
            dimensions,
            None,
            1,
        )
        .expect("validated TopK prefix");
        assert_eq!(parsed.keypoints.len(), 1);
        assert_eq!((parsed.keypoints[0].x, parsed.keypoints[0].y), (4.0, 4.0));
        assert_eq!(parsed.scores, [0.75]);
        assert_eq!(parsed.descriptors.len(), 1);

        assert_output_error(
            parse_fixture(
                &[1, 2, 2],
                &[4, 4, 3, 9],
                &[1, 2],
                &[0.75, 0.5],
                &[1, 2, DESCRIPTOR_DIM as i64],
                &descriptors,
                dimensions,
                dimensions,
                None,
                1,
            ),
            "keypoints",
        );

        let mut invalid_tail = descriptors;
        invalid_tail[DESCRIPTOR_DIM + 7] = f32::NAN;
        let error = parse_fixture(
            &[1, 2, 2],
            &[4, 4, 17, 9],
            &[1, 2],
            &[0.75, 0.5],
            &[1, 2, DESCRIPTOR_DIM as i64],
            &invalid_tail,
            dimensions,
            dimensions,
            None,
            1,
        )
        .expect_err("discarded descriptor rows are still validated");
        assert!(matches!(
            error,
            InferenceError::DescriptorOutput {
                descriptor_index: 1,
                source: crate::DescriptorError::NonFiniteComponent { index: 7, value },
                ..
            } if value.is_nan()
        ));
    }

    #[test]
    fn coordinate_parser_enforces_effective_grid_border_and_top_left_scaling() {
        let model_dimensions = FrameDimensions::try_new(27, 19).expect("valid dimensions");
        let descriptors = vec![0.0; 2 * DESCRIPTOR_DIM];
        let accepted = parse_fixture(
            &[1, 2, 2],
            &[4, 4, 19, 11],
            &[1, 2],
            &[1.0, 0.5],
            &[1, 2, DESCRIPTOR_DIM as i64],
            &descriptors,
            model_dimensions,
            model_dimensions,
            None,
            usize::MAX,
        )
        .expect("inclusive lower and exclusive upper graph bounds");
        assert_eq!(
            (accepted.keypoints[1].x, accepted.keypoints[1].y),
            (19.0, 11.0)
        );

        for [x, y] in [
            [-1, 4],
            [i64::MAX, 4],
            [3, 4],
            [20, 4],
            [26, 4],
            [4, 3],
            [4, 12],
            [4, 18],
        ] {
            assert_output_error(
                parse_fixture(
                    &[1, 1, 2],
                    &[x, y],
                    &[1, 1],
                    &[0.5],
                    &[1, 1, DESCRIPTOR_DIM as i64],
                    &[0.0; DESCRIPTOR_DIM],
                    model_dimensions,
                    model_dimensions,
                    None,
                    usize::MAX,
                ),
                "keypoints",
            );
        }

        let model_dimensions = FrameDimensions::try_new(320, 240).expect("valid dimensions");
        let frame_dimensions = FrameDimensions::try_new(640, 480).expect("valid dimensions");
        let factor = DownscaleFactor::try_from(2).expect("nonzero scale");
        let scaled = parse_fixture(
            &[1, 1, 2],
            &[315, 235],
            &[1, 1],
            &[0.5],
            &[1, 1, DESCRIPTOR_DIM as i64],
            &[0.0; DESCRIPTOR_DIM],
            model_dimensions,
            frame_dimensions,
            Some(factor),
            usize::MAX,
        )
        .expect("integer top-left mapping");
        assert_eq!(
            (scaled.keypoints[0].x, scaled.keypoints[0].y),
            (630.0, 470.0)
        );
    }

    #[test]
    fn coordinate_parser_rejects_unrepresentable_scaled_f32_pixel() {
        const EXACT_MODEL_X: i64 = 5_592_406;
        const INEXACT_MODEL_X: i64 = EXACT_MODEL_X + 1;
        let model_dimensions =
            FrameDimensions::try_new(5_592_424, 16).expect("valid model dimensions");
        let frame_dimensions =
            FrameDimensions::try_new(16_777_272, 48).expect("valid frame dimensions");
        let factor = DownscaleFactor::try_from(3).expect("nonzero scale");
        let descriptors = [0.0; DESCRIPTOR_DIM];

        parse_fixture(
            &[1, 1, 2],
            &[EXACT_MODEL_X, 4],
            &[1, 1],
            &[0.5],
            &[1, 1, DESCRIPTOR_DIM as i64],
            &descriptors,
            model_dimensions,
            frame_dimensions,
            Some(factor),
            usize::MAX,
        )
        .expect("representable scaled pixel");

        assert!(matches!(
            parse_fixture(
                &[1, 1, 2],
                &[INEXACT_MODEL_X, 4],
                &[1, 1],
                &[0.5],
                &[1, 1, DESCRIPTOR_DIM as i64],
                &descriptors,
                model_dimensions,
                frame_dimensions,
                Some(factor),
                usize::MAX,
            ),
            Err(InferenceError::KeypointCoordinateUnrepresentable {
                model: "superpoint",
                index: 0,
                axis: "x",
                coordinate: 16_777_221,
            })
        ));
    }

    #[test]
    fn score_parser_enforces_exact_graph_threshold_range_and_topk_order() {
        let threshold = CANONICAL_CONTRACT.score_threshold;
        assert_eq!(threshold.to_bits(), 0x3a03_2000);
        let descriptors = [0.0; DESCRIPTOR_DIM];
        for score in [
            f32::NAN,
            f32::NEG_INFINITY,
            f32::INFINITY,
            -f32::from_bits(1),
            threshold,
            f32::from_bits(threshold.to_bits() - 1),
            f32::from_bits(1.0_f32.to_bits() + 1),
        ] {
            assert_output_error(
                parse_original_scale(
                    &[1, 1, 2],
                    &[4, 4],
                    &[1, 1],
                    &[score],
                    &[1, 1, DESCRIPTOR_DIM as i64],
                    &descriptors,
                ),
                "scores",
            );
        }
        for score in [f32::from_bits(threshold.to_bits() + 1), 1.0] {
            parse_original_scale(
                &[1, 1, 2],
                &[4, 4],
                &[1, 1],
                &[score],
                &[1, 1, DESCRIPTOR_DIM as i64],
                &descriptors,
            )
            .expect("valid selected confidence");
        }

        assert_output_error(
            parse_original_scale(
                &[1, 2, 2],
                &[4, 4, 5, 5],
                &[1, 2],
                &[0.5, 0.75],
                &[1, 2, DESCRIPTOR_DIM as i64],
                &[0.0; 2 * DESCRIPTOR_DIM],
            ),
            "scores",
        );
    }

    #[test]
    fn legacy_profile_retains_its_exact_threshold_and_512_row_capacity() {
        assert_eq!(LEGACY_CONTRACT.score_threshold.to_bits(), 0x3a03_126f);
        assert_eq!(LEGACY_CONTRACT.capacity, 512);
        let dimensions = FrameDimensions::try_new(640, 480).expect("valid dimensions");
        let legacy_only_score = f32::from_bits(LEGACY_CONTRACT.score_threshold.to_bits() + 1);

        parse_fixture_with_contract(
            &[1, 1, 2],
            &[4, 4],
            &[1, 1],
            &[legacy_only_score],
            &[1, 1, DESCRIPTOR_DIM as i64],
            &[0.0; DESCRIPTOR_DIM],
            dimensions,
            dimensions,
            None,
            usize::MAX,
            LEGACY_CONTRACT,
        )
        .expect("score selected by the legacy f32 graph");
        assert_output_error(
            parse_fixture(
                &[1, 1, 2],
                &[4, 4],
                &[1, 1],
                &[legacy_only_score],
                &[1, 1, DESCRIPTOR_DIM as i64],
                &[0.0; DESCRIPTOR_DIM],
                dimensions,
                dimensions,
                None,
                usize::MAX,
            ),
            "scores",
        );

        let rejected = (LEGACY_CONTRACT.capacity + 1) as i64;
        assert_output_error(
            parse_fixture_with_contract(
                &[1, rejected, 2],
                &[],
                &[1, rejected],
                &[],
                &[1, rejected, DESCRIPTOR_DIM as i64],
                &[],
                dimensions,
                dimensions,
                None,
                usize::MAX,
                LEGACY_CONTRACT,
            ),
            "keypoints",
        );
    }

    #[test]
    fn dense_descriptor_normalization_preserves_regular_and_small_norm_behavior() {
        let regular: Vec<f32> = (1..=DESCRIPTOR_DIM).map(|value| value as f32).collect();
        let descriptor = sample_dense_descriptor(&regular, 1, 1, 8, 8, 3.5, 3.5)
            .expect("regular dense descriptor");
        let direct_norm = regular
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        for (&actual, &input) in descriptor.as_slice().iter().zip(&regular) {
            assert!((actual - input / direct_norm).abs() < 1e-6);
        }

        let small = [1e-8_f32; DESCRIPTOR_DIM];
        let descriptor = sample_dense_descriptor(&small, 1, 1, 8, 8, 3.5, 3.5)
            .expect("small-norm dense descriptor");
        for &value in descriptor.as_slice() {
            assert!((value - 0.01).abs() < 1e-7);
        }
    }

    #[test]
    fn dense_descriptor_normalization_does_not_overflow_on_extreme_finite_input() {
        let descriptor = sample_dense_descriptor(&[f32::MAX; DESCRIPTOR_DIM], 1, 1, 8, 8, 3.5, 3.5)
            .expect("scaled normalization keeps finite dense output representable");

        let expected = (DESCRIPTOR_DIM as f32).sqrt().recip();
        assert!(descriptor.as_slice().iter().all(|value| value.is_finite()));
        for &value in descriptor.as_slice() {
            assert!((value - expected).abs() < 1e-6);
        }
    }
}
