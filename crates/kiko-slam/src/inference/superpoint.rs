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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SuperPointModelKind {
    SparseOutputs(SparseKeypointKind),
    DenseHeads,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SparseKeypointKind {
    Float32,
    Int64,
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

impl SuperPoint {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, InferenceError> {
        Self::new_with_backend(path, InferenceBackend::auto())
    }

    pub fn new_with_backend(
        path: impl AsRef<Path>,
        backend: InferenceBackend,
    ) -> Result<Self, InferenceError> {
        let path = path.as_ref();
        let (session, selected, diagnostics) = build_session(path, backend)?;
        let run_options = build_run_options(selected)?;
        let (kind, input_kind) =
            parse_superpoint_model_interface(session.inputs(), session.outputs())?;
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

const SUPERPOINT_INTERFACE_DESCRIPTION: &str = "an image tensor input [1, 1, height, width] (dynamic dimensions allowed) of f32 or u8 and either sparse outputs keypoints:(f32|i64), scores:f32, descriptors:f32 or dense outputs dense_scores:f32, dense_descriptors:f32";

fn parse_superpoint_model_interface(
    inputs: &[Outlet],
    outputs: &[Outlet],
) -> Result<(SuperPointModelKind, SuperPointInputKind), InferenceError> {
    let input_kind = match inputs
        .iter()
        .find(|outlet| outlet.name() == "image")
        .map(Outlet::dtype)
    {
        Some(ValueType::Tensor {
            ty: TensorElementType::Float32,
            shape,
            ..
        }) if superpoint_input_shape_supported(shape) => SuperPointInputKind::Float32,
        Some(ValueType::Tensor {
            ty: TensorElementType::Uint8,
            shape,
            ..
        }) if superpoint_input_shape_supported(shape) => SuperPointInputKind::Uint8,
        _ => return Err(unsupported_superpoint_interface(inputs, outputs)),
    };

    let kind = if outlet_tensor_element_type(outputs, "dense_scores")
        == Some(TensorElementType::Float32)
        && outlet_tensor_element_type(outputs, "dense_descriptors")
            == Some(TensorElementType::Float32)
    {
        SuperPointModelKind::DenseHeads
    } else {
        let keypoint_kind = match outlet_tensor_element_type(outputs, "keypoints") {
            Some(TensorElementType::Float32) => SparseKeypointKind::Float32,
            Some(TensorElementType::Int64) => SparseKeypointKind::Int64,
            _ => return Err(unsupported_superpoint_interface(inputs, outputs)),
        };
        if outlet_tensor_element_type(outputs, "scores") != Some(TensorElementType::Float32)
            || outlet_tensor_element_type(outputs, "descriptors")
                != Some(TensorElementType::Float32)
        {
            return Err(unsupported_superpoint_interface(inputs, outputs));
        }
        SuperPointModelKind::SparseOutputs(keypoint_kind)
    };

    Ok((kind, input_kind))
}

fn superpoint_input_shape_supported(shape: &ort::value::Shape) -> bool {
    shape.len() == 4
        && matches!(shape[0], -1 | 1)
        && matches!(shape[1], -1 | 1)
        && (shape[2] == -1 || shape[2] > 0)
        && (shape[3] == -1 || shape[3] > 0)
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
        SuperPointModelKind::SparseOutputs(keypoint_kind) => run_sparse_inference(
            session,
            run_options,
            frame,
            input_tensor,
            keypoint_kind,
            input_dimensions,
            downscale,
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

fn run_sparse_inference<T>(
    session: &mut Session,
    run_options: &RunOptions,
    frame: &Frame,
    input_tensor: TensorRef<'_, T>,
    keypoint_kind: SparseKeypointKind,
    input_dimensions: FrameDimensions,
    downscale: Option<DownscaleFactor>,
    diagnostics: InferenceRunDiagnostics,
) -> Result<Detections, InferenceError>
where
    T: PrimitiveTensorElementType + std::fmt::Debug,
{
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

    // `ort` may return a null backing pointer for empty tensors; avoid raw extraction in that case.
    let scores = match tensor_num_elements(scores_value, "scores")? {
        0 => Vec::new(),
        _ => super::extract_tensor::<f32>(scores_value, "scores")?
            .1
            .to_vec(),
    };
    let keypoints_pairs = match tensor_num_elements(keypoints_value, "keypoints")? {
        0 => Vec::new(),
        _ => match keypoint_kind {
            SparseKeypointKind::Float32 => {
                let (shape, data) = super::extract_tensor::<f32>(keypoints_value, "keypoints")?;
                parse_keypoint_pairs(shape, data, "keypoints", |index, value| {
                    finite_keypoint_coordinate("keypoints", index, value)
                })?
            }
            SparseKeypointKind::Int64 => {
                let (shape, data) = super::extract_tensor::<i64>(keypoints_value, "keypoints")?;
                parse_keypoint_pairs(shape, data, "keypoints", |index, value| {
                    super::exact_i64_output_f32("keypoints", index, value)
                })?
            }
        },
    };
    let mut keypoints = to_keypoints(
        &keypoints_pairs,
        input_dimensions.width() as f32,
        input_dimensions.height() as f32,
    );
    if let Some(scale) = downscale {
        let factor = scale.get() as f32;
        for kp in &mut keypoints {
            kp.x *= factor;
            kp.y *= factor;
        }
    }
    let descriptors = match tensor_num_elements(descriptors_value, "descriptors")? {
        0 => Vec::new(),
        _ => {
            let descriptors_raw = super::extract_tensor::<f32>(descriptors_value, "descriptors")?;
            parse_descriptors(descriptors_raw.1, "descriptors")?
        }
    };

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

fn tensor_num_elements(
    value: &ort::value::Value<ort::value::DynValueTypeMarker>,
    output_name: &str,
) -> Result<usize, InferenceError> {
    match value.dtype() {
        ValueType::Tensor { shape, .. } => Ok(shape.num_elements()),
        dtype => Err(InferenceError::UnexpectedOutput {
            name: output_name.to_string(),
            expected: "tensor output".to_string(),
            actual: format!("{dtype:?}"),
        }),
    }
}

fn parse_descriptors(data: &[f32], output_name: &str) -> Result<Vec<Descriptor>, InferenceError> {
    if data.len() % DESCRIPTOR_DIM != 0 {
        return Err(InferenceError::UnexpectedOutput {
            name: output_name.to_string(),
            expected: format!(
                "tensor with element count divisible by {DESCRIPTOR_DIM} (descriptor dimension)"
            ),
            actual: format!("tensor with {} elements", data.len()),
        });
    }

    let mut descriptors = Vec::with_capacity(data.len() / DESCRIPTOR_DIM);
    for (descriptor_index, chunk) in data.chunks_exact(DESCRIPTOR_DIM).enumerate() {
        let mut descriptor = [0.0_f32; DESCRIPTOR_DIM];
        descriptor.copy_from_slice(chunk);
        descriptors.push(Descriptor::try_new(descriptor).map_err(|source| {
            InferenceError::DescriptorOutput {
                name: output_name.to_string(),
                descriptor_index,
                source,
            }
        })?);
    }
    Ok(descriptors)
}

#[derive(Clone, Copy, Debug)]
enum Normalization {
    None,
    ZeroToOne,
    NegOneToOne,
}

fn parse_keypoint_pairs<T: Copy>(
    shape: &ort::value::Shape,
    data: &[T],
    output_name: &str,
    mut parse_coordinate: impl FnMut(usize, T) -> Result<f32, InferenceError>,
) -> Result<Vec<[f32; 2]>, InferenceError> {
    let expected_len = shape.num_elements();
    if expected_len != 0 && expected_len != data.len() {
        return Err(InferenceError::UnexpectedOutput {
            name: output_name.to_string(),
            expected: format!("tensor with {expected_len} elements"),
            actual: format!("tensor with {} elements", data.len()),
        });
    }

    if data.len() % 2 != 0 {
        return Err(InferenceError::UnexpectedOutput {
            name: output_name.to_string(),
            expected: "even-sized tensor".to_string(),
            actual: format!("tensor with {} elements", data.len()),
        });
    }

    let dims = &shape[..];
    let count = data.len() / 2;
    let mut pairs = Vec::with_capacity(count);

    if dims.last().copied() == Some(2) {
        for i in 0..count {
            pairs.push([
                parse_coordinate(2 * i, data[2 * i])?,
                parse_coordinate(2 * i + 1, data[2 * i + 1])?,
            ]);
        }
        return Ok(pairs);
    }

    if dims.first().copied() == Some(2) {
        let (first, second) = data.split_at(count);
        for i in 0..count {
            pairs.push([
                parse_coordinate(i, first[i])?,
                parse_coordinate(count + i, second[i])?,
            ]);
        }
        return Ok(pairs);
    }

    Err(InferenceError::UnexpectedOutput {
        name: output_name.to_string(),
        expected: "tensor with a leading or trailing dimension of size 2".to_string(),
        actual: format!("{shape}"),
    })
}

fn finite_keypoint_coordinate(
    output_name: &str,
    index: usize,
    value: f32,
) -> Result<f32, InferenceError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(InferenceError::UnexpectedOutput {
            name: output_name.to_string(),
            expected: "finite keypoint coordinates".to_string(),
            actual: format!("coordinate {index} is {value}"),
        })
    }
}

fn extract_xy(
    pair: &[f32; 2],
    width: f32,
    height: f32,
    norm: Normalization,
    swap: bool,
) -> (f32, f32) {
    if swap {
        (
            scale_coordinate(pair[1], width, norm),
            scale_coordinate(pair[0], height, norm),
        )
    } else {
        (
            scale_coordinate(pair[0], width, norm),
            scale_coordinate(pair[1], height, norm),
        )
    }
}

fn to_keypoints(pairs: &[[f32; 2]], width: f32, height: f32) -> Vec<Keypoint> {
    let norm = detect_normalization(pairs);
    let score_xy = count_in_bounds(pairs, width, height, norm, false);
    let score_yx = count_in_bounds(pairs, width, height, norm, true);
    let swap = score_yx > score_xy;

    pairs
        .iter()
        .map(|pair| {
            let (x, y) = extract_xy(pair, width, height, norm, swap);
            Keypoint { x, y }
        })
        .collect()
}

fn detect_normalization(pairs: &[[f32; 2]]) -> Normalization {
    let mut min_value = f32::INFINITY;
    let mut max_value = f32::NEG_INFINITY;

    for [a, b] in pairs {
        min_value = min_value.min(*a).min(*b);
        max_value = max_value.max(*a).max(*b);
    }

    let epsilon = 1e-3_f32;
    if min_value >= -epsilon && max_value <= 1.0 + epsilon {
        return Normalization::ZeroToOne;
    }
    if min_value >= -1.0 - epsilon && max_value <= 1.0 + epsilon {
        return Normalization::NegOneToOne;
    }

    Normalization::None
}

fn count_in_bounds(
    pairs: &[[f32; 2]],
    width: f32,
    height: f32,
    norm: Normalization,
    swap: bool,
) -> usize {
    pairs
        .iter()
        .filter(|pair| {
            let (x, y) = extract_xy(pair, width, height, norm, swap);
            x >= 0.0 && x < width && y >= 0.0 && y < height
        })
        .count()
}

fn scale_coordinate(value: f32, dim: f32, norm: Normalization) -> f32 {
    let extent = if dim > 1.0 { dim - 1.0 } else { 1.0 };
    match norm {
        Normalization::None => value,
        Normalization::ZeroToOne => value * extent,
        Normalization::NegOneToOne => (value + 1.0) * 0.5 * extent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    fn tensor_outlet(name: &str, ty: TensorElementType) -> Outlet {
        tensor_outlet_with_shape(name, ty, [-1_i64, -1, -1, -1])
    }

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

    #[test]
    fn superpoint_interface_is_parsed_once_into_supported_kinds() {
        let float_input = [tensor_outlet("image", TensorElementType::Float32)];
        let uint8_input = [tensor_outlet("image", TensorElementType::Uint8)];
        let sparse_i64 = [
            tensor_outlet("keypoints", TensorElementType::Int64),
            tensor_outlet("scores", TensorElementType::Float32),
            tensor_outlet("descriptors", TensorElementType::Float32),
        ];
        let sparse_f32 = [
            tensor_outlet("keypoints", TensorElementType::Float32),
            tensor_outlet("scores", TensorElementType::Float32),
            tensor_outlet("descriptors", TensorElementType::Float32),
        ];
        let dense = [
            tensor_outlet("dense_scores", TensorElementType::Float32),
            tensor_outlet("dense_descriptors", TensorElementType::Float32),
        ];

        assert_eq!(
            parse_superpoint_model_interface(&float_input, &sparse_i64).expect("sparse i64"),
            (
                SuperPointModelKind::SparseOutputs(SparseKeypointKind::Int64),
                SuperPointInputKind::Float32
            )
        );
        assert_eq!(
            parse_superpoint_model_interface(&uint8_input, &sparse_f32).expect("sparse f32"),
            (
                SuperPointModelKind::SparseOutputs(SparseKeypointKind::Float32),
                SuperPointInputKind::Uint8
            )
        );
        assert_eq!(
            parse_superpoint_model_interface(&float_input, &dense).expect("dense"),
            (
                SuperPointModelKind::DenseHeads,
                SuperPointInputKind::Float32
            )
        );
    }

    #[test]
    fn superpoint_interface_rejects_missing_partial_and_wrong_typed_models() {
        let float_input = [tensor_outlet("image", TensorElementType::Float32)];
        let wrong_input = [tensor_outlet("image", TensorElementType::Int32)];
        let wrong_shape_input = [tensor_outlet_with_shape(
            "image",
            TensorElementType::Float32,
            [1_i64, 3, -1, -1],
        )];
        let sparse = [
            tensor_outlet("keypoints", TensorElementType::Int64),
            tensor_outlet("scores", TensorElementType::Float32),
            tensor_outlet("descriptors", TensorElementType::Float32),
        ];
        let partial_dense = [tensor_outlet("dense_scores", TensorElementType::Float32)];
        let wrong_sparse = [
            tensor_outlet("keypoints", TensorElementType::Int32),
            tensor_outlet("scores", TensorElementType::Float32),
            tensor_outlet("descriptors", TensorElementType::Float32),
        ];

        for error in [
            parse_superpoint_model_interface(&[], &sparse).expect_err("missing input"),
            parse_superpoint_model_interface(&wrong_input, &sparse).expect_err("wrong input"),
            parse_superpoint_model_interface(&wrong_shape_input, &sparse)
                .expect_err("wrong input shape"),
            parse_superpoint_model_interface(&float_input, &partial_dense)
                .expect_err("partial dense outputs"),
            parse_superpoint_model_interface(&float_input, &wrong_sparse)
                .expect_err("wrong sparse keypoints"),
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
    fn sparse_keypoint_parser_preserves_layout_and_rejects_nonfinite_coordinates() {
        let interleaved_shape = ort::value::Shape::new([2_i64, 2]);
        let interleaved = parse_keypoint_pairs(
            &interleaved_shape,
            &[1.0_f32, 2.0, 3.0, 4.0],
            "keypoints",
            |index, value| finite_keypoint_coordinate("keypoints", index, value),
        )
        .expect("interleaved pairs");
        assert_eq!(interleaved, [[1.0, 2.0], [3.0, 4.0]]);

        let planar_shape = ort::value::Shape::new([2_i64, 3]);
        let planar = parse_keypoint_pairs(
            &planar_shape,
            &[1_i64, 2, 3, 4, 5, 6],
            "keypoints",
            |index, value| super::super::exact_i64_output_f32("keypoints", index, value),
        )
        .expect("planar pairs");
        assert_eq!(planar, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);

        assert!(matches!(
            parse_keypoint_pairs(
                &interleaved_shape,
                &[1.0_f32, f32::NAN, 3.0, 4.0],
                "keypoints",
                |index, value| finite_keypoint_coordinate("keypoints", index, value),
            ),
            Err(InferenceError::UnexpectedOutput { name, actual, .. })
                if name == "keypoints" && actual.contains("coordinate 1")
        ));
    }

    #[test]
    fn parse_descriptors_accepts_complete_chunks() {
        let data: Vec<f32> = (0..(DESCRIPTOR_DIM * 2)).map(|v| v as f32).collect();
        let descriptors = parse_descriptors(&data, "descriptors").expect("complete chunks");

        assert_eq!(descriptors.len(), 2);
        assert_eq!(descriptors[0].as_slice()[0], 0.0);
        assert_eq!(descriptors[1].as_slice()[0], DESCRIPTOR_DIM as f32);
    }

    #[test]
    fn parse_descriptors_preserves_output_index_and_domain_source() {
        let mut data = vec![0.0_f32; DESCRIPTOR_DIM * 2];
        data[DESCRIPTOR_DIM + 7] = f32::NAN;

        let error = parse_descriptors(&data, "descriptors")
            .expect_err("nonfinite model output must fail at the descriptor boundary");
        assert!(matches!(
            &error,
            InferenceError::DescriptorOutput {
                name,
                descriptor_index: 1,
                source: crate::DescriptorError::NonFiniteComponent { index: 7, value },
            } if name == "descriptors" && value.is_nan()
        ));
        assert!(error.to_string().contains("descriptor 1"));
        assert!(error.to_string().contains("'descriptors'"));
        assert!(
            error
                .source()
                .expect("descriptor domain source")
                .to_string()
                .contains("component 7")
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

    #[test]
    fn parse_descriptors_rejects_partial_chunk() {
        let data: Vec<f32> = (0..(DESCRIPTOR_DIM + 1)).map(|v| v as f32).collect();
        let err = parse_descriptors(&data, "descriptors").expect_err("partial chunk");

        match err {
            InferenceError::UnexpectedOutput {
                name,
                expected,
                actual,
            } => {
                assert_eq!(name, "descriptors");
                assert!(expected.contains("divisible"));
                assert!(actual.contains(&(DESCRIPTOR_DIM + 1).to_string()));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
