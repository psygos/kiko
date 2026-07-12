use super::{
    InferenceBackend, InferenceError, InferenceRunDiagnostics, build_session, inference_env,
};
use crate::{Descriptor, Detections, DownscaleFactor, Frame, Keypoint};
use ort::session::Session;
use ort::value::PrimitiveTensorElementType;
use ort::value::TensorElementType;
use ort::value::TensorRef;
use ort::value::ValueType;
use std::num::NonZeroUsize;
use std::path::Path;

use crate::DESCRIPTOR_DIM;

pub struct SuperPoint {
    session: Session,
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
    SparseOutputs,
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
        let kind = if session
            .outputs()
            .iter()
            .any(|output| output.name() == "dense_scores")
            && session
                .outputs()
                .iter()
                .any(|output| output.name() == "dense_descriptors")
        {
            SuperPointModelKind::DenseHeads
        } else {
            SuperPointModelKind::SparseOutputs
        };
        let input_kind = match session.inputs().first().map(|input| input.dtype()) {
            Some(ValueType::Tensor {
                ty: TensorElementType::Uint8,
                ..
            }) => SuperPointInputKind::Uint8,
            _ => SuperPointInputKind::Float32,
        };
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
                let expected = (frame.width() as usize) * (frame.height() as usize);
                self.scratch.resize(expected, 0.0);
                crate::preprocess::normalise_into(frame.data(), &mut self.scratch)?;

                let input_tensor = TensorRef::from_array_view((
                    [1, 1, frame.height() as usize, frame.width() as usize],
                    self.scratch.as_slice(),
                ))?;
                run_with_tensor(
                    self.kind,
                    &mut self.session,
                    frame,
                    input_tensor,
                    frame.width(),
                    frame.height(),
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
                ))?;
                run_with_tensor(
                    self.kind,
                    &mut self.session,
                    frame,
                    input_tensor,
                    frame.width(),
                    frame.height(),
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
                    frame.data(),
                    frame.width(),
                    frame.height(),
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
                ))?;
                run_with_tensor(
                    self.kind,
                    &mut self.session,
                    frame,
                    input_tensor,
                    dimensions.width(),
                    dimensions.height(),
                    Some(downscale),
                    &mut self.candidates,
                    &mut self.dense_score_map,
                    max_keypoints,
                    self.dense_candidate_cap,
                    self.diagnostics,
                )
            }
            SuperPointInputKind::Uint8 => {
                let dimensions = crate::preprocess::downscale_u8_into(
                    frame.data(),
                    frame.width(),
                    frame.height(),
                    downscale,
                    &mut self.scratch_u8,
                )
                .map_err(InferenceError::from)?;

                let input_tensor = TensorRef::from_array_view((
                    [
                        1,
                        1,
                        dimensions.height() as usize,
                        dimensions.width() as usize,
                    ],
                    self.scratch_u8.as_slice(),
                ))?;
                run_with_tensor(
                    self.kind,
                    &mut self.session,
                    frame,
                    input_tensor,
                    dimensions.width(),
                    dimensions.height(),
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

// Keep the session, frame metadata, tensor, and reusable scratch buffers explicit.
#[allow(clippy::too_many_arguments)]
fn run_with_tensor<T>(
    kind: SuperPointModelKind,
    session: &mut Session,
    frame: &Frame,
    input_tensor: TensorRef<'_, T>,
    width: u32,
    height: u32,
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
        SuperPointModelKind::SparseOutputs => run_sparse_inference(
            session,
            frame,
            input_tensor,
            width,
            height,
            downscale,
            diagnostics,
        ),
        SuperPointModelKind::DenseHeads => run_dense_inference(
            session,
            frame,
            input_tensor,
            width,
            height,
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
    frame: &Frame,
    input_tensor: TensorRef<'_, T>,
    width: u32,
    height: u32,
    downscale: Option<DownscaleFactor>,
    diagnostics: InferenceRunDiagnostics,
) -> Result<Detections, InferenceError>
where
    T: PrimitiveTensorElementType + std::fmt::Debug,
{
    let outputs = super::run_with_slow_call_diagnostics(diagnostics, "superpoint", || {
        session
            .run(ort::inputs!["image" => input_tensor])
            .map_err(InferenceError::Execution)
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
        _ => scores_value.try_extract_tensor::<f32>()?.1.to_vec(),
    };
    let keypoints_pairs = match tensor_num_elements(keypoints_value, "keypoints")? {
        0 => Vec::new(),
        _ => {
            if let Ok((shape, data)) = keypoints_value.try_extract_tensor::<f32>() {
                parse_keypoint_pairs(shape, data, "keypoints")?
            } else if let Ok((shape, data)) = keypoints_value.try_extract_tensor::<i64>() {
                let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
                parse_keypoint_pairs(shape, &data_f32, "keypoints")?
            } else {
                return Err(InferenceError::UnexpectedOutput {
                    name: "keypoints".to_string(),
                    expected: "tensor of f32 or i64".to_string(),
                    actual: format!("{:?}", keypoints_value.dtype()),
                });
            }
        }
    };
    let mut keypoints = to_keypoints(&keypoints_pairs, width as f32, height as f32);
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
            let descriptors_raw = descriptors_value.try_extract_tensor::<f32>()?;
            parse_descriptors(descriptors_raw.1, "descriptors")?
        }
    };

    Detections::new(
        frame.sensor_id(),
        frame.frame_id(),
        frame.width(),
        frame.height(),
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
    frame: &Frame,
    input_tensor: TensorRef<'_, T>,
    width: u32,
    height: u32,
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
            .run(ort::inputs!["image" => input_tensor])
            .map_err(InferenceError::Execution)
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

    let (scores_shape, scores_data) = scores_value.try_extract_tensor::<f32>()?;
    let (desc_shape, desc_data) = descriptors_value.try_extract_tensor::<f32>()?;
    let (grid_h, grid_w) = dense_scores_shape(scores_shape, scores_data.len())?;
    dense_descriptors_shape(desc_shape, desc_data.len(), grid_h, grid_w)?;

    collect_dense_candidates(
        scores_data,
        grid_h,
        grid_w,
        width,
        height,
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
    for candidate in candidates.iter().copied() {
        let mut x = candidate.x as f32;
        let mut y = candidate.y as f32;
        if let Some(scale) = downscale {
            let factor = scale.get() as f32;
            x *= factor;
            y *= factor;
        }
        keypoints.push(Keypoint { x, y });
        scores.push(candidate.score);
        descriptors.push(sample_dense_descriptor(
            desc_data,
            grid_h,
            grid_w,
            width,
            height,
            candidate.x as f32,
            candidate.y as f32,
        ));
    }

    Detections::new(
        frame.sensor_id(),
        frame.frame_id(),
        frame.width(),
        frame.height(),
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
) -> Descriptor {
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

    let mut out = [0.0_f32; DESCRIPTOR_DIM];
    let mut norm2 = 0.0_f32;
    for (channel, value) in out.iter_mut().enumerate() {
        let base = channel * grid_h * grid_w;
        let sample = w00 * desc[base + y0 * grid_w + x0]
            + w10 * desc[base + y0 * grid_w + x1]
            + w01 * desc[base + y1 * grid_w + x0]
            + w11 * desc[base + y1 * grid_w + x1];
        *value = sample;
        norm2 += sample * sample;
    }
    let inv_norm = norm2.max(1e-12).sqrt().recip();
    for value in &mut out {
        *value *= inv_norm;
    }
    Descriptor(out)
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
    for chunk in data.chunks_exact(DESCRIPTOR_DIM) {
        let mut descriptor = [0.0_f32; DESCRIPTOR_DIM];
        descriptor.copy_from_slice(chunk);
        descriptors.push(Descriptor(descriptor));
    }
    Ok(descriptors)
}

#[derive(Clone, Copy, Debug)]
enum Normalization {
    None,
    ZeroToOne,
    NegOneToOne,
}

fn parse_keypoint_pairs(
    shape: &ort::value::Shape,
    data: &[f32],
    output_name: &str,
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
            pairs.push([data[2 * i], data[2 * i + 1]]);
        }
        return Ok(pairs);
    }

    if dims.first().copied() == Some(2) {
        let (first, second) = data.split_at(count);
        for i in 0..count {
            pairs.push([first[i], second[i]]);
        }
        return Ok(pairs);
    }

    Err(InferenceError::UnexpectedOutput {
        name: output_name.to_string(),
        expected: "tensor with a leading or trailing dimension of size 2".to_string(),
        actual: format!("{shape}"),
    })
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

    #[test]
    fn parse_descriptors_accepts_complete_chunks() {
        let data: Vec<f32> = (0..(DESCRIPTOR_DIM * 2)).map(|v| v as f32).collect();
        let descriptors = parse_descriptors(&data, "descriptors").expect("complete chunks");

        assert_eq!(descriptors.len(), 2);
        assert_eq!(descriptors[0].0[0], 0.0);
        assert_eq!(descriptors[1].0[0], DESCRIPTOR_DIM as f32);
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
