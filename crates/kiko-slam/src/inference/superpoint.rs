use super::InferenceError;
use crate::{Descriptor, Detections, Frame, Keypoint};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

pub struct SuperPoint {
    session: Session,
}

impl SuperPoint {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, InferenceError> {
        let path = path.as_ref();
        let session = Session::builder()
            .map_err(|e| InferenceError::LoadFailed {
                path: path.to_path_buf(),
                source: e,
            })?
            .commit_from_file(path)
            .map_err(|e| InferenceError::LoadFailed {
                path: path.to_path_buf(),
                source: e,
            })?;
        Ok(Self { session })
    }

    pub fn detect(&mut self, frame: &Frame) -> Result<Detections, InferenceError> {
        let input_data: Vec<f32> = crate::preprocess::normalise(frame.data());
        let input_tensor = Tensor::from_array((
            [1, 1, frame.height() as usize, frame.width() as usize],
            input_data,
        ))?;

        let outputs = self.session.run(ort::inputs!["image" => input_tensor])?;

        let keypoints_value = &outputs["keypoints"];
        let scores_raw = outputs["scores"].try_extract_tensor::<f32>()?;
        let descriptors_raw = outputs["descriptors"].try_extract_tensor::<f32>()?;

        let scores = scores_raw.1.to_vec();
        let keypoints_pairs = if let Ok((shape, data)) = keypoints_value.try_extract_tensor::<f32>()
        {
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
        };
        let keypoints = to_keypoints(
            &keypoints_pairs,
            frame.width() as f32,
            frame.height() as f32,
        );
        let descriptors = descriptors_raw
            .1
            .chunks(256)
            .map(|chunk| Descriptor(chunk.try_into().unwrap()))
            .collect();

        Detections::new(
            frame.sensor_id(),
            frame.frame_id(),
            keypoints,
            scores,
            descriptors,
        )
            .map_err(|e| InferenceError::Domain(format!("{e:?}")))
    }
}

#[derive(Clone, Copy, Debug)]
enum Normalization {
    None,
    ZeroToOne,
    NegOneToOne,
}

fn parse_keypoint_pairs(
    shape: &ort::tensor::Shape,
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

fn to_keypoints(pairs: &[[f32; 2]], width: f32, height: f32) -> Vec<Keypoint> {
    let norm = detect_normalization(pairs);
    let score_xy = count_in_bounds(pairs, width, height, norm, false);
    let score_yx = count_in_bounds(pairs, width, height, norm, true);
    let swap = score_yx > score_xy;

    pairs
        .iter()
        .map(|pair| {
            let (x, y) = if swap {
                (
                    scale_coordinate(pair[1], width, norm),
                    scale_coordinate(pair[0], height, norm),
                )
            } else {
                (
                    scale_coordinate(pair[0], width, norm),
                    scale_coordinate(pair[1], height, norm),
                )
            };
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
    let mut count = 0;
    for pair in pairs {
        let (x, y) = if swap {
            (
                scale_coordinate(pair[1], width, norm),
                scale_coordinate(pair[0], height, norm),
            )
        } else {
            (
                scale_coordinate(pair[0], width, norm),
                scale_coordinate(pair[1], height, norm),
            )
        };
        if x >= 0.0 && x < width && y >= 0.0 && y < height {
            count += 1;
        }
    }
    count
}

fn scale_coordinate(value: f32, dim: f32, norm: Normalization) -> f32 {
    let extent = if dim > 1.0 { dim - 1.0 } else { 1.0 };
    match norm {
        Normalization::None => value,
        Normalization::ZeroToOne => value * extent,
        Normalization::NegOneToOne => (value + 1.0) * 0.5 * extent,
    }
}
