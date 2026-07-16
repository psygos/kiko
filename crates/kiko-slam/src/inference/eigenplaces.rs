use std::path::Path;

use ort::session::RunOptions;
use ort::value::{Tensor, TensorElementType};

use crate::Frame;
use crate::loop_closure::{GLOBAL_DESCRIPTOR_DIM, GlobalDescriptor};

use super::{
    InferenceBackend, InferenceError, ManagedSession, PlaceDescriptorExtractor, build_session,
};

const INPUT_SIZE: usize = 224;
const INPUT_CHANNELS: usize = 3;
const INPUT_PLANE_ELEMENTS: usize = INPUT_SIZE * INPUT_SIZE;
const INPUT_ELEMENTS: usize = INPUT_CHANNELS * INPUT_PLANE_ELEMENTS;
const INPUT_SHAPE: [usize; 4] = [1, INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE];
const U8_SCALE: f32 = 255.0;

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

pub struct EigenPlaces {
    session: ManagedSession,
    backend: InferenceBackend,
    input: Tensor<f32>,
}

impl EigenPlaces {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, InferenceError> {
        Self::new_with_backend(path, InferenceBackend::auto())
    }

    pub fn new_with_backend(
        path: impl AsRef<Path>,
        backend: InferenceBackend,
    ) -> Result<Self, InferenceError> {
        let path = path.as_ref();
        std::fs::metadata(path).map_err(|source| InferenceError::ModelFileUnavailable {
            path: path.to_path_buf(),
            source,
        })?;
        let (session, selected) = build_session(path, backend)?;
        let input = Tensor::from_array((INPUT_SHAPE, vec![0.0_f32; INPUT_ELEMENTS]))?;
        Ok(Self {
            session,
            backend: selected,
            input,
        })
    }

    pub fn backend(&self) -> InferenceBackend {
        self.backend
    }

    pub fn compute(&mut self, frame: &Frame) -> Result<GlobalDescriptor, InferenceError> {
        let Self { session, input, .. } = self;
        session.run("eigenplaces", |session| {
            // Keep preprocessing inside ManagedSession::run. After a timeout,
            // ort's async context retains an Arc to this tensor's ValueInner
            // (including its Vec backing) until the callback. Quarantining the
            // session also prevents a later call from mutating that storage
            // while the timed-out provider may still be reading it.
            let (_, input_data) = input.extract_tensor_mut();
            preprocess_frame_to_nchw(frame, input_data);

            // EigenPlaces ONNX exports use `input` for the image tensor.
            let run_options = RunOptions::new().map_err(InferenceError::Execution)?;
            let outputs = super::run_with_watchdog("eigenplaces", || {
                session.run_async(ort::inputs!["input" => &*input], &run_options)
            })?;

            // Exporters may expose auxiliary tensors. Identify the descriptor
            // by its complete contract instead of relying on output order or
            // an exporter-specific name, and reject genuinely ambiguous sets.
            let expected_shape = [1, GLOBAL_DESCRIPTOR_DIM as i64];
            let mut descriptor_output: Option<(&str, usize)> = None;
            for (index, (name, value)) in outputs.iter().enumerate() {
                if value.dtype().tensor_type() != Some(TensorElementType::Float32) {
                    continue;
                }
                let (shape, raw_descriptor) = value
                    .try_extract_tensor::<f32>()
                    .map_err(InferenceError::Execution)?;
                if shape.as_ref() != expected_shape || raw_descriptor.len() != GLOBAL_DESCRIPTOR_DIM
                {
                    continue;
                }
                if let Some((previous_name, _)) = descriptor_output {
                    return Err(InferenceError::UnexpectedOutput {
                        name: "eigenplaces-output-set".to_string(),
                        expected: format!(
                            "one unambiguous f32 tensor with shape [1, {GLOBAL_DESCRIPTOR_DIM}]"
                        ),
                        actual: format!("multiple matching outputs: {previous_name}, {name}"),
                    });
                }
                descriptor_output = Some((name, index));
            }

            let Some((name, index)) = descriptor_output else {
                let actual = if outputs.len() == 0 {
                    "no outputs".to_string()
                } else {
                    outputs
                        .iter()
                        .map(|(output_name, output)| format!("{output_name}: {}", output.dtype()))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                return Err(InferenceError::UnexpectedOutput {
                    name: "eigenplaces-output-set".to_string(),
                    expected: format!(
                        "one unambiguous f32 tensor with shape [1, {GLOBAL_DESCRIPTOR_DIM}]"
                    ),
                    actual,
                });
            };
            let (shape, raw_descriptor) = outputs[index]
                .try_extract_tensor::<f32>()
                .map_err(InferenceError::Execution)?;
            parse_descriptor_output(name, shape, raw_descriptor)
        })
    }
}

impl PlaceDescriptorExtractor for EigenPlaces {
    fn backend_name(&self) -> &'static str {
        "eigenplaces"
    }

    fn compute_descriptor(&mut self, frame: &Frame) -> Result<GlobalDescriptor, InferenceError> {
        self.compute(frame)
    }
}

/// Converts a mono intensity in `[0, 255]` to ImageNet-normalized RGB channels.
fn normalized_grayscale(sample: u8) -> [f32; INPUT_CHANNELS] {
    let value = f32::from(sample) / U8_SCALE;
    std::array::from_fn(|channel| (value - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel])
}

/// Replicates the mono image into normalized RGB planes in NCHW order.
///
/// Resize coordinates use the existing top-left-aligned integer rule
/// `floor(destination * source_extent / INPUT_SIZE)` on each axis.
fn preprocess_frame_to_nchw(frame: &Frame, out: &mut [f32]) {
    debug_assert_eq!(out.len(), INPUT_ELEMENTS);
    let src_width = frame.width() as usize;
    let src_height = frame.height() as usize;
    let src = frame.data();

    for y in 0..INPUT_SIZE {
        let src_y = y * src_height / INPUT_SIZE;
        let src_row = src_y * src_width;
        let dst_row = y * INPUT_SIZE;
        for x in 0..INPUT_SIZE {
            let src_x = x * src_width / INPUT_SIZE;
            let normalized = normalized_grayscale(src[src_row + src_x]);
            let dst = dst_row + x;
            out[dst] = normalized[0];
            out[INPUT_PLANE_ELEMENTS + dst] = normalized[1];
            out[2 * INPUT_PLANE_ELEMENTS + dst] = normalized[2];
        }
    }
}

fn parse_descriptor_output(
    output_name: &str,
    shape: &[i64],
    raw_descriptor: &[f32],
) -> Result<GlobalDescriptor, InferenceError> {
    let expected_shape = [1, GLOBAL_DESCRIPTOR_DIM as i64];
    if shape != expected_shape {
        return Err(InferenceError::UnexpectedOutput {
            name: output_name.to_string(),
            expected: format!("tensor shape [1, {GLOBAL_DESCRIPTOR_DIM}]"),
            actual: format!("tensor shape {shape:?}"),
        });
    }

    let descriptor_array: [f32; GLOBAL_DESCRIPTOR_DIM] =
        raw_descriptor
            .try_into()
            .map_err(|_| InferenceError::UnexpectedOutput {
                name: output_name.to_string(),
                expected: format!("descriptor length {GLOBAL_DESCRIPTOR_DIM}"),
                actual: format!("descriptor length {}", raw_descriptor.len()),
            })?;
    GlobalDescriptor::try_new(descriptor_array).map_err(InferenceError::GlobalDescriptor)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        EigenPlaces, GLOBAL_DESCRIPTOR_DIM, IMAGENET_MEAN, IMAGENET_STD, INPUT_ELEMENTS,
        INPUT_PLANE_ELEMENTS, INPUT_SIZE, U8_SCALE, normalized_grayscale, parse_descriptor_output,
        preprocess_frame_to_nchw,
    };
    use crate::inference::InferenceError;
    use crate::loop_closure::GlobalDescriptorError;
    use crate::{Frame, FrameId, InferenceBackend, SensorId, Timestamp};

    fn unique_temp_file(name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should advance")
            .as_nanos();
        path.push(format!("kiko-eigenplaces-{name}-{nanos}.onnx"));
        path
    }

    fn frame(width: u32, height: u32, data: Vec<u8>) -> Frame {
        Frame::new(
            SensorId::StereoLeft,
            FrameId::new(1),
            Timestamp::from_nanos(1),
            width,
            height,
            data,
        )
        .expect("valid frame")
    }

    fn preprocess(frame: &Frame) -> Vec<f32> {
        let mut out = vec![f32::NAN; INPUT_ELEMENTS];
        preprocess_frame_to_nchw(frame, &mut out);
        out
    }

    fn expected_normalized(sample: u8, channel: usize) -> f32 {
        (f32::from(sample) / U8_SCALE - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
    }

    #[test]
    fn grayscale_normalization_is_finite_and_strictly_monotonic_for_every_byte() {
        let mut previous = [f32::NEG_INFINITY; 3];
        for value in 0_u16..=u16::from(u8::MAX) {
            let sample = u8::try_from(value).expect("bounded by u8::MAX");
            let normalized = normalized_grayscale(sample);
            for channel in 0..3 {
                assert!(normalized[channel].is_finite());
                assert!(normalized[channel] > previous[channel]);
                assert_eq!(normalized[channel], expected_normalized(sample, channel));
            }
            previous = normalized;
        }
    }

    #[test]
    fn preprocess_eigenplaces_overwrites_the_complete_fixed_tensor() {
        let data: Vec<u8> = (0_u16..(32 * 24))
            .map(|value| u8::try_from(value % 256).expect("reduced modulo 256"))
            .collect();
        let frame = frame(32, 24, data);
        let out = preprocess(&frame);
        assert_eq!(out.len(), 3 * 224 * 224);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn preprocess_eigenplaces_uses_nchw_layout_and_top_left_integer_sampling() {
        let out = preprocess(&frame(2, 2, vec![0, 64, 128, 255]));
        let samples = [
            (0, 0, 0),
            (INPUT_SIZE - 1, 0, 64),
            (0, INPUT_SIZE - 1, 128),
            (INPUT_SIZE - 1, INPUT_SIZE - 1, 255),
            (INPUT_SIZE / 2, INPUT_SIZE / 2, 255),
        ];

        for (x, y, sample) in samples {
            let pixel = y * INPUT_SIZE + x;
            for channel in 0..3 {
                assert_eq!(
                    out[channel * INPUT_PLANE_ELEMENTS + pixel],
                    expected_normalized(sample, channel),
                    "channel={channel}, x={x}, y={y}"
                );
            }
        }
    }

    #[test]
    fn preprocess_eigenplaces_handles_minimum_frame_dimensions() {
        let out = preprocess(&frame(1, 1, vec![200]));
        for channel in 0..3 {
            let expected = expected_normalized(200, channel);
            assert!(
                out[channel * INPUT_PLANE_ELEMENTS..(channel + 1) * INPUT_PLANE_ELEMENTS]
                    .iter()
                    .all(|&value| value == expected)
            );
        }
    }

    #[test]
    fn loading_nonexistent_model_preserves_path_and_io_error() {
        let missing = unique_temp_file("missing");
        assert!(!missing.exists());
        let err = match EigenPlaces::new_with_backend(&missing, InferenceBackend::Cpu) {
            Ok(_) => panic!("missing model must not load"),
            Err(err) => err,
        };
        match err {
            super::InferenceError::ModelFileUnavailable { path, source } => {
                assert_eq!(path, missing);
                assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
            }
            other => panic!("expected model-file error, got {other:?}"),
        }
    }

    #[test]
    fn loading_invalid_model_returns_an_error() {
        let invalid = unique_temp_file("invalid");
        fs::write(&invalid, b"not-an-onnx-model").expect("write invalid model");
        assert!(EigenPlaces::new_with_backend(&invalid, InferenceBackend::Cpu).is_err());
        fs::remove_file(&invalid).expect("cleanup invalid model");
    }

    #[test]
    fn parse_descriptor_output_rejects_non_finite_descriptor() {
        let mut raw = [0.0_f32; GLOBAL_DESCRIPTOR_DIM];
        raw[0] = f32::NAN;
        let err = parse_descriptor_output("descriptor", &[1, GLOBAL_DESCRIPTOR_DIM as i64], &raw)
            .expect_err("non-finite descriptor should fail");
        match err {
            InferenceError::GlobalDescriptor(GlobalDescriptorError::NonFiniteValue {
                index: 0,
                value,
            }) if value.is_nan() => {}
            other => panic!("expected domain error, got {other:?}"),
        }
    }

    #[test]
    fn parse_descriptor_output_rejects_wrong_shapes_even_with_the_right_length() {
        let raw = [1.0_f32; GLOBAL_DESCRIPTOR_DIM];
        let wrong_shapes: &[&[i64]] = &[
            &[GLOBAL_DESCRIPTOR_DIM as i64],
            &[2, (GLOBAL_DESCRIPTOR_DIM / 2) as i64],
            &[1, 1, GLOBAL_DESCRIPTOR_DIM as i64],
            &[GLOBAL_DESCRIPTOR_DIM as i64, 1],
        ];

        for &shape in wrong_shapes {
            let err = parse_descriptor_output("descriptor", shape, &raw)
                .expect_err("wrong tensor shape must fail");
            assert!(matches!(err, InferenceError::UnexpectedOutput { .. }));
        }
    }

    #[test]
    fn parse_descriptor_output_rejects_shape_data_length_disagreement() {
        let shape = [1, GLOBAL_DESCRIPTOR_DIM as i64];
        let short = vec![1.0; GLOBAL_DESCRIPTOR_DIM - 1];
        let long = vec![1.0; GLOBAL_DESCRIPTOR_DIM + 1];
        for raw in [&short[..], &long[..]] {
            let err = parse_descriptor_output("descriptor", &shape, raw)
                .expect_err("wrong descriptor length must fail");
            assert!(matches!(err, InferenceError::UnexpectedOutput { .. }));
        }
    }

    #[test]
    fn parse_descriptor_output_normalizes_extreme_finite_values() {
        let descriptor = parse_descriptor_output(
            "descriptor",
            &[1, GLOBAL_DESCRIPTOR_DIM as i64],
            &[f32::MAX; GLOBAL_DESCRIPTOR_DIM],
        )
        .expect("extreme finite descriptor");
        let norm_sq = descriptor
            .as_array()
            .iter()
            .map(|&value| f64::from(value).powi(2))
            .sum::<f64>();
        assert!(descriptor.as_array().iter().all(|value| value.is_finite()));
        assert!((norm_sq - 1.0).abs() < 1e-6, "norm_sq={norm_sq}");
    }
}
