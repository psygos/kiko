use std::path::Path;

use ort::session::{RunOptions, Session};
use ort::value::{TensorElementType, TensorRef, ValueType};

use crate::Frame;
use crate::loop_closure::{GLOBAL_DESCRIPTOR_DIM, GlobalDescriptor};

use super::{
    InferenceBackend, InferenceError, InferenceRunDiagnostics, PlaceDescriptorExtractor,
    build_run_options, build_session,
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
    session: Session,
    run_options: RunOptions,
    backend: InferenceBackend,
    diagnostics: InferenceRunDiagnostics,
    scratch: Vec<f32>,
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
        let (session, selected, diagnostics) = build_session(path, backend)?;
        let run_options = build_run_options(selected)?;
        Ok(Self {
            session,
            run_options,
            backend: selected,
            diagnostics,
            scratch: vec![0.0; INPUT_ELEMENTS],
        })
    }

    pub fn backend(&self) -> InferenceBackend {
        self.backend
    }

    pub fn compute(&mut self, frame: &Frame) -> Result<GlobalDescriptor, InferenceError> {
        preprocess_frame_to_nchw(frame, &mut self.scratch);
        let input_tensor = TensorRef::from_array_view((INPUT_SHAPE, self.scratch.as_slice()))
            .map_err(|source| InferenceError::InputTensor {
                name: "input",
                source,
            })?;

        // EigenPlaces ONNX exports use `input` for the image tensor.
        let outputs =
            super::run_with_slow_call_diagnostics(self.diagnostics, "eigenplaces", || {
                self.session
                    .run_with_options(ort::inputs!["input" => input_tensor], &self.run_options)
                    .map_err(|source| InferenceError::SessionRun {
                        model: "eigenplaces",
                        source,
                    })
            })?;

        extract_global_descriptor(&outputs)
    }
}

impl PlaceDescriptorExtractor for EigenPlaces {
    fn compute_descriptor(&mut self, frame: &Frame) -> Result<GlobalDescriptor, InferenceError> {
        self.compute(frame)
    }
}

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
            let value = f32::from(src[src_row + src_x]) / U8_SCALE;
            let dst = dst_row + x;
            for channel in 0..INPUT_CHANNELS {
                let dst_idx = channel * INPUT_PLANE_ELEMENTS + dst;
                out[dst_idx] = (value - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel];
            }
        }
    }
}

fn extract_global_descriptor(
    outputs: &ort::session::SessionOutputs<'_>,
) -> Result<GlobalDescriptor, InferenceError> {
    let expected_shape = [1, GLOBAL_DESCRIPTOR_DIM as i64];
    let mut found: Option<(String, GlobalDescriptor)> = None;
    for (name, value) in outputs.iter() {
        if !matches!(
            value.dtype(),
            ValueType::Tensor {
                ty: TensorElementType::Float32,
                ..
            }
        ) {
            continue;
        }
        let (shape, data) = super::extract_tensor::<f32>(&value, name)?;
        if shape.as_ref() != expected_shape || data.len() != GLOBAL_DESCRIPTOR_DIM {
            continue;
        }
        if let Some((previous, _)) = &found {
            return Err(InferenceError::UnexpectedOutput {
                name: "eigenplaces-output".to_string(),
                expected: format!(
                    "one unambiguous f32 tensor with shape [1, {GLOBAL_DESCRIPTOR_DIM}]"
                ),
                actual: format!("multiple matching outputs: {previous}, {name}"),
            });
        }
        let descriptor = parse_descriptor_output(name, &expected_shape, data)?;
        found = Some((name.to_string(), descriptor));
    }
    found
        .map(|(_, descriptor)| descriptor)
        .ok_or_else(|| InferenceError::UnexpectedOutput {
            name: "eigenplaces-output".to_string(),
            expected: format!("one unambiguous f32 tensor with shape [1, {GLOBAL_DESCRIPTOR_DIM}]"),
            actual: "no matching output".to_string(),
        })
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
        EigenPlaces, GLOBAL_DESCRIPTOR_DIM, INPUT_ELEMENTS, parse_descriptor_output,
        preprocess_frame_to_nchw,
    };
    use crate::inference::InferenceError;
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

    #[test]
    fn preprocess_eigenplaces_produces_expected_tensor_shape() {
        let frame = Frame::new(
            SensorId::StereoLeft,
            FrameId::new(1),
            Timestamp::from_nanos(1),
            16,
            12,
            vec![127_u8; 16 * 12],
        )
        .expect("frame");
        let mut out = vec![0.0; INPUT_ELEMENTS];
        preprocess_frame_to_nchw(&frame, &mut out);
        assert_eq!(out.len(), 3 * 224 * 224);
    }

    #[test]
    fn preprocess_eigenplaces_output_is_finite() {
        let data: Vec<u8> = (0..(32 * 24)).map(|i| (i % 255) as u8).collect();
        let frame = Frame::new(
            SensorId::StereoLeft,
            FrameId::new(2),
            Timestamp::from_nanos(2),
            32,
            24,
            data,
        )
        .expect("frame");
        let mut out = vec![0.0; INPUT_ELEMENTS];
        preprocess_frame_to_nchw(&frame, &mut out);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn preprocess_eigenplaces_channels_match_for_grayscale_input() {
        let frame = Frame::new(
            SensorId::StereoLeft,
            FrameId::new(3),
            Timestamp::from_nanos(3),
            8,
            8,
            vec![200_u8; 8 * 8],
        )
        .expect("frame");
        let mut out = vec![0.0; INPUT_ELEMENTS];
        preprocess_frame_to_nchw(&frame, &mut out);
        let hw = 224 * 224;
        let a = out[0];
        let b = out[hw];
        let c = out[2 * hw];
        assert!(a.is_finite() && b.is_finite() && c.is_finite());
        assert!((a - b).abs() > 0.0 || (b - c).abs() > 0.0);
    }

    #[test]
    fn loading_nonexistent_model_preserves_path_and_io_error() {
        let missing = unique_temp_file("missing");
        assert!(!missing.exists());
        let error = match EigenPlaces::new_with_backend(&missing, InferenceBackend::Cpu) {
            Ok(_) => panic!("missing model must fail with its source"),
            Err(error) => error,
        };
        match error {
            super::InferenceError::ModelFileUnavailable { path, source } => {
                assert_eq!(path, missing);
                assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
            }
            other => panic!("expected model-file error, got {other:?}"),
        }
    }

    #[test]
    fn loading_invalid_model_returns_error() {
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
            InferenceError::GlobalDescriptor(_) => {}
            other => panic!("expected domain error, got {other:?}"),
        }
    }

    #[test]
    fn parse_descriptor_output_rejects_wrong_shape_even_with_the_right_length() {
        let raw = [1.0_f32; GLOBAL_DESCRIPTOR_DIM];
        for shape in [
            vec![GLOBAL_DESCRIPTOR_DIM as i64],
            vec![2, (GLOBAL_DESCRIPTOR_DIM / 2) as i64],
            vec![GLOBAL_DESCRIPTOR_DIM as i64, 1],
        ] {
            assert!(matches!(
                parse_descriptor_output("descriptor", &shape, &raw),
                Err(InferenceError::UnexpectedOutput { .. })
            ));
        }
    }
}
