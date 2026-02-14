use std::path::Path;

use ort::session::Session;
use ort::value::TensorRef;

use crate::loop_closure::GlobalDescriptor;
use crate::Frame;

use super::{build_session, InferenceBackend, InferenceError, PlaceDescriptorExtractor};

const INPUT_SIZE: usize = 224;
const INPUT_CHANNELS: usize = 3;
const OUTPUT_DIM: usize = 512;

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

pub struct EigenPlaces {
    session: Session,
    backend: InferenceBackend,
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
        let (session, selected) = build_session(path, backend)?;
        Ok(Self {
            session,
            backend: selected,
            scratch: Vec::new(),
        })
    }

    pub fn try_load(path: impl AsRef<Path>, backend: InferenceBackend) -> Option<Self> {
        let path_ref = path.as_ref();
        if !path_ref.exists() {
            return None;
        }
        Self::new_with_backend(path_ref, backend).ok()
    }

    pub fn backend(&self) -> InferenceBackend {
        self.backend
    }

    pub fn compute(&mut self, frame: &Frame) -> Result<GlobalDescriptor, InferenceError> {
        preprocess_frame_to_nchw(frame, &mut self.scratch);
        let input_tensor = TensorRef::from_array_view((
            [1, INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE],
            self.scratch.as_slice(),
        ))?;

        // EigenPlaces ONNX exports use `input` for the image tensor.
        let outputs = self.session.run(ort::inputs!["input" => input_tensor])?;

        let mut raw_descriptor: Option<Vec<f32>> = None;
        for (_, value) in outputs.iter() {
            if let Ok((_, data)) = value.try_extract_tensor::<f32>() {
                raw_descriptor = Some(data.to_vec());
                break;
            }
        }
        let raw_descriptor = raw_descriptor.ok_or_else(|| InferenceError::UnexpectedOutput {
            name: "eigenplaces-output".to_string(),
            expected: "at least one f32 tensor output".to_string(),
            actual: "no f32 tensor output".to_string(),
        })?;
        if raw_descriptor.len() != OUTPUT_DIM {
            return Err(InferenceError::UnexpectedOutput {
                name: "eigenplaces-output".to_string(),
                expected: format!("descriptor length {}", OUTPUT_DIM),
                actual: format!("descriptor length {}", raw_descriptor.len()),
            });
        }
        let descriptor_array: [f32; OUTPUT_DIM] =
            raw_descriptor.try_into().expect("length checked");
        GlobalDescriptor::try_new(descriptor_array)
            .map_err(|err| InferenceError::Domain(format!("invalid global descriptor: {err}")))
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

fn preprocess_frame_to_nchw(frame: &Frame, out: &mut Vec<f32>) {
    let src_width = frame.width() as usize;
    let src_height = frame.height() as usize;
    let src = frame.data();
    out.resize(INPUT_CHANNELS * INPUT_SIZE * INPUT_SIZE, 0.0);

    for y in 0..INPUT_SIZE {
        let src_y = y * src_height / INPUT_SIZE;
        for x in 0..INPUT_SIZE {
            let src_x = x * src_width / INPUT_SIZE;
            let src_idx = src_y * src_width + src_x;
            let value = src.get(src_idx).copied().unwrap_or(0) as f32 / 255.0;
            for channel in 0..INPUT_CHANNELS {
                let dst_idx = channel * INPUT_SIZE * INPUT_SIZE + y * INPUT_SIZE + x;
                out[dst_idx] = (value - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::preprocess_frame_to_nchw;
    use crate::{Frame, FrameId, SensorId, Timestamp};

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
        let mut out = Vec::new();
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
        let mut out = Vec::new();
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
        let mut out = Vec::new();
        preprocess_frame_to_nchw(&frame, &mut out);
        let hw = 224 * 224;
        let a = out[0];
        let b = out[hw];
        let c = out[2 * hw];
        assert!(a.is_finite() && b.is_finite() && c.is_finite());
        assert!((a - b).abs() > 0.0 || (b - c).abs() > 0.0);
    }
}
