use super::{
    InferenceBackend, InferenceError, ManagedSession, PinnedOrtRuntime,
    SUPERPOINT_MAXIMUM_OUTPUT_KEYPOINTS, SUPERPOINT_MINIMUM_INPUT_AXIS_PX, build_session,
    build_session_from_memory,
};
use crate::{Descriptor, Detections, DownscaleFactor, Frame, FrameDimensions, Keypoint};
use ort::session::RunOptions;
use ort::value::{Shape, Tensor};
use std::path::Path;

use crate::DESCRIPTOR_DIM;

const MAX_OUTPUT_KEYPOINTS: usize = SUPERPOINT_MAXIMUM_OUTPUT_KEYPOINTS as usize;
const MODEL_STRIDE: u32 = SUPERPOINT_MINIMUM_INPUT_AXIS_PX;
const EXCLUDED_BORDER: u32 = 4;
const SCORE_THRESHOLD: f32 = 0.0005;

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

        let scale = downscale.map_or(1, DownscaleFactor::get_u32);
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

pub struct SuperPoint {
    session: ManagedSession,
    backend: InferenceBackend,
    scratch: Vec<f32>,
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
        let (session, selected) = build_session(path, backend)?;
        Ok(Self {
            session,
            backend: selected,
            scratch: Vec::new(),
        })
    }

    /// Build a session directly from retained ONNX bytes.
    ///
    /// The runtime proof ensures this path cannot silently consult
    /// `ORT_DYLIB_PATH` or reopen a deployment pathname.
    pub fn new_from_memory(
        model_bytes: &[u8],
        runtime: PinnedOrtRuntime,
    ) -> Result<Self, InferenceError> {
        Self::new_from_memory_with_backend(model_bytes, runtime, InferenceBackend::auto())
    }

    pub fn new_from_memory_with_backend(
        model_bytes: &[u8],
        runtime: PinnedOrtRuntime,
        backend: InferenceBackend,
    ) -> Result<Self, InferenceError> {
        let (session, selected) =
            build_session_from_memory("superpoint", model_bytes, backend, runtime)?;
        Ok(Self {
            session,
            backend: selected,
            scratch: Vec::new(),
        })
    }

    pub fn backend(&self) -> InferenceBackend {
        self.backend
    }

    pub fn detect(&mut self, frame: &Frame) -> Result<Detections, InferenceError> {
        let input_domain =
            SuperPointInputDomain::try_new(frame.dimensions(), frame.dimensions(), None)?;
        crate::preprocess::normalise_frame_into(frame, &mut self.scratch);

        let input_tensor = Tensor::from_array((
            [1, 1, frame.height() as usize, frame.width() as usize],
            self.scratch.clone(),
        ))?;

        run_inference(&mut self.session, frame, input_tensor, input_domain)
    }

    pub fn detect_with_downscale(
        &mut self,
        frame: &Frame,
        downscale: DownscaleFactor,
    ) -> Result<Detections, InferenceError> {
        if downscale.get() == 1 {
            return self.detect(frame);
        }

        let dimensions =
            crate::preprocess::normalise_downscale_into(frame, downscale, &mut self.scratch)
                .map_err(InferenceError::Downscale)?;
        let input_domain =
            SuperPointInputDomain::try_new(dimensions, frame.dimensions(), Some(downscale))?;

        let input_tensor = Tensor::from_array((
            [
                1,
                1,
                dimensions.height() as usize,
                dimensions.width() as usize,
            ],
            self.scratch.clone(),
        ))?;

        run_inference(&mut self.session, frame, input_tensor, input_domain)
    }
}

fn run_inference(
    session: &mut ManagedSession,
    frame: &Frame,
    input_tensor: Tensor<f32>,
    input_domain: SuperPointInputDomain,
) -> Result<Detections, InferenceError> {
    session.run("superpoint", |session| {
        let run_options = RunOptions::new().map_err(InferenceError::Execution)?;
        let outputs = super::run_with_watchdog("superpoint", || {
            session.run_async(ort::inputs!["image" => input_tensor], &run_options)
        })?;

        let keypoints_raw = outputs
            .get("keypoints")
            .ok_or_else(|| InferenceError::UnexpectedOutput {
                name: "keypoints".to_string(),
                expected: "named output tensor".to_string(),
                actual: "missing output".to_string(),
            })?
            .try_extract_tensor::<i64>()
            .map_err(|source| InferenceError::OutputDecode {
                name: "keypoints",
                source,
            })?;
        let scores_raw = outputs
            .get("scores")
            .ok_or_else(|| InferenceError::UnexpectedOutput {
                name: "scores".to_string(),
                expected: "named output tensor".to_string(),
                actual: "missing output".to_string(),
            })?
            .try_extract_tensor::<f32>()
            .map_err(|source| InferenceError::OutputDecode {
                name: "scores",
                source,
            })?;
        let descriptors_raw = outputs
            .get("descriptors")
            .ok_or_else(|| InferenceError::UnexpectedOutput {
                name: "descriptors".to_string(),
                expected: "named output tensor".to_string(),
                actual: "missing output".to_string(),
            })?
            .try_extract_tensor::<f32>()
            .map_err(|source| InferenceError::OutputDecode {
                name: "descriptors",
                source,
            })?;

        let parsed = parse_outputs(keypoints_raw, scores_raw, descriptors_raw, input_domain)?;

        Ok(Detections::from_parsed_components(
            frame.sensor_id(),
            frame.frame_id(),
            frame.dimensions(),
            parsed.keypoints,
            parsed.scores,
            parsed.descriptors,
        ))
    })
}

#[derive(Debug)]
struct ParsedSuperPointOutput {
    keypoints: Vec<Keypoint>,
    scores: Vec<f32>,
    descriptors: Vec<Descriptor>,
}

fn parse_outputs(
    keypoints: (&Shape, &[i64]),
    scores: (&Shape, &[f32]),
    descriptors: (&Shape, &[f32]),
    input_domain: SuperPointInputDomain,
) -> Result<ParsedSuperPointOutput, InferenceError> {
    let (keypoint_shape, keypoint_data) = keypoints;
    let count_i64 = match &keypoint_shape[..] {
        [1, count, 2] if *count >= 0 => *count,
        _ => {
            return Err(InferenceError::UnexpectedOutput {
                name: "keypoints".to_string(),
                expected: format!(
                    "i64 tensor shape [1, N, 2] with 0 <= N <= {MAX_OUTPUT_KEYPOINTS}"
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
    if count > MAX_OUTPUT_KEYPOINTS {
        return Err(InferenceError::UnexpectedOutput {
            name: "keypoints".to_string(),
            expected: format!("at most {MAX_OUTPUT_KEYPOINTS} keypoints"),
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
        if !score.is_finite() || score <= SCORE_THRESHOLD || score > 1.0 {
            return Err(InferenceError::UnexpectedOutput {
                name: "scores".to_string(),
                expected: format!(
                    "finite confidence scores within ({SCORE_THRESHOLD}, 1] in non-increasing order"
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
    if let Some((flat_index, &value)) = descriptors
        .1
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(InferenceError::UnexpectedOutput {
            name: "descriptors".to_string(),
            expected: "finite descriptor components".to_string(),
            actual: format!(
                "descriptor {} component {} contains {value}",
                flat_index / DESCRIPTOR_DIM,
                flat_index % DESCRIPTOR_DIM
            ),
        });
    }

    let mut parsed_keypoints = Vec::with_capacity(count);
    for (index, pair) in keypoint_data.chunks_exact(2).enumerate() {
        parsed_keypoints.push(Keypoint {
            x: parse_coordinate(
                pair[0],
                input_domain.effective_width,
                input_domain.scale,
                index,
                "x",
            )?,
            y: parse_coordinate(
                pair[1],
                input_domain.effective_height,
                input_domain.scale,
                index,
                "y",
            )?,
        });
    }

    let mut parsed_descriptors = Vec::with_capacity(count);
    for chunk in descriptors.1.chunks_exact(DESCRIPTOR_DIM) {
        let mut descriptor = [0.0_f32; DESCRIPTOR_DIM];
        descriptor.copy_from_slice(chunk);
        parsed_descriptors.push(Descriptor(descriptor));
    }

    Ok(ParsedSuperPointOutput {
        keypoints: parsed_keypoints,
        scores: scores.1.to_vec(),
        descriptors: parsed_descriptors,
    })
}

fn require_shape_and_length(
    name: &str,
    shape: &Shape,
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

fn parse_coordinate(
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

#[cfg(test)]
mod tests {
    use super::*;

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
    ) -> Result<ParsedSuperPointOutput, InferenceError> {
        let keypoint_shape = Shape::new(keypoint_shape.iter().copied());
        let score_shape = Shape::new(score_shape.iter().copied());
        let descriptor_shape = Shape::new(descriptor_shape.iter().copied());
        let input_domain =
            SuperPointInputDomain::try_new(model_dimensions, frame_dimensions, downscale)?;
        parse_outputs(
            (&keypoint_shape, keypoint_data),
            (&score_shape, score_data),
            (&descriptor_shape, descriptor_data),
            input_domain,
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
        let dimensions = FrameDimensions::new(640, 480);
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
        )
    }

    fn assert_output_error(
        result: Result<ParsedSuperPointOutput, InferenceError>,
        expected_name: &str,
    ) {
        assert_unexpected_output(result.expect_err("malformed model output"), expected_name);
    }

    fn assert_unexpected_output(error: InferenceError, expected_name: &str) {
        match error {
            InferenceError::UnexpectedOutput {
                name: actual_name, ..
            } => {
                assert_eq!(actual_name, expected_name);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    fn assert_unrepresentable_coordinate(
        error: InferenceError,
        expected_index: usize,
        expected_axis: &'static str,
        expected_coordinate: u32,
    ) {
        assert!(matches!(
            error,
            InferenceError::KeypointCoordinateUnrepresentable {
                model: "superpoint",
                index,
                axis,
                coordinate,
            } if index == expected_index
                && axis == expected_axis
                && coordinate == expected_coordinate
        ));
    }

    #[test]
    fn input_domain_rejects_only_dimensions_below_the_model_minimum() {
        for (width, height) in [(7, 8), (8, 7)] {
            assert!(matches!(
                SuperPointInputDomain::try_new(
                    FrameDimensions::new(width, height),
                    FrameDimensions::new(width, height),
                    None,
                ),
                Err(InferenceError::InputDimensionsTooSmall {
                    model: "superpoint",
                    width: actual_width,
                    height: actual_height,
                    minimum: MODEL_STRIDE,
                }) if actual_width == width && actual_height == height
            ));
        }

        for (width, height) in [(8, 8), (15, 15), (27, 19)] {
            SuperPointInputDomain::try_new(
                FrameDimensions::new(width, height),
                FrameDimensions::new(width, height),
                None,
            )
            .expect("supported nonzero feature grid");
        }

        let factor = DownscaleFactor::try_from(2).expect("nonzero scale");
        assert!(matches!(
            SuperPointInputDomain::try_new(
                FrameDimensions::new(7, 7),
                FrameDimensions::new(14, 14),
                Some(factor),
            ),
            Err(InferenceError::InputDimensionsTooSmall {
                width: 7,
                height: 7,
                minimum: MODEL_STRIDE,
                ..
            })
        ));
    }

    #[test]
    fn canonical_outputs_preserve_absolute_xy_and_row_alignment() {
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
        .expect("canonical output contract");

        assert_eq!(parsed.keypoints.len(), 2);
        assert_eq!((parsed.keypoints[0].x, parsed.keypoints[0].y), (4.0, 4.0));
        assert_eq!((parsed.keypoints[1].x, parsed.keypoints[1].y), (17.0, 9.0));
        assert_eq!(parsed.scores, [1.0, 0.5]);
        assert_eq!(parsed.descriptors[0].0, [0.125; DESCRIPTOR_DIM]);
        assert_eq!(parsed.descriptors[1].0, [-0.5; DESCRIPTOR_DIM]);
    }

    #[test]
    fn canonical_empty_output_is_valid() {
        let parsed = parse_original_scale(
            &[1, 0, 2],
            &[],
            &[1, 0],
            &[],
            &[1, 0, DESCRIPTOR_DIM as i64],
            &[],
        )
        .expect("empty canonical output");

        assert!(parsed.keypoints.is_empty());
        assert!(parsed.scores.is_empty());
        assert!(parsed.descriptors.is_empty());
    }

    #[test]
    fn downscaled_coordinates_map_to_top_left_source_pixels() {
        let descriptors = vec![0.0; 2 * DESCRIPTOR_DIM];
        let factor = DownscaleFactor::try_from(2).expect("nonzero scale");
        let parsed = parse_fixture(
            &[1, 2, 2],
            &[4, 4, 315, 235],
            &[1, 2],
            &[0.5, 0.25],
            &[1, 2, DESCRIPTOR_DIM as i64],
            &descriptors,
            FrameDimensions::new(320, 240),
            FrameDimensions::new(640, 480),
            Some(factor),
        )
        .expect("integer top-left mapping");

        assert_eq!((parsed.keypoints[0].x, parsed.keypoints[0].y), (8.0, 8.0));
        assert_eq!(
            (parsed.keypoints[1].x, parsed.keypoints[1].y),
            (630.0, 470.0)
        );
    }

    #[test]
    fn downscale_mapping_rejects_a_nonexact_f32_pixel() {
        const EXACT_MODEL_X: i64 = 5_592_406;
        const INEXACT_MODEL_X: i64 = EXACT_MODEL_X + 1;
        let model_dimensions = FrameDimensions::new(5_592_424, 16);
        let frame_dimensions = FrameDimensions::new(16_777_272, 48);
        let factor = DownscaleFactor::try_from(3).expect("nonzero scale");
        let descriptors = [0.0; DESCRIPTOR_DIM];

        let exact = parse_fixture(
            &[1, 1, 2],
            &[EXACT_MODEL_X, 4],
            &[1, 1],
            &[0.5],
            &[1, 1, DESCRIPTOR_DIM as i64],
            &descriptors,
            model_dimensions,
            frame_dimensions,
            Some(factor),
        )
        .expect("even pixel above 2^24 remains exactly representable");
        assert_eq!(exact.keypoints[0].x, 16_777_218.0);

        assert_unrepresentable_coordinate(
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
            )
            .expect_err("odd scaled pixel beyond f32's exact range"),
            0,
            "x",
            16_777_221,
        );
    }

    #[test]
    fn coordinate_parser_enforces_the_effective_grid_and_excluded_border() {
        let model_dimensions = FrameDimensions::new(27, 19);
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
        )
        .expect("inclusive lower and exclusive upper graph bounds");
        assert_eq!(
            (accepted.keypoints[0].x, accepted.keypoints[0].y),
            (4.0, 4.0)
        );
        assert_eq!(
            (accepted.keypoints[1].x, accepted.keypoints[1].y),
            (19.0, 11.0)
        );

        for [x, y] in [[3, 4], [20, 4], [24, 4], [26, 4], [4, 3], [4, 12], [4, 18]] {
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
                ),
                "keypoints",
            );
        }
    }

    #[test]
    fn output_parser_requires_exact_aligned_shapes_and_lengths() {
        let descriptors = vec![0.0; DESCRIPTOR_DIM];

        for shape in [&[1, 2][..], &[2, 1, 2], &[1, 1, 2, 1], &[1, 1, 3]] {
            assert_output_error(
                parse_original_scale(
                    shape,
                    &[1, 1],
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
                &[1],
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
                    &[1, 1],
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
                parse_original_scale(&[1, 1, 2], &[1, 1], &[1, 1], &[0.5], shape, data),
                "descriptors",
            );
        }
    }

    #[test]
    fn output_keypoint_count_is_bounded_before_copying() {
        let count = MAX_OUTPUT_KEYPOINTS;
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
        .expect("canonical model cap is accepted");
        assert_eq!(parsed.keypoints.len(), MAX_OUTPUT_KEYPOINTS);

        let rejected = (MAX_OUTPUT_KEYPOINTS + 1) as i64;
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
    fn coordinate_parser_checks_integer_bounds_before_narrowing() {
        for value in [-1, 1, 3, i64::MIN, i64::MAX, 636, 640] {
            assert_output_error(
                parse_original_scale(
                    &[1, 1, 2],
                    &[value, 4],
                    &[1, 1],
                    &[0.5],
                    &[1, 1, DESCRIPTOR_DIM as i64],
                    &[0.0; DESCRIPTOR_DIM],
                ),
                "keypoints",
            );
        }

        let width = 16_777_217;
        let dimensions = FrameDimensions::new(width, 16);
        assert_output_error(
            parse_fixture(
                &[1, 1, 2],
                &[i64::from(width), 4],
                &[1, 1],
                &[0.5],
                &[1, 1, DESCRIPTOR_DIM as i64],
                &[0.0; DESCRIPTOR_DIM],
                dimensions,
                dimensions,
                None,
            ),
            "keypoints",
        );
    }

    #[test]
    fn coordinate_parser_rejects_only_unrepresentable_f32_integers() {
        const TWO_POW_24: i64 = 16_777_216;
        let effective_extent = u32::try_from(TWO_POW_24 + 24).expect("u32 extent");

        assert_eq!(
            parse_coordinate(TWO_POW_24, effective_extent, 1, 0, "x").expect("exact power of two"),
            TWO_POW_24 as f32
        );
        assert_unrepresentable_coordinate(
            parse_coordinate(TWO_POW_24 + 1, effective_extent, 1, 0, "x")
                .expect_err("odd integer beyond f32's exact range"),
            0,
            "x",
            16_777_217,
        );
        assert_eq!(
            parse_coordinate(TWO_POW_24 + 2, effective_extent, 1, 0, "x")
                .expect("the next even integer remains exact"),
            (TWO_POW_24 + 2) as f32
        );
    }

    #[test]
    fn score_parser_enforces_threshold_range_and_topk_order() {
        let descriptors = [0.0; DESCRIPTOR_DIM];
        for score in [
            f32::NAN,
            f32::NEG_INFINITY,
            f32::INFINITY,
            -f32::from_bits(1),
            SCORE_THRESHOLD,
            f32::from_bits(SCORE_THRESHOLD.to_bits() - 1),
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

        for score in [f32::from_bits(SCORE_THRESHOLD.to_bits() + 1), 1.0] {
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

        let descriptors = [0.0; 2 * DESCRIPTOR_DIM];
        parse_original_scale(
            &[1, 2, 2],
            &[4, 4, 5, 5],
            &[1, 2],
            &[0.5, 0.5],
            &[1, 2, DESCRIPTOR_DIM as i64],
            &descriptors,
        )
        .expect("TopK ties remain valid");
        assert_output_error(
            parse_original_scale(
                &[1, 2, 2],
                &[4, 4, 5, 5],
                &[1, 2],
                &[0.5, 0.75],
                &[1, 2, DESCRIPTOR_DIM as i64],
                &descriptors,
            ),
            "scores",
        );
    }

    #[test]
    fn descriptor_parser_reports_the_exact_nonfinite_component() {
        let mut descriptors = [0.0; DESCRIPTOR_DIM];
        descriptors[17] = f32::NAN;
        let error = parse_original_scale(
            &[1, 1, 2],
            &[4, 4],
            &[1, 1],
            &[0.5],
            &[1, 1, DESCRIPTOR_DIM as i64],
            &descriptors,
        )
        .expect_err("nonfinite descriptor");

        assert!(matches!(
            error,
            InferenceError::UnexpectedOutput { name, actual, .. }
                if name == "descriptors"
                    && actual.contains("descriptor 0 component 17")
        ));
    }
}
