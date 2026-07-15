const U8_SCALE: f32 = 255.0;

// since camera outputs a u8, we can just precompute all the float conversion and static it into
// the binary
const fn build_float_lut() -> [f32; 256] {
    let mut table = [0 as f32; 256];
    let mut i: usize = 0;

    while i < 256 {
        table[i] = i as f32 / U8_SCALE;
        i += 1
    }
    table
}

static LUT: [f32; 256] = build_float_lut();

pub fn normalise_frame_into(frame: &crate::Frame, out: &mut Vec<f32>) {
    out.resize(frame.dimensions().area(), 0.0);
    for (dst, &src) in out.iter_mut().zip(frame.data()) {
        *dst = LUT[src as usize];
    }
}

pub fn normalise_downscale_into(
    frame: &crate::Frame,
    factor: crate::DownscaleFactor,
    out: &mut Vec<f32>,
) -> Result<crate::FrameDimensions, crate::DownscaleError> {
    let dimensions = frame.dimensions();
    let width = dimensions.width();
    let height = dimensions.height();
    let factor_u32 = factor.get_u32();
    if !width.is_multiple_of(factor_u32) || !height.is_multiple_of(factor_u32) {
        return Err(crate::DownscaleError::NonDivisible {
            width,
            height,
            factor: factor.get(),
        });
    }

    let out_width = width / factor_u32;
    let out_height = height / factor_u32;
    let out_dims = crate::FrameDimensions::new(out_width, out_height);
    let out_len = out_dims.area();
    out.resize(out_len, 0.0);

    let stride = width as usize;
    let step = factor.get();

    let mut out_idx = 0usize;
    for y in 0..out_height as usize {
        let src_y = y * step;
        let row = src_y * stride;
        for x in 0..out_width as usize {
            let src_x = x * step;
            let idx = row + src_x;
            out[out_idx] = LUT[frame.data()[idx] as usize];
            out_idx += 1;
        }
    }

    Ok(out_dims)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DownscaleError, Frame, FrameId, SensorId, Timestamp};

    fn frame(width: u32, height: u32, data: Vec<u8>) -> Frame {
        Frame::new(
            SensorId::StereoLeft,
            FrameId::new(1),
            Timestamp::from_nanos(0),
            width,
            height,
            data,
        )
        .expect("valid test frame")
    }

    #[test]
    fn normalise_frame_resizes_and_overwrites_the_complete_destination() {
        let frame = frame(2, 2, vec![0, 64, 128, 255]);

        for mut output in [vec![42.0], vec![42.0; 8]] {
            normalise_frame_into(&frame, &mut output);
            assert_eq!(output, vec![0.0, 64.0 / U8_SCALE, 128.0 / U8_SCALE, 1.0]);
        }
    }

    #[test]
    fn downscale_samples_the_top_left_pixel_of_each_block() {
        let frame = frame(4, 4, (0_u8..16).collect());
        let factor = crate::DownscaleFactor::try_from(2).expect("valid factor");
        let mut output = vec![42.0; 7];

        let dimensions =
            normalise_downscale_into(&frame, factor, &mut output).expect("divisible dimensions");

        assert_eq!((dimensions.width(), dimensions.height()), (2, 2));
        assert_eq!(
            output,
            vec![0.0, 2.0 / U8_SCALE, 8.0 / U8_SCALE, 10.0 / U8_SCALE]
        );
    }

    #[test]
    fn downscale_dimension_error_does_not_mutate_destination() {
        let frame = frame(3, 2, (0_u8..6).collect());
        let factor = crate::DownscaleFactor::try_from(2).expect("valid factor");
        let mut output = vec![42.0, 43.0];
        let before = output.clone();

        assert!(matches!(
            normalise_downscale_into(&frame, factor, &mut output),
            Err(DownscaleError::NonDivisible {
                width: 3,
                height: 2,
                factor: 2,
            })
        ));
        assert_eq!(output, before);
    }
}
