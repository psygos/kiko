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

pub fn normalise_into(data: &[u8], out: &mut [f32]) -> Result<(), crate::FrameError> {
    if data.len() != out.len() {
        return Err(crate::FrameError::DimensionMismatch {
            expected: out.len(),
            actual: data.len(),
        });
    }
    for (dst, &src) in out.iter_mut().zip(data.iter()) {
        *dst = LUT[src as usize];
    }
    Ok(())
}

pub fn normalise_downscale_into(
    data: &[u8],
    width: u32,
    height: u32,
    factor: crate::DownscaleFactor,
    out: &mut Vec<f32>,
) -> Result<crate::FrameDimensions, crate::DownscaleError> {
    let input_dims = crate::FrameDimensions::try_new(width, height)
        .map_err(crate::DownscaleError::InvalidDimensions)?;
    let expected_len = input_dims.area();
    if data.len() != expected_len {
        return Err(crate::DownscaleError::InputLenMismatch {
            expected: expected_len,
            actual: data.len(),
        });
    }

    let factor_u32 = factor.as_u32();
    if width % factor_u32 != 0 || height % factor_u32 != 0 {
        return Err(crate::DownscaleError::NonDivisible {
            width,
            height,
            factor: factor.get(),
        });
    }

    let out_width = width / factor_u32;
    let out_height = height / factor_u32;
    let out_dims = crate::FrameDimensions::try_new(out_width, out_height)
        .map_err(crate::DownscaleError::InvalidDimensions)?;
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
            out[out_idx] = LUT[data[idx] as usize];
            out_idx += 1;
        }
    }

    Ok(out_dims)
}

pub fn downscale_u8_into(
    data: &[u8],
    width: u32,
    height: u32,
    factor: crate::DownscaleFactor,
    out: &mut Vec<u8>,
) -> Result<crate::FrameDimensions, crate::DownscaleError> {
    let input_dims = crate::FrameDimensions::try_new(width, height)
        .map_err(crate::DownscaleError::InvalidDimensions)?;
    let expected_len = input_dims.area();
    if data.len() != expected_len {
        return Err(crate::DownscaleError::InputLenMismatch {
            expected: expected_len,
            actual: data.len(),
        });
    }

    let factor_u32 = factor.as_u32();
    if width % factor_u32 != 0 || height % factor_u32 != 0 {
        return Err(crate::DownscaleError::NonDivisible {
            width,
            height,
            factor: factor.get(),
        });
    }

    let out_width = width / factor_u32;
    let out_height = height / factor_u32;
    let out_dims = crate::FrameDimensions::try_new(out_width, out_height)
        .map_err(crate::DownscaleError::InvalidDimensions)?;
    out.resize(out_dims.area(), 0);

    let stride = width as usize;
    let step = factor.get();
    let mut out_idx = 0usize;
    for y in 0..out_height as usize {
        let src_y = y * step;
        let row = src_y * stride;
        for x in 0..out_width as usize {
            out[out_idx] = data[row + x * step];
            out_idx += 1;
        }
    }

    Ok(out_dims)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalise_into_rejects_length_mismatch() {
        let mut out = [0.0_f32; 3];
        let err = normalise_into(&[0, 1, 2, 3], &mut out).expect_err("length mismatch");
        assert!(matches!(
            err,
            crate::FrameError::DimensionMismatch {
                expected: 3,
                actual: 4
            }
        ));
    }

    #[test]
    fn normalise_downscale_into_rejects_input_length_mismatch() {
        let err = normalise_downscale_into(
            &[0; 3],
            2,
            2,
            crate::DownscaleFactor::try_from(1).expect("factor"),
            &mut Vec::new(),
        )
        .expect_err("length mismatch");
        assert!(matches!(
            err,
            crate::DownscaleError::InputLenMismatch {
                expected: 4,
                actual: 3
            }
        ));
    }

    #[test]
    fn downscale_u8_into_samples_expected_pixels() {
        let mut out = Vec::new();
        let dims = downscale_u8_into(
            &[0, 1, 2, 3, 4, 5, 6, 7, 8],
            3,
            3,
            crate::DownscaleFactor::try_from(3).expect("factor"),
            &mut out,
        )
        .expect("downscale");
        assert_eq!(dims.width(), 1);
        assert_eq!(dims.height(), 1);
        assert_eq!(out, vec![0]);
    }
}
