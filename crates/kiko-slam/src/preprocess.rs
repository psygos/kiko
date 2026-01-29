// since camera outputs a u8, we can just precompute all the float conversion and static it into
// the binary
const fn build_float_lut() -> [f32; 256] {
    let mut table = [0 as f32; 256];
    let mut i: usize = 0;

    while i < 256 {
        table[i] = i as f32 / 255.0;
        i += 1
    }
    table
}

static LUT: [f32; 256] = build_float_lut();

pub fn normalise(data: &[u8]) -> Vec<f32> {
    data.iter().map(|&b| LUT[b as usize]).collect()
}

pub fn normalise_downscale(
    data: &[u8],
    width: u32,
    height: u32,
    factor: crate::DownscaleFactor,
) -> Result<(Vec<f32>, u32, u32), crate::DownscaleError> {
    let factor_u32 = factor.get() as u32;
    if width % factor_u32 != 0 || height % factor_u32 != 0 {
        return Err(crate::DownscaleError::NonDivisible {
            width,
            height,
            factor: factor.get(),
        });
    }

    let out_width = width / factor_u32;
    let out_height = height / factor_u32;
    let mut out = Vec::with_capacity((out_width * out_height) as usize);

    let stride = width as usize;
    let step = factor.get();

    for y in 0..out_height as usize {
        let src_y = y * step;
        let row = src_y * stride;
        for x in 0..out_width as usize {
            let src_x = x * step;
            let idx = row + src_x;
            out.push(LUT[data[idx] as usize]);
        }
    }

    Ok((out, out_width, out_height))
}
