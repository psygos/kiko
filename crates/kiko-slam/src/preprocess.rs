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
