pub(crate) fn mat_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut r = [[0.0_f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    r
}

pub(crate) fn mat_mul_vec(r: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ]
}

pub(crate) fn transform_point(r: [[f32; 3]; 3], t: [f32; 3], v: [f32; 3]) -> [f32; 3] {
    let rv = mat_mul_vec(r, v);
    [rv[0] + t[0], rv[1] + t[1], rv[2] + t[2]]
}
