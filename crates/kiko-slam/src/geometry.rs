use std::marker::PhantomData;

use crate::math::Pose64;

const MATRIX_SYMMETRY_EPSILON: f64 = 1e-12;
const UNIT_RAY_NORM_EPSILON: f64 = 1e-12;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MapFrame;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OdomFrame;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BodyFrame;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CamLFrame;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CamRFrame;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VoxelFrame;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageFrame;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GeometryError {
    NonFiniteScalar { context: &'static str, value: f64 },
    NonPositiveScalar { context: &'static str, value: f64 },
    ZeroNormRay,
    MatrixNotSymmetric { max_asymmetry: f64 },
    MatrixNotPositiveDefinite,
    NonFiniteTransform,
}

impl std::fmt::Display for GeometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeometryError::NonFiniteScalar { context, value } => {
                write!(f, "{context} must be finite, got {value}")
            }
            GeometryError::NonPositiveScalar { context, value } => {
                write!(f, "{context} must be > 0, got {value}")
            }
            GeometryError::ZeroNormRay => write!(f, "unit ray direction must have non-zero norm"),
            GeometryError::MatrixNotSymmetric { max_asymmetry } => write!(
                f,
                "matrix must be symmetric within tolerance, max asymmetry={max_asymmetry:e}"
            ),
            GeometryError::MatrixNotPositiveDefinite => {
                write!(f, "matrix must be positive definite")
            }
            GeometryError::NonFiniteTransform => {
                write!(f, "transform must contain only finite values")
            }
        }
    }
}

impl std::error::Error for GeometryError {}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct PositiveF64(f64);

impl PositiveF64 {
    pub fn new(value: f64, context: &'static str) -> Result<Self, GeometryError> {
        ensure_finite(value, context)?;
        if value <= 0.0 {
            return Err(GeometryError::NonPositiveScalar { context, value });
        }
        Ok(Self(value))
    }

    pub fn get(self) -> f64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Variance(PositiveF64);

impl Variance {
    pub fn new(value: f64) -> Result<Self, GeometryError> {
        Ok(Self(PositiveF64::new(value, "variance")?))
    }

    pub fn from_std_dev(std_dev: StdDev) -> Self {
        let value = std_dev.get();
        Self(PositiveF64(value * value))
    }

    pub fn get(self) -> f64 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct StdDev(PositiveF64);

impl StdDev {
    pub fn new(value: f64) -> Result<Self, GeometryError> {
        Ok(Self(PositiveF64::new(value, "standard deviation")?))
    }

    pub fn from_variance(variance: Variance) -> Self {
        Self(PositiveF64(variance.get().sqrt()))
    }

    pub fn get(self) -> f64 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point3d<Frame> {
    coords: [f64; 3],
    _frame: PhantomData<Frame>,
}

impl<Frame> Point3d<Frame> {
    pub fn try_from_xyz(x: f64, y: f64, z: f64) -> Result<Self, GeometryError> {
        ensure_finite(x, "point.x")?;
        ensure_finite(y, "point.y")?;
        ensure_finite(z, "point.z")?;
        Ok(Self {
            coords: [x, y, z],
            _frame: PhantomData,
        })
    }

    pub fn try_from_array(coords: [f64; 3]) -> Result<Self, GeometryError> {
        Self::try_from_xyz(coords[0], coords[1], coords[2])
    }

    pub fn as_array(self) -> [f64; 3] {
        self.coords
    }

    pub fn x(self) -> f64 {
        self.coords[0]
    }

    pub fn y(self) -> f64 {
        self.coords[1]
    }

    pub fn z(self) -> f64 {
        self.coords[2]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3d<Frame> {
    coords: [f64; 3],
    _frame: PhantomData<Frame>,
}

impl<Frame> Vec3d<Frame> {
    pub fn try_from_xyz(x: f64, y: f64, z: f64) -> Result<Self, GeometryError> {
        ensure_finite(x, "vector.x")?;
        ensure_finite(y, "vector.y")?;
        ensure_finite(z, "vector.z")?;
        Ok(Self {
            coords: [x, y, z],
            _frame: PhantomData,
        })
    }

    pub fn try_from_array(coords: [f64; 3]) -> Result<Self, GeometryError> {
        Self::try_from_xyz(coords[0], coords[1], coords[2])
    }

    pub fn as_array(self) -> [f64; 3] {
        self.coords
    }

    pub fn dot(&self, other: &Self) -> f64 {
        self.coords[0] * other.coords[0]
            + self.coords[1] * other.coords[1]
            + self.coords[2] * other.coords[2]
    }

    pub fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitRay3d<Frame> {
    direction: [f64; 3],
    _frame: PhantomData<Frame>,
}

impl<Frame> UnitRay3d<Frame> {
    pub fn try_from_vector(vector: Vec3d<Frame>) -> Result<Self, GeometryError> {
        let norm = vector.norm();
        if norm <= UNIT_RAY_NORM_EPSILON {
            return Err(GeometryError::ZeroNormRay);
        }
        Ok(Self {
            direction: [
                vector.coords[0] / norm,
                vector.coords[1] / norm,
                vector.coords[2] / norm,
            ],
            _frame: PhantomData,
        })
    }

    pub fn direction(self) -> Vec3d<Frame> {
        Vec3d {
            coords: self.direction,
            _frame: PhantomData,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cov3<Frame> {
    matrix: [[f64; 3]; 3],
    _frame: PhantomData<Frame>,
}

impl<Frame> Cov3<Frame> {
    pub fn try_from_array(matrix: [[f64; 3]; 3]) -> Result<Self, GeometryError> {
        validate_spd_3x3(matrix)?;
        Ok(Self {
            matrix,
            _frame: PhantomData,
        })
    }

    pub fn as_array(self) -> [[f64; 3]; 3] {
        self.matrix
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Info3<Frame> {
    matrix: [[f64; 3]; 3],
    _frame: PhantomData<Frame>,
}

impl<Frame> Info3<Frame> {
    pub fn try_from_array(matrix: [[f64; 3]; 3]) -> Result<Self, GeometryError> {
        validate_spd_3x3(matrix)?;
        Ok(Self {
            matrix,
            _frame: PhantomData,
        })
    }

    pub fn as_array(self) -> [[f64; 3]; 3] {
        self.matrix
    }
}

/// Frame-safe rigid transform in `f64`.
///
/// ```compile_fail
/// use kiko_slam::{BodyFrame, MapFrame, OdomFrame, Point3d, Pose64, Transform3d};
///
/// let map_from_odom = Transform3d::<MapFrame, OdomFrame>::from_pose64(Pose64::identity()).unwrap();
/// let body_point = Point3d::<BodyFrame>::try_from_xyz(1.0, 2.0, 3.0).unwrap();
/// let _ = map_from_odom.transform_point(body_point);
/// ```
///
/// ```compile_fail
/// use kiko_slam::{BodyFrame, CamLFrame, MapFrame, OdomFrame, Pose64, Transform3d};
///
/// let map_from_odom = Transform3d::<MapFrame, OdomFrame>::from_pose64(Pose64::identity()).unwrap();
/// let cam_from_body = Transform3d::<CamLFrame, BodyFrame>::from_pose64(Pose64::identity()).unwrap();
/// let _ = map_from_odom.compose(cam_from_body);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform3d<To, From> {
    pose: Pose64,
    _to: PhantomData<To>,
    _from: PhantomData<From>,
}

impl<Frame> Transform3d<Frame, Frame> {
    pub fn identity() -> Self {
        Self {
            pose: Pose64::identity(),
            _to: PhantomData,
            _from: PhantomData,
        }
    }
}

impl<To, From> Transform3d<To, From> {
    pub fn from_pose64(pose: Pose64) -> Result<Self, GeometryError> {
        if !pose_is_finite(pose) {
            return Err(GeometryError::NonFiniteTransform);
        }
        Ok(Self {
            pose,
            _to: PhantomData,
            _from: PhantomData,
        })
    }

    pub fn into_pose64(self) -> Pose64 {
        self.pose
    }

    pub fn rotation(self) -> [[f64; 3]; 3] {
        self.pose.rotation()
    }

    pub fn translation(self) -> Vec3d<To> {
        Vec3d {
            coords: self.pose.translation(),
            _frame: PhantomData,
        }
    }

    pub fn inverse(self) -> Transform3d<From, To> {
        Transform3d {
            pose: self.pose.inverse(),
            _to: PhantomData,
            _from: PhantomData,
        }
    }

    pub fn compose<Source>(self, other: Transform3d<From, Source>) -> Transform3d<To, Source> {
        Transform3d {
            pose: self.pose.compose(other.pose),
            _to: PhantomData,
            _from: PhantomData,
        }
    }

    pub fn transform_point(self, point: Point3d<From>) -> Point3d<To> {
        let rotated = mat_mul_vec_f64_local(self.pose.rotation(), point.coords);
        Point3d {
            coords: [
                rotated[0] + self.pose.translation()[0],
                rotated[1] + self.pose.translation()[1],
                rotated[2] + self.pose.translation()[2],
            ],
            _frame: PhantomData,
        }
    }

    pub fn transform_vector(self, vector: Vec3d<From>) -> Vec3d<To> {
        Vec3d {
            coords: mat_mul_vec_f64_local(self.pose.rotation(), vector.coords),
            _frame: PhantomData,
        }
    }
}

fn ensure_finite(value: f64, context: &'static str) -> Result<(), GeometryError> {
    if !value.is_finite() {
        return Err(GeometryError::NonFiniteScalar { context, value });
    }
    Ok(())
}

fn pose_is_finite(pose: Pose64) -> bool {
    pose.rotation().into_iter().flatten().all(f64::is_finite)
        && pose.translation().into_iter().all(f64::is_finite)
}

fn mat_mul_vec_f64_local(r: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ]
}

fn validate_spd_3x3(matrix: [[f64; 3]; 3]) -> Result<(), GeometryError> {
    for row in matrix {
        for value in row {
            ensure_finite(value, "matrix entry")?;
        }
    }

    let max_asymmetry = (0..3)
        .flat_map(|row| (0..3).map(move |col| (matrix[row][col] - matrix[col][row]).abs()))
        .fold(0.0_f64, f64::max);
    if max_asymmetry > MATRIX_SYMMETRY_EPSILON {
        return Err(GeometryError::MatrixNotSymmetric { max_asymmetry });
    }

    let _ = cholesky_3x3(matrix)?;
    Ok(())
}

fn cholesky_3x3(matrix: [[f64; 3]; 3]) -> Result<[[f64; 3]; 3], GeometryError> {
    let mut lower = [[0.0_f64; 3]; 3];
    for row in 0..3 {
        for col in 0..=row {
            let mut sum = matrix[row][col];
            for k in 0..col {
                sum -= lower[row][k] * lower[col][k];
            }
            if row == col {
                if sum <= 0.0 {
                    return Err(GeometryError::MatrixNotPositiveDefinite);
                }
                lower[row][col] = sum.sqrt();
            } else {
                if lower[col][col] <= 0.0 {
                    return Err(GeometryError::MatrixNotPositiveDefinite);
                }
                lower[row][col] = sum / lower[col][col];
            }
        }
    }
    Ok(lower)
}

#[cfg(test)]
mod tests {
    use super::{
        BodyFrame, Cov3, GeometryError, MapFrame, OdomFrame, Point3d, PositiveF64, StdDev,
        Transform3d, UnitRay3d, Variance, Vec3d,
    };
    use crate::Pose64;

    #[test]
    fn positive_scalar_and_uncertainty_wrappers_are_lawful() {
        let positive = PositiveF64::new(2.0, "test").expect("positive");
        assert_eq!(positive.get(), 2.0);

        let variance = Variance::new(9.0).expect("variance");
        let std_dev = StdDev::from_variance(variance);
        assert!((std_dev.get() - 3.0).abs() < 1e-12);
        let variance_round_trip = Variance::from_std_dev(std_dev);
        assert!((variance_round_trip.get() - 9.0).abs() < 1e-12);
    }

    #[test]
    fn point_and_vector_reject_non_finite_values() {
        assert!(matches!(
            Point3d::<MapFrame>::try_from_xyz(f64::NAN, 0.0, 0.0),
            Err(GeometryError::NonFiniteScalar { .. })
        ));
        assert!(matches!(
            Vec3d::<MapFrame>::try_from_xyz(0.0, f64::INFINITY, 0.0),
            Err(GeometryError::NonFiniteScalar { .. })
        ));
    }

    #[test]
    fn unit_ray_normalizes_direction() {
        let vector = Vec3d::<BodyFrame>::try_from_xyz(0.0, 3.0, 4.0).expect("vector");
        let ray = UnitRay3d::try_from_vector(vector).expect("ray");
        let direction = ray.direction().as_array();
        assert!((direction[0] - 0.0).abs() < 1e-12);
        assert!((direction[1] - 0.6).abs() < 1e-12);
        assert!((direction[2] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn transform_round_trip_preserves_point() {
        let pose = Pose64::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            [1.0, 2.0, 3.0],
        );
        let map_from_odom = Transform3d::<MapFrame, OdomFrame>::from_pose64(pose).expect("pose");
        let odom_point = Point3d::<OdomFrame>::try_from_xyz(0.5, -1.0, 2.0).expect("point");
        let map_point = map_from_odom.transform_point(odom_point);
        let recovered = map_from_odom.inverse().transform_point(map_point);
        let recovered = recovered.as_array();
        assert!((recovered[0] - 0.5).abs() < 1e-12);
        assert!((recovered[1] + 1.0).abs() < 1e-12);
        assert!((recovered[2] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn covariance_requires_spd_matrix() {
        Cov3::<MapFrame>::try_from_array([[2.0, 0.1, 0.0], [0.1, 1.5, 0.0], [0.0, 0.0, 0.5]])
            .expect("spd");

        assert!(matches!(
            Cov3::<MapFrame>::try_from_array([[1.0, 0.5, 0.0], [0.25, 1.0, 0.0], [0.0, 0.0, 1.0],]),
            Err(GeometryError::MatrixNotSymmetric { .. })
        ));

        assert!(matches!(
            Cov3::<MapFrame>::try_from_array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0],]),
            Err(GeometryError::MatrixNotPositiveDefinite)
        ));
    }
}
