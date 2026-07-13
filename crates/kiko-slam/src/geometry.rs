use std::marker::PhantomData;

use crate::math::{Pose64, se3_exp_f64, se3_log_f64};
use crate::{Pose, Pose64Error};

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
    NonFiniteScalar {
        context: &'static str,
        value: f64,
    },
    NonPositiveScalar {
        context: &'static str,
        value: f64,
    },
    ZeroNormRay,
    MatrixNotSymmetric {
        max_asymmetry: f64,
    },
    MatrixNotPositiveDefinite,
    NonFiniteTransformRotation {
        operation: &'static str,
        row: usize,
        col: usize,
        value: f64,
    },
    NonFiniteTransformTranslation {
        operation: &'static str,
        axis: usize,
        value: f64,
    },
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
            GeometryError::NonFiniteTransformRotation {
                operation,
                row,
                col,
                value,
            } => write!(
                f,
                "{operation} produced non-finite rotation[{row}][{col}]={value}"
            ),
            GeometryError::NonFiniteTransformTranslation {
                operation,
                axis,
                value,
            } => write!(
                f,
                "{operation} produced non-finite translation axis {axis}={value}"
            ),
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
        let coords = [x, y, z];
        validate_coordinates(coords, ["point.x", "point.y", "point.z"])?;
        Ok(Self {
            coords,
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
        let coords = [x, y, z];
        validate_coordinates(coords, ["vector.x", "vector.y", "vector.z"])?;
        Ok(Self {
            coords,
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
/// let _ = map_from_odom.try_transform_point(body_point);
/// ```
///
/// ```compile_fail
/// use kiko_slam::{BodyFrame, CamLFrame, MapFrame, OdomFrame, Pose64, Transform3d};
///
/// let map_from_odom = Transform3d::<MapFrame, OdomFrame>::from_pose64(Pose64::identity()).unwrap();
/// let cam_from_body = Transform3d::<CamLFrame, BodyFrame>::from_pose64(Pose64::identity()).unwrap();
/// let _ = map_from_odom.try_compose(cam_from_body);
/// ```
///
/// ```compile_fail
/// use kiko_slam::{MapFrame, OdomFrame, Transform3d};
///
/// let _ = Transform3d::<MapFrame, OdomFrame>::identity();
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform3d<To, From> {
    pose: Pose64,
    _to: PhantomData<To>,
    _from: PhantomData<From>,
}

impl<Frame> Transform3d<Frame, Frame> {
    pub fn identity() -> Self {
        Self::identity_between_frames()
    }
}

impl<To, From> Transform3d<To, From> {
    /// Declare two distinct frame origins coincident at an internal trust boundary.
    pub(crate) fn identity_between_frames() -> Self {
        Self {
            pose: Pose64::identity(),
            _to: PhantomData,
            _from: PhantomData,
        }
    }

    pub fn from_pose64(pose: Pose64) -> Result<Self, GeometryError> {
        validate_pose_is_finite(pose, "frame-typed transform construction")?;
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

    pub fn try_inverse(self) -> Result<Transform3d<From, To>, GeometryError> {
        let pose = self.pose.inverse();
        validate_pose_is_finite(pose, "frame-typed transform inversion")?;
        Ok(Transform3d {
            pose,
            _to: PhantomData,
            _from: PhantomData,
        })
    }

    pub fn try_compose<Source>(
        self,
        other: Transform3d<From, Source>,
    ) -> Result<Transform3d<To, Source>, GeometryError> {
        let pose = self.pose.compose(other.pose);
        validate_pose_is_finite(pose, "frame-typed transform composition")?;
        Ok(Transform3d {
            pose,
            _to: PhantomData,
            _from: PhantomData,
        })
    }

    pub fn try_transform_point(self, point: Point3d<From>) -> Result<Point3d<To>, GeometryError> {
        let rotated = mat_mul_vec_f64_local(self.pose.rotation(), point.coords);
        let coords = [
            rotated[0] + self.pose.translation()[0],
            rotated[1] + self.pose.translation()[1],
            rotated[2] + self.pose.translation()[2],
        ];
        validate_coordinates(
            coords,
            [
                "transformed point.x",
                "transformed point.y",
                "transformed point.z",
            ],
        )?;
        Ok(Point3d {
            coords,
            _frame: PhantomData,
        })
    }

    pub fn try_transform_vector(self, vector: Vec3d<From>) -> Result<Vec3d<To>, GeometryError> {
        let coords = mat_mul_vec_f64_local(self.pose.rotation(), vector.coords);
        validate_coordinates(
            coords,
            [
                "transformed vector.x",
                "transformed vector.y",
                "transformed vector.z",
            ],
        )?;
        Ok(Vec3d {
            coords,
            _frame: PhantomData,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Se3TangentPart {
    TranslationTangentMeters,
    RotationVectorRadians,
}

impl std::fmt::Display for Se3TangentPart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TranslationTangentMeters => f.write_str("translation tangent (m)"),
            Self::RotationVectorRadians => f.write_str("rotation vector (rad)"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Se3TangentError {
    NonFiniteComponent {
        part: Se3TangentPart,
        axis: usize,
        value: f64,
    },
    NonFiniteNorm {
        part: Se3TangentPart,
        value: f64,
    },
    InvalidPose {
        operation: &'static str,
        source: Pose64Error,
    },
}

impl std::fmt::Display for Se3TangentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteComponent { part, axis, value } => {
                write!(f, "SE(3) {part} axis {axis} must be finite, got {value}")
            }
            Self::NonFiniteNorm { part, value } => {
                write!(f, "SE(3) {part} norm is not finite: {value}")
            }
            Self::InvalidPose { operation, source } => {
                write!(f, "{operation} is not a representable rigid pose: {source}")
            }
        }
    }
}

impl std::error::Error for Se3TangentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidPose { source, .. } => Some(source),
            Self::NonFiniteComponent { .. } | Self::NonFiniteNorm { .. } => None,
        }
    }
}

/// Finite `se(3)` tangent with translation-tangent components in meters and
/// rotation-vector components in radians.
///
/// The first three components are the logarithm's translation tangent, not
/// necessarily the relative transform's Cartesian translation for large
/// rotations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Se3Tangent64 {
    translation_tangent_m: [f64; 3],
    rotation_vector_rad: [f64; 3],
}

impl Se3Tangent64 {
    pub(crate) fn try_from_meters_radians(components: [f64; 6]) -> Result<Self, Se3TangentError> {
        let [tx, ty, tz, rx, ry, rz] = components;
        let translation_tangent_m = [tx, ty, tz];
        let rotation_vector_rad = [rx, ry, rz];
        validate_se3_components(
            translation_tangent_m,
            Se3TangentPart::TranslationTangentMeters,
        )?;
        validate_se3_components(rotation_vector_rad, Se3TangentPart::RotationVectorRadians)?;
        Ok(Self {
            translation_tangent_m,
            rotation_vector_rad,
        })
    }

    pub(crate) fn try_between_metric_poses(from: Pose, to: Pose) -> Result<Self, Se3TangentError> {
        let from =
            Pose64::try_from_pose32(from).map_err(|source| Se3TangentError::InvalidPose {
                operation: "SE(3) logarithm source pose",
                source,
            })?;
        let to = Pose64::try_from_pose32(to).map_err(|source| Se3TangentError::InvalidPose {
            operation: "SE(3) logarithm destination pose",
            source,
        })?;
        Self::try_from_meters_radians(se3_log_f64(to.compose(from.inverse())))
    }

    pub fn components_m_rad(self) -> [f64; 6] {
        let [tx, ty, tz] = self.translation_tangent_m;
        let [rx, ry, rz] = self.rotation_vector_rad;
        [tx, ty, tz, rx, ry, rz]
    }

    pub fn translation_tangent_m(self) -> [f64; 3] {
        self.translation_tangent_m
    }

    pub fn rotation_vector_rad(self) -> [f64; 3] {
        self.rotation_vector_rad
    }

    pub fn try_translation_tangent_norm_m(self) -> Result<f64, Se3TangentError> {
        try_norm3_f64(
            self.translation_tangent_m,
            Se3TangentPart::TranslationTangentMeters,
        )
    }

    pub fn try_rotation_vector_norm_rad(self) -> Result<f64, Se3TangentError> {
        try_norm3_f64(
            self.rotation_vector_rad,
            Se3TangentPart::RotationVectorRadians,
        )
    }

    /// Apply this tangent as the left update `Exp(delta) * pose`.
    pub(crate) fn try_apply_left_to_metric_pose(self, pose: Pose) -> Result<Pose, Se3TangentError> {
        let pose =
            Pose64::try_from_pose32(pose).map_err(|source| Se3TangentError::InvalidPose {
                operation: "SE(3) left-update base pose",
                source,
            })?;
        se3_exp_f64(self.components_m_rad())
            .compose(pose)
            .try_to_pose32()
            .map_err(|source| Se3TangentError::InvalidPose {
                operation: "SE(3) left-update result",
                source,
            })
    }
}

fn validate_se3_components(
    components: [f64; 3],
    part: Se3TangentPart,
) -> Result<(), Se3TangentError> {
    for (axis, value) in components.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(Se3TangentError::NonFiniteComponent { part, axis, value });
        }
    }
    Ok(())
}

fn try_norm3_f64(vector: [f64; 3], part: Se3TangentPart) -> Result<f64, Se3TangentError> {
    let value = vector[0].hypot(vector[1]).hypot(vector[2]);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(Se3TangentError::NonFiniteNorm { part, value })
    }
}

fn ensure_finite(value: f64, context: &'static str) -> Result<(), GeometryError> {
    if !value.is_finite() {
        return Err(GeometryError::NonFiniteScalar { context, value });
    }
    Ok(())
}

fn validate_coordinates(
    values: [f64; 3],
    contexts: [&'static str; 3],
) -> Result<(), GeometryError> {
    for (value, context) in values.into_iter().zip(contexts) {
        ensure_finite(value, context)?;
    }
    Ok(())
}

fn validate_pose_is_finite(pose: Pose64, operation: &'static str) -> Result<(), GeometryError> {
    for (row, values) in pose.rotation().into_iter().enumerate() {
        for (col, value) in values.into_iter().enumerate() {
            if !value.is_finite() {
                return Err(GeometryError::NonFiniteTransformRotation {
                    operation,
                    row,
                    col,
                    value,
                });
            }
        }
    }
    for (axis, value) in pose.translation().into_iter().enumerate() {
        if !value.is_finite() {
            return Err(GeometryError::NonFiniteTransformTranslation {
                operation,
                axis,
                value,
            });
        }
    }
    Ok(())
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
        BodyFrame, Cov3, GeometryError, MapFrame, OdomFrame, Point3d, PositiveF64, Se3Tangent64,
        Se3TangentError, Se3TangentPart, StdDev, Transform3d, UnitRay3d, Variance, Vec3d,
    };
    use crate::{Pose, Pose64, Pose64Error};

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
    fn se3_tangent_rejects_non_finite_components_with_units() {
        assert!(matches!(
            Se3Tangent64::try_from_meters_radians([0.0, 0.0, 0.0, 0.0, f64::NAN, 0.0]),
            Err(Se3TangentError::NonFiniteComponent {
                part: Se3TangentPart::RotationVectorRadians,
                axis: 1,
                value,
            }) if value.is_nan()
        ));
    }

    #[test]
    fn se3_tangent_round_trips_left_update_and_log() {
        let components = [0.2, -0.1, 0.05, 0.03, -0.02, 0.01];
        let tangent = Se3Tangent64::try_from_meters_radians(components).expect("finite tangent");
        let updated = tangent
            .try_apply_left_to_metric_pose(Pose::identity())
            .expect("representable update");
        let recovered = Se3Tangent64::try_between_metric_poses(Pose::identity(), updated)
            .expect("finite recovered tangent")
            .components_m_rad();

        for (axis, (actual, expected)) in recovered.into_iter().zip(components).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "tangent component {axis}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn se3_tangent_reports_unrepresentable_left_update_result() {
        let tangent = Se3Tangent64::try_from_meters_radians([f64::MAX, 0.0, 0.0, 0.0, 0.0, 0.0])
            .expect("finite f64 tangent");

        assert!(matches!(
            tangent.try_apply_left_to_metric_pose(Pose::identity()),
            Err(Se3TangentError::InvalidPose {
                operation: "SE(3) left-update result",
                source: Pose64Error::TranslationOutOfF32Range {
                    axis: 0,
                    value,
                },
            }) if value == f64::MAX
        ));
    }

    #[test]
    fn se3_tangent_reports_unrepresentable_norm() {
        let tangent =
            Se3Tangent64::try_from_meters_radians([f64::MAX, f64::MAX, 0.0, 0.0, 0.0, 0.0])
                .expect("finite components");

        assert!(matches!(
            tangent.try_translation_tangent_norm_m(),
            Err(Se3TangentError::NonFiniteNorm {
                part: Se3TangentPart::TranslationTangentMeters,
                value,
            }) if value.is_infinite()
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
        let map_point = map_from_odom
            .try_transform_point(odom_point)
            .expect("map point");
        let recovered = map_from_odom
            .try_inverse()
            .expect("inverse")
            .try_transform_point(map_point)
            .expect("recovered point");
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

    #[test]
    fn transform_composition_rejects_finite_input_translation_overflow() {
        let max_translation = Pose64::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [f64::MAX, 0.0, 0.0],
        );
        let map_from_odom =
            Transform3d::<MapFrame, OdomFrame>::from_pose64(max_translation).expect("finite pose");
        let odom_from_body =
            Transform3d::<OdomFrame, BodyFrame>::from_pose64(max_translation).expect("finite pose");

        assert!(matches!(
            map_from_odom.try_compose(odom_from_body),
            Err(GeometryError::NonFiniteTransformTranslation {
                operation: "frame-typed transform composition",
                axis: 0,
                value,
            }) if value.is_infinite()
        ));
    }

    #[test]
    fn transform_point_rejects_finite_input_coordinate_overflow() {
        let pose = Pose64::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [f64::MAX, 0.0, 0.0],
        );
        let map_from_odom =
            Transform3d::<MapFrame, OdomFrame>::from_pose64(pose).expect("finite pose");
        let point = Point3d::<OdomFrame>::try_from_xyz(f64::MAX, 0.0, 0.0).expect("finite point");

        assert!(matches!(
            map_from_odom.try_transform_point(point),
            Err(GeometryError::NonFiniteScalar {
                context: "transformed point.x",
                value,
            }) if value.is_infinite()
        ));
    }

    #[test]
    fn transform_inverse_rejects_finite_translation_overflow() {
        let half_sqrt_two = std::f64::consts::FRAC_1_SQRT_2;
        let pose = Pose64::try_from_rt(
            [
                [half_sqrt_two, -half_sqrt_two, 0.0],
                [half_sqrt_two, half_sqrt_two, 0.0],
                [0.0, 0.0, 1.0],
            ],
            [f64::MAX, f64::MAX, 0.0],
        )
        .expect("valid finite pose");
        let map_from_odom =
            Transform3d::<MapFrame, OdomFrame>::from_pose64(pose).expect("finite pose");

        assert!(matches!(
            map_from_odom.try_inverse(),
            Err(GeometryError::NonFiniteTransformTranslation {
                operation: "frame-typed transform inversion",
                value,
                ..
            }) if !value.is_finite()
        ));
    }

    #[test]
    fn transform_vector_rejects_finite_coordinate_overflow() {
        let half_sqrt_two = std::f64::consts::FRAC_1_SQRT_2;
        let pose = Pose64::try_from_rt(
            [
                [half_sqrt_two, -half_sqrt_two, 0.0],
                [half_sqrt_two, half_sqrt_two, 0.0],
                [0.0, 0.0, 1.0],
            ],
            [0.0; 3],
        )
        .expect("valid finite pose");
        let map_from_odom =
            Transform3d::<MapFrame, OdomFrame>::from_pose64(pose).expect("finite pose");
        let vector =
            Vec3d::<OdomFrame>::try_from_xyz(f64::MAX, f64::MAX, 0.0).expect("finite vector");

        assert!(matches!(
            map_from_odom.try_transform_vector(vector),
            Err(GeometryError::NonFiniteScalar {
                context: "transformed vector.y",
                value,
            }) if value.is_infinite()
        ));
    }
}
