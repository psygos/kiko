mod sealed {
    pub trait Sealed {}
}

/// A coordinate frame used to distinguish geometrically incompatible values.
pub trait CoordinateFrame: sealed::Sealed {
    #[doc(hidden)]
    type Scalar;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraFrame {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorldFrame {}

impl sealed::Sealed for CameraFrame {}
impl sealed::Sealed for WorldFrame {}

impl CoordinateFrame for CameraFrame {
    type Scalar = f32;
}

impl CoordinateFrame for WorldFrame {
    type Scalar = f32;
}

/// A 3D point whose coordinate frame is part of its type.
///
/// The default is the world frame to keep map and optimizer code concise. Code
/// that handles camera-local geometry must name [`CameraPoint3`] explicitly.
/// This legacy carrier keeps public coordinates for source compatibility, so
/// finiteness is not a type-level invariant. Boundary code that requires a
/// finite point must parse with [`Point3::try_new`], [`Point3::try_from_array`],
/// or [`Point3::validate`]. Map storage and checked transforms do so before
/// accepting a point.
///
/// ```compile_fail
/// use kiko_slam::{CameraPoint3, WorldPoint3};
///
/// fn store_map_point(_: WorldPoint3) {}
/// store_map_point(CameraPoint3::new(1.0, 2.0, 3.0));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point3<F: CoordinateFrame = WorldFrame> {
    pub x: F::Scalar,
    pub y: F::Scalar,
    pub z: F::Scalar,
}

/// Error returned when weakly typed coordinates cannot be parsed as a finite
/// 3D point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Point3Error {
    NonFinite { axis: usize, value: f32 },
    NotRepresentable { axis: usize, value: f64 },
}

impl std::fmt::Display for Point3Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite { axis, value } => {
                write!(
                    f,
                    "point coordinate on axis {axis} must be finite, got {value}"
                )
            }
            Self::NotRepresentable { axis, value } => write!(
                f,
                "point coordinate on axis {axis} is not representable as a finite f32: {value}"
            ),
        }
    }
}

impl std::error::Error for Point3Error {}

impl<F> Point3<F>
where
    F: CoordinateFrame<Scalar = f32>,
{
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Parse weakly typed coordinates into a finite point.
    pub fn try_new(x: f32, y: f32, z: f32) -> Result<Self, Point3Error> {
        Self::try_from_array([x, y, z])
    }

    pub const fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    pub const fn from_array(coordinates: [f32; 3]) -> Self {
        Self {
            x: coordinates[0],
            y: coordinates[1],
            z: coordinates[2],
        }
    }

    /// Parse a coordinate array into a finite point.
    pub fn try_from_array(coordinates: [f32; 3]) -> Result<Self, Point3Error> {
        if let Some(axis) = coordinates.iter().position(|value| !value.is_finite()) {
            return Err(Point3Error::NonFinite {
                axis,
                value: coordinates[axis],
            });
        }
        Ok(Self::from_array(coordinates))
    }

    /// Narrow computed f64 coordinates exactly once at the point boundary.
    pub fn try_from_f64(coordinates: [f64; 3]) -> Result<Self, Point3Error> {
        let mut narrowed = [0.0_f32; 3];
        for (axis, output) in narrowed.iter_mut().enumerate() {
            let value = coordinates[axis];
            let candidate = value as f32;
            if !value.is_finite() || !candidate.is_finite() {
                return Err(Point3Error::NotRepresentable { axis, value });
            }
            *output = candidate;
        }
        Ok(Self::from_array(narrowed))
    }

    pub fn validate(self) -> Result<Self, Point3Error> {
        Self::try_from_array(self.to_array())
    }
}

pub type CameraPoint3 = Point3<CameraFrame>;
pub type WorldPoint3 = Point3<WorldFrame>;

#[cfg(test)]
mod tests {
    use super::*;

    fn accepts_camera_point(_: CameraPoint3) {}
    fn accepts_world_point(_: WorldPoint3) {}

    #[test]
    fn point_frame_is_preserved_by_construction() {
        let camera = CameraPoint3::new(1.0, 2.0, 3.0);
        let world = WorldPoint3::from_array([4.0, 5.0, 6.0]);

        accepts_camera_point(camera);
        accepts_world_point(world);
        assert_eq!(camera.to_array(), [1.0, 2.0, 3.0]);
        assert_eq!(world.to_array(), [4.0, 5.0, 6.0]);
    }

    #[test]
    fn checked_construction_rejects_each_nonfinite_axis() {
        for (axis, value) in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY]
            .into_iter()
            .enumerate()
        {
            let mut coordinates = [1.0, 2.0, 3.0];
            coordinates[axis] = value;
            assert!(matches!(
                WorldPoint3::try_from_array(coordinates),
                Err(Point3Error::NonFinite {
                    axis: error_axis,
                    value: error_value,
                }) if error_axis == axis && error_value.to_bits() == value.to_bits()
            ));
        }
    }

    #[test]
    fn f64_construction_rejects_finite_values_outside_f32_range() {
        let value = f64::from(f32::MAX) * 2.0;
        assert_eq!(
            WorldPoint3::try_from_f64([0.0, value, 1.0]),
            Err(Point3Error::NotRepresentable { axis: 1, value })
        );
    }
}
