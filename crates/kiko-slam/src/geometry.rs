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

impl<F> Point3<F>
where
    F: CoordinateFrame<Scalar = f32>,
{
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
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
}
