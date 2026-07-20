//! Typed OAK-camera to neutral-head gaze geometry.
//!
//! Frame convention:
//!
//! - OAK camera: `+x` image-right, `+y` image-down, `+z` forward;
//! - neutral head: `+x` yaw-right, `+y` pitch-down, `+z` forward; and
//! - `head_origin_in_camera_m` is the head origin expressed in the OAK camera
//!   frame, in metres.
//!
//! A raw point, ray, depth, or extrinsic is parsed once into a domain type.
//! Projection accepts only those parsed types, so non-finite values, a camera
//! target at non-positive depth, and a degenerate rotation cannot cross the
//! boundary. Kiko-specific metre bounds reject implausible assembly offsets,
//! targets outside the local gaze envelope, and common unit mistakes. A camera
//! ray also needs an explicit bounded positive camera-forward depth: an origin
//! offset makes gaze depend on range, so a direction alone is not a truthful
//! head-relative target.

use core::fmt;

use libm::{atan2, cos, fma, hypot, sin};

/// Largest admitted camera-to-head origin distance, in metres.
///
/// Kiko's current declared assembly offset is about `0.32 m`. A `1 m` limit
/// leaves more than three times that mechanical-layout margin while rejecting
/// common centimetre-as-metre and millimetre-as-metre configuration mistakes.
pub const MAX_HEAD_ORIGIN_DISTANCE_M: f64 = 1.0;

/// Largest absolute camera-target `x` or `y` coordinate, in metres.
///
/// This is a host gaze-policy bound, not an OAK ranging-performance claim.
pub const MAX_CAMERA_TARGET_AXIS_ABS_M: f64 = 10.0;

/// Nearest admitted positive camera-forward target depth, in metres.
pub const MIN_CAMERA_FORWARD_DEPTH_M: f64 = 0.1;

/// Farthest admitted camera-forward target depth, in metres.
///
/// Together with the lateral-axis limit, this catches typical centimetre or
/// millimetre values accidentally supplied to the metre-valued API.
pub const MAX_CAMERA_FORWARD_DEPTH_M: f64 = 10.0;

/// Cartesian component in the documented camera/head axis convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CartesianAxis {
    X,
    Y,
    Z,
}

impl CartesianAxis {
    const ALL: [Self; 3] = [Self::X, Self::Y, Self::Z];

    const fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }
}

/// Component order of a quaternion represented as `[x, y, z, w]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuaternionComponent {
    X,
    Y,
    Z,
    W,
}

impl QuaternionComponent {
    const ALL: [Self; 4] = [Self::X, Self::Y, Self::Z, Self::W];

    const fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
            Self::W => 3,
        }
    }
}

/// Weakly typed deployment input for one camera-to-neutral-head extrinsic.
///
/// The quaternion rotates a camera-frame vector into the neutral-head frame.
/// It need not be unit length; parsing normalizes any finite, non-zero value
/// with scale-first arithmetic that does not overflow for large components.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraToHeadGazeExtrinsicsInput {
    pub head_origin_in_camera_m: [f64; 3],
    pub neutral_head_from_camera_quaternion_xyzw: [f64; 4],
}

/// Why raw camera/head extrinsics cannot become a projection authority.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GazeExtrinsicsParseError {
    NonFiniteHeadOrigin {
        axis: CartesianAxis,
        value_m: f64,
    },
    NonFiniteRotationQuaternion {
        component: QuaternionComponent,
        value: f64,
    },
    HeadOriginDistanceOutOfRange {
        distance_m: f64,
        maximum_m: f64,
    },
    DegenerateRotationQuaternion,
}

impl fmt::Display for GazeExtrinsicsParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid camera-to-head gaze extrinsics: {self:?}"
        )
    }
}

impl core::error::Error for GazeExtrinsicsParseError {}

/// One finite OAK-camera target point, in metres, with positive camera depth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OakCameraTargetPoint {
    coordinates_m: [f64; 3],
}

impl OakCameraTargetPoint {
    pub fn parse(coordinates_m: [f64; 3]) -> Result<Self, CameraGazeTargetError> {
        for axis in [CartesianAxis::X, CartesianAxis::Y] {
            let value_m = coordinates_m[axis.index()];
            if !value_m.is_finite() {
                return Err(CameraGazeTargetError::NonFinitePointCoordinate { axis, value_m });
            }
            if value_m.abs() > MAX_CAMERA_TARGET_AXIS_ABS_M {
                return Err(CameraGazeTargetError::PointCoordinateMagnitudeOutOfRange {
                    axis,
                    value_m,
                    maximum_abs_m: MAX_CAMERA_TARGET_AXIS_ABS_M,
                });
            }
        }
        CameraForwardDepthMeters::parse(coordinates_m[CartesianAxis::Z.index()])?;
        Ok(Self { coordinates_m })
    }

    pub const fn coordinates_m(self) -> [f64; 3] {
        self.coordinates_m
    }
}

// Parsing excludes NaN, so IEEE equality is reflexive for every admitted
// point. Preserve `Eq` for downstream policy types without claiming an order.
impl Eq for OakCameraTargetPoint {}

/// A finite, normalized OAK-camera ray whose forward component is positive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OakCameraTargetRay {
    unit_direction: [f64; 3],
}

// Parsing excludes NaN from the retained normalized direction.
impl Eq for OakCameraTargetRay {}

impl OakCameraTargetRay {
    /// Parse any finite, non-zero positive-forward direction. Magnitude is
    /// deliberately discarded; range is supplied separately as camera depth.
    pub fn parse(direction: [f64; 3]) -> Result<Self, CameraGazeTargetError> {
        for axis in CartesianAxis::ALL {
            let value = direction[axis.index()];
            if !value.is_finite() {
                return Err(CameraGazeTargetError::NonFiniteRayComponent { axis, value });
            }
        }

        let scale = direction
            .iter()
            .fold(0.0_f64, |current, value| current.max(value.abs()));
        if scale == 0.0 {
            return Err(CameraGazeTargetError::DegenerateRay);
        }
        if direction[CartesianAxis::Z.index()] <= 0.0 {
            return Err(CameraGazeTargetError::NonPositiveRayForwardComponent {
                forward: direction[CartesianAxis::Z.index()],
            });
        }

        let scaled = direction.map(|value| value / scale);
        let norm = hypot(hypot(scaled[0], scaled[1]), scaled[2]);
        debug_assert!(norm.is_finite() && norm > 0.0);
        let unit_direction = scaled.map(|value| value / norm);
        if unit_direction[2] <= 0.0 {
            return Err(CameraGazeTargetError::RayDirectionNotRepresentable { direction });
        }
        Ok(Self { unit_direction })
    }

    pub const fn unit_direction(self) -> [f64; 3] {
        self.unit_direction
    }

    /// Intersect this ray with a plane of constant positive camera `z` depth.
    pub fn point_at_forward_depth(
        self,
        depth: CameraForwardDepthMeters,
    ) -> Result<OakCameraTargetPoint, CameraGazeTargetError> {
        let scale = depth.get() / self.unit_direction[2];
        let x_m = self.unit_direction[0] * scale;
        let y_m = self.unit_direction[1] * scale;
        if !scale.is_finite() || !x_m.is_finite() || !y_m.is_finite() {
            return Err(CameraGazeTargetError::RayPointNotRepresentable {
                direction: self.unit_direction,
                depth_m: depth.get(),
            });
        }
        OakCameraTargetPoint::parse([x_m, y_m, depth.get()])
    }
}

/// Positive finite camera-forward depth, in metres.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct CameraForwardDepthMeters(f64);

impl CameraForwardDepthMeters {
    pub fn parse(depth_m: f64) -> Result<Self, CameraGazeTargetError> {
        if !depth_m.is_finite() {
            return Err(CameraGazeTargetError::NonFiniteForwardDepth { depth_m });
        }
        if depth_m <= 0.0 {
            return Err(CameraGazeTargetError::NonPositiveCameraDepth { depth_m });
        }
        if !(MIN_CAMERA_FORWARD_DEPTH_M..=MAX_CAMERA_FORWARD_DEPTH_M).contains(&depth_m) {
            return Err(CameraGazeTargetError::CameraForwardDepthOutOfRange {
                depth_m,
                minimum_m: MIN_CAMERA_FORWARD_DEPTH_M,
                maximum_m: MAX_CAMERA_FORWARD_DEPTH_M,
            });
        }
        Ok(Self(depth_m))
    }

    pub const fn get(self) -> f64 {
        self.0
    }
}

// Parsing excludes NaN and non-positive zero, making equality reflexive.
impl Eq for CameraForwardDepthMeters {}

/// Why weakly typed target geometry cannot become a camera target domain type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CameraGazeTargetError {
    NonFinitePointCoordinate {
        axis: CartesianAxis,
        value_m: f64,
    },
    PointCoordinateMagnitudeOutOfRange {
        axis: CartesianAxis,
        value_m: f64,
        maximum_abs_m: f64,
    },
    NonPositiveCameraDepth {
        depth_m: f64,
    },
    CameraForwardDepthOutOfRange {
        depth_m: f64,
        minimum_m: f64,
        maximum_m: f64,
    },
    NonFiniteRayComponent {
        axis: CartesianAxis,
        value: f64,
    },
    DegenerateRay,
    NonPositiveRayForwardComponent {
        forward: f64,
    },
    RayDirectionNotRepresentable {
        direction: [f64; 3],
    },
    NonFiniteForwardDepth {
        depth_m: f64,
    },
    RayPointNotRepresentable {
        direction: [f64; 3],
        depth_m: f64,
    },
}

impl fmt::Display for CameraGazeTargetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid OAK camera gaze target: {self:?}")
    }
}

impl core::error::Error for CameraGazeTargetError {}

/// Which head-relative angle could not be represented inside its strict open
/// forward-hemisphere interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HeadGazeAngle {
    YawRight,
    PitchDown,
}

/// Parsed, reusable camera-to-neutral-head projection geometry.
///
/// The row-major rotation is precomputed during parsing, so every target uses
/// the same checked extrinsic without quaternion renormalization or repeated
/// validation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraToHeadGazeExtrinsics {
    head_origin_in_camera_m: [f64; 3],
    neutral_head_from_camera_rotation_rows: [[f64; 3]; 3],
}

// Parsing excludes NaN from both retained fields. Quaternion normalization and
// the bounded matrix construction retain only finite values.
impl Eq for CameraToHeadGazeExtrinsics {}

impl CameraToHeadGazeExtrinsics {
    pub fn parse(input: CameraToHeadGazeExtrinsicsInput) -> Result<Self, GazeExtrinsicsParseError> {
        for axis in CartesianAxis::ALL {
            let value_m = input.head_origin_in_camera_m[axis.index()];
            if !value_m.is_finite() {
                return Err(GazeExtrinsicsParseError::NonFiniteHeadOrigin { axis, value_m });
            }
        }
        for component in QuaternionComponent::ALL {
            let value = input.neutral_head_from_camera_quaternion_xyzw[component.index()];
            if !value.is_finite() {
                return Err(GazeExtrinsicsParseError::NonFiniteRotationQuaternion {
                    component,
                    value,
                });
            }
        }

        let [origin_x_m, origin_y_m, origin_z_m] = input.head_origin_in_camera_m;
        let origin_distance_m = hypot(hypot(origin_x_m, origin_y_m), origin_z_m);
        if origin_distance_m > MAX_HEAD_ORIGIN_DISTANCE_M {
            return Err(GazeExtrinsicsParseError::HeadOriginDistanceOutOfRange {
                distance_m: origin_distance_m,
                maximum_m: MAX_HEAD_ORIGIN_DISTANCE_M,
            });
        }

        let raw = input.neutral_head_from_camera_quaternion_xyzw;
        let scale = raw
            .iter()
            .fold(0.0_f64, |current, value| current.max(value.abs()));
        if scale == 0.0 {
            return Err(GazeExtrinsicsParseError::DegenerateRotationQuaternion);
        }
        let scaled = raw.map(|value| value / scale);
        let norm = hypot(hypot(hypot(scaled[0], scaled[1]), scaled[2]), scaled[3]);
        debug_assert!(norm.is_finite() && norm > 0.0);
        let [x, y, z, w] = scaled.map(|value| value / norm);

        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let xz = x * z;
        let yz = y * z;
        let xw = x * w;
        let yw = y * w;
        let zw = z * w;
        let rotation = [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - zw), 2.0 * (xz + yw)],
            [2.0 * (xy + zw), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - xw)],
            [2.0 * (xz - yw), 2.0 * (yz + xw), 1.0 - 2.0 * (xx + yy)],
        ];

        Ok(Self {
            head_origin_in_camera_m: input.head_origin_in_camera_m,
            neutral_head_from_camera_rotation_rows: rotation,
        })
    }

    pub const fn head_origin_in_camera_m(self) -> [f64; 3] {
        self.head_origin_in_camera_m
    }

    pub const fn neutral_head_from_camera_rotation_rows(self) -> [[f64; 3]; 3] {
        self.neutral_head_from_camera_rotation_rows
    }

    /// Transform a positive-depth camera point into neutral-head yaw and pitch.
    ///
    /// The parsed physical bounds make subtraction and matrix multiplication
    /// finite without an overflow fallback. Fused dot products retain more
    /// information near the head's forward plane. A result whose exact
    /// front/plane classification or strict open-interval angle cannot be
    /// represented is rejected rather than rounded into a valid gaze.
    pub fn project_point(
        self,
        target: OakCameraTargetPoint,
    ) -> Result<HeadRelativeGaze, HeadGazeProjectionError> {
        let point = target.coordinates_m;
        let camera_delta_m = [
            point[0] - self.head_origin_in_camera_m[0],
            point[1] - self.head_origin_in_camera_m[1],
            point[2] - self.head_origin_in_camera_m[2],
        ];
        if camera_delta_m == [0.0; 3] {
            return Err(HeadGazeProjectionError::DegenerateHeadRelativeVector);
        }
        let rows = self.neutral_head_from_camera_rotation_rows;
        let head_m = rows.map(|row| dot3_fused(row, camera_delta_m));
        if head_m == [0.0; 3] {
            return Err(
                HeadGazeProjectionError::HeadRelativeVectorNotRepresentable { camera_delta_m },
            );
        }
        if head_m[2] < 0.0 {
            return Err(HeadGazeProjectionError::TargetNotInFrontOfHead {
                head_forward_component_m: head_m[2],
            });
        }
        if head_m[2] == 0.0 {
            return Err(
                HeadGazeProjectionError::HeadForwardClassificationNotRepresentable {
                    head_vector_m: head_m,
                },
            );
        }

        let yaw_right_rad = atan2(head_m[0], head_m[2]);
        let pitch_down_rad = atan2(head_m[1], hypot(head_m[0], head_m[2]));
        ensure_open_forward_angle(HeadGazeAngle::YawRight, yaw_right_rad)?;
        ensure_open_forward_angle(HeadGazeAngle::PitchDown, pitch_down_rad)?;
        Ok(HeadRelativeGaze {
            yaw_right_rad,
            pitch_down_rad,
        })
    }

    pub fn project_ray_at_forward_depth(
        self,
        ray: OakCameraTargetRay,
        depth: CameraForwardDepthMeters,
    ) -> Result<HeadRelativeGaze, RayHeadGazeProjectionError> {
        let target = ray
            .point_at_forward_depth(depth)
            .map_err(RayHeadGazeProjectionError::Target)?;
        self.project_point(target)
            .map_err(RayHeadGazeProjectionError::Projection)
    }
}

fn dot3_fused(left: [f64; 3], right: [f64; 3]) -> f64 {
    fma(
        left[0],
        right[0],
        fma(left[1], right[1], fma(left[2], right[2], 0.0)),
    )
}

fn ensure_open_forward_angle(
    angle: HeadGazeAngle,
    angle_rad: f64,
) -> Result<(), HeadGazeProjectionError> {
    use core::f64::consts::FRAC_PI_2;

    if !angle_rad.is_finite() || angle_rad <= -FRAC_PI_2 || angle_rad >= FRAC_PI_2 {
        return Err(
            HeadGazeProjectionError::HeadAngleNotRepresentableInOpenForwardRange {
                angle,
                angle_rad,
            },
        );
    }
    Ok(())
}

/// Head-relative neutral-axis angles in radians.
///
/// Positive yaw turns toward camera/head `+x` (image-right); positive pitch
/// turns toward `+y` (image-down). Both angles are strictly within
/// `(-pi/2, pi/2)` because targets on or behind the head's `z=0` plane are
/// rejected.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeadRelativeGaze {
    yaw_right_rad: f64,
    pitch_down_rad: f64,
}

// Construction is private and admits only finite angles strictly inside the
// open forward interval, so equality is reflexive.
impl Eq for HeadRelativeGaze {}

impl HeadRelativeGaze {
    pub const fn yaw_right_rad(self) -> f64 {
        self.yaw_right_rad
    }

    pub const fn pitch_down_rad(self) -> f64 {
        self.pitch_down_rad
    }

    /// Reconstruct the corresponding unit direction in the neutral-head frame.
    pub fn unit_direction_in_head(self) -> [f64; 3] {
        let sin_yaw = sin(self.yaw_right_rad);
        let cos_yaw = cos(self.yaw_right_rad);
        let sin_pitch = sin(self.pitch_down_rad);
        let cos_pitch = cos(self.pitch_down_rad);
        [cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw]
    }
}

/// Why a parsed camera target does not define an admissible neutral-head gaze.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HeadGazeProjectionError {
    DegenerateHeadRelativeVector,
    HeadRelativeVectorNotRepresentable {
        camera_delta_m: [f64; 3],
    },
    HeadForwardClassificationNotRepresentable {
        head_vector_m: [f64; 3],
    },
    TargetNotInFrontOfHead {
        head_forward_component_m: f64,
    },
    HeadAngleNotRepresentableInOpenForwardRange {
        angle: HeadGazeAngle,
        angle_rad: f64,
    },
}

impl fmt::Display for HeadGazeProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "cannot project camera target into head gaze: {self:?}"
        )
    }
}

impl core::error::Error for HeadGazeProjectionError {}

/// Composed error for a camera ray whose explicit depth becomes a head gaze.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RayHeadGazeProjectionError {
    Target(CameraGazeTargetError),
    Projection(HeadGazeProjectionError),
}

impl fmt::Display for RayHeadGazeProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "cannot project camera ray and depth into head gaze: {self:?}"
        )
    }
}

impl core::error::Error for RayHeadGazeProjectionError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Target(source) => Some(source),
            Self::Projection(source) => Some(source),
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use core::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2};

    use super::*;

    const TOLERANCE: f64 = 2.0e-14;

    fn assembly_extrinsics() -> CameraToHeadGazeExtrinsics {
        CameraToHeadGazeExtrinsics::parse(CameraToHeadGazeExtrinsicsInput {
            head_origin_in_camera_m: [0.0, -0.25, -0.20],
            neutral_head_from_camera_quaternion_xyzw: [0.0, 0.0, 0.0, 1.0],
        })
        .expect("finite parallel assembly extrinsics")
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= TOLERANCE,
            "actual={actual:.17e}, expected={expected:.17e}"
        );
    }

    #[test]
    fn assembly_translation_uses_metres_and_documented_camera_axes() {
        let extrinsics = assembly_extrinsics();
        assert_eq!(extrinsics.head_origin_in_camera_m(), [0.0, -0.25, -0.20]);
        assert_eq!(
            extrinsics.neutral_head_from_camera_rotation_rows(),
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        );

        let optical_axis = OakCameraTargetPoint::parse([0.0, 0.0, 1.0]).unwrap();
        let gaze = extrinsics.project_point(optical_axis).unwrap();
        assert_eq!(gaze.yaw_right_rad(), 0.0);
        assert_close(gaze.pitch_down_rad(), atan2(0.25, 1.20));

        let level_ahead = OakCameraTargetPoint::parse([0.0, -0.25, 1.0]).unwrap();
        let gaze = extrinsics.project_point(level_ahead).unwrap();
        assert_eq!(gaze.yaw_right_rad(), 0.0);
        assert_eq!(gaze.pitch_down_rad(), 0.0);
    }

    #[test]
    fn yaw_right_and_pitch_down_signs_follow_the_frame_contract() {
        let extrinsics = assembly_extrinsics();
        let right = extrinsics
            .project_point(OakCameraTargetPoint::parse([0.5, -0.25, 1.0]).unwrap())
            .unwrap();
        let left = extrinsics
            .project_point(OakCameraTargetPoint::parse([-0.5, -0.25, 1.0]).unwrap())
            .unwrap();
        let down = extrinsics
            .project_point(OakCameraTargetPoint::parse([0.0, 0.0, 1.0]).unwrap())
            .unwrap();
        let up = extrinsics
            .project_point(OakCameraTargetPoint::parse([0.0, -0.5, 1.0]).unwrap())
            .unwrap();

        assert!(right.yaw_right_rad() > 0.0);
        assert!(left.yaw_right_rad() < 0.0);
        assert!(down.pitch_down_rad() > 0.0);
        assert!(up.pitch_down_rad() < 0.0);
    }

    #[test]
    fn nonfinite_and_degenerate_extrinsics_are_rejected_componentwise() {
        for (axis, value) in
            CartesianAxis::ALL
                .into_iter()
                .zip([f64::NAN, f64::INFINITY, f64::NEG_INFINITY])
        {
            let mut input = CameraToHeadGazeExtrinsicsInput {
                head_origin_in_camera_m: [0.0, -0.25, -0.20],
                neutral_head_from_camera_quaternion_xyzw: [0.0, 0.0, 0.0, 1.0],
            };
            input.head_origin_in_camera_m[axis.index()] = value;
            assert!(matches!(
                CameraToHeadGazeExtrinsics::parse(input),
                Err(GazeExtrinsicsParseError::NonFiniteHeadOrigin {
                    axis: actual,
                    value_m,
                }) if actual == axis && value_m.to_bits() == value.to_bits()
            ));
        }

        for component in QuaternionComponent::ALL {
            let mut quaternion = [0.0, 0.0, 0.0, 1.0];
            quaternion[component.index()] = f64::NAN;
            assert!(matches!(
                CameraToHeadGazeExtrinsics::parse(CameraToHeadGazeExtrinsicsInput {
                    head_origin_in_camera_m: [0.0; 3],
                    neutral_head_from_camera_quaternion_xyzw: quaternion,
                }),
                Err(GazeExtrinsicsParseError::NonFiniteRotationQuaternion {
                    component: actual,
                    value,
                }) if actual == component && value.is_nan()
            ));
        }

        assert_eq!(
            CameraToHeadGazeExtrinsics::parse(CameraToHeadGazeExtrinsicsInput {
                head_origin_in_camera_m: [0.0; 3],
                neutral_head_from_camera_quaternion_xyzw: [0.0; 4],
            }),
            Err(GazeExtrinsicsParseError::DegenerateRotationQuaternion)
        );
    }

    #[test]
    fn head_origin_distance_is_bounded_in_metres() {
        let identity = [0.0, 0.0, 0.0, 1.0];
        CameraToHeadGazeExtrinsics::parse(CameraToHeadGazeExtrinsicsInput {
            head_origin_in_camera_m: [MAX_HEAD_ORIGIN_DISTANCE_M, 0.0, 0.0],
            neutral_head_from_camera_quaternion_xyzw: identity,
        })
        .expect("inclusive one-metre policy boundary");

        for origin_m in [[1.5, 1.5, 0.0], [0.0, -25.0, -20.0]] {
            assert!(matches!(
                CameraToHeadGazeExtrinsics::parse(CameraToHeadGazeExtrinsicsInput {
                    head_origin_in_camera_m: origin_m,
                    neutral_head_from_camera_quaternion_xyzw: identity,
                }),
                Err(GazeExtrinsicsParseError::HeadOriginDistanceOutOfRange {
                    distance_m,
                    maximum_m: MAX_HEAD_ORIGIN_DISTANCE_M,
                }) if distance_m > MAX_HEAD_ORIGIN_DISTANCE_M
            ));
        }
    }

    #[test]
    fn quaternion_normalization_is_scale_stable_and_rotation_is_applied() {
        for scale in [f64::MIN_POSITIVE, 1.0, f64::MAX] {
            let extrinsics = CameraToHeadGazeExtrinsics::parse(CameraToHeadGazeExtrinsicsInput {
                head_origin_in_camera_m: [0.0; 3],
                neutral_head_from_camera_quaternion_xyzw: [
                    0.0,
                    FRAC_1_SQRT_2 * scale,
                    0.0,
                    -FRAC_1_SQRT_2 * scale,
                ],
            })
            .unwrap();
            let gaze = extrinsics
                .project_point(OakCameraTargetPoint::parse([1.0, 0.0, 1.0]).unwrap())
                .unwrap();
            assert_close(gaze.yaw_right_rad(), -FRAC_PI_2 / 2.0);
            assert_eq!(gaze.pitch_down_rad(), 0.0);
        }
    }

    #[test]
    fn targets_rays_and_depths_reject_invalid_boundary_values() {
        for axis in [CartesianAxis::X, CartesianAxis::Y] {
            let mut point = [0.0, 0.0, 1.0];
            point[axis.index()] = f64::NAN;
            assert!(matches!(
                OakCameraTargetPoint::parse(point),
                Err(CameraGazeTargetError::NonFinitePointCoordinate {
                    axis: actual,
                    value_m,
                }) if actual == axis && value_m.is_nan()
            ));
        }
        assert!(matches!(
            OakCameraTargetPoint::parse([0.0, 0.0, f64::NAN]),
            Err(CameraGazeTargetError::NonFiniteForwardDepth { depth_m })
                if depth_m.is_nan()
        ));
        for depth_m in [0.0, -0.0, -1.0] {
            assert_eq!(
                OakCameraTargetPoint::parse([0.0, 0.0, depth_m]),
                Err(CameraGazeTargetError::NonPositiveCameraDepth { depth_m })
            );
        }
        OakCameraTargetPoint::parse([
            MAX_CAMERA_TARGET_AXIS_ABS_M,
            -MAX_CAMERA_TARGET_AXIS_ABS_M,
            MIN_CAMERA_FORWARD_DEPTH_M,
        ])
        .expect("inclusive coordinate and near-depth policy boundaries");
        OakCameraTargetPoint::parse([0.0, 0.0, MAX_CAMERA_FORWARD_DEPTH_M])
            .expect("inclusive far-depth policy boundary");
        assert_eq!(
            OakCameraTargetPoint::parse([MAX_CAMERA_TARGET_AXIS_ABS_M + 1.0, 0.0, 1.0]),
            Err(CameraGazeTargetError::PointCoordinateMagnitudeOutOfRange {
                axis: CartesianAxis::X,
                value_m: MAX_CAMERA_TARGET_AXIS_ABS_M + 1.0,
                maximum_abs_m: MAX_CAMERA_TARGET_AXIS_ABS_M,
            })
        );
        for depth_m in [
            MIN_CAMERA_FORWARD_DEPTH_M / 2.0,
            MAX_CAMERA_FORWARD_DEPTH_M + 1.0,
        ] {
            assert_eq!(
                CameraForwardDepthMeters::parse(depth_m),
                Err(CameraGazeTargetError::CameraForwardDepthOutOfRange {
                    depth_m,
                    minimum_m: MIN_CAMERA_FORWARD_DEPTH_M,
                    maximum_m: MAX_CAMERA_FORWARD_DEPTH_M,
                })
            );
            assert!(matches!(
                OakCameraTargetPoint::parse([0.0, 0.0, depth_m]),
                Err(CameraGazeTargetError::CameraForwardDepthOutOfRange {
                    depth_m: actual,
                    ..
                }) if actual == depth_m
            ));
        }
        assert_eq!(
            OakCameraTargetRay::parse([0.0; 3]),
            Err(CameraGazeTargetError::DegenerateRay)
        );
        assert_eq!(
            OakCameraTargetRay::parse([1.0, 0.0, -1.0]),
            Err(CameraGazeTargetError::NonPositiveRayForwardComponent { forward: -1.0 })
        );
        assert_eq!(
            OakCameraTargetRay::parse([f64::MAX, 0.0, f64::MIN_POSITIVE]),
            Err(CameraGazeTargetError::RayDirectionNotRepresentable {
                direction: [f64::MAX, 0.0, f64::MIN_POSITIVE]
            })
        );
        assert!(matches!(
            CameraForwardDepthMeters::parse(f64::INFINITY),
            Err(CameraGazeTargetError::NonFiniteForwardDepth { depth_m })
                if depth_m == f64::INFINITY
        ));
        assert_eq!(
            CameraForwardDepthMeters::parse(0.0),
            Err(CameraGazeTargetError::NonPositiveCameraDepth { depth_m: 0.0 })
        );
    }

    #[test]
    fn rays_require_depth_and_match_equivalent_points() {
        let extrinsics = assembly_extrinsics();
        for direction in [[0.0, 0.0, 1.0], [0.2, -0.1, 1.0], [-2.0, 1.0, 4.0]] {
            let ray = OakCameraTargetRay::parse(direction).unwrap();
            let depth = CameraForwardDepthMeters::parse(1.5).unwrap();
            let point = ray.point_at_forward_depth(depth).unwrap();
            assert_close(point.coordinates_m()[2], 1.5);
            let from_ray = extrinsics.project_ray_at_forward_depth(ray, depth).unwrap();
            let from_point = extrinsics.project_point(point).unwrap();
            assert_close(from_ray.yaw_right_rad(), from_point.yaw_right_rad());
            assert_close(from_ray.pitch_down_rad(), from_point.pitch_down_rad());
        }
    }

    #[test]
    fn behind_head_and_degenerate_head_relative_targets_are_rejected() {
        let extrinsics = CameraToHeadGazeExtrinsics::parse(CameraToHeadGazeExtrinsicsInput {
            head_origin_in_camera_m: [0.0, 0.0, 1.0],
            neutral_head_from_camera_quaternion_xyzw: [0.0, 0.0, 0.0, 1.0],
        })
        .unwrap();
        assert_eq!(
            extrinsics.project_point(OakCameraTargetPoint::parse([0.0, 0.0, 1.0]).unwrap()),
            Err(HeadGazeProjectionError::DegenerateHeadRelativeVector)
        );
        assert!(matches!(
            extrinsics.project_point(OakCameraTargetPoint::parse([0.0, 0.0, 0.5]).unwrap()),
            Err(HeadGazeProjectionError::TargetNotInFrontOfHead {
                head_forward_component_m,
            }) if head_forward_component_m < 0.0
        ));
    }

    #[test]
    fn near_axis_bounded_coordinates_remain_stable() {
        let identity = CameraToHeadGazeExtrinsics::parse(CameraToHeadGazeExtrinsicsInput {
            head_origin_in_camera_m: [0.0; 3],
            neutral_head_from_camera_quaternion_xyzw: [0.0, 0.0, 0.0, 1.0],
        })
        .unwrap();
        for epsilon in [f64::MIN_POSITIVE, 1.0e-300, 1.0e-20] {
            let gaze = identity
                .project_point(OakCameraTargetPoint::parse([epsilon, -epsilon, 1.0]).unwrap())
                .unwrap();
            assert!(gaze.yaw_right_rad().is_finite() && gaze.yaw_right_rad() > 0.0);
            assert!(gaze.pitch_down_rad().is_finite() && gaze.pitch_down_rad() < 0.0);
        }
    }

    #[test]
    fn unrepresentable_forward_plane_and_strict_angles_are_typed_errors() {
        let smallest_positive = f64::from_bits(1);
        let extrinsics = CameraToHeadGazeExtrinsics::parse(CameraToHeadGazeExtrinsicsInput {
            head_origin_in_camera_m: [0.0, 0.0, 1.0],
            neutral_head_from_camera_quaternion_xyzw: [smallest_positive, 0.0, 0.0, 1.0],
        })
        .unwrap();

        let underflowed_forward = OakCameraTargetPoint::parse([0.0, 0.25, 1.0]).unwrap();
        assert!(matches!(
            extrinsics.project_point(underflowed_forward),
            Err(HeadGazeProjectionError::HeadForwardClassificationNotRepresentable {
                head_vector_m,
            }) if head_vector_m[1] > 0.0 && head_vector_m[2] == 0.0
        ));

        let boundary_angle = OakCameraTargetPoint::parse([0.0, 1.0, 1.0]).unwrap();
        assert_eq!(
            extrinsics.project_point(boundary_angle),
            Err(
                HeadGazeProjectionError::HeadAngleNotRepresentableInOpenForwardRange {
                    angle: HeadGazeAngle::PitchDown,
                    angle_rad: FRAC_PI_2,
                }
            )
        );
    }

    #[test]
    fn translation_precedes_nonidentity_camera_to_head_rotation() {
        let extrinsics = CameraToHeadGazeExtrinsics::parse(CameraToHeadGazeExtrinsicsInput {
            head_origin_in_camera_m: [0.3, -0.4, -0.2],
            neutral_head_from_camera_quaternion_xyzw: [0.0, 0.0, FRAC_1_SQRT_2, FRAC_1_SQRT_2],
        })
        .unwrap();
        let gaze = extrinsics
            .project_point(OakCameraTargetPoint::parse([1.3, 0.1, 1.8]).unwrap())
            .unwrap();

        // camera delta [1, 0.5, 2] rotated +90 degrees around +z is
        // head vector [-0.5, 1, 2]. Rotating the point before subtracting the
        // camera-frame origin produces different angles and fails this check.
        assert_close(gaze.yaw_right_rad(), atan2(-0.5, 2.0));
        assert_close(gaze.pitch_down_rad(), atan2(1.0, hypot(0.5, 2.0)));
    }

    #[test]
    fn parsed_float_domain_types_preserve_reflexive_equality() {
        fn require_eq<T: Eq>() {}

        require_eq::<CameraForwardDepthMeters>();
        require_eq::<OakCameraTargetPoint>();
        require_eq::<OakCameraTargetRay>();
        require_eq::<CameraToHeadGazeExtrinsics>();
        require_eq::<HeadRelativeGaze>();
    }

    #[test]
    fn yaw_pitch_direction_round_trip_holds_over_the_forward_hemisphere() {
        let origin = [0.0, -0.25, -0.20];
        let extrinsics = assembly_extrinsics();
        for yaw_step in -13..=13 {
            for pitch_step in -13..=13 {
                let yaw = f64::from(yaw_step) / 10.0;
                let pitch = f64::from(pitch_step) / 10.0;
                let sin_yaw = sin(yaw);
                let cos_yaw = cos(yaw);
                let sin_pitch = sin(pitch);
                let cos_pitch = cos(pitch);
                let distance_m = 10.0;
                let direction = [cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw];
                let point = OakCameraTargetPoint::parse([
                    origin[0] + distance_m * direction[0],
                    origin[1] + distance_m * direction[1],
                    origin[2] + distance_m * direction[2],
                ])
                .unwrap();
                let gaze = extrinsics.project_point(point).unwrap();
                assert_close(gaze.yaw_right_rad(), yaw);
                assert_close(gaze.pitch_down_rad(), pitch);
                let reconstructed = gaze.unit_direction_in_head();
                for axis in 0..3 {
                    assert_close(reconstructed[axis], direction[axis]);
                }
            }
        }
    }
}
