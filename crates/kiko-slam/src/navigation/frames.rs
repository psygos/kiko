//! Strongly typed planar coordinate frames and checked SE(2) transforms.

use std::marker::PhantomData;

mod sealed {
    pub trait Sealed {}
}

/// A coordinate frame admitted by the planar navigation boundary.
///
/// This trait is sealed so external code cannot accidentally introduce an
/// unreviewed frame while still using the checked transform API.
pub trait PlanarFrame:
    sealed::Sealed + Clone + Copy + std::fmt::Debug + Eq + Send + Sync + 'static
{
    const NAME: &'static str;
}

/// The metric frame in which the displayed occupancy map is expressed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MapFrame;

/// The locally continuous frame used by state estimation and control.
///
/// Global corrections belong in an odom-to-map transform rather than as jumps
/// in this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct OdomFrame;

/// The robot body/base frame used by planar navigation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BaseFrame;

/// The robot-base-aligned frame frozen at one local-costmap capture time.
///
/// This is deliberately distinct from [`BaseFrame`]: a point expressed in the
/// robot's current moving base frame cannot be used to query an older local
/// costmap without an explicit time-aligned transform.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LocalCostmapFrame;

impl sealed::Sealed for MapFrame {}
impl sealed::Sealed for OdomFrame {}
impl sealed::Sealed for BaseFrame {}
impl sealed::Sealed for LocalCostmapFrame {}

impl PlanarFrame for MapFrame {
    const NAME: &'static str = "map";
}

impl PlanarFrame for OdomFrame {
    const NAME: &'static str = "odom";
}

impl PlanarFrame for BaseFrame {
    const NAME: &'static str = "base";
}

impl PlanarFrame for LocalCostmapFrame {
    const NAME: &'static str = "local_costmap_at_capture";
}

/// A finite point, in metres, expressed in exactly one planar frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanarPoint<Frame: PlanarFrame> {
    x_m: f64,
    y_m: f64,
    frame: PhantomData<fn() -> Frame>,
}

impl<Frame: PlanarFrame> PlanarPoint<Frame> {
    pub fn try_new(x_m: f64, y_m: f64) -> Result<Self, PlanarPointError> {
        for (axis, value) in [(PlanarAxis::X, x_m), (PlanarAxis::Y, y_m)] {
            if !value.is_finite() {
                return Err(PlanarPointError::NonFiniteCoordinate {
                    frame: Frame::NAME,
                    axis,
                    value,
                });
            }
        }
        Ok(Self {
            x_m,
            y_m,
            frame: PhantomData,
        })
    }

    pub fn origin() -> Self {
        Self {
            x_m: 0.0,
            y_m: 0.0,
            frame: PhantomData,
        }
    }

    pub fn x_m(self) -> f64 {
        self.x_m
    }

    pub fn y_m(self) -> f64 {
        self.y_m
    }

    pub fn as_array(self) -> [f64; 2] {
        [self.x_m, self.y_m]
    }

    pub fn frame_name() -> &'static str {
        Frame::NAME
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlanarPointError {
    NonFiniteCoordinate {
        frame: &'static str,
        axis: PlanarAxis,
        value: f64,
    },
}

impl std::fmt::Display for PlanarPointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteCoordinate { frame, axis, value } => write!(
                f,
                "{frame}-frame point coordinate {axis:?} must be finite metres, got {value}"
            ),
        }
    }
}

impl std::error::Error for PlanarPointError {}

/// A checked transform that maps source-frame coordinates into destination-
/// frame coordinates.
///
/// For `T: PlanarTransform<From, To>`, points are transformed as
/// `p_to = R(source_yaw_in_destination) * p_from + source_origin_in_destination`.
/// Positive yaw rotates +x toward +y.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanarTransform<From: PlanarFrame, To: PlanarFrame> {
    source_origin_x_in_destination_m: f64,
    source_origin_y_in_destination_m: f64,
    source_yaw_in_destination_rad: f64,
    frames: PhantomData<fn(From) -> To>,
}

impl<From: PlanarFrame, To: PlanarFrame> PlanarTransform<From, To> {
    pub fn try_new(
        source_origin_x_in_destination_m: f64,
        source_origin_y_in_destination_m: f64,
        source_yaw_in_destination_rad: f64,
    ) -> Result<Self, PlanarTransformError> {
        for (component, value) in [
            (
                PlanarTransformComponent::SourceOriginXInDestination,
                source_origin_x_in_destination_m,
            ),
            (
                PlanarTransformComponent::SourceOriginYInDestination,
                source_origin_y_in_destination_m,
            ),
            (
                PlanarTransformComponent::SourceYawInDestination,
                source_yaw_in_destination_rad,
            ),
        ] {
            if !value.is_finite() {
                return Err(PlanarTransformError::NonFiniteInput {
                    source_frame: From::NAME,
                    destination_frame: To::NAME,
                    component,
                    value,
                });
            }
        }

        Ok(Self {
            source_origin_x_in_destination_m,
            source_origin_y_in_destination_m,
            source_yaw_in_destination_rad: normalize_yaw(source_yaw_in_destination_rad),
            frames: PhantomData,
        })
    }

    pub fn source_origin_x_in_destination_m(self) -> f64 {
        self.source_origin_x_in_destination_m
    }

    pub fn source_origin_y_in_destination_m(self) -> f64 {
        self.source_origin_y_in_destination_m
    }

    pub fn source_origin_in_destination_m(self) -> [f64; 2] {
        [
            self.source_origin_x_in_destination_m,
            self.source_origin_y_in_destination_m,
        ]
    }

    pub fn source_yaw_in_destination_rad(self) -> f64 {
        self.source_yaw_in_destination_rad
    }

    pub fn source_frame_name() -> &'static str {
        From::NAME
    }

    pub fn destination_frame_name() -> &'static str {
        To::NAME
    }

    /// Transform one point from `From` coordinates into `To` coordinates.
    pub fn transform_point(
        self,
        point: PlanarPoint<From>,
    ) -> Result<PlanarPoint<To>, PlanarTransformError> {
        if self.source_origin_x_in_destination_m == 0.0
            && self.source_origin_y_in_destination_m == 0.0
            && self.source_yaw_in_destination_rad == 0.0
        {
            return Ok(PlanarPoint {
                x_m: point.x_m,
                y_m: point.y_m,
                frame: PhantomData,
            });
        }
        let (sin_yaw, cos_yaw) = self.source_yaw_in_destination_rad.sin_cos();
        let x_m = cos_yaw.mul_add(
            point.x_m,
            (-sin_yaw).mul_add(point.y_m, self.source_origin_x_in_destination_m),
        );
        let y_m = sin_yaw.mul_add(
            point.x_m,
            cos_yaw.mul_add(point.y_m, self.source_origin_y_in_destination_m),
        );
        checked_output::<From, To>(
            PlanarTransformOperation::TransformPoint,
            PlanarTransformComponent::PointX,
            x_m,
        )?;
        checked_output::<From, To>(
            PlanarTransformOperation::TransformPoint,
            PlanarTransformComponent::PointY,
            y_m,
        )?;
        Ok(PlanarPoint {
            x_m,
            y_m,
            frame: PhantomData,
        })
    }

    /// Compute `self ∘ before`.
    ///
    /// `before` maps `Start -> From`; `self` maps `From -> To`; the returned
    /// transform therefore maps `Start -> To`.
    pub fn compose<Start: PlanarFrame>(
        self,
        before: PlanarTransform<Start, From>,
    ) -> Result<PlanarTransform<Start, To>, PlanarTransformError> {
        let (sin_yaw, cos_yaw) = self.source_yaw_in_destination_rad.sin_cos();
        let x_m = cos_yaw.mul_add(
            before.source_origin_x_in_destination_m,
            (-sin_yaw).mul_add(
                before.source_origin_y_in_destination_m,
                self.source_origin_x_in_destination_m,
            ),
        );
        let y_m = sin_yaw.mul_add(
            before.source_origin_x_in_destination_m,
            cos_yaw.mul_add(
                before.source_origin_y_in_destination_m,
                self.source_origin_y_in_destination_m,
            ),
        );
        checked_output::<Start, To>(
            PlanarTransformOperation::Compose,
            PlanarTransformComponent::SourceOriginXInDestination,
            x_m,
        )?;
        checked_output::<Start, To>(
            PlanarTransformOperation::Compose,
            PlanarTransformComponent::SourceOriginYInDestination,
            y_m,
        )?;

        let yaw_rad = normalize_yaw(
            self.source_yaw_in_destination_rad + before.source_yaw_in_destination_rad,
        );
        checked_output::<Start, To>(
            PlanarTransformOperation::Compose,
            PlanarTransformComponent::SourceYawInDestination,
            yaw_rad,
        )?;
        Ok(PlanarTransform {
            source_origin_x_in_destination_m: x_m,
            source_origin_y_in_destination_m: y_m,
            source_yaw_in_destination_rad: yaw_rad,
            frames: PhantomData,
        })
    }

    /// Apply `self`, then `after`.
    pub fn then<Destination: PlanarFrame>(
        self,
        after: PlanarTransform<To, Destination>,
    ) -> Result<PlanarTransform<From, Destination>, PlanarTransformError> {
        after.compose(self)
    }

    pub fn inverse(self) -> Result<PlanarTransform<To, From>, PlanarTransformError> {
        let (sin_yaw, cos_yaw) = self.source_yaw_in_destination_rad.sin_cos();
        let x_m = (-cos_yaw).mul_add(
            self.source_origin_x_in_destination_m,
            -sin_yaw * self.source_origin_y_in_destination_m,
        );
        let y_m = sin_yaw.mul_add(
            self.source_origin_x_in_destination_m,
            -cos_yaw * self.source_origin_y_in_destination_m,
        );
        checked_output::<To, From>(
            PlanarTransformOperation::Inverse,
            PlanarTransformComponent::SourceOriginXInDestination,
            x_m,
        )?;
        checked_output::<To, From>(
            PlanarTransformOperation::Inverse,
            PlanarTransformComponent::SourceOriginYInDestination,
            y_m,
        )?;
        let yaw_rad = normalize_yaw(-self.source_yaw_in_destination_rad);
        checked_output::<To, From>(
            PlanarTransformOperation::Inverse,
            PlanarTransformComponent::SourceYawInDestination,
            yaw_rad,
        )?;
        Ok(PlanarTransform {
            source_origin_x_in_destination_m: x_m,
            source_origin_y_in_destination_m: y_m,
            source_yaw_in_destination_rad: yaw_rad,
            frames: PhantomData,
        })
    }
}

impl<Frame: PlanarFrame> PlanarTransform<Frame, Frame> {
    pub fn identity() -> Self {
        Self {
            source_origin_x_in_destination_m: 0.0,
            source_origin_y_in_destination_m: 0.0,
            source_yaw_in_destination_rad: 0.0,
            frames: PhantomData,
        }
    }
}

pub type OdomToMap = PlanarTransform<OdomFrame, MapFrame>;
pub type MapToOdom = PlanarTransform<MapFrame, OdomFrame>;
pub type BaseToOdom = PlanarTransform<BaseFrame, OdomFrame>;
pub type OdomToBase = PlanarTransform<OdomFrame, BaseFrame>;
pub type LocalCostmapToOdom = PlanarTransform<LocalCostmapFrame, OdomFrame>;
pub type OdomToLocalCostmap = PlanarTransform<OdomFrame, LocalCostmapFrame>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanarAxis {
    X,
    Y,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanarTransformOperation {
    TransformPoint,
    Compose,
    Inverse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanarTransformComponent {
    PointX,
    PointY,
    SourceOriginXInDestination,
    SourceOriginYInDestination,
    SourceYawInDestination,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlanarTransformError {
    NonFiniteInput {
        source_frame: &'static str,
        destination_frame: &'static str,
        component: PlanarTransformComponent,
        value: f64,
    },
    NonFiniteOutput {
        source_frame: &'static str,
        destination_frame: &'static str,
        operation: PlanarTransformOperation,
        component: PlanarTransformComponent,
        value: f64,
    },
}

impl std::fmt::Display for PlanarTransformError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteInput {
                source_frame,
                destination_frame,
                component,
                value,
            } => write!(
                f,
                "{source_frame}-to-{destination_frame} planar transform input {component:?} must be finite, got {value}"
            ),
            Self::NonFiniteOutput {
                source_frame,
                destination_frame,
                operation,
                component,
                value,
            } => write!(
                f,
                "{operation:?} produced nonfinite {component:?}={value} for a {source_frame}-to-{destination_frame} planar transform"
            ),
        }
    }
}

impl std::error::Error for PlanarTransformError {}

fn checked_output<From: PlanarFrame, To: PlanarFrame>(
    operation: PlanarTransformOperation,
    component: PlanarTransformComponent,
    value: f64,
) -> Result<(), PlanarTransformError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(PlanarTransformError::NonFiniteOutput {
            source_frame: From::NAME,
            destination_frame: To::NAME,
            operation,
            component,
            value,
        })
    }
}

/// Canonicalize an angle to `[-pi, pi)`. Both `+pi` and `-pi` map to `-pi`,
/// and both signed zero representations map to positive zero.
fn normalize_yaw(yaw_rad: f64) -> f64 {
    debug_assert!(yaw_rad.is_finite());
    let positive = yaw_rad.rem_euclid(std::f64::consts::TAU);
    let normalized = if positive >= std::f64::consts::PI {
        positive - std::f64::consts::TAU
    } else {
        positive
    };
    if normalized == 0.0 { 0.0 } else { normalized }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance,
            "expected {expected:.17e}, got {actual:.17e}, error {error:.17e} > {tolerance:.17e}"
        );
    }

    fn assert_point_close<Frame: PlanarFrame>(
        actual: PlanarPoint<Frame>,
        expected: PlanarPoint<Frame>,
        tolerance: f64,
    ) {
        assert_close(actual.x_m(), expected.x_m(), tolerance);
        assert_close(actual.y_m(), expected.y_m(), tolerance);
    }

    #[test]
    fn points_reject_every_nonfinite_coordinate() {
        for axis in [PlanarAxis::X, PlanarAxis::Y] {
            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let (x, y) = match axis {
                    PlanarAxis::X => (value, 1.0),
                    PlanarAxis::Y => (1.0, value),
                };
                assert!(matches!(
                    PlanarPoint::<MapFrame>::try_new(x, y),
                    Err(PlanarPointError::NonFiniteCoordinate {
                        frame: "map",
                        axis: actual_axis,
                        value: actual_value,
                    }) if actual_axis == axis && actual_value.to_bits() == value.to_bits()
                ));
            }
        }
    }

    #[test]
    fn transform_constructor_rejects_each_nonfinite_parameter() {
        for component in 0..3 {
            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut values = [1.0, 2.0, 0.5];
                values[component] = value;
                assert!(matches!(
                    OdomToMap::try_new(values[0], values[1], values[2]),
                    Err(PlanarTransformError::NonFiniteInput {
                        source_frame: "odom",
                        destination_frame: "map",
                        value: actual_value,
                        ..
                    }) if actual_value.to_bits() == value.to_bits()
                ));
            }
        }
    }

    #[test]
    fn yaw_normalization_is_canonical_at_pi_and_zero_boundaries() {
        let pi = std::f64::consts::PI;
        let below_pi = f64::from_bits(pi.to_bits() - 1);
        let above_pi = f64::from_bits(pi.to_bits() + 1);

        for input in [pi, -pi, 3.0 * pi, -3.0 * pi] {
            assert_eq!(
                OdomToMap::try_new(0.0, 0.0, input)
                    .expect("finite yaw")
                    .source_yaw_in_destination_rad(),
                -pi
            );
        }
        assert_eq!(
            OdomToMap::try_new(0.0, 0.0, below_pi)
                .expect("finite yaw")
                .source_yaw_in_destination_rad(),
            below_pi
        );
        let normalized_above_pi = OdomToMap::try_new(0.0, 0.0, above_pi)
            .expect("finite yaw")
            .source_yaw_in_destination_rad();
        assert!(normalized_above_pi > -pi);
        assert!(normalized_above_pi < 0.0);
        for input in [0.0, -0.0, std::f64::consts::TAU, -std::f64::consts::TAU] {
            assert_eq!(
                OdomToMap::try_new(0.0, 0.0, input)
                    .expect("finite yaw")
                    .source_yaw_in_destination_rad()
                    .to_bits(),
                0.0_f64.to_bits()
            );
        }
    }

    #[test]
    fn yaw_normalization_is_bounded_and_idempotent_for_extreme_finite_values() {
        for yaw in [
            f64::MIN,
            -1.0e300,
            -10.0,
            -std::f64::consts::PI,
            0.0,
            std::f64::consts::PI,
            10.0,
            1.0e300,
            f64::MAX,
        ] {
            let normalized = OdomToMap::try_new(0.0, 0.0, yaw)
                .expect("finite yaw")
                .source_yaw_in_destination_rad();
            assert!(normalized >= -std::f64::consts::PI);
            assert!(normalized < std::f64::consts::PI);
            assert_eq!(normalize_yaw(normalized).to_bits(), normalized.to_bits());
        }
    }

    #[test]
    fn identity_preserves_finite_points_exactly() {
        let identity = PlanarTransform::<OdomFrame, OdomFrame>::identity();
        for [x, y] in [
            [0.0, -0.0],
            [1.0, -2.0],
            [f64::MIN, f64::MAX],
            [f64::MIN_POSITIVE, -f64::MIN_POSITIVE],
        ] {
            let point = PlanarPoint::try_new(x, y).expect("finite point");
            let transformed = identity
                .transform_point(point)
                .expect("identity remains finite");
            assert_eq!(transformed.x_m().to_bits(), x.to_bits());
            assert_eq!(transformed.y_m().to_bits(), y.to_bits());
        }
    }

    #[test]
    fn inverse_round_trip_holds_across_representative_transforms_and_points() {
        let yaws = [
            -std::f64::consts::PI,
            -2.2,
            -0.5,
            0.0,
            0.5,
            2.2,
            std::f64::consts::PI,
        ];
        for yaw in yaws {
            for [tx, ty] in [[0.0, 0.0], [1.5, -3.0], [-1000.0, 2000.0]] {
                let forward = OdomToMap::try_new(tx, ty, yaw).expect("finite transform");
                let inverse: MapToOdom = forward.inverse().expect("finite inverse");
                for [x, y] in [[0.0, 0.0], [1.0, -2.0], [-400.0, 800.0]] {
                    let point = PlanarPoint::<OdomFrame>::try_new(x, y).expect("finite point");
                    let round_trip = inverse
                        .transform_point(
                            forward
                                .transform_point(point)
                                .expect("finite forward point"),
                        )
                        .expect("finite inverse point");
                    let scale = 1.0_f64
                        .max(x.abs())
                        .max(y.abs())
                        .max(tx.abs())
                        .max(ty.abs());
                    assert_point_close(round_trip, point, 32.0 * f64::EPSILON * scale);
                }
            }
        }
    }

    #[test]
    fn composition_matches_sequential_application() {
        let base_to_odom = BaseToOdom::try_new(1.0, -2.0, 0.4).expect("finite base-to-odom");
        let odom_to_map = OdomToMap::try_new(-5.0, 8.0, -1.2).expect("finite odom-to-map");
        let base_to_map = base_to_odom.then(odom_to_map).expect("finite composition");
        let point = PlanarPoint::<BaseFrame>::try_new(3.0, 4.0).expect("finite base point");

        let sequential = odom_to_map
            .transform_point(
                base_to_odom
                    .transform_point(point)
                    .expect("finite odom point"),
            )
            .expect("finite map point");
        let composed = base_to_map
            .transform_point(point)
            .expect("finite composed point");
        assert_point_close(composed, sequential, 16.0 * f64::EPSILON);
    }

    #[test]
    fn transform_inverse_composes_to_identity_action() {
        for yaw in [-2.0, -0.1, 0.0, 0.1, 2.0] {
            let transform = OdomToMap::try_new(3.0, -4.0, yaw).expect("finite transform");
            let identity = transform
                .inverse()
                .expect("finite inverse")
                .compose(transform)
                .expect("finite identity composition");
            let point = PlanarPoint::<OdomFrame>::try_new(9.0, -7.0).expect("finite point");
            assert_point_close(
                identity
                    .transform_point(point)
                    .expect("finite identity action"),
                point,
                32.0 * f64::EPSILON,
            );
        }
    }

    #[test]
    fn finite_inputs_that_overflow_each_operation_return_typed_errors() {
        let point = PlanarPoint::<OdomFrame>::try_new(f64::MAX, 0.0).expect("finite point");
        let transform = OdomToMap::try_new(f64::MAX, 0.0, 0.0).expect("finite transform");
        assert!(matches!(
            transform.transform_point(point),
            Err(PlanarTransformError::NonFiniteOutput {
                operation: PlanarTransformOperation::TransformPoint,
                component: PlanarTransformComponent::PointX,
                ..
            })
        ));

        let base_to_odom = BaseToOdom::try_new(f64::MAX, 0.0, 0.0).expect("finite first transform");
        let odom_to_map = OdomToMap::try_new(f64::MAX, 0.0, 0.0).expect("finite second transform");
        assert!(matches!(
            base_to_odom.then(odom_to_map),
            Err(PlanarTransformError::NonFiniteOutput {
                operation: PlanarTransformOperation::Compose,
                component: PlanarTransformComponent::SourceOriginXInDestination,
                ..
            })
        ));

        let inverse_overflow = OdomToMap::try_new(f64::MAX, f64::MAX, std::f64::consts::FRAC_PI_4)
            .expect("finite transform");
        assert!(matches!(
            inverse_overflow.inverse(),
            Err(PlanarTransformError::NonFiniteOutput {
                operation: PlanarTransformOperation::Inverse,
                component: PlanarTransformComponent::SourceOriginXInDestination,
                ..
            })
        ));
    }
}
