use core::fmt;
use core::num::NonZeroU16;

use crate::{
    CommandFrame, FullTelemetry, GoalSpeedTicksPerSecond, PositionAgreementTicks, PositionTicks,
    ServoId, TorqueLimitPermille, TorqueSwitch, ValidatedPresentPosition,
    build_full_telemetry_read, build_goal_with_speed_write, build_torque_limit_write,
    build_torque_switch_write,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum HeadJoint {
    Bow = 0,
    Curl = 1,
    Yaw = 2,
    Roll = 3,
}

impl HeadJoint {
    pub const ALL: [Self; 4] = [Self::Bow, Self::Curl, Self::Yaw, Self::Roll];

    /// Exact physical ID assignment qualified by the source demo rig.
    pub const fn servo_id(self) -> ServoId {
        match self {
            Self::Bow => ServoId::known(1),
            Self::Curl => ServoId::known(2),
            Self::Yaw => ServoId::known(3),
            Self::Roll => ServoId::known(4),
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

/// Exact four-joint pose in canonical bow/curl/yaw/roll order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadPose {
    positions: [PositionTicks; 4],
}

impl HeadPose {
    /// Admit a pose only from four redundant reads carrying the exact expected
    /// servo IDs in canonical order.
    pub fn try_from_validated(
        positions: [ValidatedPresentPosition; 4],
    ) -> Result<Self, HeadPoseError> {
        let mut admitted = [PositionTicks::MIN; 4];
        for (index, (joint, position)) in HeadJoint::ALL.into_iter().zip(positions).enumerate() {
            let expected = joint.servo_id();
            if position.id() != expected {
                return Err(HeadPoseError::ServoIdMismatch {
                    joint,
                    expected,
                    actual: position.id(),
                });
            }
            admitted[index] = position.admitted();
        }
        Ok(Self {
            positions: admitted,
        })
    }

    pub const fn position(self, joint: HeadJoint) -> PositionTicks {
        self.positions[joint.index()]
    }

    pub const fn positions(self) -> [PositionTicks; 4] {
        self.positions
    }

    /// Admit the freshest positions from two complete telemetry sets only when
    /// both sets carry the exact canonical IDs and each joint agrees inside the
    /// already parsed tolerance. Runtime-specific status, stopped-state, and
    /// timestamp admission occurs before this protocol-level pose constructor.
    pub fn try_from_telemetry_pair(
        first: [FullTelemetry; 4],
        second: [FullTelemetry; 4],
        tolerance: PositionAgreementTicks,
    ) -> Result<Self, HeadPoseError> {
        let mut positions = [PositionTicks::MIN; 4];
        for (index, joint) in HeadJoint::ALL.into_iter().enumerate() {
            let expected = joint.servo_id();
            for (sample, actual) in [
                (TelemetryPoseSample::First, first[index].id()),
                (TelemetryPoseSample::Second, second[index].id()),
            ] {
                if actual != expected {
                    return Err(HeadPoseError::TelemetryServoIdMismatch {
                        joint,
                        sample,
                        expected,
                        actual,
                    });
                }
            }
            let difference = first[index]
                .position()
                .get()
                .abs_diff(second[index].position().get());
            if difference > tolerance.get() {
                return Err(HeadPoseError::TelemetrySamplesDisagree {
                    joint,
                    first: first[index].position(),
                    second: second[index].position(),
                    difference_ticks: difference,
                    tolerance,
                });
            }
            positions[index] = second[index].position();
        }
        Ok(Self { positions })
    }
}

/// Exact reviewed command target in canonical bow/curl/yaw/roll order.
///
/// This is deliberately distinct from [`HeadPose`]: an observed pose can only
/// be constructed from redundant servo reads, while a command target enters at
/// a configuration boundary and must retain that different provenance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactHeadTargetPose {
    positions: [PositionTicks; 4],
}

impl ExactHeadTargetPose {
    /// Construct an exact target from positions that have already crossed the
    /// protocol tick boundary. Naming every joint avoids an implicit array
    /// order while preserving the `0..=4095` invariant of [`PositionTicks`].
    pub const fn from_positions(
        bow: PositionTicks,
        curl: PositionTicks,
        yaw: PositionTicks,
        roll: PositionTicks,
    ) -> Self {
        Self {
            positions: [bow, curl, yaw, roll],
        }
    }

    pub fn try_from_ticks(ticks: [u16; 4]) -> Result<Self, ExactHeadTargetPoseError> {
        let mut positions = [PositionTicks::MIN; 4];
        for (index, joint) in HeadJoint::ALL.into_iter().enumerate() {
            positions[index] = PositionTicks::try_new(ticks[index]).map_err(|source| {
                ExactHeadTargetPoseError::Position {
                    joint,
                    value: ticks[index],
                    source,
                }
            })?;
        }
        Ok(Self { positions })
    }

    pub const fn position(self, joint: HeadJoint) -> PositionTicks {
        self.positions[joint.index()]
    }

    pub const fn positions(self) -> [PositionTicks; 4] {
        self.positions
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactHeadTargetPoseError {
    Position {
        joint: HeadJoint,
        value: u16,
        source: crate::FrameBuildError,
    },
}

impl fmt::Display for ExactHeadTargetPoseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid exact Kiko head target: {self:?}")
    }
}

impl core::error::Error for ExactHeadTargetPoseError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Position { source, .. } => Some(source),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadPoseError {
    ServoIdMismatch {
        joint: HeadJoint,
        expected: ServoId,
        actual: ServoId,
    },
    TelemetryServoIdMismatch {
        joint: HeadJoint,
        sample: TelemetryPoseSample,
        expected: ServoId,
        actual: ServoId,
    },
    TelemetrySamplesDisagree {
        joint: HeadJoint,
        first: PositionTicks,
        second: PositionTicks,
        difference_ticks: u16,
        tolerance: PositionAgreementTicks,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TelemetryPoseSample {
    First,
    Second,
}

impl fmt::Display for HeadPoseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid exact Kiko head pose: {self:?}")
    }
}

impl core::error::Error for HeadPoseError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HeadTorqueLimits {
    limits: [TorqueLimitPermille; 4],
}

impl HeadTorqueLimits {
    pub const fn new(
        bow: TorqueLimitPermille,
        curl: TorqueLimitPermille,
        yaw: TorqueLimitPermille,
        roll: TorqueLimitPermille,
    ) -> Self {
        Self {
            limits: [bow, curl, yaw, roll],
        }
    }

    pub const fn for_joint(self, joint: HeadJoint) -> TorqueLimitPermille {
        self.limits[joint.index()]
    }
}

/// Complete write/read plan for locking the exact validated present pose.
///
/// The write responses are intentionally absent because the deployed servos'
/// response level is zero. `verification_reads` are mandatory follow-up
/// requests; only their successfully parsed telemetry can establish observed
/// post-write state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NaturalHoldFrames {
    goal_writes: [CommandFrame; 4],
    torque_limit_writes: [CommandFrame; 4],
    torque_enable_writes: [CommandFrame; 4],
    verification_reads: [CommandFrame; 4],
}

impl NaturalHoldFrames {
    pub const fn goal_writes(&self) -> &[CommandFrame; 4] {
        &self.goal_writes
    }

    pub const fn torque_limit_writes(&self) -> &[CommandFrame; 4] {
        &self.torque_limit_writes
    }

    pub const fn torque_enable_writes(&self) -> &[CommandFrame; 4] {
        &self.torque_enable_writes
    }

    pub const fn verification_reads(&self) -> &[CommandFrame; 4] {
        &self.verification_reads
    }
}

pub fn build_natural_hold_frames(
    pose: HeadPose,
    torque_limits: HeadTorqueLimits,
    speed: GoalSpeedTicksPerSecond,
) -> NaturalHoldFrames {
    // Keep construction explicit: array order is part of the head wiring
    // contract and must stay reviewable beside `HeadJoint::servo_id`.
    let goal_writes = HeadJoint::ALL
        .map(|joint| build_goal_with_speed_write(joint.servo_id(), pose.position(joint), speed));
    let torque_limit_writes = HeadJoint::ALL
        .map(|joint| build_torque_limit_write(joint.servo_id(), torque_limits.for_joint(joint)));
    let torque_enable_writes = HeadJoint::ALL
        .map(|joint| build_torque_switch_write(joint.servo_id(), TorqueSwitch::Enabled));
    let verification_reads =
        HeadJoint::ALL.map(|joint| build_full_telemetry_read(joint.servo_id()));
    NaturalHoldFrames {
        goal_writes,
        torque_limit_writes,
        torque_enable_writes,
        verification_reads,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum JointDirection {
    Negative = -1,
    Positive = 1,
}

impl JointDirection {
    const fn multiplier(self) -> f64 {
        self as i8 as f64
    }
}

/// Finite angle in radians. Joint-specific travel admission happens at the
/// calibration boundary because different axes have different limits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AngleRadians(f64);

impl AngleRadians {
    pub fn try_new(value: f64) -> Result<Self, JointCalibrationError> {
        if value.is_finite() {
            Ok(Self(value))
        } else {
            Err(JointCalibrationError::NonFiniteAngle { value })
        }
    }

    pub const fn get(self) -> f64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointLimitsRadians {
    minimum: f64,
    maximum: f64,
}

impl JointLimitsRadians {
    pub fn try_new(minimum: f64, maximum: f64) -> Result<Self, JointCalibrationError> {
        if !minimum.is_finite() {
            return Err(JointCalibrationError::NonFiniteLimit {
                bound: "minimum",
                value: minimum,
            });
        }
        if !maximum.is_finite() {
            return Err(JointCalibrationError::NonFiniteLimit {
                bound: "maximum",
                value: maximum,
            });
        }
        if minimum >= maximum {
            return Err(JointCalibrationError::LimitsNotIncreasing { minimum, maximum });
        }
        if minimum > 0.0 || maximum < 0.0 {
            return Err(JointCalibrationError::LimitsExcludeNaturalZero { minimum, maximum });
        }
        Ok(Self { minimum, maximum })
    }

    pub const fn minimum(self) -> f64 {
        self.minimum
    }

    pub const fn maximum(self) -> f64 {
        self.maximum
    }

    const fn contains(self, angle: f64) -> bool {
        angle >= self.minimum && angle <= self.maximum
    }
}

/// Parsed mapping from one physical encoder to one named joint axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointCalibration {
    joint: HeadJoint,
    zero: PositionTicks,
    ticks_per_radian: f64,
    direction: JointDirection,
    limits: JointLimitsRadians,
}

impl JointCalibration {
    pub fn try_new(
        joint: HeadJoint,
        zero: PositionTicks,
        ticks_per_radian: f64,
        direction: JointDirection,
        limits: JointLimitsRadians,
    ) -> Result<Self, JointCalibrationError> {
        if !ticks_per_radian.is_finite() || ticks_per_radian <= 0.0 {
            return Err(JointCalibrationError::InvalidTicksPerRadian {
                value: ticks_per_radian,
            });
        }
        let calibration = Self {
            joint,
            zero,
            ticks_per_radian,
            direction,
            limits,
        };
        calibration.position_for_value(limits.minimum, "minimum")?;
        calibration.position_for_value(limits.maximum, "maximum")?;
        Ok(calibration)
    }

    pub const fn joint(self) -> HeadJoint {
        self.joint
    }

    pub const fn servo_id(self) -> ServoId {
        self.joint.servo_id()
    }

    pub const fn zero(self) -> PositionTicks {
        self.zero
    }

    pub const fn ticks_per_radian(self) -> f64 {
        self.ticks_per_radian
    }

    pub const fn direction(self) -> JointDirection {
        self.direction
    }

    pub const fn limits(self) -> JointLimitsRadians {
        self.limits
    }

    pub fn angle_for_position(self, position: PositionTicks) -> AngleRadians {
        let tick_delta = f64::from(position.get()) - f64::from(self.zero.get());
        // Parsed calibration guarantees a finite positive denominator.
        AngleRadians((tick_delta / self.ticks_per_radian) * self.direction.multiplier())
    }

    pub fn position_for_angle(
        self,
        angle: AngleRadians,
    ) -> Result<PositionTicks, JointCalibrationError> {
        if !self.limits.contains(angle.get()) {
            return Err(JointCalibrationError::AngleOutsideJointLimits {
                joint: self.joint,
                value: angle.get(),
                minimum: self.limits.minimum,
                maximum: self.limits.maximum,
            });
        }
        self.position_for_value(angle.get(), "target")
    }

    fn position_for_value(
        self,
        angle: f64,
        boundary: &'static str,
    ) -> Result<PositionTicks, JointCalibrationError> {
        let raw = self.direction.multiplier() * angle * self.ticks_per_radian
            + f64::from(self.zero.get());
        if !(0.0..=f64::from(PositionTicks::MAX.get())).contains(&raw) {
            return Err(JointCalibrationError::AngleMapsOutsideEncoder {
                joint: self.joint,
                boundary,
                angle,
                raw_ticks: raw,
            });
        }
        // `raw` is finite and nonnegative here. Adding one half before the
        // integer conversion gives deterministic nearest-tick rounding in a
        // `no_std` build without introducing a floating-point helper crate.
        let ticks = (raw + 0.5) as u16;
        PositionTicks::try_new(ticks).map_err(|_| JointCalibrationError::AngleMapsOutsideEncoder {
            joint: self.joint,
            boundary,
            angle,
            raw_ticks: raw,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JointCalibrationError {
    NonFiniteAngle {
        value: f64,
    },
    NonFiniteLimit {
        bound: &'static str,
        value: f64,
    },
    LimitsNotIncreasing {
        minimum: f64,
        maximum: f64,
    },
    LimitsExcludeNaturalZero {
        minimum: f64,
        maximum: f64,
    },
    InvalidTicksPerRadian {
        value: f64,
    },
    AngleOutsideJointLimits {
        joint: HeadJoint,
        value: f64,
        minimum: f64,
        maximum: f64,
    },
    AngleMapsOutsideEncoder {
        joint: HeadJoint,
        boundary: &'static str,
        angle: f64,
        raw_ticks: f64,
    },
    InvalidPositionStepLimit {
        value: u16,
    },
}

impl fmt::Display for JointCalibrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid Kiko joint calibration or target: {self:?}"
        )
    }
}

impl core::error::Error for JointCalibrationError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PositionStepLimit(NonZeroU16);

impl PositionStepLimit {
    pub const fn try_new(value: u16) -> Result<Self, JointCalibrationError> {
        match NonZeroU16::new(value) {
            Some(value) if value.get() <= PositionTicks::MAX.get() => Ok(Self(value)),
            _ => Err(JointCalibrationError::InvalidPositionStepLimit { value }),
        }
    }

    pub const fn get(self) -> u16 {
        self.0.get()
    }

    /// Move toward `target` without crossing it or wrapping the encoder.
    pub fn advance(self, current: PositionTicks, target: PositionTicks) -> PositionTicks {
        let current = current.get();
        let target = target.get();
        let next = if current < target {
            current.saturating_add(self.get()).min(target)
        } else {
            current.saturating_sub(self.get()).max(target)
        };
        PositionTicks::try_new(next).expect("bounded step remains in the encoder domain")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PositionAgreementTicks, PresentPosition};

    fn response(id: ServoId, position: u16) -> [u8; 8] {
        let position = position.to_le_bytes();
        let mut bytes = [0xff, 0xff, id.get(), 4, 0, position[0], position[1], 0];
        bytes[7] = !bytes[2..7]
            .iter()
            .fold(0_u8, |sum, byte| sum.wrapping_add(*byte));
        bytes
    }

    fn qualified(joint: HeadJoint, first: u16, second: u16) -> ValidatedPresentPosition {
        let id = joint.servo_id();
        let first = PresentPosition::parse(&response(id, first), id).expect("first");
        let second = PresentPosition::parse(&response(id, second), id).expect("second");
        ValidatedPresentPosition::try_from_pair(
            first,
            second,
            PositionAgreementTicks::DEMO_QUALIFIED_MAXIMUM,
        )
        .expect("qualified")
    }

    fn full_telemetry(joint: HeadJoint, position: u16) -> FullTelemetry {
        let mut bytes = [0_u8; 21];
        bytes[..5].copy_from_slice(&[0xff, 0xff, joint.servo_id().get(), 17, 0]);
        bytes[5..7].copy_from_slice(&position.to_le_bytes());
        bytes[20] = !bytes[2..20]
            .iter()
            .fold(0_u8, |sum, byte| sum.wrapping_add(*byte));
        FullTelemetry::parse(&bytes, joint.servo_id()).expect("full telemetry")
    }

    #[test]
    fn canonical_pose_rejects_reordered_or_duplicate_ids() {
        let valid = [
            qualified(HeadJoint::Bow, 2_120, 2_127),
            qualified(HeadJoint::Curl, 2_550, 2_558),
            qualified(HeadJoint::Yaw, 2_920, 2_925),
            qualified(HeadJoint::Roll, 2_925, 2_930),
        ];
        let pose = HeadPose::try_from_validated(valid).expect("canonical pose");
        assert_eq!(pose.position(HeadJoint::Yaw).get(), 2_925);

        let reordered = [valid[1], valid[0], valid[2], valid[3]];
        assert!(matches!(
            HeadPose::try_from_validated(reordered),
            Err(HeadPoseError::ServoIdMismatch {
                joint: HeadJoint::Bow,
                ..
            })
        ));
    }

    #[test]
    fn telemetry_pose_requires_two_canonical_agreeing_sets_and_uses_the_freshest() {
        let first: [FullTelemetry; 4] = core::array::from_fn(|index| {
            full_telemetry(HeadJoint::ALL[index], [2_120, 2_550, 2_920, 2_925][index])
        });
        let second: [FullTelemetry; 4] = core::array::from_fn(|index| {
            full_telemetry(HeadJoint::ALL[index], [2_127, 2_558, 2_925, 2_930][index])
        });
        let tolerance = PositionAgreementTicks::try_new(10).expect("tolerance");
        let pose = HeadPose::try_from_telemetry_pair(first, second, tolerance)
            .expect("canonical agreeing telemetry");
        assert_eq!(
            pose.positions().map(PositionTicks::get),
            [2_127, 2_558, 2_925, 2_930]
        );

        let mut reordered = first;
        reordered.swap(0, 1);
        assert!(matches!(
            HeadPose::try_from_telemetry_pair(reordered, second, tolerance),
            Err(HeadPoseError::TelemetryServoIdMismatch {
                joint: HeadJoint::Bow,
                sample: TelemetryPoseSample::First,
                ..
            })
        ));

        let disagreeing: [FullTelemetry; 4] = core::array::from_fn(|index| {
            full_telemetry(HeadJoint::ALL[index], [2_131, 2_558, 2_925, 2_930][index])
        });
        assert!(matches!(
            HeadPose::try_from_telemetry_pair(first, disagreeing, tolerance),
            Err(HeadPoseError::TelemetrySamplesDisagree {
                joint: HeadJoint::Bow,
                difference_ticks: 11,
                ..
            })
        ));
    }

    #[test]
    fn natural_hold_only_targets_observed_positions_and_requires_readback() {
        let pose = HeadPose::try_from_validated([
            qualified(HeadJoint::Bow, 2_120, 2_127),
            qualified(HeadJoint::Curl, 2_550, 2_558),
            qualified(HeadJoint::Yaw, 2_920, 2_925),
            qualified(HeadJoint::Roll, 2_925, 2_930),
        ])
        .expect("pose");
        let limits = HeadTorqueLimits::new(
            TorqueLimitPermille::try_new(600).expect("bow limit"),
            TorqueLimitPermille::try_new(400).expect("curl limit"),
            TorqueLimitPermille::try_new(400).expect("yaw limit"),
            TorqueLimitPermille::try_new(400).expect("roll limit"),
        );
        let plan = build_natural_hold_frames(
            pose,
            limits,
            GoalSpeedTicksPerSecond::try_new(100).expect("speed"),
        );
        for (index, joint) in HeadJoint::ALL.into_iter().enumerate() {
            assert_eq!(
                plan.goal_writes()[index].as_bytes()[2],
                joint.servo_id().get()
            );
            assert_eq!(plan.goal_writes()[index].as_bytes()[5], 42);
            assert_eq!(plan.torque_limit_writes()[index].as_bytes()[5], 48);
            assert_eq!(
                plan.torque_enable_writes()[index].as_bytes()[5..=6],
                [40, 1]
            );
            assert_eq!(
                plan.verification_reads()[index].as_bytes()[4..=6],
                [2, 56, 15]
            );
        }
    }

    #[test]
    fn calibration_checks_units_limits_direction_and_encoder_bounds() {
        let limits = JointLimitsRadians::try_new(-0.5, 0.5).expect("limits");
        let calibration = JointCalibration::try_new(
            HeadJoint::Bow,
            PositionTicks::try_new(2_127).expect("zero"),
            4096.0 / core::f64::consts::TAU,
            JointDirection::Negative,
            limits,
        )
        .expect("calibration");
        let positive = AngleRadians::try_new(0.25).expect("angle");
        let goal = calibration.position_for_angle(positive).expect("goal");
        assert!(goal.get() < calibration.zero().get());
        let roundtrip = calibration.angle_for_position(goal).get();
        assert!((roundtrip - 0.25).abs() <= 0.5 / calibration.ticks_per_radian());

        assert!(AngleRadians::try_new(f64::NAN).is_err());
        assert!(JointLimitsRadians::try_new(1.0, 2.0).is_err());
        assert!(
            JointCalibration::try_new(
                HeadJoint::Yaw,
                PositionTicks::try_new(4_000).expect("zero"),
                1_000.0,
                JointDirection::Positive,
                limits,
            )
            .is_err()
        );
        assert!(matches!(
            calibration.position_for_angle(AngleRadians::try_new(0.75).expect("finite")),
            Err(JointCalibrationError::AngleOutsideJointLimits { .. })
        ));
    }

    #[test]
    fn bounded_step_never_overshoots_or_wraps() {
        let step = PositionStepLimit::try_new(50).expect("step");
        let low = PositionTicks::try_new(10).expect("low");
        let high = PositionTicks::try_new(4_090).expect("high");
        assert_eq!(step.advance(low, high).get(), 60);
        assert_eq!(step.advance(high, low).get(), 4_040);
        assert_eq!(
            step.advance(
                PositionTicks::try_new(100).expect("current"),
                PositionTicks::try_new(120).expect("target")
            )
            .get(),
            120
        );
    }

    #[test]
    fn exact_target_parses_canonical_ticks_once() {
        let target = ExactHeadTargetPose::try_from_ticks([2_174, 2_570, 1_637, 3_047])
            .expect("reviewed target");
        assert_eq!(
            target.positions().map(PositionTicks::get),
            [2_174, 2_570, 1_637, 3_047]
        );

        assert!(matches!(
            ExactHeadTargetPose::try_from_ticks([2_174, 4_096, 1_637, 3_047]),
            Err(ExactHeadTargetPoseError::Position {
                joint: HeadJoint::Curl,
                value: 4_096,
                ..
            })
        ));
    }

    #[test]
    fn property_style_calibration_roundtrips_with_half_tick_error_bound() {
        for direction in [JointDirection::Negative, JointDirection::Positive] {
            let calibration = JointCalibration::try_new(
                HeadJoint::Curl,
                PositionTicks::try_new(2_048).expect("zero"),
                4096.0 / core::f64::consts::TAU,
                direction,
                JointLimitsRadians::try_new(-1.0, 1.0).expect("limits"),
            )
            .expect("calibration");
            for step in -1_000_i32..=1_000 {
                let requested = f64::from(step) / 1_000.0;
                let angle = AngleRadians::try_new(requested).expect("finite angle");
                let position = calibration.position_for_angle(angle).expect("in limits");
                let reconstructed = calibration.angle_for_position(position).get();
                assert!(
                    (requested - reconstructed).abs()
                        <= 0.5 / calibration.ticks_per_radian() + f64::EPSILON,
                    "direction={direction:?} requested={requested} reconstructed={reconstructed}"
                );
            }
        }
    }
}
