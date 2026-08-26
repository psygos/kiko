//! Deterministic jerk-bounded target shaping for the four-axis head.
//!
//! Policies enter in physical SI-rate units and are bound once to the exact
//! control period. Runtime integration uses signed fixed-point microticks; it
//! has no floating-point state, allocation, wall-clock lookup, or catch-up
//! path. A caller services exactly one declared control interval per step.

use core::{fmt, num::NonZeroU16, time::Duration};

use kiko_head_protocol::{ExactHeadTargetPose, HeadJoint, PositionTicks};

use crate::gaze_control::{HeadControlPeriod, HeadMotionLimits};

const JOINT_COUNT: usize = 4;
const MICRO: i64 = 1_000_000;
const TAU_MICRO: u128 = 6_283_185;
const NANOS_PER_SECOND: u128 = 1_000_000_000;
const ATTACK_TRIGGER_PERCENT: u64 = 5;
const ATTACK_RESPONSE_GAIN_MICRO: i64 = 1_500_000;
const ATTACK_DAMPING_CUT_MICRO: i64 = 250_000;
const ATTACK_ACCELERATION_GAIN_MICRO: i64 = 1_000_000;
const ATTACK_JERK_GAIN_MICRO: i64 = 1_500_000;
const ATTACK_DECAY_TIME: Duration = Duration::from_millis(350);
const POSITION_SETTLE_MICROTICKS: i64 = 10_000;

const fn joint_index(joint: HeadJoint) -> usize {
    match joint {
        HeadJoint::Bow => 0,
        HeadJoint::Curl => 1,
        HeadJoint::Yaw => 2,
        HeadJoint::Roll => 3,
    }
}

/// Weak boundary values for one organic motion axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrganicJointMotionPolicyInput {
    pub response_millihertz: u32,
    pub damping_permille: u32,
    pub maximum_velocity_ticks_per_second: u32,
    pub maximum_acceleration_ticks_per_second_squared: u32,
    pub maximum_jerk_ticks_per_second_cubed: u32,
}

/// Which boundary field failed to become an organic axis policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrganicJointMotionPolicyField {
    ResponseMilliHertz,
    DampingPermille,
    MaximumVelocityTicksPerSecond,
    MaximumAccelerationTicksPerSecondSquared,
    MaximumJerkTicksPerSecondCubed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrganicJointMotionPolicyError {
    Zero {
        field: OrganicJointMotionPolicyField,
    },
    ResponseAboveMaximum {
        actual_millihertz: u32,
        maximum_millihertz: u32,
    },
    DampingOutsideReviewedRange {
        actual_permille: u32,
        minimum_permille: u32,
        maximum_permille: u32,
    },
    AboveRepresentableMaximum {
        field: OrganicJointMotionPolicyField,
        actual: u32,
        maximum: u32,
    },
}

impl fmt::Display for OrganicJointMotionPolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid organic joint motion policy: {self:?}")
    }
}

impl std::error::Error for OrganicJointMotionPolicyError {}

/// Parsed SI-rate policy for one head joint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrganicJointMotionPolicy {
    response_millihertz: NonZeroU16,
    damping_permille: NonZeroU16,
    maximum_velocity_ticks_per_second: NonZeroU16,
    maximum_acceleration_ticks_per_second_squared: NonZeroU16,
    maximum_jerk_ticks_per_second_cubed: NonZeroU16,
}

impl OrganicJointMotionPolicy {
    pub const MAXIMUM_RESPONSE_MILLIHERTZ: u32 = 10_000;
    pub const MINIMUM_DAMPING_PERMILLE: u32 = 500;
    pub const MAXIMUM_DAMPING_PERMILLE: u32 = 2_000;

    pub fn parse(
        input: OrganicJointMotionPolicyInput,
    ) -> Result<Self, OrganicJointMotionPolicyError> {
        let positive = |value, field| {
            let value = u16::try_from(value).map_err(|_| {
                OrganicJointMotionPolicyError::AboveRepresentableMaximum {
                    field,
                    actual: value,
                    maximum: u32::from(u16::MAX),
                }
            })?;
            NonZeroU16::new(value).ok_or(OrganicJointMotionPolicyError::Zero { field })
        };
        let response_millihertz = positive(
            input.response_millihertz,
            OrganicJointMotionPolicyField::ResponseMilliHertz,
        )?;
        if u32::from(response_millihertz.get()) > Self::MAXIMUM_RESPONSE_MILLIHERTZ {
            return Err(OrganicJointMotionPolicyError::ResponseAboveMaximum {
                actual_millihertz: u32::from(response_millihertz.get()),
                maximum_millihertz: Self::MAXIMUM_RESPONSE_MILLIHERTZ,
            });
        }
        let damping_permille = positive(
            input.damping_permille,
            OrganicJointMotionPolicyField::DampingPermille,
        )?;
        if !(Self::MINIMUM_DAMPING_PERMILLE..=Self::MAXIMUM_DAMPING_PERMILLE)
            .contains(&u32::from(damping_permille.get()))
        {
            return Err(OrganicJointMotionPolicyError::DampingOutsideReviewedRange {
                actual_permille: u32::from(damping_permille.get()),
                minimum_permille: Self::MINIMUM_DAMPING_PERMILLE,
                maximum_permille: Self::MAXIMUM_DAMPING_PERMILLE,
            });
        }
        Ok(Self {
            response_millihertz,
            damping_permille,
            maximum_velocity_ticks_per_second: positive(
                input.maximum_velocity_ticks_per_second,
                OrganicJointMotionPolicyField::MaximumVelocityTicksPerSecond,
            )?,
            maximum_acceleration_ticks_per_second_squared: positive(
                input.maximum_acceleration_ticks_per_second_squared,
                OrganicJointMotionPolicyField::MaximumAccelerationTicksPerSecondSquared,
            )?,
            maximum_jerk_ticks_per_second_cubed: positive(
                input.maximum_jerk_ticks_per_second_cubed,
                OrganicJointMotionPolicyField::MaximumJerkTicksPerSecondCubed,
            )?,
        })
    }

    pub const fn response_millihertz(self) -> u32 {
        self.response_millihertz.get() as u32
    }

    pub const fn damping_permille(self) -> u32 {
        self.damping_permille.get() as u32
    }

    pub const fn maximum_velocity_ticks_per_second(self) -> u32 {
        self.maximum_velocity_ticks_per_second.get() as u32
    }

    pub const fn maximum_acceleration_ticks_per_second_squared(self) -> u32 {
        self.maximum_acceleration_ticks_per_second_squared.get() as u32
    }

    pub const fn maximum_jerk_ticks_per_second_cubed(self) -> u32 {
        self.maximum_jerk_ticks_per_second_cubed.get() as u32
    }
}

/// Named organic motion policy for all four physical head joints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrganicHeadMotionPolicy {
    joints: [OrganicJointMotionPolicy; JOINT_COUNT],
}

impl OrganicHeadMotionPolicy {
    pub const fn new(
        bow: OrganicJointMotionPolicy,
        curl: OrganicJointMotionPolicy,
        yaw: OrganicJointMotionPolicy,
        roll: OrganicJointMotionPolicy,
    ) -> Self {
        Self {
            joints: [bow, curl, yaw, roll],
        }
    }

    pub const fn joint(self, joint: HeadJoint) -> OrganicJointMotionPolicy {
        self.joints[joint_index(joint)]
    }

    /// Prove that this SI-rate policy can be safely composed with one exact
    /// fixed-period planner declaration without exposing internal coefficients.
    pub fn admit_for_control(
        self,
        period: HeadControlPeriod,
        motion_limits: HeadMotionLimits,
    ) -> Result<(), OrganicHeadMotionBindingError> {
        self.bind(period, motion_limits).map(|_| ())
    }

    pub(crate) fn bind(
        self,
        period: HeadControlPeriod,
        motion_limits: HeadMotionLimits,
    ) -> Result<BoundOrganicHeadMotionPolicy, OrganicHeadMotionBindingError> {
        let joints = [
            bind_joint(
                HeadJoint::Bow,
                self.joint(HeadJoint::Bow),
                period,
                motion_limits,
            )?,
            bind_joint(
                HeadJoint::Curl,
                self.joint(HeadJoint::Curl),
                period,
                motion_limits,
            )?,
            bind_joint(
                HeadJoint::Yaw,
                self.joint(HeadJoint::Yaw),
                period,
                motion_limits,
            )?,
            bind_joint(
                HeadJoint::Roll,
                self.joint(HeadJoint::Roll),
                period,
                motion_limits,
            )?,
        ];
        Ok(BoundOrganicHeadMotionPolicy { joints })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrganicDerivedQuantity {
    AngularResponse,
    MaximumVelocity,
    MaximumAcceleration,
    MaximumJerk,
    SettleVelocity,
    SettleAcceleration,
    AttackDecay,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrganicHeadMotionBindingError {
    ArithmeticOverflow {
        joint: HeadJoint,
        quantity: OrganicDerivedQuantity,
    },
    BelowOneMicroUnit {
        joint: HeadJoint,
        quantity: OrganicDerivedQuantity,
    },
    VelocityExceedsPlanner {
        joint: HeadJoint,
        organic_microticks_per_tick: u64,
        planner_microticks_per_tick: u64,
    },
    AccelerationExceedsPlanner {
        joint: HeadJoint,
        organic_microticks_per_tick_squared: u64,
        planner_microticks_per_tick_squared: u64,
    },
}

impl fmt::Display for OrganicHeadMotionBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "organic head motion binding failed: {self:?}")
    }
}

impl std::error::Error for OrganicHeadMotionBindingError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BoundOrganicHeadMotionPolicy {
    joints: [BoundOrganicJointMotionPolicy; JOINT_COUNT],
}

impl BoundOrganicHeadMotionPolicy {
    const fn joint(self, joint: HeadJoint) -> BoundOrganicJointMotionPolicy {
        self.joints[joint_index(joint)]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BoundOrganicJointMotionPolicy {
    minimum_microticks: i64,
    maximum_microticks: i64,
    omega_dt_micro: i64,
    damping_micro: i64,
    maximum_velocity_microticks_per_tick: i64,
    maximum_acceleration_microticks_per_tick_squared: i64,
    maximum_jerk_microticks_per_tick_cubed: i64,
    attack_trigger_ticks: u16,
    attack_decay_micro: i64,
    settle_velocity_microticks_per_tick: i64,
    settle_acceleration_microticks_per_tick_squared: i64,
}

fn bind_joint(
    joint: HeadJoint,
    declared: OrganicJointMotionPolicy,
    period: HeadControlPeriod,
    motion_limits: HeadMotionLimits,
) -> Result<BoundOrganicJointMotionPolicy, OrganicHeadMotionBindingError> {
    let limits = motion_limits.joint(joint);
    let period_ns = period.get().as_nanos();
    let omega_dt_micro = derived_positive(
        joint,
        OrganicDerivedQuantity::AngularResponse,
        u128::from(declared.response_millihertz())
            .checked_mul(period_ns)
            .and_then(|value| value.checked_mul(TAU_MICRO)),
        1_000 * NANOS_PER_SECOND,
    )?;
    let maximum_velocity = derived_positive(
        joint,
        OrganicDerivedQuantity::MaximumVelocity,
        u128::from(declared.maximum_velocity_ticks_per_second())
            .checked_mul(period_ns)
            .and_then(|value| value.checked_mul(MICRO as u128)),
        NANOS_PER_SECOND,
    )?;
    let period_squared = period_ns.checked_mul(period_ns).ok_or(
        OrganicHeadMotionBindingError::ArithmeticOverflow {
            joint,
            quantity: OrganicDerivedQuantity::MaximumAcceleration,
        },
    )?;
    let maximum_acceleration = derived_positive(
        joint,
        OrganicDerivedQuantity::MaximumAcceleration,
        u128::from(declared.maximum_acceleration_ticks_per_second_squared())
            .checked_mul(period_squared)
            .and_then(|value| value.checked_mul(MICRO as u128)),
        NANOS_PER_SECOND * NANOS_PER_SECOND,
    )?;
    let period_cubed = period_squared.checked_mul(period_ns).ok_or(
        OrganicHeadMotionBindingError::ArithmeticOverflow {
            joint,
            quantity: OrganicDerivedQuantity::MaximumJerk,
        },
    )?;
    let maximum_jerk = derived_positive(
        joint,
        OrganicDerivedQuantity::MaximumJerk,
        u128::from(declared.maximum_jerk_ticks_per_second_cubed())
            .checked_mul(period_cubed)
            .and_then(|value| value.checked_mul(MICRO as u128)),
        NANOS_PER_SECOND * NANOS_PER_SECOND * NANOS_PER_SECOND,
    )?;

    let planner_velocity = u64::try_from(
        i64::from(limits.maximum_velocity().get())
            .min(i64::from(limits.maximum_position_step().get()))
            * MICRO,
    )
    .expect("positive planner velocity fits u64");
    if maximum_velocity > planner_velocity {
        return Err(OrganicHeadMotionBindingError::VelocityExceedsPlanner {
            joint,
            organic_microticks_per_tick: maximum_velocity,
            planner_microticks_per_tick: planner_velocity,
        });
    }
    let planner_acceleration =
        u64::try_from(i64::from(limits.maximum_acceleration().get()) * MICRO)
            .expect("positive planner acceleration fits u64");
    let maximum_attacked_acceleration = maximum_acceleration.checked_mul(2).ok_or(
        OrganicHeadMotionBindingError::ArithmeticOverflow {
            joint,
            quantity: OrganicDerivedQuantity::MaximumAcceleration,
        },
    )?;
    if maximum_attacked_acceleration > planner_acceleration {
        return Err(OrganicHeadMotionBindingError::AccelerationExceedsPlanner {
            joint,
            organic_microticks_per_tick_squared: maximum_attacked_acceleration,
            planner_microticks_per_tick_squared: planner_acceleration,
        });
    }

    let settle_velocity = derived_positive(
        joint,
        OrganicDerivedQuantity::SettleVelocity,
        50_u128
            .checked_mul(period_ns)
            .and_then(|value| value.checked_mul(MICRO as u128)),
        1_000 * NANOS_PER_SECOND,
    )?;
    let settle_acceleration = derived_positive(
        joint,
        OrganicDerivedQuantity::SettleAcceleration,
        200_u128
            .checked_mul(period_squared)
            .and_then(|value| value.checked_mul(MICRO as u128)),
        1_000 * NANOS_PER_SECOND * NANOS_PER_SECOND,
    )?;
    let decay_denominator = ATTACK_DECAY_TIME.as_nanos().checked_add(period_ns).ok_or(
        OrganicHeadMotionBindingError::ArithmeticOverflow {
            joint,
            quantity: OrganicDerivedQuantity::AttackDecay,
        },
    )?;
    let attack_decay = derived_positive(
        joint,
        OrganicDerivedQuantity::AttackDecay,
        ATTACK_DECAY_TIME.as_nanos().checked_mul(MICRO as u128),
        decay_denominator,
    )?;
    let span = u32::from(limits.maximum().get()) - u32::from(limits.minimum().get());
    let attack_trigger_ticks =
        u16::try_from((u64::from(span) * ATTACK_TRIGGER_PERCENT).div_ceil(100))
            .expect("u16 position span percentage fits u16")
            .max(1);

    Ok(BoundOrganicJointMotionPolicy {
        minimum_microticks: i64::from(limits.minimum().get()) * MICRO,
        maximum_microticks: i64::from(limits.maximum().get()) * MICRO,
        omega_dt_micro: i64::try_from(omega_dt_micro).expect("derived value admitted to i64"),
        damping_micro: i64::from(declared.damping_permille()) * MICRO / 1_000,
        maximum_velocity_microticks_per_tick: i64::try_from(maximum_velocity)
            .expect("derived value admitted to i64"),
        maximum_acceleration_microticks_per_tick_squared: i64::try_from(maximum_acceleration)
            .expect("derived value admitted to i64"),
        maximum_jerk_microticks_per_tick_cubed: i64::try_from(maximum_jerk)
            .expect("derived value admitted to i64"),
        attack_trigger_ticks,
        attack_decay_micro: i64::try_from(attack_decay).expect("derived value admitted to i64"),
        settle_velocity_microticks_per_tick: i64::try_from(settle_velocity)
            .expect("derived value admitted to i64"),
        settle_acceleration_microticks_per_tick_squared: i64::try_from(settle_acceleration)
            .expect("derived value admitted to i64"),
    })
}

fn derived_positive(
    joint: HeadJoint,
    quantity: OrganicDerivedQuantity,
    numerator: Option<u128>,
    denominator: u128,
) -> Result<u64, OrganicHeadMotionBindingError> {
    let numerator =
        numerator.ok_or(OrganicHeadMotionBindingError::ArithmeticOverflow { joint, quantity })?;
    let rounded = numerator
        .checked_add(denominator / 2)
        .ok_or(OrganicHeadMotionBindingError::ArithmeticOverflow { joint, quantity })?
        / denominator;
    if rounded == 0 {
        return Err(OrganicHeadMotionBindingError::BelowOneMicroUnit { joint, quantity });
    }
    let rounded = u64::try_from(rounded)
        .map_err(|_| OrganicHeadMotionBindingError::ArithmeticOverflow { joint, quantity })?;
    if rounded > i64::MAX as u64 {
        return Err(OrganicHeadMotionBindingError::ArithmeticOverflow { joint, quantity });
    }
    Ok(rounded)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OrganicHeadMotionState {
    axes: [OrganicAxisState; JOINT_COUNT],
}

impl OrganicHeadMotionState {
    pub(crate) fn new(initial: ExactHeadTargetPose) -> Self {
        Self {
            axes: HeadJoint::ALL.map(|joint| OrganicAxisState::new(initial.position(joint))),
        }
    }

    pub(crate) fn step(
        &mut self,
        target: ExactHeadTargetPose,
        policy: BoundOrganicHeadMotionPolicy,
    ) -> Result<ExactHeadTargetPose, OrganicMotionStepError> {
        let mut positions = target.positions();
        for joint in HeadJoint::ALL {
            positions[joint_index(joint)] = self.axes[joint_index(joint)].step(
                joint,
                target.position(joint),
                policy.joint(joint),
            )?;
        }
        Ok(ExactHeadTargetPose::from_positions(
            positions[0],
            positions[1],
            positions[2],
            positions[3],
        ))
    }

    pub(crate) fn is_settled_at(&self, target: ExactHeadTargetPose) -> bool {
        HeadJoint::ALL
            .into_iter()
            .all(|joint| self.axes[joint_index(joint)].is_settled_at(target.position(joint)))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OrganicAxisState {
    position_microticks: i64,
    velocity_microticks_per_tick: i64,
    acceleration_microticks_per_tick_squared: i64,
    attack_micro: i64,
    previous_target: PositionTicks,
}

impl OrganicAxisState {
    fn new(initial: PositionTicks) -> Self {
        Self {
            position_microticks: i64::from(initial.get()) * MICRO,
            velocity_microticks_per_tick: 0,
            acceleration_microticks_per_tick_squared: 0,
            attack_micro: 0,
            previous_target: initial,
        }
    }

    fn step(
        &mut self,
        joint: HeadJoint,
        target: PositionTicks,
        policy: BoundOrganicJointMotionPolicy,
    ) -> Result<PositionTicks, OrganicMotionStepError> {
        let target_jump = target.get().abs_diff(self.previous_target.get());
        if target_jump >= policy.attack_trigger_ticks {
            self.attack_micro = MICRO;
        }
        self.previous_target = target;

        let response_multiplier =
            MICRO + scaled(ATTACK_RESPONSE_GAIN_MICRO, self.attack_micro, joint)?;
        let omega_dt = scaled(policy.omega_dt_micro, response_multiplier, joint)?;
        let damping_multiplier =
            MICRO - scaled(ATTACK_DAMPING_CUT_MICRO, self.attack_micro, joint)?;
        let damping = scaled(policy.damping_micro, damping_multiplier, joint)?;
        let stiffness = scaled(omega_dt, omega_dt, joint)?;
        let damping_coefficient = scaled(2 * damping, omega_dt, joint)?;
        let target_microticks = i64::from(target.get()) * MICRO;
        let error = target_microticks
            .checked_sub(self.position_microticks)
            .ok_or(OrganicMotionStepError::ArithmeticOverflow { joint })?;
        let spring = scaled(stiffness, error, joint)?;
        let drag = scaled(
            damping_coefficient,
            self.velocity_microticks_per_tick,
            joint,
        )?;
        let acceleration_multiplier =
            MICRO + scaled(ATTACK_ACCELERATION_GAIN_MICRO, self.attack_micro, joint)?;
        let acceleration_cap = scaled(
            policy.maximum_acceleration_microticks_per_tick_squared,
            acceleration_multiplier,
            joint,
        )?;
        let desired_acceleration = spring
            .checked_sub(drag)
            .ok_or(OrganicMotionStepError::ArithmeticOverflow { joint })?
            .clamp(-acceleration_cap, acceleration_cap);
        let jerk_multiplier = MICRO + scaled(ATTACK_JERK_GAIN_MICRO, self.attack_micro, joint)?;
        let jerk_cap = scaled(
            policy.maximum_jerk_microticks_per_tick_cubed,
            jerk_multiplier,
            joint,
        )?;
        let next_acceleration = move_toward(
            self.acceleration_microticks_per_tick_squared,
            desired_acceleration,
            jerk_cap,
        );
        let mean_acceleration = midpoint(
            self.acceleration_microticks_per_tick_squared,
            next_acceleration,
            joint,
        )?;
        let next_velocity = self
            .velocity_microticks_per_tick
            .checked_add(mean_acceleration)
            .ok_or(OrganicMotionStepError::ArithmeticOverflow { joint })?
            .clamp(
                -policy.maximum_velocity_microticks_per_tick,
                policy.maximum_velocity_microticks_per_tick,
            );
        let mean_velocity = midpoint(self.velocity_microticks_per_tick, next_velocity, joint)?;
        let mut next_position = self
            .position_microticks
            .checked_add(mean_velocity)
            .ok_or(OrganicMotionStepError::ArithmeticOverflow { joint })?;
        let mut next_velocity = next_velocity;
        let mut next_acceleration = next_acceleration;

        let new_error = target_microticks
            .checked_sub(next_position)
            .ok_or(OrganicMotionStepError::ArithmeticOverflow { joint })?;
        if new_error.abs() <= POSITION_SETTLE_MICROTICKS
            && next_velocity.abs() <= policy.settle_velocity_microticks_per_tick
            && next_acceleration.abs() <= policy.settle_acceleration_microticks_per_tick_squared
        {
            next_position = target_microticks;
            next_velocity = 0;
            next_acceleration = 0;
        }
        if next_position <= policy.minimum_microticks {
            next_position = policy.minimum_microticks;
            next_velocity = next_velocity.max(0);
            next_acceleration = next_acceleration.max(0);
        } else if next_position >= policy.maximum_microticks {
            next_position = policy.maximum_microticks;
            next_velocity = next_velocity.min(0);
            next_acceleration = next_acceleration.min(0);
        }

        self.position_microticks = next_position;
        self.velocity_microticks_per_tick = next_velocity;
        self.acceleration_microticks_per_tick_squared = next_acceleration;
        self.attack_micro = scaled(self.attack_micro, policy.attack_decay_micro, joint)?;
        PositionTicks::try_new(self.rounded_position())
            .map_err(|_| OrganicMotionStepError::ArithmeticOverflow { joint })
    }

    fn rounded_position(self) -> u16 {
        u16::try_from((self.position_microticks + MICRO / 2) / MICRO)
            .expect("bounded nonnegative servo position fits u16")
    }

    fn is_settled_at(self, target: PositionTicks) -> bool {
        self.position_microticks == i64::from(target.get()) * MICRO
            && self.velocity_microticks_per_tick == 0
            && self.acceleration_microticks_per_tick_squared == 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OrganicMotionStepError {
    ArithmeticOverflow { joint: HeadJoint },
}

fn scaled(left: i64, right: i64, joint: HeadJoint) -> Result<i64, OrganicMotionStepError> {
    let numerator = i128::from(left) * i128::from(right);
    let magnitude = numerator.unsigned_abs();
    let rounded = (magnitude + (MICRO as u128) / 2) / MICRO as u128;
    let signed = if numerator.is_negative() {
        -(i128::try_from(rounded)
            .map_err(|_| OrganicMotionStepError::ArithmeticOverflow { joint })?)
    } else {
        i128::try_from(rounded).map_err(|_| OrganicMotionStepError::ArithmeticOverflow { joint })?
    };
    i64::try_from(signed).map_err(|_| OrganicMotionStepError::ArithmeticOverflow { joint })
}

fn midpoint(left: i64, right: i64, joint: HeadJoint) -> Result<i64, OrganicMotionStepError> {
    left.checked_add(right)
        .ok_or(OrganicMotionStepError::ArithmeticOverflow { joint })
        .map(|sum| sum / 2)
}

fn move_toward(current: i64, target: i64, maximum_delta: i64) -> i64 {
    current + (target - current).clamp(-maximum_delta, maximum_delta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaze_control::{
        HeadJointMotionLimits, ServoAccelerationLimitTicksPerControlTickSquared,
        ServoVelocityLimitTicksPerControlTick,
    };
    use kiko_head_protocol::PositionStepLimit;

    fn position(value: u16) -> PositionTicks {
        PositionTicks::try_new(value).expect("test position")
    }

    fn declared(
        response_millihertz: u32,
        damping_permille: u32,
        velocity: u32,
        acceleration: u32,
        jerk: u32,
    ) -> OrganicJointMotionPolicy {
        OrganicJointMotionPolicy::parse(OrganicJointMotionPolicyInput {
            response_millihertz,
            damping_permille,
            maximum_velocity_ticks_per_second: velocity,
            maximum_acceleration_ticks_per_second_squared: acceleration,
            maximum_jerk_ticks_per_second_cubed: jerk,
        })
        .expect("test organic policy")
    }

    fn limits() -> HeadMotionLimits {
        let joint = HeadJointMotionLimits::try_new(
            position(900),
            position(1_100),
            ServoVelocityLimitTicksPerControlTick::try_new(8).unwrap(),
            ServoAccelerationLimitTicksPerControlTickSquared::try_new(2).unwrap(),
            PositionStepLimit::try_new(8).unwrap(),
        )
        .unwrap();
        HeadMotionLimits::new(joint, joint, joint, joint)
    }

    fn fable_policy() -> OrganicHeadMotionPolicy {
        OrganicHeadMotionPolicy::new(
            declared(400, 1_400, 100, 400, 3_200),
            declared(850, 1_400, 170, 640, 5_200),
            declared(1_050, 1_150, 320, 1_600, 14_000),
            declared(900, 850, 200, 1_000, 9_000),
        )
    }

    #[test]
    fn weak_policy_values_are_parsed_once() {
        assert_eq!(
            OrganicJointMotionPolicy::parse(OrganicJointMotionPolicyInput {
                response_millihertz: 0,
                damping_permille: 1_400,
                maximum_velocity_ticks_per_second: 100,
                maximum_acceleration_ticks_per_second_squared: 400,
                maximum_jerk_ticks_per_second_cubed: 3_200,
            }),
            Err(OrganicJointMotionPolicyError::Zero {
                field: OrganicJointMotionPolicyField::ResponseMilliHertz,
            })
        );
        assert!(matches!(
            OrganicJointMotionPolicy::parse(OrganicJointMotionPolicyInput {
                response_millihertz: 400,
                damping_permille: 499,
                maximum_velocity_ticks_per_second: 100,
                maximum_acceleration_ticks_per_second_squared: 400,
                maximum_jerk_ticks_per_second_cubed: 3_200,
            }),
            Err(OrganicJointMotionPolicyError::DampingOutsideReviewedRange { .. })
        ));
        assert!(matches!(
            OrganicJointMotionPolicy::parse(OrganicJointMotionPolicyInput {
                response_millihertz: 400,
                damping_permille: 1_400,
                maximum_velocity_ticks_per_second: u32::from(u16::MAX) + 1,
                maximum_acceleration_ticks_per_second_squared: 400,
                maximum_jerk_ticks_per_second_cubed: 3_200,
            }),
            Err(OrganicJointMotionPolicyError::AboveRepresentableMaximum {
                field: OrganicJointMotionPolicyField::MaximumVelocityTicksPerSecond,
                actual: 65_536,
                maximum: 65_535,
            })
        ));
    }

    #[test]
    fn fable_si_rates_bind_to_exact_twenty_millisecond_microtick_units() {
        let period = HeadControlPeriod::try_new(Duration::from_millis(20)).unwrap();
        let bound = fable_policy().bind(period, limits()).unwrap();
        let bow = bound.joint(HeadJoint::Bow);
        assert_eq!(bow.omega_dt_micro, 50_265);
        assert_eq!(bow.maximum_velocity_microticks_per_tick, 2_000_000);
        assert_eq!(
            bow.maximum_acceleration_microticks_per_tick_squared,
            160_000
        );
        assert_eq!(bow.maximum_jerk_microticks_per_tick_cubed, 25_600);
        assert_eq!(bow.attack_trigger_ticks, 10);
        assert_eq!(bow.attack_decay_micro, 945_946);
    }

    #[test]
    fn binding_rejects_a_prefilter_that_can_outrun_the_safety_planner() {
        let period = HeadControlPeriod::try_new(Duration::from_millis(20)).unwrap();
        let too_fast = declared(400, 1_400, 1_000, 400, 3_200);
        let policy = OrganicHeadMotionPolicy::new(too_fast, too_fast, too_fast, too_fast);
        assert!(matches!(
            policy.bind(period, limits()),
            Err(OrganicHeadMotionBindingError::VelocityExceedsPlanner {
                joint: HeadJoint::Bow,
                ..
            })
        ));

        let attack_accelerates_too_hard = declared(400, 1_400, 100, 4_000, 3_200);
        let policy = OrganicHeadMotionPolicy::new(
            attack_accelerates_too_hard,
            attack_accelerates_too_hard,
            attack_accelerates_too_hard,
            attack_accelerates_too_hard,
        );
        assert!(matches!(
            policy.bind(period, limits()),
            Err(OrganicHeadMotionBindingError::AccelerationExceedsPlanner {
                joint: HeadJoint::Bow,
                organic_microticks_per_tick_squared: 3_200_000,
                planner_microticks_per_tick_squared: 2_000_000,
            })
        ));

        assert!(matches!(
            fable_policy().bind(HeadControlPeriod::try_new(Duration::MAX).unwrap(), limits()),
            Err(OrganicHeadMotionBindingError::ArithmeticOverflow { .. })
        ));
    }

    #[test]
    fn target_jump_is_jerk_bounded_and_every_state_stays_inside_the_envelope() {
        let period = HeadControlPeriod::try_new(Duration::from_millis(20)).unwrap();
        let bound = fable_policy().bind(period, limits()).unwrap();
        let neutral = ExactHeadTargetPose::try_from_ticks([1_000; 4]).unwrap();
        let target = ExactHeadTargetPose::try_from_ticks([1_100, 900, 1_100, 900]).unwrap();
        let mut state = OrganicHeadMotionState::new(neutral);
        let first = state.step(target, bound).unwrap();
        assert_eq!(first, neutral, "subtick launch must not become a step jump");
        for joint in HeadJoint::ALL {
            let axis = state.axes[joint_index(joint)];
            let policy = bound.joint(joint);
            assert!(
                axis.acceleration_microticks_per_tick_squared.abs()
                    <= policy.maximum_jerk_microticks_per_tick_cubed * 5 / 2
            );
        }

        for _ in 0..2_000 {
            let output = state.step(target, bound).unwrap();
            for joint in HeadJoint::ALL {
                assert!((900..=1_100).contains(&output.position(joint).get()));
                let axis = state.axes[joint_index(joint)];
                let policy = bound.joint(joint);
                assert!(
                    axis.velocity_microticks_per_tick.abs()
                        <= policy.maximum_velocity_microticks_per_tick
                );
                assert!(
                    axis.acceleration_microticks_per_tick_squared.abs()
                        <= 2 * policy.maximum_acceleration_microticks_per_tick_squared
                );
            }
        }
        assert!(state.is_settled_at(target));
    }

    #[test]
    fn repeated_runs_are_bit_exact() {
        let period = HeadControlPeriod::try_new(Duration::from_millis(20)).unwrap();
        let bound = fable_policy().bind(period, limits()).unwrap();
        let neutral = ExactHeadTargetPose::try_from_ticks([1_000; 4]).unwrap();
        let mut left = OrganicHeadMotionState::new(neutral);
        let mut right = OrganicHeadMotionState::new(neutral);
        for step in 0..5_000_u16 {
            let offset = step.wrapping_mul(37) % 201;
            let target = ExactHeadTargetPose::try_from_ticks([
                900 + offset,
                1_100 - offset,
                900 + (offset * 3 % 201),
                1_100 - (offset * 7 % 201),
            ])
            .unwrap();
            assert_eq!(left.step(target, bound), right.step(target, bound));
            assert_eq!(left, right);
        }
    }
}
