//! Transport-free, four-axis compliant-hold planning.
//!
//! This controller lets an already torque-limited head yield to a deliberate
//! external displacement, briefly settle with it, and return to the pose that
//! was active when contact began. It never interprets the STS `load_raw` or
//! `current_raw` registers as force: their sign and physical units are not
//! qualified by Kiko's protocol contract. Contact evidence is instead a
//! repeatable encoder error against the sole owner's verified goal.
//!
//! The controller is transactional. Preparing a step advances only a private
//! candidate snapshot. The sole bus owner must apply and verify the complete
//! four-joint goal before committing it. A partial or uncertain write must be
//! fault-aborted, never committed.

use std::{
    fmt,
    num::{NonZeroU8, NonZeroU16, NonZeroU64},
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use kiko_head_protocol::{
    ExactHeadTargetPose, FullTelemetry, HeadJoint, HeadTorqueLimits, PositionTicks,
};

use crate::energized_temperature::EnergizedTemperatureAdmission;
use crate::{HeadTelemetrySafetyLimits, HeadTelemetrySafetyViolation, MonotonicTime};

const JOINT_COUNT: usize = 4;
const INTERPOLATION_SCALE: u128 = 1_000_000;
const YIELD_TORQUE_FLOOR_PERMILLE: [u16; JOINT_COUNT] = [300, 200, 150, 150];
static NEXT_COMPLIANT_CONTROLLER_ID: AtomicU64 = AtomicU64::new(1);

const fn joint_index(joint: HeadJoint) -> usize {
    joint as usize
}

/// Encoder-domain admission and travel policy for one joint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompliantJointPolicy {
    minimum: PositionTicks,
    maximum: PositionTicks,
    contact_entry_error_ticks: NonZeroU16,
    contact_release_error_ticks: u16,
    maximum_yield_ticks: NonZeroU16,
    maximum_command_step_ticks: NonZeroU16,
    maximum_observed_step_ticks: NonZeroU16,
}

impl CompliantJointPolicy {
    pub fn try_new(
        minimum: PositionTicks,
        maximum: PositionTicks,
        contact_entry_error_ticks: u16,
        contact_release_error_ticks: u16,
        maximum_yield_ticks: u16,
        maximum_command_step_ticks: u16,
        maximum_observed_step_ticks: u16,
    ) -> Result<Self, CompliantJointPolicyError> {
        if minimum >= maximum {
            return Err(CompliantJointPolicyError::EmptyEnvelope { minimum, maximum });
        }
        let contact_entry_error_ticks = NonZeroU16::new(contact_entry_error_ticks)
            .ok_or(CompliantJointPolicyError::ZeroContactEntryError)?;
        if contact_release_error_ticks >= contact_entry_error_ticks.get() {
            return Err(CompliantJointPolicyError::ReleaseNotInsideEntry {
                release_ticks: contact_release_error_ticks,
                entry_ticks: contact_entry_error_ticks.get(),
            });
        }
        let maximum_yield_ticks = NonZeroU16::new(maximum_yield_ticks)
            .ok_or(CompliantJointPolicyError::ZeroMaximumYield)?;
        if maximum_yield_ticks.get() < contact_entry_error_ticks.get() {
            return Err(CompliantJointPolicyError::YieldSmallerThanEntry {
                maximum_yield_ticks: maximum_yield_ticks.get(),
                entry_ticks: contact_entry_error_ticks.get(),
            });
        }
        let envelope_span = maximum.get() - minimum.get();
        if maximum_yield_ticks.get() > envelope_span {
            return Err(CompliantJointPolicyError::YieldExceedsEnvelopeSpan {
                maximum_yield_ticks: maximum_yield_ticks.get(),
                envelope_span_ticks: envelope_span,
            });
        }
        let maximum_command_step_ticks = NonZeroU16::new(maximum_command_step_ticks)
            .ok_or(CompliantJointPolicyError::ZeroMaximumCommandStep)?;
        let maximum_observed_step_ticks = NonZeroU16::new(maximum_observed_step_ticks)
            .ok_or(CompliantJointPolicyError::ZeroMaximumObservedStep)?;
        if maximum_observed_step_ticks.get() < maximum_command_step_ticks.get() {
            return Err(
                CompliantJointPolicyError::ObservedStepSmallerThanCommandStep {
                    maximum_observed_step_ticks: maximum_observed_step_ticks.get(),
                    maximum_command_step_ticks: maximum_command_step_ticks.get(),
                },
            );
        }
        if maximum_observed_step_ticks.get() < contact_entry_error_ticks.get() {
            return Err(CompliantJointPolicyError::ObservedStepSmallerThanEntry {
                maximum_observed_step_ticks: maximum_observed_step_ticks.get(),
                entry_ticks: contact_entry_error_ticks.get(),
            });
        }
        Ok(Self {
            minimum,
            maximum,
            contact_entry_error_ticks,
            contact_release_error_ticks,
            maximum_yield_ticks,
            maximum_command_step_ticks,
            maximum_observed_step_ticks,
        })
    }

    pub const fn minimum(self) -> PositionTicks {
        self.minimum
    }

    pub const fn maximum(self) -> PositionTicks {
        self.maximum
    }

    pub const fn contact_entry_error_ticks(self) -> u16 {
        self.contact_entry_error_ticks.get()
    }

    pub const fn contact_release_error_ticks(self) -> u16 {
        self.contact_release_error_ticks
    }

    pub const fn maximum_yield_ticks(self) -> u16 {
        self.maximum_yield_ticks.get()
    }

    pub const fn maximum_command_step_ticks(self) -> u16 {
        self.maximum_command_step_ticks.get()
    }

    pub const fn maximum_observed_step_ticks(self) -> u16 {
        self.maximum_observed_step_ticks.get()
    }

    const fn contains(self, value: PositionTicks) -> bool {
        value.get() >= self.minimum.get() && value.get() <= self.maximum.get()
    }

    /// Lowest physically observable position admitted while a safe command
    /// remains at the command envelope edge.
    ///
    /// Commands never use this wider range. A person can backdrive a joint by
    /// at most the separately reviewed yield distance beyond an edge, so
    /// rejecting that observation would turn the compliant controller's own
    /// permitted interaction into an absorbing fault.
    pub const fn observation_minimum(self) -> PositionTicks {
        let value = self
            .minimum
            .get()
            .saturating_sub(self.maximum_yield_ticks.get());
        match PositionTicks::try_new(value) {
            Ok(position) => position,
            Err(_) => panic!("a saturating subtraction stays inside the encoder domain"),
        }
    }

    /// Highest physically observable position admitted while a safe command
    /// remains at the command envelope edge.
    pub const fn observation_maximum(self) -> PositionTicks {
        let candidate = self
            .maximum
            .get()
            .saturating_add(self.maximum_yield_ticks.get());
        let value = if candidate > PositionTicks::MAX.get() {
            PositionTicks::MAX.get()
        } else {
            candidate
        };
        match PositionTicks::try_new(value) {
            Ok(position) => position,
            Err(_) => panic!("the observation maximum is capped to the encoder domain"),
        }
    }

    const fn contains_observation(self, value: PositionTicks) -> bool {
        value.get() >= self.observation_minimum().get()
            && value.get() <= self.observation_maximum().get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompliantJointPolicyError {
    EmptyEnvelope {
        minimum: PositionTicks,
        maximum: PositionTicks,
    },
    ZeroContactEntryError,
    ReleaseNotInsideEntry {
        release_ticks: u16,
        entry_ticks: u16,
    },
    ZeroMaximumYield,
    YieldSmallerThanEntry {
        maximum_yield_ticks: u16,
        entry_ticks: u16,
    },
    YieldExceedsEnvelopeSpan {
        maximum_yield_ticks: u16,
        envelope_span_ticks: u16,
    },
    ZeroMaximumCommandStep,
    ZeroMaximumObservedStep,
    ObservedStepSmallerThanCommandStep {
        maximum_observed_step_ticks: u16,
        maximum_command_step_ticks: u16,
    },
    ObservedStepSmallerThanEntry {
        maximum_observed_step_ticks: u16,
        entry_ticks: u16,
    },
}

impl fmt::Display for CompliantJointPolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid compliant joint policy: {self:?}")
    }
}

impl std::error::Error for CompliantJointPolicyError {}

/// Pet-specific policy for one joint.
///
/// A stopped, bounded encoder bias is learned before contact is armed. This
/// keeps pose-dependent gravity sag distinct from a person's displacement.
/// Rest offsets are signed offsets from the expression target captured when
/// the contact episode begins; directional offsets lean with the admitted
/// contact direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompliantPetJointPolicy {
    maximum_baseline_error_ticks: NonZeroU16,
    rest_offset_ticks: i16,
    directional_rest_offset_ticks: u16,
}

impl CompliantPetJointPolicy {
    pub fn try_new(
        maximum_baseline_error_ticks: u16,
        rest_offset_ticks: i16,
        directional_rest_offset_ticks: u16,
    ) -> Result<Self, CompliantPetJointPolicyError> {
        let maximum_baseline_error_ticks = NonZeroU16::new(maximum_baseline_error_ticks)
            .ok_or(CompliantPetJointPolicyError::ZeroMaximumBaselineError)?;
        if rest_offset_ticks.unsigned_abs() > PositionTicks::MAX.get() {
            return Err(
                CompliantPetJointPolicyError::RestOffsetOutsideEncoderDomain {
                    ticks: rest_offset_ticks,
                },
            );
        }
        if directional_rest_offset_ticks > PositionTicks::MAX.get() {
            return Err(
                CompliantPetJointPolicyError::DirectionalRestOffsetOutsideEncoderDomain {
                    ticks: directional_rest_offset_ticks,
                },
            );
        }
        Ok(Self {
            maximum_baseline_error_ticks,
            rest_offset_ticks,
            directional_rest_offset_ticks,
        })
    }

    pub const fn maximum_baseline_error_ticks(self) -> u16 {
        self.maximum_baseline_error_ticks.get()
    }

    pub const fn rest_offset_ticks(self) -> i16 {
        self.rest_offset_ticks
    }

    pub const fn directional_rest_offset_ticks(self) -> u16 {
        self.directional_rest_offset_ticks
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompliantPetJointPolicyError {
    ZeroMaximumBaselineError,
    RestOffsetOutsideEncoderDomain { ticks: i16 },
    DirectionalRestOffsetOutsideEncoderDomain { ticks: u16 },
}

impl fmt::Display for CompliantPetJointPolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid compliant pet joint policy: {self:?}")
    }
}

impl std::error::Error for CompliantPetJointPolicyError {}

/// Field-qualified contact choreography layered on compliant hold.
///
/// Construction parses timing and torque relationships once. Binding the
/// profile to [`HeadCompliantHoldConfig`] performs the remaining checks that
/// depend on command envelopes, follow gain, control period, and holding
/// torque. A controller therefore cannot observe a half-validated pet policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompliantPetProfile {
    joints: [CompliantPetJointPolicy; JOINT_COUNT],
    rest_dwell: Duration,
    rest_per_additional_joint: Duration,
    maximum_rest_dwell: Duration,
    recovery_per_additional_joint_permille: u16,
    static_release_dwell: Duration,
    maximum_yield_dwell: Duration,
    residual_stillness_ticks: NonZeroU16,
    comfort_roll_tilt_ticks: u16,
    yield_torque_limits: HeadTorqueLimits,
    tap_maximum_contact_duration: Duration,
    tap_recovery_duration: Duration,
}

impl CompliantPetProfile {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        bow: CompliantPetJointPolicy,
        curl: CompliantPetJointPolicy,
        yaw: CompliantPetJointPolicy,
        roll: CompliantPetJointPolicy,
        rest_dwell: Duration,
        rest_per_additional_joint: Duration,
        maximum_rest_dwell: Duration,
        recovery_per_additional_joint_permille: u16,
        static_release_dwell: Duration,
        maximum_yield_dwell: Duration,
        residual_stillness_ticks: u16,
        comfort_roll_tilt_ticks: u16,
        yield_torque_limits: HeadTorqueLimits,
        tap_maximum_contact_duration: Duration,
        tap_recovery_duration: Duration,
    ) -> Result<Self, CompliantPetProfileError> {
        if rest_dwell.is_zero() {
            return Err(CompliantPetProfileError::ZeroRestDwell);
        }
        if maximum_rest_dwell < rest_dwell {
            return Err(CompliantPetProfileError::MaximumRestDwellBelowBase {
                base: rest_dwell,
                maximum: maximum_rest_dwell,
            });
        }
        if recovery_per_additional_joint_permille > 1_000 {
            return Err(
                CompliantPetProfileError::RecoveryAdditionalPermilleOutOfRange {
                    value: recovery_per_additional_joint_permille,
                },
            );
        }
        if static_release_dwell.is_zero() {
            return Err(CompliantPetProfileError::ZeroStaticReleaseDwell);
        }
        if maximum_yield_dwell <= static_release_dwell {
            return Err(CompliantPetProfileError::MaximumYieldDwellNotAboveStatic {
                static_release: static_release_dwell,
                maximum_yield: maximum_yield_dwell,
            });
        }
        let residual_stillness_ticks = NonZeroU16::new(residual_stillness_ticks)
            .ok_or(CompliantPetProfileError::ZeroResidualStillness)?;
        if tap_maximum_contact_duration.is_zero() {
            return Err(CompliantPetProfileError::ZeroTapMaximumContactDuration);
        }
        if tap_recovery_duration.is_zero() {
            return Err(CompliantPetProfileError::ZeroTapRecoveryDuration);
        }
        Ok(Self {
            joints: [bow, curl, yaw, roll],
            rest_dwell,
            rest_per_additional_joint,
            maximum_rest_dwell,
            recovery_per_additional_joint_permille,
            static_release_dwell,
            maximum_yield_dwell,
            residual_stillness_ticks,
            comfort_roll_tilt_ticks,
            yield_torque_limits,
            tap_maximum_contact_duration,
            tap_recovery_duration,
        })
    }

    pub const fn joint(self, joint: HeadJoint) -> CompliantPetJointPolicy {
        self.joints[joint_index(joint)]
    }

    pub const fn rest_dwell(self) -> Duration {
        self.rest_dwell
    }

    pub const fn rest_per_additional_joint(self) -> Duration {
        self.rest_per_additional_joint
    }

    pub const fn maximum_rest_dwell(self) -> Duration {
        self.maximum_rest_dwell
    }

    pub const fn recovery_per_additional_joint_permille(self) -> u16 {
        self.recovery_per_additional_joint_permille
    }

    pub const fn static_release_dwell(self) -> Duration {
        self.static_release_dwell
    }

    pub const fn maximum_yield_dwell(self) -> Duration {
        self.maximum_yield_dwell
    }

    pub const fn residual_stillness_ticks(self) -> u16 {
        self.residual_stillness_ticks.get()
    }

    pub const fn comfort_roll_tilt_ticks(self) -> u16 {
        self.comfort_roll_tilt_ticks
    }

    pub const fn yield_torque_limits(self) -> HeadTorqueLimits {
        self.yield_torque_limits
    }

    pub const fn tap_maximum_contact_duration(self) -> Duration {
        self.tap_maximum_contact_duration
    }

    pub const fn tap_recovery_duration(self) -> Duration {
        self.tap_recovery_duration
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompliantPetProfileError {
    ZeroRestDwell,
    MaximumRestDwellBelowBase {
        base: Duration,
        maximum: Duration,
    },
    RecoveryAdditionalPermilleOutOfRange {
        value: u16,
    },
    ZeroStaticReleaseDwell,
    MaximumYieldDwellNotAboveStatic {
        static_release: Duration,
        maximum_yield: Duration,
    },
    ZeroResidualStillness,
    ZeroTapMaximumContactDuration,
    ZeroTapRecoveryDuration,
}

impl fmt::Display for CompliantPetProfileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid compliant pet profile: {self:?}")
    }
}

impl std::error::Error for CompliantPetProfileError {}

/// A complete four-axis compliant-hold policy.
///
/// `holding_torque_limits` are evidence binding, not adaptive output. They
/// must exactly match the torque limits installed by the owning head runtime.
/// Physical commissioning remains responsible for finding the lowest limits
/// that safely support the assembly, especially bow and curl against gravity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadCompliantHoldConfig {
    joints: [CompliantJointPolicy; JOINT_COUNT],
    holding_torque_limits: HeadTorqueLimits,
    control_period: Duration,
    observation_transaction_timeout: Duration,
    maximum_observation_span: Duration,
    observation_ttl: Duration,
    contact_arm_dwell: Duration,
    contact_acquisition_samples: NonZeroU8,
    release_dwell: Duration,
    recovery_duration: Duration,
    follow_permille: NonZeroU16,
    pet_profile: Option<CompliantPetProfile>,
}

impl HeadCompliantHoldConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        bow: CompliantJointPolicy,
        curl: CompliantJointPolicy,
        yaw: CompliantJointPolicy,
        roll: CompliantJointPolicy,
        holding_torque_limits: HeadTorqueLimits,
        control_period: Duration,
        observation_transaction_timeout: Duration,
        maximum_observation_span: Duration,
        observation_ttl: Duration,
        contact_arm_dwell: Duration,
        contact_acquisition_samples: u8,
        release_dwell: Duration,
        recovery_duration: Duration,
        follow_permille: u16,
    ) -> Result<Self, HeadCompliantHoldConfigError> {
        if control_period.is_zero() {
            return Err(HeadCompliantHoldConfigError::ZeroControlPeriod);
        }
        if observation_transaction_timeout.is_zero() {
            return Err(HeadCompliantHoldConfigError::ZeroObservationTransactionTimeout);
        }
        if observation_transaction_timeout > control_period {
            return Err(
                HeadCompliantHoldConfigError::ObservationTransactionExceedsControlPeriod {
                    transaction_timeout: observation_transaction_timeout,
                    control_period,
                },
            );
        }
        if maximum_observation_span.is_zero() {
            return Err(HeadCompliantHoldConfigError::ZeroMaximumObservationSpan);
        }
        if maximum_observation_span > observation_transaction_timeout {
            return Err(
                HeadCompliantHoldConfigError::ObservationSpanExceedsTransaction {
                    maximum_span: maximum_observation_span,
                    transaction_timeout: observation_transaction_timeout,
                },
            );
        }
        if observation_ttl.is_zero() {
            return Err(HeadCompliantHoldConfigError::ZeroObservationTtl);
        }
        if maximum_observation_span >= observation_ttl {
            return Err(HeadCompliantHoldConfigError::ObservationSpanNotInsideTtl {
                maximum_span: maximum_observation_span,
                ttl: observation_ttl,
            });
        }
        if contact_arm_dwell.is_zero() {
            return Err(HeadCompliantHoldConfigError::ZeroContactArmDwell);
        }
        let contact_acquisition_samples = NonZeroU8::new(contact_acquisition_samples)
            .ok_or(HeadCompliantHoldConfigError::ZeroContactAcquisitionSamples)?;
        if release_dwell.is_zero() {
            return Err(HeadCompliantHoldConfigError::ZeroReleaseDwell);
        }
        if recovery_duration.is_zero() {
            return Err(HeadCompliantHoldConfigError::ZeroRecoveryDuration);
        }
        let follow_permille = NonZeroU16::new(follow_permille)
            .ok_or(HeadCompliantHoldConfigError::ZeroFollowPermille)?;
        if follow_permille.get() > 1_000 {
            return Err(HeadCompliantHoldConfigError::FollowPermilleOutOfRange {
                value: follow_permille.get(),
            });
        }
        Ok(Self {
            joints: [bow, curl, yaw, roll],
            holding_torque_limits,
            control_period,
            observation_transaction_timeout,
            maximum_observation_span,
            observation_ttl,
            contact_arm_dwell,
            contact_acquisition_samples,
            release_dwell,
            recovery_duration,
            follow_permille,
            pet_profile: None,
        })
    }

    /// Bind field-qualified pet behavior to this complete hold policy.
    ///
    /// This consumes and returns the configuration so invalid combinations
    /// never escape as a temporarily usable value.
    pub fn try_with_pet_profile(
        mut self,
        profile: CompliantPetProfile,
    ) -> Result<Self, HeadCompliantHoldConfigError> {
        let minimum_static_evidence = self
            .control_period
            .checked_mul(3)
            .ok_or(HeadCompliantHoldConfigError::PetTimingOverflow)?;
        if profile.static_release_dwell() < minimum_static_evidence {
            return Err(
                HeadCompliantHoldConfigError::StaticReleaseBelowThreeControlPeriods {
                    static_release: profile.static_release_dwell(),
                    minimum: minimum_static_evidence,
                },
            );
        }
        if profile.tap_recovery_duration() > self.recovery_duration {
            return Err(
                HeadCompliantHoldConfigError::TapRecoveryExceedsFullRecovery {
                    tap: profile.tap_recovery_duration(),
                    full: self.recovery_duration,
                },
            );
        }
        let maximum_recovery_scale =
            1_000_u32 + 3 * u32::from(profile.recovery_per_additional_joint_permille());
        if scale_duration_permille(self.recovery_duration, maximum_recovery_scale).is_none() {
            return Err(HeadCompliantHoldConfigError::PetTimingOverflow);
        }
        if profile
            .maximum_rest_dwell()
            .checked_add(self.maximum_rest_travel_duration())
            .is_none()
        {
            return Err(HeadCompliantHoldConfigError::PetTimingOverflow);
        }
        for joint in HeadJoint::ALL {
            let hold = self.joint(joint);
            let pet = profile.joint(joint);
            let rest = u32::from(pet.rest_offset_ticks().unsigned_abs());
            let directional = u32::from(pet.directional_rest_offset_ticks());
            if rest + directional > u32::from(hold.maximum_yield_ticks()) {
                return Err(HeadCompliantHoldConfigError::PetRestExceedsMaximumYield {
                    joint,
                    rest_offset_ticks: pet.rest_offset_ticks(),
                    directional_offset_ticks: pet.directional_rest_offset_ticks(),
                    maximum_yield_ticks: hold.maximum_yield_ticks(),
                });
            }
            if profile.residual_stillness_ticks() > hold.maximum_command_step_ticks() {
                return Err(
                    HeadCompliantHoldConfigError::StillnessExceedsMaximumCommandStep {
                        joint,
                        stillness_ticks: profile.residual_stillness_ticks(),
                        maximum_command_step_ticks: hold.maximum_command_step_ticks(),
                    },
                );
            }
            let yielded_at_entry = div_round_nearest(
                i64::from(hold.contact_entry_error_ticks()) * i64::from(self.follow_permille()),
                1_000,
            );
            let retained_error = i64::from(hold.contact_entry_error_ticks()) - yielded_at_entry;
            if retained_error <= i64::from(hold.contact_release_error_ticks()) {
                return Err(HeadCompliantHoldConfigError::FollowCollapsesHysteresis {
                    joint,
                    retained_error_ticks: u16::try_from(retained_error)
                        .expect("positive entry and admitted gain retain a nonnegative error"),
                    release_error_ticks: hold.contact_release_error_ticks(),
                });
            }
            let yield_torque = profile.yield_torque_limits().for_joint(joint).get();
            let holding_torque = self.holding_torque_limits.for_joint(joint).get();
            let floor = YIELD_TORQUE_FLOOR_PERMILLE[joint_index(joint)];
            if yield_torque < floor {
                return Err(
                    HeadCompliantHoldConfigError::YieldTorqueBelowMeasuredFloor {
                        joint,
                        actual_permille: yield_torque,
                        minimum_permille: floor,
                    },
                );
            }
            if yield_torque > holding_torque {
                return Err(HeadCompliantHoldConfigError::YieldTorqueExceedsHolding {
                    joint,
                    yield_permille: yield_torque,
                    holding_permille: holding_torque,
                });
            }
        }
        let roll_pet = profile.joint(HeadJoint::Roll);
        let roll_total = u32::from(roll_pet.rest_offset_ticks().unsigned_abs())
            + u32::from(roll_pet.directional_rest_offset_ticks())
            + u32::from(profile.comfort_roll_tilt_ticks());
        let roll_yield = u32::from(self.joint(HeadJoint::Roll).maximum_yield_ticks());
        if roll_total > roll_yield {
            return Err(
                HeadCompliantHoldConfigError::ComfortRollExceedsMaximumYield {
                    total_ticks: roll_total,
                    maximum_yield_ticks: u16::try_from(roll_yield)
                        .expect("joint policy yield is u16"),
                },
            );
        }
        self.pet_profile = Some(profile);
        Ok(self)
    }

    pub const fn joint(self, joint: HeadJoint) -> CompliantJointPolicy {
        self.joints[joint_index(joint)]
    }

    pub const fn holding_torque_limits(self) -> HeadTorqueLimits {
        self.holding_torque_limits
    }

    pub const fn control_period(self) -> Duration {
        self.control_period
    }

    pub const fn observation_transaction_timeout(self) -> Duration {
        self.observation_transaction_timeout
    }

    pub const fn maximum_observation_span(self) -> Duration {
        self.maximum_observation_span
    }

    pub const fn observation_ttl(self) -> Duration {
        self.observation_ttl
    }

    pub const fn contact_arm_dwell(self) -> Duration {
        self.contact_arm_dwell
    }

    pub const fn contact_acquisition_samples(self) -> u8 {
        self.contact_acquisition_samples.get()
    }

    pub const fn release_dwell(self) -> Duration {
        self.release_dwell
    }

    pub const fn recovery_duration(self) -> Duration {
        self.recovery_duration
    }

    pub const fn follow_permille(self) -> u16 {
        self.follow_permille.get()
    }

    pub const fn pet_profile(self) -> Option<CompliantPetProfile> {
        self.pet_profile
    }

    fn maximum_rest_travel_duration(self) -> Duration {
        let maximum_slots = HeadJoint::ALL
            .into_iter()
            .map(|joint| {
                let policy = self.joint(joint);
                let travel = u32::from(policy.maximum_yield_ticks()) * 2;
                let step = u32::from(policy.maximum_command_step_ticks());
                travel.div_ceil(step)
            })
            .max()
            .expect("a Kiko head always has four joints");
        self.control_period
            .checked_mul(maximum_slots)
            .unwrap_or(Duration::MAX)
    }

    pub fn admit_runtime_torque_limits(
        self,
        actual: HeadTorqueLimits,
    ) -> Result<(), HeadCompliantTorqueBindingError> {
        for joint in HeadJoint::ALL {
            let required = self.holding_torque_limits.for_joint(joint);
            let actual = actual.for_joint(joint);
            if actual != required {
                return Err(HeadCompliantTorqueBindingError::Mismatch {
                    joint,
                    required,
                    actual,
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadCompliantHoldConfigError {
    ZeroControlPeriod,
    ZeroObservationTransactionTimeout,
    ObservationTransactionExceedsControlPeriod {
        transaction_timeout: Duration,
        control_period: Duration,
    },
    ZeroMaximumObservationSpan,
    ObservationSpanExceedsTransaction {
        maximum_span: Duration,
        transaction_timeout: Duration,
    },
    ZeroObservationTtl,
    ObservationSpanNotInsideTtl {
        maximum_span: Duration,
        ttl: Duration,
    },
    ZeroContactArmDwell,
    ZeroContactAcquisitionSamples,
    ZeroReleaseDwell,
    ZeroRecoveryDuration,
    ZeroFollowPermille,
    FollowPermilleOutOfRange {
        value: u16,
    },
    PetTimingOverflow,
    StaticReleaseBelowThreeControlPeriods {
        static_release: Duration,
        minimum: Duration,
    },
    TapRecoveryExceedsFullRecovery {
        tap: Duration,
        full: Duration,
    },
    PetRestExceedsMaximumYield {
        joint: HeadJoint,
        rest_offset_ticks: i16,
        directional_offset_ticks: u16,
        maximum_yield_ticks: u16,
    },
    StillnessExceedsMaximumCommandStep {
        joint: HeadJoint,
        stillness_ticks: u16,
        maximum_command_step_ticks: u16,
    },
    FollowCollapsesHysteresis {
        joint: HeadJoint,
        retained_error_ticks: u16,
        release_error_ticks: u16,
    },
    YieldTorqueBelowMeasuredFloor {
        joint: HeadJoint,
        actual_permille: u16,
        minimum_permille: u16,
    },
    YieldTorqueExceedsHolding {
        joint: HeadJoint,
        yield_permille: u16,
        holding_permille: u16,
    },
    ComfortRollExceedsMaximumYield {
        total_ticks: u32,
        maximum_yield_ticks: u16,
    },
}

impl fmt::Display for HeadCompliantHoldConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid compliant-hold configuration: {self:?}")
    }
}

impl std::error::Error for HeadCompliantHoldConfigError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadCompliantTorqueBindingError {
    Mismatch {
        joint: HeadJoint,
        required: kiko_head_protocol::TorqueLimitPermille,
        actual: kiko_head_protocol::TorqueLimitPermille,
    },
}

impl fmt::Display for HeadCompliantTorqueBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "compliant-hold torque binding failed: {self:?}")
    }
}

impl std::error::Error for HeadCompliantTorqueBindingError {}

/// One complete telemetry observation admitted for compliant control.
///
/// Raw load/current are retained for diagnostics and future calibration only.
/// They do not influence contact, yield, or recovery decisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompliantHeadObservation {
    observed_at: MonotonicTime,
    positions: [PositionTicks; JOINT_COUNT],
    moving: [bool; JOINT_COUNT],
    load_raw: [u16; JOINT_COUNT],
    current_raw: [u16; JOINT_COUNT],
}

impl CompliantHeadObservation {
    pub fn try_from_timed_telemetry(
        samples: [FullTelemetry; JOINT_COUNT],
        received_at: [MonotonicTime; JOINT_COUNT],
        admitted_at: MonotonicTime,
        safety: HeadTelemetrySafetyLimits,
        maximum_span: Duration,
        ttl: Duration,
    ) -> Result<Self, CompliantHeadObservationError> {
        Self::try_from_timed_telemetry_inner(
            samples,
            received_at,
            admitted_at,
            safety,
            maximum_span,
            ttl,
            None,
        )
    }

    pub(crate) fn try_from_supervised_timed_telemetry(
        samples: [FullTelemetry; JOINT_COUNT],
        received_at: [MonotonicTime; JOINT_COUNT],
        admitted_at: MonotonicTime,
        safety: HeadTelemetrySafetyLimits,
        maximum_span: Duration,
        ttl: Duration,
        temperature_admission: EnergizedTemperatureAdmission,
    ) -> Result<Self, CompliantHeadObservationError> {
        Self::try_from_timed_telemetry_inner(
            samples,
            received_at,
            admitted_at,
            safety,
            maximum_span,
            ttl,
            Some(temperature_admission),
        )
    }

    fn try_from_timed_telemetry_inner(
        samples: [FullTelemetry; JOINT_COUNT],
        received_at: [MonotonicTime; JOINT_COUNT],
        admitted_at: MonotonicTime,
        safety: HeadTelemetrySafetyLimits,
        maximum_span: Duration,
        ttl: Duration,
        temperature_admission: Option<EnergizedTemperatureAdmission>,
    ) -> Result<Self, CompliantHeadObservationError> {
        let mut positions = [PositionTicks::MIN; JOINT_COUNT];
        let mut moving = [false; JOINT_COUNT];
        let mut load_raw = [0; JOINT_COUNT];
        let mut current_raw = [0; JOINT_COUNT];
        let mut first_temperature_violation = None;
        for index in 1..JOINT_COUNT {
            if received_at[index] < received_at[index - 1] {
                return Err(CompliantHeadObservationError::ClockRegression {
                    previous_joint: HeadJoint::ALL[index - 1],
                    previous: received_at[index - 1],
                    actual_joint: HeadJoint::ALL[index],
                    actual: received_at[index],
                });
            }
        }
        let span = received_at[JOINT_COUNT - 1]
            .checked_duration_since(received_at[0])
            .expect("ordered timestamps have a duration");
        if span > maximum_span {
            return Err(CompliantHeadObservationError::SetSpanExceeded {
                first: received_at[0],
                last: received_at[JOINT_COUNT - 1],
                span,
                maximum: maximum_span,
            });
        }
        let Some(age) = admitted_at.checked_duration_since(received_at[0]) else {
            return Err(CompliantHeadObservationError::AdmittedBeforeObservation {
                first: received_at[0],
                admitted_at,
            });
        };
        if age >= ttl {
            return Err(CompliantHeadObservationError::SetExpired {
                first: received_at[0],
                admitted_at,
                age,
                ttl,
            });
        }
        for (index, joint) in HeadJoint::ALL.into_iter().enumerate() {
            let sample = samples[index];
            if sample.id() != joint.servo_id() {
                return Err(CompliantHeadObservationError::ServoIdMismatch {
                    joint,
                    expected: joint.servo_id(),
                    actual: sample.id(),
                });
            }
            if sample.device_status_raw() != 0 {
                return Err(CompliantHeadObservationError::DeviceStatus {
                    joint,
                    raw: sample.device_status_raw(),
                });
            }
            let telemetry_admission = if let Some(temperature_admission) = temperature_admission {
                if !temperature_admission.admits(
                    index,
                    sample.temperature_raw(),
                    received_at[index],
                ) {
                    return Err(
                        CompliantHeadObservationError::TemperatureAdmissionMismatch {
                            joint,
                            observed_raw: sample.temperature_raw(),
                            observed_at: received_at[index],
                        },
                    );
                }
                safety.admit_energized_voltage(sample.voltage_raw())
            } else {
                safety.admit_energized(sample.voltage_raw(), sample.temperature_raw())
            };
            match telemetry_admission {
                Ok(()) => {}
                Err(
                    source @ HeadTelemetrySafetyViolation::EnergizedTemperatureAtOrAboveExclusiveMaximum {
                        ..
                    },
                ) => {
                    first_temperature_violation.get_or_insert((joint, source));
                }
                Err(source) => {
                    return Err(CompliantHeadObservationError::TelemetrySafety {
                        joint,
                        source,
                    });
                }
            }
            positions[index] = sample.position();
            moving[index] = sample.is_moving();
            load_raw[index] = sample.load_raw();
            current_raw[index] = sample.current_raw();
        }
        if let Some((joint, source)) = first_temperature_violation {
            return Err(CompliantHeadObservationError::TelemetrySafety { joint, source });
        }
        Ok(Self {
            observed_at: received_at[JOINT_COUNT - 1],
            positions,
            moving,
            load_raw,
            current_raw,
        })
    }

    #[cfg(test)]
    fn from_parts(
        observed_at: MonotonicTime,
        positions: [u16; JOINT_COUNT],
        moving: [bool; JOINT_COUNT],
    ) -> Self {
        Self {
            observed_at,
            positions: positions.map(|value| PositionTicks::try_new(value).unwrap()),
            moving,
            load_raw: [0; JOINT_COUNT],
            current_raw: [0; JOINT_COUNT],
        }
    }

    pub const fn observed_at(self) -> MonotonicTime {
        self.observed_at
    }

    pub const fn position(self, joint: HeadJoint) -> PositionTicks {
        self.positions[joint_index(joint)]
    }

    pub const fn is_moving(self, joint: HeadJoint) -> bool {
        self.moving[joint_index(joint)]
    }

    pub const fn load_raw(self, joint: HeadJoint) -> u16 {
        self.load_raw[joint_index(joint)]
    }

    pub const fn current_raw(self, joint: HeadJoint) -> u16 {
        self.current_raw[joint_index(joint)]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompliantHeadObservationError {
    ClockRegression {
        previous_joint: HeadJoint,
        previous: MonotonicTime,
        actual_joint: HeadJoint,
        actual: MonotonicTime,
    },
    SetSpanExceeded {
        first: MonotonicTime,
        last: MonotonicTime,
        span: Duration,
        maximum: Duration,
    },
    AdmittedBeforeObservation {
        first: MonotonicTime,
        admitted_at: MonotonicTime,
    },
    SetExpired {
        first: MonotonicTime,
        admitted_at: MonotonicTime,
        age: Duration,
        ttl: Duration,
    },
    ServoIdMismatch {
        joint: HeadJoint,
        expected: kiko_head_protocol::ServoId,
        actual: kiko_head_protocol::ServoId,
    },
    DeviceStatus {
        joint: HeadJoint,
        raw: u8,
    },
    TelemetrySafety {
        joint: HeadJoint,
        source: HeadTelemetrySafetyViolation,
    },
    TemperatureAdmissionMismatch {
        joint: HeadJoint,
        observed_raw: u8,
        observed_at: MonotonicTime,
    },
}

impl fmt::Display for CompliantHeadObservationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "compliant head observation rejected: {self:?}")
    }
}

impl std::error::Error for CompliantHeadObservationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::TelemetrySafety { source, .. } => Some(source),
            Self::ClockRegression { .. }
            | Self::SetSpanExceeded { .. }
            | Self::AdmittedBeforeObservation { .. }
            | Self::SetExpired { .. }
            | Self::ServoIdMismatch { .. }
            | Self::DeviceStatus { .. } => None,
            Self::TemperatureAdmissionMismatch { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompliantHoldState {
    FollowingExpression,
    ConfirmingContact,
    Yielding,
    ReleaseDwell,
    Resting,
    Recovering,
    FaultHeld,
}

impl CompliantHoldState {
    pub const fn suppresses_expression_motion(self) -> bool {
        !matches!(self, Self::FollowingExpression)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompliantHoldFault {
    ObservationClockRegression {
        previous: MonotonicTime,
        actual: MonotonicTime,
    },
    ObservationOutsideEnvelope {
        joint: HeadJoint,
        observed: PositionTicks,
        minimum: PositionTicks,
        maximum: PositionTicks,
    },
    ObservationDiscontinuity {
        joint: HeadJoint,
        previous: PositionTicks,
        actual: PositionTicks,
        difference_ticks: u16,
        maximum_ticks: u16,
    },
    ApplicationUncertain,
    GenerationExhausted,
    NextServiceTimestampOverflow {
        serviced_at: MonotonicTime,
        control_period: Duration,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompliantHoldDisposition {
    FollowingExpression,
    ContactCandidate {
        consecutive_samples: u8,
    },
    Yielding {
        envelope_limited: [bool; JOINT_COUNT],
        command_step_limited: [bool; JOINT_COUNT],
    },
    ReleaseDwell,
    Resting {
        command_step_limited: [bool; JOINT_COUNT],
        at_rest_target: bool,
    },
    Recovering {
        progress_millionths: u32,
        command_step_limited: [bool; JOINT_COUNT],
    },
    ReturnedToExpression,
}

/// One semantic edge emitted by the compliant planner.
///
/// This is factual state-machine evidence. The character layer may render an
/// eye/head response from it; the compliant controller itself never chooses a
/// social expression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompliantPetEvent {
    Ready,
    Candidate,
    Contact,
    Tap,
    ReleaseStatic,
    YieldTimeout,
    Resting,
    Comfy,
    Recontact,
    Recovering,
    Returned,
}

/// Exact integer evidence for one completed contact episode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompliantPetEpisodeSummary {
    started_at: MonotonicTime,
    completed_at: MonotonicTime,
    yield_entries: u16,
    samples: u64,
    peak_residual_ticks: [u16; JOINT_COUNT],
    accumulated_max_delta_ticks: u64,
    delta_samples: u64,
    reached_rest: bool,
    reached_comfy: bool,
    tap: bool,
}

impl CompliantPetEpisodeSummary {
    pub const fn started_at(self) -> MonotonicTime {
        self.started_at
    }

    pub const fn completed_at(self) -> MonotonicTime {
        self.completed_at
    }

    pub fn duration(self) -> Duration {
        self.completed_at
            .checked_duration_since(self.started_at)
            .expect("completed pet episode timestamps are monotonic")
    }

    pub const fn yield_entries(self) -> u16 {
        self.yield_entries
    }

    pub const fn samples(self) -> u64 {
        self.samples
    }

    pub const fn peak_residual_ticks(self) -> [u16; JOINT_COUNT] {
        self.peak_residual_ticks
    }

    pub const fn accumulated_max_delta_ticks(self) -> u64 {
        self.accumulated_max_delta_ticks
    }

    pub const fn delta_samples(self) -> u64 {
        self.delta_samples
    }

    pub const fn reached_rest(self) -> bool {
        self.reached_rest
    }

    pub const fn reached_comfy(self) -> bool {
        self.reached_comfy
    }

    pub const fn was_tap(self) -> bool {
        self.tap
    }
}

impl CompliantHoldDisposition {
    pub const fn suppresses_expression_motion(self) -> bool {
        !matches!(self, Self::FollowingExpression | Self::ReturnedToExpression)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ContactCandidate {
    return_target: ExactHeadTargetPose,
    directions: [i8; JOINT_COUNT],
    consecutive_samples: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ActiveContact {
    return_target: ExactHeadTargetPose,
    directions: [i8; JOINT_COUNT],
    rest_offsets: [i16; JOINT_COUNT],
    rest_duration: Duration,
    recovery_duration: Duration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CompliantPetEpisodeAccumulator {
    started_at: MonotonicTime,
    yield_entries: u16,
    samples: u64,
    peak_residual_ticks: [u16; JOINT_COUNT],
    accumulated_max_delta_ticks: u64,
    delta_samples: u64,
    previous_residual: Option<[i32; JOINT_COUNT]>,
    reached_rest: bool,
    reached_comfy: bool,
    tap: bool,
}

impl CompliantPetEpisodeAccumulator {
    const fn new(started_at: MonotonicTime) -> Self {
        Self {
            started_at,
            yield_entries: 0,
            samples: 0,
            peak_residual_ticks: [0; JOINT_COUNT],
            accumulated_max_delta_ticks: 0,
            delta_samples: 0,
            previous_residual: None,
            reached_rest: false,
            reached_comfy: false,
            tap: false,
        }
    }

    fn observe(&mut self, residual: [i32; JOINT_COUNT]) {
        self.samples = self.samples.saturating_add(1);
        for (peak, actual) in self.peak_residual_ticks.iter_mut().zip(residual) {
            let magnitude = u16::try_from(actual.unsigned_abs()).unwrap_or(u16::MAX);
            *peak = (*peak).max(magnitude);
        }
        if let Some(previous) = self.previous_residual {
            let maximum_delta = previous
                .into_iter()
                .zip(residual)
                .map(|(previous, actual)| previous.abs_diff(actual))
                .max()
                .unwrap_or(0);
            self.accumulated_max_delta_ticks = self
                .accumulated_max_delta_ticks
                .saturating_add(u64::from(maximum_delta));
            self.delta_samples = self.delta_samples.saturating_add(1);
        }
        self.previous_residual = Some(residual);
    }

    const fn complete(self, completed_at: MonotonicTime) -> CompliantPetEpisodeSummary {
        CompliantPetEpisodeSummary {
            started_at: self.started_at,
            completed_at,
            yield_entries: self.yield_entries,
            samples: self.samples,
            peak_residual_ticks: self.peak_residual_ticks,
            accumulated_max_delta_ticks: self.accumulated_max_delta_ticks,
            delta_samples: self.delta_samples,
            reached_rest: self.reached_rest,
            reached_comfy: self.reached_comfy,
            tap: self.tap,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ControllerPhase {
    FollowingExpression {
        quiescent_since: Option<MonotonicTime>,
        contact_armed: bool,
    },
    Confirming(ContactCandidate),
    Yielding {
        contact: ActiveContact,
        quiet_since: Option<MonotonicTime>,
        entered_at: MonotonicTime,
        stable_since: MonotonicTime,
        stable_reference: Option<[i32; JOINT_COUNT]>,
    },
    Resting {
        contact: ActiveContact,
        entered_at: MonotonicTime,
        settled_at: Option<MonotonicTime>,
        previous_residual: Option<[i32; JOINT_COUNT]>,
        reacquisition: Option<ContactCandidate>,
    },
    Recovering {
        contact: ActiveContact,
        recovery_start: ExactHeadTargetPose,
        started_at: MonotonicTime,
        duration: Duration,
        previous_residual: Option<[i32; JOINT_COUNT]>,
        reacquisition: Option<ContactCandidate>,
    },
}

const fn following_expression_unarmed() -> ControllerPhase {
    ControllerPhase::FollowingExpression {
        quiescent_since: None,
        contact_armed: false,
    }
}

#[derive(Clone, Debug)]
pub struct HeadCompliantHoldController {
    instance_id: NonZeroU64,
    generation: u64,
    config: HeadCompliantHoldConfig,
    phase: ControllerPhase,
    committed_target: ExactHeadTargetPose,
    baseline_error_ticks: [i32; JOINT_COUNT],
    comfort_roll_side: Option<i8>,
    episode: Option<CompliantPetEpisodeAccumulator>,
    last_completed_episode: Option<CompliantPetEpisodeSummary>,
    previous_observation: Option<CompliantHeadObservation>,
    next_service_due: MonotonicTime,
    latest_boundary_at: MonotonicTime,
    fault: Option<CompliantHoldFault>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CompliantHoldPreparedToken {
    controller_instance: NonZeroU64,
    generation: u64,
}

#[derive(Debug)]
pub struct PreparedCompliantHoldStep {
    token: CompliantHoldPreparedToken,
    state: CompliantHoldState,
    target: ExactHeadTargetPose,
    disposition: CompliantHoldDisposition,
    desired_torque_limits: HeadTorqueLimits,
    pet_event: Option<CompliantPetEvent>,
    completed_episode: Option<CompliantPetEpisodeSummary>,
    observation: CompliantHeadObservation,
    candidate: HeadCompliantHoldController,
}

impl PreparedCompliantHoldStep {
    pub const fn token(&self) -> CompliantHoldPreparedToken {
        self.token
    }

    pub const fn state(&self) -> CompliantHoldState {
        self.state
    }

    pub const fn target(&self) -> ExactHeadTargetPose {
        self.target
    }

    pub const fn disposition(&self) -> CompliantHoldDisposition {
        self.disposition
    }

    pub const fn desired_torque_limits(&self) -> HeadTorqueLimits {
        self.desired_torque_limits
    }

    pub const fn pet_event(&self) -> Option<CompliantPetEvent> {
        self.pet_event
    }

    pub const fn completed_episode(&self) -> Option<CompliantPetEpisodeSummary> {
        self.completed_episode
    }

    pub const fn observation(&self) -> CompliantHeadObservation {
        self.observation
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompliantHoldCommitReceipt {
    token: CompliantHoldPreparedToken,
    state: CompliantHoldState,
    committed_target: ExactHeadTargetPose,
    disposition: CompliantHoldDisposition,
    desired_torque_limits: HeadTorqueLimits,
    pet_event: Option<CompliantPetEvent>,
    completed_episode: Option<CompliantPetEpisodeSummary>,
    observation: CompliantHeadObservation,
}

impl CompliantHoldCommitReceipt {
    pub const fn state(self) -> CompliantHoldState {
        self.state
    }

    pub const fn committed_target(self) -> ExactHeadTargetPose {
        self.committed_target
    }

    pub const fn disposition(self) -> CompliantHoldDisposition {
        self.disposition
    }

    pub const fn desired_torque_limits(self) -> HeadTorqueLimits {
        self.desired_torque_limits
    }

    pub const fn pet_event(self) -> Option<CompliantPetEvent> {
        self.pet_event
    }

    pub const fn completed_episode(self) -> Option<CompliantPetEpisodeSummary> {
        self.completed_episode
    }

    pub const fn observation(self) -> CompliantHeadObservation {
        self.observation
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompliantHoldPrepareError {
    FaultHeld(CompliantHoldFault),
    FaultLatched(CompliantHoldFault),
    ObservationInFuture {
        observed_at: MonotonicTime,
        serviced_at: MonotonicTime,
    },
    ObservationExpired {
        observed_at: MonotonicTime,
        serviced_at: MonotonicTime,
        age: Duration,
        ttl: Duration,
    },
    BeforeScheduledService {
        scheduled_for: MonotonicTime,
        observed_at: MonotonicTime,
    },
    ExpressionTargetOutsideEnvelope {
        joint: HeadJoint,
        target: PositionTicks,
        minimum: PositionTicks,
        maximum: PositionTicks,
    },
}

impl fmt::Display for CompliantHoldPrepareError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "compliant-hold preparation failed: {self:?}")
    }
}

impl std::error::Error for CompliantHoldPrepareError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompliantHoldCommitError {
    WrongController,
    StaleGeneration { current: u64, prepared_from: u64 },
    FutureGeneration { current: u64, prepared_from: u64 },
    FaultHeld(CompliantHoldFault),
}

impl fmt::Display for CompliantHoldCommitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "prepared compliant-hold step rejected: {self:?}")
    }
}

impl std::error::Error for CompliantHoldCommitError {}

impl HeadCompliantHoldController {
    pub fn try_new(
        config: HeadCompliantHoldConfig,
        initial_committed_target: ExactHeadTargetPose,
        started_at: MonotonicTime,
    ) -> Result<Self, CompliantHoldPrepareError> {
        admit_target(config, initial_committed_target)?;
        let raw = NEXT_COMPLIANT_CONTROLLER_ID.fetch_add(1, Ordering::Relaxed);
        let instance_id = NonZeroU64::new(raw).ok_or({
            CompliantHoldPrepareError::FaultLatched(CompliantHoldFault::GenerationExhausted)
        })?;
        Ok(Self {
            instance_id,
            generation: 0,
            config,
            phase: ControllerPhase::FollowingExpression {
                quiescent_since: None,
                contact_armed: false,
            },
            committed_target: initial_committed_target,
            baseline_error_ticks: [0; JOINT_COUNT],
            comfort_roll_side: None,
            episode: None,
            last_completed_episode: None,
            previous_observation: None,
            next_service_due: started_at,
            latest_boundary_at: started_at,
            fault: None,
        })
    }

    pub const fn config(&self) -> HeadCompliantHoldConfig {
        self.config
    }

    pub const fn committed_target(&self) -> ExactHeadTargetPose {
        self.committed_target
    }

    pub const fn state(&self) -> CompliantHoldState {
        match (self.fault, self.phase) {
            (Some(_), _) => CompliantHoldState::FaultHeld,
            (None, ControllerPhase::FollowingExpression { .. }) => {
                CompliantHoldState::FollowingExpression
            }
            (None, ControllerPhase::Confirming(_)) => CompliantHoldState::ConfirmingContact,
            (
                None,
                ControllerPhase::Yielding {
                    quiet_since: None, ..
                },
            ) => CompliantHoldState::Yielding,
            (
                None,
                ControllerPhase::Yielding {
                    quiet_since: Some(_),
                    ..
                },
            ) => CompliantHoldState::ReleaseDwell,
            (None, ControllerPhase::Resting { .. }) => CompliantHoldState::Resting,
            (None, ControllerPhase::Recovering { .. }) => CompliantHoldState::Recovering,
        }
    }

    pub const fn fault(&self) -> Option<CompliantHoldFault> {
        self.fault
    }

    pub const fn next_service_due(&self) -> MonotonicTime {
        self.next_service_due
    }

    pub const fn baseline_error_ticks(&self) -> [i32; JOINT_COUNT] {
        self.baseline_error_ticks
    }

    pub const fn last_completed_episode(&self) -> Option<CompliantPetEpisodeSummary> {
        self.last_completed_episode
    }

    /// Torque profile that must be installed for the currently committed
    /// planner state.
    pub fn desired_torque_limits(&self) -> HeadTorqueLimits {
        desired_torque_limits(self.config, self.phase)
    }

    /// Prepare one observation-driven command.
    ///
    /// `expression_quiet` must be true only when the lower-priority gaze or
    /// character planner has zero commanded velocity and has retained its
    /// target. This prevents normal commanded motion from being misclassified
    /// as touch. Once contact is acquired, the captured return target remains
    /// authoritative until recovery completes.
    pub fn prepare(
        &mut self,
        serviced_at: MonotonicTime,
        expression_target: ExactHeadTargetPose,
        expression_quiet: bool,
        observation: CompliantHeadObservation,
    ) -> Result<PreparedCompliantHoldStep, CompliantHoldPrepareError> {
        if let Some(fault) = self.fault {
            return Err(CompliantHoldPrepareError::FaultHeld(fault));
        }
        if serviced_at < self.next_service_due {
            return Err(CompliantHoldPrepareError::BeforeScheduledService {
                scheduled_for: self.next_service_due,
                observed_at: serviced_at,
            });
        }
        admit_target(self.config, expression_target)?;
        self.admit_observation_time(serviced_at, observation)?;

        let Some(next_generation) = self.generation.checked_add(1) else {
            let fault = CompliantHoldFault::GenerationExhausted;
            self.fault = Some(fault);
            return Err(CompliantHoldPrepareError::FaultLatched(fault));
        };
        let token = CompliantHoldPreparedToken {
            controller_instance: self.instance_id,
            generation: self.generation,
        };
        let mut candidate = self.clone();
        let previously_completed_episode = candidate.last_completed_episode;
        let (disposition, pet_event) = candidate.advance(
            serviced_at,
            expression_target,
            expression_quiet,
            observation,
        )?;
        candidate.generation = next_generation;
        candidate.previous_observation = Some(observation);
        candidate.latest_boundary_at = serviced_at;
        candidate.next_service_due = checked_time_add(serviced_at, self.config.control_period())
            .ok_or_else(|| {
                let fault = CompliantHoldFault::NextServiceTimestampOverflow {
                    serviced_at,
                    control_period: self.config.control_period(),
                };
                self.fault = Some(fault);
                CompliantHoldPrepareError::FaultLatched(fault)
            })?;
        let target = candidate.committed_target;
        let state = candidate.state();
        let desired_torque_limits = candidate.desired_torque_limits();
        let completed_episode = (candidate.last_completed_episode != previously_completed_episode)
            .then_some(candidate.last_completed_episode)
            .flatten();
        Ok(PreparedCompliantHoldStep {
            token,
            state,
            target,
            disposition,
            desired_torque_limits,
            pet_event,
            completed_episode,
            observation,
            candidate,
        })
    }

    pub fn commit(
        &mut self,
        prepared: PreparedCompliantHoldStep,
    ) -> Result<CompliantHoldCommitReceipt, CompliantHoldCommitError> {
        self.validate_token(prepared.token)?;
        let PreparedCompliantHoldStep {
            token,
            state,
            target,
            disposition,
            desired_torque_limits,
            pet_event,
            completed_episode,
            observation,
            candidate,
        } = prepared;
        *self = candidate;
        Ok(CompliantHoldCommitReceipt {
            token,
            state,
            committed_target: target,
            disposition,
            desired_torque_limits,
            pet_event,
            completed_episode,
            observation,
        })
    }

    pub fn abort_with_application_uncertain(
        &mut self,
        prepared: PreparedCompliantHoldStep,
    ) -> Result<(), CompliantHoldCommitError> {
        self.validate_token(prepared.token)?;
        self.fault = Some(CompliantHoldFault::ApplicationUncertain);
        Ok(())
    }

    fn validate_token(
        &self,
        token: CompliantHoldPreparedToken,
    ) -> Result<(), CompliantHoldCommitError> {
        if token.controller_instance != self.instance_id {
            return Err(CompliantHoldCommitError::WrongController);
        }
        if let Some(fault) = self.fault {
            return Err(CompliantHoldCommitError::FaultHeld(fault));
        }
        if token.generation < self.generation {
            return Err(CompliantHoldCommitError::StaleGeneration {
                current: self.generation,
                prepared_from: token.generation,
            });
        }
        if token.generation > self.generation {
            return Err(CompliantHoldCommitError::FutureGeneration {
                current: self.generation,
                prepared_from: token.generation,
            });
        }
        Ok(())
    }

    fn admit_observation_time(
        &mut self,
        serviced_at: MonotonicTime,
        observation: CompliantHeadObservation,
    ) -> Result<(), CompliantHoldPrepareError> {
        if serviced_at < self.latest_boundary_at {
            let fault = CompliantHoldFault::ObservationClockRegression {
                previous: self.latest_boundary_at,
                actual: serviced_at,
            };
            self.fault = Some(fault);
            return Err(CompliantHoldPrepareError::FaultLatched(fault));
        }
        let Some(age) = serviced_at.checked_duration_since(observation.observed_at()) else {
            return Err(CompliantHoldPrepareError::ObservationInFuture {
                observed_at: observation.observed_at(),
                serviced_at,
            });
        };
        if age >= self.config.observation_ttl() {
            return Err(CompliantHoldPrepareError::ObservationExpired {
                observed_at: observation.observed_at(),
                serviced_at,
                age,
                ttl: self.config.observation_ttl(),
            });
        }
        for joint in HeadJoint::ALL {
            let policy = self.config.joint(joint);
            let observed = observation.position(joint);
            if !policy.contains_observation(observed) {
                let fault = CompliantHoldFault::ObservationOutsideEnvelope {
                    joint,
                    observed,
                    minimum: policy.observation_minimum(),
                    maximum: policy.observation_maximum(),
                };
                self.fault = Some(fault);
                return Err(CompliantHoldPrepareError::FaultLatched(fault));
            }
            if let Some(previous) = self.previous_observation {
                let previous = previous.position(joint);
                let difference_ticks = previous.get().abs_diff(observed.get());
                if difference_ticks > policy.maximum_observed_step_ticks() {
                    let fault = CompliantHoldFault::ObservationDiscontinuity {
                        joint,
                        previous,
                        actual: observed,
                        difference_ticks,
                        maximum_ticks: policy.maximum_observed_step_ticks(),
                    };
                    self.fault = Some(fault);
                    return Err(CompliantHoldPrepareError::FaultLatched(fault));
                }
            }
        }
        Ok(())
    }

    fn advance(
        &mut self,
        now: MonotonicTime,
        expression_target: ExactHeadTargetPose,
        expression_quiet: bool,
        observation: CompliantHeadObservation,
    ) -> Result<(CompliantHoldDisposition, Option<CompliantPetEvent>), CompliantHoldPrepareError>
    {
        match self.phase {
            ControllerPhase::FollowingExpression {
                quiescent_since,
                contact_armed,
            } => {
                let expression_target_changed = expression_target != self.committed_target;
                self.committed_target = expression_target;
                if !expression_quiet || expression_target_changed {
                    self.reset_to_following();
                    return Ok((CompliantHoldDisposition::FollowingExpression, None));
                }
                if !contact_armed {
                    let candidate_baseline = HeadJoint::ALL.map(|joint| {
                        signed_error(
                            observation.position(joint),
                            self.committed_target.position(joint),
                        )
                    });
                    let settled = if let Some(profile) = self.config.pet_profile() {
                        HeadJoint::ALL.into_iter().all(|joint| {
                            !observation.is_moving(joint)
                                && candidate_baseline[joint_index(joint)].unsigned_abs()
                                    <= u32::from(
                                        profile.joint(joint).maximum_baseline_error_ticks(),
                                    )
                        })
                    } else {
                        inside_release_band(
                            self.config,
                            self.committed_target,
                            observation,
                            self.baseline_error_ticks,
                        ) && HeadJoint::ALL
                            .into_iter()
                            .all(|joint| !observation.is_moving(joint))
                    };
                    let quiescent_since = if settled {
                        quiescent_since.or(Some(now))
                    } else {
                        None
                    };
                    self.baseline_error_ticks = if settled && self.config.pet_profile().is_some() {
                        candidate_baseline
                    } else {
                        [0; JOINT_COUNT]
                    };
                    let contact_armed = quiescent_since.is_some_and(|started| {
                        now.checked_duration_since(started)
                            .is_some_and(|elapsed| elapsed >= self.config.contact_arm_dwell())
                    });
                    self.phase = ControllerPhase::FollowingExpression {
                        quiescent_since,
                        contact_armed,
                    };
                    let event = contact_armed
                        .then_some(CompliantPetEvent::Ready)
                        .filter(|_| self.config.pet_profile().is_some());
                    return Ok((CompliantHoldDisposition::FollowingExpression, event));
                }
                let directions = contact_directions(
                    self.config,
                    self.committed_target,
                    observation,
                    self.baseline_error_ticks,
                );
                if directions == [0; JOINT_COUNT] {
                    return Ok((CompliantHoldDisposition::FollowingExpression, None));
                }
                let candidate = ContactCandidate {
                    return_target: self.committed_target,
                    directions,
                    consecutive_samples: 1,
                };
                if self.config.contact_acquisition_samples() == 1 {
                    return self.enter_yield(now, candidate, observation, false);
                }
                self.phase = ControllerPhase::Confirming(candidate);
                Ok((
                    CompliantHoldDisposition::ContactCandidate {
                        consecutive_samples: 1,
                    },
                    self.config
                        .pet_profile()
                        .map(|_| CompliantPetEvent::Candidate),
                ))
            }
            ControllerPhase::Confirming(candidate) => {
                if !expression_quiet || expression_target != candidate.return_target {
                    self.committed_target = expression_target;
                    self.reset_to_following();
                    return Ok((CompliantHoldDisposition::FollowingExpression, None));
                }
                let directions = contact_directions(
                    self.config,
                    self.committed_target,
                    observation,
                    self.baseline_error_ticks,
                );
                if !directions_continue(candidate.directions, directions) {
                    self.reset_to_following();
                    return Ok((CompliantHoldDisposition::FollowingExpression, None));
                }
                let consecutive_samples = candidate.consecutive_samples.saturating_add(1);
                let directions = merge_directions(candidate.directions, directions);
                if consecutive_samples >= self.config.contact_acquisition_samples() {
                    return self.enter_yield(
                        now,
                        ContactCandidate {
                            directions,
                            consecutive_samples,
                            ..candidate
                        },
                        observation,
                        false,
                    );
                }
                self.phase = ControllerPhase::Confirming(ContactCandidate {
                    directions,
                    consecutive_samples,
                    ..candidate
                });
                Ok((
                    CompliantHoldDisposition::ContactCandidate {
                        consecutive_samples,
                    },
                    None,
                ))
            }
            ControllerPhase::Yielding {
                contact,
                quiet_since,
                entered_at,
                stable_since,
                stable_reference,
            } => {
                let residual = residual_errors(
                    self.committed_target,
                    observation,
                    self.baseline_error_ticks,
                );
                if let Some(episode) = self.episode.as_mut() {
                    episode.observe(residual);
                }
                let released = inside_release_band(
                    self.config,
                    self.committed_target,
                    observation,
                    self.baseline_error_ticks,
                ) && HeadJoint::ALL
                    .into_iter()
                    .all(|joint| !observation.is_moving(joint));
                let quiet_since = if released {
                    quiet_since.or(Some(now))
                } else {
                    None
                };
                if released {
                    if let Some(profile) = self.config.pet_profile()
                        && self.episode.is_some_and(|episode| {
                            episode.yield_entries == 1
                                && !episode.reached_rest
                                && now
                                    .checked_duration_since(entered_at)
                                    .is_some_and(|elapsed| {
                                        elapsed <= profile.tap_maximum_contact_duration()
                                    })
                        })
                    {
                        if let Some(episode) = self.episode.as_mut() {
                            episode.tap = true;
                        }
                        self.enter_recovery(contact, now, profile.tap_recovery_duration());
                        return Ok((
                            CompliantHoldDisposition::Recovering {
                                progress_millionths: 0,
                                command_step_limited: [false; JOINT_COUNT],
                            },
                            Some(CompliantPetEvent::Tap),
                        ));
                    }
                    if quiet_since.is_some_and(|started| {
                        now.checked_duration_since(started)
                            .is_some_and(|elapsed| elapsed >= self.config.release_dwell())
                    }) {
                        if self.config.pet_profile().is_some() {
                            self.enter_rest(contact, now);
                            return Ok((
                                CompliantHoldDisposition::Resting {
                                    command_step_limited: [false; JOINT_COUNT],
                                    at_rest_target: false,
                                },
                                Some(CompliantPetEvent::Resting),
                            ));
                        }
                        self.enter_recovery(contact, now, self.config.recovery_duration());
                        return Ok((
                            CompliantHoldDisposition::Recovering {
                                progress_millionths: 0,
                                command_step_limited: [false; JOINT_COUNT],
                            },
                            None,
                        ));
                    }
                    self.phase = ControllerPhase::Yielding {
                        contact,
                        quiet_since,
                        entered_at,
                        stable_since: now,
                        stable_reference: None,
                    };
                    return Ok((CompliantHoldDisposition::ReleaseDwell, None));
                }

                if let Some(profile) = self.config.pet_profile() {
                    let stable = stable_reference.is_some_and(|reference| {
                        residual
                            .into_iter()
                            .zip(reference)
                            .all(|(actual, reference)| {
                                actual.abs_diff(reference)
                                    <= u32::from(profile.residual_stillness_ticks())
                            })
                    });
                    let (stable_since, stable_reference) = if stable {
                        (stable_since, stable_reference)
                    } else {
                        (now, Some(residual))
                    };
                    let static_release = now
                        .checked_duration_since(stable_since)
                        .is_some_and(|elapsed| elapsed >= profile.static_release_dwell());
                    let timed_out = now
                        .checked_duration_since(entered_at)
                        .is_some_and(|elapsed| elapsed >= profile.maximum_yield_dwell());
                    if static_release || timed_out {
                        self.enter_rest(contact, now);
                        return Ok((
                            CompliantHoldDisposition::Resting {
                                command_step_limited: [false; JOINT_COUNT],
                                at_rest_target: false,
                            },
                            Some(if static_release {
                                CompliantPetEvent::ReleaseStatic
                            } else {
                                CompliantPetEvent::YieldTimeout
                            }),
                        ));
                    }
                    self.phase = ControllerPhase::Yielding {
                        contact,
                        quiet_since: None,
                        entered_at,
                        stable_since,
                        stable_reference,
                    };
                } else {
                    self.phase = ControllerPhase::Yielding {
                        contact,
                        quiet_since: None,
                        entered_at,
                        stable_since,
                        stable_reference,
                    };
                }
                let (target, envelope_limited, command_step_limited) = yield_target(
                    self.config,
                    contact,
                    self.committed_target,
                    observation,
                    self.baseline_error_ticks,
                );
                self.committed_target = target;
                Ok((
                    CompliantHoldDisposition::Yielding {
                        envelope_limited,
                        command_step_limited,
                    },
                    None,
                ))
            }
            ControllerPhase::Resting {
                contact,
                entered_at,
                settled_at,
                previous_residual,
                reacquisition,
            } => {
                let (confirmed, previous_residual, reacquisition) =
                    self.recontact(contact, observation, previous_residual, reacquisition);
                if let Some(candidate) = confirmed {
                    return self.enter_yield(now, candidate, observation, true);
                }
                let desired = rest_target(self.config, contact);
                let (target, command_step_limited) =
                    command_step_target(self.config, self.committed_target, desired);
                self.committed_target = target;
                let reached_now = settled_at.is_none() && target == desired;
                let settled_at = if reached_now { Some(now) } else { settled_at };
                if reached_now && let Some(episode) = self.episode.as_mut() {
                    episode.reached_comfy = true;
                }
                let settled = settled_at.is_some_and(|settled_at| {
                    now.checked_duration_since(settled_at)
                        .is_some_and(|elapsed| elapsed >= contact.rest_duration)
                });
                let overdue = now
                    .checked_duration_since(entered_at)
                    .is_some_and(|elapsed| {
                        elapsed
                            >= self
                                .config
                                .pet_profile()
                                .expect("Resting exists only with a pet profile")
                                .maximum_rest_dwell()
                                .checked_add(self.config.maximum_rest_travel_duration())
                                .expect("pet profile binding rejects timing overflow")
                    });
                if settled || overdue {
                    self.enter_recovery(contact, now, contact.recovery_duration);
                    return Ok((
                        CompliantHoldDisposition::Recovering {
                            progress_millionths: 0,
                            command_step_limited: [false; JOINT_COUNT],
                        },
                        Some(CompliantPetEvent::Recovering),
                    ));
                }
                self.phase = ControllerPhase::Resting {
                    contact,
                    entered_at,
                    settled_at,
                    previous_residual,
                    reacquisition,
                };
                Ok((
                    CompliantHoldDisposition::Resting {
                        command_step_limited,
                        at_rest_target: target == desired,
                    },
                    reached_now.then_some(CompliantPetEvent::Comfy),
                ))
            }
            ControllerPhase::Recovering {
                contact,
                recovery_start,
                started_at,
                duration,
                previous_residual,
                reacquisition,
            } => {
                let (confirmed, previous_residual, reacquisition) =
                    if self.config.pet_profile().is_some() {
                        self.recontact(contact, observation, previous_residual, reacquisition)
                    } else {
                        let directions = contact_directions(
                            self.config,
                            self.committed_target,
                            observation,
                            self.baseline_error_ticks,
                        );
                        let reacquisition =
                            advance_reacquisition(contact.return_target, reacquisition, directions);
                        let confirmed = reacquisition.filter(|candidate| {
                            candidate.consecutive_samples
                                >= self.config.contact_acquisition_samples()
                        });
                        (confirmed, previous_residual, reacquisition)
                    };
                if let Some(candidate) = confirmed {
                    return self.enter_yield(now, candidate, observation, true);
                }
                let elapsed = now
                    .checked_duration_since(started_at)
                    .expect("observation time admission prevents clock regression");
                let progress = minimum_jerk_progress(elapsed, duration);
                let desired = interpolate_pose(recovery_start, contact.return_target, progress);
                let (target, command_step_limited) =
                    command_step_target(self.config, self.committed_target, desired);
                self.committed_target = target;
                if elapsed >= duration && target == contact.return_target {
                    self.phase = following_expression_unarmed();
                    self.baseline_error_ticks = [0; JOINT_COUNT];
                    self.comfort_roll_side = None;
                    if let Some(episode) = self.episode.take() {
                        let summary = episode.complete(now);
                        self.last_completed_episode = Some(summary);
                    }
                    return Ok((
                        CompliantHoldDisposition::ReturnedToExpression,
                        self.config
                            .pet_profile()
                            .map(|_| CompliantPetEvent::Returned),
                    ));
                }
                self.phase = ControllerPhase::Recovering {
                    contact,
                    recovery_start,
                    started_at,
                    duration,
                    previous_residual,
                    reacquisition,
                };
                Ok((
                    CompliantHoldDisposition::Recovering {
                        progress_millionths: u32::try_from(progress)
                            .expect("fixed interpolation scale fits u32"),
                        command_step_limited,
                    },
                    None,
                ))
            }
        }
    }

    fn enter_yield(
        &mut self,
        now: MonotonicTime,
        candidate: ContactCandidate,
        observation: CompliantHeadObservation,
        recontact: bool,
    ) -> Result<(CompliantHoldDisposition, Option<CompliantPetEvent>), CompliantHoldPrepareError>
    {
        let contact =
            self.active_contact(candidate.return_target, candidate.directions, observation);
        let (target, envelope_limited, command_step_limited) = yield_target(
            self.config,
            contact,
            self.committed_target,
            observation,
            self.baseline_error_ticks,
        );
        self.committed_target = target;
        self.phase = ControllerPhase::Yielding {
            contact,
            quiet_since: None,
            entered_at: now,
            stable_since: now,
            stable_reference: None,
        };
        if self.config.pet_profile().is_some() {
            let episode = self
                .episode
                .get_or_insert_with(|| CompliantPetEpisodeAccumulator::new(now));
            episode.yield_entries = episode.yield_entries.saturating_add(1);
            if recontact {
                episode.tap = false;
            }
        }
        Ok((
            CompliantHoldDisposition::Yielding {
                envelope_limited,
                command_step_limited,
            },
            self.config.pet_profile().map(|_| {
                if recontact {
                    CompliantPetEvent::Recontact
                } else {
                    CompliantPetEvent::Contact
                }
            }),
        ))
    }

    fn active_contact(
        &mut self,
        return_target: ExactHeadTargetPose,
        directions: [i8; JOINT_COUNT],
        observation: CompliantHeadObservation,
    ) -> ActiveContact {
        let Some(profile) = self.config.pet_profile() else {
            return ActiveContact {
                return_target,
                directions,
                rest_offsets: [0; JOINT_COUNT],
                rest_duration: Duration::ZERO,
                recovery_duration: self.config.recovery_duration(),
            };
        };
        let active_joints = directions
            .into_iter()
            .filter(|direction| *direction != 0)
            .count()
            .max(1);
        let additional_joints = u32::try_from(active_joints - 1).expect("four joints fit u32");
        let mut rest_offsets = HeadJoint::ALL.map(|joint| {
            let pet = profile.joint(joint);
            let directional = i32::from(directions[joint_index(joint)])
                * i32::from(pet.directional_rest_offset_ticks());
            i16::try_from(i32::from(pet.rest_offset_ticks()) + directional)
                .expect("bound pet rest offsets fit i16")
        });
        if directions[joint_index(HeadJoint::Roll)] == 0 && profile.comfort_roll_tilt_ticks() > 0 {
            let residual = residual_errors(
                self.committed_target,
                observation,
                self.baseline_error_ticks,
            );
            let roll = residual[joint_index(HeadJoint::Roll)];
            let yaw = residual[joint_index(HeadJoint::Yaw)];
            let hint = if roll.unsigned_abs() >= 3 { roll } else { yaw };
            let side = *self
                .comfort_roll_side
                .get_or_insert(if hint < 0 { -1 } else { 1 });
            let roll_index = joint_index(HeadJoint::Roll);
            rest_offsets[roll_index] = i16::try_from(
                i32::from(rest_offsets[roll_index])
                    + i32::from(side) * i32::from(profile.comfort_roll_tilt_ticks()),
            )
            .expect("bound comfort roll offset fits i16");
        }
        let additional_rest = profile
            .rest_per_additional_joint()
            .checked_mul(additional_joints)
            .expect("pet profile binding rejects timing overflow");
        let rest_duration = profile
            .rest_dwell()
            .checked_add(additional_rest)
            .expect("pet profile binding rejects timing overflow")
            .min(profile.maximum_rest_dwell());
        let recovery_scale =
            1_000 + additional_joints * u32::from(profile.recovery_per_additional_joint_permille());
        let recovery_duration =
            scale_duration_permille(self.config.recovery_duration(), recovery_scale)
                .expect("pet profile binding rejects timing overflow");
        ActiveContact {
            return_target,
            directions,
            rest_offsets,
            rest_duration,
            recovery_duration,
        }
    }

    fn enter_rest(&mut self, contact: ActiveContact, now: MonotonicTime) {
        if let Some(episode) = self.episode.as_mut() {
            episode.reached_rest = true;
        }
        self.phase = ControllerPhase::Resting {
            contact,
            entered_at: now,
            settled_at: None,
            previous_residual: None,
            reacquisition: None,
        };
    }

    fn enter_recovery(&mut self, contact: ActiveContact, now: MonotonicTime, duration: Duration) {
        self.phase = ControllerPhase::Recovering {
            contact,
            recovery_start: self.committed_target,
            started_at: now,
            duration,
            previous_residual: None,
            reacquisition: None,
        };
    }

    fn recontact(
        &self,
        contact: ActiveContact,
        observation: CompliantHeadObservation,
        previous_residual: Option<[i32; JOINT_COUNT]>,
        reacquisition: Option<ContactCandidate>,
    ) -> (
        Option<ContactCandidate>,
        Option<[i32; JOINT_COUNT]>,
        Option<ContactCandidate>,
    ) {
        let profile = self
            .config
            .pet_profile()
            .expect("stillness-gated recontact requires a pet profile");
        let directions = contact_directions(
            self.config,
            self.committed_target,
            observation,
            self.baseline_error_ticks,
        );
        let residual = residual_errors(
            self.committed_target,
            observation,
            self.baseline_error_ticks,
        );
        let varying = previous_residual.is_some_and(|previous| {
            residual
                .into_iter()
                .zip(previous)
                .any(|(actual, previous)| {
                    actual.abs_diff(previous) > u32::from(profile.residual_stillness_ticks())
                })
        });
        if directions != [0; JOINT_COUNT] && !varying {
            return (None, Some(residual), reacquisition);
        }
        let reacquisition = advance_reacquisition(contact.return_target, reacquisition, directions);
        let confirmed = reacquisition.filter(|candidate| {
            candidate.consecutive_samples >= self.config.contact_acquisition_samples()
        });
        (confirmed, Some(residual), reacquisition)
    }

    fn reset_to_following(&mut self) {
        self.phase = following_expression_unarmed();
        self.baseline_error_ticks = [0; JOINT_COUNT];
        self.comfort_roll_side = None;
        self.episode = None;
    }
}

fn admit_target(
    config: HeadCompliantHoldConfig,
    target: ExactHeadTargetPose,
) -> Result<(), CompliantHoldPrepareError> {
    for joint in HeadJoint::ALL {
        let policy = config.joint(joint);
        let position = target.position(joint);
        if !policy.contains(position) {
            return Err(CompliantHoldPrepareError::ExpressionTargetOutsideEnvelope {
                joint,
                target: position,
                minimum: policy.minimum(),
                maximum: policy.maximum(),
            });
        }
    }
    Ok(())
}

fn desired_torque_limits(
    config: HeadCompliantHoldConfig,
    phase: ControllerPhase,
) -> HeadTorqueLimits {
    let Some(profile) = config.pet_profile() else {
        return config.holding_torque_limits();
    };
    let ControllerPhase::Yielding { contact, .. } = phase else {
        return config.holding_torque_limits();
    };
    let for_joint = |joint: HeadJoint| {
        let touched = contact.directions[joint_index(joint)] != 0;
        if touched || matches!(joint, HeadJoint::Yaw | HeadJoint::Roll) {
            profile.yield_torque_limits().for_joint(joint)
        } else {
            config.holding_torque_limits().for_joint(joint)
        }
    };
    HeadTorqueLimits::new(
        for_joint(HeadJoint::Bow),
        for_joint(HeadJoint::Curl),
        for_joint(HeadJoint::Yaw),
        for_joint(HeadJoint::Roll),
    )
}

fn signed_error(actual: PositionTicks, reference: PositionTicks) -> i32 {
    i32::from(actual.get()) - i32::from(reference.get())
}

fn contact_directions(
    config: HeadCompliantHoldConfig,
    reference: ExactHeadTargetPose,
    observation: CompliantHeadObservation,
    baseline_error_ticks: [i32; JOINT_COUNT],
) -> [i8; JOINT_COUNT] {
    HeadJoint::ALL.map(|joint| {
        let error = signed_error(observation.position(joint), reference.position(joint))
            - baseline_error_ticks[joint_index(joint)];
        if error.unsigned_abs() >= u32::from(config.joint(joint).contact_entry_error_ticks()) {
            error.signum() as i8
        } else {
            0
        }
    })
}

fn residual_errors(
    command: ExactHeadTargetPose,
    observation: CompliantHeadObservation,
    baseline_error_ticks: [i32; JOINT_COUNT],
) -> [i32; JOINT_COUNT] {
    HeadJoint::ALL.map(|joint| {
        signed_error(observation.position(joint), command.position(joint))
            - baseline_error_ticks[joint_index(joint)]
    })
}

fn directions_continue(previous: [i8; JOINT_COUNT], actual: [i8; JOINT_COUNT]) -> bool {
    previous
        .into_iter()
        .zip(actual)
        .any(|(previous, actual)| previous != 0 && previous == actual)
        && previous
            .into_iter()
            .zip(actual)
            .all(|(previous, actual)| previous == 0 || actual == 0 || previous == actual)
}

fn merge_directions(previous: [i8; JOINT_COUNT], actual: [i8; JOINT_COUNT]) -> [i8; JOINT_COUNT] {
    std::array::from_fn(|index| {
        if previous[index] == 0 {
            actual[index]
        } else {
            previous[index]
        }
    })
}

fn advance_reacquisition(
    return_target: ExactHeadTargetPose,
    previous: Option<ContactCandidate>,
    directions: [i8; JOINT_COUNT],
) -> Option<ContactCandidate> {
    if directions == [0; JOINT_COUNT] {
        return None;
    }
    match previous {
        Some(candidate) if directions_continue(candidate.directions, directions) => {
            Some(ContactCandidate {
                directions: merge_directions(candidate.directions, directions),
                consecutive_samples: candidate.consecutive_samples.saturating_add(1),
                ..candidate
            })
        }
        _ => Some(ContactCandidate {
            return_target,
            directions,
            consecutive_samples: 1,
        }),
    }
}

fn inside_release_band(
    config: HeadCompliantHoldConfig,
    command: ExactHeadTargetPose,
    observation: CompliantHeadObservation,
    baseline_error_ticks: [i32; JOINT_COUNT],
) -> bool {
    residual_errors(command, observation, baseline_error_ticks)
        .into_iter()
        .zip(HeadJoint::ALL)
        .all(|(error, joint)| {
            error.unsigned_abs() <= u32::from(config.joint(joint).contact_release_error_ticks())
        })
}

fn yield_target(
    config: HeadCompliantHoldConfig,
    contact: ActiveContact,
    current_target: ExactHeadTargetPose,
    observation: CompliantHeadObservation,
    baseline_error_ticks: [i32; JOINT_COUNT],
) -> (
    ExactHeadTargetPose,
    [bool; JOINT_COUNT],
    [bool; JOINT_COUNT],
) {
    let mut positions = contact.return_target.positions();
    let mut limited = [false; JOINT_COUNT];
    for joint in HeadJoint::ALL {
        let index = joint_index(joint);
        let policy = config.joint(joint);
        let rest = i64::from(contact.rest_offsets[index]);
        let displacement_from_rest = i64::from(signed_error(
            observation.position(joint),
            contact.return_target.position(joint),
        )) - i64::from(baseline_error_ticks[index])
            - rest;
        let scaled = div_round_nearest(
            displacement_from_rest * i64::from(config.follow_permille()),
            1_000,
        );
        let maximum_yield = i64::from(policy.maximum_yield_ticks());
        let requested_offset = rest + scaled;
        let yield_limited = requested_offset.clamp(-maximum_yield, maximum_yield);
        limited[index] |= yield_limited != requested_offset;
        let raw = i64::from(contact.return_target.position(joint).get()) + yield_limited;
        let envelope_limited = raw.clamp(
            i64::from(policy.minimum().get()),
            i64::from(policy.maximum().get()),
        );
        limited[index] |= envelope_limited != raw;
        positions[index] = PositionTicks::try_new(
            u16::try_from(envelope_limited).expect("admitted encoder envelope fits u16"),
        )
        .expect("admitted encoder envelope is inside protocol range");
    }
    let desired =
        ExactHeadTargetPose::from_positions(positions[0], positions[1], positions[2], positions[3]);
    let (target, command_step_limited) = command_step_target(config, current_target, desired);
    (target, limited, command_step_limited)
}

fn rest_target(config: HeadCompliantHoldConfig, contact: ActiveContact) -> ExactHeadTargetPose {
    let positions = HeadJoint::ALL.map(|joint| {
        let index = joint_index(joint);
        let policy = config.joint(joint);
        let raw = i32::from(contact.return_target.position(joint).get())
            + i32::from(contact.rest_offsets[index]);
        let admitted = raw.clamp(
            i32::from(policy.minimum().get()),
            i32::from(policy.maximum().get()),
        );
        PositionTicks::try_new(u16::try_from(admitted).expect("admitted rest target fits u16"))
            .expect("admitted rest target is inside encoder domain")
    });
    ExactHeadTargetPose::from_positions(positions[0], positions[1], positions[2], positions[3])
}

fn command_step_target(
    config: HeadCompliantHoldConfig,
    current: ExactHeadTargetPose,
    desired: ExactHeadTargetPose,
) -> (ExactHeadTargetPose, [bool; JOINT_COUNT]) {
    let mut positions = current.positions();
    let mut limited = [false; JOINT_COUNT];
    for joint in HeadJoint::ALL {
        let index = joint_index(joint);
        let current = i64::from(current.position(joint).get());
        let desired = i64::from(desired.position(joint).get());
        let maximum = i64::from(config.joint(joint).maximum_command_step_ticks());
        let delta = desired - current;
        let bounded = delta.clamp(-maximum, maximum);
        limited[index] = bounded != delta;
        positions[index] = PositionTicks::try_new(
            u16::try_from(current + bounded)
                .expect("bounded step between admitted positions stays inside u16"),
        )
        .expect("bounded step between admitted positions stays in encoder domain");
    }
    (
        ExactHeadTargetPose::from_positions(positions[0], positions[1], positions[2], positions[3]),
        limited,
    )
}

fn div_round_nearest(numerator: i64, denominator: i64) -> i64 {
    debug_assert!(denominator > 0);
    if numerator >= 0 {
        (numerator + denominator / 2) / denominator
    } else {
        (numerator - denominator / 2) / denominator
    }
}

fn checked_time_add(time: MonotonicTime, duration: Duration) -> Option<MonotonicTime> {
    time.duration_since_origin()
        .checked_add(duration)
        .map(MonotonicTime::from_duration_since_origin)
}

fn scale_duration_permille(duration: Duration, permille: u32) -> Option<Duration> {
    let nanoseconds = duration
        .as_nanos()
        .checked_mul(u128::from(permille))?
        .checked_add(500)?
        / 1_000;
    let seconds = nanoseconds / 1_000_000_000;
    let subsecond_nanoseconds = nanoseconds % 1_000_000_000;
    Some(Duration::new(
        u64::try_from(seconds).ok()?,
        u32::try_from(subsecond_nanoseconds).ok()?,
    ))
}

/// Quintic minimum-jerk progress in fixed millionths.
fn minimum_jerk_progress(elapsed: Duration, total: Duration) -> u128 {
    if elapsed >= total {
        return INTERPOLATION_SCALE;
    }
    let elapsed_ns = elapsed.as_nanos();
    let total_ns = total.as_nanos();
    let u = elapsed_ns.saturating_mul(INTERPOLATION_SCALE) / total_ns;
    let u2 = u * u;
    let u3 = u2 * u;
    let u4 = u3 * u;
    let u5 = u4 * u;
    let scale2 = INTERPOLATION_SCALE * INTERPOLATION_SCALE;
    let scale4 = scale2 * scale2;
    (6 * u5 + 10 * u3 * scale2 - 15 * u4 * INTERPOLATION_SCALE) / scale4
}

fn interpolate_pose(
    start: ExactHeadTargetPose,
    end: ExactHeadTargetPose,
    progress_millionths: u128,
) -> ExactHeadTargetPose {
    let positions = HeadJoint::ALL.map(|joint| {
        let start = i64::from(start.position(joint).get());
        let delta = i64::from(end.position(joint).get()) - start;
        let scaled = div_round_nearest(
            delta * i64::try_from(progress_millionths).expect("progress fits i64"),
            i64::try_from(INTERPOLATION_SCALE).expect("scale fits i64"),
        );
        PositionTicks::try_new(
            u16::try_from(start + scaled).expect("interpolation stays between admitted endpoints"),
        )
        .expect("interpolation stays inside protocol encoder range")
    });
    ExactHeadTargetPose::from_positions(positions[0], positions[1], positions[2], positions[3])
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiko_head_protocol::{HeadTorqueLimits, TorqueLimitPermille};

    fn at(milliseconds: u64) -> MonotonicTime {
        MonotonicTime::from_duration_since_origin(Duration::from_millis(milliseconds))
    }

    fn pose(values: [u16; 4]) -> ExactHeadTargetPose {
        ExactHeadTargetPose::try_from_ticks(values).unwrap()
    }

    fn joint(minimum: u16, maximum: u16, maximum_yield: u16) -> CompliantJointPolicy {
        CompliantJointPolicy::try_new(
            PositionTicks::try_new(minimum).unwrap(),
            PositionTicks::try_new(maximum).unwrap(),
            20,
            6,
            maximum_yield,
            100,
            100,
        )
        .unwrap()
    }

    fn config(acquisition: u8) -> HeadCompliantHoldConfig {
        HeadCompliantHoldConfig::try_new(
            joint(2_064, 2_284, 80),
            joint(2_390, 2_750, 100),
            joint(1_157, 2_117, 180),
            joint(2_887, 3_207, 90),
            HeadTorqueLimits::new(
                TorqueLimitPermille::try_new(600).unwrap(),
                TorqueLimitPermille::try_new(400).unwrap(),
                TorqueLimitPermille::try_new(400).unwrap(),
                TorqueLimitPermille::try_new(400).unwrap(),
            ),
            Duration::from_millis(10),
            Duration::from_millis(10),
            Duration::from_millis(10),
            Duration::from_millis(30),
            Duration::from_millis(10),
            acquisition,
            Duration::from_millis(100),
            Duration::from_millis(1_000),
            800,
        )
        .unwrap()
    }

    fn deployed_pet_joint(
        minimum: u16,
        maximum: u16,
        entry: u16,
        release: u16,
        maximum_yield: u16,
        command_step: u16,
        observed_step: u16,
    ) -> CompliantJointPolicy {
        CompliantJointPolicy::try_new(
            PositionTicks::try_new(minimum).unwrap(),
            PositionTicks::try_new(maximum).unwrap(),
            entry,
            release,
            maximum_yield,
            command_step,
            observed_step,
        )
        .unwrap()
    }

    fn pet_joint(maximum_baseline: u16, rest: i16, directional: u16) -> CompliantPetJointPolicy {
        CompliantPetJointPolicy::try_new(maximum_baseline, rest, directional).unwrap()
    }

    fn deployed_pet_profile() -> CompliantPetProfile {
        CompliantPetProfile::try_new(
            pet_joint(32, -24, 0),
            pet_joint(40, 30, 0),
            pet_joint(24, 0, 20),
            pet_joint(24, 0, 16),
            Duration::from_millis(1_200),
            Duration::from_millis(350),
            Duration::from_secs(3),
            150,
            Duration::from_millis(1_800),
            Duration::from_secs(30),
            3,
            14,
            HeadTorqueLimits::new(
                TorqueLimitPermille::try_new(450).unwrap(),
                TorqueLimitPermille::try_new(350).unwrap(),
                TorqueLimitPermille::try_new(220).unwrap(),
                TorqueLimitPermille::try_new(250).unwrap(),
            ),
            Duration::from_millis(1_200),
            Duration::from_millis(800),
        )
        .unwrap()
    }

    fn pet_base_config(acquisition: u8) -> HeadCompliantHoldConfig {
        HeadCompliantHoldConfig::try_new(
            deployed_pet_joint(2_064, 2_284, 18, 5, 40, 3, 64),
            deployed_pet_joint(2_390, 2_750, 24, 7, 48, 4, 80),
            deployed_pet_joint(1_157, 2_117, 32, 4, 80, 8, 160),
            deployed_pet_joint(2_887, 3_207, 18, 4, 36, 3, 64),
            HeadTorqueLimits::new(
                TorqueLimitPermille::try_new(650).unwrap(),
                TorqueLimitPermille::try_new(550).unwrap(),
                TorqueLimitPermille::try_new(400).unwrap(),
                TorqueLimitPermille::try_new(400).unwrap(),
            ),
            Duration::from_millis(100),
            Duration::from_millis(80),
            Duration::from_millis(60),
            Duration::from_millis(150),
            Duration::from_secs(1),
            acquisition,
            Duration::from_millis(600),
            Duration::from_secs(3),
            650,
        )
        .unwrap()
    }

    fn pet_config(acquisition: u8) -> HeadCompliantHoldConfig {
        pet_base_config(acquisition)
            .try_with_pet_profile(deployed_pet_profile())
            .unwrap()
    }

    fn observation(at_ms: u64, values: [u16; 4], moving: bool) -> CompliantHeadObservation {
        CompliantHeadObservation::from_parts(at(at_ms), values, [moving; 4])
    }

    fn prepare_commit(
        controller: &mut HeadCompliantHoldController,
        at_ms: u64,
        expression: ExactHeadTargetPose,
        quiet: bool,
        positions: [u16; 4],
        moving: bool,
    ) -> CompliantHoldCommitReceipt {
        let prepared = controller
            .prepare(
                at(at_ms),
                expression,
                quiet,
                observation(at_ms, positions, moving),
            )
            .unwrap();
        controller.commit(prepared).unwrap()
    }

    fn arm_at_natural(controller: &mut HeadCompliantHoldController, natural: ExactHeadTargetPose) {
        let positions = natural.positions().map(PositionTicks::get);
        let first = prepare_commit(controller, 0, natural, true, positions, false);
        assert_eq!(first.state(), CompliantHoldState::FollowingExpression);
        let armed = prepare_commit(controller, 10, natural, true, positions, false);
        assert_eq!(armed.state(), CompliantHoldState::FollowingExpression);
    }

    fn arm_pet_with_baseline(
        controller: &mut HeadCompliantHoldController,
        natural: ExactHeadTargetPose,
        positions: [u16; 4],
    ) {
        for time in (0..=1_000).step_by(100) {
            let receipt = prepare_commit(controller, time, natural, true, positions, false);
            assert_eq!(receipt.state(), CompliantHoldState::FollowingExpression);
            if time == 1_000 {
                assert_eq!(receipt.pet_event(), Some(CompliantPetEvent::Ready));
            }
        }
    }

    #[test]
    fn field_pet_profile_preserves_all_measured_constants() {
        let config = pet_config(2);
        let profile = config.pet_profile().expect("pet profile");
        assert_eq!(config.follow_permille(), 650);
        assert_eq!(profile.static_release_dwell(), Duration::from_millis(1_800));
        assert_eq!(profile.maximum_yield_dwell(), Duration::from_secs(30));
        assert_eq!(profile.rest_dwell(), Duration::from_millis(1_200));
        assert_eq!(
            profile.rest_per_additional_joint(),
            Duration::from_millis(350)
        );
        assert_eq!(profile.maximum_rest_dwell(), Duration::from_secs(3));
        assert_eq!(profile.recovery_per_additional_joint_permille(), 150);
        assert_eq!(profile.comfort_roll_tilt_ticks(), 14);
        assert_eq!(
            profile.tap_maximum_contact_duration(),
            Duration::from_millis(1_200)
        );
        assert_eq!(profile.tap_recovery_duration(), Duration::from_millis(800));
        assert_eq!(profile.joint(HeadJoint::Bow).rest_offset_ticks(), -24);
        assert_eq!(profile.joint(HeadJoint::Curl).rest_offset_ticks(), 30);
        assert_eq!(
            profile
                .joint(HeadJoint::Yaw)
                .directional_rest_offset_ticks(),
            20
        );
        assert_eq!(
            profile
                .joint(HeadJoint::Roll)
                .directional_rest_offset_ticks(),
            16
        );
        assert_eq!(
            HeadJoint::ALL.map(|joint| profile.yield_torque_limits().for_joint(joint).get()),
            [450, 350, 220, 250]
        );
    }

    #[test]
    fn pet_profile_binding_rejects_unsafe_cross_field_combinations() {
        let below_floor = CompliantPetProfile {
            yield_torque_limits: HeadTorqueLimits::new(
                TorqueLimitPermille::try_new(299).unwrap(),
                TorqueLimitPermille::try_new(350).unwrap(),
                TorqueLimitPermille::try_new(220).unwrap(),
                TorqueLimitPermille::try_new(250).unwrap(),
            ),
            ..deployed_pet_profile()
        };
        assert!(matches!(
            pet_base_config(1).try_with_pet_profile(below_floor),
            Err(
                HeadCompliantHoldConfigError::YieldTorqueBelowMeasuredFloor {
                    joint: HeadJoint::Bow,
                    actual_permille: 299,
                    minimum_permille: 300,
                }
            )
        ));

        let too_much_roll = CompliantPetProfile {
            comfort_roll_tilt_ticks: 21,
            ..deployed_pet_profile()
        };
        assert!(matches!(
            pet_base_config(1).try_with_pet_profile(too_much_roll),
            Err(
                HeadCompliantHoldConfigError::ComfortRollExceedsMaximumYield {
                    total_ticks: 37,
                    maximum_yield_ticks: 36,
                }
            )
        ));

        let too_short_static = CompliantPetProfile {
            static_release_dwell: Duration::from_millis(200),
            ..deployed_pet_profile()
        };
        assert!(matches!(
            pet_base_config(1).try_with_pet_profile(too_short_static),
            Err(
                HeadCompliantHoldConfigError::StaticReleaseBelowThreeControlPeriods {
                    static_release,
                    minimum,
                }
            ) if static_release == Duration::from_millis(200)
                && minimum == Duration::from_millis(300)
        ));

        let mut collapsed_hysteresis = pet_base_config(1);
        collapsed_hysteresis.follow_permille = NonZeroU16::new(800).unwrap();
        assert!(matches!(
            collapsed_hysteresis.try_with_pet_profile(deployed_pet_profile()),
            Err(HeadCompliantHoldConfigError::FollowCollapsesHysteresis {
                joint: HeadJoint::Bow,
                retained_error_ticks: 4,
                release_error_ticks: 5,
            })
        ));
    }

    #[test]
    fn bounded_stopped_bias_is_learned_before_contact_arms() {
        let natural = pose([2_174, 2_570, 1_637, 3_047]);
        let mut controller =
            HeadCompliantHoldController::try_new(pet_config(1), natural, at(0)).unwrap();
        arm_pet_with_baseline(&mut controller, natural, [2_191, 2_585, 1_637, 3_047]);
        assert_eq!(controller.baseline_error_ticks(), [17, 15, 0, 0]);

        let contact = prepare_commit(
            &mut controller,
            1_100,
            natural,
            true,
            [2_210, 2_585, 1_637, 3_047],
            true,
        );
        assert_eq!(contact.state(), CompliantHoldState::Yielding);
        assert_eq!(contact.pet_event(), Some(CompliantPetEvent::Contact));
        assert_eq!(
            HeadJoint::ALL.map(|joint| contact.desired_torque_limits().for_joint(joint).get()),
            [450, 550, 220, 250]
        );
    }

    #[test]
    fn bias_outside_its_own_envelope_never_arms_or_becomes_touch() {
        let natural = pose([2_174, 2_570, 1_637, 3_047]);
        let mut controller =
            HeadCompliantHoldController::try_new(pet_config(1), natural, at(0)).unwrap();
        for time in (0..=1_500).step_by(100) {
            let receipt = prepare_commit(
                &mut controller,
                time,
                natural,
                true,
                [2_207, 2_570, 1_637, 3_047],
                false,
            );
            assert_eq!(receipt.state(), CompliantHoldState::FollowingExpression);
            assert_eq!(receipt.pet_event(), None);
        }
        assert_eq!(controller.baseline_error_ticks(), [0; JOINT_COUNT]);
    }

    #[test]
    fn quick_single_touch_skips_rest_and_emits_exact_episode_evidence() {
        let natural = pose([2_174, 2_570, 1_637, 3_047]);
        let mut controller =
            HeadCompliantHoldController::try_new(pet_config(1), natural, at(0)).unwrap();
        arm_pet_with_baseline(
            &mut controller,
            natural,
            natural.positions().map(PositionTicks::get),
        );
        let contact = prepare_commit(
            &mut controller,
            1_100,
            natural,
            true,
            [2_204, 2_570, 1_637, 3_047],
            true,
        );
        assert_eq!(contact.pet_event(), Some(CompliantPetEvent::Contact));
        let yielded = contact.committed_target();
        let tap = prepare_commit(
            &mut controller,
            1_200,
            natural,
            true,
            yielded.positions().map(PositionTicks::get),
            false,
        );
        assert_eq!(tap.state(), CompliantHoldState::Recovering);
        assert_eq!(tap.pet_event(), Some(CompliantPetEvent::Tap));

        let mut completed = None;
        for time in (1_300..=2_100).step_by(100) {
            let positions = controller
                .committed_target()
                .positions()
                .map(PositionTicks::get);
            let receipt = prepare_commit(&mut controller, time, natural, true, positions, false);
            assert_ne!(receipt.state(), CompliantHoldState::Resting);
            if let Some(summary) = receipt.completed_episode() {
                assert_eq!(receipt.pet_event(), Some(CompliantPetEvent::Returned));
                completed = Some(summary);
                break;
            }
        }
        let summary = completed.expect("tap returns after its 800 ms recovery");
        assert!(summary.was_tap());
        assert_eq!(summary.yield_entries(), 1);
        assert!(!summary.reached_rest());
        assert!(!summary.reached_comfy());
        assert_eq!(controller.last_completed_episode(), Some(summary));
    }

    fn drive_static_pet_to_rest(
        controller: &mut HeadCompliantHoldController,
        natural: ExactHeadTargetPose,
    ) -> (u64, CompliantHoldCommitReceipt) {
        let mut contact = prepare_commit(
            controller,
            1_100,
            natural,
            true,
            [2_214, 2_570, 1_637, 3_047],
            true,
        );
        let first_yield_time = if contact.state() == CompliantHoldState::ConfirmingContact {
            contact = prepare_commit(
                controller,
                1_200,
                natural,
                true,
                [2_214, 2_570, 1_637, 3_047],
                true,
            );
            1_200
        } else {
            1_100
        };
        assert_eq!(contact.state(), CompliantHoldState::Yielding);
        for time in ((first_yield_time + 100)..=5_000).step_by(100) {
            let receipt = prepare_commit(
                controller,
                time,
                natural,
                true,
                [2_214, 2_570, 1_637, 3_047],
                false,
            );
            if receipt.state() == CompliantHoldState::Resting {
                assert_eq!(receipt.pet_event(), Some(CompliantPetEvent::ReleaseStatic));
                return (time, receipt);
            }
        }
        panic!("static contact did not enter rest");
    }

    #[test]
    fn static_contact_releases_then_pauses_from_actual_rest_arrival() {
        let natural = pose([2_174, 2_570, 1_637, 3_047]);
        let mut controller =
            HeadCompliantHoldController::try_new(pet_config(1), natural, at(0)).unwrap();
        arm_pet_with_baseline(
            &mut controller,
            natural,
            natural.positions().map(PositionTicks::get),
        );
        let (rest_entered_at, _) = drive_static_pet_to_rest(&mut controller, natural);

        let mut comfy_at = None;
        for time in ((rest_entered_at + 100)..=(rest_entered_at + 3_000)).step_by(100) {
            let positions = controller
                .committed_target()
                .positions()
                .map(PositionTicks::get);
            let receipt = prepare_commit(&mut controller, time, natural, true, positions, false);
            if receipt.pet_event() == Some(CompliantPetEvent::Comfy) {
                assert_eq!(receipt.state(), CompliantHoldState::Resting);
                assert_eq!(
                    receipt.committed_target().position(HeadJoint::Bow).get(),
                    2_150
                );
                assert_eq!(
                    receipt.committed_target().position(HeadJoint::Curl).get(),
                    2_600
                );
                assert_eq!(
                    receipt.committed_target().position(HeadJoint::Roll).get(),
                    3_061
                );
                comfy_at = Some(time);
                break;
            }
        }
        let comfy_at = comfy_at.expect("rest target is reached through bounded steps");
        assert!(comfy_at > rest_entered_at);

        for time in ((comfy_at + 100)..(comfy_at + 1_200)).step_by(100) {
            let positions = controller
                .committed_target()
                .positions()
                .map(PositionTicks::get);
            let receipt = prepare_commit(&mut controller, time, natural, true, positions, false);
            assert_eq!(receipt.state(), CompliantHoldState::Resting);
        }
        let positions = controller
            .committed_target()
            .positions()
            .map(PositionTicks::get);
        let recovery = prepare_commit(
            &mut controller,
            comfy_at + 1_200,
            natural,
            true,
            positions,
            false,
        );
        assert_eq!(recovery.state(), CompliantHoldState::Recovering);
        assert_eq!(recovery.pet_event(), Some(CompliantPetEvent::Recovering));
    }

    #[test]
    fn static_residual_during_rest_does_not_knead_cycle_back_into_yield() {
        let natural = pose([2_174, 2_570, 1_637, 3_047]);
        let mut controller =
            HeadCompliantHoldController::try_new(pet_config(2), natural, at(0)).unwrap();
        arm_pet_with_baseline(
            &mut controller,
            natural,
            natural.positions().map(PositionTicks::get),
        );
        let (rest_entered_at, _) = drive_static_pet_to_rest(&mut controller, natural);
        for time in ((rest_entered_at + 100)..=(rest_entered_at + 2_500)).step_by(100) {
            let mut positions = controller
                .committed_target()
                .positions()
                .map(PositionTicks::get);
            positions[joint_index(HeadJoint::Bow)] += 23;
            let receipt = prepare_commit(&mut controller, time, natural, true, positions, false);
            assert_ne!(receipt.pet_event(), Some(CompliantPetEvent::Recontact));
            assert_ne!(receipt.state(), CompliantHoldState::Yielding);
        }
    }

    #[test]
    fn only_varying_contact_can_reacquire_during_rest() {
        let natural = pose([2_174, 2_570, 1_637, 3_047]);
        let mut controller =
            HeadCompliantHoldController::try_new(pet_config(2), natural, at(0)).unwrap();
        arm_pet_with_baseline(
            &mut controller,
            natural,
            natural.positions().map(PositionTicks::get),
        );
        let (rest_entered_at, _) = drive_static_pet_to_rest(&mut controller, natural);
        let base = controller.committed_target().position(HeadJoint::Bow).get();
        let first = prepare_commit(
            &mut controller,
            rest_entered_at + 100,
            natural,
            true,
            [base + 30, 2_570, 1_637, 3_047],
            true,
        );
        assert_eq!(first.state(), CompliantHoldState::Resting);
        let second_base = controller.committed_target().position(HeadJoint::Bow).get();
        let second = prepare_commit(
            &mut controller,
            rest_entered_at + 200,
            natural,
            true,
            [second_base + 40, 2_570, 1_637, 3_047],
            true,
        );
        assert_eq!(second.state(), CompliantHoldState::Resting);
        let third_base = controller.committed_target().position(HeadJoint::Bow).get();
        let third = prepare_commit(
            &mut controller,
            rest_entered_at + 300,
            natural,
            true,
            [third_base + 50, 2_570, 1_637, 3_047],
            true,
        );
        assert_eq!(third.state(), CompliantHoldState::Yielding);
        assert_eq!(third.pet_event(), Some(CompliantPetEvent::Recontact));
    }

    #[test]
    fn continuously_varying_contact_hits_the_declared_yield_timeout() {
        let natural = pose([2_174, 2_570, 1_637, 3_047]);
        let mut controller =
            HeadCompliantHoldController::try_new(pet_config(1), natural, at(0)).unwrap();
        arm_pet_with_baseline(
            &mut controller,
            natural,
            natural.positions().map(PositionTicks::get),
        );
        prepare_commit(
            &mut controller,
            1_100,
            natural,
            true,
            [2_214, 2_570, 1_637, 3_047],
            true,
        );
        let mut timed_out = None;
        for time in (1_200..=31_200).step_by(100) {
            let displacement = if (time / 100) % 2 == 0 { 40 } else { 50 };
            let receipt = prepare_commit(
                &mut controller,
                time,
                natural,
                true,
                [2_174 + displacement, 2_570, 1_637, 3_047],
                true,
            );
            if receipt.pet_event() == Some(CompliantPetEvent::YieldTimeout) {
                timed_out = Some((time, receipt.state()));
                break;
            }
        }
        assert_eq!(timed_out, Some((31_100, CompliantHoldState::Resting)));
    }

    #[test]
    fn config_rejects_hysteresis_without_an_inner_release_band() {
        assert_eq!(
            CompliantJointPolicy::try_new(
                PositionTicks::try_new(1_000).unwrap(),
                PositionTicks::try_new(1_200).unwrap(),
                20,
                20,
                40,
                40,
                40,
            ),
            Err(CompliantJointPolicyError::ReleaseNotInsideEntry {
                release_ticks: 20,
                entry_ticks: 20,
            })
        );
    }

    #[test]
    fn touch_observation_may_cross_a_command_edge_only_by_reviewed_yield() {
        let natural = pose([2_174, 2_570, 1_637, 3_047]);
        let policy = config(1).joint(HeadJoint::Bow);
        assert_eq!(policy.minimum().get(), 2_064);
        assert_eq!(policy.observation_minimum().get(), 1_984);

        let mut exact_edge =
            HeadCompliantHoldController::try_new(config(1), natural, at(0)).unwrap();
        exact_edge
            .prepare(
                at(0),
                natural,
                true,
                observation(0, [1_984, 2_570, 1_637, 3_047], false),
            )
            .expect("the exact command edge plus reviewed yield is observable");

        let mut beyond = HeadCompliantHoldController::try_new(config(1), natural, at(0)).unwrap();
        assert_eq!(
            beyond
                .prepare(
                    at(0),
                    natural,
                    true,
                    observation(0, [1_983, 2_570, 1_637, 3_047], false),
                )
                .expect_err("one tick beyond reviewed yield must fault"),
            CompliantHoldPrepareError::FaultLatched(
                CompliantHoldFault::ObservationOutsideEnvelope {
                    joint: HeadJoint::Bow,
                    observed: PositionTicks::try_new(1_983).unwrap(),
                    minimum: PositionTicks::try_new(1_984).unwrap(),
                    maximum: PositionTicks::try_new(2_364).unwrap(),
                }
            )
        );
    }

    #[test]
    fn expression_command_cannot_use_the_observation_excursion() {
        let natural = pose([2_174, 2_570, 1_637, 3_047]);
        let unsafe_expression = pose([2_063, 2_570, 1_637, 3_047]);
        let mut controller =
            HeadCompliantHoldController::try_new(config(1), natural, at(0)).unwrap();
        assert_eq!(
            controller
                .prepare(
                    at(0),
                    unsafe_expression,
                    true,
                    observation(0, [2_063, 2_570, 1_637, 3_047], false),
                )
                .expect_err("physical observation latitude must never widen commands"),
            CompliantHoldPrepareError::ExpressionTargetOutsideEnvelope {
                joint: HeadJoint::Bow,
                target: PositionTicks::try_new(2_063).unwrap(),
                minimum: PositionTicks::try_new(2_064).unwrap(),
                maximum: PositionTicks::try_new(2_284).unwrap(),
            }
        );
    }

    #[test]
    fn two_consistent_samples_are_required_before_yield() {
        let natural = pose([2_174, 2_570, 1_637, 3_047]);
        let mut controller =
            HeadCompliantHoldController::try_new(config(2), natural, at(0)).unwrap();
        arm_at_natural(&mut controller, natural);
        let first = prepare_commit(
            &mut controller,
            20,
            natural,
            true,
            [2_204, 2_570, 1_637, 3_047],
            true,
        );
        assert_eq!(
            first.disposition(),
            CompliantHoldDisposition::ContactCandidate {
                consecutive_samples: 1
            }
        );
        assert_eq!(first.committed_target(), natural);

        let second = prepare_commit(
            &mut controller,
            30,
            natural,
            true,
            [2_214, 2_570, 1_637, 3_047],
            true,
        );
        assert_eq!(second.state(), CompliantHoldState::Yielding);
        assert_eq!(
            second.committed_target().position(HeadJoint::Bow).get(),
            2_206
        );
    }

    #[test]
    fn commanded_expression_motion_cannot_be_misclassified_as_contact() {
        let natural = pose([2_174, 2_570, 1_637, 3_047]);
        let expression = pose([2_200, 2_550, 1_600, 3_080]);
        let mut controller =
            HeadCompliantHoldController::try_new(config(2), natural, at(0)).unwrap();
        let receipt = prepare_commit(
            &mut controller,
            10,
            expression,
            false,
            natural.positions().map(PositionTicks::get),
            true,
        );
        assert_eq!(receipt.state(), CompliantHoldState::FollowingExpression);
        assert_eq!(receipt.committed_target(), expression);
    }

    #[test]
    fn contact_cannot_arm_until_the_head_is_stationary_and_inside_release_band() {
        let natural = pose([2_174, 2_570, 1_637, 3_047]);
        let mut controller =
            HeadCompliantHoldController::try_new(config(1), natural, at(0)).unwrap();
        for (time, moving) in [(0, true), (10, false)] {
            let receipt = prepare_commit(
                &mut controller,
                time,
                natural,
                true,
                [2_204, 2_570, 1_637, 3_047],
                moving,
            );
            assert_eq!(receipt.state(), CompliantHoldState::FollowingExpression);
            assert_eq!(receipt.committed_target(), natural);
        }

        let positions = natural.positions().map(PositionTicks::get);
        prepare_commit(&mut controller, 20, natural, true, positions, false);
        prepare_commit(&mut controller, 30, natural, true, positions, false);
        let contact = prepare_commit(
            &mut controller,
            40,
            natural,
            true,
            [2_204, 2_570, 1_637, 3_047],
            true,
        );
        assert_eq!(contact.state(), CompliantHoldState::Yielding);
    }

    #[test]
    fn yield_is_bounded_and_reports_limiting() {
        let natural = pose([2_174, 2_570, 1_637, 3_047]);
        let mut controller =
            HeadCompliantHoldController::try_new(config(1), natural, at(0)).unwrap();
        arm_at_natural(&mut controller, natural);
        let receipt = prepare_commit(
            &mut controller,
            20,
            natural,
            true,
            [2_274, 2_470, 1_737, 3_147],
            true,
        );
        assert_eq!(
            receipt.committed_target().positions(),
            [
                PositionTicks::try_new(2_254).unwrap(),
                PositionTicks::try_new(2_490).unwrap(),
                PositionTicks::try_new(1_717).unwrap(),
                PositionTicks::try_new(3_127).unwrap(),
            ]
        );
        assert_eq!(
            receipt.disposition(),
            CompliantHoldDisposition::Yielding {
                envelope_limited: [false; 4],
                command_step_limited: [false; 4],
            }
        );
    }

    #[test]
    fn stationary_release_dwells_then_recovers_with_exact_minimum_jerk_endpoints() {
        let natural = pose([2_174, 2_570, 1_637, 3_047]);
        let mut controller =
            HeadCompliantHoldController::try_new(config(1), natural, at(0)).unwrap();
        arm_at_natural(&mut controller, natural);
        let yielded = prepare_commit(
            &mut controller,
            20,
            natural,
            true,
            [2_224, 2_570, 1_637, 3_047],
            true,
        )
        .committed_target();
        assert_eq!(yielded.position(HeadJoint::Bow).get(), 2_214);

        let dwell = prepare_commit(
            &mut controller,
            30,
            natural,
            true,
            yielded.positions().map(PositionTicks::get),
            false,
        );
        assert_eq!(dwell.state(), CompliantHoldState::ReleaseDwell);
        let recovery_start = prepare_commit(
            &mut controller,
            130,
            natural,
            true,
            yielded.positions().map(PositionTicks::get),
            false,
        );
        assert_eq!(
            recovery_start.disposition(),
            CompliantHoldDisposition::Recovering {
                progress_millionths: 0,
                command_step_limited: [false; 4],
            }
        );
        assert_eq!(recovery_start.committed_target(), yielded);

        let midpoint = prepare_commit(
            &mut controller,
            630,
            natural,
            true,
            yielded.positions().map(PositionTicks::get),
            false,
        );
        assert_eq!(
            midpoint.disposition(),
            CompliantHoldDisposition::Recovering {
                progress_millionths: 500_000,
                command_step_limited: [false; 4],
            }
        );
        assert_eq!(
            midpoint.committed_target().position(HeadJoint::Bow).get(),
            2_194
        );

        let complete = prepare_commit(
            &mut controller,
            1_130,
            natural,
            true,
            midpoint
                .committed_target()
                .positions()
                .map(PositionTicks::get),
            false,
        );
        assert_eq!(
            complete.disposition(),
            CompliantHoldDisposition::ReturnedToExpression
        );
        assert_eq!(complete.committed_target(), natural);
    }

    #[test]
    fn continued_hand_resistance_reacquires_instead_of_fighting() {
        let natural = pose([2_174, 2_570, 1_637, 3_047]);
        let mut controller =
            HeadCompliantHoldController::try_new(config(1), natural, at(0)).unwrap();
        arm_at_natural(&mut controller, natural);
        let yielded = prepare_commit(
            &mut controller,
            20,
            natural,
            true,
            [2_224, 2_570, 1_637, 3_047],
            true,
        )
        .committed_target();
        prepare_commit(
            &mut controller,
            30,
            natural,
            true,
            yielded.positions().map(PositionTicks::get),
            false,
        );
        prepare_commit(
            &mut controller,
            130,
            natural,
            true,
            yielded.positions().map(PositionTicks::get),
            false,
        );
        let resisted = prepare_commit(
            &mut controller,
            230,
            natural,
            true,
            [2_234, 2_570, 1_637, 3_047],
            true,
        );
        assert_eq!(resisted.state(), CompliantHoldState::Yielding);
        assert_eq!(
            resisted.committed_target().position(HeadJoint::Bow).get(),
            2_222
        );
    }

    #[test]
    fn stale_observation_does_not_advance_transactional_state() {
        let natural = pose([2_174, 2_570, 1_637, 3_047]);
        let mut controller =
            HeadCompliantHoldController::try_new(config(2), natural, at(0)).unwrap();
        assert!(matches!(
            controller.prepare(
                at(40),
                natural,
                true,
                observation(10, [2_204, 2_570, 1_637, 3_047], true),
            ),
            Err(CompliantHoldPrepareError::ObservationExpired { .. })
        ));
        assert_eq!(controller.state(), CompliantHoldState::FollowingExpression);
        assert_eq!(controller.committed_target(), natural);
    }

    #[test]
    fn uncertain_application_is_absorbing() {
        let natural = pose([2_174, 2_570, 1_637, 3_047]);
        let mut controller =
            HeadCompliantHoldController::try_new(config(1), natural, at(0)).unwrap();
        arm_at_natural(&mut controller, natural);
        let prepared = controller
            .prepare(
                at(20),
                natural,
                true,
                observation(20, [2_204, 2_570, 1_637, 3_047], true),
            )
            .unwrap();
        controller
            .abort_with_application_uncertain(prepared)
            .unwrap();
        assert_eq!(controller.state(), CompliantHoldState::FaultHeld);
        assert_eq!(
            controller.fault(),
            Some(CompliantHoldFault::ApplicationUncertain)
        );
    }

    #[test]
    fn minimum_jerk_is_monotonic_symmetric_and_endpoint_exact() {
        let total = Duration::from_secs(2);
        assert_eq!(minimum_jerk_progress(Duration::ZERO, total), 0);
        assert_eq!(minimum_jerk_progress(total, total), INTERPOLATION_SCALE);
        let mut previous = 0;
        for millisecond in 0..=2_000 {
            let progress = minimum_jerk_progress(Duration::from_millis(millisecond), total);
            assert!(progress >= previous);
            previous = progress;
        }
        for millisecond in [1, 25, 100, 333, 750, 1_000] {
            let forward = minimum_jerk_progress(Duration::from_millis(millisecond), total);
            let reverse = minimum_jerk_progress(Duration::from_millis(2_000 - millisecond), total);
            assert!(forward.abs_diff(INTERPOLATION_SCALE - reverse) <= 1);
        }
    }

    #[test]
    fn interpolation_never_overshoots_either_endpoint() {
        let start = pose([2_250, 2_470, 1_800, 3_120]);
        let end = pose([2_174, 2_570, 1_637, 3_047]);
        for progress in (0..=1_000_000).step_by(997) {
            let result = interpolate_pose(start, end, progress);
            for joint in HeadJoint::ALL {
                let low = start.position(joint).min(end.position(joint));
                let high = start.position(joint).max(end.position(joint));
                assert!(result.position(joint) >= low);
                assert!(result.position(joint) <= high);
            }
        }
    }

    #[test]
    fn command_step_bound_is_monotonic_and_eventually_reaches_every_axis() {
        let config = config(1);
        let mut current = pose([2_174, 2_570, 1_637, 3_047]);
        let desired = pose([2_284, 2_390, 2_117, 3_207]);
        let mut iterations = 0;
        while current != desired {
            let previous = current;
            let (next, limited) = command_step_target(config, current, desired);
            for joint in HeadJoint::ALL {
                let step = next
                    .position(joint)
                    .get()
                    .abs_diff(previous.position(joint).get());
                assert!(step <= config.joint(joint).maximum_command_step_ticks());
                let before = previous
                    .position(joint)
                    .get()
                    .abs_diff(desired.position(joint).get());
                let after = next
                    .position(joint)
                    .get()
                    .abs_diff(desired.position(joint).get());
                assert!(after <= before);
                assert_eq!(
                    limited[joint_index(joint)],
                    before > config.joint(joint).maximum_command_step_ticks()
                );
            }
            current = next;
            iterations += 1;
            assert!(iterations <= 5, "bounded convergence must make progress");
        }
    }
}
