//! Hysteretic pitch-workload derating for Kiko's bow and curl servos.
//!
//! This is deliberately separate from the energized-temperature safety
//! supervisor. The safety supervisor decides whether continued actuation is
//! admissible at all; this controller removes pitch workload before that hard
//! boundary. Samples are already protocol-parsed raw temperature bytes.

use core::fmt;
use core::num::NonZeroU8;

use crate::energized_temperature::EnergizedTemperatureSample;

const BOW_INDEX: usize = 0;
const CURL_INDEX: usize = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadThermalDeratePolicy {
    engage_temperature_raw: u8,
    clear_temperature_raw: u8,
    abort_temperature_raw_exclusive: u8,
    engage_samples: NonZeroU8,
    clear_samples: NonZeroU8,
    maximum_plausible_temperature_raw_inclusive: u8,
}

impl HeadThermalDeratePolicy {
    pub fn try_new(
        engage_temperature_raw: u8,
        clear_temperature_raw: u8,
        abort_temperature_raw_exclusive: u8,
        engage_samples: NonZeroU8,
        clear_samples: NonZeroU8,
        maximum_plausible_temperature_raw_inclusive: u8,
    ) -> Result<Self, HeadThermalDeratePolicyError> {
        if engage_temperature_raw == 0 || engage_temperature_raw >= abort_temperature_raw_exclusive
        {
            return Err(HeadThermalDeratePolicyError::EngageOutsideSafeBand {
                engage_temperature_raw,
                abort_temperature_raw_exclusive,
            });
        }
        if clear_temperature_raw >= engage_temperature_raw {
            return Err(HeadThermalDeratePolicyError::ClearNotBelowEngage {
                clear_temperature_raw,
                engage_temperature_raw,
            });
        }
        if maximum_plausible_temperature_raw_inclusive < abort_temperature_raw_exclusive {
            return Err(
                HeadThermalDeratePolicyError::PlausibilityCeilingBelowAbort {
                    maximum_plausible_temperature_raw_inclusive,
                    abort_temperature_raw_exclusive,
                },
            );
        }
        Ok(Self {
            engage_temperature_raw,
            clear_temperature_raw,
            abort_temperature_raw_exclusive,
            engage_samples,
            clear_samples,
            maximum_plausible_temperature_raw_inclusive,
        })
    }

    pub const fn kiko_field_profile() -> Self {
        Self {
            engage_temperature_raw: 60,
            clear_temperature_raw: 56,
            abort_temperature_raw_exclusive: 65,
            engage_samples: NonZeroU8::new(3).expect("three is nonzero"),
            clear_samples: NonZeroU8::new(10).expect("ten is nonzero"),
            maximum_plausible_temperature_raw_inclusive: 95,
        }
    }

    pub const fn engage_temperature_raw(self) -> u8 {
        self.engage_temperature_raw
    }

    pub const fn clear_temperature_raw(self) -> u8 {
        self.clear_temperature_raw
    }

    pub const fn abort_temperature_raw_exclusive(self) -> u8 {
        self.abort_temperature_raw_exclusive
    }

    pub const fn engage_samples(self) -> NonZeroU8 {
        self.engage_samples
    }

    pub const fn clear_samples(self) -> NonZeroU8 {
        self.clear_samples
    }

    pub const fn maximum_plausible_temperature_raw_inclusive(self) -> u8 {
        self.maximum_plausible_temperature_raw_inclusive
    }

    pub fn admit_runtime_supervision(
        self,
        actual_raw_exclusive: u8,
        actual_maximum_plausible_raw_inclusive: u8,
    ) -> Result<(), HeadThermalDerateBindingError> {
        if self.abort_temperature_raw_exclusive != actual_raw_exclusive {
            return Err(HeadThermalDerateBindingError::AbortThresholdMismatch {
                declared_raw_exclusive: self.abort_temperature_raw_exclusive,
                runtime_raw_exclusive: actual_raw_exclusive,
            });
        }
        if self.maximum_plausible_temperature_raw_inclusive
            != actual_maximum_plausible_raw_inclusive
        {
            return Err(HeadThermalDerateBindingError::PlausibilityCeilingMismatch {
                declared_raw_inclusive: self.maximum_plausible_temperature_raw_inclusive,
                runtime_raw_inclusive: actual_maximum_plausible_raw_inclusive,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadThermalDerateState {
    Nominal,
    PitchDerated,
}

impl HeadThermalDerateState {
    pub const fn pitch_derated(self) -> bool {
        matches!(self, Self::PitchDerated)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadThermalDerateTransition {
    Engaged,
    Cleared,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadThermalDerateObservation {
    Admitted { pitch_hottest_raw: u8 },
    HeldImplausible { bow_raw: u8, curl_raw: u8 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadThermalDerateStep {
    state: HeadThermalDerateState,
    observation: HeadThermalDerateObservation,
    transition: Option<HeadThermalDerateTransition>,
}

impl HeadThermalDerateStep {
    pub const fn state(self) -> HeadThermalDerateState {
        self.state
    }

    pub const fn observation(self) -> HeadThermalDerateObservation {
        self.observation
    }

    pub const fn transition(self) -> Option<HeadThermalDerateTransition> {
        self.transition
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HeadThermalDerateController {
    policy: HeadThermalDeratePolicy,
    state: HeadThermalDerateState,
    engage_count: u8,
    clear_count: u8,
}

impl HeadThermalDerateController {
    pub(crate) const fn new(policy: HeadThermalDeratePolicy) -> Self {
        Self {
            policy,
            state: HeadThermalDerateState::Nominal,
            engage_count: 0,
            clear_count: 0,
        }
    }

    pub(crate) const fn state(self) -> HeadThermalDerateState {
        self.state
    }

    pub(crate) fn observe(
        &mut self,
        samples: [EnergizedTemperatureSample; 4],
    ) -> HeadThermalDerateStep {
        let bow_raw = samples[BOW_INDEX].raw();
        let curl_raw = samples[CURL_INDEX].raw();
        if bow_raw > self.policy.maximum_plausible_temperature_raw_inclusive
            || curl_raw > self.policy.maximum_plausible_temperature_raw_inclusive
        {
            return HeadThermalDerateStep {
                state: self.state,
                observation: HeadThermalDerateObservation::HeldImplausible { bow_raw, curl_raw },
                transition: None,
            };
        }

        let pitch_hottest_raw = bow_raw.max(curl_raw);
        let mut transition = None;
        match self.state {
            HeadThermalDerateState::Nominal => {
                self.clear_count = 0;
                self.engage_count = if pitch_hottest_raw >= self.policy.engage_temperature_raw {
                    self.engage_count.saturating_add(1)
                } else {
                    0
                };
                if self.engage_count >= self.policy.engage_samples.get() {
                    self.state = HeadThermalDerateState::PitchDerated;
                    self.engage_count = 0;
                    transition = Some(HeadThermalDerateTransition::Engaged);
                }
            }
            HeadThermalDerateState::PitchDerated => {
                self.engage_count = 0;
                if pitch_hottest_raw <= self.policy.clear_temperature_raw {
                    self.clear_count = self.clear_count.saturating_add(1);
                } else {
                    // Preserve Fable's field behavior: one warm sample costs
                    // one cool count instead of discarding the whole trend.
                    self.clear_count = self.clear_count.saturating_sub(1);
                }
                if self.clear_count >= self.policy.clear_samples.get() {
                    self.state = HeadThermalDerateState::Nominal;
                    self.clear_count = 0;
                    transition = Some(HeadThermalDerateTransition::Cleared);
                }
            }
        }
        HeadThermalDerateStep {
            state: self.state,
            observation: HeadThermalDerateObservation::Admitted { pitch_hottest_raw },
            transition,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadThermalDeratePolicyError {
    EngageOutsideSafeBand {
        engage_temperature_raw: u8,
        abort_temperature_raw_exclusive: u8,
    },
    ClearNotBelowEngage {
        clear_temperature_raw: u8,
        engage_temperature_raw: u8,
    },
    PlausibilityCeilingBelowAbort {
        maximum_plausible_temperature_raw_inclusive: u8,
        abort_temperature_raw_exclusive: u8,
    },
}

impl fmt::Display for HeadThermalDeratePolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid head thermal-derate policy: {self:?}")
    }
}

impl core::error::Error for HeadThermalDeratePolicyError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadThermalDerateBindingError {
    AbortThresholdMismatch {
        declared_raw_exclusive: u8,
        runtime_raw_exclusive: u8,
    },
    PlausibilityCeilingMismatch {
        declared_raw_inclusive: u8,
        runtime_raw_inclusive: u8,
    },
}

impl fmt::Display for HeadThermalDerateBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "head thermal-derate binding failed: {self:?}")
    }
}

impl core::error::Error for HeadThermalDerateBindingError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::MonotonicTime;
    use std::time::Duration;

    fn sample(raw: [u8; 4], sequence: u64) -> [EnergizedTemperatureSample; 4] {
        raw.map(|raw| {
            EnergizedTemperatureSample::new(
                raw,
                MonotonicTime::from_duration_since_origin(Duration::from_millis(sequence)),
            )
        })
    }

    #[test]
    fn field_profile_matches_the_attended_fable_policy() {
        let policy = HeadThermalDeratePolicy::kiko_field_profile();
        assert_eq!(policy.engage_temperature_raw(), 60);
        assert_eq!(policy.clear_temperature_raw(), 56);
        assert_eq!(policy.abort_temperature_raw_exclusive(), 65);
        assert_eq!(policy.engage_samples().get(), 3);
        assert_eq!(policy.clear_samples().get(), 10);
        assert_eq!(policy.maximum_plausible_temperature_raw_inclusive(), 95);
    }

    #[test]
    fn invalid_threshold_relationships_fail_closed() {
        let one = NonZeroU8::new(1).unwrap();
        assert!(matches!(
            HeadThermalDeratePolicy::try_new(65, 56, 65, one, one, 95),
            Err(HeadThermalDeratePolicyError::EngageOutsideSafeBand { .. })
        ));
        assert!(matches!(
            HeadThermalDeratePolicy::try_new(60, 60, 65, one, one, 95),
            Err(HeadThermalDeratePolicyError::ClearNotBelowEngage { .. })
        ));
        assert!(matches!(
            HeadThermalDeratePolicy::try_new(60, 56, 65, one, one, 64),
            Err(HeadThermalDeratePolicyError::PlausibilityCeilingBelowAbort { .. })
        ));
    }

    #[test]
    fn three_hot_pitch_samples_engage_and_non_pitch_heat_does_not() {
        let mut controller =
            HeadThermalDerateController::new(HeadThermalDeratePolicy::kiko_field_profile());
        for sequence in 0..8 {
            assert_eq!(
                controller
                    .observe(sample([45, 45, 90, 90], sequence))
                    .state(),
                HeadThermalDerateState::Nominal
            );
        }
        for sequence in 8..10 {
            assert_eq!(
                controller
                    .observe(sample([60, 59, 40, 40], sequence))
                    .state(),
                HeadThermalDerateState::Nominal
            );
        }
        let engaged = controller.observe(sample([61, 59, 40, 40], 10));
        assert_eq!(engaged.state(), HeadThermalDerateState::PitchDerated);
        assert_eq!(
            engaged.transition(),
            Some(HeadThermalDerateTransition::Engaged)
        );
    }

    #[test]
    fn implausible_pitch_bytes_hold_both_streaks() {
        let mut controller =
            HeadThermalDerateController::new(HeadThermalDeratePolicy::kiko_field_profile());
        for sequence in 0..2 {
            controller.observe(sample([60, 60, 40, 40], sequence));
        }
        let held = controller.observe(sample([150, 44, 40, 40], 2));
        assert!(matches!(
            held.observation(),
            HeadThermalDerateObservation::HeldImplausible { .. }
        ));
        let engaged = controller.observe(sample([60, 60, 40, 40], 3));
        assert_eq!(
            engaged.transition(),
            Some(HeadThermalDerateTransition::Engaged)
        );

        for sequence in 4..13 {
            controller.observe(sample([50, 50, 40, 40], sequence));
        }
        controller.observe(sample([150, 44, 40, 40], 13));
        let cleared = controller.observe(sample([50, 50, 40, 40], 14));
        assert_eq!(
            cleared.transition(),
            Some(HeadThermalDerateTransition::Cleared)
        );
    }

    #[test]
    fn isolated_warm_sample_decrements_instead_of_resetting_clear_progress() {
        let mut controller =
            HeadThermalDerateController::new(HeadThermalDeratePolicy::kiko_field_profile());
        for sequence in 0..3 {
            controller.observe(sample([60, 60, 40, 40], sequence));
        }
        for sequence in 3..12 {
            controller.observe(sample([56, 55, 40, 40], sequence));
        }
        assert_eq!(
            controller.observe(sample([57, 55, 40, 40], 12)).state(),
            HeadThermalDerateState::PitchDerated
        );
        assert_eq!(
            controller.observe(sample([56, 55, 40, 40], 13)).state(),
            HeadThermalDerateState::PitchDerated
        );
        assert_eq!(
            controller
                .observe(sample([56, 55, 40, 40], 14))
                .transition(),
            Some(HeadThermalDerateTransition::Cleared)
        );
    }
}
