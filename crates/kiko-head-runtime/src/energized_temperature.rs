//! Stateful energized-temperature supervision for the compliant head owner.
//!
//! The STS bus has produced checksum-valid temperature bytes in a corruption
//! band while every other register remained plausible. A single byte therefore
//! cannot safely mean either "hot" or "cool". This controller parses each raw
//! sample once into one of three meanings and carries the evidence across
//! control slots without sleeping in the control path.

use std::num::{NonZeroU8, NonZeroU16};

use kiko_head_protocol::HeadJoint;

use crate::MonotonicTime;

const JOINT_COUNT: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EnergizedTemperatureSample {
    raw: u8,
    received_at: MonotonicTime,
}

impl EnergizedTemperatureSample {
    pub(crate) const fn new(raw: u8, received_at: MonotonicTime) -> Self {
        Self { raw, received_at }
    }

    pub const fn raw(self) -> u8 {
        self.raw
    }

    pub const fn received_at(self) -> MonotonicTime {
        self.received_at
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnergizedTemperatureChannelStatus {
    Normal(EnergizedTemperatureSample),
    PlausibleHot {
        sample: EnergizedTemperatureSample,
        consecutive_plausible_hot: NonZeroU8,
        abort_after: NonZeroU8,
    },
    Implausible {
        sample: EnergizedTemperatureSample,
        consecutive_implausible: NonZeroU16,
        abort_after: NonZeroU16,
        maximum_plausible_raw_inclusive: u8,
    },
}

/// Exact set of temperature bytes and receive times already classified by the
/// stateful supervisor. Only this module can construct the proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EnergizedTemperatureAdmission {
    samples: [EnergizedTemperatureSample; JOINT_COUNT],
}

impl EnergizedTemperatureAdmission {
    pub(crate) fn admits(self, index: usize, raw: u8, received_at: MonotonicTime) -> bool {
        self.samples[index].raw == raw && self.samples[index].received_at == received_at
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum EnergizedTemperatureSupervision {
    Admitted {
        channels: [EnergizedTemperatureChannelStatus; JOINT_COUNT],
        admission: EnergizedTemperatureAdmission,
    },
    AdmittedWithImplausible {
        channels: [EnergizedTemperatureChannelStatus; JOINT_COUNT],
        admission: EnergizedTemperatureAdmission,
    },
    Deferred {
        channels: [EnergizedTemperatureChannelStatus; JOINT_COUNT],
    },
    ConfirmedOvertemperature {
        joint: HeadJoint,
        samples: [EnergizedTemperatureSample; 3],
        maximum_raw_exclusive: u8,
    },
    Unreadable {
        joint: HeadJoint,
        last_sample: EnergizedTemperatureSample,
        consecutive_implausible: NonZeroU16,
        maximum_plausible_raw_inclusive: u8,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct EnergizedTemperatureSupervisor {
    maximum_normal_raw_exclusive: u8,
    maximum_plausible_raw_inclusive: u8,
    plausible_hot_abort_samples: NonZeroU8,
    implausible_abort_samples: NonZeroU16,
    plausible_hot_counts: [u8; JOINT_COUNT],
    implausible_counts: [u16; JOINT_COUNT],
    plausible_hot_samples: [[Option<EnergizedTemperatureSample>; 3]; JOINT_COUNT],
}

impl EnergizedTemperatureSupervisor {
    pub(crate) fn kiko_field_profile(maximum_normal_raw_exclusive: u8) -> Self {
        const MAXIMUM_PLAUSIBLE_RAW_INCLUSIVE: u8 = 95;
        const PLAUSIBLE_HOT_ABORT_SAMPLES: u8 = 3;
        const IMPLAUSIBLE_ABORT_SAMPLES: u16 = 300;

        assert!(maximum_normal_raw_exclusive > 0);
        assert!(maximum_normal_raw_exclusive <= MAXIMUM_PLAUSIBLE_RAW_INCLUSIVE);
        Self {
            maximum_normal_raw_exclusive,
            maximum_plausible_raw_inclusive: MAXIMUM_PLAUSIBLE_RAW_INCLUSIVE,
            plausible_hot_abort_samples: NonZeroU8::new(PLAUSIBLE_HOT_ABORT_SAMPLES)
                .expect("the Kiko hot-streak requirement is non-zero"),
            implausible_abort_samples: NonZeroU16::new(IMPLAUSIBLE_ABORT_SAMPLES)
                .expect("the Kiko unreadable-streak requirement is non-zero"),
            plausible_hot_counts: [0; JOINT_COUNT],
            implausible_counts: [0; JOINT_COUNT],
            plausible_hot_samples: [[None; 3]; JOINT_COUNT],
        }
    }

    pub(crate) const fn maximum_plausible_raw_inclusive(&self) -> u8 {
        self.maximum_plausible_raw_inclusive
    }

    pub(crate) fn observe(
        &mut self,
        samples: [EnergizedTemperatureSample; JOINT_COUNT],
    ) -> EnergizedTemperatureSupervision {
        let mut channels = samples.map(EnergizedTemperatureChannelStatus::Normal);
        for (index, sample) in samples.into_iter().enumerate() {
            if sample.raw < self.maximum_normal_raw_exclusive {
                self.plausible_hot_counts[index] = 0;
                self.plausible_hot_samples[index] = [None; 3];
                self.implausible_counts[index] = 0;
                continue;
            }

            if sample.raw > self.maximum_plausible_raw_inclusive {
                self.implausible_counts[index] = self.implausible_counts[index].saturating_add(1);
                channels[index] = EnergizedTemperatureChannelStatus::Implausible {
                    sample,
                    consecutive_implausible: NonZeroU16::new(self.implausible_counts[index])
                        .expect("an observed implausible sample starts a non-zero streak"),
                    abort_after: self.implausible_abort_samples,
                    maximum_plausible_raw_inclusive: self.maximum_plausible_raw_inclusive,
                };
                // An implausible byte is evidence of neither heat nor cooling:
                // retain any real plausible-hot streak across this slot.
                continue;
            }

            self.implausible_counts[index] = 0;
            let next = self.plausible_hot_counts[index].saturating_add(1);
            self.plausible_hot_counts[index] = next;
            let slot = usize::from(next.saturating_sub(1))
                .min(self.plausible_hot_samples[index].len() - 1);
            self.plausible_hot_samples[index][slot] = Some(sample);
            channels[index] = EnergizedTemperatureChannelStatus::PlausibleHot {
                sample,
                consecutive_plausible_hot: NonZeroU8::new(next)
                    .expect("an observed plausible-hot sample starts a non-zero streak"),
                abort_after: self.plausible_hot_abort_samples,
            };
        }

        for (index, joint) in HeadJoint::ALL.into_iter().enumerate() {
            if self.plausible_hot_counts[index] >= self.plausible_hot_abort_samples.get() {
                let samples = self.plausible_hot_samples[index].map(|sample| {
                    sample.expect("the three-sample hot streak retains every admitted sample")
                });
                return EnergizedTemperatureSupervision::ConfirmedOvertemperature {
                    joint,
                    samples,
                    maximum_raw_exclusive: self.maximum_normal_raw_exclusive,
                };
            }
        }
        for (index, joint) in HeadJoint::ALL.into_iter().enumerate() {
            if self.implausible_counts[index] >= self.implausible_abort_samples.get() {
                return EnergizedTemperatureSupervision::Unreadable {
                    joint,
                    last_sample: samples[index],
                    consecutive_implausible: NonZeroU16::new(self.implausible_counts[index])
                        .expect("the unreadable fault has a non-zero streak"),
                    maximum_plausible_raw_inclusive: self.maximum_plausible_raw_inclusive,
                };
            }
        }

        let admission = EnergizedTemperatureAdmission { samples };
        if channels
            .iter()
            .all(|status| matches!(status, EnergizedTemperatureChannelStatus::Normal(_)))
        {
            EnergizedTemperatureSupervision::Admitted {
                channels,
                admission,
            }
        } else if self.plausible_hot_counts.iter().all(|count| *count == 0) {
            EnergizedTemperatureSupervision::AdmittedWithImplausible {
                channels,
                admission,
            }
        } else {
            EnergizedTemperatureSupervision::Deferred { channels }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn at(milliseconds: u64) -> MonotonicTime {
        MonotonicTime::from_duration_since_origin(Duration::from_millis(milliseconds))
    }

    fn samples(at_ms: u64, raw: [u8; JOINT_COUNT]) -> [EnergizedTemperatureSample; JOINT_COUNT] {
        raw.map(|value| EnergizedTemperatureSample::new(value, at(at_ms)))
    }

    #[test]
    fn three_plausible_hot_slots_are_required_and_retain_exact_times() {
        let mut supervisor = EnergizedTemperatureSupervisor::kiko_field_profile(65);
        for slot in 1..3 {
            assert!(matches!(
                supervisor.observe(samples(slot * 100, [66, 40, 40, 40])),
                EnergizedTemperatureSupervision::Deferred { .. }
            ));
        }
        assert_eq!(
            supervisor.observe(samples(300, [67, 40, 40, 40])),
            EnergizedTemperatureSupervision::ConfirmedOvertemperature {
                joint: HeadJoint::Bow,
                samples: [
                    EnergizedTemperatureSample::new(66, at(100)),
                    EnergizedTemperatureSample::new(66, at(200)),
                    EnergizedTemperatureSample::new(67, at(300)),
                ],
                maximum_raw_exclusive: 65,
            }
        );
    }

    #[test]
    fn cool_breaks_hot_streak_but_corruption_does_not() {
        let mut supervisor = EnergizedTemperatureSupervisor::kiko_field_profile(65);
        supervisor.observe(samples(100, [66, 40, 40, 40]));
        supervisor.observe(samples(200, [64, 40, 40, 40]));
        supervisor.observe(samples(300, [66, 40, 40, 40]));
        assert!(matches!(
            supervisor.observe(samples(400, [66, 40, 40, 40])),
            EnergizedTemperatureSupervision::Deferred { .. }
        ));

        let mut supervisor = EnergizedTemperatureSupervisor::kiko_field_profile(65);
        supervisor.observe(samples(100, [66, 40, 40, 40]));
        supervisor.observe(samples(200, [150, 40, 40, 40]));
        supervisor.observe(samples(300, [66, 40, 40, 40]));
        assert!(matches!(
            supervisor.observe(samples(400, [66, 40, 40, 40])),
            EnergizedTemperatureSupervision::ConfirmedOvertemperature {
                joint: HeadJoint::Bow,
                samples,
                ..
            } if samples.map(EnergizedTemperatureSample::raw) == [66, 66, 66]
        ));
    }

    #[test]
    fn plausible_sample_resets_only_the_unreadable_streak() {
        let mut supervisor = EnergizedTemperatureSupervisor::kiko_field_profile(65);
        for slot in 1..300 {
            assert!(matches!(
                supervisor.observe(samples(slot, [150, 40, 40, 40])),
                EnergizedTemperatureSupervision::AdmittedWithImplausible { .. }
            ));
        }
        assert!(matches!(
            supervisor.observe(samples(300, [95, 40, 40, 40])),
            EnergizedTemperatureSupervision::Deferred { channels }
                if matches!(
                    channels[0],
                    EnergizedTemperatureChannelStatus::PlausibleHot {
                        consecutive_plausible_hot,
                        ..
                    } if consecutive_plausible_hot.get() == 1
                )
        ));
        assert!(matches!(
            supervisor.observe(samples(301, [150, 40, 40, 40])),
            EnergizedTemperatureSupervision::Deferred { channels }
                if matches!(
                    channels[0],
                    EnergizedTemperatureChannelStatus::Implausible {
                        consecutive_implausible,
                        ..
                    } if consecutive_implausible.get() == 1
                )
        ));
    }

    #[test]
    fn three_hundred_consecutive_implausible_slots_fault() {
        let mut supervisor = EnergizedTemperatureSupervisor::kiko_field_profile(65);
        for slot in 1..300 {
            supervisor.observe(samples(slot, [150, 40, 40, 40]));
        }
        assert!(matches!(
            supervisor.observe(samples(300, [151, 40, 40, 40])),
            EnergizedTemperatureSupervision::Unreadable {
                joint: HeadJoint::Bow,
                last_sample,
                consecutive_implausible,
                maximum_plausible_raw_inclusive: 95,
            } if last_sample.raw() == 151 && consecutive_implausible.get() == 300
        ));
    }

    #[test]
    fn all_normal_channels_are_admitted_and_reset_both_streak_classes() {
        let mut supervisor = EnergizedTemperatureSupervisor::kiko_field_profile(65);
        supervisor.observe(samples(100, [66, 150, 40, 40]));
        assert!(matches!(
            supervisor.observe(samples(200, [40; JOINT_COUNT])),
            EnergizedTemperatureSupervision::Admitted { channels, .. }
                if channels.iter().all(|status| matches!(
                    status,
                    EnergizedTemperatureChannelStatus::Normal(sample) if sample.raw() == 40
                ))
        ));
    }
}
