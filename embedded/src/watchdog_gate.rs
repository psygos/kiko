//! Pure policy for deciding whether one completed main-loop iteration may feed
//! an independent watchdog.
//!
//! A target adapter must still arrange the actual IWDG configuration and feed.
//! This module intentionally provides no interrupt-side feed path.

use core::num::NonZeroU64;

use crate::controller::ControllerWatchdogStatus;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct LoopIteration(NonZeroU64);

impl LoopIteration {
    pub const FIRST: Self = Self(NonZeroU64::MIN);

    pub fn try_new(value: u64) -> Result<Self, LoopIterationError> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or(LoopIterationError::Zero)
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }

    pub fn checked_successor(self) -> Result<Self, LoopIterationError> {
        self.get()
            .checked_add(1)
            .and_then(NonZeroU64::new)
            .map(Self)
            .ok_or(LoopIterationError::Exhausted)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopIterationError {
    Zero,
    Exhausted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompletedLoopSafety {
    controller: ControllerWatchdogStatus,
    motor_state_synchronized: bool,
    rx_stream_valid: bool,
    critical_report_path_ready: bool,
}

impl CompletedLoopSafety {
    pub const fn new(
        controller: ControllerWatchdogStatus,
        motor_state_synchronized: bool,
        rx_stream_valid: bool,
        critical_report_path_ready: bool,
    ) -> Self {
        Self {
            controller,
            motor_state_synchronized,
            rx_stream_valid,
            critical_report_path_ready,
        }
    }

    pub const fn controller(self) -> ControllerWatchdogStatus {
        self.controller
    }

    /// Whether the target adapter successfully executed every motor directive
    /// issued through this iteration.  This is software synchronization
    /// evidence, not proof that a physical motor or wheel responded.
    pub const fn motor_state_synchronized(self) -> bool {
        self.motor_state_synchronized
    }

    pub const fn rx_stream_valid(self) -> bool {
        self.rx_stream_valid
    }

    pub const fn critical_report_path_ready(self) -> bool {
        self.critical_report_path_ready
    }
}

#[derive(Debug, PartialEq, Eq)]
#[must_use = "a watchdog permit must be consumed by the target feed boundary"]
pub struct WatchdogFeedPermit {
    iteration: LoopIteration,
}

impl WatchdogFeedPermit {
    pub const fn iteration(&self) -> LoopIteration {
        self.iteration
    }

    /// Consume the one-shot capability at the target feed boundary.
    pub const fn consume(self) -> LoopIteration {
        self.iteration
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WatchdogWithholdReason {
    ControllerUnsafeExpiredOrAmbiguous,
    MotorStateNotSynchronized,
    RxStreamInvalid,
    CriticalReportPathUnavailable,
}

#[derive(Debug, PartialEq, Eq)]
pub enum WatchdogDecision {
    Feed(WatchdogFeedPermit),
    Withhold(WatchdogWithholdReason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WatchdogGateError {
    IterationNotStrictlyNewer {
        previous: LoopIteration,
        received: LoopIteration,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WatchdogGate {
    last_completed: Option<LoopIteration>,
}

impl WatchdogGate {
    pub const fn new() -> Self {
        Self {
            last_completed: None,
        }
    }

    pub const fn last_completed(self) -> Option<LoopIteration> {
        self.last_completed
    }

    /// Evaluate one already-completed loop iteration exactly once.  Even a
    /// withheld iteration is committed, so callers cannot retry it after
    /// changing evidence.
    pub fn complete_iteration(
        &mut self,
        iteration: LoopIteration,
        safety: CompletedLoopSafety,
    ) -> Result<WatchdogDecision, WatchdogGateError> {
        if let Some(previous) = self.last_completed
            && iteration <= previous
        {
            return Err(WatchdogGateError::IterationNotStrictlyNewer {
                previous,
                received: iteration,
            });
        }
        self.last_completed = Some(iteration);

        if safety.controller() == ControllerWatchdogStatus::UnsafeExpiredOrAmbiguous {
            return Ok(WatchdogDecision::Withhold(
                WatchdogWithholdReason::ControllerUnsafeExpiredOrAmbiguous,
            ));
        }
        if !safety.motor_state_synchronized() {
            return Ok(WatchdogDecision::Withhold(
                WatchdogWithholdReason::MotorStateNotSynchronized,
            ));
        }
        if !safety.rx_stream_valid() {
            return Ok(WatchdogDecision::Withhold(
                WatchdogWithholdReason::RxStreamInvalid,
            ));
        }
        if !safety.critical_report_path_ready() {
            return Ok(WatchdogDecision::Withhold(
                WatchdogWithholdReason::CriticalReportPathUnavailable,
            ));
        }
        Ok(WatchdogDecision::Feed(WatchdogFeedPermit { iteration }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iteration(value: u64) -> LoopIteration {
        LoopIteration::try_new(value).expect("test iteration is nonzero")
    }

    #[test]
    fn feed_requires_every_piece_of_safe_completed_loop_evidence() {
        let controller_states = [
            ControllerWatchdogStatus::SafeOutputsDisabled,
            ControllerWatchdogStatus::SafeTransitionWithLiveLease,
            ControllerWatchdogStatus::SafeDrivingWithLiveLease,
            ControllerWatchdogStatus::UnsafeExpiredOrAmbiguous,
        ];
        let mut next_iteration = 1_u64;
        let mut gate = WatchdogGate::new();
        for controller in controller_states {
            for motor_state_synchronized in [false, true] {
                for rx_stream_valid in [false, true] {
                    for critical_report_path_ready in [false, true] {
                        let decision = gate
                            .complete_iteration(
                                iteration(next_iteration),
                                CompletedLoopSafety::new(
                                    controller,
                                    motor_state_synchronized,
                                    rx_stream_valid,
                                    critical_report_path_ready,
                                ),
                            )
                            .expect("iterations are strictly increasing");
                        next_iteration += 1;

                        let should_feed = controller
                            != ControllerWatchdogStatus::UnsafeExpiredOrAmbiguous
                            && motor_state_synchronized
                            && rx_stream_valid
                            && critical_report_path_ready;
                        assert_eq!(matches!(decision, WatchdogDecision::Feed(_)), should_feed);
                    }
                }
            }
        }
    }

    #[test]
    fn an_iteration_is_one_shot_even_when_feed_is_withheld() {
        let mut gate = WatchdogGate::new();
        let unsafe_iteration = iteration(1);
        let decision = gate
            .complete_iteration(
                unsafe_iteration,
                CompletedLoopSafety::new(
                    ControllerWatchdogStatus::SafeOutputsDisabled,
                    true,
                    false,
                    true,
                ),
            )
            .expect("first observation");
        assert_eq!(
            decision,
            WatchdogDecision::Withhold(WatchdogWithholdReason::RxStreamInvalid)
        );
        assert!(matches!(
            gate.complete_iteration(
                unsafe_iteration,
                CompletedLoopSafety::new(
                    ControllerWatchdogStatus::SafeOutputsDisabled,
                    true,
                    true,
                    true,
                ),
            ),
            Err(WatchdogGateError::IterationNotStrictlyNewer { .. })
        ));
    }

    #[test]
    fn feed_permit_is_bound_to_the_exact_completed_iteration() {
        let mut gate = WatchdogGate::new();
        let expected = iteration(7);
        let WatchdogDecision::Feed(permit) = gate
            .complete_iteration(
                expected,
                CompletedLoopSafety::new(
                    ControllerWatchdogStatus::SafeDrivingWithLiveLease,
                    true,
                    true,
                    true,
                ),
            )
            .expect("safe first iteration")
        else {
            panic!("complete safe evidence must issue a permit")
        };
        assert_eq!(permit.iteration(), expected);
        assert_eq!(permit.consume(), expected);
    }

    #[test]
    fn iteration_domain_rejects_zero_and_exhaustion() {
        assert_eq!(LoopIteration::try_new(0), Err(LoopIterationError::Zero));
        assert_eq!(
            iteration(u64::MAX).checked_successor(),
            Err(LoopIterationError::Exhausted)
        );
    }
}
