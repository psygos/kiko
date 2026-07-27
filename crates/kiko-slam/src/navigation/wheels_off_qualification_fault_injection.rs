//! Typed, one-shot faults available only in the manually invoked wheels-off
//! qualifier.
//!
//! These declarations are command-line inputs, not deployment configuration.
//! The production `nano-agent` subcommand has no matching argument, and this
//! entire module is compiled out unless `nano-wheels-off-qualification` is
//! enabled.

use std::fmt;
use std::str::FromStr;

use robot_protocol::v2::TimerPwm;
use robot_server::OperatorSupervisedCandidateSerialFaultInjection;

/// One deterministic fault session. A run can select at most one fault, which
/// prevents overlapping injected causes from making the stop evidence
/// ambiguous.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffQualificationFaultInjection {
    HostMonotonicClockRegressionOnFirstNonzeroCommand,
    PartialUartRecordOnFirstNonzeroCommand,
    StaleDepthOnFirstNonzeroCommand,
    LocalizationLossOnFirstNonzeroCommand,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WheelsOffQualificationHostClockFaultInjection {
    RegressionOnFirstNonzeroCommand,
}

/// The observable live-runtime mutation selected by a qualifier declaration.
///
/// These are synthetic software seams. Neither variant claims that a physical
/// sensor disconnected or that the OAK stopped producing frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffQualificationLiveFaultTrigger {
    StaleDepthOnFirstNonzeroCommand,
    LocalizationLossOnFirstNonzeroCommand,
}

impl WheelsOffQualificationLiveFaultTrigger {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StaleDepthOnFirstNonzeroCommand => "stale-depth-on-first-nonzero-command",
            Self::LocalizationLossOnFirstNonzeroCommand => {
                "localization-loss-on-first-nonzero-command"
            }
        }
    }

    pub const fn declaration(self) -> WheelsOffQualificationFaultInjection {
        match self {
            Self::StaleDepthOnFirstNonzeroCommand => {
                WheelsOffQualificationFaultInjection::StaleDepthOnFirstNonzeroCommand
            }
            Self::LocalizationLossOnFirstNonzeroCommand => {
                WheelsOffQualificationFaultInjection::LocalizationLossOnFirstNonzeroCommand
            }
        }
    }
}

/// One-shot live qualifier state retained from the parsed CLI declaration.
///
/// A trigger is admitted only from a controller-confirmed applied step. Zero
/// receipts do not consume it. Once triggered, the state remains latched for
/// the process lifetime so later sensor frames cannot make the injected
/// condition appear recovered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WheelsOffQualificationLiveFaultState {
    selected: Option<WheelsOffQualificationFaultInjection>,
    triggered: Option<WheelsOffQualificationLiveFaultTrigger>,
}

impl WheelsOffQualificationLiveFaultState {
    pub const fn new(selected: Option<WheelsOffQualificationFaultInjection>) -> Self {
        Self {
            selected,
            triggered: None,
        }
    }

    pub const fn selected(self) -> Option<WheelsOffQualificationFaultInjection> {
        self.selected
    }

    pub const fn triggered(self) -> Option<WheelsOffQualificationLiveFaultTrigger> {
        self.triggered
    }

    /// Return a selected declaration that cannot truthfully complete through
    /// the qualifier's normal-exit path.
    ///
    /// The two live-state seams may finish normally only after their exact
    /// trigger has latched. Clock and partial-UART seams terminate through
    /// their typed injected-fault errors, so reaching normal teardown while
    /// either is selected means it was never exercised.
    pub const fn unexercised_on_normal_exit(self) -> Option<WheelsOffQualificationFaultInjection> {
        match (self.selected, self.triggered) {
            (
                Some(WheelsOffQualificationFaultInjection::StaleDepthOnFirstNonzeroCommand),
                Some(WheelsOffQualificationLiveFaultTrigger::StaleDepthOnFirstNonzeroCommand),
            )
            | (
                Some(WheelsOffQualificationFaultInjection::LocalizationLossOnFirstNonzeroCommand),
                Some(WheelsOffQualificationLiveFaultTrigger::LocalizationLossOnFirstNonzeroCommand),
            )
            | (None, None) => None,
            (selected, _) => selected,
        }
    }

    pub const fn suppresses_depth_admission(self) -> bool {
        matches!(
            self.triggered,
            Some(WheelsOffQualificationLiveFaultTrigger::StaleDepthOnFirstNonzeroCommand)
        )
    }

    pub const fn forces_localization_lost(self) -> bool {
        matches!(
            self.triggered,
            Some(WheelsOffQualificationLiveFaultTrigger::LocalizationLossOnFirstNonzeroCommand)
        )
    }

    pub fn observe_controller_confirmed_applied_step(
        &mut self,
        actual_applied: TimerPwm,
        localized: bool,
    ) -> Result<
        Option<WheelsOffQualificationLiveFaultTrigger>,
        WheelsOffQualificationLiveFaultTriggerError,
    > {
        if actual_applied.is_zero() || self.triggered.is_some() {
            return Ok(None);
        }
        let trigger = match self.selected {
            Some(WheelsOffQualificationFaultInjection::StaleDepthOnFirstNonzeroCommand) => {
                WheelsOffQualificationLiveFaultTrigger::StaleDepthOnFirstNonzeroCommand
            }
            Some(WheelsOffQualificationFaultInjection::LocalizationLossOnFirstNonzeroCommand) => {
                if !localized {
                    return Err(
                        WheelsOffQualificationLiveFaultTriggerError::LocalizationWasNotEstablished,
                    );
                }
                WheelsOffQualificationLiveFaultTrigger::LocalizationLossOnFirstNonzeroCommand
            }
            Some(
                WheelsOffQualificationFaultInjection::HostMonotonicClockRegressionOnFirstNonzeroCommand
                | WheelsOffQualificationFaultInjection::PartialUartRecordOnFirstNonzeroCommand,
            )
            | None => return Ok(None),
        };
        self.triggered = Some(trigger);
        Ok(Some(trigger))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffQualificationLiveFaultTriggerError {
    LocalizationWasNotEstablished,
}

impl WheelsOffQualificationLiveFaultTriggerError {
    pub const fn selected(self) -> WheelsOffQualificationFaultInjection {
        match self {
            Self::LocalizationWasNotEstablished => {
                WheelsOffQualificationFaultInjection::LocalizationLossOnFirstNonzeroCommand
            }
        }
    }
}

impl fmt::Display for WheelsOffQualificationLiveFaultTriggerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LocalizationWasNotEstablished => formatter.write_str(
                "localization-loss-on-first-nonzero-command cannot trigger because localization was not established at the controller-confirmed nonzero applied step",
            ),
        }
    }
}

impl std::error::Error for WheelsOffQualificationLiveFaultTriggerError {}

impl WheelsOffQualificationFaultInjection {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HostMonotonicClockRegressionOnFirstNonzeroCommand => {
                "host-monotonic-clock-regression-on-first-nonzero-command"
            }
            Self::PartialUartRecordOnFirstNonzeroCommand => {
                "partial-uart-record-on-first-nonzero-command"
            }
            Self::StaleDepthOnFirstNonzeroCommand => "stale-depth-on-first-nonzero-command",
            Self::LocalizationLossOnFirstNonzeroCommand => {
                "localization-loss-on-first-nonzero-command"
            }
        }
    }

    pub(crate) const fn host_clock_fault(
        self,
    ) -> Option<WheelsOffQualificationHostClockFaultInjection> {
        match self {
            Self::HostMonotonicClockRegressionOnFirstNonzeroCommand => {
                Some(WheelsOffQualificationHostClockFaultInjection::RegressionOnFirstNonzeroCommand)
            }
            Self::PartialUartRecordOnFirstNonzeroCommand
            | Self::StaleDepthOnFirstNonzeroCommand
            | Self::LocalizationLossOnFirstNonzeroCommand => None,
        }
    }

    pub(crate) const fn serial_fault(
        self,
    ) -> Option<OperatorSupervisedCandidateSerialFaultInjection> {
        match self {
            Self::HostMonotonicClockRegressionOnFirstNonzeroCommand
            | Self::StaleDepthOnFirstNonzeroCommand
            | Self::LocalizationLossOnFirstNonzeroCommand => None,
            Self::PartialUartRecordOnFirstNonzeroCommand => Some(
                OperatorSupervisedCandidateSerialFaultInjection::PartialUartRecordOnFirstNonzeroCommand,
            ),
        }
    }
}

impl FromStr for WheelsOffQualificationFaultInjection {
    type Err = WheelsOffQualificationFaultInjectionParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "host-monotonic-clock-regression-on-first-nonzero-command" => {
                Ok(Self::HostMonotonicClockRegressionOnFirstNonzeroCommand)
            }
            "partial-uart-record-on-first-nonzero-command" => {
                Ok(Self::PartialUartRecordOnFirstNonzeroCommand)
            }
            "stale-depth-on-first-nonzero-command" => Ok(Self::StaleDepthOnFirstNonzeroCommand),
            "localization-loss-on-first-nonzero-command" => {
                Ok(Self::LocalizationLossOnFirstNonzeroCommand)
            }
            _ => Err(WheelsOffQualificationFaultInjectionParseError),
        }
    }
}

impl fmt::Display for WheelsOffQualificationFaultInjection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WheelsOffQualificationFaultInjectionParseError;

impl fmt::Display for WheelsOffQualificationFaultInjectionParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(
            "fault injection must be exactly host-monotonic-clock-regression-on-first-nonzero-command, partial-uart-record-on-first-nonzero-command, stale-depth-on-first-nonzero-command, or localization-loss-on-first-nonzero-command",
        )
    }
}

impl std::error::Error for WheelsOffQualificationFaultInjectionParseError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declarations_parse_once_into_closed_fault_kinds() {
        for expected in [
            WheelsOffQualificationFaultInjection::HostMonotonicClockRegressionOnFirstNonzeroCommand,
            WheelsOffQualificationFaultInjection::PartialUartRecordOnFirstNonzeroCommand,
            WheelsOffQualificationFaultInjection::StaleDepthOnFirstNonzeroCommand,
            WheelsOffQualificationFaultInjection::LocalizationLossOnFirstNonzeroCommand,
        ] {
            assert_eq!(expected.as_str().parse(), Ok(expected));
            assert_eq!(expected.to_string(), expected.as_str());
        }
        assert!(
            "clock-regression"
                .parse::<WheelsOffQualificationFaultInjection>()
                .is_err()
        );
        assert!(
            "partial-uart-record-on-first-nonzero-command=3"
                .parse::<WheelsOffQualificationFaultInjection>()
                .is_err()
        );
    }

    #[test]
    fn each_declaration_selects_exactly_one_runtime_seam() {
        let clock =
            WheelsOffQualificationFaultInjection::HostMonotonicClockRegressionOnFirstNonzeroCommand;
        assert_eq!(
            clock.host_clock_fault(),
            Some(WheelsOffQualificationHostClockFaultInjection::RegressionOnFirstNonzeroCommand)
        );
        assert_eq!(clock.serial_fault(), None);

        let serial = WheelsOffQualificationFaultInjection::PartialUartRecordOnFirstNonzeroCommand;
        assert_eq!(serial.host_clock_fault(), None);
        assert_eq!(
            serial.serial_fault(),
            Some(
                OperatorSupervisedCandidateSerialFaultInjection::PartialUartRecordOnFirstNonzeroCommand
            )
        );

        for live in [
            WheelsOffQualificationFaultInjection::StaleDepthOnFirstNonzeroCommand,
            WheelsOffQualificationFaultInjection::LocalizationLossOnFirstNonzeroCommand,
        ] {
            assert_eq!(live.host_clock_fault(), None);
            assert_eq!(live.serial_fault(), None);
        }
    }

    #[test]
    fn live_faults_wait_for_a_confirmed_nonzero_step_and_latch_once() {
        let zero = TimerPwm::ZERO;
        let nonzero = TimerPwm::try_new(1, 0).expect("fixture PWM is valid");
        let mut depth = WheelsOffQualificationLiveFaultState::new(Some(
            WheelsOffQualificationFaultInjection::StaleDepthOnFirstNonzeroCommand,
        ));
        assert_eq!(
            depth.observe_controller_confirmed_applied_step(zero, false),
            Ok(None)
        );
        assert_eq!(
            depth.observe_controller_confirmed_applied_step(nonzero, false),
            Ok(Some(
                WheelsOffQualificationLiveFaultTrigger::StaleDepthOnFirstNonzeroCommand
            ))
        );
        assert!(depth.suppresses_depth_admission());
        assert_eq!(
            depth.observe_controller_confirmed_applied_step(nonzero, true),
            Ok(None)
        );
        assert_eq!(
            depth.triggered(),
            Some(WheelsOffQualificationLiveFaultTrigger::StaleDepthOnFirstNonzeroCommand)
        );
        assert_eq!(depth.unexercised_on_normal_exit(), None);
    }

    #[test]
    fn localization_fault_requires_a_preexisting_localized_state() {
        let nonzero = TimerPwm::try_new(1, 0).expect("fixture PWM is valid");
        let mut rejected = WheelsOffQualificationLiveFaultState::new(Some(
            WheelsOffQualificationFaultInjection::LocalizationLossOnFirstNonzeroCommand,
        ));
        assert_eq!(
            rejected.observe_controller_confirmed_applied_step(nonzero, false),
            Err(WheelsOffQualificationLiveFaultTriggerError::LocalizationWasNotEstablished)
        );
        assert_eq!(rejected.triggered(), None);

        let mut admitted = WheelsOffQualificationLiveFaultState::new(Some(
            WheelsOffQualificationFaultInjection::LocalizationLossOnFirstNonzeroCommand,
        ));
        assert_eq!(
            admitted.observe_controller_confirmed_applied_step(nonzero, true),
            Ok(Some(
                WheelsOffQualificationLiveFaultTrigger::LocalizationLossOnFirstNonzeroCommand
            ))
        );
        assert!(admitted.forces_localization_lost());
        assert_eq!(admitted.unexercised_on_normal_exit(), None);
    }

    #[test]
    fn normal_exit_rejects_every_selected_fault_that_was_not_exercised() {
        for selected in [
            WheelsOffQualificationFaultInjection::HostMonotonicClockRegressionOnFirstNonzeroCommand,
            WheelsOffQualificationFaultInjection::PartialUartRecordOnFirstNonzeroCommand,
            WheelsOffQualificationFaultInjection::StaleDepthOnFirstNonzeroCommand,
            WheelsOffQualificationFaultInjection::LocalizationLossOnFirstNonzeroCommand,
        ] {
            assert_eq!(
                WheelsOffQualificationLiveFaultState::new(Some(selected))
                    .unexercised_on_normal_exit(),
                Some(selected)
            );
        }
        assert_eq!(
            WheelsOffQualificationLiveFaultState::new(None).unexercised_on_normal_exit(),
            None
        );
    }
}
