//! Typed, one-shot faults available only in the manually invoked wheels-off
//! qualifier.
//!
//! These declarations are command-line inputs, not deployment configuration.
//! The production `nano-agent` subcommand has no matching argument, and this
//! entire module is compiled out unless `nano-wheels-off-qualification` is
//! enabled.

use std::fmt;
use std::str::FromStr;

use robot_server::OperatorSupervisedCandidateSerialFaultInjection;

/// One deterministic fault session. A run can select at most one fault, which
/// prevents overlapping injected causes from making the stop evidence
/// ambiguous.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffQualificationFaultInjection {
    HostMonotonicClockRegressionOnFirstNonzeroCommand,
    PartialUartRecordOnFirstNonzeroCommand,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WheelsOffQualificationHostClockFaultInjection {
    RegressionOnFirstNonzeroCommand,
}

impl WheelsOffQualificationFaultInjection {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HostMonotonicClockRegressionOnFirstNonzeroCommand => {
                "host-monotonic-clock-regression-on-first-nonzero-command"
            }
            Self::PartialUartRecordOnFirstNonzeroCommand => {
                "partial-uart-record-on-first-nonzero-command"
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
            Self::PartialUartRecordOnFirstNonzeroCommand => None,
        }
    }

    pub(crate) const fn serial_fault(
        self,
    ) -> Option<OperatorSupervisedCandidateSerialFaultInjection> {
        match self {
            Self::HostMonotonicClockRegressionOnFirstNonzeroCommand => None,
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
            "fault injection must be exactly host-monotonic-clock-regression-on-first-nonzero-command or partial-uart-record-on-first-nonzero-command",
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
    }
}
