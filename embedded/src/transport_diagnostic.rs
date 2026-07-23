//! Pure fail-closed admission for motor-inert KRP2 transport probes.

use robot_protocol::v2::{ControllerFaults, OutputState, TimerPwm, TransportDiagnosticResultCode};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransportDiagnosticGateSnapshot {
    pub identity_matches: bool,
    pub capability_available: bool,
    pub profile_grants_motion_authority: bool,
    pub session_active: bool,
    pub output_state: OutputState,
    pub timer_pwm: TimerPwm,
    pub faults: ControllerFaults,
}

impl TransportDiagnosticGateSnapshot {
    /// Applies the frozen denial precedence used by the hardware adapter.
    ///
    /// An accepted probe is observational only. It does not create a control
    /// epoch, arm a deadline, or authorize an output transition.
    pub const fn classify(self) -> TransportDiagnosticResultCode {
        if !self.identity_matches {
            TransportDiagnosticResultCode::DeniedIdentityMismatch
        } else if !self.capability_available {
            TransportDiagnosticResultCode::DeniedCapabilityUnavailable
        } else if self.profile_grants_motion_authority {
            TransportDiagnosticResultCode::DeniedMotionCapableProfile
        } else if self.session_active {
            TransportDiagnosticResultCode::DeniedSessionActive
        } else if !self.output_state.is_safe() || !self.timer_pwm.is_zero() {
            TransportDiagnosticResultCode::DeniedUnsafeOutput
        } else if !self.faults.is_clear() {
            TransportDiagnosticResultCode::DeniedControllerFault
        } else {
            TransportDiagnosticResultCode::EchoedMotorInert
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_protocol::v2::ControllerFaults;

    fn safe_snapshot() -> TransportDiagnosticGateSnapshot {
        TransportDiagnosticGateSnapshot {
            identity_matches: true,
            capability_available: true,
            profile_grants_motion_authority: false,
            session_active: false,
            output_state: OutputState::Disabled,
            timer_pwm: TimerPwm::ZERO,
            faults: ControllerFaults::NONE,
        }
    }

    #[test]
    fn only_the_exact_motor_inert_idle_safe_state_echoes() {
        assert_eq!(
            safe_snapshot().classify(),
            TransportDiagnosticResultCode::EchoedMotorInert
        );

        let mutations = [
            (
                TransportDiagnosticGateSnapshot {
                    identity_matches: false,
                    ..safe_snapshot()
                },
                TransportDiagnosticResultCode::DeniedIdentityMismatch,
            ),
            (
                TransportDiagnosticGateSnapshot {
                    capability_available: false,
                    ..safe_snapshot()
                },
                TransportDiagnosticResultCode::DeniedCapabilityUnavailable,
            ),
            (
                TransportDiagnosticGateSnapshot {
                    profile_grants_motion_authority: true,
                    ..safe_snapshot()
                },
                TransportDiagnosticResultCode::DeniedMotionCapableProfile,
            ),
            (
                TransportDiagnosticGateSnapshot {
                    session_active: true,
                    ..safe_snapshot()
                },
                TransportDiagnosticResultCode::DeniedSessionActive,
            ),
            (
                TransportDiagnosticGateSnapshot {
                    output_state: OutputState::NonzeroPwm,
                    timer_pwm: TimerPwm::try_new(1, 0).expect("valid nonzero PWM"),
                    ..safe_snapshot()
                },
                TransportDiagnosticResultCode::DeniedUnsafeOutput,
            ),
            (
                TransportDiagnosticGateSnapshot {
                    faults: ControllerFaults::try_from_bits(ControllerFaults::SERIAL_INTEGRITY)
                        .expect("known fault"),
                    ..safe_snapshot()
                },
                TransportDiagnosticResultCode::DeniedControllerFault,
            ),
        ];
        for (snapshot, expected) in mutations {
            assert_eq!(snapshot.classify(), expected);
        }
    }

    #[test]
    fn denial_precedence_is_stable_for_every_boolean_gate_combination() {
        for bits in 0_u8..64 {
            let snapshot = TransportDiagnosticGateSnapshot {
                identity_matches: bits & 1 != 0,
                capability_available: bits & 2 != 0,
                profile_grants_motion_authority: bits & 4 != 0,
                session_active: bits & 8 != 0,
                output_state: if bits & 16 == 0 {
                    OutputState::Disabled
                } else {
                    OutputState::NonzeroPwm
                },
                timer_pwm: if bits & 16 == 0 {
                    TimerPwm::ZERO
                } else {
                    TimerPwm::try_new(1, 0).expect("valid nonzero PWM")
                },
                faults: if bits & 32 == 0 {
                    ControllerFaults::NONE
                } else {
                    ControllerFaults::try_from_bits(ControllerFaults::INTERNAL)
                        .expect("known fault")
                },
            };
            let expected = if !snapshot.identity_matches {
                TransportDiagnosticResultCode::DeniedIdentityMismatch
            } else if !snapshot.capability_available {
                TransportDiagnosticResultCode::DeniedCapabilityUnavailable
            } else if snapshot.profile_grants_motion_authority {
                TransportDiagnosticResultCode::DeniedMotionCapableProfile
            } else if snapshot.session_active {
                TransportDiagnosticResultCode::DeniedSessionActive
            } else if !snapshot.output_state.is_safe() {
                TransportDiagnosticResultCode::DeniedUnsafeOutput
            } else if !snapshot.faults.is_clear() {
                TransportDiagnosticResultCode::DeniedControllerFault
            } else {
                TransportDiagnosticResultCode::EchoedMotorInert
            };
            assert_eq!(snapshot.classify(), expected);
        }
    }
}
