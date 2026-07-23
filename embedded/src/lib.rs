#![no_std]

pub mod attended_wheel_on_commissioning;
pub mod boot_journal;
pub mod controller;
pub mod motor;
pub mod motor_inert_profile;
pub mod provisional_four_pwm;
pub mod transport_diagnostic;
pub mod transport_scheduler;
pub mod watchdog_gate;

use robot_protocol::{
    PwmPercent,
    v2::{
        ControllerCapabilities, ControllerSessionAdmission, ControllerSessionClass,
        ControllerSessionReadiness, MaxAbsPwmPercent, PhysicalStopSemantics, ReadinessFlags,
    },
};

/// Independent-watchdog period advertised by the canonical V2 firmware.
///
/// Both the motor-inert and operator-supervised candidate images use this
/// exact period. Deployment manifests must match it; it is not a tunable
/// transport timeout.
pub const FIRMWARE_V2_WATCHDOG_NOMINAL_PERIOD_MS: u16 = 250;

/// Derive the only truthful readiness flags for an admitted established
/// session.
///
/// A stopped session deliberately omits `DEADLINE_ARMED`; a live transition or
/// drive must include it. Motion-disabled or internally inconsistent profile
/// claims do not produce established-session readiness.
pub fn established_session_readiness(
    capabilities: ControllerCapabilities,
    max_abs_pwm_percent: MaxAbsPwmPercent,
    physical_stop_semantics: PhysicalStopSemantics,
    deadline_armed: bool,
) -> Option<ReadinessFlags> {
    let session_class = match capabilities
        .classify_session_admission(max_abs_pwm_percent, physical_stop_semantics)
        .ok()?
    {
        ControllerSessionAdmission::OperatorSupervisedFourPwmCandidate => {
            ControllerSessionClass::OperatorSupervisedFourPwmCandidate
        }
        ControllerSessionAdmission::AttendedWheelOnCommissioning => {
            ControllerSessionClass::AttendedWheelOnCommissioning
        }
        ControllerSessionAdmission::ProductionExternalInterlocks => {
            ControllerSessionClass::ProductionExternalInterlocks
        }
        ControllerSessionAdmission::MotionDisabled => return None,
    };
    Some(ReadinessFlags::for_established_session(
        session_class,
        if deadline_armed {
            ControllerSessionReadiness::DeadlineArmed
        } else {
            ControllerSessionReadiness::Stopped
        },
    ))
}

/// Applies one pending timer update using the timer's currently reported direction.
///
/// The direction is an assumption: a quadrature input can reverse after setting
/// the update flag and before this snapshot reads the direction bit.
pub const fn encoder_wraps_with_pending_direction_assumption(
    committed: i64,
    update_pending: bool,
    counting_down: bool,
) -> i64 {
    if !update_pending {
        committed
    } else if counting_down {
        committed.wrapping_sub(1)
    } else {
        committed.wrapping_add(1)
    }
}

pub fn pwm_duty(pwm_percent: PwmPercent, maximum_duty: u16) -> u16 {
    let scaled = u32::from(pwm_percent.get().unsigned_abs()) * u32::from(maximum_duty) / 100;
    // PwmPercent proves an absolute value at most 100, so scaled cannot exceed
    // maximum_duty and the narrowing conversion preserves the value.
    scaled as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_protocol::v2::MAX_OPERATOR_SUPERVISED_FOUR_PWM_PWM_PERCENT;

    #[test]
    fn pending_encoder_wrap_assumption_is_directional_and_wraps_explicitly() {
        assert_eq!(
            encoder_wraps_with_pending_direction_assumption(10, false, false),
            10
        );
        assert_eq!(
            encoder_wraps_with_pending_direction_assumption(10, true, false),
            11
        );
        assert_eq!(
            encoder_wraps_with_pending_direction_assumption(10, true, true),
            9
        );
        assert_eq!(
            encoder_wraps_with_pending_direction_assumption(i64::MAX, true, false),
            i64::MIN
        );
        assert_eq!(
            encoder_wraps_with_pending_direction_assumption(i64::MIN, true, true),
            i64::MAX
        );
    }

    #[test]
    fn pwm_scaling_is_bounded_for_every_valid_integer_percent() {
        for raw in -100..=100 {
            let pwm = PwmPercent::try_new(raw).expect("loop covers the complete valid domain");
            for maximum_duty in [0, 1, 99, 100, 1_000, u16::MAX] {
                assert!(pwm_duty(pwm, maximum_duty) <= maximum_duty);
            }
        }
        assert_eq!(pwm_duty(PwmPercent::ZERO, u16::MAX), 0);
        assert_eq!(
            pwm_duty(PwmPercent::try_new(100).expect("valid endpoint"), u16::MAX),
            u16::MAX
        );
        assert_eq!(
            pwm_duty(PwmPercent::try_new(-100).expect("valid endpoint"), u16::MAX),
            u16::MAX
        );
    }

    #[test]
    fn established_candidate_readiness_distinguishes_stopped_from_live_deadline() {
        let capabilities = ControllerCapabilities::try_from_bits(
            ControllerCapabilities::SOFTWARE_GUARD_BITS
                | ControllerCapabilities::OPERATOR_SUPERVISED_FOUR_PWM_CANDIDATE,
        )
        .expect("candidate capability contract");
        let maximum =
            MaxAbsPwmPercent::try_new(MAX_OPERATOR_SUPERVISED_FOUR_PWM_PWM_PERCENT).expect("cap");

        let stopped = established_session_readiness(
            capabilities,
            maximum,
            PhysicalStopSemantics::Unverified,
            false,
        )
        .expect("candidate session is admitted");
        let live = established_session_readiness(
            capabilities,
            maximum,
            PhysicalStopSemantics::Unverified,
            true,
        )
        .expect("candidate session is admitted");

        assert!(stopped.is_stopped_ready_for_session(
            ControllerSessionClass::OperatorSupervisedFourPwmCandidate
        ));
        assert!(!stopped.is_deadline_ready_for_session(
            ControllerSessionClass::OperatorSupervisedFourPwmCandidate
        ));
        assert!(live.is_deadline_ready_for_session(
            ControllerSessionClass::OperatorSupervisedFourPwmCandidate
        ));
        assert!(!live.is_stopped_ready_for_session(
            ControllerSessionClass::OperatorSupervisedFourPwmCandidate
        ));
    }

    #[test]
    fn no_established_readiness_is_minted_for_a_motion_disabled_profile() {
        let capabilities =
            ControllerCapabilities::try_from_bits(ControllerCapabilities::SOFTWARE_GUARD_BITS)
                .expect("software guard capability contract");
        let zero = MaxAbsPwmPercent::try_new(0).expect("zero is the motion-disabled cap");

        assert_eq!(
            established_session_readiness(
                capabilities,
                zero,
                PhysicalStopSemantics::Unverified,
                false,
            ),
            None
        );
    }
}
