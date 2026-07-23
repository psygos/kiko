//! Pure contract for the attended wheel-on commissioning firmware image.
//!
//! This is a non-production, physical-stop-unverified profile. It binds a
//! distinct firmware identity to the evidenced PA0/PA1 + PB4/PB5 timer
//! topology, a hard ±20% timer-duty range, and an explicit command-step bound.
//! It does not claim that a duty produces motion, velocity, torque, or a
//! particular wheel direction.

pub use robot_protocol::v2::{
    ATTENDED_WHEEL_ON_COMMISSIONING_FINGERPRINT_BYTES,
    ATTENDED_WHEEL_ON_COMMISSIONING_FIRMWARE_BUILD_ID,
};
use robot_protocol::v2::{
    ActuatorConfigFingerprint, ControllerCapabilities, ControllerSessionAdmission, DomainError,
    MAX_ATTENDED_WHEEL_ON_COMMISSIONING_PWM_PERCENT, MaxAbsPwmPercent, PhysicalStopSemantics,
};

use crate::motor::{
    ActuatorEnvelope, AttendedWheelOnFourPwmEnvelope, ProvisionalFourPwmEnvelopeError,
};

pub const ATTENDED_WHEEL_ON_COMMISSIONING_CAPABILITY_BITS: u32 =
    ControllerCapabilities::SOFTWARE_GUARD_BITS
        | ControllerCapabilities::ATTENDED_WHEEL_ON_COMMISSIONING;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttendedWheelOnCommissioningProfile {
    fingerprint: ActuatorConfigFingerprint,
    envelope: ActuatorEnvelope,
    capabilities: ControllerCapabilities,
    max_abs_pwm: MaxAbsPwmPercent,
    physical_stop_semantics: PhysicalStopSemantics,
}

impl AttendedWheelOnCommissioningProfile {
    pub fn try_new() -> Result<Self, AttendedWheelOnCommissioningProfileError> {
        let fingerprint =
            ActuatorConfigFingerprint::try_new(ATTENDED_WHEEL_ON_COMMISSIONING_FINGERPRINT_BYTES)
                .map_err(AttendedWheelOnCommissioningProfileError::Protocol)?;
        let capabilities =
            ControllerCapabilities::try_from_bits(ATTENDED_WHEEL_ON_COMMISSIONING_CAPABILITY_BITS)
                .map_err(AttendedWheelOnCommissioningProfileError::Protocol)?;
        let max_abs_pwm =
            MaxAbsPwmPercent::try_new(MAX_ATTENDED_WHEEL_ON_COMMISSIONING_PWM_PERCENT)
                .map_err(AttendedWheelOnCommissioningProfileError::Protocol)?;
        let envelope = AttendedWheelOnFourPwmEnvelope::try_new(fingerprint)
            .map_err(AttendedWheelOnCommissioningProfileError::Envelope)?
            .into_actuator_envelope();
        let physical_stop_semantics = PhysicalStopSemantics::Unverified;
        let admission = capabilities
            .classify_session_admission(max_abs_pwm, physical_stop_semantics)
            .map_err(AttendedWheelOnCommissioningProfileError::Protocol)?;
        if admission != ControllerSessionAdmission::AttendedWheelOnCommissioning {
            return Err(AttendedWheelOnCommissioningProfileError::AdmissionInvariant);
        }
        Ok(Self {
            fingerprint,
            envelope,
            capabilities,
            max_abs_pwm,
            physical_stop_semantics,
        })
    }

    pub const fn firmware_build_id(self) -> u32 {
        ATTENDED_WHEEL_ON_COMMISSIONING_FIRMWARE_BUILD_ID
    }

    pub const fn fingerprint(self) -> ActuatorConfigFingerprint {
        self.fingerprint
    }

    pub const fn envelope(self) -> ActuatorEnvelope {
        self.envelope
    }

    pub const fn capabilities(self) -> ControllerCapabilities {
        self.capabilities
    }

    pub const fn max_abs_pwm(self) -> MaxAbsPwmPercent {
        self.max_abs_pwm
    }

    pub const fn physical_stop_semantics(self) -> PhysicalStopSemantics {
        self.physical_stop_semantics
    }

    pub fn grants_attended_session(self, per_boot_identity_is_session_unique: bool) -> bool {
        per_boot_identity_is_session_unique
            && self.envelope.is_attended_wheel_on_commissioning()
            && self
                .capabilities
                .classify_session_admission(self.max_abs_pwm, self.physical_stop_semantics)
                == Ok(ControllerSessionAdmission::AttendedWheelOnCommissioning)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttendedWheelOnCommissioningProfileError {
    Protocol(DomainError),
    Envelope(ProvisionalFourPwmEnvelopeError),
    AdmissionInvariant,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motor::ATTENDED_WHEEL_ON_MAX_COMMAND_STEP_PERCENT;
    use robot_protocol::{
        PwmPercent,
        v2::{
            ATTENDED_WHEEL_ON_COMMISSIONING_MAX_COMMAND_STEP_PERCENT, ControllerSessionClass,
            ReadinessFlags, TimerPwm,
        },
    };

    fn pair(left: i8, right: i8) -> crate::motor::PwmPair {
        crate::motor::PwmPair::from_validated(
            PwmPercent::try_new(left).expect("left PWM"),
            PwmPercent::try_new(right).expect("right PWM"),
        )
    }

    #[test]
    fn identity_class_and_bounds_are_exact_and_non_production() {
        let profile =
            AttendedWheelOnCommissioningProfile::try_new().expect("canonical attended profile");
        assert_eq!(profile.firmware_build_id(), 0x0002_2001);
        assert_eq!(profile.fingerprint().as_bytes(), b"KIKO-WHEELON-CM1");
        assert_eq!(
            profile.max_abs_pwm().get(),
            MAX_ATTENDED_WHEEL_ON_COMMISSIONING_PWM_PERCENT
        );
        assert_eq!(
            profile.physical_stop_semantics(),
            PhysicalStopSemantics::Unverified
        );
        assert!(
            profile
                .capabilities()
                .supports_attended_wheel_on_commissioning()
        );
        assert!(!profile.capabilities().supports_required_safety());
        assert_eq!(
            ATTENDED_WHEEL_ON_MAX_COMMAND_STEP_PERCENT,
            ATTENDED_WHEEL_ON_COMMISSIONING_MAX_COMMAND_STEP_PERCENT
        );
    }

    #[test]
    fn unique_boot_identity_is_required_for_motion_admission() {
        let profile =
            AttendedWheelOnCommissioningProfile::try_new().expect("canonical attended profile");
        assert!(profile.grants_attended_session(true));
        assert!(!profile.grants_attended_session(false));
    }

    #[test]
    fn envelope_rejects_above_cap_without_clamping_and_zero_always_stops() {
        let profile =
            AttendedWheelOnCommissioningProfile::try_new().expect("canonical attended profile");
        let envelope = profile.envelope();
        assert_eq!(
            envelope.validate_transition(pair(20, -20), pair(0, 0)),
            Ok(())
        );
        assert!(
            envelope
                .validate_transition(pair(0, 0), pair(21, 0))
                .is_err()
        );
        assert!(
            envelope
                .validate_transition(pair(0, 0), pair(0, -21))
                .is_err()
        );
    }

    #[test]
    fn readiness_bit_pattern_cannot_masquerade_as_production_or_candidate() {
        let attended = ReadinessFlags::for_established_session(
            ControllerSessionClass::AttendedWheelOnCommissioning,
            robot_protocol::v2::ControllerSessionReadiness::DeadlineArmed,
        );
        assert!(
            attended.is_deadline_ready_for_session(
                ControllerSessionClass::AttendedWheelOnCommissioning
            )
        );
        assert!(
            !attended.is_deadline_ready_for_session(
                ControllerSessionClass::ProductionExternalInterlocks
            )
        );
        assert!(!attended.is_deadline_ready_for_session(
            ControllerSessionClass::OperatorSupervisedFourPwmCandidate
        ));
        assert_eq!(TimerPwm::ZERO.left().get(), 0);
    }
}
