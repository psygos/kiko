//! Exact contract for the default motor-inert KRP2 firmware profile.

use robot_protocol::v2::{
    ActuatorConfigFingerprint, ControllerCapabilities, ControllerSessionAdmission, DomainError,
    MaxAbsPwmPercent, PhysicalStopSemantics,
};

use crate::motor::ActuatorEnvelope;

pub const MOTOR_INERT_FIRMWARE_BUILD_ID: u32 = 0x0002_0002;
pub const MOTOR_INERT_FINGERPRINT_BYTES: [u8; 16] = *b"KIKO-NO-ACT-V1!!";
pub const MOTOR_INERT_CAPABILITY_BITS: u32 = ControllerCapabilities::SOFTWARE_GUARD_BITS
    | ControllerCapabilities::MOTOR_INERT_TRANSPORT_DIAGNOSTICS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MotorInertProfile {
    fingerprint: ActuatorConfigFingerprint,
    capabilities: ControllerCapabilities,
    max_abs_pwm: MaxAbsPwmPercent,
}

impl MotorInertProfile {
    pub fn try_new() -> Result<Self, DomainError> {
        let fingerprint = ActuatorConfigFingerprint::try_new(MOTOR_INERT_FINGERPRINT_BYTES)?;
        let capabilities = ControllerCapabilities::try_from_bits(MOTOR_INERT_CAPABILITY_BITS)?;
        let max_abs_pwm = MaxAbsPwmPercent::try_new(0)?;
        if capabilities
            .classify_session_admission(max_abs_pwm, PhysicalStopSemantics::Unverified)?
            != ControllerSessionAdmission::MotionDisabled
        {
            return Err(DomainError::MotionAuthorityWithoutAdmissibleSafetyClass {
                bits: capabilities.bits(),
                max_abs_pwm_percent: max_abs_pwm.get(),
            });
        }
        Ok(Self {
            fingerprint,
            capabilities,
            max_abs_pwm,
        })
    }

    pub const fn firmware_build_id(self) -> u32 {
        MOTOR_INERT_FIRMWARE_BUILD_ID
    }

    pub const fn fingerprint(self) -> ActuatorConfigFingerprint {
        self.fingerprint
    }

    pub const fn envelope(self) -> ActuatorEnvelope {
        ActuatorEnvelope::Unvalidated
    }

    pub const fn capabilities(self) -> ControllerCapabilities {
        self.capabilities
    }

    pub const fn max_abs_pwm(self) -> MaxAbsPwmPercent {
        self.max_abs_pwm
    }

    pub const fn physical_stop_semantics(self) -> PhysicalStopSemantics {
        PhysicalStopSemantics::Unverified
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_default_identity_remains_motor_inert_and_diagnostic_only() {
        let profile = MotorInertProfile::try_new().expect("canonical default profile");
        assert_eq!(profile.firmware_build_id(), 0x0002_0002);
        assert_eq!(profile.fingerprint().as_bytes(), b"KIKO-NO-ACT-V1!!");
        assert_eq!(profile.max_abs_pwm().get(), 0);
        assert!(!profile.max_abs_pwm().grants_motion_authority());
        assert_eq!(profile.envelope(), ActuatorEnvelope::Unvalidated);
        assert!(
            profile
                .capabilities()
                .supports_motor_inert_transport_diagnostics()
        );
        assert!(
            !profile
                .capabilities()
                .supports_operator_supervised_four_pwm_candidate()
        );
        assert!(!profile.capabilities().supports_required_safety());
        assert_eq!(
            profile.capabilities().classify_session_admission(
                profile.max_abs_pwm(),
                profile.physical_stop_semantics()
            ),
            Ok(ControllerSessionAdmission::MotionDisabled)
        );
    }
}
