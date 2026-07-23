//! Pure contract for the operator-supervised four-PWM candidate firmware.
//!
//! This profile records only the currently evidenced timer-pin topology:
//! PA0/PA1 (TIM2 CH1/CH2) and PB4/PB5 (TIM3 CH1/CH2). It deliberately does
//! not claim an external driver-enable gate, a driver-fault input, wheel signs,
//! a useful-duty threshold, a PWM-to-velocity calibration, or physical
//! coast/brake semantics.

use robot_protocol::v2::{
    ActuatorConfigFingerprint, ControllerCapabilities, ControllerSessionAdmission, DomainError,
    MAX_OPERATOR_SUPERVISED_FOUR_PWM_PWM_PERCENT, MaxAbsPwmPercent, PhysicalStopSemantics,
};
pub use robot_protocol::v2::{
    OPERATOR_SUPERVISED_FOUR_PWM_FINGERPRINT_BYTES, OPERATOR_SUPERVISED_FOUR_PWM_FIRMWARE_BUILD_ID,
};

use crate::motor::{
    ActuatorEnvelope, ProvisionalBoundedFourPwmEnvelope, ProvisionalFourPwmEnvelopeError,
};

pub const OPERATOR_SUPERVISED_FOUR_PWM_CAPABILITY_BITS: u32 =
    ControllerCapabilities::SOFTWARE_GUARD_BITS
        | ControllerCapabilities::OPERATOR_SUPERVISED_FOUR_PWM_CANDIDATE;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperatorSupervisedFourPwmProfile {
    fingerprint: ActuatorConfigFingerprint,
    envelope: ActuatorEnvelope,
    capabilities: ControllerCapabilities,
    max_abs_pwm: MaxAbsPwmPercent,
    physical_stop_semantics: PhysicalStopSemantics,
}

impl OperatorSupervisedFourPwmProfile {
    pub fn try_new() -> Result<Self, OperatorSupervisedFourPwmProfileError> {
        let fingerprint =
            ActuatorConfigFingerprint::try_new(OPERATOR_SUPERVISED_FOUR_PWM_FINGERPRINT_BYTES)
                .map_err(OperatorSupervisedFourPwmProfileError::Protocol)?;
        let capabilities =
            ControllerCapabilities::try_from_bits(OPERATOR_SUPERVISED_FOUR_PWM_CAPABILITY_BITS)
                .map_err(OperatorSupervisedFourPwmProfileError::Protocol)?;
        let max_abs_pwm = MaxAbsPwmPercent::try_new(MAX_OPERATOR_SUPERVISED_FOUR_PWM_PWM_PERCENT)
            .map_err(OperatorSupervisedFourPwmProfileError::Protocol)?;
        let envelope = ProvisionalBoundedFourPwmEnvelope::try_new(fingerprint)
            .map_err(OperatorSupervisedFourPwmProfileError::Envelope)?
            .into_actuator_envelope();
        let physical_stop_semantics = PhysicalStopSemantics::Unverified;
        let admission = capabilities
            .classify_session_admission(max_abs_pwm, physical_stop_semantics)
            .map_err(OperatorSupervisedFourPwmProfileError::Protocol)?;
        if admission != ControllerSessionAdmission::OperatorSupervisedFourPwmCandidate {
            return Err(OperatorSupervisedFourPwmProfileError::AdmissionInvariant);
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
        OPERATOR_SUPERVISED_FOUR_PWM_FIRMWARE_BUILD_ID
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

    pub fn grants_operator_supervised_session(
        self,
        per_boot_identity_is_session_unique: bool,
    ) -> bool {
        per_boot_identity_is_session_unique
            && self.envelope.is_operator_supervised_four_pwm_candidate()
            && self
                .capabilities
                .classify_session_admission(self.max_abs_pwm, self.physical_stop_semantics)
                == Ok(ControllerSessionAdmission::OperatorSupervisedFourPwmCandidate)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorSupervisedFourPwmProfileError {
    Protocol(DomainError),
    Envelope(ProvisionalFourPwmEnvelopeError),
    AdmissionInvariant,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        controller::{Controller, ControllerCommand, ControllerConfig, ControllerEvent, FaultCode},
        motor::{
            DurationMs, MotorDirective, MotorTiming, PROVISIONAL_FOUR_PWM_MAX_COMMAND_STEP_PERCENT,
            PwmPair,
        },
        transport_diagnostic::TransportDiagnosticGateSnapshot,
    };
    use robot_protocol::{
        ControllerUptimeMsWrapping, PwmPercent,
        v2::{
            ControlEpoch, ControllerDeadlineMsWrapping, ControllerFaults, OutputState, TimerPwm,
            TransportDiagnosticResultCode, V2CommandLeaseMs, V2CommandSequence,
        },
    };

    fn now(value: u32) -> ControllerUptimeMsWrapping {
        ControllerUptimeMsWrapping::new(value)
    }

    fn deadline(value: u32) -> ControllerDeadlineMsWrapping {
        ControllerDeadlineMsWrapping::new(value)
    }

    fn pair(left: i8, right: i8) -> PwmPair {
        PwmPair::from_validated(
            PwmPercent::try_new(left).expect("left test PWM"),
            PwmPercent::try_new(right).expect("right test PWM"),
        )
    }

    fn candidate_controller() -> Controller {
        let profile = OperatorSupervisedFourPwmProfile::try_new().expect("canonical profile");
        Controller::new(ControllerConfig::new(
            ControlEpoch::try_new(7).expect("epoch"),
            profile.envelope(),
            V2CommandLeaseMs::try_new(100).expect("lease"),
            MotorTiming::new(
                DurationMs::try_new(2).expect("neutral hold"),
                DurationMs::try_new(1).expect("preload latch"),
            ),
        ))
    }

    fn acquire(controller: &mut Controller) {
        controller.mark_ready().expect("boot-safe");
        let step = controller.accept_command(
            ControllerCommand::acquire(
                ControlEpoch::try_new(7).expect("epoch"),
                PwmPair::STOP,
                deadline(50),
                controller.config().actuator().fingerprint(),
            ),
            now(0),
        );
        assert_eq!(step.event(), ControllerEvent::ZeroAcquisitionAccepted);
        assert_eq!(step.motor(), MotorDirective::DisableAndZero);
    }

    #[test]
    fn identity_and_capability_claims_are_exact_and_distinct_from_motor_inert_default() {
        let profile = OperatorSupervisedFourPwmProfile::try_new().expect("canonical profile");
        assert_eq!(profile.firmware_build_id(), 0x0002_1001);
        assert_eq!(profile.fingerprint().as_bytes(), b"KIKO-4PWM-CAND1!");
        assert_ne!(profile.fingerprint().as_bytes(), b"KIKO-NO-ACT-V1!!");
        assert_eq!(
            profile.capabilities().bits(),
            OPERATOR_SUPERVISED_FOUR_PWM_CAPABILITY_BITS
        );
        assert!(
            profile
                .capabilities()
                .supports_operator_supervised_four_pwm_candidate()
        );
        assert!(!profile.capabilities().supports_required_safety());
        assert!(
            !profile
                .capabilities()
                .supports_motor_inert_transport_diagnostics()
        );
        assert_eq!(
            profile.max_abs_pwm().get(),
            MAX_OPERATOR_SUPERVISED_FOUR_PWM_PWM_PERCENT
        );
        assert_eq!(
            profile.physical_stop_semantics(),
            PhysicalStopSemantics::Unverified
        );
        assert_eq!(PROVISIONAL_FOUR_PWM_MAX_COMMAND_STEP_PERCENT, 5);
    }

    #[test]
    fn checked_in_candidate_manifest_matches_compiled_firmware_contract() {
        let manifest: serde_json::Value = serde_json::from_str(include_str!(
            "../../configs/nano-wheels-off-qualification-template/controller-server-candidate-v2.json.template"
        ))
        .expect("checked-in candidate manifest template is JSON");
        let profile = OperatorSupervisedFourPwmProfile::try_new().expect("canonical profile");

        assert_eq!(
            manifest["firmware_build_id"].as_u64(),
            Some(u64::from(profile.firmware_build_id()))
        );
        assert_eq!(
            manifest["actuator_config_fingerprint_hex"].as_str(),
            Some("4b494b4f2d3450574d2d43414e443121")
        );
        assert_eq!(
            manifest["expected_max_abs_pwm_percent"].as_u64(),
            Some(u64::from(profile.max_abs_pwm().get()))
        );
        assert_eq!(
            manifest["expected_watchdog_nominal_timeout_ms"].as_u64(),
            Some(u64::from(crate::FIRMWARE_V2_WATCHDOG_NOMINAL_PERIOD_MS))
        );
        assert_eq!(
            manifest["expected_physical_stop_semantics"].as_str(),
            Some("unverified")
        );
        assert_eq!(
            manifest["controller_session_class"].as_str(),
            Some("operator_supervised_four_pwm_candidate")
        );
    }

    #[test]
    fn only_a_session_unique_boot_identity_completes_candidate_grant_conditions() {
        let profile = OperatorSupervisedFourPwmProfile::try_new().expect("canonical profile");
        assert!(profile.grants_operator_supervised_session(true));
        assert!(!profile.grants_operator_supervised_session(false));
    }

    #[test]
    fn candidate_never_advertises_or_echoes_motor_inert_diagnostics() {
        let profile = OperatorSupervisedFourPwmProfile::try_new().expect("canonical profile");
        assert!(
            !profile
                .capabilities()
                .supports_motor_inert_transport_diagnostics()
        );
        let result = TransportDiagnosticGateSnapshot {
            identity_matches: true,
            capability_available: profile
                .capabilities()
                .supports_motor_inert_transport_diagnostics(),
            profile_grants_motion_authority: profile.max_abs_pwm().grants_motion_authority(),
            session_active: false,
            output_state: OutputState::Disabled,
            timer_pwm: TimerPwm::ZERO,
            faults: ControllerFaults::NONE,
        }
        .classify();
        assert_eq!(
            result,
            TransportDiagnosticResultCode::DeniedCapabilityUnavailable
        );
    }

    #[test]
    fn candidate_zero_and_lease_expiry_bypass_motion_immediately() {
        let mut zero = candidate_controller();
        acquire(&mut zero);
        let started = zero.accept_command(
            ControllerCommand::apply(
                ControlEpoch::try_new(7).expect("epoch"),
                V2CommandSequence::new(1),
                pair(5, 5),
                deadline(20),
            ),
            now(1),
        );
        assert_eq!(started.event(), ControllerEvent::TransitionStarted);
        let stopped = zero.accept_command(
            ControllerCommand::apply(
                ControlEpoch::try_new(7).expect("epoch"),
                V2CommandSequence::new(2),
                PwmPair::STOP,
                deadline(20),
            ),
            now(1),
        );
        assert_eq!(stopped.event(), ControllerEvent::StopApplied);
        assert_eq!(stopped.motor(), MotorDirective::DisableAndZero);

        let mut expiry = candidate_controller();
        acquire(&mut expiry);
        let started = expiry.accept_command(
            ControllerCommand::apply(
                ControlEpoch::try_new(7).expect("epoch"),
                V2CommandSequence::new(1),
                pair(5, -5),
                deadline(3),
            ),
            now(1),
        );
        assert_eq!(started.event(), ControllerEvent::TransitionStarted);
        assert_eq!(expiry.tick(now(2)).event(), ControllerEvent::MotionApplied);
        let expired = expiry.tick(now(3));
        assert_eq!(
            expired.event(),
            ControllerEvent::FaultLatched(FaultCode::CommandLeaseExpired)
        );
        assert_eq!(expired.motor(), MotorDirective::DisableAndZero);
    }

    #[test]
    fn candidate_slew_or_direction_bypass_attempt_faults_to_zero() {
        let mut excessive_slew = candidate_controller();
        acquire(&mut excessive_slew);
        let excessive = excessive_slew.accept_command(
            ControllerCommand::apply(
                ControlEpoch::try_new(7).expect("epoch"),
                V2CommandSequence::new(1),
                pair(6, 0),
                deadline(20),
            ),
            now(1),
        );
        assert!(matches!(
            excessive.event(),
            ControllerEvent::FaultLatched(FaultCode::MotionEnvelope(_))
        ));
        assert_eq!(excessive.motor(), MotorDirective::DisableAndZero);

        let mut sign_change = candidate_controller();
        acquire(&mut sign_change);
        let started = sign_change.accept_command(
            ControllerCommand::apply(
                ControlEpoch::try_new(7).expect("epoch"),
                V2CommandSequence::new(1),
                pair(5, 0),
                deadline(20),
            ),
            now(1),
        );
        assert_eq!(started.event(), ControllerEvent::TransitionStarted);
        assert_eq!(
            sign_change.tick(now(2)).event(),
            ControllerEvent::MotionApplied
        );
        let reversal = sign_change.accept_command(
            ControllerCommand::apply(
                ControlEpoch::try_new(7).expect("epoch"),
                V2CommandSequence::new(2),
                pair(-1, 0),
                deadline(20),
            ),
            now(3),
        );
        assert!(matches!(
            reversal.event(),
            ControllerEvent::FaultLatched(FaultCode::MotionEnvelope(_))
        ));
        assert_eq!(reversal.motor(), MotorDirective::DisableAndZero);
    }

    #[test]
    fn in_envelope_candidate_direction_change_is_break_before_make() {
        let mut controller = candidate_controller();
        acquire(&mut controller);
        let started = controller.accept_command(
            ControllerCommand::apply(
                ControlEpoch::try_new(7).expect("epoch"),
                V2CommandSequence::new(1),
                pair(1, 0),
                deadline(20),
            ),
            now(1),
        );
        assert!(matches!(
            started.motor(),
            MotorDirective::PreloadWhileDisabled(_)
        ));
        assert_eq!(
            controller.tick(now(2)).event(),
            ControllerEvent::MotionApplied
        );

        let reversal = controller.accept_command(
            ControllerCommand::apply(
                ControlEpoch::try_new(7).expect("epoch"),
                V2CommandSequence::new(2),
                pair(-1, 0),
                deadline(20),
            ),
            now(3),
        );
        assert_eq!(reversal.event(), ControllerEvent::TransitionStarted);
        assert_eq!(reversal.motor(), MotorDirective::DisableAndZero);
        assert_eq!(controller.tick(now(4)).motor(), MotorDirective::Hold);
        assert!(matches!(
            controller.tick(now(5)).motor(),
            MotorDirective::PreloadWhileDisabled(_)
        ));
        assert!(matches!(
            controller.tick(now(6)).motor(),
            MotorDirective::EnablePreloaded(_)
        ));
    }
}
