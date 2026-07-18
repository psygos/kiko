//! Pure fail-safe controller state and command admission.
//!
//! The target binary is responsible for parsing bytes once into these domain
//! values and for executing every [`MotorDirective`].  This module makes no
//! claim about GPIO, timer, driver, or wheel behavior.

use robot_protocol::{
    ControllerUptimeMsWrapping,
    v2::{
        ActuatorConfigFingerprint, ControlEpoch, ControllerDeadlineMsWrapping, DeadlineRelation,
        V2CommandLeaseMs, V2CommandSequence,
    },
};

use crate::motor::{
    ActuatorEnvelope, DeadlineStatus, MAX_UNAMBIGUOUS_WRAPPING_TICKS, MotionEnvelopeError,
    MotorDirective, MotorPoll, MotorTiming, MotorTransition, MotorTransitionError, PwmPair,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandKind {
    /// The only command accepted from `AwaitingArm`.  It must use sequence zero,
    /// request zero PWM, and name the active actuator fingerprint exactly.
    Acquire {
        expected_actuator_fingerprint: Option<ActuatorConfigFingerprint>,
    },
    Apply,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControllerCommand {
    control_epoch: ControlEpoch,
    sequence: V2CommandSequence,
    pwm: PwmPair,
    expires_at: ControllerDeadlineMsWrapping,
    kind: CommandKind,
}

impl ControllerCommand {
    pub const fn acquire(
        control_epoch: ControlEpoch,
        pwm: PwmPair,
        expires_at: ControllerDeadlineMsWrapping,
        expected_actuator_fingerprint: Option<ActuatorConfigFingerprint>,
    ) -> Self {
        Self {
            control_epoch,
            sequence: V2CommandSequence::FIRST,
            pwm,
            expires_at,
            kind: CommandKind::Acquire {
                expected_actuator_fingerprint,
            },
        }
    }

    pub const fn apply(
        control_epoch: ControlEpoch,
        sequence: V2CommandSequence,
        pwm: PwmPair,
        expires_at: ControllerDeadlineMsWrapping,
    ) -> Self {
        Self {
            control_epoch,
            sequence,
            pwm,
            expires_at,
            kind: CommandKind::Apply,
        }
    }

    pub const fn control_epoch(self) -> ControlEpoch {
        self.control_epoch
    }

    pub const fn sequence(self) -> V2CommandSequence {
        self.sequence
    }

    pub const fn pwm(self) -> PwmPair {
        self.pwm
    }

    pub const fn expires_at(self) -> ControllerDeadlineMsWrapping {
        self.expires_at
    }

    pub const fn kind(self) -> CommandKind {
        self.kind
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControllerConfig {
    control_epoch: ControlEpoch,
    actuator: ActuatorEnvelope,
    maximum_command_lease: V2CommandLeaseMs,
    motor_timing: MotorTiming,
}

impl ControllerConfig {
    pub const fn new(
        control_epoch: ControlEpoch,
        actuator: ActuatorEnvelope,
        maximum_command_lease: V2CommandLeaseMs,
        motor_timing: MotorTiming,
    ) -> Self {
        Self {
            control_epoch,
            actuator,
            maximum_command_lease,
            motor_timing,
        }
    }

    pub const fn control_epoch(self) -> ControlEpoch {
        self.control_epoch
    }

    pub const fn actuator(self) -> ActuatorEnvelope {
        self.actuator
    }

    pub const fn maximum_command_lease(self) -> V2CommandLeaseMs {
        self.maximum_command_lease
    }

    pub const fn motor_timing(self) -> MotorTiming {
        self.motor_timing
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisarmReason {
    BootCompleted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActiveCommandLease {
    accepted_at: ControllerUptimeMsWrapping,
    expires_at: ControllerDeadlineMsWrapping,
}

/// Coherent target-timer evidence sampled while the motor-effect boundary is
/// protected from the deadline ISR.
///
/// The pure controller owns the semantic lease. The target adapter must also
/// prove that the hardware compare still represents that same lease before
/// and after any MMIO that can enable PWM. Keeping this decision pure makes
/// the otherwise timing-sensitive adapter rule exhaustively testable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeadlineTimerSnapshot {
    lease_armed: bool,
    compare_interrupt_enabled: bool,
    compare_pending: bool,
    programmed_deadline: ControllerDeadlineMsWrapping,
    observed_at: ControllerUptimeMsWrapping,
}

impl DeadlineTimerSnapshot {
    pub const fn new(
        lease_armed: bool,
        compare_interrupt_enabled: bool,
        compare_pending: bool,
        programmed_deadline: ControllerDeadlineMsWrapping,
        observed_at: ControllerUptimeMsWrapping,
    ) -> Self {
        Self {
            lease_armed,
            compare_interrupt_enabled,
            compare_pending,
            programmed_deadline,
            observed_at,
        }
    }

    /// Motion is permitted only when every hardware observation names the
    /// expected, still-future lease. A pending compare is expiry evidence even
    /// when the ISR has not run because interrupts are masked.
    pub const fn permits_motion_until(
        self,
        expected_deadline: ControllerDeadlineMsWrapping,
    ) -> bool {
        self.lease_armed
            && self.compare_interrupt_enabled
            && !self.compare_pending
            && self.programmed_deadline.get() == expected_deadline.get()
            && matches!(
                expected_deadline.relation_to(self.observed_at),
                DeadlineRelation::Future { .. }
            )
    }

    pub const fn indicates_expiry_of(
        self,
        expected_deadline: ControllerDeadlineMsWrapping,
    ) -> bool {
        self.programmed_deadline.get() == expected_deadline.get()
            && (self.compare_pending
                || !matches!(
                    expected_deadline.relation_to(self.observed_at),
                    DeadlineRelation::Future { .. }
                ))
    }
}

impl ActiveCommandLease {
    pub const fn new(
        accepted_at: ControllerUptimeMsWrapping,
        expires_at: ControllerDeadlineMsWrapping,
    ) -> Self {
        Self {
            accepted_at,
            expires_at,
        }
    }

    pub const fn accepted_at(self) -> ControllerUptimeMsWrapping {
        self.accepted_at
    }

    pub const fn expires_at(self) -> ControllerDeadlineMsWrapping {
        self.expires_at
    }

    pub const fn status_at(self, now: ControllerUptimeMsWrapping) -> DeadlineStatus {
        let observed_elapsed = now.get().wrapping_sub(self.accepted_at.get());
        if observed_elapsed > MAX_UNAMBIGUOUS_WRAPPING_TICKS {
            return DeadlineStatus::ObservationGap;
        }
        match self.expires_at.relation_to(now) {
            DeadlineRelation::Future { .. } => DeadlineStatus::Pending,
            DeadlineRelation::Expired => DeadlineStatus::Reached,
            DeadlineRelation::AmbiguousHalfRange => DeadlineStatus::ObservationGap,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControllerMode {
    BootSafe,
    AwaitingArm {
        reason: DisarmReason,
    },
    ArmedStopped {
        last_sequence: V2CommandSequence,
    },
    Transitioning {
        sequence: V2CommandSequence,
        transition: MotorTransition,
        lease: ActiveCommandLease,
    },
    Driving {
        sequence: V2CommandSequence,
        applied: PwmPair,
        lease: ActiveCommandLease,
    },
    FaultLatched {
        fault: FaultCode,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultCode {
    CommandBeforeReady,
    WrongControlEpoch,
    AcquisitionMustUseSequenceZero,
    AcquisitionMustRequestZero,
    AcquisitionFingerprintMismatch,
    AcquisitionRequired,
    UnexpectedAcquisition,
    LeaseAboveFirmwareMaximum { received_ms: u32, maximum_ms: u16 },
    DuplicateConflict { sequence: u32 },
    SequenceGap { expected: u32, received: u32 },
    SequenceOlder { previous: u32, received: u32 },
    SequenceAmbiguousHalfRange { previous: u32, received: u32 },
    SequenceExhausted { previous: u32 },
    MotionEnvelope(MotionEnvelopeError),
    CommandDuringTransition,
    CommandLeaseExpired,
    ClockObservationGap,
    MotorTransition(MotorTransitionError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControllerEvent {
    ReadyForZeroAcquisition,
    ZeroAcquisitionAccepted,
    DuplicateIgnoredWithoutLeaseRenewal,
    StopApplied,
    TransitionStarted,
    TransitionAdvanced,
    MotionApplied,
    FaultLatched(FaultCode),
    AlreadyFaultLatched(FaultCode),
    NoChange,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControllerStep {
    event: ControllerEvent,
    motor: MotorDirective,
}

impl ControllerStep {
    const fn new(event: ControllerEvent, motor: MotorDirective) -> Self {
        Self { event, motor }
    }

    pub const fn event(self) -> ControllerEvent {
        self.event
    }

    pub const fn motor(self) -> MotorDirective {
        self.motor
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadyError {
    NotBootSafe,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SequenceAdmission {
    Next,
    Duplicate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControllerWatchdogStatus {
    SafeOutputsDisabled,
    SafeTransitionWithLiveLease,
    SafeDrivingWithLiveLease,
    UnsafeExpiredOrAmbiguous,
}

#[derive(Clone, Copy, Debug)]
pub struct Controller {
    config: ControllerConfig,
    mode: ControllerMode,
    last_command: Option<ControllerCommand>,
}

impl Controller {
    pub const fn new(config: ControllerConfig) -> Self {
        Self {
            config,
            mode: ControllerMode::BootSafe,
            last_command: None,
        }
    }

    pub const fn config(&self) -> ControllerConfig {
        self.config
    }

    pub const fn mode(&self) -> ControllerMode {
        self.mode
    }

    pub const fn last_command(&self) -> Option<ControllerCommand> {
        self.last_command
    }

    pub fn mark_ready(&mut self) -> Result<ControllerStep, ReadyError> {
        if self.mode != ControllerMode::BootSafe {
            return Err(ReadyError::NotBootSafe);
        }
        self.mode = ControllerMode::AwaitingArm {
            reason: DisarmReason::BootCompleted,
        };
        Ok(ControllerStep::new(
            ControllerEvent::ReadyForZeroAcquisition,
            MotorDirective::DisableAndZero,
        ))
    }

    pub fn accept_command(
        &mut self,
        command: ControllerCommand,
        now: ControllerUptimeMsWrapping,
    ) -> ControllerStep {
        if let ControllerMode::FaultLatched { fault } = self.mode {
            return ControllerStep::new(
                ControllerEvent::AlreadyFaultLatched(fault),
                MotorDirective::DisableAndZero,
            );
        }
        if self.mode == ControllerMode::BootSafe {
            return self.latch_fault(FaultCode::CommandBeforeReady);
        }
        if let Some(fault) = self.active_motion_lease_fault_at(now) {
            return self.latch_fault(fault);
        }
        if command.control_epoch() != self.config.control_epoch() {
            return self.latch_fault(FaultCode::WrongControlEpoch);
        }
        match command.expires_at().relation_to(now) {
            DeadlineRelation::Future { remaining_ms }
                if remaining_ms > u32::from(self.config.maximum_command_lease().get()) =>
            {
                return self.latch_fault(FaultCode::LeaseAboveFirmwareMaximum {
                    received_ms: remaining_ms,
                    maximum_ms: self.config.maximum_command_lease().get(),
                });
            }
            DeadlineRelation::Future { .. } => {}
            DeadlineRelation::Expired => {
                return self.latch_fault(FaultCode::CommandLeaseExpired);
            }
            DeadlineRelation::AmbiguousHalfRange => {
                return self.latch_fault(FaultCode::ClockObservationGap);
            }
        }

        match self.mode {
            ControllerMode::AwaitingArm { .. } => self.accept_acquisition(command),
            ControllerMode::ArmedStopped { last_sequence } => {
                self.accept_after_sequence(command, now, last_sequence, PwmPair::STOP, false)
            }
            ControllerMode::Driving {
                sequence, applied, ..
            } => self.accept_after_sequence(command, now, sequence, applied, false),
            ControllerMode::Transitioning { sequence, .. } => {
                self.accept_after_sequence(command, now, sequence, PwmPair::STOP, true)
            }
            // These branches are already handled before command validation.
            // Keep the fallback fail-safe instead of relying on a panic-only
            // internal invariant.
            ControllerMode::BootSafe => self.latch_fault(FaultCode::CommandBeforeReady),
            ControllerMode::FaultLatched { fault } => ControllerStep::new(
                ControllerEvent::AlreadyFaultLatched(fault),
                MotorDirective::DisableAndZero,
            ),
        }
    }

    fn accept_acquisition(&mut self, command: ControllerCommand) -> ControllerStep {
        let CommandKind::Acquire {
            expected_actuator_fingerprint,
        } = command.kind()
        else {
            return self.latch_fault(FaultCode::AcquisitionRequired);
        };
        if command.sequence().get() != 0 {
            return self.latch_fault(FaultCode::AcquisitionMustUseSequenceZero);
        }
        if !command.pwm().is_stop() {
            return self.latch_fault(FaultCode::AcquisitionMustRequestZero);
        }
        if expected_actuator_fingerprint != self.config.actuator().fingerprint() {
            return self.latch_fault(FaultCode::AcquisitionFingerprintMismatch);
        }

        self.last_command = Some(command);
        self.mode = ControllerMode::ArmedStopped {
            last_sequence: command.sequence(),
        };
        ControllerStep::new(
            ControllerEvent::ZeroAcquisitionAccepted,
            MotorDirective::DisableAndZero,
        )
    }

    fn accept_after_sequence(
        &mut self,
        command: ControllerCommand,
        now: ControllerUptimeMsWrapping,
        previous_sequence: V2CommandSequence,
        currently_applied: PwmPair,
        transition_in_progress: bool,
    ) -> ControllerStep {
        let admission = match self.classify_sequence(previous_sequence, command) {
            Ok(admission) => admission,
            Err(fault) => return self.latch_fault(fault),
        };
        if admission == SequenceAdmission::Duplicate {
            return ControllerStep::new(
                ControllerEvent::DuplicateIgnoredWithoutLeaseRenewal,
                MotorDirective::Hold,
            );
        }
        if matches!(command.kind(), CommandKind::Acquire { .. }) {
            return self.latch_fault(FaultCode::UnexpectedAcquisition);
        }

        if command.pwm().is_stop() {
            self.last_command = Some(command);
            self.mode = ControllerMode::ArmedStopped {
                last_sequence: command.sequence(),
            };
            return ControllerStep::new(
                ControllerEvent::StopApplied,
                MotorDirective::DisableAndZero,
            );
        }
        if transition_in_progress {
            return self.latch_fault(FaultCode::CommandDuringTransition);
        }
        if let Err(source) = self
            .config
            .actuator()
            .validate_transition(currently_applied, command.pwm())
        {
            return self.latch_fault(FaultCode::MotionEnvelope(source));
        }

        let (transition, motor) = match MotorTransition::start(
            currently_applied,
            command.pwm(),
            now,
            self.config.motor_timing(),
        ) {
            Ok(value) => value,
            Err(source) => return self.latch_fault(FaultCode::MotorTransition(source)),
        };
        self.last_command = Some(command);
        self.mode = ControllerMode::Transitioning {
            sequence: command.sequence(),
            transition,
            lease: ActiveCommandLease::new(now, command.expires_at()),
        };
        ControllerStep::new(ControllerEvent::TransitionStarted, motor)
    }

    fn classify_sequence(
        &self,
        previous: V2CommandSequence,
        command: ControllerCommand,
    ) -> Result<SequenceAdmission, FaultCode> {
        if command.sequence() == previous {
            return if self.last_command == Some(command) {
                Ok(SequenceAdmission::Duplicate)
            } else {
                Err(FaultCode::DuplicateConflict {
                    sequence: command.sequence().get(),
                })
            };
        }

        let Some(expected) = previous.checked_successor() else {
            return Err(FaultCode::SequenceExhausted {
                previous: previous.get(),
            });
        };
        if command.sequence() == expected {
            return Ok(SequenceAdmission::Next);
        }
        let delta = command.sequence().get().wrapping_sub(previous.get());
        match delta {
            1..0x8000_0000 => Err(FaultCode::SequenceGap {
                expected: expected.get(),
                received: command.sequence().get(),
            }),
            0x8000_0000 => Err(FaultCode::SequenceAmbiguousHalfRange {
                previous: previous.get(),
                received: command.sequence().get(),
            }),
            _ => Err(FaultCode::SequenceOlder {
                previous: previous.get(),
                received: command.sequence().get(),
            }),
        }
    }

    pub fn tick(&mut self, now: ControllerUptimeMsWrapping) -> ControllerStep {
        match self.mode {
            ControllerMode::Transitioning {
                sequence,
                mut transition,
                lease,
            } => {
                match lease.status_at(now) {
                    DeadlineStatus::Reached => {
                        return self.latch_fault(FaultCode::CommandLeaseExpired);
                    }
                    DeadlineStatus::ObservationGap => {
                        return self.latch_fault(FaultCode::ClockObservationGap);
                    }
                    DeadlineStatus::Pending => {}
                }
                match transition.poll(now, self.config.motor_timing()) {
                    Ok(MotorPoll::Pending(motor)) => {
                        self.mode = ControllerMode::Transitioning {
                            sequence,
                            transition,
                            lease,
                        };
                        ControllerStep::new(ControllerEvent::TransitionAdvanced, motor)
                    }
                    Ok(MotorPoll::Applied { pwm, directive, .. }) => {
                        self.mode = ControllerMode::Driving {
                            sequence,
                            applied: pwm,
                            lease,
                        };
                        ControllerStep::new(ControllerEvent::MotionApplied, directive)
                    }
                    Err(source) => self.latch_fault(FaultCode::MotorTransition(source)),
                }
            }
            ControllerMode::Driving { lease, .. } => match lease.status_at(now) {
                DeadlineStatus::Pending => {
                    ControllerStep::new(ControllerEvent::NoChange, MotorDirective::Hold)
                }
                DeadlineStatus::Reached => self.latch_fault(FaultCode::CommandLeaseExpired),
                DeadlineStatus::ObservationGap => self.latch_fault(FaultCode::ClockObservationGap),
            },
            ControllerMode::BootSafe
            | ControllerMode::AwaitingArm { .. }
            | ControllerMode::ArmedStopped { .. } => {
                ControllerStep::new(ControllerEvent::NoChange, MotorDirective::Hold)
            }
            ControllerMode::FaultLatched { fault } => ControllerStep::new(
                ControllerEvent::AlreadyFaultLatched(fault),
                MotorDirective::DisableAndZero,
            ),
        }
    }

    pub fn watchdog_status_at(&self, now: ControllerUptimeMsWrapping) -> ControllerWatchdogStatus {
        match self.mode {
            ControllerMode::BootSafe
            | ControllerMode::AwaitingArm { .. }
            | ControllerMode::ArmedStopped { .. }
            | ControllerMode::FaultLatched { .. } => ControllerWatchdogStatus::SafeOutputsDisabled,
            ControllerMode::Transitioning { lease, .. } => {
                if lease.status_at(now) == DeadlineStatus::Pending {
                    ControllerWatchdogStatus::SafeTransitionWithLiveLease
                } else {
                    ControllerWatchdogStatus::UnsafeExpiredOrAmbiguous
                }
            }
            ControllerMode::Driving { lease, .. } => {
                if lease.status_at(now) == DeadlineStatus::Pending {
                    ControllerWatchdogStatus::SafeDrivingWithLiveLease
                } else {
                    ControllerWatchdogStatus::UnsafeExpiredOrAmbiguous
                }
            }
        }
    }

    fn active_motion_lease_fault_at(&self, now: ControllerUptimeMsWrapping) -> Option<FaultCode> {
        let lease = match self.mode {
            ControllerMode::Transitioning { lease, .. } | ControllerMode::Driving { lease, .. } => {
                lease
            }
            ControllerMode::BootSafe
            | ControllerMode::AwaitingArm { .. }
            | ControllerMode::ArmedStopped { .. }
            | ControllerMode::FaultLatched { .. } => return None,
        };
        match lease.status_at(now) {
            DeadlineStatus::Pending => None,
            DeadlineStatus::Reached => Some(FaultCode::CommandLeaseExpired),
            DeadlineStatus::ObservationGap => Some(FaultCode::ClockObservationGap),
        }
    }

    fn latch_fault(&mut self, fault: FaultCode) -> ControllerStep {
        self.mode = ControllerMode::FaultLatched { fault };
        ControllerStep::new(
            ControllerEvent::FaultLatched(fault),
            MotorDirective::DisableAndZero,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motor::{DurationMs, PwmRange, PwmStepLimit, WheelDrive};
    use robot_protocol::PwmPercent;

    fn pwm(value: i8) -> PwmPercent {
        PwmPercent::try_new(value).expect("test PWM is valid")
    }

    fn pair(left: i8, right: i8) -> PwmPair {
        PwmPair::from_validated(pwm(left), pwm(right))
    }

    fn duration(value: u32) -> DurationMs {
        DurationMs::try_new(value).expect("test duration is valid")
    }

    fn epoch(value: u32) -> ControlEpoch {
        ControlEpoch::try_new(value).expect("test epoch is nonzero")
    }

    fn fingerprint() -> ActuatorConfigFingerprint {
        ActuatorConfigFingerprint::try_new([9; 16]).expect("test fingerprint is nonzero")
    }

    const fn now(value: u32) -> ControllerUptimeMsWrapping {
        ControllerUptimeMsWrapping::new(value)
    }

    const fn deadline(start: u32, remaining_ms: u32) -> ControllerDeadlineMsWrapping {
        ControllerDeadlineMsWrapping::new(start.wrapping_add(remaining_ms))
    }

    fn validated_envelope(maximum_step: u8) -> ActuatorEnvelope {
        ActuatorEnvelope::validated(
            fingerprint(),
            PwmRange::try_new(pwm(-50), pwm(50)).expect("left range"),
            PwmRange::try_new(pwm(-50), pwm(50)).expect("right range"),
            PwmStepLimit::try_new(maximum_step, maximum_step).expect("step limit"),
        )
    }

    fn controller_with(actuator: ActuatorEnvelope) -> Controller {
        Controller::new(ControllerConfig::new(
            epoch(1),
            actuator,
            V2CommandLeaseMs::try_new(100).expect("valid firmware maximum"),
            MotorTiming::new(duration(5), duration(2)),
        ))
    }

    fn acquire(controller: &mut Controller) {
        controller.mark_ready().expect("boot-safe controller");
        let step = controller.accept_command(
            ControllerCommand::acquire(
                epoch(1),
                PwmPair::STOP,
                deadline(0, 50),
                controller.config().actuator().fingerprint(),
            ),
            now(0),
        );
        assert_eq!(step.event(), ControllerEvent::ZeroAcquisitionAccepted);
    }

    fn start_motion(controller: &mut Controller, now: u32, sequence: u32) -> ControllerCommand {
        let command = ControllerCommand::apply(
            epoch(1),
            V2CommandSequence::new(sequence),
            pair(10, 10),
            deadline(now, 20),
        );
        let step = controller.accept_command(command, self::now(now));
        assert_eq!(step.event(), ControllerEvent::TransitionStarted);
        command
    }

    fn finish_initial_preload(controller: &mut Controller, now: u32) {
        let step = controller.tick(self::now(now));
        assert_eq!(step.event(), ControllerEvent::MotionApplied);
        let MotorDirective::EnablePreloaded(output) = step.motor() else {
            panic!("stopped start must enable a preloaded output")
        };
        assert!(matches!(output.left(), WheelDrive::Forward(_)));
        assert!(matches!(output.right(), WheelDrive::Forward(_)));
    }

    #[test]
    fn boot_requires_exact_seq_zero_stop_and_fingerprint() {
        let mut controller = controller_with(validated_envelope(20));
        assert_eq!(controller.mode(), ControllerMode::BootSafe);
        assert_eq!(
            controller
                .mark_ready()
                .expect("first ready transition")
                .event(),
            ControllerEvent::ReadyForZeroAcquisition
        );
        assert_eq!(controller.mark_ready(), Err(ReadyError::NotBootSafe));

        let mut wrong_fingerprint = controller_with(validated_envelope(20));
        wrong_fingerprint.mark_ready().expect("ready");
        let step = wrong_fingerprint.accept_command(
            ControllerCommand::acquire(
                epoch(1),
                PwmPair::STOP,
                deadline(0, 10),
                Some(ActuatorConfigFingerprint::try_new([8; 16]).expect("nonzero")),
            ),
            now(0),
        );
        assert!(matches!(
            step.event(),
            ControllerEvent::FaultLatched(FaultCode::AcquisitionFingerprintMismatch)
        ));
        assert_eq!(step.motor(), MotorDirective::DisableAndZero);

        let mut nonzero = controller_with(validated_envelope(20));
        nonzero.mark_ready().expect("ready");
        let step = nonzero.accept_command(
            ControllerCommand::acquire(epoch(1), pair(1, 0), deadline(0, 10), Some(fingerprint())),
            now(0),
        );
        assert!(matches!(
            step.event(),
            ControllerEvent::FaultLatched(FaultCode::AcquisitionMustRequestZero)
        ));
    }

    #[test]
    fn wrong_epoch_and_command_before_ready_fault_to_zero() {
        for (ready, wrong_epoch, expected) in [
            (false, false, FaultCode::CommandBeforeReady),
            (true, true, FaultCode::WrongControlEpoch),
        ] {
            let mut controller = controller_with(validated_envelope(20));
            if ready {
                controller.mark_ready().expect("ready");
            }
            let command = ControllerCommand::acquire(
                if wrong_epoch { epoch(2) } else { epoch(1) },
                PwmPair::STOP,
                deadline(0, 10),
                Some(fingerprint()),
            );
            let step = controller.accept_command(command, now(0));
            assert_eq!(step.event(), ControllerEvent::FaultLatched(expected));
            assert_eq!(step.motor(), MotorDirective::DisableAndZero);
        }
    }

    #[test]
    fn unvalidated_controller_can_acquire_stopped_but_cannot_move() {
        let mut controller = controller_with(ActuatorEnvelope::unvalidated());
        acquire(&mut controller);
        let step = controller.accept_command(
            ControllerCommand::apply(
                epoch(1),
                V2CommandSequence::new(1),
                pair(1, 0),
                deadline(1, 10),
            ),
            now(1),
        );
        assert!(matches!(
            step.event(),
            ControllerEvent::FaultLatched(FaultCode::MotionEnvelope(
                MotionEnvelopeError::MotionDisabledUntilValidated
            ))
        ));
        assert_eq!(step.motor(), MotorDirective::DisableAndZero);
    }

    #[test]
    fn duplicate_is_idempotent_and_never_renews_the_original_lease() {
        let mut controller = controller_with(validated_envelope(20));
        acquire(&mut controller);
        let command = start_motion(&mut controller, 10, 1);
        finish_initial_preload(&mut controller, 12);
        let original_lease = match controller.mode() {
            ControllerMode::Driving { lease, .. } => lease,
            _ => panic!("expected driving"),
        };

        let step = controller.accept_command(command, now(20));
        assert_eq!(
            step.event(),
            ControllerEvent::DuplicateIgnoredWithoutLeaseRenewal
        );
        let retained_lease = match controller.mode() {
            ControllerMode::Driving { lease, .. } => lease,
            _ => panic!("duplicate changed mode"),
        };
        assert_eq!(retained_lease, original_lease);
        let expired = controller.tick(now(30));
        assert_eq!(
            expired.event(),
            ControllerEvent::FaultLatched(FaultCode::CommandLeaseExpired)
        );
        assert_eq!(expired.motor(), MotorDirective::DisableAndZero);
    }

    #[test]
    fn a_new_command_cannot_hide_an_already_expired_motion_lease() {
        let mut controller = controller_with(validated_envelope(20));
        acquire(&mut controller);
        start_motion(&mut controller, 10, 1);
        finish_initial_preload(&mut controller, 12);

        let step = controller.accept_command(
            ControllerCommand::apply(
                epoch(1),
                V2CommandSequence::new(2),
                pair(20, 20),
                deadline(30, 20),
            ),
            now(30),
        );
        assert_eq!(
            step.event(),
            ControllerEvent::FaultLatched(FaultCode::CommandLeaseExpired)
        );
        assert_eq!(step.motor(), MotorDirective::DisableAndZero);
    }

    #[test]
    fn duplicate_payload_conflict_gap_old_and_half_range_all_fault_to_zero() {
        enum Case {
            Conflict,
            Gap,
            Old,
            Half,
        }
        for case in [Case::Conflict, Case::Gap, Case::Old, Case::Half] {
            let mut controller = controller_with(validated_envelope(200));
            acquire(&mut controller);
            let command = match case {
                Case::Conflict => ControllerCommand::apply(
                    epoch(1),
                    V2CommandSequence::new(0),
                    pair(1, 0),
                    deadline(1, 10),
                ),
                Case::Gap => ControllerCommand::apply(
                    epoch(1),
                    V2CommandSequence::new(2),
                    PwmPair::STOP,
                    deadline(1, 10),
                ),
                Case::Old => ControllerCommand::apply(
                    epoch(1),
                    V2CommandSequence::new(u32::MAX),
                    PwmPair::STOP,
                    deadline(1, 10),
                ),
                Case::Half => ControllerCommand::apply(
                    epoch(1),
                    V2CommandSequence::new(1_u32 << 31),
                    PwmPair::STOP,
                    deadline(1, 10),
                ),
            };
            let step = controller.accept_command(command, now(1));
            assert!(matches!(step.event(), ControllerEvent::FaultLatched(_)));
            assert_eq!(step.motor(), MotorDirective::DisableAndZero);
        }
    }

    #[test]
    fn zero_bypasses_slew_and_aborts_a_transition_immediately() {
        let mut controller = controller_with(validated_envelope(10));
        acquire(&mut controller);
        start_motion(&mut controller, 10, 1);
        let stop = ControllerCommand::apply(
            epoch(1),
            V2CommandSequence::new(2),
            PwmPair::STOP,
            deadline(11, 1),
        );
        let step = controller.accept_command(stop, now(11));
        assert_eq!(step.event(), ControllerEvent::StopApplied);
        assert_eq!(step.motor(), MotorDirective::DisableAndZero);
        assert!(matches!(
            controller.mode(),
            ControllerMode::ArmedStopped { .. }
        ));
    }

    #[test]
    fn nonzero_command_during_transition_faults_instead_of_hiding_retargeting() {
        let mut controller = controller_with(validated_envelope(20));
        acquire(&mut controller);
        start_motion(&mut controller, 10, 1);
        let step = controller.accept_command(
            ControllerCommand::apply(
                epoch(1),
                V2CommandSequence::new(2),
                pair(20, 20),
                deadline(11, 20),
            ),
            now(11),
        );
        assert_eq!(
            step.event(),
            ControllerEvent::FaultLatched(FaultCode::CommandDuringTransition)
        );
        assert_eq!(step.motor(), MotorDirective::DisableAndZero);
    }

    #[test]
    fn lease_expiry_and_clock_gap_are_fail_closed_across_counter_wrap() {
        let mut expiry = controller_with(validated_envelope(20));
        acquire(&mut expiry);
        start_motion(&mut expiry, u32::MAX - 2, 1);
        finish_initial_preload(&mut expiry, u32::MAX);
        assert_eq!(expiry.tick(now(16)).event(), ControllerEvent::NoChange);
        let step = expiry.tick(now(17));
        assert_eq!(
            step.event(),
            ControllerEvent::FaultLatched(FaultCode::CommandLeaseExpired)
        );

        let mut gap = controller_with(validated_envelope(20));
        acquire(&mut gap);
        start_motion(&mut gap, 100, 1);
        let step = gap.tick(now(99));
        assert_eq!(
            step.event(),
            ControllerEvent::FaultLatched(FaultCode::ClockObservationGap)
        );
        assert_eq!(step.motor(), MotorDirective::DisableAndZero);
    }

    #[test]
    fn a_pre_admission_timestamp_cannot_tick_a_newly_accepted_lease() {
        let lease = ActiveCommandLease::new(now(101), deadline(101, 20));
        assert_eq!(lease.status_at(now(100)), DeadlineStatus::ObservationGap);
        assert_eq!(lease.status_at(now(101)), DeadlineStatus::Pending);
    }

    #[test]
    fn deadline_timer_snapshot_requires_matching_live_hardware_evidence() {
        let expected = ControllerDeadlineMsWrapping::new(120);
        let live = DeadlineTimerSnapshot::new(
            true,
            true,
            false,
            expected,
            ControllerUptimeMsWrapping::new(119),
        );
        assert!(live.permits_motion_until(expected));

        for denied in [
            DeadlineTimerSnapshot::new(false, true, false, expected, now(119)),
            DeadlineTimerSnapshot::new(true, false, false, expected, now(119)),
            // This is the masked-interrupt race: the compare became pending,
            // but TIM5 has not yet run and cleared the software armed flag.
            DeadlineTimerSnapshot::new(true, true, true, expected, now(120)),
            DeadlineTimerSnapshot::new(
                true,
                true,
                false,
                ControllerDeadlineMsWrapping::new(121),
                now(119),
            ),
            DeadlineTimerSnapshot::new(true, true, false, expected, now(120)),
        ] {
            assert!(!denied.permits_motion_until(expected));
        }
    }

    #[test]
    fn sequence_exhaustion_requires_stop_and_reacquisition() {
        let previous = V2CommandSequence::new(u32::MAX);
        let command = ControllerCommand::apply(
            epoch(1),
            V2CommandSequence::FIRST,
            PwmPair::STOP,
            deadline(0, 1),
        );
        let mut controller = controller_with(validated_envelope(20));
        controller.last_command = Some(ControllerCommand::apply(
            epoch(1),
            previous,
            PwmPair::STOP,
            deadline(0, 1),
        ));
        assert_eq!(
            controller.classify_sequence(previous, command),
            Err(FaultCode::SequenceExhausted { previous: u32::MAX })
        );
    }

    #[test]
    fn watchdog_status_refuses_expired_motion_before_tick_processes_the_fault() {
        let mut controller = controller_with(validated_envelope(20));
        acquire(&mut controller);
        start_motion(&mut controller, 10, 1);
        finish_initial_preload(&mut controller, 12);
        assert_eq!(
            controller.watchdog_status_at(now(29)),
            ControllerWatchdogStatus::SafeDrivingWithLiveLease
        );
        assert_eq!(
            controller.watchdog_status_at(now(30)),
            ControllerWatchdogStatus::UnsafeExpiredOrAmbiguous
        );
    }
}
